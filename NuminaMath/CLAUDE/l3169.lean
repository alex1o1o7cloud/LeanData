import Mathlib

namespace NUMINAMATH_CALUDE_prob_ratio_balls_in_bins_l3169_316936

def factorial (n : ℕ) : ℕ := sorry

def multinomial_coefficient (n : ℕ) (x : List ℕ) : ℝ := sorry

def p (n : ℕ) (k : ℕ) : ℝ := 
  multinomial_coefficient n [3, 6, 5, 4, 2, 10]

def q (n : ℕ) (k : ℕ) : ℝ := 
  multinomial_coefficient n [5, 5, 5, 5, 5, 5]

theorem prob_ratio_balls_in_bins : 
  p 30 6 / q 30 6 = 0.125 := by sorry

end NUMINAMATH_CALUDE_prob_ratio_balls_in_bins_l3169_316936


namespace NUMINAMATH_CALUDE_inequality_condition_l3169_316983

theorem inequality_condition (a b c : ℝ) :
  (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b → (a + Real.sqrt (b + c) > b + Real.sqrt (a + c))) ↔ c > (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3169_316983


namespace NUMINAMATH_CALUDE_existence_of_representation_l3169_316934

theorem existence_of_representation (m : ℤ) :
  ∃ (a b k : ℤ), Odd a ∧ Odd b ∧ k ≥ 0 ∧ 2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_representation_l3169_316934


namespace NUMINAMATH_CALUDE_store_shirts_count_l3169_316975

theorem store_shirts_count (shirts_sold : ℕ) (shirts_left : ℕ) :
  shirts_sold = 21 →
  shirts_left = 28 →
  shirts_sold + shirts_left = 49 :=
by sorry

end NUMINAMATH_CALUDE_store_shirts_count_l3169_316975


namespace NUMINAMATH_CALUDE_park_fencing_cost_l3169_316938

theorem park_fencing_cost (length width area perimeter total_cost : ℝ) : 
  length / width = 3 / 2 →
  length * width = 3750 →
  perimeter = 2 * (length + width) →
  total_cost = 175 →
  (total_cost / perimeter) * 100 = 70 :=
by sorry

end NUMINAMATH_CALUDE_park_fencing_cost_l3169_316938


namespace NUMINAMATH_CALUDE_burrito_count_l3169_316973

theorem burrito_count (cheese_per_burrito cheese_per_taco total_cheese : ℕ) 
  (h1 : cheese_per_burrito = 4)
  (h2 : cheese_per_taco = 9)
  (h3 : total_cheese = 37) :
  ∃ (num_burritos : ℕ), 
    num_burritos * cheese_per_burrito + cheese_per_taco = total_cheese ∧ 
    num_burritos = 7 := by
  sorry

end NUMINAMATH_CALUDE_burrito_count_l3169_316973


namespace NUMINAMATH_CALUDE_total_leaves_on_our_farm_l3169_316915

/-- Represents a farm with trees -/
structure Farm :=
  (num_trees : ℕ)
  (branches_per_tree : ℕ)
  (sub_branches_per_branch : ℕ)
  (leaves_per_sub_branch : ℕ)

/-- Calculates the total number of leaves on all trees in the farm -/
def total_leaves (f : Farm) : ℕ :=
  f.num_trees * f.branches_per_tree * f.sub_branches_per_branch * f.leaves_per_sub_branch

/-- The farm described in the problem -/
def our_farm : Farm :=
  { num_trees := 4
  , branches_per_tree := 10
  , sub_branches_per_branch := 40
  , leaves_per_sub_branch := 60 }

/-- Theorem stating that the total number of leaves on all trees in our farm is 96,000 -/
theorem total_leaves_on_our_farm : total_leaves our_farm = 96000 := by
  sorry

end NUMINAMATH_CALUDE_total_leaves_on_our_farm_l3169_316915


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3169_316956

theorem smallest_part_of_proportional_division (total : ℝ) (a b c : ℝ) 
  (h_total : total = 120)
  (h_prop : a + b + c = 15)
  (h_a : a = 3)
  (h_b : b = 5)
  (h_c : c = 7) :
  min (total * a / (a + b + c)) (min (total * b / (a + b + c)) (total * c / (a + b + c))) = 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3169_316956


namespace NUMINAMATH_CALUDE_semicircle_radius_l3169_316950

theorem semicircle_radius (x y z : ℝ) (h_right_angle : x^2 + y^2 = z^2)
  (h_xy_area : π * x^2 / 2 = 12 * π) (h_xz_arc : π * y = 10 * π) :
  z / 2 = 2 * Real.sqrt 31 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l3169_316950


namespace NUMINAMATH_CALUDE_sofia_shopping_cost_l3169_316909

/-- The total cost of Sofia's shopping trip -/
def total_cost (shirt_price : ℕ) : ℕ :=
  let shoe_price : ℕ := shirt_price + 3
  let shirts_and_shoes : ℕ := 2 * shirt_price + shoe_price
  let bag_price : ℕ := shirts_and_shoes / 2
  2 * shirt_price + shoe_price + bag_price

/-- Theorem stating that Sofia's total cost is $36 -/
theorem sofia_shopping_cost : total_cost 7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sofia_shopping_cost_l3169_316909


namespace NUMINAMATH_CALUDE_cubic_function_max_l3169_316944

/-- Given a cubic function with specific properties, prove its maximum value on [-3, 3] -/
theorem cubic_function_max (a b c : ℝ) : 
  (∀ x, (∃ y, y = a * x^3 + b * x + c)) →  -- f(x) = ax³ + bx + c
  (∃ y, y = 8 * a + 2 * b + c ∧ y = c - 16) →  -- f(2) = c - 16
  (3 * a * 2^2 + b = 0) →  -- f'(2) = 0 (extremum condition)
  (a = 1 ∧ b = -12) →  -- Values of a and b
  (∃ x, ∀ y, a * x^3 + b * x + c ≥ y ∧ a * x^3 + b * x + c = 28) →  -- Maximum value is 28
  (∃ x, x ∈ Set.Icc (-3) 3 ∧ 
    ∀ y ∈ Set.Icc (-3) 3, a * x^3 + b * x + c ≥ a * y^3 + b * y + c ∧ 
    a * x^3 + b * x + c = 28) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_max_l3169_316944


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3169_316989

-- Define the variables
variable (a : ℕ+) -- a is a positive integer
variable (A B : ℝ) -- A and B are real numbers
variable (x y z : ℕ+) -- x, y, z are positive integers

-- Define the system of equations
def equation1 (x y z : ℕ+) (a : ℕ+) : Prop :=
  (x : ℝ)^2 + (y : ℝ)^2 + (z : ℝ)^2 = (13 * (a : ℝ))^2

def equation2 (x y z : ℕ+) (a : ℕ+) (A B : ℝ) : Prop :=
  (x : ℝ)^2 * (A * (x : ℝ)^2 + B * (y : ℝ)^2) +
  (y : ℝ)^2 * (A * (y : ℝ)^2 + B * (z : ℝ)^2) +
  (z : ℝ)^2 * (A * (z : ℝ)^2 + B * (x : ℝ)^2) =
  1/4 * (2 * A + B) * (13 * (a : ℝ))^4

-- Theorem statement
theorem necessary_and_sufficient_condition :
  (∃ x y z : ℕ+, equation1 x y z a ∧ equation2 x y z a A B) ↔ B = 2 * A :=
sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3169_316989


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3169_316984

/-- A rectangle with an inscribed ellipse -/
structure RectangleWithEllipse where
  -- Rectangle dimensions
  x : ℝ
  y : ℝ
  -- Ellipse semi-major and semi-minor axes
  a : ℝ
  b : ℝ
  -- Conditions
  rectangle_area : x * y = 4024
  ellipse_area : π * a * b = 4024 * π
  foci_distance : x^2 + y^2 = 4 * (a^2 - b^2)
  major_axis : x + y = 2 * a

/-- The perimeter of a rectangle with an inscribed ellipse is 8√2012 -/
theorem rectangle_perimeter (r : RectangleWithEllipse) : r.x + r.y = 8 * Real.sqrt 2012 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3169_316984


namespace NUMINAMATH_CALUDE_open_set_classification_l3169_316923

-- Define the concept of an open set in R²
def is_open_set (A : Set (ℝ × ℝ)) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ A → ∃ (r : ℝ), r > 0 ∧ 
    {q : ℝ × ℝ | Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2) < r} ⊆ A

-- Define the sets
def set1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def set2 : Set (ℝ × ℝ) := {p | p.1 + p.2 + 2 > 0}
def set3 : Set (ℝ × ℝ) := {p | |p.1 + p.2| ≤ 6}
def set4 : Set (ℝ × ℝ) := {p | 0 < p.1^2 + (p.2 - Real.sqrt 2)^2 ∧ p.1^2 + (p.2 - Real.sqrt 2)^2 < 1}

-- State the theorem
theorem open_set_classification :
  ¬(is_open_set set1) ∧
  (is_open_set set2) ∧
  ¬(is_open_set set3) ∧
  (is_open_set set4) :=
sorry

end NUMINAMATH_CALUDE_open_set_classification_l3169_316923


namespace NUMINAMATH_CALUDE_coin_difference_l3169_316966

/-- Represents the number of coins of a specific denomination a person has -/
structure CoinCount where
  fiveRuble : ℕ
  twoRuble : ℕ

/-- Calculates the total value in rubles for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  5 * coins.fiveRuble + 2 * coins.twoRuble

/-- Represents the coin counts for Petya and Vanya -/
structure CoinDistribution where
  petya : CoinCount
  vanya : CoinCount

/-- Checks if the coin distribution satisfies the problem conditions -/
def isValidDistribution (dist : CoinDistribution) : Prop :=
  dist.vanya.fiveRuble = dist.petya.twoRuble ∧
  dist.vanya.twoRuble = dist.petya.fiveRuble ∧
  totalValue dist.petya = totalValue dist.vanya + 60

theorem coin_difference (dist : CoinDistribution) 
  (h : isValidDistribution dist) : 
  dist.petya.fiveRuble - dist.petya.twoRuble = 20 :=
sorry

end NUMINAMATH_CALUDE_coin_difference_l3169_316966


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3169_316916

theorem simplify_square_roots : 
  Real.sqrt (5 * 3) * Real.sqrt (3^4 * 5^2) = 45 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3169_316916


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3169_316935

theorem geometric_sequence_fourth_term (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h2 : a 2 = 9) (h3 : a 5 = 243) : a 4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3169_316935


namespace NUMINAMATH_CALUDE_function_value_range_l3169_316962

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * a + 1

-- State the theorem
theorem function_value_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ 1 ∧ -1 ≤ x₂ ∧ x₂ ≤ 1 ∧ f a x₁ < 0 ∧ 0 < f a x₂) →
  -1 < a ∧ a < -1/3 := by
  sorry


end NUMINAMATH_CALUDE_function_value_range_l3169_316962


namespace NUMINAMATH_CALUDE_graceGardenTopBedRows_l3169_316980

/-- Represents the garden structure and seed distribution --/
structure Garden where
  totalSeeds : ℕ
  topBedSeedsPerRow : ℕ
  mediumBedRows : ℕ
  mediumBedSeedsPerRow : ℕ
  numMediumBeds : ℕ

/-- Calculates the number of rows in the top bed --/
def topBedRows (g : Garden) : ℕ :=
  (g.totalSeeds - g.numMediumBeds * g.mediumBedRows * g.mediumBedSeedsPerRow) / g.topBedSeedsPerRow

/-- Theorem stating that for Grace's garden, the top bed can hold 8 rows --/
theorem graceGardenTopBedRows :
  let g : Garden := {
    totalSeeds := 320,
    topBedSeedsPerRow := 25,
    mediumBedRows := 3,
    mediumBedSeedsPerRow := 20,
    numMediumBeds := 2
  }
  topBedRows g = 8 := by
  sorry

end NUMINAMATH_CALUDE_graceGardenTopBedRows_l3169_316980


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3169_316957

theorem inequality_equivalence (x : ℝ) : (1 / (x - 1) > 1) ↔ (1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3169_316957


namespace NUMINAMATH_CALUDE_pairing_ways_eq_divisors_of_30_l3169_316996

/-- The number of ways to pair 60 cards with the given condition -/
def pairingWays : ℕ := sorry

/-- The set of positive divisors of 30 -/
def divisorsOf30 : Finset ℕ := sorry

/-- Theorem stating that the number of pairing ways is equal to the number of positive divisors of 30 -/
theorem pairing_ways_eq_divisors_of_30 : pairingWays = Finset.card divisorsOf30 := by sorry

end NUMINAMATH_CALUDE_pairing_ways_eq_divisors_of_30_l3169_316996


namespace NUMINAMATH_CALUDE_vlad_sister_height_l3169_316997

/-- Converts feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Converts total inches to feet (discarding remaining inches) -/
def inches_to_feet (inches : ℕ) : ℕ := inches / 12

theorem vlad_sister_height :
  let vlad_height := height_to_inches 6 3
  let height_diff := 41
  let sister_inches := vlad_height - height_diff
  inches_to_feet sister_inches = 2 := by sorry

end NUMINAMATH_CALUDE_vlad_sister_height_l3169_316997


namespace NUMINAMATH_CALUDE_new_sequence_69th_is_original_18th_l3169_316990

/-- Given a sequence, insert_between n seq inserts n elements between each pair of adjacent elements in seq -/
def insert_between (n : ℕ) (seq : ℕ → ℕ) : ℕ → ℕ := sorry

/-- The original sequence -/
def original_sequence : ℕ → ℕ := sorry

/-- The new sequence with 3 elements inserted between each pair of adjacent elements -/
def new_sequence : ℕ → ℕ := insert_between 3 original_sequence

theorem new_sequence_69th_is_original_18th :
  new_sequence 69 = original_sequence 18 := by sorry

end NUMINAMATH_CALUDE_new_sequence_69th_is_original_18th_l3169_316990


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3169_316937

theorem isosceles_right_triangle_area (DE DF : ℝ) (angle_EDF : ℝ) :
  DE = 5 →
  DF = 5 →
  angle_EDF = Real.pi / 2 →
  (1 / 2) * DE * DF = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3169_316937


namespace NUMINAMATH_CALUDE_power_of_square_l3169_316925

theorem power_of_square (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l3169_316925


namespace NUMINAMATH_CALUDE_box_volume_theorem_l3169_316939

theorem box_volume_theorem : ∃ (x y z : ℕ+), 
  (x : ℚ) / 2 = (y : ℚ) / 5 ∧ (y : ℚ) / 5 = (z : ℚ) / 7 ∧ 
  (x : ℕ) * y * z = 70 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_theorem_l3169_316939


namespace NUMINAMATH_CALUDE_joan_snow_volume_l3169_316985

/-- The volume of snow on a rectangular driveway -/
def snow_volume (length width depth : ℚ) : ℚ :=
  length * width * depth

/-- Proof that the volume of snow on Joan's driveway is 90 cubic feet -/
theorem joan_snow_volume :
  snow_volume 40 3 (3/4) = 90 := by
  sorry

end NUMINAMATH_CALUDE_joan_snow_volume_l3169_316985


namespace NUMINAMATH_CALUDE_C_is_circle_when_k_is_2_C_hyperbola_y_axis_implies_k_less_than_neg_one_l3169_316988

-- Define the curve C
def C (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (5 - k) = 1

-- Theorem 1: When k=2, curve C is a circle
theorem C_is_circle_when_k_is_2 :
  ∃ (r : ℝ), ∀ (x y : ℝ), C 2 x y ↔ x^2 + y^2 = r^2 :=
sorry

-- Theorem 2: If curve C is a hyperbola with foci on the y-axis, then k < -1
theorem C_hyperbola_y_axis_implies_k_less_than_neg_one :
  (∃ (a b : ℝ), ∀ (x y : ℝ), C k x y ↔ y^2 / a^2 - x^2 / b^2 = 1) → k < -1 :=
sorry

end NUMINAMATH_CALUDE_C_is_circle_when_k_is_2_C_hyperbola_y_axis_implies_k_less_than_neg_one_l3169_316988


namespace NUMINAMATH_CALUDE_halloween_goodie_bags_minimum_cost_l3169_316945

/-- Represents the theme of a Halloween goodie bag -/
inductive Theme
| Vampire
| Pumpkin

/-- Represents the purchase options available -/
inductive PurchaseOption
| Package
| Individual

theorem halloween_goodie_bags_minimum_cost 
  (total_students : ℕ)
  (vampire_requests : ℕ)
  (pumpkin_requests : ℕ)
  (package_price : ℕ)
  (package_size : ℕ)
  (individual_price : ℕ)
  (discount_buy : ℕ)
  (discount_free : ℕ)
  (h1 : total_students = 25)
  (h2 : vampire_requests = 11)
  (h3 : pumpkin_requests = 14)
  (h4 : vampire_requests + pumpkin_requests = total_students)
  (h5 : package_price = 3)
  (h6 : package_size = 5)
  (h7 : individual_price = 1)
  (h8 : discount_buy = 3)
  (h9 : discount_free = 1) :
  (∃ (vampire_packages vampire_individuals pumpkin_packages : ℕ),
    vampire_packages * package_size + vampire_individuals ≥ vampire_requests ∧
    pumpkin_packages * package_size ≥ pumpkin_requests ∧
    (vampire_packages * package_price + vampire_individuals * individual_price +
     (pumpkin_packages / discount_buy * (discount_buy - discount_free) + pumpkin_packages % discount_buy) * package_price = 13)) :=
by sorry


end NUMINAMATH_CALUDE_halloween_goodie_bags_minimum_cost_l3169_316945


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l3169_316928

/-- The volume of a sphere inscribed in a cube with edge length 10 inches -/
theorem inscribed_sphere_volume :
  let cube_edge : ℝ := 10
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = (500 / 3) * π := by
sorry


end NUMINAMATH_CALUDE_inscribed_sphere_volume_l3169_316928


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l3169_316951

/-- Given a rectangle with perimeter 60 feet and area 221 square feet, 
    the length of the longer side is 17 feet. -/
theorem rectangle_longer_side (x y : ℝ) 
  (h_perimeter : 2 * x + 2 * y = 60) 
  (h_area : x * y = 221) 
  (h_longer : x ≥ y) : x = 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l3169_316951


namespace NUMINAMATH_CALUDE_evaluate_F_with_f_l3169_316991

-- Define function f
def f (a : ℝ) : ℝ := a^2 - 1

-- Define function F
def F (a b : ℝ) : ℝ := 3*b^2 + 2*a

-- Theorem statement
theorem evaluate_F_with_f : F 2 (f 3) = 196 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_F_with_f_l3169_316991


namespace NUMINAMATH_CALUDE_complex_fraction_sum_zero_l3169_316999

theorem complex_fraction_sum_zero : 
  let i : ℂ := Complex.I
  ((1 + i) / (1 - i)) ^ 2017 + ((1 - i) / (1 + i)) ^ 2017 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_zero_l3169_316999


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l3169_316978

/-- Calculates the total amount given by a customer in a restaurant scenario -/
def total_given (check_amount : ℝ) (tax_rate : ℝ) (tip : ℝ) : ℝ :=
  check_amount * (1 + tax_rate) + tip

/-- Proves that given specific values for check amount, tax rate, and tip, 
    the total amount given by the customer is $20.00 -/
theorem restaurant_bill_proof : 
  total_given 15 0.2 2 = 20 := by sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l3169_316978


namespace NUMINAMATH_CALUDE_average_of_remaining_two_l3169_316926

theorem average_of_remaining_two (total_avg : ℝ) (avg1 : ℝ) (avg2 : ℝ) :
  total_avg = 3.95 →
  avg1 = 4.2 →
  avg2 = 3.8000000000000007 →
  (6 * total_avg - 2 * avg1 - 2 * avg2) / 2 = 3.85 := by
sorry


end NUMINAMATH_CALUDE_average_of_remaining_two_l3169_316926


namespace NUMINAMATH_CALUDE_min_value_theorem_l3169_316922

theorem min_value_theorem (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) 
  (h_inequality : b + c ≥ a + d) : 
  (b / (c + d)) + (c / (a + b)) ≥ Real.sqrt 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3169_316922


namespace NUMINAMATH_CALUDE_average_visitors_theorem_l3169_316953

/-- Calculates the average number of visitors per day in a 30-day month starting with a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalDays : ℕ := 30
  let sundays : ℕ := 4
  let otherDays : ℕ := totalDays - sundays
  let totalVisitors : ℕ := sundayVisitors * sundays + otherDayVisitors * otherDays
  (totalVisitors : ℚ) / totalDays

theorem average_visitors_theorem (sundayVisitors otherDayVisitors : ℕ) 
  (h1 : sundayVisitors = 510) (h2 : otherDayVisitors = 240) : 
  averageVisitorsPerDay sundayVisitors otherDayVisitors = 276 := by
  sorry

#eval averageVisitorsPerDay 510 240

end NUMINAMATH_CALUDE_average_visitors_theorem_l3169_316953


namespace NUMINAMATH_CALUDE_function_max_value_l3169_316942

theorem function_max_value (x : ℝ) (h : x < 5/4) :
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_function_max_value_l3169_316942


namespace NUMINAMATH_CALUDE_inequality_group_C_equality_group_A_equality_group_B_equality_group_D_l3169_316998

theorem inequality_group_C (a b : ℝ) : ∃ a b, 3 * (a + b) ≠ 3 * a + b :=
sorry

theorem equality_group_A (a b : ℝ) : a + b = b + a :=
sorry

theorem equality_group_B (a : ℝ) : 3 * a = a + a + a :=
sorry

theorem equality_group_D (a : ℝ) : a ^ 3 = a * a * a :=
sorry

end NUMINAMATH_CALUDE_inequality_group_C_equality_group_A_equality_group_B_equality_group_D_l3169_316998


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l3169_316901

-- Define the lines
def line1 (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 1 = 0
def line2 (a x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define parallelism
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂, f x₁ y₁ → f x₂ y₂ → g x₁ y₁ → g x₂ y₂ → 
    (y₂ - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x₂ - x₁)

-- Theorem statement
theorem parallel_lines_condition (a : ℝ) :
  parallel (line1 a) (line2 a) → a = -1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l3169_316901


namespace NUMINAMATH_CALUDE_machine_operation_l3169_316910

theorem machine_operation (x : ℤ) : 26 + x - 6 = 35 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_machine_operation_l3169_316910


namespace NUMINAMATH_CALUDE_conference_attendees_payment_registration_l3169_316930

theorem conference_attendees_payment_registration (
  early_registration : Real) 
  (mid_registration : Real)
  (late_registration : Real)
  (credit_card_percent : Real)
  (debit_card_percent : Real)
  (other_payment_percent : Real) :
  early_registration = 80 →
  mid_registration = 12 →
  late_registration = 100 - early_registration - mid_registration →
  credit_card_percent + debit_card_percent + other_payment_percent = 100 →
  credit_card_percent = 20 →
  debit_card_percent = 60 →
  other_payment_percent = 20 →
  early_registration + mid_registration = 
    (credit_card_percent + debit_card_percent + other_payment_percent) * 
    (early_registration + mid_registration) / 100 :=
by sorry

end NUMINAMATH_CALUDE_conference_attendees_payment_registration_l3169_316930


namespace NUMINAMATH_CALUDE_min_sum_tangents_l3169_316955

theorem min_sum_tangents (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  a = 2 * b * Real.sin C →  -- Given condition
  8 ≤ Real.tan A + Real.tan B + Real.tan C ∧
  (∃ (A' B' C' : Real), 0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = π ∧
    Real.tan A' + Real.tan B' + Real.tan C' = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_tangents_l3169_316955


namespace NUMINAMATH_CALUDE_parameterized_to_ordinary_equation_l3169_316976

theorem parameterized_to_ordinary_equation :
  ∀ (x y t : ℝ),
  (x = Real.sqrt t ∧ y = 2 * Real.sqrt (1 - t)) →
  (x^2 + y^2 / 4 = 1 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_parameterized_to_ordinary_equation_l3169_316976


namespace NUMINAMATH_CALUDE_marble_theorem_l3169_316961

def marble_problem (wolfgang ludo michael shania gabriel : ℕ) : Prop :=
  wolfgang = 16 ∧
  ludo = wolfgang + wolfgang / 4 ∧
  michael = 2 * (wolfgang + ludo) / 3 ∧
  shania = 2 * ludo ∧
  gabriel = wolfgang + ludo + michael + shania - 1 ∧
  (wolfgang + ludo + michael + shania + gabriel) / 5 = 39

theorem marble_theorem : ∃ wolfgang ludo michael shania gabriel : ℕ,
  marble_problem wolfgang ludo michael shania gabriel := by
  sorry

end NUMINAMATH_CALUDE_marble_theorem_l3169_316961


namespace NUMINAMATH_CALUDE_star_ratio_equals_two_thirds_l3169_316965

-- Define the ⋆ operation
def star (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem star_ratio_equals_two_thirds :
  (star 3 2 : ℚ) / (star 2 3 : ℚ) = 2/3 := by sorry

end NUMINAMATH_CALUDE_star_ratio_equals_two_thirds_l3169_316965


namespace NUMINAMATH_CALUDE_age_difference_is_two_l3169_316971

/-- The age difference between Jayson's dad and mom -/
def age_difference (jayson_age : ℕ) (mom_age_at_birth : ℕ) : ℕ :=
  (4 * jayson_age) - (mom_age_at_birth + jayson_age)

/-- Theorem stating the age difference between Jayson's dad and mom is 2 years -/
theorem age_difference_is_two :
  age_difference 10 28 = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_two_l3169_316971


namespace NUMINAMATH_CALUDE_box_dimensions_l3169_316958

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  h₁ : 0 < a₁
  h₂ : 0 < a₂
  h₃ : 0 < a₃
  h₄ : a₁ ≤ a₂
  h₅ : a₂ ≤ a₃

/-- The volume of a cube -/
def cubeVolume : ℝ := 2

/-- The proportion of the box filled by cubes -/
def fillProportion : ℝ := 0.4

/-- Checks if the given dimensions satisfy the cube-filling condition -/
def satisfiesCubeFilling (d : BoxDimensions) : Prop :=
  ∃ (n : ℕ), n * cubeVolume = fillProportion * (d.a₁ * d.a₂ * d.a₃)

/-- The theorem stating the possible box dimensions -/
theorem box_dimensions : 
  ∀ d : BoxDimensions, satisfiesCubeFilling d → 
    (d.a₁ = 2 ∧ d.a₂ = 3 ∧ d.a₃ = 5) ∨ (d.a₁ = 2 ∧ d.a₂ = 5 ∧ d.a₃ = 6) := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_l3169_316958


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3169_316986

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3169_316986


namespace NUMINAMATH_CALUDE_binary_to_decimal_l3169_316970

theorem binary_to_decimal (b : List Bool) :
  (b.reverse.enum.map (λ (i, x) => if x then 2^i else 0)).sum = 45 :=
sorry

end NUMINAMATH_CALUDE_binary_to_decimal_l3169_316970


namespace NUMINAMATH_CALUDE_race_distance_proof_l3169_316906

/-- Represents the total distance of a race in meters. -/
def race_distance : ℝ := 88

/-- Represents the time taken by Runner A to complete the race in seconds. -/
def time_A : ℝ := 20

/-- Represents the time taken by Runner B to complete the race in seconds. -/
def time_B : ℝ := 25

/-- Represents the distance by which Runner A beats Runner B in meters. -/
def beating_distance : ℝ := 22

theorem race_distance_proof : 
  race_distance = 88 ∧ 
  (race_distance / time_A) * time_B = race_distance + beating_distance :=
sorry

end NUMINAMATH_CALUDE_race_distance_proof_l3169_316906


namespace NUMINAMATH_CALUDE_max_value_of_product_sum_l3169_316929

theorem max_value_of_product_sum (w x y z : ℝ) 
  (nonneg_w : 0 ≤ w) (nonneg_x : 0 ≤ x) (nonneg_y : 0 ≤ y) (nonneg_z : 0 ≤ z)
  (sum_condition : w + x + y + z = 200) :
  w * x + w * y + y * z + z * x ≤ 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_product_sum_l3169_316929


namespace NUMINAMATH_CALUDE_triangle_inequality_l3169_316964

theorem triangle_inequality (a b c r R s : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ R > 0 ∧ s > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : s = (a + b + c) / 2)
  (h_inradius : r = (a * b * c) / (4 * s))
  (h_circumradius : R = (a * b * c) / (4 * area))
  (h_area : area = Real.sqrt (s * (s - a) * (s - b) * (s - c))) :
  (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≤ 
  (r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s)) ∧
  ((1 / (a + b) + 1 / (a + c) + 1 / (b + c) = 
    r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s)) ↔ 
   (a = b ∧ b = c)) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3169_316964


namespace NUMINAMATH_CALUDE_xy_value_l3169_316941

theorem xy_value (x y : ℝ) (h : |x^3 - 1/8| + Real.sqrt (y - 4) = 0) : x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3169_316941


namespace NUMINAMATH_CALUDE_angle_B_obtuse_l3169_316993

theorem angle_B_obtuse (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Given conditions
  (c / b < Real.cos A) ∧ (0 < A) ∧ (A < Real.pi) →
  -- Conclusion: B is obtuse
  Real.pi / 2 < B :=
by
  sorry

end NUMINAMATH_CALUDE_angle_B_obtuse_l3169_316993


namespace NUMINAMATH_CALUDE_rectangle_area_is_eight_l3169_316902

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a rectangle on a 2D grid -/
structure GridRectangle where
  bottomLeft : GridPoint
  topRight : GridPoint

/-- Calculates the area of a grid rectangle -/
def gridRectangleArea (rect : GridRectangle) : ℤ :=
  (rect.topRight.x - rect.bottomLeft.x) * (rect.topRight.y - rect.bottomLeft.y)

theorem rectangle_area_is_eight :
  let rect : GridRectangle := {
    bottomLeft := { x := 0, y := 0 },
    topRight := { x := 4, y := 2 }
  }
  gridRectangleArea rect = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_eight_l3169_316902


namespace NUMINAMATH_CALUDE_slip_4_5_in_R_l3169_316900

-- Define the set of slips
def slips : List ℝ := [1, 1.5, 1.5, 2, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 5, 5.5]

-- Define the boxes
inductive Box
| P | Q | R | S | T | U

-- Define a distribution of slips to boxes
def Distribution := Box → List ℝ

-- Define the constraint that the sum in each box is an integer
def sumIsInteger (d : Distribution) : Prop :=
  ∀ b : Box, ∃ n : ℤ, (d b).sum = n

-- Define the constraint that the sums are consecutive integers
def consecutiveSums (d : Distribution) : Prop :=
  ∃ n : ℤ, (d Box.P).sum = n ∧
           (d Box.Q).sum = n + 1 ∧
           (d Box.R).sum = n + 2 ∧
           (d Box.S).sum = n + 3 ∧
           (d Box.T).sum = n + 4 ∧
           (d Box.U).sum = n + 5

-- Define the constraint that 1 is in box U and 2 is in box Q
def fixedSlips (d : Distribution) : Prop :=
  1 ∈ d Box.U ∧ 2 ∈ d Box.Q

-- Main theorem
theorem slip_4_5_in_R (d : Distribution) 
  (h1 : d Box.P ++ d Box.Q ++ d Box.R ++ d Box.S ++ d Box.T ++ d Box.U = slips)
  (h2 : sumIsInteger d)
  (h3 : consecutiveSums d)
  (h4 : fixedSlips d) :
  4.5 ∈ d Box.R :=
sorry

end NUMINAMATH_CALUDE_slip_4_5_in_R_l3169_316900


namespace NUMINAMATH_CALUDE_sum_digits_base8_888_l3169_316979

/-- Represents a number in a given base as a list of digits --/
def Digits := List Nat

/-- Converts a natural number to its representation in a given base --/
def toBase (n : Nat) (base : Nat) : Digits :=
  sorry

/-- Sums the digits in a list --/
def sumDigits (digits : Digits) : Nat :=
  sorry

/-- The sum of digits in the base 8 representation of 888₁₀ is 13 --/
theorem sum_digits_base8_888 :
  sumDigits (toBase 888 8) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_base8_888_l3169_316979


namespace NUMINAMATH_CALUDE_soccer_league_games_l3169_316908

/-- Calculate the number of games in a soccer league --/
theorem soccer_league_games (n : ℕ) (h : n = 11) : n * (n - 1) / 2 = 55 := by
  sorry

#check soccer_league_games

end NUMINAMATH_CALUDE_soccer_league_games_l3169_316908


namespace NUMINAMATH_CALUDE_equality_of_polynomials_l3169_316977

theorem equality_of_polynomials (a b : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 5 = (x - 2)^2 + a*(x - 2) + b) → 
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_equality_of_polynomials_l3169_316977


namespace NUMINAMATH_CALUDE_time_for_c_is_48_l3169_316952

/-- The time it takes for worker c to complete the work alone -/
def time_for_c (time_ab time_bc time_ca : ℚ) : ℚ :=
  let a := (1 / time_ab + 1 / time_ca - 1 / time_bc) / 2
  let b := (1 / time_ab + 1 / time_bc - 1 / time_ca) / 2
  let c := (1 / time_bc + 1 / time_ca - 1 / time_ab) / 2
  1 / c

/-- Theorem stating that given the conditions, c will take 48 days to do the work alone -/
theorem time_for_c_is_48 :
  time_for_c 6 8 12 = 48 := by sorry

end NUMINAMATH_CALUDE_time_for_c_is_48_l3169_316952


namespace NUMINAMATH_CALUDE_value_of_w_l3169_316943

theorem value_of_w (j p t q s w : ℝ) 
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (w / 100) * p)
  (h4 : q = 1.15 * p)
  (h5 : q = 0.70 * j)
  (h6 : s = 1.40 * t)
  (h7 : s = 0.90 * q) :
  w = 6.25 := by
sorry

end NUMINAMATH_CALUDE_value_of_w_l3169_316943


namespace NUMINAMATH_CALUDE_age_difference_proof_l3169_316924

/-- Proves that given the conditions in the problem, z is 1.5 decades younger than x -/
theorem age_difference_proof (x y z w : ℝ) 
  (h1 : x + y = y + z + 15)
  (h2 : ∃ k : ℕ+, w = k * (x + z))
  (h3 : x = 3 * z) :
  (x - z) / 10 = 1.5 := by sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3169_316924


namespace NUMINAMATH_CALUDE_largest_integer_divisibility_l3169_316967

theorem largest_integer_divisibility : ∃ (n : ℕ), n = 14 ∧ 
  (∀ (m : ℕ), m > n → ¬(∃ (k : ℤ), (m - 2)^2 * (m + 1) = k * (2*m - 1))) ∧
  (∃ (k : ℤ), (n - 2)^2 * (n + 1) = k * (2*n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_divisibility_l3169_316967


namespace NUMINAMATH_CALUDE_jamies_father_weight_loss_l3169_316974

/-- Jamie's father's weight loss problem -/
theorem jamies_father_weight_loss 
  (calories_burned_per_day : ℕ)
  (calories_eaten_per_day : ℕ)
  (calories_per_pound : ℕ)
  (days_to_lose_weight : ℕ)
  (h1 : calories_burned_per_day = 2500)
  (h2 : calories_eaten_per_day = 2000)
  (h3 : calories_per_pound = 3500)
  (h4 : days_to_lose_weight = 35) :
  (days_to_lose_weight * (calories_burned_per_day - calories_eaten_per_day)) / calories_per_pound = 5 := by
  sorry


end NUMINAMATH_CALUDE_jamies_father_weight_loss_l3169_316974


namespace NUMINAMATH_CALUDE_tan_product_equals_two_l3169_316968

theorem tan_product_equals_two : 
  (1 + Real.tan (23 * π / 180)) * (1 + Real.tan (22 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_equals_two_l3169_316968


namespace NUMINAMATH_CALUDE_expression_simplification_l3169_316918

theorem expression_simplification (p : ℝ) 
  (h1 : p^3 - p^2 + 2*p + 16 ≠ 0) 
  (h2 : p^2 + 2*p + 6 ≠ 0) : 
  (p^3 + 4*p^2 + 10*p + 12) / (p^3 - p^2 + 2*p + 16) * 
  (p^3 - 3*p^2 + 8*p) / (p^2 + 2*p + 6) = p := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3169_316918


namespace NUMINAMATH_CALUDE_complex_quadrant_l3169_316914

theorem complex_quadrant (z : ℂ) (h : (1 + Complex.I) * z = Complex.abs (1 + Complex.I)) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3169_316914


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3169_316927

theorem trigonometric_product_equals_one : 
  (1 - 1 / Real.cos (30 * π / 180)) * 
  (1 + 1 / Real.sin (60 * π / 180)) * 
  (1 - 1 / Real.sin (30 * π / 180)) * 
  (1 + 1 / Real.cos (60 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3169_316927


namespace NUMINAMATH_CALUDE_square_clock_area_l3169_316913

-- Define the side length of the square clock
def clock_side_length : ℝ := 30

-- Define the area of the square clock
def clock_area : ℝ := clock_side_length * clock_side_length

-- Theorem to prove
theorem square_clock_area : clock_area = 900 := by
  sorry

end NUMINAMATH_CALUDE_square_clock_area_l3169_316913


namespace NUMINAMATH_CALUDE_cleanup_solution_l3169_316949

/-- The time spent cleaning up eggs and toilet paper -/
def cleanup_problem (time_per_roll : ℕ) (total_time : ℕ) (num_eggs : ℕ) (num_rolls : ℕ) : Prop :=
  ∃ (time_per_egg : ℕ),
    time_per_egg * num_eggs + time_per_roll * num_rolls * 60 = total_time * 60 ∧
    time_per_egg = 15

/-- Theorem stating the solution to the cleanup problem -/
theorem cleanup_solution :
  cleanup_problem 30 225 60 7 := by
  sorry

end NUMINAMATH_CALUDE_cleanup_solution_l3169_316949


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l3169_316912

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 * a 5 = a 3 * a 6 →
  a 9 * a 10 = -8 →
  a 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l3169_316912


namespace NUMINAMATH_CALUDE_total_wheels_count_l3169_316987

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of wheels on a unicycle -/
def unicycle_wheels : ℕ := 1

/-- The number of wheels on a four-wheeled scooter -/
def scooter_wheels : ℕ := 4

/-- The number of adults riding bicycles -/
def adults_on_bicycles : ℕ := 6

/-- The number of children riding tricycles -/
def children_on_tricycles : ℕ := 15

/-- The number of teenagers riding unicycles -/
def teenagers_on_unicycles : ℕ := 3

/-- The number of children riding four-wheeled scooters -/
def children_on_scooters : ℕ := 8

/-- The total number of wheels Dimitri saw at the park -/
def total_wheels : ℕ := 
  adults_on_bicycles * bicycle_wheels +
  children_on_tricycles * tricycle_wheels +
  teenagers_on_unicycles * unicycle_wheels +
  children_on_scooters * scooter_wheels

theorem total_wheels_count : total_wheels = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_count_l3169_316987


namespace NUMINAMATH_CALUDE_james_total_money_l3169_316920

-- Define the currency types
inductive Currency
| USD
| EUR

-- Define the money type
structure Money where
  amount : ℚ
  currency : Currency

-- Define the exchange rate
def exchange_rate : ℚ := 1.20

-- Define James's wallet contents
def wallet_contents : List Money := [
  ⟨50, Currency.USD⟩,
  ⟨20, Currency.USD⟩,
  ⟨5, Currency.USD⟩
]

-- Define James's pocket contents
def pocket_contents : List Money := [
  ⟨20, Currency.USD⟩,
  ⟨10, Currency.USD⟩,
  ⟨5, Currency.EUR⟩
]

-- Define James's coin contents
def coin_contents : List Money := [
  ⟨0.25, Currency.USD⟩,
  ⟨0.25, Currency.USD⟩,
  ⟨0.10, Currency.USD⟩,
  ⟨0.10, Currency.USD⟩,
  ⟨0.10, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩
]

-- Function to convert EUR to USD
def convert_to_usd (m : Money) : Money :=
  match m.currency with
  | Currency.USD => m
  | Currency.EUR => ⟨m.amount * exchange_rate, Currency.USD⟩

-- Function to sum up all money in USD
def total_usd (money_list : List Money) : ℚ :=
  (money_list.map convert_to_usd).foldl (fun acc m => acc + m.amount) 0

-- Theorem statement
theorem james_total_money :
  total_usd (wallet_contents ++ pocket_contents ++ coin_contents) = 111.85 := by
  sorry

end NUMINAMATH_CALUDE_james_total_money_l3169_316920


namespace NUMINAMATH_CALUDE_N_subset_M_l3169_316994

-- Define the sets M and N
def M : Set ℝ := {x | x < 9}
def N : Set ℝ := {x | x^2 < 9}

-- State the theorem
theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_N_subset_M_l3169_316994


namespace NUMINAMATH_CALUDE_third_to_first_l3169_316932

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Definition of a point being in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Theorem: If P is in the third quadrant, then Q(-a, -b) is in the first quadrant -/
theorem third_to_first (P : Point) (hP : isInThirdQuadrant P) :
  let Q : Point := ⟨-P.x, -P.y⟩
  isInFirstQuadrant Q := by
  sorry

end NUMINAMATH_CALUDE_third_to_first_l3169_316932


namespace NUMINAMATH_CALUDE_unique_b_value_l3169_316995

theorem unique_b_value : ∃! b : ℚ, ∀ x : ℚ, 5 * (3 * x - b) = 3 * (5 * x - 9) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_b_value_l3169_316995


namespace NUMINAMATH_CALUDE_equidistant_function_property_l3169_316917

open Complex

theorem equidistant_function_property (a b : ℝ) :
  (∀ z : ℂ, abs ((a + b * I) * z - z) = abs ((a + b * I) * z - I)) →
  abs (a + b * I) = 10 →
  b^2 = (1/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_equidistant_function_property_l3169_316917


namespace NUMINAMATH_CALUDE_price_reduction_proof_l3169_316921

/-- Given the initial price of a box of cereal, the number of boxes bought, and the total amount paid,
    prove that the price reduction per box is correct. -/
theorem price_reduction_proof (initial_price : ℕ) (boxes_bought : ℕ) (total_paid : ℕ) :
  initial_price = 104 →
  boxes_bought = 20 →
  total_paid = 1600 →
  initial_price - (total_paid / boxes_bought) = 24 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_proof_l3169_316921


namespace NUMINAMATH_CALUDE_A_subset_B_l3169_316905

/-- Set A is defined as {x | x(x-1) < 0} -/
def A : Set ℝ := {x | x * (x - 1) < 0}

/-- Set B is defined as {y | y = x^2 for some real x} -/
def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

/-- Theorem: A is a subset of B -/
theorem A_subset_B : A ⊆ B := by sorry

end NUMINAMATH_CALUDE_A_subset_B_l3169_316905


namespace NUMINAMATH_CALUDE_increasing_function_implies_a_geq_5_l3169_316982

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem increasing_function_implies_a_geq_5 (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y < 4 → f a x < f a y) →
  a ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_increasing_function_implies_a_geq_5_l3169_316982


namespace NUMINAMATH_CALUDE_nadia_hannah_distance_ratio_l3169_316954

/-- Proves the ratio of Nadia's distance to Hannah's distance -/
theorem nadia_hannah_distance_ratio :
  ∀ (nadia_distance hannah_distance : ℕ) (k : ℕ),
    nadia_distance = 18 →
    nadia_distance + hannah_distance = 27 →
    nadia_distance = k * hannah_distance →
    nadia_distance / hannah_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_nadia_hannah_distance_ratio_l3169_316954


namespace NUMINAMATH_CALUDE_customer_difference_l3169_316933

theorem customer_difference (X Y Z : ℕ) 
  (h1 : X - Y = 10) 
  (h2 : 10 - Z = 4) : 
  X - 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_customer_difference_l3169_316933


namespace NUMINAMATH_CALUDE_semicircle_problem_l3169_316992

theorem semicircle_problem (N : ℕ) (r : ℝ) (h_positive : r > 0) : 
  let A := (N * π * r^2) / 2
  let B := (π * r^2 / 2) * (N^2 - N)
  (N ≥ 1) → (A / B = 1 / 24) → (N = 25) := by
sorry

end NUMINAMATH_CALUDE_semicircle_problem_l3169_316992


namespace NUMINAMATH_CALUDE_sweater_discount_percentage_l3169_316919

/-- Proves that the discount percentage is approximately 15.5% given the conditions -/
theorem sweater_discount_percentage (markup : ℝ) (profit : ℝ) :
  markup = 0.5384615384615385 →
  profit = 0.3 →
  let normal_price := 1 + markup
  let discounted_price := 1 + profit
  let discount := (normal_price - discounted_price) / normal_price
  abs (discount - 0.155) < 0.001 := by
sorry

end NUMINAMATH_CALUDE_sweater_discount_percentage_l3169_316919


namespace NUMINAMATH_CALUDE_range_of_m_solution_when_m_minimum_l3169_316931

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |5 - 2*x| - |2*x - 1|

-- Theorem for the range of m
theorem range_of_m :
  (∃ x, f m x = 0) → m ∈ Set.Ici 4 :=
sorry

-- Theorem for the solution of the inequality when m is minimum
theorem solution_when_m_minimum :
  let m : ℝ := 4
  ∀ x, |x - 3| + |x + m| ≤ 2*m ↔ x ∈ Set.Icc (-9/2) (7/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_solution_when_m_minimum_l3169_316931


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l3169_316903

/-- Given a mixture of water and salt solution, calculate the volume of salt solution needed --/
theorem salt_solution_mixture (x : ℝ) : 
  (1 : ℝ) + x > 0 →  -- Total volume is positive
  0.6 * x = 0.2 * (1 + x) → -- Salt conservation equation
  x = 0.5 := by
sorry


end NUMINAMATH_CALUDE_salt_solution_mixture_l3169_316903


namespace NUMINAMATH_CALUDE_project_duration_is_four_days_l3169_316960

/-- Calculates the number of days taken to finish a project given the number of naps, hours per nap, and working hours. -/
def projectDuration (numNaps : ℕ) (hoursPerNap : ℕ) (workingHours : ℕ) : ℚ :=
  let totalHours := numNaps * hoursPerNap + workingHours
  totalHours / 24

/-- Theorem stating that under the given conditions, the project duration is 4 days. -/
theorem project_duration_is_four_days :
  projectDuration 6 7 54 = 4 := by
  sorry

#eval projectDuration 6 7 54

end NUMINAMATH_CALUDE_project_duration_is_four_days_l3169_316960


namespace NUMINAMATH_CALUDE_pyramid_face_area_l3169_316972

-- Define the pyramid
structure SquareBasedPyramid where
  baseEdge : ℝ
  lateralEdge : ℝ

-- Define the problem
theorem pyramid_face_area (p : SquareBasedPyramid) 
  (h_base : p.baseEdge = 8)
  (h_lateral : p.lateralEdge = 7) : 
  Real.sqrt ((4 * p.baseEdge * Real.sqrt (p.lateralEdge ^ 2 - (p.baseEdge / 2) ^ 2)) ^ 2) = 16 * Real.sqrt 33 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_face_area_l3169_316972


namespace NUMINAMATH_CALUDE_vector_magnitude_cosine_sine_l3169_316904

theorem vector_magnitude_cosine_sine (α : Real) : 
  let a : Fin 2 → Real := ![Real.cos α, Real.sin α]
  ‖a‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_cosine_sine_l3169_316904


namespace NUMINAMATH_CALUDE_line_equation_forms_l3169_316907

/-- Given a line with equation (3x-2)/4 - (2y-1)/2 = 1, prove its various forms -/
theorem line_equation_forms (x y : ℝ) :
  (3*x - 2)/4 - (2*y - 1)/2 = 1 →
  (3*x - 8*y - 2 = 0) ∧
  (y = (3/8)*x - 1/4) ∧
  (x/(2/3) + y/(-1/4) = 1) ∧
  ((3/Real.sqrt 73)*x - (8/Real.sqrt 73)*y - (2/Real.sqrt 73) = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_forms_l3169_316907


namespace NUMINAMATH_CALUDE_sams_letters_l3169_316947

theorem sams_letters (letters_tuesday : ℕ) (average_per_day : ℕ) (total_days : ℕ) :
  letters_tuesday = 7 →
  average_per_day = 5 →
  total_days = 2 →
  (average_per_day * total_days - letters_tuesday : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sams_letters_l3169_316947


namespace NUMINAMATH_CALUDE_john_mission_duration_l3169_316946

theorem john_mission_duration :
  let initial_duration : ℝ := 5
  let first_mission_duration : ℝ := initial_duration * (1 + 0.6)
  let second_mission_duration : ℝ := first_mission_duration * 0.5
  let third_mission_duration : ℝ := min (2 * second_mission_duration) (first_mission_duration * 0.8)
  let fourth_mission_duration : ℝ := 3 + (third_mission_duration * 0.5)
  first_mission_duration + second_mission_duration + third_mission_duration + fourth_mission_duration = 24.6 :=
by sorry

end NUMINAMATH_CALUDE_john_mission_duration_l3169_316946


namespace NUMINAMATH_CALUDE_b_investment_is_4000_l3169_316959

/-- Represents the investment and profit distribution in a partnership --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that under given conditions, B's investment is 4000 --/
theorem b_investment_is_4000 (p : Partnership)
  (h1 : p.a_investment = 8000)
  (h2 : p.c_investment = 2000)
  (h3 : p.total_profit = 252000)
  (h4 : p.c_profit_share = 36000)
  (h5 : p.c_investment * p.total_profit = p.c_profit_share * (p.a_investment + p.b_investment + p.c_investment)) :
  p.b_investment = 4000 := by
  sorry


end NUMINAMATH_CALUDE_b_investment_is_4000_l3169_316959


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l3169_316948

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 16) :
  ∃ (max : ℝ), (∀ a b : ℝ, a^2 + b^2 = 16 → |a| + |b| ≤ max) ∧ max = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l3169_316948


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l3169_316963

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 2)
  (h_sum : a 3 + a 5 = 10) :
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l3169_316963


namespace NUMINAMATH_CALUDE_number_equation_l3169_316981

theorem number_equation (x : ℝ) : 2 * x + 5 = 17 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3169_316981


namespace NUMINAMATH_CALUDE_magic_king_seasons_l3169_316911

theorem magic_king_seasons (total_episodes : ℕ) 
  (episodes_first_half : ℕ) (episodes_second_half : ℕ) :
  total_episodes = 225 ∧ 
  episodes_first_half = 20 ∧ 
  episodes_second_half = 25 →
  ∃ (seasons : ℕ), 
    seasons = 10 ∧
    total_episodes = (seasons / 2) * episodes_first_half + 
                     (seasons / 2) * episodes_second_half :=
by sorry

end NUMINAMATH_CALUDE_magic_king_seasons_l3169_316911


namespace NUMINAMATH_CALUDE_smallest_number_l3169_316940

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def is_smallest (n : Nat) (lst : List Nat) : Prop :=
  ∀ m ∈ lst, n ≤ m

theorem smallest_number :
  let n1 := base_to_decimal [8, 5] 9
  let n2 := base_to_decimal [2, 1, 0] 6
  let n3 := base_to_decimal [1, 0, 0, 0] 4
  let n4 := base_to_decimal [1, 1, 1, 1, 1, 1] 2
  is_smallest n4 [n1, n2, n3, n4] := by
sorry

end NUMINAMATH_CALUDE_smallest_number_l3169_316940


namespace NUMINAMATH_CALUDE_medicine_box_theorem_l3169_316969

/-- Represents the number of tablets of each medicine type in a box -/
structure MedicineBox where
  tabletA : ℕ
  tabletB : ℕ

/-- Calculates the minimum number of tablets to extract to ensure at least two of each type -/
def minExtract (box : MedicineBox) : ℕ :=
  box.tabletA + 6

theorem medicine_box_theorem (box : MedicineBox) 
  (h1 : box.tabletA = 10)
  (h2 : minExtract box = 16) :
  box.tabletB ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_medicine_box_theorem_l3169_316969
