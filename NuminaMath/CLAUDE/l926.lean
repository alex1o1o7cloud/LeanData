import Mathlib

namespace NUMINAMATH_CALUDE_flower_bed_weeds_count_l926_92656

/-- The number of weeds in the flower bed -/
def flower_bed_weeds : ℕ := 11

/-- The number of weeds in the vegetable patch -/
def vegetable_patch_weeds : ℕ := 14

/-- The number of weeds in the grass around the fruit trees -/
def grass_weeds : ℕ := 32

/-- The amount Lucille earns per weed in cents -/
def cents_per_weed : ℕ := 6

/-- The cost of the soda in cents -/
def soda_cost : ℕ := 99

/-- The amount of money Lucille has left in cents -/
def money_left : ℕ := 147

theorem flower_bed_weeds_count : 
  flower_bed_weeds = 11 :=
by sorry

end NUMINAMATH_CALUDE_flower_bed_weeds_count_l926_92656


namespace NUMINAMATH_CALUDE_compute_expression_l926_92601

theorem compute_expression : 16 * (125 / 2 + 25 / 4 + 9 / 16 + 1) = 1125 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l926_92601


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l926_92696

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ,
    n > 1 ∧
    ¬ Prime n ∧
    (∀ p : ℕ, Prime p → p < 20 → ¬ p ∣ n) ∧
    (∀ m : ℕ, m > 1 → ¬ Prime m → (∀ q : ℕ, Prime q → q < 20 → ¬ q ∣ m) → m ≥ n) ∧
    500 < n ∧
    n ≤ 600 :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l926_92696


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l926_92639

theorem polynomial_uniqueness (P : ℝ → ℝ) : 
  (∀ x, P x = P 0 + P 1 * x + P 3 * x^3) → 
  P (-1) = 3 → 
  ∀ x, P x = 3 + x + x^3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l926_92639


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l926_92653

theorem quadratic_equation_solution :
  let x₁ := -2 + Real.sqrt 2
  let x₂ := -2 - Real.sqrt 2
  x₁^2 + 4*x₁ + 2 = 0 ∧ x₂^2 + 4*x₂ + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l926_92653


namespace NUMINAMATH_CALUDE_equal_area_and_perimeter_l926_92610

-- Define the quadrilaterals
def quadrilateralA : List (ℝ × ℝ) := [(0,0), (3,0), (3,2), (0,3)]
def quadrilateralB : List (ℝ × ℝ) := [(0,0), (3,0), (3,3), (0,2)]

-- Function to calculate area of a quadrilateral
def area (quad : List (ℝ × ℝ)) : ℝ := sorry

-- Function to calculate perimeter of a quadrilateral
def perimeter (quad : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the areas and perimeters are equal
theorem equal_area_and_perimeter :
  area quadrilateralA = area quadrilateralB ∧
  perimeter quadrilateralA = perimeter quadrilateralB := by
  sorry

end NUMINAMATH_CALUDE_equal_area_and_perimeter_l926_92610


namespace NUMINAMATH_CALUDE_sum_product_bound_l926_92651

theorem sum_product_bound (a b c d : ℝ) (h : a + b + c + d = 1) :
  ∃ (x : ℝ), x ≤ 0.5 ∧ (ab + ac + ad + bc + bd + cd ≤ x) ∧
  ∀ (y : ℝ), ∃ (a' b' c' d' : ℝ), a' + b' + c' + d' = 1 ∧
  a'*b' + a'*c' + a'*d' + b'*c' + b'*d' + c'*d' < y :=
sorry

end NUMINAMATH_CALUDE_sum_product_bound_l926_92651


namespace NUMINAMATH_CALUDE_overall_mean_score_l926_92687

/-- Given the mean scores and ratios of students in three classes, prove the overall mean score --/
theorem overall_mean_score (m a e : ℕ) (M A E : ℝ) : 
  M = 78 → A = 68 → E = 82 →
  (m : ℝ) / a = 4 / 5 →
  ((m : ℝ) + a) / e = 9 / 2 →
  (M * m + A * a + E * e) / (m + a + e : ℝ) = 74.4 := by
  sorry

#check overall_mean_score

end NUMINAMATH_CALUDE_overall_mean_score_l926_92687


namespace NUMINAMATH_CALUDE_not_sufficient_for_parallelogram_l926_92613

/-- A quadrilateral with vertices A, B, C, and D -/
structure Quadrilateral (V : Type*) :=
  (A B C D : V)

/-- Parallelism relation between line segments -/
def Parallel {V : Type*} (AB CD : V × V) : Prop := sorry

/-- Equality of line segments -/
def SegmentEqual {V : Type*} (AB CD : V × V) : Prop := sorry

/-- Definition of a parallelogram -/
def IsParallelogram {V : Type*} (quad : Quadrilateral V) : Prop := sorry

/-- The main theorem: AB parallel to CD and AD = BC does not imply ABCD is a parallelogram -/
theorem not_sufficient_for_parallelogram {V : Type*} (quad : Quadrilateral V) :
  Parallel (quad.A, quad.B) (quad.C, quad.D) →
  SegmentEqual (quad.A, quad.D) (quad.B, quad.C) →
  ¬ (IsParallelogram quad) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_for_parallelogram_l926_92613


namespace NUMINAMATH_CALUDE_half_percent_of_160_l926_92678

theorem half_percent_of_160 : (1 / 2 * 1 / 100) * 160 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_half_percent_of_160_l926_92678


namespace NUMINAMATH_CALUDE_function_uniqueness_l926_92677

theorem function_uniqueness (f : ℝ → ℝ) (a : ℝ) : 
  ∃! y, f a = y :=
sorry

end NUMINAMATH_CALUDE_function_uniqueness_l926_92677


namespace NUMINAMATH_CALUDE_chord_length_on_circle_l926_92614

/-- The length of the chord intercepted by y=x on (x-0)^2+(y-2)^2=4 is 2√2 -/
theorem chord_length_on_circle (x y : ℝ) : 
  (x - 0)^2 + (y - 2)^2 = 4 → y = x → 
  ∃ (a b : ℝ), (a - 0)^2 + (b - 2)^2 = 4 ∧ b = a ∧ 
  Real.sqrt ((a - x)^2 + (b - y)^2) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_on_circle_l926_92614


namespace NUMINAMATH_CALUDE_pokemon_cards_total_l926_92695

def jenny_cards : ℕ := 6

def orlando_cards (jenny : ℕ) : ℕ := jenny + 2

def richard_cards (orlando : ℕ) : ℕ := orlando * 3

def total_cards (jenny orlando richard : ℕ) : ℕ := jenny + orlando + richard

theorem pokemon_cards_total :
  total_cards jenny_cards (orlando_cards jenny_cards) (richard_cards (orlando_cards jenny_cards)) = 38 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_total_l926_92695


namespace NUMINAMATH_CALUDE_jolene_bicycle_purchase_l926_92641

structure Income where
  babysitting : Nat
  babysittingRate : Nat
  carWashing : Nat
  carWashingRate : Nat
  dogWalking : Nat
  dogWalkingRate : Nat
  cashGift : Nat

structure BicycleOption where
  price : Nat
  discount : Nat

def calculateTotalIncome (income : Income) : Nat :=
  income.babysitting * income.babysittingRate +
  income.carWashing * income.carWashingRate +
  income.dogWalking * income.dogWalkingRate +
  income.cashGift

def calculateDiscountedPrice (option : BicycleOption) : Nat :=
  option.price - (option.price * option.discount / 100)

def canAfford (income : Nat) (price : Nat) : Prop :=
  income ≥ price

theorem jolene_bicycle_purchase (income : Income)
  (optionA optionB optionC : BicycleOption) :
  income.babysitting = 4 ∧
  income.babysittingRate = 30 ∧
  income.carWashing = 5 ∧
  income.carWashingRate = 12 ∧
  income.dogWalking = 3 ∧
  income.dogWalkingRate = 15 ∧
  income.cashGift = 40 ∧
  optionA.price = 250 ∧
  optionA.discount = 0 ∧
  optionB.price = 300 ∧
  optionB.discount = 10 ∧
  optionC.price = 350 ∧
  optionC.discount = 15 →
  canAfford (calculateTotalIncome income) (calculateDiscountedPrice optionA) ∧
  calculateTotalIncome income - calculateDiscountedPrice optionA = 15 :=
by sorry


end NUMINAMATH_CALUDE_jolene_bicycle_purchase_l926_92641


namespace NUMINAMATH_CALUDE_special_circle_equation_l926_92652

/-- A circle with specific properties -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_in_first_quadrant : 0 < center.1 ∧ 0 < center.2
  tangent_to_line : |4 * center.1 - 3 * center.2| = 5 * radius
  tangent_to_x_axis : center.2 = radius
  radius_is_one : radius = 1

/-- The standard equation of a circle given its center and radius -/
def circle_equation (c : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - c.1)^2 + (y - c.2)^2 = r^2

/-- Theorem stating that a SpecialCircle has the standard equation (x-2)^2 + (y-1)^2 = 1 -/
theorem special_circle_equation (C : SpecialCircle) (x y : ℝ) :
  circle_equation (2, 1) 1 x y ↔ circle_equation C.center C.radius x y :=
sorry

end NUMINAMATH_CALUDE_special_circle_equation_l926_92652


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l926_92671

def X : ℕ := 4444^4444

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def A : ℕ := sum_of_digits X

def B : ℕ := sum_of_digits A

theorem sum_of_digits_of_B_is_seven :
  sum_of_digits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l926_92671


namespace NUMINAMATH_CALUDE_power_function_through_2_4_l926_92633

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Theorem statement
theorem power_function_through_2_4 (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = 4) : 
  f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_2_4_l926_92633


namespace NUMINAMATH_CALUDE_box_height_is_15_l926_92690

/-- Proves that the height of a box is 15 inches given specific conditions --/
theorem box_height_is_15 (base_length : ℝ) (base_width : ℝ) (total_volume : ℝ) 
  (cost_per_box : ℝ) (min_spend : ℝ) (h : ℝ) : 
  base_length = 20 ∧ base_width = 20 ∧ total_volume = 3060000 ∧ 
  cost_per_box = 1.3 ∧ min_spend = 663 →
  h = 15 := by
  sorry

#check box_height_is_15

end NUMINAMATH_CALUDE_box_height_is_15_l926_92690


namespace NUMINAMATH_CALUDE_glass_bowls_percentage_gain_l926_92698

/-- Calculate the percentage gain from buying and selling glass bowls -/
theorem glass_bowls_percentage_gain 
  (total_bought : ℕ) 
  (cost_price : ℚ) 
  (total_sold : ℕ) 
  (selling_price : ℚ) 
  (broken : ℕ) 
  (h1 : total_bought = 250)
  (h2 : cost_price = 18)
  (h3 : total_sold = 200)
  (h4 : selling_price = 25)
  (h5 : broken = 30)
  (h6 : total_sold + broken ≤ total_bought) :
  (((total_sold : ℚ) * selling_price - (total_bought : ℚ) * cost_price) / 
   ((total_bought : ℚ) * cost_price)) * 100 = 100 / 9 := by
sorry

#eval (100 : ℚ) / 9  -- To show the approximate result

end NUMINAMATH_CALUDE_glass_bowls_percentage_gain_l926_92698


namespace NUMINAMATH_CALUDE_water_drinking_ratio_l926_92693

/-- Proof of the water drinking ratio problem -/
theorem water_drinking_ratio :
  let morning_water : ℝ := 1.5
  let total_water : ℝ := 6
  let afternoon_water : ℝ := total_water - morning_water
  afternoon_water / morning_water = 3 := by
  sorry

end NUMINAMATH_CALUDE_water_drinking_ratio_l926_92693


namespace NUMINAMATH_CALUDE_quadratic_intersection_count_l926_92616

/-- The quadratic function f(x) = x^2 - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The number of intersection points between f and the coordinate axes -/
def intersection_count : ℕ := 2

theorem quadratic_intersection_count :
  (∃! x, f x = 0) ∧ (f 0 ≠ 0) → intersection_count = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_count_l926_92616


namespace NUMINAMATH_CALUDE_cube_plus_linear_inequality_l926_92668

theorem cube_plus_linear_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4 * a * b := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_linear_inequality_l926_92668


namespace NUMINAMATH_CALUDE_battle_station_staffing_ways_l926_92620

/-- Represents the number of job openings -/
def num_jobs : ℕ := 5

/-- Represents the total number of candidates considered -/
def total_candidates : ℕ := 18

/-- Represents the number of candidates skilled in one area only -/
def specialized_candidates : ℕ := 6

/-- Represents the number of versatile candidates -/
def versatile_candidates : ℕ := total_candidates - specialized_candidates

/-- Represents the number of ways to select the specialized candidates -/
def specialized_selection_ways : ℕ := 2 * 2 * 1 * 1

/-- The main theorem stating the number of ways to staff the battle station -/
theorem battle_station_staffing_ways :
  specialized_selection_ways * versatile_candidates * (versatile_candidates - 1) = 528 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_ways_l926_92620


namespace NUMINAMATH_CALUDE_perfect_square_natural_number_l926_92626

theorem perfect_square_natural_number (n : ℕ) :
  (∃ k : ℕ, n^2 + 5*n + 13 = k^2) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_natural_number_l926_92626


namespace NUMINAMATH_CALUDE_extra_interest_proof_l926_92603

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

def investment_amount : ℝ := 7000
def high_rate : ℝ := 0.18
def low_rate : ℝ := 0.12
def investment_time : ℝ := 2

theorem extra_interest_proof :
  simple_interest investment_amount high_rate investment_time -
  simple_interest investment_amount low_rate investment_time = 840 := by
  sorry

end NUMINAMATH_CALUDE_extra_interest_proof_l926_92603


namespace NUMINAMATH_CALUDE_function_properties_l926_92607

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem function_properties (a b c : ℝ) :
  (∃ y, -3 < y ∧ y ≤ 0 ∧ f a b c (-1) = y ∧ f a b c 1 = y ∧ f a b c 2 = y) →
  a = -2 ∧ b = -1 ∧ -1 < c ∧ c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l926_92607


namespace NUMINAMATH_CALUDE_some_beautiful_objects_are_colorful_l926_92640

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Rose Beautiful Colorful : U → Prop)

-- State the theorem
theorem some_beautiful_objects_are_colorful :
  (∀ x, Rose x → Beautiful x) →  -- All roses are beautiful
  (∃ x, Colorful x ∧ Rose x) →   -- Some colorful objects are roses
  (∃ x, Beautiful x ∧ Colorful x) -- Some beautiful objects are colorful
  := by sorry

end NUMINAMATH_CALUDE_some_beautiful_objects_are_colorful_l926_92640


namespace NUMINAMATH_CALUDE_swimmer_speed_l926_92679

theorem swimmer_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 35) 
  (h2 : upstream_distance = 20) (h3 : downstream_time = 5) (h4 : upstream_time = 5) :
  ∃ (speed_still_water : ℝ), speed_still_water = 5.5 ∧
  ∃ (stream_speed : ℝ),
    (speed_still_water + stream_speed) * downstream_time = downstream_distance ∧
    (speed_still_water - stream_speed) * upstream_time = upstream_distance :=
by sorry

end NUMINAMATH_CALUDE_swimmer_speed_l926_92679


namespace NUMINAMATH_CALUDE_triangle_minimum_shortest_side_l926_92631

theorem triangle_minimum_shortest_side :
  ∀ a b : ℕ,
  a < b ∧ b < 3 * a →  -- Condition for unequal sides
  a + b + 3 * a = 120 →  -- Total number of matches
  a ≥ 18 →  -- Minimum value of shortest side
  ∃ (a₀ : ℕ), a₀ = 18 ∧ 
    ∃ (b₀ : ℕ), a₀ < b₀ ∧ b₀ < 3 * a₀ ∧ 
    a₀ + b₀ + 3 * a₀ = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_minimum_shortest_side_l926_92631


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l926_92673

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  ∃ (min : ℝ), min = 3 + 2 * Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → 2*x + y = 1 → 1/x + 1/y ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l926_92673


namespace NUMINAMATH_CALUDE_gold_weight_is_ten_l926_92691

def weights : List ℕ := List.range 19

theorem gold_weight_is_ten (iron_weights bronze_weights : List ℕ) 
  (h1 : iron_weights.length = 9)
  (h2 : bronze_weights.length = 9)
  (h3 : iron_weights ⊆ weights)
  (h4 : bronze_weights ⊆ weights)
  (h5 : (iron_weights.sum - bronze_weights.sum) = 90)
  (h6 : iron_weights ∩ bronze_weights = [])
  : weights.sum - iron_weights.sum - bronze_weights.sum = 10 := by
  sorry

#check gold_weight_is_ten

end NUMINAMATH_CALUDE_gold_weight_is_ten_l926_92691


namespace NUMINAMATH_CALUDE_slope_of_line_l926_92622

theorem slope_of_line (x y : ℝ) :
  4 * y = -6 * x + 12 → (y - 3 = -3/2 * (x - 0)) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l926_92622


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_l926_92669

/-- The minimum distance between a point on the ellipse x²/8 + y²/4 = 1 
    and the line 4x + 3y - 24 = 0 is (24 - 2√41) / 5 -/
theorem min_distance_ellipse_line : 
  let ellipse := {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}
  let line := {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 - 24 = 0}
  ∃ (d : ℝ), d = (24 - 2 * Real.sqrt 41) / 5 ∧ 
    (∀ p ∈ ellipse, ∀ q ∈ line, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ p ∈ ellipse, ∃ q ∈ line, d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_l926_92669


namespace NUMINAMATH_CALUDE_intersection_A_B_l926_92612

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | x^2 - 1 > 0}

theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l926_92612


namespace NUMINAMATH_CALUDE_strawberry_harvest_l926_92632

/-- Proves that the number of strawberries harvested from each plant is 14 --/
theorem strawberry_harvest (
  strawberry_plants : ℕ)
  (tomato_plants : ℕ)
  (tomatoes_per_plant : ℕ)
  (fruits_per_basket : ℕ)
  (strawberry_basket_price : ℕ)
  (tomato_basket_price : ℕ)
  (total_revenue : ℕ)
  (h1 : strawberry_plants = 5)
  (h2 : tomato_plants = 7)
  (h3 : tomatoes_per_plant = 16)
  (h4 : fruits_per_basket = 7)
  (h5 : strawberry_basket_price = 9)
  (h6 : tomato_basket_price = 6)
  (h7 : total_revenue = 186) :
  (total_revenue - tomato_basket_price * (tomato_plants * tomatoes_per_plant / fruits_per_basket)) / strawberry_basket_price * fruits_per_basket / strawberry_plants = 14 :=
by sorry

end NUMINAMATH_CALUDE_strawberry_harvest_l926_92632


namespace NUMINAMATH_CALUDE_square_side_length_l926_92683

theorem square_side_length (area : ℚ) (side : ℚ) : 
  area = 9 / 16 → side * side = area → side = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l926_92683


namespace NUMINAMATH_CALUDE_infinitely_many_common_terms_l926_92602

-- Define the arithmetic sequence
def a (n : ℕ) : ℤ := 3*n - 1

-- Define the geometric sequence
def b (n : ℕ) : ℕ := 2^n

-- State the properties of the sequences
axiom a2_eq_5 : a 2 = 5
axiom a8_eq_23 : a 8 = 23
axiom b1_eq_2 : b 1 = 2
axiom b_mul (s t : ℕ) : b (s + t) = b s * b t

-- Theorem statement
theorem infinitely_many_common_terms :
  ∀ m : ℕ, ∃ k : ℕ, k > m ∧ ∃ n : ℕ, b k = a n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_common_terms_l926_92602


namespace NUMINAMATH_CALUDE_max_cubes_in_box_l926_92681

/-- The volume of a rectangular box -/
def box_volume (length width height : ℕ) : ℕ :=
  length * width * height

/-- The volume of a cube -/
def cube_volume (side : ℕ) : ℕ :=
  side ^ 3

/-- The maximum number of cubes that can fit in a box -/
def max_cubes (box_length box_width box_height cube_side : ℕ) : ℕ :=
  (box_volume box_length box_width box_height) / (cube_volume cube_side)

theorem max_cubes_in_box :
  max_cubes 8 9 12 3 = 32 :=
by sorry

end NUMINAMATH_CALUDE_max_cubes_in_box_l926_92681


namespace NUMINAMATH_CALUDE_city_park_highest_difference_l926_92643

/-- Snowfall data for different locations --/
structure SnowfallData where
  mrsHilt : Float
  brecknockSchool : Float
  townLibrary : Float
  cityPark : Float

/-- Calculate the absolute difference between two snowfall measurements --/
def snowfallDifference (a b : Float) : Float :=
  (a - b).abs

/-- Determine the location with the highest snowfall difference compared to Mrs. Hilt's house --/
def highestSnowfallDifference (data : SnowfallData) : String :=
  let schoolDiff := snowfallDifference data.mrsHilt data.brecknockSchool
  let libraryDiff := snowfallDifference data.mrsHilt data.townLibrary
  let parkDiff := snowfallDifference data.mrsHilt data.cityPark
  if parkDiff > schoolDiff && parkDiff > libraryDiff then
    "City Park"
  else if schoolDiff > libraryDiff then
    "Brecknock Elementary School"
  else
    "Town Library"

/-- Theorem: The city park has the highest snowfall difference compared to Mrs. Hilt's house --/
theorem city_park_highest_difference (data : SnowfallData)
  (h1 : data.mrsHilt = 29.7)
  (h2 : data.brecknockSchool = 17.3)
  (h3 : data.townLibrary = 23.8)
  (h4 : data.cityPark = 12.6) :
  highestSnowfallDifference data = "City Park" := by
  sorry

end NUMINAMATH_CALUDE_city_park_highest_difference_l926_92643


namespace NUMINAMATH_CALUDE_circle_equation_l926_92628

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y = 0
def line2 (x y : ℝ) : Prop := x - y - 4 = 0
def line3 (x y : ℝ) : Prop := x + y = 0

-- Define tangency
def isTangent (c : Circle) (l : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), l x y ∧ ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)

-- Main theorem
theorem circle_equation (C : Circle) 
  (h1 : isTangent C line1)
  (h2 : isTangent C line2)
  (h3 : line3 C.center.1 C.center.2) :
  ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 2 ↔ ((x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l926_92628


namespace NUMINAMATH_CALUDE_circle_intersection_l926_92697

/-- The number of intersection points between two circles -/
def intersectionPoints (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : ℕ :=
  sorry

/-- Theorem: The circle centered at (0, 3) with radius 3 and the circle centered at (5, 0) with radius 5 intersect at 4 points -/
theorem circle_intersection :
  intersectionPoints (0, 3) (5, 0) 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_l926_92697


namespace NUMINAMATH_CALUDE_problem_solution_l926_92667

theorem problem_solution (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k)*(x - k) = x^3 - k*(x^2 + x + 3)) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l926_92667


namespace NUMINAMATH_CALUDE_base_b_square_l926_92629

theorem base_b_square (b : ℕ) (h : b > 1) :
  (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 3 * b + 1 → b = 7 :=
by sorry

end NUMINAMATH_CALUDE_base_b_square_l926_92629


namespace NUMINAMATH_CALUDE_crab_meat_per_dish_l926_92672

/-- Proves that given the conditions of Johnny's crab dish production, he uses 1.5 pounds of crab meat per dish. -/
theorem crab_meat_per_dish (dishes_per_day : ℕ) (crab_cost_per_pound : ℚ) 
  (weekly_crab_cost : ℚ) (operating_days : ℕ) :
  dishes_per_day = 40 →
  crab_cost_per_pound = 8 →
  weekly_crab_cost = 1920 →
  operating_days = 4 →
  (weekly_crab_cost / crab_cost_per_pound) / operating_days / dishes_per_day = 3/2 := by
  sorry

#check crab_meat_per_dish

end NUMINAMATH_CALUDE_crab_meat_per_dish_l926_92672


namespace NUMINAMATH_CALUDE_cookies_per_batch_l926_92642

/-- Given a bag of chocolate chips with 81 chips, used to make 3 batches of cookies,
    where each cookie contains 9 chips, prove that there are 3 cookies in each batch. -/
theorem cookies_per_batch (total_chips : ℕ) (num_batches : ℕ) (chips_per_cookie : ℕ) 
  (h1 : total_chips = 81)
  (h2 : num_batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips % num_batches = 0)
  (h5 : (total_chips / num_batches) % chips_per_cookie = 0) :
  (total_chips / num_batches) / chips_per_cookie = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_batch_l926_92642


namespace NUMINAMATH_CALUDE_thirteen_step_staircase_l926_92692

/-- 
Represents a staircase where each step is made of toothpicks following an arithmetic sequence.
The first step uses 3 toothpicks, and each subsequent step uses 2 more toothpicks than the previous one.
-/
def Staircase (n : ℕ) : ℕ := n * (n + 2)

/-- A staircase with 5 steps uses 55 toothpicks -/
axiom five_step_staircase : Staircase 5 = 55

/-- Theorem: A staircase with 13 steps uses 210 toothpicks -/
theorem thirteen_step_staircase : Staircase 13 = 210 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_step_staircase_l926_92692


namespace NUMINAMATH_CALUDE_distance_to_origin_l926_92658

theorem distance_to_origin : let M : ℝ × ℝ := (-3, 4)
                             Real.sqrt ((M.1 - 0)^2 + (M.2 - 0)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l926_92658


namespace NUMINAMATH_CALUDE_chromosome_size_homology_l926_92654

/-- Represents a chromosome -/
structure Chromosome where
  size : ℕ
  is_homologous : Bool
  has_centromere : Bool
  gene_order : List ℕ

/-- Represents a pair of chromosomes -/
structure ChromosomePair where
  chromosome1 : Chromosome
  chromosome2 : Chromosome

/-- Defines what it means for chromosomes to be homologous -/
def are_homologous (c1 c2 : Chromosome) : Prop :=
  c1.is_homologous = true ∧ c2.is_homologous = true

/-- Defines what it means for chromosomes to be sister chromatids -/
def are_sister_chromatids (c1 c2 : Chromosome) : Prop :=
  c1.size = c2.size ∧ c1.gene_order = c2.gene_order

/-- Defines a tetrad -/
def is_tetrad (cp : ChromosomePair) : Prop :=
  are_homologous cp.chromosome1 cp.chromosome2

theorem chromosome_size_homology (c1 c2 : Chromosome) :
  c1.size = c2.size → are_homologous c1 c2 → False :=
sorry

#check chromosome_size_homology

end NUMINAMATH_CALUDE_chromosome_size_homology_l926_92654


namespace NUMINAMATH_CALUDE_modulus_of_w_l926_92624

theorem modulus_of_w (w : ℂ) (h : w^2 = 48 - 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_w_l926_92624


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l926_92670

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_terms :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 17 6 n = 101 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l926_92670


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_skew_lines_parallel_planes_l926_92604

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem for proposition ②
theorem perpendicular_planes_parallel
  (n : Line) (α β : Plane)
  (h1 : perpendicular n α)
  (h2 : perpendicular n β) :
  parallel α β :=
sorry

-- Theorem for proposition ⑤
theorem skew_lines_parallel_planes
  (m n : Line) (α β : Plane)
  (h1 : skew m n)
  (h2 : contains α n)
  (h3 : line_parallel n β)
  (h4 : contains β m)
  (h5 : line_parallel m α) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_skew_lines_parallel_planes_l926_92604


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l926_92689

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  -(-(1 : ℚ) / n) = 1 / n := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l926_92689


namespace NUMINAMATH_CALUDE_equation_representations_l926_92635

-- Define the equations
def equation1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0
def equation2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- Define what it means for an equation to represent a line and a circle
def represents_line_and_circle (f : ℝ → ℝ → Prop) : Prop :=
  (∃ (a : ℝ), ∀ y, f a y) ∧ 
  (∃ (h k r : ℝ), ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2)

-- Define what it means for an equation to represent two points
def represents_two_points (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∨ y1 ≠ y2 ∧ 
    (∀ x y, f x y ↔ (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

-- State the theorem
theorem equation_representations : 
  represents_line_and_circle equation1 ∧ represents_two_points equation2 := by
  sorry

end NUMINAMATH_CALUDE_equation_representations_l926_92635


namespace NUMINAMATH_CALUDE_exists_valid_expression_l926_92623

def Expression := List (Fin 4 → ℕ)

def applyOps (nums : Fin 4 → ℕ) (ops : Fin 3 → Char) : ℕ :=
  let e1 := match ops 0 with
    | '+' => nums 0 + nums 1
    | '-' => nums 0 - nums 1
    | '×' => nums 0 * nums 1
    | _ => 0
  let e2 := match ops 1 with
    | '+' => e1 + nums 2
    | '-' => e1 - nums 2
    | '×' => e1 * nums 2
    | _ => 0
  match ops 2 with
    | '+' => e2 + nums 3
    | '-' => e2 - nums 3
    | '×' => e2 * nums 3
    | _ => 0

def isValidOps (ops : Fin 3 → Char) : Prop :=
  (ops 0 = '+' ∨ ops 0 = '-' ∨ ops 0 = '×') ∧
  (ops 1 = '+' ∨ ops 1 = '-' ∨ ops 1 = '×') ∧
  (ops 2 = '+' ∨ ops 2 = '-' ∨ ops 2 = '×') ∧
  (ops 0 ≠ ops 1) ∧ (ops 1 ≠ ops 2) ∧ (ops 0 ≠ ops 2)

theorem exists_valid_expression : ∃ (ops : Fin 3 → Char),
  isValidOps ops ∧ applyOps (λ i => [5, 4, 6, 3][i]) ops = 19 := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_expression_l926_92623


namespace NUMINAMATH_CALUDE_area_of_large_square_l926_92609

/-- Given three squares with side lengths a, b, and c, prove that the area of the largest square is 100 --/
theorem area_of_large_square (a b c : ℝ) 
  (h1 : a^2 = b^2 + 32)
  (h2 : 4*a = 4*c + 16) : 
  a^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_area_of_large_square_l926_92609


namespace NUMINAMATH_CALUDE_gardener_hourly_rate_l926_92659

/-- Gardening project cost calculation -/
theorem gardener_hourly_rate 
  (num_rose_bushes : ℕ) 
  (cost_per_rose_bush : ℚ) 
  (hours_per_day : ℕ) 
  (num_days : ℕ) 
  (soil_volume : ℕ) 
  (cost_per_cubic_foot : ℚ) 
  (total_project_cost : ℚ) : 
  num_rose_bushes = 20 →
  cost_per_rose_bush = 150 →
  hours_per_day = 5 →
  num_days = 4 →
  soil_volume = 100 →
  cost_per_cubic_foot = 5 →
  total_project_cost = 4100 →
  (total_project_cost - (num_rose_bushes * cost_per_rose_bush + soil_volume * cost_per_cubic_foot)) / (hours_per_day * num_days) = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_gardener_hourly_rate_l926_92659


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l926_92664

theorem smallest_constant_inequality (x y z : ℝ) :
  ∃ D : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ D * (x + y + z)) ∧
  D = -Real.sqrt (72 / 11) ∧
  ∀ E : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ E * (x + y + z)) → D ≤ E :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l926_92664


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l926_92627

/-- The maximum distance from the center of the circle x² + y² = 4 to the line mx + (5-2m)y - 2 = 0, where m ∈ ℝ, is 2√5/5. -/
theorem max_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  let line (m : ℝ) := {(x, y) : ℝ × ℝ | m*x + (5 - 2*m)*y - 2 = 0}
  ∀ m : ℝ, (⨆ p ∈ line m, dist (0, 0) p) = 2 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l926_92627


namespace NUMINAMATH_CALUDE_multiplication_properties_l926_92684

theorem multiplication_properties : 
  (∀ n : ℝ, n * 0 = 0) ∧ 
  (∀ n : ℝ, n * 1 = n) ∧ 
  (∀ n : ℝ, n * (-1) = -n) ∧ 
  (∃ a b : ℝ, a + b = 0 ∧ a * b ≠ 1) := by
sorry

end NUMINAMATH_CALUDE_multiplication_properties_l926_92684


namespace NUMINAMATH_CALUDE_constant_term_expansion_l926_92648

theorem constant_term_expansion : 
  let f (x : ℝ) := (x^2 + 2) * (x - 1/x)^6
  ∃ (g : ℝ → ℝ), (∀ x ≠ 0, f x = g x) ∧ g 0 = -25 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l926_92648


namespace NUMINAMATH_CALUDE_domain_relation_l926_92649

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+2)
def domain_f_plus_2 : Set ℝ := Set.Ioo (-2) 2

-- Theorem stating the relationship between the domains
theorem domain_relation (h : ∀ x, f (x + 2) ∈ domain_f_plus_2 ↔ x ∈ domain_f_plus_2) :
  ∀ x, f (x - 3) ∈ Set.Ioo 3 7 ↔ x ∈ Set.Ioo 3 7 :=
sorry

end NUMINAMATH_CALUDE_domain_relation_l926_92649


namespace NUMINAMATH_CALUDE_marks_radiator_cost_l926_92666

/-- Calculates the total cost for Mark's car radiator replacement. -/
def total_cost (labor_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  labor_hours * hourly_rate + part_cost

/-- Proves that the total cost for Mark's car radiator replacement is $300. -/
theorem marks_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end NUMINAMATH_CALUDE_marks_radiator_cost_l926_92666


namespace NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l926_92688

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 5x^2 - 11x + 2 -/
def quadratic_equation (x : ℝ) : ℝ := 5*x^2 - 11*x + 2

theorem discriminant_of_specific_quadratic :
  discriminant 5 (-11) 2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l926_92688


namespace NUMINAMATH_CALUDE_clinton_belts_l926_92650

/-- Represents the number of items Clinton has in his wardrobe -/
structure Wardrobe where
  shoes : ℕ
  belts : ℕ
  hats : ℕ

/-- Clinton's wardrobe satisfies the given conditions -/
def clinton_wardrobe (w : Wardrobe) : Prop :=
  w.shoes = 2 * w.belts ∧
  ∃ n : ℕ, w.belts = w.hats + n ∧
  w.hats = 5 ∧
  w.shoes = 14

theorem clinton_belts :
  ∀ w : Wardrobe, clinton_wardrobe w → w.belts = 7 := by
  sorry

end NUMINAMATH_CALUDE_clinton_belts_l926_92650


namespace NUMINAMATH_CALUDE_binomial_eight_five_l926_92661

theorem binomial_eight_five : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_eight_five_l926_92661


namespace NUMINAMATH_CALUDE_tucker_tissues_left_l926_92637

/-- The number of tissues Tucker has left -/
def tissues_left (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_used : ℕ) : ℕ :=
  boxes_bought * tissues_per_box - tissues_used

/-- Theorem: Tucker has 270 tissues left -/
theorem tucker_tissues_left :
  tissues_left 160 3 210 = 270 := by
  sorry

end NUMINAMATH_CALUDE_tucker_tissues_left_l926_92637


namespace NUMINAMATH_CALUDE_f_5_solution_set_l926_92600

def f (x : ℝ) : ℝ := x^2 + 12*x + 30

def f_5 (x : ℝ) : ℝ := f (f (f (f (f x))))

theorem f_5_solution_set :
  ∀ x : ℝ, f_5 x = 0 ↔ x = -6 - (6 : ℝ)^(1/32) ∨ x = -6 + (6 : ℝ)^(1/32) := by
  sorry

end NUMINAMATH_CALUDE_f_5_solution_set_l926_92600


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l926_92682

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / 4) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l926_92682


namespace NUMINAMATH_CALUDE_max_distinct_distance_selection_l926_92617

/-- A regular polygon inscribed in a circle -/
structure RegularPolygon :=
  (sides : ℕ)
  (vertices : Fin sides → ℝ × ℝ)

/-- The distance between two vertices of a regular polygon -/
def distance (p : RegularPolygon) (i j : Fin p.sides) : ℝ := sorry

/-- A selection of vertices from a regular polygon -/
def VertexSelection (p : RegularPolygon) := Fin p.sides → Bool

/-- The number of vertices in a selection -/
def selectionSize (p : RegularPolygon) (s : VertexSelection p) : ℕ := sorry

/-- Whether all distances between selected vertices are distinct -/
def distinctDistances (p : RegularPolygon) (s : VertexSelection p) : Prop := sorry

theorem max_distinct_distance_selection (p : RegularPolygon) 
  (h : p.sides = 21) :
  (∃ (s : VertexSelection p), selectionSize p s = 5 ∧ distinctDistances p s) ∧
  (∀ (s : VertexSelection p), selectionSize p s > 5 → ¬ distinctDistances p s) :=
sorry

end NUMINAMATH_CALUDE_max_distinct_distance_selection_l926_92617


namespace NUMINAMATH_CALUDE_f_has_no_boundary_point_l926_92619

-- Define the concept of a boundary point
def has_boundary_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ ≠ 0 ∧
    (∃ x₁ x₂ : ℝ, x₁ < x₀ ∧ x₀ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0)

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem stating that f does not have a boundary point
theorem f_has_no_boundary_point : ¬ has_boundary_point f := by
  sorry


end NUMINAMATH_CALUDE_f_has_no_boundary_point_l926_92619


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l926_92655

/-- Proves that given a selling price of Rs. 12,000 for 200 meters of cloth
    and a loss of Rs. 6 per meter, the cost price for one meter of cloth is Rs. 66. -/
theorem cost_price_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (h1 : total_meters = 200)
  (h2 : selling_price = 12000)
  (h3 : loss_per_meter = 6) :
  (selling_price + total_meters * loss_per_meter) / total_meters = 66 := by
  sorry

#check cost_price_per_meter

end NUMINAMATH_CALUDE_cost_price_per_meter_l926_92655


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l926_92686

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- Theorem: If a_1, a_3, and a_7 of an arithmetic sequence form a geometric sequence,
    then the common ratio of this geometric sequence is 2 -/
theorem arithmetic_geometric_ratio
  (seq : ArithmeticSequence)
  (h_geometric : (seq.a 3) ^ 2 = (seq.a 1) * (seq.a 7)) :
  (seq.a 3) / (seq.a 1) = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l926_92686


namespace NUMINAMATH_CALUDE_data_fraction_less_than_value_l926_92638

theorem data_fraction_less_than_value (data : List ℝ) (fraction : ℝ) (value : ℝ) : 
  data = [1, 2, 3, 4, 5, 5, 5, 5, 7, 11, 21] →
  fraction = 0.36363636363636365 →
  (data.filter (· < value)).length / data.length = fraction →
  value = 4 := by
  sorry

end NUMINAMATH_CALUDE_data_fraction_less_than_value_l926_92638


namespace NUMINAMATH_CALUDE_triangle_angle_bounds_l926_92644

noncomputable def largest_angle (a b : ℝ) : ℝ := 
  Real.arccos (a / (2 * b))

noncomputable def smallest_angle_case1 (a b : ℝ) : ℝ := 
  Real.arcsin (a / b)

noncomputable def smallest_angle_case2 (a b : ℝ) : ℝ := 
  Real.arccos (b / (2 * a))

theorem triangle_angle_bounds (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (x y : ℝ),
    (largest_angle a b ≤ x ∧ x < π) ∧
    ((b ≥ a * Real.sqrt 2 → 0 < y ∧ y ≤ smallest_angle_case1 a b) ∧
     (b ≤ a * Real.sqrt 2 → 0 < y ∧ y ≤ smallest_angle_case2 a b)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_bounds_l926_92644


namespace NUMINAMATH_CALUDE_bacteria_growth_l926_92636

theorem bacteria_growth (division_time : ℕ) (total_time : ℕ) (initial_count : ℕ) : 
  division_time = 20 → 
  total_time = 180 → 
  initial_count = 1 → 
  2 ^ (total_time / division_time) = 512 :=
by
  sorry

#check bacteria_growth

end NUMINAMATH_CALUDE_bacteria_growth_l926_92636


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l926_92694

theorem coconut_grove_problem (x : ℝ) 
  (yield_1 : (x + 2) * 40 = (x + 2) * 40)
  (yield_2 : x * 120 = x * 120)
  (yield_3 : (x - 2) * 180 = (x - 2) * 180)
  (average_yield : ((x + 2) * 40 + x * 120 + (x - 2) * 180) / (3 * x) = 100) :
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l926_92694


namespace NUMINAMATH_CALUDE_exists_valid_marking_l926_92630

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Fin 8) (y : Fin 8)

/-- Represents a marking of squares on the chessboard -/
def BoardMarking := Position → Bool

/-- Calculates the minimum number of rook moves between two positions given a board marking -/
def minRookMoves (start finish : Position) (marking : BoardMarking) : ℕ :=
  sorry

/-- Theorem stating the existence of a board marking satisfying the given conditions -/
theorem exists_valid_marking : 
  ∃ (marking : BoardMarking),
    (minRookMoves ⟨0, 0⟩ ⟨2, 3⟩ marking = 3) ∧ 
    (minRookMoves ⟨2, 3⟩ ⟨7, 7⟩ marking = 2) ∧
    (minRookMoves ⟨0, 0⟩ ⟨7, 7⟩ marking = 4) :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_marking_l926_92630


namespace NUMINAMATH_CALUDE_abs_five_implies_plus_minus_five_l926_92625

theorem abs_five_implies_plus_minus_five (x : ℝ) : |x| = 5 → x = 5 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_abs_five_implies_plus_minus_five_l926_92625


namespace NUMINAMATH_CALUDE_cos_double_angle_on_graph_l926_92611

-- Define the angle α
variable (α : Real)

-- Define the condition that the terminal side of α lies on y = -3x
def terminal_side_on_graph (α : Real) : Prop :=
  ∃ x : Real, Real.tan α = -3 ∧ x ≠ 0

-- State the theorem
theorem cos_double_angle_on_graph (α : Real) 
  (h : terminal_side_on_graph α) : Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_on_graph_l926_92611


namespace NUMINAMATH_CALUDE_fourth_tree_height_l926_92665

/-- Represents a row of trees with specific properties -/
structure TreeRow where
  tallestHeight : ℝ
  shortestHeight : ℝ
  angleTopLine : ℝ
  equalSpacing : Bool

/-- Calculates the height of the nth tree from the left -/
def heightOfNthTree (row : TreeRow) (n : ℕ) : ℝ :=
  sorry

/-- The main theorem stating the height of the 4th tree -/
theorem fourth_tree_height (row : TreeRow) 
  (h1 : row.tallestHeight = 2.8)
  (h2 : row.shortestHeight = 1.4)
  (h3 : row.angleTopLine = 45)
  (h4 : row.equalSpacing = true) :
  heightOfNthTree row 4 = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_tree_height_l926_92665


namespace NUMINAMATH_CALUDE_rectangular_to_polar_equivalence_l926_92645

/-- Given a curve C in the xy-plane, prove that its rectangular coordinate equation
    x^2 + y^2 - 2x = 0 is equivalent to the polar coordinate equation ρ = 2cosθ. -/
theorem rectangular_to_polar_equivalence :
  ∀ (x y ρ θ : ℝ),
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x^2 + y^2 - 2*x = 0) ↔ (ρ = 2 * Real.cos θ) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_equivalence_l926_92645


namespace NUMINAMATH_CALUDE_hyperbola_equation_l926_92606

/-- A hyperbola is defined by its equation and properties -/
structure Hyperbola where
  -- The equation of the hyperbola in the form (y²/a² - x²/b² = 1)
  a : ℝ
  b : ℝ
  -- The hyperbola passes through the point (2, -2)
  passes_through : a^2 * 4 - b^2 * 4 = a^2 * b^2
  -- The hyperbola has asymptotes y = ± (√2/2)x
  asymptotes : a / b = Real.sqrt 2 / 2

/-- The equation of the hyperbola is y²/2 - x²/4 = 1 -/
theorem hyperbola_equation (h : Hyperbola) : h.a^2 = 2 ∧ h.b^2 = 4 :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l926_92606


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l926_92605

theorem quadratic_equation_problem (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0) →
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9) →
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x * y = 15) →
  m = 3 * n →
  m + n = 180 := by
sorry


end NUMINAMATH_CALUDE_quadratic_equation_problem_l926_92605


namespace NUMINAMATH_CALUDE_rainfall_ratio_is_two_l926_92646

-- Define the parameters
def total_rainfall : ℝ := 180
def first_half_daily_rainfall : ℝ := 4
def days_in_november : ℕ := 30
def first_half_days : ℕ := 15

-- Define the theorem
theorem rainfall_ratio_is_two :
  let first_half_total := first_half_daily_rainfall * first_half_days
  let second_half_total := total_rainfall - first_half_total
  let second_half_days := days_in_november - first_half_days
  let second_half_daily_rainfall := second_half_total / second_half_days
  (second_half_daily_rainfall / first_half_daily_rainfall) = 2 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_ratio_is_two_l926_92646


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l926_92674

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (x : ℕ), x ≤ 9 ∧ (427398 - x) % 10 = 0 ∧ 
  ∀ (y : ℕ), y < x → (427398 - y) % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l926_92674


namespace NUMINAMATH_CALUDE_data_transmission_time_l926_92608

-- Define the number of packets
def num_packets : ℕ := 100

-- Define the number of bytes per packet
def bytes_per_packet : ℕ := 256

-- Define the transmission rate in bytes per second
def transmission_rate : ℕ := 200

-- Define the number of seconds in a minute
def seconds_per_minute : ℕ := 60

-- Theorem to prove
theorem data_transmission_time :
  (num_packets * bytes_per_packet) / transmission_rate / seconds_per_minute = 2 := by
  sorry


end NUMINAMATH_CALUDE_data_transmission_time_l926_92608


namespace NUMINAMATH_CALUDE_cos_double_angle_from_series_sum_l926_92676

theorem cos_double_angle_from_series_sum (θ : ℝ) :
  (∑' n, (Real.cos θ) ^ (2 * n) = 8) → Real.cos (2 * θ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_from_series_sum_l926_92676


namespace NUMINAMATH_CALUDE_fixed_OC_length_l926_92660

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point P inside the circle
def P (c : Circle) : ℝ × ℝ := sorry

-- Distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the chord AB
def chord (c : Circle) (p : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define point C
def pointC (c : Circle) (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem fixed_OC_length (c : Circle) : 
  let o := c.center
  let r := c.radius
  let p := P c
  let d := distance o p
  let oc_length := distance o (pointC c p)
  oc_length = Real.sqrt (2 * r^2 - d^2) := by sorry

end NUMINAMATH_CALUDE_fixed_OC_length_l926_92660


namespace NUMINAMATH_CALUDE_product_of_powers_inequality_l926_92675

theorem product_of_powers_inequality (a b : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) (hn : n ≥ 2) :
  (a^n + 1) * (b^n + 1) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_inequality_l926_92675


namespace NUMINAMATH_CALUDE_fraction_simplification_l926_92647

theorem fraction_simplification : 
  (1/3 + 1/5) / ((2/7) * (3/4) - 1/7) = 112/15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l926_92647


namespace NUMINAMATH_CALUDE_arctg_sum_pi_half_l926_92685

theorem arctg_sum_pi_half : Real.arctan 1 + Real.arctan (1/2) + Real.arctan (1/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctg_sum_pi_half_l926_92685


namespace NUMINAMATH_CALUDE_armands_guessing_game_l926_92618

theorem armands_guessing_game (x : ℤ) : x = 33 ↔ 3 * x = 2 * 51 - 3 := by
  sorry

end NUMINAMATH_CALUDE_armands_guessing_game_l926_92618


namespace NUMINAMATH_CALUDE_integral_curves_of_differential_equation_l926_92615

/-- The differential equation -/
def differential_equation (x y : ℝ) (dx dy : ℝ) : Prop :=
  6 * x * dx - 6 * y * dy = 2 * x^2 * y * dy - 3 * x * y^2 * dx

/-- The integral curve equation -/
def integral_curve (x y : ℝ) (C : ℝ) : Prop :=
  (x^2 + 3)^3 / (2 + y^2) = C

/-- Theorem stating that the integral curves of the given differential equation
    are described by the integral_curve equation -/
theorem integral_curves_of_differential_equation :
  ∀ (x y : ℝ) (C : ℝ),
  (∀ (dx dy : ℝ), differential_equation x y dx dy) →
  ∃ (C : ℝ), integral_curve x y C :=
sorry

end NUMINAMATH_CALUDE_integral_curves_of_differential_equation_l926_92615


namespace NUMINAMATH_CALUDE_range_of_m_l926_92657

theorem range_of_m (m : ℝ) : 
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) →
  (m ≥ 4 ∨ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l926_92657


namespace NUMINAMATH_CALUDE_vase_capacity_l926_92621

/-- The number of flowers each vase can hold -/
def flowers_per_vase (carnations roses vases : ℕ) : ℕ :=
  (carnations + roses) / vases

/-- Theorem: Given 7 carnations, 47 roses, and 9 vases, each vase can hold 6 flowers -/
theorem vase_capacity :
  flowers_per_vase 7 47 9 = 6 := by
sorry

end NUMINAMATH_CALUDE_vase_capacity_l926_92621


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l926_92662

theorem fraction_sum_theorem (x y : ℝ) (h : x ≠ y) :
  (x + y) / (x - y) + (x - y) / (x + y) = 3 →
  (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = 13/6 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l926_92662


namespace NUMINAMATH_CALUDE_no_x_squared_term_l926_92663

theorem no_x_squared_term (a : ℚ) : 
  (∀ x, (x + 1) * (x^2 - 5*a*x + a) = x^3 + (-5*a + 1)*x^2 + (-9*a)*x + a) → 
  a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l926_92663


namespace NUMINAMATH_CALUDE_violets_family_size_l926_92680

/-- Proves the number of children in Violet's family given ticket prices and total cost -/
theorem violets_family_size (adult_ticket : ℕ) (child_ticket : ℕ) (total_cost : ℕ) :
  adult_ticket = 35 →
  child_ticket = 20 →
  total_cost = 155 →
  ∃ (num_children : ℕ), adult_ticket + num_children * child_ticket = total_cost ∧ num_children = 6 :=
by sorry

end NUMINAMATH_CALUDE_violets_family_size_l926_92680


namespace NUMINAMATH_CALUDE_two_roots_condition_l926_92699

open Real

theorem two_roots_condition (k : ℝ) : 
  (∃ x y, x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧ 
          y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧ 
          x ≠ y ∧ 
          x * log x - k * x + 1 = 0 ∧ 
          y * log y - k * y + 1 = 0) ↔ 
  k ∈ Set.Ioo 1 (1 + 1/Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_two_roots_condition_l926_92699


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l926_92634

theorem quadratic_inequality_solution (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  ∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0 ↔ x < 1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l926_92634
