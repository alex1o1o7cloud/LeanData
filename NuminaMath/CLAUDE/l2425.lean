import Mathlib

namespace NUMINAMATH_CALUDE_wage_period_theorem_l2425_242556

/-- Represents the number of days a sum of money can pay wages -/
structure WagePeriod where
  b : ℕ  -- Days for B's wages
  c : ℕ  -- Days for C's wages
  both : ℕ  -- Days for both B and C's wages

/-- Given conditions on wage periods, proves the number of days both can be paid -/
theorem wage_period_theorem (w : WagePeriod) (hb : w.b = 12) (hc : w.c = 24) :
  w.both = 8 := by
  sorry

#check wage_period_theorem

end NUMINAMATH_CALUDE_wage_period_theorem_l2425_242556


namespace NUMINAMATH_CALUDE_sequence_is_quadratic_l2425_242585

/-- Checks if a sequence is consistent with a quadratic function --/
def is_quadratic_sequence (seq : List ℕ) : Prop :=
  let first_differences := List.zipWith (·-·) (seq.tail) seq
  let second_differences := List.zipWith (·-·) (first_differences.tail) first_differences
  second_differences.all (· = second_differences.head!)

/-- The given sequence of function values --/
def given_sequence : List ℕ := [1600, 1764, 1936, 2116, 2304, 2500, 2704, 2916]

theorem sequence_is_quadratic :
  is_quadratic_sequence given_sequence :=
sorry

end NUMINAMATH_CALUDE_sequence_is_quadratic_l2425_242585


namespace NUMINAMATH_CALUDE_hawks_score_l2425_242539

/-- The number of touchdowns scored by the Hawks -/
def touchdowns : ℕ := 3

/-- The number of points awarded for each touchdown -/
def points_per_touchdown : ℕ := 7

/-- The total points scored by the Hawks -/
def total_points : ℕ := touchdowns * points_per_touchdown

/-- Theorem stating that the total points scored by the Hawks is 21 -/
theorem hawks_score : total_points = 21 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l2425_242539


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l2425_242512

theorem polar_to_cartesian (ρ θ x y : Real) :
  ρ * Real.cos θ = 1 ↔ x + y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l2425_242512


namespace NUMINAMATH_CALUDE_flight_passengers_l2425_242582

theorem flight_passengers :
  ∀ (total_passengers : ℕ),
    (total_passengers : ℝ) * 0.4 = total_passengers * 0.4 →
    (total_passengers : ℝ) * 0.1 = total_passengers * 0.1 →
    (total_passengers : ℝ) * 0.9 = total_passengers - total_passengers * 0.1 →
    (total_passengers * 0.1 : ℝ) * (2/3) = total_passengers * 0.1 * (2/3) →
    (total_passengers : ℝ) * 0.4 - total_passengers * 0.1 * (2/3) = 40 →
    total_passengers = 120 :=
by sorry

end NUMINAMATH_CALUDE_flight_passengers_l2425_242582


namespace NUMINAMATH_CALUDE_parallel_lines_l2425_242588

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle Γ
def circle_gamma (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 8

-- Define point M on Γ
def point_on_gamma (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  circle_gamma x y

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define circle ⊙F
def circle_F (M : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  let (x_m, y_m) := M
  let (x_p, y_p) := P
  (x_p - 1)^2 + y_p^2 = (x_m - 1)^2 + y_m^2

-- Define line l tangent to ⊙F at M
def line_l (M A B : ℝ × ℝ) : Prop :=
  ∃ (k b : ℝ), ∀ (x y : ℝ),
    ((x, y) = M ∨ (x, y) = A ∨ (x, y) = B) → y = k*x + b

-- Define lines l₁ and l₂
def line_l1_l2 (A B Q R : ℝ × ℝ) : Prop :=
  ∃ (k1 b1 k2 b2 : ℝ),
    (∀ (x y : ℝ), ((x, y) = A ∨ (x, y) = Q) → y = k1*x + b1) ∧
    (∀ (x y : ℝ), ((x, y) = B ∨ (x, y) = R) → y = k2*x + b2)

-- Main theorem
theorem parallel_lines
  (M A B Q R : ℝ × ℝ)
  (h_M : point_on_gamma M)
  (h_A : parabola A.1 A.2)
  (h_B : parabola B.1 B.2)
  (h_l : line_l M A B)
  (h_F : circle_F M Q)
  (h_F' : circle_F M R)
  (h_l1_l2 : line_l1_l2 A B Q R) :
  -- Conclusion: l₁ is parallel to l₂
  ∃ (k1 b1 k2 b2 : ℝ),
    (∀ (x y : ℝ), ((x, y) = A ∨ (x, y) = Q) → y = k1*x + b1) ∧
    (∀ (x y : ℝ), ((x, y) = B ∨ (x, y) = R) → y = k2*x + b2) ∧
    k1 = k2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_l2425_242588


namespace NUMINAMATH_CALUDE_point_on_line_with_given_x_l2425_242569

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a straight line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem point_on_line_with_given_x (l : Line) (x : ℝ) :
  l.slope = 2 →
  l.yIntercept = 2 →
  x = 269 →
  ∃ p : Point, p.x = x ∧ pointOnLine p l ∧ p.y = 540 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_with_given_x_l2425_242569


namespace NUMINAMATH_CALUDE_fish_pond_problem_l2425_242561

theorem fish_pond_problem (initial_fish : ℕ) : 
  (∃ (initial_tadpoles : ℕ),
    initial_tadpoles = 3 * initial_fish ∧
    initial_tadpoles / 2 = (initial_fish - 7) + 32) →
  initial_fish = 50 := by
sorry

end NUMINAMATH_CALUDE_fish_pond_problem_l2425_242561


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2425_242501

/-- The interval of segmentation for systematic sampling -/
def intervalOfSegmentation (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The interval of segmentation for a population of 2000 and sample size of 40 is 50 -/
theorem systematic_sampling_interval :
  intervalOfSegmentation 2000 40 = 50 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2425_242501


namespace NUMINAMATH_CALUDE_garden_tomato_percentage_l2425_242523

theorem garden_tomato_percentage :
  let total_plants : ℕ := 20 + 15
  let second_garden_tomatoes : ℕ := 15 / 3
  let total_tomatoes : ℕ := (total_plants * 20) / 100
  let first_garden_tomatoes : ℕ := total_tomatoes - second_garden_tomatoes
  (first_garden_tomatoes : ℚ) / 20 * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_garden_tomato_percentage_l2425_242523


namespace NUMINAMATH_CALUDE_banana_price_reduction_l2425_242599

/-- Proves that a price reduction resulting in 64 more bananas for Rs. 40.00001 
    and a new price of Rs. 3 per dozen represents a 40% reduction from the original price. -/
theorem banana_price_reduction (original_price : ℚ) : 
  (40.00001 / 3 - 40.00001 / original_price = 64 / 12) →
  (3 / original_price = 0.6) := by
  sorry

#eval (1 - 3/5) * 100 -- Should evaluate to 40

end NUMINAMATH_CALUDE_banana_price_reduction_l2425_242599


namespace NUMINAMATH_CALUDE_approx48000_accurate_to_thousand_l2425_242579

/-- Represents an approximate value with its numerical value and accuracy -/
structure ApproximateValue where
  value : ℕ
  accuracy : ℕ

/-- Checks if the given approximate value is accurate to thousand -/
def isAccurateToThousand (av : ApproximateValue) : Prop :=
  av.accuracy = 1000

/-- The approximate value 48,000 -/
def approx48000 : ApproximateValue :=
  { value := 48000, accuracy := 1000 }

/-- Theorem stating that 48,000 is accurate to thousand -/
theorem approx48000_accurate_to_thousand :
  isAccurateToThousand approx48000 := by
  sorry

end NUMINAMATH_CALUDE_approx48000_accurate_to_thousand_l2425_242579


namespace NUMINAMATH_CALUDE_integral_x4_over_2minusx2_32_l2425_242563

theorem integral_x4_over_2minusx2_32 :
  ∫ x in (0:ℝ)..1, x^4 / (2 - x^2)^(3/2) = 5/2 - 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_integral_x4_over_2minusx2_32_l2425_242563


namespace NUMINAMATH_CALUDE_smallest_n_for_Bn_radius_greater_than_two_l2425_242542

theorem smallest_n_for_Bn_radius_greater_than_two :
  (∃ n : ℕ+, (∀ k : ℕ+, k < n → Real.sqrt k - 1 ≤ 2) ∧ Real.sqrt n - 1 > 2) ∧
  (∀ n : ℕ+, (∀ k : ℕ+, k < n → Real.sqrt k - 1 ≤ 2) ∧ Real.sqrt n - 1 > 2 → n = 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_Bn_radius_greater_than_two_l2425_242542


namespace NUMINAMATH_CALUDE_distance_between_X_and_Y_l2425_242535

/-- The distance between points X and Y in miles -/
def D : ℝ := sorry

/-- Yolanda's walking rate in miles per hour -/
def yolanda_rate : ℝ := 5

/-- Bob's walking rate in miles per hour -/
def bob_rate : ℝ := 6

/-- The time in hours that Bob walks before meeting Yolanda -/
def bob_time : ℝ := sorry

/-- The distance Bob walks before meeting Yolanda in miles -/
def bob_distance : ℝ := 30

theorem distance_between_X_and_Y : D = 60 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_X_and_Y_l2425_242535


namespace NUMINAMATH_CALUDE_magician_earnings_l2425_242555

theorem magician_earnings 
  (price_per_deck : ℕ) 
  (initial_decks : ℕ) 
  (final_decks : ℕ) :
  price_per_deck = 2 →
  initial_decks = 5 →
  final_decks = 3 →
  (initial_decks - final_decks) * price_per_deck = 4 :=
by sorry

end NUMINAMATH_CALUDE_magician_earnings_l2425_242555


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l2425_242537

theorem unique_positive_integer_solution (m : ℤ) : 
  (∃! x : ℤ, x > 0 ∧ 6 * x^2 + 2 * (m - 13) * x + 12 - m = 0) ↔ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l2425_242537


namespace NUMINAMATH_CALUDE_age_difference_l2425_242559

/-- The age difference between A and C, given the condition about total ages -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 11) : A = C + 11 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2425_242559


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2425_242597

/-- A quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x + m

/-- The range of f is [0, +∞) -/
def has_range_zero_to_infinity (m : ℝ) : Prop :=
  ∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f m x = y

theorem quadratic_function_range (m : ℝ) :
  has_range_zero_to_infinity m → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l2425_242597


namespace NUMINAMATH_CALUDE_no_such_function_exists_l2425_242513

theorem no_such_function_exists : ¬∃ (f : ℤ → ℤ), ∀ (x y : ℤ), f (x + f y) = f x - y := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l2425_242513


namespace NUMINAMATH_CALUDE_existence_of_100_pairs_l2425_242527

def has_all_digits_at_least_6 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (n.digits 10) → d ≥ 6

theorem existence_of_100_pairs :
  ∃ S : Finset (ℕ × ℕ),
    S.card = 100 ∧
    (∀ (a b : ℕ), (a, b) ∈ S →
      has_all_digits_at_least_6 a ∧
      has_all_digits_at_least_6 b ∧
      has_all_digits_at_least_6 (a * b)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_100_pairs_l2425_242527


namespace NUMINAMATH_CALUDE_julia_age_after_ten_years_l2425_242572

/-- Given the ages and relationships of siblings, calculate Julia's age after 10 years -/
theorem julia_age_after_ten_years 
  (justin_age : ℕ)
  (jessica_age_when_justin_born : ℕ)
  (james_age_diff_jessica : ℕ)
  (julia_age_diff_justin : ℕ)
  (h1 : justin_age = 26)
  (h2 : jessica_age_when_justin_born = 6)
  (h3 : james_age_diff_jessica = 7)
  (h4 : julia_age_diff_justin = 8) :
  justin_age - julia_age_diff_justin + 10 = 28 :=
by sorry

end NUMINAMATH_CALUDE_julia_age_after_ten_years_l2425_242572


namespace NUMINAMATH_CALUDE_smallest_satisfying_polygon_l2425_242580

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def satisfies_conditions (n : ℕ) : Prop :=
  (number_of_diagonals n) * 4 = n * 7 ∧
  (number_of_diagonals n + n) % 2 = 0 ∧
  number_of_diagonals n + n > 50

theorem smallest_satisfying_polygon : 
  satisfies_conditions 12 ∧ 
  ∀ m : ℕ, m < 12 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_polygon_l2425_242580


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2425_242598

/-- Represents a hyperbola with equation x^2 - y^2/3 = 1 -/
def Hyperbola := {(x, y) : ℝ × ℝ | x^2 - y^2/3 = 1}

/-- The equation of asymptotes for the given hyperbola -/
def AsymptoteEquation (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- Theorem stating that the given equation represents the asymptotes of the hyperbola -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x, y) ∈ Hyperbola → (AsymptoteEquation x y ↔ (x, y) ∈ closure Hyperbola \ Hyperbola) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2425_242598


namespace NUMINAMATH_CALUDE_rectangle_y_value_l2425_242508

/-- A rectangle with vertices at (1, y), (9, y), (1, 5), and (9, 5), where y is positive and the area is 64 square units, has y = 13. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) (h2 : (9 - 1) * (y - 5) = 64) : y = 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l2425_242508


namespace NUMINAMATH_CALUDE_divisor_remainders_l2425_242543

theorem divisor_remainders (n : ℕ) 
  (h : ∀ i ∈ Finset.range 1012, ∃ (d : ℕ), d ∣ n ∧ d % 2013 = 1001 + i) :
  ∀ k ∈ Finset.range 2012, ∃ (d : ℕ), d ∣ n^2 ∧ d % 2013 = k + 1 := by
sorry

end NUMINAMATH_CALUDE_divisor_remainders_l2425_242543


namespace NUMINAMATH_CALUDE_fuel_cost_per_liter_l2425_242530

/-- The cost per liter of fuel given the tank capacity, initial fuel amount, and money spent. -/
theorem fuel_cost_per_liter
  (tank_capacity : ℝ)
  (initial_fuel : ℝ)
  (money_spent : ℝ)
  (h1 : tank_capacity = 150)
  (h2 : initial_fuel = 38)
  (h3 : money_spent = 336)
  : (money_spent / (tank_capacity - initial_fuel)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_fuel_cost_per_liter_l2425_242530


namespace NUMINAMATH_CALUDE_gnome_count_l2425_242521

/-- The number of garden gnomes with red hats, small noses, and striped shirts -/
def redHatSmallNoseStripedShirt (totalGnomes redHats bigNoses blueHatBigNoses : ℕ) : ℕ :=
  let blueHats := totalGnomes - redHats
  let smallNoses := totalGnomes - bigNoses
  let redHatSmallNoses := smallNoses - (blueHats - blueHatBigNoses)
  redHatSmallNoses / 2

/-- Theorem stating the number of garden gnomes with red hats, small noses, and striped shirts -/
theorem gnome_count : redHatSmallNoseStripedShirt 28 21 14 6 = 6 := by
  sorry

#eval redHatSmallNoseStripedShirt 28 21 14 6

end NUMINAMATH_CALUDE_gnome_count_l2425_242521


namespace NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l2425_242565

theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height := (2 : ℝ) / 3 * h
  let water_radius := (2 : ℝ) / 3 * r
  let cone_volume := (1 : ℝ) / 3 * π * r^2 * h
  let water_volume := (1 : ℝ) / 3 * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l2425_242565


namespace NUMINAMATH_CALUDE_toy_shopping_total_l2425_242576

def calculate_total_spent (prices : List Float) (discount_rate : Float) (tax_rate : Float) : Float :=
  let total_before_discount := prices.sum
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let sales_tax := tax_rate * total_after_discount
  total_after_discount + sales_tax

theorem toy_shopping_total (prices : List Float) 
  (h1 : prices = [8.25, 6.59, 12.10, 15.29, 23.47])
  (h2 : calculate_total_spent prices 0.10 0.06 = 62.68) : 
  calculate_total_spent prices 0.10 0.06 = 62.68 := by
  sorry

end NUMINAMATH_CALUDE_toy_shopping_total_l2425_242576


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_i_plus_two_l2425_242587

theorem imaginary_part_of_i_times_i_plus_two (i : ℂ) : 
  Complex.im (i * (i + 2)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_i_plus_two_l2425_242587


namespace NUMINAMATH_CALUDE_natural_fraction_condition_l2425_242557

theorem natural_fraction_condition (k n : ℕ) :
  (∃ m : ℕ, (7 * k + 15 * n - 1) = m * (3 * k + 4 * n)) ↔
  (∃ a : ℕ, k = 3 * a - 2 ∧ n = 2 * a - 1) :=
by sorry

end NUMINAMATH_CALUDE_natural_fraction_condition_l2425_242557


namespace NUMINAMATH_CALUDE_find_M_l2425_242548

theorem find_M : ∃ M : ℚ, (10 + 11 + 12) / 3 = (2022 + 2023 + 2024) / M ∧ M = 551 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l2425_242548


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2425_242546

theorem greatest_value_quadratic_inequality :
  ∀ a : ℝ, -a^2 + 9*a - 20 ≥ 0 → a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2425_242546


namespace NUMINAMATH_CALUDE_zero_function_solution_l2425_242544

def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x - f y

theorem zero_function_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_function_solution_l2425_242544


namespace NUMINAMATH_CALUDE_consecutive_integers_operation_l2425_242564

theorem consecutive_integers_operation (n : ℕ) (h1 : n = 9) : 
  let f : ℕ → ℕ → ℕ := λ x y => x + y + 162
  f n (n + 1) = n * (n + 1) + 91 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_operation_l2425_242564


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l2425_242545

theorem largest_angle_in_triangle (a b y : ℝ) : 
  a = 60 ∧ b = 70 ∧ a + b + y = 180 → 
  max a (max b y) = 70 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l2425_242545


namespace NUMINAMATH_CALUDE_smallest_n_complex_equality_l2425_242552

theorem smallest_n_complex_equality (a b : ℝ) (c : ℕ+) 
  (h_a : a > 0) (h_b : b > 0) 
  (h_smallest : ∀ k : ℕ+, k < 3 → (c * a + b * Complex.I) ^ k.val ≠ (c * a - b * Complex.I) ^ k.val) 
  (h_equal : (c * a + b * Complex.I) ^ 3 = (c * a - b * Complex.I) ^ 3) :
  b / (c * a) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_complex_equality_l2425_242552


namespace NUMINAMATH_CALUDE_triangular_prism_properties_l2425_242562

/-- Represents a triangular prism -/
structure TriangularPrism where
  AB : ℝ
  AC : ℝ
  AA₁ : ℝ
  angleCAB : ℝ

/-- The volume of a triangular prism -/
def volume (p : TriangularPrism) : ℝ := sorry

/-- The surface area of a triangular prism -/
def surfaceArea (p : TriangularPrism) : ℝ := sorry

theorem triangular_prism_properties (p : TriangularPrism)
    (h1 : p.AB = 1)
    (h2 : p.AC = 1)
    (h3 : p.AA₁ = Real.sqrt 2)
    (h4 : p.angleCAB = 2 * π / 3) : -- 120° in radians
  volume p = Real.sqrt 6 / 4 ∧
  surfaceArea p = 2 * Real.sqrt 2 + Real.sqrt 6 + Real.sqrt 3 / 2 := by
  sorry

#check triangular_prism_properties

end NUMINAMATH_CALUDE_triangular_prism_properties_l2425_242562


namespace NUMINAMATH_CALUDE_power_2_2013_mod_11_l2425_242592

theorem power_2_2013_mod_11 : 2^2013 % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_2_2013_mod_11_l2425_242592


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2425_242515

/-- The complex number i -/
def i : ℂ := Complex.I

/-- The given equation (1-2i)z = 5 -/
def given_equation (z : ℂ) : Prop := (1 - 2*i) * z = 5

/-- A complex number is in the first quadrant if its real and imaginary parts are both positive -/
def in_first_quadrant (z : ℂ) : Prop := 0 < z.re ∧ 0 < z.im

/-- 
If (1-2i)z = 5, then z is in the first quadrant of the complex plane
-/
theorem z_in_first_quadrant (z : ℂ) (h : given_equation z) : in_first_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2425_242515


namespace NUMINAMATH_CALUDE_shortest_side_right_triangle_l2425_242578

theorem shortest_side_right_triangle (a b c : ℝ) (ha : a = 9) (hb : b = 12) (hc : c^2 = a^2 + b^2) :
  min a (min b c) = 9 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_right_triangle_l2425_242578


namespace NUMINAMATH_CALUDE_cube_cut_forms_regular_hexagons_l2425_242503

-- Define a cube
structure Cube where
  side : ℝ
  side_positive : side > 0

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a regular hexagon
structure RegularHexagon where
  side : ℝ
  side_positive : side > 0

-- Function to get midpoints of cube edges
def getMidpoints (c : Cube) : List Point3D :=
  sorry

-- Function to define a plane through midpoints
def planeThroughMidpoints (midpoints : List Point3D) : Plane3D :=
  sorry

-- Function to determine if a plane intersects a cube to form regular hexagons
def intersectionFormsRegularHexagons (c : Cube) (p : Plane3D) : Prop :=
  sorry

-- Theorem statement
theorem cube_cut_forms_regular_hexagons (c : Cube) :
  let midpoints := getMidpoints c
  let cuttingPlane := planeThroughMidpoints midpoints
  intersectionFormsRegularHexagons c cuttingPlane :=
sorry

end NUMINAMATH_CALUDE_cube_cut_forms_regular_hexagons_l2425_242503


namespace NUMINAMATH_CALUDE_twenty_squares_in_four_by_five_grid_l2425_242532

/-- Represents a grid of points -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Counts the number of squares of a given size in a grid -/
def countSquares (g : Grid) (size : Nat) : Nat :=
  (g.rows - size + 1) * (g.cols - size + 1)

/-- The total number of squares in a grid -/
def totalSquares (g : Grid) : Nat :=
  countSquares g 1 + countSquares g 2 + countSquares g 3

/-- Theorem: In a 4x5 grid, the total number of squares is 20 -/
theorem twenty_squares_in_four_by_five_grid :
  totalSquares ⟨4, 5⟩ = 20 := by
  sorry

#eval totalSquares ⟨4, 5⟩

end NUMINAMATH_CALUDE_twenty_squares_in_four_by_five_grid_l2425_242532


namespace NUMINAMATH_CALUDE_largest_square_area_l2425_242571

/-- Given a configuration of 7 squares where the smallest square has area 9,
    the largest square has area 324. -/
theorem largest_square_area (num_squares : ℕ) (smallest_area : ℝ) : 
  num_squares = 7 → smallest_area = 9 → ∃ (largest_area : ℝ), largest_area = 324 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l2425_242571


namespace NUMINAMATH_CALUDE_triangle_height_problem_l2425_242505

theorem triangle_height_problem (base1 height1 base2 : ℝ) 
  (h_base1 : base1 = 15)
  (h_height1 : height1 = 12)
  (h_base2 : base2 = 20)
  (h_area_relation : base2 * (base1 * height1) = 2 * base1 * (base2 * height1)) :
  ∃ height2 : ℝ, height2 = 18 ∧ base2 * height2 = 2 * (base1 * height1) := by
sorry

end NUMINAMATH_CALUDE_triangle_height_problem_l2425_242505


namespace NUMINAMATH_CALUDE_sum_of_ages_l2425_242589

theorem sum_of_ages (a b c : ℕ) : 
  a = b + c + 16 → 
  a^2 = (b + c)^2 + 1632 → 
  a + b + c = 102 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2425_242589


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2425_242529

theorem inequality_system_solution (x : ℝ) :
  (5 * x - 1 > 3 * (x + 1) ∧ x - 1 ≤ 7 - x) → (2 < x ∧ x ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2425_242529


namespace NUMINAMATH_CALUDE_sugar_per_chocolate_bar_l2425_242558

/-- Given a company that produces chocolate bars, this theorem proves
    the amount of sugar needed per bar based on production rate and sugar usage. -/
theorem sugar_per_chocolate_bar
  (bars_per_minute : ℕ)
  (sugar_per_two_minutes : ℕ)
  (h1 : bars_per_minute = 36)
  (h2 : sugar_per_two_minutes = 108) :
  (sugar_per_two_minutes : ℚ) / ((bars_per_minute : ℚ) * 2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_per_chocolate_bar_l2425_242558


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_25_l2425_242551

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem smallest_prime_with_digit_sum_25 :
  ∃ p : ℕ, is_prime p ∧ digit_sum p = 25 ∧
  ∀ q : ℕ, is_prime q ∧ digit_sum q = 25 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_25_l2425_242551


namespace NUMINAMATH_CALUDE_puppy_cost_proof_l2425_242553

/-- Given a purchase of puppies with specific conditions, prove the cost of non-sale puppies. -/
theorem puppy_cost_proof (total_cost : ℕ) (sale_price : ℕ) (num_puppies : ℕ) (num_sale_puppies : ℕ) :
  total_cost = 800 →
  sale_price = 150 →
  num_puppies = 5 →
  num_sale_puppies = 3 →
  ∃ (non_sale_price : ℕ), 
    non_sale_price * (num_puppies - num_sale_puppies) + sale_price * num_sale_puppies = total_cost ∧
    non_sale_price = 175 := by
  sorry

end NUMINAMATH_CALUDE_puppy_cost_proof_l2425_242553


namespace NUMINAMATH_CALUDE_high_speed_rail_distance_scientific_notation_l2425_242583

theorem high_speed_rail_distance_scientific_notation :
  9280000000 = 9.28 * (10 ^ 9) := by
  sorry

end NUMINAMATH_CALUDE_high_speed_rail_distance_scientific_notation_l2425_242583


namespace NUMINAMATH_CALUDE_alissa_presents_l2425_242533

theorem alissa_presents (ethan_presents : ℝ) (difference : ℝ) (alissa_presents : ℝ) : 
  ethan_presents = 31.0 → 
  difference = 22.0 → 
  alissa_presents = ethan_presents - difference → 
  alissa_presents = 9.0 :=
by sorry

end NUMINAMATH_CALUDE_alissa_presents_l2425_242533


namespace NUMINAMATH_CALUDE_range_of_a_l2425_242526

theorem range_of_a (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 5 ∧ ∀ x ∈ s, (1 + a ≤ x ∧ x < 2)) → 
  -5 < a ∧ a ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2425_242526


namespace NUMINAMATH_CALUDE_students_just_passed_l2425_242570

theorem students_just_passed (total_students : ℕ) 
  (first_division_percentage : ℚ) (second_division_percentage : ℚ) : 
  total_students = 300 →
  first_division_percentage = 30 / 100 →
  second_division_percentage = 54 / 100 →
  (total_students : ℚ) * (1 - first_division_percentage - second_division_percentage) = 48 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l2425_242570


namespace NUMINAMATH_CALUDE_median_and_mean_of_set_l2425_242514

theorem median_and_mean_of_set (m : ℝ) (h : m + 4 = 16) :
  let S : Finset ℝ := {m, m + 2, m + 4, m + 11, m + 18}
  (S.sum id) / S.card = 19 := by
sorry

end NUMINAMATH_CALUDE_median_and_mean_of_set_l2425_242514


namespace NUMINAMATH_CALUDE_base_of_first_term_l2425_242575

/-- Given a positive integer h that is divisible by both 225 and 216,
    and can be expressed as h = x^a * 3^b * 5^c where x, a, b, and c are positive integers,
    and the least possible value of a + b + c is 8,
    prove that x must be 2. -/
theorem base_of_first_term (h : ℕ+) (x : ℕ+) (a b c : ℕ+) 
  (h_div_225 : 225 ∣ h)
  (h_div_216 : 216 ∣ h)
  (h_eq : h = x^(a:ℕ) * 3^(b:ℕ) * 5^(c:ℕ))
  (abc_min : a + b + c = 8 ∧ ∀ a' b' c' : ℕ+, a' + b' + c' ≥ 8) : x = 2 :=
sorry

end NUMINAMATH_CALUDE_base_of_first_term_l2425_242575


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l2425_242531

/-- Given a two-digit number with digit sum 6, if the product of this number and
    the number formed by swapping its digits is 1008, then the original number
    is either 42 or 24. -/
theorem two_digit_number_puzzle (n : ℕ) : 
  (n ≥ 10 ∧ n < 100) →  -- n is a two-digit number
  (n / 10 + n % 10 = 6) →  -- digit sum is 6
  (n * (10 * (n % 10) + (n / 10)) = 1008) →  -- product condition
  (n = 42 ∨ n = 24) := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l2425_242531


namespace NUMINAMATH_CALUDE_solve_grocery_problem_l2425_242577

def grocery_problem (total_brought chicken veggies eggs dog_food left_after meat : ℕ) : Prop :=
  total_brought = 167 ∧
  chicken = 22 ∧
  veggies = 43 ∧
  eggs = 5 ∧
  dog_food = 45 ∧
  left_after = 35 ∧
  meat = total_brought - (chicken + veggies + eggs + dog_food + left_after)

theorem solve_grocery_problem :
  ∃ meat, grocery_problem 167 22 43 5 45 35 meat ∧ meat = 17 := by sorry

end NUMINAMATH_CALUDE_solve_grocery_problem_l2425_242577


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_min_value_achievable_l2425_242528

theorem min_value_quadratic_form (x y z : ℝ) :
  3 * x^2 + 2*x*y + 3 * y^2 + 2*y*z + 3 * z^2 - 3*x + 3*y - 3*z + 9 ≥ (3/2 : ℝ) :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), 3 * x^2 + 2*x*y + 3 * y^2 + 2*y*z + 3 * z^2 - 3*x + 3*y - 3*z + 9 = (3/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_min_value_achievable_l2425_242528


namespace NUMINAMATH_CALUDE_product_xyz_equals_negative_two_l2425_242517

theorem product_xyz_equals_negative_two
  (x y z : ℝ)
  (h1 : x + 2 / y = 2)
  (h2 : y + 2 / z = 2) :
  x * y * z = -2 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_negative_two_l2425_242517


namespace NUMINAMATH_CALUDE_average_salary_l2425_242594

def salary_A : ℕ := 10000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def num_individuals : ℕ := 5

theorem average_salary :
  (salary_A + salary_B + salary_C + salary_D + salary_E) / num_individuals = 8600 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_l2425_242594


namespace NUMINAMATH_CALUDE_flag_count_l2425_242595

/-- The number of colors available for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flags : ℕ := num_colors ^ num_stripes

theorem flag_count : total_flags = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_count_l2425_242595


namespace NUMINAMATH_CALUDE_no_equilateral_grid_triangle_l2425_242584

/-- A point with integer coordinates -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three grid points -/
structure GridTriangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint

/-- Check if a triangle is equilateral -/
def isEquilateral (t : GridTriangle) : Prop :=
  let d1 := (t.a.x - t.b.x)^2 + (t.a.y - t.b.y)^2
  let d2 := (t.b.x - t.c.x)^2 + (t.b.y - t.c.y)^2
  let d3 := (t.c.x - t.a.x)^2 + (t.c.y - t.a.y)^2
  d1 = d2 ∧ d2 = d3

/-- The main theorem: no equilateral triangle exists on the integer grid -/
theorem no_equilateral_grid_triangle :
  ¬ ∃ t : GridTriangle, isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_no_equilateral_grid_triangle_l2425_242584


namespace NUMINAMATH_CALUDE_triangle_side_value_l2425_242536

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.b^2 - t.c^2 + 2*t.a = 0 ∧ Real.tan t.C / Real.tan t.B = 3

theorem triangle_side_value (t : Triangle) (h : TriangleConditions t) : t.a = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_value_l2425_242536


namespace NUMINAMATH_CALUDE_intersection_forms_line_l2425_242541

-- Define the equations
def hyperbola (x y : ℝ) : Prop := x * y = 12
def ellipse (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 36) = 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ hyperbola x y ∧ ellipse x y}

-- Theorem statement
theorem intersection_forms_line :
  ∃ (a b : ℝ), ∀ (p : ℝ × ℝ), p ∈ intersection_points → 
  (p.1 = a * p.2 + b ∨ p.2 = a * p.1 + b) :=
sorry

end NUMINAMATH_CALUDE_intersection_forms_line_l2425_242541


namespace NUMINAMATH_CALUDE_hawks_touchdowns_l2425_242511

theorem hawks_touchdowns (total_points : ℕ) (points_per_touchdown : ℕ) 
  (h1 : total_points = 21) 
  (h2 : points_per_touchdown = 7) : 
  total_points / points_per_touchdown = 3 := by
  sorry

end NUMINAMATH_CALUDE_hawks_touchdowns_l2425_242511


namespace NUMINAMATH_CALUDE_school_election_votes_l2425_242549

theorem school_election_votes (total_votes : ℕ) (brenda_votes : ℕ) : 
  brenda_votes = 50 → 
  4 * brenda_votes = total_votes →
  total_votes = 200 := by
sorry

end NUMINAMATH_CALUDE_school_election_votes_l2425_242549


namespace NUMINAMATH_CALUDE_initial_peanuts_count_l2425_242525

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Mary adds to the box -/
def peanuts_added : ℕ := 12

/-- The final number of peanuts in the box after Mary adds more -/
def final_peanuts : ℕ := 16

/-- Theorem stating that the initial number of peanuts is 4 -/
theorem initial_peanuts_count : initial_peanuts = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_peanuts_count_l2425_242525


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2425_242554

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2425_242554


namespace NUMINAMATH_CALUDE_total_folded_sheets_l2425_242574

theorem total_folded_sheets (initial_sheets : ℕ) (additional_sheets : ℕ) : 
  initial_sheets = 45 → additional_sheets = 18 → initial_sheets + additional_sheets = 63 := by
sorry

end NUMINAMATH_CALUDE_total_folded_sheets_l2425_242574


namespace NUMINAMATH_CALUDE_range_of_m_l2425_242590

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (m - 1 < x ∧ x < m + 1) → (x^2 - 2*x - 3 > 0)) ∧ 
  (∃ x : ℝ, (x^2 - 2*x - 3 > 0) ∧ ¬(m - 1 < x ∧ x < m + 1)) ↔ 
  (m ≤ -2 ∨ m ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2425_242590


namespace NUMINAMATH_CALUDE_optimal_strategy_is_down_l2425_242516

/-- Represents the direction of movement on the escalator -/
inductive Direction
  | Up
  | Down

/-- Represents the state of Petya and his hat on the escalators -/
structure EscalatorState where
  petyaPosition : ℝ  -- Position of Petya (0 = bottom, 1 = top)
  hatPosition : ℝ    -- Position of the hat (0 = bottom, 1 = top)
  petyaSpeed : ℝ     -- Petya's movement speed
  escalatorSpeed : ℝ  -- Speed of the escalator

/-- Calculates the time for Petya to reach his hat -/
def timeToReachHat (state : EscalatorState) (direction : Direction) : ℝ :=
  sorry

/-- Theorem stating that moving downwards is the optimal strategy -/
theorem optimal_strategy_is_down (state : EscalatorState) :
  state.petyaPosition = 0.5 →
  state.hatPosition = 1 →
  state.petyaSpeed > state.escalatorSpeed →
  state.petyaSpeed < 2 * state.escalatorSpeed →
  timeToReachHat state Direction.Down < timeToReachHat state Direction.Up :=
sorry

#check optimal_strategy_is_down

end NUMINAMATH_CALUDE_optimal_strategy_is_down_l2425_242516


namespace NUMINAMATH_CALUDE_expression_evaluation_l2425_242524

theorem expression_evaluation (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (2*x + y)^2 + (x + y)*(x - y) - x^2 = -4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2425_242524


namespace NUMINAMATH_CALUDE_sum_of_ages_l2425_242509

-- Define Rose's age
def rose_age : ℕ := 25

-- Define Rose's mother's age
def mother_age : ℕ := 75

-- Theorem: The sum of Rose's age and her mother's age is 100
theorem sum_of_ages : rose_age + mother_age = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2425_242509


namespace NUMINAMATH_CALUDE_jenny_weight_capacity_l2425_242518

/-- Represents the recycling problem Jenny faces --/
structure RecyclingProblem where
  bottle_weight : ℕ
  can_weight : ℕ
  num_cans : ℕ
  bottle_price : ℕ
  can_price : ℕ
  total_earnings : ℕ

/-- Calculates the total weight Jenny can carry --/
def total_weight (p : RecyclingProblem) : ℕ :=
  let num_bottles := (p.total_earnings - p.num_cans * p.can_price) / p.bottle_price
  num_bottles * p.bottle_weight + p.num_cans * p.can_weight

/-- Theorem stating that Jenny can carry 100 ounces --/
theorem jenny_weight_capacity :
  ∃ (p : RecyclingProblem),
    p.bottle_weight = 6 ∧
    p.can_weight = 2 ∧
    p.num_cans = 20 ∧
    p.bottle_price = 10 ∧
    p.can_price = 3 ∧
    p.total_earnings = 160 ∧
    total_weight p = 100 := by
  sorry


end NUMINAMATH_CALUDE_jenny_weight_capacity_l2425_242518


namespace NUMINAMATH_CALUDE_abc_relationship_l2425_242591

-- Define the constants a, b, and c
noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log 3

-- State the theorem
theorem abc_relationship : b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_abc_relationship_l2425_242591


namespace NUMINAMATH_CALUDE_cheese_division_theorem_l2425_242534

theorem cheese_division_theorem (a b c d e f : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_order : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) :
  ∃ (S₁ S₂ : Finset ℝ), 
    S₁.card = 3 ∧ 
    S₂.card = 3 ∧ 
    S₁ ∩ S₂ = ∅ ∧ 
    S₁ ∪ S₂ = {a, b, c, d, e, f} ∧
    (S₁.sum id = S₂.sum id) :=
sorry

end NUMINAMATH_CALUDE_cheese_division_theorem_l2425_242534


namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l2425_242593

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the vertical shift
def vertical_shift : ℝ := 5

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := original_parabola x + vertical_shift

-- Theorem stating that the shifted parabola is equivalent to y = x^2 + 5
theorem shifted_parabola_equation :
  ∀ x : ℝ, shifted_parabola x = x^2 + 5 := by sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l2425_242593


namespace NUMINAMATH_CALUDE_alarm_clock_probability_l2425_242596

theorem alarm_clock_probability (A B : ℝ) (hA : A = 0.80) (hB : B = 0.90) :
  1 - (1 - A) * (1 - B) = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_alarm_clock_probability_l2425_242596


namespace NUMINAMATH_CALUDE_brown_dogs_count_l2425_242581

/-- Proves the number of brown dogs in a kennel with specific conditions -/
theorem brown_dogs_count (total : ℕ) (long_fur : ℕ) (neither : ℕ) (long_fur_brown : ℕ)
  (h1 : total = 45)
  (h2 : long_fur = 26)
  (h3 : neither = 8)
  (h4 : long_fur_brown = 19) :
  total - long_fur + long_fur_brown = 30 := by
  sorry

#check brown_dogs_count

end NUMINAMATH_CALUDE_brown_dogs_count_l2425_242581


namespace NUMINAMATH_CALUDE_special_octagon_regions_l2425_242507

/-- Represents an octagon with specific properties -/
structure SpecialOctagon where
  angles : Fin 8 → ℝ
  sides : Fin 8 → ℝ
  all_angles_135 : ∀ i, angles i = 135
  alternating_sides : ∀ i, sides i = if i % 2 = 0 then 1 else Real.sqrt 2

/-- Counts the regions formed by drawing all sides and diagonals of the octagon -/
def count_regions (o : SpecialOctagon) : ℕ :=
  84

/-- Theorem stating that the special octagon is divided into 84 regions -/
theorem special_octagon_regions (o : SpecialOctagon) : 
  count_regions o = 84 := by sorry

end NUMINAMATH_CALUDE_special_octagon_regions_l2425_242507


namespace NUMINAMATH_CALUDE_tangent_roots_sum_identity_l2425_242520

theorem tangent_roots_sum_identity (p q : ℝ) (α β : ℝ) :
  (Real.tan α + Real.tan β = -p) →
  (Real.tan α * Real.tan β = q) →
  Real.sin (α + β)^2 + p * Real.sin (α + β) * Real.cos (α + β) + q * Real.cos (α + β)^2 = q := by
  sorry

end NUMINAMATH_CALUDE_tangent_roots_sum_identity_l2425_242520


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l2425_242586

theorem smallest_right_triangle_area :
  let side1 : ℝ := 6
  let side2 : ℝ := 8
  let area1 : ℝ := (1/2) * side1 * side2
  let area2 : ℝ := (1/2) * side1 * Real.sqrt (side2^2 - side1^2)
  min area1 area2 = 6 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l2425_242586


namespace NUMINAMATH_CALUDE_person_y_speed_l2425_242504

-- Define the river and docks
structure River :=
  (current_speed : ℝ)

structure Dock :=
  (position : ℝ)

-- Define the persons and their boats
structure Person :=
  (rowing_speed : ℝ)
  (starting_dock : Dock)

-- Define the scenario
def Scenario (river : River) (x y : Person) :=
  (x.rowing_speed = 6) ∧ 
  (x.starting_dock.position < y.starting_dock.position) ∧
  (∃ t : ℝ, t > 0 ∧ t * (x.rowing_speed - river.current_speed) = t * (y.rowing_speed + river.current_speed)) ∧
  (∃ t : ℝ, t > 0 ∧ t * (y.rowing_speed + river.current_speed) = t * (x.rowing_speed + river.current_speed) + 4 * (y.rowing_speed - x.rowing_speed)) ∧
  (4 * (x.rowing_speed - river.current_speed + y.rowing_speed + river.current_speed) = 16 * (y.rowing_speed - x.rowing_speed))

-- Theorem statement
theorem person_y_speed (river : River) (x y : Person) 
  (h : Scenario river x y) : y.rowing_speed = 10 :=
sorry

end NUMINAMATH_CALUDE_person_y_speed_l2425_242504


namespace NUMINAMATH_CALUDE_a_share_is_one_third_l2425_242567

/-- Represents the investment and profit distribution scenario -/
structure InvestmentScenario where
  initial_investment : ℝ
  annual_gain : ℝ
  months_in_year : ℕ

/-- Calculates the effective investment value for a partner -/
def effective_investment (scenario : InvestmentScenario) 
  (investment_multiplier : ℝ) (investment_duration : ℕ) : ℝ :=
  scenario.initial_investment * investment_multiplier * investment_duration

/-- Theorem stating that A's share of the gain is one-third of the total gain -/
theorem a_share_is_one_third (scenario : InvestmentScenario) 
  (h1 : scenario.months_in_year = 12)
  (h2 : scenario.annual_gain > 0) : 
  let a_investment := effective_investment scenario 1 scenario.months_in_year
  let b_investment := effective_investment scenario 2 6
  let c_investment := effective_investment scenario 3 4
  let total_effective_investment := a_investment + b_investment + c_investment
  scenario.annual_gain / 3 = (a_investment / total_effective_investment) * scenario.annual_gain := by
  sorry

#check a_share_is_one_third

end NUMINAMATH_CALUDE_a_share_is_one_third_l2425_242567


namespace NUMINAMATH_CALUDE_caravan_keepers_count_l2425_242540

/-- Represents the number of feet for different animals and humans -/
def feet_count : Nat → Nat
| 0 => 2  -- humans and hens
| 1 => 4  -- goats and camels
| _ => 0

/-- The caravan problem -/
theorem caravan_keepers_count 
  (hens goats camels : Nat) 
  (hens_count : hens = 60)
  (goats_count : goats = 35)
  (camels_count : camels = 6)
  (feet_head_diff : 
    ∃ keepers : Nat, 
      hens * feet_count 0 + 
      goats * feet_count 1 + 
      camels * feet_count 1 + 
      keepers * feet_count 0 = 
      hens + goats + camels + keepers + 193) :
  ∃ keepers : Nat, keepers = 10 := by
sorry

end NUMINAMATH_CALUDE_caravan_keepers_count_l2425_242540


namespace NUMINAMATH_CALUDE_triangle_expression_simplification_l2425_242560

theorem triangle_expression_simplification
  (a b c : ℝ)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  |a + c - b| - |a + b + c| + |2*b + c| = c :=
sorry

end NUMINAMATH_CALUDE_triangle_expression_simplification_l2425_242560


namespace NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l2425_242522

/-- Given a geometric sequence where the first term is 4 and the second term is 16/3,
    the 10th term of this sequence is 1048576/19683. -/
theorem tenth_term_of_geometric_sequence :
  let a₁ : ℚ := 4
  let a₂ : ℚ := 16/3
  let r : ℚ := a₂ / a₁
  let a₁₀ : ℚ := a₁ * r^9
  a₁₀ = 1048576/19683 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l2425_242522


namespace NUMINAMATH_CALUDE_tangent_perpendicular_theorem_l2425_242538

noncomputable def f (x : ℝ) : ℝ := x^4

def perpendicular_line (x y : ℝ) : Prop := x + 4*y - 8 = 0

def tangent_line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

theorem tangent_perpendicular_theorem :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = f x₀ ∧ 
    (∃ (a b c : ℝ), tangent_line a b c x₀ y₀ ∧ 
      (∀ (x y : ℝ), perpendicular_line x y → 
        (a*1 + b*4 = 0))) → 
    (∃ (x y : ℝ), tangent_line 4 (-1) (-3) x y) :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_theorem_l2425_242538


namespace NUMINAMATH_CALUDE_tangent_line_and_max_value_l2425_242568

-- Define the function f(x) = x³ - ax²
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem tangent_line_and_max_value (a : ℝ) :
  f' a 1 = 3 →
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x + x ≥ 0) →
  (∃ (m b : ℝ), m = 3 ∧ b = -2 ∧
    ∀ x : ℝ, f a x = m * (x - 1) + f a 1) ∧
  (∃ M : ℝ, M = 8 ∧
    ∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x ≤ M) ∧
  a ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_tangent_line_and_max_value_l2425_242568


namespace NUMINAMATH_CALUDE_min_value_theorem_l2425_242550

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : 2 = Real.sqrt (4^a * 2^b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 = Real.sqrt (4^x * 2^y) → 
    (2/a + 1/b) ≤ (2/x + 1/y) ∧ 
    (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2 = Real.sqrt (4^a₀ * 2^b₀) ∧ 2/a₀ + 1/b₀ = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2425_242550


namespace NUMINAMATH_CALUDE_store_profit_maximization_l2425_242573

/-- Represents the store's profit function --/
def profit_function (x : ℕ) : ℚ := -10 * x^2 + 800 * x + 20000

/-- Represents the constraint on the price increase --/
def valid_price_increase (x : ℕ) : Prop := x ≤ 100

theorem store_profit_maximization :
  ∃ (x : ℕ), valid_price_increase x ∧
    (∀ (y : ℕ), valid_price_increase y → profit_function y ≤ profit_function x) ∧
    x = 40 ∧ profit_function x = 36000 := by
  sorry

#check store_profit_maximization

end NUMINAMATH_CALUDE_store_profit_maximization_l2425_242573


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2425_242500

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches -/
theorem cylinder_surface_area : 
  ∀ (h r : ℝ), 
  h = 8 → 
  r = 3 → 
  2 * π * r * h + 2 * π * r^2 = 66 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2425_242500


namespace NUMINAMATH_CALUDE_copresidents_count_l2425_242566

/-- Represents a club with members distributed across departments. -/
structure Club where
  total_members : ℕ
  num_departments : ℕ
  members_per_department : ℕ
  h_total : total_members = num_departments * members_per_department

/-- The number of ways to choose co-presidents from different departments. -/
def choose_copresidents (c : Club) : ℕ :=
  (c.num_departments * c.members_per_department * (c.num_departments - 1) * c.members_per_department) / 2

/-- Theorem stating the number of ways to choose co-presidents for the given club configuration. -/
theorem copresidents_count (c : Club) 
  (h_total : c.total_members = 24)
  (h_departments : c.num_departments = 4)
  (h_distribution : c.members_per_department = 6) : 
  choose_copresidents c = 54 := by
  sorry

#eval choose_copresidents ⟨24, 4, 6, rfl⟩

end NUMINAMATH_CALUDE_copresidents_count_l2425_242566


namespace NUMINAMATH_CALUDE_store_inventory_difference_l2425_242510

theorem store_inventory_difference : 
  ∀ (apples regular_soda diet_soda : ℕ),
    apples = 36 →
    regular_soda = 80 →
    diet_soda = 54 →
    regular_soda + diet_soda - apples = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_store_inventory_difference_l2425_242510


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l2425_242519

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    is S_n = 3^(n-2) + m, prove that m = -1/9 -/
theorem geometric_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℝ) :
  (∀ n, S n = 3^(n-2) + m) →
  (∀ n, a (n+1) / a n = a (n+2) / a (n+1)) →
  (a 1 = S 1) →
  (∀ n, a (n+1) = S (n+1) - S n) →
  m = -1/9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l2425_242519


namespace NUMINAMATH_CALUDE_fraction_equality_l2425_242506

theorem fraction_equality (p q : ℝ) (h : p / q - q / p = 21 / 10) :
  4 * p / q + 4 * q / p = 16.8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2425_242506


namespace NUMINAMATH_CALUDE_simplify_radical_fraction_l2425_242502

theorem simplify_radical_fraction (x : ℝ) (h1 : x < 0) :
  ((-x^3).sqrt / x) = -(-x).sqrt := by sorry

end NUMINAMATH_CALUDE_simplify_radical_fraction_l2425_242502


namespace NUMINAMATH_CALUDE_influenza_spread_l2425_242547

theorem influenza_spread (x : ℝ) : (1 + x)^2 = 100 → x = 9 := by sorry

end NUMINAMATH_CALUDE_influenza_spread_l2425_242547
