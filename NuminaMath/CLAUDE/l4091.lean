import Mathlib

namespace NUMINAMATH_CALUDE_function_inequality_l4091_409126

/-- Given a function f : ℝ → ℝ with derivative f', prove that if f'(x) < f(x) for all x,
    then f(2) < e^2 * f(0) and f(2012) < e^2012 * f(0) -/
theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (hf' : ∀ x, deriv f x = f' x) (h : ∀ x, f' x < f x) : 
    f 2 < Real.exp 2 * f 0 ∧ f 2012 < Real.exp 2012 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l4091_409126


namespace NUMINAMATH_CALUDE_sin_sixteen_thirds_pi_l4091_409111

theorem sin_sixteen_thirds_pi : Real.sin (16 * Real.pi / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixteen_thirds_pi_l4091_409111


namespace NUMINAMATH_CALUDE_angle_function_value_l4091_409163

/-- Given m < 0 and point M(3m, -m) on the terminal side of angle α, 
    prove that 1 / (2sin(α)cos(α) + cos²(α)) = 10/3 -/
theorem angle_function_value (m : ℝ) (α : ℝ) :
  m < 0 →
  let M : ℝ × ℝ := (3 * m, -m)
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_function_value_l4091_409163


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4091_409184

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 2 * a 6 + 2 * a 4 * a 5 + a 1 * a 9 = 25 →
  a 4 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4091_409184


namespace NUMINAMATH_CALUDE_restaurant_production_difference_l4091_409127

/-- Represents the daily production of a restaurant -/
structure DailyProduction where
  pizzas : ℕ
  hotDogs : ℕ
  pizzasMoreThanHotDogs : pizzas > hotDogs

/-- Represents the monthly production of a restaurant -/
def MonthlyProduction (d : DailyProduction) (days : ℕ) : ℕ :=
  days * (d.pizzas + d.hotDogs)

/-- Theorem: The restaurant makes 40 more pizzas than hot dogs every day -/
theorem restaurant_production_difference (d : DailyProduction) 
    (h1 : d.hotDogs = 60)
    (h2 : MonthlyProduction d 30 = 4800) :
  d.pizzas - d.hotDogs = 40 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_production_difference_l4091_409127


namespace NUMINAMATH_CALUDE_light_distance_theorem_l4091_409166

/-- The distance light travels in one year in miles -/
def light_year_miles : ℝ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℝ := 500

/-- The conversion factor from miles to kilometers -/
def miles_to_km : ℝ := 1.60934

/-- The distance light travels in the given number of years in kilometers -/
def light_distance_km : ℝ := light_year_miles * years * miles_to_km

theorem light_distance_theorem : 
  light_distance_km = 4.723e15 := by sorry

end NUMINAMATH_CALUDE_light_distance_theorem_l4091_409166


namespace NUMINAMATH_CALUDE_factors_multiple_of_180_l4091_409197

/-- The number of natural-number factors of m that are multiples of 180 -/
def count_factors (m : ℕ) : ℕ :=
  sorry

theorem factors_multiple_of_180 :
  let m : ℕ := 2^12 * 3^15 * 5^9
  count_factors m = 1386 := by
  sorry

end NUMINAMATH_CALUDE_factors_multiple_of_180_l4091_409197


namespace NUMINAMATH_CALUDE_total_cost_of_items_l4091_409102

theorem total_cost_of_items (wallet_cost : ℕ) 
  (h1 : wallet_cost = 22)
  (purse_cost : ℕ) 
  (h2 : purse_cost = 4 * wallet_cost - 3)
  (shoes_cost : ℕ) 
  (h3 : shoes_cost = wallet_cost + purse_cost + 7) :
  wallet_cost + purse_cost + shoes_cost = 221 := by
sorry

end NUMINAMATH_CALUDE_total_cost_of_items_l4091_409102


namespace NUMINAMATH_CALUDE_area_between_circles_l4091_409199

/-- The area between two concentric circles, where the larger circle's radius is three times 
    the smaller circle's radius, and the smaller circle's diameter is 6 units, 
    is equal to 72π square units. -/
theorem area_between_circles (π : ℝ) : 
  let small_diameter : ℝ := 6
  let small_radius : ℝ := small_diameter / 2
  let large_radius : ℝ := 3 * small_radius
  let area_large : ℝ := π * large_radius ^ 2
  let area_small : ℝ := π * small_radius ^ 2
  area_large - area_small = 72 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_circles_l4091_409199


namespace NUMINAMATH_CALUDE_tangent_line_to_sqrt_curve_l4091_409116

theorem tangent_line_to_sqrt_curve (x y : ℝ) :
  (∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧
    (a * 1 + b * 2 + c = 0) ∧
    (∃ (x₀ : ℝ), x₀ > 0 ∧ 
      a * x₀ + b * Real.sqrt x₀ + c = 0 ∧
      a + b * (1 / (2 * Real.sqrt x₀)) = 0)) ↔
  ((x - (4 + 2 * Real.sqrt 3) * y + (7 + 4 * Real.sqrt 3) = 0) ∨
   (x - (4 - 2 * Real.sqrt 3) * y + (7 - 4 * Real.sqrt 3) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_sqrt_curve_l4091_409116


namespace NUMINAMATH_CALUDE_fixed_point_exists_l4091_409180

/-- For any a > 0 and a ≠ 1, the function f(x) = ax - 5 has a fixed point at x = 2 -/
theorem fixed_point_exists (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x : ℝ, a * x - 5 = x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exists_l4091_409180


namespace NUMINAMATH_CALUDE_expression_simplification_l4091_409124

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 3 * x + y / 3 ≠ 0) :
  (3 * x + y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹) = (3 * x * y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4091_409124


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l4091_409141

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem tenth_term_of_sequence
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 2)
  (h3 : a 1 = 1) :
  a 10 = 19 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l4091_409141


namespace NUMINAMATH_CALUDE_trajectory_is_straight_line_l4091_409185

/-- The trajectory of a point P(x, y) equidistant from M(-2, 0) and the line x = -2 is a straight line y = 0 -/
theorem trajectory_is_straight_line :
  ∀ (x y : ℝ), 
    (|x + 2| = Real.sqrt ((x + 2)^2 + y^2)) → 
    y = 0 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_straight_line_l4091_409185


namespace NUMINAMATH_CALUDE_sum_of_sixth_root_arguments_l4091_409142

open Complex

/-- The complex number whose sixth power is equal to -1/√3 - i√(2/3) -/
def z : ℂ := sorry

/-- The argument of z^6 in radians -/
def arg_z6 : ℝ := sorry

/-- The list of arguments of the sixth roots of z^6 in radians -/
def root_args : List ℝ := sorry

theorem sum_of_sixth_root_arguments :
  (root_args.sum * (180 / Real.pi)) = 1140 ∧ 
  (∀ φ ∈ root_args, 0 ≤ φ * (180 / Real.pi) ∧ φ * (180 / Real.pi) < 360) ∧
  (List.length root_args = 6) ∧
  (∀ φ ∈ root_args, Complex.exp (φ * Complex.I) ^ 6 = z^6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sixth_root_arguments_l4091_409142


namespace NUMINAMATH_CALUDE_factorization_of_x2y_minus_4y_l4091_409131

theorem factorization_of_x2y_minus_4y (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x2y_minus_4y_l4091_409131


namespace NUMINAMATH_CALUDE_hair_cut_second_day_l4091_409106

/-- The amount of hair cut off on the second day, given the total amount cut off and the amount cut off on the first day. -/
theorem hair_cut_second_day 
  (total_cut : ℝ) 
  (first_day_cut : ℝ) 
  (h1 : total_cut = 0.875) 
  (h2 : first_day_cut = 0.375) : 
  total_cut - first_day_cut = 0.500 := by
sorry

end NUMINAMATH_CALUDE_hair_cut_second_day_l4091_409106


namespace NUMINAMATH_CALUDE_triangle_formation_l4091_409103

theorem triangle_formation (a b c : ℝ) : 
  a = 4 ∧ b = 9 ∧ c = 9 →
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l4091_409103


namespace NUMINAMATH_CALUDE_least_N_for_P_condition_l4091_409164

/-- The probability that at least 3/5 of N green balls are on the same side of a red ball
    in a random arrangement of N green balls and one red ball. -/
def P (N : ℕ) : ℚ :=
  (↑⌈(3 * N : ℚ) / 5 + 1⌉) / (N + 1)

/-- 480 is the least positive multiple of 5 for which P(N) < 321/400 -/
theorem least_N_for_P_condition : ∀ N : ℕ,
  N > 0 ∧ N % 5 = 0 ∧ P N < 321 / 400 → N ≥ 480 :=
sorry

end NUMINAMATH_CALUDE_least_N_for_P_condition_l4091_409164


namespace NUMINAMATH_CALUDE_minimum_at_two_implies_m_geq_five_range_of_m_l4091_409152

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - 1| + m * |x - 2| + 6 * |x - 3|

/-- The theorem stating that if f attains its minimum at x = 2, then m ≥ 5 -/
theorem minimum_at_two_implies_m_geq_five (m : ℝ) :
  (∀ x : ℝ, f m x ≥ f m 2) → m ≥ 5 := by
  sorry

/-- The main theorem describing the range of m -/
theorem range_of_m :
  {m : ℝ | ∀ x : ℝ, f m x ≥ f m 2} = {m : ℝ | m ≥ 5} := by
  sorry

end NUMINAMATH_CALUDE_minimum_at_two_implies_m_geq_five_range_of_m_l4091_409152


namespace NUMINAMATH_CALUDE_symmetric_complex_division_l4091_409170

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_to_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- Given two complex numbers z₁ and z₂ symmetric with respect to the imaginary axis,
    where z₁ = -1 + i, prove that z₁ / z₂ = i. -/
theorem symmetric_complex_division (z₁ z₂ : ℂ) 
    (h_sym : symmetric_to_imaginary_axis z₁ z₂) 
    (h_z₁ : z₁ = -1 + Complex.I) : 
  z₁ / z₂ = Complex.I := by
  sorry


end NUMINAMATH_CALUDE_symmetric_complex_division_l4091_409170


namespace NUMINAMATH_CALUDE_sequence_sum_property_l4091_409117

theorem sequence_sum_property (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) (k : ℕ+) :
  (∀ n : ℕ+, S n = a n / n) →
  (1 < S k ∧ S k < 9) →
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l4091_409117


namespace NUMINAMATH_CALUDE_johns_trip_distance_l4091_409154

theorem johns_trip_distance : ∃ (total_distance : ℝ), 
  (total_distance / 2) + 40 + (total_distance / 4) = total_distance ∧ 
  total_distance = 160 := by
sorry

end NUMINAMATH_CALUDE_johns_trip_distance_l4091_409154


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l4091_409195

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + (a-1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x : ℝ | x^2 - m*x + 2 = 0}

theorem set_equality_implies_values (a m : ℝ) 
  (h1 : A ∪ B a = A) 
  (h2 : A ∩ C m = C m) : 
  (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l4091_409195


namespace NUMINAMATH_CALUDE_triangle_side_count_l4091_409172

theorem triangle_side_count (a b : ℕ) (ha : a = 8) (hb : b = 5) :
  ∃! n : ℕ, n = (Finset.range (a + b - 1) \ Finset.range (a - b + 1)).card :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_count_l4091_409172


namespace NUMINAMATH_CALUDE_distance_between_3rd_and_21st_red_lights_l4091_409191

/-- Represents the pattern of lights on the string -/
inductive LightColor
| Red
| Green

/-- Defines the repeating pattern of lights -/
def lightPattern : List LightColor :=
  [LightColor.Red, LightColor.Red, LightColor.Green, LightColor.Green, LightColor.Green]

/-- The spacing between lights in inches -/
def lightSpacing : ℕ := 6

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- Function to get the position of the nth red light -/
def nthRedLightPosition (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the distance between the 3rd and 21st red lights -/
theorem distance_between_3rd_and_21st_red_lights :
  (nthRedLightPosition 21 - nthRedLightPosition 3) * lightSpacing / inchesPerFoot = 22 :=
sorry

end NUMINAMATH_CALUDE_distance_between_3rd_and_21st_red_lights_l4091_409191


namespace NUMINAMATH_CALUDE_median_trigonometric_values_max_condition_implies_range_l4091_409187

def median (a b c : ℝ) : ℝ := sorry

def max3 (a b c : ℝ) : ℝ := sorry

theorem median_trigonometric_values :
  median (Real.sin (30 * π / 180)) (Real.cos (45 * π / 180)) (Real.tan (60 * π / 180)) = Real.sqrt 2 / 2 := by sorry

theorem max_condition_implies_range (x : ℝ) :
  max3 5 (2*x - 3) (-10 - 3*x) = 5 → -5 ≤ x ∧ x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_median_trigonometric_values_max_condition_implies_range_l4091_409187


namespace NUMINAMATH_CALUDE_initially_calculated_average_weight_l4091_409178

/-- Given a class of boys with a misread weight, prove the initially calculated average weight. -/
theorem initially_calculated_average_weight
  (n : ℕ) -- number of boys
  (correct_avg : ℝ) -- correct average weight
  (misread_weight : ℝ) -- misread weight
  (correct_weight : ℝ) -- correct weight
  (h1 : n = 20)
  (h2 : correct_avg = 58.7)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 62)
  : ∃ (initial_avg : ℝ), initial_avg = 58.4 := by
  sorry

end NUMINAMATH_CALUDE_initially_calculated_average_weight_l4091_409178


namespace NUMINAMATH_CALUDE_weavers_count_proof_l4091_409130

/-- The number of weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of weavers in the second group -/
def second_group_weavers : ℕ := 6

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 9

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 6

theorem weavers_count_proof :
  (first_group_mats : ℚ) / (first_group_weavers * first_group_days) =
  (second_group_mats : ℚ) / (second_group_weavers * second_group_days) →
  first_group_weavers = 4 := by
  sorry

end NUMINAMATH_CALUDE_weavers_count_proof_l4091_409130


namespace NUMINAMATH_CALUDE_smartpup_academy_total_dogs_l4091_409181

/-- Represents the number of dogs at Smartpup Tricks Academy with various skill combinations -/
structure DogSkills where
  fetch : ℕ
  jump : ℕ
  play_dead : ℕ
  fetch_and_jump : ℕ
  jump_and_play_dead : ℕ
  fetch_and_play_dead : ℕ
  all_three : ℕ
  none : ℕ

/-- Calculates the total number of dogs at the academy -/
def total_dogs (skills : DogSkills) : ℕ :=
  skills.all_three +
  (skills.fetch_and_play_dead - skills.all_three) +
  (skills.jump_and_play_dead - skills.all_three) +
  (skills.fetch_and_jump - skills.all_three) +
  (skills.fetch - skills.fetch_and_jump - skills.fetch_and_play_dead + skills.all_three) +
  (skills.jump - skills.fetch_and_jump - skills.jump_and_play_dead + skills.all_three) +
  (skills.play_dead - skills.fetch_and_play_dead - skills.jump_and_play_dead + skills.all_three) +
  skills.none

/-- The main theorem stating that the total number of dogs is 75 -/
theorem smartpup_academy_total_dogs :
  let skills : DogSkills := {
    fetch := 40,
    jump := 35,
    play_dead := 22,
    fetch_and_jump := 14,
    jump_and_play_dead := 10,
    fetch_and_play_dead := 16,
    all_three := 6,
    none := 12
  }
  total_dogs skills = 75 := by
  sorry

end NUMINAMATH_CALUDE_smartpup_academy_total_dogs_l4091_409181


namespace NUMINAMATH_CALUDE_car_trade_profit_percentage_l4091_409149

/-- Calculates the profit percentage on the original price when a car is bought at a discount and sold at an increase. -/
theorem car_trade_profit_percentage 
  (original_price : ℝ) 
  (discount_percentage : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : discount_percentage = 20) 
  (h2 : increase_percentage = 50) 
  : (((1 - discount_percentage / 100) * (1 + increase_percentage / 100) - 1) * 100 = 20) := by
sorry

end NUMINAMATH_CALUDE_car_trade_profit_percentage_l4091_409149


namespace NUMINAMATH_CALUDE_cosine_sum_pentagon_l4091_409135

theorem cosine_sum_pentagon : 
  Real.cos (5 * π / 180) + Real.cos (77 * π / 180) + Real.cos (149 * π / 180) + 
  Real.cos (221 * π / 180) + Real.cos (293 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_pentagon_l4091_409135


namespace NUMINAMATH_CALUDE_subcommittee_formation_ways_senate_subcommittee_formation_l4091_409179

theorem subcommittee_formation_ways (total_republicans : Nat) (total_democrats : Nat) 
  (subcommittee_republicans : Nat) (subcommittee_democrats : Nat) : Nat :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem senate_subcommittee_formation : 
  subcommittee_formation_ways 10 8 4 3 = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_ways_senate_subcommittee_formation_l4091_409179


namespace NUMINAMATH_CALUDE_travis_apples_count_l4091_409119

/-- The number of apples that fit in each box -/
def apples_per_box : ℕ := 50

/-- The price of each box of apples in dollars -/
def price_per_box : ℕ := 35

/-- The total amount Travis takes home in dollars -/
def total_revenue : ℕ := 7000

/-- The number of apples Travis has -/
def travis_apples : ℕ := total_revenue / price_per_box * apples_per_box

theorem travis_apples_count : travis_apples = 10000 := by
  sorry

end NUMINAMATH_CALUDE_travis_apples_count_l4091_409119


namespace NUMINAMATH_CALUDE_line_through_origin_and_third_quadrant_l4091_409133

/-- A line in 2D space represented by the equation Ax - By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a point (x, y) lies on a given line -/
def Line.contains (L : Line) (x y : ℝ) : Prop :=
  L.A * x - L.B * y + L.C = 0

/-- Predicate to check if a line passes through the origin -/
def Line.passes_through_origin (L : Line) : Prop :=
  L.contains 0 0

/-- Predicate to check if a line passes through the third quadrant -/
def Line.passes_through_third_quadrant (L : Line) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ L.contains x y

/-- Theorem stating the properties of a line passing through the origin and third quadrant -/
theorem line_through_origin_and_third_quadrant (L : Line) :
  L.passes_through_origin ∧ L.passes_through_third_quadrant →
  L.A * L.B < 0 ∧ L.C = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_through_origin_and_third_quadrant_l4091_409133


namespace NUMINAMATH_CALUDE_work_ratio_l4091_409162

/-- The time (in days) it takes for worker A to complete the task alone -/
def time_A : ℝ := 6

/-- The time (in days) it takes for worker B to complete the task alone -/
def time_B : ℝ := 30

/-- The time (in days) it takes for workers A and B to complete the task together -/
def time_together : ℝ := 5

theorem work_ratio : 
  (1 / time_A + 1 / time_B = 1 / time_together) → 
  (time_A / time_B = 1 / 5) := by
  sorry

end NUMINAMATH_CALUDE_work_ratio_l4091_409162


namespace NUMINAMATH_CALUDE_box_neg_two_two_neg_one_l4091_409192

def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) - (b ^ c : ℚ) + (c ^ a : ℚ)

theorem box_neg_two_two_neg_one : box (-2) 2 (-1) = (7 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_box_neg_two_two_neg_one_l4091_409192


namespace NUMINAMATH_CALUDE_frequency_count_calculation_l4091_409182

/-- Given a sample of size 1000 divided into several groups,
    if the frequency of a particular group is 0.4,
    then the frequency count of that group is 400. -/
theorem frequency_count_calculation (sample_size : ℕ) (group_frequency : ℝ) :
  sample_size = 1000 →
  group_frequency = 0.4 →
  (sample_size : ℝ) * group_frequency = 400 := by
  sorry

end NUMINAMATH_CALUDE_frequency_count_calculation_l4091_409182


namespace NUMINAMATH_CALUDE_third_turtle_lying_l4091_409136

-- Define the type for turtles
inductive Turtle : Type
  | T1 : Turtle
  | T2 : Turtle
  | T3 : Turtle

-- Define the relative position of turtles
inductive Position : Type
  | Front : Position
  | Behind : Position

-- Define a function to represent the statement of each turtle
def turtleStatement (t : Turtle) : List (Turtle × Position) :=
  match t with
  | Turtle.T1 => [(Turtle.T2, Position.Behind), (Turtle.T3, Position.Behind)]
  | Turtle.T2 => [(Turtle.T1, Position.Front), (Turtle.T3, Position.Behind)]
  | Turtle.T3 => [(Turtle.T1, Position.Front), (Turtle.T2, Position.Front), (Turtle.T3, Position.Behind)]

-- Define a function to check if a turtle's statement is consistent with its position
def isConsistent (t : Turtle) (position : Nat) : Prop :=
  match t, position with
  | Turtle.T1, 0 => true
  | Turtle.T2, 1 => true
  | Turtle.T3, 2 => false
  | _, _ => false

-- Theorem: The third turtle's statement is inconsistent
theorem third_turtle_lying :
  ¬ (isConsistent Turtle.T3 2) :=
  sorry


end NUMINAMATH_CALUDE_third_turtle_lying_l4091_409136


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l4091_409173

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem: There are 10 ways to distribute 6 indistinguishable balls into 3 distinguishable boxes,
    with each box containing at least one ball. -/
theorem six_balls_three_boxes :
  distribute_balls 6 3 = 10 := by
  sorry

#eval distribute_balls 6 3

end NUMINAMATH_CALUDE_six_balls_three_boxes_l4091_409173


namespace NUMINAMATH_CALUDE_sequence_is_geometric_from_second_term_l4091_409169

def is_geometric_from_second_term (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 2 → a (n + 1) = r * a n

theorem sequence_is_geometric_from_second_term
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : S 1 = 1)
  (h2 : S 2 = 2)
  (h3 : ∀ n : ℕ, n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0)
  (h4 : ∀ n : ℕ, S (n + 1) - S n = a (n + 1))
  : is_geometric_from_second_term a :=
by
  sorry

#check sequence_is_geometric_from_second_term

end NUMINAMATH_CALUDE_sequence_is_geometric_from_second_term_l4091_409169


namespace NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l4091_409174

theorem and_sufficient_not_necessary_for_or (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ∃ (r s : Prop), (r ∨ s) ∧ ¬(r ∧ s) := by
  sorry

end NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l4091_409174


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l4091_409171

theorem max_sum_of_factors (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 144 →
  a + b + c ≤ 75 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l4091_409171


namespace NUMINAMATH_CALUDE_remaining_amount_l4091_409129

def initial_amount : ℝ := 100.00
def spent_amount : ℝ := 15.00

theorem remaining_amount :
  initial_amount - spent_amount = 85.00 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_l4091_409129


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l4091_409186

theorem polynomial_division_degree (f d q r : Polynomial ℝ) :
  (Polynomial.degree f = 15) →
  (f = d * q + r) →
  (Polynomial.degree q = 8) →
  (r = 5 * X^4 + 3 * X^2 - 2 * X + 7) →
  (Polynomial.degree d = 7) := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l4091_409186


namespace NUMINAMATH_CALUDE_football_game_attendance_l4091_409100

/-- Proves that the number of children attending a football game is 80, given the ticket prices, total attendance, and total money collected. -/
theorem football_game_attendance
  (adult_price : ℕ) -- Price of adult ticket in cents
  (child_price : ℕ) -- Price of child ticket in cents
  (total_attendance : ℕ) -- Total number of attendees
  (total_revenue : ℕ) -- Total revenue in cents
  (h1 : adult_price = 60)
  (h2 : child_price = 25)
  (h3 : total_attendance = 280)
  (h4 : total_revenue = 14000) :
  ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adults * adult_price + children * child_price = total_revenue ∧
    children = 80 :=
by sorry

end NUMINAMATH_CALUDE_football_game_attendance_l4091_409100


namespace NUMINAMATH_CALUDE_equation_solutions_l4091_409109

theorem equation_solutions :
  -- Equation 1
  (∀ x : ℝ, x^2 - 5*x = 0 ↔ x = 0 ∨ x = 5) ∧
  -- Equation 2
  (∀ x : ℝ, (2*x + 1)^2 = 4 ↔ x = -3/2 ∨ x = 1/2) ∧
  -- Equation 3
  (∀ x : ℝ, x*(x - 1) + 3*(x - 1) = 0 ↔ x = 1 ∨ x = -3) ∧
  -- Equation 4
  (∀ x : ℝ, x^2 - 2*x - 8 = 0 ↔ x = -2 ∨ x = 4) := by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l4091_409109


namespace NUMINAMATH_CALUDE_exists_k_undecided_tournament_l4091_409189

/-- A tournament is represented as a function that takes two players and returns true if the first player defeats the second, and false otherwise. -/
def Tournament (n : ℕ) := Fin n → Fin n → Bool

/-- A tournament is k-undecided if for any set of k players, there exists a player who has defeated all of them. -/
def IsKUndecided (k : ℕ) (n : ℕ) (t : Tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k →
    ∃ (p : Fin n), ∀ (a : Fin n), a ∈ A → t p a = true

/-- For any positive integer k, there exists a k-undecided tournament with more than k players. -/
theorem exists_k_undecided_tournament (k : ℕ+) :
  ∃ (n : ℕ), n > k ∧ ∃ (t : Tournament n), IsKUndecided k n t :=
sorry

end NUMINAMATH_CALUDE_exists_k_undecided_tournament_l4091_409189


namespace NUMINAMATH_CALUDE_chicken_cost_per_person_l4091_409112

def grocery_cost : ℝ := 16
def beef_price_per_pound : ℝ := 4
def beef_pounds : ℝ := 3
def oil_price : ℝ := 1
def number_of_people : ℕ := 3

theorem chicken_cost_per_person (chicken_cost : ℝ) : 
  chicken_cost = grocery_cost - (beef_price_per_pound * beef_pounds + oil_price) →
  chicken_cost / number_of_people = 1 := by sorry

end NUMINAMATH_CALUDE_chicken_cost_per_person_l4091_409112


namespace NUMINAMATH_CALUDE_equal_kite_areas_condition_l4091_409153

/-- An isosceles triangle with perpendiculars from the intersection point of angle bisectors -/
structure IsoscelesTriangleWithPerpendiculars where
  /-- Length of the legs of the isosceles triangle -/
  leg_length : ℝ
  /-- Length of the base of the isosceles triangle -/
  base_length : ℝ
  /-- The triangle is isosceles -/
  isosceles : leg_length > 0
  /-- The perpendiculars divide the triangle into two smaller kites and one larger kite -/
  has_kites : True

/-- The theorem stating the condition for equal areas of kites -/
theorem equal_kite_areas_condition (t : IsoscelesTriangleWithPerpendiculars) :
  (∃ (small_kite_area larger_kite_area : ℝ),
    small_kite_area > 0 ∧ larger_kite_area > 0 ∧
    2 * small_kite_area = larger_kite_area) ↔
  t.base_length = 2/3 * t.leg_length :=
sorry

end NUMINAMATH_CALUDE_equal_kite_areas_condition_l4091_409153


namespace NUMINAMATH_CALUDE_log_inequality_l4091_409101

theorem log_inequality (h1 : 5^5 < 8^4) (h2 : 13^4 < 8^5) :
  Real.log 3 / Real.log 5 < Real.log 5 / Real.log 8 ∧
  Real.log 5 / Real.log 8 < Real.log 8 / Real.log 13 := by
sorry

end NUMINAMATH_CALUDE_log_inequality_l4091_409101


namespace NUMINAMATH_CALUDE_incorrect_guess_is_20th_bear_prove_incorrect_guess_is_20th_bear_l4091_409159

/-- Represents the color of a bear -/
inductive BearColor
| White
| Brown
| Black

/-- Represents a row of 1000 bears -/
def BearRow := Fin 1000 → BearColor

/-- Predicate to check if three consecutive bears have all three colors -/
def hasAllColors (row : BearRow) (i : Fin 998) : Prop :=
  ∃ (c1 c2 c3 : BearColor), 
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    row i = c1 ∧ row (i + 1) = c2 ∧ row (i + 2) = c3

/-- The main theorem stating that the 20th bear's color must be the incorrect guess -/
theorem incorrect_guess_is_20th_bear (row : BearRow) : Prop :=
  (∀ i : Fin 998, hasAllColors row i) →
  (row 1 = BearColor.White) →
  (row 399 = BearColor.Black) →
  (row 599 = BearColor.Brown) →
  (row 799 = BearColor.White) →
  (row 19 ≠ BearColor.Brown)

-- The proof of the theorem
theorem prove_incorrect_guess_is_20th_bear :
  ∃ (row : BearRow), incorrect_guess_is_20th_bear row :=
sorry

end NUMINAMATH_CALUDE_incorrect_guess_is_20th_bear_prove_incorrect_guess_is_20th_bear_l4091_409159


namespace NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l4091_409115

theorem smallest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  (a : ℝ) / 4 = (b : ℝ) / 5 →
  (a : ℝ) / 4 = (c : ℝ) / 7 →
  a + b + c = 180 →
  a = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l4091_409115


namespace NUMINAMATH_CALUDE_sum_of_unique_decimals_sum_of_unique_decimals_proof_l4091_409146

/-- The sum of all unique decimals formed by 4 distinct digit cards and 1 decimal point card -/
theorem sum_of_unique_decimals : ℝ :=
  let digit_sum := (0 : ℕ) + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
  let num_permutations := 24
  let num_decimal_positions := 4
  666.6

/-- The number of unique decimals that can be formed -/
def num_unique_decimals : ℕ := 72

theorem sum_of_unique_decimals_proof :
  sum_of_unique_decimals = 666.6 ∧ num_unique_decimals = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_unique_decimals_sum_of_unique_decimals_proof_l4091_409146


namespace NUMINAMATH_CALUDE_log_27_3_l4091_409188

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_27_3_l4091_409188


namespace NUMINAMATH_CALUDE_number_problem_l4091_409104

theorem number_problem :
  ∃ x : ℝ, x = (1/4) * x + 93.33333333333333 ∧ x = 124.44444444444444 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4091_409104


namespace NUMINAMATH_CALUDE_expression_value_l4091_409118

theorem expression_value (x y : ℝ) (h1 : x ≠ y) (h2 : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (x * y + 1)) :
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (x * y + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4091_409118


namespace NUMINAMATH_CALUDE_min_value_expression_l4091_409158

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  3 * a^2 + 3 * b^2 + 1 / (a + b)^2 + 4 / (a^2 * b^2) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l4091_409158


namespace NUMINAMATH_CALUDE_radio_price_reduction_l4091_409108

theorem radio_price_reduction (x : ℝ) :
  (∀ (P Q : ℝ), P > 0 ∧ Q > 0 →
    P * (1 - x / 100) * (Q * 1.8) = P * Q * 1.44) →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_radio_price_reduction_l4091_409108


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l4091_409150

-- Define the polynomial
def p (x : ℝ) : ℝ := (x^2 - 5*x + 6) * x * (x - 4) * (x - 6)

-- State the theorem
theorem roots_of_polynomial : 
  {x : ℝ | p x = 0} = {0, 2, 3, 4, 6} := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l4091_409150


namespace NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l4091_409194

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 4 = 0
def circle2 (x y : ℝ) : Prop := 3*x^2 + 3*y^2 + 6*x + 3*y - 15 = 0

-- Theorem for the first circle
theorem circle1_properties :
  ∃ (h k r : ℝ), 
    (h = -1 ∧ k = -2 ∧ r = 3) ∧
    ∀ (x y : ℝ), circle1 x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

-- Theorem for the second circle
theorem circle2_properties :
  ∃ (h k r : ℝ), 
    (h = -1 ∧ k = -1/2 ∧ r = 5/2) ∧
    ∀ (x y : ℝ), circle2 x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l4091_409194


namespace NUMINAMATH_CALUDE_parking_lot_problem_l4091_409125

theorem parking_lot_problem :
  ∀ (medium_cars small_cars : ℕ),
    medium_cars + small_cars = 36 →
    6 * medium_cars + 4 * small_cars = 176 →
    medium_cars = 16 ∧ small_cars = 20 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l4091_409125


namespace NUMINAMATH_CALUDE_sum_of_c_values_l4091_409120

theorem sum_of_c_values (b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, z^2 + b*z + c = 0 ↔ (z = x ∨ z = y)) →
  b = c - 1 →
  ∃ c₁ c₂ : ℝ, (∀ c' : ℝ, (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, z^2 + (c' - 1)*z + c' = 0 ↔ (z = x ∨ z = y)) ↔ (c' = c₁ ∨ c' = c₂)) ∧
  c₁ + c₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_c_values_l4091_409120


namespace NUMINAMATH_CALUDE_third_class_proportion_l4091_409147

theorem third_class_proportion (first_class second_class third_class : ℕ) 
  (h1 : first_class = 30)
  (h2 : second_class = 50)
  (h3 : third_class = 20) :
  (third_class : ℚ) / (first_class + second_class + third_class : ℚ) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_third_class_proportion_l4091_409147


namespace NUMINAMATH_CALUDE_sqrt_3_5_7_not_arithmetic_sequence_l4091_409155

theorem sqrt_3_5_7_not_arithmetic_sequence : 
  ¬ ∃ (d : ℝ), Real.sqrt 5 - Real.sqrt 3 = d ∧ Real.sqrt 7 - Real.sqrt 5 = d :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_5_7_not_arithmetic_sequence_l4091_409155


namespace NUMINAMATH_CALUDE_factor_polynomial_l4091_409139

theorem factor_polynomial (x : ℝ) : 80 * x^5 - 180 * x^9 = 20 * x^5 * (4 - 9 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l4091_409139


namespace NUMINAMATH_CALUDE_inequalities_for_ordered_reals_l4091_409122

theorem inequalities_for_ordered_reals 
  (a b c d : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 0 > c) 
  (h4 : c > d) : 
  (a + c > b + d) ∧ 
  (a * d^2 > b * c^2) ∧ 
  ((1 : ℝ) / (b * c) < (1 : ℝ) / (a * d)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_ordered_reals_l4091_409122


namespace NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l4091_409138

theorem x_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → (x - 1) * (x + 2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l4091_409138


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l4091_409113

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l4091_409113


namespace NUMINAMATH_CALUDE_inequality_proof_l4091_409167

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (a + b) / Real.sqrt (a * b * (1 - a * b)) + 
  (b + c) / Real.sqrt (b * c * (1 - b * c)) + 
  (c + a) / Real.sqrt (c * a * (1 - c * a)) ≤ 
  Real.sqrt 2 / (a * b * c) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4091_409167


namespace NUMINAMATH_CALUDE_bill_difference_l4091_409183

theorem bill_difference (mike_tip joe_tip : ℝ) (mike_percent joe_percent : ℝ) 
  (h1 : mike_tip = 2)
  (h2 : joe_tip = 2)
  (h3 : mike_percent = 0.1)
  (h4 : joe_percent = 0.2)
  (h5 : mike_tip = mike_percent * mike_bill)
  (h6 : joe_tip = joe_percent * joe_bill)
  : mike_bill - joe_bill = 10 := by
  sorry

end NUMINAMATH_CALUDE_bill_difference_l4091_409183


namespace NUMINAMATH_CALUDE_lake_superior_depth_l4091_409176

/-- The depth of a lake given its water surface elevation above sea level and lowest point below sea level -/
def lake_depth (water_surface_elevation : ℝ) (lowest_point_below_sea : ℝ) : ℝ :=
  water_surface_elevation + lowest_point_below_sea

/-- Theorem: The depth of Lake Superior at its deepest point is 400 meters -/
theorem lake_superior_depth :
  lake_depth 180 220 = 400 := by
  sorry

end NUMINAMATH_CALUDE_lake_superior_depth_l4091_409176


namespace NUMINAMATH_CALUDE_pigeonhole_socks_l4091_409190

theorem pigeonhole_socks (red blue : ℕ) (h1 : red = 10) (h2 : blue = 10) :
  ∃ n : ℕ, n = 3 ∧ 
  (∀ m : ℕ, m < n → ∃ f : Fin m → Bool, Function.Injective f) ∧
  (∀ f : Fin n → Bool, ¬Function.Injective f) :=
sorry

end NUMINAMATH_CALUDE_pigeonhole_socks_l4091_409190


namespace NUMINAMATH_CALUDE_skirt_cut_amount_l4091_409148

/-- The amount cut off the pants in inches -/
def pants_cut : ℝ := 0.5

/-- The additional amount cut off the skirt compared to the pants in inches -/
def additional_skirt_cut : ℝ := 0.25

/-- The total amount cut off the skirt in inches -/
def skirt_cut : ℝ := pants_cut + additional_skirt_cut

theorem skirt_cut_amount : skirt_cut = 0.75 := by sorry

end NUMINAMATH_CALUDE_skirt_cut_amount_l4091_409148


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l4091_409132

-- Define a regular decagon
structure RegularDecagon :=
  (vertices : Finset (ℕ × ℕ))
  (is_regular : vertices.card = 10)

-- Define a triangle formed by three vertices of the decagon
def Triangle (d : RegularDecagon) :=
  {t : Finset (ℕ × ℕ) // t ⊆ d.vertices ∧ t.card = 3}

-- Define a predicate for a triangle not sharing sides with the decagon
def NoSharedSides (d : RegularDecagon) (t : Triangle d) : Prop := sorry

-- Define the probability function
def Probability (d : RegularDecagon) : ℚ := sorry

-- State the theorem
theorem decagon_triangle_probability (d : RegularDecagon) :
  Probability d = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l4091_409132


namespace NUMINAMATH_CALUDE_reciprocals_multiply_to_one_no_real_roots_when_m_greater_than_one_l4091_409134

-- Definition of reciprocals
def are_reciprocals (x y : ℝ) : Prop := x * y = 1

-- Statement 1
theorem reciprocals_multiply_to_one (x y : ℝ) :
  are_reciprocals x y → x * y = 1 :=
sorry

-- Definition for real roots
def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c = 0

-- Statement 2
theorem no_real_roots_when_m_greater_than_one (m : ℝ) :
  m > 1 → ¬(has_real_roots 1 (-2) m) :=
sorry

end NUMINAMATH_CALUDE_reciprocals_multiply_to_one_no_real_roots_when_m_greater_than_one_l4091_409134


namespace NUMINAMATH_CALUDE_amys_candy_problem_l4091_409198

/-- Amy's candy problem -/
theorem amys_candy_problem (candy_given : ℕ) (difference : ℕ) : 
  candy_given = 6 → difference = 1 → candy_given - difference = 5 := by
  sorry

end NUMINAMATH_CALUDE_amys_candy_problem_l4091_409198


namespace NUMINAMATH_CALUDE_remainder_problem_l4091_409140

theorem remainder_problem (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℕ, 1816 = k * x + 6) : 
  ∃ l : ℕ, 1442 = l * x + 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4091_409140


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l4091_409107

theorem modulo_eleven_residue :
  (332 + 6 * 44 + 8 * 176 + 3 * 22) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l4091_409107


namespace NUMINAMATH_CALUDE_original_number_proof_l4091_409128

theorem original_number_proof (x : ℝ) : 1 + 1/x = 11/5 → x = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l4091_409128


namespace NUMINAMATH_CALUDE_edward_earnings_l4091_409156

theorem edward_earnings : 
  let lawn_pay : ℕ := 8
  let garden_pay : ℕ := 12
  let lawns_mowed : ℕ := 5
  let gardens_cleaned : ℕ := 3
  let fuel_cost : ℕ := 10
  let equipment_cost : ℕ := 15
  let initial_savings : ℕ := 7

  let total_earnings := lawn_pay * lawns_mowed + garden_pay * gardens_cleaned
  let total_expenses := fuel_cost + equipment_cost
  let final_amount := total_earnings + initial_savings - total_expenses

  final_amount = 58 := by sorry

end NUMINAMATH_CALUDE_edward_earnings_l4091_409156


namespace NUMINAMATH_CALUDE_kishore_savings_l4091_409137

/-- Proves that given the total expenses and the fact that they represent 90% of the salary,
    the 10% savings amount to the correct value. -/
theorem kishore_savings (total_expenses : ℕ) (monthly_salary : ℕ) : 
  total_expenses = 20700 →
  total_expenses = (90 * monthly_salary) / 100 →
  (10 * monthly_salary) / 100 = 2300 :=
by sorry

end NUMINAMATH_CALUDE_kishore_savings_l4091_409137


namespace NUMINAMATH_CALUDE_pet_show_big_dogs_l4091_409165

theorem pet_show_big_dogs :
  ∀ (big_dogs small_dogs : ℕ),
  (big_dogs : ℚ) / small_dogs = 3 / 17 →
  big_dogs + small_dogs = 80 →
  big_dogs = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_show_big_dogs_l4091_409165


namespace NUMINAMATH_CALUDE_cryptarithm_solution_exists_l4091_409123

theorem cryptarithm_solution_exists : ∃ (Φ E B P A J : ℕ), 
  Φ < 10 ∧ E < 10 ∧ B < 10 ∧ P < 10 ∧ A < 10 ∧ J < 10 ∧
  Φ ≠ E ∧ Φ ≠ B ∧ Φ ≠ P ∧ Φ ≠ A ∧ Φ ≠ J ∧
  E ≠ B ∧ E ≠ P ∧ E ≠ A ∧ E ≠ J ∧
  B ≠ P ∧ B ≠ A ∧ B ≠ J ∧
  P ≠ A ∧ P ≠ J ∧
  A ≠ J ∧
  E ≠ 0 ∧ A ≠ 0 ∧ J ≠ 0 ∧
  (Φ : ℚ) / E + (B * 10 + P : ℚ) / (A * J) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_exists_l4091_409123


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4091_409196

theorem partial_fraction_decomposition :
  ∃ (A B C D : ℚ),
    (A = 1/15) ∧ (B = 5/2) ∧ (C = -59/6) ∧ (D = 42/5) ∧
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 →
      (x^3 - 7) / ((x - 2) * (x - 3) * (x - 5) * (x - 7)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x - 7)) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4091_409196


namespace NUMINAMATH_CALUDE_exists_large_ratio_l4091_409193

def sequence_property (a b : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a n > 0 ∧ b n > 0) ∧
  (∀ n : ℕ+, a (n + 1) * b (n + 1) = a n ^ 2 + b n ^ 2) ∧
  (∀ n : ℕ+, a (n + 1) + b (n + 1) = a n * b n) ∧
  (∀ n : ℕ+, a n ≥ b n)

theorem exists_large_ratio (a b : ℕ+ → ℝ) (h : sequence_property a b) :
  ∃ n : ℕ+, a n / b n > 2023^2023 := by
  sorry

end NUMINAMATH_CALUDE_exists_large_ratio_l4091_409193


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l4091_409110

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  startingNumber : ℕ

/-- Generates the list of selected student numbers -/
def generateSample (s : SystematicSampling) : List ℕ :=
  let interval := s.totalStudents / s.sampleSize
  List.range s.sampleSize |>.map (fun i => s.startingNumber + i * interval)

theorem systematic_sampling_result :
  ∀ (s : SystematicSampling),
    s.totalStudents = 50 →
    s.sampleSize = 5 →
    s.startingNumber = 3 →
    generateSample s = [3, 13, 23, 33, 43] :=
by
  sorry

#eval generateSample ⟨50, 5, 3⟩

end NUMINAMATH_CALUDE_systematic_sampling_result_l4091_409110


namespace NUMINAMATH_CALUDE_set_union_problem_l4091_409160

theorem set_union_problem (a b : ℕ) :
  let M : Set ℕ := {3, 2^a}
  let N : Set ℕ := {a, b}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l4091_409160


namespace NUMINAMATH_CALUDE_longest_line_segment_in_quarter_pie_l4091_409151

theorem longest_line_segment_in_quarter_pie (d : ℝ) (h : d = 16) :
  let r := d / 2
  let θ := π / 2
  let chord_length := 2 * r * Real.sin (θ / 2)
  chord_length ^ 2 = 128 :=
by sorry

end NUMINAMATH_CALUDE_longest_line_segment_in_quarter_pie_l4091_409151


namespace NUMINAMATH_CALUDE_billboard_count_l4091_409143

theorem billboard_count (h1 : ℕ) (h2 : ℕ) (h3 : ℕ) (total_hours : ℕ) (avg : ℕ) 
  (h1_count : h1 = 17)
  (h2_count : h2 = 20)
  (hours : total_hours = 3)
  (average : avg = 20)
  (avg_def : avg * total_hours = h1 + h2 + h3) :
  h3 = 23 := by
sorry

end NUMINAMATH_CALUDE_billboard_count_l4091_409143


namespace NUMINAMATH_CALUDE_nine_points_interior_lattice_point_l4091_409175

/-- A lattice point in 3D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ
  z : ℤ

/-- The statement that there exists an interior lattice point -/
def exists_interior_lattice_point (points : Finset LatticePoint) : Prop :=
  ∃ p q : LatticePoint, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧
    ∃ r : LatticePoint, r.x = (p.x + q.x) / 2 ∧ 
                        r.y = (p.y + q.y) / 2 ∧ 
                        r.z = (p.z + q.z) / 2

/-- The main theorem -/
theorem nine_points_interior_lattice_point 
  (points : Finset LatticePoint) 
  (h : points.card = 9) : 
  exists_interior_lattice_point points := by
  sorry

#check nine_points_interior_lattice_point

end NUMINAMATH_CALUDE_nine_points_interior_lattice_point_l4091_409175


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l4091_409145

theorem simplify_and_rationalize : 
  (Real.sqrt 8 / Real.sqrt 3) * (Real.sqrt 25 / Real.sqrt 30) * (Real.sqrt 16 / Real.sqrt 21) = 
  4 * Real.sqrt 14 / 63 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l4091_409145


namespace NUMINAMATH_CALUDE_bake_sale_group_composition_l4091_409177

theorem bake_sale_group_composition (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls = total / 2) →  -- Initially, 50% of the group are girls
  (initial_girls - 3 = (total * 2) / 5) →  -- After changes, 40% are girls
  (initial_girls = 15) :=
by
  sorry

#check bake_sale_group_composition

end NUMINAMATH_CALUDE_bake_sale_group_composition_l4091_409177


namespace NUMINAMATH_CALUDE_unique_factors_of_2013_l4091_409161

theorem unique_factors_of_2013 (m n : ℕ) (h1 : m < n) (h2 : n < 2 * m) (h3 : m * n = 2013) :
  m = 33 ∧ n = 61 :=
sorry

end NUMINAMATH_CALUDE_unique_factors_of_2013_l4091_409161


namespace NUMINAMATH_CALUDE_waiter_customers_l4091_409168

theorem waiter_customers (non_tipping_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : 
  non_tipping_customers = 5 →
  tip_amount = 8 →
  total_tips = 32 →
  non_tipping_customers + (total_tips / tip_amount) = 9 :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l4091_409168


namespace NUMINAMATH_CALUDE_one_root_cubic_equation_a_range_l4091_409114

theorem one_root_cubic_equation_a_range (a : ℝ) : 
  (∃! x : ℝ, x^3 + (1-3*a)*x^2 + 2*a^2*x - 2*a*x + x + a^2 - a = 0) → 
  (-Real.sqrt 3 / 2 < a ∧ a < Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_one_root_cubic_equation_a_range_l4091_409114


namespace NUMINAMATH_CALUDE_tenth_term_value_l4091_409121

theorem tenth_term_value (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n : ℕ, S n = n * (2 * n + 1)) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  a 10 = 39 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_value_l4091_409121


namespace NUMINAMATH_CALUDE_johnny_marble_selection_l4091_409144

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of marbles in the collection -/
def total_marbles : ℕ := 10

/-- The number of marbles chosen in the first step -/
def first_choice : ℕ := 4

/-- The number of marbles chosen in the second step -/
def second_choice : ℕ := 2

/-- The theorem stating the total number of ways Johnny can complete the selection process -/
theorem johnny_marble_selection :
  (choose total_marbles first_choice) * (choose first_choice second_choice) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_johnny_marble_selection_l4091_409144


namespace NUMINAMATH_CALUDE_factorization_of_x_power_difference_l4091_409105

theorem factorization_of_x_power_difference (m : ℕ) (x : ℝ) (hm : m > 1) :
  x^m - x^(m-2) = x^(m-2) * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_power_difference_l4091_409105


namespace NUMINAMATH_CALUDE_geric_bills_l4091_409157

theorem geric_bills (jessa kyla geric : ℕ) : 
  geric = 2 * kyla →
  kyla = jessa - 2 →
  jessa - 3 = 7 →
  geric = 16 := by
sorry

end NUMINAMATH_CALUDE_geric_bills_l4091_409157
