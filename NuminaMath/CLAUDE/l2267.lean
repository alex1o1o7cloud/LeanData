import Mathlib

namespace NUMINAMATH_CALUDE_ratio_calculation_l2267_226708

theorem ratio_calculation : 
  let numerator := (14^4 + 484) * (26^4 + 484) * (38^4 + 484) * (50^4 + 484) * (62^4 + 484)
  let denominator := (8^4 + 484) * (20^4 + 484) * (32^4 + 484) * (44^4 + 484) * (56^4 + 484)
  numerator / denominator = -423 := by
sorry

end NUMINAMATH_CALUDE_ratio_calculation_l2267_226708


namespace NUMINAMATH_CALUDE_problem_statement_l2267_226775

theorem problem_statement (a b : ℕ) : 
  a = 105 → a^3 = 21 * 35 * 45 * b → b = 105 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2267_226775


namespace NUMINAMATH_CALUDE_wall_area_calculation_l2267_226737

/-- The area of a rectangular wall with width and height of 4 feet is 16 square feet. -/
theorem wall_area_calculation : 
  ∀ (width height area : ℝ), 
  width = 4 → 
  height = 4 → 
  area = width * height → 
  area = 16 :=
by sorry

end NUMINAMATH_CALUDE_wall_area_calculation_l2267_226737


namespace NUMINAMATH_CALUDE_combined_average_age_l2267_226784

theorem combined_average_age (room_a_count : ℕ) (room_b_count : ℕ) 
  (room_a_avg : ℝ) (room_b_avg : ℝ) :
  room_a_count = 8 →
  room_b_count = 3 →
  room_a_avg = 45 →
  room_b_avg = 20 →
  (room_a_count * room_a_avg + room_b_count * room_b_avg) / (room_a_count + room_b_count) = 38 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l2267_226784


namespace NUMINAMATH_CALUDE_parking_space_unpainted_side_l2267_226763

/-- Represents a rectangular parking space with three painted sides. -/
structure ParkingSpace where
  width : ℝ
  length : ℝ
  painted_sum : ℝ
  area : ℝ

/-- The length of the unpainted side of a parking space. -/
def unpainted_side_length (p : ParkingSpace) : ℝ := p.length

theorem parking_space_unpainted_side
  (p : ParkingSpace)
  (h1 : p.painted_sum = 37)
  (h2 : p.area = 126)
  (h3 : p.painted_sum = 2 * p.width + p.length)
  (h4 : p.area = p.width * p.length) :
  unpainted_side_length p = 9 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_unpainted_side_l2267_226763


namespace NUMINAMATH_CALUDE_quadratic_sum_theorem_l2267_226761

/-- The quadratic function -3x^2 - 27x + 81 -/
def f (x : ℝ) : ℝ := -3 * x^2 - 27 * x + 81

/-- The quadratic function a(x+b)^2 + c -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum_theorem :
  ∃ (a b c : ℝ), (∀ x, f x = g a b c x) ∧ (a + b + c = 143.25) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_theorem_l2267_226761


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2267_226797

def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {0, 2, 3, 5}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2267_226797


namespace NUMINAMATH_CALUDE_age_difference_proof_l2267_226711

/-- Given the ages of Katie's daughter, Lavinia's daughter, and Lavinia's son, prove that Lavinia's son is 22 years older than Lavinia's daughter. -/
theorem age_difference_proof (katie_daughter_age lavinia_daughter_age lavinia_son_age : ℕ) :
  katie_daughter_age = 12 →
  lavinia_daughter_age = katie_daughter_age - 10 →
  lavinia_son_age = 2 * katie_daughter_age →
  lavinia_son_age - lavinia_daughter_age = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2267_226711


namespace NUMINAMATH_CALUDE_largest_multiple_of_nine_l2267_226709

theorem largest_multiple_of_nine (n : ℤ) : 
  (n % 9 = 0 ∧ -n > -100) → n ≤ 99 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_nine_l2267_226709


namespace NUMINAMATH_CALUDE_problem_solution_l2267_226735

theorem problem_solution (p q r u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2267_226735


namespace NUMINAMATH_CALUDE_student_representatives_distribution_l2267_226733

theorem student_representatives_distribution (n m : ℕ) : 
  n = 6 ∧ m = 4 → (Nat.choose (n + m - 2) (m - 1) = Nat.choose 5 3) := by
  sorry

end NUMINAMATH_CALUDE_student_representatives_distribution_l2267_226733


namespace NUMINAMATH_CALUDE_tim_grew_44_cantaloupes_l2267_226723

/-- The number of cantaloupes Fred grew -/
def fred_cantaloupes : ℕ := 38

/-- The total number of cantaloupes Fred and Tim grew together -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Tim grew -/
def tim_cantaloupes : ℕ := total_cantaloupes - fred_cantaloupes

/-- Proof that Tim grew 44 cantaloupes -/
theorem tim_grew_44_cantaloupes : tim_cantaloupes = 44 := by
  sorry

end NUMINAMATH_CALUDE_tim_grew_44_cantaloupes_l2267_226723


namespace NUMINAMATH_CALUDE_min_throws_correct_l2267_226785

/-- The probability of hitting the target in a single throw -/
def p : ℝ := 0.6

/-- The desired minimum probability of hitting the target at least once -/
def target_prob : ℝ := 0.9

/-- The minimum number of throws needed to exceed the target probability -/
def min_throws : ℕ := 3

/-- Theorem stating that min_throws is the minimum number of throws needed -/
theorem min_throws_correct :
  (∀ n : ℕ, n < min_throws → 1 - (1 - p)^n ≤ target_prob) ∧
  (1 - (1 - p)^min_throws > target_prob) :=
sorry

end NUMINAMATH_CALUDE_min_throws_correct_l2267_226785


namespace NUMINAMATH_CALUDE_peanut_butter_candy_count_l2267_226778

/-- The number of candy pieces in the banana jar -/
def banana_candy : ℕ := 43

/-- The number of candy pieces in the grape jar -/
def grape_candy : ℕ := banana_candy + 5

/-- The number of candy pieces in the peanut butter jar -/
def peanut_butter_candy : ℕ := 4 * grape_candy

/-- Theorem: The peanut butter jar contains 192 pieces of candy -/
theorem peanut_butter_candy_count : peanut_butter_candy = 192 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_candy_count_l2267_226778


namespace NUMINAMATH_CALUDE_series_sum_is_36118_l2267_226764

/-- The sign of a term in the series based on its position -/
def sign (n : ℕ) : ℤ :=
  if n ≤ 8 then 1
  else if n ≤ 35 then -1
  else if n ≤ 80 then 1
  else if n ≤ 143 then -1
  -- Continue this pattern up to 10003
  else if n ≤ 9801 then -1
  else 1

/-- The nth term of the series -/
def term (n : ℕ) : ℤ := sign n * n

/-- The sum of the series from 1 to 10003 -/
def seriesSum : ℤ := (List.range 10003).map term |>.sum

theorem series_sum_is_36118 : seriesSum = 36118 := by
  sorry

#eval seriesSum

end NUMINAMATH_CALUDE_series_sum_is_36118_l2267_226764


namespace NUMINAMATH_CALUDE_reciprocal_equality_l2267_226773

theorem reciprocal_equality (a b : ℝ) : 
  (1 / a = -8) → (1 / (-b) = 8) → (a = b) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equality_l2267_226773


namespace NUMINAMATH_CALUDE_total_students_l2267_226705

theorem total_students (n : ℕ) (male_students : ℕ) (selected_students : ℕ) (male_participants : ℕ) (female_participants : ℕ) : 
  male_students = 550 →
  selected_students = n / 10 →
  male_participants = female_participants + 10 →
  male_participants = male_students / 10 →
  selected_students = male_participants + female_participants →
  n = 1000 := by
sorry

end NUMINAMATH_CALUDE_total_students_l2267_226705


namespace NUMINAMATH_CALUDE_percent_of_double_is_nine_l2267_226713

theorem percent_of_double_is_nine (x : ℝ) : 
  x > 0 → (0.01 * x * (2 * x) = 9) → x = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_double_is_nine_l2267_226713


namespace NUMINAMATH_CALUDE_inequality_solution_minimum_value_minimum_value_condition_l2267_226722

-- Part 1: Inequality solution
theorem inequality_solution (x : ℝ) :
  (2 * x + 1) / (3 - x) ≥ 1 ↔ x ≤ 1 ∨ x > 2 :=
sorry

-- Part 2: Minimum value
theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) ≥ 25 :=
sorry

theorem minimum_value_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) = 25 ↔ x = 2/5 ∧ y = 3/5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_minimum_value_minimum_value_condition_l2267_226722


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l2267_226720

/-- The inradius of a right triangle with sides 12, 16, and 20 is 4 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 12 ∧ b = 16 ∧ c = 20 →  -- Side lengths
  a^2 + b^2 = c^2 →           -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l2267_226720


namespace NUMINAMATH_CALUDE_reflect_P_x_axis_l2267_226727

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system. -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The point P in the Cartesian coordinate system. -/
def P : ℝ × ℝ := (1, -2)

/-- Theorem stating that reflecting P(1,-2) across the x-axis results in (1,2). -/
theorem reflect_P_x_axis : reflect_x P = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_x_axis_l2267_226727


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2267_226788

/-- The repeating decimal 0.3̄45 as a real number -/
def repeating_decimal : ℚ := 3/10 + 45/990

/-- The fraction 83/110 -/
def target_fraction : ℚ := 83/110

/-- Theorem stating that the repeating decimal 0.3̄45 is equal to the fraction 83/110 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2267_226788


namespace NUMINAMATH_CALUDE_smallest_value_of_root_products_l2267_226741

def g (x : ℝ) : ℝ := x^4 + 16*x^3 + 69*x^2 + 112*x + 64

theorem smallest_value_of_root_products (w₁ w₂ w₃ w₄ : ℝ) 
  (h₁ : g w₁ = 0) (h₂ : g w₂ = 0) (h₃ : g w₃ = 0) (h₄ : g w₄ = 0) :
  ∃ (min : ℝ), min = 8 ∧ ∀ (p : ℝ), p = |w₁*w₂ + w₃*w₄| → p ≥ min :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_root_products_l2267_226741


namespace NUMINAMATH_CALUDE_min_sum_squares_l2267_226749

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) 
  (pos₁ : y₁ > 0) (pos₂ : y₂ > 0) (pos₃ : y₃ > 0)
  (sum_constraint : y₁ + 3*y₂ + 4*y₃ = 72) :
  ∃ (min : ℝ), min = 2592/13 ∧ 
  ∀ (z₁ z₂ z₃ : ℝ), z₁ > 0 → z₂ > 0 → z₃ > 0 → 
  z₁ + 3*z₂ + 4*z₃ = 72 → 
  z₁^2 + z₂^2 + z₃^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2267_226749


namespace NUMINAMATH_CALUDE_black_card_fraction_l2267_226769

theorem black_card_fraction (total : ℕ) (red_fraction : ℚ) (green : ℕ) : 
  total = 120 → 
  red_fraction = 2 / 5 → 
  green = 32 → 
  (5 : ℚ) / 9 = (total - (red_fraction * total) - green) / (total - (red_fraction * total)) := by
  sorry

end NUMINAMATH_CALUDE_black_card_fraction_l2267_226769


namespace NUMINAMATH_CALUDE_square_root_of_4096_l2267_226752

theorem square_root_of_4096 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 4096) : x = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_4096_l2267_226752


namespace NUMINAMATH_CALUDE_evaluate_expression_l2267_226718

theorem evaluate_expression (x y z : ℚ) 
  (hx : x = 1/4) (hy : y = 4/5) (hz : z = -2) : 
  x^3 * y^2 * z^2 = 1/25 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2267_226718


namespace NUMINAMATH_CALUDE_water_consumption_theorem_l2267_226700

/-- The amount of water drunk by the traveler and his camel in gallons -/
def total_water_gallons (traveler_ounces : ℕ) (camel_multiplier : ℕ) (ounces_per_gallon : ℕ) : ℚ :=
  (traveler_ounces + traveler_ounces * camel_multiplier) / ounces_per_gallon

/-- Theorem stating that the total water drunk is 2 gallons -/
theorem water_consumption_theorem :
  total_water_gallons 32 7 128 = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_theorem_l2267_226700


namespace NUMINAMATH_CALUDE_function_properties_l2267_226706

def f (a b c x : ℝ) := a * x^4 + b * x^2 + c

theorem function_properties (a b c : ℝ) :
  f a b c 0 = 1 ∧
  (∀ x, x = 1 → f a b c x + 2 = x) ∧
  f a b c 1 = -1 →
  a = 5/2 ∧ c = 1 ∧
  ∀ x, (- (3 * Real.sqrt 10) / 10 < x ∧ x < 0) ∨ (3 * Real.sqrt 10 / 10 < x) →
    ∀ y, x < y → f a b c x < f a b c y :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2267_226706


namespace NUMINAMATH_CALUDE_inverse_proportion_l2267_226783

/-- Given that γ is inversely proportional to δ, prove that if γ = 5 when δ = 15, then γ = 5/3 when δ = 45. -/
theorem inverse_proportion (γ δ : ℝ) (h : ∃ k : ℝ, ∀ x y, γ * x = k ∧ y * δ = k) 
  (h1 : γ = 5 ∧ δ = 15) : 
  (γ = 5/3 ∧ δ = 45) :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_l2267_226783


namespace NUMINAMATH_CALUDE_boxes_sold_saturday_l2267_226798

theorem boxes_sold_saturday (saturday_sales : ℕ) (sunday_sales : ℕ) : 
  sunday_sales = saturday_sales + saturday_sales / 2 →
  saturday_sales + sunday_sales = 150 →
  saturday_sales = 60 := by
sorry

end NUMINAMATH_CALUDE_boxes_sold_saturday_l2267_226798


namespace NUMINAMATH_CALUDE_min_value_expression_l2267_226746

theorem min_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  4 ≤ a^2 + 2 * Real.sqrt (a * b) + Real.rpow (a^2 * b * c) (1/3) ∧
  ∃ a' b' c' : ℝ, 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 3 ∧
    a'^2 + 2 * Real.sqrt (a' * b') + Real.rpow (a'^2 * b' * c') (1/3) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2267_226746


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l2267_226743

theorem gcd_power_two_minus_one :
  Nat.gcd (2^1998 - 1) (2^1989 - 1) = 2^9 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l2267_226743


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l2267_226729

/-- A function that returns the digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with 37 -/
def ends_with_37 (n : ℕ) : Prop := sorry

theorem smallest_number_with_conditions : 
  ∀ n : ℕ, 
    n ≥ 99937 → 
    (ends_with_37 n ∧ digit_sum n = 37 ∧ n % 37 = 0) → 
    n = 99937 := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l2267_226729


namespace NUMINAMATH_CALUDE_doubled_container_volume_l2267_226744

/-- A cylindrical container that can hold water -/
structure Container :=
  (volume : ℝ)
  (isOriginal : Bool)

/-- Double the dimensions of a container -/
def doubleContainer (c : Container) : Container :=
  { volume := 8 * c.volume, isOriginal := false }

theorem doubled_container_volume (c : Container) 
  (h1 : c.isOriginal = true) 
  (h2 : c.volume = 3) : 
  (doubleContainer c).volume = 24 := by
sorry

end NUMINAMATH_CALUDE_doubled_container_volume_l2267_226744


namespace NUMINAMATH_CALUDE_angle_bisector_length_l2267_226703

/-- Given a triangle PQR with side lengths PQ and PR, and the cosine of angle P,
    calculate the length of the angle bisector PS. -/
theorem angle_bisector_length (PQ PR : ℝ) (cos_P : ℝ) (h_PQ : PQ = 4) (h_PR : PR = 8) (h_cos_P : cos_P = 1/9) :
  ∃ (PS : ℝ), PS = Real.sqrt ((43280 - 128 * Real.sqrt 41) / 81) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l2267_226703


namespace NUMINAMATH_CALUDE_ice_cream_volume_l2267_226779

/-- The volume of ice cream in a cone with a spherical top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let sphere_volume := (4 / 3) * π * r^3
  h = 12 ∧ r = 3 → cone_volume + sphere_volume = 72 * π := by sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l2267_226779


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l2267_226701

theorem quadratic_monotonicity (a b c : ℝ) (h_a : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y) →
  (f 1 < f 5) ∧
  ¬ ((f 1 < f 5) → (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l2267_226701


namespace NUMINAMATH_CALUDE_peanuts_in_box_l2267_226774

/-- Given a box with an initial number of peanuts and a number of peanuts added,
    calculate the total number of peanuts in the box. -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: If there are 4 peanuts in a box and 2 more are added,
    the total number of peanuts in the box is 6. -/
theorem peanuts_in_box : total_peanuts 4 2 = 6 := by sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l2267_226774


namespace NUMINAMATH_CALUDE_tree_arrangement_probability_l2267_226742

def maple_trees : ℕ := 4
def oak_trees : ℕ := 5
def birch_trees : ℕ := 6
def total_trees : ℕ := maple_trees + oak_trees + birch_trees

def valid_arrangements : ℕ := (Nat.choose 7 maple_trees) * 1

def total_arrangements : ℕ := (Nat.factorial total_trees) / 
  (Nat.factorial maple_trees * Nat.factorial oak_trees * Nat.factorial birch_trees)

theorem tree_arrangement_probability : 
  (valid_arrangements : ℚ) / total_arrangements = 7 / 166320 := by sorry

end NUMINAMATH_CALUDE_tree_arrangement_probability_l2267_226742


namespace NUMINAMATH_CALUDE_inequality_proof_l2267_226766

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2267_226766


namespace NUMINAMATH_CALUDE_acute_angles_sum_l2267_226777

theorem acute_angles_sum (a b : Real) : 
  0 < a ∧ a < π/2 →
  0 < b ∧ b < π/2 →
  4 * Real.sin a ^ 2 + 3 * Real.sin b ^ 2 = 1 →
  4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 0 →
  2 * a + 3 * b = π/2 := by
sorry

end NUMINAMATH_CALUDE_acute_angles_sum_l2267_226777


namespace NUMINAMATH_CALUDE_smallest_consecutive_cubes_with_square_difference_l2267_226782

theorem smallest_consecutive_cubes_with_square_difference :
  (∀ n : ℕ, n < 7 → ¬∃ k : ℕ, (n + 1)^3 - n^3 = k^2) ∧
  ∃ k : ℕ, 8^3 - 7^3 = k^2 := by
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_cubes_with_square_difference_l2267_226782


namespace NUMINAMATH_CALUDE_total_apples_correct_l2267_226740

/-- The number of apples Bill picked from the orchard -/
def total_apples : ℕ := 56

/-- The number of children Bill has -/
def num_children : ℕ := 2

/-- The number of apples each child takes for teachers -/
def apples_per_child : ℕ := 3

/-- The number of teachers each child gives apples to -/
def num_teachers : ℕ := 2

/-- The number of pies Jill bakes -/
def num_pies : ℕ := 2

/-- The number of apples used per pie -/
def apples_per_pie : ℕ := 10

/-- The number of apples Bill has left -/
def apples_left : ℕ := 24

/-- Theorem stating that the total number of apples Bill picked is correct -/
theorem total_apples_correct :
  total_apples = 
    num_children * apples_per_child * num_teachers +
    num_pies * apples_per_pie +
    apples_left :=
by sorry

end NUMINAMATH_CALUDE_total_apples_correct_l2267_226740


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l2267_226702

theorem unique_three_digit_divisible_by_11 : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 10 = 3 ∧          -- units digit is 3
  n / 100 = 6 ∧         -- hundreds digit is 6
  n % 11 = 0 ∧          -- divisible by 11
  n = 693               -- the number is 693
  := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l2267_226702


namespace NUMINAMATH_CALUDE_shopping_total_proof_l2267_226772

def toy_count : ℕ := 5
def toy_price : ℚ := 3
def toy_discount : ℚ := 0.20

def book_count : ℕ := 3
def book_price : ℚ := 8
def book_discount : ℚ := 0.15

def shirt_count : ℕ := 2
def shirt_price : ℚ := 12
def shirt_discount : ℚ := 0.25

def total_paid : ℚ := 50.40

theorem shopping_total_proof :
  (toy_count : ℚ) * toy_price * (1 - toy_discount) +
  (book_count : ℚ) * book_price * (1 - book_discount) +
  (shirt_count : ℚ) * shirt_price * (1 - shirt_discount) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_shopping_total_proof_l2267_226772


namespace NUMINAMATH_CALUDE_solution_product_l2267_226734

theorem solution_product (p q : ℝ) : 
  (p - 6) * (3 * p + 10) = p^2 - 19 * p + 50 →
  (q - 6) * (3 * q + 10) = q^2 - 19 * q + 50 →
  p ≠ q →
  (p + 2) * (q + 2) = 108 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l2267_226734


namespace NUMINAMATH_CALUDE_pencil_difference_l2267_226716

/-- The cost of pencils Jamar bought -/
def jamar_cost : ℚ := 325 / 100

/-- The cost of pencils Sharona bought -/
def sharona_cost : ℚ := 425 / 100

/-- The minimum number of pencils Jamar bought -/
def jamar_min_pencils : ℕ := 15

/-- The cost difference between Sharona's and Jamar's purchases -/
def cost_difference : ℚ := sharona_cost - jamar_cost

/-- The theorem stating the difference in the number of pencils bought -/
theorem pencil_difference : ∃ (jamar_pencils sharona_pencils : ℕ) (price_per_pencil : ℚ), 
  jamar_pencils ≥ jamar_min_pencils ∧
  price_per_pencil > 1 / 100 ∧
  jamar_cost = jamar_pencils * price_per_pencil ∧
  sharona_cost = sharona_pencils * price_per_pencil ∧
  sharona_pencils - jamar_pencils = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_difference_l2267_226716


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_A_union_B_intersect_C_nonempty_iff_l2267_226754

open Set

-- Define the sets A, B, and C
def A : Set ℝ := Ioc 2 3
def B : Set ℝ := Ioo 1 3
def C (m : ℝ) : Set ℝ := Ici m

-- Statement for part (1)
theorem complement_A_intersect_B : (Aᶜ ∩ B) = Ico 1 2 := by sorry

-- Statement for part (2)
theorem A_union_B_intersect_C_nonempty_iff (m : ℝ) :
  ((A ∪ B) ∩ C m).Nonempty ↔ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_A_union_B_intersect_C_nonempty_iff_l2267_226754


namespace NUMINAMATH_CALUDE_triangle_right_angle_l2267_226768

theorem triangle_right_angle (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- sides are positive
  0 < A ∧ 0 < B ∧ 0 < C →  -- angles are positive
  A + B + C = π →  -- sum of angles in a triangle
  a * Real.cos A + b * Real.cos B = c * Real.cos C →  -- given condition
  a^2 = b^2 + c^2  -- conclusion: right triangle with a as hypotenuse
  := by sorry

end NUMINAMATH_CALUDE_triangle_right_angle_l2267_226768


namespace NUMINAMATH_CALUDE_mischievous_polynomial_at_two_l2267_226786

/-- A quadratic polynomial of the form x^2 - px + q -/
structure MischievousPolynomial where
  p : ℝ
  q : ℝ

/-- Predicate for a mischievous polynomial -/
def isMischievous (poly : MischievousPolynomial) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ (x : ℝ), (x^2 - poly.p * (x^2 - poly.p * x + poly.q) + poly.q = 0) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))

/-- The product of roots of a quadratic polynomial -/
def rootProduct (poly : MischievousPolynomial) : ℝ := poly.q

/-- The value of the polynomial at x = 2 -/
def evalAtTwo (poly : MischievousPolynomial) : ℝ := 4 - 2 * poly.p + poly.q

/-- The main theorem -/
theorem mischievous_polynomial_at_two :
  ∀ (poly : MischievousPolynomial),
    isMischievous poly →
    (∀ (other : MischievousPolynomial), isMischievous other → rootProduct poly ≤ rootProduct other) →
    evalAtTwo poly = -1 := by
  sorry

end NUMINAMATH_CALUDE_mischievous_polynomial_at_two_l2267_226786


namespace NUMINAMATH_CALUDE_complex_number_equality_l2267_226748

theorem complex_number_equality (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z + 3) ∧ 
  Complex.abs (z - 1) = Complex.abs (z - Complex.I) →
  z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2267_226748


namespace NUMINAMATH_CALUDE_square_areas_square_areas_concrete_l2267_226717

/-- Given a square with area 100, prove the areas of the inscribed square and right triangle --/
theorem square_areas (S : Real) (h1 : S^2 = 100) :
  let small_square_area := S^2 / 4
  let right_triangle_area := S^2 / 16
  (small_square_area = 50) ∧ (right_triangle_area = 12.5) := by
  sorry

/-- Alternative formulation using concrete numbers --/
theorem square_areas_concrete :
  let large_square_area := 100
  let small_square_area := large_square_area / 2
  let right_triangle_area := large_square_area / 8
  (small_square_area = 50) ∧ (right_triangle_area = 12.5) := by
  sorry

end NUMINAMATH_CALUDE_square_areas_square_areas_concrete_l2267_226717


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2267_226725

/-- Given vectors a, b, and c in ℝ², prove that if (a + k * c) is parallel to (2 * b - a), then k = -16/13 -/
theorem parallel_vectors_k_value (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (3, 2)) 
    (hb : b = (-1, 2)) 
    (hc : c = (4, 1)) 
    (h_parallel : ∃ (t : ℝ), t • (a.1 + k * c.1, a.2 + k * c.2) = (2 * b.1 - a.1, 2 * b.2 - a.2)) :
  k = -16/13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2267_226725


namespace NUMINAMATH_CALUDE_total_cows_count_l2267_226728

/-- The number of cows in each herd -/
def cows_per_herd : ℕ := 150

/-- The number of herds in the farm -/
def number_of_herds : ℕ := 32

/-- The total number of cows on the farm -/
def total_cows : ℕ := cows_per_herd * number_of_herds

theorem total_cows_count : total_cows = 4800 := by
  sorry

end NUMINAMATH_CALUDE_total_cows_count_l2267_226728


namespace NUMINAMATH_CALUDE_fraction_of_girls_l2267_226745

theorem fraction_of_girls (total_students : ℕ) (boys : ℕ) (h1 : total_students = 160) (h2 : boys = 120) :
  (total_students - boys : ℚ) / total_students = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_girls_l2267_226745


namespace NUMINAMATH_CALUDE_min_value_expression_l2267_226791

/-- Given positive real numbers a and b satisfying a + 3b = 7, 
    the expression 1/(1+a) + 4/(2+b) has a minimum value of (13 + 4√3)/14 -/
theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : a + 3*b = 7) :
  (1 / (1 + a) + 4 / (2 + b)) ≥ (13 + 4 * Real.sqrt 3) / 14 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 7 ∧
    1 / (1 + a₀) + 4 / (2 + b₀) = (13 + 4 * Real.sqrt 3) / 14 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2267_226791


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2267_226704

-- Define a quadratic function with integer coefficients
def QuadraticFunction (a b c : ℤ) : ℤ → ℤ := fun x ↦ a * x^2 + b * x + c

-- Define the set of possible values for f(0), f(3), and f(4)
def PossibleValues : Set ℤ := {2, 20, 202, 2022}

-- Theorem statement
theorem quadratic_function_property (a b c : ℤ) :
  let f := QuadraticFunction a b c
  (f 0 ∈ PossibleValues) ∧
  (f 3 ∈ PossibleValues) ∧
  (f 4 ∈ PossibleValues) ∧
  (f 0 ≠ f 3) ∧
  (f 0 ≠ f 4) ∧
  (f 3 ≠ f 4) →
  (f 1 = -80) ∨ (f 1 = -990) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2267_226704


namespace NUMINAMATH_CALUDE_train_length_l2267_226799

/-- Given a train with speed 72 km/hr crossing a 250 m long platform in 26 seconds,
    prove that the length of the train is 270 meters. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (1000 / 3600) → 
  platform_length = 250 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 270 := by
  sorry

#eval (72 * (1000 / 3600) * 26) - 250  -- Should output 270

end NUMINAMATH_CALUDE_train_length_l2267_226799


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l2267_226712

theorem complex_equation_solutions :
  (∃ (x y : ℝ), (x + y) + (y - 1) * I = (2 * x + 3 * y) + (2 * y + 1) * I ∧ x = 4 ∧ y = -2) ∧
  (∃ (x y : ℝ), (x + y - 3) + (x - 2) * I = 0 ∧ x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l2267_226712


namespace NUMINAMATH_CALUDE_least_b_with_conditions_l2267_226780

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The least positive integer with a given number of factors -/
def least_with_factors (k : ℕ) : ℕ+ := sorry

theorem least_b_with_conditions (a b : ℕ+) 
  (ha : num_factors a = 4)
  (hb : num_factors b = 2 * (num_factors a))
  (hdiv : b.val % a.val = 0) :
  b ≥ 60 ∧ ∃ (a₀ b₀ : ℕ+), 
    num_factors a₀ = 4 ∧ 
    num_factors b₀ = 2 * (num_factors a₀) ∧ 
    b₀.val % a₀.val = 0 ∧ 
    b₀ = 60 := by sorry

end NUMINAMATH_CALUDE_least_b_with_conditions_l2267_226780


namespace NUMINAMATH_CALUDE_parents_gift_ratio_equal_l2267_226724

/-- Represents the spending on Christmas gifts -/
structure ChristmasGifts where
  sibling_cost : ℕ  -- Cost per sibling's gift
  num_siblings : ℕ  -- Number of siblings
  total_spent : ℕ  -- Total amount spent on all gifts
  parent_cost : ℕ  -- Cost per parent's gift

/-- Theorem stating that the ratio of gift values for Mia's parents is 1:1 -/
theorem parents_gift_ratio_equal (gifts : ChristmasGifts)
  (h1 : gifts.sibling_cost = 30)
  (h2 : gifts.num_siblings = 3)
  (h3 : gifts.total_spent = 150)
  (h4 : gifts.parent_cost = 30) :
  gifts.parent_cost / gifts.parent_cost = 1 := by
  sorry

#check parents_gift_ratio_equal

end NUMINAMATH_CALUDE_parents_gift_ratio_equal_l2267_226724


namespace NUMINAMATH_CALUDE_factor_implies_k_equals_five_l2267_226759

theorem factor_implies_k_equals_five (m k : ℤ) : 
  (∃ (A B : ℤ), m^3 - k*m^2 - 24*m + 16 = (m^2 - 8*m) * (A*m + B)) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_k_equals_five_l2267_226759


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_l2267_226750

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ) :
  square_area = 2025 →
  rectangle_area = 180 →
  rectangle_breadth = 10 →
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_l2267_226750


namespace NUMINAMATH_CALUDE_pet_store_cages_l2267_226770

/-- Calculates the number of cages needed for a given animal type -/
def cages_needed (num_animals : ℕ) (animals_per_cage : ℕ) : ℕ :=
  (num_animals + animals_per_cage - 1) / animals_per_cage

theorem pet_store_cages :
  let num_snakes : ℕ := 4
  let num_parrots : ℕ := 6
  let num_rabbits : ℕ := 8
  let snakes_per_cage : ℕ := 2
  let parrots_per_cage : ℕ := 3
  let rabbits_per_cage : ℕ := 4
  
  (cages_needed num_snakes snakes_per_cage) +
  (cages_needed num_parrots parrots_per_cage) +
  (cages_needed num_rabbits rabbits_per_cage) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2267_226770


namespace NUMINAMATH_CALUDE_theater_bills_count_l2267_226762

-- Define the problem parameters
def total_tickets : ℕ := 300
def ticket_price : ℕ := 40
def total_revenue : ℕ := total_tickets * ticket_price

-- Define the variables for the number of each type of bill
def num_20_bills : ℕ := 238
def num_10_bills : ℕ := 2 * num_20_bills
def num_5_bills : ℕ := num_10_bills + 20

-- Define the theorem
theorem theater_bills_count :
  -- Conditions
  (20 * num_20_bills + 10 * num_10_bills + 5 * num_5_bills = total_revenue) →
  (num_10_bills = 2 * num_20_bills) →
  (num_5_bills = num_10_bills + 20) →
  -- Conclusion
  (num_20_bills + num_10_bills + num_5_bills = 1210) :=
by sorry

end NUMINAMATH_CALUDE_theater_bills_count_l2267_226762


namespace NUMINAMATH_CALUDE_solve_for_k_l2267_226757

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x^3 - 2 * x + 4
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + x^2 - 5 * x - k

-- State the theorem
theorem solve_for_k : ∃ k : ℝ, f 3 - g k 3 = 14 ∧ k = -17 := by sorry

end NUMINAMATH_CALUDE_solve_for_k_l2267_226757


namespace NUMINAMATH_CALUDE_trees_in_yard_l2267_226721

/-- The number of trees in a yard with given length and tree spacing -/
def numberOfTrees (yardLength : ℕ) (treeSpacing : ℕ) : ℕ :=
  (yardLength / treeSpacing) + 1

/-- Theorem: In a 225-meter long yard with trees spaced 10 meters apart, there are 24 trees -/
theorem trees_in_yard : numberOfTrees 225 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l2267_226721


namespace NUMINAMATH_CALUDE_solve_equation_l2267_226760

theorem solve_equation (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2267_226760


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2267_226736

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function g
def g (a b k x : ℝ) : ℝ := f a b x - k * x

-- State the theorem
theorem quadratic_function_properties (a b : ℝ) (ha : a ≠ 0) :
  (f a b (-1) = 0) →
  (∀ x : ℝ, f a b x ≥ 0) →
  (a = 1 ∧ b = 2) ∧
  (∀ k : ℝ, (∀ x ∈ Set.Icc (-2) 2, Monotone (g a b k)) ↔ k ≤ -2 ∨ k ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2267_226736


namespace NUMINAMATH_CALUDE_min_value_theorem_l2267_226753

theorem min_value_theorem (a b c d : ℝ) 
  (h1 : 0 ≤ a ∧ a < 2^(1/4))
  (h2 : 0 ≤ b ∧ b < 2^(1/4))
  (h3 : 0 ≤ c ∧ c < 2^(1/4))
  (h4 : 0 ≤ d ∧ d < 2^(1/4))
  (h5 : a^3 + b^3 + c^3 + d^3 = 2) :
  (a / Real.sqrt (2 - a^4)) + (b / Real.sqrt (2 - b^4)) + 
  (c / Real.sqrt (2 - c^4)) + (d / Real.sqrt (2 - d^4)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2267_226753


namespace NUMINAMATH_CALUDE_orange_cost_l2267_226793

/-- Given the cost of 3 dozen oranges, calculate the cost of 5 dozen oranges at the same rate -/
theorem orange_cost (cost_3_dozen : ℝ) (h : cost_3_dozen = 28.80) :
  let cost_per_dozen := cost_3_dozen / 3
  let cost_5_dozen := 5 * cost_per_dozen
  cost_5_dozen = 48 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_l2267_226793


namespace NUMINAMATH_CALUDE_range_of_g_l2267_226790

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l2267_226790


namespace NUMINAMATH_CALUDE_ellipse_equation_l2267_226787

/-- Given an ellipse with eccentricity √7/4 and distance 4 from one endpoint of the minor axis to the right focus, prove its standard equation is x²/16 + y²/9 = 1 -/
theorem ellipse_equation (e : ℝ) (d : ℝ) (x y : ℝ) :
  e = Real.sqrt 7 / 4 →
  d = 4 →
  x^2 / 16 + y^2 / 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2267_226787


namespace NUMINAMATH_CALUDE_total_players_is_60_l2267_226707

/-- Represents the total number of players in each sport and their intersections --/
structure SportPlayers where
  cricket : ℕ
  hockey : ℕ
  football : ℕ
  softball : ℕ
  cricket_hockey : ℕ
  cricket_football : ℕ
  cricket_softball : ℕ
  hockey_football : ℕ
  hockey_softball : ℕ
  football_softball : ℕ
  cricket_hockey_football : ℕ

/-- Calculate the total number of unique players given the sport participation data --/
def totalUniquePlayers (sp : SportPlayers) : ℕ :=
  sp.cricket + sp.hockey + sp.football + sp.softball
  - sp.cricket_hockey - sp.cricket_football - sp.cricket_softball
  - sp.hockey_football - sp.hockey_softball - sp.football_softball
  + sp.cricket_hockey_football

/-- The main theorem stating that given the specific sport participation data,
    the total number of unique players is 60 --/
theorem total_players_is_60 (sp : SportPlayers)
  (h1 : sp.cricket = 25)
  (h2 : sp.hockey = 20)
  (h3 : sp.football = 30)
  (h4 : sp.softball = 18)
  (h5 : sp.cricket_hockey = 5)
  (h6 : sp.cricket_football = 8)
  (h7 : sp.cricket_softball = 3)
  (h8 : sp.hockey_football = 4)
  (h9 : sp.hockey_softball = 6)
  (h10 : sp.football_softball = 9)
  (h11 : sp.cricket_hockey_football = 2) :
  totalUniquePlayers sp = 60 := by
  sorry


end NUMINAMATH_CALUDE_total_players_is_60_l2267_226707


namespace NUMINAMATH_CALUDE_parabolas_intersection_sum_l2267_226796

/-- The parabolas y = x^2 + 15x + 32 and x = y^2 + 49y + 593 meet at one point (x₀, y₀). -/
theorem parabolas_intersection_sum (x₀ y₀ : ℝ) :
  y₀ = x₀^2 + 15*x₀ + 32 ∧ 
  x₀ = y₀^2 + 49*y₀ + 593 →
  x₀ + y₀ = -33 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_sum_l2267_226796


namespace NUMINAMATH_CALUDE_basketball_probability_l2267_226726

/-- The probability of a basketball player scoring a basket -/
def p : ℚ := 2/3

/-- The number of attempts -/
def n : ℕ := 3

/-- The maximum number of successful baskets we're considering -/
def k : ℕ := 1

/-- The probability of scoring at most once in three attempts -/
def prob_at_most_one : ℚ := 7/27

theorem basketball_probability :
  (Finset.sum (Finset.range (k + 1)) (λ i => Nat.choose n i * p^i * (1 - p)^(n - i))) = prob_at_most_one :=
sorry

end NUMINAMATH_CALUDE_basketball_probability_l2267_226726


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l2267_226731

theorem fourth_root_equation_solutions :
  {x : ℝ | (57 - 2*x)^(1/4) + (45 + 2*x)^(1/4) = 4} = {27, -17} := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l2267_226731


namespace NUMINAMATH_CALUDE_brother_catch_up_l2267_226789

/-- The time it takes for the younger brother to reach school (in minutes) -/
def younger_time : ℝ := 25

/-- The time it takes for the older brother to reach school (in minutes) -/
def older_time : ℝ := 15

/-- The time difference between when the older brother leaves after the younger brother (in minutes) -/
def time_difference : ℝ := 8

/-- The time when the older brother catches up to the younger brother (in minutes after the younger brother leaves) -/
def catch_up_time : ℝ := 20

theorem brother_catch_up :
  let younger_speed := 1 / younger_time
  let older_speed := 1 / older_time
  younger_speed * catch_up_time = older_speed * (catch_up_time - time_difference) := by sorry

end NUMINAMATH_CALUDE_brother_catch_up_l2267_226789


namespace NUMINAMATH_CALUDE_debby_water_bottles_l2267_226767

def water_bottle_problem (initial_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ) : Prop :=
  let days : ℕ := (initial_bottles - remaining_bottles) / bottles_per_day
  days = 1

theorem debby_water_bottles :
  water_bottle_problem 301 144 157 :=
sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l2267_226767


namespace NUMINAMATH_CALUDE_sqrt_equality_l2267_226732

theorem sqrt_equality (a b x : ℝ) (h1 : a < b) (h2 : -b ≤ x) (h3 : x ≤ -a) :
  Real.sqrt (-(x+a)^3*(x+b)) = -(x+a) * Real.sqrt (-(x+a)*(x+b)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equality_l2267_226732


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2267_226715

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v1 v2 : Fin 2 → ℝ) : Prop :=
  v1 0 * v2 1 = v1 1 * v2 0

/-- The problem statement -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  are_parallel (λ i => if i = 0 then 1 else 2) (λ i => if i = 0 then 2*x else -3) →
  x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2267_226715


namespace NUMINAMATH_CALUDE_angle_pcq_is_45_deg_l2267_226794

/-- Given a unit square ABCD with points P on AB and Q on AD forming
    triangle APQ with perimeter 2, angle PCQ is 45 degrees. -/
theorem angle_pcq_is_45_deg (A B C D P Q : ℝ × ℝ) : 
  -- Square ABCD is a unit square
  A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1) →
  -- P is on AB
  ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ P = (a, 0) →
  -- Q is on AD
  ∃ b : ℝ, 0 ≤ b ∧ b ≤ 1 ∧ Q = (0, b) →
  -- Perimeter of APQ is 2
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) +
  Real.sqrt ((A.1 - Q.1)^2 + (A.2 - Q.2)^2) = 2 →
  -- Angle PCQ is 45 degrees
  (Real.arctan ((C.2 - P.2) / (C.1 - P.1)) -
   Real.arctan ((C.1 - Q.1) / (C.2 - Q.2))) * (180 / Real.pi) = 45 := by
  sorry


end NUMINAMATH_CALUDE_angle_pcq_is_45_deg_l2267_226794


namespace NUMINAMATH_CALUDE_percentage_difference_l2267_226781

theorem percentage_difference : 
  (38 / 100 * 80) - (12 / 100 * 160) = 11.2 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l2267_226781


namespace NUMINAMATH_CALUDE_zhang_qiu_jian_gold_distribution_l2267_226730

/-- Represents the gold distribution problem from "Zhang Qiu Jian Suan Jing" -/
theorem zhang_qiu_jian_gold_distribution 
  (n : ℕ) 
  (gold : ℕ → ℚ) 
  (h1 : n = 10) 
  (h2 : ∀ i j, i < j → gold i < gold j) 
  (h3 : gold 8 + gold 9 + gold 10 = 4) 
  (h4 : gold 4 + gold 5 + gold 6 + gold 7 = 3) 
  (h5 : ∀ i j, j = i + 1 → gold j - gold i = gold (i+1) - gold i) :
  gold 5 + gold 6 + gold 7 = 83/26 := by
  sorry

end NUMINAMATH_CALUDE_zhang_qiu_jian_gold_distribution_l2267_226730


namespace NUMINAMATH_CALUDE_common_point_properties_l2267_226714

open Real

noncomputable section

variables (a b : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 + 2*a*x

def g (a b : ℝ) (x : ℝ) : ℝ := 3*a^2 * log x + b

def common_point (a b : ℝ) : Prop :=
  ∃ x > 0, f a x = g a b x ∧ (deriv (f a)) x = (deriv (g a b)) x

theorem common_point_properties (h : a > 0) (h_common : common_point a b) :
  (a = 1 → b = 5/2) ∧
  (b = 5/2 * a^2 - 3*a^2 * log a) ∧
  (b ≤ 3/2 * exp (2/3)) :=
sorry

end

end NUMINAMATH_CALUDE_common_point_properties_l2267_226714


namespace NUMINAMATH_CALUDE_square_sum_value_l2267_226776

theorem square_sum_value (x y : ℝ) 
  (eq1 : y + 9 = 3 * (x - 1)^2)
  (eq2 : x + 9 = 3 * (y - 1)^2)
  (neq : x ≠ y) : 
  x^2 + y^2 = 71/9 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l2267_226776


namespace NUMINAMATH_CALUDE_rhombuses_in_5x5_grid_l2267_226758

/-- Represents a grid of equilateral triangles -/
structure TriangleGrid where
  rows : Nat
  cols : Nat

/-- Counts the number of rhombuses in a triangle grid -/
def count_rhombuses (grid : TriangleGrid) : Nat :=
  sorry

/-- Theorem: In a 5x5 grid of equilateral triangles, there are 30 rhombuses -/
theorem rhombuses_in_5x5_grid :
  let grid : TriangleGrid := { rows := 5, cols := 5 }
  count_rhombuses grid = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombuses_in_5x5_grid_l2267_226758


namespace NUMINAMATH_CALUDE_toms_bowling_score_l2267_226747

theorem toms_bowling_score (tom jerry : ℕ) : 
  tom = jerry + 30 → 
  (tom + jerry) / 2 = 90 → 
  tom = 105 := by
sorry

end NUMINAMATH_CALUDE_toms_bowling_score_l2267_226747


namespace NUMINAMATH_CALUDE_weeks_to_buy_bike_l2267_226792

def mountain_bike_cost : ℕ := 600
def birthday_money : ℕ := 60 + 40 + 20 + 30
def weekly_earnings : ℕ := 18

theorem weeks_to_buy_bike : 
  ∃ (weeks : ℕ), birthday_money + weeks * weekly_earnings = mountain_bike_cost ∧ weeks = 25 :=
by sorry

end NUMINAMATH_CALUDE_weeks_to_buy_bike_l2267_226792


namespace NUMINAMATH_CALUDE_min_value_theorem_l2267_226755

-- Define a positive term geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  is_positive_geometric_sequence a →
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  a 6 = a 5 + 2 * a 4 →
  (∀ k l : ℕ, 1 / k + 4 / l ≥ 3 / 2) ∧
  (∃ k l : ℕ, 1 / k + 4 / l = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2267_226755


namespace NUMINAMATH_CALUDE_locus_of_center_P_l2267_226710

-- Define the circle A
def circle_A (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 100

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Define that B is inside circle A
def B_inside_A : Prop := circle_A (point_B.1) (point_B.2)

-- Define circle P
def circle_P (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  -- P passes through B
  (center.1 - point_B.1)^2 + (center.2 - point_B.2)^2 = radius^2 ∧
  -- P is tangent to A internally
  ((center.1 + 3)^2 + center.2^2)^(1/2) + radius = 10

-- Theorem statement
theorem locus_of_center_P :
  ∀ (x y : ℝ), (∃ (r : ℝ), circle_P (x, y) r) ↔ x^2/25 + y^2/16 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_center_P_l2267_226710


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2267_226719

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x > 1 ∧ x < 2 → x^2 + m*x + 4 < 0) ↔ m ≤ -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2267_226719


namespace NUMINAMATH_CALUDE_no_valid_cube_labeling_l2267_226795

/-- A cube vertex labeling is a function from vertex indices to odd numbers -/
def CubeLabeling := Fin 8 → Nat

/-- Predicate to check if two numbers are adjacent on a cube -/
def adjacent (i j : Fin 8) : Prop :=
  (i.val + j.val) % 2 = 1 ∧ i ≠ j

/-- Predicate to check if a labeling satisfies the problem conditions -/
def validLabeling (f : CubeLabeling) : Prop :=
  (∀ i, 1 ≤ f i ∧ f i ≤ 600 ∧ f i % 2 = 1) ∧
  (∀ i j, adjacent i j → ∃ d > 1, d ∣ f i ∧ d ∣ f j) ∧
  (∀ i j, ¬adjacent i j → ∀ d > 1, ¬(d ∣ f i ∧ d ∣ f j)) ∧
  (∀ i j, i ≠ j → f i ≠ f j)

theorem no_valid_cube_labeling : ¬∃ f : CubeLabeling, validLabeling f :=
sorry

end NUMINAMATH_CALUDE_no_valid_cube_labeling_l2267_226795


namespace NUMINAMATH_CALUDE_fibonacci_gcd_2002_1998_l2267_226739

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => fibonacci (n + 2) + fibonacci (n + 1)

theorem fibonacci_gcd_2002_1998 : Nat.gcd (fibonacci 2002) (fibonacci 1998) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_gcd_2002_1998_l2267_226739


namespace NUMINAMATH_CALUDE_quadratic_function_unique_form_l2267_226765

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

-- Define the conditions
def symmetric_about_negative_one (f : ℝ → ℝ) : Prop :=
  ∀ k : ℝ, f (-1 + k) = f (-1 - k)

def y_intercept_at_one (f : ℝ → ℝ) : Prop :=
  f 0 = 1

def x_axis_intercept_length (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 2 * Real.sqrt 2

-- Main theorem
theorem quadratic_function_unique_form (f : ℝ → ℝ) :
  quadratic_function f →
  symmetric_about_negative_one f →
  y_intercept_at_one f →
  x_axis_intercept_length f →
  ∀ x, f x = -x^2 - 2*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_form_l2267_226765


namespace NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l2267_226756

/-- Represents a right triangle with sides a, b, and c --/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2

/-- Represents the areas of right isosceles triangles constructed on the sides of a right triangle --/
structure IsoscelesTriangleAreas (t : RightTriangle) where
  A : ℝ
  B : ℝ
  C : ℝ
  area_def_A : A = (1/2) * t.a^2
  area_def_B : B = (1/2) * t.b^2
  area_def_C : C = (1/2) * t.c^2

/-- Theorem: For a 5-12-13 right triangle with right isosceles triangles constructed on each side,
    the sum of the areas of the isosceles triangles on the two shorter sides
    equals the area of the isosceles triangle on the hypotenuse --/
theorem isosceles_triangle_areas_sum (t : RightTriangle)
  (h : t.a = 5 ∧ t.b = 12 ∧ t.c = 13)
  (areas : IsoscelesTriangleAreas t) :
  areas.A + areas.B = areas.C := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l2267_226756


namespace NUMINAMATH_CALUDE_negative_two_inequality_l2267_226738

theorem negative_two_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_inequality_l2267_226738


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_l2267_226771

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 10} := by sorry

-- Theorem for (ᶜA) ∩ B
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_l2267_226771


namespace NUMINAMATH_CALUDE_element_in_set_l2267_226751

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l2267_226751
