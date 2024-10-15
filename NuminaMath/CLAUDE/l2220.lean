import Mathlib

namespace NUMINAMATH_CALUDE_fraction_transformation_l2220_222013

theorem fraction_transformation (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  (a^3 : ℚ) / (b + 3) = 2 * (a / b) → a = 2 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2220_222013


namespace NUMINAMATH_CALUDE_cosine_equation_solution_l2220_222076

theorem cosine_equation_solution (x : ℝ) : 
  (1 + Real.cos (3 * x) = 2 * Real.cos (2 * x)) ↔ 
  (∃ k : ℤ, x = 2 * k * Real.pi ∨ x = Real.pi / 6 + k * Real.pi ∨ x = 5 * Real.pi / 6 + k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_cosine_equation_solution_l2220_222076


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2220_222090

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (2 * x^3))^2) = x^3 / 2 + 1 / (2 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2220_222090


namespace NUMINAMATH_CALUDE_mushroom_picking_profit_l2220_222088

/-- Calculates the money made on the first day of a three-day mushroom picking trip -/
theorem mushroom_picking_profit (total_mushrooms day2_mushrooms price_per_mushroom : ℕ) : 
  total_mushrooms = 65 →
  day2_mushrooms = 12 →
  price_per_mushroom = 2 →
  (total_mushrooms - day2_mushrooms - 2 * day2_mushrooms) * price_per_mushroom = 58 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_picking_profit_l2220_222088


namespace NUMINAMATH_CALUDE_polynomial_division_l2220_222044

theorem polynomial_division (x : ℝ) (h : x ≠ 0) : 2 * x^3 / x^2 = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l2220_222044


namespace NUMINAMATH_CALUDE_x_value_when_one_in_set_l2220_222035

theorem x_value_when_one_in_set (x : ℝ) : 1 ∈ ({x, x^2} : Set ℝ) → x ≠ x^2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_one_in_set_l2220_222035


namespace NUMINAMATH_CALUDE_second_day_sales_correct_l2220_222023

/-- Represents the sales of sportswear in a clothing store over two days -/
structure SportswearSales where
  first_day : ℕ
  second_day : ℕ

/-- Calculates the sales on the second day based on the first day's sales -/
def second_day_sales (m : ℕ) : ℕ := 2 * m - 3

/-- Theorem stating the relationship between first and second day sales -/
theorem second_day_sales_correct (sales : SportswearSales) :
  sales.first_day = m →
  sales.second_day = 2 * sales.first_day - 3 →
  sales.second_day = second_day_sales m :=
by sorry

end NUMINAMATH_CALUDE_second_day_sales_correct_l2220_222023


namespace NUMINAMATH_CALUDE_georges_walk_speed_l2220_222094

/-- Proves that given the conditions, George must walk at 6 mph for the last segment to arrive on time -/
theorem georges_walk_speed (total_distance : Real) (normal_speed : Real) (first_half_distance : Real) (first_half_speed : Real) :
  total_distance = 1.5 →
  normal_speed = 3 →
  first_half_distance = 0.75 →
  first_half_speed = 2 →
  (total_distance / normal_speed - first_half_distance / first_half_speed) / (total_distance - first_half_distance) = 6 := by
  sorry


end NUMINAMATH_CALUDE_georges_walk_speed_l2220_222094


namespace NUMINAMATH_CALUDE_car_hire_problem_l2220_222095

theorem car_hire_problem (total_cost : ℝ) (a_hours c_hours : ℝ) (b_cost : ℝ) :
  total_cost = 520 →
  a_hours = 7 →
  c_hours = 11 →
  b_cost = 160 →
  ∃ b_hours : ℝ,
    b_cost = (total_cost / (a_hours + b_hours + c_hours)) * b_hours ∧
    b_hours = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_hire_problem_l2220_222095


namespace NUMINAMATH_CALUDE_function_ordering_l2220_222058

/-- A function f is even with respect to x = -1 -/
def IsEvenShifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 1) = f (-x - 1)

/-- The function f is strictly decreasing in terms of its values when x > -1 -/
def IsStrictlyDecreasingShifted (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, -1 < x₁ → x₁ < x₂ → (f x₂ - f x₁) * (x₂ - x₁) < 0

theorem function_ordering (f : ℝ → ℝ) 
    (h1 : IsEvenShifted f) 
    (h2 : IsStrictlyDecreasingShifted f) : 
    f (-2) < f 1 ∧ f 1 < f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_ordering_l2220_222058


namespace NUMINAMATH_CALUDE_square_difference_area_l2220_222047

theorem square_difference_area (a b : ℝ) : 
  (a + b)^2 - a^2 = 2*a*b + b^2 := by sorry

end NUMINAMATH_CALUDE_square_difference_area_l2220_222047


namespace NUMINAMATH_CALUDE_insurance_percentage_l2220_222008

theorem insurance_percentage (salary tax_rate utility_rate remaining_amount : ℝ) 
  (h1 : salary = 2000)
  (h2 : tax_rate = 0.2)
  (h3 : utility_rate = 0.25)
  (h4 : remaining_amount = 1125)
  (h5 : ∃ insurance_rate : ℝ, 
    remaining_amount = salary * (1 - tax_rate - insurance_rate) * (1 - utility_rate)) :
  ∃ insurance_rate : ℝ, insurance_rate = 0.05 := by
sorry

end NUMINAMATH_CALUDE_insurance_percentage_l2220_222008


namespace NUMINAMATH_CALUDE_data_set_range_l2220_222030

/-- The range of a data set with maximum value 78 and minimum value 21 is 57. -/
theorem data_set_range : ℝ → ℝ → ℝ → Prop :=
  fun (max min range : ℝ) =>
    max = 78 ∧ min = 21 → range = max - min → range = 57

/-- Proof of the theorem -/
lemma prove_data_set_range : data_set_range 78 21 57 := by
  sorry

end NUMINAMATH_CALUDE_data_set_range_l2220_222030


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2220_222075

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : 
  x^3 - 1/x^3 = 125 := by sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2220_222075


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l2220_222029

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the speed of the man rowing upstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the man's speed in still water and downstream,
    his upstream speed is 20 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed)
  (h1 : s.stillWater = 24)
  (h2 : s.downstream = 28) :
  upstreamSpeed s = 20 := by
sorry

#eval upstreamSpeed { stillWater := 24, downstream := 28 }

end NUMINAMATH_CALUDE_upstream_speed_calculation_l2220_222029


namespace NUMINAMATH_CALUDE_shadow_length_sams_shadow_length_l2220_222024

/-- Given a lamp post and a person walking towards it, this theorem calculates
    the length of the person's shadow at a new position. -/
theorem shadow_length (lamp_height : ℝ) (initial_distance : ℝ) (initial_shadow : ℝ) 
                      (new_distance : ℝ) : ℝ :=
  let person_height := lamp_height * initial_shadow / (initial_distance + initial_shadow)
  let new_shadow := person_height * new_distance / (lamp_height - person_height)
  new_shadow

/-- The main theorem that proves the specific shadow length for the given scenario. -/
theorem sams_shadow_length : 
  shadow_length 8 12 4 8 = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_shadow_length_sams_shadow_length_l2220_222024


namespace NUMINAMATH_CALUDE_johns_age_multiple_l2220_222038

/-- Given the ages and relationships described in the problem, prove that John's age 3 years ago
    was twice James' age 6 years from now. -/
theorem johns_age_multiple (john_current_age james_brother_age james_brother_age_diff : ℕ)
  (h1 : john_current_age = 39)
  (h2 : james_brother_age = 16)
  (h3 : james_brother_age_diff = 4) : 
  (john_current_age - 3) = 2 * (james_brother_age - james_brother_age_diff + 6) := by
  sorry

end NUMINAMATH_CALUDE_johns_age_multiple_l2220_222038


namespace NUMINAMATH_CALUDE_percentage_difference_l2220_222037

theorem percentage_difference : (150 * 62 / 100) - (250 * 20 / 100) = 43 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2220_222037


namespace NUMINAMATH_CALUDE_triangle_area_l2220_222061

/-- Given a triangle PQR with inradius r, circumradius R, and angles P, Q, R satisfying certain conditions,
    prove that its area is (7√3201)/3 -/
theorem triangle_area (P Q R : ℝ) (r R : ℝ) (h1 : r = 7) (h2 : R = 25) 
    (h3 : 2 * Real.cos Q = Real.cos P + Real.cos R) : 
    ∃ (area : ℝ), area = (7 * Real.sqrt 3201) / 3 ∧ 
    area = r * (P + Q + R) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l2220_222061


namespace NUMINAMATH_CALUDE_area_smaller_circle_l2220_222045

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  r : ℝ  -- radius of the smaller circle
  R : ℝ  -- radius of the larger circle
  h_positive : 0 < r
  h_tangent : R = 2 * r
  h_common_tangent : ∃ (P A B : ℝ × ℝ), 
    let d := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
    d = 5 ∧ d = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

/-- The area of the smaller circle in a TangentCircles configuration is 25π/8 -/
theorem area_smaller_circle (tc : TangentCircles) : 
  π * tc.r^2 = 25 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_area_smaller_circle_l2220_222045


namespace NUMINAMATH_CALUDE_probability_two_painted_faces_is_three_eighths_l2220_222027

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  total_small_cubes : ℕ
  small_cubes_with_two_painted_faces : ℕ

/-- The probability of selecting a small cube with exactly two painted faces -/
def probability_two_painted_faces (c : CutCube) : ℚ :=
  c.small_cubes_with_two_painted_faces / c.total_small_cubes

/-- A cube cut into 64 smaller cubes -/
def cube_64 : CutCube :=
  { total_small_cubes := 64,
    small_cubes_with_two_painted_faces := 24 }

theorem probability_two_painted_faces_is_three_eighths :
  probability_two_painted_faces cube_64 = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_painted_faces_is_three_eighths_l2220_222027


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l2220_222093

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of enchanted stones available. -/
def num_stones : ℕ := 6

/-- The number of stones incompatible with a specific herb. -/
def incompatible_stones : ℕ := 3

/-- The number of herbs that have incompatible stones. -/
def herbs_with_incompatibility : ℕ := 1

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_stones - incompatible_stones * herbs_with_incompatibility

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l2220_222093


namespace NUMINAMATH_CALUDE_merchant_salt_problem_l2220_222065

theorem merchant_salt_problem (x : ℝ) : 
  (x > 0) →
  (x + 100 > x) →
  (x + 220 > x + 100) →
  (x / (x + 100) = (x + 100) / (x + 220)) →
  (x = 500) :=
by
  sorry

end NUMINAMATH_CALUDE_merchant_salt_problem_l2220_222065


namespace NUMINAMATH_CALUDE_total_worth_of_toys_l2220_222042

theorem total_worth_of_toys (total_toys : Nat) (special_toy_value : Nat) (regular_toy_value : Nat) :
  total_toys = 9 →
  special_toy_value = 12 →
  regular_toy_value = 5 →
  (total_toys - 1) * regular_toy_value + special_toy_value = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_worth_of_toys_l2220_222042


namespace NUMINAMATH_CALUDE_intersection_points_count_l2220_222036

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem intersection_points_count 
  (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_def : ∀ x ∈ Set.Icc 0 2, f x = x^3 - x) :
  (Set.Icc 0 6 ∩ {x | f x = 0}).ncard = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l2220_222036


namespace NUMINAMATH_CALUDE_num_large_beds_is_two_l2220_222040

/-- The number of seeds that can be planted in a large bed -/
def large_bed_capacity : ℕ := 100

/-- The number of seeds that can be planted in a medium bed -/
def medium_bed_capacity : ℕ := 60

/-- The number of medium beds -/
def num_medium_beds : ℕ := 2

/-- The total number of seeds that can be planted -/
def total_seeds : ℕ := 320

/-- Theorem stating that the number of large beds is 2 -/
theorem num_large_beds_is_two :
  ∃ (n : ℕ), n * large_bed_capacity + num_medium_beds * medium_bed_capacity = total_seeds ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_num_large_beds_is_two_l2220_222040


namespace NUMINAMATH_CALUDE_complementary_angles_theorem_l2220_222079

theorem complementary_angles_theorem (α β : Real) : 
  (α + β = 180) →  -- complementary angles
  (α - β / 2 = 30) →  -- half of β is 30° less than α
  (α = 80) :=  -- measure of α is 80°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_theorem_l2220_222079


namespace NUMINAMATH_CALUDE_rope_length_l2220_222089

/-- Given a rope cut into two parts with a ratio of 2:3, where the shorter part is 16 meters long,
    the total length of the rope is 40 meters. -/
theorem rope_length (shorter_part : ℝ) (ratio_short : ℝ) (ratio_long : ℝ) :
  shorter_part = 16 →
  ratio_short = 2 →
  ratio_long = 3 →
  (shorter_part / ratio_short) * (ratio_short + ratio_long) = 40 :=
by sorry

end NUMINAMATH_CALUDE_rope_length_l2220_222089


namespace NUMINAMATH_CALUDE_min_amount_spent_l2220_222014

/-- Represents the price of a volleyball in yuan -/
def volleyball_price : ℝ := 80

/-- Represents the price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := 100

/-- Represents the total number of balls to be purchased -/
def total_balls : ℕ := 50

/-- Represents the minimum number of soccer balls to be purchased -/
def min_soccer_balls : ℕ := 25

/-- Theorem stating the minimum amount spent on purchasing the balls -/
theorem min_amount_spent :
  let x := min_soccer_balls
  let y := total_balls - x
  x * soccer_ball_price + y * volleyball_price = 4500 ∧
  x ≥ y ∧
  500 / soccer_ball_price = 400 / volleyball_price ∧
  soccer_ball_price = volleyball_price + 20 := by
  sorry


end NUMINAMATH_CALUDE_min_amount_spent_l2220_222014


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2220_222063

theorem batsman_average_increase (total_runs : ℕ → ℕ) (innings : ℕ) :
  innings = 17 →
  total_runs innings = total_runs (innings - 1) + 74 →
  (total_runs innings : ℚ) / innings = 26 →
  (total_runs innings : ℚ) / innings - (total_runs (innings - 1) : ℚ) / (innings - 1) = 3 := by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l2220_222063


namespace NUMINAMATH_CALUDE_smallest_a_with_50_squares_l2220_222046

theorem smallest_a_with_50_squares : ∃ (a : ℕ), 
  (a = 4486) ∧ 
  (∀ k : ℕ, k < a → (∃ (n : ℕ), n * n > k ∧ n * n < 3 * k) → 
    (∃ (m : ℕ), m < 50)) ∧
  (∃ (l : ℕ), l = 50 ∧ 
    (∀ i : ℕ, i ≤ l → ∃ (s : ℕ), s * s > a ∧ s * s < 3 * a)) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_with_50_squares_l2220_222046


namespace NUMINAMATH_CALUDE_power_negative_product_l2220_222091

theorem power_negative_product (a : ℝ) : (-a)^3 * (-a)^5 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_power_negative_product_l2220_222091


namespace NUMINAMATH_CALUDE_donny_piggy_bank_money_l2220_222001

def initial_money (kite_price frisbee_price money_left : ℕ) : ℕ :=
  kite_price + frisbee_price + money_left

theorem donny_piggy_bank_money :
  initial_money 8 9 61 = 78 := by
  sorry

end NUMINAMATH_CALUDE_donny_piggy_bank_money_l2220_222001


namespace NUMINAMATH_CALUDE_min_value_of_f_l2220_222028

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := x^2 + 3*y^2 + 8*x - 6*y + x*y + 22

/-- Theorem stating that the minimum value of f is 3 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = 3 ∧ ∀ (x y : ℝ), f x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2220_222028


namespace NUMINAMATH_CALUDE_circle_radius_determines_m_l2220_222026

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def is_circle_equation (h k r m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 - 2*h*x - 2*k*y + (h^2 + k^2 - r^2 + m) = 0

theorem circle_radius_determines_m :
  ∀ m : ℝ, (∃ h k : ℝ, is_circle_equation h k 2 m) → m = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_determines_m_l2220_222026


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l2220_222018

theorem complex_number_imaginary_part (z : ℂ) (h : (1 + z) / (1 - z) = I) : z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l2220_222018


namespace NUMINAMATH_CALUDE_fraction_simplification_l2220_222032

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) (h3 : x ≠ 0) :
  (3 * x / (x - 2) - x / (x + 2)) * ((x^2 - 4) / x) = 2 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2220_222032


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2220_222083

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_3 + a_7 = 37, the sum a_2 + a_4 + a_6 + a_8 = 74. -/
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : 
  a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2220_222083


namespace NUMINAMATH_CALUDE_smallest_four_divisors_sum_of_squares_l2220_222052

theorem smallest_four_divisors_sum_of_squares (n : ℕ+) 
  (d1 d2 d3 d4 : ℕ+) 
  (h_div : ∀ m : ℕ+, m ∣ n → m ≥ d1 ∧ m ≥ d2 ∧ m ≥ d3 ∧ m ≥ d4)
  (h_order : d1 < d2 ∧ d2 < d3 ∧ d3 < d4)
  (h_sum : n = d1^2 + d2^2 + d3^2 + d4^2) : 
  n = 130 := by
sorry

end NUMINAMATH_CALUDE_smallest_four_divisors_sum_of_squares_l2220_222052


namespace NUMINAMATH_CALUDE_square_perimeter_l2220_222059

theorem square_perimeter (s : ℝ) (h : s = 13) : 4 * s = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2220_222059


namespace NUMINAMATH_CALUDE_quadratic_sum_bound_l2220_222077

/-- Represents a quadratic function of the form y = x^2 - (a+2)x + 2a + 1 -/
def QuadraticFunction (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - (a + 2) * x + 2 * a + 1

/-- Theorem: For a quadratic function passing through (-1, y₀) where y₀ is the minimum,
    any two different points A(m, n) and B(2-m, p) on the parabola satisfy n + p > -8 -/
theorem quadratic_sum_bound
  (a : ℝ)
  (y₀ : ℝ)
  (h1 : QuadraticFunction a (-1) = y₀)
  (h2 : ∀ x y, y = QuadraticFunction a x → y ≥ y₀)
  (m n p : ℝ)
  (h3 : n = QuadraticFunction a m)
  (h4 : p = QuadraticFunction a (2 - m))
  (h5 : m ≠ 2 - m) :
  n + p > -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_bound_l2220_222077


namespace NUMINAMATH_CALUDE_percentage_difference_l2220_222011

theorem percentage_difference : (80 / 100 * 60) - (4 / 5 * 25) = 28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2220_222011


namespace NUMINAMATH_CALUDE_min_value_expression_l2220_222041

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → 4 * x^2 + y^2 + 1 / (x * y) ≥ 17 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2220_222041


namespace NUMINAMATH_CALUDE_different_answers_for_fedya_question_l2220_222051

-- Define the types of people
inductive Person : Type
| truthTeller : Person
| liar : Person

-- Define the possible answers
inductive Answer : Type
| yes : Answer
| no : Answer

-- Define the function that determines how a person answers
def answerQuestion (p : Person) (isNameFedya : Bool) : Answer :=
  match p with
  | Person.truthTeller => if isNameFedya then Answer.yes else Answer.no
  | Person.liar => if isNameFedya then Answer.no else Answer.yes

-- State the theorem
theorem different_answers_for_fedya_question 
  (fedya : Person) 
  (vadim : Person) 
  (h1 : fedya = Person.truthTeller) 
  (h2 : vadim = Person.liar) :
  answerQuestion fedya true ≠ answerQuestion vadim false :=
sorry

end NUMINAMATH_CALUDE_different_answers_for_fedya_question_l2220_222051


namespace NUMINAMATH_CALUDE_solution_set_eq_neg_one_one_l2220_222074

-- Define the solution set of x^2 - 1 = 0
def solution_set : Set ℝ := {x : ℝ | x^2 - 1 = 0}

-- Theorem stating that the solution set is exactly {-1, 1}
theorem solution_set_eq_neg_one_one : solution_set = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_eq_neg_one_one_l2220_222074


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2220_222071

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  2 / a + 3 / b ≥ 5 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2220_222071


namespace NUMINAMATH_CALUDE_property_necessary_not_sufficient_l2220_222085

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Definition of an arithmetic sequence -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property a₃ + a₇ = 2a₅ -/
def property (a : Sequence) : Prop :=
  a 3 + a 7 = 2 * a 5

/-- The main theorem stating that the property is necessary but not sufficient -/
theorem property_necessary_not_sufficient :
  (∀ a : Sequence, is_arithmetic a → property a) ∧
  (∃ a : Sequence, ¬is_arithmetic a ∧ property a) :=
sorry

end NUMINAMATH_CALUDE_property_necessary_not_sufficient_l2220_222085


namespace NUMINAMATH_CALUDE_f_less_than_g_for_x_greater_than_one_l2220_222062

/-- Given functions f and g with specified properties, f(x) < g(x) for x > 1 -/
theorem f_less_than_g_for_x_greater_than_one 
  (f g : ℝ → ℝ)
  (h_f : ∀ x, f x = Real.log x)
  (h_g : ∃ a b : ℝ, ∀ x, g x = a * x + b / x)
  (h_common_tangent : ∃ x₀, x₀ > 0 ∧ f x₀ = g x₀ ∧ (deriv f) x₀ = (deriv g) x₀)
  (x : ℝ)
  (h_x : x > 1) :
  f x < g x := by
sorry

end NUMINAMATH_CALUDE_f_less_than_g_for_x_greater_than_one_l2220_222062


namespace NUMINAMATH_CALUDE_exponential_function_at_zero_l2220_222020

theorem exponential_function_at_zero (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  (fun x : ℝ => a^x) 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_at_zero_l2220_222020


namespace NUMINAMATH_CALUDE_new_eurasian_bridge_length_scientific_notation_l2220_222098

theorem new_eurasian_bridge_length_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 10900 = a * (10 : ℝ) ^ n ∧ a = 1.09 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_new_eurasian_bridge_length_scientific_notation_l2220_222098


namespace NUMINAMATH_CALUDE_flowerbed_perimeter_sum_l2220_222053

/-- Calculates the perimeter of a rectangle given its width and length -/
def rectanglePerimeter (width length : ℝ) : ℝ := 2 * (width + length)

/-- Proves that the total perimeter of three flowerbeds with given dimensions is 69 meters -/
theorem flowerbed_perimeter_sum : 
  let flowerbed1_width : ℝ := 4
  let flowerbed1_length : ℝ := 2 * flowerbed1_width - 1
  let flowerbed2_length : ℝ := flowerbed1_length + 3
  let flowerbed2_width : ℝ := flowerbed1_width - 2
  let flowerbed3_width : ℝ := (flowerbed1_width + flowerbed2_width) / 2
  let flowerbed3_length : ℝ := (flowerbed1_length + flowerbed2_length) / 2
  rectanglePerimeter flowerbed1_width flowerbed1_length +
  rectanglePerimeter flowerbed2_width flowerbed2_length +
  rectanglePerimeter flowerbed3_width flowerbed3_length = 69 := by
  sorry


end NUMINAMATH_CALUDE_flowerbed_perimeter_sum_l2220_222053


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2220_222007

open Set

def A : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_of_A_and_B : A ∪ B = Ioc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2220_222007


namespace NUMINAMATH_CALUDE_scientific_notation_of_31400000_l2220_222056

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_31400000 :
  toScientificNotation 31400000 = ScientificNotation.mk 3.14 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_31400000_l2220_222056


namespace NUMINAMATH_CALUDE_decimal_expansion_3_11_l2220_222006

theorem decimal_expansion_3_11 : 
  ∃ (n : ℕ) (a b : ℕ), 
    (3 : ℚ) / 11 = (a : ℚ) / (10^n - 1) ∧ 
    b = 10^n - 1 ∧ 
    n = 2 ∧
    a < b := by sorry

end NUMINAMATH_CALUDE_decimal_expansion_3_11_l2220_222006


namespace NUMINAMATH_CALUDE_negation_of_implication_l2220_222015

theorem negation_of_implication :
  ¬(x = 1 → x^2 = 1) ↔ (x = 1 → x^2 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2220_222015


namespace NUMINAMATH_CALUDE_jane_ate_12_swirls_l2220_222081

/-- Given a number of cinnamon swirls and people, calculate how many swirls each person ate. -/
def swirls_per_person (total_swirls : ℕ) (num_people : ℕ) : ℕ :=
  total_swirls / num_people

/-- Theorem stating that Jane ate 12 cinnamon swirls. -/
theorem jane_ate_12_swirls (total_swirls : ℕ) (num_people : ℕ) 
  (h1 : total_swirls = 120) 
  (h2 : num_people = 10) :
  swirls_per_person total_swirls num_people = 12 := by
  sorry

#eval swirls_per_person 120 10

end NUMINAMATH_CALUDE_jane_ate_12_swirls_l2220_222081


namespace NUMINAMATH_CALUDE_ten_team_round_robin_l2220_222099

def roundRobinGames (n : ℕ) : ℕ := n * (n - 1) / 2

theorem ten_team_round_robin :
  roundRobinGames 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_team_round_robin_l2220_222099


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2220_222000

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - x^3)^3 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2220_222000


namespace NUMINAMATH_CALUDE_martha_cards_l2220_222064

/-- The number of cards Martha initially had -/
def initial_cards : ℕ := 3

/-- The number of cards Martha received from Emily -/
def cards_from_emily : ℕ := 76

/-- The total number of cards Martha ended up with -/
def total_cards : ℕ := 79

/-- Theorem stating that the initial number of cards plus the cards received equals the total cards -/
theorem martha_cards : initial_cards + cards_from_emily = total_cards := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l2220_222064


namespace NUMINAMATH_CALUDE_total_supermarkets_l2220_222025

def FGH_chain (us canada : ℕ) : Prop :=
  (us = 49) ∧ (us = canada + 14)

theorem total_supermarkets (us canada : ℕ) (h : FGH_chain us canada) : 
  us + canada = 84 :=
sorry

end NUMINAMATH_CALUDE_total_supermarkets_l2220_222025


namespace NUMINAMATH_CALUDE_tan_alpha_minus_2beta_l2220_222021

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2/5)
  (h2 : Real.tan β = 1/2) : 
  Real.tan (α - 2*β) = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_2beta_l2220_222021


namespace NUMINAMATH_CALUDE_place_value_ratio_l2220_222086

def number : ℚ := 86572.4908

theorem place_value_ratio : 
  ∃ (tens hundredths : ℚ), 
    (tens = 10) ∧ 
    (hundredths = 0.01) ∧ 
    (tens / hundredths = 1000) :=
by sorry

end NUMINAMATH_CALUDE_place_value_ratio_l2220_222086


namespace NUMINAMATH_CALUDE_triangle_cosine_b_l2220_222003

theorem triangle_cosine_b (ω : ℝ) (A B C a b c : ℝ) :
  ω > 0 →
  (∀ x, 2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) - 2 * Real.sin (ω * x / 2) ^ 2 =
        2 * Real.sin (2 * x / 3 + π / 6) - 1) →
  a < b →
  b < c →
  Real.sqrt 3 * a = 2 * c * Real.sin A →
  2 * Real.sin (A + π / 2) - 1 = 11 / 13 →
  Real.cos B = (5 * Real.sqrt 3 + 12) / 26 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_b_l2220_222003


namespace NUMINAMATH_CALUDE_unique_four_digit_square_with_repeated_digits_l2220_222080

theorem unique_four_digit_square_with_repeated_digits : 
  ∃! n : ℕ, 
    1000 ≤ n ∧ n ≤ 9999 ∧ 
    (∃ m : ℕ, n = m^2) ∧
    (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = 1100 * a + 11 * b) ∧
    n = 7744 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_with_repeated_digits_l2220_222080


namespace NUMINAMATH_CALUDE_circle_from_equation_l2220_222034

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in general form -/
def CircleEquation (x y : ℝ) (A B C D E : ℝ) : Prop :=
  A * x^2 + B * x + C * y^2 + D * y + E = 0

theorem circle_from_equation :
  ∃ (c : Circle), 
    (∀ (x y : ℝ), CircleEquation x y 1 (-6) 1 2 (-12) ↔ 
      (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
    c.center = (3, -1) ∧
    c.radius = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_circle_from_equation_l2220_222034


namespace NUMINAMATH_CALUDE_unique_abcabc_cube_minus_square_l2220_222050

/-- A number is of the form abcabc if it equals 1001 * (100a + 10b + c) for some digits a, b, c -/
def is_abcabc (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 1001 * (100 * a + 10 * b + c)

/-- The main theorem stating that 78 is the unique positive integer x 
    such that x^3 - x^2 is a six-digit number of the form abcabc -/
theorem unique_abcabc_cube_minus_square :
  ∃! (x : ℕ), x > 0 ∧ 100000 ≤ x^3 - x^2 ∧ x^3 - x^2 < 1000000 ∧ is_abcabc (x^3 - x^2) ∧ x = 78 :=
sorry

end NUMINAMATH_CALUDE_unique_abcabc_cube_minus_square_l2220_222050


namespace NUMINAMATH_CALUDE_no_primes_of_form_l2220_222033

theorem no_primes_of_form (m : ℕ) (hm : m > 0) : 
  ¬ Prime (2^(5*m) + 2^m + 1) := by
sorry

end NUMINAMATH_CALUDE_no_primes_of_form_l2220_222033


namespace NUMINAMATH_CALUDE_cost_per_topping_is_two_l2220_222019

/-- Represents the cost of a pizza order with toppings and tip --/
def pizza_order_cost (large_pizza_cost : ℝ) (num_pizzas : ℕ) (toppings_per_pizza : ℕ) 
  (tip_percentage : ℝ) (topping_cost : ℝ) : ℝ :=
  let base_cost := large_pizza_cost * num_pizzas
  let total_toppings := num_pizzas * toppings_per_pizza
  let toppings_cost := total_toppings * topping_cost
  let subtotal := base_cost + toppings_cost
  let tip := subtotal * tip_percentage
  subtotal + tip

/-- The cost per topping is $2 --/
theorem cost_per_topping_is_two :
  ∃ (topping_cost : ℝ),
    pizza_order_cost 14 2 3 0.25 topping_cost = 50 ∧ 
    topping_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_topping_is_two_l2220_222019


namespace NUMINAMATH_CALUDE_min_sum_xyz_l2220_222097

theorem min_sum_xyz (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) :
  ∀ a b c : ℤ, (a - 10) * (b - 5) * (c - 2) = 1000 → x + y + z ≤ a + b + c ∧ x + y + z = 92 :=
sorry

end NUMINAMATH_CALUDE_min_sum_xyz_l2220_222097


namespace NUMINAMATH_CALUDE_consecutive_squares_not_equal_consecutive_fourth_powers_l2220_222049

theorem consecutive_squares_not_equal_consecutive_fourth_powers :
  ∀ x y : ℕ+, x^2 + (x + 1)^2 ≠ y^4 + (y + 1)^4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_squares_not_equal_consecutive_fourth_powers_l2220_222049


namespace NUMINAMATH_CALUDE_jungkook_total_sheets_l2220_222039

/-- The number of sheets in a bundle of colored paper -/
def sheets_per_bundle : ℕ := 10

/-- The number of bundles Jungkook has -/
def bundles : ℕ := 3

/-- The number of additional individual sheets Jungkook has -/
def individual_sheets : ℕ := 8

/-- Theorem stating the total number of sheets Jungkook has -/
theorem jungkook_total_sheets :
  bundles * sheets_per_bundle + individual_sheets = 38 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_total_sheets_l2220_222039


namespace NUMINAMATH_CALUDE_certain_value_problem_l2220_222054

theorem certain_value_problem (n v : ℝ) : n = 10 → (1/2) * n + v = 11 → v = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_problem_l2220_222054


namespace NUMINAMATH_CALUDE_smallest_multiple_of_twelve_power_l2220_222002

theorem smallest_multiple_of_twelve_power (k : ℕ) : 
  (3^k - k^3 = 1) → (∀ n : ℕ, n > 0 ∧ 12^k ∣ n → n ≥ 144) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_twelve_power_l2220_222002


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_l2220_222031

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_l2220_222031


namespace NUMINAMATH_CALUDE_ratio_equality_implies_fraction_value_l2220_222084

theorem ratio_equality_implies_fraction_value 
  (x y z : ℝ) 
  (h : x / 3 = y / 5 ∧ y / 5 = z / 7) : 
  (y + z) / (3 * x - y) = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_implies_fraction_value_l2220_222084


namespace NUMINAMATH_CALUDE_geometric_progression_quadratic_vertex_l2220_222043

/-- Given a geometric progression a, b, c, d and a quadratic function,
    prove that ad = 3 --/
theorem geometric_progression_quadratic_vertex (a b c d : ℝ) :
  (∃ (r : ℝ), b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric progression condition
  (2 * b^2 - 4 * b + 5 = c) →                      -- vertex condition
  a * d = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_quadratic_vertex_l2220_222043


namespace NUMINAMATH_CALUDE_simplify_expression_l2220_222004

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a + b) :
  a / b + b / a - 1 / (a * b) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l2220_222004


namespace NUMINAMATH_CALUDE_abc_sum_l2220_222067

theorem abc_sum (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a * b = 2 * (a + b)) (hbc : b * c = 3 * (b + c)) (hca : c * a = 4 * (c + a)) :
  a + b + c = 1128 / 35 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_l2220_222067


namespace NUMINAMATH_CALUDE_irwin_score_product_l2220_222048

/-- Represents the types of baskets in Jamshid and Irwin's basketball game -/
inductive BasketType
  | Two
  | Five
  | Eleven
  | Thirteen

/-- Returns the point value of a given basket type -/
def basketValue (b : BasketType) : ℕ :=
  match b with
  | BasketType.Two => 2
  | BasketType.Five => 5
  | BasketType.Eleven => 11
  | BasketType.Thirteen => 13

/-- Irwin's score at halftime -/
def irwinScore : ℕ := 2 * basketValue BasketType.Eleven

theorem irwin_score_product : irwinScore = 22 := by
  sorry

end NUMINAMATH_CALUDE_irwin_score_product_l2220_222048


namespace NUMINAMATH_CALUDE_f_sum_negative_l2220_222078

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_property_1 : ∀ x : ℝ, f (-x) = -f (x + 4)
axiom f_property_2 : ∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 2 → f x₁ > f x₂

-- Define the theorem
theorem f_sum_negative (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ < 4) 
  (h2 : (x₁ - 2) * (x₂ - 2) < 0) : 
  f x₁ + f x₂ < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_negative_l2220_222078


namespace NUMINAMATH_CALUDE_max_pieces_on_chessboard_l2220_222087

/-- Represents a chessboard configuration -/
def ChessboardConfiguration := Fin 8 → Fin 8 → Bool

/-- Checks if a given position is on the board -/
def isOnBoard (row col : ℕ) : Prop := row < 8 ∧ col < 8

/-- Checks if a piece is placed at a given position -/
def hasPiece (config : ChessboardConfiguration) (row col : Fin 8) : Prop :=
  config row col = true

/-- Counts the number of pieces on a given diagonal -/
def piecesOnDiagonal (config : ChessboardConfiguration) (startRow startCol : Fin 8) (rowStep colStep : Int) : ℕ :=
  sorry

/-- Checks if the configuration is valid (no more than 3 pieces on any diagonal) -/
def isValidConfiguration (config : ChessboardConfiguration) : Prop :=
  ∀ (startRow startCol : Fin 8) (rowStep colStep : Int),
    piecesOnDiagonal config startRow startCol rowStep colStep ≤ 3

/-- Counts the total number of pieces on the board -/
def totalPieces (config : ChessboardConfiguration) : ℕ :=
  sorry

/-- The main theorem -/
theorem max_pieces_on_chessboard :
  ∃ (config : ChessboardConfiguration),
    isValidConfiguration config ∧
    totalPieces config = 38 ∧
    ∀ (otherConfig : ChessboardConfiguration),
      isValidConfiguration otherConfig →
      totalPieces otherConfig ≤ 38 :=
  sorry

end NUMINAMATH_CALUDE_max_pieces_on_chessboard_l2220_222087


namespace NUMINAMATH_CALUDE_evaluate_expression_l2220_222057

theorem evaluate_expression : (2301 - 2222)^2 / 144 = 43 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2220_222057


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2220_222009

-- Define the quadratic polynomial
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, quadratic a b c x < 0) :
  b / a < c / a + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2220_222009


namespace NUMINAMATH_CALUDE_triangle_properties_l2220_222092

/-- Represents a triangle with sides x, y, and z -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if the triangle satisfies the given conditions -/
def satisfiesConditions (t : Triangle) (a : ℝ) : Prop :=
  t.x + t.y = 3 * t.z ∧
  t.z + t.y = t.x + a ∧
  t.x + t.z = 60

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties :
  ∀ (t : Triangle) (a : ℝ),
    satisfiesConditions t a →
    (0 < a ∧ a < 60) ∧
    (a = 30 → t.x = 35 ∧ t.y = 40 ∧ t.z = 25) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2220_222092


namespace NUMINAMATH_CALUDE_max_value_at_zero_l2220_222066

/-- The function f(x) = x³ - 3x² + 1 reaches its maximum value at x = 0 -/
theorem max_value_at_zero (f : ℝ → ℝ) (h : f = λ x => x^3 - 3*x^2 + 1) :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≤ f x₀ ∧ x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_at_zero_l2220_222066


namespace NUMINAMATH_CALUDE_function_defined_for_all_reals_l2220_222017

/-- The function f(t) is defined for all real numbers t. -/
theorem function_defined_for_all_reals :
  ∀ t : ℝ, ∃ y : ℝ, y = 1 / ((t - 1)^2 + (t + 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_function_defined_for_all_reals_l2220_222017


namespace NUMINAMATH_CALUDE_base_seven_addition_problem_l2220_222055

/-- Given a base 7 addition problem 3XY₇ + 52₇ = 42X₇, prove that X + Y = 6 in base 10 -/
theorem base_seven_addition_problem (X Y : Fin 7) :
  (3 * 7 * 7 + X * 7 + Y) + (5 * 7 + 2) = 4 * 7 * 7 + 2 * 7 + X →
  (X : ℕ) + (Y : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_addition_problem_l2220_222055


namespace NUMINAMATH_CALUDE_salary_calculation_correct_l2220_222082

/-- Calculates the salary after three months of increases -/
def salary_after_three_months (initial_salary : ℝ) (first_month_increase : ℝ) : ℝ :=
  let month1 := initial_salary * (1 + first_month_increase)
  let month2 := month1 * (1 + 2 * first_month_increase)
  let month3 := month2 * (1 + 4 * first_month_increase)
  month3

/-- Theorem stating that the salary after three months matches the expected value -/
theorem salary_calculation_correct : 
  salary_after_three_months 2000 0.05 = 2772 := by
  sorry


end NUMINAMATH_CALUDE_salary_calculation_correct_l2220_222082


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l2220_222072

theorem triangle_is_right_angled (a b c : ℝ) : 
  a = 3 ∧ b = 5 ∧ (3 * c^2 - 10 * c = 8) ∧ c > 0 → 
  a^2 + c^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angled_l2220_222072


namespace NUMINAMATH_CALUDE_larger_segment_is_50_l2220_222096

/-- Represents a triangle with sides a, b, c and an altitude h dropped on side c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  x : ℝ  -- shorter segment of side c
  valid_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  altitude_property : a^2 = x^2 + h^2 ∧ b^2 = (c - x)^2 + h^2

/-- The larger segment of the side c in a triangle with sides 40, 50, 90 is 50 --/
theorem larger_segment_is_50 :
  ∀ t : Triangle, t.a = 40 ∧ t.b = 50 ∧ t.c = 90 → (t.c - t.x = 50) :=
by sorry

end NUMINAMATH_CALUDE_larger_segment_is_50_l2220_222096


namespace NUMINAMATH_CALUDE_bad_carrots_count_l2220_222010

/-- The number of bad carrots in Vanessa's garden -/
def bad_carrots (vanessa_carrots mother_carrots good_carrots : ℕ) : ℕ :=
  vanessa_carrots + mother_carrots - good_carrots

theorem bad_carrots_count : bad_carrots 17 14 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l2220_222010


namespace NUMINAMATH_CALUDE_fraction_denominator_l2220_222070

theorem fraction_denominator (n : ℕ) (d : ℕ) :
  n = 35 →
  (n : ℚ) / d = 2 / 10^20 →
  d = 175 * 10^20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l2220_222070


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2220_222068

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def digit_sum (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℕ := (n / 1000) - (n % 10)

theorem unique_four_digit_number : 
  ∃! n : ℕ, 
    is_four_digit n ∧ 
    digit_sum n = 16 ∧ 
    middle_digits_sum n = 10 ∧ 
    thousands_minus_units n = 2 ∧ 
    n % 11 = 0 ∧
    n = 4642 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l2220_222068


namespace NUMINAMATH_CALUDE_variance_of_transformed_binomial_l2220_222022

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- The variance of a linear transformation of a random variable -/
def varianceLinearTransform (a b : ℝ) (v : ℝ) : ℝ := a^2 * v

theorem variance_of_transformed_binomial :
  let ξ : BinomialDistribution := ⟨100, 0.3, by norm_num⟩
  varianceLinearTransform 3 (-5) (variance ξ) = 189 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_transformed_binomial_l2220_222022


namespace NUMINAMATH_CALUDE_number_of_students_l2220_222012

theorem number_of_students (N : ℕ) : 
  (N : ℚ) * 15 = 4 * 14 + 10 * 16 + 9 → N = 15 := by
  sorry

#check number_of_students

end NUMINAMATH_CALUDE_number_of_students_l2220_222012


namespace NUMINAMATH_CALUDE_triangle_max_area_l2220_222073

/-- Given a triangle ABC with side lengths a, b, c opposite angles A, B, C respectively,
    prove that the maximum area is √3 under the given conditions. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  (Real.sin A - Real.sin B) / Real.sin C = (c - b) / (2 + b) →
  (∀ a' b' c' A' B' C',
    a' = 2 →
    (Real.sin A' - Real.sin B') / Real.sin C' = (c' - b') / (2 + b') →
    (1/2) * a' * b' * Real.sin C' ≤ Real.sqrt 3) ∧
  (∃ b' c',
    (Real.sin A - Real.sin B) / Real.sin C = (c' - b') / (2 + b') →
    (1/2) * a * b' * Real.sin C = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2220_222073


namespace NUMINAMATH_CALUDE_largest_five_digit_multiple_largest_five_digit_multiple_exists_l2220_222069

theorem largest_five_digit_multiple (n : Nat) : n ≤ 99999 ∧ n ≥ 10000 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≤ 99936 :=
by
  sorry

theorem largest_five_digit_multiple_exists : ∃ n : Nat, n = 99936 ∧ n ≤ 99999 ∧ n ≥ 10000 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_multiple_largest_five_digit_multiple_exists_l2220_222069


namespace NUMINAMATH_CALUDE_arthur_actual_weight_l2220_222016

/-- The weight shown on the scales when weighing King Arthur -/
def arthur_scale : ℕ := 19

/-- The weight shown on the scales when weighing the royal horse -/
def horse_scale : ℕ := 101

/-- The weight shown on the scales when weighing King Arthur and the horse together -/
def combined_scale : ℕ := 114

/-- The actual weight of King Arthur -/
def arthur_weight : ℕ := 13

/-- The consistent error of the scales -/
def scale_error : ℕ := 6

theorem arthur_actual_weight :
  arthur_weight + scale_error = arthur_scale ∧
  arthur_weight + (horse_scale - scale_error) + scale_error = combined_scale :=
sorry

end NUMINAMATH_CALUDE_arthur_actual_weight_l2220_222016


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2220_222060

-- Problem 1
theorem problem_1 : 4.7 + (-2.5) - (-5.3) - 7.5 = 0 := by sorry

-- Problem 2
theorem problem_2 : 18 + 48 / (-2)^2 - (-4)^2 * 5 = -50 := by sorry

-- Problem 3
theorem problem_3 : -1^4 + (-2)^2 / 4 * (5 - (-3)^2) = -5 := by sorry

-- Problem 4
theorem problem_4 : (-19 - 15/16) * 8 = -159 - 1/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2220_222060


namespace NUMINAMATH_CALUDE_yanna_afternoon_biscuits_l2220_222005

/-- The number of butter cookies Yanna baked in the afternoon -/
def afternoon_butter_cookies : ℕ := 10

/-- The difference between biscuits and butter cookies baked in the afternoon -/
def biscuit_cookie_difference : ℕ := 30

/-- The number of biscuits Yanna baked in the afternoon -/
def afternoon_biscuits : ℕ := afternoon_butter_cookies + biscuit_cookie_difference

theorem yanna_afternoon_biscuits : afternoon_biscuits = 40 := by
  sorry

end NUMINAMATH_CALUDE_yanna_afternoon_biscuits_l2220_222005
