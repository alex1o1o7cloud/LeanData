import Mathlib

namespace NUMINAMATH_CALUDE_minRainfallDay4_is_21_l3453_345372

/-- Represents the rainfall data and conditions for a 4-day storm --/
structure RainfallData where
  capacity : ℝ  -- Area capacity in inches
  drain_rate : ℝ  -- Daily drainage rate in inches
  day1_rain : ℝ  -- Rainfall on day 1 in inches
  day2_rain : ℝ  -- Rainfall on day 2 in inches
  day3_rain : ℝ  -- Rainfall on day 3 in inches

/-- Calculates the minimum rainfall on day 4 for flooding to occur --/
def minRainfallDay4 (data : RainfallData) : ℝ :=
  data.capacity - (data.day1_rain + data.day2_rain + data.day3_rain - 3 * data.drain_rate)

/-- Theorem stating the minimum rainfall on day 4 for flooding --/
theorem minRainfallDay4_is_21 (data : RainfallData) :
  data.capacity = 72 ∧
  data.drain_rate = 3 ∧
  data.day1_rain = 10 ∧
  data.day2_rain = 2 * data.day1_rain ∧
  data.day3_rain = 1.5 * data.day2_rain →
  minRainfallDay4 data = 21 := by
  sorry

#eval minRainfallDay4 {
  capacity := 72,
  drain_rate := 3,
  day1_rain := 10,
  day2_rain := 20,
  day3_rain := 30
}

end NUMINAMATH_CALUDE_minRainfallDay4_is_21_l3453_345372


namespace NUMINAMATH_CALUDE_equation_solution_l3453_345314

theorem equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∀ x : ℝ, (a * Real.sin x + b) / (b * Real.cos x + a) = (a * Real.cos x + b) / (b * Real.sin x + a) ↔
  ∃ k : ℤ, x = π/4 + π * k :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3453_345314


namespace NUMINAMATH_CALUDE_probability_one_or_two_first_20_rows_l3453_345359

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (rows : ℕ) : Type := Unit

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := if n ≥ 1 then 2 * n - 1 else 0

/-- The number of 2's in the first n rows of Pascal's Triangle -/
def countTwos (n : ℕ) : ℕ := if n ≥ 3 then 2 * (n - 2) else 0

/-- The probability of selecting either 1 or 2 from the first n rows of Pascal's Triangle -/
def probabilityOneOrTwo (n : ℕ) : ℚ :=
  (countOnes n + countTwos n : ℚ) / (totalElements n : ℚ)

theorem probability_one_or_two_first_20_rows :
  probabilityOneOrTwo 20 = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_probability_one_or_two_first_20_rows_l3453_345359


namespace NUMINAMATH_CALUDE_expression_evaluation_l3453_345307

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  (x^4 + 1) / x^2 * (y^4 + 1) / y^2 - (x^4 - 1) / y^2 * (y^4 - 1) / x^2 = 2 * x^2 / y^2 + 2 * y^2 / x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3453_345307


namespace NUMINAMATH_CALUDE_katya_magic_pen_problem_l3453_345390

theorem katya_magic_pen_problem (total_problems : ℕ) 
  (katya_prob : ℚ) (pen_prob : ℚ) (min_correct : ℕ) :
  total_problems = 20 →
  katya_prob = 4/5 →
  pen_prob = 1/2 →
  min_correct = 13 →
  ∃ (x : ℕ), x ≥ 10 ∧ 
    (x : ℚ) * katya_prob + (total_problems - x : ℚ) * pen_prob ≥ min_correct :=
by sorry

end NUMINAMATH_CALUDE_katya_magic_pen_problem_l3453_345390


namespace NUMINAMATH_CALUDE_multiple_properties_l3453_345350

-- Define the properties of c and d
def is_multiple_of_4 (n : ℤ) : Prop := ∃ k : ℤ, n = 4 * k
def is_multiple_of_8 (n : ℤ) : Prop := ∃ k : ℤ, n = 8 * k

-- Define the theorem
theorem multiple_properties (c d : ℤ) 
  (hc : is_multiple_of_4 c) (hd : is_multiple_of_8 d) : 
  (is_multiple_of_4 d) ∧ 
  (is_multiple_of_4 (c + d)) ∧ 
  (∃ k : ℤ, c + d = 2 * k) := by
  sorry


end NUMINAMATH_CALUDE_multiple_properties_l3453_345350


namespace NUMINAMATH_CALUDE_f_max_value_f_no_real_roots_l3453_345343

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 10

-- Theorem for the maximum value
theorem f_max_value :
  ∃ (x_max : ℝ), f x_max = -2 ∧ ∀ (x : ℝ), f x ≤ f x_max ∧ x_max = 2 :=
sorry

-- Theorem for no real roots
theorem f_no_real_roots :
  ∀ (x : ℝ), f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_f_max_value_f_no_real_roots_l3453_345343


namespace NUMINAMATH_CALUDE_jimmy_garden_servings_l3453_345309

/-- The number of servings produced by a carrot plant -/
def carrot_servings : ℕ := 4

/-- The number of green bean plants -/
def green_bean_plants : ℕ := 10

/-- The number of carrot plants -/
def carrot_plants : ℕ := 8

/-- The number of corn plants -/
def corn_plants : ℕ := 12

/-- The number of tomato plants -/
def tomato_plants : ℕ := 15

/-- The number of servings produced by a corn plant -/
def corn_servings : ℕ := 5 * carrot_servings

/-- The number of servings produced by a green bean plant -/
def green_bean_servings : ℕ := corn_servings / 2

/-- The number of servings produced by a tomato plant -/
def tomato_servings : ℕ := carrot_servings + 3

/-- The total number of servings in Jimmy's garden -/
def total_servings : ℕ :=
  green_bean_plants * green_bean_servings +
  carrot_plants * carrot_servings +
  corn_plants * corn_servings +
  tomato_plants * tomato_servings

theorem jimmy_garden_servings :
  total_servings = 477 := by sorry

end NUMINAMATH_CALUDE_jimmy_garden_servings_l3453_345309


namespace NUMINAMATH_CALUDE_equal_distance_point_exists_l3453_345371

-- Define the plane
variable (Plane : Type)

-- Define points on the plane
variable (P Q O A : Plane)

-- Define the speed
variable (v : ℝ)

-- Define the distance function
variable (dist : Plane → Plane → ℝ)

-- Define the time
variable (t : ℝ)

-- Define the lines as functions of time
variable (line_P line_Q : ℝ → Plane)

-- State the theorem
theorem equal_distance_point_exists :
  (∀ t, dist O (line_P t) = v * t) →  -- P moves with constant speed v
  (∀ t, dist O (line_Q t) = v * t) →  -- Q moves with constant speed v
  (∃ t₀, line_P t₀ = O ∧ line_Q t₀ = O) →  -- The lines intersect at O
  ∃ A : Plane, ∀ t, dist A (line_P t) = dist A (line_Q t) :=
by sorry

end NUMINAMATH_CALUDE_equal_distance_point_exists_l3453_345371


namespace NUMINAMATH_CALUDE_railroad_grade_reduction_l3453_345399

theorem railroad_grade_reduction (rise : ℝ) (initial_grade : ℝ) (reduced_grade : ℝ) :
  rise = 800 →
  initial_grade = 0.04 →
  reduced_grade = 0.03 →
  ⌊(rise / reduced_grade - rise / initial_grade)⌋ = 6667 := by
  sorry

end NUMINAMATH_CALUDE_railroad_grade_reduction_l3453_345399


namespace NUMINAMATH_CALUDE_custom_mult_three_four_l3453_345341

/-- Custom multiplication operation -/
def custom_mult (a b : ℤ) : ℤ := 4*a + 3*b - a*b

/-- Theorem stating that 3 * 4 = 12 under the custom multiplication -/
theorem custom_mult_three_four : custom_mult 3 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_three_four_l3453_345341


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l3453_345361

theorem square_rectangle_area_relation :
  ∃ (x₁ x₂ : ℝ),
    (x₁ - 2) * (x₁ + 5) = 3 * (x₁ - 3)^2 ∧
    (x₂ - 2) * (x₂ + 5) = 3 * (x₂ - 3)^2 ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l3453_345361


namespace NUMINAMATH_CALUDE_divisibility_quotient_l3453_345366

theorem divisibility_quotient (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h_div : (a * b) ∣ (a^2 + b^2 + 1)) : 
  (a^2 + b^2 + 1) / (a * b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_quotient_l3453_345366


namespace NUMINAMATH_CALUDE_tan_45_degrees_l3453_345381

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l3453_345381


namespace NUMINAMATH_CALUDE_f_of_five_eq_six_elevenths_l3453_345354

/-- Given a function f(x) = (x+1) / (3x-4), prove that f(5) = 6/11 -/
theorem f_of_five_eq_six_elevenths :
  let f : ℝ → ℝ := λ x ↦ (x + 1) / (3 * x - 4)
  f 5 = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_f_of_five_eq_six_elevenths_l3453_345354


namespace NUMINAMATH_CALUDE_set_A_representation_l3453_345382

def A : Set (ℤ × ℤ) := {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_representation : A = {(-1, 0), (0, -1), (1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_set_A_representation_l3453_345382


namespace NUMINAMATH_CALUDE_symmetry_about_x_axis_periodicity_symmetry_about_origin_l3453_345355

-- Define a real-valued function on reals
variable (f : ℝ → ℝ)

-- Statement 1
theorem symmetry_about_x_axis (x : ℝ) : 
  f (-1 - x) = f (-(x - 1)) := by sorry

-- Statement 2
theorem periodicity (x : ℝ) : 
  f (1 + x) = f (x - 1) → f (x + 2) = f x := by sorry

-- Statement 3
theorem symmetry_about_origin (x : ℝ) : 
  f (1 - x) = -f (x - 1) → f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_symmetry_about_x_axis_periodicity_symmetry_about_origin_l3453_345355


namespace NUMINAMATH_CALUDE_fixed_point_exists_P_on_parabola_l3453_345313

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through two points
def line_through (P Q : Point) (x y : ℝ) : Prop :=
  (y - P.y) * (Q.x - P.x) = (x - P.x) * (Q.y - P.y)

-- Define the external angle bisector of ∠APB
def external_angle_bisector (A P B : Point) (x y : ℝ) : Prop :=
  sorry -- Definition of external angle bisector

-- Define a tangent line to the parabola
def tangent_to_parabola (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = 2*(x - x₀)/y₀

-- Main theorem
theorem fixed_point_exists : ∃ (Q : Point), 
  ∀ (A B : Point), 
    parabola A.x A.y → 
    parabola B.x B.y → 
    line_through Q A A.x A.y → 
    line_through Q B B.x B.y → 
    ∃ (T : Point), 
      parabola T.x T.y ∧ 
      external_angle_bisector A P B T.x T.y ∧ 
      tangent_to_parabola T.x T.y T.x T.y :=
by
  -- Proof goes here
  sorry

-- Specific point P satisfies the parabola equation
theorem P_on_parabola : parabola 1 (-2) :=
by
  -- Proof goes here
  sorry

-- Q is the fixed point
def Q : Point := ⟨-3, 2⟩

end NUMINAMATH_CALUDE_fixed_point_exists_P_on_parabola_l3453_345313


namespace NUMINAMATH_CALUDE_max_profit_at_zero_optimal_investment_l3453_345365

/-- Profit function --/
def profit (m : ℝ) : ℝ := 28 - 3 * m

/-- Theorem: The profit function achieves its maximum when m = 0, given m ≥ 0 --/
theorem max_profit_at_zero (m : ℝ) (h : m ≥ 0) : profit 0 ≥ profit m := by
  sorry

/-- Corollary: The optimal investment for maximum profit is 0 --/
theorem optimal_investment : ∃ (m : ℝ), m = 0 ∧ ∀ (n : ℝ), n ≥ 0 → profit m ≥ profit n := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_zero_optimal_investment_l3453_345365


namespace NUMINAMATH_CALUDE_three_equidistant_points_l3453_345333

/-- A color type with two possible values -/
inductive Color
| red
| blue

/-- A point on a straight line -/
structure Point where
  x : ℝ

/-- A coloring function that assigns a color to each point on the line -/
def Coloring := Point → Color

/-- The distance between two points -/
def distance (p q : Point) : ℝ := |p.x - q.x|

theorem three_equidistant_points (c : Coloring) :
  ∃ (A B C : Point), c A = c B ∧ c B = c C ∧ distance A B = distance B C :=
sorry

end NUMINAMATH_CALUDE_three_equidistant_points_l3453_345333


namespace NUMINAMATH_CALUDE_log_product_reciprocal_l3453_345362

theorem log_product_reciprocal (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) :
  Real.log a / Real.log b * (Real.log b / Real.log a) = 1 :=
by sorry

end NUMINAMATH_CALUDE_log_product_reciprocal_l3453_345362


namespace NUMINAMATH_CALUDE_coupon1_best_discount_l3453_345323

/-- Represents the discount offered by Coupon 1 -/
def coupon1_discount (x : ℝ) : ℝ := 0.15 * x

/-- Represents the discount offered by Coupon 2 -/
def coupon2_discount : ℝ := 30

/-- Represents the discount offered by Coupon 3 -/
def coupon3_discount (x : ℝ) : ℝ := 0.22 * (x - 150)

/-- Theorem stating the condition for Coupon 1 to offer the greatest discount -/
theorem coupon1_best_discount (x : ℝ) :
  (coupon1_discount x > coupon2_discount ∧ coupon1_discount x > coupon3_discount x) ↔
  (200 < x ∧ x < 471.43) :=
sorry

end NUMINAMATH_CALUDE_coupon1_best_discount_l3453_345323


namespace NUMINAMATH_CALUDE_division_problem_l3453_345363

theorem division_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 729 ∧ quotient = 19 ∧ remainder = 7 →
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 38 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3453_345363


namespace NUMINAMATH_CALUDE_cubic_factorization_l3453_345356

theorem cubic_factorization (x : ℝ) : x^3 - 16*x = x*(x+4)*(x-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3453_345356


namespace NUMINAMATH_CALUDE_simplify_expression_l3453_345332

theorem simplify_expression : 0.2 * 0.4 + 0.6 * 0.8 = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3453_345332


namespace NUMINAMATH_CALUDE_tim_kittens_count_tim_final_kitten_count_l3453_345391

theorem tim_kittens_count : ℕ → ℕ → ℕ → ℕ
  | initial_kittens, sara_kittens, adoption_rate =>
    let kittens_after_jessica := initial_kittens - initial_kittens / 3
    let kittens_before_adoption := kittens_after_jessica + sara_kittens
    let adopted_kittens := sara_kittens * adoption_rate / 100
    kittens_before_adoption - adopted_kittens

theorem tim_final_kitten_count :
  tim_kittens_count 12 14 50 = 15 := by
  sorry

end NUMINAMATH_CALUDE_tim_kittens_count_tim_final_kitten_count_l3453_345391


namespace NUMINAMATH_CALUDE_bankers_gain_calculation_l3453_345378

/-- Banker's gain calculation -/
theorem bankers_gain_calculation (present_worth : ℝ) (interest_rate : ℝ) (time_period : ℕ) : 
  present_worth = 400 →
  interest_rate = 0.1 →
  time_period = 3 →
  (present_worth * (1 + interest_rate) ^ time_period - present_worth) = 132.4 := by
  sorry

#check bankers_gain_calculation

end NUMINAMATH_CALUDE_bankers_gain_calculation_l3453_345378


namespace NUMINAMATH_CALUDE_tangent_sum_over_cosine_l3453_345373

theorem tangent_sum_over_cosine (x : Real) :
  let a := x * π / 180  -- Convert degrees to radians
  (Real.tan a + Real.tan (2*a) + Real.tan (7*a) + Real.tan (8*a)) / Real.cos a = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_over_cosine_l3453_345373


namespace NUMINAMATH_CALUDE_negation_equivalence_l3453_345310

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, ∃ n : ℕ+, n ≥ x^2) ↔ (∃ x : ℝ, ∀ n : ℕ+, n < x^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3453_345310


namespace NUMINAMATH_CALUDE_opposite_implies_sum_l3453_345311

theorem opposite_implies_sum (x : ℝ) : 
  (3 - x) = -2 → x + 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_implies_sum_l3453_345311


namespace NUMINAMATH_CALUDE_dinner_time_l3453_345322

theorem dinner_time (total_time homework_time cleaning_time trash_time dishwasher_time : ℕ)
  (h1 : total_time = 120)
  (h2 : homework_time = 30)
  (h3 : cleaning_time = 30)
  (h4 : trash_time = 5)
  (h5 : dishwasher_time = 10) :
  total_time - (homework_time + cleaning_time + trash_time + dishwasher_time) = 45 := by
  sorry

end NUMINAMATH_CALUDE_dinner_time_l3453_345322


namespace NUMINAMATH_CALUDE_cube_root_of_number_with_given_square_roots_l3453_345344

theorem cube_root_of_number_with_given_square_roots (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (3*a + 1)^2 = x ∧ (a + 11)^2 = x) →
  ∃ (y : ℝ), y^3 = x ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_number_with_given_square_roots_l3453_345344


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3453_345305

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → a ∈ Set.Iio (-2) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3453_345305


namespace NUMINAMATH_CALUDE_bailey_towel_cost_l3453_345318

/-- Calculates the total cost of towel sets after discount -/
def towel_cost_after_discount (guest_sets : ℕ) (master_sets : ℕ) 
                               (guest_price : ℚ) (master_price : ℚ) 
                               (discount_percent : ℚ) : ℚ :=
  let total_cost := guest_sets * guest_price + master_sets * master_price
  let discount_amount := discount_percent * total_cost
  total_cost - discount_amount

/-- Theorem stating that Bailey's total cost for towel sets is $224.00 -/
theorem bailey_towel_cost :
  towel_cost_after_discount 2 4 40 50 (20 / 100) = 224 :=
by sorry

end NUMINAMATH_CALUDE_bailey_towel_cost_l3453_345318


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3453_345386

theorem unique_three_digit_number : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  (∃ (π b γ : ℕ),
    π ≠ b ∧ π ≠ γ ∧ b ≠ γ ∧
    0 ≤ π ∧ π ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ γ ∧ γ ≤ 9 ∧
    n = 100 * π + 10 * b + γ ∧
    n = (π + b + γ) * (π + b + γ + 1)) ∧
  n = 156 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3453_345386


namespace NUMINAMATH_CALUDE_increasing_function_a_bound_l3453_345320

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 1

-- State the theorem
theorem increasing_function_a_bound (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 1 3 → y ∈ Set.Icc 1 3 → x < y → f a x < f a y) →
  a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_a_bound_l3453_345320


namespace NUMINAMATH_CALUDE_principal_calculation_l3453_345319

/-- Proves that given specific conditions, the principal amount is 1200 --/
theorem principal_calculation (rate : ℝ) (time : ℝ) (amount : ℝ) :
  rate = 0.05 →
  time = 2 + 2 / 5 →
  amount = 1344 →
  amount = (1200 : ℝ) * (1 + rate * time) :=
by sorry

end NUMINAMATH_CALUDE_principal_calculation_l3453_345319


namespace NUMINAMATH_CALUDE_rongcheng_sample_points_l3453_345396

/-- Represents the number of observation points in each county -/
structure ObservationPoints where
  xiongxian : ℕ
  rongcheng : ℕ
  anxin : ℕ

/-- Checks if three numbers form an arithmetic sequence -/
def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- Checks if three numbers form a geometric sequence -/
def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Calculates the number of data points for stratified sampling -/
def stratified_sample (total_samples : ℕ) (points : ObservationPoints) (county : ℕ) : ℕ :=
  (county * total_samples) / (points.xiongxian + points.rongcheng + points.anxin)

theorem rongcheng_sample_points :
  ∀ (points : ObservationPoints),
    points.xiongxian = 6 →
    is_arithmetic_sequence points.xiongxian points.rongcheng points.anxin →
    is_geometric_sequence points.xiongxian points.rongcheng (points.anxin + 6) →
    stratified_sample 12 points points.rongcheng = 4 :=
by sorry

end NUMINAMATH_CALUDE_rongcheng_sample_points_l3453_345396


namespace NUMINAMATH_CALUDE_cubic_equations_common_root_l3453_345306

/-- Given real numbers a, b, c, if every pair of equations from 
    x³ - ax² + b = 0, x³ - bx² + c = 0, x³ - cx² + a = 0 has a common root, 
    then a = b = c. -/
theorem cubic_equations_common_root (a b c : ℝ) 
  (h1 : ∃ x : ℝ, x^3 - a*x^2 + b = 0 ∧ x^3 - b*x^2 + c = 0)
  (h2 : ∃ x : ℝ, x^3 - b*x^2 + c = 0 ∧ x^3 - c*x^2 + a = 0)
  (h3 : ∃ x : ℝ, x^3 - c*x^2 + a = 0 ∧ x^3 - a*x^2 + b = 0) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_cubic_equations_common_root_l3453_345306


namespace NUMINAMATH_CALUDE_abs_negative_two_l3453_345397

theorem abs_negative_two : |(-2 : ℤ)| = 2 := by sorry

end NUMINAMATH_CALUDE_abs_negative_two_l3453_345397


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3453_345304

/-- Prove that in a geometric sequence with first term 1 and fourth term 64, the common ratio is 4 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence condition
  a 1 = 1 →                     -- First term is 1
  a 4 = 64 →                    -- Fourth term is 64
  q = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3453_345304


namespace NUMINAMATH_CALUDE_singers_and_dancers_selection_l3453_345330

/-- Represents the number of ways to select singers and dancers from a group -/
def select_singers_and_dancers (total : ℕ) (singers : ℕ) (dancers : ℕ) : ℕ :=
  let both := singers + dancers - total
  let only_singers := singers - both
  let only_dancers := dancers - both
  (only_singers * only_dancers) +
  (both * (only_singers + only_dancers)) +
  (both * (both - 1))

/-- Theorem stating that for 9 people with 7 singers and 5 dancers, 
    there are 32 ways to select 2 people and assign one to sing and one to dance -/
theorem singers_and_dancers_selection :
  select_singers_and_dancers 9 7 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_singers_and_dancers_selection_l3453_345330


namespace NUMINAMATH_CALUDE_system_solution_l3453_345364

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 8 - 2*x)
  (eq2 : x + z = 10 - 2*y)
  (eq3 : x + y = 14 - 2*z) :
  2*x + 2*y + 2*z = 16 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3453_345364


namespace NUMINAMATH_CALUDE_closest_point_is_vertex_l3453_345349

/-- Given a parabola y² = -2x and a point A(m, 0), if the point on the parabola 
closest to A is the vertex of the parabola, then m ∈ [-1, +∞). -/
theorem closest_point_is_vertex (m : ℝ) : 
  (∀ x y : ℝ, y^2 = -2*x → 
    (∀ x' y' : ℝ, y'^2 = -2*x' → (x' - m)^2 + y'^2 ≥ (x - m)^2 + y^2) → 
    x = 0 ∧ y = 0) → 
  m ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_closest_point_is_vertex_l3453_345349


namespace NUMINAMATH_CALUDE_quadratic_inequality_impossibility_l3453_345374

/-- Given a quadratic function f(x) = ax^2 + 2ax + 1 where a ≠ 0,
    it is impossible for f(-2) > f(-1) > f(0) to be true. -/
theorem quadratic_inequality_impossibility (a : ℝ) (h : a ≠ 0) :
  ¬∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 + 2 * a * x + 1) ∧ 
  (f (-2) > f (-1) ∧ f (-1) > f 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_impossibility_l3453_345374


namespace NUMINAMATH_CALUDE_main_theorem_l3453_345398

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, Differentiable ℝ f ∧ f x + (deriv^[2] f) x > 0

/-- The main theorem -/
theorem main_theorem (f : ℝ → ℝ) (hf : satisfies_condition f) :
  ∀ a b : ℝ, a > b ↔ Real.exp a * f a > Real.exp b * f b :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l3453_345398


namespace NUMINAMATH_CALUDE_train_speed_theorem_l3453_345342

theorem train_speed_theorem (passing_pole_time passing_train_time stationary_train_length : ℝ) 
  (h1 : passing_pole_time = 12)
  (h2 : passing_train_time = 27)
  (h3 : stationary_train_length = 300) :
  let train_length := (passing_train_time * stationary_train_length) / (passing_train_time - passing_pole_time)
  let train_speed := train_length / passing_pole_time
  train_speed = 20 := by sorry

end NUMINAMATH_CALUDE_train_speed_theorem_l3453_345342


namespace NUMINAMATH_CALUDE_diesel_cost_per_gallon_l3453_345346

/-- The cost of diesel fuel per gallon, given weekly spending and bi-weekly usage -/
theorem diesel_cost_per_gallon 
  (weekly_spending : ℝ) 
  (biweekly_usage : ℝ) 
  (h1 : weekly_spending = 36) 
  (h2 : biweekly_usage = 24) : 
  weekly_spending * 2 / biweekly_usage = 3 := by
sorry

end NUMINAMATH_CALUDE_diesel_cost_per_gallon_l3453_345346


namespace NUMINAMATH_CALUDE_satisfactory_fraction_is_four_fifths_l3453_345334

/-- Represents the distribution of grades in a classroom --/
structure GradeDistribution where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  f : ℕ

/-- Calculates the fraction of satisfactory grades --/
def satisfactoryFraction (g : GradeDistribution) : ℚ :=
  let satisfactory := g.a + g.b + g.c + g.d
  let total := satisfactory + g.f
  satisfactory / total

/-- Theorem stating that for the given grade distribution, 
    the fraction of satisfactory grades is 4/5 --/
theorem satisfactory_fraction_is_four_fifths :
  let g : GradeDistribution := ⟨8, 7, 5, 4, 6⟩
  satisfactoryFraction g = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_satisfactory_fraction_is_four_fifths_l3453_345334


namespace NUMINAMATH_CALUDE_average_leaves_per_hour_l3453_345380

/-- Represents the leaf fall pattern of a tree over 3 hours -/
structure TreeLeafFall where
  hour1 : ℕ
  hour2 : ℕ
  hour3 : ℕ

/-- Calculates the total number of leaves that fell from a tree -/
def totalLeaves (tree : TreeLeafFall) : ℕ :=
  tree.hour1 + tree.hour2 + tree.hour3

/-- Represents the leaf fall patterns of two trees in Rylee's backyard -/
def ryleesBackyard : (TreeLeafFall × TreeLeafFall) :=
  (⟨7, 12, 9⟩, ⟨4, 4, 6⟩)

/-- The number of hours of observation -/
def observationHours : ℕ := 3

/-- Theorem: The average number of leaves falling per hour across both trees is 14 -/
theorem average_leaves_per_hour :
  (totalLeaves ryleesBackyard.1 + totalLeaves ryleesBackyard.2) / observationHours = 14 :=
by sorry

end NUMINAMATH_CALUDE_average_leaves_per_hour_l3453_345380


namespace NUMINAMATH_CALUDE_cylinder_diagonal_angle_l3453_345351

theorem cylinder_diagonal_angle (m n : ℝ) (h : m > 0 ∧ n > 0) :
  let α := if m / n < Real.pi / 4 
           then 2 * Real.arctan (4 * m / (Real.pi * n))
           else 2 * Real.arctan (Real.pi * n / (4 * m))
  ∃ (R H : ℝ), R > 0 ∧ H > 0 ∧ 
    (Real.pi * R^2) / (2 * R * H) = m / n ∧
    α = Real.arctan (2 * R / H) + Real.arctan (2 * R / H) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_diagonal_angle_l3453_345351


namespace NUMINAMATH_CALUDE_problem_statement_l3453_345324

theorem problem_statement (a b : ℝ) (h : |a + 2| + Real.sqrt (b - 4) = 0) : a / b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3453_345324


namespace NUMINAMATH_CALUDE_max_value_3cos_minus_sin_l3453_345308

theorem max_value_3cos_minus_sin :
  ∀ x : ℝ, 3 * Real.cos x - Real.sin x ≤ Real.sqrt 10 ∧
  ∃ x : ℝ, 3 * Real.cos x - Real.sin x = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_3cos_minus_sin_l3453_345308


namespace NUMINAMATH_CALUDE_wall_clock_interval_l3453_345329

/-- Represents a wall clock that rings at regular intervals -/
structure WallClock where
  rings_per_day : ℕ
  first_ring : ℕ
  hours_in_day : ℕ

/-- Calculates the interval between rings for a given wall clock -/
def ring_interval (clock : WallClock) : ℚ :=
  clock.hours_in_day / clock.rings_per_day

/-- Theorem: If a clock rings 8 times in a 24-hour day, starting at 1 A.M., 
    then the interval between each ring is 3 hours -/
theorem wall_clock_interval (clock : WallClock) 
    (h1 : clock.rings_per_day = 8) 
    (h2 : clock.first_ring = 1) 
    (h3 : clock.hours_in_day = 24) : 
    ring_interval clock = 3 := by
  sorry

end NUMINAMATH_CALUDE_wall_clock_interval_l3453_345329


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_product_product_of_means_2_8_l3453_345358

theorem arithmetic_geometric_mean_product (x y : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) :
  let arithmetic_mean := (x + y) / 2
  let geometric_mean := Real.sqrt (x * y)
  arithmetic_mean * geometric_mean = (x + y) * Real.sqrt (x * y) / 2 :=
by sorry

theorem product_of_means_2_8 :
  let arithmetic_mean := (2 + 8) / 2
  let geometric_mean := Real.sqrt (2 * 8)
  (arithmetic_mean * geometric_mean = 20 ∨ arithmetic_mean * geometric_mean = -20) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_product_product_of_means_2_8_l3453_345358


namespace NUMINAMATH_CALUDE_square_area_is_four_l3453_345301

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a division of the square
structure SquareDivision where
  square : Square
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  sum_areas : area1 + area2 + area3 + area4 = square.side ^ 2
  perpendicular_division : True  -- This is a placeholder for the perpendicular division condition

-- Theorem statement
theorem square_area_is_four 
  (div : SquareDivision) 
  (h1 : div.area1 = 1) 
  (h2 : div.area2 = 1) 
  (h3 : div.area3 = 1) : 
  div.square.side ^ 2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_square_area_is_four_l3453_345301


namespace NUMINAMATH_CALUDE_max_value_of_f_l3453_345328

def f (x : ℝ) : ℝ := -2 * x^2 + 8

theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3453_345328


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l3453_345369

/-- Given the total amount of paint and the amount of white paint used,
    calculate the amount of blue paint used. -/
theorem blue_paint_calculation (total_paint white_paint : ℕ) 
    (h1 : total_paint = 6689)
    (h2 : white_paint = 660) :
    total_paint - white_paint = 6029 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_calculation_l3453_345369


namespace NUMINAMATH_CALUDE_oil_demand_scientific_notation_l3453_345345

theorem oil_demand_scientific_notation :
  (735000000 : ℝ) = 7.35 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_oil_demand_scientific_notation_l3453_345345


namespace NUMINAMATH_CALUDE_inequality_proof_l3453_345383

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) :
  Real.sqrt (a * b / (c + a * b)) + 
  Real.sqrt (b * c / (a + b * c)) + 
  Real.sqrt (c * a / (b + c * a)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3453_345383


namespace NUMINAMATH_CALUDE_max_graduates_proof_l3453_345315

theorem max_graduates_proof (x : ℕ) : 
  x ≤ 210 ∧ 
  (49 + ((x - 50) / 8) * 7 : ℝ) / x > 0.9 ∧ 
  ∀ y : ℕ, y > 210 → (49 + ((y - 50) / 8) * 7 : ℝ) / y ≤ 0.9 := by
  sorry

end NUMINAMATH_CALUDE_max_graduates_proof_l3453_345315


namespace NUMINAMATH_CALUDE_corey_candies_l3453_345347

theorem corey_candies :
  let total_candies : ℝ := 66.5
  let tapanga_extra : ℝ := 8.25
  let corey_candies : ℝ := (total_candies - tapanga_extra) / 2
  corey_candies = 29.125 :=
by
  sorry

end NUMINAMATH_CALUDE_corey_candies_l3453_345347


namespace NUMINAMATH_CALUDE_parabola_tangent_theorem_l3453_345317

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line that point A is on
def line_A (x y : ℝ) : Prop := x - 2*y + 13 = 0

-- Define that A is not on the y-axis
def A_not_on_y_axis (x y : ℝ) : Prop := x ≠ 0

-- Define points M and N as tangent points on the parabola
def M_N_tangent_points (xm ym xn yn : ℝ) : Prop :=
  parabola xm ym ∧ parabola xn yn

-- Define B and C as intersection points of AM and AN with y-axis
def B_C_intersection_points (xb yb xc yc : ℝ) : Prop :=
  xb = 0 ∧ xc = 0

-- Theorem statement
theorem parabola_tangent_theorem
  (xa ya xm ym xn yn xb yb xc yc : ℝ)
  (h1 : line_A xa ya)
  (h2 : A_not_on_y_axis xa ya)
  (h3 : M_N_tangent_points xm ym xn yn)
  (h4 : B_C_intersection_points xb yb xc yc) :
  -- 1. Line MN passes through (13, 8)
  ∃ (t : ℝ), xm + t * (xn - xm) = 13 ∧ ym + t * (yn - ym) = 8 ∧
  -- 2. Circumcircle of ABC passes through (2, 0)
  (xa - 2)^2 + ya^2 = (xb - 2)^2 + yb^2 ∧ (xa - 2)^2 + ya^2 = (xc - 2)^2 + yc^2 ∧
  -- 3. Minimum radius of circumcircle is (3√5)/2
  ∃ (r : ℝ), r ≥ (3 * Real.sqrt 5) / 2 ∧
    (xa - 2)^2 + ya^2 = 4 * r^2 ∧ (xb - 2)^2 + yb^2 = 4 * r^2 ∧ (xc - 2)^2 + yc^2 = 4 * r^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_theorem_l3453_345317


namespace NUMINAMATH_CALUDE_composite_sum_product_l3453_345368

def first_composite : ℕ := 4
def second_composite : ℕ := 6
def third_composite : ℕ := 8
def fourth_composite : ℕ := 9
def fifth_composite : ℕ := 10

theorem composite_sum_product : 
  (first_composite * second_composite * third_composite) + 
  (fourth_composite * fifth_composite) = 282 := by
sorry

end NUMINAMATH_CALUDE_composite_sum_product_l3453_345368


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3453_345312

theorem polygon_sides_count (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 3 * 360 → n = 8 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3453_345312


namespace NUMINAMATH_CALUDE_hotel_cost_per_night_l3453_345339

theorem hotel_cost_per_night (nights : ℕ) (discount : ℕ) (total_paid : ℕ) (cost_per_night : ℕ) : 
  nights = 3 → 
  discount = 100 → 
  total_paid = 650 → 
  nights * cost_per_night - discount = total_paid → 
  cost_per_night = 250 := by
sorry

end NUMINAMATH_CALUDE_hotel_cost_per_night_l3453_345339


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3453_345384

theorem arithmetic_mean_problem (x y : ℚ) :
  (((3 * x + 12) + (2 * y + 18) + 5 * x + 6 * y + (3 * x + y + 16)) / 5 = 60) →
  (x = 2 * y) →
  (x = 254 / 15 ∧ y = 127 / 15) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3453_345384


namespace NUMINAMATH_CALUDE_average_after_addition_l3453_345331

theorem average_after_addition (numbers : List ℝ) (initial_average : ℝ) (addition : ℝ) : 
  numbers.length = 15 →
  initial_average = 40 →
  addition = 12 →
  (numbers.map (· + addition)).sum / numbers.length = 52 := by
  sorry

end NUMINAMATH_CALUDE_average_after_addition_l3453_345331


namespace NUMINAMATH_CALUDE_problem_statement_l3453_345337

theorem problem_statement (a b c t : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (ht : t ≥ 1) 
  (sum_eq : a + b + c = 1/2) 
  (sqrt_eq : Real.sqrt (a + 1/2 * (b - c)^2) + Real.sqrt b + Real.sqrt c = Real.sqrt (6*t) / 2) :
  a^(2*t) + b^(2*t) + c^(2*t) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3453_345337


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3453_345303

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 160 → n * (180 - interior_angle) = 360 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3453_345303


namespace NUMINAMATH_CALUDE_soap_calculation_l3453_345340

/-- Given a number of packs and bars per pack, calculates the total number of bars -/
def total_bars (packs : ℕ) (bars_per_pack : ℕ) : ℕ := packs * bars_per_pack

/-- Theorem stating that 6 packs with 5 bars each results in 30 total bars -/
theorem soap_calculation : total_bars 6 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_soap_calculation_l3453_345340


namespace NUMINAMATH_CALUDE_trig_ratios_for_point_l3453_345327

theorem trig_ratios_for_point (m : ℝ) (α : ℝ) (h : m < 0) :
  let x : ℝ := 3 * m
  let y : ℝ := -2 * m
  let r : ℝ := Real.sqrt (x^2 + y^2)
  (x, y) = (3 * m, -2 * m) →
  Real.sin α = 2 * Real.sqrt 13 / 13 ∧
  Real.cos α = -(3 * Real.sqrt 13 / 13) ∧
  Real.tan α = -2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_trig_ratios_for_point_l3453_345327


namespace NUMINAMATH_CALUDE_total_trees_in_park_l3453_345389

theorem total_trees_in_park (ancient_oaks : ℕ) (fir_trees : ℕ) (saplings : ℕ)
  (h1 : ancient_oaks = 15)
  (h2 : fir_trees = 23)
  (h3 : saplings = 58) :
  ancient_oaks + fir_trees + saplings = 96 :=
by sorry

end NUMINAMATH_CALUDE_total_trees_in_park_l3453_345389


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l3453_345325

theorem bowling_team_average_weight 
  (original_team_size : ℕ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (new_average_weight : ℝ) 
  (h1 : original_team_size = 7)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_average_weight = 106) :
  ∃ (original_average_weight : ℝ),
    (original_team_size * original_average_weight + new_player1_weight + new_player2_weight) / 
    (original_team_size + 2) = new_average_weight ∧
    original_average_weight = 112 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l3453_345325


namespace NUMINAMATH_CALUDE_front_yard_eggs_count_l3453_345393

/-- The number of eggs in June's front yard nest -/
def front_yard_eggs : ℕ := sorry

/-- The total number of eggs June found -/
def total_eggs : ℕ := 17

/-- The number of nests in the first tree -/
def nests_in_first_tree : ℕ := 2

/-- The number of eggs in each nest in the first tree -/
def eggs_per_nest_first_tree : ℕ := 5

/-- The number of nests in the second tree -/
def nests_in_second_tree : ℕ := 1

/-- The number of eggs in the nest in the second tree -/
def eggs_in_second_tree : ℕ := 3

theorem front_yard_eggs_count :
  front_yard_eggs = total_eggs - (nests_in_first_tree * eggs_per_nest_first_tree + nests_in_second_tree * eggs_in_second_tree) :=
by sorry

end NUMINAMATH_CALUDE_front_yard_eggs_count_l3453_345393


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l3453_345376

theorem base_10_to_base_7 : 
  ∃ (a b c : Nat), 
    234 = a * 7^2 + b * 7^1 + c * 7^0 ∧ 
    a < 7 ∧ b < 7 ∧ c < 7 ∧
    a = 4 ∧ b = 5 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l3453_345376


namespace NUMINAMATH_CALUDE_dans_cards_l3453_345353

theorem dans_cards (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 27 → bought = 20 → total = 88 → total - bought - initial = 41 := by
  sorry

end NUMINAMATH_CALUDE_dans_cards_l3453_345353


namespace NUMINAMATH_CALUDE_quadratic_constant_term_l3453_345335

theorem quadratic_constant_term (a : ℝ) : 
  ((∀ x, (a + 2) * x^2 - 3 * a * x + a - 6 = 0 → (a + 2) ≠ 0) ∧ 
   a - 6 = 0) → 
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_constant_term_l3453_345335


namespace NUMINAMATH_CALUDE_cards_per_page_l3453_345360

/-- Given Will's baseball card organization problem, prove that he puts 3 cards on each page. -/
theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) :
  (new_cards + old_cards) / pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_page_l3453_345360


namespace NUMINAMATH_CALUDE_intersecting_line_circle_isosceles_right_triangle_l3453_345379

/-- Given a line and a circle that intersect at two points forming an isosceles right triangle with a third point, prove the value of the parameter a. -/
theorem intersecting_line_circle_isosceles_right_triangle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 + a * A.2 - 1 = 0 ∧ (A.1 + a)^2 + (A.2 - 1)^2 = 1) ∧
    (B.1 + a * B.2 - 1 = 0 ∧ (B.1 + a)^2 + (B.2 - 1)^2 = 1) ∧
    A ≠ B) →
  (∃ C : ℝ × ℝ, 
    (C.1 + a * C.2 - 1 ≠ 0 ∨ (C.1 + a)^2 + (C.2 - 1)^2 ≠ 1) ∧
    (dist A C = dist B C ∧ dist A B = dist A C * Real.sqrt 2)) →
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_intersecting_line_circle_isosceles_right_triangle_l3453_345379


namespace NUMINAMATH_CALUDE_roots_and_p_value_l3453_345395

-- Define the polynomial
def f (p : ℝ) (x : ℝ) : ℝ := x^3 + 7*x^2 + 14*x - p

-- Define the condition of three distinct roots in geometric progression
def has_three_distinct_roots_in_gp (p : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    f p a = 0 ∧ f p b = 0 ∧ f p c = 0 ∧
    ∃ (r : ℝ), r ≠ 0 ∧ r ≠ 1 ∧ b = a * r ∧ c = b * r

-- Theorem statement
theorem roots_and_p_value (p : ℝ) :
  has_three_distinct_roots_in_gp p →
  p = -8 ∧ f p (-1) = 0 ∧ f p (-2) = 0 ∧ f p (-4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_and_p_value_l3453_345395


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l3453_345375

theorem percentage_passed_both_subjects (total_students : ℕ) 
  (failed_hindi : ℕ) (failed_english : ℕ) (failed_both : ℕ) :
  failed_hindi = (35 * total_students) / 100 →
  failed_english = (45 * total_students) / 100 →
  failed_both = (20 * total_students) / 100 →
  total_students > 0 →
  ((total_students - (failed_hindi + failed_english - failed_both)) * 100) / total_students = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l3453_345375


namespace NUMINAMATH_CALUDE_joey_age_l3453_345326

def ages : List ℕ := [4, 6, 8, 10, 12]

def is_cinema_pair (a b : ℕ) : Prop := a + b = 18 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

def is_soccer_pair (a b : ℕ) : Prop := 
  a < 11 ∧ b < 11 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ 
  ¬(∃ c d, is_cinema_pair c d ∧ (a = c ∨ a = d ∨ b = c ∨ b = d))

theorem joey_age : 
  (∃ a b c d, is_cinema_pair a b ∧ is_soccer_pair c d) →
  (∃! x, x ∈ ages ∧ x ≠ 6 ∧ ¬(∃ y z, (is_cinema_pair x y ∨ is_cinema_pair y x) ∧ 
                                     (is_soccer_pair x z ∨ is_soccer_pair z x))) →
  (∃ x, x ∈ ages ∧ x ≠ 6 ∧ ¬(∃ y z, (is_cinema_pair x y ∨ is_cinema_pair y x) ∧ 
                                    (is_soccer_pair x z ∨ is_soccer_pair z x)) ∧ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_joey_age_l3453_345326


namespace NUMINAMATH_CALUDE_problem_solution_l3453_345338

theorem problem_solution (a b c d e x : ℝ) 
  (h : ((x + a) ^ b) / c - d = e / 2) : 
  x = (c * e / 2 + c * d) ^ (1 / b) - a := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3453_345338


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l3453_345357

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : m % 2 = 0) (h3 : m % 221 = 0) :
  ∃ (d : ℕ), d > 221 ∧ m % d = 0 ∧ (∀ (x : ℕ), 221 < x ∧ x < d → m % x ≠ 0) → d = 247 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l3453_345357


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_l3453_345370

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  x + Real.sqrt 2 * y = 0

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 12 * x

-- Define the focus of the parabola
def parabola_focus (x : ℝ) : Prop :=
  x = 3

-- Define the point M on the hyperbola
def point_M (x y : ℝ) : Prop :=
  x = -3 ∧ y = Real.sqrt 6 / 2

-- Define the line F2M
def line_F2M (x y : ℝ) : Prop :=
  y = -Real.sqrt 6 / 12 * x + Real.sqrt 6 / 4

-- State the theorem
theorem hyperbola_focus_distance (a b x y : ℝ) :
  hyperbola a b x y →
  asymptote x y →
  parabola_focus a →
  point_M x y →
  line_F2M x y →
  (6 : ℝ) / 5 = abs (-Real.sqrt 6 / 12 * (-3) + Real.sqrt 6 / 4) / Real.sqrt (1 + 6 / 144) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_l3453_345370


namespace NUMINAMATH_CALUDE_potato_bag_weight_l3453_345316

/-- If a bag of potatoes weighs 12 lbs divided by half of its weight, then the weight of the bag is 24 lbs. -/
theorem potato_bag_weight (w : ℝ) (h : w = 12 / (w / 2)) : w = 24 :=
sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l3453_345316


namespace NUMINAMATH_CALUDE_problem_solution_l3453_345300

theorem problem_solution :
  (195 * 205 = 39975) ∧
  (9 * 11 * 101 * 10001 = 99999999) ∧
  (∀ a : ℝ, a^2 - 6*a + 8 = (a - 2)*(a - 4)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3453_345300


namespace NUMINAMATH_CALUDE_tangent_problem_l3453_345387

theorem tangent_problem (α β : ℝ) 
  (h1 : Real.tan (α - 2 * β) = 4)
  (h2 : Real.tan β = 2) :
  (Real.tan α - 2) / (1 + 2 * Real.tan α) = -6/7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l3453_345387


namespace NUMINAMATH_CALUDE_matrix_determinant_sixteen_l3453_345388

def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; x, 4*x]

theorem matrix_determinant_sixteen (x : ℝ) : 
  Matrix.det (matrix x) = 16 ↔ x = 4/3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_sixteen_l3453_345388


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_one_l3453_345352

-- Define the binary logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem logarithm_expression_equals_one :
  lg 2 * lg 50 + lg 25 - lg 5 * lg 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_one_l3453_345352


namespace NUMINAMATH_CALUDE_percent_error_multiplication_l3453_345392

theorem percent_error_multiplication (x : ℝ) (h : x > 0) : 
  (|12 * x - x / 3| / (x / 3)) * 100 = 3500 := by
sorry

end NUMINAMATH_CALUDE_percent_error_multiplication_l3453_345392


namespace NUMINAMATH_CALUDE_dartboard_angle_l3453_345302

/-- Given a circular dartboard, if the probability of a dart landing in a particular region is 1/4,
    then the measure of the central angle of that region is 90 degrees. -/
theorem dartboard_angle (probability : ℝ) (angle : ℝ) :
  probability = 1/4 →
  angle = probability * 360 →
  angle = 90 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_angle_l3453_345302


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3453_345321

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x^2 - 1 = 0 → x^3 - x = 0) ∧ 
  (∃ x : ℝ, x^3 - x = 0 ∧ x^2 - 1 ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3453_345321


namespace NUMINAMATH_CALUDE_rectangular_prism_inequality_l3453_345377

theorem rectangular_prism_inequality (a b c l : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hl : l > 0)
  (h_diagonal : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_inequality_l3453_345377


namespace NUMINAMATH_CALUDE_set_operation_equality_l3453_345336

universe u

def U : Set (Fin 5) := {0, 1, 2, 3, 4}

def M : Set (Fin 5) := {0, 3}

def N : Set (Fin 5) := {0, 2, 4}

theorem set_operation_equality :
  M ∪ (Mᶜ ∩ N) = {0, 2, 3, 4} :=
by sorry

end NUMINAMATH_CALUDE_set_operation_equality_l3453_345336


namespace NUMINAMATH_CALUDE_right_building_shorter_l3453_345367

def middle_height : ℝ := 100
def left_height : ℝ := 0.8 * middle_height
def total_height : ℝ := 340

theorem right_building_shorter : 
  (middle_height + left_height) - (total_height - (middle_height + left_height)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_building_shorter_l3453_345367


namespace NUMINAMATH_CALUDE_fifteen_ways_to_select_l3453_345394

/-- Represents the number of ways to select performers for singing and dancing -/
def select_performers (total : ℕ) (dancers : ℕ) (singers : ℕ) : ℕ :=
  let both := dancers + singers - total
  let pure_singers := singers - both
  let pure_dancers := dancers - both
  both * pure_dancers + pure_singers * both

/-- Theorem stating that there are 15 ways to select performers from a group of 8,
    where 6 can dance and 5 can sing -/
theorem fifteen_ways_to_select :
  select_performers 8 6 5 = 15 := by
  sorry

#eval select_performers 8 6 5

end NUMINAMATH_CALUDE_fifteen_ways_to_select_l3453_345394


namespace NUMINAMATH_CALUDE_rectangle_area_l3453_345385

theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) :
  length = 2 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 120 →
  area = length * width →
  area = 800 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3453_345385


namespace NUMINAMATH_CALUDE_candy_box_price_l3453_345348

/-- Proves that the current price of a candy box is 15 pounds given the conditions -/
theorem candy_box_price (
  soda_price : ℝ)
  (candy_increase : ℝ)
  (soda_increase : ℝ)
  (original_total : ℝ)
  (h1 : soda_price = 6)
  (h2 : candy_increase = 0.25)
  (h3 : soda_increase = 0.50)
  (h4 : original_total = 16) :
  ∃ (candy_price : ℝ), candy_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_price_l3453_345348
