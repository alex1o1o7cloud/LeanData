import Mathlib

namespace NUMINAMATH_CALUDE_number_count_with_incorrect_average_l1932_193266

theorem number_count_with_incorrect_average (n : ℕ) : 
  (n : ℝ) * 40.2 - (n : ℝ) * 40.1 = 35 → n = 350 := by
  sorry

end NUMINAMATH_CALUDE_number_count_with_incorrect_average_l1932_193266


namespace NUMINAMATH_CALUDE_remainder_1234567891011_div_210_l1932_193218

theorem remainder_1234567891011_div_210 : 1234567891011 % 210 = 31 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567891011_div_210_l1932_193218


namespace NUMINAMATH_CALUDE_smallest_number_l1932_193257

theorem smallest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -3) (hc : c = 1) (hd : d = -1) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by sorry

end NUMINAMATH_CALUDE_smallest_number_l1932_193257


namespace NUMINAMATH_CALUDE_exists_equivalent_expr_l1932_193244

/-- Represents the two possible binary operations in our system -/
inductive Op
| add
| sub

/-- Represents an expression in our system -/
inductive Expr
| var : String → Expr
| op : Op → Expr → Expr → Expr

/-- Evaluates an expression given an assignment of values to variables and a mapping of symbols to operations -/
def evaluate (e : Expr) (vars : String → ℝ) (sym_to_op : Op → Op) : ℝ :=
  match e with
  | Expr.var v => vars v
  | Expr.op o e1 e2 =>
    let v1 := evaluate e1 vars sym_to_op
    let v2 := evaluate e2 vars sym_to_op
    match sym_to_op o with
    | Op.add => v1 + v2
    | Op.sub => v1 - v2

/-- The theorem to be proved -/
theorem exists_equivalent_expr :
  ∃ (e : Expr),
    ∀ (vars : String → ℝ) (sym_to_op : Op → Op),
      evaluate e vars sym_to_op = 20 * vars "a" - 18 * vars "b" :=
sorry

end NUMINAMATH_CALUDE_exists_equivalent_expr_l1932_193244


namespace NUMINAMATH_CALUDE_sum_remainder_mod_13_l1932_193275

theorem sum_remainder_mod_13 : (9123 + 9124 + 9125 + 9126) % 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_13_l1932_193275


namespace NUMINAMATH_CALUDE_log_expression_equality_l1932_193269

-- Define lg as base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_expression_equality :
  (Real.log (Real.sqrt 27) / Real.log 3) + lg 25 + lg 4 + 7^(Real.log 2 / Real.log 7) + (-9.8)^0 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l1932_193269


namespace NUMINAMATH_CALUDE_smaller_cube_side_length_l1932_193221

/-- Given a cube of side length 9 that is painted and cut into smaller cubes,
    if there are 12 smaller cubes with paint on exactly 2 sides,
    then the side length of the smaller cubes is 4.5. -/
theorem smaller_cube_side_length 
  (large_cube_side : ℝ) 
  (small_cubes_two_sides : ℕ) 
  (small_cube_side : ℝ) : 
  large_cube_side = 9 → 
  small_cubes_two_sides = 12 → 
  small_cubes_two_sides = 12 * (large_cube_side / small_cube_side - 1) → 
  small_cube_side = 4.5 := by
sorry

end NUMINAMATH_CALUDE_smaller_cube_side_length_l1932_193221


namespace NUMINAMATH_CALUDE_great_wall_scientific_notation_l1932_193261

/-- Represents the length of the Great Wall in meters -/
def great_wall_length : ℝ := 6700010

/-- Converts a number to scientific notation with two significant figures -/
def to_scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

/-- Theorem stating that the scientific notation of the Great Wall's length
    is equal to 6.7 × 10^6 when rounded to two significant figures -/
theorem great_wall_scientific_notation :
  to_scientific_notation great_wall_length = (6.7, 6) :=
sorry

end NUMINAMATH_CALUDE_great_wall_scientific_notation_l1932_193261


namespace NUMINAMATH_CALUDE_max_value_xyz_l1932_193286

theorem max_value_xyz (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  ∃ (M : ℝ), M = 1 ∧ ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 1 → a + b^3 + c^4 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_xyz_l1932_193286


namespace NUMINAMATH_CALUDE_apple_tree_yield_l1932_193292

theorem apple_tree_yield (apple_trees peach_trees : ℕ) 
  (peach_yield total_yield : ℝ) (h1 : apple_trees = 30) 
  (h2 : peach_trees = 45) (h3 : peach_yield = 65) 
  (h4 : total_yield = 7425) : 
  (total_yield - peach_trees * peach_yield) / apple_trees = 150 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_yield_l1932_193292


namespace NUMINAMATH_CALUDE_wage_increase_percentage_l1932_193230

theorem wage_increase_percentage (original_wage new_wage : ℝ) 
  (h1 : original_wage = 34)
  (h2 : new_wage = 51) :
  (new_wage - original_wage) / original_wage * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_percentage_l1932_193230


namespace NUMINAMATH_CALUDE_power_of_two_equation_l1932_193242

theorem power_of_two_equation : ∃ x : ℕ, 
  8 * (32 ^ 10) = 2 ^ x ∧ x = 53 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l1932_193242


namespace NUMINAMATH_CALUDE_spade_calculation_l1932_193202

-- Define the ⋄ operation
def spade (x y : ℝ) : ℝ := (x + y)^2 * (x - y)

-- Theorem statement
theorem spade_calculation : spade 2 (spade 3 6) = 14229845 := by sorry

end NUMINAMATH_CALUDE_spade_calculation_l1932_193202


namespace NUMINAMATH_CALUDE_product_of_decimals_l1932_193283

theorem product_of_decimals : 0.3 * 0.7 = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1932_193283


namespace NUMINAMATH_CALUDE_f_properties_l1932_193295

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4
  else if x ≤ 4 then x^2 - 2*x
  else -x + 2

theorem f_properties :
  (f 0 = 4) ∧
  (f 5 = -3) ∧
  (f (f (f 5)) = -1) ∧
  (∃! a, f a = 8 ∧ a = 4) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1932_193295


namespace NUMINAMATH_CALUDE_expression_evaluation_l1932_193263

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^(y + 1) + 6 * y^(x + 1) = 2751 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1932_193263


namespace NUMINAMATH_CALUDE_damien_jogging_distance_l1932_193201

/-- The number of miles Damien jogs per day on weekdays -/
def miles_per_day : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 3

/-- The total distance Damien runs over three weeks -/
def total_distance : ℕ := miles_per_day * weekdays_per_week * num_weeks

theorem damien_jogging_distance :
  total_distance = 75 := by
  sorry

end NUMINAMATH_CALUDE_damien_jogging_distance_l1932_193201


namespace NUMINAMATH_CALUDE_exactly_one_statement_true_l1932_193226

/-- Polynomials A, B, C, D, and E -/
def A (x : ℝ) : ℝ := 2 * x^2
def B (x : ℝ) : ℝ := x + 1
def C (x : ℝ) : ℝ := -2 * x
def D (y : ℝ) : ℝ := y^2
def E (x y : ℝ) : ℝ := 2 * x - y

/-- Statement 1: For all positive integer y, B*C + A + D + E > 0 -/
def statement1 : Prop :=
  ∀ (x : ℝ) (y : ℕ), (B x * C x + A x + D y + E x y) > 0

/-- Statement 2: There exist real numbers x and y such that A + D + 2E = -2 -/
def statement2 : Prop :=
  ∃ (x y : ℝ), A x + D y + 2 * E x y = -2

/-- Statement 3: For all real x, if 3(A-B) + m*B*C has no linear term in x
    (where m is a constant), then 3(A-B) + m*B*C > -3 -/
def statement3 : Prop :=
  ∀ (x m : ℝ),
    (∃ (k : ℝ), 3 * (A x - B x) + m * B x * C x = k * x^2 + (3 * (A 0 - B 0) + m * B 0 * C 0)) →
    3 * (A x - B x) + m * B x * C x > -3

theorem exactly_one_statement_true :
  (statement1 ∧ ¬statement2 ∧ ¬statement3) ∨
  (¬statement1 ∧ statement2 ∧ ¬statement3) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3) := by sorry

end NUMINAMATH_CALUDE_exactly_one_statement_true_l1932_193226


namespace NUMINAMATH_CALUDE_lamp_post_ratio_l1932_193265

theorem lamp_post_ratio (k m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 9 * x = k ∧ 99 * x = m) → m / k = 11 := by
  sorry

end NUMINAMATH_CALUDE_lamp_post_ratio_l1932_193265


namespace NUMINAMATH_CALUDE_set_operation_result_l1932_193274

def A : Set Nat := {1, 2, 6}
def B : Set Nat := {2, 4}
def C : Set Nat := {1, 2, 3, 4}

theorem set_operation_result : (A ∪ B) ∩ C = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l1932_193274


namespace NUMINAMATH_CALUDE_exponent_of_5_in_30_factorial_l1932_193260

/-- The exponent of 5 in the prime factorization of n! -/
def exponent_of_5_in_factorial (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The exponent of 5 in the prime factorization of 30! is 7 -/
theorem exponent_of_5_in_30_factorial :
  exponent_of_5_in_factorial 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_of_5_in_30_factorial_l1932_193260


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1932_193205

theorem expression_simplification_and_evaluation (a b : ℚ) 
  (ha : a = -2) (hb : b = 3/2) : 
  1/2 * a - 2 * (a - 1/2 * b^2) - (3/2 * a - 1/3 * b^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1932_193205


namespace NUMINAMATH_CALUDE_rectangle_area_l1932_193258

theorem rectangle_area (width height : ℝ) (h1 : width / height = 0.875) (h2 : height = 24) :
  width * height = 504 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1932_193258


namespace NUMINAMATH_CALUDE_jennas_profit_is_4000_l1932_193287

/-- Calculates Jenna's total profit after expenses and taxes -/
def jennasProfit (
  widgetBuyCost : ℚ)
  (widgetSellPrice : ℚ)
  (monthlyRent : ℚ)
  (taxRate : ℚ)
  (workerSalary : ℚ)
  (numWorkers : ℕ)
  (widgetsSold : ℕ) : ℚ :=
  let totalRevenue := widgetSellPrice * widgetsSold
  let totalCost := widgetBuyCost * widgetsSold
  let grossProfit := totalRevenue - totalCost
  let totalExpenses := monthlyRent + (workerSalary * numWorkers)
  let netProfitBeforeTax := grossProfit - totalExpenses
  let taxes := taxRate * netProfitBeforeTax
  netProfitBeforeTax - taxes

theorem jennas_profit_is_4000 :
  jennasProfit 3 8 10000 (1/5) 2500 4 5000 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_jennas_profit_is_4000_l1932_193287


namespace NUMINAMATH_CALUDE_customers_without_tip_l1932_193224

theorem customers_without_tip (initial_customers : ℕ) (additional_customers : ℕ) (customers_with_tip : ℕ) : 
  initial_customers = 39 → additional_customers = 12 → customers_with_tip = 2 →
  initial_customers + additional_customers - customers_with_tip = 49 := by
sorry

end NUMINAMATH_CALUDE_customers_without_tip_l1932_193224


namespace NUMINAMATH_CALUDE_calculate_expression_l1932_193235

theorem calculate_expression : 
  let tan_60 : ℝ := Real.sqrt 3
  |2 - tan_60| - 1 + 4 + Real.sqrt 3 = 5 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l1932_193235


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l1932_193262

/-- The sum of y-coordinates of points where a circle intersects the y-axis -/
theorem circle_y_axis_intersection_sum (c : ℝ × ℝ) (r : ℝ) : 
  c.1 = -8 → c.2 = 3 → r = 15 → 
  ∃ y₁ y₂ : ℝ, 
    (0 - c.1)^2 + (y₁ - c.2)^2 = r^2 ∧
    (0 - c.1)^2 + (y₂ - c.2)^2 = r^2 ∧
    y₁ + y₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l1932_193262


namespace NUMINAMATH_CALUDE_binomial_coefficient_16_15_l1932_193227

theorem binomial_coefficient_16_15 : Nat.choose 16 15 = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_16_15_l1932_193227


namespace NUMINAMATH_CALUDE_marble_probability_l1932_193270

/-- The number of green marbles -/
def green_marbles : ℕ := 7

/-- The number of purple marbles -/
def purple_marbles : ℕ := 5

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of trials -/
def num_trials : ℕ := 8

/-- The number of successful outcomes (choosing green marbles) -/
def num_success : ℕ := 3

/-- The probability of choosing a green marble in a single trial -/
def p : ℚ := green_marbles / total_marbles

/-- The probability of choosing a purple marble in a single trial -/
def q : ℚ := purple_marbles / total_marbles

/-- The binomial probability of choosing exactly 3 green marbles in 8 trials -/
def binomial_prob : ℚ := (Nat.choose num_trials num_success : ℚ) * p ^ num_success * q ^ (num_trials - num_success)

theorem marble_probability : binomial_prob = 9378906 / 67184015 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1932_193270


namespace NUMINAMATH_CALUDE_periodic_even_symmetric_function_l1932_193252

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is symmetric about the line x = a if f(a - x) = f(a + x) for all x -/
def IsSymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

/-- A function f is periodic with period p if f(x + p) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem periodic_even_symmetric_function (f : ℝ → ℝ) 
  (h_nonconstant : ∃ x y, f x ≠ f y)
  (h_even : IsEven f)
  (h_symmetric : IsSymmetricAbout f (Real.sqrt 2 / 2)) :
  IsPeriodic f (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_periodic_even_symmetric_function_l1932_193252


namespace NUMINAMATH_CALUDE_abc_value_l1932_193264

theorem abc_value :
  let a := -(2017 * 2017 - 2017) / (2016 * 2016 + 2016)
  let b := -(2018 * 2018 - 2018) / (2017 * 2017 + 2017)
  let c := -(2019 * 2019 - 2019) / (2018 * 2018 + 2018)
  a * b * c = -1 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1932_193264


namespace NUMINAMATH_CALUDE_extra_food_is_zero_point_four_l1932_193297

/-- The amount of cat food needed for one cat per day -/
def food_for_one_cat : ℝ := 0.5

/-- The total amount of cat food needed for two cats per day -/
def total_food_for_two_cats : ℝ := 0.9

/-- The extra amount of cat food needed for the second cat per day -/
def extra_food_for_second_cat : ℝ := total_food_for_two_cats - food_for_one_cat

theorem extra_food_is_zero_point_four :
  extra_food_for_second_cat = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_extra_food_is_zero_point_four_l1932_193297


namespace NUMINAMATH_CALUDE_may_savings_l1932_193210

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (0-indexed)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_may_savings_l1932_193210


namespace NUMINAMATH_CALUDE_potato_bag_fraction_l1932_193289

theorem potato_bag_fraction (weight : ℝ) (x : ℝ) : 
  weight = 12 → weight / x = 12 → x = 1 := by sorry

end NUMINAMATH_CALUDE_potato_bag_fraction_l1932_193289


namespace NUMINAMATH_CALUDE_contractor_fine_calculation_l1932_193277

/-- Proves that the fine for each day of absence is 7.5 given the contract conditions --/
theorem contractor_fine_calculation (total_days : ℕ) (work_pay : ℝ) (total_earnings : ℝ) (absent_days : ℕ) :
  total_days = 30 →
  work_pay = 25 →
  total_earnings = 360 →
  absent_days = 12 →
  ∃ (fine : ℝ), fine = 7.5 ∧ 
    work_pay * (total_days - absent_days) - fine * absent_days = total_earnings :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_fine_calculation_l1932_193277


namespace NUMINAMATH_CALUDE_least_subtraction_l1932_193232

theorem least_subtraction (x : ℕ) : x = 22 ↔ 
  x ≠ 0 ∧
  (∀ y : ℕ, y < x → ¬(1398 - y) % 7 = 5 ∨ ¬(1398 - y) % 9 = 5 ∨ ¬(1398 - y) % 11 = 5) ∧
  (1398 - x) % 7 = 5 ∧
  (1398 - x) % 9 = 5 ∧
  (1398 - x) % 11 = 5 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_l1932_193232


namespace NUMINAMATH_CALUDE_probability_theorem_l1932_193212

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Define the condition x^2 < 1
def condition (x : ℝ) : Prop := x^2 < 1

-- Define the measure of the interval [-2, 2]
def totalMeasure : ℝ := 4

-- Define the measure of the solution set (-1, 1)
def solutionMeasure : ℝ := 2

-- State the theorem
theorem probability_theorem :
  (solutionMeasure / totalMeasure) = (1 / 2) := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1932_193212


namespace NUMINAMATH_CALUDE_power_sum_equality_l1932_193231

theorem power_sum_equality : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1932_193231


namespace NUMINAMATH_CALUDE_x_axis_conditions_l1932_193236

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a line is the x-axis -/
def is_x_axis (l : Line) : Prop :=
  ∀ x y : ℝ, l.A * x + l.B * y + l.C = 0 ↔ y = 0

/-- Theorem stating the conditions for a line to be the x-axis -/
theorem x_axis_conditions (l : Line) : 
  is_x_axis l ↔ l.B ≠ 0 ∧ l.A = 0 ∧ l.C = 0 := by sorry

end NUMINAMATH_CALUDE_x_axis_conditions_l1932_193236


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1932_193208

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, (x ≥ b - 1 ∧ x < a / 2) ↔ (-3 ≤ x ∧ x < 3 / 2)) → 
  a * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1932_193208


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1932_193253

theorem circle_tangent_to_line (x y : ℝ) :
  (∀ a b : ℝ, a^2 + b^2 = 2 → (b ≠ 2 - a ∨ (a - 0)^2 + (b - 0)^2 = 2)) ∧
  (∃ c d : ℝ, c^2 + d^2 = 2 ∧ d = 2 - c) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1932_193253


namespace NUMINAMATH_CALUDE_hospital_employee_arrangements_l1932_193206

theorem hospital_employee_arrangements (n : ℕ) (h : n = 6) :
  (Nat.factorial n = 720) ∧
  (Nat.factorial (n - 1) = 120) ∧
  (n * (n - 1) * (n - 2) = 120) := by
  sorry

#check hospital_employee_arrangements

end NUMINAMATH_CALUDE_hospital_employee_arrangements_l1932_193206


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l1932_193282

/-- A triangle is obtuse if one of its angles is greater than 90 degrees -/
def IsObtuseTriangle (a b c : ℝ) : Prop :=
  a > 90 ∨ b > 90 ∨ c > 90

/-- Theorem: If A, B, and C are the interior angles of a triangle, 
    and A > 3B and C < 2B, then the triangle is obtuse -/
theorem triangle_is_obtuse (a b c : ℝ) 
    (angle_sum : a + b + c = 180)
    (h1 : a > 3 * b) 
    (h2 : c < 2 * b) : 
  IsObtuseTriangle a b c := by
sorry

end NUMINAMATH_CALUDE_triangle_is_obtuse_l1932_193282


namespace NUMINAMATH_CALUDE_num_lizards_seen_l1932_193213

/-- The number of legs Borgnine wants to see at the zoo -/
def total_legs : ℕ := 1100

/-- The number of chimps Borgnine has seen -/
def num_chimps : ℕ := 12

/-- The number of lions Borgnine has seen -/
def num_lions : ℕ := 8

/-- The number of tarantulas Borgnine will see -/
def num_tarantulas : ℕ := 125

/-- The number of legs a chimp has -/
def chimp_legs : ℕ := 4

/-- The number of legs a lion has -/
def lion_legs : ℕ := 4

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

/-- The number of legs a lizard has -/
def lizard_legs : ℕ := 4

/-- The theorem stating the number of lizards Borgnine has seen -/
theorem num_lizards_seen : 
  (total_legs - (num_chimps * chimp_legs + num_lions * lion_legs + num_tarantulas * tarantula_legs)) / lizard_legs = 5 := by
  sorry

end NUMINAMATH_CALUDE_num_lizards_seen_l1932_193213


namespace NUMINAMATH_CALUDE_nancy_small_gardens_l1932_193222

/-- Given the total number of seeds, seeds planted in the big garden, and seeds per small garden,
    calculate the number of small gardens Nancy had. -/
def number_of_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Prove that Nancy had 6 small gardens given the conditions. -/
theorem nancy_small_gardens :
  number_of_small_gardens 52 28 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_nancy_small_gardens_l1932_193222


namespace NUMINAMATH_CALUDE_not_proportional_six_nine_nine_twelve_l1932_193285

/-- Two ratios a:b and c:d are proportional if a/b = c/d -/
def proportional (a b c d : ℚ) : Prop := a / b = c / d

/-- The ratios 6:9 and 9:12 -/
def ratio1 : ℚ := 6 / 9
def ratio2 : ℚ := 9 / 12

/-- Theorem stating that 6:9 and 9:12 are not proportional -/
theorem not_proportional_six_nine_nine_twelve : ¬(proportional 6 9 9 12) := by
  sorry

end NUMINAMATH_CALUDE_not_proportional_six_nine_nine_twelve_l1932_193285


namespace NUMINAMATH_CALUDE_largest_consecutive_sums_of_squares_l1932_193248

/-- A function that checks if a natural number is the sum of two squares -/
def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

/-- A function that checks if k consecutive natural numbers starting from n
    are all sums of two squares -/
def k_consecutive_sums_of_squares (k : ℕ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i < k → is_sum_of_two_squares (n + i)

/-- The theorem stating that 3 is the largest natural number k such that
    there are infinitely many sequences of k consecutive natural numbers
    where each number can be expressed as the sum of two squares -/
theorem largest_consecutive_sums_of_squares :
  (∃ k : ℕ, k > 0 ∧
    (∀ m : ℕ, ∃ n > m, k_consecutive_sums_of_squares k n) ∧
    (∀ k' : ℕ, k' > k →
      ¬(∀ m : ℕ, ∃ n > m, k_consecutive_sums_of_squares k' n))) ∧
  (∀ m : ℕ, ∃ n > m, k_consecutive_sums_of_squares 3 n) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sums_of_squares_l1932_193248


namespace NUMINAMATH_CALUDE_tyler_remaining_money_l1932_193247

/-- Calculates the remaining money after Tyler's purchases -/
def remaining_money (initial_amount : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) 
                    (eraser_count : ℕ) (eraser_price : ℕ) : ℕ :=
  initial_amount - (scissors_count * scissors_price + eraser_count * eraser_price)

/-- Theorem stating that Tyler will have $20 remaining after his purchases -/
theorem tyler_remaining_money : 
  remaining_money 100 8 5 10 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tyler_remaining_money_l1932_193247


namespace NUMINAMATH_CALUDE_students_both_correct_l1932_193273

theorem students_both_correct (total : ℕ) (physics_correct : ℕ) (chemistry_correct : ℕ) (both_incorrect : ℕ)
  (h1 : total = 50)
  (h2 : physics_correct = 40)
  (h3 : chemistry_correct = 31)
  (h4 : both_incorrect = 4) :
  total - both_incorrect - (physics_correct + chemistry_correct - total + both_incorrect) = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_both_correct_l1932_193273


namespace NUMINAMATH_CALUDE_expression_simplification_l1932_193239

theorem expression_simplification (a : ℝ) (h : a^2 + 3*a - 2 = 0) :
  ((a^2 - 4) / (a^2 - 4*a + 4) - 1 / (2 - a)) / (2 / (a^2 - 2*a)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1932_193239


namespace NUMINAMATH_CALUDE_arithmetic_progression_theorem_l1932_193250

/-- An arithmetic progression with n terms, first term a, and common difference d. -/
structure ArithmeticProgression where
  n : ℕ
  a : ℚ
  d : ℚ

/-- Sum of the first k terms of an arithmetic progression -/
def sum_first_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  k / 2 * (2 * ap.a + (k - 1) * ap.d)

/-- Sum of the last k terms of an arithmetic progression -/
def sum_last_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  k * (2 * ap.a + (ap.n - k + 1 + ap.n - 1) * ap.d / 2)

/-- Sum of all terms except the first k terms -/
def sum_without_first_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  (ap.n - k) / 2 * (2 * ap.a + (2 * k - 1 + ap.n - 1) * ap.d)

/-- Sum of all terms except the last k terms -/
def sum_without_last_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  (ap.n - k) / 2 * (2 * ap.a + (ap.n - k - 1) * ap.d)

/-- Theorem: If the sum of the first 13 terms is 50% of the sum of the last 13 terms,
    and the sum of all terms without the first 3 terms is 3/2 times the sum of all terms
    without the last 3 terms, then the number of terms in the progression is 18. -/
theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  sum_first_k ap 13 = (1/2) * sum_last_k ap 13 ∧
  sum_without_first_k ap 3 = (3/2) * sum_without_last_k ap 3 →
  ap.n = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_theorem_l1932_193250


namespace NUMINAMATH_CALUDE_apartment_number_change_l1932_193200

/-- Represents a building with apartments and entrances. -/
structure Building where
  num_entrances : ℕ
  apartments_per_entrance : ℕ

/-- Calculates the apartment number given the entrance number and apartment number within the entrance. -/
def apartment_number (b : Building) (entrance : ℕ) (apartment_in_entrance : ℕ) : ℕ :=
  (entrance - 1) * b.apartments_per_entrance + apartment_in_entrance

/-- Theorem stating that if an apartment's number changes from 636 to 242 when entrance numbering is reversed in a 5-entrance building, then the total number of apartments is 985. -/
theorem apartment_number_change (b : Building) 
  (h1 : b.num_entrances = 5)
  (h2 : ∃ (e1 e2 a : ℕ), 
    apartment_number b e1 a = 636 ∧ 
    apartment_number b (b.num_entrances - e1 + 1) a = 242) :
  b.num_entrances * b.apartments_per_entrance = 985 := by
  sorry

#check apartment_number_change

end NUMINAMATH_CALUDE_apartment_number_change_l1932_193200


namespace NUMINAMATH_CALUDE_zoltan_incorrect_answers_l1932_193203

theorem zoltan_incorrect_answers 
  (total_questions : Nat)
  (answered_questions : Nat)
  (total_score : Int)
  (correct_points : Int)
  (incorrect_points : Int)
  (unanswered_points : Int)
  (h1 : total_questions = 50)
  (h2 : answered_questions = 45)
  (h3 : total_score = 135)
  (h4 : correct_points = 4)
  (h5 : incorrect_points = -1)
  (h6 : unanswered_points = 0) :
  ∃ (incorrect : Nat),
    incorrect = 9 ∧
    (answered_questions - incorrect) * correct_points + 
    incorrect * incorrect_points + 
    (total_questions - answered_questions) * unanswered_points = total_score :=
by sorry

end NUMINAMATH_CALUDE_zoltan_incorrect_answers_l1932_193203


namespace NUMINAMATH_CALUDE_number_of_girls_l1932_193241

theorem number_of_girls (total_students : ℕ) (prob_girl : ℚ) (num_girls : ℕ) : 
  total_students = 20 →
  prob_girl = 2/5 →
  num_girls = (total_students : ℚ) * prob_girl →
  num_girls = 8 := by
sorry

end NUMINAMATH_CALUDE_number_of_girls_l1932_193241


namespace NUMINAMATH_CALUDE_kendra_change_l1932_193238

/-- Calculates the change received after a purchase -/
def calculate_change (toy_price hat_price : ℕ) (num_toys num_hats : ℕ) (paid : ℕ) : ℕ :=
  paid - (toy_price * num_toys + hat_price * num_hats)

/-- Proves that Kendra received $30 in change -/
theorem kendra_change : calculate_change 20 10 2 3 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_kendra_change_l1932_193238


namespace NUMINAMATH_CALUDE_chessboard_selections_theorem_l1932_193219

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_valid : size = 4 ∨ size = 8)

/-- Represents a selection of squares on a chessboard -/
structure Selection (board : Chessboard) :=
  (count : Nat)
  (row_count : Nat)
  (col_count : Nat)
  (is_valid : count = row_count * board.size ∧ row_count = col_count)

/-- Counts the number of valid selections on a 4x4 board -/
def count_4x4_selections (board : Chessboard) (sel : Selection board) : Nat :=
  24

/-- Counts the number of valid selections on an 8x8 board with all black squares chosen -/
def count_8x8_selections (board : Chessboard) (sel : Selection board) : Nat :=
  576

/-- The main theorem to prove -/
theorem chessboard_selections_theorem (board4 : Chessboard) (board8 : Chessboard) 
  (sel4 : Selection board4) (sel8 : Selection board8) :
  board4.size = 4 ∧ 
  board8.size = 8 ∧ 
  sel4.count = 12 ∧ 
  sel4.row_count = 3 ∧
  sel8.count = 56 ∧
  sel8.row_count = 7 →
  count_8x8_selections board8 sel8 = (count_4x4_selections board4 sel4) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_chessboard_selections_theorem_l1932_193219


namespace NUMINAMATH_CALUDE_james_marbles_distribution_l1932_193256

theorem james_marbles_distribution (initial_marbles : ℕ) (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 28)
  (h2 : remaining_marbles = 21)
  (h3 : initial_marbles > remaining_marbles) :
  ∃ (num_bags : ℕ), 
    num_bags > 1 ∧ 
    (initial_marbles - remaining_marbles) * num_bags = initial_marbles ∧
    num_bags = 4 := by
  sorry

end NUMINAMATH_CALUDE_james_marbles_distribution_l1932_193256


namespace NUMINAMATH_CALUDE_base4_arithmetic_theorem_l1932_193290

/-- Converts a number from base 4 to base 10 -/
def base4To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10To4 (n : ℕ) : ℕ := sorry

/-- Performs arithmetic operations in base 4 -/
def base4Arithmetic (a b c d : ℕ) : ℕ := 
  let a10 := base4To10 a
  let b10 := base4To10 b
  let c10 := base4To10 c
  let d10 := base4To10 d
  base10To4 (a10 + b10 * c10 / d10)

theorem base4_arithmetic_theorem : 
  base4Arithmetic 231 21 12 3 = 333 := by sorry

end NUMINAMATH_CALUDE_base4_arithmetic_theorem_l1932_193290


namespace NUMINAMATH_CALUDE_ten_lines_intersections_l1932_193214

/-- The number of intersections formed by n straight lines where no two lines are parallel
    and no three lines intersect at a single point. -/
def intersections (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else (n - 1) * (n - 2) / 2

/-- Theorem stating that 10 straight lines under the given conditions form 45 intersections -/
theorem ten_lines_intersections :
  intersections 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_lines_intersections_l1932_193214


namespace NUMINAMATH_CALUDE_sum_seven_multiples_of_12_l1932_193233

/-- Sum of arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The sum of the first seven multiples of 12 -/
theorem sum_seven_multiples_of_12 :
  arithmetic_sum 12 12 7 = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_multiples_of_12_l1932_193233


namespace NUMINAMATH_CALUDE_season_games_first_part_l1932_193294

theorem season_games_first_part 
  (total_games : ℕ) 
  (win_rate_first : ℚ) 
  (win_rate_second : ℚ) 
  (win_rate_overall : ℚ) :
  total_games = 125 →
  win_rate_first = 3/4 →
  win_rate_second = 1/2 →
  win_rate_overall = 7/10 →
  ∃ (first_part : ℕ),
    first_part = 100 ∧
    win_rate_first * first_part + win_rate_second * (total_games - first_part) = 
      win_rate_overall * total_games :=
by sorry

end NUMINAMATH_CALUDE_season_games_first_part_l1932_193294


namespace NUMINAMATH_CALUDE_max_trig_fraction_l1932_193296

theorem max_trig_fraction (x : ℝ) : 
  (Real.sin x)^4 + (Real.cos x)^4 ≤ (Real.sin x)^2 + (Real.cos x)^2 + 2*(Real.sin x)^2*(Real.cos x)^2 := by
  sorry

#check max_trig_fraction

end NUMINAMATH_CALUDE_max_trig_fraction_l1932_193296


namespace NUMINAMATH_CALUDE_value_of_b_minus_d_squared_l1932_193237

theorem value_of_b_minus_d_squared 
  (a b c d : ℤ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 3) : 
  (b - d)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_minus_d_squared_l1932_193237


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1932_193240

theorem sphere_surface_area (r : ℝ) (h : r > 0) :
  let plane_distance : ℝ := 3
  let section_area : ℝ := 16 * Real.pi
  let section_radius : ℝ := (section_area / Real.pi).sqrt
  r * r = plane_distance * plane_distance + section_radius * section_radius →
  4 * Real.pi * r * r = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1932_193240


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_range_l1932_193298

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| ≤ 1}
def B : Set ℝ := {x : ℝ | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_empty_implies_a_range (a : ℝ) : A a ∩ B = ∅ → 2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_range_l1932_193298


namespace NUMINAMATH_CALUDE_min_score_to_tie_record_l1932_193271

/-- Proves that the minimum average score per player in the final round to tie the league record
    is 12.5833 points less than the current league record. -/
theorem min_score_to_tie_record (
  league_record : ℝ)
  (team_size : ℕ)
  (season_length : ℕ)
  (current_score : ℝ)
  (bonus_points : ℕ)
  (h1 : league_record = 287.5)
  (h2 : team_size = 6)
  (h3 : season_length = 12)
  (h4 : current_score = 19350.5)
  (h5 : bonus_points = 300)
  : ∃ (min_score : ℝ), 
    league_record - min_score = 12.5833 ∧ 
    min_score * team_size + current_score + bonus_points = league_record * (team_size * season_length) :=
by
  sorry

end NUMINAMATH_CALUDE_min_score_to_tie_record_l1932_193271


namespace NUMINAMATH_CALUDE_inscribed_square_and_circle_dimensions_l1932_193211

-- Define the right triangle DEF
def triangle_DEF (DE EF DF : ℝ) : Prop :=
  DE = 5 ∧ EF = 12 ∧ DF = 13 ∧ DE ^ 2 + EF ^ 2 = DF ^ 2

-- Define the inscribed square PQRS
def inscribed_square (s : ℝ) (DE EF DF : ℝ) : Prop :=
  triangle_DEF DE EF DF ∧
  ∃ (P Q R S : ℝ × ℝ),
    -- P and Q on DF, R on DE, S on EF
    (P.1 + Q.1 = DF) ∧ (R.2 = DE) ∧ (S.1 = EF) ∧
    -- PQRS is a square with side length s
    (Q.1 - P.1 = s) ∧ (R.2 - Q.2 = s) ∧ (S.1 - R.1 = s) ∧ (P.2 - S.2 = s)

-- Define the inscribed circle
def inscribed_circle (r : ℝ) (s : ℝ) : Prop :=
  r = s / 2

-- Theorem statement
theorem inscribed_square_and_circle_dimensions :
  ∀ (DE EF DF s r : ℝ),
    inscribed_square s DE EF DF →
    inscribed_circle r s →
    s = 780 / 169 ∧ r = 390 / 338 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_and_circle_dimensions_l1932_193211


namespace NUMINAMATH_CALUDE_shortest_distance_to_circle_l1932_193225

def circle_equation (x y : ℝ) : Prop := (x - 8)^2 + (y - 7)^2 = 25

def point : ℝ × ℝ := (1, -2)

def center : ℝ × ℝ := (8, 7)

def radius : ℝ := 5

theorem shortest_distance_to_circle :
  let d := Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2) - radius
  d = Real.sqrt 130 - 5 := by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_circle_l1932_193225


namespace NUMINAMATH_CALUDE_max_value_f_on_interval_range_of_a_for_inequality_l1932_193251

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a

-- Part 1: Maximum value of f(x) when a = 2 on [-1, 1]
theorem max_value_f_on_interval :
  ∃ (M : ℝ), M = 5 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f 2 x ≤ M :=
sorry

-- Part 2: Range of a for f(x)/x ≥ 2 when x ∈ [1, 2]
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (1 : ℝ) 2, f a x / x ≥ 2) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_on_interval_range_of_a_for_inequality_l1932_193251


namespace NUMINAMATH_CALUDE_remaining_files_correct_l1932_193217

/-- Calculates the number of remaining files on a flash drive -/
def remaining_files (music_files video_files deleted_files : ℕ) : ℕ :=
  music_files + video_files - deleted_files

/-- Theorem: The number of remaining files is correct given the initial conditions -/
theorem remaining_files_correct (music_files video_files deleted_files : ℕ) :
  remaining_files music_files video_files deleted_files =
  music_files + video_files - deleted_files :=
by sorry

end NUMINAMATH_CALUDE_remaining_files_correct_l1932_193217


namespace NUMINAMATH_CALUDE_circle_equation_with_diameter_l1932_193228

/-- Given points A and B, prove the equation of the circle with AB as diameter -/
theorem circle_equation_with_diameter (A B : ℝ × ℝ) (h : A = (-4, 0) ∧ B = (0, 2)) :
  ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = 5 ↔ 
  (x - A.1)^2 + (y - A.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 4 ∧
  (x - B.1)^2 + (y - B.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_with_diameter_l1932_193228


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1932_193215

/-- The length of a bridge given train parameters -/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) :
  train_length = 100 →
  crossing_time = 34.997200223982084 →
  train_speed_kmph = 36 →
  let train_speed_ms := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 249.97200223982084 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1932_193215


namespace NUMINAMATH_CALUDE_sum_of_m_and_n_is_zero_l1932_193229

theorem sum_of_m_and_n_is_zero (m n p : ℝ) 
  (h1 : m * n + p^2 + 4 = 0) 
  (h2 : m - n = 4) : 
  m + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_and_n_is_zero_l1932_193229


namespace NUMINAMATH_CALUDE_min_coefficient_value_l1932_193267

theorem min_coefficient_value (a b box : ℤ) : 
  (∀ x : ℝ, (a * x + b) * (b * x + a) = 10 * x^2 + box * x + 10) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  box ≥ 29 :=
by sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l1932_193267


namespace NUMINAMATH_CALUDE_contest_result_l1932_193255

/-- The number of baskets made by Alex, Sandra, Hector, and Jordan -/
def total_baskets (alex sandra hector jordan : ℕ) : ℕ :=
  alex + sandra + hector + jordan

/-- Theorem stating the total number of baskets under given conditions -/
theorem contest_result : ∃ (alex sandra hector jordan : ℕ),
  alex = 8 ∧
  sandra = 3 * alex ∧
  hector = 2 * sandra ∧
  jordan = (alex + sandra + hector) / 5 ∧
  total_baskets alex sandra hector jordan = 96 := by
  sorry

end NUMINAMATH_CALUDE_contest_result_l1932_193255


namespace NUMINAMATH_CALUDE_copper_zinc_ratio_l1932_193272

/-- Given a mixture of copper and zinc, prove that the ratio of copper to zinc is 77:63 -/
theorem copper_zinc_ratio (total_weight zinc_weight : ℝ)
  (h_total : total_weight = 70)
  (h_zinc : zinc_weight = 31.5)
  : ∃ (a b : ℕ), a = 77 ∧ b = 63 ∧ (total_weight - zinc_weight) / zinc_weight = a / b := by
  sorry

end NUMINAMATH_CALUDE_copper_zinc_ratio_l1932_193272


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_9_l1932_193276

def number_set : Finset ℕ := {1, 3, 5, 7, 9}

def sum_greater_than_9 (a b : ℕ) : Prop := a + b > 9

def valid_pair (a b : ℕ) : Prop := a ∈ number_set ∧ b ∈ number_set ∧ a ≠ b

theorem probability_sum_greater_than_9 :
  Nat.card {p : ℕ × ℕ | p.1 < p.2 ∧ valid_pair p.1 p.2 ∧ sum_greater_than_9 p.1 p.2} /
  Nat.card {p : ℕ × ℕ | p.1 < p.2 ∧ valid_pair p.1 p.2} = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_9_l1932_193276


namespace NUMINAMATH_CALUDE_number_multiplied_by_15_l1932_193220

theorem number_multiplied_by_15 :
  ∃ x : ℝ, x * 15 = 150 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_number_multiplied_by_15_l1932_193220


namespace NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l1932_193245

theorem largest_whole_number_nine_times_less_than_150 : 
  ∃ (x : ℕ), x = 16 ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l1932_193245


namespace NUMINAMATH_CALUDE_min_coins_for_eternal_collection_l1932_193291

/-- Represents the JMO kingdom with its citizens and coin distribution. -/
structure Kingdom (n : ℕ) where
  /-- The number of citizens in the kingdom is 2^n. -/
  citizens : ℕ := 2^n
  /-- The value of paper bills used in the kingdom. -/
  bill_value : ℕ := 2^n
  /-- The possible values of coins in the kingdom. -/
  coin_values : List ℕ := List.range n |>.map (fun a => 2^a)

/-- The sum of digits function in base 2. -/
def sum_of_digits (a : ℕ) : ℕ := sorry

/-- Theorem stating the minimum number of coins required for the king to collect money every night eternally. -/
theorem min_coins_for_eternal_collection (n : ℕ) (h : n > 0) : 
  ∃ (S : ℕ), S = n * 2^(n-1) ∧ 
  ∀ (S' : ℕ), S' < S → ¬(∃ (distribution : ℕ → ℕ), 
    (∀ i, i < 2^n → distribution i ≤ sum_of_digits i) ∧
    (∀ t : ℕ, ∃ (new_distribution : ℕ → ℕ), 
      (∀ i, i < 2^n → new_distribution i = distribution ((i + 1) % 2^n) + 1) ∧
      (∀ i, i < 2^n → new_distribution i ≤ sum_of_digits i))) :=
sorry

end NUMINAMATH_CALUDE_min_coins_for_eternal_collection_l1932_193291


namespace NUMINAMATH_CALUDE_door_opening_probability_l1932_193223

/-- Represents the probability of opening a door on the second attempt -/
def probability_second_attempt (total_keys : ℕ) (working_keys : ℕ) (discard : Bool) : ℚ :=
  if discard then
    (working_keys : ℚ) / total_keys * working_keys / (total_keys - 1)
  else
    (working_keys : ℚ) / total_keys * working_keys / total_keys

/-- The main theorem about the probability of opening the door on the second attempt -/
theorem door_opening_probability :
  let total_keys : ℕ := 4
  let working_keys : ℕ := 2
  probability_second_attempt total_keys working_keys true = 1/3 ∧
  probability_second_attempt total_keys working_keys false = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_door_opening_probability_l1932_193223


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l1932_193279

theorem real_part_of_complex_number : 
  (1 + 2 / (Complex.I + 1)).re = 2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l1932_193279


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1932_193243

theorem inequality_system_solution_range (m : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 4 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x - m < 0 ∧ 7 - 2*x ≤ 1))) →
  (6 < m ∧ m ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1932_193243


namespace NUMINAMATH_CALUDE_vegetable_ghee_mixture_weight_l1932_193216

/-- The weight of the mixture of two brands of vegetable ghee -/
theorem vegetable_ghee_mixture_weight
  (weight_a : ℝ) (weight_b : ℝ) (ratio_a : ℝ) (ratio_b : ℝ) (total_volume : ℝ) :
  weight_a = 900 →
  weight_b = 700 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_volume = 4 →
  ((ratio_a / (ratio_a + ratio_b)) * total_volume * weight_a +
   (ratio_b / (ratio_a + ratio_b)) * total_volume * weight_b) / 1000 = 3.280 :=
by sorry

end NUMINAMATH_CALUDE_vegetable_ghee_mixture_weight_l1932_193216


namespace NUMINAMATH_CALUDE_alphametic_puzzle_solution_l1932_193299

theorem alphametic_puzzle_solution :
  ∃! (T H E B G M A : ℕ),
    T ≠ H ∧ T ≠ E ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧ T ≠ A ∧
    H ≠ E ∧ H ≠ B ∧ H ≠ G ∧ H ≠ M ∧ H ≠ A ∧
    E ≠ B ∧ E ≠ G ∧ E ≠ M ∧ E ≠ A ∧
    B ≠ G ∧ B ≠ M ∧ B ≠ A ∧
    G ≠ M ∧ G ≠ A ∧
    M ≠ A ∧
    T < 10 ∧ H < 10 ∧ E < 10 ∧ B < 10 ∧ G < 10 ∧ M < 10 ∧ A < 10 ∧
    1000 * T + 100 * H + 10 * E + T + 1000 * B + 100 * E + 10 * T + A =
    10000 * G + 1000 * A + 100 * M + 10 * M + A ∧
    T = 4 ∧ H = 9 ∧ E = 4 ∧ B = 5 ∧ G = 1 ∧ M = 8 ∧ A = 0 :=
by sorry

end NUMINAMATH_CALUDE_alphametic_puzzle_solution_l1932_193299


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1932_193234

theorem unique_solution_quadratic_inequality (p : ℝ) : 
  (∃! x : ℝ, 0 ≤ x^2 + p*x + 5 ∧ x^2 + p*x + 5 ≤ 1) → (p = 4 ∨ p = -4) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1932_193234


namespace NUMINAMATH_CALUDE_correlation_coefficient_relationship_l1932_193254

/-- The correlation coefficient type -/
def CorrelationCoefficient := { r : ℝ // -1 ≤ r ∧ r ≤ 1 }

/-- The degree of correlation between two variables -/
noncomputable def degreeOfCorrelation (r : CorrelationCoefficient) : ℝ := sorry

/-- Theorem stating the relationship between |r| and the degree of correlation -/
theorem correlation_coefficient_relationship (r1 r2 : CorrelationCoefficient) :
  (|r1.val| < |r2.val| ∧ |r2.val| ≤ 1) → degreeOfCorrelation r1 < degreeOfCorrelation r2 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_relationship_l1932_193254


namespace NUMINAMATH_CALUDE_alexas_vacation_time_l1932_193249

/-- Proves that Alexa's vacation time is 9 days given the conditions of the problem. -/
theorem alexas_vacation_time (E : ℝ) 
  (ethan_time : E > 0)
  (alexa_time : ℝ)
  (joey_time : ℝ)
  (alexa_vacation : alexa_time = 3/4 * E)
  (joey_swimming : joey_time = 1/2 * E)
  (joey_days : joey_time = 6) : 
  alexa_time = 9 := by
  sorry

end NUMINAMATH_CALUDE_alexas_vacation_time_l1932_193249


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l1932_193204

theorem sunzi_wood_measurement_problem (x y : ℝ) : 
  (x - y = 4.5 ∧ (x / 2) + 1 = y) ↔ (x - y = 4.5 ∧ y - x / 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l1932_193204


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l1932_193268

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a + b + c = 22) : 
  a*b + b*c + a*c = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l1932_193268


namespace NUMINAMATH_CALUDE_initial_bananas_count_l1932_193293

-- Define the number of bananas left on the tree
def bananas_left : ℕ := 430

-- Define the number of bananas eaten by each person
def raj_eaten : ℕ := 120
def asha_eaten : ℕ := 100
def vijay_eaten : ℕ := 80

-- Define the ratios of remaining to eaten bananas for each person
def raj_ratio : ℕ := 2
def asha_ratio : ℕ := 3
def vijay_ratio : ℕ := 4

-- Define the function to calculate the total number of bananas
def total_bananas : ℕ :=
  bananas_left +
  (raj_ratio * raj_eaten + raj_eaten) +
  (asha_ratio * asha_eaten + asha_eaten) +
  (vijay_ratio * vijay_eaten + vijay_eaten)

-- Theorem statement
theorem initial_bananas_count :
  total_bananas = 1290 :=
by sorry

end NUMINAMATH_CALUDE_initial_bananas_count_l1932_193293


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l1932_193246

theorem no_prime_roots_for_quadratic : ¬∃ k : ℤ, ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p : ℤ) + q = 57 ∧ 
  (p : ℤ) * q = k ∧
  (p : ℤ) * q = 57 * (p + q) - (p^2 + q^2) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l1932_193246


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_A_l1932_193278

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {2, 7}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {2, 4, 5, 7} := by sorry

-- Theorem for complement of A with respect to U
theorem complement_of_A : (U \ A) = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_A_l1932_193278


namespace NUMINAMATH_CALUDE_area_of_S3_l1932_193280

/-- Given a square S1 with area 16, S2 is constructed by connecting the midpoints of S1's sides,
    and S3 is constructed by connecting the midpoints of S2's sides. -/
def nested_squares (S1 S2 S3 : Real) : Prop :=
  S1 = 16 ∧ S2 = S1 / 2 ∧ S3 = S2 / 2

theorem area_of_S3 (S1 S2 S3 : Real) (h : nested_squares S1 S2 S3) : S3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_S3_l1932_193280


namespace NUMINAMATH_CALUDE_power_of_two_sum_l1932_193288

theorem power_of_two_sum : 2^3 + 2^3 + 2^3 + 2^3 = 2^5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l1932_193288


namespace NUMINAMATH_CALUDE_birth_probability_l1932_193259

theorem birth_probability (n : ℕ) (p : ℝ) (h1 : n = 5) (h2 : p = 1/2) :
  let prob_all_same := p^n
  let prob_three_two := (n.choose 3) * p^n
  let prob_four_one := 2 * (n.choose 1) * p^n
  (prob_three_two = prob_four_one) ∧ 
  (prob_three_two > prob_all_same) ∧
  (prob_four_one > prob_all_same) := by
  sorry

end NUMINAMATH_CALUDE_birth_probability_l1932_193259


namespace NUMINAMATH_CALUDE_janine_reading_theorem_l1932_193281

/-- The number of books Janine read last month -/
def books_last_month : ℕ := 5

/-- The number of books Janine read this month -/
def books_this_month : ℕ := 2 * books_last_month

/-- The number of pages in each book -/
def pages_per_book : ℕ := 10

/-- The total number of pages Janine read in two months -/
def total_pages : ℕ := (books_last_month + books_this_month) * pages_per_book

theorem janine_reading_theorem : total_pages = 150 := by
  sorry

end NUMINAMATH_CALUDE_janine_reading_theorem_l1932_193281


namespace NUMINAMATH_CALUDE_proposition_implication_l1932_193209

theorem proposition_implication (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬¬p) : 
  ¬q := by
sorry

end NUMINAMATH_CALUDE_proposition_implication_l1932_193209


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1932_193284

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ i / (1 + Real.sqrt 3 * i) = (Real.sqrt 3 / 4 : ℂ) + (1 / 4 : ℂ) * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1932_193284


namespace NUMINAMATH_CALUDE_intersection_point_in_circle_range_l1932_193207

theorem intersection_point_in_circle_range (m : ℝ) : 
  let M : ℝ × ℝ := (1, 1)
  let line1 : ℝ × ℝ → Prop := λ p => p.1 + p.2 - 2 = 0
  let line2 : ℝ × ℝ → Prop := λ p => 3 * p.1 - p.2 - 2 = 0
  let circle : ℝ × ℝ → Prop := λ p => (p.1 - m)^2 + p.2^2 < 5
  (line1 M ∧ line2 M ∧ circle M) ↔ -1 < m ∧ m < 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_in_circle_range_l1932_193207
