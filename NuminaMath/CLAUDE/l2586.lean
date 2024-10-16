import Mathlib

namespace NUMINAMATH_CALUDE_power_of_negative_product_l2586_258645

theorem power_of_negative_product (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l2586_258645


namespace NUMINAMATH_CALUDE_numeric_methods_students_count_second_year_students_count_l2586_258666

/-- The number of second-year students studying numeric methods -/
def numeric_methods_students : ℕ := 241

/-- The number of second-year students studying automatic control of airborne vehicles -/
def acav_students : ℕ := 423

/-- The number of second-year students studying both numeric methods and ACAV -/
def both_subjects_students : ℕ := 134

/-- The total number of students in the faculty -/
def total_students : ℕ := 663

/-- The proportion of second-year students in the faculty -/
def second_year_proportion : ℚ := 4/5

/-- The total number of second-year students -/
def total_second_year_students : ℕ := 530

theorem numeric_methods_students_count :
  numeric_methods_students + acav_students - both_subjects_students = total_second_year_students :=
by sorry

theorem second_year_students_count :
  total_second_year_students = (total_students : ℚ) * second_year_proportion :=
by sorry

end NUMINAMATH_CALUDE_numeric_methods_students_count_second_year_students_count_l2586_258666


namespace NUMINAMATH_CALUDE_no_real_roots_iff_k_less_than_negative_one_l2586_258673

theorem no_real_roots_iff_k_less_than_negative_one (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_k_less_than_negative_one_l2586_258673


namespace NUMINAMATH_CALUDE_tobias_driveways_shoveled_l2586_258648

/-- Calculates the number of driveways shoveled by Tobias given his income and expenses. -/
theorem tobias_driveways_shoveled 
  (original_price : ℚ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (monthly_allowance : ℚ)
  (lawn_fee : ℚ)
  (driveway_fee : ℚ)
  (hourly_wage : ℚ)
  (remaining_money : ℚ)
  (months_saved : ℕ)
  (hours_worked : ℕ)
  (lawns_mowed : ℕ)
  (h1 : original_price = 95)
  (h2 : discount_rate = 1/10)
  (h3 : tax_rate = 1/20)
  (h4 : monthly_allowance = 5)
  (h5 : lawn_fee = 15)
  (h6 : driveway_fee = 7)
  (h7 : hourly_wage = 8)
  (h8 : remaining_money = 15)
  (h9 : months_saved = 3)
  (h10 : hours_worked = 10)
  (h11 : lawns_mowed = 4) :
  ∃ (driveways_shoveled : ℕ), driveways_shoveled = 7 :=
by sorry


end NUMINAMATH_CALUDE_tobias_driveways_shoveled_l2586_258648


namespace NUMINAMATH_CALUDE_limit_x_minus_sin_x_ln_x_at_zero_l2586_258635

theorem limit_x_minus_sin_x_ln_x_at_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |(x - Real.sin x) * Real.log x| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_x_minus_sin_x_ln_x_at_zero_l2586_258635


namespace NUMINAMATH_CALUDE_max_profit_at_15_verify_conditions_l2586_258679

-- Define the relationship between price and sales quantity
def sales_quantity (x : ℤ) : ℤ := -5 * x + 150

-- Define the profit function
def profit (x : ℤ) : ℤ := sales_quantity x * (x - 8)

-- Theorem statement
theorem max_profit_at_15 :
  ∀ x : ℤ, 8 ≤ x → x ≤ 15 → profit x ≤ 525 ∧ profit 15 = 525 :=
by
  sorry

-- Verify the given conditions
theorem verify_conditions :
  sales_quantity 9 = 105 ∧ sales_quantity 11 = 95 :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_15_verify_conditions_l2586_258679


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l2586_258646

/-- Given a polynomial f(x) = ax^7 - bx^3 + cx - 5 where f(2) = 3, prove that f(-2) = -13 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^7 - b * x^3 + c * x - 5
  (f 2 = 3) → (f (-2) = -13) := by
sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l2586_258646


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l2586_258670

/-- The area of the circle represented by the polar equation r = 4cosθ - 3sinθ -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := λ θ => 4 * Real.cos θ - 3 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, (r θ * Real.cos θ - center.1)^2 + (r θ * Real.sin θ - center.2)^2 = radius^2) ∧
    π * radius^2 = 25 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l2586_258670


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2586_258689

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the set of x that satisfies the inequality
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (x^2 - 3*x - 3) < f 1}

-- Theorem statement
theorem solution_set_equivalence (f : ℝ → ℝ) (h : DecreasingFunction f) :
  SolutionSet f = {x | x < -1 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2586_258689


namespace NUMINAMATH_CALUDE_horner_method_f_at_2_l2586_258608

/-- Horner's method for polynomial evaluation -/
def horner_step (x : ℚ) (v : ℚ) (a : ℚ) : ℚ := v * x + a

/-- The polynomial f(x) = 7x^5 + 5x^4 + 3x^3 + x^2 + x + 2 -/
def f (x : ℚ) : ℚ := 7 * x^5 + 5 * x^4 + 3 * x^3 + x^2 + x + 2

/-- Theorem: Horner's method for f(x) at x = 2 yields v_3 = 83 -/
theorem horner_method_f_at_2 :
  let x : ℚ := 2
  let v₀ : ℚ := 7
  let v₁ : ℚ := horner_step x v₀ 5
  let v₂ : ℚ := horner_step x v₁ 3
  let v₃ : ℚ := horner_step x v₂ 1
  v₃ = 83 := by sorry

end NUMINAMATH_CALUDE_horner_method_f_at_2_l2586_258608


namespace NUMINAMATH_CALUDE_first_number_is_45_l2586_258628

/-- Given two positive integers with a ratio of 3:4 and LCM 180, prove the first number is 45 -/
theorem first_number_is_45 (a b : ℕ+) (h1 : a.val * 4 = b.val * 3) (h2 : Nat.lcm a.val b.val = 180) : a = 45 := by
  sorry

end NUMINAMATH_CALUDE_first_number_is_45_l2586_258628


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2586_258614

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

/-- The factorial of a non-negative integer n, denoted by n!, is the product of all positive integers less than or equal to n. -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem geometric_sequence_first_term (a : ℕ → ℚ) :
  IsGeometricSequence a →
  a 7 = factorial 8 →
  a 10 = factorial 11 →
  a 1 = 8 / 245 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2586_258614


namespace NUMINAMATH_CALUDE_opposite_of_negative_2022_l2586_258649

theorem opposite_of_negative_2022 : -((-2022 : ℤ)) = 2022 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2022_l2586_258649


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2586_258691

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2586_258691


namespace NUMINAMATH_CALUDE_product_41_reciprocal_squares_sum_l2586_258604

theorem product_41_reciprocal_squares_sum :
  ∀ a b : ℕ+,
  (a.val : ℕ) * (b.val : ℕ) = 41 →
  (1 : ℚ) / (a.val^2 : ℚ) + (1 : ℚ) / (b.val^2 : ℚ) = 1682 / 1681 :=
by sorry

end NUMINAMATH_CALUDE_product_41_reciprocal_squares_sum_l2586_258604


namespace NUMINAMATH_CALUDE_reading_homework_pages_isabel_homework_l2586_258663

theorem reading_homework_pages (math_pages : ℕ) (problems_per_page : ℕ) (total_problems : ℕ) : ℕ :=
  let reading_pages := (total_problems - math_pages * problems_per_page) / problems_per_page
  reading_pages

theorem isabel_homework :
  reading_homework_pages 2 5 30 = 4 := by
  sorry

end NUMINAMATH_CALUDE_reading_homework_pages_isabel_homework_l2586_258663


namespace NUMINAMATH_CALUDE_m_range_l2586_258615

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 + m * x + 1 > 0

def q (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  ∀ x y : ℝ, x^2 / (m - 1) + y^2 / (m - 2) = 1 ↔ 
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the set of m values
def M : Set ℝ := {m : ℝ | (0 ≤ m ∧ m ≤ 1) ∨ (2 ≤ m ∧ m < 4)}

-- State the theorem
theorem m_range : 
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ ∀ m : ℝ, m ∈ M :=
sorry

end NUMINAMATH_CALUDE_m_range_l2586_258615


namespace NUMINAMATH_CALUDE_novel_writing_rate_l2586_258607

/-- Represents the writing rate of an author -/
def writing_rate (total_words : ℕ) (writing_hours : ℕ) : ℚ :=
  total_words / writing_hours

/-- Proves that the writing rate for a 60,000-word novel written in 100 hours is 600 words per hour -/
theorem novel_writing_rate :
  writing_rate 60000 100 = 600 := by
  sorry

end NUMINAMATH_CALUDE_novel_writing_rate_l2586_258607


namespace NUMINAMATH_CALUDE_zhou_yu_age_theorem_l2586_258676

/-- Represents the equation for Zhou Yu's age at death -/
def zhou_yu_age_equation (x : ℕ) : Prop :=
  x^2 = 10 * (x - 3) + x

/-- Theorem stating the conditions and the equation for Zhou Yu's age at death -/
theorem zhou_yu_age_theorem (x : ℕ) :
  (x ≥ 10 ∧ x < 100) →  -- Two-digit number
  (x / 10 = x % 10 - 3) →  -- Tens digit is 3 less than units digit
  (x^2 = 10 * (x - 3) + x) →  -- Square of units digit equals the age
  zhou_yu_age_equation x :=
by
  sorry

#check zhou_yu_age_theorem

end NUMINAMATH_CALUDE_zhou_yu_age_theorem_l2586_258676


namespace NUMINAMATH_CALUDE_classroom_pencils_l2586_258634

/-- The number of pencils a teacher needs to give out to a classroom of students -/
def pencils_to_give_out (num_students : ℕ) (dozens_per_student : ℕ) : ℕ :=
  num_students * (dozens_per_student * 12)

/-- Theorem: Given 46 children in a classroom, with each child receiving 4 dozen pencils,
    the total number of pencils the teacher needs to give out is 2208 -/
theorem classroom_pencils : pencils_to_give_out 46 4 = 2208 := by
  sorry

end NUMINAMATH_CALUDE_classroom_pencils_l2586_258634


namespace NUMINAMATH_CALUDE_sine_addition_formula_l2586_258621

theorem sine_addition_formula (α β : Real) : 
  Real.sin (α - β) * Real.cos β + Real.cos (α - β) * Real.sin β = Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_sine_addition_formula_l2586_258621


namespace NUMINAMATH_CALUDE_prime_divisibility_condition_l2586_258665

theorem prime_divisibility_condition (p : ℕ) (x : ℕ) :
  Prime p →
  1 ≤ x ∧ x ≤ 2 * p →
  (x^(p-1) ∣ (p-1)^x + 1) ↔ 
  ((p = 2 ∧ x = 2) ∨ (p = 3 ∧ x = 3) ∨ (x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_prime_divisibility_condition_l2586_258665


namespace NUMINAMATH_CALUDE_largest_sum_l2586_258669

theorem largest_sum : 
  let expr1 := (1/4 : ℚ) + (1/5 : ℚ) * (1/2 : ℚ)
  let expr2 := (1/4 : ℚ) - (1/6 : ℚ)
  let expr3 := (1/4 : ℚ) + (1/3 : ℚ) * (1/2 : ℚ)
  let expr4 := (1/4 : ℚ) - (1/8 : ℚ)
  let expr5 := (1/4 : ℚ) + (1/7 : ℚ) * (1/2 : ℚ)
  expr3 = (5/12 : ℚ) ∧ 
  expr3 > expr1 ∧ 
  expr3 > expr2 ∧ 
  expr3 > expr4 ∧ 
  expr3 > expr5 :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_l2586_258669


namespace NUMINAMATH_CALUDE_derivative_sin_cos_product_l2586_258631

open Real

theorem derivative_sin_cos_product (x : ℝ) : 
  deriv (λ x => sin x * cos x) x = cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_product_l2586_258631


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l2586_258656

theorem quadratic_roots_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hroots : a^2 - c * a^2 + c = 0 ∧ b^2 - c * b^2 + c = 0) :
  (a * Real.sqrt (1 - 1 / b^2) + b * Real.sqrt (1 - 1 / a^2) = 2) ∨
  (a * Real.sqrt (1 - 1 / b^2) + b * Real.sqrt (1 - 1 / a^2) = -2) ∨
  (a * Real.sqrt (1 - 1 / b^2) + b * Real.sqrt (1 - 1 / a^2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l2586_258656


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l2586_258606

def boat_speed : ℝ := 20
def current_speed : ℝ := 4
def distance : ℝ := 2

theorem boat_speed_ratio :
  let downstream_speed := boat_speed + current_speed
  let upstream_speed := boat_speed - current_speed
  let downstream_time := distance / downstream_speed
  let upstream_time := distance / upstream_speed
  let total_time := downstream_time + upstream_time
  let total_distance := 2 * distance
  let average_speed := total_distance / total_time
  (average_speed / boat_speed) = 24 / 25 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l2586_258606


namespace NUMINAMATH_CALUDE_oldest_child_age_l2586_258600

theorem oldest_child_age (a b c d : ℕ) : 
  a = 6 ∧ b = 9 ∧ c = 12 ∧ (a + b + c + d : ℚ) / 4 = 9 → d = 9 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l2586_258600


namespace NUMINAMATH_CALUDE_magic_square_sum_l2586_258609

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e f g h i : ℝ)
  (row_sum : a + b + c = d + e + f ∧ d + e + f = g + h + i)
  (col_sum : a + d + g = b + e + h ∧ b + e + h = c + f + i)
  (diag_sum : a + e + i = c + e + g)

/-- The sum of all elements in a magic square -/
def total_sum (m : MagicSquare) : ℝ := m.a + m.b + m.c + m.d + m.e + m.f + m.g + m.h + m.i

/-- Theorem: Sum of remaining squares in a specific magic square -/
theorem magic_square_sum :
  ∀ (m : MagicSquare),
    m.b = 7 ∧ m.c = 2018 ∧ m.g = 4 →
    (total_sum m) - (m.b + m.c + m.g) = -11042.5 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_sum_l2586_258609


namespace NUMINAMATH_CALUDE_total_routes_to_school_l2586_258657

theorem total_routes_to_school (bus_routes subway_routes : ℕ) 
  (h1 : bus_routes = 3) 
  (h2 : subway_routes = 2) : 
  bus_routes + subway_routes = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_routes_to_school_l2586_258657


namespace NUMINAMATH_CALUDE_race_car_count_l2586_258601

theorem race_car_count (p_x p_y p_z p_combined : ℚ) (h1 : p_x = 1 / 7)
    (h2 : p_y = 1 / 3) (h3 : p_z = 1 / 5)
    (h4 : p_combined = p_x + p_y + p_z)
    (h5 : p_combined = 71 / 105) : ∃ n : ℕ, n = 105 ∧ p_x = 1 / n := by
  sorry

end NUMINAMATH_CALUDE_race_car_count_l2586_258601


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_fourth_power_l2586_258692

theorem consecutive_odd_sum_fourth_power (a b c : ℕ) : 
  (∃ n : ℕ, n < 10 ∧ a + b + c = n^4) ∧ 
  (Odd a ∧ Odd b ∧ Odd c) ∧
  (b = a + 2 ∧ c = b + 2) →
  ((a, b, c) = (25, 27, 29) ∨ (a, b, c) = (2185, 2187, 2189)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_fourth_power_l2586_258692


namespace NUMINAMATH_CALUDE_tunnel_length_is_1200_l2586_258671

/-- Calculates the length of a tunnel given train specifications and crossing times. -/
def tunnel_length (train_length platform_length : ℝ) 
                  (tunnel_time platform_time : ℝ) : ℝ :=
  3 * (train_length + platform_length) - train_length

/-- Proves that the tunnel length is 1200 meters given the specified conditions. -/
theorem tunnel_length_is_1200 :
  tunnel_length 330 180 45 15 = 1200 := by
  sorry

#eval tunnel_length 330 180 45 15

end NUMINAMATH_CALUDE_tunnel_length_is_1200_l2586_258671


namespace NUMINAMATH_CALUDE_max_value_of_function_l2586_258619

theorem max_value_of_function (x : Real) (h : x ∈ Set.Ioo 0 Real.pi) :
  (2 * Real.sin (x / 2) * (1 - Real.sin (x / 2)) * (1 + Real.sin (x / 2))^2) ≤ (107 + 51 * Real.sqrt 17) / 256 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2586_258619


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l2586_258662

theorem rectangular_field_perimeter : 
  let width : ℝ := 90
  let length : ℝ := (7 / 5) * width
  let perimeter : ℝ := 2 * (length + width)
  perimeter = 432 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l2586_258662


namespace NUMINAMATH_CALUDE_expression_equals_24_l2586_258633

/-- An arithmetic expression using integers and basic operators -/
inductive Expr where
  | const : Int → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression -/
def eval : Expr → Int
  | Expr.const n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Check if an expression uses each of the given numbers exactly once -/
def usesNumbers (e : Expr) (nums : List Int) : Bool := sorry

/-- There exists an arithmetic expression using 1, 4, 7, and 7 that evaluates to 24 -/
theorem expression_equals_24 : ∃ e : Expr, 
  usesNumbers e [1, 4, 7, 7] ∧ eval e = 24 := by sorry

end NUMINAMATH_CALUDE_expression_equals_24_l2586_258633


namespace NUMINAMATH_CALUDE_allyns_june_expenses_l2586_258683

/-- Calculates the total monthly electricity expenses for a given number of bulbs --/
def calculate_monthly_expenses (
  bulb_wattage : ℕ)  -- Wattage of each bulb
  (num_bulbs : ℕ)    -- Number of bulbs
  (days_in_month : ℕ) -- Number of days in the month
  (cost_per_watt : ℚ) -- Cost per watt in dollars
  : ℚ :=
  (bulb_wattage * num_bulbs * days_in_month : ℚ) * cost_per_watt

/-- Theorem stating that Allyn's monthly electricity expenses for June are $14400 --/
theorem allyns_june_expenses :
  calculate_monthly_expenses 60 40 30 (20 / 100) = 14400 := by
  sorry

end NUMINAMATH_CALUDE_allyns_june_expenses_l2586_258683


namespace NUMINAMATH_CALUDE_class_composition_l2586_258694

theorem class_composition (boys_avg : ℝ) (girls_avg : ℝ) (class_avg : ℝ) :
  boys_avg = 4 →
  girls_avg = 3.25 →
  class_avg = 3.6 →
  ∃ (boys girls : ℕ),
    boys + girls > 30 ∧
    boys + girls < 50 ∧
    (boys_avg * boys + girls_avg * girls) / (boys + girls) = class_avg ∧
    boys = 21 ∧
    girls = 24 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l2586_258694


namespace NUMINAMATH_CALUDE_supermarket_max_profit_l2586_258654

/-- Represents the daily profit function for the supermarket -/
def daily_profit (x : ℝ) : ℝ := -10 * x^2 + 1100 * x - 28000

/-- The maximum daily profit achievable by the supermarket -/
def max_profit : ℝ := 2250

theorem supermarket_max_profit :
  ∃ (x : ℝ), daily_profit x = max_profit ∧
  ∀ (y : ℝ), daily_profit y ≤ max_profit := by
  sorry

#check supermarket_max_profit

end NUMINAMATH_CALUDE_supermarket_max_profit_l2586_258654


namespace NUMINAMATH_CALUDE_smaller_partner_profit_theorem_l2586_258677

/-- Represents a partnership between two individuals -/
structure Partnership where
  investment_ratio : ℚ  -- Ratio of investments (larger / smaller)
  time_ratio : ℚ        -- Ratio of investment periods (longer / shorter)
  total_profit : ℕ      -- Total profit in rupees

/-- Calculates the profit of the partner with the smaller investment -/
def smaller_partner_profit (p : Partnership) : ℚ :=
  p.total_profit * (1 / (1 + p.investment_ratio * p.time_ratio))

/-- Theorem stating the profit of the partner with smaller investment -/
theorem smaller_partner_profit_theorem (p : Partnership) 
  (h1 : p.investment_ratio = 3)
  (h2 : p.time_ratio = 2)
  (h3 : p.total_profit = 35000) :
  ⌊smaller_partner_profit p⌋ = 5000 := by
  sorry

#eval ⌊smaller_partner_profit ⟨3, 2, 35000⟩⌋

end NUMINAMATH_CALUDE_smaller_partner_profit_theorem_l2586_258677


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2586_258652

theorem square_area_from_diagonal (d : ℝ) (h : d = 2) : 
  (d^2 / 2) = 2 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2586_258652


namespace NUMINAMATH_CALUDE_min_sum_given_product_l2586_258603

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 3 → (∀ x y : ℝ, x > 0 → y > 0 → x * y = x + y + 3 → a + b ≤ x + y) → a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l2586_258603


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_l2586_258687

/-- Given an equilateral triangle with side length 30 cm and a square with the same perimeter,
    the area of the square is 506.25 cm^2. -/
theorem square_area_equal_perimeter (triangle_side : ℝ) (square_side : ℝ) : 
  triangle_side = 30 →
  3 * triangle_side = 4 * square_side →
  square_side^2 = 506.25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_l2586_258687


namespace NUMINAMATH_CALUDE_sugar_weighing_l2586_258617

theorem sugar_weighing (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p ≠ q) :
  p / q + q / p > 2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_weighing_l2586_258617


namespace NUMINAMATH_CALUDE_age_problem_l2586_258698

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = 3 * a →
  a + b + c + d = 72 →
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2586_258698


namespace NUMINAMATH_CALUDE_blanket_average_price_l2586_258650

theorem blanket_average_price : 
  let blanket_group1 := (3, 100)
  let blanket_group2 := (5, 150)
  let blanket_group3 := (2, 570)
  let total_blankets := blanket_group1.1 + blanket_group2.1 + blanket_group3.1
  let total_cost := blanket_group1.1 * blanket_group1.2 + blanket_group2.1 * blanket_group2.2 + blanket_group3.1 * blanket_group3.2
  total_cost / total_blankets = 219 := by
sorry

end NUMINAMATH_CALUDE_blanket_average_price_l2586_258650


namespace NUMINAMATH_CALUDE_parallel_transitivity_l2586_258672

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define the parallel relation between lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l2586_258672


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2586_258629

theorem arithmetic_expression_equality : 5 * 7 - (3 * 2 + 5 * 4) / 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2586_258629


namespace NUMINAMATH_CALUDE_probability_at_least_two_liking_chi_square_association_l2586_258688

-- Define the total number of students and their preferences
def total_students : ℕ := 200
def students_liking : ℕ := 140
def students_disliking : ℕ := 60

-- Define the gender-based preferences
def male_liking : ℕ := 60
def male_disliking : ℕ := 40
def female_liking : ℕ := 80
def female_disliking : ℕ := 20

-- Define the significance level
def alpha : ℝ := 0.005

-- Define the critical value for α = 0.005
def critical_value : ℝ := 7.879

-- Theorem 1: Probability of selecting at least 2 students who like employment
theorem probability_at_least_two_liking :
  (Nat.choose 3 2 * (students_liking / total_students)^2 * (students_disliking / total_students) +
   (students_liking / total_students)^3) = 98 / 125 := by sorry

-- Theorem 2: Chi-square test for association between intention and gender
theorem chi_square_association :
  let n := total_students
  let a := male_liking
  let b := male_disliking
  let c := female_liking
  let d := female_disliking
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) > critical_value := by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_liking_chi_square_association_l2586_258688


namespace NUMINAMATH_CALUDE_sandy_jessica_marble_ratio_l2586_258644

/-- The number of marbles in a dozen -/
def marbles_per_dozen : ℕ := 12

/-- The number of dozens of red marbles Jessica has -/
def jessica_dozens : ℕ := 3

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := 144

/-- The ratio of Sandy's red marbles to Jessica's red marbles -/
def marble_ratio : ℚ := sandy_marbles / (jessica_dozens * marbles_per_dozen)

theorem sandy_jessica_marble_ratio :
  marble_ratio = 4 := by sorry

end NUMINAMATH_CALUDE_sandy_jessica_marble_ratio_l2586_258644


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l2586_258637

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l2586_258637


namespace NUMINAMATH_CALUDE_equation_solutions_l2586_258636

-- Define the equation
def equation (x : ℂ) : Prop :=
  (x - 2)^4 + (x - 6)^4 = 32

-- State the theorem
theorem equation_solutions :
  ∀ x : ℂ, equation x ↔ (x = 4 ∨ x = 4 + 2*Complex.I*Real.sqrt 6 ∨ x = 4 - 2*Complex.I*Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2586_258636


namespace NUMINAMATH_CALUDE_rectangle_division_perimeter_paradox_l2586_258682

/-- A rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculate the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Predicate to check if a real number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Theorem stating that there exists a rectangle with non-integer perimeter
    that can be divided into rectangles with integer perimeters -/
theorem rectangle_division_perimeter_paradox :
  ∃ (big : Rectangle) (small1 small2 : Rectangle),
    ¬isInteger big.perimeter ∧
    isInteger small1.perimeter ∧
    isInteger small2.perimeter ∧
    big.width = small1.width ∧
    big.width = small2.width ∧
    big.height = small1.height + small2.height :=
sorry

end NUMINAMATH_CALUDE_rectangle_division_perimeter_paradox_l2586_258682


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l2586_258684

/-- A right triangle with consecutive even number legs and hypotenuse 34 has leg sum 50 -/
theorem right_triangle_leg_sum (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 34 →  -- Hypotenuse is 34
  ∃ k : ℕ, a = 2*k ∧ b = 2*k + 2 →  -- Legs are consecutive even numbers
  a + b = 50 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l2586_258684


namespace NUMINAMATH_CALUDE_digit_250_of_13_over_17_is_8_l2586_258693

/-- The 250th decimal digit of 13/17 -/
def digit_250_of_13_over_17 : ℕ :=
  let decimal_expansion := (13 : ℚ) / 17
  let period := 16
  let position_in_period := 250 % period
  8

/-- Theorem: The 250th decimal digit in the decimal representation of 13/17 is 8 -/
theorem digit_250_of_13_over_17_is_8 :
  digit_250_of_13_over_17 = 8 := by
  sorry

end NUMINAMATH_CALUDE_digit_250_of_13_over_17_is_8_l2586_258693


namespace NUMINAMATH_CALUDE_circle_touches_angle_sides_l2586_258620

-- Define the angle
def Angle : Type := sorry

-- Define a circle
structure Circle (α : Type) where
  center : α
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the property of a circle touching a line
def touches_line (c : Circle Point) (l : Set Point) : Prop := sorry

-- Define the property of two circles touching each other
def circles_touch (c1 c2 : Circle Point) : Prop := sorry

-- Define the circle with diameter AB
def circle_with_diameter (A B : Point) : Circle Point := sorry

-- Define the sides of an angle
def sides_of_angle (a : Angle) : Set (Set Point) := sorry

theorem circle_touches_angle_sides 
  (θ : Angle) 
  (A B : Point) 
  (c1 c2 : Circle Point) 
  (h1 : c1.center = A)
  (h2 : c2.center = B)
  (h3 : ∀ s ∈ sides_of_angle θ, touches_line c1 s ∧ touches_line c2 s)
  (h4 : circles_touch c1 c2) :
  ∀ s ∈ sides_of_angle θ, touches_line (circle_with_diameter A B) s := by
  sorry

end NUMINAMATH_CALUDE_circle_touches_angle_sides_l2586_258620


namespace NUMINAMATH_CALUDE_new_girl_weight_l2586_258641

/-- Given a group of 25 girls, if replacing a 55 kg girl with a new girl increases
    the average weight by 1 kg, then the new girl weighs 80 kg. -/
theorem new_girl_weight (W : ℝ) (x : ℝ) : 
  (W / 25 + 1 = (W - 55 + x) / 25) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_new_girl_weight_l2586_258641


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2586_258632

/-- An arithmetic sequence of integers -/
def arithmeticSequence (b : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) :
  arithmeticSequence b d →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 5 * b 6 = 21 →
  b 4 * b 7 = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2586_258632


namespace NUMINAMATH_CALUDE_range_of_m_l2586_258638

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*(m+1)*x + m*(m+1) > 0

theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m > 2 ∨ (-2 ≤ m ∧ m < -1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2586_258638


namespace NUMINAMATH_CALUDE_log_product_equals_one_third_l2586_258695

theorem log_product_equals_one_third :
  Real.log 2 / Real.log 3 *
  Real.log 3 / Real.log 4 *
  Real.log 4 / Real.log 5 *
  Real.log 5 / Real.log 6 *
  Real.log 6 / Real.log 7 *
  Real.log 7 / Real.log 8 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_one_third_l2586_258695


namespace NUMINAMATH_CALUDE_sock_selection_l2586_258658

theorem sock_selection (n : ℕ) : 
  (Nat.choose 10 n = 90) → n = 2 := by
sorry

end NUMINAMATH_CALUDE_sock_selection_l2586_258658


namespace NUMINAMATH_CALUDE_sequence_existence_l2586_258655

theorem sequence_existence : ∃ (a b : ℕ → ℕ), 
  (∀ n : ℕ, n ≥ 1 → (
    (0 < a n ∧ a n < a (n + 1)) ∧
    (a n < b n ∧ b n < a n ^ 2) ∧
    ((b n - 1) % (a n - 1) = 0) ∧
    ((b n ^ 2 - 1) % (a n ^ 2 - 1) = 0)
  )) := by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_l2586_258655


namespace NUMINAMATH_CALUDE_b_47_mod_49_l2586_258699

/-- Definition of the sequence b_n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The remainder of b_47 when divided by 49 is 14 -/
theorem b_47_mod_49 : b 47 % 49 = 14 := by
  sorry

end NUMINAMATH_CALUDE_b_47_mod_49_l2586_258699


namespace NUMINAMATH_CALUDE_tennis_ball_order_l2586_258625

theorem tennis_ball_order (white yellow : ℕ) (h1 : white = yellow)
  (h2 : (white : ℚ) / ((yellow : ℚ) + 20) = 8 / 13) :
  white + yellow = 64 := by
  sorry

end NUMINAMATH_CALUDE_tennis_ball_order_l2586_258625


namespace NUMINAMATH_CALUDE_roses_cost_l2586_258680

theorem roses_cost (dozen : ℕ) (price_per_rose : ℚ) (discount_rate : ℚ) : 
  dozen * 12 * price_per_rose * discount_rate = 288 :=
by
  -- Assuming dozen = 5, price_per_rose = 6, and discount_rate = 0.8
  sorry

#check roses_cost

end NUMINAMATH_CALUDE_roses_cost_l2586_258680


namespace NUMINAMATH_CALUDE_equations_represent_parabola_and_ellipse_l2586_258602

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equations mx + ny² = 0 and mx² + ny² = 1 -/
def Equations (m n : ℝ) : Prop :=
  m ≠ 0 ∧ n ≠ 0 ∧
  ∃ (p : Point), m * p.x + n * p.y^2 = 0 ∧ m * p.x^2 + n * p.y^2 = 1

/-- Represents a parabola opening to the right -/
def ParabolaOpeningRight (m n : ℝ) : Prop :=
  m < 0 ∧ n > 0 ∧
  ∀ (p : Point), m * p.x + n * p.y^2 = 0 → p.x = -n / m * p.y^2

/-- Represents an ellipse centered at the origin -/
def Ellipse (m n : ℝ) : Prop :=
  m ≠ 0 ∧ n ≠ 0 ∧
  ∀ (p : Point), m * p.x^2 + n * p.y^2 = 1

/-- Theorem stating that the equations represent a parabola opening right and an ellipse -/
theorem equations_represent_parabola_and_ellipse (m n : ℝ) :
  Equations m n → ParabolaOpeningRight m n ∧ Ellipse m n :=
by sorry

end NUMINAMATH_CALUDE_equations_represent_parabola_and_ellipse_l2586_258602


namespace NUMINAMATH_CALUDE_exam_score_problem_l2586_258627

theorem exam_score_problem (mean : ℝ) (high_score : ℝ) (std_dev : ℝ) :
  mean = 74 ∧ high_score = 98 ∧ high_score = mean + 3 * std_dev →
  mean - 2 * std_dev = 58 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l2586_258627


namespace NUMINAMATH_CALUDE_pool_capacity_l2586_258681

/-- The capacity of a swimming pool given specific valve filling rates. -/
theorem pool_capacity 
  (fill_time : ℝ) 
  (valve_a_time : ℝ) 
  (valve_b_time : ℝ) 
  (valve_c_rate_diff : ℝ) 
  (valve_b_rate_diff : ℝ) 
  (h1 : fill_time = 40) 
  (h2 : valve_a_time = 180) 
  (h3 : valve_b_time = 240) 
  (h4 : valve_c_rate_diff = 75) 
  (h5 : valve_b_rate_diff = 60) : 
  ∃ T : ℝ, T = 16200 ∧ 
    T / fill_time = T / valve_a_time + T / valve_b_time + (T / valve_a_time + valve_c_rate_diff) :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_l2586_258681


namespace NUMINAMATH_CALUDE_license_plate_increase_l2586_258616

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^3
  let new_plates := 26^3 * 10^3 * 5
  new_plates / old_plates = 130 := by sorry

end NUMINAMATH_CALUDE_license_plate_increase_l2586_258616


namespace NUMINAMATH_CALUDE_display_configurations_l2586_258653

/-- The number of holes in the row -/
def num_holes : ℕ := 8

/-- The number of holes that can display at a time -/
def num_display : ℕ := 3

/-- The number of possible states for each displaying hole -/
def num_states : ℕ := 2

/-- A function that calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of possible configurations -/
def total_configurations : ℕ := sorry

theorem display_configurations :
  total_configurations = choose (num_holes - num_display + 1) num_display * num_states ^ num_display :=
sorry

end NUMINAMATH_CALUDE_display_configurations_l2586_258653


namespace NUMINAMATH_CALUDE_common_factor_l2586_258696

def expression (m n : ℕ) : ℤ := 4 * m^3 * n - 9 * m * n^3

theorem common_factor (m n : ℕ) : 
  ∃ (k : ℤ), expression m n = m * n * k ∧ 
  ¬∃ (l : ℤ), l ≠ 1 ∧ l ≠ -1 ∧ 
  ∃ (p : ℤ), expression m n = (m * n * l) * p :=
sorry

end NUMINAMATH_CALUDE_common_factor_l2586_258696


namespace NUMINAMATH_CALUDE_tangent_slope_x_squared_at_one_l2586_258675

theorem tangent_slope_x_squared_at_one : 
  let f : ℝ → ℝ := fun x ↦ x^2
  (deriv f) 1 = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_x_squared_at_one_l2586_258675


namespace NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_cube_l2586_258651

/-- The volume of a sphere with surface area equal to a cube of side length 2 -/
theorem sphere_volume_equal_surface_area_cube (r : ℝ) : 
  (4 * Real.pi * r^2 = 6 * 2^2) → 
  ((4 / 3) * Real.pi * r^3 = (8 * Real.sqrt 6) / Real.sqrt Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_cube_l2586_258651


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l2586_258622

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

-- Define the points
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (4, 0)
def point_C : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l2586_258622


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2586_258674

theorem polynomial_remainder_theorem (x : ℝ) : 
  ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ), 
    (∀ x, x^50 = (x^2 - 5*x + 6) * Q x + R x) ∧ 
    (∃ a b : ℝ, ∀ x, R x = a*x + b) ∧
    R x = (3^50 - 2^50)*x + (2^50 - 2*3^50 + 2*2^50) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2586_258674


namespace NUMINAMATH_CALUDE_integer_sequence_count_l2586_258686

def sequence_term (n : ℕ) : ℚ :=
  16200 / (5 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem integer_sequence_count : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∀ (k : ℕ), k < n → is_integer (sequence_term k)) ∧ 
    (∀ (k : ℕ), k ≥ n → ¬ is_integer (sequence_term k))) ∧
  (∃! (n : ℕ), n > 0 ∧ 
    (∀ (k : ℕ), k < n → is_integer (sequence_term k)) ∧ 
    (∀ (k : ℕ), k ≥ n → ¬ is_integer (sequence_term k))) ∧
  (∃ (n : ℕ), n = 3 ∧ 
    (∀ (k : ℕ), k < n → is_integer (sequence_term k)) ∧ 
    (∀ (k : ℕ), k ≥ n → ¬ is_integer (sequence_term k))) :=
by sorry

end NUMINAMATH_CALUDE_integer_sequence_count_l2586_258686


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l2586_258642

open Real

theorem sin_shift_equivalence (x : ℝ) : 
  sin (2 * x + π / 6) = sin (2 * (x + π / 4) - π / 3) := by sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l2586_258642


namespace NUMINAMATH_CALUDE_base7_to_base10_23456_l2586_258667

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- The given number in base 7 --/
def base7Number : List Nat := [6, 5, 4, 3, 2]

/-- Theorem stating that the base 10 equivalent of 23456 in base 7 is 6068 --/
theorem base7_to_base10_23456 :
  base7ToBase10 base7Number = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_23456_l2586_258667


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2586_258623

theorem sqrt_inequality (a : ℝ) (h : a > 1) : Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2586_258623


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2586_258697

/-- Given a line passing through two points, prove that the sum of its slope and y-intercept is -5/2 --/
theorem line_slope_intercept_sum (m b : ℚ) : 
  ((-1 : ℚ) = m * (1/2) + b) → 
  (2 = m * (-1/2) + b) → 
  m + b = -5/2 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2586_258697


namespace NUMINAMATH_CALUDE_function_linearity_l2586_258626

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h_continuous : Continuous f)
variable (h_additive : ∀ x y : ℝ, f (x + y) = f x + f y)

-- State the theorem
theorem function_linearity :
  ∃ C : ℝ, (∀ x : ℝ, f x = C * x) ∧ C = f 1 :=
sorry

end NUMINAMATH_CALUDE_function_linearity_l2586_258626


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2586_258613

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 1)
  (h_product : a 2 * a 4 = 16) :
  a 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2586_258613


namespace NUMINAMATH_CALUDE_only_rational_root_l2586_258610

def f (x : ℚ) : ℚ := 3 * x^5 - 2 * x^4 + 5 * x^3 - x^2 - 7 * x + 2

theorem only_rational_root :
  ∀ (x : ℚ), f x = 0 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_only_rational_root_l2586_258610


namespace NUMINAMATH_CALUDE_student_number_factor_l2586_258611

theorem student_number_factor (f : ℝ) : 120 * f - 138 = 102 → f = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_number_factor_l2586_258611


namespace NUMINAMATH_CALUDE_square_area_increase_l2586_258661

theorem square_area_increase (a : ℝ) (ha : a > 0) : 
  let side_b := 2 * a
  let side_c := side_b * 1.4
  let area_a := a ^ 2
  let area_b := side_b ^ 2
  let area_c := side_c ^ 2
  (area_c - (area_a + area_b)) / (area_a + area_b) = 0.568 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l2586_258661


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2586_258639

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value :
  ∀ k : ℝ, 
  let a : ℝ × ℝ := (1, k)
  let b : ℝ × ℝ := (9, k - 6)
  are_parallel a b → k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2586_258639


namespace NUMINAMATH_CALUDE_circle_condition_l2586_258659

theorem circle_condition (m : ℝ) :
  (∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔
  (m < 1/4 ∨ m > 1) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l2586_258659


namespace NUMINAMATH_CALUDE_extreme_value_implies_f_2_l2586_258624

/-- A function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- Theorem: If f(x) has an extreme value of 10 at x = 1, then f(2) = 18 -/
theorem extreme_value_implies_f_2 (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b 1 ≥ f a b x) ∧
  f a b 1 = 10 →
  f a b 2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_f_2_l2586_258624


namespace NUMINAMATH_CALUDE_xy_value_l2586_258664

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x + y) = 16)
  (h2 : (16 : ℝ)^(x + y) / (4 : ℝ)^(7 * y) = 1024) : 
  x * y = 30 := by sorry

end NUMINAMATH_CALUDE_xy_value_l2586_258664


namespace NUMINAMATH_CALUDE_complex_collinear_solution_l2586_258668

def collinear (a b c : ℂ) : Prop :=
  ∃ t : ℝ, b - a = t • (c - a) ∨ c - a = t • (b - a)

theorem complex_collinear_solution (z : ℂ) :
  collinear 1 Complex.I z ∧ Complex.abs z = 5 →
  z = 4 - 3 * Complex.I ∨ z = -3 + 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_collinear_solution_l2586_258668


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2586_258690

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 36 * x + c = 0) →
  (a + c = 41) →
  (a < c) →
  (a = (41 - Real.sqrt 385) / 2 ∧ c = (41 + Real.sqrt 385) / 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2586_258690


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_l2586_258660

theorem least_sum_of_exponents (h : ℕ+) (a b c : ℕ+) 
  (h_div_225 : 225 ∣ h) 
  (h_div_216 : 216 ∣ h) 
  (h_eq : h = 2^(a:ℕ) * 3^(b:ℕ) * 5^(c:ℕ)) : 
  (∀ a' b' c' : ℕ+, 
    (225 ∣ (2^(a':ℕ) * 3^(b':ℕ) * 5^(c':ℕ))) → 
    (216 ∣ (2^(a':ℕ) * 3^(b':ℕ) * 5^(c':ℕ))) → 
    (a:ℕ) + (b:ℕ) + (c:ℕ) ≤ (a':ℕ) + (b':ℕ) + (c':ℕ)) ∧ 
  (a:ℕ) + (b:ℕ) + (c:ℕ) = 10 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_l2586_258660


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2586_258647

theorem point_in_fourth_quadrant (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ x y : ℝ, x ≠ y ∧ a * x^2 - x - 1/4 = 0 ∧ a * y^2 - y - 1/4 = 0) :
  (a + 1 > 0) ∧ (-3 - a < 0) := by
  sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2586_258647


namespace NUMINAMATH_CALUDE_larger_number_proof_l2586_258605

theorem larger_number_proof (a b : ℕ) 
  (hcf_cond : Nat.gcd a b = 23)
  (lcm_cond : Nat.lcm a b = 23 * 13 * 16) :
  max a b = 368 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2586_258605


namespace NUMINAMATH_CALUDE_mike_tv_hours_l2586_258678

-- Define the number of hours Mike watches TV daily
def tv_hours : ℝ := 4

-- Define the number of days in a week Mike plays video games
def gaming_days : ℕ := 3

-- Define the total hours spent on both activities in a week
def total_hours : ℝ := 34

-- Theorem statement
theorem mike_tv_hours :
  -- Condition: On gaming days, Mike plays for half as long as he watches TV
  (gaming_days * (tv_hours / 2) +
  -- Condition: Mike watches TV every day of the week
   7 * tv_hours = total_hours) →
  -- Conclusion: Mike watches TV for 4 hours every day
  tv_hours = 4 := by
sorry

end NUMINAMATH_CALUDE_mike_tv_hours_l2586_258678


namespace NUMINAMATH_CALUDE_expression_evaluation_l2586_258685

theorem expression_evaluation (a b c : ℚ) 
  (ha : a = 1/3) (hb : b = 1/2) (hc : c = 1) : 
  (2*a^2 - b) - (a^2 - 4*b) - (b + c) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2586_258685


namespace NUMINAMATH_CALUDE_average_pages_proof_l2586_258630

def book_pages : List Nat := [50, 75, 80, 120, 100, 90, 110, 130]

theorem average_pages_proof :
  (book_pages.sum : ℚ) / book_pages.length = 94.375 := by
  sorry

end NUMINAMATH_CALUDE_average_pages_proof_l2586_258630


namespace NUMINAMATH_CALUDE_power_difference_sum_equals_six_l2586_258618

theorem power_difference_sum_equals_six : 3^2 - 2^2 + 1^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_sum_equals_six_l2586_258618


namespace NUMINAMATH_CALUDE_product_equals_square_l2586_258643

theorem product_equals_square : 10 * 9.99 * 0.999 * 100 = (99.9 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l2586_258643


namespace NUMINAMATH_CALUDE_domain_fxPlus2_l2586_258612

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x-3)
def domain_f2xMinus3 : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem domain_fxPlus2 (h : ∀ x ∈ domain_f2xMinus3, f (2*x - 3) = f (2*x - 3)) :
  {x : ℝ | f (x + 2) = f (x + 2)} = {x : ℝ | -9 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_domain_fxPlus2_l2586_258612


namespace NUMINAMATH_CALUDE_bacterium_diameter_nanometers_l2586_258640

/-- Conversion factor from meters to nanometers -/
def meters_to_nanometers : ℝ := 10^9

/-- The diameter of the bacterium in meters -/
def bacterium_diameter_meters : ℝ := 0.00000285

/-- Theorem stating that the diameter of the bacterium in nanometers is 2850 -/
theorem bacterium_diameter_nanometers :
  bacterium_diameter_meters * meters_to_nanometers = 2850 := by
  sorry

#eval bacterium_diameter_meters * meters_to_nanometers

end NUMINAMATH_CALUDE_bacterium_diameter_nanometers_l2586_258640
