import Mathlib

namespace NUMINAMATH_CALUDE_lucy_second_round_cookies_l2645_264540

/-- The number of cookies Lucy sold on her first round -/
def first_round : ℕ := 34

/-- The total number of cookies Lucy sold -/
def total : ℕ := 61

/-- The number of cookies Lucy sold on her second round -/
def second_round : ℕ := total - first_round

theorem lucy_second_round_cookies : second_round = 27 := by
  sorry

end NUMINAMATH_CALUDE_lucy_second_round_cookies_l2645_264540


namespace NUMINAMATH_CALUDE_unique_prime_between_squares_l2645_264558

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 6 ∧ 
  ∃ m : ℕ, p = (m + 1)^2 - 10 ∧
  m^2 < p ∧ p < (m + 1)^2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_squares_l2645_264558


namespace NUMINAMATH_CALUDE_arrangement_is_correct_l2645_264516

-- Define the metals and safes
inductive Metal
| Gold | Silver | Bronze | Platinum | Nickel

inductive Safe
| One | Two | Three | Four | Five

-- Define the arrangement as a function from Safe to Metal
def Arrangement := Safe → Metal

-- Define the statements on the safes
def statement1 (a : Arrangement) : Prop :=
  a Safe.Two = Metal.Gold ∨ a Safe.Three = Metal.Gold

def statement2 (a : Arrangement) : Prop :=
  a Safe.One = Metal.Silver

def statement3 (a : Arrangement) : Prop :=
  a Safe.Three ≠ Metal.Bronze

def statement4 (a : Arrangement) : Prop :=
  (a Safe.One = Metal.Nickel ∧ a Safe.Two = Metal.Gold) ∨
  (a Safe.Two = Metal.Nickel ∧ a Safe.Three = Metal.Gold) ∨
  (a Safe.Three = Metal.Nickel ∧ a Safe.Four = Metal.Gold) ∨
  (a Safe.Four = Metal.Nickel ∧ a Safe.Five = Metal.Gold)

def statement5 (a : Arrangement) : Prop :=
  (a Safe.One = Metal.Bronze ∧ a Safe.Two = Metal.Platinum) ∨
  (a Safe.Two = Metal.Bronze ∧ a Safe.Three = Metal.Platinum) ∨
  (a Safe.Three = Metal.Bronze ∧ a Safe.Four = Metal.Platinum) ∨
  (a Safe.Four = Metal.Bronze ∧ a Safe.Five = Metal.Platinum)

-- Define the correct arrangement
def correctArrangement : Arrangement :=
  fun s => match s with
  | Safe.One => Metal.Nickel
  | Safe.Two => Metal.Silver
  | Safe.Three => Metal.Bronze
  | Safe.Four => Metal.Platinum
  | Safe.Five => Metal.Gold

-- Theorem statement
theorem arrangement_is_correct (a : Arrangement) :
  (∃! s, a s = Metal.Gold ∧
    (s = Safe.One → statement1 a) ∧
    (s = Safe.Two → statement2 a) ∧
    (s = Safe.Three → statement3 a) ∧
    (s = Safe.Four → statement4 a) ∧
    (s = Safe.Five → statement5 a)) →
  (∀ s, a s = correctArrangement s) :=
sorry

end NUMINAMATH_CALUDE_arrangement_is_correct_l2645_264516


namespace NUMINAMATH_CALUDE_max_x_elements_l2645_264548

/-- Represents the number of elements of each type -/
structure Elements where
  fire : ℕ
  stone : ℕ
  metal : ℕ

/-- Represents the alchemical reactions -/
def reaction1 (e : Elements) : Elements :=
  { fire := e.fire - 1, stone := e.stone - 1, metal := e.metal + 1 }

def reaction2 (e : Elements) : Elements :=
  { fire := e.fire, stone := e.stone + 2, metal := e.metal - 1 }

/-- Creates an element X -/
def createX (e : Elements) : Elements :=
  { fire := e.fire - 2, stone := e.stone - 3, metal := e.metal - 1 }

/-- The initial state of elements -/
def initialElements : Elements :=
  { fire := 50, stone := 50, metal := 0 }

/-- Checks if the number of elements is non-negative -/
def isValid (e : Elements) : Prop :=
  e.fire ≥ 0 ∧ e.stone ≥ 0 ∧ e.metal ≥ 0

/-- Theorem: The maximum number of X elements that can be created is 14 -/
theorem max_x_elements : 
  ∃ (n : ℕ) (e : Elements), 
    n = 14 ∧ 
    isValid e ∧ 
    ∀ m : ℕ, m > n → 
      ¬∃ (f : Elements), isValid f ∧ 
        (∃ (seq : List (Elements → Elements)), 
          f = (seq.foldl (λ acc g => g acc) initialElements) ∧
          (createX^[m]) f = f) :=
sorry

end NUMINAMATH_CALUDE_max_x_elements_l2645_264548


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2645_264527

theorem cone_base_circumference 
  (V : ℝ) (l : ℝ) (θ : ℝ) (h : ℝ) (r : ℝ) (C : ℝ) :
  V = 27 * Real.pi ∧ 
  l = 6 ∧ 
  θ = Real.pi / 3 ∧ 
  h = l * Real.cos θ ∧ 
  V = 1/3 * Real.pi * r^2 * h ∧ 
  C = 2 * Real.pi * r
  → C = 6 * Real.sqrt 3 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2645_264527


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2645_264523

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dims : BoxDimensions) : ℕ :=
  dims.length * dims.width * dims.height

/-- Represents the carton dimensions -/
def cartonDims : BoxDimensions :=
  { length := 25, width := 35, height := 50 }

/-- Represents the soap box dimensions -/
def soapBoxDims : BoxDimensions :=
  { length := 8, width := 7, height := 6 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDims) / (boxVolume soapBoxDims) = 130 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2645_264523


namespace NUMINAMATH_CALUDE_triangle_count_is_53_l2645_264528

/-- Represents a rectangle divided into triangles --/
structure TriangulatedRectangle where
  columns : Nat
  rows : Nat
  has_full_diagonals : Bool
  has_half_diagonals : Bool

/-- Counts the number of triangles in a TriangulatedRectangle --/
def count_triangles (rect : TriangulatedRectangle) : Nat :=
  sorry

/-- The specific rectangle described in the problem --/
def problem_rectangle : TriangulatedRectangle :=
  { columns := 6
  , rows := 3
  , has_full_diagonals := true
  , has_half_diagonals := true }

theorem triangle_count_is_53 : count_triangles problem_rectangle = 53 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_is_53_l2645_264528


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2645_264525

def A : Set ℝ := {-2, -1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2645_264525


namespace NUMINAMATH_CALUDE_largest_base5_3digit_in_base10_l2645_264529

/-- The largest three-digit number in base-5 -/
def largest_base5_3digit : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

/-- Theorem: The largest three-digit number in base-5, when converted to base-10, is equal to 124 -/
theorem largest_base5_3digit_in_base10 : largest_base5_3digit = 124 := by
  sorry

end NUMINAMATH_CALUDE_largest_base5_3digit_in_base10_l2645_264529


namespace NUMINAMATH_CALUDE_total_students_count_l2645_264589

/-- Represents the arrangement of students in two rows -/
structure StudentArrangement where
  boys_count : ℕ
  girls_count : ℕ
  rajan_left_position : ℕ
  vinay_right_position : ℕ
  boys_between_rajan_vinay : ℕ
  deepa_left_position : ℕ

/-- The total number of students in both rows -/
def total_students (arrangement : StudentArrangement) : ℕ :=
  arrangement.boys_count + arrangement.girls_count

/-- The theorem stating the total number of students given the conditions -/
theorem total_students_count (arrangement : StudentArrangement) 
  (h1 : arrangement.boys_count = arrangement.girls_count)
  (h2 : arrangement.rajan_left_position = 6)
  (h3 : arrangement.vinay_right_position = 10)
  (h4 : arrangement.boys_between_rajan_vinay = 8)
  (h5 : arrangement.deepa_left_position = 5)
  : total_students arrangement = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_students_count_l2645_264589


namespace NUMINAMATH_CALUDE_equation_solutions_l2645_264545

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 3 ∧ x₂ = 2 - Real.sqrt 3 ∧
    x₁^2 - 4*x₁ + 1 = 0 ∧ x₂^2 - 4*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = 2/3 ∧
    3*y₁*(2*y₁ + 1) = 4*y₁ + 2 ∧ 3*y₂*(2*y₂ + 1) = 4*y₂ + 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2645_264545


namespace NUMINAMATH_CALUDE_inequality_relation_l2645_264584

theorem inequality_relation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relation_l2645_264584


namespace NUMINAMATH_CALUDE_factorization_equality_l2645_264507

theorem factorization_equality (a b x y : ℝ) :
  (a*x - b*y)^2 + (a*y + b*x)^2 = (x^2 + y^2) * (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2645_264507


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_less_than_negative_one_l2645_264597

theorem inequality_holds_iff_a_less_than_negative_one (a : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → x^2 - (a + 1) * x + a + 1 > 0) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_less_than_negative_one_l2645_264597


namespace NUMINAMATH_CALUDE_definite_integral_2x_plus_1_over_x_l2645_264594

theorem definite_integral_2x_plus_1_over_x :
  ∫ x in (1 : ℝ)..2, (2 * x + 1 / x) = 3 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_2x_plus_1_over_x_l2645_264594


namespace NUMINAMATH_CALUDE_base_4_9_digit_difference_l2645_264563

def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log base n).succ

theorem base_4_9_digit_difference :
  num_digits 1234 4 - num_digits 1234 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_4_9_digit_difference_l2645_264563


namespace NUMINAMATH_CALUDE_car_speed_comparison_l2645_264551

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  (2 * u * v) / (u + v) ≤ (u + v) / 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l2645_264551


namespace NUMINAMATH_CALUDE_abc_inequality_l2645_264541

theorem abc_inequality (a b c : ℝ) 
  (ha : a = (1/3)^(2/3))
  (hb : b = (1/5)^(2/3))
  (hc : c = (4/9)^(1/3)) :
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l2645_264541


namespace NUMINAMATH_CALUDE_weekly_surplus_and_monthly_income_estimate_l2645_264565

def weekly_income : List ℤ := [65, 68, 50, 66, 50, 75, 74]
def weekly_expenditure : List ℤ := [-60, -64, -63, -58, -60, -64, -65]

def calculate_surplus (income : List ℤ) (expenditure : List ℤ) : ℤ :=
  (income.sum + expenditure.sum)

def estimate_monthly_income (expenditure : List ℤ) : ℤ :=
  (expenditure.map (Int.natAbs)).sum * 30 / 7

theorem weekly_surplus_and_monthly_income_estimate :
  (calculate_surplus weekly_income weekly_expenditure = 14) ∧
  (estimate_monthly_income weekly_expenditure = 1860) := by
  sorry

#eval calculate_surplus weekly_income weekly_expenditure
#eval estimate_monthly_income weekly_expenditure

end NUMINAMATH_CALUDE_weekly_surplus_and_monthly_income_estimate_l2645_264565


namespace NUMINAMATH_CALUDE_a_less_than_one_l2645_264510

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the conditions
axiom deriv_f : ∀ x, HasDerivAt f (f' x) x
axiom symm_cond : ∀ x, f x + f (-x) = x^2
axiom deriv_gt : ∀ x ≥ 0, f' x > x
axiom ineq_cond : ∀ a, f (2 - a) + 2*a > f a + 2

-- State the theorem
theorem a_less_than_one (a : ℝ) : a < 1 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_one_l2645_264510


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2645_264580

/-- Given an arithmetic sequence {a_n} where a_2 = 3 and S_4 = 16, prove S_9 = 81 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 2 = 3 →                            -- given condition
  S 4 = 16 →                           -- given condition
  S 9 = 81 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2645_264580


namespace NUMINAMATH_CALUDE_alien_attack_probability_l2645_264556

/-- The number of aliens attacking --/
def num_aliens : ℕ := 3

/-- The number of galaxies being attacked --/
def num_galaxies : ℕ := 4

/-- The number of days of the attack --/
def num_days : ℕ := 3

/-- The probability that a specific galaxy is not chosen by any alien on a given day --/
def prob_not_chosen_day : ℚ := (3/4)^num_aliens

/-- The probability that a specific galaxy is not destroyed over all days --/
def prob_not_destroyed : ℚ := prob_not_chosen_day^num_days

/-- The probability that at least one galaxy is not destroyed --/
def prob_at_least_one_not_destroyed : ℚ := num_galaxies * prob_not_destroyed

/-- The probability that all galaxies are destroyed --/
def prob_all_destroyed : ℚ := 1 - prob_at_least_one_not_destroyed

theorem alien_attack_probability : prob_all_destroyed = 45853/65536 := by
  sorry

end NUMINAMATH_CALUDE_alien_attack_probability_l2645_264556


namespace NUMINAMATH_CALUDE_twelve_person_tournament_matches_l2645_264502

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a 12-person round-robin tournament, the number of matches is 66 -/
theorem twelve_person_tournament_matches : num_matches 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_twelve_person_tournament_matches_l2645_264502


namespace NUMINAMATH_CALUDE_total_oranges_in_box_l2645_264514

def initial_oranges : ℝ := 55.0
def added_oranges : ℝ := 35.0

theorem total_oranges_in_box : initial_oranges + added_oranges = 90.0 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_in_box_l2645_264514


namespace NUMINAMATH_CALUDE_coefficient_a_is_zero_l2645_264524

-- Define the quadratic equation
def quadratic_equation (a b c p : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c + p = 0

-- Define the condition that all roots are real and positive
def all_roots_real_positive (a b c : ℝ) : Prop :=
  ∀ p > 0, ∀ x, quadratic_equation a b c p x → x > 0

-- Theorem statement
theorem coefficient_a_is_zero (a b c : ℝ) :
  all_roots_real_positive a b c → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a_is_zero_l2645_264524


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2645_264547

/-- Given a line with equation y + 5 = -3(x + 6), 
    the sum of its x-intercept and y-intercept is -92/3 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 5 = -3 * (x + 6)) → 
  (∃ x_int y_int : ℝ, 
    (y_int + 5 = -3 * (x_int + 6)) ∧ 
    (0 + 5 = -3 * (x_int + 6)) ∧ 
    (y_int + 5 = -3 * (0 + 6)) ∧ 
    (x_int + y_int = -92/3)) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2645_264547


namespace NUMINAMATH_CALUDE_second_term_is_seven_l2645_264515

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ
  num_terms : ℕ
  sum_first_eight_eq_last_four : ℝ → Prop

/-- The theorem statement -/
theorem second_term_is_seven
  (seq : ArithmeticSequence)
  (h1 : seq.num_terms = 12)
  (h2 : seq.common_difference = 2)
  (h3 : seq.sum_first_eight_eq_last_four seq.first_term) :
  seq.first_term + seq.common_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_term_is_seven_l2645_264515


namespace NUMINAMATH_CALUDE_star_six_three_l2645_264568

def star (a b : ℝ) : ℝ := 4*a + 3*b - 2*a*b

theorem star_six_three : star 6 3 = -3 := by sorry

end NUMINAMATH_CALUDE_star_six_three_l2645_264568


namespace NUMINAMATH_CALUDE_necessary_condition_propositions_l2645_264518

-- Definition for necessary condition
def is_necessary_condition (p q : Prop) : Prop :=
  q → p

-- Proposition A
def prop_a (x y : ℝ) : Prop :=
  is_necessary_condition (x^2 > y^2) (x > y)

-- Proposition B
def prop_b (x : ℝ) : Prop :=
  is_necessary_condition (x > 5) (x > 10)

-- Proposition C
def prop_c (a b c : ℝ) : Prop :=
  is_necessary_condition (a * c = b * c) (a = b)

-- Proposition D
def prop_d (x y : ℝ) : Prop :=
  is_necessary_condition (2 * x + 1 = 2 * y + 1) (x = y)

-- Theorem stating which propositions have p as a necessary condition for q
theorem necessary_condition_propositions :
  (∃ x y : ℝ, ¬(prop_a x y)) ∧
  (∀ x : ℝ, prop_b x) ∧
  (∀ a b c : ℝ, c ≠ 0 → prop_c a b c) ∧
  (∀ x y : ℝ, prop_d x y) :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_propositions_l2645_264518


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2645_264534

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f (2 * x + y) + 5 * x * y = f (3 * x - y) + 2 * x^2 + 1

/-- The main theorem stating that any function satisfying the functional equation
    must have f(10) = -49 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 10 = -49 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2645_264534


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2645_264555

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, 7 * x^2 - 2 * x + 45 = 0 ↔ x = p + q * I) → 
  p + q^2 = 321/49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2645_264555


namespace NUMINAMATH_CALUDE_polygon_sides_possibility_l2645_264585

theorem polygon_sides_possibility : ∃ n : ℕ, n ≥ 10 ∧ (n - 3) * 180 = 1620 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_possibility_l2645_264585


namespace NUMINAMATH_CALUDE_rebecca_earnings_l2645_264567

def haircut_price : ℕ := 30
def perm_price : ℕ := 40
def dye_job_price : ℕ := 60
def hair_extension_price : ℕ := 80

def haircut_supply_cost : ℕ := 5
def dye_job_supply_cost : ℕ := 10
def hair_extension_supply_cost : ℕ := 25

def student_discount : ℚ := 0.1
def senior_discount : ℚ := 0.15
def first_time_discount : ℕ := 5

def num_haircuts : ℕ := 5
def num_student_haircuts : ℕ := 2
def num_perms : ℕ := 3
def num_senior_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def num_first_time_dye_jobs : ℕ := 1
def num_hair_extensions : ℕ := 1

def total_tips : ℕ := 75
def daily_expenses : ℕ := 45

theorem rebecca_earnings : 
  let total_revenue := 
    num_haircuts * haircut_price + 
    num_perms * perm_price + 
    num_dye_jobs * dye_job_price + 
    num_hair_extensions * hair_extension_price
  let total_discounts := 
    (num_student_haircuts * haircut_price * student_discount).floor +
    (num_senior_perms * perm_price * senior_discount).floor +
    (num_first_time_dye_jobs * first_time_discount)
  let supply_costs := 
    num_haircuts * haircut_supply_cost +
    num_dye_jobs * dye_job_supply_cost +
    num_hair_extensions * hair_extension_supply_cost
  let earnings := 
    total_revenue - total_discounts - supply_costs + total_tips - daily_expenses
  earnings = 413 := by sorry

end NUMINAMATH_CALUDE_rebecca_earnings_l2645_264567


namespace NUMINAMATH_CALUDE_tangent_circles_expression_l2645_264573

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- The distance between the centers of two tangent circles is the sum of their radii -/
def distance (c1 c2 : Circle) : ℝ := c1.radius + c2.radius

theorem tangent_circles_expression (a b c : ℝ) (A B C : Circle)
  (ha : A.radius = a)
  (hb : B.radius = b)
  (hc : C.radius = c)
  (hab : a > b)
  (hbc : b > c)
  (htangent : A.radius + B.radius = distance A B ∧ 
              B.radius + C.radius = distance B C ∧ 
              C.radius + A.radius = distance C A) :
  distance A B + distance B C - distance C A = b ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_expression_l2645_264573


namespace NUMINAMATH_CALUDE_quotient_sum_and_difference_l2645_264579

theorem quotient_sum_and_difference (a b : ℝ) (h : a / b = -1) : 
  (a + b = 0) ∧ (|a - b| = 2 * |b|) := by
  sorry

end NUMINAMATH_CALUDE_quotient_sum_and_difference_l2645_264579


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2645_264530

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), ∀ x, 4 * x^2 + 1 = 6 * x ↔ a * x^2 + b * x + c = 0 ∧ a = 4 ∧ b = -6 ∧ c = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2645_264530


namespace NUMINAMATH_CALUDE_alteration_cost_per_shoe_l2645_264574

-- Define the number of pairs of shoes
def num_pairs : ℕ := 14

-- Define the total cost of alteration
def total_cost : ℕ := 1036

-- Define the cost per shoe
def cost_per_shoe : ℕ := 37

-- Theorem statement
theorem alteration_cost_per_shoe :
  (total_cost : ℚ) / (2 * num_pairs) = cost_per_shoe := by
  sorry

end NUMINAMATH_CALUDE_alteration_cost_per_shoe_l2645_264574


namespace NUMINAMATH_CALUDE_author_earnings_calculation_l2645_264501

def author_earnings (paperback_copies : Nat) (paperback_price : Real)
                    (hardcover_copies : Nat) (hardcover_price : Real)
                    (ebook_copies : Nat) (ebook_price : Real)
                    (audiobook_copies : Nat) (audiobook_price : Real) : Real :=
  let paperback_sales := paperback_copies * paperback_price
  let hardcover_sales := hardcover_copies * hardcover_price
  let ebook_sales := ebook_copies * ebook_price
  let audiobook_sales := audiobook_copies * audiobook_price
  0.06 * paperback_sales + 0.12 * hardcover_sales + 0.08 * ebook_sales + 0.10 * audiobook_sales

theorem author_earnings_calculation :
  author_earnings 32000 0.20 15000 0.40 10000 0.15 5000 0.50 = 1474 :=
by sorry

end NUMINAMATH_CALUDE_author_earnings_calculation_l2645_264501


namespace NUMINAMATH_CALUDE_inequality_and_min_value_l2645_264512

theorem inequality_and_min_value (a b x y : ℝ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) (h4 : x > 0) (h5 : y > 0) :
  (a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧
  (a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ x / y = a / b) ∧
  (∀ x ∈ Set.Ioo 0 (1/2), 2/x + 9/(1-2*x) ≥ 25) ∧
  (2/(1/5) + 9/(1-2*(1/5)) = 25) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_min_value_l2645_264512


namespace NUMINAMATH_CALUDE_four_spheres_block_light_l2645_264520

-- Define a point in 3D space
def Point := ℝ × ℝ × ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point
  radius : ℝ
  radius_pos : radius > 0

-- Define the property of a sphere being opaque
def isOpaque (s : Sphere) : Prop := sorry

-- Define the property of two spheres being non-intersecting
def nonIntersecting (s1 s2 : Sphere) : Prop := sorry

-- Define the property of a set of spheres blocking light from a point source
def blocksLight (source : Point) (spheres : List Sphere) : Prop := sorry

-- The main theorem
theorem four_spheres_block_light :
  ∃ (s1 s2 s3 s4 : Sphere) (source : Point),
    isOpaque s1 ∧ isOpaque s2 ∧ isOpaque s3 ∧ isOpaque s4 ∧
    nonIntersecting s1 s2 ∧ nonIntersecting s1 s3 ∧ nonIntersecting s1 s4 ∧
    nonIntersecting s2 s3 ∧ nonIntersecting s2 s4 ∧ nonIntersecting s3 s4 ∧
    blocksLight source [s1, s2, s3, s4] := by
  sorry

end NUMINAMATH_CALUDE_four_spheres_block_light_l2645_264520


namespace NUMINAMATH_CALUDE_sara_gave_nine_kittens_l2645_264532

/-- The number of kittens Sara gave to Tim -/
def kittens_from_sara (initial : ℕ) (given_to_jessica : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_to_jessica)

/-- Proof that Sara gave Tim 9 kittens -/
theorem sara_gave_nine_kittens :
  kittens_from_sara 6 3 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sara_gave_nine_kittens_l2645_264532


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_specific_prism_l2645_264562

/-- A triangular prism with a regular triangular base and lateral edges perpendicular to the base -/
structure TriangularPrism where
  baseArea : ℝ
  lateralEdgeLength : ℝ

/-- The lateral surface area of a triangular prism -/
def lateralSurfaceArea (prism : TriangularPrism) : ℝ :=
  sorry

theorem lateral_surface_area_of_specific_prism :
  let prism : TriangularPrism := { baseArea := 4 * Real.sqrt 3, lateralEdgeLength := 3 }
  lateralSurfaceArea prism = 36 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_specific_prism_l2645_264562


namespace NUMINAMATH_CALUDE_tank_capacity_l2645_264550

theorem tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (added_water : ℚ) :
  initial_fraction = 1/4 →
  final_fraction = 3/4 →
  added_water = 200 →
  (final_fraction - initial_fraction) * (added_water / (final_fraction - initial_fraction)) = 400 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l2645_264550


namespace NUMINAMATH_CALUDE_number_of_friends_l2645_264526

-- Define the total number of stickers
def total_stickers : ℕ := 72

-- Define the number of stickers each friend receives
def stickers_per_friend : ℕ := 8

-- Theorem to prove the number of friends receiving stickers
theorem number_of_friends : total_stickers / stickers_per_friend = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_friends_l2645_264526


namespace NUMINAMATH_CALUDE_larger_number_problem_l2645_264598

theorem larger_number_problem (x y : ℝ) : 
  x - y = 5 → x + y = 37 → max x y = 21 := by sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2645_264598


namespace NUMINAMATH_CALUDE_point_on_curve_l2645_264508

def curve (x y : ℝ) : Prop := x^2 - x*y + 2*y + 1 = 0

theorem point_on_curve :
  curve 0 (-1/2) ∧
  ¬ curve 0 0 ∧
  ¬ curve 1 (-1) ∧
  ¬ curve 1 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_l2645_264508


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l2645_264543

/-- The total number of pencils after adding more -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Proof that the total number of pencils is 215 -/
theorem pencils_in_drawer : total_pencils 115 100 = 215 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l2645_264543


namespace NUMINAMATH_CALUDE_speed_calculation_l2645_264575

theorem speed_calculation (v : ℝ) (t : ℝ) (h1 : t > 0) :
  v * t = (v + 18) * (2/3 * t) → v = 36 :=
by sorry

end NUMINAMATH_CALUDE_speed_calculation_l2645_264575


namespace NUMINAMATH_CALUDE_xy_plus_reciprocal_minimum_l2645_264591

theorem xy_plus_reciprocal_minimum (x y : ℝ) (hx : x < 0) (hy : y < 0) (hsum : x + y = -1) :
  ∀ z, z = x * y + 1 / (x * y) → z ≥ 17/4 :=
by sorry

end NUMINAMATH_CALUDE_xy_plus_reciprocal_minimum_l2645_264591


namespace NUMINAMATH_CALUDE_select_three_roles_from_25_l2645_264503

/-- The number of ways to select three distinct roles from a squad of players. -/
def selectThreeRoles (squadSize : ℕ) : ℕ :=
  squadSize * (squadSize - 1) * (squadSize - 2)

/-- Theorem: The number of ways to select a captain, vice-captain, and goalkeeper
    from a squad of 25 players, where no player can occupy more than one role, is 13800. -/
theorem select_three_roles_from_25 : selectThreeRoles 25 = 13800 := by
  sorry

end NUMINAMATH_CALUDE_select_three_roles_from_25_l2645_264503


namespace NUMINAMATH_CALUDE_ratio_u_to_x_l2645_264590

theorem ratio_u_to_x (u v x y : ℚ) 
  (h1 : u / v = 5 / 2)
  (h2 : x / y = 4 / 1)
  (h3 : v / y = 3 / 4) :
  u / x = 15 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ratio_u_to_x_l2645_264590


namespace NUMINAMATH_CALUDE_problem_I4_1_l2645_264561

theorem problem_I4_1 (x y : ℝ) (h : (10 * x - 3 * y) / (x + 2 * y) = 2) :
  (y + x) / (y - x) = 15 :=
by sorry

end NUMINAMATH_CALUDE_problem_I4_1_l2645_264561


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposed_l2645_264511

/-- Represents a card color -/
inductive CardColor
| Red
| White
| Black

/-- Represents a person -/
inductive Person
| A
| B
| C

/-- Represents the distribution of cards to people -/
def Distribution := Person → CardColor

/-- The event "A receives the red card" -/
def event_A_red (d : Distribution) : Prop := d Person.A = CardColor.Red

/-- The event "B receives the red card" -/
def event_B_red (d : Distribution) : Prop := d Person.B = CardColor.Red

/-- The set of all possible distributions -/
def all_distributions : Set Distribution :=
  {d | ∀ c : CardColor, ∃! p : Person, d p = c}

theorem events_mutually_exclusive_but_not_opposed :
  (∀ d ∈ all_distributions, ¬(event_A_red d ∧ event_B_red d)) ∧
  (∃ d ∈ all_distributions, ¬event_A_red d ∧ ¬event_B_red d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposed_l2645_264511


namespace NUMINAMATH_CALUDE_carlos_welfare_fund_contribution_l2645_264595

/-- The amount in cents dedicated to the welfare fund per hour -/
def welfare_fund_cents (hourly_wage : ℝ) (deduction_rate : ℝ) : ℝ :=
  hourly_wage * 100 * deduction_rate

/-- Proof that Carlos' welfare fund contribution is 40 cents per hour -/
theorem carlos_welfare_fund_contribution :
  welfare_fund_cents 25 0.016 = 40 := by
  sorry

end NUMINAMATH_CALUDE_carlos_welfare_fund_contribution_l2645_264595


namespace NUMINAMATH_CALUDE_smallest_congruent_integer_l2645_264531

theorem smallest_congruent_integer (n : ℕ) : 
  (0 ≤ n ∧ n ≤ 15) ∧ n ≡ 5673 [MOD 16] → n = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_congruent_integer_l2645_264531


namespace NUMINAMATH_CALUDE_bruce_purchase_amount_l2645_264593

/-- The total amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1125 for his purchase -/
theorem bruce_purchase_amount :
  total_amount 9 70 9 55 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_bruce_purchase_amount_l2645_264593


namespace NUMINAMATH_CALUDE_min_value_of_max_sum_l2645_264544

theorem min_value_of_max_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → 
  a + b + c + d = 4 → 
  let M := max (max (a + b + c) (a + b + d)) (max (a + c + d) (b + c + d))
  3 ≤ M ∧ ∀ (M' : ℝ), (∀ (a' b' c' d' : ℝ), 
    a' > 0 → b' > 0 → c' > 0 → d' > 0 → 
    a' + b' + c' + d' = 4 → 
    let M'' := max (max (a' + b' + c') (a' + b' + d')) (max (a' + c' + d') (b' + c' + d'))
    M'' ≤ M') → 
  3 ≤ M' := by
sorry

end NUMINAMATH_CALUDE_min_value_of_max_sum_l2645_264544


namespace NUMINAMATH_CALUDE_equation_solution_l2645_264581

theorem equation_solution : 
  ∃ (x : ℚ), (3/4 : ℚ) + 4/x = 1 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2645_264581


namespace NUMINAMATH_CALUDE_nineteen_vectors_sum_zero_l2645_264577

theorem nineteen_vectors_sum_zero (v : Fin 19 → (Fin 3 → ZMod 3)) :
  ∃ i j k : Fin 19, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ v i + v j + v k = 0 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_vectors_sum_zero_l2645_264577


namespace NUMINAMATH_CALUDE_sally_fruit_spending_l2645_264569

/-- The total amount Sally spent on fruit --/
def total_spent (peach_price_after_coupon : ℝ) (peach_coupon : ℝ) (cherry_price : ℝ) (apple_price : ℝ) (apple_discount_percent : ℝ) : ℝ :=
  let peach_price := peach_price_after_coupon + peach_coupon
  let peach_and_cherry := peach_price + cherry_price
  let apple_discount := apple_price * apple_discount_percent
  let apple_price_discounted := apple_price - apple_discount
  peach_and_cherry + apple_price_discounted

/-- Theorem stating the total amount Sally spent on fruit --/
theorem sally_fruit_spending :
  total_spent 12.32 3 11.54 20 0.15 = 43.86 := by
  sorry

#eval total_spent 12.32 3 11.54 20 0.15

end NUMINAMATH_CALUDE_sally_fruit_spending_l2645_264569


namespace NUMINAMATH_CALUDE_fliers_remaining_l2645_264578

theorem fliers_remaining (initial_fliers : ℕ) 
  (morning_fraction : ℚ) (afternoon_fraction : ℚ) : 
  initial_fliers = 3000 →
  morning_fraction = 1/5 →
  afternoon_fraction = 1/4 →
  let remaining_after_morning := initial_fliers - (morning_fraction * initial_fliers).floor
  let final_remaining := remaining_after_morning - (afternoon_fraction * remaining_after_morning).floor
  final_remaining = 1800 := by
  sorry

end NUMINAMATH_CALUDE_fliers_remaining_l2645_264578


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l2645_264509

theorem geometric_sequence_second_term
  (a : ℕ+) -- first term
  (r : ℕ+) -- common ratio
  (h1 : a = 6)
  (h2 : a * r^3 = 768) :
  a * r = 24 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l2645_264509


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2645_264586

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: A point with negative x-coordinate and positive y-coordinate is in the second quadrant -/
theorem point_in_second_quadrant (p : Point) (hx : p.x < 0) (hy : p.y > 0) :
  SecondQuadrant p := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2645_264586


namespace NUMINAMATH_CALUDE_steven_skittles_count_l2645_264536

/-- The number of groups of Skittles in Steven's collection -/
def num_groups : ℕ := 77

/-- The number of Skittles in each group -/
def skittles_per_group : ℕ := 77

/-- The total number of Skittles in Steven's collection -/
def total_skittles : ℕ := num_groups * skittles_per_group

theorem steven_skittles_count : total_skittles = 5929 := by
  sorry

end NUMINAMATH_CALUDE_steven_skittles_count_l2645_264536


namespace NUMINAMATH_CALUDE_correct_ring_arrangements_l2645_264513

/-- The number of ways to arrange rings on fingers -/
def ring_arrangements (total_rings : ℕ) (rings_to_arrange : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings rings_to_arrange *
  Nat.choose (rings_to_arrange + fingers - 1) (fingers - 1) *
  Nat.factorial rings_to_arrange

/-- Theorem stating the correct number of ring arrangements -/
theorem correct_ring_arrangements :
  ring_arrangements 7 6 4 = 423360 := by
  sorry

end NUMINAMATH_CALUDE_correct_ring_arrangements_l2645_264513


namespace NUMINAMATH_CALUDE_remainder_sum_l2645_264505

theorem remainder_sum (n : ℤ) (h : n % 24 = 11) : (n % 4 + n % 6 = 8) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2645_264505


namespace NUMINAMATH_CALUDE_football_team_progress_l2645_264559

def football_progress (first_play : Int) (second_play : Int) : Int :=
  let third_play := -2 * (-first_play)
  let fourth_play := third_play / 2
  first_play + second_play + third_play + fourth_play

theorem football_team_progress :
  football_progress (-5) 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l2645_264559


namespace NUMINAMATH_CALUDE_additive_inverse_sum_zero_l2645_264560

theorem additive_inverse_sum_zero (x : ℝ) : x + (-x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverse_sum_zero_l2645_264560


namespace NUMINAMATH_CALUDE_trig_expression_equality_l2645_264571

theorem trig_expression_equality : 
  (Real.sin (24 * π / 180) * Real.cos (18 * π / 180) + Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) / 
  (Real.sin (28 * π / 180) * Real.cos (12 * π / 180) + Real.cos (152 * π / 180) * Real.cos (92 * π / 180)) = 
  Real.sin (18 * π / 180) / Real.sin (26 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l2645_264571


namespace NUMINAMATH_CALUDE_tickets_distribution_l2645_264566

theorem tickets_distribution (initial_tickets best_friend_tickets schoolmate_tickets remaining_tickets : ℕ) 
  (h1 : initial_tickets = 128)
  (h2 : best_friend_tickets = 7)
  (h3 : schoolmate_tickets = 4)
  (h4 : remaining_tickets = 11)
  : ∃ (best_friends schoolmates : ℕ), 
    initial_tickets = best_friend_tickets * best_friends + schoolmate_tickets * schoolmates + remaining_tickets ∧
    best_friends + schoolmates = 20 := by
  sorry

end NUMINAMATH_CALUDE_tickets_distribution_l2645_264566


namespace NUMINAMATH_CALUDE_cubic_identity_l2645_264517

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l2645_264517


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l2645_264537

theorem trigonometric_expression_value (m x : ℝ) (h : m * Real.tan x = 2) :
  (6 * m * Real.sin (2 * x) + 2 * m * Real.cos (2 * x)) /
  (m * Real.cos (2 * x) - 3 * m * Real.sin (2 * x)) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l2645_264537


namespace NUMINAMATH_CALUDE_exponential_equation_solutions_l2645_264533

theorem exponential_equation_solutions :
  ∀ a b c : ℕ, 2^a * 3^b = 7^c - 1 ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_exponential_equation_solutions_l2645_264533


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l2645_264557

/-- Given a point A with coordinates (-2, 3), this theorem proves that its symmetric point B
    with respect to the x-axis has coordinates (-2, -3). -/
theorem symmetric_point_wrt_x_axis :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (-2, -3)
  (A.1 = B.1) ∧ (A.2 = -B.2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l2645_264557


namespace NUMINAMATH_CALUDE_isosceles_triangles_not_necessarily_congruent_l2645_264570

/-- An isosceles triangle with acute angles -/
structure AcuteIsoscelesTriangle where
  /-- The length of the equal sides (legs) of the triangle -/
  legLength : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The angle at the apex of the triangle (in radians) -/
  apexAngle : ℝ
  /-- The apex angle is acute -/
  acuteAngle : apexAngle < Real.pi / 4
  /-- The leg length is positive -/
  legPositive : legLength > 0
  /-- The inradius is positive -/
  inradiusPositive : inradius > 0

/-- The theorem stating that two isosceles triangles with the same leg length and inradius
    are not necessarily congruent -/
theorem isosceles_triangles_not_necessarily_congruent :
  ∃ (t1 t2 : AcuteIsoscelesTriangle),
    t1.legLength = t2.legLength ∧
    t1.inradius = t2.inradius ∧
    t1.apexAngle ≠ t2.apexAngle :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangles_not_necessarily_congruent_l2645_264570


namespace NUMINAMATH_CALUDE_clothing_distribution_l2645_264588

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 59)
  (h2 : first_load = 32)
  (h3 : num_small_loads = 9)
  (h4 : first_load < total) :
  (total - first_load) / num_small_loads = 3 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l2645_264588


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l2645_264539

/-- Given a quadratic function f(x) = 3x^2 + 2x + 1, when shifted 5 units to the right,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 41 -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, (3 * (x - 5)^2 + 2 * (x - 5) + 1) = (a * x^2 + b * x + c)) →
  a + b + c = 41 := by
sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l2645_264539


namespace NUMINAMATH_CALUDE_car_wash_earnings_ratio_l2645_264535

theorem car_wash_earnings_ratio (total : ℕ) (lisa tommy : ℕ) : 
  total = 60 →
  lisa = total / 2 →
  lisa = tommy + 15 →
  Nat.gcd tommy lisa = tommy →
  (tommy : ℚ) / lisa = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_car_wash_earnings_ratio_l2645_264535


namespace NUMINAMATH_CALUDE_unique_positive_zero_implies_a_less_than_negative_two_l2645_264592

/-- Given a cubic function f(x) = ax^3 - 3x^2 + 1 with a unique positive zero,
    prove that the coefficient a must be less than -2. -/
theorem unique_positive_zero_implies_a_less_than_negative_two 
  (a : ℝ) (x₀ : ℝ) (h_unique : ∀ x : ℝ, a * x^3 - 3 * x^2 + 1 = 0 ↔ x = x₀) 
  (h_positive : x₀ > 0) : 
  a < -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_zero_implies_a_less_than_negative_two_l2645_264592


namespace NUMINAMATH_CALUDE_jackson_spending_money_l2645_264504

/-- The amount of money earned per hour of chores -/
def money_per_hour : ℝ := 5

/-- The time spent vacuuming (in hours) -/
def vacuuming_time : ℝ := 2 * 2

/-- The time spent washing dishes (in hours) -/
def dish_washing_time : ℝ := 0.5

/-- The time spent cleaning the bathroom (in hours) -/
def bathroom_cleaning_time : ℝ := 3 * dish_washing_time

/-- The total time spent on chores (in hours) -/
def total_chore_time : ℝ := vacuuming_time + dish_washing_time + bathroom_cleaning_time

/-- The theorem stating that Jackson's earned spending money is $30 -/
theorem jackson_spending_money : money_per_hour * total_chore_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_jackson_spending_money_l2645_264504


namespace NUMINAMATH_CALUDE_fraction_equality_l2645_264587

theorem fraction_equality : (3+9-27+81-243+729)/(9+27-81+243-729+2187) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2645_264587


namespace NUMINAMATH_CALUDE_find_A_l2645_264572

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_single_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def round_down_hundreds (n : ℕ) : ℕ := (n / 100) * 100

theorem find_A (A : ℕ) :
  is_single_digit A →
  is_three_digit (A * 100 + 27) →
  round_down_hundreds (A * 100 + 27) = 200 →
  A = 2 := by sorry

end NUMINAMATH_CALUDE_find_A_l2645_264572


namespace NUMINAMATH_CALUDE_paige_homework_problems_l2645_264596

/-- The initial number of homework problems Paige had -/
def initial_problems (finished : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished + remaining_pages * problems_per_page

/-- Theorem stating that Paige initially had 110 homework problems -/
theorem paige_homework_problems :
  initial_problems 47 7 9 = 110 := by
  sorry

end NUMINAMATH_CALUDE_paige_homework_problems_l2645_264596


namespace NUMINAMATH_CALUDE_volleyball_team_combinations_l2645_264506

def total_players : ℕ := 16
def num_triplets : ℕ := 3
def num_twins : ℕ := 2
def starters : ℕ := 6

def choose_two_triplets : ℕ := Nat.choose num_triplets 2
def remaining_after_triplets : ℕ := total_players - num_triplets + 1
def choose_rest_with_triplets : ℕ := Nat.choose remaining_after_triplets (starters - 2)

def choose_twins : ℕ := 1
def remaining_after_twins : ℕ := total_players - num_twins
def choose_rest_with_twins : ℕ := Nat.choose remaining_after_twins (starters - 2)

theorem volleyball_team_combinations :
  choose_two_triplets * choose_rest_with_triplets + choose_twins * choose_rest_with_twins = 3146 :=
sorry

end NUMINAMATH_CALUDE_volleyball_team_combinations_l2645_264506


namespace NUMINAMATH_CALUDE_cos_300_degrees_l2645_264582

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l2645_264582


namespace NUMINAMATH_CALUDE_range_of_m_range_of_x_l2645_264500

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x + 1 ∧ x + 1 < 1 + m ∧ m > 0

-- Theorem 1
theorem range_of_m (m : ℝ) :
  (∀ x, ¬(p x) → ¬(q x m)) →
  0 < m ∧ m ≤ 1 :=
sorry

-- Theorem 2
theorem range_of_x (x : ℝ) :
  (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) →
  (-5 ≤ x ∧ x < -1) ∨ x = 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_x_l2645_264500


namespace NUMINAMATH_CALUDE_matrix_equation_l2645_264564

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -7; 11, 4]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![(44/7), -(57/7); -(49/14), (63/14)]

theorem matrix_equation : N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l2645_264564


namespace NUMINAMATH_CALUDE_sophia_estimate_l2645_264554

theorem sophia_estimate (x y a b : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : a > 0) (h4 : b > 0) :
  (x + a) - (y - b) > x - y := by
  sorry

end NUMINAMATH_CALUDE_sophia_estimate_l2645_264554


namespace NUMINAMATH_CALUDE_codger_shoe_purchase_l2645_264538

/-- Represents the number of feet a sloth has -/
def sloth_feet : ℕ := 3

/-- Represents the number of shoes in a complete set for a sloth -/
def complete_set : ℕ := 3

/-- Represents the number of shoes in a pair -/
def shoes_per_pair : ℕ := 2

/-- Represents the number of complete sets Codger wants to have -/
def desired_sets : ℕ := 7

/-- Represents the number of shoes Codger already owns -/
def owned_shoes : ℕ := 3

/-- Represents the constraint that shoes must be bought in even-numbered sets of pairs -/
def even_numbered_pairs (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- The main theorem -/
theorem codger_shoe_purchase :
  ∃ (pairs_to_buy : ℕ),
    pairs_to_buy * shoes_per_pair + owned_shoes ≥ desired_sets * complete_set ∧
    even_numbered_pairs pairs_to_buy ∧
    ∀ (n : ℕ), n < pairs_to_buy →
      n * shoes_per_pair + owned_shoes < desired_sets * complete_set ∨
      ¬(even_numbered_pairs n) :=
sorry

end NUMINAMATH_CALUDE_codger_shoe_purchase_l2645_264538


namespace NUMINAMATH_CALUDE_fabric_equation_correct_l2645_264549

/-- Represents the fabric purchase scenario --/
structure FabricPurchase where
  total_meters : ℝ
  total_cost : ℝ
  blue_cost_per_meter : ℝ
  black_cost_per_meter : ℝ

/-- The equation correctly represents the fabric purchase scenario --/
theorem fabric_equation_correct (fp : FabricPurchase)
  (h1 : fp.total_meters = 138)
  (h2 : fp.total_cost = 540)
  (h3 : fp.blue_cost_per_meter = 3)
  (h4 : fp.black_cost_per_meter = 5) :
  ∃ x : ℝ, fp.blue_cost_per_meter * x + fp.black_cost_per_meter * (fp.total_meters - x) = fp.total_cost :=
by sorry

end NUMINAMATH_CALUDE_fabric_equation_correct_l2645_264549


namespace NUMINAMATH_CALUDE_batsman_average_increase_proof_l2645_264553

def batsman_average_increase (total_innings : ℕ) (last_innings_score : ℕ) (final_average : ℚ) : ℚ :=
  let initial_total := (total_innings - 1) * (total_innings * final_average - last_innings_score) / total_innings
  let initial_average := initial_total / (total_innings - 1)
  final_average - initial_average

theorem batsman_average_increase_proof :
  batsman_average_increase 12 65 32 = 3 := by sorry

end NUMINAMATH_CALUDE_batsman_average_increase_proof_l2645_264553


namespace NUMINAMATH_CALUDE_arccos_cos_ten_l2645_264522

open Real

-- Define the problem statement
theorem arccos_cos_ten :
  let x := 10
  let y := arccos (cos x)
  0 ≤ y ∧ y ≤ π →
  y = x - 2 * π :=
by sorry

end NUMINAMATH_CALUDE_arccos_cos_ten_l2645_264522


namespace NUMINAMATH_CALUDE_half_power_inequality_l2645_264546

theorem half_power_inequality (a : ℝ) : 
  (1/2 : ℝ)^(2*a + 1) < (1/2 : ℝ)^(3 - 2*a) → a > 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l2645_264546


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2645_264519

theorem sufficient_not_necessary (x : ℝ) : 
  (x = 0 → x^2 - 2*x = 0) ∧ (∃ y : ℝ, y ≠ 0 ∧ y^2 - 2*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2645_264519


namespace NUMINAMATH_CALUDE_solve_system_for_p_l2645_264599

theorem solve_system_for_p (p q : ℚ) 
  (eq1 : 3 * p + 4 * q = 15) 
  (eq2 : 4 * p + 3 * q = 18) : 
  p = 27 / 7 := by sorry

end NUMINAMATH_CALUDE_solve_system_for_p_l2645_264599


namespace NUMINAMATH_CALUDE_selections_with_paperback_count_l2645_264542

/-- The number of books on the shelf -/
def total_books : ℕ := 7

/-- The number of paperback books -/
def paperbacks : ℕ := 2

/-- The number of hardback books -/
def hardbacks : ℕ := 5

/-- The number of possible selections that include at least one paperback -/
def selections_with_paperback : ℕ := 96

/-- Theorem stating that the number of selections with at least one paperback
    is equal to the total number of possible selections minus the number of
    selections with no paperbacks -/
theorem selections_with_paperback_count :
  selections_with_paperback = 2^total_books - 2^hardbacks :=
by sorry

end NUMINAMATH_CALUDE_selections_with_paperback_count_l2645_264542


namespace NUMINAMATH_CALUDE_fourth_power_sum_l2645_264521

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (square_sum_condition : a^2 + b^2 + c^2 = 3)
  (cube_sum_condition : a^3 + b^3 + c^3 = 4) :
  a^4 + b^4 + c^4 = 7.833 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l2645_264521


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2645_264552

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85 := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2645_264552


namespace NUMINAMATH_CALUDE_circular_arc_length_l2645_264576

/-- The length of a circular arc with radius 10 meters and central angle 120° is 20π/3 meters. -/
theorem circular_arc_length : 
  ∀ (r : ℝ) (θ : ℝ), 
  r = 10 → 
  θ = 2 * π / 3 → 
  r * θ = 20 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_circular_arc_length_l2645_264576


namespace NUMINAMATH_CALUDE_simplify_expression_l2645_264583

theorem simplify_expression : (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2645_264583
