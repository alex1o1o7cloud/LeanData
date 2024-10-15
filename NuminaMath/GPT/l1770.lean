import Mathlib

namespace NUMINAMATH_GPT_inequality_solution_l1770_177030

theorem inequality_solution (x : ℝ) : (x^3 - 10 * x^2 > -25 * x) ↔ (0 < x ∧ x < 5) ∨ (5 < x) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1770_177030


namespace NUMINAMATH_GPT_lines_perpendicular_l1770_177062

noncomputable def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem lines_perpendicular {m : ℝ} :
  is_perpendicular (m + 2) (1 - m) (m - 1) (2 * m + 3) ↔ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_lines_perpendicular_l1770_177062


namespace NUMINAMATH_GPT_circle_area_pi_l1770_177085

def circle_eq := ∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0 → (x + 2) ^ 2 + y ^ 2 = 1

theorem circle_area_pi (h : ∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0 → (x + 2) ^ 2 + y ^ 2 = 1) :
  ∃ S : ℝ, S = π :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_area_pi_l1770_177085


namespace NUMINAMATH_GPT_remaining_days_to_complete_job_l1770_177003

-- Define the given conditions
def in_10_days (part_of_job_done : ℝ) (days : ℕ) : Prop :=
  part_of_job_done = 1 / 8 ∧ days = 10

-- Define the complete job condition
def complete_job (total_days : ℕ) : Prop :=
  total_days = 80

-- Define the remaining days to finish the job
def remaining_days (total_days_worked : ℕ) (days_worked : ℕ) (remaining : ℕ) : Prop :=
  total_days_worked = 80 ∧ days_worked = 10 ∧ remaining = 70

-- The theorem statement
theorem remaining_days_to_complete_job (part_of_job_done : ℝ) (days : ℕ) (total_days : ℕ) (total_days_worked : ℕ) (days_worked : ℕ) (remaining : ℕ) :
  in_10_days part_of_job_done days → complete_job total_days → remaining_days total_days_worked days_worked remaining :=
sorry

end NUMINAMATH_GPT_remaining_days_to_complete_job_l1770_177003


namespace NUMINAMATH_GPT_find_x_minus_y_l1770_177073

theorem find_x_minus_y (x y n : ℤ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x > y) (h4 : n / 10 < 10 ∧ n / 10 ≥ 1) 
  (h5 : 2 * n = x + y) 
  (h6 : ∃ m : ℤ, m^2 = x * y ∧ m = (n % 10) * 10 + n / 10) 
  : x - y = 66 :=
sorry

end NUMINAMATH_GPT_find_x_minus_y_l1770_177073


namespace NUMINAMATH_GPT_union_eq_l1770_177037

def setA : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def setB : Set ℝ := { x | 2 < x ∧ x ≤ 5 }

theorem union_eq : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end NUMINAMATH_GPT_union_eq_l1770_177037


namespace NUMINAMATH_GPT_total_amount_paid_l1770_177012

theorem total_amount_paid (B : ℕ) (hB : B = 232) (A : ℕ) (hA : A = 3 / 2 * B) :
  A + B = 580 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1770_177012


namespace NUMINAMATH_GPT_missing_number_l1770_177057

theorem missing_number 
  (a : ℕ) (b : ℕ) (x : ℕ)
  (h1 : a = 105) 
  (h2 : a^3 = 21 * 25 * x * b) 
  (h3 : b = 147) : 
  x = 3 :=
sorry

end NUMINAMATH_GPT_missing_number_l1770_177057


namespace NUMINAMATH_GPT_correct_proposition_l1770_177047

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_proposition :
  ¬ (∀ x : ℝ, f (x + 2 * Real.pi) = f x) ∧
  ¬ (∀ h : ℝ, f (-Real.pi / 6 + h) = f (-Real.pi / 6 - h)) ∧
  (∀ h : ℝ, f (-5 * Real.pi / 12 + h) = f (-5 * Real.pi / 12 - h)) :=
by sorry

end NUMINAMATH_GPT_correct_proposition_l1770_177047


namespace NUMINAMATH_GPT_probability_is_one_third_l1770_177038

noncomputable def probability_four_of_a_kind_or_full_house : ℚ :=
  let total_outcomes := 6
  let probability_triplet_match := 1 / total_outcomes
  let probability_pair_match := 1 / total_outcomes
  probability_triplet_match + probability_pair_match

theorem probability_is_one_third :
  probability_four_of_a_kind_or_full_house = 1 / 3 :=
by
  -- sorry
  trivial

end NUMINAMATH_GPT_probability_is_one_third_l1770_177038


namespace NUMINAMATH_GPT_expression_factorization_l1770_177052

variables (a b c : ℝ)

theorem expression_factorization :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3)
  = (a - b) * (b - c) * (c - a) * (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
sorry

end NUMINAMATH_GPT_expression_factorization_l1770_177052


namespace NUMINAMATH_GPT_math_problem_l1770_177031

def is_perfect_square (a : ℕ) : Prop :=
  ∃ b : ℕ, a = b * b

theorem math_problem (a m : ℕ) (h1: m = 2992) (h2: a = m^2 + m^2 * (m+1)^2 + (m+1)^2) : is_perfect_square a :=
  sorry

end NUMINAMATH_GPT_math_problem_l1770_177031


namespace NUMINAMATH_GPT_find_side_length_l1770_177090

def hollow_cube_formula (n : ℕ) : ℕ :=
  6 * n^2 - (n^2 + 4 * (n - 2))

theorem find_side_length :
  ∃ n : ℕ, hollow_cube_formula n = 98 ∧ n = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_side_length_l1770_177090


namespace NUMINAMATH_GPT_range_of_x_l1770_177029

theorem range_of_x (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 := 
  sorry

end NUMINAMATH_GPT_range_of_x_l1770_177029


namespace NUMINAMATH_GPT_lizette_overall_average_is_94_l1770_177094

-- Defining the given conditions
def third_quiz_score : ℕ := 92
def first_two_quizzes_average : ℕ := 95
def total_quizzes : ℕ := 3

-- Calculating total points from the conditions
def total_points : ℕ := first_two_quizzes_average * 2 + third_quiz_score

-- Defining the overall average to prove
def overall_average : ℕ := total_points / total_quizzes

-- The theorem stating Lizette's overall average after taking the third quiz
theorem lizette_overall_average_is_94 : overall_average = 94 := by
  sorry

end NUMINAMATH_GPT_lizette_overall_average_is_94_l1770_177094


namespace NUMINAMATH_GPT_find_percentage_l1770_177075

theorem find_percentage (P : ℝ) : 
  0.15 * P * (0.5 * 5600) = 126 → P = 0.3 := 
by 
  sorry

end NUMINAMATH_GPT_find_percentage_l1770_177075


namespace NUMINAMATH_GPT_abs_x_minus_one_eq_one_minus_x_implies_x_le_one_l1770_177080

theorem abs_x_minus_one_eq_one_minus_x_implies_x_le_one (x : ℝ) (h : |x - 1| = 1 - x) : x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_abs_x_minus_one_eq_one_minus_x_implies_x_le_one_l1770_177080


namespace NUMINAMATH_GPT_unique_solution_l1770_177053

def is_prime (n : ℕ) : Prop := Nat.Prime n

def eq_triple (m p q : ℕ) : Prop :=
  2 ^ m * p ^ 2 + 1 = q ^ 5

theorem unique_solution (m p q : ℕ) (h1 : m > 0) (h2 : is_prime p) (h3 : is_prime q) :
  eq_triple m p q ↔ (m, p, q) = (1, 11, 3) := by
  sorry

end NUMINAMATH_GPT_unique_solution_l1770_177053


namespace NUMINAMATH_GPT_solve_for_x_l1770_177042

theorem solve_for_x (x : ℝ) (h : (1/3 : ℝ) * (x + 8 + 5*x + 3 + 3*x + 4) = 4*x + 1) : x = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1770_177042


namespace NUMINAMATH_GPT_complete_square_solution_l1770_177072

theorem complete_square_solution :
  ∀ x : ℝ, ∃ p q : ℝ, (5 * x^2 - 30 * x - 45 = 0) → ((x + p) ^ 2 = q) ∧ (p + q = 15) :=
by
  sorry

end NUMINAMATH_GPT_complete_square_solution_l1770_177072


namespace NUMINAMATH_GPT_cost_price_per_meter_l1770_177006

theorem cost_price_per_meter (total_cost : ℝ) (total_length : ℝ) (h1 : total_cost = 397.75) (h2 : total_length = 9.25) : total_cost / total_length = 43 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l1770_177006


namespace NUMINAMATH_GPT_probability_of_next_satisfied_customer_l1770_177077

noncomputable def probability_of_satisfied_customer : ℝ :=
  let p := (0.8 : ℝ)
  let q := (0.15 : ℝ)
  let neg_reviews := (60 : ℝ)
  let pos_reviews := (20 : ℝ)
  p / (p + q) * (q / (q + p))

theorem probability_of_next_satisfied_customer :
  probability_of_satisfied_customer = 0.64 :=
sorry

end NUMINAMATH_GPT_probability_of_next_satisfied_customer_l1770_177077


namespace NUMINAMATH_GPT_gcd_polynomials_l1770_177019

def even_multiple_of_2927 (a : ℤ) : Prop := ∃ k : ℤ, a = 2 * 2927 * k

theorem gcd_polynomials (a : ℤ) (h : even_multiple_of_2927 a) :
  Int.gcd (3 * a ^ 2 + 61 * a + 143) (a + 19) = 7 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomials_l1770_177019


namespace NUMINAMATH_GPT_stream_speed_l1770_177045

def boat_speed_still : ℝ := 30
def distance_downstream : ℝ := 80
def distance_upstream : ℝ := 40

theorem stream_speed (v : ℝ) (h : (distance_downstream / (boat_speed_still + v) = distance_upstream / (boat_speed_still - v))) :
  v = 10 :=
sorry

end NUMINAMATH_GPT_stream_speed_l1770_177045


namespace NUMINAMATH_GPT_arithmetic_seq_max_n_l1770_177089

def arithmetic_seq_max_sum (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 > 0) ∧ (3 * (a 1 + 4 * d) = 5 * (a 1 + 7 * d)) ∧
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) ∧
  (S 12 = -72 * d)

theorem arithmetic_seq_max_n
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : 
  arithmetic_seq_max_sum a d S → n = 12 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_max_n_l1770_177089


namespace NUMINAMATH_GPT_ratio_of_a_to_c_l1770_177097

theorem ratio_of_a_to_c (a b c d : ℚ)
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 3) 
  (h3 : d / b = 1 / 5) : a / c = 75 / 16 := 
sorry

end NUMINAMATH_GPT_ratio_of_a_to_c_l1770_177097


namespace NUMINAMATH_GPT_unique_solution_triple_l1770_177067

theorem unique_solution_triple {a b c : ℝ} (h₀ : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h₁ : a^2 + b^2 + c^2 = 3) (h₂ : (a + b + c) * (a^2 * b + b^2 * c + c^2 * a) = 9) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ c = 1 ∧ b = 1) ∨ (b = 1 ∧ a = 1 ∧ c = 1) ∨ (b = 1 ∧ c = 1 ∧ a = 1) ∨ (c = 1 ∧ a = 1 ∧ b = 1) ∨ (c = 1 ∧ b = 1 ∧ a = 1) :=
sorry

end NUMINAMATH_GPT_unique_solution_triple_l1770_177067


namespace NUMINAMATH_GPT_linear_equation_solution_l1770_177099

theorem linear_equation_solution (m : ℝ) (x : ℝ) (h : |m| - 2 = 1) (h_ne : m ≠ 3) :
  (2 * m - 6) * x^(|m|-2) = m^2 ↔ x = -(3/4) :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_solution_l1770_177099


namespace NUMINAMATH_GPT_nicholas_paid_more_than_kenneth_l1770_177061

def price_per_yard : ℝ := 40
def kenneth_yards : ℝ := 700
def nicholas_multiplier : ℝ := 6
def discount_rate : ℝ := 0.15

def kenneth_total_cost : ℝ := price_per_yard * kenneth_yards
def nicholas_yards : ℝ := nicholas_multiplier * kenneth_yards
def nicholas_original_cost : ℝ := price_per_yard * nicholas_yards
def discount_amount : ℝ := discount_rate * nicholas_original_cost
def nicholas_discounted_cost : ℝ := nicholas_original_cost - discount_amount
def difference_in_cost : ℝ := nicholas_discounted_cost - kenneth_total_cost

theorem nicholas_paid_more_than_kenneth :
  difference_in_cost = 114800 := by
  sorry

end NUMINAMATH_GPT_nicholas_paid_more_than_kenneth_l1770_177061


namespace NUMINAMATH_GPT_A_squared_plus_B_squared_eq_one_l1770_177081

theorem A_squared_plus_B_squared_eq_one
  (A B : ℝ) (h1 : A ≠ B)
  (h2 : ∀ x : ℝ, (A * (B * x ^ 2 + A) ^ 2 + B - (B * (A * x ^ 2 + B) ^ 2 + A)) = B ^ 2 - A ^ 2) :
  A ^ 2 + B ^ 2 = 1 :=
sorry

end NUMINAMATH_GPT_A_squared_plus_B_squared_eq_one_l1770_177081


namespace NUMINAMATH_GPT_cos_double_angle_l1770_177014

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end NUMINAMATH_GPT_cos_double_angle_l1770_177014


namespace NUMINAMATH_GPT_no_real_roots_range_l1770_177044

theorem no_real_roots_range (a : ℝ) : (¬ ∃ x : ℝ, x^2 + a * x - 4 * a = 0) ↔ (-16 < a ∧ a < 0) := by
  sorry

end NUMINAMATH_GPT_no_real_roots_range_l1770_177044


namespace NUMINAMATH_GPT_Ron_book_picking_times_l1770_177087

theorem Ron_book_picking_times (couples members : ℕ) (weeks people : ℕ) (Ron wife picks_per_year : ℕ) 
  (h1 : couples = 3) 
  (h2 : members = 5) 
  (h3 : Ron = 1) 
  (h4 : wife = 1) 
  (h5 : weeks = 52) 
  (h6 : people = 2 * couples + members + Ron + wife) 
  (h7 : picks_per_year = weeks / people) 
  : picks_per_year = 4 :=
by
  -- Definition steps can be added here if needed, currently immediate from conditions h1 to h7
  sorry

end NUMINAMATH_GPT_Ron_book_picking_times_l1770_177087


namespace NUMINAMATH_GPT_original_price_l1770_177020

theorem original_price (P : ℝ) (final_price : ℝ) (percent_increase : ℝ) (h1 : final_price = 450) (h2 : percent_increase = 0.50) : 
  P + percent_increase * P = final_price → P = 300 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1770_177020


namespace NUMINAMATH_GPT_hyperbola_condition_l1770_177079

theorem hyperbola_condition (m : ℝ) : 
  (∃ a b : ℝ, a = m + 4 ∧ b = m - 3 ∧ (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)) ↔ m > 3 :=
sorry

end NUMINAMATH_GPT_hyperbola_condition_l1770_177079


namespace NUMINAMATH_GPT_solve_x_l1770_177015

theorem solve_x (x : ℝ) (h : x^2 + 6 * x + 8 = -(x + 4) * (x + 6)) : 
  x = -4 := 
by
  sorry

end NUMINAMATH_GPT_solve_x_l1770_177015


namespace NUMINAMATH_GPT_quadratic_roots_l1770_177092

-- Definitions based on the conditions provided.
def condition1 (x y : ℝ) : Prop := x^2 - 6 * x + 9 = -(abs (y - 1))

-- The main theorem we want to prove.
theorem quadratic_roots (x y : ℝ) (h : condition1 x y) : (a : ℝ) → (a - 3) * (a - 1) = a^2 - 4 * a + 3 :=
  by sorry

end NUMINAMATH_GPT_quadratic_roots_l1770_177092


namespace NUMINAMATH_GPT_initial_average_marks_l1770_177034

theorem initial_average_marks (A : ℝ) (h1 : 25 * A - 50 = 2450) : A = 100 :=
by
  sorry

end NUMINAMATH_GPT_initial_average_marks_l1770_177034


namespace NUMINAMATH_GPT_a_formula_b_formula_T_formula_l1770_177025

variable {n : ℕ}

def S (n : ℕ) := 2 * n^2

def a (n : ℕ) : ℕ := 
  if n = 1 then S 1 else S n - S (n - 1)

def b (n : ℕ) : ℕ := 
  if n = 1 then 2 else 2 * (1 / 4 ^ (n - 1))

def c (n : ℕ) : ℕ := (4 * n - 2) / (2 * 4 ^ (n - 1))

def T (n : ℕ) : ℕ := 
  (1 / 9) * ((6 * n - 5) * (4 ^ n) + 5)

theorem a_formula :
  ∀ n, a n = 4 * n - 2 := 
sorry

theorem b_formula :
  ∀ n, b n = 2 / (4 ^ (n - 1)) :=
sorry

theorem T_formula :
  ∀ n, T n = (1 / 9) * ((6 * n - 5) * (4 ^ n) + 5) :=
sorry

end NUMINAMATH_GPT_a_formula_b_formula_T_formula_l1770_177025


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l1770_177002

theorem area_of_triangle_ABC (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let x1 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let y_vertex := (4 * a * c - b^2) / (4 * a)
  0.5 * (|x2 - x1|) * |y_vertex| = (b^2 - 4 * a * c) * Real.sqrt (b^2 - 4 * a * c) / (8 * a^2) :=
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l1770_177002


namespace NUMINAMATH_GPT_circle_diameter_l1770_177018

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end NUMINAMATH_GPT_circle_diameter_l1770_177018


namespace NUMINAMATH_GPT_set_equality_l1770_177040

def P : Set ℝ := { x | x^2 = 1 }

theorem set_equality : P = {-1, 1} :=
by
  sorry

end NUMINAMATH_GPT_set_equality_l1770_177040


namespace NUMINAMATH_GPT_junior_score_is_95_l1770_177058

theorem junior_score_is_95:
  ∀ (n j s : ℕ) (x avg_total avg_seniors : ℕ),
    n = 20 →
    j = n * 15 / 100 →
    s = n * 85 / 100 →
    avg_total = 78 →
    avg_seniors = 75 →
    (j * x + s * avg_seniors) / n = avg_total →
    x = 95 :=
by
  sorry

end NUMINAMATH_GPT_junior_score_is_95_l1770_177058


namespace NUMINAMATH_GPT_area_of_inscribed_square_l1770_177069

-- Define the right triangle with segments m and n on the hypotenuse
variables {m n : ℝ}

-- Noncomputable setting for non-constructive aspects
noncomputable def inscribed_square_area (m n : ℝ) : ℝ :=
  (m * n)

-- Theorem stating that the area of the inscribed square is m * n
theorem area_of_inscribed_square (m n : ℝ) : inscribed_square_area m n = m * n :=
by sorry

end NUMINAMATH_GPT_area_of_inscribed_square_l1770_177069


namespace NUMINAMATH_GPT_compute_fraction_power_l1770_177068

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end NUMINAMATH_GPT_compute_fraction_power_l1770_177068


namespace NUMINAMATH_GPT_coin_flip_probability_l1770_177095

/--
Suppose we flip five coins simultaneously: a penny, a nickel, a dime, a quarter, and a half-dollar.
What is the probability that the penny and dime both come up heads, and the half-dollar comes up tails?
-/

theorem coin_flip_probability :
  let outcomes := 2^5
  let success := 1 * 1 * 1 * 2 * 2
  success / outcomes = (1 : ℚ) / 8 :=
by
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_coin_flip_probability_l1770_177095


namespace NUMINAMATH_GPT_geometric_series_sum_l1770_177016

noncomputable def first_term : ℝ := 6
noncomputable def common_ratio : ℝ := -2 / 3

theorem geometric_series_sum :
  (|common_ratio| < 1) → (first_term / (1 - common_ratio) = 18 / 5) :=
by
  intros h
  simp [first_term, common_ratio]
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1770_177016


namespace NUMINAMATH_GPT_solve_equation_l1770_177039

def equation_solution (x : ℝ) : Prop :=
  (x^2 + x + 1) / (x + 1) = x + 3

theorem solve_equation :
  ∃ x : ℝ, equation_solution x ∧ x = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1770_177039


namespace NUMINAMATH_GPT_min_washes_l1770_177065

theorem min_washes (x : ℕ) :
  (1 / 4)^x ≤ 1 / 100 → x ≥ 4 :=
by sorry

end NUMINAMATH_GPT_min_washes_l1770_177065


namespace NUMINAMATH_GPT_max_value_pq_qr_rs_sp_l1770_177066

variable (p q r s : ℕ)

theorem max_value_pq_qr_rs_sp :
  (p = 1 ∨ p = 3 ∨ p = 5 ∨ p = 7) →
  (q = 1 ∨ q = 3 ∨ q = 5 ∨ q = 7) →
  (r = 1 ∨ r = 3 ∨ r = 5 ∨ r = 7) →
  (s = 1 ∨ s = 3 ∨ s = 5 ∨ s = 7) →
  (p ≠ q) →
  (p ≠ r) →
  (p ≠ s) →
  (q ≠ r) →
  (q ≠ s) →
  (r ≠ s) →
  pq + qr + rs + sp ≤ 64 :=
sorry

end NUMINAMATH_GPT_max_value_pq_qr_rs_sp_l1770_177066


namespace NUMINAMATH_GPT_distance_between_trees_l1770_177055

def yard_length : ℕ := 414
def number_of_trees : ℕ := 24

theorem distance_between_trees : yard_length / (number_of_trees - 1) = 18 := 
by sorry

end NUMINAMATH_GPT_distance_between_trees_l1770_177055


namespace NUMINAMATH_GPT_students_with_both_pets_l1770_177013

theorem students_with_both_pets :
  ∀ (total_students students_with_dog students_with_cat students_with_both : ℕ),
    total_students = 45 →
    students_with_dog = 25 →
    students_with_cat = 34 →
    total_students = students_with_dog + students_with_cat - students_with_both →
    students_with_both = 14 :=
by
  intros total_students students_with_dog students_with_cat students_with_both
  sorry

end NUMINAMATH_GPT_students_with_both_pets_l1770_177013


namespace NUMINAMATH_GPT_open_spots_level4_correct_l1770_177041

noncomputable def open_spots_level_4 (total_levels : ℕ) (spots_per_level : ℕ) (open_spots_level1 : ℕ) (open_spots_level2 : ℕ) (open_spots_level3 : ℕ) (full_spots_total : ℕ) : ℕ := 
  let total_spots := total_levels * spots_per_level
  let open_spots_total := total_spots - full_spots_total 
  let open_spots_first_three := open_spots_level1 + open_spots_level2 + open_spots_level3
  open_spots_total - open_spots_first_three

theorem open_spots_level4_correct :
  open_spots_level_4 4 100 58 (58 + 2) (58 + 2 + 5) 186 = 31 :=
by
  sorry

end NUMINAMATH_GPT_open_spots_level4_correct_l1770_177041


namespace NUMINAMATH_GPT_find_m_l1770_177005

variable (a b m : ℝ)

def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem find_m 
  (h₁ : right_triangle a b 5)
  (h₂ : a + b = 2*m - 1)
  (h₃ : a * b = 4 * (m - 1)) : 
  m = 4 := 
sorry

end NUMINAMATH_GPT_find_m_l1770_177005


namespace NUMINAMATH_GPT_second_number_in_pair_l1770_177028

theorem second_number_in_pair (n m : ℕ) (h1 : (n, m) = (57, 58)) (h2 : ∃ (n m : ℕ), n < 1500 ∧ m < 1500 ∧ (n + m) % 5 = 0) : m = 58 :=
by {
  sorry
}

end NUMINAMATH_GPT_second_number_in_pair_l1770_177028


namespace NUMINAMATH_GPT_polynomial_simplification_l1770_177063

theorem polynomial_simplification (s : ℝ) : (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 4) = s^2 - 4 * s + 1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l1770_177063


namespace NUMINAMATH_GPT_expression_undefined_count_l1770_177051

theorem expression_undefined_count (x : ℝ) :
  ∃! x, (x - 1) * (x + 3) * (x - 3) = 0 :=
sorry

end NUMINAMATH_GPT_expression_undefined_count_l1770_177051


namespace NUMINAMATH_GPT_math_problem_l1770_177083

theorem math_problem :
  (Int.ceil ((18: ℚ) / 5 * (-25 / 4)) - Int.floor ((18 / 5) * Int.floor (-25 / 4))) = 4 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1770_177083


namespace NUMINAMATH_GPT_distinct_natural_numbers_circles_sum_equal_impossible_l1770_177001

theorem distinct_natural_numbers_circles_sum_equal_impossible :
  ¬∃ (f : ℕ → ℕ) (distinct : ∀ i j, i ≠ j → f i ≠ f j) (equal_sum : ∀ i j k, (f i + f j + f k = f (i+1) + f (j+1) + f (k+1))),
  true :=
  sorry

end NUMINAMATH_GPT_distinct_natural_numbers_circles_sum_equal_impossible_l1770_177001


namespace NUMINAMATH_GPT_dice_probability_sum_15_l1770_177009

def is_valid_combination (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 15

def count_outcomes : ℕ :=
  6 * 6 * 6

def count_valid_combinations : ℕ :=
  10  -- From the list of valid combinations

def probability (valid_count total_count : ℕ) : ℚ :=
  valid_count / total_count

theorem dice_probability_sum_15 : probability count_valid_combinations count_outcomes = 5 / 108 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_sum_15_l1770_177009


namespace NUMINAMATH_GPT_decreasing_interval_l1770_177032

noncomputable def func (x : ℝ) := 2 * x^3 - 6 * x^2 + 11

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → deriv func x < 0 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_l1770_177032


namespace NUMINAMATH_GPT_unique_solution_pairs_count_l1770_177026

theorem unique_solution_pairs_count :
  ∃! (p : ℝ × ℝ), (p.1 + 2 * p.2 = 2 ∧ (|abs p.1 - 2 * abs p.2| = 2) ∧
       ∃! q, (q = (2, 0) ∨ q = (0, 1)) ∧ p = q) := 
sorry

end NUMINAMATH_GPT_unique_solution_pairs_count_l1770_177026


namespace NUMINAMATH_GPT_binary_to_decimal_l1770_177056

-- Define the binary number 10011_2
def binary_10011 : ℕ := bit0 (bit1 (bit1 (bit0 (bit1 0))))

-- Define the expected decimal value
def decimal_19 : ℕ := 19

-- State the theorem to convert binary 10011 to decimal
theorem binary_to_decimal :
  binary_10011 = decimal_19 :=
sorry

end NUMINAMATH_GPT_binary_to_decimal_l1770_177056


namespace NUMINAMATH_GPT_angles_with_same_terminal_side_as_15_degree_l1770_177076

def condition1 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 90
def condition2 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 180
def condition3 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 360
def condition4 (β : ℝ) (k : ℤ) : Prop := β = 15 + 2 * k * 360

def has_same_terminal_side_as_15_degree (β : ℝ) : Prop :=
  ∃ k : ℤ, β = 15 + k * 360

theorem angles_with_same_terminal_side_as_15_degree (β : ℝ) :
  (∃ k : ℤ, condition1 β k)  ∨
  (∃ k : ℤ, condition2 β k)  ∨
  (∃ k : ℤ, condition3 β k)  ∨
  (∃ k : ℤ, condition4 β k) →
  has_same_terminal_side_as_15_degree β :=
by
  sorry

end NUMINAMATH_GPT_angles_with_same_terminal_side_as_15_degree_l1770_177076


namespace NUMINAMATH_GPT_units_digit_division_l1770_177043

theorem units_digit_division (a b c d e denom : ℕ)
  (h30 : a = 30) (h31 : b = 31) (h32 : c = 32) (h33 : d = 33) (h34 : e = 34)
  (h120 : denom = 120) :
  ((a * b * c * d * e) / denom) % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_division_l1770_177043


namespace NUMINAMATH_GPT_inequality_abc_l1770_177004

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l1770_177004


namespace NUMINAMATH_GPT_street_tree_fourth_point_l1770_177064

theorem street_tree_fourth_point (a b : ℝ) (h_a : a = 0.35) (h_b : b = 0.37) :
  (a + 4 * ((b - a) / 4)) = b :=
by 
  rw [h_a, h_b]
  sorry

end NUMINAMATH_GPT_street_tree_fourth_point_l1770_177064


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1770_177059

-- For given real numbers x and y
-- Prove the statement "at least one of x and y is greater than 1" is not necessary and not sufficient for x^2 + y^2 > 2.
noncomputable def at_least_one_gt_one (x y : ℝ) : Prop := (x > 1) ∨ (y > 1)
def sum_of_squares_gt_two (x y : ℝ) : Prop := x^2 + y^2 > 2

theorem neither_sufficient_nor_necessary (x y : ℝ) :
  ¬(at_least_one_gt_one x y → sum_of_squares_gt_two x y) ∧ ¬(sum_of_squares_gt_two x y → at_least_one_gt_one x y) :=
by
  sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1770_177059


namespace NUMINAMATH_GPT_consecutive_roots_prime_q_l1770_177088

theorem consecutive_roots_prime_q (p q : ℤ) (h1 : Prime q)
  (h2 : ∃ x1 x2 : ℤ, 
    x1 ≠ x2 ∧ 
    (x1 = x2 + 1 ∨ x1 = x2 - 1) ∧ 
    x1 + x2 = p ∧ 
    x1 * x2 = q) : (p = 3 ∨ p = -3) ∧ q = 2 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_roots_prime_q_l1770_177088


namespace NUMINAMATH_GPT_find_b_when_a_is_1600_l1770_177078

theorem find_b_when_a_is_1600 :
  ∀ (a b : ℝ), (a * b = 400) ∧ ((2 * a) * b = 600) → (1600 * b = 600) → b = 0.375 :=
by
  intro a b
  intro h
  sorry

end NUMINAMATH_GPT_find_b_when_a_is_1600_l1770_177078


namespace NUMINAMATH_GPT_largest_three_digit_number_l1770_177011

theorem largest_three_digit_number (a b c : ℕ) (h1 : a = 8) (h2 : b = 0) (h3 : c = 7) :
  ∃ (n : ℕ), ∀ (x : ℕ), (x = a * 100 + b * 10 + c) → x = 870 :=
by
  sorry

end NUMINAMATH_GPT_largest_three_digit_number_l1770_177011


namespace NUMINAMATH_GPT_sqrt_sum_l1770_177000

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end NUMINAMATH_GPT_sqrt_sum_l1770_177000


namespace NUMINAMATH_GPT_stratified_sampling_num_of_female_employees_l1770_177008

theorem stratified_sampling_num_of_female_employees :
  ∃ (total_employees male_employees sample_size female_employees_to_draw : ℕ),
    total_employees = 750 ∧
    male_employees = 300 ∧
    sample_size = 45 ∧
    female_employees_to_draw = (total_employees - male_employees) * sample_size / total_employees ∧
    female_employees_to_draw = 27 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_num_of_female_employees_l1770_177008


namespace NUMINAMATH_GPT_problem_1_problem_2_l1770_177049

theorem problem_1 : ((1 / 3 - 3 / 4 + 5 / 6) / (1 / 12)) = 5 := 
  sorry

theorem problem_2 : ((-1 : ℤ) ^ 2023 + |(1 : ℝ) - 0.5| * (-4 : ℝ) ^ 2) = 7 := 
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1770_177049


namespace NUMINAMATH_GPT_kyoko_bought_three_balls_l1770_177024

theorem kyoko_bought_three_balls
  (cost_per_ball : ℝ)
  (total_paid : ℝ)
  (number_of_balls : ℝ)
  (h_cost_per_ball : cost_per_ball = 1.54)
  (h_total_paid : total_paid = 4.62)
  (h_number_of_balls : number_of_balls = total_paid / cost_per_ball) :
  number_of_balls = 3 := by
  sorry

end NUMINAMATH_GPT_kyoko_bought_three_balls_l1770_177024


namespace NUMINAMATH_GPT_corner_coloring_condition_l1770_177021

theorem corner_coloring_condition 
  (n : ℕ) 
  (h1 : n ≥ 5) 
  (board : ℕ → ℕ → Prop) -- board(i, j) = true if cell (i, j) is black, false if white
  (h2 : ∀ i j, board i j = board (i + 1) j → board (i + 2) j = board (i + 1) j → ¬(board i j = board (i + 2) j)) -- row condition
  (h3 : ∀ i j, board i j = board i (j + 1) → board i (j + 2) = board i (j + 1) → ¬(board i j = board i (j + 2))) -- column condition
  (h4 : ∀ i j, board i j = board (i + 1) (j + 1) → board (i + 2) (j + 2) = board (i + 1) (j + 1) → ¬(board i j = board (i + 2) (j + 2))) -- diagonal condition
  (h5 : ∀ i j, board (i + 2) j = board (i + 1) (j + 1) → board (i + 2) (j + 2) = board (i + 1) (j + 1) → ¬(board (i + 2) j = board (i + 2) (j + 2))) -- anti-diagonal condition
  : ∀ i j, i + 2 < n ∧ j + 2 < n → ((board i j ∧ board (i + 2) (j + 2)) ∨ (board i (j + 2) ∧ board (i + 2) j)) :=
sorry

end NUMINAMATH_GPT_corner_coloring_condition_l1770_177021


namespace NUMINAMATH_GPT_initial_amount_l1770_177007

theorem initial_amount (X : ℝ) (h1 : 0.70 * X = 2800) : X = 4000 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_l1770_177007


namespace NUMINAMATH_GPT_next_ring_together_l1770_177054

def nextRingTime (libraryInterval : ℕ) (fireStationInterval : ℕ) (hospitalInterval : ℕ) (start : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm libraryInterval fireStationInterval) hospitalInterval + start

theorem next_ring_together : nextRingTime 18 24 30 (8 * 60) = 14 * 60 :=
by
  sorry

end NUMINAMATH_GPT_next_ring_together_l1770_177054


namespace NUMINAMATH_GPT_total_percent_decrease_l1770_177036

theorem total_percent_decrease (initial_value first_year_decrease second_year_decrease third_year_decrease : ℝ)
  (h₁ : first_year_decrease = 0.30)
  (h₂ : second_year_decrease = 0.10)
  (h₃ : third_year_decrease = 0.20) :
  let value_after_first_year := initial_value * (1 - first_year_decrease)
  let value_after_second_year := value_after_first_year * (1 - second_year_decrease)
  let value_after_third_year := value_after_second_year * (1 - third_year_decrease)
  let total_decrease := initial_value - value_after_third_year
  let total_percent_decrease := (total_decrease / initial_value) * 100
  total_percent_decrease = 49.60 := 
by
  sorry

end NUMINAMATH_GPT_total_percent_decrease_l1770_177036


namespace NUMINAMATH_GPT_even_increasing_function_inequality_l1770_177060

theorem even_increasing_function_inequality
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_increasing : ∀ {x₁ x₂ : ℝ}, x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end NUMINAMATH_GPT_even_increasing_function_inequality_l1770_177060


namespace NUMINAMATH_GPT_infinitely_many_n_l1770_177071

theorem infinitely_many_n (p : ℕ) (hp : p.Prime) (hp2 : p % 2 = 1) :
  ∃ᶠ n in at_top, p ∣ n * 2^n + 1 :=
sorry

end NUMINAMATH_GPT_infinitely_many_n_l1770_177071


namespace NUMINAMATH_GPT_min_value_5x_plus_6y_l1770_177070

theorem min_value_5x_plus_6y (x y : ℝ) (h : 3 * x ^ 2 + 3 * y ^ 2 = 20 * x + 10 * y + 10) : 
  ∃ x y, (5 * x + 6 * y = 122) :=
by
  sorry

end NUMINAMATH_GPT_min_value_5x_plus_6y_l1770_177070


namespace NUMINAMATH_GPT_watermelon_cost_100_l1770_177035

variable (a b : ℕ) -- costs of one watermelon and one melon respectively
variable (x : ℕ) -- number of watermelons in the container

theorem watermelon_cost_100 :
  (∀ x, (1 : ℚ) = x / 160 + (150 - x) / 120 ∧ 120 * a = 30 * b ∧ 120 * a + 30 * b = 24000 ∧ x = 120) →
  a = 100 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_watermelon_cost_100_l1770_177035


namespace NUMINAMATH_GPT_problem_I_problem_II_l1770_177050

/-- Proof problem I: Given f(x) = |x - 1|, prove that the inequality f(x) ≥ 4 - |x - 1| implies x ≥ 3 or x ≤ -1 -/
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) (h2 : f x ≥ 4 - |x - 1|) : x ≥ 3 ∨ x ≤ -1 :=
  sorry

/-- Proof problem II: Given f(x) = |x - 1| and 1/m + 1/(2*n) = 1 (m > 0, n > 0), prove that the minimum value of mn is 2 -/
theorem problem_II (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) (m n : ℝ) (hm : m > 0) (hn : n > 0) (h2 : 1/m + 1/(2*n) = 1) : m*n ≥ 2 :=
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1770_177050


namespace NUMINAMATH_GPT_find_n_l1770_177023

-- Define the operation ø
def op (x w : ℕ) : ℕ := (2 ^ x) / (2 ^ w)

-- Prove that n operating with 2 and then 1 equals 8 implies n = 3
theorem find_n (n : ℕ) (H : op (op n 2) 1 = 8) : n = 3 :=
by
  -- Proof will be provided later
  sorry

end NUMINAMATH_GPT_find_n_l1770_177023


namespace NUMINAMATH_GPT_oranges_in_bin_l1770_177022

variable (n₀ n_throw n_new : ℕ)

theorem oranges_in_bin (h₀ : n₀ = 50) (h_throw : n_throw = 40) (h_new : n_new = 24) : 
  n₀ - n_throw + n_new = 34 := 
by 
  sorry

end NUMINAMATH_GPT_oranges_in_bin_l1770_177022


namespace NUMINAMATH_GPT_find_digits_l1770_177033

theorem find_digits (x y z : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) (h3 : 0 ≤ z ∧ z ≤ 9) :
  (10 * x + 5) * (3 * 100 + y * 10 + z) = 7850 ↔ (x = 2 ∧ y = 1 ∧ z = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_digits_l1770_177033


namespace NUMINAMATH_GPT_annie_has_12_brownies_left_l1770_177091

noncomputable def initial_brownies := 100
noncomputable def portion_for_admin := (3 / 5 : ℚ) * initial_brownies
noncomputable def leftover_after_admin := initial_brownies - portion_for_admin
noncomputable def portion_for_carl := (1 / 4 : ℚ) * leftover_after_admin
noncomputable def leftover_after_carl := leftover_after_admin - portion_for_carl
noncomputable def portion_for_simon := 3
noncomputable def leftover_after_simon := leftover_after_carl - portion_for_simon
noncomputable def portion_for_friends := (2 / 3 : ℚ) * leftover_after_simon
noncomputable def each_friend_get := portion_for_friends / 5
noncomputable def total_given_to_friends := each_friend_get * 5
noncomputable def final_brownies := leftover_after_simon - total_given_to_friends

theorem annie_has_12_brownies_left : final_brownies = 12 := by
  sorry

end NUMINAMATH_GPT_annie_has_12_brownies_left_l1770_177091


namespace NUMINAMATH_GPT_two_digit_number_is_54_l1770_177046

theorem two_digit_number_is_54 
    (n : ℕ) 
    (h1 : 10 ≤ n ∧ n < 100) 
    (h2 : n % 2 = 0) 
    (h3 : ∃ (a b : ℕ), a * b = 20 ∧ 10 * a + b = n) : 
    n = 54 := 
by
  sorry

end NUMINAMATH_GPT_two_digit_number_is_54_l1770_177046


namespace NUMINAMATH_GPT_nat_pow_eq_iff_divides_l1770_177096

theorem nat_pow_eq_iff_divides (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : a = b^n :=
sorry

end NUMINAMATH_GPT_nat_pow_eq_iff_divides_l1770_177096


namespace NUMINAMATH_GPT_number_of_x_values_l1770_177093

theorem number_of_x_values : 
  (∃ x_values : Finset ℕ, (∀ x ∈ x_values, 10 ≤ x ∧ x < 25) ∧ x_values.card = 15) :=
by
  sorry

end NUMINAMATH_GPT_number_of_x_values_l1770_177093


namespace NUMINAMATH_GPT_total_seats_l1770_177074

-- Define the conditions
variable {S : ℝ} -- Total number of seats in the hall
variable {vacantSeats : ℝ} (h_vacant : vacantSeats = 240) -- Number of vacant seats
variable {filledPercentage : ℝ} (h_filled : filledPercentage = 0.60) -- Percentage of seats filled

-- Total seats in the hall
theorem total_seats (h : 0.40 * S = 240) : S = 600 :=
sorry

end NUMINAMATH_GPT_total_seats_l1770_177074


namespace NUMINAMATH_GPT_expression_evaluation_l1770_177086

theorem expression_evaluation :
  (1007 * (((7/4 : ℚ) / (3/4) + (3 / (9/4)) + (1/3)) /
    ((1 + 2 + 3 + 4 + 5) * 5 - 22)) / 19) = (4 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1770_177086


namespace NUMINAMATH_GPT_expression_simplification_l1770_177098

theorem expression_simplification (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 2 * x + y / 2 ≠ 0) :
  (2 * x + y / 2)⁻¹ * ((2 * x)⁻¹ + (y / 2)⁻¹) = (x * y)⁻¹ := 
sorry

end NUMINAMATH_GPT_expression_simplification_l1770_177098


namespace NUMINAMATH_GPT_school_student_count_l1770_177084

theorem school_student_count (pencils erasers pencils_per_student erasers_per_student students : ℕ) 
    (h1 : pencils = 195) 
    (h2 : erasers = 65) 
    (h3 : pencils_per_student = 3)
    (h4 : erasers_per_student = 1) :
    students = pencils / pencils_per_student ∧ students = erasers / erasers_per_student → students = 65 :=
by
  sorry

end NUMINAMATH_GPT_school_student_count_l1770_177084


namespace NUMINAMATH_GPT_polygon_with_given_angle_sums_is_hexagon_l1770_177048

theorem polygon_with_given_angle_sums_is_hexagon
  (n : ℕ)
  (h_interior : (n - 2) * 180 = 2 * 360) :
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_polygon_with_given_angle_sums_is_hexagon_l1770_177048


namespace NUMINAMATH_GPT_incorrect_height_is_151_l1770_177017

def incorrect_height (average_initial correct_height average_corrected : ℝ) : ℝ :=
  (30 * average_initial) - (30 * average_corrected) + correct_height

theorem incorrect_height_is_151 :
  incorrect_height 175 136 174.5 = 151 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_height_is_151_l1770_177017


namespace NUMINAMATH_GPT_units_digit_L_L_15_l1770_177027

def Lucas (n : ℕ) : ℕ :=
match n with
| 0 => 2
| 1 => 1
| n + 2 => Lucas n + Lucas (n + 1)

theorem units_digit_L_L_15 : (Lucas (Lucas 15)) % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_units_digit_L_L_15_l1770_177027


namespace NUMINAMATH_GPT_triangle_inequality_l1770_177082

variable {α β γ a b c: ℝ}

theorem triangle_inequality (h1 : α + β + γ = π)
  (h2 : α > 0) (h3 : β > 0) (h4 : γ > 0)
  (h5 : a > 0) (h6 : b > 0) (h7 : c > 0)
  (h8 : (α > β ∧ a > b) ∨ (α = β ∧ a = b) ∨ (α < β ∧ a < b))
  (h9 : (β > γ ∧ b > c) ∨ (β = γ ∧ b = c) ∨ (β < γ ∧ b < c))
  (h10 : (γ > α ∧ c > a) ∨ (γ = α ∧ c = a) ∨ (γ < α ∧ c < a)) :
  (π / 3) ≤ (a * α + b * β + c * γ) / (a + b + c) ∧
  (a * α + b * β + c * γ) / (a + b + c) < (π / 2) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1770_177082


namespace NUMINAMATH_GPT_exists_b_gt_a_divides_l1770_177010

theorem exists_b_gt_a_divides (a : ℕ) (h : 0 < a) :
  ∃ b : ℕ, b > a ∧ (1 + 2^a + 3^a) ∣ (1 + 2^b + 3^b) :=
sorry

end NUMINAMATH_GPT_exists_b_gt_a_divides_l1770_177010
