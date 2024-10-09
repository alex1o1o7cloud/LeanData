import Mathlib

namespace train_speed_length_l2385_238544

theorem train_speed_length (t1 t2 s : ℕ) (p : ℕ)
  (h1 : t1 = 7) 
  (h2 : t2 = 25) 
  (h3 : p = 378)
  (h4 : t2 - t1 = 18)
  (h5 : p / (t2 - t1) = 21) 
  (h6 : (p / (t2 - t1)) * t1 = 147) :
  (21, 147) = (21, 147) :=
by {
  sorry
}

end train_speed_length_l2385_238544


namespace vec_addition_l2385_238524

namespace VectorCalculation

open Real

def v1 : ℤ × ℤ := (3, -8)
def v2 : ℤ × ℤ := (2, -6)
def scalar : ℤ := 5

def scaled_v2 : ℤ × ℤ := (scalar * v2.1, scalar * v2.2)
def result : ℤ × ℤ := (v1.1 + scaled_v2.1, v1.2 + scaled_v2.2)

theorem vec_addition : result = (13, -38) := by
  sorry

end VectorCalculation

end vec_addition_l2385_238524


namespace percentage_of_temporary_workers_l2385_238599

theorem percentage_of_temporary_workers (total_workers technicians non_technicians permanent_technicians permanent_non_technicians : ℕ) 
  (h1 : total_workers = 100)
  (h2 : technicians = total_workers / 2) 
  (h3 : non_technicians = total_workers / 2) 
  (h4 : permanent_technicians = technicians / 2) 
  (h5 : permanent_non_technicians = non_technicians / 2) :
  ((total_workers - (permanent_technicians + permanent_non_technicians)) / total_workers) * 100 = 50 :=
by
  sorry

end percentage_of_temporary_workers_l2385_238599


namespace ratio_of_place_values_l2385_238580

def thousands_place_value : ℝ := 1000
def tenths_place_value : ℝ := 0.1

theorem ratio_of_place_values : thousands_place_value / tenths_place_value = 10000 := by
  sorry

end ratio_of_place_values_l2385_238580


namespace investment_duration_l2385_238560

noncomputable def log (x : ℝ) := Real.log x

theorem investment_duration 
  (P A : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (t : ℝ) 
  (hP : P = 3000) 
  (hA : A = 3630) 
  (hr : r = 0.10) 
  (hn : n = 1) 
  (ht : A = P * (1 + r / n) ^ (n * t)) :
  t = 2 :=
by
  sorry

end investment_duration_l2385_238560


namespace even_x_satisfies_remainder_l2385_238551

theorem even_x_satisfies_remainder 
  (z : ℕ) 
  (hz : z % 4 = 0) : 
  ∃ (x : ℕ), x % 2 = 0 ∧ (z * (2 + x + z) + 3) % 2 = 1 := 
by
  sorry

end even_x_satisfies_remainder_l2385_238551


namespace train_length_is_250_l2385_238540

noncomputable def train_length (V₁ V₂ V₃ : ℕ) (T₁ T₂ T₃ : ℕ) : ℕ :=
  let S₁ := (V₁ * (5/18) * T₁)
  let S₂ := (V₂ * (5/18)* T₂)
  let S₃ := (V₃ * (5/18) * T₃)
  if S₁ = S₂ ∧ S₂ = S₃ then S₁ else 0

theorem train_length_is_250 :
  train_length 50 60 70 18 20 22 = 250 := by
  -- proof omitted
  sorry

end train_length_is_250_l2385_238540


namespace perfect_square_conditions_l2385_238500

theorem perfect_square_conditions (k : ℕ) : 
  (∃ m : ℤ, k^2 - 101 * k = m^2) ↔ (k = 101 ∨ k = 2601) := 
by 
  sorry

end perfect_square_conditions_l2385_238500


namespace marbles_total_l2385_238574

-- Conditions
variables (T : ℕ) -- Total number of marbles
variables (h_red : T ≥ 12) -- At least 12 red marbles
variables (h_blue : T ≥ 8) -- At least 8 blue marbles
variables (h_prob : (T - 12 : ℚ) / T = (3 / 4 : ℚ)) -- Probability condition

-- Proof statement
theorem marbles_total : T = 48 :=
by
  -- Proof here
  sorry

end marbles_total_l2385_238574


namespace tulip_count_l2385_238512

theorem tulip_count (total_flowers : ℕ) (daisies : ℕ) (roses_ratio : ℚ)
  (tulip_count : ℕ) :
  total_flowers = 102 →
  daisies = 6 →
  roses_ratio = 5 / 6 →
  tulip_count = (total_flowers - daisies) * (1 - roses_ratio) →
  tulip_count = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end tulip_count_l2385_238512


namespace gcd_7_fact_10_fact_div_4_fact_eq_5040_l2385_238584

def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

noncomputable def quotient_fact (a b : ℕ) : ℕ := fact a / fact b

theorem gcd_7_fact_10_fact_div_4_fact_eq_5040 :
  Nat.gcd (fact 7) (quotient_fact 10 4) = 5040 := by
sorry

end gcd_7_fact_10_fact_div_4_fact_eq_5040_l2385_238584


namespace simplify_fraction_l2385_238534

theorem simplify_fraction : (5^3 + 5^5) / (5^4 - 5^2) = 65 / 12 := 
by 
  sorry

end simplify_fraction_l2385_238534


namespace ratio_S3_S9_l2385_238529

noncomputable def Sn (a r : ℝ) (n : ℕ) : ℝ := (a * (1 - r ^ n)) / (1 - r)

theorem ratio_S3_S9 (a r : ℝ) (h1 : r ≠ 1) (h2 : Sn a r 6 = 3 * Sn a r 3) :
  Sn a r 3 / Sn a r 9 = 1 / 7 :=
by
  sorry

end ratio_S3_S9_l2385_238529


namespace simplify_expression_l2385_238533

theorem simplify_expression (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^3 + 2) / x * (y^3 + 2) / y) - ((x^3 - 2) / y * (y^3 - 2) / x) = 4 * (x^2 / y + y^2 / x) :=
by sorry

end simplify_expression_l2385_238533


namespace spears_per_sapling_l2385_238566

/-- Given that a log can produce 9 spears and 6 saplings plus a log produce 27 spears,
prove that a single sapling can produce 3 spears (S = 3). -/
theorem spears_per_sapling (L S : ℕ) (hL : L = 9) (h: 6 * S + L = 27) : S = 3 :=
by
  sorry

end spears_per_sapling_l2385_238566


namespace louisa_average_speed_l2385_238508

def average_speed (v : ℝ) : Prop :=
  (350 / v) - (200 / v) = 3

theorem louisa_average_speed :
  ∃ v : ℝ, average_speed v ∧ v = 50 := 
by
  use 50
  unfold average_speed
  sorry

end louisa_average_speed_l2385_238508


namespace solve_fractional_equation_l2385_238509

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 4) : (1 / (x - 1) = 2 / (1 - x) + 1) → x = 4 :=
by
  sorry

end solve_fractional_equation_l2385_238509


namespace log_base_30_of_8_l2385_238525

theorem log_base_30_of_8 (a b : ℝ) (h1 : Real.log 5 = a) (h2 : Real.log 3 = b) : 
  Real.logb 30 8 = (3 * (1 - a)) / (1 + b) :=
by
  sorry

end log_base_30_of_8_l2385_238525


namespace exists_cubic_polynomial_with_cubed_roots_l2385_238591

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

-- Statement that we need to prove
theorem exists_cubic_polynomial_with_cubed_roots :
  ∃ (b c d : ℝ), ∀ (x : ℝ),
  (f x = 0) → (x^3 = y → x^3^3 + b * x^3^2 + c * x^3 + d = 0) :=
sorry

end exists_cubic_polynomial_with_cubed_roots_l2385_238591


namespace rectangle_area_is_30_l2385_238528

def Point := (ℤ × ℤ)

def vertices : List Point := [(-5, 1), (1, 1), (1, -4), (-5, -4)]

theorem rectangle_area_is_30 :
  let length := (vertices[1].1 - vertices[0].1).natAbs
  let width := (vertices[0].2 - vertices[2].2).natAbs
  length * width = 30 := by
  sorry

end rectangle_area_is_30_l2385_238528


namespace weeding_planting_support_l2385_238582

-- Definitions based on conditions
def initial_weeding := 31
def initial_planting := 18
def additional_support := 20

-- Let x be the number of people sent to support weeding.
variable (x : ℕ)

-- The equation to prove.
theorem weeding_planting_support :
  initial_weeding + x = 2 * (initial_planting + (additional_support - x)) :=
sorry

end weeding_planting_support_l2385_238582


namespace connie_s_problem_l2385_238549

theorem connie_s_problem (y : ℕ) (h : 3 * y = 90) : y / 3 = 10 :=
by
  sorry

end connie_s_problem_l2385_238549


namespace term_addition_k_to_kplus1_l2385_238588

theorem term_addition_k_to_kplus1 (k : ℕ) : 
  (2 * k + 2) + (2 * k + 3) = 4 * k + 5 := 
sorry

end term_addition_k_to_kplus1_l2385_238588


namespace correct_calculation_l2385_238516

theorem correct_calculation :
  (∀ a : ℝ, (a^2)^3 = a^6) ∧
  ¬(∀ a : ℝ, a * a^3 = a^3) ∧
  ¬(∀ a : ℝ, a + 2 * a^2 = 3 * a^3) ∧
  ¬(∀ (a b : ℝ), (-2 * a^2 * b)^2 = -4 * a^4 * b^2) :=
by
  sorry

end correct_calculation_l2385_238516


namespace rectangle_area_l2385_238546

-- Conditions: 
-- 1. The length of the rectangle is three times its width.
-- 2. The diagonal length of the rectangle is x.

theorem rectangle_area (x : ℝ) (w l : ℝ) (h1 : w * 3 = l) (h2 : w^2 + l^2 = x^2) :
  l * w = (3 / 10) * x^2 :=
by
  sorry

end rectangle_area_l2385_238546


namespace find_a4_l2385_238590

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) * a (n - 1) = a n * a n

def given_sequence_conditions (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 2 + a 6 = 34 ∧ a 3 * a 5 = 64

-- Statement
theorem find_a4 (a : ℕ → ℝ) (h : given_sequence_conditions a) : a 4 = 8 :=
sorry

end find_a4_l2385_238590


namespace cos_equation_solution_l2385_238568

open Real

theorem cos_equation_solution (m : ℝ) :
  (∀ x : ℝ, 4 * cos x - cos x^2 + m - 3 = 0) ↔ (0 ≤ m ∧ m ≤ 8) := by
  sorry

end cos_equation_solution_l2385_238568


namespace sufficient_but_not_necessary_condition_l2385_238502

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|)
  ∧ (∃ y : ℝ, ¬ (y ≥ 1) ∧ |y + 1| + |y - 1| = 2 * |y|) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2385_238502


namespace min_value_x1x2_squared_inequality_ab_l2385_238510

def D : Set (ℝ × ℝ) := 
  { p | ∃ x1 x2, p = (x1, x2) ∧ x1 + x2 = 2 ∧ x1 > 0 ∧ x2 > 0 }

-- Part 1: Proving the minimum value of x1^2 + x2^2 in set D is 2
theorem min_value_x1x2_squared (x1 x2 : ℝ) (h : (x1, x2) ∈ D) : 
  x1^2 + x2^2 ≥ 2 := 
sorry

-- Part 2: Proving the inequality for any (a, b) in set D
theorem inequality_ab (a b : ℝ) (h : (a, b) ∈ D) : 
  (1 / (a + 2 * b) + 1 / (2 * a + b)) ≥ (2 / 3) := 
sorry

end min_value_x1x2_squared_inequality_ab_l2385_238510


namespace seventeen_divides_9x_plus_5y_l2385_238521

theorem seventeen_divides_9x_plus_5y (x y : ℤ) (h : 17 ∣ (2 * x + 3 * y)) : 17 ∣ (9 * x + 5 * y) :=
sorry

end seventeen_divides_9x_plus_5y_l2385_238521


namespace greatest_possible_value_of_y_l2385_238556

-- Definitions according to problem conditions
variables {x y : ℤ}

-- The theorem statement to prove
theorem greatest_possible_value_of_y (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
sorry

end greatest_possible_value_of_y_l2385_238556


namespace problem_statement_l2385_238571

-- Defining the propositions p and q as Boolean variables
variables (p q : Prop)

-- Assume the given conditions
theorem problem_statement (hnp : ¬¬p) (hnpq : ¬(p ∧ q)) : p ∧ ¬q :=
by {
  -- Derived steps to satisfy the conditions are implicit within this scope
  sorry
}

end problem_statement_l2385_238571


namespace sum_of_consecutive_numbers_LCM_168_l2385_238552

theorem sum_of_consecutive_numbers_LCM_168
  (x y z : ℕ)
  (h1 : y = x + 1)
  (h2 : z = y + 1)
  (h3 : Nat.lcm (Nat.lcm x y) z = 168) :
  x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_numbers_LCM_168_l2385_238552


namespace area_of_shaded_region_l2385_238506

theorem area_of_shaded_region 
    (large_side : ℝ) (small_side : ℝ)
    (h_large : large_side = 10) 
    (h_small : small_side = 4) : 
    (large_side^2 - small_side^2) / 4 = 21 :=
by
  -- All proof steps are to be completed and checked,
  -- and sorry is used as placeholder for the final proof.
  sorry

end area_of_shaded_region_l2385_238506


namespace can_be_divided_into_6_triangles_l2385_238543

-- Define the initial rectangle dimensions
def initial_rectangle_length := 6
def initial_rectangle_width := 5

-- Define the cut-out rectangle dimensions
def cutout_rectangle_length := 2
def cutout_rectangle_width := 1

-- Total area before the cut-out
def total_area : Nat := initial_rectangle_length * initial_rectangle_width

-- Cut-out area
def cutout_area : Nat := cutout_rectangle_length * cutout_rectangle_width

-- Remaining area after the cut-out
def remaining_area : Nat := total_area - cutout_area

-- The statement to be proved
theorem can_be_divided_into_6_triangles :
  remaining_area = 28 → (∃ (triangles : List (Nat × Nat × Nat)), triangles.length = 6) :=
by 
  intros h
  sorry

end can_be_divided_into_6_triangles_l2385_238543


namespace find_value_of_expression_l2385_238527

theorem find_value_of_expression
  (a b c m : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = m)
  (h5 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2 * a)^2 + b * (m - 2 * b)^2 + c * (m - 2 * c)^2) / (a * b * c) = 12 := 
sorry

end find_value_of_expression_l2385_238527


namespace lisa_earns_more_than_tommy_l2385_238537

theorem lisa_earns_more_than_tommy {total_earnings : ℤ} (h1 : total_earnings = 60) :
  let lisa_earnings := total_earnings / 2
  let tommy_earnings := lisa_earnings / 2
  lisa_earnings - tommy_earnings = 15 :=
by
  sorry

end lisa_earns_more_than_tommy_l2385_238537


namespace Mark_has_23_kangaroos_l2385_238548

theorem Mark_has_23_kangaroos :
  ∃ K G : ℕ, G = 3 * K ∧ 2 * K + 4 * G = 322 ∧ K = 23 :=
by
  sorry

end Mark_has_23_kangaroos_l2385_238548


namespace craig_total_commission_correct_l2385_238562

-- Define the commission structures
def refrigerator_commission (price : ℝ) : ℝ := 75 + 0.08 * price
def washing_machine_commission (price : ℝ) : ℝ := 50 + 0.10 * price
def oven_commission (price : ℝ) : ℝ := 60 + 0.12 * price

-- Define total sales
def total_refrigerator_sales : ℝ := 5280
def total_washing_machine_sales : ℝ := 2140
def total_oven_sales : ℝ := 4620

-- Define number of appliances sold
def number_of_refrigerators : ℝ := 3
def number_of_washing_machines : ℝ := 4
def number_of_ovens : ℝ := 5

-- Calculate total commissions for each appliance category
def total_refrigerator_commission : ℝ := number_of_refrigerators * refrigerator_commission total_refrigerator_sales
def total_washing_machine_commission : ℝ := number_of_washing_machines * washing_machine_commission total_washing_machine_sales
def total_oven_commission : ℝ := number_of_ovens * oven_commission total_oven_sales

-- Calculate total commission for the week
def total_commission : ℝ := total_refrigerator_commission + total_washing_machine_commission + total_oven_commission

-- Prove that the total commission is as expected
theorem craig_total_commission_correct : total_commission = 5620.20 := 
by
  sorry

end craig_total_commission_correct_l2385_238562


namespace quadratic_has_one_real_solution_l2385_238594

theorem quadratic_has_one_real_solution (m : ℝ) : (∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → m = 6 :=
by
  sorry

end quadratic_has_one_real_solution_l2385_238594


namespace relationship_of_a_b_c_l2385_238558

noncomputable def a : ℝ := Real.log 3 / Real.log 2  -- a = log2(1/3)
noncomputable def b : ℝ := Real.exp (1 / 3)  -- b = e^(1/3)
noncomputable def c : ℝ := 1 / 3  -- c = e^ln(1/3) = 1/3

theorem relationship_of_a_b_c : b > c ∧ c > a :=
by
  -- Proof would go here
  sorry

end relationship_of_a_b_c_l2385_238558


namespace gcd_factorial_l2385_238569

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l2385_238569


namespace lcm_12_18_l2385_238559

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l2385_238559


namespace kamala_overestimation_l2385_238561

theorem kamala_overestimation : 
  let p := 150
  let q := 50
  let k := 2
  let d := 3
  let p_approx := 160
  let q_approx := 45
  let k_approx := 1
  let d_approx := 4
  let true_value := (p / q) - k + d
  let approx_value := (p_approx / q_approx) - k_approx + d_approx
  approx_value > true_value := 
  by 
  -- Skipping the detailed proof steps.
  sorry

end kamala_overestimation_l2385_238561


namespace highest_score_batsman_l2385_238541

variable (H L : ℕ)

theorem highest_score_batsman :
  (60 * 46) = (58 * 44 + H + L) ∧ (H - L = 190) → H = 199 :=
by
  intros h
  sorry

end highest_score_batsman_l2385_238541


namespace jacoby_lottery_expense_l2385_238536

-- Definitions based on the conditions:
def jacoby_trip_fund_needed : ℕ := 5000
def jacoby_hourly_wage : ℕ := 20
def jacoby_work_hours : ℕ := 10
def cookies_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def num_sisters : ℕ := 2
def money_still_needed : ℕ := 3214

-- The statement to prove:
theorem jacoby_lottery_expense : 
  (jacoby_hourly_wage * jacoby_work_hours) + (cookies_price * cookies_sold) +
  lottery_winnings + (sister_gift * num_sisters) 
  - (jacoby_trip_fund_needed - money_still_needed) = 10 :=
by {
  sorry
}

end jacoby_lottery_expense_l2385_238536


namespace solve_for_a_l2385_238520

theorem solve_for_a {f : ℝ → ℝ} (h1 : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 2) (h2 : f a = 7) : a = 7 :=
sorry

end solve_for_a_l2385_238520


namespace find_divisor_l2385_238503

theorem find_divisor (d : ℕ) (h1 : 2319 % d = 0) (h2 : 2304 % d = 0) (h3 : (2319 - 2304) % d = 0) : d = 3 :=
  sorry

end find_divisor_l2385_238503


namespace work_done_time_l2385_238565

/-
  Question: How many days does it take for \(a\) to do the work alone?

  Conditions:
  - \(b\) can do the work in 20 days.
  - \(c\) can do the work in 55 days.
  - \(a\) is assisted by \(b\) and \(c\) on alternate days, and the work can be done in 8 days.
  
  Correct Answer:
  - \(x = 8.8\)
-/

theorem work_done_time (x : ℝ) (h : 8 * x⁻¹ + 1 /  5 + 4 / 55 = 1): x = 8.8 :=
by sorry

end work_done_time_l2385_238565


namespace linear_function_no_second_quadrant_l2385_238511

theorem linear_function_no_second_quadrant (x y : ℝ) (h : y = 2 * x - 3) :
  ¬ ((x < 0) ∧ (y > 0)) :=
by {
  sorry
}

end linear_function_no_second_quadrant_l2385_238511


namespace remainder_86592_8_remainder_8741_13_l2385_238538

theorem remainder_86592_8 :
  86592 % 8 = 0 :=
by
  sorry

theorem remainder_8741_13 :
  8741 % 13 = 5 :=
by
  sorry

end remainder_86592_8_remainder_8741_13_l2385_238538


namespace hypotenuse_length_l2385_238583

theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : a^2 + b^2 = c^2) : c = 13 :=
by
  -- proof
  sorry

end hypotenuse_length_l2385_238583


namespace probability_of_sum_14_l2385_238581

-- Define the set of faces on a tetrahedral die
def faces : Set ℕ := {2, 4, 6, 8}

-- Define the event where the sum of two rolls equals 14
def event_sum_14 (a b : ℕ) : Prop := a + b = 14 ∧ a ∈ faces ∧ b ∈ faces

-- Define the total number of outcomes when rolling two dice
def total_outcomes : ℕ := 16

-- Define the number of successful outcomes for the event where the sum is 14
def successful_outcomes : ℕ := 2

-- The probability of rolling a sum of 14 with two such tetrahedral dice
def probability_sum_14 : ℚ := successful_outcomes / total_outcomes

-- The theorem we want to prove
theorem probability_of_sum_14 : probability_sum_14 = 1 / 8 := 
by sorry

end probability_of_sum_14_l2385_238581


namespace triangle_shape_l2385_238542

theorem triangle_shape (a b : ℝ) (A B : ℝ)
  (h1 : a ≠ 0) (h2 : b ≠ 0) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π)
  (hTriangle : A + B + (π - A - B) = π)
  (h : a * Real.cos A = b * Real.cos B) : 
  (A = B ∨ A + B = π / 2) := sorry

end triangle_shape_l2385_238542


namespace num_squares_in_6_by_6_grid_l2385_238570

def squares_in_grid (m n : ℕ) : ℕ :=
  (m - 1) * (m - 1) + (m - 2) * (m - 2) + 
  (m - 3) * (m - 3) + (m - 4) * (m - 4) + 
  (m - 5) * (m - 5)

theorem num_squares_in_6_by_6_grid : squares_in_grid 6 6 = 55 := 
by 
  sorry

end num_squares_in_6_by_6_grid_l2385_238570


namespace work_completes_in_39_days_l2385_238587

theorem work_completes_in_39_days 
  (amit_days : ℕ := 15)  -- Amit can complete work in 15 days
  (ananthu_days : ℕ := 45)  -- Ananthu can complete work in 45 days
  (amit_worked_days : ℕ := 3)  -- Amit worked for 3 days
  : (amit_worked_days + ((4 / 5) / (1 / ananthu_days))) = 39 :=
by
  sorry

end work_completes_in_39_days_l2385_238587


namespace mirasol_initial_amount_l2385_238573

/-- 
Mirasol had some money in her account. She spent $10 on coffee beans and $30 on a tumbler. She has $10 left in her account.
Prove that the initial amount of money Mirasol had in her account is $50.
-/
theorem mirasol_initial_amount (spent_coffee : ℕ) (spent_tumbler : ℕ) (left_in_account : ℕ) :
  spent_coffee = 10 → spent_tumbler = 30 → left_in_account = 10 → 
  spent_coffee + spent_tumbler + left_in_account = 50 := 
by
  sorry

end mirasol_initial_amount_l2385_238573


namespace greatest_value_NNM_l2385_238518

theorem greatest_value_NNM :
  ∃ (M : ℕ), (M * M % 10 = M) ∧ (∃ (MM : ℕ), MM = 11 * M ∧ (MM * M = 396)) :=
by
  sorry

end greatest_value_NNM_l2385_238518


namespace geometric_sequence_first_term_l2385_238563

-- Define factorial values for convenience
def fact (n : ℕ) : ℕ := Nat.factorial n
#eval fact 6 -- This should give us 720
#eval fact 7 -- This should give us 5040

-- State the hypotheses and the goal
theorem geometric_sequence_first_term (a r : ℝ)
  (h1 : a * r^2 = 720)
  (h2 : a * r^5 = 5040) :
  a = 720 / (7^(2/3 : ℝ)) :=
by
  sorry

end geometric_sequence_first_term_l2385_238563


namespace total_snakes_seen_l2385_238504

-- Define the number of snakes in each breeding ball
def snakes_in_first_breeding_ball : Nat := 15
def snakes_in_second_breeding_ball : Nat := 20
def snakes_in_third_breeding_ball : Nat := 25
def snakes_in_fourth_breeding_ball : Nat := 30
def snakes_in_fifth_breeding_ball : Nat := 35
def snakes_in_sixth_breeding_ball : Nat := 40
def snakes_in_seventh_breeding_ball : Nat := 45

-- Define the number of pairs of extra snakes
def extra_pairs_of_snakes : Nat := 23

-- Define the total number of snakes observed
def total_snakes_observed : Nat :=
  snakes_in_first_breeding_ball +
  snakes_in_second_breeding_ball +
  snakes_in_third_breeding_ball +
  snakes_in_fourth_breeding_ball +
  snakes_in_fifth_breeding_ball +
  snakes_in_sixth_breeding_ball +
  snakes_in_seventh_breeding_ball +
  (extra_pairs_of_snakes * 2)

theorem total_snakes_seen : total_snakes_observed = 256 := by
  sorry

end total_snakes_seen_l2385_238504


namespace complement_of_A_in_U_l2385_238598

noncomputable def U : Set ℤ := {-3, -1, 0, 1, 3}

noncomputable def A : Set ℤ := {x | x^2 - 2 * x - 3 = 0}

theorem complement_of_A_in_U : (U \ A) = {-3, 0, 1} :=
by sorry

end complement_of_A_in_U_l2385_238598


namespace units_digit_6_power_l2385_238545

theorem units_digit_6_power (n : ℕ) : (6^n % 10) = 6 :=
sorry

end units_digit_6_power_l2385_238545


namespace pizza_slices_left_per_person_l2385_238597

def total_slices (small: Nat) (large: Nat) : Nat := small + large

def total_eaten (phil: Nat) (andre: Nat) : Nat := phil + andre

def slices_left (total: Nat) (eaten: Nat) : Nat := total - eaten

def pieces_per_person (left: Nat) (people: Nat) : Nat := left / people

theorem pizza_slices_left_per_person :
  ∀ (small large phil andre people: Nat),
  small = 8 → large = 14 → phil = 9 → andre = 9 → people = 2 →
  pieces_per_person (slices_left (total_slices small large) (total_eaten phil andre)) people = 2 :=
by
  intros small large phil andre people h_small h_large h_phil h_andre h_people
  rw [h_small, h_large, h_phil, h_andre, h_people]
  /-
  Here we conclude the proof.
  -/
  sorry

end pizza_slices_left_per_person_l2385_238597


namespace part1_part2_l2385_238553

noncomputable def f (a : ℝ) (a_pos : a > 1) (x : ℝ) : ℝ :=
  a^x + (x - 2) / (x + 1)

-- Statement for part 1
theorem part1 (a : ℝ) (a_pos : a > 1) : ∀ x : ℝ, -1 < x → f a a_pos x ≤ f a a_pos (x + ε) → 0 < ε := sorry

-- Statement for part 2
theorem part2 (a : ℝ) (a_pos : a > 1) : ¬ ∃ x : ℝ, x < 0 ∧ f a a_pos x = 0 := sorry

end part1_part2_l2385_238553


namespace coastal_village_population_l2385_238576

variable (N : ℕ) (k : ℕ) (parts_for_males : ℕ) (total_males : ℕ)

theorem coastal_village_population 
  (h_total_population : N = 540)
  (h_division : k = 4)
  (h_parts_for_males : parts_for_males = 2)
  (h_total_males : total_males = (N / k) * parts_for_males) :
  total_males = 270 := 
by
  sorry

end coastal_village_population_l2385_238576


namespace jim_gave_away_675_cards_l2385_238579

def total_cards_gave_away
  (cards_per_set : ℕ)
  (sets_to_brother sets_to_sister sets_to_friend : ℕ)
  : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend) * cards_per_set

theorem jim_gave_away_675_cards
  (cards_per_set : ℕ)
  (sets_to_brother sets_to_sister sets_to_friend : ℕ)
  (h_brother : sets_to_brother = 15)
  (h_sister : sets_to_sister = 8)
  (h_friend : sets_to_friend = 4)
  (h_cards_per_set : cards_per_set = 25)
  : total_cards_gave_away cards_per_set sets_to_brother sets_to_sister sets_to_friend = 675 :=
by
  sorry

end jim_gave_away_675_cards_l2385_238579


namespace triangle_inequality_proof_l2385_238596

theorem triangle_inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
    a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
    sorry

end triangle_inequality_proof_l2385_238596


namespace total_selling_price_l2385_238575

theorem total_selling_price (cost_price_per_metre profit_per_metre : ℝ)
  (total_metres_sold : ℕ) :
  cost_price_per_metre = 58.02564102564102 → 
  profit_per_metre = 29 → 
  total_metres_sold = 78 →
  (cost_price_per_metre + profit_per_metre) * total_metres_sold = 6788 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  -- backend calculation, checking computation level;
  sorry

end total_selling_price_l2385_238575


namespace remainder_example_l2385_238505

def P (x : ℝ) := 8 * x^3 - 20 * x^2 + 28 * x - 26
def D (x : ℝ) := 4 * x - 8

theorem remainder_example : P 2 = 14 :=
by
  sorry

end remainder_example_l2385_238505


namespace no_solutions_for_divisibility_by_3_l2385_238532

theorem no_solutions_for_divisibility_by_3 (x y : ℤ) : ¬ (x^2 + y^2 + x + y ∣ 3) :=
sorry

end no_solutions_for_divisibility_by_3_l2385_238532


namespace sum_of_areas_of_squares_l2385_238519

theorem sum_of_areas_of_squares (a b x : ℕ) 
  (h_overlapping_min : 9 ≤ (min a b) ^ 2)
  (h_overlapping_max : (min a b) ^ 2 ≤ 25)
  (h_sum_of_sides : a + b + x = 23) :
  a^2 + b^2 + x^2 = 189 := 
sorry

end sum_of_areas_of_squares_l2385_238519


namespace combined_forgotten_angles_l2385_238530

-- Define primary conditions
def initial_angle_sum : ℝ := 2873
def correct_angle_sum : ℝ := 16 * 180

-- The theorem to prove
theorem combined_forgotten_angles : correct_angle_sum - initial_angle_sum = 7 :=
by sorry

end combined_forgotten_angles_l2385_238530


namespace water_height_in_cylinder_l2385_238507

theorem water_height_in_cylinder :
  let r_cone := 10 -- Radius of the cone in cm
  let h_cone := 15 -- Height of the cone in cm
  let r_cylinder := 20 -- Radius of the cylinder in cm
  let volume_cone := (1 / 3) * Real.pi * r_cone^2 * h_cone
  volume_cone = 500 * Real.pi -> 
  let h_cylinder := volume_cone / (Real.pi * r_cylinder^2)
  h_cylinder = 1.25 := 
by
  intros r_cone h_cone r_cylinder volume_cone h_volume
  let h_cylinder := volume_cone / (Real.pi * r_cylinder^2)
  have : h_cylinder = 1.25 := by
    sorry
  exact this

end water_height_in_cylinder_l2385_238507


namespace sequence_perfect_square_l2385_238586

theorem sequence_perfect_square (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) :
  ∃! n, ∃ k, a n = k ^ 2 :=
by
  sorry

end sequence_perfect_square_l2385_238586


namespace mrs_sheridan_initial_cats_l2385_238514

def cats_initial (cats_given_away : ℕ) (cats_left : ℕ) : ℕ :=
  cats_given_away + cats_left

theorem mrs_sheridan_initial_cats : cats_initial 14 3 = 17 :=
by
  sorry

end mrs_sheridan_initial_cats_l2385_238514


namespace sum_non_solutions_eq_neg21_l2385_238554

theorem sum_non_solutions_eq_neg21
  (A B C : ℝ)
  (h1 : ∀ x, ∃ k : ℝ, (x ≠ -C) ∧ (x ≠ -9) → (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9))
  (h2 : ∃ A B C, ∀ x, (x ≠ -C) ∧ (x ≠ -9) → (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9))
  (h3 : ∃! x, (x + C) * (x + 9) = 0)
   :
  -9 + -12 = -21 := by sorry

end sum_non_solutions_eq_neg21_l2385_238554


namespace coloring_ways_l2385_238515

-- Define the vertices and edges of the graph
def vertices : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8}

def edges : Finset (ℕ × ℕ) :=
  { (0, 1), (1, 2), (2, 0),  -- First triangle
    (3, 4), (4, 5), (5, 3),  -- Middle triangle
    (6, 7), (7, 8), (8, 6),  -- Third triangle
    (2, 5),   -- Connecting top horizontal edge
    (1, 7) }  -- Connecting bottom horizontal edge

-- Define the number of colors available
def colors := 4

-- Define a function to count the valid colorings given the vertices and edges
noncomputable def countValidColorings (vertices : Finset ℕ) (edges : Finset (ℕ × ℕ)) (colors : ℕ) : ℕ := sorry

-- The theorem statement
theorem coloring_ways : countValidColorings vertices edges colors = 3456 := 
sorry

end coloring_ways_l2385_238515


namespace meal_cost_l2385_238577

theorem meal_cost (x : ℝ) (h1 : ∀ (x : ℝ), (x / 4) - 6 = x / 9) : 
  x = 43.2 :=
by
  have h : (∀ (x : ℝ), (x / 4) - (x / 9) = 6) := sorry
  exact sorry

end meal_cost_l2385_238577


namespace cone_height_l2385_238595

noncomputable def height_of_cone (r : ℝ) (n : ℕ) : ℝ :=
  let sector_circumference := (2 * Real.pi * r) / n
  let cone_base_radius := sector_circumference / (2 * Real.pi)
  Real.sqrt (r^2 - cone_base_radius^2)

theorem cone_height
  (r_original : ℝ)
  (n : ℕ)
  (h : r_original = 10)
  (hc : n = 4) :
  height_of_cone r_original n = 5 * Real.sqrt 3 := by
  sorry

end cone_height_l2385_238595


namespace minimum_value_a_plus_b_plus_c_l2385_238517

theorem minimum_value_a_plus_b_plus_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 4 * b + 7 * c ≤ 2 * a * b * c) : a + b + c ≥ 15 / 2 :=
by
  sorry

end minimum_value_a_plus_b_plus_c_l2385_238517


namespace necessary_not_sufficient_condition_l2385_238539

theorem necessary_not_sufficient_condition (m : ℝ) 
  (h : 2 < m ∧ m < 6) :
  (∃ (x y : ℝ), (x^2 / (m - 2) + y^2 / (6 - m) = 1)) ∧ (∀ m', 2 < m' ∧ m' < 6 → ∃ (x' y' : ℝ), (x'^2 / (m' - 2) + y'^2 / (6 - m') = 1) ∧ m' ≠ 4) :=
by
  sorry

end necessary_not_sufficient_condition_l2385_238539


namespace right_triangle_hypotenuse_l2385_238572

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l2385_238572


namespace smallest_n_condition_l2385_238564

open Nat

-- Define the sum of squares formula
noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- Define the condition for being a square number
def is_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- The proof problem statement
theorem smallest_n_condition : 
  ∃ n : ℕ, n > 1 ∧ is_square (sum_of_squares n / n) ∧ (∀ m : ℕ, m > 1 ∧ is_square (sum_of_squares m / m) → n ≤ m) :=
sorry

end smallest_n_condition_l2385_238564


namespace bag_ratio_l2385_238555

noncomputable def ratio_of_costs : ℚ := 1 / 2

theorem bag_ratio :
  ∃ (shirt_cost shoes_cost total_cost bag_cost : ℚ),
    shirt_cost = 7 ∧
    shoes_cost = shirt_cost + 3 ∧
    total_cost = 2 * shirt_cost + shoes_cost ∧
    bag_cost = 36 - total_cost ∧
    bag_cost / total_cost = ratio_of_costs :=
sorry

end bag_ratio_l2385_238555


namespace arithmetic_and_geometric_mean_l2385_238567

theorem arithmetic_and_geometric_mean (x y : ℝ) (h₁ : (x + y) / 2 = 20) (h₂ : Real.sqrt (x * y) = Real.sqrt 150) : x^2 + y^2 = 1300 :=
by
  sorry

end arithmetic_and_geometric_mean_l2385_238567


namespace tv_weight_difference_l2385_238522

-- Definitions for the given conditions
def bill_tv_length : ℕ := 48
def bill_tv_width : ℕ := 100
def bob_tv_length : ℕ := 70
def bob_tv_width : ℕ := 60
def weight_per_square_inch : ℕ := 4
def ounces_per_pound : ℕ := 16

-- The statement to prove
theorem tv_weight_difference : (bill_tv_length * bill_tv_width * weight_per_square_inch)
                               - (bob_tv_length * bob_tv_width * weight_per_square_inch)
                               = 150 * ounces_per_pound := by
  sorry

end tv_weight_difference_l2385_238522


namespace abs_neg_2023_eq_2023_neg_one_pow_2023_eq_neg_one_l2385_238593

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 :=
sorry

theorem neg_one_pow_2023_eq_neg_one : (-1 : ℤ) ^ 2023 = -1 :=
sorry

end abs_neg_2023_eq_2023_neg_one_pow_2023_eq_neg_one_l2385_238593


namespace min_length_MN_l2385_238585

theorem min_length_MN (a b : ℝ) (H h : ℝ) (MN : ℝ) (midsegment_eq_4 : (a + b) / 2 = 4)
    (area_div_eq_half : (a + MN) / 2 * h = (MN + b) / 2 * H) : MN = 4 :=
by
  sorry

end min_length_MN_l2385_238585


namespace player_A_wins_l2385_238547

theorem player_A_wins (n : ℕ) : ∃ m, (m > 2 * n^2) ∧ (∀ S : Finset (ℕ × ℕ), S.card = m → ∃ (r c : Finset ℕ), r.card = n ∧ c.card = n ∧ ∀ rc ∈ r.product c, rc ∈ S → false) :=
by sorry

end player_A_wins_l2385_238547


namespace trajectory_of_B_l2385_238513

-- Define the points and the line for the given conditions
def A : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (2, -3)
def D_line (x : ℝ) (y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the statement to be proved
theorem trajectory_of_B (x y : ℝ) :
  D_line x y → ∃ Bx By, (3 * Bx - By - 20 = 0) :=
sorry

end trajectory_of_B_l2385_238513


namespace average_percentage_25_students_l2385_238523

theorem average_percentage_25_students (s1 s2 : ℕ) (p1 p2 : ℕ) (n : ℕ)
  (h1 : s1 = 15) (h2 : p1 = 75) (h3 : s2 = 10) (h4 : p2 = 95) (h5 : n = 25) :
  ((s1 * p1 + s2 * p2) / n) = 83 := 
by
  sorry

end average_percentage_25_students_l2385_238523


namespace simplify_expr_l2385_238589

variable (a b : ℤ)  -- assuming a and b are elements of the ring ℤ

theorem simplify_expr : 105 * a - 38 * a + 27 * b - 12 * b = 67 * a + 15 * b := 
by
  sorry

end simplify_expr_l2385_238589


namespace quadratic_eq_roots_quadratic_eq_positive_integer_roots_l2385_238501

theorem quadratic_eq_roots (m : ℝ) (hm : m ≠ 0 ∧ m ≤ 9 / 8) :
  ∃ x1 x2 : ℝ, (m * x1^2 + 3 * x1 + 2 = 0) ∧ (m * x2^2 + 3 * x2 + 2 = 0) :=
sorry

theorem quadratic_eq_positive_integer_roots (m : ℕ) (hm : m = 1) :
  ∃ x1 x2 : ℝ, (x1 = -1) ∧ (x2 = -2) ∧ (m * x1^2 + 3 * x1 + 2 = 0) ∧ (m * x2^2 + 3 * x2 + 2 = 0) :=
sorry

end quadratic_eq_roots_quadratic_eq_positive_integer_roots_l2385_238501


namespace yellow_balls_count_l2385_238526

theorem yellow_balls_count (purple blue total_needed : ℕ) 
  (h_purple : purple = 7) 
  (h_blue : blue = 5) 
  (h_total : total_needed = 19) : 
  ∃ (yellow : ℕ), yellow = 6 :=
by
  sorry

end yellow_balls_count_l2385_238526


namespace intersection_M_N_l2385_238578

def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | (x ∈ U) ∧ ¬(x ∈ complement_U_N)}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
sorry

end intersection_M_N_l2385_238578


namespace range_of_b_l2385_238535

noncomputable def a_n (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem range_of_b (b : ℝ) : (∀ n : ℕ, 0 < n → a_n (n+1) b > a_n n b) ↔ (-3 < b) :=
by
    sorry

end range_of_b_l2385_238535


namespace total_profit_l2385_238531

theorem total_profit (a_cap b_cap : ℝ) (a_profit : ℝ) (a_share b_share : ℝ) (P : ℝ) :
  a_cap = 15000 ∧ b_cap = 25000 ∧ a_share = 0.10 ∧ a_profit = 4200 →
  a_share * P + (a_cap / (a_cap + b_cap)) * (1 - a_share) * P = a_profit →
  P = 9600 :=
by
  intros h1 h2
  have h3 : a_share * P + (a_cap / (a_cap + b_cap)) * (1 - a_share) * P = a_profit := h2
  sorry

end total_profit_l2385_238531


namespace journey_length_l2385_238550

theorem journey_length (speed time : ℝ) (portions_covered total_portions : ℕ)
  (h_speed : speed = 40) (h_time : time = 0.7) (h_portions_covered : portions_covered = 4) (h_total_portions : total_portions = 5) :
  (speed * time / portions_covered) * total_portions = 35 :=
by
  sorry

end journey_length_l2385_238550


namespace geometric_sequence_a6_l2385_238557

variable {a : ℕ → ℝ} (h_geo : ∀ n, a (n+1) / a n = a (n+2) / a (n+1))

theorem geometric_sequence_a6 (h5 : a 5 = 2) (h7 : a 7 = 8) : a 6 = 4 ∨ a 6 = -4 :=
by
  sorry

end geometric_sequence_a6_l2385_238557


namespace count_8_digit_odd_last_l2385_238592

-- Define the constraints for the digits of the 8-digit number
def first_digit_choices := 9
def next_six_digits_choices := 10 ^ 6
def last_digit_choices := 5

-- State the theorem based on the given conditions and the solution
theorem count_8_digit_odd_last : first_digit_choices * next_six_digits_choices * last_digit_choices = 45000000 :=
by
  sorry

end count_8_digit_odd_last_l2385_238592
