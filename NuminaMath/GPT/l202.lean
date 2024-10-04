import Mathlib

namespace total_airflow_correct_l202_202126

def airflow_fan_A : ℕ := 10 * 10 * 60 * 7
def airflow_fan_B : ℕ := 15 * 20 * 60 * 5
def airflow_fan_C : ℕ := 25 * 30 * 60 * 5
def airflow_fan_D : ℕ := 20 * 15 * 60 * 2
def airflow_fan_E : ℕ := 30 * 60 * 60 * 6

def total_airflow : ℕ :=
  airflow_fan_A + airflow_fan_B + airflow_fan_C + airflow_fan_D + airflow_fan_E

theorem total_airflow_correct : total_airflow = 1041000 := by
  sorry

end total_airflow_correct_l202_202126


namespace lesser_of_two_numbers_l202_202709

theorem lesser_of_two_numbers (a b : ℕ) (h₁ : a + b = 55) (h₂ : a - b = 7) (h₃ : a > b) : b = 24 :=
by
  sorry

end lesser_of_two_numbers_l202_202709


namespace initial_deposit_l202_202409

variable (P R : ℝ)

theorem initial_deposit (h1 : P + (P * R * 3) / 100 = 11200)
                       (h2 : P + (P * (R + 2) * 3) / 100 = 11680) :
  P = 8000 :=
by
  sorry

end initial_deposit_l202_202409


namespace solution_set_of_inequalities_l202_202978

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202978


namespace parallel_lines_m_value_l202_202484

theorem parallel_lines_m_value (x y m : ℝ) (h₁ : 2 * x + m * y - 7 = 0) (h₂ : m * x + 8 * y - 14 = 0) (parallel : (2 / m = m / 8)) : m = -4 := 
sorry

end parallel_lines_m_value_l202_202484


namespace expected_value_fair_dodecahedral_die_l202_202583

theorem expected_value_fair_dodecahedral_die :
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  expected_value = 6.5 :=
by
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  have h1 : list.sum outcomes = 78 := by sorry
  have h2 : (12: ℝ) * (1 / 12: ℝ) = 1 := by sorry
  show expected_value = 6.5
  { sorry }

end expected_value_fair_dodecahedral_die_l202_202583


namespace fly_distance_from_ceiling_l202_202252

theorem fly_distance_from_ceiling (x y z : ℝ) (hx : x = 2) (hy : y = 6) (hP : x^2 + y^2 + z^2 = 100) : z = 2 * Real.sqrt 15 :=
by
  sorry

end fly_distance_from_ceiling_l202_202252


namespace exam_prob_l202_202506

noncomputable def prob_pass (n : ℕ) (p : ℚ) : ℚ :=
  ∑ k in finset.range (n + 1), (if k < 3 then (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) else 0)

theorem exam_prob :
  prob_pass 6 (1/3) = 1 - (64/729 + 64/243 + 40/243) := sorry

end exam_prob_l202_202506


namespace solution_set_of_linear_inequalities_l202_202909

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202909


namespace find_points_l202_202465

def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem find_points (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (y = x ∨ y = -x) := by
  sorry

end find_points_l202_202465


namespace factory_output_exceeds_by_20_percent_l202_202736

theorem factory_output_exceeds_by_20_percent 
  (planned_output : ℝ) (actual_output : ℝ)
  (h_planned : planned_output = 20)
  (h_actual : actual_output = 24) :
  ((actual_output - planned_output) / planned_output) * 100 = 20 := 
by
  sorry

end factory_output_exceeds_by_20_percent_l202_202736


namespace linear_inequalities_solution_l202_202868

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202868


namespace solution_set_of_inequalities_l202_202985

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202985


namespace isosceles_triangle_perimeter_l202_202185

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4) (h2 : b = 8) (h3 : ∃ p q r, p = b ∧ q = b ∧ r = a ∧ p + q > r) : 
  a + b + b = 20 := 
by 
  sorry

end isosceles_triangle_perimeter_l202_202185


namespace solution_set_system_of_inequalities_l202_202930

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202930


namespace smallest_GCD_value_l202_202565

theorem smallest_GCD_value (a b c d N : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : N > 5)
    (hc1 : Nat.gcd a b = 1 ∨ Nat.gcd a c = 1 ∨ Nat.gcd a d = 1 ∨ Nat.gcd b c = 1 ∨ Nat.gcd b d = 1 ∨ Nat.gcd c d = 1)
    (hc2 : Nat.gcd a b = 2 ∨ Nat.gcd a c = 2 ∨ Nat.gcd a d = 2 ∨ Nat.gcd b c = 2 ∨ Nat.gcd b d = 2 ∨ Nat.gcd c d = 2)
    (hc3 : Nat.gcd a b = 3 ∨ Nat.gcd a c = 3 ∨ Nat.gcd a d = 3 ∨ Nat.gcd b c = 3 ∨ Nat.gcd b d = 3 ∨ Nat.gcd c d = 3)
    (hc4 : Nat.gcd a b = 4 ∨ Nat.gcd a c = 4 ∨ Nat.gcd a d = 4 ∨ Nat.gcd b c = 4 ∨ Nat.gcd b d = 4 ∨ Nat.gcd c d = 4)
    (hc5 : Nat.gcd a b = 5 ∨ Nat.gcd a c = 5 ∨ Nat.gcd a d = 5 ∨ Nat.gcd b c = 5 ∨ Nat.gcd b d = 5 ∨ Nat.gcd c d = 5)
    (hcN : Nat.gcd a b = N ∨ Nat.gcd a c = N ∨ Nat.gcd a d = N ∨ Nat.gcd b c = N ∨ Nat.gcd b d = N ∨ Nat.gcd c d = N):
    N = 14 :=
sorry

end smallest_GCD_value_l202_202565


namespace order_of_numbers_l202_202663

theorem order_of_numbers (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : m + n < 0) : 
  -m > n ∧ n > -n ∧ -n > m := 
by
  sorry

end order_of_numbers_l202_202663


namespace least_k_divisible_480_l202_202798

theorem least_k_divisible_480 (k : ℕ) (h : k^4 % 480 = 0) : k = 101250 :=
sorry

end least_k_divisible_480_l202_202798


namespace find_multiple_of_number_l202_202414

theorem find_multiple_of_number (n : ℝ) (m : ℝ) (h1 : n ≠ 0) (h2 : n = 9) (h3 : (n + n^2) / 2 = m * n) : m = 5 :=
sorry

end find_multiple_of_number_l202_202414


namespace solution_set_system_of_inequalities_l202_202923

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202923


namespace n_minus_k_minus_l_square_number_l202_202579

variable (n k l x : ℕ)

theorem n_minus_k_minus_l_square_number (h1 : x^2 < n)
                                        (h2 : n < (x + 1)^2)
                                        (h3 : n - k = x^2)
                                        (h4 : n + l = (x + 1)^2) :
  ∃ m : ℕ, n - k - l = m ^ 2 :=
by
  sorry

end n_minus_k_minus_l_square_number_l202_202579


namespace real_ratio_sum_values_l202_202638

variables (a b c d : ℝ)

theorem real_ratio_sum_values :
  (a / b + b / c + c / d + d / a = 6) ∧
  (a / c + b / d + c / a + d / b = 8) →
  (a / b + c / d = 2 ∨ a / b + c / d = 4) :=
by
  sorry

end real_ratio_sum_values_l202_202638


namespace value_of_g_at_3_l202_202029

def g (x : ℝ) := x^2 - 2*x + 1

theorem value_of_g_at_3 : g 3 = 4 :=
by
  sorry

end value_of_g_at_3_l202_202029


namespace prob_2_lt_X_lt_4_l202_202489

noncomputable def normalDist := distribution_normal 2 (σ^2)

theorem prob_2_lt_X_lt_4 (σ : ℝ) (hσ : σ > 0) (hX : prob (λ x, x > 0) normalDist = 0.9) :
  prob (λ x, 2 < x ∧ x < 4) normalDist = 0.4 :=
by sorry

end prob_2_lt_X_lt_4_l202_202489


namespace total_loads_l202_202657

def shirts_per_load := 3
def sweaters_per_load := 2
def socks_per_load := 4

def white_shirts := 9
def colored_shirts := 12
def white_sweaters := 18
def colored_sweaters := 20
def white_socks := 16
def colored_socks := 24

def white_shirt_loads : ℕ := white_shirts / shirts_per_load
def white_sweater_loads : ℕ := white_sweaters / sweaters_per_load
def white_sock_loads : ℕ := white_socks / socks_per_load

def colored_shirt_loads : ℕ := colored_shirts / shirts_per_load
def colored_sweater_loads : ℕ := colored_sweaters / sweaters_per_load
def colored_sock_loads : ℕ := colored_socks / socks_per_load

def max_white_loads := max (max white_shirt_loads white_sweater_loads) white_sock_loads
def max_colored_loads := max (max colored_shirt_loads colored_sweater_loads) colored_sock_loads

theorem total_loads : max_white_loads + max_colored_loads = 19 := by
  sorry

end total_loads_l202_202657


namespace solve_inequalities_l202_202959

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202959


namespace paco_cookies_l202_202227

theorem paco_cookies (initial_cookies: ℕ) (eaten_cookies: ℕ) (final_cookies: ℕ) (bought_cookies: ℕ) 
  (h1 : initial_cookies = 40)
  (h2 : eaten_cookies = 2)
  (h3 : final_cookies = 75)
  (h4 : initial_cookies - eaten_cookies + bought_cookies = final_cookies) :
  bought_cookies = 37 :=
by
  rw [h1, h2, h3] at h4
  sorry

end paco_cookies_l202_202227


namespace line_intersects_y_axis_at_0_6_l202_202155

theorem line_intersects_y_axis_at_0_6 : ∃ y : ℝ, 4 * y + 3 * (0 : ℝ) = 24 ∧ (0, y) = (0, 6) :=
by
  use 6
  simp
  sorry

end line_intersects_y_axis_at_0_6_l202_202155


namespace house_to_car_ratio_l202_202463

-- Define conditions
def cost_per_night := 4000
def nights_at_hotel := 2
def cost_of_car := 30000
def total_value_of_treats := 158000

-- Prove that the ratio of the value of the house to the value of the car is 4:1
theorem house_to_car_ratio : 
  (total_value_of_treats - (nights_at_hotel * cost_per_night + cost_of_car)) / cost_of_car = 4 := by
  sorry

end house_to_car_ratio_l202_202463


namespace determinant_of_A_l202_202460

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![ -5, 8],
    ![ 3, -4]]

theorem determinant_of_A : A.det = -4 := by
  sorry

end determinant_of_A_l202_202460


namespace planes_parallel_if_any_line_parallel_l202_202284

-- Definitions for Lean statements:
variable (P1 P2 : Set Point)
variable (line : Set Point)

-- Conditions
def is_parallel_to_plane (line : Set Point) (plane : Set Point) : Prop := sorry

def is_parallel_plane (plane1 plane2 : Set Point) : Prop := sorry

-- Lean statement to be proved:
theorem planes_parallel_if_any_line_parallel (h : ∀ line, 
  line ⊆ P1 → is_parallel_to_plane line P2) : is_parallel_plane P1 P2 := sorry

end planes_parallel_if_any_line_parallel_l202_202284


namespace cosine_of_negative_135_l202_202755

theorem cosine_of_negative_135 : Real.cos (-(135 * Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end cosine_of_negative_135_l202_202755


namespace min_value_abs_function_l202_202477

theorem min_value_abs_function : ∃ x : ℝ, ∀ x, (|x - 4| + |x - 6|) ≥ 2 :=
by
  sorry

end min_value_abs_function_l202_202477


namespace minimum_value_expression_l202_202661

theorem minimum_value_expression (a : ℝ) (h : a > 0) : 
  a + (a + 4) / a ≥ 5 :=
sorry

end minimum_value_expression_l202_202661


namespace p_necessary_not_sufficient_q_l202_202181

def condition_p (x : ℝ) : Prop := abs x ≤ 2
def condition_q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem p_necessary_not_sufficient_q (x : ℝ) :
  (condition_p x → condition_q x) = false ∧ (condition_q x → condition_p x) = true :=
by
  sorry

end p_necessary_not_sufficient_q_l202_202181


namespace greatest_consecutive_integers_sum_120_l202_202431

def sum_of_consecutive_integers (n : ℤ) (a : ℤ) : ℤ :=
  n * (2 * a + n - 1) / 2

theorem greatest_consecutive_integers_sum_120 (N : ℤ) (a : ℤ) (h1 : sum_of_consecutive_integers N a = 120) : N ≤ 240 :=
by {
  -- Here we would provide the proof, but it's omitted with 'sorry'.
  sorry
}

end greatest_consecutive_integers_sum_120_l202_202431


namespace derivative_at_one_l202_202845

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * Real.log x

theorem derivative_at_one : (deriv f 1) = 5 := 
by 
  sorry

end derivative_at_one_l202_202845


namespace inequality_solution_l202_202004

theorem inequality_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : (x * (x + 1)) / ((x - 3)^2) ≥ 8) : 3 < x ∧ x ≤ 24/7 :=
sorry

end inequality_solution_l202_202004


namespace problem_statement_l202_202333

theorem problem_statement (x y : ℝ) (h : x / y = 2) : (x - y) / x = 1 / 2 :=
by
  sorry

end problem_statement_l202_202333


namespace min_total_books_l202_202247

-- Definitions based on conditions
variables (P C B : ℕ)

-- Condition 1: Ratio of physics to chemistry books is 3:2
def ratio_physics_chemistry := 3 * C = 2 * P

-- Condition 2: Ratio of chemistry to biology books is 4:3
def ratio_chemistry_biology := 4 * B = 3 * C

-- Condition 3: Total number of books is 3003
def total_books := P + C + B = 3003

-- The theorem to prove
theorem min_total_books (h1 : ratio_physics_chemistry P C) (h2 : ratio_chemistry_biology C B) (h3: total_books P C B) :
  3003 = 3003 :=
by
  sorry

end min_total_books_l202_202247


namespace find_n_l202_202558

-- Define the initial number and the transformation applied to it
def initial_number : ℕ := 12320

def appended_threes (n : ℕ) : ℕ :=
  initial_number * 10^(10*n + 1) + (3 * (10^(10*n + 1) - 1) / 9 : ℕ)

def quaternary_to_decimal (n : ℕ) : ℕ :=
  let base4_number := appended_threes n
  -- The conversion process, in base-4 representation
  let converted_number := 1 * (4^4) + 2 * (4^3) + 3 * (4^2) + 2 * (4^1) + 1 * (4^0)
  converted_number * (4^(10*n + 1))

-- Define x as the converted number minus 1
def x (n : ℕ) : ℕ :=
  quaternary_to_decimal n - 1

-- Define the proof statement in Lean
theorem find_n (n : ℕ) : 
  (∀ n : ℕ, (n = 0) → (x n).prime_factors.length = 2) :=
by
  sorry


end find_n_l202_202558


namespace calc_result_l202_202850

theorem calc_result : (-3)^2 - (-2)^3 = 17 := 
by
  sorry

end calc_result_l202_202850


namespace sum_of_reciprocals_of_geometric_sequence_is_two_l202_202325

theorem sum_of_reciprocals_of_geometric_sequence_is_two
  (a1 q : ℝ)
  (pos_terms : 0 < a1)
  (S P M : ℝ)
  (sum_eq : S = 9)
  (product_eq : P = 81 / 4)
  (sum_of_terms : S = a1 * (1 - q^4) / (1 - q))
  (product_of_terms : P = a1 * a1 * q * q * (a1*q*q) * (q*a1) )
  (sum_of_reciprocals : M = (q^4 - 1) / (a1 * (q^4 - q^3)))
  : M = 2 :=
sorry

end sum_of_reciprocals_of_geometric_sequence_is_two_l202_202325


namespace polynomial_divisible_by_7_l202_202635

theorem polynomial_divisible_by_7 (n : ℤ) : 7 ∣ ((n + 7)^2 - n^2) :=
sorry

end polynomial_divisible_by_7_l202_202635


namespace some_number_is_ten_l202_202502

theorem some_number_is_ten (x : ℕ) (h : 5 ^ 29 * 4 ^ 15 = 2 * x ^ 29) : x = 10 :=
by
  sorry

end some_number_is_ten_l202_202502


namespace mul_72519_9999_eq_725117481_l202_202478

theorem mul_72519_9999_eq_725117481 : 72519 * 9999 = 725117481 := by
  sorry

end mul_72519_9999_eq_725117481_l202_202478


namespace john_walks_further_than_nina_l202_202052

theorem john_walks_further_than_nina :
  let john_distance := 0.7
  let nina_distance := 0.4
  john_distance - nina_distance = 0.3 :=
by
  sorry

end john_walks_further_than_nina_l202_202052


namespace smallest_positive_z_l202_202701

theorem smallest_positive_z :
  (∃ x z : ℝ, cos x = 0 ∧ cos (x + z) = 1/2 ∧ ∀ w : ℝ, (cos x = 0 ∧ cos (x + w) = 1/2) → (0 < w → z ≤ w)) → z = π / 6 :=
by
  intro h
  sorry

end smallest_positive_z_l202_202701


namespace number_of_9_step_paths_l202_202600

-- Definitions and conditions
def is_white_square (i j : ℕ) : Bool := (i + j) % 2 = 0

def valid_move (i j k l : ℕ) : Bool :=
  (k = i + 1) ∧ (l = j ∨ l = j + 1 ∨ l = j - 1) ∧ is_white_square k l

def count_paths (n : ℕ) : ℕ :=
  ∑ b in Finset.range (n / 2 + 1),
    Nat.choose n b * Nat.choose (n - b) b

theorem number_of_9_step_paths :
  count_paths 9 = 457 := by
  sorry

end number_of_9_step_paths_l202_202600


namespace change_is_correct_l202_202273

def regular_ticket_cost : ℕ := 109
def child_discount : ℕ := 5
def payment_given : ℕ := 500

-- Prices for different people in the family
def child_ticket_cost (age : ℕ) : ℕ :=
  if age < 12 then regular_ticket_cost - child_discount else regular_ticket_cost

def parent_ticket_cost : ℕ := regular_ticket_cost
def family_ticket_cost : ℕ :=
  (child_ticket_cost 6) + (child_ticket_cost 10) + parent_ticket_cost + parent_ticket_cost

def change_received : ℕ := payment_given - family_ticket_cost

-- Prove that the change received is 74
theorem change_is_correct : change_received = 74 :=
by sorry

end change_is_correct_l202_202273


namespace linear_inequalities_solution_l202_202858

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202858


namespace determine_x_l202_202242

noncomputable def is_equal_mean_median_mode (x : ℕ) : Prop :=
  let s := {3, 4, 5, 6, 6, 7, x}
  let median := 6
  let mode := 6
  let mean := (3 + 4 + 5 + 6 + 6 + 7 + x) / 7
  mode = 6 ∧ median = 6 ∧ mean = 6

theorem determine_x : is_equal_mean_median_mode 11 :=
  by
  sorry

end determine_x_l202_202242


namespace yanna_change_l202_202131

theorem yanna_change :
  let shirt_cost := 5 in
  let sandel_cost := 3 in
  let total_money := 100 in
  let num_shirts := 10 in
  let num_sandels := 3 in
  let total_cost := (num_shirts * shirt_cost) + (num_sandels * sandel_cost) in
  let change := total_money - total_cost in
  change = 41 :=
by
  sorry

end yanna_change_l202_202131


namespace best_fit_of_regression_model_l202_202510

-- Define the context of regression analysis and the coefficient of determination
def regression_analysis : Type := sorry
def coefficient_of_determination (r : regression_analysis) : ℝ := sorry

-- Definitions of each option for clarity in our context
def A (r : regression_analysis) : Prop := sorry -- the linear relationship is stronger
def B (r : regression_analysis) : Prop := sorry -- the linear relationship is weaker
def C (r : regression_analysis) : Prop := sorry -- better fit of the model
def D (r : regression_analysis) : Prop := sorry -- worse fit of the model

-- The formal statement we need to prove
theorem best_fit_of_regression_model (r : regression_analysis) (R2 : ℝ) (h1 : coefficient_of_determination r = R2) (h2 : R2 = 1) : C r :=
by
  sorry

end best_fit_of_regression_model_l202_202510


namespace minimum_value_fraction_l202_202219

variable (a b c : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : b + c ≥ a)

theorem minimum_value_fraction : (b / c + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end minimum_value_fraction_l202_202219


namespace find_c_l202_202316

-- Definitions for the conditions
def is_solution (x c : ℝ) : Prop := x^2 + c * x - 36 = 0

theorem find_c (c : ℝ) (h : is_solution (-9) c) : c = 5 :=
sorry

end find_c_l202_202316


namespace find_x_solution_l202_202702

noncomputable def find_x (x y : ℝ) (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : Prop := 
  x = (3 + Real.sqrt 17) / 2

theorem find_x_solution (x y : ℝ) 
(h1 : x - y^2 = 3) 
(h2 : x^2 + y^4 = 13) 
(hx_pos : 0 < x) 
(hy_pos : 0 < y) : 
  find_x x y h1 h2 :=
sorry

end find_x_solution_l202_202702


namespace unique_triplets_l202_202292

theorem unique_triplets (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + 
               |c * x + a * y + b * z| = |x| + |y| + |z|) ↔ 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = -1)) :=
sorry

end unique_triplets_l202_202292


namespace solution_set_of_inequalities_l202_202971

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202971


namespace distance_between_sasha_and_kolya_when_sasha_finished_l202_202393

-- Definitions based on the problem conditions
def distance_sasha : ℝ := 100
def distance_lesha_when_sasha_finished : ℝ := 90
def distance_kolya_when_lesha_finished : ℝ := 90

def velocity_lesha (v_s : ℝ) : ℝ := 0.9 * v_s
def velocity_kolya (v_s : ℝ) : ℝ := 0.81 * v_s

-- Theorem statement
theorem distance_between_sasha_and_kolya_when_sasha_finished (v_s : ℝ) :
  distance_sasha - (velocity_kolya v_s * (distance_sasha / v_s)) = 19 :=
  by sorry

end distance_between_sasha_and_kolya_when_sasha_finished_l202_202393


namespace roadsters_paving_company_total_cement_l202_202692

noncomputable def cement_lexi : ℝ := 10
noncomputable def cement_tess : ℝ := cement_lexi + 0.20 * cement_lexi
noncomputable def cement_ben : ℝ := cement_tess - 0.10 * cement_tess
noncomputable def cement_olivia : ℝ := 2 * cement_ben

theorem roadsters_paving_company_total_cement :
  cement_lexi + cement_tess + cement_ben + cement_olivia = 54.4 := by
  sorry

end roadsters_paving_company_total_cement_l202_202692


namespace sum_of_coordinates_of_B_is_7_l202_202066

-- Define points and conditions
def A := (0, 0)
def B (x : ℝ) := (x, 3)
def slope (p₁ p₂ : ℝ × ℝ) : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)

-- Main theorem to prove the sum of the coordinates of point B is 7
theorem sum_of_coordinates_of_B_is_7 (x : ℝ) (h_slope : slope A (B x) = 3 / 4) : x + 3 = 7 :=
by
  -- Proof goes here, we use sorry to skip the proof steps.
  sorry

end sum_of_coordinates_of_B_is_7_l202_202066


namespace distance_between_Sasha_and_Kolya_l202_202399

/-- Sasha, Lesha, and Kolya simultaneously started a 100-meter race.
Assuming all three run at constant but unequal speeds, when Sasha
finished (100 meters), Lesha was 10 meters behind him; and when Lesha
finished, Kolya was 10 meters behind him. Thus, the distance between
Sasha and Kolya when Sasha finished is 19 meters. -/
theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL tK : ℝ), 
  vS > 0 ∧ vL > 0 ∧ vK > 0 ∧ 
  tS = 100 / vS ∧ 
  tL = 100 / vL ∧ 
  tK = 100 / vK ∧ 
  vL * tS = 90 ∧ 
  vK * tL = 90 →
  vS * tS - vK * tS = 19 :=
begin
  sorry
end

end distance_between_Sasha_and_Kolya_l202_202399


namespace find_theta_l202_202079

def equilateral_triangle_angle : ℝ := 60
def square_angle : ℝ := 90
def pentagon_angle : ℝ := 108
def total_round_angle : ℝ := 360

theorem find_theta (θ : ℝ)
  (h_eq_tri : equilateral_triangle_angle = 60)
  (h_squ : square_angle = 90)
  (h_pen : pentagon_angle = 108)
  (h_round : total_round_angle = 360) :
  θ = total_round_angle - (equilateral_triangle_angle + square_angle + pentagon_angle) :=
sorry

end find_theta_l202_202079


namespace average_of_b_and_c_l202_202841

theorem average_of_b_and_c (a b c : ℝ) 
  (h₁ : (a + b) / 2 = 50) 
  (h₂ : c - a = 40) : 
  (b + c) / 2 = 70 := 
by
  sorry

end average_of_b_and_c_l202_202841


namespace continuous_func_unique_l202_202002

theorem continuous_func_unique (f : ℝ → ℝ) (hf_cont : Continuous f)
  (hf_eqn : ∀ x : ℝ, f x + f (x^2) = 2) :
  ∀ x : ℝ, f x = 1 :=
by
  sorry

end continuous_func_unique_l202_202002


namespace solution_set_of_inequalities_l202_202967

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202967


namespace solution_set_of_linear_inequalities_l202_202916

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202916


namespace line_passes_through_fixed_point_l202_202197

theorem line_passes_through_fixed_point (a b c : ℝ) (h : a - b + c = 0) : a * 1 + b * (-1) + c = 0 := 
by sorry

end line_passes_through_fixed_point_l202_202197


namespace non_neg_int_solutions_inequality_l202_202419

theorem non_neg_int_solutions_inequality :
  {x : ℕ | -2 * (x : ℤ) > -4} = {0, 1} :=
by
  sorry

end non_neg_int_solutions_inequality_l202_202419


namespace jessica_age_proof_l202_202050

-- Definitions based on conditions
def grandmother_age (j : ℚ) : ℚ := 15 * j
def age_difference (g j : ℚ) : Prop := g - j = 60

-- Proposed age of Jessica
def jessica_age : ℚ := 30 / 7

-- Main statement to prove
theorem jessica_age_proof : ∃ j : ℚ, grandmother_age j = 15 * j ∧ age_difference (grandmother_age j) j ∧ j = jessica_age :=
by sorry

end jessica_age_proof_l202_202050


namespace B_finishes_work_in_54_days_l202_202136

-- The problem statement rewritten in Lean 4.
theorem B_finishes_work_in_54_days
  (A_eff : ℕ) -- amount of work A can do in one day
  (B_eff : ℕ) -- amount of work B can do in one day
  (work_days_together : ℕ) -- number of days A and B work together to finish the work
  (h1 : A_eff = 2 * B_eff)
  (h2 : A_eff + B_eff = 3)
  (h3 : work_days_together = 18) :
  work_days_together * (A_eff + B_eff) / B_eff = 54 :=
by
  sorry

end B_finishes_work_in_54_days_l202_202136


namespace probability_is_correct_l202_202338

open Finset

def numbers : Finset ℕ := {1, 3, 6, 10, 15, 21, 40}

def is_multiple_of_30 (s : Finset ℕ) : Prop :=
  30 ∣ (s.prod id)

def all_combinations := powerset (numbers.card.choose 3)

def valid_trios := (all_combinations.filter is_multiple_of_30).card

def probability_of_multiple_of_30 : ℚ := 
  valid_trios.to_nat / all_combinations.card.to_nat

theorem probability_is_correct :
  probability_of_multiple_of_30 = 4 / 35 :=
sorry

end probability_is_correct_l202_202338


namespace cupcakes_left_l202_202816

def pack_count := 3
def cupcakes_per_pack := 4
def cupcakes_eaten := 5

theorem cupcakes_left : (pack_count * cupcakes_per_pack - cupcakes_eaten) = 7 := 
by 
  sorry

end cupcakes_left_l202_202816


namespace jasmine_additional_cans_needed_l202_202520

theorem jasmine_additional_cans_needed
  (n_initial : ℕ)
  (n_lost : ℕ)
  (n_remaining : ℕ)
  (additional_can_coverage : ℕ)
  (n_needed : ℕ) :
  n_initial = 50 →
  n_lost = 4 →
  n_remaining = 36 →
  additional_can_coverage = 2 →
  n_needed = 7 :=
by
  sorry

end jasmine_additional_cans_needed_l202_202520


namespace jimmy_sells_less_l202_202522

-- Definitions based on conditions
def num_figures : ℕ := 5
def value_figure_1_to_4 : ℕ := 15
def value_figure_5 : ℕ := 20
def total_earned : ℕ := 55

-- Formulation of the problem statement in Lean
theorem jimmy_sells_less (total_value : ℕ := (4 * value_figure_1_to_4) + value_figure_5) (difference : ℕ := total_value - total_earned) (amount_less_per_figure : ℕ := difference / num_figures) : amount_less_per_figure = 5 := by
  sorry

end jimmy_sells_less_l202_202522


namespace polygon_sides_l202_202101

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l202_202101


namespace stratified_sampling_females_l202_202738

theorem stratified_sampling_females :
  let males := 500
  let females := 400
  let total_students := 900
  let total_surveyed := 45
  let males_surveyed := 25
  ((males_surveyed : ℚ) / males) * females = 20 := by
  sorry

end stratified_sampling_females_l202_202738


namespace range_of_a_l202_202652

noncomputable def f (x a : ℝ) : ℝ := 
  x * (a - 1 / Real.exp x)

noncomputable def gx (x : ℝ) : ℝ :=
  (1 + x) / Real.exp x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 a = 0 ∧ f x2 a = 0) →
  a < 2 / Real.exp 1 :=
by
  sorry

end range_of_a_l202_202652


namespace well_depth_and_rope_length_l202_202367

variables (x y : ℝ)

theorem well_depth_and_rope_length :
  (y = x / 4 - 3) ∧ (y = x / 5 + 1) → y = 17 ∧ x = 80 :=
by
  sorry
 
end well_depth_and_rope_length_l202_202367


namespace solve_inequalities_l202_202989

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202989


namespace cos_of_theta_cos_double_of_theta_l202_202308

noncomputable def theta : ℝ := sorry -- Placeholder for theta within the interval (0, π/2)
axiom theta_in_range : 0 < theta ∧ theta < Real.pi / 2
axiom sin_theta_eq : Real.sin theta = 1/3

theorem cos_of_theta : Real.cos theta = 2 * Real.sqrt 2 / 3 := by
  sorry

theorem cos_double_of_theta : Real.cos (2 * theta) = 7 / 9 := by
  sorry

end cos_of_theta_cos_double_of_theta_l202_202308


namespace combined_rocket_height_l202_202352

variable (h1 : ℕ) (h2 : ℕ)

-- Given conditions
def first_rocket_height : ℕ := 500
def second_rocket_height : ℕ := first_rocket_height * 2

-- Prove that the combined height is 1500 ft
theorem combined_rocket_height : first_rocket_height + second_rocket_height = 1500 := by
  sorry

end combined_rocket_height_l202_202352


namespace range_of_a_l202_202793

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + a * x + 3 ≥ a) ↔ -7 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_a_l202_202793


namespace sides_of_polygon_l202_202116

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l202_202116


namespace linear_inequalities_solution_l202_202866

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202866


namespace calc_price_per_litre_l202_202213

noncomputable def pricePerLitre (initial final totalCost : ℝ) : ℝ :=
  totalCost / (final - initial)

theorem calc_price_per_litre :
  pricePerLitre 10 50 36.60 = 91.5 :=
by
  sorry

end calc_price_per_litre_l202_202213


namespace find_point_C_find_area_triangle_ABC_l202_202201

noncomputable section

-- Given points and equations
def point_B : ℝ × ℝ := (4, 4)
def eq_angle_bisector : ℝ × ℝ → Prop := λ p => p.2 = 0
def eq_altitude : ℝ × ℝ → Prop := λ p => p.1 - 2 * p.2 + 2 = 0

-- Target coordinates of point C
def point_C : ℝ × ℝ := (10, -8)

-- Coordinates of point A derived from given conditions
def point_A : ℝ × ℝ := (-2, 0)

-- Line equations derived from conditions
def eq_line_BC : ℝ × ℝ → Prop := λ p => 2 * p.1 + p.2 - 12 = 0
def eq_line_AC : ℝ × ℝ → Prop := λ p => 2 * p.1 + 3 * p.2 + 4 = 0

-- Prove the coordinates of point C
theorem find_point_C : ∃ C : ℝ × ℝ, eq_line_BC C ∧ eq_line_AC C ∧ C = point_C := by
  sorry

-- Prove the area of triangle ABC.
theorem find_area_triangle_ABC : ∃ S : ℝ, S = 48 := by
  sorry

end find_point_C_find_area_triangle_ABC_l202_202201


namespace min_value_f_l202_202474

noncomputable def f (x : ℝ) : ℝ := |x - 4| + |x - 6|

theorem min_value_f : ∃ x : ℝ, f x ≥ 2 :=
by
  sorry

end min_value_f_l202_202474


namespace linear_inequalities_solution_l202_202863

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202863


namespace pool_water_after_45_days_l202_202448

-- Defining the initial conditions and the problem statement in Lean
noncomputable def initial_amount : ℝ := 500
noncomputable def evaporation_rate : ℝ := 0.7
noncomputable def addition_rate : ℝ := 5
noncomputable def total_days : ℕ := 45

noncomputable def final_amount : ℝ :=
  initial_amount - (evaporation_rate * total_days) +
  (addition_rate * (total_days / 3))

theorem pool_water_after_45_days : final_amount = 543.5 :=
by
  -- Inserting the proof is not required here
  sorry

end pool_water_after_45_days_l202_202448


namespace probability_divisible_by_3_l202_202725

theorem probability_divisible_by_3 :
  ∀ (n : ℤ), (1 ≤ n) ∧ (n ≤ 99) → 3 ∣ (n * (n + 1)) :=
by
  intros n hn
  -- Detailed proof would follow here
  sorry

end probability_divisible_by_3_l202_202725


namespace inequality_always_holds_true_l202_202645

theorem inequality_always_holds_true (a b c : ℝ) (h₁ : a > b) (h₂ : b > c) :
  (a / (c^2 + 1)) > (b / (c^2 + 1)) :=
by
  sorry

end inequality_always_holds_true_l202_202645


namespace solution_set_of_inequalities_l202_202966

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202966


namespace remainder_when_divided_l202_202012

theorem remainder_when_divided (P D Q R D'' Q'' R'' : ℕ) (h1 : P = Q * D + R) (h2 : Q = D'' * Q'' + R'') :
  P % (2 * D * D'') = D * R'' + R := sorry

end remainder_when_divided_l202_202012


namespace polygon_sides_l202_202110

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l202_202110


namespace second_piece_weight_l202_202152

theorem second_piece_weight (w1 : ℝ) (s1 : ℝ) (s2 : ℝ) (w2 : ℝ) :
  (s1 = 4) → (w1 = 16) → (s2 = 6) → w2 = w1 * (s2^2 / s1^2) → w2 = 36 :=
by
  intro h_s1 h_w1 h_s2 h_w2
  rw [h_s1, h_w1, h_s2] at h_w2
  norm_num at h_w2
  exact h_w2

end second_piece_weight_l202_202152


namespace find_first_year_l202_202130

-- Define sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

-- Define the conditions
def after_2020 (n : ℕ) : Prop := n > 2020
def sum_of_digits_eq (n required_sum : ℕ) : Prop := sum_of_digits n = required_sum

noncomputable def first_year_after_2020_with_digit_sum_15 : ℕ :=
  2049

-- The statement to be proved
theorem find_first_year : 
  ∃ y : ℕ, after_2020 y ∧ sum_of_digits_eq y 15 ∧ y = first_year_after_2020_with_digit_sum_15 :=
by
  sorry

end find_first_year_l202_202130


namespace total_students_l202_202578

theorem total_students (f1 f2 f3 total : ℕ)
  (h_ratio : f1 * 2 = f2)
  (h_ratio2 : f1 * 3 = f3)
  (h_f1 : f1 = 6)
  (h_total : total = f1 + f2 + f3) :
  total = 48 :=
by
  sorry

end total_students_l202_202578


namespace solution_set_of_inequalities_l202_202961

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202961


namespace number_of_married_men_at_least_11_l202_202037

-- Definitions based only on conditions from a)
def total_men := 100
def men_with_tv := 75
def men_with_radio := 85
def men_with_ac := 70
def married_with_tv_radio_ac := 11

-- Theorem that needs to be proven based on the conditions
theorem number_of_married_men_at_least_11 : total_men ≥ married_with_tv_radio_ac :=
by
  sorry

end number_of_married_men_at_least_11_l202_202037


namespace probability_center_in_convex_hull_3_points_probability_center_in_convex_hull_4_points_l202_202836

noncomputable def P_n (n : ℕ) : ℚ :=
  if n = 3 then 1 / 4
  else if n = 4 then 3 / 4
  else 0

theorem probability_center_in_convex_hull_3_points :
  P_n 3 = 1 / 4 :=
by
  sorry

theorem probability_center_in_convex_hull_4_points :
  P_n 4 = 3 / 4 :=
by
  sorry

end probability_center_in_convex_hull_3_points_probability_center_in_convex_hull_4_points_l202_202836


namespace largest_expression_l202_202722

theorem largest_expression :
  let A := 0.9387
  let B := 0.9381
  let C := 9385 / 10000
  let D := 0.9379
  let E := 0.9389
  E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  let A := 0.9387
  let B := 0.9381
  let C := 9385 / 10000
  let D := 0.9379
  let E := 0.9389
  sorry

end largest_expression_l202_202722


namespace arcsin_arccos_solution_l202_202074

theorem arcsin_arccos_solution (x : ℝ) (hx1 : |x| ≤ 1) (hx2 : |2*x| ≤ 1) :
  arcsin x + arcsin (2*x) = arccos x ↔ x = 0 ∨ x = 2 / Real.sqrt 5 ∨ x = - (2 / Real.sqrt 5) := 
sorry

end arcsin_arccos_solution_l202_202074


namespace solve_inequalities_l202_202990

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202990


namespace length_of_best_day_l202_202009

theorem length_of_best_day
  (len_raise_the_roof : Nat)
  (len_rap_battle : Nat)
  (len_best_day : Nat)
  (total_ride_duration : Nat)
  (playlist_count : Nat)
  (total_songs_length : Nat)
  (h_len_raise_the_roof : len_raise_the_roof = 2)
  (h_len_rap_battle : len_rap_battle = 3)
  (h_total_ride_duration : total_ride_duration = 40)
  (h_playlist_count : playlist_count = 5)
  (h_total_songs_length : len_raise_the_roof + len_rap_battle + len_best_day = total_songs_length)
  (h_playlist_length : total_ride_duration / playlist_count = total_songs_length) :
  len_best_day = 3 := 
sorry

end length_of_best_day_l202_202009


namespace union_complement_correctness_l202_202215

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_complement_correctness : 
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {2, 4} →
  A ∪ (U \ B) = {1, 2, 3, 5} :=
by
  intro hU hA hB
  sorry

end union_complement_correctness_l202_202215


namespace solution_set_linear_inequalities_l202_202944

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202944


namespace total_pencils_l202_202835

/-- The conditions defining the number of pencils Sarah buys each day. -/
def pencils_monday : ℕ := 20
def pencils_tuesday : ℕ := 18
def pencils_wednesday : ℕ := 3 * pencils_tuesday

/-- The hypothesis that the total number of pencils bought by Sarah is 92. -/
theorem total_pencils : pencils_monday + pencils_tuesday + pencils_wednesday = 92 :=
by
  -- calculations skipped
  sorry

end total_pencils_l202_202835


namespace distance_between_Sasha_and_Kolya_l202_202400

/-- Sasha, Lesha, and Kolya simultaneously started a 100-meter race.
Assuming all three run at constant but unequal speeds, when Sasha
finished (100 meters), Lesha was 10 meters behind him; and when Lesha
finished, Kolya was 10 meters behind him. Thus, the distance between
Sasha and Kolya when Sasha finished is 19 meters. -/
theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL tK : ℝ), 
  vS > 0 ∧ vL > 0 ∧ vK > 0 ∧ 
  tS = 100 / vS ∧ 
  tL = 100 / vL ∧ 
  tK = 100 / vK ∧ 
  vL * tS = 90 ∧ 
  vK * tL = 90 →
  vS * tS - vK * tS = 19 :=
begin
  sorry
end

end distance_between_Sasha_and_Kolya_l202_202400


namespace range_of_m_l202_202780

open Set

def setM (m : ℝ) : Set ℝ := { x | x ≤ m }
def setP : Set ℝ := { x | x ≥ -1 }

theorem range_of_m (m : ℝ) (h : setM m ∩ setP = ∅) : m < -1 := sorry

end range_of_m_l202_202780


namespace factory_needs_to_produce_l202_202737

-- Define the given conditions
def weekly_production_target : ℕ := 6500
def production_mon_tue_wed : ℕ := 3 * 1200
def production_thu : ℕ := 800
def total_production_mon_thu := production_mon_tue_wed + production_thu
def required_production_fri := weekly_production_target - total_production_mon_thu

-- The theorem we need to prove
theorem factory_needs_to_produce : required_production_fri = 2100 :=
by
  -- The proof would go here
  sorry

end factory_needs_to_produce_l202_202737


namespace div_by_3kp1_iff_div_by_3k_l202_202690

theorem div_by_3kp1_iff_div_by_3k (m n k : ℕ) (h1 : m > n) :
  (3 ^ (k + 1)) ∣ (4 ^ m - 4 ^ n) ↔ (3 ^ k) ∣ (m - n) := 
sorry

end div_by_3kp1_iff_div_by_3k_l202_202690


namespace sqrt_product_l202_202290

theorem sqrt_product : (Real.sqrt 121) * (Real.sqrt 49) * (Real.sqrt 11) = 77 * (Real.sqrt 11) := by
  -- This is just the theorem statement as requested.
  sorry

end sqrt_product_l202_202290


namespace max_roses_l202_202070

theorem max_roses (budget : ℝ) (indiv_price : ℝ) (dozen_1_price : ℝ) (dozen_2_price : ℝ) (dozen_5_price : ℝ) (hundred_price : ℝ) 
  (budget_eq : budget = 1000) (indiv_price_eq : indiv_price = 5.30) (dozen_1_price_eq : dozen_1_price = 36) 
  (dozen_2_price_eq : dozen_2_price = 50) (dozen_5_price_eq : dozen_5_price = 110) (hundred_price_eq : hundred_price = 180) : 
  ∃ max_roses : ℕ, max_roses = 548 :=
by
  sorry

end max_roses_l202_202070


namespace factor_quadratic_l202_202172

theorem factor_quadratic (x : ℝ) (m n : ℝ) 
  (hm : m^2 = 16) (hn : n^2 = 25) (hmn : 2 * m * n = 40) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := 
by sorry

end factor_quadratic_l202_202172


namespace charge_for_each_additional_fifth_mile_l202_202272

theorem charge_for_each_additional_fifth_mile
  (initial_charge : ℝ)
  (total_charge : ℝ)
  (distance_in_miles : ℕ)
  (distance_per_increment : ℝ)
  (x : ℝ) :
  initial_charge = 2.10 →
  total_charge = 17.70 →
  distance_in_miles = 8 →
  distance_per_increment = 1/5 →
  (total_charge - initial_charge) / ((distance_in_miles / distance_per_increment) - 1) = x →
  x = 0.40 :=
by
  intros h_initial_charge h_total_charge h_distance_in_miles h_distance_per_increment h_eq
  sorry

end charge_for_each_additional_fifth_mile_l202_202272


namespace solution_set_inequalities_l202_202894

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202894


namespace total_amount_divided_l202_202598

variables (T x : ℝ)
variables (h₁ : 0.06 * x + 0.05 * (T - x) = 85) 
variables (h₂ : T - x = 1100)

theorem total_amount_divided (T x : ℝ) 
  (h₁ : 0.06 * x + 0.05 * (T - x) = 85) 
  (h₂ : T - x = 1100) : 
  T = 1600 := 
sorry

end total_amount_divided_l202_202598


namespace cone_prism_volume_ratio_l202_202605

/--
Given:
- The base of the prism is a rectangle with side lengths 2r and 3r.
- The height of the prism is h.
- The base of the cone is a circle with radius r and height h.

Prove:
- The ratio of the volume of the cone to the volume of the prism is (π / 18).
-/
theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * Real.pi * r^2 * h) / (6 * r^2 * h) = Real.pi / 18 := by
  sorry

end cone_prism_volume_ratio_l202_202605


namespace sin_double_angle_value_l202_202200

theorem sin_double_angle_value (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : (1/2) * Real.cos (2 * α) = Real.sin (π/4 + α)) :
  Real.sin (2 * α) = -1 :=
by
  sorry

end sin_double_angle_value_l202_202200


namespace solve_inequalities_l202_202996

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202996


namespace overall_gain_percentage_correct_l202_202249

structure Transaction :=
  (buy_prices : List ℕ)
  (sell_prices : List ℕ)

def overallGainPercentage (trans : Transaction) : ℚ :=
  let total_cost := (trans.buy_prices.foldl (· + ·) 0 : ℚ)
  let total_sell := (trans.sell_prices.foldl (· + ·) 0 : ℚ)
  (total_sell - total_cost) / total_cost * 100

theorem overall_gain_percentage_correct
  (trans : Transaction)
  (h_buy_prices : trans.buy_prices = [675, 850, 920])
  (h_sell_prices : trans.sell_prices = [1080, 1100, 1000]) :
  overallGainPercentage trans = 30.06 := by
  sorry

end overall_gain_percentage_correct_l202_202249


namespace simplify_expression_l202_202838

theorem simplify_expression (x y : ℝ) (h : (x + 2)^2 + abs (y - 1/2) = 0) :
  (x - 2*y)*(x + 2*y) - (x - 2*y)^2 = -6 :=
by
  -- Proof will be provided here
  sorry

end simplify_expression_l202_202838


namespace solution_set_system_of_inequalities_l202_202933

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202933


namespace cos_angle_B_bounds_l202_202345

theorem cos_angle_B_bounds {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ)
  (angle_ADC : ℝ) (angle_B : ℝ)
  (h1 : AB = 2) (h2 : BC = 3) (h3 : CD = 2) (h4 : angle_ADC = 180 - angle_B) :
  (1 / 4) < Real.cos angle_B ∧ Real.cos angle_B < (3 / 4) := 
sorry -- Proof to be provided

end cos_angle_B_bounds_l202_202345


namespace common_difference_l202_202043

def arith_seq_common_difference (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference {a : ℕ → ℤ} (h₁ : a 5 = 3) (h₂ : a 6 = -2) : arith_seq_common_difference a (-5) :=
by
  intros n
  cases n with
  | zero => sorry -- base case: a 1 = a 0 + (-5), requires additional initial condition
  | succ n' => sorry -- inductive step

end common_difference_l202_202043


namespace distance_between_sasha_and_kolya_is_19_meters_l202_202403

theorem distance_between_sasha_and_kolya_is_19_meters
  (v_S v_L v_K : ℝ)
  (h1 : v_L = 0.9 * v_S)
  (h2 : v_K = 0.81 * v_S)
  (h3 : ∀ t_S : ℝ, t_S = 100 / v_S) :
  (∀ t_S : ℝ, 100 - v_K * t_S = 19) :=
by
  intros t_S
  have vL_defined : v_L = 0.9 * v_S := h1
  have vK_defined : v_K = 0.81 * v_S := h2
  have time_S : t_S = 100 / v_S := h3 t_S
  sorry

end distance_between_sasha_and_kolya_is_19_meters_l202_202403


namespace polygon_sides_l202_202100

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l202_202100


namespace fairy_tale_island_counties_l202_202514

theorem fairy_tale_island_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year_no_elf_counties := initial_dwarf_counties * 3 + initial_centaur_counties * 3 in
  let total_after_first_year := initial_elf_counties + first_year_no_elf_counties in
  let second_year_no_dwarf_counties := initial_elf_counties * 4 + initial_centaur_counties * 4 in
  let total_after_second_year := second_year_no_dwarf_counties + initial_dwarf_counties * 3 in
  let third_year_no_centaur_counties := second_year_no_dwarf_counties * 6 + initial_dwarf_counties * 6 in
  let final_total_counties := third_year_no_centaur_counties + initial_centaur_counties * 12 in
  final_total_counties = 54 :=
begin
  /- Proof is omitted -/
  sorry
end

end fairy_tale_island_counties_l202_202514


namespace find_number_of_students_l202_202264

theorem find_number_of_students (N T : ℕ) 
  (avg_mark_all : T = 80 * N) 
  (avg_mark_exclude : (T - 150) / (N - 5) = 90) : 
  N = 30 := by
  sorry

end find_number_of_students_l202_202264


namespace equal_cost_number_of_minutes_l202_202266

theorem equal_cost_number_of_minutes :
  ∃ m : ℝ, (8 + 0.25 * m = 12 + 0.20 * m) ∧ m = 80 :=
by
  sorry

end equal_cost_number_of_minutes_l202_202266


namespace expression_value_l202_202440

theorem expression_value 
  (x : ℝ)
  (h : x = 1/5) :
  (x^2 - 4) / (x^2 - 2 * x) = 11 :=
  by
  rw [h]
  sorry

end expression_value_l202_202440


namespace solve_inequalities_l202_202988

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202988


namespace relationship_among_a_b_c_l202_202810

noncomputable def a : ℝ := (1 / 2) ^ (3 / 4)
noncomputable def b : ℝ := (3 / 4) ^ (1 / 2)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_among_a_b_c : a < b ∧ b < c := 
by
  -- Skipping the proof steps
  sorry

end relationship_among_a_b_c_l202_202810


namespace train_speed_is_252_144_l202_202614

/-- Train and pedestrian problem setup -/
noncomputable def train_speed (train_length : ℕ) (cross_time : ℕ) (man_speed_kmph : ℕ) : ℝ :=
  let man_speed_mps := (man_speed_kmph : ℝ) * 1000 / 3600
  let relative_speed_mps := (train_length : ℝ) / (cross_time : ℝ)
  let train_speed_mps := relative_speed_mps - man_speed_mps
  train_speed_mps * 3600 / 1000

theorem train_speed_is_252_144 :
  train_speed 500 7 5 = 252.144 := by
  sorry

end train_speed_is_252_144_l202_202614


namespace math_problem_l202_202196

-- Definitions for the conditions
def condition1 (a b c : ℝ) : Prop := a + b + c = 0
def condition2 (a b c : ℝ) : Prop := |a| > |b| ∧ |b| > |c|

-- Theorem statement
theorem math_problem (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) : c > 0 ∧ a < 0 :=
by
  sorry

end math_problem_l202_202196


namespace value_of_c_l202_202318

theorem value_of_c (c : ℝ) : (∃ x : ℝ, x^2 + c * x - 36 = 0 ∧ x = -9) → c = 5 :=
by
  sorry

end value_of_c_l202_202318


namespace boat_speed_still_water_l202_202443

-- Define the conditions
def speed_of_stream : ℝ := 4
def distance_downstream : ℕ := 68
def time_downstream : ℕ := 4

-- State the theorem
theorem boat_speed_still_water : 
  ∃V_b : ℝ, distance_downstream = (V_b + speed_of_stream) * time_downstream ∧ V_b = 13 :=
by 
  sorry

end boat_speed_still_water_l202_202443


namespace polygon_sides_l202_202102

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l202_202102


namespace fraction_of_sum_l202_202141

theorem fraction_of_sum (l : List ℝ) (hl : l.length = 51)
  (n : ℝ) (hn : n ∈ l)
  (h : n = 7 * (l.erase n).sum / 50) :
  n / l.sum = 7 / 57 := by
  sorry

end fraction_of_sum_l202_202141


namespace part_length_proof_l202_202610

-- Define the scale length in feet and inches
def scale_length_ft : ℕ := 6
def scale_length_inch : ℕ := 8

-- Define the number of equal parts
def num_parts : ℕ := 4

-- Calculate total length in inches
def total_length_inch : ℕ := scale_length_ft * 12 + scale_length_inch

-- Calculate the length of each part in inches
def part_length_inch : ℕ := total_length_inch / num_parts

-- Prove that each part is 1 foot 8 inches long
theorem part_length_proof :
  part_length_inch = 1 * 12 + 8 :=
by
  sorry

end part_length_proof_l202_202610


namespace make_polynomial_perfect_square_l202_202576

theorem make_polynomial_perfect_square (m : ℝ) :
  m = 196 → ∃ (f : ℝ → ℝ), ∀ x : ℝ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = (f x) ^ 2 :=
by
  sorry

end make_polynomial_perfect_square_l202_202576


namespace distance_between_X_and_Y_l202_202619

theorem distance_between_X_and_Y :
  ∀ (D : ℝ), 
  (10 : ℝ) * (D / (10 : ℝ) + D / (4 : ℝ)) / (10 + 4) = 142.85714285714286 → 
  D = 1000 :=
by
  intro D
  sorry

end distance_between_X_and_Y_l202_202619


namespace police_officer_placement_l202_202343

-- The given problem's conditions
def intersections : Finset String := {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}

def streets : List (Finset String) := [
    {"A", "B", "C", "D"},        -- Horizontal streets
    {"E", "F", "G"},
    {"H", "I", "J", "K"},
    {"A", "E", "H"},             -- Vertical streets
    {"B", "F", "I"},
    {"D", "G", "J"},
    {"H", "F", "C"},             -- Diagonal streets
    {"C", "G", "K"}
]

def chosen_intersections : Finset String := {"B", "G", "H"}

-- Proof problem
theorem police_officer_placement :
  ∀ street ∈ streets, ∃ p ∈ chosen_intersections, p ∈ street := by
  sorry

end police_officer_placement_l202_202343


namespace solve_inequalities_l202_202956

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202956


namespace expected_value_of_fair_dodecahedral_die_l202_202582

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_of_fair_dodecahedral_die_l202_202582


namespace solution_set_system_of_inequalities_l202_202929

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202929


namespace solution_set_l202_202873

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202873


namespace range_g_l202_202220

def f (x: ℝ) : ℝ := 4 * x - 3
def g (x: ℝ) : ℝ := f (f (f (f (f x))))

theorem range_g (x: ℝ) (h: 0 ≤ x ∧ x ≤ 3) : -1023 ≤ g x ∧ g x ≤ 2049 :=
by
  sorry

end range_g_l202_202220


namespace increasing_sequence_a_range_l202_202778

theorem increasing_sequence_a_range (a : ℝ) (a_seq : ℕ → ℝ) (h_def : ∀ n, a_seq n = 
  if n ≤ 2 then a * n^2 - ((7 / 8) * a + 17 / 4) * n + 17 / 2
  else a ^ n) : 
  (∀ n, a_seq n < a_seq (n + 1)) → a > 2 :=
by
  sorry

end increasing_sequence_a_range_l202_202778


namespace linear_inequalities_solution_l202_202864

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202864


namespace johnson_family_seating_l202_202703

def johnson_family_boys : ℕ := 5
def johnson_family_girls : ℕ := 4
def total_chairs : ℕ := 9
def total_arrangements : ℕ := Nat.factorial total_chairs

noncomputable def seating_arrangements_with_at_least_3_boys : ℕ :=
  let three_boys_block_ways := 7 * (5 * 4 * 3) * Nat.factorial 6
  total_arrangements - three_boys_block_ways

theorem johnson_family_seating : seating_arrangements_with_at_least_3_boys = 60480 := by
  unfold seating_arrangements_with_at_least_3_boys
  sorry

end johnson_family_seating_l202_202703


namespace sequence_has_max_and_min_l202_202326

noncomputable def a_n (n : ℕ) : ℝ := (4 / 9)^(n - 1) - (2 / 3)^(n - 1)

theorem sequence_has_max_and_min : 
  (∃ N, ∀ n, a_n n ≤ a_n N) ∧ 
  (∃ M, ∀ n, a_n n ≥ a_n M) :=
sorry

end sequence_has_max_and_min_l202_202326


namespace isosceles_triangle_perimeter_l202_202205

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 6) (h2 : b = 3) (h3 : a ≠ b) :
  (2 * b + a = 15) :=
by
  sorry

end isosceles_triangle_perimeter_l202_202205


namespace sum_of_special_multiples_l202_202499

def smallest_two_digit_multiple_of_5 : ℕ := 10
def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_special_multiples :
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 :=
by
  sorry

end sum_of_special_multiples_l202_202499


namespace second_hand_travel_distance_l202_202851

theorem second_hand_travel_distance (r : ℝ) (minutes : ℝ) (π : ℝ) (h : r = 10 ∧ minutes = 45 ∧ π = Real.pi) : 
  (minutes / 60) * 60 * (2 * π * r) = 900 * π := 
by sorry

end second_hand_travel_distance_l202_202851


namespace major_axis_length_l202_202621

-- Definitions of the given conditions
structure Ellipse :=
  (focus1 focus2 : ℝ × ℝ)
  (tangent_to_x_axis : Bool)

noncomputable def length_of_major_axis (E : Ellipse) : ℝ :=
  let (x1, y1) := E.focus1
  let (x2, y2) := E.focus2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 + y1) ^ 2)

-- The theorem we want to prove given the conditions
theorem major_axis_length (E : Ellipse)
  (h1 : E.focus1 = (9, 20))
  (h2 : E.focus2 = (49, 55))
  (h3 : E.tangent_to_x_axis = true):
  length_of_major_axis E = 85 :=
by
  sorry

end major_axis_length_l202_202621


namespace smallest_possible_n_l202_202363

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def n_is_three_digit (n : ℕ) : Prop := 
  n ≥ 100 ∧ n < 1000

def prime_digits_less_than_10 (p : ℕ) : Prop :=
  p ∈ [2, 3, 5, 7]

def three_distinct_prime_factors (n a b : ℕ) : Prop :=
  a ≠ b ∧ is_prime a ∧ is_prime b ∧ is_prime (10 * a + b) ∧ n = a * b * (10 * a + b)

theorem smallest_possible_n :
  ∃ (n a b : ℕ), n_is_three_digit n ∧ prime_digits_less_than_10 a ∧ prime_digits_less_than_10 b ∧ three_distinct_prime_factors n a b ∧ n = 138 :=
by {
  sorry
}

end smallest_possible_n_l202_202363


namespace difference_Q_R_l202_202746

variable (P Q R : ℝ) (x : ℝ)

theorem difference_Q_R (h1 : 11 * x - 5 * x = 12100) : 19 * x - 11 * x = 16133.36 :=
by
  sorry

end difference_Q_R_l202_202746


namespace find_number_l202_202434

theorem find_number (x : ℝ) : (x + 1) / (x + 5) = (x + 5) / (x + 13) → x = 3 :=
sorry

end find_number_l202_202434


namespace right_triangle_hypotenuse_l202_202606

theorem right_triangle_hypotenuse {a b c : ℝ} 
  (h1: a + b + c = 60) 
  (h2: a * b = 96) 
  (h3: a^2 + b^2 = c^2) : 
  c = 28.4 := 
sorry

end right_triangle_hypotenuse_l202_202606


namespace simplify_power_l202_202695

theorem simplify_power (z : ℂ) (h₁ : z = (1 + complex.I) / (1 - complex.I)) : z ^ 1002 = -1 :=
by 
  sorry

end simplify_power_l202_202695


namespace coprime_pairs_solution_l202_202301

theorem coprime_pairs_solution (x y : ℕ) (hx : x ∣ y^2 + 210) (hy : y ∣ x^2 + 210) (hxy : Nat.gcd x y = 1) :
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) :=
by sorry

end coprime_pairs_solution_l202_202301


namespace solution_set_system_of_inequalities_l202_202927

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202927


namespace quadrilateral_area_l202_202684

theorem quadrilateral_area (a b c d : ℝ) (horizontally_vertically_apart : a = b + 1 ∧ b = c + 1 ∧ c = d + 1 ∧ d = a + 1) : 
  area_of_quadrilateral = 6 :=
sorry

end quadrilateral_area_l202_202684


namespace minimum_value_x_add_2y_minimum_value_x_add_2y_achieved_l202_202665

theorem minimum_value_x_add_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : 
  x + 2 * y ≥ 19 + 6 * Real.sqrt 2 := 
sorry

theorem minimum_value_x_add_2y_achieved (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : 
  ∃ x y, 0 < x ∧ 0 < y ∧ 1/x + 9/y = 1 ∧ x + 2 * y = 19 + 6 * Real.sqrt 2 :=
sorry

end minimum_value_x_add_2y_minimum_value_x_add_2y_achieved_l202_202665


namespace JacobNeed_l202_202675

-- Definitions of the conditions
def jobEarningsBeforeTax : ℝ := 25 * 15
def taxAmount : ℝ := 0.10 * jobEarningsBeforeTax
def jobEarningsAfterTax : ℝ := jobEarningsBeforeTax - taxAmount

def cookieEarnings : ℝ := 5 * 30

def tutoringEarnings : ℝ := 100 * 4

def lotteryWinnings : ℝ := 700 - 20
def friendShare : ℝ := 0.30 * lotteryWinnings
def netLotteryWinnings : ℝ := lotteryWinnings - friendShare

def giftFromSisters : ℝ := 700 * 2

def totalEarnings : ℝ := jobEarningsAfterTax + cookieEarnings + tutoringEarnings + netLotteryWinnings + giftFromSisters

def travelGearExpenses : ℝ := 3 + 47

def netSavings : ℝ := totalEarnings - travelGearExpenses

def tripCost : ℝ := 8000

-- Statement to be proven
theorem JacobNeed (jobEarningsBeforeTax taxAmount jobEarningsAfterTax cookieEarnings tutoringEarnings 
netLotteryWinnings giftFromSisters totalEarnings travelGearExpenses netSavings tripCost : ℝ) : 
  (jobEarningsAfterTax == (25 * 15) - (0.10 * (25 * 15))) → 
  (cookieEarnings == 5 * 30) →
  (tutoringEarnings == 100 * 4) →
  (netLotteryWinnings == (700 - 20) - (0.30 * (700 - 20))) →
  (giftFromSisters == 700 * 2) →
  (totalEarnings == jobEarningsAfterTax + cookieEarnings + tutoringEarnings + netLotteryWinnings + giftFromSisters) →
  (travelGearExpenses == 3 + 47) →
  (netSavings == totalEarnings - travelGearExpenses) →
  (tripCost == 8000) →
  (tripCost - netSavings = 5286.50) :=
by
  intros
  sorry

end JacobNeed_l202_202675


namespace count_primes_squared_in_range_l202_202781

theorem count_primes_squared_in_range : 
  (finset.card (finset.filter (λ p, p^2 ≥ 5000 ∧ p^2 ≤ 9000) 
    (finset.filter nat.prime (finset.range 95)))) = 5 := 
by sorry

end count_primes_squared_in_range_l202_202781


namespace paint_after_third_day_l202_202732

def initial_paint := 2
def paint_used_first_day (x : ℕ) := (1 / 2) * x
def remaining_after_first_day (x : ℕ) := x - paint_used_first_day x
def paint_used_second_day (y : ℕ) := (1 / 4) * y
def remaining_after_second_day (y : ℕ) := y - paint_used_second_day y
def paint_used_third_day (z : ℕ) := (1 / 3) * z
def remaining_after_third_day (z : ℕ) := z - paint_used_third_day z

theorem paint_after_third_day :
  remaining_after_third_day 
    (remaining_after_second_day 
      (remaining_after_first_day initial_paint)) = initial_paint / 2 := 
  by
  sorry

end paint_after_third_day_l202_202732


namespace exists_least_number_l202_202727

open Nat

theorem exists_least_number
  (N : ℕ)
  (h1 : N % 5 = 3)
  (h2 : N % 6 = 3)
  (h3 : N % 7 = 3)
  (h4 : N % 8 = 3)
  (h5 : N % 9 = 0)
  : N = 1683 :=
sorry

end exists_least_number_l202_202727


namespace area_of_triangle_l202_202825

theorem area_of_triangle (p : ℝ) (h_p : 0 < p ∧ p < 10) : 
    let C := (0, p)
    let O := (0, 0)
    let B := (10, 0)
    (1/2) * 10 * p = 5 * p := 
by
  sorry

end area_of_triangle_l202_202825


namespace f_at_3_l202_202187

-- Define the function f and its conditions
variable (f : ℝ → ℝ)

-- The domain of the function f is ℝ, hence f : ℝ → ℝ
-- Also given:
axiom f_symm : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom f_add : f (-1) + f (3) = 12

-- Final proof statement
theorem f_at_3 : f 3 = 6 :=
by
  sorry

end f_at_3_l202_202187


namespace solution_set_l202_202874

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202874


namespace percent_of_x_is_y_l202_202795

theorem percent_of_x_is_y 
    (x y : ℝ) 
    (h : 0.30 * (x - y) = 0.20 * (x + y)) : 
    y / x = 0.2 :=
  sorry

end percent_of_x_is_y_l202_202795


namespace solution_set_of_inequalities_l202_202963

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202963


namespace solution_set_of_inequalities_l202_202960

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202960


namespace number_of_children_l202_202735

def weekly_husband : ℕ := 335
def weekly_wife : ℕ := 225
def weeks_in_six_months : ℕ := 24
def amount_per_child : ℕ := 1680

theorem number_of_children : (weekly_husband + weekly_wife) * weeks_in_six_months / 2 / amount_per_child = 4 := by
  sorry

end number_of_children_l202_202735


namespace rent_expense_calculation_l202_202451

variable (S : ℝ)
variable (saved_amount : ℝ := 2160)
variable (milk_expense : ℝ := 1500)
variable (groceries_expense : ℝ := 4500)
variable (education_expense : ℝ := 2500)
variable (petrol_expense : ℝ := 2000)
variable (misc_expense : ℝ := 3940)
variable (salary_percent_saved : ℝ := 0.10)

theorem rent_expense_calculation 
  (h1 : salary_percent_saved * S = saved_amount) :
  S = 21600 → 
  0.90 * S - (milk_expense + groceries_expense + education_expense + petrol_expense + misc_expense) = 5000 :=
by
  sorry

end rent_expense_calculation_l202_202451


namespace polygon_sides_l202_202092

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202092


namespace fairy_island_county_problem_l202_202516

theorem fairy_island_county_problem :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  -- After the first year:
  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  -- After the second year:
  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  -- After the third year:
  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  third_year_elves + third_year_dwarves + third_year_centaurs = 54 :=
by
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  have third_year_counties := third_year_elves + third_year_dwarves + third_year_centaurs
  show third_year_counties = 54 by
    calc third_year_counties = 24 + 18 + 12 := by sorry
                          ... = 54 := by sorry

end fairy_island_county_problem_l202_202516


namespace count_edge_cubes_l202_202146

/-- 
A cube is painted red on all faces and then cut into 27 equal smaller cubes.
Prove that the number of smaller cubes that are painted on only 2 faces is 12. 
-/
theorem count_edge_cubes (c : ℕ) (inner : ℕ)  (edge : ℕ) (face : ℕ) :
  (c = 27 ∧ inner = 1 ∧ edge = 12 ∧ face = 6) → edge = 12 :=
by
  -- Given the conditions from the problem statement
  sorry

end count_edge_cubes_l202_202146


namespace trailing_zeros_30_factorial_l202_202289

-- Definitions directly from conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def trailing_zeros (n : ℕ) : ℕ :=
  let count_five_factors (k : ℕ) : ℕ :=
    k / 5 + k / 25 + k / 125 -- This generalizes for higher powers of 5 which is sufficient here.
  count_five_factors n

-- Mathematical proof problem statement
theorem trailing_zeros_30_factorial : trailing_zeros 30 = 7 := by
  sorry

end trailing_zeros_30_factorial_l202_202289


namespace units_digit_of_147_pow_is_7_some_exponent_units_digit_l202_202719

theorem units_digit_of_147_pow_is_7 (n : ℕ) : (147 ^ 25) % 10 = 7 % 10 :=
by
  sorry

theorem some_exponent_units_digit (n : ℕ) (hn : n % 4 = 2) : ((147 ^ 25) ^ n) % 10 = 9 :=
by
  have base_units_digit := units_digit_of_147_pow_is_7 25
  sorry

end units_digit_of_147_pow_is_7_some_exponent_units_digit_l202_202719


namespace combined_rocket_height_l202_202350

theorem combined_rocket_height :
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 :=
by
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  sorry

end combined_rocket_height_l202_202350


namespace count_primes_squared_in_range_l202_202782

theorem count_primes_squared_in_range : 
  (finset.card (finset.filter (λ p, p^2 ≥ 5000 ∧ p^2 ≤ 9000) 
    (finset.filter nat.prime (finset.range 95)))) = 5 := 
by sorry

end count_primes_squared_in_range_l202_202782


namespace Eric_rent_days_l202_202521

-- Define the conditions given in the problem
def daily_rate := 50.00
def rate_14_days := 500.00
def total_cost := 800.00

-- State the problem as a theorem in Lean
theorem Eric_rent_days : ∀ (d : ℕ), (d : ℕ) = 20 :=
by
  sorry

end Eric_rent_days_l202_202521


namespace polygon_sides_l202_202106

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202106


namespace solution1_solution2_l202_202805

noncomputable def problem1 (a b : ℝ) (C B c : ℝ) : Prop :=
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B

noncomputable def problem2 (A : ℝ) : Prop :=
  (Real.sqrt 3) * Real.sin (2 * A - (Real.pi / 6)) 
  - 2 * (Real.sin (C - Real.pi / 12)) ^ 2 = 0

theorem solution1 :
  problem1 2 (Real.sqrt 7) C (Real.pi / 3) 3 := sorry

theorem solution2 :
  problem2 (Real.pi / 4) := sorry

end solution1_solution2_l202_202805


namespace linear_inequalities_solution_l202_202860

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202860


namespace polygon_sides_l202_202096

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202096


namespace find_numbers_l202_202999

theorem find_numbers (a b c : ℕ) (h : a + b = 2015) (h' : a = 10 * b + c) (hc : 0 ≤ c ∧ c ≤ 9) :
  (a = 1832 ∧ b = 183) :=
sorry

end find_numbers_l202_202999


namespace solve_for_xy_l202_202033

theorem solve_for_xy (x y : ℝ) (h : 2 * x - 3 ≤ Real.log (x + y + 1) + Real.log (x - y - 2)) : x * y = -9 / 4 :=
by sorry

end solve_for_xy_l202_202033


namespace solution_set_of_linear_inequalities_l202_202910

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202910


namespace jam_jars_weight_l202_202045

noncomputable def jars_weight 
    (initial_suitcase_weight : ℝ) 
    (perfume_weight_oz : ℝ) (num_perfume : ℕ)
    (chocolate_weight_lb : ℝ)
    (soap_weight_oz : ℝ) (num_soap : ℕ)
    (total_return_weight : ℝ)
    (oz_to_lb : ℝ) : ℝ :=
  initial_suitcase_weight 
  + (num_perfume * perfume_weight_oz) / oz_to_lb 
  + chocolate_weight_lb 
  + (num_soap * soap_weight_oz) / oz_to_lb

theorem jam_jars_weight
    (initial_suitcase_weight : ℝ := 5)
    (perfume_weight_oz : ℝ := 1.2) (num_perfume : ℕ := 5)
    (chocolate_weight_lb : ℝ := 4)
    (soap_weight_oz : ℝ := 5) (num_soap : ℕ := 2)
    (total_return_weight : ℝ := 11)
    (oz_to_lb : ℝ := 16) :
    jars_weight initial_suitcase_weight perfume_weight_oz num_perfume
      chocolate_weight_lb soap_weight_oz num_soap total_return_weight oz_to_lb + (jars_weight initial_suitcase_weight perfume_weight_oz num_perfume
      chocolate_weight_lb soap_weight_oz num_soap total_return_weight oz_to_lb) = 1 :=
by
  sorry

end jam_jars_weight_l202_202045


namespace zero_points_of_gx_l202_202320

noncomputable def fx (a x : ℝ) : ℝ := (1 / 2) * x^2 - abs (x - 2 * a)
noncomputable def gx (a x : ℝ) : ℝ := 4 * a * x^2 + 2 * x + 1

theorem zero_points_of_gx (a : ℝ) (h : -1 / 4 ≤ a ∧ a ≤ 1 / 4) : 
  ∃ n, (n = 1 ∨ n = 2) ∧ (∃ x1 x2, gx a x1 = 0 ∧ gx a x2 = 0) := 
sorry

end zero_points_of_gx_l202_202320


namespace largest_digit_never_in_odd_unit_l202_202256

-- Definition for what constitutes odd number unit digits
def odd_unit_digits : set ℕ := {1, 3, 5, 7, 9}

-- Definition for the largest digit that is not in the given set of odd_unit_digits
def largest_missing_digit : ℕ :=
  let even_digits := {0, 2, 4, 6, 8} in set.max even_digits

-- The theorem stating the problem
theorem largest_digit_never_in_odd_unit : largest_missing_digit ∉ odd_unit_digits :=
by {
  -- Definition of the largest missing digit aligned with the above question and conditions
  have evens : set ℕ := {0, 2, 4, 6, 8},
  have largest : largest_missing_digit = 8 := sorry, -- This would be proved based on max of the set
  -- Now proving largest is not in the odd_unit_digits
  rw largest,
  exact dec_trivial
}

end largest_digit_never_in_odd_unit_l202_202256


namespace min_hypotenuse_l202_202800

theorem min_hypotenuse {a b : ℝ} (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 10) :
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ c ≥ 5 * Real.sqrt 2 :=
by
  sorry

end min_hypotenuse_l202_202800


namespace solution_set_inequalities_l202_202890

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202890


namespace prob_product_lt_36_l202_202532

open ProbabilityTheory

noncomputable def P_event (n: ℕ) : dist ℕ := pmf.uniform (finset.range (n+1)).erase 0

theorem prob_product_lt_36 :
  let paco := P_event 6 in
  let manu := P_event 12 in
  P (λ x : ℕ × ℕ, x.1 * x.2 < 36) (paco.prod manu) = 67 / 72 :=
by sorry

end prob_product_lt_36_l202_202532


namespace basketball_cards_per_box_l202_202760

-- Given conditions
def num_basketball_boxes : ℕ := 9
def num_football_boxes := num_basketball_boxes - 3
def cards_per_football_box : ℕ := 20
def total_cards : ℕ := 255
def total_football_cards := num_football_boxes * cards_per_football_box

-- We want to prove that the number of cards in each basketball card box is 15
theorem basketball_cards_per_box :
  (total_cards - total_football_cards) / num_basketball_boxes = 15 := by
  sorry

end basketball_cards_per_box_l202_202760


namespace solution_set_of_linear_inequalities_l202_202911

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202911


namespace value_subtracted_3_times_number_eq_1_l202_202704

variable (n : ℝ) (v : ℝ)

theorem value_subtracted_3_times_number_eq_1 (h1 : n = 1.0) (h2 : 3 * n - v = 2 * n) : v = 1 :=
by
  sorry

end value_subtracted_3_times_number_eq_1_l202_202704


namespace solution_set_linear_inequalities_l202_202897

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202897


namespace infinite_series_sum_l202_202459

noncomputable def partial_sum (n : ℕ) : ℚ := (2 * n - 1) / (n * (n + 1) * (n + 2))

theorem infinite_series_sum : (∑' n, partial_sum (n + 1)) = 3 / 4 :=
by
  sorry

end infinite_series_sum_l202_202459


namespace factorize_expression_l202_202764

variables {a x y : ℝ}

theorem factorize_expression (a x y : ℝ) : 3 * a * x ^ 2 + 6 * a * x * y + 3 * a * y ^ 2 = 3 * a * (x + y) ^ 2 :=
by
  sorry

end factorize_expression_l202_202764


namespace solution_set_of_linear_inequalities_l202_202913

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202913


namespace masking_tape_needed_l202_202627

def wall1_width : ℝ := 4
def wall1_count : ℕ := 2
def wall2_width : ℝ := 6
def wall2_count : ℕ := 2
def door_width : ℝ := 2
def door_count : ℕ := 1
def window_width : ℝ := 1.5
def window_count : ℕ := 2

def total_width_of_walls : ℝ := (wall1_count * wall1_width) + (wall2_count * wall2_width)
def total_width_of_door_and_windows : ℝ := (door_count * door_width) + (window_count * window_width)

theorem masking_tape_needed : total_width_of_walls - total_width_of_door_and_windows = 15 := by
  sorry

end masking_tape_needed_l202_202627


namespace weight_units_correct_l202_202765

-- Definitions of weights
def weight_peanut_kernel := 1 -- gram
def weight_truck_capacity := 8 -- ton
def weight_xiao_ming := 30 -- kilogram
def weight_basketball := 580 -- gram

-- Proof that the weights have correct units
theorem weight_units_correct :
  (weight_peanut_kernel = 1 ∧ weight_truck_capacity = 8 ∧ weight_xiao_ming = 30 ∧ weight_basketball = 580) :=
by {
  sorry
}

end weight_units_correct_l202_202765


namespace distance_between_Sasha_and_Kolya_l202_202397

/-- Sasha, Lesha, and Kolya simultaneously started a 100-meter race.
Assuming all three run at constant but unequal speeds, when Sasha
finished (100 meters), Lesha was 10 meters behind him; and when Lesha
finished, Kolya was 10 meters behind him. Thus, the distance between
Sasha and Kolya when Sasha finished is 19 meters. -/
theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL tK : ℝ), 
  vS > 0 ∧ vL > 0 ∧ vK > 0 ∧ 
  tS = 100 / vS ∧ 
  tL = 100 / vL ∧ 
  tK = 100 / vK ∧ 
  vL * tS = 90 ∧ 
  vK * tL = 90 →
  vS * tS - vK * tS = 19 :=
begin
  sorry
end

end distance_between_Sasha_and_Kolya_l202_202397


namespace find_a_l202_202356

def A (x : ℝ) : Prop := x^2 - 4 ≤ 0
def B (x : ℝ) (a : ℝ) : Prop := 2 * x + a ≤ 0
def IntersectAB (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 1

theorem find_a (a : ℝ) :
  (∀ x : ℝ, A x → B x a → IntersectAB x) → a = -2 :=
begin
  sorry
end

end find_a_l202_202356


namespace singer_worked_10_hours_per_day_l202_202450

noncomputable def hours_per_day_worked_on_one_song (total_songs : ℕ) (days_per_song : ℕ) (total_hours : ℕ) : ℕ :=
  total_hours / (total_songs * days_per_song)

theorem singer_worked_10_hours_per_day :
  hours_per_day_worked_on_one_song 3 10 300 = 10 := 
by
  sorry

end singer_worked_10_hours_per_day_l202_202450


namespace polygon_sides_l202_202111

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l202_202111


namespace line_equation_k_value_l202_202209

theorem line_equation_k_value (m n k : ℝ) 
    (h1 : m = 2 * n + 5) 
    (h2 : m + 5 = 2 * (n + k) + 5) : 
    k = 2.5 :=
by sorry

end line_equation_k_value_l202_202209


namespace remainder_of_150_div_k_l202_202769

theorem remainder_of_150_div_k (k : ℕ) (hk : k > 0) (h1 : 90 % (k^2) = 10) :
  150 % k = 2 := 
sorry

end remainder_of_150_div_k_l202_202769


namespace floor_sqrt_80_eq_8_l202_202468

theorem floor_sqrt_80_eq_8
  (h1 : 8^2 = 64)
  (h2 : 9^2 = 81)
  (h3 : 64 < 80 ∧ 80 < 81)
  (h4 : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) : 
  Int.floor (Real.sqrt 80) = 8 := by
  sorry

end floor_sqrt_80_eq_8_l202_202468


namespace no_solution_x_to_2n_plus_y_to_2n_eq_z_sq_l202_202016

theorem no_solution_x_to_2n_plus_y_to_2n_eq_z_sq (n : ℕ) (h : ∀ (x y z : ℕ), x^n + y^n ≠ z^n) : ∀ (x y z : ℕ), x^(2*n) + y^(2*n) ≠ z^2 :=
by 
  intro x y z
  sorry

end no_solution_x_to_2n_plus_y_to_2n_eq_z_sq_l202_202016


namespace piles_3_stones_impossible_l202_202248

theorem piles_3_stones_impossible :
  ∀ n : ℕ, ∀ piles : ℕ → ℕ,
  (piles 0 = 1001) →
  (∀ k : ℕ, k > 0 → ∃ i j : ℕ, piles (k-1) > 1 → piles k = i + j ∧ i > 0 ∧ j > 0) →
  ¬ (∀ m : ℕ, piles m ≠ 3) :=
by
  sorry

end piles_3_stones_impossible_l202_202248


namespace sum_of_three_smallest_two_digit_primes_l202_202258

theorem sum_of_three_smallest_two_digit_primes :
  11 + 13 + 17 = 41 :=
by
  sorry

end sum_of_three_smallest_two_digit_primes_l202_202258


namespace range_of_a_l202_202019

noncomputable def f (a x : ℝ) : ℝ :=
if h : a ≤ x ∧ x < 0 then -((1/2)^x)
else if h' : 0 ≤ x ∧ x ≤ 4 then -(x^2) + 2*x
else 0

theorem range_of_a (a : ℝ) (h : ∀ x, f a x ∈ Set.Icc (-8 : ℝ) (1 : ℝ)) : 
  a ∈ Set.Ico (-3 : ℝ) 0 :=
sorry

end range_of_a_l202_202019


namespace solution_set_linear_inequalities_l202_202935

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202935


namespace value_corresponds_l202_202025

-- Define the problem
def certain_number (x : ℝ) : Prop :=
  0.30 * x = 120

-- State the theorem to be proved
theorem value_corresponds (x : ℝ) (h : certain_number x) : 0.40 * x = 160 :=
by
  sorry

end value_corresponds_l202_202025


namespace calculate_expression_l202_202160

theorem calculate_expression : 
  (Real.sqrt 3) ^ 0 + 2 ^ (-1:ℤ) + Real.sqrt 2 * Real.cos (Float.pi / 4) - Real.abs (-1/2) = 2 := 
by
  sorry

end calculate_expression_l202_202160


namespace isosceles_perimeter_l202_202421

theorem isosceles_perimeter (peri_eqt : ℕ) (side_eqt : ℕ) (base_iso : ℕ) (side_iso : ℕ)
    (h1 : peri_eqt = 60)
    (h2 : side_eqt = peri_eqt / 3)
    (h3 : side_iso = side_eqt)
    (h4 : base_iso = 25) :
  2 * side_iso + base_iso = 65 :=
by
  sorry

end isosceles_perimeter_l202_202421


namespace walmart_knives_eq_three_l202_202430

variable (k : ℕ)

-- Walmart multitool
def walmart_tools : ℕ := 1 + k + 2

-- Target multitool (with twice as many knives as Walmart)
def target_tools : ℕ := 1 + 2 * k + 3 + 1

-- The condition that Target multitool has 5 more tools compared to Walmart
theorem walmart_knives_eq_three (h : target_tools k = walmart_tools k + 5) : k = 3 :=
by
  sorry

end walmart_knives_eq_three_l202_202430


namespace john_weekly_earnings_increase_l202_202354

theorem john_weekly_earnings_increase :
  let earnings_before := 60 + 100
  let earnings_after := 78 + 120
  let increase := earnings_after - earnings_before
  (increase / earnings_before : ℚ) * 100 = 23.75 :=
by
  -- Definitions
  let earnings_before := (60 : ℚ) + 100
  let earnings_after := (78 : ℚ) + 120
  let increase := earnings_after - earnings_before

  -- Calculation of percentage increase
  let percentage_increase : ℚ := (increase / earnings_before) * 100

  -- Expected result
  have expected_result : percentage_increase = 23.75 := by sorry
  exact expected_result

end john_weekly_earnings_increase_l202_202354


namespace tan_beta_minus_2alpha_l202_202010

noncomputable def tan_alpha := 1 / 2
noncomputable def tan_beta_minus_alpha := 2 / 5
theorem tan_beta_minus_2alpha (α β : ℝ) (h1 : Real.tan α = tan_alpha) (h2 : Real.tan (β - α) = tan_beta_minus_alpha) :
  Real.tan (β - 2 * α) = -1 / 12 := 
by
  sorry

end tan_beta_minus_2alpha_l202_202010


namespace distinct_integer_pairs_l202_202770

theorem distinct_integer_pairs :
  ∃ pairs : (Nat × Nat) → Prop,
  (∀ x y : Nat, pairs (x, y) → 0 < x ∧ x < y ∧ (8 * Real.sqrt 31 = Real.sqrt x + Real.sqrt y))
  ∧ (∃! p, pairs p) → (∃! q, pairs q) → (∃! r, pairs r) → true := sorry

end distinct_integer_pairs_l202_202770


namespace min_perimeter_is_676_l202_202715

-- Definitions and conditions based on the problem statement
def equal_perimeter (a b c : ℕ) : Prop :=
  2 * a + 14 * c = 2 * b + 16 * c

def equal_area (a b c : ℕ) : Prop :=
  7 * Real.sqrt (a^2 - 49 * c^2) = 8 * Real.sqrt (b^2 - 64 * c^2)

def base_ratio (b : ℕ) : ℕ := b * 8 / 7

theorem min_perimeter_is_676 :
  ∃ a b c : ℕ, equal_perimeter a b c ∧ equal_area a b c ∧ base_ratio b = a - b ∧ 
  2 * a + 14 * c = 676 :=
sorry

end min_perimeter_is_676_l202_202715


namespace num_palindromes_is_correct_l202_202022

section Palindromes

def num_alphanumeric_chars : ℕ := 10 + 26

def num_four_char_palindromes : ℕ := num_alphanumeric_chars * num_alphanumeric_chars

theorem num_palindromes_is_correct : num_four_char_palindromes = 1296 :=
by
  sorry

end Palindromes

end num_palindromes_is_correct_l202_202022


namespace largest_n_for_factoring_polynomial_l202_202473

theorem largest_n_for_factoring_polynomial :
  ∃ A B : ℤ, A * B = 120 ∧ (∀ n, (5 * 120 + 1 ≤ n → n ≤ 601)) := sorry

end largest_n_for_factoring_polynomial_l202_202473


namespace polygon_sides_l202_202097

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202097


namespace Cindy_initial_marbles_l202_202754

theorem Cindy_initial_marbles (M : ℕ) 
  (h1 : 4 * (M - 320) = 720) : M = 500 :=
by
  sorry

end Cindy_initial_marbles_l202_202754


namespace solution_set_linear_inequalities_l202_202898

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202898


namespace negation_of_symmetry_about_y_eq_x_l202_202557

theorem negation_of_symmetry_about_y_eq_x :
  ¬ (∀ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x) ↔ ∃ f : ℝ → ℝ, ∃ x : ℝ, f (f x) ≠ x :=
by sorry

end negation_of_symmetry_about_y_eq_x_l202_202557


namespace solve_inequalities_l202_202991

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202991


namespace no_positive_real_solutions_l202_202003

theorem no_positive_real_solutions 
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^3 + y^3 + z^3 = x + y + z) (h2 : x^2 + y^2 + z^2 = x * y * z) :
  false :=
by sorry

end no_positive_real_solutions_l202_202003


namespace james_total_chore_time_l202_202347

theorem james_total_chore_time : 
  let vacuuming_time := 3
  let other_chores_time := 3 * vacuuming_time
  vacuuming_time + other_chores_time = 12 :=
by 
  let vacuuming_time := 3
  let other_chores_time := 3 * vacuuming_time
  have h1 : vacuuming_time + other_chores_time = 12 := by
    calc
      vacuuming_time + other_chores_time = 3 + (3 * 3) : by rfl
      ... = 3 + 9 : by rfl
      ... = 12 : by rfl
  exact h1

end james_total_chore_time_l202_202347


namespace diff_of_squares_l202_202623

-- Define constants a and b
def a : ℕ := 65
def b : ℕ := 35

-- State the theorem to be proven using the difference of squares formula
theorem diff_of_squares : a^2 - b^2 = 3000 := by
  have h1 : a + b = 100 := by rfl -- Checking a + b
  have h2 : a - b = 30 := by rfl -- Checking a - b
  have h3 : (a+b)*(a-b) = 3000 := by
    rw [h1, h2] -- Substituting the values
    norm_num -- Simplifying the arithmetics
  exact h3 -- Providing the ultimate result

end diff_of_squares_l202_202623


namespace plane_equation_l202_202575

variable (x y z : ℝ)

def line1 := 3 * x - 2 * y + 5 * z + 3 = 0
def line2 := x + 2 * y - 3 * z - 11 = 0
def origin_plane := 18 * x - 8 * y + 23 * z = 0

theorem plane_equation : 
  (∀ x y z, line1 x y z → line2 x y z → origin_plane x y z) :=
by
  sorry

end plane_equation_l202_202575


namespace theme_park_ratio_l202_202080

theorem theme_park_ratio (a c : ℕ) (h_cost_adult : 20 * a + 15 * c = 1600) (h_eq_ratio : a * 28 = c * 59) :
  a / c = 59 / 28 :=
by
  /-
  Proof steps would go here.
  -/
  sorry

end theme_park_ratio_l202_202080


namespace total_trucks_l202_202688

theorem total_trucks {t : ℕ} (h1 : 2 * t + t = 300) : t = 100 := 
by sorry

end total_trucks_l202_202688


namespace numbers_identification_l202_202685

-- Definitions
def is_natural (n : ℤ) : Prop := n ≥ 0
def is_integer (n : ℤ) : Prop := True

-- Theorem
theorem numbers_identification :
  (is_natural 0 ∧ is_natural 2 ∧ is_natural 6 ∧ is_natural 7) ∧
  (is_integer (-15) ∧ is_integer (-3) ∧ is_integer 0 ∧ is_integer 4) :=
by
  sorry

end numbers_identification_l202_202685


namespace hyperbola_b_value_l202_202189

theorem hyperbola_b_value (b : ℝ) (h₁ : b > 0) 
  (h₂ : ∃ x y, x^2 - (y^2 / b^2) = 1 ∧ (∀ (c : ℝ), c = Real.sqrt (1 + b^2) → c / 1 = 2)) : b = Real.sqrt 3 :=
by { sorry }

end hyperbola_b_value_l202_202189


namespace amount_leaked_during_repairs_l202_202748

theorem amount_leaked_during_repairs:
  let total_leaked := 6206
  let leaked_before_repairs := 2475
  total_leaked - leaked_before_repairs = 3731 :=
by
  sorry

end amount_leaked_during_repairs_l202_202748


namespace largest_angle_l202_202556

-- Assume the conditions
def angle_a : ℝ := 50
def angle_b : ℝ := 70
def angle_c (y : ℝ) : ℝ := 180 - (angle_a + angle_b)

-- State the proposition
theorem largest_angle (y : ℝ) (h : y = angle_c y) : angle_b = 70 := by
  sorry

end largest_angle_l202_202556


namespace solution_set_l202_202880

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202880


namespace solution_set_of_inequalities_l202_202965

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202965


namespace ratio_w_y_l202_202562

theorem ratio_w_y (w x y z : ℚ) 
  (h1 : w / x = 5 / 2) 
  (h2 : y / z = 5 / 3) 
  (h3 : z / x = 1 / 6) : 
  w / y = 9 := 
by 
  sorry

end ratio_w_y_l202_202562


namespace problem_decimal_parts_l202_202322

theorem problem_decimal_parts :
  let a := 5 + Real.sqrt 7 - 7
  let b := 5 - Real.sqrt 7 - 2
  (a + b) ^ 2023 = 1 :=
by
  sorry

end problem_decimal_parts_l202_202322


namespace max_alpha_value_l202_202055

variable (a b x y α : ℝ)

theorem max_alpha_value (h1 : a = 2 * b)
    (h2 : a^2 + y^2 = b^2 + x^2)
    (h3 : b^2 + x^2 = (a - x)^2 + (b - y)^2)
    (h4 : 0 ≤ x) (h5 : x < a) (h6 : 0 ≤ y) (h7 : y < b) :
    α = a / b → α^2 = 4 := 
by
  sorry

end max_alpha_value_l202_202055


namespace black_cars_in_parking_lot_l202_202125

theorem black_cars_in_parking_lot :
  let total_cars := 3000
  let blue_percent := 0.40
  let red_percent := 0.25
  let green_percent := 0.15
  let yellow_percent := 0.10
  let black_percent := 1 - (blue_percent + red_percent + green_percent + yellow_percent)
  let number_of_black_cars := total_cars * black_percent
  number_of_black_cars = 300 :=
by
  sorry

end black_cars_in_parking_lot_l202_202125


namespace total_lambs_l202_202818

def num_initial_lambs : ℕ := 6
def num_baby_lambs_per_mother : ℕ := 2
def num_mothers : ℕ := 2
def traded_lambs : ℕ := 3
def extra_lambs : ℕ := 7

theorem total_lambs :
  num_initial_lambs + (num_baby_lambs_per_mother * num_mothers) - traded_lambs + extra_lambs = 14 :=
by
  sorry

end total_lambs_l202_202818


namespace second_hand_travel_distance_l202_202852

theorem second_hand_travel_distance (r : ℝ) (minutes : ℝ) (π : ℝ) (h : r = 10 ∧ minutes = 45 ∧ π = Real.pi) : 
  (minutes / 60) * 60 * (2 * π * r) = 900 * π := 
by sorry

end second_hand_travel_distance_l202_202852


namespace fraction_to_decimal_l202_202298

theorem fraction_to_decimal : (3 / 24 : ℚ) = 0.125 := 
by
  -- proof will be filled here
  sorry

end fraction_to_decimal_l202_202298


namespace major_axis_length_l202_202455

/-- Defines the properties of the ellipse we use in this problem. --/
def ellipse (x y : ℝ) : Prop :=
  let f1 := (5, 1 + Real.sqrt 8)
  let f2 := (5, 1 - Real.sqrt 8)
  let tangent_line_at_y := y = 1
  let tangent_line_at_x := x = 1
  tangent_line_at_y ∧ tangent_line_at_x ∧
  ((x - f1.1)^2 + (y - f1.2)^2) + ((x - f2.1)^2 + (y - f2.2)^2) = 4

/-- Proves the length of the major axis of the specific ellipse --/
theorem major_axis_length : ∃ l : ℝ, l = 4 :=
  sorry

end major_axis_length_l202_202455


namespace sample_size_of_survey_l202_202297

def eighth_grade_students : ℕ := 350
def selected_students : ℕ := 50

theorem sample_size_of_survey : selected_students = 50 :=
by sorry

end sample_size_of_survey_l202_202297


namespace ellipse_product_l202_202826

/-- Given conditions:
1. OG = 8
2. The diameter of the inscribed circle of triangle ODG is 4
3. O is the center of an ellipse with major axis AB and minor axis CD
4. Point G is one focus of the ellipse
--/
theorem ellipse_product :
  ∀ (O G D : Point) (a b : ℝ),
    OG = 8 → 
    (a^2 - b^2 = 64) →
    (a - b = 4) →
    (AB = 2*a) →
    (CD = 2*b) →
    (AB * CD = 240) :=
by
  intros O G D a b hOG h1 h2 h3 h4
  sorry

end ellipse_product_l202_202826


namespace part1_part2_l202_202327

open Real

noncomputable def f (x : ℝ) : ℝ := abs ((2 / 3) * x + 1)

theorem part1 (a : ℝ) : (∀ x, f x ≥ -abs x + a) → a ≤ 1 :=
sorry

theorem part2 (x y : ℝ) (h1 : abs (x + y + 1) ≤ 1 / 3) (h2 : abs (y - 1 / 3) ≤ 2 / 3) : 
  f x ≤ 7 / 9 :=
sorry

end part1_part2_l202_202327


namespace race_distance_l202_202379

/-- Given that Sasha, Lesha, and Kolya start a 100m race simultaneously and run at constant velocities,
when Sasha finishes, Lesha is 10m behind, and when Lesha finishes, Kolya is 10m behind.
Prove that the distance between Sasha and Kolya when Sasha finishes is 19 meters. -/
theorem race_distance
    (v_S v_L v_K : ℝ)
    (h1 : 100 / v_S - 100 / v_L = 10 / v_L)
    (h2 : 100 / v_L - 100 / v_K = 10 / v_K) :
    100 - 81 = 19 :=
by
  sorry

end race_distance_l202_202379


namespace sarah_age_is_26_l202_202546

theorem sarah_age_is_26 (mark_age billy_age ana_age : ℕ) (sarah_age : ℕ) 
  (h1 : sarah_age = 3 * mark_age - 4)
  (h2 : mark_age = billy_age + 4)
  (h3 : billy_age = ana_age / 2)
  (h4 : ana_age = 15 - 3) :
  sarah_age = 26 := 
sorry

end sarah_age_is_26_l202_202546


namespace degenerate_ellipse_single_point_l202_202462

theorem degenerate_ellipse_single_point (c : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + c = 0 → (x = -1 ∧ y = 6)) ↔ c = -39 :=
by
  sorry

end degenerate_ellipse_single_point_l202_202462


namespace right_angle_locus_l202_202186

noncomputable def P (x y : ℝ) : Prop :=
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16

theorem right_angle_locus (x y : ℝ) : P x y → x^2 + y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2 :=
by
  sorry

end right_angle_locus_l202_202186


namespace max_area_rectangle_l202_202561

theorem max_area_rectangle (P : ℝ) (hP : P = 60) (a b : ℝ) (h1 : b = 3 * a) (h2 : 2 * a + 2 * b = P) : a * b = 168.75 :=
by
  sorry

end max_area_rectangle_l202_202561


namespace total_distance_hiked_l202_202210

def distance_car_to_stream : ℝ := 0.2
def distance_stream_to_meadow : ℝ := 0.4
def distance_meadow_to_campsite : ℝ := 0.1

theorem total_distance_hiked : 
  distance_car_to_stream + distance_stream_to_meadow + distance_meadow_to_campsite = 0.7 := by
  sorry

end total_distance_hiked_l202_202210


namespace sides_of_polygon_l202_202120

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l202_202120


namespace count_primes_between_71_and_95_l202_202783

theorem count_primes_between_71_and_95 : 
  let primes := [71, 73, 79, 83, 89, 97] in
  let filtered_primes := primes.filter (λ p, 71 < p ∧ p < 95) in
  filtered_primes.length = 5 := 
by 
  sorry

end count_primes_between_71_and_95_l202_202783


namespace comb_eq_l202_202295

theorem comb_eq {n : ℕ} (h : Nat.choose 18 n = Nat.choose 18 2) : n = 2 ∨ n = 16 :=
by
  sorry

end comb_eq_l202_202295


namespace solution_set_linear_inequalities_l202_202900

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202900


namespace cost_for_paving_is_486_l202_202447

-- Definitions and conditions
def ratio_longer_side : ℝ := 4
def ratio_shorter_side : ℝ := 3
def diagonal : ℝ := 45
def cost_per_sqm : ℝ := 0.5 -- converting pence to pounds

-- Mathematical formulation
def longer_side (x : ℝ) : ℝ := ratio_longer_side * x
def shorter_side (x : ℝ) : ℝ := ratio_shorter_side * x
def area_of_rectangle (l w : ℝ) : ℝ := l * w
def cost_paving (area : ℝ) (cost_per_sqm : ℝ) : ℝ := area * cost_per_sqm

-- Main problem: given the conditions, prove that the cost is £486.
theorem cost_for_paving_is_486 (x : ℝ) 
  (h1 : (ratio_longer_side^2 + ratio_shorter_side^2) * x^2 = diagonal^2) :
  cost_paving (area_of_rectangle (longer_side x) (shorter_side x)) cost_per_sqm = 486 :=
by
  sorry

end cost_for_paving_is_486_l202_202447


namespace abs_inequality_m_eq_neg4_l202_202670

theorem abs_inequality_m_eq_neg4 (m : ℝ) : (∀ x : ℝ, |2 * x - m| ≤ |3 * x + 6|) ↔ (m = -4) :=
by
  sorry

end abs_inequality_m_eq_neg4_l202_202670


namespace sides_of_polygon_l202_202117

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l202_202117


namespace students_on_bus_after_stops_l202_202574

-- Definitions
def initial_students : ℕ := 10
def first_stop_off : ℕ := 3
def first_stop_on : ℕ := 2
def second_stop_off : ℕ := 1
def second_stop_on : ℕ := 4
def third_stop_off : ℕ := 2
def third_stop_on : ℕ := 3

-- Theorem statement
theorem students_on_bus_after_stops :
  let after_first_stop := initial_students - first_stop_off + first_stop_on
  let after_second_stop := after_first_stop - second_stop_off + second_stop_on
  let after_third_stop := after_second_stop - third_stop_off + third_stop_on
  after_third_stop = 13 := 
by
  sorry

end students_on_bus_after_stops_l202_202574


namespace line_intersects_semicircle_at_two_points_l202_202776

theorem line_intersects_semicircle_at_two_points
  (m : ℝ) :
  (3 ≤ m ∧ m < 3 * Real.sqrt 2) ↔ 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ (y₁ = -x₁ + m ∧ y₁ = Real.sqrt (9 - x₁^2)) ∧ (y₂ = -x₂ + m ∧ y₂ = Real.sqrt (9 - x₂^2))) :=
by
  -- The proof goes here
  sorry

end line_intersects_semicircle_at_two_points_l202_202776


namespace race_distance_l202_202382

/-- Given that Sasha, Lesha, and Kolya start a 100m race simultaneously and run at constant velocities,
when Sasha finishes, Lesha is 10m behind, and when Lesha finishes, Kolya is 10m behind.
Prove that the distance between Sasha and Kolya when Sasha finishes is 19 meters. -/
theorem race_distance
    (v_S v_L v_K : ℝ)
    (h1 : 100 / v_S - 100 / v_L = 10 / v_L)
    (h2 : 100 / v_L - 100 / v_K = 10 / v_K) :
    100 - 81 = 19 :=
by
  sorry

end race_distance_l202_202382


namespace negation_of_proposition_divisible_by_2_is_not_even_l202_202418

theorem negation_of_proposition_divisible_by_2_is_not_even :
  (¬ ∀ n : ℕ, n % 2 = 0 → (n % 2 = 0 → n % 2 = 0))
  ↔ ∃ n : ℕ, n % 2 = 0 ∧ n % 2 ≠ 0 := 
  by
    sorry

end negation_of_proposition_divisible_by_2_is_not_even_l202_202418


namespace polygon_sides_l202_202108

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202108


namespace largest_digit_not_in_odd_units_digits_l202_202257

-- Defining the sets of digits
def odd_units_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_units_digits : Set ℕ := {0, 2, 4, 6, 8}

-- Statement to prove
theorem largest_digit_not_in_odd_units_digits : 
  ∀ n ∈ even_units_digits, n ≤ 8 ∧ (∀ d ∈ odd_units_digits, d < n) → n = 8 :=
by
  sorry

end largest_digit_not_in_odd_units_digits_l202_202257


namespace train_cross_pole_time_l202_202616

def speed_kmh := 90 -- speed of the train in km/hr
def length_m := 375 -- length of the train in meters

/-- Convert speed from km/hr to m/s -/
def convert_speed (v_kmh : ℕ) : ℕ := v_kmh * 1000 / 3600

/-- Calculate the time it takes for the train to cross the pole -/
def time_to_cross_pole (length_m : ℕ) (speed_m_s : ℕ) : ℕ := length_m / speed_m_s

theorem train_cross_pole_time :
  time_to_cross_pole length_m (convert_speed speed_kmh) = 15 :=
by
  sorry

end train_cross_pole_time_l202_202616


namespace conic_section_pair_of_lines_l202_202846

theorem conic_section_pair_of_lines : 
  (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = 0 → (2 * x - 3 * y = 0 ∨ 2 * x + 3 * y = 0)) :=
by
  sorry

end conic_section_pair_of_lines_l202_202846


namespace major_axis_length_proof_l202_202276

-- Define the conditions
def radius : ℝ := 3
def minor_axis_length : ℝ := 2 * radius
def major_axis_length : ℝ := minor_axis_length + 0.75 * minor_axis_length

-- State the proof problem
theorem major_axis_length_proof : major_axis_length = 10.5 := 
by
  -- Proof goes here
  sorry

end major_axis_length_proof_l202_202276


namespace Sarah_copies_l202_202233

theorem Sarah_copies : 
  ∀ (copies_per_person number_of_people pages_per_contract : ℕ),
    copies_per_person = 2 →
    number_of_people = 9 →
    pages_per_contract = 20 →
    (copies_per_person * number_of_people * pages_per_contract) = 360 := 
by
  intros copies_per_person number_of_people pages_per_contract h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end Sarah_copies_l202_202233


namespace a_2018_mod_49_l202_202222

def a (n : ℕ) : ℕ := 6^n + 8^n

theorem a_2018_mod_49 : (a 2018) % 49 = 0 := by
  sorry

end a_2018_mod_49_l202_202222


namespace traveler_journey_possible_l202_202454

structure Archipelago (Island : Type) :=
  (n : ℕ)
  (fare : Island → Island → ℝ)
  (unique_ferry : ∀ i j : Island, i ≠ j → fare i j ≠ fare j i)
  (distinct_fares : ∀ i j k l: Island, i ≠ j ∧ k ≠ l → fare i j ≠ fare k l)
  (connected : ∀ i j : Island, i ≠ j → fare i j = fare j i)

theorem traveler_journey_possible {Island : Type} (arch : Archipelago Island) :
  ∃ (t : Island) (seq : List (Island × Island)), -- there exists a starting island and a sequence of journeys
    seq.length = arch.n - 1 ∧                   -- length of the sequence is n-1
    (∀ i j, (i, j) ∈ seq → j ≠ i ∧ arch.fare i j < arch.fare j i) := -- fare decreases with each journey
sorry

end traveler_journey_possible_l202_202454


namespace example_problem_l202_202590

variable (a b c d : ℝ)

theorem example_problem :
  (a + (b + c - d) = a + b + c - d) ∧
  (a - (b - c + d) = a - b + c - d) ∧
  (a - b - (c - d) ≠ a - b - c - d) ∧
  (a + b - (-c - d) = a + b + c + d) :=
by {
  sorry
}

end example_problem_l202_202590


namespace pies_sold_in_week_l202_202604

def daily_pies : ℕ := 8
def days_in_week : ℕ := 7

theorem pies_sold_in_week : daily_pies * days_in_week = 56 := by
  sorry

end pies_sold_in_week_l202_202604


namespace number_of_blocks_l202_202340

theorem number_of_blocks (total_amount : ℕ) (gift_worth : ℕ) (workers_per_block : ℕ) (h1 : total_amount = 4000) (h2 : gift_worth = 4) (h3 : workers_per_block = 100) :
  (total_amount / gift_worth) / workers_per_block = 10 :=
by
-- This part will be proven later, hence using sorry for now
sorry

end number_of_blocks_l202_202340


namespace usual_time_is_25_l202_202728

-- Definitions 
variables {S T : ℝ} (h1 : S * T = 5 / 4 * S * (T - 5))

-- Theorem statement
theorem usual_time_is_25 (h : S * T = 5 / 4 * S * (T - 5)) : T = 25 :=
by 
-- Using the assumption h, we'll derive that T = 25
sorry

end usual_time_is_25_l202_202728


namespace saved_per_bagel_l202_202078

-- Definitions of the conditions
def bagel_cost_each : ℝ := 3.50
def dozen_cost : ℝ := 38
def bakers_dozen : ℕ := 13
def discount : ℝ := 0.05

-- The conjecture we need to prove
theorem saved_per_bagel : 
  let total_cost_without_discount := dozen_cost + bagel_cost_each
  let discount_amount := discount * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount_amount
  let cost_per_bagel_without_discount := dozen_cost / 12
  let cost_per_bagel_with_discount := total_cost_with_discount / bakers_dozen
  let savings_per_bagel := cost_per_bagel_without_discount - cost_per_bagel_with_discount
  let savings_in_cents := savings_per_bagel * 100
  savings_in_cents = 13.36 :=
by
  -- Placeholder for the actual proof
  sorry

end saved_per_bagel_l202_202078


namespace expected_value_dodecahedral_die_l202_202586

noncomputable def dodecahedral_expected_value : ℝ :=
  let outcomes := (list.range (12 + 1)).tail -- List of numbers from 1 to 12
  let n_faces := 12
  let probability := 1 / (n_faces : ℝ)
  let expected_value := probability * outcomes.sum / n_faces
  expected_value

theorem expected_value_dodecahedral_die :
  dodecahedral_expected_value = 6.5 :=
by {
  unfold dodecahedral_expected_value,
  simp,
  sorry,
}

end expected_value_dodecahedral_die_l202_202586


namespace zhiqiang_series_l202_202262

theorem zhiqiang_series (a b : ℝ) (n : ℕ) (n_pos : 0 < n) (h : a * b = 1) (h₀ : b ≠ 1):
  (1 + a^n) / (1 + b^n) = ((1 + a) / (1 + b)) ^ n :=
by
  sorry

end zhiqiang_series_l202_202262


namespace profit_percentage_l202_202620

theorem profit_percentage (cost_price selling_price marked_price : ℝ)
  (h1 : cost_price = 47.50)
  (h2 : selling_price = 0.90 * marked_price)
  (h3 : selling_price = 65.97) :
  ((selling_price - cost_price) / cost_price) * 100 = 38.88 := 
by
  sorry

end profit_percentage_l202_202620


namespace minimum_value_expression_l202_202176

theorem minimum_value_expression (x y : ℝ) : ∃ (m : ℝ), ∀ x y : ℝ, x^2 + 3 * x * y + y^2 ≥ m ∧ m = 0 :=
by
  use 0
  sorry

end minimum_value_expression_l202_202176


namespace distance_between_A_and_B_l202_202592

theorem distance_between_A_and_B 
  (v_pas0 v_freight0 : ℝ) -- original speeds of passenger and freight train
  (t_freight : ℝ) -- time taken by freight train
  (d : ℝ) -- distance sought
  (h1 : t_freight = d / v_freight0) 
  (h2 : d + 288 = v_pas0 * t_freight) 
  (h3 : (d / (v_freight0 + 10)) + 2.4 = d / (v_pas0 + 10))
  : d = 360 := 
sorry

end distance_between_A_and_B_l202_202592


namespace not_periodic_l202_202371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + Real.sin (a * x)

theorem not_periodic {a : ℝ} (ha : Irrational a) : ¬ ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f a (x + T) = f a x :=
  sorry

end not_periodic_l202_202371


namespace sasha_kolya_distance_l202_202386

theorem sasha_kolya_distance
  (v_S v_L v_K : ℝ) 
  (h1 : Lesha_dist = 100 - 10) 
  (h2 : Kolya_dist = 100 - 10) 
  (h3 : v_L = (90 / 100) * v_S) 
  (h4 : v_K = (90 / 100) * v_L) 
  : v_S * (100/v_S - 10/v_S) = 19 :=
by
  sorry

end sasha_kolya_distance_l202_202386


namespace find_n_l202_202060

theorem find_n (n : ℤ) (h₁ : 50 ≤ n ∧ n ≤ 120)
               (h₂ : n % 8 = 0)
               (h₃ : n % 12 = 4)
               (h₄ : n % 7 = 4) : 
  n = 88 :=
sorry

end find_n_l202_202060


namespace linear_inequalities_solution_l202_202867

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202867


namespace simplify_expr_l202_202408

theorem simplify_expr (x y : ℝ) : 
  (3 * x - 2 * y - 4) * (x + y + 5) - (x + 2 * y + 5) * (3 * x - y - 1) = -4 * x * y - 3 * x - 7 * y - 15 := 
by 
  sorry

end simplify_expr_l202_202408


namespace inequality_problem_l202_202837

theorem inequality_problem (x : ℝ) (hx : 0 < x) : 
  1 + x ^ 2018 ≥ (2 * x) ^ 2017 / (1 + x) ^ 2016 := 
by
  sorry

end inequality_problem_l202_202837


namespace num_cubes_with_more_than_one_blue_face_l202_202153

-- Define the parameters of the problem
def block_length : ℕ := 5
def block_width : ℕ := 3
def block_height : ℕ := 1

def total_cubes : ℕ := 15
def corners : ℕ := 4
def edges : ℕ := 6
def middles : ℕ := 5

-- Define the condition that the total number of cubes painted on more than one face.
def cubes_more_than_one_blue_face : ℕ := corners + edges

-- Prove that the number of cubes painted on more than one face is 10
theorem num_cubes_with_more_than_one_blue_face :
  cubes_more_than_one_blue_face = 10 :=
by
  show (4 + 6) = 10
  sorry

end num_cubes_with_more_than_one_blue_face_l202_202153


namespace triangle_angle_contradiction_l202_202828

theorem triangle_angle_contradiction (A B C : ℝ) (hA : 60 < A) (hB : 60 < B) (hC : 60 < C) (h_sum : A + B + C = 180) : false :=
by {
  -- This would be the proof part, which we don't need to detail according to the instructions.
  sorry
}

end triangle_angle_contradiction_l202_202828


namespace smallest_whole_number_l202_202007

theorem smallest_whole_number (a b c d : ℤ)
  (h₁ : a = 3 + 1 / 3)
  (h₂ : b = 4 + 1 / 4)
  (h₃ : c = 5 + 1 / 6)
  (h₄ : d = 6 + 1 / 8)
  (h₅ : a + b + c + d - 2 > 16)
  (h₆ : a + b + c + d - 2 < 17) :
  17 > 16 + (a + b + c + d - 18) - 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 8 :=
  sorry

end smallest_whole_number_l202_202007


namespace jason_cards_l202_202047

theorem jason_cards :
  (initial_cards - bought_cards = remaining_cards) →
  initial_cards = 676 →
  bought_cards = 224 →
  remaining_cards = 452 :=
by
  intros h1 h2 h3
  sorry

end jason_cards_l202_202047


namespace distance_between_sasha_and_kolya_when_sasha_finished_l202_202389

-- Definitions based on the problem conditions
def distance_sasha : ℝ := 100
def distance_lesha_when_sasha_finished : ℝ := 90
def distance_kolya_when_lesha_finished : ℝ := 90

def velocity_lesha (v_s : ℝ) : ℝ := 0.9 * v_s
def velocity_kolya (v_s : ℝ) : ℝ := 0.81 * v_s

-- Theorem statement
theorem distance_between_sasha_and_kolya_when_sasha_finished (v_s : ℝ) :
  distance_sasha - (velocity_kolya v_s * (distance_sasha / v_s)) = 19 :=
  by sorry

end distance_between_sasha_and_kolya_when_sasha_finished_l202_202389


namespace find_common_difference_l202_202512

variable {a : ℕ → ℤ}  -- Define the arithmetic sequence as a function from natural numbers to integers
variable (d : ℤ)      -- Define the common difference

-- Assume the conditions given in the problem
axiom h1 : a 2 = 14
axiom h2 : a 5 = 5

theorem find_common_difference (n : ℕ) : d = -3 :=
by {
  -- This part will be filled in by the actual proof
  sorry
}

end find_common_difference_l202_202512


namespace solution_set_system_of_inequalities_l202_202924

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202924


namespace sum_of_powers_l202_202157

theorem sum_of_powers : 5^5 + 5^5 + 5^5 + 5^5 = 4 * 5^5 :=
by
  sorry

end sum_of_powers_l202_202157


namespace correct_operation_l202_202436

theorem correct_operation (a b x y m : Real) :
  (¬((a^2 * b)^2 = a^2 * b^2)) ∧
  (¬(a^6 / a^2 = a^3)) ∧
  (¬((x + y)^2 = x^2 + y^2)) ∧
  ((-m)^7 / (-m)^2 = -m^5) :=
by
  sorry

end correct_operation_l202_202436


namespace smallest_possible_value_of_N_l202_202570

-- Conditions definition:
variable (a b c d : ℕ)
variable (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
variable (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
variable (gcd_ab : Int.gcd a b = 1)
variable (gcd_ac : Int.gcd a c = 2)
variable (gcd_ad : Int.gcd a d = 4)
variable (gcd_bc : Int.gcd b c = 5)
variable (gcd_bd : Int.gcd b d = 3)
variable (gcd_cd : Int.gcd c d = N)
variable (hN : N > 5)

-- Statement to prove:
theorem smallest_possible_value_of_N : N = 14 := sorry

end smallest_possible_value_of_N_l202_202570


namespace problem_statement_l202_202650

noncomputable def f (x : ℝ) : ℝ :=
  1 - x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

theorem problem_statement : f (1 / 2) + f (-1 / 2) = 2 := sorry

end problem_statement_l202_202650


namespace find_a_plus_b_l202_202497

def smallest_two_digit_multiple_of_five : ℕ := 10
def smallest_three_digit_multiple_of_seven : ℕ := 105

theorem find_a_plus_b :
  let a := smallest_two_digit_multiple_of_five
  let b := smallest_three_digit_multiple_of_seven
  a + b = 115 := by
  sorry

end find_a_plus_b_l202_202497


namespace symmetric_circle_eq_l202_202015

theorem symmetric_circle_eq (x y : ℝ) :
  (x + 1)^2 + (y - 1)^2 = 1 → x - y = 1 → (x - 2)^2 + (y + 2)^2 = 1 :=
by
  sorry

end symmetric_circle_eq_l202_202015


namespace polygon_sides_l202_202090

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l202_202090


namespace evaluate_expression_l202_202630

theorem evaluate_expression:
  let a := 3
  let b := 2
  (a^b)^a - (b^a)^b = 665 :=
by
  sorry

end evaluate_expression_l202_202630


namespace largest_divisor_of_expression_l202_202766

theorem largest_divisor_of_expression :
  ∃ k : ℕ, (∀ m : ℕ, (m > k → m ∣ (1991 ^ k * 1990 ^ (1991 ^ 1992) + 1992 ^ (1991 ^ 1990)) = false))
  ∧ k = 1991 := by
sorry

end largest_divisor_of_expression_l202_202766


namespace submarine_rise_l202_202743

theorem submarine_rise (initial_depth final_depth : ℤ) (h_initial : initial_depth = -27) (h_final : final_depth = -18) :
  final_depth - initial_depth = 9 :=
by
  rw [h_initial, h_final]
  norm_num 

end submarine_rise_l202_202743


namespace train_speed_l202_202615

theorem train_speed
  (length_m : ℝ)
  (time_s : ℝ)
  (h_length : length_m = 280.0224)
  (h_time : time_s = 25.2) :
  (length_m / 1000) / (time_s / 3600) = 40.0032 :=
by
  sorry

end train_speed_l202_202615


namespace polygon_sides_l202_202099

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l202_202099


namespace white_seeds_per_slice_l202_202428

theorem white_seeds_per_slice (W : ℕ) (black_seeds_per_slice : ℕ) (number_of_slices : ℕ) 
(total_seeds : ℕ) (total_black_seeds : ℕ) (total_white_seeds : ℕ) 
(h1 : black_seeds_per_slice = 20)
(h2 : number_of_slices = 40)
(h3 : total_seeds = 1600)
(h4 : total_black_seeds = black_seeds_per_slice * number_of_slices)
(h5 : total_white_seeds = total_seeds - total_black_seeds)
(h6 : W = total_white_seeds / number_of_slices) :
W = 20 :=
by
  sorry

end white_seeds_per_slice_l202_202428


namespace range_of_function_l202_202373

theorem range_of_function (x y z : ℝ)
  (h : x^2 + y^2 + x - y = 1) :
  ∃ a b : ℝ, (a = (3 * Real.sqrt 6 + Real.sqrt 6) / 2) ∧ (b = (-3 * Real.sqrt 2 + Real.sqrt 6) / 2) ∧
    ∀ f : ℝ, f = (x - 1) * Real.cos z + (y + 1) * Real.sin z →
              b ≤ f ∧ f ≤ a := 
by
  sorry

end range_of_function_l202_202373


namespace edward_initial_amount_l202_202762

-- Defining the conditions
def cost_books : ℕ := 6
def cost_pens : ℕ := 16
def cost_notebook : ℕ := 5
def cost_pencil_case : ℕ := 3
def amount_left : ℕ := 19

-- Mathematical statement to prove
theorem edward_initial_amount : 
  cost_books + cost_pens + cost_notebook + cost_pencil_case + amount_left = 49 :=
by
  sorry

end edward_initial_amount_l202_202762


namespace find_y_l202_202420

theorem find_y (h1 : ∀ (a b : ℤ), a * b = (a - 1) * (b - 1)) (h2 : y * 12 = 110) : y = 11 :=
by
  sorry

end find_y_l202_202420


namespace eq_b_minus_a_l202_202085

   -- Definition for rotating a point counterclockwise by 180° around another point
   def rotate_180 (h k x y : ℝ) : ℝ × ℝ :=
     (2 * h - x, 2 * k - y)

   -- Definition for reflecting a point about the line y = -x
   def reflect_y_eq_neg_x (x y : ℝ) : ℝ × ℝ :=
     (-y, -x)

   -- Given point Q(a, b)
   variables (a b : ℝ)

   -- Image of Q after the transformations
   def Q_transformed :=
     (5, -1)

   -- Image of Q after reflection about y = -x
   def Q_reflected :=
     reflect_y_eq_neg_x (5) (-1)

   -- Image of Q after 180° rotation around (2,3)
   def Q_original :=
     rotate_180 (2) (3) a b

   -- Statement we want to prove:
   theorem eq_b_minus_a : b - a = 6 :=
   by
     -- Calculation steps
     sorry
   
end eq_b_minus_a_l202_202085


namespace range_a_l202_202199

noncomputable def f (x : ℝ) : ℝ := -(1 / 3) * x^3 + x

theorem range_a (a : ℝ) (h1 : a < 1) (h2 : 1 < 10 - a^2) (h3 : f a ≤ f 1) :
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_a_l202_202199


namespace sum_of_ages_in_10_years_l202_202749

-- Define the initial conditions about Ann's and Tom's ages
def AnnCurrentAge : ℕ := 6
def TomCurrentAge : ℕ := 2 * AnnCurrentAge

-- Define their ages 10 years later
def AnnAgeIn10Years : ℕ := AnnCurrentAge + 10
def TomAgeIn10Years : ℕ := TomCurrentAge + 10

-- The proof statement
theorem sum_of_ages_in_10_years : AnnAgeIn10Years + TomAgeIn10Years = 38 := by
  sorry

end sum_of_ages_in_10_years_l202_202749


namespace weather_desire_probability_l202_202236

noncomputable def probability_sunny_days_desired 
: ℚ := 135 / 2048

theorem weather_desire_probability 
(p_rain : ℚ) (n_days : ℕ) (p_sunny : ℚ) 
(one_day_prob : ℚ) (two_days_prob : ℚ) 
(desired_prob : ℚ) : 
  p_rain = 3 / 4 ∧ p_sunny = 1 / 4 ∧ n_days = 5 
  ∧ one_day_prob = 5 * (p_sunny * (p_rain ^ 4))
  ∧ two_days_prob = 10 * ((p_sunny ^ 2) * (p_rain ^ 3))
  ∧ desired_prob = one_day_prob + two_days_prob 
  → desired_prob = probability_sunny_days_desired := 
by 
  intro h 
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h_rest
  cases h_rest with h4 h_rest
  cases h_rest with h5 h_rest
  cases h_rest with h6 _
  -- continue proof, but add sorry as we only need the statement
  sorry

end weather_desire_probability_l202_202236


namespace rectangle_dimension_l202_202017

theorem rectangle_dimension (x : ℝ) (h : (x^2) * (x + 5) = 3 * (2 * (x^2) + 2 * (x + 5))) : x = 3 :=
by
  have eq1 : (x^2) * (x + 5) = x^3 + 5 * x^2 := by ring
  have eq2 : 3 * (2 * (x^2) + 2 * (x + 5)) = 6 * x^2 + 6 * x + 30 := by ring
  rw [eq1, eq2] at h
  sorry  -- Proof details omitted

end rectangle_dimension_l202_202017


namespace solve_system_of_equations_l202_202547

theorem solve_system_of_equations (x y : ℝ) (hx: x > 0) (hy: y > 0) :
  x * y = 500 ∧ x ^ (Real.log y / Real.log 10) = 25 → (x = 100 ∧ y = 5) ∨ (x = 5 ∧ y = 100) := by
  sorry

end solve_system_of_equations_l202_202547


namespace average_of_remaining_two_numbers_l202_202549

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℚ)
  (h1 : (a + b + c + d + e + f) / 6 = 6.40)
  (h2 : (a + b) / 2 = 6.2)
  (h3 : (c + d) / 2 = 6.1) : 
  ((e + f) / 2 = 6.9) :=
by
  sorry

end average_of_remaining_two_numbers_l202_202549


namespace combined_rocket_height_l202_202353

variable (h1 : ℕ) (h2 : ℕ)

-- Given conditions
def first_rocket_height : ℕ := 500
def second_rocket_height : ℕ := first_rocket_height * 2

-- Prove that the combined height is 1500 ft
theorem combined_rocket_height : first_rocket_height + second_rocket_height = 1500 := by
  sorry

end combined_rocket_height_l202_202353


namespace reflect_and_shift_l202_202328

def f : ℝ → ℝ := sorry  -- Assume f is some function from ℝ to ℝ

def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (6 - x)

theorem reflect_and_shift (f : ℝ → ℝ) (x : ℝ) : h f x = f (6 - x) :=
by
  -- provide the proof here
  sorry

end reflect_and_shift_l202_202328


namespace scientific_notation_correct_l202_202073

def million : ℝ := 10^6
def num : ℝ := 1.06
def num_in_million : ℝ := num * million
def scientific_notation : ℝ := 1.06 * 10^6

theorem scientific_notation_correct : num_in_million = scientific_notation :=
by 
  -- The proof is skipped, indicated by sorry
  sorry

end scientific_notation_correct_l202_202073


namespace distance_between_sasha_and_kolya_is_19_meters_l202_202405

theorem distance_between_sasha_and_kolya_is_19_meters
  (v_S v_L v_K : ℝ)
  (h1 : v_L = 0.9 * v_S)
  (h2 : v_K = 0.81 * v_S)
  (h3 : ∀ t_S : ℝ, t_S = 100 / v_S) :
  (∀ t_S : ℝ, 100 - v_K * t_S = 19) :=
by
  intros t_S
  have vL_defined : v_L = 0.9 * v_S := h1
  have vK_defined : v_K = 0.81 * v_S := h2
  have time_S : t_S = 100 / v_S := h3 t_S
  sorry

end distance_between_sasha_and_kolya_is_19_meters_l202_202405


namespace kindergarten_children_count_l202_202802

theorem kindergarten_children_count (D B C : ℕ) (hD : D = 18) (hB : B = 6) (hC : C + B = 12) : D + C + B = 30 :=
by
  sorry

end kindergarten_children_count_l202_202802


namespace fraction_difference_l202_202519

variable (a b : ℝ)

theorem fraction_difference (h : 1/a - 1/b = 1/(a + b)) : 
  1/a^2 - 1/b^2 = 1/(a * b) := 
  sorry

end fraction_difference_l202_202519


namespace polygon_sides_l202_202086

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l202_202086


namespace circles_disjoint_l202_202654

theorem circles_disjoint :
  ∀ (x y u v : ℝ),
  (x^2 + y^2 = 1) →
  ((u-2)^2 + (v+2)^2 = 1) →
  (2^2 + (-2)^2) > (1 + 1)^2 :=
by sorry

end circles_disjoint_l202_202654


namespace smallest_GCD_value_l202_202566

theorem smallest_GCD_value (a b c d N : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : N > 5)
    (hc1 : Nat.gcd a b = 1 ∨ Nat.gcd a c = 1 ∨ Nat.gcd a d = 1 ∨ Nat.gcd b c = 1 ∨ Nat.gcd b d = 1 ∨ Nat.gcd c d = 1)
    (hc2 : Nat.gcd a b = 2 ∨ Nat.gcd a c = 2 ∨ Nat.gcd a d = 2 ∨ Nat.gcd b c = 2 ∨ Nat.gcd b d = 2 ∨ Nat.gcd c d = 2)
    (hc3 : Nat.gcd a b = 3 ∨ Nat.gcd a c = 3 ∨ Nat.gcd a d = 3 ∨ Nat.gcd b c = 3 ∨ Nat.gcd b d = 3 ∨ Nat.gcd c d = 3)
    (hc4 : Nat.gcd a b = 4 ∨ Nat.gcd a c = 4 ∨ Nat.gcd a d = 4 ∨ Nat.gcd b c = 4 ∨ Nat.gcd b d = 4 ∨ Nat.gcd c d = 4)
    (hc5 : Nat.gcd a b = 5 ∨ Nat.gcd a c = 5 ∨ Nat.gcd a d = 5 ∨ Nat.gcd b c = 5 ∨ Nat.gcd b d = 5 ∨ Nat.gcd c d = 5)
    (hcN : Nat.gcd a b = N ∨ Nat.gcd a c = N ∨ Nat.gcd a d = N ∨ Nat.gcd b c = N ∨ Nat.gcd b d = N ∨ Nat.gcd c d = N):
    N = 14 :=
sorry

end smallest_GCD_value_l202_202566


namespace problem1_problem2_l202_202594

open Set

variable {x y z a b : ℝ}

-- Problem 1: Prove the inequality
theorem problem1 (x y z : ℝ) : 
  5 * x^2 + y^2 + z^2 ≥ 2 * x * y + 4 * x + 2 * z - 2 :=
by
  sorry

-- Problem 2: Prove the range of 10a - 5b is [−1, 20]
theorem problem2 (a b : ℝ) 
  (h1 : 1 ≤ 2 * a + b ∧ 2 * a + b ≤ 4)
  (h2 : -1 ≤ a - 2 * b ∧ a - 2 * b ≤ 2) : 
  -1 ≤ 10 * a - 5 * b ∧ 10 * a - 5 * b ≤ 20 :=
by
  sorry

end problem1_problem2_l202_202594


namespace last_student_remains_l202_202530

theorem last_student_remains (n : ℕ) (h : n = 100) :
  let students := list.range (n + 1) -- Students numbered from 1 to 100
  let remaining := whittled_down students -- Process of removing every second student
  remaining = [73] := 
sorry

end last_student_remains_l202_202530


namespace equation1_no_solution_equation2_solution_l202_202698

/-- Prove that the equation (4-x)/(x-3) + 1/(3-x) = 1 has no solution. -/
theorem equation1_no_solution (x : ℝ) : x ≠ 3 → ¬ (4 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by intro hx; sorry

/-- Prove that the equation (x+1)/(x-1) - 6/(x^2-1) = 1 has solution x = 2. -/
theorem equation2_solution (x : ℝ) : x = 2 ↔ (x + 1) / (x - 1) - 6 / (x^2 - 1) = 1 :=
by sorry

end equation1_no_solution_equation2_solution_l202_202698


namespace trig_identity_l202_202481

theorem trig_identity (α : ℝ) (h : Real.tan α = 4) : (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := by
  sorry

end trig_identity_l202_202481


namespace quadratic_no_real_roots_iff_l202_202479

theorem quadratic_no_real_roots_iff (m : ℝ) : (∀ x : ℝ, x^2 + 3 * x + m ≠ 0) ↔ m > 9 / 4 :=
by
  sorry

end quadratic_no_real_roots_iff_l202_202479


namespace part1_eq_of_line_l_part2_eq_of_line_l1_l202_202648

def intersection_point (m n : ℝ × ℝ × ℝ) : ℝ × ℝ := sorry

def line_through_point_eq_dists (P A B : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry
def line_area_triangle (P : ℝ × ℝ) (triangle_area : ℝ) : ℝ × ℝ × ℝ := sorry

-- Conditions defined:
def m : ℝ × ℝ × ℝ := (2, -1, -3)
def n : ℝ × ℝ × ℝ := (1, 1, -3)
def P : ℝ × ℝ := intersection_point m n
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 2)
def triangle_area : ℝ := 4

-- Questions translated into Lean 4 statements:
theorem part1_eq_of_line_l : ∃ l : ℝ × ℝ × ℝ, 
  (l = line_through_point_eq_dists P A B) := sorry

theorem part2_eq_of_line_l1 : ∃ l1 : ℝ × ℝ × ℝ,
  (l1 = line_area_triangle P triangle_area) := sorry

end part1_eq_of_line_l_part2_eq_of_line_l1_l202_202648


namespace angle_quadrant_l202_202774

theorem angle_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  0 < (π - α) ∧ (π - α) < π  :=
by
  sorry

end angle_quadrant_l202_202774


namespace xy_value_l202_202801

theorem xy_value (x y : ℝ) (h₁ : x + y = 2) (h₂ : x^2 * y^3 + y^2 * x^3 = 32) :
  x * y = 2^(5/3) :=
by
  sorry

end xy_value_l202_202801


namespace moles_of_H2O_combined_l202_202633

theorem moles_of_H2O_combined (mole_NH4Cl mole_NH4OH : ℕ) (reaction : mole_NH4Cl = 1 ∧ mole_NH4OH = 1) : 
  ∃ mole_H2O : ℕ, mole_H2O = 1 :=
by
  sorry

end moles_of_H2O_combined_l202_202633


namespace monica_book_ratio_theorem_l202_202683

/-
Given:
1. Monica read 16 books last year.
2. This year, she read some multiple of the number of books she read last year.
3. Next year, she will read 69 books.
4. Next year, she wants to read 5 more than twice the number of books she read this year.

Prove:
The ratio of the number of books she read this year to the number of books she read last year is 2.
-/

noncomputable def monica_book_ratio_proof : Prop :=
  let last_year_books := 16
  let next_year_books := 69
  ∃ (x : ℕ), (∃ (n : ℕ), x = last_year_books * n) ∧ (2 * x + 5 = next_year_books) ∧ (x / last_year_books = 2)

theorem monica_book_ratio_theorem : monica_book_ratio_proof :=
  by
    sorry

end monica_book_ratio_theorem_l202_202683


namespace solution_set_l202_202869

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202869


namespace sarah_problem_l202_202232

theorem sarah_problem (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 100 ≤ y ∧ y ≤ 999) 
  (h : 1000 * x + y = 11 * x * y) : x + y = 110 :=
sorry

end sarah_problem_l202_202232


namespace positive_integers_satisfy_condition_l202_202552

theorem positive_integers_satisfy_condition :
  ∃! n : ℕ, (n > 0 ∧ 30 - 6 * n > 18) :=
by
  sorry

end positive_integers_satisfy_condition_l202_202552


namespace find_angle_A_find_max_area_l202_202039

/-
  In an acute triangle ΔABC, with sides opposite angles A, B, and C being a, b, and c respectively,
  it is given that b/2 is the arithmetic mean of 2a sin A cos C and c sin 2A.
  Prove that the measure of angle A is π/6.
-/
theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h1 : sin A = 1 / 2) (h2 : 0 < A) (h3 : A < π / 2)
(h4: b / 2 = (2 * a * sin A * cos C + c * sin (2 * A)) / 2) : A = π / 6 := 
sorry

/-
  If a = 2, prove that the maximum area of the triangle ΔABC is 2+√3.
-/
theorem find_max_area (a b c : ℝ) (A B C : ℝ)
(h1: a = 2)
(h2: ∀ (b c : ℝ), b^2 + c^2 - 2*b*c*cos A ≥ b^2 + c^2 - sqrt 3 * b * c)
(h3: S = 1/2 * b * c * sin A)
: S ≤ 2 + sqrt 3 :=
sorry

end find_angle_A_find_max_area_l202_202039


namespace find_inverse_l202_202662

noncomputable def f (x : ℝ) := (x^7 - 1) / 5

theorem find_inverse :
  (f⁻¹ (-1 / 80) = (15 / 16)^(1 / 7)) :=
sorry

end find_inverse_l202_202662


namespace solution_set_linear_inequalities_l202_202939

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202939


namespace sheets_in_stack_l202_202140

theorem sheets_in_stack (sheets : ℕ) (thickness : ℝ) (h1 : sheets = 400) (h2 : thickness = 4) :
    let thickness_per_sheet := thickness / sheets
    let stack_height := 6
    (stack_height / thickness_per_sheet = 600) :=
by
  sorry

end sheets_in_stack_l202_202140


namespace solution_set_linear_inequalities_l202_202937

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202937


namespace range_of_a_l202_202503

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a) ↔ (a < 1 ∨ a > 3) := 
sorry

end range_of_a_l202_202503


namespace floor_sqrt_80_l202_202470

theorem floor_sqrt_80 : ∃ (x : ℤ), 8^2 = 64 ∧ 9^2 = 81 ∧ 8 < real.sqrt 80 ∧ real.sqrt 80 < 9 ∧ int.floor (real.sqrt 80) = x ∧ x = 8 :=
by
  sorry

end floor_sqrt_80_l202_202470


namespace find_98_real_coins_l202_202423

-- We will define the conditions as variables and state the goal as a theorem.

-- Variables:
variable (Coin : Type) -- Type representing coins
variable [Fintype Coin] -- 100 coins in total, therefore a Finite type
variable (number_of_coins : ℕ) (h100 : number_of_coins = 100)
variable (real : Coin → Prop) -- Predicate indicating if the coin is real
variable (lighter_fake : Coin → Prop) -- Predicate indicating if the coin is the lighter fake
variable (balance_scale : Coin → Coin → Prop) -- Balance scale result

-- Conditions:
axiom real_coins_count : ∃ R : Finset Coin, R.card = 99 ∧ (∀ c ∈ R, real c)
axiom fake_coin_exists : ∃ F : Coin, lighter_fake F ∧ ¬ real F

theorem find_98_real_coins : ∃ S : Finset Coin, S.card = 98 ∧ (∀ c ∈ S, real c) := by
  sorry

end find_98_real_coins_l202_202423


namespace prime_square_count_l202_202788

-- Define the set of natural numbers whose squares lie between 5000 and 9000
def within_square_range (n : ℕ) : Prop := 5000 < n * n ∧ n * n < 9000

-- Define the predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the set of prime numbers whose squares lie within the range
def primes_within_range : Finset ℕ := (Finset.filter is_prime (Finset.Ico 71 95))

-- The main statement: the number of prime numbers whose squares are between 5000 and 9000
theorem prime_square_count : (primes_within_range.filter within_square_range).card = 6 :=
sorry

end prime_square_count_l202_202788


namespace solution_set_of_inequalities_l202_202975

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202975


namespace non_associative_products_l202_202223

-- Definitions
def h (n : ℕ) : ℕ := if n = 1 then 1 else if n = 2 then 2 else sorry

-- Factorial and Catalan number are already defined in Mathlib

theorem non_associative_products (n : ℕ) (hn : h n) : 
  h n = (2 * n - 2).fact / (n - 1).fact ∧ h n = n.fact * Catalan n :=
by {
  sorry
}

end non_associative_products_l202_202223


namespace solution_set_of_linear_inequalities_l202_202920

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202920


namespace least_prime_P_with_integer_roots_of_quadratic_l202_202027

theorem least_prime_P_with_integer_roots_of_quadratic :
  ∃ P : ℕ, P.Prime ∧ (∃ m : ℤ,  m^2 = 12 * P + 60) ∧ P = 7 :=
by
  sorry

end least_prime_P_with_integer_roots_of_quadratic_l202_202027


namespace solve_inequalities_l202_202947

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202947


namespace jessica_age_l202_202048

theorem jessica_age 
  (j g : ℚ)
  (h1 : g = 15 * j) 
  (h2 : g - j = 60) : 
  j = 30 / 7 :=
by
  sorry

end jessica_age_l202_202048


namespace pairs_count_1432_1433_l202_202767

def PairsCount (n : ℕ) : ℕ :=
  -- The implementation would count the pairs (x, y) such that |x^2 - y^2| = n
  sorry

-- We write down the theorem that expresses what we need to prove
theorem pairs_count_1432_1433 : PairsCount 1432 = 8 ∧ PairsCount 1433 = 4 := by
  sorry

end pairs_count_1432_1433_l202_202767


namespace necessary_but_not_sufficient_condition_l202_202208

variable {a : ℕ → ℤ}

noncomputable def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
∀ (m n k : ℕ), a m * a k = a n * a (m + k - n)

noncomputable def is_root_of_quadratic (x y : ℤ) : Prop :=
x^2 + 3*x + 1 = 0 ∧ y^2 + 3*y + 1 = 0

theorem necessary_but_not_sufficient_condition 
  (a : ℕ → ℤ)
  (hgeo : is_geometric_sequence a)
  (hroots : is_root_of_quadratic (a 4) (a 12)) :
  a 8 = -1 ↔ (∃ x y : ℤ, is_root_of_quadratic x y ∧ x + y = -3 ∧ x * y = 1) :=
sorry

end necessary_but_not_sufficient_condition_l202_202208


namespace sum_of_special_multiples_l202_202498

def smallest_two_digit_multiple_of_5 : ℕ := 10
def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_special_multiples :
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 :=
by
  sorry

end sum_of_special_multiples_l202_202498


namespace polygon_sides_l202_202087

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l202_202087


namespace complex_number_quadrant_l202_202642

def i := Complex.I
def z := i * (1 + i)

theorem complex_number_quadrant 
  : z.re < 0 ∧ z.im > 0 := 
by
  sorry

end complex_number_quadrant_l202_202642


namespace diff_of_squares_l202_202625

-- Define constants a and b
def a : ℕ := 65
def b : ℕ := 35

-- State the theorem to be proven using the difference of squares formula
theorem diff_of_squares : a^2 - b^2 = 3000 := by
  have h1 : a + b = 100 := by rfl -- Checking a + b
  have h2 : a - b = 30 := by rfl -- Checking a - b
  have h3 : (a+b)*(a-b) = 3000 := by
    rw [h1, h2] -- Substituting the values
    norm_num -- Simplifying the arithmetics
  exact h3 -- Providing the ultimate result

end diff_of_squares_l202_202625


namespace sufficient_not_necessary_l202_202593

theorem sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (∃ x, 1 / x < 1 ∧ ¬(x > 1)) :=
by
  sorry

end sufficient_not_necessary_l202_202593


namespace solve_inequalities_l202_202950

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202950


namespace parallel_lines_slope_eq_l202_202504

theorem parallel_lines_slope_eq (k : ℝ) : 
  (∀ x : ℝ, k * x - 1 = 3 * x) → k = 3 :=
by sorry

end parallel_lines_slope_eq_l202_202504


namespace ratio_change_factor_is_5_l202_202563

-- Definitions based on problem conditions
def original_bleach : ℕ := 4
def original_detergent : ℕ := 40
def original_water : ℕ := 100

-- Simplified original ratio
def original_bleach_ratio : ℕ := original_bleach / 4
def original_detergent_ratio : ℕ := original_detergent / 4
def original_water_ratio : ℕ := original_water / 4

-- Altered conditions
def altered_detergent : ℕ := 60
def altered_water : ℕ := 300

-- Simplified altered ratio of detergent to water
def altered_detergent_ratio : ℕ := altered_detergent / 60
def altered_water_ratio : ℕ := altered_water / 60

-- Proof that the ratio change factor is 5
theorem ratio_change_factor_is_5 : 
  (original_water_ratio / altered_water_ratio) = 5
  := by
    have original_detergent_ratio : ℕ := 10
    have original_water_ratio : ℕ := 25
    have altered_detergent_ratio : ℕ := 1
    have altered_water_ratio : ℕ := 5
    sorry

end ratio_change_factor_is_5_l202_202563


namespace initial_pills_count_l202_202427

theorem initial_pills_count 
  (pills_taken_first_2_days : ℕ)
  (pills_taken_next_3_days : ℕ)
  (pills_taken_sixth_day : ℕ)
  (pills_left : ℕ)
  (h1 : pills_taken_first_2_days = 2 * 3 * 2)
  (h2 : pills_taken_next_3_days = 1 * 3 * 3)
  (h3 : pills_taken_sixth_day = 2)
  (h4 : pills_left = 27) :
  ∃ initial_pills : ℕ, initial_pills = pills_taken_first_2_days + pills_taken_next_3_days + pills_taken_sixth_day + pills_left :=
by
  sorry

end initial_pills_count_l202_202427


namespace total_area_is_71_l202_202207

-- Define the lengths of the segments
def length_left : ℕ := 7
def length_top : ℕ := 6
def length_middle_1 : ℕ := 2
def length_middle_2 : ℕ := 4
def length_right : ℕ := 1
def length_right_top : ℕ := 5

-- Define the rectangles and their areas
def area_left_rect : ℕ := length_left * length_left
def area_middle_rect : ℕ := length_middle_1 * (length_top - length_left)
def area_right_rect : ℕ := length_middle_2 * length_middle_2

-- Define the total area
def total_area : ℕ := area_left_rect + area_middle_rect + area_right_rect

-- Theorem: The total area of the figure is 71 square units
theorem total_area_is_71 : total_area = 71 := by
  sorry

end total_area_is_71_l202_202207


namespace square_area_l202_202082

noncomputable def side_length1 (x : ℝ) : ℝ := 5 * x - 20
noncomputable def side_length2 (x : ℝ) : ℝ := 25 - 2 * x

theorem square_area (x : ℝ) (h : side_length1 x = side_length2 x) :
  (side_length1 x)^2 = 7225 / 49 :=
by
  sorry

end square_area_l202_202082


namespace soda_cost_l202_202724

theorem soda_cost (x : ℝ) : 
    (1.5 * 35 + x * (87 - 35) = 78.5) → 
    x = 0.5 := 
by 
  intros h
  sorry

end soda_cost_l202_202724


namespace find_length_AB_l202_202203

-- Definitions for the problem conditions.
def angle_B : ℝ := 90
def angle_A : ℝ := 30
def BC : ℝ := 24

-- Main theorem to prove.
theorem find_length_AB (angle_B_eq : angle_B = 90) (angle_A_eq : angle_A = 30) (BC_eq : BC = 24) : 
  ∃ AB : ℝ, AB = 12 := 
by
  sorry

end find_length_AB_l202_202203


namespace yanna_change_l202_202132

theorem yanna_change :
  let shirt_cost := 5
  let sandal_cost := 3
  let num_shirts := 10
  let num_sandals := 3
  let given_amount := 100
  (given_amount - (num_shirts * shirt_cost + num_sandals * sandal_cost)) = 41 :=
by
  sorry

end yanna_change_l202_202132


namespace kyle_and_miles_total_marble_count_l202_202299

noncomputable def kyle_marble_count (F : ℕ) (K : ℕ) : Prop :=
  F = 4 * K

noncomputable def miles_marble_count (F : ℕ) (M : ℕ) : Prop :=
  F = 9 * M

theorem kyle_and_miles_total_marble_count :
  ∀ (F K M : ℕ), F = 36 → kyle_marble_count F K → miles_marble_count F M → K + M = 13 :=
by
  intros F K M hF hK hM
  sorry

end kyle_and_miles_total_marble_count_l202_202299


namespace sarah_boxes_l202_202231

theorem sarah_boxes (b : ℕ) 
  (h1 : ∀ x : ℕ, x = 7) 
  (h2 : 49 = 7 * b) :
  b = 7 :=
sorry

end sarah_boxes_l202_202231


namespace sasha_kolya_distance_l202_202385

theorem sasha_kolya_distance
  (v_S v_L v_K : ℝ) 
  (h1 : Lesha_dist = 100 - 10) 
  (h2 : Kolya_dist = 100 - 10) 
  (h3 : v_L = (90 / 100) * v_S) 
  (h4 : v_K = (90 / 100) * v_L) 
  : v_S * (100/v_S - 10/v_S) = 19 :=
by
  sorry

end sasha_kolya_distance_l202_202385


namespace solution_set_of_inequality_l202_202246

theorem solution_set_of_inequality : 
  {x : ℝ | x * (x + 3) ≥ 0} = {x : ℝ | x ≥ 0 ∨ x ≤ -3} := 
by sorry

end solution_set_of_inequality_l202_202246


namespace solution_set_l202_202879

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202879


namespace polygon_sides_l202_202104

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202104


namespace distance_between_Sasha_and_Kolya_l202_202396

/-- Sasha, Lesha, and Kolya simultaneously started a 100-meter race.
Assuming all three run at constant but unequal speeds, when Sasha
finished (100 meters), Lesha was 10 meters behind him; and when Lesha
finished, Kolya was 10 meters behind him. Thus, the distance between
Sasha and Kolya when Sasha finished is 19 meters. -/
theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL tK : ℝ), 
  vS > 0 ∧ vL > 0 ∧ vK > 0 ∧ 
  tS = 100 / vS ∧ 
  tL = 100 / vL ∧ 
  tK = 100 / vK ∧ 
  vL * tS = 90 ∧ 
  vK * tL = 90 →
  vS * tS - vK * tS = 19 :=
begin
  sorry
end

end distance_between_Sasha_and_Kolya_l202_202396


namespace probability_product_lt_36_eq_25_over_36_l202_202543

-- Definitions based on the conditions identified
def Paco_numbers : Fin 6 := ⟨i, h⟩ where i : ℕ, h : i < 6
noncomputable def Manu_numbers : Fin 12 := ⟨ j, h'⟩ where j : ℕ, h' : j < 12

-- Probability calculation
noncomputable def probability_product_less_than_36 : ℚ :=
  let outcomes := (Finset.univ : Finset (ℕ × ℕ)).filter (λ p, p.1 * p.2 < 36)
  outcomes.card.toRational / (6 * 12)

-- The proof statement
theorem probability_product_lt_36_eq_25_over_36 :
  probability_product_less_than_36 = 25 / 36 :=
by sorry

end probability_product_lt_36_eq_25_over_36_l202_202543


namespace solution_set_l202_202878

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202878


namespace inequality_proof_l202_202221

theorem inequality_proof
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_eq : a + b + c = 4 * (abc)^(1/3)) :
  2 * (ab + bc + ca) + 4 * min (a^2) (min (b^2) (c^2)) ≥ a^2 + b^2 + c^2 :=
by
  sorry

end inequality_proof_l202_202221


namespace research_question_correct_survey_method_correct_l202_202577

-- Define the conditions.
def total_students : Nat := 400
def sampled_students : Nat := 80

-- Define the research question.
def research_question : String := "To understand the vision conditions of 400 eighth-grade students in a certain school."

-- Define the survey method.
def survey_method : String := "A sampling survey method was used."

-- Prove the research_question matches the expected question given the conditions.
theorem research_question_correct :
  research_question = "To understand the vision conditions of 400 eighth-grade students in a certain school" := by
  sorry

-- Prove the survey method used matches the expected method given the conditions.
theorem survey_method_correct :
  survey_method = "A sampling survey method was used" := by
  sorry

end research_question_correct_survey_method_correct_l202_202577


namespace find_a_plus_b_l202_202496

def smallest_two_digit_multiple_of_five : ℕ := 10
def smallest_three_digit_multiple_of_seven : ℕ := 105

theorem find_a_plus_b :
  let a := smallest_two_digit_multiple_of_five
  let b := smallest_three_digit_multiple_of_seven
  a + b = 115 := by
  sorry

end find_a_plus_b_l202_202496


namespace distance_between_Sasha_and_Kolya_l202_202398

/-- Sasha, Lesha, and Kolya simultaneously started a 100-meter race.
Assuming all three run at constant but unequal speeds, when Sasha
finished (100 meters), Lesha was 10 meters behind him; and when Lesha
finished, Kolya was 10 meters behind him. Thus, the distance between
Sasha and Kolya when Sasha finished is 19 meters. -/
theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL tK : ℝ), 
  vS > 0 ∧ vL > 0 ∧ vK > 0 ∧ 
  tS = 100 / vS ∧ 
  tL = 100 / vL ∧ 
  tK = 100 / vK ∧ 
  vL * tS = 90 ∧ 
  vK * tL = 90 →
  vS * tS - vK * tS = 19 :=
begin
  sorry
end

end distance_between_Sasha_and_Kolya_l202_202398


namespace solution_set_linear_inequalities_l202_202901

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202901


namespace card_statements_has_four_true_l202_202461

noncomputable def statement1 (S : Fin 5 → Bool) : Prop := S 0 = true -> (S 1 = false ∧ S 2 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement2 (S : Fin 5 → Bool) : Prop := S 1 = true -> (S 0 = false ∧ S 2 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement3 (S : Fin 5 → Bool) : Prop := S 2 = true -> (S 0 = false ∧ S 1 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement4 (S : Fin 5 → Bool) : Prop := S 3 = true -> (S 0 = false ∧ S 1 = false ∧ S 2 = false ∧ S 4 = false)
noncomputable def statement5 (S : Fin 5 → Bool) : Prop := S 4 = true -> (S 0 = false ∧ S 1 = false ∧ S 2 = false ∧ S 3 = false)

theorem card_statements_has_four_true : ∃ (S : Fin 5 → Bool), 
  (statement1 S ∧ statement2 S ∧ statement3 S ∧ statement4 S ∧ statement5 S ∧ 
  ((S 0 = true ∨ S 1 = true ∨ S 2 = true ∨ S 3 = true ∨ S 4 = true) ∧ 
  4 = (if S 0 then 1 else 0) + (if S 1 then 1 else 0) + 
      (if S 2 then 1 else 0) + (if S 3 then 1 else 0) + 
      (if S 4 then 1 else 0))) :=
sorry

end card_statements_has_four_true_l202_202461


namespace find_a_pure_imaginary_l202_202647

theorem find_a_pure_imaginary (a : ℝ) (i : ℂ) (h1 : i = (0 : ℝ) + I) :
  (∃ b : ℝ, a - (17 / (4 - i)) = (0 + b*I)) → a = 4 :=
by
  sorry

end find_a_pure_imaginary_l202_202647


namespace kaleb_money_earned_l202_202137

-- Definitions based on the conditions
def total_games : ℕ := 10
def non_working_games : ℕ := 8
def price_per_game : ℕ := 6

-- Calculate the number of working games
def working_games : ℕ := total_games - non_working_games

-- Calculate the total money earned by Kaleb
def money_earned : ℕ := working_games * price_per_game

-- The theorem to prove
theorem kaleb_money_earned : money_earned = 12 := by sorry

end kaleb_money_earned_l202_202137


namespace twenty_mul_b_sub_a_not_integer_l202_202626

theorem twenty_mul_b_sub_a_not_integer {a b : ℝ} (hneq : a ≠ b) (hno_roots : ∀ x : ℝ,
  (x^2 + 20 * a * x + 10 * b) * (x^2 + 20 * b * x + 10 * a) ≠ 0) :
  ¬ ∃ n : ℤ, 20 * (b - a) = n :=
sorry

end twenty_mul_b_sub_a_not_integer_l202_202626


namespace equation_represents_lines_and_point_l202_202238

theorem equation_represents_lines_and_point:
    (∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 0 → (x = 1 ∧ y = -2)) ∧
    (∀ x y : ℝ, x^2 - y^2 = 0 → (x = y) ∨ (x = -y)) → 
    (∀ x y : ℝ, ((x - 1)^2 + (y + 2)^2) * (x^2 - y^2) = 0 → 
    ((x = 1 ∧ y = -2) ∨ (x + y = 0) ∨ (x - y = 0))) :=
by
  intros h1 h2 h3
  sorry

end equation_represents_lines_and_point_l202_202238


namespace triangle_side_relation_l202_202339

-- Definitions for the conditions
variable {A B C a b c : ℝ}
variable (acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
variable (sides_rel : a = (B * (1 + 2 * C)).sin)
variable (trig_eq : (B.sin * (1 + 2 * C.cos)) = (2 * A.sin * C.cos + A.cos * C.sin))

-- The statement to be proven
theorem triangle_side_relation (acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (sides_rel : a = (B * (1 + 2 * C)).sin)
  (trig_eq : (B.sin * (1 + 2 * C.cos)) = (2 * A.sin * C.cos + A.cos * C.sin)) :
  a = 2 * b := 
sorry

end triangle_side_relation_l202_202339


namespace line_equation_passing_through_and_perpendicular_l202_202445

theorem line_equation_passing_through_and_perpendicular :
  ∃ A B C : ℝ, (∀ x y : ℝ, 2 * x - 4 * y + 5 = 0 → -2 * x + y + 1 = 0 ∧ 
(x = 2 ∧ y = -1) → 2 * x + y - 3 = 0) :=
by
  sorry

end line_equation_passing_through_and_perpendicular_l202_202445


namespace trajectory_of_circle_center_l202_202311

open Real

noncomputable def circle_trajectory_equation (x y : ℝ) : Prop :=
  (y ^ 2 = 8 * x - 16)

theorem trajectory_of_circle_center (x y : ℝ) :
  (∃ C : ℝ × ℝ, (C.1 = 4 ∧ C.2 = 0) ∧
    (∃ MN : ℝ × ℝ, (MN.1 = 0 ∧ MN.2 ^ 2 = 64) ∧
    (x = C.1 ∧ y = C.2)) ∧
    circle_trajectory_equation x y) :=
sorry

end trajectory_of_circle_center_l202_202311


namespace polygon_sides_l202_202109

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202109


namespace james_chore_time_l202_202348

-- Definitions for the conditions
def t_vacuum : ℕ := 3
def t_chores : ℕ := 3 * t_vacuum
def t_total : ℕ := t_vacuum + t_chores

-- Statement
theorem james_chore_time : t_total = 12 := by
  sorry

end james_chore_time_l202_202348


namespace painted_cubes_only_two_faces_l202_202145

theorem painted_cubes_only_two_faces :
  ∀ (n : ℕ), n = 3 →
  let total_small_cubes := n * n * n in
  total_small_cubes = 27 →
  let face_painted_cubes := 6 in
  let corner_cubes := 8 in
  let inner_cubes := 1 in
  let edge_cubes := (total_small_cubes - face_painted_cubes - corner_cubes - inner_cubes) in
  edge_cubes = 12 :=
by
  intros n h1 h2 face_painted_cubes corner_cubes inner_cubes
  have h : total_small_cubes = 27 := by rw h2
  have edge_cubes_def : edge_cubes = (total_small_cubes - face_painted_cubes - corner_cubes - inner_cubes) := rfl
  have edge_cubes_result : edge_cubes = 12 := by
    simp [face_painted_cubes, corner_cubes, inner_cubes, total_small_cubes] at edge_cubes_def
    rw [←h, edge_cubes_def]
    norm_num
  exact edge_cubes_result

end painted_cubes_only_two_faces_l202_202145


namespace rotate_D_90_clockwise_l202_202312

-- Define the point D with its coordinates.
structure Point where
  x : Int
  y : Int

-- Define the original point D.
def D : Point := { x := -3, y := -8 }

-- Define the rotation transformation.
def rotate90Clockwise (p : Point) : Point :=
  { x := p.y, y := -p.x }

-- Statement to be proven.
theorem rotate_D_90_clockwise :
  rotate90Clockwise D = { x := -8, y := 3 } :=
sorry

end rotate_D_90_clockwise_l202_202312


namespace coordinates_of_point_P_l202_202018

theorem coordinates_of_point_P (x y : ℝ) (h1 : x > 0) (h2 : y < 0) (h3 : abs y = 2) (h4 : abs x = 4) : (x, y) = (4, -2) :=
by
  sorry

end coordinates_of_point_P_l202_202018


namespace max_remaining_grapes_l202_202270

theorem max_remaining_grapes (x : ℕ) : x % 7 ≤ 6 :=
  sorry

end max_remaining_grapes_l202_202270


namespace larger_cube_volume_is_512_l202_202144

def original_cube_volume := 64 -- volume in cubic feet
def scale_factor := 2 -- the factor by which the dimensions are scaled

def side_length (volume : ℕ) : ℕ := volume^(1/3) -- Assuming we have a function to compute cube root

def larger_cube_volume (original_volume : ℕ) (scale_factor : ℕ) : ℕ :=
  let original_side_length := side_length original_volume
  let larger_side_length := scale_factor * original_side_length
  larger_side_length ^ 3

theorem larger_cube_volume_is_512 :
  larger_cube_volume original_cube_volume scale_factor = 512 :=
sorry

end larger_cube_volume_is_512_l202_202144


namespace triangle_ABC_c_and_A_value_sin_2C_minus_pi_6_l202_202206

-- Define the properties and variables of the given obtuse triangle
variables (a b c : ℝ) (A C : ℝ)
-- Given conditions
axiom ha : a = 7
axiom hb : b = 3
axiom hcosC : Real.cos C = 11 / 14

-- Prove the values of c and angle A
theorem triangle_ABC_c_and_A_value (ha : a = 7) (hb : b = 3) (hcosC : Real.cos C = 11 / 14) : c = 5 ∧ A = 2 * Real.pi / 3 :=
sorry

-- Prove the value of sin(2C - π / 6)
theorem sin_2C_minus_pi_6 (ha : a = 7) (hb : b = 3) (hcosC : Real.cos C = 11 / 14) : Real.sin (2 * C - Real.pi / 6) = 71 / 98 :=
sorry

end triangle_ABC_c_and_A_value_sin_2C_minus_pi_6_l202_202206


namespace prove_expression_l202_202791

theorem prove_expression (a : ℝ) (h : a^2 + a - 1 = 0) : 2 * a^2 + 2 * a + 2008 = 2010 := by
  sorry

end prove_expression_l202_202791


namespace problem_statement_false_adjacent_complementary_l202_202588

-- Definition of straight angle, supplementary angles, and complementary angles.
def is_straight_angle (θ : ℝ) : Prop := θ = 180
def are_supplementary (θ ψ : ℝ) : Prop := θ + ψ = 180
def are_complementary (θ ψ : ℝ) : Prop := θ + ψ = 90

-- Definition of adjacent angles (for completeness, though we don't use adjacency differently right now)
def are_adjacent (θ ψ : ℝ) : Prop := ∀ x, θ + x + ψ + x = θ + ψ -- Simplified

-- Additional conditions that could be true or false -- we need one of them to be false.
def false_statement_D (θ ψ : ℝ) : Prop :=
  are_complementary θ ψ → are_adjacent θ ψ

theorem problem_statement_false_adjacent_complementary :
  ∃ (θ ψ : ℝ), ¬ false_statement_D θ ψ :=
by
  sorry

end problem_statement_false_adjacent_complementary_l202_202588


namespace solution_set_of_inequalities_l202_202970

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202970


namespace probability_fourth_roll_six_l202_202162

theorem probability_fourth_roll_six
  (fair_die : ℕ → ℝ)
  (biased_die : ℕ → ℝ)
  (prob_fair_die_six : fair_die 6 = 1 / 6)
  (prob_biased_die_six : biased_die 6 = 3 / 4)
  (prob_biased_die_other : ∀ i, i ≠ 6 → biased_die i = 1 / 20)
  (first_three_sixes : ℕ → ℝ) :
  first_three_sixes 6 = 774 / 1292 := 
sorry

end probability_fourth_roll_six_l202_202162


namespace intersection_of_M_and_N_l202_202194

open Set

variable (M N : Set ℕ)

theorem intersection_of_M_and_N :
  M = {1, 2, 4, 8, 16} →
  N = {2, 4, 6, 8} →
  M ∩ N = {2, 4, 8} :=
by
  intros hM hN
  rw [hM, hN]
  ext x
  simp
  sorry

end intersection_of_M_and_N_l202_202194


namespace sequence_a_10_l202_202772

theorem sequence_a_10 (a : ℕ → ℤ) 
  (H1 : ∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p + a q)
  (H2 : a 2 = -6) : 
  a 10 = -30 :=
sorry

end sequence_a_10_l202_202772


namespace expected_value_fair_dodecahedral_die_l202_202584

theorem expected_value_fair_dodecahedral_die : 
  let n := 12 in (1 / n) * (List.sum (List.range n).map (λ x => (x + 1))) = 6.5 :=
by
  let n := 12
  have h1 : (1 : ℚ) / n = 1 / 12 := by sorry
  have h2 : List.range n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] := by sorry
  have h3 : List.sum (List.range n).map (λ x => (x + 1)) = 78 := by sorry
  have h4 : (1 / 12) * 78 = 6.5 := by sorry
  exact h4

end expected_value_fair_dodecahedral_die_l202_202584


namespace maximum_value_frac_l202_202224

-- Let x and y be positive real numbers. Prove that (x + y)^3 / (x^3 + y^3) ≤ 4.
theorem maximum_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y)^3 / (x^3 + y^3) ≤ 4 := sorry

end maximum_value_frac_l202_202224


namespace solution_set_system_of_inequalities_l202_202931

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202931


namespace min_value_x_3y_l202_202058

noncomputable def min_value (x y : ℝ) : ℝ := x + 3 * y

theorem min_value_x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  ∃ (x y : ℝ), min_value x y = 18 + 21 * Real.sqrt 3 :=
sorry

end min_value_x_3y_l202_202058


namespace Avery_builds_in_4_hours_l202_202156

variable (A : ℝ) (TomTime : ℝ := 2) (TogetherTime : ℝ := 1) (RemainingTomTime : ℝ := 0.5)

-- Conditions:
axiom Tom_builds_in_2_hours : TomTime = 2
axiom Work_together_for_1_hour : TogetherTime = 1
axiom Tom_finishes_in_0_5_hours : RemainingTomTime = 0.5

-- Question:
theorem Avery_builds_in_4_hours : A = 4 :=
by
  sorry

end Avery_builds_in_4_hours_l202_202156


namespace exists_integers_m_n_l202_202811

theorem exists_integers_m_n (a b c p q r : ℝ) (h_a : a ≠ 0) (h_p : p ≠ 0) :
  ∃ (m n : ℤ), ∀ (x : ℝ), (a * x^2 + b * x + c = m * (p * x^2 + q * x + r) + n) := sorry

end exists_integers_m_n_l202_202811


namespace solution_set_of_inequalities_l202_202968

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202968


namespace solution_set_of_linear_inequalities_l202_202912

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202912


namespace orthocentric_tetrahedron_distance_l202_202057

open EuclideanGeometry

/-- Given H as the orthocenter of an orthocentric tetrahedron, M as the centroid of one of its faces,
    and N as one of the points where the line HM intersects the circumsphere of the tetrahedron
    (with M between H and N), prove that |MN| = 2|HM|. -/
theorem orthocentric_tetrahedron_distance (H M N : Point ℝ)
  (tetrahedron : ∀ (V : ℕ), V < 4 → Point ℝ)
  (orthocentric : orthocenter_δοκ_textahedron tetrahedron H)
  (centroid_face : ∃ face, is_face_of tetrahedron face ∧ face_center face = M)
  (circumsphere : N ∈ circumsphere tetrahedron)
  (H_collinear_MN : collinear {H, M, N})
  (M_between_HN : between H M N) :
  dist M N = 2 * dist H M :=
sorry

end orthocentric_tetrahedron_distance_l202_202057


namespace floor_sqrt_80_l202_202469

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 :=
  sorry

end floor_sqrt_80_l202_202469


namespace jake_sister_weight_ratio_l202_202797

theorem jake_sister_weight_ratio (Jake_initial_weight : ℕ) (total_weight : ℕ) (weight_loss : ℕ) (sister_weight : ℕ) 
(h₁ : Jake_initial_weight = 156) 
(h₂ : total_weight = 224) 
(h₃ : weight_loss = 20) 
(h₄ : total_weight = Jake_initial_weight + sister_weight) :
(Jake_initial_weight - weight_loss) / sister_weight = 2 := by
  sorry

end jake_sister_weight_ratio_l202_202797


namespace loaned_books_count_l202_202438

variable (x : ℕ)

def initial_books : ℕ := 75
def percentage_returned : ℝ := 0.65
def end_books : ℕ := 54
def non_returned_books : ℕ := initial_books - end_books
def percentage_non_returned : ℝ := 1 - percentage_returned

theorem loaned_books_count :
  percentage_non_returned * (x:ℝ) = non_returned_books → x = 60 :=
by
  sorry

end loaned_books_count_l202_202438


namespace average_age_l202_202710

theorem average_age (Jared Molly Hakimi : ℕ) (h1 : Jared = Hakimi + 10) (h2 : Molly = 30) (h3 : Hakimi = 40) :
  (Jared + Molly + Hakimi) / 3 = 40 :=
by
  sorry

end average_age_l202_202710


namespace complement_union_l202_202330

open Set

namespace ProofExample

def U : Set ℝ := univ

def A : Set ℝ := { x | x ≥ 1 }

def B : Set ℝ := { x | x ≤ 0 }

theorem complement_union:
  (U \ (A ∪ B)) = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end ProofExample

end complement_union_l202_202330


namespace number_of_male_students_in_sample_l202_202804

theorem number_of_male_students_in_sample 
  (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (sample_female_students : ℕ) (sample_male_students : ℕ) :
  total_students = 680 →
  male_students = 360 →
  female_students = 320 →
  sample_female_students = 16 →
  (female_students * sample_male_students = male_students * sample_female_students) →
  sample_male_students = 18 :=
by
  intros h_total h_male h_female h_sample_female h_proportion
  sorry

end number_of_male_students_in_sample_l202_202804


namespace xiaoli_estimate_larger_l202_202437

theorem xiaoli_estimate_larger (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) : 
  (1.1 * x) / (0.9 * y) > x / y :=
by
  sorry

end xiaoli_estimate_larger_l202_202437


namespace no_two_primes_sum_to_53_l202_202041

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_two_primes_sum_to_53 :
  ¬ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_two_primes_sum_to_53_l202_202041


namespace parallel_lines_eq_a2_l202_202083

theorem parallel_lines_eq_a2
  (a : ℝ)
  (h : ∀ x y : ℝ, x + a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0)
  : a = 2 := 
  sorry

end parallel_lines_eq_a2_l202_202083


namespace linear_inequalities_solution_l202_202859

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202859


namespace distinct_triangles_n_gon_l202_202544

-- Define the conditions in the form of a Lean definition or theorem

theorem distinct_triangles_n_gon (n : ℕ) (h : 3 ≤ n) :
  let T := (n^2 - n)/6 in
  T = Nat.floor (n^2 / 12) := 
by
  sorry

end distinct_triangles_n_gon_l202_202544


namespace swimming_pool_width_l202_202571

theorem swimming_pool_width
  (length : ℝ)
  (lowered_height_inches : ℝ)
  (removed_water_gallons : ℝ)
  (gallons_per_cubic_foot : ℝ)
  (volume_for_removal : ℝ)
  (width : ℝ) :
  length = 60 → 
  lowered_height_inches = 6 →
  removed_water_gallons = 4500 →
  gallons_per_cubic_foot = 7.5 →
  volume_for_removal = removed_water_gallons / gallons_per_cubic_foot →
  width = volume_for_removal / (length * (lowered_height_inches / 12)) →
  width = 20 :=
by
  intros h_length h_lowered_height h_removed_water h_gallons_per_cubic_foot h_volume_for_removal h_width
  sorry

end swimming_pool_width_l202_202571


namespace polygon_sides_l202_202093

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202093


namespace sasha_kolya_distance_l202_202383

theorem sasha_kolya_distance
  (v_S v_L v_K : ℝ) 
  (h1 : Lesha_dist = 100 - 10) 
  (h2 : Kolya_dist = 100 - 10) 
  (h3 : v_L = (90 / 100) * v_S) 
  (h4 : v_K = (90 / 100) * v_L) 
  : v_S * (100/v_S - 10/v_S) = 19 :=
by
  sorry

end sasha_kolya_distance_l202_202383


namespace evaluate_expression_l202_202030

theorem evaluate_expression (a b x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
    (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  sorry

end evaluate_expression_l202_202030


namespace percentage_solution_l202_202336

noncomputable def percentage_of_difference (P : ℚ) (x y : ℚ) : Prop :=
  (P / 100) * (x - y) = (14 / 100) * (x + y)

theorem percentage_solution (x y : ℚ) (h1 : y = 0.17647058823529413 * x)
  (h2 : percentage_of_difference P x y) : 
  P = 20 := 
by
  sorry

end percentage_solution_l202_202336


namespace solution_set_l202_202870

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202870


namespace jessica_age_l202_202049

theorem jessica_age 
  (j g : ℚ)
  (h1 : g = 15 * j) 
  (h2 : g - j = 60) : 
  j = 30 / 7 :=
by
  sorry

end jessica_age_l202_202049


namespace find_fraction_increase_l202_202167

noncomputable def present_value : ℝ := 64000
noncomputable def value_after_two_years : ℝ := 87111.11111111112

theorem find_fraction_increase (f : ℝ) :
  64000 * (1 + f) ^ 2 = 87111.11111111112 → f = 0.1666666666666667 := 
by
  intro h
  -- proof steps here
  sorry

end find_fraction_increase_l202_202167


namespace solution_set_l202_202876

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202876


namespace scientific_notation_of_19672_l202_202283

theorem scientific_notation_of_19672 :
  ∃ a b, 19672 = a * 10^b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.9672 ∧ b = 4 :=
sorry

end scientific_notation_of_19672_l202_202283


namespace ideal_type_circle_D_l202_202329

-- Define the line equation
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the distance condition for circles
def ideal_type_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ (P Q : ℝ × ℝ), 
    line_l P.1 P.2 ∧ line_l Q.1 Q.2 ∧
    dist P (0, 0) = radius ∧
    dist Q (0, 0) = radius ∧
    dist (P, Q) = 1

-- Definition of given circles A, B, C, D
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_B (x y : ℝ) : Prop := x^2 + y^2 = 16
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + (y - 4)^2 = 1
def circle_D (x y : ℝ) : Prop := (x - 4)^2 + (y - 4)^2 = 16

-- Define circle centers and radii for A, B, C, D
def center_A : ℝ × ℝ := (0, 0)
def radius_A : ℝ := 1
def center_B : ℝ × ℝ := (0, 0)
def radius_B : ℝ := 4
def center_C : ℝ × ℝ := (4, 4)
def radius_C : ℝ := 1
def center_D : ℝ × ℝ := (4, 4)
def radius_D : ℝ := 4

-- Problem Statement: Prove that option D is the "ideal type" circle
theorem ideal_type_circle_D : 
  ideal_type_circle center_D radius_D :=
sorry

end ideal_type_circle_D_l202_202329


namespace polygon_sides_l202_202091

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l202_202091


namespace complement_intersection_l202_202682

open Set -- Open the Set namespace

variable (U : Set ℝ := univ)
variable (A : Set ℝ := {x | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2})
variable (B : Set ℝ := {x | x ≤ -1 ∨ x > 2})

theorem complement_intersection :
  (U \ B) ∩ A = {x | x = 0 ∨ x = 1 ∨ x = 2} :=
by
  sorry -- Proof not required as per the instructions

end complement_intersection_l202_202682


namespace extreme_values_f_a4_no_zeros_f_on_1e_l202_202310

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (a + 2) * Real.log x - 2 / x + 2

theorem extreme_values_f_a4 :
  f 4 (1 / 2) = 6 * Real.log 2 ∧ f 4 1 = 4 := sorry

theorem no_zeros_f_on_1e (a : ℝ) :
  (a ≤ 0 ∨ a ≥ 2 / (Real.exp 1 * (Real.exp 1 - 1))) →
  ∀ x, 1 < x → x < Real.exp 1 → f a x ≠ 0 := sorry

end extreme_values_f_a4_no_zeros_f_on_1e_l202_202310


namespace distinguishable_octahedrons_l202_202611

noncomputable def number_of_distinguishable_octahedrons (total_colors : ℕ) (used_colors : ℕ) : ℕ :=
  let num_ways_choose_colors := Nat.choose total_colors (used_colors - 1)
  let num_permutations := (used_colors - 1).factorial
  let num_rotations := 3
  (num_ways_choose_colors * num_permutations) / num_rotations

theorem distinguishable_octahedrons (h : number_of_distinguishable_octahedrons 9 8 = 13440) : true := sorry

end distinguishable_octahedrons_l202_202611


namespace construct_largest_area_triangle_l202_202014

noncomputable def largest_area_triangle (a b : ℝ) (i : ℝ × ℝ) : Prop :=
∃ P Q : ℝ × ℝ, 
  (is_parallel P Q i) ∧ 
  (forms_triangle_with_center P Q (0, 0)) ∧ 
  (area_of_triangle (0, 0) P Q = (a * b) / 2)

theorem construct_largest_area_triangle 
  (a b : ℝ) (i : ℝ × ℝ) (h_ellipse : True) : largest_area_triangle a b i := 
by
  sorry

end construct_largest_area_triangle_l202_202014


namespace student_attempted_sums_l202_202612

theorem student_attempted_sums (right wrong : ℕ) (h1 : wrong = 2 * right) (h2 : right = 12) : right + wrong = 36 := sorry

end student_attempted_sums_l202_202612


namespace constant_condition_for_quadrant_I_solution_l202_202524

-- Define the given conditions
def equations (c : ℚ) (x y : ℚ) : Prop :=
  (x - 2 * y = 5) ∧ (c * x + 3 * y = 2)

-- Define the condition for the solution to be in Quadrant I
def isQuadrantI (x y : ℚ) : Prop :=
  (x > 0) ∧ (y > 0)

-- The theorem to be proved
theorem constant_condition_for_quadrant_I_solution (c : ℚ) :
  (∃ x y : ℚ, equations c x y ∧ isQuadrantI x y) ↔ (-3/2 < c ∧ c < 2/5) :=
by
  sorry

end constant_condition_for_quadrant_I_solution_l202_202524


namespace solution_set_linear_inequalities_l202_202934

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202934


namespace window_treatments_cost_l202_202679

def cost_of_sheers (n : ℕ) (cost_per_pair : ℝ) : ℝ := n * cost_per_pair
def cost_of_drapes (n : ℕ) (cost_per_pair : ℝ) : ℝ := n * cost_per_pair
def total_cost (n : ℕ) (cost_sheers : ℝ) (cost_drapes : ℝ) : ℝ :=
  cost_of_sheers n cost_sheers + cost_of_drapes n cost_drapes

theorem window_treatments_cost :
  total_cost 3 40 60 = 300 :=
by
  sorry

end window_treatments_cost_l202_202679


namespace fresh_grapes_weight_l202_202480

/-- Given fresh grapes containing 90% water by weight, 
    and dried grapes containing 20% water by weight,
    if the weight of dried grapes obtained from a certain amount of fresh grapes is 2.5 kg,
    then the weight of the fresh grapes used is 20 kg.
-/
theorem fresh_grapes_weight (F D : ℝ)
  (hD : D = 2.5)
  (fresh_water_content : ℝ := 0.90)
  (dried_water_content : ℝ := 0.20)
  (fresh_solid_content : ℝ := 1 - fresh_water_content)
  (dried_solid_content : ℝ := 1 - dried_water_content)
  (solid_mass_constancy : fresh_solid_content * F = dried_solid_content * D) : 
  F = 20 := 
  sorry

end fresh_grapes_weight_l202_202480


namespace solution_set_linear_inequalities_l202_202938

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202938


namespace midpoint_trajectory_extension_trajectory_l202_202142

-- Define the conditions explicitly

def is_midpoint (M A O : ℝ × ℝ) : Prop :=
  M = ((O.1 + A.1) / 2, (O.2 + A.2) / 2)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 - 8 * P.1 = 0

-- First problem: Trajectory equation of the midpoint M
theorem midpoint_trajectory (M O A : ℝ × ℝ) (hO : O = (0,0)) (hA : on_circle A) (hM : is_midpoint M A O) :
  M.1 ^ 2 + M.2 ^ 2 - 4 * M.1 = 0 :=
sorry

-- Define the condition for N
def extension_point (O A N : ℝ × ℝ) : Prop :=
  (A.1 - O.1) * 2 = N.1 - O.1 ∧ (A.2 - O.2) * 2 = N.2 - O.2

-- Second problem: Trajectory equation of the point N
theorem extension_trajectory (N O A : ℝ × ℝ) (hO : O = (0,0)) (hA : on_circle A) (hN : extension_point O A N) :
  N.1 ^ 2 + N.2 ^ 2 - 16 * N.1 = 0 :=
sorry

end midpoint_trajectory_extension_trajectory_l202_202142


namespace negation_of_exists_l202_202084

theorem negation_of_exists (h : ¬ ∃ x : ℝ, x^2 + x + 1 < 0) : ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l202_202084


namespace calculate_adult_chaperones_l202_202214

theorem calculate_adult_chaperones (students : ℕ) (student_fee : ℕ) (adult_fee : ℕ) (total_fee : ℕ) 
  (h_students : students = 35) 
  (h_student_fee : student_fee = 5) 
  (h_adult_fee : adult_fee = 6) 
  (h_total_fee : total_fee = 199) : 
  ∃ (A : ℕ), 35 * student_fee + A * adult_fee = 199 ∧ A = 4 := 
by
  sorry

end calculate_adult_chaperones_l202_202214


namespace matilda_initial_bars_l202_202823

theorem matilda_initial_bars (M : ℕ) 
  (shared_evenly : 5 * M = 20 * 2 / 5)
  (half_given_to_father : M / 2 * 5 = 10)
  (father_bars : 5 + 3 + 2 = 10) :
  M = 4 := 
by
  sorry

end matilda_initial_bars_l202_202823


namespace find_missing_number_l202_202442

theorem find_missing_number (n : ℝ) : n * 120 = 173 * 240 → n = 345.6 :=
by
  intros h
  sorry

end find_missing_number_l202_202442


namespace sum_original_and_correct_value_l202_202135

theorem sum_original_and_correct_value (x : ℕ) (h : x + 14 = 68) :
  x + (x + 41) = 149 := by
  sorry

end sum_original_and_correct_value_l202_202135


namespace number_in_eighth_group_l202_202618

theorem number_in_eighth_group (employees groups n l group_size numbering_drawn starting_number: ℕ) 
(h1: employees = 200) 
(h2: groups = 40) 
(h3: n = 5) 
(h4: number_in_fifth_group = 23) 
(h5: starting_number + 4 * n = number_in_fifth_group) : 
  starting_number + 7 * n = 38 :=
by
  sorry

end number_in_eighth_group_l202_202618


namespace range_of_quadratic_function_l202_202707

theorem range_of_quadratic_function :
  ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), -x^2 - 4 * x + 1 ∈ Set.Icc (-11) (5) :=
by
  sorry

end range_of_quadratic_function_l202_202707


namespace sum_of_g_31_values_l202_202361

def f (x : ℝ) : ℝ := 4 * x^2 - 3
def g (y : ℝ) : ℝ := y ^ 2 - y + 2

theorem sum_of_g_31_values :
  g 31 + g 31 = 21 := sorry

end sum_of_g_31_values_l202_202361


namespace basketball_court_perimeter_l202_202560

variables {Width Length : ℕ}

def width := 17
def length := 31

def perimeter (width length : ℕ) := 2 * (length + width)

theorem basketball_court_perimeter : 
  perimeter width length = 96 :=
sorry

end basketball_court_perimeter_l202_202560


namespace smallest_N_exists_l202_202568

theorem smallest_N_exists (
  a b c d : ℕ := list.perm [1, 2, 3, 4, 5] [gcd a b, gcd a c, gcd a d, gcd b c, gcd b d, gcd c d]
  (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_N: N > 5) : 
  N = 14 :=
by sorry

end smallest_N_exists_l202_202568


namespace area_of_circular_platform_l202_202741

theorem area_of_circular_platform (d : ℝ) (h : d = 2) : ∃ (A : ℝ), A = Real.pi ∧ A = π *(d / 2)^2 := by
  sorry

end area_of_circular_platform_l202_202741


namespace sin_seven_pi_over_six_l202_202300

theorem sin_seven_pi_over_six :
  Real.sin (7 * Real.pi / 6) = - 1 / 2 :=
by
  sorry

end sin_seven_pi_over_six_l202_202300


namespace each_friend_pays_6413_l202_202138

noncomputable def amount_each_friend_pays (total_bill : ℝ) (friends : ℕ) (first_discount : ℝ) (second_discount : ℝ) : ℝ :=
  let bill_after_first_coupon := total_bill * (1 - first_discount)
  let bill_after_second_coupon := bill_after_first_coupon * (1 - second_discount)
  bill_after_second_coupon / friends

theorem each_friend_pays_6413 :
  amount_each_friend_pays 600 8 0.10 0.05 = 64.13 :=
by
  sorry

end each_friend_pays_6413_l202_202138


namespace fairy_island_counties_l202_202518

theorem fairy_island_counties : 
  let init_elves := 1 in
  let init_dwarfs := 1 in
  let init_centaurs := 1 in

  -- After the first year
  let elves_1 := init_elves in
  let dwarfs_1 := init_dwarfs * 3 in
  let centaurs_1 := init_centaurs * 3 in

  -- After the second year
  let elves_2 := elves_1 * 4 in
  let dwarfs_2 := dwarfs_1 in
  let centaurs_2 := centaurs_1 * 4 in

  -- After the third year
  let elves_3 := elves_2 * 6 in
  let dwarfs_3 := dwarfs_2 * 6 in
  let centaurs_3 := centaurs_2 in

  -- Total counties after all events
  elves_3 + dwarfs_3 + centaurs_3 = 54 := 
by {
  sorry
}

end fairy_island_counties_l202_202518


namespace number_of_Al_atoms_l202_202143

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_Br : ℝ := 79.90
def number_of_Br_atoms : ℕ := 3
def molecular_weight : ℝ := 267

theorem number_of_Al_atoms (x : ℝ) : 
  molecular_weight = (atomic_weight_Al * x) + (atomic_weight_Br * number_of_Br_atoms) → 
  x = 1 :=
by
  sorry

end number_of_Al_atoms_l202_202143


namespace product_prob_less_than_36_is_67_over_72_l202_202535

def prob_product_less_than_36 : ℚ :=
  let P := [1, 2, 3, 4, 5, 6]
  let M := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (P.bind (λ p, M.filter (λ m, p * m < 36))).length / (P.length * M.length : ℚ)

theorem product_prob_less_than_36_is_67_over_72 :
  prob_product_less_than_36 = 67 / 72 :=
by
  sorry

end product_prob_less_than_36_is_67_over_72_l202_202535


namespace distance_between_Sasha_and_Kolya_l202_202395

/-- Sasha, Lesha, and Kolya simultaneously started a 100-meter race.
Assuming all three run at constant but unequal speeds, when Sasha
finished (100 meters), Lesha was 10 meters behind him; and when Lesha
finished, Kolya was 10 meters behind him. Thus, the distance between
Sasha and Kolya when Sasha finished is 19 meters. -/
theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL tK : ℝ), 
  vS > 0 ∧ vL > 0 ∧ vK > 0 ∧ 
  tS = 100 / vS ∧ 
  tL = 100 / vL ∧ 
  tK = 100 / vK ∧ 
  vL * tS = 90 ∧ 
  vK * tL = 90 →
  vS * tS - vK * tS = 19 :=
begin
  sorry
end

end distance_between_Sasha_and_Kolya_l202_202395


namespace remainder_3_pow_19_mod_10_l202_202729

theorem remainder_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 :=
by
  sorry

end remainder_3_pow_19_mod_10_l202_202729


namespace solution_set_linear_inequalities_l202_202904

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202904


namespace minimum_inverse_sum_l202_202309

theorem minimum_inverse_sum (a b : ℝ) (h1 : (a > 0) ∧ (b > 0)) 
  (h2 : 3 * a + 4 * b = 55) : 
  (1 / a) + (1 / b) ≥ (7 + 4 * Real.sqrt 3) / 55 :=
sorry

end minimum_inverse_sum_l202_202309


namespace impossible_network_of_triangles_l202_202068

-- Define the conditions of the problem, here we could define vertices and properties of the network
structure Vertex :=
(triangles_meeting : Nat)

def five_triangles_meeting (v : Vertex) : Prop :=
v.triangles_meeting = 5

-- The main theorem statement - it's impossible to cover the entire plane with such a network
theorem impossible_network_of_triangles :
  ¬ (∀ v : Vertex, five_triangles_meeting v) :=
sorry

end impossible_network_of_triangles_l202_202068


namespace caesars_charge_l202_202413

theorem caesars_charge :
  ∃ (C : ℕ), (C + 30 * 60 = 500 + 35 * 60) ↔ (C = 800) :=
by
  sorry

end caesars_charge_l202_202413


namespace cost_per_scarf_l202_202128

-- Define the cost of each earring
def cost_of_earring : ℕ := 6000

-- Define the number of earrings
def num_earrings : ℕ := 2

-- Define the cost of the iPhone
def cost_of_iphone : ℕ := 2000

-- Define the number of scarves
def num_scarves : ℕ := 4

-- Define the total value of the swag bag
def total_swag_bag_value : ℕ := 20000

-- Define the total value of diamond earrings and the iPhone
def total_value_of_earrings_and_iphone : ℕ := (num_earrings * cost_of_earring) + cost_of_iphone

-- Define the total value of the scarves
def total_value_of_scarves : ℕ := total_swag_bag_value - total_value_of_earrings_and_iphone

-- Define the cost of each designer scarf
def cost_of_each_scarf : ℕ := total_value_of_scarves / num_scarves

-- Prove that each designer scarf costs $1,500
theorem cost_per_scarf : cost_of_each_scarf = 1500 := by
  sorry

end cost_per_scarf_l202_202128


namespace candidates_count_l202_202151

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 90) : n = 10 :=
by
  sorry

end candidates_count_l202_202151


namespace salt_solution_concentration_l202_202734

theorem salt_solution_concentration :
  ∀ (C : ℝ),
  (∀ (mix_vol : ℝ) (pure_water : ℝ) (salt_solution_vol : ℝ),
    mix_vol = 1.5 →
    pure_water = 1 →
    salt_solution_vol = 0.5 →
    1.5 * 0.15 = 0.5 * (C / 100) →
    C = 45) :=
by
  intros C mix_vol pure_water salt_solution_vol h_mix h_pure h_salt h_eq
  sorry

end salt_solution_concentration_l202_202734


namespace problem_solution_exists_l202_202659

theorem problem_solution_exists {x : ℝ} :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ), 
    1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 =
    a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 +
    a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + 
    a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7)
  → a_2 = 56 :=
sorry

end problem_solution_exists_l202_202659


namespace arithmetic_series_first_term_l202_202178

theorem arithmetic_series_first_term :
  ∃ (a d : ℝ), (25 * (2 * a + 49 * d) = 200) ∧ (25 * (2 * a + 149 * d) = 2700) ∧ (a = -20.5) :=
by
  sorry

end arithmetic_series_first_term_l202_202178


namespace solution_set_linear_inequalities_l202_202902

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202902


namespace six_a_seven_eight_b_div_by_45_l202_202344

/-- If the number 6a78b is divisible by 45, then a + b = 6. -/
theorem six_a_seven_eight_b_div_by_45 (a b : ℕ) (h1: 0 ≤ a ∧ a < 10) (h2: 0 ≤ b ∧ b < 10)
  (h3 : (6 * 10^4 + a * 10^3 + 7 * 10^2 + 8 * 10 + b) % 45 = 0) : a + b = 6 := 
by
  sorry

end six_a_seven_eight_b_div_by_45_l202_202344


namespace not_necessarily_divisible_by_20_l202_202364

theorem not_necessarily_divisible_by_20 (k : ℤ) (h : ∃ k : ℤ, 5 ∣ k * (k+1) * (k+2)) : ¬ ∀ k : ℤ, 20 ∣ k * (k+1) * (k+2) :=
by
  sorry

end not_necessarily_divisible_by_20_l202_202364


namespace second_lady_distance_l202_202251

theorem second_lady_distance (x : ℕ) 
  (h1 : ∃ y, y = 2 * x) 
  (h2 : x + 2 * x = 12) : x = 4 := 
by 
  sorry

end second_lady_distance_l202_202251


namespace min_value_f_l202_202475

noncomputable def f (x : ℝ) : ℝ := |x - 4| + |x - 6|

theorem min_value_f : ∃ x : ℝ, f x ≥ 2 :=
by
  sorry

end min_value_f_l202_202475


namespace compute_expression_l202_202056

theorem compute_expression (w : ℂ) (hw : w = Complex.exp (Complex.I * (6 * Real.pi / 11))) (hwp : w^11 = 1) :
  (w / (1 + w^3) + w^2 / (1 + w^6) + w^3 / (1 + w^9) = -2) :=
sorry

end compute_expression_l202_202056


namespace stratified_sampling_middle_schools_l202_202803

theorem stratified_sampling_middle_schools (high_schools : ℕ) (middle_schools : ℕ) (elementary_schools : ℕ) (total_selected : ℕ) 
    (h_high_schools : high_schools = 10) (h_middle_schools : middle_schools = 30) (h_elementary_schools : elementary_schools = 60)
    (h_total_selected : total_selected = 20) : 
    middle_schools * (total_selected / (high_schools + middle_schools + elementary_schools)) = 6 := 
by 
  sorry

end stratified_sampling_middle_schools_l202_202803


namespace p_necessary_not_sufficient_for_q_l202_202643

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def p (a : ℝ) : Prop :=
  collinear (vec a (a^2)) (vec 1 2)

def q (a : ℝ) : Prop := a = 2

theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ ¬(∀ a : ℝ, p a → q a) :=
sorry

end p_necessary_not_sufficient_for_q_l202_202643


namespace negation_of_proposition_l202_202705

-- Define the original proposition and its negation
def original_proposition (x : ℝ) : Prop := x^2 - 3*x + 3 > 0
def negated_proposition (x : ℝ) : Prop := x^2 - 3*x + 3 ≤ 0

-- The theorem about the negation of the original proposition
theorem negation_of_proposition :
  ¬ (∀ x : ℝ, original_proposition x) ↔ ∃ x : ℝ, negated_proposition x :=
by
  sorry

end negation_of_proposition_l202_202705


namespace remainder_of_number_of_minimally_intersecting_triples_l202_202054

noncomputable def number_of_minimally_intersecting_triples : Nat :=
  let n := (8 * 7 * 6) * (4 ^ 5)
  n % 1000

theorem remainder_of_number_of_minimally_intersecting_triples :
  number_of_minimally_intersecting_triples = 64 := by
  sorry

end remainder_of_number_of_minimally_intersecting_triples_l202_202054


namespace pen_and_notebook_cost_l202_202678

theorem pen_and_notebook_cost :
  ∃ (p n : ℕ), 15 * p + 5 * n = 130 ∧ p > n ∧ p + n = 10 := by
  sorry

end pen_and_notebook_cost_l202_202678


namespace probability_of_product_lt_36_l202_202540

noncomputable def probability_less_than_36 : ℚ :=
  ∑ p in Finset.range 1 7, ∑ m in Finset.range 1 13, if (p * m < 36) then 1 else 0

theorem probability_of_product_lt_36 :
  probability (paco_num: ℕ, h_paco: 1 ≤ paco_num ∧ paco_num ≤ 6) *
  probability (manu_num: ℕ, h_manu: 1 ≤ manu_num ∧ manu_num ≤ 12) *
  (if paco_num * manu_num < 36 then 1 else 0) = 7 / 9 := 
sorry

end probability_of_product_lt_36_l202_202540


namespace Molly_age_now_l202_202229

/- Definitions -/
def Sandy_curr_age : ℕ := 60
def Molly_curr_age (S : ℕ) : ℕ := 3 * S / 4
def Sandy_age_in_6_years (S : ℕ) : ℕ := S + 6

/- Theorem to prove -/
theorem Molly_age_now 
  (ratio_condition : ∀ S M : ℕ, S / M = 4 / 3 → M = 3 * S / 4)
  (age_condition : Sandy_age_in_6_years Sandy_curr_age = 66) : 
  Molly_curr_age Sandy_curr_age = 45 :=
by
  sorry

end Molly_age_now_l202_202229


namespace solution_set_l202_202881

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202881


namespace problem_l202_202523

theorem problem (a : ℤ) (ha : 0 ≤ a ∧ a < 13) (hdiv : (51 ^ 2016 + a) % 13 = 0) : a = 12 :=
sorry

end problem_l202_202523


namespace solution_set_of_inequalities_l202_202983

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202983


namespace number_is_twenty_l202_202501

-- We state that if \( \frac{30}{100}x = \frac{15}{100} \times 40 \), then \( x = 20 \)
theorem number_is_twenty (x : ℝ) (h : (30 / 100) * x = (15 / 100) * 40) : x = 20 :=
by
  sorry

end number_is_twenty_l202_202501


namespace elise_spent_on_puzzle_l202_202628

-- Definitions based on the problem conditions:
def initial_money : ℕ := 8
def saved_money : ℕ := 13
def spent_on_comic : ℕ := 2
def remaining_money : ℕ := 1

-- Prove that the amount spent on the puzzle is $18.
theorem elise_spent_on_puzzle : initial_money + saved_money - spent_on_comic - remaining_money = 18 := by
  sorry

end elise_spent_on_puzzle_l202_202628


namespace mary_more_candy_initially_l202_202822

-- Definitions of the conditions
def Megan_initial_candy : ℕ := 5
def Mary_candy_after_addition : ℕ := 25
def additional_candy_Mary_adds : ℕ := 10

-- The proof problem statement
theorem mary_more_candy_initially :
  (Mary_candy_after_addition - additional_candy_Mary_adds) / Megan_initial_candy = 3 :=
by
  sorry

end mary_more_candy_initially_l202_202822


namespace value_of_f_at_3_l202_202198

def f (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 - 3 * x + 7

theorem value_of_f_at_3 : f 3 = 196 := by
  sorry

end value_of_f_at_3_l202_202198


namespace solution_set_linear_inequalities_l202_202945

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202945


namespace geometric_sequence_a1_l202_202775

theorem geometric_sequence_a1 (a : ℕ → ℝ) (q : ℝ) 
  (hq : 0 < q)
  (h1 : a 4 * a 8 = 2 * (a 5) ^ 2)
  (h2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end geometric_sequence_a1_l202_202775


namespace value_of_x_minus_y_l202_202792

theorem value_of_x_minus_y 
  (x y : ℝ)
  (h1 : x + y = 2)
  (h2 : 3 * x - y = 8) :
  x - y = 3 := by
  sorry

end value_of_x_minus_y_l202_202792


namespace necessary_and_sufficient_condition_l202_202500

theorem necessary_and_sufficient_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (m + n > m * n) ↔ (m = 1 ∨ n = 1) := by
  sorry

end necessary_and_sufficient_condition_l202_202500


namespace probability_product_lt_36_eq_25_over_36_l202_202541

-- Definitions based on the conditions identified
def Paco_numbers : Fin 6 := ⟨i, h⟩ where i : ℕ, h : i < 6
noncomputable def Manu_numbers : Fin 12 := ⟨ j, h'⟩ where j : ℕ, h' : j < 12

-- Probability calculation
noncomputable def probability_product_less_than_36 : ℚ :=
  let outcomes := (Finset.univ : Finset (ℕ × ℕ)).filter (λ p, p.1 * p.2 < 36)
  outcomes.card.toRational / (6 * 12)

-- The proof statement
theorem probability_product_lt_36_eq_25_over_36 :
  probability_product_less_than_36 = 25 / 36 :=
by sorry

end probability_product_lt_36_eq_25_over_36_l202_202541


namespace range_of_a_l202_202799

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) + 3 * x

def has_pos_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, (3 + a * Real.exp (a * x) = 0) ∧ (x > 0)

theorem range_of_a (a : ℝ) : has_pos_extremum a → a < -3 := by
  sorry

end range_of_a_l202_202799


namespace tom_total_seashells_l202_202426

-- Define the number of seashells Tom gave to Jessica.
def seashells_given_to_jessica : ℕ := 2

-- Define the number of seashells Tom still has.
def seashells_tom_has_now : ℕ := 3

-- Theorem stating that the total number of seashells Tom found is the sum of seashells_given_to_jessica and seashells_tom_has_now.
theorem tom_total_seashells : seashells_given_to_jessica + seashells_tom_has_now = 5 := 
by
  sorry

end tom_total_seashells_l202_202426


namespace y_intercept_of_line_is_minus_one_l202_202124

theorem y_intercept_of_line_is_minus_one : 
  (∀ x y : ℝ, y = 2 * x - 1 → y = -1) :=
by
  sorry

end y_intercept_of_line_is_minus_one_l202_202124


namespace probability_of_product_lt_36_l202_202539

noncomputable def probability_less_than_36 : ℚ :=
  ∑ p in Finset.range 1 7, ∑ m in Finset.range 1 13, if (p * m < 36) then 1 else 0

theorem probability_of_product_lt_36 :
  probability (paco_num: ℕ, h_paco: 1 ≤ paco_num ∧ paco_num ≤ 6) *
  probability (manu_num: ℕ, h_manu: 1 ≤ manu_num ∧ manu_num ≤ 12) *
  (if paco_num * manu_num < 36 then 1 else 0) = 7 / 9 := 
sorry

end probability_of_product_lt_36_l202_202539


namespace solution_set_linear_inequalities_l202_202907

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202907


namespace side_length_range_l202_202551

-- Define the inscribed circle diameter condition
def inscribed_circle_diameter (d : ℝ) (cir_diameter : ℝ) := cir_diameter = 1

-- Define inscribed square side condition
def inscribed_square_side (d side : ℝ) :=
  ∃ (triangle_ABC : Type) (AB AC BC : triangle_ABC → ℝ), 
    side = d ∧
    side < 1

-- Define the main theorem: The side length of the inscribed square lies within given bounds
theorem side_length_range (d : ℝ) :
  inscribed_circle_diameter d 1 → inscribed_square_side d d → (4/5) ≤ d ∧ d < 1 :=
by
  intros h1 h2
  sorry

end side_length_range_l202_202551


namespace factorial_divisibility_l202_202365

theorem factorial_divisibility 
  (n k : ℕ) 
  (p : ℕ) 
  [hp : Fact (Nat.Prime p)] 
  (h1 : 0 < n) 
  (h2 : 0 < k) 
  (h3 : p ^ k ∣ n!) : 
  (p! ^ k ∣ n!) :=
sorry

end factorial_divisibility_l202_202365


namespace students_interested_in_both_l202_202449

theorem students_interested_in_both (total_students interested_in_sports interested_in_entertainment not_interested interested_in_both : ℕ)
  (h_total_students : total_students = 1400)
  (h_interested_in_sports : interested_in_sports = 1250)
  (h_interested_in_entertainment : interested_in_entertainment = 952)
  (h_not_interested : not_interested = 60)
  (h_equation : not_interested + interested_in_both + (interested_in_sports - interested_in_both) + (interested_in_entertainment - interested_in_both) = total_students) :
  interested_in_both = 862 :=
by
  sorry

end students_interested_in_both_l202_202449


namespace sum_of_x_intersections_is_zero_l202_202323

-- Definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Definition for the x-coordinates of the intersection points with x-axis
def intersects_x_axis (f : ℝ → ℝ) (x_coords : List ℝ) : Prop :=
  (∀ x ∈ x_coords, f x = 0) ∧ (x_coords.length = 4)

-- Main theorem
theorem sum_of_x_intersections_is_zero 
  (f : ℝ → ℝ)
  (x_coords : List ℝ)
  (h1 : is_even_function f)
  (h2 : intersects_x_axis f x_coords) : 
  x_coords.sum = 0 :=
sorry

end sum_of_x_intersections_is_zero_l202_202323


namespace find_S_l202_202441

variable (R S T c : ℝ)
variable (h1 : R = c * (S^2 / T^2))
variable (c_value : c = 8)
variable (h2 : R = 2) (h3 : T = 2) (h4 : S = 1)
variable (R_new : R = 50) (T_new : T = 5)

theorem find_S : S = 12.5 := by
  sorry

end find_S_l202_202441


namespace total_legs_l202_202275

def total_heads : ℕ := 16
def num_cats : ℕ := 7
def cat_legs : ℕ := 4
def captain_legs : ℕ := 1
def human_legs : ℕ := 2

theorem total_legs : (num_cats * cat_legs + (total_heads - num_cats) * human_legs - human_legs + captain_legs) = 45 :=
by 
  -- Proof skipped
  sorry

end total_legs_l202_202275


namespace theater_ticket_sales_l202_202457

-- Definitions of the given constants and initialization
def R : ℕ := 25

-- Conditions based on the problem statement
def condition_horror (H : ℕ) := H = 3 * R + 18
def condition_action (A : ℕ) := A = 2 * R
def condition_comedy (C H : ℕ) := 4 * H = 5 * C

-- Desired outcomes based on the solutions
def desired_horror := 93
def desired_action := 50
def desired_comedy := 74

theorem theater_ticket_sales
  (H A C : ℕ)
  (h1 : condition_horror H)
  (h2 : condition_action A)
  (h3 : condition_comedy C H)
  : H = desired_horror ∧ A = desired_action ∧ C = desired_comedy :=
by {
    sorry
}

end theater_ticket_sales_l202_202457


namespace negation_proposition_l202_202166

theorem negation_proposition (x : ℝ) (hx : 0 < x) : x + 4 / x ≥ 4 :=
sorry

end negation_proposition_l202_202166


namespace alice_age_l202_202452

theorem alice_age (a m : ℕ) (h1 : a = m - 18) (h2 : a + m = 50) : a = 16 := by
  sorry

end alice_age_l202_202452


namespace vertex_angle_isosceles_triangle_l202_202342

theorem vertex_angle_isosceles_triangle (B V : ℝ) (h1 : 2 * B + V = 180) (h2 : B = 40) : V = 100 :=
by
  sorry

end vertex_angle_isosceles_triangle_l202_202342


namespace squares_of_roots_equation_l202_202757

theorem squares_of_roots_equation (a b x : ℂ) 
  (h : ab * x^2 - (a + b) * x + 1 = 0) : 
  a^2 * b^2 * x^2 - (a^2 + b^2) * x + 1 = 0 :=
sorry

end squares_of_roots_equation_l202_202757


namespace solution_set_system_of_inequalities_l202_202928

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202928


namespace eqidistant_point_on_x_axis_l202_202129

theorem eqidistant_point_on_x_axis (x : ℝ) : 
    (dist (x, 0) (-3, 0) = dist (x, 0) (2, 5)) → 
    x = 2 := by
  sorry

end eqidistant_point_on_x_axis_l202_202129


namespace problem_l202_202028

theorem problem (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + (1/r^4) = 7 := 
by
  sorry

end problem_l202_202028


namespace rotated_parabola_equation_l202_202545

def parabola_equation (x y : ℝ) : Prop := y = x^2 - 4 * x + 3

def standard_form (x y : ℝ) : Prop := y = (x - 2)^2 - 1

def after_rotation (x y : ℝ) : Prop := (y + 1)^2 = x - 2

theorem rotated_parabola_equation (x y : ℝ) (h : standard_form x y) : after_rotation x y :=
sorry

end rotated_parabola_equation_l202_202545


namespace correct_assignment_statements_l202_202453

-- Defining what constitutes an assignment statement in this context.
def is_assignment_statement (s : String) : Prop :=
  s ∈ ["x ← 1", "y ← 2", "z ← 3", "i ← i + 2"]

-- Given statements
def statements : List String :=
  ["x ← 1, y ← 2, z ← 3", "S^2 ← 4", "i ← i + 2", "x + 1 ← x"]

-- The Lean Theorem statement that these are correct assignment statements.
theorem correct_assignment_statements (s₁ s₃ : String) (h₁ : s₁ = "x ← 1, y ← 2, z ← 3") (h₃ : s₃ = "i ← i + 2") :
  is_assignment_statement s₁ ∧ is_assignment_statement s₃ :=
by
  sorry

end correct_assignment_statements_l202_202453


namespace smallest_n_for_nonzero_constant_term_l202_202031

theorem smallest_n_for_nonzero_constant_term : 
  ∃ n : ℕ, (∃ r : ℕ, n = 5 * r / 3) ∧ (n > 0) ∧ ∀ m : ℕ, (m > 0) → (∃ s : ℕ, m = 5 * s / 3) → n ≤ m :=
by sorry

end smallest_n_for_nonzero_constant_term_l202_202031


namespace solution_set_inequalities_l202_202893

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202893


namespace N_positive_l202_202026

def N (a b : ℝ) : ℝ :=
  4 * a^2 - 12 * a * b + 13 * b^2 - 6 * a + 4 * b + 13

theorem N_positive (a b : ℝ) : N a b > 0 :=
by
  sorry

end N_positive_l202_202026


namespace solve_inequalities_l202_202998

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202998


namespace polygon_sides_l202_202098

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l202_202098


namespace cos_14_pi_over_3_l202_202173

theorem cos_14_pi_over_3 : Real.cos (14 * Real.pi / 3) = -1 / 2 :=
by 
  -- Proof is omitted according to the instructions
  sorry

end cos_14_pi_over_3_l202_202173


namespace subtract_29_after_46_l202_202839

theorem subtract_29_after_46 (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 :=
sorry

end subtract_29_after_46_l202_202839


namespace isosceles_triangle_base_angle_l202_202040

theorem isosceles_triangle_base_angle (a b c : ℝ) (h : a + b + c = 180) (h_isosceles : b = c) (h_angle_a : a = 120) : b = 30 := 
by
  sorry

end isosceles_triangle_base_angle_l202_202040


namespace solution_set_of_inequality_l202_202855

theorem solution_set_of_inequality (x : ℝ) : 
  (3 * x - 4 > 2) → (x > 2) :=
by
  intro h
  sorry

end solution_set_of_inequality_l202_202855


namespace sum_of_coefficients_eq_one_l202_202708

theorem sum_of_coefficients_eq_one :
  ∀ x y : ℤ, (x - 2 * y) ^ 18 = (1 - 2 * 1) ^ 18 → (x - 2 * y) ^ 18 = 1 :=
by
  intros x y h
  sorry

end sum_of_coefficients_eq_one_l202_202708


namespace altitude_length_l202_202182

theorem altitude_length {s t : ℝ} 
  (A B C : ℝ × ℝ) 
  (hA : A = (-s, s^2))
  (hB : B = (s, s^2))
  (hC : C = (t, t^2))
  (h_parabola_A : A.snd = (A.fst)^2)
  (h_parabola_B : B.snd = (B.fst)^2)
  (h_parabola_C : C.snd = (C.fst)^2)
  (hyp_parallel : A.snd = B.snd)
  (right_triangle : (t + s) * (t - s) + (t^2 - s^2)^2 = 0) :
  (s^2 - (t^2)) = 1 :=
by
  sorry

end altitude_length_l202_202182


namespace cos_5theta_l202_202332

theorem cos_5theta (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (5*θ) = -93/3125 :=
sorry

end cos_5theta_l202_202332


namespace distance_between_sasha_and_kolya_when_sasha_finished_l202_202394

-- Definitions based on the problem conditions
def distance_sasha : ℝ := 100
def distance_lesha_when_sasha_finished : ℝ := 90
def distance_kolya_when_lesha_finished : ℝ := 90

def velocity_lesha (v_s : ℝ) : ℝ := 0.9 * v_s
def velocity_kolya (v_s : ℝ) : ℝ := 0.81 * v_s

-- Theorem statement
theorem distance_between_sasha_and_kolya_when_sasha_finished (v_s : ℝ) :
  distance_sasha - (velocity_kolya v_s * (distance_sasha / v_s)) = 19 :=
  by sorry

end distance_between_sasha_and_kolya_when_sasha_finished_l202_202394


namespace points_on_line_l202_202296

-- Define the two points the line connects
def P1 : (ℝ × ℝ) := (8, 10)
def P2 : (ℝ × ℝ) := (2, -2)

-- Define the candidate points
def A : (ℝ × ℝ) := (5, 4)
def E : (ℝ × ℝ) := (1, -4)

-- Define the line equation, given the slope and y-intercept
def line (x : ℝ) : ℝ := 2 * x - 6

theorem points_on_line :
  (A.snd = line A.fst) ∧ (E.snd = line E.fst) :=
by
  sorry

end points_on_line_l202_202296


namespace third_podcast_length_correct_l202_202693

def first_podcast_length : ℕ := 45
def fourth_podcast_length : ℕ := 60
def next_podcast_length : ℕ := 60
def total_drive_time : ℕ := 360

def second_podcast_length := 2 * first_podcast_length

def total_time_other_than_third := first_podcast_length + second_podcast_length + fourth_podcast_length + next_podcast_length

theorem third_podcast_length_correct :
  total_drive_time - total_time_other_than_third = 105 := by
  -- Proof goes here
  sorry

end third_podcast_length_correct_l202_202693


namespace maps_skipped_l202_202374

-- Definitions based on conditions
def total_pages := 372
def pages_read := 125
def pages_left := 231

-- Statement to be proven
theorem maps_skipped : total_pages - (pages_read + pages_left) = 16 :=
by
  sorry

end maps_skipped_l202_202374


namespace solution_set_of_inequalities_l202_202984

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202984


namespace factorize_diff_of_squares_l202_202631

theorem factorize_diff_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
  sorry

end factorize_diff_of_squares_l202_202631


namespace prime_squares_5000_9000_l202_202785

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_squares_5000_9000 : 
  ∃ (l : List ℕ), 
  (∀ p ∈ l, is_prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧ 
  l.length = 6 := 
by
  sorry

end prime_squares_5000_9000_l202_202785


namespace max_q_minus_r_839_l202_202848

theorem max_q_minus_r_839 : ∃ (q r : ℕ), (839 = 19 * q + r) ∧ (0 ≤ r ∧ r < 19) ∧ q - r = 41 :=
by
  sorry

end max_q_minus_r_839_l202_202848


namespace Sahil_purchase_price_l202_202376

theorem Sahil_purchase_price :
  ∃ P : ℝ, (1.5 * (P + 6000) = 25500) → P = 11000 :=
sorry

end Sahil_purchase_price_l202_202376


namespace harry_average_sleep_l202_202658

-- Conditions
def sleep_time_monday : ℕ × ℕ := (8, 15)
def sleep_time_tuesday : ℕ × ℕ := (7, 45)
def sleep_time_wednesday : ℕ × ℕ := (8, 10)
def sleep_time_thursday : ℕ × ℕ := (10, 25)
def sleep_time_friday : ℕ × ℕ := (7, 50)

-- Total sleep time calculation
def total_sleep_time : ℕ × ℕ :=
  let (h1, m1) := sleep_time_monday
  let (h2, m2) := sleep_time_tuesday
  let (h3, m3) := sleep_time_wednesday
  let (h4, m4) := sleep_time_thursday
  let (h5, m5) := sleep_time_friday
  (h1 + h2 + h3 + h4 + h5, m1 + m2 + m3 + m4 + m5)

-- Convert minutes to hours and minutes
def convert_minutes (mins : ℕ) : ℕ × ℕ :=
  (mins / 60, mins % 60)

-- Final total sleep time
def final_total_time : ℕ × ℕ :=
  let (total_hours, total_minutes) := total_sleep_time
  let (extra_hours, remaining_minutes) := convert_minutes total_minutes
  (total_hours + extra_hours, remaining_minutes)

-- Average calculation
def average_sleep_time : ℕ × ℕ :=
  let (total_hours, total_minutes) := final_total_time
  (total_hours / 5, (total_hours % 5) * 60 / 5 + total_minutes / 5)

-- The proof statement
theorem harry_average_sleep :
  average_sleep_time = (8, 29) :=
  by
    sorry

end harry_average_sleep_l202_202658


namespace covered_area_of_strips_l202_202286

/-- Four rectangular strips of paper, each 16 cm long and 2 cm wide, overlap on a table. 
    We need to prove that the total area of the table surface covered by these strips is 112 cm². --/

theorem covered_area_of_strips (length width : ℝ) (number_of_strips : ℕ) (intersections : ℕ) 
    (area_of_strip : ℝ) (total_area_without_overlap : ℝ) (overlap_area : ℝ) 
    (actual_covered_area : ℝ) :
  length = 16 →
  width = 2 →
  number_of_strips = 4 →
  intersections = 4 →
  area_of_strip = length * width →
  total_area_without_overlap = number_of_strips * area_of_strip →
  overlap_area = intersections * (width * width) →
  actual_covered_area = total_area_without_overlap - overlap_area →
  actual_covered_area = 112 := 
by
  intros
  sorry

end covered_area_of_strips_l202_202286


namespace inequality_proof_l202_202011

variable (x y z : ℝ)

theorem inequality_proof
  (h : x + 2*y + 3*z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 14 :=
by
  sorry

end inequality_proof_l202_202011


namespace roots_exist_l202_202777

theorem roots_exist (a : ℝ) : ∃ x : ℝ, a * x^2 - x = 0 := by
  sorry

end roots_exist_l202_202777


namespace other_x_intercept_l202_202637

noncomputable def quadratic_function_vertex :=
  ∃ (a b c : ℝ), ∀ (x : ℝ), (a ≠ 0) →
  (5, -3) = ((-b) / (2 * a), a * ((-b) / (2 * a))^2 + b * ((-b) / (2 * a)) + c) ∧
  (x = 1) ∧ (a * x^2 + b * x + c = 0) →
  ∃ (x2 : ℝ), x2 = 9

theorem other_x_intercept :
  quadratic_function_vertex :=
sorry

end other_x_intercept_l202_202637


namespace symmetric_line_origin_l202_202241

theorem symmetric_line_origin (a b : ℝ) :
  (∀ (m n : ℝ), a * m + 3 * n = 9 → -m + 3 * -n + b = 0) ↔ a = -1 ∧ b = -9 :=
by
  sorry

end symmetric_line_origin_l202_202241


namespace chord_length_on_parabola_eq_five_l202_202313

theorem chord_length_on_parabola_eq_five
  (A B : ℝ × ℝ)
  (hA : A.snd ^ 2 = 4 * A.fst)
  (hB : B.snd ^ 2 = 4 * B.fst)
  (hM : A.fst + B.fst = 3 ∧ A.snd + B.snd = 2 
     ∧ A.fst - B.fst = 0 ∧ A.snd - B.snd = 0) :
  dist A B = 5 :=
by
  -- Proof goes here
  sorry

end chord_length_on_parabola_eq_five_l202_202313


namespace system_solution_ratio_l202_202008

theorem system_solution_ratio (x y z : ℝ) (h_xyz_nonzero: x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h1 : x + (95/9)*y + 4*z = 0) (h2 : 4*x + (95/9)*y - 3*z = 0) (h3 : 3*x + 5*y - 4*z = 0) :
  (x * z) / (y ^ 2) = 175 / 81 := 
by sorry

end system_solution_ratio_l202_202008


namespace arithmetic_sequence_geometric_condition_l202_202184

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = 3)
  (h3 : ∃ k, a (k+3) * a k = (a (k+1)) * (a (k+2))) :
  a 2 = -9 :=
by
  sorry

end arithmetic_sequence_geometric_condition_l202_202184


namespace count_primes_with_squares_in_range_l202_202789

theorem count_primes_with_squares_in_range : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, Prime n ∧ 5000 < n^2 ∧ n^2 < 9000) ∧ 
    S.card = 5 :=
by
  sorry

end count_primes_with_squares_in_range_l202_202789


namespace expected_value_dodecahedral_die_is_6_5_l202_202581

noncomputable def expected_value_dodecahedral_die : ℝ :=
  let outcomes := finset.range 12 in
  1 / 12 * (finset.sum outcomes (λ i, (i + 1 : ℝ)))

theorem expected_value_dodecahedral_die_is_6_5 :
  expected_value_dodecahedral_die = 6.5 :=
by
  sorry

end expected_value_dodecahedral_die_is_6_5_l202_202581


namespace expected_value_fair_dodecahedral_die_l202_202585

theorem expected_value_fair_dodecahedral_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let probability := (1 : ℝ) / 12 in
  (probability * faces.sum) = 6.5 :=
by
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let probability := (1 : ℝ) / 12
  have h : faces.sum = 78 := rfl
  rw [h]
  sorry

end expected_value_fair_dodecahedral_die_l202_202585


namespace direct_proportion_point_l202_202669

theorem direct_proportion_point (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : y₁ = k * x₁) (hx₁ : x₁ = -1) (hy₁ : y₁ = 2) (hx₂ : x₂ = 1) (hy₂ : y₂ = -2) 
  : y₂ = k * x₂ := 
by
  -- sorry will skip the proof
  sorry

end direct_proportion_point_l202_202669


namespace correct_answers_count_l202_202263

theorem correct_answers_count
  (c w : ℕ)
  (h1 : c + w = 150)
  (h2 : 4 * c - 2 * w = 420) :
  c = 120 := by
  sorry

end correct_answers_count_l202_202263


namespace solve_inequalities_l202_202954

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202954


namespace skips_in_one_meter_l202_202076

variable (p q r s t u : ℕ)

theorem skips_in_one_meter (h1 : p * s * u = q * r * t) : 1 = (p * r * t) / (u * s * q) := by
  sorry

end skips_in_one_meter_l202_202076


namespace rick_iron_clothing_l202_202831

theorem rick_iron_clothing :
  let shirts_per_hour := 4
  let pants_per_hour := 3
  let jackets_per_hour := 2
  let hours_shirts := 3
  let hours_pants := 5
  let hours_jackets := 2
  let total_clothing := (shirts_per_hour * hours_shirts) + (pants_per_hour * hours_pants) + (jackets_per_hour * hours_jackets)
  total_clothing = 31 := by
  sorry

end rick_iron_clothing_l202_202831


namespace Nell_initial_cards_l202_202824

theorem Nell_initial_cards (n : ℕ) (h1 : n - 136 = 106) : n = 242 := 
by
  sorry

end Nell_initial_cards_l202_202824


namespace usual_time_is_49_l202_202429

variable (R T : ℝ)
variable (h1 : R > 0) -- Usual rate is positive
variable (h2 : T > 0) -- Usual time is positive
variable (condition : T * R = (T - 7) * (7 / 6 * R)) -- Main condition derived from the problem

theorem usual_time_is_49 (h1 : R > 0) (h2 : T > 0) (condition : T * R = (T - 7) * (7 / 6 * R)) : T = 49 := by
  sorry -- Proof goes here

end usual_time_is_49_l202_202429


namespace prime_square_count_l202_202787

-- Define the set of natural numbers whose squares lie between 5000 and 9000
def within_square_range (n : ℕ) : Prop := 5000 < n * n ∧ n * n < 9000

-- Define the predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the set of prime numbers whose squares lie within the range
def primes_within_range : Finset ℕ := (Finset.filter is_prime (Finset.Ico 71 95))

-- The main statement: the number of prime numbers whose squares are between 5000 and 9000
theorem prime_square_count : (primes_within_range.filter within_square_range).card = 6 :=
sorry

end prime_square_count_l202_202787


namespace fraction_relation_l202_202495

theorem fraction_relation 
  (m n p q : ℚ)
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / q = 1 / 14) : 
  m / q = 3 / 14 :=
by
  sorry

end fraction_relation_l202_202495


namespace profit_rate_l202_202253

variables (list_price : ℝ)
          (discount : ℝ := 0.95)
          (selling_increase : ℝ := 1.6)
          (inflation_rate : ℝ := 1.4)

theorem profit_rate (list_price : ℝ) : 
  (selling_increase / (discount * inflation_rate)) - 1 = 0.203 :=
by 
  sorry

end profit_rate_l202_202253


namespace solve_inequalities_l202_202994

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202994


namespace slope_line_point_l202_202601

theorem slope_line_point (m b : ℝ) (h_slope : m = 3) (h_point : 2 = m * 5 + b) : m + b = -10 :=
by
  sorry

end slope_line_point_l202_202601


namespace solution_set_of_inequalities_l202_202962

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202962


namespace value_of_c_l202_202317

theorem value_of_c (c : ℝ) : (∃ x : ℝ, x^2 + c * x - 36 = 0 ∧ x = -9) → c = 5 :=
by
  sorry

end value_of_c_l202_202317


namespace lcm_60_30_40_eq_120_l202_202006

theorem lcm_60_30_40_eq_120 : (Nat.lcm (Nat.lcm 60 30) 40) = 120 := 
sorry

end lcm_60_30_40_eq_120_l202_202006


namespace roots_cubic_sum_l202_202190

theorem roots_cubic_sum :
  (∃ x1 x2 x3 x4 : ℂ, (x1^4 + 5*x1^3 + 6*x1^2 + 5*x1 + 1 = 0) ∧
                       (x2^4 + 5*x2^3 + 6*x2^2 + 5*x2 + 1 = 0) ∧
                       (x3^4 + 5*x3^3 + 6*x3^2 + 5*x3 + 1 = 0) ∧
                       (x4^4 + 5*x4^3 + 6*x4^2 + 5*x4 + 1 = 0)) →
  (x1^3 + x2^3 + x3^3 + x4^3 = -54) :=
sorry

end roots_cubic_sum_l202_202190


namespace hare_wins_by_10_meters_l202_202508

def speed_tortoise := 3 -- meters per minute
def speed_hare_sprint := 12 -- meters per minute
def speed_hare_walk := 1 -- meters per minute
def time_total := 50 -- minutes
def time_hare_sprint := 10 -- minutes
def time_hare_walk := time_total - time_hare_sprint -- minutes

def distance_tortoise := speed_tortoise * time_total -- meters
def distance_hare := (speed_hare_sprint * time_hare_sprint) + (speed_hare_walk * time_hare_walk) -- meters

theorem hare_wins_by_10_meters : (distance_hare - distance_tortoise) = 10 := by
  -- Proof would go here
  sorry

end hare_wins_by_10_meters_l202_202508


namespace sasha_kolya_distance_l202_202388

theorem sasha_kolya_distance
  (v_S v_L v_K : ℝ) 
  (h1 : Lesha_dist = 100 - 10) 
  (h2 : Kolya_dist = 100 - 10) 
  (h3 : v_L = (90 / 100) * v_S) 
  (h4 : v_K = (90 / 100) * v_L) 
  : v_S * (100/v_S - 10/v_S) = 19 :=
by
  sorry

end sasha_kolya_distance_l202_202388


namespace solve_for_x_l202_202492

theorem solve_for_x : ∃ x : ℚ, -3 * x - 8 = 4 * x + 3 ∧ x = -11 / 7 :=
by
  sorry

end solve_for_x_l202_202492


namespace number_of_authors_l202_202808

/-- Define the number of books each author has and the total number of books. -/
def books_per_author : ℕ := 33
def total_books : ℕ := 198

/-- Main theorem stating that the number of authors Jack has is derived by dividing total books by the number of books per author. -/
theorem number_of_authors (n : ℕ) (h : total_books = n * books_per_author) : n = 6 := by
  sorry

end number_of_authors_l202_202808


namespace sum_xy_l202_202830

theorem sum_xy (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y + 10) : x + y = 14 ∨ x + y = -2 :=
sorry

end sum_xy_l202_202830


namespace Lily_books_on_Wednesday_l202_202686

noncomputable def booksMike : ℕ := 45

noncomputable def booksCorey : ℕ := 2 * booksMike

noncomputable def booksMikeGivenToLily : ℕ := 13

noncomputable def booksCoreyGivenToLily : ℕ := booksMikeGivenToLily + 5

noncomputable def booksEmma : ℕ := 28

noncomputable def booksEmmaGivenToLily : ℕ := booksEmma / 4

noncomputable def totalBooksLilyGot : ℕ := booksMikeGivenToLily + booksCoreyGivenToLily + booksEmmaGivenToLily

theorem Lily_books_on_Wednesday : totalBooksLilyGot = 38 := by
  sorry

end Lily_books_on_Wednesday_l202_202686


namespace expected_value_of_dodecahedral_die_is_6_5_l202_202580

noncomputable def expected_value_of_dodecahedral_die : ℝ := 
  ∑ i in Finset.range (12 + 1), (i : ℝ) / 12

theorem expected_value_of_dodecahedral_die_is_6_5 :
  expected_value_of_dodecahedral_die = 6.5 := sorry

end expected_value_of_dodecahedral_die_is_6_5_l202_202580


namespace count_primes_between_71_and_95_l202_202784

theorem count_primes_between_71_and_95 : 
  let primes := [71, 73, 79, 83, 89, 97] in
  let filtered_primes := primes.filter (λ p, 71 < p ∧ p < 95) in
  filtered_primes.length = 5 := 
by 
  sorry

end count_primes_between_71_and_95_l202_202784


namespace solution_set_of_linear_inequalities_l202_202918

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202918


namespace fairy_tale_island_counties_l202_202513

theorem fairy_tale_island_counties :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := initial_dwarves * 3
  let first_year_centaurs := initial_centaurs * 3

  let second_year_elves := first_year_elves * 4
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := first_year_centaurs * 4

  let third_year_elves := second_year_elves * 6
  let third_year_dwarves := second_year_dwarves * 6
  let third_year_centaurs := second_year_centaurs

  let total_counties := third_year_elves + third_year_dwarves + third_year_centaurs

  total_counties = 54 :=
by
  sorry

end fairy_tale_island_counties_l202_202513


namespace derivative_at_one_l202_202304

open Real

noncomputable def f (x : ℝ) : ℝ := exp x / x

theorem derivative_at_one : deriv f 1 = 0 :=
by
  sorry

end derivative_at_one_l202_202304


namespace caesars_rental_fee_l202_202410

theorem caesars_rental_fee (C : ℕ) 
  (hc : ∀ (n : ℕ), n = 60 → C + 30 * n = 500 + 35 * n) : 
  C = 800 :=
by
  sorry

end caesars_rental_fee_l202_202410


namespace problem_solution_l202_202796

theorem problem_solution (x : ℝ) (h1 : x = 12) (h2 : 5 + 7 / x = some_number - 5 / x) : some_number = 6 := 
by
  sorry

end problem_solution_l202_202796


namespace base_of_exponential_function_l202_202668

theorem base_of_exponential_function (a : ℝ) (h : ∀ x : ℝ, y = a^x) :
  (a > 1 ∧ (a - 1 / a = 1)) ∨ (0 < a ∧ a < 1 ∧ (1 / a - a = 1)) → 
  a = (1 + Real.sqrt 5) / 2 ∨ a = (Real.sqrt 5 - 1) / 2 :=
by sorry

end base_of_exponential_function_l202_202668


namespace total_value_of_coins_l202_202833

variables {p n : ℕ}

-- Ryan has 17 coins consisting of pennies and nickels
axiom coins_eq : p + n = 17

-- The number of pennies is equal to the number of nickels
axiom pennies_eq_nickels : p = n

-- Prove that the total value of Ryan's coins is 49 cents
theorem total_value_of_coins : (p * 1 + n * 5) = 49 :=
by sorry

end total_value_of_coins_l202_202833


namespace solve_inequalities_l202_202957

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202957


namespace chandu_work_days_l202_202154

theorem chandu_work_days (W : ℝ) (c : ℝ) 
  (anand_rate : ℝ := W / 7) 
  (bittu_rate : ℝ := W / 8) 
  (chandu_rate : ℝ := W / c) 
  (completed_in_7_days : 3 * anand_rate + 2 * bittu_rate + 2 * chandu_rate = W) : 
  c = 7 :=
by
  sorry

end chandu_work_days_l202_202154


namespace solution_set_inequalities_l202_202882

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202882


namespace sum_of_coordinates_of_point_B_l202_202067

theorem sum_of_coordinates_of_point_B
  (A : ℝ × ℝ) (hA : A = (0, 0))
  (B : ℝ × ℝ) (hB : ∃ x : ℝ, B = (x, 3))
  (slope_AB : ∃ x : ℝ, (3 - 0)/(x - 0) = 3/4) :
  (∃ x : ℝ, B = (x, 3)) ∧ x + 3 = 7 :=
by
  sorry

end sum_of_coordinates_of_point_B_l202_202067


namespace number_line_4_units_away_l202_202177

theorem number_line_4_units_away (x : ℝ) : |x + 3.2| = 4 ↔ (x = 0.8 ∨ x = -7.2) :=
by
  sorry

end number_line_4_units_away_l202_202177


namespace solution_set_l202_202875

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202875


namespace range_of_m_l202_202644

namespace MathProof

def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - m * x + m - 1 = 0 }

theorem range_of_m (m : ℝ) (h : A ∪ (B m) = A) : m = 3 :=
  sorry

end MathProof

end range_of_m_l202_202644


namespace valid_n_values_l202_202528

theorem valid_n_values (n x y : ℤ) (h1 : n * (x - 3) = y + 3) (h2 : x + n = 3 * (y - n)) :
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7 :=
sorry

end valid_n_values_l202_202528


namespace larger_number_value_l202_202472

theorem larger_number_value (L S : ℕ) (h1 : L - S = 20775) (h2 : L = 23 * S + 143) : L = 21713 :=
sorry

end larger_number_value_l202_202472


namespace arithmetic_sequence_a6_l202_202674

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_root1 : ∃ x : ℝ, x^2 + 12 * x - 8 = 0 ∧ a 2 = x)
  (h_root2 : ∃ x : ℝ, x^2 + 12 * x - 8 = 0 ∧ a 10 = x) : 
  a 6 = -6 := 
by
  sorry

end arithmetic_sequence_a6_l202_202674


namespace solution_of_phi_l202_202188

theorem solution_of_phi 
    (φ : ℝ) 
    (H : ∃ k : ℤ, 2 * (π / 6) + φ = k * π) :
    φ = - (π / 3) := 
sorry

end solution_of_phi_l202_202188


namespace length_of_each_part_l202_202607

-- Definitions from the conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def number_of_parts : ℕ := 4

-- Proof statement
theorem length_of_each_part : total_length_in_inches / number_of_parts = 20 :=
by
  sorry

end length_of_each_part_l202_202607


namespace upper_left_region_l202_202807

theorem upper_left_region (t : ℝ) : (2 - 2 * t + 4 ≤ 0) → (t ≤ 3) :=
by
  sorry

end upper_left_region_l202_202807


namespace inequality_solution_set_l202_202564

theorem inequality_solution_set :
  { x : ℝ | (3 * x + 1) / (x - 2) ≤ 0 } = { x : ℝ | -1/3 ≤ x ∧ x < 2 } :=
sorry

end inequality_solution_set_l202_202564


namespace solution_set_system_of_inequalities_l202_202925

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202925


namespace find_d_over_a_l202_202840

variable (a b c d : ℚ)

-- Conditions
def condition1 : Prop := a / b = 8
def condition2 : Prop := c / b = 4
def condition3 : Prop := c / d = 2 / 3

-- Theorem statement
theorem find_d_over_a (h1 : condition1 a b) (h2 : condition2 c b) (h3 : condition3 c d) : d / a = 3 / 4 :=
by
  -- Proof is omitted
  sorry

end find_d_over_a_l202_202840


namespace angle_quadrant_l202_202346

def same_terminal_side (θ α : ℝ) (k : ℤ) : Prop :=
  θ = α + 360 * k

def in_first_quadrant (α : ℝ) : Prop :=
  0 < α ∧ α < 90

theorem angle_quadrant (θ : ℝ) (k : ℤ) (h : same_terminal_side θ 12 k) : in_first_quadrant 12 :=
  by
    sorry

end angle_quadrant_l202_202346


namespace pages_copied_for_25_dollars_l202_202211

def cost_per_page := 3
def total_cents := 25 * 100

theorem pages_copied_for_25_dollars : (total_cents div cost_per_page) = 833 :=
by sorry

end pages_copied_for_25_dollars_l202_202211


namespace games_in_each_box_l202_202466

theorem games_in_each_box (start_games sold_games total_boxes remaining_games games_per_box : ℕ) 
  (h_start: start_games = 35) (h_sold: sold_games = 19) (h_boxes: total_boxes = 2) 
  (h_remaining: remaining_games = start_games - sold_games) 
  (h_per_box: games_per_box = remaining_games / total_boxes) : games_per_box = 8 :=
by
  sorry

end games_in_each_box_l202_202466


namespace solution_set_inequalities_l202_202883

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202883


namespace solution_set_l202_202871

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202871


namespace cut_wood_into_5_pieces_l202_202742

-- Definitions
def pieces_to_cuts (pieces : ℕ) : ℕ := pieces - 1
def time_per_cut (total_time : ℕ) (cuts : ℕ) : ℕ := total_time / cuts
def total_time_for_pieces (pieces : ℕ) (time_per_cut : ℕ) : ℕ := (pieces_to_cuts pieces) * time_per_cut

-- Given conditions
def conditions : Prop :=
  pieces_to_cuts 4 = 3 ∧
  time_per_cut 24 (pieces_to_cuts 4) = 8

-- Problem statement
theorem cut_wood_into_5_pieces (h : conditions) : total_time_for_pieces 5 8 = 32 :=
by sorry

end cut_wood_into_5_pieces_l202_202742


namespace base_height_is_two_inches_l202_202161

noncomputable def height_sculpture_feet : ℝ := 2 + (10 / 12)
noncomputable def combined_height_feet : ℝ := 3
noncomputable def base_height_feet : ℝ := combined_height_feet - height_sculpture_feet
noncomputable def base_height_inches : ℝ := base_height_feet * 12

theorem base_height_is_two_inches :
  base_height_inches = 2 := by
  sorry

end base_height_is_two_inches_l202_202161


namespace combined_rocket_height_l202_202351

theorem combined_rocket_height :
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 :=
by
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  sorry

end combined_rocket_height_l202_202351


namespace incorrect_major_premise_l202_202843

noncomputable def Line := Type
noncomputable def Plane := Type

-- Conditions: Definitions
variable (b a : Line) (α : Plane)

-- Assumption: Line b is parallel to Plane α
axiom parallel_to_plane (p : Line) (π : Plane) : Prop

-- Assumption: Line a is in Plane α
axiom line_in_plane (l : Line) (π : Plane) : Prop

-- Define theorem stating the incorrect major premise
theorem incorrect_major_premise 
  (hb_par_α : parallel_to_plane b α)
  (ha_in_α : line_in_plane a α) : ¬ (parallel_to_plane b α → ∀ l, line_in_plane l α → b = l) := 
sorry

end incorrect_major_premise_l202_202843


namespace linear_inequalities_solution_l202_202862

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202862


namespace probability_product_lt_36_eq_25_over_36_l202_202542

-- Definitions based on the conditions identified
def Paco_numbers : Fin 6 := ⟨i, h⟩ where i : ℕ, h : i < 6
noncomputable def Manu_numbers : Fin 12 := ⟨ j, h'⟩ where j : ℕ, h' : j < 12

-- Probability calculation
noncomputable def probability_product_less_than_36 : ℚ :=
  let outcomes := (Finset.univ : Finset (ℕ × ℕ)).filter (λ p, p.1 * p.2 < 36)
  outcomes.card.toRational / (6 * 12)

-- The proof statement
theorem probability_product_lt_36_eq_25_over_36 :
  probability_product_less_than_36 = 25 / 36 :=
by sorry

end probability_product_lt_36_eq_25_over_36_l202_202542


namespace find_initial_mangoes_l202_202733

-- Define the initial conditions
def initial_apples : Nat := 7
def initial_oranges : Nat := 8
def apples_taken : Nat := 2
def oranges_taken : Nat := 2 * apples_taken
def remaining_fruits : Nat := 14
def mangoes_remaining (M : Nat) : Nat := M / 3

-- Define the problem statement
theorem find_initial_mangoes (M : Nat) (hM : 7 - apples_taken + 8 - oranges_taken + mangoes_remaining M = remaining_fruits) : M = 15 :=
by
  sorry

end find_initial_mangoes_l202_202733


namespace jennifer_total_miles_l202_202349

theorem jennifer_total_miles (d1 d2 : ℕ) (h1 : d1 = 5) (h2 : d2 = 15) :
  2 * d1 + 2 * d2 = 40 :=
by 
  rw [h1, h2];
  norm_num

end jennifer_total_miles_l202_202349


namespace range_of_m_l202_202641

open Set Real

noncomputable def f (x m : ℝ) : ℝ := abs (x^2 - 4 * x + 9 - 2 * m) + 2 * m

theorem range_of_m
  (h1 : ∀ x ∈ Icc (0 : ℝ) 4, f x m ≤ 9) : m ≤ 7 / 2 :=
by
  sorry

end range_of_m_l202_202641


namespace solve_inequalities_l202_202952

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202952


namespace solution_set_linear_inequalities_l202_202905

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202905


namespace solution_set_inequalities_l202_202892

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202892


namespace necessary_condition_for_line_passes_quadrants_l202_202243

theorem necessary_condition_for_line_passes_quadrants (m n : ℝ) (h_line : ∀ x : ℝ, x * (m / n) - (1 / n) < 0 ∨ x * (m / n) - (1 / n) > 0) : m * n < 0 :=
by
  sorry

end necessary_condition_for_line_passes_quadrants_l202_202243


namespace petya_wins_prize_probability_atleast_one_wins_probability_l202_202689

/-- Petya and 9 other people each roll a fair six-sided die. 
    A player wins a prize if they roll a number that nobody else rolls more than once.-/
theorem petya_wins_prize_probability : (5 / 6) ^ 9 = 0.194 :=
sorry

/-- The probability that at least one player gets a prize in the game where Petya and
    9 others roll a fair six-sided die is 0.919. -/
theorem atleast_one_wins_probability : 1 - (1 / 6) ^ 9 = 0.919 :=
sorry

end petya_wins_prize_probability_atleast_one_wins_probability_l202_202689


namespace exponent_multiplication_identity_l202_202720

theorem exponent_multiplication_identity : 2^4 * 3^2 * 5^2 * 7 = 6300 := sorry

end exponent_multiplication_identity_l202_202720


namespace subject_difference_l202_202062

-- Define the problem in terms of conditions and question
theorem subject_difference (C R M : ℕ) (hC : C = 10) (hR : R = C + 4) (hM : M + R + C = 41) : M - R = 3 :=
by
  -- Lean expects a proof here, we skip it with sorry
  sorry

end subject_difference_l202_202062


namespace mimi_spending_adidas_l202_202063

theorem mimi_spending_adidas
  (total_spending : ℤ)
  (nike_to_adidas_ratio : ℤ)
  (adidas_to_skechers_ratio : ℤ)
  (clothes_spending : ℤ)
  (eq1 : total_spending = 8000)
  (eq2 : nike_to_adidas_ratio = 3)
  (eq3 : adidas_to_skechers_ratio = 5)
  (eq4 : clothes_spending = 2600) :
  ∃ A : ℤ, A + nike_to_adidas_ratio * A + adidas_to_skechers_ratio * A + clothes_spending = total_spending ∧ A = 600 := by
  sorry

end mimi_spending_adidas_l202_202063


namespace distance_between_sasha_and_kolya_is_19_meters_l202_202402

theorem distance_between_sasha_and_kolya_is_19_meters
  (v_S v_L v_K : ℝ)
  (h1 : v_L = 0.9 * v_S)
  (h2 : v_K = 0.81 * v_S)
  (h3 : ∀ t_S : ℝ, t_S = 100 / v_S) :
  (∀ t_S : ℝ, 100 - v_K * t_S = 19) :=
by
  intros t_S
  have vL_defined : v_L = 0.9 * v_S := h1
  have vK_defined : v_K = 0.81 * v_S := h2
  have time_S : t_S = 100 / v_S := h3 t_S
  sorry

end distance_between_sasha_and_kolya_is_19_meters_l202_202402


namespace solution_set_l202_202872

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202872


namespace clara_age_l202_202716

theorem clara_age (x : ℕ) (n m : ℕ) (h1 : x - 2 = n^2) (h2 : x + 3 = m^3) : x = 123 :=
by sorry

end clara_age_l202_202716


namespace quotient_base4_l202_202001

def base4_to_base10 (n : ℕ) : ℕ :=
  n % 10 + 4 * (n / 10 % 10) + 4^2 * (n / 100 % 10) + 4^3 * (n / 1000)

def base10_to_base4 (n : ℕ) : ℕ :=
  let rec convert (n acc : ℕ) : ℕ :=
    if n < 4 then n * acc
    else convert (n / 4) ((n % 4) * acc * 10 + acc)
  convert n 1

theorem quotient_base4 (a b : ℕ) (h1 : a = 2313) (h2 : b = 13) :
  base10_to_base4 ((base4_to_base10 a) / (base4_to_base10 b)) = 122 :=
by
  sorry

end quotient_base4_l202_202001


namespace solve_proof_problem_1_solve_proof_problem_2_l202_202163

noncomputable def proof_problem_1 (a b : ℝ) : Prop :=
  ((a^(3/2) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a * b^(5/6))) = -9 * a

noncomputable def proof_problem_2 : Prop :=
  (real.log 3 / (2 * real.log 2) * (real.log 2 / (2 * real.log 3)) - real.log (32^(1/4)) / real.log (1/2)) = 11 / 8

theorem solve_proof_problem_1 (a b : ℝ) : proof_problem_1 a b := by
  sorry

theorem solve_proof_problem_2 : proof_problem_2 := by
  sorry

end solve_proof_problem_1_solve_proof_problem_2_l202_202163


namespace solution_set_system_of_inequalities_l202_202926

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202926


namespace fairy_tale_counties_l202_202515

theorem fairy_tale_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year := 
    (initial_elf_counties, initial_dwarf_counties * 3, initial_centaur_counties * 3) in
  let second_year := 
    (first_year.1 * 4, first_year.2, first_year.3 * 4) in
  let third_year := 
    (second_year.1 * 6, second_year.2 * 6, second_year.3) in
  third_year.1 + third_year.2 + third_year.3 = 54 :=
by
  sorry

end fairy_tale_counties_l202_202515


namespace solution_set_linear_inequalities_l202_202940

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202940


namespace base_length_of_isosceles_l202_202550

-- Define the lengths of the sides and the perimeter of the triangle.
def side_length1 : ℝ := 10
def side_length2 : ℝ := 10
def perimeter : ℝ := 35

-- Define the problem statement to prove the length of the base.
theorem base_length_of_isosceles (b : ℝ) 
  (h1 : side_length1 = 10) 
  (h2 : side_length2 = 10) 
  (h3 : perimeter = 35) : b = 15 :=
by
  -- Skip the proof.
  sorry

end base_length_of_isosceles_l202_202550


namespace simplify_power_l202_202694

theorem simplify_power (z : ℂ) (h₁ : z = (1 + complex.I) / (1 - complex.I)) : z ^ 1002 = -1 :=
by 
  sorry

end simplify_power_l202_202694


namespace regression_coeff_nonzero_l202_202639

theorem regression_coeff_nonzero (a b r : ℝ) (h : b = 0 → r = 0) : b ≠ 0 :=
sorry

end regression_coeff_nonzero_l202_202639


namespace abs_eq_zero_solve_l202_202335

theorem abs_eq_zero_solve (a b : ℚ) (h : |a - (1/2 : ℚ)| + |b + 5| = 0) : a + b = -9 / 2 := 
by
  sorry

end abs_eq_zero_solve_l202_202335


namespace fraction_is_determined_l202_202599

theorem fraction_is_determined (y x : ℕ) (h1 : y * 3 = x - 1) (h2 : (y + 4) * 2 = x) : 
  y = 7 ∧ x = 22 :=
by
  sorry

end fraction_is_determined_l202_202599


namespace solution_set_of_linear_inequalities_l202_202919

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202919


namespace math_problem_l202_202731

theorem math_problem : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end math_problem_l202_202731


namespace find_ray_solutions_l202_202175

noncomputable def polynomial (a x : ℝ) : ℝ :=
  x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3

theorem find_ray_solutions (a : ℝ) :
  (∀ x : ℝ, polynomial a x ≥ 0 → ∃ b : ℝ, ∀ y ≥ b, polynomial a y ≥ 0) ↔ a = 1 ∨ a = -1 :=
sorry

end find_ray_solutions_l202_202175


namespace total_distance_traveled_l202_202228

theorem total_distance_traveled (d d1 d2 d3 d4 d5 : ℕ) 
  (h1 : d1 = d)
  (h2 : d2 = 2 * d)
  (h3 : d3 = 40)
  (h4 : d = 2 * d3)
  (h5 : d4 = 2 * (d1 + d2 + d3))
  (h6 : d5 = 3 * d4 / 2) 
  : d1 + d2 + d3 + d4 + d5 = 1680 :=
by
  have hd : d = 80 := sorry
  have hd1 : d1 = 80 := sorry
  have hd2 : d2 = 160 := sorry
  have hd4 : d4 = 560 := sorry
  have hd5 : d5 = 840 := sorry
  sorry

end total_distance_traveled_l202_202228


namespace smallest_positive_integer_congruence_l202_202718

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 5 * x ≡ 14 [MOD 31] ∧ 0 < x ∧ x < 31 := 
sorry

end smallest_positive_integer_congruence_l202_202718


namespace polygon_sides_l202_202088

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l202_202088


namespace prob_product_lt_36_l202_202534

open ProbabilityTheory

noncomputable def P_event (n: ℕ) : dist ℕ := pmf.uniform (finset.range (n+1)).erase 0

theorem prob_product_lt_36 :
  let paco := P_event 6 in
  let manu := P_event 12 in
  P (λ x : ℕ × ℕ, x.1 * x.2 < 36) (paco.prod manu) = 67 / 72 :=
by sorry

end prob_product_lt_36_l202_202534


namespace disjoint_union_A_B_l202_202306

def A : Set ℕ := {x | x^2 - 3*x + 2 = 0}
def B : Set ℕ := {y | ∃ x ∈ A, y = x^2 - 2*x + 3}

def symmetric_difference (M P : Set ℕ) : Set ℕ :=
  {x | (x ∈ M ∨ x ∈ P) ∧ x ∉ M ∩ P}

theorem disjoint_union_A_B :
  symmetric_difference A B = {1, 3} := by
  sorry

end disjoint_union_A_B_l202_202306


namespace proof_y_pow_x_equal_1_by_9_l202_202195

theorem proof_y_pow_x_equal_1_by_9 
  (x y : ℝ)
  (h : (x - 2)^2 + abs (y + 1/3) = 0) :
  y^x = 1/9 := by
  sorry

end proof_y_pow_x_equal_1_by_9_l202_202195


namespace math_problem_l202_202334

theorem math_problem (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 :=
by
  -- The proof will be here
  sorry

end math_problem_l202_202334


namespace prob_product_lt_36_l202_202533

open ProbabilityTheory

noncomputable def P_event (n: ℕ) : dist ℕ := pmf.uniform (finset.range (n+1)).erase 0

theorem prob_product_lt_36 :
  let paco := P_event 6 in
  let manu := P_event 12 in
  P (λ x : ℕ × ℕ, x.1 * x.2 < 36) (paco.prod manu) = 67 / 72 :=
by sorry

end prob_product_lt_36_l202_202533


namespace solution_set_of_linear_inequalities_l202_202915

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202915


namespace polygon_sides_l202_202107

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202107


namespace integer_a_values_l202_202174

theorem integer_a_values (a : ℤ) :
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x - 7 = 0) ↔ a = -70 ∨ a = -29 ∨ a = -5 ∨ a = 3 :=
by
  sorry

end integer_a_values_l202_202174


namespace solution_set_of_inequalities_l202_202980

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202980


namespace a_plus_b_eq_neg2_l202_202366

noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

variable (a b : ℝ)

axiom h1 : f a = 1
axiom h2 : f b = 19

theorem a_plus_b_eq_neg2 : a + b = -2 :=
sorry

end a_plus_b_eq_neg2_l202_202366


namespace number_of_cuboids_painted_l202_202744

/--
Suppose each cuboid has 6 outer faces and Amelia painted a total of 36 faces.
Prove that the number of cuboids Amelia painted is 6.
-/
theorem number_of_cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) 
  (h1 : total_faces = 36) (h2 : faces_per_cuboid = 6) :
  total_faces / faces_per_cuboid = 6 := 
by {
  sorry
}

end number_of_cuboids_painted_l202_202744


namespace polygon_sides_l202_202103

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l202_202103


namespace solve_inequalities_l202_202948

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202948


namespace three_two_three_zero_zero_zero_zero_in_scientific_notation_l202_202282

theorem three_two_three_zero_zero_zero_zero_in_scientific_notation :
  3230000 = 3.23 * 10^6 :=
sorry

end three_two_three_zero_zero_zero_zero_in_scientific_notation_l202_202282


namespace time_to_hit_ground_l202_202844

theorem time_to_hit_ground : ∃ t : ℝ, 
  (y = -4.9 * t^2 + 7.2 * t + 8) → (y - (-0.6 * t) * t = 0) → t = 223/110 :=
by
  sorry

end time_to_hit_ground_l202_202844


namespace simplify_sqrt_l202_202696

theorem simplify_sqrt (a b : ℝ) (hb : b > 0) : 
  Real.sqrt (20 * a^3 * b^2) = 2 * a * b * Real.sqrt (5 * a) :=
by
  sorry

end simplify_sqrt_l202_202696


namespace original_number_proof_l202_202446

-- Define the conditions
variables (x y : ℕ)
-- Given conditions
def condition1 : Prop := y = 13
def condition2 : Prop := 7 * x + 5 * y = 146

-- Goal: the original number (sum of the parts x and y)
def original_number : ℕ := x + y

-- State the problem as a theorem
theorem original_number_proof (x y : ℕ) (h1 : condition1 y) (h2 : condition2 x y) : original_number x y = 24 := by
  -- The proof will be written here
  sorry

end original_number_proof_l202_202446


namespace sum_mod_9_l202_202753

theorem sum_mod_9 :
  (8 + 77 + 666 + 5555 + 44444 + 333333 + 2222222 + 11111111) % 9 = 3 := 
by sorry

end sum_mod_9_l202_202753


namespace smallest_special_integer_l202_202602

noncomputable def is_special (N : ℕ) : Prop :=
  N > 1 ∧ 
  (N % 8 = 1) ∧ 
  (2 * 8 ^ Nat.log N (8) / 2 > N / 8 ^ Nat.log N (8)) ∧ 
  (N % 9 = 1) ∧ 
  (2 * 9 ^ Nat.log N (9) / 2 > N / 9 ^ Nat.log N (9))

theorem smallest_special_integer : ∃ (N : ℕ), is_special N ∧ N = 793 :=
by 
  use 793
  sorry

end smallest_special_integer_l202_202602


namespace second_part_lent_years_l202_202613

theorem second_part_lent_years 
  (P1 P2 T : ℝ)
  (h1 : P1 + P2 = 2743)
  (h2 : P2 = 1688)
  (h3 : P1 * 0.03 * 8 = P2 * 0.05 * T) 
  : T = 3 :=
sorry

end second_part_lent_years_l202_202613


namespace probability_of_product_lt_36_l202_202538

noncomputable def probability_less_than_36 : ℚ :=
  ∑ p in Finset.range 1 7, ∑ m in Finset.range 1 13, if (p * m < 36) then 1 else 0

theorem probability_of_product_lt_36 :
  probability (paco_num: ℕ, h_paco: 1 ≤ paco_num ∧ paco_num ≤ 6) *
  probability (manu_num: ℕ, h_manu: 1 ≤ manu_num ∧ manu_num ≤ 12) *
  (if paco_num * manu_num < 36 then 1 else 0) = 7 / 9 := 
sorry

end probability_of_product_lt_36_l202_202538


namespace solve_inequalities_l202_202986

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202986


namespace chess_games_total_l202_202239

-- Conditions
def crowns_per_win : ℕ := 8
def uncle_wins : ℕ := 4
def draws : ℕ := 5
def father_net_gain : ℤ := 24

-- Let total_games be the total number of games played
def total_games : ℕ := sorry

-- Proof that under the given conditions, total_games equals 16
theorem chess_games_total :
  total_games = uncle_wins + (father_net_gain + uncle_wins * crowns_per_win) / crowns_per_win + draws := by
  sorry

end chess_games_total_l202_202239


namespace range_of_x_l202_202815

noncomputable def f (x : ℝ) : ℝ := (5 / (x^2)) - (3 * (x^2)) + 2

theorem range_of_x :
  { x : ℝ | f 1 < f (Real.log x / Real.log 3) } = { x : ℝ | (1 / 3) < x ∧ x < 1 ∨ 1 < x ∧ x < 3 } :=
by
  sorry

end range_of_x_l202_202815


namespace sqrt_14_bounds_l202_202467

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end sqrt_14_bounds_l202_202467


namespace find_c_l202_202315

-- Definitions for the conditions
def is_solution (x c : ℝ) : Prop := x^2 + c * x - 36 = 0

theorem find_c (c : ℝ) (h : is_solution (-9) c) : c = 5 :=
sorry

end find_c_l202_202315


namespace solution_set_linear_inequalities_l202_202943

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202943


namespace original_weight_of_marble_l202_202150

variable (W: ℝ) 

theorem original_weight_of_marble (h: 0.80 * 0.82 * 0.72 * W = 85.0176): W = 144 := 
by
  sorry

end original_weight_of_marble_l202_202150


namespace sum_of_possible_values_of_g_at_31_l202_202360

def f (x : ℝ) : ℝ := 4 * x^2 - 3
def g (x : ℝ) : ℝ := x^2 - x + 2

theorem sum_of_possible_values_of_g_at_31 : 
  (g (√(8.5)) + g (-√(8.5))) = 21 :=
by
  sorry

end sum_of_possible_values_of_g_at_31_l202_202360


namespace books_read_l202_202134

theorem books_read (total_books remaining_books read_books : ℕ)
  (h_total : total_books = 14)
  (h_remaining : remaining_books = 6)
  (h_eq : read_books = total_books - remaining_books) : read_books = 8 := 
by 
  sorry

end books_read_l202_202134


namespace polygon_sides_l202_202105

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202105


namespace manage_committee_combination_l202_202149

theorem manage_committee_combination : (Nat.choose 20 3) = 1140 := by
  sorry

end manage_committee_combination_l202_202149


namespace max_value_sqrt_expression_l202_202053

noncomputable def expression_max_value (a b: ℝ) : ℝ :=
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b))

theorem max_value_sqrt_expression : 
  ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 → expression_max_value a b ≤ 1 :=
by
  intros a b h
  sorry

end max_value_sqrt_expression_l202_202053


namespace calculate_expression_l202_202158

theorem calculate_expression :
  (Real.sqrt 3) ^ 0 + 2 ^ (-1 : ℤ) + Real.sqrt 2 * Real.cos (Real.pi / 4) - |(-1:ℝ) / 2| = 2 := 
by
  sorry

end calculate_expression_l202_202158


namespace factor_quadratic_l202_202171

theorem factor_quadratic (x : ℝ) (m n : ℝ) 
  (hm : m^2 = 16) (hn : n^2 = 25) (hmn : 2 * m * n = 40) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := 
by sorry

end factor_quadratic_l202_202171


namespace polygon_sides_l202_202094

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202094


namespace tanya_efficiency_increase_l202_202230

theorem tanya_efficiency_increase 
  (s_efficiency : ℝ := 1 / 10) (t_efficiency : ℝ := 1 / 8) :
  (((t_efficiency - s_efficiency) / s_efficiency) * 100) = 25 := 
by
  sorry

end tanya_efficiency_increase_l202_202230


namespace solve_inequalities_l202_202987

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202987


namespace find_solutions_l202_202303

theorem find_solutions :
  ∀ x y : Real, 
  (3 / 20) + abs (x - (15 / 40)) < (7 / 20) →
  y = 2 * x + 1 →
  (7 / 20) < x ∧ x < (2 / 5) ∧ (17 / 10) ≤ y ∧ y ≤ (11 / 5) :=
by
  intros x y h₁ h₂
  sorry

end find_solutions_l202_202303


namespace extremum_and_monotonicity_inequality_for_c_l202_202192

noncomputable def f (x α : ℝ) : ℝ := x * Real.log x - α * x + 1

theorem extremum_and_monotonicity (α : ℝ) (h_extremum : ∀ (x : ℝ), x = Real.exp 2 → f x α = 0) :
  (∃ α : ℝ, (∀ x : ℝ, x > Real.exp 2 → f x α > 0) ∧ (∀ x : ℝ, 0 < x ∧ x < Real.exp 2 → f x α < 0)) := sorry

theorem inequality_for_c (c : ℝ) (α : ℝ) (h_extremum : α = 3)
  (h_ineq : ∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 3 → f x α < 2 * c^2 - c) :
  (1 < c) ∨ (c < -1 / 2) := sorry

end extremum_and_monotonicity_inequality_for_c_l202_202192


namespace solve_inequalities_l202_202958

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202958


namespace number_of_marbles_drawn_l202_202271

noncomputable def probability_same_color (n : ℕ) :=
  let total_marbles := 13
  let prob_2_black := (4 / total_marbles) * (3 / (total_marbles - 1))
  let prob_2_red := (3 / total_marbles) * (2 / (total_marbles - 1))
  let prob_2_green := (6 / total_marbles) * (5 / (total_marbles - 1))
  prob_2_black + prob_2_red + prob_2_green

theorem number_of_marbles_drawn :
  ∃ n, n = 2 ∧ probability_same_color 2 = 0.3076923076923077 :=
sorry

end number_of_marbles_drawn_l202_202271


namespace select_subset_divisible_by_n_l202_202370

theorem select_subset_divisible_by_n (n : ℕ) (h : n > 0) (l : List ℤ) (hl : l.length = 2 * n - 1) :
  ∃ s : Finset ℤ, s.card = n ∧ (s.sum id) % n = 0 := 
sorry

end select_subset_divisible_by_n_l202_202370


namespace sum_of_two_primes_eq_53_l202_202042

theorem sum_of_two_primes_eq_53 : 
  ∀ (p q : ℕ), prime p ∧ prime q ∧ p + q = 53 → 0 :=
by 
  sorry

end sum_of_two_primes_eq_53_l202_202042


namespace monotonicity_intervals_range_of_c_l202_202191

noncomputable def f (x α : ℝ) : ℝ := x * Real.log x - α * x + 1

theorem monotonicity_intervals (α : ℝ) : 
  (∀ x, 0 < x ∧ x < Real.exp 2 → deriv (f x α) < 0) ∧ 
  (∀ x, x > Real.exp 2 → deriv (f x α) > 0) :=
by
  sorry

theorem range_of_c (c : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ Real.exp 3 → f x 3 < 2 * c^2 - c) → 
  (c > 1 ∨ c < -1/2) :=
by
  sorry

end monotonicity_intervals_range_of_c_l202_202191


namespace sara_change_l202_202260

def cost_of_first_book : ℝ := 5.5
def cost_of_second_book : ℝ := 6.5
def amount_given : ℝ := 20.0
def total_cost : ℝ := cost_of_first_book + cost_of_second_book
def change : ℝ := amount_given - total_cost

theorem sara_change : change = 8 :=
by
  have total_cost_correct : total_cost = 12.0 := by sorry
  have change_correct : change = amount_given - total_cost := by sorry
  show change = 8
  sorry

end sara_change_l202_202260


namespace total_trucks_l202_202687

theorem total_trucks {t : ℕ} (h1 : 2 * t + t = 300) : t = 100 := 
by sorry

end total_trucks_l202_202687


namespace doctor_lindsay_daily_income_l202_202259

def patients_per_hour_adult : ℕ := 4
def patients_per_hour_child : ℕ := 3
def cost_per_adult : ℕ := 50
def cost_per_child : ℕ := 25
def work_hours_per_day : ℕ := 8

theorem doctor_lindsay_daily_income : 
  (patients_per_hour_adult * cost_per_adult + patients_per_hour_child * cost_per_child) * work_hours_per_day = 2200 := 
by
  sorry

end doctor_lindsay_daily_income_l202_202259


namespace linear_inequalities_solution_l202_202861

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202861


namespace sub_two_three_l202_202622

theorem sub_two_three : 2 - 3 = -1 := 
by 
  sorry

end sub_two_three_l202_202622


namespace assistant_professor_pencils_l202_202750

theorem assistant_professor_pencils :
  ∀ (A B P : ℕ), 
    A + B = 7 →
    2 * A + P * B = 10 →
    A + 2 * B = 11 →
    P = 1 :=
by 
  sorry

end assistant_professor_pencils_l202_202750


namespace distance_between_sasha_and_kolya_is_19_meters_l202_202406

theorem distance_between_sasha_and_kolya_is_19_meters
  (v_S v_L v_K : ℝ)
  (h1 : v_L = 0.9 * v_S)
  (h2 : v_K = 0.81 * v_S)
  (h3 : ∀ t_S : ℝ, t_S = 100 / v_S) :
  (∀ t_S : ℝ, 100 - v_K * t_S = 19) :=
by
  intros t_S
  have vL_defined : v_L = 0.9 * v_S := h1
  have vK_defined : v_K = 0.81 * v_S := h2
  have time_S : t_S = 100 / v_S := h3 t_S
  sorry

end distance_between_sasha_and_kolya_is_19_meters_l202_202406


namespace dan_has_remaining_cards_l202_202464

-- Define the initial conditions
def initial_cards : ℕ := 97
def cards_sold_to_sam : ℕ := 15

-- Define the expected result
def remaining_cards (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- State the theorem to prove
theorem dan_has_remaining_cards : remaining_cards initial_cards cards_sold_to_sam = 82 :=
by
  -- This insertion is a placeholder for the proof
  sorry

end dan_has_remaining_cards_l202_202464


namespace minimum_distance_from_lattice_point_to_line_l202_202432

theorem minimum_distance_from_lattice_point_to_line :
  let distance (x y : ℤ) := |25 * x - 15 * y + 12| / (5 * Real.sqrt 34)
  ∃ (x y : ℤ), distance x y = Real.sqrt 34 / 85 :=
sorry

end minimum_distance_from_lattice_point_to_line_l202_202432


namespace sequence_a31_value_l202_202779

theorem sequence_a31_value 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h₀ : a 1 = 0) 
  (h₁ : ∀ n, a (n + 1) = a n + b n) 
  (h₂ : b 15 + b 16 = 15)
  (h₃ : ∀ m n : ℕ, (b n - b m) = (n - m) * (b 2 - b 1)) :
  a 31 = 225 :=
by
  sorry

end sequence_a31_value_l202_202779


namespace pet_store_dogs_count_l202_202148

def initial_dogs : ℕ := 2
def sunday_received_dogs : ℕ := 5
def sunday_sold_dogs : ℕ := 2
def monday_received_dogs : ℕ := 3
def monday_returned_dogs : ℕ := 1
def tuesday_received_dogs : ℕ := 4
def tuesday_sold_dogs : ℕ := 3

theorem pet_store_dogs_count :
  initial_dogs 
  + sunday_received_dogs - sunday_sold_dogs
  + monday_received_dogs + monday_returned_dogs
  + tuesday_received_dogs - tuesday_sold_dogs = 10 := 
sorry

end pet_store_dogs_count_l202_202148


namespace caesars_charge_l202_202412

theorem caesars_charge :
  ∃ (C : ℕ), (C + 30 * 60 = 500 + 35 * 60) ↔ (C = 800) :=
by
  sorry

end caesars_charge_l202_202412


namespace solution_set_inequalities_l202_202887

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202887


namespace solve_inequalities_l202_202953

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202953


namespace solution_set_of_linear_inequalities_l202_202908

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202908


namespace ratio_a3_a6_l202_202483

variable (a : ℕ → ℝ) (d : ℝ)
-- aₙ is an arithmetic sequence
variable (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d)
-- d ≠ 0
variable (h_d_nonzero : d ≠ 0)
-- a₃² = a₁a₉
variable (h_condition : (a 2)^2 = (a 0) * (a 8))

theorem ratio_a3_a6 : (a 2) / (a 5) = 1 / 2 :=
by
  -- Proof omitted
  sorry

end ratio_a3_a6_l202_202483


namespace sides_of_polygon_l202_202119

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l202_202119


namespace combined_tickets_l202_202751

-- Definitions for the initial conditions
def stuffedTigerPrice : ℝ := 43
def keychainPrice : ℝ := 5.5
def discount1 : ℝ := 0.20 * stuffedTigerPrice
def discountedTigerPrice : ℝ := stuffedTigerPrice - discount1
def ticketsLeftDave : ℝ := 55
def spentDave : ℝ := discountedTigerPrice + keychainPrice
def initialTicketsDave : ℝ := spentDave + ticketsLeftDave

def dinoToyPrice : ℝ := 65
def discount2 : ℝ := 0.15 * dinoToyPrice
def discountedDinoToyPrice : ℝ := dinoToyPrice - discount2
def ticketsLeftAlex : ℝ := 42
def spentAlex : ℝ := discountedDinoToyPrice
def initialTicketsAlex : ℝ := spentAlex + ticketsLeftAlex

-- Lean statement proving the combined number of tickets at the start
theorem combined_tickets {dave_alex_combined : ℝ} 
    (h1 : dave_alex_combined = initialTicketsDave + initialTicketsAlex) : 
    dave_alex_combined = 192.15 := 
by 
    -- Placeholder for the actual proof
    sorry

end combined_tickets_l202_202751


namespace GCF_30_90_75_l202_202255

theorem GCF_30_90_75 : Nat.gcd (Nat.gcd 30 90) 75 = 15 := by
  sorry

end GCF_30_90_75_l202_202255


namespace total_lambs_l202_202819

def num_initial_lambs : ℕ := 6
def num_baby_lambs_per_mother : ℕ := 2
def num_mothers : ℕ := 2
def traded_lambs : ℕ := 3
def extra_lambs : ℕ := 7

theorem total_lambs :
  num_initial_lambs + (num_baby_lambs_per_mother * num_mothers) - traded_lambs + extra_lambs = 14 :=
by
  sorry

end total_lambs_l202_202819


namespace point_P_in_first_quadrant_l202_202511

def point_P := (3, 2)
def first_quadrant (p : ℕ × ℕ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_P_in_first_quadrant : first_quadrant point_P :=
by
  sorry

end point_P_in_first_quadrant_l202_202511


namespace solution_set_linear_inequalities_l202_202896

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202896


namespace seven_b_equals_ten_l202_202493

theorem seven_b_equals_ten (a b : ℚ) (h1 : 5 * a + 2 * b = 0) (h2 : a = b - 2) : 7 * b = 10 := 
sorry

end seven_b_equals_ten_l202_202493


namespace player_B_wins_l202_202368

-- Here we define the scenario and properties from the problem statement.
def initial_pile1 := 100
def initial_pile2 := 252

-- Definition of a turn, conditions and the win condition based on the problem
structure Turn :=
  (pile1 : ℕ)
  (pile2 : ℕ)
  (player_A_turn : Bool)  -- True if it's player A's turn, False if it's player B's turn

-- The game conditions and strategy for determining the winner
def will_player_B_win (initial_pile1 initial_pile2 : ℕ) : Bool :=
  -- assuming the conditions are provided and correctly analyzed, 
  -- we directly state the known result according to the optimal strategies from the solution
  true  -- B wins as per the solution's analysis if both play optimally.

-- The final theorem stating Player B wins given the initial conditions with both playing optimally and A going first.
theorem player_B_wins : will_player_B_win initial_pile1 initial_pile2 = true :=
  sorry  -- Proof omitted.

end player_B_wins_l202_202368


namespace bricklayer_wall_l202_202596

/-- 
A bricklayer lays a certain number of meters of wall per day and works for a certain number of days.
Given the daily work rate and the number of days worked, this proof shows that the total meters of 
wall laid equals the product of the daily work rate and the number of days.
-/
theorem bricklayer_wall (daily_rate : ℕ) (days_worked : ℕ) (total_meters : ℕ) 
  (h1 : daily_rate = 8) (h2 : days_worked = 15) : total_meters = 120 :=
by {
  sorry
}

end bricklayer_wall_l202_202596


namespace range_of_m_for_subset_l202_202655

open Set

variable (m : ℝ)

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | (2 * m - 1) ≤ x ∧ x ≤ (m + 1)}

theorem range_of_m_for_subset (m : ℝ) : B m ⊆ A ↔ m ∈ Icc (-(1 / 2) : ℝ) (2 : ℝ) ∨ m > (2 : ℝ) :=
by
  sorry

end range_of_m_for_subset_l202_202655


namespace min_value_abs_function_l202_202476

theorem min_value_abs_function : ∃ x : ℝ, ∀ x, (|x - 4| + |x - 6|) ≥ 2 :=
by
  sorry

end min_value_abs_function_l202_202476


namespace Mandy_older_than_Jackson_l202_202817

variable (M J A : ℕ)

-- Given conditions
variables (h1 : J = 20)
variables (h2 : A = (3 * J) / 4)
variables (h3 : (M + 10) + (J + 10) + (A + 10) = 95)

-- Prove that Mandy is 10 years older than Jackson
theorem Mandy_older_than_Jackson : M - J = 10 :=
by
  sorry

end Mandy_older_than_Jackson_l202_202817


namespace find_x_value_l202_202061

theorem find_x_value
  (y₁ y₂ z₁ z₂ x₁ x w k : ℝ)
  (h₁ : y₁ = 3) (h₂ : z₁ = 2) (h₃ : x₁ = 1)
  (h₄ : y₂ = 6) (h₅ : z₂ = 5)
  (inv_rel : ∀ y z k, x = k * (z / y^2))
  (const_prod : ∀ x w, x * w = 1) :
  x = 5 / 8 :=
by
  -- omitted proof steps
  sorry

end find_x_value_l202_202061


namespace mouse_jump_distance_l202_202240

theorem mouse_jump_distance
  (g : ℕ) 
  (f : ℕ) 
  (m : ℕ)
  (h1 : g = 25)
  (h2 : f = g + 32)
  (h3 : m = f - 26) : 
  m = 31 :=
by
  sorry

end mouse_jump_distance_l202_202240


namespace remainder_division_l202_202245

theorem remainder_division (a b : ℕ) (h1 : a > b) (h2 : (a - b) % 6 = 5) : a % 6 = 5 :=
sorry

end remainder_division_l202_202245


namespace distance_between_sasha_and_kolya_is_19_meters_l202_202404

theorem distance_between_sasha_and_kolya_is_19_meters
  (v_S v_L v_K : ℝ)
  (h1 : v_L = 0.9 * v_S)
  (h2 : v_K = 0.81 * v_S)
  (h3 : ∀ t_S : ℝ, t_S = 100 / v_S) :
  (∀ t_S : ℝ, 100 - v_K * t_S = 19) :=
by
  intros t_S
  have vL_defined : v_L = 0.9 * v_S := h1
  have vK_defined : v_K = 0.81 * v_S := h2
  have time_S : t_S = 100 / v_S := h3 t_S
  sorry

end distance_between_sasha_and_kolya_is_19_meters_l202_202404


namespace fraction_of_donations_l202_202713

def max_donation_amount : ℝ := 1200
def total_money_raised : ℝ := 3750000
def donations_from_500_people : ℝ := 500 * max_donation_amount
def fraction_of_money_raised : ℝ := 0.4 * total_money_raised
def num_donors : ℝ := 1500

theorem fraction_of_donations (f : ℝ) :
  donations_from_500_people + num_donors * f * max_donation_amount = fraction_of_money_raised → f = 1 / 2 :=
by
  sorry

end fraction_of_donations_l202_202713


namespace staples_left_in_stapler_l202_202573

def initial_staples : ℕ := 50
def reports_stapled : ℕ := 3 * 12
def staples_per_report : ℕ := 1
def remaining_staples : ℕ := initial_staples - (reports_stapled * staples_per_report)

theorem staples_left_in_stapler : remaining_staples = 14 :=
by
  sorry

end staples_left_in_stapler_l202_202573


namespace average_temperature_for_july_4th_l202_202065

def avg_temperature_july_4th : ℤ := 
  let temperatures := [90, 90, 90, 79, 71]
  let sum := List.sum temperatures
  sum / temperatures.length

theorem average_temperature_for_july_4th :
  avg_temperature_july_4th = 84 := 
by
  sorry

end average_temperature_for_july_4th_l202_202065


namespace polygon_sides_l202_202112

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l202_202112


namespace polar_to_cartesian_l202_202193

-- Definitions for the polar coordinates conversion
noncomputable def polar_to_cartesian_eq (C : ℝ → ℝ → Prop) :=
  ∀ (ρ θ : ℝ), (ρ^2 * (1 + 3 * (Real.sin θ)^2) = 4) → C (ρ * (Real.cos θ)) (ρ * (Real.sin θ))

-- Define the Cartesian equation
def cartesian_eq (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 = 1)

-- The main theorem
theorem polar_to_cartesian 
  (C : ℝ → ℝ → Prop)
  (h : polar_to_cartesian_eq C) :
  ∀ x y : ℝ, C x y ↔ cartesian_eq x y :=
by
  sorry

end polar_to_cartesian_l202_202193


namespace problem_inequality_l202_202218

variables (a b c : ℝ)
open Real

theorem problem_inequality (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
sorry

end problem_inequality_l202_202218


namespace total_collected_funds_l202_202204

theorem total_collected_funds (A B T : ℕ) (hA : A = 5) (hB : B = 3 * A + 3) (h_quotient : B / 3 = 6) (hT : T = B * (B / 3) + A) : 
  T = 113 := 
by 
  sorry

end total_collected_funds_l202_202204


namespace sam_sandwich_shop_cost_l202_202287

theorem sam_sandwich_shop_cost :
  let sandwich_cost := 4
  let soda_cost := 3
  let fries_cost := 2
  let num_sandwiches := 3
  let num_sodas := 7
  let num_fries := 5
  let total_cost := num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_fries * fries_cost
  total_cost = 43 :=
by
  sorry

end sam_sandwich_shop_cost_l202_202287


namespace solution_set_linear_inequalities_l202_202946

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202946


namespace sara_gets_change_l202_202261

theorem sara_gets_change (cost_book1 cost_book2 money_given : ℝ) :
  cost_book1 = 5.5 ∧ cost_book2 = 6.5 ∧ money_given = 20 →
  money_given - (cost_book1 + cost_book2) = 8 :=
by
  intros h,
  rcases h with ⟨hb1, hb2, hg⟩,
  rw [hb1, hb2, hg],
  norm_num
  sorry -- added to make sure the code builds successfully

end sara_gets_change_l202_202261


namespace intersection_is_correct_l202_202660

namespace IntervalProofs

def setA := {x : ℝ | 3 * x^2 - 14 * x + 16 ≤ 0}
def setB := {x : ℝ | (3 * x - 7) / x > 0}

theorem intersection_is_correct :
  {x | 7 / 3 < x ∧ x ≤ 8 / 3} = setA ∩ setB :=
by
  sorry

end IntervalProofs

end intersection_is_correct_l202_202660


namespace caesars_rental_fee_l202_202411

theorem caesars_rental_fee (C : ℕ) 
  (hc : ∀ (n : ℕ), n = 60 → C + 30 * n = 500 + 35 * n) : 
  C = 800 :=
by
  sorry

end caesars_rental_fee_l202_202411


namespace solution_set_of_linear_inequalities_l202_202917

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202917


namespace lunchroom_tables_l202_202697

/-- Given the total number of students and the number of students per table, 
    prove the number of tables in the lunchroom. -/
theorem lunchroom_tables (total_students : ℕ) (students_per_table : ℕ) 
  (h_total : total_students = 204) (h_per_table : students_per_table = 6) : 
  total_students / students_per_table = 34 := 
by
  sorry

end lunchroom_tables_l202_202697


namespace gwen_books_collection_l202_202656

theorem gwen_books_collection :
  let mystery_books := 8 * 6
  let picture_books := 5 * 4
  let science_books := 4 * 7
  let non_fiction_books := 3 * 5
  let lent_mystery_books := 2
  let lent_science_books := 3
  let borrowed_picture_books := 5
  mystery_books - lent_mystery_books + picture_books - borrowed_picture_books + borrowed_picture_books + science_books - lent_science_books + non_fiction_books = 106 := by
  sorry

end gwen_books_collection_l202_202656


namespace f_sum_positive_l202_202651

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem f_sum_positive (x₁ x₂ x₃ : ℝ) (h₁₂ : x₁ + x₂ > 0) (h₂₃ : x₂ + x₃ > 0) (h₃₁ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := 
sorry

end f_sum_positive_l202_202651


namespace peanuts_remaining_l202_202425

def initial_peanuts := 220
def brock_fraction := 1 / 4
def bonita_fraction := 2 / 5
def carlos_peanuts := 17

noncomputable def peanuts_left := initial_peanuts - (initial_peanuts * brock_fraction + ((initial_peanuts - initial_peanuts * brock_fraction) * bonita_fraction)) - carlos_peanuts

theorem peanuts_remaining : peanuts_left = 82 :=
by
  sorry

end peanuts_remaining_l202_202425


namespace highest_power_of_two_factor_13_pow_4_minus_11_pow_4_l202_202415

theorem highest_power_of_two_factor_13_pow_4_minus_11_pow_4 :
  ∃ n : ℕ, n = 5 ∧ (2 ^ n ∣ (13 ^ 4 - 11 ^ 4)) ∧ ¬ (2 ^ (n + 1) ∣ (13 ^ 4 - 11 ^ 4)) :=
sorry

end highest_power_of_two_factor_13_pow_4_minus_11_pow_4_l202_202415


namespace tan_half_difference_l202_202359

-- Given two angles a and b with the following conditions
variables (a b : ℝ)
axiom cos_cond : (Real.cos a + Real.cos b = 3 / 5)
axiom sin_cond : (Real.sin a + Real.sin b = 2 / 5)

-- Prove that tan ((a - b) / 2) = 2 / 3
theorem tan_half_difference (a b : ℝ) (cos_cond : Real.cos a + Real.cos b = 3 / 5) 
  (sin_cond : Real.sin a + Real.sin b = 2 / 5) : 
  Real.tan ((a - b) / 2) = 2 / 3 := 
sorry

end tan_half_difference_l202_202359


namespace like_terms_to_exponents_matching_l202_202319

theorem like_terms_to_exponents_matching (n m : ℕ) (h1 : n = 3) (h2 : m = 3) : m^n = 27 := by
  sorry

end like_terms_to_exponents_matching_l202_202319


namespace minimize_quadratic_sum_l202_202812

theorem minimize_quadratic_sum (a b : ℝ) : 
  ∃ x : ℝ, y = (x-a)^2 + (x-b)^2 ∧ (∀ x', (x'-a)^2 + (x'-b)^2 ≥ y) ∧ x = (a + b) / 2 := 
sorry

end minimize_quadratic_sum_l202_202812


namespace graph_is_hyperbola_l202_202294

def graph_equation (x y : ℝ) : Prop := x^2 - 16 * y^2 - 8 * x + 64 = 0

theorem graph_is_hyperbola : ∃ (a b : ℝ), ∀ x y : ℝ, graph_equation x y ↔ (x - a)^2 / 48 - y^2 / 3 = -1 :=
by
  sorry

end graph_is_hyperbola_l202_202294


namespace balance_balls_l202_202529

theorem balance_balls (G Y B W : ℝ) (h₁ : 4 * G = 10 * B) (h₂ : 3 * Y = 8 * B) (h₃ : 8 * B = 6 * W) :
  5 * G + 5 * Y + 4 * W = 31.1 * B :=
by
  sorry

end balance_balls_l202_202529


namespace gate_distance_probability_correct_l202_202164

-- Define the number of gates
def num_gates : ℕ := 15

-- Define the distance between adjacent gates
def distance_between_gates : ℕ := 80

-- Define the maximum distance Dave can walk
def max_distance : ℕ := 320

-- Define the function that calculates the probability
def calculate_probability (num_gates : ℕ) (distance_between_gates : ℕ) (max_distance : ℕ) : ℚ :=
  let total_pairs := num_gates * (num_gates - 1)
  let valid_pairs :=
    2 * (4 + 5 + 6 + 7) + 7 * 8
  valid_pairs / total_pairs

-- Assert the relevant result and stated answer
theorem gate_distance_probability_correct :
  let m := 10
  let n := 21
  let probability := calculate_probability num_gates distance_between_gates max_distance
  m + n = 31 ∧ probability = (10 / 21 : ℚ) :=
by
  sorry

end gate_distance_probability_correct_l202_202164


namespace real_roots_m_range_find_value_of_m_l202_202013

-- Part 1: Prove the discriminant condition for real roots
theorem real_roots_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - (2 * m + 3) * x + m^2 + 2 = 0) ↔ m ≥ -1/12 := 
sorry

-- Part 2: Prove the value of m given the condition on roots
theorem find_value_of_m (m : ℝ) (x1 x2 : ℝ) 
  (h : x1^2 + x2^2 = 3 * x1 * x2 - 14)
  (h_roots : x^2 - (2 * m + 3) * x + m^2 + 2 = 0 → (x = x1 ∨ x = x2)) :
  m = 13 := 
sorry

end real_roots_m_range_find_value_of_m_l202_202013


namespace inequality_log_range_of_a_l202_202730

open Real

theorem inequality_log (x : ℝ) (h₀ : 0 < x) : 
  1 - 1 / x ≤ log x ∧ log x ≤ x - 1 := sorry

theorem range_of_a (a : ℝ) (h : ∀ (x : ℝ), 0 < x ∧ x ≤ 1 → a * (1 - x^2) + x^2 * log x ≥ 0) : 
  a ≥ 1/2 := sorry

end inequality_log_range_of_a_l202_202730


namespace Sarah_copy_total_pages_l202_202234

theorem Sarah_copy_total_pages (num_people : ℕ) (copies_per_person : ℕ) (pages_per_contract : ℕ)
  (h1 : num_people = 9) (h2 : copies_per_person = 2) (h3 : pages_per_contract = 20) :
  num_people * copies_per_person * pages_per_contract = 360 :=
by
  sorry

end Sarah_copy_total_pages_l202_202234


namespace ratio_of_perimeters_of_similar_triangles_l202_202649

theorem ratio_of_perimeters_of_similar_triangles (A1 A2 P1 P2 : ℝ) (h : A1 / A2 = 16 / 9) : P1 / P2 = 4 / 3 :=
sorry

end ratio_of_perimeters_of_similar_triangles_l202_202649


namespace range_of_a_l202_202020

open Set

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 3 → x^2 - a * x - a + 1 ≥ 0) ↔ a ≤ 5 / 2 :=
sorry

end range_of_a_l202_202020


namespace oil_cylinder_capacity_l202_202456

theorem oil_cylinder_capacity
  (C : ℚ) -- total capacity of the cylinder, given as a rational number
  (h1 : 3 / 4 * C + 4 = 4 / 5 * C) -- equation representing the condition of initial and final amounts of oil in the cylinder
  : C = 80 := -- desired result showing the total capacity

sorry

end oil_cylinder_capacity_l202_202456


namespace quadratic_no_real_roots_l202_202505

theorem quadratic_no_real_roots (m : ℝ) : ¬ ∃ x : ℝ, x^2 + 2 * x - m = 0 → m < -1 := 
by {
  sorry
}

end quadratic_no_real_roots_l202_202505


namespace robert_salary_loss_l202_202832

variable (S : ℝ)

theorem robert_salary_loss : 
  let decreased_salary := 0.80 * S
  let increased_salary := decreased_salary * 1.20
  let percentage_loss := 100 - (increased_salary / S) * 100
  percentage_loss = 4 :=
by
  sorry

end robert_salary_loss_l202_202832


namespace puzzles_sold_eq_36_l202_202077

def n_science_kits : ℕ := 45
def n_puzzles : ℕ := n_science_kits - 9

theorem puzzles_sold_eq_36 : n_puzzles = 36 := by
  sorry

end puzzles_sold_eq_36_l202_202077


namespace cheolsu_initial_number_l202_202458

theorem cheolsu_initial_number (x : ℚ) (h : x + (-5/12) - (-5/2) = 1/3) : x = -7/4 :=
by 
  sorry

end cheolsu_initial_number_l202_202458


namespace solve_inequalities_l202_202951

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202951


namespace race_distance_l202_202381

/-- Given that Sasha, Lesha, and Kolya start a 100m race simultaneously and run at constant velocities,
when Sasha finishes, Lesha is 10m behind, and when Lesha finishes, Kolya is 10m behind.
Prove that the distance between Sasha and Kolya when Sasha finishes is 19 meters. -/
theorem race_distance
    (v_S v_L v_K : ℝ)
    (h1 : 100 / v_S - 100 / v_L = 10 / v_L)
    (h2 : 100 / v_L - 100 / v_K = 10 / v_K) :
    100 - 81 = 19 :=
by
  sorry

end race_distance_l202_202381


namespace base_number_of_equation_l202_202794

theorem base_number_of_equation (y : ℕ) (b : ℕ) (h1 : 16 ^ y = b ^ 14) (h2 : y = 7) : b = 4 := 
by 
  sorry

end base_number_of_equation_l202_202794


namespace minimum_sum_of_box_dimensions_l202_202632

theorem minimum_sum_of_box_dimensions :
  ∃ (a b c : ℕ), a * b * c = 2310 ∧ a + b + c = 42 ∧ 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end minimum_sum_of_box_dimensions_l202_202632


namespace least_number_added_to_divisible_l202_202265

theorem least_number_added_to_divisible (n : ℕ) (k : ℕ) : n = 1789 → k = 11 → (n + k) % Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 4 3)) = 0 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end least_number_added_to_divisible_l202_202265


namespace volleyball_team_starters_l202_202531

-- Define the team and the triplets
def total_players : ℕ := 14
def triplet_count : ℕ := 3
def remaining_players : ℕ := total_players - triplet_count

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem
theorem volleyball_team_starters : 
  C total_players 6 - C remaining_players 3 = 2838 :=
by sorry

end volleyball_team_starters_l202_202531


namespace solve_arcsin_arccos_l202_202075

open Real

theorem solve_arcsin_arccos (x : ℝ) (h_condition : - (1 / 2 : ℝ) ≤ x ∧ x ≤ 1 / 2) :
  arcsin x + arcsin (2 * x) = arccos x ↔ x = 0 :=
sorry

end solve_arcsin_arccos_l202_202075


namespace dogs_eat_each_day_l202_202629

theorem dogs_eat_each_day (h1 : 0.125 + 0.125 = 0.25) : true := by
  sorry

end dogs_eat_each_day_l202_202629


namespace fault_line_movement_l202_202285

theorem fault_line_movement
  (moved_past_year : ℝ)
  (moved_year_before : ℝ)
  (h1 : moved_past_year = 1.25)
  (h2 : moved_year_before = 5.25) :
  moved_past_year + moved_year_before = 6.50 :=
by
  sorry

end fault_line_movement_l202_202285


namespace simplify_expression_l202_202072

open Real

theorem simplify_expression (α : ℝ) : 
  (cos (4 * α - π / 2) * sin (5 * π / 2 + 2 * α)) / ((1 + cos (2 * α)) * (1 + cos (4 * α))) = tan α :=
by
  sorry

end simplify_expression_l202_202072


namespace factor_quadratic_l202_202170

theorem factor_quadratic (x : ℝ) : (16 * x^2 - 40 * x + 25) = (4 * x - 5)^2 :=
by 
  sorry

end factor_quadratic_l202_202170


namespace find_circle_equation_l202_202305

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the equation of the asymptote
def asymptote (x y : ℝ) : Prop :=
  4 * x - 3 * y = 0

-- Define the given center of the circle
def center : ℝ × ℝ :=
  (5, 0)

-- Define the radius of the circle
def radius : ℝ :=
  4

-- Define the circle in center-radius form and expand it to standard form
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * x + 9 = 0

theorem find_circle_equation 
  (x y : ℝ) 
  (h : asymptote x y)
  (h_center : (x, y) = center) 
  (h_radius : radius = 4) : circle_eq x y :=
sorry

end find_circle_equation_l202_202305


namespace sasha_kolya_distance_l202_202384

theorem sasha_kolya_distance
  (v_S v_L v_K : ℝ) 
  (h1 : Lesha_dist = 100 - 10) 
  (h2 : Kolya_dist = 100 - 10) 
  (h3 : v_L = (90 / 100) * v_S) 
  (h4 : v_K = (90 / 100) * v_L) 
  : v_S * (100/v_S - 10/v_S) = 19 :=
by
  sorry

end sasha_kolya_distance_l202_202384


namespace greatest_brownies_produced_l202_202491

theorem greatest_brownies_produced (p side_length a b brownies : ℕ) :
  (4 * side_length = p) →
  (p = 40) →
  (brownies = side_length * side_length) →
  ((side_length - a - 2) * (side_length - b - 2) = 2 * (2 * (side_length - a) + 2 * (side_length - b) - 4)) →
  (a = 4) →
  (b = 4) →
  brownies = 100 :=
by
  intros h_perimeter h_perimeter_value h_brownies h_eq h_a h_b
  sorry

end greatest_brownies_produced_l202_202491


namespace Rahul_batting_average_l202_202069

theorem Rahul_batting_average 
  (A : ℕ) (current_matches : ℕ := 12) (new_matches : ℕ := 13) (scored_today : ℕ := 78) (new_average : ℕ := 54)
  (h1 : (A * current_matches + scored_today) = new_average * new_matches) : A = 52 := 
by
  sorry

end Rahul_batting_average_l202_202069


namespace true_proposition_l202_202314

def p : Prop := ∃ x₀ : ℝ, x₀^2 < x₀
def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem true_proposition : p ∧ q :=
by 
  sorry

end true_proposition_l202_202314


namespace JodiMilesFourthWeek_l202_202676

def JodiMilesFirstWeek := 1 * 6
def JodiMilesSecondWeek := 2 * 6
def JodiMilesThirdWeek := 3 * 6
def TotalMilesFirstThreeWeeks := JodiMilesFirstWeek + JodiMilesSecondWeek + JodiMilesThirdWeek
def TotalMilesFourWeeks := 60

def MilesInFourthWeek := TotalMilesFourWeeks - TotalMilesFirstThreeWeeks
def DaysInWeek := 6

theorem JodiMilesFourthWeek : (MilesInFourthWeek / DaysInWeek) = 4 := by
  sorry

end JodiMilesFourthWeek_l202_202676


namespace tan_angle_sum_l202_202180

theorem tan_angle_sum
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 :=
by
  sorry

end tan_angle_sum_l202_202180


namespace prime_squares_5000_9000_l202_202786

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_squares_5000_9000 : 
  ∃ (l : List ℕ), 
  (∀ p ∈ l, is_prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧ 
  l.length = 6 := 
by
  sorry

end prime_squares_5000_9000_l202_202786


namespace diff_of_squares_l202_202624

-- Define constants a and b
def a : ℕ := 65
def b : ℕ := 35

-- State the theorem to be proven using the difference of squares formula
theorem diff_of_squares : a^2 - b^2 = 3000 := by
  have h1 : a + b = 100 := by rfl -- Checking a + b
  have h2 : a - b = 30 := by rfl -- Checking a - b
  have h3 : (a+b)*(a-b) = 3000 := by
    rw [h1, h2] -- Substituting the values
    norm_num -- Simplifying the arithmetics
  exact h3 -- Providing the ultimate result

end diff_of_squares_l202_202624


namespace solution_set_of_inequalities_l202_202974

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202974


namespace quadrangular_pyramid_edge_length_l202_202712

theorem quadrangular_pyramid_edge_length :
  ∃ e : ℝ, 8 * e = 14.8 ∧ e = 1.85 :=
  sorry

end quadrangular_pyramid_edge_length_l202_202712


namespace solve_for_x_l202_202433

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : (9*x)^18 = (27*x)^9 ↔ x = 1/3 :=
by sorry

end solve_for_x_l202_202433


namespace problem_l202_202358

theorem problem (C D : ℝ) (h : ∀ x : ℝ, x ≠ 4 → 
  (C / (x - 4)) + D * (x + 2) = (-2 * x^3 + 8 * x^2 + 35 * x + 48) / (x - 4)) : 
  C + D = 174 :=
sorry

end problem_l202_202358


namespace students_walk_fraction_l202_202671

theorem students_walk_fraction
  (school_bus_fraction : ℚ := 1/3)
  (car_fraction : ℚ := 1/5)
  (bicycle_fraction : ℚ := 1/8) :
  (1 - (school_bus_fraction + car_fraction + bicycle_fraction) = 41/120) :=
by
  sorry

end students_walk_fraction_l202_202671


namespace prop_D_l202_202023

variable (a b : ℝ)

theorem prop_D (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
  by
    sorry

end prop_D_l202_202023


namespace sides_of_polygon_l202_202118

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l202_202118


namespace solution_set_inequalities_l202_202884

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202884


namespace ratio_x_y_l202_202179

theorem ratio_x_y (x y : ℝ) (h : (1/x - 1/y) / (1/x + 1/y) = 2023) : (x + y) / (x - y) = -1 := 
by
  sorry

end ratio_x_y_l202_202179


namespace difference_before_exchange_l202_202071

--Definitions
variables {S B : ℤ}

-- Conditions
axiom h1 : S - 2 = B + 2
axiom h2 : B > S

theorem difference_before_exchange : B - S = 2 :=
by
-- Proof will go here
sorry

end difference_before_exchange_l202_202071


namespace race_distance_l202_202380

/-- Given that Sasha, Lesha, and Kolya start a 100m race simultaneously and run at constant velocities,
when Sasha finishes, Lesha is 10m behind, and when Lesha finishes, Kolya is 10m behind.
Prove that the distance between Sasha and Kolya when Sasha finishes is 19 meters. -/
theorem race_distance
    (v_S v_L v_K : ℝ)
    (h1 : 100 / v_S - 100 / v_L = 10 / v_L)
    (h2 : 100 / v_L - 100 / v_K = 10 / v_K) :
    100 - 81 = 19 :=
by
  sorry

end race_distance_l202_202380


namespace value_of_g_at_neg2_l202_202721

def g (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_of_g_at_neg2 : g (-2) = 15 :=
by
  -- This is where the proof steps would go, but we'll skip it
  sorry

end value_of_g_at_neg2_l202_202721


namespace length_decrease_by_33_percent_l202_202572

theorem length_decrease_by_33_percent (L W L_new : ℝ) 
  (h1 : L * W = L_new * 1.5 * W) : 
  L_new = (2 / 3) * L ∧ ((1 - (2 / 3)) * 100 = 33.33) := 
by
  sorry

end length_decrease_by_33_percent_l202_202572


namespace ramsey_6_3_3_l202_202691

open Classical

theorem ramsey_6_3_3 (G : SimpleGraph (Fin 6)) :
  ∃ (A : Finset (Fin 6)), A.card = 3 ∧ (∀ (x y : Fin 6), x ∈ A → y ∈ A → x ≠ y → G.Adj x y) ∨ ∃ (B : Finset (Fin 6)), B.card = 3 ∧ (∀ (x y : Fin 6), x ∈ B → y ∈ B → x ≠ y → ¬ G.Adj x y) :=
by
  sorry

end ramsey_6_3_3_l202_202691


namespace linear_inequalities_solution_l202_202856

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202856


namespace solution_set_linear_inequalities_l202_202941

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202941


namespace mary_lambs_count_l202_202820

def initial_lambs : Nat := 6
def baby_lambs : Nat := 2 * 2
def traded_lambs : Nat := 3
def extra_lambs : Nat := 7

theorem mary_lambs_count : initial_lambs + baby_lambs - traded_lambs + extra_lambs = 14 := by
  sorry

end mary_lambs_count_l202_202820


namespace length_of_each_part_l202_202608

-- Definitions from the conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def number_of_parts : ℕ := 4

-- Proof statement
theorem length_of_each_part : total_length_in_inches / number_of_parts = 20 :=
by
  sorry

end length_of_each_part_l202_202608


namespace problem_statement_l202_202485

-- Definitions for given conditions
variables (a b m n x : ℤ)

-- Assuming conditions: a = -b, mn = 1, and |x| = 2
axiom opp_num : a = -b
axiom recip : m * n = 1
axiom abs_x : |x| = 2

-- Problem statement to prove
theorem problem_statement :
  -2 * m * n + (a + b) / 2023 + x * x = 2 :=
by 
  sorry

end problem_statement_l202_202485


namespace find_a_b_c_l202_202525

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 37) / 2 + 3 / 2)

theorem find_a_b_c :
  ∃ a b c : ℕ, (x^80 = 2 * x^78 + 8 * x^76 + 9 * x^74 - x^40 + a * x^36 + b * x^34 + c * x^30) ∧ (a + b + c = 151) :=
by
  sorry

end find_a_b_c_l202_202525


namespace solution_set_of_inequalities_l202_202976

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202976


namespace range_of_a_l202_202490

-- Define set A
def setA (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a^2 + 1

-- Define set B
def setB (x a : ℝ) : Prop := (x - 2) * (x - (3 * a + 1)) ≤ 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, setA x a → setB x a) ↔ (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) :=
sorry

end range_of_a_l202_202490


namespace solve_system_l202_202699

open Real

theorem solve_system :
  (∃ x y : ℝ, (sin x) ^ 2 + (cos y) ^ 2 = y ^ 4 ∧ (sin y) ^ 2 + (cos x) ^ 2 = x ^ 2) → 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) := by
  sorry

end solve_system_l202_202699


namespace smallest_positive_integer_is_53_l202_202165

theorem smallest_positive_integer_is_53 :
  ∃ a : ℕ, a > 0 ∧ a % 3 = 2 ∧ a % 4 = 1 ∧ a % 5 = 3 ∧ a = 53 :=
by
  sorry

end smallest_positive_integer_is_53_l202_202165


namespace solution_set_inequalities_l202_202886

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202886


namespace find_phi_l202_202554

theorem find_phi (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  (∀ x, 2 * Real.sin (2 * x + φ - π / 6) = 2 * Real.cos (2 * x)) → φ = 5 * π / 6 :=
by
  sorry

end find_phi_l202_202554


namespace solution_set_of_inequalities_l202_202964

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202964


namespace largest_divisor_of_five_consecutive_odds_l202_202587

theorem largest_divisor_of_five_consecutive_odds (n : ℕ) (hn : n % 2 = 0) :
    ∃ d, d = 15 ∧ ∀ m, (m = (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11)) → d ∣ m :=
sorry

end largest_divisor_of_five_consecutive_odds_l202_202587


namespace positive_difference_of_squares_l202_202122

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 18) : a^2 - b^2 = 1080 :=
by
  sorry

end positive_difference_of_squares_l202_202122


namespace inequality_proof_l202_202267

variables {a1 a2 a3 b1 b2 b3 : ℝ}

theorem inequality_proof (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) 
                         (h4 : 0 < b1) (h5 : 0 < b2) (h6 : 0 < b3):
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 
  ≥ 4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) := 
sorry

end inequality_proof_l202_202267


namespace determine_condition_l202_202024

theorem determine_condition (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) 
    (h_eq : (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) : 
    b + c = 12 :=
by
  sorry

end determine_condition_l202_202024


namespace find_quadratic_expression_find_range_of_m_find_t_value_l202_202771

-- Condition 1: Minimum value occurs at x = 2
-- Condition 2: Length of line segment on x-axis is 2
-- Function f(x) = ax^2 + bx + 3
def hasMinAt (a b : ℝ) (x : ℝ) : Prop :=
  let f := λ x, a * x^2 + b * x + 3
  ∀ x', f x' ≥ f x

def lengthOfLineSegmentOnXAxis (a b : ℝ) : Prop :=
  let f := λ x, a * x^2 + b * x + 3
  ∃ x1 x2, f x1 = 0 ∧ f x2 = 0 ∧ abs (x1 - x2) = 2

-- Question 1: Find the analytical expression for f(x)
theorem find_quadratic_expression (a b : ℝ) 
  (h1 : hasMinAt a b 2) 
  (h2 : lengthOfLineSegmentOnXAxis a b) :
  ∃ a b, a = 1 ∧ b = -4 ∧ (∀ x, a * x^2 + b * x + 3 = x^2 - 4 * x + 3) := sorry

-- g(x) = f(x) - mx
-- One zero in (0,2), another zero in (2,3)
-- Find the range of m
def g (a b m x : ℝ) : ℝ := (a * x^2 + b * x + 3) - m * x

theorem find_range_of_m (a b : ℝ) 
  (h1 : a = 1) 
  (h2 : b = -4)
  (h3 : ∃ x1 : ℝ, 0 < x1 ∧ x1 < 2 ∧ g a b (-4) x1 = 0)
  (h4 : ∃ x2 : ℝ, 2 < x2 ∧ x2 < 3 ∧ g a b (-4) x2 = 0) :
  -1 / 2 < m ∧ m < 0 := sorry

-- Minimum value of f(x) on [t, t + 1] is -1/2
-- Find the value of t
def minValueInInterval (a b t : ℝ) : Prop :=
  let f := λ x, a * x^2 + b * x + 3
  ∀ x ∈ set.Icc t (t+1), f x ≥ -1/2

theorem find_t_value (a b : ℝ)
  (h1 : a = 1)
  (h2 : b = -4)
  (h3 : minValueInInterval a b t) :
  t = 1 - real.sqrt 2 / 2 ∨ t = 2 + real.sqrt 2 / 2 := sorry

end find_quadratic_expression_find_range_of_m_find_t_value_l202_202771


namespace jessica_age_proof_l202_202051

-- Definitions based on conditions
def grandmother_age (j : ℚ) : ℚ := 15 * j
def age_difference (g j : ℚ) : Prop := g - j = 60

-- Proposed age of Jessica
def jessica_age : ℚ := 30 / 7

-- Main statement to prove
theorem jessica_age_proof : ∃ j : ℚ, grandmother_age j = 15 * j ∧ age_difference (grandmother_age j) j ∧ j = jessica_age :=
by sorry

end jessica_age_proof_l202_202051


namespace sasha_kolya_distance_l202_202387

theorem sasha_kolya_distance
  (v_S v_L v_K : ℝ) 
  (h1 : Lesha_dist = 100 - 10) 
  (h2 : Kolya_dist = 100 - 10) 
  (h3 : v_L = (90 / 100) * v_S) 
  (h4 : v_K = (90 / 100) * v_L) 
  : v_S * (100/v_S - 10/v_S) = 19 :=
by
  sorry

end sasha_kolya_distance_l202_202387


namespace minimum_value_func1_minimum_value_func2_l202_202595

-- Problem (1): 
theorem minimum_value_func1 (x : ℝ) (h : x > -1) : 
  (x + 4 / (x + 1) + 6) ≥ 9 :=
sorry

-- Problem (2): 
theorem minimum_value_func2 (x : ℝ) (h : x > 1) : 
  (x^2 + 8) / (x - 1) ≥ 8 :=
sorry

end minimum_value_func1_minimum_value_func2_l202_202595


namespace negation_correct_l202_202244

namespace NegationProof

-- Define the original proposition 
def orig_prop : Prop := ∃ x : ℝ, x ≤ 0

-- Define the negation of the original proposition
def neg_prop : Prop := ∀ x : ℝ, x > 0

-- The theorem we need to prove
theorem negation_correct : ¬ orig_prop = neg_prop := by
  sorry

end NegationProof

end negation_correct_l202_202244


namespace total_population_is_3311_l202_202279

-- Definitions based on the problem's conditions
def fewer_than_6000_inhabitants (L : ℕ) : Prop :=
  L < 6000

def more_girls_than_boys (girls boys : ℕ) : Prop :=
  girls = (11 * boys) / 10

def more_men_than_women (men women : ℕ) : Prop :=
  men = (23 * women) / 20

def more_children_than_adults (children adults : ℕ) : Prop :=
  children = (6 * adults) / 5

-- Prove that the total population is 3311 given the described conditions
theorem total_population_is_3311 {L n men women children boys girls : ℕ}
  (hc : more_children_than_adults children (n + men))
  (hm : more_men_than_women men n)
  (hg : more_girls_than_boys girls boys)
  (hL : L = n + men + boys + girls)
  (hL_lt : fewer_than_6000_inhabitants L) :
  L = 3311 :=
sorry

end total_population_is_3311_l202_202279


namespace hyperbola_center_l202_202293

-- Definitions based on conditions
def hyperbola (x y : ℝ) : Prop := ((4 * x + 8) ^ 2 / 16) - ((5 * y - 5) ^ 2 / 25) = 1

-- Theorem statement
theorem hyperbola_center : ∀ x y : ℝ, hyperbola x y → (x, y) = (-2, 1) := 
  by
    sorry

end hyperbola_center_l202_202293


namespace order_of_abc_l202_202759

noncomputable def a : ℝ := Real.log 3 / Real.log 4
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log (4/3) / Real.log (3/4)

theorem order_of_abc : b > a ∧ a > c := by
  sorry

end order_of_abc_l202_202759


namespace tangent_line_equation_l202_202337

theorem tangent_line_equation (a : ℝ) (h : a ≠ 0) :
  (∃ b : ℝ, b = 2 ∧ (∀ x : ℝ, y = a * x^2) ∧ y - a = b * (x - 1)) → 
  ∃ (x y : ℝ), 2 * x - y - 1 = 0 :=
by
  sorry

end tangent_line_equation_l202_202337


namespace cost_per_lunch_is_7_l202_202809

-- Definitions of the conditions
def total_children := 35
def total_chaperones := 5
def janet := 1
def additional_lunches := 3
def total_cost := 308

-- Calculate the total number of lunches
def total_lunches : Int :=
  total_children + total_chaperones + janet + additional_lunches

-- Statement to prove that the cost per lunch is 7
theorem cost_per_lunch_is_7 : total_cost / total_lunches = 7 := by
  sorry

end cost_per_lunch_is_7_l202_202809


namespace distance_between_sasha_and_kolya_is_19_meters_l202_202401

theorem distance_between_sasha_and_kolya_is_19_meters
  (v_S v_L v_K : ℝ)
  (h1 : v_L = 0.9 * v_S)
  (h2 : v_K = 0.81 * v_S)
  (h3 : ∀ t_S : ℝ, t_S = 100 / v_S) :
  (∀ t_S : ℝ, 100 - v_K * t_S = 19) :=
by
  intros t_S
  have vL_defined : v_L = 0.9 * v_S := h1
  have vK_defined : v_K = 0.81 * v_S := h2
  have time_S : t_S = 100 / v_S := h3 t_S
  sorry

end distance_between_sasha_and_kolya_is_19_meters_l202_202401


namespace variance_of_given_data_is_2_l202_202123

-- Define the data set
def data_set : List ℕ := [198, 199, 200, 201, 202]

-- Define the mean function for a given data set
noncomputable def mean (data : List ℕ) : ℝ :=
  (data.sum : ℝ) / data.length

-- Define the variance function for a given data set
noncomputable def variance (data : List ℕ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x : ℝ) - μ) |>.map (λ x => x^2)).sum / data.length

-- Proposition that the variance of the given data set is 2
theorem variance_of_given_data_is_2 : variance data_set = 2 := by
  sorry

end variance_of_given_data_is_2_l202_202123


namespace cost_price_of_article_l202_202747

-- Define the conditions and goal as a Lean 4 statement
theorem cost_price_of_article (M C : ℝ) (h1 : 0.95 * M = 75) (h2 : 1.25 * C = 75) : 
  C = 60 := 
by 
  sorry

end cost_price_of_article_l202_202747


namespace pigs_to_cows_ratio_l202_202274

-- Define the conditions given in the problem
def G : ℕ := 11
def C : ℕ := G + 4
def total_animals : ℕ := 56

-- Define the number of pigs from the total animals equation
noncomputable def P : ℕ := total_animals - (C + G)

-- State the theorem that the ratio of the number of pigs to the number of cows is 2:1
theorem pigs_to_cows_ratio : (P : ℚ) / C = 2 :=
  by
  sorry

end pigs_to_cows_ratio_l202_202274


namespace Doug_age_l202_202829

theorem Doug_age (Q J D : ℕ) (h1 : Q = J + 6) (h2 : J = D - 3) (h3 : Q = 19) : D = 16 := by
  sorry

end Doug_age_l202_202829


namespace polygon_sides_l202_202095

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l202_202095


namespace find_a_2016_l202_202324

-- Given definition for the sequence sum
def sequence_sum (n : ℕ) : ℕ := n * n

-- Definition for a_n using the given sequence sum
def term (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

-- Stating the theorem that we need to prove
theorem find_a_2016 : term 2016 = 4031 := 
by 
  sorry

end find_a_2016_l202_202324


namespace tomatoes_ruined_and_discarded_l202_202417

theorem tomatoes_ruined_and_discarded 
  (W : ℝ)
  (C : ℝ)
  (P : ℝ)
  (S : ℝ)
  (profit_percentage : ℝ)
  (initial_cost : C = 0.80 * W)
  (remaining_tomatoes : S = 0.9956)
  (desired_profit : profit_percentage = 0.12)
  (final_cost : 0.896 = 0.80 + 0.096) :
  0.9956 * (1 - P / 100) = 0.896 :=
by
  sorry

end tomatoes_ruined_and_discarded_l202_202417


namespace solution_set_of_inequalities_l202_202981

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202981


namespace solution_set_of_inequalities_l202_202969

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202969


namespace boat_speed_in_still_water_l202_202806

/-- Given a boat's speed along the stream and against the stream, prove its speed in still water. -/
theorem boat_speed_in_still_water (b s : ℝ) 
  (h1 : b + s = 11)
  (h2 : b - s = 5) : b = 8 :=
sorry

end boat_speed_in_still_water_l202_202806


namespace yield_percentage_of_stock_is_8_percent_l202_202269

theorem yield_percentage_of_stock_is_8_percent :
  let face_value := 100
  let dividend_rate := 0.20
  let market_price := 250
  annual_dividend = dividend_rate * face_value →
  yield_percentage = (annual_dividend / market_price) * 100 →
  yield_percentage = 8 := 
by
  sorry

end yield_percentage_of_stock_is_8_percent_l202_202269


namespace income_expenditure_ratio_l202_202555

theorem income_expenditure_ratio
  (I : ℕ) (E : ℕ) (S : ℕ)
  (h1 : I = 18000)
  (h2 : S = 3600)
  (h3 : S = I - E) : I / E = 5 / 4 :=
by
  -- The actual proof is skipped.
  sorry

end income_expenditure_ratio_l202_202555


namespace solution_set_linear_inequalities_l202_202936

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202936


namespace solution_set_inequalities_l202_202891

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202891


namespace sue_shoes_probability_l202_202700

def sueShoes : List (String × ℕ) := [("black", 7), ("brown", 3), ("gray", 2)]

def total_shoes := 24

def prob_same_color (color : String) (pairs : List (String × ℕ)) : ℚ :=
  let total_pairs := pairs.foldr (λ p acc => acc + p.snd) 0
  let matching_pair := pairs.filter (λ p => p.fst = color)
  if matching_pair.length = 1 then
   let n := matching_pair.head!.snd * 2
   (n / total_shoes) * ((n / 2) / (total_shoes - 1))
  else 0

def prob_total (pairs : List (String × ℕ)) : ℚ :=
  (prob_same_color "black" pairs) + (prob_same_color "brown" pairs) + (prob_same_color "gray" pairs)

theorem sue_shoes_probability :
  prob_total sueShoes = 31 / 138 := by
  sorry

end sue_shoes_probability_l202_202700


namespace point_on_x_axis_l202_202667

theorem point_on_x_axis (m : ℝ) (P : ℝ × ℝ) (hP : P = (m + 3, m - 1)) (hx : P.2 = 0) :
  P = (4, 0) :=
by
  sorry

end point_on_x_axis_l202_202667


namespace solution_set_linear_inequalities_l202_202942

theorem solution_set_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) → (3 < x ∧ x < 4) :=
by
  -- We will start the proof here using the given conditions leading to the conclusion.
  intro h,
  sorry

end solution_set_linear_inequalities_l202_202942


namespace find_extrema_l202_202307

noncomputable def f (x : ℝ) : ℝ := x^3 + (-3/2) * x^2 + (-3) * x + 1
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 2 * (-3/2) * x + (-3)
noncomputable def g (x : ℝ) : ℝ := f' x * Real.exp x

theorem find_extrema :
  (a = -3/2 ∧ b = -3 ∧ f' (1) = (3 * (1:ℝ)^2 - 3/2 * (1:ℝ) - 3) ) ∧
  (g 1 = -3 * Real.exp 1 ∧ g (-2) = 15 * Real.exp (-2)) := 
by
  -- Sorry for skipping the proof
  sorry

end find_extrema_l202_202307


namespace smallest_possible_value_of_N_l202_202569

-- Conditions definition:
variable (a b c d : ℕ)
variable (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
variable (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
variable (gcd_ab : Int.gcd a b = 1)
variable (gcd_ac : Int.gcd a c = 2)
variable (gcd_ad : Int.gcd a d = 4)
variable (gcd_bc : Int.gcd b c = 5)
variable (gcd_bd : Int.gcd b d = 3)
variable (gcd_cd : Int.gcd c d = N)
variable (hN : N > 5)

-- Statement to prove:
theorem smallest_possible_value_of_N : N = 14 := sorry

end smallest_possible_value_of_N_l202_202569


namespace probability_is_correct_l202_202486

def is_increasing (a b : ℤ) : Prop :=
  (a = 0 ∧ b < 0) ∨ (a ≠ 0 ∧ b ≤ a)

def valid_combinations : List (ℤ × ℤ) :=
  [(0, -1), (0, 1), (0, 3), (0, 5), 
   (1, -1), (1, 1), (1, 3), (1, 5), 
   (2, -1), (2, 1), (2, 3), (2, 5)]

def satisfying_combinations : List (ℤ × ℤ) :=
  valid_combinations.filter (λ (p : ℤ × ℤ), is_increasing p.1 p.2)

def probability_increasing : ℚ :=
  satisfying_combinations.length / valid_combinations.length

theorem probability_is_correct :
  probability_increasing = 5 / 12 :=
by norm_num [probability_increasing, List.length, valid_combinations, satisfying_combinations, is_increasing]; sorry

end probability_is_correct_l202_202486


namespace initial_bottle_caps_l202_202761

theorem initial_bottle_caps (end_caps : ℕ) (eaten_caps : ℕ) (start_caps : ℕ) 
  (h1 : end_caps = 61) 
  (h2 : eaten_caps = 4) 
  (h3 : start_caps = end_caps + eaten_caps) : 
  start_caps = 65 := 
by 
  sorry

end initial_bottle_caps_l202_202761


namespace geometric_sequence_a6_l202_202183

theorem geometric_sequence_a6 : 
  ∀ (a : ℕ → ℚ), (∀ n, a n ≠ 0) → a 1 = 3 → (∀ n, 2 * a (n+1) - a n = 0) → a 6 = 3 / 32 :=
by
  intros a h1 h2 h3
  sorry

end geometric_sequence_a6_l202_202183


namespace count_valid_five_digit_numbers_l202_202362

-- Define the conditions
def is_five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def is_divisible_by (a b : ℕ) : Prop := b ∣ a

def quotient_remainder_sum_divisible_by (n q r : ℕ) : Prop :=
  (n = 100 * q + r) ∧ ((q + r) % 7 = 0)

-- Define the theorem
theorem count_valid_five_digit_numbers : 
  ∃ k, k = 8160 ∧ ∀ n, is_five_digit_number n ∧ 
    is_divisible_by n 13 ∧ 
    ∃ q r, quotient_remainder_sum_divisible_by n q r → 
    k = 8160 :=
sorry

end count_valid_five_digit_numbers_l202_202362


namespace non_fiction_vs_fiction_diff_l202_202711

def total_books : Nat := 35
def fiction_books : Nat := 5
def picture_books : Nat := 11
def autobiography_books : Nat := 2 * fiction_books

def accounted_books : Nat := fiction_books + autobiography_books + picture_books
def non_fiction_books : Nat := total_books - accounted_books

theorem non_fiction_vs_fiction_diff :
  non_fiction_books - fiction_books = 4 := by 
  sorry

end non_fiction_vs_fiction_diff_l202_202711


namespace winnieKeepsBalloons_l202_202723

-- Given conditions
def redBalloons : Nat := 24
def whiteBalloons : Nat := 39
def greenBalloons : Nat := 72
def chartreuseBalloons : Nat := 91
def totalFriends : Nat := 11

-- Total balloons
def totalBalloons : Nat := redBalloons + whiteBalloons + greenBalloons + chartreuseBalloons

-- Theorem: Prove the number of balloons Winnie keeps for herself
theorem winnieKeepsBalloons :
  totalBalloons % totalFriends = 6 :=
by
  -- Placeholder for the proof
  sorry

end winnieKeepsBalloons_l202_202723


namespace find_building_block_width_l202_202444

noncomputable def building_block_width
  (box_height box_width box_length building_block_height building_block_length : ℕ)
  (num_building_blocks : ℕ)
  (box_height_eq : box_height = 8)
  (box_width_eq : box_width = 10)
  (box_length_eq : box_length = 12)
  (building_block_height_eq : building_block_height = 3)
  (building_block_length_eq : building_block_length = 4)
  (num_building_blocks_eq : num_building_blocks = 40)
: ℕ :=
(8 * 10 * 12) / 40 / (3 * 4)

theorem find_building_block_width
  (box_height box_width box_length building_block_height building_block_length : ℕ)
  (num_building_blocks : ℕ)
  (box_height_eq : box_height = 8)
  (box_width_eq : box_width = 10)
  (box_length_eq : box_length = 12)
  (building_block_height_eq : building_block_height = 3)
  (building_block_length_eq : building_block_length = 4)
  (num_building_blocks_eq : num_building_blocks = 40) :
  building_block_width box_height box_width box_length building_block_height building_block_length num_building_blocks box_height_eq box_width_eq box_length_eq building_block_height_eq building_block_length_eq num_building_blocks_eq = 2 := 
sorry

end find_building_block_width_l202_202444


namespace hungarian_1905_l202_202422

open Nat

theorem hungarian_1905 (n p : ℕ) : (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + p * y = n ∧ x + y = p^z) ↔ 
  (p > 1 ∧ (n - 1) % (p - 1) = 0 ∧ ¬ ∃ k : ℕ, n = p^k) :=
by
  sorry

end hungarian_1905_l202_202422


namespace kim_fraction_of_shirts_given_l202_202677

open Nat

theorem kim_fraction_of_shirts_given (d : ℕ) (s_left : ℕ) (one_dozen := 12) 
  (original_shirts := 4 * one_dozen) 
  (given_shirts := original_shirts - s_left) 
  (fraction_given := given_shirts / original_shirts) 
  (hc1 : d = one_dozen) 
  (hc2 : s_left = 32) 
  : fraction_given = 1 / 3 := 
by 
  sorry

end kim_fraction_of_shirts_given_l202_202677


namespace solution_set_of_inequalities_l202_202977

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202977


namespace mother_to_father_age_ratio_l202_202758

def DarcieAge : ℕ := 4
def FatherAge : ℕ := 30
def MotherAge : ℕ := DarcieAge * 6

theorem mother_to_father_age_ratio :
  (MotherAge : ℚ) / (FatherAge : ℚ) = (4 / 5) := by
  sorry

end mother_to_father_age_ratio_l202_202758


namespace fraction_of_boys_in_clubs_l202_202064

def total_students : ℕ := 150
def girls_percent : ℚ := 0.6
def boys_percent : ℚ := 0.4
def boys_not_in_clubs : ℕ := 40

theorem fraction_of_boys_in_clubs :
  let total_boys := total_students * boys_percent
  let boys_in_clubs := total_boys - boys_not_in_clubs
  boys_in_clubs / total_boys = 1 / 3 := 
  by
    sorry

end fraction_of_boys_in_clubs_l202_202064


namespace cost_of_fencing_per_meter_l202_202416

theorem cost_of_fencing_per_meter (l b : ℕ) (total_cost : ℕ) (cost_per_meter : ℝ) : 
  (l = 66) → 
  (l = b + 32) → 
  (total_cost = 5300) → 
  (2 * l + 2 * b = 200) → 
  (cost_per_meter = total_cost / 200) → 
  cost_per_meter = 26.5 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof is omitted by design
  sorry

end cost_of_fencing_per_meter_l202_202416


namespace smallest_N_exists_l202_202567

theorem smallest_N_exists (
  a b c d : ℕ := list.perm [1, 2, 3, 4, 5] [gcd a b, gcd a c, gcd a d, gcd b c, gcd b d, gcd c d]
  (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_N: N > 5) : 
  N = 14 :=
by sorry

end smallest_N_exists_l202_202567


namespace problem_statement_l202_202217

noncomputable def omega : ℂ := sorry -- Definition placeholder for a specific nonreal root of x^4 = 1. 

theorem problem_statement (h1 : omega ^ 4 = 1) (h2 : omega ^ 2 = -1) : 
  (1 - omega + omega ^ 3) ^ 4 + (1 + omega - omega ^ 3) ^ 4 = -14 := 
sorry

end problem_statement_l202_202217


namespace committee_meeting_l202_202591

theorem committee_meeting : 
  ∃ (A B : ℕ), 2 * A + B = 7 ∧ A + 2 * B = 11 ∧ A + B = 6 :=
by 
  sorry

end committee_meeting_l202_202591


namespace product_calculation_l202_202288

theorem product_calculation :
  1500 * 2023 * 0.5023 * 50 = 306903675 :=
sorry

end product_calculation_l202_202288


namespace set_operations_l202_202526

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6})
variable (hA : A = {2, 4, 5})
variable (hB : B = {1, 2, 5})

theorem set_operations :
  (A ∩ B = {2, 5}) ∧ (A ∪ (U \ B) = {2, 3, 4, 5, 6}) :=
by
  sorry

end set_operations_l202_202526


namespace tan_of_angle_l202_202321

theorem tan_of_angle (α : ℝ) (h₁ : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (h₂ : Real.sin α = 3 / 5) : 
  Real.tan α = -3 / 4 := 
sorry

end tan_of_angle_l202_202321


namespace solution_set_inequalities_l202_202889

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202889


namespace distance_between_sasha_and_kolya_when_sasha_finished_l202_202391

-- Definitions based on the problem conditions
def distance_sasha : ℝ := 100
def distance_lesha_when_sasha_finished : ℝ := 90
def distance_kolya_when_lesha_finished : ℝ := 90

def velocity_lesha (v_s : ℝ) : ℝ := 0.9 * v_s
def velocity_kolya (v_s : ℝ) : ℝ := 0.81 * v_s

-- Theorem statement
theorem distance_between_sasha_and_kolya_when_sasha_finished (v_s : ℝ) :
  distance_sasha - (velocity_kolya v_s * (distance_sasha / v_s)) = 19 :=
  by sorry

end distance_between_sasha_and_kolya_when_sasha_finished_l202_202391


namespace marias_workday_ends_at_3_30_pm_l202_202527
open Nat

theorem marias_workday_ends_at_3_30_pm :
  let start_time := (7 : Nat)
  let lunch_start_time := (11 + (30 / 60))
  let work_duration := (8 : Nat)
  let lunch_break := (30 / 60 : Nat)
  let end_time := (15 + (30 / 60) : Nat)
  (start_time + work_duration + lunch_break) - (lunch_start_time - start_time) = end_time := by
  sorry

end marias_workday_ends_at_3_30_pm_l202_202527


namespace aarti_completes_work_multiple_l202_202281

-- Define the condition that Aarti can complete one piece of work in 9 days.
def aarti_work_rate (work_size : ℕ) : ℕ := 9

-- Define the task to find how many times she will complete the work in 27 days.
def aarti_work_multiple (total_days : ℕ) (work_size: ℕ) : ℕ :=
  total_days / (aarti_work_rate work_size)

-- The theorem to prove the number of times Aarti will complete the work.
theorem aarti_completes_work_multiple : aarti_work_multiple 27 1 = 3 := by
  sorry

end aarti_completes_work_multiple_l202_202281


namespace distance_between_sasha_and_kolya_when_sasha_finished_l202_202390

-- Definitions based on the problem conditions
def distance_sasha : ℝ := 100
def distance_lesha_when_sasha_finished : ℝ := 90
def distance_kolya_when_lesha_finished : ℝ := 90

def velocity_lesha (v_s : ℝ) : ℝ := 0.9 * v_s
def velocity_kolya (v_s : ℝ) : ℝ := 0.81 * v_s

-- Theorem statement
theorem distance_between_sasha_and_kolya_when_sasha_finished (v_s : ℝ) :
  distance_sasha - (velocity_kolya v_s * (distance_sasha / v_s)) = 19 :=
  by sorry

end distance_between_sasha_and_kolya_when_sasha_finished_l202_202390


namespace a10_plus_b10_l202_202225

noncomputable def a : ℝ := sorry -- a will be a real number satisfying the conditions
noncomputable def b : ℝ := sorry -- b will be a real number satisfying the conditions

axiom ab_condition1 : a + b = 1
axiom ab_condition2 : a^2 + b^2 = 3
axiom ab_condition3 : a^3 + b^3 = 4
axiom ab_condition4 : a^4 + b^4 = 7
axiom ab_condition5 : a^5 + b^5 = 11

theorem a10_plus_b10 : a^10 + b^10 = 123 :=
by 
  sorry

end a10_plus_b10_l202_202225


namespace solution_set_linear_inequalities_l202_202906

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202906


namespace geometric_sequence_term_6_l202_202634

-- Define the geometric sequence conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

variables 
  (a : ℕ → ℝ) -- the geometric sequence
  (r : ℝ) -- common ratio, which is 2
  (h_r : r = 2)
  (h_pos : ∀ n, 0 < a n)
  (h_condition : a 4 * a 10 = 16)

-- The proof statement
theorem geometric_sequence_term_6 :
  a 6 = 2 :=
sorry

end geometric_sequence_term_6_l202_202634


namespace ludek_unique_stamps_l202_202355

theorem ludek_unique_stamps (K M L : ℕ) (k_m_shared k_l_shared m_l_shared : ℕ)
  (hk : K + M = 101)
  (hl : K + L = 115)
  (hm : M + L = 110)
  (k_m_shared := 5)
  (k_l_shared := 12)
  (m_l_shared := 7) :
  L - k_l_shared - m_l_shared = 43 :=
by
  sorry

end ludek_unique_stamps_l202_202355


namespace sin_cos_ratio_l202_202494

open Real

theorem sin_cos_ratio
  (θ : ℝ)
  (h : (sin θ + cos θ) / (sin θ - cos θ) = 2) :
  sin θ * cos θ = 3 / 10 := 
by
  sorry

end sin_cos_ratio_l202_202494


namespace linear_inequalities_solution_l202_202865

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202865


namespace fairy_tale_island_counties_l202_202517

theorem fairy_tale_island_counties : 
  (initial_elf_counties : ℕ) 
  (initial_dwarf_counties : ℕ) 
  (initial_centaur_counties : ℕ)
  (first_year_non_elf_divide : ℕ)
  (second_year_non_dwarf_divide : ℕ)
  (third_year_non_centaur_divide : ℕ) :
  initial_elf_counties = 1 →
  initial_dwarf_counties = 1 →
  initial_centaur_counties = 1 →
  first_year_non_elf_divide = 3 →
  second_year_non_dwarf_divide = 4 →
  third_year_non_centaur_divide = 6 →
  let elf_counties_first_year := initial_elf_counties in
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide in
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide in
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide in
  let dwarf_counties_second_year := dwarf_counties_first_year in
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide in
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide in
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide in
  let centaur_counties_third_year := centaur_counties_second_year in
  elf_counties_third_year + dwarf_counties_third_year + centaur_counties_third_year = 54 := 
by {
  intros,
  let elf_counties_first_year := initial_elf_counties,
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide,
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide,
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide,
  let dwarf_counties_second_year := dwarf_counties_first_year,
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide,
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide,
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide,
  let centaur_counties_third_year := centaur_counties_second_year,
  have elf_counties := elf_counties_third_year,
  have dwarf_counties := dwarf_counties_third_year,
  have centaur_counties := centaur_counties_third_year,
  have total_counties := elf_counties + dwarf_counties + centaur_counties,
  show total_counties = 54,
  sorry
}

end fairy_tale_island_counties_l202_202517


namespace angle_in_third_quadrant_l202_202216

theorem angle_in_third_quadrant
  (α : ℝ)
  (k : ℤ)
  (h : (π / 2) + 2 * (↑k) * π < α ∧ α < π + 2 * (↑k) * π) :
  π + 2 * (↑k) * π < (π / 2) + α ∧ (π / 2) + α < (3 * π / 2) + 2 * (↑k) * π :=
by
  sorry

end angle_in_third_quadrant_l202_202216


namespace inequality_proof_equality_case_l202_202646

-- Defining that a, b, c are positive real numbers
variables (a b c : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The main theorem statement
theorem inequality_proof :
  a^2 + b^2 + c^2 + ( (1 / a) + (1 / b) + (1 / c) )^2 >= 6 * Real.sqrt 3 :=
sorry

-- Equality case
theorem equality_case :
  a = b ∧ b = c ∧ a = Real.sqrt 3^(1/4) →
  a^2 + b^2 + c^2 + ( (1 / a) + (1 / b) + (1 / c) )^2 = 6 * Real.sqrt 3 :=
sorry

end inequality_proof_equality_case_l202_202646


namespace books_left_over_l202_202672

-- Define the conditions as variables in Lean
def total_books : ℕ := 1500
def new_shelf_capacity : ℕ := 28

-- State the theorem based on these conditions
theorem books_left_over : total_books % new_shelf_capacity = 14 :=
by
  sorry

end books_left_over_l202_202672


namespace solution_set_of_inequalities_l202_202979

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202979


namespace number_of_positive_integers_l202_202636

theorem number_of_positive_integers (n : ℕ) : ∃! k : ℕ, k = 5 ∧
  (∀ n : ℕ, (1 ≤ n) → (12 % (n + 1) = 0)) :=
sorry

end number_of_positive_integers_l202_202636


namespace calculate_final_amount_l202_202005

def initial_amount : ℝ := 7500
def first_year_rate : ℝ := 0.20
def second_year_rate : ℝ := 0.25

def first_year_interest (p : ℝ) (r : ℝ) : ℝ := p * r
def amount_after_first_year (p : ℝ) (i : ℝ) : ℝ := p + i

def second_year_interest (p : ℝ) (r : ℝ) : ℝ := p * r
def amount_after_second_year (p : ℝ) (i : ℝ) : ℝ := p + i

theorem calculate_final_amount :
  let initial : ℝ := initial_amount
  let interest1 : ℝ := first_year_interest initial first_year_rate
  let amount1 : ℝ := amount_after_first_year initial interest1
  let interest2 : ℝ := second_year_interest amount1 second_year_rate
  let final_amount : ℝ := amount_after_second_year amount1 interest2
  final_amount = 11250 := by
  sorry

end calculate_final_amount_l202_202005


namespace geometric_sequence_tenth_term_l202_202291

theorem geometric_sequence_tenth_term :
  let a : ℚ := 4
  let r : ℚ := 5/3
  let n : ℕ := 10
  a * r^(n-1) = 7812500 / 19683 :=
by sorry

end geometric_sequence_tenth_term_l202_202291


namespace subset_exists_l202_202021

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {x + 2, 1}

-- Statement of the theorem
theorem subset_exists (x : ℝ) : B 2 ⊆ A 2 :=
by
  sorry

end subset_exists_l202_202021


namespace polygon_sides_l202_202115

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l202_202115


namespace solve_for_constants_l202_202813

def f (x : ℤ) (a b c : ℤ) : ℤ :=
if x > 0 then 2 * a * x + 4
else if x = 0 then a + b
else 3 * b * x + 2 * c

theorem solve_for_constants :
  ∃ a b c : ℤ, 
    f 1 a b c = 6 ∧ 
    f 0 a b c = 7 ∧ 
    f (-1) a b c = -4 ∧ 
    a + b + c = 14 :=
by
  sorry

end solve_for_constants_l202_202813


namespace each_dolphin_training_hours_l202_202847

theorem each_dolphin_training_hours
  (num_dolphins : ℕ)
  (num_trainers : ℕ)
  (hours_per_trainer : ℕ)
  (total_hours : ℕ := num_trainers * hours_per_trainer)
  (hours_per_dolphin_daily : ℕ := total_hours / num_dolphins)
  (h1 : num_dolphins = 4)
  (h2 : num_trainers = 2)
  (h3 : hours_per_trainer = 6) :
  hours_per_dolphin_daily = 3 :=
  by sorry

end each_dolphin_training_hours_l202_202847


namespace infinite_primes_p_solutions_eq_p2_l202_202827

theorem infinite_primes_p_solutions_eq_p2 :
  ∃ᶠ p in Filter.atTop, Prime p ∧ 
  (∃ (S : Finset (ZMod p × ZMod p × ZMod p)),
    S.card = p^2 ∧ ∀ (x y z : ZMod p), (3 * x^3 + 4 * y^4 + 5 * z^3 - y^4 * z = 0) ↔ (x, y, z) ∈ S) :=
sorry

end infinite_primes_p_solutions_eq_p2_l202_202827


namespace solution_set_linear_inequalities_l202_202903

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202903


namespace speed_of_man_in_still_water_l202_202740

-- Define the parameters and conditions
def speed_in_still_water (v_m : ℝ) (v_s : ℝ) : Prop :=
    (v_m + v_s = 5) ∧  -- downstream condition
    (v_m - v_s = 7)    -- upstream condition

-- The theorem statement
theorem speed_of_man_in_still_water : 
  ∃ v_m v_s : ℝ, speed_in_still_water v_m v_s ∧ v_m = 6 := 
by
  sorry

end speed_of_man_in_still_water_l202_202740


namespace solution_set_system_of_inequalities_l202_202922

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202922


namespace infinite_rational_solutions_x3_y3_9_l202_202407

theorem infinite_rational_solutions_x3_y3_9 :
  ∃ (S : Set (ℚ × ℚ)), S.Infinite ∧ (∀ (x y : ℚ), (x, y) ∈ S → x^3 + y^3 = 9) :=
sorry

end infinite_rational_solutions_x3_y3_9_l202_202407


namespace find_present_ratio_l202_202849

noncomputable def present_ratio_of_teachers_to_students : Prop :=
  ∃ (S T S' T' : ℕ),
    (T = 3) ∧
    (S = 50 * T) ∧
    (S' = S + 50) ∧
    (T' = T + 5) ∧
    (S' / T' = 25 / 1) ∧ 
    (T / S = 1 / 50)

theorem find_present_ratio : present_ratio_of_teachers_to_students :=
by
  sorry

end find_present_ratio_l202_202849


namespace angle_measure_l202_202237

theorem angle_measure : 
  ∃ (x : ℝ), (x + (3 * x + 3) = 90) ∧ x = 21.75 := by
  sorry

end angle_measure_l202_202237


namespace complex_number_quadrant_l202_202487

def imaginary_unit := Complex.I

def complex_simplification (z : Complex) : Complex :=
  z

theorem complex_number_quadrant :
  ∃ z : Complex, z = (5 * imaginary_unit) / (2 + imaginary_unit ^ 9) ∧ (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_number_quadrant_l202_202487


namespace area_first_side_l202_202548

-- Define dimensions of the box
variables (L W H : ℝ)

-- Define conditions
def area_WH : Prop := W * H = 72
def area_LH : Prop := L * H = 60
def volume_box : Prop := L * W * H = 720

-- Prove the area of the first side
theorem area_first_side (h1 : area_WH W H) (h2 : area_LH L H) (h3 : volume_box L W H) : L * W = 120 :=
by sorry

end area_first_side_l202_202548


namespace jana_winning_strategy_l202_202059

theorem jana_winning_strategy (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  (m + n) % 2 = 1 ∨ m = 1 ∨ n = 1 := sorry

end jana_winning_strategy_l202_202059


namespace polygon_sides_l202_202113

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l202_202113


namespace second_hand_travel_distance_l202_202853

theorem second_hand_travel_distance (r : ℝ) (t : ℝ) : 
  r = 10 → t = 45 → 2 * t * π * r = 900 * π :=
by
  intro r_def t_def
  sorry

end second_hand_travel_distance_l202_202853


namespace solve_inequalities_l202_202955

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202955


namespace solve_inequalities_l202_202997

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202997


namespace form_of_reasoning_is_wrong_l202_202127

-- Let's define the conditions
def some_rat_nums_are_proper_fractions : Prop :=
  ∃ q : ℚ, (q.num : ℤ) ≠ q.den ∧ (q.den : ℤ) ≠ 1 ∧ q.den ≠ 0

def integers_are_rational_numbers : Prop :=
  ∀ n : ℤ, ∃ q : ℚ, q = n

-- The major premise of the syllogism
def major_premise := some_rat_nums_are_proper_fractions

-- The minor premise of the syllogism
def minor_premise := integers_are_rational_numbers

-- The conclusion of the syllogism
def conclusion := ∀ n : ℤ, ∃ q : ℚ, (q.num : ℤ) ≠ q.den ∧ (q.den : ℤ) ≠ 1 ∧ q.den ≠ 0

-- We need to prove that the form of reasoning is wrong
theorem form_of_reasoning_is_wrong (H1 : major_premise) (H2 : minor_premise) : ¬ conclusion :=
by
  sorry -- proof to be filled in

end form_of_reasoning_is_wrong_l202_202127


namespace linear_inequalities_solution_l202_202857

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l202_202857


namespace fraction_of_female_participants_is_correct_l202_202706

-- defining conditions
def last_year_males : ℕ := 30
def male_increase_rate : ℚ := 1.1
def female_increase_rate : ℚ := 1.25
def overall_increase_rate : ℚ := 1.2

-- the statement to prove
theorem fraction_of_female_participants_is_correct :
  ∀ (y : ℕ), 
  let males_this_year := last_year_males * male_increase_rate
  let females_this_year := y * female_increase_rate
  let total_last_year := last_year_males + y
  let total_this_year := total_last_year * overall_increase_rate
  total_this_year = males_this_year + females_this_year →
  (females_this_year / total_this_year) = (25 / 36) :=
by
  intros y
  let males_this_year := last_year_males * male_increase_rate
  let females_this_year := y * female_increase_rate
  let total_last_year := last_year_males + y
  let total_this_year := total_last_year * overall_increase_rate
  intro h
  sorry

end fraction_of_female_participants_is_correct_l202_202706


namespace part_length_proof_l202_202609

-- Define the scale length in feet and inches
def scale_length_ft : ℕ := 6
def scale_length_inch : ℕ := 8

-- Define the number of equal parts
def num_parts : ℕ := 4

-- Calculate total length in inches
def total_length_inch : ℕ := scale_length_ft * 12 + scale_length_inch

-- Calculate the length of each part in inches
def part_length_inch : ℕ := total_length_inch / num_parts

-- Prove that each part is 1 foot 8 inches long
theorem part_length_proof :
  part_length_inch = 1 * 12 + 8 :=
by
  sorry

end part_length_proof_l202_202609


namespace common_ratio_geometric_series_l202_202471

-- Define the first three terms of the series
def first_term := (-3: ℚ) / 5
def second_term := (-5: ℚ) / 3
def third_term := (-125: ℚ) / 27

-- Prove that the common ratio = 25/9
theorem common_ratio_geometric_series :
  (second_term / first_term) = (25 : ℚ) / 9 :=
by
  sorry

end common_ratio_geometric_series_l202_202471


namespace solve_inequalities_l202_202949

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l202_202949


namespace repeating_decimal_as_fraction_l202_202168

/-- Define x as the repeating decimal 7.182182... -/
def x : ℚ := 
  7 + 182 / 999

/-- Define y as the fraction 7175/999 -/
def y : ℚ := 
  7175 / 999

/-- Theorem stating that the repeating decimal 7.182182... is equal to the fraction 7175/999 -/
theorem repeating_decimal_as_fraction : x = y :=
sorry

end repeating_decimal_as_fraction_l202_202168


namespace prob_xiao_ming_at_least_3_eq_xiao_yu_distribution_table_xiao_yu_expectation_eq_recommend_participant_l202_202714

namespace AviationCompetition

noncomputable theory

def xiao_ming_prob_correct : ℚ := 3 / 4

def xiao_ming_event (n : ℕ) : ℚ :=
if n = 3 then (Mathlib.Combinatorics.choose 4 3) * (xiao_ming_prob_correct^3) * ((1 - xiao_ming_prob_correct)^1)
else if n = 4 then (Mathlib.Combinatorics.choose 4 4) * (xiao_ming_prob_correct^4)
else 0

def prob_xiao_ming_at_least_3 : ℚ :=
xiao_ming_event 3 + xiao_ming_event 4

def xiao_yu_distribution : list (ℕ × ℚ) :=
[(2, 3 / 14), (3, 4 / 7), (4, 3 / 14)]

def xiao_yu_expectation : ℚ :=
2 * (3 / 14) + 3 * (4 / 7) + 4 * (3 / 14)

def xiao_yu_prob_at_least_3 : ℚ :=
4 / 7 + 3 / 14

theorem prob_xiao_ming_at_least_3_eq : prob_xiao_ming_at_least_3 = 189 / 256 := sorry

theorem xiao_yu_distribution_table : xiao_yu_distribution = [(2, 3 / 14), (3, 4 / 7), (4, 3 / 14)] := sorry

theorem xiao_yu_expectation_eq : xiao_yu_expectation = 3 := sorry

theorem recommend_participant : (prob_xiao_ming_at_least_3 < xiao_yu_prob_at_least_3) :=
by
  exact nat.lt_of_le_of_ne (show prob_xiao_ming_at_least_3 ≤ xiao_yu_prob_at_least_3 from sorry) (λ h, sorry)

end AviationCompetition

end prob_xiao_ming_at_least_3_eq_xiao_yu_distribution_table_xiao_yu_expectation_eq_recommend_participant_l202_202714


namespace son_age_is_9_l202_202034

-- Definitions for the conditions in the problem
def son_age (S F : ℕ) : Prop := S = (1 / 4 : ℝ) * F - 1
def father_age (S F : ℕ) : Prop := F = 5 * S - 5

-- Main statement of the equivalent problem
theorem son_age_is_9 : ∃ S F : ℕ, son_age S F ∧ father_age S F ∧ S = 9 :=
by
  -- We will leave the proof as an exercise
  sorry

end son_age_is_9_l202_202034


namespace evaluate_expression_l202_202763

theorem evaluate_expression :
  1 + (3 / (4 + (5 / (6 + (7 / 8))))) = 85 / 52 := 
by
  sorry

end evaluate_expression_l202_202763


namespace total_length_of_table_free_sides_l202_202603

theorem total_length_of_table_free_sides
  (L W : ℕ) -- Define lengths of the sides
  (h1 : L = 2 * W) -- The side opposite the wall is twice the length of each of the other two free sides
  (h2 : L * W = 128) -- The area of the rectangular table is 128 square feet
  : L + 2 * W = 32 -- Prove the total length of the table's free sides is 32 feet
  :=
sorry -- proof omitted

end total_length_of_table_free_sides_l202_202603


namespace smallest_collection_l202_202133

def Yoongi_collected : ℕ := 4
def Jungkook_collected : ℕ := 6 * 3
def Yuna_collected : ℕ := 5

theorem smallest_collection : Yoongi_collected = 4 ∧ Yoongi_collected ≤ Jungkook_collected ∧ Yoongi_collected ≤ Yuna_collected := by
  sorry

end smallest_collection_l202_202133


namespace meeting_time_and_location_l202_202147

/-- Define the initial conditions -/
def start_time : ℕ := 8 -- 8:00 AM
def city_distance : ℕ := 12 -- 12 kilometers
def pedestrian_speed : ℚ := 6 -- 6 km/h
def cyclist_speed : ℚ := 18 -- 18 km/h

/-- Define the conditions for meeting time and location -/
theorem meeting_time_and_location :
  ∃ (meet_time : ℕ) (meet_distance : ℚ),
    meet_time = 9 * 60 + 15 ∧   -- 9:15 AM in minutes
    meet_distance = 4.5 :=      -- 4.5 kilometers
sorry

end meeting_time_and_location_l202_202147


namespace sides_of_polygon_l202_202121

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l202_202121


namespace prod_of_consecutive_nums_divisible_by_504_l202_202372

theorem prod_of_consecutive_nums_divisible_by_504
  (a : ℕ)
  (h : ∃ b : ℕ, a = b ^ 3) :
  (a^3 - 1) * a^3 * (a^3 + 1) % 504 = 0 := 
sorry

end prod_of_consecutive_nums_divisible_by_504_l202_202372


namespace double_people_half_work_l202_202726

-- Definitions
def initial_person_count (P : ℕ) : Prop := true
def initial_time (T : ℕ) : Prop := T = 16

-- Theorem
theorem double_people_half_work (P T : ℕ) (hP : initial_person_count P) (hT : initial_time T) : P > 0 → (2 * P) * (T / 2) = P * T / 2 := by
  sorry

end double_people_half_work_l202_202726


namespace race_distance_l202_202378

/-- Given that Sasha, Lesha, and Kolya start a 100m race simultaneously and run at constant velocities,
when Sasha finishes, Lesha is 10m behind, and when Lesha finishes, Kolya is 10m behind.
Prove that the distance between Sasha and Kolya when Sasha finishes is 19 meters. -/
theorem race_distance
    (v_S v_L v_K : ℝ)
    (h1 : 100 / v_S - 100 / v_L = 10 / v_L)
    (h2 : 100 / v_L - 100 / v_K = 10 / v_K) :
    100 - 81 = 19 :=
by
  sorry

end race_distance_l202_202378


namespace solution_set_linear_inequalities_l202_202899

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202899


namespace petya_vasya_problem_l202_202559

theorem petya_vasya_problem :
  ∀ n : ℕ, (∀ x : ℕ, x = 12320 * 10 ^ (10 * n + 1) - 1 →
    (∃ p q : ℕ, (p ≠ q ∧ ∀ r : ℕ, (r ∣ x → (r = p ∨ r = q))))) → n = 0 :=
by
  sorry

end petya_vasya_problem_l202_202559


namespace first_snail_time_proof_l202_202250

-- Define the conditions
def first_snail_speed := 2 -- speed in feet per minute
def second_snail_speed := 2 * first_snail_speed
def third_snail_speed := 5 * second_snail_speed
def third_snail_time := 2 -- time in minutes
def distance := third_snail_speed * third_snail_time

-- Define the time it took the first snail
def first_snail_time := distance / first_snail_speed

-- Define the theorem to be proven
theorem first_snail_time_proof : first_snail_time = 20 := 
by
  -- Proof should be filled here
  sorry

end first_snail_time_proof_l202_202250


namespace abs_ineq_sol_set_l202_202235

theorem abs_ineq_sol_set (x : ℝ) : (|x - 2| + |x - 1| ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) :=
by
  sorry

end abs_ineq_sol_set_l202_202235


namespace unique_solution_for_equation_l202_202439

theorem unique_solution_for_equation (a b c d : ℝ) 
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d)
  (h : ∀ x : ℝ, (a * x + b) ^ 2016 + (x ^ 2 + c * x + d) ^ 1008 = 8 * (x - 2) ^ 2016) :
  a = 2 ^ (1 / 672) ∧ b = -2 * 2 ^ (1 / 672) ∧ c = -4 ∧ d = 4 :=
by
  sorry

end unique_solution_for_equation_l202_202439


namespace calculate_expression_l202_202159

theorem calculate_expression : (Real.sqrt 3)^0 + 2^(-1) + Real.sqrt 2 * Real.cos (Float.pi / 4) - abs (-1 / 2) = 2 := 
by
  sorry

end calculate_expression_l202_202159


namespace solution_set_l202_202877

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l202_202877


namespace robin_gum_total_l202_202375

theorem robin_gum_total :
  let original_gum := 18.0
  let given_gum := 44.0
  original_gum + given_gum = 62.0 := by
  sorry

end robin_gum_total_l202_202375


namespace problem_1_problem_2_l202_202653

open Real

noncomputable def f (omega : ℝ) (x : ℝ) : ℝ := 
  (cos (omega * x) * cos (omega * x) + sqrt 3 * cos (omega * x) * sin (omega * x) - 1/2)

theorem problem_1 (ω : ℝ) (hω : ω > 0):
 (f ω x = sin (2 * x + π / 6)) ∧ 
 (∀ k : ℤ, ∀ x : ℝ, (-π / 3 + ↑k * π) ≤ x ∧ x ≤ (π / 6 + ↑k * π) → f ω x = sin (2 * x + π / 6)) :=
sorry

theorem problem_2 (A b S a : ℝ) (hA : A / 2 = π / 3)
  (hb : b = 1) (hS: S = sqrt 3) :
  a = sqrt 13 :=
sorry

end problem_1_problem_2_l202_202653


namespace min_ticket_gates_l202_202000

theorem min_ticket_gates (a x y : ℕ) (h_pos: a > 0) :
  (a = 30 * x) ∧ (y = 2 * x) → ∃ n : ℕ, (n ≥ 4) ∧ (a + 5 * x ≤ 5 * n * y) :=
by
  sorry

end min_ticket_gates_l202_202000


namespace solution_set_of_inequalities_l202_202973

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202973


namespace factor_quadratic_l202_202169

theorem factor_quadratic (x : ℝ) : (16 * x^2 - 40 * x + 25) = (4 * x - 5)^2 :=
by 
  sorry

end factor_quadratic_l202_202169


namespace min_value_of_even_function_l202_202268

-- Define f(x) = (x + a)(x + b)
def f (x a b : ℝ) : ℝ := (x + a) * (x + b)

-- Given conditions
variables (a b : ℝ)
#check f  -- Ensuring the definition works

-- Prove that the minimum value of f(x) is -4 given that f(x) is an even function
theorem min_value_of_even_function (h_even : ∀ x : ℝ, f x a b = f (-x) a b)
  (h_domain : a + 4 > a) : ∃ c : ℝ, (f c a b = -4) :=
by
  -- We state that this function is even and consider the provided domain.
  sorry  -- Placeholder for the proof

end min_value_of_even_function_l202_202268


namespace bug_return_probability_twelfth_move_l202_202597

-- Conditions
def P : ℕ → ℚ
| 0       => 1
| (n + 1) => (1 : ℚ) / 3 * (1 - P n)

theorem bug_return_probability_twelfth_move :
  P 12 = 14762 / 59049 := by
sorry

end bug_return_probability_twelfth_move_l202_202597


namespace bill_sunday_miles_l202_202226

-- Define the variables
variables (B S J : ℕ) -- B for miles Bill ran on Saturday, S for miles Bill ran on Sunday, J for miles Julia ran on Sunday

-- State the conditions
def condition1 (B S : ℕ) : Prop := S = B + 4
def condition2 (B S J : ℕ) : Prop := J = 2 * S
def condition3 (B S J : ℕ) : Prop := B + S + J = 20

-- The final theorem to prove the number of miles Bill ran on Sunday
theorem bill_sunday_miles (B S J : ℕ) 
  (h1 : condition1 B S)
  (h2 : condition2 B S J)
  (h3 : condition3 B S J) : 
  S = 6 := 
sorry

end bill_sunday_miles_l202_202226


namespace solution_set_of_inequalities_l202_202982

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l202_202982


namespace arithmetic_sequence_sum_l202_202673

theorem arithmetic_sequence_sum {a : ℕ → ℝ} (d a1 : ℝ)
  (h_arith: ∀ n, a n = a1 + (n - 1) * d)
  (h_condition: a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
by {
  sorry
}

end arithmetic_sequence_sum_l202_202673


namespace given_condition_l202_202331

variable (a : ℝ)

theorem given_condition
  (h1 : (a + 1/a)^2 = 5) :
  a^2 + 1/a^2 + a^3 + 1/a^3 = 3 + 2 * Real.sqrt 5 :=
sorry

end given_condition_l202_202331


namespace correct_calculation_l202_202435

theorem correct_calculation :
  (∃ (x y : ℝ), 5 * x + 2 * y ≠ 7 * x * y) ∧
  (∃ (x : ℝ), 3 * x - 2 * x ≠ 1) ∧
  (∃ (x : ℝ), x^2 + x^5 ≠ x^7) →
  (∀ (x y : ℝ), 3 * x^2 * y - 4 * y * x^2 = -x^2 * y) :=
by
  sorry

end correct_calculation_l202_202435


namespace mary_lambs_count_l202_202821

def initial_lambs : Nat := 6
def baby_lambs : Nat := 2 * 2
def traded_lambs : Nat := 3
def extra_lambs : Nat := 7

theorem mary_lambs_count : initial_lambs + baby_lambs - traded_lambs + extra_lambs = 14 := by
  sorry

end mary_lambs_count_l202_202821


namespace jason_total_hours_l202_202046

variables (hours_after_school hours_total : ℕ)

def earnings_after_school := 4 * hours_after_school
def earnings_saturday := 6 * 8
def total_earnings := earnings_after_school + earnings_saturday

theorem jason_total_hours :
  4 * hours_after_school + earnings_saturday = 88 →
  hours_total = hours_after_school + 8 →
  total_earnings = 88 →
  hours_total = 18 :=
by
  intros h1 h2 h3
  sorry

end jason_total_hours_l202_202046


namespace short_trees_after_planting_l202_202424

-- Define the current number of short trees
def current_short_trees : ℕ := 41

-- Define the number of short trees to be planted today
def new_short_trees : ℕ := 57

-- Define the expected total number of short trees after planting
def total_short_trees_after_planting : ℕ := 98

-- The theorem to prove that the total number of short trees after planting is as expected
theorem short_trees_after_planting :
  current_short_trees + new_short_trees = total_short_trees_after_planting :=
by
  -- Proof skipped using sorry
  sorry

end short_trees_after_planting_l202_202424


namespace probability_two_females_chosen_l202_202038

theorem probability_two_females_chosen (total_contestants : ℕ) (female_contestants : ℕ) (chosen_contestants : ℕ) :
  total_contestants = 8 → female_contestants = 5 → chosen_contestants = 2 → 
  (finset.card (finset.filter (λ p : Finset (Fin 8), ∀ q ∈ p, q.val < 5) ((finset.univ : Finset (Fin 8)).powersetLen 2))) / 
  (finset.card ((finset.univ : Finset (Fin 8)).powersetLen 2).to_float) = 5 / 14 := 
by 
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end probability_two_females_chosen_l202_202038


namespace light_flash_time_l202_202739

/--
A light flashes every few seconds. In 3/4 of an hour, it flashes 300 times.
Prove that it takes 9 seconds for the light to flash once.
-/
theorem light_flash_time : 
  (3 / 4 * 60 * 60) / 300 = 9 :=
by
  sorry

end light_flash_time_l202_202739


namespace solution_set_inequalities_l202_202888

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202888


namespace sam_received_87_l202_202834

def sam_total_money : Nat :=
  sorry

theorem sam_received_87 (spent left_over : Nat) (h1 : spent = 64) (h2 : left_over = 23) :
  sam_total_money = spent + left_over :=
by
  rw [h1, h2]
  sorry

example : sam_total_money = 64 + 23 :=
  sam_received_87 64 23 rfl rfl

end sam_received_87_l202_202834


namespace total_amount_paid_l202_202278

-- Definitions based on conditions
def original_price : ℝ := 100
def discount_rate : ℝ := 0.20
def additional_discount : ℝ := 5
def sales_tax_rate : ℝ := 0.08

-- Theorem statement
theorem total_amount_paid :
  let discounted_price := original_price * (1 - discount_rate)
  let final_price := discounted_price - additional_discount
  let total_price_with_tax := final_price * (1 + sales_tax_rate)
  total_price_with_tax = 81 := sorry

end total_amount_paid_l202_202278


namespace four_fours_expressions_l202_202717

theorem four_fours_expressions :
  (4 * 4 + 4) / 4 = 5 ∧
  4 + (4 + 4) / 2 = 6 ∧
  4 + 4 - 4 / 4 = 7 ∧
  4 + 4 + 4 - 4 = 8 ∧
  4 + 4 + 4 / 4 = 9 :=
by
  sorry

end four_fours_expressions_l202_202717


namespace example_problem_l202_202589

variable (a b c d : ℝ)

theorem example_problem :
  (a + (b + c - d) = a + b + c - d) ∧
  (a - (b - c + d) = a - b + c - d) ∧
  (a - b - (c - d) ≠ a - b - c - d) ∧
  (a + b - (-c - d) = a + b + c + d) :=
by {
  sorry
}

end example_problem_l202_202589


namespace max_pages_l202_202212

theorem max_pages (cents_available : ℕ) (cents_per_page : ℕ) : 
  cents_available = 2500 → 
  cents_per_page = 3 → 
  (cents_available / cents_per_page) = 833 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end max_pages_l202_202212


namespace race_distance_l202_202377

/-- Given that Sasha, Lesha, and Kolya start a 100m race simultaneously and run at constant velocities,
when Sasha finishes, Lesha is 10m behind, and when Lesha finishes, Kolya is 10m behind.
Prove that the distance between Sasha and Kolya when Sasha finishes is 19 meters. -/
theorem race_distance
    (v_S v_L v_K : ℝ)
    (h1 : 100 / v_S - 100 / v_L = 10 / v_L)
    (h2 : 100 / v_L - 100 / v_K = 10 / v_K) :
    100 - 81 = 19 :=
by
  sorry

end race_distance_l202_202377


namespace min_value_frac_l202_202081

theorem min_value_frac (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  (1 / m + 2 / n) = 8 :=
sorry

end min_value_frac_l202_202081


namespace difference_of_squares_l202_202664

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := 
by
  sorry

end difference_of_squares_l202_202664


namespace proof_problem_l202_202768

variable {a b x y : ℝ}

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem proof_problem : dollar ((x + y) ^ 2) (y ^ 2 + x ^ 2) = 4 * x ^ 2 * y ^ 2 := by
  sorry

end proof_problem_l202_202768


namespace max_volume_of_sphere_in_cube_l202_202666

theorem max_volume_of_sphere_in_cube (a : ℝ) (h : a = 1) : 
  ∃ V, V = π / 6 ∧ 
        ∀ (r : ℝ), r = a / 2 →
        V = (4 / 3) * π * r^3 :=
by
  sorry

end max_volume_of_sphere_in_cube_l202_202666


namespace second_hand_travel_distance_l202_202854

theorem second_hand_travel_distance (r : ℝ) (t : ℝ) : 
  r = 10 → t = 45 → 2 * t * π * r = 900 * π :=
by
  intro r_def t_def
  sorry

end second_hand_travel_distance_l202_202854


namespace distinct_triangles_count_l202_202509

theorem distinct_triangles_count (n : ℕ) (hn : 0 < n) : 
  (∃ triangles_count, triangles_count = ⌊((n+1)^2 : ℝ)/4⌋) :=
sorry

end distinct_triangles_count_l202_202509


namespace five_level_pyramid_has_80_pieces_l202_202617

-- Definitions based on problem conditions
def rods_per_level (level : ℕ) : ℕ :=
  if level = 1 then 4
  else if level = 2 then 8
  else if level = 3 then 12
  else if level = 4 then 16
  else if level = 5 then 20
  else 0

def connectors_per_level_transition : ℕ := 4

-- The total rods used for a five-level pyramid
def total_rods_five_levels : ℕ :=
  rods_per_level 1 + rods_per_level 2 + rods_per_level 3 + rods_per_level 4 + rods_per_level 5

-- The total connectors used for a five-level pyramid
def total_connectors_five_levels : ℕ :=
  connectors_per_level_transition * 5

-- The total pieces required for a five-level pyramid
def total_pieces_five_levels : ℕ :=
  total_rods_five_levels + total_connectors_five_levels

-- Main theorem statement for the proof problem
theorem five_level_pyramid_has_80_pieces : 
  total_pieces_five_levels = 80 :=
by
  -- We expect the total_pieces_five_levels to be equal to 80
  sorry

end five_level_pyramid_has_80_pieces_l202_202617


namespace tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m_l202_202482

theorem tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m (m : ℝ) (α : ℝ)
  (h1 : Real.tan α = m / 3)
  (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
sorry

end tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m_l202_202482


namespace sum_of_first_4_terms_l202_202036

variable {a : ℕ → ℝ} -- Define the arithmetic sequence

-- Axiom: Definition of an arithmetic sequence
axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n, a (n + 1) = a n + d

-- Axiom: Given condition a₂ + a₃ = 6
axiom a2_a3_sum (a : ℕ → ℝ) : a 1 + a 2 = 6 -- Note: a₂ is a 1, and a₃ is a 2 because Lean is 0-based

-- Theorem: We need to prove S₄ = 12
theorem sum_of_first_4_terms (a : ℕ → ℝ) (d : ℝ) [arithmetic_sequence a d] [a2_a3_sum a] : 
  a 0 + a 1 + a 2 + a 3 = 12 := 
sorry

end sum_of_first_4_terms_l202_202036


namespace P_subset_Q_l202_202814

def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x > 0}

theorem P_subset_Q : P ⊂ Q :=
by
  sorry

end P_subset_Q_l202_202814


namespace not_p_equiv_exists_leq_sin_l202_202773

-- Define the conditions as a Lean proposition
def p : Prop := ∀ x : ℝ, x > Real.sin x

-- State the problem as a theorem to be proved
theorem not_p_equiv_exists_leq_sin : ¬p = ∃ x : ℝ, x ≤ Real.sin x := 
by sorry

end not_p_equiv_exists_leq_sin_l202_202773


namespace distance_between_sasha_and_kolya_when_sasha_finished_l202_202392

-- Definitions based on the problem conditions
def distance_sasha : ℝ := 100
def distance_lesha_when_sasha_finished : ℝ := 90
def distance_kolya_when_lesha_finished : ℝ := 90

def velocity_lesha (v_s : ℝ) : ℝ := 0.9 * v_s
def velocity_kolya (v_s : ℝ) : ℝ := 0.81 * v_s

-- Theorem statement
theorem distance_between_sasha_and_kolya_when_sasha_finished (v_s : ℝ) :
  distance_sasha - (velocity_kolya v_s * (distance_sasha / v_s)) = 19 :=
  by sorry

end distance_between_sasha_and_kolya_when_sasha_finished_l202_202392


namespace product_prob_less_than_36_is_67_over_72_l202_202537

def prob_product_less_than_36 : ℚ :=
  let P := [1, 2, 3, 4, 5, 6]
  let M := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (P.bind (λ p, M.filter (λ m, p * m < 36))).length / (P.length * M.length : ℚ)

theorem product_prob_less_than_36_is_67_over_72 :
  prob_product_less_than_36 = 67 / 72 :=
by
  sorry

end product_prob_less_than_36_is_67_over_72_l202_202537


namespace domain_f_l202_202553

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x) / ((x^2) - 4)

theorem domain_f : {x : ℝ | 0 ≤ x ∧ x ≠ 2} = {x | 0 ≤ x ∧ x < 2} ∪ {x | x > 2} :=
by sorry

end domain_f_l202_202553


namespace ab_bc_cd_da_leq_1_over_4_l202_202369

theorem ab_bc_cd_da_leq_1_over_4 (a b c d : ℝ) (h : a + b + c + d = 1) : 
  a * b + b * c + c * d + d * a ≤ 1 / 4 := 
sorry

end ab_bc_cd_da_leq_1_over_4_l202_202369


namespace quadratic_function_example_l202_202640

theorem quadratic_function_example : ∃ a b c : ℝ, 
  (∀ x : ℝ, (a * x^2 + b * x + c = 0) ↔ (x = 1 ∨ x = 5)) ∧ 
  (a * 3^2 + b * 3 + c = 8) ∧ 
  (a = -2 ∧ b = 12 ∧ c = -10) :=
by
  sorry

end quadratic_function_example_l202_202640


namespace haley_marbles_l202_202507

theorem haley_marbles (boys : ℕ) (marbles_per_boy : ℕ) (total_marbles : ℕ) 
  (h1 : boys = 11) (h2 : marbles_per_boy = 9) : total_marbles = 99 :=
by
  sorry

end haley_marbles_l202_202507


namespace polygon_sides_l202_202089

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l202_202089


namespace ines_bought_3_pounds_l202_202044

-- Define initial and remaining money of Ines
def initial_money : ℕ := 20
def remaining_money : ℕ := 14

-- Define the cost per pound of peaches
def cost_per_pound : ℕ := 2

-- The total money spent on peaches
def money_spent := initial_money - remaining_money

-- The number of pounds of peaches bought
def pounds_of_peaches := money_spent / cost_per_pound

-- The proof problem
theorem ines_bought_3_pounds :
  pounds_of_peaches = 3 :=
by
  sorry

end ines_bought_3_pounds_l202_202044


namespace shortest_distance_phenomena_explained_l202_202745

def condition1 : Prop :=
  ∀ (a b : ℕ), (exists nail1 : ℕ, exists nail2 : ℕ, nail1 ≠ nail2) → (exists wall : ℕ, wall = a + b)

def condition2 : Prop :=
  ∀ (tree1 tree2 tree3 : ℕ), tree1 ≠ tree2 → tree2 ≠ tree3 → (tree1 + tree2 + tree3) / 3 = tree2

def condition3 : Prop :=
  ∀ (A B : ℕ), ∃ (C : ℕ), C = (B - A) → (A = B - (B - A))

def condition4 : Prop :=
  ∀ (dist : ℕ), dist = 0 → exists shortest : ℕ, shortest < dist

-- The following theorem needs to be proven to match our mathematical problem
theorem shortest_distance_phenomena_explained :
  condition3 ∧ condition4 :=
by
  sorry

end shortest_distance_phenomena_explained_l202_202745


namespace count_primes_with_squares_in_range_l202_202790

theorem count_primes_with_squares_in_range : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, Prime n ∧ 5000 < n^2 ∧ n^2 < 9000) ∧ 
    S.card = 5 :=
by
  sorry

end count_primes_with_squares_in_range_l202_202790


namespace score_not_possible_l202_202341

theorem score_not_possible (c u i : ℕ) (score : ℤ) :
  c + u + i = 25 ∧ score = 79 → score ≠ 5 * c + 3 * u - 25 := by
  intro h
  sorry

end score_not_possible_l202_202341


namespace product_prob_less_than_36_is_67_over_72_l202_202536

def prob_product_less_than_36 : ℚ :=
  let P := [1, 2, 3, 4, 5, 6]
  let M := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (P.bind (λ p, M.filter (λ m, p * m < 36))).length / (P.length * M.length : ℚ)

theorem product_prob_less_than_36_is_67_over_72 :
  prob_product_less_than_36 = 67 / 72 :=
by
  sorry

end product_prob_less_than_36_is_67_over_72_l202_202536


namespace households_accommodated_l202_202277

theorem households_accommodated (floors_per_building : ℕ)
                                (households_per_floor : ℕ)
                                (number_of_buildings : ℕ)
                                (total_households : ℕ)
                                (h1 : floors_per_building = 16)
                                (h2 : households_per_floor = 12)
                                (h3 : number_of_buildings = 10)
                                : total_households = 1920 :=
by
  sorry

end households_accommodated_l202_202277


namespace solution_set_of_inequalities_l202_202972

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l202_202972


namespace sqrt_range_l202_202035

theorem sqrt_range (x : ℝ) : (1 - x ≥ 0) ↔ (x ≤ 1) := sorry

end sqrt_range_l202_202035


namespace proof_f_2017_l202_202681

-- Define the conditions provided in the problem
variable (f : ℝ → ℝ)
variable (hf : ∀ x, f (-x) = -f x) -- f is an odd function
variable (h1 : ∀ x, f (-x + 1) = f (x + 1))
variable (h2 : f (-1) = 1)

-- Define the Lean statement that proves the correct answer
theorem proof_f_2017 : f 2017 = -1 :=
sorry

end proof_f_2017_l202_202681


namespace polynomial_roots_bounds_l202_202302

theorem polynomial_roots_bounds (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ (x1^4 + 3*p*x1^3 + x1^2 + 3*p*x1 + 1 = 0) ∧ (x2^4 + 3*p*x2^3 + x2^2 + 3*p*x2 + 1 = 0)) ↔ p ∈ Set.Iio (1 / 4) := by
sorry

end polynomial_roots_bounds_l202_202302


namespace solution_set_system_of_inequalities_l202_202921

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202921


namespace solution_set_system_of_inequalities_l202_202932

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l202_202932


namespace solve_inequalities_l202_202995

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202995


namespace right_triangle_area_l202_202254

theorem right_triangle_area (a b c : ℝ) (h₀ : a = 24) (h₁ : c = 30) (h2 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 216 :=
by
  sorry

end right_triangle_area_l202_202254


namespace polygon_sides_l202_202114

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l202_202114


namespace sum_of_roots_l202_202680

theorem sum_of_roots (α β : ℝ) (h1 : α^2 - 4 * α + 3 = 0) (h2 : β^2 - 4 * β + 3 = 0) (h3 : α ≠ β) :
  α + β = 4 :=
sorry

end sum_of_roots_l202_202680


namespace find_a_value_l202_202357

theorem find_a_value 
  (A : Set ℝ := {x | x^2 - 4 ≤ 0})
  (B : Set ℝ := {x | 2 * x + a ≤ 0})
  (intersection : A ∩ B = {x | -2 ≤ x ∧ x ≤ 1}) : a = -2 :=
by
  sorry

end find_a_value_l202_202357


namespace remaining_blocks_to_walk_l202_202752

noncomputable def total_blocks : ℕ := 11 + 6 + 8
noncomputable def walked_blocks : ℕ := 5

theorem remaining_blocks_to_walk : total_blocks - walked_blocks = 20 := by
  sorry

end remaining_blocks_to_walk_l202_202752


namespace unpainted_cubes_count_l202_202139

/- Definitions of the conditions -/
def total_cubes : ℕ := 6 * 6 * 6
def painted_faces_per_face : ℕ := 4
def total_faces : ℕ := 6
def painted_faces : ℕ := painted_faces_per_face * total_faces
def overlapped_painted_faces : ℕ := 4 -- Each center four squares on one face corresponds to a center square on the opposite face.
def unique_painted_cubes : ℕ := painted_faces / 2

/- Lean Theorem statement that corresponds to proving the question asked in the problem -/
theorem unpainted_cubes_count : 
  total_cubes - unique_painted_cubes = 208 :=
  by
    sorry

end unpainted_cubes_count_l202_202139


namespace inverse_function_fixed_point_l202_202488

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the condition that graph of y = f(x-1) passes through the point (1, 2)
def passes_through (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f (a - 1) = b

-- State the main theorem to prove
theorem inverse_function_fixed_point {f : ℝ → ℝ} (h : passes_through f 1 2) :
  ∃ x, x = 2 ∧ f x = 0 :=
sorry

end inverse_function_fixed_point_l202_202488


namespace solution_set_of_linear_inequalities_l202_202914

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l202_202914


namespace solve_inequalities_l202_202993

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202993


namespace solution_set_inequalities_l202_202885

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l202_202885


namespace fallen_tree_trunk_length_l202_202280

noncomputable def tiger_speed (tiger_length : ℕ) (time_pass_grass : ℕ) : ℕ := tiger_length / time_pass_grass

theorem fallen_tree_trunk_length
  (tiger_length : ℕ)
  (time_pass_grass : ℕ)
  (time_pass_tree : ℕ)
  (speed := tiger_speed tiger_length time_pass_grass) :
  tiger_length = 5 →
  time_pass_grass = 1 →
  time_pass_tree = 5 →
  (speed * time_pass_tree) = 25 :=
by
  intros h_tiger_length h_time_pass_grass h_time_pass_tree
  sorry

end fallen_tree_trunk_length_l202_202280


namespace smallest_number_of_players_l202_202202

theorem smallest_number_of_players :
  ∃ n, n ≡ 1 [MOD 3] ∧ n ≡ 2 [MOD 4] ∧ n ≡ 4 [MOD 6] ∧ ∃ m, n = m * m ∧ ∀ k, (k ≡ 1 [MOD 3] ∧ k ≡ 2 [MOD 4] ∧ k ≡ 4 [MOD 6] ∧ ∃ m, k = m * m) → k ≥ n :=
sorry

end smallest_number_of_players_l202_202202


namespace cos_neg_135_eq_l202_202756

noncomputable def cosine_neg_135 : ℝ :=
  Real.cos (Real.Angle.ofRealDegree (-135.0))

theorem cos_neg_135_eq :
  cosine_neg_135 = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_neg_135_eq_l202_202756


namespace solution_set_linear_inequalities_l202_202895

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l202_202895


namespace solve_inequalities_l202_202992

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l202_202992


namespace length_of_BC_l202_202032

noncomputable def perimeter (a b c : ℝ) := a + b + c
noncomputable def area (b c : ℝ) (A : ℝ) := 0.5 * b * c * (Real.sin A)

theorem length_of_BC
  (a b c : ℝ)
  (h_perimeter : perimeter a b c = 20)
  (h_area : area b c (Real.pi / 3) = 10 * Real.sqrt 3) :
  a = 7 :=
by
  sorry

end length_of_BC_l202_202032


namespace increase_average_by_3_l202_202842

theorem increase_average_by_3 (x : ℕ) (average_initial : ℕ := 32) (matches_initial : ℕ := 10) (score_11th_match : ℕ := 65) :
  (matches_initial * average_initial + score_11th_match = 11 * (average_initial + x)) → x = 3 := 
sorry

end increase_average_by_3_l202_202842
