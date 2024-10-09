import Mathlib

namespace overall_loss_is_correct_l656_65630

-- Define the conditions
def worth_of_stock : ℝ := 17500
def percent_stock_sold_at_profit : ℝ := 0.20
def profit_rate : ℝ := 0.10
def percent_stock_sold_at_loss : ℝ := 0.80
def loss_rate : ℝ := 0.05

-- Define the calculations based on the conditions
def worth_sold_at_profit : ℝ := percent_stock_sold_at_profit * worth_of_stock
def profit_amount : ℝ := profit_rate * worth_sold_at_profit

def worth_sold_at_loss : ℝ := percent_stock_sold_at_loss * worth_of_stock
def loss_amount : ℝ := loss_rate * worth_sold_at_loss

-- Define the overall loss amount
def overall_loss : ℝ := loss_amount - profit_amount

-- Theorem to prove that the calculated overall loss amount matches the expected loss amount
theorem overall_loss_is_correct :
  overall_loss = 350 :=
by
  sorry

end overall_loss_is_correct_l656_65630


namespace find_last_year_rate_l656_65642

-- Define the problem setting with types and values (conditions)
def last_year_rate (r : ℝ) : Prop := 
  -- Let r be the annual interest rate last year
  1.1 * r = 0.09

-- Define the theorem to prove the interest rate last year given this year's rate
theorem find_last_year_rate :
  ∃ r : ℝ, last_year_rate r ∧ r = 0.09 / 1.1 := 
by
  sorry

end find_last_year_rate_l656_65642


namespace total_selection_methods_l656_65636

theorem total_selection_methods (synthetic_students : ℕ) (analytical_students : ℕ)
  (h_synthetic : synthetic_students = 5) (h_analytical : analytical_students = 3) :
  synthetic_students + analytical_students = 8 :=
by
  -- Proof is omitted
  sorry

end total_selection_methods_l656_65636


namespace sum_of_coords_of_circle_center_l656_65661

theorem sum_of_coords_of_circle_center (x y : ℝ) :
  (x^2 + y^2 = 4 * x - 6 * y + 9) → x + y = -1 :=
by
  sorry

end sum_of_coords_of_circle_center_l656_65661


namespace line_passes_through_fixed_point_l656_65628

theorem line_passes_through_fixed_point (p q : ℝ) (h : p + 2 * q - 1 = 0) :
  p * (1/2) + 3 * (-1/6) + q = 0 :=
by
  -- placeholders for the actual proof steps
  sorry

end line_passes_through_fixed_point_l656_65628


namespace find_x_l656_65641

noncomputable def series_sum (x : ℝ) : ℝ :=
∑' n : ℕ, (1 + 6 * n) * x^n

theorem find_x (x : ℝ) (h : series_sum x = 100) (hx : |x| < 1) : x = 3 / 5 := 
sorry

end find_x_l656_65641


namespace square_area_l656_65695

theorem square_area 
  (s r l : ℝ)
  (h_r_s : r = s)
  (h_l_r : l = (2/5) * r)
  (h_area_rect : l * 10 = 120) : 
  s^2 = 900 := by
  -- Proof will go here
  sorry

end square_area_l656_65695


namespace acme_cheaper_than_beta_l656_65648

theorem acme_cheaper_than_beta (x : ℕ) :
  (50 + 9 * x < 25 + 15 * x) ↔ (5 ≤ x) :=
by sorry

end acme_cheaper_than_beta_l656_65648


namespace math_problem_l656_65698

variable {a b : ℕ → ℕ}

-- Condition 1: a_n is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m

-- Condition 2: 2a₂ - a₇² + 2a₁₂ = 0
def satisfies_equation (a : ℕ → ℕ) : Prop :=
  2 * a 2 - (a 7)^2 + 2 * a 12 = 0

-- Condition 3: b_n is a geometric sequence
def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, b (n + m) = b n * b m

-- Condition 4: b₇ = a₇
def b7_eq_a7 (a b : ℕ → ℕ) : Prop :=
  b 7 = a 7

-- To prove: b₅ * b₉ = 16
theorem math_problem (a b : ℕ → ℕ)
  (h₁ : is_arithmetic_sequence a)
  (h₂ : satisfies_equation a)
  (h₃ : is_geometric_sequence b)
  (h₄ : b7_eq_a7 a b) :
  b 5 * b 9 = 16 :=
sorry

end math_problem_l656_65698


namespace smallest_fraction_gt_five_sevenths_l656_65677

theorem smallest_fraction_gt_five_sevenths (a b : ℕ) (h1 : 10 ≤ a ∧ a ≤ 99) (h2 : 10 ≤ b ∧ b ≤ 99) (h3 : 7 * a > 5 * b) : a = 68 ∧ b = 95 :=
sorry

end smallest_fraction_gt_five_sevenths_l656_65677


namespace find_coeff_a9_l656_65619

theorem find_coeff_a9 (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) :
  (x^3 + x^10 = a + a1 * (x + 1) + a2 * (x + 1)^2 + 
  a3 * (x + 1)^3 + a4 * (x + 1)^4 + a5 * (x + 1)^5 + 
  a6 * (x + 1)^6 + a7 * (x + 1)^7 + a8 * (x + 1)^8 + 
  a9 * (x + 1)^9 + a10 * (x + 1)^10) →
  a9 = -10 :=
sorry

end find_coeff_a9_l656_65619


namespace gcd_459_357_l656_65664

theorem gcd_459_357 : gcd 459 357 = 51 := 
sorry

end gcd_459_357_l656_65664


namespace race_positions_l656_65668

variable (nabeel marzuq arabi rafsan lian rahul : ℕ)

theorem race_positions :
  (arabi = 6) →
  (arabi = rafsan + 1) →
  (rafsan = rahul + 2) →
  (rahul = nabeel + 1) →
  (nabeel = marzuq + 6) →
  (marzuq = 8) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end race_positions_l656_65668


namespace f_constant_1_l656_65670

theorem f_constant_1 (f : ℕ → ℕ) (h1 : ∀ n : ℕ, 0 < n → f (n + f n) = f n)
  (h2 : ∃ n0 : ℕ, 0 < n0 ∧ f n0 = 1) : ∀ n : ℕ, f n = 1 := 
by
  sorry

end f_constant_1_l656_65670


namespace Debby_jogging_plan_l656_65644

def Monday_jog : ℝ := 3
def Tuesday_jog : ℝ := Monday_jog * 1.1
def Wednesday_jog : ℝ := 0
def Thursday_jog : ℝ := Tuesday_jog * 1.1
def Saturday_jog : ℝ := Thursday_jog * 2.5
def total_distance : ℝ := Monday_jog + Tuesday_jog + Thursday_jog + Saturday_jog
def weekly_goal : ℝ := 40
def Sunday_jog : ℝ := weekly_goal - total_distance

theorem Debby_jogging_plan :
  Tuesday_jog = 3.3 ∧
  Thursday_jog = 3.63 ∧
  Saturday_jog = 9.075 ∧
  Sunday_jog = 21.995 :=
by
  -- Proof goes here, but is omitted as the problem statement requires only the theorem outline.
  sorry

end Debby_jogging_plan_l656_65644


namespace quadratic_k_value_l656_65639

theorem quadratic_k_value (a b k : ℝ) (h_eq : a * b + 2 * a + 2 * b = 1)
  (h_roots : Polynomial.eval₂ (RingHom.id ℝ) a (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0 ∧
             Polynomial.eval₂ (RingHom.id ℝ) b (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0) : 
  k = -5 :=
by
  sorry

end quadratic_k_value_l656_65639


namespace general_term_of_arithmetic_seq_sum_of_first_n_terms_b_n_l656_65687

theorem general_term_of_arithmetic_seq
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (a_2_eq_3 : a_n 2 = 3)
  (S_4_eq_16 : S_n 4 = 16) :
  (∀ n, a_n n = 2 * n - 1) :=
sorry

theorem sum_of_first_n_terms_b_n
  (a_n : ℕ → ℕ)
  (b_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (general_formula_a_n : ∀ n, a_n n = 2 * n - 1)
  (b_n_definition : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1))) :
  (∀ n, T_n n = n / (2 * n + 1)) :=
sorry

end general_term_of_arithmetic_seq_sum_of_first_n_terms_b_n_l656_65687


namespace apple_and_pear_costs_l656_65645

theorem apple_and_pear_costs (x y : ℝ) (h1 : x + 2 * y = 194) (h2 : 2 * x + 5 * y = 458) : 
  y = 70 ∧ x = 54 := 
by 
  sorry

end apple_and_pear_costs_l656_65645


namespace value_of_k_through_point_l656_65632

noncomputable def inverse_proportion_function (x : ℝ) (k : ℝ) : ℝ :=
  k / x

theorem value_of_k_through_point (k : ℝ) (h : k ≠ 0) : inverse_proportion_function 2 k = 3 → k = 6 :=
by
  sorry

end value_of_k_through_point_l656_65632


namespace correct_article_usage_l656_65671

def sentence : String :=
  "While he was at ____ college, he took part in the march, and was soon thrown into ____ prison."

def rules_for_articles (context : String) (noun : String) : String → Bool
| "the" => noun ≠ "college" ∨ context = "specific"
| ""    => noun = "college" ∨ noun = "prison"
| _     => false

theorem correct_article_usage : 
  rules_for_articles "general" "college" "" ∧ 
  rules_for_articles "general" "prison" "" :=
by
  sorry

end correct_article_usage_l656_65671


namespace cone_volume_difference_l656_65676

theorem cone_volume_difference (H R : ℝ) : ΔV = (1/12) * Real.pi * R^2 * H := 
sorry

end cone_volume_difference_l656_65676


namespace expression_value_l656_65662

theorem expression_value : 3 * (15 + 7)^2 - (15^2 + 7^2) = 1178 := by
    sorry

end expression_value_l656_65662


namespace julio_salary_l656_65635

-- Define the conditions
def customers_first_week : ℕ := 35
def customers_second_week : ℕ := 2 * customers_first_week
def customers_third_week : ℕ := 3 * customers_first_week
def commission_per_customer : ℕ := 1
def bonus : ℕ := 50
def total_earnings : ℕ := 760

-- Calculate total commission and total earnings
def commission_first_week : ℕ := customers_first_week * commission_per_customer
def commission_second_week : ℕ := customers_second_week * commission_per_customer
def commission_third_week : ℕ := customers_third_week * commission_per_customer
def total_commission : ℕ := commission_first_week + commission_second_week + commission_third_week
def total_earnings_commission_bonus : ℕ := total_commission + bonus

-- Define the proof problem
theorem julio_salary : total_earnings - total_earnings_commission_bonus = 500 :=
by
  sorry

end julio_salary_l656_65635


namespace min_odd_is_1_l656_65685

def min_odd_integers (a b c d e f : ℤ) : ℤ :=
  if (a + b) % 2 = 0 ∧ 
     (a + b + c + d) % 2 = 1 ∧ 
     (a + b + c + d + e + f) % 2 = 0 then
    1
  else
    sorry -- This should be replaced by a calculation of the true minimum based on conditions.

def satisfies_conditions (a b c d e f : ℤ) :=
  a + b = 30 ∧ 
  a + b + c + d = 47 ∧ 
  a + b + c + d + e + f = 65

theorem min_odd_is_1 (a b c d e f : ℤ) (h : satisfies_conditions a b c d e f) : 
  min_odd_integers a b c d e f = 1 := 
sorry

end min_odd_is_1_l656_65685


namespace fraction_multiplication_l656_65682

theorem fraction_multiplication : (1 / 3) * (1 / 4) * (1 / 5) * 60 = 1 := by
  sorry

end fraction_multiplication_l656_65682


namespace range_of_m_l656_65602

noncomputable def abs_sum (x : ℝ) : ℝ := |x - 5| + |x - 3|

theorem range_of_m (m : ℝ) : (∃ x : ℝ, abs_sum x < m) ↔ m > 2 := 
by 
  sorry

end range_of_m_l656_65602


namespace permissible_m_values_l656_65665

theorem permissible_m_values :
  ∀ (m : ℕ) (a : ℝ), 
  (∃ k, 2 ≤ k ∧ k ≤ 4 ∧ (3 / (6 / (2 * m + 1)) ≤ k)) → m = 2 ∨ m = 3 :=
by
  sorry

end permissible_m_values_l656_65665


namespace quadratic_rewriting_l656_65629

theorem quadratic_rewriting:
  ∃ (d e f : ℤ), (∀ x : ℝ, 4 * x^2 - 28 * x + 49 = (d * x + e)^2 + f) ∧ d * e = -14 :=
by {
  sorry
}

end quadratic_rewriting_l656_65629


namespace father_l656_65613

variable (S F : ℕ)

theorem father's_age (h1 : F = 3 * S) (h2 : F + 15 = 2 * (S + 15)) : F = 45 :=
sorry

end father_l656_65613


namespace all_are_truth_tellers_l656_65640

-- Define the possible states for Alice, Bob, and Carol
inductive State
| true_teller
| liar

-- Define the predicates for each person's statements
def alice_statement (B C : State) : Prop :=
  B = State.true_teller ∨ C = State.true_teller

def bob_statement (A C : State) : Prop :=
  A = State.true_teller ∧ C = State.true_teller

def carol_statement (A B : State) : Prop :=
  A = State.true_teller → B = State.true_teller

-- The theorem to be proved
theorem all_are_truth_tellers
    (A B C : State)
    (alice: A = State.true_teller → alice_statement B C)
    (bob: B = State.true_teller → bob_statement A C)
    (carol: C = State.true_teller → carol_statement A B)
    : A = State.true_teller ∧ B = State.true_teller ∧ C = State.true_teller :=
by
  sorry

end all_are_truth_tellers_l656_65640


namespace probability_of_drawing_white_ball_l656_65601

def total_balls (red white : ℕ) : ℕ := red + white

def number_of_white_balls : ℕ := 2

def number_of_red_balls : ℕ := 3

def probability_of_white_ball (white total : ℕ) : ℚ := white / total

-- Theorem statement
theorem probability_of_drawing_white_ball :
  probability_of_white_ball number_of_white_balls (total_balls number_of_red_balls number_of_white_balls) = 2 / 5 :=
sorry

end probability_of_drawing_white_ball_l656_65601


namespace route_time_saving_zero_l656_65659

theorem route_time_saving_zero 
  (distance_X : ℝ) (speed_X : ℝ) 
  (total_distance_Y : ℝ) (construction_distance_Y : ℝ) (construction_speed_Y : ℝ)
  (normal_distance_Y : ℝ) (normal_speed_Y : ℝ)
  (hx1 : distance_X = 7)
  (hx2 : speed_X = 35)
  (hy1 : total_distance_Y = 6)
  (hy2 : construction_distance_Y = 1)
  (hy3 : construction_speed_Y = 10)
  (hy4 : normal_distance_Y = 5)
  (hy5 : normal_speed_Y = 50) :
  (distance_X / speed_X * 60) - 
  ((construction_distance_Y / construction_speed_Y * 60) + 
  (normal_distance_Y / normal_speed_Y * 60)) = 0 := 
sorry

end route_time_saving_zero_l656_65659


namespace find_a_l656_65663

theorem find_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) (h3 : (51 ^ 2016 + a) % 13 = 0) : a = 12 :=
sorry

end find_a_l656_65663


namespace team_total_points_l656_65615

-- Definitions based on conditions
def chandra_points (akiko_points : ℕ) := 2 * akiko_points
def akiko_points (michiko_points : ℕ) := michiko_points + 4
def michiko_points (bailey_points : ℕ) := bailey_points / 2
def bailey_points := 14

-- Total points scored by the team
def total_points :=
  let michiko := michiko_points bailey_points
  let akiko := akiko_points michiko
  let chandra := chandra_points akiko
  bailey_points + michiko + akiko + chandra

theorem team_total_points : total_points = 54 := by
  sorry

end team_total_points_l656_65615


namespace cost_price_eq_l656_65653

variable (SP : Real) (profit_percentage : Real)

theorem cost_price_eq : SP = 100 → profit_percentage = 0.15 → (100 / (1 + profit_percentage)) = 86.96 :=
by
  intros hSP hProfit
  sorry

end cost_price_eq_l656_65653


namespace difference_between_oranges_and_apples_l656_65655

-- Definitions of the conditions
variables (A B P O: ℕ)
variables (h1: O = 6)
variables (h2: B = 3 * A)
variables (h3: P = B / 2)
variables (h4: A + B + P + O = 28)

-- The proof problem statement
theorem difference_between_oranges_and_apples
    (A B P O: ℕ)
    (h1: O = 6)
    (h2: B = 3 * A)
    (h3: P = B / 2)
    (h4: A + B + P + O = 28) :
    O - A = 2 :=
sorry

end difference_between_oranges_and_apples_l656_65655


namespace price_of_each_rose_l656_65683

def number_of_roses_started (roses : ℕ) : Prop := roses = 9
def number_of_roses_left (roses : ℕ) : Prop := roses = 4
def amount_earned (money : ℕ) : Prop := money = 35
def selling_price_per_rose (price : ℕ) : Prop := price = 7

theorem price_of_each_rose 
  (initial_roses sold_roses left_roses total_money price_per_rose : ℕ)
  (h1 : number_of_roses_started initial_roses)
  (h2 : number_of_roses_left left_roses)
  (h3 : amount_earned total_money)
  (h4 : initial_roses - left_roses = sold_roses)
  (h5 : total_money / sold_roses = price_per_rose) :
  selling_price_per_rose price_per_rose := 
by
  sorry

end price_of_each_rose_l656_65683


namespace min_value_expression_ge_072_l656_65609

theorem min_value_expression_ge_072 (x y z : ℝ) 
  (hx : -0.5 ≤ x ∧ x ≤ 0.5) 
  (hy : |y| ≤ 0.5) 
  (hz : 0 ≤ z ∧ z < 1) :
  ((1 / ((1 - x) * (1 - y) * (1 - z))) - (1 / ((2 + x) * (2 + y) * (2 + z)))) ≥ 0.72 := sorry

end min_value_expression_ge_072_l656_65609


namespace factorize_1_factorize_2_factorize_3_solve_system_l656_65699

-- Proving the factorization identities
theorem factorize_1 (y : ℝ) : 5 * y - 10 * y^2 = 5 * y * (1 - 2 * y) :=
by
  sorry

theorem factorize_2 (m : ℝ) : (3 * m - 1)^2 - 9 = (3 * m + 2) * (3 * m - 4) :=
by
  sorry

theorem factorize_3 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2)^2 :=
by
  sorry

-- Proving the solution to the system of equations
theorem solve_system (x y : ℝ) (h1 : x - y = 3) (h2 : x - 3 * y = -1) : x = 5 ∧ y = 2 :=
by
  sorry

end factorize_1_factorize_2_factorize_3_solve_system_l656_65699


namespace car_circuit_velocity_solution_l656_65678

theorem car_circuit_velocity_solution
    (v_s v_p v_d : ℕ)
    (h1 : v_s < v_p)
    (h2 : v_p < v_d)
    (h3 : s = d)
    (h4 : s + p + d = 600)
    (h5 : (d : ℚ) / v_s + (p : ℚ) / v_p + (d : ℚ) / v_d = 50) :
    (v_s = 7 ∧ v_p = 12 ∧ v_d = 42) ∨
    (v_s = 8 ∧ v_p = 12 ∧ v_d = 24) ∨
    (v_s = 9 ∧ v_p = 12 ∧ v_d = 18) ∨
    (v_s = 10 ∧ v_p = 12 ∧ v_d = 15) :=
by
  sorry

end car_circuit_velocity_solution_l656_65678


namespace circles_non_intersecting_l656_65631

def circle1_equation (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4
def circle2_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem circles_non_intersecting :
    (∀ (x y : ℝ), ¬(circle1_equation x y ∧ circle2_equation x y)) :=
by
  sorry

end circles_non_intersecting_l656_65631


namespace correct_calculation_l656_65672

variable (a b : ℚ)

theorem correct_calculation :
  (a / b) ^ 4 = a ^ 4 / b ^ 4 := 
by
  sorry

end correct_calculation_l656_65672


namespace tree_planting_equation_l656_65649

variables (x : ℝ)

theorem tree_planting_equation (h1 : x > 50) :
  (300 / (x - 50) = 400 / x) ≠ False :=
by
  sorry

end tree_planting_equation_l656_65649


namespace gcd_polynomial_multiple_of_345_l656_65607

theorem gcd_polynomial_multiple_of_345 (b : ℕ) (h : ∃ k : ℕ, b = 345 * k) : 
  Nat.gcd (5 * b ^ 3 + 2 * b ^ 2 + 7 * b + 69) b = 69 := 
by
  sorry

end gcd_polynomial_multiple_of_345_l656_65607


namespace prime_pairs_divisibility_l656_65633

theorem prime_pairs_divisibility:
  ∀ (p q : ℕ), (Nat.Prime p ∧ Nat.Prime q ∧ p ≤ q ∧ p * q ∣ ((5 ^ p - 2 ^ p) * (7 ^ q - 2 ^ q))) ↔ 
                (p = 3 ∧ q = 5) ∨ 
                (p = 3 ∧ q = 3) ∨ 
                (p = 5 ∧ q = 37) ∨ 
                (p = 5 ∧ q = 83) := by
  sorry

end prime_pairs_divisibility_l656_65633


namespace system_of_equations_solution_l656_65690

theorem system_of_equations_solution (x y : ℚ) :
  (3 * x^2 + 2 * y^2 + 2 * x + 3 * y = 0 ∧ 4 * x^2 - 3 * y^2 - 3 * x + 4 * y = 0) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) :=
by
  sorry

end system_of_equations_solution_l656_65690


namespace ratio_of_N_to_R_l656_65625

variables (N T R k : ℝ)

theorem ratio_of_N_to_R (h1 : T = (1 / 4) * N)
                        (h2 : R = 40)
                        (h3 : N = k * R)
                        (h4 : T + R + N = 190) :
    N / R = 3 :=
by
  sorry

end ratio_of_N_to_R_l656_65625


namespace find_base_l656_65675

noncomputable def log_base (a x : ℝ) := Real.log x / Real.log a

theorem find_base (a : ℝ) (h : 1 < a) :
  (log_base a (2 * a) - log_base a a = 1 / 2) → a = 4 :=
by
  -- skipping the proof
  sorry

end find_base_l656_65675


namespace image_length_interval_two_at_least_four_l656_65622

noncomputable def quadratic_function (p q r : ℝ) : ℝ → ℝ :=
  fun x => p * (x - q)^2 + r

theorem image_length_interval_two_at_least_four (p q r : ℝ)
  (h : ∀ I : Set ℝ, (∀ a b : ℝ, I = Set.Icc a b ∨ I = Set.Ioo a b → |b - a| = 1 → |quadratic_function p q r b - quadratic_function p q r a| ≥ 1)) :
  ∀ I' : Set ℝ, (∀ a b : ℝ, I' = Set.Icc a b ∨ I' = Set.Ioo a b → |b - a| = 2 → |quadratic_function p q r b - quadratic_function p q r a| ≥ 4) :=
by
  sorry


end image_length_interval_two_at_least_four_l656_65622


namespace area_of_lune_l656_65684

theorem area_of_lune :
  let d1 := 2
  let d2 := 4
  let r1 := d1 / 2
  let r2 := d2 / 2
  let height := r2 - r1
  let area_triangle := (1 / 2) * d1 * height
  let area_semicircle_small := (1 / 2) * π * r1^2
  let area_combined := area_triangle + area_semicircle_small
  let area_sector_large := (1 / 4) * π * r2^2
  let area_lune := area_combined - area_sector_large
  area_lune = 1 - (1 / 2) * π := 
by
  sorry

end area_of_lune_l656_65684


namespace geometric_sequence_seventh_term_l656_65646

variable {G : Type*} [Field G]

def is_geometric (a : ℕ → G) (q : G) :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → G) (q : G)
  (h1 : a 0 + a 1 = 3)
  (h2 : a 1 + a 2 = 6)
  (hq : is_geometric a q) :
  a 6 = 64 := 
sorry

end geometric_sequence_seventh_term_l656_65646


namespace problem_statement_l656_65696

theorem problem_statement
  (f : ℝ → ℝ)
  (h0 : ∀ x, 0 <= x → x <= 1 → 0 <= f x)
  (h1 : ∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 
        (f x + f y) / 2 ≤ f ((x + y) / 2) + 1) :
  ∀ (u v w : ℝ), 
    0 ≤ u ∧ u < v ∧ v < w ∧ w ≤ 1 → 
    (w - v) / (w - u) * f u + (v - u) / (w - u) * f w ≤ f v + 2 :=
by
  intros u v w h
  sorry

end problem_statement_l656_65696


namespace factor_difference_of_squares_l656_65606

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l656_65606


namespace problem_solution_l656_65634

-- Define the necessary conditions
def f (x : ℤ) : ℤ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Define the main theorem
theorem problem_solution :
  (Nat.gcd 840 1785 = 105) ∧ (f 2 = 62) :=
by {
  -- We include sorry here to indicate that the proof is omitted.
  sorry
}

end problem_solution_l656_65634


namespace volleyball_lineup_ways_l656_65638

def num_ways_lineup (team_size : ℕ) (positions : ℕ) : ℕ :=
  if positions ≤ team_size then
    Nat.descFactorial team_size positions
  else
    0

theorem volleyball_lineup_ways :
  num_ways_lineup 10 5 = 30240 :=
by
  rfl

end volleyball_lineup_ways_l656_65638


namespace num_triangles_in_n_gon_l656_65686

-- Definitions for the problem in Lean based on provided conditions
def n_gon (n : ℕ) : Type := sorry  -- Define n-gon as a polygon with n sides
def non_intersecting_diagonals (n : ℕ) : Prop := sorry  -- Define the property of non-intersecting diagonals in an n-gon
def num_triangles (n : ℕ) : ℕ := sorry  -- Define a function to calculate the number of triangles formed by the diagonals in an n-gon

-- Statement of the theorem to prove
theorem num_triangles_in_n_gon (n : ℕ) (h : non_intersecting_diagonals n) : num_triangles n = n - 2 :=
by
  sorry

end num_triangles_in_n_gon_l656_65686


namespace max_problems_missed_to_pass_l656_65689

theorem max_problems_missed_to_pass (total_problems : ℕ) (min_percentage : ℚ) 
  (h_total_problems : total_problems = 40) 
  (h_min_percentage : min_percentage = 0.85) : 
  ∃ max_missed : ℕ, max_missed = total_problems - ⌈total_problems * min_percentage⌉₊ ∧ max_missed = 6 :=
by
  sorry

end max_problems_missed_to_pass_l656_65689


namespace points_per_enemy_l656_65674

theorem points_per_enemy (total_enemies : ℕ) (destroyed_enemies : ℕ) (total_points : ℕ) 
  (h1 : total_enemies = 7)
  (h2 : destroyed_enemies = total_enemies - 2)
  (h3 : destroyed_enemies = 5)
  (h4 : total_points = 40) :
  total_points / destroyed_enemies = 8 :=
by
  sorry

end points_per_enemy_l656_65674


namespace polygon_sides_in_arithmetic_progression_l656_65620

theorem polygon_sides_in_arithmetic_progression 
  (n : ℕ) 
  (d : ℕ := 3)
  (max_angle : ℕ := 150)
  (sum_of_interior_angles : ℕ := 180 * (n - 2)) 
  (a_n : ℕ := max_angle) : 
  (max_angle - d * (n - 1) + max_angle) * n / 2 = sum_of_interior_angles → 
  n = 28 :=
by 
  sorry

end polygon_sides_in_arithmetic_progression_l656_65620


namespace decrease_in_radius_l656_65692

theorem decrease_in_radius
  (dist_summer : ℝ)
  (dist_winter : ℝ)
  (radius_summer : ℝ) 
  (mile_to_inch : ℝ)
  (π : ℝ) 
  (δr : ℝ) :
  dist_summer = 560 →
  dist_winter = 570 →
  radius_summer = 20 →
  mile_to_inch = 63360 →
  π = Real.pi →
  δr = 0.33 :=
sorry

end decrease_in_radius_l656_65692


namespace smallest_value_of_c_l656_65611

/-- The polynomial x^3 - cx^2 + dx - 2550 has three positive integer roots,
    and the product of the roots is 2550. Prove that the smallest possible value of c is 42. -/
theorem smallest_value_of_c :
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 2550 ∧ c = a + b + c) → c = 42 :=
sorry

end smallest_value_of_c_l656_65611


namespace quadratic_real_equal_roots_l656_65660

theorem quadratic_real_equal_roots (m : ℝ) :
  (∃ x : ℝ, 3*x^2 + (2*m-5)*x + 12 = 0) ↔ (m = 8.5 ∨ m = -3.5) :=
sorry

end quadratic_real_equal_roots_l656_65660


namespace cubic_sum_l656_65673

theorem cubic_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 14) : x ^ 3 + y ^ 3 = 580 :=
by 
  sorry

end cubic_sum_l656_65673


namespace M_geq_N_l656_65658

variable (a b : ℝ)

def M : ℝ := a^2 + 12 * a - 4 * b
def N : ℝ := 4 * a - 20 - b^2

theorem M_geq_N : M a b ≥ N a b := by
  sorry

end M_geq_N_l656_65658


namespace f_g_5_l656_65669

def g (x : ℕ) : ℕ := 4 * x + 10

def f (x : ℕ) : ℕ := 6 * x - 12

theorem f_g_5 : f (g 5) = 168 := by
  sorry

end f_g_5_l656_65669


namespace average_birth_rate_l656_65693

theorem average_birth_rate (B : ℕ) 
  (death_rate : ℕ := 3)
  (daily_net_increase : ℕ := 86400) 
  (intervals_per_day : ℕ := 86400 / 2) 
  (net_increase : ℕ := (B - death_rate) * intervals_per_day) : 
  net_increase = daily_net_increase → 
  B = 5 := 
sorry

end average_birth_rate_l656_65693


namespace johns_share_l656_65688

theorem johns_share
  (total_amount : ℕ)
  (ratio_john : ℕ)
  (ratio_jose : ℕ)
  (ratio_binoy : ℕ)
  (total_parts : ℕ)
  (value_per_part : ℕ)
  (johns_parts : ℕ)
  (johns_share : ℕ)
  (h1 : total_amount = 4800)
  (h2 : ratio_john = 2)
  (h3 : ratio_jose = 4)
  (h4 : ratio_binoy = 6)
  (h5 : total_parts = ratio_john + ratio_jose + ratio_binoy)
  (h6 : value_per_part = total_amount / total_parts)
  (h7 : johns_parts = ratio_john)
  (h8 : johns_share = value_per_part * johns_parts) :
  johns_share = 800 := by
  sorry

end johns_share_l656_65688


namespace option_A_correct_l656_65656

theorem option_A_correct (a b : ℝ) (h : a > b) : a + 2 > b + 2 :=
by sorry

end option_A_correct_l656_65656


namespace ellipse_hyperbola_eccentricities_l656_65643

theorem ellipse_hyperbola_eccentricities :
  ∃ x y : ℝ, (2 * x^2 - 5 * x + 2 = 0) ∧ (2 * y^2 - 5 * y + 2 = 0) ∧ 
  ((2 > 1) ∧ (0 < (1/2) ∧ (1/2 < 1))) :=
by
  sorry

end ellipse_hyperbola_eccentricities_l656_65643


namespace numbers_square_and_cube_root_l656_65652

theorem numbers_square_and_cube_root (x : ℝ) : (x^2 = x ∧ x^3 = x) ↔ (x = 0 ∨ x = 1 ∨ x = -1) :=
by
  sorry

end numbers_square_and_cube_root_l656_65652


namespace points_lie_on_hyperbola_l656_65679

def point_on_hyperbola (t : ℝ) : Prop :=
  let x := 2 * Real.exp t - 2 * Real.exp (-t)
  let y := 4 * (Real.exp t + Real.exp (-t))
  (y^2) / 16 - (x^2) / 4 = 1

theorem points_lie_on_hyperbola : ∀ t : ℝ, point_on_hyperbola t :=
by
  intro t
  sorry

end points_lie_on_hyperbola_l656_65679


namespace annual_income_correct_l656_65681

-- Define the principal amounts and interest rates
def principal_1 : ℝ := 3000
def rate_1 : ℝ := 0.085

def principal_2 : ℝ := 5000
def rate_2 : ℝ := 0.064

-- Define the interest calculations for each investment
def interest_1 : ℝ := principal_1 * rate_1
def interest_2 : ℝ := principal_2 * rate_2

-- Define the total annual income
def total_annual_income : ℝ := interest_1 + interest_2

-- Proof statement
theorem annual_income_correct : total_annual_income = 575 :=
by
  sorry

end annual_income_correct_l656_65681


namespace average_of_xyz_l656_65621

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) : (x + y + z) / 3 = 16 / 3 :=
by
  sorry

end average_of_xyz_l656_65621


namespace whitewash_all_planks_not_whitewash_all_planks_l656_65600

open Finset

variable {N : ℕ} (is_power_of_two : ∃ (k : ℕ), N = 2^(k + 1))

def f (n : ℤ) : ℤ := n^2 + 3*n - 2

def f_equiv (x y : ℤ) : Prop := 2^(Nat.log2 (2 * N)) ∣ (f x - f y)

theorem whitewash_all_planks (N : ℕ) (is_power_of_two : ∃ (k : ℕ), N = 2^(k + 1)) : 
  ∀ n ∈ range N, ∃ m ∈ range N, f m = n :=
by {
  sorry
}

theorem not_whitewash_all_planks (N : ℕ) (not_power_of_two : ¬(∃ (k : ℕ), N = 2^(k + 1))) : 
  ∃ n ∈ range N, ∀ m ∈ range N, f m ≠ n :=
by {
  sorry
}

end whitewash_all_planks_not_whitewash_all_planks_l656_65600


namespace geom_seq_value_l656_65657

variable (a_n : ℕ → ℝ)
variable (r : ℝ)
variable (π : ℝ)

-- Define the conditions
axiom geom_seq : ∀ n, a_n (n + 1) = a_n n * r
axiom sum_pi : a_n 3 + a_n 5 = π

-- Statement to prove
theorem geom_seq_value : a_n 4 * (a_n 2 + 2 * a_n 4 + a_n 6) = π^2 :=
by
  sorry

end geom_seq_value_l656_65657


namespace distance_between_points_l656_65610

theorem distance_between_points:
  dist (0, 4) (3, 0) = 5 :=
by
  sorry

end distance_between_points_l656_65610


namespace exchange_ways_100_yuan_l656_65667

theorem exchange_ways_100_yuan : ∃ n : ℕ, n = 6 ∧ (∀ (x y : ℕ), 20 * x + 10 * y = 100 ↔ y = 10 - 2 * x):=
by
  sorry

end exchange_ways_100_yuan_l656_65667


namespace total_seashells_l656_65604

def joans_seashells : Nat := 6
def jessicas_seashells : Nat := 8

theorem total_seashells : joans_seashells + jessicas_seashells = 14 :=
by
  sorry

end total_seashells_l656_65604


namespace angle_B_value_l656_65616

theorem angle_B_value (a b c A B : ℝ) (h1 : Real.sqrt 3 * a = 2 * b * Real.sin A) : 
  Real.sin B = Real.sqrt 3 / 2 ↔ (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) :=
by sorry

noncomputable def find_b_value (a : ℝ) (area : ℝ) (A B c : ℝ) (h1 : a = 6) (h2 : area = 6 * Real.sqrt 3) (h3 : c = 4) (h4 : B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) : 
  ℝ := 
if B = Real.pi / 3 then 2 * Real.sqrt 7 else Real.sqrt 76

end angle_B_value_l656_65616


namespace michael_and_truck_meet_l656_65612

/--
Assume:
1. Michael walks at 6 feet per second.
2. Trash pails are every 240 feet.
3. A truck travels at 10 feet per second and stops for 36 seconds at each pail.
4. Initially, when Michael passes a pail, the truck is 240 feet ahead.

Prove:
Michael and the truck meet every 120 seconds starting from 120 seconds.
-/
theorem michael_and_truck_meet (t : ℕ) : t ≥ 120 → (t - 120) % 120 = 0 :=
sorry

end michael_and_truck_meet_l656_65612


namespace area_square_diagonal_l656_65666

theorem area_square_diagonal (d : ℝ) (k : ℝ) :
  (∀ side : ℝ, d^2 = 2 * side^2 → side^2 = (d^2)/2) →
  (∀ A : ℝ, A = (d^2)/2 → A = k * d^2) →
  k = 1/2 :=
by
  intros h1 h2
  sorry

end area_square_diagonal_l656_65666


namespace apples_in_pile_l656_65614

/-- Assuming an initial pile of 8 apples and adding 5 more apples, there should be 13 apples in total. -/
theorem apples_in_pile (initial_apples added_apples : ℕ) (h1 : initial_apples = 8) (h2 : added_apples = 5) :
  initial_apples + added_apples = 13 :=
by
  sorry

end apples_in_pile_l656_65614


namespace decimal_equivalent_of_fraction_squared_l656_65691

theorem decimal_equivalent_of_fraction_squared : (1 / 4 : ℝ) ^ 2 = 0.0625 :=
by sorry

end decimal_equivalent_of_fraction_squared_l656_65691


namespace least_times_to_eat_l656_65697

theorem least_times_to_eat (A B C : ℕ) (h1 : A = (9 * B) / 5) (h2 : B = C / 8) : 
  A = 2 ∧ B = 1 ∧ C = 8 :=
sorry

end least_times_to_eat_l656_65697


namespace initial_pennies_l656_65605

-- Defining the conditions
def pennies_spent : Nat := 93
def pennies_left : Nat := 5

-- Question: How many pennies did Sam have in his bank initially?
theorem initial_pennies : pennies_spent + pennies_left = 98 := by
  sorry

end initial_pennies_l656_65605


namespace value_of_f_prime_at_1_l656_65618

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem value_of_f_prime_at_1 : deriv f 1 = 1 :=
by
  sorry

end value_of_f_prime_at_1_l656_65618


namespace money_left_after_shopping_l656_65624

-- Define the initial amount of money Sandy took for shopping
def initial_amount : ℝ := 310

-- Define the percentage of money spent in decimal form
def percentage_spent : ℝ := 0.30

-- Define the remaining money as per the given conditions
def remaining_money : ℝ := initial_amount * (1 - percentage_spent)

-- The statement we need to prove
theorem money_left_after_shopping :
  remaining_money = 217 :=
by
  sorry

end money_left_after_shopping_l656_65624


namespace park_needs_minimum_37_nests_l656_65617

-- Defining the number of different birds
def num_sparrows : ℕ := 5
def num_pigeons : ℕ := 3
def num_starlings : ℕ := 6
def num_robins : ℕ := 2

-- Defining the nesting requirements for each bird species
def nests_per_sparrow : ℕ := 1
def nests_per_pigeon : ℕ := 2
def nests_per_starling : ℕ := 3
def nests_per_robin : ℕ := 4

-- Definition of total minimum nests required
def min_nests_required : ℕ :=
  (num_sparrows * nests_per_sparrow) +
  (num_pigeons * nests_per_pigeon) +
  (num_starlings * nests_per_starling) +
  (num_robins * nests_per_robin)

-- Proof Statement
theorem park_needs_minimum_37_nests :
  min_nests_required = 37 :=
sorry

end park_needs_minimum_37_nests_l656_65617


namespace ab_non_positive_l656_65603

-- Define the conditions as a structure if necessary.
variables {a b : ℝ}

-- State the theorem.
theorem ab_non_positive (h : 3 * a + 8 * b = 0) : a * b ≤ 0 :=
sorry

end ab_non_positive_l656_65603


namespace find_y_and_y2_l656_65627

theorem find_y_and_y2 (d y y2 : ℤ) (h1 : 3 ^ 2 = 9) (h2 : 3 ^ 4 = 81)
  (h3 : y = 9 + d) (h4 : y2 = 81 + d) (h5 : 81 = 9 + 3 * d) :
  y = 33 ∧ y2 = 105 :=
by
  sorry

end find_y_and_y2_l656_65627


namespace Karl_miles_driven_l656_65654

theorem Karl_miles_driven
  (gas_per_mile : ℝ)
  (tank_capacity : ℝ)
  (initial_gas : ℝ)
  (first_leg_miles : ℝ)
  (refuel_gallons : ℝ)
  (final_gas_fraction : ℝ)
  (total_miles_driven : ℝ) :
  gas_per_mile = 30 →
  tank_capacity = 16 →
  initial_gas = 16 →
  first_leg_miles = 420 →
  refuel_gallons = 10 →
  final_gas_fraction = 3 / 4 →
  total_miles_driven = 420 :=
by
  sorry

end Karl_miles_driven_l656_65654


namespace arctan_sum_l656_65623

theorem arctan_sum (a b : ℝ) (h1 : a = 1/3) (h2 : (a + 1) * (b + 1) = 3) : 
  Real.arctan a + Real.arctan b = Real.arctan (19 / 7) :=
by
  sorry

end arctan_sum_l656_65623


namespace find_other_num_l656_65637

variables (a b : ℕ)

theorem find_other_num (h_gcd : Nat.gcd a b = 12) (h_lcm : Nat.lcm a b = 5040) (h_a : a = 240) :
  b = 252 :=
  sorry

end find_other_num_l656_65637


namespace smallest_solution_l656_65650

theorem smallest_solution (x : ℝ) (h : x * |x| = 2 * x + 1) : x = -1 := 
by
  sorry

end smallest_solution_l656_65650


namespace inequality_proof_l656_65651

variable {x y : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hxy : x > y) :
    2 * x + 1 / (x ^ 2 - 2 * x * y + y ^ 2) ≥ 2 * y + 3 := 
  sorry

end inequality_proof_l656_65651


namespace question1_question2_l656_65626

-- Define the function representing the inequality
def inequality (a x : ℝ) : Prop := (a * x - 5) / (x - a) < 0

-- Question 1: Compute the solution set M when a=1
theorem question1 : (setOf (λ x : ℝ => inequality 1 x)) = {x : ℝ | 1 < x ∧ x < 5} :=
by
  sorry

-- Question 2: Determine the range for a such that 3 ∈ M but 5 ∉ M
theorem question2 : (setOf (λ a : ℝ => 3 ∈ (setOf (λ x : ℝ => inequality a x)) ∧ 5 ∉ (setOf (λ x : ℝ => inequality a x)))) = 
  {a : ℝ | (1 ≤ a ∧ a < 5 / 3) ∨ (3 < a ∧ a ≤ 5)} :=
by
  sorry

end question1_question2_l656_65626


namespace cuboid_volume_l656_65647

theorem cuboid_volume (x y z : ℝ)
  (h1 : 2 * (x + y) = 20)
  (h2 : 2 * (y + z) = 32)
  (h3 : 2 * (x + z) = 28) : x * y * z = 240 := 
by
  sorry

end cuboid_volume_l656_65647


namespace probability_at_least_one_woman_l656_65694

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ)
  (h1 : total_people = 10) (h2 : men = 5) (h3 : women = 5) (h4 : selected = 3) :
  (1 - (men / total_people) * ((men - 1) / (total_people - 1)) * ((men - 2) / (total_people - 2))) = 5 / 6 :=
by
  sorry

end probability_at_least_one_woman_l656_65694


namespace sum_and_num_of_factors_eq_1767_l656_65680

theorem sum_and_num_of_factors_eq_1767 (n : ℕ) (σ d : ℕ → ℕ) :
  (σ n + d n = 1767) → 
  ∃ m : ℕ, σ m + d m = 1767 :=
by 
  sorry

end sum_and_num_of_factors_eq_1767_l656_65680


namespace store_incur_loss_of_one_percent_l656_65608

theorem store_incur_loss_of_one_percent
    (a b x : ℝ)
    (h1 : x = a * 1.1)
    (h2 : x = b * 0.9)
    : (2 * x - (a + b)) / (a + b) = -0.01 :=
by
  -- Proof goes here
  sorry

end store_incur_loss_of_one_percent_l656_65608
