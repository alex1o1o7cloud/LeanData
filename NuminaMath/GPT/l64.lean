import Mathlib

namespace find_double_pieces_l64_64002

theorem find_double_pieces (x : ℕ) 
  (h1 : 100 + 2 * x + 150 + 660 = 1000) : x = 45 :=
by sorry

end find_double_pieces_l64_64002


namespace milo_skateboarding_speed_l64_64351

theorem milo_skateboarding_speed (cory_speed milo_skateboarding_speed : ℝ) 
  (h1 : cory_speed = 12) 
  (h2 : cory_speed = 2 * milo_skateboarding_speed) : 
  milo_skateboarding_speed = 6 :=
by sorry

end milo_skateboarding_speed_l64_64351


namespace analytical_expression_when_x_in_5_7_l64_64601

noncomputable def f : ℝ → ℝ := sorry

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry
lemma symmetric_about_one (x : ℝ) : f (1 - x) = f (1 + x) := sorry
lemma values_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x ≤ 1) : f x = x := sorry

theorem analytical_expression_when_x_in_5_7 (x : ℝ) (h : 5 < x ∧ x ≤ 7) :
  f x = 6 - x :=
sorry

end analytical_expression_when_x_in_5_7_l64_64601


namespace probability_exceeds_175_l64_64361

theorem probability_exceeds_175 (P_lt_160 : ℝ) (P_160_to_175 : ℝ) (h : ℝ) :
  P_lt_160 = 0.2 → P_160_to_175 = 0.5 → 1 - P_lt_160 - P_160_to_175 = 0.3 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end probability_exceeds_175_l64_64361


namespace t_n_minus_n_even_l64_64157

noncomputable def number_of_nonempty_subsets_with_integer_average (n : ℕ) : ℕ := 
  sorry

theorem t_n_minus_n_even (N : ℕ) (hN : N > 1) :
  ∃ T_n, T_n = number_of_nonempty_subsets_with_integer_average N ∧ (T_n - N) % 2 = 0 :=
by
  sorry

end t_n_minus_n_even_l64_64157


namespace division_problem_l64_64677

-- Define the involved constants and operations
def expr1 : ℚ := 5 / 2 * 3
def expr2 : ℚ := 100 / expr1

-- Formulate the final equality
theorem division_problem : expr2 = 40 / 3 :=
  by sorry

end division_problem_l64_64677


namespace perimeter_of_equilateral_triangle_l64_64493

theorem perimeter_of_equilateral_triangle (a : ℕ) (h1 : a = 12) (h2 : ∀ sides, sides = 3) : 
  3 * a = 36 := 
by
  sorry

end perimeter_of_equilateral_triangle_l64_64493


namespace length_of_plot_is_60_l64_64369

noncomputable def plot_length (b : ℝ) : ℝ :=
  b + 20

noncomputable def plot_perimeter (b : ℝ) : ℝ :=
  2 * (plot_length b + b)

noncomputable def plot_cost_eq (b : ℝ) : Prop :=
  26.50 * plot_perimeter b = 5300

theorem length_of_plot_is_60 : ∃ b : ℝ, plot_cost_eq b ∧ plot_length b = 60 :=
sorry

end length_of_plot_is_60_l64_64369


namespace perpendicular_lines_with_foot_l64_64524

theorem perpendicular_lines_with_foot (n : ℝ) : 
  (∀ x y, 10 * x + 4 * y - 2 = 0 ↔ 2 * x - 5 * y + n = 0) ∧
  (2 * 1 - 5 * (-2) + n = 0) → n = -12 := 
by sorry

end perpendicular_lines_with_foot_l64_64524


namespace round_robin_teams_l64_64199

theorem round_robin_teams (x : ℕ) (h : x * (x - 1) / 2 = 28) : x = 8 :=
sorry

end round_robin_teams_l64_64199


namespace minimum_value_l64_64284

theorem minimum_value (x : ℝ) (h : x > 0) :
  x^3 + 12*x + 81 / x^4 = 24 := 
sorry

end minimum_value_l64_64284


namespace smallest_n_7n_eq_n7_mod_3_l64_64103

theorem smallest_n_7n_eq_n7_mod_3 : ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 3]) ∧ ∀ m : ℕ, m > 0 → (7^m ≡ m^7 [MOD 3] → m ≥ n) :=
by
  sorry

end smallest_n_7n_eq_n7_mod_3_l64_64103


namespace walter_chore_days_l64_64026

-- Definitions for the conditions
variables (b w : ℕ)  -- b: days regular, w: days exceptionally well

-- Conditions
def days_eq : Prop := b + w = 15
def earnings_eq : Prop := 3 * b + 4 * w = 47

-- The theorem stating the proof problem
theorem walter_chore_days (hb : days_eq b w) (he : earnings_eq b w) : w = 2 :=
by
  -- We only need to state the theorem; the proof is omitted.
  sorry

end walter_chore_days_l64_64026


namespace remainder_of_n_div_11_is_1_l64_64333

def A : ℕ := 20072009
def n : ℕ := 100 * A

theorem remainder_of_n_div_11_is_1 :
  (n % 11) = 1 :=
sorry

end remainder_of_n_div_11_is_1_l64_64333


namespace correct_option_is_A_l64_64700

variable (a b : ℤ)

-- Option A condition
def optionA : Prop := 3 * a^2 * b / b = 3 * a^2

-- Option B condition
def optionB : Prop := a^12 / a^3 = a^4

-- Option C condition
def optionC : Prop := (a + b)^2 = a^2 + b^2

-- Option D condition
def optionD : Prop := (-2 * a^2)^3 = 8 * a^6

theorem correct_option_is_A : 
  optionA a b ∧ ¬optionB a ∧ ¬optionC a b ∧ ¬optionD a :=
by
  sorry

end correct_option_is_A_l64_64700


namespace university_diploma_percentage_l64_64956

-- Define the conditions
variables (P N JD ND : ℝ)
-- P: total population assumed as 100% for simplicity
-- N: percentage of people with university diploma
-- JD: percentage of people who have the job of their choice
-- ND: percentage of people who do not have a university diploma but have the job of their choice
variables (A : ℝ) -- A: University diploma percentage of those who do not have the job of their choice
variable (total_diploma : ℝ)
axiom country_Z_conditions : 
  (P = 100) ∧ (ND = 18) ∧ (JD = 40) ∧ (A = 25)

-- Define the proof problem
theorem university_diploma_percentage :
  (N = ND + (JD - ND) + (total_diploma * (P - JD * (P / JD) / P))) →
  N = 37 :=
by
  sorry

end university_diploma_percentage_l64_64956


namespace least_n_divisibility_condition_l64_64837

theorem least_n_divisibility_condition :
  ∃ n : ℕ, 0 < n ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k ∣ (n^2 - n + 1) ↔ (n = 5 ∧ k = 3)) := 
sorry

end least_n_divisibility_condition_l64_64837


namespace isabelle_weeks_needed_l64_64151

def total_ticket_cost : ℕ := 20 + 10 + 10
def total_savings : ℕ := 5 + 5
def weekly_earnings : ℕ := 3
def amount_needed : ℕ := total_ticket_cost - total_savings
def weeks_needed : ℕ := amount_needed / weekly_earnings

theorem isabelle_weeks_needed 
  (ticket_cost_isabelle : ℕ := 20)
  (ticket_cost_brother : ℕ := 10)
  (savings_brothers : ℕ := 5)
  (savings_isabelle : ℕ := 5)
  (earnings_weekly : ℕ := 3)
  (total_cost := ticket_cost_isabelle + 2 * ticket_cost_brother)
  (total_savings := savings_brothers + savings_isabelle)
  (needed_amount := total_cost - total_savings)
  (weeks := needed_amount / earnings_weekly) :
  weeks = 10 :=
  by
  sorry

end isabelle_weeks_needed_l64_64151


namespace sufficient_condition_for_inequality_l64_64019

theorem sufficient_condition_for_inequality (a : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a > 4 :=
by 
  sorry

end sufficient_condition_for_inequality_l64_64019


namespace remainder_9876543210_mod_101_l64_64682

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l64_64682


namespace sum_of_six_terms_l64_64123

variable (a₁ a₂ a₃ a₄ a₅ a₆ q : ℝ)

-- Conditions
def geom_seq := a₂ = q * a₁ ∧ a₃ = q * a₂ ∧ a₄ = q * a₃ ∧ a₅ = q * a₄ ∧ a₆ = q * a₅
def cond₁ : Prop := a₁ + a₃ = 5 / 2
def cond₂ : Prop := a₂ + a₄ = 5 / 4

-- Problem Statement
theorem sum_of_six_terms : geom_seq a₁ a₂ a₃ a₄ a₅ a₆ q → cond₁ a₁ a₃ → cond₂ a₂ a₄ → 
  (a₁ * (1 - q^6) / (1 - q) = 63 / 16) := 
by 
  sorry

end sum_of_six_terms_l64_64123


namespace ages_of_boys_l64_64186

theorem ages_of_boys (a b c : ℕ) (h1 : a + b + c = 29) (h2 : a = b) (h3 : c = 11) : a = 9 :=
by
  sorry

end ages_of_boys_l64_64186


namespace clothing_price_decrease_l64_64385

theorem clothing_price_decrease (P : ℝ) (h₁ : P > 0) :
  let price_first_sale := (4 / 5) * P
  let price_second_sale := (1 / 2) * P
  let price_difference := price_first_sale - price_second_sale
  let percent_decrease := (price_difference / price_first_sale) * 100
  percent_decrease = 37.5 :=
by
  sorry

end clothing_price_decrease_l64_64385


namespace probability_winning_l64_64661

-- Define the probability of losing
def P_lose : ℚ := 5 / 8

-- Define the total probability constraint
theorem probability_winning : P_lose = 5 / 8 → (1 - P_lose) = 3 / 8 := 
by
  intro h
  rw h
  norm_num
  sorry

end probability_winning_l64_64661


namespace perpendicular_vectors_m_eq_0_or_neg2_l64_64137

theorem perpendicular_vectors_m_eq_0_or_neg2
  (m : ℝ)
  (a : ℝ × ℝ := (m, 1))
  (b : ℝ × ℝ := (1, m - 1))
  (h : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) :
  m = 0 ∨ m = -2 := sorry

end perpendicular_vectors_m_eq_0_or_neg2_l64_64137


namespace value_of_expression_l64_64062

theorem value_of_expression : (85 + 32 / 113) * 113 = 9635 :=
by
  sorry

end value_of_expression_l64_64062


namespace correct_calculation_l64_64382

/-- Conditions for the given calculations -/
def cond_a : Prop := (-2) ^ 3 = 8
def cond_b : Prop := (-3) ^ 2 = -9
def cond_c : Prop := -(3 ^ 2) = -9
def cond_d : Prop := (-2) ^ 2 = 4

/-- Prove that the correct calculation among the given is -3^2 = -9 -/
theorem correct_calculation : cond_c :=
by sorry

end correct_calculation_l64_64382


namespace difference_between_local_and_face_value_of_7_in_65793_l64_64255

theorem difference_between_local_and_face_value_of_7_in_65793 :
  let numeral := 65793
  let digit := 7
  let place := 100
  let local_value := digit * place
  let face_value := digit
  local_value - face_value = 693 := 
by
  sorry

end difference_between_local_and_face_value_of_7_in_65793_l64_64255


namespace y_intercept_of_line_l64_64030

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end y_intercept_of_line_l64_64030


namespace initial_balance_l64_64870

-- Define the conditions given in the problem
def transferred_percent_of_balance (X : ℝ) : ℝ := 0.15 * X
def balance_after_transfer (X : ℝ) : ℝ := 0.85 * X
def final_balance_after_refund (X : ℝ) (refund : ℝ) : ℝ := 0.85 * X + refund

-- Define the given values
def refund : ℝ := 450
def final_balance : ℝ := 30000

-- The theorem statement to prove the initial balance
theorem initial_balance (X : ℝ) (h : final_balance_after_refund X refund = final_balance) : 
  X = 34564.71 :=
by
  sorry

end initial_balance_l64_64870


namespace smallest_odd_prime_factor_l64_64433

theorem smallest_odd_prime_factor (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  (2023 ^ 8 + 1) % p = 0 ↔ p = 17 := 
by
  sorry

end smallest_odd_prime_factor_l64_64433


namespace solve_system_of_equations_l64_64516

theorem solve_system_of_equations :
  ∃ x y : ℝ, (2 * x - 5 * y = -1) ∧ (-4 * x + y = -7) ∧ (x = 2) ∧ (y = 1) :=
by
  -- proof omitted
  sorry

end solve_system_of_equations_l64_64516


namespace determine_common_ratio_l64_64965

variable (a : ℕ → ℝ) (q : ℝ)

-- Given conditions
axiom a2 : a 2 = 1 / 2
axiom a5 : a 5 = 4
axiom geom_seq_def : ∀ n, a n = a 1 * q ^ (n - 1)

-- Prove the common ratio q == 2
theorem determine_common_ratio : q = 2 :=
by
  -- here we should unfold the proof steps given in the solution
  sorry

end determine_common_ratio_l64_64965


namespace center_of_square_l64_64000

theorem center_of_square (O : ℝ × ℝ) (A B C D : ℝ × ℝ) 
  (hAB : dist A B = 1) 
  (hA : A = (0, 0)) 
  (hB : B = (1, 0)) 
  (hC : C = (1, 1)) 
  (hD : D = (0, 1)) 
  (h_sum_squares : (dist O A)^2 + (dist O B)^2 + (dist O C)^2 + (dist O D)^2 = 2): 
  O = (1/2, 1/2) :=
by sorry

end center_of_square_l64_64000


namespace min_area_ABCD_l64_64408

section Quadrilateral

variables {S1 S2 S3 S4 : ℝ}

-- Define the areas of the triangles
def area_APB := S1
def area_BPC := S2
def area_CPD := S3
def area_DPA := S4

-- Condition: Product of the areas of ΔAPB and ΔCPD is 36
axiom prod_APB_CPD : S1 * S3 = 36

-- We need to prove that the minimum area of the quadrilateral ABCD is 24
theorem min_area_ABCD : S1 + S2 + S3 + S4 ≥ 24 :=
by
  sorry

end Quadrilateral

end min_area_ABCD_l64_64408


namespace marbles_cost_correct_l64_64796

def total_cost : ℝ := 20.52
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52

-- The problem is to prove that the amount spent on marbles is $9.05
def amount_spent_on_marbles : ℝ :=
  total_cost - (cost_football + cost_baseball)

theorem marbles_cost_correct :
  amount_spent_on_marbles = 9.05 :=
by
  -- The proof goes here.
  sorry

end marbles_cost_correct_l64_64796


namespace relationship_between_m_and_n_l64_64316

theorem relationship_between_m_and_n
  (m n : ℝ)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y - 4 = 0)
  (line_eq : ∀ (x y : ℝ), m * x + 2 * n * y - 4 = 0) :
  m - n - 2 = 0 := 
  sorry

end relationship_between_m_and_n_l64_64316


namespace sufficient_remedy_l64_64851

-- Definitions based on conditions
def aspirin_relieves_headache : Prop := true
def aspirin_relieves_knee_rheumatism : Prop := true
def aspirin_causes_heart_pain : Prop := true
def aspirin_causes_stomach_pain : Prop := true

def homeopathic_relieves_heart_issues : Prop := true
def homeopathic_relieves_stomach_issues : Prop := true
def homeopathic_causes_hip_rheumatism : Prop := true

def antibiotics_cure_migraines : Prop := true
def antibiotics_cure_heart_pain : Prop := true
def antibiotics_cause_stomach_pain : Prop := true
def antibiotics_cause_knee_pain : Prop := true
def antibiotics_cause_itching : Prop := true

def cortisone_relieves_itching : Prop := true
def cortisone_relieves_knee_rheumatism : Prop := true
def cortisone_exacerbates_hip_rheumatism : Prop := true

def warm_compress_relieves_itching : Prop := true
def warm_compress_relieves_stomach_pain : Prop := true

def severe_headache_morning : Prop := true
def impaired_ability_to_think : Prop := severe_headache_morning

-- Statement of the proof problem
theorem sufficient_remedy :
  (aspirin_relieves_headache ∧ antibiotics_cure_heart_pain ∧ warm_compress_relieves_itching ∧ warm_compress_relieves_stomach_pain) →
  (impaired_ability_to_think → true) :=
by
  sorry

end sufficient_remedy_l64_64851


namespace unique_f_l64_64780

def S : Set ℕ := { x | 1 ≤ x ∧ x ≤ 10^10 }

noncomputable def f : ℕ → ℕ := sorry

axiom f_cond (x : ℕ) (hx : x ∈ S) :
  f (x + 1) % (10^10) = (f (f x) + 1) % (10^10)

axiom f_boundary :
  f (10^10 + 1) % (10^10) = f 1

theorem unique_f (x : ℕ) (hx : x ∈ S) :
  f x % (10^10) = x % (10^10) :=
sorry

end unique_f_l64_64780


namespace cathy_wallet_left_money_l64_64569

noncomputable def amount_left_in_wallet (initial : ℝ) (dad_amount : ℝ) (book_cost : ℝ) (saving_percentage : ℝ) : ℝ :=
  let mom_amount := 2 * dad_amount
  let total_initial := initial + dad_amount + mom_amount
  let after_purchase := total_initial - book_cost
  let saved_amount := saving_percentage * after_purchase
  after_purchase - saved_amount

theorem cathy_wallet_left_money :
  amount_left_in_wallet 12 25 15 0.20 = 57.60 :=
by 
  sorry

end cathy_wallet_left_money_l64_64569


namespace even_heads_probability_is_17_over_25_l64_64081

-- Definition of the probabilities of heads and tails
def prob_tails : ℚ := 1 / 5
def prob_heads : ℚ := 4 * prob_tails

-- Definition of the probability of getting an even number of heads in two flips
def even_heads_prob (p_heads p_tails : ℚ) : ℚ :=
  p_tails * p_tails + p_heads * p_heads

-- Theorem statement
theorem even_heads_probability_is_17_over_25 :
  even_heads_prob prob_heads prob_tails = 17 / 25 := by
  sorry

end even_heads_probability_is_17_over_25_l64_64081


namespace sphere_center_x_axis_eq_l64_64383

theorem sphere_center_x_axis_eq (a : ℝ) (R : ℝ) (x y z : ℝ) :
  (x - a) ^ 2 + y ^ 2 + z ^ 2 = R ^ 2 → (0 - a) ^ 2 + (0 - 0) ^ 2 + (0 - 0) ^ 2 = R ^ 2 →
  a = R →
  (x ^ 2 - 2 * a * x + y ^ 2 + z ^ 2 = 0) :=
by
  sorry

end sphere_center_x_axis_eq_l64_64383


namespace number_of_circles_is_3_l64_64947

-- Define the radius and diameter of the circles
def radius := 4
def diameter := 2 * radius

-- Given the total horizontal length
def total_horizontal_length := 24

-- Number of circles calculated as per the given conditions
def number_of_circles := total_horizontal_length / diameter

-- The proof statement to verify
theorem number_of_circles_is_3 : number_of_circles = 3 := by
  sorry

end number_of_circles_is_3_l64_64947


namespace Ben_win_probability_l64_64665

theorem Ben_win_probability (lose_prob : ℚ) (no_tie : ¬ ∃ (p : ℚ), p ≠ lose_prob ∧ p + lose_prob = 1) 
  (h : lose_prob = 5/8) : (1 - lose_prob) = 3/8 := by
  sorry

end Ben_win_probability_l64_64665


namespace proportion_exists_x_l64_64607

theorem proportion_exists_x : ∃ x : ℕ, 1 * x = 3 * 4 :=
by
  sorry

end proportion_exists_x_l64_64607


namespace lines_intersection_l64_64858

theorem lines_intersection :
  ∃ (t u : ℚ), 
    (∃ (x y : ℚ),
    (x = 2 - t ∧ y = 3 + 4 * t) ∧ 
    (x = -1 + 3 * u ∧ y = 6 + 5 * u) ∧ 
    (x = 28 / 17 ∧ y = 75 / 17)) := sorry

end lines_intersection_l64_64858


namespace minimum_value_of_y_at_l64_64819

def y (x : ℝ) : ℝ := |x + 1| + |x + 2| + |x + 3|

theorem minimum_value_of_y_at (x : ℝ) :
  (∀ x : ℝ, y x ≥ 2) ∧ (y (-2) = 2) :=
by 
  sorry

end minimum_value_of_y_at_l64_64819


namespace sum_geometric_series_l64_64891

noncomputable def S (r : ℝ) : ℝ :=
  12 / (1 - r)

theorem sum_geometric_series (a : ℝ) (h1 : -1 < a) (h2 : a < 1) (h3 : S a * S (-a) = 2016) :
  S a + S (-a) = 336 :=
by
  sorry

end sum_geometric_series_l64_64891


namespace count_even_three_digit_numbers_l64_64939

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l64_64939


namespace probability_winning_l64_64660

-- Define the probability of losing
def P_lose : ℚ := 5 / 8

-- Define the total probability constraint
theorem probability_winning : P_lose = 5 / 8 → (1 - P_lose) = 3 / 8 := 
by
  intro h
  rw h
  norm_num
  sorry

end probability_winning_l64_64660


namespace max_product_913_l64_64203

-- Define the condition that ensures the digits are from the set {3, 5, 8, 9, 1}
def valid_digits (digits : List ℕ) : Prop :=
  digits = [3, 5, 8, 9, 1]

-- Define the predicate for a valid three-digit and two-digit integer
def valid_numbers (a b c d e : ℕ) : Prop :=
  valid_digits [a, b, c, d, e] ∧
  ∃ x y, 100 * x + 10 * 1 + y = 10 * d + e ∧ d ≠ 1 ∧ a ≠ 1

-- Define the product function
def product (a b c d e : ℕ) : ℕ :=
  (100 * a + 10 * b + c) * (10 * d + e)

-- State the theorem
theorem max_product_913 : ∀ (a b c d e : ℕ), valid_numbers a b c d e → 
(product a b c d e) ≤ (product 9 1 3 8 5) :=
by
  intros a b c d e
  unfold valid_numbers product 
  sorry

end max_product_913_l64_64203


namespace liz_spent_total_l64_64793

-- Definitions:
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def number_of_ingredients : ℕ := 5
def apron_cost : ℕ := recipe_book_cost + 1

-- Total cost calculation:
def total_cost : ℕ :=
  recipe_book_cost + baking_dish_cost + (number_of_ingredients * ingredient_cost) + apron_cost

-- Theorem Statement:
theorem liz_spent_total : total_cost = 40 := by
  sorry

end liz_spent_total_l64_64793


namespace smallest_number_of_digits_to_append_l64_64238

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l64_64238


namespace necessary_but_not_sufficient_l64_64980

-- Define the sets A, B, and C
def A : Set ℝ := { x | x - 1 > 0 }
def B : Set ℝ := { x | x < 0 }
def C : Set ℝ := { x | x * (x - 2) > 0 }

-- The set A ∪ B in terms of Lean
def A_union_B : Set ℝ := A ∪ B

-- State the necessary and sufficient conditions
theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x ∈ A_union_B → x ∈ C) ∧ ¬ (∀ x : ℝ, x ∈ C → x ∈ A_union_B) :=
sorry

end necessary_but_not_sufficient_l64_64980


namespace value_of_expression_l64_64247

-- Define the conditions
def x := -2
def y := 1
def z := 1
def w := 3

-- The main theorem statement
theorem value_of_expression : 
  (x^2 * y^2 * z^2) - (x^2 * y * z^2) + (y / w) * Real.sin (x * z) = - (1 / 3) * Real.sin 2 := by
  sorry

end value_of_expression_l64_64247


namespace exists_x_l64_64752

theorem exists_x (a b c : ℕ) (ha : 0 < a) (hc : 0 < c) :
  ∃ x : ℕ, (0 < x) ∧ (a ^ x + x) % c = b % c :=
sorry

end exists_x_l64_64752


namespace number_of_rows_containing_53_l64_64471

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l64_64471


namespace length_of_second_train_l64_64550

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (cross_time : ℝ)
  (opposite_directions : Bool) :
  speed_first_train = 120 / 3.6 →
  speed_second_train = 80 / 3.6 →
  cross_time = 9 →
  length_first_train = 260 →
  opposite_directions = true →
  ∃ (length_second_train : ℝ), length_second_train = 240 :=
by
  sorry

end length_of_second_train_l64_64550


namespace polynomial_problem_l64_64335

theorem polynomial_problem 
  (d_1 d_2 d_3 d_4 e_1 e_2 e_3 e_4 : ℝ)
  (h : ∀ (x : ℝ),
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
    (x^2 + d_1 * x + e_1) * (x^2 + d_2 * x + e_2) * (x^2 + d_3 * x + e_3) * (x^2 + d_4 * x + e_4)) :
  d_1 * e_1 + d_2 * e_2 + d_3 * e_3 + d_4 * e_4 = -1 := 
by
  sorry

end polynomial_problem_l64_64335


namespace hyperbola_eccentricity_l64_64945

theorem hyperbola_eccentricity (a : ℝ) (e : ℝ) :
  (∀ x y : ℝ, y = (1 / 8) * x^2 → x^2 = 8 * y) →
  (∀ y x : ℝ, y^2 / a - x^2 = 1 → a + 1 = 4) →
  e^2 = 4 / 3 →
  e = 2 * Real.sqrt 3 / 3 :=
by
  intros h1 h2 h3
  sorry

end hyperbola_eccentricity_l64_64945


namespace find_g3_l64_64818

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = 1

theorem find_g3 : g 3 = 0 := by
  sorry

end find_g3_l64_64818


namespace dumpling_probability_l64_64372

theorem dumpling_probability :
  let total_dumplings := 15
  let choose4 := Nat.choose total_dumplings 4
  let choose1 := Nat.choose 3 1
  let choose5_2 := Nat.choose 5 2
  let choose5_1 := Nat.choose 5 1
  (choose1 * choose5_2 * choose5_1 * choose5_1) / choose4 = 50 / 91 := by
  sorry

end dumpling_probability_l64_64372


namespace base7_product_digit_sum_l64_64527

noncomputable def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 350 => 3 * 7 + 5
  | 217 => 2 * 7 + 1
  | _ => 0

noncomputable def base10_to_base7 (n : Nat) : Nat := 
  if n = 390 then 1065 else 0

noncomputable def digit_sum_in_base7 (n : Nat) : Nat :=
  if n = 1065 then 1 + 0 + 6 + 5 else 0

noncomputable def sum_to_base7 (n : Nat) : Nat :=
  if n = 12 then 15 else 0

theorem base7_product_digit_sum :
  digit_sum_in_base7 (base10_to_base7 (base7_to_base10 350 * base7_to_base10 217)) = 15 :=
by
  sorry

end base7_product_digit_sum_l64_64527


namespace inequality_proof_l64_64510

variable (a b c : ℝ)

theorem inequality_proof :
  1 < (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 :=
sorry

end inequality_proof_l64_64510


namespace fraction_sum_equals_l64_64108

theorem fraction_sum_equals :
  (1 / 20 : ℝ) + (2 / 10 : ℝ) + (4 / 40 : ℝ) = 0.35 :=
by
  sorry

end fraction_sum_equals_l64_64108


namespace more_student_tickets_l64_64023

-- Definitions of given conditions
def student_ticket_price : ℕ := 6
def nonstudent_ticket_price : ℕ := 9
def total_sales : ℕ := 10500
def total_tickets : ℕ := 1700

-- Definitions of the variables for student and nonstudent tickets
variables (S N : ℕ)

-- Lean statement of the problem
theorem more_student_tickets (h1 : student_ticket_price * S + nonstudent_ticket_price * N = total_sales)
                            (h2 : S + N = total_tickets) : S - N = 1500 :=
by
  sorry

end more_student_tickets_l64_64023


namespace given_conditions_imply_f_neg3_gt_f_neg2_l64_64762

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem given_conditions_imply_f_neg3_gt_f_neg2
  {f : ℝ → ℝ}
  (h_even : is_even_function f)
  (h_comparison : f 2 < f 3) :
  f (-3) > f (-2) :=
by
  sorry

end given_conditions_imply_f_neg3_gt_f_neg2_l64_64762


namespace amount_spent_on_marbles_l64_64799

/-- A theorem to determine the amount Mike spent on marbles. -/
theorem amount_spent_on_marbles 
  (total_amount : ℝ) 
  (cost_football : ℝ) 
  (cost_baseball : ℝ) 
  (total_amount_eq : total_amount = 20.52)
  (cost_football_eq : cost_football = 4.95)
  (cost_baseball_eq : cost_baseball = 6.52) :
  ∃ (cost_marbles : ℝ), cost_marbles = total_amount - (cost_football + cost_baseball) 
  ∧ cost_marbles = 9.05 := 
by
  sorry

end amount_spent_on_marbles_l64_64799


namespace pascal_triangle_contains_53_l64_64463

theorem pascal_triangle_contains_53 (n : ℕ) :
  (∃ k, binomial n k = 53) ↔ n = 53 := 
sorry

end pascal_triangle_contains_53_l64_64463


namespace cirrus_clouds_count_l64_64667

theorem cirrus_clouds_count 
  (cirrus cumulus cumulonimbus : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = 12 * cumulonimbus)
  (h3 : cumulonimbus = 3) : 
  cirrus = 144 := 
by
  sorry

end cirrus_clouds_count_l64_64667


namespace inradius_of_right_triangle_l64_64008

theorem inradius_of_right_triangle (a b c r : ℝ) (h : a^2 + b^2 = c^2) :
  r = (1/2) * (a + b - c) :=
sorry

end inradius_of_right_triangle_l64_64008


namespace number_of_valid_three_digit_even_numbers_l64_64919

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l64_64919


namespace combined_selling_price_l64_64562

theorem combined_selling_price 
  (cost_price1 cost_price2 cost_price3 : ℚ)
  (profit_percentage1 profit_percentage2 profit_percentage3 : ℚ)
  (h1 : cost_price1 = 1200) (h2 : profit_percentage1 = 0.4)
  (h3 : cost_price2 = 800)  (h4 : profit_percentage2 = 0.3)
  (h5 : cost_price3 = 600)  (h6 : profit_percentage3 = 0.5) : 
  cost_price1 * (1 + profit_percentage1) +
  cost_price2 * (1 + profit_percentage2) +
  cost_price3 * (1 + profit_percentage3) = 3620 := by 
  sorry

end combined_selling_price_l64_64562


namespace apple_and_pear_costs_l64_64509

theorem apple_and_pear_costs (x y : ℝ) (h1 : x + 2 * y = 194) (h2 : 2 * x + 5 * y = 458) : 
  y = 70 ∧ x = 54 := 
by 
  sorry

end apple_and_pear_costs_l64_64509


namespace average_score_correct_l64_64174

-- Define the conditions
def simplified_scores : List Int := [10, -5, 0, 8, -3]
def base_score : Int := 90

-- Translate simplified score to actual score
def actual_score (s : Int) : Int :=
  base_score + s

-- Calculate the average of the actual scores
def average_score : Int :=
  (simplified_scores.map actual_score).sum / simplified_scores.length

-- The proof statement
theorem average_score_correct : average_score = 92 := 
by 
  -- Steps to compute the average score
  -- sorry is used since the proof steps are not required
  sorry

end average_score_correct_l64_64174


namespace smallest_digits_to_append_l64_64227

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l64_64227


namespace arithmetic_sequence_common_difference_l64_64959

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : a 1 + a 7 = 22) 
  (h2 : a 4 + a 10 = 40) 
  (h_general_term : ∀ n : ℕ, a n = a 1 + (n - 1) * d) 
  : d = 3 :=
by 
  sorry

end arithmetic_sequence_common_difference_l64_64959


namespace point_outside_circle_l64_64943

theorem point_outside_circle (a : ℝ) :
  (a > 1) → (a, a) ∉ {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 2 * a * p.1 + a^2 - a = 0} :=
by sorry

end point_outside_circle_l64_64943


namespace coefficient_of_x8y2_l64_64494

theorem coefficient_of_x8y2 :
  let term1 := (1 / x^2)
  let term2 := (3 / y)
  let expansion := (x^2 - y)^7
  let coeff1 := 21 * (x ^ 10) * (y ^ 2) * (-1)
  let coeff2 := 35 * (3 / y) * (x ^ 8) * (y ^ 3)
  let comb := coeff1 + coeff2
  comb = -84 * x ^ 8 * y ^ 2 := by
  sorry

end coefficient_of_x8y2_l64_64494


namespace count_even_three_digit_sum_tens_units_is_12_l64_64926

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l64_64926


namespace three_digit_even_sum_12_l64_64916

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l64_64916


namespace sequence_no_consecutive_ones_probability_l64_64269

/-- A sequence of length 8 that does not contain two consecutive 1s. The probability
of such a sequence can be written as m/n, where m and n are relatively prime. We prove
that m + n = 311. -/
theorem sequence_no_consecutive_ones_probability :
  ∃ m n : ℕ,
    (Nat.gcd m n = 1) ∧
    (m + n = 311) ∧
    (m : ℚ) / n = (Fib 10) / (2 ^ 8) :=
sorry

end sequence_no_consecutive_ones_probability_l64_64269


namespace find_value_of_15b_minus_2a_l64_64753

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if 1 ≤ x ∧ x < 2 then x + a / x
else if 2 ≤ x ∧ x ≤ 3 then b * x - 3
else 0

theorem find_value_of_15b_minus_2a (a b : ℝ)
  (h_periodic : ∀ x : ℝ, f x a b = f (x + 2) a b)
  (h_condition : f (7 / 2) a b = f (-7 / 2) a b) :
  15 * b - 2 * a = 41 :=
sorry

end find_value_of_15b_minus_2a_l64_64753


namespace ratio_of_men_to_women_l64_64849

variable (M W : ℕ)

theorem ratio_of_men_to_women
  (h1 : W = M + 4)
  (h2 : M + W = 20) :
  (M : ℚ) / (W : ℚ) = 2 / 3 :=
sorry

end ratio_of_men_to_women_l64_64849


namespace xy_value_l64_64202

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : xy = 21 :=
sorry

end xy_value_l64_64202


namespace solve_system_eq_l64_64850

theorem solve_system_eq (x1 x2 x3 x4 x5 : ℝ) :
  (x3 + x4 + x5)^5 = 3 * x1 ∧
  (x4 + x5 + x1)^5 = 3 * x2 ∧
  (x5 + x1 + x2)^5 = 3 * x3 ∧
  (x1 + x2 + x3)^5 = 3 * x4 ∧
  (x2 + x3 + x4)^5 = 3 * x5 →
  (x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨
  (x1 = 1/3 ∧ x2 = 1/3 ∧ x3 = 1/3 ∧ x4 = 1/3 ∧ x5 = 1/3) ∨
  (x1 = -1/3 ∧ x2 = -1/3 ∧ x3 = -1/3 ∧ x4 = -1/3 ∧ x5 = -1/3) :=
by
  sorry

end solve_system_eq_l64_64850


namespace net_percentage_change_l64_64184

-- Definitions based on given conditions
variables (P : ℝ) (P_post_decrease : ℝ) (P_post_increase : ℝ)

-- Conditions
def decreased_by_5_percent : Prop := P_post_decrease = P * (1 - 0.05)
def increased_by_10_percent : Prop := P_post_increase = P_post_decrease * (1 + 0.10)

-- Proof problem
theorem net_percentage_change (h1 : decreased_by_5_percent P P_post_decrease) (h2 : increased_by_10_percent P_post_decrease P_post_increase) : 
  ((P_post_increase - P) / P) * 100 = 4.5 :=
by
  -- The proof would go here
  sorry

end net_percentage_change_l64_64184


namespace remainder_of_large_number_l64_64690

theorem remainder_of_large_number : 
  (9876543210 : ℤ) % 101 = 73 := 
by
  unfold_coes
  unfold_norm_num
  sorry

end remainder_of_large_number_l64_64690


namespace det_projection_matrix_l64_64782

-- Definition of the projection matrix P onto vector (3, 4)
def projection_matrix (u : Vector ℝ 2) : Matrix (Fin 2) (Fin 2) ℝ :=
  let (x, y) := (u 0, u 1)
  let norm_sq := x^2 + y^2
  (Matrix.vecCons (Matrix.vecCons (x^2 / norm_sq) (x * y / norm_sq))
                  (Matrix.vecCons (x * y / norm_sq) (y^2 / norm_sq)))

-- Specific vector (3, 4)
def u : Vector ℝ 2 := ![3, 4]
def P := projection_matrix u

-- Goal: The determinant of the projection matrix P
theorem det_projection_matrix : det P = 0 :=
by {
  sorry
}

end det_projection_matrix_l64_64782


namespace smallest_digits_to_append_l64_64229

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l64_64229


namespace happy_children_count_l64_64986

-- Definitions of the conditions
def total_children : ℕ := 60
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 22
def girls : ℕ := 38
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4
def boys_neither_happy_nor_sad : ℕ := 10

-- The theorem we wish to prove
theorem happy_children_count :
  total_children - sad_children - neither_happy_nor_sad_children = 30 :=
by 
  -- Placeholder for the proof
  sorry

end happy_children_count_l64_64986


namespace michael_scored_times_more_goals_l64_64384

theorem michael_scored_times_more_goals (x : ℕ) (hb : Bruce_goals = 4) (hm : Michael_goals = 4 * x) (ht : Bruce_goals + Michael_goals = 16) : x = 3 := by
  sorry

end michael_scored_times_more_goals_l64_64384


namespace number_of_pieces_l64_64984

def pan_length : ℕ := 24
def pan_width : ℕ := 30
def brownie_length : ℕ := 3
def brownie_width : ℕ := 4

def area (length : ℕ) (width : ℕ) : ℕ := length * width

theorem number_of_pieces :
  (area pan_length pan_width) / (area brownie_length brownie_width) = 60 := by
  sorry

end number_of_pieces_l64_64984


namespace number_of_factors_l64_64364

theorem number_of_factors (b n : ℕ) (hb1 : b = 6) (hn1 : n = 15) (hb2 : b > 0) (hb3 : b ≤ 15) (hn2 : n > 0) (hn3 : n ≤ 15) :
  let factors := (15 + 1) * (15 + 1)
  factors = 256 :=
by
  sorry

end number_of_factors_l64_64364


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l64_64931

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l64_64931


namespace graph_shift_cos_function_l64_64130

theorem graph_shift_cos_function (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = 2 * Real.cos (π * x / 3 + φ)) ∧ 
  (∃ x, f x = 0 ∧ x = 2) ∧ 
  (f 1 > f 3) →
  (∀ x, f x = 2 * Real.cos (π * (x - 1/2) / 3)) :=
by
  sorry

end graph_shift_cos_function_l64_64130


namespace gcd_g_values_l64_64340

def g (x : ℤ) : ℤ := x^2 - 2 * x + 2023

theorem gcd_g_values : gcd (g 102) (g 103) = 1 := by
  sorry

end gcd_g_values_l64_64340


namespace smallest_number_of_digits_to_append_l64_64241

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l64_64241


namespace find_value_of_x_l64_64852

theorem find_value_of_x (w : ℕ) (x y z : ℕ) (h₁ : x = y / 3) (h₂ : y = z / 6) (h₃ : z = 2 * w) (hw : w = 45) : x = 5 :=
by
  sorry

end find_value_of_x_l64_64852


namespace find_coordinates_of_b_l64_64304

theorem find_coordinates_of_b
  (x y : ℝ)
  (a : ℂ) (b : ℂ)
  (sqrt3 sqrt5 sqrt10 sqrt6 : ℝ)
  (h1 : sqrt3 = Real.sqrt 3)
  (h2 : sqrt5 = Real.sqrt 5)
  (h3 : sqrt10 = Real.sqrt 10)
  (h4 : sqrt6 = Real.sqrt 6)
  (h5 : a = ⟨sqrt3, sqrt5⟩)
  (h6 : ∃ x y : ℝ, b = ⟨x, y⟩ ∧ (sqrt3 * x + sqrt5 * y = 0) ∧ (Real.sqrt (x^2 + y^2) = 2))
  : b = ⟨- sqrt10 / 2, sqrt6 / 2⟩ ∨ b = ⟨sqrt10 / 2, - sqrt6 / 2⟩ := 
  sorry

end find_coordinates_of_b_l64_64304


namespace determine_a_l64_64725

theorem determine_a (a b c : ℤ) (h_eq : ∀ x : ℤ, (x - a) * (x - 15) + 4 = (x + b) * (x + c)) :
  a = 16 ∨ a = 21 :=
  sorry

end determine_a_l64_64725


namespace course_count_l64_64270

theorem course_count (n1 n2 : ℕ) (sum_x1 sum_x2 : ℕ) :
  (n1 = 6) →
  (sum_x1 = n1 * 100) →
  (sum_x2 = n2 * 50) →
  ((sum_x1 + sum_x2) / (n1 + n2) = 77) →
  n2 = 5 :=
by
  intros h1 h2 h3 h4
  sorry

end course_count_l64_64270


namespace range_of_a_l64_64950

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + a * x - 1 ≤ 0) : -4 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l64_64950


namespace calculate_expression_l64_64418

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end calculate_expression_l64_64418


namespace problem_l64_64297

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 8

theorem problem 
  (a b c : ℝ) 
  (h : f a b c (-2) = 10) 
  : f a b c 2 = 6 :=
by
  sorry

end problem_l64_64297


namespace original_photo_dimensions_l64_64291

theorem original_photo_dimensions (squares_before : ℕ) 
    (squares_after : ℕ) 
    (vertical_length : ℕ) 
    (horizontal_length : ℕ) 
    (side_length : ℕ)
    (h1 : squares_before = 1812)
    (h2 : squares_after = 2018)
    (h3 : side_length = 1) :
    vertical_length = 101 ∧ horizontal_length = 803 :=
by
    sorry

end original_photo_dimensions_l64_64291


namespace rhombus_area_600_l64_64431

noncomputable def area_of_rhombus (x y : ℝ) : ℝ := (x * y) * 2

theorem rhombus_area_600 (x y : ℝ) (qx qy : ℝ)
  (hx : x = 15) (hy : y = 20)
  (hr1 : qx = 15) (hr2 : qy = 20)
  (h_ratio : qy / qx = 4 / 3) :
  area_of_rhombus (2 * (x + y - 2)) (x + y) = 600 :=
by
  rw [hx, hy]
  sorry

end rhombus_area_600_l64_64431


namespace ceil_sub_self_eq_half_l64_64345

theorem ceil_sub_self_eq_half (n : ℤ) (x : ℝ) (h : x = n + 1/2) : ⌈x⌉ - x = 1/2 :=
by
  sorry

end ceil_sub_self_eq_half_l64_64345


namespace complex_number_purely_imaginary_l64_64944

theorem complex_number_purely_imaginary (m : ℝ) 
  (h1 : m^2 - 5 * m + 6 = 0) 
  (h2 : m^2 - 3 * m ≠ 0) : 
  m = 2 :=
sorry

end complex_number_purely_imaginary_l64_64944


namespace smallest_digits_to_append_l64_64237

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l64_64237


namespace min_students_l64_64487

noncomputable def smallest_possible_number_of_students (b g : ℕ) : ℕ :=
if 3 * (3 * b) = 5 * (4 * g) then b + g else 0

theorem min_students (b g : ℕ) (h1 : 0 < b) (h2 : 0 < g) (h3 : 3 * (3 * b) = 5 * (4 * g)) :
  smallest_possible_number_of_students b g = 29 := sorry

end min_students_l64_64487


namespace count_even_three_digit_numbers_l64_64937

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l64_64937


namespace solution_l64_64159

def mapping (x : ℝ) : ℝ := x^2

theorem solution (x : ℝ) : mapping x = 4 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end solution_l64_64159


namespace plot_length_l64_64017

theorem plot_length (b : ℕ) (cost_per_meter total_cost : ℕ)
  (h1 : cost_per_meter = 2650 / 100)  -- Since Lean works with integers, use 2650 instead of 26.50
  (h2 : total_cost = 5300)
  (h3 : 2 * (b + 16) + 2 * b = total_cost / cost_per_meter) :
  b + 16 = 58 :=
by
  -- Above theorem aims to prove the length of the plot is 58 meters, given the conditions.
  sorry

end plot_length_l64_64017


namespace smallest_digits_to_append_l64_64213

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l64_64213


namespace system_solution_l64_64999

theorem system_solution (x y z : ℝ) 
  (h1 : x - y ≥ z)
  (h2 : x^2 + 4 * y^2 + 5 = 4 * z) :
  (x = 2 ∧ y = -0.5 ∧ z = 2.5) :=
sorry

end system_solution_l64_64999


namespace find_y_l64_64314

theorem find_y 
  (x y z : ℕ) 
  (h₁ : x + y + z = 25)
  (h₂ : x + y = 19) 
  (h₃ : y + z = 18) :
  y = 12 :=
by
  sorry

end find_y_l64_64314


namespace convert_degrees_to_radians_l64_64876

theorem convert_degrees_to_radians : 
  (-390) * (Real.pi / 180) = - (13 * Real.pi / 6) := 
by 
  sorry

end convert_degrees_to_radians_l64_64876


namespace intersection_of_A_and_B_l64_64979

noncomputable def A := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
noncomputable def B := {x : ℝ | x < 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} :=
sorry

end intersection_of_A_and_B_l64_64979


namespace original_price_l64_64183

variable (q r : ℝ)

theorem original_price (x : ℝ) (h : x * (1 + q / 100) * (1 - r / 100) = 1) :
  x = 1 / ((1 + q / 100) * (1 - r / 100)) :=
sorry

end original_price_l64_64183


namespace billy_win_probability_l64_64405

-- Definitions of states and transition probabilities
def alice_step_prob_pos : ℚ := 1 / 2
def alice_step_prob_neg : ℚ := 1 / 2
def billy_step_prob_pos : ℚ := 2 / 3
def billy_step_prob_neg : ℚ := 1 / 3

-- Definitions of states in the Markov chain
inductive State
| S0 | S1 | Sm1 | S2 | Sm2 -- Alice's states
| T0 | T1 | Tm1 | T2 | Tm2 -- Billy's states

open State

-- The theorem statement: the probability that Billy wins the game
theorem billy_win_probability : 
  ∃ (P : State → ℚ), 
  P S0 = 11 / 19 ∧ P T0 = 14 / 19 ∧ 
  P S1 = 1 / 2 * P T0 ∧
  P Sm1 = 1 / 2 * P S0 + 1 / 2 ∧
  P T0 = 2 / 3 * P T1 + 1 / 3 * P Tm1 ∧
  P T1 = 2 / 3 + 1 / 3 * P S0 ∧
  P Tm1 = 2 / 3 * P T0 ∧
  P S2 = 0 ∧ P Sm2 = 1 ∧ P T2 = 1 ∧ P Tm2 = 0 := 
by 
  sorry

end billy_win_probability_l64_64405


namespace charlie_max_success_ratio_l64_64146

-- Given:
-- Alpha scored 180 points out of 360 attempted on day one.
-- Alpha scored 120 points out of 240 attempted on day two.
-- Charlie did not attempt 360 points on the first day.
-- Charlie's success ratio on each day was less than Alpha’s.
-- Total points attempted by Charlie on both days are 600.
-- Alpha's two-day success ratio is 300/600 = 1/2.
-- Find the largest possible two-day success ratio that Charlie could have achieved.

theorem charlie_max_success_ratio:
  ∀ (x y z w : ℕ),
  0 < x ∧ 0 < z ∧ 0 < y ∧ 0 < w ∧
  y + w = 600 ∧
  (2 * x < y) ∧ (2 * z < w) ∧
  (x + z < 300) -> (299 / 600 = 299 / 600) :=
by
  sorry

end charlie_max_success_ratio_l64_64146


namespace rate_at_which_bowls_were_bought_l64_64561

theorem rate_at_which_bowls_were_bought 
    (total_bowls : ℕ) (sold_bowls : ℕ) (price_per_sold_bowl : ℝ) (remaining_bowls : ℕ) (percentage_gain : ℝ) 
    (total_bowls_eq : total_bowls = 115) 
    (sold_bowls_eq : sold_bowls = 104) 
    (price_per_sold_bowl_eq : price_per_sold_bowl = 20) 
    (remaining_bowls_eq : remaining_bowls = 11) 
    (percentage_gain_eq : percentage_gain = 0.4830917874396135) 
  : ∃ (R : ℝ), R = 18 :=
  sorry

end rate_at_which_bowls_were_bought_l64_64561


namespace probability_point_between_C_and_D_l64_64992

theorem probability_point_between_C_and_D :
  ∀ (A B C D E : ℝ), A < B ∧ C < D ∧
  (B - A = 4 * (D - A)) ∧ (B - A = 4 * (B - E)) ∧
  (D - A = C - D) ∧ (C - D = E - C) ∧ (E - C = B - E) →
  (B - A ≠ 0) → 
  (C - D) / (B - A) = 1 / 4 :=
by
  intros A B C D E hAB hNonZero
  sorry

end probability_point_between_C_and_D_l64_64992


namespace intersection_S_T_l64_64636

def set_S : Set ℝ := { x | abs x < 5 }
def set_T : Set ℝ := { x | x^2 + 4*x - 21 < 0 }

theorem intersection_S_T :
  set_S ∩ set_T = { x | -5 < x ∧ x < 3 } :=
sorry

end intersection_S_T_l64_64636


namespace work_efficiency_ratio_l64_64556

variable (A B : ℝ)
variable (h1 : A = 1 / 2 * B) 
variable (h2 : 1 / (A + B) = 13)
variable (h3 : B = 1 / 19.5)

theorem work_efficiency_ratio : A / B = 1 / 2 := by
  sorry

end work_efficiency_ratio_l64_64556


namespace perpendicular_vectors_m_value_l64_64306

theorem perpendicular_vectors_m_value : 
  ∀ (m : ℝ), ((2 : ℝ) * (1 : ℝ) + (m * (1 / 2)) + (1 * 2) = 0) → m = -8 :=
by
  intro m
  intro h
  sorry

end perpendicular_vectors_m_value_l64_64306


namespace smallest_four_digit_solution_l64_64112

theorem smallest_four_digit_solution :
  ∃ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧
  (3 * x ≡ 6 [MOD 12]) ∧
  (5 * x + 20 ≡ 25 [MOD 15]) ∧
  (3 * x - 2 ≡ 2 * x [MOD 35]) ∧
  x = 1274 :=
by
  sorry

end smallest_four_digit_solution_l64_64112


namespace geometric_series_common_ratio_l64_64087

theorem geometric_series_common_ratio
    (a : ℝ) (S : ℝ) (r : ℝ)
    (h_a : a = 512)
    (h_S : S = 3072)
    (h_sum : S = a / (1 - r)) : 
    r = 5 / 6 :=
by 
  rw [h_a] at h_sum
  rw [h_S] at h_sum
  sorry

end geometric_series_common_ratio_l64_64087


namespace number_divisible_l64_64230

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l64_64230


namespace John_study_time_second_exam_l64_64499

variable (StudyTime Score : ℝ)
variable (k : ℝ) (h1 : k = Score / StudyTime)
variable (study_first : ℝ := 3) (score_first : ℝ := 60)
variable (avg_target : ℝ := 75)
variable (total_tests : ℕ := 2)

theorem John_study_time_second_exam :
  (avg_target * total_tests - score_first) / (score_first / study_first) = 4.5 :=
by
  sorry

end John_study_time_second_exam_l64_64499


namespace express_scientific_notation_l64_64730

theorem express_scientific_notation : (152300 : ℝ) = 1.523 * 10^5 := 
by
  sorry

end express_scientific_notation_l64_64730


namespace coeff_x2_y2_in_expansion_l64_64817

noncomputable def binomial_expansion : ℚ := 
  ∑ r in finset.range 9, (binom 8 r : ℚ) * (x^(8 - r / 2 : ℚ) / y^(8 - r / 2 : ℚ)) * (- (y^(r / 2) / x^(r / 2)))

theorem coeff_x2_y2_in_expansion : 
  let c := @coeff ℚ _ _ _ (binomial_expansion) x^2 y^2 in
  c = 70 := by
  sorry

end coeff_x2_y2_in_expansion_l64_64817


namespace solve_inequality_l64_64815

theorem solve_inequality (x : ℝ) : |x - 2| > 2 - x ↔ x > 2 :=
sorry

end solve_inequality_l64_64815


namespace backyard_area_proof_l64_64349

-- Condition: Walking the length of 40 times covers 1000 meters
def length_times_40_eq_1000 (L: ℝ) : Prop := 40 * L = 1000

-- Condition: Walking the perimeter 8 times covers 1000 meters
def perimeter_times_8_eq_1000 (P: ℝ) : Prop := 8 * P = 1000

-- Given the conditions, we need to find the Length and Width of the backyard
def is_backyard_dimensions (L W: ℝ) : Prop := 
  length_times_40_eq_1000 L ∧ 
  perimeter_times_8_eq_1000 (2 * (L + W))

-- We need to calculate the area
def backyard_area (L W: ℝ) : ℝ := L * W

-- The theorem to prove
theorem backyard_area_proof (L W: ℝ) 
  (h1: length_times_40_eq_1000 L) 
  (h2: perimeter_times_8_eq_1000 (2 * (L + W))) :
  backyard_area L W = 937.5 := 
  by 
    sorry

end backyard_area_proof_l64_64349


namespace cirrus_clouds_count_l64_64666

theorem cirrus_clouds_count 
  (cirrus cumulus cumulonimbus : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = 12 * cumulonimbus)
  (h3 : cumulonimbus = 3) : 
  cirrus = 144 := 
by
  sorry

end cirrus_clouds_count_l64_64666


namespace paper_folding_possible_layers_l64_64355

theorem paper_folding_possible_layers (n : ℕ) : 16 = 2 ^ n :=
by
  sorry

end paper_folding_possible_layers_l64_64355


namespace find_missing_score_l64_64741

noncomputable def total_points (mean : ℝ) (games : ℕ) : ℝ :=
  mean * games

noncomputable def sum_of_scores (scores : List ℝ) : ℝ :=
  scores.sum

theorem find_missing_score
  (scores : List ℝ)
  (mean : ℝ)
  (games : ℕ)
  (total_points_value : ℝ)
  (sum_of_recorded_scores : ℝ)
  (missing_score : ℝ) :
  scores = [81, 73, 86, 73] →
  mean = 79.2 →
  games = 5 →
  total_points_value = total_points mean games →
  sum_of_recorded_scores = sum_of_scores scores →
  missing_score = total_points_value - sum_of_recorded_scores →
  missing_score = 83 :=
by
  intros
  exact sorry

end find_missing_score_l64_64741


namespace smallest_append_digits_l64_64214

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l64_64214


namespace parallel_lines_slope_m_l64_64318

theorem parallel_lines_slope_m (m : ℝ) : (∀ (x y : ℝ), (x - 2 * y + 5 = 0) ↔ (2 * x + m * y - 5 = 0)) → m = -4 :=
by
  intros h
  -- Add the necessary calculative steps here
  sorry

end parallel_lines_slope_m_l64_64318


namespace sum_first_9_terms_l64_64300

variable {a : ℕ → ℝ}  -- Define the geometric sequence a_n
variable {b : ℕ → ℝ}  -- Define the arithmetic sequence b_n

-- Conditions given in the problem
axiom geo_seq (a : ℕ → ℝ) : ∀ n m, a (n + m) = a n * a m
axiom condition1 : a 2 * a 8 = 4 * a 5
axiom arith_seq (b : ℕ → ℝ) : ∀ n m, m > n → b m = b n + (m - n) * (b 1 - b 0)
axiom condition2 : b 4 + b 6 = a 5

-- Statement to prove
theorem sum_first_9_terms : (Finset.range 9).sum b = 18 :=
by {
  sorry
}

end sum_first_9_terms_l64_64300


namespace cars_more_than_trucks_l64_64729

theorem cars_more_than_trucks (total_vehicles : ℕ) (trucks : ℕ) (h : total_vehicles = 69) (h' : trucks = 21) :
  (total_vehicles - trucks) - trucks = 27 :=
by
  sorry

end cars_more_than_trucks_l64_64729


namespace inequality_abc_l64_64639

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) >= 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_abc_l64_64639


namespace number_of_initial_cards_l64_64353

theorem number_of_initial_cards (x : ℝ) (h1 : x + 276.0 = 580) : x = 304 :=
by
  sorry

end number_of_initial_cards_l64_64353


namespace minimize_expression_at_c_l64_64437

theorem minimize_expression_at_c (c : ℝ) : (c = 7 / 4) → (∀ x : ℝ, 2 * c^2 - 7 * c + 4 ≤ 2 * x^2 - 7 * x + 4) :=
sorry

end minimize_expression_at_c_l64_64437


namespace distinct_real_roots_m_range_root_zero_other_root_l64_64308

open Real

-- Definitions of the quadratic equation and the conditions
def quadratic_eq (m x : ℝ) := x^2 + 2 * (m - 1) * x + m^2 - 1

-- Problem (1)
theorem distinct_real_roots_m_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0) → m < 1 :=
by
  sorry

-- Problem (2)
theorem root_zero_other_root (m x : ℝ) :
  (quadratic_eq m 0 = 0 ∧ quadratic_eq m x = 0) → (m = 1 ∧ x = 0) ∨ (m = -1 ∧ x = 4) :=
by
  sorry

end distinct_real_roots_m_range_root_zero_other_root_l64_64308


namespace find_a_l64_64751

theorem find_a (a : ℝ) : (dist (⟨-2, -1⟩ : ℝ × ℝ) (⟨a, 3⟩ : ℝ × ℝ) = 5) ↔ (a = 1 ∨ a = -5) :=
by
  sorry

end find_a_l64_64751


namespace sum_roots_eq_six_l64_64058

theorem sum_roots_eq_six : 
  ∀ x : ℝ, (x - 3) ^ 2 = 16 → (x - 3 = 4 ∨ x - 3 = -4) → (let x₁ := 3 + 4 in let x₂ := 3 - 4 in x₁ + x₂ = 6) := by
  sorry

end sum_roots_eq_six_l64_64058


namespace intersecting_diagonals_probability_l64_64768

variable (n : ℕ) (h : n > 0)

theorem intersecting_diagonals_probability (h : n > 0) :
  let V := 2 * n + 1 in
  let total_diagonals := (V * (V - 3)) / 2 in
  let pairs_of_diagonals := (total_diagonals * (total_diagonals - 1)) / 2 in
  let intersecting_pairs := ((2 * n + 1) * n * (2 * n - 1) * (n - 1)) / 24 in
  (intersecting_pairs.toRat / pairs_of_diagonals.toRat) = (n * (2 * n - 1)).toRat / (3 * (2 * n^2 - n - 2).toRat) := 
sorry

end intersecting_diagonals_probability_l64_64768


namespace like_terms_proof_l64_64757

theorem like_terms_proof (m n : ℤ) 
  (h1 : m + 10 = 3 * n - m) 
  (h2 : 7 - n = n - m) :
  m^2 - 2 * m * n + n^2 = 9 := by
  sorry

end like_terms_proof_l64_64757


namespace polynomial_has_exactly_one_real_root_l64_64511

theorem polynomial_has_exactly_one_real_root :
  ∀ (x : ℝ), (2007 * x^3 + 2006 * x^2 + 2005 * x = 0) → x = 0 :=
by
  sorry

end polynomial_has_exactly_one_real_root_l64_64511


namespace ages_of_boys_l64_64185

theorem ages_of_boys (a b c : ℕ) (h1 : a + b + c = 29) (h2 : a = b) (h3 : c = 11) : a = 9 :=
by
  sorry

end ages_of_boys_l64_64185


namespace marbles_per_friend_l64_64981

theorem marbles_per_friend (total_marbles : ℕ) (num_friends : ℕ) (h_total : total_marbles = 30) (h_friends : num_friends = 5) :
  total_marbles / num_friends = 6 :=
by
  -- Proof skipped
  sorry

end marbles_per_friend_l64_64981


namespace next_two_equations_l64_64638

-- Definitions based on the conditions in the problem
def pattern1 (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Statement to prove the continuation of the pattern
theorem next_two_equations 
: pattern1 9 40 41 ∧ pattern1 11 60 61 :=
by
  sorry

end next_two_equations_l64_64638


namespace pascals_triangle_53_rows_l64_64474

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l64_64474


namespace calculate_drift_l64_64541

theorem calculate_drift (w v t : ℝ) (hw : w = 400) (hv : v = 10) (ht : t = 50) : v * t - w = 100 :=
by
  sorry

end calculate_drift_l64_64541


namespace b_share_of_earnings_l64_64842

-- Definitions derived from conditions
def work_rate_a := 1 / 6
def work_rate_b := 1 / 8
def work_rate_c := 1 / 12
def total_earnings := 1170

-- Mathematically equivalent Lean statement
theorem b_share_of_earnings : 
  (work_rate_b / (work_rate_a + work_rate_b + work_rate_c)) * total_earnings = 390 := 
by
  sorry

end b_share_of_earnings_l64_64842


namespace sum_of_squares_of_four_consecutive_even_numbers_eq_344_l64_64389

theorem sum_of_squares_of_four_consecutive_even_numbers_eq_344 (n : ℤ) 
  (h : n + (n + 2) + (n + 4) + (n + 6) = 36) : 
  n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344 :=
by sorry

end sum_of_squares_of_four_consecutive_even_numbers_eq_344_l64_64389


namespace volume_of_sphere_l64_64131

theorem volume_of_sphere (r : ℝ) (h : r = 3) : (4 / 3) * π * r ^ 3 = 36 * π := 
by
  sorry

end volume_of_sphere_l64_64131


namespace calculate_expression_l64_64413

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end calculate_expression_l64_64413


namespace locus_of_point_C_l64_64491

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_isosceles_triangle (A B C : Point) : Prop := 
  let AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let AC := (A.x - C.x)^2 + (A.y - C.y)^2
  AB = AC

def circle_eqn (C : Point) : Prop :=
  C.x^2 + C.y^2 - 3 * C.x + C.y = 2

def not_points (C : Point) : Prop :=
  (C ≠ {x := 3, y := -2}) ∧ (C ≠ {x := 0, y := 1})

theorem locus_of_point_C :
  ∀ (A B C : Point),
    A = {x := 3, y := -2} →
    B = {x := 0, y := 1} →
    is_isosceles_triangle A B C →
    circle_eqn C ∧ not_points C :=
by
  intros A B C hA hB hIso
  sorry

end locus_of_point_C_l64_64491


namespace evaluate_expression_l64_64907

theorem evaluate_expression (a b c d m : ℤ) (h1 : a = -b) (h2 : c * d = 1) (h3 : |m| = 2) :
  3 * (a + b - 1) + (-c * d)^2023 - 2 * m = -8 ∨ 3 * (a + b - 1) + (-c * d)^2023 - 2 * m = 0 :=
by {
  sorry
}

end evaluate_expression_l64_64907


namespace pascal_triangle_contains_53_once_l64_64466

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l64_64466


namespace figure_can_be_cut_and_reassembled_into_square_l64_64098

-- Define the conditions
def is_square_area (n: ℕ) : Prop := ∃ k: ℕ, k * k = n

def can_form_square (area: ℕ) : Prop :=
area = 18 ∧ ¬ is_square_area area

-- The proof statement
theorem figure_can_be_cut_and_reassembled_into_square (area: ℕ) (hf: area = 18): 
  can_form_square area → ∃ (part1 part2 part3: Set (ℕ × ℕ)), true :=
by
  sorry

end figure_can_be_cut_and_reassembled_into_square_l64_64098


namespace combined_ticket_cost_l64_64168

variables (S K : ℕ)

theorem combined_ticket_cost (total_budget : ℕ) (samuel_food_drink : ℕ) (kevin_food : ℕ) (kevin_drink : ℕ) :
  total_budget = 20 →
  samuel_food_drink = 6 →
  kevin_food = 4 →
  kevin_drink = 2 →
  S + samuel_food_drink + K + kevin_food + kevin_drink = total_budget →
  S + K = 8 :=
by
  intros h_total_budget h_samuel_food_drink h_kevin_food h_kevin_drink h_total_spent
  /-
  We have the following conditions:
  1. total_budget = 20
  2. samuel_food_drink = 6
  3. kevin_food = 4
  4. kevin_drink = 2
  5. S + samuel_food_drink + K + kevin_food + kevin_drink = total_budget

  We need to prove that S + K = 8. We can use the conditions to derive this.
  -/
  rw [h_total_budget, h_samuel_food_drink, h_kevin_food, h_kevin_drink] at h_total_spent
  exact sorry

end combined_ticket_cost_l64_64168


namespace right_triangle_hypotenuse_product_square_l64_64811

theorem right_triangle_hypotenuse_product_square (A₁ A₂ : ℝ) (a₁ b₁ a₂ b₂ : ℝ) 
(h₁ : a₁ * b₁ / 2 = A₁) (h₂ : a₂ * b₂ / 2 = A₂) 
(h₃ : A₁ = 2) (h₄ : A₂ = 3) 
(h₅ : a₁ = a₂) (h₆ : b₂ = 2 * b₁) : 
(a₁ ^ 2 + b₁ ^ 2) * (a₂ ^ 2 + b₂ ^ 2) = 325 := 
by sorry

end right_triangle_hypotenuse_product_square_l64_64811


namespace range_of_p_l64_64449

def sequence_sum (n : ℕ) : ℚ := (-1) ^ (n + 1) * (1 / 2 ^ n)

def a_n (n : ℕ) : ℚ :=
  if h : n = 0 then sequence_sum 1 else
  sequence_sum n - sequence_sum (n - 1)

theorem range_of_p (p : ℚ) : 
  (∃ n : ℕ, 0 < n ∧ (p - a_n n) * (p - a_n (n + 1)) < 0) ↔ 
  - 3 / 4 < p ∧ p < 1 / 2 :=
sorry

end range_of_p_l64_64449


namespace arithmetic_sequence_a20_l64_64624

theorem arithmetic_sequence_a20 :
  (∀ n : ℕ, n > 0 → ∃ a : ℕ → ℕ, a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n + 2)) → 
  (∃ a : ℕ → ℕ, a 20 = 39) :=
by
  sorry

end arithmetic_sequence_a20_l64_64624


namespace sum_of_roots_l64_64387

theorem sum_of_roots (b : ℝ) (x : ℝ) (y : ℝ) :
  (x^2 - b * x + 20 = 0) ∧ (y^2 - b * y + 20 = 0) ∧ (x * y = 20) -> (x + y = b) := 
by
  sorry

end sum_of_roots_l64_64387


namespace probability_two_diagonals_intersect_l64_64772

theorem probability_two_diagonals_intersect (n : ℕ) (h : 0 < n) : 
  let vertices := 2 * n + 1 in
  let total_diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_of_diagonals := total_diagonals.choose 2 in
  let crossing_diagonals := (vertices.choose 4) in
  ((crossing_diagonals * 2) / pairs_of_diagonals : ℚ) = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  sorry

end probability_two_diagonals_intersect_l64_64772


namespace greatest_visible_unit_cubes_from_one_point_12_l64_64392

def num_unit_cubes (n : ℕ) : ℕ := n * n * n

def face_count (n : ℕ) : ℕ := n * n

def edge_count (n : ℕ) : ℕ := n

def visible_unit_cubes_from_one_point (n : ℕ) : ℕ :=
  let faces := 3 * face_count n
  let edges := 3 * (edge_count n - 1)
  let corner := 1
  faces - edges + corner

theorem greatest_visible_unit_cubes_from_one_point_12 :
  visible_unit_cubes_from_one_point 12 = 400 :=
  by
  sorry

end greatest_visible_unit_cubes_from_one_point_12_l64_64392


namespace linear_correlation_test_l64_64356

theorem linear_correlation_test (n1 n2 n3 n4 : ℕ) (r1 r2 r3 r4 : ℝ) :
  n1 = 10 ∧ r1 = 0.9533 →
  n2 = 15 ∧ r2 = 0.3012 →
  n3 = 17 ∧ r3 = 0.9991 →
  n4 = 3  ∧ r4 = 0.9950 →
  abs r1 > abs r2 ∧ abs r3 > abs r4 →
  (abs r1 > abs r2 → abs r1 > abs r4) →
  (abs r3 > abs r2 → abs r3 > abs r4) →
  abs r1 ≠ abs r2 →
  abs r3 ≠ abs r4 →
  true := 
sorry

end linear_correlation_test_l64_64356


namespace remainder_9876543210_mod_101_l64_64683

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l64_64683


namespace rectangle_perimeter_l64_64888

theorem rectangle_perimeter (a b : ℕ) : 
  (2 * a + b = 6 ∨ a + 2 * b = 6 ∨ 2 * a + b = 9 ∨ a + 2 * b = 9) → 
  2 * a + 2 * b = 10 :=
by 
  sorry

end rectangle_perimeter_l64_64888


namespace solve_fraction_eq_l64_64573

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 3) 
    (h₃ : 3 / (x - 2) = 6 / (x - 3)) : x = 1 :=
by 
  sorry

end solve_fraction_eq_l64_64573


namespace value_of_n_l64_64253

def is_3_digit_integer (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

def not_divisible_by (n k : ℕ) : Prop := ¬ (k ∣ n)

def least_common_multiple (a b c : ℕ) : Prop := Nat.lcm a b = c

theorem value_of_n (d n : ℕ) (h1 : least_common_multiple d n 690) 
  (h2 : not_divisible_by n 3) (h3 : not_divisible_by d 2) (h4 : is_3_digit_integer n) : n = 230 :=
by
  sorry

end value_of_n_l64_64253


namespace y_intercept_of_line_l64_64041

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end y_intercept_of_line_l64_64041


namespace cade_marbles_left_l64_64722

theorem cade_marbles_left (initial_marbles : ℕ) (given_away : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 350 → given_away = 175 → remaining_marbles = initial_marbles - given_away → remaining_marbles = 175 :=
by
  intros h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining

end cade_marbles_left_l64_64722


namespace value_of_m_div_x_l64_64254

variable (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ratio : a / b = 4 / 5)

def x := a + 0.25 * a
def m := b - 0.40 * b

theorem value_of_m_div_x (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ratio : a / b = 4 / 5) :
    m / x = 3 / 5 :=
by
  sorry

end value_of_m_div_x_l64_64254


namespace divide_composite_products_l64_64105

def first_eight_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List ℕ := [16, 18, 20, 21, 22, 24, 25, 26]

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem divide_composite_products :
  product first_eight_composites * 3120 = product next_eight_composites :=
by
  -- This would be the place for the proof solution
  sorry

end divide_composite_products_l64_64105


namespace pipe_a_filling_time_l64_64531

theorem pipe_a_filling_time
  (pipeA_fill_time : ℝ)
  (pipeB_fill_time : ℝ)
  (both_pipes_open : Bool)
  (pipeB_shutoff_time : ℝ)
  (overflow_time : ℝ)
  (pipeB_rate : ℝ)
  (combined_rate : ℝ)
  (a_filling_time : ℝ) :
  pipeA_fill_time = 1 / 2 :=
by
  -- Definitions directly from conditions in a)
  let pipeA_fill_time := a_filling_time
  let pipeB_fill_time := 1  -- Pipe B fills in 1 hour
  let both_pipes_open := True
  let pipeB_shutoff_time := 0.5 -- Pipe B shuts 30 minutes before overflow
  let overflow_time := 0.5  -- Tank overflows in 30 minutes
  let pipeB_rate := 1 / pipeB_fill_time
  
  -- Goal to prove
  sorry

end pipe_a_filling_time_l64_64531


namespace rectangle_sides_l64_64399

theorem rectangle_sides :
  ∀ (x : ℝ), 
    (3 * x = 8) ∧ (8 / 3 * 3 = 8) →
    ((2 * (3 * x + x) = 3 * x^2) ∧ (2 * (3 * (8 / 3) + (8 / 3)) = 3 * (8 / 3)^2) →
    x = 8 / 3
      ∧ 3 * x = 8) := 
by
  sorry

end rectangle_sides_l64_64399


namespace remainder_of_large_number_l64_64692

theorem remainder_of_large_number : 
  (9876543210 : ℤ) % 101 = 73 := 
by
  unfold_coes
  unfold_norm_num
  sorry

end remainder_of_large_number_l64_64692


namespace calculate_expression_l64_64420

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end calculate_expression_l64_64420


namespace cannot_be_external_diagonals_l64_64059

theorem cannot_be_external_diagonals (a b c : ℕ) : 
  ¬(3^2 + 4^2 = 6^2) :=
by
  sorry

end cannot_be_external_diagonals_l64_64059


namespace odd_f_neg1_l64_64336

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 
  if 0 ≤ x 
  then 2^x + 2 * x + b 
  else - (2^(-x) + 2 * (-x) + b)

theorem odd_f_neg1 (b : ℝ) (h : f 0 b = 0) : f (-1) b = -3 :=
by
  sorry

end odd_f_neg1_l64_64336


namespace minimum_number_of_kings_maximum_number_of_non_attacking_kings_l64_64846

-- Definitions for the chessboard and king placement problem

-- Problem (a): Minimum number of kings covering the board
def minimum_kings_covering_board (board_size : Nat) : Nat :=
  sorry

theorem minimum_number_of_kings (h : 6 = board_size) :
  minimum_kings_covering_board 6 = 4 := 
  sorry

-- Problem (b): Maximum number of non-attacking kings
def maximum_non_attacking_kings (board_size : Nat) : Nat :=
  sorry

theorem maximum_number_of_non_attacking_kings (h : 6 = board_size) :
  maximum_non_attacking_kings 6 = 9 :=
  sorry

end minimum_number_of_kings_maximum_number_of_non_attacking_kings_l64_64846


namespace probability_two_female_one_male_l64_64162

-- Define basic conditions
def total_contestants : Nat := 7
def female_contestants : Nat := 4
def male_contestants : Nat := 3
def choose_count : Nat := 3

-- Calculate combinations (binomial coefficients)
def comb (n k : Nat) : Nat := Nat.choose n k

-- Define the probability calculation steps in Lean
def total_ways := comb total_contestants choose_count
def favorable_ways_female := comb female_contestants 2
def favorable_ways_male := comb male_contestants 1
def favorable_ways := favorable_ways_female * favorable_ways_male

theorem probability_two_female_one_male :
  (favorable_ways : ℚ) / (total_ways : ℚ) = 18 / 35 := by
  sorry

end probability_two_female_one_male_l64_64162


namespace smallest_digits_to_append_l64_64245

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l64_64245


namespace triangle_A_and_Area_l64_64618

theorem triangle_A_and_Area :
  ∀ (a b c A B C : ℝ), 
  (b - (1 / 2) * c = a * Real.cos C) 
  → (4 * (b + c) = 3 * b * c) 
  → (a = 2 * Real.sqrt 3)
  → (A = 60) ∧ (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3) :=
by
  intros a b c A B C h1 h2 h3
  sorry

end triangle_A_and_Area_l64_64618


namespace divides_square_sum_implies_divides_l64_64941

theorem divides_square_sum_implies_divides (a b : ℤ) (h : 7 ∣ a^2 + b^2) : 7 ∣ a ∧ 7 ∣ b := 
sorry

end divides_square_sum_implies_divides_l64_64941


namespace square_of_binomial_l64_64480

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b) ^ 2 = 9 * x^2 - 18 * x + a) ↔ a = 9 :=
by
  sorry

end square_of_binomial_l64_64480


namespace shortest_side_length_triangle_l64_64320

noncomputable def triangle_min_angle_side_length (A B : ℝ) (c : ℝ) (tanA tanB : ℝ) (ha : tanA = 1 / 4) (hb : tanB = 3 / 5) (hc : c = Real.sqrt 17) : ℝ :=
   Real.sqrt 2

theorem shortest_side_length_triangle {A B c : ℝ} {tanA tanB : ℝ} 
  (ha : tanA = 1 / 4) (hb : tanB = 3 / 5) (hc : c = Real.sqrt 17) :
  triangle_min_angle_side_length A B c tanA tanB ha hb hc = Real.sqrt 2 :=
sorry

end shortest_side_length_triangle_l64_64320


namespace area_increase_percentage_l64_64948

-- Define the original dimensions l and w as non-zero real numbers
variables (l w : ℝ) (hl : l ≠ 0) (hw : w ≠ 0)

-- Define the new dimensions after increase
def new_length := 1.15 * l
def new_width := 1.25 * w

-- Define the original and new areas
def original_area := l * w
def new_area := new_length l * new_width w

-- The statement to prove
theorem area_increase_percentage :
  ((new_area l w - original_area l w) / original_area l w) * 100 = 43.75 :=
by
  sorry

end area_increase_percentage_l64_64948


namespace points_per_correct_answer_hard_round_l64_64774

theorem points_per_correct_answer_hard_round (total_points easy_points_per average_points_per hard_correct : ℕ) 
(easy_correct average_correct : ℕ) : 
  (total_points = (easy_correct * easy_points_per + average_correct * average_points_per) + (hard_correct * 5)) →
  (easy_correct = 6) →
  (easy_points_per = 2) →
  (average_correct = 2) →
  (average_points_per = 3) →
  (hard_correct = 4) →
  (total_points = 38) →
  5 = 5 := 
by
  intros
  sorry

end points_per_correct_answer_hard_round_l64_64774


namespace min_expression_value_l64_64102

open Real

theorem min_expression_value : ∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 := by
  sorry

end min_expression_value_l64_64102


namespace international_news_duration_l64_64367

theorem international_news_duration
  (total_duration : ℕ := 30)
  (national_news : ℕ := 12)
  (sports : ℕ := 5)
  (weather_forecasts : ℕ := 2)
  (advertising : ℕ := 6) :
  total_duration - national_news - sports - weather_forecasts - advertising = 5 :=
by
  sorry

end international_news_duration_l64_64367


namespace gcd_g_values_l64_64339

def g (x : ℤ) : ℤ := x^2 - 2 * x + 2023

theorem gcd_g_values : gcd (g 102) (g 103) = 1 := by
  sorry

end gcd_g_values_l64_64339


namespace semicircle_area_difference_l64_64591

theorem semicircle_area_difference 
  (A B C P D E F : Type) 
  (h₁ : S₅ - S₆ = 2) 
  (h₂ : S₁ - S₂ = 1) 
  : S₄ - S₃ = 3 :=
by
  -- Using Lean tactics to form the proof, place sorry for now.
  sorry

end semicircle_area_difference_l64_64591


namespace even_three_digit_numbers_l64_64928

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l64_64928


namespace number_of_SYTs_l64_64411

theorem number_of_SYTs (shape: List ℕ): shape = [5,4,3,2,1] →
  StandardYoungTableaux.count shape = 292864 :=
begin
  intros h,
  rw h,
  sorry,
end

end number_of_SYTs_l64_64411


namespace second_card_is_three_l64_64900

theorem second_card_is_three (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
                             (h_sum : a + b + c + d = 30)
                             (h_increasing : a < b ∧ b < c ∧ c < d)
                             (h_dennis : ∀ x y z, x = a → (y ≠ b ∨ z ≠ c ∨ d ≠ 30 - a - y - z))
                             (h_mandy : ∀ x y z, x = b → (y ≠ a ∨ z ≠ c ∨ d ≠ 30 - x - y - z))
                             (h_sandy : ∀ x y z, x = c → (y ≠ a ∨ z ≠ b ∨ d ≠ 30 - x - y - z))
                             (h_randy : ∀ x y z, x = d → (y ≠ a ∨ z ≠ b ∨ c ≠ 30 - x - y - z)) :
  b = 3 := 
sorry

end second_card_is_three_l64_64900


namespace tangent_line_at_1_1_is_5x_plus_y_minus_6_l64_64110

noncomputable def f : ℝ → ℝ :=
  λ x => x^3 - 4*x^2 + 4

def tangent_line_equation (x₀ y₀ m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y - y₀ = m * (x - x₀)

theorem tangent_line_at_1_1_is_5x_plus_y_minus_6 : 
  tangent_line_equation 1 1 (-5) = (λ x y => 5 * x + y - 6 = 0) := 
by
  sorry

end tangent_line_at_1_1_is_5x_plus_y_minus_6_l64_64110


namespace bob_raise_per_hour_l64_64412

theorem bob_raise_per_hour
  (hours_per_week : ℕ := 40)
  (monthly_housing_reduction : ℤ := 60)
  (weekly_earnings_increase : ℤ := 5)
  (weeks_per_month : ℕ := 4) :
  ∃ (R : ℚ), 40 * R - (monthly_housing_reduction / weeks_per_month) + weekly_earnings_increase = 0 ∧
              R = 0.25 := 
by
  sorry

end bob_raise_per_hour_l64_64412


namespace allocation_ways_l64_64251

theorem allocation_ways (programs : Finset ℕ) (grades : Finset ℕ) (h_programs : programs.card = 6) (h_grades : grades.card = 4) : 
  ∃ ways : ℕ, ways = 1080 := 
by 
  sorry

end allocation_ways_l64_64251


namespace smallest_integer_n_l64_64344

theorem smallest_integer_n (m n : ℕ) (r : ℝ) :
  (m = (n + r)^3) ∧ (0 < r) ∧ (r < 1 / 2000) ∧ (m = n^3 + 3 * n^2 * r + 3 * n * r^2 + r^3) →
  n = 26 :=
by 
  sorry

end smallest_integer_n_l64_64344


namespace corn_harvest_l64_64094

theorem corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) (total_bushels : ℕ) :
  rows = 5 →
  stalks_per_row = 80 →
  stalks_per_bushel = 8 →
  total_bushels = (rows * stalks_per_row) / stalks_per_bushel →
  total_bushels = 50 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, mul_comm 5 80] at h4
  norm_num at h4
  exact h4

end corn_harvest_l64_64094


namespace negation_of_universal_l64_64994
-- Import the Mathlib library to provide the necessary mathematical background

-- State the theorem that we want to prove. This will state that the negation of the universal proposition is an existential proposition
theorem negation_of_universal :
  (¬ (∀ x : ℝ, x > 0)) ↔ (∃ x : ℝ, x ≤ 0) :=
sorry

end negation_of_universal_l64_64994


namespace freshmen_count_l64_64648

theorem freshmen_count (n : ℕ) : n < 600 ∧ n % 25 = 24 ∧ n % 19 = 10 ↔ n = 574 := 
by sorry

end freshmen_count_l64_64648


namespace largest_a1_l64_64903

theorem largest_a1
  (a : ℕ+ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_eq : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h_initial : a 1 = a 10) :
  ∃ (max_a1 : ℝ), max_a1 = 16 ∧ ∀ x, x = a 1 → x ≤ 16 :=
by
  sorry

end largest_a1_l64_64903


namespace initial_population_l64_64705

theorem initial_population (P : ℝ) : 
  (0.9 * P * 0.85 = 2907) → P = 3801 := by
  sorry

end initial_population_l64_64705


namespace sin_neg_765_eq_neg_sqrt2_div_2_l64_64189

theorem sin_neg_765_eq_neg_sqrt2_div_2 :
  Real.sin (-765 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_765_eq_neg_sqrt2_div_2_l64_64189


namespace youngest_child_age_l64_64717

theorem youngest_child_age (total_bill mother_cost twin_age_cost total_age : ℕ) (twin_age youngest_age : ℕ) 
  (h1 : total_bill = 1485) (h2 : mother_cost = 695) (h3 : twin_age_cost = 65) 
  (h4 : total_age = (total_bill - mother_cost) / twin_age_cost)
  (h5 : total_age = 2 * twin_age + youngest_age) :
  youngest_age = 2 :=
by
  -- sorry: Proof to be completed later
  sorry

end youngest_child_age_l64_64717


namespace Chandler_more_rolls_needed_l64_64892

theorem Chandler_more_rolls_needed :
  let total_goal := 12
  let sold_to_grandmother := 3
  let sold_to_uncle := 4
  let sold_to_neighbor := 3
  let total_sold := sold_to_grandmother + sold_to_uncle + sold_to_neighbor
  total_goal - total_sold = 2 :=
by
  sorry

end Chandler_more_rolls_needed_l64_64892


namespace smallest_digits_to_append_l64_64236

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l64_64236


namespace GPA_of_rest_of_classroom_l64_64016

variable (n : ℕ) (x : ℝ)
variable (H1 : ∀ n, n > 0)
variable (H2 : (15 * n + 2 * n * x) / (3 * n) = 17)

theorem GPA_of_rest_of_classroom (n : ℕ) (H1 : ∀ n, n > 0) (H2 : (15 * n + 2 * n * x) / (3 * n) = 17) : x = 18 := by
  sorry

end GPA_of_rest_of_classroom_l64_64016


namespace sum_of_B_coordinates_l64_64597

theorem sum_of_B_coordinates 
  (x y : ℝ) 
  (A : ℝ × ℝ) 
  (M : ℝ × ℝ)
  (midpoint_x : (A.1 + x) / 2 = M.1) 
  (midpoint_y : (A.2 + y) / 2 = M.2) 
  (A_conds : A = (7, -1))
  (M_conds : M = (4, 3)) :
  x + y = 8 :=
by 
  sorry

end sum_of_B_coordinates_l64_64597


namespace square_window_side_length_l64_64955

-- Definitions based on the conditions
def total_panes := 8
def rows := 2
def cols := 4
def height_ratio := 3
def width_ratio := 1
def border_width := 3

-- The statement to prove
theorem square_window_side_length :
  let height := 3 * (1 : ℝ)
  let width := 1 * (1 : ℝ)
  let total_width := cols * width + (cols + 1) * border_width
  let total_height := rows * height + (rows + 1) * border_width
  total_width = total_height → total_width = 27 :=
by
  sorry

end square_window_side_length_l64_64955


namespace total_dots_on_left_faces_l64_64066

-- Define the number of dots on the faces A, B, C, and D
def d_A : ℕ := 3
def d_B : ℕ := 5
def d_C : ℕ := 6
def d_D : ℕ := 5

-- The statement we need to prove
theorem total_dots_on_left_faces : d_A + d_B + d_C + d_D = 19 := by
  sorry

end total_dots_on_left_faces_l64_64066


namespace sum_of_consecutive_odds_l64_64530

theorem sum_of_consecutive_odds (N1 N2 N3 : ℕ) (h1 : N1 % 2 = 1) (h2 : N2 % 2 = 1) (h3 : N3 % 2 = 1)
  (h_consec1 : N2 = N1 + 2) (h_consec2 : N3 = N2 + 2) (h_max : N3 = 27) : 
  N1 + N2 + N3 = 75 := by
  sorry

end sum_of_consecutive_odds_l64_64530


namespace trebled_resultant_is_correct_l64_64560

-- Definitions based on the conditions provided in step a)
def initial_number : ℕ := 5
def doubled_result : ℕ := initial_number * 2
def added_15_result : ℕ := doubled_result + 15
def trebled_resultant : ℕ := added_15_result * 3

-- We need to prove that the trebled resultant is equal to 75
theorem trebled_resultant_is_correct : trebled_resultant = 75 :=
by
  sorry

end trebled_resultant_is_correct_l64_64560


namespace total_heads_l64_64712

/-- There are H hens and C cows. Each hen has 1 head and 2 feet, and each cow has 1 head and 4 feet.
Given that the total number of feet is 140 and there are 26 hens, prove that the total number of heads is 48. -/
theorem total_heads (H C : ℕ) (h1 : 2 * H + 4 * C = 140) (h2 : H = 26) : H + C = 48 := by
  sorry

end total_heads_l64_64712


namespace number_of_men_in_engineering_department_l64_64961

theorem number_of_men_in_engineering_department (T : ℝ) (h1 : 0.30 * T = 180) : 
  0.70 * T = 420 :=
by 
  -- The proof will be done here, but for now, we skip it.
  sorry

end number_of_men_in_engineering_department_l64_64961


namespace joan_total_cost_is_correct_l64_64627

def year1_home_games := 6
def year1_away_games := 3
def year1_home_playoff_games := 1
def year1_away_playoff_games := 1

def year2_home_games := 2
def year2_away_games := 2
def year2_home_playoff_games := 1
def year2_away_playoff_games := 0

def home_game_ticket := 60
def away_game_ticket := 75
def home_playoff_ticket := 120
def away_playoff_ticket := 100

def friend_home_game_ticket := 45
def friend_away_game_ticket := 75

def home_game_transportation := 25
def away_game_transportation := 50

noncomputable def year1_total_cost : ℕ :=
  (year1_home_games * (home_game_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year1_away_games * (away_game_ticket + friend_away_game_ticket + away_game_transportation)) +
  (year1_home_playoff_games * (home_playoff_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year1_away_playoff_games * (away_playoff_ticket + friend_away_game_ticket + away_game_transportation))

noncomputable def year2_total_cost : ℕ :=
  (year2_home_games * (home_game_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year2_away_games * (away_game_ticket + friend_away_game_ticket + away_game_transportation)) +
  (year2_home_playoff_games * (home_playoff_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year2_away_playoff_games * (away_playoff_ticket + friend_away_game_ticket + away_game_transportation))

noncomputable def total_cost : ℕ := year1_total_cost + year2_total_cost

theorem joan_total_cost_is_correct : total_cost = 2645 := by
  sorry

end joan_total_cost_is_correct_l64_64627


namespace part1_part2_l64_64549
open Real

-- Part 1
theorem part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  0 < (sqrt (1 + x) + sqrt (1 - x) + 2) * (sqrt (1 - x^2) + 1) ∧
  (sqrt (1 + x) + sqrt (1 - x) + 2) * (sqrt (1 - x^2) + 1) ≤ 8 := 
sorry

-- Part 2
theorem part2 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  ∃ β > 0, β = 4 ∧ sqrt (1 + x) + sqrt (1 - x) ≤ 2 - x^2 / β :=
sorry

end part1_part2_l64_64549


namespace impossible_to_all_minus_l64_64426

def initial_grid : List (List Int) :=
  [[1, 1, -1, 1], 
   [-1, -1, 1, 1], 
   [1, 1, 1, 1], 
   [1, -1, 1, -1]]

-- Define the operation of flipping a row
def flip_row (grid : List (List Int)) (r : Nat) : List (List Int) :=
  grid.mapIdx (fun i row => if i == r then row.map (fun x => -x) else row)

-- Define the operation of flipping a column
def flip_col (grid : List (List Int)) (c : Nat) : List (List Int) :=
  grid.map (fun row => row.mapIdx (fun j x => if j == c then -x else x))

-- Predicate to check if all elements in the grid are -1
def all_minus (grid : List (List Int)) : Prop :=
  grid.all (fun row => row.all (fun x => x = -1))

-- The main theorem
theorem impossible_to_all_minus (init : List (List Int)) (hf1 : init = initial_grid) :
  ∀ grid, (grid = init ∨ ∃ r, grid = flip_row grid r ∨ ∃ c, grid = flip_col grid c) →
  ¬ all_minus grid := by
    sorry

end impossible_to_all_minus_l64_64426


namespace trajectory_equation_l64_64967

theorem trajectory_equation 
  (P : ℝ × ℝ)
  (h : (P.2 / (P.1 + 4)) * (P.2 / (P.1 - 4)) = -4 / 9) :
  P.1 ≠ 4 ∧ P.1 ≠ -4 → P.1^2 / 64 + P.2^2 / (64 / 9) = 1 :=
by
  sorry

end trajectory_equation_l64_64967


namespace bees_on_second_day_l64_64005

-- Define the number of bees on the first day
def bees_first_day : ℕ := 144

-- Define the multiplication factor
def multiplication_factor : ℕ := 3

-- Define the number of bees on the second day
def bees_second_day : ℕ := bees_first_day * multiplication_factor

-- Theorem stating the number of bees on the second day is 432
theorem bees_on_second_day : bees_second_day = 432 := 
by
  sorry

end bees_on_second_day_l64_64005


namespace prob_heart_and_club_not_king_l64_64201

-- Definitions for the deck, cards, and dealing process
def deck : finset (Σ x : ℕ, x < 52) := sorry
def is_heart (card : Σ x : ℕ, x < 52) : Prop := sorry
def is_club (card : Σ x : ℕ, x < 52) : Prop := sorry
def is_king (card : Σ x : ℕ, x < 52) : Prop := sorry

-- The probability of an event
def probability {α : Type*} (s : finset α) (p : α → Prop) [decidable_pred p] : ℝ := 
  (s.filter p).card / s.card

-- The event of dealing two cards with the given conditions
def first_is_heart (cards : list (Σ x : ℕ, x < 52)) : Prop := is_heart (cards.head)
def second_is_club_not_king (cards : list (Σ x : ℕ, x < 52)) : Prop := 
  ¬ (is_king (cards.nth 1).get_or_else (cards.head)) ∧ is_club (cards.nth 1).get_or_else (cards.head)

-- The probability of the specific chained event 
def event_prob := 
  probability deck (λ card1, is_heart card1) * 
  probability (deck.erase (deck.filter is_heart).choose sorry) (λ card2, is_club card2 ∧ ¬ is_king card2)

theorem prob_heart_and_club_not_king : event_prob = 1 / 17 :=
sorry

end prob_heart_and_club_not_king_l64_64201


namespace distance_between_C_and_A_l64_64951

theorem distance_between_C_and_A 
    (A B C : Type)
    (d_AB : ℝ) (d_BC : ℝ)
    (h1 : d_AB = 8)
    (h2 : d_BC = 10) :
    ∃ x : ℝ, 2 ≤ x ∧ x ≤ 18 ∧ ¬ (∃ y : ℝ, y = x) :=
sorry

end distance_between_C_and_A_l64_64951


namespace geom_seq_sum_l64_64589

noncomputable def geom_seq (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
a₁ * r^(n-1)

theorem geom_seq_sum (a₁ r : ℝ) (h_pos : 0 < a₁) (h_pos_r : 0 < r)
  (h : a₁ * (geom_seq a₁ r 5) + 2 * (geom_seq a₁ r 3) * (geom_seq a₁ r 6) + a₁ * (geom_seq a₁ r 11) = 16) :
  (geom_seq a₁ r 3 + geom_seq a₁ r 6) = 4 :=
sorry

end geom_seq_sum_l64_64589


namespace zoo_total_animals_l64_64276

theorem zoo_total_animals (penguins polar_bears : ℕ)
  (h1 : penguins = 21)
  (h2 : polar_bears = 2 * penguins) :
  penguins + polar_bears = 63 := by
   sorry

end zoo_total_animals_l64_64276


namespace largest_of_three_roots_l64_64197

theorem largest_of_three_roots (p q r : ℝ) (hpqr_sum : p + q + r = 3) 
    (hpqr_prod_sum : p * q + p * r + q * r = -8) (hpqr_prod : p * q * r = -15) :
    max p (max q r) = 3 := 
sorry

end largest_of_three_roots_l64_64197


namespace smallest_digits_to_append_l64_64242

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l64_64242


namespace rectangle_length_to_width_ratio_l64_64588

variables (s : ℝ)

-- Given conditions
def small_square_side := s
def large_square_side := 3 * s
def rectangle_length := large_square_side
def rectangle_width := large_square_side - 2 * small_square_side

-- Theorem to prove the ratio of the length to the width of the rectangle
theorem rectangle_length_to_width_ratio : 
  ∃ (r : ℝ), r = rectangle_length s / rectangle_width s ∧ r = 3 := 
by
  sorry

end rectangle_length_to_width_ratio_l64_64588


namespace chandler_needs_to_sell_more_rolls_l64_64895

/-- Chandler's wrapping paper selling condition. -/
def chandler_needs_to_sell : ℕ := 12

def sold_to_grandmother : ℕ := 3
def sold_to_uncle : ℕ := 4
def sold_to_neighbor : ℕ := 3

def total_sold : ℕ := sold_to_grandmother + sold_to_uncle + sold_to_neighbor

theorem chandler_needs_to_sell_more_rolls : chandler_needs_to_sell - total_sold = 2 :=
by
  sorry

end chandler_needs_to_sell_more_rolls_l64_64895


namespace coeffs_of_polynomial_l64_64366

def p (x y : ℚ) : ℚ := 3 * x * y^2 - 2 * y - 1

theorem coeffs_of_polynomial : 
  let p := p in 
  (coeff_of_linear_term : ℚ, coeff_of_constant_term : ℚ) = (-2, -1) :=
by
  sorry

end coeffs_of_polynomial_l64_64366


namespace remainder_of_9876543210_div_101_l64_64686

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l64_64686


namespace three_consecutive_arithmetic_l64_64890

def seq (n : ℕ) : ℝ := 
  if n % 2 = 1 then (n : ℝ)
  else 2 * 3^(n / 2 - 1)

theorem three_consecutive_arithmetic (m : ℕ) (h_m : seq m + seq (m+2) = 2 * seq (m+1)) : m = 1 :=
  sorry

end three_consecutive_arithmetic_l64_64890


namespace perpendicular_condition_sufficient_but_not_necessary_l64_64443

theorem perpendicular_condition_sufficient_but_not_necessary (m : ℝ) (h : m = -1) :
  (∀ x y : ℝ, mx + (2 * m - 1) * y + 1 = 0 ∧ 3 * x + m * y + 2 = 0) → (m = 0 ∨ m = -1) → (m = 0 ∨ m = -1) :=
by
  intro h1 h2
  sorry

end perpendicular_condition_sufficient_but_not_necessary_l64_64443


namespace line_through_vertex_has_two_a_values_l64_64899

-- Definitions for the line and parabola as conditions
def line_eq (a x : ℝ) : ℝ := 2 * x + a
def parabola_eq (a x : ℝ) : ℝ := x^2 + 2 * a^2

-- The proof problem
theorem line_through_vertex_has_two_a_values :
  (∃ a1 a2 : ℝ, (a1 ≠ a2) ∧ (line_eq a1 0 = parabola_eq a1 0) ∧ (line_eq a2 0 = parabola_eq a2 0)) ∧
  (∀ a : ℝ, line_eq a 0 = parabola_eq a 0 → (a = 0 ∨ a = 1/2)) :=
sorry

end line_through_vertex_has_two_a_values_l64_64899


namespace decreasing_interval_l64_64520

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 4)

theorem decreasing_interval :
  ∀ x ∈ Set.Icc 0 (2 * Real.pi), x ∈ Set.Icc (3 * Real.pi / 4) (2 * Real.pi) ↔ (∀ ε > 0, f x > f (x + ε)) := 
sorry

end decreasing_interval_l64_64520


namespace nebraska_license_plate_increase_l64_64352

open Nat

theorem nebraska_license_plate_increase :
  let old_plates : ℕ := 26 * 10^3
  let new_plates : ℕ := 26^2 * 10^4
  new_plates / old_plates = 260 :=
by
  -- Definitions based on conditions
  let old_plates : ℕ := 26 * 10^3
  let new_plates : ℕ := 26^2 * 10^4
  -- Assertion to prove
  show new_plates / old_plates = 260
  sorry

end nebraska_license_plate_increase_l64_64352


namespace number_of_rows_containing_53_l64_64472

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l64_64472


namespace amare_fabric_needed_l64_64989

-- Definitions for the conditions
def fabric_per_dress_yards : ℝ := 5.5
def number_of_dresses : ℕ := 4
def fabric_owned_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- Total fabric needed in yards
def total_fabric_needed_yards : ℝ := fabric_per_dress_yards * number_of_dresses

-- Total fabric needed in feet
def total_fabric_needed_feet : ℝ := total_fabric_needed_yards * yard_to_feet

-- Fabric still needed
def fabric_still_needed : ℝ := total_fabric_needed_feet - fabric_owned_feet

-- Proof
theorem amare_fabric_needed : fabric_still_needed = 59 := by
  sorry

end amare_fabric_needed_l64_64989


namespace geometric_sequence_sum_l64_64529

noncomputable def geometric_sequence (a : ℕ → ℝ) (r: ℝ): Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (r: ℝ)
  (h_geometric : geometric_sequence a r)
  (h_ratio : r = 2)
  (h_sum_condition : a 1 + a 4 + a 7 = 10) :
  a 3 + a 6 + a 9 = 20 := 
sorry

end geometric_sequence_sum_l64_64529


namespace weight_of_daughter_l64_64658

theorem weight_of_daughter 
  (M D C : ℝ)
  (h1 : M + D + C = 120)
  (h2 : D + C = 60)
  (h3 : C = (1 / 5) * M)
  : D = 48 :=
by
  sorry

end weight_of_daughter_l64_64658


namespace positive_diff_between_median_and_mode_eq_16_l64_64044

/-
  List of numbers derived from the stem and leaf plot:
  tens_units: List Nat := [21, 22, 22, 23, 25, 32, 32, 32, 36, 36, 41, 41, 47, 48, 49, 50, 53, 53, 54, 57, 60, 61, 65, 65, 68]
-/

def tens_units: List Nat := [21, 22, 22, 23, 25, 32, 32, 32, 36, 36, 41, 41, 47, 48, 49, 50, 53, 53, 54, 57, 60, 61, 65, 65, 68]

noncomputable def median (l : List Nat) : Nat :=
  let sorted := l.sorted
  sorted.get! (sorted.length / 2)

noncomputable def modes (l : List Nat) : List Nat := 
  l.groupBy id
  |>.map (λ g => (g.head!, g.length))
  |>.filter (λ p => p.2 = l.groupBy id |>.map List.length |>.maximum)
  |>.map Prod.fst

noncomputable def positive_differences (a : Nat) (ls : List Nat) : List Nat :=
  ls.map (λ b => if b > a then b - a else a - b)

theorem positive_diff_between_median_and_mode_eq_16 :
  positive_differences (median tens_units) (modes tens_units) = [16] ∨ positive_differences (median tens_units) (modes tens_units) = [26, 16] := 
by 
  -- proof omitted
  sorry

end positive_diff_between_median_and_mode_eq_16_l64_64044


namespace remainder_div_101_l64_64693

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l64_64693


namespace general_formula_minimum_n_l64_64905

-- Definitions based on given conditions
def arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d
def sum_arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- Conditions of the problem
def a2 : ℤ := -5
def S5 : ℤ := -20

-- Proving the general formula of the sequence
theorem general_formula :
  ∃ a₁ d, arith_seq a₁ d 2 = a2 ∧ sum_arith_seq a₁ d 5 = S5 ∧ (∀ n, arith_seq a₁ d n = n - 7) :=
by
  sorry

-- Proving the minimum value of n for which Sn > an
theorem minimum_n :
  ∃ n : ℕ, (n > 14) ∧ sum_arith_seq (-6) 1 n > arith_seq (-6) 1 n :=
by
  sorry

end general_formula_minimum_n_l64_64905


namespace num_pupils_is_40_l64_64701

-- given conditions
def incorrect_mark : ℕ := 83
def correct_mark : ℕ := 63
def mark_difference : ℕ := incorrect_mark - correct_mark
def avg_increase : ℚ := 1 / 2

-- the main problem statement to prove
theorem num_pupils_is_40 (n : ℕ) (h : (mark_difference : ℚ) / n = avg_increase) : n = 40 := 
sorry

end num_pupils_is_40_l64_64701


namespace cost_of_one_pack_of_gummy_bears_l64_64328

theorem cost_of_one_pack_of_gummy_bears
    (num_chocolate_bars : ℕ)
    (num_gummy_bears : ℕ)
    (num_chocolate_chips : ℕ)
    (total_cost : ℕ)
    (cost_per_chocolate_bar : ℕ)
    (cost_per_chocolate_chip : ℕ)
    (cost_of_one_gummy_bear_pack : ℕ)
    (h1 : num_chocolate_bars = 10)
    (h2 : num_gummy_bears = 10)
    (h3 : num_chocolate_chips = 20)
    (h4 : total_cost = 150)
    (h5 : cost_per_chocolate_bar = 3)
    (h6 : cost_per_chocolate_chip = 5)
    (h7 : num_chocolate_bars * cost_per_chocolate_bar +
          num_gummy_bears * cost_of_one_gummy_bear_pack +
          num_chocolate_chips * cost_per_chocolate_chip = total_cost) :
    cost_of_one_gummy_bear_pack = 2 := by
  sorry

end cost_of_one_pack_of_gummy_bears_l64_64328


namespace number_of_cirrus_clouds_l64_64669

def C_cb := 3
def C_cu := 12 * C_cb
def C_ci := 4 * C_cu

theorem number_of_cirrus_clouds : C_ci = 144 :=
by
  sorry

end number_of_cirrus_clouds_l64_64669


namespace range_of_a_l64_64784

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - Real.log x / x + a

theorem range_of_a (a : ℝ) :
  (∃ x > 0, f x a = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1 := by
  sorry

end range_of_a_l64_64784


namespace student_chose_number_l64_64844

theorem student_chose_number (x : ℤ) (h : 2 * x - 148 = 110) : x = 129 := 
by
  sorry

end student_chose_number_l64_64844


namespace crates_second_trip_l64_64714

theorem crates_second_trip
  (x y : Nat) 
  (h1 : x + y = 12)
  (h2 : x = 5) :
  y = 7 :=
by
  sorry

end crates_second_trip_l64_64714


namespace grace_is_14_l64_64608

def GraceAge (G F C E D : ℕ) : Prop :=
  G = F - 6 ∧ F = C + 2 ∧ E = C + 3 ∧ D = E - 4 ∧ D = 17

theorem grace_is_14 (G F C E D : ℕ) (h : GraceAge G F C E D) : G = 14 :=
by sorry

end grace_is_14_l64_64608


namespace y_intercept_of_line_l64_64029

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end y_intercept_of_line_l64_64029


namespace intersecting_diagonals_probability_l64_64766

theorem intersecting_diagonals_probability (n : ℕ) (h : n > 0) : 
  let vertices := 2 * n + 1 in
  let diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2 in
  let intersecting_pairs := ((vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24) in
  let probability := (n * (2 * n - 1) * 2) / (3 * ((2 * n ^ 2 - n - 1) * (2 * n ^ 2 - n - 2))) in
  (intersecting_pairs : ℝ) / (pairs_diagonals : ℝ) = probability :=
begin
  -- Proof to be provided
  sorry
end

end intersecting_diagonals_probability_l64_64766


namespace burglar_goods_value_l64_64354

theorem burglar_goods_value (V : ℝ) (S : ℝ) (S_increased : ℝ) (S_total : ℝ) (h1 : S = V / 5000) (h2 : S_increased = 1.25 * S) (h3 : S_total = S_increased + 2) (h4 : S_total = 12) : V = 40000 := by
  sorry

end burglar_goods_value_l64_64354


namespace line_passes_through_vertex_twice_l64_64896

theorem line_passes_through_vertex_twice :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ (∀ a, (y = 2 * x + a ∧ ∃ (x y : ℝ), y = x^2 + 2 * a^2) ↔ a = a₁ ∨ a = a₂) :=
by
  sorry

end line_passes_through_vertex_twice_l64_64896


namespace count_groups_U_AB_l64_64915

open Finset

theorem count_groups_U_AB :
  let U := ({1, 2, 3, 4, 5} : Finset ℕ) in
  let mutually_exclusive (A B : Finset ℕ) : Prop := A ∩ B = ∅ ∧ A.nonempty ∧ B.nonempty in
  let groups_U (A B : Finset ℕ) := mutually_exclusive A B ∧ U(A, B) ≠ U(B, A) in
  (card {p : Finset ℕ × Finset ℕ // groups_U p.1 p.2}) = 180 :=
by {sorry}

end count_groups_U_AB_l64_64915


namespace arithmetic_sequence_sum_formula_l64_64904

noncomputable def S (n : ℕ) := -a_n - (1/2)^(n - 1)

theorem arithmetic_sequence (h1: ∀ n : ℕ, n > 0 →  S (n) = -a_n - (1/2)^(n - 1)) : 
  ∀ n : ℕ, n > 0 → 2^n * S n = -1 + (n - 1) * (-1) :=
sorry

theorem sum_formula (h1: ∀ n : ℕ, n > 0 →  S (n) = -a_n - (1/2)^(n - 1)) : 
  ∀ n: ℕ, n > 0 → (∑ i in range n, S i) = (n + 2)/(2^n) - 2 :=
sorry

end arithmetic_sequence_sum_formula_l64_64904


namespace unique_function_satisfying_equation_l64_64879

theorem unique_function_satisfying_equation :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x^2 + f y) = y + f x^2) → ∀ x : ℝ, f x = x :=
by
  intro f h
  sorry

end unique_function_satisfying_equation_l64_64879


namespace sum_of_digits_of_k_l64_64182

theorem sum_of_digits_of_k : 
  ∃ (k : ℕ), (0 < k) ∧ (k / 12) / (15 / k) = 20 ∧ (Nat.digits 10 k).sum = 6 := 
by
  sorry

end sum_of_digits_of_k_l64_64182


namespace y_intercept_of_line_l64_64031

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end y_intercept_of_line_l64_64031


namespace odd_nat_numbers_eq_1_l64_64343

-- Definitions of conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem odd_nat_numbers_eq_1
  (a b c d : ℕ)
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : is_odd a) (h5 : is_odd b) (h6 : is_odd c) (h7 : is_odd d)
  (h8 : a * d = b * c)
  (h9 : is_power_of_two (a + d))
  (h10 : is_power_of_two (b + c)) :
  a = 1 :=
sorry

end odd_nat_numbers_eq_1_l64_64343


namespace problem_statement_l64_64164

variable (a b : ℝ)

-- Conditions
variable (h1 : a > 0) (h2 : b > 0) (h3 : ∃ x, x = (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a)))

-- The Lean theorem statement for the problem
theorem problem_statement : 
  ∀ x, (x = (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))) →
  (2 * a * Real.sqrt (1 + x^2)) / (x + Real.sqrt (1 + x^2)) = a + b := 
sorry


end problem_statement_l64_64164


namespace triangle_similarity_length_RY_l64_64971

theorem triangle_similarity_length_RY
  (P Q R X Y Z : Type)
  [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (PQ : ℝ) (XY : ℝ) (RY_length : ℝ)
  (h1 : PQ = 10)
  (h2 : XY = 6)
  (h3 : ∀ (PR QR PX QX RZ : ℝ) (angle_PY_RZ : ℝ),
    PR + RY_length = PX ∧
    QR + RY_length = QX ∧ 
    angle_PY_RZ = 120 ∧
    PR > 0 ∧ QR > 0 ∧ RY_length > 0)
  (h4 : XY / PQ = RY_length / (PQ + RY_length)) :
  RY_length = 15 := by
  sorry

end triangle_similarity_length_RY_l64_64971


namespace cos_1_approx_to_01_no_zero_points_in_interval_l64_64350

noncomputable def cos_approx (x : ℝ) := 1 - (x^2) / 2 + (x^4) / (4*3*2) - (x^6) / (6*5*4*3*2)  -- Compares with cos_taylor_series

theorem cos_1_approx_to_01 : abs (cos_approx 1 - cos 1) < 0.01 :=
by
  sorry

theorem no_zero_points_in_interval : ∀ x ∈ Set.Ioo ((2:ℝ)/3) 1, (exp x - 1/x) ≠ 0 :=
by
  sorry

end cos_1_approx_to_01_no_zero_points_in_interval_l64_64350


namespace corn_harvest_l64_64093

theorem corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) (total_bushels : ℕ) :
  rows = 5 →
  stalks_per_row = 80 →
  stalks_per_bushel = 8 →
  total_bushels = (rows * stalks_per_row) / stalks_per_bushel →
  total_bushels = 50 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, mul_comm 5 80] at h4
  norm_num at h4
  exact h4

end corn_harvest_l64_64093


namespace net_error_24x_l64_64707

theorem net_error_24x (x : ℕ) : 
  let penny_value := 1
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let error_pennies := (nickel_value - penny_value) * x
  let error_nickels := (dime_value - nickel_value) * x
  let error_dimes := (quarter_value - dime_value) * x
  let total_error := error_pennies + error_nickels + error_dimes
  total_error = 24 * x := 
by 
  sorry

end net_error_24x_l64_64707


namespace domain_of_h_l64_64679

-- Defining the function h(x)
def h (x : ℝ) : ℝ := (5 * x + 3) / (x - 4)

-- Stating the theorem
theorem domain_of_h :
  ∀ x : ℝ, x ≠ 4 ↔ x ∈ {x : ℝ | x ≠ 4} := 
sorry

end domain_of_h_l64_64679


namespace first_ship_rescued_boy_l64_64077

noncomputable def river_speed : ℝ := 3 -- River speed is 3 km/h

-- Define the speeds of the ships
def ship1_speed_upstream : ℝ := 4 
def ship2_speed_upstream : ℝ := 6 
def ship3_speed_upstream : ℝ := 10 

-- Define the distance downstream where the boy was found
def boy_distance_from_bridge : ℝ := 6

-- Define the equation for the first ship
def first_ship_equation (c : ℝ) : Prop := (10 - c) / (4 + c) = 1 + 6 / c

-- The problem to prove:
theorem first_ship_rescued_boy : first_ship_equation river_speed :=
by sorry

end first_ship_rescued_boy_l64_64077


namespace Ben_win_probability_l64_64664

theorem Ben_win_probability (lose_prob : ℚ) (no_tie : ¬ ∃ (p : ℚ), p ≠ lose_prob ∧ p + lose_prob = 1) 
  (h : lose_prob = 5/8) : (1 - lose_prob) = 3/8 := by
  sorry

end Ben_win_probability_l64_64664


namespace even_three_digit_numbers_l64_64929

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l64_64929


namespace pascal_contains_53_l64_64457

theorem pascal_contains_53 (n : ℕ) (h1 : Nat.Prime 53) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 52 ∧ nat.choose 53 k = 53) (h3 : ∀ m < 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) (h4 : ∀ m > 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) : 
  (n = 53) → (n = 1) := 
by
  intros
  sorry

end pascal_contains_53_l64_64457


namespace area_of_region_l64_64568

theorem area_of_region :
  let x := fun t : ℝ => 6 * Real.cos t
  let y := fun t : ℝ => 2 * Real.sin t
  (∫ t in (Real.pi / 3)..(Real.pi / 2), (x t) * (deriv y t)) * 2 = 2 * Real.pi - 3 * Real.sqrt 3 := by
  let x := fun t : ℝ => 6 * Real.cos t
  let y := fun t : ℝ => 2 * Real.sin t
  have h1 : ∫ t in (Real.pi / 3)..(Real.pi / 2), x t * deriv y t = 12 * ∫ t in (Real.pi / 3)..(Real.pi / 2), (1 + Real.cos (2*t)) / 2 := sorry
  have h2 : 12 * ∫ t in (Real.pi / 3)..(Real.pi / 2), (1 + Real.cos (2 * t)) / 2 = 2 * Real.pi - 3 * Real.sqrt 3 := sorry
  sorry

end area_of_region_l64_64568


namespace amare_additional_fabric_needed_l64_64990

-- Defining the conditions
def yards_per_dress : ℝ := 5.5
def num_dresses : ℝ := 4
def initial_fabric_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- The theorem to prove
theorem amare_additional_fabric_needed : 
  (yards_per_dress * num_dresses * yard_to_feet) - initial_fabric_feet = 59 := 
by
  sorry

end amare_additional_fabric_needed_l64_64990


namespace compare_expressions_l64_64872

-- Define the theorem statement
theorem compare_expressions (x : ℝ) : (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry -- The proof is omitted.

end compare_expressions_l64_64872


namespace remainder_div_101_l64_64694

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l64_64694


namespace charlie_acorns_l64_64290

theorem charlie_acorns (x y : ℕ) (hc hs : ℕ)
  (h5 : x = 5 * hc)
  (h7 : y = 7 * hs)
  (total : x + y = 145)
  (holes : hs = hc - 3) :
  x = 70 :=
by
  sorry

end charlie_acorns_l64_64290


namespace append_digits_divisible_by_all_less_than_10_l64_64219

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l64_64219


namespace blue_tickets_per_red_ticket_l64_64374

-- Definitions based on conditions
def yellow_tickets_to_win_bible : Nat := 10
def red_tickets_per_yellow_ticket : Nat := 10
def blue_tickets_needed : Nat := 163
def additional_yellow_tickets_needed (current_yellow : Nat) : Nat := yellow_tickets_to_win_bible - current_yellow
def additional_red_tickets_needed (current_red : Nat) (needed_yellow : Nat) : Nat := needed_yellow * red_tickets_per_yellow_ticket - current_red

-- Given conditions
def current_yellow_tickets : Nat := 8
def current_red_tickets : Nat := 3
def current_blue_tickets : Nat := 7
def needed_yellow_tickets : Nat := additional_yellow_tickets_needed current_yellow_tickets
def needed_red_tickets : Nat := additional_red_tickets_needed current_red_tickets needed_yellow_tickets

-- Theorem to prove
theorem blue_tickets_per_red_ticket : blue_tickets_needed / needed_red_tickets = 10 :=
by
  sorry

end blue_tickets_per_red_ticket_l64_64374


namespace price_of_ice_cream_l64_64552

theorem price_of_ice_cream (x : ℝ) :
  (225 * x + 125 * 0.52 = 200) → (x = 0.60) :=
sorry

end price_of_ice_cream_l64_64552


namespace range_of_derivative_max_value_of_a_l64_64632

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.cos x - (x - Real.pi / 2) * Real.sin x

-- Define the derivative of f
noncomputable def f' (a x : ℝ) : ℝ :=
  -(1 + a) * Real.sin x - (x - Real.pi / 2) * Real.cos x

-- Part (1): Prove the range of the derivative when a = -1 is [0, π/2]
theorem range_of_derivative (x : ℝ) (h0 : 0 ≤ x) (hπ : x ≤ Real.pi / 2) :
  (0 ≤ f' (-1) x) ∧ (f' (-1) x ≤ Real.pi / 2) := 
sorry

-- Part (2): Prove the maximum value of 'a' when f(x) ≤ 0 always holds
theorem max_value_of_a (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f a x ≤ 0) :
  a ≤ -1 := 
sorry

end range_of_derivative_max_value_of_a_l64_64632


namespace smallest_append_digits_l64_64217

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l64_64217


namespace quadratic_properties_l64_64594

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties 
  (a b c : ℝ) (ha : a ≠ 0) (h_passes_through : quadratic_function a b c 0 = 1) (h_unique_zero : quadratic_function a b c (-1) = 0) :
  quadratic_function a b c = quadratic_function 1 2 1 ∧ 
  (∀ k, ∃ g,
    (k ≤ -2 → g = k + 3) ∧ 
    (-2 < k ∧ k ≤ 6 → g = -((k^2 - 4*k) / 4)) ∧ 
    (6 < k → g = 9 - 2*k)) :=
sorry

end quadratic_properties_l64_64594


namespace range_of_m_l64_64264

def G (x y : ℤ) : ℤ :=
  if x ≥ y then x - y
  else y - x

theorem range_of_m (m : ℤ) :
  (∀ x, 0 < x → G x 1 > 4 → G (-1) x ≤ m) ↔ 9 ≤ m ∧ m < 10 :=
sorry

end range_of_m_l64_64264


namespace intersection_correct_l64_64755

-- Conditions
def M : Set ℤ := { -1, 0, 1, 3, 5 }
def N : Set ℤ := { -2, 1, 2, 3, 5 }

-- Statement to prove
theorem intersection_correct : M ∩ N = { 1, 3, 5 } :=
by
  sorry

end intersection_correct_l64_64755


namespace smallest_digits_to_append_l64_64207

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l64_64207


namespace triangle_third_side_range_l64_64675

theorem triangle_third_side_range {x : ℤ} : 
  (7 < x ∧ x < 17) → (4 ≤ x ∧ x ≤ 16) :=
by
  sorry

end triangle_third_side_range_l64_64675


namespace equivalent_discount_l64_64171

theorem equivalent_discount {x : ℝ} (h₀ : x > 0) :
    let first_discount := 0.10
    let second_discount := 0.20
    let single_discount := 0.28
    (1 - (1 - first_discount) * (1 - second_discount)) = single_discount := by
    sorry

end equivalent_discount_l64_64171


namespace notification_possible_l64_64620

-- Define the conditions
def side_length : ℝ := 2
def speed : ℝ := 3
def initial_time : ℝ := 12 -- noon
def arrival_time : ℝ := 19 -- 7 PM
def notification_time : ℝ := arrival_time - initial_time -- total available time for notification

-- Define the proof statement
theorem notification_possible :
  ∃ (partition : ℕ → ℝ) (steps : ℕ → ℝ), (∀ k, steps k * partition k < notification_time) ∧ 
  ∑' k, (steps k * partition k) ≤ 6 :=
by
  sorry

end notification_possible_l64_64620


namespace exists_nat_number_gt_1000_l64_64391

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_nat_number_gt_1000 (S : ℕ → ℕ) :
  (∀ n : ℕ, S (2^n) = sum_of_digits (2^n)) →
  ∃ n : ℕ, n > 1000 ∧ S (2^n) > S (2^(n + 1)) :=
by sorry

end exists_nat_number_gt_1000_l64_64391


namespace days_to_complete_work_l64_64483

theorem days_to_complete_work :
  ∀ (M B: ℝ) (D: ℝ),
    (M = 2 * B)
    → (13 * M + 24 * B) * 4 = (12 * M + 16 * B) * D
    → D = 5 :=
by
  intros M B D h1 h2
  sorry

end days_to_complete_work_l64_64483


namespace symm_diff_A_B_l64_64739

-- Define sets A and B
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | abs x < 2}

-- Define set difference
def set_diff (S T : Set ℤ) : Set ℤ := {x | x ∈ S ∧ x ∉ T}

-- Define symmetric difference
def symm_diff (S T : Set ℤ) : Set ℤ := (set_diff S T) ∪ (set_diff T S)

-- Define the expression we need to prove
theorem symm_diff_A_B : symm_diff A B = {-1, 0, 2} := by
  sorry

end symm_diff_A_B_l64_64739


namespace Raine_steps_to_school_l64_64167

-- Define Raine's conditions
variable (steps_total : ℕ) (days : ℕ) (round_trip_steps : ℕ)

-- Given conditions
def Raine_conditions := steps_total = 1500 ∧ days = 5 ∧ round_trip_steps = steps_total / days

-- Prove that the steps to school is 150 given Raine's conditions
theorem Raine_steps_to_school (h : Raine_conditions 1500 5 300) : (300 / 2) = 150 :=
by
  sorry

end Raine_steps_to_school_l64_64167


namespace smallest_append_digits_l64_64215

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l64_64215


namespace pascals_triangle_53_rows_l64_64475

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l64_64475


namespace total_time_equiv_l64_64358

-- Define the number of chairs
def chairs := 7

-- Define the number of tables
def tables := 3

-- Define the time spent on each piece of furniture in minutes
def time_per_piece := 4

-- Prove the total time taken to assemble all furniture
theorem total_time_equiv : chairs + tables = 10 ∧ 4 * 10 = 40 := by
  sorry

end total_time_equiv_l64_64358


namespace prime_divisor_property_l64_64505

-- Given conditions
variable (p k : ℕ)
variable (prime_p : Nat.Prime p)
variable (divisor_p : p ∣ (2 ^ (2 ^ k)) + 1)

-- The theorem we need to prove
theorem prime_divisor_property (p k : ℕ) (prime_p : Nat.Prime p) (divisor_p : p ∣ (2 ^ (2 ^ k)) + 1) : (2 ^ (k + 1)) ∣ (p - 1) := 
by 
  sorry

end prime_divisor_property_l64_64505


namespace problem_solution_l64_64825

theorem problem_solution
  (P Q R S : ℕ)
  (h1 : 2 * Q = P + R)
  (h2 : R * R = Q * S)
  (h3 : R = 4 * Q / 3) :
  P + Q + R + S = 171 :=
by sorry

end problem_solution_l64_64825


namespace binomial_coefficient_is_252_l64_64775

theorem binomial_coefficient_is_252 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_coefficient_is_252_l64_64775


namespace max_sum_of_abcd_l64_64305

noncomputable def abcd_product (a b c d : ℕ) : ℕ := a * b * c * d

theorem max_sum_of_abcd (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : abcd_product a b c d = 1995) : 
    a + b + c + d ≤ 142 :=
sorry

end max_sum_of_abcd_l64_64305


namespace minimum_groups_needed_l64_64551

theorem minimum_groups_needed :
  ∃ (g : ℕ), g = 5 ∧ ∀ n k : ℕ, n = 30 → k ≤ 7 → n / k = g :=
by
  sorry

end minimum_groups_needed_l64_64551


namespace lewis_weekly_earning_l64_64161

theorem lewis_weekly_earning
  (weeks : ℕ)
  (weekly_rent : ℤ)
  (total_savings : ℤ)
  (h1 : weeks = 1181)
  (h2 : weekly_rent = 216)
  (h3 : total_savings = 324775)
  : ∃ (E : ℤ), E = 49075 / 100 :=
by
  let E := 49075 / 100
  use E
  sorry -- The proof would go here

end lewis_weekly_earning_l64_64161


namespace chocolates_per_small_box_l64_64557

/-- A large box contains 19 small boxes and each small box contains a certain number of chocolate bars.
There are 475 chocolate bars in the large box. --/
def number_of_chocolate_bars_per_small_box : Prop :=
  ∃ x : ℕ, 475 = 19 * x ∧ x = 25

theorem chocolates_per_small_box : number_of_chocolate_bars_per_small_box :=
by
  sorry -- proof is skipped

end chocolates_per_small_box_l64_64557


namespace Alyssa_has_37_balloons_l64_64407

variable (Sandy_balloons : ℕ) (Sally_balloons : ℕ) (Total_balloons : ℕ)

-- Conditions
axiom Sandy_Condition : Sandy_balloons = 28
axiom Sally_Condition : Sally_balloons = 39
axiom Total_Condition : Total_balloons = 104

-- Definition of Alyssa's balloons
def Alyssa_balloons : ℕ := Total_balloons - (Sandy_balloons + Sally_balloons)

-- The proof statement 
theorem Alyssa_has_37_balloons 
: Alyssa_balloons Sandy_balloons Sally_balloons Total_balloons = 37 :=
by
  -- The proof body will be placed here, but we will leave it as a placeholder for now
  sorry

end Alyssa_has_37_balloons_l64_64407


namespace factor_tree_value_l64_64954

theorem factor_tree_value :
  let Q := 5 * 3
  let R := 11 * 2
  let Y := 2 * Q
  let Z := 7 * R
  let X := Y * Z
  X = 4620 :=
by
  sorry

end factor_tree_value_l64_64954


namespace find_constants_and_min_value_l64_64605

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x * Real.log x
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b * Real.log x + b
noncomputable def g (a b x : ℝ) := f a b x - Real.exp 1 * x^2

theorem find_constants_and_min_value :
  (∀ (a b : ℝ),
    -- Condition for the derivative at x = 1 and the given tangent line slope
    (f' a b 1 = 2 * Real.exp 1) ∧
    -- Condition for the function value at x = 1
    (f a b 1 = Real.exp 1) →
    -- Expected results for a and b
    (a = 1 ∧ b = Real.exp 1)) ∧

  -- Evaluating the minimum value of the function g(x)
  (∀ (x : ℝ), 0 < x →
    -- Given the minimum occurs at x = 1
    g 1 (Real.exp 1) 1 = 0 ∧
    (∀ (x : ℝ), 0 < x →
      (g 1 (Real.exp 1) x ≥ 0))) :=
sorry

end find_constants_and_min_value_l64_64605


namespace sum_of_coordinates_of_other_endpoint_l64_64163

theorem sum_of_coordinates_of_other_endpoint
  (x y : ℝ)
  (h1 : (1 + x) / 2 = 5)
  (h2 : (2 + y) / 2 = 6) :
  x + y = 19 :=
by
  sorry

end sum_of_coordinates_of_other_endpoint_l64_64163


namespace find_original_rabbits_l64_64832

theorem find_original_rabbits (R S : ℕ) (h1 : R + S = 50)
  (h2 : 4 * R + 8 * S = 2 * R + 16 * S) :
  R = 40 :=
sorry

end find_original_rabbits_l64_64832


namespace sqrt_inequality_l64_64995

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a ^ 2 / b) + Real.sqrt (b ^ 2 / a) ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end sqrt_inequality_l64_64995


namespace y_intercept_of_line_l64_64027

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end y_intercept_of_line_l64_64027


namespace problem_statement_l64_64643

theorem problem_statement (n : ℕ) (hn : n > 0) : (122 ^ n - 102 ^ n - 21 ^ n) % 2020 = 2019 :=
by
  sorry

end problem_statement_l64_64643


namespace solve_inequality_zero_solve_inequality_neg_solve_inequality_pos_l64_64998

variable (a x : ℝ)

def inequality (a x : ℝ) : Prop := (1 - a * x) ^ 2 < 1

theorem solve_inequality_zero : a = 0 → ¬∃ x, inequality a x := by
  sorry

theorem solve_inequality_neg (h : a < 0) : (∃ x, inequality a x) →
  ∀ x, inequality a x ↔ (a ≠ 0 ∧ (2 / a < x ∧ x < 0)) := by
  sorry

theorem solve_inequality_pos (h : a > 0) : (∃ x, inequality a x) →
  ∀ x, inequality a x ↔ (a ≠ 0 ∧ (0 < x ∧ x < 2 / a)) := by
  sorry

end solve_inequality_zero_solve_inequality_neg_solve_inequality_pos_l64_64998


namespace smallest_rel_prime_120_l64_64889

theorem smallest_rel_prime_120 : ∃ (x : ℕ), x > 1 ∧ Nat.gcd x 120 = 1 ∧ ∀ y, y > 1 ∧ Nat.gcd y 120 = 1 → x ≤ y :=
by
  use 7
  sorry

end smallest_rel_prime_120_l64_64889


namespace problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution_l64_64723

noncomputable def problem_1 : Int :=
  (-3) + 5 - (-3)

theorem problem_1_solution : problem_1 = 5 := by
  sorry

noncomputable def problem_2 : ℚ :=
  (-1/3 - 3/4 + 5/6) * (-24)

theorem problem_2_solution : problem_2 = 6 := by
  sorry

noncomputable def problem_3 : ℚ :=
  1 - (1/9) * (-1/2 - 2^2)

theorem problem_3_solution : problem_3 = 3/2 := by
  sorry

noncomputable def problem_4 : ℚ :=
  ((-1)^2023) * (18 - (-2) * 3) / (15 - 3^3)

theorem problem_4_solution : problem_4 = 2 := by
  sorry

end problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution_l64_64723


namespace sum_roots_eq_six_l64_64057

theorem sum_roots_eq_six : 
  ∀ x : ℝ, (x - 3) ^ 2 = 16 → (x - 3 = 4 ∨ x - 3 = -4) → (let x₁ := 3 + 4 in let x₂ := 3 - 4 in x₁ + x₂ = 6) := by
  sorry

end sum_roots_eq_six_l64_64057


namespace calculate_expression_l64_64417

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end calculate_expression_l64_64417


namespace pattern_equation_l64_64985

theorem pattern_equation (n : ℕ) (hn : n > 0) : n * (n + 2) + 1 = (n + 1) ^ 2 := 
by sorry

end pattern_equation_l64_64985


namespace genevieve_cherries_purchase_l64_64441

theorem genevieve_cherries_purchase (cherries_cost_per_kg: ℝ) (genevieve_money: ℝ) (extra_money_needed: ℝ) (total_kg: ℝ) : 
  cherries_cost_per_kg = 8 → 
  genevieve_money = 1600 →
  extra_money_needed = 400 →
  total_kg = (genevieve_money + extra_money_needed) / cherries_cost_per_kg →
  total_kg = 250 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end genevieve_cherries_purchase_l64_64441


namespace binomial_distributions_l64_64538

-- Define the conditions for each random variable
variables (n N M : ℕ) (p : ℝ)
variables (hMN : M < N) (hp : 0 ≤ p ∧ p ≤ 1)

-- Definitions of the random variables
def xi_A : ℕ → ProbabilityTheory.ProbabilitySpace ℕ := sorry -- Throws of a die
def xi_B : ℕ → ProbabilityTheory.ProbabilitySpace ℕ := sorry -- Shots to hit target
def xi_C : ℕ → ProbabilityTheory.ProbabilitySpace ℕ := sorry -- With replacement
def xi_D : ℕ → ProbabilityTheory.ProbabilitySpace ℕ := sorry -- Without replacement

-- Define the binomial distribution property
def is_binomial (xi : ℕ → ProbabilityTheory.ProbabilitySpace ℕ) (k : ℕ) (p : ℝ) :=
  ∃ n : ℕ, xi n = ProbabilityTheory.binomial n p

-- Theorem stating which variables follow the binomial distribution
theorem binomial_distributions :
  is_binomial xi_A n (1/3) ∧ is_binomial xi_C n (M/N) :=
by sorry

end binomial_distributions_l64_64538


namespace domain_of_h_l64_64734

noncomputable def h (x : ℝ) : ℝ := (x^3 - 2*x^2 + 4*x + 3) / (x^2 - 5*x + 6)

theorem domain_of_h :
  {x : ℝ | ∃ (y : ℝ), y = h x} = {x : ℝ | x < 2} ∪ {x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | x > 3} := 
sorry

end domain_of_h_l64_64734


namespace math_problem_l64_64424

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end math_problem_l64_64424


namespace smallest_integer_cube_ends_in_392_l64_64735

theorem smallest_integer_cube_ends_in_392 : ∃ n : ℕ, (n > 0) ∧ (n^3 % 1000 = 392) ∧ ∀ m : ℕ, (m > 0) ∧ (m^3 % 1000 = 392) → n ≤ m :=
by 
  sorry

end smallest_integer_cube_ends_in_392_l64_64735


namespace four_digit_number_l64_64403

-- Definitions of the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Statement of the theorem
theorem four_digit_number (x y : ℕ) (hx : is_two_digit x) (hy : is_two_digit y) :
    (100 * x + y) = 1000 * x + y := sorry

end four_digit_number_l64_64403


namespace triangle_BFD_ratio_l64_64322

theorem triangle_BFD_ratio (x : ℝ) : 
  let AF := 3 * x
  let FE := x
  let ED := x
  let DC := 3 * x
  let side_square := AF + FE
  let area_square := side_square^2
  let area_triangle_BFD := area_square - (1/2 * AF * side_square + 1/2 * side_square * FE + 1/2 * ED * DC)
  (area_triangle_BFD / area_square) = 7 / 16 := 
by
  sorry

end triangle_BFD_ratio_l64_64322


namespace son_present_age_l64_64859

-- Definitions
variables (S M : ℕ)
-- Conditions
def age_diff : Prop := M = S + 22
def future_age_condition : Prop := M + 2 = 2 * (S + 2)

-- Theorem statement with proof placeholder
theorem son_present_age (H1 : age_diff S M) (H2 : future_age_condition S M) : S = 20 :=
by sorry

end son_present_age_l64_64859


namespace sandwiches_ordered_l64_64007

-- Define the cost per sandwich
def cost_per_sandwich : ℝ := 5

-- Define the delivery fee
def delivery_fee : ℝ := 20

-- Define the tip percentage
def tip_percentage : ℝ := 0.10

-- Define the total amount received
def total_received : ℝ := 121

-- Define the equation representing the total amount received
def total_equation (x : ℝ) : Prop :=
  cost_per_sandwich * x + delivery_fee + (cost_per_sandwich * x + delivery_fee) * tip_percentage = total_received

-- Define the theorem that needs to be proved
theorem sandwiches_ordered (x : ℝ) : total_equation x ↔ x = 18 :=
sorry

end sandwiches_ordered_l64_64007


namespace symmetric_point_l64_64317

theorem symmetric_point (a b : ℝ) (h1 : a = 2) (h2 : 3 = -b) : (a + b) ^ 2023 = -1 := 
by
  sorry

end symmetric_point_l64_64317


namespace part1_part2_l64_64913

def f (a : ℝ) (x : ℝ) : ℝ := a * |x - 2| + x
def g (x : ℝ) : ℝ := |x - 2| - |2 * x - 3| + x

theorem part1 (a : ℝ) : (∀ x, f a x ≤ f a 2) ↔ a ≤ -1 :=
by sorry

theorem part2 (x : ℝ) : f 1 x < |2 * x - 3| ↔ x > 0.5 :=
by sorry

end part1_part2_l64_64913


namespace max_pN_value_l64_64788

noncomputable def max_probability_units_digit (N: ℕ) (q2 q5 q10: ℚ) : ℚ :=
  let qk (k : ℕ) := (Nat.floor (N / k) : ℚ) / N
  q10 * (2 - q10) + 2 * (q2 - q10) * (q5 - q10)

theorem max_pN_value : ∃ (a b : ℕ), (a.gcd b = 1) ∧ (∀ N q2 q5 q10, max_probability_units_digit N q2 q5 q10 ≤  27 / 100) ∧ (100 * 27 + 100 = 2800) :=
by
  sorry

end max_pN_value_l64_64788


namespace geometric_sequence_term_l64_64710

noncomputable def b_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 1 => Real.sin x ^ 2
  | 2 => Real.sin x * Real.cos x
  | 3 => Real.cos x ^ 2 / Real.sin x
  | n + 4 => (Real.cos x / Real.sin x) ^ n * Real.cos x ^ 3 / Real.sin x ^ 2
  | _ => 0 -- Placeholder to cover all case

theorem geometric_sequence_term (x : ℝ) :
  ∃ n, b_n n x = Real.cos x + Real.sin x ∧ n = 7 := by
  sorry

end geometric_sequence_term_l64_64710


namespace min_value_l64_64127

theorem min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) :
  ∃ x, x = a^2 + 1 / (b * (a - b)) ∧ x ≥ 4 :=
by
  sorry

end min_value_l64_64127


namespace remainder_9876543210_mod_101_l64_64681

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l64_64681


namespace geometric_sequence_a_10_l64_64621

noncomputable def geometric_sequence := ℕ → ℝ

def a_3 (a r : ℝ) := a * r^2 = 3
def a_5_equals_8a_7 (a r : ℝ) := a * r^4 = 8 * a * r^6

theorem geometric_sequence_a_10 (a r : ℝ) (seq : geometric_sequence) (h₁ : a_3 a r) (h₂ : a_5_equals_8a_7 a r) :
  seq 10 = a * r^9 := by
  sorry

end geometric_sequence_a_10_l64_64621


namespace inequality_cubed_l64_64293

theorem inequality_cubed (a b : ℝ) (h : a < b ∧ b < 0) : a^3 ≤ b^3 :=
sorry

end inequality_cubed_l64_64293


namespace swim_distance_l64_64565

theorem swim_distance (v d : ℝ) (c : ℝ := 2.5) :
  (8 = d / (v + c)) ∧ (8 = 24 / (v - c)) → d = 84 :=
by
  sorry

end swim_distance_l64_64565


namespace angle_BDC_correct_l64_64625

theorem angle_BDC_correct (A B C D : Type) 
  (angle_A : ℝ) (angle_B : ℝ) (angle_DBC : ℝ) : 
  angle_A = 60 ∧ angle_B = 70 ∧ angle_DBC = 40 → 
  ∃ angle_BDC : ℝ, angle_BDC = 100 := 
by
  intro h
  sorry

end angle_BDC_correct_l64_64625


namespace monotonicity_and_extrema_of_f_l64_64604

noncomputable def f (x : ℝ) : ℝ := 3 * x + 2

theorem monotonicity_and_extrema_of_f :
  (∀ (x_1 x_2 : ℝ), x_1 ∈ Set.Icc (-1 : ℝ) 2 → x_2 ∈ Set.Icc (-1 : ℝ) 2 → x_1 < x_2 → f x_1 < f x_2) ∧ 
  (f (-1) = -1) ∧ 
  (f 2 = 8) :=
by
  sorry

end monotonicity_and_extrema_of_f_l64_64604


namespace average_words_per_puzzle_l64_64721

-- Define the conditions
def uses_up_pencil_every_two_weeks : Prop := ∀ (days_used : ℕ), days_used = 14
def words_to_use_up_pencil : ℕ := 1050
def puzzles_completed_per_day : ℕ := 1

-- Problem statement: Prove the average number of words in each crossword puzzle
theorem average_words_per_puzzle :
  (words_to_use_up_pencil / 14 = 75) :=
by
  -- Definitions used directly from the conditions
  sorry

end average_words_per_puzzle_l64_64721


namespace pyramid_top_row_missing_number_l64_64281

theorem pyramid_top_row_missing_number (a b c d e f g : ℕ)
  (h₁ : b * c = 720)
  (h₂ : a * b = 240)
  (h₃ : c * d = 1440)
  (h₄ : c = 6)
  : a = 120 :=
by
  sorry

end pyramid_top_row_missing_number_l64_64281


namespace mean_of_five_numbers_l64_64829

theorem mean_of_five_numbers (x1 x2 x3 x4 x5 : ℚ) (h_sum : x1 + x2 + x3 + x4 + x5 = 1/3) : 
  (x1 + x2 + x3 + x4 + x5) / 5 = 1/15 :=
by 
  sorry

end mean_of_five_numbers_l64_64829


namespace number_of_kittens_l64_64265

-- Definitions for the given conditions.
def total_animals : ℕ := 77
def hamsters : ℕ := 15
def birds : ℕ := 30

-- The proof problem statement.
theorem number_of_kittens : total_animals - hamsters - birds = 32 := by
  sorry

end number_of_kittens_l64_64265


namespace fountain_pen_price_l64_64069

theorem fountain_pen_price
  (n_fpens : ℕ) (n_mpens : ℕ) (total_cost : ℕ) (avg_cost_mpens : ℝ)
  (hpens : n_fpens = 450) (mpens : n_mpens = 3750) 
  (htotal : total_cost = 11250) (havg_mpens : avg_cost_mpens = 2.25) : 
  (total_cost - n_mpens * avg_cost_mpens) / n_fpens = 6.25 :=
by
  sorry

end fountain_pen_price_l64_64069


namespace interest_rate_correct_l64_64315

-- Definitions based on the conditions
def interest_rate_doubles (r : ℝ) : ℝ := 70 / r

-- Given conditions
variables (r : ℝ) (initial_investment final_investment : ℝ) (years : ℝ)
hypothesis h1 : initial_investment = 5000
hypothesis h2 : final_investment = 20000
hypothesis h3 : years = 18
hypothesis h4 : final_investment = initial_investment * (2^2)

-- Statement to be proved
noncomputable def solve_interest_rate : ℝ :=
  70 / (years / 2)

theorem interest_rate_correct :
  solve_interest_rate = 7.78 :=
by
  -- omit the proof 
  sorry

end interest_rate_correct_l64_64315


namespace max_perimeter_of_polygons_l64_64674

noncomputable def largest_possible_perimeter (sides1 sides2 sides3 : Nat) (len : Nat) : Nat :=
  (sides1 + sides2 + sides3) * len

theorem max_perimeter_of_polygons
  (a b c : ℕ)
  (h1 : a % 2 = 0)
  (h2 : b % 2 = 0)
  (h3 : c % 2 = 0)
  (h4 : 180 * (a - 2) / a + 180 * (b - 2) / b + 180 * (c - 2) / c = 360)
  (h5 : ∃ (p : ℕ), ∃ q : ℕ, (a = p ∧ c = p ∧ a = q ∨ a = q ∧ b = p ∨ b = q ∧ c = p))
  : largest_possible_perimeter a b c 2 = 24 := 
sorry

end max_perimeter_of_polygons_l64_64674


namespace cost_of_country_cd_l64_64833

theorem cost_of_country_cd
  (cost_rock_cd : ℕ) (cost_pop_cd : ℕ) (cost_dance_cd : ℕ)
  (num_each : ℕ) (julia_has : ℕ) (julia_short : ℕ)
  (total_cost : ℕ) (total_other_cds : ℕ) (cost_country_cd : ℕ) :
  cost_rock_cd = 5 →
  cost_pop_cd = 10 →
  cost_dance_cd = 3 →
  num_each = 4 →
  julia_has = 75 →
  julia_short = 25 →
  total_cost = julia_has + julia_short →
  total_other_cds = num_each * cost_rock_cd + num_each * cost_pop_cd + num_each * cost_dance_cd →
  total_cost = total_other_cds + num_each * cost_country_cd →
  cost_country_cd = 7 :=
by
  intros cost_rock_cost_pop_cost_dance_num julia_diff 
         calc_total_total_other sub_total total_cds
  sorry

end cost_of_country_cd_l64_64833


namespace combination_eq_permutation_div_factorial_l64_64394

-- Step d): Lean 4 Statement

variables (n k : ℕ)

-- Define combination C_n^k is any k-element subset of an n-element set
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define permutation A_n^k is the number of ways to arrange k elements out of n elements
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Statement to prove: C_n^k = A_n^k / k!
theorem combination_eq_permutation_div_factorial :
  combination n k = permutation n k / (Nat.factorial k) :=
by
  sorry

end combination_eq_permutation_div_factorial_l64_64394


namespace ratio_of_fallen_cakes_is_one_half_l64_64868

noncomputable def ratio_fallen_to_total (total_cakes fallen_cakes pick_up destroyed_cakes : ℕ) :=
  fallen_cakes / total_cakes

theorem ratio_of_fallen_cakes_is_one_half :
  ∀ (total_cakes fallen_cakes pick_up destroyed_cakes : ℕ),
    total_cakes = 12 →
    pick_up = fallen_cakes / 2 →
    pick_up = destroyed_cakes →
    destroyed_cakes = 3 →
    ratio_fallen_to_total total_cakes fallen_cakes pick_up destroyed_cakes = 1 / 2 :=
by
  intros total_cakes fallen_cakes pick_up destroyed_cakes h1 h2 h3 h4
  rw [h1, h4, ratio_fallen_to_total]
  -- proof goes here
  sorry

end ratio_of_fallen_cakes_is_one_half_l64_64868


namespace y_intercept_of_line_l64_64042

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end y_intercept_of_line_l64_64042


namespace even_three_digit_numbers_l64_64930

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l64_64930


namespace equation_of_line_AB_l64_64074

noncomputable def circle_center : ℝ × ℝ := (1, 0)  -- center of the circle (x-1)^2 + y^2 = 1
noncomputable def circle_radius : ℝ := 1          -- radius of the circle (x-1)^2 + y^2 = 1
noncomputable def point_P : ℝ × ℝ := (3, 1)       -- point P(3,1)

theorem equation_of_line_AB :
  ∃ (AB : ℝ → ℝ → Prop),
    (∀ x y, AB x y ↔ (2 * x + y - 3 = 0)) := sorry

end equation_of_line_AB_l64_64074


namespace sum_of_roots_of_quadratic_l64_64050

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end sum_of_roots_of_quadratic_l64_64050


namespace cherries_purchase_l64_64439

theorem cherries_purchase (total_money : ℝ) (price_per_kg : ℝ) 
  (genevieve_money : ℝ) (shortage : ℝ) (clarice_money : ℝ) :
  genevieve_money = 1600 → shortage = 400 → clarice_money = 400 → price_per_kg = 8 →
  total_money = genevieve_money + shortage + clarice_money →
  total_money / price_per_kg = 250 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end cherries_purchase_l64_64439


namespace parallel_lines_eq_l64_64822

theorem parallel_lines_eq {a x y : ℝ} :
  (∀ x y : ℝ, x + a * y = 2 * a + 2) ∧ (∀ x y : ℝ, a * x + y = a + 1) →
  a = 1 :=
by
  sorry

end parallel_lines_eq_l64_64822


namespace remainder_of_9876543210_div_101_l64_64688

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l64_64688


namespace solution_set_of_f_x_gt_2_minimum_value_of_f_l64_64001

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem solution_set_of_f_x_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7} ∪ {x : ℝ | x > 5 / 3} :=
by 
  sorry

theorem minimum_value_of_f : ∃ x : ℝ, f x = -9 / 2 :=
by 
  sorry

end solution_set_of_f_x_gt_2_minimum_value_of_f_l64_64001


namespace farmer_land_l64_64507

noncomputable def farmer_land_example (A : ℝ) : Prop :=
  let cleared_land := 0.90 * A
  let barley_land := 0.70 * cleared_land
  let potatoes_land := 0.10 * cleared_land
  let corn_land := 0.10 * cleared_land
  let tomatoes_bell_peppers_land := 0.10 * cleared_land
  tomatoes_bell_peppers_land = 90 → A = 1000

theorem farmer_land (A : ℝ) (h_cleared_land : 0.90 * A = cleared_land)
  (h_barley_land : 0.70 * cleared_land = barley_land)
  (h_potatoes_land : 0.10 * cleared_land = potatoes_land)
  (h_corn_land : 0.10 * cleared_land = corn_land)
  (h_tomatoes_bell_peppers_land : 0.10 * cleared_land = 90) :
  A = 1000 :=
by
  sorry

end farmer_land_l64_64507


namespace append_digits_divisible_by_all_less_than_10_l64_64221

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l64_64221


namespace sum_of_roots_l64_64049

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end sum_of_roots_l64_64049


namespace gcd_g102_g103_eq_one_l64_64337

def g (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_g102_g103_eq_one : Nat.gcd (g 102).natAbs (g 103).natAbs = 1 := by
  sorry

end gcd_g102_g103_eq_one_l64_64337


namespace count_valid_even_numbers_with_sum_12_l64_64924

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l64_64924


namespace ratio_areas_l64_64076

theorem ratio_areas (H : ℝ) (L : ℝ) (r : ℝ) (A_rectangle : ℝ) (A_circle : ℝ) :
  H = 45 ∧ (L / H = 4 / 3) ∧ r = H / 2 ∧ A_rectangle = L * H ∧ A_circle = π * r^2 →
  (A_rectangle / A_circle = 17 / π) :=
by
  sorry

end ratio_areas_l64_64076


namespace candle_remaining_length_l64_64855

-- Define the initial length of the candle and the burn rate
def initial_length : ℝ := 20
def burn_rate : ℝ := 5

-- Define the remaining length function
def remaining_length (t : ℝ) : ℝ := initial_length - burn_rate * t

-- Prove the relationship between time and remaining length for the given range of time
theorem candle_remaining_length (t : ℝ) (ht: 0 ≤ t ∧ t ≤ 4) : remaining_length t = 20 - 5 * t :=
by
  dsimp [remaining_length]
  sorry

end candle_remaining_length_l64_64855


namespace number_of_people_l64_64177

def avg_weight_increase : ℝ := 2.5
def old_person_weight : ℝ := 45
def new_person_weight : ℝ := 65

theorem number_of_people (n : ℕ) 
  (h1 : avg_weight_increase = 2.5) 
  (h2 : old_person_weight = 45) 
  (h3 : new_person_weight = 65) :
  n = 8 :=
  sorry

end number_of_people_l64_64177


namespace max_value_expression_l64_64786

theorem max_value_expression (x y : ℝ) (h : x + y = 5) :
  ∃ (M : ℝ), (x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4) ≤ M ∧ 
                  M = 6084 / 17 :=
begin
  use 6084 / 17,
  split,
  { sorry }, -- proof of the inequality
  { refl }   -- proof of equality
end

end max_value_expression_l64_64786


namespace F_is_even_l64_64442

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (x^3 - 2*x) * f x

theorem F_is_even (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_nonzero : f 1 ≠ 0) :
  is_even_function (F f) :=
sorry

end F_is_even_l64_64442


namespace num_pairs_satisfying_inequality_l64_64454

theorem num_pairs_satisfying_inequality : 
  ∃ (s : Nat), s = 204 ∧ ∀ (m n : ℕ), m > 0 → n > 0 → m^2 + n < 50 → s = 204 :=
by
  sorry

end num_pairs_satisfying_inequality_l64_64454


namespace rabbit_parent_genotype_l64_64808

-- Define the types for alleles and genotypes
inductive Allele
| H : Allele -- Hairy allele, dominant
| h : Allele -- Hairy allele, recessive
| S : Allele -- Smooth allele, dominant
| s : Allele -- Smooth allele, recessive

structure RabbitGenotype where
  a1 : Allele
  a2 : Allele

-- Probability that the allele for hairy fur (H) occurs
def p_hairy_allele : ℝ := 0.1
-- Probability that the allele for smooth fur (S) occurs
def p_smooth_allele : ℝ := 1.0 - p_hairy_allele

-- Function to determine if a rabbit is hairy
def is_hairy (genotype : RabbitGenotype) : Prop :=
  (genotype.a1 = Allele.H) ∨ (genotype.a2 = Allele.H)

-- Mating resulted in all four offspring having hairy fur
def all_offspring_hairy (offspring : List RabbitGenotype) : Prop :=
  ∀ o ∈ offspring, is_hairy o

-- Statement of the proof problem
theorem rabbit_parent_genotype (offspring : List RabbitGenotype) (hf : offspring.length = 4) 
  (ha : all_offspring_hairy offspring) :
  ∃ (parent1 parent2 : RabbitGenotype), 
    (is_hairy parent1) ∧ 
    (¬ is_hairy parent2) ∧ 
    parent1 = { a1 := Allele.H, a2 := Allele.H } ∧ 
    parent2 = { a1 := Allele.S, a2 := Allele.h } :=
sorry

end rabbit_parent_genotype_l64_64808


namespace platform_and_train_length_equality_l64_64820

-- Definitions of the given conditions.
def speed_in_kmh : ℝ := 90
def speed_in_m_per_min : ℝ := (speed_in_kmh * 1000) / 60
def time_in_min : ℝ := 1
def length_of_train : ℝ := 750
def total_distance_covered : ℝ := speed_in_m_per_min * time_in_min

-- Assertion that length of platform is equal to length of train
theorem platform_and_train_length_equality : 
  total_distance_covered - length_of_train = length_of_train :=
by
  -- Placeholder for proof
  sorry

end platform_and_train_length_equality_l64_64820


namespace quiz_answer_key_count_l64_64386

theorem quiz_answer_key_count :
  ∃ n : ℕ, n = 480 ∧
  (∃ tf_count : ℕ, tf_count = 30 ∧
   (∃ mc_count : ℕ, mc_count = 16 ∧ 
    n = tf_count * mc_count)) :=
    sorry

end quiz_answer_key_count_l64_64386


namespace min_stamps_l64_64865

theorem min_stamps (x y : ℕ) (h : 5 * x + 7 * y = 47) : x + y ≥ 7 :=
by 
  have h₀ : ∃ x y : ℕ, 5 * x + 7 * y = 47 := sorry,
  have min_value := minstamps h₀,
  exact min_value

end min_stamps_l64_64865


namespace circle_properties_l64_64070

theorem circle_properties (D r C A : ℝ) (h1 : D = 15)
  (h2 : r = 7.5)
  (h3 : C = 15 * Real.pi)
  (h4 : A = 56.25 * Real.pi) :
  (9 ^ 2 + 12 ^ 2 = D ^ 2) ∧ (D = 2 * r) ∧ (C = Real.pi * D) ∧ (A = Real.pi * r ^ 2) :=
by
  sorry

end circle_properties_l64_64070


namespace max_length_third_side_l64_64172

open Real

theorem max_length_third_side (A B C : ℝ) (a b c : ℝ) 
  (h1 : cos (2 * A) + cos (2 * B) + cos (2 * C) = 1)
  (h2 : a = 9) 
  (h3 : b = 12)
  (h4 : a^2 + b^2 = c^2) : 
  c = 15 := 
sorry

end max_length_third_side_l64_64172


namespace find_three_digit_number_l64_64113

def digits_to_num (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem find_three_digit_number (a b c : ℕ) (h1 : 8 * a + 5 * b + c = 100) (h2 : a + b + c = 20) :
  digits_to_num a b c = 866 :=
by 
  sorry

end find_three_digit_number_l64_64113


namespace maple_trees_planted_plant_maple_trees_today_l64_64194

-- Define the initial number of maple trees
def initial_maple_trees : ℕ := 2

-- Define the number of maple trees the park will have after planting
def final_maple_trees : ℕ := 11

-- Define the number of popular trees, though it is irrelevant for the proof
def initial_popular_trees : ℕ := 5

-- The main statement to prove: number of maple trees planted today
theorem maple_trees_planted : ℕ :=
  final_maple_trees - initial_maple_trees

-- Prove that the number of maple trees planted today is 9
theorem plant_maple_trees_today :
  maple_trees_planted = 9 :=
by
  sorry

end maple_trees_planted_plant_maple_trees_today_l64_64194


namespace no_consecutive_positive_integers_have_sum_75_l64_64181

theorem no_consecutive_positive_integers_have_sum_75 :
  ∀ n a : ℕ, (n ≥ 2) → (a ≥ 1) → (n * (2 * a + n - 1) = 150) → False :=
by
  intros n a hn ha hsum
  sorry

end no_consecutive_positive_integers_have_sum_75_l64_64181


namespace sum_of_possible_values_l64_64501

theorem sum_of_possible_values (a b c d : ℝ) (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) : 
  (a - c) * (b - d) / ((c - d) * (d - a)) = 1 :=
by
  -- Solution omitted
  sorry

end sum_of_possible_values_l64_64501


namespace count_even_three_digit_numbers_l64_64938

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l64_64938


namespace factorization_sum_l64_64522

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℝ, (x + a) * (x + b) = x^2 + 13 * x + 40)
  (h2 : ∀ x : ℝ, (x - b) * (x - c) = x^2 - 19 * x + 88) :
  a + b + c = 24 := 
sorry

end factorization_sum_l64_64522


namespace pencil_cost_l64_64532

theorem pencil_cost 
  (x y : ℚ)
  (h1 : 3 * x + 2 * y = 165)
  (h2 : 4 * x + 7 * y = 303) :
  y = 19.155 := 
by
  sorry

end pencil_cost_l64_64532


namespace chandler_needs_to_sell_more_rolls_l64_64894

/-- Chandler's wrapping paper selling condition. -/
def chandler_needs_to_sell : ℕ := 12

def sold_to_grandmother : ℕ := 3
def sold_to_uncle : ℕ := 4
def sold_to_neighbor : ℕ := 3

def total_sold : ℕ := sold_to_grandmother + sold_to_uncle + sold_to_neighbor

theorem chandler_needs_to_sell_more_rolls : chandler_needs_to_sell - total_sold = 2 :=
by
  sorry

end chandler_needs_to_sell_more_rolls_l64_64894


namespace bird_families_difference_l64_64249

-- Define the conditions
def bird_families_to_africa : ℕ := 47
def bird_families_to_asia : ℕ := 94

-- The proof statement
theorem bird_families_difference : (bird_families_to_asia - bird_families_to_africa = 47) :=
by
  sorry

end bird_families_difference_l64_64249


namespace color_column_l64_64256

theorem color_column (n : ℕ) (color : ℕ) (board : ℕ → ℕ → ℕ) 
  (h_colors : ∀ i j, 1 ≤ board i j ∧ board i j ≤ n^2)
  (h_block : ∀ i j, (∀ k l : ℕ, k < n → l < n → ∃ c, ∀ a b : ℕ, k + a * n < n → l + b * n < n → board (i + k + a * n) (j + l + b * n) = c))
  (h_row : ∃ r, ∀ k, k < n → ∃ c, 1 ≤ c ∧ c ≤ n ∧ board r k = c) :
  ∃ c, (∀ j, 1 ≤ board c j ∧ board c j ≤ n) :=
sorry

end color_column_l64_64256


namespace tan_simplify_l64_64758

theorem tan_simplify (α : ℝ) (h : Real.tan α = 1 / 2) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = - 3 / 4 :=
by
  sorry

end tan_simplify_l64_64758


namespace q_join_after_days_l64_64543

noncomputable def workRate (totalWork : ℕ) (days : ℕ) : ℚ :=
  totalWork / days

theorem q_join_after_days (W : ℕ) (days_p : ℕ) (days_q : ℕ) (total_days : ℕ) (x : ℕ) :
  days_p = 80 ∧ days_q = 48 ∧ total_days = 35 ∧ 
  ((workRate W days_p) * x + (workRate W days_p + workRate W days_q) * (total_days - x) = W) 
  → x = 8 := sorry

end q_join_after_days_l64_64543


namespace cyclic_quadrilateral_angles_l64_64968

theorem cyclic_quadrilateral_angles (ABCD_cyclic : True) (P_interior : True)
  (x y z t : ℝ) (h1 : x + y + z + t = 360)
  (h2 : x + t = 180) :
  x = 180 - y - z :=
by
  sorry

end cyclic_quadrilateral_angles_l64_64968


namespace textbook_profit_l64_64083

theorem textbook_profit (cost_price selling_price : ℕ) (h1 : cost_price = 44) (h2 : selling_price = 55) :
  (selling_price - cost_price) = 11 := by
  sorry

end textbook_profit_l64_64083


namespace incorrect_statement_about_GIS_l64_64715

def statement_A := "GIS can provide information for geographic decision-making"
def statement_B := "GIS are computer systems specifically designed to process geographic spatial data"
def statement_C := "Urban management is one of the earliest and most effective fields of GIS application"
def statement_D := "GIS's main functions include data collection, data analysis, decision-making applications, etc."

def correct_answer := statement_B

theorem incorrect_statement_about_GIS:
  correct_answer = statement_B := 
sorry

end incorrect_statement_about_GIS_l64_64715


namespace coordinates_of_P_l64_64958

def P : Prod Int Int := (-1, 2)

theorem coordinates_of_P :
  P = (-1, 2) := 
  by
    -- The proof is omitted as per instructions
    sorry

end coordinates_of_P_l64_64958


namespace smallest_positive_integer_cube_ends_in_392_l64_64736

theorem smallest_positive_integer_cube_ends_in_392 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 392 ∧ ∀ m : ℕ, m > 0 ∧ m^3 % 1000 = 392 → n ≤ m :=
begin
  -- Placeholder for proof
  use 48,
  split,
  { exact dec_trivial }, -- 48 > 0
  split,
  { norm_num }, -- 48^3 % 1000 = 392
  { intros m h1 h2,
    -- We have to show 48 is the smallest such n
    sorry }
end

end smallest_positive_integer_cube_ends_in_392_l64_64736


namespace sum_fiftieth_powers_100_gon_l64_64021

noncomputable def sum_fiftieth_powers_all_sides_and_diagonals (n : ℕ) (R : ℝ) : ℝ := sorry
-- Define the sum of 50-th powers of all the sides and diagonals for a general n-gon inscribed in a circle of radius R

theorem sum_fiftieth_powers_100_gon (R : ℝ) : 
  sum_fiftieth_powers_all_sides_and_diagonals 100 R = sorry := sorry

end sum_fiftieth_powers_100_gon_l64_64021


namespace longest_segment_CD_l64_64969

theorem longest_segment_CD
  (ABD_angle : ℝ) (ADB_angle : ℝ) (BDC_angle : ℝ) (CBD_angle : ℝ)
  (angle_proof_ABD : ABD_angle = 50)
  (angle_proof_ADB : ADB_angle = 40)
  (angle_proof_BDC : BDC_angle = 35)
  (angle_proof_CBD : CBD_angle = 70) :
  true := 
by
  sorry

end longest_segment_CD_l64_64969


namespace compute_expression_l64_64289

theorem compute_expression (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end compute_expression_l64_64289


namespace expand_product_l64_64583

noncomputable def expand_poly (x : ℝ) : ℝ := (x + 3) * (x^2 + 2 * x + 4)

theorem expand_product (x : ℝ) : expand_poly x = x^3 + 5 * x^2 + 10 * x + 12 := 
by 
  -- This will be filled with the proof steps, but for now we use sorry.
  sorry

end expand_product_l64_64583


namespace smallest_digits_to_append_l64_64209

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l64_64209


namespace jonah_walked_8_miles_l64_64154

def speed : ℝ := 4
def time : ℝ := 2
def distance (s t : ℝ) : ℝ := s * t

theorem jonah_walked_8_miles : distance speed time = 8 := sorry

end jonah_walked_8_miles_l64_64154


namespace find_missing_number_l64_64603

theorem find_missing_number (n : ℝ) :
  (0.0088 * 4.5) / (0.05 * 0.1 * n) = 990 → n = 0.008 :=
by
  intro h
  sorry

end find_missing_number_l64_64603


namespace sum_of_roots_of_quadratic_l64_64055

theorem sum_of_roots_of_quadratic :
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16 in
  (∀ x, f x = 0 → x = 7 ∨ x = -1) →
  (let sum_of_roots := 7 + (-1) in sum_of_roots = 6) :=
by
  sorry

end sum_of_roots_of_quadratic_l64_64055


namespace percent_increase_stock_l64_64091

theorem percent_increase_stock (P_open P_close: ℝ) (h1: P_open = 30) (h2: P_close = 45):
  (P_close - P_open) / P_open * 100 = 50 :=
by
  sorry

end percent_increase_stock_l64_64091


namespace determine_a_range_l64_64173

variable (a : ℝ)

-- Define proposition p as a function
def p : Prop := ∀ x : ℝ, x^2 + x > a

-- Negation of Proposition q
def not_q : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 2 - a ≠ 0

-- The main theorem to be stated, proving the range of 'a'
theorem determine_a_range (h₁ : p a) (h₂ : not_q a) : -2 < a ∧ a < -1 / 4 := sorry

end determine_a_range_l64_64173


namespace quadratic_equation_roots_l64_64452

theorem quadratic_equation_roots (a b c : ℝ) : 
  (b ^ 6 > 4 * (a ^ 3) * (c ^ 3)) → (b ^ 10 > 4 * (a ^ 5) * (c ^ 5)) :=
by
  sorry

end quadratic_equation_roots_l64_64452


namespace smallest_digits_to_append_l64_64206

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l64_64206


namespace sum_a_b_l64_64759

theorem sum_a_b (a b : ℝ) (h₁ : 2 = a + b) (h₂ : 6 = a + b / 9) : a + b = 2 :=
by
  sorry

end sum_a_b_l64_64759


namespace quadratic_nonnegative_l64_64117

theorem quadratic_nonnegative (x y : ℝ) : x^2 + x * y + y^2 ≥ 0 :=
by sorry

end quadratic_nonnegative_l64_64117


namespace triangle_area_base_6_height_8_l64_64365

noncomputable def triangle_area (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem triangle_area_base_6_height_8 : triangle_area 6 8 = 24 := by
  sorry

end triangle_area_base_6_height_8_l64_64365


namespace diff_of_roots_l64_64542

-- Define the quadratic equation and its coefficients
def quadratic_eq (z : ℝ) : ℝ := 2 * z^2 + 5 * z - 12

-- Define the discriminant function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the roots of the quadratic equation using the quadratic formula
noncomputable def larger_root (a b c : ℝ) : ℝ := (-b + Real.sqrt (discriminant a b c)) / (2 * a)
noncomputable def smaller_root (a b c : ℝ) : ℝ := (-b - Real.sqrt (discriminant a b c)) / (2 * a)

-- Define the proof statement
theorem diff_of_roots : 
  ∃ (a b c z1 z2 : ℝ), 
    a = 2 ∧ b = 5 ∧ c = -12 ∧
    quadratic_eq z1 = 0 ∧ quadratic_eq z2 = 0 ∧
    z1 = smaller_root a b c ∧ z2 = larger_root a b c ∧
    z2 - z1 = 5.5 := 
by 
  sorry

end diff_of_roots_l64_64542


namespace find_smallest_sphere_radius_squared_l64_64777

noncomputable def smallest_sphere_radius_squared
  (AB CD AD BC : ℝ) (angle_ABC : ℝ) (radius_AC_squared : ℝ) : ℝ :=
if AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120 then radius_AC_squared else 0

theorem find_smallest_sphere_radius_squared
  (AB CD AD BC : ℝ) (angle_ABC : ℝ) (radius_AC_squared : ℝ) :
  (AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120) →
  radius_AC_squared = 49 :=
by
  intros h
  have h_ABCD : AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120 := h
  sorry -- The proof steps would be filled in here

end find_smallest_sphere_radius_squared_l64_64777


namespace number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l64_64204

/- Definitions for each number's expression using five eights -/
def number1 : Int := (8 / 8) ^ (8 / 8) * (8 / 8)
def number2 : Int := 8 / 8 + 8 / 8
def number3 : Int := (8 + 8 + 8) / 8
def number4 : Int := 8 / 8 + 8 / 8 + 8 / 8 + 8 / 8
def number5 : Int := (8 * 8 - 8) / 8 + 8 / 8

/- Theorem statements to be proven -/
theorem number1_is_1 : number1 = 1 := by
  sorry

theorem number2_is_2 : number2 = 2 := by
  sorry

theorem number3_is_3 : number3 = 3 := by
  sorry

theorem number4_is_4 : number4 = 4 := by
  sorry

theorem number5_is_5 : number5 = 5 := by
  sorry

end number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l64_64204


namespace remainder_when_sum_divided_by_30_l64_64699

theorem remainder_when_sum_divided_by_30 (x y z : ℕ) (hx : x % 30 = 14) (hy : y % 30 = 5) (hz : z % 30 = 21) :
  (x + y + z) % 30 = 10 :=
by
  sorry

end remainder_when_sum_divided_by_30_l64_64699


namespace incident_reflected_eqs_l64_64129

theorem incident_reflected_eqs {x y : ℝ} :
  (∃ (A B : ℝ × ℝ), A = (2, 3) ∧ B = (1, 1) ∧ 
   (∀ (P : ℝ × ℝ), (P = A ∨ P = B → (P.1 + P.2 + 1 = 0) → false)) ∧
   (∃ (line_inc line_ref : ℝ × ℝ × ℝ),
     line_inc = (5, -4, 2) ∧
     line_ref = (4, -5, 1))) :=
sorry

end incident_reflected_eqs_l64_64129


namespace people_in_group_l64_64178

-- Define the conditions as Lean definitions
def avg_weight_increase := 2.5
def replaced_weight := 45
def new_weight := 65
def weight_difference := new_weight - replaced_weight -- 20 kg

-- State the problem as a Lean theorem
theorem people_in_group :
  ∀ n : ℕ, avg_weight_increase * n = weight_difference → n = 8 :=
by
  intros n h
  sorry

end people_in_group_l64_64178


namespace smallest_digits_to_append_l64_64226

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l64_64226


namespace tori_current_height_l64_64533

theorem tori_current_height :
  let original_height := 4.4
  let growth := 2.86
  original_height + growth = 7.26 := 
by
  sorry

end tori_current_height_l64_64533


namespace edward_money_left_l64_64429

noncomputable def toy_cost : ℝ := 0.95

noncomputable def toy_quantity : ℕ := 4

noncomputable def toy_discount : ℝ := 0.15

noncomputable def race_track_cost : ℝ := 6.00

noncomputable def race_track_tax : ℝ := 0.08

noncomputable def initial_amount : ℝ := 17.80

noncomputable def total_toy_cost_before_discount : ℝ := toy_quantity * toy_cost

noncomputable def discount_amount : ℝ := toy_discount * total_toy_cost_before_discount

noncomputable def total_toy_cost_after_discount : ℝ := total_toy_cost_before_discount - discount_amount

noncomputable def race_track_tax_amount : ℝ := race_track_tax * race_track_cost

noncomputable def total_race_track_cost_after_tax : ℝ := race_track_cost + race_track_tax_amount

noncomputable def total_amount_spent : ℝ := total_toy_cost_after_discount + total_race_track_cost_after_tax

noncomputable def money_left : ℝ := initial_amount - total_amount_spent

theorem edward_money_left : money_left = 8.09 := by
  -- proof goes here
  sorry

end edward_money_left_l64_64429


namespace remainder_of_9876543210_div_101_l64_64685

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l64_64685


namespace polygon_of_T_has_4_sides_l64_64334

def T (b : ℝ) (x y : ℝ) : Prop :=
  b ≤ x ∧ x ≤ 4 * b ∧
  b ≤ y ∧ y ≤ 4 * b ∧
  x + y ≥ 3 * b ∧
  x + 2 * b ≥ 2 * y ∧
  2 * y ≥ x + b

noncomputable def sides_of_T (b : ℝ) : ℕ :=
  if b > 0 then 4 else 0

theorem polygon_of_T_has_4_sides (b : ℝ) (hb : b > 0) : sides_of_T b = 4 := by
  sorry

end polygon_of_T_has_4_sides_l64_64334


namespace garden_length_l64_64267

noncomputable def length_of_garden : ℝ := 300

theorem garden_length (P : ℝ) (b : ℝ) (A : ℝ) 
  (h₁ : P = 800) (h₂ : b = 100) (h₃ : A = 10000) : length_of_garden = 300 := 
by 
  sorry

end garden_length_l64_64267


namespace find_y_intercept_of_second_parabola_l64_64659

theorem find_y_intercept_of_second_parabola :
  ∃ D : ℝ × ℝ, D = (0, 9) ∧ 
    (∃ A : ℝ × ℝ, A = (10, 4) ∧ 
     ∃ B : ℝ × ℝ, B = (6, 0) ∧ 
     (∀ x y : ℝ, y = (-1/4) * x ^ 2 + 5 * x - 21 → A = (10, 4)) ∧ 
     (∀ x y : ℝ, y = (1/4) * (x - B.1) ^ 2 + B.2 ∧ y = 4 ∧ B = (6, 0) → A = (10, 4))) :=
  sorry

end find_y_intercept_of_second_parabola_l64_64659


namespace correct_calculation_is_c_l64_64537

theorem correct_calculation_is_c (a b : ℕ) :
  (2 * a ^ 2 * b) ^ 3 = 8 * a ^ 6 * b ^ 3 := 
sorry

end correct_calculation_is_c_l64_64537


namespace solid_color_marble_percentage_l64_64804

theorem solid_color_marble_percentage (solid striped dotted swirl red blue green yellow purple : ℝ)
  (h_solid: solid = 0.7) (h_striped: striped = 0.1) (h_dotted: dotted = 0.1) (h_swirl: swirl = 0.1)
  (h_red: red = 0.25) (h_blue: blue = 0.25) (h_green: green = 0.2) (h_yellow: yellow = 0.15) (h_purple: purple = 0.15) :
  solid * (red + blue + green) * 100 = 49 :=
by
  sorry

end solid_color_marble_percentage_l64_64804


namespace pascal_triangle_contains_53_only_once_l64_64460

theorem pascal_triangle_contains_53_only_once (n : ℕ) (k : ℕ) (h_prime : Nat.prime 53) :
  (n = 53 ∧ (k = 1 ∨ k = 52) ∨ 
   ∀ m < 53, Π l, Nat.binomial m l ≠ 53) ∧ 
  (n > 53 → (k = 0 ∨ k = n ∨ Π a b, a * 53 ≠ b * Nat.factorial (n - k + 1))) :=
sorry

end pascal_triangle_contains_53_only_once_l64_64460


namespace sum_areas_frequency_distribution_histogram_l64_64488

theorem sum_areas_frequency_distribution_histogram :
  ∀ (rectangles : List ℝ), (∀ r ∈ rectangles, 0 ≤ r ∧ r ≤ 1) → rectangles.sum = 1 := 
  by
    intro rectangles h
    sorry

end sum_areas_frequency_distribution_histogram_l64_64488


namespace smallest_sum_of_consecutive_integers_gt_420_l64_64246

theorem smallest_sum_of_consecutive_integers_gt_420 : 
  ∃ n : ℕ, (n * (n + 1) > 420) ∧ (n + (n + 1) = 43) := sorry

end smallest_sum_of_consecutive_integers_gt_420_l64_64246


namespace inequality_solution_set_l64_64670

theorem inequality_solution_set {x : ℝ} : 2 * x^2 - x - 1 > 0 ↔ (x < -1 / 2 ∨ x > 1) := 
sorry

end inequality_solution_set_l64_64670


namespace omitted_angle_of_convex_polygon_l64_64619

theorem omitted_angle_of_convex_polygon (calculated_sum : ℕ) (omitted_angle : ℕ)
    (h₁ : calculated_sum = 2583) (h₂ : omitted_angle = 2700 - 2583) :
    omitted_angle = 117 :=
by
  sorry

end omitted_angle_of_convex_polygon_l64_64619


namespace smallest_append_digits_l64_64216

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l64_64216


namespace factorization_sum_l64_64523

theorem factorization_sum :
  ∃ a b c : ℤ, (∀ x : ℝ, (x^2 + 20 * x + 96 = (x + a) * (x + b)) ∧
                      (x^2 + 18 * x + 81 = (x - b) * (x + c))) →
              (a + b + c = 30) :=
by
  sorry

end factorization_sum_l64_64523


namespace meal_combinations_correct_l64_64280

-- Define the given conditions
def number_of_entrees : Nat := 4
def number_of_drinks : Nat := 4
def number_of_desserts : Nat := 2

-- Define the total number of meal combinations to prove
def total_meal_combinations : Nat := number_of_entrees * number_of_drinks * number_of_desserts

-- The theorem we want to prove
theorem meal_combinations_correct : total_meal_combinations = 32 := 
by 
  sorry

end meal_combinations_correct_l64_64280


namespace pascal_triangle_contains_53_l64_64469

theorem pascal_triangle_contains_53:
  ∃! n, ∃ k, (n ≥ 0) ∧ (k ≥ 0) ∧ (binom n k = 53) := 
sorry

end pascal_triangle_contains_53_l64_64469


namespace smallest_positive_n_l64_64838

theorem smallest_positive_n : ∃ n : ℕ, 3 * n ≡ 8 [MOD 26] ∧ n = 20 :=
by 
  use 20
  simp
  sorry

end smallest_positive_n_l64_64838


namespace count_valid_even_numbers_with_sum_12_l64_64923

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l64_64923


namespace rectangle_sides_l64_64398

theorem rectangle_sides :
  ∀ (x : ℝ), 
    (3 * x = 8) ∧ (8 / 3 * 3 = 8) →
    ((2 * (3 * x + x) = 3 * x^2) ∧ (2 * (3 * (8 / 3) + (8 / 3)) = 3 * (8 / 3)^2) →
    x = 8 / 3
      ∧ 3 * x = 8) := 
by
  sorry

end rectangle_sides_l64_64398


namespace matrix_self_inverse_pairs_l64_64427

theorem matrix_self_inverse_pairs :
  ∃ p : Finset (ℝ × ℝ), (∀ a d, (a, d) ∈ p ↔ (∃ (m : Matrix (Fin 2) (Fin 2) ℝ), 
    m = !![a, 4; -9, d] ∧ m * m = 1)) ∧ p.card = 2 :=
by {
  sorry
}

end matrix_self_inverse_pairs_l64_64427


namespace quadratic_function_range_l64_64022

def range_of_quadratic_function : Set ℝ :=
  {y : ℝ | y ≥ 2}

theorem quadratic_function_range :
  ∀ x : ℝ, (∃ y : ℝ, y = x^2 - 4*x + 6 ∧ y ∈ range_of_quadratic_function) :=
by
  sorry

end quadratic_function_range_l64_64022


namespace planA_equals_planB_at_3_l64_64073

def planA_charge_for_first_9_minutes : ℝ := 0.24
def planA_charge (X: ℝ) (minutes: ℕ) : ℝ := if minutes <= 9 then X else X + 0.06 * (minutes - 9)
def planB_charge (minutes: ℕ) : ℝ := 0.08 * minutes

theorem planA_equals_planB_at_3 : planA_charge planA_charge_for_first_9_minutes 3 = planB_charge 3 :=
by sorry

end planA_equals_planB_at_3_l64_64073


namespace circumcenter_ABD_AC_l64_64156

open EuclideanGeometry

variables {A B C D M O P : Point}

theorem circumcenter_ABD_AC
  (h_trapezoid : Trapezoid A B C D)
  (O_circumABC : Cir_ring O A B C)
  (O_BD : Lies_on O (Line B D)) :
  ∃ Q, Cir_ring Q A B D ∧ Lies_on Q (Line A C) :=
by 
  sorry

end circumcenter_ABD_AC_l64_64156


namespace calculate_expression_l64_64419

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end calculate_expression_l64_64419


namespace constant_is_5_variables_are_n_and_S_l64_64489

-- Define the conditions
def cost_per_box : ℕ := 5
def total_cost (n : ℕ) : ℕ := n * cost_per_box

-- Define the statement to be proved
-- constant is 5
theorem constant_is_5 : cost_per_box = 5 := 
by sorry

-- variables are n and S, where S is total_cost n
theorem variables_are_n_and_S (n : ℕ) : 
    ∃ S : ℕ, S = total_cost n :=
by sorry

end constant_is_5_variables_are_n_and_S_l64_64489


namespace y_intercept_3x_minus_4y_eq_12_l64_64038

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end y_intercept_3x_minus_4y_eq_12_l64_64038


namespace sum_of_non_solutions_l64_64978

theorem sum_of_non_solutions (A B C : ℝ) :
  (∀ x : ℝ, (x ≠ -C ∧ x ≠ -10) → (x + B) * (A * x + 40) / ((x + C) * (x + 10)) = 2) →
  (A = 2 ∧ B = 10 ∧ C = 20) →
  (-10 + -20 = -30) :=
by sorry

end sum_of_non_solutions_l64_64978


namespace no_real_solution_3x2_plus_9x_le_neg12_l64_64740

/-- There are no real values of x such that 3x^2 + 9x ≤ -12. -/
theorem no_real_solution_3x2_plus_9x_le_neg12 (x : ℝ) : ¬(3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end no_real_solution_3x2_plus_9x_le_neg12_l64_64740


namespace mixed_fractions_calculation_l64_64425

theorem mixed_fractions_calculation :
  2017 + (2016 / 2017) / (2019 + (1 / 2016)) + (1 / 2017) = 1 :=
by
  sorry

end mixed_fractions_calculation_l64_64425


namespace living_room_area_is_60_l64_64854

-- Define the conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width
def coverage_fraction : ℝ := 0.60

-- Define the target area of the living room floor
def target_living_room_area (A : ℝ) : Prop :=
  coverage_fraction * A = carpet_area

-- State the Theorem
theorem living_room_area_is_60 (A : ℝ) (h : target_living_room_area A) : A = 60 := by
  -- Proof omitted
  sorry

end living_room_area_is_60_l64_64854


namespace smaller_number_l64_64848

theorem smaller_number (x y : ℕ) (h1 : x * y = 323) (h2 : x - y = 2) : y = 17 :=
sorry

end smaller_number_l64_64848


namespace number_divisible_l64_64233

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l64_64233


namespace lcm_4_6_9_l64_64111

/-- The least common multiple (LCM) of 4, 6, and 9 is 36 -/
theorem lcm_4_6_9 : Nat.lcm (Nat.lcm 4 6) 9 = 36 :=
by
  -- sorry replaces the actual proof steps
  sorry

end lcm_4_6_9_l64_64111


namespace calculate_expression_l64_64414

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end calculate_expression_l64_64414


namespace cricket_bat_cost_price_l64_64268

theorem cricket_bat_cost_price (CP_A : ℝ) (SP_B : ℝ) (SP_C : ℝ) (h1 : SP_B = CP_A * 1.20) (h2 : SP_C = SP_B * 1.25) (h3 : SP_C = 222) : CP_A = 148 := 
by
  sorry

end cricket_bat_cost_price_l64_64268


namespace cinema_meeting_day_l64_64628

-- Define the cycles for Kolya, Seryozha, and Vanya.
def kolya_cycle : ℕ := 4
def seryozha_cycle : ℕ := 5
def vanya_cycle : ℕ := 6

-- The problem statement requiring proof.
theorem cinema_meeting_day : ∃ n : ℕ, n > 0 ∧ n % kolya_cycle = 0 ∧ n % seryozha_cycle = 0 ∧ n % vanya_cycle = 0 ∧ n = 60 := 
  sorry

end cinema_meeting_day_l64_64628


namespace pascal_triangle_contains_53_only_once_l64_64458

theorem pascal_triangle_contains_53_only_once (n : ℕ) (k : ℕ) (h_prime : Nat.prime 53) :
  (n = 53 ∧ (k = 1 ∨ k = 52) ∨ 
   ∀ m < 53, Π l, Nat.binomial m l ≠ 53) ∧ 
  (n > 53 → (k = 0 ∨ k = n ∨ Π a b, a * 53 ≠ b * Nat.factorial (n - k + 1))) :=
sorry

end pascal_triangle_contains_53_only_once_l64_64458


namespace engineering_department_men_l64_64963

theorem engineering_department_men (total_students men_percentage women_count : ℕ) (h_percentage : men_percentage = 70) (h_women : women_count = 180) (h_total : total_students = (women_count * 100) / (100 - men_percentage)) : 
  (total_students * men_percentage / 100) = 420 :=
by
  sorry

end engineering_department_men_l64_64963


namespace y_value_for_equations_l64_64099

theorem y_value_for_equations (x y : ℝ) (h1 : x^2 + y^2 = 25) (h2 : x^2 + y = 10) :
  y = (1 - Real.sqrt 61) / 2 := by
  sorry

end y_value_for_equations_l64_64099


namespace trajectory_of_point_P_l64_64750

theorem trajectory_of_point_P :
  ∀ (x y : ℝ), 
  (∀ (m n : ℝ), n = 2 * m - 4 → (1 - m, -n) = (x - 1, y)) → 
  y = 2 * x :=
by
  sorry

end trajectory_of_point_P_l64_64750


namespace y_intercept_of_line_l64_64033

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end y_intercept_of_line_l64_64033


namespace complex_value_l64_64790

open Complex

theorem complex_value (z : ℂ)
  (h : 15 * normSq z = 3 * normSq (z + 3) + normSq (z^2 + 4) + 25) :
  z + (8 / z) = -4 :=
sorry

end complex_value_l64_64790


namespace permutation_count_of_word_l64_64310

-- Define the setup with the conditions
def letters : Finset (Char) := {'B', 'B', 'A', 'A', 'A', 'N', 'N'}

def count_B : ℕ := 2
def count_A : ℕ := 3
def count_N : ℕ := 2

theorem permutation_count_of_word :
  (∏ i in letters, nat.factorial (letters.count i)) = 2 * 2 * 2 → 
  finset.card (letters.permute (list.perm)) = 2520 := 
by
  intros h
  sorry

end permutation_count_of_word_l64_64310


namespace vector_subtraction_result_l64_64292

-- definition of vectors as pairs of integers
def OA : ℝ × ℝ := (1, -2)
def OB : ℝ × ℝ := (-3, 1)

-- definition of vector subtraction for pairs of reals
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- definition of the vector AB as the subtraction of OB and OA
def AB : ℝ × ℝ := vector_sub OB OA

-- statement to assert the expected result
theorem vector_subtraction_result : AB = (-4, 3) :=
by
  -- this is where the proof would go, but we use sorry to skip it
  sorry

end vector_subtraction_result_l64_64292


namespace silvia_saves_50_l64_64170

variables
  (price : ℕ := 1000) -- Suggested retail price of the guitar
  (gc_discount : ℕ := 15) -- Guitar Center discount in percentage
  (gc_shipping : ℕ := 100) -- Guitar Center shipping fee
  (sw_discount : ℕ := 10) -- Sweetwater discount in percentage
  (sw_shipping : ℕ := 0) -- Sweetwater shipping fee

def guitar_price_after_discount (price : ℕ) (discount : ℕ) : ℕ := 
  price - (price * discount / 100)

def total_cost (price : ℕ) (discount : ℕ) (shipping : ℕ) : ℕ :=
  (guitar_price_after_discount price discount) + shipping

def savings (price : ℕ) (gc_discount : ℕ) (gc_shipping : ℕ) (sw_discount : ℕ) (sw_shipping : ℕ) : ℕ :=
  total_cost price gc_discount gc_shipping - total_cost price sw_discount sw_shipping

theorem silvia_saves_50 (price : ℕ) (gc_discount : ℕ) (gc_shipping : ℕ) (sw_discount : ℕ) (sw_shipping : ℕ) :
  savings price gc_discount gc_shipping sw_discount sw_shipping = 50 := by
begin
  sorry
end

end silvia_saves_50_l64_64170


namespace correct_mean_251_l64_64847

theorem correct_mean_251
  (n : ℕ) (incorrect_mean : ℕ) (wrong_val : ℕ) (correct_val : ℕ)
  (h1 : n = 30) (h2 : incorrect_mean = 250) (h3 : wrong_val = 135) (h4 : correct_val = 165) :
  ((incorrect_mean * n + (correct_val - wrong_val)) / n) = 251 :=
by
  sorry

end correct_mean_251_l64_64847


namespace map_distance_l64_64508

variable (map_distance_km : ℚ) (map_distance_inches : ℚ) (actual_distance_km: ℚ)

theorem map_distance (h1 : actual_distance_km = 136)
                     (h2 : map_distance_inches = 42)
                     (h3 : map_distance_km = 18.307692307692307) :
  (actual_distance_km * map_distance_inches / map_distance_km = 312) :=
by sorry

end map_distance_l64_64508


namespace intersecting_diagonals_probability_l64_64769

variable (n : ℕ) (h : n > 0)

theorem intersecting_diagonals_probability (h : n > 0) :
  let V := 2 * n + 1 in
  let total_diagonals := (V * (V - 3)) / 2 in
  let pairs_of_diagonals := (total_diagonals * (total_diagonals - 1)) / 2 in
  let intersecting_pairs := ((2 * n + 1) * n * (2 * n - 1) * (n - 1)) / 24 in
  (intersecting_pairs.toRat / pairs_of_diagonals.toRat) = (n * (2 * n - 1)).toRat / (3 * (2 * n^2 - n - 2).toRat) := 
sorry

end intersecting_diagonals_probability_l64_64769


namespace parabola_focus_directrix_distance_l64_64652

theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y = (1 / 4) * x^2 → 
  (∃ p : ℝ, p = 2 ∧ x^2 = 4 * p * y) →
  ∃ d : ℝ, d = 2 :=
by
  sorry

end parabola_focus_directrix_distance_l64_64652


namespace proposition_p_is_false_iff_l64_64133

def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 3)

def p (a : ℝ) : Prop := ∃ x : ℝ, f x < a

theorem proposition_p_is_false_iff (a : ℝ) : (¬p a) ↔ (a < 5) :=
by sorry

end proposition_p_is_false_iff_l64_64133


namespace smallest_digits_to_append_l64_64235

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l64_64235


namespace smallest_digits_to_append_l64_64243

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l64_64243


namespace find_coordinates_A_l64_64906

-- Define the point A
structure Point where
  x : ℝ
  y : ℝ

def PointA (a : ℝ) : Point :=
  { x := 3 * a + 2, y := 2 * a - 4 }

-- Define the conditions
def condition1 (a : ℝ) := (PointA a).y = 4

def condition2 (a : ℝ) := |(PointA a).x| = |(PointA a).y|

-- The coordinates solutions to be proven
def valid_coordinates (p : Point) : Prop :=
  p = { x := 14, y := 4 } ∨
  p = { x := -16, y := -16 } ∨
  p = { x := 3.2, y := -3.2 }

-- Main theorem to prove
theorem find_coordinates_A (a : ℝ) :
  (condition1 a ∨ condition2 a) → valid_coordinates (PointA a) :=
by
  sorry

end find_coordinates_A_l64_64906


namespace quadratic_roots_real_and_equal_l64_64737

theorem quadratic_roots_real_and_equal (m : ℤ) :
  (∀ x : ℝ, 3 * x^2 + (2 - m) * x + 12 = 0 →
   (∃ r, x = r ∧ 3 * r^2 + (2 - m) * r + 12 = 0)) →
   (m = -10 ∨ m = 14) :=
sorry

end quadratic_roots_real_and_equal_l64_64737


namespace polar_to_rectangular_l64_64498

theorem polar_to_rectangular : 
  ∀ (r θ : ℝ), r = 2 ∧ θ = 2 * Real.pi / 3 → 
  (r * Real.cos θ, r * Real.sin θ) = (-1, Real.sqrt 3) := by
  sorry

end polar_to_rectangular_l64_64498


namespace total_students_in_lunchroom_l64_64966

theorem total_students_in_lunchroom :
  (34 * 6) + 15 = 219 :=
by
  sorry

end total_students_in_lunchroom_l64_64966


namespace impossible_equal_sums_3x3_l64_64492

theorem impossible_equal_sums_3x3 (a b c d e f g h i : ℕ) :
  a + b + c = 13 ∨ a + b + c = 14 ∨ a + b + c = 15 ∨ a + b + c = 16 ∨ a + b + c = 17 ∨ a + b + c = 18 ∨ a + b + c = 19 ∨ a + b + c = 20 →
  (a + d + g) = 13 ∨ (a + d + g) = 14 ∨ (a + d + g) = 15 ∨ (a + d + g) = 16 ∨ (a + d + g) = 17 ∨ (a + d + g) = 18 ∨ (a + d + g) = 19 ∨ (a + d + g) = 20 →
  (a + e + i) = 13 ∨ (a + e + i) = 14 ∨ (a + e + i) = 15 ∨ (a + e + i) = 16 ∨ (a + e + i) = 17 ∨ (a + e + i) = 18 ∨ (a + e + i) = 19 ∨ (a + e + i) = 20 →
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧ 1 ≤ f ∧ f ≤ 9 ∧ 1 ≤ g ∧ g ≤ 9 ∧ 1 ≤ h ∧ h ≤ 9 ∧ 1 ≤ i ∧ i ≤ 9 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ g ≠ h ∧ g ≠ i ∧ h ≠ i →
  false :=
sorry

end impossible_equal_sums_3x3_l64_64492


namespace zoo_total_animals_l64_64277

theorem zoo_total_animals (penguins polar_bears : ℕ)
  (h1 : penguins = 21)
  (h2 : polar_bears = 2 * penguins) :
  penguins + polar_bears = 63 := by
   sorry

end zoo_total_animals_l64_64277


namespace sqrt_360000_eq_600_l64_64513

theorem sqrt_360000_eq_600 : Real.sqrt 360000 = 600 := 
sorry

end sqrt_360000_eq_600_l64_64513


namespace sum_of_constants_l64_64613

variable (a b c : ℝ)

theorem sum_of_constants (h :  2 * (a - 2)^2 + 3 * (b - 3)^2 + 4 * (c - 4)^2 = 0) :
  a + b + c = 9 := 
sorry

end sum_of_constants_l64_64613


namespace rectangles_in_square_rectangles_in_three_squares_l64_64065

-- Given conditions as definitions
def positive_integer (n : ℕ) : Prop := n > 0

-- Part a
theorem rectangles_in_square (n : ℕ) (h : positive_integer n) :
  (n * (n + 1) / 2) ^ 2 = (n * (n + 1) / 2) ^ 2 :=
by sorry

-- Part b
theorem rectangles_in_three_squares (n : ℕ) (h : positive_integer n) :
  n^2 * (2 * n + 1)^2 - n^4 - n^3 * (n + 1) - (n * (n + 1) / 2)^2 = 
  n^2 * (2 * n + 1)^2 - n^4 - n^3 * (n + 1) - (n * (n + 1) / 2)^2 :=
by sorry

end rectangles_in_square_rectangles_in_three_squares_l64_64065


namespace coordinates_of_B_l64_64901
open Real

-- Define the conditions given in the problem
def A : ℝ × ℝ := (1, 6)
def d : ℝ := 4

-- Define the properties of the solution given the conditions
theorem coordinates_of_B (B : ℝ × ℝ) :
  (B = (-3, 6) ∨ B = (5, 6)) ↔
  (B.2 = A.2 ∧ (B.1 = A.1 - d ∨ B.1 = A.1 + d)) :=
by
  sorry

end coordinates_of_B_l64_64901


namespace sum_of_three_digit_positive_integers_l64_64697

noncomputable def sum_of_arithmetic_series (a l n : ℕ) : ℕ :=
  (a + l) / 2 * n

theorem sum_of_three_digit_positive_integers : 
  sum_of_arithmetic_series 100 999 900 = 494550 :=
by
  -- skipping the proof
  sorry

end sum_of_three_digit_positive_integers_l64_64697


namespace calculate_total_area_l64_64641

noncomputable def rectangle_sides := (AB BC : ℝ)
noncomputable def AB := 3
noncomputable def BC := 4

noncomputable def D_radius := Real.sqrt (AB ^ 2 + BC ^ 2)
noncomputable def M_radius := Real.sqrt (AB ^ 2 + (BC / 2) ^ 2)

noncomputable def total_area_regions_II_and_III := 30.3

theorem calculate_total_area :
  D_radius = 5 ∧ M_radius = Real.sqrt 13 ∧
  rectangle_sides (AB BC) → 
  total_area_regions_II_and_III = 30.3 :=
by
  intros
  sorry

end calculate_total_area_l64_64641


namespace geometric_sequence_b_general_term_a_l64_64748

-- Definitions of sequences and given conditions
def a (n : ℕ) : ℕ := sorry -- The sequence a_n
def S (n : ℕ) : ℕ := sorry -- The sum of the first n terms S_n

axiom a1_condition : a 1 = 2
axiom recursion_formula (n : ℕ): S (n+1) = 4 * a n + 2

def b (n : ℕ) : ℕ := a (n+1) - 2 * a n -- Definition of b_n

-- Theorem 1: Prove that b_n is a geometric sequence
theorem geometric_sequence_b (n : ℕ) : ∃ q, ∀ m, b (m+1) = q * b m :=
  sorry

-- Theorem 2: Find the general term formula for a_n
theorem general_term_a (n : ℕ) : a n = n * 2^n :=
  sorry

end geometric_sequence_b_general_term_a_l64_64748


namespace probability_xi_gt_2_l64_64118

open ProbabilityTheory MeasureTheory

noncomputable theory

def ξ : Measure ℝ := MeasureTheory.Measure.Normal.measure 0 (6^2)

theorem probability_xi_gt_2 :
  (MeasureTheory.Measure.Normal.cumulative_distribution ξ) (2) = 0.1 :=
sorry

end probability_xi_gt_2_l64_64118


namespace probability_two_diagonals_intersect_l64_64773

theorem probability_two_diagonals_intersect (n : ℕ) (h : 0 < n) : 
  let vertices := 2 * n + 1 in
  let total_diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_of_diagonals := total_diagonals.choose 2 in
  let crossing_diagonals := (vertices.choose 4) in
  ((crossing_diagonals * 2) / pairs_of_diagonals : ℚ) = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  sorry

end probability_two_diagonals_intersect_l64_64773


namespace min_radius_circle_condition_l64_64776

theorem min_radius_circle_condition (r : ℝ) (a b : ℝ) 
    (h_circle : (a - (r + 1))^2 + b^2 = r^2)
    (h_condition : b^2 ≥ 4 * a) :
    r ≥ 4 := 
sorry

end min_radius_circle_condition_l64_64776


namespace sum_of_roots_l64_64047

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end sum_of_roots_l64_64047


namespace evaluate_expression_l64_64283

def diamond (a b : ℚ) : ℚ := a - (2 / b)

theorem evaluate_expression :
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -(11 / 30) :=
by
  sorry

end evaluate_expression_l64_64283


namespace inequality_proof_l64_64908

variables (x y : ℝ) (n : ℕ)

theorem inequality_proof (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  (x^n / (x + y^3) + y^n / (x^3 + y)) ≥ (2^(4-n) / 5) := by
  sorry

end inequality_proof_l64_64908


namespace a100_gt_two_pow_99_l64_64595

theorem a100_gt_two_pow_99 
  (a : ℕ → ℤ) 
  (h1 : a 1 > a 0)
  (h2 : a 1 > 0)
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2 ^ 99 :=
sorry

end a100_gt_two_pow_99_l64_64595


namespace model_distance_comparison_l64_64064

theorem model_distance_comparison (m h c x y z : ℝ) (hm : 0 < m) (hh : 0 < h) (hc : 0 < c) (hz : 0 < z) (hx : 0 < x) (hy : 0 < y)
    (h_eq : (x - c) * z = (y - c) * (z + m) + h) :
    (if h > c * m then (x * z > y * (z + m))
     else if h < c * m then (x * z < y * (z + m))
     else (h = c * m → x * z = y * (z + m))) :=
by
  sorry

end model_distance_comparison_l64_64064


namespace find_solutions_l64_64732

theorem find_solutions (a m n : ℕ) (h : a > 0) (h₁ : m > 0) (h₂ : n > 0) :
  (a^m + 1) ∣ (a + 1)^n → 
  ((a = 1 ∧ True) ∨ (True ∧ m = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by sorry

end find_solutions_l64_64732


namespace quadratic_equality_l64_64610

theorem quadratic_equality (x : ℝ) 
  (h : 14*x + 5 - 21*x^2 = -2) : 
  6*x^2 - 4*x + 5 = 7 := 
by
  sorry

end quadratic_equality_l64_64610


namespace proof_of_problem_l64_64800

-- Define the problem conditions using a combination function
def problem_statement : Prop :=
  (Nat.choose 6 3 = 20)

theorem proof_of_problem : problem_statement :=
by
  sorry

end proof_of_problem_l64_64800


namespace ann_hill_length_l64_64347

/-- Given the conditions:
1. Mary slides down a hill that is 630 feet long at a speed of 90 feet/minute.
2. Ann slides down a hill at a rate of 40 feet/minute.
3. Ann's trip takes 13 minutes longer than Mary's.
Prove that the length of the hill Ann slides down is 800 feet. -/
theorem ann_hill_length
    (distance_Mary : ℕ) (speed_Mary : ℕ) 
    (speed_Ann : ℕ) (time_diff : ℕ)
    (h1 : distance_Mary = 630)
    (h2 : speed_Mary = 90)
    (h3 : speed_Ann = 40)
    (h4 : time_diff = 13) :
    speed_Ann * ((distance_Mary / speed_Mary) + time_diff) = 800 := 
by
    sorry

end ann_hill_length_l64_64347


namespace probability_first_prize_both_distribution_of_X_l64_64145

-- Definitions for the conditions
def total_students : ℕ := 500
def male_students : ℕ := 200
def female_students : ℕ := 300

def male_first_prize : ℕ := 10
def female_first_prize : ℕ := 25

def male_second_prize : ℕ := 15
def female_second_prize : ℕ := 25

def male_third_prize : ℕ := 15
def female_third_prize : ℕ := 40

-- Part (1): Prove the probability that both selected students receive the first prize is 1/240.
theorem probability_first_prize_both :
  (male_first_prize / male_students : ℚ) * (female_first_prize / female_students : ℚ) = 1 / 240 := 
sorry

-- Part (2): Prove the distribution of X.
def P_male_award : ℚ := (male_first_prize + male_second_prize + male_third_prize) / male_students
def P_female_award : ℚ := (female_first_prize + female_second_prize + female_third_prize) / female_students

theorem distribution_of_X :
  ∀ X : ℕ, X = 0 ∧ ((1 - P_male_award) * (1 - P_female_award) = 28 / 50) ∨ 
           X = 1 ∧ ((1 - P_male_award) * (1 - P_female_award) + (P_male_award * (1 - P_female_award)) + ((1 - P_male_award) * P_female_award) = 19 / 50) ∨ 
           X = 2 ∧ (P_male_award * P_female_award = 3 / 50) :=
sorry

end probability_first_prize_both_distribution_of_X_l64_64145


namespace harvest_bushels_l64_64096

def num_rows : ℕ := 5
def stalks_per_row : ℕ := 80
def stalks_per_bushel : ℕ := 8

theorem harvest_bushels : (num_rows * stalks_per_row) / stalks_per_bushel = 50 := by
  sorry

end harvest_bushels_l64_64096


namespace determine_values_of_abc_l64_64453

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def f_inv (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + b * x + a

theorem determine_values_of_abc 
  (a b c : ℝ) 
  (h_f : ∀ x : ℝ, f a b c (f_inv a b c x) = x)
  (h_f_inv : ∀ x : ℝ, f_inv a b c (f a b c x) = x) : 
  a = -1 ∧ b = 1 ∧ c = 0 :=
by
  sorry

end determine_values_of_abc_l64_64453


namespace amount_spent_on_marbles_l64_64798

/-- A theorem to determine the amount Mike spent on marbles. -/
theorem amount_spent_on_marbles 
  (total_amount : ℝ) 
  (cost_football : ℝ) 
  (cost_baseball : ℝ) 
  (total_amount_eq : total_amount = 20.52)
  (cost_football_eq : cost_football = 4.95)
  (cost_baseball_eq : cost_baseball = 6.52) :
  ∃ (cost_marbles : ℝ), cost_marbles = total_amount - (cost_football + cost_baseball) 
  ∧ cost_marbles = 9.05 := 
by
  sorry

end amount_spent_on_marbles_l64_64798


namespace smallest_digits_to_append_l64_64208

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l64_64208


namespace game_probability_correct_l64_64795

noncomputable def game_probability :=
  let total_outcomes := 3^3
  let outcomes_at_least_one_rock := total_outcomes - 2^3
  let outcomes_at_most_one_paper := 2^3 + 3 * 2^2
  let outcomes_intersection := outcomes_at_least_one_rock + outcomes_at_most_one_paper - total_outcomes
  let outcomes_exactly_one_scissors := 3 * 1^2
  let probability := outcomes_exactly_one_scissors / outcomes_intersection
  let m := 1
  let n := 4
  100 * m + n

theorem game_probability_correct : game_probability = 104 := by
  sorry

end game_probability_correct_l64_64795


namespace maximum_delta_value_l64_64135

-- Definition of the sequence a 
def a (n : ℕ) : ℕ := 1 + n^3

-- Definition of δ_n as the gcd of consecutive terms in the sequence a
def delta (n : ℕ) : ℕ := Nat.gcd (a (n + 1)) (a n)

-- Main theorem statement
theorem maximum_delta_value : ∃ n, delta n = 7 :=
by
  -- Insert the proof later
  sorry

end maximum_delta_value_l64_64135


namespace distinct_exponentiations_are_four_l64_64875

def power (a b : ℕ) : ℕ := a^b

def expr1 := power 3 (power 3 (power 3 3))
def expr2 := power 3 (power (power 3 3) 3)
def expr3 := power (power (power 3 3) 3) 3
def expr4 := power (power 3 (power 3 3)) 3
def expr5 := power (power 3 3) (power 3 3)

theorem distinct_exponentiations_are_four : 
  (expr1 ≠ expr2 ∧ expr1 ≠ expr3 ∧ expr1 ≠ expr4 ∧ expr1 ≠ expr5 ∧
   expr2 ≠ expr3 ∧ expr2 ≠ expr4 ∧ expr2 ≠ expr5 ∧
   expr3 ≠ expr4 ∧ expr3 ≠ expr5 ∧
   expr4 ≠ expr5) :=
sorry

end distinct_exponentiations_are_four_l64_64875


namespace simplify_and_find_ab_ratio_l64_64114

-- Given conditions
def given_expression (k : ℤ) : ℤ := 10 * k + 15

-- Simplified form
def simplified_form (k : ℤ) : ℤ := 2 * k + 3

-- Proof problem statement
theorem simplify_and_find_ab_ratio
  (k : ℤ) :
  let a := 2
  let b := 3
  (10 * k + 15) / 5 = 2 * k + 3 → 
  (a:ℚ) / (b:ℚ) = 2 / 3 := sorry

end simplify_and_find_ab_ratio_l64_64114


namespace solve_system_of_equations_l64_64997

theorem solve_system_of_equations (a b : ℝ) (h1 : a^2 ≠ 1) (h2 : b^2 ≠ 1) (h3 : a ≠ b) : 
  (∃ x y : ℝ, 
    (x - y) / (1 - x * y) = 2 * a / (1 + a^2) ∧ (x + y) / (1 + x * y) = 2 * b / (1 + b^2) ∧
    ((x = (a * b + 1) / (a + b) ∧ y = (a * b - 1) / (a - b)) ∨ 
     (x = (a + b) / (a * b + 1) ∧ y = (a - b) / (a * b - 1)))) :=
by
  sorry

end solve_system_of_equations_l64_64997


namespace find_f_of_2_l64_64902

theorem find_f_of_2 : ∃ (f : ℤ → ℤ), (∀ x : ℤ, f (x+1) = x^2 - 1) ∧ f 2 = 0 :=
by
  sorry

end find_f_of_2_l64_64902


namespace harvest_bushels_l64_64095

def num_rows : ℕ := 5
def stalks_per_row : ℕ := 80
def stalks_per_bushel : ℕ := 8

theorem harvest_bushels : (num_rows * stalks_per_row) / stalks_per_bushel = 50 := by
  sorry

end harvest_bushels_l64_64095


namespace inequality_proof_l64_64976

theorem inequality_proof (x a : ℝ) (hx : 0 < x) (ha : 0 < a) :
  (1 / Real.sqrt (x + 1)) + (1 / Real.sqrt (a + 1)) + Real.sqrt ( (a * x) / (a * x + 8) ) ≤ 2 := 
by {
  sorry
}

end inequality_proof_l64_64976


namespace range_of_a_l64_64134

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0)
  ↔ a ∈ Icc (-3/5 : ℝ) 1 :=
begin
  sorry
end

end range_of_a_l64_64134


namespace find_x_y_z_of_fold_points_l64_64593

noncomputable def area_of_fold_points_of_triangle (DE DF : ℝ) (angle_E : ℝ) : ℝ :=
  let radius_DE := DE / 2
  let radius_DF := DF / 2
  (1 / 2) * Real.pi * radius_DE^2

theorem find_x_y_z_of_fold_points (DE DF : ℝ) (angle_E : ℝ) (H_DE : DE = 20) (H_DF : DF = 40) (H_angle : angle_E = 90) :
  area_of_fold_points_of_triangle DE DF angle_E = 50 * Real.pi ∧ 50 + 0 + 1 = 51 := by
  sorry

end find_x_y_z_of_fold_points_l64_64593


namespace inclination_angle_l64_64175

theorem inclination_angle (θ : ℝ) : 
  (∃ (x y : ℝ), x + y - 3 = 0) → θ = 3 * Real.pi / 4 := 
sorry

end inclination_angle_l64_64175


namespace range_of_x_l64_64914

def f (x : ℝ) : ℝ := abs (x - 2)

theorem range_of_x (a b x : ℝ) (a_nonzero : a ≠ 0) (ab_real : a ∈ Set.univ ∧ b ∈ Set.univ) : 
  (|a + b| + |a - b| ≥ |a| • f x) ↔ (0 ≤ x ∧ x ≤ 4) :=
sorry

end range_of_x_l64_64914


namespace simplify_and_calculate_expression_l64_64362

theorem simplify_and_calculate_expression (a b : ℤ) (ha : a = -1) (hb : b = -2) :
  (2 * a + b) * (b - 2 * a) - (a - 3 * b) ^ 2 = -25 :=
by 
  -- We can use 'by' to start the proof and 'sorry' to skip it
  sorry

end simplify_and_calculate_expression_l64_64362


namespace largest_prime_factor_5292_l64_64043

theorem largest_prime_factor_5292 :
  ∃ p, nat.prime p ∧ p ∣ 5292 ∧ ∀ q, nat.prime q ∧ q ∣ 5292 → q ≤ p :=
sorry

end largest_prime_factor_5292_l64_64043


namespace find_d_and_a11_l64_64147

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_d_and_a11 (a : ℕ → ℤ) (d : ℤ) :
  arithmetic_sequence a d →
  a 5 = 6 →
  a 8 = 15 →
  d = 3 ∧ a 11 = 24 :=
by
  intros h_seq h_a5 h_a8
  sorry

end find_d_and_a11_l64_64147


namespace upper_seat_ticket_price_l64_64518

variable (U : ℝ) 

-- Conditions
def lower_seat_price : ℝ := 30
def total_tickets_sold : ℝ := 80
def total_revenue : ℝ := 2100
def lower_tickets_sold : ℝ := 50

theorem upper_seat_ticket_price :
  (lower_seat_price * lower_tickets_sold + (total_tickets_sold - lower_tickets_sold) * U = total_revenue) →
  U = 20 := by
  sorry

end upper_seat_ticket_price_l64_64518


namespace smallest_digits_to_append_l64_64211

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l64_64211


namespace volume_of_cuboid_l64_64653

theorem volume_of_cuboid (a b c : ℕ) (h_a : a = 2) (h_b : b = 5) (h_c : c = 8) : 
  a * b * c = 80 := 
by 
  sorry

end volume_of_cuboid_l64_64653


namespace original_rectangle_area_at_least_90_l64_64266

variable (a b c x y z : ℝ)
variable (hx1 : a * x = 1)
variable (hx2 : c * x = 3)
variable (hy : b * y = 10)
variable (hz : a * z = 9)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hx : 0 < x) (hy' : 0 < y) (hz' : 0 < z)

theorem original_rectangle_area_at_least_90 : ∀ {a b c x y z : ℝ},
  (a * x = 1) →
  (c * x = 3) →
  (b * y = 10) →
  (a * z = 9) →
  (0 < a) →
  (0 < b) →
  (0 < c) →
  (0 < x) →
  (0 < y) →
  (0 < z) →
  (a + b + c) * (x + y + z) ≥ 90 :=
sorry

end original_rectangle_area_at_least_90_l64_64266


namespace ratio_of_length_to_width_l64_64368

variable (L W : ℕ)
variable (H1 : W = 50)
variable (H2 : 2 * L + 2 * W = 240)

theorem ratio_of_length_to_width : L / W = 7 / 5 := 
by sorry

end ratio_of_length_to_width_l64_64368


namespace find_x_satisfying_inequality_l64_64884

open Real

theorem find_x_satisfying_inequality :
  ∀ x : ℝ, 0 < x → (x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16 ↔ x = 4) :=
by
  sorry

end find_x_satisfying_inequality_l64_64884


namespace percent_of_a_is_4b_l64_64645

variable (a b : ℝ)

theorem percent_of_a_is_4b (hab : a = 1.8 * b) :
  (4 * b / a) * 100 = 222.22 := by
  sorry

end percent_of_a_is_4b_l64_64645


namespace max_expression_l64_64787

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem max_expression (x y : ℝ) (h : x + y = 5) :
  max_value x y ≤ 6084 / 17 :=
sorry

end max_expression_l64_64787


namespace pills_supply_duration_l64_64778

open Nat

-- Definitions based on conditions
def one_third_pill_every_three_days : ℕ := 1 / 3 * 3
def pills_in_bottle : ℕ := 90
def days_per_pill : ℕ := 9
def days_per_month : ℕ := 30

-- The Lean statement to prove the question == answer given conditions
theorem pills_supply_duration : (pills_in_bottle * days_per_pill) / days_per_month = 27 := by
  sorry

end pills_supply_duration_l64_64778


namespace find_d1_l64_64341

noncomputable def E (n : ℕ) : ℕ := sorry

theorem find_d1 :
  ∃ (d4 d3 d2 d0 : ℤ), 
  (∀ (n : ℕ), n ≥ 4 ∧ n % 2 = 0 → 
     E n = d4 * n^4 + d3 * n^3 + d2 * n^2 + (12 : ℤ) * n + d0) :=
sorry

end find_d1_l64_64341


namespace remainder_of_9876543210_div_101_l64_64687

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l64_64687


namespace sum_divided_among_xyz_l64_64271

noncomputable def total_amount (x_share y_share z_share : ℝ) : ℝ :=
  x_share + y_share + z_share

theorem sum_divided_among_xyz
    (x_share : ℝ) (y_share : ℝ) (z_share : ℝ)
    (y_gets_45_paisa : y_share = 0.45 * x_share)
    (z_gets_50_paisa : z_share = 0.50 * x_share)
    (y_share_is_18 : y_share = 18) :
    total_amount x_share y_share z_share = 78 := by
  sorry

end sum_divided_among_xyz_l64_64271


namespace mark_charged_more_hours_l64_64006

theorem mark_charged_more_hours (P K M : ℕ) 
  (h_total : P + K + M = 144)
  (h_pat_kate : P = 2 * K)
  (h_pat_mark : P = M / 3) : M - K = 80 := 
by
  sorry

end mark_charged_more_hours_l64_64006


namespace crate_minimum_dimension_l64_64075

theorem crate_minimum_dimension (a : ℕ) (h1 : a ≥ 12) :
  min a (min 8 12) = 8 :=
by
  sorry

end crate_minimum_dimension_l64_64075


namespace find_x_satisfying_inequality_l64_64883

open Real

theorem find_x_satisfying_inequality :
  ∀ x : ℝ, 0 < x → (x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16 ↔ x = 4) :=
by
  sorry

end find_x_satisfying_inequality_l64_64883


namespace line_passes_through_quadrants_l64_64743

theorem line_passes_through_quadrants (a b c : ℝ) (hab : a * b < 0) (hbc : b * c < 0) : 
  ∀ (x y : ℝ), (a * x + b * y + c = 0) → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
by {
  sorry
}

end line_passes_through_quadrants_l64_64743


namespace equation_has_100_solutions_l64_64887

noncomputable theory

open Real

def num_solutions : ℝ :=
  { x : ℝ | 0 ≤ x ∧ x ≤ 100 * π ∧ cos (π / 2 + x) = (1 / 2) ^ x }.toFinset.card

theorem equation_has_100_solutions :
    num_solutions = 100 :=
sorry

end equation_has_100_solutions_l64_64887


namespace count_even_three_digit_numbers_with_sum_12_l64_64934

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l64_64934


namespace function_equivalence_l64_64296

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x) = 6 * x - 1) : ∀ x : ℝ, f x = 3 * x - 1 :=
by
  sorry

end function_equivalence_l64_64296


namespace cube_root_power_l64_64579

theorem cube_root_power (a : ℝ) (h : a = 8) : (a^(1/3))^12 = 4096 := by
  rw [h]
  have h2 : 8 = 2^3 := rfl
  rw h2
  sorry

end cube_root_power_l64_64579


namespace min_value_of_inverse_sum_l64_64142

noncomputable def min_value (a b : ℝ) := ¬(1 ≤ a + 2*b)

theorem min_value_of_inverse_sum (a b : ℝ) (h : a + 2 * b = 1) (h_nonneg : 0 < a ∧ 0 < b) :
  (1 / a + 2 / b) ≥ 9 :=
sorry

end min_value_of_inverse_sum_l64_64142


namespace bees_on_second_day_l64_64004

theorem bees_on_second_day (bees_first_day : ℕ) (tripling_factor : ℕ) (h1 : bees_first_day = 144) (h2 : tripling_factor = 3) :
  let bees_second_day := bees_first_day * tripling_factor
  in bees_second_day = 432 := 
by
  intros
  have h3 : bees_second_day = 144 * 3 := by rw [h1, h2]
  rw h3
  norm_num
  exact rfl

end bees_on_second_day_l64_64004


namespace total_rehabilitation_centers_l64_64380

def lisa_visits : ℕ := 6
def jude_visits (lisa : ℕ) : ℕ := lisa / 2
def han_visits (jude : ℕ) : ℕ := 2 * jude - 2
def jane_visits (han : ℕ) : ℕ := 2 * han + 6
def total_visits (lisa jude han jane : ℕ) : ℕ := lisa + jude + han + jane

theorem total_rehabilitation_centers :
  total_visits lisa_visits (jude_visits lisa_visits) (han_visits (jude_visits lisa_visits)) 
    (jane_visits (han_visits (jude_visits lisa_visits))) = 27 :=
by
  sorry

end total_rehabilitation_centers_l64_64380


namespace sequence_is_geometric_not_arithmetic_l64_64617

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2^(n-1)

def S_n (n : ℕ) : ℕ :=
  2^n - 1

theorem sequence_is_geometric_not_arithmetic (n : ℕ) : 
  (∀ n ≥ 2, a_n n = S_n n - S_n (n - 1)) ∧
  (a_n 1 = 1) ∧
  (∃ r : ℕ, r > 1 ∧ ∀ n ≥ 1, a_n (n + 1) = r * a_n n) ∧
  ¬(∃ d : ℤ, ∀ n, (a_n (n + 1) : ℤ) = a_n n + d) :=
by
  sorry

end sequence_is_geometric_not_arithmetic_l64_64617


namespace evaluate_expression_l64_64727

theorem evaluate_expression : (64^(1 / 6) * 16^(1 / 4) * 8^(1 / 3) = 8) :=
by
  -- sorry added to skip the proof
  sorry

end evaluate_expression_l64_64727


namespace reciprocal_of_8_l64_64528

theorem reciprocal_of_8:
  (1 : ℝ) / 8 = (1 / 8 : ℝ) := by
  sorry

end reciprocal_of_8_l64_64528


namespace math_problem_l64_64977

variables (x y z w p q : ℕ)
variables (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) (w_pos : 0 < w)

theorem math_problem
  (h1 : x^3 = y^2)
  (h2 : z^4 = w^3)
  (h3 : z - x = 22)
  (hx : x = p^2)
  (hy : y = p^3)
  (hz : z = q^3)
  (hw : w = q^4) : w - y = q^4 - p^3 :=
sorry

end math_problem_l64_64977


namespace Veronica_to_Half_Samir_Ratio_l64_64812

-- Mathematical conditions 
def Samir_stairs : ℕ := 318
def Total_stairs : ℕ := 495
def Half_Samir_stairs : ℚ := Samir_stairs / 2

-- Definition for Veronica's stairs as a multiple of half Samir's stairs
def Veronica_stairs (R: ℚ) : ℚ := R * Half_Samir_stairs

-- Lean statement to prove the ratio
theorem Veronica_to_Half_Samir_Ratio (R : ℚ) (H1 : Veronica_stairs R + Samir_stairs = Total_stairs) : R = 1.1132 := 
by
  sorry

end Veronica_to_Half_Samir_Ratio_l64_64812


namespace Genevieve_cherry_weight_l64_64440

theorem Genevieve_cherry_weight
  (cost_per_kg : ℕ) (short_of_total : ℕ) (amount_owned : ℕ) (total_kg : ℕ) :
  cost_per_kg = 8 →
  short_of_total = 400 →
  amount_owned = 1600 →
  total_kg = 250 :=
by
  intros h_cost_per_kg h_short_of_total h_amount_owned
  have h_equation : 8 * total_kg = 1600 + 400 := by
    rw [h_cost_per_kg, h_short_of_total, h_amount_owned]
    apply sorry -- This is where the exact proof mechanism would go
  sorry -- Skipping the remainder of the proof

end Genevieve_cherry_weight_l64_64440


namespace sin_identity_l64_64126

theorem sin_identity (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) :
  Real.sin ((3 * π / 4) - α) = 3 / 5 :=
by
  sorry

end sin_identity_l64_64126


namespace number_of_democrats_in_senate_l64_64831

/-
This Lean statement captures the essence of the problem: proving the number of Democrats in the Senate (S_D) is 55,
under given conditions involving the House's and Senate's number of Democrats and Republicans.
-/

theorem number_of_democrats_in_senate
  (D R S_D S_R : ℕ)
  (h1 : D + R = 434)
  (h2 : R = D + 30)
  (h3 : S_D + S_R = 100)
  (h4 : S_D * 4 = S_R * 5) :
  S_D = 55 := by
  sorry

end number_of_democrats_in_senate_l64_64831


namespace sequence_general_term_l64_64747

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (n / (n + 1 : ℝ)) * a n) : 
  ∀ n, a n = 1 / n := by
  sorry

end sequence_general_term_l64_64747


namespace sum_of_roots_of_quadratic_l64_64052

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end sum_of_roots_of_quadratic_l64_64052


namespace hyperbola_m_value_l64_64485

theorem hyperbola_m_value (m k : ℝ) (h₀ : k > 0) (h₁ : 0 < -m) 
  (h₂ : 2 * k = Real.sqrt (1 + m)) : 
  m = -3 := 
by {
  sorry
}

end hyperbola_m_value_l64_64485


namespace find_T_shirts_l64_64802

variable (T S : ℕ)

-- Given conditions
def condition1 : S = 2 * T := sorry
def condition2 : T + S - (T + 3) = 15 := sorry

-- Prove that number of T-shirts T Norma left in the washer is 9
theorem find_T_shirts (h1 : S = 2 * T) (h2 : T + S - (T + 3) = 15) : T = 9 :=
  by
    sorry

end find_T_shirts_l64_64802


namespace probability_win_l64_64663

theorem probability_win (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose) = 3 / 8 :=
by
  rw [h]
  norm_num

end probability_win_l64_64663


namespace parabola_problem_l64_64078

noncomputable def p_value_satisfy_all_conditions (p : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ) (A B : ℝ × ℝ),
    F = (p / 2, 0) ∧
    (A.2 = A.1 - p / 2 ∧ (A.2)^2 = 2 * p * A.1) ∧
    (B.2 = B.1 - p / 2 ∧ (B.2)^2 = 2 * p * B.1) ∧
    (A.1 + B.1) / 2 = 3 * p / 2 ∧
    (A.2 + B.2) / 2 = p ∧
    (p - 2 = -3 * p / 2)

theorem parabola_problem : ∃ (p : ℝ), p_value_satisfy_all_conditions p ∧ p = 4 / 5 :=
by
  sorry

end parabola_problem_l64_64078


namespace train_b_speed_l64_64534

theorem train_b_speed (v : ℝ) (t : ℝ) (d : ℝ) (sA : ℝ := 30) (start_time_diff : ℝ := 2) :
  (d = 180) -> (60 + sA*t = d) -> (v * t = d) -> v = 45 := by 
  sorry

end train_b_speed_l64_64534


namespace quadratic_root_l64_64763

theorem quadratic_root (k : ℝ) (h : (1:ℝ)^2 - 3 * (1 : ℝ) - k = 0) : k = -2 :=
sorry

end quadratic_root_l64_64763


namespace positive_root_in_range_l64_64826

theorem positive_root_in_range : ∃ x > 0, (x^2 - 2 * x - 1 = 0) ∧ (2 < x ∧ x < 3) :=
by
  sorry

end positive_root_in_range_l64_64826


namespace genotypes_of_parents_l64_64807

-- Define the possible alleles
inductive Allele
| H : Allele -- Hairy (dominant)
| h : Allele -- Hairy (recessive)
| S : Allele -- Smooth (dominant)
| s : Allele -- Smooth (recessive)

-- Define the genotype as a pair of alleles
def Genotype := (Allele × Allele)

-- Define the phenotype based on the genotype
def phenotype : Genotype → Allele
| (Allele.H, _) => Allele.H -- HH or HS results in Hairy
| (_, Allele.H) => Allele.H
| (Allele.S, _) => Allele.S -- SS results in Smooth
| (_, Allele.S) => Allele.S
| _           => Allele.s   -- Others

-- Given conditions
def p : ℝ := 0.1 -- probability of allele H
def q : ℝ := 1 - p -- probability of allele S

-- Define most likely genotypes of parents based on the conditions
def most_likely_genotype_of_parents (offspring: list Genotype) : Genotype × Genotype :=
  ((Allele.H, Allele.H), (Allele.S, Allele.h))

-- The main statement
theorem genotypes_of_parents (all_furry: ∀ g : Genotype, phenotype g = Allele.H) :
  most_likely_genotype_of_parents [(Allele.H, Allele.H), (Allele.H, Allele.s)]
  = ((Allele.H, Allele.H), (Allele.S, Allele.h)) :=
sorry

end genotypes_of_parents_l64_64807


namespace pascal_triangle_contains_53_l64_64462

theorem pascal_triangle_contains_53 (n : ℕ) :
  (∃ k, binomial n k = 53) ↔ n = 53 := 
sorry

end pascal_triangle_contains_53_l64_64462


namespace max_convex_quadrilaterals_l64_64592

-- Define the points on the plane and the conditions
variable (A : Fin 7 → (ℝ × ℝ))

-- Hypothesis that any 3 given points are not collinear
def not_collinear (P Q R : (ℝ × ℝ)) : Prop :=
  (Q.1 - P.1) * (R.2 - P.2) ≠ (Q.2 - P.2) * (R.1 - P.1)

-- Hypothesis that the convex hull of all points is \triangle A1 A2 A3
def convex_hull_triangle (A : Fin 7 → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : Fin 7), i ≠ j → j ≠ k → i ≠ k → not_collinear (A i) (A j) (A k)

-- The theorem to be proven
theorem max_convex_quadrilaterals :
  convex_hull_triangle A →
  (∀ i j k : Fin 7, i ≠ j → j ≠ k → i ≠ k → not_collinear (A i) (A j) (A k)) →
  ∃ n, n = 17 := 
by
  sorry

end max_convex_quadrilaterals_l64_64592


namespace Alyssa_spent_on_marbles_l64_64084

def total_spent_on_toys : ℝ := 12.30
def cost_of_football : ℝ := 5.71
def amount_spent_on_marbles : ℝ := 12.30 - 5.71

theorem Alyssa_spent_on_marbles :
  total_spent_on_toys - cost_of_football = amount_spent_on_marbles :=
by
  sorry

end Alyssa_spent_on_marbles_l64_64084


namespace percentage_increase_book_price_l64_64018

theorem percentage_increase_book_price (OldP NewP : ℕ) (hOldP : OldP = 300) (hNewP : NewP = 330) :
  ((NewP - OldP : ℕ) / OldP : ℚ) * 100 = 10 := by
  sorry

end percentage_increase_book_price_l64_64018


namespace find_a_l64_64606

noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := (Real.sqrt (a^2 + 3)) / a

theorem find_a (a : ℝ) (h : a > 0) (hexp : hyperbola_eccentricity a = 2) : a = 1 :=
by
  sorry

end find_a_l64_64606


namespace interest_rate_of_additional_investment_l64_64719

section
variable (r : ℝ)

theorem interest_rate_of_additional_investment
  (h : 2800 * 0.05 + 1400 * r = 0.06 * (2800 + 1400)) :
  r = 0.08 := by
  sorry
end

end interest_rate_of_additional_investment_l64_64719


namespace area_increase_percent_l64_64949

theorem area_increase_percent (l w : ℝ) :
  let original_area := l * w
  let new_length := 1.15 * l
  let new_width := 1.25 * w
  let new_area := new_length * new_width
  let increase_percent := ((new_area - original_area) / original_area) * 100
  increase_percent = 43.75 :=
by
  let original_area := l * w
  let new_length := 1.15 * l
  let new_width := 1.25 * w
  let new_area := new_length * new_width
  let increase_percent := ((new_area - original_area) / original_area) * 100
  calc
    increase_percent = ((new_area - original_area) / original_area) * 100 : rfl
    ... = ((1.15 * l * 1.25 * w - l * w) / (l * w)) * 100 : by rw [← mul_assoc, ← mul_assoc, mul_comm (1.15), mul_assoc]
    ... = 43.75 : sorry

end area_increase_percent_l64_64949


namespace angle_PTV_60_l64_64148

variables (m n TV TPV PTV : ℝ)

-- We state the conditions
axiom parallel_lines : m = n
axiom angle_TPV : TPV = 150
axiom angle_TVP_perpendicular : TV = 90

-- The goal statement to prove
theorem angle_PTV_60 : PTV = 60 :=
by
  sorry

end angle_PTV_60_l64_64148


namespace triangle_angle_C_l64_64952

theorem triangle_angle_C (A B C : Real) (h1 : A - B = 10) (h2 : B = A / 2) :
  C = 150 :=
by
  -- Proof goes here
  sorry

end triangle_angle_C_l64_64952


namespace calc_f_at_3_l64_64119

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem calc_f_at_3 : f 3 = 328 := 
sorry

end calc_f_at_3_l64_64119


namespace determine_parents_genotype_l64_64809

noncomputable def genotype := ℕ -- We use nat to uniquely represent each genotype: e.g. HH=0, HS=1, Sh=2, SS=3

def probability_of_allele_H : ℝ := 0.1
def probability_of_allele_S : ℝ := 1 - probability_of_allele_H

def is_dominant (allele: ℕ) : Prop := allele == 0 ∨ allele == 1 -- HH or HS are dominant for hairy

def offspring_is_hairy (parent1 parent2: genotype) : Prop :=
  (∃ g1 g2, (parent1 = 0 ∨ parent1 = 1) ∧ (parent2 = 1 ∨ parent2 = 2 ∨ parent2 = 3) ∧
  ((g1 = 0 ∨ g1 = 1) ∧ (g2 = 0 ∨ g2 = 1))) ∧ 
  (is_dominant 0 ∧ is_dominant 0 ∧ is_dominant 0 ∧ is_dominant 0) 

def most_likely_genotypes (hairy_parent smooth_parent : genotype) : Prop :=
  (hairy_parent = 0) ∧ (smooth_parent = 2)

theorem determine_parents_genotype :
  ∃ hairy_parent smooth_parent, offspring_is_hairy hairy_parent smooth_parent ∧ most_likely_genotypes hairy_parent smooth_parent :=
  sorry

end determine_parents_genotype_l64_64809


namespace S8_is_255_l64_64746

-- Definitions and hypotheses
def geometric_sequence_sum (a : ℕ → ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  a 0 * (1 - q^n) / (1 - q)

variables (a : ℕ → ℚ) (q : ℚ)
variable (h_geo_seq : ∀ n, a (n + 1) = a n * q)
variable (h_S2 : geometric_sequence_sum a q 2 = 3)
variable (h_S4 : geometric_sequence_sum a q 4 = 15)

-- Goal
theorem S8_is_255 : geometric_sequence_sum a q 8 = 255 := 
by {
  -- skipping the proof
  sorry
}

end S8_is_255_l64_64746


namespace arithmetic_sequence_sum_minimum_l64_64302

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ := 
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum_minimum (a1 : ℝ) (d : ℝ) :
  a1 = -20 ∧ (∀ n : ℕ, (S_n a1 d n) > (S_n a1 d 6)) → 
  (10 / 3 < d ∧ d < 4) := 
sorry

end arithmetic_sequence_sum_minimum_l64_64302


namespace find_range_of_x_l64_64446

variable (f : ℝ → ℝ) (x : ℝ)

-- Assume f is an increasing function on [-1, 1]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x ≤ y → f x ≤ f y

-- Main theorem statement based on the problem
theorem find_range_of_x (h_increasing : is_increasing_on_interval f (-1) 1)
                        (h_condition : f (x - 1) < f (1 - 3 * x)) :
  0 ≤ x ∧ x < (1 / 2) :=
sorry

end find_range_of_x_l64_64446


namespace sufficient_but_not_necessary_l64_64257

-- Define the quadratic function
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + 2*x + m

-- The problem statement to prove that "m < 1" is a sufficient condition
-- but not a necessary condition for the function f(x) to have a root.
theorem sufficient_but_not_necessary (m : ℝ) :
  (m < 1 → ∃ x : ℝ, f x m = 0) ∧ ¬(¬(m < 1) → ∃ x : ℝ, f x m = 0) :=
sorry

end sufficient_but_not_necessary_l64_64257


namespace train_length_l64_64025

noncomputable def length_of_first_train (l2 : ℝ) (v1 : ℝ) (v2 : ℝ) (t : ℝ) : ℝ :=
  let v1_m_per_s := v1 * 1000 / 3600
  let v2_m_per_s := v2 * 1000 / 3600
  let relative_speed := v1_m_per_s + v2_m_per_s
  let combined_length := relative_speed * t
  combined_length - l2

theorem train_length (l2 : ℝ) (v1 : ℝ) (v2 : ℝ) (t : ℝ) (h_l2 : l2 = 200) 
  (h_v1 : v1 = 100) (h_v2 : v2 = 200) (h_t : t = 3.6) : length_of_first_train l2 v1 v2 t = 100 := by
  sorry

end train_length_l64_64025


namespace count_triangles_l64_64476

-- Assuming the conditions are already defined and given as parameters  
-- Let's define a proposition to prove the solution

noncomputable def total_triangles_in_figure : ℕ := 68

-- Create the theorem statement:
theorem count_triangles : total_triangles_in_figure = 68 := 
by
  sorry

end count_triangles_l64_64476


namespace bernoulli_inequality_l64_64789

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x ≥ -1) (hn : n ≥ 1) : (1 + x)^n ≥ 1 + n * x :=
by sorry

end bernoulli_inequality_l64_64789


namespace pascal_contains_53_l64_64455

theorem pascal_contains_53 (n : ℕ) (h1 : Nat.Prime 53) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 52 ∧ nat.choose 53 k = 53) (h3 : ∀ m < 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) (h4 : ∀ m > 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) : 
  (n = 53) → (n = 1) := 
by
  intros
  sorry

end pascal_contains_53_l64_64455


namespace gcd_of_powers_of_three_l64_64377

theorem gcd_of_powers_of_three :
  let a := 3^1001 - 1
  let b := 3^1012 - 1
  gcd a b = 177146 := by
  sorry

end gcd_of_powers_of_three_l64_64377


namespace coordinates_respect_to_origin_l64_64957

def point_coordinates : ℝ × ℝ := (-1, 2)

theorem coordinates_respect_to_origin :
  point_coordinates = (-1, 2) :=
begin
  -- We refer to the definition of the point
  -- as stated in the conditions.
  refl,
end

end coordinates_respect_to_origin_l64_64957


namespace arithmetic_to_geometric_seq_l64_64301

theorem arithmetic_to_geometric_seq
  (d a : ℕ) 
  (h1 : d ≠ 0) 
  (a_n : ℕ → ℕ)
  (h2 : ∀ n, a_n n = a + (n - 1) * d)
  (h3 : (a + 2 * d) * (a + 2 * d) = a * (a + 8 * d))
  : (a_n 2 + a_n 4 + a_n 10) / (a_n 1 + a_n 3 + a_n 9) = 16 / 13 :=
by
  sorry

end arithmetic_to_geometric_seq_l64_64301


namespace cos_sq_plus_two_sin_double_l64_64311

theorem cos_sq_plus_two_sin_double (α : ℝ) (h : Real.tan α = 3 / 4) : Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end cos_sq_plus_two_sin_double_l64_64311


namespace math_problem_l64_64423

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end math_problem_l64_64423


namespace no_real_solution_for_x_l64_64259

theorem no_real_solution_for_x
  (y : ℝ)
  (x : ℝ)
  (h1 : y = (x^3 - 8) / (x - 2))
  (h2 : y = 3 * x) :
  ¬ ∃ x : ℝ, y = 3*x ∧ y = (x^3 - 8) / (x - 2) :=
by {
  sorry
}

end no_real_solution_for_x_l64_64259


namespace remainder_9876543210_mod_101_l64_64684

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l64_64684


namespace union_of_sets_l64_64303

def A : Set ℝ := {x | x < -1 ∨ x > 3}
def B : Set ℝ := {x | x ≥ 2}

theorem union_of_sets : A ∪ B = {x | x < -1 ∨ x ≥ 2} :=
by
  sorry

end union_of_sets_l64_64303


namespace transactions_proof_l64_64803

def transactions_problem : Prop :=
  let mabel_transactions := 90
  let anthony_transactions := mabel_transactions + (0.10 * mabel_transactions)
  let cal_transactions := (2 / 3) * anthony_transactions
  let jade_transactions := 81
  jade_transactions - cal_transactions = 15

-- The proof is omitted (replace 'sorry' with an actual proof)
theorem transactions_proof : transactions_problem := by
  sorry

end transactions_proof_l64_64803


namespace smallest_number_of_digits_to_append_l64_64239

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l64_64239


namespace tangent_line_equation_l64_64521

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / x

theorem tangent_line_equation :
  let x₀ := 1
  let y₀ := f x₀
  let m := deriv f x₀
  y₀ = 0 →
  m = 3 →
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = 3 * x - 3) :=
by
  intros x₀ y₀ m h₀ hm x y
  sorry

end tangent_line_equation_l64_64521


namespace complement_union_complement_intersection_l64_64853

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem complement_union (A B : Set ℝ) :
  (A ∪ B)ᶜ = { x : ℝ | x ≤ 2 ∨ x ≥ 10 } :=
by
  sorry

theorem complement_intersection (A B : Set ℝ) :
  (Aᶜ ∩ B) = { x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } :=
by
  sorry

end complement_union_complement_intersection_l64_64853


namespace ryan_more_hours_english_than_chinese_l64_64582

-- Definitions for the time Ryan spends on subjects
def weekday_hours_english : ℕ := 6 * 5
def weekend_hours_english : ℕ := 2 * 2
def total_hours_english : ℕ := weekday_hours_english + weekend_hours_english

def weekday_hours_chinese : ℕ := 3 * 5
def weekend_hours_chinese : ℕ := 1 * 2
def total_hours_chinese : ℕ := weekday_hours_chinese + weekend_hours_chinese

-- Theorem stating the difference in hours spent on English vs Chinese
theorem ryan_more_hours_english_than_chinese :
  (total_hours_english - total_hours_chinese) = 17 := by
  sorry

end ryan_more_hours_english_than_chinese_l64_64582


namespace common_ratio_geometric_sequence_l64_64307

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_arith : 2 * (1/2 * a 5) = a 3 + a 4) : q = (1 + Real.sqrt 5) / 2 :=
sorry

end common_ratio_geometric_sequence_l64_64307


namespace darnell_texts_l64_64100

theorem darnell_texts (T : ℕ) (unlimited_plan_cost alternative_text_cost alternative_call_cost : ℕ) 
    (call_minutes : ℕ) (cost_difference : ℕ) :
    unlimited_plan_cost = 12 →
    alternative_text_cost = 1 →
    alternative_call_cost = 3 →
    call_minutes = 60 →
    cost_difference = 1 →
    (alternative_text_cost * T / 30 + alternative_call_cost * call_minutes / 20) = 
      unlimited_plan_cost - cost_difference →
    T = 60 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end darnell_texts_l64_64100


namespace ratio_of_discretionary_income_l64_64576

theorem ratio_of_discretionary_income 
  (salary : ℝ) (D : ℝ)
  (h_salary : salary = 3500)
  (h_discretionary : 0.15 * D = 105) :
  D / salary = 1 / 5 :=
by
  sorry

end ratio_of_discretionary_income_l64_64576


namespace first_term_geometric_sequence_l64_64434

theorem first_term_geometric_sequence (a r : ℕ) (h1 : r = 3) (h2 : a * r^4 = 81) : a = 1 :=
by
  sorry

end first_term_geometric_sequence_l64_64434


namespace find_value_of_a_l64_64447

variable (a : ℝ)

def f (x : ℝ) := x^2 + 4
def g (x : ℝ) := x^2 - 2

theorem find_value_of_a (h_pos : a > 0) (h_eq : f (g a) = 12) : a = Real.sqrt (2 * (Real.sqrt 2 + 1)) := 
by
  sorry

end find_value_of_a_l64_64447


namespace sum_of_roots_of_quadratic_l64_64053

theorem sum_of_roots_of_quadratic :
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16 in
  (∀ x, f x = 0 → x = 7 ∨ x = -1) →
  (let sum_of_roots := 7 + (-1) in sum_of_roots = 6) :=
by
  sorry

end sum_of_roots_of_quadratic_l64_64053


namespace number_count_two_digit_property_l64_64404

open Nat

theorem number_count_two_digit_property : 
  (∃ (n : Finset ℕ), (∀ (x : ℕ), x ∈ n ↔ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 11 * a + 2 * b ≡ 7 [MOD 10] ∧ x = 10 * a + b) ∧ n.card = 5) :=
by
  sorry

end number_count_two_digit_property_l64_64404


namespace factorize_expression_l64_64585

variable (m n : ℝ)

theorem factorize_expression : 12 * m^2 * n - 12 * m * n + 3 * n = 3 * n * (2 * m - 1)^2 :=
by
  sorry

end factorize_expression_l64_64585


namespace billy_apples_l64_64092

def num_apples_eaten (monday_apples tuesday_apples wednesday_apples thursday_apples friday_apples total_apples : ℕ) : Prop :=
  monday_apples = 2 ∧
  tuesday_apples = 2 * monday_apples ∧
  wednesday_apples = 9 ∧
  friday_apples = monday_apples / 2 ∧
  thursday_apples = 4 * friday_apples ∧
  total_apples = monday_apples + tuesday_apples + wednesday_apples + thursday_apples + friday_apples

theorem billy_apples : num_apples_eaten 2 4 9 4 1 20 := 
by
  unfold num_apples_eaten
  sorry

end billy_apples_l64_64092


namespace find_p_l64_64495

-- Define the coordinates of the points
structure Point where
  x : Real
  y : Real

def Q := Point.mk 0 15
def A := Point.mk 3 15
def B := Point.mk 15 0
def O := Point.mk 0 0
def C (p : Real) := Point.mk 0 p

-- Given the area of triangle ABC and the coordinates of Q, A, B, O, and C, prove that p = 12.75
theorem find_p (p : Real) (h_area_ABC : 36 = 36) (h_Q : Q = Point.mk 0 15)
                (h_A : A = Point.mk 3 15) (h_B : B = Point.mk 15 0) 
                (h_O : O = Point.mk 0 0) : p = 12.75 := 
sorry

end find_p_l64_64495


namespace pythagorean_inequality_l64_64503

variables (a b c : ℝ) (n : ℕ)

theorem pythagorean_inequality (h₀ : a > b) (h₁ : b > c) (h₂ : a^2 = b^2 + c^2) (h₃ : n > 2) : a^n > b^n + c^n :=
sorry

end pythagorean_inequality_l64_64503


namespace maximize_product_l64_64250

-- Define 22 as a constant natural number
def N : ℕ := 22

-- Define what it means for sums of distinct natural numbers to sum to 22
def is_valid_split (ns : List ℕ) : Prop :=
  (ns.sum = N) ∧ (∀ (x y : ℕ), x ∈ ns → y ∈ ns → x ≠ y → nat.gcd x y = 1)

-- State the main theorem
theorem maximize_product : ∃ (ns : List ℕ), is_valid_split ns ∧ ns.prod = 1008 :=
sorry  -- proof to be filled

end maximize_product_l64_64250


namespace intersection_A_B_l64_64136

def A : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 3^x}
def B : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2^(-x)}

theorem intersection_A_B :
  A ∩ B = {p | p = (0, 1)} :=
by
  sorry

end intersection_A_B_l64_64136


namespace cuboid_diagonal_cubes_l64_64856

def num_cubes_intersecting_diagonal (a b c : ℕ) : ℕ :=
  a + b + c - 2

theorem cuboid_diagonal_cubes :
  num_cubes_intersecting_diagonal 77 81 100 = 256 :=
by
  sorry

end cuboid_diagonal_cubes_l64_64856


namespace cuboid_volume_l64_64702

/-- Given a cuboid with edges 6 cm, 5 cm, and 6 cm, the volume of the cuboid
    is 180 cm³. -/
theorem cuboid_volume (a b c : ℕ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 6) :
  a * b * c = 180 := by
  sorry

end cuboid_volume_l64_64702


namespace negation_of_universal_proposition_l64_64823
open Classical

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 1 → x^2 ≥ 3)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 < 3) := 
by
  sorry

end negation_of_universal_proposition_l64_64823


namespace rationalize_sqrt_fraction_l64_64996

theorem rationalize_sqrt_fraction {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) : 
  (Real.sqrt ((a : ℝ) / b)) = (Real.sqrt (a * (b / (b * b)))) → 
  (Real.sqrt (5 / 12)) = (Real.sqrt 15 / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l64_64996


namespace line_does_not_pass_through_third_quadrant_l64_64614

variable {a b c : ℝ}

theorem line_does_not_pass_through_third_quadrant
  (hac : a * c < 0) (hbc : b * c < 0) : ¬ ∃ x y, x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0 :=
sorry

end line_does_not_pass_through_third_quadrant_l64_64614


namespace find_positive_real_numbers_l64_64886

open Real

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16

theorem find_positive_real_numbers (x : ℝ) (hx : x > 0) :
  satisfies_inequality x ↔ 15 * x^2 + 32 * x - 256 = 0 :=
sorry

end find_positive_real_numbers_l64_64886


namespace total_animals_is_63_l64_64274

def zoo_animals (penguins polar_bears total : ℕ) : Prop :=
  (penguins = 21) ∧
  (polar_bears = 2 * penguins) ∧
  (total = penguins + polar_bears)

theorem total_animals_is_63 :
  ∃ (penguins polar_bears total : ℕ), zoo_animals penguins polar_bears total ∧ total = 63 :=
by {
  sorry
}

end total_animals_is_63_l64_64274


namespace eventually_periodic_of_rational_cubic_l64_64634

noncomputable def is_rational_sequence (P : ℚ → ℚ) (q : ℕ → ℚ) :=
  ∀ n : ℕ, q (n + 1) = P (q n)

theorem eventually_periodic_of_rational_cubic (P : ℚ → ℚ) (q : ℕ → ℚ) (hP : ∃ a b c d : ℚ, ∀ x : ℚ, P x = a * x^3 + b * x^2 + c * x + d) (hq : is_rational_sequence P q) : 
  ∃ k ≥ 1, ∀ n ≥ 1, q (n + k) = q n := 
sorry

end eventually_periodic_of_rational_cubic_l64_64634


namespace constant_term_is_21_l64_64375

def poly1 (x : ℕ) := x^3 + x^2 + 3
def poly2 (x : ℕ) := 2*x^4 + x^2 + 7
def expanded_poly (x : ℕ) := poly1 x * poly2 x

theorem constant_term_is_21 : expanded_poly 0 = 21 := by
  sorry

end constant_term_is_21_l64_64375


namespace sum_of_roots_l64_64287

-- sum of roots of first polynomial
def S1 : ℚ := -(-6 / 3)

-- sum of roots of second polynomial
def S2 : ℚ := -(8 / 4)

-- proof statement
theorem sum_of_roots : S1 + S2 = 0 :=
by
  -- placeholders
  sorry

end sum_of_roots_l64_64287


namespace surface_area_original_cube_l64_64709

theorem surface_area_original_cube
  (n : ℕ)
  (edge_length_smaller : ℕ)
  (smaller_cubes : ℕ)
  (original_surface_area : ℕ)
  (h1 : n = 3)
  (h2 : edge_length_smaller = 4)
  (h3 : smaller_cubes = 27)
  (h4 : 6 * (n * edge_length_smaller) ^ 2 = original_surface_area) :
  original_surface_area = 864 := by
  sorry

end surface_area_original_cube_l64_64709


namespace pascal_triangle_contains_53_once_l64_64464

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l64_64464


namespace y_intercept_of_line_l64_64032

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end y_intercept_of_line_l64_64032


namespace percentage_A_to_B_l64_64024

variable (A B : ℕ)
variable (total : ℕ := 570)
variable (B_amount : ℕ := 228)

theorem percentage_A_to_B :
  (A + B = total) →
  B = B_amount →
  (A = total - B_amount) →
  ((A / B_amount : ℚ) * 100 = 150) :=
sorry

end percentage_A_to_B_l64_64024


namespace total_animals_is_63_l64_64275

def zoo_animals (penguins polar_bears total : ℕ) : Prop :=
  (penguins = 21) ∧
  (polar_bears = 2 * penguins) ∧
  (total = penguins + polar_bears)

theorem total_animals_is_63 :
  ∃ (penguins polar_bears total : ℕ), zoo_animals penguins polar_bears total ∧ total = 63 :=
by {
  sorry
}

end total_animals_is_63_l64_64275


namespace find_sides_of_rectangle_l64_64396

-- Define the conditions
def isRectangle (w l : ℝ) : Prop :=
  l = 3 * w ∧ 2 * l + 2 * w = l * w

-- Main theorem statement
theorem find_sides_of_rectangle (w l : ℝ) :
  isRectangle w l → w = 8 / 3 ∧ l = 8 :=
by
  sorry

end find_sides_of_rectangle_l64_64396


namespace Chandler_more_rolls_needed_l64_64893

theorem Chandler_more_rolls_needed :
  let total_goal := 12
  let sold_to_grandmother := 3
  let sold_to_uncle := 4
  let sold_to_neighbor := 3
  let total_sold := sold_to_grandmother + sold_to_uncle + sold_to_neighbor
  total_goal - total_sold = 2 :=
by
  sorry

end Chandler_more_rolls_needed_l64_64893


namespace remainder_div_101_l64_64695

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l64_64695


namespace a_beats_b_by_4_rounds_l64_64843

variable (T_a T_b : ℝ)
variable (race_duration : ℝ) -- duration of the 4-round race in minutes
variable (time_difference : ℝ) -- Time that a beats b by in the 4-round race

open Real

-- Given conditions
def conditions :=
  (T_a = 7.5) ∧                             -- a's time to complete one round
  (race_duration = T_a * 4 + 10) ∧          -- a beats b by 10 minutes in a 4-round race
  (time_difference = T_b - T_a)             -- The time difference per round is T_b - T_a

-- Mathematical proof statement
theorem a_beats_b_by_4_rounds
  (h : conditions T_a T_b race_duration time_difference) :
  10 / time_difference = 4 := by
  sorry

end a_beats_b_by_4_rounds_l64_64843


namespace smallest_digits_to_append_l64_64212

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l64_64212


namespace most_likely_outcomes_l64_64587

open ProbabilityTheory

noncomputable def probability {n : ℕ} (k : ℕ) : ℝ :=
  (nat.choose n k) * (1 / 2) ^ n

theorem most_likely_outcomes :
  let n := 5,
      pA := probability n 0,
      pB := probability n 0,
      pC := probability n 3,
      pD := probability n 2,
      pE := 2 * probability n 1 in
  pC = pD ∧ pD = pE ∧ pC > pA ∧ pA = pB
:=
by
  sorry

end most_likely_outcomes_l64_64587


namespace determine_number_of_quarters_l64_64401

def number_of_coins (Q D : ℕ) : Prop := Q + D = 23

def total_value (Q D : ℕ) : Prop := 25 * Q + 10 * D = 335

theorem determine_number_of_quarters (Q D : ℕ) 
  (h1 : number_of_coins Q D) 
  (h2 : total_value Q D) : 
  Q = 7 :=
by
  -- Equating and simplifying using h2, we find 15Q = 105, hence Q = 7
  sorry

end determine_number_of_quarters_l64_64401


namespace count_even_three_digit_numbers_with_sum_12_l64_64936

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l64_64936


namespace no_such_function_exists_l64_64574

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = Real.sin x :=
by
  sorry

end no_such_function_exists_l64_64574


namespace largest_expression_value_l64_64379

-- Definitions of the expressions
def expr_A : ℕ := 3 + 0 + 1 + 8
def expr_B : ℕ := 3 * 0 + 1 + 8
def expr_C : ℕ := 3 + 0 * 1 + 8
def expr_D : ℕ := 3 + 0 + 1^2 + 8
def expr_E : ℕ := 3 * 0 * 1^2 * 8

-- Statement of the theorem
theorem largest_expression_value :
  max expr_A (max expr_B (max expr_C (max expr_D expr_E))) = 12 :=
by
  sorry

end largest_expression_value_l64_64379


namespace purely_imaginary_complex_number_l64_64122

theorem purely_imaginary_complex_number (a : ℝ) (i : ℂ)
  (h₁ : i * i = -1)
  (h₂ : ∃ z : ℂ, z = (a + i) / (1 - i) ∧ z.im ≠ 0 ∧ z.re = 0) :
  a = 1 :=
sorry

end purely_imaginary_complex_number_l64_64122


namespace left_handed_like_jazz_l64_64195

theorem left_handed_like_jazz (total_people left_handed like_jazz right_handed_dislike_jazz : ℕ)
    (h1 : total_people = 30)
    (h2 : left_handed = 12)
    (h3 : like_jazz = 20)
    (h4 : right_handed_dislike_jazz = 3)
    (h5 : ∀ p, p = total_people - left_handed ∧ p = total_people - (left_handed + right_handed_dislike_jazz)) :
    ∃ x, x = 5 := by
  sorry

end left_handed_like_jazz_l64_64195


namespace decipher_proof_l64_64547

noncomputable def decipher_message (n : ℕ) (hidden_message : String) :=
  if n = 2211169691162 then hidden_message = "Kiss me, dearest" else false

theorem decipher_proof :
  decipher_message 2211169691162 "Kiss me, dearest" = true :=
by
  -- Proof skipped
  sorry

end decipher_proof_l64_64547


namespace cricket_run_rate_l64_64623

theorem cricket_run_rate (r : ℝ) (o₁ T o₂ : ℕ) (r₁ : ℝ) (Rₜ : ℝ) : 
  r = 4.8 ∧ o₁ = 10 ∧ T = 282 ∧ o₂ = 40 ∧ r₁ = (T - r * o₁) / o₂ → Rₜ = 5.85 := 
by 
  intros h
  sorry

end cricket_run_rate_l64_64623


namespace days_to_complete_work_l64_64482

theorem days_to_complete_work
  (M B W : ℝ)  -- Define variables for daily work done by a man, a boy, and the total work
  (hM : M = 2 * B)  -- Condition: daily work done by a man is twice that of a boy
  (hW : (13 * M + 24 * B) * 4 = W)  -- Condition: 13 men and 24 boys complete work in 4 days
  (H : 12 * M + 16 * B) -- Help Lean infer the first group's total work per day
  (hW2 : (12 * M + 16 * B) * 5 = W)  -- Condition: first group must complete work in same time (5 days, to be proven)
  : (12 * M + 16 * B) * 5 = W := -- Prove equivalence
sorry

end days_to_complete_work_l64_64482


namespace math_problem_l64_64611

theorem math_problem (a b c d m : ℝ) (h1 : a = -b) (h2 : a ≠ 0) (h3 : c * d = 1)
  (h4 : m = -1 ∨ m = 3) : (a + b) * (c / d) + m * c * d + (b / a) = 2 ∨ (a + b) * (c / d) + m * c * d + (b / a) = -2 :=
by
  sorry

end math_problem_l64_64611


namespace projectile_reaches_40_at_first_time_l64_64654

theorem projectile_reaches_40_at_first_time : ∃ t : ℝ, 0 < t ∧ (40 = -16 * t^2 + 64 * t) ∧ (∀ t' : ℝ, 0 < t' ∧ t' < t → ¬ (40 = -16 * t'^2 + 64 * t')) ∧ t = 0.8 :=
by
  sorry

end projectile_reaches_40_at_first_time_l64_64654


namespace pascal_triangle_contains_53_l64_64468

theorem pascal_triangle_contains_53:
  ∃! n, ∃ k, (n ≥ 0) ∧ (k ≥ 0) ∧ (binom n k = 53) := 
sorry

end pascal_triangle_contains_53_l64_64468


namespace silvia_savings_l64_64169

def retail_price : ℝ := 1000
def guitar_center_discount_rate : ℝ := 0.15
def sweetwater_discount_rate : ℝ := 0.10
def guitar_center_shipping_fee : ℝ := 100
def sweetwater_shipping_fee : ℝ := 0

def guitar_center_cost : ℝ := retail_price * (1 - guitar_center_discount_rate) + guitar_center_shipping_fee
def sweetwater_cost : ℝ := retail_price * (1 - sweetwater_discount_rate) + sweetwater_shipping_fee

theorem silvia_savings : guitar_center_cost - sweetwater_cost = 50 := by
  sorry

end silvia_savings_l64_64169


namespace geometric_sequence_b_value_l64_64754

theorem geometric_sequence_b_value (a b c : ℝ) (h : 1 * a = a * b ∧ a * b = b * c ∧ b * c = c * 5) : b = Real.sqrt 5 :=
sorry

end geometric_sequence_b_value_l64_64754


namespace wuyang_volleyball_team_members_l64_64647

theorem wuyang_volleyball_team_members :
  (Finset.filter Nat.Prime (Finset.range 50)).card = 15 :=
by
  sorry

end wuyang_volleyball_team_members_l64_64647


namespace distinct_triangle_areas_l64_64631

variables (A B C D E F G : ℝ) (h : ℝ)
variables (AB BC CD EF FG AC BD AD EG : ℝ)

def is_valid_points := AB = 2 ∧ BC = 1 ∧ CD = 3 ∧ EF = 1 ∧ FG = 2 ∧ AC = AB + BC ∧ BD = BC + CD ∧ AD = AB + BC + CD ∧ EG = EF + FG

theorem distinct_triangle_areas (h_pos : 0 < h) (valid : is_valid_points AB BC CD EF FG AC BD AD EG) : 
  ∃ n : ℕ, n = 5 := 
by
  sorry

end distinct_triangle_areas_l64_64631


namespace remainder_when_sum_divided_by_7_l64_64940

theorem remainder_when_sum_divided_by_7 (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : a * b * c ≡ 1 [MOD 7])
  (h2 : 4 * c ≡ 3 [MOD 7])
  (h3 : 5 * b ≡ 4 + b [MOD 7]) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_when_sum_divided_by_7_l64_64940


namespace calculate_expression_l64_64416

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end calculate_expression_l64_64416


namespace remainder_mod_68_l64_64432

theorem remainder_mod_68 (n : ℕ) (h : 67^67 + 67 ≡ 66 [MOD n]) : n = 68 := 
by 
  sorry

end remainder_mod_68_l64_64432


namespace exist_non_special_symmetric_concat_l64_64299

-- Define the notion of a binary series being symmetric
def is_symmetric (xs : List Bool) : Prop :=
  ∀ i, i < xs.length → xs.get? i = xs.get? (xs.length - 1 - i)

-- Define the notion of a binary series being special
def is_special (xs : List Bool) : Prop :=
  (∀ x ∈ xs, x) ∨ (∀ x ∈ xs, ¬x)

-- The main theorem statement
theorem exist_non_special_symmetric_concat (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  ∃ (A B : List Bool), A.length = m ∧ B.length = n ∧ ¬is_special A ∧ ¬is_special B ∧ is_symmetric (A ++ B) :=
sorry

end exist_non_special_symmetric_concat_l64_64299


namespace calculate_expression_l64_64415

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end calculate_expression_l64_64415


namespace evaluate_expression_l64_64581

theorem evaluate_expression :
  2003^3 - 2002 * 2003^2 - 2002^2 * 2003 + 2002^3 = 4005 :=
by
  sorry

end evaluate_expression_l64_64581


namespace children_ticket_price_l64_64862

theorem children_ticket_price
  (C : ℝ)
  (adult_ticket_price : ℝ)
  (total_payment : ℝ)
  (total_tickets : ℕ)
  (children_tickets : ℕ)
  (H1 : adult_ticket_price = 8)
  (H2 : total_payment = 201)
  (H3 : total_tickets = 33)
  (H4 : children_tickets = 21)
  : C = 5 :=
by
  sorry

end children_ticket_price_l64_64862


namespace probability_diagonals_intersect_l64_64770

theorem probability_diagonals_intersect (n : ℕ) : 
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  probability = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  sorry

end probability_diagonals_intersect_l64_64770


namespace min_value_x_plus_one_over_x_plus_two_l64_64298

theorem min_value_x_plus_one_over_x_plus_two (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x + 1/(x + 2) ∧ y ≥ 0 :=
by
  sorry

end min_value_x_plus_one_over_x_plus_two_l64_64298


namespace exam_score_probability_l64_64191

open ProbabilityTheory

noncomputable def num_people_scoring_at_least_139 
  (num_people : ℝ) (μ : ℝ) (σ : ℝ) (p_interval : ℝ) :=
  let probability_ge_139 := (1 - p_interval) / 2 in
  num_people * probability_ge_139

theorem exam_score_probability
  (num_people : ℝ)
  (μ σ : ℝ)
  (p_interval : ℝ)
  (h : p_interval = 0.997)
  (h_norm : ∀ X, X ~ Normal μ σ^2) :
  num_people_scoring_at_least_139 num_people μ σ p_interval = 15 :=
by
  dsimp [num_people_scoring_at_least_139]
  rw [h]
  norm_num
  apply mul_eq_mul_right_iff.mpr
  right
  norm_num
  sorry

end exam_score_probability_l64_64191


namespace find_positive_real_numbers_l64_64885

open Real

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16

theorem find_positive_real_numbers (x : ℝ) (hx : x > 0) :
  satisfies_inequality x ↔ 15 * x^2 + 32 * x - 256 = 0 :=
sorry

end find_positive_real_numbers_l64_64885


namespace evaluate_cube_root_power_l64_64580

theorem evaluate_cube_root_power (a : ℝ) (b : ℝ) (c : ℝ) (h : a = b^(3 : ℝ)) : (cbrt a)^12 = b^12 :=
by
  sorry

example : evaluate_cube_root_power 8 2 4096 (by rfl)

end evaluate_cube_root_power_l64_64580


namespace package_cheaper_than_per_person_l64_64672

theorem package_cheaper_than_per_person (x : ℕ) :
  (90 * 6 + 10 * x < 54 * x + 8 * 3 * x) ↔ x ≥ 8 :=
by
  sorry

end package_cheaper_than_per_person_l64_64672


namespace solve_eq1_solve_eq2_l64_64011

theorem solve_eq1 (x : ℝ) : 3 * x * (x + 3) = 2 * (x + 3) ↔ (x = -3 ∨ x = 2 / 3) :=
by sorry

theorem solve_eq2 (x : ℝ) : x^2 - 4 * x - 5 = 0 ↔ (x = 5 ∨ x = -1) :=
by sorry

end solve_eq1_solve_eq2_l64_64011


namespace number_of_integer_values_l64_64502

def Q (x : ℤ) : ℤ := x^4 + 4 * x^3 + 9 * x^2 + 2 * x + 17

theorem number_of_integer_values :
  (∃ xs : List ℤ, xs.length = 4 ∧ ∀ x ∈ xs, Nat.Prime (Int.natAbs (Q x))) :=
by
  sorry

end number_of_integer_values_l64_64502


namespace part_1_part_2_l64_64598

theorem part_1 (a b A B : ℝ)
  (h : b * (Real.sin A)^2 = Real.sqrt 3 * a * Real.cos A * Real.sin B) 
  (h_sine_law : b / Real.sin B = a / Real.sin A)
  (A_in_range: A ∈ Set.Ioo 0 Real.pi):
  A = Real.pi / 3 := 
sorry

theorem part_2 (x : ℝ)
  (A : ℝ := Real.pi / 3)
  (h_sin_cos : ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
                f x = (Real.sin A * (Real.cos x)^2) - (Real.sin (A / 2))^2 * (Real.sin (2 * x))) :
  Set.image f (Set.Icc 0 (Real.pi / 2)) = Set.Icc ((Real.sqrt 3 - 2)/4) (Real.sqrt 3 / 2) :=
sorry

end part_1_part_2_l64_64598


namespace min_ab_min_inv_a_plus_2_inv_b_max_sqrt_2a_plus_sqrt_b_not_max_a_plus_1_times_b_plus_1_l64_64294

-- Condition definitions
variable {a b : ℝ}
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1)

-- Minimum value of ab is 1/8
theorem min_ab (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (a * b) ∧ y = 1 / 8 := by
  sorry

-- Minimum value of 1/a + 2/b is 8
theorem min_inv_a_plus_2_inv_b (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (1 / a + 2 / b) ∧ y = 8 := by
  sorry

-- Maximum value of sqrt(2a) + sqrt(b) is sqrt(2)
theorem max_sqrt_2a_plus_sqrt_b (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (Real.sqrt (2 * a) + Real.sqrt b) ∧ y = Real.sqrt 2 := by
  sorry

-- Maximum value of (a+1)(b+1) is not 2
theorem not_max_a_plus_1_times_b_plus_1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = ((a + 1) * (b + 1)) ∧ y ≠ 2 := by
  sorry


end min_ab_min_inv_a_plus_2_inv_b_max_sqrt_2a_plus_sqrt_b_not_max_a_plus_1_times_b_plus_1_l64_64294


namespace common_tangent_at_point_l64_64946

theorem common_tangent_at_point (x₀ b : ℝ) 
  (h₁ : 6 * x₀^2 = 6 * x₀) 
  (h₂ : 1 + 2 * x₀^3 = 3 * x₀^2 - b) :
  b = 0 ∨ b = -1 :=
sorry

end common_tangent_at_point_l64_64946


namespace train_length_is_correct_l64_64845

noncomputable def train_speed_kmh : ℝ := 40
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (5 / 18)
noncomputable def cross_time : ℝ := 25.2
noncomputable def train_length : ℝ := train_speed_ms * cross_time

theorem train_length_is_correct : train_length = 280.392 := by
  sorry

end train_length_is_correct_l64_64845


namespace mixed_operations_with_rationals_l64_64873

theorem mixed_operations_with_rationals :
  let a := 1 / 4
  let b := 1 / 2
  let c := 2 / 3
  (a - b + c) * (-12) = -8 :=
by
  sorry

end mixed_operations_with_rationals_l64_64873


namespace count_even_three_digit_sum_tens_units_is_12_l64_64927

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l64_64927


namespace average_marks_l64_64371

/-- Given that the total marks in physics, chemistry, and mathematics is 110 more than the marks obtained in physics. -/
theorem average_marks (P C M : ℕ) (h : P + C + M = P + 110) : (C + M) / 2 = 55 :=
by
  -- The proof goes here.
  sorry

end average_marks_l64_64371


namespace max_parts_from_blanks_l64_64321

-- Define the initial conditions and question as constants
constant initial_blanks : ℕ := 20
constant usage_fraction : ℚ := 2 / 3
constant waste_fraction : ℚ := 1 / 3

-- Define the maximum number of parts that can be produced and the remaining waste
constant max_parts_produced : ℕ := 29
constant remaining_waste : ℚ := 1 / 3

-- State the main theorem
theorem max_parts_from_blanks :
  (initial_blanks → (usage_fraction = 2 / 3) → (waste_fraction = 1 / 3) → max_parts_produced = 29 ∧ remaining_waste = 1 / 3) :=
  sorry

end max_parts_from_blanks_l64_64321


namespace exists_N_minimal_l64_64512

-- Assuming m and n are positive and coprime
variables (m n : ℕ)
variables (h_pos_m : 0 < m) (h_pos_n : 0 < n)
variables (h_coprime : Nat.gcd m n = 1)

-- Statement of the mathematical problem
theorem exists_N_minimal :
  ∃ N : ℕ, (∀ k : ℕ, k ≥ N → ∃ a b : ℕ, k = a * m + b * n) ∧
           (N = m * n - m - n + 1) := 
  sorry

end exists_N_minimal_l64_64512


namespace max_sum_is_1717_l64_64484

noncomputable def max_arithmetic_sum (a d : ℤ) : ℤ :=
  let n := 34
  let S : ℤ := n * (2*a + (n - 1)*d) / 2
  S

theorem max_sum_is_1717 (a d : ℤ) (h1 : a + 16 * d = 52) (h2 : a + 29 * d = 13) (hd : d = -3) (ha : a = 100) :
  max_arithmetic_sum a d = 1717 :=
by
  unfold max_arithmetic_sum
  rw [hd, ha]
  -- Add the necessary steps to prove max_arithmetic_sum 100 (-3) = 1717
  -- Sorry ensures the theorem can be checked syntactically without proof
  sorry

end max_sum_is_1717_l64_64484


namespace age_of_boy_not_included_l64_64816

theorem age_of_boy_not_included (average_age_11_boys : ℕ) (average_age_first_6 : ℕ) (average_age_last_6 : ℕ) 
(first_6_sum : ℕ) (last_6_sum : ℕ) (total_sum : ℕ) (X : ℕ):
  average_age_11_boys = 50 ∧ average_age_first_6 = 49 ∧ average_age_last_6 = 52 ∧ 
  first_6_sum = 6 * average_age_first_6 ∧ last_6_sum = 6 * average_age_last_6 ∧ 
  total_sum = 11 * average_age_11_boys ∧ first_6_sum + last_6_sum - X = total_sum →
  X = 56 :=
by
  sorry

end age_of_boy_not_included_l64_64816


namespace ages_of_boys_l64_64187

theorem ages_of_boys (a b c : ℕ) (h : a + b + c = 29) (h₁ : a = b) (h₂ : c = 11) : a = 9 ∧ b = 9 := 
by
  sorry

end ages_of_boys_l64_64187


namespace Jill_talking_time_total_l64_64626

-- Definition of the sequence of talking times
def talking_time : ℕ → ℕ 
| 0 => 5
| (n+1) => 2 * talking_time n

-- The statement we need to prove
theorem Jill_talking_time_total :
  (talking_time 0) + (talking_time 1) + (talking_time 2) + (talking_time 3) + (talking_time 4) = 155 :=
by
  sorry

end Jill_talking_time_total_l64_64626


namespace at_least_one_not_lt_one_l64_64835

theorem at_least_one_not_lt_one (a b c : ℝ) (h : a + b + c = 3) : ¬ (a < 1 ∧ b < 1 ∧ c < 1) :=
by
  sorry

end at_least_one_not_lt_one_l64_64835


namespace find_x_plus_inv_x_l64_64128

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + x⁻³ = 110) : x + x⁻¹ = 5 :=
sorry

end find_x_plus_inv_x_l64_64128


namespace remainder_div_101_l64_64696

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l64_64696


namespace probability_sum_is_odd_l64_64071

-- Define the problem
noncomputable def probability_odd_sum (n k : ℕ) (balls : Fin n) : ℝ :=
  let odd_balls := (Finset.filter (λ x : Fin n, ↑x % 2 = 1) Finset.univ).card
  let even_balls := (Finset.filter (λ x : Fin n, ↑x % 2 = 0) Finset.univ).card
  let favorable_cases := ((Finset.card (Finset.filter (λ x, ↑x % 2 = 1) (Finset.range 12).choose 5)) *
                          (Finset.card (Finset.filter (λ x, ↑x % 2 = 0) (Finset.range 12).choose 2))
                          + (Finset.card (Finset.filter (λ x, ↑x % 2 = 1) (Finset.range 12).choose 3)) *
                          (Finset.card (Finset.filter (λ x, ↑x % 2 = 0) (Finset.range 12).choose 4))
                          + (Finset.card (Finset.filter (λ x, ↑x % 2 = 1) (Finset.range 12).choose 1)) *
                          (Finset.card (Finset.filter (λ x, ↑x % 2 = 0) (Finset.range 12).choose 6)))
                          .to_nat
  let total_outcomes := (Finset.range 12).choose 7
  favorable_cases / total_outcomes.to_nat

theorem probability_sum_is_odd (n k : ℕ) (balls : Fin n) :
  probability_odd_sum 12 7 balls = 1/2 := by
  -- Provide conditions and skipped proof
  sorry

end probability_sum_is_odd_l64_64071


namespace jackson_total_souvenirs_l64_64153

-- Define the conditions
def num_hermit_crabs : ℕ := 45
def spiral_shells_per_hermit_crab : ℕ := 3
def starfish_per_spiral_shell : ℕ := 2

-- Define the calculation based on the conditions
def num_spiral_shells := num_hermit_crabs * spiral_shells_per_hermit_crab
def num_starfish := num_spiral_shells * starfish_per_spiral_shell
def total_souvenirs := num_hermit_crabs + num_spiral_shells + num_starfish

-- Prove that the total number of souvenirs is 450
theorem jackson_total_souvenirs : total_souvenirs = 450 :=
by
  sorry

end jackson_total_souvenirs_l64_64153


namespace compare_P_Q_l64_64445

noncomputable def P : ℝ := Real.sqrt 7 - 1
noncomputable def Q : ℝ := Real.sqrt 11 - Real.sqrt 5

theorem compare_P_Q : P > Q :=
sorry

end compare_P_Q_l64_64445


namespace average_speed_whole_journey_l64_64703

theorem average_speed_whole_journey (D : ℝ) (h₁ : D > 0) :
  let T1 := D / 54
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  let V_avg := total_distance / total_time
  V_avg = 64.8 :=
by
  sorry

end average_speed_whole_journey_l64_64703


namespace series_divergence_l64_64630

theorem series_divergence (a : ℕ → ℝ) (hdiv : ¬ ∃ l, ∑' n, a n = l) (hpos : ∀ n, a n > 0) (hnoninc : ∀ n m, n ≤ m → a m ≤ a n) : 
  ¬ ∃ l, ∑' n, (a n / (1 + n * a n)) = l :=
by
  sorry

end series_divergence_l64_64630


namespace bobs_share_l64_64993

theorem bobs_share 
  (r : ℕ → ℕ → ℕ → Prop) (s : ℕ) 
  (h_ratio : r 1 2 3) 
  (bill_share : s = 300) 
  (hr : ∃ p, s = 2 * p) :
  ∃ b, b = 3 * (s / 2) ∧ b = 450 := 
by
  sorry

end bobs_share_l64_64993


namespace pascal_triangle_contains_53_l64_64467

theorem pascal_triangle_contains_53:
  ∃! n, ∃ k, (n ≥ 0) ∧ (k ≥ 0) ∧ (binom n k = 53) := 
sorry

end pascal_triangle_contains_53_l64_64467


namespace geometric_series_common_ratio_l64_64089

theorem geometric_series_common_ratio (a S r : ℝ) (h1 : a = 512) (h2 : S = 3072) 
(h3 : S = a / (1 - r)) : r = 5/6 := 
sorry

end geometric_series_common_ratio_l64_64089


namespace smallest_number_append_l64_64224

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l64_64224


namespace express_2_175_billion_in_scientific_notation_l64_64987

-- Definition of scientific notation
def scientific_notation (a : ℝ) (n : ℤ) (value : ℝ) : Prop :=
  value = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

-- Theorem stating the problem
theorem express_2_175_billion_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), scientific_notation a n 2.175e9 ∧ a = 2.175 ∧ n = 9 :=
by
  sorry

end express_2_175_billion_in_scientific_notation_l64_64987


namespace select_test_point_l64_64258

theorem select_test_point (x1 x2 : ℝ) (h1 : x1 = 2 + 0.618 * (4 - 2)) (h2 : x2 = 2 + 4 - x1) :
  (x1 > x2 → x3 = 4 - 0.618 * (4 - x1)) ∨ (x1 < x2 → x3 = 6 - x3) :=
  sorry

end select_test_point_l64_64258


namespace probability_diagonals_intersect_l64_64771

theorem probability_diagonals_intersect (n : ℕ) : 
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  probability = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  sorry

end probability_diagonals_intersect_l64_64771


namespace find_m_l64_64101

-- Define the operation a * b
def star (a b : ℝ) : ℝ := a * b + a - 2 * b

theorem find_m (m : ℝ) (h : star 3 m = 17) : m = 14 :=
by
  -- Placeholder for the proof
  sorry

end find_m_l64_64101


namespace fourth_term_geom_progression_l64_64874

theorem fourth_term_geom_progression : 
  ∀ (a b c : ℝ), 
    a = 4^(1/2) → 
    b = 4^(1/3) → 
    c = 4^(1/6) → 
    ∃ d : ℝ, d = 1 ∧ b / a = c / b ∧ c / b = 4^(1/6) / 4^(1/3) :=
by
  sorry

end fourth_term_geom_progression_l64_64874


namespace shiela_paintings_l64_64090

theorem shiela_paintings (h1 : 18 % 2 = 0) : 18 / 2 = 9 := 
by sorry

end shiela_paintings_l64_64090


namespace size_of_third_file_l64_64572

theorem size_of_third_file 
  (s : ℝ) (t : ℝ) (f1 : ℝ) (f2 : ℝ) (f3 : ℝ) 
  (h1 : s = 2) (h2 : t = 120) (h3 : f1 = 80) (h4 : f2 = 90) : 
  f3 = s * t - (f1 + f2) :=
by
  sorry

end size_of_third_file_l64_64572


namespace problem_solution_l64_64378

theorem problem_solution :
  3 ^ (0 ^ (2 ^ 2)) + ((3 ^ 1) ^ 0) ^ 2 = 2 :=
by
  sorry

end problem_solution_l64_64378


namespace number_of_bugs_seen_l64_64506

-- Defining the conditions
def flowers_per_bug : ℕ := 2
def total_flowers_eaten : ℕ := 6

-- The statement to prove
theorem number_of_bugs_seen : total_flowers_eaten / flowers_per_bug = 3 :=
by
  sorry

end number_of_bugs_seen_l64_64506


namespace remainder_of_large_number_l64_64689

theorem remainder_of_large_number : 
  (9876543210 : ℤ) % 101 = 73 := 
by
  unfold_coes
  unfold_norm_num
  sorry

end remainder_of_large_number_l64_64689


namespace coefficient_j_l64_64282

theorem coefficient_j (j k : ℝ) (p : Polynomial ℝ) (h : p = Polynomial.C 400 + Polynomial.X * Polynomial.C k + Polynomial.X^2 * Polynomial.C j + Polynomial.X^4) :
  (∃ a d : ℝ, (d ≠ 0) ∧ (0 > (4*a + 6*d)) ∧ (p.eval a = 0) ∧ (p.eval (a + d) = 0) ∧ (p.eval (a + 2*d) = 0) ∧ (p.eval (a + 3*d) = 0)) → 
  j = -40 :=
by
  sorry

end coefficient_j_l64_64282


namespace problem_solution_l64_64651

theorem problem_solution : ∃ n : ℕ, (n > 0) ∧ (21 - 3 * n > 15) ∧ (∀ m : ℕ, (m > 0) ∧ (21 - 3 * m > 15) → m = n) :=
by
  sorry

end problem_solution_l64_64651


namespace interest_after_4_years_l64_64406
-- Importing the necessary library

-- Definitions based on the conditions
def initial_amount : ℝ := 1500
def annual_interest_rate : ℝ := 0.12
def number_of_years : ℕ := 4

-- Calculating the total amount after 4 years using compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- Calculating the interest earned
def interest_earned : ℝ :=
  compound_interest initial_amount annual_interest_rate number_of_years - initial_amount

-- The Lean statement to prove the interest earned is $859.25
theorem interest_after_4_years : interest_earned = 859.25 :=
by
  sorry

end interest_after_4_years_l64_64406


namespace union_M_N_l64_64504

def M : Set ℝ := { x | x^2 + 2 * x = 0 }
def N : Set ℝ := { x | x^2 - 2 * x = 0 }

theorem union_M_N : M ∪ N = {0, -2, 2} := by
  sorry

end union_M_N_l64_64504


namespace at_least_one_less_than_zero_l64_64615

theorem at_least_one_less_than_zero {a b : ℝ} (h: a + b < 0) : a < 0 ∨ b < 0 := 
by 
  sorry

end at_least_one_less_than_zero_l64_64615


namespace y_intercept_3x_minus_4y_eq_12_l64_64037

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end y_intercept_3x_minus_4y_eq_12_l64_64037


namespace sequence_is_geometric_l64_64764

theorem sequence_is_geometric {a : ℝ} (h : a ≠ 0) (S : ℕ → ℝ) (H : ∀ n, S n = a^n - 1) 
: ∃ r, ∀ n, (n ≥ 1) → S n - S (n-1) = r * (S (n-1) - S (n-2)) :=
sorry

end sequence_is_geometric_l64_64764


namespace smallest_number_is_C_l64_64196

-- Define the conditions
def A := 18 + 38
def B := A - 26
def C := B / 3

-- Proof statement: C is the smallest number among A, B, and C
theorem smallest_number_is_C : C = min A (min B C) :=
by
  sorry

end smallest_number_is_C_l64_64196


namespace AmandaWillSpend_l64_64085

/--
Amanda goes shopping and sees a sale where different items have different discounts.
She wants to buy a dress for $50 with a 30% discount, a pair of shoes for $75 with a 25% discount,
and a handbag for $100 with a 40% discount.
After applying the discounts, a 5% tax is added to the final price.
Prove that Amanda will spend $158.81 to buy all three items after the discounts and tax have been applied.
-/
noncomputable def totalAmount : ℝ :=
  let dressPrice := 50
  let dressDiscount := 0.30
  let shoesPrice := 75
  let shoesDiscount := 0.25
  let handbagPrice := 100
  let handbagDiscount := 0.40
  let taxRate := 0.05
  let dressFinalPrice := dressPrice * (1 - dressDiscount)
  let shoesFinalPrice := shoesPrice * (1 - shoesDiscount)
  let handbagFinalPrice := handbagPrice * (1 - handbagDiscount)
  let subtotal := dressFinalPrice + shoesFinalPrice + handbagFinalPrice
  let tax := subtotal * taxRate
  let totalAmount := subtotal + tax
  totalAmount

theorem AmandaWillSpend : totalAmount = 158.81 :=
by
  -- proof goes here
  sorry

end AmandaWillSpend_l64_64085


namespace recipe_calls_for_eight_cups_of_sugar_l64_64348

def cups_of_flour : ℕ := 6
def cups_of_salt : ℕ := 7
def additional_sugar_needed (salt : ℕ) : ℕ := salt + 1

theorem recipe_calls_for_eight_cups_of_sugar :
  additional_sugar_needed cups_of_salt = 8 :=
by
  -- condition 1: cups_of_flour = 6
  -- condition 2: cups_of_salt = 7
  -- condition 4: additional_sugar_needed = salt + 1
  -- prove formula: 7 + 1 = 8
  sorry

end recipe_calls_for_eight_cups_of_sugar_l64_64348


namespace polynomial_solution_characterization_l64_64880

theorem polynomial_solution_characterization (P : ℝ → ℝ → ℝ) (h : ∀ x y z : ℝ, P x (2 * y * z) + P y (2 * z * x) + P z (2 * x * y) = P (x + y + z) (x * y + y * z + z * x)) :
  ∃ (a b : ℝ), ∀ x y : ℝ, P x y = a * x + b * (x^2 + 2 * y) :=
sorry

end polynomial_solution_characterization_l64_64880


namespace trigonometric_identity_l64_64312

theorem trigonometric_identity (θ : ℝ) (h : Real.tan (θ + Real.pi / 4) = 2) : 
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = -2 := 
sorry

end trigonometric_identity_l64_64312


namespace quadratic_equality_l64_64609

theorem quadratic_equality (x : ℝ) 
  (h : 14*x + 5 - 21*x^2 = -2) : 
  6*x^2 - 4*x + 5 = 7 := 
by
  sorry

end quadratic_equality_l64_64609


namespace waiter_net_earning_l64_64718

theorem waiter_net_earning (c1 c2 c3 m : ℤ) (h1 : c1 = 3) (h2 : c2 = 2) (h3 : c3 = 1) (t1 t2 t3 : ℤ) (h4 : t1 = 8) (h5 : t2 = 10) (h6 : t3 = 12) (hmeal : m = 5):
  c1 * t1 + c2 * t2 + c3 * t3 - m = 51 := 
by 
  sorry

end waiter_net_earning_l64_64718


namespace find_x_in_sequence_l64_64325

theorem find_x_in_sequence :
  ∃ x y z : ℤ, 
    (z - 1 = 0) ∧ (y - z = -1) ∧ (x - y = 1) ∧ x = 1 :=
by
  sorry

end find_x_in_sequence_l64_64325


namespace intersecting_diagonals_probability_l64_64767

theorem intersecting_diagonals_probability (n : ℕ) (h : n > 0) : 
  let vertices := 2 * n + 1 in
  let diagonals := (vertices * (vertices - 3)) / 2 in
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2 in
  let intersecting_pairs := ((vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24) in
  let probability := (n * (2 * n - 1) * 2) / (3 * ((2 * n ^ 2 - n - 1) * (2 * n ^ 2 - n - 2))) in
  (intersecting_pairs : ℝ) / (pairs_diagonals : ℝ) = probability :=
begin
  -- Proof to be provided
  sorry
end

end intersecting_diagonals_probability_l64_64767


namespace number_of_cirrus_clouds_l64_64668

def C_cb := 3
def C_cu := 12 * C_cb
def C_ci := 4 * C_cu

theorem number_of_cirrus_clouds : C_ci = 144 :=
by
  sorry

end number_of_cirrus_clouds_l64_64668


namespace slant_asymptote_sum_l64_64436

theorem slant_asymptote_sum (x : ℝ) (hx : x ≠ 5) :
  (5 : ℝ) + (21 : ℝ) = 26 :=
by
  sorry

end slant_asymptote_sum_l64_64436


namespace three_digit_even_sum_12_l64_64917

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l64_64917


namespace quadratic_function_order_l64_64616

theorem quadratic_function_order (a b c : ℝ) (h_neg_a : a < 0) 
  (h_sym : ∀ x, (a * (x + 2)^2 + b * (x + 2) + c) = (a * (2 - x)^2 + b * (2 - x) + c)) :
  (a * (-1992)^2 + b * (-1992) + c) < (a * (1992)^2 + b * (1992) + c) ∧
  (a * (1992)^2 + b * (1992) + c) < (a * (0)^2 + b * (0) + c) :=
by
  sorry

end quadratic_function_order_l64_64616


namespace height_difference_after_3_years_l64_64559

/-- Conditions for the tree's and boy's growth rates per season. --/
def tree_spring_growth : ℕ := 4
def tree_summer_growth : ℕ := 6
def tree_fall_growth : ℕ := 2
def tree_winter_growth : ℕ := 1

def boy_spring_growth : ℕ := 2
def boy_summer_growth : ℕ := 2
def boy_fall_growth : ℕ := 0
def boy_winter_growth : ℕ := 0

/-- Initial heights. --/
def initial_tree_height : ℕ := 16
def initial_boy_height : ℕ := 24

/-- Length of each season in months. --/
def season_length : ℕ := 3

/-- Time period in years. --/
def years : ℕ := 3

/-- Prove the height difference between the tree and the boy after 3 years is 73 inches. --/
theorem height_difference_after_3_years :
    let tree_annual_growth := tree_spring_growth * season_length +
                             tree_summer_growth * season_length +
                             tree_fall_growth * season_length +
                             tree_winter_growth * season_length
    let tree_final_height := initial_tree_height + tree_annual_growth * years
    let boy_annual_growth := boy_spring_growth * season_length +
                            boy_summer_growth * season_length +
                            boy_fall_growth * season_length +
                            boy_winter_growth * season_length
    let boy_final_height := initial_boy_height + boy_annual_growth * years
    tree_final_height - boy_final_height = 73 :=
by sorry

end height_difference_after_3_years_l64_64559


namespace percentage_of_children_speaking_only_Hindi_l64_64765

/-
In a class of 60 children, 30% of children can speak only English,
20% can speak both Hindi and English, and 42 children can speak Hindi.
Prove that the percentage of children who can speak only Hindi is 50%.
-/
theorem percentage_of_children_speaking_only_Hindi :
  let total_children := 60
  let english_only := 0.30 * total_children
  let both_languages := 0.20 * total_children
  let hindi_only := 42 - both_languages
  (hindi_only / total_children) * 100 = 50 :=
by
  sorry

end percentage_of_children_speaking_only_Hindi_l64_64765


namespace ages_of_boys_l64_64188

theorem ages_of_boys (a b c : ℕ) (h : a + b + c = 29) (h₁ : a = b) (h₂ : c = 11) : a = 9 ∧ b = 9 := 
by
  sorry

end ages_of_boys_l64_64188


namespace abc_inequality_l64_64500

theorem abc_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + ac + bc :=
by
  sorry

end abc_inequality_l64_64500


namespace cos_sum_condition_l64_64974

theorem cos_sum_condition {x y z : ℝ} (h1 : Real.cos x + Real.cos y + Real.cos z = 1) (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 1 := 
by 
  sorry

end cos_sum_condition_l64_64974


namespace relationship_between_M_and_N_l64_64120

variable (x y : ℝ)

theorem relationship_between_M_and_N (h1 : x ≠ 3) (h2 : y ≠ -2)
  (M : ℝ) (hm : M = x^2 + y^2 - 6 * x + 4 * y)
  (N : ℝ) (hn : N = -13) : M > N :=
by
  sorry

end relationship_between_M_and_N_l64_64120


namespace number_divisible_l64_64231

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l64_64231


namespace students_selected_are_three_l64_64192

-- Definitions of the conditions 
variables (boys girls ways : ℕ)
variables (selection_ways : ℕ)

-- Given conditions
def boys_in_class : Prop := boys = 15
def girls_in_class : Prop := girls = 10
def ways_to_select : Prop := selection_ways = 1050

-- Define the problem statement
theorem students_selected_are_three 
  (hb : boys_in_class boys) 
  (hg : girls_in_class girls)
  (hw : ways_to_select 1050) :
  ∃ n, n = 3 := 
sorry

end students_selected_are_three_l64_64192


namespace pascal_contains_53_l64_64456

theorem pascal_contains_53 (n : ℕ) (h1 : Nat.Prime 53) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 52 ∧ nat.choose 53 k = 53) (h3 : ∀ m < 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) (h4 : ∀ m > 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) : 
  (n = 53) → (n = 1) := 
by
  intros
  sorry

end pascal_contains_53_l64_64456


namespace erin_serves_all_soup_in_15_minutes_l64_64577

noncomputable def time_to_serve_all_soup
  (ounces_per_bowl : ℕ)
  (bowls_per_minute : ℕ)
  (soup_in_gallons : ℕ)
  (ounces_per_gallon : ℕ) : ℕ :=
  let total_ounces := soup_in_gallons * ounces_per_gallon
  let total_bowls := (total_ounces + ounces_per_bowl - 1) / ounces_per_bowl -- to round up
  let total_minutes := (total_bowls + bowls_per_minute - 1) / bowls_per_minute -- to round up
  total_minutes

theorem erin_serves_all_soup_in_15_minutes :
  time_to_serve_all_soup 10 5 6 128 = 15 :=
sorry

end erin_serves_all_soup_in_15_minutes_l64_64577


namespace y_intercept_of_line_l64_64039

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end y_intercept_of_line_l64_64039


namespace ping_pong_tournament_l64_64370

theorem ping_pong_tournament :
  ∃ n: ℕ, 
    (∃ m: ℕ, m ≥ 0 ∧ m ≤ 2 ∧ 2 * n + m = 29) ∧
    n = 14 ∧
    (n + 2 = 16) := 
by {
  sorry
}

end ping_pong_tournament_l64_64370


namespace circle_radius_l64_64020

theorem circle_radius (x y : ℝ) : x^2 + y^2 - 2*y = 0 → ∃ r : ℝ, r = 1 :=
by
  sorry

end circle_radius_l64_64020


namespace marbles_cost_correct_l64_64797

def total_cost : ℝ := 20.52
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52

-- The problem is to prove that the amount spent on marbles is $9.05
def amount_spent_on_marbles : ℝ :=
  total_cost - (cost_football + cost_baseball)

theorem marbles_cost_correct :
  amount_spent_on_marbles = 9.05 :=
by
  -- The proof goes here.
  sorry

end marbles_cost_correct_l64_64797


namespace math_problem_l64_64422

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end math_problem_l64_64422


namespace math_problem_l64_64421

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end math_problem_l64_64421


namespace probability_win_l64_64662

theorem probability_win (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose) = 3 / 8 :=
by
  rw [h]
  norm_num

end probability_win_l64_64662


namespace geom_seq_val_l64_64324

noncomputable def is_geom_seq (a : ℕ → ℝ) : Prop :=
∃ q b, ∀ n, a n = b * q^n

variables (a : ℕ → ℝ)

axiom a_5_a_7 : a 5 * a 7 = 2
axiom a_2_plus_a_10 : a 2 + a 10 = 3

theorem geom_seq_val (a_geom : is_geom_seq a) :
  (a 12) / (a 4) = 2 ∨ (a 12) / (a 4) = 1 / 2 :=
sorry

end geom_seq_val_l64_64324


namespace find_values_of_a_l64_64309

-- Definitions for sets A and B
def A : Set ℝ := {x | x^2 - x - 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 6 = 0}

-- The theorem we want to prove
theorem find_values_of_a (a : ℝ) : (A ∪ B a = A) ↔ (a = -6 ∨ a = 0 ∨ a = 3) :=
by
  sorry

end find_values_of_a_l64_64309


namespace find_g_l64_64571

open Function

def linear_system (a b c d e f g : ℚ) :=
  a + b + c + d + e = 1 ∧
  b + c + d + e + f = 2 ∧
  c + d + e + f + g = 3 ∧
  d + e + f + g + a = 4 ∧
  e + f + g + a + b = 5 ∧
  f + g + a + b + c = 6 ∧
  g + a + b + c + d = 7

theorem find_g (a b c d e f g : ℚ) (h : linear_system a b c d e f g) : 
  g = 13 / 3 :=
sorry

end find_g_l64_64571


namespace lateral_surface_area_of_cone_l64_64602

theorem lateral_surface_area_of_cone (r h : ℝ) (r_is_4 : r = 4) (h_is_3 : h = 3) :
  ∃ A : ℝ, A = 20 * Real.pi := by
  sorry

end lateral_surface_area_of_cone_l64_64602


namespace remainder_of_large_number_l64_64691

theorem remainder_of_large_number : 
  (9876543210 : ℤ) % 101 = 73 := 
by
  unfold_coes
  unfold_norm_num
  sorry

end remainder_of_large_number_l64_64691


namespace maci_pays_total_amount_l64_64982

theorem maci_pays_total_amount :
  let cost_blue_pen := 10 -- cents
  let cost_red_pen := 2 * cost_blue_pen -- cents
  let blue_pen_count := 10
  let red_pen_count := 15
  let total_cost_blue_pens := blue_pen_count * cost_blue_pen -- cents
  let total_cost_red_pens := red_pen_count * cost_red_pen -- cents
  let total_cost := total_cost_blue_pens + total_cost_red_pens -- cents
  total_cost / 100 = 4 -- dollars :=
by
  sorry

end maci_pays_total_amount_l64_64982


namespace pascal_triangle_contains_53_l64_64461

theorem pascal_triangle_contains_53 (n : ℕ) :
  (∃ k, binomial n k = 53) ↔ n = 53 := 
sorry

end pascal_triangle_contains_53_l64_64461


namespace problem1_solution_problem2_solution_l64_64704

noncomputable def problem1 (a b : ℝ) : ℝ :=
  (a + b) * (a - b) + b * (a + 2 * b) - (a + b)^2

theorem problem1_solution: 
  problem1 (-Real.sqrt 2) (Real.sqrt 6) = 2 * Real.sqrt 3 :=
sorry

theorem problem2_solution (x : ℝ) (h1 : x ≠ 5) (h2 : x ≠ -5):
  (3 / (x - 5) + 2 = (x - 2) / (5 - x)) → x = 3 :=
begin
  intro h,
  have h_eq : (3 / (x - 5) + 2) = -((x - 2) / (x - 5)), by {
    rw div_neg at h,
    exact h,
  },
  rw [←sub_eq_zero, ←div_eq_iff, sub_eq_iff_eq_add] at h_eq,
  {
    have hx5 : x ≠ 5, by { intro hx, rw hx at h_eq, exact h1 hx },    
    field_simp at h_eq,
    linarith,
  }, 
end

end problem1_solution_problem2_solution_l64_64704


namespace three_digit_number_is_504_l64_64272

theorem three_digit_number_is_504 (x : ℕ) [Decidable (x = 504)] :
  100 ≤ x ∧ x ≤ 999 →
  (x - 7) % 7 = 0 ∧
  (x - 8) % 8 = 0 ∧
  (x - 9) % 9 = 0 →
  x = 504 :=
by
  sorry

end three_digit_number_is_504_l64_64272


namespace largest_y_coordinate_l64_64836

theorem largest_y_coordinate (x y : ℝ) (h : (x^2 / 25) + ((y - 3)^2 / 25) = 0) : y = 3 := by
  sorry

end largest_y_coordinate_l64_64836


namespace angle_turned_by_hour_hand_l64_64176

theorem angle_turned_by_hour_hand (rotation_degrees_per_hour : ℝ) (total_degrees_per_rotation : ℝ) :
  rotation_degrees_per_hour * 1 = -30 :=
by
  have rotation_degrees_per_hour := - total_degrees_per_rotation / 12
  have total_degrees_per_rotation := 360
  sorry

end angle_turned_by_hour_hand_l64_64176


namespace final_stamp_collection_l64_64864

section StampCollection

structure Collection :=
  (nature : ℕ)
  (architecture : ℕ)
  (animals : ℕ)
  (vehicles : ℕ)
  (famous_people : ℕ)

def initial_collections : Collection := {
  nature := 10, architecture := 15, animals := 12, vehicles := 6, famous_people := 4
}

-- define transactions as functions that take a collection and return a modified collection
def transaction1 (c : Collection) : Collection :=
  { c with nature := c.nature + 4, architecture := c.architecture + 5, animals := c.animals + 5, vehicles := c.vehicles + 2, famous_people := c.famous_people + 1 }

def transaction2 (c : Collection) : Collection := 
  { c with nature := c.nature + 2, animals := c.animals - 1 }

def transaction3 (c : Collection) : Collection := 
  { c with animals := c.animals - 5, architecture := c.architecture + 3 }

def transaction4 (c : Collection) : Collection :=
  { c with animals := c.animals - 4, nature := c.nature + 7 }

def transaction7 (c : Collection) : Collection :=
  { c with vehicles := c.vehicles - 2, nature := c.nature + 5 }

def transaction8 (c : Collection) : Collection :=
  { c with vehicles := c.vehicles + 3, famous_people := c.famous_people - 3 }

def final_collection (c : Collection) : Collection :=
  transaction8 (transaction7 (transaction4 (transaction3 (transaction2 (transaction1 c)))))

theorem final_stamp_collection :
  final_collection initial_collections = { nature := 28, architecture := 23, animals := 7, vehicles := 9, famous_people := 2 } :=
by
  -- skip the proof
  sorry

end StampCollection

end final_stamp_collection_l64_64864


namespace simplified_polynomial_l64_64410

theorem simplified_polynomial : ∀ (x : ℝ), (3 * x + 2) * (3 * x - 2) - (3 * x - 1) ^ 2 = 6 * x - 5 := by
  sorry

end simplified_polynomial_l64_64410


namespace number_of_students_drawn_from_B_l64_64953

/-
This statement sets up the problem by defining the relevant variables and conditions,
then asserts the result using a theorem with a given proof (which is omitted here).
-/

variables (total_students : ℕ) 
          (sample_size : ℕ)
          (num_A num_B num_C : ℕ) 
          (sample_A sample_B sample_C : ℕ)

axiom total_students_hypothesis : total_students = 1500
axiom arithmetic_sequence : num_A + num_B + num_C = total_students
axiom sample_size_hypothesis : sample_size = 120
axiom sample_students : sample_A + sample_B + sample_C = sample_size
axiom arithmetic_sequence_sample : sample_A + 2 * sample_B + sample_C = 3 * sample_B

theorem number_of_students_drawn_from_B : sample_B = 40 :=
by sorry

end number_of_students_drawn_from_B_l64_64953


namespace unicorn_rope_length_l64_64564

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 1500
noncomputable def c : ℕ := 3

theorem unicorn_rope_length : a + b + c = 1593 :=
by
  -- The steps to prove the theorem should go here, but as stated, we skip this with "sorry".
  sorry

end unicorn_rope_length_l64_64564


namespace abc_sum_is_32_l64_64612

theorem abc_sum_is_32 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b + c = 31) (h5 : b * c + a = 31) (h6 : a * c + b = 31) : 
  a + b + c = 32 := 
by
  -- Proof goes here
  sorry

end abc_sum_is_32_l64_64612


namespace sequence_general_formula_and_max_n_l64_64125

theorem sequence_general_formula_and_max_n {a : ℕ → ℝ} {S : ℕ → ℝ} {T : ℕ → ℝ}
  (hS2 : S 2 = (3 / 2) * a 2 - 1) 
  (hS3 : S 3 = (3 / 2) * a 3 - 1) :
  (∀ n, a n = 2 * 3^(n - 1)) ∧ 
  (∃ n : ℕ, (8 / 5) * T n + n / (5 * 3 ^ (n - 1)) ≤ 40 / 27 ∧ ∀ k > n, 
    (8 / 5) * T k + k / (5 * 3 ^ (k - 1)) > 40 / 27) :=
by
  sorry

end sequence_general_formula_and_max_n_l64_64125


namespace sum_of_terms_l64_64910

noncomputable def arithmetic_sequence : Type :=
  {a : ℕ → ℤ // ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d}

theorem sum_of_terms (a : arithmetic_sequence) (h1 : a.val 1 + a.val 3 = 2) (h2 : a.val 3 + a.val 5 = 4) :
  a.val 5 + a.val 7 = 6 :=
by
  sorry

end sum_of_terms_l64_64910


namespace noah_left_lights_on_2_hours_l64_64801

-- Define the conditions
def bedroom_light_usage : ℕ := 6
def office_light_usage : ℕ := 3 * bedroom_light_usage
def living_room_light_usage : ℕ := 4 * bedroom_light_usage
def total_energy_used : ℕ := 96
def total_energy_per_hour := bedroom_light_usage + office_light_usage + living_room_light_usage

-- Define the main theorem to prove
theorem noah_left_lights_on_2_hours : total_energy_used / total_energy_per_hour = 2 := by
  sorry

end noah_left_lights_on_2_hours_l64_64801


namespace probability_of_finding_transmitter_l64_64711

def total_license_plates : ℕ := 900
def inspected_vehicles : ℕ := 18

theorem probability_of_finding_transmitter : (inspected_vehicles : ℝ) / (total_license_plates : ℝ) = 0.02 :=
by
  sorry

end probability_of_finding_transmitter_l64_64711


namespace store_earnings_l64_64082

theorem store_earnings (num_pencils : ℕ) (num_erasers : ℕ) (price_eraser : ℝ) 
  (multiplier : ℝ) (price_pencil : ℝ) (total_earnings : ℝ) :
  num_pencils = 20 →
  price_eraser = 1 →
  num_erasers = num_pencils * 2 →
  price_pencil = (price_eraser * num_erasers) * multiplier →
  multiplier = 2 →
  total_earnings = num_pencils * price_pencil + num_erasers * price_eraser →
  total_earnings = 120 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end store_earnings_l64_64082


namespace basketball_weight_calc_l64_64106

-- Define the variables and conditions
variable (weight_basketball weight_watermelon : ℕ)
variable (h1 : 8 * weight_basketball = 4 * weight_watermelon)
variable (h2 : weight_watermelon = 32)

-- Statement to prove
theorem basketball_weight_calc : weight_basketball = 16 :=
by
  sorry

end basketball_weight_calc_l64_64106


namespace big_boxes_count_l64_64193

theorem big_boxes_count
  (soaps_per_package : ℕ)
  (packages_per_box : ℕ)
  (total_soaps : ℕ)
  (soaps_per_box : ℕ)
  (H1 : soaps_per_package = 192)
  (H2 : packages_per_box = 6)
  (H3 : total_soaps = 2304)
  (H4 : soaps_per_box = soaps_per_package * packages_per_box) :
  total_soaps / soaps_per_box = 2 :=
by
  sorry

end big_boxes_count_l64_64193


namespace unit_square_divisible_l64_64124

theorem unit_square_divisible (n : ℕ) (h: n ≥ 6) : ∃ squares : ℕ, squares = n :=
by
  sorry

end unit_square_divisible_l64_64124


namespace man_speed_proof_l64_64558

noncomputable def man_speed_to_post_office (v : ℝ) : Prop :=
  let distance := 19.999999999999996
  let time_back := distance / 4
  let total_time := 5 + 48 / 60
  v > 0 ∧ distance / v + time_back = total_time

theorem man_speed_proof : ∃ v : ℝ, man_speed_to_post_office v ∧ v = 25 := by
  sorry

end man_speed_proof_l64_64558


namespace ratio_of_engineers_to_designers_l64_64013

-- Definitions of the variables
variables (e d : ℕ)

-- Conditions:
-- 1. The average age of the group is 45
-- 2. The average age of engineers is 40
-- 3. The average age of designers is 55

theorem ratio_of_engineers_to_designers (h : (40 * e + 55 * d) / (e + d) = 45) : e / d = 2 :=
by
-- Placeholder for the proof
sorry

end ratio_of_engineers_to_designers_l64_64013


namespace total_heads_l64_64713

/-- There are H hens and C cows. Each hen has 1 head and 2 feet, and each cow has 1 head and 4 feet.
Given that the total number of feet is 140 and there are 26 hens, prove that the total number of heads is 48. -/
theorem total_heads (H C : ℕ) (h1 : 2 * H + 4 * C = 140) (h2 : H = 26) : H + C = 48 := by
  sorry

end total_heads_l64_64713


namespace pascals_triangle_53_rows_l64_64473

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l64_64473


namespace count_valid_even_numbers_with_sum_12_l64_64922

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l64_64922


namespace average_speed_of_trip_l64_64262

theorem average_speed_of_trip 
  (total_distance : ℝ)
  (first_leg_distance : ℝ)
  (first_leg_speed : ℝ)
  (second_leg_distance : ℝ)
  (second_leg_speed : ℝ)
  (h_dist : total_distance = 50)
  (h_first_leg : first_leg_distance = 25)
  (h_second_leg : second_leg_distance = 25)
  (h_first_speed : first_leg_speed = 60)
  (h_second_speed : second_leg_speed = 30) :
  (total_distance / 
   ((first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed)) = 40) :=
by
  sorry

end average_speed_of_trip_l64_64262


namespace smallest_digits_to_append_l64_64210

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l64_64210


namespace number_divisible_l64_64232

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l64_64232


namespace percentage_of_orange_and_watermelon_juice_l64_64857

-- Define the total volume of the drink
def total_volume := 150

-- Define the volume of grape juice in the drink
def grape_juice_volume := 45

-- Define the percentage calculation for grape juice
def grape_juice_percentage := (grape_juice_volume / total_volume) * 100

-- Define the remaining percentage that is made of orange and watermelon juices
def remaining_percentage := 100 - grape_juice_percentage

-- Define the percentage of orange and watermelon juice being the same
def orange_and_watermelon_percentage := remaining_percentage / 2

theorem percentage_of_orange_and_watermelon_juice : 
  orange_and_watermelon_percentage = 35 :=
by
  -- The proof steps would go here
  sorry

end percentage_of_orange_and_watermelon_juice_l64_64857


namespace sum_of_fractions_l64_64867

theorem sum_of_fractions :
  (3 / 50) + (5 / 500) + (7 / 5000) = 0.0714 :=
by
  sorry

end sum_of_fractions_l64_64867


namespace bacteria_growth_l64_64877

-- Defining the function for bacteria growth
def bacteria_count (t : ℕ) (initial_count : ℕ) (division_time : ℕ) : ℕ :=
  initial_count * 2 ^ (t / division_time)

-- The initial conditions given in the problem
def initial_bacteria : ℕ := 1
def division_interval : ℕ := 10
def total_time : ℕ := 2 * 60

-- Stating the hypothesis and the goal
theorem bacteria_growth : bacteria_count total_time initial_bacteria division_interval = 2 ^ 12 :=
by
  -- Proof would go here
  sorry

end bacteria_growth_l64_64877


namespace like_terms_exponent_l64_64477

theorem like_terms_exponent (a : ℝ) : (2 * a = a + 3) → a = 3 := 
by
  intros h
  -- Proof here
  sorry

end like_terms_exponent_l64_64477


namespace derivative_of_f_l64_64783

noncomputable def f (x : ℝ) : ℝ :=
  (Nat.choose 4 0 : ℝ) - (Nat.choose 4 1 : ℝ) * x + (Nat.choose 4 2 : ℝ) * x^2 - (Nat.choose 4 3 : ℝ) * x^3 + (Nat.choose 4 4 : ℝ) * x^4

theorem derivative_of_f : 
  ∀ (x : ℝ), (deriv f x) = 4 * (-1 + x)^3 :=
by
  sorry

end derivative_of_f_l64_64783


namespace joey_learn_swimming_time_l64_64566

variable (days_vacation_joey : ℚ)
variable (time_learned_jon_smith : ℚ)
variable (time_learned_alexa : ℚ)

-- Alexa was on vacation for 3/4ths of the time it took Ethan to learn 12 fencing tricks
h1 : time_learned_alexa = 3/4 * time_learned_jon_smith

-- Joey spent half as much time as Ethan spent to learn swimming
h2 : days_vacation_joey = (1/2) * time_learned_jon_smith

-- Alexa spent a week and 2 days on vacation
h3 : time_learned_alexa = 9

theorem joey_learn_swimming_time : days_vacation_joey = 6 := by
  sorry

end joey_learn_swimming_time_l64_64566


namespace engineering_department_men_l64_64962

theorem engineering_department_men (total_students men_percentage women_count : ℕ) (h_percentage : men_percentage = 70) (h_women : women_count = 180) (h_total : total_students = (women_count * 100) / (100 - men_percentage)) : 
  (total_students * men_percentage / 100) = 420 :=
by
  sorry

end engineering_department_men_l64_64962


namespace smallest_number_append_l64_64222

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l64_64222


namespace cube_sum_div_by_9_implies_prod_div_by_3_l64_64166

theorem cube_sum_div_by_9_implies_prod_div_by_3 
  {a1 a2 a3 a4 a5 : ℤ} 
  (h : 9 ∣ a1^3 + a2^3 + a3^3 + a4^3 + a5^3) : 
  3 ∣ a1 * a2 * a3 * a4 * a5 := by
  sorry

end cube_sum_div_by_9_implies_prod_div_by_3_l64_64166


namespace balls_in_drawers_l64_64805

theorem balls_in_drawers (n k : ℕ) (h_n : n = 5) (h_k : k = 2) : (k ^ n) = 32 :=
by
  rw [h_n, h_k]
  sorry

end balls_in_drawers_l64_64805


namespace Ak_largest_at_166_l64_64584

theorem Ak_largest_at_166 :
  let A : ℕ → ℝ := λ k, (Nat.choose 1000 k : ℝ) * (0.2 ^ k)
  A 166 > A 165 ∧ A 166 > A 167 ∧ ∀ k, k ≠ 166 → A 166 > A k :=
by
  sorry

end Ak_largest_at_166_l64_64584


namespace AlyssaBottleCaps_l64_64155

def bottleCapsKatherine := 34
def bottleCapsGivenAway (bottleCaps: ℕ) := bottleCaps / 2
def bottleCapsLost (bottleCaps: ℕ) := bottleCaps - 8

theorem AlyssaBottleCaps : bottleCapsLost (bottleCapsGivenAway bottleCapsKatherine) = 9 := 
  by 
  sorry

end AlyssaBottleCaps_l64_64155


namespace exists_segment_l64_64841

theorem exists_segment (f : ℚ → ℤ) : 
  ∃ (a b c : ℚ), a ≠ b ∧ c = (a + b) / 2 ∧ f a + f b ≤ 2 * f c :=
by 
  sorry

end exists_segment_l64_64841


namespace angle_same_terminal_side_210_l64_64149

theorem angle_same_terminal_side_210 (n : ℤ) : 
  ∃ k : ℤ, 210 = -510 + k * 360 ∧ 0 ≤ 210 ∧ 210 < 360 :=
by
  use 2
  -- proof steps will go here
  sorry

end angle_same_terminal_side_210_l64_64149


namespace g_x_equation_g_3_value_l64_64657

noncomputable def g : ℝ → ℝ := sorry

theorem g_x_equation (x : ℝ) (hx : x ≠ 1/2) : g x + g ((x + 2) / (2 - 4 * x)) = 2 * x := sorry

theorem g_3_value : g 3 = 31 / 8 :=
by
  -- Use the provided functional equation and specific input values to derive g(3)
  sorry

end g_x_equation_g_3_value_l64_64657


namespace tangent_line_parabola_l64_64115

theorem tangent_line_parabola (k : ℝ) (tangent : ∀ y : ℝ, ∃ x : ℝ, 4 * x + 3 * y + k = 0 ∧ y^2 = 12 * x) : 
  k = 27 / 4 :=
sorry

end tangent_line_parabola_l64_64115


namespace jill_earnings_l64_64388

theorem jill_earnings :
  ∀ (hourly_wage : ℝ) (tip_rate : ℝ) (num_shifts : ℕ) (hours_per_shift : ℕ) (avg_orders_per_hour : ℝ),
  hourly_wage = 4.00 →
  tip_rate = 0.15 →
  num_shifts = 3 →
  hours_per_shift = 8 →
  avg_orders_per_hour = 40 →
  (num_shifts * hours_per_shift * hourly_wage + num_shifts * hours_per_shift * avg_orders_per_hour * tip_rate = 240) :=
by
  intros hourly_wage tip_rate num_shifts hours_per_shift avg_orders_per_hour
  intros hwage_eq trip_rate_eq nshifts_eq hshift_eq avgorder_eq
  sorry

end jill_earnings_l64_64388


namespace vacation_cost_division_l64_64830

theorem vacation_cost_division 
  (total_cost : ℝ) 
  (initial_people : ℝ) 
  (initial_cost_per_person : ℝ) 
  (cost_difference : ℝ) 
  (new_cost_per_person : ℝ) 
  (new_people : ℝ) 
  (h1 : total_cost = 1000) 
  (h2 : initial_people = 4) 
  (h3 : initial_cost_per_person = total_cost / initial_people) 
  (h4 : initial_cost_per_person = 250) 
  (h5 : cost_difference = 50) 
  (h6 : new_cost_per_person = initial_cost_per_person - cost_difference) 
  (h7 : new_cost_per_person = 200) 
  (h8 : total_cost / new_people = new_cost_per_person) :
  new_people = 5 := 
sorry

end vacation_cost_division_l64_64830


namespace Liz_total_spend_l64_64794

theorem Liz_total_spend :
  let recipe_book_cost := 6
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_cost := 5 * 3
  let apron_cost := recipe_book_cost + 1
  let total_spent_cost := recipe_book_cost + baking_dish_cost + ingredients_cost + apron_cost
  total_spent_cost = 40 :=
by
  let recipe_book_cost := 6
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_cost := 5 * 3
  let apron_cost := recipe_book_cost + 1
  let total_spent_cost := recipe_book_cost + baking_dish_cost + ingredients_cost + apron_cost
  show total_spent_cost = 40 from
    sorry

end Liz_total_spend_l64_64794


namespace carlos_jogged_distance_l64_64724

def carlos_speed := 4 -- Carlos's speed in miles per hour
def jogging_time := 2 -- Time in hours

theorem carlos_jogged_distance : carlos_speed * jogging_time = 8 :=
by
  sorry

end carlos_jogged_distance_l64_64724


namespace simplify_expression_l64_64678

theorem simplify_expression (w : ℝ) :
  2 * w^2 + 3 - 4 * w^2 + 2 * w - 6 * w + 4 = -2 * w^2 - 4 * w + 7 :=
by
  sorry

end simplify_expression_l64_64678


namespace box_volume_of_pyramid_l64_64116

/-- A theorem to prove the volume of the smallest cube-shaped box that can house the given rectangular pyramid. -/
theorem box_volume_of_pyramid :
  (∀ (h l w : ℕ), h = 15 ∧ l = 8 ∧ w = 12 → (∀ (v : ℕ), v = (max h (max l w)) ^ 3 → v = 3375)) :=
by
  intros h l w h_condition v v_def
  sorry

end box_volume_of_pyramid_l64_64116


namespace integer_solutions_of_inequality_system_l64_64012

theorem integer_solutions_of_inequality_system :
  { x : ℤ | (3 * x - 2) / 3 ≥ 1 ∧ 3 * x + 5 > 4 * x - 2 } = {2, 3, 4, 5, 6} :=
by {
  sorry
}

end integer_solutions_of_inequality_system_l64_64012


namespace todd_initial_gum_l64_64200

theorem todd_initial_gum (x : ℝ)
(h1 : 150 = 0.25 * x)
(h2 : x + 150 = 890) :
x = 712 :=
by
  -- Here "by" is used to denote the beginning of proof block
  sorry -- Proof will be filled in later.

end todd_initial_gum_l64_64200


namespace appropriate_survey_method_l64_64863

def survey_method_suitability (method : String) (context : String) : Prop :=
  match context, method with
  | "daily floating population of our city", "sampling survey" => true
  | "security checks before passengers board an airplane", "comprehensive survey" => true
  | "killing radius of a batch of shells", "sampling survey" => true
  | "math scores of Class 1 in Grade 7 of a certain school", "census method" => true
  | _, _ => false

theorem appropriate_survey_method :
  survey_method_suitability "census method" "daily floating population of our city" = false ∧
  survey_method_suitability "comprehensive survey" "security checks before passengers board an airplane" = false ∧
  survey_method_suitability "sampling survey" "killing radius of a batch of shells" = false ∧
  survey_method_suitability "census method" "math scores of Class 1 in Grade 7 of a certain school" = true :=
by
  sorry

end appropriate_survey_method_l64_64863


namespace expected_coins_100_rounds_l64_64540

noncomputable def expectedCoinsAfterGame (rounds : ℕ) (initialCoins : ℕ) : ℝ :=
  initialCoins * (101 / 100) ^ rounds

theorem expected_coins_100_rounds :
  expectedCoinsAfterGame 100 1 = (101 / 100 : ℝ) ^ 100 :=
by
  sorry

end expected_coins_100_rounds_l64_64540


namespace minimum_value_y_l64_64633

noncomputable def y (x : ℚ) : ℚ := |3 - x| + |x - 2| + |-1 + x|

theorem minimum_value_y : ∃ x : ℚ, y x = 2 :=
by
  sorry

end minimum_value_y_l64_64633


namespace Eliane_schedule_combinations_l64_64107

def valid_schedule_combinations : ℕ :=
  let mornings := 6 * 3 -- 6 days (Monday to Saturday) each with 3 time slots
  let afternoons := 5 * 2 -- 5 days (Monday to Friday) each with 2 time slots
  let mon_or_fri_comb := 2 * 3 * 3 * 2 -- Morning on Monday or Friday
  let sat_comb := 1 * 3 * 4 * 2 -- Morning on Saturday
  let tue_wed_thu_comb := 3 * 3 * 2 * 2 -- Morning on Tuesday, Wednesday, or Thursday
  mon_or_fri_comb + sat_comb + tue_wed_thu_comb

theorem Eliane_schedule_combinations :
  valid_schedule_combinations = 96 := by
  sorry

end Eliane_schedule_combinations_l64_64107


namespace eggs_in_each_basket_is_four_l64_64570

theorem eggs_in_each_basket_is_four 
  (n : ℕ)
  (h1 : n ∣ 16) 
  (h2 : n ∣ 28) 
  (h3 : n ≥ 2) : 
  n = 4 :=
sorry

end eggs_in_each_basket_is_four_l64_64570


namespace range_of_a_l64_64143

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x + 6 < 2 + 3x → (a + x) / 4 > x) ∧ (∃! i : ℤ, ∃! j : ℤ, ∃! k : ℤ, 2 < i ∧ i < a / 3 ∧ 2 < j ∧ j < a / 3 ∧ 2 < k ∧ k < a / 3) → 15 < a ∧ a ≤ 18 :=
by
  sorry

end range_of_a_l64_64143


namespace serving_time_correct_l64_64578
noncomputable theory

def ounces_per_bowl := 10
def bowls_per_minute := 5
def gallons_of_soup := 6
def ounces_per_gallon := 128

def total_ounces := gallons_of_soup * ounces_per_gallon
def serving_rate := ounces_per_bowl * bowls_per_minute

def serving_time := total_ounces / serving_rate

def rounded_serving_time := Int.floor (serving_time + 0.5)

theorem serving_time_correct : rounded_serving_time = 15 := by
  sorry

end serving_time_correct_l64_64578


namespace find_a_for_parallel_lines_l64_64942

theorem find_a_for_parallel_lines (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 ↔ 2 * x + (a + 1) * y + 1 = 0) → a = -3 :=
by
  sorry

end find_a_for_parallel_lines_l64_64942


namespace correct_statement_a_l64_64060

theorem correct_statement_a (x y : ℝ) (h : x + y < 0) : x^2 - y > x :=
sorry

end correct_statement_a_l64_64060


namespace range_of_a_l64_64450

theorem range_of_a (a : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ (a^2 * x - 2 * a + 1 = 0)) ↔ (a > 1/2 ∧ a ≠ 1) :=
by
  sorry

end range_of_a_l64_64450


namespace number_of_rows_containing_53_l64_64470

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l64_64470


namespace equation_of_line_l64_64015

theorem equation_of_line (θ : ℝ) (b : ℝ) (k : ℝ) (y x : ℝ) :
  θ = Real.pi / 4 ∧ b = 2 ∧ k = Real.tan θ ∧ k = 1 ∧ y = k * x + b ↔ y = x + 2 :=
by
  intros
  sorry

end equation_of_line_l64_64015


namespace toys_per_hour_computation_l64_64395

noncomputable def total_toys : ℕ := 20500
noncomputable def monday_hours : ℕ := 8
noncomputable def tuesday_hours : ℕ := 7
noncomputable def wednesday_hours : ℕ := 9
noncomputable def thursday_hours : ℕ := 6

noncomputable def total_hours_worked : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours
noncomputable def toys_produced_each_hour : ℚ := total_toys / total_hours_worked

theorem toys_per_hour_computation :
  toys_produced_each_hour = 20500 / (8 + 7 + 9 + 6) :=
by
  -- Proof goes here
  sorry

end toys_per_hour_computation_l64_64395


namespace nandan_gain_l64_64629

theorem nandan_gain (x t : ℝ) (nandan_gain krishan_gain total_gain : ℝ)
  (h1 : krishan_gain = 12 * x * t)
  (h2 : nandan_gain = x * t)
  (h3 : total_gain = nandan_gain + krishan_gain)
  (h4 : total_gain = 78000) :
  nandan_gain = 6000 :=
by
  -- Proof goes here
  sorry

end nandan_gain_l64_64629


namespace joey_read_percentage_l64_64330

theorem joey_read_percentage : 
  ∀ (total_pages read_after_break : ℕ), 
  total_pages = 30 → read_after_break = 9 → 
  ( (total_pages - read_after_break : ℕ) / (total_pages : ℕ) * 100 ) = 70 :=
by
  intros total_pages read_after_break h_total h_after
  sorry

end joey_read_percentage_l64_64330


namespace gcd_g102_g103_eq_one_l64_64338

def g (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_g102_g103_eq_one : Nat.gcd (g 102).natAbs (g 103).natAbs = 1 := by
  sorry

end gcd_g102_g103_eq_one_l64_64338


namespace total_students_in_class_l64_64671

-- Define the initial conditions
def num_students_in_row (a b: Nat) : Nat := a + 1 + b
def num_lines : Nat := 3
noncomputable def students_in_row : Nat := num_students_in_row 2 5 

-- Theorem to prove the total number of students in the class
theorem total_students_in_class : students_in_row * num_lines = 24 :=
by
  sorry

end total_students_in_class_l64_64671


namespace smallest_number_append_l64_64225

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l64_64225


namespace solve_equation_l64_64644

theorem solve_equation (x : ℝ) (h1 : x ≠ 2 / 3) :
  (7 * x + 3) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔
  x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end solve_equation_l64_64644


namespace pascal_triangle_contains_53_only_once_l64_64459

theorem pascal_triangle_contains_53_only_once (n : ℕ) (k : ℕ) (h_prime : Nat.prime 53) :
  (n = 53 ∧ (k = 1 ∨ k = 52) ∨ 
   ∀ m < 53, Π l, Nat.binomial m l ≠ 53) ∧ 
  (n > 53 → (k = 0 ∨ k = n ∨ Π a b, a * 53 ≠ b * Nat.factorial (n - k + 1))) :=
sorry

end pascal_triangle_contains_53_only_once_l64_64459


namespace train_crosses_in_26_seconds_l64_64555

def speed_km_per_hr := 72
def length_of_train := 250
def length_of_platform := 270

def total_distance := length_of_train + length_of_platform

noncomputable def speed_m_per_s := (speed_km_per_hr * 1000 / 3600)  -- Convert km/hr to m/s

noncomputable def time_to_cross := total_distance / speed_m_per_s

theorem train_crosses_in_26_seconds :
  time_to_cross = 26 := 
sorry

end train_crosses_in_26_seconds_l64_64555


namespace dinner_guest_arrangement_l64_64329

noncomputable def number_of_ways (n k : ℕ) : ℕ :=
  if n < k then 0 else Nat.factorial n / Nat.factorial (n - k)

theorem dinner_guest_arrangement :
  let total_arrangements := number_of_ways 8 5
  let unwanted_arrangements := 7 * number_of_ways 6 3 * 2
  let valid_arrangements := total_arrangements - unwanted_arrangements
  valid_arrangements = 5040 :=
by
  -- Definitions and preliminary calculations
  let total_arrangements := number_of_ways 8 5
  let unwanted_arrangements := 7 * number_of_ways 6 3 * 2
  let valid_arrangements := total_arrangements - unwanted_arrangements

  -- This is where the proof would go, but we insert sorry to skip it for now
  sorry

end dinner_guest_arrangement_l64_64329


namespace smallest_sum_of_cubes_two_ways_l64_64839

theorem smallest_sum_of_cubes_two_ways :
  ∃ (n : ℕ) (a b c d e f : ℕ),
  n = a^3 + b^3 + c^3 ∧ n = d^3 + e^3 + f^3 ∧
  (a, b, c) ≠ (d, e, f) ∧
  (d, e, f) ≠ (a, b, c) ∧ n = 251 :=
by
  sorry

end smallest_sum_of_cubes_two_ways_l64_64839


namespace mother_used_eggs_l64_64263

variable (initial_eggs : ℕ) (eggs_after_chickens : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (current_eggs : ℕ)

theorem mother_used_eggs (h1 : initial_eggs = 10)
                        (h2 : chickens = 2)
                        (h3 : eggs_per_chicken = 3)
                        (h4 : current_eggs = 11)
                        (eggs_laid : ℕ)
                        (h5 : eggs_laid = chickens * eggs_per_chicken)
                        (eggs_used : ℕ)
                        (h6 : eggs_after_chickens = initial_eggs - eggs_used + eggs_laid)
                        : eggs_used = 7 :=
by
  -- proof steps go here
  sorry

end mother_used_eggs_l64_64263


namespace platform_length_l64_64821

theorem platform_length (speed_kmh : ℕ) (time_min : ℕ) (train_length_m : ℕ) (distance_covered_m : ℕ) : 
  speed_kmh = 90 → time_min = 1 → train_length_m = 750 → distance_covered_m = 1500 →
  train_length_m + (distance_covered_m - train_length_m) = 750 + (1500 - 750) :=
by sorry

end platform_length_l64_64821


namespace value_of_f_at_2_and_neg_log2_3_l64_64132

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then Real.log x / Real.log 2 else 2^(-x)

theorem value_of_f_at_2_and_neg_log2_3 :
  f 2 * f (-Real.log 3 / Real.log 2) = 3 := by
  sorry

end value_of_f_at_2_and_neg_log2_3_l64_64132


namespace salary_reduction_percentage_l64_64079

theorem salary_reduction_percentage
  (S : ℝ) 
  (h : S * (1 - R / 100) = S / 1.388888888888889): R = 28 :=
sorry

end salary_reduction_percentage_l64_64079


namespace number_of_men_in_engineering_department_l64_64960

theorem number_of_men_in_engineering_department (T : ℝ) (h1 : 0.30 * T = 180) : 
  0.70 * T = 420 :=
by 
  -- The proof will be done here, but for now, we skip it.
  sorry

end number_of_men_in_engineering_department_l64_64960


namespace smallest_digits_to_append_l64_64228

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l64_64228


namespace factorial_of_6_is_720_l64_64640

theorem factorial_of_6_is_720 : (Nat.factorial 6) = 720 := by
  sorry

end factorial_of_6_is_720_l64_64640


namespace amare_fabric_needed_l64_64988

-- Definitions for the conditions
def fabric_per_dress_yards : ℝ := 5.5
def number_of_dresses : ℕ := 4
def fabric_owned_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- Total fabric needed in yards
def total_fabric_needed_yards : ℝ := fabric_per_dress_yards * number_of_dresses

-- Total fabric needed in feet
def total_fabric_needed_feet : ℝ := total_fabric_needed_yards * yard_to_feet

-- Fabric still needed
def fabric_still_needed : ℝ := total_fabric_needed_feet - fabric_owned_feet

-- Proof
theorem amare_fabric_needed : fabric_still_needed = 59 := by
  sorry

end amare_fabric_needed_l64_64988


namespace find_sum_mod_7_l64_64313

open ZMod

-- Let a, b, and c be elements of the cyclic group modulo 7
def a : ZMod 7 := sorry
def b : ZMod 7 := sorry
def c : ZMod 7 := sorry

-- Conditions
axiom h1 : a * b * c = 1
axiom h2 : 4 * c = 5
axiom h3 : 5 * b = 4 + b

-- Goal
theorem find_sum_mod_7 : a + b + c = 2 := by
  sorry

end find_sum_mod_7_l64_64313


namespace min_value_PA_d_l64_64745

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_value_PA_d :
  let A : ℝ × ℝ := (3, 4)
  let parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1
  let distance_to_line (P : ℝ × ℝ) (line_x : ℝ) : ℝ := abs (P.1 - line_x)
  let d : ℝ := distance_to_line P (-1)
  ∀ P : ℝ × ℝ, parabola P → (distance P A + d) ≥ 2 * Real.sqrt 5 :=
by
  sorry

end min_value_PA_d_l64_64745


namespace simultaneous_equations_solution_exists_l64_64726

theorem simultaneous_equations_solution_exists (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 :=
by
  -- proof goes here
  sorry

end simultaneous_equations_solution_exists_l64_64726


namespace smallest_n_for_fraction_with_digits_439_l64_64014

theorem smallest_n_for_fraction_with_digits_439 (m n : ℕ) (hmn : Nat.gcd m n = 1) (hmn_pos : 0 < m ∧ m < n) (digits_439 : ∃ X : ℕ, (m : ℚ) / n = (439 + 1000 * X) / 1000) : n = 223 :=
by
  sorry

end smallest_n_for_fraction_with_digits_439_l64_64014


namespace min_x2_y2_l64_64342

theorem min_x2_y2 (x y : ℝ) (h : 2 * (x^2 + y^2) = x^2 + y + x * y) : 
  (∃ x y, x = 0 ∧ y = 0) ∨ x^2 + y^2 >= 1 := 
sorry

end min_x2_y2_l64_64342


namespace negation_of_conditional_l64_64824

-- Define the propositions
def P (x : ℝ) : Prop := x > 2015
def Q (x : ℝ) : Prop := x > 0

-- Negate the propositions
def notP (x : ℝ) : Prop := x <= 2015
def notQ (x : ℝ) : Prop := x <= 0

-- Theorem: Negation of the conditional statement
theorem negation_of_conditional (x : ℝ) : ¬ (P x → Q x) ↔ (notP x → notQ x) :=
by
  sorry

end negation_of_conditional_l64_64824


namespace queen_middle_school_teachers_l64_64010

theorem queen_middle_school_teachers
  (students : ℕ) 
  (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ)
  (students_per_class : ℕ)
  (h_students : students = 1500)
  (h_classes_per_student : classes_per_student = 6)
  (h_classes_per_teacher : classes_per_teacher = 5)
  (h_students_per_class : students_per_class = 25) : 
  (students * classes_per_student / students_per_class) / classes_per_teacher = 72 :=
by
  sorry

end queen_middle_school_teachers_l64_64010


namespace gcd_pow_of_subtraction_l64_64680

noncomputable def m : ℕ := 2^2100 - 1
noncomputable def n : ℕ := 2^1950 - 1

theorem gcd_pow_of_subtraction : Nat.gcd m n = 2^150 - 1 :=
by
  -- To be proven
  sorry

end gcd_pow_of_subtraction_l64_64680


namespace arithmetic_sequence_sum_ratio_l64_64791

noncomputable def S (n : ℕ) (a_1 : ℚ) (d : ℚ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum_ratio (a_1 d : ℚ) (h : d ≠ 0) (h_ratio : (a_1 + 5 * d) / (a_1 + 2 * d) = 2) :
  S 6 a_1 d / S 3 a_1 d = 7 / 2 :=
by
  sorry

end arithmetic_sequence_sum_ratio_l64_64791


namespace dartboard_central_angle_l64_64261

theorem dartboard_central_angle (A : ℝ) (x : ℝ) (P : ℝ) (h1 : P = 1 / 4) 
    (h2 : A > 0) : (x / 360 = 1 / 4) -> x = 90 :=
by
  sorry

end dartboard_central_angle_l64_64261


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l64_64933

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l64_64933


namespace boxes_sold_l64_64327

theorem boxes_sold (cases boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 8) :
  cases * boxes_per_case = 24 :=
by
  rw [h_cases, h_boxes_per_case]
  norm_num

end boxes_sold_l64_64327


namespace bug_visits_tiles_l64_64080

theorem bug_visits_tiles (width length : ℕ) (h_w : width = 15) (h_l : length = 25) :
  width + length - Nat.gcd width length = 35 :=
by
  rw [h_w, h_l]
  rfl

end bug_visits_tiles_l64_64080


namespace constant_term_of_product_is_21_l64_64376

def P (x : ℕ) : ℕ := x ^ 3 + x ^ 2 + 3
def Q (x : ℕ) : ℕ := 2 * x ^ 4 + x ^ 2 + 7

theorem constant_term_of_product_is_21 :
  (P 0) * (Q 0) = 21 :=
by
  rw [P, Q]
  simp
  rfl

end constant_term_of_product_is_21_l64_64376


namespace smallest_number_append_l64_64223

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l64_64223


namespace inequality_proof_l64_64596

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : 0 < c)
  : a + b ≤ 2 * c ∧ 2 * c ≤ 3 * c :=
sorry

end inequality_proof_l64_64596


namespace calculation_l64_64428

theorem calculation (a b : ℕ) (h1 : a = 7) (h2 : b = 5) : (a^2 - b^2) ^ 2 = 576 :=
by
  sorry

end calculation_l64_64428


namespace john_total_money_l64_64179

-- Variables representing the prices and quantities.
def chip_price : ℝ := 2
def corn_chip_price : ℝ := 1.5
def chips_quantity : ℕ := 15
def corn_chips_quantity : ℕ := 10

-- Hypothesis representing the total money John has.
theorem john_total_money : 
    (chips_quantity * chip_price + corn_chips_quantity * corn_chip_price) = 45 := by
  sorry

end john_total_money_l64_64179


namespace angle_slope_condition_l64_64600

theorem angle_slope_condition (α k : Real) (h₀ : k = Real.tan α) (h₁ : 0 ≤ α ∧ α < Real.pi) : 
  (α < Real.pi / 3) → (k < Real.sqrt 3) ∧ ¬((k < Real.sqrt 3) → (α < Real.pi / 3)) := 
sorry

end angle_slope_condition_l64_64600


namespace y_intercept_of_line_l64_64040

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end y_intercept_of_line_l64_64040


namespace speed_of_second_half_l64_64273

theorem speed_of_second_half (t d s1 d1 d2 : ℝ) (h_t : t = 30) (h_d : d = 672) (h_s1 : s1 = 21)
  (h_d1 : d1 = d / 2) (h_d2 : d2 = d / 2) (h_t1 : d1 / s1 = 16) (h_t2 : t - d1 / s1 = 14) :
  d2 / 14 = 24 :=
by sorry

end speed_of_second_half_l64_64273


namespace find_a_l64_64912

theorem find_a
  (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = 3 * Real.sin (2 * x - Real.pi / 3))
  (a : ℝ)
  (h₂ : 0 < a)
  (h₃ : a < Real.pi / 2)
  (h₄ : ∀ x, f (x + a) = f (-x + a)) :
  a = 5 * Real.pi / 12 :=
sorry

end find_a_l64_64912


namespace product_of_averages_is_125000_l64_64728

-- Define the problem from step a
def sum_1_to_99 : ℕ := (99 * (1 + 99)) / 2
def average_of_group (x : ℕ) : Prop := 3 * 33 * x = sum_1_to_99

-- Define the goal to prove
theorem product_of_averages_is_125000 (x : ℕ) (h : average_of_group x) : x^3 = 125000 :=
by
  sorry

end product_of_averages_is_125000_l64_64728


namespace inequation_proof_l64_64599

theorem inequation_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 = 1) :
  (a / (1 - a^2)) + (b / (1 - b^2)) + (c / (1 - c^2)) ≥ (3 * Real.sqrt 3 / 2) :=
by
  sorry

end inequation_proof_l64_64599


namespace average_words_per_puzzle_l64_64720

theorem average_words_per_puzzle (daily_puzzle : ℕ) (days_per_pencil : ℕ) (words_per_pencil : ℕ) (H1 : daily_puzzle = 1) (H2 : days_per_pencil = 14) (H3 : words_per_pencil = 1050) :
  words_per_pencil / days_per_pencil = 75 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end average_words_per_puzzle_l64_64720


namespace ab_cannot_be_specific_values_l64_64141

theorem ab_cannot_be_specific_values (a b : ℝ) (h1 : ∀ x : ℝ, x > 0 → (ln (a * x) - 1) * (exp x - b) ≥ 0) : 
  ab ≠ e ∧ ab ≠ (25 / 4) :=
  sorry

end ab_cannot_be_specific_values_l64_64141


namespace B_days_to_complete_job_alone_l64_64072

theorem B_days_to_complete_job_alone (x : ℝ) : 
  (1 / 15 + 1 / x) * 4 = 0.4666666666666667 → x = 20 :=
by
  intro h
  -- Note: The proof is omitted as we only need the statement here.
  sorry

end B_days_to_complete_job_alone_l64_64072


namespace min_score_needed_l64_64430

theorem min_score_needed 
  (s1 s2 s3 s4 s5 : ℕ)
  (next_test_goal_increment : ℕ)
  (current_scores_sum : ℕ)
  (desired_average : ℕ)
  (total_tests : ℕ)
  (required_total_sum : ℕ)
  (required_next_score : ℕ)
  (current_scores : s1 = 88 ∧ s2 = 92 ∧ s3 = 75 ∧ s4 = 85 ∧ s5 = 80)
  (increment_eq : next_test_goal_increment = 5)
  (current_sum_eq : current_scores_sum = s1 + s2 + s3 + s4 + s5)
  (desired_average_eq : desired_average = (current_scores_sum / 5) + next_test_goal_increment)
  (total_tests_eq : total_tests = 6)
  (required_total_sum_eq : required_total_sum = desired_average * total_tests)
  (required_next_score_eq : required_next_score = required_total_sum - current_scores_sum) :
  required_next_score = 114 := by
    sorry

end min_score_needed_l64_64430


namespace sequence_length_l64_64097

theorem sequence_length 
  (a : ℕ)
  (b : ℕ)
  (d : ℕ)
  (steps : ℕ)
  (h1 : a = 160)
  (h2 : b = 28)
  (h3 : d = 4)
  (h4 : (28:ℕ) = (160:ℕ) - steps * 4) :
  steps + 1 = 34 :=
by
  sorry

end sequence_length_l64_64097


namespace total_interest_after_tenth_year_l64_64545

variable {P R : ℕ}

theorem total_interest_after_tenth_year
  (h1 : (P * R * 10) / 100 = 900)
  (h2 : 5 * P * R / 100 = 450)
  (h3 : 5 * 3 * P * R / 100 = 1350) :
  (450 + 1350) = 1800 :=
by
  sorry

end total_interest_after_tenth_year_l64_64545


namespace opponent_score_value_l64_64676

-- Define the given conditions
def total_points : ℕ := 720
def games_played : ℕ := 24
def average_score := total_points / games_played
def championship_score := average_score / 2 - 2
def opponent_score := championship_score + 2

-- Lean theorem statement to prove
theorem opponent_score_value : opponent_score = 15 :=
by
  -- Proof to be filled in
  sorry

end opponent_score_value_l64_64676


namespace Lyka_savings_l64_64637

def Smartphone_cost := 800
def Initial_savings := 200
def Gym_cost_per_month := 50
def Total_months := 4
def Weeks_per_month := 4
def Savings_per_week_initial := 50
def Savings_per_week_after_raise := 80

def Total_savings : Nat :=
  let initial_savings := Savings_per_week_initial * Weeks_per_month * 2
  let increased_savings := Savings_per_week_after_raise * Weeks_per_month * 2
  initial_savings + increased_savings

theorem Lyka_savings :
  (Initial_savings + Total_savings) = 1040 := by
  sorry

end Lyka_savings_l64_64637


namespace sum_of_roots_l64_64048

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end sum_of_roots_l64_64048


namespace smallest_total_students_l64_64490

theorem smallest_total_students :
  (∃ (n : ℕ), 4 * n + (n + 2) > 50 ∧ ∀ m, 4 * m + (m + 2) > 50 → m ≥ n) → 4 * 10 + (10 + 2) = 52 :=
by
  sorry

end smallest_total_students_l64_64490


namespace count_prime_boring_lt_10000_l64_64104

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_boring (n : ℕ) : Prop := 
  let digits := n.digits 10
  match digits with
  | [] => false
  | (d::ds) => ds.all (fun x => x = d)

theorem count_prime_boring_lt_10000 : 
  ∃! n, is_prime n ∧ is_boring n ∧ n < 10000 := 
by 
  sorry

end count_prime_boring_lt_10000_l64_64104


namespace sum_of_first_six_terms_l64_64749

noncomputable def a (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * a (n - 1)

def sum_first_six_terms (a : ℕ → ℚ) : ℚ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_of_first_six_terms :
  sum_first_six_terms a = 63 / 32 :=
by
  sorry

end sum_of_first_six_terms_l64_64749


namespace cooking_oil_distribution_l64_64252

theorem cooking_oil_distribution (total_oil : ℝ) (oil_A : ℝ) (oil_B : ℝ) (oil_C : ℝ)
    (h_total_oil : total_oil = 3 * 1000) -- Total oil is 3000 milliliters
    (h_A_B : oil_A = oil_B + 200) -- A receives 200 milliliters more than B
    (h_B_C : oil_B = oil_C + 200) -- B receives 200 milliliters more than C
    : oil_B = 1000 :=              -- We need to prove B receives 1000 milliliters
by
  sorry

end cooking_oil_distribution_l64_64252


namespace slope_of_intersection_line_l64_64288

theorem slope_of_intersection_line :
  ∀ t : ℝ, ∃ x y : ℝ, (x + 2 * y = 7 * t + 3) ∧ (x - y = 2 * t - 2) → 
    (∃ m c : ℝ, (∀ t : ℝ,
      let x := (11 * t - 1) / 3 in
      let y := (5 * t + 5) / 3 in
      y = m * x + c) ∧ m = 5 / 11) :=
begin
  intros t,
  use [x, y],
  split,
  { sorry }, -- x + 2y = 7t + 3
  { sorry }, -- x - y = 2t - 2
  split,
  { intros t,
    let x := (11 * t - 1) / 3,
    let y := (5 * t + 5) / 3,
    use [m, c],
    split,
    { sorry }, -- y = mx + c
    { exact 5 / 11 } -- m = 5 / 11
  },
end

end slope_of_intersection_line_l64_64288


namespace number_of_sets_l64_64827

theorem number_of_sets (a n : ℕ) (M : Finset ℕ) (h_consecutive : ∀ x ∈ M, ∃ k, x = a + k ∧ k < n) (h_card : M.card ≥ 2) (h_sum : M.sum id = 2002) : n = 7 :=
sorry

end number_of_sets_l64_64827


namespace power_inequality_l64_64975

theorem power_inequality (a b n : ℕ) (h_ab : a > b) (h_b1 : b > 1)
  (h_odd_b : b % 2 = 1) (h_n_pos : 0 < n) (h_div : b^n ∣ a^n - 1) :
  a^b > 3^n / n :=
by
  sorry

end power_inequality_l64_64975


namespace append_digits_divisible_by_all_less_than_10_l64_64218

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l64_64218


namespace combined_variance_is_178_l64_64526

noncomputable def average_weight_A := 60
noncomputable def variance_A := 100
noncomputable def average_weight_B := 64
noncomputable def variance_B := 200
noncomputable def ratio_A_B := (1, 3)

theorem combined_variance_is_178 :
  let nA := ratio_A_B.1
  let nB := ratio_A_B.2
  let avg_comb := (nA * average_weight_A + nB * average_weight_B) / (nA + nB)
  let var_comb := (nA * (variance_A + (average_weight_A - avg_comb)^2) + 
                   nB * (variance_B + (average_weight_B - avg_comb)^2)) / 
                   (nA + nB)
  var_comb = 178 := 
by
  sorry

end combined_variance_is_178_l64_64526


namespace geometric_series_common_ratio_l64_64088

theorem geometric_series_common_ratio (a S r : ℝ) (h1 : a = 512) (h2 : S = 3072) 
(h3 : S = a / (1 - r)) : r = 5/6 := 
sorry

end geometric_series_common_ratio_l64_64088


namespace find_angle_four_l64_64400

theorem find_angle_four (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle1 + angle3 + 60 = 180)
  (h3 : angle3 = angle4) :
  angle4 = 60 :=
by sorry

end find_angle_four_l64_64400


namespace append_digits_divisible_by_all_less_than_10_l64_64220

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l64_64220


namespace find_a_l64_64911

theorem find_a (k a : ℚ) (hk : 4 * k = 60) (ha : 15 * a - 5 = 60) : a = 13 / 3 :=
by
  sorry

end find_a_l64_64911


namespace train_speed_in_kmh_l64_64563

-- Definitions from the conditions
def length_of_train : ℝ := 800 -- in meters
def time_to_cross_pole : ℝ := 20 -- in seconds
def conversion_factor : ℝ := 3.6 -- (km/h) per (m/s)

-- Statement to prove the train's speed in km/h
theorem train_speed_in_kmh :
  (length_of_train / time_to_cross_pole * conversion_factor) = 144 :=
  sorry

end train_speed_in_kmh_l64_64563


namespace jose_completion_time_l64_64539

noncomputable def rate_jose : ℚ := 1 / 30
noncomputable def rate_jane : ℚ := 1 / 6

theorem jose_completion_time :
  ∀ (J A : ℚ), 
    (J + A = 1 / 5) ∧ (J = rate_jose) ∧ (A = rate_jane) → 
    (1 / J = 30) :=
by
  intros J A h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end jose_completion_time_l64_64539


namespace james_total_payment_is_correct_l64_64973

-- Define the constants based on the conditions
def numDirtBikes : Nat := 3
def costPerDirtBike : Nat := 150
def numOffRoadVehicles : Nat := 4
def costPerOffRoadVehicle : Nat := 300
def numTotalVehicles : Nat := numDirtBikes + numOffRoadVehicles
def registrationCostPerVehicle : Nat := 25

-- Define the total calculation using the given conditions
def totalPaidByJames : Nat :=
  (numDirtBikes * costPerDirtBike) +
  (numOffRoadVehicles * costPerOffRoadVehicle) +
  (numTotalVehicles * registrationCostPerVehicle)

-- State the proof problem
theorem james_total_payment_is_correct : totalPaidByJames = 1825 := by
  sorry

end james_total_payment_is_correct_l64_64973


namespace find_f_k_l_l64_64158

noncomputable
def f : ℕ → ℕ := sorry

axiom f_condition_1 : f 1 = 1
axiom f_condition_2 : ∀ n : ℕ, 3 * f n * f (2 * n + 1) = f (2 * n) * (1 + 3 * f n)
axiom f_condition_3 : ∀ n : ℕ, f (2 * n) < 6 * f n

theorem find_f_k_l (k l : ℕ) (h : k < l) : 
  (f k + f l = 293) ↔ 
  ((k = 121 ∧ l = 4) ∨ (k = 118 ∧ l = 4) ∨ 
   (k = 109 ∧ l = 16) ∨ (k = 16 ∧ l = 109)) := 
by 
  sorry

end find_f_k_l_l64_64158


namespace fill_tub_in_seconds_l64_64438

theorem fill_tub_in_seconds 
  (faucet_rate : ℚ)
  (four_faucet_rate : ℚ := 4 * faucet_rate)
  (three_faucet_rate : ℚ := 3 * faucet_rate)
  (time_for_100_gallons_in_minutes : ℚ := 6)
  (time_for_100_gallons_in_seconds : ℚ := time_for_100_gallons_in_minutes * 60)
  (volume_100_gallons : ℚ := 100)
  (rate_per_three_faucets_in_gallons_per_second : ℚ := volume_100_gallons / time_for_100_gallons_in_seconds)
  (rate_per_faucet : ℚ := rate_per_three_faucets_in_gallons_per_second / 3)
  (rate_per_four_faucets : ℚ := 4 * rate_per_faucet)
  (volume_50_gallons : ℚ := 50)
  (expected_time_seconds : ℚ := volume_50_gallons / rate_per_four_faucets) :
  expected_time_seconds = 135 :=
sorry

end fill_tub_in_seconds_l64_64438


namespace christian_sue_need_more_money_l64_64869

-- Definitions based on the given conditions
def bottle_cost : ℕ := 50
def christian_initial : ℕ := 5
def sue_initial : ℕ := 7
def christian_mowing_rate : ℕ := 5
def christian_mowing_count : ℕ := 4
def sue_walking_rate : ℕ := 2
def sue_walking_count : ℕ := 6

-- Prove that Christian and Sue will need 6 more dollars to buy the bottle of perfume
theorem christian_sue_need_more_money :
  let christian_earning := christian_mowing_rate * christian_mowing_count
  let christian_total := christian_initial + christian_earning
  let sue_earning := sue_walking_rate * sue_walking_count
  let sue_total := sue_initial + sue_earning
  let total_money := christian_total + sue_total
  50 - total_money = 6 :=
by
  sorry

end christian_sue_need_more_money_l64_64869


namespace geometric_sequence_a10_a11_l64_64160

noncomputable def a (n : ℕ) : ℝ := sorry  -- define the geometric sequence {a_n}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n m, a (n + m) = a n * q^m

variables (a : ℕ → ℝ) (q : ℝ)

-- Conditions given in the problem
axiom h1 : a 1 + a 5 = 5
axiom h2 : a 4 + a 5 = 15
axiom geom_seq : is_geometric_sequence a q

theorem geometric_sequence_a10_a11 : a 10 + a 11 = 135 :=
by {
  sorry
}

end geometric_sequence_a10_a11_l64_64160


namespace number_of_valid_three_digit_even_numbers_l64_64921

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l64_64921


namespace rectangle_dimension_correct_l64_64525

-- Definition of the Width and Length based on given conditions
def width := 3 / 2
def length := 3

-- Perimeter and Area conditions
def perimeter_condition (w l : ℝ) := 2 * (w + l) = 2 * (w * l)
def length_condition (w l : ℝ) := l = 2 * w

-- Main theorem statement
theorem rectangle_dimension_correct :
  ∃ (w l : ℝ), perimeter_condition w l ∧ length_condition w l ∧ w = width ∧ l = length :=
by {
  -- add sorry to skip the proof
  sorry
}

end rectangle_dimension_correct_l64_64525


namespace negation_equiv_no_solution_l64_64840

-- Definition of there is at least one solution
def at_least_one_solution (P : α → Prop) : Prop := ∃ x, P x

-- Definition of no solution
def no_solution (P : α → Prop) : Prop := ∀ x, ¬ P x

-- Problem statement to prove that the negation of at_least_one_solution is equivalent to no_solution
theorem negation_equiv_no_solution (P : α → Prop) :
  ¬ at_least_one_solution P ↔ no_solution P := 
sorry

end negation_equiv_no_solution_l64_64840


namespace find_a_l64_64479

theorem find_a (a : ℝ) (x : ℝ) :
  (∃ b : ℝ, (9 * x^2 - 18 * x + a) = (3 * x + b) ^ 2) → a = 9 := by
  sorry

end find_a_l64_64479


namespace alayas_fruit_salads_l64_64716

theorem alayas_fruit_salads (A : ℕ) (H1 : 2 * A + A = 600) : A = 200 := 
by
  sorry

end alayas_fruit_salads_l64_64716


namespace people_per_car_l64_64706

theorem people_per_car (total_people : ℕ) (total_cars : ℕ) (h_people : total_people = 63) (h_cars : total_cars = 3) : 
  total_people / total_cars = 21 := by
  sorry

end people_per_car_l64_64706


namespace three_digit_even_sum_12_l64_64918

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l64_64918


namespace equilateral_triangle_of_equal_heights_and_inradius_l64_64165

theorem equilateral_triangle_of_equal_heights_and_inradius 
  {a b c h1 h2 h3 r : ℝ} (h1_eq : h1 = 2 * r * (a * b * c) / a) 
  (h2_eq : h2 = 2 * r * (a * b * c) / b) 
  (h3_eq : h3 = 2 * r * (a * b * c) / c) 
  (sum_heights_eq : h1 + h2 + h3 = 9 * r) : a = b ∧ b = c ∧ c = a :=
by
  sorry

end equilateral_triangle_of_equal_heights_and_inradius_l64_64165


namespace _l64_64881

noncomputable theorem unique_solution_x : (∃ x : ℝ, 0 < x ∧ x \sqrt(16 - x) + \sqrt(16 * x - x^3) ≥ 16) :=
  sorry

end _l64_64881


namespace simplify_expression_l64_64814

variable (a b : ℝ)

theorem simplify_expression (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) :
  (3 * (a^2 + a * b + b^2) / (4 * (a + b))) * (2 * (a^2 - b^2) / (9 * (a^3 - b^3))) = 
  1 / 6 := 
by
  -- Placeholder for proof steps
  sorry

end simplify_expression_l64_64814


namespace y_intercept_of_line_l64_64034

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end y_intercept_of_line_l64_64034


namespace train_length_l64_64402

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 52) (h2 : time_sec = 9) (h3 : length_m = 129.96) : 
  length_m = (speed_km_hr * 1000 / 3600) * time_sec := 
sorry

end train_length_l64_64402


namespace tower_count_l64_64553

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binom (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def multinomialCoeff (n : Nat) (ks : List Nat) : Nat :=
  factorial n / List.foldr (fun k acc => acc * factorial k) 1 ks

theorem tower_count :
  let totalCubes := 9
  let usedCubes := 8
  let redCubes := 2
  let blueCubes := 3
  let greenCubes := 4
  multinomialCoeff totalCubes [redCubes, blueCubes, greenCubes] = 1260 :=
by
  sorry

end tower_count_l64_64553


namespace find_c_plus_d_l64_64738

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then 2 * c * x + d else 9 - 2 * x

theorem find_c_plus_d (c d : ℝ) (h : ∀ x : ℝ, f c d (f c d x) = x) : c + d = 4.25 :=
by
  sorry

end find_c_plus_d_l64_64738


namespace inequality_solution_l64_64363

theorem inequality_solution (x : ℝ) :
  (x < -2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) ∨ 7 < x) →
  (1 / (x - 1)) - (4 / (x - 2)) + (4 / (x - 3)) - (1 / (x - 4)) < 1 / 30 :=
by
  sorry

end inequality_solution_l64_64363


namespace gcd_115_161_l64_64834

theorem gcd_115_161 : Nat.gcd 115 161 = 23 := by
  sorry

end gcd_115_161_l64_64834


namespace fraction_disliking_but_liking_l64_64567

-- Definitions based on conditions
def total_students : ℕ := 100
def like_dancing : ℕ := 70
def dislike_dancing : ℕ := total_students - like_dancing

def say_they_like_dancing (like_dancing : ℕ) : ℕ := (70 * like_dancing) / 100
def say_they_dislike_dancing (like_dancing : ℕ) : ℕ := like_dancing - say_they_like_dancing like_dancing

def dislike_and_say_dislike (dislike_dancing : ℕ) : ℕ := (80 * dislike_dancing) / 100
def say_dislike_but_like (like_dancing : ℕ) : ℕ := say_they_dislike_dancing like_dancing

def total_say_dislike : ℕ := dislike_and_say_dislike dislike_dancing + say_dislike_but_like like_dancing

noncomputable def fraction_like_but_say_dislike : ℚ := (say_dislike_but_like like_dancing : ℚ) / (total_say_dislike : ℚ)

theorem fraction_disliking_but_liking : fraction_like_but_say_dislike = 46.67 / 100 := 
by sorry

end fraction_disliking_but_liking_l64_64567


namespace loan_amount_principal_l64_64642

-- Definitions based on conditions
def rate_of_interest := 3
def time_period := 3
def simple_interest := 108

-- Question translated to Lean 4 statement
theorem loan_amount_principal : ∃ P, (simple_interest = (P * rate_of_interest * time_period) / 100) ∧ P = 1200 :=
sorry

end loan_amount_principal_l64_64642


namespace smallest_digits_to_append_l64_64234

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l64_64234


namespace abs_inequality_condition_l64_64295

theorem abs_inequality_condition (a : ℝ) : 
  (a < 2) ↔ ∀ x : ℝ, |x - 2| + |x| > a :=
sorry

end abs_inequality_condition_l64_64295


namespace sum_of_roots_of_quadratic_l64_64054

theorem sum_of_roots_of_quadratic :
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16 in
  (∀ x, f x = 0 → x = 7 ∨ x = -1) →
  (let sum_of_roots := 7 + (-1) in sum_of_roots = 6) :=
by
  sorry

end sum_of_roots_of_quadratic_l64_64054


namespace y_intercept_3x_minus_4y_eq_12_l64_64035

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end y_intercept_3x_minus_4y_eq_12_l64_64035


namespace find_multiplier_l64_64650

/-- Define the number -/
def number : ℝ := -10.0

/-- Define the multiplier m -/
def m : ℝ := 0.4

/-- Given conditions and prove the correct multiplier -/
theorem find_multiplier (number : ℝ) (m : ℝ) 
  (h1 : ∃ m : ℝ, m * number - 8 = -12) 
  (h2 : number = -10.0) : m = 0.4 :=
by
  -- We skip the actual steps and provide the answer using sorry
  sorry

end find_multiplier_l64_64650


namespace rachel_plants_lamps_l64_64359

-- Define the conditions as types
def plants : Type := { fern1 : Prop // true } × { fern2 : Prop // true } × { cactus : Prop // true }
def lamps : Type := { yellow1 : Prop // true } × { yellow2 : Prop // true } × { blue1 : Prop // true } × { blue2 : Prop // true }

-- A function that counts the distribution of plants under lamps
noncomputable def count_ways (p : plants) (l : lamps) : ℕ :=
  -- Here we should define the function that counts the number of configurations, 
  -- but since we are only defining the problem here we'll skip this part.
  sorry

-- The statement to prove
theorem rachel_plants_lamps :
  ∀ (p : plants) (l : lamps), count_ways p l = 14 :=
by
  sorry

end rachel_plants_lamps_l64_64359


namespace smallest_digits_to_append_l64_64244

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l64_64244


namespace dave_final_tickets_l64_64067

-- Define the initial number of tickets and operations
def initial_tickets : ℕ := 25
def tickets_spent_on_beanie : ℕ := 22
def tickets_won_after : ℕ := 15

-- Define the final number of tickets function
def final_tickets (initial : ℕ) (spent : ℕ) (won : ℕ) : ℕ :=
  initial - spent + won

-- Theorem stating that Dave would end up with 18 tickets given the conditions
theorem dave_final_tickets : final_tickets initial_tickets tickets_spent_on_beanie tickets_won_after = 18 :=
by
  -- Proof to be filled in
  sorry

end dave_final_tickets_l64_64067


namespace election_margin_of_victory_l64_64373

theorem election_margin_of_victory (T : ℕ) (H_winning_votes : T * 58 / 100 = 1044) :
  1044 - (T * 42 / 100) = 288 :=
by
  sorry

end election_margin_of_victory_l64_64373


namespace philip_paints_2_per_day_l64_64357

def paintings_per_day (initial_paintings total_paintings days : ℕ) : ℕ :=
  (total_paintings - initial_paintings) / days

theorem philip_paints_2_per_day :
  paintings_per_day 20 80 30 = 2 :=
by
  sorry

end philip_paints_2_per_day_l64_64357


namespace area_of_square_field_l64_64649

theorem area_of_square_field (s : ℕ) (area : ℕ) (cost_per_meter : ℕ) (total_cost : ℕ) (gate_width : ℕ) :
  (cost_per_meter = 3) →
  (total_cost = 1998) →
  (gate_width = 1) →
  (total_cost = cost_per_meter * (4 * s - 2 * gate_width)) →
  (area = s^2) →
  area = 27889 :=
by
  intros h_cost_per_meter h_total_cost h_gate_width h_cost_eq h_area_eq
  sorry

end area_of_square_field_l64_64649


namespace slope_proof_l64_64045

noncomputable def slope_between_midpoints : ℚ :=
  let p1 := (2, 3)
  let p2 := (4, 5)
  let q1 := (7, 3)
  let q2 := (8, 7)

  let midpoint (a b : ℚ × ℚ) : ℚ × ℚ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

  let m1 := midpoint p1 p2
  let m2 := midpoint q1 q2

  (m2.2 - m1.2) / (m2.1 - m1.1)

theorem slope_proof : slope_between_midpoints = 2 / 9 := by
  sorry

end slope_proof_l64_64045


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l64_64932

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l64_64932


namespace cherry_pie_probability_l64_64546

noncomputable def probability_of_cherry_pie : Real :=
  let packets := ["KK", "KV", "VV"]
  let prob :=
    (1/3 * 1/4) + -- Case KK broken, then picking from KV or VV
    (1/6 * 1/2) + -- Case KV broken (cabbage found), picking cherry from KV
    (1/3 * 1) + -- Case VV broken (cherry found), remaining cherry picked
    (1/6 * 0) -- Case KV broken (cherry found), remaining cabbage
  prob

theorem cherry_pie_probability : probability_of_cherry_pie = 2 / 3 :=
  sorry

end cherry_pie_probability_l64_64546


namespace find_a_l64_64478

theorem find_a (a : ℝ) (x : ℝ) :
  (∃ b : ℝ, (9 * x^2 - 18 * x + a) = (3 * x + b) ^ 2) → a = 9 := by
  sorry

end find_a_l64_64478


namespace work_completion_time_l64_64061

theorem work_completion_time (W : ℝ) : 
  let A_effort := 1 / 11
  let B_effort := 1 / 20
  let C_effort := 1 / 55
  (2 * A_effort + B_effort + C_effort) = 1 / 4 → 
  8 * (2 * A_effort + B_effort + C_effort) = 1 :=
by { sorry }

end work_completion_time_l64_64061


namespace number_of_valid_three_digit_even_numbers_l64_64920

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l64_64920


namespace line_passes_through_vertex_twice_l64_64897

theorem line_passes_through_vertex_twice :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ (∀ a, (y = 2 * x + a ∧ ∃ (x y : ℝ), y = x^2 + 2 * a^2) ↔ a = a₁ ∨ a = a₂) :=
by
  sorry

end line_passes_through_vertex_twice_l64_64897


namespace geometric_series_common_ratio_l64_64086

theorem geometric_series_common_ratio
    (a : ℝ) (S : ℝ) (r : ℝ)
    (h_a : a = 512)
    (h_S : S = 3072)
    (h_sum : S = a / (1 - r)) : 
    r = 5 / 6 :=
by 
  rw [h_a] at h_sum
  rw [h_S] at h_sum
  sorry

end geometric_series_common_ratio_l64_64086


namespace square_area_l64_64861

theorem square_area (side_length : ℝ) (h : side_length = 11) : side_length * side_length = 121 := 
by 
  simp [h]
  sorry

end square_area_l64_64861


namespace age_difference_l64_64390

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 13) : A = C + 13 :=
by
  sorry

end age_difference_l64_64390


namespace attendance_difference_is_85_l64_64673

def saturday_attendance : ℕ := 80
def monday_attendance : ℕ := saturday_attendance - 20
def wednesday_attendance : ℕ := monday_attendance + 50
def friday_attendance : ℕ := saturday_attendance + monday_attendance
def thursday_attendance : ℕ := 45
def expected_audience : ℕ := 350

def total_attendance : ℕ := 
  saturday_attendance + 
  monday_attendance + 
  wednesday_attendance + 
  friday_attendance + 
  thursday_attendance

def more_people_attended_than_expected : ℕ :=
  total_attendance - expected_audience

theorem attendance_difference_is_85 : more_people_attended_than_expected = 85 := 
by
  unfold more_people_attended_than_expected
  unfold total_attendance
  unfold saturday_attendance
  unfold monday_attendance
  unfold wednesday_attendance
  unfold friday_attendance
  unfold thursday_attendance
  unfold expected_audience
  exact sorry

end attendance_difference_is_85_l64_64673


namespace additional_men_joined_l64_64656

theorem additional_men_joined
    (M : ℕ) (X : ℕ)
    (h1 : M = 20)
    (h2 : M * 50 = (M + X) * 25) :
    X = 20 := by
  sorry

end additional_men_joined_l64_64656


namespace _l64_64882

noncomputable theorem unique_solution_x : (∃ x : ℝ, 0 < x ∧ x \sqrt(16 - x) + \sqrt(16 * x - x^3) ≥ 16) :=
  sorry

end _l64_64882


namespace effective_annual_rate_l64_64544

theorem effective_annual_rate (i : ℚ) (n : ℕ) (h_i : i = 0.16) (h_n : n = 2) :
  (1 + i / n) ^ n - 1 = 0.1664 :=
by {
  sorry
}

end effective_annual_rate_l64_64544


namespace evaluate_magnitude_l64_64731

noncomputable def z1 : ℂ := 3 * Real.sqrt 2 - 3 * Complex.I
noncomputable def z2 : ℂ := 2 * Real.sqrt 3 + 6 * Complex.I

theorem evaluate_magnitude :
  abs (z1 * z2) = 36 := by
sorrry

end evaluate_magnitude_l64_64731


namespace square_of_binomial_l64_64481

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b) ^ 2 = 9 * x^2 - 18 * x + a) ↔ a = 9 :=
by
  sorry

end square_of_binomial_l64_64481


namespace expression_constant_value_l64_64813

theorem expression_constant_value (a b x y : ℝ) 
  (h_a : a = Real.sqrt (1 + x^2))
  (h_b : b = Real.sqrt (1 + y^2)) 
  (h_xy : x + y = 1) : 
  (a + b + 1) * (a + b - 1) * (a - b + 1) * (-a + b + 1) = 4 := 
by 
  sorry

end expression_constant_value_l64_64813


namespace find_d_l64_64497

theorem find_d (d : ℝ) (h : ∃ (x y : ℝ), 3 * x + 5 * y + d = 0 ∧ x = -d / 3 ∧ y = -d / 5 ∧ -d / 3 + (-d / 5) = 15) : d = -225 / 8 :=
by 
  sorry

end find_d_l64_64497


namespace part_I_min_value_part_II_nonexistence_l64_64121

theorem part_I_min_value (a b : ℝ) (hab : a > 0 ∧ b > 0 ∧ a + 4 * b = (a * b)^(3/2)) : a^2 + 16 * b^2 ≥ 32 :=
by
  sorry

theorem part_II_nonexistence (a b : ℝ) (hab : a > 0 ∧ b > 0 ∧ a + 4 * b = (a * b)^(3/2)) : ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 6 :=
by
  sorry

end part_I_min_value_part_II_nonexistence_l64_64121


namespace find_alpha_plus_beta_l64_64138

open Real

theorem find_alpha_plus_beta 
  (α β : ℝ)
  (h1 : sin α = sqrt 5 / 5)
  (h2 : sin β = sqrt 10 / 10)
  (h3 : π / 2 < α ∧ α < π)
  (h4 : π / 2 < β ∧ β < π) :
  α + β = 7 * π / 4 :=
sorry

end find_alpha_plus_beta_l64_64138


namespace reflection_coordinates_l64_64519

-- Define the original coordinates of point M
def original_point : (ℝ × ℝ) := (3, -4)

-- Define the function to reflect a point across the x-axis
def reflect_across_x_axis (p: ℝ × ℝ) : (ℝ × ℝ) :=
  (p.1, -p.2)

-- State the theorem to prove the coordinates after reflection
theorem reflection_coordinates :
  reflect_across_x_axis original_point = (3, 4) :=
by
  sorry

end reflection_coordinates_l64_64519


namespace extreme_values_l64_64451

-- Define the function f(x) with symbolic constants a and b
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x

-- Given conditions
def intersects_at_1_0 (a b : ℝ) : Prop := (f 1 a b = 0)
def derivative_at_1_0 (a b : ℝ) : Prop := (3 - 2 * a - b = 0)

-- Main theorem statement
theorem extreme_values (a b : ℝ) (h1 : intersects_at_1_0 a b) (h2 : derivative_at_1_0 a b) :
  (∀ x, f x a b ≤ 4 / 27) ∧ (∀ x, 0 ≤ f x a b) :=
sorry

end extreme_values_l64_64451


namespace maci_pays_total_cost_l64_64983

def cost_blue_pen : ℝ := 0.10
def num_blue_pens : ℕ := 10
def num_red_pens : ℕ := 15
def cost_red_pen : ℝ := 2 * cost_blue_pen

def total_cost : ℝ := num_blue_pens * cost_blue_pen + num_red_pens * cost_red_pen

theorem maci_pays_total_cost : total_cost = 4 := by
  -- Proof goes here
  sorry

end maci_pays_total_cost_l64_64983


namespace math_problem_l64_64635

open Real

theorem math_problem
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (hxyz : x + y + z = 1) :
  ( (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28 / 3)^3 ) :=
by {
  sorry
}

end math_problem_l64_64635


namespace share_price_increase_l64_64409

theorem share_price_increase
  (P : ℝ)
  -- At the end of the first quarter, the share price was 20% higher than at the beginning of the year.
  (end_of_first_quarter : ℝ := 1.20 * P)
  -- The percent increase from the end of the first quarter to the end of the second quarter was 25%.
  (percent_increase_second_quarter : ℝ := 0.25)
  -- At the end of the second quarter, the share price
  (end_of_second_quarter : ℝ := end_of_first_quarter + percent_increase_second_quarter * end_of_first_quarter) :
  -- What is the percent increase in share price at the end of the second quarter compared to the beginning of the year?
  end_of_second_quarter = 1.50 * P :=
by
  sorry

end share_price_increase_l64_64409


namespace same_color_eye_proportion_l64_64279

theorem same_color_eye_proportion :
  ∀ (a b c d e f : ℝ),
  a + b + c = 0.30 →
  a + d + e = 0.40 →
  b + d + f = 0.50 →
  a + b + c + d + e + f = 1 →
  c + e + f = 0.80 :=
by
  intros a b c d e f h1 h2 h3 h4
  sorry

end same_color_eye_proportion_l64_64279


namespace point_reflection_example_l64_64323

def point := ℝ × ℝ

def reflect_x_axis (p : point) : point := (p.1, -p.2)

theorem point_reflection_example : reflect_x_axis (1, -2) = (1, 2) := sorry

end point_reflection_example_l64_64323


namespace sum_of_three_numbers_l64_64381

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 54) (h3 : c + a = 58) : 
  a + b + c = 73.5 :=
by
  sorry -- Proof is omitted

end sum_of_three_numbers_l64_64381


namespace simplify_and_evaluate_l64_64515

noncomputable def a := 2 * Real.sqrt 3 + 3
noncomputable def expr := (1 - 1 / (a - 2)) / ((a ^ 2 - 6 * a + 9) / (2 * a - 4))

theorem simplify_and_evaluate : expr = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l64_64515


namespace exists_nat_solution_for_A_415_l64_64590

theorem exists_nat_solution_for_A_415 : ∃ (m n : ℕ), 3 * m^2 * n = n^3 + 415 := by
  sorry

end exists_nat_solution_for_A_415_l64_64590


namespace smallest_number_of_digits_to_append_l64_64240

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l64_64240


namespace angle_AOC_is_minus_150_l64_64810

-- Define the conditions.
def rotate_counterclockwise (angle1 : Int) (angle2 : Int) : Int :=
  angle1 + angle2

-- The initial angle starts at 0°, rotates 120° counterclockwise, and then 270° clockwise
def angle_OA := 0
def angle_OB := rotate_counterclockwise angle_OA 120
def angle_OC := rotate_counterclockwise angle_OB (-270)

-- The theorem stating the resulting angle between OA and OC.
theorem angle_AOC_is_minus_150 : angle_OC = -150 := by
  sorry

end angle_AOC_is_minus_150_l64_64810


namespace pyramid_dihedral_angle_l64_64622

theorem pyramid_dihedral_angle 
  (k : ℝ) 
  (h_k_pos : 0 < k) :
  ∃ α : ℝ, α = 2 * Real.arccos (1 / Real.sqrt (Real.sqrt (4 * k))) :=
sorry

end pyramid_dihedral_angle_l64_64622


namespace y_intercept_of_line_l64_64028

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end y_intercept_of_line_l64_64028


namespace most_likely_parents_genotypes_l64_64806

-- Defining the probabilities of alleles in the population
def p_H : ℝ := 0.1
def q_S : ℝ := 0.9

-- Defining the genotypes and their corresponding fur types
inductive Genotype
| HH : Genotype
| HS : Genotype
| SS : Genotype
| Sh : Genotype

-- A function to determine if a given genotype results in hairy fur
def isHairy : Genotype → Prop
| Genotype.HH := true
| Genotype.HS := true
| _ := false

-- Axiom stating that all four offspring have hairy fur
axiom offspring_all_hairy (parent1 parent2 : Genotype) : 
  (isHairy parent1 ∧ isHairy parent2) ∨
  ((parent1 = Genotype.HH ∨ parent2 = Genotype.Sh) ∧ isHairy Genotype.HH) 

-- The main theorem to prove the genotypes of the parents
theorem most_likely_parents_genotypes : 
  ∃ parent1 parent2,
    parent1 = Genotype.HH ∧ parent2 = Genotype.Sh :=
begin
  sorry
end

end most_likely_parents_genotypes_l64_64806


namespace ab_leq_one_l64_64655

theorem ab_leq_one (a b x : ℝ) (h1 : (x + a) * (x + b) = 9) (h2 : x = a + b) : a * b ≤ 1 := 
sorry

end ab_leq_one_l64_64655


namespace count_even_three_digit_sum_tens_units_is_12_l64_64925

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l64_64925


namespace inverse_proportionality_l64_64548

theorem inverse_proportionality:
  (∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → y = k / x) ∧ y = 1 ∧ x = 2 →
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → y = 2 / x :=
by
  sorry

end inverse_proportionality_l64_64548


namespace find_x_plus_one_over_x_l64_64909

variable (x : ℝ)

theorem find_x_plus_one_over_x
  (h1 : x^3 + (1/x)^3 = 110)
  (h2 : (x + 1/x)^2 - 2*x - 2*(1/x) = 38) :
  x + 1/x = 5 :=
sorry

end find_x_plus_one_over_x_l64_64909


namespace find_angle_between_planes_l64_64744

noncomputable def angle_between_planes (α β : ℝ) : ℝ := Real.arcsin ((Real.sqrt 6 + 1) / 5)

theorem find_angle_between_planes (α β : ℝ) (h : α = β) : 
  (∃ (cube : Type) (A B C D A₁ B₁ C₁ D₁ : cube),
    α = Real.arcsin ((Real.sqrt 6 - 1) / 5) ∨ α = Real.arcsin ((Real.sqrt 6 + 1) / 5)) 
    :=
sorry

end find_angle_between_planes_l64_64744


namespace two_squares_always_similar_l64_64248

-- Define geometric shapes and their properties
inductive Shape
| Rectangle : Shape
| Rhombus   : Shape
| Square    : Shape
| RightAngledTriangle : Shape

-- Define similarity condition
def similar (s1 s2 : Shape) : Prop :=
  match s1, s2 with
  | Shape.Square, Shape.Square => true
  | _, _ => false

-- Prove that two squares are always similar
theorem two_squares_always_similar : similar Shape.Square Shape.Square = true :=
by
  sorry

end two_squares_always_similar_l64_64248


namespace zero_of_function_l64_64190

theorem zero_of_function : ∃ x : Real, 4 * x - 2 = 0 ∧ x = 1 / 2 :=
by
  sorry

end zero_of_function_l64_64190


namespace natural_solutions_3x_4y_eq_12_l64_64285

theorem natural_solutions_3x_4y_eq_12 :
  ∃ x y : ℕ, (3 * x + 4 * y = 12) ∧ ((x = 4 ∧ y = 0) ∨ (x = 0 ∧ y = 3)) := 
sorry

end natural_solutions_3x_4y_eq_12_l64_64285


namespace job_completion_time_l64_64140

theorem job_completion_time (initial_men : ℕ) (initial_days : ℕ) (extra_men : ℕ) (interval_days : ℕ) (total_days : ℕ) : 
  initial_men = 20 → 
  initial_days = 15 → 
  extra_men = 10 → 
  interval_days = 5 → 
  total_days = 12 → 
  ∀ n, (20 * 5 + (20 + 10) * 5 + (20 + 10 + 10) * n.succ = 300 → n + 10 + n.succ = 12) :=
by
  intro h1 h2 h3 h4 h5
  sorry

end job_completion_time_l64_64140


namespace chuck_total_time_on_trip_l64_64871

def distance_into_country : ℝ := 28.8
def rate_out : ℝ := 16
def rate_back : ℝ := 24

theorem chuck_total_time_on_trip : (distance_into_country / rate_out) + (distance_into_country / rate_back) = 3 := 
by sorry

end chuck_total_time_on_trip_l64_64871


namespace factorize_4a2_minus_9_factorize_2x2y_minus_8xy_plus_8y_l64_64878

-- Factorization of 4a^2 - 9 as (2a + 3)(2a - 3)
theorem factorize_4a2_minus_9 (a : ℝ) : 4 * a^2 - 9 = (2 * a + 3) * (2 * a - 3) :=
by 
  sorry

-- Factorization of 2x^2 y - 8xy + 8y as 2y(x-2)^2
theorem factorize_2x2y_minus_8xy_plus_8y (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2) ^ 2 :=
by 
  sorry

end factorize_4a2_minus_9_factorize_2x2y_minus_8xy_plus_8y_l64_64878


namespace pascal_triangle_contains_53_once_l64_64465

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l64_64465


namespace sum_of_numbers_l64_64046

theorem sum_of_numbers : (4.75 + 0.303 + 0.432) = 5.485 := 
by  
  sorry

end sum_of_numbers_l64_64046


namespace recommended_daily_serving_l64_64360

theorem recommended_daily_serving (mg_per_pill : ℕ) (pills_per_week : ℕ) (total_mg_week : ℕ) (days_per_week : ℕ) 
  (h1 : mg_per_pill = 50) (h2 : pills_per_week = 28) (h3 : total_mg_week = pills_per_week * mg_per_pill) 
  (h4 : days_per_week = 7) : 
  total_mg_week / days_per_week = 200 :=
by
  sorry

end recommended_daily_serving_l64_64360


namespace digit_of_fraction_l64_64535

theorem digit_of_fraction (n : ℕ) : (15 / 37 : ℝ) = 0.405 ∧ 415 % 3 = 1 → ∃ d : ℕ, d = 4 :=
by
  sorry

end digit_of_fraction_l64_64535


namespace pond_length_l64_64180

theorem pond_length (
    W L P : ℝ) 
    (h1 : L = 2 * W) 
    (h2 : L = 32) 
    (h3 : (L * W) / 8 = P^2) : 
  P = 8 := 
by 
  sorry

end pond_length_l64_64180


namespace find_a_l64_64486

theorem find_a (a : ℝ) : (∀ x : ℝ, |x - a| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  intro h
  have h1 : |(-1 : ℝ) - a| = 3 := sorry
  have h2 : |(5 : ℝ) - a| = 3 := sorry
  sorry

end find_a_l64_64486


namespace least_integer_value_l64_64536

theorem least_integer_value :
  ∃ x : ℤ, (∀ x' : ℤ, (|3 * x' + 4| <= 18) → (x' >= x)) ∧ (|3 * x + 4| <= 18) ∧ x = -7 := 
sorry

end least_integer_value_l64_64536


namespace marbles_problem_l64_64332

theorem marbles_problem (p : ℕ) (m n r : ℕ) 
(hp : Nat.Prime p) 
(h1 : p = 2017)
(h2 : N = p^m * n)
(h3 : ¬ p ∣ n)
(h4 : r = n % p) 
(h N : ∀ (N : ℕ), N = 3 * p * 632 - 1)
: p * m + r = 3913 := 
sorry

end marbles_problem_l64_64332


namespace gumballs_initial_count_l64_64575

theorem gumballs_initial_count (x : ℝ) (h : (0.75 ^ 3) * x = 27) : x = 64 :=
by
  sorry

end gumballs_initial_count_l64_64575


namespace val_of_7c_plus_7d_l64_64785

noncomputable def h (x : ℝ) : ℝ := 7 * x - 6

noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 4

noncomputable def f (c d x : ℝ) : ℝ := c * x + d

theorem val_of_7c_plus_7d (c d : ℝ) (h_eq : ∀ x, h x = f_inv x - 2) 
  (inv_prop : ∀ x, f c d (f_inv x) = x) : 7 * c + 7 * d = 5 :=
by
  sorry

end val_of_7c_plus_7d_l64_64785


namespace initial_contribution_l64_64331

theorem initial_contribution (j k l : ℝ)
  (h1 : j + k + l = 1200)
  (h2 : j - 200 + 3 * (k + l) = 1800) :
  j = 800 :=
sorry

end initial_contribution_l64_64331


namespace amare_additional_fabric_needed_l64_64991

-- Defining the conditions
def yards_per_dress : ℝ := 5.5
def num_dresses : ℝ := 4
def initial_fabric_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- The theorem to prove
theorem amare_additional_fabric_needed : 
  (yards_per_dress * num_dresses * yard_to_feet) - initial_fabric_feet = 59 := 
by
  sorry

end amare_additional_fabric_needed_l64_64991


namespace complex_values_l64_64742

open Complex

theorem complex_values (a b : ℝ) (i : ℂ) (h1 : i = Complex.I) (h2 : a - b * i = (1 + i) * i^3) : a = 1 ∧ b = -1 :=
by
  sorry

end complex_values_l64_64742


namespace probability_red_or_white_is_11_over_13_l64_64393

-- Given data
def total_marbles : ℕ := 60
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - blue_marbles - red_marbles

def blue_size : ℕ := 2
def red_size : ℕ := 1
def white_size : ℕ := 1

-- Total size value of all marbles
def total_size_value : ℕ := (blue_size * blue_marbles) + (red_size * red_marbles) + (white_size * white_marbles)

-- Probability of selecting a red or white marble
def probability_red_or_white : ℚ := (red_size * red_marbles + white_size * white_marbles) / total_size_value

-- Theorem to prove
theorem probability_red_or_white_is_11_over_13 : probability_red_or_white = 11 / 13 :=
by sorry

end probability_red_or_white_is_11_over_13_l64_64393


namespace triangle_side_length_l64_64319

theorem triangle_side_length 
  (A : ℝ) (a m n : ℝ) 
  (hA : A = 60) 
  (h1 : m + n = 7) 
  (h2 : m * n = 11) : a = 4 :=
by
  sorry

end triangle_side_length_l64_64319


namespace minimize_surface_area_l64_64444

theorem minimize_surface_area (V r h : ℝ) (hV : V = π * r^2 * h) (hA : 2 * π * r^2 + 2 * π * r * h = 2 * π * r^2 + 2 * π * r * h) : 
  (h / r) = 2 := 
by
  sorry

end minimize_surface_area_l64_64444


namespace polynomial_product_expansion_l64_64109

theorem polynomial_product_expansion (x : ℝ) : (x^2 + 3 * x + 3) * (x^2 - 3 * x + 3) = x^4 - 3 * x^2 + 9 := 
by sorry

end polynomial_product_expansion_l64_64109


namespace find_number_l64_64760

theorem find_number (x : ℚ) (h : x / 5 = 3 * (x / 6) - 40) : x = 400 / 3 :=
sorry

end find_number_l64_64760


namespace solve_log_eq_l64_64828

theorem solve_log_eq (x : ℝ) (h1 : x > 0) (h2 : 2 * x + 1 > 0) : log 10 (2 * x + 1) + log 10 x = 1 → x = 2 :=
by
  intro h
  sorry

end solve_log_eq_l64_64828


namespace count_even_three_digit_numbers_with_sum_12_l64_64935

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l64_64935


namespace find_sides_of_rectangle_l64_64397

-- Define the conditions
def isRectangle (w l : ℝ) : Prop :=
  l = 3 * w ∧ 2 * l + 2 * w = l * w

-- Main theorem statement
theorem find_sides_of_rectangle (w l : ℝ) :
  isRectangle w l → w = 8 / 3 ∧ l = 8 :=
by
  sorry

end find_sides_of_rectangle_l64_64397


namespace project_completion_time_l64_64761

-- Definitions based on conditions
def start_time : Time := ⟨8, 0⟩ -- 8:00 AM

def duration_to_join_b_and_c : Duration := 27.minutes

def time_to_complete_a := 6 -- hours
def time_to_complete_b := 4 -- hours
def time_to_complete_c := 5 -- hours

-- Theorem statement proving project completion time
theorem project_completion_time :
    ∃ end_time : Time, 
    start_time + duration_to_join_b_and_c + (37/40 / (1/6 + 1/4 + 1/5)) = end_time :=
begin
    use ⟨9, 57⟩, -- Expected completion time 9:57 AM
    sorry
end

end project_completion_time_l64_64761


namespace jenny_sold_boxes_l64_64326

-- Given conditions as definitions
def cases : ℕ := 3
def boxes_per_case : ℕ := 8

-- Mathematically equivalent proof problem
theorem jenny_sold_boxes : cases * boxes_per_case = 24 := by
  sorry

end jenny_sold_boxes_l64_64326


namespace find_original_price_l64_64517

-- Given conditions:
-- 1. 10% cashback
-- 2. $25 mail-in rebate
-- 3. Final cost is $110

def original_price (P : ℝ) (cashback : ℝ) (rebate : ℝ) (final_cost : ℝ) :=
  final_cost = P - (cashback * P + rebate)

theorem find_original_price :
  ∀ (P : ℝ), original_price P 0.10 25 110 → P = 150 :=
by
  sorry

end find_original_price_l64_64517


namespace matrix_A_pow_100_eq_l64_64779

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 1], ![-9, -2]]

theorem matrix_A_pow_100_eq : matrix_A ^ 100 = ![![301, 100], ![-900, -299]] :=
  sorry

end matrix_A_pow_100_eq_l64_64779


namespace y_intercept_3x_minus_4y_eq_12_l64_64036

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end y_intercept_3x_minus_4y_eq_12_l64_64036


namespace pipes_fill_cistern_in_12_minutes_l64_64708

noncomputable def time_to_fill_cistern_with_pipes (A_fill : ℝ) (B_fill : ℝ) (C_empty : ℝ) : ℝ :=
  let A_rate := 1 / (12 * 3)          -- Pipe A's rate
  let B_rate := 1 / (8 * 3)           -- Pipe B's rate
  let C_rate := -1 / 24               -- Pipe C's rate
  let combined_rate := A_rate + B_rate - C_rate
  (1 / 3) / combined_rate             -- Time to fill remaining one-third

theorem pipes_fill_cistern_in_12_minutes :
  time_to_fill_cistern_with_pipes 12 8 24 = 12 :=
by
  sorry

end pipes_fill_cistern_in_12_minutes_l64_64708


namespace sum_of_roots_of_quadratic_l64_64051

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end sum_of_roots_of_quadratic_l64_64051


namespace original_solution_concentration_l64_64063

variable (C : ℝ) -- Concentration of the original solution as a percentage.
variable (v_orig : ℝ := 12) -- 12 ounces of the original vinegar solution.
variable (w_added : ℝ := 50) -- 50 ounces of water added.
variable (v_final_pct : ℝ := 7) -- Final concentration of 7%.

theorem original_solution_concentration :
  (C / 100 * v_orig = v_final_pct / 100 * (v_orig + w_added)) →
  C = (v_final_pct * (v_orig + w_added)) / v_orig :=
sorry

end original_solution_concentration_l64_64063


namespace unit_circle_arc_length_l64_64970

theorem unit_circle_arc_length (r : ℝ) (A : ℝ) (θ : ℝ) : r = 1 ∧ A = 1 ∧ A = (1 / 2) * r^2 * θ → r * θ = 2 :=
by
  -- Given r = 1 (radius of unit circle) and area A = 1
  -- A = (1 / 2) * r^2 * θ is the formula for the area of the sector
  sorry

end unit_circle_arc_length_l64_64970


namespace fraction_is_two_thirds_l64_64586

noncomputable def fraction_of_price_of_ballet_slippers (f : ℚ) : Prop :=
  let price_high_heels := 60
  let num_ballet_slippers := 5
  let total_cost := 260
  price_high_heels + num_ballet_slippers * f * price_high_heels = total_cost

theorem fraction_is_two_thirds : fraction_of_price_of_ballet_slippers (2 / 3) := by
  sorry

end fraction_is_two_thirds_l64_64586


namespace minimum_n_for_factorable_polynomial_l64_64286

theorem minimum_n_for_factorable_polynomial :
  ∃ n : ℤ, (∀ A B : ℤ, 5 * A = 48 → 5 * B + A = n) ∧
  (∀ k : ℤ, (∀ A B : ℤ, 5 * A * B = 48 → 5 * B + A = k) → n ≤ k) :=
by
  sorry

end minimum_n_for_factorable_polynomial_l64_64286


namespace cubes_difference_l64_64435

theorem cubes_difference :
  let a := 642
  let b := 641
  a^3 - b^3 = 1234567 :=
by
  let a := 642
  let b := 641
  have h : a^3 - b^3 = 264609288 - 263374721 := sorry
  have h_correct : 264609288 - 263374721 = 1234567 := sorry
  exact Eq.trans h h_correct

end cubes_difference_l64_64435


namespace jackson_total_souvenirs_l64_64152

theorem jackson_total_souvenirs 
  (num_hermit_crabs : ℕ)
  (spiral_shells_per_hermit_crab : ℕ) 
  (starfish_per_spiral_shell : ℕ) :
  (num_hermit_crabs = 45) → 
  (spiral_shells_per_hermit_crab = 3) → 
  (starfish_per_spiral_shell = 2) →
  (45 + 45 * 3 + 45 * 3 * 2 = 450) :=
by
  intros h0 h1 h2
  rw [h0, h1, h2]
  rfl

end jackson_total_souvenirs_l64_64152


namespace line_through_vertex_has_two_a_values_l64_64898

-- Definitions for the line and parabola as conditions
def line_eq (a x : ℝ) : ℝ := 2 * x + a
def parabola_eq (a x : ℝ) : ℝ := x^2 + 2 * a^2

-- The proof problem
theorem line_through_vertex_has_two_a_values :
  (∃ a1 a2 : ℝ, (a1 ≠ a2) ∧ (line_eq a1 0 = parabola_eq a1 0) ∧ (line_eq a2 0 = parabola_eq a2 0)) ∧
  (∀ a : ℝ, line_eq a 0 = parabola_eq a 0 → (a = 0 ∨ a = 1/2)) :=
sorry

end line_through_vertex_has_two_a_values_l64_64898


namespace cumulative_profit_exceeds_technical_renovation_expressions_for_A_n_B_n_l64_64260

noncomputable def A_n (n : ℕ) : ℝ :=
  490 * n - 10 * n^2

noncomputable def B_n (n : ℕ) : ℝ :=
  500 * n + 400 - 500 / 2^(n-1)

theorem cumulative_profit_exceeds_technical_renovation :
  ∀ n : ℕ, n ≥ 4 → B_n n > A_n n :=
by
  sorry  -- Proof goes here

theorem expressions_for_A_n_B_n (n : ℕ) :
  A_n n = 490 * n - 10 * n^2 ∧
  B_n n = 500 * n + 400 - 500 / 2^(n-1) :=
by
  sorry  -- Proof goes here

end cumulative_profit_exceeds_technical_renovation_expressions_for_A_n_B_n_l64_64260


namespace sum_roots_eq_six_l64_64056

theorem sum_roots_eq_six : 
  ∀ x : ℝ, (x - 3) ^ 2 = 16 → (x - 3 = 4 ∨ x - 3 = -4) → (let x₁ := 3 + 4 in let x₂ := 3 - 4 in x₁ + x₂ = 6) := by
  sorry

end sum_roots_eq_six_l64_64056


namespace central_projection_preserves_lines_l64_64009

-- Define the conditions as parameters

variables (α₁ α₂ : Type) [ProjectivePlane α₁] [ProjectivePlane α₂]
variable (O : Point)

-- Define what it means for a line to be exceptional
def exceptional (l : Line α₁) : Prop :=
  ∃ P : Point, P ∈ l ∧ P = O

-- Define the central projection from α₁ to α₂.
def central_projection (P : Point α₁) : Point α₂ :=
  if P ≠ O then some_projection_function P O else some_infinity_point_on_α₂

-- Define the theorem statement
theorem central_projection_preserves_lines (l : Line α₁) (h : ¬ exceptional l) :
  ∃ m : Line α₂, ∀ P : Point, P ∈ l → (central_projection O P) ∈ m :=
begin
  sorry
end

end central_projection_preserves_lines_l64_64009


namespace rug_overlap_area_l64_64198

theorem rug_overlap_area (A S S2 S3 : ℝ) 
  (hA : A = 200)
  (hS : S = 138)
  (hS2 : S2 = 24)
  (h1 : ∃ (S1 : ℝ), S1 + S2 + S3 = S)
  (h2 : ∃ (S1 : ℝ), S1 + 2 * S2 + 3 * S3 = A) : S3 = 19 :=
by
  sorry

end rug_overlap_area_l64_64198


namespace problem_statement_l64_64448

theorem problem_statement 
  (a b c : ℤ)
  (h1 : (5 * a + 2) ^ (1/3) = 3)
  (h2 : (3 * a + b - 1) ^ (1/2) = 4)
  (h3 : c = Int.floor (Real.sqrt 13))
  : a = 5 ∧ b = 2 ∧ c = 3 ∧ Real.sqrt (3 * a - b + c) = 4 := 
by 
  sorry

end problem_statement_l64_64448


namespace minimum_stamps_to_make_47_cents_l64_64866

theorem minimum_stamps_to_make_47_cents (c f : ℕ) (h : 5 * c + 7 * f = 47) : c + f = 7 :=
sorry

end minimum_stamps_to_make_47_cents_l64_64866


namespace simplify_2A_minus_B_value_2A_minus_B_a_eq_neg1_b_eq_2_find_b_independent_of_a_l64_64756

-- Definitions from the conditions
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 + 6 * a * b + 8 * a

-- Part (1): Simplifying 2A - B
theorem simplify_2A_minus_B (a b : ℝ) : 
  2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a := 
by
  sorry

-- Part (2): Finding 2A - B for specific a and b
theorem value_2A_minus_B_a_eq_neg1_b_eq_2 : 
  2 * A (-1) 2 - B (-1) 2 = 52 := 
by 
  sorry

-- Part (3): Finding b for which 2A - B is independent of a
theorem find_b_independent_of_a (a b : ℝ) (h : 2 * A a b - B a b = 6 * b) : 
  b = -1 / 2 := 
by
  sorry

end simplify_2A_minus_B_value_2A_minus_B_a_eq_neg1_b_eq_2_find_b_independent_of_a_l64_64756


namespace expression_value_at_2_l64_64698

theorem expression_value_at_2 : (2^2 + 3 * 2 - 4) = 6 :=
by 
  sorry

end expression_value_at_2_l64_64698


namespace jade_driving_hours_per_day_l64_64972

variable (Jade Krista : ℕ)
variable (days driving_hours total_hours : ℕ)

theorem jade_driving_hours_per_day :
  (days = 3) →
  (Krista = 6) →
  (total_hours = 42) →
  (total_hours = days * Jade + days * Krista) →
  Jade = 8 :=
by
  intros h_days h_krista h_total_hours h_equation
  sorry

end jade_driving_hours_per_day_l64_64972


namespace final_speed_of_ball_l64_64860

/--
 A small rubber ball moves horizontally between two vertical walls. One wall is fixed, and the other wall moves away from it at a constant speed u.
 The ball's collisions are perfectly elastic. The initial speed of the ball is v₀. Prove that after 10 collisions with the moving wall, the ball's speed is 17 cm/s.
-/
theorem final_speed_of_ball
    (u : ℝ) (v₀ : ℝ) (n : ℕ)
    (u_val : u = 100) (v₀_val : v₀ = 2017) (n_val : n = 10) :
    v₀ - 2 * u * n = 17 := 
    by
    rw [u_val, v₀_val, n_val]
    sorry

end final_speed_of_ball_l64_64860


namespace value_of_expression_l64_64139

theorem value_of_expression (x y : ℤ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 :=
by
  -- Substitute the given values into the expression and calculate
  sorry

end value_of_expression_l64_64139


namespace tram_length_proof_l64_64003
-- Import the necessary library

-- Define the conditions
def tram_length : ℕ := 32 -- The length of the tram we want to prove

-- The main theorem to be stated
theorem tram_length_proof (L : ℕ) (v : ℕ) 
  (h1 : v = L / 4)  -- The tram passed by Misha in 4 seconds
  (h2 : v = (L + 64) / 12)  -- The tram passed through a tunnel of 64 meters in 12 seconds
  : L = tram_length :=
by
  sorry

end tram_length_proof_l64_64003


namespace calorie_allowance_correct_l64_64144

-- Definitions based on the problem's conditions
def daily_calorie_allowance : ℕ := 2000
def weekly_calorie_allowance : ℕ := 10500
def days_in_week : ℕ := 7

-- The statement to be proven
theorem calorie_allowance_correct :
  daily_calorie_allowance * days_in_week = weekly_calorie_allowance :=
by
  sorry

end calorie_allowance_correct_l64_64144


namespace inequality_solution_range_l64_64068

theorem inequality_solution_range (a : ℝ) :
  (∃ (x : ℝ), |x + 1| - |x - 2| < a^2 - 4 * a) → (a > 3 ∨ a < 1) :=
by
  sorry

end inequality_solution_range_l64_64068


namespace simplify_fraction_l64_64514

theorem simplify_fraction : 
  (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := 
by
  sorry

end simplify_fraction_l64_64514


namespace maximize_product_l64_64792

theorem maximize_product (x y z : ℝ) (h1 : x ≥ 20) (h2 : y ≥ 40) (h3 : z ≥ 1675) (h4 : x + y + z = 2015) :
  x * y * z ≤ 721480000 / 27 :=
by sorry

end maximize_product_l64_64792


namespace gecko_bug_eating_l64_64964

theorem gecko_bug_eating (G L F T : ℝ) (hL : L = G / 2)
                                      (hF : F = 3 * L)
                                      (hT : T = 1.5 * F)
                                      (hTotal : G + L + F + T = 63) :
  G = 15 :=
by
  sorry

end gecko_bug_eating_l64_64964


namespace investment_period_is_16_years_l64_64733

noncomputable def compound_interest_period (P A r : ℝ) (n : ℕ) : ℝ :=
  log (A / P) / log (1 + r / (n : ℝ))

theorem investment_period_is_16_years :
  let P : ℝ := 14800
  let A : ℝ := 19065.73
  let r : ℝ := 0.135
  let n : ℕ := 1
  compound_interest_period P A r n ≈ 16 :=
by
  sorry

end investment_period_is_16_years_l64_64733


namespace prism_volume_l64_64150

-- Define the right triangular prism conditions

variables (AB BC AC : ℝ)
variable (S : ℝ)
variable (volume : ℝ)

-- Given conditions
axiom AB_eq_2 : AB = 2
axiom BC_eq_2 : BC = 2
axiom AC_eq_2sqrt3 : AC = 2 * Real.sqrt 3
axiom circumscribed_sphere_surface_area : S = 32 * Real.pi

-- Statement to prove
theorem prism_volume : volume = 4 * Real.sqrt 3 :=
sorry

end prism_volume_l64_64150


namespace problem_solution_l64_64646

variable (x y : ℝ)

-- Conditions
axiom h1 : x ≠ 0
axiom h2 : y ≠ 0
axiom h3 : (4 * x - 3 * y) / (x + 4 * y) = 3

-- Goal
theorem problem_solution : (x - 4 * y) / (4 * x + 3 * y) = 11 / 63 :=
by
  sorry

end problem_solution_l64_64646


namespace numbers_from_1_to_20_with_five_eights_l64_64205

noncomputable def can_form_numbers (n : ℕ) (digits : ℕ) : Prop :=
  ∃ f : fin 5 → ℕ, (∀ i, f i = 8) ∧ (∃ op : ℕ → ℕ, op (f 0, f 1, f 2, f 3, f 4) = n)

theorem numbers_from_1_to_20_with_five_eights :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 20 → can_form_numbers n 8 :=
by
  intros n hn
  cases n
  { simp at hn }
  case dp.eight :

  sorry

end numbers_from_1_to_20_with_five_eights_l64_64205


namespace machine_worked_yesterday_l64_64278

noncomputable def shirts_made_per_minute : ℕ := 3
noncomputable def shirts_made_yesterday : ℕ := 9

theorem machine_worked_yesterday : 
  (shirts_made_yesterday / shirts_made_per_minute) = 3 :=
sorry

end machine_worked_yesterday_l64_64278


namespace num_disks_to_sell_l64_64346

-- Define the buying and selling price conditions.
def cost_per_disk := 6 / 5
def sell_per_disk := 7 / 4

-- Define the desired profit
def desired_profit := 120

-- Calculate the profit per disk.
def profit_per_disk := sell_per_disk - cost_per_disk

-- Statement of the problem: Determine number of disks to sell.
theorem num_disks_to_sell
  (h₁ : cost_per_disk = 6 / 5)
  (h₂ : sell_per_disk = 7 / 4)
  (h₃ : desired_profit = 120)
  (h₄ : profit_per_disk = 7 / 4 - 6 / 5) :
  ∃ disks_to_sell : ℕ, disks_to_sell = 219 ∧ 
  disks_to_sell * profit_per_disk ≥ 120 ∧
  (disks_to_sell - 1) * profit_per_disk < 120 :=
sorry

end num_disks_to_sell_l64_64346


namespace geometric_sequence_product_l64_64496

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = r * a n) (h_cond : a 7 * a 12 = 5) :
  a 8 * a 9 * a 10 * a 11 = 25 :=
by 
  sorry

end geometric_sequence_product_l64_64496


namespace sum_of_powers_seven_l64_64781

theorem sum_of_powers_seven (α1 α2 α3 : ℂ)
  (h1 : α1 + α2 + α3 = 2)
  (h2 : α1^2 + α2^2 + α3^2 = 6)
  (h3 : α1^3 + α2^3 + α3^3 = 14) :
  α1^7 + α2^7 + α3^7 = 478 := by
  sorry

end sum_of_powers_seven_l64_64781


namespace father_l64_64554

variable {son_age : ℕ} -- Son's present age
variable {father_age : ℕ} -- Father's present age

-- Conditions
def father_is_four_times_son (son_age father_age : ℕ) : Prop := father_age = 4 * son_age
def sum_of_ages_ten_years_ago (son_age father_age : ℕ) : Prop := (son_age - 10) + (father_age - 10) = 60

-- Theorem statement
theorem father's_present_age 
  (son_age father_age : ℕ)
  (h1 : father_is_four_times_son son_age father_age) 
  (h2 : sum_of_ages_ten_years_ago son_age father_age) : 
  father_age = 64 :=
sorry

end father_l64_64554
