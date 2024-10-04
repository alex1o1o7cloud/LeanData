import Mathlib

namespace closest_ratio_to_one_l146_146646

theorem closest_ratio_to_one (a c : ℕ) (h1 : 2 * a + c = 130) (h2 : a ≥ 1) (h3 : c ≥ 1) : 
  a = 43 ∧ c = 44 :=
by {
    sorry 
}

end closest_ratio_to_one_l146_146646


namespace value_of_m_l146_146717

theorem value_of_m (m : ℤ) : (|m| = 1) ∧ (m + 1 ≠ 0) → m = 1 := by
  sorry

end value_of_m_l146_146717


namespace max_value_x1_x2_l146_146449

noncomputable def f (x : ℝ) := 1 - Real.sqrt (2 - 3 * x)
noncomputable def g (x : ℝ) := 2 * Real.log x

theorem max_value_x1_x2 (x1 x2 : ℝ) (h1 : x1 ≤ 2 / 3) (h2 : x2 > 0) (h3 : x1 - x2 = (1 - Real.sqrt (2 - 3 * x1)) - (2 * Real.log x2)) :
  x1 - x2 ≤ -25 / 48 :=
sorry

end max_value_x1_x2_l146_146449


namespace maria_purse_value_l146_146026

def value_of_nickels (num_nickels : ℕ) : ℕ := num_nickels * 5
def value_of_dimes (num_dimes : ℕ) : ℕ := num_dimes * 10
def value_of_quarters (num_quarters : ℕ) : ℕ := num_quarters * 25
def total_value (num_nickels num_dimes num_quarters : ℕ) : ℕ := 
  value_of_nickels num_nickels + value_of_dimes num_dimes + value_of_quarters num_quarters
def percentage_of_dollar (value_cents : ℕ) : ℕ := value_cents * 100 / 100

theorem maria_purse_value : percentage_of_dollar (total_value 2 3 2) = 90 := by
  sorry

end maria_purse_value_l146_146026


namespace find_h_l146_146604

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l146_146604


namespace gcd_n4_plus_16_n_plus_3_eq_1_l146_146889

theorem gcd_n4_plus_16_n_plus_3_eq_1 (n : ℕ) (h : n > 16) : gcd (n^4 + 16) (n + 3) = 1 := 
sorry

end gcd_n4_plus_16_n_plus_3_eq_1_l146_146889


namespace solve_quadratic_inequality_l146_146983

theorem solve_quadratic_inequality (x : ℝ) : (-x^2 - 2 * x + 3 < 0) ↔ (x < -3 ∨ x > 1) := 
sorry

end solve_quadratic_inequality_l146_146983


namespace quadratic_no_real_roots_implies_inequality_l146_146676

theorem quadratic_no_real_roots_implies_inequality (a b c : ℝ) :
  let A := b + c
  let B := a + c
  let C := a + b
  (B^2 - 4 * A * C < 0) → 4 * a * c - b^2 ≤ 3 * a * (a + b + c) :=
by
  intro h
  sorry

end quadratic_no_real_roots_implies_inequality_l146_146676


namespace acid_percentage_in_original_mixture_l146_146227

theorem acid_percentage_in_original_mixture 
  {a w : ℕ} 
  (h1 : a / (a + w + 1) = 1 / 5) 
  (h2 : (a + 1) / (a + w + 2) = 1 / 3) : 
  a / (a + w) = 1 / 4 :=
sorry

end acid_percentage_in_original_mixture_l146_146227


namespace length_of_segment_l146_146390

theorem length_of_segment : ∃ (a b : ℝ), (|a - (16 : ℝ)^(1/5)| = 3) ∧ (|b - (16 : ℝ)^(1/5)| = 3) ∧ abs (a - b) = 6 :=
by
  sorry

end length_of_segment_l146_146390


namespace dot_product_neg_vec_n_l146_146144

-- Vector definitions
def vec_m : ℝ × ℝ := (2, -1)
def vec_n : ℝ × ℝ := (3, 2)
def neg_vec_n : ℝ × ℝ := (-vec_n.1, -vec_n.2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Proof statement
theorem dot_product_neg_vec_n :
  dot_product vec_m neg_vec_n = -4 :=
by
  -- Sorry to skip the proof
  sorry

end dot_product_neg_vec_n_l146_146144


namespace sin_C_value_l146_146610

theorem sin_C_value (A B C : Real) (AC BC : Real) (h_AC : AC = 3) (h_BC : BC = 2 * Real.sqrt 3) (h_A : A = 2 * B) :
    let C : Real := Real.pi - A - B
    Real.sin C = Real.sqrt 6 / 9 :=
  sorry

end sin_C_value_l146_146610


namespace ratio_length_breadth_l146_146049

theorem ratio_length_breadth
  (b : ℝ) (A : ℝ) (h_b : b = 11) (h_A : A = 363) :
  (∃ l : ℝ, A = l * b ∧ l / b = 3) :=
by
  sorry

end ratio_length_breadth_l146_146049


namespace linear_term_zero_implies_sum_zero_l146_146158

-- Define the condition that the product does not have a linear term
def no_linear_term (x a b : ℝ) : Prop :=
  (x + a) * (x + b) = x^2 + (a + b) * x + a * b

-- Given the condition, we need to prove that a + b = 0
theorem linear_term_zero_implies_sum_zero {a b : ℝ} (h : ∀ x : ℝ, no_linear_term x a b) : a + b = 0 :=
by 
  sorry

end linear_term_zero_implies_sum_zero_l146_146158


namespace base5_to_base10_max_l146_146083

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l146_146083


namespace complex_expression_value_l146_146542

theorem complex_expression_value {i : ℂ} (h : i^2 = -1) : i^3 * (1 + i)^2 = 2 := 
by
  sorry

end complex_expression_value_l146_146542


namespace measure_of_angle4_l146_146824

def angle1 := 62
def angle2 := 36
def angle3 := 24
def angle4 : ℕ := 122

theorem measure_of_angle4 (d e : ℕ) (h1 : angle1 + angle2 + angle3 + d + e = 180) (h2 : d + e = 58) :
  angle4 = 180 - (angle1 + angle2 + angle3 + d + e) :=
by
  sorry

end measure_of_angle4_l146_146824


namespace candidateA_prob_second_third_correct_after_first_incorrect_candidateA_more_likely_to_be_hired_based_on_variance_l146_146874

-- Definitions/Conditions from the problem
def q_total := 8
def q_select := 3
def correct_A := 6
def correct_prob_B := 3/4

noncomputable def prob_A_first_incorrect : ℚ := (q_total - correct_A) / q_total
noncomputable def prob_A_second_correct_given_first_incorrect : ℚ := correct_A / (q_total - 1)
noncomputable def prob_A_third_correct_given_second_correct : ℚ := (correct_A - 1) / (q_total - 2)
noncomputable def prob_A_B_given_A : ℚ := prob_A_first_incorrect * prob_A_second_correct_given_first_incorrect * prob_A_third_correct_given_second_correct
noncomputable def prob_B_given_A_conditional : ℚ := prob_A_B_given_A / prob_A_first_incorrect

def expectation_X_CandidateA : ℚ := 27/12  -- E(X) for Candidate A
def variance_X_CandidateA : ℚ := 45/112  -- Var(X) for Candidate A
def expectation_Y_CandidateB : ℚ := 27/12  -- E(Y) for Candidate B
def variance_Y_CandidateB : ℚ := 9/16   -- Var(Y) for Candidate B

-- Lean Statement for the first question
theorem candidateA_prob_second_third_correct_after_first_incorrect :
  prob_B_given_A_conditional = 5 / 7 := sorry

-- Lean Statement for the second question based on expectation and variance
theorem candidateA_more_likely_to_be_hired_based_on_variance :
  variance_X_CandidateA < variance_Y_CandidateB := sorry

end candidateA_prob_second_third_correct_after_first_incorrect_candidateA_more_likely_to_be_hired_based_on_variance_l146_146874


namespace sufficient_not_necessary_l146_146677

-- Define set A and set B
def setA (x : ℝ) := x > 5
def setB (x : ℝ) := x > 3

-- Statement:
theorem sufficient_not_necessary (x : ℝ) : setA x → setB x :=
by
  intro h
  exact sorry

end sufficient_not_necessary_l146_146677


namespace find_a_l146_146713

noncomputable def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

theorem find_a (a : ℕ) (h : collinear (a, 0) (0, a + 4) (1, 3)) : a = 4 :=
by
  sorry

end find_a_l146_146713


namespace ratio_of_sam_to_sue_l146_146637

-- Definitions
def Sam_age (S : ℕ) : Prop := 3 * S = 18
def Kendra_age (K : ℕ) : Prop := K = 18
def total_age_in_3_years (S U K : ℕ) : Prop := (S + 3) + (U + 3) + (K + 3) = 36

-- Theorem statement
theorem ratio_of_sam_to_sue (S U K : ℕ) (h1 : Sam_age S) (h2 : Kendra_age K) (h3 : total_age_in_3_years S U K) :
  S / U = 2 :=
sorry

end ratio_of_sam_to_sue_l146_146637


namespace right_triangle_unique_perimeter_18_l146_146724

theorem right_triangle_unique_perimeter_18 :
  ∃! (a b c : ℤ), a^2 + b^2 = c^2 ∧ a + b + c = 18 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end right_triangle_unique_perimeter_18_l146_146724


namespace monotonic_decreasing_range_l146_146371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x

theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) → a ≤ -1 :=
  sorry

end monotonic_decreasing_range_l146_146371


namespace geometric_series_ratio_l146_146788

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l146_146788


namespace print_pages_l146_146647

theorem print_pages (pages_per_cost : ℕ) (cost_cents : ℕ) (dollars : ℕ)
                    (h1 : pages_per_cost = 7) (h2 : cost_cents = 9) (h3 : dollars = 50) :
  (dollars * 100 * pages_per_cost) / cost_cents = 3888 :=
by
  sorry

end print_pages_l146_146647


namespace factor_x4_plus_81_l146_146548

theorem factor_x4_plus_81 (x : ℝ) : (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9) = x^4 + 81 := 
by 
   sorry

end factor_x4_plus_81_l146_146548


namespace light_bulb_arrangement_l146_146380

theorem light_bulb_arrangement :
  let B := 6
  let R := 7
  let W := 9
  let total_arrangements := Nat.choose (B + R) B * Nat.choose (B + R + 1) W
  total_arrangements = 3435432 :=
by
  sorry

end light_bulb_arrangement_l146_146380


namespace largest_base_5_five_digits_base_10_value_l146_146087

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l146_146087


namespace largest_d_for_g_of_minus5_l146_146885

theorem largest_d_for_g_of_minus5 (d : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + d = -5) → d ≤ -4 :=
by
-- Proof steps will be inserted here
sorry

end largest_d_for_g_of_minus5_l146_146885


namespace cos_5alpha_eq_sin_5alpha_eq_l146_146698

noncomputable def cos_five_alpha (α : ℝ) : ℝ := 16 * (Real.cos α) ^ 5 - 20 * (Real.cos α) ^ 3 + 5 * (Real.cos α)
noncomputable def sin_five_alpha (α : ℝ) : ℝ := 16 * (Real.sin α) ^ 5 - 20 * (Real.sin α) ^ 3 + 5 * (Real.sin α)

theorem cos_5alpha_eq (α : ℝ) : Real.cos (5 * α) = cos_five_alpha α :=
by sorry

theorem sin_5alpha_eq (α : ℝ) : Real.sin (5 * α) = sin_five_alpha α :=
by sorry

end cos_5alpha_eq_sin_5alpha_eq_l146_146698


namespace rectangle_area_y_value_l146_146774

theorem rectangle_area_y_value :
  ∀ (y : ℝ), 
  let P := (1, 1)
  let Q := (1, 4)
  let R := (y, 4)
  let S := (y, 1)
  (y > 1) → 
  (abs (R.1 - P.1) * abs (Q.2 - P.2) = 36) → 
  y = 13 :=
by
  intros y P Q R S hy harea
  let P := (1, 1)
  let Q := (1, 4)
  let R := (y, 4)
  let S := (y, 1)
  sorry

end rectangle_area_y_value_l146_146774


namespace units_digit_specified_expression_l146_146514

theorem units_digit_specified_expression :
  let numerator := (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11)
  let denominator := 8000
  let product := numerator * 20
  (∃ d, product / denominator = d ∧ (d % 10 = 6)) :=
by
  sorry

end units_digit_specified_expression_l146_146514


namespace area_of_garden_l146_146532

variable (w l : ℕ)
variable (h1 : l = 3 * w) 
variable (h2 : 2 * (l + w) = 72)

theorem area_of_garden : l * w = 243 := by
  sorry

end area_of_garden_l146_146532


namespace harrys_age_l146_146018

theorem harrys_age {K B J F H : ℕ} 
  (hKiarra : K = 30)
  (hKiarra_Bea : K = 2 * B)
  (hJob : J = 3 * B)
  (hFigaro : F = J + 7)
  (hHarry : H = F / 2) : 
  H = 26 := 
by 
  -- Definitions from the conditions
  have hBea : B = 15, from (by linarith : 15 = K / 2).symm,
  
  -- Continuing the proof using the provided conditions and calculating step by step
  sorry

end harrys_age_l146_146018


namespace find_y_l146_146726

theorem find_y (y : ℝ) (h : 9 * y^3 = y * 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end find_y_l146_146726


namespace johns_father_fraction_l146_146335

theorem johns_father_fraction (total_money : ℝ) (given_to_mother_fraction remaining_after_father : ℝ) :
  total_money = 200 →
  given_to_mother_fraction = 3 / 8 →
  remaining_after_father = 65 →
  ((total_money - given_to_mother_fraction * total_money) - remaining_after_father) / total_money
  = 3 / 10 :=
by
  intros h1 h2 h3
  sorry

end johns_father_fraction_l146_146335


namespace problem_solution_l146_146151

-- We assume x and y are real numbers.
variables (x y : ℝ)

-- Our conditions
def condition1 : Prop := |x| - x + y = 6
def condition2 : Prop := x + |y| + y = 8

-- The goal is to prove that x + y = 30 under the given conditions.
theorem problem_solution (hx : condition1 x y) (hy : condition2 x y) : x + y = 30 :=
sorry

end problem_solution_l146_146151


namespace benny_lunch_cost_l146_146540

theorem benny_lunch_cost :
  let person := 3;
  let cost_per_lunch := 8;
  let total_cost := person * cost_per_lunch;
  total_cost = 24 :=
by
  let person := 3;
  let cost_per_lunch := 8;
  let total_cost := person * cost_per_lunch;
  have h : total_cost = 24 := by
    sorry
  exact h

end benny_lunch_cost_l146_146540


namespace number_of_speedster_convertibles_l146_146853

def proof_problem (T : ℕ) :=
  let Speedsters := 2 * T / 3
  let NonSpeedsters := 50
  let TotalInventory := NonSpeedsters * 3
  let SpeedsterConvertibles := 4 * Speedsters / 5
  (Speedsters = 2 * TotalInventory / 3) ∧ (SpeedsterConvertibles = 4 * Speedsters / 5)

theorem number_of_speedster_convertibles : proof_problem 150 → ∃ (x : ℕ), x = 80 :=
by
  -- Provide the definition of Speedsters, NonSpeedsters, TotalInventory, and SpeedsterConvertibles
  sorry

end number_of_speedster_convertibles_l146_146853


namespace subtracted_value_l146_146860

theorem subtracted_value (N V : ℕ) (h1 : N = 800) (h2 : N / 5 - V = 6) : V = 154 :=
by
  sorry

end subtracted_value_l146_146860


namespace color_preference_blue_percentage_l146_146044

theorem color_preference_blue_percentage : 
(40 + 70 + 30 + 50 + 30 + 40 = 260) → 
(70 / 260 * 100 = 27) := 
by 
  sorry

end color_preference_blue_percentage_l146_146044


namespace little_john_height_l146_146347

theorem little_john_height :
  let m := 2 
  let cm_to_m := 8 * 0.01
  let mm_to_m := 3 * 0.001
  m + cm_to_m + mm_to_m = 2.083 := 
by
  sorry

end little_john_height_l146_146347


namespace total_snakes_l146_146948

/-
  Problem: Mary sees three breeding balls with 8 snakes each and 6 additional pairs of snakes.
           How many snakes did she see total?
  Conditions:
    - There are 3 breeding balls.
    - Each breeding ball has 8 snakes.
    - There are 6 additional pairs of snakes.
  Answer: 36 snakes
-/

theorem total_snakes (balls : ℕ) (snakes_per_ball : ℕ) (pairs : ℕ) (snakes_per_pair : ℕ) :
    balls = 3 → snakes_per_ball = 8 → pairs = 6 → snakes_per_pair = 2 →
    (balls * snakes_per_ball) + (pairs * snakes_per_pair) = 36 :=
  by 
    intros hb hspb hp hsp
    sorry

end total_snakes_l146_146948


namespace marker_cost_l146_146294

theorem marker_cost (s n c : ℕ) (h_majority : s > 20) (h_markers : n > 1) (h_cost : c > n) (h_total_cost : s * n * c = 3388) : c = 11 :=
by {
  sorry
}

end marker_cost_l146_146294


namespace simple_interest_rate_l146_146402

theorem simple_interest_rate 
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (hSI : SI = 250) (hP : P = 1500) (hT : T = 5)
  (hSIFormula : SI = (P * R * T) / 100) :
  R = 3.33 := 
by 
  sorry

end simple_interest_rate_l146_146402


namespace incorrect_proposition_C_l146_146544

open Nat

theorem incorrect_proposition_C (a : ℕ → ℕ) (h : ∀ n, n ≥ 2 → a (n+1) * a (n-1) = a n * a n) : ¬(∀ n, (n ≥ 2 → a (n+1) * a (n-1) = a n * a n) → ∃ q, ∀ n, a (n+1) = q * a n) :=
by
  intro hyp
  have ex_counter_example: ∃ a, a 0 = 0 ∧ a 1 = 0 ∧ (∀ n, a (n+1) * a (n-1) = a n * a n) := sorry
  cases ex_counter_example with a ha
  specialize hyp a
  have H := hyp ha.2
  intro contradiction
  contradiction ha
  sorry

end incorrect_proposition_C_l146_146544


namespace train_passes_jogger_time_l146_146688

noncomputable def jogger_speed_km_per_hr : ℝ := 9
noncomputable def train_speed_km_per_hr : ℝ := 75
noncomputable def jogger_head_start_m : ℝ := 500
noncomputable def train_length_m : ℝ := 300

noncomputable def km_per_hr_to_m_per_s (speed: ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def jogger_speed_m_per_s := km_per_hr_to_m_per_s jogger_speed_km_per_hr
noncomputable def train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr

noncomputable def relative_speed_m_per_s := train_speed_m_per_s - jogger_speed_m_per_s

noncomputable def total_distance_to_cover_m := jogger_head_start_m + train_length_m

theorem train_passes_jogger_time :
  let time_to_pass := total_distance_to_cover_m / relative_speed_m_per_s
  abs (time_to_pass - 43.64) < 0.01 :=
by
  sorry

end train_passes_jogger_time_l146_146688


namespace hypotenuse_length_l146_146321

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l146_146321


namespace remainder_modulo_9_l146_146503

open Int

theorem remainder_modulo_9 (a b c d : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9) (hd : d < 9)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (hinv : ∀ x ∈ {a, b, c, d}, Nat.gcd x 9 = 1) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (a * b * c * d)⁻¹) % 9 = 6 :=
sorry

end remainder_modulo_9_l146_146503


namespace girls_try_out_l146_146664

-- Given conditions
variables (boys callBacks didNotMakeCut : ℕ)
variable (G : ℕ)

-- Define the conditions
def conditions : Prop := 
  boys = 14 ∧ 
  callBacks = 2 ∧ 
  didNotMakeCut = 21 ∧ 
  G + boys = callBacks + didNotMakeCut

-- The statement of the proof
theorem girls_try_out (h : conditions boys callBacks didNotMakeCut G) : G = 9 :=
by
  sorry

end girls_try_out_l146_146664


namespace total_spent_l146_146630

variable (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ)

/-- Conditions from the problem setup --/
def conditions :=
  T_L = 40 ∧
  J_L = 0.5 * T_L ∧
  C_L = 2 * T_L ∧
  S_L = 3 * J_L ∧
  T_C = 0.25 * T_L ∧
  J_C = 3 * J_L ∧
  C_C = 0.5 * C_L ∧
  S_C = S_L ∧
  D_C = 2 * S_C ∧
  A_C = 0.5 * J_C

/-- Total spent by Lisa --/
def total_Lisa := T_L + J_L + C_L + S_L

/-- Total spent by Carly --/
def total_Carly := T_C + J_C + C_C + S_C + D_C + A_C

/-- Combined total spent by Lisa and Carly --/
theorem total_spent :
  conditions T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C →
  total_Lisa T_L J_L C_L S_L + total_Carly T_C J_C C_C S_C D_C A_C = 520 :=
by
  sorry

end total_spent_l146_146630


namespace hypotenuse_length_l146_146320

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l146_146320


namespace findLastNames_l146_146121

noncomputable def peachProblem : Prop :=
  ∃ (a b c d : ℕ),
    2 * a + 3 * b + 4 * c + 5 * d = 32 ∧
    a + b + c + d = 10 ∧
    (a = 3 ∧ b = 4 ∧ c = 1 ∧ d = 2)

theorem findLastNames :
  peachProblem :=
sorry

end findLastNames_l146_146121


namespace john_annual_profit_l146_146931

-- Definitions of monthly incomes
def TenantA_income : ℕ := 350
def TenantB_income : ℕ := 400
def TenantC_income : ℕ := 450

-- Total monthly income
def total_monthly_income : ℕ := TenantA_income + TenantB_income + TenantC_income

-- Definitions of monthly expenses
def rent_expense : ℕ := 900
def utilities_expense : ℕ := 100
def maintenance_fee : ℕ := 50

-- Total monthly expenses
def total_monthly_expense : ℕ := rent_expense + utilities_expense + maintenance_fee

-- Monthly profit
def monthly_profit : ℕ := total_monthly_income - total_monthly_expense

-- Annual profit
def annual_profit : ℕ := monthly_profit * 12

theorem john_annual_profit :
  annual_profit = 1800 := by
  -- The proof is omitted, but the statement asserts that John makes an annual profit of $1800.
  sorry

end john_annual_profit_l146_146931


namespace total_flour_needed_l146_146172

theorem total_flour_needed (Katie_flour : ℕ) (Sheila_flour : ℕ) 
  (h1 : Katie_flour = 3) 
  (h2 : Sheila_flour = Katie_flour + 2) : 
  Katie_flour + Sheila_flour = 8 := 
  by 
  sorry

end total_flour_needed_l146_146172


namespace largest_base_5_five_digit_number_in_decimal_l146_146072

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l146_146072


namespace problem_solution_l146_146117

theorem problem_solution :
  20 * ((180 / 3) + (40 / 5) + (16 / 32) + 2) = 1410 := by
  sorry

end problem_solution_l146_146117


namespace isosceles_triangle_largest_angle_l146_146739

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h1 : A = B) (h2 : C = 50) (h3 : A + B + C = 180) : max A C = 80 :=
by 
  -- Define additional facts about the triangle, e.g., A = B = 50, and sum of angles = 180.
  have h4 : A = 50 := sorry
  rw [h4, h2] at h3
  -- Prove the final result using the given conditions.
  sorry

end isosceles_triangle_largest_angle_l146_146739


namespace lassis_with_eighteen_mangoes_smoothies_with_eighteen_mangoes_and_thirtysix_bananas_l146_146543

def lassis_per_three_mangoes := 15
def smoothies_per_mango := 1
def bananas_per_smoothie := 2

-- proving the number of lassis Caroline can make with eighteen mangoes
theorem lassis_with_eighteen_mangoes :
  (18 / 3) * lassis_per_three_mangoes = 90 :=
by 
  sorry

-- proving the number of smoothies Caroline can make with eighteen mangoes and thirty-six bananas
theorem smoothies_with_eighteen_mangoes_and_thirtysix_bananas :
  min (18 / smoothies_per_mango) (36 / bananas_per_smoothie) = 18 :=
by 
  sorry

end lassis_with_eighteen_mangoes_smoothies_with_eighteen_mangoes_and_thirtysix_bananas_l146_146543


namespace tax_raise_expectation_l146_146404

noncomputable section

variables 
  (x y : ℝ) -- x: fraction of liars, y: fraction of economists
  (p1 p2 p3 p4 : ℝ) -- percentages of affirmative answers
  (expected_taxes : ℝ) -- expected fraction for taxes

-- Given conditions
def given_conditions (x y p1 p2 p3 p4 : ℝ) :=
  p1 = 0.4 ∧ p2 = 0.3 ∧ p3 = 0.5 ∧ p4 = 0.0 ∧
  y = 1 - x ∧ -- fraction of economists
  3 * x + y = 1.2 -- sum of affirmative answers

-- The statement to prove
theorem tax_raise_expectation (x y p1 p2 p3 p4 : ℝ) : 
  given_conditions x y p1 p2 p3 p4 →
  expected_taxes = 0.4 - x →
  expected_taxes = 0.3 :=
begin
  intro h, intro h_exp,
  sorry -- proof to be filled in
end

end tax_raise_expectation_l146_146404


namespace girl_needs_120_oranges_l146_146686

-- Define the cost and selling prices per pack
def cost_per_pack : ℤ := 15   -- cents
def oranges_per_pack_cost : ℤ := 4
def sell_per_pack : ℤ := 30   -- cents
def oranges_per_pack_sell : ℤ := 6

-- Define the target profit
def target_profit : ℤ := 150  -- cents

-- Calculate the cost price per orange
def cost_per_orange : ℚ := cost_per_pack / oranges_per_pack_cost

-- Calculate the selling price per orange
def sell_per_orange : ℚ := sell_per_pack / oranges_per_pack_sell

-- Calculate the profit per orange
def profit_per_orange : ℚ := sell_per_orange - cost_per_orange

-- Calculate the number of oranges needed to achieve the target profit
def oranges_needed : ℚ := target_profit / profit_per_orange

-- Lean theorem statement
theorem girl_needs_120_oranges :
  oranges_needed = 120 :=
  sorry

end girl_needs_120_oranges_l146_146686


namespace find_third_angle_of_triangle_l146_146010

theorem find_third_angle_of_triangle (a b c : ℝ) (h₁ : a = 40) (h₂ : b = 3 * c) (h₃ : a + b + c = 180) : c = 35 := 
by sorry

end find_third_angle_of_triangle_l146_146010


namespace sqrt_81_eq_9_l146_146775

theorem sqrt_81_eq_9 : Real.sqrt 81 = 9 :=
by
  sorry

end sqrt_81_eq_9_l146_146775


namespace common_ratio_of_geometric_series_l146_146793

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l146_146793


namespace exists_infinite_diff_but_not_sum_of_kth_powers_l146_146482

theorem exists_infinite_diff_but_not_sum_of_kth_powers (k : ℕ) (hk : k > 1) :
  ∃ (infinitely_many x : ℕ), (∃ (a b : ℕ), x = a^k - b^k) ∧ ¬ (∃ (c d : ℕ), x = c^k + d^k) :=
  sorry

end exists_infinite_diff_but_not_sum_of_kth_powers_l146_146482


namespace semiperimeter_inequality_l146_146034

theorem semiperimeter_inequality (p R r : ℝ) (hp : p ≥ 0) (hR : R ≥ 0) (hr : r ≥ 0) :
  p ≥ (3 / 2) * Real.sqrt (6 * R * r) :=
sorry

end semiperimeter_inequality_l146_146034


namespace find_f_l146_146756

theorem find_f (f : ℝ → ℝ) (h₁ : ∀ x y : ℝ, 0 < x → 0 < y → x ≤ y → f x ≤ f y)
  (h₂ : ∀ x : ℝ, 0 < x → f (x ^ 4) + f (x ^ 2) + f x + f 1 = x ^ 4 + x ^ 2 + x + 1) :
  ∀ x : ℝ, 0 < x → f x = x := 
sorry

end find_f_l146_146756


namespace first_number_in_a10_l146_146955

-- Define a function that captures the sequence of the first number in each sum 'a_n'.
def first_in_an (n : ℕ) : ℕ :=
  1 + 2 * (n * (n - 1)) / 2 

-- State the theorem we want to prove
theorem first_number_in_a10 : first_in_an 10 = 91 := 
  sorry

end first_number_in_a10_l146_146955


namespace triangle_inequality_l146_146036

variable {a b c : ℝ}

theorem triangle_inequality (h₁ : a + b + c = 2) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) (h₅ : a + b > c) (h₆ : a + c > b) (h₇ : b + c > a) : 
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
by
  sorry

end triangle_inequality_l146_146036


namespace alien_abduction_problem_l146_146539

theorem alien_abduction_problem:
  ∀ (total_abducted people_taken_elsewhere people_taken_home people_returned: ℕ),
  total_abducted = 200 →
  people_taken_elsewhere = 10 →
  people_taken_home = 30 →
  people_returned = total_abducted - (people_taken_elsewhere + people_taken_home) →
  (people_returned : ℕ) / total_abducted * 100 = 80 := 
by
  intros total_abducted people_taken_elsewhere people_taken_home people_returned;
  intros h_total_abducted h_taken_elsewhere h_taken_home h_people_returned;
  sorry

end alien_abduction_problem_l146_146539


namespace percentageReduction_l146_146854

variable (R P : ℝ)

def originalPrice (R : ℝ) (P : ℝ) : Prop :=
  2400 / R - 2400 / P = 8 ∧ R = 120

theorem percentageReduction : 
  originalPrice 120 P → ((P - 120) / P) * 100 = 40 := 
by
  sorry

end percentageReduction_l146_146854


namespace find_f_neg_one_l146_146571

theorem find_f_neg_one (f h : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x)
    (h2 : ∀ x, h x = f x - 9) (h3 : h 1 = 2) : f (-1) = -11 := 
by
  sorry

end find_f_neg_one_l146_146571


namespace black_area_remaining_after_changes_l146_146248

theorem black_area_remaining_after_changes :
  let initial_fraction_black := 1
  let change_factor := 8 / 9
  let num_changes := 4
  let final_fraction_black := (change_factor ^ num_changes)
  final_fraction_black = 4096 / 6561 :=
by
  sorry

end black_area_remaining_after_changes_l146_146248


namespace negation_of_P_is_exists_ge_1_l146_146896

theorem negation_of_P_is_exists_ge_1 :
  let P := ∀ x : ℤ, x < 1
  ¬P ↔ ∃ x : ℤ, x ≥ 1 := by
  sorry

end negation_of_P_is_exists_ge_1_l146_146896


namespace total_snakes_count_l146_146947

-- Define the basic conditions
def breedingBalls : Nat := 3
def snakesPerBall : Nat := 8
def pairsOfSnakes : Nat := 6
def snakesPerPair : Nat := 2

-- Define the total number of snakes
theorem total_snakes_count : 
  (breedingBalls * snakesPerBall) + (pairsOfSnakes * snakesPerPair) = 36 := 
by 
  -- we skip the proof with sorry
  sorry

end total_snakes_count_l146_146947


namespace eval_composition_l146_146478

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 - 2

theorem eval_composition : f (g 2) = -7 := 
by {
  sorry
}

end eval_composition_l146_146478


namespace slope_of_line_through_midpoints_l146_146852

theorem slope_of_line_through_midpoints :
  let P₁ := (1, 2)
  let P₂ := (3, 8)
  let P₃ := (4, 3)
  let P₄ := (7, 9)
  let M₁ := ( (P₁.1 + P₂.1)/2, (P₁.2 + P₂.2)/2 )
  let M₂ := ( (P₃.1 + P₄.1)/2, (P₃.2 + P₄.2)/2 )
  let slope := (M₂.2 - M₁.2) / (M₂.1 - M₁.1)
  slope = 2/7 :=
by
  sorry

end slope_of_line_through_midpoints_l146_146852


namespace hypotenuse_length_l146_146325

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l146_146325


namespace second_exponent_base_ends_in_1_l146_146129

theorem second_exponent_base_ends_in_1 
  (x : ℕ) 
  (h : ((1023 ^ 3923) + (x ^ 3921)) % 10 = 8) : 
  x % 10 = 1 := 
by sorry

end second_exponent_base_ends_in_1_l146_146129


namespace base5_to_base10_max_l146_146081

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l146_146081


namespace number_of_women_more_than_men_l146_146214

variables (M W : ℕ)

def ratio_condition : Prop := M * 3 = 2 * W
def total_condition : Prop := M + W = 20
def correct_answer : Prop := W - M = 4

theorem number_of_women_more_than_men 
  (h1 : ratio_condition M W) 
  (h2 : total_condition M W) : 
  correct_answer M W := 
by 
  sorry

end number_of_women_more_than_men_l146_146214


namespace arithmetic_sequence_a6_eq_1_l146_146920

theorem arithmetic_sequence_a6_eq_1
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : S 11 = 11)
  (h2 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h3 : ∃ d, ∀ n, a n = a 1 + (n - 1) * d) :
  a 6 = 1 :=
by
  sorry

end arithmetic_sequence_a6_eq_1_l146_146920


namespace angle_is_60_degrees_l146_146425

-- Definitions
def angle_is_twice_complementary (x : ℝ) : Prop := x = 2 * (90 - x)

-- Theorem statement
theorem angle_is_60_degrees (x : ℝ) (h : angle_is_twice_complementary x) : x = 60 :=
by sorry

end angle_is_60_degrees_l146_146425


namespace smallest_sum_of_squares_l146_146367

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 175) : 
  ∃ (x y : ℤ), x^2 - y^2 = 175 ∧ x^2 + y^2 = 625 :=
sorry

end smallest_sum_of_squares_l146_146367


namespace average_monthly_growth_rate_l146_146374

-- Define the conditions
variables (P : ℝ) (r : ℝ)
-- The condition that output in December is P times that of January
axiom growth_rate_condition : (1 + r)^11 = P

-- Define the goal to prove the average monthly growth rate
theorem average_monthly_growth_rate : r = (P^(1/11) - 1) :=
by
  sorry

end average_monthly_growth_rate_l146_146374


namespace common_ratio_of_geometric_series_l146_146806

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l146_146806


namespace beatrice_tv_ratio_l146_146253

theorem beatrice_tv_ratio (T1 T2 T Ttotal : ℕ)
  (h1 : T1 = 8)
  (h2 : T2 = 10)
  (h_total : Ttotal = 42)
  (h_T : T = Ttotal - T1 - T2) :
  (T / gcd T T1, T1 / gcd T T1) = (3, 1) :=
by {
  sorry
}

end beatrice_tv_ratio_l146_146253


namespace find_a22_l146_146972

variable (a : ℕ → ℝ)
variable (h : ∀ n, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
variable (h99 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
variable (h100 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
variable (h10 : a 10 = 10)

theorem find_a22 : a 22 = 10 := sorry

end find_a22_l146_146972


namespace vendor_second_day_sale_l146_146691

theorem vendor_second_day_sale (n : ℕ) :
  let sold_first_day := (50 * n) / 100
  let remaining_after_first_sale := n - sold_first_day
  let thrown_away_first_day := (20 * remaining_after_first_sale) / 100
  let remaining_after_first_day := remaining_after_first_sale - thrown_away_first_day
  let total_thrown_away := (30 * n) / 100
  let thrown_away_second_day := total_thrown_away - thrown_away_first_day
  let sold_second_day := remaining_after_first_day - thrown_away_second_day
  let percent_sold_second_day := (sold_second_day * 100) / remaining_after_first_day
  percent_sold_second_day = 50 :=
sorry

end vendor_second_day_sale_l146_146691


namespace sin_alpha_of_terminal_side_l146_146005

theorem sin_alpha_of_terminal_side (α : ℝ) (P : ℝ × ℝ) 
  (hP : P = (5, 12)) :
  Real.sin α = 12 / 13 := sorry

end sin_alpha_of_terminal_side_l146_146005


namespace concurrency_or_parallelism_of_lines_l146_146619

variables {α : Type*} [ordered_ring α]
variables {A B C A1 B1 C1 P A2 B2 C2 : point α}
variables (medians : ∀ (P : point α), intersection_condition α A B C A1 B1 C1 P A2 B2 C2)

theorem concurrency_or_parallelism_of_lines
    (T : triangle α)
    (medians_def : medians T A B C = [A1, B1, C1])
    (intersection_def : 
        (intersection A P B1 C1 = A2) ∧ 
        (intersection B P C1 A1 = B2) ∧ 
        (intersection C P A1 B1 = C2)) :
  concurrency_or_parallelism α A1 A2 B1 B2 C1 C2 :=
sorry

end concurrency_or_parallelism_of_lines_l146_146619


namespace base5_to_base10_max_l146_146080

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l146_146080


namespace find_a_for_symmetry_l146_146907

theorem find_a_for_symmetry :
  ∃ a : ℝ, (∀ x : ℝ, a * Real.sin x + Real.cos (x + π / 6) = 
                    a * Real.sin (π / 3 - x) + Real.cos (π / 3 - x + π / 6)) 
           ↔ a = 2 :=
by
  sorry

end find_a_for_symmetry_l146_146907


namespace neither_5_nice_nor_6_nice_count_l146_146438

def is_k_nice (N k : ℕ) : Prop :=
  N % k = 1

def count_5_nice (N : ℕ) : ℕ :=
  (N - 1) / 5 + 1

def count_6_nice (N : ℕ) : ℕ :=
  (N - 1) / 6 + 1

def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

def count_30_nice (N : ℕ) : ℕ :=
  (N - 1) / 30 + 1

theorem neither_5_nice_nor_6_nice_count : 
  ∀ N < 200, 
  (N - (count_5_nice 199 + count_6_nice 199 - count_30_nice 199)) = 133 := 
by
  sorry

end neither_5_nice_nor_6_nice_count_l146_146438


namespace storm_deposit_l146_146106

theorem storm_deposit (C : ℝ) (original_amount after_storm_rate before_storm_rate : ℝ) (after_storm full_capacity : ℝ) :
  before_storm_rate = 0.40 →
  after_storm_rate = 0.60 →
  original_amount = 220 * 10^9 →
  before_storm_rate * C = original_amount →
  C = full_capacity →
  after_storm = after_storm_rate * full_capacity →
  after_storm - original_amount = 110 * 10^9 :=
by
  sorry

end storm_deposit_l146_146106


namespace planks_needed_for_surface_l146_146623

theorem planks_needed_for_surface
  (total_tables : ℕ := 5)
  (total_planks : ℕ := 45)
  (planks_per_leg : ℕ := 4) :
  ∃ S : ℕ, total_tables * (planks_per_leg + S) = total_planks ∧ S = 5 :=
by
  use 5
  sorry

end planks_needed_for_surface_l146_146623


namespace total_age_difference_l146_146208

noncomputable def ages_difference (A B C : ℕ) : ℕ :=
  (A + B) - (B + C)

theorem total_age_difference (A B C : ℕ) (h₁ : A + B > B + C) (h₂ : C = A - 11) : ages_difference A B C = 11 :=
by
  sorry

end total_age_difference_l146_146208


namespace comparison_theorem_l146_146894

open Real

noncomputable def comparison (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : Prop :=
  let a := log (sin x)
  let b := sin x
  let c := exp (sin x)
  a < b ∧ b < c

theorem comparison_theorem (x : ℝ) (h : 0 < x ∧ x < π / 2) : comparison x h.1 h.2 :=
by { sorry }

end comparison_theorem_l146_146894


namespace probability_heads_tails_4_tosses_l146_146393

-- Define the probabilities of heads and tails
variables (p q : ℝ)

-- Define the conditions
def unfair_coin (p q : ℝ) : Prop :=
  p ≠ q ∧ p + q = 1 ∧ 2 * p * q = 1/2

-- Define the theorem to prove the probability of two heads and two tails
theorem probability_heads_tails_4_tosses 
  (h_unfair : unfair_coin p q) 
  : 6 * (p * q)^2 = 3 / 8 :=
by sorry

end probability_heads_tails_4_tosses_l146_146393


namespace platform_length_605_l146_146869

noncomputable def length_of_platform (speed_kmh : ℕ) (accel : ℚ) (t_platform : ℚ) (t_man : ℚ) (dist_man_from_platform : ℚ) : ℚ :=
  let speed_ms := (speed_kmh : ℚ) * 1000 / 3600
  let distance_man := speed_ms * t_man + 0.5 * accel * t_man^2
  let train_length := distance_man - dist_man_from_platform
  let distance_platform := speed_ms * t_platform + 0.5 * accel * t_platform^2
  distance_platform - train_length

theorem platform_length_605 :
  length_of_platform 54 0.5 40 20 5 = 605 := by
  sorry

end platform_length_605_l146_146869


namespace set_relationship_l146_146443

def set_M : Set ℚ := {x : ℚ | ∃ m : ℤ, x = m + 1/6}
def set_N : Set ℚ := {x : ℚ | ∃ n : ℤ, x = n/2 - 1/3}
def set_P : Set ℚ := {x : ℚ | ∃ p : ℤ, x = p/2 + 1/6}

theorem set_relationship : set_M ⊆ set_N ∧ set_N = set_P := by
  sorry

end set_relationship_l146_146443


namespace parabola_focus_l146_146784

theorem parabola_focus (a : ℝ) (p : ℝ) (x y : ℝ) :
  a = -3 ∧ p = 6 →
  (y^2 = -2 * p * x) → 
  (y^2 = -12 * x) := 
by sorry

end parabola_focus_l146_146784


namespace xyz_distinct_real_squares_l146_146502

theorem xyz_distinct_real_squares (x y z : ℝ) 
  (h1 : x^2 = 2 + y)
  (h2 : y^2 = 2 + z)
  (h3 : z^2 = 2 + x) 
  (h4 : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  x^2 + y^2 + z^2 = 5 ∨ x^2 + y^2 + z^2 = 6 ∨ x^2 + y^2 + z^2 = 9 :=
by 
  sorry

end xyz_distinct_real_squares_l146_146502


namespace athena_total_spent_l146_146743

-- Define the conditions
def sandwiches_quantity : ℕ := 3
def sandwich_price : ℝ := 3.0
def drinks_quantity : ℕ := 2
def drink_price : ℝ := 2.5

-- Define the calculations
def total_sandwich_cost : ℝ := sandwiches_quantity * sandwich_price
def total_drink_cost : ℝ := drinks_quantity * drink_price

-- Define the total cost
def total_amount_spent : ℝ := total_sandwich_cost + total_drink_cost

-- Prove the total amount spent
theorem athena_total_spent : total_amount_spent = 14 := by
  sorry

end athena_total_spent_l146_146743


namespace candles_lit_time_correct_l146_146506

noncomputable def candle_time : String :=
  let initial_length := 1 -- Since the length is uniform, we use 1
  let rateA := initial_length / (6 * 60) -- Rate at which Candle A burns out
  let rateB := initial_length / (8 * 60) -- Rate at which Candle B burns out
  let t := 320 -- The time in minutes that satisfy the condition
  let time_lit := (16 * 60 - t) / 60 -- Convert minutes to hours
  if time_lit = 10 + 40 / 60 then "10:40 AM" else "Unknown"

theorem candles_lit_time_correct :
  candle_time = "10:40 AM" := 
by
  sorry

end candles_lit_time_correct_l146_146506


namespace h_value_l146_146609

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l146_146609


namespace soccer_league_total_games_l146_146817

theorem soccer_league_total_games :
  let teams := 20
  let regular_games_per_team := 19 * 3
  let total_regular_games := (regular_games_per_team * teams) / 2
  let promotional_games_per_team := 3
  let total_promotional_games := promotional_games_per_team * teams
  let total_games := total_regular_games + total_promotional_games
  total_games = 1200 :=
by
  sorry

end soccer_league_total_games_l146_146817


namespace h_value_l146_146606

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l146_146606


namespace find_number_of_toonies_l146_146250

variable (L T : ℕ)

def condition1 : Prop := L + T = 10
def condition2 : Prop := L + 2 * T = 14

theorem find_number_of_toonies (h1 : condition1 L T) (h2 : condition2 L T) : T = 4 :=
by
  sorry

end find_number_of_toonies_l146_146250


namespace least_number_to_make_divisible_l146_146670

theorem least_number_to_make_divisible (k : ℕ) (h : 1202 + k = 1204) : (2 ∣ 1204) := 
by
  sorry

end least_number_to_make_divisible_l146_146670


namespace vincent_total_packs_l146_146400

-- Definitions based on the conditions
def packs_yesterday : ℕ := 15
def extra_packs_today : ℕ := 10

-- Total packs calculation
def packs_today : ℕ := packs_yesterday + extra_packs_today
def total_packs : ℕ := packs_yesterday + packs_today

-- Proof statement
theorem vincent_total_packs : total_packs = 40 :=
by
  -- Calculate today’s packs
  have h1 : packs_today = 25 := by
    rw [packs_yesterday, extra_packs_today]
    norm_num
  
  -- Calculate the total packs
  have h2 : total_packs = 15 + 25 := by
    rw [packs_yesterday, h1]
  
  -- Conclude the total number of packs
  show total_packs = 40
  rw [h2]
  norm_num

end vincent_total_packs_l146_146400


namespace probability_no_neighbouring_same_color_l146_146893

-- Given conditions
def red_beads : ℕ := 4
def white_beads : ℕ := 2
def blue_beads : ℕ := 2
def total_beads : ℕ := red_beads + white_beads + blue_beads

-- Total permutations
def total_orderings : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

-- Probability calculation proof
theorem probability_no_neighbouring_same_color : (30 / 420 : ℚ) = (1 / 14 : ℚ) :=
by
  -- proof steps
  sorry

end probability_no_neighbouring_same_color_l146_146893


namespace paul_spending_l146_146759

theorem paul_spending :
  let cost_of_dress_shirts := 4 * 15
  let cost_of_pants := 2 * 40
  let cost_of_suit := 150
  let cost_of_sweaters := 2 * 30
  let total_cost := cost_of_dress_shirts + cost_of_pants + cost_of_suit + cost_of_sweaters
  let store_discount := 0.2 * total_cost
  let after_store_discount := total_cost - store_discount
  let coupon_discount := 0.1 * after_store_discount
  let final_amount := after_store_discount - coupon_discount
  final_amount = 252 :=
by
  -- Mathematically equivalent proof problem.
  sorry

end paul_spending_l146_146759


namespace athena_spent_l146_146742

theorem athena_spent :
  let sandwich_cost := 3
  let num_sandwiches := 3
  let drink_cost := 2.5
  let num_drinks := 2
  let total_cost := num_sandwiches * sandwich_cost + num_drinks * drink_cost
  total_cost = 14 :=
by
  let sandwich_cost := 3
  let num_sandwiches := 3
  let drink_cost := 2.5
  let num_drinks := 2
  let total_cost := num_sandwiches * sandwich_cost + num_drinks * drink_cost
  sorry

end athena_spent_l146_146742


namespace high_heels_height_l146_146237

theorem high_heels_height (x : ℝ) :
  let height := 157
  let lower_limbs := 95
  let golden_ratio := 0.618
  (95 + x) / (157 + x) = 0.618 → x = 5.3 :=
sorry

end high_heels_height_l146_146237


namespace mass_of_1m3_l146_146651

/-- The volume of 1 gram of the substance in cubic centimeters cms_per_gram is 1.3333333333333335 cm³. -/
def cms_per_gram : ℝ := 1.3333333333333335

/-- There are 1,000,000 cubic centimeters in 1 cubic meter. -/
def cm3_per_m3 : ℕ := 1000000

/-- Given the volume of 1 gram of the substance, find the mass of 1 cubic meter of the substance. -/
theorem mass_of_1m3 (h1 : cms_per_gram = 1.3333333333333335) (h2 : cm3_per_m3 = 1000000) :
  ∃ m : ℝ, m = 750 :=
by
  sorry

end mass_of_1m3_l146_146651


namespace multiple_of_5_digits_B_l146_146441

theorem multiple_of_5_digits_B (B : ℕ) : B = 0 ∨ B = 5 ↔ 23 * 10 + B % 5 = 0 :=
by
  sorry

end multiple_of_5_digits_B_l146_146441


namespace dozen_Pokemon_cards_per_friend_l146_146772

theorem dozen_Pokemon_cards_per_friend
  (total_cards : ℕ) (num_friends : ℕ) (cards_per_dozen : ℕ)
  (h1 : total_cards = 432)
  (h2 : num_friends = 4)
  (h3 : cards_per_dozen = 12) :
  (total_cards / num_friends) / cards_per_dozen = 9 := 
sorry

end dozen_Pokemon_cards_per_friend_l146_146772


namespace slope_of_line_det_by_two_solutions_l146_146222

theorem slope_of_line_det_by_two_solutions (x y : ℝ) (h : 3 / x + 4 / y = 0) :
  (y = -4 * x / 3) → 
  ∀ x1 x2 y1 y2, (y1 = -4 * x1 / 3) ∧ (y2 = -4 * x2 / 3) → 
  (y2 - y1) / (x2 - x1) = -4 / 3 :=
sorry

end slope_of_line_det_by_two_solutions_l146_146222


namespace workers_contribution_l146_146850

theorem workers_contribution (W C : ℕ) (h1 : W * C = 300000) (h2 : W * (C + 50) = 360000) : W = 1200 :=
by
  sorry

end workers_contribution_l146_146850


namespace positive_difference_two_numbers_l146_146378

theorem positive_difference_two_numbers (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 2 * y - 3 * x = 5) : abs (y - x) = 8 := 
sorry

end positive_difference_two_numbers_l146_146378


namespace find_h_l146_146599

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l146_146599


namespace find_value_correct_l146_146938

-- Definitions for the given conditions
def equation1 (a b : ℚ) : Prop := 3 * a - b = 8
def equation2 (a b : ℚ) : Prop := 4 * b + 7 * a = 13

-- Definition for the question
def find_value (a b : ℚ) : ℚ := 2 * a + b

-- Statement of the proof
theorem find_value_correct (a b : ℚ) (h1 : equation1 a b) (h2 : equation2 a b) : find_value a b = 73 / 19 := 
by 
  sorry

end find_value_correct_l146_146938


namespace P_P_eq_P_eight_equals_58_l146_146488

open Polynomial

noncomputable def P(x : ℚ) : ℚ := x^2 - x + 2

theorem P_P_eq :
  (P ∘ P)(x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4 := sorry

theorem P_eight_equals_58 :
  P 8 = 58 := sorry

end P_P_eq_P_eight_equals_58_l146_146488


namespace chinese_character_equation_l146_146617

noncomputable def units_digit (n: ℕ) : ℕ :=
  n % 10

noncomputable def tens_digit (n: ℕ) : ℕ :=
  (n / 10) % 10

noncomputable def hundreds_digit (n: ℕ) : ℕ :=
  (n / 100) % 10

def Math : ℕ := 25
def LoveMath : ℕ := 125
def ILoveMath : ℕ := 3125

theorem chinese_character_equation :
  Math * LoveMath = ILoveMath :=
by
  have h_units_math := units_digit Math
  have h_units_lovemath := units_digit LoveMath
  have h_units_ilovemath := units_digit ILoveMath
  
  have h_tens_math := tens_digit Math
  have h_tens_lovemath := tens_digit LoveMath
  have h_tens_ilovemath := tens_digit ILoveMath

  have h_hundreds_lovemath := hundreds_digit LoveMath
  have h_hundreds_ilovemath := hundreds_digit ILoveMath

  -- Check conditions:
  -- h_units_* should be 0, 1, 5 or 6
  -- h_tens_math == h_tens_lovemath == h_tens_ilovemath
  -- h_hundreds_lovemath == h_hundreds_ilovemath

  sorry -- Proof would go here

end chinese_character_equation_l146_146617


namespace jerry_needs_money_l146_146926

theorem jerry_needs_money 
  (current_count : ℕ) (total_needed : ℕ) (cost_per_action_figure : ℕ)
  (h1 : current_count = 7) 
  (h2 : total_needed = 16) 
  (h3 : cost_per_action_figure = 8) :
  (total_needed - current_count) * cost_per_action_figure = 72 :=
by sorry

end jerry_needs_money_l146_146926


namespace infinite_series_limit_l146_146546
  
noncomputable def series_limit := 2 + (1 / (3 - 1)) / 3 + (1 / (9 - 1)) / 9

theorem infinite_series_limit : 
  series_limit = 21 / 8 :=
by 
  sorry

end infinite_series_limit_l146_146546


namespace gcd_840_1764_gcd_459_357_l146_146851

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := sorry

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := sorry

end gcd_840_1764_gcd_459_357_l146_146851


namespace swim_club_members_l146_146096

theorem swim_club_members (X : ℝ) 
  (h1 : 0.30 * X = 0.30 * X)
  (h2 : 0.70 * X = 42) : X = 60 :=
sorry

end swim_club_members_l146_146096


namespace minimum_empty_cells_face_move_minimum_empty_cells_diagonal_move_l146_146414

-- Definition for Problem Part (a)
def box_dimensions := (3, 5, 7)
def initial_cockchafers := 3 * 5 * 7 -- or 105

-- Defining the theorem for part (a)
theorem minimum_empty_cells_face_move (d : (ℕ × ℕ × ℕ)) (n : ℕ) :
  d = box_dimensions →
  n = initial_cockchafers →
  ∃ k ≥ 1, k = 1 :=
by
  intros hdim hn
  sorry

-- Definition for Problem Part (b)
def row_odd_cells := 2 * 5 * 7  
def row_even_cells := 1 * 5 * 7  

-- Defining the theorem for part (b)
theorem minimum_empty_cells_diagonal_move (r_odd r_even : ℕ) :
  r_odd = row_odd_cells →
  r_even = row_even_cells →
  ∃ m ≥ 35, m = 35 :=
by
  intros ho he
  sorry

end minimum_empty_cells_face_move_minimum_empty_cells_diagonal_move_l146_146414


namespace surface_area_of_4cm_cube_after_corner_removal_l146_146700

noncomputable def surface_area_after_corner_removal (cube_side original_surface_length corner_cube_side : ℝ) : ℝ := 
  let num_faces : ℕ := 6
  let num_corners : ℕ := 8
  let surface_area_one_face := cube_side * cube_side
  let original_surface_area := num_faces * surface_area_one_face
  let corner_surface_area_one_face := 3 * (corner_cube_side * corner_cube_side)
  let exposed_surface_area_one_face := 3 * (corner_cube_side * corner_cube_side)
  let net_change_per_corner_cube := -corner_surface_area_one_face + exposed_surface_area_one_face
  let total_change := num_corners * net_change_per_corner_cube
  original_surface_area + total_change

theorem surface_area_of_4cm_cube_after_corner_removal : 
  ∀ (cube_side original_surface_length corner_cube_side : ℝ), 
  cube_side = 4 ∧ original_surface_length = 4 ∧ corner_cube_side = 2 →
  surface_area_after_corner_removal cube_side original_surface_length corner_cube_side = 96 :=
by
  intros cube_side original_surface_length corner_cube_side h
  rcases h with ⟨hs, ho, hc⟩
  rw [hs, ho, hc]
  sorry

end surface_area_of_4cm_cube_after_corner_removal_l146_146700


namespace quadratic_inequality_solution_set_l146_146132

theorem quadratic_inequality_solution_set
  (a b x : ℝ)
  (h1 : ∀ x, a * (x + b) * (x + 5 / a) > 0 ↔ x < -1 ∨ 3 < x) :
  (x^2 + b*x - 2*a < 0) ↔ (-2 < x ∧ x < 5) := 
by
  sorry

end quadratic_inequality_solution_set_l146_146132


namespace students_selecting_water_l146_146873

-- Definitions of percentages and given values.
def p : ℝ := 0.7
def q : ℝ := 0.1
def n : ℕ := 140

-- The Lean statement to prove the number of students who selected water.
theorem students_selecting_water (p_eq : p = 0.7) (q_eq : q = 0.1) (n_eq : n = 140) :
  ∃ w : ℕ, w = (q / p) * n ∧ w = 20 :=
by sorry

end students_selecting_water_l146_146873


namespace goods_amount_decreased_initial_goods_amount_total_fees_l146_146856

-- Define the conditions as variables
def tonnages : List Int := [31, -31, -16, 34, -38, -20]
def final_goods : Int := 430
def fee_per_ton : Int := 5

-- Prove that the amount of goods in the warehouse has decreased
theorem goods_amount_decreased : (tonnages.sum < 0) := by
  sorry

-- Prove the initial amount of goods in the warehouse
theorem initial_goods_amount : (final_goods + tonnages.sum = 470) := by
  sorry

-- Prove the total loading and unloading fees
theorem total_fees : (tonnages.map Int.natAbs).sum * fee_per_ton = 850 := by
  sorry

end goods_amount_decreased_initial_goods_amount_total_fees_l146_146856


namespace hypotenuse_length_l146_146319

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l146_146319


namespace count_words_200_l146_146624

theorem count_words_200 : 
  let single_word_numbers := 29
  let compound_words_21_to_99 := 144
  let compound_words_100_to_199 := 54 + 216
  single_word_numbers + compound_words_21_to_99 + compound_words_100_to_199 = 443 :=
by
  sorry

end count_words_200_l146_146624


namespace number_of_cows_is_six_l146_146848

variable (C H : Nat) -- C for cows and H for chickens

-- Number of legs is 12 more than twice the number of heads.
def cows_count_condition : Prop :=
  4 * C + 2 * H = 2 * (C + H) + 12

theorem number_of_cows_is_six (h : cows_count_condition C H) : C = 6 :=
sorry

end number_of_cows_is_six_l146_146848


namespace minimum_value_l146_146445

variable (a b : ℝ)

-- Assume a and b are positive real numbers
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)

-- Given the condition a + b = 2
variable (h₂ : a + b = 2)

theorem minimum_value : (1 / a) + (2 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end minimum_value_l146_146445


namespace find_h_l146_146598

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l146_146598


namespace product_469157_9999_l146_146113

theorem product_469157_9999 : 469157 * 9999 = 4690872843 := by
  -- computation and its proof would go here
  sorry

end product_469157_9999_l146_146113


namespace hypotenuse_length_l146_146309

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l146_146309


namespace abs_expression_equals_l146_146118

theorem abs_expression_equals (h : Real.pi < 12) : 
  abs (Real.pi - abs (Real.pi - 12)) = 12 - 2 * Real.pi := 
by
  sorry

end abs_expression_equals_l146_146118


namespace mean_of_added_numbers_l146_146964

theorem mean_of_added_numbers (mean_seven : ℝ) (mean_ten : ℝ) (x y z : ℝ)
    (h1 : mean_seven = 40)
    (h2 : mean_ten = 55) :
    (mean_seven * 7 + x + y + z) / 10 = mean_ten → (x + y + z) / 3 = 90 :=
by sorry

end mean_of_added_numbers_l146_146964


namespace sampling_interval_divisor_l146_146006

theorem sampling_interval_divisor (P : ℕ) (hP : P = 524) (k : ℕ) (hk : k ∣ P) : k = 4 :=
by
  sorry

end sampling_interval_divisor_l146_146006


namespace total_video_hours_in_june_l146_146107

-- Definitions for conditions
def upload_rate_first_half : ℕ := 10 -- one-hour videos per day
def upload_rate_second_half : ℕ := 20 -- doubled one-hour videos per day
def days_in_half_month : ℕ := 15
def total_days_in_june : ℕ := 30

-- Number of video hours uploaded in the first half of the month
def video_hours_first_half : ℕ := upload_rate_first_half * days_in_half_month

-- Number of video hours uploaded in the second half of the month
def video_hours_second_half : ℕ := upload_rate_second_half * days_in_half_month

-- Total number of video hours in June
theorem total_video_hours_in_june : video_hours_first_half + video_hours_second_half = 450 :=
by {
  sorry
}

end total_video_hours_in_june_l146_146107


namespace distributeCandies_l146_146015

-- Define the conditions as separate definitions.

-- Number of candies
def candies : ℕ := 10

-- Number of boxes
def boxes : ℕ := 5

-- Condition that each box gets at least one candy
def atLeastOne (candyDist : Fin boxes → ℕ) : Prop :=
  ∀ b, candyDist b > 0

-- Function to count the number of ways to distribute candies
noncomputable def countWaysToDistribute (candies : ℕ) (boxes : ℕ) : ℕ :=
  -- Function to compute the number of ways
  -- (assuming a correct implementation is provided)
  sorry -- Placeholder for the actual counting implementation

-- Theorem to prove the number of distributions
theorem distributeCandies : countWaysToDistribute candies boxes = 7 := 
by {
  -- Proof omitted
  sorry
}

end distributeCandies_l146_146015


namespace m_range_circle_l146_146714

noncomputable def circle_equation (m : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + 2 * x + 4 * y + m = 0

theorem m_range_circle (m : ℝ) : circle_equation m → m < 5 := by
  sorry

end m_range_circle_l146_146714


namespace parallelogram_area_formula_l146_146636

noncomputable def parallelogram_area (ha hb : ℝ) (γ : ℝ) : ℝ := 
  ha * hb / Real.sin γ

theorem parallelogram_area_formula (ha hb γ : ℝ) (a b : ℝ) 
  (h₁ : Real.sin γ ≠ 0) :
  (parallelogram_area ha hb γ = ha * hb / Real.sin γ) := by
  sorry

end parallelogram_area_formula_l146_146636


namespace find_y_l146_146727

theorem find_y (y : ℝ) (h : 9 * y^3 = y * 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end find_y_l146_146727


namespace avg_of_two_numbers_l146_146200

theorem avg_of_two_numbers (a b c d : ℕ) (h_different: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_positive: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_average: (a + b + c + d) / 4 = 4)
  (h_max_diff: ∀ x y : ℕ, (x ≠ y ∧ x > 0 ∧ y > 0 ∧ x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ x ≠ d ∧ y ≠ a ∧ y ≠ b ∧ y ≠ c ∧ y ≠ d) → (max x y - min x y <= max a d - min a d)) : 
  (a + b + c + d - min a (min b (min c d)) - max a (max b (max c d))) / 2 = 5 / 2 :=
by sorry

end avg_of_two_numbers_l146_146200


namespace range_of_x_when_y_lt_0_l146_146466

variable (a b c n m : ℝ)

-- The definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions given in the problem
axiom value_at_neg1 : quadratic_function a b c (-1) = 4
axiom value_at_0 : quadratic_function a b c 0 = 0
axiom value_at_1 : quadratic_function a b c 1 = n
axiom value_at_2 : quadratic_function a b c 2 = m
axiom value_at_3 : quadratic_function a b c 3 = 4

-- Proof statement
theorem range_of_x_when_y_lt_0 : ∀ (x : ℝ), quadratic_function a b c x < 0 ↔ 0 < x ∧ x < 2 :=
sorry

end range_of_x_when_y_lt_0_l146_146466


namespace percent_runs_by_running_between_wickets_l146_146526

theorem percent_runs_by_running_between_wickets :
  (132 - (12 * 4 + 2 * 6)) / 132 * 100 = 54.54545454545455 :=
by
  sorry

end percent_runs_by_running_between_wickets_l146_146526


namespace cylinder_unoccupied_volume_l146_146058

theorem cylinder_unoccupied_volume (r h_cylinder h_cone : ℝ) 
  (h : r = 10 ∧ h_cylinder = 30 ∧ h_cone = 15) :
  (π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π) :=
by
  rcases h with ⟨rfl, rfl, rfl⟩
  simp
  sorry

end cylinder_unoccupied_volume_l146_146058


namespace cube_volumes_total_l146_146089

theorem cube_volumes_total :
  let v1 := 5^3
  let v2 := 6^3
  let v3 := 7^3
  v1 + v2 + v3 = 684 := by
  -- Here will be the proof using Lean's tactics
  sorry

end cube_volumes_total_l146_146089


namespace friend_spent_more_l146_146844

variable (total_spent : ℕ)
variable (friend_spent : ℕ)
variable (you_spent : ℕ)

-- Conditions
axiom total_is_11 : total_spent = 11
axiom friend_is_7 : friend_spent = 7
axiom spending_relation : total_spent = friend_spent + you_spent

-- Question
theorem friend_spent_more : friend_spent - you_spent = 3 :=
by
  sorry -- Here should be the formal proof

end friend_spent_more_l146_146844


namespace complex_expression_value_l146_146273

theorem complex_expression_value :
  (i^3 * (1 + i)^2 = 2) :=
by
  sorry

end complex_expression_value_l146_146273


namespace value_of_expression_l146_146226

theorem value_of_expression (a : ℝ) (h : a = 1/2) : 
  (2 * a⁻¹ + a⁻¹ / 2) / a = 10 :=
by
  sorry

end value_of_expression_l146_146226


namespace paulsons_income_increase_percentage_l146_146764

-- Definitions of conditions
variables {I : ℝ} -- Original income
def E := 0.75 * I -- Original expenditure
def S := I - E -- Original savings

-- New income defined by percentage increase
variables {x : ℝ} -- percentage increase in income in fraction (0.20 for 20%)
def I_new := I * (1 + x)

-- New expenditure
def E_new := E * 1.10

-- New savings
def S_new := I_new - E_new

-- Percentage increase in savings given as 49.99999999999996% ~= 50%
def percentage_increase_in_savings := 0.50

-- New savings in terms of original savings
def S_new_from_percentage := S * (1 + percentage_increase_in_savings)

-- Theorem to prove that the percentage increase in income is 20%
theorem paulsons_income_increase_percentage (h : S_new = S_new_from_percentage) : x = 0.20 :=
by sorry

end paulsons_income_increase_percentage_l146_146764


namespace joshua_total_payment_is_correct_l146_146016

noncomputable def total_cost : ℝ := 
  let t_shirt_price := 8
  let sweater_price := 18
  let jacket_price := 80
  let jeans_price := 35
  let shoes_price := 60
  let jacket_discount := 0.10
  let shoes_discount := 0.15
  let clothing_tax_rate := 0.05
  let shoes_tax_rate := 0.08

  let t_shirt_count := 6
  let sweater_count := 4
  let jacket_count := 5
  let jeans_count := 3
  let shoes_count := 2

  let t_shirts_subtotal := t_shirt_price * t_shirt_count
  let sweaters_subtotal := sweater_price * sweater_count
  let jackets_subtotal := jacket_price * jacket_count
  let jeans_subtotal := jeans_price * jeans_count
  let shoes_subtotal := shoes_price * shoes_count

  let jackets_discounted := jackets_subtotal * (1 - jacket_discount)
  let shoes_discounted := shoes_subtotal * (1 - shoes_discount)

  let total_before_tax := t_shirts_subtotal + sweaters_subtotal + jackets_discounted + jeans_subtotal + shoes_discounted

  let t_shirts_tax := t_shirts_subtotal * clothing_tax_rate
  let sweaters_tax := sweaters_subtotal * clothing_tax_rate
  let jackets_tax := jackets_discounted * clothing_tax_rate
  let jeans_tax := jeans_subtotal * clothing_tax_rate
  let shoes_tax := shoes_discounted * shoes_tax_rate

  total_before_tax + t_shirts_tax + sweaters_tax + jackets_tax + jeans_tax + shoes_tax

theorem joshua_total_payment_is_correct : total_cost = 724.41 := by
  sorry

end joshua_total_payment_is_correct_l146_146016


namespace sum_of_first_49_primes_l146_146271

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                                   61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 
                                   137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 
                                   199, 211, 223, 227]

theorem sum_of_first_49_primes : first_49_primes.sum = 10787 :=
by
  -- Proof to be filled in
  sorry

end sum_of_first_49_primes_l146_146271


namespace area_of_circle_l146_146696

theorem area_of_circle 
  (r : ℝ → ℝ)
  (h : ∀ θ : ℝ, r θ = 3 * Real.cos θ - 4 * Real.sin θ) :
  ∃ A : ℝ, A = (25 / 4) * Real.pi :=
by
  sorry

end area_of_circle_l146_146696


namespace color_set_no_arith_prog_same_color_l146_146639

def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 1987}

def colors : Fin 4 := sorry  -- Color indexing set (0, 1, 2, 3)

def valid_coloring (c : ℕ → Fin 4) : Prop :=
  ∀ (a d : ℕ) (h₁ : a ∈ M) (h₂ : d ≠ 0) (h₃ : ∀ k, a + k * d ∈ M ∧ k < 10), 
  ¬ ∀ k, c (a + k * d) = c a

theorem color_set_no_arith_prog_same_color :
  ∃ (c : ℕ → Fin 4), valid_coloring c :=
sorry

end color_set_no_arith_prog_same_color_l146_146639


namespace crop_yield_solution_l146_146334

variable (x y : ℝ)

axiom h1 : 3 * x + 6 * y = 4.7
axiom h2 : 5 * x + 3 * y = 5.5

theorem crop_yield_solution :
  x = 0.9 ∧ y = 1/3 :=
by
  sorry

end crop_yield_solution_l146_146334


namespace card_draw_probability_l146_146382

theorem card_draw_probability : 
  let P1 := (12 / 52 : ℚ) * (4 / 51 : ℚ) * (13 / 50 : ℚ)
  let P2 := (1 / 52 : ℚ) * (3 / 51 : ℚ) * (13 / 50 : ℚ)
  P1 + P2 = (63 / 107800 : ℚ) :=
by
  sorry

end card_draw_probability_l146_146382


namespace find_product_of_variables_l146_146518

variables (a b c d : ℚ)

def system_of_equations (a b c d : ℚ) :=
  3 * a + 4 * b + 6 * c + 9 * d = 45 ∧
  4 * (d + c) = b + 1 ∧
  4 * b + 2 * c = a ∧
  2 * c - 2 = d

theorem find_product_of_variables :
  system_of_equations a b c d → a * b * c * d = 162 / 185 :=
by sorry

end find_product_of_variables_l146_146518


namespace find_h_l146_146603

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l146_146603


namespace common_ratio_of_geometric_series_l146_146792

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l146_146792


namespace sum_of_prime_factors_of_143_l146_146996

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l146_146996


namespace geometric_series_ratio_l146_146786

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l146_146786


namespace intersection_of_lines_l146_146128

theorem intersection_of_lines : ∃ (x y : ℚ), 8 * x - 5 * y = 20 ∧ 6 * x + 2 * y = 18 ∧ x = 65 / 23 ∧ y = 1 / 2 :=
by {
  -- The solution to the theorem is left as an exercise
  sorry
}

end intersection_of_lines_l146_146128


namespace find_value_of_6b_l146_146914

theorem find_value_of_6b (a b : ℝ) (h1 : 10 * a = 20) (h2 : 120 * a * b = 800) : 6 * b = 20 :=
by
  sorry

end find_value_of_6b_l146_146914


namespace range_of_a_for_inequality_l146_146157

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 4) :=
by sorry

end range_of_a_for_inequality_l146_146157


namespace arithmetic_sequence_common_diff_sum_of_five_terms_l146_146566

-- Definitions of arithmetic sequence and initial terms
def arithmetic_sequence (a d : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := arithmetic_sequence a d n + d

-- Sum of the first n terms of arithmetic sequence
def sum_of_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  (n + 1) * a + n * (n + 1) / 2 * d

-- Problem condition, a_5 + a_6 = 2 * exp(a_4)
def problem_condition (a d : ℝ) :=
  let a4 := arithmetic_sequence a d 4
  let a5 := arithmetic_sequence a d 5
  let a6 := arithmetic_sequence a d 6
  a5 + a6 = 2 * Real.exp a4

-- Lean statements of the proof problems
theorem arithmetic_sequence_common_diff (a d : ℝ) (h : problem_condition a d) : d ≥ 2 / 3 := sorry

theorem sum_of_five_terms (a d : ℝ) (h : problem_condition a d) : sum_of_first_n_terms a d 4 < 0 := sorry

end arithmetic_sequence_common_diff_sum_of_five_terms_l146_146566


namespace find_h_l146_146601

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l146_146601


namespace distance_between_points_l146_146779

theorem distance_between_points : abs (3 - (-2)) = 5 := 
by
  sorry

end distance_between_points_l146_146779


namespace sufficient_but_not_necessary_l146_146341

variable {a b : ℝ}

theorem sufficient_but_not_necessary (h : b < a ∧ a < 0) : 1 / a < 1 / b :=
by
  sorry

end sufficient_but_not_necessary_l146_146341


namespace largest_base5_to_base10_l146_146078

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l146_146078


namespace harrys_age_l146_146017

-- Definitions of the ages
variable (Kiarra Bea Job Figaro Harry : ℕ)

-- Given conditions
variable (h1 : Kiarra = 2 * Bea)
variable (h2 : Job = 3 * Bea)
variable (h3 : Figaro = Job + 7)
variable (h4 : Harry = Figaro / 2)
variable (h5 : Kiarra = 30)

-- The statement to prove
theorem harrys_age : Harry = 26 := sorry

end harrys_age_l146_146017


namespace village_population_l146_146411

theorem village_population (P : ℝ) (h : 0.8 * P = 64000) : P = 80000 :=
sorry

end village_population_l146_146411


namespace find_n_l146_146522

theorem find_n (n : ℕ) (hn : (Nat.choose n 2 : ℚ) / 2^n = 10 / 32) : n = 5 :=
by
  sorry

end find_n_l146_146522


namespace hypotenuse_length_l146_146310

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l146_146310


namespace complex_number_identity_l146_146906

theorem complex_number_identity (z : ℂ) (h : z = 1 + (1 : ℂ) * I) : z^2 + z = 1 + 3 * I := 
sorry

end complex_number_identity_l146_146906


namespace prob_match_ends_two_games_A_wins_prob_match_ends_four_games_prob_A_wins_overall_l146_146956

noncomputable def prob_A_wins_game := 2 / 3
noncomputable def prob_B_wins_game := 1 / 3

/-- The probability that the match ends after two games with player A's victory is 4/9. -/
theorem prob_match_ends_two_games_A_wins :
  prob_A_wins_game * prob_A_wins_game = 4 / 9 := by
  sorry

/-- The probability that the match ends exactly after four games is 20/81. -/
theorem prob_match_ends_four_games :
  2 * prob_A_wins_game * prob_B_wins_game * (prob_A_wins_game^2 + prob_B_wins_game^2) = 20 / 81 := by
  sorry

/-- The probability that player A wins the match overall is 74/81. -/
theorem prob_A_wins_overall :
  (prob_A_wins_game^2 + 2 * prob_A_wins_game * prob_B_wins_game * prob_A_wins_game^2
  + 2 * prob_A_wins_game * prob_B_wins_game * prob_A_wins_game * prob_B_wins_game) / (prob_A_wins_game + prob_B_wins_game) = 74 / 81 := by
  sorry

end prob_match_ends_two_games_A_wins_prob_match_ends_four_games_prob_A_wins_overall_l146_146956


namespace factorization_analysis_l146_146883

variable (a b c : ℝ)

theorem factorization_analysis : a^2 - 2 * a * b + b^2 - c^2 = (a - b + c) * (a - b - c) := 
sorry

end factorization_analysis_l146_146883


namespace abs_add_opposite_signs_l146_146577

theorem abs_add_opposite_signs (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a * b < 0) : |a + b| = 1 := 
sorry

end abs_add_opposite_signs_l146_146577


namespace total_age_difference_l146_146207

noncomputable def ages_difference (A B C : ℕ) : ℕ :=
  (A + B) - (B + C)

theorem total_age_difference (A B C : ℕ) (h₁ : A + B > B + C) (h₂ : C = A - 11) : ages_difference A B C = 11 :=
by
  sorry

end total_age_difference_l146_146207


namespace income_increase_percentage_l146_146765

theorem income_increase_percentage (I : ℝ) (P : ℝ) (h1 : 0 < I)
  (h2 : 0 ≤ P) (h3 : 0.75 * I + 0.075 * I = 0.825 * I) 
  (h4 : 1.5 * (0.25 * I) = ((I * (1 + P / 100)) - 0.825 * I)) 
  : P = 20 := by
sorry

end income_increase_percentage_l146_146765


namespace athena_total_spent_l146_146746

def cost_of_sandwiches (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) : ℝ :=
  num_sandwiches * cost_per_sandwich

def cost_of_drinks (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  num_drinks * cost_per_drink

def total_cost (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  cost_of_sandwiches num_sandwiches cost_per_sandwich + cost_of_drinks num_drinks cost_per_drink

theorem athena_total_spent :
  total_cost 3 3 2 2.5 = 14 :=
by 
  sorry

end athena_total_spent_l146_146746


namespace correlated_relationships_l146_146423

-- Definitions for the conditions are arbitrary
-- In actual use cases, these would be replaced with real mathematical conditions
def great_teachers_produce_outstanding_students : Prop := sorry
def volume_of_sphere_with_radius : Prop := sorry
def apple_production_climate : Prop := sorry
def height_and_weight : Prop := sorry
def taxi_fare_distance_traveled : Prop := sorry
def crows_cawing_bad_omen : Prop := sorry

-- The final theorem statement
theorem correlated_relationships : 
  great_teachers_produce_outstanding_students ∧
  apple_production_climate ∧
  height_and_weight ∧
  ¬ volume_of_sphere_with_radius ∧ 
  ¬ taxi_fare_distance_traveled ∧ 
  ¬ crows_cawing_bad_omen :=
sorry

end correlated_relationships_l146_146423


namespace total_DVDs_CDs_sold_l146_146008

theorem total_DVDs_CDs_sold (C D : ℕ) (h1 : D = 1.6 * C) (h2 : D = 168) : 
  D + C = 273 :=
by
  sorry

end total_DVDs_CDs_sold_l146_146008


namespace range_of_x_for_expression_meaningful_l146_146156

theorem range_of_x_for_expression_meaningful (x : ℝ) :
  (x - 1 > 0 ∧ x ≠ 1) ↔ x > 1 :=
by
  sorry

end range_of_x_for_expression_meaningful_l146_146156


namespace smallest_positive_multiple_l146_146512

theorem smallest_positive_multiple (a : ℕ) (h : 17 * a % 53 = 7) : 17 * a = 544 :=
sorry

end smallest_positive_multiple_l146_146512


namespace megan_savings_days_l146_146932

theorem megan_savings_days :
  let josiah_saving_rate : ℝ := 0.25
  let josiah_days : ℕ := 24
  let josiah_total := josiah_saving_rate * josiah_days

  let leah_saving_rate : ℝ := 0.5
  let leah_days : ℕ := 20
  let leah_total := leah_saving_rate * leah_days

  let total_savings : ℝ := 28.0
  let josiah_leah_total := josiah_total + leah_total
  let megan_total := total_savings - josiah_leah_total

  let megan_saving_rate := 2 * leah_saving_rate
  let megan_days := megan_total / megan_saving_rate
  
  megan_days = 12 :=
by
  sorry

end megan_savings_days_l146_146932


namespace evaporation_rate_is_200_ml_per_hour_l146_146924

-- Definitions based on the given conditions
def faucet_drip_rate : ℕ := 40 -- ml per minute
def running_time : ℕ := 9 -- hours
def dumped_water : ℕ := 12000 -- ml (converted from liters)
def water_left : ℕ := 7800 -- ml

-- Alias for total water dripped in running_time
noncomputable def total_dripped_water : ℕ := faucet_drip_rate * 60 * running_time

-- Total water that should have been in the bathtub without evaporation
noncomputable def total_without_evaporation : ℕ := total_dripped_water - dumped_water

-- Water evaporated
noncomputable def evaporated_water : ℕ := total_without_evaporation - water_left

-- Evaporation rate in ml/hour
noncomputable def evaporation_rate : ℕ := evaporated_water / running_time

-- The goal theorem statement
theorem evaporation_rate_is_200_ml_per_hour : evaporation_rate = 200 := by
  -- proof here
  sorry

end evaporation_rate_is_200_ml_per_hour_l146_146924


namespace min_value_expr_l146_146554

theorem min_value_expr : ∃ (x : ℝ), (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 ∧ 
  ∀ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by
  sorry

end min_value_expr_l146_146554


namespace smallest_n_sqrt_12n_integer_l146_146903

theorem smallest_n_sqrt_12n_integer : ∃ n : ℕ, (n > 0) ∧ (∃ k : ℕ, 12 * n = k^2) ∧ n = 3 := by
  sorry

end smallest_n_sqrt_12n_integer_l146_146903


namespace solution_set_x2_minus_x_lt_0_l146_146658

theorem solution_set_x2_minus_x_lt_0 :
  ∀ x : ℝ, (0 < x ∧ x < 1) ↔ x^2 - x < 0 := 
by
  sorry

end solution_set_x2_minus_x_lt_0_l146_146658


namespace complement_A_in_U_l146_146909

noncomputable def U : Set ℕ := {0, 1, 2}
noncomputable def A : Set ℕ := {x | x^2 - x = 0}
noncomputable def complement_U (A : Set ℕ) : Set ℕ := U \ A

theorem complement_A_in_U : 
  complement_U {x | x^2 - x = 0} = {2} := 
sorry

end complement_A_in_U_l146_146909


namespace joan_games_l146_146471

theorem joan_games (games_this_year games_total games_last_year : ℕ) 
  (h1 : games_this_year = 4) 
  (h2 : games_total = 9) 
  (h3 : games_total = games_this_year + games_last_year) :
  games_last_year = 5 :=
by {
  -- The proof goes here
  sorry
}

end joan_games_l146_146471


namespace hypotenuse_length_l146_146322

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l146_146322


namespace vincent_total_packs_l146_146401

-- Definitions based on the conditions
def packs_yesterday : ℕ := 15
def extra_packs_today : ℕ := 10

-- Total packs calculation
def packs_today : ℕ := packs_yesterday + extra_packs_today
def total_packs : ℕ := packs_yesterday + packs_today

-- Proof statement
theorem vincent_total_packs : total_packs = 40 :=
by
  -- Calculate today’s packs
  have h1 : packs_today = 25 := by
    rw [packs_yesterday, extra_packs_today]
    norm_num
  
  -- Calculate the total packs
  have h2 : total_packs = 15 + 25 := by
    rw [packs_yesterday, h1]
  
  -- Conclude the total number of packs
  show total_packs = 40
  rw [h2]
  norm_num

end vincent_total_packs_l146_146401


namespace negation_proposition_equivalence_l146_146033

theorem negation_proposition_equivalence : 
    (¬ ∃ x_0 : ℝ, (x_0^2 + 1 > 0) ∨ (x_0 > Real.sin x_0)) ↔ 
    (∀ x : ℝ, (x^2 + 1 ≤ 0) ∧ (x ≤ Real.sin x)) :=
by 
    sorry

end negation_proposition_equivalence_l146_146033


namespace unoccupied_volume_in_container_l146_146537

-- defining constants
def side_length_container := 12
def side_length_ice_cube := 3
def number_of_ice_cubes := 8
def water_fill_fraction := 3 / 4

-- defining volumes
def volume_container := side_length_container ^ 3
def volume_water := volume_container * water_fill_fraction
def volume_ice_cube := side_length_ice_cube ^ 3
def total_volume_ice := volume_ice_cube * number_of_ice_cubes
def volume_unoccupied := volume_container - (volume_water + total_volume_ice)

-- The theorem to be proved
theorem unoccupied_volume_in_container : volume_unoccupied = 216 := by
  -- Proof steps will go here
  sorry

end unoccupied_volume_in_container_l146_146537


namespace real_y_iff_x_l146_146626

open Real

-- Definitions based on the conditions
def quadratic_eq (y x : ℝ) : ℝ := 9 * y^2 - 3 * x * y + x + 8

-- The main theorem to prove
theorem real_y_iff_x (x : ℝ) : (∃ y : ℝ, quadratic_eq y x = 0) ↔ x ≤ -4 ∨ x ≥ 8 := 
sorry

end real_y_iff_x_l146_146626


namespace cube_volume_correct_l146_146174

-- Define the height and base dimensions of the pyramid
def pyramid_height := 15
def pyramid_base_length := 12
def pyramid_base_width := 8

-- Define the side length of the cube-shaped box
def cube_side_length := max pyramid_height pyramid_base_length

-- Define the volume of the cube-shaped box
def cube_volume := cube_side_length ^ 3

-- Theorem statement: the volume of the smallest cube-shaped box that can fit the pyramid is 3375 cubic inches
theorem cube_volume_correct : cube_volume = 3375 := by
  sorry

end cube_volume_correct_l146_146174


namespace find_f_minus_1_l146_146143

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_at_2 : f 2 = 4

theorem find_f_minus_1 : f (-1) = -2 := 
by 
  sorry

end find_f_minus_1_l146_146143


namespace correct_average_marks_l146_146613

-- Define all the given conditions
def average_marks : ℕ := 92
def number_of_students : ℕ := 25
def wrong_mark : ℕ := 75
def correct_mark : ℕ := 30

-- Define variables for total marks calculations
def total_marks_with_wrong : ℕ := average_marks * number_of_students
def total_marks_with_correct : ℕ := total_marks_with_wrong - wrong_mark + correct_mark

-- Goal: Prove that the correct average marks is 90.2
theorem correct_average_marks :
  (total_marks_with_correct : ℝ) / (number_of_students : ℝ) = 90.2 :=
by
  sorry

end correct_average_marks_l146_146613


namespace total_peanut_cost_l146_146252

def peanut_cost_per_pound : ℝ := 3
def minimum_pounds : ℝ := 15
def extra_pounds : ℝ := 20

theorem total_peanut_cost :
  (minimum_pounds + extra_pounds) * peanut_cost_per_pound = 105 :=
by
  sorry

end total_peanut_cost_l146_146252


namespace find_C_coordinates_l146_146567

-- Define the points A, B, and the vector relationship
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)
def C : ℝ × ℝ := (-3, 9)

-- The condition stating vector AC is twice vector AB
def vector_condition (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1, C.2 - A.2) = (2 * (B.1 - A.1), 2 * (B.2 - A.2))

-- The theorem we need to prove
theorem find_C_coordinates (A B C : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (-1, 5))
  (hCondition : vector_condition A B C) : C = (-3, 9) :=
by
  rw [hA, hB] at hCondition
  -- sorry here skips the proof
  sorry

end find_C_coordinates_l146_146567


namespace largest_base5_to_base10_l146_146076

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l146_146076


namespace games_given_away_l146_146284

/-- Gwen had ninety-eight DS games. 
    After she gave some to her friends she had ninety-one left.
    Prove that she gave away 7 DS games. -/
theorem games_given_away (original_games : ℕ) (games_left : ℕ) (games_given : ℕ) 
  (h1 : original_games = 98) 
  (h2 : games_left = 91) 
  (h3 : games_given = original_games - games_left) : 
  games_given = 7 :=
sorry

end games_given_away_l146_146284


namespace cos_alpha_value_l146_146292

theorem cos_alpha_value (α β γ: ℝ) (h1: β = 2 * α) (h2: γ = 4 * α)
 (h3: 2 * (Real.sin β) = (Real.sin α + Real.sin γ)) : Real.cos α = -1/2 := 
by
  sorry

end cos_alpha_value_l146_146292


namespace percent_of_whole_l146_146409

theorem percent_of_whole (Part Whole : ℝ) (Percent : ℝ) (hPart : Part = 160) (hWhole : Whole = 50) :
  Percent = (Part / Whole) * 100 → Percent = 320 :=
by
  rw [hPart, hWhole]
  sorry

end percent_of_whole_l146_146409


namespace sara_total_cents_l146_146638

-- Define the conditions as constants
def quarters : ℕ := 11
def value_per_quarter : ℕ := 25

-- Define the total amount formula based on the conditions
def total_cents (q : ℕ) (v : ℕ) : ℕ := q * v

-- The theorem to be proven
theorem sara_total_cents : total_cents quarters value_per_quarter = 275 :=
by
  -- Proof goes here
  sorry

end sara_total_cents_l146_146638


namespace unoccupied_volume_eq_l146_146059

-- Define the radii and heights of the cones and cylinder
variable (r_cone : ℝ) (h_cone : ℝ) (h_cylinder : ℝ)
variable (r_cone_eq : r_cone = 10) (h_cone_eq : h_cone = 15) (h_cylinder_eq : h_cylinder = 30)

-- Define the volumes of the cylinder and the two cones
def volume_cylinder (r h : ℝ) : ℝ := π * r ^ 2 * h
def volume_cone (r h : ℝ) : ℝ := 1 / 3 * π * r ^ 2 * h
def volume_unoccupied : ℝ := volume_cylinder r_cone h_cylinder - 2 * volume_cone r_cone h_cone

-- Expression of the final result
theorem unoccupied_volume_eq : volume_unoccupied r_cone h_cone h_cylinder = 2000 * π :=
by
  rw [r_cone_eq, h_cone_eq, h_cylinder_eq]
  unfold volume_cylinder volume_cone volume_unoccupied
  norm_num

end unoccupied_volume_eq_l146_146059


namespace geometric_series_ratio_l146_146789

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l146_146789


namespace mary_saw_total_snakes_l146_146942

theorem mary_saw_total_snakes :
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  totalSnakes = 36 :=
by
  /- Definitions -/ 
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  /- Main proof statement -/
  show totalSnakes = 36
  sorry

end mary_saw_total_snakes_l146_146942


namespace total_seeds_l146_146186

theorem total_seeds (seeds_per_watermelon : ℕ) (number_of_watermelons : ℕ) 
(seeds_each : seeds_per_watermelon = 100)
(watermelons_count : number_of_watermelons = 4) :
(seeds_per_watermelon * number_of_watermelons) = 400 := by
  sorry

end total_seeds_l146_146186


namespace coordinates_of_AC_l146_146895

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p1.x - p2.x,
    y := p1.y - p2.y,
    z := p1.z - p2.z }

def scalar_mult (k : ℝ) (v : Point3D) : Point3D :=
  { x := k * v.x,
    y := k * v.y,
    z := k * v.z }

noncomputable def A : Point3D := { x := 1, y := 2, z := 3 }
noncomputable def B : Point3D := { x := 4, y := 5, z := 9 }

theorem coordinates_of_AC : vector_sub B A = { x := 3, y := 3, z := 6 } →
  scalar_mult (1 / 3) (vector_sub B A) = { x := 1, y := 1, z := 2 } :=
by
  sorry

end coordinates_of_AC_l146_146895


namespace smallest_possible_sum_S_l146_146773

theorem smallest_possible_sum_S (n : ℕ) (S : ℕ) :
  ∃ n S, (prob_sum n 2027 = prob_sum n S) → (S = 339 ∧ 6 * n ≥ 2027 ∧ n > 0) :=
begin
  use 338,
  use 339,
  sorry
end

end smallest_possible_sum_S_l146_146773


namespace find_a_l146_146569

theorem find_a (a : ℤ) (A B : Set ℤ) (hA : A = {0, 1}) (hB : B = {-1, 0, a + 3}) (h : A ⊆ B) : a = -2 := by
  sorry

end find_a_l146_146569


namespace original_number_increased_l146_146101

theorem original_number_increased (x : ℝ) (h : (1.10 * x) * 1.15 = 632.5) : x = 500 :=
sorry

end original_number_increased_l146_146101


namespace number_of_grade11_students_l146_146416

-- Define the total number of students in the high school.
def total_students : ℕ := 900

-- Define the total number of students selected in the sample.
def sample_students : ℕ := 45

-- Define the number of Grade 10 students in the sample.
def grade10_students_sample : ℕ := 20

-- Define the number of Grade 12 students in the sample.
def grade12_students_sample : ℕ := 10

-- Prove the number of Grade 11 students in the school is 300.
theorem number_of_grade11_students :
  (sample_students - grade10_students_sample - grade12_students_sample) * (total_students / sample_students) = 300 :=
by
  sorry

end number_of_grade11_students_l146_146416


namespace gcd_1855_1120_l146_146061

theorem gcd_1855_1120 : Int.gcd 1855 1120 = 35 :=
by
  sorry

end gcd_1855_1120_l146_146061


namespace more_pairs_B_than_A_l146_146662

theorem more_pairs_B_than_A :
    let pairs_per_box := 20
    let boxes_A := 8
    let pairs_A := boxes_A * pairs_per_box
    let pairs_B := 5 * pairs_A
    let more_pairs := pairs_B - pairs_A
    more_pairs = 640
:= by
    sorry

end more_pairs_B_than_A_l146_146662


namespace train_speed_kmph_l146_146104

noncomputable def train_speed_mps : ℝ := 60.0048

def conversion_factor : ℝ := 3.6

theorem train_speed_kmph : train_speed_mps * conversion_factor = 216.01728 := by
  sorry

end train_speed_kmph_l146_146104


namespace find_a_l146_146843

def possible_scores : List ℕ := [103, 104, 105, 106, 107, 108, 109, 110]

def is_possible_score (a : ℕ) (n : ℕ) : Prop :=
  ∃ (k8 k0 ka : ℕ), k8 * 8 + ka * a + k0 * 0 = n

def is_impossible_score (a : ℕ) (n : ℕ) : Prop :=
  ¬ is_possible_score a n

theorem find_a : ∀ (a : ℕ), a ≠ 0 → a ≠ 8 →
  (∀ n ∈ possible_scores, is_possible_score a n) →
  is_impossible_score a 83 →
  a = 13 := by
  intros a ha1 ha2 hpossible himpossible
  sorry

end find_a_l146_146843


namespace product_mnp_l146_146497

theorem product_mnp (a x y z c : ℕ) (m n p : ℕ) :
  (a ^ 8 * x * y * z - a ^ 7 * y * z - a ^ 6 * x * z = a ^ 5 * (c ^ 5 - 1) ∧
   (a ^ m * x * z - a ^ n) * (a ^ p * y * z - a ^ 3) = a ^ 5 * c ^ 5) →
  m = 5 ∧ n = 4 ∧ p = 3 ∧ m * n * p = 60 :=
by
  sorry

end product_mnp_l146_146497


namespace triangle_cos_area_l146_146166

/-- In triangle ABC, with angles A, B, and C, opposite sides a, b, and c respectively, given the condition 
    a * cos C = (2 * b - c) * cos A, prove: 
    1. cos A = 1/2
    2. If a = 6 and b + c = 8, then the area of triangle ABC is 7 * sqrt 3 / 3 --/
theorem triangle_cos_area (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos C = (2 * b - c) * Real.cos A)
  (h2 : a = 6) (h3 : b + c = 8) :
  Real.cos A = 1 / 2 ∧ ∃ area : ℝ, area = 7 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end triangle_cos_area_l146_146166


namespace cylinder_radius_eq_3_l146_146432

theorem cylinder_radius_eq_3 (r : ℝ) : 
  (π * (r + 4)^2 * 3 = π * r^2 * 11) ∧ (r >= 0) → r = 3 :=
by 
  sorry

end cylinder_radius_eq_3_l146_146432


namespace min_value_4a2_b2_plus_1_div_2a_minus_b_l146_146712

variable (a b : ℝ)

theorem min_value_4a2_b2_plus_1_div_2a_minus_b (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a > b) (h4 : a * b = 1 / 2) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 3 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x > y → x * y = 1 / 2 → (4 * x^2 + y^2 + 1) / (2 * x - y) ≥ c) :=
sorry

end min_value_4a2_b2_plus_1_div_2a_minus_b_l146_146712


namespace pow_mod_eq_l146_146391

theorem pow_mod_eq (n : ℕ) : 
  (3^n % 5 = 3 % 5) → 
  (3^(n+1) % 5 = (3 * 3^n) % 5) → 
  (3^(n+2) % 5 = (3 * 3^(n+1)) % 5) → 
  (3^(n+3) % 5 = (3 * 3^(n+2)) % 5) → 
  (3^4 % 5 = 1 % 5) → 
  (2023 % 4 = 3) → 
  (3^2023 % 5 = 2 % 5) :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pow_mod_eq_l146_146391


namespace largest_base_5_five_digits_base_10_value_l146_146084

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l146_146084


namespace common_ratio_of_geometric_series_l146_146808

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l146_146808


namespace problems_finished_equals_45_l146_146535

/-- Mathematical constants and conditions -/
def ratio_finished_left (F L : ℕ) : Prop := F = 9 * (L / 4)
def total_problems (F L : ℕ) : Prop := F + L = 65

/-- Lean theorem to prove the problem statement -/
theorem problems_finished_equals_45 :
  ∃ F L : ℕ, ratio_finished_left F L ∧ total_problems F L ∧ F = 45 :=
by
  sorry

end problems_finished_equals_45_l146_146535


namespace range_of_a_l146_146201

theorem range_of_a (a : ℝ) : 
  ( ∃ x y : ℝ, (x^2 + 4 * (y - a)^2 = 4) ∧ (x^2 = 4 * y)) ↔ a ∈ Set.Ico (-1 : ℝ) (5 / 4 : ℝ) := 
sorry

end range_of_a_l146_146201


namespace Bruce_grape_purchase_l146_146426

theorem Bruce_grape_purchase
  (G : ℕ)
  (total_paid : ℕ)
  (cost_per_kg_grapes : ℕ)
  (kg_mangoes : ℕ)
  (cost_per_kg_mangoes : ℕ)
  (total_mango_cost : ℕ)
  (total_grape_cost : ℕ)
  (total_amount : ℕ)
  (h1 : cost_per_kg_grapes = 70)
  (h2 : kg_mangoes = 10)
  (h3 : cost_per_kg_mangoes = 55)
  (h4 : total_paid = 1110)
  (h5 : total_mango_cost = kg_mangoes * cost_per_kg_mangoes)
  (h6 : total_grape_cost = G * cost_per_kg_grapes)
  (h7 : total_amount = total_mango_cost + total_grape_cost)
  (h8 : total_amount = total_paid) :
  G = 8 := by
  sorry

end Bruce_grape_purchase_l146_146426


namespace impossible_circle_arrangement_l146_146922

theorem impossible_circle_arrangement :
  ¬ ∃ (arrangement : List ℕ), arrangement.length = 2017 ∧ (∀ (i : ℕ), 
    (17 : ℤ) ∣ (arrangement.nth_le i (by simp [*]) - arrangement.nth_le ((i + 1) % 2017) (by simp [*])) ∨ 
    (21 : ℤ) ∣ (arrangement.nth_le i (by simp [*]) - arrangement.nth_le ((i + 1) % 2017) (by simp [*]))) :=
sorry

end impossible_circle_arrangement_l146_146922


namespace f_at_1_is_neg7007_l146_146568

variable (a b c : ℝ)

def g (x : ℝ) := x^3 + a * x^2 + x + 10
def f (x : ℝ) := x^4 + x^3 + b * x^2 + 100 * x + c

theorem f_at_1_is_neg7007
  (a b c : ℝ)
  (h1 : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ g a (r1) = 0 ∧ g a (r2) = 0 ∧ g a (r3) = 0)
  (h2 : ∀ x, f x = 0 → g x = 0) :
  f 1 = -7007 := 
sorry

end f_at_1_is_neg7007_l146_146568


namespace largest_base_5_five_digits_base_10_value_l146_146086

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l146_146086


namespace Pam_current_balance_l146_146364

-- Given conditions as definitions
def initial_balance : ℕ := 400
def tripled_balance : ℕ := 3 * initial_balance
def current_balance : ℕ := tripled_balance - 250

-- The theorem to be proved
theorem Pam_current_balance : current_balance = 950 := by
  sorry

end Pam_current_balance_l146_146364


namespace calculate_sheep_l146_146846

-- Conditions as definitions
def cows : Nat := 24
def goats : Nat := 113
def total_animals_to_transport (groups size_per_group : Nat) : Nat := groups * size_per_group
def cows_and_goats (cows goats : Nat) : Nat := cows + goats

-- The problem statement: Calculate the number of sheep such that the total number of animals matches the target.
theorem calculate_sheep
  (groups : Nat) (size_per_group : Nat) (cows goats : Nat) (transportation_total animals_present : Nat) 
  (h1 : groups = 3) (h2 : size_per_group = 48) (h3 : cows = 24) (h4 : goats = 113) 
  (h5 : animals_present = cows + goats) (h6 : transportation_total = groups * size_per_group) :
  transportation_total - animals_present = 7 :=
by 
  -- To be proven 
  sorry

end calculate_sheep_l146_146846


namespace common_ratio_of_geometric_series_l146_146791

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l146_146791


namespace common_ratio_of_geometric_series_l146_146807

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l146_146807


namespace remainder_7n_mod_4_l146_146837

theorem remainder_7n_mod_4 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := sorry

end remainder_7n_mod_4_l146_146837


namespace students_in_class_l146_146215

theorem students_in_class (n : ℕ) (S : ℕ) (h_avg_students : S / n = 14) (h_avg_including_teacher : (S + 45) / (n + 1) = 15) : n = 30 :=
by
  sorry

end students_in_class_l146_146215


namespace find_g_inv_neg_fifteen_sixtyfour_l146_146148

noncomputable def g (x : ℝ) : ℝ := (x^6 - 1) / 4

theorem find_g_inv_neg_fifteen_sixtyfour : g⁻¹ (-15/64) = 1/2 :=
by
  sorry  -- Proof is not required

end find_g_inv_neg_fifteen_sixtyfour_l146_146148


namespace probability_of_black_given_not_white_l146_146816

variable (total_balls white_balls black_balls red_balls : ℕ)
variable (ball_is_not_white : Prop)

theorem probability_of_black_given_not_white 
  (h1 : total_balls = 10)
  (h2 : white_balls = 5)
  (h3 : black_balls = 3)
  (h4 : red_balls = 2)
  (h5 : ball_is_not_white) :
  (3 : ℚ) / 5 = (black_balls : ℚ) / (total_balls - white_balls) :=
by
  simp only [h1, h2, h3, h4]
  sorry

end probability_of_black_given_not_white_l146_146816


namespace min_value_ineq_inequality_proof_l146_146710

variable (a b x1 x2 : ℝ)
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hx1_pos : 0 < x1) (hx2_pos : 0 < x2) (hab_sum : a + b = 1)

-- First problem: Prove that the minimum value of the given expression is 6.
theorem min_value_ineq : (x1 / a) + (x2 / b) + (2 / (x1 * x2)) ≥ 6 := by
  sorry

-- Second problem: Prove the given inequality.
theorem inequality_proof : (a * x1 + b * x2) * (a * x2 + b * x1) ≥ x1 * x2 := by
  sorry

end min_value_ineq_inequality_proof_l146_146710


namespace class_a_winning_probability_best_of_three_l146_146695

theorem class_a_winning_probability_best_of_three :
  let p := (3 : ℚ) / 5
  let win_first_two := p * p
  let win_first_and_third := p * ((1 - p) * p)
  let win_last_two := (1 - p) * (p * p)
  p * p + p * ((1 - p) * p) + (1 - p) * (p * p) = 81 / 125 :=
by
  sorry

end class_a_winning_probability_best_of_three_l146_146695


namespace total_dog_food_needed_per_day_l146_146187

theorem total_dog_food_needed_per_day :
  let weights := [20, 40, 10, 30, 50]
  let food_per_pound := 1 / 10
  let food_needed := list.sum (weights.map (λ w, w * food_per_pound))
  food_needed = 15 :=
by
  let weights := [20, 40, 10, 30, 50]
  let food_per_pound := 1 / 10
  let food_needed := list.sum (weights.map (λ w, w * food_per_pound))
  have : food_needed = 2 + 4 + 1 + 3 + 5, by sorry
  have : 2 + 4 + 1 + 3 + 5 = 15, by sorry
  exact this.trans this

end total_dog_food_needed_per_day_l146_146187


namespace line_param_func_l146_146050

theorem line_param_func (t : ℝ) : 
    ∃ f : ℝ → ℝ, (∀ t, (20 * t - 14) = 2 * (f t) - 30) ∧ (f t = 10 * t + 8) := by
  sorry

end line_param_func_l146_146050


namespace bacon_suggestion_l146_146642

theorem bacon_suggestion (x y : ℕ) (h1 : x = 479) (h2 : y = x + 10) : y = 489 := 
by {
  sorry
}

end bacon_suggestion_l146_146642


namespace find_m_l146_146204

theorem find_m {m : ℝ} :
  (4 - m) / (m + 2) = 1 → m = 1 :=
by
  sorry

end find_m_l146_146204


namespace cannon_hit_probability_l146_146525

theorem cannon_hit_probability
  (P1 P2 P3 : ℝ)
  (h1 : P1 = 0.2)
  (h3 : P3 = 0.3)
  (h_none_hit : (1 - P1) * (1 - P2) * (1 - P3) = 0.27999999999999997) :
  P2 = 0.5 :=
by
  sorry

end cannon_hit_probability_l146_146525


namespace binomial_coefficient_sum_l146_146878

theorem binomial_coefficient_sum :
  Nat.choose 10 3 + Nat.choose 10 2 = 165 := by
  sorry

end binomial_coefficient_sum_l146_146878


namespace max_abs_sum_l146_146581

theorem max_abs_sum (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_l146_146581


namespace E_runs_is_20_l146_146295

-- Definitions of runs scored by each batsman as multiples of 4
def a := 28
def e := 20
def d := e + 12
def b := d + e
def c := 107 - b
def total_runs := a + b + c + d + e

-- Adding conditions
axiom A_max: a > b ∧ a > c ∧ a > d ∧ a > e
axiom runs_multiple_of_4: ∀ (x : ℕ), x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e → x % 4 = 0
axiom average_runs: total_runs = 180
axiom d_condition: d = e + 12
axiom e_condition: e = a - 8
axiom b_condition: b = d + e
axiom bc_condition: b + c = 107

theorem E_runs_is_20 : e = 20 := by
  sorry

end E_runs_is_20_l146_146295


namespace remainder_invariance_l146_146355

theorem remainder_invariance (S A K : ℤ) (h : ∃ B r : ℤ, S = A * B + r ∧ 0 ≤ r ∧ r < |A|) :
  (∃ B' r' : ℤ, S + A * K = A * B' + r' ∧ r' = r) ∧ (∃ B'' r'' : ℤ, S - A * K = A * B'' + r'' ∧ r'' = r) :=
by
  sorry

end remainder_invariance_l146_146355


namespace remove_five_yields_average_10_5_l146_146550

def numberList : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def averageRemaining (l : List ℕ) : ℚ :=
  (List.sum l : ℚ) / l.length

theorem remove_five_yields_average_10_5 :
  averageRemaining (numberList.erase 5) = 10.5 :=
sorry

end remove_five_yields_average_10_5_l146_146550


namespace ratio_of_areas_l146_146370

theorem ratio_of_areas (side_length : ℝ) (num_corrals : ℕ)
  (corral_perimeter : ℝ) (total_fencing : ℝ)
  (large_corral_side_length : ℝ) (small_corral_area : ℝ) (total_small_corrals_area : ℝ) (large_corral_area : ℝ) :
  side_length = 10 →
  num_corrals = 6 →
  corral_perimeter = 3 * side_length →
  total_fencing = num_corrals * corral_perimeter →
  large_corral_side_length = total_fencing / 3 →
  small_corral_area = (sqrt 3 / 4) * side_length^2 →
  total_small_corrals_area = num_corrals * small_corral_area →
  large_corral_area = (sqrt 3 / 4) * large_corral_side_length^2 →
  total_small_corrals_area / large_corral_area = 1 / 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end ratio_of_areas_l146_146370


namespace molecular_weight_ammonia_l146_146872

def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.008
def count_N : ℕ := 1
def count_H : ℕ := 3

theorem molecular_weight_ammonia :
  (count_N * atomic_weight_N) + (count_H * atomic_weight_H) = 17.034 :=
by
  sorry

end molecular_weight_ammonia_l146_146872


namespace band_total_earnings_l146_146233

variables (earnings_per_gig_per_member : ℕ)
variables (number_of_members : ℕ)
variables (number_of_gigs : ℕ)

theorem band_total_earnings :
  earnings_per_gig_per_member = 20 →
  number_of_members = 4 →
  number_of_gigs = 5 →
  earnings_per_gig_per_member * number_of_members * number_of_gigs = 400 :=
by
  intros
  sorry

end band_total_earnings_l146_146233


namespace probability_diff_suits_l146_146463

theorem probability_diff_suits (n : ℕ) (h₁ : n = 65) (suits : ℕ) (h₂ : suits = 5) (cards_per_suit : ℕ) (h₃ : cards_per_suit = n / suits) : 
  (52 : ℚ) / (64 : ℚ) = (13 : ℚ) / (16 : ℚ) := 
by 
  sorry

end probability_diff_suits_l146_146463


namespace average_of_consecutive_odds_is_24_l146_146363

theorem average_of_consecutive_odds_is_24 (a b c d : ℤ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : d = 27) 
  (h5 : b = d - 2) (h6 : c = d - 4) (h7 : a = d - 6) 
  (h8 : ∀ x : ℤ, x % 2 = 1) :
  ((a + b + c + d) / 4) = 24 :=
by {
  sorry
}

end average_of_consecutive_odds_is_24_l146_146363


namespace complement_of_65_degrees_l146_146290

def angle_complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_65_degrees : angle_complement 65 = 25 := by
  -- Proof would follow here, but it's omitted since 'sorry' is added.
  sorry

end complement_of_65_degrees_l146_146290


namespace candy_totals_l146_146769

-- Definitions of the conditions
def sandra_bags := 2
def sandra_pieces_per_bag := 6

def roger_bags1 := 11
def roger_bags2 := 3

def emily_bags1 := 4
def emily_bags2 := 7
def emily_bags3 := 5

-- Definitions of total pieces of candy
def sandra_total_candy := sandra_bags * sandra_pieces_per_bag
def roger_total_candy := roger_bags1 + roger_bags2
def emily_total_candy := emily_bags1 + emily_bags2 + emily_bags3

-- The proof statement
theorem candy_totals :
  sandra_total_candy = 12 ∧ roger_total_candy = 14 ∧ emily_total_candy = 16 :=
by
  -- Here we would provide the proof but we'll use sorry to skip it
  sorry

end candy_totals_l146_146769


namespace opposite_of_negative_fraction_l146_146051

theorem opposite_of_negative_fraction :
  -(-1 / 2023) = (1 / 2023) :=
by
  sorry

end opposite_of_negative_fraction_l146_146051


namespace flour_quantity_l146_146412

-- Define the recipe ratio of eggs to flour
def recipe_ratio : ℚ := 3 / 2

-- Define the number of eggs needed
def eggs_needed := 9

-- Prove that the number of cups of flour needed is 6
theorem flour_quantity (r : ℚ) (n : ℕ) (F : ℕ) 
  (hr : r = 3 / 2) (hn : n = 9) : F = 6 :=
by
  sorry

end flour_quantity_l146_146412


namespace fraction_relation_l146_146937

-- Definitions for arithmetic sequences and their sums
noncomputable def a_n (a₁ d₁ n : ℕ) := a₁ + (n - 1) * d₁
noncomputable def b_n (b₁ d₂ n : ℕ) := b₁ + (n - 1) * d₂

noncomputable def A_n (a₁ d₁ n : ℕ) := n * a₁ + n * (n - 1) * d₁ / 2
noncomputable def B_n (b₁ d₂ n : ℕ) := n * b₁ + n * (n - 1) * d₂ / 2

-- Theorem statement
theorem fraction_relation (a₁ d₁ b₁ d₂ : ℕ) (h : ∀ n : ℕ, B_n a₁ d₁ n ≠ 0 → A_n a₁ d₁ n / B_n b₁ d₂ n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, b_n b₁ d₂ n ≠ 0 → a_n a₁ d₁ n / b_n b₁ d₂ n = (4 * n - 3) / (6 * n - 2) :=
sorry

end fraction_relation_l146_146937


namespace largest_base5_number_conversion_l146_146071

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l146_146071


namespace sum_of_circle_areas_l146_146985

theorem sum_of_circle_areas (r s t : ℝ) (h1 : r + s = 5) (h2 : s + t = 12) (h3 : r + t = 13) :
  real.pi * (r^2 + s^2 + t^2) = 113 * real.pi :=
by
  sorry

end sum_of_circle_areas_l146_146985


namespace craig_age_l146_146258

theorem craig_age (C M : ℕ) (h1 : C = M - 24) (h2 : C + M = 56) : C = 16 := 
by
  sorry

end craig_age_l146_146258


namespace how_many_cubes_needed_l146_146576

def cube_volume (side_len : ℕ) : ℕ :=
  side_len ^ 3

theorem how_many_cubes_needed (Vsmall Vlarge Vsmall_cube num_small_cubes : ℕ) 
  (h1 : Vsmall = cube_volume 8) 
  (h2 : Vlarge = cube_volume 12) 
  (h3 : Vsmall_cube = cube_volume 2) 
  (h4 : num_small_cubes = (Vlarge - Vsmall) / Vsmall_cube) :
  num_small_cubes = 152 :=
by
  sorry

end how_many_cubes_needed_l146_146576


namespace largest_integer_satisfying_conditions_l146_146437

theorem largest_integer_satisfying_conditions (n : ℤ) (m : ℤ) :
  n^2 = (m + 1)^3 - m^3 ∧ ∃ k : ℤ, 2 * n + 103 = k^2 → n = 313 := 
by 
  sorry

end largest_integer_satisfying_conditions_l146_146437


namespace inequality_solution_eq_l146_146708

theorem inequality_solution_eq :
  ∀ y : ℝ, 2 ≤ |y - 5| ∧ |y - 5| ≤ 8 ↔ (-3 ≤ y ∧ y ≤ 3) ∨ (7 ≤ y ∧ y ≤ 13) :=
by
  sorry

end inequality_solution_eq_l146_146708


namespace min_value_arith_geom_seq_l146_146251

theorem min_value_arith_geom_seq (x y a1 a2 b1 b2 : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h_arith : x + y = 2 * a1) (h_geom : x * y = b1 * b2) :
  (x = y) → (a_1 + a_2) / (real.sqrt (b_1 * b_2)) = 2 :=
begin
  sorry
end

end min_value_arith_geom_seq_l146_146251


namespace mixture_percentage_l146_146673

variable (P : ℝ)
variable (x_ryegrass_percent : ℝ := 0.40)
variable (y_ryegrass_percent : ℝ := 0.25)
variable (final_mixture_ryegrass_percent : ℝ := 0.32)

theorem mixture_percentage (h : 0.40 * P + 0.25 * (1 - P) = 0.32) : P = 0.07 / 0.15 := by
  sorry

end mixture_percentage_l146_146673


namespace train_crossing_time_l146_146536

theorem train_crossing_time
    (train_speed_kmph : ℕ)
    (platform_length_meters : ℕ)
    (crossing_time_platform_seconds : ℕ)
    (crossing_time_man_seconds : ℕ)
    (train_speed_mps : ℤ)
    (train_length_meters : ℤ)
    (T : ℤ)
    (h1 : train_speed_kmph = 72)
    (h2 : platform_length_meters = 340)
    (h3 : crossing_time_platform_seconds = 35)
    (h4 : train_speed_mps = 20)
    (h5 : train_length_meters = 360)
    (h6 : train_length_meters = train_speed_mps * crossing_time_man_seconds)
    : T = 18 :=
by
  sorry

end train_crossing_time_l146_146536


namespace main_theorem_l146_146547

open Nat

-- Define the conditions
def conditions (p q n : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Odd p ∧ Odd q ∧ n > 1 ∧
  (q^(n+2) % p^n = 3^(n+2) % p^n) ∧ (p^(n+2) % q^n = 3^(n+2) % q^n)

-- Define the conclusion
def conclusion (p q n : ℕ) : Prop :=
  (p = 3 ∧ q = 3)

-- Define the main problem
theorem main_theorem : ∀ p q n : ℕ, conditions p q n → conclusion p q n :=
  by
    intros p q n h
    sorry

end main_theorem_l146_146547


namespace garden_area_l146_146529

theorem garden_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end garden_area_l146_146529


namespace complement_M_in_U_l146_146024

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_M_in_U :
  U \ M = {3, 5, 6} :=
by sorry

end complement_M_in_U_l146_146024


namespace triangle_inequality_l146_146961

variable {a b c S n : ℝ}

theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
(habc : a + b > c) (habc' : a + c > b) (habc'' : b + c > a)
(hS : 2 * S = a + b + c) (hn : n ≥ 1) :
  (a^n / (b + c)) + (b^n / (c + a)) + (c^n / (a + b)) ≥ ((2 / 3)^(n - 2)) * S^(n - 1) :=
by
  sorry

end triangle_inequality_l146_146961


namespace brenda_initial_points_l146_146881

theorem brenda_initial_points
  (b : ℕ)  -- points scored by Brenda in her play
  (initial_advantage :ℕ := 22)  -- Brenda is initially 22 points ahead
  (david_score : ℕ := 32)  -- David scores 32 points
  (final_advantage : ℕ := 5)  -- Brenda is 5 points ahead after both plays
  (h : initial_advantage + b - david_score = final_advantage) :
  b = 15 :=
by
  sorry

end brenda_initial_points_l146_146881


namespace fraction_girls_at_meet_l146_146660

-- Define the conditions of the problem
def numStudentsMaplewood : ℕ := 300
def ratioBoysGirlsMaplewood : ℕ × ℕ := (3, 2)
def numStudentsRiverview : ℕ := 240
def ratioBoysGirlsRiverview : ℕ × ℕ := (3, 5)

-- Define the combined number of students and number of girls
def totalStudentsMaplewood := numStudentsMaplewood
def totalStudentsRiverview := numStudentsRiverview

def numGirlsMaplewood : ℕ :=
  let (b, g) := ratioBoysGirlsMaplewood
  (totalStudentsMaplewood * g) / (b + g)

def numGirlsRiverview : ℕ :=
  let (b, g) := ratioBoysGirlsRiverview
  (totalStudentsRiverview * g) / (b + g)

def totalGirls := numGirlsMaplewood + numGirlsRiverview
def totalStudents := totalStudentsMaplewood + totalStudentsRiverview

-- Formalize the actual proof statement
theorem fraction_girls_at_meet : 
  (totalGirls : ℚ) / totalStudents = 1 / 2 := by
  sorry

end fraction_girls_at_meet_l146_146660


namespace molecular_weight_compound_l146_146510

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def molecular_weight (n_H n_Br n_O : ℕ) : ℝ :=
  n_H * atomic_weight_H + n_Br * atomic_weight_Br + n_O * atomic_weight_O

theorem molecular_weight_compound : 
  molecular_weight 1 1 3 = 128.91 :=
by
  -- This is where the proof would go
  sorry

end molecular_weight_compound_l146_146510


namespace hypotenuse_length_l146_146311

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l146_146311


namespace max_sum_of_arithmetic_sequence_l146_146444

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ)
  (d : ℤ) (h_a : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 2 + a 3 = 156)
  (h2 : a 2 + a 3 + a 4 = 147) :
  ∃ n : ℕ, n = 19 ∧ (∀ m : ℕ, S m ≤ S n) :=
sorry

end max_sum_of_arithmetic_sequence_l146_146444


namespace num_dress_designs_l146_146527

-- Define the number of fabric colors and patterns
def fabric_colors : ℕ := 4
def patterns : ℕ := 5

-- Define the number of possible dress designs
def total_dress_designs : ℕ := fabric_colors * patterns

-- State the theorem that needs to be proved
theorem num_dress_designs : total_dress_designs = 20 := by
  sorry

end num_dress_designs_l146_146527


namespace find_a22_l146_146974

-- Definitions and conditions
noncomputable def seq (n : ℕ) : ℝ := if n = 0 then 0 else sorry

axiom seq_conditions
  (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) : True

theorem find_a22 (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 :=
sorry

end find_a22_l146_146974


namespace number_of_true_propositions_is_three_l146_146545

-- Define each proposition as a Lean term
def proposition1 : Prop := ¬((∃ x : ℝ, log x = 0) → (x = 1)) ↔ (¬∃ x : ℝ, log x = 0 → x ≠ 1)
def proposition2 : Prop := (¬(True ∧ True)) → (True ∧ False)
def proposition3 : Prop := (∃ x : ℝ, sin x > 1) ↔ (¬∀ x : ℝ, sin x ≤ 1)
def proposition4 : Prop := ∀ x : ℝ, (x > 2 → 1 / x < 1 / 2) ∧ (¬(2 > x → 1 / x < 1 / 2))

-- Define the overall proof problem
theorem number_of_true_propositions_is_three : 
  (nat.succ (nat.succ (nat.succ nat.zero))) = 
  (CondCount [proposition1, proposition2, proposition3, proposition4] (λ x, x = true)) :-
begin
  -- The proof will be skipped
  sorry
end

end number_of_true_propositions_is_three_l146_146545


namespace AmpersandDoubleCalculation_l146_146561

def ampersand (x : Int) : Int := 7 - x
def doubleAmpersand (x : Int) : Int := (x - 7)

theorem AmpersandDoubleCalculation : doubleAmpersand (ampersand 12) = -12 :=
by
  -- This is where the proof would go, which shows the steps described in the solution.
  sorry

end AmpersandDoubleCalculation_l146_146561


namespace value_of_4_and_2_l146_146654

noncomputable def custom_and (a b : ℕ) : ℕ :=
  ((a + b) * (a - b)) ^ 2

theorem value_of_4_and_2 : custom_and 4 2 = 144 :=
  sorry

end value_of_4_and_2_l146_146654


namespace hash_op_example_l146_146180

def hash_op (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem hash_op_example : hash_op 2 5 3 = 1 := 
by 
  sorry

end hash_op_example_l146_146180


namespace three_pow_124_mod_7_l146_146826

theorem three_pow_124_mod_7 : (3^124) % 7 = 4 := by
  sorry

end three_pow_124_mod_7_l146_146826


namespace cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_l146_146229

-- Part (a)
theorem cupSaucersCombination :
  (5 : ℕ) * (3 : ℕ) = 15 :=
by
  -- Proof goes here
  sorry

-- Part (b)
theorem cupSaucerSpoonCombination :
  (5 : ℕ) * (3 : ℕ) * (4 : ℕ) = 60 :=
by
  -- Proof goes here
  sorry

-- Part (c)
theorem twoDifferentItemsCombination :
  (5 * 3 + 5 * 4 + 3 * 4 : ℕ) = 47 :=
by
  -- Proof goes here
  sorry

end cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_l146_146229


namespace hypotenuse_length_l146_146313

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l146_146313


namespace sin_C_value_area_of_triangle_l146_146611

open Real
open Classical

variable {A B C a b c : ℝ}

-- Given conditions
axiom h1 : b = sqrt 2
axiom h2 : c = 1
axiom h3 : cos B = 3 / 4

-- Proof statements
theorem sin_C_value : sin C = sqrt 14 / 8 := sorry

theorem area_of_triangle : 1 / 2 * b * c * sin (B + C) = sqrt 7 / 4 := sorry

end sin_C_value_area_of_triangle_l146_146611


namespace factorize_polynomial_l146_146517

noncomputable def polynomial_factorization : Prop :=
  ∀ x : ℤ, (x^12 + x^9 + 1) = (x^4 + x^3 + x^2 + x + 1) * (x^8 - x^7 + x^6 - x^5 + x^3 - x^2 + x - 1)

theorem factorize_polynomial : polynomial_factorization :=
by
  sorry

end factorize_polynomial_l146_146517


namespace sum_of_first_49_primes_l146_146270

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                                   61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 
                                   137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 
                                   199, 211, 223, 227]

theorem sum_of_first_49_primes : first_49_primes.sum = 10787 :=
by
  -- Proof to be filled in
  sorry

end sum_of_first_49_primes_l146_146270


namespace parabola_directrix_eq_l146_146202

def parabola_directrix (p : ℝ) : ℝ := -p

theorem parabola_directrix_eq (x y p : ℝ) (h : y ^ 2 = 8 * x) (hp : 2 * p = 8) : 
  parabola_directrix p = -2 :=
by
  sorry

end parabola_directrix_eq_l146_146202


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l146_146306

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l146_146306


namespace pentagon_rectangle_ratio_l146_146241

theorem pentagon_rectangle_ratio :
  ∀ (p w l : ℝ), 
  5 * p = 20 → 
  2 * (w + l) = 20 →
  l = 2 * w →
  p / w = 6 / 5 :=
by
  intros p w l h₁ h₂ h₃
  have p_value : p = 4 := 
    by linarith
  have w_value : w = 10 / 3 := 
    by linarith
  rw [p_value, w_value]
  norm_num
  sorry

end pentagon_rectangle_ratio_l146_146241


namespace proposition_5_l146_146232

/-! 
  Proposition 5: If there are four points A, B, C, D in a plane, 
  then the vector addition relation: \overrightarrow{AC} + \overrightarrow{BD} = \overrightarrow{BC} + \overrightarrow{AD} must hold.
--/

variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables (AC BD BC AD : A)

-- Theorem Statement in Lean 4
theorem proposition_5 (AC BD BC AD : A)
  : AC + BD = BC + AD := by
  -- Proof by congruence and equality, will add actual steps here
  sorry

end proposition_5_l146_146232


namespace volume_ratio_l146_146418

noncomputable def V_D (s : ℝ) := (15 + 7 * Real.sqrt 5) * s^3 / 4
noncomputable def a (s : ℝ) := s / 2 * (1 + Real.sqrt 5)
noncomputable def V_I (a : ℝ) := 5 * (3 + Real.sqrt 5) * a^3 / 12

theorem volume_ratio (s : ℝ) (h₁ : 0 < s) :
  V_I (a s) / V_D s = (5 * (3 + Real.sqrt 5) * (1 + Real.sqrt 5)^3) / (12 * 2 * (15 + 7 * Real.sqrt 5)) :=
by
  sorry

end volume_ratio_l146_146418


namespace rational_sum_eq_one_l146_146917

theorem rational_sum_eq_one (a b : ℚ) (h : |3 - a| + (b + 2)^2 = 0) : a + b = 1 := 
by
  sorry

end rational_sum_eq_one_l146_146917


namespace functional_equation_holds_l146_146474

def f (p q : ℕ) : ℝ :=
  if p = 0 ∨ q = 0 then 0 else (p * q : ℝ)

theorem functional_equation_holds (p q : ℕ) : 
  f p q = 
    if p = 0 ∨ q = 0 then 0 
    else 1 + (1 / 2) * f (p + 1) (q - 1) + (1 / 2) * f (p - 1) (q + 1) :=
  by 
    sorry

end functional_equation_holds_l146_146474


namespace unique_toy_value_l146_146246

/-- Allie has 9 toys in total. The total worth of these toys is $52. 
One toy has a certain value "x" dollars and the remaining 8 toys each have a value of $5. 
Prove that the value of the unique toy is $12. -/
theorem unique_toy_value (x : ℕ) (h1 : 1 + 8 = 9) (h2 : x + 8 * 5 = 52) : x = 12 :=
by
  sorry

end unique_toy_value_l146_146246


namespace jesse_stamps_l146_146621

variable (A E : Nat)

theorem jesse_stamps :
  E = 3 * A ∧ E + A = 444 → E = 333 :=
by
  sorry

end jesse_stamps_l146_146621


namespace average_runs_l146_146043

/-- The average runs scored by the batsman in the first 20 matches is 40,
and in the next 10 matches is 30. We want to prove the average runs scored
by the batsman in all 30 matches is 36.67. --/
theorem average_runs (avg20 avg10 : ℕ) (num_matches_20 num_matches_10 : ℕ)
  (h1 : avg20 = 40) (h2 : avg10 = 30) (h3 : num_matches_20 = 20) (h4 : num_matches_10 = 10) :
  ((num_matches_20 * avg20 + num_matches_10 * avg10 : ℕ) : ℚ) / (num_matches_20 + num_matches_10 : ℕ) = 36.67 := by
  sorry

end average_runs_l146_146043


namespace find_m_l146_146142

open Real

noncomputable def x_values : List ℝ := [1, 3, 4, 5, 7]
noncomputable def y_values (m : ℝ) : List ℝ := [1, m, 2 * m + 1, 2 * m + 3, 10]

noncomputable def mean (l : List ℝ) : ℝ :=
l.sum / l.length

theorem find_m (m : ℝ) :
  mean x_values = 4 →
  mean (y_values m) = m + 3 →
  (1.3 * 4 + 0.8 = m + 3) →
  m = 3 :=
by
  intros h1 h2 h3
  sorry

end find_m_l146_146142


namespace david_produces_8_more_widgets_l146_146030

variable (w t : ℝ)

def widgets_monday (w t : ℝ) : ℝ :=
  w * t

def widgets_tuesday (w t : ℝ) : ℝ :=
  (w + 4) * (t - 2)

theorem david_produces_8_more_widgets (h : w = 2 * t) : 
  widgets_monday w t - widgets_tuesday w t = 8 :=
by
  sorry

end david_produces_8_more_widgets_l146_146030


namespace polynomial_expansion_a6_l146_146753

theorem polynomial_expansion_a6 :
  let p := x^2 + x^7
  ∃ (a : ℕ → ℝ), p = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 ∧ a 6 = -7 := 
sorry

end polynomial_expansion_a6_l146_146753


namespace comic_story_books_proportion_l146_146467

theorem comic_story_books_proportion (x : ℕ) :
  let initial_comic_books := 140
  let initial_story_books := 100
  let borrowed_books_per_day := 4
  let comic_books_after_x_days := initial_comic_books - borrowed_books_per_day * x
  let story_books_after_x_days := initial_story_books - borrowed_books_per_day * x
  (comic_books_after_x_days = 3 * story_books_after_x_days) -> x = 20 :=
by
  sorry

end comic_story_books_proportion_l146_146467


namespace depth_of_channel_l146_146776

theorem depth_of_channel (a b A : ℝ) (h : ℝ) (h_area : A = (1 / 2) * (a + b) * h)
  (ha : a = 12) (hb : b = 6) (hA : A = 630) : h = 70 :=
by
  sorry

end depth_of_channel_l146_146776


namespace inversely_proportional_y_value_l146_146500

theorem inversely_proportional_y_value (x y k : ℝ)
  (h1 : ∀ x y : ℝ, x * y = k)
  (h2 : ∃ y : ℝ, x = 3 * y ∧ x + y = 36 ∧ x * y = k)
  (h3 : x = -9) : y = -27 := 
by
  sorry

end inversely_proportional_y_value_l146_146500


namespace apples_per_friend_l146_146112

def Benny_apples : Nat := 5
def Dan_apples : Nat := 2 * Benny_apples
def Total_apples : Nat := Benny_apples + Dan_apples
def Number_of_friends : Nat := 3

theorem apples_per_friend : Total_apples / Number_of_friends = 5 := by
  sorry

end apples_per_friend_l146_146112


namespace remainder_7n_mod_4_l146_146835

theorem remainder_7n_mod_4 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := sorry

end remainder_7n_mod_4_l146_146835


namespace b_bound_for_tangent_parallel_l146_146718

theorem b_bound_for_tangent_parallel (b : ℝ) (c : ℝ) :
  (∃ x : ℝ, 3 * x^2 - x + b = 0) → b ≤ 1/12 :=
by
  intros h
  -- Placeholder proof
  sorry

end b_bound_for_tangent_parallel_l146_146718


namespace find_a4_l146_146564

-- Given expression of x^5
def polynomial_expansion (x a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  x^5 = a_0 + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5

theorem find_a4 (x a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (h : polynomial_expansion x a_0 a_1 a_2 a_3 a_4 a_5) : a_4 = -5 :=
  sorry

end find_a4_l146_146564


namespace cost_of_apples_and_oranges_correct_l146_146168

-- Define the initial money jasmine had
def initial_money : ℝ := 100.00

-- Define the remaining money after purchase
def remaining_money : ℝ := 85.00

-- Define the cost of apples and oranges
def cost_of_apples_and_oranges : ℝ := initial_money - remaining_money

-- This is our theorem statement that needs to be proven
theorem cost_of_apples_and_oranges_correct :
  cost_of_apples_and_oranges = 15.00 :=
by
  sorry

end cost_of_apples_and_oranges_correct_l146_146168


namespace maria_total_eggs_l146_146757

def total_eggs (boxes : ℕ) (eggs_per_box : ℕ) : ℕ :=
  boxes * eggs_per_box

theorem maria_total_eggs :
  total_eggs 3 7 = 21 :=
by
  -- Here, you would normally show the steps of computation
  -- which we can skip with sorry
  sorry

end maria_total_eggs_l146_146757


namespace find_triples_l146_146704

theorem find_triples (a m n : ℕ) (k : ℕ):
  a ≥ 2 ∧ m ≥ 2 ∧ a^n + 203 ≡ 0 [MOD a^m + 1] ↔ 
  (a = 2 ∧ ((n = 4 * k + 1 ∧ m = 2) ∨ (n = 6 * k + 2 ∧ m = 3) ∨ (n = 8 * k + 8 ∧ m = 4) ∨ (n = 12 * k + 9 ∧ m = 6))) ∨
  (a = 3 ∧ n = 4 * k + 3 ∧ m = 2) ∨
  (a = 4 ∧ n = 4 * k + 4 ∧ m = 2) ∨
  (a = 5 ∧ n = 4 * k + 1 ∧ m = 2) ∨
  (a = 8 ∧ n = 4 * k + 3 ∧ m = 2) ∨
  (a = 10 ∧ n = 4 * k + 2 ∧ m = 2) ∨
  (a = 203 ∧ n = (2 * k + 1) * m + 1 ∧ m ≥ 2) := by sorry

end find_triples_l146_146704


namespace quadratic_form_h_l146_146592

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l146_146592


namespace tan_A_tan_B_l146_146501

theorem tan_A_tan_B (A B C : ℝ) (R : ℝ) (H F : ℝ)
  (HF : H + F = 26) (h1 : 2 * R * Real.cos A * Real.cos B = 8)
  (h2 : 2 * R * Real.sin A * Real.sin B = 26) :
  Real.tan A * Real.tan B = 13 / 4 :=
by
  sorry

end tan_A_tan_B_l146_146501


namespace car_time_passed_l146_146234

variable (speed : ℝ) (distance : ℝ) (time_passed : ℝ)

theorem car_time_passed (h_speed : speed = 2) (h_distance : distance = 2) :
  time_passed = distance / speed := by
  rw [h_speed, h_distance]
  norm_num
  sorry

end car_time_passed_l146_146234


namespace range_of_a_l146_146203

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
if h : x < 1 then a * x^2 - 6 * x + a^2 + 1 else x^(5 - 2 * a)

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (5/2 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l146_146203


namespace range_of_a_l146_146616

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

def sibling_point_pair (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  A.2 = f a A.1 ∧ B.2 = f a B.1 ∧ A.1 = -B.1 ∧ A.2 = -B.2

theorem range_of_a (a : ℝ) :
  (∃ A B : ℝ × ℝ, sibling_point_pair a A B) ↔ a > 1 :=
sorry

end range_of_a_l146_146616


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l146_146302

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l146_146302


namespace find_nth_number_in_s_l146_146023

def s (k : ℕ) : ℕ := 8 * k + 5

theorem find_nth_number_in_s (n : ℕ) (number_in_s : ℕ) (h : number_in_s = 573) :
  ∃ k : ℕ, s k = number_in_s ∧ n = k + 1 := 
sorry

end find_nth_number_in_s_l146_146023


namespace find_m_l146_146645

theorem find_m :
  ∃ m : ℕ, 264 * 391 % 100 = m ∧ 0 ≤ m ∧ m < 100 ∧ m = 24 :=
by
  sorry

end find_m_l146_146645


namespace XiaoMing_reading_problem_l146_146092

theorem XiaoMing_reading_problem :
  ∀ (total_pages days first_days first_rate remaining_rate : ℕ),
    total_pages = 72 →
    days = 10 →
    first_days = 2 →
    first_rate = 5 →
    (first_days * first_rate) + ((days - first_days) * remaining_rate) ≥ total_pages →
    remaining_rate ≥ 8 :=
by
  intros total_pages days first_days first_rate remaining_rate
  intro h1 h2 h3 h4 h5
  sorry

end XiaoMing_reading_problem_l146_146092


namespace problem_inequality_l146_146480

theorem problem_inequality (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_abc : a * b * c = 1) :
  (a - 1 + (1 / b)) * (b - 1 + (1 / c)) * (c - 1 + (1 / a)) ≤ 1 :=
sorry

end problem_inequality_l146_146480


namespace average_of_multiples_l146_146705

theorem average_of_multiples :
  let sum_of_first_7_multiples_of_9 := 9 + 18 + 27 + 36 + 45 + 54 + 63
  let sum_of_first_5_multiples_of_11 := 11 + 22 + 33 + 44 + 55
  let sum_of_first_3_negative_multiples_of_13 := -13 + -26 + -39
  let total_sum := sum_of_first_7_multiples_of_9 + sum_of_first_5_multiples_of_11 + sum_of_first_3_negative_multiples_of_13
  let average := total_sum / 3
  average = 113 :=
by
  sorry

end average_of_multiples_l146_146705


namespace veronica_initial_marbles_l146_146431

variable {D M P V : ℕ}

theorem veronica_initial_marbles (hD : D = 14) (hM : M = 20) (hP : P = 19)
  (h_total : D + M + P + V = 60) : V = 7 :=
by
  sorry

end veronica_initial_marbles_l146_146431


namespace greatest_b_value_l146_146496

def equation_has_integer_solutions (b : ℕ) : Prop :=
  ∃ (x : ℤ), x * (x + b) = -20

theorem greatest_b_value : ∃ (b : ℕ), b = 21 ∧ equation_has_integer_solutions b :=
by
  sorry

end greatest_b_value_l146_146496


namespace geometric_series_common_ratio_l146_146800

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l146_146800


namespace luncheon_cost_l146_146495

theorem luncheon_cost
  (s c p : ℝ)
  (h1 : 3 * s + 7 * c + p = 3.15)
  (h2 : 4 * s + 10 * c + p = 4.20) :
  s + c + p = 1.05 :=
by sorry

end luncheon_cost_l146_146495


namespace parking_space_area_l146_146519

theorem parking_space_area (L W : ℕ) (h1 : L = 9) (h2 : L + 2 * W = 37) : L * W = 126 := by
  sorry

end parking_space_area_l146_146519


namespace h_value_l146_146608

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l146_146608


namespace solution_set_l146_146278

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom monotone_increasing : ∀ x y, x < y → f x ≤ f y
axiom f_at_3 : f 3 = 2

-- Proof statement
theorem solution_set : {x : ℝ | -2 ≤ f (3 - x) ∧ f (3 - x) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 6} :=
by {
  sorry
}

end solution_set_l146_146278


namespace probability_rain_at_most_3_days_l146_146375

noncomputable theory

open ProbabilityTheory

def binomial_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem probability_rain_at_most_3_days :
  let p := (1 : ℝ) / 20
  let days := 62
  binomial_probability days 0 p +
  binomial_probability days 1 p +
  binomial_probability days 2 p +
  binomial_probability days 3 p ≈ 0.5383 :=
by
  sorry

end probability_rain_at_most_3_days_l146_146375


namespace certain_number_l146_146913

theorem certain_number (N : ℝ) (k : ℝ) 
  (h1 : (1 / 2) ^ 22 * N ^ k = 1 / 18 ^ 22) 
  (h2 : k = 11) 
  : N = 81 := 
by
  sorry

end certain_number_l146_146913


namespace stickers_total_l146_146397

theorem stickers_total (yesterday_packs : ℕ) (increment_packs : ℕ) (today_packs : ℕ) (total_packs : ℕ) :
  yesterday_packs = 15 → increment_packs = 10 → today_packs = yesterday_packs + increment_packs → total_packs = yesterday_packs + today_packs → total_packs = 40 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h1, h3] at h4
  exact h4

end stickers_total_l146_146397


namespace complete_the_square_h_value_l146_146586

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l146_146586


namespace smallest_n_terminating_decimal_l146_146828

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (n > 0) ∧ (∀ d : ℕ, (n + 110 = d) → (∀ p : ℕ, Prime p → p ∣ d → (p = 2 ∨ p = 5)) ∧ n = 15) :=
begin
  sorry
end

end smallest_n_terminating_decimal_l146_146828


namespace sum_of_prime_factors_143_l146_146997

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l146_146997


namespace sophomores_stratified_sampling_l146_146682

theorem sophomores_stratified_sampling 
  (total_students freshmen sophomores seniors selected_total : ℕ) 
  (H1 : total_students = 2800) 
  (H2 : freshmen = 970) 
  (H3 : sophomores = 930) 
  (H4 : seniors = 900) 
  (H_selected_total : selected_total = 280) : 
  (sophomores / total_students) * selected_total = 93 :=
by sorry

end sophomores_stratified_sampling_l146_146682


namespace ratio_of_volumes_l146_146898

theorem ratio_of_volumes (r : ℝ) (π : ℝ) (V1 V2 : ℝ) 
  (h1 : V2 = (4 / 3) * π * r^3) 
  (h2 : V1 = 2 * π * r^3) : 
  V1 / V2 = 3 / 2 :=
by
  sorry

end ratio_of_volumes_l146_146898


namespace sum_of_prime_factors_143_l146_146998

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l146_146998


namespace find_a_l146_146047

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a = 0 ↔ 3 * x^4 - 48 = 0) → a = 4 :=
  by
    intros h
    sorry

end find_a_l146_146047


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l146_146303

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l146_146303


namespace vincent_total_packs_l146_146399

noncomputable def total_packs (yesterday today_addition: ℕ) : ℕ :=
  let today := yesterday + today_addition
  yesterday + today

theorem vincent_total_packs
  (yesterday_packs : ℕ)
  (today_addition: ℕ)
  (hyesterday: yesterday_packs = 15)
  (htoday_addition: today_addition = 10) :
  total_packs yesterday_packs today_addition = 40 :=
by
  rw [hyesterday, htoday_addition]
  unfold total_packs
  -- at this point it simplifies to 15 + (15 + 10) = 40
  sorry

end vincent_total_packs_l146_146399


namespace quadratic_form_h_l146_146593

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l146_146593


namespace no_couples_next_to_each_other_l146_146329

def factorial (n: Nat): Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (m n p q: Nat): Nat :=
  factorial m - n * factorial (m - 1) + p * factorial (m - 2) - q * factorial (m - 3)

theorem no_couples_next_to_each_other :
  arrangements 7 8 24 32 + 16 * factorial 3 = 1488 :=
by
  -- Here we state that the calculation of special arrangements equals 1488.
  sorry

end no_couples_next_to_each_other_l146_146329


namespace rahim_sequence_final_value_l146_146541

theorem rahim_sequence_final_value :
  ∃ (a : ℕ) (b : ℕ), a ^ b = 5 ^ 16 :=
sorry

end rahim_sequence_final_value_l146_146541


namespace problem_solution_l146_146002

-- Define the conditions
variables {a c b d x y z q : Real}
axiom h1 : a^x = c^q ∧ c^q = b
axiom h2 : c^y = a^z ∧ a^z = d

-- State the theorem
theorem problem_solution : xy = zq :=
by
  sorry

end problem_solution_l146_146002


namespace correct_operation_l146_146091

variable (x y a : ℝ)

lemma correct_option_C :
  -4 * x^5 * y^3 / (2 * x^3 * y) = -2 * x^2 * y^2 :=
by sorry

lemma wrong_option_A :
  x * (2 * x + 3) ≠ 2 * x^2 + 3 :=
by sorry

lemma wrong_option_B :
  a^2 + a^3 ≠ a^5 :=
by sorry

lemma wrong_option_D :
  x^3 * x^2 ≠ x^6 :=
by sorry

theorem correct_operation :
  ((-4 * x^5 * y^3 / (2 * x^3 * y) = -2 * x^2 * y^2) ∧
   (x * (2 * x + 3) ≠ 2 * x^2 + 3) ∧
   (a^2 + a^3 ≠ a^5) ∧
   (x^3 * x^2 ≠ x^6)) :=
by
  exact ⟨correct_option_C x y, wrong_option_A x, wrong_option_B a, wrong_option_D x⟩

end correct_operation_l146_146091


namespace correct_option_l146_146394

-- Definitions of the options as Lean statements
def optionA : Prop := (-1 : ℝ) / 6 > (-1 : ℝ) / 7
def optionB : Prop := (-4 : ℝ) / 3 < (-3 : ℝ) / 2
def optionC : Prop := (-2 : ℝ)^3 = -2^3
def optionD : Prop := -(-4.5 : ℝ) > abs (-4.6 : ℝ)

-- Theorem stating that optionC is the correct statement among the provided options
theorem correct_option : optionC :=
by
  unfold optionC
  rw [neg_pow, neg_pow, pow_succ, pow_succ]
  sorry  -- The proof is omitted as per instructions

end correct_option_l146_146394


namespace common_ratio_of_geometric_series_l146_146790

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l146_146790


namespace gcd_45_75_105_l146_146666

theorem gcd_45_75_105 : Nat.gcd (45 : ℕ) (Nat.gcd 75 105) = 15 := 
by
  sorry

end gcd_45_75_105_l146_146666


namespace sum_of_radii_l146_146097

noncomputable def tangency_equation (r : ℝ) : Prop :=
  (r - 5)^2 + r^2 = (r + 1.5)^2

theorem sum_of_radii : ∀ (r1 r2 : ℝ), tangency_equation r1 ∧ tangency_equation r2 →
  r1 + r2 = 13 :=
by
  intros r1 r2 h
  sorry

end sum_of_radii_l146_146097


namespace train_length_proof_l146_146420

def speed_kmph : ℝ := 54
def time_seconds : ℝ := 54.995600351971845
def bridge_length_m : ℝ := 660
def train_length_approx : ℝ := 164.93

noncomputable def speed_m_s : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_m_s * time_seconds
noncomputable def train_length : ℝ := total_distance - bridge_length_m

theorem train_length_proof :
  abs (train_length - train_length_approx) < 0.01 :=
by
  sorry

end train_length_proof_l146_146420


namespace geometric_series_common_ratio_l146_146802

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l146_146802


namespace largest_base_5_five_digit_number_in_decimal_l146_146075

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l146_146075


namespace min_max_expression_l146_146477

variable (a b c d e : ℝ)

def expression (a b c d e : ℝ) : ℝ :=
  5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4)

theorem min_max_expression :
  a + b + c + d + e = 10 →
  a^2 + b^2 + c^2 + d^2 + e^2 = 20 →
  expression a b c d e = 120 := by
  sorry

end min_max_expression_l146_146477


namespace maximum_take_home_pay_l146_146163

noncomputable def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - ((x + 10) / 100 * 1000 * x)

theorem maximum_take_home_pay : 
  ∃ x : ℝ, (take_home_pay x = 20250) ∧ (45000 = 1000 * x) :=
by
  sorry

end maximum_take_home_pay_l146_146163


namespace athena_spent_l146_146741

theorem athena_spent :
  let sandwich_cost := 3
  let num_sandwiches := 3
  let drink_cost := 2.5
  let num_drinks := 2
  let total_cost := num_sandwiches * sandwich_cost + num_drinks * drink_cost
  total_cost = 14 :=
by
  let sandwich_cost := 3
  let num_sandwiches := 3
  let drink_cost := 2.5
  let num_drinks := 2
  let total_cost := num_sandwiches * sandwich_cost + num_drinks * drink_cost
  sorry

end athena_spent_l146_146741


namespace solve_system_of_equations_l146_146198

theorem solve_system_of_equations (x y z : ℝ) :
  x + y + z = 2 →
  x * y * z = 2 * (x * y + y * z + z * x) →
  ((x = -y ∧ z = 2) ∨ (y = -z ∧ x = 2) ∨ (z = -x ∧ y = 2)) :=
by
  intros h1 h2
  sorry

end solve_system_of_equations_l146_146198


namespace range_of_a_l146_146905

-- Definitions
def domain_f : Set ℝ := {x : ℝ | x ≤ -4 ∨ x ≥ 4}
def range_g (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ y = x^2 - 2*x + a}

-- Theorem to prove the range of values for a
theorem range_of_a :
  (∀ x : ℝ, x ∈ domain_f ∨ (∃ y : ℝ, ∃ a : ℝ, y ∈ range_g a ∧ x = y)) ↔ (-4 ≤ a ∧ a ≤ -3) :=
sorry

end range_of_a_l146_146905


namespace binary_to_decimal_l146_146429

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 := by
  sorry

end binary_to_decimal_l146_146429


namespace polynomial_value_at_8_l146_146489

noncomputable def P : ℝ → ℝ := λ x, x^2 - x + 2

theorem polynomial_value_at_8 :
  (P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) ∧ (P = λ x, x^2 - x + 2) → P 8 = 58 :=
by
  sorry

end polynomial_value_at_8_l146_146489


namespace impossible_arrangement_l146_146923

theorem impossible_arrangement (s : Finset ℕ) (h₁ : s = Finset.range 2018 \ {0})
  (h₂ : ∀ a ∈ s, ∀ b ∈ s, a ≠ b ∧ (b = a + 17 ∨ b = a + 21 ∨ b = a - 17 ∨ b = a - 21)) : False :=
by
  sorry

end impossible_arrangement_l146_146923


namespace scientific_notation_of_510000000_l146_146379

theorem scientific_notation_of_510000000 :
  (510000000 : ℝ) = 5.1 * 10^8 := 
sorry

end scientific_notation_of_510000000_l146_146379


namespace increase_80_by_135_percent_l146_146829

theorem increase_80_by_135_percent : 
  let original := 80 
  let increase := 1.35 
  original + (increase * original) = 188 := 
by
  sorry

end increase_80_by_135_percent_l146_146829


namespace min_value_eq_216_l146_146022

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c)

theorem min_value_eq_216 {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  min_value a b c = 216 :=
sorry

end min_value_eq_216_l146_146022


namespace steve_can_answer_38_questions_l146_146055

theorem steve_can_answer_38_questions (total_questions S : ℕ) 
  (h1 : total_questions = 45)
  (h2 : total_questions - S = 7) :
  S = 38 :=
by {
  -- The proof goes here
  sorry
}

end steve_can_answer_38_questions_l146_146055


namespace minimum_games_pasha_wins_l146_146485

noncomputable def pasha_initial_money : Nat := 9 -- Pasha has a single-digit amount
noncomputable def igor_initial_money : Nat := 1000 -- Igor has a four-digit amount
noncomputable def pasha_final_money : Nat := 100 -- Pasha has a three-digit amount
noncomputable def igor_final_money : Nat := 99 -- Igor has a two-digit amount

theorem minimum_games_pasha_wins :
  ∃ (games_won_by_pasha : Nat), 
    (games_won_by_pasha >= 7) ∧
    (games_won_by_pasha <= 7) := sorry

end minimum_games_pasha_wins_l146_146485


namespace fraction_to_decimal_l146_146130

theorem fraction_to_decimal : (7 : ℝ) / 250 = 0.028 := 
sorry

end fraction_to_decimal_l146_146130


namespace rate_percent_is_10_l146_146511

theorem rate_percent_is_10
  (SI : ℕ) (P : ℕ) (T : ℕ) (R : ℕ) 
  (h1 : SI = 2500) (h2 : P = 5000) (h3 : T = 5) :
  R = 10 :=
by
  sorry

end rate_percent_is_10_l146_146511


namespace number_of_spiders_l146_146110

theorem number_of_spiders (total_legs birds dogs snakes : ℕ) (legs_per_bird legs_per_dog legs_per_snake legs_per_spider : ℕ) (h1 : total_legs = 34)
  (h2 : birds = 3) (h3 : dogs = 5) (h4 : snakes = 4) (h5 : legs_per_bird = 2) (h6 : legs_per_dog = 4)
  (h7 : legs_per_snake = 0) (h8 : legs_per_spider = 8) : 
  (total_legs - (birds * legs_per_bird + dogs * legs_per_dog + snakes * legs_per_snake)) / legs_per_spider = 1 :=
by sorry

end number_of_spiders_l146_146110


namespace find_principal_amount_l146_146865

theorem find_principal_amount
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (h1 : SI = 4034.25)
  (h2 : R = 9)
  (h3 : T = 5) :
  P = 8965 :=
by
  sorry

end find_principal_amount_l146_146865


namespace largest_base_5_five_digits_base_10_value_l146_146085

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l146_146085


namespace min_value_of_expression_l146_146557

noncomputable def target_expression (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x)

theorem min_value_of_expression : (∀ x : ℝ, target_expression x ≥ -784) ∧ (∃ x : ℝ, target_expression x = -784) :=
by
  sorry

end min_value_of_expression_l146_146557


namespace sum_of_possible_values_l146_146376

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 8) = 4) : ∃ S : ℝ, S = 8 :=
sorry

end sum_of_possible_values_l146_146376


namespace sticks_difference_l146_146119

-- Definitions of the conditions
def d := 14  -- number of sticks Dave picked up
def a := 9   -- number of sticks Amy picked up
def total := 50  -- initial total number of sticks in the yard

-- The proof problem statement
theorem sticks_difference : (d + a) - (total - (d + a)) = 4 :=
by
  sorry

end sticks_difference_l146_146119


namespace jessica_found_seashells_l146_146027

-- Define the given conditions
def mary_seashells : ℕ := 18
def total_seashells : ℕ := 59

-- Define the goal for the number of seashells Jessica found
def jessica_seashells (mary_seashells total_seashells : ℕ) : ℕ := total_seashells - mary_seashells

-- The theorem stating Jessica found 41 seashells
theorem jessica_found_seashells : jessica_seashells mary_seashells total_seashells = 41 := by
  -- We assume the conditions and skip the proof
  sorry

end jessica_found_seashells_l146_146027


namespace smallest_multiple_17_7_more_53_l146_146513

theorem smallest_multiple_17_7_more_53 : 
  ∃ a : ℕ, (17 * a ≡ 7 [MOD 53]) ∧ 17 * a = 187 := 
by
  have h : 17* 11 = 187 := by norm_num
  use 11
  constructor
  · norm_num
  · exact h
  sorry

end smallest_multiple_17_7_more_53_l146_146513


namespace combined_CD_length_l146_146620

def CD1 := 1.5
def CD2 := 1.5
def CD3 := 2 * CD1

theorem combined_CD_length : CD1 + CD2 + CD3 = 6 := 
by
  sorry

end combined_CD_length_l146_146620


namespace find_investment_period_l146_146037

variable (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)

theorem find_investment_period (hP : P = 12000)
                               (hr : r = 0.10)
                               (hn : n = 2)
                               (hA : A = 13230) :
                               ∃ t : ℝ, A = P * (1 + r / n)^(n * t) ∧ t = 1 := 
by
  sorry

end find_investment_period_l146_146037


namespace find_plane_through_points_and_perpendicular_l146_146436

-- Definitions for points and plane conditions
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def point1 : Point3D := ⟨2, -2, 2⟩
def point2 : Point3D := ⟨0, 2, -1⟩

def normal_vector_of_given_plane : Point3D := ⟨2, -1, 2⟩

-- Lean 4 statement
theorem find_plane_through_points_and_perpendicular :
  ∃ (A B C D : ℤ), 
  (∀ (p : Point3D), (p = point1 ∨ p = point2) → A * p.x + B * p.y + C * p.z + D = 0) ∧
  (A * 2 + B * -1 + C * 2 = 0) ∧ 
  A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 ∧ 
  (A = 5 ∧ B = -2 ∧ C = 6 ∧ D = -26) :=
by
  sorry

end find_plane_through_points_and_perpendicular_l146_146436


namespace area_of_garden_l146_146531

variable (w l : ℕ)
variable (h1 : l = 3 * w) 
variable (h2 : 2 * (l + w) = 72)

theorem area_of_garden : l * w = 243 := by
  sorry

end area_of_garden_l146_146531


namespace hypotenuse_length_l146_146312

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l146_146312


namespace hyperbola_decreasing_l146_146891

-- Defining the variables and inequalities
variables (x : ℝ) (m : ℝ)
hypothesis (hx : x > 0)

-- The statement we need to prove
theorem hyperbola_decreasing (hx : x > 0) : (∀ x, x > 0 → (λ (x : ℝ), (1 - m) / x) x > (λ (x' : ℝ), (1 - m) / (x + x')) x) ↔ m < 1 :=
begin
  sorry
end

end hyperbola_decreasing_l146_146891


namespace math_problem_solution_l146_146123

theorem math_problem_solution (a b n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_eq : a ^ 2013 + b ^ 2013 = p ^ n) :
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ n = 2013 * k + 1 ∧ p = 2 :=
sorry

end math_problem_solution_l146_146123


namespace Kelly_needs_to_give_away_l146_146337

variable (n k : Nat)

theorem Kelly_needs_to_give_away (h_n : n = 20) (h_k : k = 12) : n - k = 8 := 
by
  sorry

end Kelly_needs_to_give_away_l146_146337


namespace sequence_property_l146_146978

theorem sequence_property (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n+1) + 2021 * a (n+2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := 
sorry

end sequence_property_l146_146978


namespace determinant_zero_implies_sum_l146_146755

open Matrix

noncomputable def matrix_example (a b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![2, 5, 8],
    ![4, a, b],
    ![4, b, a]
  ]

theorem determinant_zero_implies_sum (a b : ℝ) (h : a ≠ b) (h_det : det (matrix_example a b) = 0) : a + b = 26 :=
by
  sorry

end determinant_zero_implies_sum_l146_146755


namespace pie_remaining_portion_l146_146877

theorem pie_remaining_portion (carlos_portion maria_portion remaining_portion : ℝ)
  (h1 : carlos_portion = 0.6) 
  (h2 : remaining_portion = 1 - carlos_portion)
  (h3 : maria_portion = 0.5 * remaining_portion) :
  remaining_portion - maria_portion = 0.2 := 
by
  sorry

end pie_remaining_portion_l146_146877


namespace triangle_condition_A_triangle_condition_B_triangle_condition_C_triangle_condition_D_problem_solution_l146_146842

def triangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b

def right_triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem triangle_condition_A (a b c : ℝ) (h : triangle a b c) : 
  b^2 = (a + c) * (c - a) → right_triangle a c b := 
sorry

theorem triangle_condition_B (A B C : ℝ) (h : A + B + C = 180) : 
  A = B + C → 90 = A :=
sorry

theorem triangle_condition_C (A B C : ℝ) (h : A + B + C = 180) : 
  3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C → 
  ¬ (right_triangle A B C) :=
sorry

theorem triangle_condition_D : 
  right_triangle 6 8 10 := 
sorry

theorem problem_solution (a b c : ℝ) (A B C : ℝ) (hABC : triangle a b c) : 
  (b^2 = (a + c) * (c - a) → right_triangle a c b) ∧
  ((A + B + C = 180) ∧ (A = B + C) → 90 = A) ∧
  (3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C → ¬ right_triangle a b c) ∧
  (right_triangle 6 8 10) → 
  ∃ (cond : Prop), cond = (3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C) := 
sorry

end triangle_condition_A_triangle_condition_B_triangle_condition_C_triangle_condition_D_problem_solution_l146_146842


namespace largest_multiple_of_15_less_than_400_l146_146667

theorem largest_multiple_of_15_less_than_400 (x : ℕ) (k : ℕ) (h : x = 15 * k) (h1 : x < 400) (h2 : ∀ m : ℕ, (15 * m < 400) → m ≤ k) : x = 390 :=
by
  sorry

end largest_multiple_of_15_less_than_400_l146_146667


namespace dragon_cake_votes_l146_146331

theorem dragon_cake_votes (W U D : ℕ) (x : ℕ) 
  (hW : W = 7) 
  (hU : U = 3 * W) 
  (hD : D = W + x) 
  (hTotal : W + U + D = 60) 
  (hx : x = D - W) : 
  x = 25 := 
by
  sorry

end dragon_cake_votes_l146_146331


namespace paul_spending_l146_146760

theorem paul_spending :
  let cost_of_dress_shirts := 4 * 15
  let cost_of_pants := 2 * 40
  let cost_of_suit := 150
  let cost_of_sweaters := 2 * 30
  let total_cost := cost_of_dress_shirts + cost_of_pants + cost_of_suit + cost_of_sweaters
  let store_discount := 0.2 * total_cost
  let after_store_discount := total_cost - store_discount
  let coupon_discount := 0.1 * after_store_discount
  let final_amount := after_store_discount - coupon_discount
  final_amount = 252 :=
by
  -- Mathematically equivalent proof problem.
  sorry

end paul_spending_l146_146760


namespace train_car_speed_ratio_l146_146381

theorem train_car_speed_ratio
  (distance_bus : ℕ) (time_bus : ℕ) (distance_car : ℕ) (time_car : ℕ)
  (speed_bus := distance_bus / time_bus)
  (speed_train := speed_bus / (3 / 4))
  (speed_car := distance_car / time_car)
  (ratio := (speed_train : ℚ) / (speed_car : ℚ))
  (h1 : distance_bus = 480)
  (h2 : time_bus = 8)
  (h3 : distance_car = 450)
  (h4 : time_car = 6) :
  ratio = 16 / 15 :=
by
  sorry

end train_car_speed_ratio_l146_146381


namespace sales_in_second_month_l146_146415

-- Given conditions:
def sales_first_month : ℕ := 6400
def sales_third_month : ℕ := 6800
def sales_fourth_month : ℕ := 7200
def sales_fifth_month : ℕ := 6500
def sales_sixth_month : ℕ := 5100
def average_sales : ℕ := 6500

-- Statement to prove:
theorem sales_in_second_month :
  ∃ (sales_second_month : ℕ), 
    average_sales * 6 = sales_first_month + sales_second_month + sales_third_month 
    + sales_fourth_month + sales_fifth_month + sales_sixth_month 
    ∧ sales_second_month = 7000 :=
  sorry

end sales_in_second_month_l146_146415


namespace pauls_total_cost_is_252_l146_146761

variable (price_shirt : ℕ) (num_shirts : ℕ)
variable (price_pants : ℕ) (num_pants : ℕ)
variable (price_suit : ℕ) (num_suit : ℕ)
variable (price_sweater : ℕ) (num_sweaters : ℕ)
variable (store_discount : ℕ) (coupon_discount : ℕ)

-- Define the given prices and discounts
def total_cost_before_discounts : ℕ :=
  (price_shirt * num_shirts) +
  (price_pants * num_pants) +
  (price_suit * num_suit) +
  (price_sweater * num_sweaters)

def store_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def coupon_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def total_cost_after_discounts : ℕ :=
  let initial_total := total_cost_before_discounts price_shirt num_shirts price_pants num_pants price_suit num_suit price_sweater num_sweaters
  let store_discount_value := store_discount_amount initial_total store_discount
  let subtotal_after_store_discount := initial_total - store_discount_value
  let coupon_discount_value := coupon_discount_amount subtotal_after_store_discount coupon_discount
  subtotal_after_store_discount - coupon_discount_value

theorem pauls_total_cost_is_252 :
  total_cost_after_discounts 15 4 40 2 150 1 30 2 20 10 = 252 := by
  sorry

end pauls_total_cost_is_252_l146_146761


namespace necessarily_negative_l146_146193

theorem necessarily_negative
  (a b c : ℝ)
  (ha : -2 < a ∧ a < -1)
  (hb : 0 < b ∧ b < 1)
  (hc : -1 < c ∧ c < 0) :
  b + c < 0 :=
sorry

end necessarily_negative_l146_146193


namespace total_discount_is_15_l146_146783

structure Item :=
  (price : ℝ)      -- Regular price
  (discount_rate : ℝ) -- Discount rate in decimal form

def t_shirt : Item := {price := 25, discount_rate := 0.3}
def jeans : Item := {price := 75, discount_rate := 0.1}

def discount (item : Item) : ℝ :=
  item.discount_rate * item.price

def total_discount (items : List Item) : ℝ :=
  items.map discount |>.sum

theorem total_discount_is_15 :
  total_discount [t_shirt, jeans] = 15 := by
  sorry

end total_discount_is_15_l146_146783


namespace product_nonzero_except_cases_l146_146644

theorem product_nonzero_except_cases (n : ℤ) (h : n ≠ 5 ∧ n ≠ 17 ∧ n ≠ 257) : 
  (n - 5) * (n - 17) * (n - 257) ≠ 0 :=
by
  sorry

end product_nonzero_except_cases_l146_146644


namespace replace_asterisk_l146_146671

theorem replace_asterisk (x : ℕ) (h : (42 / 21) * (42 / x) = 1) : x = 84 := by
  sorry

end replace_asterisk_l146_146671


namespace quadratic_polynomial_value_l146_146492

theorem quadratic_polynomial_value (P : ℝ → ℝ) (hP : ∀ x, P(P(x)) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) :
  P 8 = 58 :=
sorry

end quadratic_polynomial_value_l146_146492


namespace ratio_of_55_to_11_l146_146825

theorem ratio_of_55_to_11 : (55 / 11) = 5 := 
by
  sorry

end ratio_of_55_to_11_l146_146825


namespace max_consecutive_sum_l146_146821

theorem max_consecutive_sum (a N : ℤ) (h₀ : N > 0) (h₁ : N * (2 * a + N - 1) = 90) : N = 90 :=
by
  -- Proof to be provided
  sorry

end max_consecutive_sum_l146_146821


namespace difficult_vs_easy_l146_146702

theorem difficult_vs_easy (x y z : ℕ) (h1 : x + y + z = 100) (h2 : x + 3 * y + 2 * z = 180) :
  x - y = 20 :=
by sorry

end difficult_vs_easy_l146_146702


namespace painter_total_fence_painted_l146_146167

theorem painter_total_fence_painted : 
  ∀ (L T W Th F : ℕ), 
  (T = W) → (W = Th) → 
  (L = T / 2) → 
  (F = 2 * T * (6 / 8)) → 
  (F = L + 300) → 
  (L + T + W + Th + F = 1500) :=
by
  sorry

end painter_total_fence_painted_l146_146167


namespace apples_per_sandwich_l146_146562

-- Define the conditions
def sam_sandwiches_per_day : Nat := 10
def days_in_week : Nat := 7
def total_apples_in_week : Nat := 280

-- Calculate total sandwiches in a week
def total_sandwiches_in_week := sam_sandwiches_per_day * days_in_week

-- Prove that Sam eats 4 apples for each sandwich
theorem apples_per_sandwich : total_apples_in_week / total_sandwiches_in_week = 4 :=
  by
    sorry

end apples_per_sandwich_l146_146562


namespace cuboid_surface_area_500_l146_146819

def surface_area (w l h : ℝ) : ℝ :=
  2 * l * w + 2 * l * h + 2 * w * h

theorem cuboid_surface_area_500 :
  ∀ (w l h : ℝ), w = 4 → l = w + 6 → h = l + 5 →
  surface_area w l h = 500 :=
by
  intros w l h hw hl hh
  unfold surface_area
  rw [hw, hl, hh]
  norm_num
  sorry

end cuboid_surface_area_500_l146_146819


namespace carolyn_initial_marbles_l146_146116

theorem carolyn_initial_marbles (x : ℕ) (h1 : x - 42 = 5) : x = 47 :=
by
  sorry

end carolyn_initial_marbles_l146_146116


namespace range_of_m_l146_146447

/-- The range of the real number m such that the equation x^2/m + y^2/(2m - 1) = 1 represents an ellipse with foci on the x-axis is (1/2, 1). -/
theorem range_of_m (m : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, x^2 / m + y^2 / (2 * m - 1) = 1 → x^2 / a^2 + y^2 / b^2 = 1 ∧ b^2 < a^2))
  ↔ 1 / 2 < m ∧ m < 1 :=
sorry

end range_of_m_l146_146447


namespace lucas_50_mod_5_l146_146041

-- Define the Lucas sequence
def lucas : ℕ → ℕ
| 0       := 1
| 1       := 3
| (n + 2) := lucas n + lucas (n + 1)

-- Proof statement
theorem lucas_50_mod_5 : (lucas 50) % 5 = 3 :=
by sorry

end lucas_50_mod_5_l146_146041


namespace common_ratio_of_geometric_series_l146_146814

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l146_146814


namespace xy_value_l146_146019

theorem xy_value : 
  ∀ (x y : ℝ),
  (∀ (A B C : ℝ × ℝ), A = (1, 8) ∧ B = (x, y) ∧ C = (6, 3) → 
  (C.1 = (A.1 + B.1) / 2) ∧ (C.2 = (A.2 + B.2) / 2)) → 
  x * y = -22 :=
sorry

end xy_value_l146_146019


namespace distance_between_neg2_and_3_l146_146778
-- Import the necessary Lean libraries

-- State the theorem to prove the distance between -2 and 3 is 5
theorem distance_between_neg2_and_3 : abs (3 - (-2)) = 5 := by
  sorry

end distance_between_neg2_and_3_l146_146778


namespace solve_for_n_l146_146732

theorem solve_for_n (n : ℕ) (h : 2 * 2 * 3 * 3 * 5 * 6 = 5 * 6 * n * n) : n = 6 := by
  sorry

end solve_for_n_l146_146732


namespace largest_base5_number_conversion_l146_146069

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l146_146069


namespace geometric_series_common_ratio_l146_146801

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l146_146801


namespace hypotenuse_length_l146_146316

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l146_146316


namespace hypotenuse_length_l146_146324

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l146_146324


namespace rotate_A_180_about_B_l146_146354

-- Define the points A, B, and C
def A : ℝ × ℝ := (-4, 1)
def B : ℝ × ℝ := (-1, 4)
def C : ℝ × ℝ := (-1, 1)

-- Define the 180 degrees rotation about B
def rotate_180_about (p q : ℝ × ℝ) : ℝ × ℝ :=
  let translated_p := (p.1 - q.1, p.2 - q.2) 
  let rotated_p := (-translated_p.1, -translated_p.2)
  (rotated_p.1 + q.1, rotated_p.2 + q.2)

-- Prove the image of point A after a 180 degrees rotation about point B
theorem rotate_A_180_about_B : rotate_180_about A B = (2, 7) :=
by
  sorry

end rotate_A_180_about_B_l146_146354


namespace angle_Z_proof_l146_146346

-- Definitions of the given conditions
variables {p q : Type} [Parallel p q]
variables {X Y Z : ℝ}
variables (mAngleX : X = 100)
variables (mAngleY : Y = 130)

-- Statement of the proof problem
theorem angle_Z_proof (hpq : Parallel p q) (hX : X = 100) (hY : Y = 130) : Z = 130 :=
sorry

end angle_Z_proof_l146_146346


namespace local_language_letters_l146_146009

theorem local_language_letters (n : ℕ) (h : 1 + 2 * n = 139) : n = 69 :=
by
  -- Proof skipped
  sorry

end local_language_letters_l146_146009


namespace cos_neg_pi_over_3_eq_one_half_sin_eq_sqrt3_over_2_solutions_l146_146254

noncomputable def cos_negative_pi_over_3 : Real :=
  Real.cos (-Real.pi / 3)

theorem cos_neg_pi_over_3_eq_one_half :
  cos_negative_pi_over_3 = 1 / 2 :=
  by
    sorry

noncomputable def solutions_sin_eq_sqrt3_over_2 (x : Real) : Prop :=
  Real.sin x = Real.sqrt 3 / 2 ∧ 0 ≤ x ∧ x < 2 * Real.pi

theorem sin_eq_sqrt3_over_2_solutions :
  {x : Real | solutions_sin_eq_sqrt3_over_2 x} = {Real.pi / 3, 2 * Real.pi / 3} :=
  by
    sorry

end cos_neg_pi_over_3_eq_one_half_sin_eq_sqrt3_over_2_solutions_l146_146254


namespace baseball_card_decrease_l146_146524

noncomputable def percentDecrease (V : ℝ) (P : ℝ) : ℝ :=
  V * (P / 100)

noncomputable def valueAfterDecrease (V : ℝ) (D : ℝ) : ℝ :=
  V - D

theorem baseball_card_decrease (V : ℝ) (H1 : V > 0) :
  let D1 := percentDecrease V 50
  let V1 := valueAfterDecrease V D1
  let D2 := percentDecrease V1 10
  let V2 := valueAfterDecrease V1 D2
  let totalDecrease := V - V2
  totalDecrease / V * 100 = 55 := sorry

end baseball_card_decrease_l146_146524


namespace Harold_speed_is_one_more_l146_146245

variable (Adrienne_speed Harold_speed : ℝ)
variable (distance_when_Harold_catches_Adr : ℝ)
variable (time_difference : ℝ)

axiom Adrienne_speed_def : Adrienne_speed = 3
axiom Harold_catches_distance : distance_when_Harold_catches_Adr = 12
axiom time_difference_def : time_difference = 1

theorem Harold_speed_is_one_more :
  Harold_speed - Adrienne_speed = 1 :=
by 
  have Adrienne_time := (distance_when_Harold_catches_Adr - Adrienne_speed * time_difference) / Adrienne_speed 
  have Harold_time := distance_when_Harold_catches_Adr / Harold_speed
  have := Adrienne_time = Harold_time - time_difference
  sorry

end Harold_speed_is_one_more_l146_146245


namespace housewife_saving_l146_146100

theorem housewife_saving :
  let total_money := 450
  let groceries_fraction := 3 / 5
  let household_items_fraction := 1 / 6
  let personal_care_items_fraction := 1 / 10
  let groceries_expense := groceries_fraction * total_money
  let household_items_expense := household_items_fraction * total_money
  let personal_care_items_expense := personal_care_items_fraction * total_money
  let total_expense := groceries_expense + household_items_expense + personal_care_items_expense
  total_money - total_expense = 60 :=
by
  sorry

end housewife_saving_l146_146100


namespace cans_of_beans_is_two_l146_146694

-- Define the problem parameters
variable (C B T : ℕ)

-- Conditions based on the problem statement
axiom chili_can : C = 1
axiom tomato_to_bean_ratio : T = 3 * B / 2
axiom quadruple_batch_cans : 4 * (C + B + T) = 24

-- Prove the number of cans of beans is 2
theorem cans_of_beans_is_two : B = 2 :=
by
  -- Include conditions
  have h1 : C = 1 := by sorry
  have h2 : T = 3 * B / 2 := by sorry
  have h3 : 4 * (C + B + T) = 24 := by sorry
  -- Derive the answer (Proof omitted)
  sorry

end cans_of_beans_is_two_l146_146694


namespace geometric_sequence_sum_l146_146013

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 3 + a 5 = 20)
  (h2 : a 4 = 8)
  (h3 : ∀ n, a (n + 1) = a n * q) :
  a 2 + a 6 = 34 := 
sorry

end geometric_sequence_sum_l146_146013


namespace max_abs_sum_l146_146580

theorem max_abs_sum (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_l146_146580


namespace find_a22_l146_146975

-- Definitions and conditions
noncomputable def seq (n : ℕ) : ℝ := if n = 0 then 0 else sorry

axiom seq_conditions
  (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) : True

theorem find_a22 (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 :=
sorry

end find_a22_l146_146975


namespace math_problem_l146_146357

variable (a a' b b' c c' : ℝ)

theorem math_problem 
  (h1 : a * a' > 0) 
  (h2 : a * c ≥ b * b) 
  (h3 : a' * c' ≥ b' * b') : 
  (a + a') * (c + c') ≥ (b + b') * (b + b') := 
by
  sorry

end math_problem_l146_146357


namespace probability_three_even_l146_146111

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the probability of exactly three dice showing an even number
noncomputable def prob_exactly_three_even (n : ℕ) (k : ℕ) (p : ℚ) : ℚ := 
  (binomial n k : ℚ) * (p^k) * ((1 - p)^(n - k))

-- The main theorem stating the desired probability
theorem probability_three_even (n : ℕ) (p : ℚ) (k : ℕ) (h₁ : n = 6) (h₂ : p = 1/2) (h₃ : k = 3) :
  prob_exactly_three_even n k p = 5 / 16 := by
  sorry

-- Include required definitions and expected values for the theorem
#check binomial
#check prob_exactly_three_even
#check probability_three_even

end probability_three_even_l146_146111


namespace ordered_pairs_of_positive_integers_l146_146205

theorem ordered_pairs_of_positive_integers (x y : ℕ) (h : x * y = 2800) :
  2^4 * 5^2 * 7 = 2800 → ∃ (n : ℕ), n = 30 ∧ (∃ x y : ℕ, x * y = 2800 ∧ n = 30) :=
by
  sorry

end ordered_pairs_of_positive_integers_l146_146205


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l146_146304

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l146_146304


namespace num_pigs_on_farm_l146_146723

variables (P : ℕ)
def cows := 2 * P - 3
def goats := (2 * P - 3) + 6
def total_animals := P + cows P + goats P

theorem num_pigs_on_farm (h : total_animals P = 50) : P = 10 :=
sorry

end num_pigs_on_farm_l146_146723


namespace min_value_expr_l146_146555

theorem min_value_expr : ∃ (x : ℝ), (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 ∧ 
  ∀ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by
  sorry

end min_value_expr_l146_146555


namespace geometric_sequence_general_formula_l146_146615

theorem geometric_sequence_general_formula (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h1 : a 1 = 2)
  (h_rec : ∀ n, (a (n + 2))^2 + 4 * (a n)^2 = 4 * (a (n + 1))^2) :
  ∀ n, a n = 2^(n + 1) / 2 := 
sorry

end geometric_sequence_general_formula_l146_146615


namespace problem_1_part_1_problem_1_part_2_l146_146342

-- Define the function f
def f (x a : ℝ) := |x - a| + 3 * x

-- The first problem statement - Part (Ⅰ)
theorem problem_1_part_1 (x : ℝ) : { x | x ≥ 3 ∨ x ≤ -1 } = { x | f x 1 ≥ 3 * x + 2 } :=
by {
  sorry
}

-- The second problem statement - Part (Ⅱ)
theorem problem_1_part_2 : { x | x ≤ -1 } = { x | f x 2 ≤ 0 } :=
by {
  sorry
}

end problem_1_part_1_problem_1_part_2_l146_146342


namespace find_certain_number_l146_146679

theorem find_certain_number (x : ℝ) (h : 0.80 * x = (4 / 5 * 20) + 16) : x = 40 :=
by sorry

end find_certain_number_l146_146679


namespace rectangle_area_eq_six_l146_146967

-- Define the areas of the small squares
def smallSquareArea : ℝ := 1

-- Define the number of small squares
def numberOfSmallSquares : ℤ := 2

-- Define the area of the larger square
def largeSquareArea : ℝ := (2 ^ 2)

-- Define the area of rectangle ABCD
def areaRectangleABCD : ℝ :=
  (numberOfSmallSquares * smallSquareArea) + largeSquareArea

-- The theorem we want to prove
theorem rectangle_area_eq_six :
  areaRectangleABCD = 6 := by sorry

end rectangle_area_eq_six_l146_146967


namespace consecutive_integers_product_no_3n_plus_1_product_consecutive_no_n_cubed_plus_5n_plus_4_product_consecutive_no_permutation_23456780_product_consecutive_l146_146847

-- 6(a): Prove that the product of two consecutive integers is either divisible by 6 or gives a remainder of 2 when divided by 18.
theorem consecutive_integers_product (n : ℕ) : n * (n + 1) % 18 = 0 ∨ n * (n + 1) % 18 = 2 := 
sorry

-- 6(b): Prove that there does not exist an integer n such that the number 3n + 1 is the product of two consecutive integers.
theorem no_3n_plus_1_product_consecutive : ¬ ∃ n : ℕ, ∃ m : ℕ, 3 * m + 1 = m * (m + 1) := 
sorry

-- 6(c): Prove that for no integer n, the number n^3 + 5n + 4 can be the product of two consecutive integers.
theorem no_n_cubed_plus_5n_plus_4_product_consecutive : ¬ ∃ n : ℕ, ∃ m : ℕ, n^3 + 5 * n + 4 = m * (m + 1) := 
sorry

-- 6(d): Prove that none of the numbers resulting from the rearrangement of the digits in 23456780 is the product of two consecutive integers.
def is_permutation (m : ℕ) (n : ℕ) : Prop := 
-- This function definition should check that m is a permutation of the digits of n
sorry

theorem no_permutation_23456780_product_consecutive : 
  ∀ m : ℕ, is_permutation m 23456780 → ¬ ∃ n : ℕ, m = n * (n + 1) := 
sorry

end consecutive_integers_product_no_3n_plus_1_product_consecutive_no_n_cubed_plus_5n_plus_4_product_consecutive_no_permutation_23456780_product_consecutive_l146_146847


namespace geometric_series_common_ratio_l146_146803

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l146_146803


namespace total_flour_needed_l146_146173

theorem total_flour_needed (Katie_flour : ℕ) (Sheila_flour : ℕ) 
  (h1 : Katie_flour = 3) 
  (h2 : Sheila_flour = Katie_flour + 2) : 
  Katie_flour + Sheila_flour = 8 := 
  by 
  sorry

end total_flour_needed_l146_146173


namespace geometric_sequence_common_ratio_is_2_l146_146137

variable {a : ℕ → ℝ} (h : ∀ n : ℕ, a n * a (n + 1) = 4 ^ n)

theorem geometric_sequence_common_ratio_is_2 : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_is_2_l146_146137


namespace beads_bracelet_rotational_symmetry_l146_146011

theorem beads_bracelet_rotational_symmetry :
  let n := 8
  let factorial := Nat.factorial
  (factorial n / n = 5040) := by
  sorry

end beads_bracelet_rotational_symmetry_l146_146011


namespace problem_I_problem_II_l146_146574

def intervalA := { x : ℝ | -2 < x ∧ x < 5 }
def intervalB (m : ℝ) := { x : ℝ | m < x ∧ x < m + 3 }

theorem problem_I (m : ℝ) :
  (intervalB m ⊆ intervalA) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by sorry

theorem problem_II (m : ℝ) :
  (intervalA ∩ intervalB m ≠ ∅) ↔ (-5 < m ∧ m < 2) :=
by sorry

end problem_I_problem_II_l146_146574


namespace quadratic_P_value_l146_146490

noncomputable def P (x : ℝ) : ℝ :=
  x^2 - x + 2

theorem quadratic_P_value :
  P (P 8) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 ∧ P 8 = 58 := 
by
  have h1 : P(P(8)) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 := sorry
  have h2 : P(8) = 58 := sorry
  exact ⟨h1, h2⟩  

end quadratic_P_value_l146_146490


namespace Taimour_paint_time_l146_146470

theorem Taimour_paint_time (T : ℝ) (H1 : ∀ t : ℝ, t = 2 / T → t ≠ 0) (H2 : (1 / T + 2 / T) = 1 / 3) : T = 9 :=
by
  sorry

end Taimour_paint_time_l146_146470


namespace max_min_values_of_f_l146_146267

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_min_values_of_f :
  (∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x = 2) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x ≥ -18) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x = -18)
:= by
  sorry  -- To be replaced with the actual proof

end max_min_values_of_f_l146_146267


namespace passengers_landed_in_virginia_l146_146424

theorem passengers_landed_in_virginia
  (P_start : ℕ) (D_Texas : ℕ) (C_Texas : ℕ) (D_NC : ℕ) (C_NC : ℕ) (C : ℕ)
  (hP_start : P_start = 124)
  (hD_Texas : D_Texas = 58)
  (hC_Texas : C_Texas = 24)
  (hD_NC : D_NC = 47)
  (hC_NC : C_NC = 14)
  (hC : C = 10) :
  P_start - D_Texas + C_Texas - D_NC + C_NC + C = 67 := by
  sorry

end passengers_landed_in_virginia_l146_146424


namespace age_difference_l146_146210

variable (A B C : ℕ)

-- Conditions: C is 11 years younger than A
axiom h1 : C = A - 11

-- Statement: Prove the difference (A + B) - (B + C) is 11
theorem age_difference : (A + B) - (B + C) = 11 := by
  sorry

end age_difference_l146_146210


namespace factor_b_value_l146_146146

theorem factor_b_value (a b : ℤ) (h : ∀ x : ℂ, (x^2 - x - 1) ∣ (a*x^3 + b*x^2 + 1)) : b = -2 := 
sorry

end factor_b_value_l146_146146


namespace abs_inequality_solution_set_l146_146659

theorem abs_inequality_solution_set :
  { x : ℝ | |x - 1| + |x + 2| ≥ 5 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end abs_inequality_solution_set_l146_146659


namespace milk_drinks_on_weekdays_l146_146025

-- Defining the number of boxes Lolita drinks on a weekday as a variable W
variable (W : ℕ)

-- Condition: Lolita drinks 30 boxes of milk per week.
axiom total_milk_per_week : 5 * W + 2 * W + 3 * W = 30

-- Proof (Statement) that Lolita drinks 15 boxes of milk on weekdays.
theorem milk_drinks_on_weekdays : 5 * W = 15 :=
by {
  -- Use the given axiom to derive the solution
  sorry
}

end milk_drinks_on_weekdays_l146_146025


namespace hypotenuse_length_l146_146315

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l146_146315


namespace combined_work_time_l146_146095

-- Define the time taken by Paul and Rose to complete the work individually
def paul_days : ℕ := 80
def rose_days : ℕ := 120

-- Define the work rates of Paul and Rose
def paul_rate := 1 / (paul_days : ℚ)
def rose_rate := 1 / (rose_days : ℚ)

-- Define the combined work rate
def combined_rate := paul_rate + rose_rate

-- Statement to prove: Together they can complete the work in 48 days.
theorem combined_work_time : combined_rate = 1 / 48 := by 
  sorry

end combined_work_time_l146_146095


namespace point_coordinates_l146_146330

theorem point_coordinates (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : abs y = 5) (h4 : abs x = 2) : x = -2 ∧ y = 5 :=
by
  sorry

end point_coordinates_l146_146330


namespace rabbit_total_distance_l146_146386

theorem rabbit_total_distance 
  (r₁ r₂ : ℝ) 
  (h1 : r₁ = 7) 
  (h2 : r₂ = 15) 
  (q : ∀ (x : ℕ), x = 4) 
  : (3.5 * π + 8 + 7.5 * π + 8 + 3.5 * π + 8) = 14.5 * π + 24 := 
by
  sorry

end rabbit_total_distance_l146_146386


namespace find_alpha_l146_146333

theorem find_alpha (α : ℝ) :
    7 * α + 8 * α + 45 = 180 →
    α = 9 :=
by
  sorry

end find_alpha_l146_146333


namespace problem_part_1_problem_part_2_l146_146282

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x

noncomputable def g (x : ℝ) : ℝ := Real.log ((x + 2) / (x - 2))

theorem problem_part_1 :
  ∀ (x₁ x₂ : ℝ), 0 < x₂ ∧ x₂ < x₁ → Real.log x₁ + 2 * x₁ > Real.log x₂ + 2 * x₂ :=
sorry

theorem problem_part_2 :
  ∃ k : ℕ, ∀ (x₁ : ℝ), 0 < x₁ ∧ x₁ < 1 → (∃ (x₂ : ℝ), x₂ ∈ Set.Ioo (k : ℝ) (k + 1) ∧ Real.log x₁ + 2 * x₁ < Real.log ((x₂ + 2) / (x₂ - 2))) → k = 2 :=
sorry

end problem_part_1_problem_part_2_l146_146282


namespace sum_of_first_five_terms_l146_146442

noncomputable -- assuming non-computable for general proof involving sums
def arithmetic_sequence_sum (a_n : ℕ → ℤ) := ∃ d m : ℤ, ∀ n : ℕ, a_n = m + n * d

theorem sum_of_first_five_terms 
(a_n : ℕ → ℤ) 
(h_arith : arithmetic_sequence_sum a_n)
(h_cond : a_n 5 + a_n 8 - a_n 10 = 2)
: ((a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) = 10) := 
by 
  sorry

end sum_of_first_five_terms_l146_146442


namespace find_f_neg_one_l146_146141

theorem find_f_neg_one (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_symm : ∀ x, f (4 - x) = -f x)
  (h_f3 : f 3 = 3) :
  f (-1) = 3 := 
sorry

end find_f_neg_one_l146_146141


namespace isosceles_triangle_largest_angle_l146_146736

theorem isosceles_triangle_largest_angle (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = 50) :
  A + B + C = 180 →
  C = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l146_146736


namespace initial_average_weight_l146_146494

theorem initial_average_weight 
    (W : ℝ)
    (a b c d e : ℝ)
    (h1 : (a + b + c) / 3 = W)
    (h2 : (a + b + c + d) / 4 = W)
    (h3 : (b + c + d + (d + 3)) / 4 = 68)
    (h4 : a = 81) :
    W = 70 := 
sorry

end initial_average_weight_l146_146494


namespace find_a_of_perpendicular_lines_l146_146459

theorem find_a_of_perpendicular_lines (a : ℝ) :
  let line1 : ℝ := a * x + y - 1
  let line2 : ℝ := 4 * x + (a - 3) * y - 2
  (∀ x y : ℝ, (line1 = 0 → line2 ≠ 0 → line1 * line2 = -1)) → a = 3 / 5 :=
by
  sorry

end find_a_of_perpendicular_lines_l146_146459


namespace simplify_expression_l146_146408

theorem simplify_expression :
  (1 / 2^2 + (2 / 3^3 * (3 / 2)^2) + 4^(1/2)) - 8 / (4^2 - 3^2) = 107 / 84 :=
by
  -- Skip the proof
  sorry

end simplify_expression_l146_146408


namespace minimum_possible_base_maximum_possible_base_total_possible_bases_l146_146653

open Nat

def num_trailing_zeroes_in_factorial (n : Nat) : Nat :=
  let rec count_factors (n k acc : Nat) :=
    let div := n / k
    if div = 0 then acc else count_factors n (k * 5) (acc + div)
  count_factors n 5 0

theorem minimum_possible_base (n : Nat) (H : n = 2 + 2^96)
        (trailing_zeroes : Nat) (Hz : trailing_zeroes = 2^93) :
  ∃ B, (num_trailing_zeroes_in_factorial n = trailing_zeroes) → B = 16 := by
  sorry

theorem maximum_possible_base (n : Nat) (H : n = 2 + 2^96)
        (trailing_zeroes : Nat) (Hz : trailing_zeroes = 2^93) :
  ∃ B, (num_trailing_zeroes_in_factorial n = trailing_zeroes) → B = 5040 := by
  sorry

theorem total_possible_bases (n : Nat) (H : n = 2 + 2^96)
        (trailing_zeroes : Nat) (Hz : trailing_zeroes = 2^93) :
  ∃ count, (count = 12) := by
  sorry

end minimum_possible_base_maximum_possible_base_total_possible_bases_l146_146653


namespace sum_of_solutions_l146_146669

def equation (x : ℝ) : Prop := (6 * x) / 30 = 8 / x

theorem sum_of_solutions : ∀ x1 x2 : ℝ, equation x1 → equation x2 → x1 + x2 = 0 := by
  sorry

end sum_of_solutions_l146_146669


namespace cylinder_surface_area_l146_146652

variable (height1 height2 radius1 radius2 : ℝ)
variable (π : ℝ)
variable (C1 : height1 = 6 * π)
variable (C2 : radius1 = 3)
variable (C3 : height2 = 4 * π)
variable (C4 : radius2 = 2)

theorem cylinder_surface_area : 
  (6 * π * 4 * π + 2 * π * radius1 ^ 2) = 24 * π ^ 2 + 18 * π ∨
  (4 * π * 6 * π + 2 * π * radius2 ^ 2) = 24 * π ^ 2 + 8 * π :=
by
  intros
  sorry

end cylinder_surface_area_l146_146652


namespace quadratic_expression_rewriting_l146_146583

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l146_146583


namespace ball_center_distance_traveled_l146_146413

theorem ball_center_distance_traveled (d : ℝ) (r1 r2 r3 r4 : ℝ) (R1 R2 R3 R4 : ℝ) :
  d = 6 → 
  R1 = 120 → 
  R2 = 50 → 
  R3 = 90 → 
  R4 = 70 → 
  r1 = R1 - 3 → 
  r2 = R2 + 3 → 
  r3 = R3 - 3 → 
  r4 = R4 + 3 → 
  (1/2) * 2 * π * r1 + (1/2) * 2 * π * r2 + (1/2) * 2 * π * r3 + (1/2) * 2 * π * r4 = 330 * π :=
by
  sorry

end ball_center_distance_traveled_l146_146413


namespace greatest_product_of_digits_l146_146164

theorem greatest_product_of_digits :
  ∀ a b : ℕ, (10 * a + b) % 35 = 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  ∃ ab_max : ℕ, ab_max = a * b ∧ ab_max = 15 :=
by
  sorry

end greatest_product_of_digits_l146_146164


namespace right_triangle_area_l146_146032

theorem right_triangle_area (a b c r : ℝ) (h1 : a = 15) (h2 : r = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_right : a ^ 2 + b ^ 2 = c ^ 2) (h_incircle : r = (a + b - c) / 2) : 
  1 / 2 * a * b = 60 :=
by
  sorry

end right_triangle_area_l146_146032


namespace find_fraction_l146_146185

-- Let's define the conditions
variables (F N : ℝ)
axiom condition1 : (1 / 4) * (1 / 3) * F * N = 15
axiom condition2 : 0.4 * N = 180

-- theorem to prove the fraction F
theorem find_fraction : F = 2 / 5 :=
by
  -- proof steps would go here, but we're adding sorry to skip the proof.
  sorry

end find_fraction_l146_146185


namespace find_y_l146_146286

theorem find_y 
  (x y : ℝ) 
  (h1 : 3 * x - 2 * y = 18) 
  (h2 : x + 2 * y = 10) : 
  y = 1.5 := 
by 
  sorry

end find_y_l146_146286


namespace find_c_l146_146968

def is_midpoint (p1 p2 mid : ℝ × ℝ) : Prop :=
(mid.1 = (p1.1 + p2.1) / 2) ∧ (mid.2 = (p1.2 + p2.2) / 2)

def is_perpendicular_bisector (line : ℝ → ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop := 
∃ mid : ℝ × ℝ, 
is_midpoint p1 p2 mid ∧ line mid.1 mid.2 = 0

theorem find_c (c : ℝ) : 
is_perpendicular_bisector (λ x y => 3 * x - y - c) (2, 4) (6, 8) → c = 6 :=
by
  sorry

end find_c_l146_146968


namespace decreasing_hyperbola_l146_146892

theorem decreasing_hyperbola (m : ℝ) (x : ℝ) (hx : x > 0) (y : ℝ) (h_eq : y = (1 - m) / x) :
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > x₁ → (1 - m) / x₂ < (1 - m) / x₁) ↔ m < 1 :=
by
  sorry

end decreasing_hyperbola_l146_146892


namespace find_h_l146_146602

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l146_146602


namespace cyclic_quadrilateral_AD_correct_l146_146711

noncomputable def cyclic_quadrilateral_AD_length : ℝ :=
  let R := 200 * Real.sqrt 2
  let AB := 200
  let BC := 200
  let CD := 200
  let AD := 500
  sorry

theorem cyclic_quadrilateral_AD_correct (R AB BC CD AD : ℝ) (hR : R = 200 * Real.sqrt 2) 
  (hAB : AB = 200) (hBC : BC = 200) (hCD : CD = 200) : AD = 500 :=
by
  have hRABBCDC: R = 200 * Real.sqrt 2 ∧ AB = 200 ∧ BC = 200 ∧ CD = 200 := ⟨hR, hAB, hBC, hCD⟩
  sorry

end cyclic_quadrilateral_AD_correct_l146_146711


namespace sum_gcf_lcm_36_56_84_l146_146392

def gcf (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem sum_gcf_lcm_36_56_84 :
  let gcf_36_56_84 := gcf 36 56 84
  let lcm_36_56_84 := lcm 36 56 84
  gcf_36_56_84 + lcm_36_56_84 = 516 :=
by
  let gcf_36_56_84 := gcf 36 56 84
  let lcm_36_56_84 := lcm 36 56 84
  show gcf_36_56_84 + lcm_36_56_84 = 516
  sorry

end sum_gcf_lcm_36_56_84_l146_146392


namespace total_spent_l146_146890

theorem total_spent (B D : ℝ) (h1 : D = 0.7 * B) (h2 : B = D + 15) : B + D = 85 :=
sorry

end total_spent_l146_146890


namespace taxi_ride_cost_l146_146867

theorem taxi_ride_cost :
  let base_fare : ℝ := 2.00
  let cost_per_mile_first_3 : ℝ := 0.30
  let cost_per_mile_additional : ℝ := 0.40
  let total_distance : ℕ := 8
  let first_3_miles_cost : ℝ := base_fare + 3 * cost_per_mile_first_3
  let additional_miles_cost : ℝ := (total_distance - 3) * cost_per_mile_additional
  let total_cost : ℝ := first_3_miles_cost + additional_miles_cost
  total_cost = 4.90 :=
by
  sorry

end taxi_ride_cost_l146_146867


namespace geometric_series_ratio_l146_146787

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l146_146787


namespace quadratic_expression_rewriting_l146_146584

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l146_146584


namespace largest_base5_number_to_base10_is_3124_l146_146066

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l146_146066


namespace equilateral_triangle_l146_146052

theorem equilateral_triangle (a b c : ℝ) (h1 : b^2 = a * c) (h2 : 2 * b = a + c) : a = b ∧ b = c ∧ a = c := by
  sorry

end equilateral_triangle_l146_146052


namespace cistern_filled_in_12_hours_l146_146098

def fill_rate := 1 / 6
def empty_rate := 1 / 12
def net_rate := fill_rate - empty_rate

theorem cistern_filled_in_12_hours :
  (1 / net_rate) = 12 :=
by
  -- Proof omitted for clarity
  sorry

end cistern_filled_in_12_hours_l146_146098


namespace problem_eight_sided_polygon_interiors_l146_146155

-- Define the condition of the problem
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- One interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℚ := sum_of_interior_angles n / n

-- The main theorem stating the problem
theorem problem_eight_sided_polygon_interiors (n : ℕ) (h1: diagonals_from_vertex n = 5) : 
  one_interior_angle n = 135 :=
by
  -- Proof would go here
  sorry

end problem_eight_sided_polygon_interiors_l146_146155


namespace product_divisible_by_third_l146_146021

theorem product_divisible_by_third (a b c : Int)
    (h1 : (a + b + c)^2 = -(a * b + a * c + b * c))
    (h2 : a + b ≠ 0) (h3 : b + c ≠ 0) (h4 : a + c ≠ 0) :
    ((a + b) * (a + c) % (b + c) = 0) ∧ ((a + b) * (b + c) % (a + c) = 0) ∧ ((a + c) * (b + c) % (a + b) = 0) :=
  sorry

end product_divisible_by_third_l146_146021


namespace one_interior_angle_of_polygon_with_five_diagonals_l146_146153

theorem one_interior_angle_of_polygon_with_five_diagonals (n : ℕ) (h : n - 3 = 5) :
  let interior_angle := 180 * (n - 2) / n
  interior_angle = 135 :=
by
  sorry

end one_interior_angle_of_polygon_with_five_diagonals_l146_146153


namespace collinear_A_P_Q_l146_146628

variables {A B C D E P F G Q : Point}
variables [f : Euclidean_Geometry]

-- Conditions
axiom D_on_AB : OnSegment A B D
axiom E_on_AC : OnSegment A C E
axiom DE_parallel_BC : Parallel (Line.mk D E) (Line.mk B C)
axiom P_in_ADE : InTriangle A D E P
axiom F_int_DE_BP : Intersects (Line.mk D E) (Line.mk B P) F
axiom G_int_DE_CP : Intersects (Line.mk D E) (Line.mk C P) G
axiom Q_circ_PDG_PFE : SecIntersection (Circ P D G) (Circ P F E) Q

-- Theorem to prove
theorem collinear_A_P_Q : Collinear A P Q :=
  sorry

end collinear_A_P_Q_l146_146628


namespace hypotenuse_length_l146_146317

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l146_146317


namespace tan_of_angle_in_third_quadrant_l146_146001

open Real

theorem tan_of_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : sin (π + α) = 3/5) 
  (h2 : π < α ∧ α < 3 * π / 2) : 
  tan α = 3 / 4 :=
by
  sorry

end tan_of_angle_in_third_quadrant_l146_146001


namespace sum_x_y_l146_146462

theorem sum_x_y (x y : ℤ) (h1 : x - y = 40) (h2 : x = 32) : x + y = 24 := by
  sorry

end sum_x_y_l146_146462


namespace sum_of_areas_of_circles_l146_146987

-- Definitions and given conditions
variables (r s t : ℝ)
variables (h1 : r + s = 5)
variables (h2 : r + t = 12)
variables (h3 : s + t = 13)

-- The sum of the areas
theorem sum_of_areas_of_circles : 
  π * r^2 + π * s^2 + π * t^2 = 113 * π :=
  by
    sorry

end sum_of_areas_of_circles_l146_146987


namespace no_b_satisfies_l146_146900

theorem no_b_satisfies (b : ℝ) : ¬ (2 * 1 - b * (-2) + 1 ≤ 0 ∧ 2 * (-1) - b * 2 + 1 ≤ 0) :=
by
  sorry

end no_b_satisfies_l146_146900


namespace hypotenuse_length_l146_146308

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l146_146308


namespace other_person_age_l146_146181

variable {x : ℕ} -- age of the other person
variable {y : ℕ} -- Marco's age

-- Conditions given in the problem.
axiom marco_age : y = 2 * x + 1
axiom sum_ages : x + y = 37

-- Goal: Prove that the age of the other person is 12.
theorem other_person_age : x = 12 :=
by
  -- Proof is skipped
  sorry

end other_person_age_l146_146181


namespace swimmer_speed_is_4_4_l146_146866

noncomputable def swimmer_speed_in_still_water (distance : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
(distance / time) + current_speed

theorem swimmer_speed_is_4_4 :
  swimmer_speed_in_still_water 7 2.5 3.684210526315789 = 4.4 :=
by
  -- This part would contain the proof to show that the calculated speed is 4.4
  sorry

end swimmer_speed_is_4_4_l146_146866


namespace remainder_7n_mod_4_l146_146840

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by sorry

end remainder_7n_mod_4_l146_146840


namespace mary_saw_total_snakes_l146_146944

theorem mary_saw_total_snakes :
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  totalSnakes = 36 :=
by
  /- Definitions -/ 
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  /- Main proof statement -/
  show totalSnakes = 36
  sorry

end mary_saw_total_snakes_l146_146944


namespace total_stars_l146_146212

theorem total_stars (students stars_per_student : ℕ) (h_students : students = 124) (h_stars_per_student : stars_per_student = 3) : students * stars_per_student = 372 := by
  sorry

end total_stars_l146_146212


namespace divisible_by_27000_l146_146249

theorem divisible_by_27000 (k : ℕ) (h₁ : k = 30) : ∃ n : ℕ, k^3 = 27000 * n :=
by {
  sorry
}

end divisible_by_27000_l146_146249


namespace geometric_series_common_ratio_l146_146799

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l146_146799


namespace additional_people_needed_l146_146265

-- Definition of the conditions
def person_hours (people: ℕ) (hours: ℕ) : ℕ := people * hours

-- Assertion that 8 people can paint the fence in 3 hours
def eight_people_three_hours : Prop := person_hours 8 3 = 24

-- Definition of the additional people required
def additional_people (initial_people required_people: ℕ) : ℕ := required_people - initial_people

-- Main theorem stating the problem
theorem additional_people_needed : eight_people_three_hours → additional_people 8 12 = 4 :=
by
  sorry

end additional_people_needed_l146_146265


namespace red_grapes_count_l146_146734

-- Definitions of variables and conditions
variables (G R Ra B P : ℕ)
variables (cond1 : R = 3 * G + 7)
variables (cond2 : Ra = G - 5)
variables (cond3 : B = 4 * Ra)
variables (cond4 : P = (1 / 2) * B + 5)
variables (cond5 : G + R + Ra + B + P = 350)

-- Theorem statement
theorem red_grapes_count : R = 100 :=
by sorry

end red_grapes_count_l146_146734


namespace p_sufficient_for_q_iff_l146_146136

-- Definitions based on conditions
def p (x : ℝ) : Prop := x^2 - 2 * x - 8 ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) ≤ 0
def m_condition (m : ℝ) : Prop := m < 0

-- The statement to prove
theorem p_sufficient_for_q_iff (m : ℝ) :
  (∀ x, p x → q x m) ↔ m <= -3 :=
by
  sorry

-- noncomputable theory is not necessary here since all required functions are computable.

end p_sufficient_for_q_iff_l146_146136


namespace children_attended_play_l146_146217

variables (A C : ℕ)

theorem children_attended_play
  (h1 : A + C = 610)
  (h2 : 2 * A + C = 960) : 
  C = 260 := 
by 
  -- Proof goes here
  sorry

end children_attended_play_l146_146217


namespace vehicle_combinations_count_l146_146162

theorem vehicle_combinations_count :
  ∃ (x y : ℕ), (4 * x + y = 79) ∧ (∃ (n : ℕ), n = 19) :=
sorry

end vehicle_combinations_count_l146_146162


namespace pi_minus_five_floor_value_l146_146970

noncomputable def greatest_integer_function (x : ℝ) : ℤ := Int.floor x

theorem pi_minus_five_floor_value :
  greatest_integer_function (Real.pi - 5) = -2 :=
by
  -- The proof is omitted
  sorry

end pi_minus_five_floor_value_l146_146970


namespace find_b_l146_146108

theorem find_b (b : ℝ) (x y : ℝ) (h1 : 2 * x^2 + b * x = 12) (h2 : y = x + 5.5) (h3 : y^2 * x + y * x^2 + y * (b * x) = 12) :
  b = -5 :=
sorry

end find_b_l146_146108


namespace gcd_of_128_144_480_is_16_l146_146389

-- Define the three numbers
def a := 128
def b := 144
def c := 480

-- Define the problem statement in Lean
theorem gcd_of_128_144_480_is_16 : Int.gcd (Int.gcd a b) c = 16 :=
by
  -- Definitions using given conditions
  -- use Int.gcd function to define the problem precisely.
  -- The proof will be left as "sorry" since we don't need to solve it
  sorry

end gcd_of_128_144_480_is_16_l146_146389


namespace slope_of_line_det_by_two_solutions_l146_146221

theorem slope_of_line_det_by_two_solutions (x y : ℝ) (h : 3 / x + 4 / y = 0) :
  (y = -4 * x / 3) → 
  ∀ x1 x2 y1 y2, (y1 = -4 * x1 / 3) ∧ (y2 = -4 * x2 / 3) → 
  (y2 - y1) / (x2 - x1) = -4 / 3 :=
sorry

end slope_of_line_det_by_two_solutions_l146_146221


namespace constant_term_binomial_expansion_l146_146965

theorem constant_term_binomial_expansion (n : ℕ) (hn : n = 6) :
  (2 : ℤ) * (x : ℝ) - (1 : ℤ) / (2 : ℝ) / (x : ℝ) ^ n == -20 := by
  sorry

end constant_term_binomial_expansion_l146_146965


namespace jake_sister_weight_ratio_l146_146731

theorem jake_sister_weight_ratio
  (jake_present_weight : ℕ)
  (total_weight : ℕ)
  (weight_lost : ℕ)
  (sister_weight : ℕ)
  (jake_weight_after_loss : ℕ)
  (ratio : ℕ) :
  jake_present_weight = 188 →
  total_weight = 278 →
  weight_lost = 8 →
  jake_weight_after_loss = jake_present_weight - weight_lost →
  sister_weight = total_weight - jake_present_weight →
  ratio = jake_weight_after_loss / sister_weight →
  ratio = 2 := by
  sorry

end jake_sister_weight_ratio_l146_146731


namespace correct_choice_l146_146963

theorem correct_choice
  (options : List String)
  (correct : String)
  (is_correct : correct = "that") :
  "The English spoken in the United States is only slightly different from ____ spoken in England." = 
  "The English spoken in the United States is only slightly different from that spoken in England." :=
by
  sorry

end correct_choice_l146_146963


namespace total_snakes_count_l146_146945

-- Define the basic conditions
def breedingBalls : Nat := 3
def snakesPerBall : Nat := 8
def pairsOfSnakes : Nat := 6
def snakesPerPair : Nat := 2

-- Define the total number of snakes
theorem total_snakes_count : 
  (breedingBalls * snakesPerBall) + (pairsOfSnakes * snakesPerPair) = 36 := 
by 
  -- we skip the proof with sorry
  sorry

end total_snakes_count_l146_146945


namespace smallest_n_l146_146003

def n_expr (n : ℕ) : ℕ :=
  n * (2^7) * (3^2) * (7^3)

theorem smallest_n (n : ℕ) (h1: 25 ∣ n_expr n) (h2: 27 ∣ n_expr n) : n = 75 :=
sorry

end smallest_n_l146_146003


namespace problem_l146_146563

noncomputable def x : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem problem (x y : ℝ) (hx : x = Real.sqrt 3 + Real.sqrt 2) (hy : y = Real.sqrt 3 - Real.sqrt 2) :
  x * y^2 - x^2 * y = -2 * Real.sqrt 2 :=
by
  rw [hx, hy]
  sorry

end problem_l146_146563


namespace johns_family_total_members_l146_146336

theorem johns_family_total_members (n_f : ℕ) (h_f : n_f = 10) (n_m : ℕ) (h_m : n_m = (13 * n_f) / 10) :
  n_f + n_m = 23 := by
  rw [h_f, h_m]
  norm_num
  sorry

end johns_family_total_members_l146_146336


namespace can_invent_1001_sad_stories_l146_146493

-- Definitions
def is_natural (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 17

def is_sad_story (a b c : ℕ) : Prop :=
  ∀ x y : ℤ, a * x + b * y ≠ c

-- The Statement
theorem can_invent_1001_sad_stories :
  ∃ stories : Finset (ℕ × ℕ × ℕ),
    (∀ (a b c : ℕ), (a, b, c) ∈ stories → is_natural a ∧ is_natural b ∧ is_natural c ∧ is_sad_story a b c) ∧
    stories.card ≥ 1001 :=
by
  sorry

end can_invent_1001_sad_stories_l146_146493


namespace solution_triple_root_system_l146_146771

theorem solution_triple_root_system (x y z : ℝ) :
  (x - 1) * (y - 1) * (z - 1) = x * y * z - 1 ∧
  (x - 2) * (y - 2) * (z - 2) = x * y * z - 2 →
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  intro h
  sorry

end solution_triple_root_system_l146_146771


namespace parabola_intercepts_min_max_l146_146908

noncomputable def f (θ : ℝ) : ℝ :=
  -((Real.sin θ + 2) ^ 2) + 7

theorem parabola_intercepts_min_max :
  ∀ θ : ℝ, (-(Real.sin θ + 2) ^ 2 + 7 ≥ -2) ∧ (-(Real.sin θ + 2) ^ 2 + 7 ≤ 6) :=
by
  intro θ
  split
  -- Proof for minimum value
  sorry
  -- Proof for maximum value
  sorry

end parabola_intercepts_min_max_l146_146908


namespace sequence_is_geometric_not_arithmetic_l146_146428

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

def is_geometric_sequence (a b c : ℕ) : Prop :=
  b / a = c / b

theorem sequence_is_geometric_not_arithmetic :
  ∀ (a₁ a₂ an : ℕ), a₁ = 3 ∧ a₂ = 9 ∧ an = 729 →
    ¬ is_arithmetic_sequence a₁ a₂ an ∧ is_geometric_sequence a₁ a₂ an :=
by
  intros a₁ a₂ an h
  sorry

end sequence_is_geometric_not_arithmetic_l146_146428


namespace largest_base5_to_base10_l146_146079

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l146_146079


namespace sector_area_l146_146446

theorem sector_area (alpha : ℝ) (r : ℝ) (h_alpha : alpha = Real.pi / 3) (h_r : r = 2) : 
  (1 / 2) * (alpha * r) * r = (2 * Real.pi) / 3 := 
by
  sorry

end sector_area_l146_146446


namespace kimberly_loan_l146_146752

theorem kimberly_loan :
  ∃ (t : ℕ), (1.06 : ℝ)^t > 3 ∧ ∀ (t' : ℕ), t' < t → (1.06 : ℝ)^t' ≤ 3 :=
by
sorry

end kimberly_loan_l146_146752


namespace initial_percentage_of_water_l146_146859

variable (V : ℝ) (W : ℝ) (P : ℝ)

theorem initial_percentage_of_water 
  (h1 : V = 120) 
  (h2 : W = 8)
  (h3 : (V + W) * 0.25 = ((P / 100) * V) + W) : 
  P = 20 :=
by
  sorry

end initial_percentage_of_water_l146_146859


namespace largest_base5_number_to_base10_is_3124_l146_146065

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l146_146065


namespace max_abs_sum_on_circle_l146_146578

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by sorry

end max_abs_sum_on_circle_l146_146578


namespace hypotenuse_length_l146_146298

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l146_146298


namespace sum_of_squares_of_roots_l146_146256

theorem sum_of_squares_of_roots (s1 s2 : ℝ) (h1 : s1 * s2 = 4) (h2 : s1 + s2 = 16) : s1^2 + s2^2 = 248 :=
by
  sorry

end sum_of_squares_of_roots_l146_146256


namespace fraction_beans_remain_l146_146815

theorem fraction_beans_remain (J B B_remain : ℝ) 
  (h1 : J = 0.10 * (J + B)) 
  (h2 : J + B_remain = 0.60 * (J + B)) : 
  B_remain / B = 5 / 9 := 
by 
  sorry

end fraction_beans_remain_l146_146815


namespace complete_the_square_3x2_9x_20_l146_146594

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l146_146594


namespace part_I_part_II_l146_146138

noncomputable def a (n : Nat) : Nat := sorry

def is_odd (n : Nat) : Prop := n % 2 = 1

theorem part_I
  (h : a 1 = 19) :
  a 2014 = 98 := by
  sorry

theorem part_II
  (h1: ∀ n : Nat, is_odd (a n))
  (h2: ∀ n m : Nat, a n = a m) -- constant sequence
  (h3: ∀ n : Nat, a n > 1) :
  ∃ k : Nat, a k = 5 := by
  sorry


end part_I_part_II_l146_146138


namespace geometric_series_common_ratio_l146_146795

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l146_146795


namespace find_prime_triple_l146_146124

def is_prime (n : ℕ) : Prop := n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_prime_triple :
  ∃ (I M C : ℕ), is_prime I ∧ is_prime M ∧ is_prime C ∧ I ≤ M ∧ M ≤ C ∧ 
  I * M * C = I + M + C + 1007 ∧ (I = 2 ∧ M = 2 ∧ C = 337) :=
by
  sorry

end find_prime_triple_l146_146124


namespace solve_for_y_l146_146729

theorem solve_for_y (y : ℝ) (h : 9 / y^3 = y / 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end solve_for_y_l146_146729


namespace probability_of_c_between_l146_146632

noncomputable def probability_c_between (a b : ℝ) (hab : 0 < a ∧ a ≤ 1 ∧ 0 < b ∧ b ≤ 1) : ℝ :=
  let c := a / (a + b)
  if (1 / 4 : ℝ) ≤ c ∧ c ≤ (3 / 4 : ℝ) then sorry else sorry
  
theorem probability_of_c_between (a b : ℝ) (hab : 0 < a ∧ a ≤ 1 ∧ 0 < b ∧ b ≤ 1) : 
  probability_c_between a b hab = (2 / 3 : ℝ) :=
sorry

end probability_of_c_between_l146_146632


namespace heesu_has_greatest_sum_l146_146361

theorem heesu_has_greatest_sum :
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  have h1 : Heesu_sum > Sora_sum := by sorry
  have h2 : Heesu_sum > Jiyeon_sum := by sorry
  exact And.intro h1 h2

end heesu_has_greatest_sum_l146_146361


namespace find_e_l146_146272

theorem find_e (x y e : ℝ) (h1 : x / (2 * y) = 5 / e) (h2 : (7 * x + 4 * y) / (x - 2 * y) = 13) : e = 2 := 
by
  sorry

end find_e_l146_146272


namespace quadratic_polynomial_P_l146_146487

noncomputable def P : Polynomial := {
  to_fun := λ x : ℝ, x^2 - x + 2,
  degree := 2
}

theorem quadratic_polynomial_P (P : ℝ → ℝ) 
  (h : ∀ x, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) : 
  P 8 = 58 := 
by
  sorry

end quadratic_polynomial_P_l146_146487


namespace slope_of_line_between_solutions_l146_146223

theorem slope_of_line_between_solutions (x1 y1 x2 y2 : ℝ) (h1 : 3 / x1 + 4 / y1 = 0) (h2 : 3 / x2 + 4 / y2 = 0) (h3 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -4 / 3 := 
sorry

end slope_of_line_between_solutions_l146_146223


namespace profit_per_meter_is_20_l146_146103

-- Define given conditions
def selling_price_total (n : ℕ) (price : ℕ) : ℕ := n * price
def cost_price_per_meter : ℕ := 85
def selling_price_total_85_meters : ℕ := 8925

-- Define the expected profit per meter
def expected_profit_per_meter : ℕ := 20

-- Rewrite the problem statement: Prove that with given conditions the profit per meter is Rs. 20
theorem profit_per_meter_is_20 
  (n : ℕ := 85)
  (sp : ℕ := selling_price_total_85_meters)
  (cp_pm : ℕ := cost_price_per_meter) 
  (expected_profit : ℕ := expected_profit_per_meter) :
  (sp - n * cp_pm) / n = expected_profit :=
by
  sorry

end profit_per_meter_is_20_l146_146103


namespace remainder_7n_mod_4_l146_146838

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by sorry

end remainder_7n_mod_4_l146_146838


namespace problem_proof_l146_146481

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 2

theorem problem_proof : f (1 + g 3) = 32 := by
  sorry

end problem_proof_l146_146481


namespace ellipse_properties_l146_146139

theorem ellipse_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b ≥ 0)
  (e : ℝ)
  (hc : e = 4 / 5)
  (directrix : ℝ)
  (hd : directrix = 25 / 4)
  (x y : ℝ)
  (hx : (x - 6)^2 / 25 + (y - 6)^2 / 9 = 1) :
  x^2 / 25 + y^2 / 9 = 1 :=
sorry

end ellipse_properties_l146_146139


namespace remainder_of_7n_div_4_l146_146833

theorem remainder_of_7n_div_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
sorry

end remainder_of_7n_div_4_l146_146833


namespace total_balloons_after_gift_l146_146057

-- Definitions for conditions
def initial_balloons := 26
def additional_balloons := 34

-- Proposition for the total number of balloons
theorem total_balloons_after_gift : initial_balloons + additional_balloons = 60 := 
by
  -- Proof omitted, adding sorry
  sorry

end total_balloons_after_gift_l146_146057


namespace tree_distance_l146_146266

theorem tree_distance 
  (num_trees : ℕ) (dist_first_to_fifth : ℕ) (length_of_road : ℤ) 
  (h1 : num_trees = 8) 
  (h2 : dist_first_to_fifth = 100) 
  (h3 : length_of_road = (dist_first_to_fifth * (num_trees - 1)) / 4 + 3 * dist_first_to_fifth) 
  :
  length_of_road = 175 := 
sorry

end tree_distance_l146_146266


namespace melanie_total_plums_l146_146183

namespace Melanie

def initial_plums : ℝ := 7.0
def plums_given_by_sam : ℝ := 3.0

theorem melanie_total_plums : initial_plums + plums_given_by_sam = 10.0 :=
by
  sorry

end Melanie

end melanie_total_plums_l146_146183


namespace cost_of_each_taco_l146_146093

variables (T E : ℝ)

-- Conditions
axiom condition1 : 2 * T + 3 * E = 7.80
axiom condition2 : 3 * T + 5 * E = 12.70

-- Question to prove
theorem cost_of_each_taco : T = 0.90 :=
by
  sorry

end cost_of_each_taco_l146_146093


namespace sum_of_inscribed_angles_l146_146045

theorem sum_of_inscribed_angles 
  (n : ℕ) 
  (total_degrees : ℝ)
  (arcs : ℕ)
  (x_arcs : ℕ)
  (y_arcs : ℕ) 
  (arc_angle : ℝ)
  (x_central_angle : ℝ)
  (y_central_angle : ℝ)
  (x_inscribed_angle : ℝ)
  (y_inscribed_angle : ℝ)
  (total_inscribed_angles : ℝ) :
  n = 18 →
  total_degrees = 360 →
  x_arcs = 3 →
  y_arcs = 5 →
  arc_angle = total_degrees / n →
  x_central_angle = x_arcs * arc_angle →
  y_central_angle = y_arcs * arc_angle →
  x_inscribed_angle = x_central_angle / 2 →
  y_inscribed_angle = y_central_angle / 2 →
  total_inscribed_angles = x_inscribed_angle + y_inscribed_angle →
  total_inscribed_angles = 80 := sorry

end sum_of_inscribed_angles_l146_146045


namespace range_of_m_l146_146281

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x then (1 / 3)^(-x) - 2 
  else 2 * Real.log x / Real.log 3

theorem range_of_m :
  {m : ℝ | f m > 1} = {m : ℝ | m < -Real.sqrt 3} ∪ {m : ℝ | 1 < m} :=
by
  sorry

end range_of_m_l146_146281


namespace seq_a22_l146_146982

def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0) ∧
  (a 10 = 10)

theorem seq_a22 : ∀ (a : ℕ → ℝ), seq a → a 22 = 10 :=
by
  intros a h,
  have h1 := h.1,
  have h99 := h.2.1,
  have h100 := h.2.2.1,
  have h_eq := h.2.2.2,
  sorry

end seq_a22_l146_146982


namespace time_period_principal_1000_amount_1120_interest_5_l146_146886

-- Definitions based on the conditions
def principal : ℝ := 1000
def amount : ℝ := 1120
def interest_rate : ℝ := 0.05

-- Lean 4 statement asserting the time period
theorem time_period_principal_1000_amount_1120_interest_5
  (P : ℝ) (A : ℝ) (r : ℝ) (T : ℝ) 
  (hP : P = principal)
  (hA : A = amount)
  (hr : r = interest_rate) :
  (A - P) * 100 / (P * r * 100) = 2.4 :=
by 
  -- The proof is filled in by 'sorry'
  sorry

end time_period_principal_1000_amount_1120_interest_5_l146_146886


namespace same_color_probability_is_correct_l146_146725

-- Define the variables and conditions
def total_sides : ℕ := 12
def pink_sides : ℕ := 3
def green_sides : ℕ := 4
def blue_sides : ℕ := 5

-- Calculate individual probabilities
def pink_probability : ℚ := (pink_sides : ℚ) / total_sides
def green_probability : ℚ := (green_sides : ℚ) / total_sides
def blue_probability : ℚ := (blue_sides : ℚ) / total_sides

-- Calculate the probabilities that both dice show the same color
def both_pink_probability : ℚ := pink_probability ^ 2
def both_green_probability : ℚ := green_probability ^ 2
def both_blue_probability : ℚ := blue_probability ^ 2

-- The final probability that both dice come up the same color
def same_color_probability : ℚ := both_pink_probability + both_green_probability + both_blue_probability

theorem same_color_probability_is_correct : same_color_probability = 25 / 72 := by
  sorry

end same_color_probability_is_correct_l146_146725


namespace largest_base5_number_to_base10_is_3124_l146_146064

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l146_146064


namespace find_h_l146_146605

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l146_146605


namespace Heesu_has_greatest_sum_l146_146359

-- Define the numbers collected by each individual
def Sora_collected : (Nat × Nat) := (4, 6)
def Heesu_collected : (Nat × Nat) := (7, 5)
def Jiyeon_collected : (Nat × Nat) := (3, 8)

-- Calculate the sums
def Sora_sum : Nat := Sora_collected.1 + Sora_collected.2
def Heesu_sum : Nat := Heesu_collected.1 + Heesu_collected.2
def Jiyeon_sum : Nat := Jiyeon_collected.1 + Jiyeon_collected.2

-- The theorem to prove that Heesu has the greatest sum
theorem Heesu_has_greatest_sum :
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  sorry

end Heesu_has_greatest_sum_l146_146359


namespace monotonic_subsequence_exists_l146_146899

theorem monotonic_subsequence_exists (n : ℕ) (a : Fin ((2^n : ℕ) + 1) → ℕ)
  (h : ∀ k : Fin (2^n + 1), a k ≤ k.val) : 
  ∃ (b : Fin (n + 2) → Fin (2^n + 1)),
    (∀ i j : Fin (n + 2), i ≤ j → b i ≤ b j) ∧
    (∀ i j : Fin (n + 2), i < j → a (b i) ≤ a ( b j)) :=
by
  sorry

end monotonic_subsequence_exists_l146_146899


namespace sum_even_then_diff_even_sum_odd_then_diff_odd_l146_146159

theorem sum_even_then_diff_even (a b : ℤ) (h : (a + b) % 2 = 0) : (a - b) % 2 = 0 := by
  sorry

theorem sum_odd_then_diff_odd (a b : ℤ) (h : (a + b) % 2 = 1) : (a - b) % 2 = 1 := by
  sorry

end sum_even_then_diff_even_sum_odd_then_diff_odd_l146_146159


namespace common_ratio_of_geometric_series_l146_146812

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l146_146812


namespace solution_interval_l146_146552

theorem solution_interval (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ (5 / 2 : ℝ) < x ∧ x ≤ (14 / 5 : ℝ) := 
by
  sorry

end solution_interval_l146_146552


namespace volume_between_spheres_l146_146505

theorem volume_between_spheres (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 8) : 
  (4 / 3) * Real.pi * (r_large ^ 3) - (4 / 3) * Real.pi * (r_small ^ 3) = (1792 / 3) * Real.pi := 
by
  rw [h_small, h_large]
  sorry

end volume_between_spheres_l146_146505


namespace other_root_of_quadratic_eq_l146_146427

namespace QuadraticEquation

variables {a b c : ℝ}

theorem other_root_of_quadratic_eq
  (h : ∀ x, a * (b + c) * x^2 - b * (c + a) * x - c * (a + b) = 0)
  (root1 : -1) :
  ∃ root2, root2 = (c * (a + b)) / (a * (b + c)) := by
  sorry

end QuadraticEquation

end other_root_of_quadratic_eq_l146_146427


namespace equivalent_problem_l146_146476

noncomputable def problem_statement : Prop :=
  ∀ (a b c d : ℝ), a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ∀ (ω : ℂ), ω^4 = 1 → ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / (1 + ω)) →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2)

#check problem_statement

-- Expected output for type checking without providing the proof
theorem equivalent_problem : problem_statement :=
  sorry

end equivalent_problem_l146_146476


namespace original_square_perimeter_l146_146244

theorem original_square_perimeter (P : ℝ) (x : ℝ) (h1 : 4 * x * 2 + 4 * x = 56) : P = 32 :=
by
  sorry

end original_square_perimeter_l146_146244


namespace garden_area_l146_146530

theorem garden_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end garden_area_l146_146530


namespace sum_of_areas_of_circles_l146_146986

-- Definitions and given conditions
variables (r s t : ℝ)
variables (h1 : r + s = 5)
variables (h2 : r + t = 12)
variables (h3 : s + t = 13)

-- The sum of the areas
theorem sum_of_areas_of_circles : 
  π * r^2 + π * s^2 + π * t^2 = 113 * π :=
  by
    sorry

end sum_of_areas_of_circles_l146_146986


namespace mat_weaves_problem_l146_146643

theorem mat_weaves_problem (S1 S2: ℕ) (days1 days2: ℕ) (mats1 mats2: ℕ) (H1: S1 = 1)
    (H2: S2 = 8) (H3: days1 = 4) (H4: days2 = 8) (H5: mats1 = 4) (H6: mats2 = 16) 
    (rate_consistency: (mats1 / days1) = (mats2 / days2 / S2)): S1 = 4 := 
by
  sorry

end mat_weaves_problem_l146_146643


namespace hyperbola_vertices_distance_l146_146884

noncomputable def distance_between_vertices : ℝ :=
  2 * Real.sqrt 7.5

theorem hyperbola_vertices_distance :
  ∀ (x y : ℝ), 4 * x^2 - 24 * x - y^2 + 6 * y - 3 = 0 →
  distance_between_vertices = 2 * Real.sqrt 7.5 :=
by sorry

end hyperbola_vertices_distance_l146_146884


namespace simplify_fraction_result_l146_146962

theorem simplify_fraction_result : (130 / 16900) * 65 = 1 / 2 :=
by sorry

end simplify_fraction_result_l146_146962


namespace total_cats_in_center_l146_146109

def cats_training_center : ℕ := 45
def cats_can_fetch : ℕ := 25
def cats_can_meow : ℕ := 40
def cats_jump_and_fetch : ℕ := 15
def cats_fetch_and_meow : ℕ := 20
def cats_jump_and_meow : ℕ := 23
def cats_all_three : ℕ := 10
def cats_none : ℕ := 5

theorem total_cats_in_center :
  (cats_training_center - (cats_jump_and_fetch + cats_jump_and_meow - cats_all_three)) +
  (cats_all_three) +
  (cats_fetch_and_meow - cats_all_three) +
  (cats_jump_and_fetch - cats_all_three) +
  (cats_jump_and_meow - cats_all_three) +
  cats_none = 67 := by
  sorry

end total_cats_in_center_l146_146109


namespace abc_divisible_by_7_l146_146373

theorem abc_divisible_by_7 (a b c : ℤ) (h : 7 ∣ (a^3 + b^3 + c^3)) : 7 ∣ (a * b * c) :=
sorry

end abc_divisible_by_7_l146_146373


namespace paul_reading_novel_l146_146189

theorem paul_reading_novel (x : ℕ) 
  (h1 : x - ((1 / 6) * x + 10) - ((1 / 5) * (x - ((1 / 6) * x + 10)) + 14) - ((1 / 4) * ((x - ((1 / 6) * x + 10) - ((1 / 5) * (x - ((1 / 6) * x + 10)) + 14)) + 16)) = 48) : 
  x = 161 :=
by sorry

end paul_reading_novel_l146_146189


namespace complete_the_square_3x2_9x_20_l146_146596

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l146_146596


namespace simplify_fraction_l146_146640

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l146_146640


namespace two_thirds_of_5_times_9_l146_146707

theorem two_thirds_of_5_times_9 : (2 / 3) * (5 * 9) = 30 :=
by
  sorry

end two_thirds_of_5_times_9_l146_146707


namespace geometric_series_common_ratio_l146_146797

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l146_146797


namespace binomial_middle_term_coefficient_l146_146332

noncomputable def middle_term_coefficient (n : ℕ) : ℤ :=
  (nat.choose n (n / 2)) * ((-2) ^ (n / 2))

theorem binomial_middle_term_coefficient :
  ∀ n : ℕ, (∑ k in finset.range (n / 2 + 1), (nat.choose n (2 * k)) * (1 - 2 * 0)^ (n - 2 * k) * (-2) ^ (2 * k)) = 128 →
  n = 8 →
  middle_term_coefficient n = 1120 :=
by
  intros n h1 h2
  rw h2 at *
  simp [middle_term_coefficient, nat.choose]
  sorry

end binomial_middle_term_coefficient_l146_146332


namespace base5_to_base10_max_l146_146082

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l146_146082


namespace remainder_7n_mod_4_l146_146836

theorem remainder_7n_mod_4 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := sorry

end remainder_7n_mod_4_l146_146836


namespace kimberly_skittles_proof_l146_146338

variable (SkittlesInitial : ℕ) (SkittlesBought : ℕ) (OrangesBought : ℕ)

/-- Kimberly's initial number of Skittles --/
def kimberly_initial_skittles := SkittlesInitial

/-- Skittles Kimberly buys --/
def kimberly_skittles_bought := SkittlesBought

/-- Oranges Kimbery buys (irrelevant for Skittles count) --/
def kimberly_oranges_bought := OrangesBought

/-- Total Skittles Kimberly has --/
def kimberly_total_skittles (SkittlesInitial SkittlesBought : ℕ) : ℕ :=
  SkittlesInitial + SkittlesBought

/-- Proof statement --/
theorem kimberly_skittles_proof (h1 : SkittlesInitial = 5) (h2 : SkittlesBought = 7) : 
  kimberly_total_skittles SkittlesInitial SkittlesBought = 12 :=
by
  rw [h1, h2]
  exact rfl

end kimberly_skittles_proof_l146_146338


namespace polygon_sides_l146_146352

-- Define the conditions
def sum_interior_angles (x : ℕ) : ℝ := 180 * (x - 2)
def sum_given_angles (x : ℕ) : ℝ := 160 + 112 * (x - 1)

-- State the theorem
theorem polygon_sides (x : ℕ) (h : sum_interior_angles x = sum_given_angles x) : x = 6 := by
  sorry

end polygon_sides_l146_146352


namespace henry_wins_l146_146452

-- Definitions of conditions
def total_games : ℕ := 14
def losses : ℕ := 2
def draws : ℕ := 10

-- Statement of the theorem
theorem henry_wins : (total_games - losses - draws) = 2 :=
by
  -- Proof goes here
  sorry

end henry_wins_l146_146452


namespace prob_X_gt_2_l146_146720

noncomputable def X : ℝ → ℝ := sorry
axiom X_normal : ∀ (x : ℝ), X(x) = pdf (NormalDist.mk 1 4) x

theorem prob_X_gt_2 :
  (∀ x, P(0 ≤ X ≤ 2) = 0.68) →
  P (X > 2) = 0.16 := 
by sorry

end prob_X_gt_2_l146_146720


namespace find_positive_k_l146_146160

noncomputable def polynomial_with_equal_roots (k: ℚ) : Prop := 
  ∃ a b : ℚ, a ≠ b ∧ 2 * a + b = -3 ∧ 2 * a * b + a^2 = -50 ∧ k = -2 * a^2 * b

theorem find_positive_k : ∃ k : ℚ, polynomial_with_equal_roots k ∧ 0 < k ∧ k = 950 / 27 :=
by
  sorry

end find_positive_k_l146_146160


namespace largest_base_5_five_digit_number_in_decimal_l146_146074

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l146_146074


namespace common_ratio_of_geometric_series_l146_146794

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l146_146794


namespace min_41x_2y_eq_nine_l146_146565

noncomputable def min_value_41x_2y (x y : ℝ) : ℝ :=
  41*x + 2*y

theorem min_41x_2y_eq_nine (x y : ℝ) (h : ∀ n : ℕ, 0 < n →  n*x + (1/n)*y ≥ 1) :
  min_value_41x_2y x y = 9 :=
sorry

end min_41x_2y_eq_nine_l146_146565


namespace prime_iff_even_and_power_of_two_l146_146020

theorem prime_iff_even_and_power_of_two (a n : ℕ) (h_pos_a : a > 1) (h_pos_n : n > 0) :
  Nat.Prime (a^n + 1) → (∃ k : ℕ, a = 2 * k) ∧ (∃ m : ℕ, n = 2^m) :=
by 
  sorry

end prime_iff_even_and_power_of_two_l146_146020


namespace sum_d_e_f_equals_23_l146_146648

theorem sum_d_e_f_equals_23
  (d e f : ℤ)
  (h1 : ∀ x : ℝ, x^2 + 9 * x + 20 = (x + d) * (x + e))
  (h2 : ∀ x : ℝ, x^2 + 11 * x - 60 = (x + e) * (x - f)) :
  d + e + f = 23 :=
by
  sorry

end sum_d_e_f_equals_23_l146_146648


namespace age_difference_l146_146209

variable (A B C : ℕ)

-- Conditions: C is 11 years younger than A
axiom h1 : C = A - 11

-- Statement: Prove the difference (A + B) - (B + C) is 11
theorem age_difference : (A + B) - (B + C) = 11 := by
  sorry

end age_difference_l146_146209


namespace Diego_annual_savings_l146_146264

theorem Diego_annual_savings :
  let monthly_income := 5000
  let monthly_expenses := 4600
  let monthly_savings := monthly_income - monthly_expenses
  let months_in_year := 12
  let annual_savings := monthly_savings * months_in_year
  annual_savings = 4800 :=
by
  -- Definitions based on the conditions and required result
  let monthly_income := 5000
  let monthly_expenses := 4600
  let monthly_savings := monthly_income - monthly_expenses
  let months_in_year := 12
  let annual_savings := monthly_savings * months_in_year

  -- Assertion to check the correctness of annual savings
  have h : annual_savings = 4800 := by
    have h1 : monthly_savings = monthly_income - monthly_expenses := rfl
    have h2 : monthly_savings = 400 := by simp [monthly_income, monthly_expenses, h1]
    have h3 : annual_savings = monthly_savings * months_in_year := rfl
    simp [h2, months_in_year, h3]
  exact h

end Diego_annual_savings_l146_146264


namespace find_x_l146_146450

-- Definitions for the problem
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b : ℝ × ℝ := (-2, 1)

-- Theorem statement
theorem find_x (x : ℝ) (h : ∃ k : ℝ, a x = k • b) : x = -1/2 := by
  sorry

end find_x_l146_146450


namespace unique_function_satisfying_conditions_l146_146340

open Nat

theorem unique_function_satisfying_conditions (k : ℕ) (h : 0 < k) : 
  ∀ (f : ℕ → ℕ),
  (∃ᶠ p in at_top, ∃ c : ℕ, f(c) = p ^ k) ∧ 
  (∀ m n : ℕ, f(m) + f(n) ∣ f(m + n)) ->
  (∀ n : ℕ, f(n) = n) :=
by
  sorry

end unique_function_satisfying_conditions_l146_146340


namespace find_starting_number_of_range_l146_146211

theorem find_starting_number_of_range :
  ∃ x, (∀ n, 0 ≤ n ∧ n < 10 → 65 - 5 * n = x + 5 * (9 - n)) ∧ x = 15 := 
by
  sorry

end find_starting_number_of_range_l146_146211


namespace area_difference_of_square_screens_l146_146657

theorem area_difference_of_square_screens (d1 d2 : ℝ) (A1 A2 : ℝ) 
  (h1 : d1 = 18) (h2 : d2 = 16) 
  (hA1 : A1 = d1^2 / 2) (hA2 : A2 = d2^2 / 2) : 
  A1 - A2 = 34 := by
  sorry

end area_difference_of_square_screens_l146_146657


namespace water_formed_l146_146435

theorem water_formed (n_HCl : ℕ) (n_CaCO3: ℕ) (n_H2O: ℕ) 
  (balance_eqn: ∀ (n : ℕ), 
    (2 * n_CaCO3) ≤ n_HCl ∧
    n_H2O = n_CaCO3 ):
  n_HCl = 4 ∧ n_CaCO3 = 2 → n_H2O = 2 :=
by
  intros h0
  obtain ⟨h1, h2⟩ := h0
  sorry

end water_formed_l146_146435


namespace h_value_l146_146607

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l146_146607


namespace six_people_theorem_l146_146767

theorem six_people_theorem  {G : SimpleGraph (Fin 6)} :
  ∀ (color : G.Edge → Prop),
  (∀ e, color e ∨ ¬ color e) →
  ∃ (x y z : Fin 6), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  ((color (G.edge x y) ∧ color (G.edge y z) ∧ color (G.edge x z)) ∨
   (¬color (G.edge x y) ∧ ¬color (G.edge y z) ∧ ¬color (G.edge x z))) :=
begin
  sorry
end

end six_people_theorem_l146_146767


namespace work_completion_time_for_A_l146_146230

theorem work_completion_time_for_A 
  (B_work_rate : ℝ)
  (combined_work_rate : ℝ)
  (x : ℝ) 
  (B_work_rate_def : B_work_rate = 1 / 6)
  (combined_work_rate_def : combined_work_rate = 3 / 10) :
  (1 / x) + B_work_rate = combined_work_rate →
  x = 7.5 := 
by
  sorry

end work_completion_time_for_A_l146_146230


namespace incenter_x_coordinate_eq_l146_146689

theorem incenter_x_coordinate_eq (x y : ℝ) :
  (x = y) ∧ 
  (y = -x + 3) → 
  x = 3 / 2 := 
sorry

end incenter_x_coordinate_eq_l146_146689


namespace marked_price_correct_l146_146528

noncomputable def marked_price (original_price discount_percent purchase_price profit_percent final_price_percent : ℝ) := 
  (purchase_price * (1 + profit_percent)) / final_price_percent

theorem marked_price_correct
  (original_price : ℝ)
  (discount_percent : ℝ)
  (profit_percent : ℝ)
  (final_price_percent : ℝ)
  (purchase_price : ℝ := original_price * (1 - discount_percent))
  (expected_marked_price : ℝ) :
  original_price = 40 →
  discount_percent = 0.15 →
  profit_percent = 0.25 →
  final_price_percent = 0.90 →
  expected_marked_price = 47.20 →
  marked_price original_price discount_percent purchase_price profit_percent final_price_percent = expected_marked_price := 
by
  intros
  sorry

end marked_price_correct_l146_146528


namespace total_flour_needed_l146_146171

-- Definitions of flour needed by Katie and Sheila
def katie_flour : ℕ := 3
def sheila_flour : ℕ := katie_flour + 2

-- Statement of the theorem
theorem total_flour_needed : katie_flour + sheila_flour = 8 := by
  -- The proof would go here
  sorry

end total_flour_needed_l146_146171


namespace spring_length_relationship_maximum_mass_l146_146703

theorem spring_length_relationship (x y : ℝ) : 
  (y = 0.5 * x + 12) ↔ y = 12 + 0.5 * x := 
by sorry

theorem maximum_mass (x y : ℝ) : 
  (y = 0.5 * x + 12) → (y ≤ 20) → (x ≤ 16) :=
by sorry

end spring_length_relationship_maximum_mass_l146_146703


namespace common_ratio_of_geometric_series_l146_146813

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l146_146813


namespace action_figure_value_l146_146169

theorem action_figure_value (
    V1 V2 V3 V4 : ℝ
) : 5 * 15 = 75 ∧ 
    V1 - 5 + V2 - 5 + V3 - 5 + V4 - 5 + (20 - 5) = 55 ∧
    V1 + V2 + V3 + V4 + 20 = 80 → 
    ∀ i, i = 15 := by
    sorry

end action_figure_value_l146_146169


namespace nadine_spent_money_l146_146952

theorem nadine_spent_money (table_cost : ℕ) (chair_cost : ℕ) (num_chairs : ℕ) 
    (h_table_cost : table_cost = 34) 
    (h_chair_cost : chair_cost = 11) 
    (h_num_chairs : num_chairs = 2) : 
    table_cost + num_chairs * chair_cost = 56 :=
by
  sorry

end nadine_spent_money_l146_146952


namespace hypotenuse_length_l146_146307

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l146_146307


namespace max_abs_sum_on_circle_l146_146579

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by sorry

end max_abs_sum_on_circle_l146_146579


namespace cost_of_each_pair_of_jeans_l146_146339

-- Conditions
def costWallet : ℕ := 50
def costSneakers : ℕ := 100
def pairsSneakers : ℕ := 2
def costBackpack : ℕ := 100
def totalSpent : ℕ := 450
def pairsJeans : ℕ := 2

-- Definitions
def totalSpentLeonard := costWallet + pairsSneakers * costSneakers
def totalSpentMichaelWithoutJeans := costBackpack

-- Goal: Prove the cost of each pair of jeans
theorem cost_of_each_pair_of_jeans :
  let totalCostJeans := totalSpent - (totalSpentLeonard + totalSpentMichaelWithoutJeans)
  let costPerPairJeans := totalCostJeans / pairsJeans
  costPerPairJeans = 50 :=
by
  intros
  let totalCostJeans := totalSpent - (totalSpentLeonard + totalSpentMichaelWithoutJeans)
  let costPerPairJeans := totalCostJeans / pairsJeans
  show costPerPairJeans = 50
  sorry

end cost_of_each_pair_of_jeans_l146_146339


namespace hypotenuse_length_l146_146299

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l146_146299


namespace coefficient_of_x_neg_2_in_binomial_expansion_l146_146366

theorem coefficient_of_x_neg_2_in_binomial_expansion :
  let x := (x : ℚ)
  let term := (x^3 - (2 / x))^6
  (coeff_of_term : Int) ->
  (coeff_of_term = -192) :=
by
  -- Placeholder for the proof
  sorry

end coefficient_of_x_neg_2_in_binomial_expansion_l146_146366


namespace total_snakes_l146_146949

/-
  Problem: Mary sees three breeding balls with 8 snakes each and 6 additional pairs of snakes.
           How many snakes did she see total?
  Conditions:
    - There are 3 breeding balls.
    - Each breeding ball has 8 snakes.
    - There are 6 additional pairs of snakes.
  Answer: 36 snakes
-/

theorem total_snakes (balls : ℕ) (snakes_per_ball : ℕ) (pairs : ℕ) (snakes_per_pair : ℕ) :
    balls = 3 → snakes_per_ball = 8 → pairs = 6 → snakes_per_pair = 2 →
    (balls * snakes_per_ball) + (pairs * snakes_per_pair) = 36 :=
  by 
    intros hb hspb hp hsp
    sorry

end total_snakes_l146_146949


namespace leoCurrentWeight_l146_146288

def currentWeightProblem (L K : Real) : Prop :=
  (L + 15 = 1.75 * K) ∧ (L + K = 250)

theorem leoCurrentWeight (L K : Real) (h : currentWeightProblem L K) : L = 154 :=
by
  sorry

end leoCurrentWeight_l146_146288


namespace solve_inequalities_l146_146260

theorem solve_inequalities (x : ℝ) (h₁ : 5 * x - 8 > 12 - 2 * x) (h₂ : |x - 1| ≤ 3) : 
  (20 / 7) < x ∧ x ≤ 4 :=
by
  sorry

end solve_inequalities_l146_146260


namespace product_of_roots_l146_146293

noncomputable def is_root (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem product_of_roots :
  ∀ (x1 x2 : ℝ), is_root 1 (-4) 3 x1 ∧ is_root 1 (-4) 3 x2 ∧ x1 ≠ x2 → x1 * x2 = 3 :=
by
  intros x1 x2 h
  sorry

end product_of_roots_l146_146293


namespace range_of_f_l146_146831

theorem range_of_f (x : ℝ) (h : x ∈ Set.Icc (-3 : ℝ) 3) : 
  ∃ y, y ∈ Set.Icc (0 : ℝ) 25 ∧ ∀ z, z = (x^2 - 4*x + 4) → y = z :=
sorry

end range_of_f_l146_146831


namespace jars_needed_l146_146469

def hives : ℕ := 5
def honey_per_hive : ℕ := 20
def jar_capacity : ℝ := 0.5
def friend_ratio : ℝ := 0.5

theorem jars_needed : (hives * honey_per_hive) / 2 / jar_capacity = 100 := 
by sorry

end jars_needed_l146_146469


namespace money_problem_l146_146133

variable (a b : ℝ)

theorem money_problem (h1 : 4 * a + b = 68) 
                      (h2 : 2 * a - b < 16) 
                      (h3 : a + b > 22) : 
                      a < 14 ∧ b > 12 := 
by 
  sorry

end money_problem_l146_146133


namespace number_of_integers_divisible_by_18_or_21_but_not_both_l146_146912

theorem number_of_integers_divisible_by_18_or_21_but_not_both :
  let num_less_2019_div_by_18 := 112
  let num_less_2019_div_by_21 := 96
  let num_less_2019_div_by_both := 16
  num_less_2019_div_by_18 + num_less_2019_div_by_21 - 2 * num_less_2019_div_by_both = 176 :=
by
  sorry

end number_of_integers_divisible_by_18_or_21_but_not_both_l146_146912


namespace radius_of_scrap_cookie_l146_146029

theorem radius_of_scrap_cookie
  (r_cookies : ℝ) (n_cookies : ℕ) (radius_layout : Prop)
  (circle_diameter_twice_width : Prop) :
  (r_cookies = 0.5 ∧ n_cookies = 9 ∧ radius_layout ∧ circle_diameter_twice_width)
  →
  (∃ r_scrap : ℝ, r_scrap = Real.sqrt 6.75) :=
by
  sorry

end radius_of_scrap_cookie_l146_146029


namespace min_value_expression_l146_146558

noncomputable def expression (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2)

theorem min_value_expression : ∃ x : ℝ, expression x = 2 * Real.sqrt 5 :=
by
  sorry

end min_value_expression_l146_146558


namespace dummies_remainder_l146_146433

/-
  Prove that if the number of Dummies in one bag is such that when divided among 10 kids, 3 pieces are left over,
  then the number of Dummies in four bags when divided among 10 kids leaves 2 pieces.
-/
theorem dummies_remainder (n : ℕ) (h : n % 10 = 3) : (4 * n) % 10 = 2 := 
by {
  sorry
}

end dummies_remainder_l146_146433


namespace determine_phi_l146_146498

theorem determine_phi (phi : ℝ) (h : 0 < phi ∧ phi < π) :
  (∃ k : ℤ, phi = 2*k*π + (3*π/4)) :=
by
  sorry

end determine_phi_l146_146498


namespace percentage_tax_raise_expecting_population_l146_146403

def percentage_affirmative_responses_tax : ℝ := 0.4
def percentage_affirmative_responses_money : ℝ := 0.3
def percentage_affirmative_responses_bonds : ℝ := 0.5
def percentage_affirmative_responses_gold : ℝ := 0.0

def fraction_liars : ℝ := 0.1
def fraction_economists : ℝ := 1 - fraction_liars

theorem percentage_tax_raise_expecting_population : 
  (percentage_affirmative_responses_tax - fraction_liars) = 0.3 :=
by
  sorry

end percentage_tax_raise_expecting_population_l146_146403


namespace pyramid_volume_is_232_l146_146863

noncomputable def pyramid_volume (length : ℝ) (width : ℝ) (slant_height : ℝ) : ℝ :=
  (1 / 3) * (length * width) * (Real.sqrt ((slant_height)^2 - ((length / 2)^2 + (width / 2)^2)))

theorem pyramid_volume_is_232 :
  pyramid_volume 5 10 15 = 232 := 
by
  sorry

end pyramid_volume_is_232_l146_146863


namespace reciprocal_of_sum_of_fraction_l146_146150

theorem reciprocal_of_sum_of_fraction (y : ℚ) (h : y = 6 + 1/6) : 1 / y = 6 / 37 := by
  sorry

end reciprocal_of_sum_of_fraction_l146_146150


namespace ratio_slices_l146_146875

theorem ratio_slices (total_slices : ℕ) (calories_per_slice : ℕ) (total_calories_eaten : ℕ) :
  total_slices = 8 →
  calories_per_slice = 300 →
  total_calories_eaten = 1200 →
  (total_calories_eaten / calories_per_slice) = (1 / 2) * total_slices :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end ratio_slices_l146_146875


namespace problem_eight_sided_polygon_interiors_l146_146154

-- Define the condition of the problem
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- One interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℚ := sum_of_interior_angles n / n

-- The main theorem stating the problem
theorem problem_eight_sided_polygon_interiors (n : ℕ) (h1: diagonals_from_vertex n = 5) : 
  one_interior_angle n = 135 :=
by
  -- Proof would go here
  sorry

end problem_eight_sided_polygon_interiors_l146_146154


namespace sum_of_reciprocal_squares_l146_146369

theorem sum_of_reciprocal_squares
  (p q r : ℝ)
  (h1 : p + q + r = 9)
  (h2 : p * q + q * r + r * p = 8)
  (h3 : p * q * r = -2) :
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 25 := by
  sorry

end sum_of_reciprocal_squares_l146_146369


namespace find_n_divisibility_l146_146882

theorem find_n_divisibility :
  ∃ n : ℕ, n < 10 ∧ (6 * 10000 + n * 1000 + 2 * 100 + 7 * 10 + 2) % 11 = 0 ∧ (6 * 10000 + n * 1000 + 2 * 100 + 7 * 10 + 2) % 5 = 0 :=
by
  use 3
  sorry

end find_n_divisibility_l146_146882


namespace xyz_inequality_l146_146035

theorem xyz_inequality (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  3 * (x^2 * y^2 + x^2 * z^2 + y^2 * z^2) - 2 * x * y * z * (x + y + z) ≤ 3 := by
  sorry

end xyz_inequality_l146_146035


namespace Petya_bonus_points_l146_146220

def bonus_points (p : ℕ) : ℕ :=
  if p < 1000 then
    (20 * p) / 100
  else if p ≤ 2000 then
    200 + (30 * (p - 1000)) / 100
  else
    200 + 300 + (50 * (p - 2000)) / 100

theorem Petya_bonus_points : bonus_points 2370 = 685 :=
by sorry

end Petya_bonus_points_l146_146220


namespace sum_of_x_and_y_greater_equal_twice_alpha_l146_146191

theorem sum_of_x_and_y_greater_equal_twice_alpha (x y α : ℝ) 
  (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) :
  x + y ≥ 2 * α :=
sorry

end sum_of_x_and_y_greater_equal_twice_alpha_l146_146191


namespace hypotenuse_length_l146_146300

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l146_146300


namespace common_ratio_of_geometric_series_l146_146810

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l146_146810


namespace prime_factorization_of_expression_l146_146627

theorem prime_factorization_of_expression (p n : ℕ) (hp : Nat.Prime p) (hdiv : p^2 ∣ 2^(p-1) - 1) : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) ∧ 
  a ∣ (p-1) ∧ b ∣ (p! + 2^n) ∧ c ∣ (p! + 2^n) := 
sorry

end prime_factorization_of_expression_l146_146627


namespace fencing_required_l146_146240

theorem fencing_required (length width area : ℕ) (length_eq : length = 30) (area_eq : area = 810) 
  (field_area : length * width = area) : 2 * length + width = 87 := 
by
  sorry

end fencing_required_l146_146240


namespace isosceles_triangle_largest_angle_l146_146737

theorem isosceles_triangle_largest_angle (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = 50) :
  A + B + C = 180 →
  C = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l146_146737


namespace largest_base5_to_base10_l146_146077

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l146_146077


namespace find_h_l146_146600

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l146_146600


namespace solve_equation_l146_146126

-- Define the conditions of the problem.
def equation (x : ℝ) : Prop := (5 - x / 3)^(1/3) = -2

-- Define the main theorem to prove that x = 39 is the solution to the equation.
theorem solve_equation : ∃ x : ℝ, equation x ∧ x = 39 :=
by
  existsi 39
  intros
  simp [equation]
  sorry

end solve_equation_l146_146126


namespace johns_previous_salary_l146_146749

-- Conditions
def johns_new_salary : ℝ := 70
def percent_increase : ℝ := 0.16666666666666664

-- Statement
theorem johns_previous_salary :
  ∃ x : ℝ, x + percent_increase * x = johns_new_salary ∧ x = 60 :=
by
  sorry

end johns_previous_salary_l146_146749


namespace find_J_salary_l146_146849

variable (J F M A : ℝ)

theorem find_J_salary (h1 : (J + F + M + A) / 4 = 8000) (h2 : (F + M + A + 6500) / 4 = 8900) :
  J = 2900 := by
  sorry

end find_J_salary_l146_146849


namespace number_of_divisors_that_are_multiples_of_2_l146_146453

-- Define the prime factorization of 540
def prime_factorization_540 : ℕ × ℕ × ℕ := (2, 3, 5)

-- Define the constraints for a divisor to be a multiple of 2
def valid_divisor_form (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1

noncomputable def count_divisors (prime_info : ℕ × ℕ × ℕ) : ℕ :=
  let (p1, p2, p3) := prime_info
  2 * 4 * 2 -- Correspond to choices for \( a \), \( b \), and \( c \)

theorem number_of_divisors_that_are_multiples_of_2 (p1 p2 p3 : ℕ) (h : prime_factorization_540 = (p1, p2, p3)) :
  ∃ (count : ℕ), count = 16 :=
by
  use count_divisors (2, 3, 5)
  sorry

end number_of_divisors_that_are_multiples_of_2_l146_146453


namespace mr_a_loss_l146_146631

noncomputable def house_initial_value := 12000
noncomputable def first_transaction_loss := 15 / 100
noncomputable def second_transaction_gain := 20 / 100

def house_value_after_first_transaction (initial_value loss : ℝ) : ℝ :=
  initial_value * (1 - loss)

def house_value_after_second_transaction (value_after_first gain : ℝ) : ℝ :=
  value_after_first * (1 + gain)

theorem mr_a_loss :
  let initial_value := house_initial_value
  let loss := first_transaction_loss
  let gain := second_transaction_gain
  let value_after_first := house_value_after_first_transaction initial_value loss
  let value_after_second := house_value_after_second_transaction value_after_first gain
  value_after_second - initial_value = 240 :=
by
  sorry

end mr_a_loss_l146_146631


namespace lamp_turn_off_ways_l146_146990

theorem lamp_turn_off_ways : 
  ∃ (ways : ℕ), ways = 10 ∧
  (∃ (n : ℕ) (m : ℕ), 
    n = 6 ∧  -- 6 lamps in a row
    m = 2 ∧  -- turn off 2 of them
    ways = Nat.choose (n - m + 1) m) := -- 2 adjacent lamps cannot be turned off
by
  -- Proof will be provided here.
  sorry

end lamp_turn_off_ways_l146_146990


namespace car_trip_proof_l146_146991

def initial_oil_quantity (y : ℕ → ℕ) : Prop :=
  y 0 = 50

def consumption_rate (y : ℕ → ℕ) : Prop :=
  ∀ t, y t = y (t - 1) - 5

def relationship_between_y_and_t (y : ℕ → ℕ) : Prop :=
  ∀ t, y t = 50 - 5 * t

def oil_left_after_8_hours (y : ℕ → ℕ) : Prop :=
  y 8 = 10

theorem car_trip_proof (y : ℕ → ℕ) :
  initial_oil_quantity y ∧ consumption_rate y ∧ relationship_between_y_and_t y ∧ oil_left_after_8_hours y :=
by
  -- the proof goes here
  sorry

end car_trip_proof_l146_146991


namespace line_through_point_equal_intercepts_l146_146781

theorem line_through_point_equal_intercepts (x y a b : ℝ) :
  ∀ (x y : ℝ), 
    (x - 1) = a → 
    (y - 2) = b →
    (a = -1 ∨ a = 2) → 
    ((x + y - 3 = 0) ∨ (2 * x - y = 0)) := by
  sorry

end line_through_point_equal_intercepts_l146_146781


namespace percent_kindergarten_combined_l146_146053

-- Define the constants provided in the problem
def studentsPinegrove : ℕ := 150
def studentsMaplewood : ℕ := 250

def percentKindergartenPinegrove : ℝ := 18.0
def percentKindergartenMaplewood : ℝ := 14.0

-- The proof statement
theorem percent_kindergarten_combined :
  (27.0 + 35.0) / (150.0 + 250.0) * 100.0 = 15.5 :=
by 
  sorry

end percent_kindergarten_combined_l146_146053


namespace Alice_favorite_number_l146_146538

theorem Alice_favorite_number :
  ∃ n : ℕ, (30 ≤ n ∧ n ≤ 70) ∧ (7 ∣ n) ∧ ¬(3 ∣ n) ∧ (4 ∣ (n / 10 + n % 10)) ∧ n = 35 :=
by
  sorry

end Alice_favorite_number_l146_146538


namespace football_team_total_members_l146_146040

-- Definitions from the problem conditions
def initialMembers : ℕ := 42
def newMembers : ℕ := 17

-- Mathematical equivalent proof problem
theorem football_team_total_members : initialMembers + newMembers = 59 := by
  sorry

end football_team_total_members_l146_146040


namespace paths_mat8_l146_146921

-- Define variables
def grid := [
  ["M", "A", "M", "A", "M"],
  ["A", "T", "A", "T", "A"],
  ["M", "A", "M", "A", "M"],
  ["A", "T", "A", "T", "A"],
  ["M", "A", "M", "A", "M"]
]

def is_adjacent (x1 y1 x2 y2 : Nat): Bool :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 = y2 - 1)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 = x2 - 1))

def count_paths (grid: List (List String)): Nat :=
  -- implementation to count number of paths
  4 * 4 * 2

theorem paths_mat8 (grid: List (List String)): count_paths grid = 32 := by
  sorry

end paths_mat8_l146_146921


namespace parallel_vectors_cosine_identity_l146_146910

-- Defining the problem in Lean 4

theorem parallel_vectors_cosine_identity :
  ∀ α : ℝ, (∃ k : ℝ, (1 / 3, Real.tan α) = (k * Real.cos α, k)) →
  Real.cos (Real.pi / 2 + α) = -1 / 3 :=
by
  sorry

end parallel_vectors_cosine_identity_l146_146910


namespace min_value_ab_sum_l146_146176

theorem min_value_ab_sum (a b : ℤ) (h : a * b = 100) : a + b ≥ -101 :=
  sorry

end min_value_ab_sum_l146_146176


namespace problem_l146_146218

-- Definition for condition 1
def condition1 (uniform_band : Prop) (appropriate_model : Prop) := 
  uniform_band → appropriate_model

-- Definition for condition 2
def condition2 (smaller_residual : Prop) (better_fit : Prop) :=
  smaller_residual → better_fit

-- Formal statement of the problem
theorem problem (uniform_band appropriate_model smaller_residual better_fit : Prop)
  (h1 : condition1 uniform_band appropriate_model)
  (h2 : condition2 smaller_residual better_fit)
  (h3 : uniform_band ∧ smaller_residual) :
  appropriate_model ∧ better_fit :=
  sorry

end problem_l146_146218


namespace compare_y1_y2_l146_146656

-- Definitions for the conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + 3
axiom a_neg (a : ℝ) : a < 0

-- The proof problem statement
theorem compare_y1_y2 (a : ℝ) (h_a : a_neg a)
  (y1 : ℝ) (y2 : ℝ) (h_y1 : y1 = quadratic_function a (-1))
  (h_y2 : y2 = quadratic_function a 2) : y1 < y2 :=
sorry

end compare_y1_y2_l146_146656


namespace remainder_of_7n_div_4_l146_146834

theorem remainder_of_7n_div_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
sorry

end remainder_of_7n_div_4_l146_146834


namespace select_3_representatives_l146_146960

theorem select_3_representatives (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 4) :
  (∑ g in {2, 3}, (Nat.choose girls g) * (Nat.choose boys (3 - g))) = 28 :=
by
  sorry

end select_3_representatives_l146_146960


namespace universal_quantifiers_and_propositions_l146_146388

-- Definitions based on conditions
def universal_quantifiers_phrases := ["for all", "for any"]
def universal_quantifier_symbol := "∀"
def universal_proposition := "Universal Proposition"
def universal_proposition_representation := "∀ x ∈ M, p(x)"

-- Main theorem
theorem universal_quantifiers_and_propositions :
  universal_quantifiers_phrases = ["for all", "for any"]
  ∧ universal_quantifier_symbol = "∀"
  ∧ universal_proposition = "Universal Proposition"
  ∧ universal_proposition_representation = "∀ x ∈ M, p(x)" :=
by
  sorry

end universal_quantifiers_and_propositions_l146_146388


namespace gcd_9240_12240_33720_l146_146127

theorem gcd_9240_12240_33720 : Nat.gcd (Nat.gcd 9240 12240) 33720 = 240 := by
  sorry

end gcd_9240_12240_33720_l146_146127


namespace free_fall_height_and_last_second_distance_l146_146681

theorem free_fall_height_and_last_second_distance :
  let time := 11
  let initial_distance := 4.9
  let increment := 9.8
  let total_height := (initial_distance * time + increment * (time * (time - 1)) / 2)
  let last_second_distance := initial_distance + increment * (time - 1)
  total_height = 592.9 ∧ last_second_distance = 102.9 :=
by
  sorry

end free_fall_height_and_last_second_distance_l146_146681


namespace fraction_to_decimal_l146_146880

theorem fraction_to_decimal (numerator : ℚ) (denominator : ℚ) (h : numerator = 5 ∧ denominator = 40) : 
  (numerator / denominator) = 0.125 :=
sorry

end fraction_to_decimal_l146_146880


namespace transformation_composition_l146_146740

def f (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def g (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 - p.2)

theorem transformation_composition :
  f (g (-1, 2)) = (1, -3) :=
by {
  sorry
}

end transformation_composition_l146_146740


namespace total_surface_area_of_pyramid_l146_146243

noncomputable def base_length_ab : ℝ := 8 -- Length of side AB
noncomputable def base_length_ad : ℝ := 6 -- Length of side AD
noncomputable def height_pf : ℝ := 15 -- Perpendicular height from peak P to the base's center F

noncomputable def base_area : ℝ := base_length_ab * base_length_ad
noncomputable def fm_distance : ℝ := Real.sqrt ((base_length_ab / 2)^2 + (base_length_ad / 2)^2)
noncomputable def slant_height_pm : ℝ := Real.sqrt (height_pf^2 + fm_distance^2)

noncomputable def lateral_area_ab : ℝ := 2 * (0.5 * base_length_ab * slant_height_pm)
noncomputable def lateral_area_ad : ℝ := 2 * (0.5 * base_length_ad * slant_height_pm)
noncomputable def total_surface_area : ℝ := base_area + lateral_area_ab + lateral_area_ad

theorem total_surface_area_of_pyramid :
  total_surface_area = 48 + 55 * Real.sqrt 10 := by
  sorry

end total_surface_area_of_pyramid_l146_146243


namespace configuration_count_l146_146684

theorem configuration_count :
  (∃ (w h s : ℕ), 2 * (w + h + 2 * s) = 120 ∧ w < h ∧ s % 2 = 0) →
  ∃ n, n = 196 := 
sorry

end configuration_count_l146_146684


namespace sum_of_b_values_l146_146274

theorem sum_of_b_values (b1 b2 : ℝ) : 
  (∀ x : ℝ, (9 * x^2 + (b1 + 15) * x + 16 = 0 ∨ 9 * x^2 + (b2 + 15) * x + 16 = 0) ∧ 
           (b1 + 15)^2 - 4 * 9 * 16 = 0 ∧ 
           (b2 + 15)^2 - 4 * 9 * 16 = 0) → 
  (b1 + b2) = -30 := 
sorry

end sum_of_b_values_l146_146274


namespace first_car_speed_l146_146507

-- Definitions based on problem conditions
def highway_length : ℕ := 105
def second_car_speed : ℕ := 20
def meeting_time : ℕ := 3

-- The question is to prove the speed of the first car
theorem first_car_speed : ∃ v : ℕ, 3 * v + 3 * second_car_speed = highway_length ∧ v = 15 :=
by
  use 15
  split
  { calc
      3 * 15 + 3 * 20 = highway_length : by norm_num
  }
  { refl }

end first_car_speed_l146_146507


namespace log_prime_factor_inequality_l146_146709

open Real

-- Define p(n) such that it returns the number of prime factors of n.
noncomputable def p (n: ℕ) : ℕ := sorry  -- This will be defined contextually for now

theorem log_prime_factor_inequality (n : ℕ) (hn : n > 0) : 
  log n ≥ (p n) * log 2 :=
by 
  sorry

end log_prime_factor_inequality_l146_146709


namespace sum_of_g1_l146_146479

-- Define the main conditions
variable {g : ℝ → ℝ}
variable (h_nonconst : ∀ a b : ℝ, a ≠ b → g a ≠ g b)
axiom main_condition : ∀ x : ℝ, x ≠ 0 → g (x - 1) + g x + g (x + 1) = (g x) ^ 2 / (2025 * x)

-- Define the goal
theorem sum_of_g1 :
  g 1 = 6075 :=
sorry

end sum_of_g1_l146_146479


namespace sum_of_first_49_primes_is_10787_l146_146268

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                                79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
                                167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227]

theorem sum_of_first_49_primes_is_10787:
  first_49_primes.sum = 10787 :=
by
  -- Proof would go here.
  -- This is just a placeholder as per the requirements.
  sorry

end sum_of_first_49_primes_is_10787_l146_146268


namespace average_cars_given_per_year_l146_146353

/-- Definition of initial conditions and the proposition -/
def initial_cars : ℕ := 3500
def final_cars : ℕ := 500
def years : ℕ := 60

theorem average_cars_given_per_year : (initial_cars - final_cars) / years = 50 :=
by
  sorry

end average_cars_given_per_year_l146_146353


namespace fisherman_gets_8_red_snappers_l146_146782

noncomputable def num_red_snappers (R : ℕ) : Prop :=
  let cost_red_snapper := 3
  let cost_tuna := 2
  let num_tunas := 14
  let total_earnings := 52
  (R * cost_red_snapper) + (num_tunas * cost_tuna) = total_earnings

theorem fisherman_gets_8_red_snappers : num_red_snappers 8 :=
by
  sorry

end fisherman_gets_8_red_snappers_l146_146782


namespace common_ratio_of_geometric_series_l146_146811

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l146_146811


namespace first_term_geometric_sequence_l146_146054

theorem first_term_geometric_sequence (a r : ℚ) 
  (h3 : a * r^(3-1) = 24)
  (h4 : a * r^(4-1) = 36) :
  a = 32 / 3 :=
by
  sorry

end first_term_geometric_sequence_l146_146054


namespace julio_fish_count_l146_146750

theorem julio_fish_count : 
  ∀ (catches_per_hour : ℕ) (hours_fishing : ℕ) (fish_lost : ℕ) (total_fish : ℕ), 
  catches_per_hour = 7 →
  hours_fishing = 9 →
  fish_lost = 15 →
  total_fish = (catches_per_hour * hours_fishing) - fish_lost →
  total_fish = 48 :=
by
  intros catches_per_hour hours_fishing fish_lost total_fish
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

# Jason had jotted down the following proof statement:

end julio_fish_count_l146_146750


namespace value_of_f2009_l146_146147

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f2009 
  (h_ineq1 : ∀ x : ℝ, f x ≤ f (x+4) + 4)
  (h_ineq2 : ∀ x : ℝ, f (x+2) ≥ f x + 2)
  (h_f1 : f 1 = 0) :
  f 2009 = 2008 :=
sorry

end value_of_f2009_l146_146147


namespace total_pieces_of_clothes_l146_146456

theorem total_pieces_of_clothes (shirts_per_pant pants : ℕ) (h1 : shirts_per_pant = 6) (h2 : pants = 40) : 
  shirts_per_pant * pants + pants = 280 :=
by
  rw [h1, h2]
  sorry

end total_pieces_of_clothes_l146_146456


namespace min_chips_to_A10_l146_146062

theorem min_chips_to_A10 (n : ℕ) (A : ℕ → ℕ) (hA1 : A 1 = n) :
  (∃ (σ : ℕ → ℕ), 
    (∀ i, 1 ≤ i ∧ i < 10 → (σ i = A i - 2) ∧ (σ (i + 1) = A (i + 1) + 1)) ∨ 
    (∀ i, 1 ≤ i ∧ i < 9 → (σ (i + 1) = A (i + 1) - 2) ∧ (σ (i + 2) = A (i + 2) + 1) ∧ (σ i = A i + 1)) ∧ 
    (∃ (k : ℕ), k = 10 ∧ σ k = 1)) →
  n ≥ 46 := sorry

end min_chips_to_A10_l146_146062


namespace cole_drive_time_l146_146257

theorem cole_drive_time (D T1 T2 : ℝ) (h1 : T1 = D / 75) 
  (h2 : T2 = D / 105) (h3 : T1 + T2 = 6) : 
  (T1 * 60 = 210) :=
by sorry

end cole_drive_time_l146_146257


namespace shaded_area_l146_146296

open Real

theorem shaded_area (AH HF GF : ℝ) (AH_eq : AH = 12) (HF_eq : HF = 16) (GF_eq : GF = 4) 
  (DG : ℝ) (DG_eq : DG = 3) (area_triangle_DGF : ℝ) (area_triangle_DGF_eq : area_triangle_DGF = 6) :
  let area_square : ℝ := 4 * 4
  let shaded_area : ℝ := area_square - area_triangle_DGF
  shaded_area = 10 := by
    sorry

end shaded_area_l146_146296


namespace simplify_expression_l146_146115

theorem simplify_expression : 
  2 ^ (-1: ℤ) + Real.sqrt 16 - (3 - Real.sqrt 3) ^ 0 + |Real.sqrt 2 - 1 / 2| = 3 + Real.sqrt 2 := by
  sorry

end simplify_expression_l146_146115


namespace find_n_for_perfect_square_l146_146140

theorem find_n_for_perfect_square :
  ∃ (n : ℕ), n > 0 ∧ ∃ (m : ℤ), n^2 + 5 * n + 13 = m^2 ∧ n = 4 :=
by
  sorry

end find_n_for_perfect_square_l146_146140


namespace cos_alpha_value_l146_146285

-- Definitions for conditions and theorem statement

def condition_1 (α : ℝ) : Prop :=
  0 < α ∧ α < Real.pi / 2

def condition_2 (α : ℝ) : Prop :=
  Real.cos (Real.pi / 3 + α) = 1 / 3

theorem cos_alpha_value (α : ℝ) (h1 : condition_1 α) (h2 : condition_2 α) :
  Real.cos α = (1 + 2 * Real.sqrt 6) / 6 := sorry

end cos_alpha_value_l146_146285


namespace sequence_property_l146_146979

theorem sequence_property (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n+1) + 2021 * a (n+2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := 
sorry

end sequence_property_l146_146979


namespace triangle_height_l146_146231

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 615) (h_base : base = 123) 
  (area_formula : area = (base * height) / 2) : height = 10 := 
by 
  sorry

end triangle_height_l146_146231


namespace probability_two_even_multiples_of_five_drawn_l146_146523

-- Definition of conditions
def toys : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
                      39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

def isEvenMultipleOfFive (n : ℕ) : Bool := n % 10 == 0

-- Collect all such numbers from the list
def evenMultiplesOfFive : List ℕ := toys.filter isEvenMultipleOfFive

-- Number of such even multiples of 5
def countEvenMultiplesOfFive : ℕ := evenMultiplesOfFive.length

theorem probability_two_even_multiples_of_five_drawn :
  (countEvenMultiplesOfFive / 50) * ((countEvenMultiplesOfFive - 1) / 49) = 2 / 245 :=
  by sorry

end probability_two_even_multiples_of_five_drawn_l146_146523


namespace range_of_b_l146_146460

theorem range_of_b (b : ℝ) : (¬ ∃ a < 0, a + 1/a > b) → b ≥ -2 := 
by {
  sorry
}

end range_of_b_l146_146460


namespace find_number_l146_146239

theorem find_number (X a b : ℕ) (hX : X = 10 * a + b) 
  (h1 : a * b = 24) (h2 : 10 * b + a = X + 18) : X = 46 :=
by
  sorry

end find_number_l146_146239


namespace regular_polygon_sides_l146_146464

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 = 144 * n) : n = 10 := 
by 
  sorry

end regular_polygon_sides_l146_146464


namespace common_factor_polynomials_l146_146939

theorem common_factor_polynomials (a : ℝ) :
  (∀ p : ℝ, p ≠ 0 ∧ 
           (p^3 - p - a = 0) ∧ 
           (p^2 + p - a = 0)) → 
  (a = 0 ∨ a = 10 ∨ a = -2) := by
  sorry

end common_factor_polynomials_l146_146939


namespace shelves_used_l146_146102

-- Definitions from conditions
def initial_bears : ℕ := 6
def shipment_bears : ℕ := 18
def bears_per_shelf : ℕ := 6

-- Theorem statement
theorem shelves_used : (initial_bears + shipment_bears) / bears_per_shelf = 4 := by
  sorry

end shelves_used_l146_146102


namespace men_in_group_initial_l146_146042

variable (M : ℕ)  -- Initial number of men in the group
variable (A : ℕ)  -- Initial average age of the group

theorem men_in_group_initial : (2 * 50 - (18 + 22) = 60) → ((M + 6) = 60 / 6) → (M = 10) :=
by
  sorry

end men_in_group_initial_l146_146042


namespace function_symmetry_property_l146_146722

noncomputable def f (x : ℝ) : ℝ :=
  x ^ 2

def symmetry_property := 
  ∀ (x : ℝ), (-1 < x ∧ x ≤ 1) →
    (¬ (f (-x) = f x) ∧ ¬ (f (-x) = -f x))

theorem function_symmetry_property :
  symmetry_property :=
by
  sorry

end function_symmetry_property_l146_146722


namespace remainder_7n_mod_4_l146_146839

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by sorry

end remainder_7n_mod_4_l146_146839


namespace quadratic_equation_unique_solution_l146_146206

theorem quadratic_equation_unique_solution
  (a c : ℝ)
  (h_discriminant : 100 - 4 * a * c = 0)
  (h_sum : a + c = 12)
  (h_lt : a < c) :
  (a, c) = (6 - Real.sqrt 11, 6 + Real.sqrt 11) :=
sorry

end quadratic_equation_unique_solution_l146_146206


namespace total_snakes_count_l146_146946

-- Define the basic conditions
def breedingBalls : Nat := 3
def snakesPerBall : Nat := 8
def pairsOfSnakes : Nat := 6
def snakesPerPair : Nat := 2

-- Define the total number of snakes
theorem total_snakes_count : 
  (breedingBalls * snakesPerBall) + (pairsOfSnakes * snakesPerPair) = 36 := 
by 
  -- we skip the proof with sorry
  sorry

end total_snakes_count_l146_146946


namespace depth_of_lost_ship_l146_146419

theorem depth_of_lost_ship (rate_of_descent : ℕ) (time_taken : ℕ) (h1 : rate_of_descent = 60) (h2 : time_taken = 60) :
  rate_of_descent * time_taken = 3600 :=
by {
  /-
  Proof steps would go here.
  -/
  sorry
}

end depth_of_lost_ship_l146_146419


namespace police_officer_can_catch_gangster_l146_146862

theorem police_officer_can_catch_gangster
  (a : ℝ) -- length of the side of the square
  (v_police : ℝ) -- maximum speed of the police officer
  (v_gangster : ℝ) -- maximum speed of the gangster
  (h_gangster_speed : v_gangster = 2.9 * v_police) :
  ∃ (t : ℝ), t ≥ 0 ∧ (a / (2 * v_police)) = t := sorry

end police_officer_can_catch_gangster_l146_146862


namespace votes_for_Crow_l146_146919

theorem votes_for_Crow 
  (J : ℕ)
  (P V K : ℕ)
  (ε1 ε2 ε3 ε4 : ℤ)
  (h₁ : P + V = 15 + ε1)
  (h₂ : V + K = 18 + ε2)
  (h₃ : K + P = 20 + ε3)
  (h₄ : P + V + K = 59 + ε4)
  (bound₁ : |ε1| ≤ 13)
  (bound₂ : |ε2| ≤ 13)
  (bound₃ : |ε3| ≤ 13)
  (bound₄ : |ε4| ≤ 13)
  : V = 13 :=
sorry

end votes_for_Crow_l146_146919


namespace distance_between_neg2_and_3_l146_146777
-- Import the necessary Lean libraries

-- State the theorem to prove the distance between -2 and 3 is 5
theorem distance_between_neg2_and_3 : abs (3 - (-2)) = 5 := by
  sorry

end distance_between_neg2_and_3_l146_146777


namespace find_angle_C_l146_146377

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (triangle_ABC : Type)

-- Given conditions
axiom ten_a_cos_B_eq_three_b_cos_A : 10 * a * Real.cos B = 3 * b * Real.cos A
axiom cos_A_value : Real.cos A = 5 * Real.sqrt 26 / 26

-- Required to prove
theorem find_angle_C : C = 3 * Real.pi / 4 := by
  sorry

end find_angle_C_l146_146377


namespace geometric_sequence_a5_l146_146618

variable {a : ℕ → ℝ}
variable (h₁ : a 3 * a 7 = 3)
variable (h₂ : a 3 + a 7 = 4)

theorem geometric_sequence_a5 : a 5 = Real.sqrt 3 := 
sorry

end geometric_sequence_a5_l146_146618


namespace difference_is_1343_l146_146966

-- Define the larger number L and the relationship with the smaller number S.
def L : ℕ := 1608
def quotient : ℕ := 6
def remainder : ℕ := 15

-- Define the relationship: L = 6S + 15
def relationship (S : ℕ) : Prop := L = quotient * S + remainder

-- The theorem we want to prove: The difference between the larger and smaller number is 1343
theorem difference_is_1343 (S : ℕ) (h_rel : relationship S) : L - S = 1343 :=
by
  sorry

end difference_is_1343_l146_146966


namespace triangle_side_length_l146_146014

noncomputable def sine (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180) -- Define sine function explicitly (degrees to radians)

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ)
  (hA : A = 30) (hC : C = 45) (ha : a = 4) :
  c = 4 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l146_146014


namespace quadratic_expression_rewriting_l146_146582

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l146_146582


namespace right_triangle_other_side_l146_146291

theorem right_triangle_other_side (c a : ℝ) (h_c : c = 10) (h_a : a = 6) : ∃ b : ℝ, b^2 = c^2 - a^2 ∧ b = 8 :=
by
  use 8
  rw [h_c, h_a]
  simp
  sorry

end right_triangle_other_side_l146_146291


namespace total_luggage_l146_146031

theorem total_luggage (ne nb nf : ℕ)
  (leconomy lbusiness lfirst : ℕ)
  (Heconomy : ne = 10) 
  (Hbusiness : nb = 7) 
  (Hfirst : nf = 3)
  (Heconomy_luggage : leconomy = 5)
  (Hbusiness_luggage : lbusiness = 8)
  (Hfirst_luggage : lfirst = 12) : 
  (ne * leconomy + nb * lbusiness + nf * lfirst) = 142 :=
by
  sorry

end total_luggage_l146_146031


namespace expected_value_dice_sum_l146_146845

theorem expected_value_dice_sum :
  ∀ (d1 d2 d3 : ℕ), 
    (1 ≤ d1 ∧ d1 ≤ 6) → 
    (1 ≤ d2 ∧ d2 ≤ 6) → 
    (1 ≤ d3 ∧ d3 ≤ 6) → 
    max d1 (max d2 d3) = 5 → 
    ∃ (a b : ℕ), 
    (a = 645) ∧ (b = 61) ∧ (a + b = 706) ∧ 
    (expected_value (sum_dice d1 d2 d3) = (645 / 61) : ℚ) :=
by
  sorry

end expected_value_dice_sum_l146_146845


namespace cricketer_wickets_l146_146672

noncomputable def initial_average (R W : ℝ) : ℝ := R / W

noncomputable def new_average (R W : ℝ) (additional_runs additional_wickets : ℝ) : ℝ :=
  (R + additional_runs) / (W + additional_wickets)

theorem cricketer_wickets (R W : ℝ) 
(h1 : initial_average R W = 12.4) 
(h2 : new_average R W 26 5 = 12.0) : 
  W = 85 :=
sorry

end cricketer_wickets_l146_146672


namespace sequence_property_l146_146977

theorem sequence_property (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n+1) + 2021 * a (n+2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := 
sorry

end sequence_property_l146_146977


namespace max_intersections_l146_146992

theorem max_intersections (n1 n2 k : ℕ) 
  (h1 : n1 ≤ n2)
  (h2 : k ≤ n1) : 
  ∃ max_intersections : ℕ, 
  max_intersections = k * n2 :=
by
  sorry

end max_intersections_l146_146992


namespace seq_a22_l146_146980

def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0) ∧
  (a 10 = 10)

theorem seq_a22 : ∀ (a : ℕ → ℝ), seq a → a 22 = 10 :=
by
  intros a h,
  have h1 := h.1,
  have h99 := h.2.1,
  have h100 := h.2.2.1,
  have h_eq := h.2.2.2,
  sorry

end seq_a22_l146_146980


namespace decrement_value_each_observation_l146_146499

theorem decrement_value_each_observation 
  (n : ℕ) 
  (original_mean updated_mean : ℝ) 
  (n_pos : n = 50) 
  (original_mean_value : original_mean = 200)
  (updated_mean_value : updated_mean = 153) :
  (original_mean * n - updated_mean * n) / n = 47 :=
by
  sorry

end decrement_value_each_observation_l146_146499


namespace wall_building_time_l146_146748

variable (r : ℝ) -- rate at which one worker can build the wall
variable (W : ℝ) -- the wall in units, let’s denote one whole wall as 1 unit

theorem wall_building_time:
  (∀ (w t : ℝ), W = (60 * r) * t → W = (30 * r) * 6) :=
by
  sorry

end wall_building_time_l146_146748


namespace bread_calories_l146_146056

theorem bread_calories (total_calories : Nat) (pb_calories : Nat) (pb_servings : Nat) (bread_pieces : Nat) (bread_calories : Nat)
  (h1 : total_calories = 500)
  (h2 : pb_calories = 200)
  (h3 : pb_servings = 2)
  (h4 : bread_pieces = 1)
  (h5 : total_calories = pb_servings * pb_calories + bread_pieces * bread_calories) : 
  bread_calories = 100 :=
by
  sorry

end bread_calories_l146_146056


namespace min_reciprocal_sum_l146_146936

theorem min_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy_sum : x + y = 12) (hxy_neq : x ≠ y) : 
  ∃ c : ℝ, c = 1 / 3 ∧ (1 / x + 1 / y ≥ c) :=
sorry

end min_reciprocal_sum_l146_146936


namespace find_r_l146_146131

theorem find_r (r : ℝ) (h1 : ∃ s : ℝ, 8 * x^3 - 4 * x^2 - 42 * x + 45 = 8 * (x - r)^2 * (x - s)) :
  r = 3 / 2 :=
by
  sorry

end find_r_l146_146131


namespace hypotenuse_length_l146_146314

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l146_146314


namespace John_total_weekly_consumption_l146_146927

/-
  Prove that John's total weekly consumption of water, milk, and juice in quarts is 49.25 quarts, 
  given the specified conditions on his daily and periodic consumption.
-/

def John_consumption_problem (gallons_per_day : ℝ) (pints_every_other_day : ℝ) (ounces_every_third_day : ℝ) 
  (quarts_per_gallon : ℝ) (quarts_per_pint : ℝ) (quarts_per_ounce : ℝ) : ℝ :=
  let water_per_day := gallons_per_day * quarts_per_gallon
  let water_per_week := water_per_day * 7
  let milk_per_other_day := pints_every_other_day * quarts_per_pint
  let milk_per_week := milk_per_other_day * 4 -- assuming he drinks milk 4 times a week
  let juice_per_third_day := ounces_every_third_day * quarts_per_ounce
  let juice_per_week := juice_per_third_day * 2 -- assuming he drinks juice 2 times a week
  water_per_week + milk_per_week + juice_per_week

theorem John_total_weekly_consumption :
  John_consumption_problem 1.5 3 20 4 (1/2) (1/32) = 49.25 :=
by
  sorry

end John_total_weekly_consumption_l146_146927


namespace XT_value_l146_146407

noncomputable def AB := 15
noncomputable def BC := 20
noncomputable def height_P := 30
noncomputable def volume_ratio := 9

theorem XT_value 
  (AB BC height_P : ℕ)
  (volume_ratio : ℕ)
  (h1 : AB = 15)
  (h2 : BC = 20)
  (h3 : height_P = 30)
  (h4 : volume_ratio = 9) : 
  ∃ (m n : ℕ), m + n = 97 ∧ m.gcd n = 1 :=
by sorry

end XT_value_l146_146407


namespace shaded_area_of_circles_l146_146012

theorem shaded_area_of_circles :
  let R := 10
  let r1 := R / 2
  let r2 := R / 2
  (π * R^2 - (π * r1^2 + π * r1^2 + π * r2^2)) = 25 * π :=
by
  sorry

end shaded_area_of_circles_l146_146012


namespace gcd_of_gy_and_y_l146_146572

theorem gcd_of_gy_and_y (y : ℕ) (h : ∃ k : ℕ, y = k * 3456) :
  gcd ((5 * y + 4) * (9 * y + 1) * (12 * y + 6) * (3 * y + 9)) y = 216 :=
by {
  sorry
}

end gcd_of_gy_and_y_l146_146572


namespace rectangle_area_is_200000_l146_146165

structure Point :=
  (x : ℝ)
  (y : ℝ)

def isRectangle (P Q R S : Point) : Prop :=
  (P.x - Q.x) * (P.x - Q.x) + (P.y - Q.y) * (P.y - Q.y) = 
  (R.x - S.x) * (R.x - S.x) + (R.y - S.y) * (R.y - S.y) ∧
  (P.x - S.x) * (P.x - S.x) + (P.y - S.y) * (P.y - S.y) = 
  (Q.x - R.x) * (Q.x - R.x) + (Q.y - R.y) * (Q.y - R.y) ∧
  (P.x - Q.x) * (P.x - S.x) + (P.y - Q.y) * (P.y - S.y) = 0

theorem rectangle_area_is_200000:
  ∀ (P Q R S : Point),
  P = ⟨-15, 30⟩ →
  Q = ⟨985, 230⟩ →
  R.x = 985 → 
  S.x = -13 →
  R.y = S.y → 
  isRectangle P Q R S →
  ( ( (Q.x - P.x)^2 + (Q.y - P.y)^2 ).sqrt *
    ( (S.x - P.x)^2 + (S.y - P.y)^2 ).sqrt ) = 200000 :=
by
  intros P Q R S hP hQ hxR hxS hyR hRect
  sorry

end rectangle_area_is_200000_l146_146165


namespace min_abs_sum_l146_146625

-- Definitions based on given conditions for the problem
variable (p q r s : ℤ)
variable (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
variable (h : (matrix 2 2 ℤ ![(p, q), (r, s)]) ^ 2 = matrix 2 2 ℤ ![(9, 0), (0, 9)])

-- Statement of the proof problem
theorem min_abs_sum :
  |p| + |q| + |r| + |s| = 8 :=
by
  sorry

end min_abs_sum_l146_146625


namespace andy_solves_49_problems_l146_146094

theorem andy_solves_49_problems : ∀ (a b : ℕ), a = 78 → b = 125 → b - a + 1 = 49 :=
by
  introv ha hb
  rw [ha, hb]
  norm_num
  sorry

end andy_solves_49_problems_l146_146094


namespace Albert_eats_48_slices_l146_146871

theorem Albert_eats_48_slices (large_pizzas : ℕ) (small_pizzas : ℕ) (slices_large : ℕ) (slices_small : ℕ) 
  (h1 : large_pizzas = 2) (h2 : small_pizzas = 2) (h3 : slices_large = 16) (h4 : slices_small = 8) :
  (large_pizzas * slices_large + small_pizzas * slices_small) = 48 := 
by 
  -- sorry is used to skip the proof.
  sorry

end Albert_eats_48_slices_l146_146871


namespace complete_the_square_h_value_l146_146589

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l146_146589


namespace rectangle_other_side_l146_146362

theorem rectangle_other_side
  (a b : ℝ)
  (Area : ℝ := 12 * a ^ 2 - 6 * a * b)
  (side1 : ℝ := 3 * a)
  (side2 : ℝ := Area / side1) :
  side2 = 4 * a - 2 * b :=
by
  sorry

end rectangle_other_side_l146_146362


namespace solve_for_y_l146_146728

theorem solve_for_y (y : ℝ) (h : 9 / y^3 = y / 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end solve_for_y_l146_146728


namespace people_remaining_at_end_l146_146820

def total_people_start : ℕ := 600
def girls_start : ℕ := 240
def boys_start : ℕ := total_people_start - girls_start
def boys_left_early : ℕ := boys_start / 4
def girls_left_early : ℕ := girls_start / 8
def total_left_early : ℕ := boys_left_early + girls_left_early
def people_remaining : ℕ := total_people_start - total_left_early

theorem people_remaining_at_end : people_remaining = 480 := by
  sorry

end people_remaining_at_end_l146_146820


namespace slope_of_given_line_eq_l146_146699

theorem slope_of_given_line_eq : (∀ x y : ℝ, (4 / x + 5 / y = 0) → (x ≠ 0 ∧ y ≠ 0) → ∀ y x : ℝ, y = - (5 * x / 4) → ∃ m, m = -5/4) :=
by
  sorry

end slope_of_given_line_eq_l146_146699


namespace john_annual_profit_is_1800_l146_146929

def tenant_A_monthly_payment : ℕ := 350
def tenant_B_monthly_payment : ℕ := 400
def tenant_C_monthly_payment : ℕ := 450
def john_monthly_rent : ℕ := 900
def utility_cost : ℕ := 100
def maintenance_fee : ℕ := 50

noncomputable def annual_profit : ℕ :=
  let total_monthly_income := tenant_A_monthly_payment + tenant_B_monthly_payment + tenant_C_monthly_payment
  let total_monthly_expenses := john_monthly_rent + utility_cost + maintenance_fee
  let monthly_profit := total_monthly_income - total_monthly_expenses
  monthly_profit * 12

theorem john_annual_profit_is_1800 : annual_profit = 1800 := by
  sorry

end john_annual_profit_is_1800_l146_146929


namespace determine_angle_A_l146_146465

noncomputable section

open Real

-- Definition of an acute triangle and its sides
variables {A B : ℝ} {a b : ℝ}

-- Additional conditions that are given before providing the theorem
variables (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
          (h5 : 2 * a * sin B = sqrt 3 * b)

-- Theorem statement
theorem determine_angle_A (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
  (h5 : 2 * a * sin B = sqrt 3 * b) : A = π / 3 :=
sorry

end determine_angle_A_l146_146465


namespace sum_of_prime_factors_of_143_l146_146999

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l146_146999


namespace fraction_decomposition_l146_146549

theorem fraction_decomposition :
  ∃ (A B : ℚ), 
  (A = 27 / 10) ∧ (B = -11 / 10) ∧ 
  (∀ x : ℚ, 
    7 * x - 13 = A * (3 * x - 4) + B * (x + 2)) := 
  sorry

end fraction_decomposition_l146_146549


namespace investment_amount_first_rate_l146_146864

theorem investment_amount_first_rate : ∀ (x y : ℝ) (r : ℝ),
  x + y = 15000 → -- Condition 1 (Total investments)
  8200 * r + 6800 * 0.075 = 1023 → -- Condition 2 (Interest yield)
  x = 8200 → -- Condition 3 (Amount invested at first rate)
  x = 8200 := -- Question (How much was invested)
by
  intros x y r h₁ h₂ h₃
  exact h₃

end investment_amount_first_rate_l146_146864


namespace orange_juice_fraction_l146_146219

def capacity_small_pitcher := 500 -- mL
def orange_juice_fraction_small := 1 / 4
def capacity_large_pitcher := 800 -- mL
def orange_juice_fraction_large := 1 / 2

def total_orange_juice_volume := 
  (capacity_small_pitcher * orange_juice_fraction_small) + 
  (capacity_large_pitcher * orange_juice_fraction_large)
def total_volume := capacity_small_pitcher + capacity_large_pitcher

theorem orange_juice_fraction :
  (total_orange_juice_volume / total_volume) = (21 / 52) := 
by 
  sorry

end orange_juice_fraction_l146_146219


namespace Q_gets_less_than_P_l146_146915

theorem Q_gets_less_than_P (x : Real) (hx : x > 0) (hP : P = 1.25 * x): 
  Q = P * 0.8 := 
sorry

end Q_gets_less_than_P_l146_146915


namespace melanie_sale_revenue_correct_l146_146350

noncomputable def melanie_revenue : ℝ :=
let red_cost := 0.08
let green_cost := 0.10
let yellow_cost := 0.12
let red_gumballs := 15
let green_gumballs := 18
let yellow_gumballs := 22
let total_gumballs := red_gumballs + green_gumballs + yellow_gumballs
let total_cost := (red_cost * red_gumballs) + (green_cost * green_gumballs) + (yellow_cost * yellow_gumballs)
let discount := if total_gumballs >= 20 then 0.30 else if total_gumballs >= 10 then 0.20 else 0
let final_cost := total_cost * (1 - discount)
final_cost

theorem melanie_sale_revenue_correct : melanie_revenue = 3.95 :=
by
  -- All calculations and proofs omitted for brevity, as per instructions above
  sorry

end melanie_sale_revenue_correct_l146_146350


namespace problem_statement_l146_146629

-- Definitions of the sets P and Q
def P : Set ℝ := {x : ℝ | x > 1}
def Q : Set ℝ := {x : ℝ | abs x > 0}

-- Statement of the problem to prove that P is not a subset of Q
theorem problem_statement : ¬ (P ⊆ Q) :=
sorry

end problem_statement_l146_146629


namespace num_sets_satisfying_union_l146_146000

theorem num_sets_satisfying_union : 
  ∃! (A : Set ℕ), ({1, 3} ∪ A = {1, 3, 5}) :=
by
  sorry

end num_sets_satisfying_union_l146_146000


namespace fish_remaining_l146_146751

def fish_caught_per_hour := 7
def hours_fished := 9
def fish_lost := 15

theorem fish_remaining : 
  (fish_caught_per_hour * hours_fished - fish_lost) = 48 :=
by
  sorry

end fish_remaining_l146_146751


namespace system_real_solutions_l146_146887

theorem system_real_solutions (a b c : ℝ) :
  (∃ x : ℝ, 
    a * x^2 + b * x + c = 0 ∧ 
    b * x^2 + c * x + a = 0 ∧ 
    c * x^2 + a * x + b = 0) ↔ 
  a + b + c = 0 :=
sorry

end system_real_solutions_l146_146887


namespace part1_part2_l146_146693

-- Part 1
theorem part1 (x y : ℝ) : (2 * x - 3 * y) ^ 2 - (y + 3 * x) * (3 * x - y) = -5 * x ^ 2 - 12 * x * y + 10 * y ^ 2 := 
sorry

-- Part 2
theorem part2 : (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) - 2 ^ 16 = -1 := 
sorry

end part1_part2_l146_146693


namespace smallest_sum_of_big_in_circle_l146_146633

theorem smallest_sum_of_big_in_circle (arranged_circle : Fin 8 → ℕ) (h_circle : ∀ n, arranged_circle n ∈ Finset.range (9) ∧ arranged_circle n > 0) :
  (∀ n, (arranged_circle n > arranged_circle (n + 1) % 8 ∧ arranged_circle n > arranged_circle (n + 7) % 8) ∨ (arranged_circle n < arranged_circle (n + 1) % 8 ∧ arranged_circle n < arranged_circle (n + 7) % 8)) →
  ∃ big_indices : Finset (Fin 8), big_indices.card = 4 ∧ big_indices.sum arranged_circle = 23 :=
by
  sorry

end smallest_sum_of_big_in_circle_l146_146633


namespace albert_pizza_slices_l146_146870

theorem albert_pizza_slices :
  let large_pizzas := 2
  let slices_per_large_pizza := 16
  let small_pizzas := 2
  let slices_per_small_pizza := 8
  (large_pizzas * slices_per_large_pizza + small_pizzas * slices_per_small_pizza) = 48 :=
by
  have h1 : large_pizzas * slices_per_large_pizza = 32 := by sorry
  have h2 : small_pizzas * slices_per_small_pizza = 16 := by sorry
  have ht : 32 + 16 = 48 := by sorry
  exact ht

end albert_pizza_slices_l146_146870


namespace find_m_value_l146_146575

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def is_perpendicular (v1 v2 : vector) : Prop :=
  dot_product v1 v2 = 0

theorem find_m_value (a b : vector) (m : ℝ) (h: a = (2, -1)) (h2: b = (1, 3))
  (h3: is_perpendicular a (a.1 + m * b.1, a.2 + m * b.2)) : m = 5 :=
sorry

end find_m_value_l146_146575


namespace find_a_prove_inequality_l146_146283

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.exp x + 2 * x + a * Real.log x

theorem find_a (a : ℝ) (h : (2 * Real.exp 1 + 2 + a) * (-1 / 2) = -1) : a = -2 * Real.exp 1 :=
by
  sorry

theorem prove_inequality (a : ℝ) (h1 : a = -2 * Real.exp 1) :
    ∀ x : ℝ, x > 0 → f x a > x^2 + 2 :=
by
  sorry

end find_a_prove_inequality_l146_146283


namespace athena_total_spent_l146_146745

def cost_of_sandwiches (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) : ℝ :=
  num_sandwiches * cost_per_sandwich

def cost_of_drinks (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  num_drinks * cost_per_drink

def total_cost (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  cost_of_sandwiches num_sandwiches cost_per_sandwich + cost_of_drinks num_drinks cost_per_drink

theorem athena_total_spent :
  total_cost 3 3 2 2.5 = 14 :=
by 
  sorry

end athena_total_spent_l146_146745


namespace gcd_combination_l146_146969

theorem gcd_combination (a b d : ℕ) (h : d = Nat.gcd a b) : 
  Nat.gcd (5 * a + 3 * b) (13 * a + 8 * b) = d := 
by
  sorry

end gcd_combination_l146_146969


namespace combined_length_of_trains_l146_146387

theorem combined_length_of_trains
  (speed_A_kmph : ℕ) (speed_B_kmph : ℕ)
  (platform_length : ℕ) (time_A_sec : ℕ) (time_B_sec : ℕ)
  (h_speed_A : speed_A_kmph = 72) (h_speed_B : speed_B_kmph = 90)
  (h_platform_length : platform_length = 300)
  (h_time_A : time_A_sec = 30) (h_time_B : time_B_sec = 24) :
  let speed_A_ms := speed_A_kmph * 5 / 18
  let speed_B_ms := speed_B_kmph * 5 / 18
  let distance_A := speed_A_ms * time_A_sec
  let distance_B := speed_B_ms * time_B_sec
  let length_A := distance_A - platform_length
  let length_B := distance_B - platform_length
  length_A + length_B = 600 :=
by
  sorry

end combined_length_of_trains_l146_146387


namespace percentage_slump_in_business_l146_146674

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.05 * Y = 0.04 * X) : (X > 0) → (Y > 0) → (X - Y) / X * 100 = 20 := 
by
  sorry

end percentage_slump_in_business_l146_146674


namespace stickers_total_l146_146396

theorem stickers_total (yesterday_packs : ℕ) (increment_packs : ℕ) (today_packs : ℕ) (total_packs : ℕ) :
  yesterday_packs = 15 → increment_packs = 10 → today_packs = yesterday_packs + increment_packs → total_packs = yesterday_packs + today_packs → total_packs = 40 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h1, h3] at h4
  exact h4

end stickers_total_l146_146396


namespace hypotenuse_length_l146_146301

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l146_146301


namespace solve_frac_difference_of_squares_l146_146225

theorem solve_frac_difference_of_squares :
  (108^2 - 99^2) / 9 = 207 := by
  sorry

end solve_frac_difference_of_squares_l146_146225


namespace number_of_rectangles_containing_cell_l146_146439

theorem number_of_rectangles_containing_cell (m n p q : ℕ) (hp : 1 ≤ p ∧ p ≤ m) (hq : 1 ≤ q ∧ q ≤ n) :
    ∃ count : ℕ, count = p * q * (m - p + 1) * (n - q + 1) := 
    sorry

end number_of_rectangles_containing_cell_l146_146439


namespace distance_between_points_l146_146780

theorem distance_between_points : abs (3 - (-2)) = 5 := 
by
  sorry

end distance_between_points_l146_146780


namespace larger_of_two_numbers_l146_146521

theorem larger_of_two_numbers (A B : ℕ) (HCF : ℕ) (factor1 factor2 : ℕ) (h_hcf : HCF = 23) (h_factor1 : factor1 = 13) (h_factor2 : factor2 = 14)
(hA : A = HCF * factor1) (hB : B = HCF * factor2) :
  max A B = 322 :=
by
  sorry

end larger_of_two_numbers_l146_146521


namespace find_a22_l146_146971

variable (a : ℕ → ℝ)
variable (h : ∀ n, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
variable (h99 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
variable (h100 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
variable (h10 : a 10 = 10)

theorem find_a22 : a 22 = 10 := sorry

end find_a22_l146_146971


namespace complete_the_square_h_value_l146_146587

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l146_146587


namespace complete_the_square_3x2_9x_20_l146_146597

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l146_146597


namespace trisha_money_l146_146385

theorem trisha_money (money_meat money_chicken money_veggies money_eggs money_dogfood money_left : ℤ)
  (h_meat : money_meat = 17)
  (h_chicken : money_chicken = 22)
  (h_veggies : money_veggies = 43)
  (h_eggs : money_eggs = 5)
  (h_dogfood : money_dogfood = 45)
  (h_left : money_left = 35) :
  let total_spent := money_meat + money_chicken + money_veggies + money_eggs + money_dogfood
  in total_spent + money_left = 167 :=
by
  sorry

end trisha_money_l146_146385


namespace number_4_div_p_equals_l146_146730

-- Assume the necessary conditions
variables (p q : ℝ)
variables (h1 : 4 / q = 18) (h2 : p - q = 0.2777777777777778)

-- Define the proof problem
theorem number_4_div_p_equals (N : ℝ) (hN : 4 / p = N) : N = 8 :=
by 
  sorry

end number_4_div_p_equals_l146_146730


namespace rows_identical_l146_146635

theorem rows_identical {n : ℕ} {a : Fin n → ℝ} {k : Fin n → Fin n}
  (h_inc : ∀ i j : Fin n, i < j → a i < a j)
  (h_perm : ∀ i j : Fin n, k i ≠ k j → a (k i) ≠ a (k j))
  (h_sum_inc : ∀ i j : Fin n, i < j → a i + a (k i) < a j + a (k j)) :
  ∀ i : Fin n, a i = a (k i) :=
by
  sorry

end rows_identical_l146_146635


namespace positional_relationship_perpendicular_l146_146897

theorem positional_relationship_perpendicular 
  (a b c : ℝ) 
  (A B C : ℝ)
  (h : b * Real.sin A - a * Real.sin B = 0) :
  (∀ x y : ℝ, (x * Real.sin A + a * y + c = 0) ↔ (b * x - y * Real.sin B + Real.sin C = 0)) :=
sorry

end positional_relationship_perpendicular_l146_146897


namespace articles_bought_l146_146458

theorem articles_bought (C : ℝ) (N : ℝ) (h1 : (N * C) = (30 * ((5 / 3) * C))) : N = 50 :=
by
  sorry

end articles_bought_l146_146458


namespace loan_to_scholarship_ratio_l146_146184

noncomputable def tuition := 22000
noncomputable def parents_contribution := tuition / 2
noncomputable def scholarship := 3000
noncomputable def wage_per_hour := 10
noncomputable def working_hours := 200
noncomputable def earnings := wage_per_hour * working_hours
noncomputable def total_scholarship_and_work := scholarship + earnings
noncomputable def remaining_tuition := tuition - parents_contribution - total_scholarship_and_work
noncomputable def student_loan := remaining_tuition

theorem loan_to_scholarship_ratio :
  (student_loan / scholarship) = 2 := 
by
  sorry

end loan_to_scholarship_ratio_l146_146184


namespace max_points_in_equilateral_property_set_l146_146690

theorem max_points_in_equilateral_property_set (Γ : Finset (ℝ × ℝ)) :
  (∀ (A B : (ℝ × ℝ)), A ∈ Γ → B ∈ Γ → 
    ∃ C : (ℝ × ℝ), C ∈ Γ ∧ 
    dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B) → Γ.card ≤ 3 :=
by
  intro h
  sorry

end max_points_in_equilateral_property_set_l146_146690


namespace problem_solution_l146_146504

def is_invertible_modulo_9 (a : ℕ) : Prop := Int.gcd a 9 = 1

theorem problem_solution (a b c d : ℕ) 
  (h1 : a < 9) (h2 : b < 9) (h3 : c < 9) (h4 : d < 9)
  (h5 : a ≠ b) (h6 : a ≠ c) (h7 : a ≠ d)
  (h8 : b ≠ c) (h9 : b ≠ d) (h10 : c ≠ d)
  (h11 : is_invertible_modulo_9 a)
  (h12 : is_invertible_modulo_9 b)
  (h13 : is_invertible_modulo_9 c)
  (h14 : is_invertible_modulo_9 d) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) *
   Nat.gcd_a (a * b * c * d) 9) % 9 = 6 :=
by sorry

end problem_solution_l146_146504


namespace min_days_equal_shifts_l146_146365

theorem min_days_equal_shifts (k n : ℕ) (h : 9 * k + 10 * n = 66) : k + n = 7 :=
sorry

end min_days_equal_shifts_l146_146365


namespace group_c_right_angled_triangle_l146_146692

theorem group_c_right_angled_triangle :
  (3^2 + 4^2 = 5^2) := by
  sorry

end group_c_right_angled_triangle_l146_146692


namespace minimum_value_7a_4b_l146_146904

noncomputable def original_cond (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (2 / (3 * a + b)) + (1 / (a + 2 * b)) = 4

theorem minimum_value_7a_4b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  original_cond a b ha hb → 7 * a + 4 * b = 9 / 4 :=
by
  sorry

end minimum_value_7a_4b_l146_146904


namespace total_sold_l146_146007

theorem total_sold (D C : ℝ) (h1 : D = 1.6 * C) (h2 : D = 168) : D + C = 273 :=
by
  sorry

end total_sold_l146_146007


namespace lowest_score_l146_146356

-- Define the conditions
def test_scores (s1 s2 s3 : ℕ) := s1 = 86 ∧ s2 = 112 ∧ s3 = 91
def max_score := 120
def target_average := 95
def num_tests := 5
def total_points_needed := target_average * num_tests

-- Define the proof statement
theorem lowest_score 
  (s1 s2 s3 : ℕ)
  (condition1 : test_scores s1 s2 s3)
  (max_pts : ℕ := max_score) 
  (target_avg : ℕ := target_average) 
  (num_tests : ℕ := num_tests)
  (total_needed : ℕ := total_points_needed) :
  ∃ s4 s5 : ℕ, s4 ≤ max_pts ∧ s5 ≤ max_pts ∧ s4 + s5 + s1 + s2 + s3 = total_needed ∧ (s4 = 66 ∨ s5 = 66) :=
by
  sorry

end lowest_score_l146_146356


namespace sphere_surface_area_ratio_l146_146916

axiom prism_has_circumscribed_sphere : Prop
axiom prism_has_inscribed_sphere : Prop

theorem sphere_surface_area_ratio 
  (h1 : prism_has_circumscribed_sphere)
  (h2 : prism_has_inscribed_sphere) : 
  ratio_surface_area_of_circumscribed_to_inscribed_sphere = 5 :=
sorry

end sphere_surface_area_ratio_l146_146916


namespace find_a_l146_146715

-- Definitions given in the conditions
def f (x : ℝ) : ℝ := x^2 - 2
def g (x : ℝ) : ℝ := x^2 + 6

-- The main theorem to show
theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : f (g a) = 18) : a = Real.sqrt 14 := sorry

end find_a_l146_146715


namespace trisha_bought_amount_initially_l146_146384

-- Define the amounts spent on each item
def meat : ℕ := 17
def chicken : ℕ := 22
def veggies : ℕ := 43
def eggs : ℕ := 5
def dogs_food : ℕ := 45
def amount_left : ℕ := 35

-- Define the total amount spent
def total_spent : ℕ := meat + chicken + veggies + eggs + dogs_food

-- Define the amount brought at the beginning
def amount_brought_at_beginning : ℕ := total_spent + amount_left

-- Theorem stating the amount Trisha brought at the beginning is 167
theorem trisha_bought_amount_initially : amount_brought_at_beginning = 167 := by
  -- Formal proof would go here, we use sorry to skip the proof
  sorry

end trisha_bought_amount_initially_l146_146384


namespace solve_for_x0_l146_146179

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 2 then x^2 + 2 else 2 * x

theorem solve_for_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 ∨ x0 = - Real.sqrt 6 :=
  by
  sorry

end solve_for_x0_l146_146179


namespace maximum_area_of_flower_bed_l146_146758

-- Definitions based on conditions
def length_of_flower_bed : ℝ := 150
def total_fencing : ℝ := 450

-- Question reframed as a proof statement
theorem maximum_area_of_flower_bed :
  ∀ (w : ℝ), 2 * w + length_of_flower_bed = total_fencing → (length_of_flower_bed * w = 22500) :=
by
  intro w h
  sorry

end maximum_area_of_flower_bed_l146_146758


namespace quadratic_roots_ratio_l146_146754

theorem quadratic_roots_ratio (k : ℝ) (k1 k2 : ℝ) (a b : ℝ) 
  (h_roots : ∀ x : ℝ, k * x * x + (1 - 6 * k) * x + 8 = 0 ↔ (x = a ∨ x = b))
  (h_ab : a ≠ b)
  (h_cond : a / b + b / a = 3 / 7)
  (h_ks : k^1 - 6 * (k1 + k2) + 8 = 0)
  (h_vieta : k1 + k2 = 200 / 36 ∧ k1 * k2 = 49 / 36) : 
  (k1 / k2 + k2 / k1 = 6.25) :=
by sorry

end quadratic_roots_ratio_l146_146754


namespace MarysTotalCandies_l146_146348

-- Definitions for the conditions
def MegansCandies : Nat := 5
def MarysInitialCandies : Nat := 3 * MegansCandies
def MarysCandiesAfterAdding : Nat := MarysInitialCandies + 10

-- Theorem to prove that Mary has 25 pieces of candy in total
theorem MarysTotalCandies : MarysCandiesAfterAdding = 25 :=
by
  sorry

end MarysTotalCandies_l146_146348


namespace perpendicular_lines_l146_146918

theorem perpendicular_lines (m : ℝ) : 
    (∀ x y : ℝ, x - 2 * y + 5 = 0) ∧ (∀ x y : ℝ, 2 * x + m * y - 6 = 0) → m = -1 :=
by
  sorry

end perpendicular_lines_l146_146918


namespace range_of_m_l146_146475

noncomputable def G (x : ℝ) (m : ℝ) : ℝ := (8 * x ^ 2 + 24 * x + 5 * m) / 8

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ c : ℝ, G x m = (x + c) ^ 2 ∧ c ^ 2 = 3) → 4 ≤ m ∧ m ≤ 5 := 
by
  sorry

end range_of_m_l146_146475


namespace carrots_picked_next_day_l146_146941

theorem carrots_picked_next_day :
  ∀ (initial_picked thrown_out additional_picked total : ℕ),
    initial_picked = 48 →
    thrown_out = 11 →
    total = 52 →
    additional_picked = total - (initial_picked - thrown_out) →
    additional_picked = 15 :=
by
  intros initial_picked thrown_out additional_picked total h_ip h_to h_total h_ap
  sorry

end carrots_picked_next_day_l146_146941


namespace one_interior_angle_of_polygon_with_five_diagonals_l146_146152

theorem one_interior_angle_of_polygon_with_five_diagonals (n : ℕ) (h : n - 3 = 5) :
  let interior_angle := 180 * (n - 2) / n
  interior_angle = 135 :=
by
  sorry

end one_interior_angle_of_polygon_with_five_diagonals_l146_146152


namespace sum_of_first_49_primes_is_10787_l146_146269

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                                79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
                                167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227]

theorem sum_of_first_49_primes_is_10787:
  first_49_primes.sum = 10787 :=
by
  -- Proof would go here.
  -- This is just a placeholder as per the requirements.
  sorry

end sum_of_first_49_primes_is_10787_l146_146269


namespace no_perfect_power_l146_146959

theorem no_perfect_power (n m : ℕ) (hn : 0 < n) (hm : 1 < m) : 102 ^ 1991 + 103 ^ 1991 ≠ n ^ m := 
sorry

end no_perfect_power_l146_146959


namespace pentagon_rectangle_ratio_l146_146242

theorem pentagon_rectangle_ratio (p w l : ℝ) (h₁ : 5 * p = 20) (h₂ : l = 2 * w) (h₃ : 2 * l + 2 * w = 20) : p / w = 6 / 5 :=
by
  sorry

end pentagon_rectangle_ratio_l146_146242


namespace total_snakes_l146_146950

/-
  Problem: Mary sees three breeding balls with 8 snakes each and 6 additional pairs of snakes.
           How many snakes did she see total?
  Conditions:
    - There are 3 breeding balls.
    - Each breeding ball has 8 snakes.
    - There are 6 additional pairs of snakes.
  Answer: 36 snakes
-/

theorem total_snakes (balls : ℕ) (snakes_per_ball : ℕ) (pairs : ℕ) (snakes_per_pair : ℕ) :
    balls = 3 → snakes_per_ball = 8 → pairs = 6 → snakes_per_pair = 2 →
    (balls * snakes_per_ball) + (pairs * snakes_per_pair) = 36 :=
  by 
    intros hb hspb hp hsp
    sorry

end total_snakes_l146_146950


namespace optimal_saving_is_45_cents_l146_146533

def initial_price : ℝ := 18
def fixed_discount : ℝ := 3
def percentage_discount : ℝ := 0.15

def price_after_fixed_discount (price fixed_discount : ℝ) : ℝ :=
  price - fixed_discount

def price_after_percentage_discount (price percentage_discount : ℝ) : ℝ :=
  price * (1 - percentage_discount)

def optimal_saving (initial_price fixed_discount percentage_discount : ℝ) : ℝ :=
  let price1 := price_after_fixed_discount initial_price fixed_discount
  let final_price1 := price_after_percentage_discount price1 percentage_discount
  let price2 := price_after_percentage_discount initial_price percentage_discount
  let final_price2 := price_after_fixed_discount price2 fixed_discount
  final_price1 - final_price2

theorem optimal_saving_is_45_cents : optimal_saving initial_price fixed_discount percentage_discount = 0.45 :=
by 
  sorry

end optimal_saving_is_45_cents_l146_146533


namespace complete_the_square_3x2_9x_20_l146_146595

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l146_146595


namespace hypotenuse_length_l146_146326

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l146_146326


namespace marys_total_candy_l146_146349

/-
Given:
1. Mary has 3 times as much candy as Megan.
2. Megan has 5 pieces of candy.
3. Mary adds 10 more pieces of candy to her collection.

Show that the total number of pieces of candy Mary has is 25.
-/

theorem marys_total_candy :
  (∀ (mary_candy megan_candy : ℕ), mary_candy = 3 * megan_candy → megan_candy = 5 → mary_candy + 10 = 25) :=
begin
  intros mary_candy megan_candy h1 h2,
  rw h2 at h1,
  rw h1,
  norm_num,
end

end marys_total_candy_l146_146349


namespace students_ages_average_l146_146289

variables (a b c : ℕ)

theorem students_ages_average (h1 : (14 * a + 13 * b + 12 * c) = 13 * (a + b + c)) : a = c :=
by
  sorry

end students_ages_average_l146_146289


namespace marks_in_math_l146_146697

theorem marks_in_math (e p c b : ℕ) (avg : ℚ) (n : ℕ) (total_marks_other_subjects : ℚ) :
  e = 45 →
  p = 52 →
  c = 47 →
  b = 55 →
  avg = 46.8 →
  n = 5 →
  total_marks_other_subjects = (e + p + c + b : ℕ) →
  (avg * n) - total_marks_other_subjects = 35 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end marks_in_math_l146_146697


namespace ordered_pairs_condition_l146_146279

theorem ordered_pairs_condition (m n : ℕ) (hmn : m ≥ n) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_eq : 3 * m * n = 8 * (m + n - 1)) :
    (m, n) = (16, 3) ∨ (m, n) = (6, 4) := by
  sorry

end ordered_pairs_condition_l146_146279


namespace cylinder_radius_l146_146048

theorem cylinder_radius (h r: ℝ) (S: ℝ) (S_eq: S = 130 * Real.pi) (h_eq: h = 8) 
    (surface_area_eq: S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) : 
    r = 5 :=
by {
  -- Placeholder for proof steps.
  sorry
}

end cylinder_radius_l146_146048


namespace common_ratio_of_geometric_series_l146_146805

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l146_146805


namespace students_like_both_l146_146451

-- Definitions based on given conditions
def total_students : ℕ := 500
def students_like_mountains : ℕ := 289
def students_like_sea : ℕ := 337
def students_like_neither : ℕ := 56

-- Statement to prove
theorem students_like_both : 
  students_like_mountains + students_like_sea - 182 + students_like_neither = total_students := 
by
  sorry

end students_like_both_l146_146451


namespace quadratic_form_h_l146_146590

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l146_146590


namespace ellipse_m_gt_5_l146_146747

theorem ellipse_m_gt_5 (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2) → m > 5 :=
by
  intros h
  sorry

end ellipse_m_gt_5_l146_146747


namespace ten_year_old_dog_is_64_human_years_l146_146665

namespace DogYears

-- Definition of the conditions
def first_year_in_human_years : ℕ := 15
def second_year_in_human_years : ℕ := 9
def subsequent_year_in_human_years : ℕ := 5

-- Definition of the total human years for a 10-year-old dog.
def dog_age_in_human_years (dog_age : ℕ) : ℕ :=
  if dog_age = 1 then first_year_in_human_years
  else if dog_age = 2 then first_year_in_human_years + second_year_in_human_years
  else first_year_in_human_years + second_year_in_human_years + (dog_age - 2) * subsequent_year_in_human_years

-- The statement to prove
theorem ten_year_old_dog_is_64_human_years : dog_age_in_human_years 10 = 64 :=
  by
    sorry

end DogYears

end ten_year_old_dog_is_64_human_years_l146_146665


namespace solve_for_t_l146_146196

theorem solve_for_t : ∃ t : ℝ, 3 * 3^t + Real.sqrt (9 * 9^t) = 18 ∧ t = 1 :=
by
  sorry

end solve_for_t_l146_146196


namespace find_m_l146_146570

theorem find_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 ^ a = m) (h4 : 3 ^ b = m) (h5 : 2 * a * b = a + b) : m = Real.sqrt 6 :=
sorry

end find_m_l146_146570


namespace john_annual_profit_is_1800_l146_146928

def tenant_A_monthly_payment : ℕ := 350
def tenant_B_monthly_payment : ℕ := 400
def tenant_C_monthly_payment : ℕ := 450
def john_monthly_rent : ℕ := 900
def utility_cost : ℕ := 100
def maintenance_fee : ℕ := 50

noncomputable def annual_profit : ℕ :=
  let total_monthly_income := tenant_A_monthly_payment + tenant_B_monthly_payment + tenant_C_monthly_payment
  let total_monthly_expenses := john_monthly_rent + utility_cost + maintenance_fee
  let monthly_profit := total_monthly_income - total_monthly_expenses
  monthly_profit * 12

theorem john_annual_profit_is_1800 : annual_profit = 1800 := by
  sorry

end john_annual_profit_is_1800_l146_146928


namespace john_spends_6_dollars_l146_146472

-- Let treats_per_day, cost_per_treat, and days_in_month be defined by the conditions of the problem.
def treats_per_day : ℕ := 2
def cost_per_treat : ℝ := 0.1
def days_in_month : ℕ := 30

-- The total expenditure should be defined as the number of treats multiplied by their cost.
def total_number_of_treats := treats_per_day * days_in_month
def total_expenditure := total_number_of_treats * cost_per_treat

-- The statement to be proven: John spends $6 on the treats.
theorem john_spends_6_dollars :
  total_expenditure = 6 :=
sorry

end john_spends_6_dollars_l146_146472


namespace remainder_when_dividing_928927_by_6_l146_146827

theorem remainder_when_dividing_928927_by_6 :
  928927 % 6 = 1 :=
by
  sorry

end remainder_when_dividing_928927_by_6_l146_146827


namespace range_cos_2alpha_cos_2beta_l146_146902

variable (α β : ℝ)
variable (h : Real.sin α + Real.cos β = 3 / 2)

theorem range_cos_2alpha_cos_2beta :
  -3/2 ≤ Real.cos (2 * α) + Real.cos (2 * β) ∧ Real.cos (2 * α) + Real.cos (2 * β) ≤ 3/2 :=
sorry

end range_cos_2alpha_cos_2beta_l146_146902


namespace P_eight_value_l146_146491

def quadratic_polynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℤ, P(x) = a * x^2 + b * x + c

theorem P_eight_value (P : ℤ → ℤ)
  (H : ∀ x : ℤ, P(P(x)) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4)
  (H_quad : quadratic_polynomial P) :
  P 8 = 58 :=
sorry

end P_eight_value_l146_146491


namespace area_shaded_region_l146_146614

-- Define the conditions in Lean

def semicircle_radius_ADB : ℝ := 2
def semicircle_radius_BEC : ℝ := 2
def midpoint_arc_ADB (D : ℝ) : Prop := D = semicircle_radius_ADB
def midpoint_arc_BEC (E : ℝ) : Prop := E = semicircle_radius_BEC
def semicircle_radius_DFE : ℝ := 1
def midpoint_arc_DFE (F : ℝ) : Prop := F = semicircle_radius_DFE

-- Given the mentioned conditions, we want to show the area of the shaded region is 8 square units
theorem area_shaded_region 
  (D E F : ℝ) 
  (hD : midpoint_arc_ADB D)
  (hE : midpoint_arc_BEC E)
  (hF : midpoint_arc_DFE F) : 
  ∃ (area : ℝ), area = 8 := 
sorry

end area_shaded_region_l146_146614


namespace inradius_of_triangle_l146_146327

variable (A : ℝ) (p : ℝ) (r : ℝ) (s : ℝ)

theorem inradius_of_triangle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_l146_146327


namespace point_M_coordinates_l146_146190

theorem point_M_coordinates :
  (∃ (M : ℝ × ℝ), M.1 < 0 ∧ M.2 > 0 ∧ abs M.2 = 2 ∧ abs M.1 = 1 ∧ M = (-1, 2)) :=
by
  use (-1, 2)
  sorry

end point_M_coordinates_l146_146190


namespace ratio_Andrea_Jude_l146_146213

-- Definitions
def number_of_tickets := 100
def tickets_left := 40
def tickets_sold := number_of_tickets - tickets_left

def Jude_tickets := 16
def Sandra_tickets := 4 + 1/2 * Jude_tickets
def Andrea_tickets := tickets_sold - (Jude_tickets + Sandra_tickets)

-- Assertion that needs proof
theorem ratio_Andrea_Jude : 
  (Andrea_tickets / Jude_tickets) = 2 := by
  sorry

end ratio_Andrea_Jude_l146_146213


namespace largest_base5_number_conversion_l146_146070

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l146_146070


namespace tank_overflow_time_l146_146520

noncomputable def pipeARate : ℚ := 1 / 32
noncomputable def pipeBRate : ℚ := 3 * pipeARate
noncomputable def combinedRate (rateA rateB : ℚ) : ℚ := rateA + rateB

theorem tank_overflow_time : 
  combinedRate pipeARate pipeBRate = 1 / 8 ∧ (1 / combinedRate pipeARate pipeBRate = 8) :=
by
  sorry

end tank_overflow_time_l146_146520


namespace find_angleZ_l146_146345

-- Definitions of angles
def angleX : ℝ := 100
def angleY : ℝ := 130

-- Define Z angle based on the conditions given
def angleZ : ℝ := 130

theorem find_angleZ (p q : Prop) (parallel_pq : p ∧ q)
  (h1 : angleX = 100)
  (h2 : angleY = 130) :
  angleZ = 130 :=
by
  sorry

end find_angleZ_l146_146345


namespace prob_all_same_room_correct_prob_at_least_two_same_room_correct_l146_146818

-- Given conditions: three people and four rooms, each equally likely for each person.
def people := {1, 2, 3}
def rooms := {1, 2, 3, 4}
def assignments := {f : people → rooms // ∀ p ∈ people, f p ∈ rooms}

-- Define the probability space
noncomputable def prob_space : ProbabilityMassFunction assignments := sorry

-- Probability that all three people are assigned to the same room.
noncomputable def prob_all_same_room : ℚ :=
  let event := {f : assignments // ∃ r ∈ rooms, ∀ p ∈ people, f.val p = r}
  ProbabilityMassFunction.probability prob_space event

-- Probability that at least two people are assigned to the same room.
noncomputable def prob_at_least_two_same_room : ℚ :=
  1 - ProbabilityMassFunction.probability prob_space
    {f : assignments // ∀ p1 p2 ∈ people, p1 ≠ p2 → f.val p1 ≠ f.val p2}

-- Theorems
theorem prob_all_same_room_correct : prob_all_same_room = 1 / 16 := sorry

theorem prob_at_least_two_same_room_correct : prob_at_least_two_same_room = 5 / 8 := sorry

end prob_all_same_room_correct_prob_at_least_two_same_room_correct_l146_146818


namespace value_of_m_l146_146716

theorem value_of_m (m : ℤ) : (|m| = 1) ∧ (m + 1 ≠ 0) → m = 1 := by
  sorry

end value_of_m_l146_146716


namespace range_of_a_l146_146004

-- Definitions derived from conditions
def is_ellipse_with_foci_on_x_axis (a : ℝ) : Prop := a^2 > a + 6 ∧ a + 6 > 0

-- Theorem representing the proof problem
theorem range_of_a (a : ℝ) (h : is_ellipse_with_foci_on_x_axis a) :
  (a > 3) ∨ (-6 < a ∧ a < -2) :=
sorry

end range_of_a_l146_146004


namespace quadratic_polynomial_P8_l146_146486

theorem quadratic_polynomial_P8 :
  ∃ (a b c : ℝ), 
  (∀ x : ℝ, P x = a * x^2 + b * x + c) ∧ 
  (∀ x : ℝ, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) ∧ 
  P 8 = 58 :=
begin
  sorry
end

end quadratic_polynomial_P8_l146_146486


namespace solve_for_y_l146_146195

theorem solve_for_y (y : ℤ) : 7 * (4 * y + 5) - 4 = -3 * (2 - 9 * y) → y = -37 :=
by
  intro h
  sorry

end solve_for_y_l146_146195


namespace choose_students_l146_146236

/-- There are 50 students in the class, including one class president and one vice-president. 
    We want to select 5 students to participate in an activity such that at least one of 
    the class president or vice-president is included. We assert that there are exactly 2 
    distinct methods for making this selection. -/
theorem choose_students (students : Finset ℕ) (class_president vice_president : ℕ) (students_card : students.card = 50)
  (students_ex : class_president ∈ students ∧ vice_president ∈ students) : 
  ∃ valid_methods : Finset (Finset ℕ), valid_methods.card = 2 :=
by
  sorry

end choose_students_l146_146236


namespace f_2018_eq_2017_l146_146685

-- Define f(1) and f(2)
def f : ℕ → ℕ 
| 1 => 1
| 2 => 1
| n => if h : n ≥ 3 then (f (n - 1) - f (n - 2) + n) else 0

-- State the theorem to prove f(2018) = 2017
theorem f_2018_eq_2017 : f 2018 = 2017 := 
by 
  sorry

end f_2018_eq_2017_l146_146685


namespace largest_base5_number_conversion_l146_146068

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l146_146068


namespace athena_total_spent_l146_146744

-- Define the conditions
def sandwiches_quantity : ℕ := 3
def sandwich_price : ℝ := 3.0
def drinks_quantity : ℕ := 2
def drink_price : ℝ := 2.5

-- Define the calculations
def total_sandwich_cost : ℝ := sandwiches_quantity * sandwich_price
def total_drink_cost : ℝ := drinks_quantity * drink_price

-- Define the total cost
def total_amount_spent : ℝ := total_sandwich_cost + total_drink_cost

-- Prove the total amount spent
theorem athena_total_spent : total_amount_spent = 14 := by
  sorry

end athena_total_spent_l146_146744


namespace cone_height_l146_146683

theorem cone_height (r : ℝ) (n : ℕ) (circumference : ℝ) 
  (sector_circumference : ℝ) (base_radius : ℝ) (slant_height : ℝ) 
  (h : ℝ) : 
  r = 8 →
  n = 4 →
  circumference = 2 * Real.pi * r →
  sector_circumference = circumference / n →
  base_radius = sector_circumference / (2 * Real.pi) →
  slant_height = r →
  h = Real.sqrt (slant_height^2 - base_radius^2) →
  h = 2 * Real.sqrt 15 := 
by
  intros
  sorry

end cone_height_l146_146683


namespace natalie_height_l146_146953

variable (height_Natalie height_Harpreet height_Jiayin : ℝ)
variable (h1 : height_Natalie = height_Harpreet)
variable (h2 : height_Jiayin = 161)
variable (h3 : (height_Natalie + height_Harpreet + height_Jiayin) / 3 = 171)

theorem natalie_height : height_Natalie = 176 :=
by 
  sorry

end natalie_height_l146_146953


namespace simplify_absolute_value_l146_146038

theorem simplify_absolute_value : abs (-(5^2) + 6 * 2) = 13 := by
  sorry

end simplify_absolute_value_l146_146038


namespace plant_ways_count_l146_146661

theorem plant_ways_count :
  ∃ (solutions : Finset (Fin 7 → ℕ)), 
    (∀ x ∈ solutions, (x 0 + x 1 + x 2 + x 3 + x 4 + x 5 = 10) ∧ 
                       (100 * x 0 + 200 * x 1 + 300 * x 2 + 150 * x 3 + 125 * x 4 + 125 * x 5 = 2500)) ∧
    (solutions.card = 8) :=
sorry

end plant_ways_count_l146_146661


namespace probability_white_given_popped_l146_146680

theorem probability_white_given_popped :
  (3/4 : ℚ) * (3/5 : ℚ) / ((3/4 : ℚ) * (3/5 : ℚ) + (1/4 : ℚ) * (1/2 : ℚ)) = 18/23 := by
  sorry

end probability_white_given_popped_l146_146680


namespace problem_l146_146120

def op (x y : ℝ) : ℝ := x^2 - y

theorem problem (h : ℝ) : op h (op h h) = h :=
by
  sorry

end problem_l146_146120


namespace fixed_points_l146_146275

noncomputable def f (x : ℝ) : ℝ := x^2 - x - 3

theorem fixed_points : { x : ℝ | f x = x } = { -1, 3 } :=
by
  sorry

end fixed_points_l146_146275


namespace minimum_value_of_f_l146_146721

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ -1 / Real.exp 1) ∧ (∃ x : ℝ, x > 0 ∧ f x = -1 / Real.exp 1) :=
by
  sorry

end minimum_value_of_f_l146_146721


namespace intersection_A_B_l146_146901

def A : Set ℝ := { x | x^2 - 2*x < 0 }
def B : Set ℝ := { x | |x| > 1 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end intersection_A_B_l146_146901


namespace rational_coefficient_exists_in_binomial_expansion_l146_146280

theorem rational_coefficient_exists_in_binomial_expansion :
  ∃! (n : ℕ), n > 0 ∧ (∀ r, (r % 3 = 0 → (n - r) % 2 = 0 → n = 7)) :=
by
  sorry

end rational_coefficient_exists_in_binomial_expansion_l146_146280


namespace lily_milk_amount_l146_146343

def initial_milk : ℚ := 5
def milk_given_to_james : ℚ := 18 / 4
def milk_received_from_neighbor : ℚ := 7 / 4

theorem lily_milk_amount : (initial_milk - milk_given_to_james + milk_received_from_neighbor) = 9 / 4 :=
by
  sorry

end lily_milk_amount_l146_146343


namespace remainder_of_7n_div_4_l146_146832

theorem remainder_of_7n_div_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
sorry

end remainder_of_7n_div_4_l146_146832


namespace minimize_sum_l146_146440

noncomputable def objective_function (x : ℝ) : ℝ := x + x^2

theorem minimize_sum : ∃ x : ℝ, (objective_function x = x + x^2) ∧ (∀ y : ℝ, objective_function y ≥ objective_function (-1/2)) :=
by
  sorry

end minimize_sum_l146_146440


namespace prob_exceeds_2100_l146_146934

open ProbabilityTheory MeasureTheory

noncomputable def normal_dist (μ σ : ℝ) : Measure ℝ := {
  to_fun := λ s, ∫ x in s, PDF (normalPDF μ σ) x,
  zero := by sorry,
  add := by sorry
}

theorem prob_exceeds_2100 :
  ∀ (ξ : Measure ℝ), 
    (ξ = normal_dist 2000 100) → 
    (∫⁻ x, ↑(if x > 2100 then 1 else 0) ∂ξ) = 0.1587 :=
by sorry

end prob_exceeds_2100_l146_146934


namespace maximum_value_expression_l146_146823

theorem maximum_value_expression (a b c : ℕ) (ha : 0 < a ∧ a ≤ 9) (hb : 0 < b ∧ b ≤ 9) (hc : 0 < c ∧ c ≤ 9) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ∃ (v : ℚ), v = (1 / (a + 2010 / (b + 1 / c : ℚ))) ∧ v ≤ (1 / 203) :=
sorry

end maximum_value_expression_l146_146823


namespace difference_received_from_parents_l146_146560

-- Define conditions
def amount_from_mom := 8
def amount_from_dad := 5

-- Question: Prove the difference between amount_from_mom and amount_from_dad is 3
theorem difference_received_from_parents : (amount_from_mom - amount_from_dad) = 3 :=
by
  sorry

end difference_received_from_parents_l146_146560


namespace approx_change_in_y_l146_146993

-- Definition of the function
def y (x : ℝ) : ℝ := x^3 - 7 * x^2 + 80

-- Derivative of the function, calculated manually
def y_prime (x : ℝ) : ℝ := 3 * x^2 - 14 * x

-- The change in x
def delta_x : ℝ := 0.01

-- The given value of x
def x_initial : ℝ := 5

-- To be proved: the approximate change in y
theorem approx_change_in_y : (y_prime x_initial) * delta_x = 0.05 :=
by
  -- Imported and recognized theorem verifications skipped
  sorry

end approx_change_in_y_l146_146993


namespace sufficient_not_necessary_condition_m_eq_1_sufficient_m_eq_1_not_necessary_l146_146911

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (4, -2)

def perp_vectors (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem sufficient_not_necessary_condition :
  perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) ↔ (m = 1 ∨ m = -3) :=
by
  sorry

theorem m_eq_1_sufficient :
  (m = 1) → perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) :=
by
  sorry

theorem m_eq_1_not_necessary :
  perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) → (m = 1 ∨ m = -3) :=
by
  sorry

end sufficient_not_necessary_condition_m_eq_1_sufficient_m_eq_1_not_necessary_l146_146911


namespace intersection_of_sets_l146_146178

-- Defining set M
def M : Set ℝ := { x | x^2 + x - 2 < 0 }

-- Defining set N
def N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

-- Theorem stating the solution
theorem intersection_of_sets : M ∩ N = { x : ℝ | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_sets_l146_146178


namespace common_ratio_of_geometric_series_l146_146809

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l146_146809


namespace solve_equation_solve_inequality_system_l146_146406

theorem solve_equation :
  ∃ x, 2 * x^2 - 4 * x - 1 = 0 ∧ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
sorry

theorem solve_inequality_system : 
  ∀ x, (2 * x + 3 > 1 → -1 < x) ∧
       (x - 2 ≤ (1 / 2) * (x + 2) → x ≤ 6) ∧ 
       (2 * x + 3 > 1 ∧ x - 2 ≤ (1 / 2) * (x + 2) ↔ (-1 < x ∧ x ≤ 6)) :=
sorry

end solve_equation_solve_inequality_system_l146_146406


namespace initial_markers_count_l146_146028

   -- Let x be the initial number of markers Megan had.
   variable (x : ℕ)

   -- Conditions:
   def robert_gave_109_markers : Prop := true
   def total_markers_after_adding : ℕ := 326
   def markers_added_by_robert : ℕ := 109

   -- The total number of markers Megan has now is 326.
   def total_markers_eq (x : ℕ) : Prop := x + markers_added_by_robert = total_markers_after_adding

   -- Prove that initially Megan had 217 markers.
   theorem initial_markers_count : total_markers_eq 217 := by
     sorry
   
end initial_markers_count_l146_146028


namespace hypotenuse_length_l146_146318

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l146_146318


namespace walking_east_of_neg_west_l146_146461

-- Define the representation of directions
def is_walking_west (d : ℕ) (x : ℤ) : Prop := x = d
def is_walking_east (d : ℕ) (x : ℤ) : Prop := x = -d

-- Given the condition and states the relationship is the proposition to prove.
theorem walking_east_of_neg_west (d : ℕ) (x : ℤ) (h : is_walking_west 2 2) : is_walking_east 5 (-5) :=
by
  sorry

end walking_east_of_neg_west_l146_146461


namespace technicians_count_l146_146328

/-- Given a workshop with 49 workers, where the average salary of all workers 
    is Rs. 8000, the average salary of the technicians is Rs. 20000, and the
    average salary of the rest is Rs. 6000, prove that the number of 
    technicians is 7. -/
theorem technicians_count (T R : ℕ) (h1 : T + R = 49) (h2 : 10 * T + 3 * R = 196) : T = 7 := 
by
  sorry

end technicians_count_l146_146328


namespace find_solutions_l146_146125

theorem find_solutions :
  ∀ (x n : ℕ), 0 < x → 0 < n → x^(n+1) - (x + 1)^n = 2001 → (x, n) = (13, 2) :=
by
  intros x n hx hn heq
  sorry

end find_solutions_l146_146125


namespace total_deposit_amount_l146_146182

def markDeposit : ℕ := 88
def bryanDeposit (markAmount : ℕ) : ℕ := 5 * markAmount - 40
def totalDeposit (markAmount bryanAmount : ℕ) : ℕ := markAmount + bryanAmount

theorem total_deposit_amount : totalDeposit markDeposit (bryanDeposit markDeposit) = 488 := 
by sorry

end total_deposit_amount_l146_146182


namespace pairball_playing_time_l146_146194

-- Define the conditions of the problem
def num_children : ℕ := 7
def total_minutes : ℕ := 105
def total_child_minutes : ℕ := 2 * total_minutes

-- Define the theorem to prove
theorem pairball_playing_time : total_child_minutes / num_children = 30 :=
by sorry

end pairball_playing_time_l146_146194


namespace no_real_solutions_for_eqn_l146_146197

theorem no_real_solutions_for_eqn :
  ¬ ∃ x : ℝ, (x + 4) ^ 2 = 3 * (x - 2) := 
by 
  sorry

end no_real_solutions_for_eqn_l146_146197


namespace team_points_l146_146534

theorem team_points (wins losses ties : ℕ) (points_per_win points_per_loss points_per_tie : ℕ) :
  wins = 9 → losses = 3 → ties = 4 → points_per_win = 2 → points_per_loss = 0 → points_per_tie = 1 →
  (points_per_win * wins + points_per_loss * losses + points_per_tie * ties = 22) :=
by
  intro h_wins h_losses h_ties h_points_per_win h_points_per_loss h_points_per_tie
  sorry

end team_points_l146_146534


namespace trig_identity_l146_146255

noncomputable def trig_expr := 
  4.34 * (Real.cos (28 * Real.pi / 180) * Real.cos (56 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) + 
  (Real.cos (2 * Real.pi / 180) * Real.cos (4 * Real.pi / 180) / Real.sin (28 * Real.pi / 180))

theorem trig_identity : 
  trig_expr = (Real.sqrt 3 * Real.sin (38 * Real.pi / 180)) / (4 * Real.sin (2 * Real.pi / 180) * Real.sin (28 * Real.pi / 180)) :=
by 
  sorry

end trig_identity_l146_146255


namespace seq_a22_l146_146981

def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0) ∧
  (a 10 = 10)

theorem seq_a22 : ∀ (a : ℕ → ℝ), seq a → a 22 = 10 :=
by
  intros a h,
  have h1 := h.1,
  have h99 := h.2.1,
  have h100 := h.2.2.1,
  have h_eq := h.2.2.2,
  sorry

end seq_a22_l146_146981


namespace problem_1_problem_2_l146_146994

noncomputable def k_value (θ₀ θ₁ θ : ℝ) (t : ℝ) :=
  real.log ((θ₁ - θ₀) / (θ - θ₀)) / -t

theorem problem_1 : 
  let θ₀ := 20 
      θ₁ := 98 
      θ := 71.2 
      t := 1 
  in abs (k_value θ₀ θ₁ θ t - 0.029) < 0.001 := 
by 
  sorry

noncomputable def room_temp (k θ₁ θ : ℝ) (t : ℝ) :=
  θ₁ + (θ - θ₁) * real.exp (k * t)

theorem problem_2 :
  let k := 0.01 
      θ₁ := 100 
      θ := 40 
      t := 2.5 
  in abs (room_temp k θ₁ θ t - 20.0) < 0.1 := 
by 
  sorry

end problem_1_problem_2_l146_146994


namespace complete_the_square_h_value_l146_146588

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l146_146588


namespace European_to_American_swallow_ratio_l146_146039

theorem European_to_American_swallow_ratio (a e : ℝ) (n_E : ℕ) 
  (h1 : a = 5)
  (h2 : 2 * n_E + n_E = 90)
  (h3 : 60 * a + 30 * e = 600) :
  e / a = 2 := 
by
  sorry

end European_to_American_swallow_ratio_l146_146039


namespace A_three_two_l146_146430

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m+1, 0 => A m 2
| m+1, n+1 => A m (A (m + 1) n)

theorem A_three_two : A 3 2 = 5 := 
by 
  sorry

end A_three_two_l146_146430


namespace total_flour_needed_l146_146170

-- Definitions of flour needed by Katie and Sheila
def katie_flour : ℕ := 3
def sheila_flour : ℕ := katie_flour + 2

-- Statement of the theorem
theorem total_flour_needed : katie_flour + sheila_flour = 8 := by
  -- The proof would go here
  sorry

end total_flour_needed_l146_146170


namespace sum_distances_l146_146351

noncomputable def lengthAB : ℝ := 2
noncomputable def lengthA'B' : ℝ := 5
noncomputable def midpointAB : ℝ := lengthAB / 2
noncomputable def midpointA'B' : ℝ := lengthA'B' / 2
noncomputable def distancePtoD : ℝ := 0.5
noncomputable def proportionality_constant : ℝ := lengthA'B' / lengthAB

theorem sum_distances : distancePtoD + (proportionality_constant * distancePtoD) = 1.75 := by
  sorry

end sum_distances_l146_146351


namespace evaluate_expression_l146_146122

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 := by
  sorry

end evaluate_expression_l146_146122


namespace find_g_l146_146551

noncomputable def g (x : ℝ) := -4 * x ^ 4 + x ^ 3 - 6 * x ^ 2 + x - 1

theorem find_g (x : ℝ) :
  4 * x ^ 4 + 2 * x ^ 2 - x + 7 + g x = x ^ 3 - 4 * x ^ 2 + 6 :=
by
  sorry

end find_g_l146_146551


namespace hypotenuse_length_l146_146323

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l146_146323


namespace initial_amount_liquid_A_l146_146238

-- Definitions and conditions
def initial_ratio (a : ℕ) (b : ℕ) := a = 4 * b
def replaced_mixture_ratio (a : ℕ) (b : ℕ) (r₀ r₁ : ℕ) := 4 * r₀ = 2 * (r₁ + 20)

-- Theorem to prove the initial amount of liquid A
theorem initial_amount_liquid_A (a b r₀ r₁ : ℕ) :
  initial_ratio a b → replaced_mixture_ratio a b r₀ r₁ → a = 16 := 
by
  sorry

end initial_amount_liquid_A_l146_146238


namespace exists_collinear_B_points_l146_146434

noncomputable def intersection (A B C D : Point) : Point :=
sorry

noncomputable def collinearity (P Q R S T : Point) : Prop :=
sorry

def convex_pentagon (A1 A2 A3 A4 A5 : Point) : Prop :=
-- Condition ensuring A1, A2, A3, A4, A5 form a convex pentagon, to be precisely defined
sorry

theorem exists_collinear_B_points :
  ∃ (A1 A2 A3 A4 A5 : Point),
    convex_pentagon A1 A2 A3 A4 A5 ∧
    collinearity
      (intersection A1 A4 A2 A3)
      (intersection A2 A5 A3 A4)
      (intersection A3 A1 A4 A5)
      (intersection A4 A2 A5 A1)
      (intersection A5 A3 A1 A2) :=
sorry

end exists_collinear_B_points_l146_146434


namespace total_clothes_count_l146_146457

theorem total_clothes_count (shirts_per_pants : ℕ) (pants : ℕ) (shirts : ℕ) : shirts_per_pants = 6 → pants = 40 → shirts = shirts_per_pants * pants → shirts + pants = 280 := by
  intro h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end total_clothes_count_l146_146457


namespace geometric_series_common_ratio_l146_146804

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l146_146804


namespace cost_difference_l146_146951

def TMobile_cost (num_lines : ℕ) : ℝ :=
  let base_cost := 50
  let additional_line_cost := 16
  let discount := 0.1
  let data_charge := 3
  let monthly_cost_before_discount := base_cost + (additional_line_cost * (num_lines - 2))
  let total_monthly_cost := monthly_cost_before_discount + (data_charge * num_lines)
  (total_monthly_cost * (1 - discount)) * 12

def MMobile_cost (num_lines : ℕ) : ℝ :=
  let base_cost := 45
  let additional_line_cost := 14
  let activation_fee := 20
  let monthly_cost := base_cost + (additional_line_cost * (num_lines - 2))
  (monthly_cost * 12) + (activation_fee * num_lines)

theorem cost_difference (num_lines : ℕ) (h : num_lines = 5) :
  TMobile_cost num_lines - MMobile_cost num_lines = 76.40 :=
  sorry

end cost_difference_l146_146951


namespace percentage_increase_l146_146925

variable (T : ℕ) (total_time : ℕ)

theorem percentage_increase (h1 : T = 4) (h2 : total_time = 10) : 
  ∃ P : ℕ, (T + P / 100 * T = total_time - T) → P = 50 := 
by 
  sorry

end percentage_increase_l146_146925


namespace num_comics_liked_by_males_l146_146484

-- Define the problem conditions
def num_comics : ℕ := 300
def percent_liked_by_females : ℕ := 30
def percent_disliked_by_both : ℕ := 30

-- Define the main theorem to prove
theorem num_comics_liked_by_males :
  let percent_liked_by_at_least_one_gender := 100 - percent_disliked_by_both
  let num_comics_liked_by_females := percent_liked_by_females * num_comics / 100
  let num_comics_liked_by_at_least_one_gender := percent_liked_by_at_least_one_gender * num_comics / 100
  num_comics_liked_by_at_least_one_gender - num_comics_liked_by_females = 120 :=
by
  sorry

end num_comics_liked_by_males_l146_146484


namespace ratio_blue_to_gold_l146_146508

-- Define the number of brown stripes
def brown_stripes : Nat := 4

-- Given condition: There are three times as many gold stripes as brown stripes
def gold_stripes : Nat := 3 * brown_stripes

-- Given condition: There are 60 blue stripes
def blue_stripes : Nat := 60

-- The actual statement to prove
theorem ratio_blue_to_gold : blue_stripes / gold_stripes = 5 := by
  -- Proof would go here
  sorry

end ratio_blue_to_gold_l146_146508


namespace percentage_profit_l146_146417

theorem percentage_profit (cp sp : ℝ) (h1 : cp = 1200) (h2 : sp = 1680) : ((sp - cp) / cp) * 100 = 40 := 
by 
  sorry

end percentage_profit_l146_146417


namespace diagonal_lt_half_perimeter_l146_146957

theorem diagonal_lt_half_perimeter (AB BC CD DA AC : ℝ) (h1 : AB > 0) (h2 : BC > 0) (h3 : CD > 0) (h4 : DA > 0) 
  (h_triangle1 : AC < AB + BC) (h_triangle2 : AC < AD + DC) :
  AC < (AB + BC + CD + DA) / 2 :=
by {
  sorry
}

end diagonal_lt_half_perimeter_l146_146957


namespace range_of_a_l146_146649

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := 
sorry

end range_of_a_l146_146649


namespace hypotenuse_length_l146_146297

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l146_146297


namespace natural_numbers_satisfy_inequality_l146_146553

theorem natural_numbers_satisfy_inequality:
  ∃ (a b c : ℕ), 
    a = 5 ∧ b = 9 ∧ c = 4 ∧ 
    ∀ n : ℕ, n > 2 → 
      b - (c / ((n-2) !)) < (∑ k in Finset.range (n+1).filter (λ x, x > 1), (k^3 - a) / (k !)) ∧ 
      (∑ k in Finset.range (n+1).filter (λ x, x > 1), (k^3 - a) / (k !)) < b :=
begin
  sorry
end

end natural_numbers_satisfy_inequality_l146_146553


namespace equivalence_of_X_conditions_l146_146935

theorem equivalence_of_X_conditions {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}
  {ξ X : Ω → ℝ}
  (h_ind : Independent ξ X)
  (h_dist_ξ : P {ω | ξ ω = 1} = 1 / 2 ∧ P {ω | ξ ω = -1} = 1 / 2) :
  (X ∼ᵈ (-X)) ↔ (X ∼ᵈ (ξ * X)) ↔ (X ∼ᵈ (ξ * |X|)) :=
sorry

end equivalence_of_X_conditions_l146_146935


namespace Peter_drew_more_l146_146768

theorem Peter_drew_more :
  ∃ (P : ℕ), 5 + P + (P + 20) = 41 ∧ (P - 5 = 3) :=
sorry

end Peter_drew_more_l146_146768


namespace circle_units_diff_l146_146855

-- Define the context where we verify the claim about the circle

noncomputable def radius : ℝ := 3
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def area (r : ℝ) := Real.pi * r ^ 2

-- Lean Theorem statement that needs to be proved
theorem circle_units_diff (r : ℝ) (h₀ : r = radius) :
  circumference r ≠ area r :=
by sorry

end circle_units_diff_l146_146855


namespace real_imaginary_part_above_x_axis_polynomial_solutions_l146_146678

-- Question 1: For what values of the real number m is (m^2 - 2m - 15) > 0
theorem real_imaginary_part_above_x_axis (m : ℝ) : 
  (m^2 - 2 * m - 15 > 0) ↔ (m < -3 ∨ m > 5) :=
sorry

-- Question 2: For what values of the real number m does 2m^2 + 3m - 4=0?
theorem polynomial_solutions (m : ℝ) : 
  (2 * m^2 + 3 * m - 4 = 0) ↔ (m = -3 ∨ m = 2) :=
sorry

end real_imaginary_part_above_x_axis_polynomial_solutions_l146_146678


namespace find_a0_find_a2_find_sum_a1_a2_a3_a4_l146_146134

lemma problem_conditions (x : ℝ) : 
  (x - 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 :=
sorry

theorem find_a0 :
  a_0 = 16 :=
sorry

theorem find_a2 :
  a_2 = 24 :=
sorry

theorem find_sum_a1_a2_a3_a4 :
  a_1 + a_2 + a_3 + a_4 = -15 :=
sorry

end find_a0_find_a2_find_sum_a1_a2_a3_a4_l146_146134


namespace arithmetic_seq_a6_l146_146276

theorem arithmetic_seq_a6 (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (0 < q) →
  a 1 = 1 →
  S 3 = 7/4 →
  S n = (1 - q^n) / (1 - q) →
  (∀ n, a n = 1 * q^(n - 1)) →
  a 6 = 1 / 32 :=
by
  sorry

end arithmetic_seq_a6_l146_146276


namespace circle_areas_sum_l146_146989

theorem circle_areas_sum {r s t : ℝ}
  (h1 : r + s = 5)
  (h2 : s + t = 12)
  (h3 : r + t = 13) :
  (Real.pi * r ^ 2 + Real.pi * s ^ 2 + Real.pi * t ^ 2) = 81 * Real.pi :=
by sorry

end circle_areas_sum_l146_146989


namespace complement_of_irreducible_proper_fraction_is_irreducible_l146_146766

theorem complement_of_irreducible_proper_fraction_is_irreducible 
  (a b : ℤ) (h0 : 0 < a) (h1 : a < b) (h2 : Int.gcd a b = 1) : Int.gcd (b - a) b = 1 :=
sorry

end complement_of_irreducible_proper_fraction_is_irreducible_l146_146766


namespace probability_of_drawing_K_is_2_over_27_l146_146668

-- Define the total number of cards in a standard deck of 54 cards
def total_cards : ℕ := 54

-- Define the number of "K" cards in the standard deck
def num_K_cards : ℕ := 4

-- Define the probability function for drawing a "K"
def probability_drawing_K (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

-- Prove that the probability of drawing a "K" is 2/27
theorem probability_of_drawing_K_is_2_over_27 :
  probability_drawing_K total_cards num_K_cards = 2 / 27 :=
by
  sorry

end probability_of_drawing_K_is_2_over_27_l146_146668


namespace quadratic_expression_rewriting_l146_146585

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l146_146585


namespace triangle_area_DEF_l146_146063

def point : Type := ℝ × ℝ

def D : point := (-2, 2)
def E : point := (8, 2)
def F : point := (6, -4)

theorem triangle_area_DEF :
  let base : ℝ := abs (D.1 - E.1)
  let height : ℝ := abs (F.2 - 2)
  let area := 1/2 * base * height
  area = 30 := 
by 
  sorry

end triangle_area_DEF_l146_146063


namespace problem_statement_l146_146879

theorem problem_statement 
  (x1 y1 x2 y2 x3 y3 x4 y4 a b c : ℝ)
  (h1 : x1 > 0) (h2 : y1 > 0)
  (h3 : x2 < 0) (h4 : y2 > 0)
  (h5 : x3 < 0) (h6 : y3 < 0)
  (h7 : x4 > 0) (h8 : y4 < 0)
  (h9 : (x1 - a)^2 + (y1 - b)^2 ≤ c^2)
  (h10 : (x2 - a)^2 + (y2 - b)^2 ≤ c^2)
  (h11 : (x3 - a)^2 + (y3 - b)^2 ≤ c^2)
  (h12 : (x4 - a)^2 + (y4 - b)^2 ≤ c^2) : a^2 + b^2 < c^2 :=
by sorry

end problem_statement_l146_146879


namespace ticket_price_divisors_count_l146_146687

theorem ticket_price_divisors_count :
  ∃ (x : ℕ), (36 % x = 0) ∧ (60 % x = 0) ∧ (Nat.divisors (Nat.gcd 36 60)).card = 6 := 
by
  sorry

end ticket_price_divisors_count_l146_146687


namespace largest_base5_number_to_base10_is_3124_l146_146067

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l146_146067


namespace sum_of_circle_areas_l146_146984

theorem sum_of_circle_areas (r s t : ℝ) (h1 : r + s = 5) (h2 : s + t = 12) (h3 : r + t = 13) :
  real.pi * (r^2 + s^2 + t^2) = 113 * real.pi :=
by
  sorry

end sum_of_circle_areas_l146_146984


namespace geometric_series_common_ratio_l146_146796

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l146_146796


namespace task2_probability_l146_146516

variable (P_task1_on_time P_task2_on_time : ℝ)

theorem task2_probability 
  (h1 : P_task1_on_time = 5 / 8)
  (h2 : (P_task1_on_time * (1 - P_task2_on_time)) = 0.25) :
  P_task2_on_time = 3 / 5 := by
  sorry

end task2_probability_l146_146516


namespace heesu_has_greatest_sum_l146_146360

theorem heesu_has_greatest_sum :
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  have h1 : Heesu_sum > Sora_sum := by sorry
  have h2 : Heesu_sum > Jiyeon_sum := by sorry
  exact And.intro h1 h2

end heesu_has_greatest_sum_l146_146360


namespace DongfangElementary_total_students_l146_146262

theorem DongfangElementary_total_students (x y : ℕ) 
  (h1 : x = y + 2)
  (h2 : 10 * (y + 2) = 22 * 11 * (y - 22))
  (h3 : x - x / 11 = 2 * (y - 22)) :
  x + y = 86 :=
by
  sorry

end DongfangElementary_total_students_l146_146262


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l146_146305

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l146_146305


namespace arithmetic_evaluation_l146_146876

theorem arithmetic_evaluation :
  (3.2 - 2.95) / (0.25 * 2 + 1/4) + (2 * 0.3) / (2.3 - (1 + 2/5)) = 1 := by
  sorry

end arithmetic_evaluation_l146_146876


namespace problem8x_eq_5_200timesreciprocal_l146_146287

theorem problem8x_eq_5_200timesreciprocal (x : ℚ) (h : 8 * x = 5) : 200 * (1 / x) = 320 := 
by 
  sorry

end problem8x_eq_5_200timesreciprocal_l146_146287


namespace largest_4_digit_divisible_by_12_l146_146509

theorem largest_4_digit_divisible_by_12 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ 12 ∣ n ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ 12 ∣ m → m ≤ n :=
sorry

end largest_4_digit_divisible_by_12_l146_146509


namespace solve_for_x_l146_146455

theorem solve_for_x (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 := 
sorry

end solve_for_x_l146_146455


namespace distinct_digit_numbers_count_l146_146145

def numDistinctDigitNumbers : Nat := 
  let first_digit_choices := 10
  let second_digit_choices := 9
  let third_digit_choices := 8
  let fourth_digit_choices := 7
  first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices

theorem distinct_digit_numbers_count : numDistinctDigitNumbers = 5040 :=
by
  sorry

end distinct_digit_numbers_count_l146_146145


namespace music_store_cellos_l146_146235

/-- 
A certain music store stocks 600 violas. 
There are 100 cello-viola pairs, such that a cello and a viola were both made with wood from the same tree. 
The probability that the two instruments are made with wood from the same tree is 0.00020833333333333335. 
Prove that the store stocks 800 cellos.
-/
theorem music_store_cellos (V : ℕ) (P : ℕ) (Pr : ℚ) (C : ℕ) 
  (h1 : V = 600) 
  (h2 : P = 100) 
  (h3 : Pr = 0.00020833333333333335) 
  (h4 : Pr = P / (C * V)): C = 800 :=
by
  sorry

end music_store_cellos_l146_146235


namespace obtuse_right_triangle_cannot_exist_l146_146395

-- Definitions of various types of triangles

def is_acute (θ : ℕ) : Prop := θ < 90
def is_right (θ : ℕ) : Prop := θ = 90
def is_obtuse (θ : ℕ) : Prop := θ > 90

def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ a = c
def is_scalene (a b c : ℕ) : Prop := ¬ (a = b) ∧ ¬ (b = c) ∧ ¬ (a = c)
def is_triangle (a b c : ℕ) : Prop := a + b + c = 180

-- Propositions for the types of triangles given in the problem

def acute_isosceles_triangle (a b : ℕ) : Prop :=
  is_triangle a a (180 - 2 * a) ∧ is_acute a ∧ is_isosceles a a (180 - 2 * a)

def isosceles_right_triangle (a : ℕ) : Prop :=
  is_triangle a a 90 ∧ is_right 90 ∧ is_isosceles a a 90

def obtuse_right_triangle (a b : ℕ) : Prop :=
  is_triangle a 90 (180 - 90 - a) ∧ is_right 90 ∧ is_obtuse (180 - 90 - a)

def scalene_right_triangle (a b : ℕ) : Prop :=
  is_triangle a b 90 ∧ is_right 90 ∧ is_scalene a b 90

def scalene_obtuse_triangle (a b : ℕ) : Prop :=
  is_triangle a b (180 - a - b) ∧ is_obtuse (180 - a - b) ∧ is_scalene a b (180 - a - b)

-- The final theorem stating that obtuse right triangle cannot exist

theorem obtuse_right_triangle_cannot_exist (a b : ℕ) :
  ¬ exists (a b : ℕ), obtuse_right_triangle a b :=
by
  sorry

end obtuse_right_triangle_cannot_exist_l146_146395


namespace water_flow_into_sea_per_minute_l146_146228

noncomputable def river_flow_rate_kmph : ℝ := 4
noncomputable def river_depth_m : ℝ := 5
noncomputable def river_width_m : ℝ := 19
noncomputable def hours_to_minutes : ℝ := 60
noncomputable def km_to_m : ℝ := 1000

noncomputable def flow_rate_m_per_min : ℝ := (river_flow_rate_kmph * km_to_m) / hours_to_minutes
noncomputable def cross_sectional_area_m2 : ℝ := river_depth_m * river_width_m
noncomputable def volume_per_minute_m3 : ℝ := cross_sectional_area_m2 * flow_rate_m_per_min

theorem water_flow_into_sea_per_minute :
  volume_per_minute_m3 = 6333.65 := by 
  -- Proof would go here
  sorry

end water_flow_into_sea_per_minute_l146_146228


namespace probability_of_black_yellow_green_probability_of_not_red_or_green_l146_146612

namespace ProbabilityProof

/- Definitions of events A, B, C, D representing probabilities as real numbers -/
variables (P_A P_B P_C P_D : ℝ)

/- Conditions stated in the problem -/
def conditions (h1 : P_A = 1 / 3)
               (h2 : P_B + P_C = 5 / 12)
               (h3 : P_C + P_D = 5 / 12)
               (h4 : P_A + P_B + P_C + P_D = 1) :=
  true

/- Proof that P(B) = 1/4, P(C) = 1/6, and P(D) = 1/4 given the conditions -/
theorem probability_of_black_yellow_green
  (P_A P_B P_C P_D : ℝ)
  (h1 : P_A = 1 / 3)
  (h2 : P_B + P_C = 5 / 12)
  (h3 : P_C + P_D = 5 / 12)
  (h4 : P_A + P_B + P_C + P_D = 1) :
  P_B = 1 / 4 ∧ P_C = 1 / 6 ∧ P_D = 1 / 4 :=
by
  sorry

/- Proof that the probability of not drawing a red or green ball is 5/12 -/
theorem probability_of_not_red_or_green
  (P_A P_B P_C P_D : ℝ)
  (h1 : P_A = 1 / 3)
  (h2 : P_B + P_C = 5 / 12)
  (h3 : P_C + P_D = 5 / 12)
  (h4 : P_A + P_B + P_C + P_D = 1)
  (h5 : P_B = 1 / 4)
  (h6 : P_C = 1 / 6)
  (h7 : P_D = 1 / 4) :
  1 - (P_A + P_D) = 5 / 12 :=
by
  sorry

end ProbabilityProof

end probability_of_black_yellow_green_probability_of_not_red_or_green_l146_146612


namespace find_a22_l146_146976

-- Definitions and conditions
noncomputable def seq (n : ℕ) : ℝ := if n = 0 then 0 else sorry

axiom seq_conditions
  (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) : True

theorem find_a22 (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 :=
sorry

end find_a22_l146_146976


namespace trapezoid_side_lengths_l146_146405

theorem trapezoid_side_lengths
  (isosceles : ∀ (A B C D : ℝ) (height BE : ℝ), height = 2 → BE = 2 → A = 2 * Real.sqrt 2 → D = A → 12 = 0.5 * (B + C) * BE → A = D)
  (area : ∀ (BC AD : ℝ), 12 = 0.5 * (BC + AD) * 2)
  (height : ∀ (BE : ℝ), BE = 2)
  (intersect_right_angle : ∀ (A B C D : ℝ), 90 = 45 + 45) :
  ∃ A B C D, A = 2 * Real.sqrt 2 ∧ B = 4 ∧ C = 8 ∧ D = 2 * Real.sqrt 2 :=
by
  sorry

end trapezoid_side_lengths_l146_146405


namespace non_degenerate_ellipse_l146_146372

theorem non_degenerate_ellipse (k : ℝ) : (∃ (x y : ℝ), x^2 + 4*y^2 - 10*x + 56*y = k) ↔ k > -221 :=
sorry

end non_degenerate_ellipse_l146_146372


namespace price_per_working_game_eq_six_l146_146473

-- Define the total number of video games
def total_games : Nat := 10

-- Define the number of non-working video games
def non_working_games : Nat := 8

-- Define the total income from selling working games
def total_earning : Nat := 12

-- Calculate the number of working video games
def working_games : Nat := total_games - non_working_games

-- Define the expected price per working game
def expected_price_per_game : Nat := 6

-- Theorem statement: Prove that the price per working game is $6
theorem price_per_working_game_eq_six :
  total_earning / working_games = expected_price_per_game :=
by sorry

end price_per_working_game_eq_six_l146_146473


namespace geometric_series_ratio_l146_146785

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l146_146785


namespace simplify_abs_expression_l146_146277

theorem simplify_abs_expression
  (a b : ℝ)
  (h1 : a < 0)
  (h2 : a * b < 0)
  : |a - b - 3| - |4 + b - a| = -1 := by
  sorry

end simplify_abs_expression_l146_146277


namespace cost_of_watermelon_and_grapes_l146_146634

variable (x y z f : ℕ)

theorem cost_of_watermelon_and_grapes (h1 : x + y + z + f = 45) 
                                    (h2 : f = 3 * x) 
                                    (h3 : z = x + y) :
    y + z = 9 := by
  sorry

end cost_of_watermelon_and_grapes_l146_146634


namespace total_money_l146_146421

theorem total_money (A B C : ℕ) (h1 : A + C = 400) (h2 : B + C = 750) (hC : C = 250) :
  A + B + C = 900 :=
sorry

end total_money_l146_146421


namespace john_annual_profit_l146_146930

-- Definitions of monthly incomes
def TenantA_income : ℕ := 350
def TenantB_income : ℕ := 400
def TenantC_income : ℕ := 450

-- Total monthly income
def total_monthly_income : ℕ := TenantA_income + TenantB_income + TenantC_income

-- Definitions of monthly expenses
def rent_expense : ℕ := 900
def utilities_expense : ℕ := 100
def maintenance_fee : ℕ := 50

-- Total monthly expenses
def total_monthly_expense : ℕ := rent_expense + utilities_expense + maintenance_fee

-- Monthly profit
def monthly_profit : ℕ := total_monthly_income - total_monthly_expense

-- Annual profit
def annual_profit : ℕ := monthly_profit * 12

theorem john_annual_profit :
  annual_profit = 1800 := by
  -- The proof is omitted, but the statement asserts that John makes an annual profit of $1800.
  sorry

end john_annual_profit_l146_146930


namespace min_value_of_expression_l146_146556

noncomputable def target_expression (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x)

theorem min_value_of_expression : (∀ x : ℝ, target_expression x ≥ -784) ∧ (∃ x : ℝ, target_expression x = -784) :=
by
  sorry

end min_value_of_expression_l146_146556


namespace original_price_of_shirt_l146_146701

variables (S C P : ℝ)

def shirt_condition := S = C / 3
def pants_condition := S = P / 2
def total_paid := 0.90 * S + 0.95 * C + P = 900

theorem original_price_of_shirt :
  shirt_condition S C →
  pants_condition S P →
  total_paid S C P →
  S = 900 / 5.75 :=
by
  sorry

end original_price_of_shirt_l146_146701


namespace num_three_digit_primes_end_with_3_l146_146454

theorem num_three_digit_primes_end_with_3 : 
  (Finset.filter (λ n : ℕ, n.digits.length = 3 ∧ n % 10 = 3 ∧ n.Prime) (Finset.range 1000)).card = 70 := by
sorry

end num_three_digit_primes_end_with_3_l146_146454


namespace last_digit_2_to_2010_l146_146954

theorem last_digit_2_to_2010 : (2 ^ 2010) % 10 = 4 := 
by
  -- proofs and lemmas go here
  sorry

end last_digit_2_to_2010_l146_146954


namespace trigonometric_identity_l146_146114

theorem trigonometric_identity :
  3 * Real.arcsin (Real.sqrt 3 / 2) - Real.arctan (-1) - Real.arccos 0 = (3 * Real.pi) / 4 := 
by
  sorry

end trigonometric_identity_l146_146114


namespace shaded_percentage_correct_l146_146515

def total_squares : ℕ := 6 * 6
def shaded_squares : ℕ := 18
def percentage_shaded (total shaded : ℕ) : ℕ := (shaded * 100) / total

theorem shaded_percentage_correct : percentage_shaded total_squares shaded_squares = 50 := by
  sorry

end shaded_percentage_correct_l146_146515


namespace paula_aunt_money_l146_146763

theorem paula_aunt_money
  (shirts_cost : ℕ := 2 * 11)
  (pants_cost : ℕ := 13)
  (money_left : ℕ := 74) : 
  shirts_cost + pants_cost + money_left = 109 :=
by
  sorry

end paula_aunt_money_l146_146763


namespace find_k_l146_146177

def f (a b c x : Int) : Int := a * x^2 + b * x + c

theorem find_k (a b c k : Int)
  (h₁ : f a b c 2 = 0)
  (h₂ : 100 < f a b c 7 ∧ f a b c 7 < 110)
  (h₃ : 120 < f a b c 8 ∧ f a b c 8 < 130)
  (h₄ : 6000 * k < f a b c 100 ∧ f a b c 100 < 6000 * (k + 1)) :
  k = 0 := 
sorry

end find_k_l146_146177


namespace largest_base_5_five_digit_number_in_decimal_l146_146073

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l146_146073


namespace distance_between_tangency_points_l146_146060

theorem distance_between_tangency_points
  (circle_radius : ℝ) (M_distance : ℝ) (A_distance : ℝ) 
  (h1 : circle_radius = 7)
  (h2 : M_distance = 25)
  (h3 : A_distance = 7) :
  ∃ AB : ℝ, AB = 48 :=
by
  -- Definitions and proofs will go here.
  sorry

end distance_between_tangency_points_l146_146060


namespace determine_a_for_quadratic_l146_146830

theorem determine_a_for_quadratic (a : ℝ) : 
  (∃ x : ℝ, 3 * x ^ (a - 1) - x = 5 ∧ a - 1 = 2) → a = 3 := 
sorry

end determine_a_for_quadratic_l146_146830


namespace ratio_of_x_to_y_l146_146261

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 3 / 5) : x / y = 16 / 15 :=
sorry

end ratio_of_x_to_y_l146_146261


namespace Karen_baked_50_cookies_l146_146933

def Karen_kept_cookies : ℕ := 10
def Karen_grandparents_cookies : ℕ := 8
def people_in_class : ℕ := 16
def cookies_per_person : ℕ := 2

theorem Karen_baked_50_cookies :
  Karen_kept_cookies + Karen_grandparents_cookies + (people_in_class * cookies_per_person) = 50 :=
by 
  sorry

end Karen_baked_50_cookies_l146_146933


namespace find_a22_l146_146973

variable (a : ℕ → ℝ)
variable (h : ∀ n, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
variable (h99 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
variable (h100 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
variable (h10 : a 10 = 10)

theorem find_a22 : a 22 = 10 := sorry

end find_a22_l146_146973


namespace college_students_count_l146_146733

theorem college_students_count (girls boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
(h_ratio : ratio_boys = 6) (h_ratio_girls : ratio_girls = 5)
(h_girls : girls = 200)
(h_boys : boys = ratio_boys * (girls / ratio_girls)) :
  boys + girls = 440 := by
  sorry

end college_students_count_l146_146733


namespace minimum_value_l146_146719

theorem minimum_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  (∃ (m : ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → m ≤ (y / x + 1 / y)) ∧
   m = 3 ∧ (∀ (x : ℝ), 0 < x → 0 < (1 - x) → (1 - x) + x = 1 → (y / x + 1 / y = m) ↔ x = 1 / 2)) :=
by
  sorry

end minimum_value_l146_146719


namespace max_consecutive_integers_sum_45_l146_146822

theorem max_consecutive_integers_sum_45 :
  ∃ N : ℕ, (∃ a : ℤ, 45 = N * a + (N * (N - 1)) / 2) ∧ N ∈ divisors 90 ∧ (∀ M : ℕ, (∃ b : ℤ, 45 = M * b + (M * (M - 1)) / 2) ∧ M ∈ divisors 90 → M ≤ N) :=
begin
  existsi 90,
  split,
  { existsi -44,
    -- Proof part omitted
    sorry },
  split,
  { -- Proof part omitted
    sorry },
  { intros M hM,
    -- Proof part omitted
    sorry }
end

end max_consecutive_integers_sum_45_l146_146822


namespace find_k_l146_146135

theorem find_k (x y k : ℤ) (h₁ : x = -1) (h₂ : y = 2) (h₃ : 2 * x + k * y = 6) :
  k = 4 :=
by
  sorry

end find_k_l146_146135


namespace bridge_length_l146_146105

noncomputable def train_length : ℝ := 250 -- in meters
noncomputable def train_speed_kmh : ℝ := 60 -- in km/hr
noncomputable def crossing_time : ℝ := 20 -- in seconds

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600 -- converting to m/s

noncomputable def total_distance_covered : ℝ := train_speed_ms * crossing_time -- distance covered in 20 seconds

theorem bridge_length : total_distance_covered - train_length = 83.4 :=
by
  -- The proof would go here
  sorry

end bridge_length_l146_146105


namespace num_individuals_eliminated_l146_146383

theorem num_individuals_eliminated (pop_size : ℕ) (sample_size : ℕ) :
  (pop_size % sample_size) = 2 :=
by
  -- Given conditions
  let pop_size := 1252
  let sample_size := 50
  -- Proof skipped
  sorry

end num_individuals_eliminated_l146_146383


namespace denominator_of_second_fraction_l146_146161

theorem denominator_of_second_fraction (y x : ℝ) (h_cond : y > 0) (h_eq : (y / 20) + (3 * y / x) = 0.35 * y) : x = 10 :=
sorry

end denominator_of_second_fraction_l146_146161


namespace incorrect_statement_l146_146247

theorem incorrect_statement :
  ¬ (∀ (l1 l2 l3 : ℝ → ℝ → Prop), 
      (∀ (x y : ℝ), l3 x y → l1 x y) ∧ 
      (∀ (x y : ℝ), l3 x y → l2 x y) → 
      (∀ (x y : ℝ), l1 x y → l2 x y)) :=
by sorry

end incorrect_statement_l146_146247


namespace dog_food_l146_146188

theorem dog_food (weights : List ℕ) (h_weights : weights = [20, 40, 10, 30, 50]) (h_ratio : ∀ w ∈ weights, 1 ≤ w / 10):
  (weights.sum / 10) = 15 := by
  sorry

end dog_food_l146_146188


namespace least_number_to_add_l146_146088

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) : n = 1100 → d = 23 → r = n % d → (r ≠ 0) → (d - r) = 4 :=
by
  intros h₀ h₁ h₂ h₃
  simp [h₀, h₁] at h₂
  sorry

end least_number_to_add_l146_146088


namespace total_time_to_complete_work_l146_146940

noncomputable def mahesh_work_rate (W : ℕ) := W / 40
noncomputable def mahesh_work_done_in_20_days (W : ℕ) := 20 * (mahesh_work_rate W)
noncomputable def remaining_work (W : ℕ) := W - (mahesh_work_done_in_20_days W)
noncomputable def rajesh_work_rate (W : ℕ) := (remaining_work W) / 30

theorem total_time_to_complete_work (W : ℕ) :
    (mahesh_work_rate W) + (rajesh_work_rate W) = W / 24 →
    (mahesh_work_done_in_20_days W) = W / 2 →
    (remaining_work W) = W / 2 →
    (rajesh_work_rate W) = W / 60 →
    20 + 30 = 50 :=
by 
  intros _ _ _ _
  sorry

end total_time_to_complete_work_l146_146940


namespace sum_of_prime_factors_of_143_l146_146995

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l146_146995


namespace sqrt_sum_inequality_l146_146192

theorem sqrt_sum_inequality (x y α : ℝ) (h : sqrt (1 + x) + sqrt (1 + y) = 2 * sqrt (1 + α)) : x + y ≥ 2 * α :=
sorry

end sqrt_sum_inequality_l146_146192


namespace slope_of_line_between_solutions_l146_146224

theorem slope_of_line_between_solutions (x1 y1 x2 y2 : ℝ) (h1 : 3 / x1 + 4 / y1 = 0) (h2 : 3 / x2 + 4 / y2 = 0) (h3 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -4 / 3 := 
sorry

end slope_of_line_between_solutions_l146_146224


namespace circle_areas_sum_l146_146988

theorem circle_areas_sum {r s t : ℝ}
  (h1 : r + s = 5)
  (h2 : s + t = 12)
  (h3 : r + t = 13) :
  (Real.pi * r ^ 2 + Real.pi * s ^ 2 + Real.pi * t ^ 2) = 81 * Real.pi :=
by sorry

end circle_areas_sum_l146_146988


namespace john_total_spent_l146_146622

-- Defining the conditions from part a)
def vacuum_cleaner_original_price : ℝ := 250
def vacuum_cleaner_discount_rate : ℝ := 0.20
def dishwasher_price : ℝ := 450
def special_offer_discount : ℝ := 75
def sales_tax_rate : ℝ := 0.07

-- The adesso to formalize part c noncomputably.
noncomputable def total_amount_spent : ℝ :=
  let vacuum_cleaner_discount := vacuum_cleaner_original_price * vacuum_cleaner_discount_rate
  let vacuum_cleaner_final_price := vacuum_cleaner_original_price - vacuum_cleaner_discount
  let total_before_special_offer := vacuum_cleaner_final_price + dishwasher_price
  let total_after_special_offer := total_before_special_offer - special_offer_discount
  let sales_tax := total_after_special_offer * sales_tax_rate
  total_after_special_offer + sales_tax

-- The proof statement
theorem john_total_spent : total_amount_spent = 615.25 := by
  sorry

end john_total_spent_l146_146622


namespace tiling_problem_l146_146149

theorem tiling_problem (b c f : ℕ) (h : b * c = f) : c * (b^2 / f) = b :=
by 
  sorry

end tiling_problem_l146_146149


namespace pauls_total_cost_is_252_l146_146762

variable (price_shirt : ℕ) (num_shirts : ℕ)
variable (price_pants : ℕ) (num_pants : ℕ)
variable (price_suit : ℕ) (num_suit : ℕ)
variable (price_sweater : ℕ) (num_sweaters : ℕ)
variable (store_discount : ℕ) (coupon_discount : ℕ)

-- Define the given prices and discounts
def total_cost_before_discounts : ℕ :=
  (price_shirt * num_shirts) +
  (price_pants * num_pants) +
  (price_suit * num_suit) +
  (price_sweater * num_sweaters)

def store_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def coupon_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def total_cost_after_discounts : ℕ :=
  let initial_total := total_cost_before_discounts price_shirt num_shirts price_pants num_pants price_suit num_suit price_sweater num_sweaters
  let store_discount_value := store_discount_amount initial_total store_discount
  let subtotal_after_store_discount := initial_total - store_discount_value
  let coupon_discount_value := coupon_discount_amount subtotal_after_store_discount coupon_discount
  subtotal_after_store_discount - coupon_discount_value

theorem pauls_total_cost_is_252 :
  total_cost_after_discounts 15 4 40 2 150 1 30 2 20 10 = 252 := by
  sorry

end pauls_total_cost_is_252_l146_146762


namespace equation_pattern_l146_146483

theorem equation_pattern (n : ℕ) (h : n = 999999) : n^2 = (n + 1) * (n - 1) + 1 :=
by
  sorry

end equation_pattern_l146_146483


namespace interview_room_count_l146_146216

-- Define the number of people in the waiting room
def people_in_waiting_room : ℕ := 22

-- Define the increase in number of people
def extra_people_arrive : ℕ := 3

-- Define the total number of people after more people arrive
def total_people_after_arrival : ℕ := people_in_waiting_room + extra_people_arrive

-- Define the relationship between people in waiting room and interview room
def relation (x : ℕ) : Prop := total_people_after_arrival = 5 * x

theorem interview_room_count : ∃ x : ℕ, relation x ∧ x = 5 :=
by
  -- The proof will be provided here
  sorry

end interview_room_count_l146_146216


namespace proof_custom_operations_l146_146199

def customOp1 (a b : ℕ) : ℕ := a * b / (a + b)
def customOp2 (a b : ℕ) : ℕ := a * a + b * b

theorem proof_custom_operations :
  customOp2 (customOp1 7 14) 2 = 200 := 
by 
  sorry

end proof_custom_operations_l146_146199


namespace unique_solution_l146_146259

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  (2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1)) ∧
  (2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1)) ∧
  (2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1))

theorem unique_solution : ∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) → system_of_equations x y z → (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  intro x y z hx hy hz h
  sorry

end unique_solution_l146_146259


namespace markup_percentage_l146_146841

-- Define the wholesale cost
def wholesale_cost : ℝ := sorry

-- Define the retail cost
def retail_cost : ℝ := sorry

-- Condition given in the problem: selling at 60% discount nets a 20% profit
def discount_condition (W R : ℝ) : Prop :=
  0.40 * R = 1.20 * W

-- We need to prove the markup percentage is 200%
theorem markup_percentage (W R : ℝ) (h : discount_condition W R) : 
  ((R - W) / W) * 100 = 200 :=
by sorry

end markup_percentage_l146_146841


namespace evaluate_expression_l146_146175

variable (a b : ℝ) (h : a > b ∧ b > 0)

theorem evaluate_expression (h : a > b ∧ b > 0) : 
  (a^2 * b^3) / (b^2 * a^3) = (a / b)^(2 - 3) :=
  sorry

end evaluate_expression_l146_146175


namespace units_digit_difference_l146_146090

-- Conditions based on the problem statement
def units_digit_of_power_of_5 (n : ℕ) : ℕ := 5

def units_digit_of_power_of_3 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0     => 1
  | 1     => 3
  | 2     => 9
  | 3     => 7
  | _     => 0  -- impossible due to mod 4

-- Problem statement in Lean as a theorem
theorem units_digit_difference : (5^2019 - 3^2019) % 10 = 8 :=
by
  have h1 : (5^2019 % 10) = units_digit_of_power_of_5 2019 := sorry
  have h2 : (3^2019 % 10) = units_digit_of_power_of_3 2019 := sorry
  -- The core proof step will go here
  sorry

end units_digit_difference_l146_146090


namespace Alice_min_speed_l146_146368

theorem Alice_min_speed (d : ℝ) (v_bob : ℝ) (delta_t : ℝ) (v_alice : ℝ) :
  d = 180 ∧ v_bob = 40 ∧ delta_t = 0.5 ∧ 0 < v_alice ∧ v_alice * (d / v_bob - delta_t) ≥ d →
  v_alice > 45 :=
by
  sorry

end Alice_min_speed_l146_146368


namespace girls_together_girls_separated_girls_not_both_ends_girls_not_both_ends_simul_l146_146410

-- Definition of the primary condition
def girls := 3
def boys := 5

-- Statement for each part of the problem
theorem girls_together (A : ℕ → ℕ → ℕ) : 
  A (girls + boys - 1) girls * A girls girls = 4320 := 
sorry

theorem girls_separated (A : ℕ → ℕ → ℕ) : 
  A boys boys * A (girls + boys - 1) girls = 14400 := 
sorry

theorem girls_not_both_ends (A : ℕ → ℕ → ℕ) : 
  A boys 2 * A (girls + boys - 2) (girls + boys - 2) = 14400 := 
sorry

theorem girls_not_both_ends_simul (P : ℕ → ℕ → ℕ) (A : ℕ → ℕ → ℕ) : 
  P (girls + boys) (girls + boys) - A girls 2 * A (girls + boys - 2) (girls + boys - 2) = 36000 := 
sorry

end girls_together_girls_separated_girls_not_both_ends_girls_not_both_ends_simul_l146_146410


namespace Isabella_speed_is_correct_l146_146468

-- Definitions based on conditions
def distance_km : ℝ := 17.138
def time_s : ℝ := 38

-- Conversion factor
def conversion_factor : ℝ := 1000

-- Distance in meters
def distance_m : ℝ := distance_km * conversion_factor

-- Correct answer (speed in m/s)
def correct_speed : ℝ := 451

-- Statement to prove
theorem Isabella_speed_is_correct : distance_m / time_s = correct_speed :=
by
  sorry

end Isabella_speed_is_correct_l146_146468


namespace friends_playing_video_game_l146_146857

def total_lives : ℕ := 64
def lives_per_player : ℕ := 8

theorem friends_playing_video_game (num_friends : ℕ) :
  num_friends = total_lives / lives_per_player :=
sorry

end friends_playing_video_game_l146_146857


namespace product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers_l146_146958

theorem product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers (n : ℤ) :
  let T := (n - 1) * n * (n + 1) * (n + 2)
  let M := n * (n + 1)
  T = (M - 2) * M :=
by
  -- proof here
  sorry

end product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers_l146_146958


namespace vincent_total_packs_l146_146398

noncomputable def total_packs (yesterday today_addition: ℕ) : ℕ :=
  let today := yesterday + today_addition
  yesterday + today

theorem vincent_total_packs
  (yesterday_packs : ℕ)
  (today_addition: ℕ)
  (hyesterday: yesterday_packs = 15)
  (htoday_addition: today_addition = 10) :
  total_packs yesterday_packs today_addition = 40 :=
by
  rw [hyesterday, htoday_addition]
  unfold total_packs
  -- at this point it simplifies to 15 + (15 + 10) = 40
  sorry

end vincent_total_packs_l146_146398


namespace mary_saw_total_snakes_l146_146943

theorem mary_saw_total_snakes :
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  totalSnakes = 36 :=
by
  /- Definitions -/ 
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  /- Main proof statement -/
  show totalSnakes = 36
  sorry

end mary_saw_total_snakes_l146_146943


namespace fermat_little_theorem_l146_146675

theorem fermat_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) : a ^ p ≡ a [MOD p] :=
sorry

end fermat_little_theorem_l146_146675


namespace Heesu_has_greatest_sum_l146_146358

-- Define the numbers collected by each individual
def Sora_collected : (Nat × Nat) := (4, 6)
def Heesu_collected : (Nat × Nat) := (7, 5)
def Jiyeon_collected : (Nat × Nat) := (3, 8)

-- Calculate the sums
def Sora_sum : Nat := Sora_collected.1 + Sora_collected.2
def Heesu_sum : Nat := Heesu_collected.1 + Heesu_collected.2
def Jiyeon_sum : Nat := Jiyeon_collected.1 + Jiyeon_collected.2

-- The theorem to prove that Heesu has the greatest sum
theorem Heesu_has_greatest_sum :
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  sorry

end Heesu_has_greatest_sum_l146_146358


namespace largest_regular_hexagon_proof_l146_146099

noncomputable def largest_regular_hexagon_side_length (x : ℝ) (H : ConvexHexagon) 
  (hx : -5 < x ∧ x < 6) : ℝ := 11 / 2

-- Convex Hexagon Definition
structure ConvexHexagon :=
  (sides : Vector ℝ 6)
  (is_convex : true)  -- Placeholder for convex property

theorem largest_regular_hexagon_proof (x : ℝ) (H : ConvexHexagon) 
  (hx : -5 < x ∧ x < 6)
  (H_sides_length : H.sides = ⟨[5, 6, 7, 5+x, 6-x, 7+x], by simp⟩) :
  largest_regular_hexagon_side_length x H hx = 11 / 2 :=
sorry

end largest_regular_hexagon_proof_l146_146099


namespace geometric_series_common_ratio_l146_146798

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l146_146798


namespace number_of_students_l146_146663

variables (m d r : ℕ) (k : ℕ)

theorem number_of_students :
  (30 < m + d ∧ m + d < 40) → (r = 3 * m) → (r = 5 * d) → m + d = 32 :=
by 
  -- The proof body is not necessary here according to instructions.
  sorry

end number_of_students_l146_146663


namespace quadratic_form_h_l146_146591

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l146_146591


namespace parallelogram_circumference_l146_146650

-- Define the lengths of the sides of the parallelogram.
def side1 : ℝ := 18
def side2 : ℝ := 12

-- Define the formula for the circumference (or perimeter) of the parallelogram.
def circumference (a b : ℝ) : ℝ :=
  2 * (a + b)

-- Statement of the proof problem:
theorem parallelogram_circumference : circumference side1 side2 = 60 := 
  by
    sorry

end parallelogram_circumference_l146_146650


namespace hyperbola_asymptote_eccentricity_l146_146046

-- Problem statement: We need to prove that the eccentricity of hyperbola 
-- given the specific asymptote is sqrt(5).

noncomputable def calc_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_asymptote_eccentricity 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_asymptote : b = 2 * a) :
  calc_eccentricity a b = Real.sqrt 5 := 
by
  -- Insert the proof step here
  sorry

end hyperbola_asymptote_eccentricity_l146_146046


namespace parabola_equation_l146_146706

theorem parabola_equation :
  (∃ h k : ℝ, h^2 = 3 ∧ k^2 = 6) →
  (∃ c : ℝ, c^2 = (3 + 6)) →
  (∃ x y : ℝ, x = 3 ∧ y = 0) →
  (y^2 = 12 * x) :=
sorry

end parabola_equation_l146_146706


namespace probability_distance_greater_than_2_l146_146573

theorem probability_distance_greater_than_2 :
  let D := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  let area_square := 9
  let area_sector := Real.pi
  let area_shaded := area_square - area_sector
  let P := area_shaded / area_square
  P = (9 - Real.pi) / 9 :=
by
  sorry

end probability_distance_greater_than_2_l146_146573


namespace present_value_of_machine_l146_146858

theorem present_value_of_machine (r : ℝ) (t : ℕ) (V : ℝ) (P : ℝ) (h1 : r = 0.10) (h2 : t = 2) (h3 : V = 891) :
  V = P * (1 - r)^t → P = 1100 :=
by
  intro h
  rw [h3, h1, h2] at h
  -- The steps to solve for P are omitted as instructed
  sorry

end present_value_of_machine_l146_146858


namespace no_nonconstant_arithmetic_progression_l146_146559

theorem no_nonconstant_arithmetic_progression (x : ℝ) :
  2 * (2 : ℝ)^(x^2) ≠ (2 : ℝ)^x + (2 : ℝ)^(x^3) :=
sorry

end no_nonconstant_arithmetic_progression_l146_146559


namespace diego_annual_savings_l146_146263

-- Definitions based on conditions
def monthly_deposit := 5000
def monthly_expense := 4600
def months_in_year := 12

-- Prove that Diego's annual savings is $4800
theorem diego_annual_savings : (monthly_deposit - monthly_expense) * months_in_year = 4800 := by
  sorry

end diego_annual_savings_l146_146263


namespace water_consumption_l146_146868

theorem water_consumption (x y : ℝ)
  (h1 : 120 + 20 * x = 3200000 * y)
  (h2 : 120 + 15 * x = 3000000 * y) :
  x = 200 ∧ y = 50 :=
by
  sorry

end water_consumption_l146_146868


namespace isosceles_triangle_largest_angle_l146_146738

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h1 : A = B) (h2 : C = 50) (h3 : A + B + C = 180) : max A C = 80 :=
by 
  -- Define additional facts about the triangle, e.g., A = B = 50, and sum of angles = 180.
  have h4 : A = 50 := sorry
  rw [h4, h2] at h3
  -- Prove the final result using the given conditions.
  sorry

end isosceles_triangle_largest_angle_l146_146738


namespace simplify_expression_l146_146641

variables (a b : ℝ)

theorem simplify_expression (h₁ : a = 2) (h₂ : b = -1) :
  (2 * a^2 - a * b - b^2) - 2 * (a^2 - 2 * a * b + b^2) = -5 :=
by
  sorry

end simplify_expression_l146_146641


namespace compare_y1_y2_l146_146655

theorem compare_y1_y2 (a : ℝ) (y1 y2 : ℝ) (h₁ : a < 0) (h₂ : y1 = a * (-1 - 1)^2 + 3) (h₃ : y2 = a * (2 - 1)^2 + 3) : 
  y1 < y2 :=
by
  sorry

end compare_y1_y2_l146_146655


namespace Lincoln_High_School_max_principals_l146_146344

def max_principals (total_years : ℕ) (term_length : ℕ) (max_principals_count : ℕ) : Prop :=
  ∀ (period : ℕ), period = total_years → 
                  term_length = 4 → 
                  max_principals_count = 3

theorem Lincoln_High_School_max_principals 
  (total_years term_length max_principals_count : ℕ) :
  max_principals total_years term_length max_principals_count :=
by 
  intros period h1 h2
  have h3 : period = 10 := sorry
  have h4 : term_length = 4 := sorry
  have h5 : max_principals_count = 3 := sorry
  sorry

end Lincoln_High_School_max_principals_l146_146344


namespace good_or_bad_of_prime_divides_l146_146888

-- Define the conditions
variables (k n n' : ℕ)
variables (h1 : k ≥ 2) (h2 : n ≥ k) (h3 : n' ≥ k)
variables (prime_divides : ∀ p, prime p → p ≤ k → (p ∣ n ↔ p ∣ n'))

-- Define what it means for a number to be good or bad
def is_good (m : ℕ) : Prop := ∃ strategy : ℕ → Prop, strategy m

-- Prove that either both n and n' are good or both are bad
theorem good_or_bad_of_prime_divides :
  (is_good n ∧ is_good n') ∨ (¬is_good n ∧ ¬is_good n') :=
sorry

end good_or_bad_of_prime_divides_l146_146888


namespace sum_of_distances_minimized_l146_146735

theorem sum_of_distances_minimized (x : ℝ) (h : 0 ≤ x ∧ x ≤ 50) : 
  abs (x - 0) + abs (x - 50) = 50 := 
by
  sorry

end sum_of_distances_minimized_l146_146735


namespace transformation_correct_l146_146770

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

-- Define the transformation functions
noncomputable def shift_right_by_pi_over_10 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - Real.pi / 10)
noncomputable def stretch_x_by_factor_of_2 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x / 2)

-- Define the transformed function
noncomputable def transformed_function : ℝ → ℝ :=
  stretch_x_by_factor_of_2 (shift_right_by_pi_over_10 original_function)

-- Define the expected resulting function
noncomputable def expected_function (x : ℝ) : ℝ := Real.sin (x / 2 - Real.pi / 10)

-- State the theorem
theorem transformation_correct :
  ∀ x : ℝ, transformed_function x = expected_function x :=
by
  sorry

end transformation_correct_l146_146770


namespace abs_less_than_zero_impossible_l146_146422

theorem abs_less_than_zero_impossible (x : ℝ) : |x| < 0 → false :=
by
  sorry

end abs_less_than_zero_impossible_l146_146422


namespace person_speed_kmh_l146_146861

-- Given conditions
def distance_meters : ℝ := 1000
def time_minutes : ℝ := 10

-- Proving the speed in km/h
theorem person_speed_kmh :
  (distance_meters / 1000) / (time_minutes / 60) = 6 :=
  sorry

end person_speed_kmh_l146_146861


namespace range_of_a_l146_146448

def f (a x : ℝ) : ℝ := -x^3 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → -3 * x^2 + a ≥ 0) → a ≥ 3 := 
by
  sorry

end range_of_a_l146_146448
