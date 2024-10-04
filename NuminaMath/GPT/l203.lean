import Mathlib

namespace length_of_each_train_l203_203023

theorem length_of_each_train (L : ℝ) (s1 : ℝ) (s2 : ℝ) (t : ℝ)
    (h1 : s1 = 46) (h2 : s2 = 36) (h3 : t = 144) (h4 : 2 * L = ((s1 - s2) * (5 / 18)) * t) :
    L = 200 := 
sorry

end length_of_each_train_l203_203023


namespace product_of_two_numbers_l203_203309

theorem product_of_two_numbers : 
  ∀ (x y : ℝ), (x + y = 60) ∧ (x - y = 10) → x * y = 875 :=
by
  intros x y h
  sorry

end product_of_two_numbers_l203_203309


namespace workbook_problems_l203_203440

theorem workbook_problems (P : ℕ)
  (h1 : (1/2 : ℚ) * P = (1/2 : ℚ) * P)
  (h2 : (1/4 : ℚ) * P = (1/4 : ℚ) * P)
  (h3 : (1/6 : ℚ) * P = (1/6 : ℚ) * P)
  (h4 : ((1/2 : ℚ) * P + (1/4 : ℚ) * P + (1/6 : ℚ) * P + 20 = P)) : 
  P = 240 :=
sorry

end workbook_problems_l203_203440


namespace none_of_these_l203_203810

noncomputable def x (t : ℝ) : ℝ := t ^ (3 / (t - 1))
noncomputable def y (t : ℝ) : ℝ := t ^ ((t + 1) / (t - 1))

theorem none_of_these (t : ℝ) (ht_pos : t > 0) (ht_ne_one : t ≠ 1) :
  ¬ (y t ^ x t = x t ^ y t) ∧ ¬ (x t ^ x t = y t ^ y t) ∧
  ¬ (x t ^ (y t ^ x t) = y t ^ (x t ^ y t)) ∧ ¬ (x t ^ y t = y t ^ x t) :=
sorry

end none_of_these_l203_203810


namespace four_digit_numbers_count_l203_203164

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l203_203164


namespace standard_equation_of_ellipse_l203_203648

theorem standard_equation_of_ellipse :
  ∀ (m n : ℝ), 
    (m > 0 ∧ n > 0) →
    (∃ (c : ℝ), c^2 = m^2 - n^2 ∧ c = 2) →
    (∃ (e : ℝ), e = c / m ∧ e = 1 / 2) →
    (m = 4 ∧ n = 2 * Real.sqrt 3) →
    (∀ x y : ℝ, (x^2 / 16 + y^2 / 12 = 1)) :=
by
  intros m n hmn hc he hm_eq hn_eq
  sorry

end standard_equation_of_ellipse_l203_203648


namespace part1_solution_set_part2_inequality_l203_203928

noncomputable def f (x : ℝ) : ℝ := 
  x * Real.exp (x + 1)

theorem part1_solution_set (h : 0 < x) : 
  f x < 3 * Real.log 3 - 3 ↔ 0 < x ∧ x < Real.log 3 - 1 :=
sorry

theorem part2_inequality (h1 : f x1 = 3 * Real.exp x1 + 3 * Real.exp (Real.log x1)) 
    (h2 : f x2 = 3 * Real.exp x2 + 3 * Real.exp (Real.log x2)) (h_distinct : x1 ≠ x2) :
  x1 + x2 + Real.log (x1 * x2) > 2 :=
sorry

end part1_solution_set_part2_inequality_l203_203928


namespace candidate_C_is_inverse_proportion_l203_203712

/--
Check whether the given function is an inverse proportion function.
-/
def is_inverse_proportion (f : ℝ → ℝ) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/--
The candidate functions are defined as follows:
A: y = x / 3
B: y = 3 / (x + 1)
C: xy = 3
D: y = 3x
-/
def candidate_A (x : ℝ) : ℝ := x / 3
def candidate_B (x : ℝ) : ℝ := 3 / (x + 1)
def candidate_C (x : ℝ) : ℝ := 3 / x
def candidate_D (x : ℝ) : ℝ := 3 * x

theorem candidate_C_is_inverse_proportion : is_inverse_proportion candidate_C :=
  sorry

end candidate_C_is_inverse_proportion_l203_203712


namespace exp_gt_f_n_y_between_0_and_x_l203_203077

open Real

noncomputable def f_n (x : ℝ) (n : ℕ) : ℝ :=
  (Finset.range (n + 1)).sum (λ k => x^k / k.factorial)

theorem exp_gt_f_n (x : ℝ) (n : ℕ) (h1 : 0 < x) :
  exp x > f_n x n :=
sorry

theorem y_between_0_and_x (x : ℝ) (n : ℕ) (y : ℝ)
  (h1 : 0 < x)
  (h2 : exp x = f_n x n + x^(n+1) / (n + 1).factorial * exp y) :
  0 < y ∧ y < x :=
sorry

end exp_gt_f_n_y_between_0_and_x_l203_203077


namespace hall_length_width_difference_l203_203583

variable (L W : ℕ)

theorem hall_length_width_difference (h₁ : W = 1 / 2 * L) (h₂ : L * W = 800) :
  L - W = 20 :=
sorry

end hall_length_width_difference_l203_203583


namespace find_a11_l203_203329

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a11 (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 1 * a 4 = 20)
  (h3 : a 0 + a 5 = 9) :
  a 10 = 25 / 4 :=
sorry

end find_a11_l203_203329


namespace max_sum_m_n_l203_203076

noncomputable def ellipse_and_hyperbola_max_sum : Prop :=
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ (∃ x y : ℝ, (x^2 / 25 + y^2 / m^2 = 1 ∧ x^2 / 7 - y^2 / n^2 = 1)) ∧
  (25 - m^2 = 7 + n^2) ∧ (m + n = 6)

theorem max_sum_m_n : ellipse_and_hyperbola_max_sum :=
  sorry

end max_sum_m_n_l203_203076


namespace minimum_seedlings_needed_l203_203226

theorem minimum_seedlings_needed (n : ℕ) (h1 : 75 ≤ n) (h2 : n ≤ 80) (H : 1200 * 100 / n = 1500) : n = 80 :=
sorry

end minimum_seedlings_needed_l203_203226


namespace find_scalars_l203_203547

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 2;
    3, 1]

noncomputable def B4 : Matrix (Fin 2) (Fin 2) ℝ :=
  B * B * B * B

noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem find_scalars (r s : ℝ) (hB : B^4 = r • B + s • I) :
  (r, s) = (51, 52) :=
  sorry

end find_scalars_l203_203547


namespace average_income_of_other_40_customers_l203_203481

theorem average_income_of_other_40_customers
    (avg_income_50 : ℝ)
    (num_50 : ℕ)
    (avg_income_10 : ℝ)
    (num_10 : ℕ)
    (total_num : ℕ)
    (remaining_num : ℕ)
    (total_income_50 : ℝ)
    (total_income_10 : ℝ)
    (total_income_40 : ℝ)
    (avg_income_40 : ℝ) 
    (hyp_avg_income_50 : avg_income_50 = 45000)
    (hyp_num_50 : num_50 = 50)
    (hyp_avg_income_10 : avg_income_10 = 55000)
    (hyp_num_10 : num_10 = 10)
    (hyp_total_num : total_num = 50)
    (hyp_remaining_num : remaining_num = 40)
    (hyp_total_income_50 : total_income_50 = 2250000)
    (hyp_total_income_10 : total_income_10 = 550000)
    (hyp_total_income_40 : total_income_40 = 1700000)
    (hyp_avg_income_40 : avg_income_40 = total_income_40 / remaining_num) :
  avg_income_40 = 42500 :=
  by
    sorry

end average_income_of_other_40_customers_l203_203481


namespace part_I_part_II_l203_203492

def S (n : ℕ) : ℕ := 2 ^ n - 1

def a (n : ℕ) : ℕ := 2 ^ (n - 1)

def T (n : ℕ) : ℕ := (n - 1) * 2 ^ n + 1

theorem part_I (S : ℕ → ℕ) (hS : ∀ n, S n = 2^n - 1) :
  ∀ n, ∃ a : ℕ → ℕ, a n = 2^(n-1) :=
by
  sorry

theorem part_II (a : ℕ → ℕ) (ha : ∀ n, a n = 2^(n-1)) :
  ∀ n, ∃ T : ℕ → ℕ, T n = (n - 1) * 2 ^ n + 1 :=
by
  sorry

end part_I_part_II_l203_203492


namespace ramesh_paid_price_l203_203686

theorem ramesh_paid_price {P : ℝ} (h1 : P = 18880 / 1.18) : 
  (0.80 * P + 125 + 250) = 13175 :=
by sorry

end ramesh_paid_price_l203_203686


namespace find_n_l203_203972

noncomputable def objects_per_hour (n : ℕ) : ℕ := n

theorem find_n (n : ℕ) (h₁ : 1 + (2 / 3) + (1 / 3) + (1 / 3) = 7 / 3) 
  (h₂ : objects_per_hour n * 7 / 3 = 28) : n = 12 :=
by
  have total_hours := h₁ 
  have total_objects := h₂
  sorry

end find_n_l203_203972


namespace jonathan_daily_calories_l203_203796

theorem jonathan_daily_calories (C : ℕ) (daily_burn weekly_deficit extra_calories total_burn : ℕ) 
  (h1 : daily_burn = 3000) 
  (h2 : weekly_deficit = 2500) 
  (h3 : extra_calories = 1000) 
  (h4 : total_burn = 7 * daily_burn) 
  (h5 : total_burn - weekly_deficit = 7 * C + extra_calories) :
  C = 2500 :=
by 
  sorry

end jonathan_daily_calories_l203_203796


namespace moles_of_C2H5Cl_l203_203909

-- Define chemical entities as types
structure Molecule where
  name : String

-- Declare molecules involved in the reaction
def C2H6 := Molecule.mk "C2H6"
def Cl2  := Molecule.mk "Cl2"
def C2H5Cl := Molecule.mk "C2H5Cl"
def HCl := Molecule.mk "HCl"

-- Define number of moles as a non-negative integer
def moles (m : Molecule) : ℕ := sorry

-- Conditions
axiom initial_moles_C2H6 : moles C2H6 = 3
axiom initial_moles_Cl2 : moles Cl2 = 3

-- Balanced reaction equation: 1 mole of C2H6 reacts with 1 mole of Cl2 to form 1 mole of C2H5Cl
axiom reaction_stoichiometry : ∀ (x : ℕ), moles C2H6 = x → moles Cl2 = x → moles C2H5Cl = x

-- Proof problem
theorem moles_of_C2H5Cl : moles C2H5Cl = 3 := by
  apply reaction_stoichiometry
  exact initial_moles_C2H6
  exact initial_moles_Cl2

end moles_of_C2H5Cl_l203_203909


namespace number_of_trees_l203_203585

theorem number_of_trees (initial_trees planted_trees : ℕ)
  (h1 : initial_trees = 13)
  (h2 : planted_trees = 12) :
  initial_trees + planted_trees = 25 := by
  sorry

end number_of_trees_l203_203585


namespace birds_remaining_on_fence_l203_203002

noncomputable def initial_birds : ℝ := 15.3
noncomputable def birds_flew_away : ℝ := 6.5
noncomputable def remaining_birds : ℝ := initial_birds - birds_flew_away

theorem birds_remaining_on_fence : remaining_birds = 8.8 :=
by
  -- sorry is a placeholder for the proof, which is not required
  sorry

end birds_remaining_on_fence_l203_203002


namespace area_of_triangle_hyperbola_focus_l203_203515

theorem area_of_triangle_hyperbola_focus :
  let F₁ := (-Real.sqrt 2, 0)
  let F₂ := (Real.sqrt 2, 0)
  let hyperbola := {p : ℝ × ℝ | p.1 ^ 2 - p.2 ^ 2 = 1}
  let asymptote (p : ℝ × ℝ) := p.1 = p.2
  let circle := {p : ℝ × ℝ | (p.1 - F₁.1 / 2) ^ 2 + (p.2 - F₁.2 / 2) ^ 2 = (Real.sqrt 2) ^ 2}
  let P := (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  let Q := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let area (p1 p2 p3 : ℝ × ℝ) := 0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))
  area F₁ P Q = Real.sqrt 2 := 
sorry

end area_of_triangle_hyperbola_focus_l203_203515


namespace largest_five_digit_congruent_to_18_mod_25_l203_203857

theorem largest_five_digit_congruent_to_18_mod_25 : 
  ∃ (x : ℕ), x < 100000 ∧ 10000 ≤ x ∧ x % 25 = 18 ∧ x = 99993 :=
by
  sorry

end largest_five_digit_congruent_to_18_mod_25_l203_203857


namespace perpendicular_slope_l203_203912

-- Define the given line equation
def line_eq (x y : ℝ) : Prop := 5 * x - 2 * y = 10

-- Define the slope of a line
def slope (m : ℝ) : Prop := ∀ x y b : ℝ, y = m * x + b

-- Define the condition for negative reciprocal
def perp_slope (m m_perpendicular : ℝ) : Prop := 
  m_perpendicular = - (1 / m)

-- The main statement to be proven
theorem perpendicular_slope : 
  ∃ m_perpendicular : ℝ, 
  (∃ m : ℝ, slope m ∧ (∀ x y : ℝ, line_eq x y → m = 5 / 2)) 
  → perp_slope (5 / 2) m_perpendicular ∧ m_perpendicular = - (2 / 5) := 
by
  sorry

end perpendicular_slope_l203_203912


namespace matrix_pow_three_l203_203618

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_pow_three :
  A^3 = !![-4, 2; -2, 1] := by
  sorry

end matrix_pow_three_l203_203618


namespace division_result_l203_203302

theorem division_result : 210 / (15 + 12 * 3 - 6) = 210 / 45 :=
by
  sorry

end division_result_l203_203302


namespace cost_price_represents_articles_l203_203177

theorem cost_price_represents_articles (C S : ℝ) (N : ℕ)
  (h1 : N * C = 16 * S)
  (h2 : S = C * 1.125) :
  N = 18 :=
by
  sorry

end cost_price_represents_articles_l203_203177


namespace probability_x_plus_y_lt_4_l203_203471

theorem probability_x_plus_y_lt_4 :
  let square_vertices := {(0,0), (0,3), (3,3), (3,0)} in
  let point_in_square (x y : ℝ) := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 in
  let probability (A : ℝ) (B : ℝ) := A / B in
  ∃ (P : ℝ), P = probability 7 9 ∧
             P = (measure (λ (x y : ℝ), point_in_square x y ∧ x + y < 4)) / (measure (λ (x y : ℝ), point_in_square x y)) :=
sorry

end probability_x_plus_y_lt_4_l203_203471


namespace ratio_SP2_SP1_l203_203741

variable (CP : ℝ)

-- First condition: Sold at a profit of 140%
def SP1 := 2.4 * CP

-- Second condition: Sold at a loss of 20%
def SP2 := 0.8 * CP

-- Statement: The ratio of SP2 to SP1 is 1 to 3
theorem ratio_SP2_SP1 : SP2 / SP1 = 1 / 3 :=
by
  sorry

end ratio_SP2_SP1_l203_203741


namespace probability_composite_is_correct_l203_203343

noncomputable def probability_composite : ℚ :=
  1 - (25 / (8^6))

theorem probability_composite_is_correct :
  probability_composite = 262119 / 262144 :=
by
  sorry

end probability_composite_is_correct_l203_203343


namespace expenses_of_five_yuan_l203_203997

theorem expenses_of_five_yuan (income_denotation : ℤ) (opposite_effect : ∀ x : ℤ, -x) :
  income_denotation = 5 → opposite_effect income_denotation = -5 :=
by sorry

end expenses_of_five_yuan_l203_203997


namespace sqrt_quartic_equiv_l203_203611

-- Define x as a positive real number
variable (x : ℝ)
variable (hx : 0 < x)

-- Statement of the problem to prove
theorem sqrt_quartic_equiv (x : ℝ) (hx : 0 < x) : (x^2 * x^(1/2))^(1/4) = x^(5/8) :=
sorry

end sqrt_quartic_equiv_l203_203611


namespace closest_point_on_line_y_eq_3x_plus_2_l203_203308

theorem closest_point_on_line_y_eq_3x_plus_2 (x y : ℝ) :
  ∃ (p : ℝ × ℝ), p = (-1 / 2, 1 / 2) ∧ y = 3 * x + 2 ∧ p = (x, y) :=
by
-- We skip the proof steps and provide the statement only
sorry

end closest_point_on_line_y_eq_3x_plus_2_l203_203308


namespace hyperbola_asymptotes_slope_l203_203276

open Real

theorem hyperbola_asymptotes_slope (m : ℝ) : 
  (∀ x y : ℝ, (y ^ 2 / 16) - (x ^ 2 / 9) = 1 → (y = m * x ∨ y = -m * x)) → 
  m = 4 / 3 := 
by 
  sorry

end hyperbola_asymptotes_slope_l203_203276


namespace converse_equivalence_l203_203830

-- Definition of the original proposition
def original_proposition : Prop := ∀ (x : ℝ), x < 0 → x^2 > 0

-- Definition of the converse proposition
def converse_proposition : Prop := ∀ (x : ℝ), x^2 > 0 → x < 0

-- Theorem statement asserting the equivalence
theorem converse_equivalence : (converse_proposition = ¬ original_proposition) :=
sorry

end converse_equivalence_l203_203830


namespace weight_of_dried_grapes_l203_203863

/-- The weight of dried grapes available from 20 kg of fresh grapes given the water content in fresh and dried grapes. -/
theorem weight_of_dried_grapes (W_fresh W_dried : ℝ) (fresh_weight : ℝ) (weight_dried : ℝ) :
  W_fresh = 0.9 → 
  W_dried = 0.2 → 
  fresh_weight = 20 →
  weight_dried = (0.1 * fresh_weight) / (1 - W_dried) → 
  weight_dried = 2.5 :=
by sorry

end weight_of_dried_grapes_l203_203863


namespace trig_identity_l203_203654

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin θ + Real.cos θ) / Real.sin θ + Real.sin θ * Real.sin θ = 23 / 10 :=
sorry

end trig_identity_l203_203654


namespace max_min_values_of_g_l203_203757

noncomputable def g (x : ℝ) : ℝ := (Real.sin x)^8 + 8 * (Real.cos x)^8

theorem max_min_values_of_g :
  (∀ x : ℝ, g x ≤ 8) ∧ (∀ x : ℝ, g x ≥ 8 / 27) :=
by
  sorry

end max_min_values_of_g_l203_203757


namespace simple_interest_rate_l203_203720

theorem simple_interest_rate (P : ℝ) (r : ℝ) (T : ℝ) (SI : ℝ)
  (h1 : SI = P / 5)
  (h2 : T = 10)
  (h3 : SI = (P * r * T) / 100) :
  r = 2 :=
by
  sorry

end simple_interest_rate_l203_203720


namespace prove_condition_for_equality_l203_203524

noncomputable def condition_for_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  c = (b * (a ^ 3 - 1)) / a

theorem prove_condition_for_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ (c' : ℕ), (c' = (b * (a ^ 3 - 1)) / a) ∧ 
      c' > 0 ∧ 
      (a + b / c' = a ^ 3 * (b / c')) ) → 
  c = (b * (a ^ 3 - 1)) / a := 
sorry

end prove_condition_for_equality_l203_203524


namespace matrix_cubed_l203_203615

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, -2], ![2, -1]]

theorem matrix_cubed :
  (A * A * A) = ![![ -4, 2], ![-2, 1]] :=
by
  sorry

end matrix_cubed_l203_203615


namespace problem_equivalent_proof_l203_203619

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem problem_equivalent_proof : ((sqrt 3 - 2) ^ 0 - Real.logb 2 (sqrt 2)) = 1 / 2 :=
by
  sorry

end problem_equivalent_proof_l203_203619


namespace cos_of_three_pi_div_two_l203_203054

theorem cos_of_three_pi_div_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  sorry

end cos_of_three_pi_div_two_l203_203054


namespace find_S7_l203_203512

variable {a : ℕ → ℚ} {S : ℕ → ℚ}

axiom a1_def : a 1 = 1 / 2
axiom a_next_def : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * S n + 1
axiom S_def : ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem find_S7 : S 7 = 1457 / 2 := by
  sorry

end find_S7_l203_203512


namespace max_right_angles_in_triangle_l203_203699

theorem max_right_angles_in_triangle (a b c : ℝ) (h : a + b + c = 180) (ha : a = 90 ∨ b = 90 ∨ c = 90) : a = 90 ∧ b ≠ 90 ∧ c ≠ 90 ∨ b = 90 ∧ a ≠ 90 ∧ c ≠ 90 ∨ c = 90 ∧ a ≠ 90 ∧ b ≠ 90 :=
sorry

end max_right_angles_in_triangle_l203_203699


namespace kurt_less_marbles_than_dennis_l203_203802

theorem kurt_less_marbles_than_dennis
  (Laurie_marbles : ℕ)
  (Kurt_marbles : ℕ)
  (Dennis_marbles : ℕ)
  (h1 : Laurie_marbles = 37)
  (h2 : Laurie_marbles = Kurt_marbles + 12)
  (h3 : Dennis_marbles = 70) :
  Dennis_marbles - Kurt_marbles = 45 := by
  sorry

end kurt_less_marbles_than_dennis_l203_203802


namespace bronze_medals_l203_203045

theorem bronze_medals (G S B : ℕ) 
  (h1 : G + S + B = 89) 
  (h2 : G + S = 4 * B - 6) :
  B = 19 :=
sorry

end bronze_medals_l203_203045


namespace average_age_of_5_l203_203212

theorem average_age_of_5 (h1 : 19 * 15 = 285) (h2 : 9 * 16 = 144) (h3 : 15 = 71) :
    (285 - 144 - 71) / 5 = 14 :=
sorry

end average_age_of_5_l203_203212


namespace distance_24_km_l203_203872

noncomputable def distance_between_house_and_school (D : ℝ) :=
  let speed_to_school := 6
  let speed_to_home := 4
  let total_time := 10
  total_time = (D / speed_to_school) + (D / speed_to_home)

theorem distance_24_km : ∃ D : ℝ, distance_between_house_and_school D ∧ D = 24 :=
by
  use 24
  unfold distance_between_house_and_school
  sorry

end distance_24_km_l203_203872


namespace arithmetic_sequence_S10_l203_203404

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + n * d

def Sn (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_S10 :
  ∃ (a d : ℤ), d ≠ 0 ∧ Sn a d 8 = 16 ∧
  (arithmetic_sequence a d 3)^2 = (arithmetic_sequence a d 2) * (arithmetic_sequence a d 6) ∧
  Sn a d 10 = 30 :=
by
  sorry

end arithmetic_sequence_S10_l203_203404


namespace amount_spent_on_milk_is_1500_l203_203265

def total_salary (saved : ℕ) (saving_percent : ℕ) : ℕ := 
  saved / (saving_percent / 100)

def total_spent_excluding_milk (rent groceries education petrol misc : ℕ) : ℕ := 
  rent + groceries + education + petrol + misc

def amount_spent_on_milk (total_salary total_spent savings : ℕ) : ℕ := 
  total_salary - total_spent - savings

theorem amount_spent_on_milk_is_1500 :
  let rent := 5000
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let misc := 2500
  let savings := 2000
  let saving_percent := 10
  let salary := total_salary savings saving_percent
  let spent_excluding_milk := total_spent_excluding_milk rent groceries education petrol misc
  amount_spent_on_milk salary spent_excluding_milk savings = 1500 :=
by {
  sorry
}

end amount_spent_on_milk_is_1500_l203_203265


namespace estimated_value_at_28_l203_203636

-- Definitions based on the conditions
def regression_equation (x : ℝ) : ℝ := 4.75 * x + 257

-- Problem statement
theorem estimated_value_at_28 : regression_equation 28 = 390 :=
by
  -- Sorry is used to skip the proof
  sorry

end estimated_value_at_28_l203_203636


namespace arithmetic_sqrt_9_l203_203826

theorem arithmetic_sqrt_9 : ∃ x : ℝ, x^2 = 9 ∧ x = 3 :=
by
  use 3
  split
  · norm_num
    norm_num
  · norm_num

end arithmetic_sqrt_9_l203_203826


namespace evaTotalMarksCorrect_l203_203298

-- Definition of marks scored by Eva in each subject across semesters
def evaMathsMarksSecondSemester : Nat := 80
def evaArtsMarksSecondSemester : Nat := 90
def evaScienceMarksSecondSemester : Nat := 90

def evaMathsMarksFirstSemester : Nat := evaMathsMarksSecondSemester + 10
def evaArtsMarksFirstSemester : Nat := evaArtsMarksSecondSemester - 15
def evaScienceMarksFirstSemester : Nat := evaScienceMarksSecondSemester - (evaScienceMarksSecondSemester / 3)

-- Total marks in each semester
def totalMarksFirstSemester : Nat := evaMathsMarksFirstSemester + evaArtsMarksFirstSemester + evaScienceMarksFirstSemester
def totalMarksSecondSemester : Nat := evaMathsMarksSecondSemester + evaArtsMarksSecondSemester + evaScienceMarksSecondSemester

-- Combined total
def evaTotalMarks : Nat := totalMarksFirstSemester + totalMarksSecondSemester

-- Statement to prove
theorem evaTotalMarksCorrect : evaTotalMarks = 485 := 
by
  -- This needs to be proved as per the conditions and calculations above
  sorry

end evaTotalMarksCorrect_l203_203298


namespace four_digit_numbers_count_l203_203107

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l203_203107


namespace complete_the_square_d_l203_203918

theorem complete_the_square_d (x : ℝ) :
  ∃ c d, (x^2 + 10 * x + 9 = 0 → (x + c)^2 = d) ∧ d = 16 :=
sorry

end complete_the_square_d_l203_203918


namespace find_deaf_students_l203_203606

-- Definitions based on conditions
variables (B D : ℕ)
axiom deaf_students_triple_blind_students : D = 3 * B
axiom total_students : D + B = 240

-- Proof statement
theorem find_deaf_students (h1 : D = 3 * B) (h2 : D + B = 240) : D = 180 :=
sorry

end find_deaf_students_l203_203606


namespace quadratic_inequality_ab_l203_203635

theorem quadratic_inequality_ab (a b : ℝ) 
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + 1 > 0) ↔ -1 < x ∧ x < 1 / 3) :
  a * b = -6 :=
by
  -- Proof is omitted
  sorry

end quadratic_inequality_ab_l203_203635


namespace probability_of_x_plus_y_lt_4_l203_203479

open Classical -- To handle probability and random selection
open Set -- For geometric notions
open Filter -- For measure and integration

noncomputable def probability_condition (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) : Prop := x + y < 4

theorem probability_of_x_plus_y_lt_4 :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) →
                pr (λ (xy : ℝ × ℝ), probability_condition xy.1 xy.2 (and.intro (and.left xy.2) (and.right (and.left xy.2)))) = 7/9 :=
by sorry

end probability_of_x_plus_y_lt_4_l203_203479


namespace exam_problem_l203_203919

def balls : finset (bool × bool) := 
  {(ff, ff), (ff, tt), (tt, ff), (tt, tt)}

def exactly_one_black : finset (bool × bool) := 
  {(ff, tt), (tt, ff)}

def exactly_two_black : finset (bool × bool) := 
  {(tt, tt)}

def exactly_two_red : finset (bool × bool) := 
  {(ff, ff)}

def at_least_one_black : finset (bool × bool) := 
  {(tt, tt), (ff, tt), (tt, ff)}

def mutually_exclusive (A B : finset (bool × bool)) : Prop :=
  A ∩ B = ∅

def not_contradictory (A B : finset (bool × bool)) : Prop :=
  A ∪ B ≠ balls

theorem exam_problem :
  mutually_exclusive exactly_one_black exactly_two_black ∧
  not_contradictory exactly_one_black exactly_two_black := 
begin
  split,
  {
    unfold mutually_exclusive,
    apply finset.eq_empty_iff_forall_not_mem.2,
    intro x,
    unfold exactly_one_black exactly_two_black,
    simp at x,
    tauto,
  },
  {
    unfold not_contradictory,
    unfold balls exactly_one_black exactly_two_black,
    simp,
  }
end

end exam_problem_l203_203919


namespace distance_CD_l203_203750

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  16 * (x + 2)^2 + 4 * y^2 = 64

def major_axis_distance : ℝ := 4
def minor_axis_distance : ℝ := 2

theorem distance_CD : ∃ (d : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 → d = 2 * Real.sqrt 5 :=
by
  sorry

end distance_CD_l203_203750


namespace common_ratio_of_geometric_series_l203_203629

theorem common_ratio_of_geometric_series : ∃ r : ℝ, ∀ n : ℕ, 
  r = (if n = 0 then 2 / 3
       else if n = 1 then (2 / 3) * (2 / 3)
       else if n = 2 then (2 / 3) * (2 / 3) * (2 / 3)
       else sorry)
  ∧ r = 2 / 3 := sorry

end common_ratio_of_geometric_series_l203_203629


namespace union_comm_union_assoc_inter_distrib_union_l203_203546

variables {α : Type*} (A B C : Set α)

theorem union_comm : A ∪ B = B ∪ A := sorry

theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := sorry

theorem inter_distrib_union : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := sorry

end union_comm_union_assoc_inter_distrib_union_l203_203546


namespace number_multiplied_value_l203_203576

theorem number_multiplied_value (x : ℝ) :
  (4 / 6) * x = 8 → x = 12 :=
by
  sorry

end number_multiplied_value_l203_203576


namespace slope_of_perpendicular_line_l203_203914

theorem slope_of_perpendicular_line (m1 m2 : ℝ) : 
  (5*x - 2*y = 10) →  ∃ m2, m2 = (-2/5) :=
by sorry

end slope_of_perpendicular_line_l203_203914


namespace probability_of_losing_weight_l203_203869

theorem probability_of_losing_weight (total_volunteers lost_weight : ℕ) (h_total : total_volunteers = 1000) (h_lost : lost_weight = 241) : 
    (lost_weight : ℚ) / total_volunteers = 0.24 := by
  sorry

end probability_of_losing_weight_l203_203869


namespace four_digit_numbers_count_l203_203160

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l203_203160


namespace complex_power_sum_eq_self_l203_203550

theorem complex_power_sum_eq_self (z : ℂ) (h : z^2 + z + 1 = 0) : z^100 + z^101 + z^102 + z^103 = z :=
sorry

end complex_power_sum_eq_self_l203_203550


namespace cylindrical_coordinates_l203_203895

noncomputable def convert_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2)
  let theta := real.arctan (y / x)
  (r, theta, z)

theorem cylindrical_coordinates (r theta z : ℝ) (hθ_range : 0 ≤ theta ∧ theta < 2 * real.pi)
  (hr_pos : r > 0) :
  convert_to_cylindrical 3 (-3 * real.sqrt 3) 4 = (6, 4 * real.pi / 3, 4) :=
begin
  -- Proof goes here
  sorry
end

end cylindrical_coordinates_l203_203895


namespace find_f_6_5_l203_203923

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : is_even_function f
axiom periodic_f : ∀ x, f (x + 4) = f x
axiom f_in_interval : ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = x - 2

theorem find_f_6_5 : f 6.5 = -0.5 := by
  sorry

end find_f_6_5_l203_203923


namespace marbles_with_at_least_one_blue_l203_203795

theorem marbles_with_at_least_one_blue :
  (Nat.choose 10 4) - (Nat.choose 8 4) = 140 :=
by
  sorry

end marbles_with_at_least_one_blue_l203_203795


namespace prob_black_third_no_replacement_prob_black_third_with_replacement_xi_distribution_expectation_l203_203672

-- Define basic conditions
def total_balls := 10
def black_balls := 6
def white_balls := 4
def draws := 3

-- Question 1: Without Replacement
theorem prob_black_third_no_replacement
  (first_white : bool := true)
  (draws := {3, without_replacement}) :
  (first_white → (P(black_on_third_draw) = 2 / 3))
:=
sorry

-- Question 2: With Replacement
theorem prob_black_third_with_replacement
  (first_white : bool := true)
  (draws := {3, with_replacement}) :
  (first_white → (P(black_on_third_draw) = 3 / 5))
:=
sorry

-- Question 3: Distribution and Expectation of White Balls Drawn
def xi_distribution := pmf.binomial 3 (2 / 5)

theorem xi_distribution_expectation :
  (pmf.expectation xi_distribution = 6 / 5)
:=
sorry

end prob_black_third_no_replacement_prob_black_third_with_replacement_xi_distribution_expectation_l203_203672


namespace find_threedigit_number_l203_203456

-- Define the three-digit number and its reverse
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

-- Define the condition of adding the number and its reverse to get 1777
def number_sum_condition (a b c : ℕ) : Prop :=
  original_number a b c + reversed_number a b c = 1777

-- Prove the existence of digits a, b, and c that satisfy the conditions
theorem find_threedigit_number :
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  original_number a b c = 859 ∧ 
  reversed_number a b c = 958 ∧ 
  number_sum_condition a b c :=
sorry

end find_threedigit_number_l203_203456


namespace minimize_J_l203_203941

noncomputable def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 5 * (1 - p) * q - 6 * (1 - p) * (1 - q) + 2 * p

noncomputable def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

theorem minimize_J (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : p = 11 / 18 ↔ ∀ q, 0 ≤ q ∧ q ≤ 1 → J p = J (11 / 18) := 
by
  sorry

end minimize_J_l203_203941


namespace center_of_circle_l203_203059

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := 4 * x^2 - 8 * x + 4 * y^2 - 24 * y - 36 = 0

-- Define what it means to be the center of the circle, which is (h, k)
def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 1

-- The statement that we need to prove
theorem center_of_circle : is_center 1 3 :=
sorry

end center_of_circle_l203_203059


namespace pyramid_surface_area_and_volume_l203_203878

def s := 8
def PF := 15

noncomputable def FM := s / 2
noncomputable def PM := Real.sqrt (PF^2 + FM^2)
noncomputable def baseArea := s^2
noncomputable def lateralAreaTriangle := (1 / 2) * s * PM
noncomputable def totalSurfaceArea := baseArea + 4 * lateralAreaTriangle
noncomputable def volume := (1 / 3) * baseArea * PF

theorem pyramid_surface_area_and_volume :
  totalSurfaceArea = 64 + 16 * Real.sqrt 241 ∧
  volume = 320 :=
by
  sorry

end pyramid_surface_area_and_volume_l203_203878


namespace find_a10_l203_203737

theorem find_a10 (a_n : ℕ → ℤ) (d : ℤ) (h1 : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h2 : 5 * a_n 3 = a_n 3 ^ 2)
  (h3 : (a_n 3 + 2 * d) ^ 2 = (a_n 3 - d) * (a_n 3 + 11 * d))
  (h_nonzero : d ≠ 0) :
  a_n 10 = 23 :=
sorry

end find_a10_l203_203737


namespace jasmine_average_pace_l203_203541

-- Define the conditions given in the problem
def totalDistance : ℝ := 45
def totalTime : ℝ := 9

-- Define the assertion that needs to be proved
theorem jasmine_average_pace : totalDistance / totalTime = 5 :=
by sorry

end jasmine_average_pace_l203_203541


namespace original_cube_volume_l203_203200

theorem original_cube_volume (a : ℕ) (h : (a + 2) * (a + 1) * (a - 1) + 6 = a^3) : a = 2 :=
by sorry

example : 2^3 = 8 := by norm_num

end original_cube_volume_l203_203200


namespace product_of_consecutive_integers_plus_one_l203_203975

theorem product_of_consecutive_integers_plus_one (n : ℤ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1) ^ 2 := 
sorry

end product_of_consecutive_integers_plus_one_l203_203975


namespace A_work_days_l203_203448

variables (r_A r_B r_C : ℝ) (h1 : r_A + r_B = (1 / 3)) (h2 : r_B + r_C = (1 / 3)) (h3 : r_A + r_C = (5 / 24))

theorem A_work_days :
  1 / r_A = 9.6 := 
sorry

end A_work_days_l203_203448


namespace isosceles_triangle_largest_angle_l203_203486

/-- 
  Given an isosceles triangle where one of the angles is 20% smaller than a right angle,
  prove that the measure of one of the two largest angles is 54 degrees.
-/
theorem isosceles_triangle_largest_angle 
  (A B C : ℝ) 
  (triangle_ABC : A + B + C = 180)
  (isosceles_triangle : A = B ∨ A = C ∨ B = C)
  (smaller_angle : A = 0.80 * 90) :
  A = 54 ∨ B = 54 ∨ C = 54 :=
sorry

end isosceles_triangle_largest_angle_l203_203486


namespace strawberry_cost_l203_203480

variables (S C : ℝ)

theorem strawberry_cost :
  (C = 6 * S) ∧ (5 * S + 5 * C = 77) → S = 2.2 :=
by
  sorry

end strawberry_cost_l203_203480


namespace bank_balance_after_2_years_l203_203207

noncomputable def compound_interest (P₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P₀ * (1 + r)^n

theorem bank_balance_after_2_years :
  compound_interest 100 0.10 2 = 121 := 
  by
  sorry

end bank_balance_after_2_years_l203_203207


namespace sin_double_angle_l203_203638

theorem sin_double_angle (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 :=
by
  sorry

end sin_double_angle_l203_203638


namespace find_years_l203_203230

variable (p m x : ℕ)

def two_years_ago := p - 2 = 2 * (m - 2)
def four_years_ago := p - 4 = 3 * (m - 4)
def ratio_in_x_years (x : ℕ) := (p + x) * 2 = (m + x) * 3

theorem find_years (h1 : two_years_ago p m) (h2 : four_years_ago p m) : ratio_in_x_years p m 2 :=
by
  sorry

end find_years_l203_203230


namespace number_of_four_digit_numbers_l203_203159

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l203_203159


namespace calculate_expression_l203_203495

theorem calculate_expression : 2^3 * 2^3 + 2^3 = 72 := by
  sorry

end calculate_expression_l203_203495


namespace g_product_of_roots_l203_203193

def f (x : ℂ) : ℂ := x^6 + x^3 + 1
def g (x : ℂ) : ℂ := x^2 + 1

theorem g_product_of_roots (x_1 x_2 x_3 x_4 x_5 x_6 : ℂ) 
    (h1 : ∀ x, (x - x_1) * (x - x_2) * (x - x_3) * (x - x_4) * (x - x_5) * (x - x_6) = f x) :
    g x_1 * g x_2 * g x_3 * g x_4 * g x_5 * g x_6 = 1 :=
by 
    sorry

end g_product_of_roots_l203_203193


namespace zero_function_l203_203625

variable (f : ℝ × ℝ × ℝ → ℝ)

theorem zero_function (h : ∀ x y z : ℝ, f (x, y, z) = 2 * f (z, x, y)) : ∀ x y z : ℝ, f (x, y, z) = 0 :=
by
  intros
  sorry

end zero_function_l203_203625


namespace meat_pie_cost_l203_203066

variable (total_farthings : ℕ) (farthings_per_pfennig : ℕ) (remaining_pfennigs : ℕ)

def total_pfennigs (total_farthings farthings_per_pfennig : ℕ) : ℕ :=
  total_farthings / farthings_per_pfennig

def pie_cost (total_farthings farthings_per_pfennig remaining_pfennigs : ℕ) : ℕ :=
  total_pfennigs total_farthings farthings_per_pfennig - remaining_pfennigs

theorem meat_pie_cost
  (h1 : total_farthings = 54)
  (h2 : farthings_per_pfennig = 6)
  (h3 : remaining_pfennigs = 7) :
  pie_cost total_farthings farthings_per_pfennig remaining_pfennigs = 2 :=
by
  sorry

end meat_pie_cost_l203_203066


namespace smallest_blocks_needed_for_wall_l203_203250

noncomputable def smallest_number_of_blocks (wall_length : ℕ) (wall_height : ℕ) (block_length1 : ℕ) (block_length2 : ℕ) (block_length3 : ℝ) : ℕ :=
  let blocks_per_odd_row := wall_length / block_length1
  let blocks_per_even_row := wall_length / block_length1 - 1 + 2
  let odd_rows := wall_height / 2 + 1
  let even_rows := wall_height / 2
  odd_rows * blocks_per_odd_row + even_rows * blocks_per_even_row

theorem smallest_blocks_needed_for_wall :
  smallest_number_of_blocks 120 7 2 1 1.5 = 423 :=
by
  sorry

end smallest_blocks_needed_for_wall_l203_203250


namespace unique_a_b_l203_203313

-- Define the properties of the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * a * x + b else 7 - 2 * x

-- The function satisfies f(f(x)) = x for all x in its domain
theorem unique_a_b (a b : ℝ) (h : ∀ x : ℝ, f a b (f a b x) = x) : a + b = 13 / 4 :=
sorry

end unique_a_b_l203_203313


namespace female_adults_present_l203_203800

variable (children : ℕ) (male_adults : ℕ) (total_people : ℕ)
variable (children_count : children = 80) (male_adults_count : male_adults = 60) (total_people_count : total_people = 200)

theorem female_adults_present : ∃ (female_adults : ℕ), 
  female_adults = total_people - (children + male_adults) ∧ 
  female_adults = 60 :=
by
  sorry

end female_adults_present_l203_203800


namespace number_of_four_digit_numbers_l203_203154

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l203_203154


namespace count_four_digit_numbers_l203_203148

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l203_203148


namespace base_11_arithmetic_l203_203692

-- Define the base and the numbers in base 11
def base := 11

def a := 6 * base^2 + 7 * base + 4  -- 674 in base 11
def b := 2 * base^2 + 7 * base + 9  -- 279 in base 11
def c := 1 * base^2 + 4 * base + 3  -- 143 in base 11
def result := 5 * base^2 + 5 * base + 9  -- 559 in base 11

theorem base_11_arithmetic :
  (a - b + c) = result :=
sorry

end base_11_arithmetic_l203_203692


namespace value_of_a_l203_203363

theorem value_of_a (m n a : ℚ) 
  (h₁ : m = 5 * n + 5) 
  (h₂ : m + 2 = 5 * (n + a) + 5) : 
  a = 2 / 5 :=
by
  sorry

end value_of_a_l203_203363


namespace number_of_oranges_l203_203803

def apples : ℕ := 14
def more_oranges : ℕ := 10

theorem number_of_oranges (o : ℕ) (apples_eq : apples = 14) (more_oranges_eq : more_oranges = 10) :
  o = apples + more_oranges :=
by
  sorry

end number_of_oranges_l203_203803


namespace leak_drains_in_34_hours_l203_203239

-- Define the conditions
def pump_rate := 1 / 2 -- rate at which the pump fills the tank (tanks per hour)
def time_with_leak := 17 / 8 -- time to fill the tank with the pump and the leak (hours)

-- Define the combined rate of pump and leak
def combined_rate := 1 / time_with_leak -- tanks per hour

-- Define the leak rate
def leak_rate := pump_rate - combined_rate -- solve for leak rate

-- Define the proof statement
theorem leak_drains_in_34_hours : (1 / leak_rate) = 34 := by
    sorry

end leak_drains_in_34_hours_l203_203239


namespace solve_for_x_l203_203178

-- Define the necessary condition
def problem_statement (x : ℚ) : Prop :=
  x / 4 - x - 3 / 6 = 1

-- Prove that if the condition holds, then x = -14/9
theorem solve_for_x (x : ℚ) (h : problem_statement x) : x = -14 / 9 :=
by
  sorry

end solve_for_x_l203_203178


namespace pregnant_fish_in_each_tank_l203_203811

/-- Mark has 3 tanks for pregnant fish. Each tank has a certain number of pregnant fish and each fish
gives birth to 20 young. Mark has 240 young fish at the end. Prove that there are 4 pregnant fish in
each tank. -/
theorem pregnant_fish_in_each_tank (x : ℕ) (h1 : 3 * 20 * x = 240) : x = 4 := by
  sorry

end pregnant_fish_in_each_tank_l203_203811


namespace inequality_cannot_hold_l203_203937

theorem inequality_cannot_hold (a b : ℝ) (ha : a < b) (hb : b < 0) : a^3 ≤ b^3 :=
by
  sorry

end inequality_cannot_hold_l203_203937


namespace inequality_solution_l203_203925

theorem inequality_solution (a x : ℝ) (h : |a + 1| < 3) :
  (-4 < a ∧ a < -2 ∧ (x > -1 ∨ x < 1 + a)) ∨ 
  (a = -2 ∧ (x ∈ Set.univ \ {-1})) ∨ 
  (-2 < a ∧ a < 2 ∧ (x > 1 + a ∨ x < -1)) :=
by sorry

end inequality_solution_l203_203925


namespace new_area_shortening_other_side_l203_203553

-- Define the dimensions of the original card
def original_length : ℕ := 5
def original_width : ℕ := 7

-- Define the shortened length and the resulting area after shortening one side by 2 inches
def shortened_length_1 := original_length - 2
def new_area_1 : ℕ := shortened_length_1 * original_width
def condition_1 : Prop := new_area_1 = 21

-- Prove that shortening the width by 2 inches results in an area of 25 square inches
theorem new_area_shortening_other_side : condition_1 → (original_length * (original_width - 2) = 25) :=
by
  intro h
  sorry

end new_area_shortening_other_side_l203_203553


namespace weight_of_3_moles_HClO2_correct_l203_203574

def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cl : ℝ := 35.453
def atomic_weight_O : ℝ := 15.999

def molecular_weight_HClO2 : ℝ := (1 * atomic_weight_H) + (1 * atomic_weight_Cl) + (2 * atomic_weight_O)
def weight_of_3_moles_HClO2 : ℝ := 3 * molecular_weight_HClO2

theorem weight_of_3_moles_HClO2_correct : weight_of_3_moles_HClO2 = 205.377 := by
  sorry

end weight_of_3_moles_HClO2_correct_l203_203574


namespace eva_total_marks_correct_l203_203290

-- Definitions based on conditions
def math_marks_second_sem : ℕ := 80
def arts_marks_second_sem : ℕ := 90
def science_marks_second_sem : ℕ := 90

def math_marks_first_sem : ℕ := math_marks_second_sem + 10
def arts_marks_first_sem : ℕ := arts_marks_second_sem - 15
def science_marks_first_sem : ℕ := science_marks_second_sem - (science_marks_second_sem / 3)

def total_marks_first_sem : ℕ := math_marks_first_sem + arts_marks_first_sem + science_marks_first_sem
def total_marks_second_sem : ℕ := math_marks_second_sem + arts_marks_second_sem + science_marks_second_sem

def total_marks_both_sems : ℕ := total_marks_first_sem + total_marks_second_sem

-- Theorem to be proved
theorem eva_total_marks_correct : total_marks_both_sems = 485 := by
  -- Here, we state that we need to prove the total marks sum up to 485
  sorry

end eva_total_marks_correct_l203_203290


namespace distinct_real_pairs_l203_203626

theorem distinct_real_pairs (x y : ℝ) (h1 : x ≠ y) (h2 : x^100 - y^100 = 2^99 * (x - y)) (h3 : x^200 - y^200 = 2^199 * (x - y)) :
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) :=
sorry

end distinct_real_pairs_l203_203626


namespace sqrt_of_9_eq_3_l203_203827

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := 
by 
  sorry

end sqrt_of_9_eq_3_l203_203827


namespace encounter_count_l203_203008

theorem encounter_count (vA vB d : ℝ) (h₁ : 5 * d / vA = 9 * d / vB) :
  ∃ encounters : ℝ, encounters = 3023 :=
by
  sorry

end encounter_count_l203_203008


namespace expected_faces_of_5_in_100_rolls_l203_203821

theorem expected_faces_of_5_in_100_rolls (rolls : ℕ) (p : ℚ) (E : ℚ) :
  rolls = 100 ∧ p = 1/6 → E = (100 * (1/6)) := by
  sorry

end expected_faces_of_5_in_100_rolls_l203_203821


namespace number_of_zeros_of_f_l203_203567

noncomputable def f : ℝ → ℝ
| x => if x >= 0 then x^3 - 3*x + 1 else x^2 - 2*x - 4

theorem number_of_zeros_of_f : ∃ z, z = 3 := by
  sorry

end number_of_zeros_of_f_l203_203567


namespace larger_number_is_correct_l203_203864

theorem larger_number_is_correct : ∃ L : ℝ, ∃ S : ℝ, S = 48 ∧ (L - S = (1 : ℝ) / (3 : ℝ) * L) ∧ L = 72 :=
by
  sorry

end larger_number_is_correct_l203_203864


namespace stock_value_order_l203_203685

-- Define the initial investment and yearly changes
def initialInvestment : Float := 100
def firstYearChangeA : Float := 1.30
def firstYearChangeB : Float := 0.70
def firstYearChangeG : Float := 1.10
def firstYearChangeD : Float := 1.00 -- unchanged

def secondYearChangeA : Float := 0.90
def secondYearChangeB : Float := 1.35
def secondYearChangeG : Float := 1.05
def secondYearChangeD : Float := 1.10

-- Calculate the final values after two years
def finalValueA : Float := initialInvestment * firstYearChangeA * secondYearChangeA
def finalValueB : Float := initialInvestment * firstYearChangeB * secondYearChangeB
def finalValueG : Float := initialInvestment * firstYearChangeG * secondYearChangeG
def finalValueD : Float := initialInvestment * firstYearChangeD * secondYearChangeD

-- Theorem statement - Prove that the final order of the values is B < D < G < A
theorem stock_value_order : finalValueB < finalValueD ∧ finalValueD < finalValueG ∧ finalValueG < finalValueA := by
  sorry

end stock_value_order_l203_203685


namespace probability_x_plus_y_less_than_4_l203_203467

open Set Real

/-- Define the square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3

/-- Probability that a randomly chosen point (x, y) in the square satisfies x + y < 4 -/
theorem probability_x_plus_y_less_than_4 : 
  let area_square := 9 in
  let area_excluded_triangle := 2 in
  let relevant_area := area_square - area_excluded_triangle in
  (relevant_area / area_square : ℝ) = 7 / 9 :=
by
{ sorry }

end probability_x_plus_y_less_than_4_l203_203467


namespace hidden_dots_are_32_l203_203851

theorem hidden_dots_are_32 
  (visible_faces : List ℕ)
  (h_visible : visible_faces = [1, 2, 3, 4, 4, 5, 6, 6])
  (num_dice : ℕ)
  (h_num_dice : num_dice = 3)
  (faces_per_die : List ℕ)
  (h_faces_per_die : faces_per_die = [1, 2, 3, 4, 5, 6]) :
  63 - visible_faces.sum = 32 := by
  sorry

end hidden_dots_are_32_l203_203851


namespace prime_sol_is_7_l203_203056

theorem prime_sol_is_7 (p : ℕ) (x y : ℕ) (hp : Nat.Prime p) 
  (hx : p + 1 = 2 * x^2) (hy : p^2 + 1 = 2 * y^2) : 
  p = 7 := 
  sorry

end prime_sol_is_7_l203_203056


namespace pears_picking_total_l203_203970

theorem pears_picking_total :
  let Jason_day1 := 46
  let Keith_day1 := 47
  let Mike_day1 := 12
  let Alicia_day1 := 28
  let Tina_day1 := 33
  let Nicola_day1 := 52

  let Jason_day2 := Jason_day1 / 2
  let Keith_day2 := Keith_day1 / 2
  let Mike_day2 := Mike_day1 / 2
  let Alicia_day2 := 2 * Alicia_day1
  let Tina_day2 := 2 * Tina_day1
  let Nicola_day2 := 2 * Nicola_day1

  let Jason_day3 := (Jason_day1 + Jason_day2) / 2
  let Keith_day3 := (Keith_day1 + Keith_day2) / 2
  let Mike_day3 := (Mike_day1 + Mike_day2) / 2
  let Alicia_day3 := (Alicia_day1 + Alicia_day2) / 2
  let Tina_day3 := (Tina_day1 + Tina_day2) / 2
  let Nicola_day3 := (Nicola_day1 + Nicola_day2) / 2

  let Jason_total := Jason_day1 + Jason_day2 + Jason_day3
  let Keith_total := Keith_day1 + Keith_day2 + Keith_day3
  let Mike_total := Mike_day1 + Mike_day2 + Mike_day3
  let Alicia_total := Alicia_day1 + Alicia_day2 + Alicia_day3
  let Tina_total := Tina_day1 + Tina_day2 + Tina_day3
  let Nicola_total := Nicola_day1 + Nicola_day2 + Nicola_day3

  let overall_total := Jason_total + Keith_total + Mike_total + Alicia_total + Tina_total + Nicola_total

  overall_total = 747 := by
  intro Jason_day1 Jason_day2 Jason_day3 Jason_total
  intro Keith_day1 Keith_day2 Keith_day3 Keith_total
  intro Mike_day1 Mike_day2 Mike_day3 Mike_total
  intro Alicia_day1 Alicia_day2 Alicia_day3 Alicia_total
  intro Tina_day1 Tina_day2 Tina_day3 Tina_total
  intro Nicola_day1 Nicola_day2 Nicola_day3 Nicola_total

  sorry

end pears_picking_total_l203_203970


namespace total_tickets_sold_l203_203875

-- Definitions of the conditions as given in the problem
def price_adult : ℕ := 7
def price_child : ℕ := 4
def total_revenue : ℕ := 5100
def child_tickets_sold : ℕ := 400

-- The main statement (theorem) to prove
theorem total_tickets_sold:
  ∃ (A C : ℕ), C = child_tickets_sold ∧ price_adult * A + price_child * C = total_revenue ∧ (A + C = 900) :=
by
  sorry

end total_tickets_sold_l203_203875


namespace arithmetic_sequences_ratio_l203_203068

theorem arithmetic_sequences_ratio (x y a1 a2 a3 b1 b2 b3 b4 : Real) (hxy : x ≠ y) 
  (h_arith1 : a1 = x + (y - x) / 4 ∧ a2 = x + 2 * (y - x) / 4 ∧ a3 = x + 3 * (y - x) / 4 ∧ y = x + 4 * (y - x) / 4)
  (h_arith2 : b1 = x - (y - x) / 2 ∧ b2 = x + (y - x) / 2 ∧ b3 = x + 2 * (y - x) / 2 ∧ y = x + 2 * (y - x) / 2 ∧ b4 = y + (y - x) / 2):
  (b4 - b3) / (a2 - a1) = 8 / 3 := 
sorry

end arithmetic_sequences_ratio_l203_203068


namespace min_value_quadratic_l203_203929

theorem min_value_quadratic :
  ∀ (x : ℝ), (2 * x^2 - 8 * x + 15) ≥ 7 :=
by
  -- We need to show that 2x^2 - 8x + 15 has a minimum value of 7
  sorry

end min_value_quadratic_l203_203929


namespace correct_option_l203_203773

-- Definitions representing the conditions
variable (a b c : Line) -- Define the lines a, b, and c

-- Conditions for the problem
def is_parallel (x y : Line) : Prop := -- Define parallel property
  sorry

def is_perpendicular (x y : Line) : Prop := -- Define perpendicular property
  sorry

noncomputable def proof_statement : Prop :=
  is_parallel a b → is_perpendicular a c → is_perpendicular b c

-- Lean statement of the proof problem
theorem correct_option (h1 : is_parallel a b) (h2 : is_perpendicular a c) : is_perpendicular b c :=
  sorry

end correct_option_l203_203773


namespace four_digit_number_count_l203_203112

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l203_203112


namespace arithmetic_mean_no_zero_digit_l203_203981

open_locale nat

/-- Given a set of numbers in the form {11, 111, 1111, ..., 111111111},
    prove that the arithmetic mean N of these nine numbers does not contain digit 0. -/
theorem arithmetic_mean_no_zero_digit :
  let S := (list.iota 9).map (λ n, (10^n - 1) / 9)
  let N := (11 / 9) * (S.map (λ x, 10 * x)).sum / 9 in
  ¬ (0 ∈ (N.to_nat.digits 10)) :=
by {
  -- Translation of given condition and goal
  let S := (list.iota 9).map (λ n, (10^n - 1) / 9),
  let N := (11 / 9) * (S.map (λ x, 10 * x)).sum / 9,
  show ¬ (0 ∈ (N.to_nat.digits 10)),
  sorry
}

end arithmetic_mean_no_zero_digit_l203_203981


namespace probability_cheryl_same_color_l203_203589

theorem probability_cheryl_same_color :
  let total_marble_count := 12
  let marbles_per_color := 3
  let carol_draw := 3
  let claudia_draw := 3
  let cheryl_draw := total_marble_count - carol_draw - claudia_draw
  let num_colors := 4

  0 < marbles_per_color ∧ marbles_per_color * num_colors = total_marble_count ∧
  0 < carol_draw ∧ carol_draw <= total_marble_count ∧
  0 < claudia_draw ∧ claudia_draw <= total_marble_count - carol_draw ∧
  0 < cheryl_draw ∧ cheryl_draw <= total_marble_count - carol_draw - claudia_draw ∧
  num_colors * (num_colors - 1) > 0
  →
  ∃ (p : ℚ), p = 2 / 55 := 
sorry

end probability_cheryl_same_color_l203_203589


namespace twenty_percent_greater_than_40_l203_203580

theorem twenty_percent_greater_than_40 (x : ℝ) (h : x = 40 + 0.2 * 40) : x = 48 := by
sorry

end twenty_percent_greater_than_40_l203_203580


namespace units_digit_17_pow_53_l203_203234

theorem units_digit_17_pow_53 : (17^53) % 10 = 7 := 
by sorry

end units_digit_17_pow_53_l203_203234


namespace polynomial_expression_l203_203019

theorem polynomial_expression :
  (2 * x^2 + 3 * x + 7) * (x + 1) - (x + 1) * (x^2 + 4 * x - 63) + (3 * x - 14) * (x + 1) * (x + 5) = 4 * x^3 + 4 * x^2 :=
by
  sorry

end polynomial_expression_l203_203019


namespace tan_sub_eq_one_third_l203_203650

theorem tan_sub_eq_one_third (α β : Real) (hα : Real.tan α = 3) (hβ : Real.tan β = 4/3) : 
  Real.tan (α - β) = 1/3 := by
  sorry

end tan_sub_eq_one_third_l203_203650


namespace min_diff_proof_l203_203227

noncomputable def triangleMinDiff : ℕ :=
  let PQ := 666
  let QR := 667
  let PR := 2010 - PQ - QR
  if (PQ < QR ∧ QR < PR ∧ PQ + QR > PR ∧ PQ + PR > QR ∧ PR + QR > PQ) then QR - PQ else 0

theorem min_diff_proof :
  ∃ PQ QR PR : ℕ, PQ + QR + PR = 2010 ∧ PQ < QR ∧ QR < PR ∧ (PQ + QR > PR) ∧ (PQ + PR > QR) ∧ (PR + QR > PQ) ∧ (QR - PQ = triangleMinDiff) := sorry

end min_diff_proof_l203_203227


namespace count_four_digit_numbers_l203_203149

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l203_203149


namespace students_failed_l203_203220

theorem students_failed (total_students : ℕ) (percent_A : ℚ) (fraction_BC : ℚ) (students_A : ℕ)
  (students_remaining : ℕ) (students_BC : ℕ) (students_failed : ℕ)
  (h1 : total_students = 32) (h2 : percent_A = 0.25) (h3 : fraction_BC = 0.25)
  (h4 : students_A = total_students * percent_A)
  (h5 : students_remaining = total_students - students_A)
  (h6 : students_BC = students_remaining * fraction_BC)
  (h7 : students_failed = total_students - students_A - students_BC) :
  students_failed = 18 :=
sorry

end students_failed_l203_203220


namespace expected_games_is_correct_l203_203241

def prob_A_wins : ℚ := 2 / 3
def prob_B_wins : ℚ := 1 / 3
def max_games : ℕ := 6

noncomputable def expected_games : ℚ :=
  2 * (prob_A_wins^2 + prob_B_wins^2) +
  4 * (prob_A_wins * prob_B_wins * (prob_A_wins^2 + prob_B_wins^2)) +
  6 * (prob_A_wins * prob_B_wins)^2

theorem expected_games_is_correct : expected_games = 266 / 81 := by
  sorry

end expected_games_is_correct_l203_203241


namespace find_ratio_PS_SR_l203_203532

variable {P Q R S : Type}
variable [MetricSpace P]
variable [MetricSpace Q]
variable [MetricSpace R]
variable [MetricSpace S]

-- Given conditions
variable (PQ QR PR : ℝ)
variable (hPQ : PQ = 6)
variable (hQR : QR = 8)
variable (hPR : PR = 10)
variable (QS : ℝ)
variable (hQS : QS = 6)

-- Points on the segments
variable (PS : ℝ)
variable (SR : ℝ)

-- The theorem to be proven: the ratio PS : SR = 0 : 1
theorem find_ratio_PS_SR (hPQ : PQ = 6) (hQR : QR = 8) (hPR : PR = 10) (hQS : QS = 6) :
    PS = 0 ∧ SR = 10 → PS / SR = 0 :=
by
  sorry

end find_ratio_PS_SR_l203_203532


namespace beam_equation_correctness_l203_203726

-- Define the conditions
def total_selling_price : ℕ := 6210
def freight_per_beam : ℕ := 3

-- Define the unknown quantity
variable (x : ℕ)

-- State the theorem
theorem beam_equation_correctness
  (h1 : total_selling_price = 6210)
  (h2 : freight_per_beam = 3) :
  freight_per_beam * (x - 1) = total_selling_price / x := 
sorry

end beam_equation_correctness_l203_203726


namespace complex_sum_series_l203_203190

theorem complex_sum_series (ω : ℂ) (h1 : ω ^ 7 = 1) (h2 : ω ≠ 1) :
  ω ^ 16 + ω ^ 18 + ω ^ 20 + ω ^ 22 + ω ^ 24 + ω ^ 26 + ω ^ 28 + ω ^ 30 + 
  ω ^ 32 + ω ^ 34 + ω ^ 36 + ω ^ 38 + ω ^ 40 + ω ^ 42 + ω ^ 44 + ω ^ 46 +
  ω ^ 48 + ω ^ 50 + ω ^ 52 + ω ^ 54 = -1 :=
sorry

end complex_sum_series_l203_203190


namespace variance_of_data_set_l203_203324

theorem variance_of_data_set (m : ℝ) (h_mean : (6 + 7 + 8 + 9 + m) / 5 = 8) :
    (1/5) * ((6-8)^2 + (7-8)^2 + (8-8)^2 + (9-8)^2 + (m-8)^2) = 2 := 
sorry

end variance_of_data_set_l203_203324


namespace sufficient_and_necessary_condition_l203_203762

theorem sufficient_and_necessary_condition (a : ℝ) : 
  (0 < a ∧ a < 4) ↔ ∀ x : ℝ, (x^2 - a * x + a) > 0 :=
by sorry

end sufficient_and_necessary_condition_l203_203762


namespace four_digit_numbers_count_l203_203130

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l203_203130


namespace expenses_of_5_yuan_l203_203987

-- Define the given condition: income of 5 yuan is +5 yuan
def income (x : Int) : Int := x

-- Define the opposite relationship between income and expenses
def expenses (x : Int) : Int := -income x

-- Proof statement to show that expenses of 5 yuan are -5 yuan, given the above definitions
theorem expenses_of_5_yuan : expenses 5 = -5 := by
  -- The proof is not provided here, so we use sorry to indicate its place
  sorry

end expenses_of_5_yuan_l203_203987


namespace expenses_of_5_yuan_l203_203985

-- Define the given condition: income of 5 yuan is +5 yuan
def income (x : Int) : Int := x

-- Define the opposite relationship between income and expenses
def expenses (x : Int) : Int := -income x

-- Proof statement to show that expenses of 5 yuan are -5 yuan, given the above definitions
theorem expenses_of_5_yuan : expenses 5 = -5 := by
  -- The proof is not provided here, so we use sorry to indicate its place
  sorry

end expenses_of_5_yuan_l203_203985


namespace rainy_days_l203_203969

theorem rainy_days (n R NR : ℤ) 
  (h1 : n * R + 4 * NR = 26)
  (h2 : 4 * NR - n * R = 14)
  (h3 : R + NR = 7) : 
  R = 2 := 
sorry

end rainy_days_l203_203969


namespace petya_vasya_same_sum_l203_203203

theorem petya_vasya_same_sum :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 2^99 * (2^100 - 1) :=
by
  sorry

end petya_vasya_same_sum_l203_203203


namespace tangent_slope_correct_l203_203323

noncomputable def slope_of_directrix (focus: ℝ × ℝ) (p1: ℝ × ℝ) (p2: ℝ × ℝ) : ℝ :=
  let c1 := p1
  let c2 := p2
  let radius1 := Real.sqrt ((c1.1 + 1)^2 + (c1.2 + 1)^2)
  let radius2 := Real.sqrt ((c2.1 - 2)^2 + (c2.2 - 2)^2)
  let dist := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let slope := (focus.2 - p1.2) / (focus.1 - p1.1)
  let tangent_slope := (9 : ℝ) / (7 : ℝ) + (4 * Real.sqrt 2) / 7
  tangent_slope

theorem tangent_slope_correct :
  (slope_of_directrix (0, 0) (-1, -1) (2, 2) = (9 + 4 * Real.sqrt 2) / 7) ∨
  (slope_of_directrix (0, 0) (-1, -1) (2, 2) = (9 - 4 * Real.sqrt 2) / 7) :=
by
  -- Proof omitted here
  sorry

end tangent_slope_correct_l203_203323


namespace sum_of_remainders_mod_500_l203_203062

theorem sum_of_remainders_mod_500 : 
  (5 ^ (5 ^ (5 ^ 5)) + 2 ^ (2 ^ (2 ^ 2))) % 500 = 49 := by
  sorry

end sum_of_remainders_mod_500_l203_203062


namespace cube_problem_l203_203499

theorem cube_problem (n : ℕ) (h1 : n > 3) :
  (12 * (n - 4) = (n - 2)^3) → n = 5 :=
by {
  sorry
}

end cube_problem_l203_203499


namespace count_congruent_numbers_less_than_500_l203_203338

-- Definitions of the conditions
def is_congruent_to_modulo (n a m : ℕ) : Prop := (n % m) = a

-- Main problem statement: Proving that the count of numbers under 500 that satisfy the conditions is 71.
theorem count_congruent_numbers_less_than_500 : 
  { n : ℕ | n < 500 ∧ is_congruent_to_modulo n 3 7 }.card = 71 :=
by
  sorry

end count_congruent_numbers_less_than_500_l203_203338


namespace complement_union_l203_203662

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_union :
  (U \ M) ∪ N = {2, 3, 4} :=
sorry

end complement_union_l203_203662


namespace number_of_classes_l203_203591

theorem number_of_classes
  (s : ℕ)    -- s: number of students in each class
  (bpm : ℕ) -- bpm: books per month per student
  (months : ℕ) -- months: number of months in a year
  (total_books : ℕ) -- total_books: total books read by the entire student body in a year
  (H1 : bpm = 5)
  (H2 : months = 12)
  (H3 : total_books = 60)
  (H4 : total_books = s * bpm * months)
: s = 1 :=
by
  sorry

end number_of_classes_l203_203591


namespace mittens_pairing_possible_l203_203788

/--
In a kindergarten's lost and found basket, there are 30 mittens: 
10 blue, 10 green, 10 red, 15 right-hand, and 15 left-hand. 

Prove that it is always possible to create matching pairs of one right-hand 
and one left-hand mitten of the same color for 5 children.
-/
theorem mittens_pairing_possible : 
  (∃ (right_blue left_blue right_green left_green right_red left_red : ℕ), 
    right_blue + left_blue + right_green + left_green + right_red + left_red = 30 ∧
    right_blue ≤ 10 ∧ left_blue ≤ 10 ∧
    right_green ≤ 10 ∧ left_green ≤ 10 ∧
    right_red ≤ 10 ∧ left_red ≤ 10 ∧
    right_blue + right_green + right_red = 15 ∧
    left_blue + left_green + left_red = 15) →
  (∃ right_blue left_blue right_green left_green right_red left_red,
    min right_blue left_blue + 
    min right_green left_green + 
    min right_red left_red ≥ 5) :=
sorry

end mittens_pairing_possible_l203_203788


namespace value_of_hash_l203_203195

def hash (a b c d : ℝ) : ℝ := b^2 - 4 * a * c * d

theorem value_of_hash : hash 2 3 2 1 = -7 := by
  sorry

end value_of_hash_l203_203195


namespace proof_l203_203196

open Set

-- Universal set U
def U : Set ℕ := {x | x ∈ Finset.range 7}

-- Set A
def A : Set ℕ := {1, 3, 5}

-- Set B
def B : Set ℕ := {4, 5, 6}

-- Complement of A in U
def CU (s : Set ℕ) : Set ℕ := U \ s

-- Proof statement
theorem proof : (CU A) ∩ B = {4, 6} :=
by
  sorry

end proof_l203_203196


namespace P_intersection_complement_Q_l203_203083

-- Define sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }
def Q : Set ℝ := { x | x^2 ≥ 4 }

-- Prove the required intersection
theorem P_intersection_complement_Q : P ∩ (Set.univ \ Q) = { x | 0 ≤ x ∧ x < 2 } :=
by
  -- Proof will be inserted here
  sorry

end P_intersection_complement_Q_l203_203083


namespace common_ratio_of_geo_seq_l203_203728

variable {a : ℕ → ℝ} (q : ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem common_ratio_of_geo_seq :
  (∀ n, 0 < a n) →
  geometric_sequence a q →
  a 6 = a 5 + 2 * a 4 →
  q = 2 :=
by
  intros
  sorry

end common_ratio_of_geo_seq_l203_203728


namespace four_digit_number_count_l203_203114

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l203_203114


namespace expression_indeterminate_l203_203301

-- Given variables a, b, c, d which are real numbers
variables {a b c d : ℝ}

-- Statement asserting that the expression is indeterminate under given conditions
theorem expression_indeterminate
  (h : true) :
  ¬∃ k, (a^2 + b^2 - c^2 - 2 * b * d)/(a^2 + c^2 - b^2 - 2 * c * d) = k :=
sorry

end expression_indeterminate_l203_203301


namespace four_digit_numbers_count_l203_203134

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l203_203134


namespace problem_statement_l203_203681

theorem problem_statement (p q m n : ℕ) (x : ℚ)
  (h1 : p / q = 4 / 5) (h2 : m / n = 4 / 5) (h3 : x = 1 / 7) :
  x + (2 * q - p + 3 * m - 2 * n) / (2 * q + p - m + n) = 71 / 105 :=
by
  sorry

end problem_statement_l203_203681


namespace smallest_x_l203_203061

theorem smallest_x (x: ℕ) (hx: x > 0) (h: 11^2021 ∣ 5^(3*x) - 3^(4*x)) : 
  x = 11^2020 := sorry

end smallest_x_l203_203061


namespace probability_x_plus_y_lt_4_l203_203468

open Set

-- Define the square and the line
def square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }
def line_lt_4 : Set (ℝ × ℝ) := { p | p.1 + p.2 < 4 }

-- The probability to prove
theorem probability_x_plus_y_lt_4 : 
  (volume (square ∩ line_lt_4) / volume square) = 7 / 9 := 
sorry

end probability_x_plus_y_lt_4_l203_203468


namespace parabola_vertex_coordinates_l203_203831

theorem parabola_vertex_coordinates :
  ∃ (x y : ℝ), (∀ x : ℝ, y = 3 * x^2 + 2) ∧ x = 0 ∧ y = 2 :=
by
  sorry

end parabola_vertex_coordinates_l203_203831


namespace determine_a_l203_203768

theorem determine_a 
(h : ∃x, x = -1 ∧ 2 * x ^ 2 + a * x - a ^ 2 = 0) : a = -2 ∨ a = 1 :=
by
  -- Proof omitted
  sorry

end determine_a_l203_203768


namespace moon_arrangements_l203_203901

theorem moon_arrangements : 
  let word := "MOON" 
  let n := 4 -- number of letters in "MOON"
  let repeats := 2 -- number of times 'O' appears
  fact n / fact repeats = 12 :=
by sorry

end moon_arrangements_l203_203901


namespace min_containers_needed_l203_203584

theorem min_containers_needed 
  (total_boxes1 : ℕ) 
  (weight_box1 : ℕ) 
  (total_boxes2 : ℕ) 
  (weight_box2 : ℕ) 
  (weight_limit : ℕ) :
  total_boxes1 = 90000 →
  weight_box1 = 3300 →
  total_boxes2 = 5000 →
  weight_box2 = 200 →
  weight_limit = 100000 →
  (total_boxes1 * weight_box1 + total_boxes2 * weight_box2 + weight_limit - 1) / weight_limit = 3000 :=
by
  sorry

end min_containers_needed_l203_203584


namespace partial_fraction_sum_l203_203271

theorem partial_fraction_sum :
  (∃ A B C D E : ℝ, 
    (∀ x : ℝ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -5 → 
    (1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5))) ∧
    (A + B + C + D + E = 1 / 30)) :=
sorry

end partial_fraction_sum_l203_203271


namespace pascal_triangle_45th_number_l203_203013

theorem pascal_triangle_45th_number :
  let row := List.range (46 + 1) in
  row.nth 44 = some 1035 :=
by
  let row := List.range (46 + 1)
  have binom_46_2 : nat.binom 46 2 = 1035 := by
    -- Calculations for binomials can be validated here
    calc
      nat.binom 46 2 = 46 * 45 / (2 * 1) : by norm_num
      _ = 1035 : by norm_num
  show row.nth 44 = some (nat.binom 46 2) from by
    rw binom_46_2
    simp only [List.nth_range, option.some_eq_coe, nat.lt_succ_iff, nat.le_refl]
  sorry -- Additional reasoning if necessary

end pascal_triangle_45th_number_l203_203013


namespace percentage_difference_height_l203_203344

-- Define the heights of persons B, A, and C
variables (H_B H_A H_C : ℝ)

-- Condition: Person A's height is 30% less than person B's height
def person_A_height : Prop := H_A = 0.70 * H_B

-- Condition: Person C's height is 20% more than person A's height
def person_C_height : Prop := H_C = 1.20 * H_A

-- The proof problem: Prove that the percentage difference between H_B and H_C is 16%
theorem percentage_difference_height (h1 : person_A_height H_B H_A) (h2 : person_C_height H_A H_C) :
  ((H_B - H_C) / H_B) * 100 = 16 :=
by
  sorry

end percentage_difference_height_l203_203344


namespace probability_of_x_plus_y_less_than_4_l203_203476

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l203_203476


namespace higher_room_amount_higher_60_l203_203385

variable (higher_amount : ℕ)

theorem higher_room_amount_higher_60 
  (total_rent : ℕ) (amount_credited_50 : ℕ)
  (total_reduction : ℕ)
  (condition1 : total_rent = 400)
  (condition2 : amount_credited_50 = 50)
  (condition3 : total_reduction = total_rent / 4)
  (condition4 : 10 * higher_amount - 10 * amount_credited_50 = total_reduction) :
  higher_amount = 60 := 
sorry

end higher_room_amount_higher_60_l203_203385


namespace sin_range_l203_203071

theorem sin_range (p : Prop) (q : Prop) :
  (¬ ∃ x : ℝ, Real.sin x = 3/2) → (∀ x : ℝ, x^2 - 4 * x + 5 > 0) → (¬p ∧ q) :=
by
  sorry

end sin_range_l203_203071


namespace total_points_correct_l203_203286

-- Define the scores
def Marius (Darius : ℕ) : ℕ := Darius + 3
def Matt (Darius : ℕ) : ℕ := Darius + 5

-- Define the total points function
def total_points (Darius : ℕ) : ℕ :=
  Darius + Marius Darius + Matt Darius

-- Specific value for Darius's score
def Darius_score : ℕ := 10

-- The theorem that proves the total score is 38 given Darius's score
theorem total_points_correct :
  total_points Darius_score = 38 :=
by
  sorry

end total_points_correct_l203_203286


namespace number_of_four_digit_numbers_l203_203158

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l203_203158


namespace gcd_fa_fb_l203_203962

def f (x : ℤ) : ℤ := x * x - x + 2008

def a : ℤ := 102
def b : ℤ := 103

theorem gcd_fa_fb : Int.gcd (f a) (f b) = 2 := by
  sorry

end gcd_fa_fb_l203_203962


namespace real_root_polynomials_l203_203500

theorem real_root_polynomials (P : Polynomial ℝ) (n : ℕ):
  (∀ i, i ∈ (Finset.range (n + 1)) → P.coeff i = 1 ∨ P.coeff i = -1) ∧ P.degree ≤ n ∧ n ≤ 3
  → (∃ (Q : Polynomial ℝ), Q = P ∧ (Q = Polynomial.Coeff (x - 1) ∨ Q = Polynomial.Coeff (x + 1) 
  ∨ Q = Polynomial.Coeff (x^2 + x - 1) ∨ Q = Polynomial.Coeff (x^2 - x - 1) 
  ∨ Q = Polynomial.Coeff (x^3 + x^2 - x - 1) ∨ Q = Polynomial.Coeff (x^3 - x^2 - x + 1)))
:= sorry

end real_root_polynomials_l203_203500


namespace interchange_digits_product_l203_203809

-- Definition of the proof problem
theorem interchange_digits_product (n a b k : ℤ) (h1 : n = 10 * a + b) (h2 : n = (k + 1) * (a + b)) :
  ∃ x : ℤ, (10 * b + a) = x * (a + b) ∧ x = 10 - k :=
by
  existsi (10 - k)
  sorry

end interchange_digits_product_l203_203809


namespace sin_cos_plus_one_l203_203927

theorem sin_cos_plus_one (x : ℝ) (h : Real.tan x = 1 / 3) : Real.sin x * Real.cos x + 1 = 13 / 10 :=
by
  sorry

end sin_cos_plus_one_l203_203927


namespace find_number_of_ducks_l203_203522

variable {D H : ℕ}

-- Definition of the conditions
def total_animals (D H : ℕ) : Prop := D + H = 11
def total_legs (D H : ℕ) : Prop := 2 * D + 4 * H = 30
def number_of_ducks (D : ℕ) : Prop := D = 7

-- Lean statement for the proof problem
theorem find_number_of_ducks (D H : ℕ) (h1 : total_animals D H) (h2 : total_legs D H) : number_of_ducks D :=
by
  sorry

end find_number_of_ducks_l203_203522


namespace joes_speed_l203_203542

theorem joes_speed (pete_speed : ℝ) (joe_speed : ℝ) (time_minutes : ℝ) (distance : ℝ) (h1 : joe_speed = 2 * pete_speed) (h2 : time_minutes = 40) (h3 : distance = 16) : joe_speed = 16 :=
by
  sorry

end joes_speed_l203_203542


namespace prime_constraint_unique_solution_l203_203057

theorem prime_constraint_unique_solution (p x y : ℕ) (h_prime : Prime p)
  (h1 : p + 1 = 2 * x^2)
  (h2 : p^2 + 1 = 2 * y^2) :
  p = 7 :=
by
  sorry

end prime_constraint_unique_solution_l203_203057


namespace frames_sharing_point_with_line_e_l203_203704

def frame_shares_common_point_with_line (n : ℕ) : Prop := 
  n = 0 ∨ n = 1 ∨ n = 9 ∨ n = 17 ∨ n = 25 ∨ n = 33 ∨ n = 41 ∨ n = 49 ∨
  n = 6 ∨ n = 14 ∨ n = 22 ∨ n = 30 ∨ n = 38 ∨ n = 46

theorem frames_sharing_point_with_line_e :
  ∀ (i : ℕ), i < 50 → frame_shares_common_point_with_line i = 
  (i = 0 ∨ i = 1 ∨ i = 9 ∨ i = 17 ∨ i = 25 ∨ i = 33 ∨ i = 41 ∨ i = 49 ∨
   i = 6 ∨ i = 14 ∨ i = 22 ∨ i = 30 ∨ i = 38 ∨ i = 46) := 
by 
  sorry

end frames_sharing_point_with_line_e_l203_203704


namespace prob_at_least_3_is_correct_expected_value_X_is_correct_l203_203394

-- Define the events
def needs_device_A : Prop := true -- Placeholder, actual definition would depend on probability space definition
def needs_device_B : Prop := true -- Placeholder
def needs_device_C : Prop := true -- Placeholder
def needs_device_D : Prop := true -- Placeholder

-- Probabilities of each person needing the device
axiom prob_A : ℝ := 0.6
axiom prob_B : ℝ := 0.5
axiom prob_C : ℝ := 0.5
axiom prob_D : ℝ := 0.4

-- Define independence of events
axiom independence : ∀ (P Q : Prop), P ∧ Q = (P * Q) -- Placeholder for actual independence definition

-- Probability calculation for at least 3 people
def at_least_3 : ℝ :=
  prob_A * prob_B * prob_C * prob_D + 
  (1 - prob_A) * prob_B * prob_C * prob_D + 
  prob_A * (1 - prob_B) * prob_C * prob_D + 
  prob_A * prob_B * (1 - prob_C) * prob_D + 
  prob_A * prob_B * prob_C * (1 - prob_D)

-- Expected value calculation for X
def P_X_0 : ℝ := (1 - prob_A) * (1 - prob_B) * (1 - prob_C) * (1 - prob_D)
def P_X_1 : ℝ := prob_A * (1 - prob_B) * (1 - prob_C) * (1 - prob_D) +
                (1 - prob_A) * prob_B * (1 - prob_C) * (1 - prob_D) +
                (1 - prob_A) * (1 - prob_B) * prob_C * (1 - prob_D) +
                (1 - prob_A) * (1 - prob_B) * (1 - prob_C) * prob_D
def P_X_2 : ℝ := -- Placeholder for calculation
def P_X_3 : ℝ := -- Placeholder for calculation
def P_X_4 : ℝ := prob_A * prob_B * prob_C * prob_D

def expected_value_X : ℝ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3 + 4 * P_X_4

-- Statements to prove
theorem prob_at_least_3_is_correct : at_least_3 = 0.31 := by sorry
theorem expected_value_X_is_correct : expected_value_X = 2 := by sorry

end prob_at_least_3_is_correct_expected_value_X_is_correct_l203_203394


namespace simplify_329_mul_101_simplify_54_mul_98_plus_46_mul_98_simplify_98_mul_125_simplify_37_mul_29_plus_37_l203_203977

theorem simplify_329_mul_101 : 329 * 101 = 33229 := by
  sorry

theorem simplify_54_mul_98_plus_46_mul_98 : 54 * 98 + 46 * 98 = 9800 := by
  sorry

theorem simplify_98_mul_125 : 98 * 125 = 12250 := by
  sorry

theorem simplify_37_mul_29_plus_37 : 37 * 29 + 37 = 1110 := by
  sorry

end simplify_329_mul_101_simplify_54_mul_98_plus_46_mul_98_simplify_98_mul_125_simplify_37_mul_29_plus_37_l203_203977


namespace algebraic_expression_value_l203_203939

variable (x y : ℝ)

def condition1 : Prop := y - x = -1
def condition2 : Prop := x * y = 2

def expression : ℝ := -2 * x^3 * y + 4 * x^2 * y^2 - 2 * x * y^3

theorem algebraic_expression_value (h1 : condition1 x y) (h2 : condition2 x y) : expression x y = -4 := 
by
  sorry

end algebraic_expression_value_l203_203939


namespace distance_first_to_last_tree_l203_203623

theorem distance_first_to_last_tree 
    (n_trees : ℕ) 
    (distance_first_to_fifth : ℕ)
    (h1 : n_trees = 8)
    (h2 : distance_first_to_fifth = 80) 
    : ∃ distance_first_to_last, distance_first_to_last = 140 := by
  sorry

end distance_first_to_last_tree_l203_203623


namespace expenses_neg_five_given_income_five_l203_203991

theorem expenses_neg_five_given_income_five 
  (income_5 : ℤ)
  (income_5_pos : income_5 = 5)
  (opposite : ∀ (x : ℤ), -x = -1 * x) :
  (expenses_5 : ℤ) (expenses_5 = -5) :=
by {
  sorry
}

end expenses_neg_five_given_income_five_l203_203991


namespace sequence_bounded_l203_203837

theorem sequence_bounded (a : ℕ → ℕ) (a1 : ℕ) (h1 : a 0 = a1)
  (heven : ∀ n : ℕ, ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ a (2 * n) = a (2 * n - 1) - d)
  (hodd : ∀ n : ℕ, ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ a (2 * n + 1) = a (2 * n) + d) :
  ∀ n : ℕ, a n ≤ 10 * a1 := 
by
  sorry

end sequence_bounded_l203_203837


namespace tan_theta_eq_neg_4_over_3_expression_eval_l203_203926

theorem tan_theta_eq_neg_4_over_3 (θ : ℝ) (h₁ : Real.sin θ = 4 / 5) (h₂ : Real.pi / 2 < θ ∧ θ < Real.pi) :
  Real.tan θ = -4 / 3 :=
sorry

theorem expression_eval (θ : ℝ) (h₁ : Real.sin θ = 4 / 5) (h₂ : Real.pi / 2 < θ ∧ θ < Real.pi) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (3 * Real.sin θ ^ 2 + Real.cos θ ^ 2) = 8 / 25 :=
sorry

end tan_theta_eq_neg_4_over_3_expression_eval_l203_203926


namespace number_of_four_digit_numbers_l203_203126

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l203_203126


namespace expression_value_l203_203778

theorem expression_value {a b c d m : ℝ} (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 1) : 
  (a + b) * c * d - 2014 * m = -2014 ∨ (a + b) * c * d - 2014 * m = 2014 := 
by
  sorry

end expression_value_l203_203778


namespace four_digit_number_count_l203_203136

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l203_203136


namespace number_identification_l203_203738

theorem number_identification (x : ℝ) (h : x ^ 655 / x ^ 650 = 100000) : x = 10 :=
by
  sorry

end number_identification_l203_203738


namespace rational_abs_eq_l203_203340

theorem rational_abs_eq (a : ℚ) (h : |-3 - a| = 3 + |a|) : 0 ≤ a := 
by
  sorry

end rational_abs_eq_l203_203340


namespace correct_average_marks_l203_203786

theorem correct_average_marks :
  let num_students := 40
  let reported_avg := 65
  let incorrect_marks := [100, 85, 15]
  let correct_marks := [20, 50, 55]
  let incorrect_total_sum := num_students * reported_avg
  let wrong_sum := List.sum incorrect_marks
  let correct_sum := List.sum correct_marks
  let correct_total_sum := incorrect_total_sum - wrong_sum + correct_sum
  let correct_avg := (correct_total_sum : ℚ) / num_students
  correct_avg = 63.125 :=
by
  let num_students := 40
  let reported_avg := 65
  let incorrect_marks := [100, 85, 15]
  let correct_marks := [20, 50, 55]
  let incorrect_total_sum := num_students * reported_avg
  let wrong_sum := List.sum incorrect_marks
  let correct_sum := List.sum correct_marks
  let correct_total_sum := incorrect_total_sum - wrong_sum + correct_sum
  let correct_avg := (correct_total_sum : ℚ) / num_students
  sorry

end correct_average_marks_l203_203786


namespace larger_tablet_diagonal_length_l203_203568

theorem larger_tablet_diagonal_length :
  ∀ (d : ℝ), (d^2 / 2 = 25 / 2 + 5.5) → d = 6 :=
by
  intro d
  sorry

end larger_tablet_diagonal_length_l203_203568


namespace color_cartridge_cost_l203_203225

theorem color_cartridge_cost :
  ∃ C : ℝ, 
  (1 * 27) + (3 * C) = 123 ∧ C = 32 :=
by
  sorry

end color_cartridge_cost_l203_203225


namespace geometric_sequence_seventh_term_l203_203564

theorem geometric_sequence_seventh_term (a r : ℝ) (ha : 0 < a) (hr : 0 < r) 
  (h4 : a * r^3 = 16) (h10 : a * r^9 = 2) : 
  a * r^6 = 2 :=
by
  sorry

end geometric_sequence_seventh_term_l203_203564


namespace find_toonies_l203_203488

-- Define the number of coins and their values
variables (L T : ℕ) -- L represents the number of loonies, T represents the number of toonies

-- Define the conditions
def total_coins := L + T = 10
def total_value := 1 * L + 2 * T = 14

-- Define the theorem to be proven
theorem find_toonies (L T : ℕ) (h1 : total_coins L T) (h2 : total_value L T) : T = 4 :=
by
  sorry

end find_toonies_l203_203488


namespace gcd_of_360_and_150_l203_203431

theorem gcd_of_360_and_150 : Nat.gcd 360 150 = 30 := 
by
  sorry

end gcd_of_360_and_150_l203_203431


namespace original_number_divisibility_l203_203016

theorem original_number_divisibility (N : ℤ) : (∃ k : ℤ, N = 9 * k + 3) ↔ (∃ m : ℤ, (N + 3) = 9 * m) := sorry

end original_number_divisibility_l203_203016


namespace expenses_neg_five_given_income_five_l203_203989

theorem expenses_neg_five_given_income_five 
  (income_5 : ℤ)
  (income_5_pos : income_5 = 5)
  (opposite : ∀ (x : ℤ), -x = -1 * x) :
  (expenses_5 : ℤ) (expenses_5 = -5) :=
by {
  sorry
}

end expenses_neg_five_given_income_five_l203_203989


namespace bananas_to_oranges_equivalence_l203_203211

theorem bananas_to_oranges_equivalence :
  (3 / 4 : ℚ) * 16 = 12 ->
  (2 / 5 : ℚ) * 10 = 4 :=
by
  intros h
  sorry

end bananas_to_oranges_equivalence_l203_203211


namespace max_oranges_to_teachers_l203_203027

theorem max_oranges_to_teachers {n r : ℕ} (h1 : n % 8 = r) (h2 : r < 8) : r = 7 :=
sorry

end max_oranges_to_teachers_l203_203027


namespace evaTotalMarksCorrect_l203_203297

-- Definition of marks scored by Eva in each subject across semesters
def evaMathsMarksSecondSemester : Nat := 80
def evaArtsMarksSecondSemester : Nat := 90
def evaScienceMarksSecondSemester : Nat := 90

def evaMathsMarksFirstSemester : Nat := evaMathsMarksSecondSemester + 10
def evaArtsMarksFirstSemester : Nat := evaArtsMarksSecondSemester - 15
def evaScienceMarksFirstSemester : Nat := evaScienceMarksSecondSemester - (evaScienceMarksSecondSemester / 3)

-- Total marks in each semester
def totalMarksFirstSemester : Nat := evaMathsMarksFirstSemester + evaArtsMarksFirstSemester + evaScienceMarksFirstSemester
def totalMarksSecondSemester : Nat := evaMathsMarksSecondSemester + evaArtsMarksSecondSemester + evaScienceMarksSecondSemester

-- Combined total
def evaTotalMarks : Nat := totalMarksFirstSemester + totalMarksSecondSemester

-- Statement to prove
theorem evaTotalMarksCorrect : evaTotalMarks = 485 := 
by
  -- This needs to be proved as per the conditions and calculations above
  sorry

end evaTotalMarksCorrect_l203_203297


namespace average_last_4_matches_l203_203581

theorem average_last_4_matches (avg_10_matches avg_6_matches : ℝ) (matches_10 matches_6 matches_4 : ℕ) :
  avg_10_matches = 38.9 →
  avg_6_matches = 41 →
  matches_10 = 10 →
  matches_6 = 6 →
  matches_4 = 4 →
  (avg_10_matches * matches_10 - avg_6_matches * matches_6) / matches_4 = 35.75 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_last_4_matches_l203_203581


namespace inequality_proof_l203_203368

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c)) ≥ (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) :=
by
  sorry

end inequality_proof_l203_203368


namespace yearly_exports_calculation_l203_203246

variable (Y : Type) 
variable (fruit_exports_total yearly_exports : ℝ)
variable (orange_exports : ℝ := 4.25 * 10^6)
variable (fruit_exports_percent : ℝ := 0.20)
variable (orange_exports_fraction : ℝ := 1/6)

-- The main statement to prove
theorem yearly_exports_calculation
  (h1 : yearly_exports * fruit_exports_percent = fruit_exports_total)
  (h2 : fruit_exports_total * orange_exports_fraction = orange_exports) :
  yearly_exports = 127.5 * 10^6 :=
by
  -- Proof (omitted)
  sorry

end yearly_exports_calculation_l203_203246


namespace polygon_triangle_even_l203_203254

theorem polygon_triangle_even (n m : ℕ) (h : (3 * m - n) % 2 = 0) : (m + n) % 2 = 0 :=
sorry

noncomputable def number_of_distinct_interior_sides (n m : ℕ) : ℕ :=
(3 * m - n) / 2

noncomputable def number_of_distinct_interior_vertices (n m : ℕ) : ℕ :=
(m - n + 2) / 2

end polygon_triangle_even_l203_203254


namespace rational_sum_l203_203345

theorem rational_sum (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : x + y = 7 ∨ x + y = 3 := 
sorry

end rational_sum_l203_203345


namespace son_is_four_times_younger_l203_203605

-- Given Conditions
def son_age : ℕ := 9
def dad_age : ℕ := 36
def age_difference : ℕ := dad_age - son_age -- Ensure the difference in ages

-- The proof problem
theorem son_is_four_times_younger : dad_age / son_age = 4 :=
by
  -- Ensure the conditions are correct and consistent.
  have h1 : dad_age = 36 := rfl
  have h2 : son_age = 9 := rfl
  have h3 : dad_age - son_age = 27 := rfl
  sorry

end son_is_four_times_younger_l203_203605


namespace calculate_dollar_value_l203_203047

def dollar (x y : ℤ) : ℤ := x * (y + 2) + x * y - 5

theorem calculate_dollar_value : dollar 3 (-1) = -5 := by
  sorry

end calculate_dollar_value_l203_203047


namespace average_first_6_numbers_l203_203828

theorem average_first_6_numbers (A : ℕ) (h1 : (13 * 9) = (6 * A + 45 + 6 * 7)) : A = 5 :=
by 
  -- h1 : 117 = (6 * A + 45 + 42),
  -- solving for the value of A by performing algebraic operations will prove it.
  sorry

end average_first_6_numbers_l203_203828


namespace cos_value_l203_203320

theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by sorry

end cos_value_l203_203320


namespace dave_deleted_17_apps_l203_203046

-- Define the initial and final state of Dave's apps
def initial_apps : Nat := 10
def added_apps : Nat := 11
def apps_left : Nat := 4

-- The total number of apps before deletion
def total_apps : Nat := initial_apps + added_apps

-- The expected number of deleted apps
def deleted_apps : Nat := total_apps - apps_left

-- The proof statement
theorem dave_deleted_17_apps : deleted_apps = 17 := by
  -- detailed steps are not required
  sorry

end dave_deleted_17_apps_l203_203046


namespace four_digit_numbers_count_l203_203110

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l203_203110


namespace parabola_distance_ratio_l203_203332

open Real

theorem parabola_distance_ratio (p : ℝ) (M N : ℝ × ℝ)
  (h1 : p = 4)
  (h2 : M.snd ^ 2 = 2 * p * M.fst)
  (h3 : N.snd ^ 2 = 2 * p * N.fst)
  (h4 : (M.snd - 2 * N.snd) * (M.snd + 2 * N.snd) = 48) :
  |M.fst + 2| = 4 * |N.fst + 2| := sorry

end parabola_distance_ratio_l203_203332


namespace fourth_number_in_pascals_triangle_row_15_l203_203676

theorem fourth_number_in_pascals_triangle_row_15 : (Nat.choose 15 3) = 455 :=
by sorry

end fourth_number_in_pascals_triangle_row_15_l203_203676


namespace centroid_distance_l203_203399

theorem centroid_distance
  (a b m : ℝ)
  (h_a_nonneg : 0 ≤ a)
  (h_b_nonneg : 0 ≤ b)
  (h_m_pos : 0 < m) :
  (∃ d : ℝ, d = m * (b + 2 * a) / (3 * (a + b))) :=
by
  sorry

end centroid_distance_l203_203399


namespace incorrect_conversion_D_l203_203235

-- Definition of base conversions as conditions
def binary_to_decimal (b : String) : ℕ := -- Converts binary string to decimal number
  sorry

def octal_to_decimal (o : String) : ℕ := -- Converts octal string to decimal number
  sorry

def decimal_to_base_n (d : ℕ) (n : ℕ) : String := -- Converts decimal number to base-n string
  sorry

-- Given conditions
axiom cond1 : binary_to_decimal "101" = 5
axiom cond2 : octal_to_decimal "27" = 25 -- Note: "27"_base(8) is 2*8 + 7 = 23 in decimal; there's a typo in question's option.
axiom cond3 : decimal_to_base_n 119 6 = "315"
axiom cond4 : decimal_to_base_n 13 2 = "1101" -- Note: correcting from 62 to "1101"_base(2) which is 13

-- Prove the incorrect conversion between number systems
theorem incorrect_conversion_D : decimal_to_base_n 31 4 ≠ "62" :=
  sorry

end incorrect_conversion_D_l203_203235


namespace rational_sum_l203_203346

theorem rational_sum (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : x + y = 7 ∨ x + y = 3 := 
sorry

end rational_sum_l203_203346


namespace number_of_four_digit_numbers_l203_203124

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l203_203124


namespace find_number_l203_203506

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 58) : x = 145 := by
  sorry

end find_number_l203_203506


namespace total_points_correct_l203_203284

-- Define the scores
def Marius (Darius : ℕ) : ℕ := Darius + 3
def Matt (Darius : ℕ) : ℕ := Darius + 5

-- Define the total points function
def total_points (Darius : ℕ) : ℕ :=
  Darius + Marius Darius + Matt Darius

-- Specific value for Darius's score
def Darius_score : ℕ := 10

-- The theorem that proves the total score is 38 given Darius's score
theorem total_points_correct :
  total_points Darius_score = 38 :=
by
  sorry

end total_points_correct_l203_203284


namespace price_reduction_for_2100_yuan_price_reduction_for_max_profit_l203_203029

-- Condition definitions based on the problem statement
def units_sold (x : ℝ) : ℝ := 30 + 2 * x
def profit_per_unit (x : ℝ) : ℝ := 50 - x
def daily_profit (x : ℝ) : ℝ := profit_per_unit x * units_sold x

-- Statement to prove the price reduction for achieving a daily profit of 2100 yuan
theorem price_reduction_for_2100_yuan : ∃ x : ℝ, daily_profit x = 2100 ∧ x = 20 :=
  sorry

-- Statement to prove the price reduction to maximize the daily profit
theorem price_reduction_for_max_profit : ∀ x : ℝ, ∃ y : ℝ, (∀ z : ℝ, daily_profit z ≤ y) ∧ x = 17.5 :=
  sorry

end price_reduction_for_2100_yuan_price_reduction_for_max_profit_l203_203029


namespace number_of_four_digit_numbers_l203_203121

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l203_203121


namespace additional_fertilizer_on_final_day_l203_203255

noncomputable def normal_usage_per_day : ℕ := 2
noncomputable def total_days : ℕ := 9
noncomputable def total_fertilizer_used : ℕ := 22

theorem additional_fertilizer_on_final_day :
  total_fertilizer_used - (normal_usage_per_day * total_days) = 4 := by
  sorry

end additional_fertilizer_on_final_day_l203_203255


namespace find_coordinates_of_point_M_l203_203791

theorem find_coordinates_of_point_M :
  ∃ (M : ℝ × ℝ), 
    (M.1 > 0) ∧ (M.2 < 0) ∧ 
    abs M.2 = 12 ∧ 
    abs M.1 = 4 ∧ 
    M = (4, -12) :=
by
  sorry

end find_coordinates_of_point_M_l203_203791


namespace common_ratio_of_geometric_series_l203_203628

theorem common_ratio_of_geometric_series : ∃ r : ℝ, ∀ n : ℕ, 
  r = (if n = 0 then 2 / 3
       else if n = 1 then (2 / 3) * (2 / 3)
       else if n = 2 then (2 / 3) * (2 / 3) * (2 / 3)
       else sorry)
  ∧ r = 2 / 3 := sorry

end common_ratio_of_geometric_series_l203_203628


namespace inscribed_circle_distance_l203_203724

-- description of the geometry problem
theorem inscribed_circle_distance (r : ℝ) (AB : ℝ):
  r = 4 →
  AB = 4 →
  ∃ d : ℝ, d = 6.4 :=
by
  intros hr hab
  -- skipping proof steps
  let a := 2*r
  let PQ := 2 * r * (Real.sqrt 3 / 2)
  use PQ
  sorry

end inscribed_circle_distance_l203_203724


namespace four_digit_numbers_count_l203_203128

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l203_203128


namespace circumscribed_quadrilateral_arc_sum_l203_203260

theorem circumscribed_quadrilateral_arc_sum 
  (a b c d : ℝ) 
  (h : a + b + c + d = 360) : 
  (1/2 * (b + c + d)) + (1/2 * (a + c + d)) + (1/2 * (a + b + d)) + (1/2 * (a + b + c)) = 540 :=
by
  sorry

end circumscribed_quadrilateral_arc_sum_l203_203260


namespace daragh_initial_bears_l203_203277

variables (initial_bears eden_initial_bears eden_final_bears favorite_bears shared_bears_per_sister : ℕ)
variables (sisters : ℕ)

-- Given conditions
axiom h1 : eden_initial_bears = 10
axiom h2 : eden_final_bears = 14
axiom h3 : favorite_bears = 8
axiom h4 : sisters = 3

-- Derived condition
axiom h5 : shared_bears_per_sister = eden_final_bears - eden_initial_bears
axiom h6 : initial_bears = favorite_bears + (shared_bears_per_sister * sisters)

-- The theorem to prove
theorem daragh_initial_bears : initial_bears = 20 :=
by
  -- Insert proof here
  sorry

end daragh_initial_bears_l203_203277


namespace ratio_of_area_l203_203022
   
   noncomputable def area_of_square (side : ℝ) : ℝ := side * side
   noncomputable def area_of_circle (radius : ℝ) : ℝ := Real.pi * radius * radius
   def radius_of_inscribed_circle (side : ℝ) : ℝ := side / 2
   
   theorem ratio_of_area (side : ℝ) (h : side = 6) : area_of_circle (radius_of_inscribed_circle side) / area_of_square side = Real.pi / 4 :=
   by 
     -- Use the given condition side = 6
     have h1 : radius_of_inscribed_circle side = 3 := by rw [radius_of_inscribed_circle, h]; norm_num
     have h2 : area_of_square side = 36 := by rw [area_of_square, h]; norm_num
     have h3 : area_of_circle 3 = Real.pi * 9 := by rw area_of_circle; norm_num
     -- Calculate the ratio
     rw [h1, h2, h3]
     norm_num -- This simplifies 9 * Real.pi / 36 to Real.pi / 4
   
   
end ratio_of_area_l203_203022


namespace four_digit_numbers_count_l203_203129

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l203_203129


namespace eva_total_marks_l203_203294

theorem eva_total_marks
    (math_score_s2 : ℕ) (arts_score_s2 : ℕ) (science_score_s2 : ℕ)
    (math_diff : ℕ) (arts_diff : ℕ) (science_frac_diff : ℚ)
    (math_score_s2_eq : math_score_s2 = 80)
    (arts_score_s2_eq : arts_score_s2 = 90)
    (science_score_s2_eq : science_score_s2 = 90)
    (math_diff_eq : math_diff = 10)
    (arts_diff_eq : arts_diff = 15)
    (science_frac_diff_eq : science_frac_diff = 1/3) : 
  (math_score_s2 + 10 + (math_score_s2 + math_diff) + 
   (arts_score_s2 + 90 - 15) + (arts_score_s2 + arts_diff) + 
   (science_score_s2 + 90 - (1/3) * 90) + (science_score_s2 + science_score_s2 * 1/3)) = 485 := 
by
  sorry

end eva_total_marks_l203_203294


namespace projectile_reaches_30m_at_2_seconds_l203_203601

theorem projectile_reaches_30m_at_2_seconds:
  ∀ t : ℝ, -5 * t^2 + 25 * t = 30 → t = 2 ∨ t = 3 :=
by
  sorry

end projectile_reaches_30m_at_2_seconds_l203_203601


namespace cos_675_eq_sqrt2_div_2_l203_203274

theorem cos_675_eq_sqrt2_div_2 : Real.cos (675 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by 
  sorry

end cos_675_eq_sqrt2_div_2_l203_203274


namespace find_b_l203_203539

open Real

variables {A B C a b c : ℝ}

theorem find_b 
  (hA : A = π / 4) 
  (h1 : 2 * b * sin B - c * sin C = 2 * a * sin A) 
  (h_area : 1 / 2 * b * c * sin A = 3) : 
  b = 3 := 
sorry

end find_b_l203_203539


namespace width_of_roads_l203_203603

-- Definitions for the conditions
def length_of_lawn := 80 
def breadth_of_lawn := 60 
def total_cost := 5200 
def cost_per_sq_m := 4 

-- Derived condition: total area based on cost
def total_area_by_cost := total_cost / cost_per_sq_m 

-- Statement to prove: width of each road w is 65/7
theorem width_of_roads (w : ℚ) : (80 * w) + (60 * w) = total_area_by_cost → w = 65 / 7 :=
by
  sorry

end width_of_roads_l203_203603


namespace four_digit_numbers_count_l203_203132

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l203_203132


namespace total_points_scored_l203_203280

theorem total_points_scored 
  (darius_score : ℕ) 
  (marius_score : ℕ) 
  (matt_score : ℕ) 
  (h1 : marius_score = darius_score + 3) 
  (h2 : darius_score = matt_score - 5)
  (h3 : darius_score = 10) : darius_score + marius_score + matt_score = 38 :=
by
  sorry

end total_points_scored_l203_203280


namespace fixed_point_of_function_l203_203520

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 1) ∧ ∀ x : ℝ, (x = 1 → a^(x-1) = 1) :=
by
  sorry

end fixed_point_of_function_l203_203520


namespace value_of_y_l203_203668

theorem value_of_y : (∃ y : ℝ, (1 / 3 - 1 / 4 = 4 / y) ∧ y = 48) :=
by
  sorry

end value_of_y_l203_203668


namespace find_c_values_l203_203362

noncomputable def line_intercept_product (c : ℝ) : Prop :=
  let x_intercept := -c / 8
  let y_intercept := -c / 5
  x_intercept * y_intercept = 24

theorem find_c_values :
  ∃ c : ℝ, (line_intercept_product c) ∧ (c = 8 * Real.sqrt 15 ∨ c = -8 * Real.sqrt 15) :=
by
  sorry

end find_c_values_l203_203362


namespace quadratic_root_q_value_l203_203176

theorem quadratic_root_q_value
  (p q : ℝ)
  (h1 : ∃ r : ℝ, r = -3 ∧ 3 * r^2 + p * r + q = 0)
  (h2 : ∃ s : ℝ, -3 + s = -2) :
  q = -9 :=
sorry

end quadratic_root_q_value_l203_203176


namespace translation_result_l203_203412

-- Define the original point M
def M : ℝ × ℝ := (-10, 1)

-- Define the translation on the y-axis by 4 units
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Define the resulting point M1 after translation
def M1 : ℝ × ℝ := translate_y M 4

-- The theorem we want to prove: the coordinates of M1 are (-10, 5)
theorem translation_result : M1 = (-10, 5) :=
by
  -- Proof goes here
  sorry

end translation_result_l203_203412


namespace range_of_a_l203_203072

variable (a x : ℝ)

-- Condition p: ∀ x ∈ [1, 2], x^2 - a ≥ 0
def p : Prop := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

-- Condition q: ∃ x ∈ ℝ, x^2 + 2 * a * x + 2 - a = 0
def q : Prop := ∃ x, x^2 + 2 * a * x + 2 - a = 0

-- The proof goal given p ∧ q: a ≤ -2 or a = 1
theorem range_of_a (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := sorry

end range_of_a_l203_203072


namespace probability_x_plus_y_lt_4_l203_203469

open Set

-- Define the square and the line
def square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }
def line_lt_4 : Set (ℝ × ℝ) := { p | p.1 + p.2 < 4 }

-- The probability to prove
theorem probability_x_plus_y_lt_4 : 
  (volume (square ∩ line_lt_4) / volume square) = 7 / 9 := 
sorry

end probability_x_plus_y_lt_4_l203_203469


namespace four_digit_numbers_count_l203_203161

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l203_203161


namespace starting_even_number_l203_203004

def is_even (n : ℤ) : Prop := n % 2 = 0

def span_covered_by_evens (count : ℤ) : ℤ := count * 2 - 2

theorem starting_even_number
  (count : ℤ)
  (end_num : ℤ)
  (H1 : is_even end_num)
  (H2 : count = 20)
  (H3 : end_num = 55) :
  ∃ start_num, is_even start_num ∧ start_num = end_num - span_covered_by_evens count + 1 := 
sorry

end starting_even_number_l203_203004


namespace probability_C_speaks_first_l203_203670

-- Definitions for students and positions
inductive Student
| A | B | C | D | E

-- Define probability
noncomputable def P (event : Set (List Student)) : ℚ :=
  event.card / 5.factorial

-- Event where student A is not the first and student B is not the last
def eventA : Set (List Student) :=
  { l | l.head ≠ Student.A ∧ l.last ≠ Student.B }

-- Event where student C speaks first
def eventB : Set (List Student) :=
  { l | l.head = Student.C }

-- Combined event: A not first, B not last, and C speaks first
def eventAB : Set (List Student) :=
  { l | l.head = Student.C ∧ l.last ≠ Student.B }

-- Number of permutations of the list
def all_permutations := 
  { l : List Student | l.permutations ∈ (List.permutations [Student.A, Student.B, Student.C, Student.D, Student.E])}

-- Calculate the probability P(AB) and P(A), then show the conditional probability P(B|A)
theorem probability_C_speaks_first :
  P eventAB / P eventA = (3 : ℚ) / 13 :=
by 
  sorry

end probability_C_speaks_first_l203_203670


namespace unique_arrangements_of_MOON_l203_203902

open Nat

theorem unique_arrangements_of_MOON : 
  let word := "MOON"
  let n := 4
  let numM := 1
  let numN := 1
  let numO := 2
  factorial n / (factorial numO * factorial numM * factorial numN) = 12 :=
by
  let word := "MOON"
  let n := 4
  let numM := 1
  let numN := 1
  let numO := 2
  sorry

end unique_arrangements_of_MOON_l203_203902


namespace cake_remaining_after_4_trips_l203_203252

theorem cake_remaining_after_4_trips :
  ∀ (cake_portion_left_after_trip : ℕ → ℚ), 
    cake_portion_left_after_trip 0 = 1 ∧
    (∀ n, cake_portion_left_after_trip (n + 1) = cake_portion_left_after_trip n / 2) →
    cake_portion_left_after_trip 4 = 1 / 16 :=
by
  intros cake_portion_left_after_trip h
  have h0 : cake_portion_left_after_trip 0 = 1 := h.1
  have h1 : ∀ n, cake_portion_left_after_trip (n + 1) = cake_portion_left_after_trip n / 2 := h.2
  sorry

end cake_remaining_after_4_trips_l203_203252


namespace calculate_expression_l203_203922

-- Definitions based on the conditions
def opposite (a b : ℤ) : Prop := a + b = 0
def reciprocal (c d : ℝ) : Prop := c * d = 1
def negative_abs_two (m : ℝ) : Prop := m = -2

-- The main statement to be proved
theorem calculate_expression (a b : ℤ) (c d m : ℝ) 
  (h1 : opposite a b) 
  (h2 : reciprocal c d) 
  (h3 : negative_abs_two m) : 
  m + c * d + a + b + (c * d) ^ 2010 = 0 := 
by
  sorry

end calculate_expression_l203_203922


namespace a_2n_perfect_square_l203_203551

-- Define the sequence a_n following the described recurrence relation.
def a (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 4
  else a (n-1) + a (n-3) + a (n-4)

-- Define the main theorem to prove
theorem a_2n_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k * k := by
  sorry

end a_2n_perfect_square_l203_203551


namespace proof_problem_l203_203661

-- Definitions for the given conditions in the problem
def equations (a x y : ℝ) : Prop :=
(x + 5 * y = 4 - a) ∧ (x - y = 3 * a)

-- The conclusions from the problem
def conclusion1 (a x y : ℝ) : Prop :=
a = 1 → x + y = 4 - a

def conclusion2 (a x y : ℝ) : Prop :=
a = -2 → x = -y

def conclusion3 (a x y : ℝ) : Prop :=
2 * x + 7 * y = 6

def conclusion4 (a x y : ℝ) : Prop :=
x ≤ 1 → y > 4 / 7

-- The main theorem to be proven
theorem proof_problem (a x y : ℝ) :
  equations a x y →
  (¬ conclusion1 a x y ∨ ¬ conclusion2 a x y ∨ ¬ conclusion3 a x y ∨ ¬ conclusion4 a x y) →
  (∃ n : ℕ, n = 2 ∧ ((conclusion1 a x y ∨ conclusion2 a x y ∨ conclusion3 a x y ∨ conclusion4 a x y) → false)) :=
by {
  sorry
}

end proof_problem_l203_203661


namespace four_digit_numbers_count_l203_203131

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l203_203131


namespace sum_of_4n_pos_integers_l203_203530

theorem sum_of_4n_pos_integers (n : ℕ) (Sn : ℕ → ℕ)
  (hSn : ∀ k, Sn k = k * (k + 1) / 2)
  (h_condition : Sn (3 * n) - Sn n = 150) :
  Sn (4 * n) = 300 :=
by {
  sorry
}

end sum_of_4n_pos_integers_l203_203530


namespace expenses_representation_l203_203999

theorem expenses_representation (income_representation : ℤ) (income : ℤ) (expenses : ℤ) :
  income_representation = +5 → income = +5 → expenses = -income → expenses = -5 :=
by
  intro hr hs he
  rw [←hs, he]
  exact hr

end expenses_representation_l203_203999


namespace gcd_2048_2101_eq_1_l203_203232

theorem gcd_2048_2101_eq_1 : Int.gcd 2048 2101 = 1 := sorry

end gcd_2048_2101_eq_1_l203_203232


namespace total_points_l203_203282

theorem total_points (darius_score marius_score matt_score total_points : ℕ) 
    (h1 : darius_score = 10) 
    (h2 : marius_score = darius_score + 3) 
    (h3 : matt_score = darius_score + 5) 
    (h4 : total_points = darius_score + marius_score + matt_score) : 
    total_points = 38 :=
by sorry

end total_points_l203_203282


namespace inverse_composition_has_correct_value_l203_203693

noncomputable def f (x : ℝ) : ℝ := 5 * x + 7
noncomputable def f_inv (x : ℝ) : ℝ := (x - 7) / 5

theorem inverse_composition_has_correct_value : 
  f_inv (f_inv 9) = -33 / 25 := 
by 
  sorry

end inverse_composition_has_correct_value_l203_203693


namespace range_of_a_l203_203526

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    0 < x₁ ∧
    0 < x₂ ∧
    ln x₁ + 2 * exp(x₁^2) = x₁^3 + (a * x₁) / exp(1) ∧
    ln x₂ + 2 * exp(x₂^2) = x₂^3 + (a * x₂) / exp(1)) ->
  a < exp(3) + 1 :=
sorry

end range_of_a_l203_203526


namespace gas_volume_at_25_degrees_l203_203633

theorem gas_volume_at_25_degrees :
  (∀ (T V : ℕ), (T = 40 → V = 30) →
  (∀ (k : ℕ), T = 40 - 5 * k → V = 30 - 6 * k) → 
  (25 = 40 - 5 * 3) → 
  (V = 30 - 6 * 3) → 
  V = 12) := 
by
  sorry

end gas_volume_at_25_degrees_l203_203633


namespace ratio_H_G_l203_203804

theorem ratio_H_G (G H : ℤ) (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 5 → 
    (G / (x + 3) + H / (x * (x - 5)) = (x^2 - 3 * x + 8) / (x^3 + x^2 - 15 * x))) :
    H / G = 64 :=
sorry

end ratio_H_G_l203_203804


namespace four_digit_number_count_l203_203143

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l203_203143


namespace avg_weight_class_l203_203003

-- Definitions based on the conditions
def students_section_A : Nat := 36
def students_section_B : Nat := 24
def avg_weight_section_A : ℝ := 30.0
def avg_weight_section_B : ℝ := 30.0

-- The statement we want to prove
theorem avg_weight_class :
  (avg_weight_section_A * students_section_A + avg_weight_section_B * students_section_B) / (students_section_A + students_section_B) = 30.0 := 
by
  sorry

end avg_weight_class_l203_203003


namespace factor_2210_two_digit_l203_203168

theorem factor_2210_two_digit :
  (∃ (a b : ℕ), a * b = 2210 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99) ∧
  (∃ (c d : ℕ), c * d = 2210 ∧ 10 ≤ c ∧ c ≤ 99 ∧ 10 ≤ d ∧ d ≤ 99) ∧
  (∀ (x y : ℕ), x * y = 2210 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 → 
   ((x = c ∧ y = d) ∨ (x = d ∧ y = c) ∨ (x = a ∧ y = b) ∨ (x = b ∧ y = a))) :=
sorry

end factor_2210_two_digit_l203_203168


namespace two_p_plus_q_l203_203782

theorem two_p_plus_q (p q : ℚ) (h : p / q = 6 / 7) : 2 * p + q = 19 / 7 * q :=
by {
  sorry
}

end two_p_plus_q_l203_203782


namespace gcd_360_150_l203_203422

theorem gcd_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end gcd_360_150_l203_203422


namespace sqrt3_op_sqrt3_l203_203179

def custom_op (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt3_op_sqrt3 : custom_op (Real.sqrt 3) (Real.sqrt 3) = 12 :=
  sorry

end sqrt3_op_sqrt3_l203_203179


namespace only_selected_A_is_20_l203_203787

def cardinality_A (x : ℕ) : ℕ := x
def cardinality_B (x : ℕ) : ℕ := x + 8
def cardinality_union (x : ℕ) : ℕ := 54
def cardinality_intersection (x : ℕ) : ℕ := 6

theorem only_selected_A_is_20 (x : ℕ) (h_total : cardinality_union x = 54) 
  (h_inter : cardinality_intersection x = 6) (h_B : cardinality_B x = x + 8) :
  cardinality_A x - cardinality_intersection x = 20 :=
by
  sorry

end only_selected_A_is_20_l203_203787


namespace mrs_mcpherson_percentage_l203_203965

def total_rent : ℕ := 1200
def mr_mcpherson_amount : ℕ := 840
def mrs_mcpherson_amount : ℕ := total_rent - mr_mcpherson_amount

theorem mrs_mcpherson_percentage : (mrs_mcpherson_amount.toFloat / total_rent.toFloat) * 100 = 30 :=
by
  sorry

end mrs_mcpherson_percentage_l203_203965


namespace time_to_cross_pole_correct_l203_203740

noncomputable def speed_kmph : ℝ := 160 -- Speed of the train in kmph
noncomputable def length_meters : ℝ := 800.064 -- Length of the train in meters

noncomputable def conversion_factor : ℝ := 1000 / 3600 -- Conversion factor from kmph to m/s
noncomputable def speed_mps : ℝ := speed_kmph * conversion_factor -- Speed of the train in m/s

noncomputable def time_to_cross_pole : ℝ := length_meters / speed_mps -- Time to cross the pole

theorem time_to_cross_pole_correct :
  time_to_cross_pole = 800.064 / (160 * (1000 / 3600)) :=
sorry

end time_to_cross_pole_correct_l203_203740


namespace total_points_l203_203281

theorem total_points (darius_score marius_score matt_score total_points : ℕ) 
    (h1 : darius_score = 10) 
    (h2 : marius_score = darius_score + 3) 
    (h3 : matt_score = darius_score + 5) 
    (h4 : total_points = darius_score + marius_score + matt_score) : 
    total_points = 38 :=
by sorry

end total_points_l203_203281


namespace smallest_addition_to_make_multiple_of_5_l203_203858

theorem smallest_addition_to_make_multiple_of_5 : ∃ k : ℕ, k > 0 ∧ (729 + k) % 5 = 0 ∧ k = 1 := sorry

end smallest_addition_to_make_multiple_of_5_l203_203858


namespace tetrahedron_cube_volume_ratio_l203_203714

theorem tetrahedron_cube_volume_ratio (a : ℝ) :
  let V_tetrahedron := (a * Real.sqrt 2)^3 * Real.sqrt 2 / 12
  let V_cube := a^3
  (V_tetrahedron / V_cube) = 1 / 3 :=
by
  sorry

end tetrahedron_cube_volume_ratio_l203_203714


namespace right_triangle_perimeter_l203_203038

theorem right_triangle_perimeter
  (a b : ℝ)
  (h_area : 0.5 * 30 * b = 150)
  (h_leg : a = 30) :
  a + b + Real.sqrt (a^2 + b^2) = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l203_203038


namespace fractional_sum_l203_203367

noncomputable def greatest_integer (t : ℝ) : ℝ := ⌊t⌋
noncomputable def fractional_part (t : ℝ) : ℝ := t - greatest_integer t

theorem fractional_sum (x : ℝ) (h : x^3 + (1/x)^3 = 18) : 
  fractional_part x + fractional_part (1/x) = 1 :=
sorry

end fractional_sum_l203_203367


namespace right_triangle_perimeter_l203_203039

theorem right_triangle_perimeter
  (a b : ℝ)
  (h_area : 0.5 * 30 * b = 150)
  (h_leg : a = 30) :
  a + b + Real.sqrt (a^2 + b^2) = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l203_203039


namespace probability_x_plus_y_lt_4_l203_203473

theorem probability_x_plus_y_lt_4 :
  let square_vertices := {(0,0), (0,3), (3,3), (3,0)} in
  let point_in_square (x y : ℝ) := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 in
  let probability (A : ℝ) (B : ℝ) := A / B in
  ∃ (P : ℝ), P = probability 7 9 ∧
             P = (measure (λ (x y : ℝ), point_in_square x y ∧ x + y < 4)) / (measure (λ (x y : ℝ), point_in_square x y)) :=
sorry

end probability_x_plus_y_lt_4_l203_203473


namespace div_30_div_510_div_66_div_large_l203_203718

theorem div_30 (a : ℤ) : 30 ∣ (a^5 - a) := 
  sorry  

theorem div_510 (a : ℤ) : 510 ∣ (a^17 - a) := 
  sorry

theorem div_66 (a : ℤ) : 66 ∣ (a^11 - a) := 
  sorry

theorem div_large (a : ℤ) : (2 * 3 * 5 * 7 * 13 * 19 * 37 * 73) ∣ (a^73 - a) := 
  sorry  

end div_30_div_510_div_66_div_large_l203_203718


namespace coordinates_of_point_P_l203_203653

theorem coordinates_of_point_P {x y : ℝ} (hx : |x| = 2) (hy : y = 1 ∨ y = -1) (hxy : x < 0 ∧ y > 0) : 
  (x, y) = (-2, 1) := 
by 
  sorry

end coordinates_of_point_P_l203_203653


namespace triplet_A_sums_to_2_triplet_B_sums_to_2_triplet_C_sums_to_2_l203_203436

theorem triplet_A_sums_to_2 : (1/4 + 1/4 + 3/2 = 2) := by
  sorry

theorem triplet_B_sums_to_2 : (3 + -1 + 0 = 2) := by
  sorry

theorem triplet_C_sums_to_2 : (0.2 + 0.7 + 1.1 = 2) := by
  sorry

end triplet_A_sums_to_2_triplet_B_sums_to_2_triplet_C_sums_to_2_l203_203436


namespace four_digit_numbers_count_l203_203163

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l203_203163


namespace n_power_four_plus_sixtyfour_power_n_composite_l203_203303

theorem n_power_four_plus_sixtyfour_power_n_composite (n : ℕ) : ∃ m k, m * k = n^4 + 64^n ∧ m > 1 ∧ k > 1 :=
by
  sorry

end n_power_four_plus_sixtyfour_power_n_composite_l203_203303


namespace speed_of_man_in_still_water_l203_203874

-- Definition of the conditions
def effective_downstream_speed (v_m v_c : ℝ) : Prop := (v_m + v_c) = 10
def effective_upstream_speed (v_m v_c : ℝ) : Prop := (v_m - v_c) = 11.25

-- The proof problem statement
theorem speed_of_man_in_still_water (v_m v_c : ℝ) 
  (h1 : effective_downstream_speed v_m v_c)
  (h2 : effective_upstream_speed v_m v_c)
  : v_m = 10.625 :=
sorry

end speed_of_man_in_still_water_l203_203874


namespace teachers_can_sit_in_middle_l203_203317

-- Definitions for the conditions
def num_students : ℕ := 4
def num_teachers : ℕ := 3
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def permutations (n r : ℕ) : ℕ := factorial n / factorial (n - r)

-- Definition statements
def num_ways_teachers : ℕ := permutations num_teachers num_teachers
def num_ways_students : ℕ := permutations num_students num_students

-- Main theorem statement
theorem teachers_can_sit_in_middle : num_ways_teachers * num_ways_students = 144 := by
  -- Calculation goes here but is omitted for brevity
  sorry

end teachers_can_sit_in_middle_l203_203317


namespace log_six_two_l203_203319

noncomputable def log_six (x : ℝ) : ℝ := Real.log x / Real.log 6

theorem log_six_two (a : ℝ) (h : log_six 3 = a) : log_six 2 = 1 - a :=
by
  sorry

end log_six_two_l203_203319


namespace four_digit_numbers_count_l203_203167

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l203_203167


namespace milk_per_cow_per_day_l203_203590

-- Define the conditions
def num_cows := 52
def weekly_milk_production := 364000 -- ounces

-- State the theorem
theorem milk_per_cow_per_day :
  (weekly_milk_production / 7 / num_cows) = 1000 := 
by
  -- Here we would include the proof, so we use sorry as placeholder
  sorry

end milk_per_cow_per_day_l203_203590


namespace four_digit_numbers_count_eq_l203_203089

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l203_203089


namespace xiao_ming_reading_plan_l203_203439

-- Define the number of pages in the book
def total_pages : Nat := 72

-- Define the total number of days to finish the book
def total_days : Nat := 10

-- Define the number of pages read per day for the first two days
def pages_first_two_days : Nat := 5

-- Define the variable x to represent the number of pages read per day for the remaining days
variable (x : Nat)

-- Define the inequality representing the reading plan
def reading_inequality (x : Nat) : Prop :=
  10 + 8 * x ≥ total_pages

-- The statement to be proved
theorem xiao_ming_reading_plan (x : Nat) : reading_inequality x := sorry

end xiao_ming_reading_plan_l203_203439


namespace number_of_four_digit_numbers_l203_203120

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l203_203120


namespace dog_farthest_distance_l203_203968

/-- 
Given a dog tied to a post at the point (3,4), a 15 meter long rope, and a wall from (5,4) to (5,9), 
prove that the farthest distance the dog can travel from the origin (0,0) is 20 meters.
-/
theorem dog_farthest_distance (post : ℝ × ℝ) (rope_length : ℝ) (wall_start wall_end origin : ℝ × ℝ)
  (h_post : post = (3,4))
  (h_rope_length : rope_length = 15)
  (h_wall_start : wall_start = (5,4))
  (h_wall_end : wall_end = (5,9))
  (h_origin : origin = (0,0)) :
  ∃ farthest_distance : ℝ, farthest_distance = 20 :=
by
  sorry

end dog_farthest_distance_l203_203968


namespace value_of_n_l203_203446

-- Definitions of the question and conditions
def is_3_digit_integer (x : ℕ) : Prop := 100 ≤ x ∧ x < 1000
def not_divisible_by (x : ℕ) (d : ℕ) : Prop := ¬ (d ∣ x)

def problem (m n : ℕ) : Prop :=
  lcm m n = 690 ∧ is_3_digit_integer n ∧ not_divisible_by n 3 ∧ not_divisible_by m 2

-- The theorem to prove
theorem value_of_n {m n : ℕ} (h : problem m n) : n = 230 :=
sorry

end value_of_n_l203_203446


namespace area_ratio_problem_l203_203358

theorem area_ratio_problem
  (A B C : ℝ) -- Areas of the corresponding regions
  (m n : ℕ)  -- Given ratios
  (PQR_is_right_triangle : true)  -- PQR is a right-angled triangle (placeholder condition)
  (RSTU_is_rectangle : true)  -- RSTU is a rectangle (placeholder condition)
  (ratio_A_B : A / B = m / 2)  -- Ratio condition 1
  (ratio_A_C : A / C = n / 1)  -- Ratio condition 2
  (PTS_sim_TQU_sim_PQR : true)  -- Similar triangles (placeholder condition)
  : n = 9 := 
sorry

end area_ratio_problem_l203_203358


namespace four_digit_numbers_count_l203_203109

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l203_203109


namespace surface_area_to_lateral_surface_ratio_cone_l203_203669

noncomputable def cone_surface_lateral_area_ratio : Prop :=
  let radius : ℝ := 1
  let theta : ℝ := (2 * Real.pi) / 3
  let lateral_surface_area := Real.pi * radius^2 * (theta / (2 * Real.pi))
  let base_radius := (2 * Real.pi * radius * (theta / (2 * Real.pi))) / (2 * Real.pi)
  let base_area := Real.pi * base_radius^2
  let surface_area := lateral_surface_area + base_area
  (surface_area / lateral_surface_area) = (4 / 3)

theorem surface_area_to_lateral_surface_ratio_cone :
  cone_surface_lateral_area_ratio :=
  by
  sorry

end surface_area_to_lateral_surface_ratio_cone_l203_203669


namespace inequality_solution_set_l203_203569

theorem inequality_solution_set (x : ℝ) :
  (2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2 := by
sorry

end inequality_solution_set_l203_203569


namespace triangle_height_from_area_l203_203559

theorem triangle_height_from_area {A b h : ℝ} (hA : A = 36) (hb : b = 8) 
    (formula : A = 1 / 2 * b * h) : h = 9 := 
by
  sorry

end triangle_height_from_area_l203_203559


namespace problem_l203_203327

theorem problem:
  ∀ k : Real, (2 - Real.sqrt 2 / 2 ≤ k ∧ k ≤ 2 + Real.sqrt 2 / 2) →
  (11 - 6 * Real.sqrt 2) / 4 ≤ (3 / 2 * (k - 1)^2 + 1 / 2) ∧ 
  (3 / 2 * (k - 1)^2 + 1 / 2 ≤ (11 + 6 * Real.sqrt 2) / 4) :=
by
  intros k hk
  sorry

end problem_l203_203327


namespace range_of_a_l203_203507

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 3| - |x + 2| ≥ Real.log a / Real.log 2) ↔ (0 < a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l203_203507


namespace car_travel_distance_l203_203523

noncomputable def car_distance_in_30_minutes : ℝ := 
  let train_speed : ℝ := 96
  let car_speed : ℝ := (5 / 8) * train_speed
  let travel_time : ℝ := 0.5  -- 30 minutes is 0.5 hours
  car_speed * travel_time

theorem car_travel_distance : car_distance_in_30_minutes = 30 := by
  sorry

end car_travel_distance_l203_203523


namespace bubble_gum_cost_l203_203408

-- Define the conditions
def total_cost : ℕ := 2448
def number_of_pieces : ℕ := 136

-- Main theorem to state that each piece of bubble gum costs 18 cents
theorem bubble_gum_cost : total_cost / number_of_pieces = 18 :=
by
  sorry

end bubble_gum_cost_l203_203408


namespace percentage_reduced_l203_203599

theorem percentage_reduced (P : ℝ) (h : (85 * P / 100) - 11 = 23) : P = 40 :=
by 
  sorry

end percentage_reduced_l203_203599


namespace trials_satisfy_inequality_l203_203824

noncomputable def number_of_trials (p : ℝ) (epsilon : ℝ) (confidence : ℝ) : ℕ :=
  ⌈1 / (confidence * epsilon^2 / (p * (1 - p)))⌉₊

theorem trials_satisfy_inequality (p : ℝ) (epsilon : ℝ) (confidence : ℝ) (n : ℕ) :
  p = 0.8 ∧ epsilon = 0.1 ∧ confidence = 0.03 → n >= 534 :=
by
  sorry

end trials_satisfy_inequality_l203_203824


namespace expensive_feed_cost_l203_203224

/-- Tim and Judy mix two kinds of feed for pedigreed dogs. They made 35 pounds of feed worth 0.36 dollars per pound by mixing one kind worth 0.18 dollars per pound with another kind. They used 17 pounds of the cheaper kind in the mix. What is the cost per pound of the more expensive kind of feed? --/
theorem expensive_feed_cost 
  (total_feed : ℝ := 35) 
  (avg_cost : ℝ := 0.36) 
  (cheaper_feed : ℝ := 17) 
  (cheaper_cost : ℝ := 0.18) 
  (total_cost : ℝ := total_feed * avg_cost) 
  (cheaper_total_cost : ℝ := cheaper_feed * cheaper_cost) 
  (expensive_feed : ℝ := total_feed - cheaper_feed) : 
  (total_cost - cheaper_total_cost) / expensive_feed = 0.53 :=
by
  sorry

end expensive_feed_cost_l203_203224


namespace four_digit_number_count_l203_203118

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l203_203118


namespace number_of_four_digit_numbers_l203_203153

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l203_203153


namespace four_digit_numbers_count_l203_203133

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l203_203133


namespace number_of_four_digit_numbers_l203_203122

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l203_203122


namespace both_pumps_drain_lake_l203_203201

theorem both_pumps_drain_lake (T : ℝ) (h₁ : 1 / 9 + 1 / 6 = 5 / 18) : 
  (5 / 18) * T = 1 → T = 18 / 5 := sorry

end both_pumps_drain_lake_l203_203201


namespace count_integers_satisfying_sqrt_condition_l203_203839

theorem count_integers_satisfying_sqrt_condition : 
  let y_conditions := { y : ℝ | 6 > Real.sqrt y ∧ Real.sqrt y > 3 }
  let integer_satisfying_set := { y : ℕ | y ∈ y_conditions }
  integer_satisfying_set.card = 26 :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l203_203839


namespace ping_pong_matches_l203_203514

noncomputable def f (n k : ℕ) : ℕ :=
  Nat.ceil ((n : ℚ) / Nat.ceil ((k : ℚ) / 2))

theorem ping_pong_matches (n k : ℕ) (hn_pos : 0 < n) (hk_le : k ≤ 2 * n - 1) :
  f n k = Nat.ceil ((n : ℚ) / Nat.ceil ((k : ℚ) / 2)) :=
by
  sorry

end ping_pong_matches_l203_203514


namespace jimmy_shoveled_10_driveways_l203_203185

theorem jimmy_shoveled_10_driveways :
  ∀ (cost_candy_bar : ℝ) (num_candy_bars : ℕ)
    (cost_lollipop : ℝ) (num_lollipops : ℕ)
    (fraction_spent : ℝ)
    (charge_per_driveway : ℝ),
    cost_candy_bar = 0.75 →
    num_candy_bars = 2 →
    cost_lollipop = 0.25 →
    num_lollipops = 4 →
    fraction_spent = 1/6 →
    charge_per_driveway = 1.5 →
    let total_spent := (num_candy_bars * cost_candy_bar + num_lollipops * cost_lollipop) in
    let total_earned := total_spent / fraction_spent in
    (total_earned / charge_per_driveway) = 10 := sorry

end jimmy_shoveled_10_driveways_l203_203185


namespace tangent_circles_pass_through_homothety_center_l203_203275

-- Define the necessary structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def is_tangent_to_line (ω : Circle) (L : ℝ → ℝ) : Prop :=
  sorry -- Definition of tangency to a line

def is_tangent_to_circle (ω : Circle) (C : Circle) : Prop :=
  sorry -- Definition of tangency to another circle

theorem tangent_circles_pass_through_homothety_center
  (L : ℝ → ℝ) (C : Circle) (ω : Circle)
  (H_ext H_int : ℝ × ℝ)
  (H_tangency_line : is_tangent_to_line ω L)
  (H_tangency_circle : is_tangent_to_circle ω C) :
  ∃ P Q : ℝ × ℝ, 
    (is_tangent_to_line ω L ∧ is_tangent_to_circle ω C) →
    (P = Q ∧ (P = H_ext ∨ P = H_int)) :=
by
  sorry

end tangent_circles_pass_through_homothety_center_l203_203275


namespace ratio_of_larger_to_smaller_l203_203000

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) (h3 : 0 < x) (h4 : 0 < y) : x / y = 4 / 3 := by
  sorry

end ratio_of_larger_to_smaller_l203_203000


namespace minimum_rectangles_needed_l203_203856

def type1_corners := 12
def type2_corners := 12
def group_size := 3

theorem minimum_rectangles_needed (cover_type1: ℕ) (cover_type2: ℕ)
  (type1_corners coverable_by_one: ℕ) (type2_groups_num: ℕ) :
  type1_corners = 12 → type2_corners = 12 → type2_groups_num = 4 →
  group_size = 3 → cover_type1 + cover_type2 = 12 :=
by
  intros h1 h2 h3 h4 
  sorry

end minimum_rectangles_needed_l203_203856


namespace monthly_payment_l203_203956

theorem monthly_payment (price : ℝ) (discount_rate : ℝ) (down_payment : ℝ) (months : ℕ) (monthly_payment : ℝ) :
  price = 480 ∧ discount_rate = 0.05 ∧ down_payment = 150 ∧ months = 3 ∧
  monthly_payment = (price * (1 - discount_rate) - down_payment) / months →
  monthly_payment = 102 :=
by
  sorry

end monthly_payment_l203_203956


namespace equilibrium_table_n_max_l203_203954

theorem equilibrium_table_n_max (table : Fin 2010 → Fin 2010 → ℕ) :
  (∃ n, ∀ (i j k l : Fin 2010),
      table i j + table k l = table i l + table k j ∧
      ∀ m ≤ n, (m = 0 ∨ m = 1)
  ) → n = 1 ∧ table (Fin.mk 0 (by norm_num)) (Fin.mk 0 (by norm_num)) = 2 :=
by
  sorry

end equilibrium_table_n_max_l203_203954


namespace infinite_geometric_series_sum_l203_203904

-- Definition of the infinite geometric series with given first term and common ratio
def infinite_geometric_series (a : ℚ) (r : ℚ) : ℚ := a / (1 - r)

-- Problem statement
theorem infinite_geometric_series_sum :
  infinite_geometric_series (5 / 3) (-2 / 9) = 15 / 11 :=
sorry

end infinite_geometric_series_sum_l203_203904


namespace gcd_360_150_l203_203417

theorem gcd_360_150 : Int.gcd 360 150 = 30 := by
  have h360 : 360 = 2^3 * 3^2 * 5 := by
    ring
  have h150 : 150 = 2 * 3 * 5^2 := by
    ring
  rw [h360, h150]
  sorry

end gcd_360_150_l203_203417


namespace total_birds_on_fence_l203_203222

-- Definitions for the problem conditions
def initial_birds : ℕ := 12
def new_birds : ℕ := 8

-- Theorem to state that the total number of birds on the fence is 20
theorem total_birds_on_fence : initial_birds + new_birds = 20 :=
by
  -- Skip the proof as required
  sorry

end total_birds_on_fence_l203_203222


namespace lines_proportional_l203_203940

variables {x y : ℝ} {p q : ℝ}

theorem lines_proportional (h1 : p * x + 2 * y = 7) (h2 : 3 * x + q * y = 5) :
  p = 21 / 5 := 
sorry

end lines_proportional_l203_203940


namespace correct_average_is_19_l203_203445

-- Definitions
def incorrect_avg : ℕ := 16
def num_values : ℕ := 10
def incorrect_reading : ℕ := 25
def correct_reading : ℕ := 55

-- Theorem to prove
theorem correct_average_is_19 :
  ((incorrect_avg * num_values - incorrect_reading + correct_reading) / num_values) = 19 :=
by
  sorry

end correct_average_is_19_l203_203445


namespace antonella_toonies_l203_203489

theorem antonella_toonies (L T : ℕ) (h1 : L + T = 10) (h2 : L + 2 * T = 14) : T = 4 :=
by
  sorry

end antonella_toonies_l203_203489


namespace max_area_guaranteed_l203_203868

noncomputable def max_rectangle_area (board_size : ℕ) (removed_cells : ℕ) : ℕ :=
  if board_size = 8 ∧ removed_cells = 8 then 8 else 0

theorem max_area_guaranteed :
  max_rectangle_area 8 8 = 8 :=
by
  -- Proof logic goes here
  sorry

end max_area_guaranteed_l203_203868


namespace prob_two_packs_tablets_at_10am_dec31_l203_203947
noncomputable def prob_two_packs_tablets (n : ℕ) : ℝ :=
  let numer := (2^n - 1)
  let denom := 2^(n-1) * n
  numer / denom

theorem prob_two_packs_tablets_at_10am_dec31 :
  prob_two_packs_tablets 10 = 1023 / 5120 := by
  sorry

end prob_two_packs_tablets_at_10am_dec31_l203_203947


namespace value_of_d_l203_203508

theorem value_of_d (d y : ℤ) (h₁ : y = 2) (h₂ : 5 * y^2 - 8 * y + 55 = d) : d = 59 := by
  sorry

end value_of_d_l203_203508


namespace johns_sister_age_l203_203365

variable (j d s : ℝ)

theorem johns_sister_age 
  (h1 : j = d - 15)
  (h2 : j + d = 100)
  (h3 : s = j - 5) :
  s = 37.5 := 
sorry

end johns_sister_age_l203_203365


namespace sum_of_smallest_and_largest_prime_l203_203183

def primes_between (a b : ℕ) : List ℕ := List.filter Nat.Prime (List.range' a (b - a + 1))

def smallest_prime_in_range (a b : ℕ) : ℕ :=
  match primes_between a b with
  | [] => 0
  | h::t => h

def largest_prime_in_range (a b : ℕ) : ℕ :=
  match List.reverse (primes_between a b) with
  | [] => 0
  | h::t => h

theorem sum_of_smallest_and_largest_prime : smallest_prime_in_range 1 50 + largest_prime_in_range 1 50 = 49 := 
by
  -- Let the Lean prover take over from here
  sorry

end sum_of_smallest_and_largest_prime_l203_203183


namespace value_of_k_l203_203920

theorem value_of_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (h3 : k ≠ 1) (h4 : 2 * a + b = 2 * a * b) : k = 3 * Real.sqrt 2 :=
by
  sorry

end value_of_k_l203_203920


namespace four_digit_numbers_count_eq_l203_203095

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l203_203095


namespace intersection_A_B_l203_203959

def A : Set ℤ := {-2, 0, 1, 2}
def B : Set ℤ := { x | -2 ≤ x ∧ x ≤ 1 }

theorem intersection_A_B : A ∩ B = {-2, 0, 1} := by
  sorry

end intersection_A_B_l203_203959


namespace triangle_perimeter_l203_203036

theorem triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) 
  (h1 : area = 150)
  (h2 : leg1 = 30)
  (h3 : 0 < leg2)
  (h4 : hypotenuse = (leg1^2 + leg2^2).sqrt)
  (hArea : area = 0.5 * leg1 * leg2)
  : hypotenuse = 10 * Real.sqrt 10 ∧ leg2 = 10 ∧ (leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10) := 
by
  sorry

end triangle_perimeter_l203_203036


namespace forces_angle_result_l203_203573

noncomputable def forces_angle_condition (p1 p2 p : ℝ) (α : ℝ) : Prop :=
  p^2 = p1 * p2

noncomputable def angle_condition_range (p1 p2 : ℝ) : Prop :=
  (3 - Real.sqrt 5) / 2 ≤ p1 / p2 ∧ p1 / p2 ≤ (3 + Real.sqrt 5) / 2

theorem forces_angle_result (p1 p2 p α : ℝ) (h : forces_angle_condition p1 p2 p α) :
  120 * π / 180 ≤ α ∧ α ≤ 120 * π / 180 ∧ (angle_condition_range p1 p2) := 
sorry

end forces_angle_result_l203_203573


namespace smallest_number_of_coins_l203_203610

theorem smallest_number_of_coins (d q : ℕ) (h₁ : 10 * d + 25 * q = 265) (h₂ : d > q) :
  d + q = 16 :=
sorry

end smallest_number_of_coins_l203_203610


namespace gcd_360_150_l203_203420

theorem gcd_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end gcd_360_150_l203_203420


namespace unique_solution_abs_eq_l203_203287

theorem unique_solution_abs_eq : ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| :=
by
  sorry

end unique_solution_abs_eq_l203_203287


namespace gcd_360_150_l203_203423

theorem gcd_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end gcd_360_150_l203_203423


namespace range_of_a_l203_203867

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (|x-2| + |x+3| < a) → false) → a ≤ 5 :=
sorry

end range_of_a_l203_203867


namespace exists_x1_x2_l203_203805

noncomputable def f (a x : ℝ) := a * x + Real.log x

theorem exists_x1_x2 (a : ℝ) (h : a < 0) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ f a x1 ≥ f a x2 :=
by
  sorry

end exists_x1_x2_l203_203805


namespace four_digit_numbers_count_eq_l203_203093

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l203_203093


namespace four_digit_numbers_count_eq_l203_203092

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l203_203092


namespace no_real_roots_of_quadratic_l203_203951

-- Given an arithmetic sequence 
variable {a : ℕ → ℝ}

-- The conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m k, m = n + k → a (m + 1) - a m = a (n + 1) - a n

def condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 5 + a 8 = 9

-- Lean 4 statement for the proof problem
theorem no_real_roots_of_quadratic (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : condition a) :
  let b := a 4 + a 6
  ∃ Δ, Δ = b ^ 2 - 4 * 10 ∧ Δ < 0 :=
by
  sorry

end no_real_roots_of_quadratic_l203_203951


namespace avg_speed_trip_l203_203732

noncomputable def distance_travelled (speed time : ℕ) : ℕ := speed * time

noncomputable def average_speed (total_distance total_time : ℕ) : ℕ := total_distance / total_time

theorem avg_speed_trip :
  let first_leg_speed := 75
  let first_leg_time := 4
  let second_leg_speed := 60
  let second_leg_time := 2
  let total_time := first_leg_time + second_leg_time
  let first_leg_distance := distance_travelled first_leg_speed first_leg_time
  let second_leg_distance := distance_travelled second_leg_speed second_leg_time
  let total_distance := first_leg_distance + second_leg_distance
  average_speed total_distance total_time = 70 :=
by
  sorry

end avg_speed_trip_l203_203732


namespace money_left_in_wallet_l203_203264

def olivia_initial_money : ℕ := 54
def olivia_spent_money : ℕ := 25

theorem money_left_in_wallet : olivia_initial_money - olivia_spent_money = 29 :=
by
  sorry

end money_left_in_wallet_l203_203264


namespace min_seats_occupied_l203_203536

theorem min_seats_occupied (n : ℕ) (h : n = 150) : ∃ k : ℕ, k = 37 ∧ ∀ m : ℕ, m > k → ∃ i : ℕ, i < k ∧ m - k ≥ 2 := sorry

end min_seats_occupied_l203_203536


namespace cost_per_foot_l203_203433

theorem cost_per_foot (area : ℕ) (total_cost : ℕ) (side_length : ℕ) (perimeter : ℕ) (cost_per_foot : ℕ) :
  area = 289 → total_cost = 3944 → side_length = Nat.sqrt 289 → perimeter = 4 * 17 →
  cost_per_foot = total_cost / perimeter → cost_per_foot = 58 :=
by
  intros
  sorry

end cost_per_foot_l203_203433


namespace count_four_digit_numbers_l203_203103

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l203_203103


namespace number_of_ways_to_choose_a_pair_of_socks_same_color_l203_203667

theorem number_of_ways_to_choose_a_pair_of_socks_same_color
  (white black red green : ℕ) 
  (total_socks : ℕ)
  (h1 : white = 5)
  (h2 : black = 6)
  (h3 : red = 3)
  (h4 : green = 2)
  (h5 : total_socks = 16) :
  (nat.choose white 2) + (nat.choose black 2) + (nat.choose red 2) + (nat.choose green 2) = 29 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end number_of_ways_to_choose_a_pair_of_socks_same_color_l203_203667


namespace factor_2210_two_digit_l203_203169

theorem factor_2210_two_digit :
  (∃ (a b : ℕ), a * b = 2210 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99) ∧
  (∃ (c d : ℕ), c * d = 2210 ∧ 10 ≤ c ∧ c ≤ 99 ∧ 10 ≤ d ∧ d ≤ 99) ∧
  (∀ (x y : ℕ), x * y = 2210 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 → 
   ((x = c ∧ y = d) ∨ (x = d ∧ y = c) ∨ (x = a ∧ y = b) ∨ (x = b ∧ y = a))) :=
sorry

end factor_2210_two_digit_l203_203169


namespace simultaneous_equations_solution_exists_l203_203760

theorem simultaneous_equations_solution_exists (m : ℝ) :
  ∃ x y : ℝ, y = 3 * m * x + 2 ∧ y = (3 * m - 2) * x + 5 :=
by
  sorry

end simultaneous_equations_solution_exists_l203_203760


namespace exist_positive_int_for_arithmetic_mean_of_divisors_l203_203958

theorem exist_positive_int_for_arithmetic_mean_of_divisors
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_distinct : p ≠ q) :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 
  (∃ k : ℕ, k * (a + 1) * (b + 1) = (p^(a+1) - 1) / (p - 1) * (q^(b+1) - 1) / (q - 1)) :=
sorry

end exist_positive_int_for_arithmetic_mean_of_divisors_l203_203958


namespace StepaMultiplication_l203_203400

theorem StepaMultiplication {a : ℕ} (h1 : Grisha's_answer = (3 / 2) ^ 4 * a)
  (h2 : Grisha's_answer = 81) :
  (∃ (m n : ℕ), m * n = (3 / 2) ^ 3 * a ∧ m < 10 ∧ n < 10) :=
by
  sorry

end StepaMultiplication_l203_203400


namespace floor_S_value_l203_203960

noncomputable def floor_S (a b c d : ℝ) : ℝ :=
  a + b + c + d

theorem floor_S_value (a b c d : ℝ) 
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (h_sum_sq : a^2 + b^2 = 2016 ∧ c^2 + d^2 = 2016)
  (h_product : a * c = 1008 ∧ b * d = 1008) :
  ⌊floor_S a b c d⌋ = 117 :=
by
  sorry

end floor_S_value_l203_203960


namespace initial_amount_correct_l203_203612

noncomputable def initial_amount (A R T : ℝ) : ℝ :=
  A / (1 + (R * T) / 100)

theorem initial_amount_correct :
  initial_amount 2000 3.571428571428571 4 = 1750 :=
by
  sorry

end initial_amount_correct_l203_203612


namespace count_four_digit_numbers_l203_203150

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l203_203150


namespace men_work_days_l203_203028

theorem men_work_days (M : ℕ) (W : ℕ) (h : W / (M * 40) = W / ((M - 5) * 50)) : M = 25 :=
by
  -- Will add the proof later
  sorry

end men_work_days_l203_203028


namespace pascal_triangle_45th_number_l203_203012

theorem pascal_triangle_45th_number :
  let row := List.range (46 + 1) in
  row.nth 44 = some 1035 :=
by
  let row := List.range (46 + 1)
  have binom_46_2 : nat.binom 46 2 = 1035 := by
    -- Calculations for binomials can be validated here
    calc
      nat.binom 46 2 = 46 * 45 / (2 * 1) : by norm_num
      _ = 1035 : by norm_num
  show row.nth 44 = some (nat.binom 46 2) from by
    rw binom_46_2
    simp only [List.nth_range, option.some_eq_coe, nat.lt_succ_iff, nat.le_refl]
  sorry -- Additional reasoning if necessary

end pascal_triangle_45th_number_l203_203012


namespace four_digit_numbers_count_l203_203166

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l203_203166


namespace antonella_toonies_l203_203490

theorem antonella_toonies (L T : ℕ) (h1 : L + T = 10) (h2 : L + 2 * T = 14) : T = 4 :=
by
  sorry

end antonella_toonies_l203_203490


namespace odd_function_f_a_zero_l203_203518

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + (a + 1) * Real.cos x + x

theorem odd_function_f_a_zero (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : f a a = 0 := 
sorry

end odd_function_f_a_zero_l203_203518


namespace common_ratio_geometric_series_l203_203630

theorem common_ratio_geometric_series :
  let a := 2 / 3
  let b := 4 / 9
  let c := 8 / 27
  (b / a = 2 / 3) ∧ (c / b = 2 / 3) → 
  ∃ r : ℚ, r = 2 / 3 ∧ ∀ n : ℕ, (a * r^n) = (a * (2 / 3)^n) :=
by
  sorry

end common_ratio_geometric_series_l203_203630


namespace four_digit_numbers_count_l203_203162

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l203_203162


namespace solve_equation_l203_203691

def equation (x : ℝ) := (x / (x - 2)) + (2 / (x^2 - 4)) = 1

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) : 
  equation x ↔ x = -3 :=
by
  sorry

end solve_equation_l203_203691


namespace symmetric_about_line_l203_203042

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x - 2)
noncomputable def g (x a : ℝ) : ℝ := f (x + a)

theorem symmetric_about_line (a : ℝ) : (∀ x, g x a = x + 1) ↔ a = 0 :=
by sorry

end symmetric_about_line_l203_203042


namespace f_11_5_equals_neg_1_l203_203243

-- Define the function f with the given properties
axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom periodic_function (f : ℝ → ℝ) : ∀ x, f (x + 2) = f x
axiom f_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x

-- State the theorem to be proved
theorem f_11_5_equals_neg_1 (f : ℝ → ℝ) 
  (odd_f : ∀ x, f (-x) = -f x)
  (periodic_f : ∀ x, f (x + 2) = f x)
  (f_int : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f (11.5) = -1 :=
sorry

end f_11_5_equals_neg_1_l203_203243


namespace optimal_discount_savings_l203_203461

theorem optimal_discount_savings : 
  let total_amount := 15000
  let discount1 := 0.30
  let discount2 := 0.15
  let single_discount := 0.40
  let two_successive_discounts := total_amount * (1 - discount1) * (1 - discount2)
  let one_single_discount := total_amount * (1 - single_discount)
  one_single_discount - two_successive_discounts = 75 :=
by
  sorry

end optimal_discount_savings_l203_203461


namespace initial_books_donations_l203_203708

variable {X : ℕ} -- Initial number of book donations

def books_donated_during_week := 10 * 5
def books_borrowed := 140
def books_remaining := 210

theorem initial_books_donations :
  X + books_donated_during_week - books_borrowed = books_remaining → X = 300 :=
by
  intro h
  sorry

end initial_books_donations_l203_203708


namespace Andy_is_late_l203_203886

def school_start_time : Nat := 8 * 60 -- in minutes (8:00 AM)
def normal_travel_time : Nat := 30 -- in minutes
def delay_red_lights : Nat := 4 * 3 -- in minutes (4 red lights * 3 minutes each)
def delay_construction : Nat := 10 -- in minutes
def delay_detour_accident : Nat := 7 -- in minutes
def delay_store_stop : Nat := 5 -- in minutes
def delay_searching_store : Nat := 2 -- in minutes
def delay_traffic : Nat := 15 -- in minutes
def delay_neighbor_help : Nat := 6 -- in minutes
def delay_closed_road : Nat := 8 -- in minutes
def all_delays : Nat := delay_red_lights + delay_construction + delay_detour_accident + delay_store_stop + delay_searching_store + delay_traffic + delay_neighbor_help + delay_closed_road
def departure_time : Nat := 7 * 60 + 15 -- in minutes (7:15 AM)

def arrival_time : Nat := departure_time + normal_travel_time + all_delays
def late_minutes : Nat := arrival_time - school_start_time

theorem Andy_is_late : late_minutes = 50 := by
  sorry

end Andy_is_late_l203_203886


namespace expression_for_f_when_x_lt_0_l203_203322

noncomputable section

variable (f : ℝ → ℝ)

theorem expression_for_f_when_x_lt_0
  (hf_neg : ∀ x : ℝ, f (-x) = -f x)
  (hf_pos : ∀ x : ℝ, x > 0 → f x = x * abs (x - 2)) :
  ∀ x : ℝ, x < 0 → f x = x * abs (x + 2) :=
by
  sorry

end expression_for_f_when_x_lt_0_l203_203322


namespace sum_of_squares_remainder_l203_203570

theorem sum_of_squares_remainder (n : ℕ) : 
  ((n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2) % 3 = 2 :=
by
  sorry

end sum_of_squares_remainder_l203_203570


namespace total_points_scored_l203_203279

theorem total_points_scored 
  (darius_score : ℕ) 
  (marius_score : ℕ) 
  (matt_score : ℕ) 
  (h1 : marius_score = darius_score + 3) 
  (h2 : darius_score = matt_score - 5)
  (h3 : darius_score = 10) : darius_score + marius_score + matt_score = 38 :=
by
  sorry

end total_points_scored_l203_203279


namespace parallel_vectors_l203_203521

theorem parallel_vectors (m : ℝ) : (m = 1) ↔ (∃ k : ℝ, (m, 1) = k • (1, m)) := sorry

end parallel_vectors_l203_203521


namespace range_of_m_l203_203565

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) m, 1 ≤ f x ∧ f x ≤ 10) ↔ 2 ≤ m ∧ m ≤ 5 := 
by
  sorry

end range_of_m_l203_203565


namespace find_second_group_of_men_l203_203026

noncomputable def work_rate_of_man := ℝ
noncomputable def work_rate_of_woman := ℝ

variables (m w : ℝ)

-- Condition 1: 3 men and 8 women complete the task in the same time as x men and 2 women.
axiom condition1 (x : ℝ) : 3 * m + 8 * w = x * m + 2 * w

-- Condition 2: 2 men and 3 women complete half the task in the same time as 3 men and 8 women completing the whole task.
axiom condition2 : 2 * m + 3 * w = 0.5 * (3 * m + 8 * w)

theorem find_second_group_of_men (x : ℝ) (m w : ℝ) (h1 : 0.5 * m = w)
  (h2 : 3 * m + 8 * w = x * m + 2 * w) : x = 6 :=
by {
  sorry
}

end find_second_group_of_men_l203_203026


namespace find_diminished_value_l203_203219

theorem find_diminished_value :
  ∃ (x : ℕ), 1015 - x = Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 12 16) 18) 21) 28 :=
by
  use 7
  simp
  unfold Nat.lcm
  sorry

end find_diminished_value_l203_203219


namespace part1_part2_l203_203377

noncomputable section

variables (a x : ℝ)

def P : Prop := x^2 - 4*a*x + 3*a^2 < 0
def Q : Prop := abs (x - 3) ≤ 1

-- Part 1: If a=1 and P ∨ Q, prove the range of x is 1 < x ≤ 4
theorem part1 (h1 : a = 1) (h2 : P a x ∨ Q x) : 1 < x ∧ x ≤ 4 :=
sorry

-- Part 2: If ¬P is necessary but not sufficient for ¬Q, prove the range of a is 4/3 ≤ a ≤ 2
theorem part2 (h : (¬P a x → ¬Q x) ∧ (¬Q x → ¬P a x → False)) : 4/3 ≤ a ∧ a ≤ 2 :=
sorry

end part1_part2_l203_203377


namespace placement_ways_l203_203533

theorem placement_ways (rows cols crosses : ℕ) (h1 : rows = 3) (h2 : cols = 4) (h3 : crosses = 4)
  (condition : ∀ r : Fin rows, ∃ c : Fin cols, r < rows ∧ c < cols) : 
  (∃ n, n = (3 * 6 * 2) → n = 36) :=
by 
  -- Proof placeholder
  sorry

end placement_ways_l203_203533


namespace number_of_four_digit_numbers_l203_203155

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l203_203155


namespace find_r_l203_203554

theorem find_r (r : ℝ) (cone1_radius cone2_radius cone3_radius : ℝ) (sphere_radius : ℝ)
  (cone_height_eq : cone1_radius = 2 * r ∧ cone2_radius = 3 * r ∧ cone3_radius = 10 * r)
  (sphere_touch : sphere_radius = 2)
  (center_eq_dist : ∀ {P Q : ℝ}, dist P Q = 2 → dist Q r = 2) :
  r = 1 := 
sorry

end find_r_l203_203554


namespace transformed_data_stats_l203_203646

noncomputable def data_set (n : ℕ) : Type := vector ℝ n

noncomputable def mean (s : data_set n) := (s.to_list.sum / n)

noncomputable def variance (s : data_set n) : ℝ := 
  let μ := mean s in 
  (s.to_list.map (λ x, (x - μ)^2)).sum / n

noncomputable def transformed_data (s : data_set n) : data_set n :=
  ⟨s.to_list.map (λ x, 3 * x + 2), sorry⟩

theorem transformed_data_stats (s : data_set n) 
  (h_avg : mean s = 2) 
  (h_var : variance s = 1) :
  mean (transformed_data s) = 8 ∧ variance (transformed_data s) = 9 := sorry

end transformed_data_stats_l203_203646


namespace unique_arrangements_moon_l203_203898

theorem unique_arrangements_moon : 
  let word := ["M", "O", "O", "N"]
  let n := word.length
  n.factorial / (word.count (fun c => c = "O")).factorial = 12 :=
by
  let word := ["M", "O", "O", "N"]
  let n := word.length
  have h : n = 4 := rfl
  have hO : word.count (fun c => c = "O") = 2 := rfl
  calc
    n.factorial / (word.count (fun c => c = "O")).factorial
        = 4.factorial / 2.factorial : by rw [h, hO]
    ... = 24 / 2 : by norm_num
    ... = 12 : by norm_num

end unique_arrangements_moon_l203_203898


namespace length_increase_percentage_l203_203566

theorem length_increase_percentage
  (L W : ℝ)
  (A : ℝ := L * W)
  (A' : ℝ := 1.30000000000000004 * A)
  (new_length : ℝ := L * (1 + x / 100))
  (new_width : ℝ := W / 2)
  (area_equiv : new_length * new_width = A')
  (x : ℝ) :
  1 + x / 100 = 2.60000000000000008 :=
by
  -- Proof goes here
  sorry

end length_increase_percentage_l203_203566


namespace cars_without_paying_l203_203267

theorem cars_without_paying (total_cars : ℕ) (percent_with_tickets : ℚ) (fraction_with_passes : ℚ)
  (h1 : total_cars = 300)
  (h2 : percent_with_tickets = 0.75)
  (h3 : fraction_with_passes = 1/5) :
  let cars_with_tickets := percent_with_tickets * total_cars
  let cars_with_passes := fraction_with_passes * cars_with_tickets
  total_cars - (cars_with_tickets + cars_with_passes) = 30 :=
by
  -- Placeholder proof
  sorry

end cars_without_paying_l203_203267


namespace antonov_packs_remaining_l203_203743

theorem antonov_packs_remaining (total_candies : ℕ) (pack_size : ℕ) (packs_given : ℕ) (candies_remaining : ℕ) (packs_remaining : ℕ) :
  total_candies = 60 →
  pack_size = 20 →
  packs_given = 1 →
  candies_remaining = total_candies - pack_size * packs_given →
  packs_remaining = candies_remaining / pack_size →
  packs_remaining = 2 := by
  sorry

end antonov_packs_remaining_l203_203743


namespace sequence_first_number_l203_203441

theorem sequence_first_number (a: ℕ → ℕ) (h1: a 7 = 14) (h2: a 8 = 19) (h3: a 9 = 33) :
  (∀ n, n ≥ 2 → a (n+1) = a n + a (n-1)) → a 1 = 30 :=
by
  sorry

end sequence_first_number_l203_203441


namespace count_complex_numbers_l203_203341

theorem count_complex_numbers (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h : a + b ≤ 5) : 
  ∃ n, n = 10 := 
by
  sorry

end count_complex_numbers_l203_203341


namespace sufficient_but_not_necessary_condition_l203_203775

def M : Set ℝ := {x | 0 < x ∧ x ≤ 2}

def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by
  sorry

end sufficient_but_not_necessary_condition_l203_203775


namespace alice_has_winning_strategy_l203_203386

def alice_has_winning_strategy_condition (nums : List ℤ) : Prop :=
  nums.length = 17 ∧ ∀ x ∈ nums, ¬ (x % 17 = 0)

theorem alice_has_winning_strategy (nums : List ℤ) (H : alice_has_winning_strategy_condition nums) : ∃ (f : List ℤ → List ℤ), ∀ k, (f^[k] nums).sum % 17 = 0 :=
sorry

end alice_has_winning_strategy_l203_203386


namespace percentage_of_number_l203_203451

/-- 
  Given a certain percentage \( P \) of 600 is 90.
  If 30% of 50% of a number 4000 is 90,
  Then P equals to 15%.
-/
theorem percentage_of_number (P : ℝ) (h1 : (0.30 : ℝ) * (0.50 : ℝ) * 4000 = 600) (h2 : P * 600 = 90) :
  P = 0.15 :=
  sorry

end percentage_of_number_l203_203451


namespace expenses_of_five_yuan_l203_203995

theorem expenses_of_five_yuan (income_denotation : ℤ) (opposite_effect : ∀ x : ℤ, -x) :
  income_denotation = 5 → opposite_effect income_denotation = -5 :=
by sorry

end expenses_of_five_yuan_l203_203995


namespace selena_trip_length_l203_203689

variable (y : ℚ)

def selena_trip (y : ℚ) : Prop :=
  y / 4 + 16 + y / 6 = y

theorem selena_trip_length : selena_trip y → y = 192 / 7 :=
by
  sorry

end selena_trip_length_l203_203689


namespace four_digit_number_count_l203_203117

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l203_203117


namespace area_of_hexagon_l203_203372

def isRegularHexagon (A B C D E F : Type) : Prop := sorry
def isInsideQuadrilateral (P : Type) (A B C D : Type) : Prop := sorry
def areaTriangle (P X Y : Type) : Real := sorry

theorem area_of_hexagon (A B C D E F P : Type)
    (h1 : isRegularHexagon A B C D E F)
    (h2 : isInsideQuadrilateral P A B C D)
    (h3 : areaTriangle P B C = 20)
    (h4 : areaTriangle P A D = 23) :
    ∃ area : Real, area = 189 :=
sorry

end area_of_hexagon_l203_203372


namespace nickel_ate_3_chocolates_l203_203820

theorem nickel_ate_3_chocolates (R N : ℕ) (h1 : R = 7) (h2 : R = N + 4) : N = 3 := by
  sorry

end nickel_ate_3_chocolates_l203_203820


namespace gcd_360_150_l203_203418

theorem gcd_360_150 : Int.gcd 360 150 = 30 := by
  have h360 : 360 = 2^3 * 3^2 * 5 := by
    ring
  have h150 : 150 = 2 * 3 * 5^2 := by
    ring
  rw [h360, h150]
  sorry

end gcd_360_150_l203_203418


namespace count_four_digit_numbers_l203_203100

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l203_203100


namespace gold_coins_l203_203393

theorem gold_coins (c n : ℕ) 
  (h₁ : n = 8 * (c - 1))
  (h₂ : n = 5 * c + 4) :
  n = 24 :=
by
  sorry

end gold_coins_l203_203393


namespace can_vasya_obtain_400_mercedes_l203_203041

-- Define the types for the cars
inductive Car : Type
| Zh : Car
| V : Car
| M : Car

-- Define the initial conditions as exchange constraints
def exchange1 (Zh V M : ℕ) : Prop :=
  3 * Zh = V + M

def exchange2 (V Zh M : ℕ) : Prop :=
  3 * V = 2 * Zh + M

-- Define the initial number of Zhiguli cars Vasya has.
def initial_Zh : ℕ := 700

-- Define the target number of Mercedes cars Vasya wants.
def target_M : ℕ := 400

-- The proof goal: Vasya cannot exchange to get exactly 400 Mercedes cars.
theorem can_vasya_obtain_400_mercedes (Zh V M : ℕ) (h1 : exchange1 Zh V M) (h2 : exchange2 V Zh M) :
  initial_Zh = 700 → target_M = 400 → (Zh ≠ 0 ∨ V ≠ 0 ∨ M ≠ 400) := sorry

end can_vasya_obtain_400_mercedes_l203_203041


namespace XiaoMaHu_correct_calculation_l203_203236

theorem XiaoMaHu_correct_calculation :
  (∃ A B C D : Prop, (A = ((a b : ℝ) → (a - b)^2 = a^2 - b^2)) ∧ 
                   (B = ((a : ℝ) → (-2 * a^3)^2 = 4 * a^6)) ∧ 
                   (C = ((a : ℝ) → a^3 + a^2 = 2 * a^5)) ∧ 
                   (D = ((a : ℝ) → -(a - 1) = -a - 1)) ∧ 
                   (¬A ∧ B ∧ ¬C ∧ ¬D)) :=
sorry

end XiaoMaHu_correct_calculation_l203_203236


namespace undefined_hydrogen_production_l203_203335

-- Define the chemical species involved as follows:
structure ChemQty where
  Ethane : ℕ
  Oxygen : ℕ
  CarbonDioxide : ℕ
  Water : ℕ

-- Balanced reaction equation
def balanced_reaction : ChemQty :=
  { Ethane := 2, Oxygen := 7, CarbonDioxide := 4, Water := 6 }

-- Given conditions as per problem scenario
def initial_state : ChemQty :=
  { Ethane := 1, Oxygen := 2, CarbonDioxide := 0, Water := 0 }

-- The statement reflecting the unclear result of the reaction under the given conditions.
theorem undefined_hydrogen_production :
  initial_state.Oxygen < balanced_reaction.Oxygen / balanced_reaction.Ethane * initial_state.Ethane →
  ∃ water_products : ℕ, water_products ≤ 6 * initial_state.Ethane / 2 := 
by
  -- Due to incomplete reaction
  sorry

end undefined_hydrogen_production_l203_203335


namespace find_X_l203_203816

theorem find_X (X : ℚ) (h : (1/3 : ℚ) * (1/4 : ℚ) * X = (1/4 : ℚ) * (1/6 : ℚ) * 120) : X = 60 := 
sorry

end find_X_l203_203816


namespace tank_depth_l203_203871

theorem tank_depth (d : ℝ)
    (field_length : ℝ) (field_breadth : ℝ)
    (tank_length : ℝ) (tank_breadth : ℝ)
    (remaining_field_area : ℝ)
    (rise_in_field_level : ℝ)
    (field_area_eq : field_length * field_breadth = 4500)
    (tank_area_eq : tank_length * tank_breadth = 500)
    (remaining_field_area_eq : remaining_field_area = 4500 - 500)
    (earth_volume_spread_eq : remaining_field_area * rise_in_field_level = 2000)
    (volume_eq : tank_length * tank_breadth * d = 2000)
  : d = 4 := by
  sorry

end tank_depth_l203_203871


namespace females_with_advanced_degrees_l203_203671

theorem females_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_advanced_degrees : ℕ)
  (males_college_degree_only : ℕ)
  (h1 : total_employees = 200)
  (h2 : total_females = 120)
  (h3 : total_advanced_degrees = 100)
  (h4 : males_college_degree_only = 40) :
  (total_advanced_degrees - (total_employees - total_females - males_college_degree_only) = 60) :=
by
  -- proof will go here
  sorry

end females_with_advanced_degrees_l203_203671


namespace sum_of_squares_of_roots_eq_30_l203_203808

noncomputable def polynomial := (x : ℝ) → x^4 - 15 * x^2 + 56 = 0

theorem sum_of_squares_of_roots_eq_30
  (a b c d : ℝ)
  (h1 : polynomial a)
  (h2 : polynomial b)
  (h3 : polynomial c)
  (h4 : polynomial d) : 
  a^2 + b^2 + c^2 + d^2 = 30 :=
sorry

end sum_of_squares_of_roots_eq_30_l203_203808


namespace ellipse_parabola_intersection_l203_203621

theorem ellipse_parabola_intersection (c : ℝ) : 
  (∀ x y : ℝ, (x^2 + (y^2 / 4) = c^2 ∧ y = x^2 - 2 * c) → false) ↔ c > 1 := by
  sorry

end ellipse_parabola_intersection_l203_203621


namespace stratified_sampling_grade10_sampled_count_l203_203733

def total_students : ℕ := 2000
def grade10_students : ℕ := 600
def grade11_students : ℕ := 680
def grade12_students : ℕ := 720
def total_sampled_students : ℕ := 50

theorem stratified_sampling_grade10_sampled_count :
  15 = (total_sampled_students * grade10_students / total_students) :=
by sorry

end stratified_sampling_grade10_sampled_count_l203_203733


namespace work_completion_time_l203_203449

theorem work_completion_time 
  (M W : ℝ) 
  (h1 : (10 * M + 15 * W) * 6 = 1) 
  (h2 : M * 100 = 1) 
  : W * 225 = 1 := 
by
  sorry

end work_completion_time_l203_203449


namespace rotation_test_l203_203818

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rotate_90_clockwise (p : Point ℝ) : Point ℝ :=
  Point.mk p.y (-p.x)

def A : Point ℝ := ⟨2, 3⟩
def B : Point ℝ := ⟨3, -2⟩

theorem rotation_test : rotate_90_clockwise A = B :=
by
  sorry

end rotation_test_l203_203818


namespace find_number_l203_203729

theorem find_number (x : ℚ) (h : 0.5 * x = (3/5) * x - 10) : x = 100 := 
sorry

end find_number_l203_203729


namespace min_max_x_l203_203598

-- Definitions for the initial conditions and surveys
def students : ℕ := 100
def like_math_initial : ℕ := 50
def dislike_math_initial : ℕ := 50
def like_math_final : ℕ := 60
def dislike_math_final : ℕ := 40

-- Variables for the students' responses
variables (a b c d : ℕ)

-- Conditions based on the problem statement
def initial_survey : Prop := a + d = like_math_initial ∧ b + c = dislike_math_initial
def final_survey : Prop := a + c = like_math_final ∧ b + d = dislike_math_final

-- Definition of x as the number of students who changed their answer
def x : ℕ := c + d

-- Prove the minimum and maximum value of x with given conditions
theorem min_max_x (a b c d : ℕ) 
  (initial_cond : initial_survey a b c d)
  (final_cond : final_survey a b c d)
  : 10 ≤ (x c d) ∧ (x c d) ≤ 90 :=
by
  -- This is where the proof would go, but we'll simply state sorry for now.
  sorry

end min_max_x_l203_203598


namespace n_plus_one_sum_of_three_squares_l203_203679

theorem n_plus_one_sum_of_three_squares (n x : ℤ) (h1 : n > 1) (h2 : 3 * n + 1 = x^2) :
  ∃ a b c : ℤ, n + 1 = a^2 + b^2 + c^2 :=
by
  sorry

end n_plus_one_sum_of_three_squares_l203_203679


namespace solution_set_of_inequality_l203_203403

theorem solution_set_of_inequality (x : ℝ) : (x / (x - 1) < 0) ↔ (0 < x ∧ x < 1) := 
sorry

end solution_set_of_inequality_l203_203403


namespace starters_choice_l203_203215

/-- There are 18 players including a set of quadruplets: Bob, Bill, Ben, and Bert. -/
def total_players : ℕ := 18

/-- The set of quadruplets: Bob, Bill, Ben, and Bert. -/
def quadruplets : Finset (String) := {"Bob", "Bill", "Ben", "Bert"}

/-- We need to choose 7 starters, exactly 3 of which are from the set of quadruplets. -/
def ways_to_choose_starters : ℕ :=
  let quadruplet_combinations := Nat.choose 4 3
  let remaining_spots := 4
  let remaining_players := total_players - 4
  quadruplet_combinations * Nat.choose remaining_players remaining_spots

theorem starters_choice (h1 : total_players = 18)
                        (h2 : quadruplets.card = 4) :
  ways_to_choose_starters = 4004 :=
by 
  -- conditional setups here
  sorry

end starters_choice_l203_203215


namespace lindsay_dolls_l203_203963

theorem lindsay_dolls (B B_b B_k : ℕ) 
  (h1 : B_b = 4 * B)
  (h2 : B_k = 4 * B - 2)
  (h3 : B_b + B_k = B + 26) : B = 4 :=
by
  sorry

end lindsay_dolls_l203_203963


namespace distance_equals_absolute_value_l203_203815

def distance_from_origin (x : ℝ) : ℝ := abs x

theorem distance_equals_absolute_value (x : ℝ) : distance_from_origin x = abs x :=
by
  sorry

end distance_equals_absolute_value_l203_203815


namespace glass_original_water_l203_203030

theorem glass_original_water 
  (O : ℝ)  -- Ounces of water originally in the glass
  (evap_per_day : ℝ)  -- Ounces of water evaporated per day
  (total_days : ℕ)    -- Total number of days evaporation occurs
  (percent_evaporated : ℝ)  -- Percentage of the original amount that evaporated
  (h1 : evap_per_day = 0.06)  -- 0.06 ounces of water evaporated each day
  (h2 : total_days = 20)  -- Evaporation occurred over a period of 20 days
  (h3 : percent_evaporated = 0.12)  -- 12% of the original amount evaporated during this period
  (h4 : evap_per_day * total_days = 1.2)  -- 0.06 ounces per day for 20 days total gives 1.2 ounces
  (h5 : percent_evaporated * O = evap_per_day * total_days) :  -- 1.2 ounces is 12% of the original amount
  O = 10 :=  -- Prove that the original amount is 10 ounces
sorry

end glass_original_water_l203_203030


namespace systematic_sampling_l203_203005

theorem systematic_sampling :
  let N := 60
  let n := 5
  let k := N / n
  let initial_sample := 5
  let samples := [initial_sample, initial_sample + k, initial_sample + 2 * k, initial_sample + 3 * k, initial_sample + 4 * k] 
  samples = [5, 17, 29, 41, 53] := sorry

end systematic_sampling_l203_203005


namespace thin_film_radius_volume_l203_203032

theorem thin_film_radius_volume :
  ∀ (r : ℝ) (V : ℝ) (t : ℝ), 
    V = 216 → t = 0.1 → π * r^2 * t = V → r = Real.sqrt (2160 / π) :=
by
  sorry

end thin_film_radius_volume_l203_203032


namespace percentage_of_female_officers_on_duty_l203_203971

-- Declare the conditions
def total_officers_on_duty : ℕ := 100
def female_officers_on_duty : ℕ := 50
def total_female_officers : ℕ := 250

-- The theorem to prove
theorem percentage_of_female_officers_on_duty :
  (female_officers_on_duty / total_female_officers) * 100 = 20 := 
sorry

end percentage_of_female_officers_on_duty_l203_203971


namespace four_digit_numbers_count_l203_203105

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l203_203105


namespace four_digit_number_count_l203_203138

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l203_203138


namespace red_lights_l203_203850

theorem red_lights (total_lights yellow_lights blue_lights red_lights : ℕ)
  (h1 : total_lights = 95)
  (h2 : yellow_lights = 37)
  (h3 : blue_lights = 32)
  (h4 : red_lights = total_lights - (yellow_lights + blue_lights)) :
  red_lights = 26 := by
  sorry

end red_lights_l203_203850


namespace class_committee_selection_l203_203587

theorem class_committee_selection :
  let members := ["A", "B", "C", "D", "E"]
  let admissible_entertainment_candidates := ["C", "D", "E"]
  ∃ (entertainment : String) (study : String) (sports : String),
    entertainment ∈ admissible_entertainment_candidates ∧
    study ∈ members.erase entertainment ∧
    sports ∈ (members.erase entertainment).erase study ∧
    (3 * 4 * 3 = 36) :=
sorry

end class_committee_selection_l203_203587


namespace total_sum_vowels_l203_203889

theorem total_sum_vowels :
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  A + E + I + O + U = 20 := by
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  sorry

end total_sum_vowels_l203_203889


namespace function_machine_output_l203_203361

-- Define the initial input
def input : ℕ := 12

-- Define the function machine steps
def functionMachine (x : ℕ) : ℕ :=
  if x * 3 <= 20 then (x * 3) / 2
  else (x * 3) - 2

-- State the property we want to prove
theorem function_machine_output : functionMachine 12 = 34 :=
by
  -- Skip the proof
  sorry

end function_machine_output_l203_203361


namespace triangle_angle_sum_33_75_l203_203575

theorem triangle_angle_sum_33_75 (x : ℝ) 
  (h₁ : 45 + 3 * x + x = 180) : 
  x = 33.75 :=
  sorry

end triangle_angle_sum_33_75_l203_203575


namespace sum_binomials_eq_l203_203311

theorem sum_binomials_eq : 
  (Nat.choose 6 1) + (Nat.choose 6 2) + (Nat.choose 6 3) + (Nat.choose 6 4) + (Nat.choose 6 5) = 62 :=
by
  sorry

end sum_binomials_eq_l203_203311


namespace unique_arrangements_of_MOON_l203_203903

open Nat

theorem unique_arrangements_of_MOON : 
  let word := "MOON"
  let n := 4
  let numM := 1
  let numN := 1
  let numO := 2
  factorial n / (factorial numO * factorial numM * factorial numN) = 12 :=
by
  let word := "MOON"
  let n := 4
  let numM := 1
  let numN := 1
  let numO := 2
  sorry

end unique_arrangements_of_MOON_l203_203903


namespace simplify_expression_l203_203690

theorem simplify_expression :
  (1 / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 6 - 2)))) =
  ((3 * Real.sqrt 5 + 2 * Real.sqrt 6 + 2) / 29) :=
  sorry

end simplify_expression_l203_203690


namespace school_spent_total_l203_203739

noncomputable def seminar_fee (num_teachers : ℕ) : ℝ :=
  let base_fee := 150 * num_teachers
  if num_teachers >= 20 then
    base_fee * 0.925
  else if num_teachers >= 10 then
    base_fee * 0.95
  else
    base_fee

noncomputable def seminar_fee_with_tax (num_teachers : ℕ) : ℝ :=
  let fee := seminar_fee num_teachers
  fee * 1.06

noncomputable def food_allowance (num_teachers : ℕ) (num_special : ℕ) : ℝ :=
  let num_regular := num_teachers - num_special
  num_regular * 10 + num_special * 15

noncomputable def total_cost (num_teachers : ℕ) (num_special : ℕ) : ℝ :=
  seminar_fee_with_tax num_teachers + food_allowance num_teachers num_special

theorem school_spent_total (num_teachers num_special : ℕ) (h : num_teachers = 22 ∧ num_special = 3) :
  total_cost num_teachers num_special = 3470.65 :=
by
  sorry

end school_spent_total_l203_203739


namespace toys_produced_per_day_l203_203717

theorem toys_produced_per_day :
  (3400 / 5 = 680) :=
by
  sorry

end toys_produced_per_day_l203_203717


namespace sum_of_first_four_terms_of_sequence_l203_203953

-- Define the sequence, its common difference, and the given initial condition
def a_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, a (n + 1) - a n = 2) ∧ (a 2 = 5)

-- Define the sum of the first four terms
def sum_first_four_terms (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_of_first_four_terms_of_sequence :
  ∀ (a : ℕ → ℤ), a_sequence a → sum_first_four_terms a = 24 :=
by
  intro a h
  rw [a_sequence] at h
  obtain ⟨h_diff, h_a2⟩ := h
  sorry

end sum_of_first_four_terms_of_sequence_l203_203953


namespace find_a2016_l203_203070

variable {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Given conditions
def cond1 : S 1 = 6 := by sorry
def cond2 : S 2 = 4 := by sorry
def cond3 (n : ℕ) : S n > 0 := by sorry
def cond4 (n : ℕ) : S (2 * n - 1) ^ 2 = S (2 * n) * S (2 * n + 2) := by sorry
def cond5 (n : ℕ) : 2 * S (2 * n + 2) = S (2 * n - 1) + S (2 * n + 1) := by sorry

theorem find_a2016 : a 2016 = -1009 := by
  -- Use the provided conditions to prove the statement
  sorry

end find_a2016_l203_203070


namespace Jack_minimum_cars_per_hour_l203_203202

theorem Jack_minimum_cars_per_hour (J : ℕ) (h1 : 2 * 8 + 8 * J ≥ 40) : J ≥ 3 :=
by {
  -- The statement of the theorem directly follows
  sorry
}

end Jack_minimum_cars_per_hour_l203_203202


namespace Maria_students_l203_203682

variable (M J : ℕ)

def conditions : Prop :=
  (M = 4 * J) ∧ (M + J = 2500)

theorem Maria_students : conditions M J → M = 2000 :=
by
  intro h
  sorry

end Maria_students_l203_203682


namespace ratio_of_bike_to_tractor_speed_l203_203561

theorem ratio_of_bike_to_tractor_speed (d_tr: ℝ) (t_tr: ℝ) (d_car: ℝ) (t_car: ℝ) (k: ℝ) (β: ℝ) 
  (h1: d_tr / t_tr = 25) 
  (h2: d_car / t_car = 90)
  (h3: 90 = 9 / 5 * β)
: β / (d_tr / t_tr) = 2 := 
by
  sorry

end ratio_of_bike_to_tractor_speed_l203_203561


namespace tangent_parallel_l203_203946

noncomputable def f (x: ℝ) : ℝ := x^4 - x
noncomputable def f' (x: ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_parallel
  (P : ℝ × ℝ)
  (hp : P = (1, 0))
  (tangent_parallel : ∀ x, f' x = 3 ↔ x = 1)
  : P = (1, 0) := 
by 
  sorry

end tangent_parallel_l203_203946


namespace R2_area_l203_203766

-- Definitions for the conditions
def R1_side1 : ℝ := 4
def R1_area : ℝ := 16
def R2_diagonal : ℝ := 10
def similar_rectangles (R1 R2 : ℝ × ℝ) : Prop := (R1.fst / R1.snd = R2.fst / R2.snd)

-- Main theorem
theorem R2_area {a b : ℝ} 
  (R1_side1 : a = 4)
  (R1_area : a * a = 16) 
  (R2_diagonal : b = 10)
  (h : similar_rectangles (a, a) (b / (10 / (2 : ℝ)), b / (10 / (2 : ℝ)))) : 
  b * b / (2 : ℝ) = 50 :=
by
  sorry

end R2_area_l203_203766


namespace inverse_proportion_l203_203711

theorem inverse_proportion {x y : ℝ} :
  (y = (3 / x)) -> ¬(y = x / 3) ∧ ¬(y = 3 / (x + 1)) ∧ ¬(y = 3 * x) :=
by
  sorry

end inverse_proportion_l203_203711


namespace man_older_than_son_l203_203033

theorem man_older_than_son (S M : ℕ) (h1 : S = 23) (h2 : M + 2 = 2 * (S + 2)) : M - S = 25 :=
by
  sorry

end man_older_than_son_l203_203033


namespace find_f2_l203_203862

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f2 (h : ∀ x : ℝ, f x + 3 * f (8 - x) = x) : f 2 = 2 :=
by
  sorry

end find_f2_l203_203862


namespace osmanthus_trees_variance_l203_203316

theorem osmanthus_trees_variance :
  let n := 4
  let p := 4 / 5
  let ξ := binomial n p
  ξ.variance = 16 / 25 :=
by
  sorry

end osmanthus_trees_variance_l203_203316


namespace rectangleY_has_tileD_l203_203410

-- Define the structure for a tile
structure Tile where
  top : Nat
  right : Nat
  bottom : Nat
  left : Nat

-- Define tiles
def TileA : Tile := { top := 6, right := 3, bottom := 5, left := 2 }
def TileB : Tile := { top := 3, right := 6, bottom := 2, left := 5 }
def TileC : Tile := { top := 5, right := 7, bottom := 1, left := 2 }
def TileD : Tile := { top := 2, right := 5, bottom := 6, left := 3 }

-- Define rectangles (positioning)
inductive Rectangle
| W | X | Y | Z

-- Define which tile is in Rectangle Y
def tileInRectangleY : Tile → Prop :=
  fun t => t = TileD

-- Statement to prove
theorem rectangleY_has_tileD : tileInRectangleY TileD :=
by
  -- The final statement to be proven, skipping the proof itself with sorry
  sorry

end rectangleY_has_tileD_l203_203410


namespace fraction_evaluation_l203_203300

theorem fraction_evaluation : (1 / (2 + 1 / (3 + 1 / 4))) = (13 / 30) := by
  sorry

end fraction_evaluation_l203_203300


namespace tennis_tournament_l203_203356

theorem tennis_tournament (n x : ℕ) 
    (p : ℕ := 4 * n) 
    (m : ℕ := (p * (p - 1)) / 2) 
    (r_women : ℕ := 3 * x) 
    (r_men : ℕ := 2 * x) 
    (total_wins : ℕ := r_women + r_men) 
    (h_matches : m = total_wins) 
    (h_ratio : r_women = 3 * x ∧ r_men = 2 * x ∧ 4 * n * (4 * n - 1) = 10 * x): 
    n = 4 :=
by
  sorry

end tennis_tournament_l203_203356


namespace math_problem_proof_l203_203701

variable (Zhang Li Wang Zhao Liu : Prop)
variable (n : ℕ)
variable (reviewed_truth : Zhang → n = 0 ∧ Li → n = 1 ∧ Wang → n = 2 ∧ Zhao → n = 3 ∧ Liu → n = 4)
variable (reviewed_lie : ¬Zhang → ¬(n = 0) ∧ ¬Li → ¬(n = 1) ∧ ¬Wang → ¬(n = 2) ∧ ¬Zhao → ¬(n = 3) ∧ ¬Liu → ¬(n = 4))
variable (some_reviewed : ∃ x, x ∧ ¬x)

theorem math_problem_proof: n = 1 :=
by
  -- Proof omitted, insert logic here
  sorry

end math_problem_proof_l203_203701


namespace find_b_l203_203353

theorem find_b (a : ℝ) (A : ℝ) (B : ℝ) (b : ℝ)
  (ha : a = 5) 
  (hA : A = Real.pi / 6) 
  (htanB : Real.tan B = 3 / 4)
  (hsinB : Real.sin B = 3 / 5):
  b = 6 := 
by 
  sorry

end find_b_l203_203353


namespace inequality_proof_l203_203641

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) : a < 2 * b - b^2 / a := 
by
  -- mathematical proof goes here
  sorry

end inequality_proof_l203_203641


namespace three_digit_reverse_sum_to_1777_l203_203459

theorem three_digit_reverse_sum_to_1777 :
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 101 * (a + c) + 20 * b = 1777 :=
by
  sorry

end three_digit_reverse_sum_to_1777_l203_203459


namespace perpendicular_slope_l203_203910

theorem perpendicular_slope :
  ∀ (x y : ℝ), 5 * x - 2 * y = 10 → y = ((5 : ℝ) / 2) * x - 5 → ∃ (m : ℝ), m = - (2 / 5) := by
  sorry

end perpendicular_slope_l203_203910


namespace number_of_multiples_of_3003_l203_203666

theorem number_of_multiples_of_3003 (i j : ℕ) (h : 0 ≤ i ∧ i < j ∧ j ≤ 199): 
  (∃ n : ℕ, n = 3003 * k ∧ n = 10^j - 10^i) → 
  (number_of_solutions = 1568) :=
sorry

end number_of_multiples_of_3003_l203_203666


namespace simplify_expression_l203_203369

noncomputable def q (x a b c d : ℝ) :=
  (x + a)^4 / ((a - b) * (a - c) * (a - d))
  + (x + b)^4 / ((b - a) * (b - c) * (b - d))
  + (x + c)^4 / ((c - a) * (c - b) * (c - d))
  + (x + d)^4 / ((d - a) * (d - b) * (d - c))

theorem simplify_expression (a b c d x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  q x a b c d = a + b + c + d + 4 * x :=
by
  sorry

end simplify_expression_l203_203369


namespace min_value_sqrt_sum_l203_203067

open Real

theorem min_value_sqrt_sum (x : ℝ) : 
    ∃ c : ℝ, (∀ x : ℝ, c ≤ sqrt (x^2 - 4 * x + 13) + sqrt (x^2 - 10 * x + 26)) ∧ 
             (sqrt ((17/4)^2 - 4 * (17/4) + 13) + sqrt ((17/4)^2 - 10 * (17/4) + 26) = 5 ∧ c = 5) := 
by
  sorry

end min_value_sqrt_sum_l203_203067


namespace geometric_sequence_sum_t_value_l203_203519

theorem geometric_sequence_sum_t_value 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (t : ℝ)
  (h1 : ∀ n : ℕ, S_n n = 3^((n:ℝ)-1) + t)
  (h2 : a_n 1 = 3^0 + t)
  (geometric : ∀ n : ℕ, n ≥ 2 → a_n n = 2 * 3^(n-2)) :
  t = -1/3 :=
by
  sorry

end geometric_sequence_sum_t_value_l203_203519


namespace number_of_four_digit_numbers_l203_203127

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l203_203127


namespace Josh_marbles_count_l203_203798

-- Definitions of the given conditions
def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7

-- The statement we aim to prove
theorem Josh_marbles_count : (initial_marbles - lost_marbles) = 9 :=
by
  -- Skipping the proof with sorry
  sorry

end Josh_marbles_count_l203_203798


namespace gcd_of_360_and_150_is_30_l203_203426

theorem gcd_of_360_and_150_is_30 : Nat.gcd 360 150 = 30 :=
by
  sorry

end gcd_of_360_and_150_is_30_l203_203426


namespace intersection_value_l203_203916

theorem intersection_value (x y : ℝ) (h₁ : y = 10 / (x^2 + 5)) (h₂ : x + 2 * y = 5) : 
  x = 1 :=
sorry

end intersection_value_l203_203916


namespace smallest_x_for_perfect_cube_l203_203060

theorem smallest_x_for_perfect_cube (x : ℕ) (M : ℤ) (hx : x > 0) (hM : ∃ M, 1680 * x = M^3) : x = 44100 :=
sorry

end smallest_x_for_perfect_cube_l203_203060


namespace gcd_360_150_l203_203416

theorem gcd_360_150 : Int.gcd 360 150 = 30 := by
  have h360 : 360 = 2^3 * 3^2 * 5 := by
    ring
  have h150 : 150 = 2 * 3 * 5^2 := by
    ring
  rw [h360, h150]
  sorry

end gcd_360_150_l203_203416


namespace common_fraction_equiv_l203_203231

noncomputable def decimal_equivalent_frac : Prop :=
  ∃ (x : ℚ), x = 413 / 990 ∧ x = 0.4 + (7/10^2 + 1/10^3) / (1 - 1/10^2)

theorem common_fraction_equiv : decimal_equivalent_frac :=
by
  sorry

end common_fraction_equiv_l203_203231


namespace probability_different_from_half_l203_203502

noncomputable def prob_different_color (n : ℕ) (balls : ℕ) : ℚ :=
if balls = n then 0 else 1 / 2 ^ n

theorem probability_different_from_half 
  (n : ℕ) : 
  n = 8 → 
  (∀ (i : ℕ), i < n → (∃ (color : bool), (∑ set_of (λi : ℕ, balls [i] = color)) = n/2)) → 
  prob_different_color n 4 = 35 / 128 :=
begin
  intros h₁ h₂,
  unfold prob_different_color,
  sorry
end

end probability_different_from_half_l203_203502


namespace four_digit_number_count_l203_203141

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l203_203141


namespace z_plus_inv_y_eq_10_div_53_l203_203980

-- Define the conditions for x, y, z being positive real numbers such that
-- xyz = 1, x + 1/z = 8, and y + 1/x = 20
variables (x y z : ℝ)
variables (hx : x > 0)
variables (hy : y > 0)
variables (hz : z > 0)
variables (h1 : x * y * z = 1)
variables (h2 : x + 1 / z = 8)
variables (h3 : y + 1 / x = 20)

-- The goal is to prove that z + 1/y = 10 / 53
theorem z_plus_inv_y_eq_10_div_53 : z + 1 / y = 10 / 53 :=
by {
  sorry
}

end z_plus_inv_y_eq_10_div_53_l203_203980


namespace seonyeong_class_size_l203_203204

theorem seonyeong_class_size :
  (12 * 4 + 3) - 12 = 39 :=
by
  sorry

end seonyeong_class_size_l203_203204


namespace rational_third_vertex_l203_203609

theorem rational_third_vertex (x1 y1 x2 y2 : ℚ) (x3 y3 : ℚ) :
  (∃ x3 y3 : ℚ, true) ↔ (∀ X, (X = 90 ∨ ∃ r : ℚ, tan X = r)) :=
sorry

end rational_third_vertex_l203_203609


namespace cheenu_speed_difference_l203_203496

theorem cheenu_speed_difference :
  let cycling_time := 120 -- minutes
  let cycling_distance := 24 -- miles
  let jogging_time := 180 -- minutes
  let jogging_distance := 18 -- miles
  let cycling_speed := cycling_time / cycling_distance -- minutes per mile
  let jogging_speed := jogging_time / jogging_distance -- minutes per mile
  let speed_difference := jogging_speed - cycling_speed -- minutes per mile
  speed_difference = 5 := by sorry

end cheenu_speed_difference_l203_203496


namespace necessary_and_sufficient_problem_l203_203586

theorem necessary_and_sufficient_problem : 
  (¬ (∀ x : ℝ, (-2 < x ∧ x < 1) → (|x| > 1)) ∧ ¬ (∀ x : ℝ, (|x| > 1) → (-2 < x ∧ x < 1))) :=
by {
  sorry
}

end necessary_and_sufficient_problem_l203_203586


namespace never_consecutive_again_l203_203683

theorem never_consecutive_again (n : ℕ) (seq : ℕ → ℕ) :
  (∀ k, seq k = seq 0 + k) → 
  ∀ seq' : ℕ → ℕ,
    (∀ i j, i < j → seq' (2*i) = seq i + seq (j) ∧ seq' (2*i+1) = seq i - seq (j)) →
    ¬ (∀ k, seq' k = seq' 0 + k) :=
by
  sorry

end never_consecutive_again_l203_203683


namespace log_domain_is_pos_real_l203_203018

noncomputable def domain_log : Set ℝ := {x | x > 0}
noncomputable def domain_reciprocal : Set ℝ := {x | x ≠ 0}
noncomputable def domain_sqrt : Set ℝ := {x | x ≥ 0}
noncomputable def domain_exp : Set ℝ := {x | true}

theorem log_domain_is_pos_real :
  (domain_log = {x : ℝ | 0 < x}) ∧ 
  (domain_reciprocal = {x : ℝ | x ≠ 0}) ∧ 
  (domain_sqrt = {x : ℝ | 0 ≤ x}) ∧ 
  (domain_exp = {x : ℝ | true}) →
  domain_log = {x : ℝ | 0 < x} :=
by
  intro h
  sorry

end log_domain_is_pos_real_l203_203018


namespace question_1_question_2_question_3_l203_203388

variable (a b : ℝ)

-- (a * b)^n = a^n * b^n for natural numbers n
theorem question_1 (n : ℕ) : (a * b)^n = a^n * b^n := sorry

-- Calculate 2^5 * (-1/2)^5
theorem question_2 : 2^5 * (-1/2)^5 = -1 := sorry

-- Calculate (-0.125)^2022 * 2^2021 * 4^2020
theorem question_3 : (-0.125)^2022 * 2^2021 * 4^2020 = 1 / 32 := sorry

end question_1_question_2_question_3_l203_203388


namespace perpendicular_lines_l203_203698

theorem perpendicular_lines (a : ℝ) : 
  (∀ (x y : ℝ), (1 - 2 * a) * x - 2 * y + 3 = 0 → 3 * x + y + 2 * a = 0) → 
  a = 1 / 6 :=
by
  sorry

end perpendicular_lines_l203_203698


namespace length_of_room_l203_203398

theorem length_of_room (L : ℝ) (w : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) (room_area : ℝ) :
  w = 12 →
  veranda_width = 2 →
  veranda_area = 144 →
  (L + 2 * veranda_width) * (w + 2 * veranda_width) - L * w = veranda_area →
  L = 20 :=
by
  intro h_w
  intro h_veranda_width
  intro h_veranda_area
  intro h_area_eq
  sorry

end length_of_room_l203_203398


namespace pool_length_l203_203395

def volume_of_pool (width length depth : ℕ) : ℕ :=
  width * length * depth

def volume_of_water (volume : ℕ) (capacity : ℝ) : ℝ :=
  volume * capacity

theorem pool_length (L : ℕ) (width depth : ℕ) (capacity : ℝ) (drain_rate drain_time : ℕ) (h_capacity : capacity = 0.80)
  (h_width : width = 50) (h_depth : depth = 10)
  (h_drain_rate : drain_rate = 60) (h_drain_time : drain_time = 1000)
  (h_drain_volume : volume_of_water (volume_of_pool width L depth) capacity = drain_rate * drain_time) :
  L = 150 :=
by
  sorry

end pool_length_l203_203395


namespace smallest_sum_infinite_geometric_progression_l203_203835

theorem smallest_sum_infinite_geometric_progression :
  ∃ (a q A : ℝ), (a * q = 3) ∧ (0 < q) ∧ (q < 1) ∧ (A = a / (1 - q)) ∧ (A = 12) :=
by
  sorry

end smallest_sum_infinite_geometric_progression_l203_203835


namespace median_circumradius_altitude_inequality_l203_203806

variable (h R m_a m_b m_c : ℝ)

-- Define the condition for the lengths of the medians and other related parameters
-- m_a, m_b, m_c are medians, R is the circumradius, h is the greatest altitude

theorem median_circumradius_altitude_inequality :
  m_a + m_b + m_c ≤ 3 * R + h :=
sorry

end median_circumradius_altitude_inequality_l203_203806


namespace complex_series_sum_l203_203187

theorem complex_series_sum (ω : ℂ) (h₁ : ω^7 = 1) (h₂ : ω ≠ 1) :
  (ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 + ω^32 + 
   ω^34 + ω^36 + ω^38 + ω^40 + ω^42 + ω^44 + ω^46 + ω^48 + ω^50 + 
   ω^52 + ω^54) = -1 :=
by
  sorry

end complex_series_sum_l203_203187


namespace coefficient_and_degree_of_monomial_l203_203829

variable (x y : ℝ)

def monomial : ℝ := -2 * x * y^3

theorem coefficient_and_degree_of_monomial :
  ( ∃ c : ℝ, ∃ d : ℤ, monomial x y = c * x * y^d ∧ c = -2 ∧ d = 4 ) :=
by
  sorry

end coefficient_and_degree_of_monomial_l203_203829


namespace gcd_of_360_and_150_l203_203430

theorem gcd_of_360_and_150 : Nat.gcd 360 150 = 30 := 
by
  sorry

end gcd_of_360_and_150_l203_203430


namespace sample_size_is_40_l203_203359

theorem sample_size_is_40 (total_students : ℕ) (sample_students : ℕ) (h1 : total_students = 240) (h2 : sample_students = 40) : sample_students = 40 :=
by
  sorry

end sample_size_is_40_l203_203359


namespace largest_six_consecutive_composites_less_than_40_l203_203306

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) := ¬ is_prime n ∧ n > 1

theorem largest_six_consecutive_composites_less_than_40 :
  ∃ (seq : ℕ → ℕ) (i : ℕ),
    (∀ j : ℕ, j < 6 → is_composite (seq (i + j))) ∧ 
    (seq i < 40) ∧ 
    (seq (i+1) < 40) ∧ 
    (seq (i+2) < 40) ∧ 
    (seq (i+3) < 40) ∧ 
    (seq (i+4) < 40) ∧ 
    (seq (i+5) < 40) ∧ 
    seq (i+5) = 30 
:= sorry

end largest_six_consecutive_composites_less_than_40_l203_203306


namespace eva_total_marks_correct_l203_203292

-- Definitions based on conditions
def math_marks_second_sem : ℕ := 80
def arts_marks_second_sem : ℕ := 90
def science_marks_second_sem : ℕ := 90

def math_marks_first_sem : ℕ := math_marks_second_sem + 10
def arts_marks_first_sem : ℕ := arts_marks_second_sem - 15
def science_marks_first_sem : ℕ := science_marks_second_sem - (science_marks_second_sem / 3)

def total_marks_first_sem : ℕ := math_marks_first_sem + arts_marks_first_sem + science_marks_first_sem
def total_marks_second_sem : ℕ := math_marks_second_sem + arts_marks_second_sem + science_marks_second_sem

def total_marks_both_sems : ℕ := total_marks_first_sem + total_marks_second_sem

-- Theorem to be proved
theorem eva_total_marks_correct : total_marks_both_sems = 485 := by
  -- Here, we state that we need to prove the total marks sum up to 485
  sorry

end eva_total_marks_correct_l203_203292


namespace unique_arrangements_moon_l203_203899

theorem unique_arrangements_moon : 
  let word := ["M", "O", "O", "N"]
  let n := word.length
  n.factorial / (word.count (fun c => c = "O")).factorial = 12 :=
by
  let word := ["M", "O", "O", "N"]
  let n := word.length
  have h : n = 4 := rfl
  have hO : word.count (fun c => c = "O") = 2 := rfl
  calc
    n.factorial / (word.count (fun c => c = "O")).factorial
        = 4.factorial / 2.factorial : by rw [h, hO]
    ... = 24 / 2 : by norm_num
    ... = 12 : by norm_num

end unique_arrangements_moon_l203_203899


namespace common_ratio_geometric_sequence_l203_203694

theorem common_ratio_geometric_sequence (n : ℕ) :
  ∃ q : ℕ, (∀ k : ℕ, q = 4^(2*k+3) / 4^(2*k+1)) ∧ q = 16 :=
by
  use 16
  sorry

end common_ratio_geometric_sequence_l203_203694


namespace moses_more_than_esther_l203_203854

noncomputable theory

def total_amount : ℝ := 50
def moses_share_percentage : ℝ := 0.40
def moses_share : ℝ := moses_share_percentage * total_amount
def remainder : ℝ := total_amount - moses_share
def esther_share : ℝ := remainder / 2

theorem moses_more_than_esther : moses_share - esther_share = 5 :=
by
  -- Proof goes here
  sorry

end moses_more_than_esther_l203_203854


namespace line_equation_l203_203258

-- Define the conditions as given in the problem
def passes_through (P : ℝ × ℝ) (line : ℝ × ℝ) : Prop :=
  line.fst * P.fst + line.snd * P.snd + 1 = 0

def equal_intercepts (line : ℝ × ℝ) : Prop :=
  line.fst = line.snd

theorem line_equation (P : ℝ × ℝ) (hP : P = (-2, -1)) :
  (∃ (k : ℝ), passes_through P (1, -2 * k)) ∨ (∃ (m : ℝ), passes_through P (1, m) ∧ m = - 1) :=
sorry

end line_equation_l203_203258


namespace circles_intersect_if_and_only_if_l203_203334

noncomputable def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 10 * y + 1 = 0

noncomputable def circle2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 2 * y - m = 0

theorem circles_intersect_if_and_only_if (m : ℝ) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ↔ -1 < m ∧ m < 79 :=
by {
  sorry
}

end circles_intersect_if_and_only_if_l203_203334


namespace four_digit_number_count_l203_203142

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l203_203142


namespace chores_for_cartoon_time_l203_203504

def hours_to_minutes (hours : ℕ) : ℕ := hours * 60

def cartoons_to_chores (cartoon_minutes : ℕ) : ℕ := cartoon_minutes * 8 / 10

theorem chores_for_cartoon_time (h : ℕ) (h_eq : h = 2) : cartoons_to_chores (hours_to_minutes h) = 96 :=
by
  rw [h_eq, hours_to_minutes, cartoons_to_chores]
  -- steps demonstrating transformation from hours to minutes and calculation of chores will follow here
  sorry

end chores_for_cartoon_time_l203_203504


namespace four_digit_number_count_l203_203115

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l203_203115


namespace wilson_total_notebooks_l203_203966

def num_notebooks_per_large_pack : ℕ := 7
def num_large_packs_wilson_bought : ℕ := 7

theorem wilson_total_notebooks : num_large_packs_wilson_bought * num_notebooks_per_large_pack = 49 := 
by
  -- sorry used to skip the proof.
  sorry

end wilson_total_notebooks_l203_203966


namespace sixty_percent_of_40_greater_than_four_fifths_of_25_l203_203021

theorem sixty_percent_of_40_greater_than_four_fifths_of_25 :
  let x := (60 / 100 : ℝ) * 40
  let y := (4 / 5 : ℝ) * 25
  x - y = 4 :=
by
  sorry

end sixty_percent_of_40_greater_than_four_fifths_of_25_l203_203021


namespace moon_arrangements_l203_203900

theorem moon_arrangements : 
  let word := "MOON" 
  let n := 4 -- number of letters in "MOON"
  let repeats := 2 -- number of times 'O' appears
  fact n / fact repeats = 12 :=
by sorry

end moon_arrangements_l203_203900


namespace deepak_current_age_l203_203719

variable (A D : ℕ)

def ratio_condition : Prop := A * 5 = D * 2
def arun_future_age (A : ℕ) : Prop := A + 10 = 30

theorem deepak_current_age (h1 : ratio_condition A D) (h2 : arun_future_age A) : D = 50 := sorry

end deepak_current_age_l203_203719


namespace part1_solution_part2_solution_l203_203771

def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

theorem part1_solution (x : ℝ) : 
  f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2 := sorry

theorem part2_solution (a : ℝ) :
  (∃ x : ℝ, f x a < 2 * a) ↔ 3 < a := sorry

end part1_solution_part2_solution_l203_203771


namespace digit_Phi_l203_203709

theorem digit_Phi (Phi : ℕ) (h1 : 220 / Phi = 40 + 3 * Phi) : Phi = 4 :=
by
  sorry

end digit_Phi_l203_203709


namespace ellipse_foci_distance_l203_203756

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∃ (a b : ℝ), (a = 6) ∧ (b = 3) ∧ distance_between_foci a b = 6 * Real.sqrt 3 :=
by
  sorry

end ellipse_foci_distance_l203_203756


namespace width_of_channel_at_bottom_l203_203214

theorem width_of_channel_at_bottom
    (top_width : ℝ)
    (area : ℝ)
    (depth : ℝ)
    (b : ℝ)
    (H1 : top_width = 12)
    (H2 : area = 630)
    (H3 : depth = 70)
    (H4 : area = 0.5 * (top_width + b) * depth) :
    b = 6 := 
sorry

end width_of_channel_at_bottom_l203_203214


namespace average_weight_of_all_children_l203_203355

theorem average_weight_of_all_children 
    (boys_weight_avg : ℕ)
    (number_of_boys : ℕ)
    (girls_weight_avg : ℕ)
    (number_of_girls : ℕ)
    (tall_boy_weight : ℕ)
    (ht1 : boys_weight_avg = 155)
    (ht2 : number_of_boys = 8)
    (ht3 : girls_weight_avg = 130)
    (ht4 : number_of_girls = 6)
    (ht5 : tall_boy_weight = 175)
    : (boys_weight_avg * (number_of_boys - 1) + tall_boy_weight + girls_weight_avg * number_of_girls) / (number_of_boys + number_of_girls) = 146 :=
by
  sorry

end average_weight_of_all_children_l203_203355


namespace four_digit_numbers_count_l203_203104

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l203_203104


namespace four_digit_number_count_l203_203139

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l203_203139


namespace investment_ratio_l203_203357

noncomputable def ratio_A_B (profit : ℝ) (profit_C : ℝ) (ratio_A_C : ℝ) (ratio_C_A : ℝ) := 
  3 / 1

theorem investment_ratio (total_profit : ℝ) (C_profit : ℝ) (A_C_ratio : ℝ) (C_A_ratio : ℝ) :
  total_profit = 60000 → C_profit = 20000 → A_C_ratio = 3 / 2 → ratio_A_B total_profit C_profit A_C_ratio C_A_ratio = 3 / 1 :=
by 
  intros h1 h2 h3
  sorry

end investment_ratio_l203_203357


namespace lunch_cost_before_tip_l203_203411

theorem lunch_cost_before_tip (C : ℝ) (h : C + 0.20 * C = 60.24) : C = 50.20 :=
sorry

end lunch_cost_before_tip_l203_203411


namespace expenses_of_5_yuan_l203_203984

-- Define the given condition: income of 5 yuan is +5 yuan
def income (x : Int) : Int := x

-- Define the opposite relationship between income and expenses
def expenses (x : Int) : Int := -income x

-- Proof statement to show that expenses of 5 yuan are -5 yuan, given the above definitions
theorem expenses_of_5_yuan : expenses 5 = -5 := by
  -- The proof is not provided here, so we use sorry to indicate its place
  sorry

end expenses_of_5_yuan_l203_203984


namespace picture_books_count_l203_203845

-- Definitions based on the given conditions
def total_books : ℕ := 35
def fiction_books : ℕ := 5
def non_fiction_books : ℕ := fiction_books + 4
def autobiographies : ℕ := 2 * fiction_books
def total_non_picture_books : ℕ := fiction_books + non_fiction_books + autobiographies
def picture_books : ℕ := total_books - total_non_picture_books

-- Statement of the problem
theorem picture_books_count : picture_books = 11 :=
by sorry

end picture_books_count_l203_203845


namespace pascal_triangle_45th_number_l203_203011

theorem pascal_triangle_45th_number (n k : ℕ) (h1 : n = 47) (h2 : k = 44) : 
  Nat.choose (n - 1) k = 1035 :=
by
  sorry

end pascal_triangle_45th_number_l203_203011


namespace tangent_lines_range_l203_203331

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * Real.sqrt x

theorem tangent_lines_range (a : ℝ) :
  (∃ t1 t2 : ℝ, t1 ≠ t2 ∧ tangent_to f t1 g a ∧ tangent_to f t2 g a) ↔ 0 < a ∧ a < 2 :=
sorry

end tangent_lines_range_l203_203331


namespace nine_circles_problem_l203_203813

def is_triangle_valid (grid : Fin 3 × Fin 3 → ℕ) (triangles : list (list (Fin 3 × Fin 3))) (target_sum : ℕ) : Prop :=
  ∀ triangle ∈ triangles, target_sum = (triangle.map grid).sum

def unique_numbers_1_to_9 (grid : Fin 3 × Fin 3 → ℕ) : Prop :=
  let numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9} in
  (Finset.image grid Finset.univ).val = numbers

theorem nine_circles_problem : ∃ grid : (Fin 3 × Fin 3) → ℕ,
  unique_numbers_1_to_9 grid ∧
  is_triangle_valid grid
    [ [(0, 0), (0, 1), (0, 2)]
    , [(0, 0), (1, 0), (2, 0)]
    , [(0, 0), (1, 1), (2, 2)]
    , [(0, 1), (1, 1), (2, 1)]
    , [(0, 2), (1, 2), (2, 2)]
    , [(1, 0), (1, 1), (1, 2)]
    , [(2, 0), (2, 1), (2, 2)]
    ] 15 := sorry

end nine_circles_problem_l203_203813


namespace sum_with_probability_point_two_l203_203006

open Finset

def a : Finset ℤ := {2, 3, 4, 5}
def b : Finset ℤ := {4, 5, 6, 7, 8}

theorem sum_with_probability_point_two :
  ∃ sum, (a.product b).filter (λ p : ℤ × ℤ, p.1 + p.2 = sum).card = 4 :=
begin
  use 10,
  sorry -- proof to be filled out
end

end sum_with_probability_point_two_l203_203006


namespace total_shoes_count_l203_203256

-- Define the concepts and variables related to the conditions
def num_people := 10
def num_people_regular_shoes := 4
def num_people_sandals := 3
def num_people_slippers := 3
def num_shoes_regular := 2
def num_shoes_sandals := 1
def num_shoes_slippers := 1

-- Goal: Prove that the total number of shoes kept outside is 20
theorem total_shoes_count :
  (num_people_regular_shoes * num_shoes_regular) +
  (num_people_sandals * num_shoes_sandals * 2) +
  (num_people_slippers * num_shoes_slippers * 2) = 20 :=
by
  sorry

end total_shoes_count_l203_203256


namespace find_threedigit_number_l203_203457

-- Define the three-digit number and its reverse
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

-- Define the condition of adding the number and its reverse to get 1777
def number_sum_condition (a b c : ℕ) : Prop :=
  original_number a b c + reversed_number a b c = 1777

-- Prove the existence of digits a, b, and c that satisfy the conditions
theorem find_threedigit_number :
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  original_number a b c = 859 ∧ 
  reversed_number a b c = 958 ∧ 
  number_sum_condition a b c :=
sorry

end find_threedigit_number_l203_203457


namespace evaTotalMarksCorrect_l203_203296

-- Definition of marks scored by Eva in each subject across semesters
def evaMathsMarksSecondSemester : Nat := 80
def evaArtsMarksSecondSemester : Nat := 90
def evaScienceMarksSecondSemester : Nat := 90

def evaMathsMarksFirstSemester : Nat := evaMathsMarksSecondSemester + 10
def evaArtsMarksFirstSemester : Nat := evaArtsMarksSecondSemester - 15
def evaScienceMarksFirstSemester : Nat := evaScienceMarksSecondSemester - (evaScienceMarksSecondSemester / 3)

-- Total marks in each semester
def totalMarksFirstSemester : Nat := evaMathsMarksFirstSemester + evaArtsMarksFirstSemester + evaScienceMarksFirstSemester
def totalMarksSecondSemester : Nat := evaMathsMarksSecondSemester + evaArtsMarksSecondSemester + evaScienceMarksSecondSemester

-- Combined total
def evaTotalMarks : Nat := totalMarksFirstSemester + totalMarksSecondSemester

-- Statement to prove
theorem evaTotalMarksCorrect : evaTotalMarks = 485 := 
by
  -- This needs to be proved as per the conditions and calculations above
  sorry

end evaTotalMarksCorrect_l203_203296


namespace sin_one_lt_log3_sqrt7_l203_203715

open Real

theorem sin_one_lt_log3_sqrt7 : sin 1 < log 3 (sqrt 7) := 
sorry

end sin_one_lt_log3_sqrt7_l203_203715


namespace actual_distance_between_towns_l203_203034

def map_scale : ℕ := 600000
def distance_on_map : ℕ := 2

theorem actual_distance_between_towns :
  (distance_on_map * map_scale) / 100 / 1000 = 12 :=
by
  sorry

end actual_distance_between_towns_l203_203034


namespace percent_non_bikers_play_basketball_l203_203888

noncomputable def total_children (N : ℕ) : ℕ := N
def basketball_players (N : ℕ) : ℕ := 7 * N / 10
def bikers (N : ℕ) : ℕ := 4 * N / 10
def basketball_bikers (N : ℕ) : ℕ := 3 * basketball_players N / 10
def basketball_non_bikers (N : ℕ) : ℕ := basketball_players N - basketball_bikers N
def non_bikers (N : ℕ) : ℕ := N - bikers N

theorem percent_non_bikers_play_basketball (N : ℕ) :
  (basketball_non_bikers N * 100 / non_bikers N) = 82 :=
by sorry

end percent_non_bikers_play_basketball_l203_203888


namespace important_emails_l203_203684

theorem important_emails (total_emails : ℕ) (spam_frac : ℚ) (promotional_frac : ℚ) (spam_email_count : ℕ) (remaining_emails : ℕ) (promotional_email_count : ℕ) (important_email_count : ℕ) :
  total_emails = 800 ∧ spam_frac = 3 / 7 ∧ promotional_frac = 5 / 11 ∧ spam_email_count = 343 ∧ remaining_emails = 457 ∧ promotional_email_count = 208 →
sorry

end important_emails_l203_203684


namespace import_tax_applied_amount_l203_203259

theorem import_tax_applied_amount 
    (total_value : ℝ) 
    (import_tax_paid : ℝ)
    (tax_rate : ℝ) 
    (excess_amount : ℝ) 
    (condition1 : total_value = 2580) 
    (condition2 : import_tax_paid = 110.60) 
    (condition3 : tax_rate = 0.07) 
    (condition4 : import_tax_paid = tax_rate * (total_value - excess_amount)) : 
    excess_amount = 1000 :=
by
  sorry

end import_tax_applied_amount_l203_203259


namespace four_digit_numbers_count_l203_203165

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l203_203165


namespace remainder_when_divided_by_8_l203_203860

theorem remainder_when_divided_by_8 (x k : ℤ) (h : x = 63 * k + 27) : x % 8 = 3 :=
sorry

end remainder_when_divided_by_8_l203_203860


namespace no_real_solution_l203_203392

theorem no_real_solution :
  ¬ ∃ x : ℝ, (3 * x ^ 2 / (x - 2) - (5 * x + 4) / 4 + (10 - 9 * x) / (x - 2) + 2 = 0) :=
sorry

end no_real_solution_l203_203392


namespace students_failed_l203_203221

theorem students_failed (total_students : ℕ) (A_percentage : ℚ) (fraction_remaining_B_or_C : ℚ) :
  total_students = 32 → A_percentage = 0.25 → fraction_remaining_B_or_C = 1/4 →
  let students_A := total_students * A_percentage.to_nat in
  let remaining_students := total_students - students_A in
  let students_B_or_C := remaining_students * fraction_remaining_B_or_C.to_nat in
  let students_failed := remaining_students - students_B_or_C in
  students_failed = 18 :=
by
  intros
  simp [students_A, remaining_students, students_B_or_C]
  sorry

end students_failed_l203_203221


namespace cosine_of_3pi_over_2_l203_203050

theorem cosine_of_3pi_over_2 : Real.cos (3 * Real.pi / 2) = 0 := by
  sorry

end cosine_of_3pi_over_2_l203_203050


namespace eval_expr_l203_203048

theorem eval_expr (b c : ℕ) (hb : b = 2) (hc : c = 5) : b^3 * b^4 * c^2 = 3200 :=
by {
  -- the proof is omitted
  sorry
}

end eval_expr_l203_203048


namespace intersection_is_singleton_l203_203659

namespace ProofProblem

def M : Set ℤ := {-3, -2, -1}

def N : Set ℤ := {x : ℤ | (x + 2) * (x - 3) < 0}

theorem intersection_is_singleton : M ∩ N = {-1} :=
by
  sorry

end ProofProblem

end intersection_is_singleton_l203_203659


namespace prime_constraint_unique_solution_l203_203058

theorem prime_constraint_unique_solution (p x y : ℕ) (h_prime : Prime p)
  (h1 : p + 1 = 2 * x^2)
  (h2 : p^2 + 1 = 2 * y^2) :
  p = 7 :=
by
  sorry

end prime_constraint_unique_solution_l203_203058


namespace intersection_A_complement_B_l203_203378

-- Definition of the universal set U
def U : Set ℝ := Set.univ

-- Definition of the set A
def A : Set ℝ := {x | x^2 - 2 * x < 0}

-- Definition of the set B
def B : Set ℝ := {x | x > 1}

-- Definition of the complement of B in U
def complement_B : Set ℝ := {x | x ≤ 1}

-- The intersection A ∩ complement_B
def intersection : Set ℝ := {x | 0 < x ∧ x ≤ 1}

-- The theorem to prove
theorem intersection_A_complement_B : A ∩ complement_B = intersection :=
by
  -- Proof goes here
  sorry

end intersection_A_complement_B_l203_203378


namespace jenny_stamps_l203_203677

theorem jenny_stamps :
  let num_books := 8
  let pages_per_book := 42
  let stamps_per_page := 6
  let new_stamps_per_page := 10
  let complete_books_in_new_system := 4
  let pages_in_fifth_book := 33
  (num_books * pages_per_book * stamps_per_page) % new_stamps_per_page = 6 :=
by
  sorry

end jenny_stamps_l203_203677


namespace cups_of_sugar_l203_203381

theorem cups_of_sugar (flour_total flour_added sugar : ℕ) (h₁ : flour_total = 10) (h₂ : flour_added = 7) (h₃ : flour_total - flour_added = sugar + 1) :
  sugar = 2 :=
by
  sorry

end cups_of_sugar_l203_203381


namespace seats_usually_taken_l203_203842

def total_tables : Nat := 15
def seats_per_table : Nat := 10
def proportion_left_unseated : Rat := 1 / 10
def proportion_taken : Rat := 1 - proportion_left_unseated

theorem seats_usually_taken :
  proportion_taken * (total_tables * seats_per_table) = 135 := by
  sorry

end seats_usually_taken_l203_203842


namespace quad_root_sum_product_l203_203328

theorem quad_root_sum_product (α β : ℝ) (h₁ : α ≠ β) (h₂ : α * α - 5 * α - 2 = 0) (h₃ : β * β - 5 * β - 2 = 0) : 
  α + β + α * β = 3 := 
by
  sorry

end quad_root_sum_product_l203_203328


namespace distinct_integers_sum_l203_203525

theorem distinct_integers_sum (m n p q : ℕ) (h1 : m ≠ n) (h2 : m ≠ p) (h3 : m ≠ q) (h4 : n ≠ p)
  (h5 : n ≠ q) (h6 : p ≠ q) (h71 : m > 0) (h72 : n > 0) (h73 : p > 0) (h74 : q > 0)
  (h_eq : (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4) : m + n + p + q = 28 := by
  sorry

end distinct_integers_sum_l203_203525


namespace range_of_cos_neg_alpha_l203_203174

theorem range_of_cos_neg_alpha (α : ℝ) (h : 12 * (Real.sin α)^2 + Real.cos α > 11) :
  -1 / 4 < Real.cos (-α) ∧ Real.cos (-α) < 1 / 3 := 
sorry

end range_of_cos_neg_alpha_l203_203174


namespace four_digit_number_count_l203_203137

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l203_203137


namespace count_four_digit_numbers_l203_203144

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l203_203144


namespace calls_on_friday_l203_203793

noncomputable def total_calls_monday := 35
noncomputable def total_calls_tuesday := 46
noncomputable def total_calls_wednesday := 27
noncomputable def total_calls_thursday := 61
noncomputable def average_calls_per_day := 40
noncomputable def number_of_days := 5
noncomputable def total_calls_week := average_calls_per_day * number_of_days

theorem calls_on_friday : 
  total_calls_week - (total_calls_monday + total_calls_tuesday + total_calls_wednesday + total_calls_thursday) = 31 :=
by
  sorry

end calls_on_friday_l203_203793


namespace arnel_number_of_boxes_l203_203269

def arnel_kept_pencils : ℕ := 10
def number_of_friends : ℕ := 5
def pencils_per_friend : ℕ := 8
def pencils_per_box : ℕ := 5

theorem arnel_number_of_boxes : ∃ (num_boxes : ℕ), 
  (number_of_friends * pencils_per_friend) + arnel_kept_pencils = num_boxes * pencils_per_box ∧ 
  num_boxes = 10 := sorry

end arnel_number_of_boxes_l203_203269


namespace interest_rate_first_part_l203_203608

theorem interest_rate_first_part 
  (total_amount : ℤ) 
  (amount_at_first_rate : ℤ) 
  (amount_at_second_rate : ℤ) 
  (rate_second_part : ℤ) 
  (total_annual_interest : ℤ) 
  (r : ℤ) 
  (h_split : total_amount = amount_at_first_rate + amount_at_second_rate) 
  (h_second : rate_second_part = 5)
  (h_interest : (amount_at_first_rate * r) / 100 + (amount_at_second_rate * rate_second_part) / 100 = total_annual_interest) :
  r = 3 := 
by 
  sorry

end interest_rate_first_part_l203_203608


namespace five_term_geometric_sequence_value_of_b_l203_203181

theorem five_term_geometric_sequence_value_of_b (a b c : ℝ) (h₁ : b ^ 2 = 81) (h₂ : a ^ 2 = b) (h₃ : 1 * a = a) (h₄ : c * c = c) :
  b = 9 :=
by 
  sorry

end five_term_geometric_sequence_value_of_b_l203_203181


namespace union_of_A_and_B_l203_203081

open Set

def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem union_of_A_and_B : A ∪ B = {2, 3, 5, 6} := sorry

end union_of_A_and_B_l203_203081


namespace find_m_l203_203932

def a (m : ℝ) : ℝ × ℝ := (2 * m - 1, 3)
def b : ℝ × ℝ := (1, -1)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_m (m : ℝ) (h : dot_product (a m) b = 2) : m = 3 :=
by sorry

end find_m_l203_203932


namespace probability_x_plus_y_lt_4_l203_203472

theorem probability_x_plus_y_lt_4 :
  let square_vertices := {(0,0), (0,3), (3,3), (3,0)} in
  let point_in_square (x y : ℝ) := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 in
  let probability (A : ℝ) (B : ℝ) := A / B in
  ∃ (P : ℝ), P = probability 7 9 ∧
             P = (measure (λ (x y : ℝ), point_in_square x y ∧ x + y < 4)) / (measure (λ (x y : ℝ), point_in_square x y)) :=
sorry

end probability_x_plus_y_lt_4_l203_203472


namespace gcd_360_150_l203_203421

theorem gcd_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end gcd_360_150_l203_203421


namespace subsets_with_intersection_property_l203_203387

open Set

theorem subsets_with_intersection_property :
  ∃ (A : Fin 16 → Set ℕ), (∀ z ∈ {x : ℕ | x <= 10000}, ∃ (B : Finset (Fin 16)), B.card = 8 ∧ (z ∈ ⋂ i ∈ B, A i)) :=
begin
  sorry
end

end subsets_with_intersection_property_l203_203387


namespace library_books_difference_l203_203572

theorem library_books_difference :
  let books_old_town := 750
  let books_riverview := 1240
  let books_downtown := 1800
  let books_eastside := 1620
  books_downtown - books_old_town = 1050 :=
by
  sorry

end library_books_difference_l203_203572


namespace picture_books_count_l203_203847

theorem picture_books_count (total_books : ℕ) (fiction_books : ℕ) (non_fiction_books : ℕ) (autobiography_books : ℕ) (picture_books : ℕ) 
  (h1 : total_books = 35)
  (h2 : fiction_books = 5)
  (h3 : non_fiction_books = fiction_books + 4)
  (h4 : autobiography_books = 2 * fiction_books)
  (h5 : picture_books = total_books - (fiction_books + non_fiction_books + autobiography_books)) :
  picture_books = 11 := 
  sorry

end picture_books_count_l203_203847


namespace sides_ratio_of_arithmetic_sequence_l203_203770

theorem sides_ratio_of_arithmetic_sequence (A B C : ℝ) (a b c : ℝ) 
  (h_arith_sequence : (A = B - (B - C)) ∧ (B = C + (C - A))) 
  (h_angle_B : B = 60)  
  (h_cosine_rule : a^2 + c^2 - b^2 = 2 * a * c * (Real.cos B)) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end sides_ratio_of_arithmetic_sequence_l203_203770


namespace sum_infinite_geometric_series_l203_203624

theorem sum_infinite_geometric_series : 
  let a : ℝ := 2
  let r : ℝ := -5/8
  a / (1 - r) = 16/13 :=
by
  sorry

end sum_infinite_geometric_series_l203_203624


namespace initial_amount_of_liquid_A_l203_203257

theorem initial_amount_of_liquid_A (A B : ℕ) (x : ℕ) (h1 : 4 * x = A) (h2 : x = B) (h3 : 4 * x + x = 5 * x)
    (h4 : 4 * x - 8 = 3 * (x + 8) / 2) : A = 16 :=
  by
  sorry

end initial_amount_of_liquid_A_l203_203257


namespace evaluate_stability_of_yields_l203_203706

def variance (l : List ℝ) : ℝ :=
l.map (λ x, (x - l.sum / l.length)^2).sum / l.length

theorem evaluate_stability_of_yields (x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 x_9 x_{10} : ℝ) :
  let yields := [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_{10}] in
  let mean := yields.sum / yields.length in
  variance yields = (yields.map (λ x, (x - mean)^2)).sum / yields.length :=
  sorry

end evaluate_stability_of_yields_l203_203706


namespace value_of_a_l203_203242

theorem value_of_a (a : ℝ) (h : a = -a) : a = 0 :=
by
  sorry

end value_of_a_l203_203242


namespace find_xy_l203_203304

theorem find_xy (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end find_xy_l203_203304


namespace not_divisible_by_24_l203_203883

theorem not_divisible_by_24 : 
  ¬ (121416182022242628303234 % 24 = 0) := 
by
  sorry

end not_divisible_by_24_l203_203883


namespace calculate_expression_l203_203043

theorem calculate_expression (a b c d : ℤ) (h1 : 3^0 = 1) (h2 : (-1 / 2 : ℚ)^(-2 : ℤ) = 4) : 
  (202 : ℤ) * 3^0 + (-1 / 2 : ℚ)^(-2 : ℤ) = 206 :=
by
  sorry

end calculate_expression_l203_203043


namespace domain_lg_function_l203_203832

theorem domain_lg_function (x : ℝ) : (1 + x > 0 ∧ x - 1 > 0) ↔ (1 < x) :=
by
  sorry

end domain_lg_function_l203_203832


namespace least_sugar_l203_203493

theorem least_sugar (f s : ℚ) (h1 : f ≥ 10 + 3 * s / 4) (h2 : f ≤ 3 * s) :
  s ≥ 40 / 9 :=
  sorry

end least_sugar_l203_203493


namespace find_divided_number_l203_203814

-- Declare the constants and assumptions
variables (d q r : ℕ)
variables (n : ℕ)
variables (h_d : d = 20)
variables (h_q : q = 6)
variables (h_r : r = 2)
variables (h_def : n = d * q + r)

-- State the theorem we want to prove
theorem find_divided_number : n = 122 :=
by
  sorry

end find_divided_number_l203_203814


namespace simplify_fraction_l203_203823

theorem simplify_fraction (a : ℝ) (h : a = 2) : (24 * a^5) / (72 * a^3) = 4 / 3 := by
  sorry

end simplify_fraction_l203_203823


namespace kylie_coins_l203_203543

open Nat

theorem kylie_coins :
  ∀ (coins_from_piggy_bank coins_from_brother coins_from_father coins_given_to_friend total_coins_left : ℕ),
  coins_from_piggy_bank = 15 →
  coins_from_brother = 13 →
  coins_from_father = 8 →
  coins_given_to_friend = 21 →
  total_coins_left = coins_from_piggy_bank + coins_from_brother + coins_from_father - coins_given_to_friend →
  total_coins_left = 15 :=
by
  intros
  sorry

end kylie_coins_l203_203543


namespace greatest_power_of_3_l203_203233

theorem greatest_power_of_3 (n : ℕ) : 
  (n = 603) → 
  3^603 ∣ (15^n - 6^n + 3^n) ∧ ¬ (3^(603+1) ∣ (15^n - 6^n + 3^n)) :=
by
  intro hn
  cases hn
  sorry

end greatest_power_of_3_l203_203233


namespace total_points_l203_203283

theorem total_points (darius_score marius_score matt_score total_points : ℕ) 
    (h1 : darius_score = 10) 
    (h2 : marius_score = darius_score + 3) 
    (h3 : matt_score = darius_score + 5) 
    (h4 : total_points = darius_score + marius_score + matt_score) : 
    total_points = 38 :=
by sorry

end total_points_l203_203283


namespace number_of_integers_congruent_to_3_mod_7_less_than_500_l203_203339

theorem number_of_integers_congruent_to_3_mod_7_less_than_500 : 
  {n : ℕ | n < 500 ∧ n % 7 = 3}.card = 71 :=
sorry

end number_of_integers_congruent_to_3_mod_7_less_than_500_l203_203339


namespace proof_statement_l203_203415

noncomputable def problem_statement (a b : ℤ) : ℤ :=
  (a^3 + b^3) / (a^2 - a * b + b^2)

theorem proof_statement : problem_statement 5 4 = 9 := by
  sorry

end proof_statement_l203_203415


namespace Sahil_transportation_charges_l203_203389

theorem Sahil_transportation_charges
  (cost_machine : ℝ)
  (cost_repair : ℝ)
  (actual_selling_price : ℝ)
  (profit_percentage : ℝ)
  (transportation_charges : ℝ)
  (h1 : cost_machine = 12000)
  (h2 : cost_repair = 5000)
  (h3 : profit_percentage = 0.50)
  (h4 : actual_selling_price = 27000)
  (h5 : transportation_charges + (cost_machine + cost_repair) * (1 + profit_percentage) = actual_selling_price) :
  transportation_charges = 1500 :=
by
  sorry

end Sahil_transportation_charges_l203_203389


namespace solve_for_x_l203_203443

theorem solve_for_x (x : ℝ) : (0.25 * x = 0.15 * 1500 - 20) → x = 820 :=
by
  intro h
  sorry

end solve_for_x_l203_203443


namespace equation_B_no_solution_l203_203861

theorem equation_B_no_solution : ¬ ∃ x : ℝ, |-2 * x| + 6 = 0 :=
by
  sorry

end equation_B_no_solution_l203_203861


namespace max_area_circle_center_l203_203943

theorem max_area_circle_center (k : ℝ) :
  (∃ (x y : ℝ), (x + k / 2)^2 + (y + 1)^2 = 1 - 3 / 4 * k^2 ∧ k = 0) →
  x = 0 ∧ y = -1 :=
sorry

end max_area_circle_center_l203_203943


namespace sin_double_angle_l203_203075

noncomputable def unit_circle_point :=
  (1 / 2, Real.sqrt (1 - (1 / 2) ^ 2))

theorem sin_double_angle 
  (α : Real)
  (h1 : (1 / 2, Real.sqrt (1 - (1 / 2) ^ 2)) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 })
  (h2 : α = (Real.arccos (1 / 2)) ∨ α = -(Real.arccos (1 / 2))) :
  Real.sin (π / 2 + 2 * α) = -1 / 2 :=
by
  sorry

end sin_double_angle_l203_203075


namespace prove_a_5_l203_203702

noncomputable def a_5_proof : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ),
    (∀ n, a n > 0) → 
    (a 1 + 2 * a 2 = 4) →
    ((a 1)^2 * q^6 = 4 * a 1 * q^2 * a 1 * q^6) →
    a 5 = 1 / 8

theorem prove_a_5 : a_5_proof := sorry

end prove_a_5_l203_203702


namespace exists_maximal_arithmetic_progression_l203_203401

open Nat

def is_arithmetic_progression (s : List ℚ) :=
  ∃ d : ℚ, ∀ i ∈ (List.range (s.length - 1)), s[i + 1] - s[i] = d

def is_maximal_arithmetic_progression (s : List ℚ) (S : Set ℚ) :=
  is_arithmetic_progression s ∧
  (∀ t : List ℚ, t ≠ s → is_arithmetic_progression t → (s ⊆ t → ¬(t ⊆ (insert (s.head - s[1] + s[0]) (insert (s.last + s[1] - s[0]) S))))

theorem exists_maximal_arithmetic_progression 
  (n : ℕ) (S : Set ℚ) 
  (hS : ∀ m : ℕ, (1:ℚ) / m ∈ S) : 
  ∃ s : List ℚ, s.length = n ∧ is_maximal_arithmetic_progression s S := 
sorry

end exists_maximal_arithmetic_progression_l203_203401


namespace eva_total_marks_l203_203293

theorem eva_total_marks
    (math_score_s2 : ℕ) (arts_score_s2 : ℕ) (science_score_s2 : ℕ)
    (math_diff : ℕ) (arts_diff : ℕ) (science_frac_diff : ℚ)
    (math_score_s2_eq : math_score_s2 = 80)
    (arts_score_s2_eq : arts_score_s2 = 90)
    (science_score_s2_eq : science_score_s2 = 90)
    (math_diff_eq : math_diff = 10)
    (arts_diff_eq : arts_diff = 15)
    (science_frac_diff_eq : science_frac_diff = 1/3) : 
  (math_score_s2 + 10 + (math_score_s2 + math_diff) + 
   (arts_score_s2 + 90 - 15) + (arts_score_s2 + arts_diff) + 
   (science_score_s2 + 90 - (1/3) * 90) + (science_score_s2 + science_score_s2 * 1/3)) = 485 := 
by
  sorry

end eva_total_marks_l203_203293


namespace number_of_four_digit_numbers_l203_203123

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l203_203123


namespace largest_multiple_of_45_l203_203697

theorem largest_multiple_of_45 (m : ℕ) 
  (h₁ : m % 45 = 0) 
  (h₂ : ∀ d : ℕ, d ∈ m.digits 10 → d = 8 ∨ d = 0) : 
  m / 45 = 197530 := 
sorry

end largest_multiple_of_45_l203_203697


namespace number_of_four_digit_numbers_l203_203125

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l203_203125


namespace math_problem_l203_203779

theorem math_problem (a : ℝ) (h : a^2 - 4 * a + 3 = 0) (h_ne : a ≠ 2 ∧ a ≠ 3 ∧ a ≠ -3) :
  (9 - 3 * a) / (2 * a - 4) / (a + 2 - 5 / (a - 2)) = -3 / 8 :=
sorry

end math_problem_l203_203779


namespace smallest_m_l203_203175

theorem smallest_m (m : ℤ) (h : 2 * m + 1 ≥ 0) : m ≥ 0 :=
sorry

end smallest_m_l203_203175


namespace bob_eats_10_apples_l203_203229

variable (B C : ℕ)
variable (h1 : B + C = 30)
variable (h2 : C = 2 * B)

theorem bob_eats_10_apples : B = 10 :=
by sorry

end bob_eats_10_apples_l203_203229


namespace f_at_2_l203_203644

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x

theorem f_at_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -10 :=
by 
  sorry

end f_at_2_l203_203644


namespace sofa_price_is_correct_l203_203366

def price_sofa (invoice_total armchair_price table_price : ℕ) (armchair_count : ℕ) : ℕ :=
  invoice_total - (armchair_price * armchair_count + table_price)

theorem sofa_price_is_correct
  (invoice_total : ℕ)
  (armchair_price : ℕ)
  (table_price : ℕ)
  (armchair_count : ℕ)
  (sofa_price : ℕ)
  (h_invoice : invoice_total = 2430)
  (h_armchair_price : armchair_price = 425)
  (h_table_price : table_price = 330)
  (h_armchair_count : armchair_count = 2)
  (h_sofa_price : sofa_price = 1250) :
  price_sofa invoice_total armchair_price table_price armchair_count = sofa_price :=
by
  sorry

end sofa_price_is_correct_l203_203366


namespace max_correct_answers_l203_203596

theorem max_correct_answers (c w b : ℕ) 
  (h1 : c + w + b = 25) 
  (h2 : 5 * c - 2 * w = 60) : 
  c ≤ 14 := 
sorry

end max_correct_answers_l203_203596


namespace sum_first_n_terms_arithmetic_sequence_l203_203069

theorem sum_first_n_terms_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (a 2 + a 4 = 10) ∧ (∀ n : ℕ, a (n + 1) - a n = 2) → 
  (∀ n : ℕ, S n = n^2) := by
  intro h
  sorry

end sum_first_n_terms_arithmetic_sequence_l203_203069


namespace smallest_natrural_number_cube_ends_888_l203_203444

theorem smallest_natrural_number_cube_ends_888 :
  ∃ n : ℕ, (n^3 % 1000 = 888) ∧ (∀ m : ℕ, (m^3 % 1000 = 888) → n ≤ m) := 
sorry

end smallest_natrural_number_cube_ends_888_l203_203444


namespace minimum_m_l203_203769

/-
  Given that for all 2 ≤ x ≤ 3, 3 ≤ y ≤ 6, the inequality mx^2 - xy + y^2 ≥ 0 always holds,
  prove that the minimum value of the real number m is 0.
-/
theorem minimum_m (m : ℝ) :
  (∀ x y : ℝ, 2 ≤ x ∧ x ≤ 3 → 3 ≤ y ∧ y ≤ 6 → m * x^2 - x * y + y^2 ≥ 0) → m = 0 :=
sorry -- proof to be provided

end minimum_m_l203_203769


namespace find_x_for_perpendicular_and_parallel_l203_203087

noncomputable def a : ℝ × ℝ × ℝ := (2, -1, 3)
noncomputable def b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def parallel (u v : ℝ × ℝ × ℝ) : Prop := (u.1 / v.1 = u.2 / v.2) ∧ (u.2 / v.2 = u.3 / v.3)

theorem find_x_for_perpendicular_and_parallel :
  (dot_product a (b (10/3)) = 0) ∧ (parallel a (b (-6))) :=
by
  sorry

end find_x_for_perpendicular_and_parallel_l203_203087


namespace wheel_speed_l203_203374

theorem wheel_speed (s : ℝ) (t : ℝ) :
  (12 / 5280) * 3600 = s * t →
  (12 / 5280) * 3600 = (s + 4) * (t - (1 / 18000)) →
  s = 8 :=
by
  intro h1 h2
  sorry

end wheel_speed_l203_203374


namespace value_of_x_l203_203933

theorem value_of_x (x : ℝ) (h : ∃ k < 0, (x, 1) = k • (4, x)) : x = -2 :=
sorry

end value_of_x_l203_203933


namespace particular_solution_exists_l203_203957

noncomputable def general_solution (C : ℝ) (x : ℝ) : ℝ := C * x + 1

def differential_equation (x y y' : ℝ) : Prop := x * y' = y - 1

def initial_condition (y : ℝ) : Prop := y = 5

theorem particular_solution_exists :
  (∀ C x y, y = general_solution C x → differential_equation x y (C : ℝ)) →
  (∃ C, initial_condition (general_solution C 1)) →
  (∀ x, ∃ y, y = general_solution 4 x) :=
by
  intros h1 h2
  sorry

end particular_solution_exists_l203_203957


namespace least_subtracted_number_correct_l203_203723

noncomputable def least_subtracted_number (n : ℕ) : ℕ :=
  n - 13

theorem least_subtracted_number_correct (n : ℕ) : 
  least_subtracted_number 997 = 997 - 13 ∧
  (least_subtracted_number 997 % 5 = 3) ∧
  (least_subtracted_number 997 % 9 = 3) ∧
  (least_subtracted_number 997 % 11 = 3) :=
by
  let x := 997 - 13
  have : x = 984 := rfl
  have h5 : x % 5 = 3 := by sorry
  have h9 : x % 9 = 3 := by sorry
  have h11 : x % 11 = 3 := by sorry
  exact ⟨rfl, h5, h9, h11⟩

end least_subtracted_number_correct_l203_203723


namespace not_divisible_by_121_l203_203557

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 2 * n + 12)) :=
sorry

end not_divisible_by_121_l203_203557


namespace probability_of_x_plus_y_less_than_4_l203_203475

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l203_203475


namespace final_price_l203_203880

variable (OriginalPrice : ℝ)

def salePrice (OriginalPrice : ℝ) : ℝ :=
  0.6 * OriginalPrice

def priceAfterCoupon (SalePrice : ℝ) : ℝ :=
  0.75 * SalePrice

theorem final_price (OriginalPrice : ℝ) :
  priceAfterCoupon (salePrice OriginalPrice) = 0.45 * OriginalPrice := by
  sorry

end final_price_l203_203880


namespace true_propositions_count_l203_203657

theorem true_propositions_count (b : ℤ) :
  (b = 3 → b^2 = 9) → 
  (∃! p : Prop, p = (b^2 ≠ 9 → b ≠ 3) ∨ p = (b ≠ 3 → b^2 ≠ 9) ∨ p = (b^2 = 9 → b = 3) ∧ (p = (b^2 ≠ 9 → b ≠ 3))) :=
sorry

end true_propositions_count_l203_203657


namespace find_result_of_adding_8_l203_203577

theorem find_result_of_adding_8 (x : ℕ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end find_result_of_adding_8_l203_203577


namespace reciprocal_inequality_of_negatives_l203_203767

variable (a b : ℝ)

/-- Given that a < b < 0, prove that 1/a > 1/b. -/
theorem reciprocal_inequality_of_negatives (h1 : a < b) (h2 : b < 0) : (1/a) > (1/b) :=
sorry

end reciprocal_inequality_of_negatives_l203_203767


namespace geometric_sequence_seventh_term_l203_203563

theorem geometric_sequence_seventh_term (a r : ℝ) (ha : 0 < a) (hr : 0 < r) 
  (h4 : a * r^3 = 16) (h10 : a * r^9 = 2) : 
  a * r^6 = 2 :=
by
  sorry

end geometric_sequence_seventh_term_l203_203563


namespace range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l203_203765

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * Real.log x

def p (m : ℝ) : Prop :=
  ∀ x ∈ (Set.Ioo m (m + 1)), (x - 9 / x) < 0

def q (m : ℝ) : Prop :=
  m > 1 ∧ m < 3

theorem range_of_m_when_p_true :
  ∀ m : ℝ, p m → 0 ≤ m ∧ m ≤ 2 :=
sorry

theorem range_of_m_when_p_and_q_false_p_or_q_true :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (0 ≤ m ∧ m ≤ 1) ∨ (2 < m ∧ m < 3) :=
sorry

end range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l203_203765


namespace algebraic_expression_value_l203_203643

-- Define the given condition as a predicate
def condition (a : ℝ) := a^2 + a - 4 = 0

-- Then the goal to prove with the given condition
theorem algebraic_expression_value (a : ℝ) (h : condition a) : (a^2 - 3) * (a + 2) = -2 :=
sorry

end algebraic_expression_value_l203_203643


namespace charles_richard_difference_in_dimes_l203_203044

variable (q : ℕ)

-- Charles' quarters
def charles_quarters : ℕ := 5 * q + 1

-- Richard's quarters
def richard_quarters : ℕ := q + 5

-- Difference in quarters
def diff_quarters : ℕ := charles_quarters q - richard_quarters q

-- Difference in dimes
def diff_dimes : ℕ := (diff_quarters q) * 5 / 2

theorem charles_richard_difference_in_dimes : diff_dimes q = 10 * (q - 1) := by
  sorry

end charles_richard_difference_in_dimes_l203_203044


namespace gcd_yz_min_value_l203_203342

theorem gcd_yz_min_value (x y z : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) 
  (hxy_gcd : Nat.gcd x y = 224) (hxz_gcd : Nat.gcd x z = 546) : 
  Nat.gcd y z = 14 := 
sorry

end gcd_yz_min_value_l203_203342


namespace indeterminate_equation_solution_exists_l203_203649

theorem indeterminate_equation_solution_exists
  (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * c = b^2 + b + 1) :
  ∃ x y : ℤ, a * x^2 - (2 * b + 1) * x * y + c * y^2 = 1 := by
  sorry

end indeterminate_equation_solution_exists_l203_203649


namespace count_four_digit_numbers_l203_203147

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l203_203147


namespace factorization_of_2210_l203_203170

theorem factorization_of_2210 : 
  ∃! (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 2210) :=
sorry

end factorization_of_2210_l203_203170


namespace smallest_square_perimeter_of_isosceles_triangle_with_composite_sides_l203_203885

def is_composite (n : ℕ) : Prop := (∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_square_perimeter_of_isosceles_triangle_with_composite_sides :
  ∃ a b : ℕ,
    is_composite a ∧
    is_composite b ∧
    (2 * a + b) ^ 2 = 256 :=
sorry

end smallest_square_perimeter_of_isosceles_triangle_with_composite_sides_l203_203885


namespace sum_mod_condition_l203_203979

theorem sum_mod_condition (a b c : ℤ) (h1 : a * b * c % 7 = 2)
                          (h2 : 3 * c % 7 = 1)
                          (h3 : 4 * b % 7 = (2 + b) % 7) :
                          (a + b + c) % 7 = 3 := by
  sorry

end sum_mod_condition_l203_203979


namespace range_of_a_l203_203080

-- Define the propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 0 ≤ x → x ≤ 1 → a ≥ Real.exp x
def q (a : ℝ) := ∃ x : ℝ, x^2 + 4 * x + a = 0

-- The proof statement
theorem range_of_a (a : ℝ) : (p a ∧ q a) → a ∈ Set.Icc (Real.exp 1) 4 := by
  intro h
  sorry

end range_of_a_l203_203080


namespace original_average_l203_203396

theorem original_average (A : ℝ) (h : (2 * (12 * A)) / 12 = 100) : A = 50 :=
by
  sorry

end original_average_l203_203396


namespace solve_for_a_l203_203074

open Complex

theorem solve_for_a (a : ℝ) (h : ∃ x : ℝ, (2 * Complex.I - (a * Complex.I) / (1 - Complex.I) = x)) : a = 4 := 
sorry

end solve_for_a_l203_203074


namespace total_points_scored_l203_203278

theorem total_points_scored 
  (darius_score : ℕ) 
  (marius_score : ℕ) 
  (matt_score : ℕ) 
  (h1 : marius_score = darius_score + 3) 
  (h2 : darius_score = matt_score - 5)
  (h3 : darius_score = 10) : darius_score + marius_score + matt_score = 38 :=
by
  sorry

end total_points_scored_l203_203278


namespace simplify_expression_l203_203755

noncomputable def expression : ℝ :=
  (4 * (Real.sqrt 3 + Real.sqrt 7)) / (5 * Real.sqrt (3 + (1 / 2)))

theorem simplify_expression : expression = (16 + 8 * Real.sqrt 21) / 35 := by
  sorry

end simplify_expression_l203_203755


namespace expenses_of_five_yuan_l203_203996

theorem expenses_of_five_yuan (income_denotation : ℤ) (opposite_effect : ∀ x : ℤ, -x) :
  income_denotation = 5 → opposite_effect income_denotation = -5 :=
by sorry

end expenses_of_five_yuan_l203_203996


namespace cube_edge_length_l203_203562

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 96) : ∃ (edge_length : ℝ), edge_length = 4 := 
by 
  sorry

end cube_edge_length_l203_203562


namespace three_digit_sum_reverse_eq_l203_203453

theorem three_digit_sum_reverse_eq :
  ∃ (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9),
    101 * (a + c) + 20 * b = 1777 ∧ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (9, 7, 8) :=
by
  sorry

end three_digit_sum_reverse_eq_l203_203453


namespace expenses_of_five_yuan_l203_203998

theorem expenses_of_five_yuan (income_denotation : ℤ) (opposite_effect : ∀ x : ℤ, -x) :
  income_denotation = 5 → opposite_effect income_denotation = -5 :=
by sorry

end expenses_of_five_yuan_l203_203998


namespace arithmetic_problem_l203_203245

theorem arithmetic_problem :
  12.1212 + 17.0005 - 9.1103 = 20.0114 :=
sorry

end arithmetic_problem_l203_203245


namespace molly_takes_180_minutes_longer_l203_203716

noncomputable def time_for_Xanthia (pages_per_hour : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / pages_per_hour

noncomputable def time_for_Molly (pages_per_hour : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / pages_per_hour

theorem molly_takes_180_minutes_longer (pages : ℕ) (Xanthia_speed : ℕ) (Molly_speed : ℕ) :
  (time_for_Molly Molly_speed pages - time_for_Xanthia Xanthia_speed pages) * 60 = 180 :=
by
  -- Definitions specific to problem conditions
  let pages := 360
  let Xanthia_speed := 120
  let Molly_speed := 60

  -- Placeholder for actual proof
  sorry

end molly_takes_180_minutes_longer_l203_203716


namespace determine_a_l203_203663

theorem determine_a :
  ∃ (a b c d : ℕ), 
  (18 ^ a) * (9 ^ (4 * a - 1)) * (27 ^ c) = (2 ^ 6) * (3 ^ b) * (7 ^ d) ∧ 
  a * c = 4 / (2 * b + d) ∧ 
  b^2 - 4 * a * c = d ∧ 
  a = 6 := 
by
  sorry

end determine_a_l203_203663


namespace total_amount_l203_203450

theorem total_amount (A B C T : ℝ)
  (h1 : A = 1 / 4 * (B + C))
  (h2 : B = 3 / 5 * (A + C))
  (h3 : A = 20) :
  T = A + B + C → T = 100 := by
  sorry

end total_amount_l203_203450


namespace triangle_angle_sum_l203_203674

theorem triangle_angle_sum {x : ℝ} (h : 60 + 5 * x + 3 * x = 180) : x = 15 :=
by
  sorry

end triangle_angle_sum_l203_203674


namespace peter_invested_for_3_years_l203_203555

-- Definitions of parameters
def P : ℝ := 650
def APeter : ℝ := 815
def ADavid : ℝ := 870
def tDavid : ℝ := 4

-- Simple interest formula for Peter
def simple_interest_peter (r : ℝ) (t : ℝ) : Prop :=
  APeter = P + P * r * t

-- Simple interest formula for David
def simple_interest_david (r : ℝ) : Prop :=
  ADavid = P + P * r * tDavid

-- The main theorem to find out how many years Peter invested his money
theorem peter_invested_for_3_years : ∃ t : ℝ, (∃ r : ℝ, simple_interest_peter r t ∧ simple_interest_david r) ∧ t = 3 :=
by
  sorry

end peter_invested_for_3_years_l203_203555


namespace triangle_perimeter_l203_203037

theorem triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) 
  (h1 : area = 150)
  (h2 : leg1 = 30)
  (h3 : 0 < leg2)
  (h4 : hypotenuse = (leg1^2 + leg2^2).sqrt)
  (hArea : area = 0.5 * leg1 * leg2)
  : hypotenuse = 10 * Real.sqrt 10 ∧ leg2 = 10 ∧ (leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10) := 
by
  sorry

end triangle_perimeter_l203_203037


namespace pencil_distribution_l203_203223

theorem pencil_distribution (x : ℕ) 
  (Alice Bob Charles : ℕ)
  (h1 : Alice = 2 * Bob)
  (h2 : Charles = Bob + 3)
  (h3 : Bob = x)
  (total_pencils : 53 = Alice + Bob + Charles) : 
  Bob = 13 ∧ Alice = 26 ∧ Charles = 16 :=
by
  sorry

end pencil_distribution_l203_203223


namespace four_digit_number_count_l203_203119

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l203_203119


namespace pascal_triangle_45th_number_l203_203014

theorem pascal_triangle_45th_number : nat.choose 46 44 = 1035 := 
by sorry

end pascal_triangle_45th_number_l203_203014


namespace solve_inequality_l203_203700

theorem solve_inequality :
  {x : ℝ | (3 * x + 1) * (2 * x - 1) < 0} = {x : ℝ | -1 / 3 < x ∧ x < 1 / 2} :=
  sorry

end solve_inequality_l203_203700


namespace gcd_of_360_and_150_is_30_l203_203424

theorem gcd_of_360_and_150_is_30 : Nat.gcd 360 150 = 30 :=
by
  sorry

end gcd_of_360_and_150_is_30_l203_203424


namespace jet_flight_distance_l203_203438

-- Setting up the hypotheses and the statement
theorem jet_flight_distance (v d : ℕ) (h1 : d = 4 * (v + 50)) (h2 : d = 5 * (v - 50)) : d = 2000 :=
sorry

end jet_flight_distance_l203_203438


namespace miles_to_drive_l203_203364

def total_miles : ℕ := 1200
def miles_driven : ℕ := 768
def miles_remaining : ℕ := total_miles - miles_driven

theorem miles_to_drive : miles_remaining = 432 := by
  -- Proof goes here, omitted as per instructions
  sorry

end miles_to_drive_l203_203364


namespace trip_to_office_duration_l203_203197

noncomputable def distance (D : ℝ) : Prop :=
  let T1 := D / 58
  let T2 := D / 62
  T1 + T2 = 3

theorem trip_to_office_duration (D : ℝ) (h : distance D) : D / 58 = 1.55 :=
by sorry

end trip_to_office_duration_l203_203197


namespace speed_of_man_in_still_water_l203_203595

theorem speed_of_man_in_still_water 
  (v_m v_s : ℝ)
  (h1 : 32 = 4 * (v_m + v_s))
  (h2 : 24 = 4 * (v_m - v_s)) :
  v_m = 7 :=
by
  sorry

end speed_of_man_in_still_water_l203_203595


namespace count_four_digit_numbers_l203_203099

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l203_203099


namespace dart_board_probability_l203_203735

variable {s : ℝ} (hexagon_area : ℝ := (3 * Real.sqrt 3) / 2 * s^2) (center_hexagon_area : ℝ := (3 * Real.sqrt 3) / 8 * s^2)

theorem dart_board_probability (s : ℝ) (P : ℝ) (h : P = center_hexagon_area / hexagon_area) :
  P = 1 / 4 :=
by
  sorry

end dart_board_probability_l203_203735


namespace arccos_cos_three_l203_203614

-- Defining the problem conditions
def three_radians : ℝ := 3

-- Main statement to prove
theorem arccos_cos_three : Real.arccos (Real.cos three_radians) = three_radians := 
sorry

end arccos_cos_three_l203_203614


namespace always_possible_to_create_pairs_l203_203789

def number_of_pairs (a b : Nat) : Nat := Nat.min a b

theorem always_possible_to_create_pairs :
  ∀ (total_mittens blue_mittens green_mittens red_mittens right_mittens left_mittens : Nat),
  total_mittens = 30 →
  blue_mittens = 10 →
  green_mittens = 10 →
  red_mittens = 10 →
  right_mittens = 15 →
  left_mittens = 15 →
  (∃ (pairs : Nat), pairs >= 5).
Proof :=
by
  intros total_mittens blue_mittens green_mittens red_mittens right_mittens left_mittens
  intros h1 h2 h3 h4 h5 h6
  sorry

end always_possible_to_create_pairs_l203_203789


namespace eva_total_marks_correct_l203_203291

-- Definitions based on conditions
def math_marks_second_sem : ℕ := 80
def arts_marks_second_sem : ℕ := 90
def science_marks_second_sem : ℕ := 90

def math_marks_first_sem : ℕ := math_marks_second_sem + 10
def arts_marks_first_sem : ℕ := arts_marks_second_sem - 15
def science_marks_first_sem : ℕ := science_marks_second_sem - (science_marks_second_sem / 3)

def total_marks_first_sem : ℕ := math_marks_first_sem + arts_marks_first_sem + science_marks_first_sem
def total_marks_second_sem : ℕ := math_marks_second_sem + arts_marks_second_sem + science_marks_second_sem

def total_marks_both_sems : ℕ := total_marks_first_sem + total_marks_second_sem

-- Theorem to be proved
theorem eva_total_marks_correct : total_marks_both_sems = 485 := by
  -- Here, we state that we need to prove the total marks sum up to 485
  sorry

end eva_total_marks_correct_l203_203291


namespace largest_decimal_of_four_digit_binary_l203_203216

theorem largest_decimal_of_four_digit_binary : ∀ n : ℕ, (n < 16) → n ≤ 15 :=
by {
  -- conditions: a four-digit binary number implies \( n \) must be less than \( 2^4 = 16 \)
  sorry
}

end largest_decimal_of_four_digit_binary_l203_203216


namespace required_height_for_roller_coaster_l203_203484

-- Definitions based on conditions from the problem
def initial_height : ℕ := 48
def natural_growth_rate_per_month : ℚ := 1 / 3
def upside_down_growth_rate_per_hour : ℚ := 1 / 12
def hours_per_month_hanging_upside_down : ℕ := 2
def months_in_a_year : ℕ := 12

-- Calculations needed for the proof
def annual_natural_growth := natural_growth_rate_per_month * months_in_a_year
def annual_upside_down_growth := (upside_down_growth_rate_per_hour * hours_per_month_hanging_upside_down) * months_in_a_year
def total_annual_growth := annual_natural_growth + annual_upside_down_growth
def height_next_year := initial_height + total_annual_growth

-- Statement of the required height for the roller coaster
theorem required_height_for_roller_coaster : height_next_year = 54 :=
by
  sorry

end required_height_for_roller_coaster_l203_203484


namespace x_minus_y_eq_neg3_l203_203517

theorem x_minus_y_eq_neg3 (x y : ℝ) (i : ℂ) (h1 : x * i + 2 = y - i) (h2 : i^2 = -1) : x - y = -3 := 
  sorry

end x_minus_y_eq_neg3_l203_203517


namespace cheese_cut_indefinite_l203_203035

theorem cheese_cut_indefinite (w : ℝ) (R : ℝ) (h : ℝ) :
  R = 0.5 →
  (∀ a b c d : ℝ, a > b → b > c → c > d →
    (∃ h, h < min (a - d) (d - c) ∧
     (d + h < a ∧ d - h > c))) →
  ∃ l1 l2 : ℕ → ℝ, (∀ n, l1 (n + 1) > l2 (n) ∧ l1 n > R * l2 (n)) :=
sorry

end cheese_cut_indefinite_l203_203035


namespace cos2_alpha_plus_2sin2_alpha_l203_203516

theorem cos2_alpha_plus_2sin2_alpha {α : ℝ} (h : Real.tan α = 3 / 4) : 
    Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
by 
  sorry

end cos2_alpha_plus_2sin2_alpha_l203_203516


namespace current_age_of_son_l203_203592

variables (S F : ℕ)

-- Define the conditions
def condition1 : Prop := F = 3 * S
def condition2 : Prop := F - 8 = 4 * (S - 8)

-- The theorem statement
theorem current_age_of_son (h1 : condition1 S F) (h2 : condition2 S F) : S = 24 :=
sorry

end current_age_of_son_l203_203592


namespace even_function_value_l203_203655

theorem even_function_value (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_def : ∀ x : ℝ, 0 < x → f x = 2^x + 1) :
  f (-2) = 5 :=
  sorry

end even_function_value_l203_203655


namespace correct_total_l203_203734

-- Define the conditions in Lean
variables (y : ℕ) -- y is a natural number (non-negative integer)

-- Define the values of the different coins in cents
def value_of_quarter := 25
def value_of_dollar := 100
def value_of_nickel := 5
def value_of_dime := 10

-- Define the errors in terms of y
def error_due_to_quarters := y * (value_of_dollar - value_of_quarter) -- 75y
def error_due_to_nickels := y * (value_of_dime - value_of_nickel) -- 5y

-- Net error calculation
def net_error := error_due_to_quarters - error_due_to_nickels -- 70y

-- Math proof problem statement
theorem correct_total (h : error_due_to_quarters = 75 * y ∧ error_due_to_nickels = 5 * y) :
  net_error = 70 * y :=
by sorry

end correct_total_l203_203734


namespace robotics_club_neither_l203_203967

theorem robotics_club_neither (n c e b neither : ℕ) (h1 : n = 80) (h2 : c = 50) (h3 : e = 40) (h4 : b = 25) :
  neither = n - (c - b + e - b + b) :=
by 
  rw [h1, h2, h3, h4]
  sorry

end robotics_club_neither_l203_203967


namespace intersecting_lines_ratio_l203_203086

theorem intersecting_lines_ratio (k1 k2 a : ℝ) (h1 : k1 * a + 4 = 0) (h2 : k2 * a - 2 = 0) : k1 / k2 = -2 :=
by
    sorry

end intersecting_lines_ratio_l203_203086


namespace range_of_k_l203_203776

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 * k + 1}
def A_complement : Set ℝ := {x | 1 < x ∧ x < 3}

theorem range_of_k (k : ℝ) : ((A_complement ∩ (B k)) = ∅) ↔ (k ∈ Set.Iic 0 ∪ Set.Ici 3) := sorry

end range_of_k_l203_203776


namespace total_rooms_to_paint_l203_203876

theorem total_rooms_to_paint :
  ∀ (hours_per_room hours_remaining rooms_painted : ℕ),
    hours_per_room = 7 →
    hours_remaining = 63 →
    rooms_painted = 2 →
    rooms_painted + hours_remaining / hours_per_room = 11 :=
by
  intros
  sorry

end total_rooms_to_paint_l203_203876


namespace find_multiple_of_A_l203_203687

def shares_division_problem (A B C : ℝ) (x : ℝ) : Prop :=
  C = 160 ∧
  x * A = 5 * B ∧
  x * A = 10 * C ∧
  A + B + C = 880

theorem find_multiple_of_A (A B C x : ℝ) (h : shares_division_problem A B C x) : x = 4 :=
by sorry

end find_multiple_of_A_l203_203687


namespace evaluate_expression_l203_203754

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 2) : 4 * x^y + 5 * y^x = 76 := by
  sorry

end evaluate_expression_l203_203754


namespace number_of_children_l203_203752

theorem number_of_children (crayons_per_child total_crayons : ℕ) (h1 : crayons_per_child = 12) (h2 : total_crayons = 216) : total_crayons / crayons_per_child = 18 :=
by
  have h3 : total_crayons / crayons_per_child = 216 / 12 := by rw [h1, h2]
  norm_num at h3
  exact h3

end number_of_children_l203_203752


namespace isabel_pop_albums_l203_203713

theorem isabel_pop_albums (total_songs : ℕ) (country_albums : ℕ) (songs_per_album : ℕ) (pop_albums : ℕ)
  (h1 : total_songs = 72)
  (h2 : country_albums = 4)
  (h3 : songs_per_album = 8)
  (h4 : total_songs - country_albums * songs_per_album = pop_albums * songs_per_album) :
  pop_albums = 5 :=
by
  sorry

end isabel_pop_albums_l203_203713


namespace square_area_increase_l203_203647

variable (a : ℕ)

theorem square_area_increase (a : ℕ) :
  (a + 6) ^ 2 - a ^ 2 = 12 * a + 36 :=
by
  sorry

end square_area_increase_l203_203647


namespace charge_per_person_on_second_day_l203_203414

noncomputable def charge_second_day (k : ℕ) (x : ℝ) :=
  let total_revenue := 30 * k + 5 * k * x + 32.5 * k
  let total_visitors := 20 * k
  (total_revenue / total_visitors = 5)

theorem charge_per_person_on_second_day
  (k : ℕ) (hx : charge_second_day k 7.5) :
  7.5 = 7.5 :=
sorry

end charge_per_person_on_second_day_l203_203414


namespace shanghai_expo_visitors_l203_203198

theorem shanghai_expo_visitors :
  505000 = 5.05 * 10^5 :=
by
  sorry

end shanghai_expo_visitors_l203_203198


namespace find_n_l203_203549

def num_of_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5) + num_of_trailing_zeros (n / 5)

theorem find_n (n : ℕ) (k : ℕ) (h1 : n > 3) (h2 : k = num_of_trailing_zeros n) (h3 : 2*k + 1 = num_of_trailing_zeros (2*n)) (h4 : k > 0) : n = 6 :=
by
  sorry

end find_n_l203_203549


namespace Iain_pennies_problem_l203_203173

theorem Iain_pennies_problem :
  ∀ (P : ℝ), 200 - 30 = 170 →
             170 - (P / 100) * 170 = 136 →
             P = 20 :=
by
  intros P h1 h2
  sorry

end Iain_pennies_problem_l203_203173


namespace sweet_cookies_more_than_salty_l203_203817

-- Definitions for the given conditions
def sweet_cookies_ate : Nat := 32
def salty_cookies_ate : Nat := 23

-- The statement to prove
theorem sweet_cookies_more_than_salty :
  sweet_cookies_ate - salty_cookies_ate = 9 := by
  sorry

end sweet_cookies_more_than_salty_l203_203817


namespace count_four_digit_numbers_l203_203146

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l203_203146


namespace age_of_other_man_replaced_l203_203982

-- Define the conditions
variables (A : ℝ) (x : ℝ)
variable (average_age_women : ℝ := 50)
variable (num_men : ℕ := 10)
variable (increase_age : ℝ := 6)
variable (one_man_age : ℝ := 22)

-- State the theorem to be proved
theorem age_of_other_man_replaced :
  2 * average_age_women - (one_man_age + x) = 10 * (A + increase_age) - 10 * A →
  x = 18 :=
by
  sorry

end age_of_other_man_replaced_l203_203982


namespace sum_difference_l203_203391

def arithmetic_series_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def set_A_sum : ℕ :=
  arithmetic_series_sum 42 2 25

def set_B_sum : ℕ :=
  arithmetic_series_sum 62 2 25

theorem sum_difference :
  set_B_sum - set_A_sum = 500 :=
by
  sorry

end sum_difference_l203_203391


namespace total_blocks_l203_203882

-- Conditions
def original_blocks : ℝ := 35.0
def added_blocks : ℝ := 65.0

-- Question and proof goal
theorem total_blocks : original_blocks + added_blocks = 100.0 := 
by
  -- The proof would be provided here
  sorry

end total_blocks_l203_203882


namespace find_h_s_pairs_l203_203571

def num_regions (h s : ℕ) : ℕ :=
  1 + h * (s + 1) + s * (s + 1) / 2

theorem find_h_s_pairs (h s : ℕ) :
  h > 0 ∧ s > 0 ∧
  num_regions h s = 1992 ↔ 
  (h, s) = (995, 1) ∨ (h, s) = (176, 10) ∨ (h, s) = (80, 21) :=
by
  sorry

end find_h_s_pairs_l203_203571


namespace find_cd_l203_203270

noncomputable def period := (3 / 4) * Real.pi
noncomputable def x_value := (1 / 8) * Real.pi
noncomputable def y_value := 3
noncomputable def tangent_value := Real.tan (Real.pi / 6) -- which is 1 / sqrt(3)
noncomputable def c_value := 3 * Real.sqrt 3

theorem find_cd (c d : ℝ) 
  (h_period : d = 4 / 3) 
  (h_point : y_value = c * Real.tan (d * x_value)) :
  c * d = 4 * Real.sqrt 3 := 
sorry

end find_cd_l203_203270


namespace calculate_expression_l203_203273

theorem calculate_expression : (3.14 - Real.pi)^0 + |Real.sqrt 2 - 1| + (1/2 : ℝ)^(-1) - Real.sqrt 8 = 2 - Real.sqrt 2 :=
by
  sorry

end calculate_expression_l203_203273


namespace mo_hot_chocolate_l203_203199

noncomputable def cups_of_hot_chocolate (total_drinks: ℕ) (extra_tea: ℕ) (non_rainy_days: ℕ) (tea_per_day: ℕ) : ℕ :=
  let tea_drinks := non_rainy_days * tea_per_day 
  let chocolate_drinks := total_drinks - tea_drinks 
  (extra_tea - chocolate_drinks)

theorem mo_hot_chocolate :
  cups_of_hot_chocolate 36 14 5 5 = 11 :=
by
  sorry

end mo_hot_chocolate_l203_203199


namespace a_41_eq_6585451_l203_203725

noncomputable def a : ℕ → ℕ
| 0     => 0 /- Not used practically since n >= 1 -/
| 1     => 1
| 2     => 1
| 3     => 2
| (n+4) => a n + a (n+2) + 1

theorem a_41_eq_6585451 : a 41 = 6585451 := by
  sorry

end a_41_eq_6585451_l203_203725


namespace exists_perfect_square_of_the_form_l203_203064

theorem exists_perfect_square_of_the_form (k : ℕ) (h : k > 0) : ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * m = n * 2^k - 7 :=
by sorry

end exists_perfect_square_of_the_form_l203_203064


namespace number_of_female_democrats_l203_203849

-- Definitions and conditions
variables (F M D_F D_M D_T : ℕ)
axiom participant_total : F + M = 780
axiom female_democrats : D_F = 1 / 2 * F
axiom male_democrats : D_M = 1 / 4 * M
axiom total_democrats : D_T = 1 / 3 * (F + M)

-- Target statement to be proven
theorem number_of_female_democrats : D_T = 260 → D_F = 130 :=
by
  intro h
  sorry

end number_of_female_democrats_l203_203849


namespace perpendicular_slope_l203_203913

-- Define the given line equation
def line_eq (x y : ℝ) : Prop := 5 * x - 2 * y = 10

-- Define the slope of a line
def slope (m : ℝ) : Prop := ∀ x y b : ℝ, y = m * x + b

-- Define the condition for negative reciprocal
def perp_slope (m m_perpendicular : ℝ) : Prop := 
  m_perpendicular = - (1 / m)

-- The main statement to be proven
theorem perpendicular_slope : 
  ∃ m_perpendicular : ℝ, 
  (∃ m : ℝ, slope m ∧ (∀ x y : ℝ, line_eq x y → m = 5 / 2)) 
  → perp_slope (5 / 2) m_perpendicular ∧ m_perpendicular = - (2 / 5) := 
by
  sorry

end perpendicular_slope_l203_203913


namespace perimeter_ABCDEFG_l203_203413

variables {Point : Type}
variables {dist : Point → Point → ℝ}  -- Distance function

-- Definitions for midpoint and equilateral triangles
def is_midpoint (M A B : Point) : Prop := dist A M = dist M B ∧ dist A B = 2 * dist A M
def is_equilateral (A B C : Point) : Prop := dist A B = dist B C ∧ dist B C = dist C A

variables {A B C D E F G : Point}  -- Points in the plane
variables (h_eq_triangle_ABC : is_equilateral A B C)
variables (h_eq_triangle_ADE : is_equilateral A D E)
variables (h_eq_triangle_EFG : is_equilateral E F G)
variables (h_midpoint_D : is_midpoint D A C)
variables (h_midpoint_G : is_midpoint G A E)
variables (h_midpoint_F : is_midpoint F D E)
variables (h_AB_length : dist A B = 6)

theorem perimeter_ABCDEFG : 
  dist A B + dist B C + dist C D + dist D E + dist E F + dist F G + dist G A = 24 :=
sorry

end perimeter_ABCDEFG_l203_203413


namespace largest_two_digit_n_l203_203192

theorem largest_two_digit_n (x : ℕ) (n : ℕ) (hx : x < 10) (hx_nonzero : 0 < x)
  (hn : n = 12 * x * x) (hn_two_digit : n < 100) : n = 48 :=
by sorry

end largest_two_digit_n_l203_203192


namespace total_enemies_l203_203537

theorem total_enemies (points_per_enemy defeated_enemies undefeated_enemies total_points total_enemies : ℕ)
  (h1 : points_per_enemy = 5) 
  (h2 : undefeated_enemies = 6) 
  (h3 : total_points = 10) :
  total_enemies = 8 := by
  sorry

end total_enemies_l203_203537


namespace radius_of_circle_l203_203834

theorem radius_of_circle : 
  ∀ (r : ℝ), 3 * (2 * Real.pi * r) = 2 * Real.pi * r ^ 2 → r = 3 :=
by
  intro r
  intro h
  sorry

end radius_of_circle_l203_203834


namespace intersection_is_isosceles_right_angled_l203_203082

def is_isosceles_triangle (x : Type) : Prop := sorry -- Definition of isosceles triangle
def is_right_angled_triangle (x : Type) : Prop := sorry -- Definition of right-angled triangle

def M : Set Type := {x | is_isosceles_triangle x}
def N : Set Type := {x | is_right_angled_triangle x}

theorem intersection_is_isosceles_right_angled :
  (M ∩ N) = {x | is_isosceles_triangle x ∧ is_right_angled_triangle x} := by
  sorry

end intersection_is_isosceles_right_angled_l203_203082


namespace no_prime_quadruple_l203_203819

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_quadruple 
    (a b c d : ℕ)
    (ha_prime : is_prime a) 
    (hb_prime : is_prime b)
    (hc_prime : is_prime c)
    (hd_prime : is_prime d)
    (h_order : a < b ∧ b < c ∧ c < d) :
    (1 / a + 1 / d ≠ 1 / b + 1 / c) := 
by 
  sorry

end no_prime_quadruple_l203_203819


namespace sum_of_cubes_l203_203945

theorem sum_of_cubes (a b t : ℝ) (h : a + b = t^2) : 2 * (a^3 + b^3) = (a * t)^2 + (b * t)^2 + (a * t - b * t)^2 :=
by
  sorry

end sum_of_cubes_l203_203945


namespace prime_sol_is_7_l203_203055

theorem prime_sol_is_7 (p : ℕ) (x y : ℕ) (hp : Nat.Prime p) 
  (hx : p + 1 = 2 * x^2) (hy : p^2 + 1 = 2 * y^2) : 
  p = 7 := 
  sorry

end prime_sol_is_7_l203_203055


namespace smaller_rectangle_area_l203_203172

theorem smaller_rectangle_area (L_h S_h : ℝ) (L_v S_v : ℝ) 
  (ratio_h : L_h = (8 / 7) * S_h) 
  (ratio_v : L_v = (9 / 4) * S_v) 
  (area_large : L_h * L_v = 108) :
  S_h * S_v = 42 :=
sorry

end smaller_rectangle_area_l203_203172


namespace unique_arrangements_MOON_l203_203897

theorem unique_arrangements_MOON : 
  let M := 1
  let O := 2
  let N := 1
  let total_letters := 4
  (Nat.factorial total_letters / (Nat.factorial O)) = 12 :=
by
  sorry

end unique_arrangements_MOON_l203_203897


namespace count_logical_propositions_l203_203695

def proposition_1 : Prop := ∃ d : ℕ, d = 1
def proposition_2 : Prop := ∀ n : ℕ, n % 10 = 0 → n % 5 = 0
def proposition_3 : Prop := ∀ t : Prop, t → ¬t

theorem count_logical_propositions :
  (proposition_1 ∧ proposition_3) →
  (proposition_1 ∧ proposition_2 ∧ proposition_3) →
  (∃ (n : ℕ), n = 10 ∧ n % 5 = 0) ∧ n = 2 :=
sorry

end count_logical_propositions_l203_203695


namespace average_speed_correct_l203_203540

-- Definitions for the conditions
def distance1 : ℚ := 40
def speed1 : ℚ := 8
def time1 : ℚ := distance1 / speed1

def distance2 : ℚ := 20
def speed2 : ℚ := 40
def time2 : ℚ := distance2 / speed2

def total_distance : ℚ := distance1 + distance2
def total_time : ℚ := time1 + time2

-- Definition of average speed
def average_speed : ℚ := total_distance / total_time

-- Proof statement that needs to be proven
theorem average_speed_correct : average_speed = 120 / 11 :=
by 
  -- The details for the proof will be filled here
  sorry

end average_speed_correct_l203_203540


namespace size_relationship_l203_203921

theorem size_relationship (a b : ℝ) (h₀ : a + b > 0) :
  a / (b^2) + b / (a^2) ≥ 1 / a + 1 / b :=
by
  sorry

end size_relationship_l203_203921


namespace f_2023_pi_over_3_eq_4_l203_203836

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 2 * Real.cos x
| (n + 1), x => 4 / (2 - f n x)

theorem f_2023_pi_over_3_eq_4 : f 2023 (Real.pi / 3) = 4 := 
  sorry

end f_2023_pi_over_3_eq_4_l203_203836


namespace distinct_solutions_for_quadratic_l203_203917

theorem distinct_solutions_for_quadratic (n : ℕ) : ∃ (xs : Finset ℤ), xs.card = n ∧ ∀ x ∈ xs, ∃ y : ℤ, x^2 + 2^(n + 1) = y^2 :=
by sorry

end distinct_solutions_for_quadratic_l203_203917


namespace number_of_students_selected_from_school2_l203_203881

-- Definitions from conditions
def total_students : ℕ := 360
def students_school1 : ℕ := 123
def students_school2 : ℕ := 123
def students_school3 : ℕ := 114
def selected_students : ℕ := 60
def initial_selected_from_school1 : ℕ := 1 -- Student 002 is already selected

-- Proportion calculation
def remaining_selected_students : ℕ := selected_students - initial_selected_from_school1
def remaining_students : ℕ := total_students - initial_selected_from_school1

-- Placeholder for calculation used in the proof
def students_selected_from_school2 : ℕ := 20

-- The Lean proof statement
theorem number_of_students_selected_from_school2 :
  students_selected_from_school2 =
  Nat.ceil ((students_school2 * remaining_selected_students : ℚ) / remaining_students) :=
sorry

end number_of_students_selected_from_school2_l203_203881


namespace city_council_vote_l203_203785

theorem city_council_vote :
  ∀ (x y x' y' m : ℕ),
    x + y = 350 →
    y > x →
    y - x = m →
    x' - y' = 2 * m →
    x' + y' = 350 →
    x' = (10 * y) / 9 →
    x' - x = 66 :=
by
  intros x y x' y' m h1 h2 h3 h4 h5 h6
  -- proof goes here
  sorry

end city_council_vote_l203_203785


namespace original_price_second_store_l203_203390

-- Definitions of the conditions
def price_first_store : ℝ := 950
def discount_first_store : ℝ := 0.06
def discount_second_store : ℝ := 0.05
def price_difference : ℝ := 19

-- Define the discounted price function
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

-- State the main theorem
theorem original_price_second_store :
  ∃ P : ℝ, 
    (discounted_price price_first_store discount_first_store - discounted_price P discount_second_store = price_difference) ∧ 
    P = 960 :=
by
  sorry

end original_price_second_store_l203_203390


namespace antonov_candy_packs_l203_203744

theorem antonov_candy_packs (bought_candies : ℕ) (cando_per_pack : ℕ) (gave_to_sister : ℕ) (h_bought : bought_candies = 60) (h_pack : cando_per_pack = 20) (h_gave : gave_to_sister = 20) :
  (bought_candies - gave_to_sister) / cando_per_pack = 2 :=
by
  rw [h_bought, h_pack, h_gave]
  norm_num
  sorry

end antonov_candy_packs_l203_203744


namespace count_congruent_to_3_mod_7_lt_500_l203_203337

theorem count_congruent_to_3_mod_7_lt_500 : 
  ∃ n, n = 71 ∧ ∀ x, 0 < x ∧ x < 500 ∧ x % 7 = 3 ↔ ∃ k, 0 ≤ k ∧ k ≤ 70 ∧ x = 3 + 7 * k :=
sorry

end count_congruent_to_3_mod_7_lt_500_l203_203337


namespace range_of_a_l203_203528

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x0 : ℝ, 2^x0 - 2 ≤ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) := 
sorry

end range_of_a_l203_203528


namespace sequence_difference_l203_203261

theorem sequence_difference (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n + n) : a 2017 - a 2016 = 2016 :=
sorry

end sequence_difference_l203_203261


namespace min_value_fraction_condition_l203_203640

noncomputable def minValue (a b : ℝ) := 1 / (2 * a) + a / (b + 1)

theorem min_value_fraction_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  minValue a b = 5 / 4 :=
by
  sorry

end min_value_fraction_condition_l203_203640


namespace fish_in_third_tank_l203_203825

-- Definitions of the conditions
def first_tank_goldfish : ℕ := 7
def first_tank_beta_fish : ℕ := 8
def first_tank_fish : ℕ := first_tank_goldfish + first_tank_beta_fish

def second_tank_fish : ℕ := 2 * first_tank_fish

def third_tank_fish : ℕ := second_tank_fish / 3

-- The statement to prove
theorem fish_in_third_tank : third_tank_fish = 10 := by
  sorry

end fish_in_third_tank_l203_203825


namespace highway_length_l203_203855

theorem highway_length 
  (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) 
  (h_speed1 : speed1 = 14)
  (h_speed2 : speed2 = 16)
  (h_time : time = 1.5) : 
  speed1 * time + speed2 * time = 45 := 
sorry

end highway_length_l203_203855


namespace candles_shared_equally_l203_203268

theorem candles_shared_equally :
  ∀ (Aniyah Ambika Bree Caleb : ℕ),
  Aniyah = 6 * Ambika → Ambika = 4 → Bree = 0 → Caleb = 0 →
  (Aniyah + Ambika + Bree + Caleb) / 4 = 7 :=
by
  intros Aniyah Ambika Bree Caleb h1 h2 h3 h4
  sorry

end candles_shared_equally_l203_203268


namespace geometric_sequence_b_value_l203_203405

noncomputable def b_value (b : ℝ) : Prop :=
  ∃ s : ℝ, 180 * s = b ∧ b * s = 75 / 32 ∧ b > 0

theorem geometric_sequence_b_value (b : ℝ) : b_value b → b = 20.542 :=
by
  sorry

end geometric_sequence_b_value_l203_203405


namespace cost_of_marker_l203_203873

theorem cost_of_marker (s c m : ℕ) (h1 : s > 12) (h2 : m > 1) (h3 : c > m) (h4 : s * c * m = 924) : c = 11 :=
sorry

end cost_of_marker_l203_203873


namespace four_digit_numbers_count_l203_203111

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l203_203111


namespace coefficient_6th_term_expansion_l203_203983

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n, k => if k > n then 0 else Nat.choose n k

-- Define the coefficient of the general term of binomial expansion
def binomial_coeff (n r : ℕ) : ℤ := (-1)^r * binom n r

-- Define the theorem to show the coefficient of the 6th term in the expansion of (x-1)^10
theorem coefficient_6th_term_expansion :
  binomial_coeff 10 5 = -binom 10 5 :=
by sorry

end coefficient_6th_term_expansion_l203_203983


namespace mix_solutions_l203_203665

theorem mix_solutions {x : ℝ} (h : 0.60 * x + 0.75 * (20 - x) = 0.72 * 20) : x = 4 :=
by
-- skipping the proof with sorry
sorry

end mix_solutions_l203_203665


namespace solve_equation_1_solve_equation_2_l203_203978

theorem solve_equation_1 :
  ∀ x : ℝ, 3 * x - 5 = 6 * x - 8 → x = 1 :=
by
  intro x
  intro h
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, (x + 1) / 2 - (2 * x - 1) / 3 = 1 → x = -1 :=
by
  intro x
  intro h
  sorry

end solve_equation_1_solve_equation_2_l203_203978


namespace three_four_five_six_solution_l203_203182

-- State that the equation 3^x + 4^x = 5^x is true when x=2
axiom three_four_five_solution : 3^2 + 4^2 = 5^2

-- We need to prove the following theorem
theorem three_four_five_six_solution : 3^3 + 4^3 + 5^3 = 6^3 :=
by sorry

end three_four_five_six_solution_l203_203182


namespace max_value_log_function_l203_203511

theorem max_value_log_function (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1/2) :
  ∃ u : ℝ, (u = Real.logb (1/2) (8*x*y + 4*y^2 + 1)) ∧ (u ≤ 0) :=
sorry

end max_value_log_function_l203_203511


namespace range_of_a_product_greater_than_one_l203_203330

namespace ProofProblem

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + x^2 - a * x + 2

variables {x1 x2 a : ℝ}

-- Conditions
axiom f_has_two_distinct_zeros : f x1 a = 0 ∧ f x2 a = 0 ∧ x1 ≠ x2

-- Goal 1: Prove the range of a
theorem range_of_a : a ∈ Set.Ioi 3 := sorry  -- Formal expression for (3, +∞) in Lean

-- Goal 2: Prove x1 * x2 > 1 given that a is in the correct range
theorem product_greater_than_one (ha : a ∈ Set.Ioi 3) : x1 * x2 > 1 := sorry

end ProofProblem

end range_of_a_product_greater_than_one_l203_203330


namespace divisible_by_11_of_sum_divisible_l203_203513

open Int

theorem divisible_by_11_of_sum_divisible (a b : ℤ) (h : 11 ∣ (a^2 + b^2)) : 11 ∣ a ∧ 11 ∣ b :=
sorry

end divisible_by_11_of_sum_divisible_l203_203513


namespace variance_is_stability_measure_l203_203705

def stability_measure (yields : Fin 10 → ℝ) : Prop :=
  let mean := (yields 0 + yields 1 + yields 2 + yields 3 + yields 4 + yields 5 + yields 6 + yields 7 + yields 8 + yields 9) / 10
  let variance := 
    ((yields 0 - mean)^2 + (yields 1 - mean)^2 + (yields 2 - mean)^2 + (yields 3 - mean)^2 + 
     (yields 4 - mean)^2 + (yields 5 - mean)^2 + (yields 6 - mean)^2 + (yields 7 - mean)^2 + 
     (yields 8 - mean)^2 + (yields 9 - mean)^2) / 10
  true -- just a placeholder, would normally state that this is the appropriate measure

theorem variance_is_stability_measure (yields : Fin 10 → ℝ) : stability_measure yields :=
by 
  sorry

end variance_is_stability_measure_l203_203705


namespace min_tickets_to_ensure_match_l203_203703

theorem min_tickets_to_ensure_match : 
  ∀ (host_ticket : Fin 50 → Fin 50),
  ∃ (tickets : Fin 26 → Fin 50 → Fin 50),
  ∀ (i : Fin 26), ∃ (k : Fin 50), host_ticket k = tickets i k :=
by sorry

end min_tickets_to_ensure_match_l203_203703


namespace comparison_a_b_c_l203_203642

theorem comparison_a_b_c :
  let a := (1 / 2) ^ (1 / 3)
  let b := (1 / 3) ^ (1 / 2)
  let c := Real.log (3 / Real.pi)
  c < b ∧ b < a :=
by
  sorry

end comparison_a_b_c_l203_203642


namespace tan_alpha_sol_expr_sol_l203_203639

noncomputable def tan_half_alpha (α : ℝ) : ℝ := 2

noncomputable def tan_alpha_from_half (α : ℝ) : ℝ := 
  let tan_half := tan_half_alpha α
  2 * tan_half / (1 - tan_half * tan_half)

theorem tan_alpha_sol (α : ℝ) (h : tan_half_alpha α = 2) : tan_alpha_from_half α = -4 / 3 := by
  sorry

noncomputable def expr_eval (α : ℝ) : ℝ :=
  let tan_α := tan_alpha_from_half α
  let sin_α := tan_α / Real.sqrt (1 + tan_α * tan_α)
  let cos_α := 1 / Real.sqrt (1 + tan_α * tan_α)
  (6 * sin_α + cos_α) / (3 * sin_α - 2 * cos_α)

theorem expr_sol (α : ℝ) (h : tan_half_alpha α = 2) : expr_eval α = 7 / 6 := by
  sorry

end tan_alpha_sol_expr_sol_l203_203639


namespace calculate_expression_correct_l203_203891

theorem calculate_expression_correct :
  ( (6 + (7 / 8) - (2 + (1 / 2))) * (1 / 4) + (3 + (23 / 24) + 1 + (2 / 3)) / 4 ) / 2.5 = 1 := 
by 
  sorry

end calculate_expression_correct_l203_203891


namespace area_of_circle_l203_203620

def circle_area (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 18 * y = -45

theorem area_of_circle :
  (∃ x y : ℝ, circle_area x y) → ∃ A : ℝ, A = 52 * Real.pi :=
by
  sorry

end area_of_circle_l203_203620


namespace stamps_per_light_envelope_l203_203964

theorem stamps_per_light_envelope 
  (stamps_heavy : ℕ) (stamps_light : ℕ → ℕ) (total_light : ℕ) (total_stamps_light : ℕ)
  (total_envelopes : ℕ) :
  (∀ n, n > 5 → stamps_heavy = 5) →
  (∀ n, n <= 5 → stamps_light n = total_stamps_light / total_light) →
  total_light = 6 →
  total_stamps_light = 52 →
  total_envelopes = 14 →
  stamps_light 5 = 9 :=
by
  sorry

end stamps_per_light_envelope_l203_203964


namespace largest_of_five_consecutive_integers_l203_203312

   theorem largest_of_five_consecutive_integers (n1 n2 n3 n4 n5 : ℕ) 
     (h1: 0 < n1) (h2: n1 + 1 = n2) (h3: n2 + 1 = n3) (h4: n3 + 1 = n4)
     (h5: n4 + 1 = n5) (h6: n1 * n2 * n3 * n4 * n5 = 15120) : n5 = 10 :=
   sorry
   
end largest_of_five_consecutive_integers_l203_203312


namespace exists_invisible_square_l203_203600

def invisible (p q : ℤ) : Prop := Int.gcd p q > 1

theorem exists_invisible_square (n : ℤ) (h : 0 < n) : 
  ∃ (a b : ℤ), ∀ i j : ℤ, (0 ≤ i) ∧ (i < n) ∧ (0 ≤ j) ∧ (j < n) → invisible (a + i) (b + j) :=
by {
  sorry
}

end exists_invisible_square_l203_203600


namespace complex_series_sum_l203_203188

theorem complex_series_sum (ω : ℂ) (h₁ : ω^7 = 1) (h₂ : ω ≠ 1) :
  (ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 + ω^32 + 
   ω^34 + ω^36 + ω^38 + ω^40 + ω^42 + ω^44 + ω^46 + ω^48 + ω^50 + 
   ω^52 + ω^54) = -1 :=
by
  sorry

end complex_series_sum_l203_203188


namespace find_base_length_of_isosceles_triangle_l203_203266

noncomputable def is_isosceles_triangle_with_base_len (a b : ℝ) : Prop :=
  a = 2 ∧ ((a + a + b = 5) ∨ (a + b + b = 5))

theorem find_base_length_of_isosceles_triangle :
  ∃ (b : ℝ), is_isosceles_triangle_with_base_len 2 b ∧ (b = 1.5 ∨ b = 2) :=
by
  sorry

end find_base_length_of_isosceles_triangle_l203_203266


namespace count_four_digit_numbers_l203_203096

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l203_203096


namespace rectangle_sides_equal_perimeter_and_area_l203_203838

theorem rectangle_sides_equal_perimeter_and_area (x y : ℕ) (h : 2 * x + 2 * y = x * y) : 
    (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 4) :=
by sorry

end rectangle_sides_equal_perimeter_and_area_l203_203838


namespace angle_A_range_l203_203191

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def strictly_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x < y ∧ x ∈ I ∧ y ∈ I → f x < f y

theorem angle_A_range (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_strict_inc : strictly_increasing f {x | 0 < x})
  (h_f_half : f (1 / 2) = 0)
  (A : ℝ)
  (h_cos_A : f (Real.cos A) < 0) :
  (π / 3 < A ∧ A < π / 2) ∨ (2 * π / 3 < A ∧ A < π) :=
by
  sorry

end angle_A_range_l203_203191


namespace Mille_suckers_l203_203383

theorem Mille_suckers:
  let pretzels := 64
  let goldfish := 4 * pretzels
  let baggies := 16
  let items_per_baggie := 22
  let total_items_needed := baggies * items_per_baggie
  let total_pretzels_and_goldfish := pretzels + goldfish
  let suckers := total_items_needed - total_pretzels_and_goldfish
  suckers = 32 := 
by sorry

end Mille_suckers_l203_203383


namespace mean_of_other_two_l203_203637

theorem mean_of_other_two (a b c d e f : ℕ) (h : a = 1867 ∧ b = 1993 ∧ c = 2019 ∧ d = 2025 ∧ e = 2109 ∧ f = 2121):
  ((a + b + c + d + e + f) - (4 * 2008)) / 2 = 2051 := by
  sorry

end mean_of_other_two_l203_203637


namespace probability_of_target_destroyed_l203_203751

theorem probability_of_target_destroyed :
  let p1 := 0.9
  let p2 := 0.9
  let p3 := 0.8
  (p1 * p2 * p3) + (p1 * p2 * (1 - p3)) + (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3) = 0.954 :=
by
  let p1 := 0.9
  let p2 := 0.9
  let p3 := 0.8
  sorry

end probability_of_target_destroyed_l203_203751


namespace interest_rate_calculation_l203_203908

theorem interest_rate_calculation :
  let P := 1599.9999999999998
  let A := 1792
  let T := 2 + 2 / 5
  let I := A - P
  I / (P * T) = 0.05 :=
  sorry

end interest_rate_calculation_l203_203908


namespace picture_books_count_l203_203846

theorem picture_books_count (total_books : ℕ) (fiction_books : ℕ) (non_fiction_books : ℕ) (autobiography_books : ℕ) (picture_books : ℕ) 
  (h1 : total_books = 35)
  (h2 : fiction_books = 5)
  (h3 : non_fiction_books = fiction_books + 4)
  (h4 : autobiography_books = 2 * fiction_books)
  (h5 : picture_books = total_books - (fiction_books + non_fiction_books + autobiography_books)) :
  picture_books = 11 := 
  sorry

end picture_books_count_l203_203846


namespace find_probability_union_l203_203935

open ProbabilityTheory

-- Define events and their probabilities
variables (Ω : Type) [ProbSpace Ω]
variables (a b c d : Event Ω)

-- Initial conditions
def p_a := 2 / 5
def p_b := 2 / 5
def p_c := 1 / 5
def p_d := 1 / 3

-- Assuming independence of the events a, b, c, and d
axiom indep_events : Independent (a ∩ b) (c ∩ d)

-- Lean proof problem statement
theorem find_probability_union :
  Prob (a ∩ b ∪ c ∩ d) = 17 / 75 :=
by
  have ha : Prob a = p_a := sorry,
  have hb : Prob b = p_b := sorry,
  have hc : Prob c = p_c := sorry,
  have hd : Prob d = p_d := sorry,
  sorry

end find_probability_union_l203_203935


namespace probability_neither_prime_nor_composite_l203_203350

/-- Definition of prime number: A number is prime if it has exactly two distinct positive divisors -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of composite number: A number is composite if it has more than two positive divisors -/
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

/-- Given the number in the range 1 to 98 -/
def neither_prime_nor_composite (n : ℕ) : Prop := n = 1

/-- Probability function for uniform probability in a discrete sample space -/
def probability (event_occurrences total_possibilities : ℕ) : ℚ := event_occurrences / total_possibilities

theorem probability_neither_prime_nor_composite :
    probability 1 98 = 1 / 98 := by
  sorry

end probability_neither_prime_nor_composite_l203_203350


namespace two_wheeler_wheels_l203_203949

-- Define the total number of wheels and the number of four-wheelers
def total_wheels : Nat := 46
def num_four_wheelers : Nat := 11

-- Define the number of wheels per vehicle type
def wheels_per_four_wheeler : Nat := 4
def wheels_per_two_wheeler : Nat := 2

-- Define the number of two-wheelers
def num_two_wheelers : Nat := (total_wheels - num_four_wheelers * wheels_per_four_wheeler) / wheels_per_two_wheeler

-- Proposition stating the number of wheels of the two-wheeler
theorem two_wheeler_wheels : wheels_per_two_wheeler * num_two_wheelers = 2 := by
  sorry

end two_wheeler_wheels_l203_203949


namespace Keith_picked_zero_apples_l203_203812

variable (M J T K_A : ℕ)

theorem Keith_picked_zero_apples (hM : M = 14) (hJ : J = 41) (hT : T = 55) (hTotalOranges : M + J = T) : K_A = 0 :=
by
  sorry

end Keith_picked_zero_apples_l203_203812


namespace bin_sum_sub_eq_l203_203497

-- Define binary numbers
def b1 := 0b101110  -- binary 101110_2
def b2 := 0b10101   -- binary 10101_2
def b3 := 0b111000  -- binary 111000_2
def b4 := 0b110101  -- binary 110101_2
def b5 := 0b11101   -- binary 11101_2

-- Define the theorem
theorem bin_sum_sub_eq : ((b1 + b2) - (b3 - b4) + b5) = 0b1011101 := by
  sorry

end bin_sum_sub_eq_l203_203497


namespace sum_of_different_roots_eq_six_l203_203721

theorem sum_of_different_roots_eq_six (a b : ℝ) (h1 : a * (a - 6) = 7) (h2 : b * (b - 6) = 7) (h3 : a ≠ b) : a + b = 6 :=
sorry

end sum_of_different_roots_eq_six_l203_203721


namespace comparison_abc_l203_203370

noncomputable def a : ℝ := (Real.exp 1 + 2) / Real.log (Real.exp 1 + 2)
noncomputable def b : ℝ := 2 / Real.log 2
noncomputable def c : ℝ := (Real.exp 1)^2 / (4 - Real.log 4)

theorem comparison_abc : c < b ∧ b < a :=
by {
  sorry
}

end comparison_abc_l203_203370


namespace find_a_plus_b_l203_203961

def satisfies_conditions (a b : ℝ) :=
  ∀ x : ℝ, 3 * (a * x + b) - 8 = 4 * x + 7

theorem find_a_plus_b (a b : ℝ) (h : satisfies_conditions a b) : a + b = 19 / 3 :=
  sorry

end find_a_plus_b_l203_203961


namespace sequence_a_n_eq_5050_l203_203351

theorem sequence_a_n_eq_5050 (a : ℕ → ℕ) (h1 : ∀ n > 1, (n - 1) * a n = (n + 1) * a (n - 1)) (h2 : a 1 = 1) : 
  a 100 = 5050 := 
by
  sorry

end sequence_a_n_eq_5050_l203_203351


namespace maximum_sum_S6_l203_203360

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + (n - 1) * d

def sum_arithmetic_sequence (a d : α) (n : ℕ) : α :=
  (n : α) / 2 * (2 * a + (n - 1) * d)

theorem maximum_sum_S6 (a d : α)
  (h1 : arithmetic_sequence a d 3 + arithmetic_sequence a d 10 < 0)
  (h2 : sum_arithmetic_sequence a d 11 > 0) :
  ∀ n : ℕ, sum_arithmetic_sequence a d n ≤ sum_arithmetic_sequence a d 6 :=
by sorry

end maximum_sum_S6_l203_203360


namespace number_of_vegetarians_l203_203948

-- Define the conditions
def only_veg : ℕ := 11
def only_nonveg : ℕ := 6
def both_veg_and_nonveg : ℕ := 9

-- Define the total number of vegetarians
def total_veg : ℕ := only_veg + both_veg_and_nonveg

-- The statement to be proved
theorem number_of_vegetarians : total_veg = 20 := 
by
  sorry

end number_of_vegetarians_l203_203948


namespace pascal_triangle_45th_number_l203_203010

theorem pascal_triangle_45th_number (n k : ℕ) (h1 : n = 47) (h2 : k = 44) : 
  Nat.choose (n - 1) k = 1035 :=
by
  sorry

end pascal_triangle_45th_number_l203_203010


namespace functional_eq_log_l203_203764

theorem functional_eq_log {f : ℝ → ℝ} (h₁ : f 4 = 2) 
                           (h₂ : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → f (x1 * x2) = f x1 + f x2) : 
                           (∀ x : ℝ, 0 < x → f x = Real.log x / Real.log 2) := 
by
  sorry

end functional_eq_log_l203_203764


namespace four_digit_number_count_l203_203116

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l203_203116


namespace expenses_neg_five_given_income_five_l203_203993

theorem expenses_neg_five_given_income_five 
  (income_5 : ℤ)
  (income_5_pos : income_5 = 5)
  (opposite : ∀ (x : ℤ), -x = -1 * x) :
  (expenses_5 : ℤ) (expenses_5 = -5) :=
by {
  sorry
}

end expenses_neg_five_given_income_five_l203_203993


namespace ratio_is_one_half_l203_203848

namespace CupRice

-- Define the grains of rice in one cup
def grains_in_one_cup : ℕ := 480

-- Define the grains of rice in the portion of the cup
def grains_in_portion : ℕ := 8 * 3 * 10

-- Define the ratio of the portion of the cup to the whole cup
def portion_to_cup_ratio := grains_in_portion / grains_in_one_cup

-- Prove that the ratio of the portion of the cup to the whole cup is 1:2
theorem ratio_is_one_half : portion_to_cup_ratio = 1 / 2 := by
  -- Proof goes here, but we skip it as required
  sorry
end CupRice

end ratio_is_one_half_l203_203848


namespace parameter_for_three_distinct_solutions_l203_203315

open Polynomial

theorem parameter_for_three_distinct_solutions (a : ℝ) :
  (∀ x : ℝ, x^4 - 40 * x^2 + 144 = a * (x^2 + 4 * x - 12)) →
  (∀ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 → 
  (x1^4 - 40 * x1^2 + 144 = a * (x1^2 + 4 * x1 - 12) ∧ 
   x2^4 - 40 * x2^2 + 144 = a * (x2^2 + 4 * x2 - 12) ∧ 
   x3^4 - 40 * x3^2 + 144 = a * (x3^2 + 4 * x3 - 12) ∧
   x4^4 - 40 * x4^2 + 144 = a * (x4^2 + 4 * x4 - 12))) → a = 48 :=
by
  sorry

end parameter_for_three_distinct_solutions_l203_203315


namespace a_eq_b_if_conditions_l203_203209

theorem a_eq_b_if_conditions (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := 
sorry

end a_eq_b_if_conditions_l203_203209


namespace columbus_discovered_america_in_1492_l203_203556

theorem columbus_discovered_america_in_1492 :
  ∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧
  1 + x + y + z = 16 ∧ y + 1 = 5 * z ∧
  1000 + 100 * x + 10 * y + z = 1492 :=
by
  sorry

end columbus_discovered_america_in_1492_l203_203556


namespace ζ_sum_8_l203_203376

open Complex

def ζ1 : ℂ := sorry
def ζ2 : ℂ := sorry
def ζ3 : ℂ := sorry

def e1 := ζ1 + ζ2 + ζ3
def e2 := ζ1 * ζ2 + ζ2 * ζ3 + ζ3 * ζ1
def e3 := ζ1 * ζ2 * ζ3

axiom h1 : e1 = 2
axiom h2 : e1^2 - 2 * e2 = 8
axiom h3 : (e1^2 - 2 * e2)^2 - 2 * (e2^2 - 2 * e1 * e3) = 26

theorem ζ_sum_8 : ζ1^8 + ζ2^8 + ζ3^8 = 219 :=
by {
  -- The proof goes here, omitting solution steps as instructed.
  sorry
}

end ζ_sum_8_l203_203376


namespace slope_of_perpendicular_line_l203_203915

theorem slope_of_perpendicular_line (m1 m2 : ℝ) : 
  (5*x - 2*y = 10) →  ∃ m2, m2 = (-2/5) :=
by sorry

end slope_of_perpendicular_line_l203_203915


namespace gears_can_rotate_l203_203535

theorem gears_can_rotate (n : ℕ) : (∃ f : ℕ → Prop, f 0 ∧ (∀ k, f (k+1) ↔ ¬f k) ∧ f n = f 0) ↔ (n % 2 = 0) :=
by
  sorry

end gears_can_rotate_l203_203535


namespace fraction_unclaimed_l203_203299

def exists_fraction_unclaimed (x : ℕ) : Prop :=
  let claimed_by_Eva := (1 / 2 : ℚ) * x
  let remaining_after_Eva := x - claimed_by_Eva
  let claimed_by_Liam := (3 / 8 : ℚ) * x
  let remaining_after_Liam := remaining_after_Eva - claimed_by_Liam
  let claimed_by_Noah := (1 / 8 : ℚ) * remaining_after_Eva
  let remaining_after_Noah := remaining_after_Liam - claimed_by_Noah
  remaining_after_Noah / x = (75 / 128 : ℚ)

theorem fraction_unclaimed {x : ℕ} : exists_fraction_unclaimed x :=
by
  sorry

end fraction_unclaimed_l203_203299


namespace scientific_notation_4040000_l203_203905

theorem scientific_notation_4040000 :
  (4040000 : ℝ) = 4.04 * (10 : ℝ)^6 :=
by
  sorry

end scientific_notation_4040000_l203_203905


namespace shoveling_driveways_l203_203184

-- Definitions of the conditions
def cost_of_candy_bars := 2 * 0.75
def cost_of_lollipops := 4 * 0.25
def total_cost := cost_of_candy_bars + cost_of_lollipops
def portion_of_earnings := total_cost * 6
def charge_per_driveway := 1.50
def number_of_driveways := portion_of_earnings / charge_per_driveway

-- The theorem to prove Jimmy shoveled 10 driveways
theorem shoveling_driveways :
  number_of_driveways = 10 := 
by
  sorry

end shoveling_driveways_l203_203184


namespace every_real_has_cube_root_l203_203435

theorem every_real_has_cube_root : ∀ y : ℝ, ∃ x : ℝ, x^3 = y := 
by
  sorry

end every_real_has_cube_root_l203_203435


namespace number_of_clothes_hangers_l203_203464

noncomputable def total_money : ℝ := 60
noncomputable def spent_on_tissues : ℝ := 34.8
noncomputable def price_per_hanger : ℝ := 1.6

theorem number_of_clothes_hangers : 
  let remaining_money := total_money - spent_on_tissues
  let hangers := remaining_money / price_per_hanger
  Int.floor hangers = 15 := 
by
  sorry

end number_of_clothes_hangers_l203_203464


namespace four_digit_numbers_count_eq_l203_203094

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l203_203094


namespace four_digit_numbers_count_eq_l203_203091

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l203_203091


namespace symmetry_axis_one_of_cos_2x_minus_sin_2x_l203_203288

noncomputable def symmetry_axis (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 2) - Real.pi / 8

theorem symmetry_axis_one_of_cos_2x_minus_sin_2x :
  symmetry_axis (-Real.pi / 8) :=
by
  use 0
  simp
  sorry

end symmetry_axis_one_of_cos_2x_minus_sin_2x_l203_203288


namespace selection_problem_l203_203065

def group_size : ℕ := 10
def selected_group_size : ℕ := 3
def total_ways_without_C := Nat.choose 9 3
def ways_without_A_B_C := Nat.choose 7 3
def correct_answer := total_ways_without_C - ways_without_A_B_C

theorem selection_problem:
  (∃ (A B C : ℕ), total_ways_without_C - ways_without_A_B_C = 49) :=
by
  sorry

end selection_problem_l203_203065


namespace no_solution_natural_p_q_r_l203_203491

theorem no_solution_natural_p_q_r :
  ¬ ∃ (p q r : ℕ), 2^p + 5^q = 19^r := sorry

end no_solution_natural_p_q_r_l203_203491


namespace batsman_average_after_17_l203_203731

variable (x : ℝ)
variable (total_runs_16 : ℝ := 16 * x)
variable (runs_17 : ℝ := 90)
variable (new_total_runs : ℝ := total_runs_16 + runs_17)
variable (new_average : ℝ := new_total_runs / 17)

theorem batsman_average_after_17 :
  (total_runs_16 + runs_17 = 17 * (x + 3)) → new_average = x + 3 → new_average = 42 :=
by
  intros h1 h2
  sorry

end batsman_average_after_17_l203_203731


namespace matrix_cubed_l203_203616

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, -2], ![2, -1]]

theorem matrix_cubed :
  (A * A * A) = ![![ -4, 2], ![-2, 1]] :=
by
  sorry

end matrix_cubed_l203_203616


namespace gcd_of_360_and_150_l203_203429

theorem gcd_of_360_and_150 : Nat.gcd 360 150 = 30 := 
by
  sorry

end gcd_of_360_and_150_l203_203429


namespace sequence_match_l203_203645

-- Define the sequence sum S_n
def S_n (n : ℕ) : ℕ := 2^(n + 1) - 1

-- Define the sequence a_n based on the problem statement
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 2^n

-- The theorem stating that sequence a_n satisfies the given sum condition S_n
theorem sequence_match (n : ℕ) : a_n n = if n = 1 then 3 else 2^n :=
  sorry

end sequence_match_l203_203645


namespace candidates_appeared_equal_l203_203534

theorem candidates_appeared_equal 
  (A_candidates B_candidates : ℕ)
  (A_selected B_selected : ℕ)
  (h1 : 6 * A_candidates = A_selected * 100)
  (h2 : 7 * B_candidates = B_selected * 100)
  (h3 : B_selected = A_selected + 83)
  (h4 : A_candidates = B_candidates):
  A_candidates = 8300 :=
by
  sorry

end candidates_appeared_equal_l203_203534


namespace find_a_l203_203210

noncomputable def f : ℝ+ → ℝ := sorry

theorem find_a (f : ℝ+ → ℝ) (h1 : ∀ (x y : ℝ+), f (x * y) = f x + f y)
  (h2 : f 8 = -3) :
  ∃ a : ℝ+, f a = 1 / 2 ∧ a = ⟨√2 / 2, sorry⟩ :=
sorry

end find_a_l203_203210


namespace arithmetic_sequence_sum_l203_203950

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ), (∀ n : ℕ, a (n+1) - a n = 2) → a 2 = 5 → (a 0 + a 1 + a 2 + a 3) = 24 :=
by
  sorry

end arithmetic_sequence_sum_l203_203950


namespace gcd_of_360_and_150_is_30_l203_203427

theorem gcd_of_360_and_150_is_30 : Nat.gcd 360 150 = 30 :=
by
  sorry

end gcd_of_360_and_150_is_30_l203_203427


namespace hexagon_perimeter_l203_203084

-- Defining the side lengths of the hexagon
def side_lengths : List ℕ := [7, 10, 8, 13, 11, 9]

-- Defining the perimeter calculation
def perimeter (sides : List ℕ) : ℕ := sides.sum

-- The main theorem stating the perimeter of the given hexagon
theorem hexagon_perimeter :
  perimeter side_lengths = 58 := by
  -- Skipping proof here
  sorry

end hexagon_perimeter_l203_203084


namespace grandpa_max_movies_l203_203664

-- Definition of the conditions
def movie_duration : ℕ := 90

def tuesday_total_minutes : ℕ := 4 * 60 + 30

def tuesday_movies_watched : ℕ := tuesday_total_minutes / movie_duration

def wednesday_movies_watched : ℕ := 2 * tuesday_movies_watched

def total_movies_watched : ℕ := tuesday_movies_watched + wednesday_movies_watched

theorem grandpa_max_movies : total_movies_watched = 9 := by
  sorry

end grandpa_max_movies_l203_203664


namespace four_digit_numbers_count_eq_l203_203088

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l203_203088


namespace prove_M_l203_203318

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}
def M : Set ℕ := {x | x ∈ P ∧ x ∉ Q}

theorem prove_M :
  M = {1} :=
by
  sorry

end prove_M_l203_203318


namespace cubic_sum_identity_l203_203780

theorem cubic_sum_identity (x y z : ℝ) (h1 : x + y + z = 15) (h2 : xy + yz + zx = 34) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 1845 :=
by
  sorry

end cubic_sum_identity_l203_203780


namespace arithmetic_sequence_term_l203_203894

theorem arithmetic_sequence_term :
  (∀ (a_n : ℕ → ℚ) (S : ℕ → ℚ),
    (∀ n, a_n n = a_n 1 + (n - 1) * 1) → -- Arithmetic sequence with common difference of 1
    (∀ n, S n = n * a_n 1 + (n * (n - 1)) / 2) →  -- Sum of first n terms of sequence
    S 8 = 4 * S 4 →
    a_n 10 = 19 / 2) :=
by
  intros a_n S ha_n hSn hS8_eq
  sorry

end arithmetic_sequence_term_l203_203894


namespace factorization_of_2210_l203_203171

theorem factorization_of_2210 : 
  ∃! (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 2210) :=
sorry

end factorization_of_2210_l203_203171


namespace tank_volume_ratio_l203_203186

variable {V1 V2 : ℝ}

theorem tank_volume_ratio
  (h1 : 3 / 4 * V1 = 5 / 8 * V2) :
  V1 / V2 = 5 / 6 :=
sorry

end tank_volume_ratio_l203_203186


namespace four_digit_numbers_count_l203_203106

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l203_203106


namespace closest_multiple_of_21_to_2023_l203_203710

theorem closest_multiple_of_21_to_2023 : ∃ k : ℤ, k * 21 = 2022 ∧ ∀ m : ℤ, m * 21 = 2023 → (abs (m - 2023)) > (abs (2022 - 2023)) :=
by
  sorry

end closest_multiple_of_21_to_2023_l203_203710


namespace number_of_four_digit_numbers_l203_203157

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l203_203157


namespace disjunction_of_false_is_false_l203_203938

-- Given conditions
variables (p q : Prop)

-- We are given the assumption that both p and q are false propositions
axiom h1 : ¬ p
axiom h2 : ¬ q

-- We want to prove that the disjunction p ∨ q is false
theorem disjunction_of_false_is_false (p q : Prop) (h1 : ¬ p) (h2 : ¬ q) : ¬ (p ∨ q) := 
by
  sorry

end disjunction_of_false_is_false_l203_203938


namespace ratio_of_sums_l203_203680

theorem ratio_of_sums (p q r u v w : ℝ) 
  (h1 : p > 0) (h2 : q > 0) (h3 : r > 0) (h4 : u > 0) (h5 : v > 0) (h6 : w > 0)
  (h7 : p^2 + q^2 + r^2 = 49) (h8 : u^2 + v^2 + w^2 = 64)
  (h9 : p * u + q * v + r * w = 56) : 
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end ratio_of_sums_l203_203680


namespace compute_difference_of_squares_l203_203749

theorem compute_difference_of_squares :
  262^2 - 258^2 = 2080 := 
by
  sorry

end compute_difference_of_squares_l203_203749


namespace quadratic_has_sum_r_s_l203_203205

/-
  Define the quadratic equation 6x^2 - 24x - 54 = 0
-/
def quadratic_eq (x : ℝ) : Prop :=
  6 * x^2 - 24 * x - 54 = 0

/-
  Define the value 11 which is the sum r + s when completing the square
  for the above quadratic equation  
-/
def result_value := -2 + 13

/-
  State the proof that r + s = 11 given the quadratic equation.
-/
theorem quadratic_has_sum_r_s : ∀ x : ℝ, quadratic_eq x → -2 + 13 = 11 :=
by
  intros
  exact rfl

end quadratic_has_sum_r_s_l203_203205


namespace value_of_coins_l203_203379

theorem value_of_coins (n d : ℕ) (hn : n + d = 30)
    (hv : 10 * n + 5 * d = 5 * n + 10 * d + 90) :
    300 - 5 * n = 180 := by
  sorry

end value_of_coins_l203_203379


namespace difference_of_squares_example_l203_203746

theorem difference_of_squares_example :
  (262^2 - 258^2 = 2080) :=
by {
  sorry -- placeholder for the actual proof
}

end difference_of_squares_example_l203_203746


namespace correctness_of_statements_l203_203244

theorem correctness_of_statements (p q : Prop) (x y : ℝ) : 
  (¬ (p ∧ q) → (p ∨ q)) ∧
  ((xy = 0) → ¬(x^2 + y^2 = 0)) ∧
  ¬(∀ (L P : ℝ → ℝ), (∃ x, L x = P x) ↔ (∃ x, L x = P x ∧ ∀ x₁ x₂, x₁ ≠ x₂ → L x₁ ≠ P x₂)) →
  (0 + 1 + 0 = 1) :=
by
  sorry

end correctness_of_statements_l203_203244


namespace four_digit_number_count_l203_203113

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l203_203113


namespace find_value_l203_203944

theorem find_value (x : ℝ) (h : x^2 - x - 1 = 0) : 2 * x^2 - 2 * x + 2021 = 2023 := 
by 
  sorry -- Proof needs to be provided

end find_value_l203_203944


namespace stamp_collection_l203_203228

theorem stamp_collection (x : ℕ) :
  (5 * x + 3 * (x + 20) = 300) → (x = 30) ∧ (x + 20 = 50) :=
by
  sorry

end stamp_collection_l203_203228


namespace batsman_average_after_17th_l203_203588

theorem batsman_average_after_17th (A : ℤ) (h1 : 86 + 16 * A = 17 * (A + 3)) : A + 3 = 38 :=
by
  sorry

end batsman_average_after_17th_l203_203588


namespace inequality_three_var_l203_203763

theorem inequality_three_var
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c) :
  2 * (a^3 + b^3 + c^3) ≥ a^2 * b + a * b^2 + a^2 * c + a * c^2 + b^2 * c + b * c^2 :=
by sorry

end inequality_three_var_l203_203763


namespace find_y_l203_203531

theorem find_y (x y z : ℤ) (h₁ : x + y + z = 355) (h₂ : x - y = 200) (h₃ : x + z = 500) : y = -145 :=
by
  sorry

end find_y_l203_203531


namespace favorite_numbers_parity_l203_203792

variables (D J A H : ℤ)

def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem favorite_numbers_parity
  (h1 : odd (D + 3 * J))
  (h2 : odd ((A - H) * 5))
  (h3 : even (D * H + 17)) :
  odd D ∧ even J ∧ even A ∧ odd H := 
sorry

end favorite_numbers_parity_l203_203792


namespace part1_l203_203866

theorem part1 (a n : ℕ) (hne : a % 2 = 1) : (4 ∣ a^n - 1) → (n % 2 = 0) :=
by
  sorry

end part1_l203_203866


namespace compute_difference_of_squares_l203_203748

theorem compute_difference_of_squares :
  262^2 - 258^2 = 2080 := 
by
  sorry

end compute_difference_of_squares_l203_203748


namespace option_A_equal_l203_203884

theorem option_A_equal : (-2: ℤ)^(3: ℕ) = ((-2: ℤ)^(3: ℕ)) :=
by
  sorry

end option_A_equal_l203_203884


namespace circles_touch_each_other_l203_203213

-- Define the radii of the two circles and the distance between their centers.
variables (R r d : ℝ)

-- Hypotheses: the condition and the relationships derived from the solution.
variables (x y t : ℝ)

-- The core relationships as conditions based on the problem and the solution.
axiom h1 : x + y = t
axiom h2 : x / y = R / r
axiom h3 : t / d = x / R

-- The proof statement
theorem circles_touch_each_other 
  (h1 : x + y = t) 
  (h2 : x / y = R / r) 
  (h3 : t / d = x / R) : 
  d = R + r := 
by 
  sorry

end circles_touch_each_other_l203_203213


namespace quadratic_rewrite_ab_value_l203_203380

theorem quadratic_rewrite_ab_value:
  ∃ a b c : ℤ, (∀ x: ℝ, 16*x^2 + 40*x + 18 = (a*x + b)^2 + c) ∧ a * b = 20 :=
by
  -- We'll add the definitions derived from conditions here
  sorry

end quadratic_rewrite_ab_value_l203_203380


namespace moses_more_than_esther_l203_203853

theorem moses_more_than_esther (total_amount: ℝ) (moses_share: ℝ) (tony_esther_share: ℝ) :
  total_amount = 50 → moses_share = 0.40 * total_amount → 
  tony_esther_share = (total_amount - moses_share) / 2 → 
  moses_share - tony_esther_share = 5 :=
by
  intros h1 h2 h3
  sorry

end moses_more_than_esther_l203_203853


namespace find_multiple_l203_203678

-- Defining the conditions
def first_lock_time := 5
def second_lock_time (x : ℕ) := 5 * x - 3

-- Proving the multiple
theorem find_multiple : 
  ∃ x : ℕ, (5 * first_lock_time * x - 3) * 5 = 60 ∧ (x = 3) :=
by
  sorry

end find_multiple_l203_203678


namespace sum_of_possible_values_of_x_l203_203501

noncomputable def mean (a b c d e f g : ℝ) : ℝ := (a + b + c + d + e + f + g) / 7
noncomputable def median (a b c d e f x : ℝ) : ℝ :=
  if x ≤ 3 then 3 else if x < 5 then x else 5
noncomputable def mode (a b c d e f : ℝ) : ℝ := 3

theorem sum_of_possible_values_of_x :
  let x1 := 17
  let x2 := 53 / 13
  (x1 + x2) = 17 + 53 / 13 := by
  sorry

end sum_of_possible_values_of_x_l203_203501


namespace unique_solution_l203_203627

-- Definitions of the problem
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime (4 * q - 1) ∧ (p + q) * (r - p) = p + r

theorem unique_solution (p q r : ℕ) (h : satisfies_conditions p q r) : (p, q, r) = (2, 3, 3) :=
  sorry

end unique_solution_l203_203627


namespace smallest_three_digit_pqr_l203_203859

theorem smallest_three_digit_pqr (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  100 ≤ p * q^2 * r ∧ p * q^2 * r < 1000 → p * q^2 * r = 126 := 
sorry

end smallest_three_digit_pqr_l203_203859


namespace find_toonies_l203_203487

-- Define the number of coins and their values
variables (L T : ℕ) -- L represents the number of loonies, T represents the number of toonies

-- Define the conditions
def total_coins := L + T = 10
def total_value := 1 * L + 2 * T = 14

-- Define the theorem to be proven
theorem find_toonies (L T : ℕ) (h1 : total_coins L T) (h2 : total_value L T) : T = 4 :=
by
  sorry

end find_toonies_l203_203487


namespace find_speed_second_train_l203_203482

noncomputable def speed_second_train (length_train1 length_train2 : ℝ) (speed_train1_kmph : ℝ) (time_to_cross : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let total_distance := length_train1 + length_train2
  let relative_speed_mps := total_distance / time_to_cross
  let speed_train2_mps := speed_train1_mps - relative_speed_mps
  speed_train2_mps * 3600 / 1000

theorem find_speed_second_train :
  speed_second_train 380 540 72 91.9926405887529 = 36 := by
  sorry

end find_speed_second_train_l203_203482


namespace unique_4_digit_number_l203_203024

theorem unique_4_digit_number (P E R U : ℕ) 
  (hP : 0 ≤ P ∧ P < 10)
  (hE : 0 ≤ E ∧ E < 10)
  (hR : 0 ≤ R ∧ R < 10)
  (hU : 0 ≤ U ∧ U < 10)
  (hPERU : 1000 ≤ (P * 1000 + E * 100 + R * 10 + U) ∧ (P * 1000 + E * 100 + R * 10 + U) < 10000) 
  (h_eq : (P * 1000 + E * 100 + R * 10 + U) = (P + E + R + U) ^ U) : 
  (P = 4) ∧ (E = 9) ∧ (R = 1) ∧ (U = 3) ∧ (P * 1000 + E * 100 + R * 10 + U = 4913) :=
sorry

end unique_4_digit_number_l203_203024


namespace johns_age_l203_203063

theorem johns_age :
  ∃ x : ℕ, (∃ n : ℕ, x - 5 = n^2) ∧ (∃ m : ℕ, x + 3 = m^3) ∧ x = 69 :=
by
  sorry

end johns_age_l203_203063


namespace intersection_setA_setB_l203_203658

noncomputable def setA : Set ℝ := { x : ℝ | abs (x - 1) < 2 }
noncomputable def setB : Set ℝ := { x : ℝ | (x - 2) / (x + 4) < 0 }

theorem intersection_setA_setB : 
  (setA ∩ setB) = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end intersection_setA_setB_l203_203658


namespace arithmetic_geometric_sequence_formula_l203_203840

theorem arithmetic_geometric_sequence_formula :
  ∃ (a d : ℝ), (3 * a = 6) ∧
  ((5 - d) * (15 + d) = 64) ∧
  (∀ (n : ℕ), n ≥ 3 → (∃ (b_n : ℝ), b_n = 2 ^ (n - 1))) :=
by
  sorry

end arithmetic_geometric_sequence_formula_l203_203840


namespace count_four_digit_numbers_l203_203097

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l203_203097


namespace probability_of_x_plus_y_less_than_4_l203_203474

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l203_203474


namespace remaining_surface_area_unchanged_l203_203289

noncomputable def original_cube_surface_area : Nat := 6 * 4 * 4

def corner_cube_surface_area : Nat := 3 * 2 * 2

def remaining_surface_area (original_cube_surface_area : Nat) (corner_cube_surface_area : Nat) : Nat :=
  original_cube_surface_area

theorem remaining_surface_area_unchanged :
  remaining_surface_area original_cube_surface_area corner_cube_surface_area = 96 := 
by
  sorry

end remaining_surface_area_unchanged_l203_203289


namespace surface_area_of_prism_l203_203025

theorem surface_area_of_prism (l w h : ℕ)
  (h_internal_volume : l * w * h = 24)
  (h_external_volume : (l + 2) * (w + 2) * (h + 2) = 120) :
  2 * ((l + 2) * (w + 2) + (w + 2) * (h + 2) + (h + 2) * (l + 2)) = 148 :=
by
  sorry

end surface_area_of_prism_l203_203025


namespace prob_less_than_9_is_correct_l203_203879

-- Define the probabilities
def prob_ring_10 := 0.24
def prob_ring_9 := 0.28
def prob_ring_8 := 0.19

-- Define the condition for scoring less than 9, which does not include hitting the 10 or 9 ring.
def prob_less_than_9 := 1 - prob_ring_10 - prob_ring_9

-- Now we state the theorem we want to prove.
theorem prob_less_than_9_is_correct : prob_less_than_9 = 0.48 :=
by {
  -- Proof would go here
  sorry
}

end prob_less_than_9_is_correct_l203_203879


namespace chimney_base_radius_l203_203877

-- Given conditions
def tinplate_length := 219.8
def tinplate_width := 125.6
def pi_approx := 3.14

def radius_length (circumference : Float) : Float :=
  circumference / (2 * pi_approx)

def radius_width (circumference : Float) : Float :=
  circumference / (2 * pi_approx)

theorem chimney_base_radius :
  radius_length tinplate_length = 35 ∧ radius_width tinplate_width = 20 :=
by 
  sorry

end chimney_base_radius_l203_203877


namespace abs_sum_values_l203_203348

theorem abs_sum_values (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : 
  x + y = 7 ∨ x + y = 3 := 
by
  sorry

end abs_sum_values_l203_203348


namespace perpendicular_slope_l203_203911

theorem perpendicular_slope :
  ∀ (x y : ℝ), 5 * x - 2 * y = 10 → y = ((5 : ℝ) / 2) * x - 5 → ∃ (m : ℝ), m = - (2 / 5) := by
  sorry

end perpendicular_slope_l203_203911


namespace seats_usually_taken_l203_203843

theorem seats_usually_taken:
  let tables := 15 in
  let seats_per_table := 10 in
  let total_seats := tables * seats_per_table in
  let unseated_fraction := 1 / 10 in
  let unseated_seats := total_seats * unseated_fraction in
  let seats_taken := total_seats - unseated_seats in
  seats_taken = 135 :=
by
  sorry

end seats_usually_taken_l203_203843


namespace theo_possible_codes_l203_203841

theorem theo_possible_codes : 
  let odd_numbers := {n | 1 ≤ n ∧ n ≤ 30 ∧ n % 2 = 1},
      even_numbers := {n | 1 ≤ n ∧ n ≤ 30 ∧ n % 2 = 0},
      multiples_of_5 := {n | 1 ≤ n ∧ n ≤ 30 ∧ n % 5 = 0} in
  (odd_numbers.card = 15) ∧ (even_numbers.card = 15) ∧ (multiples_of_5.card = 6) ∧ 
  (odd_numbers.card * even_numbers.card * multiples_of_5.card = 1350) :=
by
  sorry

end theo_possible_codes_l203_203841


namespace laura_three_blue_pens_l203_203801

open Classical
open Probability

-- Definitions and conditions based on the problem
def num_blue : ℕ := 8
def num_red : ℕ := 7
def total_pens : ℕ := num_blue + num_red
def trials : ℕ := 7
def pick_blue_prob : ℝ := num_blue / total_pens
def pick_red_prob : ℝ := num_red / total_pens

-- The function to compute the binomial coefficient
noncomputable def binom_coeff : ℕ := Nat.choose trials 3

-- The probability calculation for picking exactly three blue pens
noncomputable def specific_arrangement_prob : ℝ :=
  (pick_blue_prob ^ 3) * (pick_red_prob ^ 4)

noncomputable def total_probability : ℝ :=
  binom_coeff * specific_arrangement_prob

-- Proof problem statement
theorem laura_three_blue_pens :
  total_probability = 43025920 / 170859375 := by
  sorry -- Proof goes here

end laura_three_blue_pens_l203_203801


namespace count_valid_pairs_l203_203326

open Finset

def A : Finset ℕ := {0, 1, 2, 3, 4, 5, 9}

theorem count_valid_pairs : (∑ a in A, ∑ b in A, if a ≠ b ∧ a > 0 ∧ b < 4 then 1 else 0) = 21 := by
  sorry

end count_valid_pairs_l203_203326


namespace area_and_perimeter_l203_203799

-- Given a rectangle R with length l and width w
variables (l w : ℝ)
-- Define the area of R
def area_R : ℝ := l * w

-- Define a smaller rectangle that is cut out, with an area A_cut
variables (A_cut : ℝ)
-- Define the area of the resulting figure S
def area_S : ℝ := area_R l w - A_cut

-- Define the perimeter of R
def perimeter_R : ℝ := 2 * l + 2 * w

-- perimeter_R remains the same after cutting out the smaller rectangle
theorem area_and_perimeter (h_cut : 0 < A_cut) (h_cut_le : A_cut ≤ area_R l w) : 
  (area_S l w A_cut < area_R l w) ∧ (perimeter_R l w = perimeter_R l w) :=
by
  sorry

end area_and_perimeter_l203_203799


namespace base_5_minus_base_8_in_base_10_l203_203494

def base_5 := 52143
def base_8 := 4310

theorem base_5_minus_base_8_in_base_10 :
  (5 * 5^4 + 2 * 5^3 + 1 * 5^2 + 4 * 5^1 + 3 * 5^0) -
  (4 * 8^3 + 3 * 8^2 + 1 * 8^1 + 0 * 8^0)
  = 1175 := by
  sorry

end base_5_minus_base_8_in_base_10_l203_203494


namespace bank_balance_after_two_years_l203_203208

theorem bank_balance_after_two_years :
  let P := 100 -- initial deposit
  let r := 0.1 -- annual interest rate
  let t := 2   -- time in years
  in P * (1 + r) ^ t = 121 :=
by
  sorry

end bank_balance_after_two_years_l203_203208


namespace geometric_sequence_value_l203_203538

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)
variable (r : α)
variable (a_pos : ∀ n, a n > 0)
variable (h1 : a 1 = 2)
variable (h99 : a 99 = 8)
variable (geom_seq : ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_value :
  a 20 * a 50 * a 80 = 64 := by
  sorry

end geometric_sequence_value_l203_203538


namespace angle_B_max_area_triangle_l203_203931
noncomputable section

open Real

variables {A B C a b c : ℝ}

-- Prove B = π / 3 given b sin A = √3 a cos B
theorem angle_B (h1 : b * sin A = sqrt 3 * a * cos B) : B = π / 3 :=
sorry

-- Prove if b = 2√3, the maximum area of triangle ABC is 3√3
theorem max_area_triangle (h1 : b * sin A = sqrt 3 * a * cos B) (h2 : b = 2 * sqrt 3) : 
    (1 / 2) * a * (a : ℝ) *  (sqrt 3 / 2 : ℝ) ≤ 3 * sqrt 3 :=
sorry

end angle_B_max_area_triangle_l203_203931


namespace gcd_of_360_and_150_is_30_l203_203425

theorem gcd_of_360_and_150_is_30 : Nat.gcd 360 150 = 30 :=
by
  sorry

end gcd_of_360_and_150_is_30_l203_203425


namespace correct_expression_must_hold_l203_203924

variable {f : ℝ → ℝ}

-- Conditions
axiom increasing_function : ∀ x y : ℝ, x < y → f x < f y
axiom positive_function : ∀ x : ℝ, f x > 0

-- Problem Statement
theorem correct_expression_must_hold : 3 * f (-2) > 2 * f (-3) := by
  sorry

end correct_expression_must_hold_l203_203924


namespace find_a4_l203_203325

variable (a : ℕ → ℤ)

def S (n : ℕ) : ℤ := (n * (a 1 + a n)) / 2

theorem find_a4 (h₁ : S 5 = 25) (h₂ : a 2 = 3) : a 4 = 7 := by
  sorry

end find_a4_l203_203325


namespace original_side_length_l203_203579

theorem original_side_length (x : ℝ) (h1 : (x - 6) * (x - 5) = 120) : x = 15 :=
sorry

end original_side_length_l203_203579


namespace four_digit_numbers_count_l203_203135

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l203_203135


namespace figure8_squares_figure12_perimeter_no_figure_C_figure29_figureD_ratio_l203_203397

-- Given conditions:
def initial_squares : ℕ := 3
def initial_perimeter : ℕ := 8
def squares_per_step : ℕ := 2
def perimeter_per_step : ℕ := 4

-- Statement proving Figure 8 has 17 squares
theorem figure8_squares : 3 + 2 * (8 - 1) = 17 := by sorry

-- Statement proving Figure 12 has a perimeter of 52 cm
theorem figure12_perimeter : 8 + 4 * (12 - 1) = 52 := by sorry

-- Statement proving no positive integer C yields perimeter of 38 cm
theorem no_figure_C : ¬∃ C : ℕ, 8 + 4 * (C - 1) = 38 := by sorry
  
-- Statement proving closest D giving the ratio for perimeter between Figure 29 and Figure D
theorem figure29_figureD_ratio : (8 + 4 * (29 - 1)) * 11 = 4 * (8 + 4 * (81 - 1)) := by sorry

end figure8_squares_figure12_perimeter_no_figure_C_figure29_figureD_ratio_l203_203397


namespace percentage_discount_of_retail_price_l203_203604

theorem percentage_discount_of_retail_price {wp rp sp discount : ℝ} (h1 : wp = 99) (h2 : rp = 132) (h3 : sp = wp + 0.20 * wp) (h4 : discount = (rp - sp) / rp * 100) : discount = 10 := 
by 
  sorry

end percentage_discount_of_retail_price_l203_203604


namespace cos_sum_identity_l203_203651

theorem cos_sum_identity (θ : ℝ) (h1 : Real.tan θ = -5 / 12) (h2 : θ ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  Real.cos (θ + Real.pi / 4) = 17 * Real.sqrt 2 / 26 :=
sorry

end cos_sum_identity_l203_203651


namespace fourth_number_in_15th_row_of_pascals_triangle_l203_203675

-- Here we state and prove the theorem about the fourth entry in the 15th row of Pascal's Triangle.
theorem fourth_number_in_15th_row_of_pascals_triangle : 
    (nat.choose 15 3) = 455 := 
by 
    sorry -- Proof is omitted as per instructions

end fourth_number_in_15th_row_of_pascals_triangle_l203_203675


namespace max_value_f_l203_203936

noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.sin (2 * x) + (a - 2) * Real.cos (2 * x)

theorem max_value_f (a : ℝ) (h : a < 0)
  (symm : ∀ x, f a (x - π / 4) = f a (-x - π / 4)) :
  ∃ x, f a x = 4 * Real.sqrt 2 :=
sorry

end max_value_f_l203_203936


namespace chores_minutes_proof_l203_203505

-- Definitions based on conditions
def minutes_of_cartoon_per_hour := 60
def cartoon_watched_hours := 2
def cartoon_watched_minutes := cartoon_watched_hours * minutes_of_cartoon_per_hour
def ratio_of_cartoon_to_chores := 10 / 8

-- Definition based on the question
def chores_minutes (cartoon_minutes : ℕ) : ℕ := (8 * cartoon_minutes) / 10

theorem chores_minutes_proof : chores_minutes cartoon_watched_minutes = 96 := 
by sorry 

end chores_minutes_proof_l203_203505


namespace greatest_overlap_l203_203247

-- Defining the conditions based on the problem statement
def percentage_internet (n : ℕ) : Prop := n = 35
def percentage_snacks (m : ℕ) : Prop := m = 70

-- The theorem to prove the greatest possible overlap
theorem greatest_overlap (n m k : ℕ) (hn : percentage_internet n) (hm : percentage_snacks m) : 
  k ≤ 35 :=
by sorry

end greatest_overlap_l203_203247


namespace park_area_l203_203402

variable (length width : ℝ)
variable (cost_per_meter total_cost : ℝ)
variable (ratio_length ratio_width : ℝ)
variable (x : ℝ)

def rectangular_park_ratio (length width : ℝ) (ratio_length ratio_width : ℝ) : Prop :=
  length / width = ratio_length / ratio_width

def fencing_cost (cost_per_meter total_cost : ℝ) (perimeter : ℝ) : Prop :=
  total_cost = cost_per_meter * perimeter

theorem park_area (length width : ℝ) (cost_per_meter total_cost : ℝ)
  (ratio_length ratio_width : ℝ) (x : ℝ)
  (h1 : rectangular_park_ratio length width ratio_length ratio_width)
  (h2 : cost_per_meter = 0.70)
  (h3 : total_cost = 175)
  (h4 : ratio_length = 3)
  (h5 : ratio_width = 2)
  (h6 : length = 3 * x)
  (h7 : width = 2 * x)
  (h8 : fencing_cost cost_per_meter total_cost (2 * (length + width))) :
  length * width = 3750 := by
  sorry

end park_area_l203_203402


namespace difference_of_squares_example_l203_203747

theorem difference_of_squares_example :
  (262^2 - 258^2 = 2080) :=
by {
  sorry -- placeholder for the actual proof
}

end difference_of_squares_example_l203_203747


namespace calendar_sum_l203_203180

theorem calendar_sum (n : ℕ) : 
    n + (n + 7) + (n + 14) = 3 * n + 21 :=
by sorry

end calendar_sum_l203_203180


namespace x_y_iff_pos_l203_203865

theorem x_y_iff_pos (x y : ℝ) : x + y > |x - y| ↔ x > 0 ∧ y > 0 := by
  sorry

end x_y_iff_pos_l203_203865


namespace expenses_neg_five_given_income_five_l203_203990

theorem expenses_neg_five_given_income_five 
  (income_5 : ℤ)
  (income_5_pos : income_5 = 5)
  (opposite : ∀ (x : ℤ), -x = -1 * x) :
  (expenses_5 : ℤ) (expenses_5 = -5) :=
by {
  sorry
}

end expenses_neg_five_given_income_five_l203_203990


namespace exists_special_sequence_l203_203622

open List
open Finset
open BigOperators

theorem exists_special_sequence :
  ∃ s : ℕ → ℕ,
    (∀ n, s n > 0) ∧
    (∀ i j, i ≠ j → s i ≠ s j) ∧
    (∀ k, (∑ i in range (k + 1), s i) % (k + 1) = 0) :=
sorry  -- Proof from the provided solution steps.

end exists_special_sequence_l203_203622


namespace tom_watching_days_l203_203707

def show_a_season_1_time : Nat := 20 * 22
def show_a_season_2_time : Nat := 18 * 24
def show_a_season_3_time : Nat := 22 * 26
def show_a_season_4_time : Nat := 15 * 30

def show_b_season_1_time : Nat := 24 * 42
def show_b_season_2_time : Nat := 16 * 48
def show_b_season_3_time : Nat := 12 * 55

def show_c_season_1_time : Nat := 10 * 60
def show_c_season_2_time : Nat := 13 * 58
def show_c_season_3_time : Nat := 15 * 50
def show_c_season_4_time : Nat := 11 * 52
def show_c_season_5_time : Nat := 9 * 65

def show_a_total_time : Nat :=
  show_a_season_1_time + show_a_season_2_time +
  show_a_season_3_time + show_a_season_4_time

def show_b_total_time : Nat :=
  show_b_season_1_time + show_b_season_2_time + show_b_season_3_time

def show_c_total_time : Nat :=
  show_c_season_1_time + show_c_season_2_time +
  show_c_season_3_time + show_c_season_4_time +
  show_c_season_5_time

def total_time : Nat := show_a_total_time + show_b_total_time + show_c_total_time

def daily_watch_time : Nat := 120

theorem tom_watching_days : (total_time + daily_watch_time - 1) / daily_watch_time = 64 := sorry

end tom_watching_days_l203_203707


namespace modulus_of_power_l203_203375

open Complex

theorem modulus_of_power (x y : ℚ) (h : (x ^ 2 + y ^ 2 = 1)) :
  ∀ n : ℤ, ∃ r : ℚ, |(x + y * Complex.I) ^ (2 * n) - 1| = r :=
by
  sorry

end modulus_of_power_l203_203375


namespace probability_x_plus_y_lt_4_l203_203470

open Set

-- Define the square and the line
def square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }
def line_lt_4 : Set (ℝ × ℝ) := { p | p.1 + p.2 < 4 }

-- The probability to prove
theorem probability_x_plus_y_lt_4 : 
  (volume (square ∩ line_lt_4) / volume square) = 7 / 9 := 
sorry

end probability_x_plus_y_lt_4_l203_203470


namespace initial_discount_percentage_l203_203736

variable (d : ℝ) (x : ℝ)
variable (h1 : 0 < d) (h2 : 0 ≤ x) (h3 : x ≤ 100)
variable (h4 : (1 - x / 100) * 0.6 * d = 0.33 * d)

theorem initial_discount_percentage : x = 45 :=
by
  sorry

end initial_discount_percentage_l203_203736


namespace count_four_digit_numbers_l203_203101

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l203_203101


namespace number_of_four_digit_numbers_l203_203156

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l203_203156


namespace find_sum_of_squares_of_roots_l203_203807

theorem find_sum_of_squares_of_roots:
  ∀ (a b c d : ℝ), (a^2 * b^2 * c^2 * d^2 - 15 * a * b * c * d + 56 = 0) → 
  a^2 + b^2 + c^2 + d^2 = 30 := by
  intros a b c d h
  sorry

end find_sum_of_squares_of_roots_l203_203807


namespace gcd_of_360_and_150_l203_203428

theorem gcd_of_360_and_150 : Nat.gcd 360 150 = 30 := 
by
  sorry

end gcd_of_360_and_150_l203_203428


namespace boys_in_class_l203_203218

theorem boys_in_class (r : ℕ) (g b : ℕ) (h1 : g/b = 4/3) (h2 : g + b = 35) : b = 15 :=
  sorry

end boys_in_class_l203_203218


namespace tan_degree_identity_l203_203073

theorem tan_degree_identity (k : ℝ) (hk : Real.cos (Real.pi * -80 / 180) = k) : 
  Real.tan (Real.pi * 100 / 180) = - (Real.sqrt (1 - k^2) / k) := 
by 
  sorry

end tan_degree_identity_l203_203073


namespace women_in_luxury_suites_count_l203_203822

noncomputable def passengers : ℕ := 300
noncomputable def percentage_women : ℝ := 70 / 100
noncomputable def percentage_luxury : ℝ := 15 / 100

noncomputable def women_on_ship : ℝ := passengers * percentage_women
noncomputable def women_in_luxury_suites : ℝ := women_on_ship * percentage_luxury

theorem women_in_luxury_suites_count : 
  round women_in_luxury_suites = 32 :=
by sorry

end women_in_luxury_suites_count_l203_203822


namespace complement_M_l203_203085

open Set

-- Define the universal set U as the set of all real numbers
def U := ℝ

-- Define the set M as {x | |x| > 2}
def M : Set ℝ := {x | |x| > 2}

-- State that the complement of M (in the universal set U) is [-2, 2]
theorem complement_M : Mᶜ = {x | -2 ≤ x ∧ x ≤ 2} :=
by
  sorry

end complement_M_l203_203085


namespace three_digit_reverse_sum_to_1777_l203_203458

theorem three_digit_reverse_sum_to_1777 :
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 101 * (a + c) + 20 * b = 1777 :=
by
  sorry

end three_digit_reverse_sum_to_1777_l203_203458


namespace sum_of_tens_and_ones_digits_pow_l203_203432

theorem sum_of_tens_and_ones_digits_pow : 
  let n := 7
  let exp := 12
  (n^exp % 100) / 10 + (n^exp % 10) = 1 :=
by
  sorry

end sum_of_tens_and_ones_digits_pow_l203_203432


namespace train_speed_l203_203240

theorem train_speed (l t: ℝ) (h1: l = 441) (h2: t = 21) : l / t = 21 := by
  sorry

end train_speed_l203_203240


namespace cos_of_three_pi_div_two_l203_203053

theorem cos_of_three_pi_div_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  sorry

end cos_of_three_pi_div_two_l203_203053


namespace ziggy_rap_requests_l203_203578

variables (total_songs electropop dance rock oldies djs_choice rap : ℕ)

-- Given conditions
axiom total_songs_eq : total_songs = 30
axiom electropop_eq : electropop = total_songs / 2
axiom dance_eq : dance = electropop / 3
axiom rock_eq : rock = 5
axiom oldies_eq : oldies = rock - 3
axiom djs_choice_eq : djs_choice = oldies / 2

-- Proof statement
theorem ziggy_rap_requests : rap = total_songs - electropop - dance - rock - oldies - djs_choice :=
by
  -- Apply the axioms and conditions to prove the resulting rap count
  sorry

end ziggy_rap_requests_l203_203578


namespace eva_total_marks_l203_203295

theorem eva_total_marks
    (math_score_s2 : ℕ) (arts_score_s2 : ℕ) (science_score_s2 : ℕ)
    (math_diff : ℕ) (arts_diff : ℕ) (science_frac_diff : ℚ)
    (math_score_s2_eq : math_score_s2 = 80)
    (arts_score_s2_eq : arts_score_s2 = 90)
    (science_score_s2_eq : science_score_s2 = 90)
    (math_diff_eq : math_diff = 10)
    (arts_diff_eq : arts_diff = 15)
    (science_frac_diff_eq : science_frac_diff = 1/3) : 
  (math_score_s2 + 10 + (math_score_s2 + math_diff) + 
   (arts_score_s2 + 90 - 15) + (arts_score_s2 + arts_diff) + 
   (science_score_s2 + 90 - (1/3) * 90) + (science_score_s2 + science_score_s2 * 1/3)) = 485 := 
by
  sorry

end eva_total_marks_l203_203295


namespace gcd_360_150_l203_203419

theorem gcd_360_150 : Int.gcd 360 150 = 30 := by
  have h360 : 360 = 2^3 * 3^2 * 5 := by
    ring
  have h150 : 150 = 2 * 3 * 5^2 := by
    ring
  rw [h360, h150]
  sorry

end gcd_360_150_l203_203419


namespace coefficient_of_friction_correct_l203_203251

noncomputable def coefficient_of_friction (R Fg: ℝ) (α: ℝ) : ℝ :=
  (1 - R * real.cos α) / (R * real.sin α)

theorem coefficient_of_friction_correct
  (Fg: ℝ)
  (α: ℝ)
  (R: ℝ := 11 * Fg)
  (hα: α = real.pi * 80 / 180):
  coefficient_of_friction R Fg α = 0.17 :=
by
  sorry

end coefficient_of_friction_correct_l203_203251


namespace probability_green_jelly_bean_l203_203248

theorem probability_green_jelly_bean :
  let red := 10
  let green := 9
  let yellow := 5
  let blue := 7
  let total := red + green + yellow + blue
  (green : ℚ) / (total : ℚ) = 9 / 31 := by
  sorry

end probability_green_jelly_bean_l203_203248


namespace expenses_of_five_yuan_l203_203994

theorem expenses_of_five_yuan (income_denotation : ℤ) (opposite_effect : ∀ x : ℤ, -x) :
  income_denotation = 5 → opposite_effect income_denotation = -5 :=
by sorry

end expenses_of_five_yuan_l203_203994


namespace expenses_of_5_yuan_l203_203986

-- Define the given condition: income of 5 yuan is +5 yuan
def income (x : Int) : Int := x

-- Define the opposite relationship between income and expenses
def expenses (x : Int) : Int := -income x

-- Proof statement to show that expenses of 5 yuan are -5 yuan, given the above definitions
theorem expenses_of_5_yuan : expenses 5 = -5 := by
  -- The proof is not provided here, so we use sorry to indicate its place
  sorry

end expenses_of_5_yuan_l203_203986


namespace remainder_of_division_l203_203310

noncomputable def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 1
noncomputable def g (x : ℝ) : ℝ := (x - 3) ^ 2
noncomputable def remainder (x : ℝ) : ℝ := 324 * x - 488

theorem remainder_of_division :
  ∀ (x : ℝ), (f x) % (g x) = remainder x :=
sorry

end remainder_of_division_l203_203310


namespace years_since_marriage_l203_203560

theorem years_since_marriage (x : ℕ) (ave_age_husband_wife_at_marriage : ℕ)
  (total_family_age_now : ℕ) (child_age : ℕ) (family_members : ℕ) :
  ave_age_husband_wife_at_marriage = 23 →
  total_family_age_now = 19 →
  child_age = 1 →
  family_members = 3 →
  (46 + 2 * x) + child_age = 57 →
  x = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end years_since_marriage_l203_203560


namespace cosine_of_3pi_over_2_l203_203051

theorem cosine_of_3pi_over_2 : Real.cos (3 * Real.pi / 2) = 0 := by
  sorry

end cosine_of_3pi_over_2_l203_203051


namespace minimum_positive_Sn_l203_203652

theorem minimum_positive_Sn (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ n, a (n+1) = a n + d) →
  a 11 / a 10 < -1 →
  (∃ N, ∀ n > N, S n < S (n + 1) ∧ S 1 ≤ S n ∧ ∀ n > N, S n < 0) →
  S 19 > 0 ∧ ∀ k < 19, S k > S 19 → S 19 < 0 →
  n = 19 :=
by
  sorry

end minimum_positive_Sn_l203_203652


namespace tg_equation_solution_l203_203020

noncomputable def tg (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tg_equation_solution (x : ℝ) (n : ℤ) 
    (h₀ : Real.cos x ≠ 0) :
    (tg x + tg (50 * Real.pi / 180) + tg (70 * Real.pi / 180) = 
    tg x * tg (50 * Real.pi / 180) * tg (70 * Real.pi / 180)) →
    (∃ n : ℤ, x = 60 * Real.pi / 180 + n * Real.pi) :=
sorry

end tg_equation_solution_l203_203020


namespace solve_cyclist_return_speed_l203_203870

noncomputable def cyclist_return_speed (D : ℝ) (V : ℝ) : Prop :=
  let avg_speed := 9.5
  let out_speed := 10
  let T_out := D / out_speed
  let T_back := D / V
  2 * D / (T_out + T_back) = avg_speed

theorem solve_cyclist_return_speed : ∀ (D : ℝ), cyclist_return_speed D (20 / 2.1) :=
by
  intro D
  sorry

end solve_cyclist_return_speed_l203_203870


namespace count_four_digit_numbers_l203_203098

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l203_203098


namespace new_person_weight_l203_203582

-- The conditions from part (a)
variables (average_increase: ℝ) (num_people: ℕ) (weight_lost_person: ℝ)
variables (total_increase: ℝ) (new_weight: ℝ)

-- Assigning the given conditions
axiom h1 : average_increase = 2.5
axiom h2 : num_people = 8
axiom h3 : weight_lost_person = 45
axiom h4 : total_increase = num_people * average_increase
axiom h5 : new_weight = weight_lost_person + total_increase

-- The proof goal: proving that the new person's weight is 65 kg
theorem new_person_weight : new_weight = 65 :=
by
  -- Proof steps go here
  sorry

end new_person_weight_l203_203582


namespace smallest_x_satisfies_abs_eq_l203_203758

theorem smallest_x_satisfies_abs_eq (x : ℝ) :
  (|2 * x + 5| = 21) → (x = -13) :=
sorry

end smallest_x_satisfies_abs_eq_l203_203758


namespace find_threedigit_number_l203_203455

-- Define the three-digit number and its reverse
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

-- Define the condition of adding the number and its reverse to get 1777
def number_sum_condition (a b c : ℕ) : Prop :=
  original_number a b c + reversed_number a b c = 1777

-- Prove the existence of digits a, b, and c that satisfy the conditions
theorem find_threedigit_number :
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  original_number a b c = 859 ∧ 
  reversed_number a b c = 958 ∧ 
  number_sum_condition a b c :=
sorry

end find_threedigit_number_l203_203455


namespace daughter_work_alone_12_days_l203_203594

/-- Given a man, his wife, and their daughter working together on a piece of work. The man can complete the work in 4 days, the wife in 6 days, and together with their daughter, they can complete it in 2 days. Prove that the daughter alone would take 12 days to complete the work. -/
theorem daughter_work_alone_12_days (h1 : (1/4 : ℝ) + (1/6) + D = 1/2) : D = 1/12 :=
by
  sorry

end daughter_work_alone_12_days_l203_203594


namespace three_digit_sum_reverse_eq_l203_203454

theorem three_digit_sum_reverse_eq :
  ∃ (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9),
    101 * (a + c) + 20 * b = 1777 ∧ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (9, 7, 8) :=
by
  sorry

end three_digit_sum_reverse_eq_l203_203454


namespace total_points_correct_l203_203285

-- Define the scores
def Marius (Darius : ℕ) : ℕ := Darius + 3
def Matt (Darius : ℕ) : ℕ := Darius + 5

-- Define the total points function
def total_points (Darius : ℕ) : ℕ :=
  Darius + Marius Darius + Matt Darius

-- Specific value for Darius's score
def Darius_score : ℕ := 10

-- The theorem that proves the total score is 38 given Darius's score
theorem total_points_correct :
  total_points Darius_score = 38 :=
by
  sorry

end total_points_correct_l203_203285


namespace alex_casey_meet_probability_l203_203483

noncomputable def probability_meet : ℚ :=
  let L := (1:ℚ) / 3;
  let area_of_square := 1;
  let area_of_triangles := (1 / 2) * L ^ 2;
  let area_of_meeting_region := area_of_square - 2 * area_of_triangles;
  area_of_meeting_region / area_of_square

theorem alex_casey_meet_probability :
  probability_meet = 8 / 9 :=
by
  sorry

end alex_casey_meet_probability_l203_203483


namespace picture_books_count_l203_203844

-- Definitions based on the given conditions
def total_books : ℕ := 35
def fiction_books : ℕ := 5
def non_fiction_books : ℕ := fiction_books + 4
def autobiographies : ℕ := 2 * fiction_books
def total_non_picture_books : ℕ := fiction_books + non_fiction_books + autobiographies
def picture_books : ℕ := total_books - total_non_picture_books

-- Statement of the problem
theorem picture_books_count : picture_books = 11 :=
by sorry

end picture_books_count_l203_203844


namespace fixed_point_l203_203696

noncomputable def f (a : ℝ) (x : ℝ) := a^(x - 2) - 3

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 2 = -2 :=
by
  sorry

end fixed_point_l203_203696


namespace line_symmetric_fixed_point_l203_203942

theorem line_symmetric_fixed_point (k : ℝ) :
  (∀ x, (∃ y, y = k * (x - 4))) →
  (∃ p : ℝ × ℝ, p = (2, 1) ∧ ∀ x, (∃ y, y = k * (x - 4))) →
  (∃ p : ℝ × ℝ, p = (2, 1)) →
  (∃ q : ℝ × ℝ, q = (0, 2)) →
  True := 
by sorry

end line_symmetric_fixed_point_l203_203942


namespace part_a_part_b_part_c_l203_203974

-- Part a
theorem part_a (n: ℕ) (h: n = 1): (n^2 - 5 * n + 4) / (n - 4) = 0 := by sorry

-- Part b
theorem part_b (n: ℕ) (h: (n^2 - 5 * n + 4) / (n - 4) = 5): n = 6 := 
  by sorry

-- Part c
theorem part_c (n: ℕ) (h : n ≠ 4): (n^2 - 5 * n + 4) / (n - 4) ≠ 3 := 
  by sorry

end part_a_part_b_part_c_l203_203974


namespace becky_to_aliyah_ratio_l203_203409

def total_school_days : ℕ := 180
def days_aliyah_packs_lunch : ℕ := total_school_days / 2
def days_becky_packs_lunch : ℕ := 45

theorem becky_to_aliyah_ratio :
  (days_becky_packs_lunch : ℚ) / days_aliyah_packs_lunch = 1 / 2 := by
  sorry

end becky_to_aliyah_ratio_l203_203409


namespace problem_statement_l203_203371

theorem problem_statement 
  (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z = 945) :
  2 * w + 3 * x + 5 * y + 7 * z = 21 :=
by
  sorry

end problem_statement_l203_203371


namespace integer_values_of_n_summing_to_24_l203_203907

theorem integer_values_of_n_summing_to_24 :
  {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13} = {11, 13} ∧ 11 + 13 = 24 :=
by
  sorry

end integer_values_of_n_summing_to_24_l203_203907


namespace can_form_triangle_l203_203434

theorem can_form_triangle (a b c : ℕ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 10) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  rw [h1, h2, h3]
  repeat {sorry}

end can_form_triangle_l203_203434


namespace parabola_focus_condition_l203_203527

theorem parabola_focus_condition (m : ℝ) : (∃ (x y : ℝ), x + y - 2 = 0 ∧ y = (1 / (4 * m))) → m = 1 / 8 :=
by
  sorry

end parabola_focus_condition_l203_203527


namespace smallest_sum_arith_geo_sequence_l203_203217

theorem smallest_sum_arith_geo_sequence :
  ∃ (X Y Z W : ℕ),
    X < Y ∧ Y < Z ∧ Z < W ∧
    (2 * Y = X + Z) ∧
    (Y ^ 2 = Z * X) ∧
    (Z / Y = 7 / 4) ∧
    (X + Y + Z + W = 97) :=
by
  sorry

end smallest_sum_arith_geo_sequence_l203_203217


namespace probability_of_x_plus_y_lt_4_l203_203478

open Classical -- To handle probability and random selection
open Set -- For geometric notions
open Filter -- For measure and integration

noncomputable def probability_condition (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) : Prop := x + y < 4

theorem probability_of_x_plus_y_lt_4 :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) →
                pr (λ (xy : ℝ × ℝ), probability_condition xy.1 xy.2 (and.intro (and.left xy.2) (and.right (and.left xy.2)))) = 7/9 :=
by sorry

end probability_of_x_plus_y_lt_4_l203_203478


namespace econ_not_feasible_l203_203510

theorem econ_not_feasible (x y p q: ℕ) (h_xy : 26 * x + 29 * y = 687) (h_pq : 27 * p + 31 * q = 687) : p + q ≥ x + y := by
  sorry

end econ_not_feasible_l203_203510


namespace real_roots_exists_of_a_ne_zero_real_roots_exists_of_a_ne_one_l203_203314

theorem real_roots_exists_of_a_ne_zero (a : ℝ) : 
  (a ≠ 0) → (a < -5 - 2*Real.sqrt 6 ∨ 2*Real.sqrt 6 - 5 < a ∧ a < 0 ∨ a > 0) ↔ 
  (∃ x : ℝ, a*x^2 + (a+1)*x - 2 = 0) := sorry

theorem real_roots_exists_of_a_ne_one (a : ℝ) : 
  (a ≠ 1) → (a < 1 ∨ (1 < a ∧ a < 3) ∨ a > 3) ↔ 
  (∃ x : ℝ, (1 - a)*x^2 + (a + 1)*x - 2 = 0) := sorry

end real_roots_exists_of_a_ne_zero_real_roots_exists_of_a_ne_one_l203_203314


namespace probability_x_plus_y_less_than_4_l203_203465

open Set Real

/-- Define the square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3

/-- Probability that a randomly chosen point (x, y) in the square satisfies x + y < 4 -/
theorem probability_x_plus_y_less_than_4 : 
  let area_square := 9 in
  let area_excluded_triangle := 2 in
  let relevant_area := area_square - area_excluded_triangle in
  (relevant_area / area_square : ℝ) = 7 / 9 :=
by
{ sorry }

end probability_x_plus_y_less_than_4_l203_203465


namespace middle_rungs_widths_l203_203001

theorem middle_rungs_widths (a : ℕ → ℝ) (d : ℝ) :
  a 1 = 33 ∧ a 12 = 110 ∧ (∀ n, a (n + 1) = a n + 7) →
  (a 2 = 40 ∧ a 3 = 47 ∧ a 4 = 54 ∧ a 5 = 61 ∧
   a 6 = 68 ∧ a 7 = 75 ∧ a 8 = 82 ∧ a 9 = 89 ∧
   a 10 = 96 ∧ a 11 = 103) :=
by
  sorry

end middle_rungs_widths_l203_203001


namespace vector_expression_l203_203783

variables (a b c : ℝ × ℝ)
variables (m n : ℝ)

noncomputable def vec_a : ℝ × ℝ := (1, 1)
noncomputable def vec_b : ℝ × ℝ := (1, -1)
noncomputable def vec_c : ℝ × ℝ := (-1, 2)

/-- Prove that vector c can be expressed in terms of vectors a and b --/
theorem vector_expression : 
  vec_c = m • vec_a + n • vec_b → (m = 1/2 ∧ n = -3/2) :=
sorry

end vector_expression_l203_203783


namespace problem1_problem2_l203_203447

noncomputable def expression1 : ℝ :=
  (0.064: ℝ) ^ (-1/3: ℝ) - (-7/8: ℝ) ^ (0: ℝ) + ((-2: ℝ) ^ 3) ^ (-4/3: ℝ) + (16: ℝ) ^ (-0.25: ℝ)

theorem problem1 : expression1 = 33 / 16 := by
  sorry

noncomputable def expression2 : ℝ :=
  abs ((4 / 9: ℝ) ^ (-1/2) - real.log 5) + real.sqrt ((real.log 2) ^ 2 - real.log 4 + 1) - 3 ^ (1 - real.log 3 2)

theorem problem2 : expression2 = 0 := by
  sorry

end problem1_problem2_l203_203447


namespace find_m_l203_203930

noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2*k + 1

theorem find_m (m : ℕ) (h₀ : 0 < m) (h₁ : (m ^ 2 - 2 * m - 3:ℤ) < 0) (h₂ : is_odd (m ^ 2 - 2 * m - 3)) : m = 2 := 
sorry

end find_m_l203_203930


namespace quadratic_no_real_roots_l203_203602

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_no_real_roots
  (a b c: ℝ)
  (h1: ((b - 1)^2 - 4 * a * (c + 1) = 0))
  (h2: ((b + 2)^2 - 4 * a * (c - 2) = 0)) :
  ∀ x : ℝ, f a b c x ≠ 0 := 
sorry

end quadratic_no_real_roots_l203_203602


namespace games_within_division_l203_203249

theorem games_within_division (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 2 * N + 6 * M = 76) : 2 * N = 40 :=
by {
  sorry
}

end games_within_division_l203_203249


namespace prob_event_A_given_B_l203_203237

def EventA (visits : Fin 4 → Fin 4) : Prop :=
  Function.Injective visits

def EventB (visits : Fin 4 → Fin 4) : Prop :=
  visits 0 = 0

theorem prob_event_A_given_B :
  ∀ (visits : Fin 4 → Fin 4),
  (∃ f : (Fin 4 → Fin 4) → Prop, f visits → (EventA visits ∧ EventB visits)) →
  (∃ P : ℚ, P = 2 / 9) :=
by
  intros visits h
  -- Proof omitted
  sorry

end prob_event_A_given_B_l203_203237


namespace sum_abcd_l203_203545

variable (a b c d : ℝ)

theorem sum_abcd :
  (∃ y : ℝ, 2 * a + 3 = y ∧ 2 * b + 4 = y ∧ 2 * c + 5 = y ∧ 2 * d + 6 = y ∧ a + b + c + d + 10 = y) →
  a + b + c + d = -11 :=
by
  sorry

end sum_abcd_l203_203545


namespace billy_age_l203_203890

theorem billy_age (B J : ℕ) (h1 : B = 3 * J) (h2 : B + J = 60) : B = 45 :=
by
  sorry

end billy_age_l203_203890


namespace coordinates_of_P_l203_203973

-- Define a structure for a 2D point
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- Define the distance from a point to the x-axis
def distance_to_x_axis (P : Point) : ℝ :=
  |P.y|

-- Define the distance from a point to the y-axis
def distance_to_y_axis (P : Point) : ℝ :=
  |P.x|

-- The main proof statement
theorem coordinates_of_P (P : Point) :
  in_third_quadrant P →
  distance_to_x_axis P = 2 →
  distance_to_y_axis P = 5 →
  P = { x := -5, y := -2 } :=
by
  intros h1 h2 h3
  sorry

end coordinates_of_P_l203_203973


namespace people_in_room_after_2019_minutes_l203_203790

theorem people_in_room_after_2019_minutes :
  ∀ (P : Nat → Int), 
    P 0 = 0 -> 
    (∀ t, P (t+1) = P t + 2 ∨ P (t+1) = P t - 1) -> 
    P 2019 ≠ 2018 :=
by
  intros P hP0 hP_changes
  sorry

end people_in_room_after_2019_minutes_l203_203790


namespace sum_of_values_l203_203406

theorem sum_of_values :
  1 + 0.01 + 0.0001 = 1.0101 :=
by sorry

end sum_of_values_l203_203406


namespace wine_barrels_l203_203952

theorem wine_barrels :
  ∃ x y : ℝ, (6 * x + 4 * y = 48) ∧ (5 * x + 3 * y = 38) :=
by
  -- Proof is left out
  sorry

end wine_barrels_l203_203952


namespace four_digit_number_count_l203_203140

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l203_203140


namespace joan_dimes_l203_203437

theorem joan_dimes :
  ∀ (total_dimes_jacket : ℕ) (total_money : ℝ) (value_per_dime : ℝ),
    total_dimes_jacket = 15 →
    total_money = 1.90 →
    value_per_dime = 0.10 →
    ((total_money - (total_dimes_jacket * value_per_dime)) / value_per_dime) = 4 :=
by
  intros total_dimes_jacket total_money value_per_dime h1 h2 h3
  sorry

end joan_dimes_l203_203437


namespace unique_arrangements_MOON_l203_203896

theorem unique_arrangements_MOON : 
  let M := 1
  let O := 2
  let N := 1
  let total_letters := 4
  (Nat.factorial total_letters / (Nat.factorial O)) = 12 :=
by
  sorry

end unique_arrangements_MOON_l203_203896


namespace part_I_part_II_l203_203552

open Real

noncomputable def f (x : ℝ) : ℝ := log ((2 / (x + 1)) - 1)

def g (x a : ℝ) : ℝ := -x^2 + 2 * x + a

-- Domain of function f
def A : Set ℝ := {x | -1 < x ∧ x < 1}

-- Range of function g with a given condition on x
def B (a : ℝ) : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ y = g x a}

theorem part_I : f (1 / 2015) + f (-1 / 2015) = 0 := sorry

theorem part_II (a : ℝ) : (A ∩ B a) = ∅ ↔ a ≤ -2 ∨ a ≥ 4 := sorry

end part_I_part_II_l203_203552


namespace Josanna_min_avg_score_l203_203797

theorem Josanna_min_avg_score (scores : List ℕ) (cur_avg target_avg : ℚ)
  (next_test_bonus : ℚ) (additional_avg_points : ℚ) : ℚ :=
  let cur_avg := (92 + 81 + 75 + 65 + 88) / 5
  let target_avg := cur_avg + 6
  let needed_total := target_avg * 7
  let additional_points := 401 + 5
  let needed_sum := needed_total - additional_points
  needed_sum / 2

noncomputable def min_avg_score : ℚ :=
  Josanna_min_avg_score [92, 81, 75, 65, 88] 80.2 86.2 5 6

example : min_avg_score = 99 :=
by
  sorry

end Josanna_min_avg_score_l203_203797


namespace range_of_m_l203_203529

variable {R : Type*} [LinearOrderedField R]

def discriminant (a b c : R) := b * b - 4 * a * c

theorem range_of_m (m : R) : (∀ x : R, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by sorry

end range_of_m_l203_203529


namespace three_digit_reverse_sum_to_1777_l203_203460

theorem three_digit_reverse_sum_to_1777 :
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 101 * (a + c) + 20 * b = 1777 :=
by
  sorry

end three_digit_reverse_sum_to_1777_l203_203460


namespace intersection_eq_l203_203656

-- Universal set and its sets M and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 > 9}
def N : Set ℝ := {x | -1 < x ∧ x < 4}
def complement_N : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}

-- Prove the intersection
theorem intersection_eq :
  M ∩ complement_N = {x | x < -3 ∨ x ≥ 4} :=
by
  sorry

end intersection_eq_l203_203656


namespace bananas_first_day_l203_203593

theorem bananas_first_day (x : ℕ) (h : x + (x + 6) + (x + 12) + (x + 18) + (x + 24) = 100) : x = 8 := by
  sorry

end bananas_first_day_l203_203593


namespace find_number_l203_203253

theorem find_number (x : ℕ) (h1 : x > 7) (h2 : x ≠ 8) : x = 9 := by
  sorry

end find_number_l203_203253


namespace simplify_log_expression_l203_203976

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem simplify_log_expression :
  let term1 := 1 / (log_base 20 3 + 1)
  let term2 := 1 / (log_base 12 5 + 1)
  let term3 := 1 / (log_base 8 7 + 1)
  term1 + term2 + term3 = 2 :=
by
  sorry

end simplify_log_expression_l203_203976


namespace intersection_of_M_and_N_l203_203660

open Set

noncomputable def M := {x : ℝ | ∃ y:ℝ, y = Real.log (2 - x)}
noncomputable def N := {x : ℝ | x^2 - 3*x - 4 ≤ 0 }
noncomputable def I := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = I := 
  sorry

end intersection_of_M_and_N_l203_203660


namespace f_is_periodic_l203_203373

noncomputable def f : ℝ → ℝ := sorry

def a : ℝ := sorry

axiom exists_a_gt_zero : a > 0

axiom functional_eq (x : ℝ) : f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)

theorem f_is_periodic : ∀ x : ℝ, f (x + 2 * a) = f x := sorry

end f_is_periodic_l203_203373


namespace union_of_complements_l203_203333

def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {x | x^2 + 4 = 5 * x}
def complement_U (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem union_of_complements :
  complement_U A ∪ complement_U B = {0, 2, 3, 4, 5} := by
sorry

end union_of_complements_l203_203333


namespace combinations_of_coins_l203_203777

noncomputable def count_combinations (target : ℕ) : ℕ :=
  (30 - 0*0) -- As it just returns 45 combinations

theorem combinations_of_coins : count_combinations 30 = 45 :=
  sorry

end combinations_of_coins_l203_203777


namespace symmetric_points_origin_l203_203673

theorem symmetric_points_origin (a b : ℝ) (h1 : a = -(-2)) (h2 : 1 = -b) : a + b = 1 :=
by
  sorry

end symmetric_points_origin_l203_203673


namespace zero_count_non_decreasing_zero_count_tends_to_2N_l203_203238

noncomputable def f (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  ∑ k in Finset.range (Nat.succ N), a k * Real.sin (2 * k * π * x)

def number_of_zeros (f : ℝ → ℝ) (i : ℕ) (interval : Set.Ici (0 : ℝ)) : ℕ :=
  (Set.Icc (0 : ℝ) 1).count {x | deriv^[i] f x = 0}

theorem zero_count_non_decreasing (a : ℕ → ℝ) (h : ∀ k, k ≥ N → a k = 0) (h_nonzero : a N ≠ 0) 
  (i : ℕ) :
  ∀ j ≥ i, number_of_zeros (f a) j (set.Ici 0) ≥ number_of_zeros (f a) i (set.Ici 0) :=
sorry

theorem zero_count_tends_to_2N (a : ℕ → ℝ) (h : ∀ k, k ≥ N → a k = 0) (h_nonzero : a N ≠ 0) :
  limit (λ i, number_of_zeros (f a) i (set.Ici 0)) at_top = 2 * N :=
sorry

end zero_count_non_decreasing_zero_count_tends_to_2N_l203_203238


namespace samBill_l203_203597

def textMessageCostPerText := 8 -- cents
def extraMinuteCostPerMinute := 15 -- cents
def planBaseCost := 25 -- dollars
def includedPlanHours := 25
def centToDollar (cents: Nat) : Nat := cents / 100

def totalBill (texts: Nat) (hours: Nat) : Nat :=
  let textCost := centToDollar (texts * textMessageCostPerText)
  let extraHours := if hours > includedPlanHours then hours - includedPlanHours else 0
  let extraMinutes := extraHours * 60
  let extraMinuteCost := centToDollar (extraMinutes * extraMinuteCostPerMinute)
  planBaseCost + textCost + extraMinuteCost

theorem samBill :
  totalBill 150 26 = 46 := 
sorry

end samBill_l203_203597


namespace count_four_digit_numbers_l203_203145

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l203_203145


namespace annulus_area_of_tangent_segments_l203_203321

theorem annulus_area_of_tangent_segments (r : ℝ) (l : ℝ) (region_area : ℝ) 
  (h_rad : r = 3) (h_len : l = 6) : region_area = 9 * Real.pi :=
sorry

end annulus_area_of_tangent_segments_l203_203321


namespace inverse_proportion_graph_l203_203772

theorem inverse_proportion_graph (m n : ℝ) (h : n = -2 / m) : m = -2 / n :=
by
  sorry

end inverse_proportion_graph_l203_203772


namespace count_four_digit_numbers_l203_203102

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l203_203102


namespace max_height_reached_threat_to_object_at_70km_l203_203833

noncomputable def initial_acceleration : ℝ := 20 -- m/s^2
noncomputable def duration : ℝ := 50 -- seconds
noncomputable def gravity : ℝ := 10 -- m/s^2
noncomputable def height_at_max_time : ℝ := 75000 -- meters (75km)

-- Proof that the maximum height reached is 75 km
theorem max_height_reached (a τ g : ℝ) (H : ℝ) (h₀: a = initial_acceleration) (h₁: τ = duration) (h₂: g = gravity) (h₃: H = height_at_max_time) :
  H = 75 * 1000 := 
sorry

-- Proof that the rocket poses a threat to an object located at 70 km
theorem threat_to_object_at_70km (a τ g : ℝ) (H : ℝ) (h₀: a = initial_acceleration) (h₁: τ = duration) (h₂: g = gravity) (h₃: H = height_at_max_time) :
  H > 70 * 1000 :=
sorry

end max_height_reached_threat_to_object_at_70km_l203_203833


namespace sum_of_squares_of_ages_l203_203781

theorem sum_of_squares_of_ages {a b c : ℕ} (h1 : 5 * a + b = 3 * c) (h2 : 3 * c^2 = 2 * a^2 + b^2) 
  (relatively_prime : Nat.gcd (Nat.gcd a b) c = 1) : 
  a^2 + b^2 + c^2 = 374 :=
by
  sorry

end sum_of_squares_of_ages_l203_203781


namespace grid_3x3_unique_72_l203_203730

theorem grid_3x3_unique_72 :
  ∃ (f : Fin 3 → Fin 3 → ℕ), 
    (∀ (i j : Fin 3), 1 ≤ f i j ∧ f i j ≤ 9) ∧
    (∀ (i j k : Fin 3), j < k → f i j < f i k) ∧
    (∀ (i j k : Fin 3), i < k → f i j < f k j) ∧
    f 0 0 = 1 ∧ f 1 1 = 5 ∧ f 2 2 = 8 ∧
    (∃! (g : Fin 3 → Fin 3 → ℕ), 
      (∀ (i j : Fin 3), 1 ≤ g i j ∧ g i j ≤ 9) ∧
      (∀ (i j k : Fin 3), j < k → g i j < g i k) ∧
      (∀ (i j k : Fin 3), i < k → g i j < g k j) ∧
      g 0 0 = 1 ∧ g 1 1 = 5 ∧ g 2 2 = 8) :=
sorry

end grid_3x3_unique_72_l203_203730


namespace probability_digits_different_l203_203485

noncomputable def probability_all_digits_different : ℚ :=
  have tens_digits_probability := (9 / 9) * (8 / 9) * (7 / 9)
  have ones_digits_probability := (10 / 10) * (9 / 10) * (8 / 10)
  (tens_digits_probability * ones_digits_probability)

theorem probability_digits_different :
  probability_all_digits_different = 112 / 225 :=
by 
  -- The proof would go here, but it is not required for this task.
  sorry

end probability_digits_different_l203_203485


namespace domain_all_real_numbers_l203_203305

theorem domain_all_real_numbers (k : ℝ) :
  (∀ x : ℝ, -7 * x^2 - 4 * x + k ≠ 0) ↔ k < -4 / 7 := by
  sorry

end domain_all_real_numbers_l203_203305


namespace sqrt_solution_range_l203_203503

theorem sqrt_solution_range : 
  7 < (Real.sqrt 32) * (Real.sqrt (1 / 2)) + (Real.sqrt 12) ∧ (Real.sqrt 32) * (Real.sqrt (1 / 2)) + (Real.sqrt 12) < 8 := 
by
  sorry

end sqrt_solution_range_l203_203503


namespace xyz_divides_xyz_squared_l203_203544

theorem xyz_divides_xyz_squared (x y z p : ℕ) (hxyz : x < y ∧ y < z ∧ z < p) (hp : Nat.Prime p) (hx3 : x^3 ≡ y^3 [MOD p])
    (hy3 : y^3 ≡ z^3 [MOD p]) (hz3 : z^3 ≡ x^3 [MOD p]) : (x + y + z) ∣ (x^2 + y^2 + z^2) :=
by
  sorry

end xyz_divides_xyz_squared_l203_203544


namespace line_tangent_to_parabola_l203_203031

theorem line_tangent_to_parabola (d : ℝ) :
  (∀ x y: ℝ, y = 3 * x + d ↔ y^2 = 12 * x) → d = 1 :=
by
  sorry

end line_tangent_to_parabola_l203_203031


namespace expenses_neg_five_given_income_five_l203_203992

theorem expenses_neg_five_given_income_five 
  (income_5 : ℤ)
  (income_5_pos : income_5 = 5)
  (opposite : ∀ (x : ℤ), -x = -1 * x) :
  (expenses_5 : ℤ) (expenses_5 = -5) :=
by {
  sorry
}

end expenses_neg_five_given_income_five_l203_203992


namespace completing_the_square_l203_203017

theorem completing_the_square (x : ℝ) :
  4 * x^2 - 2 * x - 1 = 0 → (x - 1/4)^2 = 5/16 := 
by
  sorry

end completing_the_square_l203_203017


namespace expenses_of_5_yuan_l203_203988

-- Define the given condition: income of 5 yuan is +5 yuan
def income (x : Int) : Int := x

-- Define the opposite relationship between income and expenses
def expenses (x : Int) : Int := -income x

-- Proof statement to show that expenses of 5 yuan are -5 yuan, given the above definitions
theorem expenses_of_5_yuan : expenses 5 = -5 := by
  -- The proof is not provided here, so we use sorry to indicate its place
  sorry

end expenses_of_5_yuan_l203_203988


namespace andy_wrong_questions_l203_203887

theorem andy_wrong_questions
  (a b c d : ℕ)
  (h1 : a + b = c + d + 6)
  (h2 : a + d = b + c + 4)
  (h3 : c = 10) :
  a = 15 :=
by
  sorry

end andy_wrong_questions_l203_203887


namespace cos_of_three_pi_div_two_l203_203052

theorem cos_of_three_pi_div_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  sorry

end cos_of_three_pi_div_two_l203_203052


namespace percentage_non_honda_red_cars_l203_203784

theorem percentage_non_honda_red_cars 
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (toyota_cars : ℕ)
  (ford_cars : ℕ)
  (other_cars : ℕ)
  (perc_red_honda : ℕ)
  (perc_red_toyota : ℕ)
  (perc_red_ford : ℕ)
  (perc_red_other : ℕ)
  (perc_total_red : ℕ)
  (hyp_total_cars : total_cars = 900)
  (hyp_honda_cars : honda_cars = 500)
  (hyp_toyota_cars : toyota_cars = 200)
  (hyp_ford_cars : ford_cars = 150)
  (hyp_other_cars : other_cars = 50)
  (hyp_perc_red_honda : perc_red_honda = 90)
  (hyp_perc_red_toyota : perc_red_toyota = 75)
  (hyp_perc_red_ford : perc_red_ford = 30)
  (hyp_perc_red_other : perc_red_other = 20)
  (hyp_perc_total_red : perc_total_red = 60) :
  (205 / 400) * 100 = 51.25 := 
by {
  sorry
}

end percentage_non_honda_red_cars_l203_203784


namespace odd_positive_93rd_l203_203009

theorem odd_positive_93rd : 
  (2 * 93 - 1) = 185 := 
by sorry

end odd_positive_93rd_l203_203009


namespace number_of_dimes_l203_203688

theorem number_of_dimes (x : ℕ) (h1 : 10 * x + 25 * x + 50 * x = 2040) : x = 24 :=
by {
  -- The proof will go here if you need to fill it out.
  sorry
}

end number_of_dimes_l203_203688


namespace number_of_four_digit_numbers_l203_203152

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l203_203152


namespace count_four_digit_numbers_l203_203151

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l203_203151


namespace range_of_p_l203_203194

def A := {x : ℝ | x^2 - x - 2 > 0}
def B := {x : ℝ | (3 / x) - 1 ≥ 0}
def intersection := {x : ℝ | x ∈ A ∧ x ∈ B}
def C (p : ℝ) := {x : ℝ | 2 * x + p ≤ 0}

theorem range_of_p (p : ℝ) : (∀ x : ℝ, x ∈ intersection → x ∈ C p) → p < -6 := by
  sorry

end range_of_p_l203_203194


namespace four_digit_numbers_count_l203_203108

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l203_203108


namespace more_bags_found_l203_203007

def bags_Monday : ℕ := 7
def bags_nextDay : ℕ := 12

theorem more_bags_found : bags_nextDay - bags_Monday = 5 := by
  -- Proof Skipped
  sorry

end more_bags_found_l203_203007


namespace factor_expression_l203_203906

-- Define the expression to be factored
def expr (b : ℝ) := 348 * b^2 + 87 * b + 261

-- Define the supposedly factored form of the expression
def factored_expr (b : ℝ) := 87 * (4 * b^2 + b + 3)

-- The theorem stating that the original expression is equal to its factored form
theorem factor_expression (b : ℝ) : expr b = factored_expr b := 
by
  unfold expr factored_expr
  sorry

end factor_expression_l203_203906


namespace computation_result_l203_203498

theorem computation_result :
  2 + 8 * 3 - 4 + 7 * 2 / 2 * 3 = 43 :=
by
  sorry

end computation_result_l203_203498


namespace carolyn_sum_of_removed_numbers_eq_31_l203_203892

theorem carolyn_sum_of_removed_numbers_eq_31 :
  let initial_list := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let carolyn_first_turn := 4
  let carolyn_numbers_removed := [4, 9, 10, 8]
  let sum := carolyn_numbers_removed.sum
  sum = 31 :=
by
  sorry

end carolyn_sum_of_removed_numbers_eq_31_l203_203892


namespace problem_part_1_problem_part_2_l203_203761

theorem problem_part_1 (a b c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x ^ 2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_cond : ∀ x, -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) :
  |c| ≤ 1 :=
by
  sorry

theorem problem_part_2 (a b c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x ^ 2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_cond : ∀ x, -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → |g x| ≤ 2 :=
by
  sorry

end problem_part_1_problem_part_2_l203_203761


namespace min_value_of_y_l203_203307

noncomputable def y (x : ℝ) : ℝ :=
  2 * Real.sin (Real.pi / 3 - x) - Real.cos (Real.pi / 6 + x)

theorem min_value_of_y : ∃ x : ℝ, y x = -1 := by
  sorry

end min_value_of_y_l203_203307


namespace expression_evaluation_l203_203272

theorem expression_evaluation : 
  (3.14 - Real.pi)^0 + abs (Real.sqrt 2 - 1) + (1 / 2)^(-1:ℤ) - Real.sqrt 8 = 2 - Real.sqrt 2 :=
by sorry

end expression_evaluation_l203_203272


namespace uniquely_determine_T_l203_203509

theorem uniquely_determine_T'_n (b e : ℤ) (S' T' : ℕ → ℤ)
  (hb : ∀ n, S' n = n * (2 * b + (n - 1) * e) / 2)
  (ht : ∀ n, T' n = n * (n + 1) * (3 * b + (n - 1) * e) / 6)
  (h3028 : S' 3028 = 3028 * (b + 1514 * e)) :
  T' 4543 = (4543 * (4543 + 1) * (3 * b + 4542 * e)) / 6 :=
by
  sorry

end uniquely_determine_T_l203_203509


namespace inequality_solution_l203_203558

theorem inequality_solution (x : ℝ) : |x - 3| + |x - 5| ≥ 4 → x ≥ 6 ∨ x ≤ 2 :=
by
  sorry

end inequality_solution_l203_203558


namespace number_of_toys_l203_203463

-- Definitions based on conditions
def selling_price : ℝ := 18900
def cost_price_per_toy : ℝ := 900
def gain_per_toy : ℝ := 3 * cost_price_per_toy

-- The number of toys sold
noncomputable def number_of_toys_sold (SP CP gain : ℝ) : ℝ :=
  (SP - gain) / CP

-- The theorem statement to prove
theorem number_of_toys (SP CP gain : ℝ) : number_of_toys_sold SP CP gain = 18 :=
by
  have h1: SP = 18900 := by sorry
  have h2: CP = 900 := by sorry
  have h3: gain = 3 * CP := by sorry
  -- Further steps to establish the proof
  sorry

end number_of_toys_l203_203463


namespace probability_x_plus_y_less_than_4_l203_203466

open Set Real

/-- Define the square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3

/-- Probability that a randomly chosen point (x, y) in the square satisfies x + y < 4 -/
theorem probability_x_plus_y_less_than_4 : 
  let area_square := 9 in
  let area_excluded_triangle := 2 in
  let relevant_area := area_square - area_excluded_triangle in
  (relevant_area / area_square : ℝ) = 7 / 9 :=
by
{ sorry }

end probability_x_plus_y_less_than_4_l203_203466


namespace four_digit_numbers_count_eq_l203_203090

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l203_203090


namespace probability_all_three_dice_twenty_l203_203442

theorem probability_all_three_dice_twenty (d1 d2 d3 d4 d5 : ℕ)
  (h1 : 1 ≤ d1 ∧ d1 ≤ 20) (h2 : 1 ≤ d2 ∧ d2 ≤ 20) (h3 : 1 ≤ d3 ∧ d3 ≤ 20)
  (h4 : 1 ≤ d4 ∧ d4 ≤ 20) (h5 : 1 ≤ d5 ∧ d5 ≤ 20)
  (h6 : d1 = 20) (h7 : d2 = 19)
  (h8 : (if d1 = 20 then 1 else 0) + (if d2 = 20 then 1 else 0) +
        (if d3 = 20 then 1 else 0) + (if d4 = 20 then 1 else 0) +
        (if d5 = 20 then 1 else 0) ≥ 3) :
  (1 / 58 : ℚ) = (if d3 = 20 ∧ d4 = 20 ∧ d5 = 20 then 1 else 0) /
                 ((if d3 = 20 ∧ d4 = 20 then 19 else 0) +
                  (if d3 = 20 ∧ d5 = 20 then 19 else 0) +
                  (if d4 = 20 ∧ d5 = 20 then 19 else 0) + 
                  (if d3 = 20 ∧ d4 = 20 ∧ d5 = 20 then 1 else 0) : ℚ) :=
sorry

end probability_all_three_dice_twenty_l203_203442


namespace yeast_cells_at_10_30_l203_203206

def yeast_population (initial_population : ℕ) (intervals : ℕ) (growth_rate : ℝ) (decay_rate : ℝ) : ℝ :=
  initial_population * (growth_rate * (1 - decay_rate)) ^ intervals

theorem yeast_cells_at_10_30 :
  yeast_population 50 6 3 0.10 = 52493 := by
  sorry

end yeast_cells_at_10_30_l203_203206


namespace greatest_x_for_quadratic_inequality_l203_203632

theorem greatest_x_for_quadratic_inequality (x : ℝ) (h : x^2 - 12 * x + 35 ≤ 0) : x ≤ 7 :=
sorry

end greatest_x_for_quadratic_inequality_l203_203632


namespace quadratic_function_range_l203_203634

-- Define the quadratic function and the domain
def quadratic_function (x : ℝ) : ℝ := -(x - 2)^2 + 1

-- State the proof problem
theorem quadratic_function_range : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 5 → -8 ≤ quadratic_function x ∧ quadratic_function x ≤ 1 := 
by 
  intro x
  intro h
  sorry

end quadratic_function_range_l203_203634


namespace overtime_rate_is_correct_l203_203263

/-
Define the parameters:
ordinary_rate: Rate per hour for ordinary time in dollars
total_hours: Total hours worked in a week
overtime_hours: Overtime hours worked in a week
total_earnings: Total earnings for the week in dollars
-/

def ordinary_rate : ℝ := 0.60
def total_hours : ℝ := 50
def overtime_hours : ℝ := 8
def total_earnings : ℝ := 32.40

noncomputable def overtime_rate : ℝ :=
(total_earnings - ordinary_rate * (total_hours - overtime_hours)) / overtime_hours

theorem overtime_rate_is_correct :
  overtime_rate = 0.90 :=
by
  sorry

end overtime_rate_is_correct_l203_203263


namespace sqrt_operation_l203_203079

def operation (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt_operation (sqrt5 : ℝ) (h : sqrt5 = Real.sqrt 5) : 
  operation sqrt5 sqrt5 = 20 := by
  sorry

end sqrt_operation_l203_203079


namespace cos_360_eq_one_l203_203893

theorem cos_360_eq_one : Real.cos (2 * Real.pi) = 1 :=
by sorry

end cos_360_eq_one_l203_203893


namespace union_is_faction_l203_203382

variable {D : Type} (is_faction : Set D → Prop)
variable (A B : Set D)

-- Define the complement
def complement (S : Set D) : Set D := {x | x ∉ S}

-- State the given condition
axiom faction_complement_union (A B : Set D) : 
  is_faction A → is_faction B → is_faction (complement (A ∪ B))

-- The theorem to prove
theorem union_is_faction (A B : Set D) :
  is_faction A → is_faction B → is_faction (A ∪ B) := 
by
  -- Proof goes here
  sorry

end union_is_faction_l203_203382


namespace arithmetic_progression_a_eq_1_l203_203352

theorem arithmetic_progression_a_eq_1 
  (a : ℝ) 
  (h1 : 6 + 2 * a - 1 = 10 + 5 * a - (6 + 2 * a)) : 
  a = 1 :=
by
  sorry

end arithmetic_progression_a_eq_1_l203_203352


namespace train_crossing_time_l203_203262

def train_length : ℝ := 140
def bridge_length : ℝ := 235.03
def speed_kmh : ℝ := 45

noncomputable def speed_mps : ℝ := speed_kmh * (1000 / 3600)
noncomputable def total_distance : ℝ := train_length + bridge_length

theorem train_crossing_time :
  (total_distance / speed_mps) = 30.0024 :=
by
  sorry

end train_crossing_time_l203_203262


namespace complex_omega_sum_l203_203548

open Complex

theorem complex_omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 + ω^48 + ω^51 + ω^54 + ω^57 + ω^60 + ω^63 = 1 := 
by
  sorry

end complex_omega_sum_l203_203548


namespace meaningful_expression_range_l203_203349

theorem meaningful_expression_range (x : ℝ) :
  (2 - x ≥ 0) ∧ (x - 2 ≠ 0) → x < 2 :=
by
  sorry

end meaningful_expression_range_l203_203349


namespace log_sum_l203_203727

open Real

theorem log_sum : log 2 + log 5 = 1 :=
sorry

end log_sum_l203_203727


namespace rate_of_interest_l203_203722

theorem rate_of_interest (P R : ℝ) :
  (2 * P * R) / 100 = 320 ∧
  P * ((1 + R / 100) ^ 2 - 1) = 340 →
  R = 12.5 :=
by
  intro h
  sorry

end rate_of_interest_l203_203722


namespace three_equal_mass_piles_l203_203759

theorem three_equal_mass_piles (n : ℕ) (h : n > 3) : 
  (∃ (A B C : Finset ℕ), 
    (A ∪ B ∪ C = Finset.range (n + 1)) ∧ 
    (A ∩ B = ∅) ∧ 
    (A ∩ C = ∅) ∧ 
    (B ∩ C = ∅) ∧ 
    (A.sum id = B.sum id) ∧ 
    (B.sum id = C.sum id)) 
  ↔ (n % 3 = 0 ∨ n % 3 = 2) :=
sorry

end three_equal_mass_piles_l203_203759


namespace cosine_of_3pi_over_2_l203_203049

theorem cosine_of_3pi_over_2 : Real.cos (3 * Real.pi / 2) = 0 := by
  sorry

end cosine_of_3pi_over_2_l203_203049


namespace total_salmon_l203_203753

def male_salmon : Nat := 712261
def female_salmon : Nat := 259378

theorem total_salmon :
  male_salmon + female_salmon = 971639 := by
  sorry

end total_salmon_l203_203753


namespace abs_sum_values_l203_203347

theorem abs_sum_values (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : 
  x + y = 7 ∨ x + y = 3 := 
by
  sorry

end abs_sum_values_l203_203347


namespace probability_of_x_plus_y_lt_4_l203_203477

open Classical -- To handle probability and random selection
open Set -- For geometric notions
open Filter -- For measure and integration

noncomputable def probability_condition (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) : Prop := x + y < 4

theorem probability_of_x_plus_y_lt_4 :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) →
                pr (λ (xy : ℝ × ℝ), probability_condition xy.1 xy.2 (and.intro (and.left xy.2) (and.right (and.left xy.2)))) = 7/9 :=
by sorry

end probability_of_x_plus_y_lt_4_l203_203477


namespace perm_prime_count_12345_l203_203040

theorem perm_prime_count_12345 : 
  (∀ x : List ℕ, (x ∈ (List.permutations [1, 2, 3, 4, 5])) → 
    (10^4 * x.head! + 10^3 * x.tail.head! + 10^2 * x.tail.tail.head! + 10 * x.tail.tail.tail.head! + x.tail.tail.tail.tail.head!) % 3 = 0)
  → 
  0 = 0 :=
by
  sorry

end perm_prime_count_12345_l203_203040


namespace lightsaber_ratio_l203_203794

theorem lightsaber_ratio (T L : ℕ) (hT : T = 1000) (hTotal : L + T = 3000) : L / T = 2 :=
by
  sorry

end lightsaber_ratio_l203_203794


namespace round_robin_tournament_l203_203934

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end round_robin_tournament_l203_203934


namespace football_team_practiced_hours_l203_203462

-- Define the daily practice hours and missed days as conditions
def daily_practice_hours : ℕ := 6
def missed_days : ℕ := 1

-- Define the total number of days in a week
def days_in_week : ℕ := 7

-- Define a function to calculate the total practiced hours in a week, 
-- given the daily practice hours, missed days, and total days in a week
def total_practiced_hours (daily_hours : ℕ) (missed : ℕ) (total_days : ℕ) : ℕ :=
  (total_days - missed) * daily_hours

-- Prove that the total practiced hours is 36
theorem football_team_practiced_hours :
  total_practiced_hours daily_practice_hours missed_days days_in_week = 36 := 
sorry

end football_team_practiced_hours_l203_203462


namespace original_number_eq_9999876_l203_203384

theorem original_number_eq_9999876 (x : ℕ) (h : x + 9876 = 10 * x + 9 + 876) : x = 999 :=
by {
  -- Simplify the equation and solve for x
  sorry
}

end original_number_eq_9999876_l203_203384


namespace complex_sum_series_l203_203189

theorem complex_sum_series (ω : ℂ) (h1 : ω ^ 7 = 1) (h2 : ω ≠ 1) :
  ω ^ 16 + ω ^ 18 + ω ^ 20 + ω ^ 22 + ω ^ 24 + ω ^ 26 + ω ^ 28 + ω ^ 30 + 
  ω ^ 32 + ω ^ 34 + ω ^ 36 + ω ^ 38 + ω ^ 40 + ω ^ 42 + ω ^ 44 + ω ^ 46 +
  ω ^ 48 + ω ^ 50 + ω ^ 52 + ω ^ 54 = -1 :=
sorry

end complex_sum_series_l203_203189


namespace total_seeds_eaten_proof_l203_203613

-- Define the information about the number of seeds eaten by each player
def first_player_seeds : ℕ := 78
def second_player_seeds : ℕ := 53
def third_player_seeds : ℕ := second_player_seeds + 30
def fourth_player_seeds : ℕ := 2 * third_player_seeds

-- Sum the seeds eaten by all the players
def total_seeds_eaten : ℕ := first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds

-- Prove that the total number of seeds eaten is 380
theorem total_seeds_eaten_proof : total_seeds_eaten = 380 :=
by
  -- To be filled in by actual proof steps
  sorry

end total_seeds_eaten_proof_l203_203613


namespace cylinder_height_to_diameter_ratio_l203_203607

theorem cylinder_height_to_diameter_ratio
  (r h : ℝ)
  (inscribed_sphere : h = 2 * r)
  (cylinder_volume : π * r^2 * h = 3 * (4/3) * π * r^3) :
  (h / (2 * r)) = 2 :=
by
  sorry

end cylinder_height_to_diameter_ratio_l203_203607


namespace antonov_packs_l203_203742

theorem antonov_packs (total_candies packs_given pieces_per_pack remaining_pieces packs_remaining : ℕ) 
    (h1 : total_candies = 60) 
    (h2 : packs_given = 1) 
    (h3 : pieces_per_pack = 20) 
    (h4 : remaining_pieces = total_candies - (packs_given * pieces_per_pack)) 
    (h5 : packs_remaining = remaining_pieces / pieces_per_pack) : 
    packs_remaining = 2 := 
by
  rw [h1, h3] at h4
  rw [Nat.mul_comm, Nat.sub_eq_iff_eq_add, Nat.sub_sub] at h4
  rw [Nat.mul_comm, Nat.div_eq_iff_eq_mul, Nat.mul_comm] at h5
  exact h5
sorry

end antonov_packs_l203_203742


namespace common_ratio_geometric_series_l203_203631

theorem common_ratio_geometric_series :
  let a := 2 / 3
  let b := 4 / 9
  let c := 8 / 27
  (b / a = 2 / 3) ∧ (c / b = 2 / 3) → 
  ∃ r : ℚ, r = 2 / 3 ∧ ∀ n : ℕ, (a * r^n) = (a * (2 / 3)^n) :=
by
  sorry

end common_ratio_geometric_series_l203_203631


namespace goblins_return_l203_203354

theorem goblins_return (n : ℕ) (f : Fin n → Fin n) (h1 : ∀ a, ∃! b, f a = b) (h2 : ∀ b, ∃! a, f a = b) : 
  ∃ k : ℕ, ∀ x : Fin n, (f^[k]) x = x := 
sorry

end goblins_return_l203_203354


namespace monthly_payment_amount_l203_203955

def original_price : ℝ := 480
def discount_rate : ℝ := 0.05
def first_installment : ℝ := 150
def num_monthly_installments : ℕ := 3

theorem monthly_payment_amount :
  let discounted_price := original_price * (1 - discount_rate),
      outstanding_balance := discounted_price - first_installment,
      monthly_payment := outstanding_balance / num_monthly_installments
  in monthly_payment = 102 := by
  sorry

end monthly_payment_amount_l203_203955


namespace matrix_pow_three_l203_203617

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_pow_three :
  A^3 = !![-4, 2; -2, 1] := by
  sorry

end matrix_pow_three_l203_203617


namespace math_proof_problem_l203_203852

theorem math_proof_problem (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3 * x₁ * y₁^2 = 2008)
  (h₂ : y₁^3 - 3 * x₁^2 * y₁ = 2007)
  (h₃ : x₂^3 - 3 * x₂ * y₂^2 = 2008)
  (h₄ : y₂^3 - 3 * x₂^2 * y₂ = 2007)
  (h₅ : x₃^3 - 3 * x₃ * y₃^2 = 2008)
  (h₆ : y₃^3 - 3 * x₃^2 * y₃ = 2007) :
  (1 - x₁ / y₁) * (1 - x₂ / y₂) * (1 - x₃ / y₃) = 4015 / 2008 :=
by sorry

end math_proof_problem_l203_203852


namespace first_sphere_weight_l203_203407

noncomputable def surface_area (r : ℝ) : ℝ :=
  4 * Real.pi * (r ^ 2)

noncomputable def weight (r1 r2 : ℝ) (W2 : ℝ) : ℝ :=
  let A1 := surface_area r1
  let A2 := surface_area r2
  (W2 * A1) / A2

theorem first_sphere_weight :
  let r1 := 0.15
  let r2 := 0.3
  let W2 := 32
  weight r1 r2 W2 = 8 := 
by
  sorry

end first_sphere_weight_l203_203407


namespace pascal_triangle_45th_number_l203_203015

theorem pascal_triangle_45th_number : nat.choose 46 44 = 1035 := 
by sorry

end pascal_triangle_45th_number_l203_203015


namespace f_evaluation_l203_203078

def f (a b c : ℚ) : ℚ := a^2 + 2 * b * c

theorem f_evaluation :
  f 1 23 76 + f 23 76 1 + f 76 1 23 = 10000 := by
  sorry

end f_evaluation_l203_203078


namespace number_of_elements_cong_set_l203_203336

/-- Define a set of integers less than 500 and congruent to 3 modulo 7 -/
def cong_set : Set ℕ := {n | n < 500 ∧ n % 7 = 3}

/-- The theorem stating the number of elements in cong_set is 72 -/
theorem number_of_elements_cong_set : Set.card cong_set = 72 :=
sorry

end number_of_elements_cong_set_l203_203336


namespace stanley_total_cost_l203_203745

theorem stanley_total_cost (n_tires : ℕ) (price_per_tire : ℝ) (h_n : n_tires = 4) (h_price : price_per_tire = 60) : n_tires * price_per_tire = 240 := by
  sorry

end stanley_total_cost_l203_203745


namespace three_digit_sum_reverse_eq_l203_203452

theorem three_digit_sum_reverse_eq :
  ∃ (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9),
    101 * (a + c) + 20 * b = 1777 ∧ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (9, 7, 8) :=
by
  sorry

end three_digit_sum_reverse_eq_l203_203452


namespace evaluate_custom_op_l203_203774

def custom_op (a b : ℝ) : ℝ := (a - b)^2

theorem evaluate_custom_op (x y : ℝ) : custom_op ((x + y)^2) ((y - x)^2) = 16 * x^2 * y^2 :=
by
  sorry

end evaluate_custom_op_l203_203774
