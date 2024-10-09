import Mathlib

namespace tan_theta_minus_pi_over_4_l1619_161956

theorem tan_theta_minus_pi_over_4 (θ : Real) (h1 : θ ∈ Set.Ioc (-(π / 2)) 0)
  (h2 : Real.sin (θ + π / 4) = 3 / 5) : Real.tan (θ - π / 4) = - (4 / 3) :=
by
  /- Proof goes here -/
  sorry

end tan_theta_minus_pi_over_4_l1619_161956


namespace inequality_holds_l1619_161981

theorem inequality_holds (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧ (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
sorry

end inequality_holds_l1619_161981


namespace unique_diff_subset_l1619_161960

noncomputable def exists_unique_diff_subset : Prop :=
  ∃ S : Set ℕ, 
    (∀ n : ℕ, n > 0 → ∃! (a b : ℕ), a ∈ S ∧ b ∈ S ∧ n = a - b)

theorem unique_diff_subset : exists_unique_diff_subset :=
  sorry

end unique_diff_subset_l1619_161960


namespace probability_of_exactly_9_correct_matches_is_zero_l1619_161909

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∃ (P : ℕ → ℕ → ℕ), 
    (∀ (total correct : ℕ), 
      total = 10 → 
      correct = 9 → 
      P total correct = 0) := 
by {
  sorry
}

end probability_of_exactly_9_correct_matches_is_zero_l1619_161909


namespace george_earnings_after_deductions_l1619_161962

noncomputable def george_total_earnings : ℕ := 35 + 12 + 20 + 21

noncomputable def tax_deduction (total_earnings : ℕ) : ℚ := total_earnings * 0.10

noncomputable def uniform_fee : ℚ := 15

noncomputable def final_earnings (total_earnings : ℕ) (tax_deduction : ℚ) (uniform_fee : ℚ) : ℚ :=
  total_earnings - tax_deduction - uniform_fee

theorem george_earnings_after_deductions : 
  final_earnings george_total_earnings (tax_deduction george_total_earnings) uniform_fee = 64.2 := 
  by
  sorry

end george_earnings_after_deductions_l1619_161962


namespace sequence_a_n_eq_5050_l1619_161928

theorem sequence_a_n_eq_5050 (a : ℕ → ℕ) (h1 : ∀ n > 1, (n - 1) * a n = (n + 1) * a (n - 1)) (h2 : a 1 = 1) : 
  a 100 = 5050 := 
by
  sorry

end sequence_a_n_eq_5050_l1619_161928


namespace find_max_sum_pair_l1619_161955

theorem find_max_sum_pair :
  ∃ a b : ℕ, 2 * a * b + 3 * b = b^2 + 6 * a + 6 ∧ (∀ a' b' : ℕ, 2 * a' * b' + 3 * b' = b'^2 + 6 * a' + 6 → a + b ≥ a' + b') ∧ a = 5 ∧ b = 9 :=
by {
  sorry
}

end find_max_sum_pair_l1619_161955


namespace correct_calculation_l1619_161983

theorem correct_calculation :
  (∀ (x : ℝ), (x^3 * 2 * x^4 = 2 * x^7) ∧
  (x^6 / x^3 = x^2) ∧
  ((x^3)^4 = x^7) ∧
  (x^2 + x = x^3)) → 
  (∀ (x : ℝ), x^3 * 2 * x^4 = 2 * x^7) :=
by
  intros h x
  have A := h x
  exact A.1

end correct_calculation_l1619_161983


namespace point_coordinates_l1619_161966

/-- Given the vector from point A to point B, if point A is the origin, then point B will have coordinates determined by the vector. -/
theorem point_coordinates (A B: ℝ × ℝ) (v: ℝ × ℝ) 
  (h: A = (0, 0)) (h_v: v = (-2, 4)) (h_ab: B = (A.1 + v.1, A.2 + v.2)): 
  B = (-2, 4) :=
by
  sorry

end point_coordinates_l1619_161966


namespace proof_second_number_is_30_l1619_161908

noncomputable def second_number_is_30 : Prop :=
  ∃ (a b c : ℕ), 
    a + b + c = 98 ∧ 
    (a / (gcd a b) = 2) ∧ (b / (gcd a b) = 3) ∧
    (b / (gcd b c) = 5) ∧ (c / (gcd b c) = 8) ∧
    b = 30

theorem proof_second_number_is_30 : second_number_is_30 :=
  sorry

end proof_second_number_is_30_l1619_161908


namespace sum_of_two_consecutive_negative_integers_l1619_161912

theorem sum_of_two_consecutive_negative_integers (n : ℤ) (h : n * (n + 1) = 2210) (hn : n < 0) : n + (n + 1) = -95 := 
sorry

end sum_of_two_consecutive_negative_integers_l1619_161912


namespace equal_area_centroid_S_l1619_161995

noncomputable def P : ℝ × ℝ := (-4, 3)
noncomputable def Q : ℝ × ℝ := (7, -5)
noncomputable def R : ℝ × ℝ := (0, 6)
noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

theorem equal_area_centroid_S (x y : ℝ) (h : (x, y) = centroid P Q R) :
  10 * x + y = 34 / 3 := by
  sorry

end equal_area_centroid_S_l1619_161995


namespace find_values_of_a_l1619_161915

noncomputable def has_one_real_solution (a : ℝ) : Prop :=
  ∃ x: ℝ, (x^3 - a*x^2 - 3*a*x + a^2 - 1 = 0) ∧ (∀ y: ℝ, (y^3 - a*y^2 - 3*a*y + a^2 - 1 = 0) → y = x)

theorem find_values_of_a : ∀ a: ℝ, has_one_real_solution a ↔ a < -(5 / 4) :=
by
  sorry

end find_values_of_a_l1619_161915


namespace range_of_m_l1619_161927

-- Definitions based on the conditions
def p (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 > 0

theorem range_of_m (m : ℝ) (hm : m > 0) :
  (∀ x : ℝ, p x → q x m) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m) ↔ 0 < m ∧ m ≤ 2 := sorry

end range_of_m_l1619_161927


namespace sum_terms_sequence_l1619_161990

noncomputable def geometric_sequence := ℕ → ℝ

variables (a : geometric_sequence)
variables (r : ℝ) (h_pos : ∀ n, a n > 0)

-- Geometric sequence condition
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r

-- Given condition
axiom h_condition : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100

-- The goal is to prove that a_4 + a_6 = 10
theorem sum_terms_sequence : a 4 + a 6 = 10 :=
by
  sorry

end sum_terms_sequence_l1619_161990


namespace amusement_park_ticket_length_l1619_161914

theorem amusement_park_ticket_length (Area Width Length : ℝ) (h₀ : Area = 1.77) (h₁ : Width = 3) (h₂ : Area = Width * Length) : Length = 0.59 :=
by
  -- Proof will go here
  sorry

end amusement_park_ticket_length_l1619_161914


namespace salary_calculation_l1619_161987

variable {A B : ℝ}

theorem salary_calculation (h1 : A + B = 6000) (h2 : 0.05 * A = 0.15 * B) : A = 4500 :=
by
  sorry

end salary_calculation_l1619_161987


namespace no_solutions_for_a_ne_4_solutions_for_a_eq_4_infinite_l1619_161926

-- Part (i)
theorem no_solutions_for_a_ne_4 (a : ℕ) (h : a ≠ 4) :
  ¬∃ (u v : ℕ), (u > 0 ∧ v > 0 ∧ u^2 + v^2 - a * u * v + 2 = 0) :=
by sorry

-- Part (ii)
theorem solutions_for_a_eq_4_infinite :
  ∃ (a_seq : ℕ → ℕ),
    (a_seq 0 = 1 ∧ a_seq 1 = 3 ∧
     ∀ n, a_seq (n + 2) = 4 * a_seq (n + 1) - a_seq n ∧
    ∀ n, (a_seq n) > 0 ∧ (a_seq (n + 1)) > 0 ∧ (a_seq n)^2 + (a_seq (n + 1))^2 - 4 * (a_seq n) * (a_seq (n + 1)) + 2 = 0) :=
by sorry

end no_solutions_for_a_ne_4_solutions_for_a_eq_4_infinite_l1619_161926


namespace part_i_l1619_161907

theorem part_i (n : ℕ) (a : Fin (n+1) → ℤ) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (a i - a j) % n = 0 := by
  sorry

end part_i_l1619_161907


namespace vessel_capacity_proof_l1619_161945

variable (V1_capacity : ℕ) (V2_capacity : ℕ) (total_mixture : ℕ) (final_vessel_capacity : ℕ)
variable (A1_percentage : ℕ) (A2_percentage : ℕ)

theorem vessel_capacity_proof
  (h1 : V1_capacity = 2)
  (h2 : A1_percentage = 35)
  (h3 : V2_capacity = 6)
  (h4 : A2_percentage = 50)
  (h5 : total_mixture = 8)
  (h6 : final_vessel_capacity = 10)
  : final_vessel_capacity = 10 := 
by
  sorry

end vessel_capacity_proof_l1619_161945


namespace steel_bar_lengths_l1619_161903

theorem steel_bar_lengths
  (x y z : ℝ)
  (h1 : 2 * x + y + 3 * z = 23)
  (h2 : x + 4 * y + 5 * z = 36) :
  x + 2 * y + 3 * z = 22 := 
sorry

end steel_bar_lengths_l1619_161903


namespace smallest_product_of_two_distinct_primes_greater_than_50_l1619_161920

theorem smallest_product_of_two_distinct_primes_greater_than_50 : 
  ∃ (p q : ℕ), p > 50 ∧ q > 50 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 3127 :=
by 
  sorry

end smallest_product_of_two_distinct_primes_greater_than_50_l1619_161920


namespace speed_of_stream_l1619_161911

variable (b s : ℝ)

theorem speed_of_stream (h1 : 110 = (b + s + 3) * 5)
                        (h2 : 85 = (b - s + 2) * 6) : s = 3.4 :=
by
  sorry

end speed_of_stream_l1619_161911


namespace r_has_money_l1619_161947

-- Define the variables and the conditions in Lean
variable (p q r : ℝ)
variable (h1 : p + q + r = 4000)
variable (h2 : r = (2/3) * (p + q))

-- Define the proof statement
theorem r_has_money : r = 1600 := 
  by
    sorry

end r_has_money_l1619_161947


namespace c_linear_combination_of_a_b_l1619_161957

-- Definitions of vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, 2)

-- Theorem stating the relationship between vectors a, b, and c
theorem c_linear_combination_of_a_b :
  c = (1 / 2 : ℝ) • a + (-3 / 2 : ℝ) • b :=
  sorry

end c_linear_combination_of_a_b_l1619_161957


namespace correct_calculation_l1619_161965

theorem correct_calculation (a : ℝ) : (-a)^10 / (-a)^3 = -a^7 :=
by sorry

end correct_calculation_l1619_161965


namespace max_xy_l1619_161977

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 * x + 2 * y = 12) : 
  xy ≤ 6 :=
sorry

end max_xy_l1619_161977


namespace quadratic_nonneg_range_l1619_161973

theorem quadratic_nonneg_range (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end quadratic_nonneg_range_l1619_161973


namespace pow_mod_26_l1619_161952

theorem pow_mod_26 (a b n : ℕ) (hn : n = 2023) (h₁ : a = 17) (h₂ : b = 26) :
  a ^ n % b = 7 := by
  sorry

end pow_mod_26_l1619_161952


namespace avg_income_pr_l1619_161919

theorem avg_income_pr (P Q R : ℝ) 
  (h_avgPQ : (P + Q) / 2 = 5050) 
  (h_avgQR : (Q + R) / 2 = 6250)
  (h_P : P = 4000) 
  : (P + R) / 2 = 5200 := 
by 
  sorry

end avg_income_pr_l1619_161919


namespace find_c_d_of_cubic_common_roots_l1619_161959

theorem find_c_d_of_cubic_common_roots 
  (c d : ℝ)
  (h1 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + c * r ^ 2 + 12 * r + 7 = 0) ∧ (s ^ 3 + c * s ^ 2 + 12 * s + 7 = 0))
  (h2 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + d * r ^ 2 + 15 * r + 9 = 0) ∧ (s ^ 3 + d * s ^ 2 + 15 * s + 9 = 0)) :
  c = 5 ∧ d = 4 :=
sorry

end find_c_d_of_cubic_common_roots_l1619_161959


namespace no_solution_fractions_eq_l1619_161902

open Real

theorem no_solution_fractions_eq (x : ℝ) :
  (x-2)/(2*x-1) + 1 = 3/(2-4*x) → False :=
by
  intro h
  have h1 : ¬ (2*x - 1 = 0) := by
    -- 2*x - 1 ≠ 0
    sorry
  have h2 : ¬ (2 - 4*x = 0) := by
    -- 2 - 4*x ≠ 0
    sorry
  -- Solve the equation and show no solutions exist without contradicting the conditions
  sorry

end no_solution_fractions_eq_l1619_161902


namespace swimming_pool_width_l1619_161949

theorem swimming_pool_width 
  (V_G : ℝ) (G_CF : ℝ) (height_inch : ℝ) (L : ℝ) (V_CF : ℝ) (height_ft : ℝ) (A : ℝ) (W : ℝ) :
  V_G = 3750 → G_CF = 7.48052 → height_inch = 6 → L = 40 →
  V_CF = V_G / G_CF → height_ft = height_inch / 12 →
  A = L * W → V_CF = A * height_ft →
  W = 25.067 :=
by
  intros hV hG hH hL hVC hHF hA hVF
  sorry

end swimming_pool_width_l1619_161949


namespace initial_trees_l1619_161943

theorem initial_trees (DeadTrees CutTrees LeftTrees : ℕ) (h1 : DeadTrees = 15) (h2 : CutTrees = 23) (h3 : LeftTrees = 48) :
  DeadTrees + CutTrees + LeftTrees = 86 :=
by
  sorry

end initial_trees_l1619_161943


namespace length_of_leg_of_isosceles_right_triangle_l1619_161954

def is_isosceles_right_triangle (a b h : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = h^2

def median_to_hypotenuse (m h : ℝ) : Prop :=
  m = h / 2

theorem length_of_leg_of_isosceles_right_triangle (m : ℝ) (h a : ℝ)
  (h1 : median_to_hypotenuse m h)
  (h2 : h = 2 * m)
  (h3 : is_isosceles_right_triangle a a h) :
  a = 15 * Real.sqrt 2 :=
by
  -- Skipping the proof
  sorry

end length_of_leg_of_isosceles_right_triangle_l1619_161954


namespace isosceles_triangle_perimeter_1_isosceles_triangle_perimeter_2_l1619_161939

-- Problem 1
theorem isosceles_triangle_perimeter_1 (a b : ℕ) (h1: a = 4 ∨ a = 6) (h2: b = 4 ∨ b = 6) (h3: a ≠ b): 
  (a + b + b = 14 ∨ a + b + b = 16) :=
sorry

-- Problem 2
theorem isosceles_triangle_perimeter_2 (a b : ℕ) (h1: a = 2 ∨ a = 6) (h2: b = 2 ∨ b = 6) (h3: a ≠ b ∨ (a = 2 ∧ 2 + 2 ≥ 6 ∧ 6 = b)):
  (a + b + b = 14) :=
sorry

end isosceles_triangle_perimeter_1_isosceles_triangle_perimeter_2_l1619_161939


namespace plan_b_more_cost_effective_l1619_161917

theorem plan_b_more_cost_effective (x : ℕ) : 
  (12 * x : ℤ) > (3000 + 8 * x : ℤ) → x ≥ 751 :=
sorry

end plan_b_more_cost_effective_l1619_161917


namespace percent_of_x_is_y_l1619_161951

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y / x = 0.25 :=
by
  -- proof omitted
  sorry

end percent_of_x_is_y_l1619_161951


namespace problem_translation_l1619_161970

variables {a : ℕ → ℤ} (S : ℕ → ℤ)

-- Definition of the arithmetic sequence and its sum function
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ (d : ℤ), ∀ (n m : ℕ), a (n + 1) = a n + d

-- Sum of the first n terms defined recursively
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  if n = 0 then 0 else a n + sum_first_n_terms a (n - 1)

-- Conditions
axiom h1 : is_arithmetic_sequence a
axiom h2 : S 5 > S 6

-- To be proved: Option D does not necessarily hold
theorem problem_translation : ¬(a 3 + a 6 + a 12 < 2 * a 7) := sorry

end problem_translation_l1619_161970


namespace candy_peanut_butter_is_192_l1619_161992

/-
   Define the conditions and the statement to be proved.
   The definitions follow directly from the problem's conditions.
-/
def candy_problem : Prop :=
  ∃ (peanut_butter_jar grape_jar banana_jar coconut_jar : ℕ),
    banana_jar = 43 ∧
    grape_jar = banana_jar + 5 ∧
    peanut_butter_jar = 4 * grape_jar ∧
    coconut_jar = 2 * banana_jar - 10 ∧
    peanut_butter_jar = 192
  -- The tuple (question, conditions, correct answer) is translated into this lemma

theorem candy_peanut_butter_is_192 : candy_problem :=
  by
    -- Skipping the actual proof as requested
    sorry

end candy_peanut_butter_is_192_l1619_161992


namespace max_blocks_fit_l1619_161976

-- Defining the dimensions of the box and blocks
def box_length : ℝ := 4
def box_width : ℝ := 3
def box_height : ℝ := 2

def block_length : ℝ := 3
def block_width : ℝ := 1
def block_height : ℝ := 1

-- Theorem stating the maximum number of blocks that fit
theorem max_blocks_fit : (24 / 3 = 8) ∧ (1 * 3 * 2 = 6) → 6 = 6 := 
by
  sorry

end max_blocks_fit_l1619_161976


namespace flutes_tried_out_l1619_161931

theorem flutes_tried_out (flutes clarinets trumpets pianists : ℕ) 
  (percent_flutes_in : ℕ → ℕ) (percent_clarinets_in : ℕ → ℕ) 
  (percent_trumpets_in : ℕ → ℕ) (percent_pianists_in : ℕ → ℕ) 
  (total_in_band : ℕ) :
  percent_flutes_in flutes = 80 / 100 * flutes ∧
  percent_clarinets_in clarinets = 30 / 2 ∧
  percent_trumpets_in trumpets = 60 / 3 ∧
  percent_pianists_in pianists = 20 / 10 ∧
  total_in_band = 53 →
  flutes = 20 :=
by
  sorry

end flutes_tried_out_l1619_161931


namespace trains_meet_in_2067_seconds_l1619_161984

def length_of_train1 : ℝ := 100  -- Length of Train 1 in meters
def length_of_train2 : ℝ := 200  -- Length of Train 2 in meters
def initial_distance : ℝ := 630  -- Initial distance between trains in meters
def speed_of_train1_kmh : ℝ := 90  -- Speed of Train 1 in km/h
def speed_of_train2_kmh : ℝ := 72  -- Speed of Train 2 in km/h

noncomputable def speed_of_train1_ms : ℝ := speed_of_train1_kmh * (1000 / 3600)
noncomputable def speed_of_train2_ms : ℝ := speed_of_train2_kmh * (1000 / 3600)
noncomputable def relative_speed : ℝ := speed_of_train1_ms + speed_of_train2_ms
noncomputable def total_distance : ℝ := initial_distance + length_of_train1 + length_of_train2
noncomputable def time_to_meet : ℝ := total_distance / relative_speed

theorem trains_meet_in_2067_seconds : time_to_meet = 20.67 := 
by
  sorry

end trains_meet_in_2067_seconds_l1619_161984


namespace find_x_l1619_161998

theorem find_x : ∃ x : ℝ, (0.40 * x - 30 = 50) ∧ x = 200 :=
by
  sorry

end find_x_l1619_161998


namespace integer_values_b_l1619_161963

theorem integer_values_b (b : ℤ) : 
  (∃ (x1 x2 : ℤ), x1 + x2 = -b ∧ x1 * x2 = 7 * b) ↔ b = 0 ∨ b = 36 ∨ b = -28 ∨ b = -64 :=
by
  sorry

end integer_values_b_l1619_161963


namespace gcd_90_405_l1619_161936

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l1619_161936


namespace number_of_marbles_pat_keeps_l1619_161910

theorem number_of_marbles_pat_keeps 
  (x : ℕ) 
  (h1 : x / 6 = 9) 
  : x / 3 = 18 :=
by
  sorry

end number_of_marbles_pat_keeps_l1619_161910


namespace container_volume_ratio_l1619_161929

theorem container_volume_ratio
  (A B C : ℝ)
  (h1 : (3 / 4) * A - (5 / 8) * B = (7 / 8) * C - (1 / 2) * C)
  (h2 : B =  (5 / 8) * B)
  (h3 : (5 / 8) * B =  (3 / 8) * C)
  (h4 : A =  (24 / 40) * C) : 
  A / C = 4 / 5 := sorry

end container_volume_ratio_l1619_161929


namespace parallel_lines_necessary_and_sufficient_l1619_161997

-- Define the lines l1 and l2
def line1 (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x - y + a = 0

-- State the theorem
theorem parallel_lines_necessary_and_sufficient (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ↔ a = 2 :=
by
  -- Proof omitted
  sorry

end parallel_lines_necessary_and_sufficient_l1619_161997


namespace inequality_solution_set_l1619_161944

theorem inequality_solution_set :
  {x : ℝ | (x - 5) * (x + 1) > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 5} :=
by
  sorry

end inequality_solution_set_l1619_161944


namespace marcia_savings_l1619_161996

def hat_price := 60
def regular_price (n : ℕ) := n * hat_price
def discount_price (discount_percentage: ℕ) (price: ℕ) := price - (price * discount_percentage) / 100
def promotional_price := hat_price + discount_price 25 hat_price + discount_price 35 hat_price

theorem marcia_savings : (regular_price 3 - promotional_price) * 100 / regular_price 3 = 20 :=
by
  -- The proof steps would follow here.
  sorry

end marcia_savings_l1619_161996


namespace fg_at_3_l1619_161946

-- Define the functions f and g according to the conditions
def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x + 2)^2

theorem fg_at_3 : f (g 3) = 103 :=
by
  sorry

end fg_at_3_l1619_161946


namespace Dongdong_test_score_l1619_161989

theorem Dongdong_test_score (a b c : ℕ) (h1 : a + b + c = 280) : a ≥ 94 ∨ b ≥ 94 ∨ c ≥ 94 :=
by
  sorry

end Dongdong_test_score_l1619_161989


namespace Frank_work_hours_l1619_161969

def hoursWorked (h_monday h_tuesday h_wednesday h_thursday h_friday h_saturday : Nat) : Nat :=
  h_monday + h_tuesday + h_wednesday + h_thursday + h_friday + h_saturday

theorem Frank_work_hours
  (h_monday : Nat := 8)
  (h_tuesday : Nat := 10)
  (h_wednesday : Nat := 7)
  (h_thursday : Nat := 9)
  (h_friday : Nat := 6)
  (h_saturday : Nat := 4) :
  hoursWorked h_monday h_tuesday h_wednesday h_thursday h_friday h_saturday = 44 :=
by
  unfold hoursWorked
  sorry

end Frank_work_hours_l1619_161969


namespace minimum_w_coincide_after_translation_l1619_161901

noncomputable def period_of_cosine (w : ℝ) : ℝ := (2 * Real.pi) / w

theorem minimum_w_coincide_after_translation
  (w : ℝ) (h_w_pos : 0 < w) :
  period_of_cosine w = (4 * Real.pi) / 3 → w = 3 / 2 :=
by
  sorry

end minimum_w_coincide_after_translation_l1619_161901


namespace band_members_minimum_n_l1619_161967

theorem band_members_minimum_n 
  (n : ℕ) 
  (h1 : n % 6 = 3) 
  (h2 : n % 8 = 5) 
  (h3 : n % 9 = 7) : 
  n ≥ 165 := 
sorry

end band_members_minimum_n_l1619_161967


namespace midpoint_fraction_l1619_161935

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (a + b) / 2 = 19/24 := by
  sorry

end midpoint_fraction_l1619_161935


namespace Andrey_Gleb_distance_l1619_161930

theorem Andrey_Gleb_distance (AB VG : ℕ) (AG : ℕ) (BV : ℕ) (cond1 : AB = 600) (cond2 : VG = 600) (cond3 : AG = 3 * BV) :
  AG = 900 ∨ AG = 1800 := 
sorry

end Andrey_Gleb_distance_l1619_161930


namespace determine_x1_l1619_161979

theorem determine_x1
  (x1 x2 x3 x4 : ℝ)
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1/3) :
  x1 = 4/5 :=
by
  sorry

end determine_x1_l1619_161979


namespace percent_absent_students_l1619_161971

def total_students : ℕ := 180
def num_boys : ℕ := 100
def num_girls : ℕ := 80
def fraction_boys_absent : ℚ := 1 / 5
def fraction_girls_absent : ℚ := 1 / 4

theorem percent_absent_students : 
  (fraction_boys_absent * num_boys + fraction_girls_absent * num_girls) / total_students = 22.22 / 100 := 
  sorry

end percent_absent_students_l1619_161971


namespace g_g_g_of_3_eq_neg_6561_l1619_161900

def g (x : ℤ) : ℤ := -x^2

theorem g_g_g_of_3_eq_neg_6561 : g (g (g 3)) = -6561 := by
  sorry

end g_g_g_of_3_eq_neg_6561_l1619_161900


namespace dollar_symmetric_l1619_161906

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_symmetric {x y : ℝ} : dollar (x + y) (y + x) = 0 :=
by
  sorry

end dollar_symmetric_l1619_161906


namespace proof_inequality_l1619_161916

noncomputable def problem (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1 → a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d

theorem proof_inequality (a b c d : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_inequality_l1619_161916


namespace number_of_trees_in_park_l1619_161964

def number_of_trees (length width area_per_tree : ℕ) : ℕ :=
  (length * width) / area_per_tree

theorem number_of_trees_in_park :
  number_of_trees 1000 2000 20 = 100000 :=
by
  sorry

end number_of_trees_in_park_l1619_161964


namespace train_length_is_900_l1619_161999

def train_length_crossing_pole (L V : ℕ) : Prop :=
  L = V * 18

def train_length_crossing_platform (L V : ℕ) : Prop :=
  L + 1050 = V * 39

theorem train_length_is_900 (L V : ℕ) (h1 : train_length_crossing_pole L V) (h2 : train_length_crossing_platform L V) : L = 900 := 
by
  sorry

end train_length_is_900_l1619_161999


namespace odd_square_diff_div_by_eight_l1619_161993

theorem odd_square_diff_div_by_eight (n p : ℤ) : 
  (2 * n + 1)^2 - (2 * p + 1)^2 % 8 = 0 := 
by 
-- Here we declare the start of the proof.
  sorry

end odd_square_diff_div_by_eight_l1619_161993


namespace car_travel_distance_l1619_161924

-- Define the conditions
def speed : ℝ := 23
def time : ℝ := 3

-- Define the formula for distance
def distance_traveled (s : ℝ) (t : ℝ) : ℝ := s * t

-- State the theorem to prove the distance the car traveled
theorem car_travel_distance : distance_traveled speed time = 69 :=
by
  -- The proof would normally go here, but we're skipping it as per the instructions
  sorry

end car_travel_distance_l1619_161924


namespace inverse_proportion_decreasing_l1619_161950

theorem inverse_proportion_decreasing (k : ℝ) (x : ℝ) (hx : x > 0) :
  (y = (k - 1) / x) → (k > 1) :=
by
  sorry

end inverse_proportion_decreasing_l1619_161950


namespace cube_root_sum_lt_sqrt_sum_l1619_161961

theorem cube_root_sum_lt_sqrt_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^3 + b^3)^(1/3) < (a^2 + b^2)^(1/2) := by
    sorry

end cube_root_sum_lt_sqrt_sum_l1619_161961


namespace mean_of_four_integers_l1619_161974

theorem mean_of_four_integers (x : ℝ) (h : (78 + 83 + 82 + x) / 4 = 80) : x = 77 ∧ x = 80 - 3 :=
by
  have h1 : 78 + 83 + 82 + x = 4 * 80 := by sorry
  have h2 : 78 + 83 + 82 = 243 := by sorry
  have h3 : 243 + x = 320 := by sorry
  have h4 : x = 320 - 243 := by sorry
  have h5 : x = 77 := by sorry
  have h6 : x = 80 - 3 := by sorry
  exact ⟨h5, h6⟩

end mean_of_four_integers_l1619_161974


namespace angle_A_is_60_degrees_triangle_area_l1619_161925

-- Define the basic setup for the triangle and its angles
variables (a b c : ℝ) -- internal angles of the triangle ABC
variables (B C : ℝ) -- sides opposite to angles b and c respectively

-- Given conditions
axiom equation_1 : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a
axiom perimeter_condition : a + b + c = 8
axiom circumradius_condition : ∃ R : ℝ, R = Real.sqrt 3

-- Question 1: Prove the measure of angle A is 60 degrees
theorem angle_A_is_60_degrees (h : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a) : 
  a = 60 :=
sorry

-- Question 2: Prove the area of triangle ABC
theorem triangle_area (h : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a)
(h_perimeter : a + b + c = 8) (h_circumradius : ∃ R : ℝ, R = Real.sqrt 3) :
  ∃ S : ℝ, S = 4 * Real.sqrt 3 / 3 :=
sorry

end angle_A_is_60_degrees_triangle_area_l1619_161925


namespace find_point_P_l1619_161913

structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def vector (P Q : Point) : Point :=
⟨Q.x - P.x, Q.y - P.y⟩

def magnitude_ratio (P A B : Point) (r : ℝ) : Prop :=
  let AP := vector A P
  let PB := vector P B
  (AP.x, AP.y) = (r * PB.x, r * PB.y)

theorem find_point_P (P : Point) : 
  magnitude_ratio P A B (4/3) → (P.x = 10 ∧ P.y = -21) :=
sorry

end find_point_P_l1619_161913


namespace sum_of_three_consecutive_integers_product_504_l1619_161904

theorem sum_of_three_consecutive_integers_product_504 : 
  ∃ n : ℤ, n * (n + 1) * (n + 2) = 504 ∧ n + (n + 1) + (n + 2) = 24 := 
by
  sorry

end sum_of_three_consecutive_integers_product_504_l1619_161904


namespace tangent_line_intersect_x_l1619_161958

noncomputable def tangent_intercept_x : ℚ := 9/2

theorem tangent_line_intersect_x (x : ℚ)
  (h₁ : x > 0)
  (h₂ : ∃ r₁ r₂ d : ℚ, r₁ = 3 ∧ r₂ = 5 ∧ d = 12 ∧ x = (r₂ * d) / (r₁ + r₂)) :
  x = tangent_intercept_x :=
by
  sorry

end tangent_line_intersect_x_l1619_161958


namespace three_digit_numbers_proof_l1619_161975

-- Definitions and conditions
def are_digits_distinct (A B C : ℕ) := (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C)

def is_arithmetic_mean (A B C : ℕ) := 2 * B = A + C

def geometric_mean_property (A B C : ℕ) := 
  (100 * A + 10 * B + C) * (100 * C + 10 * A + B) = (100 * B + 10 * C + A)^2

-- statement of the proof problem
theorem three_digit_numbers_proof :
  ∃ A B C : ℕ, (10 ≤ A) ∧ (A ≤ 99) ∧ (10 ≤ B) ∧ (B ≤ 99) ∧ (10 ≤ C) ∧ (C ≤ 99) ∧
  (A * 100 + B * 10 + C = 432 ∨ A * 100 + B * 10 + C = 864) ∧
  are_digits_distinct A B C ∧
  is_arithmetic_mean A B C ∧
  geometric_mean_property A B C :=
by {
  -- The Lean proof goes here
  sorry
}

end three_digit_numbers_proof_l1619_161975


namespace range_of_a_l1619_161968

theorem range_of_a (a b : ℝ) :
  (∀ x : ℝ, (a * x^2 + b * x + 1 < 2)) ∧ (a - b + 1 = 1) → (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l1619_161968


namespace max_odd_numbers_in_pyramid_l1619_161994

-- Define the properties of the pyramid
def is_sum_of_immediate_below (p : Nat → Nat → Nat) : Prop :=
  ∀ r c : Nat, r > 0 → p r c = p (r - 1) c + p (r - 1) (c + 1)

-- Define what it means for a number to be odd
def is_odd (n : Nat) : Prop := n % 2 = 1

-- Define the pyramid structure and number of rows
def pyramid (n : Nat) := { p : Nat → Nat → Nat // is_sum_of_immediate_below p ∧ n = 6 }

-- Theorem statement
theorem max_odd_numbers_in_pyramid (p : Nat → Nat → Nat) (h : is_sum_of_immediate_below p ∧ 6 = 6) : ∃ k : Nat, (∀ i j, is_odd (p i j) → k ≤ 14) := 
sorry

end max_odd_numbers_in_pyramid_l1619_161994


namespace parabola_distance_l1619_161986

theorem parabola_distance (a : ℝ) :
  (abs (1 + (1 / (4 * a))) = 2 → a = 1 / 4) ∨ 
  (abs (1 - (1 / (4 * a))) = 2 → a = -1 / 12) := by 
  sorry

end parabola_distance_l1619_161986


namespace triangle_number_arrangement_l1619_161980

noncomputable def numbers := [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

theorem triangle_number_arrangement : 
  ∃ (f : Fin 9 → Fin 9), 
    (numbers[f 0] + numbers[f 1] + numbers[f 2] = 
     numbers[f 3] + numbers[f 4] + numbers[f 5] ∧ 
     numbers[f 3] + numbers[f 4] + numbers[f 5] = 
     numbers[f 6] + numbers[f 7] + numbers[f 8]) :=
sorry

end triangle_number_arrangement_l1619_161980


namespace area_ratio_of_square_side_multiplied_by_10_l1619_161934

theorem area_ratio_of_square_side_multiplied_by_10 (s : ℝ) (A_original A_resultant : ℝ) 
  (h1 : A_original = s^2)
  (h2 : A_resultant = (10 * s)^2) :
  (A_original / A_resultant) = (1 / 100) :=
by
  sorry

end area_ratio_of_square_side_multiplied_by_10_l1619_161934


namespace four_integers_sum_product_odd_impossible_l1619_161982

theorem four_integers_sum_product_odd_impossible (a b c d : ℤ) :
  ¬ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ 
     (a + b + c + d) % 2 = 1) :=
by
  sorry

end four_integers_sum_product_odd_impossible_l1619_161982


namespace miae_closer_than_hyori_l1619_161991

def bowl_volume : ℝ := 1000
def miae_estimate : ℝ := 1100
def hyori_estimate : ℝ := 850

def miae_difference : ℝ := abs (miae_estimate - bowl_volume)
def hyori_difference : ℝ := abs (bowl_volume - hyori_estimate)

theorem miae_closer_than_hyori : miae_difference < hyori_difference :=
by
  sorry

end miae_closer_than_hyori_l1619_161991


namespace all_three_white_probability_l1619_161905

noncomputable def box_probability : ℚ :=
  let total_white := 4
  let total_black := 7
  let total_balls := total_white + total_black
  let draw_count := 3
  let total_combinations := (total_balls.choose draw_count : ℕ)
  let favorable_combinations := (total_white.choose draw_count : ℕ)
  (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem all_three_white_probability :
  box_probability = 4 / 165 :=
by
  sorry

end all_three_white_probability_l1619_161905


namespace can_weight_is_two_l1619_161948

theorem can_weight_is_two (c : ℕ) (h1 : 100 = 20 * c + 6 * ((100 - 20 * c) / 6)) (h2 : 160 = 10 * ((100 - 20 * c) / 6) + 3 * 20) : c = 2 :=
by
  sorry

end can_weight_is_two_l1619_161948


namespace chuck_play_area_l1619_161922

-- Define the conditions for the problem in Lean
def shed_length1 : ℝ := 3
def shed_length2 : ℝ := 4
def leash_length : ℝ := 4

-- State the theorem we want to prove
theorem chuck_play_area :
  let sector_area1 := (3 / 4) * Real.pi * (leash_length ^ 2)
  let sector_area2 := (1 / 4) * Real.pi * (1 ^ 2)
  sector_area1 + sector_area2 = (49 / 4) * Real.pi := 
by
  -- The proof is omitted for brevity
  sorry

end chuck_play_area_l1619_161922


namespace add_expression_l1619_161941

theorem add_expression {k : ℕ} :
  (2 * k + 2) + (2 * k + 3) = (2 * k + 2) + (2 * k + 3) := sorry

end add_expression_l1619_161941


namespace problem_statement_l1619_161978

noncomputable def f_B (x : ℝ) : ℝ := -x^2
noncomputable def f_D (x : ℝ) : ℝ := Real.cos x

theorem problem_statement :
  (∀ x : ℝ, f_B (-x) = f_B x) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f_B x1 > f_B x2) ∧
  (∀ x : ℝ, f_D (-x) = f_D x) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f_D x1 > f_D x2) :=
  sorry

end problem_statement_l1619_161978


namespace parameter_a_range_l1619_161923

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * a * x + 2 * a + 1

theorem parameter_a_range :
  (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → quadratic_function a x ≥ 1) ↔ (0 ≤ a) :=
by
  sorry

end parameter_a_range_l1619_161923


namespace parity_of_exponentiated_sum_l1619_161985

theorem parity_of_exponentiated_sum
  : (1 ^ 1994 + 9 ^ 1994 + 8 ^ 1994 + 6 ^ 1994) % 2 = 0 := 
by
  sorry

end parity_of_exponentiated_sum_l1619_161985


namespace avg_decrease_by_one_l1619_161953

noncomputable def average_decrease (obs : Fin 7 → ℕ) : ℕ :=
  let sum6 := 90
  let seventh := 8
  let new_sum := sum6 + seventh
  let new_avg := new_sum / 7
  let old_avg := 15
  old_avg - new_avg

theorem avg_decrease_by_one :
  (average_decrease (fun _ => 0)) = 1 :=
by
  sorry

end avg_decrease_by_one_l1619_161953


namespace initial_orange_balloons_l1619_161921

-- Definitions
variable (x : ℕ)
variable (h1 : x - 2 = 7)

-- Theorem to prove
theorem initial_orange_balloons (h1 : x - 2 = 7) : x = 9 :=
sorry

end initial_orange_balloons_l1619_161921


namespace sally_more_cards_than_dan_l1619_161933

theorem sally_more_cards_than_dan :
  let sally_initial := 27
  let sally_bought := 20
  let dan_cards := 41
  sally_initial + sally_bought - dan_cards = 6 :=
by
  sorry

end sally_more_cards_than_dan_l1619_161933


namespace minimum_handshakes_l1619_161972

noncomputable def min_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

theorem minimum_handshakes (n k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  min_handshakes n k = 45 :=
by
  -- We provide the conditions directly
  -- n = 30, k = 3
  rw [h1, h2]
  -- then show that min_handshakes 30 3 = 45
  show min_handshakes 30 3 = 45
  sorry 

end minimum_handshakes_l1619_161972


namespace satellite_modular_units_24_l1619_161988

-- Define basic parameters
variables (U N S : ℕ)
def fraction_upgraded : ℝ := 0.2

-- Define the conditions as Lean premises
axiom non_upgraded_per_unit_eq_sixth_total_upgraded : N = S / 6
axiom fraction_sensors_upgraded : (S : ℝ) = fraction_upgraded * (S + U * N)

-- The main statement to be proved
theorem satellite_modular_units_24 (h1 : N = S / 6) (h2 : (S : ℝ) = fraction_upgraded * (S + U * N)) : U = 24 :=
by
  -- The actual proof steps will be written here.
  sorry

end satellite_modular_units_24_l1619_161988


namespace inches_repaired_before_today_l1619_161937

-- Definitions and assumptions based on the conditions.
def total_inches_repaired : ℕ := 4938
def inches_repaired_today : ℕ := 805

-- Target statement that needs to be proven.
theorem inches_repaired_before_today : total_inches_repaired - inches_repaired_today = 4133 :=
by
  sorry

end inches_repaired_before_today_l1619_161937


namespace missing_fraction_is_73_div_60_l1619_161942

-- Definition of the given fractions
def fraction1 : ℚ := 1/3
def fraction2 : ℚ := 1/2
def fraction3 : ℚ := -5/6
def fraction4 : ℚ := 1/5
def fraction5 : ℚ := 1/4
def fraction6 : ℚ := -5/6

-- Total sum provided in the problem
def total_sum : ℚ := 50/60  -- 0.8333333333333334 in decimal form

-- The summation of given fractions
def sum_of_fractions : ℚ := fraction1 + fraction2 + fraction3 + fraction4 + fraction5 + fraction6

-- The statement to prove that the missing fraction is 73/60
theorem missing_fraction_is_73_div_60 : (total_sum - sum_of_fractions) = 73/60 := by
  sorry

end missing_fraction_is_73_div_60_l1619_161942


namespace bowling_ball_weight_l1619_161932

theorem bowling_ball_weight :
  (∃ (b c : ℝ), 8 * b = 4 * c ∧ 2 * c = 64) → ∃ b : ℝ, b = 16 :=
by
  sorry

end bowling_ball_weight_l1619_161932


namespace number_of_paths_A_to_D_l1619_161918

-- Definition of conditions
def ways_A_to_B : Nat := 2
def ways_B_to_C : Nat := 2
def ways_C_to_D : Nat := 2
def direct_A_to_D : Nat := 1

-- Theorem statement for the total number of paths from A to D
theorem number_of_paths_A_to_D : ways_A_to_B * ways_B_to_C * ways_C_to_D + direct_A_to_D = 9 := by
  sorry

end number_of_paths_A_to_D_l1619_161918


namespace probability_both_cards_are_diamonds_l1619_161940

-- Conditions definitions
def total_cards : ℕ := 52
def diamonds_in_deck : ℕ := 13
def two_draws : ℕ := 2

-- Calculation definitions
def total_possible_outcomes : ℕ := (total_cards * (total_cards - 1)) / two_draws
def favorable_outcomes : ℕ := (diamonds_in_deck * (diamonds_in_deck - 1)) / two_draws

-- Definition of the probability asked in the question
def probability_both_diamonds : ℚ := favorable_outcomes / total_possible_outcomes

theorem probability_both_cards_are_diamonds :
  probability_both_diamonds = 1 / 17 := 
sorry

end probability_both_cards_are_diamonds_l1619_161940


namespace constant_sum_of_distances_l1619_161938

open Real

theorem constant_sum_of_distances (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
    (ellipse_condition : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → ∀ A B : ℝ × ℝ, A.2 > 0 ∧ B.2 > 0)
    (foci : (ℝ × ℝ) × (ℝ × ℝ) := ((-c, 0), (c, 0)))
    (points_AB : ∃ (A B : ℝ × ℝ), A.2 > 0 ∧ B.2 > 0 ∧ (A.1 - c)^2 / a^2 + A.2^2 / b^2 = 1 ∧ (B.1 - -c)^2 / a^2 + B.2^2 / b^2 = 1)
    (AF1_parallel_BF2 : ∀ (A B : ℝ × ℝ), (A.1 - -c) * (B.2 - 0) - (A.2 - 0) * (B.1 - c) = 0)
    (intersection_P: ∀ (A B : ℝ × ℝ), ∃ P : ℝ × ℝ, ((A.1 - c) * (B.2 - 0) = (A.2 - 0) * (P.1 - c)) ∧ ((B.1 - -c) * (A.2 - 0) = (B.2 - 0) * (P.1 - -c))) :
    ∃ k : ℝ, ∀ (P : ℝ × ℝ), dist P (foci.fst) + dist P (foci.snd) = k := 
sorry

end constant_sum_of_distances_l1619_161938
