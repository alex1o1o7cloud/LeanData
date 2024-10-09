import Mathlib

namespace simplification_of_expression_l1109_110931

theorem simplification_of_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  ( (x - 2) / (x^2 - 2 * x + 1) / (x / (x - 1)) + 1 / (x^2 - x) ) = 1 / x := 
by 
  sorry

end simplification_of_expression_l1109_110931


namespace typing_speed_ratio_l1109_110917

theorem typing_speed_ratio (T t : ℝ) (h1 : T + t = 12) (h2 : T + 1.25 * t = 14) : t / T = 2 :=
by
  sorry

end typing_speed_ratio_l1109_110917


namespace not_perfect_square_l1109_110942

theorem not_perfect_square (x y : ℤ) : ¬ ∃ k : ℤ, k^2 = (x^2 + x + 1)^2 + (y^2 + y + 1)^2 :=
by
  sorry

end not_perfect_square_l1109_110942


namespace parabola_and_hyperbola_equation_l1109_110978

theorem parabola_and_hyperbola_equation (a b c : ℝ)
    (ha : a > 0)
    (hb : b > 0)
    (hp_eq : c = 2)
    (intersect : (3 / 2, Real.sqrt 6) ∈ {p : ℝ × ℝ | p.2^2 = 4 * c * p.1}
                ∧ (3 / 2, Real.sqrt 6) ∈ {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) - (p.2 ^ 2 / b ^ 2) = 1}) :
    (∀ x y : ℝ, y^2 = 4*x ↔ c = 1)
    ∧ (∃ a', a' = 1 / 2 ∧ ∀ x y : ℝ, 4 * x^2 - (4 * y^2) / 3 = 1 ↔ a = a') := 
by 
  -- Proof will be here
  sorry

end parabola_and_hyperbola_equation_l1109_110978


namespace short_pencil_cost_l1109_110910

theorem short_pencil_cost (x : ℝ)
  (h1 : 200 * 0.8 + 40 * 0.5 + 35 * x = 194) : x = 0.4 :=
by {
  sorry
}

end short_pencil_cost_l1109_110910


namespace race_head_start_l1109_110912

theorem race_head_start (Va Vb L H : ℚ) (h : Va = 30 / 17 * Vb) :
  H = 13 / 30 * L :=
by
  sorry

end race_head_start_l1109_110912


namespace difference_of_squares_not_2018_l1109_110996

theorem difference_of_squares_not_2018 (a b : ℕ) : a^2 - b^2 ≠ 2018 :=
by
  sorry

end difference_of_squares_not_2018_l1109_110996


namespace total_earnings_l1109_110966

theorem total_earnings : 
  let wage : ℕ := 10
  let hours_monday : ℕ := 7
  let tips_monday : ℕ := 18
  let hours_tuesday : ℕ := 5
  let tips_tuesday : ℕ := 12
  let hours_wednesday : ℕ := 7
  let tips_wednesday : ℕ := 20
  let total_hours : ℕ := hours_monday + hours_tuesday + hours_wednesday
  let earnings_from_wage : ℕ := total_hours * wage
  let total_tips : ℕ := tips_monday + tips_tuesday + tips_wednesday
  let total_earnings : ℕ := earnings_from_wage + total_tips
  total_earnings = 240 :=
by
  sorry

end total_earnings_l1109_110966


namespace part1_is_geometric_part2_is_arithmetic_general_formula_for_a_sum_of_first_n_terms_l1109_110960

open Nat

variable {α : Type*}
variables (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom a1 : a 1 = 1
axiom S_def : ∀ (n : ℕ), S (n + 1) = 4 * a n + 2 

def b (n : ℕ) : ℕ := a (n + 1) - 2 * a n

def c (n : ℕ) : ℚ := a n / 2^n

theorem part1_is_geometric :
  ∃ r, ∀ n, b n = r * b (n - 1) := sorry

theorem part2_is_arithmetic :
  ∃ d, ∀ n, c n - c (n - 1) = d := sorry

theorem general_formula_for_a :
  ∀ n, a n = (1 / 4) * (3 * n - 1) * 2 ^ n := sorry

theorem sum_of_first_n_terms :
  ∀ n, S n = (1 / 4) * (8 + (3 * n - 4) * 2 ^ (n + 1)) := sorry

end part1_is_geometric_part2_is_arithmetic_general_formula_for_a_sum_of_first_n_terms_l1109_110960


namespace barycentric_identity_l1109_110965

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def barycentric (α β γ : ℝ) (a b c : V) : V := 
  α • a + β • b + γ • c

theorem barycentric_identity 
  (A B C X : V) 
  (α β γ : ℝ)
  (h : α + β + γ = 1)
  (hXA : X = barycentric α β γ A B C) :
  X - A = β • (B - A) + γ • (C - A) :=
by
  sorry

end barycentric_identity_l1109_110965


namespace sum_5n_is_630_l1109_110959

variable (n : ℕ)

def sum_first_k (k : ℕ) : ℕ :=
  k * (k + 1) / 2

theorem sum_5n_is_630 (h : sum_first_k (3 * n) = sum_first_k n + 210) : sum_first_k (5 * n) = 630 := sorry

end sum_5n_is_630_l1109_110959


namespace food_price_before_tax_and_tip_l1109_110980

theorem food_price_before_tax_and_tip (total_paid : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (P : ℝ) (h1 : total_paid = 198) (h2 : tax_rate = 0.10) (h3 : tip_rate = 0.20) : 
  P = 150 :=
by
  -- Given that total_paid = 198, tax_rate = 0.10, tip_rate = 0.20,
  -- we should show that the actual price of the food before tax
  -- and tip is $150.
  sorry

end food_price_before_tax_and_tip_l1109_110980


namespace original_denominator_is_two_l1109_110981

theorem original_denominator_is_two (d : ℕ) : 
  (∃ d : ℕ, 2 * (d + 4) = 6) → d = 2 :=
by sorry

end original_denominator_is_two_l1109_110981


namespace number_of_boys_in_school_l1109_110937

theorem number_of_boys_in_school (B : ℕ) (girls : ℕ) (difference : ℕ) 
    (h1 : girls = 697) (h2 : girls = B + 228) : B = 469 := 
by
  sorry

end number_of_boys_in_school_l1109_110937


namespace find_m_value_l1109_110907

theorem find_m_value (m : ℝ) : (∃ A B : ℝ × ℝ, A = (-2, m) ∧ B = (m, 4) ∧ (∃ k : ℝ, k = (4 - m) / (m + 2) ∧ k = -2) ∧ (∃ l : ℝ, l = -2 ∧ 2 * l + l - 1 = 0)) → m = -8 :=
by
  sorry

end find_m_value_l1109_110907


namespace willy_crayons_eq_l1109_110983

def lucy_crayons : ℕ := 3971
def more_crayons : ℕ := 1121

theorem willy_crayons_eq : 
  ∀ willy_crayons : ℕ, willy_crayons = lucy_crayons + more_crayons → willy_crayons = 5092 :=
by
  sorry

end willy_crayons_eq_l1109_110983


namespace sequence_non_positive_l1109_110992

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h₀ : a 0 = 0) 
  (hₙ : a n = 0)
  (h : ∀ k, 1 ≤ k ∧ k < n → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) : 
  ∀ k, k ≤ n → a k ≤ 0 :=
by
  sorry

end sequence_non_positive_l1109_110992


namespace magician_assistant_strategy_l1109_110940

-- Define the possible states of a coin
inductive CoinState | heads | tails

-- Define the circle of coins
def coinCircle := Fin 11 → CoinState

-- The strategy ensures there are adjacent coins with the same state
theorem magician_assistant_strategy (c : coinCircle) :
  ∃ i : Fin 11, c i = c ((i + 1) % 11) := by sorry

end magician_assistant_strategy_l1109_110940


namespace triangle_areas_l1109_110995

theorem triangle_areas (S₁ S₂ : ℝ) :
  ∃ (ABC : ℝ), ABC = Real.sqrt (S₁ * S₂) :=
sorry

end triangle_areas_l1109_110995


namespace g_is_even_l1109_110976

noncomputable def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end g_is_even_l1109_110976


namespace Mobius_speed_without_load_l1109_110949

theorem Mobius_speed_without_load
  (v : ℝ)
  (distance : ℝ := 143)
  (load_speed : ℝ := 11)
  (rest_time : ℝ := 2)
  (total_time : ℝ := 26) :
  (total_time - rest_time = (distance / load_speed + distance / v)) → v = 13 :=
by
  intros h
  exact sorry

end Mobius_speed_without_load_l1109_110949


namespace relationship_among_x_y_z_w_l1109_110926

theorem relationship_among_x_y_z_w (x y z w : ℝ) (h : (x + y) / (y + z) = (z + w) / (w + x)) :
  x = z ∨ x + y + w + z = 0 :=
sorry

end relationship_among_x_y_z_w_l1109_110926


namespace ratio_mets_redsox_l1109_110924

theorem ratio_mets_redsox 
    (Y M R : ℕ) 
    (h1 : Y = 3 * (M / 2))
    (h2 : M = 88)
    (h3 : Y + M + R = 330) : 
    M / R = 4 / 5 := 
by 
    sorry

end ratio_mets_redsox_l1109_110924


namespace ticket_cost_l1109_110984

theorem ticket_cost (a : ℝ) (h1 : (6 * a + 5 * (2 / 3 * a) = 47.25)) :
  10 * a + 8 * (2 / 3 * a) = 77.625 :=
by
  sorry

end ticket_cost_l1109_110984


namespace find_maximum_marks_l1109_110903

theorem find_maximum_marks (M : ℝ) 
  (h1 : 0.60 * M = 270)
  (h2 : ∀ x : ℝ, 220 + 50 = x → x = 270) : 
  M = 450 :=
by
  sorry

end find_maximum_marks_l1109_110903


namespace tangent_line_parabola_d_l1109_110909

theorem tangent_line_parabola_d (d : ℝ) :
  (∀ x y : ℝ, (y = 3 * x + d) → (y^2 = 12 * x) → ∃! x, 9 * x^2 + (6 * d - 12) * x + d^2 = 0) → d = 1 :=
by
  sorry

end tangent_line_parabola_d_l1109_110909


namespace quadratic_function_inequality_l1109_110967

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem quadratic_function_inequality (a b c : ℝ) (h_a : a > 0) (h_symm : ∀ x : ℝ, quadratic_function a b c x = quadratic_function a b c (2 - x)) :
  ∀ x : ℝ, quadratic_function a b c (2 ^ x) < quadratic_function a b c (3 ^ x) :=
by
  sorry

end quadratic_function_inequality_l1109_110967


namespace cucumber_weight_l1109_110939

theorem cucumber_weight (W : ℝ)
  (h1 : W * 0.99 + W * 0.01 = W)
  (h2 : (W * 0.01) / 20 = 1 / 95) :
  W = 100 :=
by
  sorry

end cucumber_weight_l1109_110939


namespace sqrt_of_sixteen_l1109_110974

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_sixteen_l1109_110974


namespace circle_in_quad_radius_l1109_110923

theorem circle_in_quad_radius (AB BC CD DA : ℝ) (r : ℝ) (h₁ : AB = 15) (h₂ : BC = 10) (h₃ : CD = 8) (h₄ : DA = 13) :
  r = 2 * Real.sqrt 10 := 
by {
  sorry
  }

end circle_in_quad_radius_l1109_110923


namespace total_enemies_l1109_110943

theorem total_enemies (points_per_enemy defeated_enemies undefeated_enemies total_points total_enemies : ℕ)
  (h1 : points_per_enemy = 5) 
  (h2 : undefeated_enemies = 6) 
  (h3 : total_points = 10) :
  total_enemies = 8 := by
  sorry

end total_enemies_l1109_110943


namespace paper_clips_distribution_l1109_110977

theorem paper_clips_distribution (P c b : ℕ) (hP : P = 81) (hc : c = 9) (hb : b = P / c) : b = 9 :=
by
  rw [hP, hc] at hb
  simp at hb
  exact hb

end paper_clips_distribution_l1109_110977


namespace area_of_square_l1109_110918

theorem area_of_square (A_circle : ℝ) (hA_circle : A_circle = 39424) (cm_to_inch : ℝ) (hcm_to_inch : cm_to_inch = 2.54) :
  ∃ (A_square : ℝ), A_square = 121.44 := 
by
  sorry

end area_of_square_l1109_110918


namespace downstream_speed_l1109_110919

-- Define the given conditions
def V_m : ℝ := 40 -- speed of the man in still water in kmph
def V_up : ℝ := 32 -- speed of the man upstream in kmph

-- Question to be proved as a statement
theorem downstream_speed : 
  ∃ (V_c V_down : ℝ), V_c = V_m - V_up ∧ V_down = V_m + V_c ∧ V_down = 48 :=
by
  -- Provide statement without proof as specified
  sorry

end downstream_speed_l1109_110919


namespace value_range_of_m_for_equation_l1109_110920

theorem value_range_of_m_for_equation 
    (x : ℝ) 
    (cos_x : ℝ) 
    (h1: cos_x = Real.cos x) :
    ∃ (m : ℝ), (0 ≤ m ∧ m ≤ 8) ∧ (4 * cos_x + Real.sin x ^ 2 + m - 4 = 0) := sorry

end value_range_of_m_for_equation_l1109_110920


namespace number_of_white_balls_l1109_110999

theorem number_of_white_balls (r w : ℕ) (h_r : r = 8) (h_prob : (r : ℚ) / (r + w) = 2 / 5) : w = 12 :=
by sorry

end number_of_white_balls_l1109_110999


namespace find_second_number_l1109_110913

theorem find_second_number (x : ℕ) (h1 : ∀ d : ℕ, d ∣ 60 → d ∣ x → d ∣ 18) 
                           (h2 : 60 % 18 = 6) (h3 : x % 18 = 10) 
                           (h4 : x > 60) : 
  x = 64 := 
by
  sorry

end find_second_number_l1109_110913


namespace cos_pi_minus_2alpha_l1109_110948

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (Real.pi - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_2alpha_l1109_110948


namespace graph_of_equation_l1109_110964

theorem graph_of_equation (x y : ℝ) :
  x^3 * (x + y + 2) = y^3 * (x + y + 2) →
  (x + y + 2 ≠ 0 ∧ (x = y ∨ x^2 + x * y + y^2 = 0)) ∨
  (x + y + 2 = 0 ∧ y = -x - 2) →
  (y = x ∨ y = -x - 2) := 
sorry

end graph_of_equation_l1109_110964


namespace sum_of_fractions_l1109_110987

theorem sum_of_fractions (a b c d : ℚ) (ha : a = 2 / 5) (hb : b = 3 / 8) :
  (a + b = 31 / 40) :=
by
  sorry

end sum_of_fractions_l1109_110987


namespace sector_area_correct_l1109_110928

noncomputable def sector_area (r θ : ℝ) : ℝ := 0.5 * θ * r^2

theorem sector_area_correct (r θ : ℝ) (hr : r = 2) (hθ : θ = 2 * Real.pi / 3) :
  sector_area r θ = 4 * Real.pi / 3 :=
by
  subst hr
  subst hθ
  sorry

end sector_area_correct_l1109_110928


namespace calculate_length_QR_l1109_110921

noncomputable def length_QR (A : ℝ) (h : ℝ) (PQ : ℝ) (RS : ℝ) : ℝ :=
  21 - 0.5 * (Real.sqrt (PQ ^ 2 - h ^ 2) + Real.sqrt (RS ^ 2 - h ^ 2))

theorem calculate_length_QR :
  length_QR 210 10 12 21 = 21 - 0.5 * (Real.sqrt 44 + Real.sqrt 341) :=
by
  sorry

end calculate_length_QR_l1109_110921


namespace percent_absent_l1109_110997

-- Given conditions
def total_students : ℕ := 120
def boys : ℕ := 72
def girls : ℕ := 48
def absent_boys_fraction : ℚ := 1 / 8
def absent_girls_fraction : ℚ := 1 / 4

-- Theorem to prove
theorem percent_absent : 100 * ((absent_boys_fraction * boys + absent_girls_fraction * girls) / total_students) = 17.5 := 
sorry

end percent_absent_l1109_110997


namespace alt_fib_factorial_seq_last_two_digits_eq_85_l1109_110982

noncomputable def alt_fib_factorial_seq_last_two_digits : ℕ :=
  let f0 := 1   -- 0!
  let f1 := 1   -- 1!
  let f2 := 2   -- 2!
  let f3 := 6   -- 3!
  let f5 := 120 -- 5! (last two digits 20)
  (f0 - f1 + f1 - f2 + f3 - (f5 % 100)) % 100

theorem alt_fib_factorial_seq_last_two_digits_eq_85 :
  alt_fib_factorial_seq_last_two_digits = 85 :=
by 
  sorry

end alt_fib_factorial_seq_last_two_digits_eq_85_l1109_110982


namespace lemonade_calories_is_correct_l1109_110911

def lemon_juice_content := 150
def sugar_content := 150
def water_content := 450

def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 400
def water_calories_per_100g := 0

def total_weight := lemon_juice_content + sugar_content + water_content
def caloric_density :=
  (lemon_juice_content * lemon_juice_calories_per_100g / 100) +
  (sugar_content * sugar_calories_per_100g / 100) +
  (water_content * water_calories_per_100g / 100)
def calories_per_gram := caloric_density / total_weight

def calories_in_300_grams := 300 * calories_per_gram

theorem lemonade_calories_is_correct : calories_in_300_grams = 258 := by
  sorry

end lemonade_calories_is_correct_l1109_110911


namespace necessary_and_sufficient_condition_l1109_110957

-- Define the arithmetic sequence
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  a_1 + (n - 1) * d

-- Define the sum of the first k terms of an arithmetic sequence
def sum_arithmetic_seq (a_1 : ℤ) (d : ℤ) (k : ℤ) : ℤ :=
  (k * (2 * a_1 + (k - 1) * d)) / 2

-- Prove that d > 0 is a necessary and sufficient condition for S_3n - S_2n > S_2n - S_n
/-- Necessary and sufficient condition for the inequality S_{3n} - S_{2n} > S_{2n} - S_n -/
theorem necessary_and_sufficient_condition {a_1 d n : ℤ} :
  d > 0 ↔ sum_arithmetic_seq a_1 d (3 * n) - sum_arithmetic_seq a_1 d (2 * n) > 
             sum_arithmetic_seq a_1 d (2 * n) - sum_arithmetic_seq a_1 d n :=
by sorry

end necessary_and_sufficient_condition_l1109_110957


namespace find_complement_l1109_110900

-- Define predicate for a specific universal set U and set A
def universal_set (a : ℤ) (x : ℤ) : Prop :=
  x = a^2 - 2 ∨ x = 2 ∨ x = 1

def set_A (a : ℤ) (x : ℤ) : Prop :=
  x = a ∨ x = 1

-- Define complement of A with respect to U
def complement_U_A (a : ℤ) (x : ℤ) : Prop :=
  universal_set a x ∧ ¬ set_A a x

-- Main theorem statement
theorem find_complement (a : ℤ) (h : a ≠ 2) : { x | complement_U_A a x } = {2} :=
by
  sorry

end find_complement_l1109_110900


namespace percentage_increase_l1109_110908

theorem percentage_increase (new_wage original_wage : ℝ) (h₁ : new_wage = 42) (h₂ : original_wage = 28) :
  ((new_wage - original_wage) / original_wage) * 100 = 50 := by
  sorry

end percentage_increase_l1109_110908


namespace arithmetic_sequence_property_l1109_110994

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_property
  (h1 : a 3 + 3 * a 8 + a 13 = 120)
  (h2 : a 3 + a 13 = 2 * a 8) :
  a 3 + a 13 - a 8 = 24 := by
  sorry

end arithmetic_sequence_property_l1109_110994


namespace smallest_number_divisible_by_11_and_remainder_1_l1109_110989

theorem smallest_number_divisible_by_11_and_remainder_1 {n : ℕ} :
  (n % 2 = 1) ∧ 
  (n % 3 = 1) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 11 = 0) -> n = 121 :=
sorry

end smallest_number_divisible_by_11_and_remainder_1_l1109_110989


namespace cos_identity_l1109_110972

theorem cos_identity (x : ℝ) : 
  4 * Real.cos x * Real.cos (x + π / 3) * Real.cos (x - π / 3) = Real.cos (3 * x) :=
by
  sorry

end cos_identity_l1109_110972


namespace second_round_score_l1109_110986

/-- 
  Given the scores in three rounds of darts, where the second round score is twice the
  first round score, and the third round score is 1.5 times the second round score,
  prove that the score in the second round is 48, given that the maximum score in the 
  third round is 72.
-/
theorem second_round_score (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 1.5 * y) (h3 : z = 72) : y = 48 :=
sorry

end second_round_score_l1109_110986


namespace equivalent_set_complement_intersection_l1109_110955

def setM : Set ℝ := {x | -3 < x ∧ x < 1}
def setN : Set ℝ := {x | x ≤ 3}
def givenSet : Set ℝ := {x | x ≤ -3 ∨ x ≥ 1}

theorem equivalent_set_complement_intersection :
  givenSet = (setM ∩ setN)ᶜ :=
sorry

end equivalent_set_complement_intersection_l1109_110955


namespace problem_statement_l1109_110998

open Real

theorem problem_statement (t : ℝ) :
  cos (2 * t) ≠ 0 ∧ sin (2 * t) ≠ 0 →
  cos⁻¹ (2 * t) + sin⁻¹ (2 * t) + cos⁻¹ (2 * t) * sin⁻¹ (2 * t) = 5 →
  (∃ k : ℤ, t = arctan (1/2) + π * k) ∨ (∃ n : ℤ, t = arctan (1/3) + π * n) :=
by
  sorry

end problem_statement_l1109_110998


namespace rectangle_area_stage_8_l1109_110969

def square_side_length : ℕ := 4
def stage_count : ℕ := 8

-- The function to compute the area of one square
def square_area (side_length: ℕ) : ℕ :=
  side_length * side_length

-- The function to compute the total area at a given stage
def total_area_at_stage (side_length: ℕ) (stages: ℕ) : ℕ :=
  stages * (square_area side_length)

theorem rectangle_area_stage_8 :
  total_area_at_stage square_side_length stage_count = 128 :=
  by
    sorry

end rectangle_area_stage_8_l1109_110969


namespace smallest_portion_quantity_l1109_110906

-- Define the conditions for the problem
def conditions (a1 a2 a3 a4 a5 d : ℚ) : Prop :=
  a2 = a1 + d ∧
  a3 = a1 + 2 * d ∧
  a4 = a1 + 3 * d ∧
  a5 = a1 + 4 * d ∧
  5 * a1 + 10 * d = 100 ∧
  (a3 + a4 + a5) = (1/7) * (a1 + a2)

-- Lean theorem statement
theorem smallest_portion_quantity : 
  ∃ (a1 a2 a3 a4 a5 d : ℚ), conditions a1 a2 a3 a4 a5 d ∧ a1 = 5 / 3 :=
by
  sorry

end smallest_portion_quantity_l1109_110906


namespace sue_driving_days_l1109_110950

-- Define the conditions as constants or variables
def total_cost : ℕ := 2100
def sue_payment : ℕ := 900
def sister_days : ℕ := 4
def total_days_in_week : ℕ := 7

-- Prove that the number of days Sue drives the car (x) equals 3
theorem sue_driving_days : ∃ x : ℕ, x = 3 ∧ sue_payment * sister_days = x * (total_cost - sue_payment) := 
by
  sorry

end sue_driving_days_l1109_110950


namespace compute_expression_l1109_110970

theorem compute_expression (x : ℝ) (h : x = 7) : (x^6 - 36*x^3 + 324) / (x^3 - 18) = 325 := 
by
  sorry

end compute_expression_l1109_110970


namespace problem_C_l1109_110934

theorem problem_C (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a * b :=
by sorry

end problem_C_l1109_110934


namespace apples_needed_for_two_weeks_l1109_110914

theorem apples_needed_for_two_weeks :
  ∀ (apples_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ),
  apples_per_day = 1 → days_per_week = 7 → weeks = 2 →
  apples_per_day * days_per_week * weeks = 14 :=
by
  intros apples_per_day days_per_week weeks h1 h2 h3
  sorry

end apples_needed_for_two_weeks_l1109_110914


namespace ratio_of_girls_more_than_boys_l1109_110946

theorem ratio_of_girls_more_than_boys 
  (B : ℕ := 50) 
  (P : ℕ := 123) 
  (driver_assistant_teacher := 3) 
  (h : P = driver_assistant_teacher + B + (P - driver_assistant_teacher - B)) : 
  (P - driver_assistant_teacher - B) - B = 21 → 
  (P - driver_assistant_teacher - B) % B = 21 / 50 := 
sorry

end ratio_of_girls_more_than_boys_l1109_110946


namespace tyler_puppies_l1109_110990

theorem tyler_puppies (dogs : ℕ) (puppies_per_dog : ℕ) (total_puppies : ℕ) 
  (h1 : dogs = 15) (h2 : puppies_per_dog = 5) : total_puppies = 75 :=
by {
  sorry
}

end tyler_puppies_l1109_110990


namespace perimeter_not_55_l1109_110938

def is_valid_perimeter (a b p : ℕ) : Prop :=
  ∃ x : ℕ, a + b > x ∧ a + x > b ∧ b + x > a ∧ p = a + b + x

theorem perimeter_not_55 (a b : ℕ) (h1 : a = 18) (h2 : b = 10) : ¬ is_valid_perimeter a b 55 :=
by
  rw [h1, h2]
  sorry

end perimeter_not_55_l1109_110938


namespace solve_fractional_equation_l1109_110927

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l1109_110927


namespace quadratic_function_negative_values_l1109_110979

theorem quadratic_function_negative_values (a : ℝ) : 
  (∃ x : ℝ, (x^2 - a*x + 1) < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end quadratic_function_negative_values_l1109_110979


namespace arithmetic_progression_primes_l1109_110971

theorem arithmetic_progression_primes (p₁ p₂ p₃ : ℕ) (d : ℕ) 
  (hp₁ : Prime p₁) (hp₁_cond : 3 < p₁) 
  (hp₂ : Prime p₂) (hp₂_cond : 3 < p₂) 
  (hp₃ : Prime p₃) (hp₃_cond : 3 < p₃) 
  (h_prog_1 : p₂ = p₁ + d) (h_prog_2 : p₃ = p₁ + 2 * d) : 
  d % 6 = 0 :=
sorry

end arithmetic_progression_primes_l1109_110971


namespace mukesh_total_debt_l1109_110904

-- Define the initial principal, additional loan, interest rate, and time periods
def principal₁ : ℝ := 10000
def principal₂ : ℝ := 12000
def rate : ℝ := 0.06
def time₁ : ℝ := 2
def time₂ : ℝ := 3

-- Define the interest calculations
def interest₁ : ℝ := principal₁ * rate * time₁
def total_after_2_years : ℝ := principal₁ + interest₁ + principal₂
def interest₂ : ℝ := total_after_2_years * rate * time₂

-- Define the total amount owed after 5 years
def amount_owed : ℝ := total_after_2_years + interest₂

-- The goal is to prove that Mukesh owes 27376 Rs after 5 years
theorem mukesh_total_debt : amount_owed = 27376 := by sorry

end mukesh_total_debt_l1109_110904


namespace abs_sum_leq_abs_l1109_110916

theorem abs_sum_leq_abs (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  |a| + |b| ≤ |a + b| :=
sorry

end abs_sum_leq_abs_l1109_110916


namespace average_speed_l1109_110956

theorem average_speed (x : ℝ) (h₀ : x > 0) : 
  let time1 := x / 90
  let time2 := 2 * x / 20
  let total_distance := 3 * x
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 27 := 
by
  sorry

end average_speed_l1109_110956


namespace max_consecutive_integers_sum_lt_1000_l1109_110951

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end max_consecutive_integers_sum_lt_1000_l1109_110951


namespace parallelogram_height_l1109_110945

/-- The cost of leveling a field in the form of a parallelogram is Rs. 50 per 10 sq. meter, 
    with the base being 54 m and a certain perpendicular distance from the other side. 
    The total cost is Rs. 6480. What is the perpendicular distance from the other side 
    of the parallelogram? -/
theorem parallelogram_height
  (cost_per_10_sq_meter : ℝ)
  (base_length : ℝ)
  (total_cost : ℝ)
  (height : ℝ)
  (h1 : cost_per_10_sq_meter = 50)
  (h2 : base_length = 54)
  (h3 : total_cost = 6480)
  (area : ℝ)
  (h4 : area = (total_cost / cost_per_10_sq_meter) * 10)
  (h5 : area = base_length * height) :
  height = 24 :=
by { sorry }

end parallelogram_height_l1109_110945


namespace ferry_heading_to_cross_perpendicularly_l1109_110991

theorem ferry_heading_to_cross_perpendicularly (river_speed ferry_speed : ℝ) (river_speed_val : river_speed = 12.5) (ferry_speed_val : ferry_speed = 25) : 
  angle_to_cross = 30 :=
by
  -- Definitions for the problem
  let river_velocity : ℝ := river_speed
  let ferry_velocity : ℝ := ferry_speed
  have river_velocity_def : river_velocity = 12.5 := river_speed_val
  have ferry_velocity_def : ferry_velocity = 25 := ferry_speed_val
  -- The actual proof would go here
  sorry

end ferry_heading_to_cross_perpendicularly_l1109_110991


namespace ab_cd_value_l1109_110961

theorem ab_cd_value (a b c d: ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 5)
  (h3 : a + c + d = 14)
  (h4 : b + c + d = 9) :
  a * b + c * d = 338 / 9 := 
sorry

end ab_cd_value_l1109_110961


namespace curve_symmetric_reflection_l1109_110952

theorem curve_symmetric_reflection (f : ℝ → ℝ → ℝ) :
  (∀ x y, f x y = 0 ↔ f (y + 3) (x - 3) = 0) → 
  (∀ x y, (x - y - 3 = 0) → (f (y + 3) (x - 3) = 0)) :=
sorry

end curve_symmetric_reflection_l1109_110952


namespace eleven_billion_in_scientific_notation_l1109_110993

-- Definition: "Billion" is 10^9
def billion : ℝ := 10^9

-- Theorem: 11 billion can be represented as 1.1 * 10^10
theorem eleven_billion_in_scientific_notation : 11 * billion = 1.1 * 10^10 := by
  sorry

end eleven_billion_in_scientific_notation_l1109_110993


namespace small_ball_rubber_bands_l1109_110941

theorem small_ball_rubber_bands (S : ℕ) 
    (large_ball : ℕ := 300) 
    (initial_rubber_bands : ℕ := 5000) 
    (small_balls : ℕ := 22) 
    (large_balls : ℕ := 13) :
  (small_balls * S + large_balls * large_ball = initial_rubber_bands) → S = 50 := by
    sorry

end small_ball_rubber_bands_l1109_110941


namespace xy_sum_cases_l1109_110933

theorem xy_sum_cases (x y : ℕ) (hxy1 : 0 < x) (hxy2 : x < 30)
                      (hy1 : 0 < y) (hy2 : y < 30)
                      (h : x + y + x * y = 119) : (x + y = 24) ∨ (x + y = 20) :=
sorry

end xy_sum_cases_l1109_110933


namespace simplify_power_l1109_110963

theorem simplify_power (x : ℝ) : (3 * x^4)^4 = 81 * x^16 :=
by sorry

end simplify_power_l1109_110963


namespace irrational_power_to_nat_l1109_110988

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log (Real.sqrt 2) 

theorem irrational_power_to_nat 
  (ha_irr : ¬ ∃ (q : ℚ), a = q)
  (hb_irr : ¬ ∃ (q : ℚ), b = q) : (a ^ b) = 3 := by
  -- \[a = \sqrt{2}, b = \log_{\sqrt{2}}(3)\]
  sorry

end irrational_power_to_nat_l1109_110988


namespace impossible_equal_sums_3x3_l1109_110925

theorem impossible_equal_sums_3x3 (a b c d e f g h i : ℕ) :
  a + b + c = 13 ∨ a + b + c = 14 ∨ a + b + c = 15 ∨ a + b + c = 16 ∨ a + b + c = 17 ∨ a + b + c = 18 ∨ a + b + c = 19 ∨ a + b + c = 20 →
  (a + d + g) = 13 ∨ (a + d + g) = 14 ∨ (a + d + g) = 15 ∨ (a + d + g) = 16 ∨ (a + d + g) = 17 ∨ (a + d + g) = 18 ∨ (a + d + g) = 19 ∨ (a + d + g) = 20 →
  (a + e + i) = 13 ∨ (a + e + i) = 14 ∨ (a + e + i) = 15 ∨ (a + e + i) = 16 ∨ (a + e + i) = 17 ∨ (a + e + i) = 18 ∨ (a + e + i) = 19 ∨ (a + e + i) = 20 →
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧ 1 ≤ f ∧ f ≤ 9 ∧ 1 ≤ g ∧ g ≤ 9 ∧ 1 ≤ h ∧ h ≤ 9 ∧ 1 ≤ i ∧ i ≤ 9 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ g ≠ h ∧ g ≠ i ∧ h ≠ i →
  false :=
sorry

end impossible_equal_sums_3x3_l1109_110925


namespace intersection_A_complementB_l1109_110954

universe u

def R : Type := ℝ

def A (x : ℝ) : Prop := 0 < x ∧ x < 2

def B (x : ℝ) : Prop := x ≥ 1

def complement_B (x : ℝ) : Prop := x < 1

theorem intersection_A_complementB : 
  ∀ x : ℝ, (A x ∧ complement_B x) ↔ (0 < x ∧ x < 1) := 
by 
  sorry

end intersection_A_complementB_l1109_110954


namespace simplify_and_evaluate_expression_l1109_110936

theorem simplify_and_evaluate_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  ( (2 * x - 3) / (x - 2) - 1 ) / ( (x^2 - 2 * x + 1) / (x - 2) ) = 1 / 2 :=
by {
  sorry
}

end simplify_and_evaluate_expression_l1109_110936


namespace day_crew_fraction_correct_l1109_110973

-- Given conditions
variables (D W : ℕ)
def night_boxes_per_worker := (5 : ℚ) / 8 * D
def night_workers := (3 : ℚ) / 5 * W

-- Total boxes loaded
def total_day_boxes := D * W
def total_night_boxes := night_boxes_per_worker D * night_workers W

-- Fraction of boxes loaded by day crew
def fraction_loaded_by_day_crew := total_day_boxes D W / (total_day_boxes D W + total_night_boxes D W)

-- Theorem to prove
theorem day_crew_fraction_correct (D W : ℕ) : fraction_loaded_by_day_crew D W = (8 : ℚ) / 11 :=
by
  sorry

end day_crew_fraction_correct_l1109_110973


namespace CarlosAndDianaReceivedAs_l1109_110947

variables (Alan Beth Carlos Diana : Prop)
variable (num_A : ℕ)

-- Condition 1: Alan => Beth
axiom AlanImpliesBeth : Alan → Beth

-- Condition 2: Beth => Carlos
axiom BethImpliesCarlos : Beth → Carlos

-- Condition 3: Carlos => Diana
axiom CarlosImpliesDiana : Carlos → Diana

-- Condition 4: Only two students received an A
axiom OnlyTwoReceivedAs : num_A = 2

-- Theorem: Carlos and Diana received A's
theorem CarlosAndDianaReceivedAs : ((Alan ∧ Beth ∧ Carlos ∧ Diana → False) ∧
                                   (Beth ∧ Carlos ∧ Diana → False) ∧
                                   (Alan ∧ Beth ∧ Diana → False) ∧
                                   (Alan ∧ Beth ∧ Carlos → False) ∧
                                   (Alan ∧ Diana → False) ∧
                                   (Beth ∧ Carlos → False) ∧
                                   (Alan ∧ Carlos → False) ∧
                                   (Beth ∧ Diana → False)) → (Carlos ∧ Diana) :=
by
  intros h
  have h1 := AlanImpliesBeth
  have h2 := BethImpliesCarlos
  have h3 := CarlosImpliesDiana
  have h4 := OnlyTwoReceivedAs
  sorry

end CarlosAndDianaReceivedAs_l1109_110947


namespace candles_time_l1109_110953

/-- Prove that if two candles of equal length are lit at a certain time,
and by 6 PM one of the stubs is three times the length of the other,
the correct time to light the candles is 4:00 PM. -/

theorem candles_time :
  ∀ (ℓ : ℝ) (t : ℝ),
  (∀ t1 t2 : ℝ, t = t1 + t2 → 
    (180 - t1) = 3 * (300 - t2) / 3 → 
    18 <= 6 ∧ 0 <= t → ℓ / 180 * (180 - (t - 180)) = 3 * (ℓ / 300 * (300 - (6 - t))) →
    t = 4
  ) := 
by 
  sorry

end candles_time_l1109_110953


namespace no_common_points_l1109_110975

theorem no_common_points (x0 y0 : ℝ) (h : x0^2 < 4 * y0) :
  ∀ (x y : ℝ), (x^2 = 4 * y) → (x0 * x = 2 * (y + y0)) →
  false := 
by
  sorry

end no_common_points_l1109_110975


namespace largest_divisible_number_l1109_110958

theorem largest_divisible_number : ∃ n, n = 9950 ∧ n ≤ 9999 ∧ (∀ m, m ≤ 9999 ∧ m % 50 = 0 → m ≤ n) :=
by {
  sorry
}

end largest_divisible_number_l1109_110958


namespace min_distinct_sums_l1109_110929

theorem min_distinct_sums (n : ℕ) (hn : n ≥ 5) (s : Finset ℕ) 
  (hs : s.card = n) : 
  ∃ (t : Finset ℕ), (∀ (x y : ℕ), x ∈ s → y ∈ s → x < y → (x + y) ∈ t) ∧ t.card = 2 * n - 3 :=
by
  sorry

end min_distinct_sums_l1109_110929


namespace nicky_profit_l1109_110930

theorem nicky_profit (value_traded_away value_received : ℤ)
  (h1 : value_traded_away = 2 * 8)
  (h2 : value_received = 21) :
  value_received - value_traded_away = 5 :=
by
  sorry

end nicky_profit_l1109_110930


namespace solution_set_inequality_l1109_110915

theorem solution_set_inequality (x : ℝ) : (3 * x^2 + 7 * x ≤ 2) ↔ (-2 ≤ x ∧ x ≤ 1 / 3) :=
by
  sorry

end solution_set_inequality_l1109_110915


namespace wire_ratio_l1109_110985

theorem wire_ratio (a b : ℝ) (h_eq_area : (a / 4)^2 = 2 * (b / 8)^2 * (1 + Real.sqrt 2)) :
  a / b = Real.sqrt (2 + Real.sqrt 2) / 2 :=
by
  sorry

end wire_ratio_l1109_110985


namespace complex_product_polar_form_l1109_110968

theorem complex_product_polar_form :
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 360 ∧ 
  (r = 12 ∧ θ = 245) :=
by
  sorry

end complex_product_polar_form_l1109_110968


namespace mean_score_of_students_who_failed_l1109_110944

noncomputable def mean_failed_score : ℝ := sorry

theorem mean_score_of_students_who_failed (t p proportion_passed proportion_failed : ℝ) (h1 : t = 6) (h2 : p = 8) (h3 : proportion_passed = 0.6) (h4 : proportion_failed = 0.4) : mean_failed_score = 3 :=
by
  sorry

end mean_score_of_students_who_failed_l1109_110944


namespace sum_of_remainders_l1109_110902

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 47 = 25) (h2 : b % 47 = 20) (h3 : c % 47 = 3) : 
  (a + b + c) % 47 = 1 := 
by {
  sorry
}

end sum_of_remainders_l1109_110902


namespace point_Q_and_d_l1109_110962

theorem point_Q_and_d :
  ∃ (a b c d : ℝ),
    (∀ x y z : ℝ, (x - 2)^2 + (y - 3)^2 + (z + 4)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2) ∧
    (8 * a - 6 * b + 32 * c = d) ∧ a = 6 ∧ b = 0 ∧ c = 12 ∧ d = 151 :=
by
  existsi 6, 0, 12, 151
  sorry

end point_Q_and_d_l1109_110962


namespace sum_arithmetic_sequence_l1109_110905

theorem sum_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith_seq : ∀ n, S (n + 1) - S n = a n)
  (h_S2 : S 2 = 4) 
  (h_S4 : S 4 = 16) 
: a 5 + a 6 = 20 :=
sorry

end sum_arithmetic_sequence_l1109_110905


namespace complete_the_square_l1109_110932

theorem complete_the_square (d e f : ℤ) (h1 : d > 0)
  (h2 : 25 * d * d = 25)
  (h3 : 10 * d * e = 30)
  (h4 : 25 * d * d * (d * x + e) * (d * x + e) = 25 * x * x * 25 + 30 * x * 25 * d + 25 * e * e - 9)
  : d + e + f = 41 := 
  sorry

end complete_the_square_l1109_110932


namespace age_of_youngest_child_l1109_110922

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 65) : x = 7 :=
sorry

end age_of_youngest_child_l1109_110922


namespace factory_output_decrease_l1109_110901

noncomputable def original_output (O : ℝ) : ℝ :=
  O

noncomputable def increased_output_10_percent (O : ℝ) : ℝ :=
  O * 1.1

noncomputable def increased_output_30_percent (O : ℝ) : ℝ :=
  increased_output_10_percent O * 1.3

noncomputable def percentage_decrease_needed (original new_output : ℝ) : ℝ :=
  ((new_output - original) / new_output) * 100

theorem factory_output_decrease (O : ℝ) : 
  abs (percentage_decrease_needed (original_output O) (increased_output_30_percent O) - 30.07) < 0.01 :=
by
  sorry

end factory_output_decrease_l1109_110901


namespace Lauren_total_revenue_l1109_110935

noncomputable def LaurenMondayEarnings (views subscriptions : ℕ) : ℝ :=
  (views * 0.40) + (subscriptions * 0.80)

noncomputable def LaurenTuesdayEarningsEUR (views subscriptions : ℕ) : ℝ :=
  (views * 0.40) + (subscriptions * 0.75)

noncomputable def convertEURtoUSD (eur : ℝ) : ℝ :=
  eur * (1 / 0.85)

noncomputable def convertGBPtoUSD (gbp : ℝ) : ℝ :=
  gbp * 1.38

noncomputable def LaurenWeekendEarnings (sales : ℝ) : ℝ :=
  (sales * 0.10)

theorem Lauren_total_revenue :
  let monday_views := 80
  let monday_subscriptions := 20
  let tuesday_views := 100
  let tuesday_subscriptions := 27
  let weekend_sales := 100

  let monday_earnings := LaurenMondayEarnings monday_views monday_subscriptions
  let tuesday_earnings_eur := LaurenTuesdayEarningsEUR tuesday_views tuesday_subscriptions
  let tuesday_earnings_usd := convertEURtoUSD tuesday_earnings_eur
  let weekend_earnings_gbp := LaurenWeekendEarnings weekend_sales
  let weekend_earnings_usd := convertGBPtoUSD weekend_earnings_gbp

  monday_earnings + tuesday_earnings_usd + weekend_earnings_usd = 132.68 :=
by
  sorry

end Lauren_total_revenue_l1109_110935
