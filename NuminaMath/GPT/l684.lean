import Mathlib

namespace bob_paid_24_percent_of_SRP_l684_684087

theorem bob_paid_24_percent_of_SRP
  (P : ℝ) -- Suggested Retail Price (SRP)
  (MP : ℝ) -- Marked Price (MP)
  (price_bob_paid : ℝ) -- Price Bob Paid
  (h1 : MP = 0.60 * P) -- Condition 1: MP is 60% of SRP
  (h2 : price_bob_paid = 0.40 * MP) -- Condition 2: Bob paid 40% of the MP
  : (price_bob_paid / P) * 100 = 24 := -- Bob paid 24% of the SRP
by
  sorry

end bob_paid_24_percent_of_SRP_l684_684087


namespace units_digit_of_50_factorial_is_0_l684_684167

theorem units_digit_of_50_factorial_is_0 : 
  (∃ n : ℕ, 50! ≡ n [MOD 10]) ∧ (n = 0) := sorry

end units_digit_of_50_factorial_is_0_l684_684167


namespace eunseo_change_correct_l684_684644

-- Define the given values
def r : ℕ := 3
def p_r : ℕ := 350
def b : ℕ := 2
def p_b : ℕ := 180
def P : ℕ := 2000

-- Define the total cost of candies and the change
def total_cost := r * p_r + b * p_b
def change := P - total_cost

-- Theorem statement
theorem eunseo_change_correct : change = 590 := by
  -- proof not required, so using sorry
  sorry

end eunseo_change_correct_l684_684644


namespace definite_integral_eq_e_l684_684626

theorem definite_integral_eq_e : ∫ x in 1..Real.exp 1, (1 + 1 / x) = Real.exp 1 := by
sorry

end definite_integral_eq_e_l684_684626


namespace combined_investment_yield_l684_684857

-- Definitions of the conditions
def yield_A : ℝ := 0.14
def value_A : ℝ := 500

def yield_B : ℝ := 0.08
def value_B : ℝ := 750

def yield_C : ℝ := 0.12
def value_C : ℝ := 1000

-- Complete the proof
theorem combined_investment_yield :
  let income_A := yield_A * value_A in
  let income_B := yield_B * value_B in
  let income_C := yield_C * value_C in
  let total_income := income_A + income_B + income_C in
  let total_value := value_A + value_B + value_C in
  (total_income / total_value) = 0.1111 :=
by
  -- establish the required calculations
  let income_A := yield_A * value_A
  let income_B := yield_B * value_B
  let income_C := yield_C * value_C
  let total_income := income_A + income_B + income_C
  let total_value := value_A + value_B + value_C
  -- reduced form of our problem
  have h : (total_income / total_value) = 0.1111 := sorry
  exact h

end combined_investment_yield_l684_684857


namespace seventh_term_l684_684526

def nth_term (n : ℕ) (a : ℝ) : ℝ :=
  (-2) ^ n * a ^ (2 * n - 1)

theorem seventh_term (a : ℝ) : nth_term 7 a = -128 * a ^ 13 :=
by sorry

end seventh_term_l684_684526


namespace find_length_LP_l684_684534

def length_AC := 800
def length_BC := 500
def length_AK := length_AC / 2
def length_KC := length_AC / 2
def am_length := 240
def ratio_AL_LB := length_AC / length_BC -- 8/5
def ratio_AM_LP := ratio_AL_LB + 1 -- 13/5

theorem find_length_LP : 
  ∃ LP : ℝ, (am_length / LP = ratio_AM_LP) ∧ (LP = 1200 / 13) :=
by
  use 1200 / 13
  calc
    am_length / (1200 / 13) = 240 / (1200 / 13)   : by rfl
    ... = 240 * 13 / 1200   : by ring
    ... = 13 / 5            : by norm_num
  split
  exact rfl
  exact rfl

end find_length_LP_l684_684534


namespace probability_selected_ball_is_multiple_of_4_9_or_both_l684_684052

-- Define the conditions: total number of balls and the probability in question
def total_balls := 70
def probability_multiple_4_or_9_or_both : ℚ := 23 / 70

-- Define what it means for a number to be a multiple of 4, 9, or both
def is_multiple_of_4_9_or_both (n : ℕ) : Prop :=
  n % 4 = 0 ∨ n % 9 = 0

-- Define the problem statement to prove
theorem probability_selected_ball_is_multiple_of_4_9_or_both :
  let chosen_ball (n : ℕ) := n ∈ finset.range (total_balls + 1) ∧ is_multiple_of_4_9_or_both n,
  (finset.card (finset.filter chosen_ball (finset.range (total_balls + 1))) : ℚ) / total_balls = probability_multiple_4_or_9_or_both :=
begin
  sorry
end

end probability_selected_ball_is_multiple_of_4_9_or_both_l684_684052


namespace subset_with_difference_l684_684100

theorem subset_with_difference {S : Finset ℕ} (h : S = {1, 2, 3, 4, 5}) :
  ∃ A B : Finset ℕ, A ∪ B = S ∧ A ∩ B = ∅ ∧ 
  ((∃ x y ∈ A, (x - y) ∈ A ∨ (y - x) ∈ A) ∨ (∃ x y ∈ B, (x - y) ∈ B ∨ (y - x) ∈ B)) :=
by
  sorry

end subset_with_difference_l684_684100


namespace gcd_282_470_l684_684914

theorem gcd_282_470 : Int.gcd 282 470 = 94 := by
  sorry

end gcd_282_470_l684_684914


namespace problem_ineq_l684_684485

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : |f (m + n) - f m| ≤ (n : ℝ) / (m : ℝ)

theorem problem_ineq (k : ℕ) (hk : 0 < k) :
  ∑ i in Finset.range k + 1, |f (2^k) - f (2^i)| ≤ (k * (k - 1)) / 2 := sorry

end problem_ineq_l684_684485


namespace find_odd_natural_numbers_l684_684596

-- Definition of a friendly number
def is_friendly (n : ℕ) : Prop :=
  ∀ i, (n / 10^i) % 10 = (n / 10^(i + 1)) % 10 + 1 ∨ (n / 10^i) % 10 = (n / 10^(i + 1)) % 10 - 1

-- Given condition: n is divisible by 64m
def is_divisible_by_64m (n m : ℕ) : Prop :=
  64 * m ∣ n

-- Proof problem statement
theorem find_odd_natural_numbers (m : ℕ) (hm1 : m % 2 = 1) :
  (5 ∣ m → ¬ ∃ n, is_friendly n ∧ is_divisible_by_64m n m) ∧ 
  (¬ 5 ∣ m → ∃ n, is_friendly n ∧ is_divisible_by_64m n m) :=
by
  sorry

end find_odd_natural_numbers_l684_684596


namespace corridor_length_correct_l684_684394

/-- Scale representation in the blueprint: 1 cm represents 10 meters. --/
def scale_cm_to_m (cm: ℝ): ℝ := cm * 10

/-- Length of the corridor in the blueprint. --/
def blueprint_length_cm: ℝ := 9.5

/-- Real-life length of the corridor. --/
def real_life_length: ℝ := 95

/-- Proof that the real-life length of the corridor is correctly calculated. --/
theorem corridor_length_correct :
  scale_cm_to_m blueprint_length_cm = real_life_length :=
by
  sorry

end corridor_length_correct_l684_684394


namespace prove_tangency_l684_684027

noncomputable def parabola (p : ℝ) (hp : p > 0) : set (ℝ × ℝ) :=
  { z | z.snd^2 = 2 * p * z.fst }

structure Focus (p : ℝ) (hp : p > 0) : Type :=
(point : ℝ × ℝ)
(is_focus : point = (p / 2, 0))

structure PointOutsideParabola (p : ℝ) (hp : p > 0) : Type :=
(point : ℝ × ℝ)
(is_outside : ¬∃ x ∈ parabola p hp, point = x ∧ point.snd = 0)

structure TangentPoints (p : ℝ) (hp : p > 0) (P : PointOutsideParabola p hp) : Type :=
(A B : ℝ × ℝ)
(tangentA : A ∈ parabola p hp)
(tangentB : B ∈ parabola p hp)
(tangent_condition_A : ∃ l : line ℝ, l.contains P.point ∧ l.contains A)
(tangent_condition_B : ∃ l : line ℝ, l.contains P.point ∧ l.contains B)

structure IntersectionY (Q : ℝ × ℝ) : Type :=
(C D : ℝ × ℝ)
(intersects_y_axis_C : C.fst = 0)
(intersects_y_axis_D : D.fst = 0)
(line_QA : line ℝ)
(line_QB : line ℝ)
(intersection_C : ∃ A, tangents_to_A line_QA A ∧ P.tangentA)
(intersection_D : ∃ B, tangents_to_B line_QB B ∧ P.tangentB)

structure CircumcenterQAB (Q : PointOutsideParabola p hp) (T : TangentPoints p hp Q) : Type :=
(M : ℝ × ℝ)
(is_circumcenter: ∃ circumcircle : circle ℝ, circumcenter_of_triangle M Q.point T.A T.B)

theorem prove_tangency 
  (p : ℝ) (hp : p > 0)
  (F : Focus p hp)
  (Q : PointOutsideParabola p hp)
  (T : TangentPoints p hp Q)
  (I : IntersectionY Q.point)
  (CQC : CircumcenterQAB Q T)
  : tangent_to_circumcircle F.point CQC.M I.C I.D :=
sorry

end prove_tangency_l684_684027


namespace fg_of_1_eq_15_l684_684743

def f (x : ℝ) := 2 * x - 3
def g (x : ℝ) := (x + 2) ^ 2

theorem fg_of_1_eq_15 : f (g 1) = 15 :=
by
  sorry

end fg_of_1_eq_15_l684_684743


namespace eccentricity_of_ellipse_l684_684686

theorem eccentricity_of_ellipse (m n : ℝ) (h1 : m > 0) (h2: n > 0)
  (h3: 1/m + 2/n = 1) : sqrt(1 - (2 / 4) ^ 2) = sqrt(3) / 2 :=
by sorry

end eccentricity_of_ellipse_l684_684686


namespace intersection_M_N_l684_684681

def M : Set ℝ := {y | ∃ x : ℝ, y = x - |x|}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {0} :=
  sorry

end intersection_M_N_l684_684681


namespace units_digit_of_product_1_to_50_is_zero_l684_684151

theorem units_digit_of_product_1_to_50_is_zero :
  Nat.digits 10 (∏ i in Finset.range 51, i) = [0] :=
sorry

end units_digit_of_product_1_to_50_is_zero_l684_684151


namespace vector_dot_product_l684_684365

theorem vector_dot_product :
  ∀ x : ℝ, let a := (1, 2 : ℝ × ℝ), b := (x, -1 : ℝ × ℝ) in
  (a.1 * (a.1 - b.1) = a.2 * (a.2 - b.2)) →
  (a.1 * b.1 + a.2 * b.2 = -5/2) :=
by
  intros x a b h
  simp [a, b] at h
  sorry

end vector_dot_product_l684_684365


namespace irrational_seq_for_all_n_ge_1_limit_of_seq_exists_and_is_rational_l684_684265

noncomputable def seq : ℕ → ℝ
| 0       := 2
| (n + 1) := real.sqrt (2 * seq n - 1)

theorem irrational_seq_for_all_n_ge_1 (n : ℕ) (hn : n ≥ 1) : irrational (seq n) :=
sorry

theorem limit_of_seq_exists_and_is_rational : ∃ l : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |seq n - l| < ε) ∧ l = 1 :=
sorry

end irrational_seq_for_all_n_ge_1_limit_of_seq_exists_and_is_rational_l684_684265


namespace units_digit_of_product_1_to_50_is_zero_l684_684155

theorem units_digit_of_product_1_to_50_is_zero :
  Nat.digits 10 (∏ i in Finset.range 51, i) = [0] :=
sorry

end units_digit_of_product_1_to_50_is_zero_l684_684155


namespace cylindrical_to_cartesian_l684_684679

theorem cylindrical_to_cartesian :
  ∀ (ρ θ z : ℝ), (ρ = sqrt 2) ∧ (θ = 5 * Real.pi / 4) ∧ (z = sqrt 2) →
  let x := ρ * Real.cos θ in
  let y := ρ * Real.sin θ in
  (x, y, z) = (-1, -1, sqrt 2) :=
by
  sorry

end cylindrical_to_cartesian_l684_684679


namespace veronica_more_stairs_l684_684051

theorem veronica_more_stairs (S : ℕ) (V : ℕ) (X : ℕ) 
  (h1 : S = 318) 
  (h2 : V = (S / 2) + X) 
  (h3 : S + V = 495) : X = 18 :=
by
  have h4 : 318 + 159 + X = 495 := by rw [←h1, Nat.div_mul_left 318 (by norm_num), h2, h3]
  have h5 : 477 + X = 495 := by rw [show (318 + 159) = 477, from rfl, h4]
  sorry

end veronica_more_stairs_l684_684051


namespace symmetric_point_on_line_l684_684889

noncomputable theory
open_locale classical

variables {A B C H E C' B' O I F : Point}
variables (ABC : Triangle)
variables [IsScalene ABC]

/-- Define the orthocenter of the triangle ABC --/
def orthocenter (ABC : Triangle) : Point := sorry

/-- Define the circumcenter of the triangle ABC --/
def circumcenter (ABC : Triangle) : Point := sorry

/-- Define the incenter of the triangle ABC --/
def incenter (ABC : Triangle) : Point := sorry

/-- Define the incircle of the triangle ABC --/
def incircle (ABC : Triangle) : Circle := sorry

/-- Definition of point E as the midpoint of AH --/
def midpoint_of_AH (A H : Point) : Point := sorry

/-- Define the tangent points of the incircle with sides AB and AC --/
def tangent_at_AB (incircle : Circle) (A B : Point) : Point := sorry
def tangent_at_AC (incircle : Circle) (A C : Point) : Point := sorry

/-- The main theorem stating F lies on the line passing through both the circumcenter and the incenter --/
theorem symmetric_point_on_line (ABC : Triangle) [IsScalene ABC]
  (H := orthocenter ABC)
  (O := circumcenter ABC)
  (I := incenter ABC)
  (incircle := incircle ABC)
  (E := midpoint_of_AH A H)
  (C' := tangent_at_AB incircle A B)
  (B' := tangent_at_AC incircle A C)
  (F : Point)
  (hF : symmetric F E B' C') :
  collinear [F, O, I] :=
sorry

end symmetric_point_on_line_l684_684889


namespace percentage_less_than_l684_684187

theorem percentage_less_than (x y : ℝ) (h : y = x * 1.6) : (y - x) / y * 100 = 37.5 :=
by simp [h] sorry

end percentage_less_than_l684_684187


namespace tangent_line_at_a_eq_neg1_monotonicity_intervals_extremum_range_of_a_l684_684698

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / x - a * Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 + f a x

-- (I) Tangent line equation at point (1, f(1)) when a = -1.
theorem tangent_line_at_a_eq_neg1 : 
  ∃ (line_eq : ℝ → ℝ), a = -1 ∧ line_eq = (λ x, -x + 3) := 
sorry

-- (II) Monotonicity intervals of f.
theorem monotonicity_intervals (a : ℝ): 
  ((a ≥ 0) → (∀ x ∈ Ioi (0 : ℝ), ∀ y ∈ Ioi (0 : ℝ), x < y → f a y < f a x))
  ∧ ((a < 0) → ((∀ x ∈ Ioo 0 (-2 / a), ∀ y ∈ Ioo 0 (-2 / a), x < y → f a y < f a x) 
  ∧ (∀ x ∈ Ioi (-2 / a), ∀ y ∈ Ioi (-2 / a), x < y → f a x < f a y))) := 
sorry

-- (III) The range of values for a such that g has an extremum in (0,1).
theorem extremum_range_of_a: 
  (∃ (a : ℝ), g a 0 = 0) ↔ a < 0 := 
sorry

end tangent_line_at_a_eq_neg1_monotonicity_intervals_extremum_range_of_a_l684_684698


namespace triple_composition_g_eq_107_l684_684731

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_g_eq_107 : g(g(g(3))) = 107 := by
  sorry

end triple_composition_g_eq_107_l684_684731


namespace two_pow_15000_mod_1250_l684_684931

theorem two_pow_15000_mod_1250 (h : 2 ^ 500 ≡ 1 [MOD 1250]) :
  2 ^ 15000 ≡ 1 [MOD 1250] :=
sorry

end two_pow_15000_mod_1250_l684_684931


namespace units_digit_of_product_1_to_50_is_zero_l684_684152

theorem units_digit_of_product_1_to_50_is_zero :
  Nat.digits 10 (∏ i in Finset.range 51, i) = [0] :=
sorry

end units_digit_of_product_1_to_50_is_zero_l684_684152


namespace largest_integer_base_7_l684_684789

theorem largest_integer_base_7 :
  let M := 66 in
  M ^ 2 = 48 ^ 2 :=
by
  let M := (6 * 7 + 6) in
  have h : M ^ 2 = 48 ^ 2 := rfl
  sorry -- Proof not required.

end largest_integer_base_7_l684_684789


namespace sale_in_second_month_l684_684212

theorem sale_in_second_month 
  (sale_first_month: ℕ := 2500)
  (sale_third_month: ℕ := 3540)
  (sale_fourth_month: ℕ := 1520)
  (average_sale: ℕ := 2890)
  (total_sales: ℕ := 11560) :
  sale_first_month + sale_third_month + sale_fourth_month + (sale_second_month: ℕ) = total_sales → 
  sale_second_month = 4000 := 
by
  intros h
  sorry

end sale_in_second_month_l684_684212


namespace refrigerator_cost_is_15000_l684_684834

theorem refrigerator_cost_is_15000 (R : ℝ) 
  (phone_cost : ℝ := 8000)
  (phone_profit : ℝ := 0.10) 
  (fridge_loss : ℝ := 0.03) 
  (overall_profit : ℝ := 350) :
  (0.97 * R + phone_cost * (1 + phone_profit) = (R + phone_cost) + overall_profit) →
  (R = 15000) :=
by
  sorry

end refrigerator_cost_is_15000_l684_684834


namespace probability_two_points_one_unit_apart_l684_684913

theorem probability_two_points_one_unit_apart :
  let n := 12 -- number of points
  let total_ways := nat.choose n 2 -- total ways to choose 2 points from 12
  let favorable_ways := 12 -- number of favorable outcomes where points are one unit apart
  (favorable_ways : ℚ) / total_ways = (2 : ℚ) / 11 :=
by
  -- Definitions and calculations would go here.
  sorry

end probability_two_points_one_unit_apart_l684_684913


namespace minimum_small_bottles_l684_684239

-- Define the capacities of the bottles
def small_bottle_capacity : ℕ := 35
def large_bottle_capacity : ℕ := 500

-- Define the number of small bottles needed to fill a large bottle
def small_bottles_needed_to_fill_large : ℕ := 
  (large_bottle_capacity + small_bottle_capacity - 1) / small_bottle_capacity

-- Statement of the theorem
theorem minimum_small_bottles : small_bottles_needed_to_fill_large = 15 := by
  sorry

end minimum_small_bottles_l684_684239


namespace snickers_cost_l684_684760

variable (S : ℝ)

def cost_of_snickers (n : ℝ) : Prop :=
  2 * n + 3 * (2 * n) = 12

theorem snickers_cost (h : cost_of_snickers S) : S = 1.50 :=
by
  sorry

end snickers_cost_l684_684760


namespace sum_numerator_denominator_eq_10475_l684_684180

def repeating_decimal_fraction_sum : ℚ := 0.047604760476.repeating -- Define the repeating decimal

theorem sum_numerator_denominator_eq_10475 :
  let frac := repeating_decimal_fraction_sum -- Simplify the fraction
  in frac.num + frac.denom = 10475 := by
sorry

end sum_numerator_denominator_eq_10475_l684_684180


namespace volleyball_match_probabilities_l684_684862

noncomputable def probability_of_team_A_winning : ℚ := (2 / 3) ^ 3
noncomputable def probability_of_team_B_winning_3_0 : ℚ := 1 / 3
noncomputable def probability_of_team_B_winning_3_1 : ℚ := (2 / 3) * (1 / 3)
noncomputable def probability_of_team_B_winning_3_2 : ℚ := (2 / 3) ^ 2 * (1 / 3)

theorem volleyball_match_probabilities :
  probability_of_team_A_winning = 8 / 27 ∧
  probability_of_team_B_winning_3_0 = 1 / 3 ∧
  probability_of_team_B_winning_3_1 ≠ 1 / 9 ∧
  probability_of_team_B_winning_3_2 ≠ 4 / 9 :=
by
  sorry

end volleyball_match_probabilities_l684_684862


namespace problem_1_problem_2_l684_684841

-- Define the first problem
theorem problem_1 :
  0.064 ^ (- (1 / 3)) - ((-1 / 8) ^ 0) + 16 ^ (3 / 4) + 0.25 ^ (1 / 2) = 10 :=
by
  sorry

-- Define the second problem
theorem problem_2 :
  (1 / 2) * log 25 10 + log 2 10 + (1 / 3) ^ (log 2 3) - log 9 2 * log 2 3 = -1 / 2 :=
by
  sorry

end problem_1_problem_2_l684_684841


namespace distinct_cubes_count_l684_684724

theorem distinct_cubes_count : 
  ∃ N, N = 1680 ∧ ∀ (cubes : Finset (Fin 8)), Finset.card cubes = 8 →
  (∃! G : Finset (Fin 24), true) := sorry

end distinct_cubes_count_l684_684724


namespace Lisa_goal_achievable_l684_684617

open Nat

theorem Lisa_goal_achievable :
  ∀ (total_quizzes quizzes_with_A goal_percentage : ℕ),
  total_quizzes = 60 →
  quizzes_with_A = 25 →
  goal_percentage = 85 →
  (quizzes_with_A < goal_percentage * total_quizzes / 100) →
  (∃ remaining_quizzes, goal_percentage * total_quizzes / 100 - quizzes_with_A > remaining_quizzes) :=
by
  intros total_quizzes quizzes_with_A goal_percentage h_total h_A h_goal h_lack
  let needed_quizzes := goal_percentage * total_quizzes / 100
  let remaining_quizzes := total_quizzes - 35
  have h_needed := needed_quizzes - quizzes_with_A
  use remaining_quizzes
  sorry

end Lisa_goal_achievable_l684_684617


namespace proof_problem_l684_684897

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∀ x, (a * x^2 + b * x + 2 > 0) ↔ (x ∈ Set.Ioo (-1/2 : ℝ) (1/3 : ℝ))) 

theorem proof_problem (a b : ℝ) (h : problem_statement a b) : a + b = -14 :=
sorry

end proof_problem_l684_684897


namespace community_children_count_l684_684873

theorem community_children_count (total_members : ℕ) (pct_adult_men : ℝ) (ratio_adult_women : ℝ) :
  total_members = 2000 → pct_adult_men = 0.3 → ratio_adult_women = 2 →
  let num_adult_men := (pct_adult_men * total_members).to_nat in
  let num_adult_women := (ratio_adult_women * num_adult_men).to_nat in
  let total_adults := num_adult_men + num_adult_women in
  let num_children := total_members - total_adults in
  num_children = 200 :=
by
  intro h1 h2 h3
  let num_adult_men := (0.3 * 2000).to_nat
  let num_adult_women := (2 * num_adult_men).to_nat
  let total_adults := num_adult_men + num_adult_women
  let num_children := 2000 - total_adults
  -- we need to skip the proof part
  sorry

end community_children_count_l684_684873


namespace circular_arrangement_exists_l684_684998

theorem circular_arrangement_exists :
  ∃ (f : Fin 9 → ℕ), (∀ i, f i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  Function.Bijective f ∧
  ∀ i, ¬( (f i + f ((i + 1) % 9)) % 3 = 0 ∨ (f i + f ((i + 1) % 9)) % 5 = 0 ∨ (f i + f ((i + 1) % 9)) % 7 = 0) :=
sorry

end circular_arrangement_exists_l684_684998


namespace Amy_total_crumbs_eq_3z_l684_684114

variable (T C z : ℕ)

-- Given conditions
def total_crumbs_Arthur := T * C = z
def trips_Amy := 2 * T
def crumbs_per_trip_Amy := 3 * C / 2

-- Problem statement
theorem Amy_total_crumbs_eq_3z (h : total_crumbs_Arthur T C z) :
  (trips_Amy T) * (crumbs_per_trip_Amy C) = 3 * z :=
sorry

end Amy_total_crumbs_eq_3z_l684_684114


namespace perfect_power_prime_divisor_product_l684_684891

theorem perfect_power_prime_divisor_product (n : ℕ) (h1 : ∏ d in (fintype.pi_finset (λ d, d ∣ n)), d = 1024) (h2 : ∃ p k : ℕ, (Nat.Prime p) ∧ (n = p^k)) : n = 1024 :=
by
  sorry

end perfect_power_prime_divisor_product_l684_684891


namespace slant_height_base_plane_angle_l684_684892

noncomputable def angle_between_slant_height_and_base_plane (R : ℝ) : ℝ :=
  Real.arcsin ((Real.sqrt 13 - 1) / 3)

theorem slant_height_base_plane_angle (R : ℝ) (h : R = R) : angle_between_slant_height_and_base_plane R = Real.arcsin ((Real.sqrt 13 - 1) / 3) :=
by
  -- Here we assume that the mathematical conditions and transformations hold true.
  -- According to the solution steps provided:
  -- We found that γ = arcsin ((sqrt(13) - 1) / 3)
  sorry

end slant_height_base_plane_angle_l684_684892


namespace cannot_represent_sequence_l684_684923

theorem cannot_represent_sequence :
  ¬ (∀ n : ℕ+, (∃ k : ℕ+, n = 2*k - 1 ∧ (-1 : ℤ)^n + 1 = 2) ∨ (∃ k : ℕ+, n = 2*k ∧ (-1 : ℤ)^n + 1 = 0)) :=
by sorry

end cannot_represent_sequence_l684_684923


namespace volume_of_increased_box_l684_684232

theorem volume_of_increased_box {l w h : ℝ} (vol : l * w * h = 4860) (sa : l * w + w * h + l * h = 930) (sum_dim : l + w + h = 56) :
  (l + 2) * (w + 3) * (h + 1) = 5964 :=
by
  sorry

end volume_of_increased_box_l684_684232


namespace monotonic_increasing_interval_l684_684090

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin ((2 * Real.pi) / 3 - 2 * x)

theorem monotonic_increasing_interval :
  ∀ x, f x = 3 * Real.sin ((2 * Real.pi) / 3 - 2 * x) →
  (∃ x, ∀ x, (7 * Real.pi / 12) ≤ x ∧ x ≤ (13 * Real.pi / 12)) :=
sorry

end monotonic_increasing_interval_l684_684090


namespace captain_prize_amount_l684_684198

-- Definitions based on the conditions
def team_member_prize : ℕ := 200

def total_members : ℕ := 10
def other_members : ℕ := 9
def captain_more_than_average : ℕ := 90

-- The proof statement
theorem captain_prize_amount : 
  ∀ (x : ℕ), 
  x = 90 + (x + 9 * 200) / 10 → 
  x = 300 := 
by
  intro x
  -- introducing the given conditions
  have h0 : (x + 9 * 200) / 10 + 90 = x := 
    by assumption
  sorry

end captain_prize_amount_l684_684198


namespace rectangle_with_perpendicular_diagonals_is_square_l684_684183

-- Define rectangle and its properties
structure Rectangle where
  length : ℝ
  width : ℝ
  opposite_sides_equal : length = width

-- Define the condition that the diagonals of the rectangle are perpendicular
axiom perpendicular_diagonals {r : Rectangle} : r.length = r.width → True

-- Define the square property that a rectangle with all sides equal is a square
structure Square extends Rectangle where
  all_sides_equal : length = width

-- The main theorem to be proven
theorem rectangle_with_perpendicular_diagonals_is_square (r : Rectangle) (h : r.length = r.width) : Square := by
  sorry

end rectangle_with_perpendicular_diagonals_is_square_l684_684183


namespace units_digit_factorial_50_is_0_l684_684144

theorem units_digit_factorial_50_is_0 : (nat.factorial 50) % 10 = 0 := by
  sorry

end units_digit_factorial_50_is_0_l684_684144


namespace angle_bisectors_equal_length_l684_684536

open Triangle

-- Given two congruent triangles
variables {A B C A' B' C' : Point}
variables {ABC A'B'C' : Triangle}

-- Assume triangles are congruent
variables (h_cong : CongruentTriangle ABC A'B'C')

-- Let AD and A'D' be the angle bisectors of ∠BAC and ∠B'A'C'
variables {D D' : Point}
variables (h_bisector1 : IsAngleBisector (∠ B A C) D)
variables (h_bisector2 : IsAngleBisector (∠ B' A' C') D')

-- The proof statement asserting that the angle bisectors AD and A'D' are equal
theorem angle_bisectors_equal_length :
  (TriangleAngleBisector ABC h_bisector1).length = (TriangleAngleBisector A'B'C' h_bisector2).length :=
sorry

end angle_bisectors_equal_length_l684_684536


namespace triangle_sides_and_angles_l684_684101

theorem triangle_sides_and_angles (a : Real) (α β : Real) :
  (a ≥ 0) →
  let sides := [a, a + 1, a + 2]
  let angles := [α, β, 2 * α]
  (∀ s, s ∈ sides) → (∀ θ, θ ∈ angles) →
  a = 4 ∧ a + 1 = 5 ∧ a + 2 = 6 := 
by {
  sorry
}

end triangle_sides_and_angles_l684_684101


namespace necessary_but_not_sufficient_l684_684574

def condition1 (a b : ℝ) : Prop :=
  a > b

def statement (a b : ℝ) : Prop :=
  a > b + 1

theorem necessary_but_not_sufficient (a b : ℝ) (h : condition1 a b) : 
  (∀ a b : ℝ, statement a b → condition1 a b) ∧ ¬ (∀ a b : ℝ, condition1 a b → statement a b) :=
by 
  -- Proof skipped
  sorry

end necessary_but_not_sufficient_l684_684574


namespace fifteen_triangular_number_sum_fifteen_sixteen_triangular_numbers_l684_684092

theorem fifteen_triangular_number :
  let T := λ n : ℕ, n * (n + 1) / 2 in
  T 15 = 120 := 
by
  let T := λ n : ℕ, n * (n + 1) / 2
  sorry

theorem sum_fifteen_sixteen_triangular_numbers :
  let T := λ n : ℕ, n * (n + 1) / 2 in
  T 15 + T 16 = 256 := 
by
  let T := λ n : ℕ, n * (n + 1) / 2
  sorry

end fifteen_triangular_number_sum_fifteen_sixteen_triangular_numbers_l684_684092


namespace introspective_mult_introspective_prod_l684_684939

noncomputable theory
open Polynomial

variables (P Q : Polynomial ℤ) (k l : ℕ) (p : ℕ)

theorem introspective_mult (h1 : ∀ (X : ℕ), (P.eval (X ^ k)) ≡ (P.eval X) ^ k [MOD p])
  (h2 : ∀ (X : ℕ), (P.eval (X ^ l)) ≡ (P.eval X) ^ l [MOD p]) :
  ∀ (X : ℕ), P.eval (X ^ (k * l)) ≡ (P.eval X) ^ (k * l) [MOD p] :=
by sorry

theorem introspective_prod (h1 : ∀ (X : ℕ), (P.eval (X ^ k)) ≡ (P.eval X) ^ k [MOD p])
  (h2 : ∀ (X : ℕ), (Q.eval (X ^ k)) ≡ (Q.eval X) ^ k [MOD p]) :
  ∀ (X : ℕ), (P * Q).eval (X ^ k) ≡ (P * Q).eval X ^ k [MOD p] :=
by sorry

end introspective_mult_introspective_prod_l684_684939


namespace symmetric_circle_equation_l684_684744

theorem symmetric_circle_equation :
  ∀ C : Type,
    (∃ c : ℝ × ℝ, (x_1 + 2)^2 + (y_1 - 1)^2 = 1 ∧
      (c.1, c.2) = (2, -1)) →
    ((x - c.1)^2 + (y - c.2)^2 = 1) → (x - 2)^2 + (y + 1)^2 = 1 :=
by sorry

end symmetric_circle_equation_l684_684744


namespace integral_1_integral_2_integral_3_integral_4_integral_5_l684_684649

-- Definition using the conditions as standard integral formulas
def power_rule (n : ℝ) (C : ℝ) (x : ℝ) (h : n ≠ -1)  := (x^(n+1)) / (n+1) + C
def arcsin_integral (a : ℝ) (C : ℝ) (x : ℝ) := Math.sin⁻¹ (x/a) + C
def exponential_integral (a : ℝ) (C : ℝ) (x : ℝ) := a^x / (Real.log a) + C
def sqrt_integral' (C : ℝ) (u : ℝ) := (2 / 3) * u^(3/2) + C
def rational_integral (a : ℝ) (C : ℝ) (x : ℝ) := (1 / (2 * a)) * Real.log (Real.abs ((x - a) / (x + a))) + C

-- Proofs
theorem integral_1 (C : ℝ) : ∫ (fun x => 1/x^3) = fun x => -1/(2*x^2) + C := by sorry

theorem integral_2 (C : ℝ) : ∫ (fun x => 1/(Real.sqrt (2 - x^2))) = fun x => arcsin_integral (Real.sqrt 2) C x := by sorry

theorem integral_3 (C : ℝ) : ∫ (fun t => 3^t * 5^t) = fun t => exponential_integral 15 C t := by sorry

theorem integral_4 (C : ℝ) : ∫ (fun y => Real.sqrt (y + 1)) = fun y => sqrt_integral' C (y + 1) := by sorry

theorem integral_5 (C : ℝ) : ∫ (fun x => 1/(2 * x^2 - 6)) = fun x => (1 / (2 * Real.sqrt 3)) * Real.log (Real.abs ((x - Real.sqrt 3) / (x + Real.sqrt 3))) + C := by sorry

end integral_1_integral_2_integral_3_integral_4_integral_5_l684_684649


namespace sin_18_proof_l684_684451

def x : ℝ := Real.sin (Real.pi / 10)

theorem sin_18_proof : 4 * x^2 + 2 * x = 1 := by
  sorry

end sin_18_proof_l684_684451


namespace sqrt_15_minus_1_range_l684_684281

theorem sqrt_15_minus_1_range : (9 : ℝ) < 15 ∧ 15 < 16 → 2 < real.sqrt 15 - 1 ∧ real.sqrt 15 - 1 < 3 := 
by
  sorry

end sqrt_15_minus_1_range_l684_684281


namespace height_difference_correct_l684_684529

-- Definitions based on problem conditions
def diameter : ℝ := 15
def height_crate_A : ℝ := 15 * diameter
def height_increment : ℝ := sqrt(diameter^2 - (diameter / 2)^2)
def height_crate_B : ℝ := height_increment * 14 + diameter

-- Establishing a dummy equivalent for the given description of Crate C
def height_crate_C : ℝ := height_crate_B

def height_difference : ℝ := height_crate_A - min height_crate_B height_crate_C

-- Lean 4 statement for the math proof problem
theorem height_difference_correct : height_difference = 43.14 := by
  sorry

end height_difference_correct_l684_684529


namespace semicircle_length_EF_eq_sqrt3_AB_l684_684470

open EuclideanGeometry

theorem semicircle_length_EF_eq_sqrt3_AB
  (A B S C D E F : Point)
  (h1 : Diameter S A B)
  (h2 : OnSemicircle C S A D)
  (h3 : Angle C S D = 120)
  (h4 : IntersectsAt E A C B D)
  (h5 : IntersectsAt F A D B C) :
  Length E F = Real.sqrt 3 * Length A B := 
  sorry

end semicircle_length_EF_eq_sqrt3_AB_l684_684470


namespace part1_avg_score_paper2_part2_reasonable_estimate_part3_cong_winning_prob_l684_684863

-- Define constants and conditions
def total_score := 150
def est_difficulty_coeffs := [0.7, 0.64, 0.6, 0.6, 0.55]
def sample_avg_scores := [102, 99, 93, 93, 87]

-- Part 1: Estimate average score for test paper 2
theorem part1_avg_score_paper2 :
  (1 - est_difficulty_coeffs[1]) * total_score = 96 :=
by
  sorry

-- Part 2: Reasonableness of estimated difficulty coefficients
def calculate_actual_difficulty (avg_score : ℕ) : ℚ :=
  1 - (total_score - avg_score).toRat / total_score

def estimated_difficulties : List ℚ :=
  sample_avg_scores.map calculate_actual_difficulty

def S := (estimated_difficulties.zip est_difficulty_coeffs).foldr (λ ⟨real, est⟩ acc, acc + (real - est) ^ 2) 0 / 5

theorem part2_reasonable_estimate :
  S < 0.001 :=
by
  sorry

-- Part 3: Probability of Cong winning with a score of 3:1
def P1 := (2 / 3 : ℚ) * (1 / 2) * (2 / 3) * (1 / 2)
def P2 := (1 / 3 : ℚ) * (1 / 2) * (2 / 3) * (1 / 2)
def winning_probability := P1 + P2

theorem part3_cong_winning_prob :
  winning_probability = 1 / 6 :=
by
  sorry

end part1_avg_score_paper2_part2_reasonable_estimate_part3_cong_winning_prob_l684_684863


namespace arithmetic_mean_six_expressions_l684_684380

theorem arithmetic_mean_six_expressions (x : ℝ) :
  (x + 10 + 17 + 2 * x + 15 + 2 * x + 6 + 3 * x - 5) / 6 = 30 →
  x = 137 / 8 :=
by
  sorry

end arithmetic_mean_six_expressions_l684_684380


namespace min_value_of_expression_l684_684548

open Real

noncomputable def min_expression_value : ℝ :=
  let expr := λ (x y : ℝ), x^2 + y^2 - 8 * x + 6 * y + 25
  0

theorem min_value_of_expression : ∃ x y : ℝ, x = 4 ∧ y = -3 ∧ (x^2 + y^2 - 8 * x + 6 * y + 25) = min_expression_value :=
by {
  use [4, -3],
  split,
  { refl },
  split,
  { refl },
  sorry
}

end min_value_of_expression_l684_684548


namespace sum_reciprocals_l684_684909

noncomputable def sequence (n : ℕ) : ℕ := 
  if n = 1 then 2 
  else n * (n + 1)

theorem sum_reciprocals :
  (∑ n in Finset.range 2022, 1 / (sequence (n + 2) : ℚ)) = (2023 / 4048 : ℚ) :=
by
  sorry

end sum_reciprocals_l684_684909


namespace min_sum_ab_l684_684797

theorem min_sum_ab (a b : ℤ) (h : a * b = 196) : a + b = -197 :=
sorry

end min_sum_ab_l684_684797


namespace problem_statement_l684_684182

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def range_is_nonnegative (f : ℝ → ℝ) : Prop := ∀ y, ∃ x, f x = y ∧ y ≥ 0

def f1 (x : ℝ) : ℝ := |x|
def f2 (x : ℝ) : ℝ := x^3
def f3 (x : ℝ) : ℝ := 2^(|x|)
def f4 (x : ℝ) : ℝ := x^2 + |x|

theorem problem_statement :
  (is_even f1 ∧ range_is_nonnegative f1) ∧
  (is_even f4 ∧ range_is_nonnegative f4) :=
by
  sorry

end problem_statement_l684_684182


namespace x_2016_value_l684_684635

noncomputable def a : Real := 8
noncomputable def b : Real := 8
noncomputable def c : Real := 4
noncomputable def x₁ : Real := 4

def x_seq (n : Nat) : Real :=
  x₁ + n * 4

theorem x_2016_value : x_seq 2015 = 8064 :=
by
  sorry

end x_2016_value_l684_684635


namespace bake_sale_total_money_l684_684623

def dozens_to_pieces (dozens : Nat) : Nat :=
  dozens * 12

def total_money_raised
  (betty_chocolate_chip_dozen : Nat)
  (betty_oatmeal_raisin_dozen : Nat)
  (betty_brownies_dozen : Nat)
  (paige_sugar_cookies_dozen : Nat)
  (paige_blondies_dozen : Nat)
  (paige_cream_cheese_brownies_dozen : Nat)
  (price_per_cookie : Rat)
  (price_per_brownie_blondie : Rat) : Rat :=
let betty_cookies := dozens_to_pieces betty_chocolate_chip_dozen + dozens_to_pieces betty_oatmeal_raisin_dozen
let paige_cookies := dozens_to_pieces paige_sugar_cookies_dozen
let total_cookies := betty_cookies + paige_cookies
let betty_brownies := dozens_to_pieces betty_brownies_dozen
let paige_brownies_blondies := dozens_to_pieces paige_blondies_dozen + dozens_to_pieces paige_cream_cheese_brownies_dozen
let total_brownies_blondies := betty_brownies + paige_brownies_blondies
(total_cookies * price_per_cookie) + (total_brownies_blondies * price_per_brownie_blondie)

theorem bake_sale_total_money :
  total_money_raised 4 6 2 6 3 5 1 2 = 432 :=
by
  sorry

end bake_sale_total_money_l684_684623


namespace no_such_point_C_exists_l684_684776

noncomputable def point := (ℝ × ℝ)

def A : point := (-2, 0)
def B : point := (2, 0)

def distance (p q : point) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def triangle_perimeter (A B C : point) : ℝ :=
  distance A B + distance A C + distance B C

theorem no_such_point_C_exists :
  ¬ ∃ C : point, triangle_perimeter A B C = 8 :=
by
  sorry

end no_such_point_C_exists_l684_684776


namespace limit_of_difference_l684_684456

variable {ℝ : Type} [RealField ℝ]

def differentiable_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ (f' : ℝ), ∀ (ε > 0), ∃ (δ > 0), ∀ x, abs (x - x₀) < δ → abs ((f x - f x₀) / (x - x₀) - f') < ε
  
theorem limit_of_difference (f : ℝ → ℝ) (x₀ : ℝ) (h : differentiable_at f x₀) :
  tendsto (λ x, (f(x₀ + x) - f(x₀ - 3 * x)) / x) (nhds 0) (nhds (4 * f'(x₀))) := 
sorry

end limit_of_difference_l684_684456


namespace find_number_l684_684659

def perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def sum_of_perfect_square_divisors_eq (N : ℕ) (sum : ℕ) : Prop :=
  sum = ∑ d in (Finset.filter perfect_square (Finset.divisors N)), d

theorem find_number (N : ℕ) : 
  (1 ∣ N) ∧ (4 ∣ N) ∧ (16 ∣ N) ∧ sum_of_perfect_square_divisors_eq N 21 → N = 16 :=
by
  sorry

end find_number_l684_684659


namespace charlie_banana_consumption_l684_684261

theorem charlie_banana_consumption :
  (∃ b : ℚ, (∃ (n : ℕ) (d : ℚ), n = 7 ∧ d = 4 ∧ 
    ∑ i in Finset.range (n - 1), b + i * d + b + (n - 1) * d / 2 + (b + (b + (n - 1) * d)) / 2).toReal = 150) 
    → ((b + 24 : ℚ) = (33 + 4 / 7 : ℚ)) :=
by
  sorry

end charlie_banana_consumption_l684_684261


namespace exists_circle_through_M_and_intersects_exists_tangent_line_AB_l684_684680

def point_M : ℝ × ℝ := (2, -2)
def circle_O : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 3
def circle_1 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + 3 * x = 0

def circle_through_M (x y : ℝ) : Prop := 3*x^2 + 3*y^2 - 5*x - 14 = 0
def line_AB (x y : ℝ) : Prop := 2*x - 2*y = 3

theorem exists_circle_through_M_and_intersects : 
  (∃ (x y : ℝ), point_M = (x, y) ∧ circle_O x y ∧ circle_1 x y) → 
  (∃ (x y : ℝ), circle_through_M x y) :=
by {
  sorry
}

theorem exists_tangent_line_AB : 
  (∃ (x y : ℝ), point_M = (x, y) ∧ circle_O x y) → 
  (∃ (x y : ℝ), line_AB x y) :=
by {
  sorry
}

end exists_circle_through_M_and_intersects_exists_tangent_line_AB_l684_684680


namespace fixed_point_of_line_l684_684707

theorem fixed_point_of_line (k : ℝ) : 
  ∀ (k : ℝ), ∃ x y : ℝ, (1 + 4 * k) * x - (2 - 3 * k) * y + (2 - 3 * k) = 0 ∧ (x, y) = (0, 1) :=
by
  assume k : ℝ
  use 0, 1
  have h1 : (1 + 4 * k) * 0 - (2 - 3 * k) * 1 + (2 - 3 * k) = 0 :=
    by calc
      (1 + 4 * k) * 0 - (2 - 3 * k) * 1 + (2 - 3 * k)
        = 0 - (2 - 3 * k) * 1 + (2 - 3 * k) : by ring
    ... = - (2 - 3 * k) + (2 - 3 * k) : by ring
    ... = 0 : by ring
  exact ⟨h1, rfl⟩

end fixed_point_of_line_l684_684707


namespace max_value_y_range_k_one_root_range_a_l684_684705

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def y (a : ℝ) (x : ℝ) : ℝ := f a x * g x

theorem max_value_y (a : ℝ) (h : a = -1) : 
    ∃ x ∈ Set.Icc (-1 : ℝ) 2, y a x = 3 * Real.exp 2 := 
sorry

theorem range_k_one_root (a : ℝ) (h : a = -1) : 
    {k : ℝ | ∃ x, f a x = k * g x ∧ ( ∃! x, f a x = k * g x )} = 
    {k : ℝ | k > 3 / (Real.exp 2) ∨ (0 < k ∧ k < (1 / Real.exp 1)) } := 
sorry

theorem range_a (h : ∀ x1 x2 ∈ Set.Icc (0 : ℝ) 2, x1 ≠ x2 → |f x1 - f x2| < |g x1 - g x2|) : 
    ∀ a, -1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2 :=
sorry

end max_value_y_range_k_one_root_range_a_l684_684705


namespace domain_of_f_l684_684121

noncomputable def f (x : ℝ) := Real.log 2 (Real.log 5 (Real.log 3 x))

theorem domain_of_f : ∀ x : ℝ, 
  (0 < Real.log 3 x) → 
  (0 < Real.log 5 (Real.log 3 x)) → 
  (0 < Real.log 2 (Real.log 5 (Real.log 3 x))) → 
  (3 < x) :=
by
  sorry

end domain_of_f_l684_684121


namespace call_cost_inequalities_min_call_cost_correct_l684_684949

noncomputable def call_cost_before (x : ℝ) : ℝ :=
  if x ≤ 3 then 0.2 else 0.4

noncomputable def call_cost_after (x : ℝ) : ℝ :=
  if x ≤ 3 then 0.2
  else if x ≤ 4 then 0.2 + 0.1 * (x - 3)
  else 0.3 + 0.1 * (x - 4)

theorem call_cost_inequalities : 
  (call_cost_before 4 = 0.4 ∧ call_cost_after 4 = 0.3) ∧
  (call_cost_before 4.3 = 0.4 ∧ call_cost_after 4.3 = 0.4) ∧
  (call_cost_before 5.8 = 0.4 ∧ call_cost_after 5.8 = 0.5) ∧
  (∀ x, (0 < x ∧ x ≤ 3) ∨ x > 4 → call_cost_before x ≤ call_cost_after x) :=
by
  sorry

noncomputable def min_call_cost_plan (m : ℝ) (n : ℕ) : ℝ :=
  if 3 * n - 1 < m ∧ m ≤ 3 * n then 0.2 * n
  else if 3 * n < m ∧ m ≤ 3 * n + 1 then 0.2 * n + 0.1
  else if 3 * n + 1 < m ∧ m ≤ 3 * n + 2 then 0.2 * n + 0.2
  else 0.0  -- Fallback, though not necessary as per the conditions

theorem min_call_cost_correct (m : ℝ) (n : ℕ) (h : m > 5) :
  (3 * n - 1 < m ∧ m ≤ 3 * n → min_call_cost_plan m n = 0.2 * n) ∧
  (3 * n < m ∧ m ≤ 3 * n + 1 → min_call_cost_plan m n = 0.2 * n + 0.1) ∧
  (3 * n + 1 < m ∧ m ≤ 3 * n + 2 → min_call_cost_plan m n = 0.2 * n + 0.2) :=
by
  sorry

end call_cost_inequalities_min_call_cost_correct_l684_684949


namespace john_finishes_task_at_six_pm_l684_684425

noncomputable def task_finish_time (start_time : Time) (finish_time_third : Time) : Time :=
  let total_duration_three_tasks := Time.diff finish_time_third start_time 
  let duration_one_task := total_duration_three_tasks / 3 
  Time.add finish_time_third duration_one_task

theorem john_finishes_task_at_six_pm 
  (start_time : Time := Time.mk 14 0)
  (finish_time_third : Time := Time.mk 17 0)
  (finish_time_fourth : Time := Time.mk 18 0) :
  task_finish_time start_time finish_time_third = finish_time_fourth := 
sorry

end john_finishes_task_at_six_pm_l684_684425


namespace find_value_5a1_a7_l684_684328

-- Define the arithmetic sequence with common difference and sum of terms
def arithmetic_sequence (a d : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the sequence
def sum_terms (a d : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  S 3 = 3 * a 0 + 3 * d

-- Define the specific problem statement
theorem find_value_5a1_a7 (a d S : ℕ → ℝ) (h_seq : arithmetic_sequence a d 1)
  (h_sum : sum_terms a d S) (h_S3 : S 3 = 6) : 
  5 * a 0 + a 6 = 12 := 
by 
  sorry

end find_value_5a1_a7_l684_684328


namespace why_build_offices_l684_684289

structure Company where
  name : String
  hasSkillfulEmployees : Prop
  uniqueComfortableWorkEnvironment : Prop
  integratedWorkLeisureSpaces : Prop
  reducedEmployeeStress : Prop
  flexibleWorkSchedules : Prop
  increasesProfit : Prop

theorem why_build_offices (goog_fb : Company)
  (h1 : goog_fb.hasSkillfulEmployees)
  (h2 : goog_fb.uniqueComfortableWorkEnvironment)
  (h3 : goog_fb.integratedWorkLeisureSpaces)
  (h4 : goog_fb.reducedEmployeeStress)
  (h5 : goog_fb.flexibleWorkSchedules) :
  goog_fb.increasesProfit := 
sorry

end why_build_offices_l684_684289


namespace line_equation_length_AB_l684_684708

variables {x y : ℝ}

-- Conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def point_M (x y : ℝ) : Prop := x = 2 ∧ y = 1
def midpoint (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 1
def line_through_point (mx my : ℝ) (a b c : ℝ) : Prop := a * mx + b * my + c = 0

-- Questions translated to lean
theorem line_equation : 
  ∃ a b c : ℝ, ∃ x1 y1 x2 y2 : ℝ,
  (parabola x1 y1 ∧ parabola x2 y2) ∧ 
  (midpoint x1 y1 x2 y2) ∧ 
  (line_through_point 2 1 a b c) ∧ 
  (a = 2) ∧ (b = -1) ∧ (c = -3) := 
sorry

theorem length_AB : 
  ∃ x1 y1 x2 y2 : ℝ,
  (parabola x1 y1 ∧ parabola x2 y2) ∧ 
  (midpoint x1 y1 x2 y2) ∧ 
  dist (x1, y1) (x2, y2) = sqrt 35 :=
sorry

end line_equation_length_AB_l684_684708


namespace positive_integer_solutions_l684_684847

theorem positive_integer_solutions
  (x : ℤ) :
  (5 + 3 * x < 13) ∧ ((x + 2) / 3 - (x - 1) / 2 <= 2) →
  (x = 1 ∨ x = 2) :=
by
  sorry

end positive_integer_solutions_l684_684847


namespace min_small_bottles_needed_l684_684237

theorem min_small_bottles_needed (small_capacity large_capacity : ℕ) 
    (h_small_capacity : small_capacity = 35) (h_large_capacity : large_capacity = 500) : 
    ∃ n, n = 15 ∧ large_capacity <= n * small_capacity :=
by 
  sorry

end min_small_bottles_needed_l684_684237


namespace solve_for_t_l684_684346

variable (A P0 r t : ℝ)

theorem solve_for_t (h : A = P0 * Real.exp (r * t)) : t = (Real.log (A / P0)) / r :=
  by
  sorry

end solve_for_t_l684_684346


namespace soap_box_length_l684_684958

def VolumeOfEachSoapBox (L : ℝ) := 30 * L
def VolumeOfCarton := 25 * 42 * 60
def MaximumSoapBoxes := 300

theorem soap_box_length :
  ∀ L : ℝ,
  MaximumSoapBoxes * VolumeOfEachSoapBox L = VolumeOfCarton → 
  L = 7 :=
by
  intros L h
  sorry

end soap_box_length_l684_684958


namespace union_of_A_and_B_l684_684746

open Set

variable (A B : Set ℤ)

theorem union_of_A_and_B (hA : A = {0, 1}) (hB : B = {0, -1}) : A ∪ B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l684_684746


namespace units_digit_50_factorial_l684_684160

theorem units_digit_50_factorial : (nat.factorial 50) % 10 = 0 :=
by
  sorry

end units_digit_50_factorial_l684_684160


namespace ways_to_go_home_via_library_l684_684091

def ways_from_school_to_library := 2
def ways_from_library_to_home := 3

theorem ways_to_go_home_via_library : 
  ways_from_school_to_library * ways_from_library_to_home = 6 :=
by 
  sorry

end ways_to_go_home_via_library_l684_684091


namespace successful_balls_count_l684_684822

-- Define the initial conditions
def totalBalls : Nat := 100
def ballsWithHoles : Nat := 40 * totalBalls / 100
def ballsRemainAfterHoles : Nat := totalBalls - ballsWithHoles
def overinflatedBalls : Nat := 20 * ballsRemainAfterHoles / 100
def successfullyInflatedBalls : Nat := ballsRemainAfterHoles - overinflatedBalls

-- The theorem we are proving
theorem successful_balls_count : successfullyInflatedBalls = 48 := by
  unfold totalBalls ballsWithHoles ballsRemainAfterHoles overinflatedBalls successfullyInflatedBalls
  rfl

end successful_balls_count_l684_684822


namespace Y_minus_X_eq_92_l684_684433

def arithmetic_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def X : ℕ := arithmetic_sum 10 2 46
def Y : ℕ := arithmetic_sum 12 2 46

theorem Y_minus_X_eq_92 : Y - X = 92 := by
  sorry

end Y_minus_X_eq_92_l684_684433


namespace trapezoid_area_l684_684759

-- Geometry setup
variable (outer_area : ℝ) (inner_height_ratio : ℝ)

-- Conditions
def outer_triangle_area := outer_area = 36
def inner_height_to_outer_height := inner_height_ratio = 2 / 3

-- Conclusion: Area of one trapezoid
theorem trapezoid_area (outer_area inner_height_ratio : ℝ) 
  (h_outer : outer_triangle_area outer_area) 
  (h_inner : inner_height_to_outer_height inner_height_ratio) : 
  (outer_area - 16 * Real.sqrt 3) / 3 = (36 - 16 * Real.sqrt 3) / 3 := 
sorry

end trapezoid_area_l684_684759


namespace square_difference_l684_684334

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 36) 
  (h₂ : x * y = 8) : 
  (x - y)^2 = 4 :=
by
  sorry

end square_difference_l684_684334


namespace g_min_value_l684_684710

-- Definition of the function f(x) = x^α where it passes through point (2, 1/2)
def f (x : ℝ) (α : ℝ) : ℝ := x^α

-- Given f(2) = 1/2 to find α, we state the condition
lemma α_eq : ∃ α : ℝ, (2:ℝ) ^ α = 1 / 2 :=
by { use -1, norm_num }

-- Definition of the function g(x) = (x-2)f(x)
def g (x : ℝ) : ℝ := (x - 2) / x

-- Theorem to state the minimum value of g(x) on the interval [1/2, 1] is -3
theorem g_min_value : ∀ x ∈ set.Icc (1 / 2 : ℝ) 1, g x >= -3 := 
by sorry

-- The minimum value is actually attained at x = 1/2
example : g (1 / 2) = -3 := by norm_num

end g_min_value_l684_684710


namespace there_exists_l_l684_684808

theorem there_exists_l (m n : ℕ) (h1 : m ≠ 0) (h2 : n ≠ 0) 
  (h3 : ∀ k : ℕ, 0 < k → Nat.gcd (17 * k - 1) m = Nat.gcd (17 * k - 1) n) :
  ∃ l : ℤ, m = (17 : ℕ) ^ l.natAbs * n := 
sorry

end there_exists_l_l684_684808


namespace count_1_left_of_2_and_3_l684_684877

theorem count_1_left_of_2_and_3 : 
  let digits := {1, 2, 3, 4, 5, 6}
  in ∃! count : ℕ, 
    (∀ perm : list ℕ, perm.perm digits.toList → 
       perm.nodup ∧ perm.length = 6 ∧ 
       (∃ idx1 idx2 idx3, idx1 < idx2 ∧ idx1 < idx3 ∧ 
          perm.nth idx1 = some 1 ∧ perm.nth idx2 = some 2 ∧ perm.nth idx3 = some 3)) → 
    count = 240 :=
begin
  let digits := {1, 2, 3, 4, 5, 6},
  let count := 240,
  have : ∀ perm : list ℕ, perm.perm digits.toList → 
         perm.nodup ∧ perm.length = 6 ∧ 
         (∃ idx1 idx2 idx3, idx1 < idx2 ∧ idx1 < idx3 ∧ 
            perm.nth idx1 = some 1 ∧ perm.nth idx2 = some 2 ∧ perm.nth idx3 = some 3),
  { sorry },
  use count,
  split,
  { exact 240 },
  { sorry }
end

end count_1_left_of_2_and_3_l684_684877


namespace number_of_non_consecutive_triples_l684_684129

-- Define a set from 1 to 10
def numbers : Finset ℕ := Finset.range 10

-- Define a predicate for non-consecutive numbers
def nonConsecutive (x y z : ℕ) : Prop := (x < y) ∧ (y < z) ∧ (x + 1 < y) ∧ (y + 1 < z)

-- Define the set of all triples (x, y, z) of non-consecutive numbers from the set
def validTriples : Finset (ℕ × ℕ × ℕ) := 
  numbers.product (numbers.product numbers) |>.filter (λ (t : ℕ × (ℕ × ℕ)), 
    nonConsecutive t.1 t.2.1 t.2.2)

-- Theorem statement
theorem number_of_non_consecutive_triples : validTriples.card = 56 :=
by
  sorry -- Proof goes here

end number_of_non_consecutive_triples_l684_684129


namespace length_of_AC_l684_684390

-- Definitions of the given conditions
variable (A B C D E : Type)
variable (AB AC BC AD AE : ℝ)
variable (BD DE EC : ℝ)
variable (triangle_ABC : is_triangle A B C)
variable (AD_bisects_BAC : bisects A D B C)
variable (AE_bisects_BAD : bisects A E B D)
variable (BD_length : BD = 4)
variable (DE_length : DE = 2)
variable (EC_length : EC = 9)

-- The theorem to prove the length of AC
theorem length_of_AC :
  AC = 6 * Real.sqrt 6 :=
by
  sorry

end length_of_AC_l684_684390


namespace sum_of_angles_in_pentagon_l684_684684

-- Define the given conditions
def angle_A : ℝ := 30
def angle_AFG : ℝ := 75 / 2  -- Since we know it is 75°
def angle_AGF : ℝ := angle_AFG
def angle_BFD : ℝ := 180 - angle_AFG
def angle_GAF : ℝ := 180 - 2 * angle_AFG  -- Derived from the previous conditions

-- Statement to be proven
theorem sum_of_angles_in_pentagon :
  (angle_AFG = angle_AGF ∧ 
   angle_AFG + angle_AGF + angle_GAF = 180 ∧ 
   angle_BFD + angle_AFG = 180 ∧ 
   angle_A = 30
  ) → (angle_BFD + 2 * angle_AFG = 75) :=
  by
  sorry

end sum_of_angles_in_pentagon_l684_684684


namespace isosceles_triangle_ABC_l684_684448

open EuclideanGeometry
open Classical
noncomputable theory

theorem isosceles_triangle_ABC (A B C E D : Point) 
    (h1 : AngleBisector A B C E) 
    (h2 : AngleBisector C D A B) 
    (h3 : ∠ B E D = 2 * ∠ A E D) 
    (h4 : ∠ B D E = 2 * ∠ E D C) :
  isIsosceles A B C := 
sorry

end isosceles_triangle_ABC_l684_684448


namespace volume_of_T_l684_684105

/-- The set T consisting of all points (x, y, z) such that |x| + |y| ≤ 2, |x| + |z| ≤ 2,
    and |y| + |z| ≤ 2, has a volume of 16√3 / 3. -/
theorem volume_of_T : 
  ∃ (T : set (ℝ × ℝ × ℝ)), 
  (∀ (x y z : ℝ), (x, y, z) ∈ T → |x| + |y| ≤ 2 ∧ |x| + |z| ≤ 2 ∧ |y| + |z| ≤ 2) ∧
  (volume T = 16 * real.sqrt 3 / 3) :=
sorry

end volume_of_T_l684_684105


namespace polynomial_root_sum_square_l684_684014

theorem polynomial_root_sum_square (a b c d : ℝ) 
  (h : ∀ x : ℝ, Polynomial.eval x (Polynomial.ofReal [7, -35, 50, -24, 1]) = 0 → 
    x = a ∨ x = b ∨ x = c ∨ x = d) : 
  (a + b + c)^2 + (b + c + d)^2 + (c + d + a)^2 + (d + a + b)^2 = 2104 := 
by 
  sorry

end polynomial_root_sum_square_l684_684014


namespace part1_condition1_part1_condition2_part1_condition3_part2_l684_684999

theorem part1_condition1 (a b : ℝ) (A B : ℝ) (h : a * sin B - sqrt 3 * b * cos A = 0) (hA : A ∈ (0 : ℝ)..π) :
  A = π / 3 :=
sorry

theorem part1_condition2 (a b c : ℝ) (A B C : ℝ) (h : (sin B - sin C)^2 = sin^2 A - sin B * sin C) :
  A = π / 3 :=
sorry

theorem part1_condition3 (a b c : ℝ) (A B C : ℝ) (h : 2 * cos A * (c * cos B + b * cos C) = a) :
  A = π / 3 :=
sorry

theorem part2 (a b c A B C : ℝ) (ha : a = sqrt 3) (hC : sin C = 2 * sin B) (hA : A = π / 3) :
  let S := (1/2) * b * c * sin A in S = sqrt 3 / 2 :=
sorry

end part1_condition1_part1_condition2_part1_condition3_part2_l684_684999


namespace shaded_area_percentage_l684_684920

theorem shaded_area_percentage (side : ℕ) (total_shaded_area : ℕ) (expected_percentage : ℕ)
  (h1 : side = 5)
  (h2 : total_shaded_area = 15)
  (h3 : expected_percentage = 60) :
  ((total_shaded_area : ℚ) / (side * side) * 100) = expected_percentage :=
by
  sorry

end shaded_area_percentage_l684_684920


namespace common_chord_of_circles_l684_684500

theorem common_chord_of_circles (r : ℝ) (h : r > 0) :
    ∀ (ρ θ : ℝ), 
    (ρ = r ∨ ρ = -2 * r * sin(θ + π / 4)) → 
    (√2 * ρ * (sin θ + cos θ) = r) := 
sorry

end common_chord_of_circles_l684_684500


namespace project_completion_time_l684_684583

/--
A can complete a project in 20 days, B can complete the same project in 30 days, and C can complete the project in 40 days.
A quits 5 days before the project is completed while C quits 3 days before the project is completed.
Prove that the total number of days to complete the project is 13.
-/
theorem project_completion_time :
  ∃ (D : ℕ), (∀ (A B C : ℕ), A = 20 ∧ B = 30 ∧ C = 40 → (D > 5) ∧ (∃ (work_per_day : ℚ), work_per_day = 13 / 120 ∧ ((work_per_day * (D - 5) + 7 / 120 * 2 + 4 / 120 * 2) = 1)) → D = 13) :=
begin
  use 13,
  sorry
end

end project_completion_time_l684_684583


namespace complex_expression_l684_684254

theorem complex_expression : 
  ( (1 + Complex.i) / (1 - Complex.i) )^2017 + ( (1 - Complex.i) / (1 + Complex.i) )^2017 = 0 := 
by 
  have h1 : (1 + Complex.i) / (1 - Complex.i) = Complex.i := 
    sorry 
  have h2 : (1 - Complex.i) / (1 + Complex.i) = -Complex.i := 
    sorry 
  have h3 : (Complex.i ^ 4) = 1 := 
    sorry 
  sorry

end complex_expression_l684_684254


namespace maximum_area_of_garden_l684_684489

def length (w : ℝ) : ℝ := 400 - 2 * w
def area (w : ℝ) : ℝ := w * length w

theorem maximum_area_of_garden : ∀ w : ℝ, w = 100 → area w = 20000 := 
by 
  intro w
  intro hw
  rw [hw, length, area]
  simp
  sorry

end maximum_area_of_garden_l684_684489


namespace sum_of_squares_of_roots_lower_bound_l684_684915

theorem sum_of_squares_of_roots_lower_bound {n : ℕ} {a : ℝ} (hn : n ≥ 1) :
  (∑ i in (finset.range n), (λ r, r^2)) ≥ -4 * a * a + 2 * a :=
  sorry

end sum_of_squares_of_roots_lower_bound_l684_684915


namespace log_expression_simplifies_l684_684933

theorem log_expression_simplifies (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (log a b + log b a + 2) * (log a b - log (a * b) b) * log b a - 1 = log a b :=
by sorry

end log_expression_simplifies_l684_684933


namespace tan_theta_is_minus_3_l684_684678

-- Define the line whose equation is given
def line1 (x y : ℝ) : Prop := x - 3 * y + 1 = 0

-- Define the inclination angle θ and its tangent
def theta_inclination (θ : ℝ) : Prop := ∀ m : ℝ, (m = -3 ↔ line1 (m * tan θ) 1) 

-- Prove that the tangent of θ is -3
theorem tan_theta_is_minus_3 : ∃ θ : ℝ, theta_inclination θ ∧ tan θ = -3 :=
sorry

end tan_theta_is_minus_3_l684_684678


namespace shortest_distance_ln_curve_to_line_l684_684656

variable (x : ℝ) (point : ℝ × ℝ) (A B C : ℝ)

def curve_eq (x : ℝ) : ℝ := Real.log x

def distance_formula (point : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  abs (A * point.1 + B * point.2 + C) / Real.sqrt (A^2 + B^2)

theorem shortest_distance_ln_curve_to_line : 
  ∃ (point : ℝ × ℝ), point = (1, 0) ∧ distance_formula (1, 0) (-1) 1 (-1) = Real.sqrt 2 :=
by
  let point := (1, 0)
  have h1 : point = (1, 0), from rfl
  rw [distance_formula, h1]
  have h2 : abs ((-1) * 1 + 1 * 0 + (-1)) = abs (-2), by simp
  have h3 : abs (-2) = 2, from abs_neg 2
  rw [h2, h3]
  have h4 : (-1)^2 + 1^2 = 2, by simp
  rw [h4, Real.sqrt_two_mul_two, div_self, one_mul]
  exact Real.sqrt_two_ne_zero

end shortest_distance_ln_curve_to_line_l684_684656


namespace exists_fixed_point_l684_684683

noncomputable def ellipse := (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (P : ℝ × ℝ) 
  (h3 : P = (1, sqrt 2 / 2))
  (hasi_sequence : sqrt 2 * (dist P ⟨0, 0⟩) = dist ⟨0, 0⟩ ⟨a, 0⟩ 
  + sqrt 2 * (dist P ⟨a, 0⟩)) : 
  a = sqrt 2 ∧ b = 1 ∧ c = 1 

theorem exists_fixed_point (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (P : ℝ × ℝ) (h3 : P = (1, sqrt 2 / 2))
  (hasi_sequence : sqrt 2 * (dist P ⟨0, 0⟩) = dist ⟨0, 0⟩ ⟨a, 0⟩ 
  + sqrt 2 * (dist P ⟨a, 0⟩)) 
  (h_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) : 
  ∃ Q : ℝ × ℝ, Q = (5/4, 0) := 
begin 
  sorry 
end 

end exists_fixed_point_l684_684683


namespace prob_both_white_is_two_fifth_prob_one_white_one_black_is_eight_fifteenth_l684_684106

-- Conditions
def total_balls : ℕ := 6
def white_balls : ℕ := 4
def black_balls : ℕ := 2

-- Events
def total_outcomes : ℕ := (total_balls * (total_balls - 1)) / 2
def white_white_outcomes : ℕ := (white_balls * (white_balls - 1)) / 2
def white_black_outcomes : ℕ := white_balls * black_balls

-- Probabilities
def prob_both_white : ℚ := (white_white_outcomes : ℚ) / (total_outcomes : ℚ)
def prob_one_white_one_black : ℚ := (white_black_outcomes : ℚ) / (total_outcomes : ℚ)

theorem prob_both_white_is_two_fifth 
  (h1 : total_balls = 6)
  (h2 : white_balls = 4)
  (h3 : black_balls = 2)
  (h4 : white_white_outcomes = 6)
  (h5 : total_outcomes = 15) :
  prob_both_white = 2 / 5 := by
  sorry

theorem prob_one_white_one_black_is_eight_fifteenth
  (h1 : total_balls = 6)
  (h2 : white_balls = 4)
  (h3 : black_balls = 2)
  (h4 : white_black_outcomes = 8)
  (h5 : total_outcomes = 15) :
  prob_one_white_one_black = 8 / 15 := by
  sorry

end prob_both_white_is_two_fifth_prob_one_white_one_black_is_eight_fifteenth_l684_684106


namespace lines_parallel_value_of_a_l684_684379

theorem lines_parallel_value_of_a (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 → (∃ m₁ : ℝ, y = -a / 3 * x + m₁))
  → (∀ x y : ℝ, 2 * x + (a + 1) * y + 1 = 0 → (∃ m₂ : ℝ, y = -2 / (a + 1) * x + m₂))
  → a = -3 :=
by 
  intros h1 h2
  have h3 : ∃ m₁ : ℝ, ∃ m₂ : ℝ, m₁ = m₂ :=
    by 
      obtain ⟨m₁, hm₁⟩ := h1 0 0 (by simp)
      obtain ⟨m₂, hm₂⟩ := h2 0 0 (by simp)
      use [m₁, m₂]
      sorry
  sorry

end lines_parallel_value_of_a_l684_684379


namespace tank_capacity_l684_684589

-- Define the conditions given in the problem.
def tank_full_capacity (x : ℝ) : Prop :=
  (0.25 * x = 60) ∧ (0.15 * x = 36)

-- State the theorem that needs to be proved.
theorem tank_capacity : ∃ x : ℝ, tank_full_capacity x ∧ x = 240 := 
by 
  sorry

end tank_capacity_l684_684589


namespace volume_of_T_l684_684104

/-- The set T consisting of all points (x, y, z) such that |x| + |y| ≤ 2, |x| + |z| ≤ 2,
    and |y| + |z| ≤ 2, has a volume of 16√3 / 3. -/
theorem volume_of_T : 
  ∃ (T : set (ℝ × ℝ × ℝ)), 
  (∀ (x y z : ℝ), (x, y, z) ∈ T → |x| + |y| ≤ 2 ∧ |x| + |z| ≤ 2 ∧ |y| + |z| ≤ 2) ∧
  (volume T = 16 * real.sqrt 3 / 3) :=
sorry

end volume_of_T_l684_684104


namespace spinner_probability_l684_684592

theorem spinner_probability :
  let p_A := (1 / 4)
  let p_B := (1 / 3)
  let p_C := (5 / 12)
  let p_D := 1 - (p_A + p_B + p_C)
  p_D = 0 :=
by
  sorry

end spinner_probability_l684_684592


namespace eval_power_imaginary_unit_l684_684645

noncomputable def i : ℂ := Complex.I

theorem eval_power_imaginary_unit :
  i^20 + i^39 = 1 - i := by
  -- Skipping the proof itself, indicating it with "sorry"
  sorry

end eval_power_imaginary_unit_l684_684645


namespace number_of_ones_in_sum_g_subsets_l684_684720

-- Define bitwise XOR
def bitwise_xor (x y : ℕ) : ℕ := Nat.bitwise xor x y

-- Define f(S) as the XOR of all elements in S
def f (S : List ℕ) : ℕ := S.foldr bitwise_xor 0

-- Define g(S) as the number of divisors of f(S) that are at most 2018
-- but greater than or equal to the largest element in S.
def g (S : List ℕ) : ℕ :=
  if S = [] then 2018
  else
    let max_element := List.maximum' S
    let f_S := f S
    List.length (List.filter (λ d, d ≥ max_element ∧ d ≤ 2018) (List.Ico 1 (2018 + 1)).filter (λ d, f_S % d = 0))

-- Sum g(S) over all subsets S of the set {1, 2, ... ,2018}
def sum_g {α} [DecidableEq α] (l : List α) : ℕ :=
  List.foldr (λ x acc, acc + g x) 0 (List.subsets l)

-- Compute the number of 1s in the binary representation of a number
def count_ones (n : ℕ) : ℕ :=
  Integer.to_biginteger n.flash.to_nat.toBinary.count_ones

-- The statement to prove
theorem number_of_ones_in_sum_g_subsets : count_ones (sum_g (List.range 2018)) = 10 := sorry

end number_of_ones_in_sum_g_subsets_l684_684720


namespace triangles_congruent_l684_684411

variable (A B C A' B' C' O1 O2 O3 I1 I2 I3 : Point)
variable (ABC : Triangle)
variable (AB'C' CA'B B'CA' : Triangle)

-- Conditions
def midpoints_of_triangle (ABC : Triangle) :=
  midpoint (side BC ABC) = A' ∧ midpoint (side CA ABC) = B' ∧ midpoint (side AB ABC) = C'

def circumcenters_of_subtriangles :=
  circumcenter (triangle AB'C') = O1 ∧ circumcenter (triangle CA'B) = O2 ∧ circumcenter (triangle B'CA') = O3

def incenters_of_subtriangles :=
  incenter (triangle AB'C') = I1 ∧ incenter (triangle CA'B) = I2 ∧ incenter (triangle B'CA') = I3

theorem triangles_congruent (h1 : midpoints_of_triangle ABC)
                            (h2 : circumcenters_of_subtriangles O1 O2 O3)
                            (h3 : incenters_of_subtriangles I1 I2 I3) :
   congruent (triangle O1 O2 O3) (triangle I1 I2 I3) := 
sorry

end triangles_congruent_l684_684411


namespace companies_increase_profitability_l684_684291

-- Define the three main arguments as conditions
def increased_retention_and_attraction : Prop := 
  ∀ (company : Type), ∃ (environment : Type),
    (company = Google ∨ company = Facebook) → 
    environment provides_comfortable_and_flexible_workplace → 
    increases_profitability company

def enhanced_productivity_and_creativity : Prop := 
  ∀ (company : Type), ∃ (environment : Type),
    (company = Google ∨ company = Facebook) → 
    environment provides_work_and_leisure_integration → 
    increases_profitability company

def better_work_life_integration : Prop := 
  ∀ (company : Type), ∃ (environment : Type),
    (company = Google ∨ company = Facebook) → 
    environment provides_work_life_integration → 
    increases_profitability company

-- Define the overall theorem we need to prove
theorem companies_increase_profitability 
  (company : Type)
  (h1 : increased_retention_and_attraction)
  (h2 : enhanced_productivity_and_creativity)
  (h3 : better_work_life_integration) 
: ∃ (environment : Type), 
    (company = Google ∨ company = Facebook) → 
    environment enhances_work_environment → 
    increases_profitability company :=
by
  sorry

end companies_increase_profitability_l684_684291


namespace max_sum_remaining_numbers_l684_684479

theorem max_sum_remaining_numbers : 
  ∃ (S : ℕ), ∀ (erased_numbers : set ℕ), 
  (erased_numbers ⊆ {n | 7 ≤ n ∧ n ≤ 17}) ∧ erased_numbers ≠ ∅ → 
  let remaining_numbers := {n | 7 ≤ n ∧ n ≤ 17} \ erased_numbers in 
  S = ∑ n in remaining_numbers, id n ∧ 
  ¬ ∃ (grouped_numbers : set (set ℕ)), 
  grouped_numbers ⊆ powerset remaining_numbers ∧ 
  (∀ G ∈ grouped_numbers, ∃ x : ℕ, ∑ n in G, id n = x) ∧ 
  (∀ (G1 G2 : set ℕ), G1 ≠ G2 → G1 ∩ G2 = ∅) ∧ 
  (⋃₀ grouped_numbers) = remaining_numbers 
  ∧ S = 121 :=
sorry

end max_sum_remaining_numbers_l684_684479


namespace largest_integer_base_7_l684_684788

theorem largest_integer_base_7 :
  let M := 66 in
  M ^ 2 = 48 ^ 2 :=
by
  let M := (6 * 7 + 6) in
  have h : M ^ 2 = 48 ^ 2 := rfl
  sorry -- Proof not required.

end largest_integer_base_7_l684_684788


namespace convex_polygon_triangulation_l684_684831

theorem convex_polygon_triangulation (n : ℕ) (h_div : n % 3 = 0) :
  ∃ (T : set (set ℕ)), (∀ t ∈ T, ∃ (a b c : ℕ), t = {a, b, c} ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (∀ t1 t2 ∈ T, t1 ≠ t2 → t1 ∩ t2 ⊆ ∅) ∧ 
  (∀ v ∈ (finset.range n), (finset.filter (λ t, v ∈ t) T).card % 2 = 1) := 
sorry

end convex_polygon_triangulation_l684_684831


namespace am_gm_inequality_l684_684048

theorem am_gm_inequality
  (n : ℕ)
  (a : fin n → ℝ)
  (h_pos : ∀ i, 0 < a i)
  (h_prod : ∏ i, a i = 1) :
  (∏ i, 1 + a i) ≥ 2^n :=
by sorry

end am_gm_inequality_l684_684048


namespace pirate_schooner_problem_l684_684223

theorem pirate_schooner_problem (p : ℕ) (h1 : 10 < p) 
  (h2 : 0.54 * (p - 10) = (54 : ℝ) / 100 * (p - 10)) 
  (h3 : 0.34 * (p - 10) = (34 : ℝ) / 100 * (p - 10)) 
  (h4 : 2 / 3 * p = (2 : ℝ) / 3 * p) : 
  p = 60 := 
sorry

end pirate_schooner_problem_l684_684223


namespace interior_diagonals_sum_l684_684231

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 52)
  (h2 : 2 * (a * b + b * c + c * a) = 118) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 51 := 
by
  sorry

end interior_diagonals_sum_l684_684231


namespace taxi_ride_cost_l684_684974

-- Definitions given in the conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 10

-- The theorem we need to prove
theorem taxi_ride_cost : base_fare + (cost_per_mile * distance_traveled) = 5.00 :=
by
  sorry

end taxi_ride_cost_l684_684974


namespace smallest_pos_int_multiple_7_and_4_l684_684553

theorem smallest_pos_int_multiple_7_and_4 : ∃ (n : ℕ), n > 0 ∧ (n % 7 = 0) ∧ (n % 4 = 0) ∧ ∀ (m : ℕ), m > 0 ∧ (m % 7 = 0) ∧ (m % 4 = 0) → n ≤ m :=
  exists.intro 28
    (and.intro
      (by simp)  -- 28 > 0
      (and.intro
        (by norm_num)  -- 28 % 7 = 0
        (and.intro
          (by norm_num)  -- 28 % 4 = 0
          (by intros m hmn0 hmd7 hmd4; exact nat.le_of_dvd hmn0 (nat.dvd_of_mod_eq_zero (nat.gcd_dvd_right _ _))))))

end smallest_pos_int_multiple_7_and_4_l684_684553


namespace apartments_decrease_l684_684615

theorem apartments_decrease (p_initial e_initial p e q : ℕ) (h1: p_initial = 5) (h2: e_initial = 2) (h3: q = 1)
    (first_mod: p = p_initial - 2) (e_first_mod: e = e_initial + 3) (q_eq: q = 1)
    (second_mod: p = p - 2) (e_second_mod: e = e + 3) :
    p_initial * e_initial * q > p * e * q := by
  sorry

end apartments_decrease_l684_684615


namespace tan_alpha_eq_l684_684398

variables {ABC : Type*} [right_angled_triangle ABC]
variables {a : ℝ} (BC := a)
variables {n : ℕ} (odd_n : n % 2 = 1) (n_pos : 0 < n) 
variables {h : ℝ} (altitude_h : is_altitude ABC BC h)
variables {α : ℝ} (angle_α : is_angle_at_A_midpoint_of_hypotenuse BC α)

theorem tan_alpha_eq : 
  tan α = (4 * n * h) / ((n^2 - 1) * a) :=
sorry

end tan_alpha_eq_l684_684398


namespace quadratic_poly_correct_l684_684304

noncomputable def quadratic_poly : polynomial ℝ :=
  3 * polynomial.C 1 * ((polynomial.X - polynomial.C (4 + 2 * complex.I)) * 
  (polynomial.X - polynomial.C (4 - 2 * complex.I)))

theorem quadratic_poly_correct : 
  quadratic_poly = 3 * polynomial.X ^ 2 - 24 * polynomial.X + 60 := by
  sorry

end quadratic_poly_correct_l684_684304


namespace sum_int_values_binom_eq_l684_684128

theorem sum_int_values_binom_eq :
  (∑ n in finset.range 16, if (Nat.choose 15 n + Nat.choose 15 7 = Nat.choose 16 8) then n else 0) = 8 := by
  sorry

end sum_int_values_binom_eq_l684_684128


namespace animal_arrangement_l684_684065

theorem animal_arrangement :
  let chickens := 3
  let dogs := 3
  let cats := 4 
  let rabbits := 2
  let total_animals := chickens + dogs + cats + rabbits
  let factorial := Nat.factorial
  total_animals = 12 ∧
  (factorial 4 * factorial chickens * factorial dogs * factorial cats * factorial rabbits = 41472) :=
by
  { sorry }

end animal_arrangement_l684_684065


namespace molly_age_l684_684475

theorem molly_age (S M : ℕ) (h1 : S / M = 4 / 3) (h2 : S + 6 = 34) : M = 21 :=
by
  sorry

end molly_age_l684_684475


namespace min_trips_for_jeffrey_min_total_trips_for_pairs_l684_684216

-- Definitions for the problem
def students : ℕ := 12
def trip_size : ℕ := 6
def jeffrey : Prop := sorry -- Encoding Jeffrey's necessary trips

-- (a) Define the problem of the minimum number of trips Jeffrey could go on
theorem min_trips_for_jeffrey (trips : ℕ) :
  (∀ s : Finset ℕ, s.card = students - 1 → 
   ∃ t1 t2 t3 : Finset ℕ, (∀ t, t ∈ {t1, t2, t3} → t.card = trip_size + 1) ∧ 
   ∀ x ∈ s, ∃ t ∈ {t1, t2, t3}, x ∈ t) → 
  trips = 3 := sorry

-- (b) Define the problem of the minimum number of field trips such that each pair has been on at least one trip.
theorem min_total_trips_for_pairs (total_trips : ℕ) :
  (∀ s : Finset ℕ, s.card = students →
   ∃ trips : Finset (Finset ℕ), trips.card = total_trips ∧ 
   (∀ t ∈ trips, t.card = trip_size) ∧
   ∀ x y ∈ s, ∃ t ∈ trips, {x, y} ⊆ t) →
  total_trips = 6 := sorry

end min_trips_for_jeffrey_min_total_trips_for_pairs_l684_684216


namespace largest_number_of_HCF_LCM_l684_684492

theorem largest_number_of_HCF_LCM (HCF : ℕ) (k1 k2 : ℕ) (n1 n2 : ℕ) 
  (hHCF : HCF = 50)
  (hk1 : k1 = 11) 
  (hk2 : k2 = 12) 
  (hn1 : n1 = HCF * k1) 
  (hn2 : n2 = HCF * k2) :
  max n1 n2 = 600 := by
  sorry

end largest_number_of_HCF_LCM_l684_684492


namespace malcolm_initial_white_lights_l684_684086

-- Definitions based on the conditions
def red_lights : Nat := 12
def blue_lights : Nat := 3 * red_lights
def green_lights : Nat := 6
def total_colored_lights := red_lights + blue_lights + green_lights
def lights_left_to_buy : Nat := 5
def initially_white_lights := total_colored_lights + lights_left_to_buy

-- Proof statement
theorem malcolm_initial_white_lights : initially_white_lights = 59 := by
  sorry

end malcolm_initial_white_lights_l684_684086


namespace Willey_Farm_Available_Capital_l684_684866

theorem Willey_Farm_Available_Capital 
  (total_acres : ℕ)
  (cost_per_acre_corn : ℕ)
  (cost_per_acre_wheat : ℕ)
  (acres_wheat : ℕ)
  (available_capital : ℕ) :
  total_acres = 4500 →
  cost_per_acre_corn = 42 →
  cost_per_acre_wheat = 35 →
  acres_wheat = 3400 →
  available_capital = (acres_wheat * cost_per_acre_wheat) + 
                      ((total_acres - acres_wheat) * cost_per_acre_corn) →
  available_capital = 165200 := sorry

end Willey_Farm_Available_Capital_l684_684866


namespace find_angle_AOB_l684_684357

-- Define the parabola and chord conditions as given.
axiom parabola (p : ℝ) (h : p > 0) (theta : ℝ) (h_theta : 0 < theta ∧ theta ≤ π / 2) : Type

-- Define the fact that the chord AB passes through the focus F.
axiom chord_through_focus (AB : parabola p h theta h_theta) : True

-- Define the goal to prove
theorem find_angle_AOB (p : ℝ) (h : p > 0) (theta : ℝ) (h_theta : 0 < theta ∧ theta ≤ π / 2) :
  let AOB := ℝ -- Angle AOB
  let angle := Parabola p h theta h_theta in
  chord_through_focus angle →
  AOB = π - arctan (4 / (3 * sin theta)) := by
  sorry

end find_angle_AOB_l684_684357


namespace smoothie_supplements_combinations_l684_684603

/-- The number of combinations of one type of smoothie and three different supplements -/
theorem smoothie_supplements_combinations : 
  let smoothies := 7
  let supplements := 8
  smoothies * Nat.choose supplements 3 = 392 :=
by
  intros
  have h := Nat.choose_eq_factorial_div_factorial (m := supplements) 3
  rw [h]
  simp
  sorry

end smoothie_supplements_combinations_l684_684603


namespace quadratic_roots_eq_l684_684447

theorem quadratic_roots_eq (a : ℝ) (b : ℝ) :
  (∀ x, (2 * x^2 - 3 * x - 8 = 0) → 
         ((x + 3)^2 + a * (x + 3) + b = 0)) → 
  b = 9.5 :=
by
  sorry

end quadratic_roots_eq_l684_684447


namespace pears_for_strawberries_l684_684828

-- Define the exchange conditions as equalities.
-- Strawberries to raspberries
def exchange_1 : ℕ → ℕ → Prop := λ s r, 11 * s = 14 * r

-- Cherries to raspberries
def exchange_2 : ℕ → ℕ → Prop := λ c r, 22 * c = 21 * r

-- Cherries to bananas
def exchange_3 : ℕ → ℕ → Prop := λ c b, 10 * c = 3 * b

-- Pears to bananas
def exchange_4 : ℕ → ℕ → Prop := λ p b, 5 * p = 2 * b

-- Define the problem statement
theorem pears_for_strawberries (s : ℕ) (p : ℕ) : 
  exchange_1 s 14 ∧ exchange_2 (21 * 3) 21 ∧ exchange_3 (21 * 3) 3 ∧ exchange_4 p 2 →
  s = 7 → p = 7 :=
by sorry

end pears_for_strawberries_l684_684828


namespace always_possible_to_win_l684_684560

-- Define the finite states for the switches
inductive SwitchState
| up
| down

open SwitchState

-- Define the initial configuration with 4 switches
def initialConfiguration : List SwitchState := [up, down, up, down] -- placeholder

-- Define operations: flipping two switches
def flip (s1 s2 : SwitchState) : SwitchState × SwitchState :=
  match s1, s2 with
  | up, up     => (down, down)
  | down, down => (up, up)
  | up, down   => (down, up)
  | down, up   => (up, down)

-- Define a function to check if all switches are in the same state
def allSameState (config : List SwitchState) : Bool :=
  config.all (λ s, s = up) || config.all (λ s, s = down)

-- Problem statement: it is always possible to win the game
theorem always_possible_to_win : ∃ finalConfiguration, allSameState finalConfiguration :=
  sorry

end always_possible_to_win_l684_684560


namespace largest_integer_base_7_l684_684790

theorem largest_integer_base_7 :
  let M := 66 in
  M ^ 2 = 48 ^ 2 :=
by
  let M := (6 * 7 + 6) in
  have h : M ^ 2 = 48 ^ 2 := rfl
  sorry -- Proof not required.

end largest_integer_base_7_l684_684790


namespace total_money_raised_l684_684621

-- Define the baked goods quantities
def betty_chocolate_chip_cookies := 4
def betty_oatmeal_raisin_cookies := 6
def betty_regular_brownies := 2
def paige_sugar_cookies := 6
def paige_blondies := 3
def paige_cream_cheese_swirled_brownies := 5

-- Define the price of goods
def cookie_price := 1
def brownie_price := 2

-- State the total money raised
theorem total_money_raised :
  let total_cookies := 12 * (betty_chocolate_chip_cookies + betty_oatmeal_raisin_cookies + paige_sugar_cookies),
      total_brownies := 12 * (betty_regular_brownies + paige_blondies + paige_cream_cheese_swirled_brownies),
      money_from_cookies := total_cookies * cookie_price,
      money_from_brownies := total_brownies * brownie_price
  in money_from_cookies + money_from_brownies = 432 := by
  sorry

end total_money_raised_l684_684621


namespace matching_shoes_probability_is_one_ninth_l684_684564

def total_shoes : ℕ := 10
def pairs_of_shoes : ℕ := 5
def total_combinations : ℕ := (total_shoes * (total_shoes - 1)) / 2
def matching_combinations : ℕ := pairs_of_shoes

def matching_shoes_probability : ℚ := matching_combinations / total_combinations

theorem matching_shoes_probability_is_one_ninth :
  matching_shoes_probability = 1 / 9 :=
by
  sorry

end matching_shoes_probability_is_one_ninth_l684_684564


namespace malcolm_initial_white_lights_l684_684083

theorem malcolm_initial_white_lights :
  ∀ (red blue green remaining total_initial : ℕ),
    red = 12 →
    blue = 3 * red →
    green = 6 →
    remaining = 5 →
    total_initial = red + blue + green + remaining →
    total_initial = 59 :=
by
  intros red blue green remaining total_initial h1 h2 h3 h4 h5
  -- Add details if necessary for illustration
  -- sorry typically as per instructions
  sorry

end malcolm_initial_white_lights_l684_684083


namespace units_digit_factorial_50_is_0_l684_684145

theorem units_digit_factorial_50_is_0 : (nat.factorial 50) % 10 = 0 := by
  sorry

end units_digit_factorial_50_is_0_l684_684145


namespace first_group_number_l684_684535

variable (x : ℕ)

def number_of_first_group :=
  x = 6

theorem first_group_number (H1 : ∀ k : ℕ, k = 8 * 15 + x)
                          (H2 : k = 126) : 
                          number_of_first_group x :=
by
  sorry

end first_group_number_l684_684535


namespace partition_equilateral_triangle_l684_684796

theorem partition_equilateral_triangle (A B C : Point) (S : Set Point) 
  (h1 : is_equilateral_triangle A B C)
  (h2 : ∀ P ∈ S, P ∈ segment A B ∨ P ∈ segment B C ∨ P ∈ segment C A)
  (h3 : ∀ P ∈ {A, B, C}, P ∈ S)
  (partition : S → Prop) :
  (∃ T₁ T₂ : Set Point, T₁ ∪ T₂ = S ∧ T₁ ∩ T₂ = ∅ ∧
   ((∃ P Q R, {P, Q, R} ⊆ T₁ ∧ forms_right_triangle P Q R) ∨
    (∃ P Q R, {P, Q, R} ⊆ T₂ ∧ forms_right_triangle P Q R))) :=
sorry

end partition_equilateral_triangle_l684_684796


namespace largest_integer_with_4_digit_square_in_base_7_l684_684785

theorem largest_integer_with_4_digit_square_in_base_7 (M : ℕ) :
  (∀ m : ℕ, m < 240 ∧ 49 ≤ m → m ≤ 239) ∧ nat.to_digits 7 239 = [4, 6, 1] :=
begin
  sorry
end

end largest_integer_with_4_digit_square_in_base_7_l684_684785


namespace correct_sum_of_digits_l684_684867

theorem correct_sum_of_digits :
  ∃ (d e : ℕ), 
    (d ≠ e) ∧ 
    ( ∃ f g h,  
          364765 = f ∧ 
          951872 = g ∧ 
          1496637 = h ∧ 
          ∀ (n : ℕ), f + g = h ↔
          (n = d → replaceDigit f d e + replaceDigit g d e = replaceDigit h d e)
    ) ∧ (d + e = 7) :=
sorry

-- A helper function can be defined to simulate the digit replacement in the numbers:
def replaceDigit (n : ℕ) (d e : ℕ) : ℕ :=
sorry

end correct_sum_of_digits_l684_684867


namespace max_value_log_div_x_l684_684088

noncomputable def func (x : ℝ) := (Real.log x) / x

theorem max_value_log_div_x : ∃ x > 0, func x = 1 / Real.exp 1 ∧ 
(∀ t > 0, t ≠ x → func t ≤ func x) :=
sorry

end max_value_log_div_x_l684_684088


namespace double_acute_angle_is_positive_and_less_than_180_l684_684670

variable (α : ℝ) (h : 0 < α ∧ α < π / 2)

theorem double_acute_angle_is_positive_and_less_than_180 :
  0 < 2 * α ∧ 2 * α < π :=
by
  sorry

end double_acute_angle_is_positive_and_less_than_180_l684_684670


namespace g_g_g_3_equals_107_l684_684736

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_g_g_3_equals_107 : g (g (g 3)) = 107 := 
by 
  sorry

end g_g_g_3_equals_107_l684_684736


namespace intersection_M_N_l684_684029

open Set

def M : Set ℝ := {x | x ≥ 0}
def N : Set ℝ := {x | x^2 < 1}

theorem intersection_M_N : M ∩ N = Ico 0 1 := 
sorry

end intersection_M_N_l684_684029


namespace minimum_value_sin4_cos4_l684_684299

theorem minimum_value_sin4_cos4 (x : ℝ) : 
    ∃ y : ℝ, ∀ x : ℝ, (sin x)^4 + 2 * (cos x)^4 ≥ (sin y)^4 + 2 * (cos y)^4 ∧ (sin y)^4 + 2 * (cos y)^4 = 2 / 3 :=
by {
  -- Lean statement only, no proof will be provided
  sorry
}

end minimum_value_sin4_cos4_l684_684299


namespace minimum_filtration_processes_l684_684606

theorem minimum_filtration_processes : ∃ (n : ℕ), (1 - 0.20)^n < 0.05 ∧ ∀ m < n, (1 - 0.20)^m >= 0.05 :=
sorry

end minimum_filtration_processes_l684_684606


namespace students_calculation_correct_l684_684410

structure SecondYearStudents :=
  (NM : ℕ)
  (ACAV : ℕ)
  (AR : ℕ)
  (NM_ACAV : ℕ)
  (NM_AR : ℕ)
  (ACAV_AR : ℕ)
  (NM_ACAV_AR : ℕ)
  (percent_of_total : ℝ)

def total_students (sy : SecondYearStudents) : ℕ :=
  let total_studying := 
    sy.NM + sy.ACAV + sy.AR - sy.NM_ACAV - sy.NM_AR - sy.ACAV_AR + sy.NM_ACAV_AR
  in rational_to_int (total_studying / sy.percent_of_total)

theorem students_calculation_correct (sy : SecondYearStudents) :
  sy = { NM := 240, ACAV := 423, AR := 365, NM_ACAV := 134, NM_AR := 75,
         ACAV_AR := 95, NM_ACAV_AR := 45, percent_of_total := 0.8 } →
  total_students sy = 905 :=
by
  sorry

end students_calculation_correct_l684_684410


namespace integral_eval_l684_684189

noncomputable def integrand (x : ℝ) : ℝ :=
  (2 * (Real.cot x) + 1) / ((2 * Real.sin x + Real.cos x) ^ 2)

noncomputable def integral_value : ℝ :=
  (Real.integral (integrand) (Real.arccos (4 / Real.sqrt 17)) (Real.pi / 4))

theorem integral_eval :
  integral_value = 1 / 2 :=
sorry

end integral_eval_l684_684189


namespace proof_l684_684925

-- Define the expression
def expr : ℕ :=
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128)

-- Define the conjectured result
def result : ℕ := 5^128 - 4^128

-- Assert their equality
theorem proof : expr = result :=
by
    sorry

end proof_l684_684925


namespace solve_for_x_l684_684895

theorem solve_for_x :
  ∃ x : ℝ, (x > 0) ∧ (x ≠ 1) ∧ (x ≠ 5) ∧ (x ≠ 25) ∧ (x^{Real.logb 5 x} = x^2 / 25) :=
by
  sorry

end solve_for_x_l684_684895


namespace probability_abs_diff_gt_one_l684_684474

-- Define and prove the probability problem
theorem probability_abs_diff_gt_one :
  let coin_flip_probability := 1/2
  let choose_number_probability := 1/2
  let interval := set.Icc 0 2 -- Closed interval [0, 2]
  let x_choice := if (coin_flip_probability = 1/2) then 2 else 0
  let y_choice := if (coin_flip_probability = 1/2) then 2 else 0
  let x := if (coin_flip_probability = 1/2) then if (choose_number_probability = 1/2) then 2 else 0 else (set.Icc 0 2)
  let y := if (coin_flip_probability = 1/2) then if (choose_number_probability = 1/2) then 2 else 0 else (set.Icc 0 2)
  in (|x - y| > 1) =
      (9/16) :=
  sorry

end probability_abs_diff_gt_one_l684_684474


namespace martin_goldfish_count_l684_684814

-- Define the initial number of goldfish
def initial_goldfish := 18

-- Define the number of goldfish that die each week
def goldfish_die_per_week := 5

-- Define the number of goldfish purchased each week
def goldfish_purchased_per_week := 3

-- Define the number of weeks
def weeks := 7

-- Calculate the expected number of goldfish after 7 weeks
noncomputable def final_goldfish := initial_goldfish - (goldfish_die_per_week * weeks) + (goldfish_purchased_per_week * weeks)

-- State the theorem and the proof target
theorem martin_goldfish_count : final_goldfish = 4 := 
sorry

end martin_goldfish_count_l684_684814


namespace limit_unique_l684_684046

open Filter Set

theorem limit_unique {f : ℝ → ℝ} {x₀ : ℝ} 
  (h : ∃ L, Tendsto f (𝓝 x₀) (𝓝 L)) :
  ∃! L, Tendsto f (𝓝 x₀) (𝓝 L) :=
by
  sorry

end limit_unique_l684_684046


namespace largest_M_in_base_7_l684_684793

-- Define the base and the bounds for M^2
def base : ℕ := 7
def lower_bound : ℕ := base^3
def upper_bound : ℕ := base^4

-- Define M and its maximum value.
def M : ℕ := 48

-- Define a function to convert a number to its base 7 representation
def to_base_7 (n : ℕ) : List ℕ :=
  if n == 0 then [0] else
  let rec digits (n : ℕ) : List ℕ :=
    if n == 0 then [] else (n % 7) :: digits (n / 7)
  digits n |>.reverse

-- Define the base 7 representation of 48
def M_base_7 : List ℕ := to_base_7 M

-- The main statement asserting the conditions and the solution
theorem largest_M_in_base_7 :
  lower_bound ≤ M^2 ∧ M^2 < upper_bound ∧ M_base_7 = [6, 6] :=
by
  sorry

end largest_M_in_base_7_l684_684793


namespace smallest_range_of_sample_l684_684966

open Real

theorem smallest_range_of_sample {a b c d e f g : ℝ}
  (h1 : (a + b + c + d + e + f + g) / 7 = 8)
  (h2 : d = 10)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ f ∧ f ≤ g) :
  ∃ r, r = g - a ∧ r = 8 :=
by
  sorry

end smallest_range_of_sample_l684_684966


namespace jack_christina_speed_l684_684422

noncomputable def speed_of_jack_christina (d_jack_christina : ℝ) (v_lindy : ℝ) (d_lindy : ℝ) (relative_speed_factor : ℝ := 2) : ℝ :=
d_lindy * relative_speed_factor / d_jack_christina

theorem jack_christina_speed :
  speed_of_jack_christina 240 10 400 = 3 := by
  sorry

end jack_christina_speed_l684_684422


namespace small_squares_overlap_l684_684397

theorem small_squares_overlap 
  (A : Fin 9 → Set ℝ^2) -- The 9 small squares
  (plane : Set ℝ^2) -- The planar shape
  (h_total_area : measure_theory.measure_space.measure plane = 1)
  (h_each_square : ∀ i, measure_theory.measure_space.measure (A i) = 1/5)
  (h_within_plane : ∀ i, A i ⊆ plane):
  ∃ (i j : Fin 9), i ≠ j ∧ measure_theory.measure_space.measure (A i ∩ A j) ≥ 1/45 :=
begin
  sorry
end

end small_squares_overlap_l684_684397


namespace find_line_equation_l684_684345

theorem find_line_equation 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1)
  (P : ℝ × ℝ) (P_coord : P = (1, 3/2))
  (line_l : ∀ x : ℝ, ℝ)
  (line_eq : ∀ x : ℝ, y = k * x + b) 
  (intersects : ∀ A B : ℝ × ℝ, A ≠ P ∧ B ≠ P)
  (perpendicular : ∀ A B : ℝ × ℝ, (A.1 - 1) * (B.1 - 1) + (A.2 - 3 / 2) * (B.2 - 3 / 2) = 0)
  (bisected_by_y_axis : ∀ A B : ℝ × ℝ, A.1 + B.1 = 0) :
  ∃ k : ℝ, k = 3 / 2 ∨ k = -3 / 2 :=
sorry

end find_line_equation_l684_684345


namespace find_k_l684_684594

noncomputable def vector_line_through (a b : ℝ^n) (k : ℝ) : Prop :=
  ∃ t : ℝ, k * a + (1/2) * b = a + t * (b - a)

theorem find_k (a b : ℝ^n) (h : ∃ k : ℝ, vector_line_through a b k) :
  ∃ k : ℝ, k = 1 / 2 :=
  sorry

end find_k_l684_684594


namespace find_triangle_angles_l684_684893

theorem find_triangle_angles (α β γ : ℝ)
  (h1 : (180 - α) / (180 - β) = 13 / 9)
  (h2 : β - α = 45)
  (h3 : α + β + γ = 180) :
  (α = 33.75) ∧ (β = 78.75) ∧ (γ = 67.5) :=
by
  sorry

end find_triangle_angles_l684_684893


namespace is_rth_power_l684_684430

noncomputable def f : ℤ → ℤ := sorry -- Placeholder for the given function f.

theorem is_rth_power (M a b r : ℕ) (h1 : a ≥ 2) (h2 : r ≥ 2) 
    (hf1 : ∀ n : ℤ, (iterate r f n) = (a * n + b))
    (hf2 : ∀ n : ℤ, (M : ℤ) ≤ n → 0 ≤ f n)
    (hf3 : ∀ n m : ℤ, (M : ℤ) < m ∧ m ≤ n → ((m - n) ∣ (f m - f n))) : ∃ k : ℕ, a = k ^ r := 
sorry

end is_rth_power_l684_684430


namespace cos_angle_of_a_b_eq_neg4_over_9_projection_of_a_on_b_eq_neg2_over_9_b_find_m_n_if_parallel_l684_684721

def vector_a : ℝ × ℝ × ℝ := (1, 2, -2)
def vector_b : ℝ × ℝ × ℝ := (4, -2, 4)
def vector_c (m n : ℝ) : ℝ × ℝ × ℝ := (3, m, n)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def cos_angle (u v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let scalar := dot_product u v / (magnitude v ^ 2) in
  (scalar * v.1, scalar * v.2, scalar * v.3)

theorem cos_angle_of_a_b_eq_neg4_over_9 :
  cos_angle vector_a vector_b = -4 / 9 :=
sorry

theorem projection_of_a_on_b_eq_neg2_over_9_b :
  projection vector_a vector_b = (-2 / 9 * vector_b.1, -2 / 9 * vector_b.2, -2 / 9 * vector_b.3) :=
sorry

theorem find_m_n_if_parallel :
  ∃ (m n : ℝ), vector_c m n = 3 * vector_a :=
sorry

end cos_angle_of_a_b_eq_neg4_over_9_projection_of_a_on_b_eq_neg2_over_9_b_find_m_n_if_parallel_l684_684721


namespace freds_average_book_cost_l684_684314

theorem freds_average_book_cost :
  ∀ (initial_amount spent_amount num_books remaining_amount avg_cost : ℕ),
    initial_amount = 236 →
    remaining_amount = 14 →
    num_books = 6 →
    spent_amount = initial_amount - remaining_amount →
    avg_cost = spent_amount / num_books →
    avg_cost = 37 :=
by
  intros initial_amount spent_amount num_books remaining_amount avg_cost h_init h_rem h_books h_spent h_avg
  sorry

end freds_average_book_cost_l684_684314


namespace unique_solution_triple_l684_684293

theorem unique_solution_triple (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xy / z : ℚ) + (yz / x) + (zx / y) = 3 → (x = 1 ∧ y = 1 ∧ z = 1) := 
by 
  sorry

end unique_solution_triple_l684_684293


namespace sqrt_15_minus_1_bounds_l684_684283

theorem sqrt_15_minus_1_bounds :
  (sqrt 15) > 3 ∧ (sqrt 15) < 4 → 2 < (sqrt 15) - 1 ∧ (sqrt 15) - 1 < 3 :=
by
  intro h
  have h1 : 3 < sqrt 15 := h.left
  have h2 : sqrt 15 < 4 := h.right
  apply And.intro
  { -- 2 < sqrt 15 - 1
    linarith
  } 
  { -- sqrt 15 - 1 < 3
    linarith
  }
  sorry

end sqrt_15_minus_1_bounds_l684_684283


namespace translation_phenomena_l684_684557

inductive Phenomenon
| SwingingOnSwing : Phenomenon
| ElevatorGoingUp : Phenomenon
| MovementOfPlanets : Phenomenon
| ParcelsOnConveyorBelt : Phenomenon

open Phenomenon

def is_translation (p : Phenomenon) : Prop :=
match p with
| SwingingOnSwing => False
| ElevatorGoingUp => True
| MovementOfPlanets => False
| ParcelsOnConveyorBelt => True
end

theorem translation_phenomena :
  is_translation ElevatorGoingUp ∧ is_translation ParcelsOnConveyorBelt ∧
  ¬ is_translation SwingingOnSwing ∧ ¬ is_translation MovementOfPlanets :=
by
  repeat {sorry}

end translation_phenomena_l684_684557


namespace find_y_l684_684662

theorem find_y : ∃ y : ℝ, 12^y * 6^4 / 432 = 5184 ∧ y = 9 := 
by 
  sorry

end find_y_l684_684662


namespace largest_intersection_value_l684_684886

-- Define the polynomial
def poly (a b c x : ℝ) : ℝ := x^6 - 14 * x^5 + 45 * x^4 - 30 * x^3 + a * x^2 + b * x + c

-- Define the condition that polynomial intersects the line at three points
def intersects_line (x1 x2 x3 : ℝ) (a b c d e : ℝ) : Prop :=
  poly a b c x1 = d * x1 + e ∧ poly a b c x2 = d * x2 + e ∧ poly a b c x3 = d * x3 + e

-- State the theorem that the largest of these values is 8
theorem largest_intersection_value (a b c d e : ℝ) (x1 x2 x3 : ℝ) (h1 : 3 = x1) (h2 : 6 = x2) 
(h3 : 8 = x3) (h_intersec : intersects_line x1 x2 x3 a b c d e) : x3 = 8 :=
by
  exact h3

end largest_intersection_value_l684_684886


namespace material_used_is_correct_l684_684887

def total_bought : ℚ := 2/9 + 1/8
def leftover : ℚ := 4/18
def used : ℚ := total_bought - leftover

theorem material_used_is_correct : used = 1/8 :=
  by
    unfold total_bought leftover used
    sorry

end material_used_is_correct_l684_684887


namespace minimum_at_2_l684_684506

open Real

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem minimum_at_2 : 
  (∀ x : ℝ, ((x ≠ 2) → (f(2) ≤ f(x)))) :=
by
  sorry

end minimum_at_2_l684_684506


namespace sum_n_values_binom_l684_684125

open Nat

theorem sum_n_values_binom (h : ∑ n in finset.range 16, if (choose 15 n + choose 15 7 = choose 16 8) then n else 0) = 8 :=
by
  sorry

end sum_n_values_binom_l684_684125


namespace line_AM_equation_angle_OMA_OMB_l684_684809

def ellipse : Type := { C : ℝ → ℝ → Prop // ∀ x y, C x y ↔ (x^2 / 2 + y^2 = 1) }

def right_focus (e : ellipse) : (ℝ × ℝ) :=
(1, 0) -- Since semi-major axis sqrt(2), semi-minor axis 1

def intersect_line_ellipse (e : ellipse) (l : ℝ → Prop) : set (ℝ × ℝ) :=
{ p | ∃ x y, l x ∧ e.val x y }

def line_l : (ℝ → Prop) :=
fun x => x = 1 -- Perpendicular to the x-axis

def point_M := (2, 0) : (ℝ × ℝ)

theorem line_AM_equation :
  let e : ellipse := ⟨λ x y, x^2 / 2 + y^2 = 1, sorry⟩ in
  let l := line_l in
  let A := (1, sqrt 2 / 2) in -- Intersection point from calculations
  let B :=  (1, -sqrt 2 / 2) in -- Intersection point from calculations
  let AM_equation := (λ x y, y = (sqrt 2 / 2) * (x - 2)) in
  let F := right_focus e in
  F = (1, 0) → intersect_line_ellipse e l = {A, B} →
  AM_equation 1 (sqrt 2 / 2) := by sorry

theorem angle_OMA_OMB :
  let e : ellipse := ⟨λ x y, x^2 / 2 + y^2 = 1, sorry⟩ in
  let l := line_l in
  ∀ O M A B, O = (0, 0) ∧ M = point_M →
  let A := (1, sqrt 2 / 2) in -- Generic form depending on l 
  let B := (1, -sqrt 2 / 2) in -- Intersection points
  (∠ O M A = ∠ O M B) := by sorry

end line_AM_equation_angle_OMA_OMB_l684_684809


namespace find_k_l684_684591

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n + 5 else (n + 1) / 2

theorem find_k (k : ℤ) (h1 : k % 2 = 0) (h2 : g (g (g k)) = 61) : k = 236 :=
by
  sorry

end find_k_l684_684591


namespace minimum_value_expression_l684_684550

theorem minimum_value_expression (x y : ℝ) : ∃ (x y : ℝ), x = 4 ∧ y = -3 ∧ (x^2 + y^2 - 8 * x + 6 * y + 25) = 0 :=
by
  use 4, -3
  split
  · rfl
  split
  · rfl
  calc
    4^2 + (-3)^2 - 8 * 4 + 6 * (-3) + 25
      = 16 + 9 - 32 - 18 + 25 : by norm_num
  ... = 0 : by norm_num
  done

end minimum_value_expression_l684_684550


namespace max_value_of_expression_l684_684338

theorem max_value_of_expression (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 2 * a + b = 1) : 
  2 * Real.sqrt (a * b) - 4 * a ^ 2 - b ^ 2 ≤ (Real.sqrt 2 - 1) / 2 :=
sorry

end max_value_of_expression_l684_684338


namespace insurance_coverage_percentage_l684_684111

noncomputable def doctor_visit_cost : ℝ := 300
noncomputable def cat_visit_cost : ℝ := 120
noncomputable def pet_insurance_covered : ℝ := 60
noncomputable def tim_paid : ℝ := 135

theorem insurance_coverage_percentage :
  let total_cost_before_insurance := doctor_visit_cost + cat_visit_cost,
      total_insurance_covered := total_cost_before_insurance - tim_paid,
      insurance_for_doctor_visit := total_insurance_covered - pet_insurance_covered in
  (insurance_for_doctor_visit / doctor_visit_cost) * 100 = 75 := by sorry

end insurance_coverage_percentage_l684_684111


namespace total_apples_correct_example_l684_684618

-- Definitions for the conditions
def Benny_apples_per_tree := 2
def trees_Benny_picked_from := 4

def Dan_apples_per_tree := 9
def trees_Dan_picked_from := 5

def Sarah_ratio_to_Dan := 1 / 2

-- Calculate the total number of apples picked
def Benny_total_apples := Benny_apples_per_tree * trees_Benny_picked_from
def Dan_total_apples := Dan_apples_per_tree * trees_Dan_picked_from
def Sarah_total_apples := int ∘ floor (Sarah_ratio_to_Dan * Dan_total_apples)

def total_apples := Benny_total_apples + Dan_total_apples + Sarah_total_apples

theorem total_apples_correct : total_apples = 75 := by
  have hBenny : Benny_total_apples = 8 := by
    simp [Benny_apples_per_tree, trees_Benny_picked_from]
  have hDan: Dan_total_apples = 45 := by
    simp [Dan_apples_per_tree, trees_Dan_picked_from]
  have hSarah: Sarah_total_apples = 22 := by
    simp [Sarah_ratio_to_Dan, Dan_total_apples]
    norm_cast
  simp [total_apples, hBenny, hDan, hSarah]

-- Placeholder for proof
theorem example : total_apples = 75 := by sorry

end total_apples_correct_example_l684_684618


namespace part_I_part_II_l684_684811

-- Define the function f
noncomputable def f (x : ℝ) (m : ℤ) : ℝ := x - log (x + m)

-- Part (I)
theorem part_I (m : ℤ) (hx : x > -m) : m ≤ 1 → ∀ x : ℝ, f x m ≥ 0 := 
begin
  sorry
end

-- Part (II)
theorem part_II (m : ℤ) (hm : m > 1) :
  ∃ x1 x2 : ℝ, (exp(-m) - m) < x1 ∧ x1 < (1 - m) ∧ 
               f x1 m = 0 ∧ (1 - m) < x2 ∧ x2 < (exp(2 * m) - m) ∧ 
               f x2 m = 0 := 
begin
  sorry
end

end part_I_part_II_l684_684811


namespace only_tan_neg_x_has_properties_l684_684244

theorem only_tan_neg_x_has_properties :
  (∀ f : ℝ → ℝ,
      (f = (λ x, Real.sin (2 * x)) ∨
       f = (λ x, 2 * Real.abs (Real.cos x)) ∨
       f = (λ x, Real.cos (x / 2)) ∨
       f = (λ x, Real.tan (-x))) →
      (∀ T : ℝ, (T > 0) → (∀ x, f (x + T) = f x)) = T = π →
      (∀ a b : ℝ, (a < b) → (a > π / 2) → (b < π) → (f a > f b)) = f = (λ x, Real.tan (-x))) :=
by
  sorry

end only_tan_neg_x_has_properties_l684_684244


namespace stamp_total_cost_l684_684619

theorem stamp_total_cost :
  let price_A := 2
  let price_B := 3
  let price_C := 5
  let num_A := 150
  let num_B := 90
  let num_C := 60
  let discount_A := if num_A > 100 then 0.20 else 0
  let discount_B := if num_B > 50 then 0.15 else 0
  let discount_C := if num_C > 30 then 0.10 else 0
  let cost_A := num_A * price_A * (1 - discount_A)
  let cost_B := num_B * price_B * (1 - discount_B)
  let cost_C := num_C * price_C * (1 - discount_C)
  cost_A + cost_B + cost_C = 739.50 := sorry

end stamp_total_cost_l684_684619


namespace warehouse_cement_l684_684204

theorem warehouse_cement :
  ∃ (A B : ℕ), A + B = 462 ∧ A = 4 * B + 32 ∧ A = 376 ∧ B = 86 :=
by
  use 376
  use 86
  split
  · exact rfl
  split
  · exact rfl
  split
  sorry

end warehouse_cement_l684_684204


namespace probability_of_shaded_triangle_l684_684409

def triangle (name: String) := name

def triangles := ["AEC", "AEB", "BED", "BEC", "BDC", "ABD"]
def shaded_triangles := ["BEC", "BDC", "ABD"]

theorem probability_of_shaded_triangle :
  (shaded_triangles.length : ℚ) / (triangles.length : ℚ) = 1 / 2 := 
by
  sorry

end probability_of_shaded_triangle_l684_684409


namespace units_digit_of_50_factorial_l684_684175

theorem units_digit_of_50_factorial : 
  ∃ d, (d = List.prod (List.range 1 51)) ∧ (d % 10 = 0) :=
by
  sorry

end units_digit_of_50_factorial_l684_684175


namespace maximum_distance_point_to_line_l684_684443

-- Define the conditions of the problem
variables (m : ℝ)
noncomputable def P : ℝ × ℝ :=
  let y := (2 * m - 4) / (1 + m^2) in
  let x := -m * y in (x, y)

def line_l (θ : ℝ) (x y : ℝ) : Prop :=
  (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3

-- Define the statement of the mathematical proof
theorem maximum_distance_point_to_line (m θ : ℝ) (P : ℝ × ℝ)
  (hP : P = let y := (2 * m - 4) / (1 + m^2) in
            let x := -m * y in (x, y))
  (h_intersect : ∃ P : ℝ × ℝ, line_l θ P.1 P.2)
  (hdist : ∀ x y : ℝ, y = (2 * m - 4) / (1 + m^2) → (x, y) = P) :
  ∃ d : ℝ, d = 3 + Real.sqrt 5 :=
sorry

end maximum_distance_point_to_line_l684_684443


namespace determine_numbers_l684_684457

theorem determine_numbers (A B n : ℤ) (h1 : 0 ≤ n ∧ n ≤ 9) (h2 : A = 10 * B + n) (h3 : A + B = 2022) : 
  A = 1839 ∧ B = 183 :=
by
  -- proof will be filled in here
  sorry

end determine_numbers_l684_684457


namespace sequence_bound_l684_684445

-- Define the sequence {a_k}
noncomputable def a_seq (n : ℕ) (k : ℕ) : ℚ :=
  if k = 0 then 1 / 2 else (a_seq n (k - 1)) + (a_seq n (k - 1))^2 / n

-- State the theorem
theorem sequence_bound (n : ℕ) (hn : 0 < n) : 
  let a_n := a_seq n n in 1 - (1 : ℚ) / n < a_n ∧ a_n < 1 := 
by 
  sorry

end sequence_bound_l684_684445


namespace geometric_seq_general_term_sum_of_transformed_seq_l684_684712

open Nat

-- Define the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n, n ≥ 2 → a n = 2 * a (n - 1) - 1

-- Prove that the sequence $\{a_n - 1\}$ is a geometric sequence with common ratio 2
theorem geometric_seq (a : ℕ → ℕ) (h : seq a) :
  ∃ q, q = 2 ∧ (∀ n, n ≥ 1 → a (n + 1) - 1 = q * (a n - 1)) :=
sorry

-- Prove that the general term formula for the sequence is $a_n = 2^{n-1} + 1$
theorem general_term (a : ℕ → ℕ) (h : seq a) :
  ∀ n, a n = 2 ^ (n - 1) + 1 :=
sorry

-- Define the transformed sequence $b_n = n \cdot a_n - n$
def b (a : ℕ → ℕ) (n : ℕ) :=
  n * a n - n

-- Prove that the sum of the first $n$ terms of the sequence $b_n$ is $(n-1) \cdot 2^n + 1$
theorem sum_of_transformed_seq (a : ℕ → ℕ) (h : seq a) :
  ∀ n, (∑ k in Finset.range n, k * a k - k) = (n - 1) * 2^n + 1 :=
sorry

end geometric_seq_general_term_sum_of_transformed_seq_l684_684712


namespace find_coefficient_l684_684341

theorem find_coefficient (a : ℝ) : 
  let coeff := 10 + 5 * a 
  in coeff = 5 → a = -1 :=
by
  intro h
  sorry

end find_coefficient_l684_684341


namespace find_percentage_l684_684946

-- Define the necessary constants and conditions
def n : ℕ := 456
def m : ℕ := 120
def p : ℕ := 48 -- We know 40% of 120 is 48
def extra : ℕ := 180

-- Define the percentage that we want to prove 
def P : ℕ := 50 -- This is the value we are going to prove

-- State the theorem
theorem find_percentage (P : ℕ) (h₁ : p = 0.4 * m) (h₂ : P / 100 * n = p + extra) : P = 50 := by
  sorry


end find_percentage_l684_684946


namespace combined_investment_yield_l684_684860

theorem combined_investment_yield :
  let income_A := 0.14 * 500
  let income_B := 0.08 * 750
  let income_C := 0.12 * 1000
  let total_income := income_A + income_B + income_C
  let total_market_value := 500 + 750 + 1000
  (total_income / total_market_value) * 100 = 11.11 :=
by
  let income_A := 0.14 * 500
  let income_B := 0.08 * 750
  let income_C := 0.12 * 1000
  let total_income := income_A + income_B + income_C
  let total_market_value := 500 + 750 + 1000
  have : total_income = 250 := by norm_num
  have : total_market_value = 2250 := by norm_num
  have : (total_income / total_market_value) * 100 = (250 / 2250) * 100 := by congr
  have : (250 / 2250) * 100 = 11.11 := by norm_num
  exact this

end combined_investment_yield_l684_684860


namespace proof_of_problem_l684_684184

/-- Proposition A: Vertical angles are equal. -/
def proposition_A : Prop := ∀ (θ₁ θ₂ : ℝ), is_vertical_angle θ₁ θ₂ → θ₁ = θ₂

/-- Proposition B: Corresponding angles are equal. -/
def proposition_B : Prop := ∀ (θ₁ θ₂ : ℝ), is_corresponding_angle θ₁ θ₂ → θ₁ = θ₂

/-- Proposition C: If a² = b², then a = b. -/
def proposition_C : Prop := ∀ (a b : ℝ), a^2 = b^2 → a = b

/-- Proposition D: If x² > 0, then x > 0. -/
def proposition_D : Prop := ∀ (x : ℝ), x^2 > 0 → x > 0

/-- The overall proposition with the correct choice being A. -/
def problem_statement : Prop := proposition_A ∧ ¬proposition_B ∧ ¬proposition_C ∧ ¬proposition_D

theorem proof_of_problem : problem_statement := 
by
  sorry

end proof_of_problem_l684_684184


namespace cosine_and_shifted_sine_even_and_bounded_l684_684197

theorem cosine_and_shifted_sine_even_and_bounded :
  ∀ (x : ℝ), ∃ M m : ℝ, y = cos (2 * x) + sin (π / 2 - x) ∧
  (∀ x, y ≤ M) ∧ (∀ x, m ≤ y) :=
by
  sorry

end cosine_and_shifted_sine_even_and_bounded_l684_684197


namespace range_of_f_value_of_omega_strictly_increasing_intervals_l684_684701

def f (ω x : ℝ) : ℝ := sin (ω * x + π / 6) + sin (ω * x - π / 6) - 2 * cos (ω * x / 2) ^ 2

theorem range_of_f (ω : ℝ) (hω : ω > 0) :
  set.range (f ω) = set.Icc (-3 : ℝ) 1 :=
sorry

theorem value_of_omega (ω : ℝ) (h : ∀ a : ℝ, set.card (set_of (λ x, f ω x = -1) ∩ set.Ioo a (a + π)) = 2) :
  ω = 2 :=
sorry

theorem strictly_increasing_intervals (ω : ℝ) (hω : ω = 2) :
  ∀ k : ℤ, ∃ I : set ℝ, I = set.Icc (k * π - π / 6) (k * π + π / 3) ∧ 
  ∀ x₁ x₂ ∈ I, x₁ < x₂ → f ω x₁ < f ω x₂ :=
sorry

end range_of_f_value_of_omega_strictly_increasing_intervals_l684_684701


namespace gray_region_area_l684_684538

theorem gray_region_area (d_small r_large r_small π : ℝ) (h1 : d_small = 6)
    (h2 : r_large = 3 * r_small) (h3 : r_small = d_small / 2) :
    (π * r_large ^ 2 - π * r_small ^ 2) = 72 * π := 
by
  -- The proof will be filled here
  sorry

end gray_region_area_l684_684538


namespace color_change_arrangements_eq_l684_684249

-- Definitions and assumptions directly from the conditions
variables (n k : ℕ)
variables (hn : n > 0) (hk : 0 < k) (hk' : k < n)

-- State the theorem
theorem color_change_arrangements_eq (hnk : n > 0 ∧ 0 < k ∧ k < n) :
  let arrangements : ℕ → ℕ := λ v, if v % 2 = 1 then
      2 * nat.choose (n - 1) ((v - 1) / 2)^2
    else
      2 * nat.choose (n - 1) (v / 2) * nat.choose (n - 1) ((v - 2) / 2)

  in arrangements (n - k) = arrangements (n + k) :=
sorry

end color_change_arrangements_eq_l684_684249


namespace sheep_to_horses_ratio_l684_684990

-- Define the known quantities
def number_of_sheep := 32
def total_horse_food := 12880
def food_per_horse := 230

-- Calculate number of horses
def number_of_horses := total_horse_food / food_per_horse

-- Calculate and simplify the ratio of sheep to horses
def ratio_of_sheep_to_horses := (number_of_sheep : ℚ) / (number_of_horses : ℚ)

-- Define the expected simplified ratio
def expected_ratio_of_sheep_to_horses := (4 : ℚ) / (7 : ℚ)

-- The statement we want to prove
theorem sheep_to_horses_ratio : ratio_of_sheep_to_horses = expected_ratio_of_sheep_to_horses :=
by
  -- Proof will be here
  sorry

end sheep_to_horses_ratio_l684_684990


namespace Caden_total_money_l684_684993

theorem Caden_total_money (p n d q : ℕ) (hp : p = 120)
    (hn : p = 3 * n) 
    (hd : n = 5 * d)
    (hq : q = 2 * d) :
    (p * 1 / 100 + n * 5 / 100 + d * 10 / 100 + q * 25 / 100) = 8 := 
by
  sorry

end Caden_total_money_l684_684993


namespace find_k_l684_684353

theorem find_k (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (kx - A.1 - y + A.2 = 0) 
    ∧ (kx - B.1 - y + B.2 = 0) 
    ∧ (A.1 ^ 2 + A.2 ^ 2 = 4) 
    ∧ (B.1 ^ 2 + B.2 ^ 2 = 4)
    ∧ (A.1 × B.1 + A.2 × B.2 = 0)) 
  → (k = 1 ∨ k = -1) :=
by
  sorry

end find_k_l684_684353


namespace mrs_martin_pays_l684_684864

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def mr_martin_scoops : ℕ := 1
def mrs_martin_scoops : ℕ := 1
def children_scoops : ℕ := 2
def teenage_children_scoops : ℕ := 3

def total_cost : ℕ :=
  (mr_martin_scoops + mrs_martin_scoops) * regular_scoop_cost +
  children_scoops * kiddie_scoop_cost +
  teenage_children_scoops * double_scoop_cost

theorem mrs_martin_pays : total_cost = 32 :=
  by sorry

end mrs_martin_pays_l684_684864


namespace power_rounding_l684_684836

theorem power_rounding : (Real.pow 1.003 4).roundTo 3 = 1.012 :=
by
  sorry

end power_rounding_l684_684836


namespace proof_complement_intersection_l684_684715

-- Define the universal set
def U := set.univ : set ℝ         

-- Define set M in terms of logarithmic inequality
def M := {x : ℝ | real.log (x - 1) / real.log (1/2) > -1}

-- Define set N in terms of exponential inequality
def N := {x : ℝ | 1 < real.exp x ∧ real.exp x < 4 }

-- Define the complement of M in U
def complement_M_U := {x : ℝ | x ≤ 1 ∨ x ≥ 3 }

-- Define the intersection to be proved
def intersection := {x : ℝ | 0 < x ∧ x ≤ 1}

-- The final proof statement to be provided
theorem proof_complement_intersection : (complement_M_U ∩ N) = intersection := by
  sorry

end proof_complement_intersection_l684_684715


namespace validate_correct_statement_l684_684185

-- Define the data conditions for analysis.

def data_set := [2, 5, 4, 5, 6, 7]

-- Define the given variances.
def S_A_sq := 0.02
def S_B_sq := 0.01

-- Define each statement as a proposition.
def statement_A := "It rains a lot during the Qingming Festival is a certain event."
def statement_B := "To understand the service life of a batch of light tubes, a census survey can be conducted."
def statement_C := "The mode, median, and mean of a set of data 2, 5, 4, 5, 6, 7 are all 5."
def statement_D := "The variances of the heights of the team members in groups A and B are S_A^2=0.02 and S_B^2=0.01 respectively. Therefore, the heights of the team members in group B are more uniform."

-- Define the correct statement as D.
def correct_statement : Prop := statement_D

-- The theorem to be proven: among all statements, D is the correct one.
theorem validate_correct_statement : correct_statement := by
  sorry

end validate_correct_statement_l684_684185


namespace total_doctors_and_nurses_l684_684108

theorem total_doctors_and_nurses (ratio_doctors_nurses : 5 / 9)
                                  (nurses_count : ℕ)
                                  (nurses_at_hospital : nurses_count = 180) :
  let doctors_count := 5 * nurses_count / 9
  let total_count := doctors_count + nurses_count
  total_count = 280 :=
by
  -- Proof here
  sorry

end total_doctors_and_nurses_l684_684108


namespace ratio_S13_S7_l684_684343

-- Definitions for conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)
-- Arithmetic sequence conditions
axiom sum_def : ∀ n, S n = n * a n
axiom a7_eq_7a4 : a 7 = 7 * a 4

-- Statement of the problem
theorem ratio_S13_S7 : S 13 / S 7 = 13 :=
by
  -- Conditions being used
  have h1 : S 13 = 13 * a 7 := sum_def 13
  have h2 : S 7 = 7 * a 4 := sum_def 7
  have h3 : a 7 = 7 * a 4 := a7_eq_7a4
  -- Proof placeholder
  sorry

end ratio_S13_S7_l684_684343


namespace units_digit_of_50_factorial_l684_684142

theorem units_digit_of_50_factorial : (nat.factorial 50) % 10 = 0 := 
by 
  sorry

end units_digit_of_50_factorial_l684_684142


namespace min_cars_for_adults_5_min_cars_for_adults_8_l684_684392

-- Define the condition: each car must be off the road at least one day a week
-- Define the minimum number of cars required for the given number of adults

theorem min_cars_for_adults_5 : 
  (∀ (k : ℕ), (k ≥ 6 → (¬ each_day k (5 : ℕ)))) :=
by
  -- condition: each car must rest one day a week
  -- goal: Prove that 6 cars are needed for 5 adults
  sorry

theorem min_cars_for_adults_8 : 
  (∀ (k : ℕ), (k ≥ 10 → (¬ each_day k (8 : ℕ)))) :=
by
  -- condition: each car must rest one day a week
  -- goal: Prove that 10 cars are needed for 8 adults
  sorry

end min_cars_for_adults_5_min_cars_for_adults_8_l684_684392


namespace min_a2_b2_condition_l684_684747

problem:
def min_value_a2_plus_b2 (a b : ℝ) : ℝ :=
if (∃ (a b : ℝ) , (∑ i in finset.range 7, nat.choose 6 (i) * (a^6 - 2*i) * (b^i) * (6 - 2*i) = -160 )), then a^2 + b^2 else 4

theorem min_a2_b2_condition {a b : ℝ}
  (h : ∑ i in finset.range 7, nat.choose 6 (i) * (a^6 - 2*i) * (b^i) * (6 - 2*i) = -160) :
  a^2 + b^2 = 4 :=
begin
  unfold min_value_a2_plus_b2,
  sorry,
end

end min_a2_b2_condition_l684_684747


namespace f_f_sqrt2_eq_sqrt3_l684_684348

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/3)^x else Real.logBase (1/2) x

theorem f_f_sqrt2_eq_sqrt3 : f (f (Real.sqrt 2)) = Real.sqrt 3 := 
  by
  -- proof
  sorry

end f_f_sqrt2_eq_sqrt3_l684_684348


namespace fibonacci_sum_div_l684_684879

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_sum_div {a : ℕ → ℕ} :
  a 1 = 1 →
  a 2 = 1 →
  (∀ (n : ℕ), a (n + 2) = a (n + 1) + a n) →
  (∑ i in finset.range 2017, (a (i + 1))^2) / (a 2017) = a 2018 :=
by
  sorry

end fibonacci_sum_div_l684_684879


namespace probability_log2_x_between_1_and_2_l684_684229

noncomputable def probability_log_between : ℝ :=
  let favorable_range := (4:ℝ) - (2:ℝ)
  let total_range := (6:ℝ) - (0:ℝ)
  favorable_range / total_range

theorem probability_log2_x_between_1_and_2 :
  probability_log_between = 1 / 3 :=
sorry

end probability_log2_x_between_1_and_2_l684_684229


namespace min_value_expression_l684_684542

open Real

theorem min_value_expression : ∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 :=
by
  intro x y
  sorry

end min_value_expression_l684_684542


namespace smallest_number_of_coins_l684_684469

theorem smallest_number_of_coins : ∃ (n : ℕ), 
  n ≡ 2 [MOD 5] ∧ 
  n ≡ 1 [MOD 4] ∧ 
  n ≡ 0 [MOD 3] ∧ 
  n = 57 := 
by
  sorry

end smallest_number_of_coins_l684_684469


namespace smallest_possible_perimeter_is_52_l684_684230

-- Define the problem
def smallest_rectangle_perimeter : ℕ :=
  let a := 1
  let b := 3 * a
  let length := 2 * a + 3 * b
  let width := 3 * a + 4 * b
  2 * (length + width)

-- Prove the statement
theorem smallest_possible_perimeter_is_52 :
  smallest_rectangle_perimeter = 52 :=
by
  -- Simplify the expressions
  have h1: 3 * a = 3 * 1 := rfl
  have h2: 2 * 1 + 3 * (3 * 1) = 11 := by norm_num
  have h3: 3 * 1 + 4 * (3 * 1) = 15 := by norm_num
  have h4: 2 * (11 + 15) = 52 := by norm_num
  rw [h1, h2, h3, h4]
  rfl

end smallest_possible_perimeter_is_52_l684_684230


namespace evaluate_expression_l684_684285

variable (a : ℝ)
variable (x : ℝ)

theorem evaluate_expression (h : x = a + 9) : x - a + 6 = 15 := by
  sorry

end evaluate_expression_l684_684285


namespace total_participants_l684_684042

theorem total_participants (Petya Vasya total : ℕ) 
  (h1 : Petya = Vasya + 1) 
  (h2 : Petya = 10)
  (h3 : Vasya + 15 = total + 1) : 
  total = 23 :=
by
  sorry

end total_participants_l684_684042


namespace parallelogram_area_l684_684366

def vector := (ℝ × ℝ × ℝ)

def magnitude (v : vector) :=
  match v with
  | (x, y, z) => real.sqrt (x^2 + y^2 + z^2)

def dot_product (v1 v2 : vector) :=
  match v1, v2 with
  | (x1, y1, z1), (x2, y2, z2) => x1 * x2 + y1 * y2 + z1 * z2

def area_of_parallelogram (v1 v2 : vector) :=
  let a := magnitude v1
  let b := magnitude v2
  (a * b)

theorem parallelogram_area (a b : vector) (ha : a = (2, -1, 1)) (hb : b = (1, 3, 1)) :
  area_of_parallelogram a b = real.sqrt 66 :=
by
  sorry

end parallelogram_area_l684_684366


namespace probability_of_2_out_of_5_accurate_probability_of_2_out_of_5_with_third_accurate_l684_684627

-- Defining the conditions
def p : ℚ := 4 / 5
def n : ℕ := 5
def k1 : ℕ := 2
def k2 : ℕ := 1

-- Binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Binomial probability function
def binom_prob (k n : ℕ) (p : ℚ) : ℚ :=
  binomial n k * p^k * (1 - p)^(n - k)

-- The first proof problem:
-- Prove that the probability of exactly 2 out of 5 forecasts being accurate is 0.05 given the accuracy rate
theorem probability_of_2_out_of_5_accurate :
  binom_prob k1 n p = 0.05 := by
  sorry

-- The second proof problem:
-- Prove that the probability of exactly 2 out of 5 forecasts being accurate, with the third forecast being one of the accurate ones, is 0.02 given the accuracy rate
theorem probability_of_2_out_of_5_with_third_accurate :
  binom_prob k2 (n - 1) p = 0.02 := by
  sorry

end probability_of_2_out_of_5_accurate_probability_of_2_out_of_5_with_third_accurate_l684_684627


namespace perpendicular_vectors_l684_684321

def vec := ℝ × ℝ

def dot_product (a b : vec) : ℝ :=
  a.1 * b.1 + a.2 * b.2

variables (m : ℝ)
def a : vec := (1, 2)
def b : vec := (m, 1)

theorem perpendicular_vectors (h : dot_product a (b m) = 0) : m = -2 :=
sorry

end perpendicular_vectors_l684_684321


namespace sum_digits_of_consecutive_numbers_l684_684191

-- Define the sum of digits function
def sum_digits (n : ℕ) : ℕ := sorry -- Placeholder, define the sum of digits function

-- Given conditions
variables (N : ℕ)
axiom h1 : sum_digits N + sum_digits (N + 1) = 200
axiom h2 : sum_digits (N + 2) + sum_digits (N + 3) = 105

-- Theorem statement to be proved
theorem sum_digits_of_consecutive_numbers : 
  sum_digits (N + 1) + sum_digits (N + 2) = 103 := 
sorry  -- Proof to be provided

end sum_digits_of_consecutive_numbers_l684_684191


namespace effective_radius_approx_52_08_l684_684241

-- Definitions based on the given conditions
def distance_miles : ℝ := 120
def revolutions : ℕ := 2300
def friction_force : ℝ := -8
def incline_angle_deg : ℝ := 10

-- Conversion factors
def miles_to_inches : ℝ := 5280 * 12
def pi : ℝ := Real.pi

-- Query for the effective radius in inches
noncomputable def effective_radius_in_inches (distance_miles : ℝ) (revolutions : ℕ) (pi : ℝ) : ℝ :=
  let C := (distance_miles * miles_to_inches) / revolutions
  C / (2 * pi)

-- Assertion of the result
theorem effective_radius_approx_52_08 :
  effective_radius_in_inches distance_miles revolutions pi ≈ 52.08 :=
by
  sorry

end effective_radius_approx_52_08_l684_684241


namespace function_range_ge_4_l684_684095

variable {x : ℝ}

theorem function_range_ge_4 (h : x > 0) : 2 * x + 2 * x⁻¹ ≥ 4 :=
sorry

end function_range_ge_4_l684_684095


namespace proof_problem_l684_684801

noncomputable theory
open Real

def f (x a b : ℝ) : ℝ := x + a * x^2 + b * log x

def tangent_slope (x a b : ℝ) : ℝ := 1 + 2 * a * x + b / x

theorem proof_problem :
  ∃ (a b : ℝ), 
    (f 1 a b = 0) ∧ (tangent_slope 1 a b = 2) ∧ 
    (∀ x > 0, f x (-1) 3 ≤ 2 * x - 2) :=
begin
  use [-1, 3],
  split,
  { simp [f], norm_num },
  split,
  { simp [tangent_slope], norm_num },
  { intros x hx,
    -- The following proof sketch will assume g(x) = -x^2 + 3 * log x - x + 2 ≤ 0 as proven
    sorry
  }
end

end proof_problem_l684_684801


namespace MorseCodeDistinctSymbolsCount_l684_684769

theorem MorseCodeDistinctSymbolsCount :
  let S := {0, 1, 2} -- Let {dot, dash, blank} be represented by {0, 1, 2}
  ∃ (n : Nat), n = 3 + (3 * 3) + (3 * 3 * 3) ∧ n = 39 := 
by
s ''Sorry

end MorseCodeDistinctSymbolsCount_l684_684769


namespace sequence_bounded_l684_684512

theorem sequence_bounded (a : ℕ → ℝ) :
  a 0 = 2 →
  (∀ n, a (n+1) = (2 * a n + 1) / (a n + 2)) →
  ∀ n, 1 < a n ∧ a n < 1 + 1 / 3^n :=
by
  intro h₀ h₁
  sorry

end sequence_bounded_l684_684512


namespace plane_equation_l684_684648

def point := ℝ × ℝ × ℝ

def plane_eq (A B C D : ℤ) (p : point) : Prop :=
  ∃ (x y z : ℝ), p = (x, y, z) ∧ (A : ℝ) * x + (B : ℝ) * y + (C : ℝ) * z + (D : ℝ) = 0

theorem plane_equation (A B C D : ℤ)
  (hA : A > 0)
  (hgcd : Int.gcd A (Int.gcd B (Int.gcd C D)) = 1)
  (p1 p2 p3 : point)
  (hp1 : p1 = (2, -1, 3))
  (hp2 : p2 = (5, -1, 1))
  (hp3 : p3 = (7, 0, 2)) :
  plane_eq 2 -7 3 (-20) p1 ∧ plane_eq 2 -7 3 (-20) p2 ∧ plane_eq 2 -7 3 (-20) p3 :=
by
  sorry

end plane_equation_l684_684648


namespace probability_not_bought_by_Jim_l684_684968

open Finset

theorem probability_not_bought_by_Jim
  (total_pictures : ℕ) (bought_pictures : ℕ) (pick_pictures : ℕ)
  (h_total : total_pictures = 10) (h_bought : bought_pictures = 3) (h_pick : pick_pictures = 2) :
  (choose (total_pictures - bought_pictures) pick_pictures) / (choose total_pictures pick_pictures) = (7 / 15 : ℚ) :=
by
  sorry

end probability_not_bought_by_Jim_l684_684968


namespace half_n_lt_m_lt_two_n_l684_684434

theorem half_n_lt_m_lt_two_n (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n)
  (h : ∃ x : ℤ, (x + m) * (x + n) = x + m + n) :
  1 / 2 * n < m ∧ m < 2 * n :=
sorry

end half_n_lt_m_lt_two_n_l684_684434


namespace total_polishing_time_is_correct_l684_684992

-- Definitions of quantities and their properties
def total_pairs_shoes : ℕ := 10
def total_belts : ℕ := 5
def total_watches : ℕ := 3

def time_per_shoe : ℕ := 10
def time_per_belt : ℕ := 15
def time_per_watch : ℕ := 5

def percentage_polished_shoes : ℚ := 0.45
def percentage_polished_belts : ℚ := 0.60
def percentage_polished_watches : ℚ := 0.20

-- Remaining items
def remaining_shoes : ℕ := total_pairs_shoes * 2 - nat.floor (percentage_polished_shoes * (total_pairs_shoes * 2).to_rat)
def remaining_belts : ℕ := total_belts - nat.floor (percentage_polished_belts * (total_belts).to_rat)
def remaining_watches : ℕ := total_watches - nat.floor (percentage_polished_watches * (total_watches).to_rat)

-- Total time needed to polish remaining items
def polishing_time : ℕ := remaining_shoes * time_per_shoe + remaining_belts * time_per_belt + remaining_watches * time_per_watch

theorem total_polishing_time_is_correct :
  polishing_time = 150 :=
by
  sorry

end total_polishing_time_is_correct_l684_684992


namespace cube_cut_edges_l684_684587

theorem cube_cut_edges : 
  let original_edges := 12 in
  let vertices := 8 in
  let new_edges_per_vertex := 4 in
  let total_new_edges := vertices * new_edges_per_vertex in
  let total_edges := original_edges + total_new_edges in
  total_edges = 44 :=
by
  let original_edges := 12
  let vertices := 8
  let new_edges_per_vertex := 4
  let total_new_edges := vertices * new_edges_per_vertex
  let total_edges := original_edges + total_new_edges
  have h : total_edges = 44 := by
    simp [original_edges, vertices, new_edges_per_vertex, total_new_edges, total_edges]
  exact h

end cube_cut_edges_l684_684587


namespace find_value_of_c_in_triangle_l684_684755

noncomputable def value_of_c (a : ℝ) (A : ℝ) (cos_B : ℝ) : ℝ :=
  let sin_A := Real.sin A
  let cos_A := Real.cos A
  let sin_B := Real.sin (Real.acos cos_B)
  let sin_C := sin_A * cos_B + cos_A * sin_B in
  a / sin_A * sin_C

theorem find_value_of_c_in_triangle 
  (a : ℝ) (A : ℝ) (cos_B : ℝ) 
  (ha : a = 5) 
  (hA : A = Real.pi / 4) 
  (hcosB : cos_B = 3 / 5) : 
  value_of_c a A cos_B = 7 :=
by 
  rw [ha, hA, hcosB]
  sorry

end find_value_of_c_in_triangle_l684_684755


namespace number_of_possible_bases_l684_684853

theorem number_of_possible_bases (n : ℕ) (h1 : n >= 2) : n = 3 :=
  let S := {c : ℕ // c^3 ≤ 250 ∧ 250 < c^4 ∧ c ≥ 2 } in
  Finset.card S = 3 :=
  sorry

end number_of_possible_bases_l684_684853


namespace contrapositive_proposition_contrapositive_equiv_l684_684075

theorem contrapositive_proposition (x : ℝ) (h : -1 < x ∧ x < 1) : (x^2 < 1) :=
sorry

theorem contrapositive_equiv (x : ℝ) (h : x^2 ≥ 1) : x ≥ 1 ∨ x ≤ -1 :=
sorry

end contrapositive_proposition_contrapositive_equiv_l684_684075


namespace product_mnp_l684_684498

theorem product_mnp (a x y b : ℝ) (m n p : ℕ):
  (a ^ 8 * x * y - 2 * a ^ 7 * y - 3 * a ^ 6 * x = 2 * a ^ 5 * (b ^ 5 - 2)) ∧
  (a ^ 8 * x * y - 2 * a ^ 7 * y - 3 * a ^ 6 * x + 6 * a ^ 5 = (a ^ m * x - 2 * a ^ n) * (a ^ p * y - 3 * a ^ 3)) →
  m = 5 ∧ n = 4 ∧ p = 3 ∧ m * n * p = 60 :=
by
  intros h
  sorry

end product_mnp_l684_684498


namespace spaces_per_tray_l684_684278

-- Conditions
def num_ice_cubes_glass : ℕ := 8
def num_ice_cubes_pitcher : ℕ := 2 * num_ice_cubes_glass
def total_ice_cubes_used : ℕ := num_ice_cubes_glass + num_ice_cubes_pitcher
def num_trays : ℕ := 2

-- Proof statement
theorem spaces_per_tray : total_ice_cubes_used / num_trays = 12 :=
by
  sorry

end spaces_per_tray_l684_684278


namespace parabola_shape_formed_l684_684021

variable (a c : ℝ) (h₀ : 0 < a) (h₁ : 0 < c)

/-- Proving that the shape formed by the vertices \((x_t, y_t)\) of the parabolas
\(y = ax^2 + tx + c\) for all real \(t\) is a parabola -/
theorem parabola_shape_formed (t : ℝ) :
  let x_t := -t / (2 * a),
      y_t := c - (t^2 / (4 * a))
  in ∃ k : ℝ, y_t = -a * x_t^2 + k := by
  sorry

end parabola_shape_formed_l684_684021


namespace isosceles_right_triangle_to_isosceles_trapezoids_l684_684044

-- Definitions and conditions

-- Define what constitutes an isosceles right triangle
def is_isosceles_right_triangle (ABC : Triangle) : Prop :=
  ABC.is_isosceles ∧ ABC.angle_A = 90

-- Define what constitutes an isosceles trapezoid
def is_isosceles_trapezoid (trapezoid : Quadrilateral) : Prop :=
  trapezoid.is_trapezoid ∧ trapezoid.non_parallel_sides_equal

-- Main theorem
theorem isosceles_right_triangle_to_isosceles_trapezoids (ABC : Triangle)
  (h : is_isosceles_right_triangle ABC) : 
  ∃ T : Fin 7 → Quadrilateral, 
  (∀ i : Fin 7, is_isosceles_trapezoid (T i)) ∧
  (ABC.divided_into_seven_trapezoids_by T) :=
sorry

end isosceles_right_triangle_to_isosceles_trapezoids_l684_684044


namespace decreasing_function_iff_l684_684507

-- Let's define the function f
def f (a x : ℝ) : ℝ := a * x^2 + 4 * (a + 1) * x - 3

-- Define the main theorem
theorem decreasing_function_iff :
  ∀ a : ℝ, (∀ x ∈ set.Ici (2:ℝ), ∀ y ∈ set.Ici (x), f a y ≤ f a x) ↔ a ≤ -1/2 :=
begin
  sorry
end

end decreasing_function_iff_l684_684507


namespace divide_P_Q_l684_684696

noncomputable def sequence_of_ones (n : ℕ) : ℕ := (10 ^ n - 1) / 9

theorem divide_P_Q (n : ℕ) (h : 1997 ∣ sequence_of_ones n) :
  1997 ∣ (sequence_of_ones (n + 1) * (10^(3*n) + 9 * 10^(2*n) + 9 * 10^n + 7)) ∧
  1997 ∣ (sequence_of_ones (n + 1) * (10^(3*(n + 1)) + 9 * 10^(2*(n + 1)) + 9 * 10^(n + 1) + 7)) := 
by
  sorry

end divide_P_Q_l684_684696


namespace polynomial_division_example_l684_684305

noncomputable def polynomial_div (f g : Polynomial ℤ) :=
  Polynomial.divModByMonic f g

theorem polynomial_division_example :
  polynomial_div (Polynomial.of_fn [12, -16, 13, -27, 5, -4, 1]) (Polynomial.of_fn [-3, 1]) =
  (Polynomial.of_fn [-166, -50, -21, 2, -1, 1], Polynomial.of_fn [-486]) :=
by
  sorry

end polynomial_division_example_l684_684305


namespace line_intersects_circle_but_not_center_l684_684709

noncomputable def circle (t : ℝ) : ℝ × ℝ := (-1 + 2 * Real.cos t, 3 + 2 * Real.sin t)
noncomputable def line (m : ℝ) : ℝ × ℝ := (2 * m - 1, 6 * m - 1)

def circle_center := (-1, 3 : ℝ)
def circle_radius := 2 : ℝ

theorem line_intersects_circle_but_not_center :
  let center := circle_center 
  let radius := circle_radius 
  let line_equation := (3 * x - y + 2 = 0) 
  let A := 3 
  let B := -1 
  let C := 2 
  let d := abs (A * center.1 + B * center.2 + C) / Real.sqrt (A ^ 2 + B ^ 2) 
  d < radius ∧ d ≠ 0 :=
sorry

end line_intersects_circle_but_not_center_l684_684709


namespace min_value_of_expression_l684_684547

open Real

noncomputable def min_expression_value : ℝ :=
  let expr := λ (x y : ℝ), x^2 + y^2 - 8 * x + 6 * y + 25
  0

theorem min_value_of_expression : ∃ x y : ℝ, x = 4 ∧ y = -3 ∧ (x^2 + y^2 - 8 * x + 6 * y + 25) = min_expression_value :=
by {
  use [4, -3],
  split,
  { refl },
  split,
  { refl },
  sorry
}

end min_value_of_expression_l684_684547


namespace find_AC_of_triangle_l684_684516

/-
The radius of the circumcircle of the acute-angled triangle \(ABC\) is 1.
The center of another circle passing through vertices \(A\), \(C\), and the orthocenter of triangle \(ABC\) lies on this circumcircle.
Prove that \(AC = \sqrt{3}\).

Problem restated in Lean 4:
-/

theorem find_AC_of_triangle (A B C H O : Type) (R : real) (circ_radius : R = 1) :
  (AC : real) = real.sqrt 3 := 
by 
  sorry

end find_AC_of_triangle_l684_684516


namespace tangent_line_at_1_minimum_value_a_l684_684699

noncomputable def f (x : ℝ) := 2 * Real.log x - 3 * x^2 - 11 * x

-- (1) Prove the equation of the tangent line to the curve y = f(x) at the point (1, f(1))
theorem tangent_line_at_1 : 
  let df := (2 / x - 6 * x - 11) in
  let f1 := f 1 in
  let df1 := df 1 in
  ∃ m b, df1 = m ∧ f1 = b + m * 1 ∧ ∀ x, f1 + m * (x - 1) = -15 * x + 1 :=
sorry

-- (2) Prove the minimum value of the positive integer a such that f(x) ≤ (a - 3)x^2 + (2a - 13)x - 2
theorem minimum_value_a : 
  (∀ x : ℝ, f x ≤ (a - 3) * x^2 + (2a - 13) * x - 2) → (a : ℕ) ≥ 1 :=
sorry

end tangent_line_at_1_minimum_value_a_l684_684699


namespace vector_magnitude_proof_l684_684364

variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions
variables (a b : V)
variables (ha : ∥a∥ = 1)
variables (hab : ∥a - b∥ = real.sqrt 3)
variables (hinner : ⟪a, a - b⟫ = 0)

theorem vector_magnitude_proof :
  ∥(2 : ℝ) • a + b∥ = 2 * real.sqrt 3 :=
sorry

end vector_magnitude_proof_l684_684364


namespace smallest_angle_pentagon_l684_684089

theorem smallest_angle_pentagon (x : ℝ) (h : 16 * x = 540) : 2 * x = 67.5 := 
by 
  sorry

end smallest_angle_pentagon_l684_684089


namespace boxes_sold_l684_684001

theorem boxes_sold (start_boxes sold_boxes left_boxes : ℕ) (h1 : start_boxes = 10) (h2 : left_boxes = 5) (h3 : start_boxes - sold_boxes = left_boxes) : sold_boxes = 5 :=
by
  sorry

end boxes_sold_l684_684001


namespace find_m_l684_684885

def f (x m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

theorem find_m :
  let m := 10 / 7
  3 * f 5 m = 2 * g 5 m :=
by
  sorry

end find_m_l684_684885


namespace magic_numbers_range_1_to_50_l684_684932

def double_or_subtract (n : ℕ) : ℕ :=
  if n ≤ 30 then 2 * n else n - 15

def sequence_contains (start target : ℕ) : Prop :=
  ∃ k, (Nat.iterate double_or_subtract k start = target)

def is_magic_number (G : ℕ) : Prop :=
  ¬sequence_contains G 20

def num_magic_numbers (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter is_magic_number |>.length

theorem magic_numbers_range_1_to_50 : num_magic_numbers 50 = 39 := by
  sorry

end magic_numbers_range_1_to_50_l684_684932


namespace alpha_in_second_quadrant_l684_684370

noncomputable theory

open real

def is_second_quadrant (θ : ℝ) : Prop := 
  (cos θ < 0) ∧ (sin θ > 0)

theorem alpha_in_second_quadrant (α : ℝ) :
  (cos α < 0) ∧ (sin α > 0) → is_second_quadrant α :=
by {
  sorry
}

end alpha_in_second_quadrant_l684_684370


namespace number_of_values_l684_684513

/-- Given:
  - The mean of some values was 190.
  - One value 165 was wrongly copied as 130 for the computation of the mean.
  - The correct mean is 191.4.
  Prove: the total number of values is 25. --/
theorem number_of_values (n : ℕ) (h₁ : (190 : ℝ) = ((190 * n) - (165 - 130)) / n) (h₂ : (191.4 : ℝ) = ((190 * n + 35) / n)) : n = 25 :=
sorry

end number_of_values_l684_684513


namespace find_x_l684_684584

theorem find_x (x : ℝ) (h : x * 1.6 - (2 * 1.4) / 1.3 = 4) : x = 3.846154 :=
sorry

end find_x_l684_684584


namespace letters_in_afternoon_l684_684778

theorem letters_in_afternoon (morning_letters afternoon_letters : ℕ) (h1 : morning_letters = 8) (h2 : morning_letters = afternoon_letters + 1) :
  afternoon_letters = 7 :=
by
  rw [h1, h2]
  sorry

end letters_in_afternoon_l684_684778


namespace sum_of_intervals_eq_one_l684_684664

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℝ :=
  floor x * (2013 ^ (x - floor x) - 1)

theorem sum_of_intervals_eq_one :
  (∑ k in Finset.range 2012, Real.log 2013 (k + 1) / k) = 1 := 
sorry

end sum_of_intervals_eq_one_l684_684664


namespace range_of_x_l684_684737

theorem range_of_x 
  (x : ℝ)
  (h1 : 1 / x < 4) 
  (h2 : 1 / x > -6) 
  (h3 : x < 0) : 
  -1 / 6 < x ∧ x < 0 := 
by 
  sorry

end range_of_x_l684_684737


namespace alex_min_additional_coins_l684_684978

theorem alex_min_additional_coins (n m k : ℕ) (h_n : n = 15) (h_m : m = 120) :
  k = 0 ↔ m = (n * (n + 1)) / 2 :=
by
  sorry

end alex_min_additional_coins_l684_684978


namespace max_intersections_l684_684122

/-- Given two different circles and three different straight lines, the maximum number of
points of intersection on a plane is 17. -/
theorem max_intersections (c1 c2 : Circle) (l1 l2 l3 : Line) (h_distinct_cir : c1 ≠ c2) (h_distinct_lines : ∀ (l1 l2 : Line), l1 ≠ l2) :
  ∃ (n : ℕ), n = 17 :=
by
  sorry

end max_intersections_l684_684122


namespace min_value_of_expression_l684_684545

open Real

noncomputable def min_expression_value : ℝ :=
  let expr := λ (x y : ℝ), x^2 + y^2 - 8 * x + 6 * y + 25
  0

theorem min_value_of_expression : ∃ x y : ℝ, x = 4 ∧ y = -3 ∧ (x^2 + y^2 - 8 * x + 6 * y + 25) = min_expression_value :=
by {
  use [4, -3],
  split,
  { refl },
  split,
  { refl },
  sorry
}

end min_value_of_expression_l684_684545


namespace andy_last_problem_l684_684982

theorem andy_last_problem (s t : ℕ) (start : s = 75) (total : t = 51) : (s + t - 1) = 125 :=
by
  sorry

end andy_last_problem_l684_684982


namespace exactly_four_horses_meet_at_210_l684_684908

theorem exactly_four_horses_meet_at_210 (horses : List ℕ) 
  (h₁ : horses = [2, 3, 5, 7, 11, 13, 17]) : 
  ∃ (T : ℕ), T = 210 ∧ 
  ((horses.filter (λ n, T % n = 0)).length = 4) ∧ 
  ∀ T' > 0, ((horses.filter (λ n, T' % n = 0)).length = 4) → T' ≥ 210 :=
by
  sorry

end exactly_four_horses_meet_at_210_l684_684908


namespace eccentricity_of_hyperbola_l684_684868

variables (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
def hyperbola_asymptote_distance := (x : ℝ) (y : ℝ) (a b : ℝ) := 
  (a_constraint : a > 0) 
  (b_constraint : b > 0) 
  (h₃ : (x - sqrt 3) ^ 2 + (y - 1) ^ 2 = 1) 
  (h₄ : x = sqrt 3)
  (h₅ : y = 1)

# The condition that asymptotes of the hyperbola touch the circle
theorem eccentricity_of_hyperbola (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h_asymptote_tangent : ∃ b, ∀ x y, (x - sqrt 3) ^ 2 + (y - 1) ^ 2 = 1 → 
    (|sqrt 3 * b - a| / sqrt (a^2 + b^2) = 1 ∨ |sqrt 3 * b + a| / sqrt (a^2 + b^2) = 1)) :
  ∃ a, (b = sqrt 3 * a ∧ sqrt (a^2 + b^2) / abs a = 2) :=
sorry

end eccentricity_of_hyperbola_l684_684868


namespace insurance_premium_l684_684219

variables (a p x : ℝ)  -- representing real numbers

-- Conditions
def payout := a  -- payout amount if event E occurs 
def probability := p  -- probability of event E occurring within a year
def expected_revenue := 0.1 * a  -- expected revenue is 10% of a

-- Company's annual profit random variable distribution
def profit (x : ℝ) (a : ℝ) (p : ℝ) : ℝ :=
  x * (1 - p) + (x - a) * p

-- Statement to prove
theorem insurance_premium (h_profit : profit x a p = expected_revenue) :
  x = a * (p + 0.1) :=
  sorry

end insurance_premium_l684_684219


namespace monthly_income_l684_684067

def average_expenditure_6_months (expenditure_6_months : ℕ) (average : ℕ) : Prop :=
  average = expenditure_6_months / 6

def expenditure_next_4_months (expenditure_4_months : ℕ) (monthly_expense : ℕ) : Prop :=
  expenditure_4_months = 4 * monthly_expense

def cleared_debt_and_saved (income_4_months : ℕ) (debt : ℕ) (savings : ℕ)  (condition : ℕ) : Prop :=
  income_4_months = debt + savings + condition

theorem monthly_income 
(income : ℕ) 
(avg_6m_exp : ℕ) 
(exp_4m : ℕ) 
(debt: ℕ) 
(savings: ℕ )
(condition: ℕ) 
    (h1 : average_expenditure_6_months avg_6m_exp 85) 
    (h2 : expenditure_next_4_months exp_4m 60) 
    (h3 : cleared_debt_and_saved (income * 4) debt savings 30) 
    (h4 : income * 6 < 6 * avg_6m_exp) 
    : income = 78 :=
sorry

end monthly_income_l684_684067


namespace triple_composition_g_eq_107_l684_684733

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_g_eq_107 : g(g(g(3))) = 107 := by
  sorry

end triple_composition_g_eq_107_l684_684733


namespace factorial_fraction_is_integer_l684_684437

theorem factorial_fraction_is_integer (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ∃ k : ℤ, (a.factorial * b.factorial * (a + b).factorial) / (2 * a).factorial * (2 * b).factorial = k := 
sorry

end factorial_fraction_is_integer_l684_684437


namespace probability_equation_l684_684668

def group := {A1, A2, B1, B2}
def boys := {A1, A2}
def girls := {B1, B2}
def selections := {set.union {A1, A2} {A1, B1}, set.union {A1, B2} {A2, B1}, set.union {A2, B2}}

def probability_one_boy_one_girl_given_one_boy_selected : ℚ :=
  let valid_selections := [{A1, A2}, {A1, B1}, {A1, B2}, {A2, B1}, {A2, B2}] in
  let favorable_selections := [{A1, B1}, {A1, B2}, {A2, B1}, {A2, B2}] in
  (favorable_selections.length) / (valid_selections.length)

theorem probability_equation :
  probability_one_boy_one_girl_given_one_boy_selected = 2 / 3 :=
begin
  sorry
end

end probability_equation_l684_684668


namespace binary_to_decimal_l684_684637

theorem binary_to_decimal : 
  let b := [1, 1, 0, 1] in
  let decimal := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 in
  decimal = 13 := by
  let b := [1, 1, 0, 1]
  let decimal := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  show decimal = 13 from
  sorry

end binary_to_decimal_l684_684637


namespace triangle_problem_l684_684327

-- Given the conditions:
-- 1. Triangle ABC with sides opposite angles A, B, C being a, b, c respectively
-- 2. a * cos C + (1 / 2) * c = b
-- 3. a = 1

-- Define the conditions as Lean structures
variables {A B C : ℝ} -- angles in radians
variables {a b c : ℝ} -- side lengths

-- Condition that a * cos(C) + (1/2) * c = b
def condition1 (a b c C : ℝ) : Prop := a * real.cos C + (1 / 2) * c = b

-- Condition that a = 1
def condition2 (a : ℝ) : Prop := a = 1

-- State the theorem
theorem triangle_problem :
  ∀ (A B C : ℝ) (a b c : ℝ),
    condition1 a b c C →
    condition2 a →
    A = 60 * (π / 180) ∧ 2 < 1 + 2 * real.sin (B + 30 * (π / 180)) ∧ 1 + 2 * real.sin (B + 30 * (π / 180)) ≤ 3 :=
by
  sorry

end triangle_problem_l684_684327


namespace problem_I4_1_l684_684387

theorem problem_I4_1 (a : ℝ) : ((∃ y : ℝ, x + 2 * y + 3 = 0) ∧ (∃ y : ℝ, 4 * x - a * y + 5 = 0) ∧ 
  (∃ m1 m2 : ℝ, m1 = -(1 / 2) ∧ m2 = 4 / a ∧ m1 * m2 = -1)) → a = 2 :=
sorry

end problem_I4_1_l684_684387


namespace coin_flip_probability_l684_684955

theorem coin_flip_probability :
  let total_flips := 8
  let num_heads := 6
  let total_outcomes := (2: ℝ) ^ total_flips
  let favorable_outcomes := (Nat.choose total_flips num_heads)
  let probability := favorable_outcomes / total_outcomes
  probability = (7 / 64 : ℝ) :=
by
  sorry

end coin_flip_probability_l684_684955


namespace oranges_per_store_visit_l684_684849

theorem oranges_per_store_visit (total_oranges : ℕ) (store_visits : ℕ) (h1 : total_oranges = 16) (h2 : store_visits = 8) :
  (total_oranges / store_visits) = 2 :=
by
  rw [h1, h2],
  exact Nat.div_eq_of_eq_mul_right (by norm_num) (by norm_num : 16 = 8 * 2)

end oranges_per_store_visit_l684_684849


namespace Amanda_notebooks_l684_684243

theorem Amanda_notebooks (initial ordered lost final : ℕ) 
  (h_initial: initial = 65) 
  (h_ordered: ordered = 23) 
  (h_lost: lost = 14) : 
  final = 74 := 
by 
  -- calculation and proof will go here
  sorry 

end Amanda_notebooks_l684_684243


namespace angle_PBA_eq_angle_DBH_l684_684487

open EuclideanGeometry

variables (A B C D H P : Point)
variables [parallelogram ABCD]
variables [angle_eq (∠DAC) 90°]
variables [foot H A D C]
variables [point_on_line P A C]
variables [tangent (line_through P D) (circumcircle_of_triangle A B D)]

theorem angle_PBA_eq_angle_DBH :
  angle P B A = angle D B H :=
sorry

end angle_PBA_eq_angle_DBH_l684_684487


namespace measure_of_B_value_of_b2_over_ac_l684_684389

-- Define the given conditions
variables {A B C : ℝ} (a b c : ℝ)

-- Problem 1: Prove the measure of angle B
theorem measure_of_B
(h1 : sin A / cos A + sin B / cos B = sqrt 2 * (sin C / cos A))
(h2 : 0 < B ∧ B < π) :
B = π / 4 :=
sorry

-- Problem 2: Prove the value of b^2 / ac
theorem value_of_b2_over_ac
(h3 : sin A / sin C + sin C / sin A = 2)
(h4 : a = c * real.sin B)
(h5 : b^2 = a^2 + c^2 - 2 * a * c * cos B) :
b^2 / (a * c) = 2 - sqrt 2 :=
sorry

end measure_of_B_value_of_b2_over_ac_l684_684389


namespace range_of_m_l684_684749

-- Define the conditions
variables (x m : ℝ)

-- The equation
def equation := (2 * x - 1) / (x + 1) = 3 - m / (x + 1)

-- The solution x must be a negative number
def x_negative := x < 0

-- Lean statement to prove the range of values for m
theorem range_of_m (h : equation) (hx : x_negative) : m < 4 ∧ m ≠ 3 :=
sorry

end range_of_m_l684_684749


namespace units_digit_50_factorial_l684_684163

theorem units_digit_50_factorial : (nat.factorial 50) % 10 = 0 :=
by
  sorry

end units_digit_50_factorial_l684_684163


namespace min_a_for_positive_distinct_roots_l684_684358

theorem min_a_for_positive_distinct_roots:
  ∃ (a : ℕ), 
    (1 ≤ a) ∧ ∀ (b c : ℝ), 
      (1 ≤ c) → 
      (1 ≤ a + b + c) → 
      (∃ x1 x2 : ℝ, (0 < x1) ∧ (x1 < 1) ∧ (0 < x2) ∧ (x2 < 1) ∧ (a * x1 ^ 2 + b * x1 + c = 0) ∧ (a * x2 ^ 2 + b * x2 + c = 0)) → 
    a = 4 :=
begin
  sorry
end

end min_a_for_positive_distinct_roots_l684_684358


namespace sum_of_ages_is_20_l684_684032

-- Let t be the age of one of the twin sisters, and l be the age of Lilian
def sum_of_ages (t l : ℕ) : ℕ :=
  2 * t + l

theorem sum_of_ages_is_20 (t l : ℕ) (h1 : t > l) (h2 : t^2 * l = 162) :
  sum_of_ages t l = 20 :=
begin
  -- Proof omitted
  sorry
end

end sum_of_ages_is_20_l684_684032


namespace clea_time_on_escalator_l684_684630

variable (d c s : ℝ)
variable (h1 : d = 90 * c)
variable (h2 : d = 36 * (c + s))

theorem clea_time_on_escalator (h : s = 3 / 2 * c) : (d/s = 60) := by
  calc 
    d / s = (90 * c) / (3 / 2 * c) : by rw [h1, h]
        ... = 60                 : by field_simp [ne_of_gt (by norm_num), h1, h2]

end clea_time_on_escalator_l684_684630


namespace minimized_MN_length_maximized_CMXN_area_l684_684359

-- Definition: Given a right triangle ABC with hypotenuse AB and a point X on AB,
-- M and N are the projections of X onto the legs AC and BC respectively.
variable (A B C X M N : Point)
variable (right_triangle : right_triangle A B C)
variable (on_hypotenuse : on_line_segment X A B)
variable (projections : projections X M N A C B C)

-- Part (a): Prove that the length of segment MN is minimized when X is the foot of the perpendicular from C to AB.
theorem minimized_MN_length : is_foot_of_perpendicular X C A B → minimized_length (segment_length M N) :=
by
  sorry

-- Part (b): Prove that the area of quadrilateral CMXN is maximized when X is the midpoint of AB.
theorem maximized_CMXN_area : is_midpoint X A B → maximized_area (area_of_quadrilateral C M X N) :=
by
  sorry

end minimized_MN_length_maximized_CMXN_area_l684_684359


namespace intervals_of_monotonic_increase_find_sin_alpha_l684_684019

-- Define the function f(x)
def f (x : ℝ) : ℝ := cos x * (2 * √3 * sin x - cos x) + sin x ^ 2

-- The intervals of monotonic increase
theorem intervals_of_monotonic_increase (k : ℤ) :
  let a := -π/6 + k * π,
      b := π/3 + k * π
  in strict_mono_incr_on f (Icc a b) := sorry

-- Given conditions for sin(alpha)
variables (α : ℝ) (h1 : π/6 < α) (h2 : α < 2 * π / 3) (h3 : f (α / 2) = 1/2)

-- The value of sin(alpha)
theorem find_sin_alpha : sin α = (√3 + √15) / 8 := sorry

end intervals_of_monotonic_increase_find_sin_alpha_l684_684019


namespace sequence_value_series_value_l684_684704

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

def sequence (n : Nat) : ℝ :=
  if n = 1 then (1 / 2 : ℝ)
  else 2 * (sequence (n - 1)) / (1 + (sequence (n - 1))^2)

axiom f_defined_on_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → ∃ y : ℝ, f y = f x
axiom f_initial_condition : f (1 / 2) = -1
axiom functional_equation : ∀ x y : ℝ, -1 < x ∧ x < 1 → -1 < y ∧ y < 1 → f x + f y = f ((x + y) / (1 + x * y))

theorem sequence_value (n : Nat) : f (sequence n) = -2^(n-1) := by
  sorry

theorem series_value (n : Nat) : 1 + (∑ k in Finset.range n, f (1 / ((k + 1)^2 + 3*(k + 1) + 1))) + f (1 / (n + 2)) = f 0 := by
  sorry

end sequence_value_series_value_l684_684704


namespace option_B_not_congruent_l684_684181

variables {A B C A' B' C' : Type}
variables [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C]
variables [euclidean_geometry A'] [euclidean_geometry B'] [euclidean_geometry C']
variables {angleA angleB angleC : α} {sideAB sideBC sideAC : α}
variables {angleA' angleB' angleC' : α} {sideA'B' sideB'C' sideA'C' : α}

def triangle_congruent (A B C A' B' C' : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] : Prop :=
  A ≈ A' ∧ B ≈ B' ∧ C ≈ C'

theorem option_B_not_congruent 
  (h1 : angleA = angleA')
  (h2 : sideAB = sideA'B')
  (h3 : sideBC = sideB'C') : 
  ¬ triangle_congruent A B C A' B' C' :=
by
  sorry

end option_B_not_congruent_l684_684181


namespace dishes_mode_and_median_l684_684063

def number_of_dishes : List ℕ := [3, 5, 4, 6, 3, 3, 4]

noncomputable def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x, max acc (l.count x)) 0

noncomputable def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· < ·)
  sorted.get (sorted.length / 2)

theorem dishes_mode_and_median :
  mode number_of_dishes = 3 ∧ median number_of_dishes = 4 :=
by
  sorry

end dishes_mode_and_median_l684_684063


namespace circles_intersect_l684_684275

noncomputable def C1 := {center := (-2, 2), radius := 4}
noncomputable def C2 := {center := (1, -2), radius := 2}

theorem circles_intersect :
    let d := (λ (x1 y1 x2 y2 : ℝ), real.sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2))
    in d (-2) 2 1 (-2) = 5 ∧ 4 - 2 < 5 ∧ 5 < 4 + 2 :=
by
  sorry

end circles_intersect_l684_684275


namespace slope_is_negative_sqrt_three_l684_684355

noncomputable def slope_of_line_through_focus (A B : ℝ × ℝ) (h : abs (A.1 - B.1) = 16 / 3) : Prop :=
  let C := { p : ℝ × ℝ | p.2^2 = 4 * p.1 }
  let F : ℝ × ℝ := (1, 0)
  let l := { p : ℝ × ℝ | ∃ k : ℝ, k ≠ 0 ∧ p.2 = k * (p.1 - 1) }
  let obtuse_angle (k : ℝ) := k < -1 ∨ k > 1
  l F ∧ ((A ∈ C ∧ B ∈ C) ∧ (A ∈ l ∧ B ∈ l)) ∧ h → slope_of_line_through_focus A B = -√3

theorem slope_is_negative_sqrt_three (A B : ℝ × ℝ) :
  let C := { p : ℝ × ℝ | p.2^2 = 4 * p.1 }
  let F : ℝ × ℝ := (1, 0)
  let l := { p : ℝ × ℝ | ∃ k : ℝ, k ≠ 0 ∧ p.2 = k * (p.1 - 1) }
  (A ∈ C ∧ B ∈ C) ∧ (A ∈ l ∧ B ∈ l) ∧ abs (A.1 - B.1) = 16 / 3 ∧ (line_obtuse_angle l) →
  (∃ k : ℝ, k ≠ 0 ∧ slope_of_line_through_focus A B = k ∧ (k < -1 ∨ k > 1)) := 
  sorry

end slope_is_negative_sqrt_three_l684_684355


namespace ants_species_A_count_l684_684984

theorem ants_species_A_count (a b : ℕ) (h1 : a + b = 30) (h2 : 2^5 * a + 3^5 * b = 3281) : 32 * a = 608 :=
by
  sorry

end ants_species_A_count_l684_684984


namespace min_value_correct_l684_684690

noncomputable def min_value (a b x y : ℝ) [Fact (a > 0)] [Fact (b > 0)] [Fact (x > 0)] [Fact (y > 0)] : ℝ :=
  if x + y = 1 then (a / x + b / y) else 0

theorem min_value_correct (a b x y : ℝ) [Fact (a > 0)] [Fact (b > 0)] [Fact (x > 0)] [Fact (y > 0)]
  (h : x + y = 1) : min_value a b x y = (Real.sqrt a + Real.sqrt b)^2 :=
by
  sorry

end min_value_correct_l684_684690


namespace minimum_value_expression_l684_684551

theorem minimum_value_expression (x y : ℝ) : ∃ (x y : ℝ), x = 4 ∧ y = -3 ∧ (x^2 + y^2 - 8 * x + 6 * y + 25) = 0 :=
by
  use 4, -3
  split
  · rfl
  split
  · rfl
  calc
    4^2 + (-3)^2 - 8 * 4 + 6 * (-3) + 25
      = 16 + 9 - 32 - 18 + 25 : by norm_num
  ... = 0 : by norm_num
  done

end minimum_value_expression_l684_684551


namespace probability_of_two_approvals_l684_684395

theorem probability_of_two_approvals (P_A : ℝ) : P_A = 0.6 → 
  ∑ (k in Finset.range(5)), 
    if k = 2 then (((nat.choose 4 k) : ℝ) * (P_A^k) * (1 - P_A)^(4-k)) 
    else 0 = 0.3456 :=
by
  intro h
  rw h
  sorry

end probability_of_two_approvals_l684_684395


namespace common_factor_of_polynomials_l684_684614

-- Define the polynomials
def p1 (m : ℝ) : ℝ := m * (m - 3) + 2 * (3 - m)
def p2 (m : ℝ) : ℝ := m^2 - 4 * m + 4
def p3 (m : ℝ) : ℝ := m^4 - 16

-- Definition of common factor
def is_common_factor (f : ℝ → ℝ) (g1 g2 g3 : ℝ → ℝ) : Prop :=
  ∀ m, g1 m = 0 → g2 m = 0 → g3 m = 0 → f m = 0

-- Now we state the problem
theorem common_factor_of_polynomials : is_common_factor (λ m, m - 2) p1 p2 p3 :=
sorry

end common_factor_of_polynomials_l684_684614


namespace students_on_bus_after_stop_l684_684110

variable (x : ℝ)
variable (students_initial : ℝ) (students_final : ℝ)

-- Conditions
axiom h1 : students_initial = 28
axiom h2 : 0.40 * x = 12
axiom h3 : students_final = students_initial + x

-- Proof goal
theorem students_on_bus_after_stop : students_final = 58 :=
by
  rw [h3, h1]
  have hx : x = 30 := by sorry
  rw [hx]
  norm_num

end students_on_bus_after_stop_l684_684110


namespace problem_l684_684700

noncomputable def f (x a : ℝ) : ℝ := log x + x^2 - a * x

theorem problem 
  (a : ℝ)
  (x : ℝ)
  (h1 : deriv (fun y => log y + y^2 - a * y) 1 = 0) :
  a = 3 ∧
  (∀ (a : ℝ), (0 < a ∧ a ≤ 2) → ∀ x > 0, (deriv (fun y => (log y + y^2 - a * y)) x > 0) → true) ∧
  (∀ (a : ℝ) (x0 : ℝ), (1 < a ∧ a < 2) ∧ (1 ≤ x0 ∧ x0 ≤ 2) → 
    ∀ m : ℝ, (f x0 a > m * log a) → m ≤ -log 2 e) :=
by 
  sorry

end problem_l684_684700


namespace total_toucans_l684_684580

def initial_toucans : Nat := 2

def new_toucans : Nat := 1

theorem total_toucans : initial_toucans + new_toucans = 3 := by
  sorry

end total_toucans_l684_684580


namespace range_of_slope_angle_proof_l684_684593

noncomputable def range_of_slope_angle (k : ℝ) : Prop :=
  ∃ A : ℝ × ℝ, A = (sqrt 3, 1) ∧ ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ y - 1 = k * (x - sqrt 3)

theorem range_of_slope_angle_proof :
  (∀ (k : ℝ), ∃ θ : ℝ, θ = real.arctan k ∧ 0 ≤ θ ∧ θ ≤ real.pi / 3) ↔
  range_of_slope_angle :=
sorry

end range_of_slope_angle_proof_l684_684593


namespace find_given_number_l684_684242

theorem find_given_number (x : ℕ) : 10 * x + 2 = 3 * (x + 200000) → x = 85714 :=
by
  sorry

end find_given_number_l684_684242


namespace min_pipes_required_l684_684963

noncomputable def volume_3_inch_pipe (h : ℝ) : ℝ := π * (1.5 ^ 2) * h
noncomputable def volume_12_inch_pipe (h : ℝ) : ℝ := π * (6 ^ 2) * h

theorem min_pipes_required (h : ℝ) : ∃ (x : ℕ), volume_3_inch_pipe h * x = volume_12_inch_pipe h ∧ x = 16 :=
by
  sorry

end min_pipes_required_l684_684963


namespace binary_to_decimal_to_base_5_l684_684267

theorem binary_to_decimal_to_base_5 (n : ℕ) (h : n = 51) :
  (51 : ℕ) = 2 * 5^2 + 0 * 5^1 + 1 * 5^0 := by 
  -- Calculation for binary to decimal provided
  have h₁ : (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1) = 51 := by sorry
  rw [← h, h₁]
  -- Calculation for decimal to base 5
  symmetry,
  apply nat.eq_of_dvd_of_add_mul_eq_pow,
  -- 5^2 + 5^0 part
  have : 51 = 2 * 5^2 + 1 := by sorry,
  -- 5^1 part
  have : 2 * 5^2 + 1 = 2 * 5^2 + 0 * 5 + 1 := by sorry,
  -- Concluding proof
  rwa [nat.add_mul_eq_add_mul],
  sorry

end binary_to_decimal_to_base_5_l684_684267


namespace julie_mowed_20_lawns_l684_684426

theorem julie_mowed_20_lawns :
  ∀ (cost_bike saved leftover num_newspapers num_dogs pay_lawn pay_newspaper pay_dog : ℕ),
  cost_bike = 2345 →
  saved = 1500 →
  leftover = 155 →
  num_newspapers = 600 →
  num_dogs = 24 →
  pay_lawn = 20 →
  pay_newspaper = 40 →
  pay_dog = 15 →
  ((saved + num_newspapers * pay_newspaper / 100 + num_dogs * pay_dog) + (pay_lawn * ?lawns) = cost_bike + leftover) →
  ?lawns = 20 :=
by
  intros
  sorry

end julie_mowed_20_lawns_l684_684426


namespace granddaughter_is_worst_l684_684758

inductive Player
| grandmother
| daughter
| grandson
| granddaughter

namespace family_tournament

def is_worst_player (p : Player) : Prop := sorry
def is_best_player (p : Player) : Prop := sorry
def is_twin (p1 p2 : Player) : Prop := sorry
def is_opposite_sex (p1 p2 : Player) : Prop := sorry
def is_same_age (p1 p2 : Player) : Prop := sorry

axiom players : set Player := { Player.grandmother, Player.daughter, Player.grandson, Player.granddaughter }

axiom condition1 (p : Player) (t : Player) (b : Player) :
  is_worst_player p ∧ is_twin p t ∧ is_best_player b ∧ t ∈ players ∧ b ∈ players → is_opposite_sex t b

axiom condition2 (p : Player) (b : Player) :
  is_worst_player p ∧ is_best_player b ∧ p ∈ players ∧ b ∈ players → is_same_age p b

noncomputable def worst_player : Player :=
  Player.granddaughter

theorem granddaughter_is_worst :
  is_worst_player worst_player :=
sorry

end family_tournament

end granddaughter_is_worst_l684_684758


namespace freds_average_book_cost_l684_684315

theorem freds_average_book_cost :
  ∀ (initial_amount spent_amount num_books remaining_amount avg_cost : ℕ),
    initial_amount = 236 →
    remaining_amount = 14 →
    num_books = 6 →
    spent_amount = initial_amount - remaining_amount →
    avg_cost = spent_amount / num_books →
    avg_cost = 37 :=
by
  intros initial_amount spent_amount num_books remaining_amount avg_cost h_init h_rem h_books h_spent h_avg
  sorry

end freds_average_book_cost_l684_684315


namespace largest_M_in_base_7_l684_684794

-- Define the base and the bounds for M^2
def base : ℕ := 7
def lower_bound : ℕ := base^3
def upper_bound : ℕ := base^4

-- Define M and its maximum value.
def M : ℕ := 48

-- Define a function to convert a number to its base 7 representation
def to_base_7 (n : ℕ) : List ℕ :=
  if n == 0 then [0] else
  let rec digits (n : ℕ) : List ℕ :=
    if n == 0 then [] else (n % 7) :: digits (n / 7)
  digits n |>.reverse

-- Define the base 7 representation of 48
def M_base_7 : List ℕ := to_base_7 M

-- The main statement asserting the conditions and the solution
theorem largest_M_in_base_7 :
  lower_bound ≤ M^2 ∧ M^2 < upper_bound ∧ M_base_7 = [6, 6] :=
by
  sorry

end largest_M_in_base_7_l684_684794


namespace units_digit_of_50_factorial_is_0_l684_684170

theorem units_digit_of_50_factorial_is_0 : 
  (∃ n : ℕ, 50! ≡ n [MOD 10]) ∧ (n = 0) := sorry

end units_digit_of_50_factorial_is_0_l684_684170


namespace focus_directrix_distance_l684_684878

def parabola_focus (x : ℝ) (y : ℝ) : Prop := x^2 = -4 * y
def focus_coordinates := (0, -1 : ℝ)
def directrix := (y : ℝ) -> y = 1
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.norm (p1.1 - p2.1, p1.2 - p2.2)

theorem focus_directrix_distance : parabola_focus 0 (-1) -> distance focus_coordinates (0, 1) = 2 := 
by
  sorry

end focus_directrix_distance_l684_684878


namespace function_characterization_l684_684272

theorem function_characterization (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), f(x^4 - y^4) + 4 * f(x * y)^2 = f(x^4 + y^4)) : 
  (∀ x, f(x) = 0) ∨ (∀ x, f(x) = x^2) :=
sorry

end function_characterization_l684_684272


namespace problem_natasha_divisible_l684_684201

theorem problem_natasha_divisible 
  (Natasha_divisible_by_15 : ∀ (n : ℕ), (Natasha n → (n % 15 = 0)))
  (one_correct_boy_and_girl : ∀ boy girl other1 other2 : ℕ, one_correct boy girl other1 other2)
  (Kolya_correct : Kolya 9 → ¬divisible_by_15 9)
  (Roman_correct : ∀ n : ℕ, Roman n → (n % 2 = 0) → prime n → ¬divisible_by_15 n)
  : ∃ n : ℕ, true := by
  -- Implicitly deducing the number satisfying the conditions.
  let n := 2
  -- Expected number.
  exact Exists.intro n trivial

end problem_natasha_divisible_l684_684201


namespace rectangle_area_integer_l684_684412

/-- In the figure, AB ⟂ BC, BC ⟂ CD, and BC is tangent to the circle with center O and diameter AD. 
    Prove that when AB = 9 and CD = 4, the area of the rectangle ABCD is an integer. -/
theorem rectangle_area_integer (AB CD AD: ℝ) (h_perpendicular1: ∀ AB BC : ℝ, AB ⟂ BC) 
    (h_perpendicular2: ∀ BC CD : ℝ, BC ⟂ CD) (h_tangent: ∀ BC : ℝ, is_tangent BC (Circle O AD)) 
    (h_diameter : AD = 2 * BC) (h_AB: AB = 9) (h_CD: CD = 4):
    (AB * CD) ∈ ℤ :=
by
  -- Proof is to be provided based on the conditions
  sorry

end rectangle_area_integer_l684_684412


namespace bacteria_growth_rate_l684_684777

theorem bacteria_growth_rate
  (r : ℝ) 
  (h1 : ∃ B D : ℝ, B * r^30 = D) 
  (h2 : ∃ B D : ℝ, B * r^25 = D / 32) :
  r = 2 := 
by 
  sorry

end bacteria_growth_rate_l684_684777


namespace T_value_l684_684806

theorem T_value:
  let T := (1 / (3 - Real.sqrt 8)) - (1 / (Real.sqrt 8 - Real.sqrt 7)) + (1 / (Real.sqrt 7 - Real.sqrt 6)) - (1 / (Real.sqrt 6 - Real.sqrt 5)) + (1 / (Real.sqrt 5 - 2))
  in T = 5 :=
by
  sorry

end T_value_l684_684806


namespace locus_of_centers_of_equilateral_triangles_through_ABC_is_circle_l684_684296

variable {Point : Type}
variable [MetricSpace Point]

variables (A B C : Point)

theorem locus_of_centers_of_equilateral_triangles_through_ABC_is_circle :
  ∃ (Circle : Set Point), 
    (∀ Triangle : Set Point, 
      (is_equilateral Triangle ∧ ∀ {x : Point}, x ∈ Triangle → x ∈ ({A, B, C} : Set Point)) 
      → center Triangle ∈ Circle)
:= sorry

end locus_of_centers_of_equilateral_triangles_through_ABC_is_circle_l684_684296


namespace exists_x0_l684_684835

noncomputable def f (x : Real) (a : Real) : Real :=
  Real.exp x - a * Real.sin x

theorem exists_x0 (a : Real) (h : a = 1) :
  ∃ x0 ∈ Set.Ioo (-Real.pi / 2) 0, 1 < f x0 a ∧ f x0 a < Real.sqrt 2 :=
  sorry

end exists_x0_l684_684835


namespace find_x_values_l684_684372

theorem find_x_values (x k : ℝ) 
  (h1 : log 10 (3 * x^2 - 5 * x + k) = 2) 
  (h2 : k = 2 * x + 5) : 
  x = 37 / 6 ∨ x = -31 / 6 := 
by
  sorry

end find_x_values_l684_684372


namespace first_person_always_wins_l684_684827

noncomputable section

def polynomial_has_three_integer_roots (a b c : ℤ) : Prop :=
  ∃ x y z : ℤ, (x^3 + a*x^2 + b*x + c = 0) ∧ (y^3 + a*y^2 + b*y + c = 0) ∧ (z^3 + a*z^2 + b*z + c = 0)

theorem first_person_always_wins :
  ∀ (A : ℤ), ∀ (B : ℤ) (B ≠ 0), ∃ C : ℤ, polynomial_has_three_integer_roots A B C :=
by
  intros A B hB
  use -A  -- or could alternatively use -B if B is meant to be the chosen non-zero integer 
  sorry

end first_person_always_wins_l684_684827


namespace solve_equation_l684_684481

theorem solve_equation (x : ℝ) : (x - 1) * (x + 3) = 5 ↔ x = 2 ∨ x = -4 := by
  sorry

end solve_equation_l684_684481


namespace problem_solution_l684_684310

-- Definitions of the homotheties
def T (k : ℕ) : ℝ × ℝ := (k * (k + 1), 0)

def homothety (k : ℕ) (P : ℝ × ℝ) : ℝ × ℝ :=
  let r := if k % 2 = 1 then 1 / 2 else 2 / 3
  let T_k := T k
  ((P.1 - T_k.1) * r + T_k.1, P.2 * r)

def apply_homothety_sequence (P : ℝ × ℝ) : ℝ × ℝ :=
  homothety 4 (homothety 3 (homothety 2 (homothety 1 P)))

theorem problem_solution (x y : ℝ) (P : ℝ × ℝ) (hP : P = (x, y)) :
  apply_homothety_sequence P = (20, 20) → x + y = 256 := by
  sorry

end problem_solution_l684_684310


namespace max_marked_cells_no_same_strip_n10_max_marked_cells_no_same_strip_n9_l684_684279

-- Declare the equilateral triangle and the conditions about the markings
structure EquilateralTriangle (n : ℕ) :=
  (segments : ℕ)
  (cells : ℕ := n * n)
  (strips : ℕ := 3 * n (* each side's parallel stripes *))

-- The math problem's proof statement for n = 10
theorem max_marked_cells_no_same_strip_n10 : 
  ∀ (T : EquilateralTriangle 10), 
  ∃ (max_marked : ℕ), max_marked = 7 := 
by
  sorry

-- The math problem's proof statement for n = 9
theorem max_marked_cells_no_same_strip_n9 :
  ∀ (T : EquilateralTriangle 9),
  ∃ (max_marked : ℕ), max_marked = 6 := 
by
  sorry

end max_marked_cells_no_same_strip_n10_max_marked_cells_no_same_strip_n9_l684_684279


namespace find_m_plus_n_l684_684969

def cone (radius height : ℝ) := 
  (volume : ℝ := (1/3) * Real.pi * radius^2 * height,
   surface_area : ℝ := Real.pi * radius^2 + Real.pi * radius * (Real.sqrt (radius^2 + height^2)))

def smaller_cone (x : ℝ) (ratio : ℝ) := 
  (volume : ℝ := (1/3) * Real.pi * x^2 * (h * x / r),
   surface_area : ℝ := Real.pi * x^2 + Real.pi * x * (Real.sqrt ((x^2 * h^2 / r^2) + (x^2))))

def frustum_surface_area (total_surface_area smaller_surface_area : ℝ) := 
  total_surface_area - smaller_surface_area

def frustum_volume (total_volume smaller_volume : ℝ) := 
  total_volume - smaller_volume

theorem find_m_plus_n 
  (r h : ℝ) 
  (total_volume : ℝ := cone r h .volume) 
  (total_surface_area : ℝ := cone r h .surface_area) 
  (x : ℝ) 
  (smaller_volume : ℝ := smaller_cone x r .volume) 
  (smaller_surface_area : ℝ := smaller_cone x r .surface_area)
  (frustum_surface_area := frustum_surface_area total_surface_area smaller_surface_area)
  (frustum_volume := frustum_volume total_volume smaller_volume)
  (k : ℚ)
  (ratio_eq : k = (smaller_volume / frustum_volume) ∧ k = (smaller_surface_area / frustum_surface_area))
  (m n : ℕ)
  (rel_prime : Nat.coprime m n)
  (k_def : k = m / n) :
  m + n = 512 := 
sorry

end find_m_plus_n_l684_684969


namespace units_digit_factorial_50_is_0_l684_684148

theorem units_digit_factorial_50_is_0 : (nat.factorial 50) % 10 = 0 := by
  sorry

end units_digit_factorial_50_is_0_l684_684148


namespace experiment_procedure_arrangements_count_l684_684248

theorem experiment_procedure_arrangements_count :
  let procedures := ["A", "B", "C", "D", "E"]
  let constrained_arrangements := 
  -- A is either the first or the last step.
  {arr | (arr.head = "A" ∨ arr.last = "A") ∧ 
         -- C and D are consecutive.
         ((/"CD" /∈ arr ∧ "DC" /∈ arr) )}
  -- Counting the number of valid arrangements.
  count constrained_arrangements procedures = 24 := 
sorry

end experiment_procedure_arrangements_count_l684_684248


namespace train_speed_l684_684942

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor: ℝ)
  (h_length : length = 100) 
  (h_time : time = 5) 
  (h_conversion : conversion_factor = 3.6) :
  (length / time * conversion_factor) = 72 :=
by
  sorry

end train_speed_l684_684942


namespace find_x_l684_684669

variables (x : ℝ)
def a : ℝ × ℝ := (-1, 3)
def b : ℝ × ℝ := (x + 1, -4)

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def is_parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 - u.2 * v.1 = 0

theorem find_x (h_parallel : is_parallel (vector_add a b) b) : x = 1 / 3 :=
by 
  sorry

end find_x_l684_684669


namespace zero_in_interval_l684_684612

noncomputable def f (x : ℝ) : ℝ := 2 * x - 4 + 3^x

theorem zero_in_interval : ∃ c ∈ Set.Ioo (1/2 : ℝ) 1, f c = 0 :=
by
  have h1 : f (1/2) < 0 := calc
    f (1/2) = 2 * (1/2) - 4 + 3^(1/2) : rfl
         ... = -3 + Real.sqrt 3      : by norm_num
         ... < 0                     : by linarith [Real.sqrt_pos.mpr (by norm_num)]
  have h2 : f 1 > 0 := calc
    f 1 = 2 * 1 - 4 + 3 : rfl
        ... = 1         : by norm_num
        ... > 0         : by norm_num
  apply exists_Ioo_of_exists_Ico _ h1 h2
  exact continuous_id.smul continuous_const.add (continuous_pow 3).continuous_on
  sorry

end zero_in_interval_l684_684612


namespace incorrect_propositions_l684_684245

-- Definitions based on conditions
def no_common_points (l1 l2 : Line) : Prop := ¬∃ p : Point, p ∈ l1 ∧ p ∈ l2
def intersect_skew_lines (l1 l2 sl1 sl2 : Line) : Prop :=
  (∃ p1, p1 ∈ l1 ∧ p1 ∈ sl1) ∧ (∃ p2, p2 ∈ l2 ∧ p2 ∈ sl2) ∧ no_common_points l1 l2
def parallel_to_skew (l : Line) (s1 s2 : Line) : Prop := 
  (∃ p1, p1 ∈ s1) ∧ (∃ p2, p2 ∈ s2) ∧ ¬Parallel l s1 ∧ ¬Parallel l s2
def intersect_with_skew_lines (l : Line) (s1 s2 : Line) : Prop :=
  (∃ p1, p1 ∈ l ∧ p1 ∈ s1) ∧ (∃ p2, p2 ∈ l ∧ p2 ∈ s2)

-- Proposition Correctness Statements
def incorrect_proposition_1 (l1 l2 : Line) : Prop :=
  no_common_points l1 l2 → ¬Parallel l1 l2 → Skew l1 l2

def incorrect_proposition_2 (l1 l2 sl1 sl2 : Line) : Prop :=
  intersect_skew_lines l1 l2 sl1 sl2 → no_common_points l1 l2

def proposition_3 (l : Line) (s1 s2 : Line) : Prop :=
  Parallel l s1 → ¬Parallel l s2

def proposition_4 (l : Line) (s1 s2 : Line) : Prop :=
  intersect_with_skew_lines l s1 s2 → True  -- Simplified representation

-- The theorem to be proved: which propositions are incorrect
theorem incorrect_propositions (l1 l2 sl1 sl2 l : Line) (s1 s2 : Line) :
  incorrect_proposition_1 l1 l2 ∨ incorrect_proposition_2 l1 l2 sl1 sl2 :=
begin
sorry
end

end incorrect_propositions_l684_684245


namespace strictly_increasing_interval_l684_684273

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 9)

theorem strictly_increasing_interval :
  ∀ x : ℝ, (3 < x) → StrictlyMonotonic (fun x => f x) (3, +∞) :=
sorry -- proof is not required

end strictly_increasing_interval_l684_684273


namespace isosceles_triangle_perimeter_l684_684766

-- Definitions based on the conditions
def isosceles_triangle (a b : ℕ) : Prop := (a = b) ∨ (a ≠ b ∧ (a = 4 ∧ b = 8) ∨ (a = 8 ∧ b = 4))

-- Statement of the problem
theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), isosceles_triangle a b ∧ (a + b + c = 20) := by
  have a_eq : 4 ∧ b_eq : 8 := by sorry  -- These would represent the side lengths considered in the problem.
  exact ⟨4, 8, 8, ⟨a_eq, b_eq⟩, by sorry⟩

end isosceles_triangle_perimeter_l684_684766


namespace value_of_Y_l684_684371

def P : ℝ := 3012 / 4
def Q : ℝ := P / 2
def Y : ℝ := P - Q

theorem value_of_Y : Y = 376.5 :=
by
  sorry

end value_of_Y_l684_684371


namespace amount_paid_correct_l684_684062

-- Defining the conditions and constants
def hourly_rate : ℕ := 60
def hours_per_day : ℕ := 3
def total_days : ℕ := 14

-- The proof statement
theorem amount_paid_correct : hourly_rate * hours_per_day * total_days = 2520 := by
  sorry

end amount_paid_correct_l684_684062


namespace num_possibilities_for_asima_integer_l684_684250

theorem num_possibilities_for_asima_integer (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 65) :
  ∃ (n : ℕ), n = 64 :=
by
  sorry

end num_possibilities_for_asima_integer_l684_684250


namespace sum_of_coefficients_of_f_l684_684894

theorem sum_of_coefficients_of_f :
  ∃ (f g : ℝ[x]), (∀ x : ℝ, (1 + complex.I * x) ^ 2001 = f.eval x + complex.I * g.eval x) →
  f.eval 1 = 2^1000 :=
  begin
    sorry
  end

end sum_of_coefficients_of_f_l684_684894


namespace anya_hair_growth_l684_684986

theorem anya_hair_growth (wash_loss : ℕ) (brush_loss : ℕ) (total_loss : ℕ) : wash_loss = 32 → brush_loss = wash_loss / 2 → total_loss = wash_loss + brush_loss → total_loss + 1 = 49 :=
by
  sorry

end anya_hair_growth_l684_684986


namespace area_of_trapezoid_EFGH_l684_684179

-- Define the vertices of the trapezoid
structure Point where
  x : ℤ
  y : ℤ

def E : Point := ⟨-2, -3⟩
def F : Point := ⟨-2, 2⟩
def G : Point := ⟨4, 5⟩
def H : Point := ⟨4, 0⟩

-- Define the formula for the area of a trapezoid
def trapezoid_area (b1 b2 height : ℤ) : ℤ :=
  (b1 + b2) * height / 2

-- The proof statement
theorem area_of_trapezoid_EFGH : trapezoid_area (F.y - E.y) (G.y - H.y) (G.x - E.x) = 30 := by
  sorry -- proof not required

end area_of_trapezoid_EFGH_l684_684179


namespace count_wonderful_two_digit_numbers_l684_684031

def is_wonderful (n : ℕ) : Prop :=
  ∃ (k p : ℕ), odd (p ∧ prime p) ∧ n = 2^k * p^2 ∧ ∃ (d1 d2 d3 : ℕ), {d1, d2, d3} = {1, p, p^2}

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem count_wonderful_two_digit_numbers :
  card {n : ℕ | is_wonderful n ∧ is_two_digit n} = 7 :=
sorry

end count_wonderful_two_digit_numbers_l684_684031


namespace average_chemistry_mathematics_l684_684899

variable {P C M : ℝ}

theorem average_chemistry_mathematics (h : P + C + M = P + 150) : (C + M) / 2 = 75 :=
by
  sorry

end average_chemistry_mathematics_l684_684899


namespace g_can_achieve_max_l684_684884

noncomputable def f (x : ℝ) (ϕ : ℝ) := 3 * Real.sin (3 * x + ϕ)
noncomputable def g (x : ℝ) (ϕ : ℝ) := 2 * Real.cos (2 * x + ϕ)

theorem g_can_achieve_max
  {a b ϕ : ℝ} 
  (h_f_increasing : ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b → f x₁ ϕ < f x₂ ϕ)
  (h_f_at_a : f a ϕ = -2)
  (h_f_at_b : f b ϕ = 2) :
  ∃ c ∈ set.Icc a b, ∀ x ∈ set.Icc a b, g x ϕ ≤ g c ϕ :=
by
  sorry

end g_can_achieve_max_l684_684884


namespace arithmetic_sequence_and_sum_properties_l684_684813

noncomputable def a_n (n : ℕ) : ℤ := 30 - 2 * n
noncomputable def S_n (n : ℕ) : ℤ := -n^2 + 29 * n

theorem arithmetic_sequence_and_sum_properties :
  (a_n 3 = 24 ∧ a_n 6 = 18) ∧
  (∀ n : ℕ, (S_n n = (n * (a_n 1 + a_n n)) / 2) ∧ ((a_n 3 = 24 ∧ a_n 6 = 18) → ∀ n : ℕ, a_n n = 30 - 2 * n)) ∧
  (S_n 14 = 210) :=
by 
  -- Proof omitted.
  sorry

end arithmetic_sequence_and_sum_properties_l684_684813


namespace constant_term_in_expansion_l684_684774

theorem constant_term_in_expansion (a : ℝ) : a + 1 = 4 → 
  (let s : ℝ := (x + a / x) * (2 * x - 1 / x) ^ 5 in 
   constant_term_in_expansion s = 200) :=
begin
  sorry
end

end constant_term_in_expansion_l684_684774


namespace worm_length_l684_684861

theorem worm_length (l1 l2 : ℝ) (h1 : l1 = 0.8) (h2 : l2 = l1 + 0.7) : l1 = 0.8 :=
by
  exact h1

end worm_length_l684_684861


namespace range_of_f_l684_684748

theorem range_of_f (φ : ℝ) (hφ : φ > 0) : ∃ a : ℝ, (a ∈ [-2, 1)) ∧ (∃ x0 : ℝ, x0 ∈ (0, π / 2) ∧ f x0 = a) :=
  sorry

def f (x : ℝ) : ℝ :=
  2 * cos (2 * x + φ)

end range_of_f_l684_684748


namespace bg_over_gh_l684_684752

-- Definitions and Conditions
variable {A B C G H P : Type}
variable [linear_ordered_field ℝ]

-- Points G and H are on line segments AB and BC, respectively
variable G_on_AB : G ∈ ℝ 
variable H_on_BC : H ∈ ℝ 

-- Intersection ratios
variable (AP PG CP PH : ℝ)
variable h₁ : AP / PG = 5
variable h₂ : CP / PH = 3

-- Conclusion to be proven
theorem bg_over_gh : ∀ (BG GH : ℝ), (BG / GH = 3 / 2) :=
by
  sorry

end bg_over_gh_l684_684752


namespace cereal_consumption_time_l684_684036

theorem cereal_consumption_time:
  let rate_fat := (1 : ℝ) / 25
  let rate_thin := (1 : ℝ) / 35
  let combined_rate := rate_fat + rate_thin
  let total_cereal := 5
  let time := total_cereal / combined_rate
  time ≈ 73 :=
by sorry

end cereal_consumption_time_l684_684036


namespace center_of_symmetry_for_cubic_l684_684665

-- Define the cubic function f(x)
def f (x : ℝ) := x^3 - 3*x^2 + 3*x

-- Define the second derivative of f
noncomputable def f'' (x : ℝ) := deriv (deriv f x)

-- Define the center of symmetry based on f''(x) = 0 and (x0, f x0) being the center
noncomputable def center_of_symmetry (x0 : ℝ) : Prop :=
  f'' x0 = 0 ∧ (x0, f x0) = (1, 1)

-- Prove the center of symmetry statement for f(x)
theorem center_of_symmetry_for_cubic : ∃ x0, center_of_symmetry x0 :=
by
  sorry

end center_of_symmetry_for_cubic_l684_684665


namespace quadratic_solution_l684_684376

theorem quadratic_solution (x : ℝ) (h1 : x^2 - 6 * x + 8 = 0) (h2 : x ≠ 0) :
  x = 2 ∨ x = 4 :=
sorry

end quadratic_solution_l684_684376


namespace socks_combinations_correct_l684_684907

noncomputable def num_socks_combinations (colors patterns pairs : ℕ) : ℕ :=
  colors * (colors - 1) * patterns * (patterns - 1)

theorem socks_combinations_correct :
  num_socks_combinations 5 4 20 = 240 :=
by
  sorry

end socks_combinations_correct_l684_684907


namespace find_B_l684_684851

noncomputable def A : ℝ := 1 / 49
noncomputable def C : ℝ := -(1 / 7)

theorem find_B :
  (∀ x : ℝ, 1 / (x^3 + 2 * x^2 - 25 * x - 50) 
            = (A / (x - 2)) + (B / (x + 5)) + (C / ((x + 5)^2))) 
    → B = - (11 / 490) :=
sorry

end find_B_l684_684851


namespace sum_binom_remainder_l684_684807

theorem sum_binom_remainder :
  let T := ∑ n in Finset.range 1002, (-1)^n * Nat.choose 3003 (3 * n)
  T % 1000 = 6 :=
by
  -- Proof would be inserted here
  sorry

end sum_binom_remainder_l684_684807


namespace quadrilateral_area_l684_684830

theorem quadrilateral_area (EFGH : Type) [Plane EFGH] (square_side : ℝ)
  (P Q R S : Point EFGH) (EPF FQG GRH HSE : Triangle EFGH)
  (h1 : is_equilateral_triangle EPF)
  (h2 : is_equilateral_triangle FQG)
  (h3 : is_equilateral_triangle GRH)
  (h4 : is_equilateral_triangle HSE)
  (h5 : side_length EFGH = 8)
  (h6 : on_side P EFGH)
  (h7 : on_side Q FGH)
  (h8 : on_side R GHF)
  (h9 : on_side S HFE) :
  area_quadrilateral PQRS = 48 := sorry

end quadrilateral_area_l684_684830


namespace gray_area_correct_l684_684540

-- Define the conditions of the problem
def diameter_small : ℝ := 6
def radius_small : ℝ := diameter_small / 2
def radius_large : ℝ := 3 * radius_small

-- Define the areas based on the conditions
def area_small : ℝ := Real.pi * radius_small^2
def area_large : ℝ := Real.pi * radius_large^2
def gray_area : ℝ := area_large - area_small

-- Write the theorem that proves the required area of the gray region
theorem gray_area_correct : gray_area = 72 * Real.pi :=
by
  sorry

end gray_area_correct_l684_684540


namespace prob1_prob2_l684_684260

-- Definition and theorems related to the calculations of the given problem.
theorem prob1 : ((-12) - 5 + (-14) - (-39)) = 8 := by 
  sorry

theorem prob2 : (-2^2 * 5 - (-12) / 4 - 4) = -21 := by
  sorry

end prob1_prob2_l684_684260


namespace transportation_cost_correct_l684_684865

-- Define the cost per kilogram constant
def cost_per_kg : ℝ := 18000

-- Define the weight of the instrument in grams
def weight_g : ℝ := 400

-- Convert the weight to kilograms
def weight_kg : ℝ := weight_g / 1000

-- Define the correct answer to the problem (expected transportation cost)
def expected_cost : ℝ := 7200

-- Define the theorem stating that the cost of transportation is equal to the expected cost
theorem transportation_cost_correct :
  (weight_kg * cost_per_kg) = expected_cost :=
by sorry

end transportation_cost_correct_l684_684865


namespace compare_M_N_l684_684672

variable {a b : ℝ}

theorem compare_M_N (h_a : 0 < a ∧ a < 1) (h_b : 0 < b ∧ b < 1) :
  let M := a * b
  let N := a + b - 1
  in M > N :=
by
  sorry

end compare_M_N_l684_684672


namespace min_value_at_2_l684_684504

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_at_2 : ∃ x : ℝ, f x = 2 :=
sorry

end min_value_at_2_l684_684504


namespace tan_15_simplification_l684_684839

theorem tan_15_simplification :
  (1 + Real.tan (Real.pi / 12)) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
by
  sorry

end tan_15_simplification_l684_684839


namespace expected_cards_drawn_l684_684983

theorem expected_cards_drawn 
  (deck_size : ℕ) 
  (distinct_factors : set ℕ) 
  (total_factors : distinct_factors.card = deck_size ∧ ∀ x ∈ distinct_factors, ∃ p, x | 2002 ∧ nat.prime p ∧ p ∈ {2, 7, 11, 13}) 
  (draw_until_perfect_square : ∀ (drawn_cards : finset ℕ), 
    ∃ subset (S : finset ℕ) (hS : S ≠ ∅), 
    (drawn_cards ⊆ distinct_factors) → ((S.prod id : ℕ) ^ 2) ∣ (drawn_cards.prod id)^2 ) :
  deck_size = 16 → 
  4.0625 :=
by open_locale big_operators sorry

end expected_cards_drawn_l684_684983


namespace triangle_third_side_l684_684400

theorem triangle_third_side (a b : ℝ) (x : ℝ) (h₀ : a = 3) (h₁ : b = 7) 
  (h₂ : 0 < x) (triangle_ineq : a + b > x ∧ a + x > b ∧ b + x > a) : 
  4 < x ∧ x < 10 := 
by 
  have h₃ : 3 + 7 > x := by linarith [triangle_ineq.left]
  have h₄ : x > 4 := by linarith [triangle_ineq.right.left, triangle_ineq.right.right]
  exact ⟨h₄, h₃⟩

end triangle_third_side_l684_684400


namespace farmer_picked_potatoes_l684_684211

noncomputable def initial_tomatoes := 175
noncomputable def initial_potatoes := 77
noncomputable def remaining_total := 80
noncomputable def picked_potatoes := 172

theorem farmer_picked_potatoes :
  let total_initial := initial_tomatoes + initial_potatoes in
  let total_picked := total_initial - remaining_total in
  total_picked = picked_potatoes :=
by
  sorry

end farmer_picked_potatoes_l684_684211


namespace number_of_poles_l684_684818

theorem number_of_poles (side_length : ℝ) (distance_between_poles : ℝ) 
  (h1 : side_length = 150) (h2 : distance_between_poles = 30) : 
  ((4 * side_length) / distance_between_poles) = 20 :=
by 
  -- Placeholder to indicate missing proof
  sorry

end number_of_poles_l684_684818


namespace production_exceeds_60000_in_2022_l684_684210

/-- 
Given:
1. The factory's annual production in 2015 is 20,000 units.
2. The production increases by 20% every year starting from 2016.
3. \(\log_{10}(2) = 0.3010\)
4. \(\log_{10}(3) = 0.4771\)

Prove that:
The annual production will exceed 60,000 units in the year 2022.
-/
theorem production_exceeds_60000_in_2022 (log10_2 : ℝ) (log10_3 : ℝ) 
  (log10_2_eq : log10_2 = 0.3010) (log10_3_eq : log10_3 = 0.4771) : 
  ∃ n : ℕ, 2015 + n = 2022 ∧ 20000 * (1.2 ^ n) > 60000 := 
by 
  sorry

end production_exceeds_60000_in_2022_l684_684210


namespace all_numbers_positive_l684_684467

noncomputable def condition (a : Fin 9 → ℝ) : Prop :=
  ∀ (S : Finset (Fin 9)), S.card = 4 → S.sum (a : Fin 9 → ℝ) < (Finset.univ \ S).sum (a : Fin 9 → ℝ)

theorem all_numbers_positive (a : Fin 9 → ℝ) (h : condition a) : ∀ i, 0 < a i :=
by
  sorry

end all_numbers_positive_l684_684467


namespace total_arrangements_l684_684585

-- Define the primary days of the week
inductive Day
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day

-- Schools involved in the survey
inductive School
| A : School
| B : School
| C : School
| D : School
| E : School
| F : School
| G : School

-- Define conditions
def is_week (days : List Day) : Prop := 
  days = [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday]

def at_least_one_school_per_day (schedule : Day → List School) : Prop :=
  ∀ d, schedule d ≠ []

def school_A_on_monday (schedule : Day → List School) : Prop :=
  School.A ∈ schedule Day.Monday

def school_B_on_tuesday (schedule : Day → List School) : Prop :=
  School.B ∈ schedule Day.Tuesday

def school_C_and_D_same_day (schedule : Day → List School) : Prop :=
  ∃ (d : Day), School.C ∈ schedule d ∧ School.D ∈ schedule d

def school_E_not_on_friday (schedule : Day → List School) : Prop :=
  School.E ∉ schedule Day.Friday

-- The main goal is to prove that the number of different arrangements satisfying all conditions is 60
theorem total_arrangements :
  ∃ (schedule : Day → List School),
    is_week [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday] ∧
    at_least_one_school_per_day schedule ∧
    school_A_on_monday schedule ∧
    school_B_on_tuesday schedule ∧
    school_C_and_D_same_day schedule ∧
    school_E_not_on_friday schedule ∧
    (some function to count schedules) = 60 :=
sorry

end total_arrangements_l684_684585


namespace number_of_successful_arrangements_l684_684767

-- Definition of successful arrangement in a square grid of size (2^n - 1) x (2^n - 1).
def is_successful {n : ℕ} (grid : Array (Array Int)) : Prop :=
  ∀ i j : ℕ, i < (2^n - 1) → j < (2^n - 1) →
    grid[i][j] = (if i > 0 then grid[i-1][j] else 1) *
                (if i < (2^n - 2) then grid[i+1][j] else 1) *
                (if j > 0 then grid[i][j-1] else 1) *
                (if j < (2^n - 2) then grid[i][j+1] else 1)

-- Definition of the problem statement.
theorem number_of_successful_arrangements {n : ℕ} : 
  ∃ grid, (∀ i j, i < (2^n - 1) → j < (2^n - 1) → grid[i][j] = 1)
  ∧ (is_successful grid) :=
sorry

end number_of_successful_arrangements_l684_684767


namespace units_digit_factorial_50_is_0_l684_684149

theorem units_digit_factorial_50_is_0 : (nat.factorial 50) % 10 = 0 := by
  sorry

end units_digit_factorial_50_is_0_l684_684149


namespace problem_statement_l684_684799

def seriesWithBase2Powers_sum : ℚ :=
  ∑' n, (2 + 4 * n : ℚ) / (2 ^ (2 + 2 * n))

def seriesWithBase3Powers_sum : ℚ :=
  ∑' n, (4 + 4 * n : ℚ) / (3 ^ (1 + 2 * n))

theorem problem_statement : 
  ∀ a b : ℕ, Nat.coprime a b ∧ (↑a / ↑b = seriesWithBase2Powers_sum + seriesWithBase3Powers_sum) →
  a + b = 83 :=
by
  sorry

end problem_statement_l684_684799


namespace total_people_present_l684_684988

theorem total_people_present (A B : ℕ) 
  (h1 : 2 * A + B = 10) 
  (h2 : A + 2 * B = 14) :
  A + B = 8 :=
sorry

end total_people_present_l684_684988


namespace division_value_unique_l684_684921

theorem division_value_unique :
  ∃ x : ℝ, (5.5 / x) * 12 = 11 ∧ x = 6 :=
begin
  use 6,
  split,
  { have hx : (5.5 / 6) * 12 = 11, by norm_num,
    exact hx, },
  refl,
end

end division_value_unique_l684_684921


namespace collinear_abd_l684_684030

-- Definitions of vectors
variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (e₁ e₂ : V) (A B C D : V)
variables (nonzero_e₁ : e₁ ≠ 0) (nonzero_e₂ : e₂ ≠ 0) (lin_indep : ¬(e₁ = e₂))

-- Given conditions
def ab : V := e₁ + 2 • e₂
def bc : V := 2 • e₁ + 7 • e₂
def cd : V := 3 • (e₁ + e₂)

-- Definition of collinearity
def collinear (u v w : V) : Prop :=
  ∃ (r : ℝ), u + r • v = w

-- Statement to be proved
theorem collinear_abd : collinear ab (bc + cd) 0 := by
  sorry

end collinear_abd_l684_684030


namespace rectangle_length_increase_decrease_l684_684598

theorem rectangle_length_increase_decrease
  (L : ℝ)
  (width : ℝ)
  (increase_percentage : ℝ)
  (decrease_percentage : ℝ)
  (new_width : ℝ)
  (initial_area : ℝ)
  (new_length : ℝ)
  (new_area : ℝ)
  (HLW : width = 40)
  (Hinc : increase_percentage = 0.30)
  (Hdec : decrease_percentage = 0.17692307692307693)
  (Hnew_width : new_width = 40 - (decrease_percentage * 40))
  (Hinitial_area : initial_area = L * 40)
  (Hnew_length : new_length = 1.30 * L)
  (Hequal_area : new_length * new_width = L * 40) :
  L = 30.76923076923077 :=
by
  sorry

end rectangle_length_increase_decrease_l684_684598


namespace option_B_correct_l684_684927

theorem option_B_correct : 1 ∈ ({0, 1} : Set ℕ) := 
by
  sorry

end option_B_correct_l684_684927


namespace dot_product_zero_l684_684431

-- Define the graph of the function
def function_graph (x : ℝ) : ℝ := x + 2 / x

-- P is a point on the graph
variable (x : ℝ) (hx : x > 0)

-- Define points P, A, B
def P : ℝ × ℝ := (x, function_graph x)
def A : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
def B : ℝ × ℝ := (0, function_graph x)

-- Define vectors PA and PB
def vec_PA : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)
def vec_PB : ℝ × ℝ := (B.1 - P.1, B.2 - P.2)

-- Function to calculate dot product
def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

-- The final proof statement
theorem dot_product_zero : dot_product (vec_PA x) (vec_PB x) = 0 :=
by
  sorry

end dot_product_zero_l684_684431


namespace larry_channel_reduction_l684_684006

theorem larry_channel_reduction
  (initial_channels new_channels final_channels sports_package supreme_sports_package channels_at_end : ℕ)
  (h_initial : initial_channels = 150)
  (h_adjustment : new_channels = initial_channels - 20 + 12)
  (h_sports : sports_package = 8)
  (h_supreme_sports : supreme_sports_package = 7)
  (h_channels_at_end : channels_at_end = 147)
  (h_final : final_channels = channels_at_end - sports_package - supreme_sports_package) :
  initial_channels - 20 + 12 - final_channels = 10 := 
sorry

end larry_channel_reduction_l684_684006


namespace eccentricity_is_half_l684_684719

noncomputable def eccentricity_of_ellipse (x : ℝ) (y : ℝ) : ℝ :=
let F1 := (-1, 0) in
let F2 := (1, 0) in
let a := 2 in
let c := 1 in
-- Since |PF1| + |PF2| = 4, P is on the ellipse with foci F1 and F2
-- The eccentricity e = c / a
c / a

theorem eccentricity_is_half (x : ℝ) (y : ℝ) :
  let F1 := (-1, 0) in
  let F2 := (1, 0) in
  let a := 2 in
  let c := 1 in
  2 * dist F1 F2 = dist (x, y) F1 + dist (x, y) F2 → 
  eccentricity_of_ellipse x y = 1 / 2 :=
by
  intros F1 F2 a c h
  rw [dist_eq (x, y) F1 (x, y) F2, dist_eq F1 F2] at h
  sorry

end eccentricity_is_half_l684_684719


namespace midpoint_A_l684_684093

-- Define the initial points A, J, H
def A : ℝ × ℝ := (2, 2)
def J : ℝ × ℝ := (3, 5)
def H : ℝ × ℝ := (6, 2)

-- Define the translated points A', J', H'
def A' : ℝ × ℝ := (A.1 - 6, A.2 + 3)
def J' : ℝ × ℝ := (J.1 - 6, J.2 + 3)
def H' : ℝ × ℝ := (H.1 - 6, H.2 + 3)

-- Define the midpoint of A'H'
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Prove the midpoint of A' and H' is (-2, 5)
theorem midpoint_A'_H' : midpoint A' H' = (-2, 5) :=
by
  have A'_coords : A' = (-4, 5) := rfl
  have H'_coords : H' = (0, 5) := rfl
  rw [A'_coords, H'_coords]
  exact (rfl)


end midpoint_A_l684_684093


namespace max_marks_l684_684970

theorem max_marks (M : ℝ) (pass_percent : ℝ) (obtained_marks : ℝ) (failed_by : ℝ) (pass_marks : ℝ) 
  (h1 : pass_percent = 0.40) 
  (h2 : obtained_marks = 150) 
  (h3 : failed_by = 50) 
  (h4 : pass_marks = 200) 
  (h5 : pass_marks = obtained_marks + failed_by) 
  : M = 500 :=
by 
  -- Placeholder for the proof
  sorry

end max_marks_l684_684970


namespace units_digit_of_product_1_to_50_is_zero_l684_684154

theorem units_digit_of_product_1_to_50_is_zero :
  Nat.digits 10 (∏ i in Finset.range 51, i) = [0] :=
sorry

end units_digit_of_product_1_to_50_is_zero_l684_684154


namespace find_d_minus_r_l684_684375

theorem find_d_minus_r :
  ∃ d r : ℕ, d > 1 ∧ (1059 % d = r) ∧ (1417 % d = r) ∧ (2312 % d = r) ∧ (d - r = 15) :=
sorry

end find_d_minus_r_l684_684375


namespace probability_of_forming_phrase_l684_684524

theorem probability_of_forming_phrase :
  let cards := ["中", "国", "梦"]
  let n := 6
  let m := 1
  ∃ (p : ℚ), p = (m / n : ℚ) ∧ p = 1 / 6 :=
by
  sorry

end probability_of_forming_phrase_l684_684524


namespace coeff_x6_expansion_l684_684120

theorem coeff_x6_expansion : 
  (3 - 4 * x ^ 3) ^ 4 = (864 * x ^ 6 + ...) := 
sorry

end coeff_x6_expansion_l684_684120


namespace table_height_is_five_l684_684240

def height_of_table (l h w : ℕ) : Prop :=
  l + h + w = 45 ∧ 2 * w + h = 40

theorem table_height_is_five (l w : ℕ) : height_of_table l 5 w :=
by
  sorry

end table_height_is_five_l684_684240


namespace max_food_per_guest_l684_684989

theorem max_food_per_guest (total_food : ℝ) (min_guests : ℕ) (h1 : total_food = 325) (h2 : min_guests = 163) :
  total_food / min_guests ≈ 2 :=
by 
  simp [h1, h2, Real.div_eq]
  sorry

end max_food_per_guest_l684_684989


namespace sum_interior_numbers_sum_interior_row_8_sum_interior_row_9_l684_684660

theorem sum_interior_numbers (n : ℕ) : (2^(n-1) - 2) = sum_interior_of_row_n :=
by sorry

def sum_interior_of_row_8 : ℕ := 126
def sum_interior_of_row_9 : ℕ := 254

theorem sum_interior_row_8 : sum_interior_numbers 8 = sum_interior_of_row_8 := by
sorry

theorem sum_interior_row_9 : sum_interior_numbers 9 = sum_interior_of_row_9 := by
sorry

end sum_interior_numbers_sum_interior_row_8_sum_interior_row_9_l684_684660


namespace train_stops_14_4_min_per_hour_l684_684929

def distance (D : ℕ) : Prop :=
  let t_no_stop := D / 250 in
  let t_with_stop := D / 125 in
  let t_stop := t_with_stop - t_no_stop in
  let t_stop_min := t_stop * 60 in
  t_stop_min = 14.4

theorem train_stops_14_4_min_per_hour (D : ℕ) : distance D :=
by sorry

end train_stops_14_4_min_per_hour_l684_684929


namespace soja_book_page_count_l684_684937

theorem soja_book_page_count (P : ℕ) (h1 : P > 0) (h2 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 100) : P = 300 :=
by
  -- The Lean proof is not required, so we just add sorry to skip the proof
  sorry

end soja_book_page_count_l684_684937


namespace geometric_arithmetic_sequence_property_l684_684438

-- Definitions of conditions
variables (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
variable (h4 : a * c = b * b) -- Geometric sequence condition: a/c = c/b
variable (h5 : c = 4 * a)    -- Given c = 4a

-- Statements to be proved
theorem geometric_arithmetic_sequence_property :
  a = c ^ (1/4) ∧
  ( (log c a) = ((log b c) + (log a b)) / 2
  ∧ (log b c) - (log c a) = 5 / 4) :=
by sorry

end geometric_arithmetic_sequence_property_l684_684438


namespace cone_radius_not_triple_cylinder_radius_l684_684511

/-- Define the geometric shapes -/
structure Cylinder :=
(base_radius : ℝ)
(height : ℝ)

structure Cone :=
(base_radius : ℝ)
(height : ℝ)

/-- Define the volume functions for the shapes -/
def volume_cylinder (c : Cylinder) : ℝ := π * c.base_radius ^ 2 * c.height

def volume_cone (k : Cone) : ℝ := (1 / 3) * π * k.base_radius ^ 2 * k.height

/-- Given conditions -/
noncomputable def cylinder := Cylinder.mk r h
noncomputable def cone := Cone.mk (3 * r) h

/-- The conjecture: the volumes are not equal if height is the same and r_k = 3 * r_c -/
theorem cone_radius_not_triple_cylinder_radius (r h : ℝ) (h_eq : cone.height = cylinder.height) 
  (vol_eq : volume_cylinder cylinder = volume_cone cone) : false :=
sorry

end cone_radius_not_triple_cylinder_radius_l684_684511


namespace polynomial_q_l684_684018

-- Given conditions
def poly1 := 2 * x ^ 6 + 5 * x ^ 4 + 11 * x ^ 2 + 6 * x
def poly2 := 4 * x ^ 4 + 16 * x ^ 3 + 36 * x ^ 2 + 10 * x + 4

-- The polynomial we need to prove
def q (x : ℝ) := -2 * x ^ 6 - x ^ 4 + 16 * x ^ 3 + 25 * x ^ 2 + 4 * x + 4

theorem polynomial_q (x : ℝ) : q x + poly1 = poly2 := 
by 
  calc
    q x + poly1 = ... -- add the remaining required steps    
  sorry

end polynomial_q_l684_684018


namespace non_coplanar_triangles_in_cube_l684_684368

noncomputable def binomial (n k : ℕ) : ℕ :=
if k > n then 0 else Nat.choose n k

/-- 
  Given a cube with 8 vertices, prove that the number of distinct non-coplanar triangles formed by connecting any three different vertices is 50.
-/
theorem non_coplanar_triangles_in_cube : 
  let total_triangles := binomial 8 3 in
  let coplanar_triangles := 6 in
  let non_coplanar_triangles := total_triangles - coplanar_triangles in
  non_coplanar_triangles = 50 := 
by 
  let total_triangles := binomial 8 3
  let coplanar_triangles := 6
  let non_coplanar_triangles := total_triangles - coplanar_triangles
  show non_coplanar_triangles = 50
  sorry

end non_coplanar_triangles_in_cube_l684_684368


namespace rectangle_from_circles_l684_684007

open EuclideanGeometry

theorem rectangle_from_circles
  (k : Circle ℂ)
  (k1 k2 k3 k4 : Circle ℂ)
  (O1 O2 O3 O4 : ℂ)
  (A1 A2 A3 A4 B1 B2 B3 B4 : ℂ)
  (hO1 : O1 ∈ k)
  (hO2 : O2 ∈ k)
  (hO3 : O3 ∈ k)
  (hO4 : O4 ∈ k)
  (hA1 : A1 ∈ k)
  (hA2 : A2 ∈ k)
  (hA3 : A3 ∈ k)
  (hA4 : A4 ∈ k)
  (hIntersect1 : (A1 ∈ k1) ∧ (B1 ∈ k1) ∧ (A1 ∈ k2) ∧ (B1 ∈ k2))
  (hIntersect2 : (A2 ∈ k2) ∧ (B2 ∈ k2) ∧ (A2 ∈ k3) ∧ (B2 ∈ k3))
  (hIntersect3 : (A3 ∈ k3) ∧ (B3 ∈ k3) ∧ (A3 ∈ k4) ∧ (B3 ∈ k4))
  (hIntersect4 : (A4 ∈ k4) ∧ (B4 ∈ k4) ∧ (A4 ∈ k1) ∧ (B4 ∈ k1))
  (hOrder : [O1, A1, O2, A2, O3, A3, O4, A4].pairwise (≠)) :
  parallelogram k B1 B2 B3 B4 ∧ all_right_angle B1 B2 B3 B4 :=
sorry

end rectangle_from_circles_l684_684007


namespace segment_AB_length_l684_684418

-- Let h be the height of the trapezoid

theorem segment_AB_length (h : ℝ) (AB CD : ℝ)
  (ratio_areas : 8 * (1/2) * AB * h = 2 * (1/2) * CD * h)
  (sum_length : AB + CD = 150)
  (ratio_length : AB = 3 * CD) :
  AB = 120 :=
by 
  -- Equations derived from conditions
  have area_eq := (8: ℝ) * (1/2) * AB * h = (2: ℝ) * (1/2) * CD * h,
  have length_eq1 := AB + CD = 150,
  have length_eq2 := AB = 3 * CD,
  -- Skipping all detailed steps of the proof
  sorry

end segment_AB_length_l684_684418


namespace rational_polynomials_l684_684292

-- Let p be a polynomial with real coefficients
variable (p : ℝ[X])

-- Define rationality in terms of input and output
def rational_if_and_only_if (p : ℝ[X]) : Prop :=
  ∀ x : ℝ, (∃ r : ℚ, x = r) ↔ (∃ s : ℚ, polynomial.eval x p = s)

-- The theorem statement
theorem rational_polynomials (p : ℝ[X]) :
  rational_if_and_only_if p ↔ (∃ a b : ℚ, p = polynomial.C (a:ℝ) + polynomial.C (b:ℝ) * polynomial.X) :=
begin
  sorry
end

end rational_polynomials_l684_684292


namespace michael_will_meet_two_times_l684_684816

noncomputable def michael_meetings : ℕ :=
  let michael_speed := 6 -- feet per second
  let pail_distance := 300 -- feet
  let truck_speed := 12 -- feet per second
  let truck_stop_time := 20 -- seconds
  let initial_distance := pail_distance -- feet
  let michael_position (t: ℕ) := michael_speed * t
  let truck_position (cycle: ℕ) := pail_distance * cycle
  let truck_cycle_time := pail_distance / truck_speed + truck_stop_time -- seconds per cycle
  let truck_position_at_time (t: ℕ) := 
    let cycle := t / truck_cycle_time
    let remaining_time := t % truck_cycle_time
    if remaining_time < (pail_distance / truck_speed) then 
      truck_position cycle + truck_speed * remaining_time
    else 
      truck_position cycle + pail_distance
  let distance_between := 
    λ (t: ℕ) => truck_position_at_time t - michael_position t
  let meet_time := 
    λ (t: ℕ) => if distance_between t = 0 then 1 else 0
  let total_meetings := 
    (List.range 300).map meet_time -- estimating within 300 seconds
    |> List.sum
  total_meetings

theorem michael_will_meet_two_times : michael_meetings = 2 :=
  sorry

end michael_will_meet_two_times_l684_684816


namespace hyperbola_standard_equation_l684_684079

theorem hyperbola_standard_equation (a c : ℝ) (h1 : a + c = 9) (h2 : b = 3) 
(h3 : b^2 = 9) (h4 : c^2 = a^2 + b^2) :
  (a = 4) → (c = 5) → (∀ x y : ℝ, (y^2 / 16 - x^2 / 9 = 1)) :=
begin
  sorry
end

end hyperbola_standard_equation_l684_684079


namespace min_value_expression_l684_684541

open Real

theorem min_value_expression : ∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 :=
by
  intro x y
  sorry

end min_value_expression_l684_684541


namespace sum_of_digits_of_second_multiple_lcm_l684_684454

theorem sum_of_digits_of_second_multiple_lcm : 
  let N := Nat.lcm (List.range 8).tail 
  in N * 2 = 840 → (840.digits.sum = 12) :=
by
  sorry

end sum_of_digits_of_second_multiple_lcm_l684_684454


namespace book_arrangement_l684_684573

theorem book_arrangement {n : ℕ} (h1 : n = 11) : 
  let shelves := 3 
  let total_arrangements := Nat.fact (11 + 2) / Nat.fact 2 
  total_arrangements - (shelves * Nat.fact 11) = 75 * Nat.fact 11 :=
by
  sorry

end book_arrangement_l684_684573


namespace minimum_distance_PQ_l684_684263

-- Definition of the problem conditions
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def C : ℝ × ℝ × ℝ := (1/2, Real.sqrt 3 / 2, 0)
def D : ℝ × ℝ × ℝ := (1/2, Real.sqrt 3 / 6, Real.sqrt (2 / 3))

variable (t s : ℝ)
variable (ht : 0 ≤ t ∧ t ≤ 1)
variable (hs : 0 ≤ s ∧ s ≤ 1)

def P : ℝ × ℝ × ℝ := (t, 0, 0)
def Q : ℝ × ℝ × ℝ := (1/2 * s + 1/2, Real.sqrt 3 / 2 * s, Real.sqrt (2 / 3) * s)

-- Lean 4 statement to prove the minimum distance PQ is sqrt(2)/2
theorem minimum_distance_PQ : ∃ (t s : ℝ), (0 ≤ t ∧ t ≤ 1) ∧ (0 ≤ s ∧ s ≤ 1) ∧ 
  Real.sqrt ((t - 1/2 * s - 1/2) ^ 2 + (Real.sqrt 3 / 2 * s) ^ 2 + (Real.sqrt (2 / 3) * s) ^ 2) = Real.sqrt 2 / 2 :=
by
  sorry

end minimum_distance_PQ_l684_684263


namespace multiply_1546_by_100_l684_684819

theorem multiply_1546_by_100 : 15.46 * 100 = 1546 :=
by
  sorry

end multiply_1546_by_100_l684_684819


namespace running_distance_l684_684944

theorem running_distance (D : ℕ) 
  (hA_time : ∀ (A_time : ℕ), A_time = 28) 
  (hB_time : ∀ (B_time : ℕ), B_time = 32) 
  (h_lead : ∀ (lead : ℕ), lead = 28) 
  (hA_speed : ∀ (A_speed : ℚ), A_speed = D / 28) 
  (hB_speed : ∀ (B_speed : ℚ), B_speed = D / 32) 
  (hB_dist : ∀ (B_dist : ℚ), B_dist = D - 28) 
  (h_eq : ∀ (B_dist : ℚ), B_dist = D * (28 / 32)) :
  D = 224 :=
by 
  sorry

end running_distance_l684_684944


namespace triangle_ratio_l684_684773

-- We define the geometrical configurations and properties
noncomputable def length_AB (AC BC : ℝ) : ℝ := Real.sqrt (AC^2 + BC^2)
def is_right_triangle (A B C : ℝ × ℝ) : Prop := A.x^2 + B.x^2 = C.x^2

theorem triangle_ratio 
    (AC BC AD DE DB: ℝ)
    (h1 : AC = 5) 
    (h2 : BC = 12) 
    (h3 : length_AB AC BC = 13) 
    (h4 : AD = 20) 
    (h5 : is_right_triangle (5, 12) (12, 5) (13, 0))
    (h6 : DE / DB = (12 : ℝ) / 13) :
    (12 + 13 = 25) :=
    by
    sorry

end triangle_ratio_l684_684773


namespace interior_edges_sum_l684_684964

-- Definitions based on conditions
def frame_width : ℕ := 2
def frame_area : ℕ := 32
def outer_edge_length : ℕ := 8

-- Mathematically equivalent proof problem
theorem interior_edges_sum :
  ∃ (y : ℕ),  (frame_width * 2) * (y - frame_width * 2) = 32 ∧ (outer_edge_length * y - (outer_edge_length - 2 * frame_width) * (y - 2 * frame_width)) = 32 -> 4 + 4 + 0 + 0 = 8 :=
sorry

end interior_edges_sum_l684_684964


namespace solution_set_of_inequality_l684_684657

theorem solution_set_of_inequality :
  {x : ℝ | (x^2 - 2*x - 3) * (x^2 + 1) < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end solution_set_of_inequality_l684_684657


namespace least_possible_k_l684_684745

theorem least_possible_k (k : ℤ) (h : 0.00010101 * 10^k > 10) : k ≥ 6 :=
by {
  sorry
}

end least_possible_k_l684_684745


namespace gcd_of_repeated_ones_l684_684444

theorem gcd_of_repeated_ones (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  Nat.gcd (Nat.repeat 1 m : ℕ) (Nat.repeat 1 n : ℕ) = Nat.repeat 1 (Nat.gcd m n) :=
sorry

end gcd_of_repeated_ones_l684_684444


namespace bob_sells_silver_cube_for_4455_l684_684624

noncomputable def side_length := 3 -- Condition 1: Side length of the cube
noncomputable def density := 6 -- Condition 2: Weight of a cubic inch of silver in ounces
noncomputable def price_per_ounce := 25 -- Condition 3: Price per ounce of silver in dollars
noncomputable def sale_percentage := 1.1 -- Condition 4: Selling price is 110% of the silver value

noncomputable def volume := side_length * side_length * side_length

noncomputable def weight := volume * density

noncomputable def silver_value := weight * price_per_ounce

noncomputable def selling_price := silver_value * sale_percentage

theorem bob_sells_silver_cube_for_4455 : selling_price = 4455 :=
by
  sorry

end bob_sells_silver_cube_for_4455_l684_684624


namespace solution_exist_l684_684697

noncomputable def solve_eq (x : ℝ) : Prop :=
  Real.sin x ≠ 0 ∧
  Real.cos x ≠ 0 ∧
  Real.cos (2 * x) ≠ 0 ∧
  Real.cos (4 * x) ≠ 0

theorem solution_exist (x : ℝ) (k : ℤ) :
  solve_eq x →
  (Real.cot x - Real.tan x - 2 * Real.tan (2 * x) - 4 * Real.tan (4 * x) + 8 = 0)
  ↔ (∃ k : ℤ, x = (nat.pi : ℝ) / 32 * (4 * k + 3)) :=
begin
  sorry
end

end solution_exist_l684_684697


namespace jet_return_trip_time_l684_684928

def jet_tailwind_distance := 2000 -- miles
def tailwind_travel_time := 4 -- hours
def wind_speed := 50 -- mph

def jet_speed_in_still_air := (jet_tailwind_distance / tailwind_travel_time) - wind_speed
def jet_distance_against_wind := 2000 -- miles

theorem jet_return_trip_time :
  (jet_speed_in_still_air - wind_speed) * 5 = jet_distance_against_wind :=
by
  simp [jet_speed_in_still_air, jet_distance_against_wind, wind_speed, jet_tailwind_distance, tailwind_travel_time]
  sorry

end jet_return_trip_time_l684_684928


namespace simplify_and_evaluate_l684_684057

theorem simplify_and_evaluate (x : ℝ) (h : x^2 - 3*x - 2 = 0) :
  (x + 1) * (x - 1) - (x + 3)^2 + 2 * x^2 = -6 := 
by {
  sorry
}

end simplify_and_evaluate_l684_684057


namespace age_of_25th_student_l684_684900

theorem age_of_25th_student 
(A : ℤ) (B : ℤ) (C : ℤ) (D : ℤ)
(total_students : ℤ)
(total_age : ℤ)
(age_all_students : ℤ)
(avg_age_all_students : ℤ)
(avg_age_7_students : ℤ)
(avg_age_12_students : ℤ)
(avg_age_5_students : ℤ)
:
total_students = 25 →
avg_age_all_students = 18 →
avg_age_7_students = 20 →
avg_age_12_students = 16 →
avg_age_5_students = 19 →
total_age = total_students * avg_age_all_students →
age_all_students = total_age - (7 * avg_age_7_students + 12 * avg_age_12_students + 5 * avg_age_5_students) →
A = 7 * avg_age_7_students →
B = 12 * avg_age_12_students →
C = 5 * avg_age_5_students →
D = total_age - (A + B + C) →
D = 23 :=
by {
  sorry
}

end age_of_25th_student_l684_684900


namespace find_x_l684_684965

-- Define the lengths of the edges of the rectangular solid
def a : ℝ := 3
def b : ℝ := 2
def S : ℝ := 18 * Real.pi

-- Define the radius from the sphere's surface area
def radius : ℝ := Real.sqrt (S / (4 * Real.pi))

-- Hypothesis for the radius formula
def hyp_radius_eq : Prop := radius = (3 * Real.sqrt 2) / 2

-- Hypothesis for the spherical condition
def hyp_spherical_condition (x : ℝ) : Prop := 2 * radius = Real.sqrt (a^2 + b^2 + x^2)

-- Target to prove
theorem find_x : ∃ x : ℝ, hyp_spherical_condition x ∧ hyp_radius_eq ∧ x = Real.sqrt 5 :=
by
  use Real.sqrt 5
  split
  . sorry
  . split
    . sorry
    . rfl

end find_x_l684_684965


namespace tangent_line_to_circle_l684_684888

theorem tangent_line_to_circle (c : ℝ) (h : 0 < c) : 
  (∃ (x y : ℝ), x^2 + y^2 = 8 ∧ x + y = c) ↔ c = 4 :=
by sorry

end tangent_line_to_circle_l684_684888


namespace find_k_hyperbola_l684_684384

-- Define the given conditions
variables (k : ℝ)
def condition1 : Prop := k < 0
def condition2 : Prop := 2 * k^2 + k - 2 = -1

-- State the proof goal
theorem find_k_hyperbola (h1 : condition1 k) (h2 : condition2 k) : k = -1 :=
by
  sorry

end find_k_hyperbola_l684_684384


namespace move_all_objects_to_one_box_l684_684435

-- Define the type for objects in the boxes
inductive Object
| rock 
| paper
| scissors

-- Define the relationship for which object beats which
def beats : Object → Object → Prop
| Object.rock, Object.scissors => True
| Object.scissors, Object.paper => True
| Object.paper, Object.rock => True
| _, _ => False

-- Define the main theorem
theorem move_all_objects_to_one_box (n : ℕ) (hn : n ≥ 3) 
    (boxes : List Object) (hlen : boxes.length = n)
    (adj_distinct : ∀ i, boxes[i % n] ≠ boxes[(i + 1) % n])
    (contains_all : ∃ r, boxes.contains Object.rock ∧ 
                     boxes.contains Object.paper ∧ 
                     boxes.contains Object.scissors) :
  ∃ box_id, ∀ i : ℕ, (i ∈ boxes) → i = box_id :=
begin
  sorry -- Proof goes here
end

end move_all_objects_to_one_box_l684_684435


namespace probability_A_l684_684571

open ProbabilityTheory MeasureTheory

noncomputable def fair_coin : MeasureTheory.Measure Ω := 
  MeasureTheory.Measure.dirac (λ ω : Ω, (ω = 1/2))

def flip_n_times (n : Nat) : PMF (Fin n → Bool) :=
  PMF.finRange n >>= λ i, PMF.coin (1/2)

def event_A (n m : Nat) : Event (flip_n_times n × flip_n_times m) :=
  {ω | (ω.2.toFinset.filter id).card > (ω.1.toFinset.filter id).card}

open ProbabilityTheory.ProbabilityMeasure Event

theorem probability_A (n : Nat) (m : Nat) (hnm : n = 10) (hmm : m = 11):
  (probability (flip_n_times n) (event_A n m)) = 1 / 2 :=
by
  sorry

end probability_A_l684_684571


namespace collinear_vectors_value_l684_684856

theorem collinear_vectors_value (e1 e2 : ℝ^3) (k : ℝ) (h1 : e1 ≠ 0) (h2 : e2 ≠ 0) (h3 : ¬ collinear e1 e2) :
  collinear (k • e1 + e2) (e1 + k • e2) → k = 1 ∨ k = -1 :=
sorry

end collinear_vectors_value_l684_684856


namespace fraction_meaningful_condition_l684_684501

theorem fraction_meaningful_condition (x : ℝ) : (4 / (x + 2) ≠ 0) ↔ (x ≠ -2) := 
by 
  sorry

end fraction_meaningful_condition_l684_684501


namespace common_intersection_implies_cd_l684_684510

theorem common_intersection_implies_cd (a b c d : ℝ) (h : a ≠ b) (x y : ℝ) 
  (H1 : y = a * x + a) (H2 : y = b * x + b) (H3 : y = c * x + d) : c = d := by
  sorry

end common_intersection_implies_cd_l684_684510


namespace range_of_m_l684_684323

theorem range_of_m (x m : ℝ) : (|x - 3| ≤ 2) → ((x - m + 1) * (x - m - 1) ≤ 0) → 
  (¬(|x - 3| ≤ 2) → ¬((x - m + 1) * (x - m - 1) ≤ 0)) → 2 ≤ m ∧ m ≤ 4 :=
by
  intro h1 h2 h3
  sorry

end range_of_m_l684_684323


namespace oranges_per_store_visit_l684_684850

theorem oranges_per_store_visit (total_oranges : ℕ) (store_visits : ℕ) (h1 : total_oranges = 16) (h2 : store_visits = 8) :
  (total_oranges / store_visits) = 2 :=
by
  rw [h1, h2],
  exact Nat.div_eq_of_eq_mul_right (by norm_num) (by norm_num : 16 = 8 * 2)

end oranges_per_store_visit_l684_684850


namespace boys_play_theater_with_Ocho_l684_684468

variables (Ocho_friends : ℕ) (half_girls : Ocho_friends / 2 = 4)

theorem boys_play_theater_with_Ocho : (Ocho_friends / 2) = 4 := by
  -- Ocho_friends is the total number of Ocho's friends
  -- half_girls is given as a condition that half of Ocho's friends are girls
  -- thus, we directly use this to conclude that the number of boys is 4
  sorry

end boys_play_theater_with_Ocho_l684_684468


namespace units_digit_of_50_factorial_is_0_l684_684171

theorem units_digit_of_50_factorial_is_0 : 
  (∃ n : ℕ, 50! ≡ n [MOD 10]) ∧ (n = 0) := sorry

end units_digit_of_50_factorial_is_0_l684_684171


namespace max_value_of_g_l684_684309

noncomputable def g (x : ℝ) : ℝ := min (min (3 * x + 3) ((1 / 3) * x + 1)) (-2 / 3 * x + 8)

theorem max_value_of_g : ∃ x : ℝ, g x = 10 / 3 :=
by
  sorry

end max_value_of_g_l684_684309


namespace pencils_loss_equates_20_l684_684041

/--
Patrick purchased 70 pencils and sold them at a loss equal to the selling price of some pencils. The cost of 70 pencils is 1.2857142857142856 times the selling price of 70 pencils. Prove that the loss equates to the selling price of 20 pencils.
-/
theorem pencils_loss_equates_20 
  (C S : ℝ) 
  (h1 : C = 1.2857142857142856 * S) :
  (70 * C - 70 * S) = 20 * S :=
by
  sorry

end pencils_loss_equates_20_l684_684041


namespace log_inequality_l684_684251

theorem log_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) : log (1 + x + y) < x + y :=
sorry

end log_inequality_l684_684251


namespace integral_arctg_sqrt_l684_684256

theorem integral_arctg_sqrt (C : ℝ) :
  ∫ (x : ℝ), arctan (sqrt (3 * x - 1)) dx = (λ x, x * arctan (sqrt (3 * x - 1)) - (sqrt (3 * x - 1)) / 3 + C) :=
sorry

end integral_arctg_sqrt_l684_684256


namespace two_exponent_sum_square_l684_684193

theorem two_exponent_sum_square {n : ℕ} :
  let N := 2^10 + 2^13 + 2^14 + 3 * 2^n in
  (∃ k : ℕ, N = k^2) → (n = 13 ∨ n = 15) :=
by
  intros N hN
  sorry

end two_exponent_sum_square_l684_684193


namespace apple_harvest_l684_684882

theorem apple_harvest (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 →
  num_sections = 8 →
  total_sacks = sacks_per_section * num_sections →
  total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end apple_harvest_l684_684882


namespace least_number_to_add_l684_684938

theorem least_number_to_add (n : ℕ) (divisor : ℕ) (modulus : ℕ) (h1 : n = 1076) (h2 : divisor = 23) (h3 : n % divisor = 18) :
  modulus = divisor - (n % divisor) ∧ modulus = 5 := 
sorry

end least_number_to_add_l684_684938


namespace simplify_expression_l684_684556

theorem simplify_expression (x : ℝ) : (3 * x + 2) - 2 * (2 * x - 1) = 3 * x + 2 - 4 * x + 2 := 
by sorry

end simplify_expression_l684_684556


namespace inequality_correct_statement_l684_684980

theorem inequality_correct_statement
  (a b : ℝ)
  (hA : a > b → (1/a) < (1/b))
  (hB : a > b → a^2 > b^2)
  (hC : 0 > a > b → (1/a) < (1/b))
  (hD : 0 > a > b → a^2 > b^2) : 
  (∃ (h : 0 > a > b), (1/a) < (1/b)) := by
  sorry

end inequality_correct_statement_l684_684980


namespace minimum_value_expression_l684_684552

theorem minimum_value_expression (x y : ℝ) : ∃ (x y : ℝ), x = 4 ∧ y = -3 ∧ (x^2 + y^2 - 8 * x + 6 * y + 25) = 0 :=
by
  use 4, -3
  split
  · rfl
  split
  · rfl
  calc
    4^2 + (-3)^2 - 8 * 4 + 6 * (-3) + 25
      = 16 + 9 - 32 - 18 + 25 : by norm_num
  ... = 0 : by norm_num
  done

end minimum_value_expression_l684_684552


namespace surface_area_of_cube_l684_684233

-- Define the dimensions of the rectangular prism
def length : ℝ := 8
def width : ℝ := 2
def height : ℝ := 32

-- Define the volume of the rectangular prism
def prismVolume : ℝ := length * width * height

-- Define the edge length of the cube with the same volume as the rectangular prism
def cubeEdgeLength : ℝ := (prismVolume)^(1/3)

-- Define the surface area of the cube
def cubeSurfaceArea : ℝ := 6 * (cubeEdgeLength ^ 2)

theorem surface_area_of_cube : cubeSurfaceArea = 384 := 
by
  -- We can write out the proof steps or just skip with sorry
  sorry

end surface_area_of_cube_l684_684233


namespace equal_distances_to_circumcenter_l684_684010

open EuclideanGeometry

noncomputable def midpoint (a b : Point) : Point :=
  (a + b) / 2

theorem equal_distances_to_circumcenter
  (A B C M B1 C1 O : Point)
  (hM : M = midpoint B C)
  (hcircle_ABM : ∃ K : Circle, ∀ P, P ∈ K.to_set ↔ P = A ∨ P = B ∨ P = M)
  (hB1 : B1 ∈ (circumcircle_of_triangle A B M))
  (hcircle_ACM : ∃ K : Circle, ∀ P, P ∈ K.to_set ↔ P = A ∨ P = C ∨ P = M)
  (hC1 : C1 ∈ (circumcircle_of_triangle A C M))
  (hO : is_circumcenter O (triangle A B1 C1)) :
  dist O B = dist O C := by
  sorry

end equal_distances_to_circumcenter_l684_684010


namespace solve_equation_l684_684480

noncomputable def equation (x : ℝ) : Prop := x * (x - 2) + x - 2 = 0

theorem solve_equation : ∀ x, equation x ↔ (x = 2 ∨ x = -1) :=
by sorry

end solve_equation_l684_684480


namespace daniel_initial_noodles_l684_684268

theorem daniel_initial_noodles (noodles_given noodles_left : ℕ) 
  (h_given : noodles_given = 12)
  (h_left : noodles_left = 54) :
  ∃ x, x - noodles_given = noodles_left ∧ x = 66 := 
by 
sory

end daniel_initial_noodles_l684_684268


namespace find_particular_number_l684_684725

theorem find_particular_number (x : ℤ) (h : x - 29 + 64 = 76) : x = 41 :=
by
  sorry

end find_particular_number_l684_684725


namespace shortest_side_second_triangle_l684_684601

noncomputable def length_of_other_leg (a b : ℕ) (h : a ^ 2 + b ^ 2 = c ^ 2) : ℕ :=
  sqrt ((c ^ 2) - (a ^ 2))

-- Define the conditions
def right_triangle_first (side : ℕ) (hypotenuse : ℕ) (side = 15) (hypotenuse = 17) : Prop :=
  side ^ 2 + length_of_other_leg side hypotenuse = hypotenuse ^ 2

def similar_triangle_first_second (hypotenuse1 hypotenuse2 : ℕ) (hypotenuse1 = 17) (hypotenuse2 = 102): Prop :=
  hypotenuse2 = hypotenuse1 * 6

-- Prove the shortest side length of the second triangle is 48 cm
theorem shortest_side_second_triangle (side1 hypotenuse1 hypotenuse2 : ℕ)
  (h1 : right_triangle_first 15 17) (h2 : similar_triangle_first_second 17 102) : 
  side2 = 48 :=
  sorry

end shortest_side_second_triangle_l684_684601


namespace integral_sqrt_1_minus_x_sq_l684_684255

noncomputable def unitCircleArea : ℝ := π / 4

theorem integral_sqrt_1_minus_x_sq :
  ∫ x in 0..1, Real.sqrt (1 - x^2) = unitCircleArea :=
by
  simp [unitCircleArea]
  sorry

end integral_sqrt_1_minus_x_sq_l684_684255


namespace min_students_scoring_90_l684_684523

theorem min_students_scoring_90 {total_students : ℕ} {min_score max_score n_90 : ℕ}
  (h1 : total_students = 120)
  (h2 : min_score = 60)
  (h3 : max_score = 98)
  (h4 : min_score ≤ 90 ∧ 90 ≤ max_score)
  (h5 : ∀ s, s ≠ 90 → min_score ≤ s ∧ s ≤ max_score → s ≠ 90 ∧ n_90 > students_scoring s) :
  n_90 ≥ 5 :=
sorry

end min_students_scoring_90_l684_684523


namespace sqrt_15_minus_1_range_l684_684280

theorem sqrt_15_minus_1_range : (9 : ℝ) < 15 ∧ 15 < 16 → 2 < real.sqrt 15 - 1 ∧ real.sqrt 15 - 1 < 3 := 
by
  sorry

end sqrt_15_minus_1_range_l684_684280


namespace sheets_given_to_Charles_l684_684313

theorem sheets_given_to_Charles (initial_sheets : ℕ) (received_sheets : ℕ) (sheets_left : ℕ) (total_sheets : ℕ) (sheets_given : ℕ) :
  initial_sheets = 212 →
  received_sheets = 307 →
  sheets_left = 363 →
  (initial_sheets + received_sheets = total_sheets) →
  (total_sheets - sheets_left = sheets_given) →
  sheets_given = 156 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4 h5
  sorry

end sheets_given_to_Charles_l684_684313


namespace percentage_of_ll_watchers_l684_684038

theorem percentage_of_ll_watchers 
  (T : ℕ) 
  (IS : ℕ) 
  (ME : ℕ) 
  (E2 : ℕ) 
  (A3 : ℕ) 
  (total_residents : T = 600)
  (is_watchers : IS = 210)
  (me_watchers : ME = 300)
  (e2_watchers : E2 = 108)
  (a3_watchers : A3 = 21)
  (at_least_one_show : IS + (by sorry) + ME - E2 + A3 = T) :
  ∃ x : ℕ, (x * 100 / T) = 115 :=
by sorry

end percentage_of_ll_watchers_l684_684038


namespace find_b_l684_684568

theorem find_b (n b : ℝ) (h1 : n = 2 ^ 0.15) (h2 : n ^ b = 8) : b = 20 :=
by
  sorry

end find_b_l684_684568


namespace minimum_value_sin4_cos4_l684_684300

theorem minimum_value_sin4_cos4 (x : ℝ) : 
    ∃ y : ℝ, ∀ x : ℝ, (sin x)^4 + 2 * (cos x)^4 ≥ (sin y)^4 + 2 * (cos y)^4 ∧ (sin y)^4 + 2 * (cos y)^4 = 2 / 3 :=
by {
  -- Lean statement only, no proof will be provided
  sorry
}

end minimum_value_sin4_cos4_l684_684300


namespace sufficient_but_not_necessary_condition_l684_684195

theorem sufficient_but_not_necessary_condition (x : ℝ) : 
  (x > 2 → (x-1)^2 > 1) ∧ (∃ (y : ℝ), y ≤ 2 ∧ (y-1)^2 > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l684_684195


namespace gg_of_3_is_107_l684_684730

-- Define the function g
def g (x : ℕ) : ℕ := 3 * x + 2

-- State that g(g(g(3))) equals 107
theorem gg_of_3_is_107 : g (g (g 3)) = 107 := by
  sorry

end gg_of_3_is_107_l684_684730


namespace triple_composition_g_eq_107_l684_684732

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_g_eq_107 : g(g(g(3))) = 107 := by
  sorry

end triple_composition_g_eq_107_l684_684732


namespace limit_of_sequence_l684_684024

theorem limit_of_sequence 
  (k1 k2 : ℕ) (h : k1 ≠ k2)
  (x : ℕ → ℕ)
  (H : ∀ m n : ℕ, x n * x m + k1 * k2 ≤ k1 * x n + k2 * x m) :
  tendsto (λ n, ((n.factorial : ℝ) * (-1)^(1 + n) * (x n)^2) / (n^n : ℝ)) at_top (𝓝 0) :=
sorry

end limit_of_sequence_l684_684024


namespace problem_part_1_problem_part_2_l684_684676

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f (x + 1) a + g x

-- Problem Part (1)
theorem problem_part_1 (a : ℝ) (h_pos : 0 < a) :
  (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 :=
sorry

-- Problem Part (2)
theorem problem_part_2 (a : ℝ) (h_cond : ∀ x, 0 ≤ x → h x a ≥ 1) :
  a ≤ 2 :=
sorry

end problem_part_1_problem_part_2_l684_684676


namespace distance_to_circle_center_l684_684995

-- We define the circle using the given equation and the point
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8 * x - 2 * y + 16
def point : ℝ × ℝ := (-3, 4)

-- We state the theorem for the distance calculation
theorem distance_to_circle_center : 
  let center : ℝ × ℝ := (4, -1) in
  dist center point = Real.sqrt 74 :=
by
  sorry

end distance_to_circle_center_l684_684995


namespace triangle_LMN_length_l684_684486

theorem triangle_LMN_length (LMN : Type) [right_triangle_with_angle (LMN) N] 
  (h1 : sin N = 3 / 5) (h2 : length LM = 12) : 
  length LN = 20 :=
sorry

end triangle_LMN_length_l684_684486


namespace integral_of_reciprocal_l684_684286

theorem integral_of_reciprocal (a b : ℝ) (h_eq : a = 1) (h_eb : b = Real.exp 1) : ∫ x in a..b, 1/x = 1 :=
by 
  rw [h_eq, h_eb]
  sorry

end integral_of_reciprocal_l684_684286


namespace selected_students_correct_l684_684910

def student_selection (table : List (List ℕ)) (start_row start_col : ℕ) (n_rows n_cols : ℕ) (valid_range : Finset ℕ) : List ℕ :=
  let lt_row_limit r := r < n_rows
  let lt_col_limit c := c < n_cols
  let in_valid_range num := num ∈ valid_range
  let start_num := table[start_row][start_col]
  let rec read_from_table row col acc :=
    if acc.length = 5 then acc
    else
      let num := table[row][col]
      let new_acc := if in_valid_range num then acc.concat num else acc
      let new_col := col + 1
      let new_row := if new_col = n_cols then row + 1 else row
      let new_col := new_col % n_cols
      read_from_table new_row new_col new_acc
  read_from_table start_row start_col []

noncomputable def selected_students : List ℕ :=
  student_selection [[0, 1, 5, 4], [3, 2, 8, 7], [6, 5, 9, 5], [4, 2, 8, 7], [5, 3, 4, 6], [7, 9, 5, 3], [2, 5, 8, 6], [5, 7, 4, 1], [3, 3, 6, 9], [8, 3, 2, 4], 
                     [4, 5, 9, 7], [7, 3, 8, 6], [5, 2, 4, 4], [3, 5, 7, 8], [6, 2, 4, 1]] 0 2 15 4 (Finset.range 53)

theorem selected_students_correct :
  selected_students = [32, 42, 53, 46, 25] :=
by sorry

end selected_students_correct_l684_684910


namespace dorothy_score_l684_684490

theorem dorothy_score (T I D : ℝ) 
  (hT : T = 2 * I)
  (hI : I = (3 / 5) * D)
  (hSum : T + I + D = 252) : 
  D = 90 := 
by {
  sorry
}

end dorothy_score_l684_684490


namespace pirate_schooner_problem_l684_684222

theorem pirate_schooner_problem (p : ℕ) (h1 : 10 < p) 
  (h2 : 0.54 * (p - 10) = (54 : ℝ) / 100 * (p - 10)) 
  (h3 : 0.34 * (p - 10) = (34 : ℝ) / 100 * (p - 10)) 
  (h4 : 2 / 3 * p = (2 : ℝ) / 3 * p) : 
  p = 60 := 
sorry

end pirate_schooner_problem_l684_684222


namespace sin_alpha_plus_beta_l684_684685

theorem sin_alpha_plus_beta (α β : ℝ) 
    (h1 : π / 4 < α) (h2 : α < 3 * π / 4) 
    (h3 : 0 < β) (h4 : β < π / 4) 
    (h5 : cos (π / 4 + α) = -3 / 5) 
    (h6 : sin (3 * π / 4 + β) = 5 / 13) : 
      sin (α + β) = 63 / 65 := 
by 
  sorry

end sin_alpha_plus_beta_l684_684685


namespace financial_outcome_l684_684817

theorem financial_outcome :
  let initial_value : ℝ := 12000
  let selling_price : ℝ := initial_value * 1.20
  let buying_price : ℝ := selling_price * 0.85
  let financial_outcome : ℝ := buying_price - initial_value
  financial_outcome = 240 :=
by
  sorry

end financial_outcome_l684_684817


namespace min_value_of_expression_l684_684546

open Real

noncomputable def min_expression_value : ℝ :=
  let expr := λ (x y : ℝ), x^2 + y^2 - 8 * x + 6 * y + 25
  0

theorem min_value_of_expression : ∃ x y : ℝ, x = 4 ∧ y = -3 ∧ (x^2 + y^2 - 8 * x + 6 * y + 25) = min_expression_value :=
by {
  use [4, -3],
  split,
  { refl },
  split,
  { refl },
  sorry
}

end min_value_of_expression_l684_684546


namespace average_runs_l684_684391

theorem average_runs (games : ℕ) (runs1 matches1 runs2 matches2 runs3 matches3 : ℕ)
  (h1 : runs1 = 1) 
  (h2 : matches1 = 1) 
  (h3 : runs2 = 4) 
  (h4 : matches2 = 2)
  (h5 : runs3 = 5) 
  (h6 : matches3 = 3) 
  (h_games : games = matches1 + matches2 + matches3) :
  (runs1 * matches1 + runs2 * matches2 + runs3 * matches3) / games = 4 :=
by
  sorry

end average_runs_l684_684391


namespace large_planter_holds_seeds_l684_684040

theorem large_planter_holds_seeds (total_seeds : ℕ) (small_planter_capacity : ℕ) (num_small_planters : ℕ) (num_large_planters : ℕ) 
  (h1 : total_seeds = 200)
  (h2 : small_planter_capacity = 4)
  (h3 : num_small_planters = 30)
  (h4 : num_large_planters = 4) : 
  (total_seeds - num_small_planters * small_planter_capacity) / num_large_planters = 20 := by
  sorry

end large_planter_holds_seeds_l684_684040


namespace convex_hexagon_diagonal_l684_684473

theorem convex_hexagon_diagonal (S : ℝ) (hex : set (set ℝ)) (h_convex : convex ℝ hex) (h_area : measure_space.volume hex = S) :
  ∃ diagonal, ∃ triangle, triangle ⊆ hex ∧ convex ℝ triangle ∧ measure_space.volume triangle ≤ S / 6 :=
by sorry

end convex_hexagon_diagonal_l684_684473


namespace overlapping_area_l684_684956

theorem overlapping_area (square : set (ℝ×ℝ))
  (triangle : set (ℝ×ℝ)) :
  (∀ p : ℝ×ℝ, p ∈ square ↔ 
    (p = (0,0) ∨ p = (2,0) ∨ p = (2,2) ∨ p = (0,2))) → 
  (∀ p : ℝ×ℝ, p ∈ triangle ↔ 
    (p = (3,0) ∨ p = (1,2) ∨ p = (2,1))) →
  (∃ overlap : set (ℝ×ℝ), area overlap = 0.15) :=
by
  -- Defining the vertices of the square and the triangle.
  let vertices_square := {(0,0), (2,0), (2,2), (0,2)}
  let vertices_triangle := {(3,0), ((1,2): ℝ×ℝ), (2,1)}
  -- Ensuring conditions hold.
  have hsquare : ∀ p : ℝ×ℝ, p ∈ square ↔ p ∈ vertices_square := sorry
  have htriangle : ∀ p : ℝ×ℝ, p ∈ triangle ↔ p ∈ vertices_triangle := sorry
  -- Defining the overlapping region.
  let overlap := square ∩ triangle
  -- Calculating the area of the overlapping region.
  have area_overlap : area overlap = 0.15 := sorry
  -- Proving the theorem.
  exact ⟨overlap, area_overlap⟩

end overlapping_area_l684_684956


namespace total_cat_count_l684_684991

noncomputable def totalCats : ℕ :=
  let j := 60
  let s := 40
  let p := 35
  let f := 45
  let js := 20
  let sp := 15
  let pf := 10
  let fj := 18
  let jsp := 5
  let spf := 3
  let pfj := 7
  let jfs := 10
  let all_four := 2
  let none := 12 in
  let exclusiveJ := j - (js + jsp + fj - 2*all_four)
  let exclusiveS := s - (js + sp + jfs - 2*all_four)
  let exclusiveP := p - (sp + pf + jsp - 2*all_four)
  let exclusiveF := f - (fj + pf + jfs - 2*all_four)
  let two_way_js := js - (jsp + all_four)
  let two_way_sp := sp - (spf + all_four)
  let two_way_pf := pf - (pfj + all_four)
  let two_way_fj := fj - (jfs + all_four)
  let three_way_jsp := jsp - all_four
  let three_way_spf := spf - all_four
  let three_way_pfj := pfj - all_four
  let three_way_jfs := jfs - all_four in
  none + exclusiveJ + exclusiveS + exclusiveP + exclusiveF +
    two_way_js + two_way_sp + two_way_pf + two_way_fj +
    three_way_jsp + three_way_spf + three_way_pfj + three_way_jfs + all_four 

theorem total_cat_count : totalCats = 143 :=
by sorry

end total_cat_count_l684_684991


namespace range_of_quadratic_function_is_geq_11_over_4_l684_684099

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - x + 3

-- Define the range of the quadratic function
def range_of_quadratic_function := {y : ℝ | ∃ x : ℝ, quadratic_function x = y}

-- Prove the statement
theorem range_of_quadratic_function_is_geq_11_over_4 : range_of_quadratic_function = {y : ℝ | y ≥ 11 / 4} :=
by
  sorry

end range_of_quadratic_function_is_geq_11_over_4_l684_684099


namespace hyperbola_eccentricity_l684_684691

noncomputable def eccentricity_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_point : (4 / (a^2) - 1 / (b^2)) = 1) : ℝ :=
    let e := sqrt (1 + (1 + b^2) / 4) in
    e

theorem hyperbola_eccentricity (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_point : (4 / (a^2) - 1 / (b^2)) = 1) :
    eccentricity_range a b h_a_pos h_b_pos h_point > sqrt(5) / 2 :=
sorry

end hyperbola_eccentricity_l684_684691


namespace units_digit_factorial_50_is_0_l684_684150

theorem units_digit_factorial_50_is_0 : (nat.factorial 50) % 10 = 0 := by
  sorry

end units_digit_factorial_50_is_0_l684_684150


namespace min_value_sin_cos_l684_684297

theorem min_value_sin_cos (x : ℝ) : sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 := 
by 
  sorry

end min_value_sin_cos_l684_684297


namespace son_current_age_l684_684217

theorem son_current_age (S : ℕ) (hf_now : 38) (hf_past : 28) (cond : 28 = 7 * (S - 10)) : S = 14 := by
  sorry

end son_current_age_l684_684217


namespace closest_point_to_line_l684_684301

noncomputable def closest_point (line_pt dir_vec target : ℝ × ℝ × ℝ) (t : ℝ) :=
  let lx := line_pt.1 + t * dir_vec.1 in
  let ly := line_pt.2 + t * dir_vec.2 in
  let lz := line_pt.3 + t * dir_vec.3 in
  (lx, ly, lz)

theorem closest_point_to_line : 
  ∀ (line_pt : ℝ × ℝ × ℝ) (dir_vec : ℝ × ℝ × ℝ) (target : ℝ × ℝ × ℝ),
  line_pt = (5, 1, -2) →
  dir_vec = (-3, 7, -4) →
  target = (1, 2, -1) →
  closest_point line_pt dir_vec target (23 / 74) = (287 / 74, 215 / 74, -314 / 74) :=
  by 
  intros line_pt dir_vec target h_line_pt h_dir_vec h_target
  rw [h_line_pt, h_dir_vec, h_target]
  sorry

end closest_point_to_line_l684_684301


namespace sachin_rahul_age_ratio_l684_684050

theorem sachin_rahul_age_ratio 
(S_age : ℕ) 
(R_age : ℕ) 
(h1 : R_age = S_age + 4) 
(h2 : S_age = 14) : 
S_age / Int.gcd S_age R_age = 7 ∧ R_age / Int.gcd S_age R_age = 9 := 
by 
sorry

end sachin_rahul_age_ratio_l684_684050


namespace square_perimeter_ratio_l684_684494

theorem square_perimeter_ratio (a₁ a₂ s₁ s₂ : ℝ) 
  (h₁ : a₁ / a₂ = 16 / 25)
  (h₂ : a₁ = s₁^2)
  (h₃ : a₂ = s₂^2) :
  (4 : ℝ) / 5 = s₁ / s₂ :=
by sorry

end square_perimeter_ratio_l684_684494


namespace difference_of_squares_l684_684979

theorem difference_of_squares (a b : ℝ) : -4 * a^2 + b^2 = (b + 2 * a) * (b - 2 * a) :=
by
  sorry

end difference_of_squares_l684_684979


namespace count_solutions_l684_684689

theorem count_solutions : 
  (∃ (n : ℕ), ∀ (x : ℕ), (x + 17) % 43 = 71 % 43 ∧ x < 150 → n = 4) := 
sorry

end count_solutions_l684_684689


namespace determine_values_l684_684579

-- Assuming the given conditions:
def digits_satisfy (x y z b : ℕ) : Prop :=
  x * b^2 + y * b + z = 1987 ∧ x + y + z = 25

-- Full possible values
theorem determine_values (x y z b : ℕ) (hx : x = 5) (hy : y = 9) (hz : z = 11) (hb : b = 19) :
  digits_satisfy x y z b :=
by {
  rw [hx, hy, hz, hb],
  split,
  { linarith },
  { linarith },
}

end determine_values_l684_684579


namespace shuttle_speed_conversion_l684_684604

-- Define the speed of the space shuttle in kilometers per second
def shuttle_speed_km_per_sec : ℕ := 6

-- Define the number of seconds in an hour
def seconds_per_hour : ℕ := 3600

-- Define the expected speed in kilometers per hour
def expected_speed_km_per_hour : ℕ := 21600

-- Prove that the speed converted to kilometers per hour is equal to the expected speed
theorem shuttle_speed_conversion : shuttle_speed_km_per_sec * seconds_per_hour = expected_speed_km_per_hour :=
by
    sorry

end shuttle_speed_conversion_l684_684604


namespace circle_equation_l684_684340

noncomputable def midpoint (A B : (ℝ × ℝ)) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem circle_equation (A B : ℝ × ℝ) (hA : A.2 = 0) (hB : B.1 = 0) (center : ℝ × ℝ) (hcenter : (center = (2, -3))) :
  midpoint A B = center → (A.1 - 2)^2 + (-3)^2 + (B.2 + 3)^2 = 13 := 
sorry

end circle_equation_l684_684340


namespace problem_l684_684060

def a := 1 / 4
def b := 1 / 2
def c := -3 / 4

def a_n (n : ℕ) : ℚ := 2 * n + 1
def S_n (n : ℕ) : ℚ := (n + 2) * n
def f (n : ℕ) : ℚ := 4 * a * n^2 + (4 * a + 2 * b) * n + (a + b + c)

theorem problem : ∀ n : ℕ, f n = S_n n := by
  sorry

end problem_l684_684060


namespace min_balls_to_ensure_10_of_one_color_l684_684393

theorem min_balls_to_ensure_10_of_one_color (r b y k : ℕ) (h1 : r = 20) (h2 : b = 20) (h3 : y = 20) (h4 : k = 10) :
  ∀ (total : ℕ), total = r + b + y + k + k → (∑ c : ℕ in {r, b, y, k}, c < ∑ c in {r, b, y, k}, c + 28) ∨ (∑ c : ℕ in {r, b, y, k}, c = ∑ c in {r, b, y, k}, c + 38) → 
  38 := 
sorry

end min_balls_to_ensure_10_of_one_color_l684_684393


namespace fixed_point_of_parabolas_l684_684450

theorem fixed_point_of_parabolas (x y t : ℝ) (h : ∀ t : ℝ, y = 5 * x^2 + 4 * t * x - 3 * t) :
  x = 3 / 4 ∧ y = 45 / 16 :=
by {
  assume t,
  sorry
}

end fixed_point_of_parabolas_l684_684450


namespace interval_representation_l684_684098

theorem interval_representation (x : ℝ) : 
  (∃ x, -1 < x ∧ x < 1) → (∀ x, -1 < x ∧ x < 1 ↔ x ∈ set.Ioo (-1 : ℝ) (1 : ℝ)) := by
    sorry

end interval_representation_l684_684098


namespace find_cos_alpha_minus_pi_over_3_l684_684350

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin(2 * x + π / 6)

-- Define the angle alpha and corresponding function value
constant α : ℝ
axiom α_acute : 0 < α ∧ α < π / 2
axiom f_alpha : f (α / 2 - π / 12) = 3 / 5

-- The main statement capturing the problem
theorem find_cos_alpha_minus_pi_over_3 : 
  (ω = 2) → (ϕ = π / 6) → 
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ f (α / 2 - π / 12) = 3 / 5 ∧
  Real.cos (α - π / 3) = (4 + 3 * Real.sqrt 3) / 10 :=
by 
  intros ω_eq ϕ_eq
  use α
  exact ⟨α_acute.1, α_acute.2, f_alpha, sorry⟩

end find_cos_alpha_minus_pi_over_3_l684_684350


namespace quadratic_poly_correct_l684_684303

noncomputable def quadratic_poly : polynomial ℝ :=
  3 * polynomial.C 1 * ((polynomial.X - polynomial.C (4 + 2 * complex.I)) * 
  (polynomial.X - polynomial.C (4 - 2 * complex.I)))

theorem quadratic_poly_correct : 
  quadratic_poly = 3 * polynomial.X ^ 2 - 24 * polynomial.X + 60 := by
  sorry

end quadratic_poly_correct_l684_684303


namespace hyperbola_eccentricity_l684_684382

theorem hyperbola_eccentricity (k : ℝ) (h_eq : ∀ x y : ℝ, x^2 + k * y^2 = 1) (h_eccentricity : ∀ e : ℝ, e = 2) :
    k = -1 / 3 := 
sorry

end hyperbola_eccentricity_l684_684382


namespace probability_twins_on_street_probability_twins_in_family_expected_twins_among_N_first_graders_l684_684578

-- Condition: Probability of twins being born in Schwambrania
variable (p : ℝ)
-- Given that triplets are not born in Schwambrania, we do not need to define it explicitly.

-- a) Prove: probability that a randomly encountered person on the street is one of a pair of twins
theorem probability_twins_on_street (h : 0 ≤ p ∧ p ≤ 1) : 
  (\frac{2p}{p+1} : ℝ) = (\frac{2p}{p+1}) :=
sorry

-- b: Prove: In a family with three children, the probability that there is a pair of twins among them
theorem probability_twins_in_family (h : 0 ≤ p ∧ p ≤ 1) : 
  (\frac{2p}{2p + (1-p)^3} : ℝ) = (\frac{2p}{2p + (1-p)^3}) :=
sorry
  
-- c: Prove: the expected number of pairs of twins among N first-graders
theorem expected_twins_among_N_first_graders (N : ℕ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (\frac{Np}{p+1} : ℝ) = (\frac{Np}{p+1}) :=
sorry

end probability_twins_on_street_probability_twins_in_family_expected_twins_among_N_first_graders_l684_684578


namespace pirate_schooner_problem_l684_684221

theorem pirate_schooner_problem (p : ℕ) (h1 : 10 < p) 
  (h2 : 0.54 * (p - 10) = (54 : ℝ) / 100 * (p - 10)) 
  (h3 : 0.34 * (p - 10) = (34 : ℝ) / 100 * (p - 10)) 
  (h4 : 2 / 3 * p = (2 : ℝ) / 3 * p) : 
  p = 60 := 
sorry

end pirate_schooner_problem_l684_684221


namespace tank_emptying_time_correct_l684_684977

noncomputable def tank_emptying_time : ℝ :=
  let initial_volume := 1 / 5
  let fill_rate := 1 / 15
  let empty_rate := 1 / 6
  let combined_rate := fill_rate - empty_rate
  initial_volume / combined_rate

theorem tank_emptying_time_correct :
  tank_emptying_time = 2 :=
by
  -- Proof will be provided here
  sorry

end tank_emptying_time_correct_l684_684977


namespace units_digit_50_factorial_l684_684158

theorem units_digit_50_factorial : (nat.factorial 50) % 10 = 0 :=
by
  sorry

end units_digit_50_factorial_l684_684158


namespace emails_after_deleting_old_l684_684000

-- Definitions for the conditions in the problem
def d1 := 50
def r1 := 15
def d2 := 20
def r2 := 5
def e_final := 30

-- Statement to prove the number of additional emails
theorem emails_after_deleting_old : 
  e_final - (r1 + r2) = 10 :=
by
  -- calculation steps
  rw [e_final, r1, r2]
  exact Nat.sub_eq_of_eq_add 10 (by norm_num)


end emails_after_deleting_old_l684_684000


namespace min_value_sin_cos_l684_684298

theorem min_value_sin_cos (x : ℝ) : sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 := 
by 
  sorry

end min_value_sin_cos_l684_684298


namespace largest_M_in_base_7_l684_684792

-- Define the base and the bounds for M^2
def base : ℕ := 7
def lower_bound : ℕ := base^3
def upper_bound : ℕ := base^4

-- Define M and its maximum value.
def M : ℕ := 48

-- Define a function to convert a number to its base 7 representation
def to_base_7 (n : ℕ) : List ℕ :=
  if n == 0 then [0] else
  let rec digits (n : ℕ) : List ℕ :=
    if n == 0 then [] else (n % 7) :: digits (n / 7)
  digits n |>.reverse

-- Define the base 7 representation of 48
def M_base_7 : List ℕ := to_base_7 M

-- The main statement asserting the conditions and the solution
theorem largest_M_in_base_7 :
  lower_bound ≤ M^2 ∧ M^2 < upper_bound ∧ M_base_7 = [6, 6] :=
by
  sorry

end largest_M_in_base_7_l684_684792


namespace lim_arithmetic_seq_l684_684329

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def condition1 (h₂ h₇ h₈ h₁₁: ℝ) : Prop := h₂ + h₇ + h₈ + h₁₁ = 48

def condition2 (h₃ h₁₁: ℝ) : Prop := h₃ / h₁₁ = 1 / 2

def Sn (n : ℕ) : ℝ := S n -- Sum of the first n terms definition

-- Given conditions
def c1 := ∃ (h₂ h₇ h₈ h₁₁ : ℝ), condition1 h₂ h₇ h₈ h₁₁
def c2 := ∃ (h₃ h₁₁ : ℝ), condition2 h₃ h₁₁

theorem lim_arithmetic_seq (c1 : c1) (c2 : c2) : 
    tendsto (λ n, (n : ℝ) * a n / S (2 * n)) at_top (𝓝 (1 / 2)) := by
  sorry

end lim_arithmetic_seq_l684_684329


namespace max_f_geq_fraction_3_sqrt3_over_2_l684_684311

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * x) + Real.sin (3 * x)

theorem max_f_geq_fraction_3_sqrt3_over_2 : ∃ x : ℝ, f x ≥ (3 + Real.sqrt 3) / 2 := 
sorry

end max_f_geq_fraction_3_sqrt3_over_2_l684_684311


namespace concurrent_lines_l684_684333

theorem concurrent_lines
  (c : circle)
  (A B O C E D H Z S I K L M P U R X T Q : point)
  (tangent_c_C : tangent C)
  (C_on_c : C ∈ c)
  (E_intersection : ∃ E, tangent_c_C ∩ line_through A B = {E})
  (D_dir_perp : is_perp D (line_through C D) (line_through A B))
  (CD_eq_CH_CZ : ∃ H Z, C D = C H ∧ C H = C Z ∧ H ∈ c ∧ Z ∈ c)
  (intersections_HZ : ∃ S I K, line_through H Z ∩ line_through C O = {S} ∧ line_through H Z ∩ line_through C D = {I} ∧ line_through H Z ∩ line_through A B = {K})
  (parallel_line_I_AB : ∃ L M, line_through I parallel line_through A B ∩ line_through C O = {L} ∧ line_through I parallel line_through A B ∩ line_through C K = {M})
  (circumcircle_LMD : circle_of_triangle L M D∩line_through A B = {P} ∧ circle_of_triangle L M D∩line_through C K = {U})
  (tangents_k : ∃ (e1 e2 e3 : line), tangent L e1 ∧ tangent M e2 ∧ tangent P e3)
  (RXT_intersections : ∃ R X T, intersection e1 e2 = {R} ∧ intersection e2 e3 = {X} ∧ intersection e1 e3 = {T})
  (Q_center : Q = center_of_circle (circle_of_triangle L M D))
  :
  concurrent (line_through R D) (line_through T U) (line_through X S) ∧ on_line (point_of_concurrency (line_through R D) (line_through T U) (line_through X S)) (line_through I Q) := by
  sorry

end concurrent_lines_l684_684333


namespace no_consecutive_squares_of_arithmetic_progression_l684_684870

theorem no_consecutive_squares_of_arithmetic_progression (d : ℕ):
  (d % 10000 = 2019) →
  (∀ a b c : ℕ, a < b ∧ b < c → b^2 - a^2 = d ∧ c^2 - b^2 = d →
  false) :=
sorry

end no_consecutive_squares_of_arithmetic_progression_l684_684870


namespace gain_percent_after_discount_l684_684188

variable (MP : ℝ)
def CP := 0.64 * MP
def SP := 0.88 * MP
def profit := SP - CP
def gain_percent := (profit / CP) * 100

theorem gain_percent_after_discount : gain_percent = 37.5 := by
  sorry

end gain_percent_after_discount_l684_684188


namespace prime_factor_of_sum_l684_684654

theorem prime_factor_of_sum (n : ℤ) : ∃ p : ℕ, Nat.Prime p ∧ p = 2 ∧ (2 * n + 1 + 2 * n + 3 + 2 * n + 5 + 2 * n + 7) % p = 0 :=
by
  sorry

end prime_factor_of_sum_l684_684654


namespace vision_approx_l684_684491

theorem vision_approx (L : ℝ) (V : ℝ) (h1 : L = 5 + Real.log10 V) (h2 : L = 4.9) : V ≈ 0.8 :=
by
  sorry

end vision_approx_l684_684491


namespace diagonals_in_octadecagon_l684_684109

def num_sides : ℕ := 18

def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_octadecagon : num_diagonals num_sides = 135 := by 
  sorry

end diagonals_in_octadecagon_l684_684109


namespace Mr_Brown_selling_price_l684_684464

theorem Mr_Brown_selling_price 
  (initial_price : ℕ := 100000)
  (profit_percentage : ℚ := 10)
  (loss_percentage : ℚ := 10) :
  let profit := initial_price * (profit_percentage / 100),
      price_to_Brown := initial_price + profit,
      loss := price_to_Brown * (loss_percentage / 100),
      selling_price := price_to_Brown - loss
  in selling_price = 99000 := by
    sorry

end Mr_Brown_selling_price_l684_684464


namespace A_inter_complement_B_l684_684714

variable (U : Set ℝ := {x | True}) -- Universal set U = ℝ
def A : Set ℝ := { x : ℝ | x^2 - 2 * x < 0 }
def B : Set ℝ := { x : ℝ | x - 1 ≥ 0 }

theorem A_inter_complement_B (x : ℝ) : x ∈ (A U) ∩ (U \ B) ↔ x > 0 ∧ x < 1 := by
  sorry

end A_inter_complement_B_l684_684714


namespace problem_1_problem_2_l684_684337

open Nat

theorem problem_1 (a : ℤ) (S : ℕ → ℤ) (a_n : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = 2^(n + 1) + a) 
  (ha_geom : ∀ n : ℕ, a_n (n + 1) = (if n = 0 then S 1 else S(n + 1) - S n))
  (hgeom : ∀ n : ℕ, S n = (a_n 1 + S 0) * n) :
  a = -1 ∧ (∀ n, a_n n = 2^(n - 1)) :=
sorry

theorem problem_2 (T_n b_n a_n : ℕ → ℤ)
  (hb_n : ∀ n, b_n n = (2 * n - 1) * a_n n)
  (ha_n : ∀ n, a_n n = 2^(n - 1)) :
  ∀ n, T_n n = (2 * n - 3) * (2^n) + 3 :=
sorry

end problem_1_problem_2_l684_684337


namespace units_digit_factorial_50_is_0_l684_684147

theorem units_digit_factorial_50_is_0 : (nat.factorial 50) % 10 = 0 := by
  sorry

end units_digit_factorial_50_is_0_l684_684147


namespace meal_combinations_total_l684_684252

theorem meal_combinations_total : 
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let pizza_drinks := 1
  let general_combinations := (main_courses - 1) * sides * drinks -- non-Pizza main courses
  let pizza_combinations := 1 * sides * pizza_drinks -- Pizza main course
  in general_combinations + pizza_combinations = 27 :=
by
  sorry

end meal_combinations_total_l684_684252


namespace solve_equation_l684_684898

theorem solve_equation 
  (x : ℝ) 
  (h : (2 * x - 1)^2 - (1 - 3 * x)^2 = 5 * (1 - x) * (x + 1)) : 
  x = 5 / 2 := 
sorry

end solve_equation_l684_684898


namespace problem_1_problem_2_l684_684703

-- Defining the functions f and g
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
def g (x : ℝ) : ℝ := (1 / 2) * x + 3

-- Problem (1)
theorem problem_1 (x : ℝ) : 0 < x ∧ x < 4 ↔ f x (-2) < g x :=
sorry

-- Problem (2)
theorem problem_2 (a : ℝ) (h : a > -1) :
  (∃ x : ℝ, -a ≤ x ∧ x ≤ 1 ∧ f x a ≤ g x) ↔ -1 < a ∧ a ≤ 5 / 2 :=
sorry

end problem_1_problem_2_l684_684703


namespace largest_b_value_l684_684903

/- Definitions -/
def conditions (a b c : ℕ) : Prop :=
  1 < c ∧ c < b ∧ b < a ∧ a * b * c = 360

noncomputable def largest_b : ℕ :=
  Nat.find_max' (λ b, ∃ a c, conditions a b c) sorry

/- Theorem -/
theorem largest_b_value : largest_b = 12 :=
sorry

end largest_b_value_l684_684903


namespace arithmetic_geometric_mean_l684_684798

variable {x : ℝ}

theorem arithmetic_geometric_mean (h : x > 0) : 
  let a := 2 * x,
      b := 2 * x^2,
      AM := (a + b) / 2,
      GM := Real.sqrt (a * b)
  in AM = x + x^2 ∧ GM = 2 * x ^ (3 / 2) :=
by 
  intro a b AM GM
  dsimp [AM, GM, a, b]
  split
  { rw [add_mul, add_comm], sorry }
  { rw [eq_comm, Real.sqrt_eq_iff_sq_eq], sorry }

end arithmetic_geometric_mean_l684_684798


namespace units_digit_of_50_factorial_l684_684137

theorem units_digit_of_50_factorial : (nat.factorial 50) % 10 = 0 := 
by 
  sorry

end units_digit_of_50_factorial_l684_684137


namespace students_between_min_and_hos_l684_684577

theorem students_between_min_and_hos
  (total_students : ℕ)
  (minyoung_left_position : ℕ)
  (hoseok_right_position : ℕ)
  (total_students_eq : total_students = 13)
  (minyoung_left_position_eq : minyoung_left_position = 8)
  (hoseok_right_position_eq : hoseok_right_position = 9) :
  (minyoung_left_position - (total_students - hoseok_right_position + 1) - 1) = 2 := 
by
  sorry

end students_between_min_and_hos_l684_684577


namespace sum_of_angles_is_210_l684_684408

namespace Geometry

-- Variables and given conditions
variables (A B C D P Q R : Type)
variables [rectangle : Rectangle A B C D] -- ABCD is a rectangle
variables [OnBC : PointOn P (line B C)]
variables [OnCD : PointOn Q (line C D)]
variables [Inside : PointIn R (Rect A B C D)]

-- Angles given in the problem
variables (w x y z : ℝ)
variables (angle_PRQ : angle P R Q = 30)
variables (angle_RQD : angle R Q D = w)
variables (angle_PQC : angle P Q C = x)
variables (angle_CPQ : angle C P Q = y)
variables (angle_BPR : angle B P R = z)

-- Prove that the sum of these angles is 210 degrees
theorem sum_of_angles_is_210 : w + x + y + z = 210 :=
by sorry

end Geometry

end sum_of_angles_is_210_l684_684408


namespace max_distance_cat_l684_684039

theorem max_distance_cat
  (post : ℝ × ℝ)
  (rope_length : ℝ)
  (origin : ℝ × ℝ := (0, 0))
  (h_post : post = (5, -2))
  (h_rope : rope_length = 12) :
  let distance_to_center := real.sqrt ((post.1 - origin.1)^2 + (post.2 - origin.2)^2) in
  distance_to_center = real.sqrt 29 →
  rope_length + distance_to_center = 12 + real.sqrt 29 :=
by
  intros distance_to_center h_dist
  rw [h_post, h_rope, h_dist]
  sorry

end max_distance_cat_l684_684039


namespace chessboard_signs_all_plus_l684_684186

theorem chessboard_signs_all_plus (board : Fin 8 → Fin 8 → Bool) :
  (∃ ops : List (Fin 8 → Fin 8 → Bool → Bool), 
     ∀ (row col : Fin 8), 
     (board row col = false → board_after_ops row col = true)) → False :=
sorry

end chessboard_signs_all_plus_l684_684186


namespace sum_of_mixed_numbers_is_between_18_and_19_l684_684520

theorem sum_of_mixed_numbers_is_between_18_and_19 :
  let a := 2 + 3 / 8;
  let b := 4 + 1 / 3;
  let c := 5 + 2 / 21;
  let d := 6 + 1 / 11;
  18 < a + b + c + d ∧ a + b + c + d < 19 :=
by
  sorry

end sum_of_mixed_numbers_is_between_18_and_19_l684_684520


namespace correct_quotient_l684_684757

theorem correct_quotient (Q : ℤ) (D : ℤ) (h1 : D = 21 * Q) (h2 : D = 12 * 35) : Q = 20 :=
by {
  sorry
}

end correct_quotient_l684_684757


namespace mod_arithmetic_problem_l684_684804

open Nat

theorem mod_arithmetic_problem :
  ∃ n : ℤ, 0 ≤ n ∧ n < 29 ∧ (5 * n ≡ 1 [ZMOD 29]) ∧ ((3 ^ n) ^ 2 - 3 ≡ 13 [ZMOD 29]) :=
by
  sorry

end mod_arithmetic_problem_l684_684804


namespace winning_majority_vote_l684_684765

def total_votes : ℕ := 600

def winning_percentage : ℝ := 0.70

def losing_percentage : ℝ := 0.30

theorem winning_majority_vote : (0.70 * (total_votes : ℝ) - 0.30 * (total_votes : ℝ)) = 240 := 
by
  sorry

end winning_majority_vote_l684_684765


namespace TA_eq_TM_l684_684009

-- Definitions based on conditions
variable (ABC : Type) [triangle ABC]
variable (B C : ABC) [midpoint M (B, C)]
variable (D : ABC) [foot D B]
variable (E : ABC) [foot E C]
variable (X : ABC) [midpoint X (E, M)]
variable (Y : ABC) [midpoint Y (D, M)]
variable (A : ABC) [parallel_line (BC, A)]
variable (T : ABC) [intersection T (line XY) (parallel_line BC A)]

-- The theorem stating the conclusion
theorem TA_eq_TM : dist T A = dist T M := 
sorry

end TA_eq_TM_l684_684009


namespace units_digit_50_factorial_l684_684162

theorem units_digit_50_factorial : (nat.factorial 50) % 10 = 0 :=
by
  sorry

end units_digit_50_factorial_l684_684162


namespace apple_harvest_l684_684883

theorem apple_harvest (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 →
  num_sections = 8 →
  total_sacks = sacks_per_section * num_sections →
  total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end apple_harvest_l684_684883


namespace candy_mixture_price_l684_684253

theorem candy_mixture_price
  (price_first_per_kg : ℝ) (price_second_per_kg : ℝ) (weight_ratio : ℝ) (weight_second : ℝ) 
  (h1 : price_first_per_kg = 10) 
  (h2 : price_second_per_kg = 15) 
  (h3 : weight_ratio = 3) 
  : (price_first_per_kg * weight_ratio * weight_second + price_second_per_kg * weight_second) / 
    (weight_ratio * weight_second + weight_second) = 11.25 :=
by
  sorry

end candy_mixture_price_l684_684253


namespace max_area_equilateral_in_rectangle_l684_684519

-- Define the dimensions of the rectangle
def length_efgh : ℕ := 15
def width_efgh : ℕ := 8

-- The maximum possible area of an equilateral triangle inscribed in the rectangle
theorem max_area_equilateral_in_rectangle : 
  ∃ (s : ℝ), 
  s = ((16 * Real.sqrt 3) / 3) ∧ 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ length_efgh → 
    (∃ (area : ℝ), area = (Real.sqrt 3 / 4 * s^2) ∧
      area = 64 * Real.sqrt 3)) :=
by sorry

end max_area_equilateral_in_rectangle_l684_684519


namespace john_thrice_tom_years_ago_l684_684004

-- Define the ages of Tom and John
def T : ℕ := 16
def J : ℕ := 36

-- Condition that John will be 2 times Tom's age in 4 years
def john_twice_tom_in_4_years (J T : ℕ) : Prop := J + 4 = 2 * (T + 4)

-- The number of years ago John was thrice as old as Tom
def years_ago (J T x : ℕ) : Prop := J - x = 3 * (T - x)

-- Prove that the number of years ago John was thrice as old as Tom is 6
theorem john_thrice_tom_years_ago (h1 : john_twice_tom_in_4_years 36 16) : years_ago 36 16 6 :=
by
  -- Import initial values into the context
  unfold john_twice_tom_in_4_years at h1
  unfold years_ago
  -- Solve the steps, more details in the actual solution
  sorry

end john_thrice_tom_years_ago_l684_684004


namespace combined_investment_yield_l684_684858

-- Definitions of the conditions
def yield_A : ℝ := 0.14
def value_A : ℝ := 500

def yield_B : ℝ := 0.08
def value_B : ℝ := 750

def yield_C : ℝ := 0.12
def value_C : ℝ := 1000

-- Complete the proof
theorem combined_investment_yield :
  let income_A := yield_A * value_A in
  let income_B := yield_B * value_B in
  let income_C := yield_C * value_C in
  let total_income := income_A + income_B + income_C in
  let total_value := value_A + value_B + value_C in
  (total_income / total_value) = 0.1111 :=
by
  -- establish the required calculations
  let income_A := yield_A * value_A
  let income_B := yield_B * value_B
  let income_C := yield_C * value_C
  let total_income := income_A + income_B + income_C
  let total_value := value_A + value_B + value_C
  -- reduced form of our problem
  have h : (total_income / total_value) = 0.1111 := sorry
  exact h

end combined_investment_yield_l684_684858


namespace Zhang_Qiang_distance_when_bus_started_l684_684562

theorem Zhang_Qiang_distance_when_bus_started :
  ∀ (speed_zhang : ℕ) (speed_bus : ℕ) (stop_time : ℕ) (interval : ℕ) (lead_time : ℕ) (time_to_catch_up : ℕ)
  (effective_speed_bus : ℤ) (relative_speed : ℤ) (initial_distance : ℤ),
  speed_zhang = 250 →
  speed_bus = 450 →
  stop_time = 1 →
  interval = 6 →
  lead_time = 15 →
  time_to_catch_up = 15 →
  effective_speed_bus = (speed_bus * interval) / (interval + stop_time) →
  relative_speed = effective_speed_bus - speed_zhang →
  initial_distance = speed_zhang * lead_time →
  initial_distance = 2100 :=
by
  -- noting variables
  intros speed_zhang speed_bus stop_time interval lead_time time_to_catch_up
         effective_speed_bus relative_speed initial_distance hz hb hs_hi hz_sr hi
  -- Given conditions: 
  have hz : speed_zhang = 250 := by assumption
  have hb : speed_bus = 450 := by assumption
  have hi : (interval + stop_time) ≠ 0 := by linarith
  have h_effective_speed_bus : effective_speed_bus = (speed_bus * interval) / (interval + stop_time) :=
    by rw hi; exact effective_speed_bus
  have h_relative_speed : relative_speed = effective_speed_bus - speed_zhang := by rw relative_speed

  have h_initial_distance : initial_distance = speed_zhang * lead_time := by exact initial_distance

  -- Final assertion
  rw [hz, hb, hs_hi, hz_sr]
  have final_distance := (250 * 15)
  have : final_distance = 2100 := by  exact 2100 -- placeholdered as per the given problem
  exact sorry -- relying on assumptions.assert final result

end Zhang_Qiang_distance_when_bus_started_l684_684562


namespace volume_of_sphere_in_cube_l684_684605

theorem volume_of_sphere_in_cube (r : ℝ)
  (h1 : ∃ center, ∀ edge, is_edge_of_cube_and_sphere_touches r edge)
  (h2 : ∀ (a : ℝ), a = 3 → edge_length_of_cube edge a)
  (h3 : ∀ (d : ℝ), d = 3 * (sqrt 2)) :
  (V = 4 / 3 * π * r^3) :=
by
  have r := (3 * sqrt 2) / 2
  have V := 4 / 3 * π * r^3
  exact V

end volume_of_sphere_in_cube_l684_684605


namespace total_cakes_served_l684_684234

-- Define the conditions as constants
constant cakes_served_lunch_today : ℕ := 5
constant cakes_served_dinner_today : ℕ := 6
constant cakes_served_yesterday : ℕ := 3

-- Define the theorem
theorem total_cakes_served :
  cakes_served_lunch_today + cakes_served_dinner_today + cakes_served_yesterday = 14 :=
by
  -- Add steps to reach the conclusion
  sorry

end total_cakes_served_l684_684234


namespace mr_brown_selling_price_l684_684466

noncomputable def initial_price : ℝ := 100000
noncomputable def profit_percentage : ℝ := 0.10
noncomputable def loss_percentage : ℝ := 0.10

def selling_price_mr_brown (initial_price profit_percentage : ℝ) : ℝ :=
  initial_price * (1 + profit_percentage)

def selling_price_to_friend (selling_price_mr_brown loss_percentage : ℝ) : ℝ :=
  selling_price_mr_brown * (1 - loss_percentage)

theorem mr_brown_selling_price :
  selling_price_to_friend (selling_price_mr_brown initial_price profit_percentage) loss_percentage = 99000 :=
by
  sorry

end mr_brown_selling_price_l684_684466


namespace choose_4_from_15_with_restriction_l684_684404

-- Definitions based on the problem conditions
def total_number_of_ways_select_4_from_15 : ℕ := Nat.choose 15 4
def number_of_ways_select_2_from_remaining_13 : ℕ := Nat.choose 13 2

-- Statement of the problem in Lean 4
theorem choose_4_from_15_with_restriction (A B : Type*) [DecidableEq A] [Fintype A] (club : Finset A)
  (h1 : Fintype.card club = 15) (h2 : A ≠ B) :
  let total_ways := total_number_of_ways_select_4_from_15
  let restricted_ways := number_of_ways_select_2_from_remaining_13
  total_ways - restricted_ways = 1287 := 
by 
  let total_ways := total_number_of_ways_select_4_from_15
  let restricted_ways := number_of_ways_select_2_from_remaining_13
  sorry

end choose_4_from_15_with_restriction_l684_684404


namespace next_correct_time_l684_684950

def clock_shows_correct_time (start_date : String) (start_time : String) (time_lost_per_hour : Int) : String :=
  if start_date = "March 21" ∧ start_time = "12:00 PM" ∧ time_lost_per_hour = 25 then
    "June 1, 12:00 PM"
  else
    "unknown"

theorem next_correct_time :
  clock_shows_correct_time "March 21" "12:00 PM" 25 = "June 1, 12:00 PM" :=
by sorry

end next_correct_time_l684_684950


namespace region_probability_l684_684640

theorem region_probability :
  ∀ (x y : ℝ),
  (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ e) →
  (x + y ≥ 1) ∧ (e^x - y ≥ 0) →
  let N : ℝ := 1 * e in
  let M : ℝ := 1.965 in
  M / N = 1 - 3 / (2 * e) := 
by 
  sorry

end region_probability_l684_684640


namespace measurable_length_l684_684611

-- Definitions of lines, rays, and line segments

-- A line is infinitely long with no endpoints.
def isLine (l : Type) : Prop := ∀ x y : l, (x ≠ y)

-- A line segment has two endpoints and a finite length.
def isLineSegment (ls : Type) : Prop := ∃ a b : ls, a ≠ b ∧ ∃ d : ℝ, d > 0

-- A ray has one endpoint and is infinitely long.
def isRay (r : Type) : Prop := ∃ e : r, ∀ x : r, x ≠ e

-- Problem statement
theorem measurable_length (x : Type) : isLineSegment x → (∃ d : ℝ, d > 0) :=
by
  -- Proof is not required
  sorry

end measurable_length_l684_684611


namespace find_points_l684_684716

def acute_triangle (A B C : ℝ × ℝ × ℝ) : Prop :=
  -- Definition to ensure that the triangle formed by A, B, and C is an acute-angled triangle.
  sorry -- This would be formalized ensuring all angles are less than 90 degrees.

def no_three_collinear (A B C D E : ℝ × ℝ × ℝ) : Prop :=
  -- Definition that ensures no three points among A, B, C, D, and E are collinear.
  sorry

def line_normal_to_plane (P Q R S : ℝ × ℝ × ℝ) : Prop :=
  -- Definition to ensure that the line through any two points P, Q is normal to the plane containing R, S, and the other point.
  sorry

theorem find_points (A B C : ℝ × ℝ × ℝ) (h_acute : acute_triangle A B C) :
  ∃ (D E : ℝ × ℝ × ℝ), no_three_collinear A B C D E ∧
    (∀ (P Q R R' : ℝ × ℝ × ℝ), 
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E) →
      (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E) →
      (R' = A ∨ R' = B ∨ R' = C ∨ R' = D ∨ R' = E) →
      P ≠ Q → Q ≠ R → R ≠ R' →
      line_normal_to_plane P Q R R') :=
sorry

end find_points_l684_684716


namespace correct_statements_l684_684440

/-- Let f(x) be an even function defined on ℝ such that f(x + 1) = f(x - 1) for any x in ℝ. 
  Given that f(x) = 2^x for x in [0, 1], determine which of the following statements are true:
  1. 2 is a period of the function f(x);
  2. The function f(x) is decreasing on (1, 2) and increasing on (2, 3);
  3. The maximum value of the function f(x) is 1, and the minimum value is 0.
-/
theorem correct_statements (f : ℝ → ℝ)
  (h1 : ∀ x, f x = f (-x))
  (h2 : ∀ x, f (x + 1) = f (x - 1))
  (h3 : ∀ x ∈ set.Icc 0 1, f x = 2 ^ x) :
  list ℕ := 
if (∀ x, f (x + 2) = f x) ∧ 
   (∀ x, (1 < x ∧ x < 2 → f (x - 1) > f x) ∧ (2 < x ∧ x < 3 → f (x - 1) < f x)) then 
  [1, 2]
else []

example : correct_statements f _ _ _ = [1, 2] :=
by sorry

end correct_statements_l684_684440


namespace fitted_bowling_ball_volume_l684_684203

theorem fitted_bowling_ball_volume :
  let r_bowl := 20 -- radius of the bowling ball in cm
  let r_hole1 := 1 -- radius of the first hole in cm
  let r_hole2 := 2 -- radius of the second hole in cm
  let r_hole3 := 2 -- radius of the third hole in cm
  let depth := 10 -- depth of each hole in cm
  let V_bowl := (4/3) * Real.pi * r_bowl^3
  let V_hole1 := Real.pi * r_hole1^2 * depth
  let V_hole2 := Real.pi * r_hole2^2 * depth
  let V_hole3 := Real.pi * r_hole3^2 * depth
  let V_holes := V_hole1 + V_hole2 + V_hole3
  let V_fitted := V_bowl - V_holes
  V_fitted = (31710 / 3) * Real.pi :=
by sorry

end fitted_bowling_ball_volume_l684_684203


namespace construct_triangle_l684_684636

theorem construct_triangle (A B C O : Point) (circumcircle : ∀ P : Triangle ABC, O = circumcenter P) 
  (euler_line : Line) (parallel_to_ext_bisector_A : parallel euler_line (external_bisector_angle_A A B C))
  (equal_segments : ∀ P : Point, belongs P euler_line → distance A P = distance A (symmetric_counterpart P A B C)) :
  (∀ P1 P2 P3 : Point, triangle A B C other_info P1 P2 P3) :=
by 
  sorry

end construct_triangle_l684_684636


namespace projection_of_sum_on_a_l684_684675

variables {𝕜 : Type*} [real_field 𝕜] [inner_product_space 𝕜 (euclidean_space 𝕜)] 
variables {a b : euclidean_space 𝕜}
def magnitude (v : euclidean_space 𝕜) : 𝕜 := real.sqrt (inner_product_space.to_inner v v)

noncomputable def angle (v₁ v₂ : euclidean_space 𝕜) : 𝕜 :=
(real.arccos ((inner_product_space.to_inner v₁ v₂) / (magnitude v₁ * magnitude v₂)))

theorem projection_of_sum_on_a (h_a : magnitude a = 2) (h_b : magnitude b = 2) (h_angle : angle a b = real.pi / 3) :
  (magnitude (a + b)) * real.cos (angle (a + b) a) = 3 :=
sorry

end projection_of_sum_on_a_l684_684675


namespace why_build_offices_l684_684288

structure Company where
  name : String
  hasSkillfulEmployees : Prop
  uniqueComfortableWorkEnvironment : Prop
  integratedWorkLeisureSpaces : Prop
  reducedEmployeeStress : Prop
  flexibleWorkSchedules : Prop
  increasesProfit : Prop

theorem why_build_offices (goog_fb : Company)
  (h1 : goog_fb.hasSkillfulEmployees)
  (h2 : goog_fb.uniqueComfortableWorkEnvironment)
  (h3 : goog_fb.integratedWorkLeisureSpaces)
  (h4 : goog_fb.reducedEmployeeStress)
  (h5 : goog_fb.flexibleWorkSchedules) :
  goog_fb.increasesProfit := 
sorry

end why_build_offices_l684_684288


namespace cone_surface_area_l684_684208

theorem cone_surface_area (h l : ℝ) (h_pos : h > 0) (l_pos : l > 0) : 
  (π * (h^3) / l) = (surface_area_of_cone h l) :=
sorry

end cone_surface_area_l684_684208


namespace sufficient_but_not_necessary_l684_684022

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1/2 → 2 * x^2 + x - 1 > 0) ∧ ¬(2 * x^2 + x - 1 > 0 → x > 1 / 2) := 
by
  sorry

end sufficient_but_not_necessary_l684_684022


namespace range_of_a_no_real_roots_l684_684360

theorem range_of_a_no_real_roots (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 + ax + 1 = 0)) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_no_real_roots_l684_684360


namespace worker_surveys_per_week_l684_684607

theorem worker_surveys_per_week :
  let regular_rate := 30
  let cellphone_rate := regular_rate + 0.20 * regular_rate
  let surveys_with_cellphone := 50
  let earnings := 3300
  cellphone_rate = regular_rate + 0.20 * regular_rate →
  earnings = surveys_with_cellphone * cellphone_rate →
  regular_rate = 30 →
  surveys_with_cellphone = 50 →
  earnings = 3300 →
  surveys_with_cellphone = 50 := sorry

end worker_surveys_per_week_l684_684607


namespace sum_fn_eq_exp_exp_l684_684638

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => Real.exp x
| (n + 1), x => x * (f n x).deriv

theorem sum_fn_eq_exp_exp :
  (∑' n, (f n 1) / n.factorial) = Real.exp (Real.exp 1) :=
by
  sorry

end sum_fn_eq_exp_exp_l684_684638


namespace solve_for_s_l684_684270

def F (x y z : ℝ) := x * y^z

theorem solve_for_s : ∃ s > 0, F s s 2 = 144 ∧ s = 2^(4/3) * 3^(2/3) :=
by
  sorry

end solve_for_s_l684_684270


namespace first_three_digits_right_of_decimal_l684_684916

-- Define the number (10^2003 + 1) and its exponent (11/7)
def base := 10 ^ 2003 + 1
def exponent := 11 / 7

-- Define the target number as (10^2003 + 1) ^ (11 / 7)
noncomputable def target := base ^ exponent

-- State the goal to prove
theorem first_three_digits_right_of_decimal (digits := "571") :
  let dec_part := (target % 10 ^ (-3)) * 10 ^ 3 in
  dec_part = 571 :=
sorry

end first_three_digits_right_of_decimal_l684_684916


namespace probability_three_draws_exceeds_nine_l684_684582

/-- A box contains 6 chips numbered from 1 to 6. Chips are drawn randomly one at a time 
    without replacement until the sum of the values drawn exceeds 9. 
    Prove that the probability that exactly 3 draws are required is 3/5. -/
theorem probability_three_draws_exceeds_nine : 
  (∃ (chips : set ℕ) (h1 : chips = {1, 2, 3, 4, 5, 6}), 
   ∃ (draws : list ℕ) (h2 : ∀ i, draws.nth i ∈ chips ∧ (draws.nodup ∧ draws.sum > 9)), 
   (draws.length = 3 ∧ proba_of_three_draws_3_5) := sorry

end probability_three_draws_exceeds_nine_l684_684582


namespace partition_students_l684_684112

theorem partition_students (V : Type) (E : V → V → Prop) (h_symmetric : symmetric E) (h_count : (finset.pairwise (finset.univ : finset V) E).card = k) :
  ∃ (V1 V2 : finset V), disjoint V1 V2 ∧ V1 ∪ V2 = finset.univ ∧
  (finset.pairwise V1 E).card ≤ k / 3 ∧
  (finset.pairwise V2 E).card ≤ k / 3 :=
sorry

end partition_students_l684_684112


namespace find_positive_integer_l684_684555

variable (z : ℕ)

theorem find_positive_integer
  (h1 : (4 * z)^2 - z = 2345)
  (h2 : 0 < z) :
  z = 7 :=
sorry

end find_positive_integer_l684_684555


namespace find_f_neg_a_l684_684322

def f (x : ℝ) : ℝ := x + (1 / x) - 1

theorem find_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -4 := sorry

end find_f_neg_a_l684_684322


namespace companies_increase_profitability_l684_684290

-- Define the three main arguments as conditions
def increased_retention_and_attraction : Prop := 
  ∀ (company : Type), ∃ (environment : Type),
    (company = Google ∨ company = Facebook) → 
    environment provides_comfortable_and_flexible_workplace → 
    increases_profitability company

def enhanced_productivity_and_creativity : Prop := 
  ∀ (company : Type), ∃ (environment : Type),
    (company = Google ∨ company = Facebook) → 
    environment provides_work_and_leisure_integration → 
    increases_profitability company

def better_work_life_integration : Prop := 
  ∀ (company : Type), ∃ (environment : Type),
    (company = Google ∨ company = Facebook) → 
    environment provides_work_life_integration → 
    increases_profitability company

-- Define the overall theorem we need to prove
theorem companies_increase_profitability 
  (company : Type)
  (h1 : increased_retention_and_attraction)
  (h2 : enhanced_productivity_and_creativity)
  (h3 : better_work_life_integration) 
: ∃ (environment : Type), 
    (company = Google ∨ company = Facebook) → 
    environment enhances_work_environment → 
    increases_profitability company :=
by
  sorry

end companies_increase_profitability_l684_684290


namespace monthly_income_of_labourer_l684_684068

variable (I : ℕ) -- Monthly income

-- Conditions: 
def condition1 := (85 * 6) - (6 * I) -- A boolean expression depicting the labourer fell into debt
def condition2 := (60 * 4) + (85 * 6 - 6 * I) + 30 -- Total income covers debt and saving 30

-- Statement to be proven
theorem monthly_income_of_labourer : 
  ∃ I : ℕ, condition1 I = 0 ∧ condition2 I = 4 * I → I = 78 :=
by
  sorry

end monthly_income_of_labourer_l684_684068


namespace largest_number_with_hcf_lcm_factors_l684_684569

theorem largest_number_with_hcf_lcm_factors (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) (largest_number : ℕ)
  (hcf_eq : hcf = 40) (factor1_eq : factor1 = 11) (factor2_eq : factor2 = 12) (largest_number_eq : largest_number = 480) :
  largest_number = hcf * factor2 := 
by 
  rw [hcf_eq, factor2_eq, largest_number_eq]
  exact rfl

end largest_number_with_hcf_lcm_factors_l684_684569


namespace line_passes_second_intersection_l684_684528

--definitions based on the problem statement
structure Circle := 
  (center : Point) 
  (radius : ℝ)

structure Point := 
  (x : ℝ)
  (y : ℝ)

def passes_through (L : Line) (P : Point) : Prop := sorry

def touches_internally (C1 C2 : Circle) : Prop := sorry

def second_intersection_point (C1 C2 : Circle) (S : Point) : Point := sorry

variable (S T1 T2 T3 : Point)
variable (ω ω1 ω2 ω3 : Circle)
variable (R r : ℝ)

--constraints on the circles, points and radii
axiom h_radius_ω : ω.radius = R
axiom h_radius_ω1 : ω1.radius = r
axiom h_radius_ω2 : ω2.radius = r
axiom h_radius_ω3 : ω3.radius = r
axiom h_passing_ω1 : ω1.center ≠ ω.center
axiom h_passing_ω2 : ω2.center ≠ ω.center
axiom h_passing_ω3 : ω3.center ≠ ω.center
axiom h_touch_ω1 : touches_internally ω ω1
axiom h_touch_ω2 : touches_internally ω ω2
axiom h_touch_ω3 : touches_internally ω ω3
axiom h_point_S : passes_through ω1 S ∧ passes_through ω2 S ∧ passes_through ω3 S
axiom h_touch_point1 : passes_through ω1 T1
axiom h_touch_point2 : passes_through ω2 T2
axiom h_touch_point3 : passes_through ω3 T3

-- theorem statement
theorem line_passes_second_intersection :
  let M := second_intersection_point ω1 ω2 S in
  passes_through (Line.mk T1 T2) M :=
sorry

end line_passes_second_intersection_l684_684528


namespace multiple_of_power_of_two_l684_684056

theorem multiple_of_power_of_two (n : ℕ) (h : n ≥ 1) : 
  ∃ a : ℕ, (∀ d ∈ (nat.digits 10 a), d = 1 ∨ d = 2) ∧ (nat.num_digits 10 a = n) ∧ (2^n ∣ a) :=
sorry

end multiple_of_power_of_two_l684_684056


namespace ani_winning_strategy_for_even_n_ge_6_l684_684855

open Nat

-- Lean statement of proof problem
theorem ani_winning_strategy_for_even_n_ge_6 (n : ℕ) (h_n : n ≥ 6 ∧ n % 2 = 0):
  ∃ (a b c : ℕ), a + b + c = n ∧ (min a (min b c) ≥ 1) ∧ (
    ∀ (budi_turn: Π (x: ℕ), x = 1 ∨ x = 2 ∨ x = 3 → ∃ (a' b' c' : ℕ), a' + b' + c' = n - x ∧ (min a' (min b' c') ≥ 0)), 
    ∃ (ani_winning: Π (x: ℕ), x = 1 ∨ x = 2 ∨ x = 3 → ∃ (a'' b'' c'' : ℕ), a'' + b'' + c'' = n - x ∧ (min a'' (min b'' c'') ≥ 0) ∧
    ∀ (a b c : ℕ), a + b + c = n → 
    ((a ≡ 0 [MOD 4]) ∧ (b ≡ 0 [MOD 4]) ∧ (c ≡ 0 [MOD 4]) ∨ 
    (a ≡ 1 [MOD 4] ∧ b ≡ 2 [MOD 4] ∧ c ≡ 3 [MOD 4]) ∨ 
    (a ≡ 2 [MOD 4] ∧ b ≡ 3 [MOD 4] ∧ c ≡ 1 [MOD 4]) ∨ 
    (a ≡ 3 [MOD 4] ∧ b ≡ 1 [MOD 4] ∧ c ≡ 2 [MOD 4]) → false
  ))) := 
sorry

end ani_winning_strategy_for_even_n_ge_6_l684_684855


namespace calculate_xy_l684_684688

theorem calculate_xy (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x * y = 32 :=
by
  sorry

end calculate_xy_l684_684688


namespace successful_balls_count_l684_684823

-- Define the initial conditions
def totalBalls : Nat := 100
def ballsWithHoles : Nat := 40 * totalBalls / 100
def ballsRemainAfterHoles : Nat := totalBalls - ballsWithHoles
def overinflatedBalls : Nat := 20 * ballsRemainAfterHoles / 100
def successfullyInflatedBalls : Nat := ballsRemainAfterHoles - overinflatedBalls

-- The theorem we are proving
theorem successful_balls_count : successfullyInflatedBalls = 48 := by
  unfold totalBalls ballsWithHoles ballsRemainAfterHoles overinflatedBalls successfullyInflatedBalls
  rfl

end successful_balls_count_l684_684823


namespace find_cos_alpha_l684_684671

theorem find_cos_alpha
  (α : ℝ)
  (h1 : sin (α + π / 6) = -4 / 5)
  (h2 : -π / 2 < α ∧ α < 0) :
  cos α = (3 * sqrt 3 - 4) / 10 :=
by
  sorry

end find_cos_alpha_l684_684671


namespace sum_int_values_binom_eq_l684_684127

theorem sum_int_values_binom_eq :
  (∑ n in finset.range 16, if (Nat.choose 15 n + Nat.choose 15 7 = Nat.choose 16 8) then n else 0) = 8 := by
  sorry

end sum_int_values_binom_eq_l684_684127


namespace function_proof_l684_684080

noncomputable def function_properties {α : Type*} (f : α → ℝ) (a b : α) :=
  f a = f b ∧ ∀ {x y : α}, x ≠ y → |f x - f y| < |x - y|

theorem function_proof {f : ℝ → ℝ} (h0 : function_properties f 0 1) :
  ∀ x₁ x₂ ∈ Icc 0 1, x₁ ≠ x₂ → |f x₂ - f x₁| < 1 / 2 :=
by
  intro x₁ hx₁ x₂ hx₂ hneq
  -- start the proof here
  sorry

end function_proof_l684_684080


namespace days_engaged_on_job_l684_684218

-- Definitions for the conditions
variable (x y : ℕ)
-- y is given to be 7
def y_is_7 : y = 7 := rfl

-- The net earnings equation provided in the problem
def net_earnings : 10 * x - 2 * y = 216 

-- The theorem to prove: x = 23 given the conditions
theorem days_engaged_on_job : net_earnings x y ∧ y_is_7 y → x = 23 := by
  sorry

end days_engaged_on_job_l684_684218


namespace find_k_l684_684726

theorem find_k (a k : ℤ) (h1 : 3^a = 0.618) (h2 : a ∈ set.Ico k (k+1)) : k = -1 :=
sorry

end find_k_l684_684726


namespace units_digit_of_50_factorial_is_0_l684_684166

theorem units_digit_of_50_factorial_is_0 : 
  (∃ n : ℕ, 50! ≡ n [MOD 10]) ∧ (n = 0) := sorry

end units_digit_of_50_factorial_is_0_l684_684166


namespace cos_plus_sin_value_l684_684258

theorem cos_plus_sin_value :
  cos (π / 2 + π / 3) + sin (-π - π / 6) = - (√3 / 2) - (1 / 2) :=
by
  sorry -- To be proved

end cos_plus_sin_value_l684_684258


namespace sum_of_solutions_l684_684554

theorem sum_of_solutions : 
  let eq := (4 * x + 3) * (3 * x - 8) = 0
  in (eq -> sum_of_roots eq = 23 / 12) :=
by {
  -- Definition of the equation for convenience
  let eq := (4 * x + 3) * (3 * x - 8) = 0;
  -- We skip the proof here
  sorry
}

end sum_of_solutions_l684_684554


namespace min_small_bottles_needed_l684_684236

theorem min_small_bottles_needed (small_capacity large_capacity : ℕ) 
    (h_small_capacity : small_capacity = 35) (h_large_capacity : large_capacity = 500) : 
    ∃ n, n = 15 ∧ large_capacity <= n * small_capacity :=
by 
  sorry

end min_small_bottles_needed_l684_684236


namespace width_to_length_ratio_l684_684772

variables {w l P : ℕ}

theorem width_to_length_ratio :
  l = 10 → P = 30 → P = 2 * (l + w) → (w : ℚ) / l = 1 / 2 :=
by
  intro h1 h2 h3
  -- Noncomputable definition for rational division
  -- (ℚ is used for exact rational division)
  sorry

#check width_to_length_ratio

end width_to_length_ratio_l684_684772


namespace library_book_configurations_l684_684959

def number_of_valid_configurations (total_books : ℕ) (min_in_library : ℕ) (min_checked_out : ℕ) : ℕ :=
  (total_books - (min_in_library + min_checked_out + 1)) + 1

theorem library_book_configurations : number_of_valid_configurations 8 2 2 = 5 :=
by
  -- Here we would write the Lean proof, but since we are only interested in the statement:
  sorry

end library_book_configurations_l684_684959


namespace orchard_harvest_l684_684881

theorem orchard_harvest (sacks_per_section : ℕ) (sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 → sections = 8 → total_sacks = sacks_per_section * sections → total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end orchard_harvest_l684_684881


namespace community_children_count_l684_684874

theorem community_children_count (total_members : ℕ) (pct_adult_men : ℝ) (ratio_adult_women : ℝ) :
  total_members = 2000 → pct_adult_men = 0.3 → ratio_adult_women = 2 →
  let num_adult_men := (pct_adult_men * total_members).to_nat in
  let num_adult_women := (ratio_adult_women * num_adult_men).to_nat in
  let total_adults := num_adult_men + num_adult_women in
  let num_children := total_members - total_adults in
  num_children = 200 :=
by
  intro h1 h2 h3
  let num_adult_men := (0.3 * 2000).to_nat
  let num_adult_women := (2 * num_adult_men).to_nat
  let total_adults := num_adult_men + num_adult_women
  let num_children := 2000 - total_adults
  -- we need to skip the proof part
  sorry

end community_children_count_l684_684874


namespace average_people_per_hour_l684_684416

theorem average_people_per_hour 
    (total_people : ℕ) 
    (days : ℕ) 
    (hours_per_day : ℕ) 
    (h1 : total_people = 3000) 
    (h2 : days = 5) 
    (h3 : hours_per_day = 24) :
    total_people / (days * hours_per_day) = 25 :=
begin
    -- The proof would go here, but it is omitted as per the instructions.
    sorry
end

end average_people_per_hour_l684_684416


namespace sarah_earnings_l684_684478

-- Conditions
def monday_hours : ℚ := 1 + 3 / 4
def wednesday_hours : ℚ := 65 / 60
def thursday_hours : ℚ := 2 + 45 / 60
def friday_hours : ℚ := 45 / 60
def saturday_hours : ℚ := 2

def weekday_rate : ℚ := 4
def weekend_rate : ℚ := 6

-- Definition for total earnings
def total_weekday_earnings : ℚ :=
  (monday_hours + wednesday_hours + thursday_hours + friday_hours) * weekday_rate

def total_weekend_earnings : ℚ :=
  saturday_hours * weekend_rate

def total_earnings : ℚ :=
  total_weekday_earnings + total_weekend_earnings

-- Statement to prove
theorem sarah_earnings : total_earnings = 37.3332 := by
  sorry

end sarah_earnings_l684_684478


namespace num_elements_M_l684_684028

open Set 

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {2, 3, 4}

def M : Set ℕ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a * b}

theorem num_elements_M : (M : Set ℕ).toFinset.card = 7 :=
by {
  sorry
}

end num_elements_M_l684_684028


namespace find_number_l684_684961

theorem find_number (x : ℕ) : 
  (let sum := 555 + 445 in
   let diff := 555 - 445 in
   x / sum = 2 * diff ∧ x % sum = 25) → 
  x = 220025 :=
by
  intro h
  have sum_eq : 555 + 445 = 1000 := rfl
  have diff_eq : 555 - 445 = 110 := rfl
  have quotient_eq := mul_nat.trans (mul_nat.trans h.1 sum_eq) diff_eq
  have remainder_eq := h.2
  calc
    x = (1000 * 220) + 25 : by rw [quotient_eq, remainder_eq]
    ... = 220025 : by norm_num

end find_number_l684_684961


namespace black_cars_count_l684_684906

noncomputable def total_cars : ℕ := 1824
noncomputable def blue_cars : ℕ := (2 / 5 : ℚ) * total_cars
noncomputable def red_cars : ℕ := (1 / 3 : ℚ) * total_cars
noncomputable def green_cars : ℕ := (1 / 8 : ℚ) * total_cars
noncomputable def non_black_cars : ℕ := blue_cars + red_cars + green_cars
noncomputable def black_cars : ℕ := total_cars - non_black_cars

theorem black_cars_count : black_cars = 259 :=
by
  sorry

end black_cars_count_l684_684906


namespace part1_part2_part3_l684_684677

noncomputable def f (x : ℝ) := (1/3) * x^3 - x

def g (x : ℝ) := 33 * f(x) + 3 * x

theorem part1:
  ∀ x : ℝ, f(x) = (1/3) * x^3 - x :=
sorry

theorem part2 (hx : x > 0):
  (1 + 1 / g(x)) ^ g(x) < Real.exp 1 :=
sorry

noncomputable def b (n : ℕ) : ℝ := g(n)^(1/g(n + 1))

theorem part3:
  ∃ n m : ℕ, n ≠ m ∧ b(n) = b(m) :=
  exists.intro 2 (exists.intro 8 (and.intro (by norm_num) sorry))

end part1_part2_part3_l684_684677


namespace find_pirates_l684_684226

def pirate_problem (p : ℕ) (nonfighters : ℕ) (arm_loss_percent both_loss_percent : ℝ) (leg_loss_fraction : ℚ) : Prop :=
  let participants := p - nonfighters in
  let arm_loss := arm_loss_percent * participants in
  let both_loss := both_loss_percent * participants in
  let leg_loss (p : ℕ) := leg_loss_fraction * p in
  let only_leg_loss (p : ℕ) := leg_loss p - both_loss in
  arm_loss + only_leg_loss p + both_loss = leg_loss p

theorem find_pirates : ∃ p : ℕ, pirate_problem p 10 0.54 0.34 (2/3) :=
sorry

end find_pirates_l684_684226


namespace nadia_distance_graph_l684_684820

theorem nadia_distance_graph :
  ∃ (graph : ℕ → ℝ),
    (∀ t : ℕ, 0 ≤ graph t) ∧
    (graph 0 = 0) ∧
    (graph (1/4) = 1) ∧
    (graph (1/2) = real.sqrt 2) ∧
    (graph (3/4) = 1) ∧
    (graph 1 = 0) ∧
    (∀ t : ℝ, (t ≤ 1/4 ∨ (3/4 ≤ t ∧ t ≤ 1)) → graph t = t * 4) ∧
    (∀ t : ℝ, (1/4 ≤ t ∧ t ≤ 3/4) → graph t = real.sqrt (2 - 4 * (t - 1/2)^2)) →
  graph = λ t, match t with
                 | x if x >= 0 ∧ x < 1/4 => 4 * x
                 | x if x >= 1/4 ∧ x < 1/2 => real.sqrt 2
                 | x if x >= 1/2 ∧ x < 3/4 => real.sqrt 2
                 | x if x >= 3/4 ∧ x < 1 => 4 * x - 3
                 | _ => 0
               end := sorry

end nadia_distance_graph_l684_684820


namespace tom_speed_B_to_C_l684_684911

theorem tom_speed_B_to_C :
  ∀ (d v : ℝ),
  (60 : ℝ) > 0 ∧
  (36 : ℝ) > 0 ∧
  (2 * d : ℝ) / 60 + d / v = 3 * d / 36 →
  v = 64.8 :=
begin
  intros d v h,
  sorry
end

end tom_speed_B_to_C_l684_684911


namespace mary_trip_duration_l684_684815

theorem mary_trip_duration :
  let uber_home := 10 -- in minutes
  let uber_airport := 5 * uber_home
  let check_bag := 15
  let sec_check := 3 * check_bag
  let wait_boarding := 20
  let wait_takeoff := 2 * wait_boarding
  let pre_flight := uber_home + uber_airport + check_bag + sec_check + wait_boarding + wait_takeoff
  let layover1 := 3 * 60 + 25
  let delay := 45
  let layover2 := 1 * 60 + 50
  let layovers := layover1 + delay + layover2
  let timezone_diff := 3 * 60
  let total_minutes := pre_flight + layovers + timezone_diff
  let total_hours := total_minutes / 60
  in total_hours = 12 := 
by 
  sorry

end mary_trip_duration_l684_684815


namespace count_polynomials_l684_684453

noncomputable def numDistRootsPolys (n : ℕ) := sorry

theorem count_polynomials : numDistRootsPolys 528 = 132 := sorry

end count_polynomials_l684_684453


namespace hyperbola_asymptote_slope_l684_684081

theorem hyperbola_asymptote_slope :
  (∀ x y : ℝ, (x^2 / 100 - y^2 / 64 = 1) → y = (4/5) * x ∨ y = -(4/5) * x) :=
by
  sorry

end hyperbola_asymptote_slope_l684_684081


namespace min_components_ineq_l684_684429

theorem min_components_ineq (n l : ℕ) :
  ∀ (A : set (Σ (i j : ℕ), i < n ∧ j < n)),
    (∃ A' : finset (Σ (i j : ℕ), i < n ∧ j < n),
      A' ∈ A ∧ A'.card ≥ l) →
    (let V : finset (Σ (i j : ℕ), i < n ∧ j < n) := {v | ∃ e ∈ A, v = e.1} in
     let j : ℕ := sorry in  -- definition of j(A), the number of connected components.
     ∃ (min_val : ℕ), min_val = |V|.card - j ∧
       ∀ m, m = min_val →
         ∃ min_bound max_bound,
           min_bound = l / 2 ∧
           max_bound = l / 2 + (nat.sqrt (l / 2)) + 1 ∧
           min_bound ≤ m ∧ m ≤ max_bound) :=
sorry  -- proof goes here

end min_components_ineq_l684_684429


namespace m_value_l684_684363

-- Definitions of the lines based on the given conditions
def line1 (m : ℝ) : ℝ → ℝ → Prop := λ x y, x + (1 + m) * y = 2 - m
def line2 (m : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * m * x + 4 * y = -16

-- Parallelism condition for two lines
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x y, l1 x y → l2 (k * x) (k * y))

-- The main statement to prove
theorem m_value (m : ℝ) (h : parallel (line1 m) (line2 m)) : m = 1 :=
  sorry

end m_value_l684_684363


namespace symmetric_point_F1_ellipse_C_l684_684597

noncomputable def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)
def line_l (x y : ℝ) : Prop := x + 2*y + 6 = 0
def symmetric_point (F : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ :=
  let (x, y) := F in
  let (x', y') := (2 * (2 * y - x + 6) / 5 + x, 2 * (x - 2 * y - 6) / 5 + y) in
  (x', y')

theorem symmetric_point_F1 :
  symmetric_point F1 line_l = (-3, -4) :=
sorry

theorem ellipse_C :
  ∃ (a b : ℝ), a = 2 * Real.sqrt 2 ∧ b = Real.sqrt 7 ∧
  ∀ (x y : ℝ), (x^2 / 8 + y^2 / 7 = 1 → line_l x y → 
  let F'_1 := symmetric_point F1 line_l in
  dist' (x, y) F1 + dist' (x, y) F2 = dist' F'_1 F2) :=
sorry

end symmetric_point_F1_ellipse_C_l684_684597


namespace problem_l684_684673

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 4
noncomputable def c : ℝ := Real.log 9 / Real.log 4

theorem problem : a = c ∧ a > b :=
by
  sorry

end problem_l684_684673


namespace no_isometry_spherical_cap_to_plane_l684_684832

-- Define the spherical cap and the plane
namespace Geometry

def spherical_cap (S : Type) [metric_space S] := ∃ R : ℝ, ∀ A B : S, dist A B = 2 * R * real.asin ((dist A B) / (2 * R))
def plane (ℝ² : Type) [metric_space ℝ²] := ∀ A B : ℝ², dist A B = sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

-- Define the problem statement
theorem no_isometry_spherical_cap_to_plane {S ℝ² : Type} [metric_space S] [metric_space ℝ²] :
  spherical_cap S → plane ℝ² → ¬∃ f : S → ℝ², ∀ A B : S, dist A B = dist (f A) (f B) :=
by
  -- Add a placeholder for proof
  sorry

end no_isometry_spherical_cap_to_plane_l684_684832


namespace find_y_l684_684738

theorem find_y (x y : ℝ) (h₁ : x^2 - 2 * x + 5 = y + 3) (h₂ : x = 5) : y = 17 :=
by
  sorry

end find_y_l684_684738


namespace units_digit_factorial_50_l684_684132

theorem units_digit_factorial_50 : Nat.unitsDigit (Nat.factorial 50) = 0 := 
  sorry

end units_digit_factorial_50_l684_684132


namespace bob_sells_silver_cube_for_4455_l684_684625

noncomputable def side_length := 3 -- Condition 1: Side length of the cube
noncomputable def density := 6 -- Condition 2: Weight of a cubic inch of silver in ounces
noncomputable def price_per_ounce := 25 -- Condition 3: Price per ounce of silver in dollars
noncomputable def sale_percentage := 1.1 -- Condition 4: Selling price is 110% of the silver value

noncomputable def volume := side_length * side_length * side_length

noncomputable def weight := volume * density

noncomputable def silver_value := weight * price_per_ounce

noncomputable def selling_price := silver_value * sale_percentage

theorem bob_sells_silver_cube_for_4455 : selling_price = 4455 :=
by
  sorry

end bob_sells_silver_cube_for_4455_l684_684625


namespace calculate_sum_l684_684257

theorem calculate_sum : 5 * 12 + 7 * 15 + 13 * 4 + 6 * 9 = 271 :=
by
  sorry

end calculate_sum_l684_684257


namespace line_through_points_l684_684717

theorem line_through_points (a1 b1 a2 b2 : ℝ) :
  (2 * a1 + 3 * b1 + 1 = 0) →
  (2 * a2 + 3 * b2 + 1 = 0) →
  ∀ (x y : ℝ), (x = a1 ∧ y = b1) ∨ (x = a2 ∧ y = b2) → (2 * x + 3 * y + 1 = 0) :=
begin
  intros h1 h2 x y p,
  cases p;
  { subst_vars,
    assumption }
end

end line_through_points_l684_684717


namespace conjugate_of_complex_number_l684_684495

theorem conjugate_of_complex_number :
  let z := 1 / (1 - Complex.i) in Complex.conj z = (1 / 2) - (1 / 2) * Complex.i :=
by
  sorry

end conjugate_of_complex_number_l684_684495


namespace max_non_triangulated_segments_correct_l684_684616

open Classical

/-
Problem description:
Given an equilateral triangle divided into smaller equilateral triangles with side length 1, 
we need to define the maximum number of 1-unit segments that can be marked such that no 
triangular subregion has all its sides marked.
-/

def total_segments (n : ℕ) : ℕ :=
  (3 * n * (n + 1)) / 2

def max_non_triangular_segments (n : ℕ) : ℕ :=
  n * (n + 1)

theorem max_non_triangulated_segments_correct (n : ℕ) :
  max_non_triangular_segments n = n * (n + 1) := by sorry

end max_non_triangulated_segments_correct_l684_684616


namespace simplify_expression_l684_684059

theorem simplify_expression (x : ℝ) : 2 * (x - 3) - (-x + 4) = 3 * x - 10 :=
by
  -- The proof is omitted, so use sorry to skip it
  sorry

end simplify_expression_l684_684059


namespace sum_of_last_three_digits_9_pow_15_plus_15_pow_15_l684_684918

theorem sum_of_last_three_digits_9_pow_15_plus_15_pow_15 :
  (9 ^ 15 + 15 ^ 15) % 1000 = 24 :=
by
  sorry

end sum_of_last_three_digits_9_pow_15_plus_15_pow_15_l684_684918


namespace digits_right_of_decimal_l684_684567

theorem digits_right_of_decimal :
  let expr := (10 ^ 4 * 3.456789) in
  let result := expr ^ 9 in
  (result - Real.floor result) * 10 ^ 10 % 1 = 0 :=
by
  let expr := (10 ^ 4 * 3.456789)
  let result := expr ^ 9
  sorry

end digits_right_of_decimal_l684_684567


namespace min_cuts_to_pay_max_days_with_n_cuts_l684_684572

-- (a) Minimum number of rings he must cut
theorem min_cuts_to_pay (chain_rings days : ℕ) (H_chain : chain_rings = 11) (H_days : days = 11)
  (H_payment_increase : ∀ n, n < days -> n + 1) : ∃ cuts, cuts = 2 :=
sorry

-- (b) Maximum number of days the traveler can stay given n cuts
theorem max_days_with_n_cuts (cuts : ℕ) : ∃ rings, rings = (cuts + 1) * 2^cuts - 1 :=
sorry

end min_cuts_to_pay_max_days_with_n_cuts_l684_684572


namespace find_a_values_l684_684294

theorem find_a_values :
  ∀ (a : ℝ),
  (∃ (x₁ x₂ : ℝ),
    -√6 / 2 ≤ x₁ ∧ x₁ ≤ √2 ∧ -√6 / 2 ≤ x₂ ∧ x₂ ≤ √2 ∧ x₁ ≠ x₂ ∧
    ((1 - x₁^2)^2 + 2 * a^2 + 5 * a)^7 - ((3 * a + 2) * (1 - x₁^2) + 3)^7 = 
    5 - 2 * a - (3 * a + 2) * x₁^2 - 2 * a^2 - (1 - x₁^2)^2 ∧ 
    ((1 - x₂^2)^2 + 2 * a^2 + 5 * a)^7 - ((3 * a + 2) * (1 - x₂^2) + 3)^7 = 
    5 - 2 * a - (3 * a + 2) * x₂^2 - 2 * a^2 - (1 - x₂^2)^2) ↔ 
  (0.25 ≤ a ∧ a < 1) ∨ (-3.5 ≤ a ∧ a < -2) := sorry

end find_a_values_l684_684294


namespace avg_of_other_two_l684_684869

-- Definitions and conditions from the problem
def avg (l : List ℕ) : ℕ := l.sum / l.length

variables {A B C D E : ℕ}
variables (h_avg_five : avg [A, B, C, D, E] = 20)
variables (h_sum_three : A + B + C = 48)
variables (h_twice : A = 2 * B)

-- Theorem to prove
theorem avg_of_other_two (A B C D E : ℕ) 
  (h_avg_five : avg [A, B, C, D, E] = 20)
  (h_sum_three : A + B + C = 48)
  (h_twice : A = 2 * B) :
  avg [D, E] = 26 := 
  sorry

end avg_of_other_two_l684_684869


namespace min_payment_to_verify_weights_l684_684761

theorem min_payment_to_verify_weights :
  ∃ (cost : ℕ), cost = 800 ∧
    let diamonds := (list.range 15).map (λ n, n + 1)
    ∧ let weights := [1, 2, 4, 8]
    ∧ ∀ d ∈ diamonds, ∃ ws ⊆ weights, sum ws = d ∧ 100 * ws.length ≤ cost :=
sorry

end min_payment_to_verify_weights_l684_684761


namespace calculate_expression_l684_684259

theorem calculate_expression :
  2 * (7:ℝ)^(1 / 3) + (16:ℝ)^(3 / 4) + (((4:ℝ) / (Real.sqrt 3 - 1))^0) + (-3:ℝ)^(-1) = 44 / 3 :=
by
  sorry

end calculate_expression_l684_684259


namespace items_from_B_l684_684695

noncomputable def totalItems : ℕ := 1200
noncomputable def ratioA : ℕ := 3
noncomputable def ratioB : ℕ := 4
noncomputable def ratioC : ℕ := 5
noncomputable def totalRatio : ℕ := ratioA + ratioB + ratioC
noncomputable def sampledItems : ℕ := 60
noncomputable def numberB := sampledItems * ratioB / totalRatio

theorem items_from_B :
  numberB = 20 :=
by
  sorry

end items_from_B_l684_684695


namespace pet_store_solution_l684_684962

def pet_store_problem 
    (total_puppies : ℕ) 
    (puppies_per_cage : ℕ) 
    (cages_used : ℕ) 
    : Prop :=
    total_puppies = 18 ∧ puppies_per_cage = 5 ∧ cages_used = 3 → 
    let puppies_in_cages := cages_used * puppies_per_cage in
    let puppies_sold := total_puppies - puppies_in_cages in
    puppies_sold = 3

theorem pet_store_solution : pet_store_problem 18 5 3 :=
by
  sorry

end pet_store_solution_l684_684962


namespace range_of_a_function_greater_than_exp_neg_x_l684_684349

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a / x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ f x a = 0) → (0 < a ∧ a ≤ 1 / Real.exp 1) :=
sorry

theorem function_greater_than_exp_neg_x (a : ℝ) (h : a ≥ 2 / Real.exp 1) (x : ℝ) (hx : 0 < x) : f x a > Real.exp (-x) :=
sorry

end range_of_a_function_greater_than_exp_neg_x_l684_684349


namespace Jane_probability_wins_l684_684424

noncomputable def probability_jane_wins : ℚ :=
  let sectors := {1, 2, 3, 4, 5, 6}
  let total_outcomes := 36
  let losing_outcomes := 6
  let winning_outcomes := total_outcomes - losing_outcomes
  rationalize winning_outcomes total_outcomes := winning_outcomes / total_outcomes

theorem Jane_probability_wins : probability_jane_wins = 5 / 6 := sorry

end Jane_probability_wins_l684_684424


namespace units_digit_of_product_1_to_50_is_zero_l684_684156

theorem units_digit_of_product_1_to_50_is_zero :
  Nat.digits 10 (∏ i in Finset.range 51, i) = [0] :=
sorry

end units_digit_of_product_1_to_50_is_zero_l684_684156


namespace find_integers_with_sum_and_gcd_l684_684713

theorem find_integers_with_sum_and_gcd {a b : ℕ} (h_sum : a + b = 104055) (h_gcd : Nat.gcd a b = 6937) :
  (a = 6937 ∧ b = 79118) ∨ (a = 13874 ∧ b = 90181) ∨ (a = 27748 ∧ b = 76307) ∨ (a = 48559 ∧ b = 55496) :=
sorry

end find_integers_with_sum_and_gcd_l684_684713


namespace polynomial_divisibility_l684_684264

-- Define the polynomial f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^3 - 8 * x^2 + m * x - 16

-- Prove that f(x) is divisible by x-2 if and only if m=8
theorem polynomial_divisibility (m : ℝ) :
  (∀ (x : ℝ), (x - 2) ∣ f x m) ↔ m = 8 := 
by
  sorry

end polynomial_divisibility_l684_684264


namespace no_third_quadrant_l684_684711

def quadratic_no_real_roots (b : ℝ) : Prop :=
  16 - 4 * b < 0

def passes_through_third_quadrant (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = -2 * x + b ∧ x < 0 ∧ y < 0

theorem no_third_quadrant (b : ℝ) (h : quadratic_no_real_roots b) : ¬ passes_through_third_quadrant b := 
by {
  sorry
}

end no_third_quadrant_l684_684711


namespace taxi_ride_cost_l684_684973

theorem taxi_ride_cost (base_fare : ℚ) (cost_per_mile : ℚ) (distance : ℕ) :
  base_fare = 2 ∧ cost_per_mile = 0.30 ∧ distance = 10 →
  base_fare + cost_per_mile * distance = 5 :=
by
  sorry

end taxi_ride_cost_l684_684973


namespace sqrt_A_approx_20_sig_figs_l684_684783

theorem sqrt_A_approx_20_sig_figs :
  let A := 0.1111111111
  in Real.sqrt A = 0.33333333331666666666 :=
by
  -- provide the necessary steps and calculations
  let A := 0.1111111111
  -- Typically we would show the detailed steps here, but 
  -- since the problem statement says to include a proof, we provide:
  sorry

end sqrt_A_approx_20_sig_figs_l684_684783


namespace range_of_cars_l684_684521

def fuel_vehicle_cost_per_km (x : ℕ) : ℚ := (40 * 9) / x
def new_energy_vehicle_cost_per_km (x : ℕ) : ℚ := (60 * 0.6) / x

theorem range_of_cars : ∃ x : ℕ, fuel_vehicle_cost_per_km x = new_energy_vehicle_cost_per_km x + 0.54 ∧ x = 600 := 
by {
  sorry
}

end range_of_cars_l684_684521


namespace tetrahedrons_with_equal_face_areas_have_unequal_volumes_l684_684419

theorem tetrahedrons_with_equal_face_areas_have_unequal_volumes :
  ∃ (T1 T2 : EuclideanGeometry.tetrahedron), 
  (∀ f1 ∈ T1.faces, ∃ f2 ∈ T2.faces, f1.area = f2.area) ∧ T1.volume ≠ T2.volume := by
sorry

end tetrahedrons_with_equal_face_areas_have_unequal_volumes_l684_684419


namespace sides_of_triangle_with_120_degree_angle_l684_684247

theorem sides_of_triangle_with_120_degree_angle:
  ∃ (x y z : ℕ), x = 3 ∧ y = 6 ∧ z = 7 ∧ (x^2 + x * y + y^2 = z^2) :=
by
  existsi 3
  existsi 6
  existsi 7
  split
  . rfl
  split
  . rfl
  split
  . rfl
  have h : 3^2 + 3 * 6 + 6^2 = 7^2 := by norm_num
  exact h
  done

end sides_of_triangle_with_120_degree_angle_l684_684247


namespace smallest_positive_period_pi_increasing_intervals_in_0_to_pi_range_of_m_l684_684026

noncomputable def f (x m : ℝ) := 2 * (Real.cos x) ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + m

theorem smallest_positive_period_pi (m : ℝ) :
  ∀ x : ℝ, f (x + π) m = f x m := sorry

theorem increasing_intervals_in_0_to_pi (m : ℝ) :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 6) ∨ (2 * π / 3 ≤ x ∧ x ≤ π) →
  ∀ y : ℝ, ((0 ≤ y ∧ y ≤ π / 6 ∨ (2 * π / 3 ≤ y ∧ y ≤ π)) ∧ x < y) → f x m < f y m := sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 6) → -4 < f x m ∧ f x m < 4) ↔ (-6 < m ∧ m < 1) := sorry

end smallest_positive_period_pi_increasing_intervals_in_0_to_pi_range_of_m_l684_684026


namespace raking_time_together_l684_684421

theorem raking_time_together (your_rate : ℝ) (brother_rate : ℝ) 
  (hyour_rate : your_rate = 1/30) (hbrother_rate : brother_rate = 1/45) :
  let combined_rate := your_rate + brother_rate in
  let time_to_rake := 1 / combined_rate in
  time_to_rake = 18 :=
by
  sorry

end raking_time_together_l684_684421


namespace runs_percentage_proof_l684_684935

def runs_total : ℕ := 136
def runs_boundaries : ℕ := 12 * 4
def runs_sixes : ℕ := 2 * 6

def runs_by_running : ℕ := runs_total - (runs_boundaries + runs_sixes)

def percentage_runs_by_running : ℚ := (runs_by_running.toRat / runs_total.toRat) * 100

theorem runs_percentage_proof : percentage_runs_by_running ≈ 55.88 := sorry

end runs_percentage_proof_l684_684935


namespace chromosomes_mitosis_late_stage_l684_684945

/-- A biological cell with 24 chromosomes at the late stage of the second meiotic division. -/
def cell_chromosomes_meiosis_late_stage : ℕ := 24

/-- The number of chromosomes in this organism at the late stage of mitosis is double that at the late stage of the second meiotic division. -/
theorem chromosomes_mitosis_late_stage : cell_chromosomes_meiosis_late_stage * 2 = 48 :=
by
  -- We will add the necessary proof here.
  sorry

end chromosomes_mitosis_late_stage_l684_684945


namespace tangents_at_origin_line_AB_at_t_l684_684324

universe u

variable {R : Type u} [LinearOrderedField R]

def circle (x y : R) : Prop := (x - 2) ^ 2 + y ^ 2 = 1

def is_tangent_to_circle (tangent : R → R → Prop) : Prop :=
  ∃ x y, circle x y ∧ tangent x y


theorem tangents_at_origin : 
  ∀ Q : ℝ × ℝ, (Q = (0,0)) → 
  (is_tangent_to_circle (λ x y, y = (sqrt 3 / 3) * x)) ∧ 
  (is_tangent_to_circle (λ x y, y = - (sqrt 3 / 3) * x)) :=
sorry

theorem line_AB_at_t : 
  ∀ t : ℝ, 
  t ∈ ℝ → 
  (∃ A B : ℝ × ℝ, A ≠ B ∧ circle A.1 A.2 ∧ circle B.1 B.2 ∧ 
  Q = (t, t) ∧ 
  segment_tangent A B Q → 
  ∀ x y : ℝ, ((t - 2) * x + t * y = 2 * t - 3)) :=
sorry


end tangents_at_origin_line_AB_at_t_l684_684324


namespace necessary_but_not_sufficient_l684_684875

open Set Real

theorem necessary_but_not_sufficient (x : ℝ) : (0 < x ∧ x < 3) → (ln (x - 2) < 0) → (∃ x, 0 < x ∧ x < 3 ∧ ln (x - 2) < 0) :=
by
  intro h1 h2
  have hA : Set.Ioo (0 : ℝ) 3 = { x | 0 < x ∧ x < 3 } := rfl
  have hB : Set.Ioo (2 : ℝ) 3 = { x | 2 < x ∧ x < 3 } := rfl
  have subset_B_A : Set.Ioo (2 : ℝ) 3 ⊆ Set.Ioo (0 : ℝ) 3 := 
    by 
      intros y hy
      rw [hA, hB] at *
      exact ⟨(lt_of_lt_of_le (lt_of_lt hy.1 zero_le_two) le_rfl), hy.2⟩
  exact ⟨x, h1.1, h1.2, h2⟩

end necessary_but_not_sufficient_l684_684875


namespace george_speed_to_school_l684_684317

theorem george_speed_to_school :
  ∀ (D S_1 S_2 D_1 S_x : ℝ),
  D = 1.5 ∧ S_1 = 3 ∧ S_2 = 2 ∧ D_1 = 0.75 →
  S_x = (D - D_1) / ((D / S_1) - (D_1 / S_2)) →
  S_x = 6 :=
by
  intros D S_1 S_2 D_1 S_x h1 h2
  rw [h1.1, h1.2.1, h1.2.2.1, h1.2.2.2] at *
  sorry

end george_speed_to_school_l684_684317


namespace min_value_expression_l684_684544

open Real

theorem min_value_expression : ∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 :=
by
  intro x y
  sorry

end min_value_expression_l684_684544


namespace arithmetic_sequence_sum_l684_684330

noncomputable theory
open_locale classical

variable {a : ℕ → ℝ}

-- Given conditions for the problem
def cond1 : Prop := a 1 + a 3 = 2
def cond2 : Prop := a 3 + a 5 = 4

-- Define the arithmetic sequence
def arithmetic_sequence (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- The main problem statement
theorem arithmetic_sequence_sum (d : ℝ) (h1 : arithmetic_sequence d) (h2 : cond1) (h3 : cond2) :
  a 7 + a 9 = 8 :=
sorry

end arithmetic_sequence_sum_l684_684330


namespace initial_green_hard_hats_l684_684401

noncomputable def initial_pink_hard_hats : ℕ := 26
noncomputable def initial_yellow_hard_hats : ℕ := 24
noncomputable def carl_taken_pink_hard_hats : ℕ := 4
noncomputable def john_taken_pink_hard_hats : ℕ := 6
noncomputable def john_taken_green_hard_hats (G : ℕ) : ℕ := 2 * john_taken_pink_hard_hats
noncomputable def remaining_pink_hard_hats : ℕ := initial_pink_hard_hats - carl_taken_pink_hard_hats - john_taken_pink_hard_hats
noncomputable def total_remaining_hard_hats (G : ℕ) : ℕ := remaining_pink_hard_hats + (G - john_taken_green_hard_hats G) + initial_yellow_hard_hats

theorem initial_green_hard_hats (G : ℕ) :
  total_remaining_hard_hats G = 43 ↔ G = 15 := by
  sorry

end initial_green_hard_hats_l684_684401


namespace series_sum_12100_value_l684_684642

def sign_change (n : ℕ) : ℕ :=
  (-1) ^ (⟨n, sorry⟩ / 2) -- TODO: actually define the floor(sqrt(n)/2)

noncomputable def series_sum (n : ℕ) : ℤ :=
  ∑ k in finset.range (n + 1), (sign_change k) * k

theorem series_sum_12100_value :
  series_sum 12100 = S :=
sorry

end series_sum_12100_value_l684_684642


namespace gravel_cost_l684_684367

def cost_per_cubic_foot := 8
def cubic_yards := 3
def cubic_feet_per_cubic_yard := 27

theorem gravel_cost :
  (cubic_yards * cubic_feet_per_cubic_yard) * cost_per_cubic_foot = 648 :=
by sorry

end gravel_cost_l684_684367


namespace range_of_expression_l684_684754

-- Definitions for the geometric context
variables {A B C : ℝ} -- Angles of triangle ABC
variables {a b c : ℝ} -- Sides opposite to angles A, B, C respectively

-- Conditions of the problem
axiom triangle_ABC (ha : a > 0) (hb : b > 0) (hc : c > 0) 
                   (habc : a ≠ b + c) (hbca : b ≠ a + c) (hcab : c ≠ a + b)

axiom side_relation (h : b ^ 2 = a * c)

-- Function representing the simplified expression to analyze
noncomputable def expression := (sin A + cos A * tan C) / (sin B + cos B * tan C)

-- The main statement to prove
theorem range_of_expression [triangle_ABC ha hb hc habc hbca hcab] [side_relation h] :
  ∃ lower upper, lower = (sqrt 5 - 1) / 2 ∧ upper = (sqrt 5 + 1) / 2 ∧ lower < expression ∧ expression < upper :=
sorry

end range_of_expression_l684_684754


namespace complex_fraction_simplification_l684_684997

theorem complex_fraction_simplification : 
  ∀ (i : ℂ), i^2 = -1 → (1 + i^2017) / (1 - i) = i :=
by
  intro i h_imag_unit
  sorry

end complex_fraction_simplification_l684_684997


namespace range_j_l684_684802

def h (x : ℝ) : ℝ := 4 * x - 3
def j (x : ℝ) : ℝ := h (h (h x))

theorem range_j : ∀ x, 0 ≤ x ∧ x ≤ 3 → -63 ≤ j x ∧ j x ≤ 129 :=
by
  intro x
  intro hx
  sorry

end range_j_l684_684802


namespace total_arrangements_l684_684406

theorem total_arrangements (students : Fin 6) (A B : Fin 6) :
    A ∈ students ∧ B ∈ students ∧ (∀ i, A ≠ B) ∧
    (A ∈ {0, 1}) ∧ (B ∈ {1, 3}) ->
    ∃ l : List (Fin 6), l.length = 4 ∧ nodup l ∧
    36 = (l.filter (fun x => x = A).length + l.filter (fun x => x = B).length) :=
begin
  sorry
end

end total_arrangements_l684_684406


namespace value_of_c_l684_684351

noncomputable def f (a c x : ℝ) : ℝ := a * x^3 + c
noncomputable def f' (a x : ℝ) : ℝ := 3 * a * x^2

theorem value_of_c (a c : ℝ) (h1: f' a 1 = 6) (h2: ∀ x ∈ Icc 1 2, f a c x ≤ 20) : c = 4 :=
by {
  have ha : a = 2, from (by linarith : 3 * a * 1^2 = 6), -- resolves to a = 2
  sorry,
}

end value_of_c_l684_684351


namespace taxi_ride_cost_l684_684972

theorem taxi_ride_cost (base_fare : ℚ) (cost_per_mile : ℚ) (distance : ℕ) :
  base_fare = 2 ∧ cost_per_mile = 0.30 ∧ distance = 10 →
  base_fare + cost_per_mile * distance = 5 :=
by
  sorry

end taxi_ride_cost_l684_684972


namespace sofia_total_time_l684_684842

constant laps : ℕ
constant track_length : ℕ
constant first_part_length : ℕ
constant second_part_length : ℕ
constant first_part_speed : ℝ
constant second_part_speed : ℝ
constant time_for_first_part : ℝ
constant time_for_second_part : ℝ
constant total_time_one_lap : ℝ
constant total_time_6_laps : ℝ

axiom h1 : laps = 6
axiom h2 : track_length = 500
axiom h3 : first_part_length = 200
axiom h4 : second_part_length = 300
axiom h5 : first_part_speed = 5
axiom h6 : second_part_speed = 4
axiom h7 : time_for_first_part = first_part_length / first_part_speed
axiom h8 : time_for_second_part = second_part_length / second_part_speed
axiom h9 : total_time_one_lap = time_for_first_part + time_for_second_part
axiom h10 : total_time_6_laps = laps * total_time_one_lap

theorem sofia_total_time : total_time_6_laps = 11 * 60 + 30 := by
  sorry

end sofia_total_time_l684_684842


namespace sum_of_two_numbers_l684_684094

theorem sum_of_two_numbers (a b : ℝ) (h1 : a * b = 16) (h2 : (1 / a) = 3 * (1 / b)) (ha : 0 < a) (hb : 0 < b) :
  a + b = 16 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end sum_of_two_numbers_l684_684094


namespace log_simplification_l684_684838

theorem log_simplification :
  (1 / (Real.log 3 / Real.log 12 + 2))
  + (1 / (Real.log 2 / Real.log 8 + 2))
  + (1 / (Real.log 3 / Real.log 9 + 2)) = 2 :=
  sorry

end log_simplification_l684_684838


namespace m_value_distribution_function_expected_value_variance_l684_684515

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 ∨ x > 8 then 0
  else if 2 < x ∧ x ≤ 4 then (1 / 6) * (x - 2)
  else if 4 < x ∧ x ≤ 8 then -(1 / 12) * (x - 8)
  else 0

def F (x : ℝ) : ℝ :=
  if x ≤ 2 then 0
  else if 2 < x ∧ x ≤ 4 then (x - 2)^2 / 12
  else if 4 < x ∧ x ≤ 8 then 1 - (x - 8)^2 / 24
  else 1

theorem m_value :
  let m := 1 / 3
  (∫ x in 2..8, f x) = 1 →
  m = 1 / 3 := by sorry

theorem distribution_function :
  ∀ x : ℝ, F x = 
    if x ≤ 2 then 0
    else if 2 < x ∧ x ≤ 4 then (x - 2)^2 / 12
    else if 4 < x ∧ x ≤ 8 then 1 - (x - 8)^2 / 24
    else 1 := by sorry

theorem expected_value :
  (∫ x in 2..8, x * f x) = 14 / 3 := by sorry

theorem variance :
  (∫ x in 2..8, x^2 * f x) - (14 / 3)^2 = 14 / 9 := by sorry

end m_value_distribution_function_expected_value_variance_l684_684515


namespace AE_plus_AP_eq_PD_l684_684775

-- Definitions based on the given conditions in the problem
variables (ABC : Triangle) (O : Incircle ABC)
variables (A B C : Point) (AC : Angle) (DE : Segment) (E F D : Point)

-- Condition: ABC is a right triangle with angle ∠ACB = 90°
variable (h1 : AC = 90)

-- Incircle O touches the sides BC, CA, and AB at points D, E, and F respectively.
variable (h2 : O.touches BC at D)
variable (h3 : O.touches CA at E)
variable (h4 : O.touches AB at F)

-- AD intersects the incircle O at P
variables (P : Point) (AD : Line) (h5 : AD.intersects_at P on O)

-- BP and CP are drawn such that ∠BPC = 90°
variable (BP : Line) (CP : Line)
variable (h6 : ∠(BP, CP) = 90°)

-- Statement to prove
theorem AE_plus_AP_eq_PD :
  AE + AP = PD :=
sorry

end AE_plus_AP_eq_PD_l684_684775


namespace solution_set_equality_l684_684016

noncomputable def f : ℝ → ℝ := sorry -- Let's assume f and its derivative are defined.

axiom f_deriv : ∀ x : ℝ, has_deriv_at f (f' x) x
axiom f_cond : ∀ x : ℝ, f(x) + deriv f x > 1
axiom f_at_zero : f(0) = 2016

theorem solution_set_equality : {x : ℝ | e^x * f(x) > e^x + 2015} = {x : ℝ | x > 0} :=
by
  sorry

end solution_set_equality_l684_684016


namespace set_contains_all_rationals_l684_684428

variable (S : Set ℚ)
variable (h1 : (0 : ℚ) ∈ S)
variable (h2 : ∀ x ∈ S, x + 1 ∈ S ∧ x - 1 ∈ S)
variable (h3 : ∀ x ∈ S, x ≠ 0 → x ≠ 1 → 1 / (x * (x - 1)) ∈ S)

theorem set_contains_all_rationals : ∀ q : ℚ, q ∈ S :=
by
  sorry

end set_contains_all_rationals_l684_684428


namespace value_of_y_for_absolute_value_eq_zero_l684_684919

theorem value_of_y_for_absolute_value_eq_zero :
  ∃ (y : ℚ), |(2:ℚ) * y - 3| ≤ 0 ↔ y = 3 / 2 :=
by
  sorry

end value_of_y_for_absolute_value_eq_zero_l684_684919


namespace slope_of_line_l684_684641

theorem slope_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : (- (4 : ℝ) / 7) = -4 / 7 :=
by
  -- Sorry for the proof for completeness
  sorry

end slope_of_line_l684_684641


namespace range_function1_range_function2_l684_684196

theorem range_function1 : ∀ y : ℝ, y ∈ set.Ici (15 / 8) ↔ ∃ x : ℝ, x ≥ 1 ∧ y = 2 * x - real.sqrt (x - 1) :=
by sorry

theorem range_function2 : ∀ y : ℝ, y ≠ 3 ↔ ∃ x : ℝ, x ≠ -1 ∧ y = (3 * x - 1) / (x + 1) :=
by sorry

end range_function1_range_function2_l684_684196


namespace difference_of_smallest_integers_l684_684295

theorem difference_of_smallest_integers (n_1 n_2: ℕ) (h1 : ∀ k, 2 ≤ k ∧ k ≤ 6 → (n_1 > 1 ∧ n_1 % k = 1)) (h2 : ∀ k, 2 ≤ k ∧ k ≤ 6 → (n_2 > 1 ∧ n_2 % k = 1)) (h_smallest : n_1 = 61) (h_second_smallest : n_2 = 121) : n_2 - n_1 = 60 :=
by
  sorry

end difference_of_smallest_integers_l684_684295


namespace units_digit_of_50_factorial_l684_684143

theorem units_digit_of_50_factorial : (nat.factorial 50) % 10 = 0 := 
by 
  sorry

end units_digit_of_50_factorial_l684_684143


namespace matrix_B4_v_eq_v4_l684_684011

noncomputable def matrix_B : Matrix (Fin 2) (Fin 2) ℝ := sorry
def v : Fin 2 → ℝ := fun i => if i = 0 then 3 else -1
def v4 : Fin 2 → ℝ := fun i => if i = 0 then 243 else -81

theorem matrix_B4_v_eq_v4
  (h : matrix_B.mul_vec v = ![9, -3]) :
  (matrix_B ^ 4).mul_vec v = v4 :=
sorry

end matrix_B4_v_eq_v4_l684_684011


namespace sum_of_coefficients_binomial_l684_684336

theorem sum_of_coefficients_binomial (m : ℝ) (h : ∫ x in 1..m, (2 * x - 1) = 6) : 
  (∑ k in finset.range (9 + 1), (-1)^k * 2^k * binomial 9 k = -1) :=
sorry

end sum_of_coefficients_binomial_l684_684336


namespace problem_solution_l684_684852

variables (a b c d : ℝ)

def cond1 : Prop := a + b - c - d = 3
def cond2 : Prop := ab - 3 * bc + cd - 3 * da = 4
def cond3 : Prop := 3 * ab - bc + 3 * cd - da = 5

theorem problem_solution 
  (h1 : cond1 a b c d)
  (h2 : cond2 a b c d)
  (h3 : cond3 a b c d) : 
  11 * (a - c)^2 + 17 * (b - d)^2 = 63 :=
by
  sorry

end problem_solution_l684_684852


namespace average_marks_l684_684971

variable (M P C B : ℕ)

theorem average_marks (h1 : M + P = 20) (h2 : C = P + 20) 
  (h3 : B = 2 * M) (h4 : M ≤ 100) (h5 : P ≤ 100) (h6 : C ≤ 100) (h7 : B ≤ 100) :
  (M + C) / 2 = 20 := by
  sorry

end average_marks_l684_684971


namespace f_recurrence_l684_684502

def f : ℕ → ℝ := -- This will be defined in a proof later
  sorry

theorem f_recurrence :
  ∀ n : ℕ, (f 0 = 2) ∧ (∀ n : ℕ, (f (n + 1) - 1)^2 + (f n - 1)^2 = 2 * (f n) * (f (n + 1)) + 4) → 
  ∀ n : ℕ, f n = n^2 + n * real.sqrt 11 + 2 :=
sorry

end f_recurrence_l684_684502


namespace average_people_per_hour_l684_684415

theorem average_people_per_hour 
    (total_people : ℕ) 
    (days : ℕ) 
    (hours_per_day : ℕ) 
    (h1 : total_people = 3000) 
    (h2 : days = 5) 
    (h3 : hours_per_day = 24) :
    total_people / (days * hours_per_day) = 25 :=
begin
    -- The proof would go here, but it is omitted as per the instructions.
    sorry
end

end average_people_per_hour_l684_684415


namespace arithmetic_to_geometric_progression_l684_684045

theorem arithmetic_to_geometric_progression (x y z : ℝ) 
  (hAP : 2 * y^2 - y * x = z^2) : 
  z^2 = y * (2 * y - x) := 
  by 
  sorry

end arithmetic_to_geometric_progression_l684_684045


namespace largest_b_value_l684_684902

/- Definitions -/
def conditions (a b c : ℕ) : Prop :=
  1 < c ∧ c < b ∧ b < a ∧ a * b * c = 360

noncomputable def largest_b : ℕ :=
  Nat.find_max' (λ b, ∃ a c, conditions a b c) sorry

/- Theorem -/
theorem largest_b_value : largest_b = 12 :=
sorry

end largest_b_value_l684_684902


namespace pyramid_properties_l684_684070

theorem pyramid_properties (d : ℝ) (d_pos : d > 0) :
  ∃ (F V : ℝ), 
    F = (d^2) / 2 * (1 + Real.sqrt 7) ∧
    V = (d^3) * (Real.sqrt 3) / 12 :=
begin
  sorry
end

end pyramid_properties_l684_684070


namespace correct_proposition_l684_684800

-- Definitions based on conditions
variables (Line : Type) (Plane : Type)
variables (a b c : Line)
variables (α β γ : Plane)

-- Conditions for problem
axiom distinct_lines : a ≠ b ∧ b ≠ c ∧ a ≠ c
axiom distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Propositions
axiom parallel (x y : Type) : Prop

axiom prop1 : parallel Line Plane → parallel Line Plane → parallel Line Line
axiom prop2 : parallel Plane Line → parallel Plane Line → parallel Plane Plane
axiom prop3 : parallel Plane Plane → parallel Plane Plane → parallel Plane Plane
axiom prop4 : parallel Plane Line → parallel Line Line → parallel Plane Line

-- To prove
theorem correct_proposition : prop3 _ _ _ :=
by sorry

end correct_proposition_l684_684800


namespace problem_statement_l684_684023

noncomputable def p := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def q := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def r := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7
noncomputable def s := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7

theorem problem_statement :
  (1 / p + 1 / q + 1 / r + 1 / s)^2 = 112 / 3481 :=
sorry

end problem_statement_l684_684023


namespace sin_sum_leq_three_halves_l684_684753

-- Define the proof problem
theorem sin_sum_leq_three_halves 
  (A B C : ℝ) 
  (h1 : A + B + C = 180) 
  (h2 : 0 < C) 
  (h3 : C ≤ 60) : 
  sin (A - 30) + sin (B - 30) + sin (C - 30) ≤ 3 / 2 :=
sorry

end sin_sum_leq_three_halves_l684_684753


namespace jose_investment_l684_684532

theorem jose_investment 
  (tom_investment : ℝ)
  (jose_investment_time : ℝ → ℝ)
  (total_profit : ℝ)
  (jose_profit : ℝ)
  (tom_profit : ℝ)
  (investment_ratio : ℝ → ℝ → Prop)
  (profit_ratio : ℝ → ℝ → Prop)
  :
  ∃ (X : ℝ), jose_investment X = 45000 :=
by
  let tom_investment := 30000
  let jose_investment_time := fun X => X * 10
  let total_profit := 63000
  let jose_profit := 35000
  let tom_profit := total_profit - jose_profit
  let investment_ratio := fun t j => t / j
  let profit_ratio := fun t j => t / j
  have Hr : investment_ratio (tom_investment * 12) (jose_investment_time 45000) = profit_ratio tom_profit jose_profit, sorry
  use 45000
  sorry

end jose_investment_l684_684532


namespace distribution_methods_l684_684987

def people := {A, B, C, D, E}
def activities := {Activity1, Activity2, Activity3}

def participate (p : {A, B, C, D, E}) (a : {Activity1, Activity2, Activity3}) : Prop

def valid_distribution (dist : people → activities) : Prop :=
  (dist A ≠ dist B) ∧
  (∃ a, dist a = Activity1 ∧ a ∈ people ∧ ∃ b, dist b = Activity1 ∧ b ≠ a) ∧
  (∃ c, dist c = Activity2 ∧ c ∈ people ∧ ∃ d, dist d = Activity2 ∧ d ≠ c) ∧
  (∃ e, dist e = Activity3 ∧ e ∈ people)

theorem distribution_methods : ∃ dist_methods, valid_distribution dist_methods ∧ dist_methods = 24 := sorry

end distribution_methods_l684_684987


namespace longest_segment_in_cylinder_correct_l684_684588

noncomputable def longest_segment_in_cylinder (r h : ℝ) : ℝ :=
  real.sqrt (h^2 + (2*r)^2)

theorem longest_segment_in_cylinder_correct : longest_segment_in_cylinder 5 12 = 2 * real.sqrt 61 :=
by
  -- Conditions
  let r := 5
  let h := 12
  
  -- Assertion that needs proof
  have : longest_segment_in_cylinder r h = 2 * real.sqrt 61
  exact sorry

  exact this

end longest_segment_in_cylinder_correct_l684_684588


namespace triangle_constructibility_l684_684266

variables (a b c γ : ℝ)

-- definition of the problem conditions
def valid_triangle_constructibility_conditions (a b_c_diff γ : ℝ) : Prop :=
  γ < 90 ∧ b_c_diff < a * Real.cos γ

-- constructibility condition
def is_constructible (a b c γ : ℝ) : Prop :=
  b - c < a * Real.cos γ

-- final theorem statement
theorem triangle_constructibility (a b c γ : ℝ) (h1 : γ < 90) (h2 : b > c) :
  (b - c < a * Real.cos γ) ↔ valid_triangle_constructibility_conditions a (b - c) γ :=
by sorry

end triangle_constructibility_l684_684266


namespace minimum_at_2_l684_684505

open Real

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem minimum_at_2 : 
  (∀ x : ℝ, ((x ≠ 2) → (f(2) ≤ f(x)))) :=
by
  sorry

end minimum_at_2_l684_684505


namespace units_digit_of_50_factorial_l684_684172

theorem units_digit_of_50_factorial : 
  ∃ d, (d = List.prod (List.range 1 51)) ∧ (d % 10 = 0) :=
by
  sorry

end units_digit_of_50_factorial_l684_684172


namespace units_digit_50_factorial_l684_684161

theorem units_digit_50_factorial : (nat.factorial 50) % 10 = 0 :=
by
  sorry

end units_digit_50_factorial_l684_684161


namespace insulation_optimization_l684_684531

def construction_cost_per_cm := 60000
def annual_energy_no_insulation := 8 -- in ten thousand yuan

def energy_consumption_cost (x : ℝ) (k : ℝ) := k / (3 * x + 5)

def f (x : ℝ) (k : ℝ) : ℝ :=
  (construction_cost_per_cm / 10000) * x + 20 * (energy_consumption_cost x k)

theorem insulation_optimization :
  ∃ k : ℝ, 
  k = 40 ∧ 
  (f 0 40 = 8) ∧ 
  ( ∀ x : ℝ, 0 ≤ x ∧ x ≤ 10 → f x 40 = 6 * x + 800 / (3 * x + 5) ) ∧ 
  ( ∃ x : ℝ, 0 ≤ x ∧ x ≤ 10 ∧ f x 40 = 70 ∧ (∀ y : ℝ, 0 ≤ y ∧ y ≤ 10 → f y 40 ≥ 70) )
  :=
sorry

end insulation_optimization_l684_684531


namespace simplify_expression_l684_684058

theorem simplify_expression (x y : ℝ) : 
  8 * x + 3 * y - 2 * x + y + 20 + 15 = 6 * x + 4 * y + 35 :=
by
  sorry

end simplify_expression_l684_684058


namespace eccentricity_of_ellipse_l684_684981

-- Definitions from conditions
def ellipse (a b : ℝ) : Set (ℝ × ℝ) := 
  {p | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1}

def focus1 (a b : ℝ) : ℝ × ℝ :=
  (-real.sqrt (a ^ 2 - b ^ 2), 0)

def focus2 (a b : ℝ) : ℝ × ℝ := 
  (real.sqrt (a ^ 2 - b ^ 2), 0)

def point_P (a b : ℝ) : Set (ℝ × ℝ) := 
  {p | p ∈ ellipse a b ∧ p.1 = -real.sqrt (a ^ 2 - b ^ 2)}

def angle_F1PF2_is_60 (a b : ℝ) : Prop :=
  let c := real.sqrt (a ^ 2 - b ^ 2)
  ∃ P ∈ point_P a b, ∠focus1 a b (P.fst, P.snd) (focus2 a b) = real.pi / 3

-- Translate math proof problem
theorem eccentricity_of_ellipse (a b : ℝ) (ha : a > b) (hb : b > 0)
  (h60 : angle_F1PF2_is_60 a b) : 
  let e := real.sqrt (a^2 - b^2) / a
  e = real.sqrt 3 / 3 := 
sorry

end eccentricity_of_ellipse_l684_684981


namespace domain_of_ln_sqrt_function_l684_684077

/-- Define the domain conditions -/
def condition_1 (x : ℝ) : Prop := sin x > 0
def condition_2 (x : ℝ) : Prop := cos x - sqrt 2 / 2 ≥ 0

/-- Define the main problem statement -/
theorem domain_of_ln_sqrt_function :
  (∀ x : ℝ, condition_1 x ∧ condition_2 x ↔ ∃ k : ℤ, 2 * k * π < x ∧ x <= 2 * k * π + π / 4) :=
sorry

end domain_of_ln_sqrt_function_l684_684077


namespace spoiled_milk_percentage_l684_684423

theorem spoiled_milk_percentage (p_egg p_flour p_all_good : ℝ) (h_egg : p_egg = 0.40) (h_flour : p_flour = 0.75) (h_all_good : p_all_good = 0.24) : 
  (1 - (p_all_good / (p_egg * p_flour))) = 0.20 :=
by
  sorry

end spoiled_milk_percentage_l684_684423


namespace complex_product_polar_form_l684_684996

noncomputable def complex_cis (r : ℝ) (θ : ℝ) : ℂ := r * complex.exp (complex.I * θ * π / 180)

theorem complex_product_polar_form :
  let z1 := complex_cis 4 160
  let z2 := complex_cis 5 210
  z1 * z2 = complex_cis 20 10 :=
by
  sorry

end complex_product_polar_form_l684_684996


namespace probability_roots_condition_l684_684595

-- Define the conditions
def quadratic_eq (k : ℝ) (x : ℝ) : ℝ :=
  (k^2 - 2*k - 3) * x^2 + (3*k - 5) * x + 2

def define_interval (k : ℝ) : Prop :=
  3 ≤ k ∧ k ≤ 8

-- Proof statement of the probability
theorem probability_roots_condition :
  (∀ k : ℝ, define_interval k → 
    let a := k^2 - 2*k - 3,
        b := 3*k - 5,
        c := 2,
        x1_x2_sum := -b / a,
        x1_x2_prod := c / a in
    let x1 := (x1_x2_sum / 3),
        x2 := (x1_x2_prod ** (1/2)) in
    (x1 ≤ 2 * x2) ∧ 
    (k ∈ [3, 8])
  ) → 
  ∃ (P : ℝ), P = 4/15 :=
sorry

end probability_roots_condition_l684_684595


namespace arithmetic_seq_cos_l684_684017

def f (x : ℝ) : ℝ := 2 * x - Real.cos x

def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_cos (a1 : ℝ) (d : ℝ) 
  (h1 : d = π / 8) 
  (h2 : f (arithmetic_sequence a1 d 1) + f (arithmetic_sequence a1 d 2) + 
        f (arithmetic_sequence a1 d 3) + f (arithmetic_sequence a1 d 4) + 
        f (arithmetic_sequence a1 d 5) = 5 * π) :
  [f (arithmetic_sequence a1 d 3)]^2 - 
  (arithmetic_sequence a1 d 1) * (arithmetic_sequence a1 d 5) = 
  (13 / 16) * π^2 :=
by
  sorry

end arithmetic_seq_cos_l684_684017


namespace gg_of_3_is_107_l684_684729

-- Define the function g
def g (x : ℕ) : ℕ := 3 * x + 2

-- State that g(g(g(3))) equals 107
theorem gg_of_3_is_107 : g (g (g 3)) = 107 := by
  sorry

end gg_of_3_is_107_l684_684729


namespace obtain_1972_from_4_obtain_any_from_4_l684_684559

-- Conditions translated as helper functions
def append_digit_4 (n : ℕ) : ℕ := n * 10 + 4
def append_digit_0 (n : ℕ) : ℕ := n * 10
def halve_if_even (n : ℕ) : ℕ := if n % 2 = 0 then n / 2 else n

-- Prove that starting from 4, one can obtain 1972 using the allowed operations
theorem obtain_1972_from_4 : ∃ (ops : list (ℕ → ℕ)), (ops.foldl (λ n f => f n) 4) = 1972 := 
sorry

-- Prove that starting from 4, one can obtain any natural number using the allowed operations
theorem obtain_any_from_4 (n : ℕ) : ∃ (ops : list (ℕ → ℕ)), (ops.foldl (λ n f => f n) 4) = n :=
sorry

end obtain_1972_from_4_obtain_any_from_4_l684_684559


namespace abs_eq_neg_of_nonpos_l684_684246

theorem abs_eq_neg_of_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
by
  have ha : |a| ≥ 0 := abs_nonneg a
  rw [h] at ha
  exact neg_nonneg.mp ha

end abs_eq_neg_of_nonpos_l684_684246


namespace probability_event_occurs_l684_684833

def in_interval (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 * Real.pi

def event_occurs (x : ℝ) : Prop :=
  Real.cos (x + Real.pi / 3) + Real.sqrt 3 * Real.sin (x + Real.pi / 3) ≥ 1

theorem probability_event_occurs : 
  (∀ x, in_interval x → event_occurs x) → 
  (∃ p, p = 1/3) :=
by
  intros h
  sorry

end probability_event_occurs_l684_684833


namespace cells_after_9_days_l684_684750

noncomputable def remaining_cells (initial : ℕ) (days : ℕ) : ℕ :=
  let rec divide_and_decay (cells: ℕ) (remaining_days: ℕ) : ℕ :=
    if remaining_days = 0 then cells
    else
      let divided := cells * 2
      let decayed := (divided * 9) / 10
      divide_and_decay decayed (remaining_days - 3)
  divide_and_decay initial days

theorem cells_after_9_days :
  remaining_cells 5 9 = 28 := by
  sorry

end cells_after_9_days_l684_684750


namespace locus_of_c_l684_684718

-- Definitions of the problem
variables (e f : Line) (A B : Point) (P : Point)
variable (C : Point)
variable (d : ℝ) -- distance between the parallel lines

-- Conditions
axiom parallel_e_f : parallel e f
axiom A_on_e : on_line A e
axiom B_on_f : on_line B f
axiom midpoint_P : is_midpoint A B P
axiom AC_perpendicular : perpendicular AC (line_through A C)
axiom BC_perpendicular : perpendicular BC (line_through B C)
axiom AC_length : length_of_side C A = d
axiom BC_length : length_of_side C B = d

-- The theorem stating the proof problem
theorem locus_of_c : ∀ C, 
  (∃ points QR_1 QR_2 RQ_1 RQ_2 : Line, 
  (on_line C QR_1 ∨ on_line C QR_2 ∨ on_line C RQ_1 ∨ on_line C RQ_2) ∧ 
  ¬(C = Q) ∧ ¬(C = R)) :=
sorry

end locus_of_c_l684_684718


namespace volume_of_T_eq_32_div_3_l684_684102

section
variable {R : Type*} [LinearOrderedField R]

def T (x y z : R) : Prop :=
  abs x + abs y ≤ 2 ∧ abs x + abs z ≤ 2 ∧ abs y + abs z ≤ 2

theorem volume_of_T_eq_32_div_3 :
  let volume : R := 32/3
  in ∀ R : Type*, LinearOrderedField R → volume = 32 / 3 :=
by
  sorry
end

end volume_of_T_eq_32_div_3_l684_684102


namespace probability_inequality_l684_684192

noncomputable def random_variable (Ω : Type) [MeasureSpace Ω] : Type := Ω → ℝ

variables {Ω : Type} [MeasureSpace Ω]
variables (ξ : random_variable Ω)

axiom E_xi_squared_lt_infty : ∫⁻ (ω : Ω), (ξ ω)^2 ∂(volume) < ∞

theorem probability_inequality (h : E_xi_squared_lt_infty) :
  ∫⁻ (ω : Ω), (ξ ω) ∂(volume) = 0 ≤ (∫⁻ (ω : Ω), (ξ ω)^2 ∂(volume))^2 - ∫⁻ (ω : Ω), (ξ(ω))^2 ∂(volume) := sorry

end probability_inequality_l684_684192


namespace complementary_three_card_sets_count_l684_684287

open Set

-- Definitions for the conditions
def Deck : Set (String × String × String) :=
  {("star", c, s) | c ∈ {"red", "yellow", "blue", "green"} ∧ s ∈ {"light", "medium", "dark"}} ∪
  {("heart", c, s) | c ∈ {"red", "yellow", "blue", "green"} ∧ s ∈ {"light", "medium", "dark"}} ∪
  {("diamond", c, s) | c ∈ {"red", "yellow", "blue", "green"} ∧ s ∈ {"light", "medium", "dark"}}

def is_complementary (cards : Finset (String × String × String)) : Prop :=
  (∃ symb1 symb2 symb3, 
    symb1 ≠ symb2 ∧ symb2 ≠ symb3 ∧ symb1 ≠ symb3 ∧ 
    ∀ card ∈ cards, card.fst = symb1 ∨ card.fst = symb2 ∨ card.fst = symb3) ∨
  (∃ symb, ∀ card ∈ cards, card.fst = symb) ∧
  (
    (∃ col1 col2 col3, 
    col1 ≠ col2 ∧ col2 ≠ col3 ∧ col1 ≠ col3 ∧ 
    ∀ card ∈ cards, card.snd.fst = col1 ∨ card.snd.fst = col2 ∨ card.snd.fst = col3) ∨
    (∃ col, ∀ card ∈ cards, card.snd.fst = col)
  ) ∧
  (
    (∃ shade1 shade2 shade3, 
      shade1 ≠ shade2 ∧ shade2 ≠ shade3 ∧ shade1 ≠ shade3 ∧ 
      ∀ card ∈ cards, card.snd.snd = shade1 ∨ card.snd.snd = shade2 ∨ card.snd.snd = shade3
  ) ∨
  (∃ shade, ∀ card ∈ cards, card.snd.snd = shade)

theorem complementary_three_card_sets_count : 
  {s : Finset (String × String × String) | s.card = 3 ∧ is_complementary s}.card = 972 := sorry

end complementary_three_card_sets_count_l684_684287


namespace range_of_m_l684_684674

variable (m : ℝ)
def p (x : ℝ) : Prop := x^2 + 2 * x - m > 0

theorem range_of_m (hm1 : ¬ p 1) (hm2 : p 2) : 3 ≤ m ∧ m < 8 := sorry

end range_of_m_l684_684674


namespace percentage_markup_is_20_l684_684235

noncomputable def wholesale_cost := 200
noncomputable def employee_discount := 0.1
noncomputable def employee_paid_price := 216

theorem percentage_markup_is_20 :
  ∃ x : ℝ, 0 < x ∧ (let retail_price := wholesale_cost + (wholesale_cost * x / 100) in
                   0.9 * retail_price = employee_paid_price) ∧ x = 20 :=
by
  sorry

end percentage_markup_is_20_l684_684235


namespace tangent_slope_at_1_l684_684896

def f (x : ℝ) : ℝ := x^3 + x^2 + 1

theorem tangent_slope_at_1 : (deriv f 1) = 5 := by
  sorry

end tangent_slope_at_1_l684_684896


namespace trailing_zeros_in_100_factorial_l684_684123

theorem trailing_zeros_in_100_factorial :
  ∀ (n : ℕ), (n = 100) → (number_of_trailing_zeros (fact n) = 24) := 
begin
  intro n,
  intro h,
  subst h,
  -- We assume a function 'number_of_trailing_zeros'.
  have h₁ : number_of_trailing_zeros (fact 100) = 24,
  {
    -- Proof is omitted
    sorry,
  },
  exact h₁,
end

end trailing_zeros_in_100_factorial_l684_684123


namespace sequence_loop_l684_684012

def hyperbola (x y : ℝ) : Prop := y^2 - 4 * x^2 = 4

def P (n : ℕ) : ℝ

def ell_n (x_n : ℝ) (x y : ℝ) : Prop := y = 2 * (x - x_n)

noncomputable def next_P (x_n : ℝ) : ℝ :=
let x := (x_n^2 + 1) / (2 * x_n) in x

noncomputable def sequence_term (x_0 : ℝ) (n : ℕ) : ℝ := ηnotation  -- Definition of the sequence entry

def theta_0 (k : ℕ) (N : ℕ) : ℝ := k * π / (2^N - 1)

theorem sequence_loop (N : ℕ) :
  (∃ (x_0 : ℝ), P_0 := sequence_term x_0, 1004) = x_0 ) 
    → ∃ (k : ℕ), k ∈ finset.range (1, 2^N - 1)) := 
    ∑ (k ∈ finset.range (1, 2^1004 - 1)) :=
sorry

end sequence_loop_l684_684012


namespace no_simple_form_l684_684840

noncomputable def trig_expr_simplification (θ₁ θ₂ θ₃ θ₄ : ℝ) : Prop :=
  (sin θ₁ + sin θ₂ + sin θ₃ + sin θ₄) / (cos (θ₁ / 2) * cos (θ₂ / 2) * cos (θ₃ / 2) * cos (θ₄ / 2)) = θ₁ / cos (θ₂ / 2)

-- Using sum-to-product identities
lemma sin_sum_to_product_1 : sin (20 * (real.pi / 180)) + sin (80 * (real.pi / 180)) = 2 * sin (50 * (real.pi / 180)) * cos (30 * (real.pi / 180)) :=
by sorry

lemma sin_sum_to_product_2 : sin (40 * (real.pi / 180)) + sin (60 * (real.pi / 180)) = 2 * sin (50 * (real.pi / 180)) * cos (10 * (real.pi / 180)) :=
by sorry

-- Combining cosines using sum-to-product
lemma cos_sum_to_product : cos (30 * (real.pi / 180)) + cos (10 * (real.pi / 180)) = 2 * cos (20 * (real.pi / 180)) * cos (10 * (real.pi / 180)) :=
by sorry

-- Basic trigonometric identities
lemma cos_30_id : cos (30 * (real.pi / 180)) = real.sqrt 3 / 2 :=
by sorry

lemma cos_40_id : cos (40 * (real.pi / 180)) = sin (50 * (real.pi / 180)) * cos (10 * (real.pi / 180)) - cos (50 * (real.pi / 180)) * sin (10 * (real.pi / 180)) :=
by sorry

-- Main theorem
theorem no_simple_form :
  ¬ ∃ simplification, trig_expr_simplification (20 * (real.pi / 180)) (40 * (real.pi / 180)) (60 * (real.pi / 180)) (80 * (real.pi / 180)) = simplification :=
by sorry

end no_simple_form_l684_684840


namespace evaluate_g_at_5_l684_684508

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem evaluate_g_at_5 : g 5 = 15 :=
by
    -- proof steps here
    sorry

end evaluate_g_at_5_l684_684508


namespace min_value_expression_l684_684543

open Real

theorem min_value_expression : ∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 :=
by
  intro x y
  sorry

end min_value_expression_l684_684543


namespace sin_210_deg_l684_684522

theorem sin_210_deg : sin (210 * (Real.pi / 180)) = - (1 / 2) := 
by
  -- Given the trigonometric property and the value for sine 30 degrees, the theorem can be stated and proven.
  sorry

end sin_210_deg_l684_684522


namespace monthly_income_l684_684066

def average_expenditure_6_months (expenditure_6_months : ℕ) (average : ℕ) : Prop :=
  average = expenditure_6_months / 6

def expenditure_next_4_months (expenditure_4_months : ℕ) (monthly_expense : ℕ) : Prop :=
  expenditure_4_months = 4 * monthly_expense

def cleared_debt_and_saved (income_4_months : ℕ) (debt : ℕ) (savings : ℕ)  (condition : ℕ) : Prop :=
  income_4_months = debt + savings + condition

theorem monthly_income 
(income : ℕ) 
(avg_6m_exp : ℕ) 
(exp_4m : ℕ) 
(debt: ℕ) 
(savings: ℕ )
(condition: ℕ) 
    (h1 : average_expenditure_6_months avg_6m_exp 85) 
    (h2 : expenditure_next_4_months exp_4m 60) 
    (h3 : cleared_debt_and_saved (income * 4) debt savings 30) 
    (h4 : income * 6 < 6 * avg_6m_exp) 
    : income = 78 :=
sorry

end monthly_income_l684_684066


namespace trapezoid_area_l684_684976

def height := ℝ
def base1 (h : height) := 4 * h
def base2 (h : height) := 5 * h

theorem trapezoid_area (h : height) : 
  (1 / 2) * (base1 h + base2 h) * h = (9 * h ^ 2) / 2 := 
by 
  -- Placeholder for proof
  sorry

end trapezoid_area_l684_684976


namespace probability_intersection_correct_l684_684629

noncomputable def probability_of_intersection_line_circle : ℝ :=
  let lower_bound := (-1 / 2 : ℝ)
  let upper_bound := (1 / 2 : ℝ)
  let radius := (1 : ℝ)
  let k_interval := set.Icc lower_bound upper_bound

  let intersect_condition (k : ℝ) : Prop := 
    (abs (3 * k) / real.sqrt (k^2 + 1)) < radius

  let k_valid_interval := set.Icc (-real.sqrt 2 / 4) (real.sqrt 2 / 4)
  (real.volume (set.Icc (-real.sqrt 2 / 4) (real.sqrt 2 / 4)) / real.volume k_interval)

theorem probability_intersection_correct :
  probability_of_intersection_line_circle = real.sqrt 2 / 2 :=
sorry

end probability_intersection_correct_l684_684629


namespace equation_of_parallel_line_passing_through_point_l684_684078

-- Definitions for the problem
def point0 := (0 : ℝ, 3 : ℝ)

def slope_parallel := -4

def line1 (x y : ℝ) := y = slope_parallel * x + 1

def line2 (x y : ℝ) := y = slope_parallel * x + 3

def equation_line2 (x y : ℝ) := 4 * x + y - 3 = 0

-- Theorem to prove
theorem equation_of_parallel_line_passing_through_point :
  ∀ (x y : ℝ), ( (x, y) = point0 → line2 x y ) → equation_line2 x y :=
sorry

end equation_of_parallel_line_passing_through_point_l684_684078


namespace chord_length_of_ellipse_cut_by_line_l684_684071

-- Conditions
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 16
def line (x y : ℝ) : Prop := y = x + 1

-- Goal
theorem chord_length_of_ellipse_cut_by_line :
  ∀ (answer : ℝ), (∃ x1 y1 x2 y2 : ℝ,
   ellipse x1 y1 ∧ ellipse x2 y2 ∧ line x1 y1 ∧ line x2 y2 ∧ x1 ≠ x2 ∧
   answer = (real.sqrt ((x2 - x1)^2 + (y2 - y1)^2))) :=
sorry

end chord_length_of_ellipse_cut_by_line_l684_684071


namespace median_A_value_mode_A_value_variance_B_value_l684_684581

def scores_A : List ℕ := [85, 78, 86, 79, 72, 91, 79, 71, 70, 89]
def scores_B : List ℕ := [85, 80, 77, 85, 80, 73, 90, 74, 75, 81]
def students_A : ℕ := 45
def students_B : ℕ := 40
def prize_threshold : ℕ := 80

theorem median_A_value :
  List.median scores_A = 79 := by
  sorry

theorem mode_A_value :
  List.mode scores_A = 79 := by
  sorry

noncomputable def variance (l : List ℕ) : ℝ :=
  let mean := (l.sum / l.length : Int).toRat
  (l.map (λ x, (x - mean) ^ 2)).sum / l.length

theorem variance_B_value :
  variance scores_B = 27 := by
  sorry

end median_A_value_mode_A_value_variance_B_value_l684_684581


namespace sum_of_exponents_is_16_l684_684025

def consecutive_multiples_of_3 (x y : ℕ) : List ℕ :=
  List.filter (λ n, n % 3 = 0) (List.range' x (y + 1 - x))

def product_of_consecutive_multiples_of_3 (x y z : ℕ) : ℕ :=
  (consecutive_multiples_of_3 x y).prod * (consecutive_multiples_of_3 y z).prod

theorem sum_of_exponents_is_16 :
  let product := product_of_consecutive_multiples_of_3 21 30 36 in
  ∃ (factors : ℕ → ℕ), product = ∏ p in (Multiset.map factors (unique_factorization_monoid.factors product)).to_finset,
  ∑ p in (Multiset.map factors (unique_factorization_monoid.factors product)).to_finset, factors p = 16 ∧
  ∑ p in (Multiset.map factors (unique_factorization_monoid.factors product)).to_finset.filter (λ p, p ≠ 3), factors p % 2 = 0 :=
begin
  sorry,
end

end sum_of_exponents_is_16_l684_684025


namespace equation_of_curve_C_length_segment_AB_l684_684362

-- Definition of circles M and N
def circle_M (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 1

def circle_N (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 9

-- Definition of curve C (trajectory of the center of circle P)
def curve_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2

def line_tangent_circle_M (k : ℝ) : Prop :=
  abs (3 * k) / Real.sqrt (1 + k^2) = 1

def line_l (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k = sqrt 2 / 4 ∧ y = k * (x + 4) ∨ k = -sqrt 2 / 4 ∧ y = k * (x + 4)

-- Theorem 1: Prove the equation of curve C
theorem equation_of_curve_C :
  curve_C x y ↔ x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2 :=
by
  sorry

-- Theorem 2: Prove the length of segment |AB|
theorem length_segment_AB (A B : ℝ × ℝ) :
  let l_AB := dist A B
  line_l r ∘ A.1 = 0 ∧ line_l r ∘ B.1 = 0 → l_AB = 18 / 7 :=
by
  sorry

end equation_of_curve_C_length_segment_AB_l684_684362


namespace best_coupon1_discount_price_l684_684209

noncomputable def coupon1_discount (x : ℝ) : ℝ :=
  if x >= 60 then 0.12 * x else 0

noncomputable def coupon2_discount (x : ℝ) : ℝ :=
  if x >= 150 then 25 else 0

noncomputable def coupon3_discount (x : ℝ) : ℝ :=
  if x >= 150 then 0.20 * (Real.floor ((x - 150) / 10) * 10) else 0

def listed_prices : List ℝ := [189.95, 209.95, 229.95, 249.95, 269.95]

theorem best_coupon1_discount_price :
  ∃ x ∈ listed_prices, coupon1_discount x > coupon2_discount x ∧ coupon1_discount x > coupon3_discount x :=
by
  sorry

end best_coupon1_discount_price_l684_684209


namespace range_of_a_l684_684655

theorem range_of_a :
  ∀ (x : ℝ) (θ : ℝ), (0 ≤ θ ∧ θ ≤ π / 2) →
  ∀ (a : ℝ), (a ≤ √6 ∨ a ≥ 7 / 2) →
  (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1 / 8 :=
by
  intros x θ hθ a ha
  sorry

end range_of_a_l684_684655


namespace trajectory_of_M_max_area_of_triangle_AOB_l684_684471

-- Problem 1: Prove the equation of curve C and its eccentricity
theorem trajectory_of_M 
  (x y x0 y0 : ℝ)
  (P_on_circle : x0^2 + y0^2 = 3)
  (PD_MD_condition : y0 = sqrt 3 * y) :
  (x^2 + 3 * y^2 = 3) ∧ (sqrt 6 / 3 = sqrt 6 / 3) :=
  sorry

-- Problem 2: Prove the maximum area of triangle AOB
theorem max_area_of_triangle_AOB 
  (A B O : ℝ × ℝ)
  (distance_from_origin_to_line : ∀ k m : ℝ, (m^2 = 3/4 * (k^2 + 1)) → abs m / (sqrt (1 + k^2)) = sqrt 3 / 2) 
  (C : set (ℝ × ℝ): ℝ × ℝ → Prop := λ p, (p.1^2 / 3) + (p.2^2) = 1)
  (intersect_AB : ∃ k m : ℝ, y = k * x + m ∧ y = k * x - m) :
  (∀ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C → 
  2 = max (dist A B))
  ∧ 
  (area_of_triangle O A B = (sqrt 3 / 2)) :=
  sorry

end trajectory_of_M_max_area_of_triangle_AOB_l684_684471


namespace polynomial_with_real_roots_maintained_l684_684049

noncomputable def polynomial_has_all_real_roots (p : Polynomial ℝ) : Prop :=
  ∀ (x : ℝ), Polynomial.degree p = (Polynomial.degree_leading_coeff p).natDegree

theorem polynomial_with_real_roots_maintained
  {f : Polynomial ℝ}
  (h : polynomial_has_all_real_roots f)
  (λ : ℝ) :
  polynomial_has_all_real_roots (f + λ • polynomial.derivative f) :=
sorry

end polynomial_with_real_roots_maintained_l684_684049


namespace no_bijective_function_l684_684432

open Set

def is_bijective {α β : Type*} (f : α → β) : Prop :=
  Function.Bijective f

def are_collinear {P : Type*} (A B C : P) : Prop :=
  sorry -- placeholder for the collinearity predicate on points

def are_parallel_or_concurrent {L : Type*} (l₁ l₂ l₃ : L) : Prop :=
  sorry -- placeholder for the condition that lines are parallel or concurrent

theorem no_bijective_function (P : Type*) (D : Type*) :
  ¬ ∃ (f : P → D), is_bijective f ∧
    ∀ A B C : P, are_collinear A B C → are_parallel_or_concurrent (f A) (f B) (f C) :=
by
  sorry

end no_bijective_function_l684_684432


namespace area_of_triangle_AEF_l684_684912

variable (A B C D E F G : Type)
variable [HasArea A B C D : ℝ]
variable [HasArea E F G : ℝ]

-- Trapezoid ABCD with area 50
axiom trapezoid_ABCD : trapezoid A B C D
axiom area_ABCD : area trapezoid_ABCD = 50

-- Points E and F are midpoints
axiom midpoint_A_first : midpoint A D E
axiom midpoint_B_first : midpoint B C F

-- Trapezoid ABEF with area 20
axiom trapezoid_ABEF : trapezoid A B E F
axiom area_ABEF : area trapezoid_ABEF = 20

-- EF is parallel to AB and CD
axiom parallel_AB_EF_CD : parallel AB EF
axiom parallel_CD_EF_AB : parallel CD EF

-- Area of triangle AEF
def area_triangle_AEF : ℝ := 15

-- Theorem stating the area of triangle AEF
theorem area_of_triangle_AEF :
  area triangle A E F = 15 := 
  sorry

end area_of_triangle_AEF_l684_684912


namespace gcd_markers_l684_684459

variable (n1 n2 n3 : ℕ)

-- Let the markers Mary, Luis, and Ali bought be represented by n1, n2, and n3
def MaryMarkers : ℕ := 36
def LuisMarkers : ℕ := 45
def AliMarkers : ℕ := 75

theorem gcd_markers : Nat.gcd (Nat.gcd MaryMarkers LuisMarkers) AliMarkers = 3 := by
  sorry

end gcd_markers_l684_684459


namespace mr_brown_selling_price_l684_684465

noncomputable def initial_price : ℝ := 100000
noncomputable def profit_percentage : ℝ := 0.10
noncomputable def loss_percentage : ℝ := 0.10

def selling_price_mr_brown (initial_price profit_percentage : ℝ) : ℝ :=
  initial_price * (1 + profit_percentage)

def selling_price_to_friend (selling_price_mr_brown loss_percentage : ℝ) : ℝ :=
  selling_price_mr_brown * (1 - loss_percentage)

theorem mr_brown_selling_price :
  selling_price_to_friend (selling_price_mr_brown initial_price profit_percentage) loss_percentage = 99000 :=
by
  sorry

end mr_brown_selling_price_l684_684465


namespace total_songs_l684_684666

variable (H : String) (M : String) (A : String) (T : String)

def num_songs (s : String) : ℕ :=
  if s = H then 9 else
  if s = M then 5 else
  if s = A ∨ s = T then 
    if H ≠ s ∧ M ≠ s then 6 else 7 
  else 0

theorem total_songs 
  (hH : num_songs H = 9)
  (hM : num_songs M = 5)
  (hA : 5 < num_songs A ∧ num_songs A < 9)
  (hT : 5 < num_songs T ∧ num_songs T < 9) :
  (num_songs H + num_songs M + num_songs A + num_songs T) / 3 = 10 :=
sorry

end total_songs_l684_684666


namespace max_probability_at_4_l684_684117
noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  if h : k ≤ n then nat.choose n k else 0

noncomputable def probability (k : ℕ) : ℚ :=
  binomial_coefficient 5 k * ((3 / 4) ^ k) * ((1 / 4) ^ (5 - k))

theorem max_probability_at_4 : 
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ 5 → probability k ≤ probability 4) :=
begin
  intro k,
  sorry
end

end max_probability_at_4_l684_684117


namespace probability_three_kings_or_at_least_two_aces_l684_684741

variable (deck_size : ℕ := 52)
variable (aces_count : ℕ := 4)
variable (kings_count : ℕ := 4)

theorem probability_three_kings_or_at_least_two_aces :
  (real.to_rat (
    (kings_count.to_real / deck_size.to_real) *
    ((kings_count - 1).to_real / (deck_size - 1).to_real) *
    ((kings_count - 2).to_real / (deck_size - 2).to_real) +
    ((
      ((binomial_coefficient aces_count 2) * (binomial_coefficient (deck_size - aces_count) 1)).to_real /
      (binomial_coefficient deck_size 3).to_real
    ) +
    ((aces_count.to_real / deck_size.to_real) *
    ((aces_count - 1).to_real / (deck_size - 1).to_real) *
    ((aces_count - 2).to_real / (deck_size - 2).to_real / 3))
    ).numerator.to_real /
    ((deck_size * (deck_size - 1) * (deck_size - 2) / 6).to_real)
  )) = 74/5525 := sorry

end probability_three_kings_or_at_least_two_aces_l684_684741


namespace electricity_cost_one_kilometer_minimum_electricity_kilometers_l684_684205

-- Part 1: Cost of traveling one kilometer using electricity only
theorem electricity_cost_one_kilometer (x : ℝ) (fuel_cost : ℝ) (electricity_cost : ℝ) 
  (total_fuel_cost : ℝ) (total_electricity_cost : ℝ) 
  (fuel_per_km_more_than_electricity : ℝ) (distance_fuel : ℝ) (distance_electricity : ℝ)
  (h1 : total_fuel_cost = distance_fuel * fuel_cost)
  (h2 : total_electricity_cost = distance_electricity * electricity_cost)
  (h3 : fuel_per_km_more_than_electricity = 0.5)
  (h4 : fuel_cost = electricity_cost + fuel_per_km_more_than_electricity)
  (h5 : distance_fuel = 76 / (electricity_cost + 0.5))
  (h6 : distance_electricity = 26 / electricity_cost) : 
  x = 0.26 :=
sorry

-- Part 2: Minimum kilometers traveled using electricity
theorem minimum_electricity_kilometers (total_trip_cost : ℝ) (electricity_per_km : ℝ) 
  (hybrid_total_km : ℝ) (max_total_cost : ℝ) (fuel_per_km : ℝ) (y : ℝ)
  (h1 : electricity_per_km = 0.26)
  (h2 : fuel_per_km = 0.26 + 0.5)
  (h3 : hybrid_total_km = 100)
  (h4 : max_total_cost = 39)
  (h5 : total_trip_cost = electricity_per_km * y + (hybrid_total_km - y) * fuel_per_km)
  (h6 : total_trip_cost ≤ max_total_cost) :
  y ≥ 74 :=
sorry

end electricity_cost_one_kilometer_minimum_electricity_kilometers_l684_684205


namespace units_digit_of_50_factorial_l684_684177

theorem units_digit_of_50_factorial : 
  ∃ d, (d = List.prod (List.range 1 51)) ∧ (d % 10 = 0) :=
by
  sorry

end units_digit_of_50_factorial_l684_684177


namespace sum_of_angles_l684_684072

theorem sum_of_angles (x y : ℝ) (n : ℕ) :
  n = 16 →
  (∃ k l : ℕ, k = 3 ∧ l = 5 ∧ 
  x = (k * (360 / n)) / 2 ∧ y = (l * (360 / n)) / 2) →
  x + y = 90 :=
by
  intros
  sorry

end sum_of_angles_l684_684072


namespace incircle_incenter_on_segment_iff_circumcircle_circumcenter_on_segment_l684_684764

variables {A B C H_a H_b W_a W_b I O : Type}
variables [AcuteAngledTriangle A B C]
variables [FootOfAltitude H_a A B C] [FootOfAltitude H_b B A C]
variables [IntersectionOfAngleBisector W_a A C] [IntersectionOfAngleBisector W_b B A]
variables [Incenter I A B C] [Circumcenter O A B C]

theorem incircle_incenter_on_segment_iff_circumcircle_circumcenter_on_segment :
  collinear I H_a H_b ↔ collinear O W_a W_b :=
sorry

end incircle_incenter_on_segment_iff_circumcircle_circumcenter_on_segment_l684_684764


namespace quadratic_roots_l684_684682

noncomputable def complex_roots_of_quadratic (b c : ℝ) : Prop :=
  (∃ z : ℂ, z = 1 + complex.i ∧ (z ^ 2 + b * z + c = (0 : ℂ))) ∧
  (∃ z : ℂ, z = 1 - complex.i ∧ (z ^ 2 + b * z + c = (0 : ℂ)))

theorem quadratic_roots (b c : ℝ) (hb : b = -2) (hc : c = 2) : complex_roots_of_quadratic b c :=
by {
  sorry
}

end quadratic_roots_l684_684682


namespace units_digit_factorial_50_l684_684135

theorem units_digit_factorial_50 : Nat.unitsDigit (Nat.factorial 50) = 0 := 
  sorry

end units_digit_factorial_50_l684_684135


namespace doctor_visit_cost_l684_684533

variable (pills_per_day : ℕ)
variable (days_per_year : ℕ)
variable (pill_cost : ℕ)
variable (insurance_coverage : ℚ)
variable (total_annual_cost : ℕ)
variable (doctor_visits_per_year : ℕ)

theorem doctor_visit_cost
  (h1 : pills_per_day = 2)
  (h2 : days_per_year = 365)
  (h3 : pill_cost = 5)
  (h4 : insurance_coverage = 0.80)
  (h5 : total_annual_cost = 1530)
  (h6 : doctor_visits_per_year = 2) :
  
  let annual_pills := pills_per_day * days_per_year in
  let annual_pill_cost := annual_pills * pill_cost in
  let annual_patient_pill_cost := (1 - insurance_coverage) * annual_pill_cost in
  let total_doctor_cost := total_annual_cost - annual_patient_pill_cost in
  total_doctor_cost / doctor_visits_per_year = 400 :=

by
  have annual_pills_eq : annual_pills = 2 * 365 := by rw [←h1, ←h2]
  have annual_pill_cost_eq : annual_pill_cost = 730 * 5 := by rw [annual_pills_eq, ←h3]
  have annual_patient_pill_cost_eq : annual_patient_pill_cost = (1 - 0.80) * 3650 := by rw [annual_pill_cost_eq, ←h4]
  have total_doctor_cost_eq : total_doctor_cost = 1530 - 730 := by rw [annual_patient_pill_cost_eq, ←h5]
  have cost_per_visit_eq : total_doctor_cost / doctor_visits_per_year = 400 := by rw [total_doctor_cost_eq, ←h6]
  exact cost_per_visit_eq

end doctor_visit_cost_l684_684533


namespace max_value_of_trig_function_l684_684922

theorem max_value_of_trig_function :
  ∀ x ∈ Set.Ico 0 (2 * Real.pi), (sin x - (Real.sqrt 3) * cos x) ≤ 2 ∧ 
  ∃ x ∈ Set.Ico 0 (2 * Real.pi), (sin x - (Real.sqrt 3) * cos x) = 2 ∧ x = (5 * Real.pi) / 6 :=
by
  sorry

end max_value_of_trig_function_l684_684922


namespace quadratic_poly_correct_l684_684302

noncomputable def quadratic_poly : polynomial ℝ :=
  3 * polynomial.C 1 * ((polynomial.X - polynomial.C (4 + 2 * complex.I)) * 
  (polynomial.X - polynomial.C (4 - 2 * complex.I)))

theorem quadratic_poly_correct : 
  quadratic_poly = 3 * polynomial.X ^ 2 - 24 * polynomial.X + 60 := by
  sorry

end quadratic_poly_correct_l684_684302


namespace intervals_of_monotonicity_and_extreme_values_l684_684639

def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 1

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x ∈ set.Icc (0:ℝ) (2:ℝ), f x ≤ 1) ∧
  (∀ x ∈ set.Icc (2:ℝ) (3:ℝ), f 2 ≤ f x) ∧
  (f 0 = 1) ∧
  (f 3 = 1) ∧
  (f 2 = -7) ∧
  (f ∈ ((-2:ℝ), 0) ↗) ∧
  (f ∈ ((0:ℝ), 2) ↘) ∧
  (f ∈ ((2:ℝ), 3) ↗) := 
sorry

end intervals_of_monotonicity_and_extreme_values_l684_684639


namespace total_money_raised_l684_684620

-- Define the baked goods quantities
def betty_chocolate_chip_cookies := 4
def betty_oatmeal_raisin_cookies := 6
def betty_regular_brownies := 2
def paige_sugar_cookies := 6
def paige_blondies := 3
def paige_cream_cheese_swirled_brownies := 5

-- Define the price of goods
def cookie_price := 1
def brownie_price := 2

-- State the total money raised
theorem total_money_raised :
  let total_cookies := 12 * (betty_chocolate_chip_cookies + betty_oatmeal_raisin_cookies + paige_sugar_cookies),
      total_brownies := 12 * (betty_regular_brownies + paige_blondies + paige_cream_cheese_swirled_brownies),
      money_from_cookies := total_cookies * cookie_price,
      money_from_brownies := total_brownies * brownie_price
  in money_from_cookies + money_from_brownies = 432 := by
  sorry

end total_money_raised_l684_684620


namespace minnie_takes_more_time_l684_684035

def minnie_speed_flat : ℝ := 25
def minnie_speed_downhill : ℝ := 35
def minnie_speed_uphill : ℝ := 10
def penny_speed_flat : ℝ := 35
def penny_speed_downhill : ℝ := 45
def penny_speed_uphill : ℝ := 15

def distance_A_to_B : ℝ := 15
def distance_B_to_D : ℝ := 20
def distance_D_to_C : ℝ := 25

def distance_C_to_B : ℝ := 20
def distance_D_to_A : ℝ := 25

noncomputable def time_minnie : ℝ :=
(distance_A_to_B / minnie_speed_uphill) + 
(distance_B_to_D / minnie_speed_downhill) + 
(distance_D_to_C / minnie_speed_flat)

noncomputable def time_penny : ℝ :=
(distance_C_to_B / penny_speed_uphill) + 
(distance_B_to_D / penny_speed_downhill) + 
(distance_D_to_A / penny_speed_flat)

noncomputable def time_diff : ℝ := (time_minnie - time_penny) * 60

theorem minnie_takes_more_time : time_diff = 10 := by
  sorry

end minnie_takes_more_time_l684_684035


namespace units_digit_of_50_factorial_is_0_l684_684169

theorem units_digit_of_50_factorial_is_0 : 
  (∃ n : ℕ, 50! ≡ n [MOD 10]) ∧ (n = 0) := sorry

end units_digit_of_50_factorial_is_0_l684_684169


namespace trajectory_of_midpoint_l684_684344

theorem trajectory_of_midpoint (t : ℝ):
  (∃ (A B : ℝ × ℝ), 
     (A.1^2 + 2 * A.2^2 = 4) ∧ 
     (B.1^2 + 2 * B.2^2 = 4) ∧ 
     (A.2 = 2 * A.1 + t) ∧ 
     (B.2 = 2 * B.1 + t)) → 
  ∀ x y : ℝ, 
    ((-4 * real.sqrt 2 / 3 < x) ∧ 
    (x < 4 * real.sqrt 2 / 3)) → 
    y = -1 / 4 * x :=
by sorry

end trajectory_of_midpoint_l684_684344


namespace line_passes_through_fixed_point_minimum_triangle_area_and_line_equation_l684_684354

-- Given line l with parameter k in real number domain and the point O as origin
variables (k : ℝ)

def line_l (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

theorem line_passes_through_fixed_point :
  line_l k (-2) 1 :=
by sorry

theorem minimum_triangle_area_and_line_equation :
  ∃ (S : ℝ) (k' : ℝ), (line_l (k') k' = (k' = 1/2)) ∧ S = 4 :=
by sorry

end line_passes_through_fixed_point_minimum_triangle_area_and_line_equation_l684_684354


namespace units_digit_50_factorial_l684_684164

theorem units_digit_50_factorial : (nat.factorial 50) % 10 = 0 :=
by
  sorry

end units_digit_50_factorial_l684_684164


namespace necessary_but_not_sufficient_condition_l684_684805

theorem necessary_but_not_sufficient_condition (x y : ℝ) : 
  (x ≠ y → cos x ≠ cos y) ∧ (∃ x y : ℝ, x ≠ y ∧ cos x = cos y) :=
by
  sorry

end necessary_but_not_sufficient_condition_l684_684805


namespace parity_of_fx_minus_one_monotonicity_of_f_value_of_f5_l684_684271

-- Conditions
variable (f : ℝ → ℝ)
variable (H1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y) - 1)
variable (H2 : ∀ x < 0, f(x) > 1)

-- Proof statements
theorem parity_of_fx_minus_one : (∀ x, (f(x) - 1) = -(f(-x) - 1)) :=
sorry

theorem monotonicity_of_f : (∀ x1 x2 : ℝ, x1 < x2 → f(x1) > f(x2)) :=
sorry

theorem value_of_f5 : f(5) = -13 / 2 :=
sorry

end parity_of_fx_minus_one_monotonicity_of_f_value_of_f5_l684_684271


namespace meet_time_correct_l684_684930

variable (circumference : ℕ) (speed_yeonjeong speed_donghun : ℕ)

def meet_time (circumference speed_yeonjeong speed_donghun : ℕ) : ℕ :=
  circumference / (speed_yeonjeong + speed_donghun)

theorem meet_time_correct
  (h_circumference : circumference = 3000)
  (h_speed_yeonjeong : speed_yeonjeong = 100)
  (h_speed_donghun : speed_donghun = 150) :
  meet_time circumference speed_yeonjeong speed_donghun = 12 :=
by
  rw [h_circumference, h_speed_yeonjeong, h_speed_donghun]
  norm_num
  sorry

end meet_time_correct_l684_684930


namespace part_i_part_ii_l684_684442

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + x - a
noncomputable def g (x a : ℝ) : ℝ := Real.sqrt (f x a)

theorem part_i (a : ℝ) :
  (∀ x ∈ Set.Icc (0:ℝ) (1:ℝ), f x a ≥ 0) ↔ (a ≤ 1) :=
by {
  -- Suppose it is already known that theorem is true.
  sorry
}

theorem part_ii (a : ℝ) :
  (∃ x0 y0 : ℝ, (x0, y0) ∈ (Set.Icc (-1) 1) ∧ y0 = Real.cos (2 * x0) ∧ g (g y0 a) a = y0) ↔ (1 ≤ a ∧ a ≤ Real.exp 1) :=
by {
  -- Suppose it is already known that theorem is true.
  sorry
}

end part_i_part_ii_l684_684442


namespace track_length_l684_684829

theorem track_length (speedA speedB : ℕ) (time : ℕ) (meetings : ℕ) 
  (hA : speedA = 180) (hB : speedB = 240) (hTime : time = 30) (hMeetings : meetings = 24) :
  ∃ length : ℕ, length = 525 := 
by
  have combined_speed := speedA + speedB
  have total_distance := combined_speed * time
  have length := total_distance / meetings
  use length
  have h_length : length = 525
  sorry

end track_length_l684_684829


namespace quadratic_sum_l684_684917

theorem quadratic_sum (x : ℝ) (h : x^2 = 16*x - 9) : x = 8 ∨ x = 9 := sorry

end quadratic_sum_l684_684917


namespace units_digit_factorial_50_l684_684136

theorem units_digit_factorial_50 : Nat.unitsDigit (Nat.factorial 50) = 0 := 
  sorry

end units_digit_factorial_50_l684_684136


namespace numbers_not_equal_l684_684420

theorem numbers_not_equal
  (a b c S : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h1 : a + b^2 + c^2 = S)
  (h2 : b + a^2 + c^2 = S)
  (h3 : c + a^2 + b^2 = S) :
  ¬ (a = b ∧ b = c) :=
by sorry

end numbers_not_equal_l684_684420


namespace sum_seven_consecutive_integers_l684_684854

theorem sum_seven_consecutive_integers (m : ℕ) :
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) + (m + 6) = 7 * m + 21 :=
by
  -- Sorry to skip the actual proof steps.
  sorry

end sum_seven_consecutive_integers_l684_684854


namespace solution_count_l684_684482

theorem solution_count (a : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ sqrt x + sqrt (6 - 2 * x) = a) ↔ 
  (a = 3 ∨ (0 ≤ a ∧ a < real.sqrt 3 ∧ ∃ x1 : ℝ, sqrt x1 + sqrt (6 - 2 * x1) = a ∧ 0 ≤ x1 ≤ 3) ∨
  (real.sqrt 3 ≤ a ∧ a < 3 ∧ ∃ x2 x3 : ℝ, sqrt x2 + sqrt (6 - 2 * x2) = a ∧ sqrt x3 + sqrt (6 - 2 * x3) = a ∧ x2 ≠ x3 ∧ 0 ≤ x2 ≤ 3 ∧ 0 ≤ x3 ≤ 3)) :=
sorry

end solution_count_l684_684482


namespace johns_final_push_time_l684_684002

-- Definitions and assumptions
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.8
def initial_gap : ℝ := 15
def final_gap : ℝ := 2

theorem johns_final_push_time :
  ∃ t : ℝ, john_speed * t = steve_speed * t + initial_gap + final_gap ∧ t = 42.5 :=
by
  sorry

end johns_final_push_time_l684_684002


namespace bin_10101_eq_21_l684_684517

-- Define the binary number
def binaryNumber : List ℕ := [1, 0, 1, 0, 1]

-- Function to convert binary to decimal
def bin_to_dec (b : List ℕ) : ℕ :=
  b.reverse.foldl (λ acc bit, acc * 2 + bit) 0

-- The statement to prove
theorem bin_10101_eq_21 : bin_to_dec binaryNumber = 21 :=
by
  sorry

end bin_10101_eq_21_l684_684517


namespace corresponding_angles_relation_l684_684335

theorem corresponding_angles_relation (a1 a2 : ℝ) (h : ∃ l1 l2 t : Type, corresponding_angles l1 l2 t a1 a2) :
  a1 = a2 ∨ a1 > a2 ∨ a1 < a2 := by
  sorry

end corresponding_angles_relation_l684_684335


namespace solve_quadratic_and_linear_l684_684377

theorem solve_quadratic_and_linear :
  ∃ y z : ℝ, y^2 - 6*y + 9 = 0 ∧ y + z = 11 ∧ y = 3 ∧ z = 8 :=
by
  use [3, 8]
  split
  · rw [sub_self]
    norm_num
  split
  · norm_num
  split
  · refl
  · refl

end solve_quadratic_and_linear_l684_684377


namespace at_most_three_lambdas_l684_684020

theorem at_most_three_lambdas
  (P Q : RealPolynomial)
  (h_coprime : P.coprime Q)
  (h_nonconstant_P : P.degree > 0)
  (h_nonconstant_Q : Q.degree > 0) :
  ∃ (Λ : Finset ℝ) (hsize : Λ.card ≤ 3), ∀ λ ∈ Λ, ∃ A : RealPolynomial, P + λ * Q = A^2 := 
  sorry

end at_most_three_lambdas_l684_684020


namespace max_sides_cross_section_of_hexagonal_prism_l684_684632

-- Definition of a regular hexagonal prism with 8 faces.
def regular_hexagonal_prism : Type := sorry -- This would be a concrete definition in practice.

-- Definition of a cross-section.
def cross_section (P : regular_hexagonal_prism) (plane : ℝ → ℝ → ℝ → ℝ) : polygon := sorry

-- Statement of the problem.
theorem max_sides_cross_section_of_hexagonal_prism (P : regular_hexagonal_prism) (plane : ℝ → ℝ → ℝ → ℝ) : 
  ∃ n, (cross_section P plane).sides = n ∧ n ≤ 8 :=
sorry

end max_sides_cross_section_of_hexagonal_prism_l684_684632


namespace initial_principal_solution_l684_684947

noncomputable def initial_principal (P : ℝ) : Prop :=
  let A1 := P * 1.18
  let A1_prime := A1 + 0.25 * P
  let A2 := A1_prime * (1.15 ^ 2)
  let B1 := P * 1.12
  let B1_prime := B1 + 0.25 * P
  let B2 := B1_prime * (1.12 ^ 2)
  (A2 - B2 = 500)

theorem initial_principal_solution :
  ∃ P : ℝ, initial_principal P ∧ P ≈ 2896.77 :=
by
  use 2896.77
  rw initial_principal
  -- Further calculations and steps would go here
  sorry

end initial_principal_solution_l684_684947


namespace log_base_2_min_value_l684_684575

theorem log_base_2_min_value : ∀ x > 0, f(x) = log 2 x → (∀ y > 0, log 2 y ≥ 0) :=
by 
  intro x hx hf
  sorry

end log_base_2_min_value_l684_684575


namespace solve_proof_problem_l684_684373

variables (a b c d : ℝ)

noncomputable def proof_problem : Prop :=
  a = 3 * b ∧ b = 3 * c ∧ c = 5 * d → (a * c) / (b * d) = 15

theorem solve_proof_problem : proof_problem a b c d :=
by
  sorry

end solve_proof_problem_l684_684373


namespace twelve_months_game_probability_l684_684525

/-- The card game "Twelve Months" involves turning over cards according to a set of rules.
Given the rules, we are asked to find the probability that all 12 columns of cards can be fully turned over. -/
def twelve_months_probability : ℚ :=
  1 / 12

theorem twelve_months_game_probability :
  twelve_months_probability = 1 / 12 :=
by
  -- The conditions and their representations are predefined.
  sorry

end twelve_months_game_probability_l684_684525


namespace external_side_length_is_correct_l684_684780

noncomputable def internal_volume_cubic_inches := 4 * 1728
noncomputable def internal_side_length := (internal_volume_cubic_inches : ℝ)^(1 / 3)
noncomputable def wall_thickness := 1
noncomputable def external_side_length := internal_side_length + 2 * wall_thickness

theorem external_side_length_is_correct :
  external_side_length ≈ 21.08 :=
begin
  sorry
end

end external_side_length_is_correct_l684_684780


namespace spending_on_other_items_is_30_percent_l684_684825

-- Define the total amount Jill spent excluding taxes
variable (T : ℝ)

-- Define the amounts spent on clothing, food, and other items as percentages of T
def clothing_spending : ℝ := 0.50 * T
def food_spending : ℝ := 0.20 * T
def other_items_spending (x : ℝ) : ℝ := x * T

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0.0
def other_items_tax_rate : ℝ := 0.10

-- Define the taxes paid on each category
def clothing_tax : ℝ := clothing_tax_rate * clothing_spending T
def food_tax : ℝ := food_tax_rate * food_spending T
def other_items_tax (x : ℝ) : ℝ := other_items_tax_rate * other_items_spending T x

-- Define the total tax paid as a percentage of the total amount spent excluding taxes
def total_tax_paid : ℝ := 0.05 * T

-- The main theorem stating that the percentage of the amount spent on other items is 30%
theorem spending_on_other_items_is_30_percent (x : ℝ) (h : total_tax_paid T = clothing_tax T + other_items_tax T x) :
  x = 0.30 :=
sorry

end spending_on_other_items_is_30_percent_l684_684825


namespace roof_area_l684_684096

theorem roof_area (w l : ℕ) (h1 : l = 4 * w) (h2 : l - w = 42) : l * w = 784 :=
by
  sorry

end roof_area_l684_684096


namespace quadrilateral_equality_l684_684570

variables {A B C D M N P Q : Type}
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D]
variables [AffineSpace ℝ M] [AffineSpace ℝ N] [AffineSpace ℝ P] [AffineSpace ℝ Q]

-- Conditions
def is_midpoint (M : A → B → Prop) (N : D → C → Prop) : Prop :=
  M ≠ N

def intersection_point (P Q : P → Q → Prop) : Prop :=
  ∃ x y : Line, x ∩ y = {P, Q}

-- Angles
def angle_eq (B M Q : ∠ BQM = ∠ APM) : Prop :=
  ∠ BQM = ∠ APM

theorem quadrilateral_equality (A B C D M N P Q : A, B, C, D, M, N, P, Q) :
  is_midpoint (A, B, M) ∧ is_midpoint (D, C, N) ∧ 
  intersection_point (A, D, P) ∧ intersection_point (B, C, Q) ∧ 
  angle_eq (B, M, Q) (A, M, P) →
  distance B C = distance A D :=
sorry

end quadrilateral_equality_l684_684570


namespace Luke_spent_money_l684_684458

theorem Luke_spent_money : ∀ (initial_money additional_money current_money x : ℕ),
  initial_money = 48 →
  additional_money = 21 →
  current_money = 58 →
  (initial_money + additional_money - current_money) = x →
  x = 11 :=
by
  intros initial_money additional_money current_money x h1 h2 h3 h4
  sorry

end Luke_spent_money_l684_684458


namespace cubic_identity_l684_684446

noncomputable def roots := {p q r : ℝ // polynomial.root_set (polynomial.C 2 + polynomial.C -3 * polynomial.X + polynomial.X^3) ⊆ {p, q, r}}

theorem cubic_identity (p q r : ℝ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -3) (h3 : p * q * r = -2) :
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 12 := 
by 
  sorry

end cubic_identity_l684_684446


namespace values_of_n_le_100_with_g30_eq_18_l684_684308

-- Define the function g_1 as thrice the number of divisors of n
def g_1 (n : ℕ) : ℕ := 3 * (Nat.divisors n).length

-- Define the function g_j recursively
def g (j : ℕ) (n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | k+1 => g_1 (g k n)

-- Define the main problem statement
theorem values_of_n_le_100_with_g30_eq_18 : (Set.filter (λ n => g 30 n = 18) (Finset.range 101).toSet).card = 1 := 
  sorry

end values_of_n_le_100_with_g30_eq_18_l684_684308


namespace daniel_initial_noodles_l684_684269

theorem daniel_initial_noodles (noodles_given noodles_left : ℕ) 
  (h_given : noodles_given = 12)
  (h_left : noodles_left = 54) :
  ∃ x, x - noodles_given = noodles_left ∧ x = 66 := 
by 
sory

end daniel_initial_noodles_l684_684269


namespace hyperbola_eccentricity_l684_684352

theorem hyperbola_eccentricity (a b c : ℝ)
  (hyp : ∀ {x y : ℝ}, (x^2 / a^2 - y^2 / b^2 = 1) → (y = ± b / a * x))
  (parabola : ∀ {x y : ℝ}, (y = x^2 + 1) → (x^2 ± b / a * x + 1 = 0))
  (intersect_one_point : (b^2 / a^2 - 4 = 0)) :
  (c = √ (a^2 + b^2)) → 
  ((√ (a^2 + b^2)) / a = √ 5) :=
by sorry

end hyperbola_eccentricity_l684_684352


namespace algae_growth_l684_684207

theorem algae_growth (initial_cells : ℕ) (target_cells : ℕ) (rate : ℕ) (tripling_period : ℕ) 
  (h_initial : initial_cells = 200) (h_target : target_cells = 145800) (h_rate : rate = 3)
  (h_tripling_period : tripling_period = 5):
  ∃ (hours : ℕ), hours = 30 ∧ ((initial_cells * rate ^ (hours / tripling_period)) ≥ target_cells) := 
begin
  -- Proof goes here
  sorry
end

end algae_growth_l684_684207


namespace units_digit_of_product_1_to_50_is_zero_l684_684153

theorem units_digit_of_product_1_to_50_is_zero :
  Nat.digits 10 (∏ i in Finset.range 51, i) = [0] :=
sorry

end units_digit_of_product_1_to_50_is_zero_l684_684153


namespace correct_option_is_B_l684_684613

-- Define the given options
def optionA := (a + b) / c
def optionB := | 4 | / 3
def optionC := (x - 2 * y) / 2
def optionD := x * y * 5

-- Prove that the correct option is B
theorem correct_option_is_B (a b c x y : ℝ) : 
  (optionB = | 4 | / 3) ∧ 
  (optionA ≠ (a + b) / c) ∧ 
  (optionC ≠ (x - 2 * y) / 2) ∧ 
  (optionD ≠ 5 * x * y) := 
  by 
    -- Add sorry to skip the actual proof
    sorry

end correct_option_is_B_l684_684613


namespace midpoint_of_AB_l684_684356

theorem midpoint_of_AB (xA xB : ℝ) (p : ℝ) (h_parabola : ∀ y, y^2 = 4 * xA → y^2 = 4 * xB)
  (h_focus : (2 : ℝ) = p)
  (h_length_AB : (abs (xB - xA)) = 5) :
  (xA + xB) / 2 = 3 / 2 :=
sorry

end midpoint_of_AB_l684_684356


namespace sequence_y_is_arithmetic_Tn_greater_than_bound_l684_684332

-- Step 1: Proving the sequence {y_n} is arithmetic
theorem sequence_y_is_arithmetic (n : ℕ) (y : ℕ → ℝ) :
  (∀ m : ℕ, y m = 1/2 * m + 1) → (∃ d : ℝ, ∀ m : ℕ, y (m + 1) - y m = d) :=
by 
  intro h
  use 1/2
  intro m
  calc 
    y (m + 1) - y m = (1 / 2 * (m + 1) + 1) - (1 / 2 * m + 1) : by rw [h (m + 1), h m]
                  ... = 1/2

-- Step 2: Expressing S_{2n-1}
noncomputable def S_odd (a : ℝ) (n : ℕ) : ℝ := (1 - a) * (2 * n + 1) / 2

#check S_odd

-- Step 3: Proving T_n > 8n / (3n + 4)
def T (S_odd S_even : ℕ → ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, 1 / (S_odd k * S_even k)

theorem Tn_greater_than_bound (a : ℝ) (S_odd S_even : ℕ → ℝ) (n : ℕ) :
  T S_odd S_even n > 8 * n / (3 * n + 4) :=
by sorry

end sequence_y_is_arithmetic_Tn_greater_than_bound_l684_684332


namespace problem_l684_684441

noncomputable def f : ℝ → ℝ := sorry

theorem problem :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 2) = -f x) →
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) →
  f 3 = -1 ∧
  ∫ x in -4..4, abs (f x) = 4 :=
begin
  sorry
end

end problem_l684_684441


namespace student_solution_correct_l684_684936

variable (x : ℕ)
def twice_as_many_wrong_as_right (x : ℕ) : Prop := 2 * x + x = 54

theorem student_solution_correct : ∃ x, twice_as_many_wrong_as_right x ∧ x = 18 :=
by
  use 18
  simp [twice_as_many_wrong_as_right]
  sorry

end student_solution_correct_l684_684936


namespace f_neg_t_l684_684702

def f (x : ℝ) : ℝ := x^3 + 2 * x + (1 / x) - 3

theorem f_neg_t (t : ℝ) (ht : f t = 4) : f (-t) = -10 :=
by
  sorry

end f_neg_t_l684_684702


namespace inequality_solution_l684_684483

theorem inequality_solution (x : ℝ) : 
  (x < 2 ∨ x = 3) ↔ (x - 3) / ((x - 2) * (x - 3)) ≤ 0 := 
by {
  sorry
}

end inequality_solution_l684_684483


namespace Mr_Brown_selling_price_l684_684463

theorem Mr_Brown_selling_price 
  (initial_price : ℕ := 100000)
  (profit_percentage : ℚ := 10)
  (loss_percentage : ℚ := 10) :
  let profit := initial_price * (profit_percentage / 100),
      price_to_Brown := initial_price + profit,
      loss := price_to_Brown * (loss_percentage / 100),
      selling_price := price_to_Brown - loss
  in selling_price = 99000 := by
    sorry

end Mr_Brown_selling_price_l684_684463


namespace shaded_region_perimeter_of_tangent_circles_l684_684667

-- Conditions
def circumference (r : ℝ) := 2 * real.pi * r

def radius_of_circle_with_circumference (c : ℝ) := c / (2 * real.pi)

def half_circumference (c : ℝ) := c / 2

-- Question (theorem statement)
theorem shaded_region_perimeter_of_tangent_circles : 
  ∀ (r : ℝ), 
    circumference r = 24 → 
    4 * half_circumference (circumference r) = 24 :=
by
  intros r h
  -- This line states that the proof is admitted (i.e., not provided)
  sorry

end shaded_region_perimeter_of_tangent_circles_l684_684667


namespace units_digit_of_50_factorial_l684_684173

theorem units_digit_of_50_factorial : 
  ∃ d, (d = List.prod (List.range 1 51)) ∧ (d % 10 = 0) :=
by
  sorry

end units_digit_of_50_factorial_l684_684173


namespace units_digit_factorial_50_l684_684130

theorem units_digit_factorial_50 : Nat.unitsDigit (Nat.factorial 50) = 0 := 
  sorry

end units_digit_factorial_50_l684_684130


namespace average_people_per_hour_l684_684414

theorem average_people_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) (total_hours : ℕ) (average_per_hour : ℕ) :
  total_people = 3000 ∧ days = 5 ∧ hours_per_day = 24 ∧ total_hours = days * hours_per_day ∧ average_per_hour = total_people / total_hours → 
  average_per_hour = 25 :=
by
  sorry

end average_people_per_hour_l684_684414


namespace ratio_of_rectangular_paper_l684_684600

theorem ratio_of_rectangular_paper (x y : ℝ) (h : x > 0 ∧ y > 0)
  (cut_condition : ∀ A B C D : Point, rect_paper A B C D →
    (cut_start_vertex A D A B) →
    (angle_bisector_cut A D A B meets (rectangle_paper_other_edge C D) at (E point)) →
    (area_ratio (A B C E) (A D E) = 4)) :
  y / x = 5 / 2 :=
sorry

end ratio_of_rectangular_paper_l684_684600


namespace area_ratio_of_squares_eq_eight_l684_684948

theorem area_ratio_of_squares_eq_eight (s₁ : ℝ)
  (h₁ : ∀ c : ℝ, (s₁ ^ 2 = c)) -- given that the side length squared is the area of original square
  (h₂ : ∀ r, r = 2 * s₁ ∧ 4 * s₁ = r) -- The circle diameter is same as the perimeter of square and diameter equal to diagonal of other square
  (h₃: ∀ d : ℝ, d = 2 * sqrt 2 * s₁) -- the second square side length equated
  : 8 :=
  sorry

end area_ratio_of_squares_eq_eight_l684_684948


namespace number_of_possible_values_of_a_l684_684488

theorem number_of_possible_values_of_a :
  ∃ a_values : Finset ℕ, 
    (∀ a ∈ a_values, 5 ∣ a) ∧ 
    (∀ a ∈ a_values, a ∣ 30) ∧ 
    (∀ a ∈ a_values, 0 < a) ∧ 
    a_values.card = 4 :=
by
  sorry

end number_of_possible_values_of_a_l684_684488


namespace units_digit_50_factorial_l684_684159

theorem units_digit_50_factorial : (nat.factorial 50) % 10 = 0 :=
by
  sorry

end units_digit_50_factorial_l684_684159


namespace largest_angle_in_pentagon_l684_684770

-- Define the angles and sum condition
variables (x : ℝ) {P Q R S T : ℝ}

-- Conditions
def angle_P : P = 90 := sorry
def angle_Q : Q = 70 := sorry
def angle_R : R = x := sorry
def angle_S : S = x := sorry
def angle_T : T = 2*x + 20 := sorry
def sum_of_angles : P + Q + R + S + T = 540 := sorry

-- Prove the largest angle
theorem largest_angle_in_pentagon (hP : P = 90) (hQ : Q = 70)
    (hR : R = x) (hS : S = x) (hT : T = 2*x + 20) 
    (h_sum : P + Q + R + S + T = 540) : T = 200 :=
by
  sorry

end largest_angle_in_pentagon_l684_684770


namespace gray_region_area_l684_684537

theorem gray_region_area (d_small r_large r_small π : ℝ) (h1 : d_small = 6)
    (h2 : r_large = 3 * r_small) (h3 : r_small = d_small / 2) :
    (π * r_large ^ 2 - π * r_small ^ 2) = 72 * π := 
by
  -- The proof will be filled here
  sorry

end gray_region_area_l684_684537


namespace complex_number_equality_l684_684073

-- Define complex numbers
def complex_exp : ℂ := 2 * complex.I * (1 + complex.I) ^ 2

-- Define the target value
def target_value : ℂ := -4

-- Prove the equality
theorem complex_number_equality : complex_exp = target_value := by
  sorry  -- proof is omitted

end complex_number_equality_l684_684073


namespace serving_calculation_correct_l684_684608

def prepared_orange_juice_servings (cans_of_concentrate : ℕ) 
                                  (oz_per_concentrate_can : ℕ) 
                                  (water_ratio : ℕ) 
                                  (oz_per_serving : ℕ) : ℕ :=
  let total_concentrate := cans_of_concentrate * oz_per_concentrate_can
  let total_water := cans_of_concentrate * water_ratio * oz_per_concentrate_can
  let total_juice := total_concentrate + total_water
  total_juice / oz_per_serving

theorem serving_calculation_correct :
  prepared_orange_juice_servings 60 5 3 6 = 200 := by
  sorry

end serving_calculation_correct_l684_684608


namespace largest_integer_with_4_digit_square_in_base_7_l684_684787

theorem largest_integer_with_4_digit_square_in_base_7 (M : ℕ) :
  (∀ m : ℕ, m < 240 ∧ 49 ≤ m → m ≤ 239) ∧ nat.to_digits 7 239 = [4, 6, 1] :=
begin
  sorry
end

end largest_integer_with_4_digit_square_in_base_7_l684_684787


namespace men_in_first_group_l684_684206

theorem men_in_first_group (M : ℕ) :
  (16 * 30) * (1 : ℚ) / 320 = (15 * 1) * M / 40 → M = 4 :=
by
  intro condition
  have h1 : 16 * 30 = 480 := by norm_num
  have h2 : 320 = 320 := by norm_num
  have h3 : 480 / 320 = 3 / 2 := by norm_num
  have h4 : (15 * 1) * M / 40 = 15 * M / 40 := by norm_num
  rw [h3, h4] at condition
  have h5 : 15 * M / 40 = 3 / 2 → M = 4 := sorry
  apply h5
  assumption

end men_in_first_group_l684_684206


namespace units_digit_factorial_50_l684_684131

theorem units_digit_factorial_50 : Nat.unitsDigit (Nat.factorial 50) = 0 := 
  sorry

end units_digit_factorial_50_l684_684131


namespace determine_moles_Al2O3_formed_l684_684634

noncomputable def initial_moles_Al : ℝ := 10
noncomputable def initial_moles_Fe2O3 : ℝ := 6
noncomputable def balanced_eq (moles_Al moles_Fe2O3 moles_Al2O3 moles_Fe : ℝ) : Prop :=
  2 * moles_Al + moles_Fe2O3 = moles_Al2O3 + 2 * moles_Fe

theorem determine_moles_Al2O3_formed :
  ∃ moles_Al2O3 : ℝ, balanced_eq 10 6 moles_Al2O3 (moles_Al2O3 * 2) ∧ moles_Al2O3 = 5 := 
  by 
  sorry

end determine_moles_Al2O3_formed_l684_684634


namespace monthly_income_of_labourer_l684_684069

variable (I : ℕ) -- Monthly income

-- Conditions: 
def condition1 := (85 * 6) - (6 * I) -- A boolean expression depicting the labourer fell into debt
def condition2 := (60 * 4) + (85 * 6 - 6 * I) + 30 -- Total income covers debt and saving 30

-- Statement to be proven
theorem monthly_income_of_labourer : 
  ∃ I : ℕ, condition1 I = 0 ∧ condition2 I = 4 * I → I = 78 :=
by
  sorry

end monthly_income_of_labourer_l684_684069


namespace atomic_weight_O_l684_684650

-- We define the atomic weights of sodium and chlorine
def atomic_weight_Na : ℝ := 22.99
def atomic_weight_Cl : ℝ := 35.45

-- We define the molecular weight of the compound
def molecular_weight_compound : ℝ := 74.0

-- We want to prove that the atomic weight of oxygen (O) is 15.56 given the above conditions
theorem atomic_weight_O : 
  (molecular_weight_compound = atomic_weight_Na + atomic_weight_Cl + w -> w = 15.56) :=
by
  sorry

end atomic_weight_O_l684_684650


namespace no_rational_xyz_satisfies_l684_684643

theorem no_rational_xyz_satisfies:
  ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
  (1 / (x - y) ^ 2 + 1 / (y - z) ^ 2 + 1 / (z - x) ^ 2 = 2014) :=
by
  -- The proof will go here
  sorry

end no_rational_xyz_satisfies_l684_684643


namespace seating_arrangements_l684_684768

def total_seating (n : ℕ) : ℕ :=
  n!

def group_seating (n_super_person : ℕ) (n_remaining : ℕ) : ℕ :=
  (n_super_person + 1)! * n_remaining!

noncomputable def acceptable_seating : ℕ :=
  total_seating 10 - (group_seating 1 6 * 4!)

theorem seating_arrangements :
  acceptable_seating = 3507840 :=
by
  sorry

end seating_arrangements_l684_684768


namespace problem1_problem2_problem3_problem4_l684_684844

-- Problem 1
theorem problem1 (x : ℤ) (h : 4 * x = 20) : x = 5 :=
sorry

-- Problem 2
theorem problem2 (x : ℤ) (h : x - 18 = 40) : x = 58 :=
sorry

-- Problem 3
theorem problem3 (x : ℤ) (h : x / 7 = 12) : x = 84 :=
sorry

-- Problem 4
theorem problem4 (n : ℚ) (h : 8 * n / 2 = 15) : n = 15 / 4 :=
sorry

end problem1_problem2_problem3_problem4_l684_684844


namespace Leah_age_is_41_l684_684484

def LeahAgeProof : Prop :=
  ∃ x ∈ {26, 29, 31, 33, 35, 39, 41, 43, 45, 50} |  -- Leah's possible ages.
    (prime x) ∧                                  -- Leah's age is a prime number.
    (∃ a b c, a + 1 = x ∧ b - 1 = x ∧ c - 1 = x ∧ -- There are three ages off by exactly one.
      a ∈ {26, 29, 31, 33, 35, 39, 41, 43, 45, 50} ∧ 
      b ∈ {26, 29, 31, 33, 35, 39, 41, 43, 45, 50} ∧ 
      c ∈ {26, 29, 31, 33, 35, 39, 41, 43, 45, 50}) ∧ 
    (∃ n, n = {y ∈ {26, 29, 31, 33, 35, 39, 41, 43, 45, 50} | y < x}.card ∧ 
      n > {26, 29, 31, 33, 35, 39, 41, 43, 45, 50}.card / 2) -- More than half the guesses are too low.

theorem Leah_age_is_41 : LeahAgeProof := sorry

end Leah_age_is_41_l684_684484


namespace daily_renovation_length_additional_renovation_length_required_additional_daily_length_l684_684113

open Real

-- Definitions based on given conditions
def totalLength := 3600 -- meters
def increasedEfficiency := 1.2 -- 20% higher efficiency
def daysSaved := 10 -- days saved due to increased efficiency
def initialConstructionDays := totalLength / (totalLength / initialDailyLength) - daysSaved

-- Main theorem statements to prove
theorem daily_renovation_length (initialDailyLength : ℝ) (actualDailyLength : ℝ) : 
  initialDailyLength = 60 → actualDailyLength = 72 := by
  -- The proof should go here.
  sorry

theorem additional_renovation_length_required (additionalDailyLength : ℝ) :
  additionalDailyLength ≥ 36 := by
  -- The proof should go here.
  sorry

-- Defining the initial and actual daily renovation lengths from the conditions
def initialDailyLength := (3600 / (3600 / x - 10))
def actualDailyLength := initialDailyLength * 1.2

-- Defining inequality for the additional renovation length
def additionalRenovationLength (remainingDays : ℕ) :=
  let workDone := 72 * 20
  let remainingWork := totalLength - workDone
  let remainingDailyLength := remainingWork / remainingDays
  remainingDailyLength - 72

-- Main theorem proving the required additional daily length
theorem additional_daily_length (remainingDays : ℕ) (remainingDailyLength : ℝ) :
  additionalRenovationLength remainingDays = remainingDailyLength →
  remainingDailyLength ≥ 36 := by
  have : remainingDays = 20, from sorry
  -- Proof should go here.
  sorry

end daily_renovation_length_additional_renovation_length_required_additional_daily_length_l684_684113


namespace soap_remaining_days_l684_684202

theorem soap_remaining_days 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0)
  (daily_consumption : ℝ)
  (h4 : daily_consumption = a * b * c / 8) 
  (h5 : ∀ t : ℝ, t > 0 → t ≤ 7 → daily_consumption = (a * b * c - (a * b * c) * (1 / 8))) :
  ∃ t : ℝ, t = 1 :=
by 
  sorry

end soap_remaining_days_l684_684202


namespace community_cleaning_children_l684_684872

theorem community_cleaning_children (total_members adult_men_ratio adult_women_ratio : ℕ) 
(h_total : total_members = 2000)
(h_men_ratio : adult_men_ratio = 30) 
(h_women_ratio : adult_women_ratio = 2) :
  (total_members - (adult_men_ratio * total_members / 100 + 
  adult_women_ratio * (adult_men_ratio * total_members / 100))) = 200 :=
by
  sorry

end community_cleaning_children_l684_684872


namespace number_of_whole_numbers_between_200_and_500_contain_digit_3_l684_684723

open Set

def contains_digit_3 (n : ℕ) : Prop := 
  let digits := [n / 100, (n / 10) % 10, n % 10]
  3 ∈ digits

def whole_numbers_between_200_and_500_contain_3 : ℕ := 
  Finset.card (Finset.filter contains_digit_3 (Finset.Ico 200 500))

theorem number_of_whole_numbers_between_200_and_500_contain_digit_3 :
  whole_numbers_between_200_and_500_contain_3 = 138 := sorry

end number_of_whole_numbers_between_200_and_500_contain_digit_3_l684_684723


namespace coin_flip_probability_l684_684954

theorem coin_flip_probability :
  let total_flips := 8
  let num_heads := 6
  let total_outcomes := (2: ℝ) ^ total_flips
  let favorable_outcomes := (Nat.choose total_flips num_heads)
  let probability := favorable_outcomes / total_outcomes
  probability = (7 / 64 : ℝ) :=
by
  sorry

end coin_flip_probability_l684_684954


namespace max_min_sum_difference_l684_684033

-- The statement that we need to prove
theorem max_min_sum_difference : 
  ∃ (max_sum min_sum: ℕ), (∀ (RST UVW XYZ : ℕ),
   -- Constraints for Max's and Minnie's sums respectively
   (RST = 100 * 9 + 10 * 6 + 3 ∧ UVW = 100 * 8 + 10 * 5 + 2 ∧ XYZ = 100 * 7 + 10 * 4 + 1 → max_sum = 2556) ∧ 
   (RST = 100 * 1 + 10 * 0 + 6 ∧ UVW = 100 * 2 + 10 * 4 + 7 ∧ XYZ = 100 * 3 + 10 * 5 + 8 → min_sum = 711)) → 
    max_sum - min_sum = 1845 :=
by
  sorry

end max_min_sum_difference_l684_684033


namespace inequality_solution_system_of_inequalities_solution_l684_684846

noncomputable section

-- First problem
theorem inequality_solution (x : ℝ) : 
  (3 - x) / (5 + 2 * x) ≤ 0 → x < -5 / 2 ∨ x ≥ 3 :=
begin
  sorry
end

-- Second problem
theorem system_of_inequalities_solution (x : ℝ) : 
  (1 / x ≤ x) ∧ (x^2 - 3 * x < 0) → (1 ≤ x ∧ x < 3) :=
begin
  sorry
end

end inequality_solution_system_of_inequalities_solution_l684_684846


namespace mike_interest_earned_l684_684460

theorem mike_interest_earned
  (total_invested : ℝ) (investment_9_percent : ℝ) (investment_11_percent : ℝ)
  (interest_rate_9 : ℝ) (interest_rate_11 : ℝ)
  (total_invested_eq : total_invested = 6000)
  (investment_9_percent_eq : investment_9_percent = 1800)
  (investment_11_percent_eq : investment_11_percent = total_invested - investment_9_percent)
  (interest_rate_9_eq : interest_rate_9 = 0.09)
  (interest_rate_11_eq : interest_rate_11 = 0.11) :
  (investment_9_percent * interest_rate_9 + investment_11_percent * interest_rate_11) = 624 :=
by
  sorry

end mike_interest_earned_l684_684460


namespace compute_expression_l684_684013

noncomputable theory
open Complex

theorem compute_expression (ω : ℂ) (hω : ω ^ 3 = 1) (h_nonreal : ω.im ≠ 0) :
  let a := 1 - 2 * ω + 3 * ω ^ 2
      b := 1 + 3 * ω - 2 * ω ^ 2 in
  (a ^ 4 + b ^ 4) = 9375 * ω + 2722 :=
by
  have h1 : ω ^ 2 + ω + 1 = 0 := sorry
  have a_eq : a = -5 * ω - 2 := sorry
  have b_eq : b = 5 * ω + 3 := sorry
  calc
    (a ^ 4 + b ^ 4)
        = (-5 * ω - 2) ^ 4 + (5 * ω + 3) ^ 4 : by rw [a_eq, b_eq]
    ... = 9375 * ω + 2722 : sorry

end compute_expression_l684_684013


namespace remainder_1509_mod_1000_l684_684633

/--
Given an increasing sequence of positive integers \( b_1 \le b_2 \le \cdots \le b_{15} \le 3005 \)
such that \( b_i - i \) is odd for \( 1 \le i \le 15 \),
prove that the number of ways to choose these integers can be expressed as \({1509 \choose 15}\)
and the remainder of \( 1509 \) when divided by \( 1000 \) is \( 509 \).
-/
theorem remainder_1509_mod_1000 :
  let p := 1509 in
  p % 1000 = 509 :=
by
  sorry

end remainder_1509_mod_1000_l684_684633


namespace total_profit_percentage_l684_684602

def shopkeeper_apples : ℝ := 100
def first_half_profit_percentage : ℝ := 25
def second_half_profit_percentage : ℝ := 30

theorem total_profit_percentage :
  let cost_price_per_kg := 1 in
  let total_cost_price := shopkeeper_apples * cost_price_per_kg in
  let first_half_cost := (shopkeeper_apples / 2) * cost_price_per_kg in
  let first_half_profit := first_half_cost * (first_half_profit_percentage / 100) in
  let second_half_cost := (shopkeeper_apples / 2) * cost_price_per_kg in
  let second_half_profit := second_half_cost * (second_half_profit_percentage / 100) in
  let total_profit := first_half_profit + second_half_profit in
  (total_profit / total_cost_price) * 100 = 27.5 :=
by
  sorry

end total_profit_percentage_l684_684602


namespace gray_area_correct_l684_684539

-- Define the conditions of the problem
def diameter_small : ℝ := 6
def radius_small : ℝ := diameter_small / 2
def radius_large : ℝ := 3 * radius_small

-- Define the areas based on the conditions
def area_small : ℝ := Real.pi * radius_small^2
def area_large : ℝ := Real.pi * radius_large^2
def gray_area : ℝ := area_large - area_small

-- Write the theorem that proves the required area of the gray region
theorem gray_area_correct : gray_area = 72 * Real.pi :=
by
  sorry

end gray_area_correct_l684_684539


namespace find_pirates_l684_684224

def pirate_problem (p : ℕ) (nonfighters : ℕ) (arm_loss_percent both_loss_percent : ℝ) (leg_loss_fraction : ℚ) : Prop :=
  let participants := p - nonfighters in
  let arm_loss := arm_loss_percent * participants in
  let both_loss := both_loss_percent * participants in
  let leg_loss (p : ℕ) := leg_loss_fraction * p in
  let only_leg_loss (p : ℕ) := leg_loss p - both_loss in
  arm_loss + only_leg_loss p + both_loss = leg_loss p

theorem find_pirates : ∃ p : ℕ, pirate_problem p 10 0.54 0.34 (2/3) :=
sorry

end find_pirates_l684_684224


namespace roots_relationship_l684_684518

theorem roots_relationship (p q : ℝ) : 
  (∀ x1 x2 : ℝ, x1 + x2 = -p ∧ x1 * x2 = q → 
    (((x1 + 1) + (x2 + 1) = x1 + x2 + 2) = -p^2) ∧ 
    (((x1 + 1) * (x2 + 1) = (x1 * x2 + (x1 + x2) + 1) = p * q))) →
  (p = 2 ∧ q = -1) ∨ (p = -1 ∧ q = -1) :=
sorry

end roots_relationship_l684_684518


namespace fixed_point_of_linear_function_l684_684326

theorem fixed_point_of_linear_function (k b : ℝ) (h : 3*k - b = 2) : ( -3, -2 ) ∈ (λ x, (k * x + b) : ℝ → ℝ) :=
by
  -- Proof omitted
  sorry

end fixed_point_of_linear_function_l684_684326


namespace malcolm_initial_white_lights_l684_684084

theorem malcolm_initial_white_lights :
  ∀ (red blue green remaining total_initial : ℕ),
    red = 12 →
    blue = 3 * red →
    green = 6 →
    remaining = 5 →
    total_initial = red + blue + green + remaining →
    total_initial = 59 :=
by
  intros red blue green remaining total_initial h1 h2 h3 h4 h5
  -- Add details if necessary for illustration
  -- sorry typically as per instructions
  sorry

end malcolm_initial_white_lights_l684_684084


namespace sally_total_cards_l684_684837

theorem sally_total_cards (initial_cards : ℕ) (dan_cards : ℕ) (bought_cards : ℕ) :
  initial_cards = 27 →
  dan_cards = 41 →
  bought_cards = 20 →
  initial_cards + dan_cards + bought_cards = 88 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end sally_total_cards_l684_684837


namespace difference_in_volume_l684_684118

-- Definitions
def edge_length : ℝ := 1
def small_sphere_radius : ℝ := 1/2
def volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3
def large_sphere_radius : ℝ := (Real.sqrt 3 + 1) / 2
def small_sphere_radius_internal : ℝ := (Real.sqrt 3 - 1) / 2

-- Lean 4 Statement
theorem difference_in_volume : 
  volume large_sphere_radius - volume small_sphere_radius_internal = (10 / 3) * Real.pi := 
  sorry

end difference_in_volume_l684_684118


namespace real_part_of_z_l684_684692

noncomputable def complex_multiply (z : ℂ) : ℂ := (1 - complex.I) * z
noncomputable def complex_norm : ℂ := complex.abs (3 - 4 * complex.I)

theorem real_part_of_z (z : ℂ) (hz : complex_multiply z = complex_norm) :
  z.re = 5 / 2 :=
by
  -- Problem conditions and proof go here
  sorry

end real_part_of_z_l684_684692


namespace apple_sale_discrepancy_l684_684116

theorem apple_sale_discrepancy
  (vendor1_apples : ℕ)
  (vendor1_price_per_three : ℕ)
  (vendor2_apples : ℕ)
  (vendor2_price_per_two : ℕ)
  (total_apples : ℕ)
  (friend_price_per_five : ℕ)
  (individual_revenue : ℕ)
  (friend_revenue : ℕ)
  (missing_cent : ℕ) :
  vendor1_apples = 30 →
  vendor1_price_per_three = 1 →
  vendor2_apples = 30 →
  vendor2_price_per_two = 1 →
  total_apples = 60 →
  friend_price_per_five = 2 →
  individual_revenue = 25 →
  friend_revenue = 24 →
  missing_cent = individual_revenue - friend_revenue :=
begin
  sorry
end

end apple_sale_discrepancy_l684_684116


namespace minimum_small_bottles_l684_684238

-- Define the capacities of the bottles
def small_bottle_capacity : ℕ := 35
def large_bottle_capacity : ℕ := 500

-- Define the number of small bottles needed to fill a large bottle
def small_bottles_needed_to_fill_large : ℕ := 
  (large_bottle_capacity + small_bottle_capacity - 1) / small_bottle_capacity

-- Statement of the theorem
theorem minimum_small_bottles : small_bottles_needed_to_fill_large = 15 := by
  sorry

end minimum_small_bottles_l684_684238


namespace positive_pairs_solution_l684_684803

noncomputable def is_factor (n : ℕ) (a b : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + a*x + b = 0 → a * x^(2 * n) + (a * x + b)^(2 * n) = 0

theorem positive_pairs_solution (n : ℕ) (a b : ℝ) (k : ℕ) (h : n ≥ 2) :
  let α := 2 * real.cos ((2 * k + 1) * real.pi / (2 * n))
  n < 2 * k + 1 ∧ 2 * k + 1 < 3 * n →
  a = α^(2 * n / (2 * n - 1)) ∧ b = α^(2 / (2 * n - 1)) →
  a > 0 ∧ b > 0 →
  is_factor n a b :=
by
  intros α h_range h_vals h_pos
  sorry

end positive_pairs_solution_l684_684803


namespace units_digit_of_50_factorial_l684_684178

theorem units_digit_of_50_factorial : 
  ∃ d, (d = List.prod (List.range 1 51)) ∧ (d % 10 = 0) :=
by
  sorry

end units_digit_of_50_factorial_l684_684178


namespace find_pirates_l684_684225

def pirate_problem (p : ℕ) (nonfighters : ℕ) (arm_loss_percent both_loss_percent : ℝ) (leg_loss_fraction : ℚ) : Prop :=
  let participants := p - nonfighters in
  let arm_loss := arm_loss_percent * participants in
  let both_loss := both_loss_percent * participants in
  let leg_loss (p : ℕ) := leg_loss_fraction * p in
  let only_leg_loss (p : ℕ) := leg_loss p - both_loss in
  arm_loss + only_leg_loss p + both_loss = leg_loss p

theorem find_pirates : ∃ p : ℕ, pirate_problem p 10 0.54 0.34 (2/3) :=
sorry

end find_pirates_l684_684225


namespace tan_alpha_eq_3_expression_value_l684_684318

variable (α : ℝ)

-- Condition
def cond : Prop := (sin α + cos α) / (sin α - cos α) = 2

-- Proof statement 1: tan α = 3
theorem tan_alpha_eq_3 (h : cond α) : tan α = 3 := sorry

-- Proof statement 2: sin^2 α - 2sin α cos α + 1 = 13/10
theorem expression_value (h : cond α) : sin α ^ 2 - 2 * sin α * cos α + 1 = 13 / 10 := sorry

end tan_alpha_eq_3_expression_value_l684_684318


namespace find_max_m_l684_684015

-- We define real numbers a, b, c that satisfy the given conditions
variable (a b c m : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum : a + b + c = 12)
variable (h_prod_sum : a * b + b * c + c * a = 30)
variable (m_def : m = min (a * b) (min (b * c) (c * a)))

-- We state the main theorem to be proved
theorem find_max_m : m ≤ 2 :=
by
  sorry

end find_max_m_l684_684015


namespace find_a_in_terms_of_y_l684_684727

theorem find_a_in_terms_of_y (a b y : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * y^3) (h3 : a - b = 3 * y) :
  a = 3 * y :=
sorry

end find_a_in_terms_of_y_l684_684727


namespace probability_at_least_one_defective_probability_at_most_one_defective_l684_684417

noncomputable def machine_a_defect_rate : ℝ := 0.05
noncomputable def machine_b_defect_rate : ℝ := 0.1

/-- 
Prove the probability that there is at least one defective part among the two parts
given the defect rates of machine A and machine B
--/
theorem probability_at_least_one_defective (pA pB : ℝ) (hA : pA = machine_a_defect_rate) (hB : pB = machine_b_defect_rate) : 
  (1 - (1 - pA) * (1 - pB)) = 0.145 :=
  sorry

/-- 
Prove the probability that there is at most one defective part among the two parts
given the defect rates of machine A and machine B
--/
theorem probability_at_most_one_defective (pA pB : ℝ) (hA : pA = machine_a_defect_rate) (hB : pB = machine_b_defect_rate) : 
  (1 - pA * pB) = 0.995 :=
  sorry

end probability_at_least_one_defective_probability_at_most_one_defective_l684_684417


namespace part1_part2_l684_684687

-- Given conditions and definitions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x
def satisfies_equation (f g : ℝ → ℝ) := ∀ x, f x + g x = 2 * Real.log2 (1 - x)

-- Definitions based on the problem statement and solution
def f (x : ℝ) : ℝ := Real.log2 ((1 - x) / (1 + x))
def g (x : ℝ) : ℝ := Real.log2 ((1 - x) ^ 2)

-- Part 1: Prove the analytical expressions for f(x) and g(x)
theorem part1 (f g : ℝ → ℝ) (h1 : is_odd f) (h2 : is_even g) (h3 : satisfies_equation f g) :
  f = λ x, Real.log2 ((1 - x) / (1 + x)) ∧ g = λ x, Real.log2 ((1 - x) ^ 2) :=
by
  sorry

-- Part 2: Prove the range of real number values for m such that f(2^x) = m has a solution
theorem part2 (m : ℝ) :
  (∃ x : ℝ, f (2 ^ x) = m) ↔ m < 0 :=
by
  sorry

end part1_part2_l684_684687


namespace knocks_to_knicks_l684_684742

variable (knicks knacks knocks : ℝ)

def knicks_eq_knacks : Prop := 
  8 * knicks = 3 * knacks

def knacks_eq_knocks : Prop := 
  4 * knacks = 5 * knocks

theorem knocks_to_knicks
  (h1 : knicks_eq_knacks knicks knacks)
  (h2 : knacks_eq_knocks knacks knocks) :
  20 * knocks = 320 / 15 * knicks :=
  sorry

end knocks_to_knicks_l684_684742


namespace subset_sum_divisible_by_2008_l684_684055

theorem subset_sum_divisible_by_2008 (a : Fin 2008 → ℤ) :
  ∃ (s : Finset (Fin 2008)), (∑ i in s, a i) % 2008 = 0 :=
by
  sorry

end subset_sum_divisible_by_2008_l684_684055


namespace work_completion_time_l684_684563

theorem work_completion_time 
(w : ℝ)  -- total amount of work
(A B : ℝ)  -- work rate of a and b per day
(h1 : A + B = w / 30)  -- combined work rate
(h2 : 20 * (A + B) + 20 * A = w) : 
  (1 / A = 60) :=
sorry

end work_completion_time_l684_684563


namespace chocolate_bar_min_breaks_l684_684561

theorem chocolate_bar_min_breaks (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  ∃ k, k = m * n - 1 := by
  sorry

end chocolate_bar_min_breaks_l684_684561


namespace solve_differential_eq_l684_684651

noncomputable def particular_solution (x : ℝ) : ℝ :=
  x^3 / 3 - x + 14 / 3

theorem solve_differential_eq :
  ∀ x, (differential_eq : ∀ y, y' = x^2 - 1) 
    → particular_solution(1) = 4 
    → y(particular_solution x) = particular_solution(x) :=
by sorry

end solve_differential_eq_l684_684651


namespace repeating_decimal_sum_l684_684628

theorem repeating_decimal_sum :
  (0.6666666666 : ℝ) + (0.7777777777 : ℝ) = (13 : ℚ) / 9 := by
  sorry

end repeating_decimal_sum_l684_684628


namespace ratio_lateral_surface_areas_l684_684325

-- Definitions and assumptions
variables (r h : ℝ)
-- There is a cone and a cylinder with equal base radius (r) and height (h)
-- The axis section of the cone is an equilateral triangle
def is_equilateral_triangle_axis_section (r : ℝ) : Prop :=
  ∀ (l : ℝ), l = 2 * r -- slant height is 2 * radius

-- Lateral surface area calculations (for the proof, this is inferred beforehand)
def lateral_surface_area_cone (r : ℝ) : ℝ := 
  2 * π * r^2

def lateral_surface_area_cylinder(r h : ℝ) : ℝ := 
  2 * π * r * (r * sqrt 3)

-- Prove the ratio of the lateral surface areas equals sqrt(3)/3
theorem ratio_lateral_surface_areas (r : ℝ) (h : ℝ) :
  is_equilateral_triangle_axis_section r → h = r * sqrt 3 →
  (lateral_surface_area_cone r / lateral_surface_area_cylinder r h) = sqrt 3 / 3 :=
by
  intro h1 h2
  sorry

end ratio_lateral_surface_areas_l684_684325


namespace S9_equals_27_l684_684694

variable {a : ℕ → ℤ} -- sequence a_n

-- conditions
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ (a₁ d : ℤ), ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), a i

theorem S9_equals_27 (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_condition : a 1 = 3 * a 3 - 6) :
  sum_of_first_n_terms a 8 = 27 :=
by
  sorry

end S9_equals_27_l684_684694


namespace total_cans_collected_l684_684194

-- Definitions based on conditions
def bags_on_saturday : ℕ := 6
def bags_on_sunday : ℕ := 3
def cans_per_bag : ℕ := 8

-- The theorem statement
theorem total_cans_collected : bags_on_saturday + bags_on_sunday * cans_per_bag = 72 :=
by
  sorry

end total_cans_collected_l684_684194


namespace min_value_at_2_l684_684503

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_at_2 : ∃ x : ℝ, f x = 2 :=
sorry

end min_value_at_2_l684_684503


namespace max_x_value_l684_684843

noncomputable def max_x : ℝ :=
  let p := -5
  let q := 5
  let r := 66
  let s := 7
  in (p + q * Real.sqrt r) / s

theorem max_x_value :
  ∃ p q r s,
    (p + q * Real.sqrt r) / s = max_x ∧
    (7 * max_x / 5 + 2 = 4 / max_x) ∧
    (p, q, r, s ∈ Int) ∧
    (prs := p * r * s) ∧
    (frac_prs_q := prs / q) ∧
    (frac_prs_q = -462) :=
by
  sorry

end max_x_value_l684_684843


namespace bulbs_still_on_after_toggling_l684_684107

theorem bulbs_still_on_after_toggling :
  { n | n ∈ finset.range 101 ∧ (∃ k: ℕ, k ∈ finset.range 11 ∧ n = k^2) } = 
  { 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 } :=
by
  sorry

end bulbs_still_on_after_toggling_l684_684107


namespace volume_of_T_eq_32_div_3_l684_684103

section
variable {R : Type*} [LinearOrderedField R]

def T (x y z : R) : Prop :=
  abs x + abs y ≤ 2 ∧ abs x + abs z ≤ 2 ∧ abs y + abs z ≤ 2

theorem volume_of_T_eq_32_div_3 :
  let volume : R := 32/3
  in ∀ R : Type*, LinearOrderedField R → volume = 32 / 3 :=
by
  sorry
end

end volume_of_T_eq_32_div_3_l684_684103


namespace community_cleaning_children_l684_684871

theorem community_cleaning_children (total_members adult_men_ratio adult_women_ratio : ℕ) 
(h_total : total_members = 2000)
(h_men_ratio : adult_men_ratio = 30) 
(h_women_ratio : adult_women_ratio = 2) :
  (total_members - (adult_men_ratio * total_members / 100 + 
  adult_women_ratio * (adult_men_ratio * total_members / 100))) = 200 :=
by
  sorry

end community_cleaning_children_l684_684871


namespace gg_of_3_is_107_l684_684728

-- Define the function g
def g (x : ℕ) : ℕ := 3 * x + 2

-- State that g(g(g(3))) equals 107
theorem gg_of_3_is_107 : g (g (g 3)) = 107 := by
  sorry

end gg_of_3_is_107_l684_684728


namespace find_train_speed_l684_684565

-- Define the given conditions
def train_length : ℕ := 2500  -- length of the train in meters
def time_to_cross_pole : ℕ := 100  -- time to cross the pole in seconds

-- Define the expected speed
def expected_speed : ℕ := 25  -- expected speed in meters per second

-- The theorem we need to prove
theorem find_train_speed : 
  (train_length / time_to_cross_pole) = expected_speed := 
by 
  sorry

end find_train_speed_l684_684565


namespace subsets_exist_l684_684449

theorem subsets_exist (n : Fin 2000 → ℕ) (h : ∀ i j : Fin 2000, i < j → n i < n j)
  (h_bound : n 1999 < 10^100) :
  ∃ (A B : Fin 2000 → Bool), disjoint A B ∧ ∑ i, ite (A i) (n i) 0 = ∑ i, ite (B i) (n i) 0 ∧
  (∑ i, ite (A i) ((n i)^2) 0) = ∑ i, ite (B i) ((n i)^2) 0 :=
sorry

end subsets_exist_l684_684449


namespace units_digit_of_50_factorial_l684_684139

theorem units_digit_of_50_factorial : (nat.factorial 50) % 10 = 0 := 
by 
  sorry

end units_digit_of_50_factorial_l684_684139


namespace solve_quadratic_1_solve_quadratic_2_l684_684845

-- 1. Prove that the solutions to the equation x^2 - 4x - 1 = 0 are x = 2 + sqrt(5) and x = 2 - sqrt(5)
theorem solve_quadratic_1 (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
sorry

-- 2. Prove that the solutions to the equation 3(x - 1)^2 = 2(x - 1) are x = 1 and x = 5/3
theorem solve_quadratic_2 (x : ℝ) : 3 * (x - 1) ^ 2 = 2 * (x - 1) ↔ x = 1 ∨ x = 5 / 3 :=
sorry

end solve_quadratic_1_solve_quadratic_2_l684_684845


namespace sound_propagation_in_all_directions_l684_684403

noncomputable theory

-- Definitions based on the conditions
def is_mechanical_wave (s : Type) : Prop :=
  -- Sound is a mechanical wave
  ∃ (θ : s → Prop), ∀ (x : s), θ x

def travels_through_medium (s : Type) : Prop :=
  -- Sound travels through mediums like air, water, or solids
  ∃ (θ : s → Prop), ∀ (x : s), θ x

def causes_particles_to_vibrate (s : Type) : Prop :=
  -- Sound causes particles of the medium to vibrate
  ∃ (θ : s → Prop), ∀ (x : s), θ x

def cannot_travel_through_vacuum (s : Type) : Prop :=
  -- Sound cannot travel through a vacuum
  ∃ (θ : s → Prop), ∀ (x : s), ¬θ x

def can_reflect_off_surfaces (s : Type) : Prop :=
  -- Sound can reflect off surfaces
  ∃ (θ : s → Prop), ∀ (x : s), θ x

-- The statement to prove that sound propagates in all directions
theorem sound_propagation_in_all_directions (s : Type) :
  is_mechanical_wave s →
  travels_through_medium s →
  causes_particles_to_vibrate s →
  cannot_travel_through_vacuum s →
  can_reflect_off_surfaces s →
  ∀ (x : s), True :=
by
  intros _ _ _ _ _
  triv

end sound_propagation_in_all_directions_l684_684403


namespace sqrt_two_not_in_A_l684_684810

def A := {x : ℚ | x > -1}

theorem sqrt_two_not_in_A : ¬(real.sqrt 2 ∈ A) :=
sorry

end sqrt_two_not_in_A_l684_684810


namespace find_A_l684_684934

def alpha : ℝ := Real.arcsin (3/5)
def beta : ℝ := Real.arcsin (4/5)

def A : ℝ :=
  (Real.cos (3 * Real.pi / 2 - alpha / 2))^6 
  - (Real.cos (5 * Real.pi / 2 + beta / 2))^6

theorem find_A : A / 3.444 = -0.007 := by
  sorry

end find_A_l684_684934


namespace bearWeightGainRatio_l684_684943

variables (B : ℝ) (acorns smallAnimals totalWeight : ℝ)

def bearWeightGain := ∃ B : ℝ, 
  let acorns := 2 * B,
      smallAnimals := 200,
      totalWeight := 1000 in
  let remainingWeight := totalWeight - (B + acorns + smallAnimals),
      salmon := remainingWeight / 2 in
  B + acorns + smallAnimals + salmon = 1000 ∧ (B / totalWeight = 1 / 3.75)

theorem bearWeightGainRatio : bearWeightGain :=
sorry

end bearWeightGainRatio_l684_684943


namespace probability_of_6_heads_in_8_flips_l684_684952

theorem probability_of_6_heads_in_8_flips :
  let n : ℕ := 8
  let k : ℕ := 6
  let total_outcomes := 2 ^ n
  let successful_outcomes := Nat.choose n k
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 7 / 64 := by
  sorry

end probability_of_6_heads_in_8_flips_l684_684952


namespace papers_on_saturday_equals_45_l684_684472

-- Define the conditions
def total_papers_on_weekend : ℕ := 110
def papers_delivered_difference : ℕ := 20

-- The number of papers Peter delivers on Saturday
def papers_on_saturday (S : ℕ) : Prop :=
  (S + (S + papers_delivered_difference)) = total_papers_on_weekend

-- The theorem to prove 
theorem papers_on_saturday_equals_45 : ∃ (S : ℕ), papers_on_saturday S ∧ S = 45 :=
begin
  sorry
end

end papers_on_saturday_equals_45_l684_684472


namespace not_equal_to_seven_sixths_l684_684926

theorem not_equal_to_seven_sixths : 
  (\frac{14}{12} = \frac{7}{6}) ∧ 
  (1 + \frac{1}{6} = \frac{7}{6}) ∧ 
  (1 + \frac{2}{12} = \frac{7}{6}) ∧ 
  (1 + \frac{3}{18} = \frac{7}{6}) ∧ 
  ¬ (1 + \frac{3}{5} = \frac{7}{6}) :=
by
  sorry

end not_equal_to_seven_sixths_l684_684926


namespace equal_piles_l684_684497

theorem equal_piles (initial_rocks final_piles : ℕ) (moves : ℕ) (total_rocks : ℕ) (rocks_per_pile : ℕ) :
  initial_rocks = 36 →
  final_piles = 7 →
  moves = final_piles - 1 →
  total_rocks = initial_rocks + moves →
  rocks_per_pile = total_rocks / final_piles →
  rocks_per_pile = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end equal_piles_l684_684497


namespace combined_height_of_rockets_l684_684003

noncomputable def height_of_rocket (a t : ℝ) : ℝ := (1/2) * a * t^2

theorem combined_height_of_rockets
  (h_A_ft : ℝ)
  (fuel_type_B_coeff : ℝ)
  (g : ℝ)
  (ft_to_m : ℝ)
  (h_combined : ℝ) :
  h_A_ft = 850 →
  fuel_type_B_coeff = 1.7 →
  g = 9.81 →
  ft_to_m = 0.3048 →
  h_combined = 348.96 :=
by sorry

end combined_height_of_rockets_l684_684003


namespace derivative_of_y_l684_684647

noncomputable def y (x : ℝ) : ℝ := Real.sin x + Real.exp x * Real.cos x

theorem derivative_of_y (x : ℝ) : derivative (λ x, y x) x = (1 + Real.exp x) * Real.cos x - Real.exp x * Real.sin x := by
  sorry

end derivative_of_y_l684_684647


namespace men_hours_per_day_l684_684941

theorem men_hours_per_day
  (H : ℕ)
  (men_days := 15 * 21 * H)
  (women_days := 21 * 20 * 9)
  (conversion_ratio := 3 / 2)
  (equivalent_man_hours := women_days * conversion_ratio)
  (same_work : men_days = equivalent_man_hours) :
  H = 8 :=
by
  sorry

end men_hours_per_day_l684_684941


namespace number_of_special_integers_l684_684722

theorem number_of_special_integers :
  ∃ N : ℕ, N = 2 ∧ (∀ n : ℕ, (n < 500 ∧ n = 7 * (nat.digits 10 n).sum ∧ nat.prime ((nat.digits 10 n).sum)) → n = 21 ∨ n = 133) :=
sorry

end number_of_special_integers_l684_684722


namespace sum_of_sqrt_times_ints_not_zero_l684_684190

open Classical
noncomputable theory

theorem sum_of_sqrt_times_ints_not_zero
  (m : ℕ)
  (a : Fin m → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_not_divisible : ∀ i, ∀ n > 1, ¬ (n * n ∣ a i))
  (b : Fin m → ℤ)
  (h_nonzero : ∀ i, b i ≠ 0) :
  (Finset.univ.sum (λ i, (real.sqrt (a i) * b i : ℝ))) ≠ 0 := by
  sorry

end sum_of_sqrt_times_ints_not_zero_l684_684190


namespace g_g_g_3_equals_107_l684_684734

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_g_g_3_equals_107 : g (g (g 3)) = 107 := 
by 
  sorry

end g_g_g_3_equals_107_l684_684734


namespace time_with_DE_only_l684_684763

theorem time_with_DE_only (d e f : ℚ) 
  (h₁ : d + e + f = 1/2) 
  (h₂ : d + f = 1/3) 
  (h₃ : e + f = 1/4) : 
  1 / (d + e) = 12 / 5 :=
begin
  sorry
end

end time_with_DE_only_l684_684763


namespace g_possible_values_l684_684452

noncomputable def g (x y z : ℝ) : ℝ :=
  (x + y) / x + (y + z) / y + (z + x) / z

theorem g_possible_values (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  6 ≤ g x y z :=
by
  sorry

end g_possible_values_l684_684452


namespace quiz_sum_correct_l684_684762

theorem quiz_sum_correct (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (h_sub : x - y = 4) (h_mul : x * y = 104) :
  x + y = 20 := by
  sorry

end quiz_sum_correct_l684_684762


namespace two_digit_numbers_rearranged_at_least_twice_l684_684369

/--
The number of two-digit numbers which increase by at least twice when their digits are rearranged.
-/
theorem two_digit_numbers_rearranged_at_least_twice : 
  let num_pairs := 
    (list.range' 1 9).bind (fun a => 
    (list.range' 0 10).filter (fun b => 10 * b + a ≥ 2 * (10 * a + b))).length
  in num_pairs = 14 :=
by
  sorry

end two_digit_numbers_rearranged_at_least_twice_l684_684369


namespace belts_count_l684_684631

-- Definitions based on conditions
variable (shoes belts hats : ℕ)

-- Conditions from the problem
axiom shoes_eq_14 : shoes = 14
axiom hat_count : hats = 5
axiom shoes_double_of_belts : shoes = 2 * belts

-- Definition of the theorem to prove the number of belts
theorem belts_count : belts = 7 :=
by
  sorry

end belts_count_l684_684631


namespace rplus_vector_space_l684_684277

open Real

def add_op (a b : ℝ) : ℝ := a * b
def scal_mul (α : ℝ) (a : ℝ) : ℝ := a^α

theorem rplus_vector_space : 
  ∃ (zero : ℝ) (inv : ℝ → ℝ), 
    (∀ (a b : ℝ), 0 < a ∧ 0 < b → add_op a b = add_op b a) ∧
    (∀ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c → add_op (add_op a b) c = add_op a (add_op b c)) ∧
    (∀ (a : ℝ), 0 < a → add_op a zero = a) ∧
    (∀ (a : ℝ), 0 < a → add_op a (inv a) = zero) ∧
    (∀ (α β : ℝ) (a : ℝ), 0 < a → scal_mul α (scal_mul β a) = scal_mul (α * β) a) ∧
    (∀ (a : ℝ), 0 < a → scal_mul 1 a = a) ∧
    (∀ (α β : ℝ) (a : ℝ), 0 < a → scal_mul (α + β) a = add_op (scal_mul α a) (scal_mul β a)) ∧
    (∀ (α : ℝ) (a b : ℝ), 0 < a ∧ 0 < b → scal_mul α (add_op a b) = add_op (scal_mul α a) (scal_mul α b)) :=
begin
  sorry
end

end rplus_vector_space_l684_684277


namespace sqrt_meaningful_iff_ge_one_l684_684388

theorem sqrt_meaningful_iff_ge_one (x : ℝ) : (∃ y, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end sqrt_meaningful_iff_ge_one_l684_684388


namespace quadrilateral_angles_l684_684876

theorem quadrilateral_angles (O B A C D : Type) (α β γ δ : ℝ)
  (h1 : α = 30) (h2 : β = 45) (h3 : γ = 45) (h4 : δ = 30) :
  (∠A = 60 ∧ ∠B = 105 ∧ ∠C = 60 ∧ ∠D = 135) ∨ (∠A = 120 ∧ ∠B = 45 ∧ ∠C = 120 ∧ ∠D = 75) := by
  sorry

end quadrilateral_angles_l684_684876


namespace third_price_reduction_l684_684530

theorem third_price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h1 : (original_price * (1 - x)^2 = final_price))
  (h2 : final_price = 100)
  (h3 : original_price = 100 / (1 - 0.19)) :
  (original_price * (1 - x)^3 = 90) :=
by
  sorry

end third_price_reduction_l684_684530


namespace trig_identity_l684_684901

theorem trig_identity : 
  Real.sin (600 * Real.pi / 180) + Real.tan (240 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l684_684901


namespace orchard_harvest_l684_684880

theorem orchard_harvest (sacks_per_section : ℕ) (sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 → sections = 8 → total_sacks = sacks_per_section * sections → total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end orchard_harvest_l684_684880


namespace sum_n_values_binom_l684_684126

open Nat

theorem sum_n_values_binom (h : ∑ n in finset.range 16, if (choose 15 n + choose 15 7 = choose 16 8) then n else 0) = 8 :=
by
  sorry

end sum_n_values_binom_l684_684126


namespace units_digit_of_50_factorial_l684_684138

theorem units_digit_of_50_factorial : (nat.factorial 50) % 10 = 0 := 
by 
  sorry

end units_digit_of_50_factorial_l684_684138


namespace minimum_number_of_triangles_l684_684951

-- Define the initial conditions
def chessboard_size : ℕ := 64
def removed_square : ℕ := 1
def remaining_squares : ℕ := 63

-- Define the conditions for the congruent triangles
def triangle_area : ℚ := 7 / 2

-- Define the statement to be proven
theorem minimum_number_of_triangles : chessboard_size - removed_square = remaining_squares →
  ∃ n : ℕ, n = remaining_squares / triangle_area ∧ n = 18 :=
begin
  sorry
end

end minimum_number_of_triangles_l684_684951


namespace minimum_value_expression_l684_684549

theorem minimum_value_expression (x y : ℝ) : ∃ (x y : ℝ), x = 4 ∧ y = -3 ∧ (x^2 + y^2 - 8 * x + 6 * y + 25) = 0 :=
by
  use 4, -3
  split
  · rfl
  split
  · rfl
  calc
    4^2 + (-3)^2 - 8 * 4 + 6 * (-3) + 25
      = 16 + 9 - 32 - 18 + 25 : by norm_num
  ... = 0 : by norm_num
  done

end minimum_value_expression_l684_684549


namespace taxi_ride_cost_l684_684975

-- Definitions given in the conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 10

-- The theorem we need to prove
theorem taxi_ride_cost : base_fare + (cost_per_mile * distance_traveled) = 5.00 :=
by
  sorry

end taxi_ride_cost_l684_684975


namespace function_is_even_l684_684342

variable (f : ℝ → ℝ)

-- Conditions
def periodic_4 (f : ℝ → ℝ) := ∀ x : ℝ, f(x) = f(x + 4)
def symmetric_about_2 (f : ℝ → ℝ) := ∀ x : ℝ, f(2 + x) = f(2 - x)

-- Theorem statement
theorem function_is_even (periodic_4 f) (symmetric_about_2 f) : ∀ x : ℝ, f(x) = f(-x) :=
by
  sorry

end function_is_even_l684_684342


namespace frost_cupcakes_together_in_8_minutes_l684_684994

def Cagney_frosting_rate := 1/15 -- cupcakes per second
def Lacey_frosting_rate := 1/35 -- cupcakes per second (effective rate: includes rest time)
def combined_frosting_rate := Cagney_frosting_rate + Lacey_frosting_rate -- combined rate

theorem frost_cupcakes_together_in_8_minutes (time_minutes : ℕ) (time_seconds : ℕ) (combined_rate : ℚ) :
  time_minutes = 8 →
  time_seconds = time_minutes * 60 →
  combined_rate = Cagney_frosting_rate + Lacey_frosting_rate →
  (time_seconds * combined_rate).natAbs = 45 :=
by
  -- proof goes here
  sorry

end frost_cupcakes_together_in_8_minutes_l684_684994


namespace units_digit_factorial_50_is_0_l684_684146

theorem units_digit_factorial_50_is_0 : (nat.factorial 50) % 10 = 0 := by
  sorry

end units_digit_factorial_50_is_0_l684_684146


namespace malcolm_initial_white_lights_l684_684085

-- Definitions based on the conditions
def red_lights : Nat := 12
def blue_lights : Nat := 3 * red_lights
def green_lights : Nat := 6
def total_colored_lights := red_lights + blue_lights + green_lights
def lights_left_to_buy : Nat := 5
def initially_white_lights := total_colored_lights + lights_left_to_buy

-- Proof statement
theorem malcolm_initial_white_lights : initially_white_lights = 59 := by
  sorry

end malcolm_initial_white_lights_l684_684085


namespace dave_shirts_not_washed_l684_684940

variable (short_sleeve_shirts long_sleeve_shirts washed_shirts : ℕ)

theorem dave_shirts_not_washed (h1 : short_sleeve_shirts = 9) (h2 : long_sleeve_shirts = 27) (h3 : washed_shirts = 20) :
  (short_sleeve_shirts + long_sleeve_shirts - washed_shirts = 16) :=
by {
  -- sorry indicates the proof is omitted
  sorry
}

end dave_shirts_not_washed_l684_684940


namespace chord_length_of_curve_by_line_l684_684215

theorem chord_length_of_curve_by_line :
  let x (t : ℝ) := 2 + 2 * t
  let y (t : ℝ) := -t
  let curve_eq (θ : ℝ) := 4 * Real.cos θ
  ∃ a b : ℝ, (x a = 2 + 2 * a ∧ y a = -a) ∧ (x b = 2 + 2 * b ∧ y b = -b) ∧
  ((x a - x b)^2 + (y a - y b)^2 = 4^2) :=
by
  sorry

end chord_length_of_curve_by_line_l684_684215


namespace midpoint_of_BK_l684_684782

noncomputable def triangle := {A B C : Point}

variables {A B C X K Z M : Point}

-- Define the points and conditions
variables (h_iso : AB = AC)
variables (h_X_on_AC : X ∈ AC)
variables (h_K_on_AB : K ∈ AB)
variables (h_KX_CX : KX = CX)
variables (h_bisector_AKX : bisector (angle AKX) ∩ line BC = Z)
variables (h_M_def : M = KB ∩ XZ)

-- Statement to prove
theorem midpoint_of_BK (h_iso : AB = AC) 
                       (h_X_on_AC : X ∈ AC) 
                       (h_K_on_AB : K ∈ AB) 
                       (h_KX_CX : KX = CX) 
                       (h_bisector_AKX : bisector (angle AKX) ∩ line BC = Z) 
                       (h_M_def : M = KB ∩ XZ) : 
                       midpoint M BK :=
sorry

end midpoint_of_BK_l684_684782


namespace mahi_received_exact_amount_l684_684200

noncomputable def mahi_receives : ℤ :=
  let x := 2167 / 30 in
  let M := 6 * x + 4 in
  let mile_bonus := 7 * 10 in
  M + mile_bonus

theorem mahi_received_exact_amount :
  mahi_receives = 507.38 := by
  -- Provided conditions
  -- Total amount divided: 2200
  -- Share ratios after removing specific amounts and the age and distance considerations
  sorry

end mahi_received_exact_amount_l684_684200


namespace combined_investment_yield_l684_684859

theorem combined_investment_yield :
  let income_A := 0.14 * 500
  let income_B := 0.08 * 750
  let income_C := 0.12 * 1000
  let total_income := income_A + income_B + income_C
  let total_market_value := 500 + 750 + 1000
  (total_income / total_market_value) * 100 = 11.11 :=
by
  let income_A := 0.14 * 500
  let income_B := 0.08 * 750
  let income_C := 0.12 * 1000
  let total_income := income_A + income_B + income_C
  let total_market_value := 500 + 750 + 1000
  have : total_income = 250 := by norm_num
  have : total_market_value = 2250 := by norm_num
  have : (total_income / total_market_value) * 100 = (250 / 2250) * 100 := by congr
  have : (250 / 2250) * 100 = 11.11 := by norm_num
  exact this

end combined_investment_yield_l684_684859


namespace largest_integer_base_7_l684_684791

theorem largest_integer_base_7 :
  let M := 66 in
  M ^ 2 = 48 ^ 2 :=
by
  let M := (6 * 7 + 6) in
  have h : M ^ 2 = 48 ^ 2 := rfl
  sorry -- Proof not required.

end largest_integer_base_7_l684_684791


namespace initial_population_approximately_l684_684890

noncomputable def initial_population_from (doubling_time : ℝ) (final_population : ℝ) (time : ℝ) : ℝ :=
  final_population / (2^(time / doubling_time))

theorem initial_population_approximately :
  initial_population_from 4 500000 35.86 ≈ 1009 :=
sorry

end initial_population_approximately_l684_684890


namespace probability_of_6_heads_in_8_flips_l684_684953

theorem probability_of_6_heads_in_8_flips :
  let n : ℕ := 8
  let k : ℕ := 6
  let total_outcomes := 2 ^ n
  let successful_outcomes := Nat.choose n k
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 7 / 64 := by
  sorry

end probability_of_6_heads_in_8_flips_l684_684953


namespace angle_ACB_l684_684751

open Real
open Locale.Real

/-- In triangle ABC, with angle ABC = 30 degrees, point D on segment BC such that 3 * BD = 2 * CD, 
and angle DAB = 10 degrees, we want to prove that angle ACB = 10 degrees. -/
theorem angle_ACB (A B C D : Point) (ABC_30 : angle B A C = 30) (D_on_BC : collinear B D C) 
                (BD_2_3_CD : 3 * (dist B D) = 2 * (dist D C)) (DAB_10 : angle D A B = 10) : 
                angle A C B = 10 := 
sorry

end angle_ACB_l684_684751


namespace stream_speed_difference_is_15_l684_684812

noncomputable def speed_of_stream_first_boat (stream_speed : ℝ) : Prop :=
  let boat_speed := 36
  in (1 / (boat_speed - stream_speed) = 2 * (1 / (boat_speed + stream_speed)))

noncomputable def speed_of_stream_second_boat (stream_speed : ℝ) : Prop :=
  let boat_speed := 54
  in (1 / (boat_speed - stream_speed) = 3 * (1 / (boat_speed + stream_speed)))

noncomputable def difference_in_stream_speeds : ℝ :=
  let first_stream_speed := 12
  let second_stream_speed := 27
  in second_stream_speed - first_stream_speed

theorem stream_speed_difference_is_15 : difference_in_stream_speeds = 15 :=
sorry

end stream_speed_difference_is_15_l684_684812


namespace probability_prime_and_multiple_of_13_l684_684053

open Set

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def cards := {n : ℕ | n ≥ 1 ∧ n ≤ 75}
def favorable_outcome := {n : ℕ | n ∈ cards ∧ n % 13 = 0 ∧ is_prime n}

theorem probability_prime_and_multiple_of_13 :
  (favorable_outcome.card : ℚ) / (cards.card : ℚ) = 1 / 75 :=
by
  sorry

end probability_prime_and_multiple_of_13_l684_684053


namespace sqrt_15_minus_1_bounds_l684_684282

theorem sqrt_15_minus_1_bounds :
  (sqrt 15) > 3 ∧ (sqrt 15) < 4 → 2 < (sqrt 15) - 1 ∧ (sqrt 15) - 1 < 3 :=
by
  intro h
  have h1 : 3 < sqrt 15 := h.left
  have h2 : sqrt 15 < 4 := h.right
  apply And.intro
  { -- 2 < sqrt 15 - 1
    linarith
  } 
  { -- sqrt 15 - 1 < 3
    linarith
  }
  sorry

end sqrt_15_minus_1_bounds_l684_684282


namespace arithmetic_expression_eq2016_l684_684043

theorem arithmetic_expression_eq2016 :
  (1 / 8 : ℚ) * (1 / 9) * (1 / 28) = 1 / 2016 ∨ ((1 / 8 - 1 / 9) * (1 / 28) = 1 / 2016) := 
by sorry

end arithmetic_expression_eq2016_l684_684043


namespace average_people_per_hour_l684_684413

theorem average_people_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) (total_hours : ℕ) (average_per_hour : ℕ) :
  total_people = 3000 ∧ days = 5 ∧ hours_per_day = 24 ∧ total_hours = days * hours_per_day ∧ average_per_hour = total_people / total_hours → 
  average_per_hour = 25 :=
by
  sorry

end average_people_per_hour_l684_684413


namespace largest_possible_b_l684_684904

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
sorry

end largest_possible_b_l684_684904


namespace sandwich_count_l684_684527

theorem sandwich_count 
  (S_total : ℕ)
  (S_sold : ℕ)
  (h_total : S_total = 50)
  (h_sold : S_sold = 33) :
  S_total - S_sold = 17 := by
  rw [h_total, h_sold]
  norm_num

end sandwich_count_l684_684527


namespace largest_integer_with_4_digit_square_in_base_7_l684_684784

theorem largest_integer_with_4_digit_square_in_base_7 (M : ℕ) :
  (∀ m : ℕ, m < 240 ∧ 49 ≤ m → m ≤ 239) ∧ nat.to_digits 7 239 = [4, 6, 1] :=
begin
  sorry
end

end largest_integer_with_4_digit_square_in_base_7_l684_684784


namespace largest_possible_b_l684_684905

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
sorry

end largest_possible_b_l684_684905


namespace segment_PQ_length_l684_684407

open Real

noncomputable def semicircle_parametric_eq : ℝ → ℝ × ℝ := λ φ, ((1 + cos φ), sin φ)

def polar_line_eq (ρ θ : ℝ) : Prop := ρ * (sin θ + sqrt 3 * cos θ) = 5 * sqrt 3

def polar_semi_circular_eq (ρ θ : ℝ) : Prop := ρ = 2 * cos θ

theorem segment_PQ_length :
  (∀ φ, 0 ≤ φ ∧ φ ≤ π → ∃ x y, (x = 1 + cos φ ∧ y = sin φ)) →
  (∀ θ, (θ = π / 3) → ∃ ρ, polar_semi_circular_eq ρ θ) →
  (∀ θ, (θ = π / 3) → ∃ ρ, polar_line_eq ρ θ) →
  |1 - 5| = 4 :=
by
  sorry

end segment_PQ_length_l684_684407


namespace find_other_number_l684_684082

theorem find_other_number (x y : ℕ) (h_gcd : Nat.gcd x y = 22) (h_lcm : Nat.lcm x y = 5940) (h_x : x = 220) :
  y = 594 :=
sorry

end find_other_number_l684_684082


namespace factor_expression_l684_684646

theorem factor_expression (a b c : ℝ) :
  3*a^3*(b^2 - c^2) - 2*b^3*(c^2 - a^2) + c^3*(a^2 - b^2) =
  (a - b)*(b - c)*(c - a)*(3*a^2 - 2*b^2 - 3*a^3/c + c) :=
sorry

end factor_expression_l684_684646


namespace probability_theorem_l684_684663

noncomputable def probability_odd_sum_given_even_product : ℚ :=
  let total_ways := 8^5
  let ways_no_even := 4^5
  let ways_at_least_one_even := total_ways - ways_no_even
  let ways_odd_sum_given_even (n : ℕ) : ℕ :=
    match n with 
    | 1 => 5 * 4^1 * 4^4
    | 3 => 10 * 4^3 * 4^2
    | _ => 0
  let favorable_outcomes := ways_odd_sum_given_even 1 + ways_odd_sum_given_even 3
  favorable_outcomes / ways_at_least_one_even

theorem probability_theorem : probability_odd_sum_given_even_product = 15 / 31 := 
  sorry

end probability_theorem_l684_684663


namespace connor_cats_l684_684034

theorem connor_cats (j : ℕ) (a : ℕ) (m : ℕ) (c : ℕ) (co : ℕ) (x : ℕ) 
  (h1 : a = j / 3)
  (h2 : m = 2 * a)
  (h3 : c = a / 2)
  (h4 : c = co + 5)
  (h5 : j = 90)
  (h6 : x = j + a + m + c + co) : 
  co = 10 := 
by
  sorry

end connor_cats_l684_684034


namespace compute_series_sum_l684_684262

-- Define the series term
def series_term (n : ℕ) : ℝ :=
  1 / (n * Real.sqrt (n + 1) + (n + 1) * Real.sqrt n)

-- State the theorem
theorem compute_series_sum :
  (∑ n in Finset.range' 3 198, series_term n) = 1 / Real.sqrt 3 - 1 / Real.sqrt 201 :=
by
  sorry

end compute_series_sum_l684_684262


namespace animath_workshop_lists_l684_684064

/-- The 79 trainees of the Animath workshop each choose an activity for the free afternoon 
among 5 offered activities. It is known that:
- The swimming pool was at least as popular as soccer.
- The students went shopping in groups of 5.
- No more than 4 students played cards.
- At most one student stayed in their room.
We write down the number of students who participated in each activity.
How many different lists could we have written? --/
theorem animath_workshop_lists :
  ∃ (l : ℕ), l = Nat.choose 81 2 := 
sorry

end animath_workshop_lists_l684_684064


namespace tan_330_eq_neg_sqrt3_div_3_l684_684306

theorem tan_330_eq_neg_sqrt3_div_3 :
  Real.tan (330 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_330_eq_neg_sqrt3_div_3_l684_684306


namespace fraction_of_water_in_vat_l684_684781

noncomputable theory
open_locale big_operators

theorem fraction_of_water_in_vat :
  ∃ (bottle : ℕ → ℝ) (vat : ℕ → ℝ),
  bottle 0 = 1 ∧
  (∀ n, bottle n = 0.5 * bottle (n - 1) if n > 0 else bottle 0) ∧
  (vat 0 = 0 ∧
  ∀ n, vat (n + 1) = vat n + 0.5 * bottle n) ∧
  bottle 10 = 1 / 1024 ∧
  vat 10 = 5 - 0.5 * bottle 10 ∧
  (vat 10 / (vat 10 + bottle 10)) = 1023 / 1024 :=
sorry

end fraction_of_water_in_vat_l684_684781


namespace shiela_neighbors_l684_684054

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) (neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : drawings_per_neighbor = 9) : neighbors = total_drawings / drawings_per_neighbor :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end shiela_neighbors_l684_684054


namespace train_passes_jogger_in_37_seconds_l684_684214

-- Define the parameters
def jogger_speed_kmph : ℝ := 9
def train_speed_kmph : ℝ := 45
def headstart : ℝ := 250
def train_length : ℝ := 120

-- Convert speeds from km/h to m/s
noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

-- Calculate relative speed in m/s
noncomputable def relative_speed : ℝ :=
  train_speed_mps - jogger_speed_mps

-- Calculate total distance to be covered in meters
def total_distance : ℝ :=
  headstart + train_length

-- Calculate time taken to pass the jogger in seconds
noncomputable def time_to_pass : ℝ :=
  total_distance / relative_speed

theorem train_passes_jogger_in_37_seconds :
  time_to_pass = 37 :=
by
  -- Proof would be here
  sorry

end train_passes_jogger_in_37_seconds_l684_684214


namespace intersect_or_parallel_lines_through_A1B1C1_l684_684047

-- Definition of two triangles
variable {A B C A1 B1 C1 : Type}

-- Conditions: lines through vertices of triangle ABC parallel to lines B1C1, C1A1, A1B1 intersect at a point
axiom lines_through_ABC_parallel_intersect :
  (∀ P : Type, ∀ Q : Type, ∀ R : Type, 
   (P, Q, R) = ((A → (B1 → C1)) ∧ (B → (C1 → A1)) ∧ (C → (A1 → B1))) ∧ 
   ∃ O : Type, (∀ P' ∈ P, ∀ Q' ∈ Q, ∀ R' ∈ R, P'.intersects_at Q' R' O))

-- Goal: lines through vertices of triangle A1B1C1 parallel to lines BC, CA, AB intersect at a point (or are parallel)
theorem intersect_or_parallel_lines_through_A1B1C1 :
  (∃ O' : Type, (A1 → BC).intersects_at (B1 → CA) (C1 → AB) O') ∨ 
  (∀ P : Type, ∀ Q : Type, ∀ R : Type, 
   (P, Q, R) = ((A1 → BC) ∧ (B1 → CA) ∧ (C1 → AB)) ∧ 
   (∀ P' ∈ P, ∀ Q' ∈ Q, ∀ R' ∈ R, P'.parallel_to Q' R')) :=
sorry

end intersect_or_parallel_lines_through_A1B1C1_l684_684047


namespace train_stop_time_per_hour_l684_684566

theorem train_stop_time_per_hour (v_excl_stop v_incl_stop : ℝ) (h1 : v_excl_stop = 48) (h2 : v_incl_stop = 36) : 
  ((v_excl_stop - v_incl_stop) / (v_excl_stop / 60)) = 15 :=
by 
  rw [h1, h2] 
  norm_num 
  sorry

end train_stop_time_per_hour_l684_684566


namespace units_digit_of_50_factorial_l684_684140

theorem units_digit_of_50_factorial : (nat.factorial 50) % 10 = 0 := 
by 
  sorry

end units_digit_of_50_factorial_l684_684140


namespace container_remaining_content_l684_684374

theorem container_remaining_content:
  let initial_content : ℚ := 1 in
  let content_after_first_day : ℚ := initial_content - (2 / 3) * initial_content in
  let content_after_second_day : ℚ := content_after_first_day - (1 / 4) * content_after_first_day in
  let content_after_third_day : ℚ := content_after_second_day - (1 / 5) * content_after_second_day in
  let content_after_fourth_day : ℚ := content_after_third_day - (1 / 3) * content_after_third_day in
  content_after_fourth_day = 2 / 15 :=
by sorry

end container_remaining_content_l684_684374


namespace min_value_fraction_108_l684_684455

noncomputable def min_value_fraction (x y z w : ℝ) : ℝ :=
(x + y) / (x * y * z * w)

theorem min_value_fraction_108 (x y z w : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : w > 0) (h_sum : x + y + z + w = 1) :
  min_value_fraction x y z w = 108 :=
sorry

end min_value_fraction_108_l684_684455


namespace largest_prime_divisor_of_sum_of_squares_l684_684576

theorem largest_prime_divisor_of_sum_of_squares :
  let n := 17^2 + 144^2 
  n = 21025 →
  ∃ p : ℕ, nat.prime p ∧ p ∣ n ∧ 
           ∀ q : ℕ, nat.prime q ∧ q ∣ n → q ≤ p
:=
by
  simp only [←nat.pow_two, nat.pow_two, mul_eq_mul_left_iff, nat.pow_succ, nat.add_comm, pow_add_products];
  existsi 29;
  split; sorry


end largest_prime_divisor_of_sum_of_squares_l684_684576


namespace sum_of_g_35_l684_684439

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 - 3
noncomputable def g (y : ℝ) : ℝ := y^2 + y + 1

theorem sum_of_g_35 : g 35 = 21 := 
by
  sorry

end sum_of_g_35_l684_684439


namespace math_problem_l684_684339

noncomputable def answer := 21

theorem math_problem 
  (a b c d x : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |x| = 3) : 
  2 * x^2 - (a * b - c - d) + |a * b + 3| = answer := 
sorry

end math_problem_l684_684339


namespace circumcenters_cyclic_l684_684826

open EuclideanGeometry

variables {A B C M K : Point}

-- Definitions for acute-angled triangle and angle equality
def acute_angled_triangle (A B C : Point) : Prop := 
  triangle A B C ∧
  ∠BAC < pi / 2 ∧ ∠ABC < pi / 2 ∧ ∠ACB < pi / 2

def angle_equality (A B C M K : Point) : Prop := 
  ∠ABM = ∠CBK

-- Define the centers of circumcircles
noncomputable def circumcenter_ABM (A B M : Point) : Point := circumcenter A B M
noncomputable def circumcenter_ABK (A B K : Point) : Point := circumcenter A B K
noncomputable def circumcenter_CBM (C B M : Point) : Point := circumcenter C B M
noncomputable def circumcenter_CBK (C B K : Point) : Point := circumcenter C B K

-- Problem statement in Lean: Prove the centers lie on one circle
theorem circumcenters_cyclic (
  h_acute : acute_angled_triangle A B C,
  h_angle_eq : angle_equality A B C M K
  ) :
  ∃ (O1 O2 O3 O4 : Point), 
  O1 = circumcenter_ABM A B M ∧ 
  O2 = circumcenter_ABK A B K ∧ 
  O3 = circumcenter_CBM C B M ∧ 
  O4 = circumcenter_CBK C B K ∧ 
  cyclic4 O1 O2 O3 O4 :=
sorry

end circumcenters_cyclic_l684_684826


namespace necessary_but_not_sufficient_l684_684074

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 + 2 * x - 8 > 0) ↔ (x > 2) ∨ (x < -4) := by
sorry

end necessary_but_not_sufficient_l684_684074


namespace possible_double_roots_l684_684228

-- Define the polynomial and the conditions
variables (b1 b2 b3 r : ℤ)

-- Statement of the problem
theorem possible_double_roots (h1 : (Polynomial.X ^ 4 + polynomial.C b3 * Polynomial.X ^ 3 
                                + polynomial.C b2 * Polynomial.X ^ 2
                                + polynomial.C b1 * Polynomial.X 
                                + polynomial.C 50).isRoot (r) ∧ 
                               (Polynomial.X ^ 4 + polynomial.C b3 * Polynomial.X ^ 3 
                                + polynomial.C b2 * Polynomial.X ^ 2
                                + polynomial.C b1 * Polynomial.X 
                                + polynomial.C 50).isRoot (r)) :
  r = -5 ∨ r = -1 ∨ r = 1 ∨ r = 5 :=
sorry

end possible_double_roots_l684_684228


namespace sum_of_values_not_in_domain_of_g_l684_684276

def g (x : ℝ) : ℝ := 1 / (2 - 1 / (2 + 1 / x^2))

theorem sum_of_values_not_in_domain_of_g : 
  (∑ x in { x : ℝ | ¬(x ∈ (set_of (λ x, (2 - 1 / (2 + 1 / x^2)) (x) ≠ 0))) }, x) = 0 :=
by 
  sorry

end sum_of_values_not_in_domain_of_g_l684_684276


namespace invertible_functions_product_label_l684_684509

/-- Definition of function domains and properties. -/
def domain_f3 : set ℤ := { -5, -4, -3, -2, -1, 0, 1, 2 }

def graph_f2 := "parabola opening upwards"
def graph_f3 := "discrete points in domain_f3"
def graph_f4 := "curve resembling -atan(x)"
def graph_f5 := "hyperbola symmetric about the origin"

/-- Proposition that checks invertibility of the given functions using their graphs. -/
def invertible (graph: string) : Prop :=
  match graph with
  | "parabola opening upwards" => false
  | "discrete points in domain_f3" => true
  | "curve resembling -atan(x)" => true
  | "hyperbola symmetric about the origin" => true
  | _ => false

/-- Main theorem statement: The product of the labels of the invertible functions is equal to 60. -/
theorem invertible_functions_product_label : 
  { l2 := 2, l3 := 3, l4 := 4, l5 := 5 } → 
  list.prod [l3, l4, l5] = 60 :=
by sorry

end invertible_functions_product_label_l684_684509


namespace units_digit_of_50_factorial_l684_684174

theorem units_digit_of_50_factorial : 
  ∃ d, (d = List.prod (List.range 1 51)) ∧ (d % 10 = 0) :=
by
  sorry

end units_digit_of_50_factorial_l684_684174


namespace last_integer_in_sequence_l684_684097

-- Define the initial number and the rule for the sequence
def initial_number : ℤ := 1000000

def next_number (n : ℤ) : ℤ := n / 3

-- Define the condition that for a given number, it is the last integer in the sequence
def is_last_integer (n : ℤ) : Prop := next_number n < 1

theorem last_integer_in_sequence : ∃ (x : ℤ), is_last_integer x ∧ initial_number / (3 ^ x) = 1 :=
by
  sorry

end last_integer_in_sequence_l684_684097


namespace colleague_typing_time_l684_684037

theorem colleague_typing_time (T : ℝ) : 
  (∀ me_time : ℝ, (me_time = 180) →
  (∀ my_speed my_colleague_speed : ℝ, (my_speed = T / me_time) →
  (my_colleague_speed = 4 * my_speed) →
  (T / my_colleague_speed = 45))) :=
  sorry

end colleague_typing_time_l684_684037


namespace find_percentage_l684_684661

theorem find_percentage (P : ℝ) (h : ((P / 100) * 1442 - 0.36 * 1412) + 66 = 6) : P ≈ 31.08 :=
sorry

end find_percentage_l684_684661


namespace find_a2016_l684_684693

noncomputable def sequence : ℕ → ℝ
| 0 := 1
| n+1 := (2 * (n + 1) / n) * sequence n

theorem find_a2016 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, S n = (n / 2) * a (n + 1))
  (h3 : ∀ n, S n = ∑ i in range (n + 1), a i) :
  a 2016 = 2016 := 
sorry

end find_a2016_l684_684693


namespace matrices_commute_l684_684008

variable {n : Nat}
variable (A B X : Matrix (Fin n) (Fin n) ℝ)

theorem matrices_commute (h : A * X * B + A + B = 0) : A * X * B = B * X * A :=
by
  sorry

end matrices_commute_l684_684008


namespace cheese_division_l684_684824

variable {α : Type} [LinearOrderedField α]

theorem cheese_division (masses : Fin 9 → α)
  (h_sorted : ∀ i j : Fin 9, i < j → masses i < masses j) :
  ∃ (a b : α) (m_split : Fin 10 → α), 
  (∃ i, masses i = a + b) ∧
  (m_split 0 = masses 0) ∧ (m_split 1 = masses 1) ∧ (m_split 2 = masses 2) ∧ (m_split 3 = masses 3) ∧
  (m_split 4 = masses 4) ∧ (m_split 5 = masses 5) ∧ (m_split 6 = masses 6) ∧ (m_split 7 = masses 7) ∧
  ((m_split 8 = a) ∨ (m_split 9 = a) ∨ (m_split 8 = b) ∨ (m_split 9 = b)) ∧
  ((∑ i in Finset.range 5, m_split i) = (∑ i in Finset.range 5 \ Finset.singleton 4, m_split i + m_split 9)) := 
sorry

end cheese_division_l684_684824


namespace yellow_parrots_l684_684307

variable (total_parrots : ℕ)
variable (fraction_red : ℚ)
variable (fraction_yellow : ℚ)
variable (number_yellow : ℕ)

-- Conditions
def condition1 : Prop := fraction_red = 5 / 6
def condition2 : Prop := fraction_yellow = 1 - fraction_red
def condition3 : Prop := total_parrots = 108

-- Question to prove
theorem yellow_parrots : 
  condition1 ∧ condition2 ∧ condition3 → number_yellow = 18 := 
by
  sorry

end yellow_parrots_l684_684307


namespace ratio_green_students_l684_684462

/-- 
Miss Molly surveyed her class of 30 students about their favorite color.
Some portion of the class answered green, one-third of the girls answered pink,
and the rest of the class answered yellow. There are 18 girls in the class, 
and 9 students like yellow best. Prove that the ratio of students who answered green 
to the total number of students is 1:2.
-/
theorem ratio_green_students (total_students girls pink_ratio yellow_best green_students : ℕ)
    (h_total : total_students = 30)
    (h_girls : girls = 18)
    (h_pink_ratio : pink_ratio = girls / 3)
    (h_yellow : yellow_best = 9)
    (h_green : green_students = total_students - (pink_ratio + yellow_best)) :
  green_students / total_students = 1 / 2 :=
  sorry

end ratio_green_students_l684_684462


namespace bake_sale_total_money_l684_684622

def dozens_to_pieces (dozens : Nat) : Nat :=
  dozens * 12

def total_money_raised
  (betty_chocolate_chip_dozen : Nat)
  (betty_oatmeal_raisin_dozen : Nat)
  (betty_brownies_dozen : Nat)
  (paige_sugar_cookies_dozen : Nat)
  (paige_blondies_dozen : Nat)
  (paige_cream_cheese_brownies_dozen : Nat)
  (price_per_cookie : Rat)
  (price_per_brownie_blondie : Rat) : Rat :=
let betty_cookies := dozens_to_pieces betty_chocolate_chip_dozen + dozens_to_pieces betty_oatmeal_raisin_dozen
let paige_cookies := dozens_to_pieces paige_sugar_cookies_dozen
let total_cookies := betty_cookies + paige_cookies
let betty_brownies := dozens_to_pieces betty_brownies_dozen
let paige_brownies_blondies := dozens_to_pieces paige_blondies_dozen + dozens_to_pieces paige_cream_cheese_brownies_dozen
let total_brownies_blondies := betty_brownies + paige_brownies_blondies
(total_cookies * price_per_cookie) + (total_brownies_blondies * price_per_brownie_blondie)

theorem bake_sale_total_money :
  total_money_raised 4 6 2 6 3 5 1 2 = 432 :=
by
  sorry

end bake_sale_total_money_l684_684622


namespace original_population_multiple_of_5_l684_684514

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem original_population_multiple_of_5 (x y z : ℕ) 
  (H1 : is_perfect_square (x * x)) 
  (H2 : x * x + 200 = y * y) 
  (H3 : y * y + 180 = z * z) : 
  ∃ k : ℕ, x * x = 5 * k := 
sorry

end original_population_multiple_of_5_l684_684514


namespace birds_problem_l684_684199

-- Define the initial number of birds and the total number of birds as given conditions.
def initial_birds : ℕ := 2
def total_birds : ℕ := 6

-- Define the number of new birds that came to join.
def new_birds : ℕ := total_birds - initial_birds

-- State the theorem to be proved, asserting that the number of new birds is 4.
theorem birds_problem : new_birds = 4 := 
by
  -- required proof goes here
  sorry

end birds_problem_l684_684199


namespace framed_painting_ratio_l684_684220

-- Define the conditions and the problem
theorem framed_painting_ratio:
  ∀ (x : ℝ),
    (30 + 2 * x) * (20 + 4 * x) = 1500 →
    (20 + 4 * x) / (30 + 2 * x) = 4 / 5 := 
by sorry

end framed_painting_ratio_l684_684220


namespace minimize_distance_sum_is_intersection_of_diagonals_l684_684652

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

structure Quadrilateral (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
  (convexness : ConvexQuadrilateral A B C D)

def minimize_distance_sum (Q : Quadrilateral A B C D) : A :=
  let O := intersect_diagonals Q
    in O

theorem minimize_distance_sum_is_intersection_of_diagonals
  (Q : Quadrilateral A B C D)
  (O := intersect_diagonals Q)
  (O1 : A) (hO1 : inside_quadrilateral O1 Q) :
  distance A O + distance B O + distance C O + distance D O ≤
  distance A O1 + distance B O1 + distance C O1 + distance D O1 :=
sorry

end minimize_distance_sum_is_intersection_of_diagonals_l684_684652


namespace angles_of_isosceles_triangle_l684_684610

-- Define the isosceles triangle and the angles
variables (α : ℝ) (β : ℝ)

-- Define the properties of the triangles
axiom isosceles_triangle (α β : ℝ) : α = 44 ∧ β = 92

-- Prove the angles of the original triangle given the conditions
theorem angles_of_isosceles_triangle :
  (isosceles_triangle α β) →
  α = 44 ∧ α = β :=
by
  intro h
  sorry

end angles_of_isosceles_triangle_l684_684610


namespace units_digit_of_50_factorial_l684_684141

theorem units_digit_of_50_factorial : (nat.factorial 50) % 10 = 0 := 
by 
  sorry

end units_digit_of_50_factorial_l684_684141


namespace three_people_paint_time_l684_684312

theorem three_people_paint_time :
  (∀ (n t : ℕ), n * t = 24) →
  (4 * 6 = 24) →
  (∃ t : ℕ, 3 * t = 24) →
  ∃ t : ℕ, t = 8 :=
by
  intros h1 h2 h3
  use 8
  sorry

end three_people_paint_time_l684_684312


namespace g_g_g_3_equals_107_l684_684735

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_g_g_3_equals_107 : g (g (g 3)) = 107 := 
by 
  sorry

end g_g_g_3_equals_107_l684_684735


namespace vector_b_solution_l684_684319

def vector (α : Type) := (α, α)

def parallel (a b : vector ℝ) : Prop :=
  ∃ k : ℝ, (fst b = k * fst a) ∧ (snd b = k * snd a)

def magnitude (v : vector ℝ) := 
  real.sqrt (fst v ^ 2 + snd v ^ 2)

theorem vector_b_solution :
  let a := (2, 1) in
  let b1 := (4, 2) in
  let b2 := (-4, -2) in
  (parallel a b1 ∧ magnitude b1 = 2 * real.sqrt 5) ∨
  (parallel a b2 ∧ magnitude b2 = 2 * real.sqrt 5) :=
sorry

end vector_b_solution_l684_684319


namespace angle_DAB_independent_of_triangle_l684_684756

-- We define the conditions.
variables (A B C D E : Type) [is_isosceles_triangle A B C CA CB] [is_square BCDE B C D E]

-- Now we state the theorem.
theorem angle_DAB_independent_of_triangle :
  ∀ (ABC : triangle) (A B C D E : point),
  is_isosceles_triangle ABC CA CB →
  is_square BCDE BC →
  angle DAB = 45 :=
by
  intros
  sorry

end angle_DAB_independent_of_triangle_l684_684756


namespace painted_cells_l684_684427

theorem painted_cells (k l : ℕ) (h : k * l = 74) :
  let rows := 2 * k + 1,
      columns := 2 * l + 1,
      total_cells := rows * columns,
      white_cells := 74,
      painted_cells := total_cells - white_cells
  in painted_cells = 373 ∨ painted_cells = 301 :=
sorry

end painted_cells_l684_684427


namespace largest_integer_with_4_digit_square_in_base_7_l684_684786

theorem largest_integer_with_4_digit_square_in_base_7 (M : ℕ) :
  (∀ m : ℕ, m < 240 ∧ 49 ≤ m → m ≤ 239) ∧ nat.to_digits 7 239 = [4, 6, 1] :=
begin
  sorry
end

end largest_integer_with_4_digit_square_in_base_7_l684_684786


namespace max_gcd_consecutive_terms_l684_684274

def sequence_b (n : ℕ) : ℕ := n.factorial + n^2 + 1

theorem max_gcd_consecutive_terms (n : ℕ) : 
  (nat.gcd (sequence_b n) (sequence_b (n + 1))) ≤ 1 := 
sorry

end max_gcd_consecutive_terms_l684_684274


namespace units_digit_factorial_50_l684_684134

theorem units_digit_factorial_50 : Nat.unitsDigit (Nat.factorial 50) = 0 := 
  sorry

end units_digit_factorial_50_l684_684134


namespace mid_point_coordinates_line_slope_condition_l684_684586

-- Definitions for problem 1
def alpha := Real.pi / 3
def line_l (t: Real) := (2 + t * Real.cos alpha, Real.sqrt 3 + t * Real.sin alpha)
def curve_C (rho theta: Real) := rho^2 = 4 / (1 + 3 * (Real.sin theta)^2)

-- Theorem for problem 1
theorem mid_point_coordinates :
  let t1 t2 := -28 / 13
  line_l t1 = (2 + (-28 / 13) * (1/2), sqrt(3) + (-28 / 13) * (sqrt(3) / 2))
  line_l t2 = (2 + (-28 / 13) * (1/2), sqrt(3) + (-28 / 13) * (sqrt(3) / 2))
  ∃ M : ℝ × ℝ, M = (12 / 13, -sqrt(3) / 13) := 
sorry

-- ...

-- Definitions for problem 2
def P := (2, sqrt(3) : ℝ)
def slope (α : ℝ) := Real.tan α

-- Theorem for problem 2
theorem line_slope_condition :
  (|P.1 - line_l (sqrt(5) / 4).1| * |P.2 - line_l (sqrt(5) / 4).2| = 7) → 
  slope ?α = sqrt(5) / 4 := sorry

end mid_point_coordinates_line_slope_condition_l684_684586


namespace units_digit_of_product_1_to_50_is_zero_l684_684157

theorem units_digit_of_product_1_to_50_is_zero :
  Nat.digits 10 (∏ i in Finset.range 51, i) = [0] :=
sorry

end units_digit_of_product_1_to_50_is_zero_l684_684157


namespace sand_pourings_l684_684590

theorem sand_pourings {n : ℕ} :
  (∀ i : ℕ, i ≥ 1 → remaining_sand (i-1) * (i+1)/ (i+2) = remaining_sand i) → 
  remaining_sand 0 = 1 →
  remaining_sand n = 1/5 ↔ n = 8 :=
by
  sorry

end sand_pourings_l684_684590


namespace symmetry_axis_of_quadratic_l684_684386

theorem symmetry_axis_of_quadratic (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x, (x + 4) = 0 ∧ a * (x + 4)^2 + bx + c = 0)
  (h₂ : ∃ y, (y - 1) = 0 ∧ a * (y - 1)^2 + by + c = 0) :
  -b / (2 * a) = 1.5 :=
by
  sorry

end symmetry_axis_of_quadratic_l684_684386


namespace sum_of_c_values_with_rational_roots_l684_684658

theorem sum_of_c_values_with_rational_roots :
  let c_values := {c : ℤ | -20 ≤ c ∧ c ≤ 20 ∧ ∃ k : ℤ, 81 + 4 * c = k * k}
  (∑ c in c_values, c) = -55 :=
by
  sorry

end sum_of_c_values_with_rational_roots_l684_684658


namespace largest_possible_n_l684_684960

/-- A natural number is called a prime power if it can be expressed as p^n for some prime p and natural number n -/
def is_prime_power (a : ℕ) : Prop :=
  ∃ (p : ℕ) (h : Nat.prime p) (n : ℕ), a = p ^ n

/-- A sequence of natural numbers satisfies the Fibonacci-type recurrence a_i = a_{i-1} + a_{i-2} -/
def fib_seq (a : ℕ → ℕ) : Prop :=
  ∀ n, 3 ≤ n → a n = a (n - 1) + a (n - 2)

/-- Given a sequence of prime powers a_1, a_2, ..., a_n that satisfies a_i = a_{i-1} + a_{i-2} for all 3 ≤ i ≤ n, 
    prove that the largest possible n is 7 -/
theorem largest_possible_n :
  ∃ (n : ℕ) (a : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ n → is_prime_power (a i)) ∧ fib_seq a ∧ n = 7 := 
sorry

end largest_possible_n_l684_684960


namespace value_of_a_minus_b_l684_684740

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a + b > 0) :
  (a - b = -1) ∨ (a - b = -7) :=
by
  sorry

end value_of_a_minus_b_l684_684740


namespace XiaoqiangGames_l684_684779

variable Jia Yi Bing Ding Xiaoqiang : ℕ

theorem XiaoqiangGames
  (hJia : Jia = 4)
  (hYi : Yi = 3)
  (hBing : Bing = 2)
  (hDing : Ding = 1) :
  Xiaoqiang = 2 := 
sorry

end XiaoqiangGames_l684_684779


namespace probability_male_saturday_female_sunday_l684_684316

theorem probability_male_saturday_female_sunday :
  let males := {1, 2} : Finset ℕ;
      females := {3, 4} : Finset ℕ;
      students := males ∪ females;
      total_pairs := students.pairs;
      favorable_pairs := (males.product females) ∪ (females.product males)
  in (favorable_pairs.card : ℝ) / (total_pairs.card : ℝ) = 1 / 3 :=
by
  sorry

end probability_male_saturday_female_sunday_l684_684316


namespace T3_defeat_T4_impossible_T4_defeat_T3_possible_l684_684821

open Classical

-- Definitions for teams and their scores in the tournament
structure Team :=
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)
  (points : ℕ := wins * 3 + draws)

namespace Tournament

-- Define the round-robin structure
def round_robin_tournament (teams : List Team) : Prop :=
  teams.length = 9 ∧ 
  ∀ t : Team, t ∈ teams ∧
  ∀ i j : ℕ, i ≠ j → teams.nth i ≠ none → teams.nth j ≠ none

-- Define the distinct scores constraint
def distinct_scores (teams : List Team) : Prop :=
  ∀ t1 t2 : Team, t1 ∈ teams → t2 ∈ teams → t1 ≠ t2 → t1.points ≠ t2.points

-- Conditions for T1 and T9
def T1 := {wins := 3, draws := 4, losses := 1}
def T9 := {wins := 0, draws := 5, losses := 3}

theorem T3_defeat_T4_impossible (teams : List Team) (hrr : round_robin_tournament teams) (hds: distinct_scores teams) (hT1 : T1 ∈ teams) (hT9 : T9 ∈ teams) :
  ¬ (∃ T3 T4 : Team, T3 ∈ teams ∧ T4 ∈ teams ∧ T3.points > T4.points ∧ T3.wins > T4.wins + 1) :=
sorry

theorem T4_defeat_T3_possible (teams : List Team) (hrr : round_robin_tournament teams) (hds: distinct_scores teams) (hT1 : T1 ∈ teams) (hT9 : T9 ∈ teams) :
  ∃ T4 T3 : Team, T4 ∈ teams ∧ T3 ∈ teams ∧ T4.points > T3.points ∧ T4.wins > T3.wins - 1 :=
sorry

end Tournament

end T3_defeat_T4_impossible_T4_defeat_T3_possible_l684_684821


namespace distance_between_A_and_B_l684_684115

variable (d : ℝ) -- Total distance between A and B

def car_speeds (vA vB t : ℝ) : Prop :=
vA = 80 ∧ vB = 100 ∧ t = 2

def total_covered_distance (vA vB t : ℝ) : ℝ :=
(vA + vB) * t

def percentage_distance (total_distance covered_distance : ℝ) : Prop :=
0.6 * total_distance = covered_distance

theorem distance_between_A_and_B (vA vB t : ℝ) (H1 : car_speeds vA vB t) 
  (H2 : percentage_distance d (total_covered_distance vA vB t)) : d = 600 := by
  sorry

end distance_between_A_and_B_l684_684115


namespace sara_golf_balls_l684_684477

theorem sara_golf_balls :
  let dozen := 12
  in 9 * dozen = 108 :=
by
  let dozen := 12
  show 9 * dozen = 108
  sorry

end sara_golf_balls_l684_684477


namespace sale_in_fifth_month_equals_6562_l684_684213

variable (sale1 sale2 sale3 sale4 sale5 sale6 : ℝ)
variable (avg_sale_per_month : ℝ)

-- Defining the variables as per the given conditions
def sales_for_first_four_months := sale1 + sale2 + sale3 + sale4
def total_sales_for_six_months := avg_sale_per_month * 6
def total_sales_including_sixth_month := sales_for_first_four_months + sale6
def sale_for_fifth_month := total_sales_for_six_months - total_sales_including_sixth_month

-- Given conditions
axiom sales_first_month : sale1 = 6235
axiom sales_second_month : sale2 = 6927
axiom sales_third_month : sale3 = 6855
axiom sales_fourth_month : sale4 = 7230
axiom sales_sixth_month : sale6 = 5191
axiom desired_average_sale : avg_sale_per_month = 6500

-- The theorem to prove
theorem sale_in_fifth_month_equals_6562 :
  sale_for_fifth_month sale1 sale2 sale3 sale4 sale5 sale6 avg_sale_per_month = 6562 := by
  sorry

end sale_in_fifth_month_equals_6562_l684_684213


namespace intersection_points_l684_684653

noncomputable def parabola (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 4
noncomputable def line (x : ℝ) : ℝ := -x + 2

theorem intersection_points :
  (parabola (-1 / 3) = line (-1 / 3) ∧ parabola (-2) = line (-2)) ∧
  (parabola (-1 / 3) = 7 / 3) ∧ (parabola (-2) = 4) :=
by
  sorry

end intersection_points_l684_684653


namespace distance_between_vertices_l684_684436

def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x_v := -b / (2 * a)
  let y_v := a * (x_v ^ 2) + b * x_v + c
  (x_v, y_v)

theorem distance_between_vertices :
  let C := vertex 1 6 13
  let D := vertex 1 (-4) 5
  real.sqrt ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2) = real.sqrt 34 :=
by
  sorry

end distance_between_vertices_l684_684436


namespace positive_solution_y_l684_684361

theorem positive_solution_y (x y z : ℝ) 
  (h1 : x * y = 8 - 3 * x - 2 * y) 
  (h2 : y * z = 15 - 5 * y - 3 * z) 
  (h3 : x * z = 40 - 5 * x - 4 * z) : 
  y = 4 := 
sorry

end positive_solution_y_l684_684361


namespace largest_M_in_base_7_l684_684795

-- Define the base and the bounds for M^2
def base : ℕ := 7
def lower_bound : ℕ := base^3
def upper_bound : ℕ := base^4

-- Define M and its maximum value.
def M : ℕ := 48

-- Define a function to convert a number to its base 7 representation
def to_base_7 (n : ℕ) : List ℕ :=
  if n == 0 then [0] else
  let rec digits (n : ℕ) : List ℕ :=
    if n == 0 then [] else (n % 7) :: digits (n / 7)
  digits n |>.reverse

-- Define the base 7 representation of 48
def M_base_7 : List ℕ := to_base_7 M

-- The main statement asserting the conditions and the solution
theorem largest_M_in_base_7 :
  lower_bound ≤ M^2 ∧ M^2 < upper_bound ∧ M_base_7 = [6, 6] :=
by
  sorry

end largest_M_in_base_7_l684_684795


namespace distance_from_point_to_line_is_4_l684_684496

-- Define the point and the line
def point := (2 : ℤ, 3 : ℤ)
def line (x y : ℤ) : ℤ := 3 * x + 4 * y + 2

-- Define the distance formula
def point_to_line_distance (p : ℤ × ℤ) (l : ℤ → ℤ → ℤ) : ℚ :=
  let d := l p.1 p.2
  (abs d : ℚ) / real.to_rat (real.sqrt (3^2 + 4^2))

-- Theorem to prove the distance is 4
theorem distance_from_point_to_line_is_4 : point_to_line_distance point line = 4 := by
  sorry

end distance_from_point_to_line_is_4_l684_684496


namespace min_n_such_that_product_exceeds_1024_l684_684706

theorem min_n_such_that_product_exceeds_1024 :
  ∃ n : ℕ, (∀ k : ℕ, k < n → k = 15 → 2 ^ ((∑ i in finset.range (k+1), i.succ) / 12) ≤ 1024) ∧
            2 ^ ((∑ i in finset.range (n+1), i.succ) / 12) > 1024 :=
by
  sorry

end min_n_such_that_product_exceeds_1024_l684_684706


namespace units_digit_factorial_50_l684_684133

theorem units_digit_factorial_50 : Nat.unitsDigit (Nat.factorial 50) = 0 := 
  sorry

end units_digit_factorial_50_l684_684133


namespace probability_of_sum_12_with_at_least_one_4_or_more_l684_684124

-- Definitions for the problem conditions
def outcomes := {x : ℕ × ℕ × ℕ | x.1 + x.2.1 + x.2.2 = 12 ∧ 
  (x.1 ≥ 4 ∨ x.2.1 ≥ 4 ∨ x.2.2 ≥ 4)}

def total_possibilities := 6 * 6 * 6

noncomputable def count_outcomes : ℕ :=
  Set.card outcomes

-- The final probability we need to prove matches the expected result
theorem probability_of_sum_12_with_at_least_one_4_or_more :
  (count_outcomes : ℚ) / total_possibilities = 4 / 54 :=
  sorry

end probability_of_sum_12_with_at_least_one_4_or_more_l684_684124


namespace cos_and_sin_sum_ident_l684_684320

theorem cos_and_sin_sum_ident {α : ℝ} (h1 : sin α = 4/5) (h2 : 0 < α ∧ α < π / 2) :
  cos α = 3/5 ∧ sin (α + π / 4) = 7 * sqrt 2 / 10 :=
by
  sorry

end cos_and_sin_sum_ident_l684_684320


namespace parabola_directrix_l684_684499

theorem parabola_directrix (p : ℝ) (h : p = 2) : 
  ∀ y, y^2 = 4 * p * (y * y / (4 * p) - 1) = (y ^ 2 )  = 4 * x  → x = -1 :=
  sorry

end parabola_directrix_l684_684499


namespace rotated_log_to_f_l684_684385

noncomputable def f (x : ℝ) := 10^(-x) - 1

theorem rotated_log_to_f (x : ℝ) :
  (∀ y, y = Real.log (x + 1) → y = f (-x)) :=
by
  sorry

end rotated_log_to_f_l684_684385


namespace quarters_spent_l684_684609

/-- Adam's quarters problem. -/
theorem quarters_spent (initial_quarters : ℕ) (left_quarters : ℕ) (spent_quarters : ℕ) 
  (h1 : initial_quarters = 88) 
  (h2 : left_quarters = 79) : 
  spent_quarters = initial_quarters - left_quarters := by
  sorry

/-- Specifying the problem parameters. -/
example : quarters_spent 88 79 9 := by
  repeat { sorry }

end quarters_spent_l684_684609


namespace length_of_path_traversed_by_vertex_A_l684_684399

theorem length_of_path_traversed_by_vertex_A 
  (W X Y Z A B C : ℝ) 
  (side_square : ∀ W X Y Z, dist W X = 6 ∧ dist X Y = 6 ∧ dist Y Z = 6 ∧ dist Z W = 6)
  (equilateral_triangle : ∀ A B C, dist A B = 3 ∧ dist B C = 3 ∧ dist C A = 3)
  (initial_position : vertex C = side Z W)
  (rotation_path : vertex A rotates about C, then vertex A, then vertex B consecutively around the edges of the square )
  : dist_of_travel_path_length A = 12 * π :=
sorry

end length_of_path_traversed_by_vertex_A_l684_684399


namespace result_eq_neg_one_l684_684739

theorem result_eq_neg_one (x y: ℝ) 
  (h1: y - real.sqrt (x - 2022) = real.sqrt (2022 - x) - 2023) 
  (hx: x = 2022) 
  (hy: y = -2023) :
  (x + y) ^ 2023 = -1 := 
by 
  sorry

end result_eq_neg_one_l684_684739


namespace units_digit_of_50_factorial_is_0_l684_684165

theorem units_digit_of_50_factorial_is_0 : 
  (∃ n : ℕ, 50! ≡ n [MOD 10]) ∧ (n = 0) := sorry

end units_digit_of_50_factorial_is_0_l684_684165


namespace positive_integer_solutions_l684_684848

theorem positive_integer_solutions
  (x : ℤ) :
  (5 + 3 * x < 13) ∧ ((x + 2) / 3 - (x - 1) / 2 <= 2) →
  (x = 1 ∨ x = 2) :=
by
  sorry

end positive_integer_solutions_l684_684848


namespace trays_from_second_table_l684_684476

theorem trays_from_second_table 
  (trays_per_trip : ℕ) 
  (trays_from_first_table : ℕ) 
  (trips : ℕ) 
  (total_trays_needed : ℕ) : 
  trays_per_trip = 4 →
  trays_from_first_table = 10 →
  trips = 3 →
  total_trays_needed = trips * trays_per_trip →
  total_trays_needed - trays_from_first_table = 2 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3] at h4,
  simp at h4,
  exact h4,
end

end trays_from_second_table_l684_684476


namespace eval_complex_expression_l684_684284

theorem eval_complex_expression : (Real.sqrt ((Real.cbrt 4)^4))^6 = 256 := 
by 
  sorry

end eval_complex_expression_l684_684284


namespace quadratic_root_is_imaginary_unit_l684_684378

theorem quadratic_root_is_imaginary_unit (p q : ℝ)
  (h_eq : ∀ z : ℂ, z^2 + p * z + q = 0 → (z = 1 + complex.I ∨ z = 1 - complex.I))
  : q = 2 :=
sorry

end quadratic_root_is_imaginary_unit_l684_684378


namespace range_of_a_l684_684347

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2 * a) ^ x
  else a / x + 4

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x < f a y) → a < 0 :=
by
  sorry

end range_of_a_l684_684347


namespace area_of_centroid_quadrilateral_l684_684061

theorem area_of_centroid_quadrilateral:
  ∃ (ABCD : set (ℝ × ℝ)) (A B C D P : ℝ × ℝ),
    is_square ABCD ∧ 
    length A B = 30 ∧
    length B C = 30 ∧ 
    length C D = 30 ∧ 
    length D A = 30 ∧ 
    in_square P ABCD ∧ 
    length A P = 12 ∧ 
    length B P = 26 ∧ 
    ∀ (C1 C2 C3 C4 : ℝ × ℝ),
      is_centroid A B P C1 ∧ 
      is_centroid B C P C2 ∧ 
      is_centroid C D P C3 ∧ 
      is_centroid D A P C4 →
        area (C1, C2, C3, C4) = 200 :=
sorry

end area_of_centroid_quadrilateral_l684_684061


namespace anya_hair_growth_l684_684985

theorem anya_hair_growth (wash_loss : ℕ) (brush_loss : ℕ) (total_loss : ℕ) : wash_loss = 32 → brush_loss = wash_loss / 2 → total_loss = wash_loss + brush_loss → total_loss + 1 = 49 :=
by
  sorry

end anya_hair_growth_l684_684985


namespace population_exceeds_capacity_in_60_years_l684_684405

-- Define the conditions
def max_area : ℕ := 15000
def required_acres_per_person : ℝ := 0.75
def carrying_capacity : ℕ := (max_area / required_acres_per_person).to_nat

def initial_population : ℕ := 250
def quadruples_every_n_years : ℕ := 20
def years_to_exceed_capacity : ℕ := 60

-- Define the population growth function
def population_at_year (start_year : ℕ) (initial_population : ℕ) (quadruple_years : ℕ) (years_since_start : ℕ) :=
  initial_population * 4 ^ (years_since_start / quadruple_years)

-- The proof goal
theorem population_exceeds_capacity_in_60_years :
  population_at_year 2000 initial_population quadruples_every_n_years years_to_exceed_capacity ≥ carrying_capacity :=
  sorry

end population_exceeds_capacity_in_60_years_l684_684405


namespace perimeter_of_original_rectangle_l684_684599

theorem perimeter_of_original_rectangle
  (s : ℕ)
  (h1 : 4 * s = 24)
  (l w : ℕ)
  (h2 : l = 3 * s)
  (h3 : w = s) :
  2 * (l + w) = 48 :=
by
  sorry

end perimeter_of_original_rectangle_l684_684599


namespace probability_gong_yu_same_side_jiao_l684_684402

theorem probability_gong_yu_same_side_jiao :
  let tones := ["Gong", "Shang", "Jiao", "Zhi", "Yu"],
      total_sequences := Nat.factorial 5,
      favorable_sequences := 80 in
  (favorable_sequences / total_sequences : ℚ) = 2 / 3 :=
by
  sorry

end probability_gong_yu_same_side_jiao_l684_684402


namespace units_digit_of_50_factorial_is_0_l684_684168

theorem units_digit_of_50_factorial_is_0 : 
  (∃ n : ℕ, 50! ≡ n [MOD 10]) ∧ (n = 0) := sorry

end units_digit_of_50_factorial_is_0_l684_684168


namespace min_value_on_interval_l684_684383

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem min_value_on_interval (a : ℝ) (h : ∃ x ∈ Icc (-2 : ℝ) (-1), f x a = 2) : 
  ∃ y ∈ Icc (-2 : ℝ) (-1), f y a = -5 :=
by 
  sorry

end min_value_on_interval_l684_684383


namespace identify_quadratic_equation_l684_684924

theorem identify_quadratic_equation :
  (∃ a b c : ℝ, a ≠ 0 ∧ (λ x : ℝ, a * x^2 + b * x + c) = (λ x : ℝ, x^2 - x) ∧ 
    ∀ (x : ℝ), x^2 - x = 0 ↔ (a * x^2 + b * x + c = 0)) ∧
  (¬ ∃ a b c : ℝ, a ≠ 0 ∧ (λ x : ℝ, a * x^2 + b * x + c) = (λ x : ℝ, 2 * x + 1)) ∧
  (¬ ∃ a b c : ℝ, a ≠ 0 ∧ (λ x : ℝ, a * x^2 + b * x + c) = (λ x y : ℝ, y^2 + x)) ∧
  (¬ ∃ a b c : ℝ, a ≠ 0 ∧ (λ x : ℝ, a * x^2 + b * x + c) = (λ x : ℝ, 1 / x + x)) :=
by
  -- The proof is to be filled in here.
  sorry

end identify_quadratic_equation_l684_684924


namespace circle_radius_l684_684227

theorem circle_radius (P Q R : Point) (O : Point) (r : ℝ) :
  dist P O = 15 ∧ dist P Q = 10 ∧ dist Q R = 8 ∧ dist P O = sqrt ((15 - r) * (15 + r)) →
  r = 3 * sqrt 5 :=
by
  sorry

end circle_radius_l684_684227


namespace difference_between_numbers_l684_684076

theorem difference_between_numbers :
  ∃ S : ℝ, L = 1650 ∧ L = 6 * S + 15 ∧ L - S = 1377.5 :=
sorry

end difference_between_numbers_l684_684076


namespace sale_in_fourth_month_l684_684957

-- Given conditions
def sales_first_month : ℕ := 5266
def sales_second_month : ℕ := 5768
def sales_third_month : ℕ := 5922
def sales_sixth_month : ℕ := 4937
def required_average_sales : ℕ := 5600
def number_of_months : ℕ := 6

-- Sum of the first, second, third, and sixth month's sales
def total_sales_without_fourth_fifth : ℕ := sales_first_month + sales_second_month + sales_third_month + sales_sixth_month

-- Total sales required to achieve the average required
def required_total_sales : ℕ := required_average_sales * number_of_months

-- The sale in the fourth month should be calculated as follows
def sales_fourth_month : ℕ := required_total_sales - total_sales_without_fourth_fifth

-- Proof statement
theorem sale_in_fourth_month :
  sales_fourth_month = 11707 := by
  sorry

end sale_in_fourth_month_l684_684957


namespace sequences_probability_l684_684967

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Definition of b_n, the number of valid sequences of length n
-- Note: b_n = F_{n+2} by the given solution analysis
def b (n : ℕ) : ℕ := fibonacci (n + 2)

-- The probability calculation using b_n for n=12
def probability (n : ℕ) : ℚ := (b n : ℚ) / (2^n : ℚ)

theorem sequences_probability (p q : ℕ) (hp_prime : Nat.gcd p q = 1) :
  p = 377 ∧ q = 4096 ∧ ((probability 12) = (p : ℚ) / (q : ℚ)) →
  p + q = 4473 :=
by
  intro h
  cases h with hp hq
  rw [hp, hq]
  simp [probability, b, fibonacci]
  sorry

end sequences_probability_l684_684967


namespace upgraded_video_card_multiple_l684_684005

noncomputable def multiple_of_video_card_cost (computer_cost monitor_cost_peripheral_cost base_video_card_cost total_spent upgraded_video_card_cost : ℝ) : ℝ :=
  upgraded_video_card_cost / base_video_card_cost

theorem upgraded_video_card_multiple
  (computer_cost : ℝ)
  (monitor_cost_ratio : ℝ)
  (base_video_card_cost : ℝ)
  (total_spent : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : monitor_cost_ratio = 1/5)
  (h3 : base_video_card_cost = 300)
  (h4 : total_spent = 2100) :
  multiple_of_video_card_cost computer_cost (computer_cost * monitor_cost_ratio) base_video_card_cost total_spent (total_spent - (computer_cost + computer_cost * monitor_cost_ratio)) = 1 :=
by
  sorry

end upgraded_video_card_multiple_l684_684005


namespace units_digit_of_50_factorial_l684_684176

theorem units_digit_of_50_factorial : 
  ∃ d, (d = List.prod (List.range 1 51)) ∧ (d % 10 = 0) :=
by
  sorry

end units_digit_of_50_factorial_l684_684176


namespace abs_diff_one_l684_684331

theorem abs_diff_one (a b c d : ℤ) (h : a + b + c + d = a * b + b * c + c * d + d * a + 1) :
  ∃ i j ∈ [a, b, c, d], |i - j| = 1 :=
by
  sorry

end abs_diff_one_l684_684331


namespace twenty_fifth_digit_sum_l684_684119

theorem twenty_fifth_digit_sum :
  let dec_1_5 := "0.2222222222222222222222..."
  let dec_1_6 := "0.1666666666666666666666..."
  let sum := "0.3888888888888888888888..."
  (25th_digit sum) = '8' :=
by {
  -- Assume necessary properties for dec_1_5, dec_1_6 and their sum
  -- Proof required to conclude that 25th digit in sum is '8'
  sorry
}

end twenty_fifth_digit_sum_l684_684119


namespace DiagonalsOfShapesBisectEachOther_l684_684558

structure Shape where
  bisect_diagonals : Prop

def is_parallelogram (s : Shape) : Prop := s.bisect_diagonals
def is_rectangle (s : Shape) : Prop := s.bisect_diagonals
def is_rhombus (s : Shape) : Prop := s.bisect_diagonals
def is_square (s : Shape) : Prop := s.bisect_diagonals

theorem DiagonalsOfShapesBisectEachOther (s : Shape) :
  is_parallelogram s ∨ is_rectangle s ∨ is_rhombus s ∨ is_square s → s.bisect_diagonals := by
  sorry

end DiagonalsOfShapesBisectEachOther_l684_684558


namespace all_sturns_are_toverics_and_quelds_l684_684396

-- Definitions of sets
variable (Toverics Sturns Vorks Quelds : Type)
variable {T Q S V : Toverics → Prop}
variable h1 : ∀ t : Toverics, Q t
variable h2 : ∀ s : Sturns, Q (coe s)
variable h3 : ∀ v : Vorks, T (coe v)
variable h4 : ∀ s : Sturns, T (coe s)

-- Theorem to prove
theorem all_sturns_are_toverics_and_quelds : ∀ s : Sturns, T (coe s) ∧ Q (coe s) :=
by
  assume s,
  sorry

end all_sturns_are_toverics_and_quelds_l684_684396


namespace pear_juice_processed_l684_684493

theorem pear_juice_processed
  (total_pears : ℝ)
  (export_percentage : ℝ)
  (juice_percentage_of_remainder : ℝ) :
  total_pears = 8.5 →
  export_percentage = 0.30 →
  juice_percentage_of_remainder = 0.60 →
  ((total_pears * (1 - export_percentage)) * juice_percentage_of_remainder) = 3.6 :=
by
  intros
  sorry

end pear_juice_processed_l684_684493


namespace length_of_EF_l684_684771

-- Define the basic setup of the rectangle
variables (A B C D E F : Point)
variable (AB BC : ℝ)
variables (abcd_rect : Rectangle A B C D)
-- Hypotheses based on problem conditions
hypothesis (h1 : AB = 4)
hypothesis (h2 : BC = 8)
-- Define the folding operation and resulting pentagon
variable (abe_folded : FoldAlong B A D A E F (Pentagon A B E F D))

-- Prove that EF = 4 √2
theorem length_of_EF :
  EF = 4 * Real.sqrt 2 :=
by
  sorry

end length_of_EF_l684_684771


namespace milly_wins_at_n_11_min_n_milly_wins_milly_wins_l684_684461

theorem milly_wins_at_n_11 : 
  ∀ (n : ℕ), 
  n < 11 → 
  (∀ f : ℕ → Prop, 
  (∀ x, 1 ≤ x ∧ x ≤ n → f x = red ∨ f x = blue) → 
  ¬(∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ 1 ≤ c ∧ c ≤ n ∧ 1 ≤ d ∧ d ≤ n ∧ f a = f b ∧ f b = f c ∧ f c = f d ∧ a + b + c = d)) :=
begin
  sorry
end

theorem min_n_milly_wins : 
  ∀ (n : ℕ), 
  n ≥ 11 → 
  ∃ f : ℕ → Prop, 
  ∀ x, 1 ≤ x ∧ x ≤ n → 
  (∀ a b c d : ℕ, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ 1 ≤ c ∧ c ≤ n ∧ 1 ≤ d ∧ d ≤ n → 
  (f a = f b ∧ f b = f c ∧ f c = f d → a + b + c = d)) :=
by sorry

noncomputable def milly_min_n : ℕ := 11

theorem milly_wins :
  ∀ n : ℕ, milly_min_n n ≤ 11 → 
  (∀ f : ℕ → Prop, 
  (∀ x, 1 ≤ x ∧ x ≤ n → f x = red ∨ f x = blue) → 
  (∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ 1 ≤ c ∧ c ≤ n ∧ 1 ≤ d ∧ d ≤ n ∧ f a = f b ∧ f b = f c ∧ f c = f d ∧ a + b + c = d)) :=
begin
  sorry
end

end milly_wins_at_n_11_min_n_milly_wins_milly_wins_l684_684461


namespace imaginary_part_z_l684_684381

theorem imaginary_part_z : 
  let z := (2 * Complex.i) / (1 + Complex.i) 
  in Complex.im z = 1 :=
by
  let z := (2 * Complex.i) / (1 + Complex.i)
  show Complex.im z = 1
  sorry

end imaginary_part_z_l684_684381
