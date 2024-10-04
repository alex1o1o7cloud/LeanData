import MathLib
import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.SinCos
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vector.Basic
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Init.Data.Real
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Computation
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.SolveByElim
import Mathlib.Topology.Basic
import Mathlib.Trigonometric.Basic

namespace find_b_value_l804_804297

noncomputable def triangle_ABC := (13 : ℕ, 24 : ℕ, 15 : ℕ)

def area_PQRS_expression (a b x : ℝ) := a * x - b * x^2

theorem find_b_value:
  let a := 24 * (13 / 48) in
  let b := 13 / 48 in
  (13 + 24 + 15) = 52 → 
  sqrt (26 * (26 - 13) * (26 - 15) * (26 - 24)) = 78 →
  (39 : ℝ) = area_PQRS_expression (12 * 24 * b) b 12 →
  b = 13 / 48 :=
by { intros, sorry }

end find_b_value_l804_804297


namespace proof_problem_l804_804269

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * Real.pi * x)

theorem proof_problem
  (a : ℝ)
  (h1 : ∀ x : ℝ, f (x - 1/2) = f (x + 1/2))
  (h2 : f (-1/4) = a) :
  f (9/4) = -a :=
by sorry

end proof_problem_l804_804269


namespace BC_passes_through_orthocenter_of_KDE_l804_804616

-- Define Triangle ABC
structure Triangle (α : Type*) :=
(A B C : α)

-- Assume a Point type and its different instances
variable (Point : Type*)

-- Given points B1 and C1
variables (A B C B1 C1 : Point)

-- Midpoints definition: specific to our setup
def is_midpoint (M A B : Point) : Prop :=
 ∃ (line_through_M_n : line_through M Ⓝ ⊆ line_through A Ⓑ ⋃ line_through B Ⓐ)

-- Definition of diameter of a circle
def is_diameter (circle : set Point) (A B : Point) : Prop :=
∃ (M : Point), is_midpoint M A B ∧ ∀ P : Point, P ∈ circle → (distance P M = √((distance P A)² + (distance P B)² / 4))

variables (ω1 ω2 : set Point)
variables (D E K : Point)

-- Assumptions
axiom h1 : is_midpoint B1 A C
axiom h2 : is_midpoint C1 A B
axiom h3 : is_diameter ω1 A B
axiom h4 : is_diameter ω2 A C
axiom h5 : intersection (line_through B1 C1) ω1 D
axiom h6 : intersection (line_through B1 C1) ω2 E
axiom h7 : intersection (line_through B D) (line_through C E) K

-- Conjecture
theorem BC_passes_through_orthocenter_of_KDE : 
  is_orthocenter K D E A B C B1 C1 ω1 ω2 D E K → line_through B C ⊆ orthocenter (triangle K D E) :=
sorry

end BC_passes_through_orthocenter_of_KDE_l804_804616


namespace distance_from_vertex_to_asymptote_l804_804077

noncomputable def hyperbola_distance_to_asymptote (x y : ℝ) : ℝ :=
    let a := real.sqrt 12 in
    let vertex := (2 * real.sqrt 3, 0) in
    let asymptote := λ x, (real.sqrt 3)/3 * x in
    let distance := (|2 * real.sqrt 3 * real.sqrt 3|) / (real.sqrt (3 + 9)) in
    distance

theorem distance_from_vertex_to_asymptote :
    hyperbola_distance_to_asymptote 2 (real.sqrt 3) = real.sqrt 3 :=
by
    -- Here the proof would go, but for now it is skipped with sorry
    sorry

end distance_from_vertex_to_asymptote_l804_804077


namespace rational_points_bound_l804_804244

theorem rational_points_bound (n : ℕ) (h_n_pos : 0 < n) :
  ∀ (I : set ℝ), (∃ a b, I = set.Ioo a b ∧ b - a = 1 / n) →
  ∃ (n_points : ℕ), n_points ≤ (n + 1) / 2 ∧
  ∀ points, (∀ p ∈ points, ∃ (r : ℚ), r ∈ I ∧ ∃ q, r.denom = q ∧ 1 ≤ q ∧ q ≤ n) →
  ∃ sub_points, set.finite sub_points ∧ set.card sub_points = n_points :=
by {
  sorry
}

end rational_points_bound_l804_804244


namespace regions_seven_lines_l804_804629

theorem regions_seven_lines (n : ℕ) (h_n : n = 7) : 
  ∀ (no_two_parallel : ∀ (i j : ℕ), i ≠ j → 
                          ¬ (lines i = lines j)) 
    (no_three_concurrent : ∀ (i j k : ℕ), 
                           i ≠ j → j ≠ k → i ≠ k → 
                           ¬ (lines i = lines j ∧ lines j = lines k)), 
  regions n = 29 := 
sorry

end regions_seven_lines_l804_804629


namespace length_of_platform_l804_804019

theorem length_of_platform (train_length : ℕ) (train_speed_kmph : ℕ) (crossing_time_s : ℕ) :
    (train_length = 450) →
    (train_speed_kmph = 108) →
    (crossing_time_s = 25) →
    let train_speed_mps := train_speed_kmph * 1000 / 3600 in
    let total_distance := train_speed_mps * crossing_time_s in
    let platform_length := total_distance - train_length in
    platform_length = 300 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end length_of_platform_l804_804019


namespace max_and_min_value_of_f_l804_804130

noncomputable def triangle_area (b c : ℝ) (θ : ℝ) := 0.5 * b * c * sin θ

noncomputable def satisfies_conditions (b c : ℝ) (θ : ℝ) : Prop :=
  (triangle_area b c θ = 3) ∧ (0 ≤ b * c * cos θ) ∧ (b * c * cos θ ≤ 6)

def is_in_theta_range (θ : ℝ) : Prop :=
  -- specify the actual range for θ based on tan(θ) >= 1
  sorry

noncomputable def f (θ : ℝ) : ℝ :=
  2 * sin θ ^ 2 - cos (2 * θ)

theorem max_and_min_value_of_f (b c θ : ℝ) (h : satisfies_conditions b c θ) :
  is_in_theta_range θ →
  (∃ (θ_max θ_min : ℝ), f θ_max = 3 ∧ f θ_min = 2) :=
by
  sorry

end max_and_min_value_of_f_l804_804130


namespace max_area_of_rectangle_l804_804929

theorem max_area_of_rectangle 
  (s : ℝ) 
  (hs : s ≥ 6) 
  (rect_area : ℝ := 4) 
  (circle_area : ℝ := π) : 
  let total_area := s^2 
  in total_area - (rect_area + circle_area) = 32 - π :=
by
  let total_area := s^2
  sorry

end max_area_of_rectangle_l804_804929


namespace Jessie_points_l804_804999

theorem Jessie_points (total_points team_points : ℕ) (players_points : ℕ) (P Q R : ℕ) (eq1 : total_points = 311) (eq2 : players_points = 188) (eq3 : team_points - players_points = 3 * P) (eq4 : P = Q) (eq5 : Q = R) : Q = 41 :=
by
  sorry

end Jessie_points_l804_804999


namespace XiaojunDescriptionIsCorrect_l804_804720

noncomputable def XiaojunData : Prop :=
  let available_numbers : List ℝ := [150, 1, 1.5, 1350, 50, 15]
  let weight := 50
  let height := 150
  let hours_washing_up := 1
  let distance_to_school := 1350
  let walking_speed := 1.5
  let time_to_school := 15
  weight ∈ available_numbers ∧
  height ∈ available_numbers ∧
  hours_washing_up ∈ available_numbers ∧
  distance_to_school ∈ available_numbers ∧
  walking_speed ∈ available_numbers ∧
  time_to_school ∈ available_numbers ∧
  ∀ x ∈ available_numbers, x == 50 ∨ x == 150 ∨ x == 1 ∨ x == 1350 ∨ x == 1.5 ∨ x == 15

theorem XiaojunDescriptionIsCorrect : XiaojunData :=
by
  unfold XiaojunData
  simp
  split
  exact List.mem_cons_of_mem 50 (List.single_mem_of_mem 50 [150, 1, 1.5, 1350, 15])
  split
  exact List.mem_cons 150 [1, 1.5, 1350, 50, 15]
  split
  exact List.mem_cons 1 [150, 1.5, 1350, 50, 15]
  split
  exact List.mem_cons 1350 [150, 1, 1.5, 50, 15]
  split
  exact List.mem_cons 1.5 [150, 1, 1350, 50, 15]
  split
  exact List.mem_cons 15 [150, 1, 1.5, 1350, 50]
  intro x hx
  rw ← List.mem_cons 50 [150, 1, 1.5, 1350, 15], at hx
  finish

end XiaojunDescriptionIsCorrect_l804_804720


namespace sum_of_squares_of_consecutive_integers_l804_804287

-- The sum of the squares of three consecutive positive integers equals 770.
-- We aim to prove that the largest integer among them is 17.
theorem sum_of_squares_of_consecutive_integers (n : ℕ) (h_pos : n > 0) 
    (h_sum : (n-1)^2 + n^2 + (n+1)^2 = 770) : n + 1 = 17 :=
sorry

end sum_of_squares_of_consecutive_integers_l804_804287


namespace minimum_cards_to_turn_over_l804_804790

def lower_case_on_one_side (c : Card) : Prop := sorry
def odd_integer_on_other_side (c : Card) : Prop := sorry
def even_integer_on_other_side (c : Card) : Prop := sorry

theorem minimum_cards_to_turn_over 
  (cards : List Card) 
  (h : ∀ c, lower_case_on_one_side c → odd_integer_on_other_side c) :
  ∃ k, k = 3 ∧ 
    (∃ (S : Finset Card), S.card = k ∧ 
      ∀ c ∈ S, lower_case_on_one_side c ∨ even_integer_on_other_side c) := 
sorry

end minimum_cards_to_turn_over_l804_804790


namespace jane_paid_five_l804_804528

noncomputable def cost_of_apple : ℝ := 0.75
noncomputable def change_received : ℝ := 4.25
noncomputable def amount_paid : ℝ := cost_of_apple + change_received

theorem jane_paid_five : amount_paid = 5.00 :=
by
  sorry

end jane_paid_five_l804_804528


namespace trig_expression_value_l804_804114

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 :=
by
  sorry

end trig_expression_value_l804_804114


namespace find_23_45_l804_804559

variable (a b : ℝ)

def operation (a b : ℝ) := a * b * 2 * b

axiom eq_ab : ∀ a b > 0, (a * b) * 2 * b = a * (b * 2 * b)
axiom eq_a1 : ∀ a > 0, a * 2 * a = a * 2 
axiom eq_11 : 1 * 2 = 2

theorem find_23_45 : (23 : ℝ) * 2 * (45 : ℝ) = 2070 := by
  sorry

end find_23_45_l804_804559


namespace part_I_part_II_l804_804893

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 6)

theorem part_I :
  ∃ A ω ϕ, A > 0 ∧ ω > 0 ∧ 0 ≤ ϕ ∧ ϕ < π ∧ (∀ x, f x = 2 * Real.sin (2 * x + π / 6)) ∧ 
           (∀ x ∈ set.Ici 0, f(x) <= 2) :=
by
  use 2, 2, π / 6
  sorry

theorem part_II {A B C b : ℝ} (hA: A > 0) (hf : f B = 1) (hb : b = 1) (triangle_ABC : ∀x ∈ set.Ici 0, 0 < A ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) :
  1 + Real.sqrt 3 ≤ (2 * Real.sin (A + π / 6) + 1) ∧ (2 * Real.sin (A + π / 6) + 1) ≤ 3 :=
by
  sorry

end part_I_part_II_l804_804893


namespace term_1500_of_sequence_l804_804440

theorem term_1500_of_sequence :
  (∀ (n : ℕ), n > 0 → (∑ i in finset.range n, (a i + 1) / n = n)
  → (a 1500 = 2999) :=
begin
  sorry
end

end term_1500_of_sequence_l804_804440


namespace initial_matchsticks_l804_804791

-- Define the problem conditions
def matchsticks_elvis := 4
def squares_elvis := 5
def matchsticks_ralph := 8
def squares_ralph := 3
def matchsticks_left := 6

-- Calculate the total matchsticks used by Elvis and Ralph
def total_used_elvis := matchsticks_elvis * squares_elvis
def total_used_ralph := matchsticks_ralph * squares_ralph
def total_used := total_used_elvis + total_used_ralph

-- The proof statement
theorem initial_matchsticks (matchsticks_elvis squares_elvis matchsticks_ralph squares_ralph matchsticks_left : ℕ) : total_used + matchsticks_left = 50 := 
by
  sorry

end initial_matchsticks_l804_804791


namespace area_inequality_l804_804827

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (S : ℝ)

theorem area_inequality (h : is_quadrilateral A B C D) (area_S : area_quadrilateral A B C D = S) :
  S ≤ (1 / 2) * (AB * CD + AD * BC) :=
sorry

end area_inequality_l804_804827


namespace math_problem_l804_804881

theorem math_problem (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^2 * b - 2 * a * b + a * b^2 = -1 :=
by
  sorry

end math_problem_l804_804881


namespace money_raised_is_correct_l804_804529

noncomputable def cost_per_dozen : ℚ := 2.40
noncomputable def selling_price_per_donut : ℚ := 1
noncomputable def dozens : ℕ := 10

theorem money_raised_is_correct :
  (dozens * 12 * selling_price_per_donut - dozens * cost_per_dozen) = 96 := by
sorry

end money_raised_is_correct_l804_804529


namespace side_length_rhombus_l804_804745
-- Importing the Mathlib library

-- Definitions of the given conditions and the theorem in Lean 4
variables {A B C D E F : Type} [Triangle A B C]
variables (ab_cd_common_angle_AC : ABC • c = AC)
variables (ab_cd_rhombus_side_length : Rhombus A D E F)

-- Theorem stating the equivalence
theorem side_length_rhombus
  (AB : ℝ) (AC : ℝ)
  (h_AB : AB = c) (h_AC : AC = b) :
    side_length_rhombus = (b * c) / (b + c) :=
by
  sorry

end side_length_rhombus_l804_804745


namespace simplify_expression_l804_804543

variable {a b c d : ℝ} (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0)

def x := b / c + c / b
def y := a / c + c / a
def z := a / b + b / a
def w := a / d + d / a

theorem simplify_expression : x^2 + y^2 + z^2 + w^2 - x * y * z * w = 8 :=
sorry

end simplify_expression_l804_804543


namespace weight_of_replaced_oarsman_l804_804260

noncomputable def average_weight (W : ℝ) : ℝ := W / 20

theorem weight_of_replaced_oarsman (W : ℝ) (W_avg : ℝ) (H1 : average_weight W = W_avg) (H2 : average_weight (W + 40) = W_avg + 2) : W = 40 :=
by sorry

end weight_of_replaced_oarsman_l804_804260


namespace evaluate_dollar_op_l804_804403

def dollar_op (x y : ℤ) := x * (y + 2) + 2 * x * y

theorem evaluate_dollar_op : dollar_op 4 (-1) = -4 :=
by
  -- Proof steps here
  sorry

end evaluate_dollar_op_l804_804403


namespace gary_earnings_l804_804817

theorem gary_earnings 
  (total_flour : ℕ)
  (cake_flour : ℝ)
  (cupcake_flour : ℝ)
  (cake_price : ℝ)
  (cupcake_price : ℝ)
  (leftover_flour : ℕ)
  (earnings : ℝ) 
  (h_flour : total_flour = 6)
  (h_cake_flour : cake_flour = 0.5)
  (h_cupcake_flour : cupcake_flour = 1 / 5)
  (h_cake_price : cake_price = 2.5)
  (h_cupcake_price : cupcake_price = 1)
  (h_leftover_flour : leftover_flour = 2)
  (h_earnings : earnings = 30) :
  earnings = (total_flour - leftover_flour) / cake_flour * cake_price + leftover_flour / cupcake_flour * cupcake_price :=
begin
  sorry
end

end gary_earnings_l804_804817


namespace problem_equiv_proof_l804_804819

variable (a b : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1)

theorem problem_equiv_proof :
  (2 ^ a + 2 ^ b ≥ 2 * Real.sqrt 2) ∧
  (Real.log a / Real.log 2 + Real.log b / Real.log 2 ≤ -2) ∧
  (a ^ 2 + b ^ 2 ≥ 1 / 2) :=
by
  sorry

end problem_equiv_proof_l804_804819


namespace find_tan_x_range_of_f_l804_804145

-- Definitions and conditions for question (1)
def a (ω x : ℝ) : ℝ × ℝ := ⟨1, cos (ω * x - π / 6)⟩
def b (ω x : ℝ) : ℝ × ℝ := ⟨sqrt 3, sqrt 3 * sin (ω * x - π / 6)⟩

-- Definitions for collinearity (vector parallelism)
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ u.1 = k * v.1 ∧ u.2 = k * v.2

theorem find_tan_x (x : ℝ) : 
  parallel (a 1 x) (b 1 x) → tan x = 2 + sqrt 3 := sorry

-- Definitions and conditions for question (2)
def omega : ℝ := 1
def f (x : ℝ) : ℝ := (a omega x - b omega x).1^2 + (a omega x - b omega x).2^2 - (sqrt 3 - 1)^2

theorem range_of_f (x : ℝ) : 
  (0 < x) ∧ (x < π / 2) → 
  (∀ y, y = f x → (0 ≤ y) ∧ (y < 3)) := sorry

end find_tan_x_range_of_f_l804_804145


namespace sum_YNRB_l804_804922

theorem sum_YNRB :
  ∃ (R Y B N : ℕ),
    (RY = 10 * R + Y) ∧
    (BY = 10 * B + Y) ∧
    (111 * N = (10 * R + Y) * (10 * B + Y)) →
    (Y + N + R + B = 21) :=
sorry

end sum_YNRB_l804_804922


namespace find_g_neg_one_l804_804654

theorem find_g_neg_one (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 1 / 2 → g x + g ((x + 2) / (2 - 4 * x)) = 3 * x) : 
  g (-1) = - 3 / 2 := 
sorry

end find_g_neg_one_l804_804654


namespace possible_numbers_l804_804605

theorem possible_numbers (N : ℕ) (h_digit : 10 ≤ N ∧ N ≤ 99)
  (h_claimed_divisors : ∀ d ∈ [3, 4, 5, 9, 10, 15, 18, 30], d ∣ N ∨ ¬ d ∣ N)
  (h_mistakes : Nat.countp (λ d, ¬ d ∣ N) [3, 4, 5, 9, 10, 15, 18, 30] = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 :=
by
  sorry

end possible_numbers_l804_804605


namespace area_of_region_bounded_by_sec_and_csc_l804_804801

theorem area_of_region_bounded_by_sec_and_csc (x y : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ 0 ≤ x ∧ 0 ≤ y) → 
  (∃ (area : ℝ), area = 1) :=
by 
  sorry

end area_of_region_bounded_by_sec_and_csc_l804_804801


namespace neznika_number_l804_804574

theorem neznika_number (N : ℕ) :
  10 ≤ N ∧ N ≤ 99 ∧
  (divisible_by N 3 ∧ divisible_by N 4 ∧ divisible_by N 5 ∧
   divisible_by N 9 ∧ divisible_by N 10 ∧ divisible_by N 15 ∧
   divisible_by N 18 ∧ divisible_by N 30) →
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end neznika_number_l804_804574


namespace sum_n_frac_form_l804_804939

theorem sum_n_frac_form (n : ℕ) : 
  ∑ k in Finset.range n, (1 : ℚ) / (k+1) / (k + 2) = (n : ℚ) / (n + 1) := 
by
  sorry

end sum_n_frac_form_l804_804939


namespace percentage_failed_english_l804_804510

variables (H B P E : ℝ)

theorem percentage_failed_english
 (hH : H = 0.2)
 (hB : B = 0.1)
 (hP : P = 0.2)
 (hE : 100% - P = 80%) -- equivalent to (100% - P) in English
 (hE_only : H - B + (E - B) + B = 0.8) :
 E = 0.7 := by
 sorry

end percentage_failed_english_l804_804510


namespace product_of_two_greatest_unattainable_scores_l804_804640

theorem product_of_two_greatest_unattainable_scores:
  ∀ (score1 score2 : ℕ), score1 = 31 → score2 = 39 → 
  (product_unattainable score1 score2 (λ n, ∃ a b c, n = 19 * a + 9 * b + 8 * c)) → 
  score1 * score2 = 1209 :=
by {
  intros score1 score2 h_score1 h_score2 h_unattainable,
  rw [h_score1, h_score2],
  norm_num,
  sorry
}

def product_unattainable (score1 score2: ℕ) (scorable: ℕ → Prop): Prop :=
  ∀ x, x ≥ min score1 score2 → scorable x

end product_of_two_greatest_unattainable_scores_l804_804640


namespace maximum_value_x_2y_2z_l804_804951

noncomputable def max_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : ℝ :=
  x + 2*y + 2*z

theorem maximum_value_x_2y_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : 
  max_sum x y z h ≤ 15 :=
sorry

end maximum_value_x_2y_2z_l804_804951


namespace championship_win_prob_l804_804258
open ProbabilityTheory

-- Define the probabilities and conditions
def prob_rain_win : ℚ := 2 / 3
def prob_sun_win : ℚ := 1 / 2

def prob_rain_game (k : ℕ) : ℚ :=
  (nat.choose 3 k) * (prob_rain_win ^ k) * ((1 - prob_rain_win) ^ (3 - k))

def prob_sun_game (j : ℕ) : ℚ :=
  (nat.choose 4 j) * (prob_sun_win ^ j) * ((1 - prob_sun_win) ^ (4 - j))

-- Calculate the probability that the Lions win at least 4 out of 7 games
noncomputable def total_prob_win : ℚ :=
  ∑ k in finset.range 4, ∑ j in finset.range 5, (if k + j ≥ 4 then prob_rain_game(k) * prob_sun_game(j) else 0)

-- Expected result in percentage
def LIONS_WIN_PROB_PERCENT : ℚ := 76

theorem championship_win_prob : (total_prob_win * 100).round = LIONS_WIN_PROB_PERCENT :=
by sorry  -- Skipping proof with sorry

end championship_win_prob_l804_804258


namespace trapezoid_area_l804_804176

-- Define the conditions and the final proof requirement
theorem trapezoid_area (AB CD : ℝ) (E : ℝ) 
  (area_abe : Real) (area_ade : Real) (ratio_de_be : ℝ) :
  AB = CD →
  area_abe = 75 →
  area_ade = 30 →
  ratio_de_be = 1/2 →
  let area_cde := 1/4 * area_abe in
  let area_bce := area_ade in
  let area_abcd := area_abe + area_ade + area_bce + area_cde in
  area_abcd = 153.75 :=
by
  intros h1 h2 h3 h4
  let area_cde := 1/4 * area_abe
  have h_cde : area_cde = 18.75, by norm_num [area_abe, h2]
  let area_bce := area_ade
  have h_bce : area_bce = 30, by norm_num [area_bce, h3]
  let area_abcd := area_abe + area_ade + area_bce + area_cde
  have h_abcd : area_abcd = 153.75, by norm_num [area_abe, area_ade, area_cde, area_bce, h2, h3, h_cde, h_bce]
  exact h_abcd

end trapezoid_area_l804_804176


namespace L_shaped_region_area_l804_804067

-- Define the conditions
def square_area (side_length : ℕ) : ℕ := side_length * side_length

def WXYZ_side_length : ℕ := 6
def XUVW_side_length : ℕ := 2
def TYXZ_side_length : ℕ := 3

-- Define the areas of the squares
def WXYZ_area : ℕ := square_area WXYZ_side_length
def XUVW_area : ℕ := square_area XUVW_side_length
def TYXZ_area : ℕ := square_area TYXZ_side_length

-- Lean statement to prove the area of the L-shaped region
theorem L_shaped_region_area : WXYZ_area - XUVW_area - TYXZ_area = 23 := by
  sorry

end L_shaped_region_area_l804_804067


namespace part_I_part_II_l804_804134

noncomputable def f (x a : ℝ) := 1/2 * x^2 + 2 * a * x
noncomputable def g (x a b : ℝ) := 3 * a^2 * real.log x + b

theorem part_I (a : ℝ) (b : ℝ) (h_a : a = real.exp 1) :
  ∃ x₀ : ℝ, (f x₀ a) = g x₀ a b ∧ (deriv (f x₀ a)) = deriv (g x₀ a b) → b = - (real.exp 1)^2 / 2 :=
    sorry

theorem part_II (a : ℝ) (h_a : 0 < a) :
  (∀ x : ℝ, 0 < x → f x a ≥ g x a 0) → a ≤ real.exp (5 / 6) :=
    sorry

end part_I_part_II_l804_804134


namespace savings_by_paying_cash_l804_804969

theorem savings_by_paying_cash
  (cash_price : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) (number_of_months : ℕ)
  (h1 : cash_price = 400) (h2 : down_payment = 120) (h3 : monthly_payment = 30) (h4 : number_of_months = 12) :
  cash_price + (monthly_payment * number_of_months - down_payment) - cash_price = 80 :=
by
  sorry

end savings_by_paying_cash_l804_804969


namespace complex_quadrant_l804_804919

open Complex

/-- If z = (1 + i) * (1 - 2i), then z lies in the fourth quadrant. -/
theorem complex_quadrant :
  let z := (1 + Complex.i) * (1 - 2 * Complex.i) in
  z.re > 0 ∧ z.im < 0 :=
by
  let z := (1 + Complex.i) * (1 - 2 * Complex.i)
  show z.re > 0 ∧ z.im < 0
  sorry

end complex_quadrant_l804_804919


namespace greatest_distance_C_D_l804_804514

noncomputable def C : set ℂ :=
  {w | w^4 = 16}

noncomputable def D : set ℂ :=
  {w | w^4 - 4 * w^3 + 16 = 0}

theorem greatest_distance_C_D : 
  ∃ (x ∈ C) (y ∈ D), dist x y = 6 :=
begin
  sorry
end

end greatest_distance_C_D_l804_804514


namespace lemonade_total_difference_is_1860_l804_804254

-- Define the conditions
def stanley_rate : Nat := 4
def stanley_price : Real := 1.50

def carl_rate : Nat := 7
def carl_price : Real := 1.30

def lucy_rate : Nat := 5
def lucy_price : Real := 1.80

def hours : Nat := 3

-- Compute the total amounts for each sibling
def stanley_total : Real := stanley_rate * hours * stanley_price
def carl_total : Real := carl_rate * hours * carl_price
def lucy_total : Real := lucy_rate * hours * lucy_price

-- Compute the individual differences
def diff_stanley_carl : Real := carl_total - stanley_total
def diff_stanley_lucy : Real := lucy_total - stanley_total
def diff_carl_lucy : Real := carl_total - lucy_total

-- Sum the differences
def total_difference : Real := diff_stanley_carl + diff_stanley_lucy + diff_carl_lucy

-- The proof statement
theorem lemonade_total_difference_is_1860 :
  total_difference = 18.60 :=
by
  sorry

end lemonade_total_difference_is_1860_l804_804254


namespace economical_club_l804_804294

-- Definitions of cost functions for Club A and Club B
def f (x : ℕ) : ℕ := 5 * x

def g (x : ℕ) : ℕ := if x ≤ 30 then 90 else 2 * x + 30

-- Theorem to determine the more economical club
theorem economical_club (x : ℕ) (hx : 15 ≤ x ∧ x ≤ 40) :
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 30 → f x > g x) ∧
  (30 < x ∧ x ≤ 40 → f x > g x) :=
sorry

end economical_club_l804_804294


namespace find_s_over_r_l804_804625

-- Define the function
def f (k : ℝ) : ℝ := 9 * k ^ 2 - 6 * k + 15

-- Define constants
variables (d r s : ℝ)

-- Define the main theorem to be proved
theorem find_s_over_r : 
  (∀ k : ℝ, f k = d * (k + r) ^ 2 + s) → s / r = -42 :=
by
  sorry

end find_s_over_r_l804_804625


namespace tree_classification_l804_804268

theorem tree_classification (circumference : ℝ) (width_ring_core width_ring_bark : ℝ) 
  (rings_arith_sequence : Prop) (age_classification : ℕ → string) : 
  (circumference = 3.14) → 
  (width_ring_core = 0.4) → 
  (width_ring_bark = 0.2) → 
  (age_classification 167 = "Third grade") :=
begin
  intros h_circ h_core h_bark h_class,
  -- Skipping proof steps; directly stating the result
  sorry
end

end tree_classification_l804_804268


namespace forty_sheep_forty_days_l804_804492

theorem forty_sheep_forty_days (one_sheep_one_bag_in_40_days : ∀ (t : ℕ), t = 40 → 1) :
  let total_sheep := 40 in
  let days := 40 in
  total_sheep * one_sheep_one_bag_in_40_days(days) = 40 :=
by
  sorry

end forty_sheep_forty_days_l804_804492


namespace ants_no_collision_probability_l804_804987

noncomputable def probability_no_collision 
    (initial_ants : Finset Vertex)
    (graph : Vertex → Finset Vertex)
    (move_probability : Π v : Vertex, Finset (graph v) → Probability)
    (no_collision_prob : ℚ)
    (conditions : 
      -- Adjacent vertices conditions:
      (graph A = {B, C, D, E, F}) ∧ 
      (graph F = {A, B, C}) ∧
      (graph B = {A, C, D, E, F}) ∧
      (graph C = {A, B, D, E, F}) ∧
      (graph D = {A, B, C, E}) ∧
      (graph E = {A, B, C, D})) : Prop :=
    -- Only check initial setup and probability, assuming basic Lean/FOL non-constructive probability is used
    initial_ants = {A, B, C, D, E, F} ∧
    no_collision_prob = 240 / 15625 ∧ 
    true -- Placeholder for potential other prob. calculations within full proof framework

-- The goal is to show:
theorem ants_no_collision_probability : 
    probability_no_collision initial_ants graph move_probability (240 / 15625) 
    (begin 
      split, -- conditions for connections
      all_goals { try {assumption} },
    end) :=
    sorry

end ants_no_collision_probability_l804_804987


namespace total_rainbow_nerds_l804_804347

-- Definitions based on the conditions
def num_purple_candies : ℕ := 10
def num_yellow_candies : ℕ := num_purple_candies + 4
def num_green_candies : ℕ := num_yellow_candies - 2

-- The statement to be proved
theorem total_rainbow_nerds : num_purple_candies + num_yellow_candies + num_green_candies = 36 := by
  -- Using the provided definitions to automatically infer
  sorry

end total_rainbow_nerds_l804_804347


namespace sequence_x_value_l804_804520

theorem sequence_x_value
  (z y x : ℤ)
  (h1 : z + (-2) = -1)
  (h2 : y + 1 = -2)
  (h3 : x + (-3) = 1) :
  x = 4 := 
sorry

end sequence_x_value_l804_804520


namespace sum_of_extremes_eq_10_l804_804809

theorem sum_of_extremes_eq_10 : 
  let a := 1
  let b := 3
  let c := 7
  let d := 9
  in (a + d) = 10 := 
by
  let a := 1
  let b := 3
  let c := 7
  let d := 9
  show (a + d) = 10
  sorry

end sum_of_extremes_eq_10_l804_804809


namespace max_value_of_quadratic_l804_804088

-- Define the quadratic function
def f (x : ℝ) : ℝ := 12 * x - 4 * x^2 + 2

-- State the main theorem of finding the maximum value
theorem max_value_of_quadratic : ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = 11 := sorry

end max_value_of_quadratic_l804_804088


namespace positive_integers_no_common_factor_l804_804055

theorem positive_integers_no_common_factor (X Y Z : ℕ) 
    (X_pos : 0 < X) (Y_pos : 0 < Y) (Z_pos : 0 < Z)
    (coprime_XYZ : Nat.gcd (Nat.gcd X Y) Z = 1)
    (eqn : X * (Real.log 3 / Real.log 100) + Y * (Real.log 4 / Real.log 100) = Z^2) :
    X + Y + Z = 4 :=
sorry

end positive_integers_no_common_factor_l804_804055


namespace largest_of_seven_consecutive_numbers_l804_804712

theorem largest_of_seven_consecutive_numbers (avg : ℕ) (h : avg = 20) :
  ∃ n : ℕ, n + 6 = 23 := 
by
  sorry

end largest_of_seven_consecutive_numbers_l804_804712


namespace area_of_triangle_ABC_is_five_l804_804399

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_ABC_is_five :
  ∀ (A B C : ℝ × ℝ),
  let d_AB := dist A B
  let d_BC := dist B C
  (d_AB = 1) →
  (d_BC = 9) →
  triangle_area A B C = 5 :=
begin
  intros A B C d_AB d_BC h_AB h_BC,
  -- Definitions of A, B, and C according to conditions
  let A := (-1 : ℝ, 3 : ℝ),
  let B := (0 : ℝ, 4 : ℝ),
  let C := (9 : ℝ, 5 : ℝ),
  
  -- Using the given distances and the formula for triangle area to prove the theorem
  sorry
end

end area_of_triangle_ABC_is_five_l804_804399


namespace ratio_of_sum_of_terms_l804_804540

theorem ratio_of_sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 5 / 9) : S 9 / S 5 = 1 := 
  sorry

end ratio_of_sum_of_terms_l804_804540


namespace domain_of_function_l804_804264

theorem domain_of_function :
  ∀ x : ℝ, 3 * x - 2 > 0 ∧ 2 * x - 1 > 0 ↔ x > (2 / 3) := by
  intro x
  sorry

end domain_of_function_l804_804264


namespace num_values_sum_l804_804955

noncomputable def g : ℝ → ℝ :=
sorry

theorem num_values_sum (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - 2 * x * y - 2 * x + 2) :
  ∃ n s : ℕ, (n = 1 ∧ s = 3 ∧ n * s = 3) :=
sorry

end num_values_sum_l804_804955


namespace mu4_central_moment_l804_804775

-- Define the normal distribution central moment formula
def central_moments (k : ℕ) (σ : ℝ) : ℝ :=
  if odd k then 0 else
  let l := k / 2 in (list.prod (list.map (λ n, (2*n-1)) (list.range l))) * σ^k

-- Prove that for a given σ, the 4th central moment equals 3σ^4
theorem mu4_central_moment (σ : ℝ) : central_moments 4 σ = 3 * σ^4 :=
by sorry

end mu4_central_moment_l804_804775


namespace triangle_area_144_l804_804001

theorem triangle_area_144 
  (Q : Type) -- Point Q within the triangle DEF
  (triangle_DE : Type) -- triangle DEF
  (area1 area2 area3 : ℕ) 
  (h1 : area1 = 9) 
  (h2 : area2 = 16) 
  (h3 : area3 = 25) 
  (h_divides : ∀ (q ∈ triangle_DE), q = Q) -- Q divides DEF into 3 triangles with given areas
  : area triangle_DE = 144 :=
sorry

end triangle_area_144_l804_804001


namespace minimum_nails_required_to_fix_convex_polygon_l804_804736

variable (P : Type) [convex_polygon P] 

-- A fixing set is defined as a set of points that prevents the polygon from moving
def fixing_set (P : Type) [convex_polygon P] (S : set Point) : Prop :=
  ∀ translation_rotation : Transformation, (∀ p ∈ S, ¬fixed_point translation_rotation p) → fixed P translation_rotation

-- The main theorem stating the minimum number of nails required to fix any convex polygon P
theorem minimum_nails_required_to_fix_convex_polygon (P : Type) [convex_polygon P] : ∃ (S : set Point), fixing_set P S ∧ S.card = 4 := 
sorry

end minimum_nails_required_to_fix_convex_polygon_l804_804736


namespace log_expression_value_l804_804289

theorem log_expression_value
  (h₁ : x + (Real.log 32 / Real.log 8) = 1.6666666666666667)
  (h₂ : Real.log 32 / Real.log 8 = 1.6666666666666667) :
  x = 0 :=
by
  sorry

end log_expression_value_l804_804289


namespace mixed_number_sum_interval_l804_804036

def mixed_number_sum :=
  let frac1 := (2 + 3 / 9)
  let frac2 := (3 + 3 / 4)
  let frac3 := (5 + 3 / 25)
  let sum := frac1 + frac2 + frac3
  sum
  theorem mixed_number_sum_interval :
  let frac1 := 2 + 3 / 9 in 
  let frac2 := 3 + 3 / 4 in 
  let frac3 := 5 + 3 / 25 in 
  let sum := frac1 + frac2 + frac3 in 
  8 < sum ∧ sum < 9 :=
by
  sorry

end mixed_number_sum_interval_l804_804036


namespace right_triangle_with_same_color_l804_804065

def color := ℕ -- Representing colors as natural numbers (0 for blue, 1 for red)
def square_contour := Fin 8 -- Points labeled A_1, A_2, ..., A_8

def coloring (c : square_contour → color) : Prop := 
  ∀ i : square_contour, c i = 0 ∨ c i = 1

theorem right_triangle_with_same_color 
  (c : square_contour → color) 
  (h : coloring c) : 
  ∃ i j k : square_contour, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ c i = c j ∧ c j = c k ∧ 
  (  (i = 0 ∧ j = 2 ∧ k = 4) ∨
     (i = 2 ∧ j = 4 ∧ k = 6) ∨ 
     (i = 4 ∧ j = 6 ∧ k = 0) ∨ 
     (i = 0 ∧ j = 4 ∧ k = 6) ∨
     (i = 1 ∧ j = 3 ∧ k = 5) ∨ 
     (i = 3 ∧ j = 5 ∧ k = 7) ∨ 
     (i = 5 ∧ j = 7 ∧ k = 1) ∨ 
     (i = 1 ∧ j = 5 ∧ k = 7)
  ) :=
sorry

end right_triangle_with_same_color_l804_804065


namespace find_points_D_l804_804954

-- Assume the definitions required for the problem (pyramids, regularity, etc.)
variables {Point : Type*} [metric_space Point]

-- Conditions of the problem:
-- SABC is a regular triangular pyramid
def is_regular_triangular_pyramid (S A B C : Point) : Prop :=
  dist S A = dist S B ∧ dist S B = dist S C ∧ dist A B = dist B C ∧ dist B C = dist C A

-- D not equal to S
def point_not_equal (D S : Point) : Prop :=
  D ≠ S

-- Angles with respect to tetrahedron
def angle_S_XD (S X D : Point) : ℝ := sorry -- We would define the angle appropriately here

-- Condition equation 
def condition_eq (S A B C D : Point) : Prop :=
  |cos (angle_S_XD S A D) - 2 * cos (angle_S_XD S B D) - 2 * cos (angle_S_XD S C D)| = 3

-- Plane passing through S and perpendicular to the plane ABC
def plane_through_S_perpendicular_to_ABC (S A B C : Point) : set Point :=
  {D : Point | sorry -- Some hyperplane definition involving S, and normal to the plane ABC}

-- Finally, the theorem statement
theorem find_points_D (S A B C : Point) (D : Point)
  (h1 : is_regular_triangular_pyramid S A B C)
  (h2 : point_not_equal D S)
  (h3 : condition_eq S A B C D) :
  D ∈ plane_through_S_perpendicular_to_ABC S A B C :=
sorry

end find_points_D_l804_804954


namespace least_sum_of_meanly_set_l804_804351

-- Define a set A to be meanly
def is_meanly (A : Finset ℕ) : Prop :=
  ∀ (B : Finset ℕ), B ⊆ A → B.Nonempty →
  (∃ k : ℕ, k = B.card ∧ (B.sum (λ x, x)) % k = 0)

-- Define the least possible sum function
def least_possible_sum (n : ℕ) : ℕ :=
  let T := List.lcm (List.range (n - 1)).map Nat.succ in
  n + (n * (n - 1) / 2) * T

-- The main theorem
theorem least_sum_of_meanly_set (n : ℕ) (A : Finset ℕ) (hA : A.card = n) (h_meanly : is_meanly A) :
  A.sum (λ x, x) = least_possible_sum n :=
sorry

end least_sum_of_meanly_set_l804_804351


namespace collinearity_condition_l804_804140

def vector (α : Type) := prod α α

variables (k : ℝ) (a b : vector ℝ)

def a := (k, 2 : ℝ)
def b := (1, k : ℝ)

theorem collinearity_condition :
  (k = Real.sqrt 2) ↔ ∃ m : ℝ, m > 0 ∧ a = (λ (x : ℝ × ℝ), (m * x.1, m * x.2)) b :=
sorry

end collinearity_condition_l804_804140


namespace sequence_has_four_integers_l804_804398

-- Problem statement and given conditions
def starts_with (seq : List ℕ) (n : ℕ) : Prop := seq.head = n
def divide_by_two_sequence (n : ℕ) (num_terms : ℕ) : List ℕ :=
  List.init num_terms (λ i => n / (2^i))

-- The sequence starts at 9720 and each subsequent term is n / 2^k
def given_seq := divide_by_two_sequence 9720 4

-- Verification that exactly 4 integers are in given_seq
theorem sequence_has_four_integers :
  ∃ seq : List ℕ, starts_with seq 9720 ∧ seq = [9720, 4860, 2430, 1215] :=
by
  use divide_by_two_sequence 9720 4
  have h1 : starts_with (divide_by_two_sequence 9720 4) 9720, from rfl
  have h2 : divide_by_two_sequence 9720 4 = [9720, 4860, 2430, 1215], from rfl
  exact ⟨h1, h2⟩

end sequence_has_four_integers_l804_804398


namespace local_minimum_at_x_l804_804406

noncomputable def f (p q x : ℝ) : ℝ := x^3 + x^2 + p * x + q

theorem local_minimum_at_x (p q x : ℝ) (hp : 0 ≤ p) (hp1 : p < 1 / 3) :
  let x₁ := (-1 + real.sqrt (1 - 3 * p)) / 3 in
  ∃ (x₁ : ℝ), x₁ = (-1 + real.sqrt (1 - 3 * p)) / 3 ∧
  (∀ x, f' p q x₁ = 0 → (∀ y, f' p q (y) = 0 → f'' p q (y) > 0)) := sorry

end local_minimum_at_x_l804_804406


namespace complex_sequence_z6_l804_804058

def complex_sequence : ℕ → ℂ
| 0       := 1
| (n + 1) := (complex_sequence n)^2 - 1 + complex.I

theorem complex_sequence_z6 :
  complex_sequence 5 = -289 + 35 * complex.I ∧
  complex.abs (complex_sequence 5) = 291 :=
by
  sorry

end complex_sequence_z6_l804_804058


namespace rectangle_area_y_l804_804643

theorem rectangle_area_y (y : ℝ) (h_y_pos : y > 0)
  (h_area : (3 * y = 21)) : y = 7 :=
by
  sorry

end rectangle_area_y_l804_804643


namespace find_b_plus_m_l804_804687

-- Definitions of the constants and functions based on the given conditions.
variables (m b : ℝ)

-- The first line equation passing through (5, 8).
def line1 := 8 = m * 5 + 3

-- The second line equation passing through (5, 8).
def line2 := 8 = 4 * 5 + b

-- The goal statement we need to prove.
theorem find_b_plus_m (h1 : line1 m) (h2 : line2 b) : b + m = -11 :=
sorry

end find_b_plus_m_l804_804687


namespace function_positive_range_l804_804166

theorem function_positive_range (a : ℝ) (x : ℝ) (h : a ∈ Icc (-1 : ℝ) (1 : ℝ)) : 
  (x ∈ Iio 1 ∨ x ∈ Ioi 3) ↔ (x^2 + (a - 4) * x + (4 - 2 * a) > 0) :=
by
  sorry

end function_positive_range_l804_804166


namespace petals_per_rose_l804_804527

theorem petals_per_rose
    (roses_per_bush : ℕ)
    (bushes : ℕ)
    (bottles : ℕ)
    (oz_per_bottle : ℕ)
    (petals_per_oz : ℕ)
    (petals : ℕ)
    (ounces : ℕ := bottles * oz_per_bottle)
    (total_petals : ℕ := ounces * petals_per_oz)
    (petals_per_bush : ℕ := total_petals / bushes)
    (petals_per_rose : ℕ := petals_per_bush / roses_per_bush) :
    petals_per_oz = 320 →
    roses_per_bush = 12 →
    bushes = 800 →
    bottles = 20 →
    oz_per_bottle = 12 →
    petals_per_rose = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end petals_per_rose_l804_804527


namespace original_parabola_eq_l804_804683

theorem original_parabola_eq (x : ℝ) :
  ∃ (y : ℝ), (∃ (yt : ℝ), yt = 2x^2 ∧ yt = y - 3) ↔ y = 2(x + 1)^2 - 3 :=
begin
  sorry
end

end original_parabola_eq_l804_804683


namespace math_problem_l804_804313

theorem math_problem (x y : ℤ) (h1 : x = 12) (h2 : y = 18) : (x - y) * ((x + y) ^ 2) = -5400 := by
  sorry

end math_problem_l804_804313


namespace cross_section_area_correct_l804_804644

noncomputable def cross_sectional_area (S_upper S_lower m n : ℝ) : ℝ :=
  ( (n * real.sqrt S_upper + m * real.sqrt S_lower) / (m + n) ) ^ 2

theorem cross_section_area_correct (S_upper S_lower m n : ℝ) (h1 : 0 < S_upper) (h2 : 0 < S_lower) (h3 : 0 < m) (h4 : 0 < n) :
  cross_sectional_area S_upper S_lower m n = ( (n * real.sqrt S_upper + m * real.sqrt S_lower) / (m + n) ) ^ 2 :=
by
  sorry

end cross_section_area_correct_l804_804644


namespace problem_statement_l804_804279

def op (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem problem_statement : ((op 7 4) - 12) * 5 = 105 := by
  sorry

end problem_statement_l804_804279


namespace starting_number_eq_12_l804_804292

theorem starting_number_eq_12 : 
  ∃ x : ℕ, (∃ n : ℕ, x = 300 - 9 * n ∧ n = 32) ∧ ∀ k, 1 ≤ k ∧ k ≤ 33 → 9 * k + (x - 9) ≤ 300 ∧ (9 * k + (x - 9)) mod 9 = 0 :=
sorry

end starting_number_eq_12_l804_804292


namespace cannot_serve_as_basis_l804_804022

theorem cannot_serve_as_basis :
  ¬ ( (1, 3) ∧ (2, 6) ) = (basis \mathbb{R}^2) :=
begin
  sorry
end

end cannot_serve_as_basis_l804_804022


namespace no_positive_integer_solutions_l804_804980

theorem no_positive_integer_solutions (x y : ℕ) (h : x > 0 ∧ y > 0) : x^2 + (x+1)^2 ≠ y^4 + (y+1)^4 :=
by
  intro h1
  sorry

end no_positive_integer_solutions_l804_804980


namespace find_angle_TCD_l804_804380

/-- Lean statement for the given math problem -/

def is_isosceles_trapezoid (A B C D : Type) := sorry
def is_point_inside (T A B C D : Type) := sorry
def angle (X Y Z : Type) := sorry 

theorem find_angle_TCD (A B C D T : Type) 
  (h1: is_isosceles_trapezoid A B C D)
  (h2: angle A D C = 82)
  (h3: angle C A D = 41)
  (h4: is_point_inside T A B C D)
  (h5: CT = CD)
  (h6: AT = TD) :
  angle T C D = 38 :=
sorry

end find_angle_TCD_l804_804380


namespace probability_of_first_and_second_failure_l804_804688

noncomputable def compute_probability
  (p1 p2 p3 : ℝ)
  (h1 : p1 = 0.2)
  (h2 : p2 = 0.4)
  (h3 : p3 = 0.3) :
  ℝ :=
  let P_B1 := p1 * p2 * (1 - p3) in
  let P_B2 := p1 * p3 * (1 - p2) in
  let P_B3 := p2 * p3 * (1 - p1) in
  let P_A := P_B1 + P_B2 + P_B3 in
  P_B1 / P_A

theorem probability_of_first_and_second_failure (p1 p2 p3 : ℝ)
  (h1 : p1 = 0.2)
  (h2 : p2 = 0.4)
  (h3 : p3 = 0.3) :
  compute_probability p1 p2 p3 h1 h2 h3 = 0.3 :=
sorry

end probability_of_first_and_second_failure_l804_804688


namespace neznaika_mistake_correct_numbers_l804_804611

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end neznaika_mistake_correct_numbers_l804_804611


namespace domain_transform_l804_804844

variable (f : ℝ → ℝ)

theorem domain_transform (h : ∀ x, -1 ≤ x ∧ x ≤ 4 → ∃ y, f y = x) :
  ∀ x, 0 ≤ x ∧ x ≤ 5 / 2 → ∃ y, f y = 2 * x - 1 :=
sorry

end domain_transform_l804_804844


namespace actual_order_correct_l804_804436

-- Define the actual order of the students.
def actual_order := ["E", "D", "A", "C", "B"]

-- Define the first person's prediction and conditions.
def first_person_prediction := ["A", "B", "C", "D", "E"]
def first_person_conditions (pos1 pos2 pos3 pos4 pos5 : String) : Prop :=
  (pos1 ≠ "A") ∧ (pos2 ≠ "B") ∧ (pos3 ≠ "C") ∧ (pos4 ≠ "D") ∧ (pos5 ≠ "E") ∧
  (pos1 ≠ "B") ∧ (pos2 ≠ "A") ∧ (pos2 ≠ "C") ∧ (pos3 ≠ "B") ∧ (pos3 ≠ "D") ∧
  (pos4 ≠ "C") ∧ (pos4 ≠ "E") ∧ (pos5 ≠ "D")

-- Define the second person's prediction and conditions.
def second_person_prediction := ["D", "A", "E", "C", "B"]
def second_person_conditions (pos1 pos2 pos3 pos4 pos5 : String) : Prop :=
  ((pos1 = "D") ∨ (pos2 = "D") ∨ (pos3 = "D") ∨ (pos4 = "D") ∨ (pos5 = "D")) ∧
  ((pos1 = "A") ∨ (pos2 = "A") ∨ (pos3 = "A") ∨ (pos4 = "A") ∨ (pos5 = "A")) ∧
  (pos1 ≠ "D" ∨ pos2 ≠ "A") ∧ (pos2 ≠ "A" ∨ pos3 ≠ "E") ∧ (pos3 ≠ "E" ∨ pos4 ≠ "C") ∧ (pos4 ≠ "C" ∨ pos5 ≠ "B")

-- The theorem to prove the actual order.
theorem actual_order_correct :
  ∃ (pos1 pos2 pos3 pos4 pos5 : String),
    first_person_conditions pos1 pos2 pos3 pos4 pos5 ∧
    second_person_conditions pos1 pos2 pos3 pos4 pos5 ∧
    [pos1, pos2, pos3, pos4, pos5] = actual_order :=
by sorry

end actual_order_correct_l804_804436


namespace exponent_of_2_in_S_n_l804_804450

noncomputable def P_n (n : ℕ) (k : ℕ) : (ℕ → ℕ) :=
  λ x, ∑ (i : ℕ) in (finset.range n).filter (λ m, 1 ≤ 2*m+1 ∧ 2*m+1 ≤ n), 
       nat.choose n (2*k+1) * x^(n-2*k-1) * ((x^2-1)^(k-1))

def S_n (n : ℕ) : ℕ :=
  ∑ (x : ℕ) in finset.range (n+1),
    abs (P_n n x)

theorem exponent_of_2_in_S_n (n : ℕ) [fact (nat.prime.factorization.exponent (S_n n) 2 = (nat.prime.factorization.exponent n 2) + 1)] : 
  ∀n : ℕ, nat.prime.factorization.exponent (S_n n) 2 = nat.prime.factorization.exponent n 2 + 1 := sorry

end exponent_of_2_in_S_n_l804_804450


namespace jogging_hours_at_15_mph_l804_804766

theorem jogging_hours_at_15_mph (distance total_time speed1 speed2 : ℕ) (total_distance : ℚ) :
  distance = 160 ∧ total_time = 8 ∧ speed1 = 15 ∧ speed2 = 10 →
  let hours1 := (80 : ℚ) / 5 in
  hours1 = 16 / 5 :=
by
  intro h
  sorry

end jogging_hours_at_15_mph_l804_804766


namespace price_relationship_l804_804378

-- Define the conditions
variable (initial_price : ℂ) (x : ℂ) (time_years : ℕ)

-- Given conditions: initial_price = 30 (which means 30,000 yuan), time_years = 5
def initial_price_value : initial_price = 30 := sorry
def time_years_value : time_years = 5 := sorry

-- Define the relationship between y and x
def price_after_years (initial_price : ℂ) (x : ℂ) (t : ℕ) : ℂ :=
  initial_price * (1 - x / 100) ^ t

-- Proof statement
theorem price_relationship : price_after_years initial_price x time_years = 30 * (1 - x / 100) ^ 5 :=
by
  rw [initial_price_value, time_years_value]
  sorry

end price_relationship_l804_804378


namespace probability_units_digit_prime_correct_l804_804748

noncomputable def probability_units_digit_prime : ℚ :=
  let primes := {2, 3, 5, 7}
  let total_outcomes := 10
  primes.card / total_outcomes

theorem probability_units_digit_prime_correct :
  probability_units_digit_prime = 2 / 5 := by
  sorry

end probability_units_digit_prime_correct_l804_804748


namespace quadratic_has_two_distinct_real_roots_l804_804704

theorem quadratic_has_two_distinct_real_roots (a b c : ℝ) (h_eq : 2 * a * a - 6 * a = 7) : 
  (b^2 - 4 * a * c > 0) :=
by
  have h : 2 * (2 : ℝ) * (2 : ℝ) - 6 * (2 : ℝ) = 7
  sorry

end quadratic_has_two_distinct_real_roots_l804_804704


namespace num_integers_with_factors_l804_804154

theorem num_integers_with_factors (a b lcm : ℕ) (lower upper : ℕ) (h_lcm : lcm = Nat.lcm a b) :
  (36 = Nat.lcm 12 9) → (a = 12) → (b = 9) → (lower = 200) → (upper = 500) →
  (finset.filter (λ x, x % lcm = 0) (finset.Icc lower upper)).card = 8 :=
by
  sorry

end num_integers_with_factors_l804_804154


namespace ratio_of_PB_to_AB_l804_804904

-- Define the given problem in Lean
theorem ratio_of_PB_to_AB (A B C P : Type)
  [triangle : condition A B C P]
  (h1 : AC / CB = 2 / 3)
  (h2 : is_exterior_angle_bisector_of_C P A B) : 
  PB / AB = 2 / 1 :=
sorry

end ratio_of_PB_to_AB_l804_804904


namespace B_pow_150_l804_804199

noncomputable def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem B_pow_150 : B ^ 150 = 1 :=
by
  sorry

end B_pow_150_l804_804199


namespace g1_plus_gneg1_l804_804116

variable {R : Type*} [Field R]
variable {f g : R → R}

-- Conditions of the problem
def functional_equation (f g : R → R) : Prop :=
∀ x y : R, f (x - y) = f x * g y - g x * f y

axiom f_eq_neq_zero (f g : R → R) : f (-2) = f 1 ∧ f 1 ≠ 0

theorem g1_plus_gneg1 {f g : R → R} : functional_equation f g → f_eq_neq_zero f g → g 1 + g (-1) = -1 :=
by
  intro functional_eq f_eq_neq
  sorry

end g1_plus_gneg1_l804_804116


namespace cone_frustum_ratio_l804_804009

theorem cone_frustum_ratio (h r : ℝ) (h = 6) (r = 5) 
  (k : ℝ) (m n : ℕ) (coprime : nat.coprime m n) (k = m / n) 
  (areas_ratio : k = (painted_areas_of_smaller_cone_and_frustum h r) / 
                            (painted_areas_of_frustum h r))
  (volumes_ratio : k = (volumes_of_smaller_cone_and_frustum h r) / 
                               (volumes_of_frustum h r)) : 
  m + n = 20 :=
by
  sorry

end cone_frustum_ratio_l804_804009


namespace angle_p_q_l804_804944

noncomputable def angle_between (p q: ℝ^3) : ℝ :=
  real.arccos ((p.dot q) / (p.norm * q.norm))

theorem angle_p_q (p q r : ℝ^3)
  (hp : p.norm = 1)
  (hq : q.norm = 1)
  (hr : r.norm = 1)
  (h : p + 2 • q + 2 • r = 0) :
  angle_between p q = real.arccos (-1 / 4) :=
by
  sorry

end angle_p_q_l804_804944


namespace log_expression_value_l804_804389

theorem log_expression_value :
  let logBase : ℝ → ℝ → ℝ := λ b x, Real.log x / Real.log b
  let log3_2 := logBase 3 2
  let log2_3 := logBase 2 3
  let log2_27 := 3 * log2_3
  log3_2 * log2_27 = 3 :=
by
  sorry

end log_expression_value_l804_804389


namespace circle_diameter_from_area_l804_804693

theorem circle_diameter_from_area (A : ℝ) (h : A = 225 * Real.pi) : ∃ d : ℝ, d = 30 :=
  by
  have r := Real.sqrt (225)
  have d := 2 * r
  exact ⟨d, sorry⟩

end circle_diameter_from_area_l804_804693


namespace find_a_b_l804_804719

noncomputable def limit_condition (a b : ℝ) :=
  filter.tendsto (λ x : ℝ, (real.log (2 - x))^2 / (x^2 + a*x + b)) (nhds 1) (nhds 1)

theorem find_a_b : limit_condition (-2) 1 := 
by 
  unfold limit_condition 
  -- Further steps can involve manipulation and use of real analysis theorems
  -- Proof can be added here accordingly
  sorry

end find_a_b_l804_804719


namespace coin_combinations_l804_804770

theorem coin_combinations (X : ℕ) :
  (∑ n d q, if (5 * n + 10 * d + 25 * q = 1500) ∧ (n > 0) ∧ (d > 0) ∧ (q > 0) ∧ (2 * d + 5 * q < 300) then 1 else 0) = X :=
sorry

end coin_combinations_l804_804770


namespace product_of_x_y_l804_804915

variable (x y : ℝ)

-- Condition: EF = GH
def EF_eq_GH := (x^2 + 2 * x - 8 = 45)

-- Condition: FG = EH
def FG_eq_EH := (y^2 + 8 * y + 16 = 36)

-- Condition: y > 0
def y_pos := (y > 0)

theorem product_of_x_y : EF_eq_GH x ∧ FG_eq_EH y ∧ y_pos y → 
  x * y = -2 + 6 * Real.sqrt 6 :=
sorry

end product_of_x_y_l804_804915


namespace harolds_rent_is_700_l804_804873

/-- Define constants for the salaries and expenditures -/
def monthly_income : ℝ := 2500.00
def car_payment : ℝ := 300.00
def utilities : ℝ := 1/2 * car_payment
def groceries : ℝ := 50.00
def remaining_after_retirement : ℝ := 650.00

/-- Define the expected remaining after retirement given the conditions -/
def expected_remaining_after_retirement (rent : ℝ) : ℝ :=
  let remaining_before_retirement := 
    monthly_income - rent - car_payment - utilities - groceries in
  1/2 * remaining_before_retirement

/-- Prove Harold's rent is $700 given the conditions -/
theorem harolds_rent_is_700 (R : ℝ) (h1 : expected_remaining_after_retirement R = remaining_after_retirement) : R = 700 := 
  by sorry

end harolds_rent_is_700_l804_804873


namespace arithmetic_sequence_max_Sn_l804_804228

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers

def Sn (n : ℕ) : ℝ := (n * (a 1 + a n)) / 2 -- Define the sum of the first n terms of the arithmetic sequence

theorem arithmetic_sequence_max_Sn (h1 : Sn 15 > 0) (h2 : Sn 16 < 0) : ∃ n, Sn n = S_max n := by sorry

end arithmetic_sequence_max_Sn_l804_804228


namespace odd_square_not_sum_of_five_odd_squares_l804_804249

theorem odd_square_not_sum_of_five_odd_squares :
  ∀ (n : ℤ), (∃ k : ℤ, k^2 % 8 = n % 8 ∧ n % 8 = 1) →
             ¬(∃ a b c d e : ℤ, (a^2 % 8 = 1) ∧ (b^2 % 8 = 1) ∧ (c^2 % 8 = 1) ∧ (d^2 % 8 = 1) ∧ 
               (e^2 % 8 = 1) ∧ (n % 8 = (a^2 + b^2 + c^2 + d^2 + e^2) % 8)) :=
by
  sorry

end odd_square_not_sum_of_five_odd_squares_l804_804249


namespace min_x_plus_2y_l804_804216

theorem min_x_plus_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  x + 2 * y ≥ 3 + 8 * real.sqrt 2 :=
sorry

end min_x_plus_2y_l804_804216


namespace ratio_of_playground_area_to_total_landscape_area_l804_804273

theorem ratio_of_playground_area_to_total_landscape_area {B L : ℝ} 
    (h1 : L = 8 * B)
    (h2 : L = 240)
    (h3 : 1200 = (240 * B * L) / (240 * B)) :
    1200 / (240 * B) = 1 / 6 :=
sorry

end ratio_of_playground_area_to_total_landscape_area_l804_804273


namespace ava_planted_9_trees_l804_804030

theorem ava_planted_9_trees
  (L : ℕ)
  (hAva : ∀ L, Ava = L + 3)
  (hTotal : L + (L + 3) = 15) : 
  Ava = 9 :=
by
  sorry

end ava_planted_9_trees_l804_804030


namespace sum_of_solutions_to_the_equation_l804_804428

noncomputable def sum_of_real_solutions : ℚ := 
  have h : (∀ x : ℚ, (x - 3) * (x^2 - 12 * x) = (x - 6) * (x^2 + 5 * x + 2)) → 
             (∀ x : ℚ, 14 * x^2 - 64 * x - 12 = 0) := 
  by sorry,
  (32 : ℚ) / 7

theorem sum_of_solutions_to_the_equation :
  (∀ x : ℚ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x) ↔ 
            14 * x^2 - 64 * x - 12 = 0) → 
              sum_of_real_solutions = 32 / 7 :=
by sorry

end sum_of_solutions_to_the_equation_l804_804428


namespace days_y_worked_l804_804713

theorem days_y_worked 
  (W : ℝ) 
  (x_days : ℝ) (h1 : x_days = 36)
  (y_days : ℝ) (h2 : y_days = 24)
  (x_remaining_days : ℝ) (h3 : x_remaining_days = 18)
  (d : ℝ) :
  d * (W / y_days) + x_remaining_days * (W / x_days) = W → d = 12 :=
by
  -- Mathematical proof goes here
  sorry

end days_y_worked_l804_804713


namespace sin_870_equals_half_l804_804050

theorem sin_870_equals_half :
  sin (870 * Real.pi / 180) = 1 / 2 := 
by
  -- Angle simplification
  have h₁ : 870 - 2 * 360 = 150 := by norm_num,
  -- Sine identity application
  have h₂ : sin (150 * Real.pi / 180) = sin (30 * Real.pi / 180) := by
    rw [mul_div_cancel_left 150 (ne_of_lt (by norm_num : 0 < (Real.pi : ℝ)) : (Real.pi : ℝ) ≠ 0)],
    congr,
    norm_num,

  -- Sine 30 degrees value
  have h₃ : sin (30 * Real.pi / 180) = 1 / 2 := by norm_num,

  -- Combine results
  rw [mul_div_cancel_left 870 (ne_of_lt (by norm_num : 0 < (Real.pi : ℝ)) : (Real.pi : ℝ) ≠ 0), h₁, h₂, h₃],
  sorry

end sin_870_equals_half_l804_804050


namespace problem1_problem2_l804_804986

-- Problem 1
theorem problem1 (x : ℚ) (h : x = -1/3) : 6 * x^2 + 5 * x^2 - 2 * (3 * x - 2 * x^2) = 11 / 3 :=
by sorry

-- Problem 2
theorem problem2 (a b : ℚ) (ha : a = -2) (hb : b = -1) : 5 * a^2 - a * b - 2 * (3 * a * b - (a * b - 2 * a^2)) = -6 :=
by sorry

end problem1_problem2_l804_804986


namespace tetrahedron_volume_and_height_l804_804390

def A1 : ℝ × ℝ × ℝ := (0, -1, -1)
def A2 : ℝ × ℝ × ℝ := (-2, 3, 5)
def A3 : ℝ × ℝ × ℝ := (1, -5, -9)
def A4 : ℝ × ℝ × ℝ := (-1, -6, 3)

theorem tetrahedron_volume_and_height :
  let V := 1 / 6 * |matrix.det (matrix![
    [-2, 4, 6],
    [1, -4, -8],
    [-1, -5, 4]])| in
  let S := 1 / 2 * real.norm ((vector.cross ⟨-2, 4, 6⟩ ⟨1, -4, -8⟩)) in
  (V = 5 / 3) ∧ (3 * V / S = real.sqrt 5) :=
by
  sorry

end tetrahedron_volume_and_height_l804_804390


namespace neznaika_mistake_correct_numbers_l804_804607

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end neznaika_mistake_correct_numbers_l804_804607


namespace number_of_distinct_paintings_l804_804053

theorem number_of_distinct_paintings : ∀ (octagon : Fin 8 → ℕ), 
  (∀ i, 0 ≤ octagon i ∧ octagon i ≤ 2) → 
  (Finset.card (Finset.filter (λ x, octagon x = 0) Finset.univ) = 3) →
  (Finset.card (Finset.filter (λ x, octagon x = 1) Finset.univ) = 3) →
  (Finset.card (Finset.filter (λ x, octagon x = 2) Finset.univ) = 2) →
  by sorry where
  ∃ f : fin 8 → fin 8, 
  bijective f ∧ 
  (∀ i j, (octagon i = octagon (f i)) = (octagon j = octagon (f j))) :=
  sorry

end number_of_distinct_paintings_l804_804053


namespace ratio_division_l804_804879

theorem ratio_division
  (A B C : ℕ)
  (h : (A : ℚ) / B = 3 / 2 ∧ (B : ℚ) / C = 1 / 3) :
  (5 * A + 3 * B) / (5 * C - 2 * A) = 7 / 8 :=
by
  sorry

end ratio_division_l804_804879


namespace x1_x2_product_l804_804886

theorem x1_x2_product (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : x1^2 - 2006 * x1 = 1) (h3 : x2^2 - 2006 * x2 = 1) : x1 * x2 = -1 := 
by
  sorry

end x1_x2_product_l804_804886


namespace compare_y1_y2_l804_804614

-- Define the function
def f (x : ℝ) : ℝ := -3 * x + 1

-- Define the points
def y1 := f 1
def y2 := f 3

-- The theorem to be proved
theorem compare_y1_y2 : y1 > y2 :=
by
  -- Proof placeholder
  sorry

end compare_y1_y2_l804_804614


namespace characteristic_value_leq_l804_804382

theorem characteristic_value_leq (n : ℕ) (hn : n ≥ 2)
  (table : fin n → fin n → ℕ)
  (h1 : ∀ i j, 1 ≤ table i j ∧ table i j ≤ n^2)
  (h2 : ∀ i j, ∃ a b, a > b ∧ (table i j = a ∧ table (i + 1) j = b ∨ table i (j + 1) = b))
  (row_contains : ∃ i, ∀ j, ∃ k, table i j = n^2 - n + k + 1)
  (col_contains : ∃ j, ∀ i, ∃ k, table i j = n^2 - n + k + 1) :
  ∃ λ, λ ≤ (n + 1) / n := sorry

end characteristic_value_leq_l804_804382


namespace zongzi_unit_price_l804_804066

theorem zongzi_unit_price (uA uB : ℝ) (pA pB : ℝ) : 
  pA = 1200 → pB = 800 → uA = 2 * uB → pA / uA = pB / uB - 50 → uB = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end zongzi_unit_price_l804_804066


namespace possible_numbers_l804_804601

theorem possible_numbers (N : ℕ) (h_digit : 10 ≤ N ∧ N ≤ 99)
  (h_claimed_divisors : ∀ d ∈ [3, 4, 5, 9, 10, 15, 18, 30], d ∣ N ∨ ¬ d ∣ N)
  (h_mistakes : Nat.countp (λ d, ¬ d ∣ N) [3, 4, 5, 9, 10, 15, 18, 30] = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 :=
by
  sorry

end possible_numbers_l804_804601


namespace math_problem_statement_l804_804916

-- Conditions for part (I)
noncomputable def curve_C_polar : Prop := ∀ (ρ θ : ℝ), ρ^2 = 1 + 2 * ρ * (Real.cos θ)

-- The conversion to rectangular coordinates
noncomputable def curve_C_rect : Prop :=
  ∀ (x y : ℝ), (x - 1)^2 + y^2 = 2

-- Conditions for part (II)
noncomputable def line_through_P : Prop := ∀ (t α : ℝ), (x y : ℝ),
  x = 1 + t * (Real.cos α) ∧ y = (1/2) + t * (Real.sin α)

-- Statement combining all conditions and seeking the desired maximum value
noncomputable def max_abs_PA_PB : Prop := ∀ (P : ℝ × ℝ) (C : Prop), 
  (P = (1, 1 / 2)) ∧ (C = curve_C_rect) → 
  (∃ A B : ℝ × ℝ, (A ∈ C ∧ B ∈ C ∧ (|PA| ≠ |PB|)) → 
  (max (|1 / |PA| - 1 / |PB||) = 4 / 7))

-- The combined theorem stating both parts of the problem explicitly
theorem math_problem_statement : curve_C_polar → curve_C_rect → line_through_P → max_abs_PA_PB := by
  sorry

end math_problem_statement_l804_804916


namespace range_of_a_l804_804143

variable (a x : ℝ)

def p := abs (4 * x - 3) ≤ 1
def q := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- The translated proof statement
theorem range_of_a (h1 : p a x) (h2 : q a x) (h3 : ¬p a x → ¬q a x = false) (h4 : ¬q a x → ¬p a x = true) :
  0 ≤ a ∧ a ≤ 1 / 2 :=
sorry

end range_of_a_l804_804143


namespace possible_numbers_l804_804600

theorem possible_numbers (N : ℕ) (h_digit : 10 ≤ N ∧ N ≤ 99)
  (h_claimed_divisors : ∀ d ∈ [3, 4, 5, 9, 10, 15, 18, 30], d ∣ N ∨ ¬ d ∣ N)
  (h_mistakes : Nat.countp (λ d, ¬ d ∣ N) [3, 4, 5, 9, 10, 15, 18, 30] = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 :=
by
  sorry

end possible_numbers_l804_804600


namespace coordinates_of_point_P_l804_804121

theorem coordinates_of_point_P 
  (P : ℝ × ℝ)
  (h1 : P.1 < 0 ∧ P.2 < 0) 
  (h2 : abs P.2 = 3)
  (h3 : abs P.1 = 5) :
  P = (-5, -3) :=
sorry

end coordinates_of_point_P_l804_804121


namespace original_number_of_boys_l804_804646

theorem original_number_of_boys (n : ℕ) (W : ℕ) 
  (h1 : W = n * 35) 
  (h2 : W + 40 = (n + 1) * 36) 
  : n = 4 :=
sorry

end original_number_of_boys_l804_804646


namespace min_value_of_y_on_interval_l804_804276

noncomputable def y (x : ℝ) : ℝ := 9^x - 2 * 3^x + 2

theorem min_value_of_y_on_interval :
  ∃ x₀ ∈ set.Icc (-1 : ℝ) (1 : ℝ), ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), y x₀ = 1 ∧ y x ≥ 1 :=
by
  sorry

end min_value_of_y_on_interval_l804_804276


namespace correct_quotient_remainder_sum_l804_804486

theorem correct_quotient_remainder_sum :
  ∃ N : ℕ, (N % 23 = 17 ∧ N / 23 = 3) ∧ (∃ q r : ℕ, N = 32 * q + r ∧ r < 32 ∧ q + r = 24) :=
by
  sorry

end correct_quotient_remainder_sum_l804_804486


namespace ratio_of_canoes_to_kayaks_l804_804306

theorem ratio_of_canoes_to_kayaks 
    (canoe_cost kayak_cost total_revenue : ℕ) 
    (canoe_to_kayak_ratio extra_canoes : ℕ)
    (h1 : canoe_cost = 14)
    (h2 : kayak_cost = 15)
    (h3 : total_revenue = 288)
    (h4 : extra_canoes = 4)
    (h5 : canoe_to_kayak_ratio = 3) 
    (c k : ℕ)
    (h6 : c = k + extra_canoes)
    (h7 : c = canoe_to_kayak_ratio * k)
    (h8 : canoe_cost * c + kayak_cost * k = total_revenue) :
    c / k = 3 := 
sorry

end ratio_of_canoes_to_kayaks_l804_804306


namespace multiple_time_second_artifact_is_three_l804_804932

-- Define the conditions as Lean definitions
def months_in_year : ℕ := 12
def total_time_both_artifacts_years : ℕ := 10
def total_time_first_artifact_months : ℕ := 6 + 24

-- Convert total time of both artifacts from years to months
def total_time_both_artifacts_months : ℕ := total_time_both_artifacts_years * months_in_year

-- Define the time for the second artifact
def time_second_artifact_months : ℕ :=
  total_time_both_artifacts_months - total_time_first_artifact_months

-- Define the sought multiple
def multiple_second_first : ℕ :=
  time_second_artifact_months / total_time_first_artifact_months

-- The theorem stating the required proof
theorem multiple_time_second_artifact_is_three :
  multiple_second_first = 3 :=
by
  sorry

end multiple_time_second_artifact_is_three_l804_804932


namespace problem_l804_804855

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x >= 0 then Real.log x / Real.log 3 + m else 1 / 2017

theorem problem (m := -2) (h_root : f 3 m = 0):
  f (f 6 m - 2) m = 1 / 2017 :=
by
  sorry

end problem_l804_804855


namespace number_of_integers_between_cubes_l804_804151

theorem number_of_integers_between_cubes :
  let a := 10.8
  let b := 11.0
  let lower_bound := ⌈a^3⌉
  let upper_bound := ⌊b^3⌋
  lower_bound.number_of_integers_between upper_bound = 72 :=
by
  sorry

end number_of_integers_between_cubes_l804_804151


namespace quadratic_function_properties_l804_804849

open Real

theorem quadratic_function_properties :
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 * x ^ 2 - 4 * x + 3) ∧
   (∃ x, f' x = 0 ∧ f x = 1) ∧
   f 0 = 3 ∧ f 2 = 3) → 
  (0 < a ∧ a < 1 / 2 → ¬monotonic_on (λ x, 2 * x ^ 2 - 4 * x + 3) (set.Icc (2 * a) (a + 1))) ∧
  ((∀ x ∈ set.Icc (-3 : ℝ) 0, 2 * x ^ 2 - 4 * x + 3 > 2 * x + 2 * m + 1) → m < 1) :=
by
  sorry

end quadratic_function_properties_l804_804849


namespace correct_selection_l804_804293

-- Conditions setup
def num_products := 50
def num_selected := 5
def systematic_sample(option : Nat) : List Nat :=
  match option with
  | 1 => [5, 10, 15, 20, 25]
  | 2 => [5, 15, 20, 35, 40]
  | 3 => [5, 11, 17, 23, 29]
  | 4 => [10, 20, 30, 40, 50]
  | _ => []

-- The problem statement
theorem correct_selection :
  ∃ (option : Nat), option = 4 ∧ systematic_sample(option) = [10, 20, 30, 40, 50] :=
by {
  use 4,
  split,
  trivial,
  trivial
}

end correct_selection_l804_804293


namespace number_of_viewers_in_scientific_notation_l804_804020

noncomputable def sci_notation : ℝ := 1.63 * 10^10

theorem number_of_viewers_in_scientific_notation : (16_300_000_000 : ℝ) = sci_notation := by
  sorry

end number_of_viewers_in_scientific_notation_l804_804020


namespace num_trombone_players_l804_804917

def weight_per_trumpet := 5
def weight_per_clarinet := 5
def weight_per_trombone := 10
def weight_per_tuba := 20
def weight_per_drum := 15

def num_trumpets := 6
def num_clarinets := 9
def num_tubas := 3
def num_drummers := 2
def total_weight := 245

theorem num_trombone_players : 
  let weight_trumpets := num_trumpets * weight_per_trumpet
  let weight_clarinets := num_clarinets * weight_per_clarinet
  let weight_tubas := num_tubas * weight_per_tuba
  let weight_drums := num_drummers * weight_per_drum
  let weight_others := weight_trumpets + weight_clarinets + weight_tubas + weight_drums
  let weight_trombones := total_weight - weight_others
  weight_trombones / weight_per_trombone = 8 :=
by
  sorry

end num_trombone_players_l804_804917


namespace discriminant_irrational_l804_804744

-- Given conditions
variables {a b c : ℝ}

-- The quadratic trinomial function f(x)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition: The coefficient b is rational
axiom hb_rat : ∃ (q : ℚ), b = q

-- Condition: The quadratic trinomial has no real roots, i.e., its discriminant is negative
axiom no_real_roots : b^2 - 4 * a * c < 0

-- Condition: Among the numbers c and f(c), exactly one is irrational
axiom one_irrational : (irrational c ∧ ¬ irrational (f c)) ∨ (¬ irrational c ∧ irrational (f c))

-- The statement to prove: The discriminant of the trinomial cannot be rational
theorem discriminant_irrational : ¬ ∃ (q : ℚ), b^2 - 4 * a * c = q := 
by 
  sorry

end discriminant_irrational_l804_804744


namespace problem_statement_l804_804583

-- Definitions based on conditions
def two_digit_number (N : ℕ) := N >= 10 ∧ N < 100
def divisible_by (a b : ℕ) := a % b = 0
def mistaken_exactly (N : ℕ) := (if divisible_by N 3 then 0 else 1) +
                               (if divisible_by N 4 then 0 else 1) +
                               (if divisible_by N 5 then 0 else 1) +
                               (if divisible_by N 9 then 0 else 1) +
                               (if divisible_by N 10 then 0 else 1) +
                               (if divisible_by N 15 then 0 else 1) +
                               (if divisible_by N 18 then 0 else 1) +
                               (if divisible_by N 30 then 0 else 1)

-- Lean 4 statement
theorem problem_statement (N : ℕ) (h_two_digit : two_digit_number N) (h_mistaken : mistaken_exactly N = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 := sorry

end problem_statement_l804_804583


namespace lucas_mod_prime_zero_l804_804666

-- Define the Lucas sequence
def lucas : ℕ → ℕ
| 0 => 1       -- Note that in the mathematical problem L_1 is given as 1. Therefore we adjust for 0-based index in programming.
| 1 => 3
| (n + 2) => lucas n + lucas (n + 1)

-- Main theorem statement
theorem lucas_mod_prime_zero (p : ℕ) (hp : Nat.Prime p) : (lucas p - 1) % p = 0 := by
  sorry

end lucas_mod_prime_zero_l804_804666


namespace domain_of_y_l804_804265

noncomputable def domain_function : set ℝ :=
  {x : ℝ | (abs (x - 2) ≠ 0) ∧ (6 - x - x^2 ≥ 0)}

theorem domain_of_y : domain_function = set.Icc (-3 : ℝ) 2 \ {2} :=
by
  sorry

end domain_of_y_l804_804265


namespace neznika_number_l804_804572

theorem neznika_number (N : ℕ) :
  10 ≤ N ∧ N ≤ 99 ∧
  (divisible_by N 3 ∧ divisible_by N 4 ∧ divisible_by N 5 ∧
   divisible_by N 9 ∧ divisible_by N 10 ∧ divisible_by N 15 ∧
   divisible_by N 18 ∧ divisible_by N 30) →
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end neznika_number_l804_804572


namespace exists_point_on_circle_with_sum_distances_at_least_1983_l804_804236

theorem exists_point_on_circle_with_sum_distances_at_least_1983 
  (points : Fin 1983 → ℝ × ℝ) (circle_center : ℝ × ℝ) (circle_radius : ℝ) 
  (h_radius : circle_radius = 1) :
  ∃ (p : ℝ × ℝ), 
    (dist circle_center p = 1) ∧ 
    (∑ i, dist (points i) p) ≥ 1983 :=
by 
  sorry

end exists_point_on_circle_with_sum_distances_at_least_1983_l804_804236


namespace root_sum_eq_one_l804_804878

theorem root_sum_eq_one (b c : ℝ) : (2 + complex.I) * (2 - complex.I) = c ∧ (2 + complex.I) + (2 - complex.I) = -b → b + c = 1 :=
by
  sorry

end root_sum_eq_one_l804_804878


namespace longest_chord_radius_l804_804125

theorem longest_chord_radius (O : Type*) [metric_space O] [normed_group O] 
  (longest_chord_eq : 8 = 2 * (radius : ℝ)) :
  radius = 4 := 
by 
  sorry

end longest_chord_radius_l804_804125


namespace find_a_value_l804_804142

noncomputable def A (a : ℝ) : Set ℝ := {x | x = a}
noncomputable def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else {x | a * x = 1}

theorem find_a_value (a : ℝ) :
  (A a ∩ B a = B a) → (a = 1 ∨ a = -1 ∨ a = 0) :=
by
  intro h
  sorry

end find_a_value_l804_804142


namespace problem_real_numbers_l804_804493

theorem problem_real_numbers (a b c d r : ℝ) 
  (h1 : b + c + d = r * a) 
  (h2 : a + c + d = r * b) 
  (h3 : a + b + d = r * c) 
  (h4 : a + b + c = r * d) : 
  r = 3 ∨ r = -1 :=
sorry

end problem_real_numbers_l804_804493


namespace root_is_prime_l804_804158

theorem root_is_prime (m n : ℕ) (h1 : Prime n) (h2 : ∃ r s : ℕ, r ≠ s ∧ r + s = m ∧ r * s = n) :
  (Prime (classical.some h2).1 ∨ Prime (classical.some h2).2) :=
sorry

end root_is_prime_l804_804158


namespace cone_cross_section_is_conic_section_l804_804779

-- Definition of a cone in geometric space
structure Cone :=
  (base_plane : Set Point)
  (axis : Line)
  (apex : Point)

-- Definition of a plane intersecting a cone
structure IntersectingPlane :=
  (line_intersect_base : Line)
  (point_intersect_axis : Point)
  (intersecting_with_base : line_intersect_base ∈ base_plane)

-- Prove the cross-section is one of the conic sections
theorem cone_cross_section_is_conic_section (cone : Cone) (plane : IntersectingPlane)
  (intersection_axioms : ∀ (L : Line), L ∈ cone.axis → plane.point_intersect_axis ∈ L) :
  ∃ section : ConicSection, 
  section = IntersectionOfConeAndPlane cone plane :=
sorry

end cone_cross_section_is_conic_section_l804_804779


namespace geometric_series_sum_to_15_l804_804037

-- Define the geometric series sum up to the 15th term
theorem geometric_series_sum_to_15 :
  (∑ k in Finset.range 15, (2/3) ^ (k + 1)) = 28691078 / 14348907 :=
by
  sorry

end geometric_series_sum_to_15_l804_804037


namespace solution_for_n_l804_804797

-- Define the conditions
def divides_n (n x y : ℕ) : Prop := n ∣ x^n - y^n
def divides_n_square (n x y : ℕ) : Prop := n^2 ∣ x^n - y^n
def is_square_free (m : ℕ) : Prop := ∀ p : ℕ, p.prime → p^2 ∣ m → false

theorem solution_for_n (n : ℕ) : 
  (∀ x y : ℕ, 0 < x → 0 < y → divides_n n x y → divides_n_square n x y) →
  (∃ m : ℕ, is_square_free m ∧ (n = m ∨ n = 2 * m)) :=
by sorry

end solution_for_n_l804_804797


namespace sum_of_n_values_l804_804701

theorem sum_of_n_values (n_values : List ℤ) 
  (h : ∀ n ∈ n_values, ∃ k : ℤ, 24 = k * (2 * n - 1)) : n_values.sum = 2 :=
by
  -- Proof to be provided.
  sorry

end sum_of_n_values_l804_804701


namespace annual_income_of_A_l804_804277

def monthly_income_ratios (A_income B_income : ℝ) : Prop := A_income / B_income = 5 / 2
def B_income_increase (B_income C_income : ℝ) : Prop := B_income = C_income + 0.12 * C_income

theorem annual_income_of_A (A_income B_income C_income : ℝ)
  (h1 : monthly_income_ratios A_income B_income)
  (h2 : B_income_increase B_income C_income)
  (h3 : C_income = 13000) :
  12 * A_income = 436800 :=
by 
  sorry

end annual_income_of_A_l804_804277


namespace sum_of_infinite_geometric_series_l804_804388

theorem sum_of_infinite_geometric_series (a r : ℚ) (h_a : a = 1) (h_r : r = 2 / 3) (h_abs_r_lt : |r| < 1) : (a / (1 - r)) = 3 := 
by 
  rw [h_a, h_r]
  have h1 : 1 - (2 / 3) = 1 / 3 :=
    by linarith
  rw [h1, div_div_eq_div_mul, div_self]
  · linarith
  · norm_num
  sorry

end sum_of_infinite_geometric_series_l804_804388


namespace correct_propositions_l804_804854

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 0.5 * x^2

def f_prime (x : ℝ) : ℝ := Real.log x + 1 + x

def x0_extremum (x0 : ℝ) := f_prime x0 = 0

theorem correct_propositions (x0 : ℝ) (h_ext : x0_extremum x0) :
  (0 < x0 ∧ x0 < (1 : ℝ) / Real.exp 1) ∧
  (f x0 + x0 < 0) :=
sorry

end correct_propositions_l804_804854


namespace range_of_k_not_monotonic_on_subinterval_l804_804170

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - real.log x

theorem range_of_k_not_monotonic_on_subinterval :
  ∀ k : ℝ, (¬ (∀ x y ∈ set.Ioo (k-2) (k+1), x ≤ y → f x ≤ f y) ∨ ¬ (∀ x y ∈ set.Ioo (k-2) (k+1), x ≤ y → f y ≤ f x)) → (2 ≤ k ∧ k < 5/2) :=
sorry

end range_of_k_not_monotonic_on_subinterval_l804_804170


namespace coin_flip_problem_l804_804350

def count_named_sequences : ℕ :=
  let total_flips := 10
  let at_least_heads (flips : List Bool) : Bool :=
    let heads_count := flips.count (λ b => b = tt)
    heads_count ≥ 3
  let count_sequences : ℕ :=
    let first_flip_heads_sequences := (List.range 4).bind (λ h_count =>
      if h_count ≥ 3 then [ (tt, List.replicate 4 tt ++ flips) | flips <- List.replicate (total_flips - 5) ff] else [])
    let rest_sequences_without_condition := List.replicate (2 ^ (total_flips - 1)) ff
    first_flip_heads_sequences.length + rest_sequences_without_condition.length
  672

theorem coin_flip_problem : count_named_sequences = 672 := 
  sorry

end coin_flip_problem_l804_804350


namespace parallel_condition_l804_804113

theorem parallel_condition (A B C D E : Point) (hD : LiesOnExtension D B A) (hE : LiesOnExtension E C A) :
  (AD / AB = AE / AC) → (Parallel DE BC) := 
sorry

end parallel_condition_l804_804113


namespace smallest_nat_number_l804_804082

theorem smallest_nat_number (n : ℕ) (h1 : ∃ a, 0 ≤ a ∧ a < 20 ∧ n % 20 = a ∧ n % 21 = a + 1) (h2 : n % 22 = 2) : n = 838 := by 
  sorry

end smallest_nat_number_l804_804082


namespace concyclic_points_l804_804946

-- The definitions, conditions, and theorem statement

variables (A B C P Q C₁ B₁ : Type) [Point A] [Point B] [Point C] [Point P] [Point Q] [Point C₁] [Point B₁]

-- Conditions definition
def acute_triangle (A B C P Q C₁ B₁: Type) : Prop :=
  ∀ (A B C : Type), ∃ (P Q : Type), 
  (cyclic_quadrilateral A P B C₁) ∧ 
  (parallel Q C₁ C A) ∧ 
  (opposite_sides C₁ Q A B) ∧
  (cyclic_quadrilateral A P C B₁) ∧ 
  (parallel Q B₁ B A) ∧ 
  (opposite_sides B₁ Q A C)

-- Proof statement
theorem concyclic_points (A B C P Q C₁ B₁ : Type)
  [acute_triangle A B C ]
  (cyclic_quadrilateral A P B C₁) 
  (parallel Q C₁ C A) 
  (opposite_sides C₁ Q A B) 
  (cyclic_quadrilateral A P C B₁) 
  (parallel Q B₁ B A) 
  (opposite_sides B₁ Q A C) :
  concyclic B₁ C₁ P Q :=
sorry

end concyclic_points_l804_804946


namespace plane_total_distance_l804_804641

def distance (t : ℝ) : ℝ := 60 * t - (6 / 5) * t^2

theorem plane_total_distance : ∃ t : ℝ, distance t = 750 :=
by
  sorry

end plane_total_distance_l804_804641


namespace incorrect_statements_l804_804571

theorem incorrect_statements :
  (¬ ∀ (s : ℝ), (s > 0) → 3 * s^2 = (9 * s^2)) ∧
  (¬ ∀ (r : ℝ), (r > 0) → 3 * π * r^2 = (9 * π * r^2)) :=
by {
  sorry,
}

end incorrect_statements_l804_804571


namespace smallest_side_is_10_l804_804274

noncomputable def smallest_side_of_triangle (x : ℝ) : ℝ :=
    let side1 := 10
    let side2 := 3 * x + 6
    let side3 := x + 5
    min side1 (min side2 side3)

theorem smallest_side_is_10 (x : ℝ) (h : 10 + (3 * x + 6) + (x + 5) = 60) : 
    smallest_side_of_triangle x = 10 :=
by
    sorry

end smallest_side_is_10_l804_804274


namespace integer_sol_num_integer_solutions_l804_804061

theorem integer_sol (n : ℤ) : (n + complex.I)^6 ∈ ℤ ↔ n = 0 := sorry

theorem num_integer_solutions : {n : ℤ | (n + complex.I)^6 ∈ ℤ}.finite ∧
                                {n : ℤ | (n + complex.I)^6 ∈ ℤ}.to_finset.card = 1 :=
begin
  sorry
end

end integer_sol_num_integer_solutions_l804_804061


namespace correct_assignment_statement_l804_804755

def expr1 : Prop := ¬(3=3)
def expr2 : Prop := ∃ M : ℕ, M = -M
def expr3 : Prop := ¬(B=A=2)
def expr4 : Prop := ¬(x+y=0)

theorem correct_assignment_statement : (expr1 ∧ expr3 ∧ expr4) → expr2 :=
by sorry

end correct_assignment_statement_l804_804755


namespace probability_of_binomial_distribution_l804_804126

open ProbabilityTheory

variables (ξ : ℕ)

def binomial_distribution (n : ℕ) (p : ℚ) : Distribution ℕ :=
  Distribution.binomial n p

theorem probability_of_binomial_distribution :
  (ξ ~ (binomial_distribution 3 (1 / 3))) → P(ξ = 1) = 4 / 9 :=
by
  sorry

end probability_of_binomial_distribution_l804_804126


namespace quadratic_root_conditions_l804_804171

theorem quadratic_root_conditions (m : ℝ) (h_pos : m > 0) :
  ∃ a b c : ℝ, 
    a = m ∧ b = 2*m - 1 ∧ c = -m + 2 ∧
    (a ≠ 0) ∧ 
    let Δ := b^2 - 4*a*c in
    let vertex := -b / (2*a) in
    (Δ > 0) ∧
    (vertex < 1) ∧
    (m + (2*m - 1) - m + 2 > 0) → 
    m > (3 + Real.sqrt 7) / 4 :=
begin
  sorry
end

end quadratic_root_conditions_l804_804171


namespace twenty_b_minus_a_not_integer_l804_804941

namespace Proof

variables (a b : ℝ)

theorem twenty_b_minus_a_not_integer
  (h1 : a ≠ b)
  (h2 : ¬ ∃ x : ℝ, (x^2 + 20 * a * x + 10 * b = 0) ∧ (x^2 + 20 * b * x + 10 * a = 0)) :
  20 * (b - a) ∉ ℤ :=
sorry

end Proof

end twenty_b_minus_a_not_integer_l804_804941


namespace probability_of_3_consecutive_heads_in_4_tosses_l804_804738

theorem probability_of_3_consecutive_heads_in_4_tosses :
  (∃ p : ℚ, p = 3/16 ∧ probability (λ s : vector bool 4, is_3_consecutive_heads s) = p) :=
sorry

def is_3_consecutive_heads (v : vector bool 4) : Prop :=
  (v.head = tt ∧ v.tail.head = tt ∧ v.tail.tail.head = tt) ∨
  (v.head = tt ∧ v.tail.head = tt ∧ v.tail.tail.tail.head = tt) ∨
  (v.tail.head = tt ∧ v.tail.tail.head = tt ∧ v.tail.tail.tail.head = tt)

end probability_of_3_consecutive_heads_in_4_tosses_l804_804738


namespace math_proof_part_a_math_proof_part_b_l804_804004

-- define the structure of the problem
structure Quadrilateral (A B C D : Type) := 
( inscribed_in_circle : Prop )

structure Circle (center : Type) := 
(radius : ℝ)

structure TangentPoint (L F : Type) :=
(equal_radii : ℝ)
(touches : Circle -> Prop)

noncomputable def part_a (A B C D O O1 O2 L F : Type) [q : Quadrilateral A B C D] 
  (AL : ℝ) (CF : ℝ) (r1 r2 : ℝ) (Omega1 : Circle O1) (Omega2 : Circle O2) 
  (Tangent1 : TangentPoint L F) : Prop := 
  AL = real.sqrt 2 → 
  CF = 2 * real.sqrt 2 → 
  Tangent1.equal_radii = r1 → 
  Tangent1.equal_radii = r2 → 
  Omega1.radius = Omega2.radius → 
  Omega2.radius = 2

noncomputable def part_b (A B C D O O1 O2 L F : Type) [q : Quadrilateral A B C D] 
  (circumcenter : Type) (BDC : ℝ): Prop := 
  ∀ (circumcenter_condition : circumcenter = O2),
  BDC = real.arctan ((real.sqrt 3 - 1) / real.sqrt 2)

-- using the proof problem statements to create the Lean statement
theorem math_proof_part_a (A B C D O O1 O2 L F : Type) [q : Quadrilateral A B C D] 
  (AL : ℝ) (CF : ℝ) (r1 r2 : ℝ) (Omega1 : Circle O1) (Omega2 : Circle O2) 
  (Tangent1 : TangentPoint L F) : part_a A B C D O O1 O2 L F AL CF r1 r2 Omega1 Omega2 Tangent1 :=
sorry

theorem math_proof_part_b (A B C D O O1 O2 L F : Type) [q : Quadrilateral A B C D] 
  (circumcenter : Type) (BDC : ℝ): part_b A B C D O O1 O2 L F circumcenter BDC :=
sorry

end math_proof_part_a_math_proof_part_b_l804_804004


namespace woodworker_tables_l804_804377

theorem woodworker_tables (L C_leg C T_leg : ℕ) (hL : L = 40) (hC_leg : C_leg = 4) (hC : C = 6) (hT_leg : T_leg = 4) :
  T = (L - C * C_leg) / T_leg := by
  sorry

end woodworker_tables_l804_804377


namespace equilateral_triangle_CM_eq_five_l804_804107

theorem equilateral_triangle_CM_eq_five 
  {A B C K L M : Point} 
  (h_eq_triangle : is_equilateral_triangle A B C)
  (h_K_on_AB : is_on_segment K A B)
  (h_LM_on_BC : is_on_segment L B M ∧ is_on_segment M B C)
  (h_KL_KM_eq : dist K L = dist K M)
  (h_BL_eq_two : dist B L = 2)
  (h_AK_eq_three : dist A K = 3) 
  : dist C M = 5 := 
sorry

end equilateral_triangle_CM_eq_five_l804_804107


namespace area_ratio_of_triangles_l804_804183

-- Definitions to represent the conditions
def is_diameter (A B : Point) (O : Circle) : Prop := 
  A ∈ O ∧ B ∈ O ∧ diameter O = lineSegment A B

def is_parallel (line1 line2 : Line) : Prop := 
  ∃ (θ₁ θ₂ : Angle), line1.angle θ₁ ∧ line2.angle θ₂ ∧ θ₁ = θ₂

def intersects (line1 line2 : Line) (E : Point) : Prop := 
  E ∈ line1 ∧ E ∈ line2

def is_perpendicular_to (line1 line2 : Line) : Prop :=
  ∃ θ : Angle, line1.angle θ ∧ line2.angle (90° - θ)

variables {O : Circle} {A B C D E : Point}
variables {α : ℝ}

-- Mathematical statement to prove the ratio of areas
theorem area_ratio_of_triangles 
  (h1 : is_diameter A B O) 
  (h2 : is_parallel (lineSegment C D) (lineSegment A B))
  (h3 : intersects (lineSegment A C) (lineSegment B D) E)
  (h4 : ∠ AED = 90° - α) 
  : (area (triangle C D E)) / (area (triangle A B E)) = (sin α)^2 :=
sorry

end area_ratio_of_triangles_l804_804183


namespace exists_triangle_with_sqrt_sides_area_inequality_l804_804101

theorem exists_triangle_with_sqrt_sides (a b c S : ℝ) (h_triangle: a + b > c ∧ a + c > b ∧ b + c > a) :
  ∃ A1 B1 C1 : Type, 
    (A1 = (√a : ℝ)) ∧  
    (B1 = (√b : ℝ)) ∧  
    (C1 = (√c : ℝ)) ∧
    (A1 + B1 > C1 ∧ A1 + C1 > B1 ∧ B1 + C1 > A1) :=
by sorry

theorem area_inequality (a b c S : ℝ) (h_triangle: a + b > c ∧ a + c > b ∧ b + c > a) (h_area : S = √(s * (s - a) * (s - b) * (s - c)) where s = (a + b + c) / 2):
  let A1 := (√a : ℝ),
      B1 := (√b : ℝ),
      C1 := (√c : ℝ),
      s1 := (A1 + B1 + C1) / 2
  let S1 := (√(s1 * (s1 - A1) * (s1 - B1) * (s1 - C1)) : ℝ)
  in S1^2 ≥ (S * √3 / 4) :=
by sorry

end exists_triangle_with_sqrt_sides_area_inequality_l804_804101


namespace circle_distance_to_line_l804_804164

-- Definitions for the conditions
def circle_tangent_to_axes_center (a : ℝ) (h : a > 0) : (ℝ × ℝ) := (a, a)
def circle_tangent_to_axes_radius (a : ℝ) (h : a > 0) : ℝ := a

def passes_through (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

def distance_point_to_line (x y A B C : ℝ) : ℝ :=
  |A * x + B * y + C| / real.sqrt (A^2 + B^2)

-- Theorem statement
theorem circle_distance_to_line 
    (a : ℝ) (h : a > 0)
    (pt : ℝ × ℝ) (hpt : pt = (2, 1))
    (hpass : passes_through (circle_tangent_to_axes_center a h) (circle_tangent_to_axes_radius a h) pt) :
  distance_point_to_line a a 2 (-1) (-3) = 2 * real.sqrt 5 / 5 :=
begin
  sorry
end

end circle_distance_to_line_l804_804164


namespace function_properties_l804_804137

theorem function_properties :
  let f : ℝ → ℝ := λ x, x^2 + (Real.log x / Real.log 2) in
  (∀ x > 0, f (-x) = -f x ∧ f (-x) = f x) →
  (∀ x y > 0, x < y → f x < f y) :=
by
  intros f hf
  -- here would be the proof steps, but we use sorry to skip it
  sorry

end function_properties_l804_804137


namespace sin_870_correct_l804_804046

noncomputable def sin_870_eq_half : Prop :=
  sin (870 : ℝ) = 1 / 2

theorem sin_870_correct : sin_870_eq_half :=
by
  sorry

end sin_870_correct_l804_804046


namespace real_solution_count_l804_804424

-- Define the polynomial equation
def polynomial_eq (x : ℝ) : Prop := 
  (x^1010 + 1) * (sum (i in list.range 0 505, if i % 2 = 0 then x^(1008 - 2 * i) else 0)) = 1010 * x ^ 1009

-- Define the conditions
def conditions (x : ℝ) : Prop := 
  x ≠ 0 ∧ (x^1010 + 1 > 0) ∧ (sum (i in list.range 0 505, if i % 2 = 0 then x^(1008 - 2 * i) else 0) > 0)

-- The theorem statement
theorem real_solution_count : 
  ∃! x : ℝ, conditions x ∧ polynomial_eq x := 
sorry

end real_solution_count_l804_804424


namespace possible_numbers_l804_804586

theorem possible_numbers (N : ℕ) 
    (h1 : 10 ≤ N) (h2 : N ≤ 99)
    (h3 : (N % 3 = 0) ∧ (N % 4 = 0) ∧ (N % 5 = 0) ∧ (N % 9 = 0) ∧ (N % 10 = 0) ∧ 
          (N % 15 = 0) ∧ (N % 18 = 0) ∧ (N % 30 = 0) ∨ 
          (N % 3 ≠ 0) + (N % 4 ≠ 0) + (N % 5 ≠ 0) + (N % 9 ≠ 0) + 
          (N % 10 ≠ 0) + (N % 15 ≠ 0) + (N % 18 ≠ 0) + (N % 30 ≠ 0) = 4) :
   N = 36 ∨ N = 45 ∨ N = 72 :=
by {
  sorry
}

end possible_numbers_l804_804586


namespace A_inter_B_is_empty_for_specific_a_l804_804478

def set_A (a : ℝ) : set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ (y - 3) / (x - 2) = a + 1}
def set_B (a : ℝ) : set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ (a^2 - 1) * x + (a - 1) * y = 15}

theorem A_inter_B_is_empty_for_specific_a : ∀ a : ℝ, a ∈ {-1, -4, 1, 5 / 2} → set_A a ∩ set_B a = ∅ := 
by
  exact sorry

end A_inter_B_is_empty_for_specific_a_l804_804478


namespace solution_set_of_inequality_l804_804884

def f (x : ℝ) (a : ℝ) : ℝ := exp x + a * exp (-x)
def f_is_even (a : ℝ) : Prop := ∀ x : ℝ, f x a = f (-x) a
def a_value_is_one (a : ℝ) : Prop := f_is_even a → a = 1

theorem solution_set_of_inequality (x : ℝ) (a : ℝ) (h : f_is_even a) :
  a_value_is_one a →
  (f (x-1) 1 < exp 1 + exp (-1)) ↔ (0 < x ∧ x < 2) :=
by
  intro ha
  have ha := ha h
  rw [←ha] at h
  sorry

end solution_set_of_inequality_l804_804884


namespace non_matching_pairings_l804_804392

theorem non_matching_pairings : 
  let bowls := {red, blue, yellow, green, purple}
  let glasses := {red, blue, yellow, green, purple, orange}
  (∀ b ∈ bowls, ∀ g ∈ glasses, b ≠ g) → 
  let pairings := { (b, g) | b ∈ bowls ∧ g ∈ glasses ∧ b ≠ g }
  pairings.card = 20 := 
by
  sorry

end non_matching_pairings_l804_804392


namespace problem_statement_l804_804585

-- Definitions based on conditions
def two_digit_number (N : ℕ) := N >= 10 ∧ N < 100
def divisible_by (a b : ℕ) := a % b = 0
def mistaken_exactly (N : ℕ) := (if divisible_by N 3 then 0 else 1) +
                               (if divisible_by N 4 then 0 else 1) +
                               (if divisible_by N 5 then 0 else 1) +
                               (if divisible_by N 9 then 0 else 1) +
                               (if divisible_by N 10 then 0 else 1) +
                               (if divisible_by N 15 then 0 else 1) +
                               (if divisible_by N 18 then 0 else 1) +
                               (if divisible_by N 30 then 0 else 1)

-- Lean 4 statement
theorem problem_statement (N : ℕ) (h_two_digit : two_digit_number N) (h_mistaken : mistaken_exactly N = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 := sorry

end problem_statement_l804_804585


namespace max_min_diff_eq_four_l804_804839

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * |x - a|

theorem max_min_diff_eq_four (a : ℝ) (h_a : a ≥ 2) : 
    let M := max (f a (-1)) (f a 1)
    let m := min (f a (-1)) (f a 1)
    M - m = 4 :=
by
  sorry

end max_min_diff_eq_four_l804_804839


namespace equidistant_point_z_l804_804417

theorem equidistant_point_z :
  (∀ (A B C : ℝ × ℝ × ℝ), 
    B = (10, 0, -2) ∧ C = (9, -2, 1) ∧ (A = (0, 0, z)) →
    ∃ z : ℝ, (λ (A : ℝ × ℝ × ℝ), let (xA, yA, zA) := A in
               let (xB, yB, zB) := B in
               let (xC, yC, zC) := C in
               (xB - xA)^2 + (yB - yA)^2 + (zB - zA)^2 =
               (xC - xA)^2 + (yC - yA)^2 + (zC - zA)^2) (0,0,z) z = -3): sorry

end equidistant_point_z_l804_804417


namespace total_jokes_proof_l804_804937

-- Definitions of the conditions
def jokes_jessy_last_saturday : Nat := 11
def jokes_alan_last_saturday : Nat := 7
def jokes_jessy_next_saturday : Nat := 2 * jokes_jessy_last_saturday
def jokes_alan_next_saturday : Nat := 2 * jokes_alan_last_saturday

-- Sum of jokes over two Saturdays
def total_jokes : Nat := (jokes_jessy_last_saturday + jokes_alan_last_saturday) + (jokes_jessy_next_saturday + jokes_alan_next_saturday)

-- The proof problem
theorem total_jokes_proof : total_jokes = 54 := 
by
  sorry

end total_jokes_proof_l804_804937


namespace problem_statement_l804_804466

noncomputable def curve := {p : ℝ × ℝ | let ⟨x, y⟩ := p in (x * real.abs x / (a^2)) - (y * real.abs y / (b^2)) = 1}

theorem problem_statement
  (a b : ℝ)
  (k m : ℝ)
  (x y : ℝ) :
  (∀ x, ∃! y, (x, y) ∈ curve) ∧
  (∀ x y, (x, y) ∈ curve -> ∀ P1 P2: ℝ × ℝ, P1 ∈ curve -> P2 ∈ curve -> 
    (y P1 - y P2) / (x P1 - x P2) > 0) ∧
  (∃ x : ℝ, ∃ y : ℝ, ∃ k : ℝ, ∃ m : ℝ, y = k * x + m ∧ (x, y) ∈ curve) := sorry

end problem_statement_l804_804466


namespace sequence_term_index_l804_804184

open Nat

noncomputable def arithmetic_sequence_term (a₁ d n : ℕ) : ℕ :=
a₁ + (n - 1) * d

noncomputable def term_index (a₁ d term : ℕ) : ℕ :=
1 + (term - a₁) / d

theorem sequence_term_index {a₅ a₄₅ term : ℕ}
  (h₁: a₅ = 33)
  (h₂: a₄₅ = 153)
  (h₃: ∀ n, arithmetic_sequence_term 21 3 n = if n = 5 then 33 else if n = 45 then 153 else (21 + (n - 1) * 3))
  : term_index 21 3 201 = 61 :=
sorry

end sequence_term_index_l804_804184


namespace triangle_inequality_at_vertex_l804_804563

-- Define the edge lengths of the tetrahedron and the common vertex label
variables {a b c d e f S : ℝ}

-- Conditions for the edge lengths and vertex label
axiom edge_lengths :
  a + b + c = S ∧
  a + d + e = S ∧
  b + d + f = S ∧
  c + e + f = S

-- The theorem to be proven
theorem triangle_inequality_at_vertex :
  a + b + c = S →
  a + d + e = S →
  b + d + f = S →
  c + e + f = S →
  (a ≤ b + c) ∧
  (b ≤ c + a) ∧
  (c ≤ a + b) ∧
  (a ≤ d + e) ∧
  (d ≤ e + a) ∧
  (e ≤ a + d) ∧
  (b ≤ d + f) ∧
  (d ≤ f + b) ∧
  (f ≤ b + d) ∧
  (c ≤ e + f) ∧
  (e ≤ f + c) ∧
  (f ≤ c + e) :=
sorry

end triangle_inequality_at_vertex_l804_804563


namespace find_m_eq_4_l804_804888

-- Define the initial condition
def b0 := Real.sin (Real.pi / 60) ^ 2

-- Define the recursive relationship
def b (m : ℕ) : ℝ
| 0     := b0
| (n+1) := 4 * (b n) * (1 - (b n))

-- Define the target property we want to prove
theorem find_m_eq_4 : ∃ m : ℕ, b m = b0 ∧ m = 4 := 
by 
  exists 4
  constructor
  sorry

end find_m_eq_4_l804_804888


namespace triangle_OAB_area_l804_804962

noncomputable def triangle_area (z1 z2 : ℂ) : ℝ :=
  let z2_minus_z1 := z2 - z1
  have h1 : z2_minus_z1 = (√3 * z1 * I) ∨ z2_minus_z1 = (-√3 * z1 * I) := sorry
  if h2 : |z1| = 4 then
  if h3 : 4 * z1^2 - 2 * z1 * z2 + z2^2 = 0 then
  have magnitude_abs : |z2_minus_z1| = √3 * |z1| := sorry,
  1 / 2 * |z1| * |z2_minus_z1|
  else 0
  else 0

theorem triangle_OAB_area
  (z1 z2 : ℂ)
  (h1 : |z1| = 4)
  (h2 : 4 * z1^2 - 2 * z1 * z2 + z2^2 = 0) :
  triangle_area z1 z2 = 8 * √3 := 
sorry

end triangle_OAB_area_l804_804962


namespace angle_A_is_70_l804_804190

-- Definitions of angles given as conditions in the problem
variables (BAD BAC ACB : ℝ)

def angle_BAD := 150
def angle_BAC := 80

-- The Lean 4 statement to prove the measure of angle ACB
theorem angle_A_is_70 (h1 : BAD = 150) (h2 : BAC = 80) : ACB = 70 :=
by {
  sorry
}

end angle_A_is_70_l804_804190


namespace permissible_m_values_l804_804823

theorem permissible_m_values :
  ∀ (m : ℕ) (a : ℝ), 
  (∃ k, 2 ≤ k ∧ k ≤ 4 ∧ (3 / (6 / (2 * m + 1)) ≤ k)) → m = 2 ∨ m = 3 :=
by
  sorry

end permissible_m_values_l804_804823


namespace workers_in_workshop_l804_804674

theorem workers_in_workshop (W : ℕ) (h1 : W ≤ 100) (h2 : W % 3 = 0) (h3 : W % 25 = 0)
  : W = 75 ∧ W / 3 = 25 ∧ W * 8 / 100 = 6 :=
by
  sorry

end workers_in_workshop_l804_804674


namespace problem_statement_l804_804778

-- Definitions based on the conditions
def point1 : (ℝ × ℝ) := (2, 7)
def point2 : (ℝ × ℝ) := (8, -5)
def check_point : (ℝ × ℝ) := (4, 3)
def lineB (x y : ℝ) : Prop := 5 * x - 6 * y - 2 = 0

-- The problem statement to prove
theorem problem_statement : 
  let trisection_point1 := ((point1.1 + (point2.1 - point1.1) / 3), (point1.2 + (point2.2 - point1.2) / 3)) in
    trisection_point1 = check_point ∧ lineB check_point.1 check_point.2 :=
by {
  -- Proof steps skipped
  sorry
}

end problem_statement_l804_804778


namespace f_97_equals_98_l804_804227

def f : ℕ → ℕ 
| x := if x ≥ 100 then x - 3 else f (f (x + 5))

theorem f_97_equals_98 : f 97 = 98 := 
by {
  sorry
}

end f_97_equals_98_l804_804227


namespace a_value_and_max_m_l804_804553

noncomputable def f (x a : ℝ) := (x + a) * Real.log x
noncomputable def g (x : ℝ) := (x^2) / Real.exp x
noncomputable def m (x a : ℝ) := min (f x a) (g x)

theorem a_value_and_max_m :
  (∀ a : ℝ, (deriv (f · a)) 1 = 2 → a = 1) ∧
  ∀ x a : ℝ, a = 1 →
  (∃ x_0 : ℝ, 1 < x_0 ∧ ∀ x : ℝ, m x 1 ≤ m 2 1 := 4 / Real.exp 2) :=
by
  sorry

end a_value_and_max_m_l804_804553


namespace cake_wedge_volume_l804_804737

noncomputable def volume_of_cylindrical_wedge (radius height : ℝ) : ℝ :=
  (π * radius^2 * height) / 4

theorem cake_wedge_volume:
  volume_of_cylindrical_wedge 8 10 ≈ 502.4 :=
sorry

end cake_wedge_volume_l804_804737


namespace product_of_slopes_eq_no_circle_passing_A_l804_804132

-- Defining the conditions
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
def Q : ℝ × ℝ := (-1, 0)

-- Point P (x0, y0) on the ellipse, given P ≠ A and P ≠ B
variable {x0 y0 : ℝ}
variable (hP : ellipse x0 y0)
variable (hPA : (x0, y0) ≠ A)
variable (hPB : (x0, y0) ≠ B)

-- Defining the slopes of lines PA and PB
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Defining points M and N on the ellipse where the line through Q intersects the ellipse
variable {t : ℝ}
variable {x1 y1 x2 y2 : ℝ}
variable (hMN1 : ellipse x1 y1)
variable (hMN2 : ellipse x2 y2)

-- Proof statements
theorem product_of_slopes_eq : slope (x0, y0) A * slope (x0, y0) B = -1 / 4 := sorry

theorem no_circle_passing_A : ¬∃ (x1 y1 x2 y2 : ℝ), ellipse x1 y1 ∧ ellipse x2 y2 ∧
  ((x1 - (-2))^2 + y1^2) = ((x2 - (-2))^2 + y2^2) ∧
  ∀ (M N : ℝ × ℝ), (M = (x1, y1) ∧ N = (x2, y2) ∧ (M ≠ A) ∧ (N ≠ A)) := sorry

end product_of_slopes_eq_no_circle_passing_A_l804_804132


namespace sticker_height_enlarged_l804_804031

theorem sticker_height_enlarged (orig_width orig_height new_width : ℝ)
    (h1 : orig_width = 3) (h2 : orig_height = 2) (h3 : new_width = 12) :
    new_width / orig_width * orig_height = 8 :=
by
  rw [h1, h2, h3]
  norm_num

end sticker_height_enlarged_l804_804031


namespace saturday_exclamation_l804_804028

/-- Define the exclamations for the sequence from Monday to Saturday
given the pattern identified in the solution. -/
def exclamation (n : ℕ) : String :=
  match n with
  | 0 => "A!"
  | 1 => "AU!"
  | 2 => "AUUA!"
  | 3 => "AUUAUAAU!"
  | _ => (exclamation (n - 1)).toList.reverse.map (λ c =>
    if c = 'A' then 'U'
    else if c = 'U' then 'A'
    else c
  ).asString ++ (exclamation (n - 1))

theorem saturday_exclamation : exclamation 5 = "AUUAUAAUUAUAUUAUAAUAUUAUAUUAUAAUAUUAUAUUAU" :=
by { sorry }

end saturday_exclamation_l804_804028


namespace odd_and_increasing_l804_804703

-- Define the function f(x) = e^x - e^{-x}
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- We want to prove that this function is both odd and increasing.
theorem odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
sorry

end odd_and_increasing_l804_804703


namespace max_height_l804_804003

def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height : ∃ t : ℝ, height t = 161 :=
sorry

end max_height_l804_804003


namespace train_clicks_l804_804283

/-- Considering the train speed in miles per hour, the length of each rail in feet, and the clicks heard 
when the train passes over the rail joints, we want to determine the time in seconds that corresponds 
to the speed of the train in miles per hour being equal to the number of clicks heard. -/
theorem train_clicks (speed_mph : ℝ) :
  (50 : ℝ) * (88 * speed_mph / 60) / (50 * (1.76 * speed_mph)) * 60 ≈ 34 := sorry

end train_clicks_l804_804283


namespace p_sufficient_not_necessary_q_l804_804825

-- Define the conditions p and q
def p (x : ℝ) : Prop := 2 < x ∧ x < 4
def q (x : ℝ) : Prop := x > 2 ∨ x < -3

-- Prove the relationship between p and q
theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l804_804825


namespace sum_of_x_coordinates_l804_804243

theorem sum_of_x_coordinates (a c : ℕ) (ha : 0 < a) (hc : 0 < c) 
    (h_intercept: -10 / a = -c / 4) :
    let x_values := [ -10 / a | (a, c) ∈ [(1, 40), (2, 20), (4, 10), (5, 8), (8, 5), (10, 4), (20, 2), (40, 1)] ] in 
    x_values.sum = -22.5 := sorry

end sum_of_x_coordinates_l804_804243


namespace points_coincide_after_6_steps_l804_804506

-- Definitions of the points and transformations
structure Point := (x : ℝ) (y : ℝ)

def reflect_across (p q : Point) : Point :=
  { x := 2 * q.x - p.x, y := 2 * q.y - p.y }

noncomputable def transform_A (A D B : Point) : Point := reflect_across A B
noncomputable def transform_D (D A C : Point) : Point := reflect_across D C

def initial_positions (A D B C : Point) :=
  (A.x - B.x) ^ 2 + (A.y - B.y) ^ 2 = 1 ∧
  (B.x - C.x) ^ 2 + (C.y - C.y) ^ 2 = 1 ∧
  (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 = 1 ∧
  (A.x - D.x) ^ 2 + (A.y - D.y) ^ 2 ≠ 1

def transform_steps (A0 D0 B C : Point) : (Point × Point) :=
  let A1 := transform_A A0 D0 B in
  let D1 := transform_D D0 A1 C in
  let A2 := transform_A A1 D1 B in
  let D2 := transform_D D1 A2 C in
  let A3 := transform_A A2 D2 B in
  let D3 := transform_D D2 A3 C in
  let A4 := transform_A A3 D3 B in
  let D4 := transform_D D3 A4 C in
  let A5 := transform_A A4 D4 B in
  let D5 := transform_D D4 A5 C in
  let A6 := transform_A A5 D5 B in
  let D6 := transform_D D5 A6 C in
  (A6, D6)

theorem points_coincide_after_6_steps (A D B C : Point) (h : initial_positions A D B C):
  let (A6, D6) := transform_steps A D B C in
  A6 = A ∧ D6 = D :=
sorry

end points_coincide_after_6_steps_l804_804506


namespace sin_870_correct_l804_804047

noncomputable def sin_870_eq_half : Prop :=
  sin (870 : ℝ) = 1 / 2

theorem sin_870_correct : sin_870_eq_half :=
by
  sorry

end sin_870_correct_l804_804047


namespace evaluate_expression_at_0_25_l804_804159

theorem evaluate_expression_at_0_25 : 
  let x := 0.25 in 
  625^(-x) + 25^(-2*x) + 5^(-4*x) = 3/5 :=
by
  sorry

end evaluate_expression_at_0_25_l804_804159


namespace correct_total_weight_6_moles_Al2_CO3_3_l804_804697

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

def num_atoms_Al : ℕ := 2
def num_atoms_C : ℕ := 3
def num_atoms_O : ℕ := 9

def molecular_weight_Al2_CO3_3 : ℝ :=
  (num_atoms_Al * atomic_weight_Al) +
  (num_atoms_C * atomic_weight_C) +
  (num_atoms_O * atomic_weight_O)

def num_moles : ℝ := 6

def total_weight_6_moles_Al2_CO3_3 : ℝ := num_moles * molecular_weight_Al2_CO3_3

theorem correct_total_weight_6_moles_Al2_CO3_3 :
  total_weight_6_moles_Al2_CO3_3 = 1403.94 :=
by
  unfold total_weight_6_moles_Al2_CO3_3
  unfold num_moles
  unfold molecular_weight_Al2_CO3_3
  unfold num_atoms_Al num_atoms_C num_atoms_O atomic_weight_Al atomic_weight_C atomic_weight_O
  sorry

end correct_total_weight_6_moles_Al2_CO3_3_l804_804697


namespace min_number_of_each_coin_l804_804569

def total_cost : ℝ := 1.30 + 0.75 + 0.50 + 0.45

def nickel_value : ℝ := 0.05
def dime_value : ℝ := 0.10
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50

def min_coins :=
  ∃ (n q d h : ℕ), 
  (n ≥ 1) ∧ (q ≥ 1) ∧ (d ≥ 1) ∧ (h ≥ 1) ∧ 
  ((n * nickel_value) + (q * quarter_value) + (d * dime_value) + (h * half_dollar_value) = total_cost)

theorem min_number_of_each_coin :
  min_coins ↔ (5 * half_dollar_value + 1 * quarter_value + 2 * dime_value + 1 * nickel_value = total_cost) :=
by sorry

end min_number_of_each_coin_l804_804569


namespace fractions_equivalent_iff_x_eq_zero_l804_804995

theorem fractions_equivalent_iff_x_eq_zero (x : ℝ) (h : (x + 1) / (x + 3) = 1 / 3) : x = 0 :=
by
  sorry

end fractions_equivalent_iff_x_eq_zero_l804_804995


namespace time_for_all_hoses_l804_804376

noncomputable def rate_X_Y_Z_to_fill_pool : ℝ → ℝ → ℝ → ℝ :=
  λ rate_X rate_Y rate_Z => 1 / (rate_X + rate_Y + rate_Z)

constant rate_X : ℝ
constant rate_Y : ℝ
constant rate_Z : ℝ

axiom rate_X_Y : rate_X + rate_Y = 1 / 3
axiom rate_X_Z : rate_X + rate_Z = 1 / 6
axiom rate_Y_Z : rate_Y + rate_Z = 1 / 5

theorem time_for_all_hoses :
  rate_X_Y_Z_to_fill_pool rate_X rate_Y rate_Z = 2.86 :=
by
  have h1 : rate_X + rate_Y = 1 / 3 := rate_X_Y
  have h2 : rate_X + rate_Z = 1 / 6 := rate_X_Z
  have h3 : rate_Y + rate_Z = 1 / 5 := rate_Y_Z
  sorry

end time_for_all_hoses_l804_804376


namespace find_x_l804_804668

theorem find_x (x : ℕ) (h1 : ∑ d in (finset.filter (λ d, x % d = 0) (finset.range (x + 1))), d = 56)
  (h2 : x % 4 = 0) (h3 : x % 2 = 0) : x = 28 :=
by
  sorry

end find_x_l804_804668


namespace median_bisects_and_parallel_l804_804104

theorem median_bisects_and_parallel
  (A B C M M3 A1 B1 : Point)
  (hM_CM3 : M ∈ line_segment C M3)  -- M is on the median CM3
  (hA1 : A1 ∈ line_segment B C)    -- A1 is on BC
  (hB1 : B1 ∈ line_segment A C)    -- B1 is on AC
  (hAM_A1 : A1 = line_intersection (line_through A M) (line_through B C)) -- AM intersects BC at A1 
  (hBM_B1 : B1 = line_intersection (line_through B M) (line_through A C)) -- BM intersects AC at B1
  (hCM3_bisects_AB : midpoint C AB M3) -- CM3 bisects AB at M3
  : (midpoint C A1 B1) ∧ (parallel A1 B1 A B) :=  -- CM3 bisects A1B1 and A1B1 is parallel to AB
sorry

end median_bisects_and_parallel_l804_804104


namespace max_omega_l804_804895

theorem max_omega (ω : ℕ) (T : ℝ) (h₁ : T = 2 * Real.pi / ω) (h₂ : 1 < T) (h₃ : T < 3) : ω = 6 :=
sorry

end max_omega_l804_804895


namespace solve_for_a_l804_804896

theorem solve_for_a : 
  ∀ (a : ℚ), 
  ∀ (x : ℚ), 
  (x + 2) * (x^2 - 5 * a * x + 1) = x^3 + (2 - 5 * a) * x^2 - 9 * a * x + 2 →
  (2 - 5 * a = 0 ∧ a = (2 / 5)) :=
begin
  sorry,
end

end solve_for_a_l804_804896


namespace exist_point_equal_angle_view_l804_804911

variable {Point : Type} [MetricSpace Point]

structure Circle (Point : Type) [MetricSpace Point] :=
(center : Point)
(radius : ℝ)

def circlesAtEqualAngles (k1 k2 : Circle Point) : Set Point := 
  {P : Point | dist P k1.center / dist P k2.center = k1.radius / k2.radius}

theorem exist_point_equal_angle_view (k1 k2 k3 : Circle Point)
  (h1 : ∀ x y, x ≠ y → dist k1.center k2.center > k1.radius + k2.radius)
  (h2 : ∀ x y, x ≠ y → dist k2.center k3.center > k2.radius + k3.radius)
  (h3 : ∀ x y, x ≠ y → dist k1.center k3.center > k1.radius + k3.radius) :
  ∃ P : Point, P ∈ circlesAtEqualAngles k1 k2 ∩ circlesAtEqualAngles k2 k3 :=
sorry

end exist_point_equal_angle_view_l804_804911


namespace sum_of_n_values_l804_804700

theorem sum_of_n_values (n_values : List ℤ) 
  (h : ∀ n ∈ n_values, ∃ k : ℤ, 24 = k * (2 * n - 1)) : n_values.sum = 2 :=
by
  -- Proof to be provided.
  sorry

end sum_of_n_values_l804_804700


namespace divisors_of_n_cubed_l804_804355

theorem divisors_of_n_cubed (n : ℕ) (h1 : ∃ p q : ℕ, p.prime ∧ q.prime ∧ (n = p^4 ∨ (n = p * q) ∧ p ≠ q)) :
  (∀ m : ℕ, Nat.num_divisors (n ^ 3) = m → (m = 13 ∨ m = 16)) :=
by
  sorry

end divisors_of_n_cubed_l804_804355


namespace problem_l804_804397

def expr : ℤ := 7^2 - 4 * 5 + 2^2

theorem problem : expr = 33 := by
  sorry

end problem_l804_804397


namespace triangle_angle_A_range_l804_804501

theorem triangle_angle_A_range (A B C : ℝ) (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π)
  (h_triangle : A + B + C = π)
  (h : sin^2 A ≤ sin^2 B + sin^2 C - sin B * sin C) :
  0 < A ∧ A ≤ π / 3 :=
by
  sorry

end triangle_angle_A_range_l804_804501


namespace smallest_a_plus_b_l804_804548

noncomputable def smallest_value (a b : ℝ) : ℝ :=
  a + b

theorem smallest_a_plus_b :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ x : ℝ, (x^2 + a * x + 3 * b = 0) ↔ discriminant (x^2 + a * x + 3 * b) ≥ 0) →
  (∀ x : ℝ, (x^2 + 2 * b * x + a = 0) ↔ discriminant (x^2 + 2 * b * x + a) ≥ 0) →
  smallest_value a b = 12 :=
by sorry

end smallest_a_plus_b_l804_804548


namespace range_of_k_l804_804285

noncomputable def f (x k : ℝ) : ℝ := 2^x + 3*x - k

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, 1 ≤ x ∧ x < 2 ∧ f x k = 0) ↔ 5 ≤ k ∧ k < 10 :=
by sorry

end range_of_k_l804_804285


namespace frosting_cupcakes_in_10_minutes_l804_804032

def speed_Cagney := 1 / 20 -- Cagney frosts 1 cupcake every 20 seconds
def speed_Lacey := 1 / 30 -- Lacey frosts 1 cupcake every 30 seconds
def speed_Jamie := 1 / 15 -- Jamie frosts 1 cupcake every 15 seconds

def combined_speed := speed_Cagney + speed_Lacey + speed_Jamie -- Combined frosting rate (cupcakes per second)

def total_seconds := 10 * 60 -- 10 minutes converted to seconds

def number_of_cupcakes := combined_speed * total_seconds -- Total number of cupcakes frosted in 10 minutes

theorem frosting_cupcakes_in_10_minutes :
  number_of_cupcakes = 90 := by
  sorry

end frosting_cupcakes_in_10_minutes_l804_804032


namespace saturday_exclamation_l804_804027

/-- Define the exclamations for the sequence from Monday to Saturday
given the pattern identified in the solution. -/
def exclamation (n : ℕ) : String :=
  match n with
  | 0 => "A!"
  | 1 => "AU!"
  | 2 => "AUUA!"
  | 3 => "AUUAUAAU!"
  | _ => (exclamation (n - 1)).toList.reverse.map (λ c =>
    if c = 'A' then 'U'
    else if c = 'U' then 'A'
    else c
  ).asString ++ (exclamation (n - 1))

theorem saturday_exclamation : exclamation 5 = "AUUAUAAUUAUAUUAUAAUAUUAUAUUAUAAUAUUAUAUUAU" :=
by { sorry }

end saturday_exclamation_l804_804027


namespace possible_numbers_l804_804587

theorem possible_numbers (N : ℕ) 
    (h1 : 10 ≤ N) (h2 : N ≤ 99)
    (h3 : (N % 3 = 0) ∧ (N % 4 = 0) ∧ (N % 5 = 0) ∧ (N % 9 = 0) ∧ (N % 10 = 0) ∧ 
          (N % 15 = 0) ∧ (N % 18 = 0) ∧ (N % 30 = 0) ∨ 
          (N % 3 ≠ 0) + (N % 4 ≠ 0) + (N % 5 ≠ 0) + (N % 9 ≠ 0) + 
          (N % 10 ≠ 0) + (N % 15 ≠ 0) + (N % 18 ≠ 0) + (N % 30 ≠ 0) = 4) :
   N = 36 ∨ N = 45 ∨ N = 72 :=
by {
  sorry
}

end possible_numbers_l804_804587


namespace odd_function_expression_l804_804653

theorem odd_function_expression {f : ℝ → ℝ}
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, 0 < x → f x = -x + 1) :
  ∀ x, x < 0 → f x = -x - 1 :=
by
  intro x hx
  have : 0 < -x := neg_pos.mpr hx
  rw [h1, h2 (-x) this]
  linarith

end odd_function_expression_l804_804653


namespace proof_problem_l804_804099

-- Definitions of the conditions
def log (b a : ℝ) : ℝ := Real.log a / Real.log b
def condition (a b : ℝ) : Prop := log (1/3) a + log 9 b = 0

-- Statements to be proven
def statement_B (a b : ℝ) : Prop := a * exp (Real.log a) = b
def statement_C (a b : ℝ) : Prop := b = a^2
def statement_D (a b : ℝ) : Prop := log 2 a = log 8 (a * b)

-- The final theorem
theorem proof_problem (a b : ℝ) (h : condition a b) : 
  statement_B a b ∧ 
  statement_C a b ∧ 
  statement_D a b := 
sorry

end proof_problem_l804_804099


namespace value_of_a_plus_b_l804_804496

theorem value_of_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, (x > -4 ∧ x < 1) ↔ (ax^2 + bx - 2 > 0)) → 
  a = 1/2 → 
  b = 3/2 → 
  a + b = 2 := 
by 
  intro h cond_a cond_b 
  rw [cond_a, cond_b]
  norm_num

end value_of_a_plus_b_l804_804496


namespace line_curve_intersect_l804_804517

noncomputable def parametric_equation (t : ℝ) : ℝ × ℝ :=
  let x := (√3 / 2) * t
  let y := 1 + (1 / 2) * t
  (x, y)

noncomputable def rectangular_equation (x y : ℝ) : Prop :=
  (y - 2)^2 + x^2 = 4

theorem line_curve_intersect
  (A B : ℝ × ℝ)
  (hA : rectangular_equation A.fst A.snd)
  (hB : rectangular_equation B.fst B.snd)
  (hPA : |A.fst - 0| + |A.snd - 1| = |A.snd - 1|)
  (hPB : |B.fst - 0| + |B.snd - 1| = |B.snd - 1|) :
  (1 / (real.sqrt ((A.snd - 2)^2 + (A.fst)^2))) + (1 / (real.sqrt ((B.snd - 2)^2 + (B.fst)^2))) = √13 / 3 := sorry

end line_curve_intersect_l804_804517


namespace cosine_sum_is_zero_l804_804965
-- Lean 4 statement version of the mathematical proof problem


-- Define the sequence a_i as the simplest proper fractions with denominator 60
def is_simplest_proper_fraction (a : ℚ) : Prop :=
  a.denom = 60 ∧ a.num.gcd(60) = 1

-- Define increasing sequence of positive terms
def is_increasing_sequence (seq : ℕ → ℚ) (n : ℕ) : Prop :=
  ∀ i j, (1 ≤ i ∧ i < j ∧ j ≤ n) → seq i < seq j

-- The main theorem to prove that the sum of cosines of the sequence is zero
theorem cosine_sum_is_zero (seq : ℕ → ℚ) (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → is_simplest_proper_fraction (seq i))
  (h2 : is_increasing_sequence seq n) :
  ∑ i in finset.range n, real.cos ((seq (i + 1)).val * real.pi) = 0 :=
sorry

end cosine_sum_is_zero_l804_804965


namespace equal_weight_piles_32_equal_weight_piles_22_l804_804325

theorem equal_weight_piles_32 : 
  (coins : Finset ℕ) → (h : coins.card = 32) → 
  ∃ (piles : Finset (Finset ℕ)), (piles.card = 2) ∧
  (∀ (pile ∈ piles), pile.card = 16) ∧ 
  (∀ (pile ∈ piles), ∃ (w : ℕ), ∀ (coin ∈ pile, coin = w)) :=
sorry

theorem equal_weight_piles_22 : 
  (coins : Finset ℕ) → (h : coins.card = 22) → 
  ∃ (piles : Finset (Finset ℕ)), (piles.card = 2) ∧
  (∀ (pile ∈ piles), pile.card = 11) ∧ 
  (∀ (pile ∈ piles), ∃ (w : ℕ), ∀ (coin ∈ pile, coin = w)) :=
sorry

end equal_weight_piles_32_equal_weight_piles_22_l804_804325


namespace number_of_unique_pairs_l804_804634

structure Person where
  isNextToFemale : Bool
  isNextToMale : Bool

-- Definition stating six people sitting at a round table
def sixPeopleRoundTable (people : List Person) : Prop := 
  people.length = 6  -- Ensure exactly 6 persons

-- Definitions for f and m
def f (people : List Person) : ℕ :=
  people.countp (·.isNextToFemale)

def m (people : List Person) : ℕ :=
  people.countp (·.isNextToMale)

-- Theorem to prove there are 5 unique (f, m) pairs
theorem number_of_unique_pairs :
  ∀ people : List Person, 
  sixPeopleRoundTable people → 
  ∃ pairs : Finset (ℕ × ℕ), pairs.card = 5 ∧ 
  ∀ p ∈ pairs, (f people, m people) = p :=
by
  sorry

end number_of_unique_pairs_l804_804634


namespace rectangular_block_squares_l804_804339

theorem rectangular_block_squares (total_squares : ℕ) (cut_out_squares : ℕ) (blocks : ℕ) (remaining_squares : ℕ) (block_size : ℕ) :
  total_squares = 64 → cut_out_squares = 2 → blocks = 30 → remaining_squares = total_squares - cut_out_squares → block_size = remaining_squares / blocks → (block_size = 2) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  simp at h4
  rw h4 at h5
  sorry

end rectangular_block_squares_l804_804339


namespace solveSystem_l804_804635

variable {r p q x y z : ℝ}

theorem solveSystem :
  
  -- The given system of equations
  (x + r * y - q * z = 1) ∧
  (-r * x + y + p * z = r) ∧ 
  (q * x - p * y + z = -q) →

  -- Solution equivalence using determined
  x = (1 - r ^ 2 + p ^ 2 - q ^ 2) / (1 + r ^ 2 + p ^ 2 + q ^ 2) :=
by sorry

end solveSystem_l804_804635


namespace pattern_E_cannot_be_formed_l804_804652

-- Define the basic properties of the tile and the patterns
inductive Tile
| rhombus (diag_coloring : Bool) -- representing black-and-white diagonals

inductive Pattern
| optionA
| optionB
| optionC
| optionD
| optionE

-- The given tile is a rhombus with a certain coloring scheme
def given_tile : Tile := Tile.rhombus true

-- The statement to prove
theorem pattern_E_cannot_be_formed : 
  ¬ (∃ f : Pattern → Tile, f Pattern.optionE = given_tile) :=
sorry

end pattern_E_cannot_be_formed_l804_804652


namespace conditional_probability_of_C_given_A_l804_804271

/-- 
Given:
- P(A|C) = 0.9
- P(not A|not C) = 0.9
- P(C) = 0.005

Prove: 
- P(C|A) = 9/208
-/
theorem conditional_probability_of_C_given_A :
  let P_A_given_C := 0.9
  let P_notA_given_notC := 0.9
  let P_C := 0.005
  let P_notC := 1 - P_C
  let P_A_given_notC := 1 - P_notA_given_notC
  let P_A := P_A_given_C * P_C + P_A_given_notC * P_notC
  P(C|A) = 9 / 208 :=
by
  sorry

end conditional_probability_of_C_given_A_l804_804271


namespace boundary_line_exists_l804_804860

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2
noncomputable def g (x : ℝ) : ℝ := Real.exp (Real.log x)

theorem boundary_line_exists :
  ∃ k m : ℝ, (∀ x : ℝ, f x ≥ k * x + m) ∧ (∀ x : ℝ, 0 < x → g x ≤ k * x + m) ∧ (k = Real.exp (1 / 2) ∧ m = -Real.exp 1 / 2) :=
by
  -- Condition 1: prove that f(x) ≥ kx + m for all x ∈ ℝ
  have h1 : ∃ k m : ℝ, ∀ x : ℝ, (1 / 2) * x^2 ≥ k * x + m,
  by sorry,
  
  -- Condition 2: prove that g(x) ≤ kx + m for all x ∈ (0, +∞)
  have h2 : ∃ k m : ℝ, ∀ x : ℝ, 0 < x → Real.exp (Real.log x) ≤ k * x + m,
  by sorry,

  -- Showing the specific values for k and m according to the solution steps
  use [Real.sqrt (Real.exp 1), -Real.exp 1 / 2],
  split;
  [exact h1, split; exact h2; sorry]

end boundary_line_exists_l804_804860


namespace painted_cells_possible_values_l804_804198

theorem painted_cells_possible_values (k l : ℕ) (hk : 2 * k + 1 > 0) (hl : 2 * l + 1 > 0) (h : k * l = 74) :
  (2 * k + 1) * (2 * l + 1) - 74 = 301 ∨ (2 * k + 1) * (2 * l + 1) - 74 = 373 := 
sorry

end painted_cells_possible_values_l804_804198


namespace sequence_sum_l804_804829

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def a1 : Prop := a 1 = 1
def a_rec (n : ℕ) (h : n > 0) : Prop := a n + a (n + 1) = (1/4)^n
def S_n (n : ℕ) : Prop := S n = ∑ i in range n, (4^i) * (a (i + 1))

-- Main statement to be proved
theorem sequence_sum (n : ℕ) (h : n > 0) (h_a1 : a1) (h_a_rec : ∀ m : ℕ, m > 0 → a_rec m) (h_S_n : S_n n) :
  S n - (4^n / 5) * (a n) = n / 5 := by
  sorry

end sequence_sum_l804_804829


namespace original_square_perimeter_l804_804369

theorem original_square_perimeter (p : ℕ) (x : ℕ) 
  (h1: p = 56) 
  (h2: 28 * x = p) : 4 * (2 * (x + 4 * x)) = 40 :=
by
  sorry

end original_square_perimeter_l804_804369


namespace find_constant_b_l804_804802

theorem find_constant_b (a b c : ℂ) :
  (∀ x : ℂ, (4 * x^3 - 2 * x + 5/2) * (a * x^3 + b * x^2 + c) =
           20 * x^6 - 8 * x^4 + 15 * x^3 - 5 * x^2 + 5) →
  b = -2 :=
begin
  intro h,
  sorry
end

end find_constant_b_l804_804802


namespace total_number_of_subsets_l804_804115

theorem total_number_of_subsets (X : set ℕ) :
  {1, 2} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5} → X.finite.card = 8 := by
  sorry

end total_number_of_subsets_l804_804115


namespace symmetric_y_axis_function_l804_804495

theorem symmetric_y_axis_function (f g : ℝ → ℝ) (h : ∀ (x : ℝ), g x = 3^x + 1) :
  (∀ x, f x = f (-x)) → (∀ x, f x = g (-x)) → (∀ x, f x = 3^(-x) + 1) :=
by
  intros h1 h2
  sorry

end symmetric_y_axis_function_l804_804495


namespace fourier_coeffs_l804_804256

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_cont : Continuous f)
variable (h_periodic : ∀ x, f (x + 2 * Real.pi) = f x)

-- Fourier coefficient definitions
noncomputable def alpha_0 : ℝ :=
  (1 / Real.pi) * ∫ x : ℝ in -Real.pi..Real.pi, f x

noncomputable def alpha_n (n : ℕ) : ℝ :=
  (1 / Real.pi) * ∫ x : ℝ in -Real.pi..Real.pi, f x * Real.cos (n * x)

noncomputable def beta_n (n : ℕ) : ℝ :=
  (1 / Real.pi) * ∫ x : ℝ in -Real.pi..Real.pi, f x * Real.sin (n * x)

-- Theorem statement to prove the equivalence
theorem fourier_coeffs (n : ℕ) : 
  ((1 / Real.pi) * ∫ x : ℝ in -Real.pi..Real.pi, f x) = alpha_0 ∧
  ((1 / Real.pi) * ∫ x : ℝ in -Real.pi..Real.pi, f x * Real.cos (n * x)) = alpha_n n ∧
  ((1 / Real.pi) * ∫ x : ℝ in -Real.pi..Real.pi, f x * Real.sin (n * x)) = beta_n n := by
  sorry

end fourier_coeffs_l804_804256


namespace find_unique_a_l804_804776

-- Define the arithmetic sequence and conditions
def arithmetic_seq_defined (a : ℝ) (n : ℕ) := a + (n - 1:ℕ) * d

-- Define the theorem statement based on the problem and solution
theorem find_unique_a (a d : ℝ)
  (h1 : a > 0)
  (h2 : ∀ n : ℕ, arithmetic_seq_defined a (n + 1) + (n + 1) = arithmetic_seq_defined a (n + 2) + (n + 2))
  (h3 : ∃q : ℝ, ∀t:ℕ, (q = d ↔ real.eq_one_plus (arithmetic_seq_defined a (t+1))- (arithmetic_seq_defined a (t)))) :
  a = (1/3) :=
by
  sorry

end find_unique_a_l804_804776


namespace number_of_games_l804_804673

theorem number_of_games (n : ℕ) (k : ℕ) (h_n : n = 25) (h_k : k = 12) :
  (n * (n - 1) / 2) * k = 3600 :=
by {
  have h1: n = 25, from h_n,
  have h2: k = 12, from h_k,
  rw [h1, h2],
  calc
    (25 * (25 - 1) / 2) * 12 = (25 * 24 / 2) * 12 : by simp
    ... = (25 * 24 / 2) * 12 : by simp
    ... = 300 * 12 : by norm_num
    ... = 3600 : by norm_num
}

end number_of_games_l804_804673


namespace g_eq_g_inv_at_2_l804_804059

def g (x : ℝ) : ℝ := 3 * x - 4
def g_inv (x : ℝ) : ℝ := (x + 4) / 3

theorem g_eq_g_inv_at_2 : g 2 = g_inv 2 := by
  sorry

end g_eq_g_inv_at_2_l804_804059


namespace remainder_of_4n_minus_6_l804_804889

theorem remainder_of_4n_minus_6 (n : ℕ) (h : n % 9 = 5) : (4 * n - 6) % 9 = 5 :=
sorry

end remainder_of_4n_minus_6_l804_804889


namespace expected_potato_yield_l804_804568

-- Define the problem and required conditions as constants
constant steps_to_feet : ℕ → ℕ
def steps_dim1 := 15
def steps_dim2 := 20
def step_length := 2
def yield_per_sq_ft := 1 / 2

-- Define the required values based on conditions
def length := steps_to_feet steps_dim1
def width := steps_to_feet steps_dim2

-- Convert steps to feet
def steps_to_feet (n : ℕ) : ℕ := n * step_length

-- Calculate the expected yield of potatoes
def area (length width : ℕ) : ℕ := length * width
def yield (area : ℕ) (yield_per_sq_ft : ℚ) : ℚ := area * yield_per_sq_ft

-- Prove the expected yield of potatoes is 600 pounds
theorem expected_potato_yield : yield (area length width) yield_per_sq_ft = 600 := by
  sorry

end expected_potato_yield_l804_804568


namespace other_root_of_quadratic_l804_804167

theorem other_root_of_quadratic (a : ℝ) (h : (3:ℝ) = 3) :
  (∃ x : ℝ, 3 + x = a ∧ 3 * x = -2 * a) → 
  ∃ x : ℝ, x = -6 / 5 :=
by {
  intros h1,
  sorry
}

end other_root_of_quadratic_l804_804167


namespace largest_interval_inequalities_l804_804078

theorem largest_interval_inequalities :
  ∃ M : Set ℝ, M = Set.Ici 2 ∧ 
    (∀ a b c d ∈ M, Real.sqrt (a * b) + Real.sqrt (c * d) ≥ Real.sqrt (a + b) + Real.sqrt (c + d)) ∧
    (∀ a b c d ∈ M, Real.sqrt (a * b) + Real.sqrt (c * d) ≥ Real.sqrt (a + c) + Real.sqrt (b + d)) := 
by
  use Set.Ici 2
  split
  · exact rfl
  split
  · intros a ha b hb c hc d hd
    -- Proof for the first inequality goes here
    sorry
  · intros a ha b hb c hc d hd
    -- Proof for the second inequality goes here
    sorry

end largest_interval_inequalities_l804_804078


namespace two_digit_number_possible_options_l804_804597

theorem two_digit_number_possible_options
  (N : ℕ)
  (h1 : 10 ≤ N ∧ N < 100)
  (h2 : (N % 3 = 0 ∨ N % 3 ≠ 0) ∧
        (N % 4 = 0 ∨ N % 4 ≠ 0) ∧
        (N % 5 = 0 ∨ N % 5 ≠ 0) ∧
        (N % 9 = 0 ∨ N % 9 ≠ 0) ∧
        (N % 10 = 0 ∨ N % 10 ≠ 0) ∧
        (N % 15 = 0 ∨ N % 15 ≠ 0) ∧
        (N % 18 = 0 ∨ N % 18 ≠ 0) ∧
        (N % 30 = 0 ∨ N % 30 ≠ 0)) :
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end two_digit_number_possible_options_l804_804597


namespace minimum_n_50_example_n_50_l804_804219

theorem minimum_n_50 (n : ℕ) (x : Fin n → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_sum : (∑ i, x i) = 1)
  (h_square_sum : (∑ i, (x i)^2) ≤ 1 / 50) :
  50 ≤ n :=
begin
  sorry
end

theorem example_n_50 : ∃ x : Fin 50 → ℝ, 
  (∀ i, 0 ≤ x i) ∧ 
  (∑ i, x i = 1) ∧ 
  (∑ i, (x i)^2 = 1 / 50) :=
begin
  let x := λ i : Fin 50, 1 / 50,
  use x,
  split,
  { intro i,
    exact zero_le_one },
  split,
  { simp [Finset.sum_const, Finset.card_univ, add_monoid_hom.map_mul, Finset.card_fin] },
  { simp [Finset.sum_const, Finset.card_univ, add_monoid_hom.map_mul, Finset.card_fin, mul_div_assoc, mul_comm, mul_left_comm, div_self, zero_le_one],
    norm_num }
end

end minimum_n_50_example_n_50_l804_804219


namespace max_growing_paths_product_l804_804263

-- Define the grid points
def grid_points : finset (ℕ × ℕ) := finset.univ.image (λ x, ((x / 5 : ℕ), (x % 5 : ℕ)))

-- Define the distance between two grid points
def distance (p1 p2 : ℕ × ℕ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define a growing path
def is_growing_path (path : list (ℕ × ℕ)) : Prop :=
  (∀ i < path.length - 1, distance (path.nth_le i (nat.lt_of_lt_pred i)) (path.nth_le (i + 1) (nat.lt_of_succ_lt (nat.lt_of_lt_pred i))) <
    distance (path.nth_le (i + 1) (nat.lt_of_succ_lt (nat.lt_of_lt_pred i))) (path.nth_le (i + 2) (nat.lt_of_succ_lt (nat.lt_of_lt_pred i)))) ∧
    list.nodup path

-- Define the problem statement
theorem max_growing_paths_product :
  let m := 15 in
  let r := 16 in
  m * r = 240 := by
  sorry

end max_growing_paths_product_l804_804263


namespace correct_property_of_f_l804_804470

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 - x)

theorem correct_property_of_f :
  ∃ x y : ℝ, f x = f y ∧ x = -y :=
by {
  use [x, -x],
  rw [f, f],
  simp,
  apply congr_fun,
  apply congr_arg,
  exact Real.cos,
  -- This is essentially all we need to prove symmetry
  sorry
}

end correct_property_of_f_l804_804470


namespace max_value_inequality_l804_804544

theorem max_value_inequality (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  3 * x + 4 * y + 6 * z ≤ Real.sqrt 53 := by
  sorry

end max_value_inequality_l804_804544


namespace find_op_p_q_sum_l804_804280

open Real

noncomputable theory
def perimeter (a b c : ℝ) : ℝ := a + b + c

variables (A P M O : Point) (r : ℝ)
variables (AP AM PM : Line)

-- Given conditions
axiom h1 : perimeter (dist A P) (dist P M) (dist M A) = 180
axiom h2 : angle A P M = π / 2
axiom h3 : ∃ (O : Point) (r : ℝ), r = 15 ∧ O ∈ line_through A P ∧ tangent AM (circle_centered O r) ∧ tangent PM (circle_centered O r)

-- Prove OP = 5
theorem find_op (h1 h2 h3 : Prop) : OP = 5 := sorry

-- If OP = p / q, p + q = 6
theorem p_q_sum : ∀ (p q : ℕ), ∀ (rel_prime : nat.coprime p q), OP = p / q → p + q = 6 := sorry


end find_op_p_q_sum_l804_804280


namespace monotonicity_of_f_when_a_leq_zero_find_a_for_f_monotonic_increasing_on_R_l804_804212

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x + a * sin x - a * x^2 - (1 + a) * x

theorem monotonicity_of_f_when_a_leq_zero (a : ℝ) (h : a ≤ 0) :
  (∃ c1 c2 : ℝ, c1 < 0 ∧ 0 < c2 ∧ 
    (∀ x : ℝ, x < 0 → deriv (f a) x < 0) ∧ 
    (∀ x : ℝ, x > 0 → deriv (f a) x > 0)) :=
sorry

theorem find_a_for_f_monotonic_increasing_on_R : 
  (∃ a : ℝ, a = 1/2 ∧ 
    (∀ x1 x2 : ℝ, x1 < x2 → f a x1 ≤ f a x2)) :=
sorry

end monotonicity_of_f_when_a_leq_zero_find_a_for_f_monotonic_increasing_on_R_l804_804212


namespace number_of_trapezoid_base_pairs_l804_804642

theorem number_of_trapezoid_base_pairs
    (area height : ℕ)
    (divisible_by : ℕ)
    (h_area : area = 2250)
    (h_height : height = 75)
    (h_divisible_by : divisible_by = 5) :
    (∃ b1 b2 : ℕ, b1 % divisible_by = 0 ∧ b2 % divisible_by = 0 ∧ h_height * (b1 + b2) / 2 = area) ↔ (7 : ℕ) :=
by sorry

end number_of_trapezoid_base_pairs_l804_804642


namespace player_two_minimize_diff_player_one_maximize_diff_l804_804298

def place_digits (call_digit : ℕ → ℕ) (place : ℕ → option ℕ → option ℕ) : ℕ × ℕ :=
let digit_list := λ n, call_digit n in
let a_digits := [place (digit_list 0) none, place (digit_list 1) none, place (digit_list 2) none, place (digit_list 3) none].map (λ x, option.getOrElse x 0) in
let b_digits := [place (digit_list 0) none, place (digit_list 1) none, place (digit_list 2) none, place (digit_list 3) none].map (λ x, option.getOrElse x 0) in
let a := list.foldl (λ acc d, 10 * acc + d) 0 a_digits in
let b := list.foldl (λ acc d, 10 * acc + d) 0 b_digits in
(a, b)

theorem player_two_minimize_diff (call_digit : ℕ → ℕ) (place : ℕ → option ℕ → option ℕ) :
  let (a, b) := place_digits call_digit place in
  a - b <= 4000 :=
by
  -- Proof will be provided here
  sorry

theorem player_one_maximize_diff (call_digit : ℕ → ℕ) (place : ℕ → option ℕ → option ℕ) :
  let (a, b) := place_digits call_digit place in
  a - b >= 4000 :=
by
  -- Proof will be provided here
  sorry

end player_two_minimize_diff_player_one_maximize_diff_l804_804298


namespace symmetric_line_equation_l804_804266

-- Define the conditions and question in Lean
def line_equation (x y : ℝ) : Prop := 2 * x - y + 3 = 0

def symmetric_point (M : ℝ × ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  let (mx, my) := M in
  let (px, py) := P in
  (2 * mx - px, 2 * my - py)

-- Define the theorem we want to prove
theorem symmetric_line_equation : 
  ∀ (x y : ℝ),
  let M := (-1, 2) 
  in let P := (x, y)
  in line_equation (symmetric_point M P).1 (symmetric_point M P).2 
  → (2 * x - y + 5 = 0) :=
by
  intros x y M P hp
  rw [symmetric_point] at hp
  sorry

end symmetric_line_equation_l804_804266


namespace round_robin_tournament_cycles_l804_804015

noncomputable def num_cycles (n : ℕ) : ℕ :=
  let m := n - 1 in
  (n * m / 2 * (n - 2) / 3)

theorem round_robin_tournament_cycles (n : ℕ) (w : ℕ) (l : ℕ) (games_played : ℕ) 
  (h1 : n = 21) (h2 : w = 10) (h3 : l = 10) (h4 : games_played = 20) :
  num_cycles n = 385 :=
by { rw [h1, h2, h3, h4], simp [num_cycles, div_eq_multiplication_of_inverses], sorry }

end round_robin_tournament_cycles_l804_804015


namespace compute_difference_l804_804225

noncomputable def fractional_part (z : ℝ) : ℝ := 
  z - floor z

theorem compute_difference (x y : ℝ) 
  (h1 : floor x - fractional_part y = 1.7)
  (h2 : fractional_part x + floor y = 4.4) :
  |x - y| = 1.9 :=
by
  sorry

end compute_difference_l804_804225


namespace derivative_problem1_derivative_problem2_derivative_problem3_derivative_problem4_l804_804419

theorem derivative_problem1 (x : ℝ) (hx : x < 1) : 
  deriv (λ x, (1 / (1 - sqrt x)) + (1 / (1 + sqrt x))) x = 2 / (1 - x)^2 := 
sorry

theorem derivative_problem2 (x : ℝ) : 
  deriv (λ x, (4 - sin x) / cos x) x = (4 * sin x - 1) / cos x^2 := 
sorry

theorem derivative_problem3 (x : ℝ) (hx : x > -1) : 
  deriv (λ x, ln (x + 3) - ln (x + 1)) x = -2 / ((x + 1) * (x + 3)) := 
sorry

theorem derivative_problem4 (x : ℝ) : 
  deriv (λ x, exp(x - cos x) * (1 + sin x) - (2 / x^2) - 3 * x^2) x = 
    exp(x - cos x) * (1 + sin x) - 2 * (1 / x^2) - 3 * x^2 := 
sorry

end derivative_problem1_derivative_problem2_derivative_problem3_derivative_problem4_l804_804419


namespace exponent_addition_l804_804491

theorem exponent_addition (a : ℝ) (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : (∏ i in finset.range m, a) * (∏ j in finset.range n, a) = a^(m + n) :=
sorry

end exponent_addition_l804_804491


namespace probability_two_english_teachers_l804_804334

open Nat

def num_english_teachers : ℕ := 3
def num_math_teachers : ℕ := 4
def num_social_studies_teachers : ℕ := 2
def total_teachers : ℕ := num_english_teachers + num_math_teachers + num_social_studies_teachers
def num_committees_of_size_2 : ℕ := choose total_teachers 2
def num_english_combinations_of_size_2 : ℕ := choose num_english_teachers 2

theorem probability_two_english_teachers :
  (num_english_combinations_of_size_2 : ℚ) / num_committees_of_size_2 = 1 / 12 :=
begin
  sorry
end

end probability_two_english_teachers_l804_804334


namespace two_digit_integers_congruent_1_mod_4_l804_804483

theorem two_digit_integers_congruent_1_mod_4 : 
  {n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ n % 4 = 1}.card = 22 :=
sorry

end two_digit_integers_congruent_1_mod_4_l804_804483


namespace log_sum_geom_seq_l804_804128
noncomputable theory

open Real

def geom_seq (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) = a n * a 1 / a 2

theorem log_sum_geom_seq (a : ℕ → ℝ) (h1 : geom_seq a) (h2 : 0 < ∀ n, a n) :
  a 1 * a 100 + a 3 * a 98 = 8 →
  (∑ i in Finset.range 100, log 2 (a (i + 1))) = 100 :=
by
  sorry

end log_sum_geom_seq_l804_804128


namespace minimum_distance_is_sqrt2_l804_804847

def parabola_intersection_min_distance : Prop :=
  let P := (-2, 2)
  let l (k : ℚ) : ℚ × ℚ := λ x : ℚ, (x, k * (x + 2) + 2)
  let parabola : ℚ × ℚ := λ x : ℚ, (x, real.sqrt (4 * x))
  let C : ℚ × ℚ := (4, 0)
  let A : ℚ × ℚ := _
  let B : ℚ × ℚ := _
  let Q : ℚ × ℚ := _
  let λ : ℚ := _
  ∀ P A B Q C λ, 
    (l k P.1 P.2) →              
    (y * y = 4 * x) →
    (P_A = λ P_B) →
    (Q_A = -(λ Q_B)) →
    dist Q C = sqrt 2

theorem minimum_distance_is_sqrt2 : parabola_intersection_min_distance := by
  sorry

end minimum_distance_is_sqrt2_l804_804847


namespace consecutive_hits_probability_l804_804742

theorem consecutive_hits_probability :
  (∃ (shots : ℕ) (hits : ℕ) (n_consecutive_hits : ℕ),
    shots = 5 ∧ hits = 3 ∧ n_consecutive_hits = 2 →
    (∑ x in {(HMHHM, P(HMHHM)), (HMHMH, P(HMHMH)), (MHHMH, P(MHHMH))}.val, x.2) = 3 / 5) :=
by
  sorry

end consecutive_hits_probability_l804_804742


namespace potato_yield_computation_l804_804566

noncomputable def garden_potato_yield (l_s w_s s_l y : ℕ) : ℕ :=
  let l_f := l_s * s_l -- Convert length to feet
  let w_f := w_s * s_l -- Convert width to feet
  let area := l_f * w_f -- Calculate area
  area * y -- Calculate yield

theorem potato_yield_computation : garden_potato_yield 15 20 2 0.5 = 600 :=
  by
  sorry

end potato_yield_computation_l804_804566


namespace percentage_equivalence_l804_804702

theorem percentage_equivalence (x : ℝ) : 0.3 * 0.6 * 0.7 * x = 0.126 * x :=
by
  sorry

end percentage_equivalence_l804_804702


namespace probability_of_roots_condition_satisfied_l804_804357

def quadratic_roots_condition (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 
    (k^2 + k - 90) * x1^2 + (3 * k - 8) * x1 + 2 = 0 ∧
    (k^2 + k - 90) * x2^2 + (3 * k - 8) * x2 + 2 = 0 ∧
    x1 ≤ 2 * x2

noncomputable def probability_k_condition_satisfied : ℝ :=
  let k_range := set.Icc 12 17 in
  let satisfying_k := { k : ℝ | k ∈ k_range ∧ quadratic_roots_condition k } in
  (satisfying_k.to_finset.card : ℝ) / (k_range.to_finset.card : ℝ)

theorem probability_of_roots_condition_satisfied :
  probability_k_condition_satisfied = 2 / 3 := 
sorry

end probability_of_roots_condition_satisfied_l804_804357


namespace num_divisors_720_l804_804197

def prime_factorization_720 := [(2, 4), (3, 2), (5, 1)]

def num_divisors (pf : List (ℕ × ℕ)) : ℕ :=
  pf.foldr (λ (pair : ℕ × ℕ) (acc : ℕ), (pair.2 + 1) * acc) 1

theorem num_divisors_720 : num_divisors prime_factorization_720 = 30 := 
by
  sorry

end num_divisors_720_l804_804197


namespace math_problem_l804_804033

theorem math_problem :
  (-1 : ℤ) ^ 49 + 2 ^ (4 ^ 3 + 3 ^ 2 - 7 ^ 2) = 16777215 := by
  sorry

end math_problem_l804_804033


namespace max_sides_in_13gon_l804_804907

theorem max_sides_in_13gon (n : ℕ) (h : n = 13) : 
  ∃ (s : ℕ), s = 13 ∧ 
  (∀ polygon_formed, polygon_formed ⊆ convex_polygon (13) ∧ is_subpolygon(polygon_formed) →
  sides(polygon_formed) ≤ s) :=
sorry

end max_sides_in_13gon_l804_804907


namespace inequality_proof_l804_804087

open Finset
open Real

theorem inequality_proof {n : ℕ} {a : ℕ → ℝ} (h₀ : 0 < n) (h₁ : ∀ i, 0 < a i) : 
  (∏ i in range n, 1 + 1 / (a i * (1 + a i))) ≥ (1 + 1 / (∏ i in range n, a i)^(1 / n) * (1 + (∏ i in range n, a i)^(1 / n))) ^ n :=
sorry

end inequality_proof_l804_804087


namespace correct_statements_l804_804023

-- Conditions
def synthetic_method : Prop := true -- Defined to symbolize that the synthetic method is correctly known.
def analytic_method : Prop := true -- Defined to symbolize that the analytic method is correctly known.

-- Definitions derived from conditions
def is_cause_to_effect (m : Prop) : Prop := m
def is_effect_to_cause (m : Prop) : Prop := m
def is_direct_proof (m : Prop) : Prop := m
def is_indirect_proof (m : Prop) : Prop := not (is_direct_proof m)

-- Statements
def statement_1 : Prop := is_cause_to_effect synthetic_method
def statement_2 : Prop := is_indirect_proof analytic_method
def statement_3 : Prop := is_effect_to_cause analytic_method
def statement_4 : Prop := is_direct_proof (not synthetic_method)

-- Proof
theorem correct_statements : statement_1 ∧ statement_3 ∧ not (statement_2 ∧ statement_4) :=
by {
  -- The proof will be provided here.
  sorry
}

end correct_statements_l804_804023


namespace geometric_sequence_common_ratio_l804_804890

theorem geometric_sequence_common_ratio (a q : ℝ) (h : a = a * q / (1 - q)) : q = 1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l804_804890


namespace arithmetic_geometric_sequence_l804_804679

theorem arithmetic_geometric_sequence (a d : ℤ) (h1 : ∃ a d, (a - d) * a * (a + d) = 1000)
  (h2 : ∃ a d, a^2 = 2 * (a - d) * ((a + d) + 7)) :
  d = 8 ∨ d = -15 :=
by sorry

end arithmetic_geometric_sequence_l804_804679


namespace oranges_in_second_group_l804_804632

namespace oranges_problem

-- Definitions coming from conditions
def cost_of_apple : ℝ := 0.21
def total_cost_1 : ℝ := 1.77
def total_cost_2 : ℝ := 1.27
def num_apples_group1 : ℕ := 6
def num_oranges_group1 : ℕ := 3
def num_apples_group2 : ℕ := 2
def cost_of_orange : ℝ := 0.17
def num_oranges_group2 : ℕ := 5 -- derived from the solution involving $0.85/$0.17.

-- Price calculation functions and conditions
def price_group1 (cost_of_orange : ℝ) : ℝ :=
  num_apples_group1 * cost_of_apple + num_oranges_group1 * cost_of_orange

def price_group2 (num_oranges_group2 cost_of_orange : ℝ) : ℝ :=
  num_apples_group2 * cost_of_apple + num_oranges_group2 * cost_of_orange

theorem oranges_in_second_group :
  (price_group1 cost_of_orange = total_cost_1) →
  (price_group2 num_oranges_group2 cost_of_orange = total_cost_2) →
  num_oranges_group2 = 5 :=
by
  intros h1 h2
  sorry

end oranges_problem

end oranges_in_second_group_l804_804632


namespace find_a_l804_804133

theorem find_a (f : ℝ → ℝ) (h1 : ∀ x, f (2^x) = x + 3) (h2 : f a = 5) : a = 4 := 
by
  sorry

end find_a_l804_804133


namespace peter_speed_l804_804530

variable (P : ℝ)
variable (h1 : Juan_speed = P + 3)
variable (h2 : time = 1.5)
variable (h3 : distance = 19.5)
variable (Peter_travel : Peter_travel = time * P)
variable (Juan_travel : Juan_travel = time * (P + 3))

theorem peter_speed : P = 5 :=
by
  have total_distance : distance = Peter_travel + Juan_travel := by sorry
  sorry

end peter_speed_l804_804530


namespace Question_D_condition_l804_804233

theorem Question_D_condition (P Q : Prop) (h : P → Q) : ¬ Q → ¬ P :=
by sorry

end Question_D_condition_l804_804233


namespace tan_alpha_value_l804_804835

theorem tan_alpha_value (α : ℝ) (h1 : sin α + cos α = 1/5) (h2 : 0 ≤ α ∧ α < π) : 
  tan α = -4/3 :=
by
  sorry

end tan_alpha_value_l804_804835


namespace arithmetic_contains_geometric_progression_l804_804617

theorem arithmetic_contains_geometric_progression (a d : ℕ) (h_pos : d > 0) :
  ∃ (a' : ℕ) (r : ℕ), a' = a ∧ r = 1 + d ∧ (∀ k : ℕ, ∃ n : ℕ, a' * r^k = a + (n-1)*d) :=
by
  sorry

end arithmetic_contains_geometric_progression_l804_804617


namespace find_f_neg_one_l804_804124

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem find_f_neg_one (f : ℝ → ℝ) (h_odd : is_odd f)
(h_pos : ∀ x, 0 < x → f x = x^2 + 1/x) : f (-1) = -2 := 
sorry

end find_f_neg_one_l804_804124


namespace smallest_a_condition_l804_804084

theorem smallest_a_condition:
  ∃ a: ℝ, (∀ x y z: ℝ, (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1) → a * (x^2 + y^2 + z^2) + x * y * z ≥ 10 / 27) ∧ a = 2 / 9 :=
sorry

end smallest_a_condition_l804_804084


namespace diameter_of_inscribed_circle_l804_804694

-- Define the sides of the triangle
def DE : ℝ := 13
def DF : ℝ := 8
def EF : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (DE + DF + EF) / 2

-- Define Heron's formula for the area of the triangle
def area_triangle (a b c : ℝ) : ℝ :=
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the radius of the inscribed circle in terms of the area and semiperimeter
def radius_incircle (a b c : ℝ) : ℝ :=
  area_triangle a b c / s

-- Define the diameter of the inscribed circle
def diameter_incircle (a b c : ℝ) : ℝ :=
  2 * radius_incircle a b c

-- Assert the diameter of the inscribed circle in triangle DEF is 5.16
theorem diameter_of_inscribed_circle : diameter_incircle DE DF EF = 5.16 :=
by
  sorry

end diameter_of_inscribed_circle_l804_804694


namespace cost_of_eraser_is_1_l804_804371

variables (P : ℝ)  -- Price of a pencil
variables (num_pencils : ℝ) (num_erasers : ℝ) (total_revenue : ℝ)

-- Conditions
def condition1 := num_erasers = 2 * num_pencils
def condition2 := num_pencils = 20
def condition3 := total_revenue = 80
def condition4 := total_revenue = num_pencils * P + num_erasers * (1/2 * P)

-- Theorem: The cost of an eraser is 1 dollar
theorem cost_of_eraser_is_1 : 
    (P : ℝ) → (condition1) → (condition2) → (condition3) → (condition4) → (1/2 * P) = 1 :=
by
  sorry

end cost_of_eraser_is_1_l804_804371


namespace ratio_ab_l804_804479

variable (x y a b : ℝ)
variable (h1 : 4 * x - 2 * y = a)
variable (h2 : 6 * y - 12 * x = b)
variable (h3 : b ≠ 0)

theorem ratio_ab : 4 * x - 2 * y = a ∧ 6 * y - 12 * x = b ∧ b ≠ 0 → a / b = -1 / 3 := by
  sorry

end ratio_ab_l804_804479


namespace ratio_of_areas_l804_804649

theorem ratio_of_areas (a b : ℝ) (h : a = b * Real.sqrt 2) : 
  let total_area := 4 * b^2 + 4 * b^2 * Real.sqrt 2 in
  let shaded_area := b^2 + b^2 * Real.sqrt 2 in
  shaded_area / total_area = 1 / 4 :=
by 
  sorry

end ratio_of_areas_l804_804649


namespace hedgehogs_found_baskets_l804_804686

noncomputable theory

-- Definitions from conditions
def number_of_hedgehogs : ℕ := 2
def strawberries_per_basket : ℕ := 900
def strawberries_eaten_per_hedgehog : ℕ := 1050
def portion_strawberries_eaten : ℚ := 7 / 9

-- The main theorem to prove
theorem hedgehogs_found_baskets :
  ∃ (baskets : ℕ), ((number_of_hedgehogs * strawberries_eaten_per_hedgehog * 9) / 7) = baskets * strawberries_per_basket := 
sorry

end hedgehogs_found_baskets_l804_804686


namespace number_triangle_value_of_n_l804_804923

theorem number_triangle_value_of_n:
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y = 2022 ∧ (∃ n : ℕ, n > 0 ∧ n^2 ∣ 2022 ∧ n = 1) :=
by sorry

end number_triangle_value_of_n_l804_804923


namespace find_m_l804_804000

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def point_on_x_axis_distance (x y : ℝ) : Prop :=
  y = 14

def point_distance_from_fixed_point (x y : ℝ) : Prop :=
  distance (x, y) (3, 8) = 8

def x_coordinate_condition (x : ℝ) : Prop :=
  x > 3

def m_distance (x y m : ℝ) : Prop :=
  distance (x, y) (0, 0) = m

theorem find_m (x y m : ℝ) 
  (h1 : point_on_x_axis_distance x y) 
  (h2 : point_distance_from_fixed_point x y) 
  (h3 : x_coordinate_condition x) :
  m_distance x y m → 
  m = Real.sqrt (233 + 12 * Real.sqrt 7) := by
  sorry

end find_m_l804_804000


namespace arithmetic_sequence_10th_term_l804_804639

theorem arithmetic_sequence_10th_term (a d : ℤ) :
    (a + 4 * d = 26) →
    (a + 7 * d = 50) →
    (a + 9 * d = 66) := by
  intros h1 h2
  sorry

end arithmetic_sequence_10th_term_l804_804639


namespace bridge_length_proof_l804_804750

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 49.9960003199744
noncomputable def train_speed_kmph : ℝ := 18
noncomputable def conversion_factor : ℝ := 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * conversion_factor
noncomputable def total_distance : ℝ := train_speed_mps * time_to_cross_bridge
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_proof : bridge_length = 149.980001599872 := 
by 
  sorry

end bridge_length_proof_l804_804750


namespace arithmetic_sequence_a15_l804_804918

theorem arithmetic_sequence_a15 {a : ℕ → ℝ} (d : ℝ) (a7 a23 : ℝ) 
    (h1 : a 7 = 8) (h2 : a 23 = 22) : 
    a 15 = 15 := 
by
  sorry

end arithmetic_sequence_a15_l804_804918


namespace sum_of_n_values_l804_804699

theorem sum_of_n_values : 
  ∑ n in {n : ℤ | ∃ (d : ℤ), d * (2 * n - 1) = 24}.toFinset, n = 2 :=
by
  sorry

end sum_of_n_values_l804_804699


namespace expected_potato_yield_l804_804567

-- Define the problem and required conditions as constants
constant steps_to_feet : ℕ → ℕ
def steps_dim1 := 15
def steps_dim2 := 20
def step_length := 2
def yield_per_sq_ft := 1 / 2

-- Define the required values based on conditions
def length := steps_to_feet steps_dim1
def width := steps_to_feet steps_dim2

-- Convert steps to feet
def steps_to_feet (n : ℕ) : ℕ := n * step_length

-- Calculate the expected yield of potatoes
def area (length width : ℕ) : ℕ := length * width
def yield (area : ℕ) (yield_per_sq_ft : ℚ) : ℚ := area * yield_per_sq_ft

-- Prove the expected yield of potatoes is 600 pounds
theorem expected_potato_yield : yield (area length width) yield_per_sq_ft = 600 := by
  sorry

end expected_potato_yield_l804_804567


namespace simplify_and_evaluate_l804_804252

theorem simplify_and_evaluate (m : ℝ) (h_root : m^2 + 3 * m - 2 = 0) :
  (m - 3) / (3 * m^2 - 6 * m) / (m + 2 - 5 / (m - 2)) = 1 / 6 :=
by
  sorry

end simplify_and_evaluate_l804_804252


namespace find_a_for_odd_function_l804_804856

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  a - 1 / (2^x + 1)

theorem find_a_for_odd_function :
  (∃ a : ℝ, is_odd_function (given_function a)) ↔ ∃ a : ℝ, a = 1 / 2 := sorry

end find_a_for_odd_function_l804_804856


namespace usual_time_to_school_l804_804304

-- Defining the given variables and conditions
variables (T R : ℝ)
axiom h1 : (T > 0)
axiom h2 : (R > 0)
axiom h3 : (3 / 7 * T + 2 / 5 * T) = T - 3

-- The proof statement
theorem usual_time_to_school : T = 17.5 :=
by
  have h4 : (29 / 35 * T) = T - 3 := by sorry
  have h5 : 29 * T = 35 * T - 105 := by sorry
  have h6 : 6 * T = 105 := by sorry
  have h7 : T = 105 / 6 := by sorry
  exact h7.symm.subst (by linarith)

end usual_time_to_school_l804_804304


namespace caleb_buys_41_double_burgers_l804_804709

variable (S D : ℕ)

theorem caleb_buys_41_double_burgers (h1 : 1.00 * S + 1.50 * D = 70.50) 
                                     (h2 : S + D = 50) : 
  D = 41 := 
sorry

end caleb_buys_41_double_burgers_l804_804709


namespace range_of_f_l804_804857

def f (x : ℤ) : ℤ := x ^ 2 + 2 * x

theorem range_of_f :
  (set.range (λ (x : ℤ), f x) (set.Icc (-2 : ℤ) 1) = {-1, 0, 3}) :=
by {
  sorry 
}

end range_of_f_l804_804857


namespace complex_quadrant_l804_804465

theorem complex_quadrant (a b : ℝ) (h : (a + Complex.I) / (b - Complex.I) = 2 - Complex.I) :
  (a < 0 ∧ b < 0) :=
by
  sorry

end complex_quadrant_l804_804465


namespace ab_equation_l804_804180

theorem ab_equation : 
  ∀ (BC CD AD : ℝ) (angleA angleB : ℝ) (r s : ℝ), 
  BC = 10 → CD = 15 → AD = 13 → angleA = 45 → angleB = 45 → 
  (∃ (r s : ℕ), \(AB = r + \sqrt{s}\) /\ \(AB = 27.5\) → r + s = 27.25) := 
by
  intros BC CD AD angleA angleB r s hBC hCD hAD hAngleA hAngleB hExistsAB
  obtain ⟨r, s, h⟩ := hExistsAB
  sorry

end ab_equation_l804_804180


namespace find_a_l804_804327

-- Define the given context (condition)
def condition (a : ℝ) : Prop := 0.5 / 100 * a = 75 / 100 -- since 1 paise = 1/100 rupee

-- Define the statement to prove
theorem find_a (a : ℝ) (h : condition a) : a = 150 := 
sorry

end find_a_l804_804327


namespace decisive_factor_population_size_l804_804752

-- Conditions (Defining the characteristics of the population)
variables (PopulationDensity BirthRate DeathRate AgeStructure SexRatio ImmigrationRate EmigrationRate : Type)

-- The statement we need to prove
theorem decisive_factor_population_size
  (h1: BirthRate ≠ 0)
  (h2: DeathRate ≠ 0) :
  (BirthRate + DeathRate) = (BirthRate + DeathRate) := sorry

end decisive_factor_population_size_l804_804752


namespace find_phi_l804_804488

theorem find_phi (φ : ℝ) (h1 : 0 < φ ∧ φ < 90) (h2 : √3 * sin (15 * real.pi / 180) = cos φ - sin φ) :
  φ = 15 * real.pi / 180 :=
sorry

end find_phi_l804_804488


namespace range_m_l804_804821

def p (m : ℝ) : Prop := abs (1 - m) / sqrt 2 < 1
def q (m : ℝ) : Prop := 4 - 4 * m ≥ 0
def nq (m : ℝ) : Prop := ¬ (q m)

theorem range_m (m : ℝ) (hq_false : ¬ nq m) (hpq_true : p m ∨ q m) : m ∈ Iic 1 :=
by
  sorry

end range_m_l804_804821


namespace exists_f_in_S_for_positive_rational_q_l804_804221

def is_in_S (f : ℝ → ℝ) : Prop :=
  ∃ n (g : fin n → ℝ → ℝ), (∀ i, g i = sin ∨ g i = cos ∨ g i = tan ∨ g i = asin ∨ g i = acos ∨ g i = atan) ∧
                          f = (g 0 ∘ g 1 ∘ ... ∘ g (n-1))

theorem exists_f_in_S_for_positive_rational_q (q : ℚ) (q_pos : 0 < q) : ∃ f : ℝ → ℝ, is_in_S f ∧ f 1 = q :=
by
  sorry

end exists_f_in_S_for_positive_rational_q_l804_804221


namespace cone_intersects_sphere_in_circle_l804_804620

-- Definitions of the geometric terms used in the problem
variable (δ : Type) -- Plane δ
variable (G : Type) -- Sphere G
variable (E : Type) -- Point E
variable circle : δ → Prop -- A property that holds if an object is a circle on the plane δ
variable intersects_sphere : δ → E → G → Prop -- A property that holds if a cone formed by a circle on δ and point E intersects the sphere G 

-- The statement of the proof problem in Lean 4
theorem cone_intersects_sphere_in_circle (h1 : ∀ c : δ, circle c) (h2 : E ∉ G) :
  ∀ c : δ, intersects_sphere c E G :=
sorry

end cone_intersects_sphere_in_circle_l804_804620


namespace n_times_s_eq_zero_l804_804213

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g(g(x)^2 - y^2) = g(x)^2 + g(g(y)^2 - g(-x)^2) + x^2

theorem n_times_s_eq_zero : 
  let n := {a : ℝ | ∃ x, g(x) = a}.finite.to_finset.card,
      s := {a : ℝ | ∃ x, g(x) = a}.finite.to_finset.sum id
  in n * s = 0 :=
by
  sorry

end n_times_s_eq_zero_l804_804213


namespace sum_of_real_solutions_eq_32_over_7_l804_804432

theorem sum_of_real_solutions_eq_32_over_7 :
  (∑ x in (finset.filter (λ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x)) finset.univ), x) = 32 / 7 :=
by
  sorry

end sum_of_real_solutions_eq_32_over_7_l804_804432


namespace nina_total_spending_l804_804234

-- Defining the quantities and prices of each category of items
def num_toys : Nat := 3
def price_per_toy : Nat := 10

def num_basketball_cards : Nat := 2
def price_per_card : Nat := 5

def num_shirts : Nat := 5
def price_per_shirt : Nat := 6

-- Calculating the total cost for each category
def cost_toys : Nat := num_toys * price_per_toy
def cost_cards : Nat := num_basketball_cards * price_per_card
def cost_shirts : Nat := num_shirts * price_per_shirt

-- Calculating the total amount spent
def total_cost : Nat := cost_toys + cost_cards + cost_shirts

-- The final theorem statement to verify the answer
theorem nina_total_spending : total_cost = 70 :=
by
  sorry

end nina_total_spending_l804_804234


namespace smallest_N_l804_804089

-- Conditions translated to Lean definitions
def f (n : ℕ) : ℕ := Nat.digits 5 n |>.sum
def g (n : ℕ) : ℕ := Nat.digits 9 (f n) |>.sum

-- The proof problem statement
theorem smallest_N'_mod_1000 : 
  ∃ N' : ℕ, (Nat.digits 18 (g N')).any (λ d, d = 10) ∧ N' % 1000 = 619 := by
  sorry

end smallest_N_l804_804089


namespace amy_total_equals_bob_total_l804_804502

def original_price : ℝ := 120.00
def sales_tax_rate : ℝ := 0.08
def discount_rate : ℝ := 0.25
def additional_discount : ℝ := 0.10
def num_sweaters : ℕ := 4

def calculate_amy_total (original_price : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (additional_discount : ℝ) (num_sweaters : ℕ) : ℝ :=
  let price_with_tax := original_price * (1.0 + sales_tax_rate)
  let discounted_price := price_with_tax * (1.0 - discount_rate)
  let final_price := discounted_price * (1.0 - additional_discount)
  final_price * (num_sweaters : ℝ)
  
def calculate_bob_total (original_price : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (additional_discount : ℝ) (num_sweaters : ℕ) : ℝ :=
  let discounted_price := original_price * (1.0 - discount_rate)
  let further_discounted_price := discounted_price * (1.0 - additional_discount)
  let price_with_tax := further_discounted_price * (1.0 + sales_tax_rate)
  price_with_tax * (num_sweaters : ℝ)

theorem amy_total_equals_bob_total :
  calculate_amy_total original_price sales_tax_rate discount_rate additional_discount num_sweaters =
  calculate_bob_total original_price sales_tax_rate discount_rate additional_discount num_sweaters :=
by
  sorry

end amy_total_equals_bob_total_l804_804502


namespace tangent_addition_tangent_subtraction_l804_804690

theorem tangent_addition (a b : ℝ) : 
  Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
sorry

theorem tangent_subtraction (a b : ℝ) : 
  Real.tan (a - b) = (Real.tan a - Real.tan b) / (1 + Real.tan a * Real.tan b) :=
sorry

end tangent_addition_tangent_subtraction_l804_804690


namespace line_parallel_to_x_axis_l804_804816

variable (k : ℝ)

theorem line_parallel_to_x_axis :
  let point1 := (3, 2 * k + 1)
  let point2 := (8, 4 * k - 5)
  (point1.2 = point2.2) ↔ (k = 3) :=
by
  sorry

end line_parallel_to_x_axis_l804_804816


namespace count_indices_eq_binom_l804_804785

-- Definition of the sequence a_n
def a : ℕ → ℤ
| 1 := 0
| n := if n = 0 then 0 else a (n/2) + (-1)^(n*(n+1)/2)

-- Statement of the problem
theorem count_indices_eq_binom :
  ∀ k : ℕ, 
    (finset.card ((finset.Ico (2^k) (2^(k+1))).filter (λ n, a n = 0))) = nat.choose k (k / 2) :=
sorry

end count_indices_eq_binom_l804_804785


namespace calculate_jessie_points_l804_804997

theorem calculate_jessie_points (total_points : ℕ) (some_players_points : ℕ) (players : ℕ) :
  total_points = 311 →
  some_players_points = 188 →
  players = 3 →
  (total_points - some_players_points) / players = 41 :=
by
  intros
  sorry

end calculate_jessie_points_l804_804997


namespace saleswoman_commission_l804_804439

theorem saleswoman_commission (S : ℝ)
  (h1 : (S > 500) )
  (h2 : (0.20 * 500 + 0.50 * (S - 500)) = 0.3125 * S) : 
  S = 800 :=
sorry

end saleswoman_commission_l804_804439


namespace avg_score_false_iff_unequal_ints_l804_804069

variable {a b m n : ℕ}

theorem avg_score_false_iff_unequal_ints 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (m_pos : 0 < m) 
  (n_pos : 0 < n) 
  (m_neq_n : m ≠ n) : 
  (∃ a b, (ma + nb) / (m + n) = (a + b)/2) ↔ a ≠ b := 
sorry

end avg_score_false_iff_unequal_ints_l804_804069


namespace neg_fraction_comparison_l804_804041

theorem neg_fraction_comparison : - (4 / 5 : ℝ) > - (5 / 6 : ℝ) :=
by {
  -- sorry to skip the proof
  sorry
}

end neg_fraction_comparison_l804_804041


namespace janelle_total_marbles_l804_804194

def initial_green_marbles := 26
def bags_of_blue_marbles := 12
def marbles_per_bag := 15
def gift_red_marbles := 7
def gift_green_marbles := 9
def gift_blue_marbles := 12
def gift_red_marbles_given := 3
def returned_blue_marbles := 8

theorem janelle_total_marbles :
  let total_green := initial_green_marbles - gift_green_marbles
  let total_blue := (bags_of_blue_marbles * marbles_per_bag) - gift_blue_marbles + returned_blue_marbles
  let total_red := gift_red_marbles - gift_red_marbles_given
  total_green + total_blue + total_red = 197 :=
by
  sorry

end janelle_total_marbles_l804_804194


namespace find_missing_number_l804_804029

/-- 
  Define the grid and the condition for the numbers.
-/
def grid : Type := ℕ → ℕ → Option ℕ

def valid_grid (g : grid) : Prop :=
  ∀ x y,
    (x, y) ≠ (1, 1) ∧ (x, y) ≠ (1, 2) ∧ (x, y) ≠ (x, y) ≠ (2, 1) ∧
    (x, y) ≠ (2, 2) ∧ (y1, y2) ≠ (x1, y1) ∧ (x, y) ≠ (3, 5) ∧
    (y1, y3) ≠ (x2, y2) ∧ (x, y) ≠ (4, 2) ∧ (y1, y4) ≠ (x2, y4) ∧
    (x, y) ≠ (4, 4) → 
    (∃ a b, 
      g (x - 1) (y - 1) = some a ∧ 
      g (x + 1) (y + 1) = some b ∧ 
      g x y = some (a + b))

def initial_grid : grid
| 1, 1 := some 1
| 1, 2 := none
| 1, 3 := some 3
| 1, 4 := none
| 1, 5 := some 5
| 2, 1 := none
| 2, 2 := some 16
| 2, 3 := none
| 2, 4 := some 18
| 2, 5 := none
| 3, 1 := some 17
| 3, 2 := none
| 3, 3 := none
| 3, 4 := none
| 3, 5 := some 21
| 4, 1 := none
| 4, 2 := some 23
| 4, 3 := none
| 4, 4 := some 25
| 4, 5 := none
| 5, 1 := some 6
| 5, 2 := none
| 5, 3 := some 8
| 5, 4 := none
| 5, 5 := some 10
| _, _ := none

theorem find_missing_number : ∀ (g : grid), valid_grid g → g 2 3 = some 14 :=
by
  intro g valid
  sorry -- proof to be constructed

end find_missing_number_l804_804029


namespace geom_series_eq_l804_804205

noncomputable def C (n : ℕ) := 256 * (1 - 1 / (4^n)) / (3 / 4)
noncomputable def D (n : ℕ) := 1024 * (1 - 1 / ((-2)^n)) / (3 / 2)

theorem geom_series_eq (n : ℕ) (h : n ≥ 1) : C n = D n ↔ n = 1 :=
by
  sorry

end geom_series_eq_l804_804205


namespace find_a_b_find_extreme_value_l804_804135

def f (a b c : ℝ) (x : ℝ) : ℝ := a*x^3 + b*x + c

theorem find_a_b (c : ℝ) (h₁ : f a b c 1 = c - 4) (h₂ : f a b c 1 = 0) :
  a = 2 ∧ b = -6 :=
  sorry

theorem find_extreme_value (a b c : ℝ) (h₁ : a = 2) (h₂ : b = -6) (h₃ : c = 0)
  (h₄ : ∀ x : ℝ, f a b c (-x) = -f a b c x) :
  ∃ x ∈ set.Icc (-2 : ℝ) 0, is_maximum f x 4 :=
  sorry

end find_a_b_find_extreme_value_l804_804135


namespace trapezoid_median_l804_804054

theorem trapezoid_median 
  (h : ℝ)
  (triangle_base : ℝ := 24)
  (trapezoid_base1 : ℝ := 15)
  (trapezoid_base2 : ℝ := 33)
  (triangle_area_eq_trapezoid_area : (1 / 2) * triangle_base * h = ((trapezoid_base1 + trapezoid_base2) / 2) * h)
  : (trapezoid_base1 + trapezoid_base2) / 2 = 24 :=
by
  sorry

end trapezoid_median_l804_804054


namespace jam_consumption_l804_804188

theorem jam_consumption (x y t : ℝ) :
  x + y = 100 →
  t = 45 * x / y →
  t = 20 * y / x →
  x = 40 ∧ y = 60 ∧ 
  (y / 45 = 4 / 3) ∧ 
  (x / 20 = 2) := by
  sorry

end jam_consumption_l804_804188


namespace problem_statement_l804_804582

-- Definitions based on conditions
def two_digit_number (N : ℕ) := N >= 10 ∧ N < 100
def divisible_by (a b : ℕ) := a % b = 0
def mistaken_exactly (N : ℕ) := (if divisible_by N 3 then 0 else 1) +
                               (if divisible_by N 4 then 0 else 1) +
                               (if divisible_by N 5 then 0 else 1) +
                               (if divisible_by N 9 then 0 else 1) +
                               (if divisible_by N 10 then 0 else 1) +
                               (if divisible_by N 15 then 0 else 1) +
                               (if divisible_by N 18 then 0 else 1) +
                               (if divisible_by N 30 then 0 else 1)

-- Lean 4 statement
theorem problem_statement (N : ℕ) (h_two_digit : two_digit_number N) (h_mistaken : mistaken_exactly N = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 := sorry

end problem_statement_l804_804582


namespace toys_ratio_l804_804759

theorem toys_ratio (k A M T : ℕ) (h1 : M = 6) (h2 : A = k * M) (h3 : A = T - 2) (h4 : A + M + T = 56):
  A / M = 4 :=
by
  sorry

end toys_ratio_l804_804759


namespace tanya_work_days_l804_804627

def work_rate_sakshi : ℝ := 1 / 10

def efficiency_tanya : ℝ := 1.25 * work_rate_sakshi

theorem tanya_work_days : ∃ (days : ℝ), days = 8 ∧ efficiency_tanya = 1 / days := 
by
  existsi (8 : ℝ)
  sorry

end tanya_work_days_l804_804627


namespace verify_statements_l804_804060

-- Define the condition for a doubling point
def is_doubling_point (P Q : ℝ × ℝ) : Prop :=
  2 * (P.1 + Q.1) = P.2 + Q.2

-- Define the given point P1
def P1 : ℝ × ℝ := (1, 0)

-- Define the four statements in Lean
def statement1 : Prop :=
  is_doubling_point P1 (3, 8) ∧ is_doubling_point P1 (-2, -2)

def statement2 : Prop :=
  ∀ A : ℝ × ℝ, A.2 = A.1 + 2 → is_doubling_point P1 A → A = (2, 4)

def statement3 : Prop :=
  ∃ x : ℝ, (is_doubling_point P1 (x, x^2 - 2*x - 3))

def statement4 : Prop :=
  ∀ B : ℝ × ℝ, is_doubling_point P1 B → ∃ x, B = (x, 2*(x + 1)) ∧ 
    dist P1 B = 4 * real.sqrt 5 / 5

-- Define the main theorem to verify correct statements
theorem verify_statements :
  statement1 ∧ ¬ statement2 ∧ statement3 ∧ statement4 → 
  ({statement1, ¬statement2, statement3, statement4}.count true = 3) :=
sorry

end verify_statements_l804_804060


namespace possible_numbers_l804_804590

theorem possible_numbers (N : ℕ) 
    (h1 : 10 ≤ N) (h2 : N ≤ 99)
    (h3 : (N % 3 = 0) ∧ (N % 4 = 0) ∧ (N % 5 = 0) ∧ (N % 9 = 0) ∧ (N % 10 = 0) ∧ 
          (N % 15 = 0) ∧ (N % 18 = 0) ∧ (N % 30 = 0) ∨ 
          (N % 3 ≠ 0) + (N % 4 ≠ 0) + (N % 5 ≠ 0) + (N % 9 ≠ 0) + 
          (N % 10 ≠ 0) + (N % 15 ≠ 0) + (N % 18 ≠ 0) + (N % 30 ≠ 0) = 4) :
   N = 36 ∨ N = 45 ∨ N = 72 :=
by {
  sorry
}

end possible_numbers_l804_804590


namespace sqrt_pow_product_l804_804395

variable {R : Type*} [OrderedRing R] [HasSqrt R]

theorem sqrt_pow_product (a b : R) (ha : a = 3) (hb : b = 5) : sqrt (a^2 * b^4) = 75 := 
by sorry

end sqrt_pow_product_l804_804395


namespace angle_PMN_is_55_l804_804186

-- Definitions from the conditions
variable (P Q R M N : Type)
variable [triangle : Triangle P Q R]
variable [triangle' : Triangle P M N]
variable [isosceles : IsoscelesTriangle P Q R]
variable [isosceles' : IsoscelesTriangle P M N]

-- Given conditions
def AnglePQR : Angle P Q R := 70
def IsoscelesPQR : IsoscelesTriangle P Q R := { hPQ := proof, hPR := proof, hQR := proof}
def IsoscelesPMN : IsoscelesTriangle P M N := { hPM := proof, hPN := proof, hMN := proof }

-- The theorem to be proved
theorem angle_PMN_is_55 :
  ∃ (a : Angle P M N), a = 55 := by
  sorry

end angle_PMN_is_55_l804_804186


namespace average_pushups_is_correct_l804_804401

theorem average_pushups_is_correct :
  ∀ (David Zachary Emily : ℕ),
    David = 510 →
    Zachary = David - 210 →
    Emily = David - 132 →
    (David + Zachary + Emily) / 3 = 396 :=
by
  intro David Zachary Emily hDavid hZachary hEmily
  -- All calculations and proofs will go here, but we'll leave them as sorry for now.
  sorry

end average_pushups_is_correct_l804_804401


namespace f_x_plus_3_eq_l804_804882

-- Define the function f
def f (x : ℝ) := (x * (x - 1)) / 2

-- Prove that f(x + 3) equals (x + 3) * f(x + 1) / x
theorem f_x_plus_3_eq (x : ℝ) : f(x + 3) = (x + 3) * f(x + 1) / x :=
by
  -- Proof will be placed here
  sorry

end f_x_plus_3_eq_l804_804882


namespace unit_digit_product_l804_804312

theorem unit_digit_product :
  let units := [7858413, 10864231, 45823797, 97833129, 51679957, 38213827, 75946153, 27489543, 94837311, 37621597].map (λ x, x % 10)
  in units.product % 10 = 1 :=
by
  sorry

end unit_digit_product_l804_804312


namespace percentage_increase_ticket_price_l804_804531

-- Definitions for the conditions
def last_year_income := 100.0
def clubs_share_last_year := 0.10 * last_year_income
def rental_cost := 0.90 * last_year_income
def new_clubs_share := 0.20
def new_income := rental_cost / (1 - new_clubs_share)

-- Lean 4 theorem statement
theorem percentage_increase_ticket_price : 
  new_income = 112.5 → ((new_income - last_year_income) / last_year_income * 100) = 12.5 := 
by
  sorry

end percentage_increase_ticket_price_l804_804531


namespace minimum_value_PA_PB_l804_804464

theorem minimum_value_PA_PB
  (O : Point) (P : Point)
  (AB : Line) (C : Circle)
  (hO : O = ⟨0, 0⟩)
  (h_circle_O : ∀ (x y : ℝ), x^2 + y^2 = 1)
  (h_chord_AB : length AB = sqrt 3)
  (h_circle_C : ∃ (x y : ℝ), (x - 2)^2 + (y - 3)^2 = 2)
  (h_on_circle_C : ∀ (P : Point), liesOnCircle P C) :
  minimum (sqrt (dot_product (PA - P) (PB - P) + 3/4)) = sqrt 13 - 1/2 - sqrt 2 :=
by
  sorry

end minimum_value_PA_PB_l804_804464


namespace gcd_of_three_numbers_l804_804270

theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd 324 243) 135 = 27 := 
by 
  sorry

end gcd_of_three_numbers_l804_804270


namespace round_robin_tournament_cycles_l804_804014

noncomputable def num_cycles (n : ℕ) : ℕ :=
  let m := n - 1 in
  (n * m / 2 * (n - 2) / 3)

theorem round_robin_tournament_cycles (n : ℕ) (w : ℕ) (l : ℕ) (games_played : ℕ) 
  (h1 : n = 21) (h2 : w = 10) (h3 : l = 10) (h4 : games_played = 20) :
  num_cycles n = 385 :=
by { rw [h1, h2, h3, h4], simp [num_cycles, div_eq_multiplication_of_inverses], sorry }

end round_robin_tournament_cycles_l804_804014


namespace log_sum_is_two_l804_804472

noncomputable def log_base (a x : ℝ) : ℝ := log x / log a

axiom a_pos_and_ne_one (a : ℝ) : a > 0 ∧ a ≠ 1

def func_y (a x : ℝ) : ℝ := log_base a (x - 1) + 4

def fixed_point_P (P : ℝ × ℝ) : Prop := P = (2, 4)

def power_function_f (x : ℝ) : ℝ := x ^ 2

theorem log_sum_is_two (a : ℝ) (P : ℝ × ℝ) :
  a > 0 ∧ a ≠ 1 →
  fixed_point_P P →
  P ∈ set_of (λ (p : ℝ × ℝ), p.2 = func_y a p.1) →
  (Real.log 10 / Real.log 2) * (Real.log 4 / Real.log 10 + Real.log 25 / Real.log 10) = 2 :=
begin
  sorry
end

end log_sum_is_two_l804_804472


namespace intersection_of_sets_l804_804961

open Set

variable {U : Set ℕ} {A B : Set ℕ}
variables [finite U] [finite A] [finite B]

noncomputable def complement (s : Set ℕ) (U : Set ℕ) : Set ℕ :=
  U \ s

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} 
noncomputable def A : Set ℕ := {2, 4, 6, 8, 10} 
noncomputable def C_UA : Set ℕ := complement A U 
noncomputable def C_UB : Set ℕ := {1, 4, 6, 8, 9} 
noncomputable def B : Set ℕ := complement C_UB U

theorem intersection_of_sets :
  A ∩ B = {2} :=
 by sorry

end intersection_of_sets_l804_804961


namespace average_hours_per_person_l804_804016

theorem average_hours_per_person (num_people : ℕ) (days_in_cycle : ℕ) (people_on_duty : ℕ)
    (work_days_per_person : ℕ) (hours_per_day : ℕ) :
  num_people = 8 →
  days_in_cycle = 8 →
  people_on_duty = 3 →
  work_days_per_person = 3 →
  hours_per_day = 24 →
  (hours_per_day * work_days_per_person / days_in_cycle) = 9 := 
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num

end average_hours_per_person_l804_804016


namespace average_speed_interval_l804_804278

def motion_equation (t : ℝ) : ℝ := 3 + t^2

def average_speed (s : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
  (s t2 - s t1) / (t2 - t1)

theorem average_speed_interval :
  average_speed motion_equation 2 2.1 = 4.1 := by
  sorry

end average_speed_interval_l804_804278


namespace smallest_c_value_l804_804534

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  c * x * (x - 1)

noncomputable def f_iterate (c : ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  nat.iterate (f c) n x

theorem smallest_c_value :
  ∃ c : ℝ, c > 0 ∧ (∀ n : ℕ, all_real_roots (f_iterate c x n)) ∧ c = 4 := sorry


end smallest_c_value_l804_804534


namespace find_angle_l804_804418

theorem find_angle (φ : ℝ) : cos (10 * real.pi / 180) = sin (15 * real.pi / 180) + sin φ ↔ φ = 42.5 * real.pi / 180 := 
by sorry

end find_angle_l804_804418


namespace sum_of_cubes_mod_6_l804_804425

theorem sum_of_cubes_mod_6:
  (∑ i in finset.range 151, i^3) % 6 = 3 :=
by
  have h: ∀ n, n % 6 = n^3 % 6 := sorry
  sorry

end sum_of_cubes_mod_6_l804_804425


namespace general_term_formula_Sn_formula_l804_804453

-- Given conditions
variable (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℚ)
variable (d : ℕ)
variable (a4 : a 4 = 4)
variable (a3a7 : a 3 + a 7 = 10)

noncomputable def a_n_formula (n : ℕ) : ℕ :=
  n

theorem general_term_formula : ∀ n, a n = a_n_formula n := sorry

theorem Sn_formula : ∀ S b,
  (∀ n, b n = (1 : ℚ) / (a n * a (n + 1))) →
  (S 1 = b 1) →
  (∀ n, S (n + 1) = S n + b (n + 1)) →
  ∀ n, S n = n / (n + 1) :=
sorry

end general_term_formula_Sn_formula_l804_804453


namespace maximum_players_advancing_l804_804672

theorem maximum_players_advancing
  (total_players : ℕ := 16)
  (total_games : ℕ := 120)
  (advance_points : ℚ := 10)
  (total_points : ℚ := 120)
  : ∃ (max_advancing_players : ℕ), max_advancing_players = 11 :=
by
  have h_total_players : total_players = 16 := rfl
  have h_total_games : total_games = (16 * 15) / 2 := rfl
  have h_total_points : total_points = (total_games : ℚ) := rfl
  have h_advance_points : advance_points = 10 := rfl
  use 11
  sorry

end maximum_players_advancing_l804_804672


namespace f_2m_equality_l804_804533

def f (k : ℕ) : ℕ := 
  ∑ n in Finset.range (10^k), 
    if (∃ (perm : List ℕ), 
          n.digits.decimal.isPerm₂ perm ∧
          (perm.sum ≡ 0 [MOD 11])) then 1 else 0

theorem f_2m_equality (m : ℕ) (h : m > 0) : 
  f (2 * m) = 10 * f (2 * m - 1) := 
sorry

end f_2m_equality_l804_804533


namespace collinear_midpoints_and_center_l804_804202

noncomputable def projections_on_side (X : Point) (P : Quadrilateral) : List Point := sorry -- Defines projections M, N, P, Q

structure Quadrilateral :=
(A : Point) (B : Point) (C : Point) (D : Point)
(convex : ConvexQuadrilateral A B C D)

structure Point :=
(x : ℝ) (y : ℝ)

def is_center_of_circle (L : Point) (points : List Point) : Prop :=
∃ r : ℝ, ∀ p ∈ points, dist L p = r

def is_midpoint (J : Point) (P1 P2 : Point) : Prop :=
dist J P1 = dist J P2

def lie_on_same_line (J K L : Point) : Prop :=
collinear {J, K, L}

theorem collinear_midpoints_and_center
    (P : Quadrilateral) (X : Point)
    (L : Point := center_of_circle_of_projections X P)
    (J K : Point := midpoints_of_diagonals_of P) :
    let points := projections_on_side X P in
    is_center_of_circle L points →
    (is_midpoint J (P.A P.C)) →
    (is_midpoint K (P.B P.D)) →
    lie_on_same_line J K L :=
by
    intro points points_center_midpoint_collinear sorry

end collinear_midpoints_and_center_l804_804202


namespace total_time_to_fill_tank_l804_804723

-- Define the rates of each pipe filling the tank
def rate_pipe1 : ℝ := 1 / 20
def rate_pipe2 : ℝ := 1 / 30

-- Define the combined rate without the leak
def combined_rate : ℝ := rate_pipe1 + rate_pipe2

-- Define the effective rate accounting for the leak
def effective_rate : ℝ := (2 / 3) * combined_rate

-- Prove that the total time taken to fill the tank is 18 hours
theorem total_time_to_fill_tank : 1 / effective_rate = 18 := 
by
  sorry

end total_time_to_fill_tank_l804_804723


namespace eraser_cost_l804_804373

variable (P E : ℝ)
variable (h1 : E = P / 2)
variable (h2 : 20 * P = 80)

theorem eraser_cost : E = 2 := by 
  sorry

end eraser_cost_l804_804373


namespace sin_870_equals_half_l804_804048

theorem sin_870_equals_half :
  sin (870 * Real.pi / 180) = 1 / 2 := 
by
  -- Angle simplification
  have h₁ : 870 - 2 * 360 = 150 := by norm_num,
  -- Sine identity application
  have h₂ : sin (150 * Real.pi / 180) = sin (30 * Real.pi / 180) := by
    rw [mul_div_cancel_left 150 (ne_of_lt (by norm_num : 0 < (Real.pi : ℝ)) : (Real.pi : ℝ) ≠ 0)],
    congr,
    norm_num,

  -- Sine 30 degrees value
  have h₃ : sin (30 * Real.pi / 180) = 1 / 2 := by norm_num,

  -- Combine results
  rw [mul_div_cancel_left 870 (ne_of_lt (by norm_num : 0 < (Real.pi : ℝ)) : (Real.pi : ℝ) ≠ 0), h₁, h₂, h₃],
  sorry

end sin_870_equals_half_l804_804048


namespace part1_part2_l804_804550

variable {ℝ : Type _} [LinearOrder ℝ]

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f(x) < f(y)

variables (f : ℝ → ℝ) (h_odd : odd_function f)
  (h_inc : increasing_function f) (h_f1 : f 1 = 2)

theorem part1 : f (-1) = -2 := 
sorry

theorem part2 (t : ℝ) (h : f (t^2 - 3*t + 1) < -2) : 1 < t ∧ t < 2 :=
sorry

end part1_part2_l804_804550


namespace max_abs_eq_one_vertices_l804_804070

theorem max_abs_eq_one_vertices (x y : ℝ) :
  (max (|x + y|) (|x - y|) = 1) ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = -1) ∨ (x = 0 ∧ y = 1) :=
sorry

end max_abs_eq_one_vertices_l804_804070


namespace exists_special_x_l804_804942

def satisfies_conditions (f : ℕ → ℕ) : Prop :=
  (∀ x : ℕ, 1 ≤ f(x) - x ∧ f(x) - x ≤ 2019) ∧ 
  (∀ x : ℕ, f(f(x)) % 2019 = x % 2019)

theorem exists_special_x (f : ℕ → ℕ) (h : satisfies_conditions f) : 
  ∃ x : ℕ, ∀ k : ℕ, (nat.iterate f k) x = x + 2019 * k :=
sorry

end exists_special_x_l804_804942


namespace correct_propositions_l804_804208

variables (α β γ : Plane) (l m n : Line)

-- Propositions as conditions
def prop1 : Prop := (α ⊥ γ ∧ β ⊥ γ) → α ∥ β
def prop2 : Prop := (α ∥ β ∧ l ⊆ α) → l ∥ β
def prop3 : Prop := (m ⊆ α ∧ n ⊆ α ∧ m ∥ β ∧ n ∥ β) → α ∥ β
def prop4 : Prop := (α ∩ β = l ∧ β ∩ γ = m ∧ γ ∩ α = n ∧ l ∥ γ) → m ∥ n

-- The main theorem statement
theorem correct_propositions :
  (prop2 α β l ∧ prop4 α β γ l m n) ∧ ¬(prop1 α β γ) ∧ ¬(prop3 α β m n) :=
by
  -- Using sorry to skip the proof
  sorry

end correct_propositions_l804_804208


namespace initial_interval_for_bisection_l804_804806

noncomputable def f : ℝ → ℝ := λ x, 2^x - 3

theorem initial_interval_for_bisection : 
  (∀ x, continuous_at f x) → 
  (f 1 < 0) → 
  (f 2 > 0) → 
  ∃ c ∈ Ioo 1 2, f c = 0 :=
by
  sorry

end initial_interval_for_bisection_l804_804806


namespace secret_code_count_l804_804187

-- Conditions
def num_colors : ℕ := 8
def num_slots : ℕ := 5

-- The proof statement
theorem secret_code_count : (num_colors ^ num_slots) = 32768 := by
  sorry

end secret_code_count_l804_804187


namespace constant_term_jaclyn_l804_804229

noncomputable theory

open Polynomial

-- Define the polynomials and their properties
theorem constant_term_jaclyn (p q : Polynomial ℝ) (h1 : p.natDegree = 3)
  (h2 : q.natDegree = 3) (h3 : p.leadingCoeff = 1) 
  (h4 : q.leadingCoeff = 1) (h5 : p.coeff 0 = q.coeff 0)
  (h6 : p * q = Polynomial.ofCoeffVector [9, 2, 3, 4, 3, 2, 1]) : p.coeff 0 = 3 :=
by
  sorry -- Proof would go here if required

end constant_term_jaclyn_l804_804229


namespace problem_statement_l804_804127

noncomputable def term_with_largest_binomial_coefficient
  (M N P : ℕ)
  (h_sum : M + N - P = 2016)
  (n : ℕ) : ℤ :=
-8064

noncomputable def term_with_largest_absolute_value_coefficient
  (M N P : ℕ)
  (h_sum : M + N - P = 2016)
  (n : ℕ) : ℤ × ℕ :=
(-15360, 8)

theorem problem_statement (M N P : ℕ) (h_sum : M + N - P = 2016) (n : ℕ) :
  ((term_with_largest_binomial_coefficient M N P h_sum n = -8064) ∧ 
   (term_with_largest_absolute_value_coefficient M N P h_sum n = (-15360, 8))) :=
by {
  -- proof goes here
  sorry
}

end problem_statement_l804_804127


namespace infinite_rel_prime_set_of_form_2n_minus_3_l804_804982

theorem infinite_rel_prime_set_of_form_2n_minus_3 : ∃ S : Set ℕ, (∀ x ∈ S, ∃ n : ℕ, x = 2^n - 3) ∧ 
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y → Nat.gcd x y = 1) ∧ S.Infinite := 
by
  sorry

end infinite_rel_prime_set_of_form_2n_minus_3_l804_804982


namespace general_term_formula_sum_of_first_n_terms_l804_804108

-- Define the arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℕ) := ∃ d : ℕ, ∀ n : ℕ, a (n+1) - a n = d

-- {a_n} is an increasing arithmetic sequence, a₁, a₂, a₄ form a geometric sequence, and a₁ ≠ 1
variables {a : ℕ → ℕ} (h_arith : arithmetic_seq a) (h_geom : a 2^2 = a 1 * a 4) (h_a1_ne_1 : a 1 ≠ 1)

-- Prove the general term formula for {a_n}
theorem general_term_formula :
  ∃ c, ∀ n, a n = c * n :=
begin
  sorry
end

-- Define b_n and T_n based on given conditions
noncomputable def b (a : ℕ → ℕ) (n : ℕ) := Real.log (1 + 1 / (a n)) / Real.log 2
noncomputable def T (a : ℕ → ℕ) (n : ℕ) := ∑ i in Finset.range n, b a (i + 1)

-- Define the sequence to find the sum {1/(2^{T_n} * 2^{T_{n+1}})}
def seq (a : ℕ → ℕ): ℕ → ℝ := λ n, 1 / (2^(T a n) * 2^(T a (n + 1)))

-- Prove the sum of the first n terms S_n for the sequence
theorem sum_of_first_n_terms (n : ℕ) :
  ∑ i in Finset.range n, seq a i = n / (2 * n + 4) :=
begin
  sorry
end

end general_term_formula_sum_of_first_n_terms_l804_804108


namespace range_of_a_l804_804862

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, x > a ∧ x ≤ 7 → integer x ∧ 3 ≤ x ∧ x ≤ 7) →
  (∃ N : ℕ, N = 5) →
  (∃ a', 2 ≤ a' ∧ a' < 3 ∧ a' = a) := 
by 
  assume h condition,
  sorry

end range_of_a_l804_804862


namespace statistical_data_comparison_l804_804368

variables {x1 x2 x3 x4 x5 : ℝ}

def mean (x1 x2 x3 x4 x5 : ℝ) : ℝ := (x1 + x2 + x3 + x4 + x5) / 5
def variance (x1 x2 x3 x4 x5 : ℝ) : ℝ := ((x1 - mean x1 x2 x3 x4 x5)^2 + (x2 - mean x1 x2 x3 x4 x5)^2 + (x3 - mean x1 x2 x3 x4 x5)^2 + (x4 - mean x1 x2 x3 x4 x5)^2 + (x5 - mean x1 x2 x3 x4 x5)^2) / 5
def stddev (x1 x2 x3 x4 x5 : ℝ) : ℝ := sqrt (variance x1 x2 x3 x4 x5)
def median (x1 x2 x3 x4 x5 : ℝ) : ℝ := x3

theorem statistical_data_comparison : 
  mean (2 * x1 + 3) (2 * x2 + 3) (2 * x3 + 3) (2 * x4 + 3) (2 * x5 + 3) ≠ mean x1 x2 x3 x4 x5 ∧
  stddev (2 * x1 + 3) (2 * x2 + 3) (2 * x3 + 3) (2 * x4 + 3) (2 * x5 + 3) ≠ stddev x1 x2 x3 x4 x5 ∧
  median (2 * x1 + 3) (2 * x2 + 3) (2 * x3 + 3) (2 * x4 + 3) (2 * x5 + 3) ≠ median x1 x2 x3 x4 x5 :=
by
  -- Proof is omitted as per instruction
  sorry

end statistical_data_comparison_l804_804368


namespace solution_l804_804865

def system (a b : ℝ) : Prop :=
  (2 * a + b = 3) ∧ (a - b = 1)

theorem solution (a b : ℝ) (h: system a b) : a + 2 * b = 2 :=
by
  cases h with
  | intro h1 h2 => sorry

end solution_l804_804865


namespace neznika_number_l804_804573

theorem neznika_number (N : ℕ) :
  10 ≤ N ∧ N ≤ 99 ∧
  (divisible_by N 3 ∧ divisible_by N 4 ∧ divisible_by N 5 ∧
   divisible_by N 9 ∧ divisible_by N 10 ∧ divisible_by N 15 ∧
   divisible_by N 18 ∧ divisible_by N 30) →
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end neznika_number_l804_804573


namespace range_of_b_minus_c_l804_804845

theorem range_of_b_minus_c {f : ℝ → ℝ}
  (h1 : ∃ x1 x2 x3 : ℝ, (x1 + x2 + x3 = 0) ∧ (f = λ x, x^3 + b * x^2 + c * x + c) ∧
       (f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0)) :
  (∃ lb : ℝ, (lb = 27/4) ∧ ∀ b c : ℝ, (b - c) ∈ (lb, +∞)) :=
sorry

end range_of_b_minus_c_l804_804845


namespace combined_garden_perimeter_l804_804359

theorem combined_garden_perimeter :
  let l_garden := 15;
  let w_garden := 10;
  let h_triangle := 6;
  let base_triangle := l_garden;
  let P_rectangle := 2 * (l_garden + w_garden);
  let hypotenuse_triangle := Real.sqrt (15^2 + 6^2);
  let extra_perimeter := hypotenuse_triangle + h_triangle;
  let P_total := P_rectangle + extra_perimeter - base_triangle;
  P_total = 41 + Real.sqrt(261) :=
by
  sorry

end combined_garden_perimeter_l804_804359


namespace parallel_lines_implies_m_no_perpendicular_lines_solution_l804_804144

noncomputable def parallel_slopes (m : ℝ) : Prop :=
  let y₁ := -m
  let y₂ := -2 / m
  y₁ = y₂

noncomputable def perpendicular_slopes (m : ℝ) : Prop :=
  let y₁ := -m
  let y₂ := -2 / m
  y₁ * y₂ = -1

theorem parallel_lines_implies_m (m : ℝ) : parallel_slopes m ↔ m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
by
  sorry

theorem no_perpendicular_lines_solution (m : ℝ) : perpendicular_slopes m → false :=
by
  sorry

end parallel_lines_implies_m_no_perpendicular_lines_solution_l804_804144


namespace count_similar_divisors_l804_804010

def is_integrally_similar_divisible (a b c : ℕ) : Prop :=
  ∃ x y z : ℕ, a * c = b * z ∧
  x ≤ y ∧ y ≤ z ∧
  b = 2023 ∧ a * c = 2023^2

theorem count_similar_divisors (b : ℕ) (hb : b = 2023) :
  ∃ (n : ℕ), n = 7 ∧ 
    (∀ (a c : ℕ), a ≤ b ∧ b ≤ c → is_integrally_similar_divisible a b c) :=
by
  sorry

end count_similar_divisors_l804_804010


namespace isosceles_triangle_perimeter_l804_804109

theorem isosceles_triangle_perimeter (a b c : ℝ) 
  (h1 : a = 4 ∨ b = 4 ∨ c = 4) 
  (h2 : a = 8 ∨ b = 8 ∨ c = 8) 
  (isosceles : a = b ∨ b = c ∨ a = c) : 
  a + b + c = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l804_804109


namespace circumcircles_common_point_l804_804842

noncomputable theory

variables {A B C D E F O : Type} [euclidean_space ℝ A B C D E F O]

def lines_intersect_four_triangles (A B C D E F : euclidean_space ℝ A B C D E F) : Prop :=
∃ (AE AF ED FB : set (euclidean_space ℝ A B C D E F)),
  -- Points of intersection
  AE ∩ AF = {A} ∧
  ED ∩ FB = {D} ∧
  -- Formation of triangles
  triangle A B F ∧ triangle A E D ∧ triangle B C E ∧ triangle D C F
  
def circumcircles_concur (A B C D E F O : euclidean_space ℝ A B C D E F) : Prop :=
∃ (O : euclidean_space ℝ A B C D E F),
  -- Circumcircles concurrence definitions
  circumcircle (triangle A B F) O ∧ circumcircle (triangle A E D) O ∧
  circumcircle (triangle B C E) O ∧ circumcircle (triangle D C F) O

theorem circumcircles_common_point
  (A B C D E F : euclidean_space ℝ A B C D E F) :
  lines_intersect_four_triangles A B C D E F → circumcircles_concur A B C D E F :=
begin
  intros h,
  sorry, -- The proof goes here
end

end circumcircles_common_point_l804_804842


namespace salmon_oxygen_ratio_l804_804413

theorem salmon_oxygen_ratio (v : ℝ) (O1 O2 : ℝ) (hv1 : v = 1 / 2 * log 3 (O1 / 100))
    (hv2 : v + 2 = 1 / 2 * log 3 (O2 / 100)) : O2 / O1 = 81 :=
sorry

end salmon_oxygen_ratio_l804_804413


namespace find_inverse_and_range_l804_804824

-- Define the function f
def f (x : ℝ) (h : x > 1) : ℝ := ( (x - 1) / (x + 1) ) ^ 2

-- Define its inverse function f_inv
def f_inv (y : ℝ) (h : 0 < y ∧ y < 1) : ℝ := ( real.sqrt y + 1 ) / ( 1 - real.sqrt y )

-- State the main theorem
theorem find_inverse_and_range (x a : ℝ) (hx : 1/4 ≤ x ∧ x ≤ 1/2) :
  (∃ h : x > 1, f_inv (f x h) ⟷ x) ∧
  (∀ hx' : 1/4 ≤ x ∧ x ≤ 1/2, (1 - real.sqrt x) * f_inv x ⟷ a * (a - real.sqrt x) → (-1 < a ∧ a < 3/2)) :=
sorry

end find_inverse_and_range_l804_804824


namespace sqrt_value_l804_804445

theorem sqrt_value {A B C : ℝ} (x y : ℝ) 
  (h1 : A = 5 * Real.sqrt (2 * x + 1)) 
  (h2 : B = 3 * Real.sqrt (x + 3)) 
  (h3 : C = Real.sqrt (10 * x + 3 * y)) 
  (h4 : A + B = C) 
  (h5 : 2 * x + 1 = x + 3) : 
  Real.sqrt (2 * y - x^2) = 14 :=
by
  sorry

end sqrt_value_l804_804445


namespace find_a12_a14_l804_804452

noncomputable def S (n : ℕ) (a_n : ℕ → ℝ) (b : ℝ) : ℝ := a_n n ^ 2 + b * n

noncomputable def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ (a1 : ℝ) (c : ℝ), ∀ n : ℕ, a_n n = a1 + (n - 1) * c

theorem find_a12_a14
  (a_n : ℕ → ℝ)
  (b : ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, S n = a_n n ^ 2 + b * n)
  (h2 : S 25 = 100)
  (h3 : is_arithmetic_sequence a_n) :
  a_n 12 + a_n 14 = 5 :=
sorry

end find_a12_a14_l804_804452


namespace number_of_valid_as_l804_804979

theorem number_of_valid_as : 
  ∃ (a b c d : ℕ), a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2200 ∧
  a^2 - b^2 + c^2 - d^2 = 2200 ∧
  (set.range (λ a, ∃ b c d, a > b ∧ b > c ∧ c > d ∧
    a + b + c + d = 2200 ∧
    a^2 - b^2 + c^2 - d^2 = 2200).card = 548) :=
sorry

end number_of_valid_as_l804_804979


namespace possible_numbers_l804_804589

theorem possible_numbers (N : ℕ) 
    (h1 : 10 ≤ N) (h2 : N ≤ 99)
    (h3 : (N % 3 = 0) ∧ (N % 4 = 0) ∧ (N % 5 = 0) ∧ (N % 9 = 0) ∧ (N % 10 = 0) ∧ 
          (N % 15 = 0) ∧ (N % 18 = 0) ∧ (N % 30 = 0) ∨ 
          (N % 3 ≠ 0) + (N % 4 ≠ 0) + (N % 5 ≠ 0) + (N % 9 ≠ 0) + 
          (N % 10 ≠ 0) + (N % 15 ≠ 0) + (N % 18 ≠ 0) + (N % 30 ≠ 0) = 4) :
   N = 36 ∨ N = 45 ∨ N = 72 :=
by {
  sorry
}

end possible_numbers_l804_804589


namespace range_of_a_l804_804172

-- Define the function
def f (a x : ℝ) : ℝ := x^2 - a * x - a

-- Conditions
def range_is_R (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = log (f a x)
def decreasing_on_I (a : ℝ) : Prop := ∀ x1 x2 : ℝ, (x1 < x2) → (x1 < 1 - real.sqrt 3) → (x2 < 1 - real.sqrt 3) → (f a x2 < f a x1)

-- Define the statement
theorem range_of_a (a : ℝ) :
  range_is_R a ∧ decreasing_on_I a → (0 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l804_804172


namespace circle_tangent_to_parabola_height_difference_l804_804732

theorem circle_tangent_to_parabola_height_difference (a : ℝ) : 
  -- Circle is inside and tangent to the parabola y = 2x^2 at points (a, 2a^2) and (-a, 2a^2).
  let b : ℝ := a^2 + 1/4 in
  -- The height difference between the center and the tangency points is 1/4 - a^2.
  (b - 2 * a^2 = 1/4 - a^2) :=
begin
  sorry
end

end circle_tangent_to_parabola_height_difference_l804_804732


namespace triangle_intersection_ratios_eq_one_l804_804715

theorem triangle_intersection_ratios_eq_one
  (A B C A1 B1 C1 : Type)
  [triangle ABC]
  [point_on_line C1 (line_segment A B)]
  [point_on_line A1 (line_segment B C)]
  [point_on_extension B1 (line_extension A C)] :
  (length_ratio (segment B A1) (segment A1 C)) *
  (length_ratio (segment C B1) (segment B1 A)) *
  (length_ratio (segment A C1) (segment C1 B)) = 1 :=
by sorry

end triangle_intersection_ratios_eq_one_l804_804715


namespace sin_870_eq_half_l804_804042

theorem sin_870_eq_half : Real.sin (870 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_870_eq_half_l804_804042


namespace num_of_valid_divisors_of_15_factorial_l804_804807

theorem num_of_valid_divisors_of_15_factorial : 
by { let n := (15.factorial), let valid_d := (λ d, d ∣ n ∧ Int.gcd d 60 = 5),
     have f : ∀ d, valid_d d → get_valid_divisors_count d n,
  sorry }

end num_of_valid_divisors_of_15_factorial_l804_804807


namespace candle_burn_down_time_candle_height_after_half_time_l804_804352

noncomputable def total_time (n : ℕ) : ℕ :=
  10 * ∑ k in finset.range (n + 1), k^2

theorem candle_burn_down_time :
  total_time 150 = 11_380_250 := by
  sorry

noncomputable def height_after_half_time (n : ℕ) (T : ℕ) : ℕ :=
  let half_time := T / 2
  let m := ((∑ k in finset.range (n + 1), k^2) / 10 + half_time) / (20 * half_time + 1)
  n - m

theorem candle_height_after_half_time :
  height_after_half_time 150 (total_time 150) = 68 := by
  sorry

end candle_burn_down_time_candle_height_after_half_time_l804_804352


namespace find_q_7_l804_804958

noncomputable def q (x : ℝ) : ℝ := (2/21) * x^2 - x + (76/21)

theorem find_q_7 :
  let q := λ x : ℝ, (2/21) * x^2 - x + (76/21)
  in q 7 = 9/7 :=
by
  let q := λ x : ℝ, (2/21) * x^2 - x + (76/21)
  rw [(show q 7 = (2/21) * 49 - 7 + (76/21), by norm_num), (show (2/21) * 49 = 98/21, by norm_num), (show 98/21 - 7 = 98/21 - 147/21, by norm_num), (show 98/21 - 147/21 = -49/21, by norm_num), (show -49/21 + 76/21 = 27/21, by norm_num), (show 27/21 = 9/7, by norm_num)]
  rfl

end find_q_7_l804_804958


namespace sum_series_base6_proof_l804_804068

def sum_series_base6 (a b : ℕ) : ℕ :=
  let start := a
  let end := b
  (34 * (start + end)) / 2

theorem sum_series_base6_proof (a b : ℕ) (h1 : a = 3) (h2 : b = 36) : 
  sum_series_base6 a b = 3023 :=
by
  sorry

end sum_series_base6_proof_l804_804068


namespace robert_coin_arrangements_l804_804626

-- Define the main problem statement
theorem robert_coin_arrangements:
  let gold_coins := 4 in
  let silver_coins := 4 in
  let total_coins := gold_coins + silver_coins in
  let no_adjacent_faces :=
    (∀ i ∈ finset.range(total_coins - 1), stack !i ≠ stack !i.succ) in
  (#arrangements_of_coins_with_no_adjacent_faces gold_coins silver_coins total_coins no_adjacent_faces) = 630 :=
sorry

end robert_coin_arrangements_l804_804626


namespace exponentiation_of_64_l804_804391

theorem exponentiation_of_64 : (64:ℝ)^(3/4) = 16 * real.sqrt 2 :=
by sorry

end exponentiation_of_64_l804_804391


namespace num_two_digit_integers_congruent_to_1_mod_4_l804_804484

theorem num_two_digit_integers_congruent_to_1_mod_4 : 
  let count := (range [4*k+1 | k, 3 ≤ k ∧ k ≤ 24]).length in
  count = 22 :=
by
  sorry

end num_two_digit_integers_congruent_to_1_mod_4_l804_804484


namespace mike_total_spent_l804_804090

theorem mike_total_spent (spent_on_speakers : ℝ) (spent_on_tires : ℝ) (spent_on_cds : ℝ) (num_cds : ℕ) (cd_price : ℝ) (did_not_buy_cds : num_cds = 3 ∧ cd_price = 4.58 ∧ spent_on_cds = 0) :
  spent_on_speakers = 118.54 ∧ spent_on_tires = 106.33 → 
  spent_on_speakers + spent_on_tires = 224.87 :=
by {
  intro h,
  have hs := h.left,
  have ht := h.right,
  rw [hs],
  rw [ht],
  norm_num,
  sorry
}

end mike_total_spent_l804_804090


namespace sqrt_3_between_neg_1_and_2_l804_804021

theorem sqrt_3_between_neg_1_and_2 : -1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by
  sorry

end sqrt_3_between_neg_1_and_2_l804_804021


namespace jason_total_spending_l804_804933

theorem jason_total_spending:
  let stove_cost : ℕ := 1200
  let wall_repair_cost := stove_cost / 6
  let total_repair_cost := stove_cost + wall_repair_cost
  let labor_fees := total_repair_cost * 20 / 100
  let total_cost := total_repair_cost + labor_fees
  total_cost = 1680 :=
by
  let stove_cost : ℕ := 1200
  let wall_repair_cost := stove_cost / 6
  let total_repair_cost := stove_cost + wall_repair_cost
  let labor_fees := total_repair_cost * 20 / 100
  let total_cost := total_repair_cost + labor_fees
  simp [stove_cost, wall_repair_cost, total_repair_cost, labor_fees, total_cost]
  sorry

end jason_total_spending_l804_804933


namespace T_8_is_1715_l804_804541

def T (n : Nat) : Nat :=
  if n = 1 then 0
  else (B1 n) + (B2 n) + (B3 n)

def B1 (n : Nat) : Nat :=
  if n = 1 then 2
  else T (n - 1) + B1 (n - 1)

def B2 (n : Nat) : Nat :=
  if n = 1 then 1
  else B1 (n - 1)

def B3 (n : Nat) : Nat :=
  if n = 1 then 0
  else B2 (n - 1)

theorem T_8_is_1715 : T 8 = 1715 := 
by
  -- proof goes here
  sorry

end T_8_is_1715_l804_804541


namespace eqn_of_circle_fixed_point_passing_line_l804_804475

-- Define the parabola, focus, directrix, and origin
def parabola (x y : ℝ) := y^2 = 4 * x
def directrix := λ x : ℝ, x = -1
def focus := (1 : ℝ, 0 : ℝ)
def origin := (0 : ℝ, 0 : ℝ)

-- Question 1: Circle equation
def circle_eqn (x y : ℝ) := 
  ∃ (a b r : ℝ), a = 1 / 2 ∧ r = 3 / 2 ∧ (x - a)^2 + (y - b)^2 = r^2 ∧ (b = sqrt 2 ∨ b = -sqrt 2)

-- The Lean statement for the first question
theorem eqn_of_circle :
  ∃ (a b r : ℝ), a = 1 / 2 ∧ r = 3 / 2 ∧ 
  (b = sqrt 2 ∨ b = -sqrt 2) ∧ 
  ∀ (x y : ℝ), circle_eqn x y → (x - a)^2 + (y - b)^2 = r^2 :=
sorry

-- Question 2: Line passing through fixed point
def symmetric_point (A : ℝ × ℝ) := (A.1, -A.2)
def is_on_parabola (A : ℝ × ℝ) := parabola A.1 A.2
def is_on_line (A B : ℝ × ℝ) := B.1 ≠ A.1

-- The Lean statement for the second question
theorem fixed_point_passing_line :
  ∀ (A B : ℝ × ℝ), 
  is_on_parabola A →
  is_on_parabola B →
  is_on_line focus focus ↔ 
  let A' := symmetric_point A in let M := (-1 : ℝ, 0 : ℝ) in 
  collinear A' B M :=
sorry

end eqn_of_circle_fixed_point_passing_line_l804_804475


namespace shaded_region_area_equals_l804_804733

open Set

noncomputable def radius := 4
noncomputable def side_length := 2
noncomputable def area_shaded_region : ℝ := (16 * Real.pi / 3) - 6 * Real.sqrt 3 + 4

theorem shaded_region_area_equals :
  let O : ℂ := 0
  let A : ℂ := complex.of_real side_length
  let B : ℂ := complex.of_real (side_length + I * side_length)
  let C : ℂ := complex.of_real (I * side_length)
  let D : ℂ := complex.of_real (radius * real.cos (real.pi / 3) + I * radius * real.sin (real.pi / 3))
  let E : ℂ := complex.of_real (radius * real.cos (2 * real.pi / 3) + I * radius * real.sin (2 * real.pi / 3))
  ∀ (O A B C D E : ℂ), 
    O = 0 ∧
    complex.abs O A = side_length ∧
    complex.abs O A = side_length ∧
    complex.abs O D = radius ∧
    complex.abs O E = radius  →
    area_shaded_region = (16 * Real.pi / 3) - 6 * Real.sqrt 3 + 4 :=
by
  sorry

end shaded_region_area_equals_l804_804733


namespace right_triangle_sides_l804_804364

noncomputable def triangle_sides (k : ℝ) (rho : ℝ) : ℝ × ℝ × ℝ :=
  if k = 40 ∧ rho = 3 then (15, 8, 17) else (0, 0, 0)

theorem right_triangle_sides :
  ∀ (k rho : ℝ), k = 40 → rho = 3 →
  triangle_sides k rho = (15, 8, 17) := by
  intros k rho hk hrho
  simp [triangle_sides, hk, hrho]
  rfl

end right_triangle_sides_l804_804364


namespace units_digit_is_3_l804_804810

-- Define the relevant expressions
def A : ℝ := 17 + real.sqrt 252
def B : ℝ := 17 - real.sqrt 252

-- Define the expression whose units digit we want to find
def expression (n m p : ℕ) := A^n + B^m + 3 * A^p

-- Prove that the units digit is 3
theorem units_digit_is_3 : 
  (∃ n m p,
    n = 20 ∧ 
    m = 54 ∧ 
    p = 100 ∧ 
    (expression n m p) % 10 = 3) := 
by
  use [20, 54, 100]
  have h1 : A^20 % 10 = 1 := sorry
  have h2 : B^54 % 10 = 9 := sorry
  have h3 : 3 * A^100 % 10 = 3 := sorry
  calc
    (A^20 + B^54 + 3*A^100) % 10 
        = (1 + 9 + 3) % 10 := by rw [h1, h2, h3]
    ... = 13 % 10 := by norm_num
    ... = 3 := by norm_num

end units_digit_is_3_l804_804810


namespace num_mittens_per_box_eq_six_l804_804393

theorem num_mittens_per_box_eq_six 
    (num_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ)
    (h1 : num_boxes = 4) (h2 : scarves_per_box = 2) (h3 : total_clothing = 32) :
    (total_clothing - num_boxes * scarves_per_box) / num_boxes = 6 :=
by
  sorry

end num_mittens_per_box_eq_six_l804_804393


namespace domain_of_f_l804_804695

noncomputable def f (x : ℝ) := Real.log x / Real.log 6

noncomputable def g (x : ℝ) := Real.log x / Real.log 5

noncomputable def h (x : ℝ) := Real.log x / Real.log 3

open Set

theorem domain_of_f :
  (∀ x, x > 7776 → ∃ y, y = (h ∘ g ∘ f) x) :=
by
  sorry

end domain_of_f_l804_804695


namespace num_integers_with_factors_l804_804155

theorem num_integers_with_factors (a b lcm : ℕ) (lower upper : ℕ) (h_lcm : lcm = Nat.lcm a b) :
  (36 = Nat.lcm 12 9) → (a = 12) → (b = 9) → (lower = 200) → (upper = 500) →
  (finset.filter (λ x, x % lcm = 0) (finset.Icc lower upper)).card = 8 :=
by
  sorry

end num_integers_with_factors_l804_804155


namespace find_direction_vector_l804_804353

def line_parametrization (v d : ℝ × ℝ) (t x y : ℝ) : ℝ × ℝ :=
  (v.fst + t * d.fst, v.snd + t * d.snd)

theorem find_direction_vector : 
  ∀ d: ℝ × ℝ, ∀ t: ℝ,
    ∀ (v : ℝ × ℝ) (x y : ℝ), 
    v = (-3, -1) → 
    y = (2 * x + 3) / 5 →
    x + 3 ≤ 0 →
    dist (line_parametrization v d t x y) (-3, -1) = t →
    d = (5/2, 1) :=
by
  intros d t v x y hv hy hcond hdist
  sorry

end find_direction_vector_l804_804353


namespace minimum_sin_function_l804_804684

-- Definitions and conditions
def f (x : ℝ) : ℝ := Real.sin (2 * x - π / 3)
def interval := set.Icc 0 (π / 2)

-- The statement to be proved
theorem minimum_sin_function :
  is_min_on f interval (- (Real.sqrt 3 / 2)) :=
sorry

end minimum_sin_function_l804_804684


namespace cupcakes_frosted_l804_804768

noncomputable def CagneyRate : ℝ := 1/15
noncomputable def LaceyRate : ℝ := 1/25
noncomputable def workingTimePerCycle : ℝ := 180 -- 3 minutes in seconds
noncomputable def breakTimePerCycle : ℝ := 30
noncomputable def cycleTime : ℝ := workingTimePerCycle + breakTimePerCycle
noncomputable def totalTime : ℝ := 900 -- 15 minutes in seconds

theorem cupcakes_frosted : 
  let combinedRate := CagneyRate + LaceyRate in
  let cycleEffectiveCount := (totalTime / cycleTime).floor in
  let effectiveWorkTime := cycleEffectiveCount * workingTimePerCycle in
  let combinedFrostingTime := 1 / combinedRate in
  (effectiveWorkTime / combinedFrostingTime).floor = 76 :=
by
  sorry

end cupcakes_frosted_l804_804768


namespace sum_of_x_coords_Q3_l804_804341

-- Define the initial conditions:
def initial_x_sum (n : ℕ) (x_coords : list ℝ) := x_coords.sum = 150
def scaling_factor := 1.5

-- Define the problem statement:
theorem sum_of_x_coords_Q3 (x_coords : list ℝ) (h₁ : initial_x_sum 50 x_coords) :
  let scaled_x_coords := x_coords.map (λ x, scaling_factor * x),
      Q2_x_coords := list.map₂ (λ x1 x2, (x1 + x2) / 2) scaled_x_coords (scaled_x_coords.tail ++ [scaled_x_coords.head]),
      Q3_x_coords := list.map₂ (λ x1 x2, (x1 + x2) / 2) Q2_x_coords (Q2_x_coords.tail ++ [Q2_x_coords.head])
    in Q3_x_coords.sum = 225 :=
by
  sorry

end sum_of_x_coords_Q3_l804_804341


namespace fraction_multiplication_l804_804387

theorem fraction_multiplication : (1 / 2) * (1 / 3) * (1 / 6) * 108 = 3 := by
  sorry

end fraction_multiplication_l804_804387


namespace triangle_tangent_ratio_l804_804903

theorem triangle_tangent_ratio (A B C : ℝ) (a b c : ℝ)
  (h1 : a^2 + b^2 = 2016 * c^2)
  (h2 : a = 2 * (c * Real.sin A))
  (h3 : b = 2 * (c * Real.sin B))
  (h4 : c = 2 * (c * Real.sin C)) :
  (Real.tan A * Real.tan B) / (Real.tan C * (Real.tan A + Real.tan B)) = 1003 := 
begin
  sorry
end

end triangle_tangent_ratio_l804_804903


namespace find_A_l804_804991

variable (x A B C : ℝ)

theorem find_A :
  (∃ A B C : ℝ, (∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 → 
  (1 / (x^3 + 2 * x^2 - 19 * x - 30) = 
  (A / (x + 3)) + (B / (x - 2)) + (C / (x - 2)^2)) ∧ 
  A = 1 / 25)) :=
by
  sorry

end find_A_l804_804991


namespace technician_round_trip_completion_percentage_l804_804018

theorem technician_round_trip_completion_percentage :
  ∀ (d total_d : ℝ),
  d = 1 + (0.75 * 1) + (0.5 * 1) + (0.25 * 1) →
  total_d = 4 * 2 →
  (d / total_d) * 100 = 31.25 :=
by
  intros d total_d h1 h2
  sorry

end technician_round_trip_completion_percentage_l804_804018


namespace ellipse_equation_max_area_line_l804_804843

/-- Statement 1: The equation of the ellipse **/
theorem ellipse_equation (a b : ℝ) (h1 : 2 * a * b = 2 * real.sqrt 2) (h2 : a^2 = b^2 + b^2) : (set_of (λ p : ℝ × ℝ, p.1 ^ 2 / 2 + p.2 ^ 2 = 1)) :=
begin
  sorry
end

/-- Statement 2: The equation of the line where area of triangle AOB is maximized **/
theorem max_area_line (k : ℝ) (h1 : ∃ A B : ℝ × ℝ, A ≠ B ∧ 
                                               A.2 = k * A.1 + 2 ∧ B.2 = k * B.1 + 2 ∧
                                               (A.1^2) / 2 + (A.2^2) = 1 ∧ (B.1^2) / 2 + (B.2^2) = 1) 
  (h2 : ∀ k1 k2 a b : ℝ, k1 ≠ k2 → 2 * real.sqrt (16 * k1^2 - 24) / (1 + 2 * k1^2) = real.sqrt 2 / 2 → k1 = k2)
  : (set_of (λ p : ℝ × ℝ, p.2 = (√14 / 2 * p.1) + 2) ∨ set_of (λ p : ℝ × ℝ, p.2 = (-√14 / 2 * p.1) + 2)) :=
begin
  sorry
end

end ellipse_equation_max_area_line_l804_804843


namespace rotation_obtuse_angle_l804_804659

-- Define the initial condition with the measure of angle ACB as 70 degrees
def initial_angle_ACB : ℝ := 70

-- Define the rotation angle
def rotation_angle : ℝ := 600

-- State the problem to prove the new angle after rotation
theorem rotation_obtuse_angle :
  ∃ (new_angle : ℝ), new_angle = 170 ∧ new_angle = 
  let reduced_angle := (rotation_angle % 360 : ℝ ) 
  in if reduced_angle > 0 then 180 - reduced_angle % 180 else 180 - (-reduced_angle) % 180 :=
sorry

end rotation_obtuse_angle_l804_804659


namespace sum_of_circle_numbers_l804_804822

theorem sum_of_circle_numbers (a : ℕ → ℝ) (h : ∀ i : ℕ, 0 < a i ∧ (a i + a (i + 1) = a (i + 2) * a (i + 3))) :
  (finset.range 99).sum a = 99 * 2 :=
sorry

end sum_of_circle_numbers_l804_804822


namespace num_four_digit_numbers_l804_804148

theorem num_four_digit_numbers (digits : multiset ℕ) (h1 : digits = {2, 2, 0, 5}) :
  ∃ n : ℕ, n = 15 ∧ 
    ( ∀ l : list ℕ, l ∈ digits.permutations → l.head ≠ 0 ) := 
sorry

end num_four_digit_numbers_l804_804148


namespace min_value_frac_l804_804458

theorem min_value_frac (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 10) : 
  ∃ x, (x = (1 / m) + (4 / n)) ∧ (∀ y, y = (1 / m) + (4 / n) → y ≥ 9 / 10) :=
sorry

end min_value_frac_l804_804458


namespace card_combination_exists_l804_804983

theorem card_combination_exists : ∃ (R B G : ℕ), 
  R + B + G = 20 ∧ 2 ≤ R ∧ 3 ≤ B ∧ 1 ≤ G ∧ 3*R + 5*B + 7*G = 84 := 
by
  use 16, 3, 1
  simp [Nat.add_comm] -- simplifying helps to demonstrate the example solution.
  split; norm_num
  split; norm_num
  split; norm_num
  simp [Nat.add_comm, Nat.mul_comm]
  split; norm_num
  simp [Nat.add_comm, Nat.mul_comm]
sorry

end card_combination_exists_l804_804983


namespace two_digit_integers_congruent_1_mod_4_l804_804482

theorem two_digit_integers_congruent_1_mod_4 : 
  {n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ n % 4 = 1}.card = 22 :=
sorry

end two_digit_integers_congruent_1_mod_4_l804_804482


namespace sum_difference_4041_l804_804814

def sum_of_first_n_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_difference_4041 :
  sum_of_first_n_integers 2021 - sum_of_first_n_integers 2019 = 4041 :=
by
  sorry

end sum_difference_4041_l804_804814


namespace salmon_oxygen_ratio_l804_804412

theorem salmon_oxygen_ratio (v : ℝ) (O1 O2 : ℝ) (hv1 : v = 1 / 2 * log 3 (O1 / 100))
    (hv2 : v + 2 = 1 / 2 * log 3 (O2 / 100)) : O2 / O1 = 81 :=
sorry

end salmon_oxygen_ratio_l804_804412


namespace prob_A_and_B_succeed_prob_vaccine_A_successful_l804_804774

-- Define the probabilities of success for Company A, Company B, and Company C
def P_A := (2 : ℚ) / 3
def P_B := (1 : ℚ) / 2
def P_C := (3 : ℚ) / 5

-- Define the theorem statements

-- Theorem for the probability that both Company A and Company B succeed
theorem prob_A_and_B_succeed : P_A * P_B = 1 / 3 := by
  sorry

-- Theorem for the probability that vaccine A is successfully developed
theorem prob_vaccine_A_successful : 1 - ((1 - P_A) * (1 - P_B)) = 5 / 6 := by
  sorry

end prob_A_and_B_succeed_prob_vaccine_A_successful_l804_804774


namespace two_digit_number_possible_options_l804_804595

theorem two_digit_number_possible_options
  (N : ℕ)
  (h1 : 10 ≤ N ∧ N < 100)
  (h2 : (N % 3 = 0 ∨ N % 3 ≠ 0) ∧
        (N % 4 = 0 ∨ N % 4 ≠ 0) ∧
        (N % 5 = 0 ∨ N % 5 ≠ 0) ∧
        (N % 9 = 0 ∨ N % 9 ≠ 0) ∧
        (N % 10 = 0 ∨ N % 10 ≠ 0) ∧
        (N % 15 = 0 ∨ N % 15 ≠ 0) ∧
        (N % 18 = 0 ∨ N % 18 ≠ 0) ∧
        (N % 30 = 0 ∨ N % 30 ≠ 0)) :
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end two_digit_number_possible_options_l804_804595


namespace sum_ratio_is_one_l804_804538

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)

-- Conditions
def arithmetic_sequence (a_n S_n : ℕ → ℝ) := ∀ n : ℕ, S_n = (n * (a_n 1 + a_n n)) / 2
def ratio_condition (a_n : ℕ → ℝ) := a_n 5 / a_n 3 = 5 / 9

-- The theorem to prove
theorem sum_ratio_is_one (a_n S_n : ℕ → ℝ) 
  [arithmetic_sequence a_n S_n] 
  [ratio_condition a_n] : 
  S_n 9 / S_n 5 = 1 := 
sorry

end sum_ratio_is_one_l804_804538


namespace coeff_x_squared_in_expr_l804_804076

theorem coeff_x_squared_in_expr :
  let expr := 5 * (λ x : ℕ, if x = 1 then 1 else if x = 4 then -1 else 0) 
                   - 4 * (λ x : ℕ, if x = 2 then 2 else if x = 4 then -1 else if x = 6 then 1 else 0)
                   + 3 * (λ x : ℕ, if x = 2 then 3 else if x = 10 then -1 else 0)
  in expr 2 = 1 :=
by
  sorry

end coeff_x_squared_in_expr_l804_804076


namespace quadratic_roots_ratio_l804_804092

theorem quadratic_roots_ratio (p x1 x2 : ℝ) (h_eq : x1^2 + p * x1 - 16 = 0) (h_ratio : x1 / x2 = -4) :
  p = 6 ∨ p = -6 :=
by {
  sorry
}

end quadratic_roots_ratio_l804_804092


namespace max_crosses_in_10x11_board_l804_804309

/-- A cross is defined as occupying 5 unit squares in a plus shape. -/
def cross := {c : set (ℕ × ℕ) // c = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)} }

/-- Board dimensions: 10 by 11 -/
def board_width : ℕ := 10
def board_height : ℕ := 11

/-- A condition representing the size of the board in terms of unit squares -/
def board_size : ℕ := board_width * board_height

/-- The maximal number of crosses that can fit in a 10x11 board without overlapping -/
def maximal_crosses_fit : ℕ := 14

theorem max_crosses_in_10x11_board : 
  ∀ (b : set (ℕ × ℕ)), (∀ (c1 c2 : cross), c1 ≠ c2 → disjoint c1.1 c2.1) →
  ((∀ (x y : ℕ × ℕ), x ∈ b → (1 ≤ x.1 ∧ x.1 ≤ board_width) ∧ (1 ≤ x.2 ∧ x.2 ≤ board_height)) →
  (∃ (n : ℕ), n = maximal_crosses_fit)) :=
begin
  sorry
end

end max_crosses_in_10x11_board_l804_804309


namespace interest_rate_is_correct_l804_804345

variable (principal total_repaid : ℝ)
variable (interest_rate : ℝ)

theorem interest_rate_is_correct (h1 : principal = 220) (h2 : total_repaid = 242) :
  interest_rate = 10 :=
by
  have interest := total_repaid - principal
  have rate := (interest / principal) * 100
  have : interest_rate = rate
  calc
    interest_rate = (242 - 220) / 220 * 100 : by rw [h1, h2]
               ... = 22 / 220 * 100
               ... = 0.1 * 100
               ... = 10
  sorry

end interest_rate_is_correct_l804_804345


namespace ellipse_equation_l804_804832

def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (a^2 - b^2) = 1

theorem ellipse_equation (a b x y : ℝ) 
  (h1 : a > b > 0)
  (h2 : ∃ e, e = (sqrt 3) / 3 ∧ e = 1/a)
  (h3 : ∃ F1 F2 : ℝ × ℝ, ∀ x1 y1, 
  is_ellipse a b (x1 - F1.1 + x1 - F2.1) y1 = true 
  ∧ (4 * sqrt 3) = (4 * (a)))
  : (frac x^2 3 + frac y^2 2 = 1) := 
sorry

end ellipse_equation_l804_804832


namespace num_true_propositions_l804_804156

theorem num_true_propositions : 
  (∀ (a b : ℝ), a = 0 → ab = 0) ∧
  (∀ (a b : ℝ), ab ≠ 0 → a ≠ 0) ∧
  ¬ (∀ (a b : ℝ), ab = 0 → a = 0) ∧
  ¬ (∀ (a b : ℝ), a ≠ 0 → ab ≠ 0) → 
  2 = 2 :=
by 
  sorry

end num_true_propositions_l804_804156


namespace dot_product_eq_one_l804_804900

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_eq_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_eq_one_l804_804900


namespace area_of_equilateral_triangle_DEF_l804_804193

/-- Inside an equilateral triangle DEF, there is a point Q such that QD = 9, QE = 12, and QF = 15. Prove that the area of triangle DEF to the nearest integer is 97. --/
theorem area_of_equilateral_triangle_DEF (DEF_equilateral : ∀ (D E F : Point), equilateral D E F)
  (Q_inside_DEF : ∀ (D E F Q : Point), inside Q (triangle D E F) → QD = 9 → QE = 12 → QF = 15) 
  (triangle_area : ∀ (D E F : Point), equilateral D E F → ℝ) :
  ∃ (D E F Q : Point), round (triangle_area D E F) = 97 := 
by
  sorry

end area_of_equilateral_triangle_DEF_l804_804193


namespace projection_correct_l804_804079

-- Defining the vectors a and b
def a : ℝ × ℝ × ℝ := (4, -1, 3)
def b : ℝ × ℝ × ℝ := (3, 2, -2)

-- Defining the dot product
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Defining the scalar multiplication
def scalar_mult (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

-- Defining the projection of vector a onto vector b
def proj (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let num := (dot_product a b) / (dot_product b b)
  scalar_mult num b

-- Stating the theorem to be proved
theorem projection_correct :
  proj a b = (12/17, 8/17, -8/17) := by
  sorry

end projection_correct_l804_804079


namespace total_votes_l804_804321

-- Definitions
variable (V : ℝ)
variable (candidate_votes : ℝ) := 0.35 * V
variable (rival_votes : ℝ) := candidate_votes + 2370

-- Theorem statement
theorem total_votes (h : candidate_votes + rival_votes = V) : V = 7900 := by
  -- Proof would go here
  sorry

end total_votes_l804_804321


namespace num_students_who_like_both_apple_pie_and_chocolate_cake_l804_804503

open Set

variable (S : Type) [Finite S] 
variable (students : Finset S)

variable (likes_apple_pie likes_chocolate_cake likes_pumpkin_pie likes_no_dessert : Finset S)

-- Conditions
axiom total_students : students.card = 50
axiom likes_apple_pie_card : likes_apple_pie.card = 22
axiom likes_chocolate_cake_card : likes_chocolate_cake.card = 20
axiom likes_pumpkin_pie_card : likes_pumpkin_pie.card = 17
axiom likes_no_dessert_card : likes_no_dessert.card = 15
axiom no_dessert : ∀ s, s ∈ likes_no_dessert → s ∉ likes_apple_pie ∧ s ∉ likes_chocolate_cake ∧ s ∉ likes_pumpkin_pie
axiom dessert_partition : ∀ s, s ∈ students → (s ∈ likes_apple_pie ∨ s ∈ likes_chocolate_cake ∨ s ∈ likes_pumpkin_pie ∨ s ∈ likes_no_dessert)

-- Define the proof problem
def students_who_like_both_apple_pie_and_chocolate_cake :=
  likes_apple_pie ∩ likes_chocolate_cake

theorem num_students_who_like_both_apple_pie_and_chocolate_cake : 
  (students_who_like_both_apple_pie_and_chocolate_cake likes_apple_pie likes_chocolate_cake).card = 7 :=
by
  sorry

end num_students_who_like_both_apple_pie_and_chocolate_cake_l804_804503


namespace complex_point_quadrant_l804_804117

def i : ℂ := complex.I

def z : ℂ := (1 + i) / real.sqrt 2

theorem complex_point_quadrant :
  let z_2015 := z ^ 2015 in
  z_2015.re > 0 ∧ z_2015.im < 0 := 
by 
  sorry

end complex_point_quadrant_l804_804117


namespace problem1_problem2_l804_804838

variable (α : ℝ)

def condition1 : Prop := sin α + cos α = 1 / 5
def condition2 : Prop := 0 < α ∧ α < π

theorem problem1 (h1 : condition1 α) (h2 : condition2 α) : sin α * cos α = -12 / 25 :=
by
  sorry

theorem problem2 (h1 : condition1 α) (h2 : condition2 α) : tan α = -4 / 3 :=
by
  sorry

end problem1_problem2_l804_804838


namespace total_phases_sixth_goal_scorer_l804_804749

theorem total_phases (afp bfp cgk : ℕ) (p : ℕ) 
  (h1 : afp = 12) 
  (h2 : bfp = 21) 
  (h3 : cgk = 8) 
  (h4 : p - afp + p - bfp + cgk = p) :
  p = 25 :=
by {
  rw [h1, h2, h3] at h4,
  linarith,
}

theorem sixth_goal_scorer (p : ℕ) (ak gk bk ck : ℕ)
  (h1 : p = 25) 
  (h2 : ak = p - 12) 
  (h3 : bk = p - 21)
  (h4 : ck = 8)
  (h5 : ak ∉ {6} → False) :
  (∃ f, f = "Amandine") :=
by {
  use "Amandine",
  exact h5,
}

end total_phases_sixth_goal_scorer_l804_804749


namespace find_angle_between_vectors_l804_804800

-- Defining the two vectors
def u : ℝ^3 := ⟨3, -1, 4⟩
def v : ℝ^3 := ⟨2, 6, -5⟩

-- Statement to prove the angle between the vectors
theorem find_angle_between_vectors : 
  let θ := real.arccos ((u ⬝ v) / (∥u∥ * ∥v∥)) in
  θ.toDegrees ≈ 118.94 :=
by sorry

end find_angle_between_vectors_l804_804800


namespace average_eq_one_half_l804_804328

variable (w x y : ℝ)

-- Conditions
variables (h1 : 2 / w + 2 / x = 2 / y)
variables (h2 : w * x = y)

theorem average_eq_one_half : (w + x) / 2 = 1 / 2 :=
by
  sorry

end average_eq_one_half_l804_804328


namespace school_payment_l804_804366

-- Define the number of rows and seats per row
def num_rows : Nat := 20
def seats_per_row : Nat := 15

-- Define the cost per seat
def cost_per_seat : Nat := 50

-- Define the discounts
def discount_10 : Rat := 0.10
def discount_20 : Rat := 0.15
def discount_30 : Rat := 0.20

-- Calculate the total number of seats
def total_seats : Nat := num_rows * seats_per_row

-- Calculate the total cost without discount
def total_cost : Nat := total_seats * cost_per_seat

-- Calculate the discount applicable for groups of 30 seats or more
def discount_amount : Nat := Nat.floor (discount_30 * total_cost)

-- Calculate the final cost after discount
def final_cost : Nat := total_cost - discount_amount

-- Theorem to prove
theorem school_payment : final_cost = 12000 := by
  rw [final_cost, total_cost, discount_amount, total_seats]
  have : total_seats = 300 := rfl
  have : total_cost = 15000 := by norm_num [total_seats, cost_per_seat]
  have : discount_amount = 3000 := by norm_num [discount_30, total_cost]
  have : final_cost = 12000 := by norm_num [total_cost, discount_amount]
  exact this

end school_payment_l804_804366


namespace gcd_divides_15n_and_75_l804_804324

theorem gcd_divides_15n_and_75 (n : ℤ) : 
  let k := Int.gcd (n + 5) (n^2 - 5*n + 25)
  in k ∣ 15*n ∧ k ∣ 75 :=
sorry

end gcd_divides_15n_and_75_l804_804324


namespace ce_plus_de_squared_l804_804542

noncomputable def circleGeom (r : ℝ) (BE : ℝ) (theta : ℝ) := sorry

theorem ce_plus_de_squared (r : ℝ) : circleGeom 10 6 (60 * (π / 180)) = 300 := sorry

end ce_plus_de_squared_l804_804542


namespace line_intersects_circle_l804_804662

theorem line_intersects_circle (k : ℝ) :
  let l := λ x y : ℝ, x - k * y - 1 = 0
  let C := λ x y : ℝ, x^2 + y^2 = 2
  let center := (0, 0)
  let radius := sqrt 2
  let d := abs 1 / sqrt (1 + k^2)
  d < sqrt 2 :=
by
  sorry

end line_intersects_circle_l804_804662


namespace inequality_abc_l804_804834

theorem inequality_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by {
  sorry
}

end inequality_abc_l804_804834


namespace min_room_side_length_for_table_l804_804360

theorem min_room_side_length_for_table (S : ℕ) :
  (∃ l w : ℕ, l = 9 ∧ w = 12 ∧ S ≥ int.sqrt (l^2 + w^2)) → S = 15 :=
by
  intros h
  sorry

end min_room_side_length_for_table_l804_804360


namespace conjugate_in_first_quadrant_l804_804474

open Complex

variable (z : ℂ)

noncomputable def matrix_determinant_eq_zero : Prop :=
  (z * (1 + I)) - (1 + 2 * I) * (1 - I) = 0

theorem conjugate_in_first_quadrant
  (h : matrix_determinant_eq_zero z) :
  0 < z.conj.re ∧ 0 < z.conj.im :=
sorry

end conjugate_in_first_quadrant_l804_804474


namespace largest_integer_l804_804696

def sum_digits (n : ℕ) : ℕ :=
n.digits.sum

theorem largest_integer (A : ℕ) (h: A = 19 * sum_digits A) : A ≤ 399 :=
sorry

end largest_integer_l804_804696


namespace part1_part2_l804_804521
-- Importing the entire Mathlib library for required definitions

-- Define the sequence a_n with the conditions given in the problem
def a : ℕ → ℚ
| 0       => 1
| (n + 1) => a n / (2 * a n + 1)

-- Prove the given claims
theorem part1 (n : ℕ) : a n = (1 : ℚ) / (2 * n + 1) :=
sorry

def b (n : ℕ) : ℚ := a n * a (n + 1)

-- The sum of the first n terms of the sequence b_n is denoted as T_n
def T : ℕ → ℚ
| 0       => 0
| (n + 1) => T n + b n

-- Prove the given sum
theorem part2 (n : ℕ) : T n = (n : ℚ) / (2 * n + 1) :=
sorry

end part1_part2_l804_804521


namespace first_girl_has_one_sibling_l804_804024

-- Define the known conditions
def mean_siblings : ℝ := 5.7
def known_results : List ℕ := [6, 10, 4, 3, 3, 11, 3, 10]

def total_girls : ℕ := known_results.length + 1

def total_siblings : ℕ := Real.toNat (mean_siblings * total_girls)

def sum_known_results : ℕ := known_results.sum

def first_girl_siblings : ℕ := total_siblings - sum_known_results

theorem first_girl_has_one_sibling : first_girl_siblings = 1 := by
  sorry

end first_girl_has_one_sibling_l804_804024


namespace neznaika_mistake_correct_numbers_l804_804610

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end neznaika_mistake_correct_numbers_l804_804610


namespace num_ways_select_two_numbers_l804_804149

theorem num_ways_select_two_numbers (S : set ℤ) (hS : S = {1, 2, ..., 20}) :
  (∃ (count : ℕ), count = 45 ∧
    (∀ (a b : ℤ), a ∈ S → b ∈ S → |a - b| > 10)) := sorry

end num_ways_select_two_numbers_l804_804149


namespace free_throw_percentage_l804_804344

theorem free_throw_percentage (p : ℚ) :
  (1 - p)^2 + 2 * p * (1 - p) = 16 / 25 → p = 3 / 5 :=
by
  sorry

end free_throw_percentage_l804_804344


namespace find_number_l804_804174

theorem find_number (n : ℝ) (h : (1 / 3) * n = 6) : n = 18 :=
sorry

end find_number_l804_804174


namespace count_multiples_12_9_l804_804153

theorem count_multiples_12_9 :
  ∃ n : ℕ, n = 8 ∧ (∀ x : ℕ, x % 36 = 0 ∧ 200 ≤ x ∧ x ≤ 500 ↔ ∃ y : ℕ, (x = 36 * y ∧ 200 ≤ 36 * y ∧ 36 * y ≤ 500)) :=
by
  sorry

end count_multiples_12_9_l804_804153


namespace exists_individual_with_few_friendly_pairs_l804_804912

theorem exists_individual_with_few_friendly_pairs (n q : ℕ) (friendly : ℕ) (hostile : ℕ) :
  -- Conditions
  q = friendly ∧
  (q : ℝ) = friendly ∧
  (n * (n - 1) / 2 - q = hostile) ∧
  (∀ (i j k : ℕ), {i, j, k}.size = 3 → 
                  (\forall a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c →  
                  (\{i, j, k}.filter $ \λ s, s != i ∧ s != j ∧ s != k).size <= 1)) →
  -- Conclusion
  ∃ i, ∃ enemies_of_i : ℕ, ∃ d_i : ℕ,
    d_i = (enemies_of_i - 1) ∧
    hostile = (n-1-d_i : ℝ) ∧
    d_i <= q * (1 - 4 * q / n ^ 2) :=
sorry

end exists_individual_with_few_friendly_pairs_l804_804912


namespace general_formula_sum_comparison_l804_804141

open BigOperators

-- Condition: Given sequence a_n satisfies a certain sum formula
def sequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → ∑ i in Finset.range n, 3 ^ i * a (i + 1) = (n + 1) / 3

-- Proving sequence satisfies given general formula
theorem general_formula (a : ℕ → ℚ) (h : sequence a) :
  ∀ n : ℕ, n > 0 → a n = if n = 1 then 2 / 3 else 1 / 3 ^ n :=
sorry

-- Definition of sequence b_n based on a_n
def b (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  1 / (3 ^ (n + 1) * (1 - a n) * (1 - a (n + 1)))

-- Sum of the first n terms of sequence b_n
def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, b a (i + 1)

-- Proving sum S_n is less than 7/16
theorem sum_comparison (a : ℕ → ℚ) (h : sequence a) :
  ∀ n : ℕ, n > 0 → S a n < 7 / 16 :=
sorry

end general_formula_sum_comparison_l804_804141


namespace train_on_time_speed_l804_804751

theorem train_on_time_speed :
  ∃ (v : ℝ), (70 / v) + 0.25 = 70 / 35 ∧ v = 40 :=
by
  existsi (40 : ℝ)
  split
  calc
    70 / 40 + 0.25 = 1.75 + 0.25 : by norm_num
                ... = 2           : by norm_num
  sorry

end train_on_time_speed_l804_804751


namespace smallest_sum_of_numbers_on_cube_faces_l804_804267

theorem smallest_sum_of_numbers_on_cube_faces
  (a b c d e f : ℕ)
  (adj1 : |a - b| > 1)
  (adj2 : |b - c| > 1)
  (adj3 : |c - a| > 1)
  (adj4 : |d - e| > 1)
  (adj5 : |e - f| > 1)
  (adj6 : |f - d| > 1)
  (adj7 : |a - d| > 1)
  (adj8 : |b - e| > 1)
  (adj9 : |c - f| > 1)
  (adj10 : |a - e| > 1)
  (adj11 : |b - f| > 1)
  (adj12 : |c - d| > 1) : 
  a + b + c + d + e + f = 18 :=
sorry

end smallest_sum_of_numbers_on_cube_faces_l804_804267


namespace trains_cross_time_correct_l804_804301

def length_train1 : Nat := 130
def length_train2 : Nat := 160
def speed_train1_kmh : Nat := 60
def speed_train2_kmh : Nat := 40

def relative_speed_kmh : Nat := speed_train1_kmh + speed_train2_kmh

def conversion_factor : ℚ := 1000 / 3600

def relative_speed_ms : ℚ := relative_speed_kmh * conversion_factor

def total_distance : Nat := length_train1 + length_train2

def cross_time_seconds : ℚ := total_distance / relative_speed_ms

theorem trains_cross_time_correct:
  cross_time_seconds ≈ 10.44 := 
sorry

end trains_cross_time_correct_l804_804301


namespace cricket_students_count_l804_804175

def num_students_who_like_cricket (B B_inter_C B_union_C : ℕ) : ℕ :=
  B_union_C - B + B_inter_C

theorem cricket_students_count
  (B : ℕ) (B_inter_C : ℕ) (B_union_C : ℕ) : num_students_who_like_cricket B B_inter_C B_union_C = 8 :=
by
  have B_value : B = 7 := by rfl
  have B_inter_C_value : B_inter_C = 5 := by rfl
  have B_union_C_value : B_union_C = 10 := by rfl
  rw [B_value, B_inter_C_value, B_union_C_value] -- simplify with the given values
  unfold num_students_who_like_cricket -- use the definition
  sorry

end cricket_students_count_l804_804175


namespace partition_phone_call_groups_l804_804119

theorem partition_phone_call_groups (n : ℕ) (h_n : n ≥ 6)
  (h1 : ∀ (s : Finset (Fin n)), s.card = 3 → ∃ a b, a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ (a.can_talk_to b))
  (h2 : ∀ a : Fin n, (∑ b in (Finset.filter (λ x, ¬a.can_talk_to x) (Finset.univ)), 1) ≥ (n - 1 - ⌊(n-2)/2⌋)) :
  ∃ (g1 g2 : Finset (Fin n)), g1 ∩ g2 = ∅ ∧ g1 ∪ g2 = Finset.univ ∧
  (∀ a b ∈ g1, a ≠ b → a.can_talk_to b) ∧ (∀ a b ∈ g2, a ≠ b → a.can_talk_to b) :=
begin
  sorry
end

end partition_phone_call_groups_l804_804119


namespace cot_45_correct_l804_804769

def tangent_45_eq_one : Prop := tan (real.pi / 4) = 1

def cotangent_45 : Prop := cot (real.pi / 4) = 1

theorem cot_45_correct : tangent_45_eq_one → cotangent_45 := 
by
  sorry

end cot_45_correct_l804_804769


namespace sum_positive_implies_at_least_one_positive_l804_804497

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l804_804497


namespace alien_saturday_exclamation_l804_804025

-- Define the initial exclamations
def monday_exclamation := "A!"
def tuesday_exclamation := "AU!"
def wednesday_exclamation := "AUUA!"
def thursday_exclamation := "AUUAUAAU!"

-- Define the function that generates the next exclamation based on the current one's pattern
def next_exclamation (excl : String) : String :=
  let half_len := excl.length / 2
  let first_half := excl.take half_len
  let second_half := excl.drop half_len
  let reflected_half := second_half.map (λ c, if c = 'A' then 'U' else if c = 'U' then 'A' else c)
  first_half ++ reflected_half

-- Define the exclamations for each day starting from Thursday
def friday_exclamation := next_exclamation thursday_exclamation
def saturday_exclamation := next_exclamation friday_exclamation

-- Define the expected solution for Saturday's exclamation
def expected_saturday_exclamation := "AUUAUAUAAUAUAUAAUAA!"

-- Prove that the calculated exclamation for Saturday matches the expected one
theorem alien_saturday_exclamation :
  saturday_exclamation = expected_saturday_exclamation :=
by sorry

end alien_saturday_exclamation_l804_804025


namespace sufficient_condition_for_parallel_l804_804207

-- Definitions of planes and lines
variables (α β : Type) [Plane α] [Plane β] (m n : Line)

-- Interpreting the conditions
-- Line m is parallel to plane α
def m_parallel_to_α (m : Line) (α : Plane) : Prop := sorry

-- Plane α is perpendicular to plane β
def α_perpendicular_to_β (α β : Plane) : Prop := sorry

-- Line m is perpendicular to plane β
def m_perpendicular_to_β (m : Line) (β : Plane) : Prop := sorry

-- α intersects β in line n
def α_intersects_β_in_n (α β : Plane) (n : Line) : Prop := sorry

-- Line m is parallel to line n
def m_parallel_to_n (m n : Line) : Prop := sorry

-- Plane α is parallel to plane β
def α_parallel_to_β (α β : Plane) : Prop := sorry

-- Line m is contained in plane β
def m_in_β (m : Line) (β : Plane) : Prop := sorry

-- Line n is parallel to plane α
def n_parallel_to_α (n : Line) (α : Plane) : Prop := sorry

theorem sufficient_condition_for_parallel (α β : Type) [Plane α] [Plane β] (m : Line) : 
    α_parallel_to_β α β ∧ m_in_β m β → m_parallel_to_α m α := sorry

end sufficient_condition_for_parallel_l804_804207


namespace ratio_daves_bench_to_weight_l804_804783

variables (wD bM bD bC : ℝ)

def daves_weight := wD = 175
def marks_bench_press := bM = 55
def marks_comparison_to_craig := bM = bC - 50
def craigs_comparison_to_dave := bC = 0.20 * bD

theorem ratio_daves_bench_to_weight
  (h1 : daves_weight wD)
  (h2 : marks_bench_press bM)
  (h3 : marks_comparison_to_craig bM bC)
  (h4 : craigs_comparison_to_dave bC bD) :
  (bD / wD) = 3 :=
by
  rw [daves_weight] at h1
  rw [marks_bench_press] at h2
  rw [marks_comparison_to_craig] at h3
  rw [craigs_comparison_to_dave] at h4
  -- Now we have:
  -- 1. wD = 175
  -- 2. bM = 55
  -- 3. bM = bC - 50
  -- 4. bC = 0.20 * bD
  -- We proceed to solve:
  sorry

end ratio_daves_bench_to_weight_l804_804783


namespace extremum_at_1_implies_a_eq_2_monotonically_increasing_implies_a_leq_2_zeros_of_g_based_on_a_l804_804136

section MathProblems

variable (a : ℝ)

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + a / x + real.log x

-- Define the first derivative of the function f(x)
def f' (x : ℝ) : ℝ := 1 - a / x^2 + 1 / x

-- Define the function g(x)
def g (x : ℝ) : ℝ := f' a x - x

-- The first problem
theorem extremum_at_1_implies_a_eq_2 : f' a 1 = 0 → a = 2 :=
sorry

-- The second problem
theorem monotonically_increasing_implies_a_leq_2 :
  (∀ x ∈ Ioo 1 2, f' a x ≥ 0) → a ≤ 2 :=
sorry

-- The third problem
theorem zeros_of_g_based_on_a : 
  ∀ a, (a > 1 → ∀ x, g a x ≠ 0) ∧ 
       (a = 1 → ∃ x, g a x = 0) ∧ 
       (a ≤ 0 → ∃ x, g a x = 0) ∧ 
       (0 < a ∧ a < 1 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0) :=
sorry

end MathProblems

end extremum_at_1_implies_a_eq_2_monotonically_increasing_implies_a_leq_2_zeros_of_g_based_on_a_l804_804136


namespace sum_of_real_solutions_eq_neg_32_div_7_l804_804430

theorem sum_of_real_solutions_eq_neg_32_div_7 :
  ∀ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x) →
  ∑ sol in { x | (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x) }, sol = -32 / 7 :=
by
  sorry

end sum_of_real_solutions_eq_neg_32_div_7_l804_804430


namespace product_of_tangents_l804_804536

def S := {p : ℤ × ℤ | (0 ≤ p.1 ∧ p.1 ≤ 4) ∧ (1 ≤ p.2 ∧ p.2 ≤ 5)}
def S' := {p : ℤ × ℤ | (0 ≤ p.1 ∧ p.1 ≤ 4) ∧ (0 ≤ p.2 ∧ p.2 ≤ 4)}

def isRightTriangle (A B C : ℤ × ℤ) : Prop :=
  (B.2 - A.2) * (C.2 - A.2) + (B.1 - A.1) * (C.1 - A.1) = 0

def f (t : (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ)) : ℚ :=
  sorry -- Tangent function definition here

def g (t : (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ)) : ℚ :=
  sorry -- Tangent function definition here

def T := {t : (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ) | t.1 ∈ S ∧ t.2 ∈ S ∧ t.3 ∈ S ∧ isRightTriangle t.1 t.2 t.3}
def T' := {t : (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ) | t.1 ∈ S' ∧ t.2 ∈ S' ∧ t.3 ∈ S' ∧ isRightTriangle t.1 t.2 t.3}
def T'' := T ∪ T'

theorem product_of_tangents :
  (∏ t in T'', (f t) * (g t)) = 1 :=
sorry

end product_of_tangents_l804_804536


namespace prism_faces_l804_804676

theorem prism_faces (E V F : ℕ) (n : ℕ) 
  (h1 : E + V = 40) 
  (h2 : E = 3 * F - 6) 
  (h3 : V - E + F = 2)
  (h4 : V = 2 * n)
  : F = 10 := 
by
  sorry

end prism_faces_l804_804676


namespace geometric_locus_intersection_points_locus_vertex_polygon_inscribed_polygon_l804_804323

-- Proof Problem 1
theorem geometric_locus_intersection_points (l1 l2 l3 l4 : Line) (A : Point) (B : Point)
  (M1 M2 : Point) (m : ℝ) :
  (M1 ∈ l1) → (M2 ∈ l3) → (A ∈ l1) → (B ∈ l3) → (AM1 / BM2 = m) →
  (parallel (line_through M1) l3) → (parallel (line_through M2) l4) →
  ∃ O : Point, (locus_intersection_points l1 l2 l3 l4 A B M1 M2 = line_through O) :=
sorry

-- Proof Problem 2
theorem locus_vertex_polygon (n : ℕ) (l1 l2 ... ln : Line) (A1 A2 ... An : Point)
  (A1_0 A2_0 ... An_0 : Point) :
  (parallel_sides_polygon A1 A2 ... An A1_0 A2_0 ... An_0) →
  (A1 ∈ l1) → (A2 ∈ l2) → ... → (An-1 ∈ ln-1) →
  ∃ l, (vertex_locus An = l) :=
sorry

-- Proof Problem 3
theorem inscribed_polygon (l1 l2 ... ln : Line) (A1 : Point) :
  ∃ (polygon : Polygon), (sides_parallel_to_given_lines polygon l1 l2 ... ln) ∧
  (vertices_on_lines polygon l1 l2 ... ln) :=
sorry

end geometric_locus_intersection_points_locus_vertex_polygon_inscribed_polygon_l804_804323


namespace reflected_midpoint_sum_l804_804978

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def reflect_y (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, P.2)

noncomputable def sum_coords (P : ℝ × ℝ) : ℝ :=
  P.1 + P.2

theorem reflected_midpoint_sum :
  let A := (3, 2)
  let B := (15, 18)
  let N := midpoint A B
  let A' := reflect_y A
  let B' := reflect_y B
  let N' := midpoint A' B'
  sum_coords N' = 1 :=
by
  let A := (3, 2)
  let B := (15, 18)
  let N := midpoint A B
  let A' := reflect_y A
  let B' := reflect_y B
  let N' := midpoint A' B'
  have hn'_coords : N' = (-9, 10) := sorry
  have sum := sum_coords N'
  show 1 = sum
  use hn'_coords
  use sum
  sorry

end reflected_midpoint_sum_l804_804978


namespace neznaika_mistake_correct_numbers_l804_804613

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end neznaika_mistake_correct_numbers_l804_804613


namespace percentage_germinated_is_29_l804_804437

variable (seeds_plot1 seeds_plot2 : ℕ)
variable (perc_germ_plot1 perc_germ_plot2 : ℝ)

def calculate_percentage_germinated (seeds_plot1 seeds_plot2 : ℕ) 
    (perc_germ_plot1 perc_germ_plot2 : ℝ) : ℝ :=
  let total_seeds := seeds_plot1 + seeds_plot2
  let germinated_seeds := (perc_germ_plot1 / 100) * seeds_plot1 + 
                          (perc_germ_plot2 / 100) * seeds_plot2
  (germinated_seeds / total_seeds) * 100

theorem percentage_germinated_is_29 :
  seeds_plot1 = 300 → seeds_plot2 = 200 → 
  perc_germ_plot1 = 25 → perc_germ_plot2 = 35 →
  calculate_percentage_germinated seeds_plot1 seeds_plot2 perc_germ_plot1 perc_germ_plot2 = 29 :=
by
  intros h1 h2 h3 h4
  simp only [calculate_percentage_germinated, h1, h2, h3, h4]
  sorry  -- proof skipped

end percentage_germinated_is_29_l804_804437


namespace overtime_pay_rate_l804_804507

noncomputable def regular_days_per_week := 5
noncomputable def regular_hours_per_day := 8
noncomputable def regular_pay_rate := 2.40
noncomputable def total_earnings := 432
noncomputable def total_hours := 175
noncomputable def weeks := 4

theorem overtime_pay_rate:
  let regular_hours_per_week := regular_days_per_week * regular_hours_per_day
  let regular_hours := regular_hours_per_week * weeks
  let regular_earnings := regular_hours * regular_pay_rate
  let overtime_hours := total_hours - regular_hours
  let overtime_earnings := total_earnings - regular_earnings
  overtime_hours * (overtime_earnings / overtime_hours) = 48 →
  (overtime_earnings / overtime_hours = 3.20)
:= by
  intros regular_hours_per_week regular_hours regular_earnings overtime_hours overtime_earnings h
  sorry

end overtime_pay_rate_l804_804507


namespace solve_eq_l804_804622

theorem solve_eq {x : ℝ} (h : x + 2 * Real.sqrt x - 8 = 0) : x = 4 :=
by
  sorry

end solve_eq_l804_804622


namespace min_room_side_for_table_rotation_l804_804342

-- Define table dimensions
def table_width : ℕ := 7
def table_length : ℕ := 11

-- Define diagonal of the table
def table_diagonal : ℝ := Real.sqrt (table_width^2 + table_length^2)

-- Define smallest integer side of the room
def smallest_integer_side (d : ℝ) : ℕ := Nat.ceil d

-- State the theorem
theorem min_room_side_for_table_rotation : smallest_integer_side table_diagonal = 14 :=
by
  -- Calculation and proof to be filled in
  sorry

end min_room_side_for_table_rotation_l804_804342


namespace NumberOfRootsForEquation_l804_804930

noncomputable def numRootsAbsEq : ℕ :=
  let f := (fun x : ℝ => abs (abs (abs (abs (x - 1) - 9) - 9) - 3))
  let roots : List ℝ := [27, -25, 11, -9, 9, -7]
  roots.length

theorem NumberOfRootsForEquation : numRootsAbsEq = 6 := by
  sorry

end NumberOfRootsForEquation_l804_804930


namespace no_such_x_exists_l804_804407

theorem no_such_x_exists :
  ¬ ∃ x : ℝ, (x + Real.sqrt 2).isRat ∧ (x^3 + Real.sqrt 2).isRat :=
by
  sorry

end no_such_x_exists_l804_804407


namespace graph_fixed_point_l804_804858

theorem graph_fixed_point {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1) : 
  ∃ A : ℝ × ℝ, A = (-2, -1) ∧ ∀ x : ℝ, y = a^(x + 2) - 2 ↔ (x, y) = A := 
by 
  sorry

end graph_fixed_point_l804_804858


namespace brenda_friends_l804_804386

def total_slices (pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ := pizzas * slices_per_pizza
def total_people (total_slices : ℕ) (slices_per_person : ℕ) : ℕ := total_slices / slices_per_person
def friends (total_people : ℕ) : ℕ := total_people - 1

theorem brenda_friends (pizzas : ℕ) (slices_per_pizza : ℕ) 
  (slices_per_person : ℕ) (pizzas_ordered : pizzas = 5) 
  (slices_per_pizza_value : slices_per_pizza = 4) 
  (slices_per_person_value : slices_per_person = 2) :
  friends (total_people (total_slices pizzas slices_per_pizza) slices_per_person) = 9 :=
by
  rw [pizzas_ordered, slices_per_pizza_value, slices_per_person_value]
  sorry

end brenda_friends_l804_804386


namespace tangent_distance_property_l804_804707

noncomputable def circle_tangent_distances (A B C : Point) (l : Line) (a : ℝ) (alpha : ℝ) 
(u : ℝ) (v : ℝ) (w : ℝ) : Prop :=
  tangent_to_circle A ∧ tangent_to_circle B ∧ 
  intersection_point A B C ∧ 
  tangent_line_not_passthrough A B l ∧ 
  dist(A, C) = a ∧ dist(B, C) = a ∧ 
  angle(A, C, B) = alpha ∧ 
  dist(A, l) = u ∧ dist(B, l) = v ∧ 
  dist(C, l) = w

theorem tangent_distance_property (A B C : Point) (l : Line) (a : ℝ) (alpha : ℝ) 
(u : ℝ) (v : ℝ) (w : ℝ) : 
  circle_tangent_distances A B C l a alpha u v w → 
  (u * v) / (w * w) = real.sin(alpha / 2) ^ 2 :=
by
  sorry

end tangent_distance_property_l804_804707


namespace simplify_complex_expression_l804_804985

theorem simplify_complex_expression :
  let i := complex.I in
  3 * (4 - 2 * i) + 2 * i * (3 - 2 * i) = 16 :=
by
  let i := complex.I
  calc
    3 * (4 - 2 * i) + 2 * i * (3 - 2 * i) = 3 * 4 - 3 * 2 * i + 2 * i * 3 - 2 * i * 2 * i : by ring
    ... = 12 - 6 * i + 6 * i - 4 * (i * i) : by ring
    ... = 12 - 6 * i + 6 * i - 4 * (-1) : by simp [complex.I_mul_I]
    ... = 12 - 6 * i + 6 * i + 4 : by simp
    ... = 12 + 4 : by simp
    ... = 16 : by simp

end simplify_complex_expression_l804_804985


namespace problem_l804_804396

def expr : ℤ := 7^2 - 4 * 5 + 2^2

theorem problem : expr = 33 := by
  sorry

end problem_l804_804396


namespace part1_part2_l804_804836

variables {A B C : ℝ}
variables (sin cos : ℝ → ℝ)
variables (a b : ℝ × ℝ)

-- Condition: A, B, and C are the three internal angles of triangle ABC, and A + B + C = π
axiom angle_sum : A + B + C = Real.pi

-- Vectors based on the given problem
def vector_a : ℝ × ℝ := (sin B + cos B, cos C)
def vector_b : ℝ × ℝ := (sin C, sin B - cos B)

-- Dot product of vectors a and b
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- 1. If dot product is 0, prove A = 3/4 π
theorem part1 (h : dot_product vector_a vector_b = 0) : A = 3 * Real.pi / 4 :=
by sorry

-- 2. If dot product is -1/5, prove tan(2A) = -24/7
theorem part2 (h : dot_product vector_a vector_b = -1/5) : Real.tan (2 * A) = -24/7 :=
by sorry

end part1_part2_l804_804836


namespace two_digit_number_possible_options_l804_804599

theorem two_digit_number_possible_options
  (N : ℕ)
  (h1 : 10 ≤ N ∧ N < 100)
  (h2 : (N % 3 = 0 ∨ N % 3 ≠ 0) ∧
        (N % 4 = 0 ∨ N % 4 ≠ 0) ∧
        (N % 5 = 0 ∨ N % 5 ≠ 0) ∧
        (N % 9 = 0 ∨ N % 9 ≠ 0) ∧
        (N % 10 = 0 ∨ N % 10 ≠ 0) ∧
        (N % 15 = 0 ∨ N % 15 ≠ 0) ∧
        (N % 18 = 0 ∨ N % 18 ≠ 0) ∧
        (N % 30 = 0 ∨ N % 30 ≠ 0)) :
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end two_digit_number_possible_options_l804_804599


namespace total_possible_values_of_m_l804_804451

-- Definition of the sequence {a_n}
def sequence (m : ℕ) : ℕ → ℕ
| 1     := m
| (n+1) := if sequence n % 2 = 0 then sequence n / 2 else 3 * sequence n + 1

-- The proof problem
theorem total_possible_values_of_m (m : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ (n : ℕ), a n = sequence m n) 
  (h2 : a 6 = 1) 
: ∃ (s : Finset ℕ), (∀ m, m ∈ s ↔ (∃ n, a n = 1)) ∧ s.card = 3 :=
sorry

end total_possible_values_of_m_l804_804451


namespace selene_sandwiches_l804_804011

-- Define the context and conditions in Lean
variables (S : ℕ) (sandwich_cost hamburger_cost hotdog_cost juice_cost : ℕ)
  (selene_cost tanya_cost total_cost : ℕ)

-- Each item prices
axiom sandwich_price : sandwich_cost = 2
axiom hamburger_price : hamburger_cost = 2
axiom hotdog_price : hotdog_cost = 1
axiom juice_price : juice_cost = 2

-- Purchases
axiom selene_purchase : selene_cost = sandwich_cost * S + juice_cost
axiom tanya_purchase : tanya_cost = hamburger_cost * 2 + juice_cost * 2

-- Total spending
axiom total_spending : selene_cost + tanya_cost = 16

-- Goal: Prove that Selene bought 3 sandwiches
theorem selene_sandwiches : S = 3 :=
by {
  sorry
}

end selene_sandwiches_l804_804011


namespace time_to_write_all_rearrangements_l804_804253

-- Define the problem conditions
def sophie_name_length := 6
def rearrangements_per_minute := 18

-- Define the factorial function for calculating permutations
noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the total number of rearrangements of Sophie's name
noncomputable def total_rearrangements := factorial sophie_name_length

-- Define the time in minutes to write all rearrangements
noncomputable def time_in_minutes := total_rearrangements / rearrangements_per_minute

-- Convert the time to hours
noncomputable def minutes_to_hours (minutes : ℕ) : ℚ := minutes / 60

-- Prove the time in hours to write all the rearrangements
theorem time_to_write_all_rearrangements : minutes_to_hours time_in_minutes = (2 : ℚ) / 3 := 
  sorry

end time_to_write_all_rearrangements_l804_804253


namespace area_of_triangle_XPQ_l804_804192
open Real

/-- Given a triangle XYZ with area 15 square units and points P, Q, R on sides XY, YZ, and ZX respectively,
where XP = 3, PY = 6, and triangles XPQ and quadrilateral PYRQ have equal areas, 
prove that the area of triangle XPQ is 5/3 square units. -/
theorem area_of_triangle_XPQ 
  (Area_XYZ : ℝ) (h1 : Area_XYZ = 15)
  (XP PY : ℝ) (h2 : XP = 3) (h3 : PY = 6)
  (h4 : ∃ (Area_XPQ : ℝ) (Area_PYRQ : ℝ), Area_XPQ = Area_PYRQ) :
  ∃ (Area_XPQ : ℝ), Area_XPQ = 5/3 :=
sorry

end area_of_triangle_XPQ_l804_804192


namespace hotdog_eating_ratio_l804_804909

variable (rate_first rate_second rate_third total_hotdogs time_minutes : ℕ)
variable (rate_ratio : ℕ)

def rate_first_eq : rate_first = 10 := by sorry
def rate_second_eq : rate_second = 3 * rate_first := by sorry
def total_hotdogs_eq : total_hotdogs = 300 := by sorry
def time_minutes_eq : time_minutes = 5 := by sorry
def rate_third_eq : rate_third = total_hotdogs / time_minutes := by sorry

theorem hotdog_eating_ratio :
  rate_ratio = rate_third / rate_second :=
  by sorry

end hotdog_eating_ratio_l804_804909


namespace rounding_bound_l804_804319

theorem rounding_bound (k : ℕ) (n : ℕ) (a : ℕ) (m : ℕ) 
  (h₁ : k = 10 * 106)
  (h₂ : n = Nat.digits 10 k . length) 
  (h₃ : k = a * 10^(n - 1) + m) 
  (h₄ : a ≥ 1) 
  (h₅ : m < 10^(n - 1))
  (h_bar_k :  ∀ m (h : m < n - 1),
      let r := (λ k : ℕ, round (10^m) k)
      in (r^[m]) k != r^[m + 1] k):
  ∃ (bar_k : ℕ), bar_k < (18 / 13) * k := by 
  sorry

end rounding_bound_l804_804319


namespace three_friends_expenses_l804_804678

theorem three_friends_expenses :
  let ticket_cost := 7
  let number_of_tickets := 3
  let popcorn_cost := 1.5
  let number_of_popcorn := 2
  let milk_tea_cost := 3
  let number_of_milk_tea := 3
  let total_expenses := (ticket_cost * number_of_tickets) + (popcorn_cost * number_of_popcorn) + (milk_tea_cost * number_of_milk_tea)
  let amount_per_friend := total_expenses / 3
  amount_per_friend = 11 := 
by
  sorry

end three_friends_expenses_l804_804678


namespace total_percent_decrease_baseball_card_l804_804343

theorem total_percent_decrease_baseball_card
  (original_value : ℝ)
  (first_year_decrease : ℝ := 0.20)
  (second_year_decrease : ℝ := 0.30)
  (value_after_first_year : ℝ := original_value * (1 - first_year_decrease))
  (final_value : ℝ := value_after_first_year * (1 - second_year_decrease))
  (total_percent_decrease : ℝ := ((original_value - final_value) / original_value) * 100) :
  total_percent_decrease = 44 :=
by 
  sorry

end total_percent_decrease_baseball_card_l804_804343


namespace find_a_three_balls_sum_five_probability_l804_804675

   -- Condition definitions
   def num_red_balls (a : ℕ) : ℕ := a + 1
   def num_yellow_balls (a : ℕ) : ℕ := a
   def num_blue_balls : ℕ := 1
   def total_balls (a : ℕ) : ℕ := num_red_balls a + num_yellow_balls a + num_blue_balls

   def probability_red_ball (a : ℕ) : ℚ := ↑(num_red_balls a) / ↑(total_balls a)
   def probability_yellow_ball (a : ℕ) : ℚ := ↑(num_yellow_balls a) / ↑(total_balls a)
   def probability_blue_ball (a : ℕ) : ℚ := ↑(num_blue_balls) / ↑(total_balls a)

   def expected_value_ball (a : ℕ) : ℚ :=
     1 * probability_red_ball a + 2 * probability_yellow_ball a + 3 * probability_blue_ball a

   -- Part 1: Prove the expected value condition to find a
   theorem find_a (a : ℕ) (h : expected_value_ball a = 5 / 3) : a = 2 :=
   sorry

   -- Part 2: Prove the probability of sum of scores being 5 is 3/10
   def probability_three_balls_sum_five (a : ℕ) : ℚ :=
     let total := total_balls a in
     if a = 2 then (3 / ↑(total.choose 3)) + (3 / ↑(total.choose 3)) else 0

   theorem three_balls_sum_five_probability (a : ℕ) (h : a = 2) : probability_three_balls_sum_five a = 3 / 10 :=
   sorry
   
end find_a_three_balls_sum_five_probability_l804_804675


namespace cyclical_winning_sets_l804_804012

theorem cyclical_winning_sets (teams : Finset ℕ) (h_card : teams.card = 21)
  (h_wins_losses : ∀ t ∈ teams, ∃ wins losses : Finset ℕ, wins.card = 10 ∧ losses.card = 10 ∧ wins ∩ losses = ∅ ∧ wins ∪ losses = teams.erase t) :
  ∃ (k : ℕ), k = 385 ∧ ∀ {T : Finset ℕ}, T.card = 3 → (∃ a b c : ℕ, a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a, b) ∈ wins_of t ∧ (b, c) ∈ wins_of t ∧ (c, a) ∈ wins_of t) :=
by
  sorry

end cyclical_winning_sets_l804_804012


namespace sin_alpha_is_one_third_sin_2alpha_plus_β_is_correct_l804_804461

-- Define the conditions
variables {α β : ℝ}
axiom α_range : 0 < α ∧ α < π / 2
axiom β_range : π / 2 < β ∧ β < π
axiom cos_β : cos β = -1 / 3
axiom sin_α_plus_β : sin (α + β) = 7 / 9

-- Prove that sin α = 1 / 3 given the conditions
theorem sin_alpha_is_one_third (α : ℝ) (β : ℝ) 
  (α_range : 0 < α ∧ α < π / 2) 
  (β_range : π / 2 < β ∧ β < π) 
  (cos_β : cos β = -1 / 3) 
  (sin_α_plus_β : sin (α + β) = 7 / 9) : 
  sin α = 1 / 3 :=
sorry

-- Define additional condition based on result of first part
axiom sin_α : sin α = 1 / 3

-- Prove that sin (2α + β) = 10 * sqrt 2 / 27 given the conditions
theorem sin_2alpha_plus_β_is_correct (α : ℝ) (β : ℝ) 
  (α_range : 0 < α ∧ α < π / 2) 
  (β_range : π / 2 < β ∧ β < π) 
  (cos_β : cos β = -1 / 3) 
  (sin_α_plus_β : sin (α + β) = 7 / 9) 
  (sin_α : sin α = 1 / 3) : 
  sin (2 * α + β) = 10 * real.sqrt 2 / 27 :=
sorry

end sin_alpha_is_one_third_sin_2alpha_plus_β_is_correct_l804_804461


namespace crease_points_coverage_l804_804973

variable {O A : Point}
variable {R a : ℝ}

axiom distance_OA_eq_a (O A : Point) (R a : ℝ) (O A := distance O A) : O A = a
axiom R_gt_a (R a : ℝ) : a < R
axiom circle_O (O : Point) (R : ℝ) : Circle O R

theorem crease_points_coverage (O A : Point) (R a : ℝ)
  (h1 : O A = a)
  (h2 : a < R)
  (h3 : ∀ A', is_point_on_circle A' O R → ∃ l, is_fold_line l A A' O)
  : ∀ P, is_point_outside_circle P O R ∨ is_point_on_circle P O R := sorry

end crease_points_coverage_l804_804973


namespace cone_slant_height_l804_804667

noncomputable def pi := Real.pi

theorem cone_slant_height :
  ∃ (l : ℝ), l = 10 ∧
  let CSA := 157.07963267948966 in
  let r := 5 in
  CSA = pi * r * l :=
by sorry

end cone_slant_height_l804_804667


namespace total_rainbow_nerds_is_36_l804_804349

def purple_candies : ℕ := 10
def yellow_candies : ℕ := purple_candies + 4
def green_candies : ℕ := yellow_candies - 2
def total_candies : ℕ := purple_candies + yellow_candies + green_candies

theorem total_rainbow_nerds_is_36 : total_candies = 36 := by
  sorry

end total_rainbow_nerds_is_36_l804_804349


namespace floor_eq_48_iff_l804_804416

-- Define the real number set I to be [8, 49/6)
def I : Set ℝ := { x | 8 ≤ x ∧ x < 49/6 }

-- The main statement to be proven
theorem floor_eq_48_iff (x : ℝ) : (Int.floor (x * Int.floor x) = 48) ↔ x ∈ I := 
by
  sorry

end floor_eq_48_iff_l804_804416


namespace greatest_matching_pairs_after_losing_9_shoes_l804_804966

/-- 
Marcella has 27 pairs of shoes in total. The configuration is comprised of:
1. 12 pairs of sneakers: 5 red, 4 blue, 3 green.
2. 10 pairs of sandals: 6 with straps, 4 slip-on.
3. 5 pairs of boots, all black.
Additionally, the shoes come in sizes: 
- Size 6: 7 pairs
- Size 7: 8 pairs
- Size 8: 12 pairs

To form a matching pair, shoes must be the same color or style and also the same size.
Prove that after losing 9 individual shoes randomly, Marcella can still have up to 18 complete pairs. 
-/
theorem greatest_matching_pairs_after_losing_9_shoes :
  (∃ s : Set (Set ℕ), s.card = 18 ∧ ∀ shoe ∈ s, (shoe ∈ {size_6_shoe, size_7_shoe, size_8_shoe} ) ∧ (shoe ∈ {sneaker_shoe, sandal_shoe, boot_shoe}) ) :=
sorry

end greatest_matching_pairs_after_losing_9_shoes_l804_804966


namespace shopkeepers_total_profit_percentage_l804_804747

noncomputable def calculateProfitPercentage : ℝ :=
  let oranges := 1000
  let bananas := 800
  let apples := 750
  let rotten_oranges_percentage := 0.12
  let rotten_bananas_percentage := 0.05
  let rotten_apples_percentage := 0.10
  let profit_oranges_percentage := 0.20
  let profit_bananas_percentage := 0.25
  let profit_apples_percentage := 0.15
  let cost_per_orange := 2.5
  let cost_per_banana := 1.5
  let cost_per_apple := 2.0

  let rotten_oranges := rotten_oranges_percentage * oranges
  let rotten_bananas := rotten_bananas_percentage * bananas
  let rotten_apples := rotten_apples_percentage * apples

  let good_oranges := oranges - rotten_oranges
  let good_bananas := bananas - rotten_bananas
  let good_apples := apples - rotten_apples

  let cost_oranges := cost_per_orange * oranges
  let cost_bananas := cost_per_banana * bananas
  let cost_apples := cost_per_apple * apples

  let total_cost := cost_oranges + cost_bananas + cost_apples

  let selling_price_oranges := cost_per_orange * (1 + profit_oranges_percentage) * good_oranges
  let selling_price_bananas := cost_per_banana * (1 + profit_bananas_percentage) * good_bananas
  let selling_price_apples := cost_per_apple * (1 + profit_apples_percentage) * good_apples

  let total_selling_price := selling_price_oranges + selling_price_bananas + selling_price_apples

  let total_profit := total_selling_price - total_cost

  (total_profit / total_cost) * 100

theorem shopkeepers_total_profit_percentage :
  calculateProfitPercentage = 8.03 := sorry

end shopkeepers_total_profit_percentage_l804_804747


namespace consecutive_groups_divisible_by_11_l804_804516

def sequence : List ℕ := [1, 4, 8, 10, 16, 19, 21, 25, 30, 43]

def cumulative_sum (seq : List ℕ) : List ℕ :=
  seq.scanl (+) 0

def mod_11_equiv_pairs (lst : List ℕ) : List (ℕ × ℕ) :=
  List.filter (λ (pair : ℕ × ℕ), (lst.nth pair.fst).getD 0 % 11 = (lst.nth pair.snd).getD 0 % 11)
    ((List.range lst.length).product (List.range lst.length))

theorem consecutive_groups_divisible_by_11 :
  let S := cumulative_sum sequence
  in (mod_11_equiv_pairs S).length = 7 :=
sorry

end consecutive_groups_divisible_by_11_l804_804516


namespace bedroom_curtain_width_l804_804381

theorem bedroom_curtain_width
  (initial_fabric_area : ℕ)
  (living_room_curtain_area : ℕ)
  (fabric_left : ℕ)
  (bedroom_curtain_height : ℕ)
  (bedroom_curtain_area : ℕ)
  (bedroom_curtain_width : ℕ) :
  initial_fabric_area = 16 * 12 →
  living_room_curtain_area = 4 * 6 →
  fabric_left = 160 →
  bedroom_curtain_height = 4 →
  bedroom_curtain_area = 168 - 160 →
  bedroom_curtain_area = bedroom_curtain_width * bedroom_curtain_height →
  bedroom_curtain_width = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Skipping the proof
  sorry

end bedroom_curtain_width_l804_804381


namespace at_most_70_percent_acute_l804_804100

theorem at_most_70_percent_acute (P : Finset (EuclideanSpace ℝ (Fin 2))) (hP : P.card = 100)
  (triangle : Finset (Finset P) → Triangle ℝ) :
  ∃ non_acute_triangles : Finset (Triangle ℝ), 
  (∃ card_non_acute_triangles,
    card_non_acute_triangles = non_acute_triangles.card ∧
    card_non_acute_triangles ≥ 0.3 * (Finset.powersetLen 3 P).card) :=
sorry

end at_most_70_percent_acute_l804_804100


namespace find_a_l804_804928

noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ

-- Define the given conditions
def b : ℝ := 4 * Real.sqrt 6
def B : ℝ := Real.pi / 3  -- 60 degrees in radians
def A : ℝ := Real.pi / 4  -- 45 degrees in radians

-- Define the law of sines equation
lemma law_of_sines (a : ℝ) : 
    a / sin A = b / sin B :=
begin
    sorry
end

-- Prove that a is equal to 8
theorem find_a (a : ℝ) (h : a / sin A = b / sin B) : a = 8 :=
begin
    sorry
end

end find_a_l804_804928


namespace find_omega_l804_804898

theorem find_omega
  (A ω : ℝ) (hω_pos : ω > 0)
  (h_area : ∃ x1 x2 x3, x1 < x2 ∧ x2 < x3 ∧ y = A * sin (ω * x1) 
            ∧ y = A * sin (ω * x2) ∧ y = A * sin (ω * x3) 
            ∧ is_right_triangle (y, x1, x2, x3) 
            ∧ (0.5 * 4 * A * 2 * A = 1)) : 
  ω = real.pi := sorry

end find_omega_l804_804898


namespace population_end_second_year_l804_804512

-- Define the conditions as variables and functions in Lean
def initial_population : ℕ := 10000
def increase_rate : ℝ := 0.2
def decrease_rate : ℝ := 0.2

-- Define the population after the first year
def population_after_first_year (initial : ℕ) (rate : ℝ) : ℕ :=
  initial + (initial * rate).to_nat

-- Define the population after the second year
def population_after_second_year (initial : ℕ) (increase_rate : ℝ) (decrease_rate : ℝ) : ℕ :=
  let after_first := population_after_first_year initial increase_rate
  in after_first - (after_first * decrease_rate).to_nat

-- The theorem statement to be proved: The population at the end of the 2nd year is 9600
theorem population_end_second_year :
  population_after_second_year initial_population increase_rate decrease_rate = 9600 := by
  sorry

end population_end_second_year_l804_804512


namespace carnival_loss_count_l804_804765

theorem carnival_loss_count :
  let wins1 := 28
  let wins2 := 36
  let wins3 := 15
  let ratio1 := (4, 1)
  let ratio2 := (3, 2)
  let ratio3 := (1, 3)
  let losses1 := wins1 / ratio1.1 * ratio1.2
  let losses2 := wins2 / ratio2.1 * ratio2.2
  let losses3 := wins3 * ratio3.2
  losses1 + losses2 + losses3 = 76 :=
by
  -- Condition setup
  let wins1 := 28
  let wins2 := 36
  let wins3 := 15
  let ratio1 := (4, 1)
  let ratio2 := (3, 2)
  let ratio3 := (1, 3)
  -- Calculate losses
  let losses1 := wins1 / ratio1.1 * ratio1.2
  let losses2 := wins2 / ratio2.1 * ratio2.2
  let losses3 := wins3 * ratio3.2
  -- Expected result
  have : losses1 = 7 := by simp [losses1, wins1, ratio1]
  have : losses2 = 24 := by simp [losses2, wins2, ratio2]
  have : losses3 = 45 := by simp [losses3, wins3, ratio3]
  show losses1 + losses2 + losses3 = 76, by simp [losses1, losses2, losses3]

end carnival_loss_count_l804_804765


namespace monotonicity_intervals_range_of_a_for_zeros_h_gt_2_for_a_eq_1_l804_804139

namespace Problem

-- Define the functions f and g given a constant a.
def f (a : ℝ) (x : ℝ) := Real.log x - a * x
def g (x : ℝ) := Real.exp x - x

-- Statement (I): Monotonicity intervals for f(x)
theorem monotonicity_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, 0 < f (𝐬𝐚) x) ∧ (a > 0 → (∀ x ∈ Set.Ioo 0 (1 / a), 0 < f (𝐬𝐚) x ∧ ∀ x > 1 / a, f (𝐬𝐚) x < 0)) := by
  sorry

-- Statement (II): Range of a for two distinct zeros of f(x)
theorem range_of_a_for_zeros : (1 > 0 ∧ ∀ a ∈ Set.Ioo 0 (1 / Real.exp 1), ∃ x1 x2 ∈ Set.Ioi 0, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) := by
  sorry

-- Statement (III): Prove h(x) > 2 for a = 1
theorem h_gt_2_for_a_eq_1 : let h := λ x => g x - f 1 x in ∀ x > 0, h x > 2 := by
  sorry

end Problem

end monotonicity_intervals_range_of_a_for_zeros_h_gt_2_for_a_eq_1_l804_804139


namespace possible_numbers_l804_804588

theorem possible_numbers (N : ℕ) 
    (h1 : 10 ≤ N) (h2 : N ≤ 99)
    (h3 : (N % 3 = 0) ∧ (N % 4 = 0) ∧ (N % 5 = 0) ∧ (N % 9 = 0) ∧ (N % 10 = 0) ∧ 
          (N % 15 = 0) ∧ (N % 18 = 0) ∧ (N % 30 = 0) ∨ 
          (N % 3 ≠ 0) + (N % 4 ≠ 0) + (N % 5 ≠ 0) + (N % 9 ≠ 0) + 
          (N % 10 ≠ 0) + (N % 15 ≠ 0) + (N % 18 ≠ 0) + (N % 30 ≠ 0) = 4) :
   N = 36 ∨ N = 45 ∨ N = 72 :=
by {
  sorry
}

end possible_numbers_l804_804588


namespace calculate_jessie_points_l804_804996

theorem calculate_jessie_points (total_points : ℕ) (some_players_points : ℕ) (players : ℕ) :
  total_points = 311 →
  some_players_points = 188 →
  players = 3 →
  (total_points - some_players_points) / players = 41 :=
by
  intros
  sorry

end calculate_jessie_points_l804_804996


namespace pyramid_volume_l804_804006

theorem pyramid_volume (R r : ℝ) :
  ∃ V : ℝ, 
  let H := R + sqrt (R^2 - r^2),
      S := 2 * r^2
  in V = (2 * r^2 * H) / 3 :=
begin
  sorry
end

end pyramid_volume_l804_804006


namespace area_of_PINE_l804_804438

def PI := 6
def IN := 15
def NE := 6
def EP := 25
def sum_angles := 60 

theorem area_of_PINE : 
  (∃ (area : ℝ), area = (100 * Real.sqrt 3) / 3) := 
sorry

end area_of_PINE_l804_804438


namespace number_of_valid_integers_l804_804874

def count_valid_numbers : Nat :=
  let one_digit_count : Nat := 6
  let two_digit_count : Nat := 6 * 6
  let three_digit_count : Nat := 6 * 6 * 6
  one_digit_count + two_digit_count + three_digit_count

theorem number_of_valid_integers :
  count_valid_numbers = 258 :=
sorry

end number_of_valid_integers_l804_804874


namespace ratio_large_to_small_l804_804739

-- Definitions of the conditions
def total_fries_sold : ℕ := 24
def small_fries_sold : ℕ := 4
def large_fries_sold : ℕ := total_fries_sold - small_fries_sold

-- The proof goal
theorem ratio_large_to_small : large_fries_sold / small_fries_sold = 5 :=
by
  -- Mathematical steps would go here, but we skip with sorry
  sorry

end ratio_large_to_small_l804_804739


namespace exists_fib_with_last_four_digits_zero_l804_804476

noncomputable def fib_sequence (n : ℕ) : ℕ :=
  if n = 1 then 0
  else if n = 2 then 1
  else fib_sequence (n - 1) + fib_sequence (n - 2)

theorem exists_fib_with_last_four_digits_zero :
  ∃ n < 100000001, fib_sequence n % 10000 = 0 :=
by
  sorry

end exists_fib_with_last_four_digits_zero_l804_804476


namespace pentagon_perimeter_l804_804185

-- Define the side length and number of sides for a regular pentagon
def side_length : ℝ := 5
def num_sides : ℕ := 5

-- Define the perimeter calculation as a constant
def perimeter (side_length : ℝ) (num_sides : ℕ) : ℝ := side_length * num_sides

theorem pentagon_perimeter : perimeter side_length num_sides = 25 := by
  sorry

end pentagon_perimeter_l804_804185


namespace total_rainbow_nerds_is_36_l804_804348

def purple_candies : ℕ := 10
def yellow_candies : ℕ := purple_candies + 4
def green_candies : ℕ := yellow_candies - 2
def total_candies : ℕ := purple_candies + yellow_candies + green_candies

theorem total_rainbow_nerds_is_36 : total_candies = 36 := by
  sorry

end total_rainbow_nerds_is_36_l804_804348


namespace range_of_t_l804_804138

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 3

noncomputable def g (a : ℝ) : ℝ := 2 * a^3 - 3 * a^2

theorem range_of_t (t : ℝ) :
  (∃ (a : ℝ) (h : a ≠ 0) (h : a ≠ 1), g(a) = -t - 9) ↔ (-9 < t ∧ t < 8) := sorry

end range_of_t_l804_804138


namespace coeff_x5_l804_804307

-- Define the two polynomials
def poly1 : Polynomial ℝ := Polynomial.X^5 - 4 * Polynomial.X^4 + 6 * Polynomial.X^3 - 5 * Polynomial.X^2 + 2 * Polynomial.X + 1
def poly2 : Polynomial ℝ := 3 * Polynomial.X^4 - 2 * Polynomial.X^3 + Polynomial.X^2 + 4 * Polynomial.X - 8

-- The theorem to prove the coefficient of x^5
theorem coeff_x5 : (poly1 * poly2).coeff 5 = -2 := 
  sorry

end coeff_x5_l804_804307


namespace plane_through_three_points_l804_804804

noncomputable def point (x y z : ℤ) := (x, y, z)

def plane_eq (A B C D : ℤ) (x y z : ℤ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_through_three_points :
  ∃ (A B C D : ℤ), 
    A > 0 ∧ 
    Int.gcd A (Int.gcd B (Int.gcd C D)) = 1 ∧ 
    plane_eq A B C D 2 (-1) 3 ∧ 
    plane_eq A B C D 4 (-1) 5 ∧ 
    plane_eq A B C D 1 0 2 :=
  ∃ A B C D, A = -1 ∧ B = 0 ∧ C = 1 ∧ D = -1 ∧
  sorry

end plane_through_three_points_l804_804804


namespace triangle_ce_minus_bf_equiv_half_ac_minus_ab_l804_804200

noncomputable def triangle_acute (A B C : Point) : Prop :=
  ∃ (α β γ : ℝ), α + β + γ = 180 ∧ α < 90 ∧ β < 90 ∧ γ < 90 ∧ A.angle = α ∧ B.angle = β ∧ C.angle = γ

theorem triangle_ce_minus_bf_equiv_half_ac_minus_ab
  (A B C E F : Point)
  (h_triangle_acute : triangle_acute A B C) 
  (h_angle_A : A.angle = 60) 
  (h_E_altitude_from_B : is_altitude_from B E)
  (h_F_altitude_from_C : is_altitude_from C F) :
  dist C E - dist B F = 3 / 2 * (dist A C - dist A B) := 
  sorry

end triangle_ce_minus_bf_equiv_half_ac_minus_ab_l804_804200


namespace fishing_math_problem_l804_804633

-- Define the number of fish each mathematician catches as a list of integers
variables (a : List ℕ) -- a is the list [a1, a2, a3, a4, a5, a6]

-- Given conditions:
-- 1. Six mathematicians caught 100 fish total.
-- 2. Each caught a different number of fish.
-- 3. Any one can distribute their fish so that the remaining five have equal numbers.

theorem fishing_math_problem 
  (h_length : a.length = 6) -- There are six mathematicians
  (h_sum : a.sum = 100) -- They caught 100 fish in total
  (h_distinct : a.nodup) -- Each caught a different number
  (h_divisible : ∀ i ∈ a, (100 - i) % 5 = 0) -- Any one can distribute their fish so the remaining have equal numbers
  : ∃ i ∈ a, i = 20 ∧ ∀ j ∈ a.erase i, j < 20 := -- Prove there is one with exactly 20 fish and the rest with less than 20 fish
sorry

end fishing_math_problem_l804_804633


namespace reggie_father_money_l804_804624

theorem reggie_father_money :
  let books := 5
  let cost_per_book := 2
  let amount_left := 38
  books * cost_per_book + amount_left = 48 :=
by
  sorry

end reggie_father_money_l804_804624


namespace triangle_angle_NMC_l804_804924

theorem triangle_angle_NMC (A B C M N : Type) [angle_space A B C] [point_space M A B] [point_space N A C]
  (angle_ABC: ∠ABC = 100)
  (angle_ACB: ∠ACB = 65)
  (angle_MCB: ∠MCB = 55)
  (angle_NBC: ∠NBC = 80) :
  ∠NMC = 80 := 
sorry

end triangle_angle_NMC_l804_804924


namespace minimum_value_x_2y_l804_804455

theorem minimum_value_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = x * y) : x + 2 * y = 8 :=
sorry

end minimum_value_x_2y_l804_804455


namespace positive_solution_in_interval_l804_804570

def quadratic (x : ℝ) := x^2 + 3 * x - 5

theorem positive_solution_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ quadratic x = 0 :=
sorry

end positive_solution_in_interval_l804_804570


namespace distance_center_to_line_l804_804162

noncomputable def distance_from_center_to_line
    (a : ℝ) (h_circle: (a > 0) ∧ (∀ x y : ℝ, (x - a) ^ 2 + (y - a) ^ 2 = a ^ 2) ∧ (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2)
    (line : ℝ × ℝ → ℝ) : ℝ :=
    let center := if (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2 then (a, a) else (? , ?);
    let numerator := abs (2 * center.fst - center.snd - 3) in
    let denominator := sqrt (2 ^ 2 + 1 ^ 2) in
    numerator / denominator

theorem distance_center_to_line (a : ℝ)
    (h_circle: (a > 0) ∧ (∀ x y : ℝ, (x - a) ^ 2 + (y - a) ^ 2 = a ^ 2) ∧ (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2):
    distance_from_center_to_line a h_circle (λ p, 2 * p.1 - p.2 - 3) = (2 * Real.sqrt 5) / 5 := 
    sorry

end distance_center_to_line_l804_804162


namespace leak_empties_tank_in_4_hours_l804_804711

theorem leak_empties_tank_in_4_hours
  (A_fills_in : ℝ)
  (A_with_leak_fills_in : ℝ) : 
  (∀ (L : ℝ), A_fills_in = 2 ∧ A_with_leak_fills_in = 4 → L = (1 / 4) → 1 / L = 4) :=
by 
  sorry

end leak_empties_tank_in_4_hours_l804_804711


namespace edges_after_truncation_l804_804789

-- Define a regular tetrahedron with 4 vertices and 6 edges
structure Tetrahedron :=
  (vertices : ℕ)
  (edges : ℕ)

-- Initial regular tetrahedron
def initial_tetrahedron : Tetrahedron :=
  { vertices := 4, edges := 6 }

-- Function to calculate the number of edges after truncating vertices
def truncated_edges (t : Tetrahedron) (vertex_truncations : ℕ) (new_edges_per_vertex : ℕ) : ℕ :=
  vertex_truncations * new_edges_per_vertex

-- Given a regular tetrahedron and the truncation process
def resulting_edges (t : Tetrahedron) (vertex_truncations : ℕ) :=
  truncated_edges t vertex_truncations 3

-- Problem statement: Proving the resulting figure has 12 edges
theorem edges_after_truncation :
  resulting_edges initial_tetrahedron 4 = 12 :=
  sorry

end edges_after_truncation_l804_804789


namespace similar_triangle_perimeter_l804_804365

-- Define the side lengths of the original right triangle
def side1 : ℕ := 6     -- Shortest side
def side2 : ℕ := 8
def hypotenuse : ℕ := 10

-- Define the shortest side of the similar triangle
def similar_shortest_side : ℕ := 15

-- Use the conditions and the question to form the statement
theorem similar_triangle_perimeter : ∃ (scale_factor : ℝ) (perimeter : ℕ), 
  scale_factor = similar_shortest_side / side1 ∧
  perimeter = (side1 * scale_factor).toNat + (side2 * scale_factor).toNat + (hypotenuse * scale_factor).toNat ∧
  perimeter = 60 :=
by
  sorry

end similar_triangle_perimeter_l804_804365


namespace find_cost_prices_l804_804383

-- These represent the given selling prices of the items.
def SP_computer_table : ℝ := 3600
def SP_office_chair : ℝ := 5000
def SP_bookshelf : ℝ := 1700

-- These represent the percentage markups and discounts as multipliers.
def markup_computer_table : ℝ := 1.20
def markup_office_chair : ℝ := 1.25
def discount_bookshelf : ℝ := 0.85

-- The problem requires us to find the cost prices. We will define these as variables.
variable (C O B : ℝ)

theorem find_cost_prices :
  (SP_computer_table = C * markup_computer_table) ∧
  (SP_office_chair = O * markup_office_chair) ∧
  (SP_bookshelf = B * discount_bookshelf) →
  (C = 3000) ∧ (O = 4000) ∧ (B = 2000) :=
by
  sorry

end find_cost_prices_l804_804383


namespace no_hot_dogs_l804_804008

def hamburgers_initial := 9.0
def hamburgers_additional := 3.0
def hamburgers_total := 12.0

theorem no_hot_dogs (h1 : hamburgers_initial + hamburgers_additional = hamburgers_total) : 0 = 0 :=
by
  sorry

end no_hot_dogs_l804_804008


namespace domain_M_complement_domain_M_l804_804123

noncomputable def f (x : ℝ) : ℝ :=
  1 / Real.sqrt (1 - x)

noncomputable def g (x : ℝ) : ℝ :=
  Real.log (1 + x)

def M : Set ℝ :=
  {x | 1 - x > 0}

def N : Set ℝ :=
  {x | 1 + x > 0}

def complement_M : Set ℝ :=
  {x | 1 - x ≤ 0}

theorem domain_M :
  M = {x | x < 1} := by
  sorry

theorem complement_domain_M :
  complement_M = {x | x ≥ 1} := by
  sorry

end domain_M_complement_domain_M_l804_804123


namespace neznaika_mistake_correct_numbers_l804_804612

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end neznaika_mistake_correct_numbers_l804_804612


namespace modulus_z_eq_one_l804_804547

noncomputable def z (r : ℝ) (θ : ℝ) : ℂ :=
  let z_val := (r + complex.sqrt (4 - r^2) * complex.I) / 2
  z_val

theorem modulus_z_eq_one {r : ℝ} [hr : |r| < 3] (θ : ℝ) (z : ℂ) 
 (h_r : r = 2 * real.sin θ) (h_z : z + (z⁻¹) = r) :
  |z| = 1 := by
  sorry

end modulus_z_eq_one_l804_804547


namespace sin_periodicity_and_property_l804_804288

theorem sin_periodicity_and_property (x : ℝ) : 
  ∃ n : ℤ, 
  ∀ x : ℝ, 
  sin (n * 360 + x) = sin x ∧ 
  sin (180 + x) = -sin x → 
  sin 2005 = -sin 25 :=
begin
  sorry
end

end sin_periodicity_and_property_l804_804288


namespace sum_of_angles_in_square_l804_804105

theorem sum_of_angles_in_square (A B C D O : Point) (hAB : dist A B = dist B C)
(hBC : dist B C = dist C D) (hCD : dist C D = dist D A) (hAD : dist D A = dist A B)
(hAO : dist A O ≥ 0) (hBO : dist B O ≥ 0) (hCO : dist C O ≥ 0) (hDO : dist D O ≥ 0):
135 ≤ angle O A B + angle O B C + angle O C D + angle O D A ∧
angle O A B + angle O B C + angle O C D + angle O D A ≤ 225 :=
sorry

end sum_of_angles_in_square_l804_804105


namespace Jessie_points_l804_804998

theorem Jessie_points (total_points team_points : ℕ) (players_points : ℕ) (P Q R : ℕ) (eq1 : total_points = 311) (eq2 : players_points = 188) (eq3 : team_points - players_points = 3 * P) (eq4 : P = Q) (eq5 : Q = R) : Q = 41 :=
by
  sorry

end Jessie_points_l804_804998


namespace range_g_l804_804035

open Real

noncomputable def g (x : ℝ) : ℝ := arcsin x + arccos x + arctanh x

theorem range_g : set.range g = set.Ioo_negInf_posInf := by
  sorry

end range_g_l804_804035


namespace symmetric_point_xOz_l804_804914

-- Define the original point P and the xOz plane symmetry condition
def P : ℝ × ℝ × ℝ := (3, 1, 5)

-- Define the symmetric point function w.r.t xOz plane
def symmetric_xOz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, p.3)

-- State the theorem
theorem symmetric_point_xOz : 
  symmetric_xOz P = (3, -1, 5) :=
by
  unfold symmetric_xOz
  unfold P
  rfl

end symmetric_point_xOz_l804_804914


namespace trigonometric_value_l804_804926

theorem trigonometric_value (a b c : ℝ) (B : ℝ)
  (h_collinear : c * (a - c) = (a - b) * (a + b))
  (h_cosine : cos B = (a^2 + c^2 - b^2) / (2 * a * c)) :
  2 * sin (π + B) - 4 * cos (-B) = -√3 - 2 :=
by
  sorry

end trigonometric_value_l804_804926


namespace savings_by_paying_cash_l804_804968

theorem savings_by_paying_cash
  (cash_price : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) (number_of_months : ℕ)
  (h1 : cash_price = 400) (h2 : down_payment = 120) (h3 : monthly_payment = 30) (h4 : number_of_months = 12) :
  cash_price + (monthly_payment * number_of_months - down_payment) - cash_price = 80 :=
by
  sorry

end savings_by_paying_cash_l804_804968


namespace alpha_when_beta_48_l804_804994

variable (α β : ℝ)
variable (k : ℝ)

-- α is inversely proportional to β
def inversely_proportional (α β : ℝ) := α * β = k

-- Given: α = 6 when β = 4
axiom cond1 : inversely_proportional α β ∧ α = 6 ∧ β = 4

-- Prove: α = 1/2 when β = 48
theorem alpha_when_beta_48 : (β = 48) → α = 1 / 2 :=
by
  intro hβ
  sorry

end alpha_when_beta_48_l804_804994


namespace least_natural_number_not_one_digit_l804_804091

def f (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem least_natural_number_not_one_digit :
  let n := 19999999999999999999999
  in f (f (f n)) ≥ 10 :=
sorry

end least_natural_number_not_one_digit_l804_804091


namespace absolute_value_expression_l804_804052

theorem absolute_value_expression : 
  let pi := Real.pi in 
  abs (pi - abs (pi - 7)) = 7 - 2 * pi := 
by {
  sorry
}

end absolute_value_expression_l804_804052


namespace cost_of_eraser_is_1_l804_804372

variables (P : ℝ)  -- Price of a pencil
variables (num_pencils : ℝ) (num_erasers : ℝ) (total_revenue : ℝ)

-- Conditions
def condition1 := num_erasers = 2 * num_pencils
def condition2 := num_pencils = 20
def condition3 := total_revenue = 80
def condition4 := total_revenue = num_pencils * P + num_erasers * (1/2 * P)

-- Theorem: The cost of an eraser is 1 dollar
theorem cost_of_eraser_is_1 : 
    (P : ℝ) → (condition1) → (condition2) → (condition3) → (condition4) → (1/2 * P) = 1 :=
by
  sorry

end cost_of_eraser_is_1_l804_804372


namespace lucy_needs_more_distance_l804_804967

noncomputable def mary_distance : ℝ := (3 / 8) * 24
noncomputable def edna_distance : ℝ := (2 / 3) * mary_distance
noncomputable def lucy_distance : ℝ := (5 / 6) * edna_distance

theorem lucy_needs_more_distance :
  mary_distance - lucy_distance = 4 := by
  sorry

end lucy_needs_more_distance_l804_804967


namespace propositions_correct_l804_804950

-- Define properties of lines and planes
variable {m n : Type*} [linear_order m] [linear_order n]
variable {α β γ : Type*} [linear_order α] [linear_order β] [linear_order γ]
variable {a : α}

-- Define parallel and perpendicular relationships
constant parallel : α → β → Prop
constant perpendicular : α → β → Prop

-- The propositions as given in the problem statement
def prop1 : Prop := ∀ {a β γ : Type*} [linear_order a] [linear_order β] [linear_order γ], 
  parallel a β ∧ parallel a γ → parallel β γ

def prop2 : Prop := parallel α β ∧ parallel m α → perpendicular m β

def prop3 : Prop := perpendicular α γ ∧ perpendicular β γ → parallel α β

def prop4 : Prop := parallel m n ∧ perpendicular n α → perpendicular m α

-- The theorem stating which propositions are correct
theorem propositions_correct : prop1 ∧ prop4 ∧ ¬prop2 ∧ ¬prop3 :=
by sorry

end propositions_correct_l804_804950


namespace find_q_zero_plus_q_five_l804_804215

-- Definitions of the conditions
def monic_polynomial_degree_5 (q : ℝ → ℝ) : Prop :=
  ∃ (a b c d e : ℝ), q = λ x, x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

def satisfies_conditions (q : ℝ → ℝ) : Prop :=
  monic_polynomial_degree_5 q ∧ q 1 = 20 ∧ q 2 = 40 ∧ q 3 = 60 ∧ q 4 = 80

-- Statement to be proved
theorem find_q_zero_plus_q_five (q : ℝ → ℝ) (h : satisfies_conditions q) : q 0 + q 5 = 220 := 
sorry

end find_q_zero_plus_q_five_l804_804215


namespace positive_solution_sqrt_eq_l804_804808

theorem positive_solution_sqrt_eq (y : ℝ) (hy_pos : 0 < y) : 
    (∃ a, a = y ∧ a^2 = y * a) ∧ (∃ b, b = y ∧ b^2 = y + b) ∧ y = 2 :=
by 
  sorry

end positive_solution_sqrt_eq_l804_804808


namespace remainder_f_x5_div_f_x_l804_804161

theorem remainder_f_x5_div_f_x {R : Type*} [CommRing R] (x : R) :
  let f := x^4 + x^3 + x^2 + x + 1
  in (poly_eval f (x^5) % f = 5) :=
by
  let f := x^4 + x^3 + x^2 + x + 1
  exact sorry

end remainder_f_x5_div_f_x_l804_804161


namespace complement_B_in_A_l804_804557

noncomputable def A : Set ℝ := {x | x < 2}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x < 2}

theorem complement_B_in_A : {x | x ∈ A ∧ x ∉ B} = {x | x ≤ 1} :=
by
  sorry

end complement_B_in_A_l804_804557


namespace determine_m_l804_804894

theorem determine_m (m : ℝ) :
  (∀ x ∈ set.Icc 0 3, (λ x, x^2 - 2*x + m) x ≤ 1) ∧
  (∀ x ∈ set.Icc 0 3, (λ x, x^2 - 2*x + m) x = 1 → x = 3) →
  m = -2 :=
begin
  sorry
end

end determine_m_l804_804894


namespace count_lucky_tuples_l804_804441

-- Define the set P_n
def P (n : ℕ) : Set ℕ := { k | ∃ i : ℕ, k = n^i }

-- Conditions for being a "lucky" tuple
def is_lucky (a b c m : ℕ) : Prop :=
  (a - 1 ∈ P m) ∧ (ab - 12 ∈ P m) ∧ (abc - 2015 ∈ P m)

-- Main theorem stating the number of lucky tuples
theorem count_lucky_tuples : ℕ :=
  { p : ℕ × ℕ × ℕ × ℕ // is_lucky p.1 p.2.1 p.2.2.1 p.2.2.2 }.card = 25 :=
sorry

end count_lucky_tuples_l804_804441


namespace probability_of_draw_l804_804299

noncomputable def P_A_winning : ℝ := 0.4
noncomputable def P_A_not_losing : ℝ := 0.9

theorem probability_of_draw : P_A_not_losing - P_A_winning = 0.5 :=
by 
  sorry

end probability_of_draw_l804_804299


namespace max_single_player_salary_l804_804746

/-- A semipro baseball league has teams with 21 players each.
    League rules state that a player must be paid at least $15,000 dollars,
    and that the total of all players' salaries for each team cannot exceed $700,000 dollars.
    Prove that the maximum possible salary for a single player is $400,000 dollars. -/
theorem max_single_player_salary (n : ℕ) (m s : ℕ) (h1 : n = 21) (h2 : ∀ i, i < n → s i ≥ 15000) (h3 : ∑ i in finset.range n, s i ≤ 700000) :
  nat.max (s '' finset.range n) = 400000 := by
  sorry

end max_single_player_salary_l804_804746


namespace number_of_integer_triangles_l804_804875

theorem number_of_integer_triangles :
  (card { t : ℕ × ℕ × ℕ // 
    let (a, b, c) := t in 
    a < b ∧ b < c ∧ 
    a + b + c < 15 ∧ 
    a + b > c ∧
    ¬(a = b ∧ b = c) ∧ 
    ¬(a = b ∨ b = c ∨ a = c) ∧
    ¬(a^2 + b^2 = c^2) }) = 5 := 
sorry

end number_of_integer_triangles_l804_804875


namespace number_of_points_on_parabola_l804_804423

open Nat

def parabola (x : ℕ) : ℚ := - (x^2)/4 + 9 * x + 19

def is_nat (y : ℚ) : Prop := ∃ n : ℕ, y = n

theorem number_of_points_on_parabola :
  (finset.filter (λ x => is_nat (parabola x)) (finset.range 38)).card = 18 := by
sorry

end number_of_points_on_parabola_l804_804423


namespace ratio_of_black_to_blue_l804_804247

universe u

-- Define the types of black and red pens
variables (B R : ℕ)

-- Define the conditions
def condition1 : Prop := 2 + B + R = 12
def condition2 : Prop := R = 2 * B - 2

-- Define the proof statement
theorem ratio_of_black_to_blue (h1 : condition1 B R) (h2 : condition2 B R) : B / 2 = 1 :=
by
  sorry

end ratio_of_black_to_blue_l804_804247


namespace investment_after_five_years_l804_804375

noncomputable def total_amount_after_five_years 
  (P₁ : ℝ) (r : ℝ) (n₁ : ℕ) (P₂ : ℝ) (n₂ : ℕ) : ℝ :=
(P₁ * (1 + r / 2) ^ n₁ + P₂) * (1 + r / 2) ^ n₂

theorem investment_after_five_years
  (P₁ : ℝ := 12000)
  (r : ℝ := 0.045)
  (n₁ : ℕ := 6)
  (P₂ : ℝ := 2000)
  (n₂ : ℕ := 4) :
  total_amount_after_five_years P₁ r n₁ P₂ n₂ ≈ 17172 :=
by
  sorry

end investment_after_five_years_l804_804375


namespace main_purpose_of_regulation_l804_804173

-- Define the conditions and the goal statement
variable (debt_assets equity_assets : Type)
variable (PensionFund : Type)
variable (invest_in : PensionFund → debt_assets → Prop)
variable (max_equity_investment : ℝ := 0.30)

-- Define the statement that the pension fund can invest in both types of assets
variable (invest_in_debt : ∀ (pf : PensionFund) (d : debt_assets), invest_in pf d)
variable (invest_in_equity : ∀ (pf : PensionFund) (e : equity_assets), invest_in pf e)

-- Define the condition that the proportion of equity investment should not exceed 30%
axiom equity_investment_limit (pf : PensionFund) (e : equity_assets) : Prop :=
  proportion_invested pf e ≤ max_equity_investment

-- The main purpose of the regulations
theorem main_purpose_of_regulation :
  (∀ (pf : PensionFund), invest_in_debt pf ∨ (∃ e, invest_in_equity pf e ∧ equity_investment_limit pf e)) →
  optimize_investment_portfolio :=
sorry

end main_purpose_of_regulation_l804_804173


namespace ellipse_eq_area_opq_perp_op_oq_l804_804182

-- Definitions for the conditions
def ellipse := {x y a b : ℝ // a > b ∧ 0 < b ∧ x^2 / a^2 + y^2 / b^2 = 1}

def eccentricity := {a c : ℝ // c / a = sqrt 2 / 2}

def point_on_ellipse (a b : ℝ) := {x y : ℝ // x = 2 ∧ y = 1 ∧ x^2 / a^2 + y^2 / b^2 = 1}

def circle := {x y : ℝ // x^2 + y^2 = 2}

def right_focus (a b : ℝ) := {x y : ℝ // x = sqrt (a^2 - b^2) ∧ y = 0}

-- Problem Statements
theorem ellipse_eq (a b : ℝ) (h₀ : ellipse 2 1 a b) (h₁ : eccentricity a 3.c) : x^2 / 6 + y^2 / 3 = 1 :=
  sorry

theorem area_opq (a b : ℝ) (h₀ : ellipse 2 1 a b) (h₁ : circle ∃ p q : circle, right_focus a b) : 
area_triangle O P Q = 6 * sqrt (3 / 5) :=
  sorry

theorem perp_op_oq (a b : ℝ) (h₀ : ellipse 2 1 a b) (h₁ : tangent_line_eq a b) (h₂ : right_focus a b) :
⊥ O P Q :=
  sorry

end ellipse_eq_area_opq_perp_op_oq_l804_804182


namespace probability_of_same_color_is_4_over_9_l804_804157

noncomputable def probability_same_color : ℚ :=
  let total_ways := (Nat.choose 9 2) in
  let red_ways := (Nat.choose 5 2) in
  let blue_ways := (Nat.choose 4 2) in
  let same_color_ways := red_ways + blue_ways in
  (same_color_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_same_color_is_4_over_9 : probability_same_color = 4 / 9 :=
by
  sorry

end probability_of_same_color_is_4_over_9_l804_804157


namespace x_intercept_of_perpendicular_line_l804_804657

theorem x_intercept_of_perpendicular_line 
  (a : ℝ)
  (l1 : ℝ → ℝ → Prop)
  (l1_eq : ∀ x y, l1 x y ↔ (a+3)*x + y - 4 = 0)
  (l2 : ℝ → ℝ → Prop)
  (l2_eq : ∀ x y, l2 x y ↔ x + (a-1)*y + 4 = 0)
  (perpendicular : ∀ x y, l1 x y → l2 x y → (a+3)*(a-1) = -1) :
  (∃ x : ℝ, l1 x 0 ∧ x = 2) :=
sorry

end x_intercept_of_perpendicular_line_l804_804657


namespace proof_problem_l804_804555

noncomputable theory

-- Definitions based on given conditions
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 4 = 1
def line_l (k x y : ℝ) : Prop := y = k * x
def line_AB (x y : ℝ) : Prop := x + y = 2

-- Main theorem
theorem proof_problem (e : ℝ) (a b : ℝ) (F B A : ℝ × ℝ)
  (k : ℝ) (sin_angle_AOQ : ℝ) :
  e = sqrt 5 / 3 ∧
  F = (- a * sqrt (1 - (b^2 / a^2)), 0) ∧
  B = (0, b) ∧
  A = (b, 0) ∧
  abs (a - b) * b = 6 * sqrt 2 ∧
  (line_l k).intersect (λ x y, ellipse_eq x y) (λ P, P.1 > 0 ∧ P.2 > 0) = λ Q, q = (root_of_some_eq) ∧
  abs (A.1 - Q.1) / abs (P.1 - Q.1) = (5 * sqrt 2) / 4 * sin_angle (∠ (0, 0) AOQ) ∧
  ∃ k : ℝ, k = 1/2 ∨ k = 11/28 := by
  sorry

end proof_problem_l804_804555


namespace arithmetic_progression_roots_l804_804405
noncomputable def given_conditions := -\frac{850}{27}

theorem arithmetic_progression_roots (a : ℝ) :
  (∃ (r d : ℂ), (r - d + r + r + d) = 10 / 2 ∧ (
  (r - d) * r + (r - d) * (r + d) + r * (r + d) = 30 / 2) ∧
  (-2) * (r - d) * r * (r + d) + a = 0)  →
  a = -\frac{850}{27} :=
begin
  sorry
end

end arithmetic_progression_roots_l804_804405


namespace possible_numbers_l804_804592

theorem possible_numbers (N : ℕ) 
    (h1 : 10 ≤ N) (h2 : N ≤ 99)
    (h3 : (N % 3 = 0) ∧ (N % 4 = 0) ∧ (N % 5 = 0) ∧ (N % 9 = 0) ∧ (N % 10 = 0) ∧ 
          (N % 15 = 0) ∧ (N % 18 = 0) ∧ (N % 30 = 0) ∨ 
          (N % 3 ≠ 0) + (N % 4 ≠ 0) + (N % 5 ≠ 0) + (N % 9 ≠ 0) + 
          (N % 10 ≠ 0) + (N % 15 ≠ 0) + (N % 18 ≠ 0) + (N % 30 ≠ 0) = 4) :
   N = 36 ∨ N = 45 ∨ N = 72 :=
by {
  sorry
}

end possible_numbers_l804_804592


namespace gcd_factorial_7_12_div_5_l804_804308

-- Declare the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Definition of the gcd function
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Problem statement: Prove that gcd of 7! and (12! / 5!) is 2520
theorem gcd_factorial_7_12_div_5 : gcd (factorial 7) (factorial 12 / factorial 5) = 2520 := 
by 
  sorry  -- Proof not needed according to instructions.

end gcd_factorial_7_12_div_5_l804_804308


namespace expression_equals_384_l804_804338

noncomputable def problem_expression : ℤ :=
  2021^4 - 4 * 2023^4 + 6 * 2025^4 - 4 * 2027^4 + 2029^4

theorem expression_equals_384 : problem_expression = 384 := by
  sorry

end expression_equals_384_l804_804338


namespace salmon_oxygen_ratio_l804_804414

def salmon_speed (O : ℝ) : ℝ := (1/2) * log_base 3 (O / 100)

theorem salmon_oxygen_ratio (O1 O2 : ℝ) (v : ℝ):
  salmon_speed O1 = v →
  salmon_speed O2 = v + 2 →
  O2 / O1 = 81 :=
by 
  sorry

end salmon_oxygen_ratio_l804_804414


namespace trailing_zeros_2012_factorial_l804_804767

theorem trailing_zeros_2012_factorial : 
  ∑ k in finset.range (nat.log 5 2012 + 1), 2012 / 5^k = 501 := 
sorry

end trailing_zeros_2012_factorial_l804_804767


namespace number_of_ways_to_remove_rings_l804_804689

theorem number_of_ways_to_remove_rings : 
  ∃ n, (n = 20) ∧ 
       (∀ rings : Fin₅ → 𝒫 Set,
          (∃ l_r, rings (0 : Fin₅) = {l_r}) ∧
          (∃ m_r, rings (1 : Fin₅) = {m_r}) ∧
          (∃ r_rs, rings (2 : Fin₅) = r_rs ∧ #r_rs = 3))
sorry

end number_of_ways_to_remove_rings_l804_804689


namespace regular_tetrahedron_fourth_vertex_l804_804363

theorem regular_tetrahedron_fourth_vertex :
  ∃ (x y z : ℤ), 
    ((x, y, z) = (0, 0, 6) ∨ (x, y, z) = (0, 0, -6)) ∧
    ((x - 0) ^ 2 + (y - 0) ^ 2 + (z - 0) ^ 2 = 36) ∧
    ((x - 6) ^ 2 + (y - 0) ^ 2 + (z - 0) ^ 2 = 36) ∧
    ((x - 5) ^ 2 + (y - 0) ^ 2 + (z - 6) ^ 2 = 36) := 
by
  sorry

end regular_tetrahedron_fourth_vertex_l804_804363


namespace number_of_moles_HCl_combined_is_one_l804_804422

-- Declare the types and variables
variables (HCl NaHCO₃ NaCl CO₂ H₂O : Type) 
variable number_of_moles : ℝ -- Using real numbers to for the number of moles

-- Definitions of the reaction and conditions
def reacts (a b : Type) (product1 product2 product3 : Type) : Prop :=
  a → b → product1 ∧ product2 ∧ product3

-- Specific conditions for the problem
def chemical_reaction := reacts HCl NaHCO₃ NaCl CO₂ H₂O

-- Given information in the problem
def one_mole : ℝ := 1
def condition_on_moles (m : ℝ) (reacts : reacts HCl NaHCO₃ NaCl CO₂ H₂O) : Prop :=
  m = one_mole → reacts

-- Problem statement to be proved
theorem number_of_moles_HCl_combined_is_one (h : condition_on_moles 1 chemical_reaction) :
  number_of_moles = 1 :=
sorry

end number_of_moles_HCl_combined_is_one_l804_804422


namespace lemmings_distance_average_l804_804741

noncomputable def diagonal_length (side: ℝ) : ℝ :=
  Real.sqrt (side^2 + side^2)

noncomputable def fraction_traveled (side: ℝ) (distance: ℝ) : ℝ :=
  distance / (Real.sqrt 2 * side)

noncomputable def final_coordinates (side: ℝ) (distance1: ℝ) (angle: ℝ) (distance2: ℝ) : (ℝ × ℝ) :=
  let frac := fraction_traveled side distance1
  let initial_pos := (frac * side, frac * side)
  let move_dist := distance2 * (Real.sqrt 2 / 2)
  (initial_pos.1 + move_dist, initial_pos.2 + move_dist)

noncomputable def average_shortest_distances (side: ℝ) (coords: ℝ × ℝ) : ℝ :=
  let x_dist := min coords.1 (side - coords.1)
  let y_dist := min coords.2 (side - coords.2)
  (x_dist + (side - x_dist) + y_dist + (side - y_dist)) / 4

theorem lemmings_distance_average :
  let side := 15
  let distance1 := 9.3
  let angle := 45 / 180 * Real.pi -- convert to radians
  let distance2 := 3
  let coords := final_coordinates side distance1 angle distance2
  average_shortest_distances side coords = 7.5 :=
by
  sorry

end lemmings_distance_average_l804_804741


namespace intersection_B_C_l804_804457

-- Definitions of sets A, B, and C
def A : Set ℕ := {1, 2, 3, 6, 9}
def B : Set ℕ := { x | ∃ a ∈ A, x = 3 * a }
def C : Set ℕ := { x | x ∈ ℕ ∧ (3 * x) ∈ A }

-- Prove B ∩ C = {3}
theorem intersection_B_C : B ∩ C = {3} :=
  sorry

end intersection_B_C_l804_804457


namespace subset_inequality_l804_804222

-- Definitions of the sets and projections
def is_xyz_point (p : ℤ × ℤ × ℤ) : Prop := true

def V : finset (ℤ × ℤ × ℤ) := {val := finset.univ.val.filter is_xyz_point, nodup := sorry}

def S1 : finset (ℤ × ℤ) := V.image (λ ⟨x, y, z⟩, (y, z))

def S2 : finset (ℤ × ℤ) := V.image (λ ⟨x, y, z⟩, (x, z))

def S3 : finset (ℤ × ℤ) := V.image (λ ⟨x, y, z⟩, (x, y))

-- The theorem statement
theorem subset_inequality : V.card ^ 2 ≤ S1.card * S2.card * S3.card := 
sorry

end subset_inequality_l804_804222


namespace neznika_number_l804_804576

theorem neznika_number (N : ℕ) :
  10 ≤ N ∧ N ≤ 99 ∧
  (divisible_by N 3 ∧ divisible_by N 4 ∧ divisible_by N 5 ∧
   divisible_by N 9 ∧ divisible_by N 10 ∧ divisible_by N 15 ∧
   divisible_by N 18 ∧ divisible_by N 30) →
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end neznika_number_l804_804576


namespace fraction_one_two_three_sum_l804_804905

def fraction_one_bedroom : ℝ := 0.12
def fraction_two_bedroom : ℝ := 0.26
def fraction_three_bedroom : ℝ := 0.38
def fraction_four_bedroom : ℝ := 0.24

theorem fraction_one_two_three_sum :
  fraction_one_bedroom + fraction_two_bedroom + fraction_three_bedroom = 0.76 :=
by
  sorry

end fraction_one_two_three_sum_l804_804905


namespace infinite_countable_planar_graph_coloring_l804_804691

theorem infinite_countable_planar_graph_coloring 
  (H : ∀ (G : Type) [Fintype G], ∀ (V : G → Type) (E : G → G → Prop), 
    (PlanarGraph G V E) → 
    (∃ (C : G → fin 3), ∀ (Cyc : List G), (length Cyc % 2 = 1) → (∀ v ∈ Cyc, C v = C Cyc.nth 0) → false))
    (G : Type) [Inhabited G] (V : G → Type) (E : G → G → Prop) 
    [CountablyInfinite (Σ (v : G), V v)] : 
    (PlanarGraph G V E) → 
    ∃ (C : (Σ (v : G), V v) → fin 3), ∀ (Cyc : List (Σ (v : G), V v)), 
    (length Cyc % 2 = 1) → (∀ v ∈ Cyc, C v = C Cyc.nth 0) → false :=
by
  sorry

end infinite_countable_planar_graph_coloring_l804_804691


namespace x_range_condition_l804_804656

-- Define the inequality and conditions
def inequality (x : ℝ) : Prop := x^2 + 2 * x < 8

-- The range of x must be (-4, 2)
theorem x_range_condition (x : ℝ) : inequality x → x > -4 ∧ x < 2 :=
by
  intro h
  sorry

end x_range_condition_l804_804656


namespace new_arithmetic_mean_l804_804758

theorem new_arithmetic_mean (S : Fin 60 → ℝ) (h_mean : (∑ i, S i) / 60 = 45) (h_max1 : ∃ i, S i = 70) (h_max2 : ∃ j, S j = 80) (h_ij_distinct : ∀ i j, i ≠ j → S i = 70 ∨ S i = 80 → ¬ S j = S i) : 
  (∑ i, if S i = 70 ∨ S i = 80 then 0 else S i) / 58 = 44 := 
by
  sorry

end new_arithmetic_mean_l804_804758


namespace complement_U_A_correct_l804_804868

-- Step 1: Define the universal set U
def U (x : ℝ) := x > 0

-- Step 2: Define the set A
def A (x : ℝ) := 0 < x ∧ x < 1

-- Step 3: Define the complement of A in U
def complement_U_A (x : ℝ) := U x ∧ ¬ A x

-- Step 4: Define the expected complement
def expected_complement (x : ℝ) := x ≥ 1

-- Step 5: The proof problem statement
theorem complement_U_A_correct (x : ℝ) : complement_U_A x = expected_complement x := by
  sorry

end complement_U_A_correct_l804_804868


namespace binary_to_decimal_101_l804_804780

theorem binary_to_decimal_101 : 
  (1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 5 := 
by 
  calc
    1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 4 + 0 + 1 : by norm_num
                            ... = 5         : by norm_num

end binary_to_decimal_101_l804_804780


namespace expression_not_defined_at_9_l804_804093

theorem expression_not_defined_at_9 :
  ∀ x : ℝ, x = 9 → (x^2 - 18 * x + 81 = 0) :=
begin
  intros x hx,
  rw hx,
  ring,
end

end expression_not_defined_at_9_l804_804093


namespace missing_fraction_is_two_l804_804669

theorem missing_fraction_is_two :
  (1/2) + (-5/6) + (1/5) + (1/4) + (-9/20) + (-5/6) + 2 = 0.8333333333333334 := by
  sorry

end missing_fraction_is_two_l804_804669


namespace magnitude_of_expression_l804_804963

noncomputable def z1 : ℂ := 3 + 2 * complex.I
noncomputable def z2 : ℂ := 1 - complex.I

theorem magnitude_of_expression : complex.abs (z1 + 2 / z2) = 5 :=
by
  let z1 := 3 + 2 * complex.I
  let z2 := 1 - complex.I
  have h : z1 + 2 / z2 = 4 + 3 * complex.I := by
    calc
      z1 + 2 / z2
          = 3 + 2 * complex.I + 2 / (1 - complex.I) : by rfl
      ... = 3 + 2 * complex.I + 2 * (1 + complex.I) / ((1 - complex.I) * (1 + complex.I)) : by norm_num
      ... = 3 + 2 * complex.I + 2 * (1 + complex.I) / (1 + 1) : by ring
      ... = 3 + 2 * complex.I + 1 + complex.I : by norm_num
      ... = 4 + 3 * complex.I : by ring
  have h1 := h.symm ▸ rfl
  calc
    complex.abs (4 + 3 * complex.I) = real.sqrt (4^2 + 3^2) : by rw complex.abs
    ... = real.sqrt (16 + 9) : by norm_num
    ... = real.sqrt 25 : by ring
    ... = 5 : by norm_num

end magnitude_of_expression_l804_804963


namespace cone_volume_correct_l804_804850

def cone_volume (lateral_surface_area : ℝ) (slant_height : ℝ) : ℝ :=
  let r := 2 * lateral_surface_area / slant_height / π in
  let h := Real.sqrt (slant_height ^ 2 - r ^ 2) in
  (1 / 3) * π * r ^ 2 * h

theorem cone_volume_correct :
  cone_volume (15 * π) 5 = 12 * π :=
by
  sorry

end cone_volume_correct_l804_804850


namespace find_current_l804_804509

-- Definitions based on the given conditions
def V : ℂ := 2 - 2 * Complex.I
def Z1 : ℂ := 3 + 4 * Complex.I
def Z2 : ℂ := 1 - Complex.I

-- Theorem to prove the current I
theorem find_current : 
  let Z := Z1 + Z2 in
  let I := V / Z in
  I = (14 / 25) - (14 / 25) * Complex.I :=
by
  -- Placeholder for the proof
  sorry

end find_current_l804_804509


namespace max_knots_no_cycle_l804_804532

/-- Let P be a 2019-gon such that no three of its diagonals concur at an internal point, 
    where each internal intersection point of diagonals of P is called a knot.
    This theorem proves that the maximum number of knots one can choose such that no cycle exists 
    is 2018. -/
theorem max_knots_no_cycle (P : Poly) (h : P.sides = 2019) 
  (no_concur : ∀ (x y z : Diagonal), knots_intersect x y z → false) :
  ∃ knots : Finset Knot, knots.card = 2018 ∧ acyclic_knots knots :=
sorry

end max_knots_no_cycle_l804_804532


namespace at_least_one_perpendicular_l804_804168

variables (α β : Plane) (n m : Line)

-- Definition of the conditions
def plane_perpendicular (α β : Plane) : Prop := α.perp β
def line_in_plane (l : Line) (α : Plane) : Prop := ∃ p : Point, p ∈ l ∧ p ∈ α
def line_perpendicular (l1 l2 : Line) : Prop := l1.perp l2

-- Main statement
theorem at_least_one_perpendicular 
  (h1 : plane_perpendicular α β)
  (h2 : line_in_plane n α) 
  (h3 : line_in_plane m β) 
  (h4 : line_perpendicular m n) : 
  (line_perpendicular n ⟨β⟩) ∨ (line_perpendicular m ⟨α⟩) :=
sorry

end at_least_one_perpendicular_l804_804168


namespace problem_I_problem_II_l804_804469

def f (x a : ℝ) : ℝ := abs(3 * x + 2) - abs(2 * x + a)

-- Statement (I)
theorem problem_I (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ (a = 4 / 3) :=
sorry

-- Statement (II)
theorem problem_II (a : ℝ) : (∃ x : ℝ, x ∈ set.Icc 1 2 ∧ f x a ≤ 0) ↔ (a ∈ set.Ici 3 ∪ set.Iic (-7)) :=
sorry

end problem_I_problem_II_l804_804469


namespace sum_of_solutions_to_the_equation_l804_804427

noncomputable def sum_of_real_solutions : ℚ := 
  have h : (∀ x : ℚ, (x - 3) * (x^2 - 12 * x) = (x - 6) * (x^2 + 5 * x + 2)) → 
             (∀ x : ℚ, 14 * x^2 - 64 * x - 12 = 0) := 
  by sorry,
  (32 : ℚ) / 7

theorem sum_of_solutions_to_the_equation :
  (∀ x : ℚ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x) ↔ 
            14 * x^2 - 64 * x - 12 = 0) → 
              sum_of_real_solutions = 32 / 7 :=
by sorry

end sum_of_solutions_to_the_equation_l804_804427


namespace sum_of_real_solutions_eq_neg_32_div_7_l804_804431

theorem sum_of_real_solutions_eq_neg_32_div_7 :
  ∀ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x) →
  ∑ sol in { x | (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x) }, sol = -32 / 7 :=
by
  sorry

end sum_of_real_solutions_eq_neg_32_div_7_l804_804431


namespace find_k_l804_804480

def vectors_collinear (a b : Vector 2 ℝ) : Prop :=
  a 0 * b 1 = a 1 * b 0

theorem find_k (k : ℝ) (a b : Vector 2 ℝ) : 
  (a = ![1, k]) → 
  (b = ![2, 2]) → 
  vectors_collinear (a + b) a → 
  k = 1 := 
by
  sorry

end find_k_l804_804480


namespace moles_of_reactants_and_products_l804_804481

-- Define the reactants and products as constants
constant NH4NO3 : Type
constant NaOH : Type
constant NaNO3 : Type
constant NH3 : Type
constant H2O : Type

-- Define the balanced chemical equation condition
-- Assume 1 to be the stoichiometric coefficient for simplicity
def balanced_eq (a b c d e : ℕ) := a * 1 + b * 1 = c * 1 + d * 1 + e * 1

-- Given: The balanced equation for NH4NO3 + NaOH -> NaNO3 + NH3 + H2O
axiom balanced_chemical_eq : balanced_eq 1 1 1 1 1

-- Define the specific reaction condition
def reaction_condition (moles_NaOH : ℕ) := moles_NaOH = 2

-- Define the theorem to be proven based on the problem statement
theorem moles_of_reactants_and_products (moles_NaOH : ℕ) (moles_NH4NO3 : ℕ) (moles_NaNO3 : ℕ) :
  reaction_condition moles_NaOH → balanced_eq 1 1 1 1 1 →
  moles_NH4NO3 = 2 ∧ moles_NaNO3 = 2 :=
by
  intro h1 h2
  sorry

end moles_of_reactants_and_products_l804_804481


namespace number_of_circles_through_F_M_and_tangent_to_l_l804_804892

noncomputable def parabola : set (ℝ × ℝ) := { p | p.2 ^ 2 = 8 * p.1 }

def point_F : ℝ × ℝ := (2, 0)  -- Calculated focus of the parabola y^2 = 8x
def point_M : ℝ × ℝ := (3, 3)
def directrix : ℝ × ℝ → Prop := λ p, p.1 = -2 -- Equation of the directrix x = -2

theorem number_of_circles_through_F_M_and_tangent_to_l :
  ∃ (c : set (ℝ × ℝ)), 
    ∃ (center : ℝ × ℝ), 
      center ∈ parabola ∧ 
      set.finite c ∧ 
      2 = set.card c ∧
      (∀ (p ∈ c), (dist p point_F) = (dist p center)) ∧
      (∀ (p ∈ c), (dist p point_M) = (dist p center)) ∧
      (∀ (p ∈ c), directrix (proj_to_directrix p))
        :=
sorry

end number_of_circles_through_F_M_and_tangent_to_l_l804_804892


namespace min_positive_ones_in_grid_l804_804178

theorem min_positive_ones_in_grid : 
  ∀ (grid : fin 4 → fin 4 → ℤ), 
    (∀ (r1 r2 : fin 4) (c1 c2 : fin 4), r1 ≠ r2 → c1 ≠ c2 → 
       (0 ≤ grid (⟨0, Nat.zero_lt_succ 3⟩) c1 + grid (⟨0, Nat.zero_lt_succ 3⟩) c2 + 
              grid (⟨1, Nat.succ_lt_succ Nat.zero_lt_one⟩) c1 + grid (⟨1, Nat.succ_lt_succ Nat.zero_lt_one⟩) c2)) 
    → (∑ i j, if grid i j = 1 then 1 else 0) ≥ 10 :=
by
  intro grid condition
  sorry

end min_positive_ones_in_grid_l804_804178


namespace each_episode_length_l804_804196

theorem each_episode_length (h_watch_time : ∀ d : ℕ, d = 5 → 2 * 60 * d = 600)
  (h_episodes : 20 > 0) : 600 / 20 = 30 := by
  -- Conditions used:
  -- 1. h_watch_time : John wants to finish a show in 5 days by watching 2 hours a day.
  -- 2. h_episodes : There are 20 episodes.
  -- Goal: Prove that each episode is 30 minutes long.
  sorry

end each_episode_length_l804_804196


namespace find_incircle_radius_l804_804714

-- Define the necessary parameters for the isosceles triangle and the inscribed circle
structure Triangle where
  A B C : Point
  is_isosceles : isIsosceles A B C
  base_length : length B C = 8
  height : heightDropped A B C = 3

-- Define the properties of the inscribed circle (incircle)
structure Incircle where
  triangle : Triangle
  center : Point
  radius : ℝ
  is_inscribed : isInscribedCircle center radius triangle

-- Define the proposition to be proved
theorem find_incircle_radius (T : Triangle) (I : Incircle) : I.radius = 20 / 3 := by
  -- Proof goes here
  sorry

end find_incircle_radius_l804_804714


namespace domain_of_transformed_function_l804_804169

variable (f : ℝ → ℝ)

theorem domain_of_transformed_function :
  (∃ (D : set ℝ), (∀ x, 2 * x - 1 ∈ Icc (-1 : ℝ) (1 : ℝ) ↔ x ∈ D)) →
  (∃ (D' : set ℝ), D' = {x : ℝ | 1 < x ∧ x ≤ 2}) →
  ∀ x, (x > 1 ∧ x ≤ 2 → (x - 1) ∈ D → 0 < sqrt (x -1)) →
  ∀ x, x ∈ {x : ℝ | (x > 1 ∧ x ≤ 2)} :=
by
  sorry

end domain_of_transformed_function_l804_804169


namespace fraction_of_odd_products_l804_804910

theorem fraction_of_odd_products :
  let table : list (ℕ × ℕ) := 
    (list.range 16).product (list.range 16)
  let odd_products : list ((ℕ × ℕ)) :=
    table.filter (λ ⟨x, y⟩, x % 2 = 1 ∧ y % 2 = 1)
  (odd_products.length : ℚ) / (table.length : ℚ) = 0.25 :=
by
  let table := (list.range 16).product (list.range 16)
  let odd_numbers := [1, 3, 5, 7, 9, 11, 13, 15]
  let odd_products := list.product odd_numbers odd_numbers
  have total_products : table.length = 256 := by simp
  have odd_products_count : odd_products.length = 64 := by simp
  have fraction : (odd_products.length : ℚ) / (table.length : ℚ) = 0.25 :=
    by norm_num [total_products, odd_products_count]
  exact fraction

end fraction_of_odd_products_l804_804910


namespace find_f_expression_find_f_prime_expression_l804_804447

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^2 - b * x + c

theorem find_f_expression :
  ∀ (b c : ℝ),
  (f 1 b c = 0) → (f 2 b c = -3) → (f x 6 5 = x^2 - 6 * x + 5) :=
by
  sorry

noncomputable def f' (x : ℝ) : ℝ := x^2 - 6 * x + 5

theorem find_f_prime_expression :
  ∀ (x : ℝ),
  (x > -1) → 
  (f' (1 / sqrt (x + 1)) = (1 / (x + 1)) - (6 / sqrt (x + 1)) + 5) :=
by
  sorry

end find_f_expression_find_f_prime_expression_l804_804447


namespace fourier_series_f_l804_804318

noncomputable def f (x : Real) : Real :=
  if -π < x ∧ x < π then x else if x = -π then 0 else f (x - 2*π)

theorem fourier_series_f :
  ∀ x, f x = 2 * ∑ k in Finset.range (Nat.succ (Nat.pred Natural.infinity)), 
    (1 : Real) / k * (-1)^(k + 1) * sin (k * x) := by
  sorry

end fourier_series_f_l804_804318


namespace arrangements_of_6_books_l804_804239

theorem arrangements_of_6_books : ∃ (n : ℕ), n = 720 ∧ n = Nat.factorial 6 :=
by
  use 720
  constructor
  · rfl
  · sorry

end arrangements_of_6_books_l804_804239


namespace sin_870_equals_half_l804_804049

theorem sin_870_equals_half :
  sin (870 * Real.pi / 180) = 1 / 2 := 
by
  -- Angle simplification
  have h₁ : 870 - 2 * 360 = 150 := by norm_num,
  -- Sine identity application
  have h₂ : sin (150 * Real.pi / 180) = sin (30 * Real.pi / 180) := by
    rw [mul_div_cancel_left 150 (ne_of_lt (by norm_num : 0 < (Real.pi : ℝ)) : (Real.pi : ℝ) ≠ 0)],
    congr,
    norm_num,

  -- Sine 30 degrees value
  have h₃ : sin (30 * Real.pi / 180) = 1 / 2 := by norm_num,

  -- Combine results
  rw [mul_div_cancel_left 870 (ne_of_lt (by norm_num : 0 < (Real.pi : ℝ)) : (Real.pi : ℝ) ≠ 0), h₁, h₂, h₃],
  sorry

end sin_870_equals_half_l804_804049


namespace buttons_in_sixth_box_l804_804317

theorem buttons_in_sixth_box (a₁ a₂ a₃ a₄ a₅: ℕ) (r: ℕ) 
  (h₁: a₁ = 1) (h₂: a₂ = r * a₁) (h₃: a₃ = r * a₂) 
  (h₄: a₄ = r * a₃) (h₅: a₅ = r * a₄) (hr: r = 3) (ha₅: a₅ = 81) : 
  a₆ = r * a₅ → a₆ = 243 := 
by
  intro h₆
  have ha₆ : a₆ = r * a₅ := h₆
  have r_eq_three : r = 3 := hr
  rw [r_eq_three, ha₅] at ha₆
  show a₆ = 243
  rw [ha₆, mul_comm]
  norm_num
  exact rfl


end buttons_in_sixth_box_l804_804317


namespace positive_difference_l804_804670

theorem positive_difference (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * y - 3 * x = 10) : |y - x| = 12 := by
  sorry

end positive_difference_l804_804670


namespace ellipse_equation_line_equation_final_line_equation_l804_804467

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (C : ℝ → ℝ → Prop)
  (hC : ∀ x y : ℝ, C x y ↔ (x^2 / a^2 + y^2 / b^2 = 1))
  (F : ℝ × ℝ) (hF : F = (2, 0)) 
  (P : ℝ × ℝ) (hP : P = (2, sqrt 6 / 3))
  (hP_on_C : C P.1 P.2) : 
  a^2 = 6 ∧ b^2 = 2 := 
sorry

theorem line_equation (k : ℝ) (F : ℝ × ℝ) (hF : F = (2, 0))
  (C : ℝ → ℝ → Prop) (hC : ∀ x y : ℝ, C x y ↔ (x^2 / 6 + y^2 / 2 = 1))
  (A B M : ℝ × ℝ) (O : ℝ × ℝ) (hO : O = (0, 0)) 
  (h_line_ABM_centroid : O = ((A.1 + B.1 + M.1) / 3, (A.2 + B.2 + M.2) / 3))
  (hM_on_C : C M.1 M.2)
  (h_AB_on_ellipse : ∀ x : ℝ, ∃ y : ℝ, C x y → ∃ t : ℝ, y = k * (x - 2)) : 
  k^2 = 1/5 ∨ k^2 = -1/3 := 
sorry

theorem final_line_equation (k : ℝ) (h1 : k^2 = 1/5 ∨ k^2 = -1/3) : 
  k = sqrt 5 / 5 ∨ k = -sqrt 5 / 5 :=
sorry

end ellipse_equation_line_equation_final_line_equation_l804_804467


namespace erase_six_points_l804_804974

theorem erase_six_points (points : Finset Point) (h_card : points.card = 16) :
  ∃ subset_points : Finset Point, subset_points.card = 10 ∧ 
  ∀ (s : Finset Point), s.card = 4 → s ⊆ subset_points → ¬ is_square s :=
begin
  sorry
end

end erase_six_points_l804_804974


namespace greatest_k_l804_804002

noncomputable def n : ℕ := sorry
def k : ℕ := sorry

axiom d : ℕ → ℕ

axiom h1 : d n = 72
axiom h2 : d (5 * n) = 90

theorem greatest_k : ∃ k : ℕ, (∀ m : ℕ, m > k → ¬(5^m ∣ n)) ∧ 5^k ∣ n ∧ k = 3 :=
by
  sorry

end greatest_k_l804_804002


namespace limit_integral_eq_log_l804_804404

noncomputable def distance_to_nearest_int (x : ℝ) : ℝ :=
  abs (x - round x)

theorem limit_integral_eq_log
  (H : ∏ n in (Finset.range (n + 1)).filter (λ k, k ≠ 0), (2 * n : ℝ) / (2 * n - 1) * (2 * n / (2 * n + 1)) = π / 2) :
  (real.integrable_over_Icc (λ x, distance_to_nearest_int (n / x)) 1 n) →
  tendsto (λ n : ℕ, (1 / n : ℝ) * ∫ x in 1 .. (n : ℝ), distance_to_nearest_int (n / x)) at_top (𝓝 (real.log (4 / π))) :=
 sorry

end limit_integral_eq_log_l804_804404


namespace exists_four_integers_mod_5050_l804_804096

theorem exists_four_integers_mod_5050 (S : Finset ℕ) (hS_card : S.card = 101) (hS_bound : ∀ x ∈ S, x < 5050) : 
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (a + b - c - d) % 5050 = 0 :=
sorry

end exists_four_integers_mod_5050_l804_804096


namespace exists_subset_with_elements_l804_804083

theorem exists_subset_with_elements (m : ℕ) (h : m = 56) :
  ∀ (A : Finset ℕ), (∀ m, A = Finset.range (m + 1)) →
  ∃ (A_i : Finset (Finset ℕ)), ∃ (a b : ℕ), a ∈ A_i ∧ b ∈ A_i ∧ b < a ∧ a ≤ (4 * b + 2) / 3 :=
by 
  intro A hA 
  intro A_i
  use sorry

end exists_subset_with_elements_l804_804083


namespace neg_fraction_comparison_l804_804040

theorem neg_fraction_comparison : - (4 / 5 : ℝ) > - (5 / 6 : ℝ) :=
by {
  -- sorry to skip the proof
  sorry
}

end neg_fraction_comparison_l804_804040


namespace cannot_determine_a_l804_804519

theorem cannot_determine_a 
  (n : ℝ) 
  (p : ℝ) 
  (a : ℝ) 
  (line_eq : ∀ (x y : ℝ), x = 5 * y + 5) 
  (pt1 : a = 5 * n + 5) 
  (pt2 : a + 2 = 5 * (n + p) + 5) : p = 0.4 → ¬∀ a' : ℝ, a = a' :=
by
  sorry

end cannot_determine_a_l804_804519


namespace length_pr_of_circle_l804_804242

theorem length_pr_of_circle (P Q R : Point) (O : Point) (radius : ℝ) (d : ℝ) (S : Point)
  (h_circle : distance O P = radius ∧ distance O Q = radius)
  (h_pq : distance P Q = d)
  (h_midarc : R = midpoint (major_arc P Q))
  (h_radius : radius = 13)
  (h_pqd : d = 10)
  (h_midpoint_pq : S = midpoint (line_segment P Q))
  (h_perp : is_perpendicular (line_segment O R) (line_segment P Q) ∧ distance O S = 12 ∧ distance R S = 1) :
  distance P R = sqrt 26 := by
sorry

end length_pr_of_circle_l804_804242


namespace number_of_blocks_differing_in_2_ways_l804_804731

/-- Defining the properties of the blocks -/
inductive Material
| plastic | wood | metal

inductive Size
| small | medium | large

inductive Color
| blue | green | red | yellow | purple

inductive Shape
| circle | hexagon | square | triangle

/-- Definition of a block -/
structure Block :=
  (material : Material)
  (size : Size)
  (color : Color)
  (shape : Shape)

/-- Defining the specific block 'metal medium purple circle' -/
def target_block : Block :=
  { material := Material.metal,
    size := Size.medium,
    color := Color.purple,
    shape := Shape.circle }

/-- Predicate to check if a block differs in exactly 2 ways from the target block -/
def differs_in_exactly_two_ways (b1 b2 : Block) : Prop :=
  [b1.material ≠ b2.material,
   b1.size ≠ b2.size,
   b1.color ≠ b2.color,
   b1.shape ≠ b2.shape].count true = 2

/-- Theorem stating the number of blocks differing from the target block in exactly 2 ways is 40 -/
theorem number_of_blocks_differing_in_2_ways :
  (SetOf (λ b : Block, differs_in_exactly_two_ways b target_block)).card = 40 :=
  sorry -- Proof is omitted

end number_of_blocks_differing_in_2_ways_l804_804731


namespace smallest_root_floor_eq_3_l804_804402

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x - 3 * Real.cos x + Real.tan x

-- statement of the lean proof
theorem smallest_root_floor_eq_3 :
  let s := Inf {x : ℝ | x > 0 ∧ g x = 0} in ⌊s⌋ = 3 :=
by
  sorry

end smallest_root_floor_eq_3_l804_804402


namespace product_of_sums_of_squares_l804_804250

theorem product_of_sums_of_squares (a b : ℤ) 
  (h1 : ∃ x1 y1 : ℤ, a = x1^2 + y1^2)
  (h2 : ∃ x2 y2 : ℤ, b = x2^2 + y2^2) : 
  ∃ x y : ℤ, a * b = x^2 + y^2 :=
by
  sorry

end product_of_sums_of_squares_l804_804250


namespace exists_special_sequence_l804_804813

noncomputable def Euler_totient : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 1) := n.succ.gcd n.succ_nat.pred

def N (n : ℕ) : ℕ := (finset.range (n + 1)).sum (λ k, Euler_totient k)

theorem exists_special_sequence (n : ℕ) (N := N n) :
  ∃ a : fin n → ℕ, (∀ k ≤ n, (finset.univ.filter (λ i, a i = k)).card = Euler_totient k)
    ∧ (fin n).sum (λ i, 1 / (a i * a ((i + 1) % N))) = 1 :=
begin
  sorry
end

end exists_special_sequence_l804_804813


namespace sum_of_real_solutions_eq_32_over_7_l804_804433

theorem sum_of_real_solutions_eq_32_over_7 :
  (∑ x in (finset.filter (λ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x)) finset.univ), x) = 32 / 7 :=
by
  sorry

end sum_of_real_solutions_eq_32_over_7_l804_804433


namespace polynomial_nonnegative_l804_804619

theorem polynomial_nonnegative (x : ℝ) : x^4 - x^3 + 3x^2 - 2x + 2 ≥ 0 := 
by 
  sorry

end polynomial_nonnegative_l804_804619


namespace tree_growth_per_two_weeks_l804_804231

-- Definitions based on conditions
def initial_height_meters : ℕ := 2
def initial_height_centimeters : ℕ := initial_height_meters * 100
def final_height_centimeters : ℕ := 600
def total_growth : ℕ := final_height_centimeters - initial_height_centimeters
def weeks_in_4_months : ℕ := 16
def number_of_two_week_periods : ℕ := weeks_in_4_months / 2

-- Objective: Prove that the growth every two weeks is 50 centimeters
theorem tree_growth_per_two_weeks :
  (total_growth / number_of_two_week_periods) = 50 :=
  by
  sorry

end tree_growth_per_two_weeks_l804_804231


namespace rented_room_percentage_l804_804237

theorem rented_room_percentage (total_rooms : ℕ) (h1 : 3 * total_rooms / 4 = 3 * total_rooms / 4) 
                               (h2 : 3 * total_rooms / 5 = 3 * total_rooms / 5) 
                               (h3 : 2 * (3 * total_rooms / 5) / 3 = 2 * (3 * total_rooms / 5) / 3) :
  (1 * (3 * total_rooms / 5) / 5) / (1 * total_rooms / 4) * 100 = 80 := by
  sorry

end rented_room_percentage_l804_804237


namespace range_of_a_l804_804473

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if hx : 0 ≤ x ∧ x ≤ 3 then a - x else a * Real.log x / Real.log 2

theorem range_of_a (a : ℝ) (h : f a 2 < f a 4) : a > -2 :=
by
  have h₁ : f a 2 = a - 2 := by simp [f]
  have h₂ : f a 4 = a * 2 := by simp [f, Real.log, Real.log_two_eq_log_e]
  rw [h₁, h₂] at h
  linarith

end range_of_a_l804_804473


namespace segment_area_l804_804786

theorem segment_area (p : ℝ) : 
  let α := 120 / 180 * Real.pi in -- 120 degrees in radians
  let r := 3 * p / (2 * Real.pi + 3 * Real.sqrt 3) in
  let area_of_sector := Real.pi * r^2 * α / (2 * Real.pi) in
  let area_of_triangle := r^2 * Real.sin α / 2 in
  let area_of_segment := area_of_sector - area_of_triangle in
  area_of_segment = (3 * p^2 * (4 * Real.pi - 3 * Real.sqrt 3)) / (4 * (2 * Real.pi + 3 * Real.sqrt 3)^2) :=
by {
  sorry -- Proof is not required
}

end segment_area_l804_804786


namespace sum_first_ten_nice_l804_804784

def is_nice (n : ℕ) : Prop :=
  n > 1 ∧
  (∃ p q : ℕ, p ≠ q ∧ Prime p ∧ Prime q ∧ n = p * q) ∨
  (∃ p : ℕ, Prime p ∧ n = p^3) ∨
  (∃ p q : ℕ, p ≠ q ∧ Prime p ∧ Prime q ∧ n = p^2 * q ∧ (p^3 * q^2 = n))

theorem sum_first_ten_nice : (Finset.range (100)).filter is_nice |>.take 10 |>.sum = 182 :=
by {
  sorry
}

end sum_first_ten_nice_l804_804784


namespace log_sum_of_sequence_eq_51_l804_804830

theorem log_sum_of_sequence_eq_51 (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, log (a (n + 1)) / log 2 = 1 + log (a n) / log 2) 
  (h2 : a 1 + a 2 + a 3 + a 4 + a 5 = 2) : 
  log ((a 51 + a 52 + a 53 + a 54 + a 55)) / log 2 = 51 := 
sorry

end log_sum_of_sequence_eq_51_l804_804830


namespace unique_solution_l804_804443

-- Define the system of equations
def system_of_equations (m x y : ℝ) := 
  (m + 1) * x - y - 3 * m = 0 ∧ 4 * x + (m - 1) * y + 7 = 0

-- Define the determinant condition
def determinant_nonzero (m : ℝ) := m^2 + 3 ≠ 0

-- Theorem to prove there is exactly one solution
theorem unique_solution (m x y : ℝ) : 
  determinant_nonzero m → ∃! (x y : ℝ), system_of_equations m x y :=
by
  sorry

end unique_solution_l804_804443


namespace fifth_selected_ID_is_01_l804_804282

noncomputable def populationIDs : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

noncomputable def randomNumberTable : List (List ℕ) :=
  [[78, 16, 65, 72,  8, 2, 63, 14,  7, 2, 43, 69, 97, 28,  1, 98],
   [32,  4, 92, 34, 49, 35, 82,  0, 36, 23, 48, 69, 69, 38, 74, 81]]

noncomputable def selectedIDs (table : List (List ℕ)) : List ℕ :=
  [8, 2, 14, 7, 1]  -- Derived from the selection method

theorem fifth_selected_ID_is_01 : (selectedIDs randomNumberTable).get! 4 = 1 := by
  sorry

end fifth_selected_ID_is_01_l804_804282


namespace sin_identity_l804_804320

theorem sin_identity (α : ℝ) : 
  (sin (π + α) * sin ((4 / 3) * π + α) * sin ((2 / 3) * π + α) = (1 / 4) * sin (3 * α)) := 
  sorry

end sin_identity_l804_804320


namespace area_of_triangle_DEF_l804_804525

variables {D E F V U : Type}
variables [normed_group D] [normed_group E] [normed_group F] [inner_product_space ℝ D] [inner_product_space ℝ E] [inner_product_space ℝ F]
variables (DE DF DV FU : ℝ) (med_DE_DF : DE = DF) (medians_perpendicular : ⟪DV, FU⟫ = 0) (medians_length : DV = 15 ∧ FU = 15)

theorem area_of_triangle_DEF : ∃ (area : ℝ), area = 450 :=
sorry

end area_of_triangle_DEF_l804_804525


namespace simplified_expression_l804_804718

variable (x y : ℝ)

theorem simplified_expression (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 / 5) * Real.sqrt (x * y^2) / ((-4 / 15) * Real.sqrt (y / x)) * ((-5 / 6) * Real.sqrt (x^3 * y)) =
  (15 * x^2 * y * Real.sqrt x) / 8 :=
by
  sorry

end simplified_expression_l804_804718


namespace xiao_ming_shopping_l804_804975

theorem xiao_ming_shopping :
  ∃ x : ℕ, x ≤ 16 ∧ 6 * x ≤ 100 ∧ 100 - 6 * x = 28 :=
by
  -- Given that:
  -- 1. x is the same amount spent in each of the six stores.
  -- 2. Total money spent, 6 * x, must be less than or equal to 100.
  -- 3. We seek to prove that Xiao Ming has 28 yuan left.
  sorry

end xiao_ming_shopping_l804_804975


namespace int_solutions_prime_square_l804_804864

theorem int_solutions_prime_square (n : ℤ) (p : ℤ) (h : p = n^3 - n^2 - 5 * n + 2) :
  (∃ k : ℤ, p^2 = k*k ∧ (prime k ∨ prime (-k))) ↔ n = -1 ∨ n = -3 ∨ n = 0 ∨ n = 1 ∨ n = 3 :=
by 
  sorry

end int_solutions_prime_square_l804_804864


namespace multiply_millions_l804_804435

theorem multiply_millions :
  (5 * 10^6) * (8 * 10^6) = 40 * 10^12 :=
by 
  sorry

end multiply_millions_l804_804435


namespace domain_of_f_f_is_odd_f_greater_than_0_l804_804820

noncomputable def f (x : ℝ) := Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

-- Problem 1: Prove the domain of f(x) is (-1, 1)
theorem domain_of_f : {x : ℝ | 1 - x > 0 ∧ 1 + x > 0} = set.Ioo (-1 : ℝ) 1 := by
  sorry

-- Problem 2: Prove that f(x) is an odd function
theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) := by
  sorry

-- Problem 3: Prove that f(x) > 0 for x in (-1, 0)
theorem f_greater_than_0 : ∀ x : ℝ, x ∈ set.Ioo (-1 : ℝ) 0 → f(x) > 0 := by
  sorry

end domain_of_f_f_is_odd_f_greater_than_0_l804_804820


namespace find_fruit_cost_l804_804444

-- Define the conditions
def muffin_cost : ℝ := 2
def francis_muffin_count : ℕ := 2
def francis_fruit_count : ℕ := 2
def kiera_muffin_count : ℕ := 2
def kiera_fruit_count : ℕ := 1
def total_cost : ℝ := 17

-- Define the cost of each fruit cup
variable (F : ℝ)

-- The statement to be proved
theorem find_fruit_cost (h : francis_muffin_count * muffin_cost 
                + francis_fruit_count * F 
                + kiera_muffin_count * muffin_cost 
                + kiera_fruit_count * F = total_cost) : 
                F = 1.80 :=
by {
  sorry
}

end find_fruit_cost_l804_804444


namespace grid_tinting_problem_l804_804818

theorem grid_tinting_problem :
  let M := (number of valid ways to tint a 2018 × 4 grid with each row and each column having equal number of red and blue grids)
  M % 2018 = 6 :=
sorry

end grid_tinting_problem_l804_804818


namespace staff_members_attended_meeting_l804_804409

theorem staff_members_attended_meeting
  (n_doughnuts_served : ℕ)
  (e_each_staff_member : ℕ)
  (n_doughnuts_left : ℕ)
  (h1 : n_doughnuts_served = 50)
  (h2 : e_each_staff_member = 2)
  (h3 : n_doughnuts_left = 12) :
  (n_doughnuts_served - n_doughnuts_left) / e_each_staff_member = 19 := 
by
  sorry

end staff_members_attended_meeting_l804_804409


namespace stored_energy_in_doubled_square_l804_804651

noncomputable def energy (q : ℝ) (d : ℝ) : ℝ := q^2 / d

theorem stored_energy_in_doubled_square (q d : ℝ) (h : energy q d * 4 = 20) :
  energy q (2 * d) * 4 = 10 := by
  -- Add steps: Show that energy proportional to 1/d means energy at 2d is half compared to at d
  sorry

end stored_energy_in_doubled_square_l804_804651


namespace cyclic_sum_fraction_ge_one_l804_804960

theorem cyclic_sum_fraction_ge_one (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (hineq : (a/(b+c+1) + b/(c+a+1) + c/(a+b+1)) ≤ 1) :
  (1/(b+c+1) + 1/(c+a+1) + 1/(a+b+1)) ≥ 1 :=
by sorry

end cyclic_sum_fraction_ge_one_l804_804960


namespace eraser_cost_l804_804374

variable (P E : ℝ)
variable (h1 : E = P / 2)
variable (h2 : 20 * P = 80)

theorem eraser_cost : E = 2 := by 
  sorry

end eraser_cost_l804_804374


namespace find_f_pi_plus_3_l804_804494

variables (a b : ℝ)

def f (x : ℝ) : ℝ := a * sin (2 * x) + b * tan x + 1

theorem find_f_pi_plus_3 (h : f a b (-3) = 5) : f a b (Real.pi + 3) = -3 :=
by sorry

end find_f_pi_plus_3_l804_804494


namespace unique_solution_l804_804799

theorem unique_solution (x : ℝ) (hx : 4 ≤ x) : sqrt (x + 2 - 2 * sqrt (x - 4)) + sqrt (x + 12 - 8 * sqrt (x - 4)) = 4 ↔ x = 13 :=
sorry

end unique_solution_l804_804799


namespace T_shaped_perimeter_l804_804637

/-- Consider a "T" shaped figure composed of four squares, each with a side length of 2.
    We need to prove that the perimeter of this figure is 18. -/
theorem T_shaped_perimeter : 
  let side_len := 2
  let squares_count := 4
  let horizontal_squares := 3
  let vertical_square := 1
  let horizontal_perimeter := horizontal_squares * side_len * 2 + side_len * 2
  let vertical_perimeter :=  side_len * 2 + side_len * 2
  in horizontal_perimeter + vertical_perimeter = 18 := by
  sorry

end T_shaped_perimeter_l804_804637


namespace determine_parallel_l804_804815

variables (α β : Plane) [non_coincident_planes : ¬(α = β)]
variables (l : Line) (γ : Plane)
variables (points : Finset Point)

-- Definition of the given conditions
def condition1 : Prop := l ⟂ α ∧ l ⟂ β
def condition2 : Prop := α ⟂ γ ∧ β ⟂ γ
def condition3 : Prop := (points.card = 3 ∧ ¬ collinear α points ∧ ∀ p ∈ points, distance p β = distance (p + 1) β)
def condition4 : Prop := l.parallel α ∧ l.parallel β ∧ skew_lines l γ ∧ skew_lines l β

-- Proof that proves condition2 determines α is parallel to β
theorem determine_parallel : condition2 α β γ → α.parallel β :=
by
  sorry

end determine_parallel_l804_804815


namespace product_of_chords_l804_804204

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.i / 18)

theorem product_of_chords :
  let A := 3
  let B := -3
  let D := λ k: ℕ, 3 * omega^k
  (∏ k in Finset.range 8, Complex.abs (A - D k)) *
  (∏ k in Finset.range 8, Complex.abs (B - D k)) = 4782969 :=
by
  let A := 3
  let B := -3
  let D := λ k: ℕ, 3 * omega^k
  have omega_power_9 : omega^9 = -1 := by
    have omega_power_18 : omega^18 = 1 := by
      sorry
    sorry
  have product_of_first_8_AD := ∏ k in Finset.range 8, Complex.abs (A - D k) = 3^8 * Complex.abs (Finset.range.map (λ k: ℕ, 1 - omega^k).prod) := by
    sorry
  have product_of_first_8_BD := ∏ k in Finset.range 8, Complex.abs (B - D k) = 3^8 * Complex.abs ((Finset.range.map (λ k: ℕ, 1 - omega^(k+9))).prod) := by
    sorry
  have products_multiplied : 3^16 * (Complex.abs (Finset.range.map (λ k: ℕ, 1 - omega^k).prod)) * (Complex.abs ((Finset.range.map (λ k: ℕ, 1 - omega^(k+9))).prod)) = 9 * 3^16 := by
    sorry
  exact products_multiplied

end product_of_chords_l804_804204


namespace find_number_of_rabbits_l804_804335

variable (R P : ℕ)

theorem find_number_of_rabbits (h1 : R + P = 60) (h2 : 4 * R + 2 * P = 192) : R = 36 := 
by
  sorry

end find_number_of_rabbits_l804_804335


namespace perimeter_of_triangle_OAB_l804_804131

noncomputable
def triangle_perimeter (z1 z2 : ℂ) (h1 : complex.abs z1 = 2) (h2 : z1^2 - 2 * z1 * z2 + 4 * z2^2 = 0) : ℝ :=
  3 + real.sqrt 3

theorem perimeter_of_triangle_OAB (z1 z2 : ℂ) (h1 : complex.abs z1 = 2) (h2 : z1^2 - 2 * z1 * z2 + 4 * z2^2 = 0) :
  triangle_perimeter z1 z2 h1 h2 = 3 + real.sqrt 3 :=
sorry

end perimeter_of_triangle_OAB_l804_804131


namespace find_m_l804_804098

-- Definitions derived from conditions
def z (m : ℝ) := complex.mk m 1  -- complex number z = m + i
def z_conjugate (m : ℝ) := complex.mk m (-1)  -- complex conjugate z = m - i

-- Condition that m ≥ 0
def m_ge_0 (m : ℝ) := m ≥ 0

-- Condition that the magnitude of the conjugate is sqrt(2)
def magnitude_z_conjugate (m : ℝ) := complex.abs (z_conjugate m) = real.sqrt 2

-- The main theorem to prove
theorem find_m (m : ℝ) (h1 : m_ge_0 m) (h2 : magnitude_z_conjugate m) : m = 1 :=
sorry

end find_m_l804_804098


namespace monotonicity_of_f_extreme_points_of_f_log_inequality_l804_804964

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * real.log (x + 1)

theorem monotonicity_of_f (b : ℝ) (hb : b > 1/2) : monotone_on (f b) (set.Ioi (-1)) :=
sorry

theorem extreme_points_of_f (b : ℝ) (hb : b ≠ 0) : 
  (b >= 1/2 ∧ ∀ x ∈ set.Ioi (-1), ¬(is_maximum (f b) x ∨ is_minimum (f b) x)) ∨
  (b < 0 ∧ ∃ x ∈ set.Ioi (-1), is_minimum (f b) x) ∨
  (0 < b ∧ b < 1/2 ∧ ∃ x ∈ set.Ioi (-1), is_maximum (f b) x ∧ ∃ y ∈ set.Ioi (-1), is_minimum (f b) y ) :=
sorry

theorem log_inequality (n : ℕ) (hn : 0 < n) : 
  real.log ((1 : ℝ) / n + 1) > (1 : ℝ) / (n^2) - (1 : ℝ) / (n^3) :=
sorry

end monotonicity_of_f_extreme_points_of_f_log_inequality_l804_804964


namespace theta_in_fourth_quadrant_l804_804487

open Real

theorem theta_in_fourth_quadrant (θ : ℝ) 
  (h1 : sin θ < cos θ) 
  (h2 : sin θ * cos θ < 0) : 
  (θ ∈ set.Ioo (3 * π / 2) (2 * π)) :=
by
  sorry

end theta_in_fourth_quadrant_l804_804487


namespace range_of_a_l804_804223

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f(-x) = -f(x)) →
  (∀ x : ℝ, 0 ≤ x → f(x) = x^2) →
  (∀ x : ℝ, a ≤ x ∧ x ≤ a + 2 → f(x + a) ≥ 2 * f(x)) →
  a ≥ real.sqrt 2 :=
by
  intros f_odd f_def f_ineq
  sorry

end range_of_a_l804_804223


namespace equal_diagonals_of_quadrilateral_l804_804658

-- Definitions based on conditions
structure ConvexQuadrilateral (A B C D : Type) :=
  (convex : Prop)

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

axiom equal_angles (P Q R S : ℝ × ℝ) (L midpoint P Q) (N midpoint R S) :
  ℕ → ℕ → Prop -- Placeholder for angle equality condition

theorem equal_diagonals_of_quadrilateral (A B C D : ℝ × ℝ) (quad : ConvexQuadrilateral A B C D)
  (L : ℝ × ℝ := midpoint B C) (N : ℝ × ℝ := midpoint D A)
  (angle_condition : equal_angles A C B D L N) :
  (dist A C) = (dist B D) :=
sorry

end equal_diagonals_of_quadrilateral_l804_804658


namespace A_sub_subneq_B_l804_804477

set_option pp.generalizedFieldNotation false

theorem A_sub_subneq_B : 
  (∃ a : ℝ, a = 3 ∧ 
  (∃ B : Set ℤ, B = {x | (|x| < a) ∧ x ∈ ℤ} ∧ 
  ∀ x, x ∈ A → x ∈ B ∧ A ⊂ B)) :=
sorry

end A_sub_subneq_B_l804_804477


namespace right_triangle_perimeter_l804_804340

theorem right_triangle_perimeter 
  (a b c : ℕ) (h : a = 11) (h1 : a * a + b * b = c * c) (h2 : a < c) : a + b + c = 132 :=
  sorry

end right_triangle_perimeter_l804_804340


namespace simplification_tangent_sine_cosine_l804_804251

theorem simplification_tangent_sine_cosine (α : ℝ) :
  (tan (real.pi / 4 - α)) / (1 - (tan (real.pi / 4 - α))^2) * (sin α * cos α) / (cos α ^ 2 - sin α ^ 2) = 1 / 4 :=
by sorry

end simplification_tangent_sine_cosine_l804_804251


namespace right_triangle_LN_length_l804_804257

theorem right_triangle_LN_length 
  (LM : ℝ) (LN : ℝ) 
  (h1 : LM = 15)
  (h2 : real.sin (real.atan (LM / LN)) = 3 / 5) : 
  LN = 25 := 
by
  sorry

end right_triangle_LN_length_l804_804257


namespace correct_statement_exam_l804_804906

theorem correct_statement_exam 
  (students_participated : ℕ)
  (students_sampled : ℕ)
  (statement1 : Bool)
  (statement2 : Bool)
  (statement3 : Bool)
  (statement4 : Bool)
  (cond1 : students_participated = 70000)
  (cond2 : students_sampled = 1000)
  (cond3 : statement1 = False)
  (cond4 : statement2 = False)
  (cond5 : statement3 = False)
  (cond6 : statement4 = True) :
  statement4 = True := 
sorry

end correct_statement_exam_l804_804906


namespace exist_triangle_with_points_on_sides_l804_804336

open Set

def non_collinear {α : Type*} [LinearOrderedField α] (s : Set (α × α)) :=
  ∀ (p1 p2 p3 : α × α), p1 ∈ s → p2 ∈ s → p3 ∈ s → ¬Collinear ℝ {p1, p2, p3}

theorem exist_triangle_with_points_on_sides {α : Type*} [LinearOrderedField α] 
  (s : Set (α × α)) (n : ℕ) (h_points : n ≥ 4) (h_size : s.size = n) 
  (h_non_collinear : non_collinear s) :
  ∃ (A B C : α × α), A ∈ s ∧ B ∈ s ∧ C ∈ s ∧ 
  (∀ x ∈ s, (x ≠ A ∧ x ≠ B ∧ x ≠ C) → 
  (x inside_triangle (A, B, C)) ∧ 
  (∃ y ∈ s, y ∈ line(A, B) ∧ 
  ∃ z ∈ s, z ∈ line(B, C) ∧ 
  ∃ w ∈ s, w ∈ line(C, A))) :=
sorry

end exist_triangle_with_points_on_sides_l804_804336


namespace midpoints_of_triangle_l804_804177

theorem midpoints_of_triangle
    (A B C A' B' C' : Type)
    [EuclideanGeometry A B C]
    (on_bc : A' ∈ segment B C)
    (on_ca : B' ∈ segment C A)
    (on_ab : C' ∈ segment A B)
    (angle_ACB_eq : ∠ (A, C, B') = ∠ (B', A', C))
    (angle_CBA_eq : ∠ (C, B', A') = ∠ (A', C', B))
    (angle_BAC_eq : ∠ (B, A', C') = ∠ (C', B', A)) :
    is_midpoint A' (B, C) ∧ is_midpoint B' (C, A) ∧ is_midpoint C' (A, B) := 
by
  sorry

end midpoints_of_triangle_l804_804177


namespace xyz_sum_eq_eleven_l804_804841

theorem xyz_sum_eq_eleven (x y z : ℝ) (h : (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0) : x + y + z = 11 :=
sorry

end xyz_sum_eq_eleven_l804_804841


namespace perfect_square_factors_of_product_l804_804876

open Nat

/-- Counting the number of positive perfect square factors of the product -/
theorem perfect_square_factors_of_product :
  let prod := (2 ^ 12) * (3 ^ 10) * (5 ^ 18)
  ∃ f : ℕ, (∃ n, f = 2 ^ n) ∨ (∃ n, f = 3 ^ n) ∨ (∃ n, f = 5 ^ n) ∧
  (f ≤ prod) ∧
  (∃ n, is_square f) :=
    (cardinality_of_set_of_positive_perfect_square_factors prod) = 420 := sorry

end perfect_square_factors_of_product_l804_804876


namespace apples_per_box_l804_804561

variable (A : ℕ) -- Number of apples packed in a box

-- Conditions
def normal_boxes_per_day := 50
def days_per_week := 7
def boxes_first_week := normal_boxes_per_day * days_per_week * A
def boxes_second_week := (normal_boxes_per_day * A - 500) * days_per_week
def total_apples := 24500

-- Theorem
theorem apples_per_box : boxes_first_week + boxes_second_week = total_apples → A = 40 :=
by
  sorry

end apples_per_box_l804_804561


namespace not_possible_to_tile_l804_804771

theorem not_possible_to_tile 
    (m n : ℕ) (a b : ℕ)
    (h_m : m = 2018)
    (h_n : n = 2020)
    (h_a : a = 5)
    (h_b : b = 8) :
    ¬ ∃ k : ℕ, k * (a * b) = m * n := by
sorry

end not_possible_to_tile_l804_804771


namespace white_area_proof_l804_804671

def total_area_board : ℕ := 6 * 16
def area_m : ℕ := 2 * (6 * 1) + 2 * 3
def area_a : ℕ := 1 * (6 * 1) + 2 * (1 * 2) - 1
def area_t : ℕ := 1 * (5 * 1) + 1 * (6 * 1)
def area_h : ℕ := 2 * (6 * 1) + 1 * 3
def total_black_area : ℕ := area_m + area_a + area_t + area_h
def area_white : ℕ := total_area_board - total_black_area

theorem white_area_proof : area_white = 43 := by
  -- definitions
  have h_m : area_m = 18 := rfl
  have h_a : area_a = 9  := rfl
  have h_t : area_t = 11 := rfl
  have h_h : area_h = 15 := rfl
  have h_black : total_black_area = area_m + area_a + area_t + area_h := rfl
  have h_total_black : total_black_area = 18 + 9 + 11 + 15 := by deco.color:,dec_trailing_commas: [preserve]
  have h_total_black := by rw [h_m, h_a, h_t, h_h] at total_black_area: rfl
  have h_white : area_white = total_area_board - total_black_area := rfl
  have h_total_area : total_area_board = 96 := rfl
  have h_white_calculated : area_white = 96 - 53 := by
    rw [h_total_area, h_total_black] at h_white: rfl
  show area_white = 43 from rfl

#eval white_area_proof -- Expected output: true

end white_area_proof_l804_804671


namespace joe_list_combinations_l804_804722

theorem joe_list_combinations : 
  let total_draws := 15^4,
      ways_to_choose_3_from_4 := Nat.choose 4 3
  in total_draws * ways_to_choose_3_from_4 = 202500 :=
by
  let total_draws := 15^4
  let ways_to_choose_3_from_4 := Nat.choose 4 3
  show total_draws * ways_to_choose_3_from_4 = 202500 from
  sorry

end joe_list_combinations_l804_804722


namespace expectation_of_X_is_15_max_P_a_le_X_le_b_additional_workers_needed_l804_804728

noncomputable def repair_data := [9, 15, 12, 18, 12, 18, 9, 9, 24, 12, 12, 24, 15, 15, 15, 12, 15, 15, 15, 24]

def X_distribution_table : List (ℚ × ℚ) :=
[(9, 3/20), (12, 5/20), (15, 7/20), (18, 2/20), (24, 3/20)]

def expectation_X : ℚ :=
9 * (3/20) + 12 * (5/20) + 15 * (7/20) + 18 * (2/20) + 24 * (3/20)

theorem expectation_of_X_is_15 : expectation_X = 15 :=
by
  sorry

def P_a_le_X_le_b (a b : ℕ) : ℚ :=
if (a, b) = (9, 15) then (3/20) + (5/20) + (7/20)
else if (a, b) = (12, 18) then (5/20) + (7/20) + (2/20)
else if (a, b) = (18, 24) then (2/20) + (3/20)
else 0

theorem max_P_a_le_X_le_b : 
  ∃ a b : ℕ, b - a = 6 ∧ P_a_le_X_le_b a b = 3/4 :=
by
  use 9, 15
  split
  . rfl
  . sorry

def num_of_workers_needed (E : ℚ) (max_per_worker : ℚ) : ℚ :=
E / max_per_worker

theorem additional_workers_needed : 
  num_of_workers_needed 15 4 - 2 = 2 :=
by
  sorry

end expectation_of_X_is_15_max_P_a_le_X_le_b_additional_workers_needed_l804_804728


namespace total_jokes_l804_804935

theorem total_jokes (jessy_jokes_saturday : ℕ) (alan_jokes_saturday : ℕ) 
  (jessy_next_saturday : ℕ) (alan_next_saturday : ℕ) (total_jokes_so_far : ℕ) :
  jessy_jokes_saturday = 11 → 
  alan_jokes_saturday = 7 → 
  jessy_next_saturday = 11 * 2 → 
  alan_next_saturday = 7 * 2 → 
  total_jokes_so_far = (jessy_jokes_saturday + alan_jokes_saturday) + (jessy_next_saturday + alan_next_saturday) → 
  total_jokes_so_far = 54 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end total_jokes_l804_804935


namespace polynomial_roots_sum_l804_804214

theorem polynomial_roots_sum :
  ∀ p q : ℂ, (p^2 - 5 * p + 6 = 0 ∧ q^2 - 5 * q + 6 = 0) -> (p^3 + p^5 * q + p * q^5 + q^3 = 617) := 
by
  intros p q h
  cases h with hp hq
  sorry

end polynomial_roots_sum_l804_804214


namespace max_single_player_salary_l804_804725

variable (team_size : ℕ) (min_salary total_salary : ℝ)

theorem max_single_player_salary :
  team_size = 25 →
  min_salary = 18000 →
  total_salary = 900000 →
  ∃ x : ℝ, 24 * min_salary + x = total_salary ∧ x = 468000 :=
by
  intros h1 h2 h3
  exists 468000
  construct goal with:
  - equation: 24 * min_salary + 468000 = total_salary
  - verification: algebra on equation shows 468000 is correct
  sorry

end max_single_player_salary_l804_804725


namespace replaced_person_weight_l804_804645

theorem replaced_person_weight :
  ∀ (old_avg_weight new_person_weight incr_weight : ℕ),
    old_avg_weight * 8 + incr_weight = new_person_weight →
    incr_weight = 16 →
    new_person_weight = 81 →
    (old_avg_weight - (new_person_weight - incr_weight) / 8) = 65 :=
by
  intros old_avg_weight new_person_weight incr_weight h1 h2 h3
  -- TODO: Proof goes here
  sorry

end replaced_person_weight_l804_804645


namespace correct_answer_l804_804456

def proposition_p : Prop := ∃ x : ℝ, Math.sin x < 1

def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem correct_answer : proposition_p ∧ proposition_q :=
by
  sorry

end correct_answer_l804_804456


namespace find_x_for_parallel_vectors_l804_804871

-- Definitions for the given conditions
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- The proof statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : parallel a (b x)) : x = 6 :=
  sorry

end find_x_for_parallel_vectors_l804_804871


namespace rahul_share_of_payment_l804_804621

def work_rate_rahul : ℚ := 1 / 3
def work_rate_rajesh : ℚ := 1 / 2
def total_payment : ℚ := 150

theorem rahul_share_of_payment : (work_rate_rahul / (work_rate_rahul + work_rate_rajesh)) * total_payment = 60 := by
  sorry

end rahul_share_of_payment_l804_804621


namespace mathematicians_and_physicists_arrangement_l804_804296

def num_ways_arrange (num_m num_p : Nat) (rows : Array (Array String)) : Nat :=
  sorry  -- This function intentionally leaves out the computational details.

theorem mathematicians_and_physicists_arrangement :
  let num_m := 3
  let num_p := 4
  let front_row := ["P", "P", "P"]  -- There will be only one physicist in one of the rows
  let back_row := ["M", "M", "M", "P"]
  let rows := #[front_row, back_row]
  num_ways_arrange num_m num_p rows = 432 :=
begin
  sorry  -- The proof goes here, which we are skipping as specified.
end

end mathematicians_and_physicists_arrangement_l804_804296


namespace possible_numbers_l804_804603

theorem possible_numbers (N : ℕ) (h_digit : 10 ≤ N ∧ N ≤ 99)
  (h_claimed_divisors : ∀ d ∈ [3, 4, 5, 9, 10, 15, 18, 30], d ∣ N ∨ ¬ d ∣ N)
  (h_mistakes : Nat.countp (λ d, ¬ d ∣ N) [3, 4, 5, 9, 10, 15, 18, 30] = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 :=
by
  sorry

end possible_numbers_l804_804603


namespace possible_numbers_l804_804604

theorem possible_numbers (N : ℕ) (h_digit : 10 ≤ N ∧ N ≤ 99)
  (h_claimed_divisors : ∀ d ∈ [3, 4, 5, 9, 10, 15, 18, 30], d ∣ N ∨ ¬ d ∣ N)
  (h_mistakes : Nat.countp (λ d, ¬ d ∣ N) [3, 4, 5, 9, 10, 15, 18, 30] = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 :=
by
  sorry

end possible_numbers_l804_804604


namespace possible_values_f_l804_804948

def f (n : ℕ) (x : ℕ → ℤ) : ℚ := (∑ k in Finset.range n, x (k + 1)) / n

def x_k (k : ℕ) : ℤ := (-1)^(k + 1)

theorem possible_values_f :
  ∀ n : ℕ, n > 0 → (f n x_k = 0 ∨ f n x_k = 1 / n) :=
begin
  sorry
end

end possible_values_f_l804_804948


namespace sin_b_in_triangle_l804_804500

theorem sin_b_in_triangle (a b : ℝ) (sin_A sin_B : ℝ) (h₁ : a = 2) (h₂ : b = 1) (h₃ : sin_A = 1 / 3) 
  (h₄ : sin_B = (b * sin_A) / a) : sin_B = 1 / 6 :=
by
  have h₅ : sin_B = 1 / 6 := by 
    sorry
  exact h₅

end sin_b_in_triangle_l804_804500


namespace positive_difference_balances_l804_804773

noncomputable def cedric_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

noncomputable def daniel_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem positive_difference_balances :
  let P : ℝ := 15000
  let r_cedric : ℝ := 0.06
  let r_daniel : ℝ := 0.08
  let t : ℕ := 15
  let A_cedric := cedric_balance P r_cedric t
  let A_daniel := daniel_balance P r_daniel t
  (A_daniel - A_cedric) = 11632.65 :=
by
  sorry

end positive_difference_balances_l804_804773


namespace sequence_general_term_l804_804831

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n + 1) : a n = n * n :=
by
  sorry

end sequence_general_term_l804_804831


namespace max_uncovered_sections_l804_804908

open Classical

noncomputable def corridor_length : ℝ := 100
noncomputable def num_carpet_strips : ℕ := 20
noncomputable def total_carpet_length : ℝ := 1000

theorem max_uncovered_sections : ∀ (corridor_length : ℝ) (num_carpet_strips : ℕ) (total_carpet_length : ℝ),
  corridor_length = 100 → 
  num_carpet_strips = 20 → 
  total_carpet_length = 1000 → 
  (∃ (uncovered_sections : ℕ), uncovered_sections = 11) :=
by
  intros
  exists 11
  sorry

end max_uncovered_sections_l804_804908


namespace area_increase_approx_l804_804724

noncomputable def rectangle_dimensions := (60 : ℝ, 20 : ℝ)
noncomputable def rectangle_area := (rectangle_dimensions.1 * rectangle_dimensions.2)
noncomputable def rectangle_perimeter := 2 * (rectangle_dimensions.1 + rectangle_dimensions.2)
noncomputable def circle_circumference := rectangle_perimeter
noncomputable def circle_radius := circle_circumference / (2 * Real.pi)
noncomputable def circle_area := Real.pi * (circle_radius ^ 2)

def increase_in_area := circle_area - rectangle_area

theorem area_increase_approx : increase_in_area ≈ 838.54 := by
  sorry

end area_increase_approx_l804_804724


namespace intersection_P_M_l804_804206

open Set Int

def P : Set ℤ := {x | 0 ≤ x ∧ x < 3}

def M : Set ℤ := {x | x^2 ≤ 9}

theorem intersection_P_M : P ∩ M = {0, 1, 2} := by
  sorry

end intersection_P_M_l804_804206


namespace sin_870_eq_half_l804_804044

theorem sin_870_eq_half : Real.sin (870 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_870_eq_half_l804_804044


namespace correct_conclusions_count_l804_804063

open Real

noncomputable def proposition1 (p q : Prop) : Prop :=
  ¬ (p ∧ q) → (¬ p ∧ ¬ q)

noncomputable def proposition2 : Prop :=
  ¬ (∀ x : ℝ, sin x ≤ 1) ↔ ∃ x_0 : ℝ, sin x_0 > 1

noncomputable def proposition3 : Prop :=
  ∀ x : ℝ, (tan x = 1 → x = π / 4) ∧ (x = π / 4 → tan x = 1)

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - (f x)

noncomputable def proposition4 (f : ℝ → ℝ) (h_odd : odd_function f) : Prop :=
  f (log 3 2) + f (log 2 3) = 0

theorem correct_conclusions_count 
  (p q : Prop) (f : ℝ → ℝ) (h_odd : odd_function f) :
  (¬ (p ∧ q) → (¬ p ∧ ¬ q) = false) →
  (¬ (∀ x : ℝ, sin x ≤ 1) ↔ ∃ x_0 : ℝ, sin x_0 > 1) →
  (∀ x : ℝ, (tan x = 1 → x = π / 4) ∧ (x = π / 4 → tan x = 1) = false) →
  (f (log 3 2) + f (log 2 3) = 0 = false) →
  1 :=
by
  intros h1 h2 h3 h4
  -- Proof skipped
  sorry

end correct_conclusions_count_l804_804063


namespace sin_870_correct_l804_804045

noncomputable def sin_870_eq_half : Prop :=
  sin (870 : ℝ) = 1 / 2

theorem sin_870_correct : sin_870_eq_half :=
by
  sorry

end sin_870_correct_l804_804045


namespace extremum_at_one_l804_804471

noncomputable def f (a x : ℝ) : ℝ := a * x - (2 * a + 1) * real.log x - 2 / x

theorem extremum_at_one (a : ℝ) : 
  (∃ y : ℝ, f a y = 0) ↔ a = 1 :=
by sorry

end extremum_at_one_l804_804471


namespace overall_speed_l804_804545

-- Definitions
variables (d : ℝ) (t1 t2 T : ℝ) (AB BC : ℝ)

-- Conditions
def AB_eq_BC : AB = d := sorry
def BC_eq_d : BC = d := sorry
def t1_def : t1 = d / 40 := sorry
def t2_def : t2 = d / 60 := sorry
def T_def : T = t1 + t2 := sorry
def total_distance : ℝ := 2 * d

-- Goal
theorem overall_speed (h_AB : AB = d) (h_BC : BC = d) (h_t1 : t1 = d / 40) (h_t2 : t2 = d / 60) (h_T : T = t1 + t2) : 
  total_distance / T = 48 :=
begin
  -- Define values based on conditions
  have h_total_distance := total_distance,
  calc
    total_distance / T = (2 * d) / (t1 + t2) : by rw [h_AB, h_BC, h_t1, h_t2, h_T]
                   ... = (2 * d) / (d / 40 + d / 60) : by rw [h_t1, h_t2]
                   ... = -- Simplification steps here
                   ... = 48 : sorry,
end

end overall_speed_l804_804545


namespace sqrt_sum_l804_804872

-- Define the conditions and the statement of the theorem
theorem sqrt_sum {x : ℝ} (h : sqrt (49 - x^2) - sqrt (25 - x^2) = 3) : 
  sqrt (49 - x^2) + sqrt (25 - x^2) = 8 := 
  sorry -- The proof is omitted

end sqrt_sum_l804_804872


namespace polynomial_roots_l804_804080

noncomputable def f (x : ℝ) : ℝ := 8 * x^4 + 28 * x^3 - 74 * x^2 - 8 * x + 48

theorem polynomial_roots:
  ∃ (a b c d : ℝ), a = -3 ∧ b = -1 ∧ c = -1 ∧ d = 2 ∧ 
  (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) :=
sorry

end polynomial_roots_l804_804080


namespace leila_average_speed_l804_804408

theorem leila_average_speed :
  let initial_reading := 12321
  let final_reading := 12421 
  let time_period := 3
  let distance_traveled := final_reading - initial_reading
  let average_speed := (distance_traveled:ℝ) / (time_period:ℝ)
  average_speed = 33.33 :=
begin
  let initial_reading := 12321,
  let final_reading := 12421,
  let time_period := 3,
  let distance_traveled := final_reading - initial_reading,
  have h_distance : distance_traveled = 100, by refl,
  let average_speed : ℝ := distance_traveled / time_period,
  have h_avg_speed : average_speed = 100 / 3, by rw h_distance,
  norm_num at h_avg_speed,
  -- Here we skip additional steps and use a known mathematical fact
  sorry,
end

end leila_average_speed_l804_804408


namespace num_four_place_decimals_near_fifth_l804_804262

theorem num_four_place_decimals_near_fifth : 
  let lower_bound := 0.1834
  let upper_bound := 0.2249 in
  let count := (upper_bound * 10000).floor - (lower_bound * 10000).ceil + 1 in
  count = 416 :=
by
  sorry

end num_four_place_decimals_near_fifth_l804_804262


namespace tiffany_math_homework_pages_l804_804682

def math_problems (m : ℕ) : ℕ := 3 * m
def reading_problems : ℕ := 4 * 3
def total_problems (m : ℕ) : ℕ := math_problems m + reading_problems

theorem tiffany_math_homework_pages (m : ℕ) (h : total_problems m = 30) : m = 6 :=
by
  sorry

end tiffany_math_homework_pages_l804_804682


namespace neznika_number_l804_804578

theorem neznika_number (N : ℕ) :
  10 ≤ N ∧ N ≤ 99 ∧
  (divisible_by N 3 ∧ divisible_by N 4 ∧ divisible_by N 5 ∧
   divisible_by N 9 ∧ divisible_by N 10 ∧ divisible_by N 15 ∧
   divisible_by N 18 ∧ divisible_by N 30) →
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end neznika_number_l804_804578


namespace prove_b_minus_a_l804_804281

noncomputable def point := (ℝ × ℝ)

def rotate90 (p : point) (c : point) : point :=
  let (x, y) := p
  let (h, k) := c
  (h - (y - k), k + (x - h))

def reflect_y_eq_x (p : point) : point :=
  let (x, y) := p
  (y, x)

def transformed_point (a b : ℝ) : point :=
  reflect_y_eq_x (rotate90 (a, b) (2, 6))

theorem prove_b_minus_a (a b : ℝ) (h1 : transformed_point a b = (-7, 4)) : b - a = 15 :=
by
  sorry

end prove_b_minus_a_l804_804281


namespace count_multiples_12_9_l804_804152

theorem count_multiples_12_9 :
  ∃ n : ℕ, n = 8 ∧ (∀ x : ℕ, x % 36 = 0 ∧ 200 ≤ x ∧ x ≤ 500 ↔ ∃ y : ℕ, (x = 36 * y ∧ 200 ≤ 36 * y ∧ 36 * y ≤ 500)) :=
by
  sorry

end count_multiples_12_9_l804_804152


namespace not_perfect_square_l804_804772

theorem not_perfect_square (n : ℕ) (h₁ : 100 + 200 = 300) (h₂ : ¬(300 % 9 = 0)) : ¬(∃ m : ℕ, n = m * m) :=
by
  intros
  sorry

end not_perfect_square_l804_804772


namespace integral_equivalence_l804_804781

variable (f : ℝ → ℝ → ℝ)

noncomputable def question_integral : ℝ :=
  ∫ (x : ℝ) in - (Real.pi / 2)..(Real.pi / 2), ∫ (y : ℝ) in (x / 2 - Real.pi / 4)..(Real.cos (x / 2)), f x y

noncomputable def answer_integral : ℝ :=
  ∫ (y : ℝ) in - (Real.pi / 4)..0, ∫ (x : ℝ) in - (Real.pi / 2)..(2 * y + Real.pi / 2), f x y +
  ∫ (y : ℝ) in 0..(Real.sqrt 2 / 2), ∫ (x : ℝ) in - (Real.pi / 2)..(Real.pi / 2), f x y +
  ∫ (y : ℝ) in (Real.sqrt 2 / 2)..1, ∫ (x : ℝ) in - (2 * Real.arccos y)..(2 * Real.arccos y), f x y

theorem integral_equivalence : question_integral f = answer_integral f :=
by
  sorry

end integral_equivalence_l804_804781


namespace count_Z_inter_complement_A_l804_804463

def U : Set ℝ := Set.univ
def Z : Set ℤ := Set.univ
def A : Set ℝ := {x : ℝ | x^2 - x - 6 ≥ 0}
def Z_int : Set ℝ := {n : ℝ | ∃ (m : ℤ), n = m}
def complement_A : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def intersection_set : Set ℝ := {x : ℝ | -2 < x ∧ x < 3 ∧ ∃ (m : ℤ), x = m}

theorem count_Z_inter_complement_A : Fintype.card (intersection_set) = 4 := by
  sorry

end count_Z_inter_complement_A_l804_804463


namespace coefficient_x_squared_in_expansion_l804_804261

theorem coefficient_x_squared_in_expansion :
  let x := x in 
  let expr := (sqrt x + (1 / (2 * sqrt x))) ^ 8 in
  (∃ a : ℝ, a * x^2 ∈ finset.range 9 ∧ a = 7) :=
by sorry

end coefficient_x_squared_in_expansion_l804_804261


namespace inner_square_area_l804_804921

theorem inner_square_area (AB BE : ℝ) (hAB : AB = 10) (hBE : BE = 2) : 
  let x := 4 * Real.sqrt 6 - 2 in
  let area := x * x in
  area = 100 - 16 * Real.sqrt 6 :=
by
  have hx : x = 4 * Real.sqrt 6 - 2 := rfl
  have harea : area = (4 * Real.sqrt 6 - 2) * (4 * Real.sqrt 6 - 2) := rfl
  sorry

end inner_square_area_l804_804921


namespace maximum_sum_abs_diff_l804_804956

theorem maximum_sum_abs_diff (n : ℕ) (hn : 0 < n) (x : ℕ → ℝ) 
  (hx_bounds : ∀ i, 1 ≤ i ∧ i ≤ n -> 0 < x i ∧ x i < 1) :
  (∑ i in finset.range(n), ∑ j in finset.range(n), if i < j then abs (x i - x j) else 0) = (n * (n + 1)) / 6 :=
sorry

end maximum_sum_abs_diff_l804_804956


namespace largest_possible_value_mod_z_l804_804947

theorem largest_possible_value_mod_z
  (a b c d z w : ℂ)
  (ha : |a| = |b|)
  (hb : |b| = |c|)
  (hc : |c| = |d|)
  (hd : |d| > 0)
  (h_eq : a * z^3 + b * w * z^2 + c * z + d = 0)
  (h_w : |w| = 1 / 2) :
  |z| ≤ 1 :=
sorry

end largest_possible_value_mod_z_l804_804947


namespace rotate_point_D_l804_804241

theorem rotate_point_D :
  let D : ℝ × ℝ := (2, -3)
  in (D.1, D.2) = (2, -3) →
     (D.1, D.2) rotated 180 degrees counterclockwise about origin = (-2, 3) :=
by
  -- Define the rotation operation
  let rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  -- Assume point D with given coordinates
  let D : ℝ × ℝ := (2, -3)
  -- Apply the rotation
  have h : rotate_180 D = (-2, 3) := by
    unfold rotate_180
    rfl
  -- Assert the result
  exact h

end rotate_point_D_l804_804241


namespace problem_part_1_problem_part_2_l804_804870

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

noncomputable def tan_2x_when_parallel (x : ℝ) (h : (Real.sin x + 3 / 2 * Real.cos x = 0)) : Prop :=
    Real.tan (2 * x) = 12 / 5

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2

def range_f_on_interval : Prop :=
  ∀ x ∈ Set.Icc (-Real.pi / 2) 0, -Real.sqrt 2 / 2 ≤ f x ∧ f x ≤ 1 / 2

theorem problem_part_1 (x : ℝ) (h : (Real.sin x + 3 / 2 * Real.cos x = 0)) : Real.tan (2 * x) = 12 / 5 :=
by
  sorry

theorem problem_part_2 : range_f_on_interval :=
by
  sorry

end problem_part_1_problem_part_2_l804_804870


namespace sin_inequality_iff_angle_inequality_l804_804754

section
variables {A B : ℝ} {a b : ℝ} (R : ℝ) (hA : A = Real.sin a) (hB : B = Real.sin b)

theorem sin_inequality_iff_angle_inequality (A B : ℝ) :
  (A > B) ↔ (Real.sin A > Real.sin B) :=
sorry
end

end sin_inequality_iff_angle_inequality_l804_804754


namespace no_integer_solutions_to_equation_l804_804760

theorem no_integer_solutions_to_equation :
  ∀ (m n : ℤ), ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2005) :=
by solve_by_elim [math]

end no_integer_solutions_to_equation_l804_804760


namespace right_triangle_inequality_l804_804551

theorem right_triangle_inequality (a b c m : ℝ)
  (h1 : c = sqrt (a^2 + b^2))
  (h2 : a * b = c * m) :
  a + b < c + m :=
sorry

end right_triangle_inequality_l804_804551


namespace sum_of_squares_div_l804_804953

theorem sum_of_squares_div (y : Fin 50 → ℝ) 
  (h₁ : ∑ i, y i = 1)
  (h₂ : ∑ i, (y i) / (1 - y i) = 2) :
  ∑ i, (y i)^2 / (1 - y i) = 1 :=
by
  sorry

end sum_of_squares_div_l804_804953


namespace question_proof_l804_804232

-- Define the dataset
def dates : List ℕ := 
  List.replicate 12 1 ++ 
  List.replicate 12 2 ++ 
  List.replicate 12 3 ++ 
  List.replicate 12 4 ++ 
  List.replicate 12 5 ++ 
  List.replicate 12 6 ++ 
  List.replicate 12 7 ++ 
  List.replicate 12 8 ++ 
  List.replicate 12 9 ++ 
  List.replicate 12 10 ++ 
  List.replicate 12 11 ++ 
  List.replicate 12 12 ++ 
  List.replicate 12 13 ++ 
  List.replicate 12 14 ++ 
  List.replicate 12 15 ++ 
  List.replicate 12 16 ++ 
  List.replicate 12 17 ++ 
  List.replicate 12 18 ++ 
  List.replicate 12 19 ++ 
  List.replicate 12 20 ++ 
  List.replicate 12 21 ++ 
  List.replicate 12 22 ++ 
  List.replicate 12 23 ++ 
  List.replicate 12 24 ++ 
  List.replicate 12 25 ++ 
  List.replicate 12 26 ++ 
  List.replicate 12 27 ++ 
  List.replicate 12 28 ++ 
  List.replicate 12 29 ++ 
  List.replicate 11 30 ++ 
  List.replicate 7 31 

-- Define the mean
def mean (l : List ℕ) : ℚ := 
  (l.map (λ x => (x : ℚ))).sum / l.length

-- Define the median
def median (l : List ℕ) : ℕ := 
  let sorted := l.qsort (· ≤ ·)
  sorted.get (l.length / 2)

-- Define the median of the modes
def median_of_modes (l : List ℕ) : ℕ :=
  let modes := List.range' 1 29
  median modes

noncomputable def problem_statement : Prop :=
  let M := median dates
  let μ := mean dates
  let d := median_of_modes dates
  d < μ ∧ μ < M

theorem question_proof : problem_statement :=
  by {
    sorry
  }

end question_proof_l804_804232


namespace sum_of_largest_and_third_smallest_l804_804303

def is_largest_three_digit (n : ℕ) : Prop :=
  digits_of n = [8, 6, 1]

def is_third_smallest_three_digit (n : ℕ) : Prop :=
  digits_of n = [6, 1, 8]

def digits_of (n : ℕ) : list ℕ :=
  [n / 100, (n / 10) % 10, n % 10]

theorem sum_of_largest_and_third_smallest :
  ∃ a b : ℕ, is_largest_three_digit a ∧ is_third_smallest_three_digit b ∧ a + b = 1479 :=
begin
  sorry
end

end sum_of_largest_and_third_smallest_l804_804303


namespace reading_order_l804_804913

theorem reading_order (a b c d : ℝ) 
  (h1 : a + c = b + d) 
  (h2 : a + b > c + d)
  (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c :=
by sorry

end reading_order_l804_804913


namespace probability_of_green_ball_l804_804056

theorem probability_of_green_ball :
  let P_X := 0.2
  let P_Y := 0.5
  let P_Z := 0.3
  let P_green_given_X := 5 / 10
  let P_green_given_Y := 3 / 10
  let P_green_given_Z := 8 / 10
  P_green_given_X * P_X + P_green_given_Y * P_Y + P_green_given_Z * P_Z = 0.49 :=
by {
  sorry
}

end probability_of_green_ball_l804_804056


namespace calculate_total_cups_l804_804284

variable (butter : ℕ) (flour : ℕ) (sugar : ℕ) (total_cups : ℕ)

def ratio_condition : Prop :=
  3 * butter = 2 * sugar ∧ 3 * flour = 5 * sugar

def sugar_condition : Prop :=
  sugar = 9

def total_cups_calculation : Prop :=
  total_cups = butter + flour + sugar

theorem calculate_total_cups (h1 : ratio_condition butter flour sugar) (h2 : sugar_condition sugar) :
  total_cups_calculation butter flour sugar total_cups -> total_cups = 30 := by
  sorry

end calculate_total_cups_l804_804284


namespace cube_root_of_scaled_number_l804_804880

theorem cube_root_of_scaled_number (h1 : Real.cbrt 25.36 = 2.938)
                                  (h2 : Real.cbrt 253.6 = 6.329) :
  Real.cbrt 253600 = 63.29 :=
sorry

end cube_root_of_scaled_number_l804_804880


namespace find_x_l804_804885

theorem find_x : ∃ (x : ℚ), (3 * x - 5) / 7 = 15 ∧ x = 110 / 3 := by
  sorry

end find_x_l804_804885


namespace sum_first_eight_l804_804122

-- Defining the geometric sequence and its properties
def geom_seq (a r : ℕ) (n : ℕ) : ℕ := a * r ^ (n-1)

noncomputable def S_n (a r n : ℕ) : ℕ := ∑ i in finset.range n, geom_seq a r (i + 1)

theorem sum_first_eight (a r : ℕ) (h_r_pos : r > 0) 
  (h1 : geom_seq a r 1 + geom_seq a r 2 = 2) 
  (h2 : geom_seq a r 3 + geom_seq a r 4 = 6) 
  : S_n a r 8 = 80 :=
by
  sorry

end sum_first_eight_l804_804122


namespace roots_inequality_l804_804630

theorem roots_inequality (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) :
  -1 ≤ z ∧ z ≤ 13 / 3 :=
sorry

end roots_inequality_l804_804630


namespace alien_collected_95_units_l804_804757

def convert_base_six_to_ten (n : ℕ) : ℕ :=
  match n with
  | 235 => 2 * 6^2 + 3 * 6^1 + 5 * 6^0
  | _ => 0

theorem alien_collected_95_units : convert_base_six_to_ten 235 = 95 := by
  sorry

end alien_collected_95_units_l804_804757


namespace sum_of_n_values_l804_804698

theorem sum_of_n_values : 
  ∑ n in {n : ℤ | ∃ (d : ℤ), d * (2 * n - 1) = 24}.toFinset, n = 2 :=
by
  sorry

end sum_of_n_values_l804_804698


namespace isosceles_trapezoid_l804_804245

variable {A B C D : Type*}
variables [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
variables [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D]

structure Trapezoid (A B C D : Type*) :=
(base1 base2 : A)
(angleA angleB angleC angleD : B)
(parallel_base : base1 = B ∧ base2 = C)
(eq_angles : angleA = angleD ∧ angleB = angleC)

theorem isosceles_trapezoid (A B C D : Type*) [DecidableEq A] [DecidableEq B] [DecidableEq C]
  {tr : Trapezoid A B C D} (h : tr.eq_angles) : True := sorry

end isosceles_trapezoid_l804_804245


namespace problem1_problem2_l804_804959

-- Problem 1
theorem problem1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  a * b + b * c + c * a ≤ 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) :
  2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b :=
sorry

end problem1_problem2_l804_804959


namespace emily_fishes_correct_l804_804792

/-- Given conditions:
1. Emily caught 4 trout weighing 2 pounds each.
2. Emily caught 3 catfish weighing 1.5 pounds each.
3. Bluegills weigh 2.5 pounds each.
4. Emily caught a total of 25 pounds of fish. -/
def emilyCatches : Prop :=
  ∃ (trout_count catfish_count bluegill_count : ℕ)
    (trout_weight catfish_weight bluegill_weight total_weight : ℝ),
    trout_count = 4 ∧ catfish_count = 3 ∧ 
    trout_weight = 2 ∧ catfish_weight = 1.5 ∧ 
    bluegill_weight = 2.5 ∧ 
    total_weight = 25 ∧
    (total_weight = (trout_count * trout_weight) + (catfish_count * catfish_weight) + (bluegill_count * bluegill_weight)) ∧
    bluegill_count = 5

theorem emily_fishes_correct : emilyCatches := by
  sorry

end emily_fishes_correct_l804_804792


namespace common_divisors_count_l804_804150

theorem common_divisors_count (h₁ : ∀ d, d ∣ 54 ↔ d ∣ (2 * 3^3))
                              (h₂ : ∀ d, d ∣ 81 ↔ d ∣ 3^4) :
  ∃ n, n = 10 ∧ ∀ d, (d ∣ 54 ∧ d ∣ 81) ↔ (d ∈ {-1, 1, 3, 9, 27}) :=
by
  sorry

end common_divisors_count_l804_804150


namespace evaluate_log_expression_l804_804411

theorem evaluate_log_expression :
  (3 / (log 8 (5000^4)) + 2 / (log 9 (5000^4))) = 1 / 4 :=
by 
  sorry

end evaluate_log_expression_l804_804411


namespace sin_870_eq_half_l804_804043

theorem sin_870_eq_half : Real.sin (870 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_870_eq_half_l804_804043


namespace find_c_l804_804358

-- Define conditions as Lean statements
theorem find_c :
  ∀ (c n : ℝ), 
  (n ^ 2 + 1 / 16 = 1 / 4) → 
  2 * n = c → 
  c < 0 → 
  c = - (Real.sqrt 3) / 2 :=
by
  intros c n h1 h2 h3
  sorry

end find_c_l804_804358


namespace farming_supply_cost_l804_804764

-- Define the conditions
def cost_type_a : ℝ := 12
def cost_type_b : ℝ := 8
def total_weight : ℝ := 6
def ratio_a_to_b : ℝ := 2 / 3

-- Define the expected cost
def expected_total_cost : ℝ := 57.6

-- The theorem to be proven
theorem farming_supply_cost : 
  let x := (ratio_a_to_b * total_weight) / (1 + ratio_a_to_b),
      y := total_weight - x,
      cost_a := x * cost_type_a,
      cost_b := y * cost_type_b
  in cost_a + cost_b = expected_total_cost :=
by
  sorry

end farming_supply_cost_l804_804764


namespace triangle_inscribed_circle_ratio_l804_804191

theorem triangle_inscribed_circle_ratio 
  (A B C D : Type)
  [has_eq A] [has_lt B]
  (triangle : triangle A)
  (AC BC AB : B)
  (hAC : AC = 5) (hBC : BC = 12) (hAB : AB = 13) 
  (angle_ACB_right : is_right_angle (angle A C B))
  (D_on_AB : is_on_segment D A B)
  (CD_bisects_angle_ACB : bisects (segment C D) (angle A C B))
  (r1 r2 : B) 
  (ht : inscribed_circle_radius (triangle A B C) (triangle D A C) (triangle D B C) r1 r2) :
  r1 / r2 = (5 / 13) ∨ r1 / r2 = (5 / 144) ∨ r1 / r2 = (12 / 65) ∨ r1 / r2 = (15 / 52) ∨ r1 / r2 = (12 / 13) :=
by sorry

end triangle_inscribed_circle_ratio_l804_804191


namespace enlarged_poster_height_l804_804230

def original_poster_width : ℝ := 3
def original_poster_height : ℝ := 2
def new_poster_width : ℝ := 12

theorem enlarged_poster_height :
  new_poster_width / original_poster_width * original_poster_height = 8 := 
by
  sorry

end enlarged_poster_height_l804_804230


namespace exist_c_l804_804535

theorem exist_c (p : ℕ) (r : ℤ) (a b : ℤ) [Fact (Nat.Prime p)]
  (hp1 : r^7 ≡ 1 [ZMOD p])
  (hp2 : r + 1 - a^2 ≡ 0 [ZMOD p])
  (hp3 : r^2 + 1 - b^2 ≡ 0 [ZMOD p]) :
  ∃ c : ℤ, (r^3 + 1 - c^2) ≡ 0 [ZMOD p] :=
by
  sorry

end exist_c_l804_804535


namespace lowest_common_multiple_10_14_20_l804_804421
open Nat

theorem lowest_common_multiple_10_14_20 : lcm (lcm 10 14) 20 = 140 :=
by sorry

end lowest_common_multiple_10_14_20_l804_804421


namespace proof_problem_l804_804729

-- Define the product quality indicators
structure Product :=
(x : ℕ)
(y : ℕ)
(z : ℕ)

-- Overall indicator S is defined as x + y + z
def overall_indicator (p : Product) : ℕ := p.x + p.y + p.z

-- Define the list of products in the sample
def products : List Product :=
[
  { x := 1, y := 1, z := 2 }, -- A1
  { x := 2, y := 1, z := 1 }, -- A2
  { x := 2, y := 2, z := 2 }, -- A3
  { x := 1, y := 1, z := 1 }, -- A4
  { x := 1, y := 2, z := 1 }, -- A5
  { x := 1, y := 2, z := 2 }, -- A6
  { x := 2, y := 1, z := 1 }, -- A7
  { x := 2, y := 2, z := 1 }, -- A8
  { x := 1, y := 1, z := 1 }, -- A9
  { x := 2, y := 1, z := 2 }  -- A10
]

-- Define the condition for first-class product
def is_first_class (p : Product) : Prop := overall_indicator p ≤ 4

-- List of first-class products
def first_class_products : List Product := products.filter is_first_class

-- Calculate the rate of first-class products
def first_class_rate : ℚ := (first_class_products.length : ℚ) / (products.length : ℚ)

-- Determine the pairs of selecting 2 products
def pairs (l : List Product) : List (Product × Product) :=
  l.bind (λ a, l.map (λ b, (a, b)))

-- Define the condition for event B
def event_B (pair : Product × Product) : Prop :=
  overall_indicator pair.1 = 4 ∧ overall_indicator pair.2 = 4

-- Calculate the probability of event B
def prob_event_B : ℚ :=
  let pairs := pairs first_class_products
  let B_pairs := pairs.filter event_B
  (B_pairs.length : ℚ) / (pairs.length : ℚ)

-- Lean statement for the problem
theorem proof_problem :
  first_class_rate = 3 / 5 ∧ prob_event_B = 2 / 5 :=
  sorry

end proof_problem_l804_804729


namespace thomas_loan_positive_difference_l804_804677

-- Define the parameters and conditions of the problem
def principal : ℝ := 15000
def rate_compounded : ℝ := 0.08
def rate_simple : ℝ := 0.1
def n_compounded : ℕ := 2 -- biannually compounding periods per year
def years_total : ℕ := 12
def years_half : ℕ := 6

-- Definitions for compounded loan calculations
def amount_owed_compounded (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- Definitions for simple interest loan calculations
def amount_owed_simple (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P + P * r * t

-- Definition for payment at six years for Scheme 1
def payment_at_six_years (A : ℝ) : ℝ :=
  A / 3

-- Definition for remaining amount after six years for Scheme 1
def remaining_amount (A : ℝ) (payment : ℝ) : ℝ :=
  A - payment

-- Define the positive difference that needs to be proven
def positive_difference (A_total1 : ℝ) (A_total2 : ℝ) : ℝ :=
  if A_total1 > A_total2 then A_total1 - A_total2 else A_total2 - A_total1

-- State the theorem to be proven
theorem thomas_loan_positive_difference :
  let A_6_years := amount_owed_compounded principal rate_compounded n_compounded years_half,
      payment := payment_at_six_years A_6_years,
      remaining := remaining_amount A_6_years payment,
      A_12_years := amount_owed_compounded remaining rate_compounded n_compounded years_half,
      total_paid_compounded := payment + A_12_years,
      total_paid_simple := amount_owed_simple principal rate_simple years_total in
  positive_difference total_paid_compounded total_paid_simple = 3048 :=
by {
  -- Here you would provide the proof, but we omit it with sorry.
  sorry,
}

end thomas_loan_positive_difference_l804_804677


namespace circle_diameter_from_area_l804_804692

theorem circle_diameter_from_area (A : ℝ) (h : A = 225 * Real.pi) : ∃ d : ℝ, d = 30 :=
  by
  have r := Real.sqrt (225)
  have d := 2 * r
  exact ⟨d, sorry⟩

end circle_diameter_from_area_l804_804692


namespace value_of_f_a5_l804_804448

noncomputable def f (x : ℝ) : ℝ := (x - 4)^3 + x - 1

axiom arith_seq (a : ℕ → ℝ) (d : ℝ) : (a 1 + a 9 = 2 * a 5) ∧ (a 2 + a 8 = 2 * a 5) ∧ (a 3 + a 7 = 2 * a 5) ∧ (a 4 + a 6 = 2 * a 5)

axiom sum_f (a : ℕ → ℝ) : (f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) + f (a 8) + f (a 9) = 27)

theorem value_of_f_a5 (a : ℕ → ℝ) (d : ℝ) (h_seq : arith_seq a d) (h_sum : sum_f a) : 
  f (a 5) = 3 :=
by
  sorry

end value_of_f_a5_l804_804448


namespace y_percentage_of_8950_l804_804901

noncomputable def x := 0.18 * 4750
noncomputable def y := 1.30 * x
theorem y_percentage_of_8950 : (y / 8950) * 100 = 12.42 := 
by 
  -- proof steps are omitted
  sorry

end y_percentage_of_8950_l804_804901


namespace systematic_sampling_first_two_numbers_l804_804302

/-- Given a population of size 8000, if systematic sampling selects a sample of size 50 and 
the last sampled number is 7900, then the first two sampled numbers are 60 and 220 -/
theorem systematic_sampling_first_two_numbers :
  ∀ (N n last: ℕ), 
  N = 8000 → n = 50 → last = 7900 → 
  ∃ (first second: ℕ), 
  first = 60 ∧ second = 220 :=
begin
  sorry
end

end systematic_sampling_first_two_numbers_l804_804302


namespace area_of_triangle_l804_804454

noncomputable def equilateral_hyperbola := sorry
noncomputable def point_on_hyperbola := sorry
noncomputable def foci_of_hyperbola := sorry
noncomputable def perpendicular_lines := sorry
noncomputable def triangle_area := sorry

theorem area_of_triangle {x y : ℝ}
  (h1 : equilateral_hyperbola (x, y)) 
  (h2 : point_on_hyperbola P (x, y)) 
  (h3 : foci_of_hyperbola F₁ F₂) 
  (h4 : perpendicular_lines (P, F₁) (P, F₂)) 
  : triangle_area F₁ F₂ P = 1 := 
sorry

end area_of_triangle_l804_804454


namespace main_theorem_l804_804680

-- Defining basic elements (vertices, parallel lines, reflections)
variables {A B C : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C]
variables (d_a d_b d_c : A → A) (triangleABC : Type) 
variables (triangleXYZ : Type)

-- Assuming necessary properties for parallel lines and triangle formation
variables {O H D E F : Type} [Circumcenter A O] [Orthocenter A H] 
variables [Reflect O A D] [Reflect O B E] [Reflect O C F]

-- Assuming rhye axis must be parallel for subsequent transformations and reflections
axiom reflections_form_triangleXYZ : d_a(BC) = d_b(CA) = d_c(AB)

-- Defining the locus of the incenters of triangles formed by reflections
noncomputable def locus_incenters (triangleABC: Type) (d_a d_b d_c : A → A) : Set Point :=
  {I | I ∈ Circumcircle DEF}

-- The main theorem statement
theorem main_theorem : 
  (∀ triangleABC : Type, ∃ triangleXYZ : Type, locus_incenters triangleABC d_a d_b d_c) :=
by
  sorry

end main_theorem_l804_804680


namespace angle_between_lines_eq_angle_between_circles_l804_804647

theorem angle_between_lines_eq_angle_between_circles
  (S1 S2 : Circle) (A B P1 Q1 P2 Q2 : Point) (p q : Line)
  (h1 : S1 ∩ S2 = {A, B})
  (h2 : p ∈ A)
  (h3 : q ∈ A)
  (h4 : p ∩ S1 = {P1, Q1})
  (h5 : p ∩ S2 = {P2, Q2}) :
  angle (P1, Q1) (P2, Q2) = angle_between_circles S1 S2 :=
sorry

end angle_between_lines_eq_angle_between_circles_l804_804647


namespace total_fencing_cost_l804_804005

noncomputable def shortSide : ℝ := 30
noncomputable def longSide (area : ℝ) (shortSide : ℝ) : ℝ := area / shortSide
noncomputable def diagonal (longSide : ℝ) (shortSide : ℝ) : ℝ := real.sqrt (longSide^2 + shortSide^2)

theorem total_fencing_cost 
  (area : ℝ := 1200)
  (shortSide : ℝ := 30)
  (costShortSide : ℝ := 12)
  (costLongSide : ℝ := 14)
  (costDiagonal : ℝ := 16) :
  let l := longSide area shortSide,
      d := diagonal l shortSide,
      total_cost := shortSide * costShortSide + l * costLongSide + d * costDiagonal
  in total_cost = 1720 := 
by
  sorry

end total_fencing_cost_l804_804005


namespace blue_polygon_exists_l804_804255

theorem blue_polygon_exists (n : ℕ) (hn : n ≥ 3) 
                            (red_points : Finset ℝ) 
                            (h_red : red_points.card = 2016) : 
    ∃ (blue_polygon : Finset ℝ), 
    (∀ (point ∈ blue_polygon), point ∉ red_points) ∧ (is_regular_ngon blue_polygon n) :=
by 
  sorry

def is_regular_ngon (points : Finset ℝ) (n : ℕ) : Prop := 
  sorry

end blue_polygon_exists_l804_804255


namespace area_AMCN_l804_804623

theorem area_AMCN 
  (ABCD : Type) [rect : Rectangle ABCD] 
  (A B C D M N : ABCD) 
  (h_AB : distance A B = 10) 
  (h_BC : distance B C = 6) 
  (h_mid_M : midpoint A B M) 
  (h_mid_N : midpoint C D N) :
  area_AMCN = 30 := 
sorry

end area_AMCN_l804_804623


namespace vector_magnitude_ratio_l804_804111

variable (a b : ℝ)
variable (nonzero_a : a ≠ 0)
variable (nonzero_b : b ≠ 0)
variable (angle_120 : (a * b) = - (a * b) / 2)
variable (perpendicular : (a * b - b * b) = 0)

theorem vector_magnitude_ratio : 
  (| 2 * a - b | / | 2 * a + b |) = sqrt (10 + sqrt(33)) / 3 :=
by 
  sorry

end vector_magnitude_ratio_l804_804111


namespace num_two_digit_integers_congruent_to_1_mod_4_l804_804485

theorem num_two_digit_integers_congruent_to_1_mod_4 : 
  let count := (range [4*k+1 | k, 3 ≤ k ∧ k ≤ 24]).length in
  count = 22 :=
by
  sorry

end num_two_digit_integers_congruent_to_1_mod_4_l804_804485


namespace sum_of_integers_l804_804992

variable (p q r s : ℤ)

theorem sum_of_integers :
  (p - q + r = 7) →
  (q - r + s = 8) →
  (r - s + p = 4) →
  (s - p + q = 1) →
  p + q + r + s = 20 := by
  intros h1 h2 h3 h4
  sorry

end sum_of_integers_l804_804992


namespace min_distance_point_l804_804891

-- Define the point A
def A : (ℝ × ℝ) := (3, 2)

-- Define the parabola y^2 = 2x
def on_parabola (P : ℝ × ℝ) : Prop := P.2 ^ 2 = 2 * P.1

-- Define the distance function between two points
def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Define the focus of the parabola y^2 = 2x
def F : (ℝ × ℝ) := (1 / 2, 0)

-- Define the function to minimize |PA| + |PF|
def target_function (P : ℝ × ℝ) : ℝ := distance P A + distance P F

-- The statement to prove
theorem min_distance_point : ∀ P : (ℝ × ℝ), on_parabola P → 
  target_function P = min (target_function P) → P = (2, 2) := 
sorry

end min_distance_point_l804_804891


namespace min_value_y_l804_804661

noncomputable def y (x : ℝ) : ℝ := x + 4 / (x - 1)

theorem min_value_y : ∀ x > 1, y x ≥ 5 ∧ (y x = 5 ↔ x = 3) :=
by
  intros x hx
  sorry

end min_value_y_l804_804661


namespace smallest_positive_x_for_palindrome_l804_804311

-- Define a function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Prop := 
  let s := n.to_string in 
  s = s.reverse

-- Define the main theorem to be proved
theorem smallest_positive_x_for_palindrome : ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 8321) ∧ x = 17 :=
by
  use 17
  -- Provide the conditions to be checked
  split
  -- Check the positivity of x
  { exact Nat.zero_lt_succ 16 }
  split
  -- Check if 8338 is a palindrome
  { unfold is_palindrome
    have h8321 : 8321.to_string = "8321" := rfl
    have h8338 : 8338.to_string = "8338" := rfl
    rw [h8338, h8321]
    norm_num
    sorry -- The complete proof that "8338" is a palindrome }
  -- Check the correctness of x
  { rfl }

end smallest_positive_x_for_palindrome_l804_804311


namespace max_elements_of_union_of_intersecting_sets_l804_804201

-- Define the vertices of the 2010-gon
def vertices : set ℕ := {i | 1 ≤ i ∧ i ≤ 2010}

-- Define the set of all sides and diagonals (pairs of distinct vertices)
def K : set (ℕ × ℕ) := { (x, y) | x ∈ vertices ∧ y ∈ vertices ∧ x ≠ y }

-- Define what it means for a set to be an intersecting set
def is_intersecting_set (A : set (ℕ × ℕ)) : Prop :=
  ∀ (p q : ℕ × ℕ), p ∈ A → q ∈ A → p ≠ q → (line_segments_intersect p q)

-- The proof problem statement
theorem max_elements_of_union_of_intersecting_sets :
  ∃ (A B : set (ℕ × ℕ)), is_intersecting_set A ∧ is_intersecting_set B ∧ (∃ n : ℕ, finset.card (A ∪ B) = n ∧ n = 4019) :=
sorry

end max_elements_of_union_of_intersecting_sets_l804_804201


namespace min_value_fraction_l804_804664

variables {X : ℝ → ℝ} {σ m n : ℝ}

/-- X is normally distributed with mean 10 and variance σ², and 
      P(X > 12) = m, P(8 ≤ X ≤ 10) = n. 
    We need to show: 
      min (2 / m + 1 / n) = 6 + 4 * Real.sqrt 2. -/
theorem min_value_fraction (hX : ∀ x, X x ∼ Normal 10 (σ^2))
    (hPm : P(λ x, X x > 12) = m)
    (hPn : P(λ x, 8 ≤ X x ∧ X x ≤ 10) = n) : 
    6 + 4 * Real.sqrt 2 ≤ 2 / m + 1 / n :=
sorry

end min_value_fraction_l804_804664


namespace hefei_route_assignment_l804_804146

-- Define the classes and routes
inductive Class
| Grade1 | Grade2 | Grade3 | Grade4 | Grade5

inductive Route
| Xian | Yangzhou | SouthernAnhui

-- Define the predicates for the conditions
def each_route_chosen_at_least_once (assignments : List (Class × Route)) : Prop :=
  ∀ route, route ∈ Route → ∃ class, (class, route) ∈ assignments

def classes1_and_2_different_routes (assignments : List (Class × Route)) : Prop :=
  ∀ (route : Route), (Class.Grade1, route) ∈ assignments → ¬ (Class.Grade2, route) ∈ assignments

-- The main theorem to be proven: there are 114 valid ways to assign routes to classes
theorem hefei_route_assignment : 
  ∃ (assignments : List (Class × Route)), 
    each_route_chosen_at_least_once assignments ∧ classes1_and_2_different_routes assignments ∧ (List.length assignments = 5) :=
sorry

end hefei_route_assignment_l804_804146


namespace possible_numbers_l804_804602

theorem possible_numbers (N : ℕ) (h_digit : 10 ≤ N ∧ N ≤ 99)
  (h_claimed_divisors : ∀ d ∈ [3, 4, 5, 9, 10, 15, 18, 30], d ∣ N ∨ ¬ d ∣ N)
  (h_mistakes : Nat.countp (λ d, ¬ d ∣ N) [3, 4, 5, 9, 10, 15, 18, 30] = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 :=
by
  sorry

end possible_numbers_l804_804602


namespace mean_difference_is_180_l804_804508

variable {T : ℕ} (hT : T = 3000000)
variable {n : ℕ} (hn : n = 1500)
variable {E : ℕ} (hE : E = 300000)
variable {C : ℕ} (hC : C = 30000)

def difference_between_means (T n E C : ℕ) : ℕ :=
  let S := T - C
  let mean_actual := (S + C) / n
  let mean_incorrect := (S + E) / n
  mean_incorrect - mean_actual

theorem mean_difference_is_180 :
  difference_between_means 3000000 1500 300000 30000 = 180 :=
by
  rw [difference_between_means]
  simp [Nat.sub_def, Nat.add_def, Nat.div_def]
  sorry

end mean_difference_is_180_l804_804508


namespace domain_of_function_l804_804840

theorem domain_of_function : 
  ∀ x ∈ set.Icc (0 : ℝ) (2 * Real.pi), 
    (Real.tan x ≥ 0) ∧ (-Real.cos x ≥ 0) ↔ x ∈ set.Ico Real.pi (3 * Real.pi / 2) :=
by
  sorry

end domain_of_function_l804_804840


namespace max_consecutive_sum_below_500_l804_804310

theorem max_consecutive_sum_below_500 : 
  ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ n → (∑ i in finset.range k + 10, (i + 10)) < 500) ∧ 
  (∑ i in finset.range (n + 1 + 10), (i + 10) ≥ 500) := 
sorry

end max_consecutive_sum_below_500_l804_804310


namespace car_trip_time_difference_l804_804727

theorem car_trip_time_difference
  (average_speed : ℝ)
  (distance1 distance2 : ℝ)
  (speed_60_mph : average_speed = 60)
  (dist1_540 : distance1 = 540)
  (dist2_510 : distance2 = 510) :
  ((distance1 - distance2) / average_speed) * 60 = 30 := by
  sorry

end car_trip_time_difference_l804_804727


namespace find_g_product_l804_804952

theorem find_g_product 
  (x1 x2 x3 x4 x5 : ℝ)
  (h_root1 : x1^5 - x1^3 + 1 = 0)
  (h_root2 : x2^5 - x2^3 + 1 = 0)
  (h_root3 : x3^5 - x3^3 + 1 = 0)
  (h_root4 : x4^5 - x4^3 + 1 = 0)
  (h_root5 : x5^5 - x5^3 + 1 = 0)
  (g : ℝ → ℝ) 
  (hg : ∀ x, g x = x^2 - 3) :
  g x1 * g x2 * g x3 * g x4 * g x5 = 107 := 
sorry

end find_g_product_l804_804952


namespace line_segment_connect_param_l804_804863

theorem line_segment_connect_param (a b c d : ℝ) :
  (1 = b) ∧ (-3 = d) ∧ (6 = a + b) ∧ (12 = c + d) → a + c^2 + b^2 + d^2 = 240 :=
by
  intros h,
  cases h with h_b h_rest,
  cases h_rest with h_d h_rest2,
  cases h_rest2 with h_ab h_cd,
  rw [h_b, h_d] at h_ab h_cd,
  rw [h_b, h_d],

  have ha : a = 5,
  { linarith [h_ab] },
  have hc : c = 15,
  { linarith [h_cd] },
  
  rw [ha, hc],
  norm_num,
  sorry

end line_segment_connect_param_l804_804863


namespace binary_conversion_93_l804_804400

-- Definition of the binary conversion function
def decimal_to_binary (n : ℕ) : list ℕ :=
  if n = 0 then [0] else
  let rec aux (n : ℕ) (acc : list ℕ) : list ℕ :=
    if n = 0 then acc else aux (n / 2) ((n % 2) :: acc)
  in aux n []

-- The main theorem statement
theorem binary_conversion_93 : decimal_to_binary 93 = [1, 0, 1, 1, 1, 0, 1] :=
by
  -- proof would go here
  sorry

end binary_conversion_93_l804_804400


namespace find_m_l804_804925

-- Conditions
variables (A B C E : Type) [IsTriangle A B C]
variable (AB : A = B ⟺ 5)
variable (BC : B = C ⟺ 12)
variable (AC : A = C ⟺ 13)
variable (BE_bisector : IsAngleBisectorAngle B E C)

-- Problem statement
theorem find_m (m : ℝ) : BE = m * sqrt 2 → m = 5 := by
  sorry

end find_m_l804_804925


namespace pirates_problem_l804_804740

noncomputable def initial_coins : ℕ :=
  3^9 * 5^12

def remaining_coins (x : ℕ) : ℕ :=
  x * (14.fact / 15^14)

def final_coins (x : ℕ) : ℕ :=
  x * (14.fact / 15^13)

theorem pirates_problem :
  final_coins initial_coins = 2520 :=
by
  sorry

end pirates_problem_l804_804740


namespace problem1_problem2_l804_804859

def f (x : ℝ) (a : ℝ) : ℝ := x * log x - a * x^2

theorem problem1 :
  f (1 : ℝ) (1/2 : ℝ) = 0 ∧
  ∀ x : ℝ, (0 < x) → (f_deriv (x) = log x + 1 - x) ∧ (f_deriv (x) < 0) :=
sorry

theorem problem2 (x1 x2 : ℝ) (h : x1 ≠ x2) :
  let a := 1/2 in 
  let t := x1 / x2, g := λ t, log t - 2 * (t - 1) / (t + 2) in
  f_deriv (x1) = 0 ∧ f_deriv (x2) = 0 →
  1 < t ∧ 
  (∀ t > 1, g (t) > 0) →
  x1 * (x2^2) > Real.exp (-1) :=
sorry

end problem1_problem2_l804_804859


namespace percentage_increase_l804_804038

theorem percentage_increase 
  (distance : ℝ) (time_q : ℝ) (time_y : ℝ) 
  (speed_q : ℝ) (speed_y : ℝ) 
  (percentage_increase : ℝ) 
  (h_distance : distance = 80)
  (h_time_q : time_q = 2)
  (h_time_y : time_y = 1.3333333333333333)
  (h_speed_q : speed_q = distance / time_q)
  (h_speed_y : speed_y = distance / time_y)
  (h_faster : speed_y > speed_q)
  : percentage_increase = ((speed_y - speed_q) / speed_q) * 100 :=
by
  sorry

end percentage_increase_l804_804038


namespace midpoint_of_symmetric_chord_on_ellipse_l804_804468

theorem midpoint_of_symmetric_chord_on_ellipse
  (A B : ℝ × ℝ) -- coordinates of points A and B
  (hA : (A.1^2 / 16) + (A.2^2 / 4) = 1) -- A lies on the ellipse
  (hB : (B.1^2 / 16) + (B.2^2 / 4) = 1) -- B lies on the ellipse
  (symm : 2 * (A.1 + B.1) / 2 - 2 * (A.2 + B.2) / 2 - 3 = 0) -- A and B are symmetric about the line
  : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (2, 1 / 2) :=
  sorry

end midpoint_of_symmetric_chord_on_ellipse_l804_804468


namespace wall_building_days_l804_804931

theorem wall_building_days
  (m1 d1 m2: ℕ) (d2_half: ℝ)
  (h1: m1 = 18) (h2: d1 = 6) (h3: m2 = 30)
  (k: ℝ := m1 * d1)
  (h4: k = 108):
  d2_half = (k / m2) * 2 → d2_half = 7.2 :=
by
  intros h
  rw [h4, mul_div_cancel' 108 (by norm_num : (30 : ℝ) ≠ 0)] at h
  norm_num at h
  exact h

end wall_building_days_l804_804931


namespace squares_area_comparison_l804_804333
noncomputable def area_percentage_increase (s : ℕ) : ℕ :=
  let areaA := s * s
  let sideB := 2.5 * s
  let areaB := 6.25 * s * s
  let sideC := 4 * s
  let areaC := 16 * s * s
  let total_areaAB := 7.25 * s * s
  let area_difference := areaC - total_areaAB
  let percentage_increase := (area_difference / total_areaAB) * 100
  percentage_increase

theorem squares_area_comparison (s : ℕ) : area_percentage_increase s = 120.69 :=
by
  sorry

end squares_area_comparison_l804_804333


namespace length_of_AE_l804_804526

open Real
open EuclideanGeometry

def triangle_area (a b c : ℝ) (angle : ℝ) : ℝ :=
  (1 / 2) * a * b * sin angle

theorem length_of_AE :
  ∀ (AB AC ∠BAC AD ∠DAE : ℝ),
  AB = 4 →
  AC = 5 →
  ∠BAC = π / 3 →
  AD = 2.5 →
  ∠DAE = π / 3 →
  triangle_area AB AC ∠BAC = triangle_area AD (AE) ∠DAE →
  AE = 8 :=
by
  intros AB AC ∠BAC AD ∠DAE hAB hAC hBAC hAD hDAE heq
  sorry

end length_of_AE_l804_804526


namespace election_votes_l804_804505

theorem election_votes (T V winner_votes runner_up_votes third_votes fourth_votes fifth_votes sixth_votes invalid_votes undecided_votes: ℕ)
  (h_valid_fraction : 89% of T = V)
  (h_invalid_fraction : 7% of T = invalid_votes)
  (h_undecided_fraction : 4% of T = undecided_votes)
  (h_winner_percentage : winner_votes = 35% of V)
  (h_runner_up_percentage : runner_up_votes = 25% of V)
  (h_winner_margin : winner_votes - runner_up_votes = 2000)
  (h_third_percentage : third_votes = 16% of V)
  (h_fourth_percentage : fourth_votes = 10% of V)
  (h_fifth_percentage : fifth_votes = 8% of V)
  (h_sixth_percentage : sixth_votes = 6% of V)
  : 
  (T = 22472) ∧ 
  (winner_votes =  7000) ∧ 
  (runner_up_votes = 5000) ∧ 
  (third_votes = 3200) ∧ 
  (fourth_votes = 2000) ∧ 
  (fifth_votes = 1600) ∧ 
  (sixth_votes = 1200) ∧ 
  (invalid_votes = 1572) ∧ 
  (undecided_votes = 899) := 
sorry

end election_votes_l804_804505


namespace largest_x_l804_804226

theorem largest_x (x y z : ℝ) (h₁ : x + y + z = 7) (h₂ : x * y + x * z + y * z = 12) : 
  x ≤ (14 + 2 * Real.sqrt 46) / 6 :=
sorry

end largest_x_l804_804226


namespace f2009_equals_cos_l804_804883

-- Define the sequence of functions fn(x) for n ∈ ℕ where f0(x) = sin x
noncomputable def fn : ℕ → (ℝ → ℝ)
| 0     := λ x, Real.sin x
| (n+1) := λ x, (fn n) x.derivative

-- State the main theorem: f2009(x) = cos x
theorem f2009_equals_cos (x : ℝ) :
  fn 2009 x = Real.cos x := sorry

end f2009_equals_cos_l804_804883


namespace quadratic_solution_1_quadratic_solution_2_l804_804989

theorem quadratic_solution_1 (x : ℝ) :
  x^2 + 3 * x - 1 = 0 ↔ (x = (-3 + Real.sqrt 13) / 2) ∨ (x = (-3 - Real.sqrt 13) / 2) :=
by
  sorry

theorem quadratic_solution_2 (x : ℝ) :
  (x - 2)^2 = 2 * (x - 2) ↔ (x = 2) ∨ (x = 4) :=
by
  sorry

end quadratic_solution_1_quadratic_solution_2_l804_804989


namespace toms_age_l804_804710

variable (T J : ℕ)

theorem toms_age :
  (J - 6 = 3 * (T - 6)) ∧ (J + 4 = 2 * (T + 4)) → T = 16 :=
by
  intros h
  sorry

end toms_age_l804_804710


namespace sector_area_l804_804259

theorem sector_area (r θ : ℝ) (h_r : r = 15) (h_θ : θ = 42) :
  (θ / 360) * Real.pi * r^2 ≈ 82.47 :=
by
  rw [h_r, h_θ]
  norm_num
  sorry

end sector_area_l804_804259


namespace volume_ratio_tetrahedron_octahedron_l804_804007

noncomputable def ratio_of_volumes (s : ℝ) : ℝ :=
  let t := (2 / 3) * (sqrt 3 / 2 * s) in
  let V_T := (t^3 * sqrt 2) / 12 in
  let V_O := (s^3 * sqrt 2) / 3 in
  V_T / V_O

theorem volume_ratio_tetrahedron_octahedron (s : ℝ) (h : s > 0) : ratio_of_volumes s = sqrt 3 / 12 :=
by
  sorry

end volume_ratio_tetrahedron_octahedron_l804_804007


namespace maximum_area_of_rectangular_pen_l804_804449

noncomputable def max_rect_area (P L : ℝ) (h : L ≥ 15) : ℝ :=
  let x := L in
  let y := (P / 2) - L in
  x * y

theorem maximum_area_of_rectangular_pen (h : (max_rect_area 60 15 (by linarith)) = 225) : 
  ∃ L : ℝ, L ≥ 15 ∧ max_rect_area 60 L (by linarith) = 225 :=
begin
  sorry,
end

end maximum_area_of_rectangular_pen_l804_804449


namespace domain_of_f_l804_804650

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 23)

theorem domain_of_f : ∀ x : ℝ, f x = 1 / (x - 23) →
  (∃ r : ℝ, f r ∧ r ≠ 23) :=
by
  -- Proof will go here.
  sorry

end domain_of_f_l804_804650


namespace prove_main_theorem_l804_804367

-- Definition of the sequence {a_n}
def a_sequence : ℕ → ℝ
| 0     := a₀
| 1     := 1 - a₀
| (n+1) := 1 - a_sequence n * (1 - a_sequence n)

-- Main theorem statement
noncomputable def main_theorem (n : ℕ) (a₀ : ℝ) (h₀ : a₀ ≠ 0) (h₁ : a₀ ≠ 1) : Prop :=
  let a : ℕ → ℝ := a_sequence in
  (∏ i in finset.range (n + 1), a i) * (∑ i in finset.range (n + 1), 1 / a i) = 1

theorem prove_main_theorem (n : ℕ) (a₀ : ℝ) (h₀ : a₀ ≠ 0) (h₁ : a₀ ≠ 1) : main_theorem n a₀ h₀ h₁ :=
sorry

end prove_main_theorem_l804_804367


namespace meaningful_sqrt_domain_l804_804897

theorem meaningful_sqrt_domain (x : ℝ) : (∃ y : ℝ, y = sqrt(1 / (2 * x - 3))) ↔ x > 3 / 2 :=
by sorry

end meaningful_sqrt_domain_l804_804897


namespace real_solutions_x4_plus_3_minus_x4_eq_82_l804_804071

theorem real_solutions_x4_plus_3_minus_x4_eq_82 :
  ∀ x : ℝ, x = 2.6726 ∨ x = 0.3274 → x^4 + (3 - x)^4 = 82 := by
  sorry

end real_solutions_x4_plus_3_minus_x4_eq_82_l804_804071


namespace semicircle_chord_product_l804_804220

open Complex

noncomputable def product_of_chords (r : ℝ) (n : ℕ) : ℝ :=
  let ω := exp (2 * π * I / (2 * n))
  let A := r
  let B := -r
  let C := λ k : ℕ, r * ω ^ k
  let AC := λ k, abs (A - C k)
  let BC := λ k, abs (B - C k)
  (finset.range (n - 1)).prod (λ k, AC k * BC k)

theorem semicircle_chord_product :
  product_of_chords 3 5 = 32805 := by
  sorry

end semicircle_chord_product_l804_804220


namespace savings_by_paying_cash_l804_804970

def cash_price := 400
def down_payment := 120
def monthly_installment := 30
def number_of_months := 12

theorem savings_by_paying_cash :
  let total_cost_plan := down_payment + (monthly_installment * number_of_months) in
  let savings := total_cost_plan - cash_price in
  savings = 80 := 
by
  let total_cost_plan := down_payment + monthly_installment * number_of_months
  let savings := total_cost_plan - cash_price
  sorry

end savings_by_paying_cash_l804_804970


namespace abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one_l804_804717

theorem abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one (x : ℝ) :
  |x| < 1 → x^3 < 1 ∧ (x^3 < 1 → |x| < 1 → False) :=
by
  sorry

end abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one_l804_804717


namespace no_tangential_triangle_exists_l804_804852

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the ellipse C2
def C2 (a b x y : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Additional condition that the point (1, 1) lies on C2
def point_on_C2 (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (1^2) / (a^2) + (1^2) / (b^2) = 1

-- The theorem to prove
theorem no_tangential_triangle_exists (a b : ℝ) (h : a > b ∧ b > 0) :
  point_on_C2 a b h →
  ¬ ∃ (A B C : ℝ × ℝ), 
    (C1 A.1 A.2 ∧ C1 B.1 B.2 ∧ C1 C.1 C.2) ∧ 
    (C2 a b A.1 A.2 h ∧ C2 a b B.1 B.2 h ∧ C2 a b C.1 C.2 h) :=
by sorry

end no_tangential_triangle_exists_l804_804852


namespace find_line_eq_l804_804120

-- Define the point P and the angle of inclination
def P := (2 : ℝ, 4 : ℝ)
def θ := 45 * (Real.pi / 180)  -- Converting degrees to radians

-- Define the line equation
def line_eq (x y : ℝ) := x - y + 2 = 0

-- Define the conditions: point P on the line and slope determined by angle
def line_conditions :=
  (2 - P.1) * Real.tan θ = 1 ∧ P.2 = 4

-- The statement we want to prove
theorem find_line_eq : 
  (∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ b ≠ 0 ∧ line_conditions) →
  line_eq :=
  by
  sorry

end find_line_eq_l804_804120


namespace smallest_consecutive_even_sum_l804_804286

theorem smallest_consecutive_even_sum (n : ℕ) (h : 1) (sum_even : n + (n+2) + (n+4) + (n+6) + (n+8) = 240) : 
  n = 44 :=
by 
  sorry

end smallest_consecutive_even_sum_l804_804286


namespace find_d_l804_804949

variables {a b c d : ℤ}

-- Given a polynomial with negative roots
def g (x : ℤ) : ℤ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem find_d (h1 : ∀ (r : ℤ), g r = 0 → r < 0) 
(h2 : a + b + c + d = 2031) : 
d = 1540 := 
sorry

end find_d_l804_804949


namespace correct_statements_l804_804522

def class (k : ℤ) : Set ℤ := {n | ∃ m : ℤ, n = 5 * m + k}

def statement1 : Prop := 2011 ∈ class 1
def statement2 : Prop := -3 ∈ class 3
def statement3 : Prop := ∀ z : ℤ, ∃ (k : ℤ), k ∈ {0, 1, 2, 3, 4} ∧ z ∈ class k
def statement4 : Prop := ∀ (a b : ℤ), (∃ m : ℤ, a = 5 * m + (a % 5)) ∧ (∃ n : ℤ, b = 5 * n + (b % 5)) → (a % 5 = b % 5 ↔ a - b ∈ class 0)

theorem correct_statements :
  (statement1) ∧ (¬ statement2) ∧ (statement3) ∧ (statement4) :=
by
  sorry

end correct_statements_l804_804522


namespace proof_quadrilateral_properties_l804_804459

noncomputable def quadrilateral_proof : Prop :=
  let Q : Point := (0, 0, 0)
  let A : Point := (1, 0, 0)
  let P : Point := (0, 0, sqrt 3)
  let B : Point := (0, sqrt 3, 0)
  let D : Point := (-1, sqrt 3, 0)
  let QB : Vector := (0, sqrt 3, 0)
  let M : Point := (-1/2, sqrt (3) / 2, sqrt (3) / 2)
  midpoint (Point.A, Point.D) Q ∧
  parallelogram (Point.P, Point.Q, Point.A) ∧
  contains (Plane.ABC, Line.AB) ∧
  midpoint (Point.P, Point.Q) M ∧
  angle_complementary (PQB, M) ∧
  perpendicular (Line.P, Line.AB) ∧
  dihedral_complementary_value (PQM,  sqrt (3) / 2)

theorem proof_quadrilateral_properties : quadrilateral_proof :=
  by
    sorry

end proof_quadrilateral_properties_l804_804459


namespace solve_inequality_l804_804075

noncomputable def is_solution (x : ℝ) : Prop :=
  (x ∈ set.Icc ( (76 - 3 * Real.sqrt 60) / 14 ) 5 ∪
  set.Ioc 5 ((76 + 3 * Real.sqrt 60) / 14))

theorem solve_inequality (x : ℝ) : 
  ( (x^2 + 2*x + 1) / (x-5)^2 >= 15 ) ↔ is_solution x :=
sorry

end solve_inequality_l804_804075


namespace distance_P1_P2_eq_sqrt_14_l804_804518

-- Define Point in 3D space
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

-- Distance function between two points in 3D
def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2)

-- Define point A
def A : Point3D := { x := -1, y := 2, z := -3 }

-- Define projection of A on the yOz plane (P1)
def P1 : Point3D := { x := 0, y := A.y, z := A.z }

-- Define projection of A on the x axis (P2)
def P2 : Point3D := { x := A.x, y := 0, z := 0 }

-- Problem statement: Prove that the distance between P1 and P2 is sqrt(14)
theorem distance_P1_P2_eq_sqrt_14 : distance P1 P2 = real.sqrt 14 := 
sorry

end distance_P1_P2_eq_sqrt_14_l804_804518


namespace two_digit_number_possible_options_l804_804598

theorem two_digit_number_possible_options
  (N : ℕ)
  (h1 : 10 ≤ N ∧ N < 100)
  (h2 : (N % 3 = 0 ∨ N % 3 ≠ 0) ∧
        (N % 4 = 0 ∨ N % 4 ≠ 0) ∧
        (N % 5 = 0 ∨ N % 5 ≠ 0) ∧
        (N % 9 = 0 ∨ N % 9 ≠ 0) ∧
        (N % 10 = 0 ∨ N % 10 ≠ 0) ∧
        (N % 15 = 0 ∨ N % 15 ≠ 0) ∧
        (N % 18 = 0 ∨ N % 18 ≠ 0) ∧
        (N % 30 = 0 ∨ N % 30 ≠ 0)) :
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end two_digit_number_possible_options_l804_804598


namespace find_cubic_expression_l804_804097

theorem find_cubic_expression (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 := by
  sorry

end find_cubic_expression_l804_804097


namespace tire_usage_l804_804726

theorem tire_usage (total_distance : ℕ) (num_tires : ℕ) (active_tires : ℕ) 
  (h1 : total_distance = 45000) 
  (h2 : num_tires = 5) 
  (h3 : active_tires = 4) 
  (equal_usage : (total_distance * active_tires) / num_tires = 36000) : 
  (∀ tire, tire < num_tires → used_miles_per_tire = 36000) := 
by
  sorry

end tire_usage_l804_804726


namespace solution_a_solution_b_l804_804957

def num_squares (n : ℕ) : ℕ := n + 1

def alphonse_jump (pos : ℕ) : ℕ := if pos ≥ 8 then 8 else 1

def beryl_jump (pos : ℕ) : ℕ := if pos ≥ 7 then 7 else 1

def alphonse_time (n : ℕ) : ℕ :=
  let rec jump_time (pos : ℕ) (time : ℕ) : ℕ :=
    if pos < n then jump_time (pos + alphonse_jump (n - pos)) (time + 1) else time
  jump_time 0 0

def beryl_time (n : ℕ) : ℕ :=
  let rec jump_time (pos : ℕ) (time : ℕ) : ℕ :=
    if pos < n then jump_time (pos + beryl_jump (n - pos)) (time + 1) else time
  jump_time 0 0

theorem solution_a : ∃ n > 200, beryl_time n < alphonse_time n := by
  use 209
  show 209 > 200, by decide
  show beryl_time 209 < alphonse_time 209, by sorry

theorem solution_b : ∀ n, beryl_time n ≤ alphonse_time n → n ≤ 209 := by
  intros n h
  by_cases n > 209
  case pos => show false, by sorry
  case neg => exact h

end solution_a_solution_b_l804_804957


namespace initial_ones_count_l804_804291

theorem initial_ones_count (Z O : ℕ) 
  (hZ : Z = 150)
  (steps_to_eliminate_zeroes : ℕ)
  (h_steps : steps_to_eliminate_zeroes = 76)
  (h_min_steps : ∀ (s : ℕ), s < steps_to_eliminate_zeroes → ⋆) -- captures the step requirement complexity
  (h_total_digits : ∀ (remain_zeroes remain_ones : ℕ), ⋆) -- captures the transformation over steps
  : O = 78 := 
sorry

end initial_ones_count_l804_804291


namespace problem_statement_l804_804580

-- Definitions based on conditions
def two_digit_number (N : ℕ) := N >= 10 ∧ N < 100
def divisible_by (a b : ℕ) := a % b = 0
def mistaken_exactly (N : ℕ) := (if divisible_by N 3 then 0 else 1) +
                               (if divisible_by N 4 then 0 else 1) +
                               (if divisible_by N 5 then 0 else 1) +
                               (if divisible_by N 9 then 0 else 1) +
                               (if divisible_by N 10 then 0 else 1) +
                               (if divisible_by N 15 then 0 else 1) +
                               (if divisible_by N 18 then 0 else 1) +
                               (if divisible_by N 30 then 0 else 1)

-- Lean 4 statement
theorem problem_statement (N : ℕ) (h_two_digit : two_digit_number N) (h_mistaken : mistaken_exactly N = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 := sorry

end problem_statement_l804_804580


namespace rectangle_length_width_ratio_l804_804812

-- Define the side lengths of the small squares and the large square
variables (s : ℝ)

-- Define the dimensions of the large square and the rectangle
def large_square_side : ℝ := 5 * s
def rectangle_length : ℝ := 5 * s
def rectangle_width : ℝ := s

-- State and prove the theorem
theorem rectangle_length_width_ratio : rectangle_length s / rectangle_width s = 5 :=
by sorry

end rectangle_length_width_ratio_l804_804812


namespace larger_cuboid_length_is_16_l804_804147

def volume (l w h : ℝ) : ℝ := l * w * h

def cuboid_length_proof : Prop :=
  ∀ (length_large : ℝ), 
  (volume 5 4 3 * 32 = volume length_large 10 12) → 
  length_large = 16

theorem larger_cuboid_length_is_16 : cuboid_length_proof :=
by
  intros length_large eq_volume
  sorry

end larger_cuboid_length_is_16_l804_804147


namespace isosceles_triangle_l804_804887

variables {a b c : ℝ} {α β γ : ℝ}

def is_triangle (a b c : ℝ) (α β γ : ℝ) : Prop :=
  α + β + γ = π ∧ 
  0 < α ∧ α < π ∧ 
  0 < β ∧ β < π ∧ 
  0 < γ ∧ γ < π ∧ 
  a > 0 ∧ b > 0 ∧ c > 0

def given_condition (a b : ℝ) (α β γ : ℝ) : Prop :=
  α + b = tan (γ / 2) * (a * tan α + b * tan β)

theorem isosceles_triangle 
  (h_triangle : is_triangle a b c α β γ)
  (h_condition : given_condition a b α β γ) : 
  a = b :=
sorry

end isosceles_triangle_l804_804887


namespace strawberries_remaining_l804_804295

theorem strawberries_remaining (initial : ℝ) (eaten_yesterday : ℝ) (eaten_today : ℝ) :
  initial = 1.6 ∧ eaten_yesterday = 0.8 ∧ eaten_today = 0.3 → initial - eaten_yesterday - eaten_today = 0.5 :=
by
  sorry

end strawberries_remaining_l804_804295


namespace titu_andreescu_inequality_l804_804721

theorem titu_andreescu_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 :=
sorry

end titu_andreescu_inequality_l804_804721


namespace find_f2_g2_l804_804462

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x
def equation (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x - g x = x^3 + 2^(-x)

theorem find_f2_g2 (f g : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : odd_function g)
  (h3 : equation f g) :
  f 2 + g 2 = -2 :=
sorry

end find_f2_g2_l804_804462


namespace inverse_proportion_function_l804_804848

theorem inverse_proportion_function (a k x : ℝ) 
  (line_passing_through : ∀ x, -2 * x = if x = -2 then a else 0) 
  (P : x = -2 ∧ a = 4) 
  (symmetric_to_y_axis : ∀ P P', P.1 = -2 → P.2 = 4 → P'.1 = 2 ∧ P'.2 = 4) 
  (P' : x = 2 ∧ a = 4)
  (inverse_prop_fn : ∃ k, ∀ x, y = (if x = P'.1 then k / x else 0)) : 
  k = 8 :=
by 
  sorry

end inverse_proportion_function_l804_804848


namespace find_coefficients_l804_804945

noncomputable def ω : ℂ := sorry -- Placeholder for ω, a complex number
axiom ω_fifth_root_of_unity : ω^5 = 1
axiom ω_not_one : ω ≠ 1

def α : ℂ := ω + ω^2
def β : ℂ := ω^3 + ω^4

theorem find_coefficients :
  ∃ (a b : ℂ), (x : ℂ) → (x^2 + a * x + b = 0) ↔ ((x = α) ∨ (x = β)) :=
begin
  use [1, 2],
  sorry -- Proof skipped
end

end find_coefficients_l804_804945


namespace neznika_number_l804_804575

theorem neznika_number (N : ℕ) :
  10 ≤ N ∧ N ≤ 99 ∧
  (divisible_by N 3 ∧ divisible_by N 4 ∧ divisible_by N 5 ∧
   divisible_by N 9 ∧ divisible_by N 10 ∧ divisible_by N 15 ∧
   divisible_by N 18 ∧ divisible_by N 30) →
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end neznika_number_l804_804575


namespace ratio_of_distances_l804_804705

theorem ratio_of_distances
  (w x y : ℝ)
  (hw : w > 0)
  (hx : x > 0)
  (hy : y > 0)
  (h_eq_time : y / w = x / w + (x + y) / (5 * w)) :
  x / y = 2 / 3 :=
by
  sorry

end ratio_of_distances_l804_804705


namespace log_relation_l804_804489

theorem log_relation (a b: ℝ) (h₁: a = log 4 81) (h₂: b = log 2 9) : a = b :=
by
  sorry

end log_relation_l804_804489


namespace at_least_one_casket_made_by_Cellini_son_l804_804179

-- Definitions for casket inscriptions
def golden_box := "The silver casket was made by Cellini"
def silver_box := "The golden casket was made by someone other than Cellini"

-- Predicate indicating whether a box was made by Cellini
def made_by_Cellini (box : String) : Prop :=
  box = "The golden casket was made by someone other than Cellini" ∨ box = "The silver casket was made by Cellini"

-- Our goal is to prove that at least one of the boxes was made by Cellini's son
theorem at_least_one_casket_made_by_Cellini_son :
  (¬ made_by_Cellini golden_box ∧ made_by_Cellini silver_box) ∨ (made_by_Cellini golden_box ∧ ¬ made_by_Cellini silver_box) → (¬ made_by_Cellini golden_box ∨ ¬ made_by_Cellini silver_box) :=
sorry

end at_least_one_casket_made_by_Cellini_son_l804_804179


namespace ratio_of_sum_of_terms_l804_804539

theorem ratio_of_sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 5 / 9) : S 9 / S 5 = 1 := 
  sorry

end ratio_of_sum_of_terms_l804_804539


namespace total_rainbow_nerds_l804_804346

-- Definitions based on the conditions
def num_purple_candies : ℕ := 10
def num_yellow_candies : ℕ := num_purple_candies + 4
def num_green_candies : ℕ := num_yellow_candies - 2

-- The statement to be proved
theorem total_rainbow_nerds : num_purple_candies + num_yellow_candies + num_green_candies = 36 := by
  -- Using the provided definitions to automatically infer
  sorry

end total_rainbow_nerds_l804_804346


namespace option_c_correct_l804_804316

variables (a b c : line) (partial : plane) (A : point)

def lines_in_plane : Prop := b ⊆ partial ∧ c ⊆ partial
def lines_intersect : Prop := b ∩ c = {A}
def line_perpendicular : Prop := a ⊥ b ∧ a ⊥ c

theorem option_c_correct 
  (h1 : lines_in_plane b c partial) 
  (h2 : lines_intersect b c A)
  (h3 : line_perpendicular a b c) :
  a ⊥ partial :=
sorry

end option_c_correct_l804_804316


namespace problem_statement_l804_804224

open Set

theorem problem_statement (k : ℕ) (n : ℕ) (S : Set (ℝ × ℝ)) 
    (h₀ : S.finite) 
    (h₁ : S.card = n) 
    (h₂ : ∀ P ∈ S, ∃ r : ℝ, ∃ T : Set (ℝ × ℝ), T ⊆ S ∧ T.card ≥ k ∧ (∀ Q ∈ T, dist P Q = r)) 
    (h₃ : ∀ P Q R ∈ S, P ≠ Q → P ≠ R → Q ≠ R → ¬Collinear ℝ (P, Q, R)) : 
    k < 1 / 2 + sqrt (2 * n) :=
by
  sorry

end problem_statement_l804_804224


namespace flowers_grew_correctly_l804_804977

noncomputable def front_yard_seeds : ℕ := 120
noncomputable def front_yard_success_rate : ℝ := 0.75
noncomputable def herb_garden_seeds : ℕ := 45
noncomputable def herb_garden_success_rate : ℝ := 0.9
noncomputable def backyard_seeds : ℕ := 80
noncomputable def backyard_success_rate : ℝ := 0.6
noncomputable def mailbox_seeds : ℕ := 60
noncomputable def mailbox_success_rate : ℝ := 0.8

noncomputable def front_yard_flowers := (front_yard_success_rate * front_yard_seeds.toFloat.toReal).toNat
noncomputable def herb_garden_flowers := (herb_garden_success_rate * herb_garden_seeds.toFloat.toReal).toNat
noncomputable def backyard_flowers := (backyard_success_rate * backyard_seeds.toFloat.toReal).toNat
noncomputable def mailbox_flowers := (mailbox_success_rate * mailbox_seeds.toFloat.toReal).toNat

noncomputable def total_flowers : ℕ := front_yard_flowers + herb_garden_flowers + backyard_flowers + mailbox_flowers

theorem flowers_grew_correctly : total_flowers = 226 := by
  sorry

end flowers_grew_correctly_l804_804977


namespace min_positive_period_func_l804_804660

def func (x : Real) : Real := (sin x - sqrt 3 * cos x) * (cos x - sqrt 3 * sin x)

theorem min_positive_period_func : ∃ T > 0, ∀ x, func (x + T) = func x :=
sorry

end min_positive_period_func_l804_804660


namespace speed_conversion_l804_804057

theorem speed_conversion (s : ℚ) (conv_factor : ℚ) (speed_in_ms : ℚ) : 
  s = 3.6 → speed_in_ms = 17 / 36 → speed_in_ms * s = 1.7 :=
by
  intros h_conv_factor h_speed_in_ms
  rw [h_conv_factor, h_speed_in_ms]
  norm_num
  sorry

end speed_conversion_l804_804057


namespace distance_to_highest_point_of_fifth_sphere_l804_804305

noncomputable def distance_from_plane_to_highest_point (r : ℝ) : ℝ :=
  r * (√2 + 2)

theorem distance_to_highest_point_of_fifth_sphere (r : ℝ) :
  let centers_form_square := 
    ∀ (O1 O2 O3 O4 : ℝ × ℝ × ℝ), 
      (O1 = (r, r, r)) ∧ 
      (O2 = (-r, r, r)) ∧ 
      (O3 = (-r, -r, r)) ∧ 
      (O4 = (r, -r, r)) ∧
      ((O1.1 - O3.1)^2 + (O1.2 - O3.2)^2 + (O1.3 - O3.3) ^ 2 = (2 * r * √2)^2) ∧
      ((O2.1 - O4.1)^2 + (O2.2 - O4.2)^2 + (O2.3 - O4.3) ^ 2 = (2 * r * √2)^2),
  let fifth_sphere :=
    ∀ (O5 : ℝ × ℝ × ℝ), 
      (∀ (O1 O2 O3 O4 : ℝ × ℝ × ℝ), (centers_form_square O1 O2 O3 O4) → 
        (O5 = (0, 0, r * (√2 + 1))) ∧
        ((O1.1 - O5.1)^2 + (O1.2 - O5.2)^2 + (O1.3 - O5.3)^2 = (2 * r)^2) ∧
        ((O2.1 - O5.1)^2 + (O2.2 - O5.2)^2 + (O2.3 - O5.3)^2 = (2 * r)^2) ∧
        ((O3.1 - O5.1)^2 + (O3.2 - O5.2)^2 + (O3.3 - O5.3)^2 = (2 * r)^2) ∧
        ((O4.1 - O5.1)^2 + (O4.2 - O5.2)^2 + (O4.3 - O5.3)^2 = (2 * r)^2))
  in
  (distance_from_plane_to_highest_point r) = r * (√2 + 2) :=
sorry

end distance_to_highest_point_of_fifth_sphere_l804_804305


namespace find_y_for_arithmetic_mean_l804_804811

theorem find_y_for_arithmetic_mean :
  (∀ y : ℝ, (8 + 15 + 21 + 7 + 12 + y) / 6 = 15 → y = 27) :=
by
  intro y
  assume h : (8 + 15 + 21 + 7 + 12 + y) / 6 = 15
  -- Proof steps go here
  sorry

end find_y_for_arithmetic_mean_l804_804811


namespace total_jokes_l804_804934

theorem total_jokes (jessy_jokes_saturday : ℕ) (alan_jokes_saturday : ℕ) 
  (jessy_next_saturday : ℕ) (alan_next_saturday : ℕ) (total_jokes_so_far : ℕ) :
  jessy_jokes_saturday = 11 → 
  alan_jokes_saturday = 7 → 
  jessy_next_saturday = 11 * 2 → 
  alan_next_saturday = 7 * 2 → 
  total_jokes_so_far = (jessy_jokes_saturday + alan_jokes_saturday) + (jessy_next_saturday + alan_next_saturday) → 
  total_jokes_so_far = 54 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end total_jokes_l804_804934


namespace minimal_lanterns_required_l804_804523

theorem minimal_lanterns_required (n m : ℕ) : ∃ k, 
  (∀ i j : ℕ, i < n → j < m → (∃ t1 t2 : (Σ i j : ℕ, i ≤ n ∧ j ≤ m),
    (i, j) ∈ t1 ∧ (i, j) ∈ t2)) → 
  k = ((m + 1) * n) / 2 :=
sorry

end minimal_lanterns_required_l804_804523


namespace quadrilateral_is_parallelogram_l804_804546

variables {Point Circle Parallelogram : Type}

-- Define points and the circumcenters
variable (E F G H : Point)
variable (A B C D : Point)
variable [Parallelogram ABCD]
variable (O1 O2 O3 O4 : Circle) -- Assume these are the circumcenters

-- Define segments and sides of the parallelogram
variable (AEH BEF CGF DGH : Point → Point → Point → Prop)

-- Conditions based on the problem statement
axiom circumcenter_circumcircle :
  ∀ {X Y Z : Point}, (AEH X Y Z → is_circumcenter (O1 X Y Z)) ∧
                     (BEF X Y Z → is_circumcenter (O2 X Y Z)) ∧
                     (CGF X Y Z → is_circumcenter (O3 X Y Z)) ∧
                     (DGH X Y Z → is_circumcenter (O4 X Y Z))

-- Prove that the quadrilateral O1O2O3O4 is a parallelogram
theorem quadrilateral_is_parallelogram
  (parallelogramABCD : Parallelogram ABCD)
  (circumcenter_AEH: is_circumcenter O1)
  (circumcenter_BEF: is_circumcenter O2)
  (circumcenter_CGF: is_circumcenter O3)
  (circumcenter_DGH: is_circumcenter O4) :
  is_parallelogram (Quadrilateral O1 O3 O2 O4) := 
sorry

end quadrilateral_is_parallelogram_l804_804546


namespace program_selection_count_l804_804017

theorem program_selection_count :
  let courses := ["English", "Algebra", "Geometry", "History", "Science", "Art", "Latin"]
  let english := 1
  let math_courses := ["Algebra", "Geometry"]
  let science_courses := ["Science"]
  ∃ (programs : Finset (Finset String)) (count : ℕ),
    (count = 9) ∧
    (programs.card = count) ∧
    ∀ p ∈ programs,
      "English" ∈ p ∧
      (∃ m ∈ p, m ∈ math_courses) ∧
      (∃ s ∈ p, s ∈ science_courses) ∧
      p.card = 5 :=
sorry

end program_selection_count_l804_804017


namespace smallest_n_fact_expr_l804_804081

theorem smallest_n_fact_expr : ∃ n : ℕ, (∀ m : ℕ, m = 6 → n! = (n - 4) * (n - 3) * (n - 2) * (n - 1) * n * (n + 1)) ∧ n = 23 := by
  sorry

end smallest_n_fact_expr_l804_804081


namespace lia_quadrilateral_rod_count_l804_804560

theorem lia_quadrilateral_rod_count :
  let rods := {n : ℕ | 1 ≤ n ∧ n ≤ 40}
  let selected_rods := {5, 10, 20}
  let remaining_rods := rods \ selected_rods
  rod_count = 26 ∧ (∃ d ∈ remaining_rods, 
    (5 + 10 + 20) > d ∧ (10 + 20 + d) > 5 ∧ (5 + 20 + d) > 10 ∧ (5 + 10 + d) > 20)
:=
sorry

end lia_quadrilateral_rod_count_l804_804560


namespace part1_part2_l804_804549

-- Define the triangle ABC with internal angles A, B, and C opposite to sides a, b, and c respectively.
variables {a b c : ℝ} {A B C : ℝ}

-- Problem conditions
def condition1 := (a + b + c) * (a - b + c) = a * c
def condition2 := sin A * sin C = (sqrt 3 - 1) / 4

-- The theorems to prove
theorem part1 (h1 : condition1) : B = 120 := by
  sorry

theorem part2 (h1 : condition1) (h2 : condition2) : C = 15 ∨ C = 45 := by
  sorry

end part1_part2_l804_804549


namespace intersection_x_axis_l804_804788

theorem intersection_x_axis : (∃ x, (∃ y1 y2, y1 = 2 ∧ y2 = 6 ∧ (y1 - y2) / (8 - 4) = -1 ∧ y1 = -1 * (x - 8) + 10) ∧ y x = 0) →
  x = 10 :=
by
  sorry

end intersection_x_axis_l804_804788


namespace possible_numbers_l804_804591

theorem possible_numbers (N : ℕ) 
    (h1 : 10 ≤ N) (h2 : N ≤ 99)
    (h3 : (N % 3 = 0) ∧ (N % 4 = 0) ∧ (N % 5 = 0) ∧ (N % 9 = 0) ∧ (N % 10 = 0) ∧ 
          (N % 15 = 0) ∧ (N % 18 = 0) ∧ (N % 30 = 0) ∨ 
          (N % 3 ≠ 0) + (N % 4 ≠ 0) + (N % 5 ≠ 0) + (N % 9 ≠ 0) + 
          (N % 10 ≠ 0) + (N % 15 ≠ 0) + (N % 18 ≠ 0) + (N % 30 ≠ 0) = 4) :
   N = 36 ∨ N = 45 ∨ N = 72 :=
by {
  sorry
}

end possible_numbers_l804_804591


namespace problem_statement_l804_804579

-- Definitions based on conditions
def two_digit_number (N : ℕ) := N >= 10 ∧ N < 100
def divisible_by (a b : ℕ) := a % b = 0
def mistaken_exactly (N : ℕ) := (if divisible_by N 3 then 0 else 1) +
                               (if divisible_by N 4 then 0 else 1) +
                               (if divisible_by N 5 then 0 else 1) +
                               (if divisible_by N 9 then 0 else 1) +
                               (if divisible_by N 10 then 0 else 1) +
                               (if divisible_by N 15 then 0 else 1) +
                               (if divisible_by N 18 then 0 else 1) +
                               (if divisible_by N 30 then 0 else 1)

-- Lean 4 statement
theorem problem_statement (N : ℕ) (h_two_digit : two_digit_number N) (h_mistaken : mistaken_exactly N = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 := sorry

end problem_statement_l804_804579


namespace no_integer_solutions_system_l804_804981

theorem no_integer_solutions_system :
  ¬∃ (x y z : ℤ), x^6 + x^3 + x^3 * y + y = 147^157 ∧ x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := 
sorry

end no_integer_solutions_system_l804_804981


namespace product_of_roots_l804_804218

variable {k m x1 x2 : ℝ}

theorem product_of_roots (h1 : 4 * x1 ^ 2 - k * x1 - m = 0) (h2 : 4 * x2 ^ 2 - k * x2 - m = 0) (h3 : x1 ≠ x2) :
  x1 * x2 = -m / 4 :=
sorry

end product_of_roots_l804_804218


namespace max_ab_is_nine_l804_804490

noncomputable def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- If a > 0, b > 0, and the function f(x) = 4x^3 - ax^2 - 2bx + 2 has an extremum at x = 1, then the maximum value of ab is 9. -/
theorem max_ab_is_nine {a b : ℝ}
  (ha : a > 0) (hb : b > 0)
  (extremum_x1 : deriv (f a b) 1 = 0) :
  a * b ≤ 9 :=
sorry

end max_ab_is_nine_l804_804490


namespace correct_relationship_l804_804753

variable (a b : Type)

theorem correct_relationship (a b : Type) : a ∈ ({a, b}) := 
sorry

end correct_relationship_l804_804753


namespace C₁_equation_C₂_equation_PA_PB_value_l804_804189

-- Definitions for conditions
def C₁_parametric (t : ℝ) : (ℝ × ℝ) :=
  (2 - (3 / 5) * t, -2 + (4 / 5) * t)

def C₂_polar (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = Real.tan θ

-- Prove the ordinary equation of curve C₁ is 4x + 3y - 2 = 0
theorem C₁_equation :
  ∀ (x y t : ℝ), C₁_parametric t = (x, y) → 4 * x + 3 * y - 2 = 0 := by
  sorry

-- Prove the rectangular coordinate equation of curve C₂ is y = x²
theorem C₂_equation :
  ∀ (x y ρ θ : ℝ), (ρ * Real.cos θ = Real.tan θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) → y = x ^ 2 := by
  sorry

-- Given polar coordinates of point P and intersections A, B, prove |PA| * |PB| = 0
theorem PA_PB_value :
  ∀ (P A B : ℝ × ℝ), P = (2, -2) ∧ A = (2, -2) ∧ B = (-1, -2) 
    → (Real.sqrt ((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2)) * (Real.sqrt ((P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2)) = 0 := by
  sorry

end C₁_equation_C₂_equation_PA_PB_value_l804_804189


namespace average_in_all_6_subjects_l804_804763

-- Definitions of the conditions
def average_in_5_subjects : ℝ := 74
def marks_in_6th_subject : ℝ := 104
def num_subjects_total : ℝ := 6

-- Proof that the average in all 6 subjects is 79
theorem average_in_all_6_subjects :
  (average_in_5_subjects * 5 + marks_in_6th_subject) / num_subjects_total = 79 := by
  sorry

end average_in_all_6_subjects_l804_804763


namespace probability_both_girls_l804_804734

def club_probability (total_members girls chosen_members : ℕ) : ℚ :=
  (Nat.choose girls chosen_members : ℚ) / (Nat.choose total_members chosen_members : ℚ)

theorem probability_both_girls (H1 : total_members = 12) (H2 : girls = 7) (H3 : chosen_members = 2) :
  club_probability 12 7 2 = 7 / 22 :=
by {
  sorry
}

end probability_both_girls_l804_804734


namespace amount_after_two_years_l804_804326

-- Definition of initial amount and the rate of increase
def initial_value : ℝ := 32000
def rate_of_increase : ℝ := 0.125
def time_period : ℕ := 2

-- The compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- The proof problem: Prove that after 2 years the amount is 40500
theorem amount_after_two_years : compound_interest initial_value rate_of_increase time_period = 40500 :=
sorry

end amount_after_two_years_l804_804326


namespace contract_hours_calculation_l804_804735

noncomputable def contract_completion_hours (h_day : ℝ) : ℝ :=
  let total_days := 46
  let men_initial := 117
  let hours_per_day := 8
  let days_worked := 33
  let fraction_completed := 4 / 7
  let men_additional := 81
  let total_men := men_initial + men_additional
  let days_remaining := total_days - days_worked

  -- Total man-hours required
  let total_man_hours := men_initial * hours_per_day * total_days

  -- Man-hours used so far
  let man_hours_used := fraction_completed * total_man_hours

  -- Man-hours remaining
  let man_hours_remaining := (3 / 7) * total_man_hours

  -- Equation to find hours per day
  man_hours_remaining = total_men * h_day * days_remaining

theorem contract_hours_calculation :
  contract_completion_hours 7.16 = true :=
by
  sorry

end contract_hours_calculation_l804_804735


namespace seashells_after_giving_away_l804_804195

-- Define the given conditions
def initial_seashells : ℕ := 79
def given_away_seashells : ℕ := 63

-- State the proof problem
theorem seashells_after_giving_away : (initial_seashells - given_away_seashells) = 16 :=
  by 
    sorry

end seashells_after_giving_away_l804_804195


namespace angle_EBD_l804_804499

theorem angle_EBD (A B C D E : Prop)
  (BE_bisects_ABC_externally : ∀ x, is_external_bisector x BE B)
  (angle_BAC_eq_a : ∠ BAC = a)
  (angle_BCA_eq_c : ∠ BCA = c)
  (angle_DBC_eq_90 : ∠ DBC = 90) :
  ∠ EBD = 180 - a :=
by
  sorry

end angle_EBD_l804_804499


namespace alien_saturday_exclamation_l804_804026

-- Define the initial exclamations
def monday_exclamation := "A!"
def tuesday_exclamation := "AU!"
def wednesday_exclamation := "AUUA!"
def thursday_exclamation := "AUUAUAAU!"

-- Define the function that generates the next exclamation based on the current one's pattern
def next_exclamation (excl : String) : String :=
  let half_len := excl.length / 2
  let first_half := excl.take half_len
  let second_half := excl.drop half_len
  let reflected_half := second_half.map (λ c, if c = 'A' then 'U' else if c = 'U' then 'A' else c)
  first_half ++ reflected_half

-- Define the exclamations for each day starting from Thursday
def friday_exclamation := next_exclamation thursday_exclamation
def saturday_exclamation := next_exclamation friday_exclamation

-- Define the expected solution for Saturday's exclamation
def expected_saturday_exclamation := "AUUAUAUAAUAUAUAAUAA!"

-- Prove that the calculated exclamation for Saturday matches the expected one
theorem alien_saturday_exclamation :
  saturday_exclamation = expected_saturday_exclamation :=
by sorry

end alien_saturday_exclamation_l804_804026


namespace inequality_proof_l804_804211

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) : ab < 1 ∧ 1 < (a^2 + b^2) / 2 :=
by
  sorry

end inequality_proof_l804_804211


namespace who_made_statements_and_fate_l804_804685

namespace IvanTsarevichProblem

-- Define the characters and their behaviors
inductive Animal
| Bear : Animal
| Fox : Animal
| Wolf : Animal

def always_true (s : Prop) : Prop := s
def always_false (s : Prop) : Prop := ¬s
def alternates (s1 s2 : Prop) : Prop := s1 ∧ ¬s2

-- Statements made by the animals
def statement1 (save_die : Bool) : Prop := save_die = true
def statement2 (safe_sound_save : Bool) : Prop := safe_sound_save = true
def statement3 (safe_lose : Bool) : Prop := safe_lose = true

-- Analyze truth based on behaviors
noncomputable def belongs_to (a : Animal) (s : Prop) : Prop :=
  match a with
  | Animal.Bear => always_true s
  | Animal.Fox => always_false s
  | Animal.Wolf =>
    match s with
    | ss => alternates (ss = true) (ss = false)

-- Given conditions
axiom h1 : statement1 false -- Fox lies, so "You will save the horse. But you will die." is false
axiom h2 : statement2 false -- Wolf alternates, so "You will stay safe and sound. And you will save the horse." is a mix
axiom h3 : statement3 true  -- Bear tells the truth, so "You will survive. But you will lose the horse." is true

-- Conclusion: Animal who made each statement
theorem who_made_statements_and_fate : 
  belongs_to Animal.Fox (statement1 false) ∧ 
  belongs_to Animal.Wolf (statement2 false) ∧ 
  belongs_to Animal.Bear (statement3 true) ∧ 
  (¬safe_lose) := sorry

end IvanTsarevichProblem

end who_made_statements_and_fate_l804_804685


namespace base_five_sum_of_product_product_base_five_sum_is_correct_l804_804663

theorem base_five_sum_of_product (a b : ℕ) (a_10 b_10 : ℕ) (c : ℕ) (c_10 : ℕ) :
  a = 17 ∧ b = 8 ∧ a_10 = 3 * 5 + 2 ∧ b_10 = 1 * 5 + 3 ∧ c_10 = a_10 * b_10 ∧ c = c_10 := 
begin
  sorry,
end

theorem product_base_five_sum_is_correct (a b : ℕ) (a_10 b_10 : ℕ) (c : ℕ) (c_10 : ℕ) :
  a = 17 → b = 8 → a_10 = 3 * 5 + 2 → b_10 = 1 * 5 + 3 → c_10 = a_10 * b_10 → c = 521 →
  (5 + 2 + 1 = 8) :=
by
  intros h1 h2 h3 h4 h5
  have h : (521 : ℕ) = 136 % 25 * 5^2 + 11 % 5 * 5^1 + 1 * 5 ^ 0 := by sorry
  sorry

end base_five_sum_of_product_product_base_five_sum_is_correct_l804_804663


namespace possible_numbers_l804_804606

theorem possible_numbers (N : ℕ) (h_digit : 10 ≤ N ∧ N ≤ 99)
  (h_claimed_divisors : ∀ d ∈ [3, 4, 5, 9, 10, 15, 18, 30], d ∣ N ∨ ¬ d ∣ N)
  (h_mistakes : Nat.countp (λ d, ¬ d ∣ N) [3, 4, 5, 9, 10, 15, 18, 30] = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 :=
by
  sorry

end possible_numbers_l804_804606


namespace arrangement_count_l804_804988

-- Definitions related to the problem conditions.
def leftmost_condition (p : ℕ → Prop) (c : list ℕ) : Prop :=
  p (c.head!)

def rightmost_condition (p : ℕ → Prop) (c : list ℕ) : Prop :=
  p (c.getLast sorry)

def total_arrangements (c : list ℕ) : ℕ :=
  if h : leftmost_condition (λ x, x = 0 ∨ x = 1) c ∧ ¬rightmost_condition (λ x, x = 0) c then
    216
  else
    0

-- Proving that the number of different arrangements is 216
theorem arrangement_count : ∃ c : list ℕ, total_arrangements c = 216 :=
by {
  use [0, 1, 2, 3, 4, 5],
  simp [total_arrangements, leftmost_condition, rightmost_condition],
  sorry
}

end arrangement_count_l804_804988


namespace min_value_diff_l804_804837

theorem min_value_diff (P Q F1 F : ℝ × ℝ)
  (hP_on_ellipse : P.1 ^ 2 / 4 + P.2 ^ 2 / 3 = 1)
  (hQ_proj_line : ∃ t : ℝ, Q = (P.1 + 4 * t, P.2 + 3 * t) ∧ 4 * P.1 + 3 * P.2 = 21)
  (hFocus : F = (2, 0)) :
  ∃ Q, P.1 ^ 2 / 4 + P.2 ^ 2 / 3 = 1 ∧
       Q = (P.1 + t * 4, P.2 + t * 3) ∧ 
       4 * Q.1 + 3 * Q.2 - 21 = 0 ∧ 
       min_val = 1 :=
sorry

end min_value_diff_l804_804837


namespace sum_of_real_solutions_eq_32_over_7_l804_804434

theorem sum_of_real_solutions_eq_32_over_7 :
  (∑ x in (finset.filter (λ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x)) finset.univ), x) = 32 / 7 :=
by
  sorry

end sum_of_real_solutions_eq_32_over_7_l804_804434


namespace class_B_more_uniform_l804_804730

def average_height : ℝ := 1.65
def variance_A : ℝ := 6
def variance_B : ℝ := 3.2

theorem class_B_more_uniform : variance_A > variance_B → "Class B" :=
by sorry

end class_B_more_uniform_l804_804730


namespace sequence_term_formula_l804_804828

theorem sequence_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, S n = n ^ 2) →
  (∀ n : ℕ, n > 0 → S n = a n + S (n - 1)) →
  a = (λ n, 2 * n - 1) :=
by
  intro hS hRec
  funext n
  sorry

end sequence_term_formula_l804_804828


namespace smallest_sum_in_arithmetic_sequence_l804_804086

noncomputable def smallest_S (a : ℕ → ℝ) (S : ℕ → ℝ) : ℝ :=
  if a_3 + a_8 > 0 ∧ S 9 < 0 then S 5 else sorry

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

theorem smallest_sum_in_arithmetic_sequence (h₁ : ∀ n, S n = ∑ i in range (n + 1), a i)
                                            (h₂ : a 3 + a 8 > 0)
                                            (h₃ : S 9 < 0) :
  smallest_S a S = S 5 :=
by
  -- This is where the proof would go.
  sorry

end smallest_sum_in_arithmetic_sequence_l804_804086


namespace national_day_2020_is_thursday_l804_804160

-- Condition: 2019 National Day was on a Tuesday
def day_of_week_2019 := "Tuesday"

-- Condition: 2020 is a leap year
def is_leap_year_2020 := true

-- Constants representing days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Helper function to calculate the next day's value
def next_day (d : Day) : Day :=
match d with
| Sunday    => Monday
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
end

-- Helper function to calculate the day after a given number of days
def add_days (d : Day) (n : ℕ) : Day :=
match n % 7 with
| 0 => d
| 1 => next_day d
| 2 => next_day (next_day d)
| 3 => next_day (next_day (next_day d))
| 4 => next_day (next_day (next_day (next_day d)))
| 5 => next_day (next_day (next_day (next_day (next_day d))))
| 6 => next_day (next_day (next_day (next_day (next_day (next_day d)))))
| _ => d -- This case will never happen because of the % 7 constraint
end

-- Theorem to state that National Day in 2020 will be on Thursday
theorem national_day_2020_is_thursday :
    add_days Tuesday 2 = Thursday :=
by
exact rfl

end national_day_2020_is_thursday_l804_804160


namespace number_of_intersection_elements_l804_804628

def X : set ℤ := { x | 1 ≤ x ∧ x ≤ 12 }
def Y : set ℤ := { y | 0 ≤ y ∧ y ≤ 20 }

theorem number_of_intersection_elements : (X ∩ Y).to_finset.card = 12 :=
by
  sorry

end number_of_intersection_elements_l804_804628


namespace find_angle_find_area_l804_804106

noncomputable def degree_to_radian (d : ℝ) := d * π / 180

theorem find_angle {A B C : ℝ}
  (h : 2 * (Real.sin (B + C))^2 = Real.sqrt 3 * Real.sin (2 * A)) : A = degree_to_radian 60 :=
by {
  sorry
}

theorem find_area {A B C BC AC : ℝ}
  (h₁ : B + C = π/3)
  (h₂ : BC = 7)
  (h₃ : AC = 5)
  (h₄ : A = degree_to_radian 60) :
  let S := 0.5 * AC * BC * Real.sin A in
  S = 10 * Real.sqrt 3 :=
by {
  sorry
}

end find_angle_find_area_l804_804106


namespace number_of_sets_l804_804203

variable {n : ℕ} (x : Fin n → ℝ)

def s (I : Finset (Fin n)) : ℝ :=
  I.sum (λ i => x i)

theorem number_of_sets {x : Fin n → ℝ} (h : (Finset.powerset (Finset.univ.val)).image (λ I => s x I)).card ≥ 1.8^n :
  (Finset.filter (λ I => s x I = 2019) (Finset.powerset (Finset.univ.val))).card ≤ 1.7^n :=
sorry

end number_of_sets_l804_804203


namespace sin_A_triangle_abc_l804_804902

open Real

noncomputable def sin_A_tri (A B C : ℝ) (a b c : ℝ) (h1 : B = π / 4) (h2 : (c * sin A) = 1 / 3 * c) : Real :=
  sin A

theorem sin_A_triangle_abc (a b c : ℝ) (A B C : ℝ) 
  (h_B_eq : B = π / 4) 
  (h_height : (a * sin A = 1/3 * a))
   : sin A = 3 * sqrt 10 / 10 := 
by
  sorry

end sin_A_triangle_abc_l804_804902


namespace two_digit_number_possible_options_l804_804596

theorem two_digit_number_possible_options
  (N : ℕ)
  (h1 : 10 ≤ N ∧ N < 100)
  (h2 : (N % 3 = 0 ∨ N % 3 ≠ 0) ∧
        (N % 4 = 0 ∨ N % 4 ≠ 0) ∧
        (N % 5 = 0 ∨ N % 5 ≠ 0) ∧
        (N % 9 = 0 ∨ N % 9 ≠ 0) ∧
        (N % 10 = 0 ∨ N % 10 ≠ 0) ∧
        (N % 15 = 0 ∨ N % 15 ≠ 0) ∧
        (N % 18 = 0 ∨ N % 18 ≠ 0) ∧
        (N % 30 = 0 ∨ N % 30 ≠ 0)) :
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end two_digit_number_possible_options_l804_804596


namespace simplify_sub_polynomials_l804_804631

def f (r : ℝ) : ℝ := 2 * r^3 + r^2 + 5 * r - 4
def g (r : ℝ) : ℝ := r^3 + 3 * r^2 + 7 * r - 2

theorem simplify_sub_polynomials (r : ℝ) : f r - g r = r^3 - 2 * r^2 - 2 * r - 2 := by
  sorry

end simplify_sub_polynomials_l804_804631


namespace domain_of_f_l804_804051

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.floor (2 * x^2 - 10 * x + 16))

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = ∞} = {x : ℝ | x ∈ set.Iic 2.5 ∪ set.Ici 3} :=
by
  sorry

end domain_of_f_l804_804051


namespace doctor_lawyer_ratio_l804_804504

variables {d l : ℕ} -- Number of doctors and lawyers

-- Conditions
def avg_age_group (d l : ℕ) : Prop := (40 * d + 55 * l) / (d + l) = 45

-- Theorem: Given the conditions, the ratio of doctors to lawyers is 2:1.
theorem doctor_lawyer_ratio (hdl : avg_age_group d l) : d / l = 2 :=
sorry

end doctor_lawyer_ratio_l804_804504


namespace find_x_l804_804329

theorem find_x (x y : ℤ) 
  (h1 : 3 ^ x * 4 ^ y = 19683) 
  (h2 : x - y = 9) : 
  x = 9 := 
sorry

end find_x_l804_804329


namespace sum_ratio_is_one_l804_804537

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)

-- Conditions
def arithmetic_sequence (a_n S_n : ℕ → ℝ) := ∀ n : ℕ, S_n = (n * (a_n 1 + a_n n)) / 2
def ratio_condition (a_n : ℕ → ℝ) := a_n 5 / a_n 3 = 5 / 9

-- The theorem to prove
theorem sum_ratio_is_one (a_n S_n : ℕ → ℝ) 
  [arithmetic_sequence a_n S_n] 
  [ratio_condition a_n] : 
  S_n 9 / S_n 5 = 1 := 
sorry

end sum_ratio_is_one_l804_804537


namespace line_m_parallel_n_l804_804615

/-- Plane definitions -/
variables {Point Line Plane : Type}
variables (α β γ : Plane) (m n : Line) (p : Point)

-- Conditions
variables (parallel_planes : ∀ {π₁ π₂ : Plane}, π₁ = α → π₂ = β → parallel π₁ π₂)
variables (intersect_alpha_gamma : ∀ {λ : Line}, λ = m → intersects α γ)
variables (intersect_beta_gamma : ∀ {λ : Line}, λ = n → intersects β γ)

-- Positional relationship to prove
theorem line_m_parallel_n : parallel m n := sorry

end line_m_parallel_n_l804_804615


namespace x_coordinate_of_point_on_parabola_l804_804498

-- Conditions
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1
def distance_to_focus (P : ℝ × ℝ) : Prop :=
  let focus := (2 : ℝ, 0 : ℝ)
  (Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 20)

-- Proof statement
theorem x_coordinate_of_point_on_parabola :
  ∃ (x : ℝ), ∃ (y : ℝ), parabola (x, y) ∧ distance_to_focus (x, y) ∧ x = 18 := by
  sorry

end x_coordinate_of_point_on_parabola_l804_804498


namespace max_val_of_xy_collinear_l804_804869

section
variables {x y : ℝ}
def collinear_vectors : Prop := ∃ k : ℝ, (1, x^2) = k • (-2, y^2 - 2) 

theorem max_val_of_xy_collinear (h : collinear_vectors) : 
  ∃ (max_val : ℝ), max_val = xy ∧ max_val = (√2) / 2 :=
sorry
end

end max_val_of_xy_collinear_l804_804869


namespace billy_sisters_count_l804_804385

theorem billy_sisters_count 
  (S B : ℕ) -- S is the number of sisters, B is the number of brothers
  (h1 : B = 2 * S) -- Billy has twice as many brothers as sisters
  (h2 : 2 * (B + S) = 12) -- Billy gives 2 sodas to each sibling to give out the 12 pack
  : S = 2 := 
  by sorry

end billy_sisters_count_l804_804385


namespace equation_of_line_l_l804_804552

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

-- Define the line equations as predicates
def line1 (x y : ℝ) : Prop := 2 * x - y = 1
def line2 (x y : ℝ) : Prop := 2 * x + y = 11

-- Define the center of the circle
def center : ℝ × ℝ := (3, 5)

-- Define the condition for midpoint and intersection
def is_midpoint (A B P : ℝ × ℝ) : Prop := 
  A = ((B.1 + P.1) / 2, (B.2 + P.2) / 2)

def intersects_circle (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ t, circle (line t).1 (line t).2

theorem equation_of_line_l :
  ∀ (A B P : ℝ × ℝ),
  line1 (A.1) (A.2) ∨ line2 (A.1) (A.2) →
  (A = center) → 
  intersects_circle (λ t, (A.1 + t * (P.1 - A.1), A.2 + t * (P.2 - A.2))) →
  (is_midpoint A B P) →
  (line1 (A.1) (P.2) ∨ line2 (B.1) (P.2)) :=
by
  sorry

end equation_of_line_l_l804_804552


namespace businessmen_count_l804_804716

theorem businessmen_count (n : ℕ) (d : ℕ → ℕ) 
  (H1 : ∀ i : ℕ, 1 ≤ i ∧ i < n → 
    d i = 3 * ((∑ j in finset.range (n - 1), d (j + 1)) / (n - i)))
  (H2 : d 1 = 190 * d n) : 
  n = 19 :=
by
  sorry

end businessmen_count_l804_804716


namespace geometric_sequence_sums_l804_804851

open Real

theorem geometric_sequence_sums (S T R : ℝ)
  (h1 : ∃ a r, S = a * (1 + r))
  (h2 : ∃ a r, T = a * (1 + r + r^2 + r^3))
  (h3 : ∃ a r, R = a * (1 + r + r^2 + r^3 + r^4 + r^5)) :
  S^2 + T^2 = S * (T + R) :=
by
  sorry

end geometric_sequence_sums_l804_804851


namespace proposition_2_and_3_correct_l804_804246

variables {m n : Line} {α β : Plane}

-- Definitions based on conditions
axiom parallel_line_plane {l : Line} {p : Plane} : Prop
axiom perpendicular_line_plane {l : Line} {p : Plane} : Prop
axiom parallel_planes {p1 p2 : Plane} : Prop
axiom perpendicular_planes {p1 p2 : Plane} : Prop
axiom parallel_lines {l1 l2 : Line} : Prop
axiom perpendicular_lines {l1 l2 : Line} : Prop

-- The main theorem to prove based on the conditions
theorem proposition_2_and_3_correct :
  ((parallel_line_plane m α) ∧ (parallel_line_plane n β) ∧ (parallel_planes α β) → ¬(parallel_lines m n)) ∧
  ((perpendicular_line_plane m α) ∧ (perpendicular_line_plane n β) ∧ (perpendicular_planes α β) → perpendicular_lines m n) ∧
  ((perpendicular_line_plane m α) ∧ (parallel_line_plane n β) ∧ (parallel_planes α β) → perpendicular_lines m n) ∧
  ((parallel_line_plane m α) ∧ (perpendicular_line_plane n β) ∧ (perpendicular_planes α β) → ¬(parallel_lines m n)) :=
by sorry

end proposition_2_and_3_correct_l804_804246


namespace min_sugar_l804_804384

theorem min_sugar (f s : ℝ) (h₁ : f ≥ 8 + (3/4) * s) (h₂ : f ≤ 2 * s) : s ≥ 32 / 5 :=
sorry

end min_sugar_l804_804384


namespace eval_expression_l804_804793

theorem eval_expression (a b c d e: ℕ) (h1 : a = 8) (h2 : b = 6) (h3 : c = 16) (h4 : d = -3) (h5 : e = -12) :
  ((a^b) * (c^d)) / (2^e) = 2^18 := by
  sorry

end eval_expression_l804_804793


namespace least_distance_traveled_l804_804410

theorem least_distance_traveled :
  let n := 8
  let R := 50
  let diameter := 2 * R
  let distance_2_points_apart := 2 * R  * Real.sin (2 * π / n)
  let distance_3_points_apart := 2 * R  * Real.sin (3 * π / n)
  let total_distance_per_person := diameter + 2 * distance_2_points_apart + 2 * distance_3_points_apart
  let total_distance_all_friends := n * total_distance_per_person
  in total_distance_all_friends = 800 + 800 * Real.sqrt 2 + 800 * Real.sqrt (2 + Real.sqrt 2) := 
sorry

end least_distance_traveled_l804_804410


namespace tangent_sum_l804_804777

noncomputable def f (x : ℝ) : ℝ := max (-7 * x - 25) (max (x^2 - 1) (5 * x + 7))

def p (x : ℝ) : ℝ := a * x^2 + b * x + c
-- Assuming p(x) is a quadratic polynomial

theorem tangent_sum (x1 x2 x3 : ℝ) (a b c : ℝ) (h_tangent1 : ∀ x, p x - (-7 * x - 25) = a * (x - x1)^2)
  (h_tangent2 : ∀ x, p x - (x^2 - 1) = a * (x - x2)^2)
  (h_tangent3 : ∀ x, p x - (5 * x + 7) = a * (x - x3)^2) :
  x1 + x2 + x3 = -16 / 3 := by
  sorry

end tangent_sum_l804_804777


namespace circle_eq_and_extremum_l804_804102

-- Definitions for points A and B 
def A := (0 : ℝ, 2 : ℝ)
def B := (1 : ℝ, 1 : ℝ)

-- Definition for the line equation where the center C lies
def lineC (x y : ℝ) : Prop := x + y + 5 = 0

-- Definition for the standard form of the circle
def standard_eq_circle (x y : ℝ) : Prop := (x + 3)^2 + (y + 2)^2 = 25

-- Main theorem statement for proving the standard equation of the circle
theorem circle_eq_and_extremum :
  (∃ (C : ℝ × ℝ), (C.fst + C.snd + 5 = 0)
    ∧ (∀ (A B : ℝ × ℝ), (A = (0,2)) ∧ (B = (1,1)) 
    ∧ (∃ r : ℝ, standard_eq_circle C.fst C.snd 
    ∧ (dist A C = r) ∧ (dist B C = r)) 
    ∧ (∀ (P : ℝ × ℝ), standard_eq_circle P.fst P.snd 
    → (3 * P.fst - 4 * P.snd = 24 ∨ 3 * P.fst - 4 * P.snd = -26)))) :=
sorry

end circle_eq_and_extremum_l804_804102


namespace neznaika_mistake_correct_numbers_l804_804608

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end neznaika_mistake_correct_numbers_l804_804608


namespace space_left_over_l804_804708

theorem space_left_over (D B : ℕ) (wall_length desk_length bookcase_length : ℝ) (h_wall : wall_length = 15)
  (h_desk : desk_length = 2) (h_bookcase : bookcase_length = 1.5) (h_eq : D = B)
  (h_max : 2 * D + 1.5 * B ≤ wall_length) :
  ∃ w : ℝ, w = wall_length - (D * desk_length + B * bookcase_length) ∧ w = 1 :=
by
  sorry

end space_left_over_l804_804708


namespace diameter_of_circular_field_l804_804803

theorem diameter_of_circular_field (cost_per_meter total_cost : Real) (h1 : cost_per_meter = 1.50) (h2 : total_cost = 150.80) : Real :=
  let circumference := total_cost / cost_per_meter
  let diameter := circumference / Real.pi
  diameter

example : diameter_of_circular_field 1.50 150.80 = 32 := by
  rw [diameter_of_circular_field, Real.pi]
  norm_num
  sorry

end diameter_of_circular_field_l804_804803


namespace length_AB_l804_804556

theorem length_AB :
  ∀ (A B : ℝ × ℝ) (k : ℝ),
    (A.2 = k * A.1 - 2) ∧ (B.2 = k * B.1 - 2) ∧ (A.2^2 = 8 * A.1) ∧ (B.2^2 = 8 * B.1) ∧
    ((A.1 + B.1) / 2 = 2) →
  dist A B = 2 * Real.sqrt 15 :=
by
  sorry

end length_AB_l804_804556


namespace sum_sin_squared_even_degrees_l804_804394

theorem sum_sin_squared_even_degrees : 
  (∑ i in (finset.range 46).map (function.embedding.subtype $ λ i, 2 * i ≤ 90), (Real.sin (2 * i : ℝ)).pow 2) = 23 :=
by
  sorry

end sum_sin_squared_even_degrees_l804_804394


namespace evaluate_expression_l804_804794

theorem evaluate_expression : 
  let a := 3 
  let b := 2 
  (a^2 + b)^2 - (a^2 - b)^2 + 2*a*b = 78 := 
by
  let a := 3
  let b := 2
  sorry

end evaluate_expression_l804_804794


namespace sum_lent_is_1500_l804_804706

/--
A person lent a certain sum of money at 4% per annum at simple interest.
In 4 years, the interest amounted to Rs. 1260 less than the sum lent.
Prove that the sum lent was Rs. 1500.
-/
theorem sum_lent_is_1500
  (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ)
  (h1 : r = 4) (h2 : t = 4)
  (h3 : I = P - 1260)
  (h4 : I = P * r * t / 100):
  P = 1500 :=
by
  sorry

end sum_lent_is_1500_l804_804706


namespace cinema_movie_choice_l804_804976

theorem cinema_movie_choice :
  ∃ m, (m ≠ 'B' ∧ m ≠ 'A' ∧ m ≠ 'C' ∧ m ≠ 'E') ∧ m = 'D' :=
by {
  let movies := ['A', 'B', 'C', 'D', 'E']
  let xiaoZhao := λ m, m ≠ 'B'
  let xiaoZhang := λ m, m ≠ 'A'
  let xiaoLi := λ m, m ≠ 'C'
  let xiaoLiu := λ m, m ≠ 'E'

  have sol := 'D'
  split,
  apply and.intro,
  {
    split,
    exact (xiaoZhao sol),
    split,
    exact (xiaoZhang sol),
    split,
    exact (xiaoLi sol),
    exact (xiaoLiu sol)
  },
  exact eq.refl sol
} sorry

end cinema_movie_choice_l804_804976


namespace matrix_product_identity_l804_804034

theorem matrix_product_identity (matrices : List (Matrix (Fin 2) (Fin 2) ℝ))
  (h_matrices : matrices = List.map (λ a, !![1, a; 0, 1]) (List.range' 1 3 34)):
  (matrices.foldl (\(acc m : Matrix (Fin 2) (Fin 2) ℝ), acc ⬝ m) (1 : Matrix (Fin 2) (Fin 2) ℝ))
  = !![1, 1717; 0, 1] := 
  sorry

end matrix_product_identity_l804_804034


namespace power_of_i_l804_804118

theorem power_of_i (i : ℂ) (h₀ : i^2 = -1) : i^(2016) = 1 :=
by {
  -- Proof will go here
  sorry
}

end power_of_i_l804_804118


namespace functional_equality_l804_804796

theorem functional_equality (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f y + x ^ 2 + 1) + 2 * x = y + (f (x + 1)) ^ 2) →
  (∀ x : ℝ, f x = x) := 
by
  intro h
  sorry

end functional_equality_l804_804796


namespace tangency_distance_l804_804681

noncomputable def distance_between_tangency_points (r1 r2 r3 : ℝ) (touch_condition : r1 = 1 ∧ r2 = 2 ∧ r3 = 5) : ℝ :=
  let O1O2 := r1 + r2
  let O1O3 := r1 + r3
  let O2O3 := r2 + r3
  let s := (O1O2 + O1O3 + O2O3) / 2
  let S_∆O1O2O3 := Real.sqrt (s * (s - O1O2) * (s - O1O3) * (s - O2O3))
  let AB := 2 * Real.sqrt (r1 * r2)
  let AC := 2 * Real.sqrt (r1 * r3)
  let BC := 2 * Real.sqrt (r2 * r3)
  let s_ABC := (AB + AC + BC) / 2
  let S_∆ABC := Real.sqrt (s_ABC * (s_ABC - AB) * (s_ABC - AC) * (s_ABC - BC))
  let phi_div_2 := Real.arccos (S_∆ABC / S_∆O1O2O3)
  let AD := 2 * r1 * Real.cos phi_div_2
  let distance := AD / 2
  distance

theorem tangency_distance : distance_between_tangency_points 1 2 5 (by simp) = Real.sqrt (31 / 10) :=
sorry

end tangency_distance_l804_804681


namespace odd_numbers_last_4_digits_l804_804756

theorem odd_numbers_last_4_digits (n : ℕ) (h_odd : n % 2 = 1) (h_bound : n < 10000) :
  let m := n^9 % 10000 in
  (n < m → ∃ k (h_odd_k : k % 2 = 1) (h_bound_k : k < 10000), k < 10000 ∧ k^9 % 10000 < k) ∧
  (n > m → ∃ j (h_odd_j : j % 2 = 1) (h_bound_j : j < 10000), j < 10000 ∧ j^9 % 10000 > j) := sorry

end odd_numbers_last_4_digits_l804_804756


namespace polynomial_nonnegative_l804_804618

theorem polynomial_nonnegative (x : ℝ) : x^4 - x^3 + 3x^2 - 2x + 2 ≥ 0 := 
by 
  sorry

end polynomial_nonnegative_l804_804618


namespace no_solution_exists_l804_804094

theorem no_solution_exists (p : ℝ) : (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) = (x - p) / (x - 8) → false) ↔ p = 7 :=
by sorry

end no_solution_exists_l804_804094


namespace probability_largest_num_l804_804337

/-
Given \(\frac{n(n + 1)}{2}\) distinct numbers arranged into \(n\) rows where the \(i\)-th row has \(i\) numbers, 
prove that the probability that the largest number in each row is smaller than the largest number in each subsequent row is 
\(\frac{2^n}{(n+1)!}\). 
-/
theorem probability_largest_num (n : ℕ) : 
  let num = (n * (n + 1)) / 2;
  let rows := List.range (n + 1) in
  ∀ (arrangement : List ℕ),
  arrangement.length = num →
  (∀ i, ∃ row : List ℕ, row.length = i) →
  let probability := (2^n : ℚ) / (n+1)! in
  ∃ arrangement, (arrangement.length = num) ∧ 
  (∀ i < n, arrangement.row i < arrangement.row (i + 1)) → 
  probability = (2^n / (n + 1)!) := sorry

end probability_largest_num_l804_804337


namespace two_digit_number_possible_options_l804_804593

theorem two_digit_number_possible_options
  (N : ℕ)
  (h1 : 10 ≤ N ∧ N < 100)
  (h2 : (N % 3 = 0 ∨ N % 3 ≠ 0) ∧
        (N % 4 = 0 ∨ N % 4 ≠ 0) ∧
        (N % 5 = 0 ∨ N % 5 ≠ 0) ∧
        (N % 9 = 0 ∨ N % 9 ≠ 0) ∧
        (N % 10 = 0 ∨ N % 10 ≠ 0) ∧
        (N % 15 = 0 ∨ N % 15 ≠ 0) ∧
        (N % 18 = 0 ∨ N % 18 ≠ 0) ∧
        (N % 30 = 0 ∨ N % 30 ≠ 0)) :
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end two_digit_number_possible_options_l804_804593


namespace stormi_mowing_charge_l804_804990

theorem stormi_mowing_charge (cars_washed : ℕ) (car_wash_price : ℕ) (lawns_mowed : ℕ) (bike_cost : ℕ) (money_needed_more : ℕ) 
  (total_from_cars : ℕ := cars_washed * car_wash_price)
  (total_earned : ℕ := bike_cost - money_needed_more)
  (earned_from_lawns : ℕ := total_earned - total_from_cars) :
  cars_washed = 3 → car_wash_price = 10 → lawns_mowed = 2 → bike_cost = 80 → money_needed_more = 24 → earned_from_lawns / lawns_mowed = 13 := 
by
  sorry

end stormi_mowing_charge_l804_804990


namespace neznaika_mistake_correct_numbers_l804_804609

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end neznaika_mistake_correct_numbers_l804_804609


namespace p_iff_q_l804_804833

theorem p_iff_q (a b : ℝ) : (a > b) ↔ (a^3 > b^3) :=
sorry

end p_iff_q_l804_804833


namespace fraction_equiv_to_repeating_decimal_l804_804805

theorem fraction_equiv_to_repeating_decimal :
  let a := (568 / 1000 : ℚ)
  let r := (1 / 1000 : ℚ)
  0.568.repeating_decimal = (568 / 999 : ℚ) :=
by
  sorry

end fraction_equiv_to_repeating_decimal_l804_804805


namespace sum_of_solutions_to_the_equation_l804_804426

noncomputable def sum_of_real_solutions : ℚ := 
  have h : (∀ x : ℚ, (x - 3) * (x^2 - 12 * x) = (x - 6) * (x^2 + 5 * x + 2)) → 
             (∀ x : ℚ, 14 * x^2 - 64 * x - 12 = 0) := 
  by sorry,
  (32 : ℚ) / 7

theorem sum_of_solutions_to_the_equation :
  (∀ x : ℚ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x) ↔ 
            14 * x^2 - 64 * x - 12 = 0) → 
              sum_of_real_solutions = 32 / 7 :=
by sorry

end sum_of_solutions_to_the_equation_l804_804426


namespace find_real_solutions_l804_804074

def equation (x : ℝ) : Prop := x^4 + (3 - x)^4 = 82

theorem find_real_solutions : 
  {x : ℝ // equation x} = {x | x ≈ 3.22 ∨ x ≈ -0.22} :=
by 
  sorry

end find_real_solutions_l804_804074


namespace tenth_root_of_unity_l804_804064

theorem tenth_root_of_unity (n : ℕ) (h₀ : n ≤ 9) :
  (Complex.tan (Real.pi / 4) + Complex.i) / (Complex.tan (Real.pi / 4) - Complex.i)
  = Complex.exp (Complex.i * 2 * Real.pi * ↑n / 10) → n = 3 := by
  sorry

end tenth_root_of_unity_l804_804064


namespace savings_by_paying_cash_l804_804971

def cash_price := 400
def down_payment := 120
def monthly_installment := 30
def number_of_months := 12

theorem savings_by_paying_cash :
  let total_cost_plan := down_payment + (monthly_installment * number_of_months) in
  let savings := total_cost_plan - cash_price in
  savings = 80 := 
by
  let total_cost_plan := down_payment + monthly_installment * number_of_months
  let savings := total_cost_plan - cash_price
  sorry

end savings_by_paying_cash_l804_804971


namespace distance_between_centers_l804_804272

-- Definitions and conditions
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := Nat.sqrt (a*a + b*b)
def p : ℚ := (a + b + c:ℕ) / 2
def area : ℚ :=  (a * b) / 2
def r : ℚ := area / p

-- The question (distance OQ)
def d (a b c p r : ℚ) := (p - b) / 2
def OQ (a b c p r d : ℚ) := Real.sqrt (d*d + r*r)

theorem distance_between_centers :
  OQ a b c p r (d a b c p r) = (Real.sqrt 5) / 2 :=
  sorry

end distance_between_centers_l804_804272


namespace cafeteria_extra_fruit_l804_804665

theorem cafeteria_extra_fruit 
    (red_apples : ℕ)
    (green_apples : ℕ)
    (students : ℕ)
    (total_apples := red_apples + green_apples)
    (apples_taken := students)
    (extra_apples := total_apples - apples_taken)
    (h1 : red_apples = 42)
    (h2 : green_apples = 7)
    (h3 : students = 9) :
    extra_apples = 40 := 
by 
  sorry

end cafeteria_extra_fruit_l804_804665


namespace potato_yield_computation_l804_804565

noncomputable def garden_potato_yield (l_s w_s s_l y : ℕ) : ℕ :=
  let l_f := l_s * s_l -- Convert length to feet
  let w_f := w_s * s_l -- Convert width to feet
  let area := l_f * w_f -- Calculate area
  area * y -- Calculate yield

theorem potato_yield_computation : garden_potato_yield 15 20 2 0.5 = 600 :=
  by
  sorry

end potato_yield_computation_l804_804565


namespace problem_statement_l804_804581

-- Definitions based on conditions
def two_digit_number (N : ℕ) := N >= 10 ∧ N < 100
def divisible_by (a b : ℕ) := a % b = 0
def mistaken_exactly (N : ℕ) := (if divisible_by N 3 then 0 else 1) +
                               (if divisible_by N 4 then 0 else 1) +
                               (if divisible_by N 5 then 0 else 1) +
                               (if divisible_by N 9 then 0 else 1) +
                               (if divisible_by N 10 then 0 else 1) +
                               (if divisible_by N 15 then 0 else 1) +
                               (if divisible_by N 18 then 0 else 1) +
                               (if divisible_by N 30 then 0 else 1)

-- Lean 4 statement
theorem problem_statement (N : ℕ) (h_two_digit : two_digit_number N) (h_mistaken : mistaken_exactly N = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 := sorry

end problem_statement_l804_804581


namespace smallest_possible_z_l804_804515

theorem smallest_possible_z :
  ∃ (z : ℕ), (z = 6) ∧ 
  ∃ (u w x y : ℕ), u < w ∧ w < x ∧ x < y ∧ y < z ∧ 
  u.succ = w ∧ w.succ = x ∧ x.succ = y ∧ y.succ = z ∧ 
  u^3 + w^3 + x^3 + y^3 = z^3 :=
by
  use 6
  sorry

end smallest_possible_z_l804_804515


namespace bisect_triangle_equally_l804_804940

variables {A B C H M : Point}
variables {A B C H M : Point} -- Points A, B, C, H (orthocenter), and M (midpoint of AC)
variables {triangle : Triangle A B C}
variables {orthocenter : orthocenter (triangle) = H}
variables {midpoint : is_midpoint M A C}
variables {l : Line}
variables {angle_bisector_AHC : AngleBisector (𝛼, 𝛽, 𝛾) ⟷ l}

theorem bisect_triangle_equally (condition_l : ∀ P Q, P, Q, l ∥ angle_bisector (\(<M)\), (\<H>\) ) :
  divides_triangle_into_equal_perimeters triangle l :=
by sorry

end bisect_triangle_equally_l804_804940


namespace no_student_has_more_than_2_candies_at_any_time_l804_804290

theorem no_student_has_more_than_2_candies_at_any_time :
  ∀ (students : Fin 2019 → ℕ), 
  (∀ i, students i = 1) →
  (∀ i j, 
    if students i > 0 then
      ∃ (k : ℤ), (students (i + k) = students i - 1) ∧ (students (i - k) = students i - 1)) →
  (∀ i, students i ≤ 2) :=
by
  sorry

end no_student_has_more_than_2_candies_at_any_time_l804_804290


namespace intercepts_correct_l804_804787

-- Define the equation of the line
def line_eq (x y : ℝ) := 5 * x - 2 * y - 10 = 0

-- Define the intercepts
def x_intercept : ℝ := 2
def y_intercept : ℝ := -5

-- Prove that the intercepts are as stated
theorem intercepts_correct :
  (∃ x, line_eq x 0 ∧ x = x_intercept) ∧
  (∃ y, line_eq 0 y ∧ y = y_intercept) :=
by
  sorry

end intercepts_correct_l804_804787


namespace map_scale_correct_l804_804300

theorem map_scale_correct (distance_real_m : ℕ) (distance_map_cm : ℕ) (scale : ℕ) :
  distance_real_m = 1600 →
  distance_map_cm = 8 →
  scale = 20000 →
  (distance_map_cm * scale) = (distance_real_m * 100) := 
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  sorry -- Placeholder for the calculation proof
end

end map_scale_correct_l804_804300


namespace math_proof_problem_l804_804181

-- Define curves C1 and C2
def curve_C1 (α : ℝ) : ℝ × ℝ := (2 * sqrt 5 * cos α, 2 * sin α)
def curve_C2 (ρ θ : ℝ) : Prop := ρ^2 + 4 * ρ * cos θ - 2 * ρ * sin θ + 4 = 0

-- Standard equations in Cartesian coordinates
def curve_C1_standard_eq : Prop :=
  ∀ x y, (∃ α, (x, y) = curve_C1 α) ↔ (x^2 / 20 + y^2 / 4 = 1)

def curve_C2_standard_eq : Prop :=
  ∀ x y ρ θ, (ρ = sqrt (x^2 + y^2) ∧ θ = arctan y x ∧ curve_C2 ρ θ) ↔ ((x + 2)^2 + (y - 1)^2 = 1)

-- Distance |AB| for specific line intersecting curve C2
def distance_AB : Prop :=
  let left_focus := (-4 : ℝ, 0 : ℝ) in
  let slope_angle := π / 4 in
  let line := λ t : ℝ, (left_focus.1 + sqrt 2 / 2 * t, left_focus.2 + sqrt 2 / 2 * t) in
  ∃ t1 t2, (curve_C2 (sqrt ((line t1).1^2 + (line t1).2^2)) (arctan (line t1).2 (line t1).1) ∧
            curve_C2 (sqrt ((line t2).1^2 + (line t2).2^2)) (arctan (line t2).2 (line t2).1)
           ) ∧ abs (t1 - t2) = sqrt 2
  
-- Final statement encapsulating all the proofs
theorem math_proof_problem :
  curve_C1_standard_eq ∧ curve_C2_standard_eq ∧ distance_AB :=
by sorry

end math_proof_problem_l804_804181


namespace hyperbola_asymptote_intersection_length_l804_804861

theorem hyperbola_asymptote_intersection_length (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
    (e : ℝ) (h_e : e = sqrt 2) :
    let hyperbola := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1,
        circle := ∀ x y : ℝ, x^2 + y^2 - 6 * x = 0 in
    (∃ x₁ x₂ y₁ y₂ : ℝ, 
        (circle x₁ y₁) ∧ (circle x₂ y₂) ∧ 
        (x₁ = y₁) ∧ (x₂ = y₂) ∧ 
        (abs (x₂ - x₁) = 3 * sqrt 2)) :=
sorry

end hyperbola_asymptote_intersection_length_l804_804861


namespace quadratic_roots_bc_minus_two_l804_804899

theorem quadratic_roots_bc_minus_two (b c : ℝ) 
  (h1 : 1 + -2 = -b) 
  (h2 : 1 * -2 = c) : b * c = -2 :=
by 
  sorry

end quadratic_roots_bc_minus_two_l804_804899


namespace find_points_and_max_area_l804_804103

open Real

noncomputable def parabola (a : ℝ) : set (ℝ × ℝ) := {p | p.snd ^ 2 = 4 * a * p.fst}

noncomputable def ellipse (a : ℝ) : set (ℝ × ℝ) :=
  {p | p.fst ^ 2 / (2 * a ^ 2) + p.snd ^ 2 / a ^ 2 = 1}

noncomputable def line : set (ℝ × ℝ) := {p | p.snd = p.fst - a}

def pointP (a : ℝ) := (4 * a / 3, a / 3)

def pointQ (a : ℝ) := ((3 - 2 * sqrt 2) * a, (2 - 2 * sqrt 2) * a)

noncomputable def hyperbola (a : ℝ) : set (ℝ × ℝ) :=
  {p | 7 * p.fst ^ 2 - 13 * p.snd ^ 2 = 11 * a ^ 2}

def triangleArea (a : ℝ) (t : ℝ) : ℝ :=
  let p := pointP a
  let q := pointQ a
  let (t0, t1), (p0, p1), (q0, q1) := (t, 0), p, q
  1 / 2 * abs (t0 * (p1 - q1) + p0 * (q1 - 0) + q0 * (0 - p1))

def maxArea (t : ℝ) : ℝ :=
  let c := sqrt 2 - 5 / 6
  c * (2 * t - 4)

theorem find_points_and_max_area (t : ℝ) (ht : t > 4) (a_min a_max : ℝ) (h_min : a_min = 1) (h_max : a_max = 2) :
  ∀ a ∈ set.Icc a_min a_max,
  pointP a = (4 * a / 3, a / 3) ∧
  pointQ a = ((3 - 2 * sqrt 2) * a, (2 - 2 * sqrt 2) * a) ∧
  ∀ q' : ℝ × ℝ, q' = (3 * a, 2 * a) → hyperbola a (pointP a) ∧
  hyperbola a q' ∧
  (triangleArea a t = maxArea t ↔ a = 2) :=
sorry

end find_points_and_max_area_l804_804103


namespace at_least_one_not_less_than_one_l804_804853

theorem at_least_one_not_less_than_one (m n : ℝ) :
  let f := λ x : ℝ, 2 * x^2 + m * x + n
  let f1 := f 1
  let f2 := f 2
  let f3 := f 3 in
  (f1 + f3 - 2 * f2 = 4) → (abs f1 ≥ 1 ∨ abs f2 ≥ 1 ∨ abs f3 ≥ 1) :=
by
  intro f f1 f2 f3 h
  sorry

end at_least_one_not_less_than_one_l804_804853


namespace min_value_A_l804_804446

-- Definition of A
def A (a : ℝ) : ℝ := a + 1 / (a + 2)

-- Problem statement
theorem min_value_A (a : ℝ) (h : a > -2) : ∃ b : ℝ, b = 0 ∧ ∀ x : ℝ, x > -2 → A x ≥ b := by
  sorry

end min_value_A_l804_804446


namespace tangent_angle_PC_ABC_l804_804762

variables (P A B C O O1 D : Point)
-- Given conditions
axiom PA_perp_ABC : ⊥ PA (plane A B C)
axiom angle_ABC_120 : angle A B C = 120
axiom PA_eq_4 : dist P A = 4
axiom circumscribed_sphere_radius : circumscribed_sphere_radius P A B C = 2 * sqrt 2

-- Theorem statement
theorem tangent_angle_PC_ABC : tan (angle (line P C) (plane A B C)) = (2 * sqrt 3) / 3 :=
by
  sorry

end tangent_angle_PC_ABC_l804_804762


namespace max_value_AMC_l804_804943

theorem max_value_AMC (A M C : ℕ) (h : A + M + C = 15) : 
  2 * (A * M * C) + A * M + M * C + C * A ≤ 325 := 
sorry

end max_value_AMC_l804_804943


namespace root_interval_l804_804062

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h1 : f 1 < 0) (h2 : f 1.5 > 0) (h3 : f 1.25 < 0) :
  ∃ c, 1.25 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  -- Proof by the Intermediate Value Theorem
  sorry

end root_interval_l804_804062


namespace pair_d_not_meet_goal_l804_804636

-- Defining the marks Sophie has already obtained in her first three tests
def first_three_marks : List ℕ := [73, 82, 85]

-- Function to calculate sum of a list of integers
def sum_list (lst: List ℕ) : ℕ := lst.foldr (· + ·) 0

-- Defining Sophie's goal average and the number of total tests
def goal_average : ℕ := 80
def num_tests : ℕ := 5
def total_goal_sum : ℕ := num_tests * goal_average

-- Calculating the sum of marks already obtained
def sum_first_three := sum_list first_three_marks

-- Calculating the required sum for the remaining two tests
def required_sum : ℕ := total_goal_sum - sum_first_three

-- Defining the pairs of marks for the remaining tests
def pair_a : List ℕ := [79, 82]
def pair_b : List ℕ := [70, 91]
def pair_c : List ℕ := [76, 86]
def pair_d : List ℕ := [73, 83]

-- Creating a theorem to prove that pair_d does not meet the goal
theorem pair_d_not_meet_goal : sum_list pair_d < required_sum := by
  sorry

end pair_d_not_meet_goal_l804_804636


namespace cyclical_winning_sets_l804_804013

theorem cyclical_winning_sets (teams : Finset ℕ) (h_card : teams.card = 21)
  (h_wins_losses : ∀ t ∈ teams, ∃ wins losses : Finset ℕ, wins.card = 10 ∧ losses.card = 10 ∧ wins ∩ losses = ∅ ∧ wins ∪ losses = teams.erase t) :
  ∃ (k : ℕ), k = 385 ∧ ∀ {T : Finset ℕ}, T.card = 3 → (∃ a b c : ℕ, a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a, b) ∈ wins_of t ∧ (b, c) ∈ wins_of t ∧ (c, a) ∈ wins_of t) :=
by
  sorry

end cyclical_winning_sets_l804_804013


namespace students_like_basketball_or_cricket_or_both_l804_804330

theorem students_like_basketball_or_cricket_or_both {A B C : ℕ} (hA : A = 12) (hB : B = 8) (hC : C = 3) :
    A + B - C = 17 :=
by
  sorry

end students_like_basketball_or_cricket_or_both_l804_804330


namespace maria_nickels_l804_804562

theorem maria_nickels (dimes quarters_initial quarters_additional : ℕ) (total_amount : ℚ) 
  (Hd : dimes = 4) (Hqi : quarters_initial = 4) (Hqa : quarters_additional = 5) (Htotal : total_amount = 3) : 
  (dimes * 0.10 + quarters_initial * 0.25 + quarters_additional * 0.25 + n/20) = total_amount → n = 7 :=
  sorry

end maria_nickels_l804_804562


namespace geese_percentage_non_ducks_l804_804938

theorem geese_percentage_non_ducks :
  let total_birds := 100
  let geese := 0.20 * total_birds
  let swans := 0.30 * total_birds
  let herons := 0.15 * total_birds
  let ducks := 0.25 * total_birds
  let pigeons := 0.10 * total_birds
  let non_duck_birds := total_birds - ducks
  (geese / non_duck_birds) * 100 = 27 := 
by
  sorry

end geese_percentage_non_ducks_l804_804938


namespace lemonade_stand_total_profit_l804_804379

theorem lemonade_stand_total_profit :
  let day1_revenue := 21 * 4
  let day1_expenses := 10 + 5 + 3
  let day1_profit := day1_revenue - day1_expenses

  let day2_revenue := 18 * 5
  let day2_expenses := 12 + 6 + 4
  let day2_profit := day2_revenue - day2_expenses

  let day3_revenue := 25 * 4
  let day3_expenses := 8 + 4 + 3 + 2
  let day3_profit := day3_revenue - day3_expenses

  let total_profit := day1_profit + day2_profit + day3_profit

  total_profit = 217 := by
    sorry

end lemonade_stand_total_profit_l804_804379


namespace interval_for_decreasing_log_l804_804655

noncomputable def decreasing_interval (m : ℝ) : Prop :=
∀ x y : ℝ, 
  m < x ∧ x < m + 1 ∧ m < y ∧ y < m + 1 ∧ x < y → 
  (log 0.5 (-x^2 + 6 * x - 5)) > (log 0.5 (-y^2 + 6 * y - 5))

theorem interval_for_decreasing_log : 
  ∀ m : ℝ, 
  (1 ≤ m ∧ m ≤ 2) ↔ decreasing_interval m :=
by 
  sorry

end interval_for_decreasing_log_l804_804655


namespace length_of_goods_train_l804_804322

-- Define the given conditions
def speed_kmph := 72
def platform_length := 260
def crossing_time := 26

-- Convert speed to m/s
def speed_mps := (speed_kmph * 5) / 18

-- Calculate distance covered
def distance_covered := speed_mps * crossing_time

-- Define the length of the train
def train_length := distance_covered - platform_length

theorem length_of_goods_train : train_length = 260 := by
  sorry

end length_of_goods_train_l804_804322


namespace incorrect_statements_about_g_l804_804217

noncomputable def f (x : ℝ) : ℝ := 2 / (abs (x - 1) + 1)

def g (x : ℝ) : ℤ := Int.floor (f x)

def is_even (g : ℝ → ℤ) : Prop :=
  ∀ x, g x = g (-x)

def has_min_value (g : ℝ → ℤ) (m : ℤ) : Prop :=
  ∀ x, g x ≥ m

def is_monotonic (g : ℝ → ℤ) : Prop :=
  ∀ x y, x ≤ y → g x ≤ g y

theorem incorrect_statements_about_g :
  ¬ is_even g ∧
  ¬ has_min_value g 1 ∧
  (∀ x, g x ∈ {0, 1, 2}) ∧
  ¬ is_monotonic g :=
by
  sorry

end incorrect_statements_about_g_l804_804217


namespace trapezoid_intersect_l804_804110

variables (A B C D P M Q N : Type*)
variables [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables [VAdd Q P] [VAdd P P] [VAdd R R] [Sub Q P] [Div R R] [AddCommGroup Q]
variables [vector_space Q P]

/-- Given a trapezoid ABCD with BC parallel to AD, and P, M, Q, N being the midpoints
of sides AB, BC, CD, and DA respectively, we aim to prove that the segments AQ, PD,
and MN intersect at one point. -/
theorem trapezoid_intersect (ABCD : qr) [BC parallel AD] [midpoint_of_sides P M Q N ABCD ] :
  intersect (AQ PD MN) :=
sorry

end trapezoid_intersect_l804_804110


namespace paco_more_cookies_l804_804238

def paco_cookies_difference
  (initial_cookies : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_given : ℕ) : ℕ :=
  cookies_eaten - cookies_given

theorem paco_more_cookies 
  (initial_cookies : ℕ)
  (cookies_eaten : ℕ)
  (cookies_given : ℕ)
  (h1 : initial_cookies = 17)
  (h2 : cookies_eaten = 14)
  (h3 : cookies_given = 13) :
  paco_cookies_difference initial_cookies cookies_eaten cookies_given = 1 :=
by
  rw [h2, h3]
  exact rfl

end paco_more_cookies_l804_804238


namespace companion_value_4164_smallest_N_satisfies_conditions_l804_804558

-- Define relevant functions
def G (N : ℕ) : ℕ :=
  let digits := [N / 1000 % 10, N / 100 % 10, N / 10 % 10, N % 10]
  digits.sum

def P (N : ℕ) : ℕ :=
  (N / 1000 % 10) * (N / 100 % 10)

def Q (N : ℕ) : ℕ :=
  (N / 10 % 10) * (N % 10)

def companion_value (N : ℕ) : ℚ :=
  |(G N : ℤ) / ((P N : ℤ) - (Q N : ℤ))|

-- Proof problem for part (1)
theorem companion_value_4164 : companion_value 4164 = 3 / 4 := sorry

-- Proof problem for part (2)
theorem smallest_N_satisfies_conditions :
  ∀ (N : ℕ), N > 1000 ∧ N < 10000 ∧ (∀ d, N / 10^d % 10 ≠ 0) ∧ (N / 1000 % 10 + N % 10) % 9 = 0 ∧ G N = 16 ∧ companion_value N = 4 → N = 2527 := sorry

end companion_value_4164_smallest_N_satisfies_conditions_l804_804558


namespace sum_of_coordinates_of_point_D_l804_804240

noncomputable def pointM := (4, -2 : ℤ × ℤ)
noncomputable def pointC := (-1, 5 : ℤ × ℤ)

theorem sum_of_coordinates_of_point_D (x y : ℤ) (h1 : (x + pointC.1) / 2 = pointM.1) (h2 : (y + pointC.2) / 2 = pointM.2) :
  x + y = 0 :=
sorry

end sum_of_coordinates_of_point_D_l804_804240


namespace division_example_l804_804315

theorem division_example : ∃ A B : ℕ, 23 = 6 * A + B ∧ A = 3 ∧ B < 6 := 
by sorry

end division_example_l804_804315


namespace intersection_complement_eq_l804_804867

open Set

theorem intersection_complement_eq :
  let U := {1, 2, 3, 4, 5, 6}
  let A := {1, 2, 3}
  let B := {2, 5, 6}
  A ∩ (U \ B) = {1, 3} := by
  let U := {1, 2, 3, 4, 5, 6}
  let A := {1, 2, 3}
  let B := {2, 5, 6}
  have h1 : U \ B = {1, 3, 4} := by sorry
  have h2 : A ∩ {1, 3, 4} = {1, 3} := by sorry
  rw [←h1, h2]
  rfl

end intersection_complement_eq_l804_804867


namespace repeatingDecimal_to_frac_l804_804795

noncomputable def repeatingDecimalToFrac : ℚ :=
  3 + 3 * (2 / 99 : ℚ)

theorem repeatingDecimal_to_frac :
  repeatingDecimalToFrac = 101 / 33 :=
by {
  sorry
}

end repeatingDecimal_to_frac_l804_804795


namespace total_amount_received_l804_804354

-- Define the conditions
def purchase_price_1 : ℝ := 600
def purchase_price_2 : ℝ := 800
def purchase_price_3 : ℝ := 1000

def loss_percentage_1 : ℝ := 20 / 100
def loss_percentage_2 : ℝ := 25 / 100
def loss_percentage_3 : ℝ := 30 / 100

-- Define the loss calculations
def loss_amount_1 : ℝ := loss_percentage_1 * purchase_price_1
def loss_amount_2 : ℝ := loss_percentage_2 * purchase_price_2
def loss_amount_3 : ℝ := loss_percentage_3 * purchase_price_3

-- Define the selling price calculations
def selling_price_1 : ℝ := purchase_price_1 - loss_amount_1
def selling_price_2 : ℝ := purchase_price_2 - loss_amount_2
def selling_price_3 : ℝ := purchase_price_3 - loss_amount_3

-- Define the theorem to prove the total amount received
theorem total_amount_received : selling_price_1 + selling_price_2 + selling_price_3 = 1780 := by
  sorry

end total_amount_received_l804_804354


namespace sum_of_real_solutions_eq_neg_32_div_7_l804_804429

theorem sum_of_real_solutions_eq_neg_32_div_7 :
  ∀ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x) →
  ∑ sol in { x | (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 12 * x) }, sol = -32 / 7 :=
by
  sorry

end sum_of_real_solutions_eq_neg_32_div_7_l804_804429


namespace unique_a_exists_iff_n_eq_two_l804_804798

theorem unique_a_exists_iff_n_eq_two (n : ℕ) (h1 : 1 < n) : 
  (∃ a : ℕ, 0 < a ∧ a ≤ n! ∧ n! ∣ a^n + 1 ∧ ∀ b : ℕ, (0 < b ∧ b ≤ n! ∧ n! ∣ b^n + 1) → b = a) ↔ n = 2 := 
by {
  sorry
}

end unique_a_exists_iff_n_eq_two_l804_804798


namespace Part1_Part2_l804_804511

-- For part 1
theorem Part1 : @fin.prob 0.5 (i → bool) 6 = \sum_{k=0}^{2} {binomial 6 k * (0.5)^k * (0.5)^(6-k)}
begin
  sorry
end

-- For part 2
theorem Part2 (n : nat) (E : ℝ := 0.5 * n) (D : ℝ := 0.25 * n) : 
  (1 - D / ((0.1 * n) ^ 2) ≥ 0.98) → (n ≥ 1250) :=
begin
  rw sub_ge at h,
  rw ge_eq_le at h,
  rw le_sub_iff_add_le at h,
  rw div_le_iff at h,
  norm_num at h,
  rw [pow_two, ge_eq_le] at h,
  rw mul_inv at h,
  exact h
  sorry
end


end Part1_Part2_l804_804511


namespace real_solutions_x4_plus_3_minus_x4_eq_82_l804_804072

theorem real_solutions_x4_plus_3_minus_x4_eq_82 :
  ∀ x : ℝ, x = 2.6726 ∨ x = 0.3274 → x^4 + (3 - x)^4 = 82 := by
  sorry

end real_solutions_x4_plus_3_minus_x4_eq_82_l804_804072


namespace distance_center_to_line_l804_804163

noncomputable def distance_from_center_to_line
    (a : ℝ) (h_circle: (a > 0) ∧ (∀ x y : ℝ, (x - a) ^ 2 + (y - a) ^ 2 = a ^ 2) ∧ (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2)
    (line : ℝ × ℝ → ℝ) : ℝ :=
    let center := if (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2 then (a, a) else (? , ?);
    let numerator := abs (2 * center.fst - center.snd - 3) in
    let denominator := sqrt (2 ^ 2 + 1 ^ 2) in
    numerator / denominator

theorem distance_center_to_line (a : ℝ)
    (h_circle: (a > 0) ∧ (∀ x y : ℝ, (x - a) ^ 2 + (y - a) ^ 2 = a ^ 2) ∧ (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2):
    distance_from_center_to_line a h_circle (λ p, 2 * p.1 - p.2 - 3) = (2 * Real.sqrt 5) / 5 := 
    sorry

end distance_center_to_line_l804_804163


namespace cubic_roots_sum_of_cubes_l804_804210

theorem cubic_roots_sum_of_cubes :
  ∀ (a b c : ℝ), 
  (∀ x : ℝ, 9 * x^3 + 14 * x^2 + 2047 * x + 3024 = 0 → (x = a ∨ x = b ∨ x = c)) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = -58198 / 729 :=
by
  intros a b c roota_eqn
  sorry

end cubic_roots_sum_of_cubes_l804_804210


namespace find_a_l804_804846

def tangent_condition (x a : ℝ) : Prop := 2 * x - (Real.log x + a) + 1 = 0

def slope_condition (x : ℝ) : Prop := 2 = 1 / x

theorem find_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ tangent_condition x a ∧ slope_condition x) →
  a = -2 * Real.log 2 :=
by
  intro h
  sorry

end find_a_l804_804846


namespace nonrational_ab_l804_804877

theorem nonrational_ab {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) : 
    ¬(∃ (p q r s : ℤ), q ≠ 0 ∧ s ≠ 0 ∧ a = p / q ∧ b = r / s) := by
  sorry

end nonrational_ab_l804_804877


namespace find_length_QR_l804_804984

-- Conditions
variables {D E F Q R : Type} [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace Q] [MetricSpace R]
variables {DE EF DF QR : ℝ} (tangent : Q = E ∧ R = D)
variables (t₁ : de = 5) (t₂ : ef = 12) (t₃ : df = 13)

-- Problem: Prove that QR = 5 given the conditions.
theorem find_length_QR : QR = 5 :=
sorry

end find_length_QR_l804_804984


namespace linear_function_max_value_l804_804314

theorem linear_function_max_value (m x : ℝ) (h : -1 ≤ x ∧ x ≤ 3) (y : ℝ) 
  (hl : y = m * x - 2 * m) (hy : y = 6) : m = -2 ∨ m = 6 := 
by 
  sorry

end linear_function_max_value_l804_804314


namespace oranges_picked_in_total_l804_804235

-- Define the variables representing the number of fruits picked by each person on each day
def Mary_oranges_Monday := 14
def Jason_oranges_Monday := 41
def Amanda_oranges_Monday := 56
def Keith_apples_Monday := 38

def Mary_apples_Tuesday := 22
def Jason_grapefruits_Tuesday := 15
def Amanda_oranges_Tuesday := 36
def Keith_plums_Tuesday := 47

-- Calculate the total number of oranges picked on Monday
def total_oranges_Monday := Mary_oranges_Monday + Jason_oranges_Monday + Amanda_oranges_Monday

-- Calculate the total number of oranges picked on Tuesday
def total_oranges_Tuesday := Amanda_oranges_Tuesday

-- Calculate the total number of oranges picked over both days
def total_oranges := total_oranges_Monday + total_oranges_Tuesday

-- The theorem we want to prove
theorem oranges_picked_in_total (Mary_oranges_Monday Jason_oranges_Monday Amanda_oranges_Monday Amanda_oranges_Tuesday : ℕ) 
(Mary_oranges_Monday = 14) 
(Jason_oranges_Monday = 41) 
(Amanda_oranges_Monday = 56) 
(Amanda_oranges_Tuesday = 36) : 
  total_oranges = 147 := 
  by 
  -- Proof steps would go here
  sorry

end oranges_picked_in_total_l804_804235


namespace average_temp_is_correct_l804_804248

-- Define the temperatures for each day
def sunday_temp : ℕ := 40
def monday_temp : ℕ := 50
def tuesday_temp : ℕ := 65
def wednesday_temp : ℕ := 36
def thursday_temp : ℕ := 82
def friday_temp : ℕ := 72
def saturday_temp : ℕ := 26

-- Define the total number of days in the week
def days_in_week : ℕ := 7

-- Define the total temperature for the week
def total_temperature : ℕ := sunday_temp + monday_temp + tuesday_temp + 
                             wednesday_temp + thursday_temp + friday_temp + 
                             saturday_temp

-- Define the average temperature calculation
def average_temperature : ℕ := total_temperature / days_in_week

-- The theorem to be proved
theorem average_temp_is_correct : average_temperature = 53 := by
  sorry

end average_temp_is_correct_l804_804248


namespace find_vectors_and_projection_l804_804209

noncomputable def vec_a := (-1 : ℝ, 1 : ℝ)
noncomputable def vec_d := (8 : ℝ, 6 : ℝ)
noncomputable def vec_b (x : ℝ) := (x, 3 : ℝ)
noncomputable def vec_c (y : ℝ) := (5 : ℝ, y)

theorem find_vectors_and_projection :
  -- Given conditions
  ∀ (x y : ℝ),
    let b := vec_b x,
    let c := vec_c y in
    (b.1 / vec_d.1) = (b.2 / vec_d.2) →    -- Condition 1: b is parallel to d
    (4 * vec_a + vec_d) • c = 0 →           -- Condition 2: (4a + d) is perpendicular to c
    
    -- Proving the correct values of vectors b and c
    b = (4, 3) ∧ c = (5, -2) ∧
    
    -- Projection of c in the direction of a
    let proj := (vec_a • c) / ‖vec_a‖ in
        proj = - (7 * Real.sqrt 2) / 2 :=
begin
  intros x y b c h1 h2,
  simp only [vec_b, vec_c] at *,
  split,
  { exact ⟨x = 4, y = -2⟩ },
  { sorry },
end

end find_vectors_and_projection_l804_804209


namespace triangle_angle_C_l804_804927

theorem triangle_angle_C (A B C : ℝ) (sin cos : ℝ → ℝ) 
  (h1 : 3 * sin A + 4 * cos B = 6)
  (h2 : 4 * sin B + 3 * cos A = 1)
  (triangle_sum : A + B + C = 180) :
  C = 30 :=
by
  sorry

end triangle_angle_C_l804_804927


namespace perimeter_of_regular_polygon_l804_804362

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : n = 3) (h2 : side_length = 5) (h3 : exterior_angle = 120) : 
  n * side_length = 15 :=
by
  sorry

end perimeter_of_regular_polygon_l804_804362


namespace true_for_2_and_5_l804_804782

theorem true_for_2_and_5 (x : ℝ) : ((x - 2) * (x - 5) = 0) ↔ (x = 2 ∨ x = 5) :=
by
  sorry

end true_for_2_and_5_l804_804782


namespace minimal_diagonal_sum_spiral_matrix_l804_804761

theorem minimal_diagonal_sum_spiral_matrix (n : ℕ) (M : matrix (fin n) (fin n) ℕ) 
  (spiral_fill : ∀ i j, M i j ∈ finset.range (1, n^2 + 1) ∧ spiral_order M) :
  ∀ (sel : (fin n → fin n)), (∀ i j, i ≠ j → sel i ≠ sel j) →
  (∑ i in finset.range n, M i (sel i)) ≥ (∑ i in finset.range n, M i i) :=
begin
  sorry
end

end minimal_diagonal_sum_spiral_matrix_l804_804761


namespace triangle_abc_l804_804524

-- Conditions
variables (a b c : ℝ)
variable (cosC : ℝ)
variable (sum_ab : a + b = 6)
variable (c_value : c = 2)
variable (cosC_value : cosC = 7/9)

-- Define cosC and sinC
def sinC := sqrt (1 - cosC^2)

-- Define the area function for triangle ABC
noncomputable def area_triangle (a b : ℝ) (sinC : ℝ) : ℝ :=
  0.5 * a * b * sinC

-- Statements to prove
theorem triangle_abc (a b : ℝ) (cosC : ℝ)
  (ha : sum_ab a b)
  (hc : c_value c)
  (hcos : cosC_value cosC) :
  a = 3 ∧ b = 3 ∧ area_triangle a b (sinC) = 2 * sqrt 2 :=
sorry

end triangle_abc_l804_804524


namespace problem_statement_l804_804584

-- Definitions based on conditions
def two_digit_number (N : ℕ) := N >= 10 ∧ N < 100
def divisible_by (a b : ℕ) := a % b = 0
def mistaken_exactly (N : ℕ) := (if divisible_by N 3 then 0 else 1) +
                               (if divisible_by N 4 then 0 else 1) +
                               (if divisible_by N 5 then 0 else 1) +
                               (if divisible_by N 9 then 0 else 1) +
                               (if divisible_by N 10 then 0 else 1) +
                               (if divisible_by N 15 then 0 else 1) +
                               (if divisible_by N 18 then 0 else 1) +
                               (if divisible_by N 30 then 0 else 1)

-- Lean 4 statement
theorem problem_statement (N : ℕ) (h_two_digit : two_digit_number N) (h_mistaken : mistaken_exactly N = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 := sorry

end problem_statement_l804_804584


namespace n_power_of_two_if_2_pow_n_plus_one_odd_prime_l804_804993

-- Definition: a positive integer n is a power of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

-- Theorem: if 2^n +1 is an odd prime, then n must be a power of 2
theorem n_power_of_two_if_2_pow_n_plus_one_odd_prime (n : ℕ) (hp : Prime (2^n + 1)) (hn : Odd (2^n + 1)) : is_power_of_two n :=
by
  sorry

end n_power_of_two_if_2_pow_n_plus_one_odd_prime_l804_804993


namespace find_num_female_students_l804_804331

noncomputable def numFemaleStudents (totalAvg maleAvg femaleAvg : ℕ) (numMales : ℕ) : ℕ :=
  let numFemales := (totalAvg * (numMales + (totalAvg * 0)) - (maleAvg * numMales)) / femaleAvg
  numFemales

theorem find_num_female_students :
  (totalAvg maleAvg femaleAvg : ℕ) →
  (numMales : ℕ) →
  totalAvg = 90 →
  maleAvg = 83 →
  femaleAvg = 92 →
  numMales = 8 →
  numFemaleStudents totalAvg maleAvg femaleAvg numMales = 28 := by
    intros
    sorry

end find_num_female_students_l804_804331


namespace sqrt2_plus_sqrt3_irrational_l804_804460

-- Condition definitions
def sqrt2_irrational : Prop := ¬ ∃ (q : ℚ), q^2 = 2
def sqrt3_irrational : Prop := ¬ ∃ (q : ℚ), q^2 = 3

-- Problem statement
theorem sqrt2_plus_sqrt3_irrational
  (h1 : sqrt2_irrational)
  (h2 : sqrt3_irrational) :
  ¬ ∃ (q : ℚ), q = (sqrt 2 + sqrt 3) :=
by
  sorry

end sqrt2_plus_sqrt3_irrational_l804_804460


namespace circle_distance_to_line_l804_804165

-- Definitions for the conditions
def circle_tangent_to_axes_center (a : ℝ) (h : a > 0) : (ℝ × ℝ) := (a, a)
def circle_tangent_to_axes_radius (a : ℝ) (h : a > 0) : ℝ := a

def passes_through (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

def distance_point_to_line (x y A B C : ℝ) : ℝ :=
  |A * x + B * y + C| / real.sqrt (A^2 + B^2)

-- Theorem statement
theorem circle_distance_to_line 
    (a : ℝ) (h : a > 0)
    (pt : ℝ × ℝ) (hpt : pt = (2, 1))
    (hpass : passes_through (circle_tangent_to_axes_center a h) (circle_tangent_to_axes_radius a h) pt) :
  distance_point_to_line a a 2 (-1) (-3) = 2 * real.sqrt 5 / 5 :=
begin
  sorry
end

end circle_distance_to_line_l804_804165


namespace num_logs_in_stack_l804_804370

-- Define the conditions
def initial_logs : ℕ := 20
def common_difference : ℤ := -2
def top_row_logs : ℕ := 4

-- Define the function that calculates the nth term of the arithmetic series
def nth_term (a₁ : ℕ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Define the function that calculates the sum of the first n terms of an arithmetic series
def sum_arithmetic_series (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

-- Statement, to be proved
theorem num_logs_in_stack :
  let n := 9 in -- number of rows, found from solving 20 + (n - 1)(-2) = 4
  sum_arithmetic_series initial_logs top_row_logs n = 108 :=
by
  sorry

end num_logs_in_stack_l804_804370


namespace line_intersections_with_parabola_l804_804275

theorem line_intersections_with_parabola :
  ∃! (L : ℝ → ℝ) (l_count : ℕ),  
    l_count = 3 ∧
    (∀ x : ℝ, (L x) ∈ {x | (L 0 = 2) ∧ ∃ y, y * y = 8 * x ∧ L x = y}) := sorry

end line_intersections_with_parabola_l804_804275


namespace price_reduction_percentage_l804_804361

theorem price_reduction_percentage (original_price current_price reduction_percentage : ℕ) :
  original_price = 3000 → current_price = 2400 → reduction_percentage = 20 → 
  reduction_percentage = ((original_price - current_price) * 100) / original_price :=
by
  intros h_orig h_curr h_reduc
  rw [h_orig, h_curr, h_reduc]
  sorry

end price_reduction_percentage_l804_804361


namespace coefficient_of_q_l804_804920

theorem coefficient_of_q (q' : ℤ → ℤ) (h : ∀ q, q' q = 3 * q - 3) (h₁ : q' (q' 4) = 72) : 
  ∀ q, q' q = 3 * q - 3 :=
  sorry

end coefficient_of_q_l804_804920


namespace moses_income_l804_804564

theorem moses_income (investment : ℝ) (percentage : ℝ) (dividend_rate : ℝ) (income : ℝ)
  (h1 : investment = 3000) (h2 : percentage = 0.72) (h3 : dividend_rate = 0.0504) :
  income = 210 :=
sorry

end moses_income_l804_804564


namespace complement_union_l804_804866

open Set

theorem complement_union (A B : Set ℝ) (U : Set ℝ) 
  (hU : U = univ)
  (hA : A = {x | x < 0})
  (hB : B = {x | x ≥ 2}) :
  compl (A ∪ B) = {x | 0 ≤ x ∧ x < 2} := 
  by
    rw [hA, hB, compl_union, compl_set_of, compl_set_of]
    sorry

end complement_union_l804_804866


namespace problem1_problem2_l804_804554

-- Define the piecewise function f.
def f (a x : ℝ) : ℝ :=
  if x < 1 then 2^x - a else 4 * (x - a) * (x - 2 * a)

-- State the minimum value of f when a = 1.
theorem problem1 (x : ℝ) (h : x < 1 → 2^x - 1 > f 1 x)
  : {x // x < 1} → ( ∀ k, x = k → f 1 x > -1 ) := sorry

-- State the range of a for f to have exactly 2 zeros.
theorem problem2 (a x : ℝ) 
  : ( (∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ (1 / 2 ≤ a ∧ a < 1 ∨ a ≥ 2) ) := sorry

end problem1_problem2_l804_804554


namespace find_j_l804_804085

noncomputable def g (x : ℝ) : ℝ := cot (x / 3) - cot x

theorem find_j {j : ℝ} (h : ∀ x, g x = sin (j * x) / (sin (x / 3) * sin x)) : j = 2 / 3 :=
sorry

end find_j_l804_804085


namespace total_jokes_proof_l804_804936

-- Definitions of the conditions
def jokes_jessy_last_saturday : Nat := 11
def jokes_alan_last_saturday : Nat := 7
def jokes_jessy_next_saturday : Nat := 2 * jokes_jessy_last_saturday
def jokes_alan_next_saturday : Nat := 2 * jokes_alan_last_saturday

-- Sum of jokes over two Saturdays
def total_jokes : Nat := (jokes_jessy_last_saturday + jokes_alan_last_saturday) + (jokes_jessy_next_saturday + jokes_alan_next_saturday)

-- The proof problem
theorem total_jokes_proof : total_jokes = 54 := 
by
  sorry

end total_jokes_proof_l804_804936


namespace b_sum_l804_804513

noncomputable def a (n : ℕ) : ℕ := n + 2
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

/--
In the arithmetic sequence $\{a_n\}$, $a_2=4$, $a_4+a_7=15$, and the general term formula is $a_n = n + 2$.
Let $b_{n}= \frac{1}{a_{n}a_{n+1}}$, prove that the sum $b_1 + b_2 + \ldots + b_{10}$ equals $\frac{10}{39}$.
-/
theorem b_sum :
  (b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 + b 10) = 10 / 39 :=
by {
  sorry
}

end b_sum_l804_804513


namespace neznika_number_l804_804577

theorem neznika_number (N : ℕ) :
  10 ≤ N ∧ N ≤ 99 ∧
  (divisible_by N 3 ∧ divisible_by N 4 ∧ divisible_by N 5 ∧
   divisible_by N 9 ∧ divisible_by N 10 ∧ divisible_by N 15 ∧
   divisible_by N 18 ∧ divisible_by N 30) →
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end neznika_number_l804_804577


namespace intersection_with_y_axis_l804_804648

theorem intersection_with_y_axis {x y : ℝ} (h : y = 2 * x - 5) (hx : x = 0) : y = -5 :=
by {
  rw hx at h,
  simp at h,
  exact h
}

end intersection_with_y_axis_l804_804648


namespace salmon_oxygen_ratio_l804_804415

def salmon_speed (O : ℝ) : ℝ := (1/2) * log_base 3 (O / 100)

theorem salmon_oxygen_ratio (O1 O2 : ℝ) (v : ℝ):
  salmon_speed O1 = v →
  salmon_speed O2 = v + 2 →
  O2 / O1 = 81 :=
by 
  sorry

end salmon_oxygen_ratio_l804_804415


namespace sum_of_nth_numbers_l804_804039

theorem sum_of_nth_numbers (n : ℕ) : 
  let first_set_nth := n^2,
      second_set_nth := n^3,
      third_set_nth := -2 * n^2 in
  first_set_nth + second_set_nth + third_set_nth = n^3 - n^2 := 
by 
  sorry

end sum_of_nth_numbers_l804_804039


namespace sin_angle_BAC_l804_804112

structure Point (α : Type u) := (x : α) (y : α)

def point_A : Point ℝ := ⟨1, 2⟩
def point_B : Point ℝ := ⟨3, 4⟩
def point_C : Point ℝ := ⟨5, 0⟩

noncomputable def vector (p1 p2 : Point ℝ) : Point ℝ :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

noncomputable def magnitude (v : Point ℝ) : ℝ :=
  real.sqrt (v.x^2 + v.y^2)

noncomputable def dot_product (v1 v2 : Point ℝ) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

noncomputable def cos_angle (v1 v2 : Point ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

noncomputable def sin_of_angle (cos_val : ℝ) : ℝ :=
  real.sqrt (1 - cos_val^2)

theorem sin_angle_BAC : sin_of_angle (cos_angle (vector point_A point_B) (vector point_A point_C)) = (3 * real.sqrt 10) / 10 :=
by 
  sorry

end sin_angle_BAC_l804_804112


namespace min_distance_from_curve_to_line_l804_804826

noncomputable def min_distance : ℝ :=
  let curve (x : ℝ) := x^2 - log x
  let line (x : ℝ) := x - 4
  sqrt 8

theorem min_distance_from_curve_to_line :
  ∀ (P : ℝ × ℝ), (P.snd = P.fst^2 - log P.fst) → ∃ (d : ℝ), d = min_distance := 
begin
  sorry
end

end min_distance_from_curve_to_line_l804_804826


namespace two_digit_number_possible_options_l804_804594

theorem two_digit_number_possible_options
  (N : ℕ)
  (h1 : 10 ≤ N ∧ N < 100)
  (h2 : (N % 3 = 0 ∨ N % 3 ≠ 0) ∧
        (N % 4 = 0 ∨ N % 4 ≠ 0) ∧
        (N % 5 = 0 ∨ N % 5 ≠ 0) ∧
        (N % 9 = 0 ∨ N % 9 ≠ 0) ∧
        (N % 10 = 0 ∨ N % 10 ≠ 0) ∧
        (N % 15 = 0 ∨ N % 15 ≠ 0) ∧
        (N % 18 = 0 ∨ N % 18 ≠ 0) ∧
        (N % 30 = 0 ∨ N % 30 ≠ 0)) :
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end two_digit_number_possible_options_l804_804594


namespace laborer_monthly_income_l804_804332

theorem laborer_monthly_income :
  (∃ (I D : ℤ),
    6 * I + D = 540 ∧
    4 * I - D = 270) →
  (∃ I : ℤ,
    I = 81) :=
by
  sorry

end laborer_monthly_income_l804_804332


namespace post_tax_raise_percentage_correct_l804_804356

noncomputable def salary_adjustment (S : ℝ) : ℝ :=
  (1 / 0.6192) - 1

theorem post_tax_raise_percentage_correct (S : ℝ) :
  let reduced_salary := 0.86 * S in
  let tax := 0.28 * reduced_salary in
  let post_tax_salary := reduced_salary - tax in
  let raise_percentage := salary_adjustment S in
  (1 + raise_percentage) * post_tax_salary = S :=
by
  let reduced_salary := 0.86 * S
  let tax := 0.28 * reduced_salary
  let post_tax_salary := reduced_salary - tax
  let raise_percentage := (1 / 0.6192) - 1
  have h : (1 + raise_percentage) * post_tax_salary = S :=
  by sorry
  exact h

end post_tax_raise_percentage_correct_l804_804356


namespace find_real_solutions_l804_804073

def equation (x : ℝ) : Prop := x^4 + (3 - x)^4 = 82

theorem find_real_solutions : 
  {x : ℝ // equation x} = {x | x ≈ 3.22 ∨ x ≈ -0.22} :=
by 
  sorry

end find_real_solutions_l804_804073


namespace dayOfWeek_20081001_l804_804972

def isLeapYear (y : ℕ) : Prop := (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

def daysInYear (y : ℕ) : ℕ := if isLeapYear y then 366 else 365

noncomputable def totalDays (startYear endYear : ℕ) : ℕ :=
  let years := list.range (endYear - startYear)
  let yearsAdjusted := years.map (λ n => startYear + n)
  (yearsAdjusted.map daysInYear).sum

def dayOfWeekAfterYears (startYear endYear : ℕ) (startDay : ℕ) : ℕ :=
  (startDay + totalDays startYear endYear) % 7

axiom dayOfWeek_20021001 : 2002-10-01 = 2 

theorem dayOfWeek_20081001 : dayOfWeekAfterYears 2002 2008 2 = 3 :=
begin
  sorry
end

end dayOfWeek_20081001_l804_804972


namespace sin_alpha_of_point_l804_804129

theorem sin_alpha_of_point (a : ℝ) (h : a < 0) : 
  let P := (3 * a, 4 * a) in 
  let r := Real.sqrt ((3 * a) ^ 2 + (4 * a) ^ 2) in 
  sin (Real.atan2 (4 * a) (3 * a)) = -4 / 5 :=
by
  sorry

end sin_alpha_of_point_l804_804129


namespace largest_integer_with_remainder_l804_804420

theorem largest_integer_with_remainder (n : ℕ) (h₁ : n < 150) (h₂ : n % 9 = 2) : n ≤ 146 :=
by
  have h : 146 % 9 = 2 := by norm_num [Nat.mod_eq_of_lt]; norm_num
  sorry

end largest_integer_with_remainder_l804_804420


namespace smallest_value_m_plus_n_l804_804442

theorem smallest_value_m_plus_n (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : m + n = 60 :=
sorry

end smallest_value_m_plus_n_l804_804442


namespace cost_of_Roger_cookie_l804_804095

theorem cost_of_Roger_cookie
  (art_cookie_length : ℕ := 4)
  (art_cookie_width : ℕ := 3)
  (art_cookie_count : ℕ := 10)
  (roger_cookie_side : ℕ := 3)
  (art_cookie_price : ℕ := 50)
  (same_dough_used : ℕ := art_cookie_count * art_cookie_length * art_cookie_width)
  (roger_cookie_area : ℕ := roger_cookie_side * roger_cookie_side)
  (roger_cookie_count : ℕ := same_dough_used / roger_cookie_area) :
  (500 / roger_cookie_count) = 38 := by
  sorry

end cost_of_Roger_cookie_l804_804095


namespace reflection_line_l804_804638

open Real

theorem reflection_line (m b : ℝ) :
  (image_point_reflection (-2, 3) (4, -5)) → (m + b = -1) :=
by 
  sorry

end reflection_line_l804_804638


namespace pet_store_bird_count_l804_804743

def total_birds_in_pet_store : ℕ := 51

theorem pet_store_bird_count :
  ∃ (total_birds : ℕ),
    let num_cages := 9,
        num_mixed_cages := num_cages - num_cages / 3,
        num_parakeet_only_cages := num_cages / 3,
        birds_in_mixed_cages := num_mixed_cages * (2 + 3 + 1),
        birds_in_parakeet_only_cages := num_parakeet_only_cages * 5,
        total_birds := birds_in_mixed_cages + birds_in_parakeet_only_cages
    in total_birds = total_birds_in_pet_store :=
by
  sorry

end pet_store_bird_count_l804_804743
