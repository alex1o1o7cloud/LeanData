import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.ConicSections
import Mathlib.Algebra.Fact
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.LinearEquiv
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Combinations
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Ellipse
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Triangle.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.DotProduct
import Mathlib.Logic.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Order

namespace filled_sandbag_weight_is_correct_l709_709649

-- Define the conditions
def sandbag_weight : ℝ := 250
def fill_percent : ℝ := 0.80
def heavier_factor : ℝ := 1.40

-- Define the intermediate weights
def sand_weight : ℝ := sandbag_weight * fill_percent
def extra_weight : ℝ := sand_weight * (heavier_factor - 1)
def filled_material_weight : ℝ := sand_weight + extra_weight

-- Define the total weight including the empty sandbag
def total_weight : ℝ := sandbag_weight + filled_material_weight

-- Prove the total weight is correct
theorem filled_sandbag_weight_is_correct : total_weight = 530 := 
by sorry

end filled_sandbag_weight_is_correct_l709_709649


namespace determine_a_l709_709160

def y (x : ℝ) (a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem determine_a (a : ℝ) : is_even (y _ a) → a = 2 :=
sorry

end determine_a_l709_709160


namespace train_length_is_correct_l709_709683

noncomputable def lengthOfTrain (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  speed_m_s * time_s

theorem train_length_is_correct : lengthOfTrain 60 15 = 250.05 :=
by
  sorry

end train_length_is_correct_l709_709683


namespace equal_differences_in_set_l709_709018

-- Axioms and abstractions relevant to the problem
axiom sorted (a : ℕ) (b : ℕ) : a < b
axiom in_set (a : ℕ) : Prop
axiom prop1 (a b : ℕ) (h : a > b) : in_set (a + b) ∨ in_set (a - b)

-- Given problem
theorem equal_differences_in_set (s : set ℕ) (h : ∀ a b, a ∈ s → b ∈ s → a > b → (a + b ∈ s ∨ a - b ∈ s)) :
  ∀ a1 a2 a3 ... a2003, 
    list.sorted (<) [a1, a2, a3, ..., a2003] →
    (∀ a, a ∈ s → 0 < a) → 
    (∀ (i j : ℕ), i + 1 = (j + 1) → s.to_list.nth i = some a1 ∧ s.to_list.nth (j + 1) = some a2 ∧ 
    (∀ k : ℕ, 1 ≤ k < 2003 → s.to_list.nth (k + 1).get_or_else 0 - s.to_list.nth k.get_or_else 0 = a2 - a1)) 
:= sorry

end equal_differences_in_set_l709_709018


namespace general_formula_an_l709_709285

/-- Defining the sequence a : ℕ → ℝ -/
def a : ℕ → ℝ 
| 0       := 1
| 1       := 1
| (n + 2) := (n^2 * a (n + 1) ^ 2 + 5) / ((n^2 - 1) * a n)

/-- Statement to be proven in Lean 4 -/
theorem general_formula_an :
  ∀ n : ℕ, n > 0 → 
  a n = (1 / n) * 
        ( (((63 - 13 * Real.sqrt 21) / 42) * ((5 + Real.sqrt 21) / 2) ^ n) + 
          (((63 + 13 * Real.sqrt 21) / 42) * ((5 - Real.sqrt 21) / 2) ^ n) ) := 
sorry

end general_formula_an_l709_709285


namespace calc_sub_neg_eq_add_problem_0_sub_neg_3_l709_709380

theorem calc_sub_neg_eq_add (a b : Int) : a - (-b) = a + b := by
  sorry

theorem problem_0_sub_neg_3 : 0 - (-3) = 3 := by
  exact calc_sub_neg_eq_add 0 3

end calc_sub_neg_eq_add_problem_0_sub_neg_3_l709_709380


namespace reducible_fraction_l709_709292

theorem reducible_fraction {n k : ℤ} (h : n = 13 * k + 8) :
  ∃ d : ℕ, d = 13 ∧ d ∣ (3 * n + 2) ∧ d ∣ (8 * n + 1) :=
begin
  sorry
end

end reducible_fraction_l709_709292


namespace quadratic_function_solution_l709_709218

theorem quadratic_function_solution :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, g (x + 1) - g x = 2 * x + 3 ∧ g 2 - g 6 = -40) :=
sorry

end quadratic_function_solution_l709_709218


namespace arithmetic_sequence_and_formula_l709_709982

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709982


namespace max_elements_set_l709_709978

theorem max_elements_set (M : Finset ℤ) 
(h : ∀ a b c ∈ M, ∃ x y ∈ M, x + y ∈ M) :
  M.card ≤ 2006 :=
sorry

end max_elements_set_l709_709978


namespace smallest_n_for_An_ending_in_0_l709_709804

theorem smallest_n_for_An_ending_in_0 :
  ∃ n : ℕ, (∀ m : ℕ, m < n → ¬(∏ k in finset.range m.succ, nat.choose (k^2) k) % 10 = 0)
    ∧ (∏ k in finset.range n.succ, nat.choose (k^2) k) % 10 = 0 ∧ n = 4 :=
  sorry

end smallest_n_for_An_ending_in_0_l709_709804


namespace alpha_beta_sufficient_necessary_l709_709469

theorem alpha_beta_sufficient_necessary
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  (α > β ↔ sin α > sin β) :=
by
  sorry

end alpha_beta_sufficient_necessary_l709_709469


namespace cuboid_height_l709_709412

theorem cuboid_height (SA l b : ℝ) (h : ℝ):
  SA = 720 →
  l = 12 →
  b = 6 →
  2 * (l * b + l * h + b * h) = 720 →
  h = 16 :=
by
  intros h_SA h_l h_b h_eq
  rw [h_SA, h_l, h_b] at h_eq
  sorry

end cuboid_height_l709_709412


namespace ratio_income_l709_709633

-- Define the problem conditions
variable (A B E_A E_B : ℕ)
variable (h1 : A = 2000)
variable (h2 : E_A / E_B = 3 / 2)
variable (h3 : A - E_A = 800)
variable (h4 : B - E_B = 800)

-- Translate the proof problem to Lean 4
theorem ratio_income : A / B = 5 / 4 :=
by 
  -- Using the given conditions above
  unfold h1 h2 h3 h4
  sorry

end ratio_income_l709_709633


namespace fixed_or_swapped_vertices_l709_709443

variable (V : Type) [Fintype V] [DecidableEq V]

structure Tree (V : Type) :=
  (vertices : Finset V)
  (edges : Finset (V × V))
  (is_tree : ∀ (x y : V), x ∈ vertices → y ∈ vertices → (x ≠ y → x ≠ y → ∃! (p : List V), p.head = x ∧ p.last = y ∧ ∀ (k : ℝ), k < (p.length - 1) → (p.nth k).get × (p.nth (k + 1)).get ∈ edges))

-- Define isomorphism
structure Isomorphism (T : Tree V) :=
  (f : V → V)
  (bij : Function.Bijective f)
  (preserve_edges : ∀ (x y : V), (x, y) ∈ T.edges → (f x, f y) ∈ T.edges)

variable (T : Tree V) (iso : Isomorphism T)

theorem fixed_or_swapped_vertices :
  (∃ a : V, iso.f a = a) ∨ (∃ a b : V, (a, b) ∈ T.edges ∧ iso.f a = b ∧ iso.f b = a) :=
sorry

end fixed_or_swapped_vertices_l709_709443


namespace area_swept_by_minute_hand_in_half_hour_l709_709306

-- Definitions according to given conditions
def radius := 15
def area_circle (r : ℝ) : ℝ := Real.pi * r^2

-- Lean statement for the proof problem
theorem area_swept_by_minute_hand_in_half_hour : area_circle radius / 2 = 1/2 * Real.pi * (15^2) :=
by
  sorry

end area_swept_by_minute_hand_in_half_hour_l709_709306


namespace rectangle_AE_length_l709_709522

theorem rectangle_AE_length
  (A B C D E : Point)
  (h1 : Rectangle A B C D)
  (h2 : dist A B = 20)
  (h3 : dist B C = 10)
  (h4 : E ∈ LineSegment C D)
  (h5 : ∠ C B E = 15) :
  dist A E = 20 :=
sorry

end rectangle_AE_length_l709_709522


namespace correct_statement_l709_709835

variable (p q : Prop)
variable (x y : ℝ)

-- Define propositional statement p
def p := ∀ x ≥ 0, 2^x ≥ 1

-- Define propositional statement q
def q := ∀ x y, x > y → x^2 > y^2

-- Prove the correct answer is p ∧ ¬q
theorem correct_statement : p ∧ ¬q := by
  sorry

end correct_statement_l709_709835


namespace minimum_balls_drawn_l709_709336

theorem minimum_balls_drawn :
  let
    red_balls := 21
    green_balls := 17
    yellow_balls := 24
    blue_balls := 10
    white_balls := 14
    black_balls := 14
  in
  ∃ n, (n = 84) →
   (∀ drawn : fin n, (drawn.val > 83 → (drawn.val - 83) > 0)) :=
begin
  sorry
end

end minimum_balls_drawn_l709_709336


namespace most_likely_outcome_l709_709064

theorem most_likely_outcome :
  let p := 1/2 in
  let prob_5_boys := p^5 in
  let prob_5_girls := p^5 in
  let comb_5_3 := Nat.choose 5 3 in
  let prob_3_boys_2_girls := comb_5_3 * p^5 in
  let comb_5_1 := Nat.choose 5 1 in
  let prob_4_boys_1_girl := comb_5_1 * p^5 in
  let prob_4_girls_1_boy := comb_5_1 * p^5 in
  let prob_4_1 := prob_4_boys_1_girl + prob_4_girls_1_boy in
  prob_3_boys_2_girls = 5/16 ∧ prob_4_1 = 5/16 ∧ 
  prob_5_boys = 1/32 ∧ prob_5_girls = 1/32 ∧
  prob_3_boys_2_girls = prob_4_1 ∧ prob_3_boys_2_girls > prob_5_boys ∧ prob_3_boys_2_girls > prob_5_girls := 
by
  sorry

end most_likely_outcome_l709_709064


namespace find_dividend_l709_709655

def dividend_problem (dividend divisor : ℕ) : Prop :=
  (15 * divisor + 5 = dividend) ∧ (dividend + divisor + 15 + 5 = 2169)

theorem find_dividend : ∃ dividend, ∃ divisor, dividend_problem dividend divisor ∧ dividend = 2015 :=
sorry

end find_dividend_l709_709655


namespace num_valid_arrays_l709_709883

def valid_array (A : Matrix (Fin 6) (Fin 6) ℤ) : Prop :=
  (∀ i, ∑ j, A i j = 0) ∧ (∀ j, ∑ i, A i j = 0) ∧ (∀ i j, A i j = 1 ∨ A i j = -1)

theorem num_valid_arrays : 
  ∃ (n : ℕ), n = 160400 ∧ ∀ A : Matrix (Fin 6) (Fin 6) ℤ, valid_array A → A ∈ (Matrix (λ (i j : Fin 6), {x : ℤ // x = 1 ∨ x = -1}) (Fin 6) (Fin 6)) :=
sorry

end num_valid_arrays_l709_709883


namespace line_distance_m_l709_709618

theorem line_distance_m (m : ℝ) :
    (∀ (x y : ℝ), 2 * x + 2 * y - 2 * m = 0) ∧ (∀ (x y : ℝ), 2 * x + 2 * y - 3 = 0) ∧
    (abs (2 * m - 3) / sqrt 8 = sqrt 2) → (m = -1 / 2 ∨ m = 7 / 2) :=
by
  sorry

end line_distance_m_l709_709618


namespace number_of_six_digit_decreasing_numbers_l709_709114

theorem number_of_six_digit_decreasing_numbers : 
  let S := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  in fintype.card { x : Finset ℕ // (x ⊆ S) ∧ (x.card = 6) } = 210 :=
sorry

end number_of_six_digit_decreasing_numbers_l709_709114


namespace square_area_proof_l709_709752

   theorem square_area_proof (x : ℝ) (h1 : 4 * x - 15 = 20 - 3 * x) :
     (20 - 3 * x) * (4 * x - 15) = 25 :=
   by
     sorry
   
end square_area_proof_l709_709752


namespace no_triangle_formed_l709_709507

def line1 (x y : ℝ) := y = -x
def line2 (x y : ℝ) := 4 * x + y = 3
def line3 (m x y : ℝ) := m * x + y + m - 1 = 0

theorem no_triangle_formed (m : ℝ) : ¬ (∃ x y z : ℝ, (line1 x y ∧ line2 x y ∧ line3 m x y)) ∨ (∃ t u : ℝ, (line3 t 0 u ∧ line1 t 0 u ∧ (line2 t 0 u))) → m = 4 :=
by sorry

end no_triangle_formed_l709_709507


namespace product_comparison_l709_709647

theorem product_comparison (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c := 
by
  let P_new := (1.1 * a) * (1.13 * b) * (0.8 * c)
  let P_original := a * b * c
  have h_coeff : 1.1 * 1.13 * 0.8 = 0.9944 := by norm_num
  have h : P_new = 0.9944 * P_original := by rw [←mul_assoc, h_coeff, ←mul_assoc, ←mul_assoc]
  rw h
  exact mul_lt_of_pos_left (by norm_num) (mul_pos ha (mul_pos hb hc))

end product_comparison_l709_709647


namespace algebraic_expression_evaluation_l709_709905

theorem algebraic_expression_evaluation (a b : ℤ) (h : a - 3 * b = -3) : 5 - a + 3 * b = 8 :=
by 
  sorry

end algebraic_expression_evaluation_l709_709905


namespace find_k_l709_709901

theorem find_k (k : ℝ) (h : ∫ x in 0..2, 3 * x^2 + k = 10) : k = 1 :=
sorry

end find_k_l709_709901


namespace lines_parallel_if_perpendicular_to_same_plane_l709_709876

-- Definitions and conditions
variables {Point : Type*} [MetricSpace Point]
variables {Line Plane : Type*}

def is_parallel (l₁ l₂ : Line) : Prop := sorry
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry

variables (m n : Line) (α : Plane)

-- Theorem statement
theorem lines_parallel_if_perpendicular_to_same_plane :
  is_perpendicular m α → is_perpendicular n α → is_parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l709_709876


namespace alpha_squared_plus_3alpha_plus_beta_equals_2023_l709_709899

-- Definitions and conditions
variables (α β : ℝ)
-- α and β are roots of the quadratic equation x² + 2x - 2025 = 0
def is_root_of_quadratic_1 : Prop := α^2 + 2 * α - 2025 = 0
def is_root_of_quadratic_2 : Prop := β^2 + 2 * β - 2025 = 0
-- Vieta's formula gives us α + β = -2
def sum_of_roots : Prop := α + β = -2

-- Theorem (statement) we want to prove
theorem alpha_squared_plus_3alpha_plus_beta_equals_2023 (h1 : is_root_of_quadratic_1 α)
                                                      (h2 : is_root_of_quadratic_2 β)
                                                      (h3 : sum_of_roots α β) :
                                                      α^2 + 3 * α + β = 2023 :=
by
  sorry

end alpha_squared_plus_3alpha_plus_beta_equals_2023_l709_709899


namespace option_c_equals_9_l709_709675

theorem option_c_equals_9 : (3 * 3 - 3 + 3) = 9 :=
by
  sorry

end option_c_equals_9_l709_709675


namespace polynomial_roots_cos_val_zero_l709_709609

theorem polynomial_roots_cos_val_zero 
  (a b c d : ℝ)
  (hQ : ∀ x : ℂ, x^4 + a*x^3 + b*x^2 + c*x + d = 0 ↔ 
    x = complex.cos (2 * real.pi / 9) ∨
    x = complex.cos (4 * real.pi / 9) ∨
    x = complex.cos (6 * real.pi / 9) ∨
    x = complex.cos (8 * real.pi / 9)) :
  abcd = 0 :=
sorry

end polynomial_roots_cos_val_zero_l709_709609


namespace profit_percent_l709_709332

theorem profit_percent (CP SP : ℕ) (h : CP * 5 = SP * 4) : 100 * (SP - CP) = 25 * CP :=
by
  sorry

end profit_percent_l709_709332


namespace chebyshevs_inequality_example_l709_709007

def X_vals : List ℝ := [0.3, 0.6]
def p_vals : List ℝ := [0.2, 0.8]

noncomputable def expectation_X : ℝ :=
  List.sum (List.zipWith (· *) X_vals p_vals)

noncomputable def variance_X : ℝ :=
  let M_X_sq := List.sum (List.zipWith (fun x p => (x * x) * p) X_vals p_vals)
  M_X_sq - (expectation_X * expectation_X)

noncomputable def chebyshevs_bound (ε : ℝ) : ℝ :=
  1 - variance_X / (ε * ε)

theorem chebyshevs_inequality_example :
  chebyshevs_bound 0.2 ≥ 0.64 := by
  sorry

end chebyshevs_inequality_example_l709_709007


namespace find_number_l709_709785

theorem find_number (x : ℝ) (h : x / 4 + 15 = 4 * x - 15) : x = 8 :=
sorry

end find_number_l709_709785


namespace min_ab_value_l709_709105

theorem min_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1 / a + 4 / b = Real.sqrt (a * b)) :
  a * b = 4 :=
  sorry

end min_ab_value_l709_709105


namespace range_of_a_l709_709925
noncomputable theory

def domain_is_real (a : ℝ) : Prop :=
  ∀ x : ℝ, ax^2 - 4ax + 2 > 0

theorem range_of_a (a : ℝ) : domain_is_real a ↔ a ∈ set.Ico 0 (1/2) := 
begin
  sorry
end

end range_of_a_l709_709925


namespace ben_average_score_increase_l709_709759

theorem ben_average_score_increase :
  let scores := [92, 89, 91]
  let fourth_score := 95
  let current_avg := (scores.sum / scores.length.toReal : ℚ)
  let new_avg := ((scores.sum + fourth_score) / (scores.length + 1).toReal : ℚ)
  new_avg - current_avg = 1.08 :=
by
  sorry

end ben_average_score_increase_l709_709759


namespace remaining_money_after_payments_l709_709128

-- Conditions
def initial_money : ℕ := 100
def paid_colin : ℕ := 20
def paid_helen : ℕ := 2 * paid_colin
def paid_benedict : ℕ := paid_helen / 2
def total_paid : ℕ := paid_colin + paid_helen + paid_benedict

-- Proof
theorem remaining_money_after_payments : 
  initial_money - total_paid = 20 := by
  sorry

end remaining_money_after_payments_l709_709128


namespace smallest_positive_integer_n_l709_709666

theorem smallest_positive_integer_n :
  ∃ (n : ℕ), 5 * n ≡ 1978 [MOD 26] ∧ n = 16 :=
by
  sorry

end smallest_positive_integer_n_l709_709666


namespace minimum_value_of_y_l709_709308

theorem minimum_value_of_y : ∀ x : ℝ, ∃ y : ℝ, (y = 3 * x^2 + 6 * x + 9) → y ≥ 6 :=
by
  intro x
  use (3 * (x + 1)^2 + 6)
  intro h
  sorry

end minimum_value_of_y_l709_709308


namespace ratio_of_linear_combination_l709_709087

theorem ratio_of_linear_combination (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (3 * a + 2 * b) / (b + 4 * c) = 3 / 17 :=
by
  sorry

end ratio_of_linear_combination_l709_709087


namespace addition_addends_l709_709588

theorem addition_addends (a b : ℕ) (c₁ c₂ : ℕ) (d : ℕ) : 
  a + b = c₁ ∧ a + (b - d) = c₂ ∧ d = 50 ∧ c₁ = 982 ∧ c₂ = 577 → 
  a = 450 ∧ b = 532 :=
by
  sorry

end addition_addends_l709_709588


namespace find_phi_l709_709625

theorem find_phi (ϕ : ℝ) (h1 : |ϕ| < π / 2)
  (h2 : ∃ k : ℤ, 3 * (π / 12) + ϕ = k * π + π / 2) :
  ϕ = π / 4 :=
by sorry

end find_phi_l709_709625


namespace find_a_if_y_is_even_l709_709163

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem find_a_if_y_is_even (a : ℝ) (h : ∀ x : ℝ, y x a = y (-x) a) : a = 2 :=
by
  sorry

end find_a_if_y_is_even_l709_709163


namespace printing_shop_paper_boxes_l709_709340

variable (x y : ℕ) -- Assuming x and y are natural numbers since the number of boxes can't be negative.

theorem printing_shop_paper_boxes (h1 : 80 * x + 180 * y = 2660)
                                  (h2 : x = 5 * y - 3) :
    x = 22 ∧ y = 5 := sorry

end printing_shop_paper_boxes_l709_709340


namespace average_greater_than_median_by_12_l709_709933

theorem average_greater_than_median_by_12 : 
  let weights := [85, 12, 12, 15, 18, 10] in
  let sorted_weights := [10, 12, 12, 15, 18, 85] in
  let median := (12 + 15) / 2 in
  let average := (85 + 12 + 12 + 15 + 18 + 10) / 6 in
  average - median = 12 := 
by
  sorry

end average_greater_than_median_by_12_l709_709933


namespace find_Xd_sub_Yd_l709_709256

theorem find_Xd_sub_Yd (d X Y : ℕ) (hd : 8 < d)
  (h : d * X + Y + d * X + X = d * d + 3 * d + 4) :
  (X - Y) = -2 :=
by
  sorry

end find_Xd_sub_Yd_l709_709256


namespace fifth_term_in_arithmetic_sequence_l709_709272

theorem fifth_term_in_arithmetic_sequence (x y : ℝ)
    (h1 : x - 2 * y + 4 * y = x + 2 * y)  
    (h2 : x + 2 * y + (x^2 - y^2 - (x + 2 * y)) = x^2 - y^2) : 
    ( (4 * y = x^2 - y^2 - x - 2 * y) ∧ (4 * y = (x^2 / y^2) - (x^2 - y^2)) ∧ ((x = 3 * y / (1 - y)) ∧ (y = 1/2) ∨ (y = -1/3)) ∧ (x = 3) ) →  (5 * y = 20):=
begin
  sorry
end

end fifth_term_in_arithmetic_sequence_l709_709272


namespace deriv_of_f_at_0_l709_709627

-- Definitions
def f (x : ℝ) : ℝ := 2 * Real.exp (Real.sin x)

def deriv_f_at_0 : ℝ :=
  (deriv f 0)

-- Theorem
theorem deriv_of_f_at_0 :
  deriv_f_at_0 = 2 * Real.exp 1 := by
  sorry

end deriv_of_f_at_0_l709_709627


namespace proof_probability_p_s_two_less_than_multiple_of_7_l709_709296

noncomputable def probability_p_s_two_less_than_multiple_of_7 : ℚ :=
  let possible_integers := {a : ℕ | 1 ≤ a ∧ a ≤ 100}
  let distinct_pairs := {p : ℕ × ℕ | p.1 ∈ possible_integers ∧ p.2 ∈ possible_integers ∧ p.1 ≠ p.2}
  let valid_pairs := {p : ℕ × ℕ | p ∈ distinct_pairs ∧ (let a := p.1, b := p.2 in
                        let s := a + b
                        let p := a * b
                        p + s ≡ 5 [MOD 7])}
  (fintype.card valid_pairs : ℚ) / (fintype.card distinct_pairs : ℚ)

theorem proof_probability_p_s_two_less_than_multiple_of_7 :
  probability_p_s_two_less_than_multiple_of_7 = 7 / 330 :=
sorry

end proof_probability_p_s_two_less_than_multiple_of_7_l709_709296


namespace problem1_problem2_problem3_l709_709300

/-- Problem 1: Calculate 25 * 26 * 8 and show it equals 5200 --/
theorem problem1 : 25 * 26 * 8 = 5200 := 
sorry

/-- Problem 2: Calculate 340 * 40 / 17 and show it equals 800 --/
theorem problem2 : 340 * 40 / 17 = 800 := 
sorry

/-- Problem 3: Calculate 440 * 15 + 480 * 15 + 79 * 15 + 15 and show it equals 15000 --/
theorem problem3 : 440 * 15 + 480 * 15 + 79 * 15 + 15 = 15000 := 
sorry

end problem1_problem2_problem3_l709_709300


namespace whale_sixth_hour_l709_709324

-- Conditions
variables (x : ℕ)

-- Let the plankton consumed in the i-th hour be denoted by a function f(i)
def f (i : ℕ) : ℕ :=
  if i = 1 then x
  else x + 3 * (i - 1)

-- Total consumption over 9 hours must be equal to 270
def total_consumption : ℕ := ∑ i in (finset.range 9).map ((+) 1 : ℕ → ℕ), f x i

-- Prove that the whale consumed 33 kilos on the sixth hour
theorem whale_sixth_hour : total_consumption x = 270 → f x 6 = 33 :=
by
  -- Proof steps to be added
  sorry

end whale_sixth_hour_l709_709324


namespace arithmetic_sequence_and_formula_l709_709991

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709991


namespace interest_diff_is_272_l709_709727

noncomputable def principal : ℝ := 400
noncomputable def rate : ℝ := 0.04
noncomputable def time : ℕ := 8

theorem interest_diff_is_272 : 
  let interest := principal * rate * (time : ℝ) in
  let difference := principal - interest in
  difference = 272 :=
by
  let interest : ℝ := principal * rate * (time : ℝ)
  let difference : ℝ := principal - interest
  have : difference = 272 := by sorry
  exact this

end interest_diff_is_272_l709_709727


namespace Wednesday_sleep_hours_Friday_liters_of_tea_l709_709630

-- Define the relation of inverse proportionality
def inversely_proportional (k : ℝ) (h : ℝ) (t : ℝ) : Prop :=
  t * h = k

-- The given constant from Sunday
def k : ℝ := 12 * 1.5

-- Prove the sleep hours on Wednesday
theorem Wednesday_sleep_hours (h_wednesday : ℝ) (t_wednesday : ℝ) (h_sunday : ℝ) (t_sunday : ℝ) :
  inversely_proportional k h_sunday t_sunday → t_wednesday = 2 → h_wednesday = 9 :=
by
  intros h_sunday t_sunday invert_prop t_eq
  sorry

-- Prove the liters of tea on Friday
theorem Friday_liters_of_tea (h_friday : ℝ) (t_friday : ℝ) (h_sunday : ℝ) (t_sunday : ℝ) :
  inversely_proportional k h_sunday t_sunday → h_friday = 8 → t_friday = 2.25 :=
by
  intros invert_prop h_eq
  sorry

end Wednesday_sleep_hours_Friday_liters_of_tea_l709_709630


namespace james_marbles_l709_709202

def marbles_in_bag_D (bag_C : ℕ) := 2 * bag_C - 1
def marbles_in_bag_E (bag_A : ℕ) := bag_A / 2
def marbles_in_bag_G (bag_E : ℕ) := bag_E

theorem james_marbles :
    ∀ (A B C D E F G : ℕ),
      A = 4 →
      B = 3 →
      C = 5 →
      D = marbles_in_bag_D C →
      E = marbles_in_bag_E A →
      F = 3 →
      G = marbles_in_bag_G E →
      28 - (D + F) + 4 = 20 := by
    intros A B C D E F G hA hB hC hD hE hF hG
    sorry

end james_marbles_l709_709202


namespace indistinguishable_balls_boxes_l709_709891

noncomputable def ways_to_distribute (balls : ℕ) (boxes : ℕ) : ℕ :=
  if boxes = 2 then
    if balls = 5 then
      3
    else
      sorry  -- Not needed for this problem
  else 
    sorry  -- Not needed for this problem

theorem indistinguishable_balls_boxes
  (balls : ℕ) (boxes : ℕ) : 
  boxes = 2 → balls = 5 → ways_to_distribute balls boxes = 3 :=
by
  intros h1 h2
  simp [ways_to_distribute, h1, h2]
  sorry

end indistinguishable_balls_boxes_l709_709891


namespace odd_function_property_l709_709558

theorem odd_function_property (f : ℝ → ℝ)
  (h1 : ∀ x, f (2 - x) + f x = 0)
  (h2 : ∀ x, f (-x) = -f x) :
  f 2022 + f 2023 = 0 :=
by 
  let f_periodic := λ x, calc
    f (2 + x) = f (2 - (-x)) : by rw [sub_neg_eq_add]
           ... = -f (-x) : by rw [h1 (-x)]
           ... = -(-f x) : by rw [h2 x]
           ... = f x : by rw [neg_neg]
  have f0 : f 0 = 0, from calc
    f 0 = -f 0 : by rw [←h2 0]
       ... = 0 : by linarith
  have f1 : f 1 = 0, from calc
    0 = f (2 - 1) + f 1 : by rw [h1 1]
      ... = f 1 + f 1 : by rw [sub_self]
      ... = 2 * f 1 : by ring
      ... = f 1 : by field_simp only [zero_eq_mul]

  have f2022_eq_f0 : f 2022 = f 0 := by exact f_periodic (2022-2 * 1011)
  have f2023_eq_f1 : f 2023 = f 1 := by exact f_periodic (2023-2 * 1011)
  
  calc f 2022 + f 2023 = f 0 + f 1 : by rw [f2022_eq_f0, f2023_eq_f1]
                      ... = 0 : by rw [f0, f1]

end odd_function_property_l709_709558


namespace truck_transportation_l709_709656

theorem truck_transportation
  (x y t : ℕ) 
  (h1 : xt - yt = 60)
  (h2 : (x - 4) * (t + 10) = xt)
  (h3 : (y - 3) * (t + 10) = yt)
  (h4 : xt = x * t)
  (h5 : yt = y * t) : 
  x - 4 = 8 ∧ y - 3 = 6 ∧ t + 10 = 30 := 
by
  sorry

end truck_transportation_l709_709656


namespace value_of_b_plus_a_l709_709072

theorem value_of_b_plus_a (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 2) (h3 : |a - b| = |b - a|) : b + a = -6 ∨ b + a = -10 :=
by
  sorry

end value_of_b_plus_a_l709_709072


namespace decreasing_sequence_l709_709471

def f (x : ℝ) : ℝ := (3/2) * x + (Real.log (x - 1))
def f_prime (x : ℝ) : ℝ := (3/2) + (1 / (x - 1))

def sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a (n + 1) = f_prime (a n + 1)) ∧ (∀ n : ℕ, a n > 0)

def b_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a (2 * n)

theorem decreasing_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) (h₁ : sequence a) (h₂ : b_sequence a b)
  (h₃ : 0 < a 1 ∧ a 1 < 2) : ∀ n : ℕ, b (n + 1) < b n :=
by
  sorry

end decreasing_sequence_l709_709471


namespace fill_675_cans_time_l709_709351

theorem fill_675_cans_time :
  (∀ (cans_per_batch : ℕ) (time_per_batch : ℕ) (total_cans : ℕ),
    cans_per_batch = 150 →
    time_per_batch = 8 →
    total_cans = 675 →
    total_cans / cans_per_batch * time_per_batch = 36) :=
begin
  intros cans_per_batch time_per_batch total_cans h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry
end

end fill_675_cans_time_l709_709351


namespace average_income_l709_709338

theorem average_income :
  let incomes := [400, 250, 650, 400, 500]
  ∑ x in incomes, x / incomes.length = 440 :=
by {
  sorry
}

end average_income_l709_709338


namespace g_at_2_l709_709495

def g (x : ℝ) : ℝ := x^2 - 4

theorem g_at_2 : g 2 = 0 := by
  sorry

end g_at_2_l709_709495


namespace garage_motorcycles_l709_709637

theorem garage_motorcycles (bicycles cars motorcycles total_wheels : ℕ)
  (hb : bicycles = 20)
  (hc : cars = 10)
  (hw : total_wheels = 90)
  (wb : bicycles * 2 = 40)
  (wc : cars * 4 = 40)
  (wm : motorcycles * 2 = total_wheels - (bicycles * 2 + cars * 4)) :
  motorcycles = 5 := 
  by 
  sorry

end garage_motorcycles_l709_709637


namespace villagers_proportion_half_l709_709737

theorem villagers_proportion_half (n : ℕ) (truthful_villager : ℕ → Prop) (liar_villager : ℕ → Prop)
(truth_or_liar : ∀ i, truthful_villager i ∨ liar_villager i)
(circle : ∀ i, truthful_villager i ↔ liar_villager (i+1) % n) :
  (∃ k, k = n ∧ k > 0 ∧ 
    ( ∀ i : ℕ, truthful_villager i ↔ (i < k) ∧ (∀ j : ℕ, j < n/2 → j < k))
    ∧ ( ∀ i : ℕ, liar_villager i ↔ (i ≥ k) ∧ (∀ j : ℕ, j < n/2 → j ≥ k))
  ) :=
sorry

end villagers_proportion_half_l709_709737


namespace find_A_when_A_clubsuit_7_equals_61_l709_709915

-- Define the operation
def clubsuit (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 7

-- Define the main problem statement
theorem find_A_when_A_clubsuit_7_equals_61 : 
  ∃ A : ℝ, clubsuit A 7 = 61 ∧ A = (2 * Real.sqrt 30) / 3 :=
by
  sorry

end find_A_when_A_clubsuit_7_equals_61_l709_709915


namespace max_value_of_d_l709_709217

-- Define the real numbers a, b, c, d
variables {a b c d : ℝ}

-- Define the conditions
def condition1 : Prop := a + b + c + d = 10
def condition2 : Prop := ab + ac + ad + bc + bd + cd = 20

-- Define the statement to find the maximum value of d
theorem max_value_of_d (h1 : condition1) (h2 : condition2) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_of_d_l709_709217


namespace valentines_given_l709_709590

-- Let x be the number of boys and y be the number of girls
variables (x y : ℕ)

-- Condition 1: the number of valentines is 28 more than the total number of students.
axiom valentines_eq : x * y = x + y + 28

-- Theorem: Prove that the total number of valentines given is 60.
theorem valentines_given : x * y = 60 :=
by
  sorry

end valentines_given_l709_709590


namespace chickens_and_rabbits_l709_709942

theorem chickens_and_rabbits (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x = 23 ∧ y = 12 :=
by
  sorry

end chickens_and_rabbits_l709_709942


namespace max_value_x_plus_2y_on_ellipse_l709_709552

theorem max_value_x_plus_2y_on_ellipse : 
  ∀ (x y : ℝ), (2 * x^2 + 3 * y^2 = 12) → x + 2 * y ≤ Real.sqrt 22 :=
begin
  sorry
end

end max_value_x_plus_2y_on_ellipse_l709_709552


namespace range_of_a_l709_709922

theorem range_of_a {a : ℝ} (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, (a*x + 1)/(sqrt (a*x^2 - 4*a*x + 2)) (f x)) :
  (0 ≤ a) ∧ (a < 1/2) :=
sorry

end range_of_a_l709_709922


namespace skew_lines_angle_range_l709_709280

theorem skew_lines_angle_range :
  ∀ (L₁ L₂ : ℝ → ℝ → ℝ),
  skew_lines L₁ L₂ →
  ∃ θ ∈ Ioc 0 (real.pi / 2),
    ∀ P : ℝ × ℝ,
    acute_or_right_angle (parallel_line_through_point L₁ P) (parallel_line_through_point L₂ P) θ :=
by
  sorry

end skew_lines_angle_range_l709_709280


namespace digit_at_150_is_five_l709_709669

theorem digit_at_150_is_five : 
  ∀ (n : ℕ), 
  (0 < n) → 
  (n = 150) → 
  (decimal_repr (5 / 13) = "0.384615") → 
  (repeats_every_six_digit : ("0.384615" / 6 = 25)) → -- repeats every 6 digits
  (nth_digit n (decimal_repr (5 / 13)) = 5) := 
by
  intros
  sorry

end digit_at_150_is_five_l709_709669


namespace find_ellipse_equation_l709_709463

-- Definitions based on conditions
def ellipse_centered_at_origin (x y : ℝ) (m n : ℝ) := m * x ^ 2 + n * y ^ 2 = 1

def passes_through_points_A_and_B (m n : ℝ) := 
  (ellipse_centered_at_origin 0 (-2) m n) ∧ (ellipse_centered_at_origin (3 / 2) (-1) m n)

-- Statement to be proved
theorem find_ellipse_equation : 
  ∃ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ (m ≠ n) ∧ 
  passes_through_points_A_and_B m n ∧ 
  m = 1 / 3 ∧ n = 1 / 4 :=
by sorry

end find_ellipse_equation_l709_709463


namespace remainder_n_pow_5_minus_n_mod_30_l709_709226

theorem remainder_n_pow_5_minus_n_mod_30 (n : ℤ) : (n^5 - n) % 30 = 0 := 
by sorry

end remainder_n_pow_5_minus_n_mod_30_l709_709226


namespace angle_MDC_relation_l709_709175

variables (A B C D M : Type*)
variables [euclidean_geometry A B C D M]

-- Assume that A, B, and C are points forming a triangle
variables (triangle_ABC : triangle A B C)

-- Define the median AM to side BC with M as the midpoint
def is_median (AM : segment A M) (BC : segment B C) (M : midpoint M B C) : Prop :=
segment A M = segment A (midpoint B C)

-- Extend AB to point D
def extends_to (AB : line A B) (D : point) : Prop :=
on_line AB D ∧ ∃ l, perpendicular l (segment M C) ∧ on_line l D

-- Define the external angle theorem
def angle_external_theorem (triangle_ABC : triangle A B C) (C D : point) : Prop :=
angle D M C = angle A B C + angle B A C

-- The theorem statement
theorem angle_MDC_relation (triangle_ABC : triangle A B C) (AM : segment A M) (BC : segment B C)
    (M : midpoint M B C) (D : point) (h1 : is_median AM BC M)
    (h2 : extends_to (line A B) D) : 
    angle D M C = 180 - angle A C B :=
by
  sorry

end angle_MDC_relation_l709_709175


namespace y_value_is_32_l709_709135

-- Define the conditions
variables (y : ℝ) (hy_pos : y > 0) (hy_eq : y^2 = 1024)

-- State the theorem
theorem y_value_is_32 : y = 32 :=
by
  -- The proof will be written here
  sorry

end y_value_is_32_l709_709135


namespace balls_diff_color_probability_l709_709177

-- Defining the number of each color balls in the bag
def blue_balls := 1
def red_balls := 1
def yellow_balls := 2
def total_balls := blue_balls + red_balls + yellow_balls

-- Defining the event of drawing two balls of different colors
def event_diff_color := 
  let total_draw := 2
  let total_ways := Nat.choose total_balls total_draw
  let same_yellow_ways := Nat.choose yellow_balls total_draw
  (total_ways - same_yellow_ways) / total_ways

-- Theorem statement
theorem balls_diff_color_probability : event_diff_color = (5 / 6) :=
by
  sorry

end balls_diff_color_probability_l709_709177


namespace ratio_is_one_half_l709_709239

-- Define the problem conditions as constants
def robert_age_in_2_years : ℕ := 30
def years_until_robert_is_30 : ℕ := 2
def patrick_current_age : ℕ := 14

-- Using the conditions, set up the definitions for the proof
def robert_current_age : ℕ := robert_age_in_2_years - years_until_robert_is_30

-- Define the target ratio
def ratio_of_ages : ℚ := patrick_current_age / robert_current_age

-- Prove that the ratio of Patrick's age to Robert's age is 1/2
theorem ratio_is_one_half : ratio_of_ages = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l709_709239


namespace B_completes_work_in_days_l709_709694

theorem B_completes_work_in_days (A_days : ℕ) (B_efficiency_increase : ℕ) :
  A_days = 12 → B_efficiency_increase = 75 → ∃ B_days : ℝ, B_days ≈ 6.86 :=
by
  intros hA hB
  let A_work_rate := 1 / A_days
  let B_work_rate := (1 + B_efficiency_increase / 100) * A_work_rate
  let B_days := 1 / B_work_rate
  use B_days
  sorry

end B_completes_work_in_days_l709_709694


namespace exists_pos_integer_n_l709_709494

theorem exists_pos_integer_n (n : ℕ) (hn_pos : n > 0) (h : ∃ m : ℕ, m * m = 1575 * n) : n = 7 :=
sorry

end exists_pos_integer_n_l709_709494


namespace prairie_total_area_l709_709344

theorem prairie_total_area (dust : ℕ) (untouched : ℕ) (total : ℕ) 
  (h1 : dust = 64535) (h2 : untouched = 522) : total = dust + untouched :=
by
  sorry

end prairie_total_area_l709_709344


namespace monotonicity_and_extrema_of_f_l709_709858

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

theorem monotonicity_and_extrema_of_f :
  (∀ x1 x2, x1 ∈ set.Icc (1:ℝ) 3 → x2 ∈ set.Icc (1:ℝ) 3 → x1 < x2 → f x1 < f x2) ∧
  (f 1 = 0) ∧ (f 3 = 1 / 2) := 
by
  sorry

end monotonicity_and_extrema_of_f_l709_709858


namespace arithmetic_sequence_and_formula_l709_709986

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709986


namespace January_to_November_ratio_l709_709342

variable (N D J : ℝ)

-- Condition 1: November revenue is 3/5 of December revenue
axiom revenue_Nov : N = (3 / 5) * D

-- Condition 2: December revenue is 2.5 times the average of November and January revenues
axiom revenue_Dec : D = 2.5 * (N + J) / 2

-- Goal: Prove the ratio of January revenue to November revenue is 1/3
theorem January_to_November_ratio : J / N = 1 / 3 :=
by
  -- We will use the given axioms to derive the proof
  sorry

end January_to_November_ratio_l709_709342


namespace Vasya_can_win_l709_709302

theorem Vasya_can_win 
  (a : ℕ → ℕ) -- initial sequence of natural numbers
  (x : ℕ) -- number chosen by Vasya
: ∃ (i : ℕ), ∀ (k : ℕ), ∃ (j : ℕ), (a j + k * x = 1) :=
by
  sorry

end Vasya_can_win_l709_709302


namespace find_a2_b2_c2_l709_709767

-- Defining the conditions
def is_valid_digit (n : ℕ) : Prop := n >= 0 ∧ n < 10

def all_different (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_valid_reading (a b c : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c

def odometer_reading_start (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def odometer_reading_end (a b c : ℕ) : ℕ := 100 * b + 10 * c + a

def travel_distance (a b c : ℕ) : ℕ :=
  odometer_reading_end b c a - odometer_reading_start a b c

-- Setting the values of a, b, c
def a : ℕ := 2
def b : ℕ := 3
def c : ℕ := 8

-- Prove the main statement
theorem find_a2_b2_c2 : a^2 + b^2 + c^2 = 77 :=
by
  have valid_values : is_valid_reading a b c := 
    by 
      simp [is_valid_digit, a, b, c]
      exact ⟨⟨nat.zero_le 2, nat.lt_succ_self 1⟩,
             ⟨nat.zero_le 3, nat.lt_succ_self 2⟩,
             ⟨nat.zero_le 8, nat.lt_succ_self 7⟩⟩

  have different_values : all_different a b c := 
    by 
      simp [a, b, c]

  have travel_div_60 : 60 ∣ travel_distance a b c :=
    by
      simp [travel_distance, odometer_reading_start, odometer_reading_end, a, b, c]
      exact dvd.intro 3 rfl

  show a^2 + b^2 + c^2 = 77,
    by
      simp [a, b, c]
      norm_num

-- Sorry to skip the actual proof steps
sorry

end find_a2_b2_c2_l709_709767


namespace spherical_to_rectangular_conversion_l709_709389

-- Define spherical to rectangular coordinate conversion
def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular 4 π (π / 3) = (-2 * Real.sqrt 3, 0, 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l709_709389


namespace equilateral_triangle_rectangles_l709_709213

theorem equilateral_triangle_rectangles {A B C: Point} (h : equilateral_triangle A B C) : 
  num_rectangles_share_two_vertices A B C = 3 :=
sorry

end equilateral_triangle_rectangles_l709_709213


namespace division_multiplication_calculation_l709_709376

theorem division_multiplication_calculation :
  (30 / (7 + 2 - 3)) * 4 = 20 :=
by
  sorry

end division_multiplication_calculation_l709_709376


namespace three_digit_sum_remainder_l709_709980

def S := (∑ i in (finset.filter (λ x : ℕ, (x / 100 ≠ x / 10 % 10) ∧ (x / 100 ≠ x % 10) ∧ (x / 10 % 10 ≠ x % 10)) (finset.range 1000)), i)

theorem three_digit_sum_remainder : S % 1000 = 680 :=
sorry

end three_digit_sum_remainder_l709_709980


namespace dogs_at_pet_store_is_10_l709_709354

def calculate_dogs : Nat :=
  let start := 2
  let after_sunday := start + 5 - 2
  let after_monday := after_sunday + 3 + 0.5
  let after_tuesday := after_monday + 4 - 3
  let after_wednesday := (after_tuesday * 0.4).toNat + 7
  after_wednesday

theorem dogs_at_pet_store_is_10 : calculate_dogs = 10 :=
by sorry

end dogs_at_pet_store_is_10_l709_709354


namespace polar_to_cartesian_circle_l709_709104

open Real

-- Given the polar coordinate equation of a circle, we need to prove the following:
theorem polar_to_cartesian_circle :
  (∀ θ ρ, ρ^2 - 4 * sqrt 2 * ρ * cos (θ - π/4) + 6 = 0) →
  (∀ x y, (x = 2 + sqrt 2 * cos θ) → (y = 2 + sqrt 2 * sin θ) →
  x^2 + y^2 - 4 * x - 4 * y + 6 = 0) ∧ 
  (∀ x y, (x = 2 + sqrt 2 * cos θ) → (y = 2 + sqrt 2 * sin θ) →
  (∀ t, t = sin θ + cos θ → (xy = t^2 + 2 * sqrt 2 * t + 3 ∧ 
  min xy = 1 ∧ max xy = 9)))
sorry

end polar_to_cartesian_circle_l709_709104


namespace range_a_ge_find_ab_range_a_le_l709_709474

-- Question 1
theorem range_a_ge (a : ℝ) :
  (∀ x : ℝ, f x = x^2 - (a + 1) * x + 1 ∧ f x ≥ 0) ↔ (-3 ≤ a ∧ a ≤ 1) :=
sorry

-- Question 2
theorem find_ab (a b : ℝ) :
  (∀ x f x = x^2 - (a + 1) * x + 1 ∧ (∃ b, ∀ x, f x < 0 ↔ b < x ∧ x < 2)) ↔ (a = 3 / 2 ∧ b = 1 / 2) :=
sorry

-- Question 3
theorem range_a_le (a : ℝ) (Q : Set ℝ) :
  (Q = { x | 0 ≤ x ∧ x ≤ 1 } ∧ ∀ x : ℝ, f x = x^2 - (a + 1) * x + 1 ∧ f x ≤ 0 ↔ x ∈ Q) ∧
  (∀ x, x ∈ Q → f x > 0) ↔ (a < 1) :=
sorry

end range_a_ge_find_ab_range_a_le_l709_709474


namespace max_tan_C_l709_709950

variable {A B C : ℝ}

def in_triangle (A B C : ℝ) : Prop := 
  A > 0 ∧ A < π / 2 ∧ B > 0 ∧ B < π / 2 ∧ C > 0 ∧ C < π / 2 ∧ A + B + C = π

def condition (A B C : ℝ) : Prop := 
  1 / Real.tan A + 1 / Real.tan B + Real.tan C = 0

theorem max_tan_C (A B C : ℝ) (h_triangle : in_triangle A B C) (h_condition : condition A B C) :
  Real.tan C ≤ -2 * Real.sqrt 2 :=
sorry

end max_tan_C_l709_709950


namespace three_digit_numbers_with_perfect_square_sum_l709_709123

theorem three_digit_numbers_with_perfect_square_sum :
  ∃ (n : ℕ), n = 58 ∧ ∀ (a b c : ℕ), (100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c < 1000) →
  ∃ (s : ℕ), a + b + c = s ∧ s ∈ {1, 4, 9, 16, 25} :=
by sorry

end three_digit_numbers_with_perfect_square_sum_l709_709123


namespace john_must_sell_883_disks_to_make_150_profit_l709_709204

noncomputable def cost_per_disk : ℝ := 4 / 3
noncomputable def selling_price_per_disk : ℝ := 6 / 4
noncomputable def profit_per_disk : ℝ := selling_price_per_disk - cost_per_disk
noncomputable def desired_profit : ℝ := 150

theorem john_must_sell_883_disks_to_make_150_profit :
  ceil (desired_profit / profit_per_disk) = 883 := 
sorry

end john_must_sell_883_disks_to_make_150_profit_l709_709204


namespace football_group_stage_teams_l709_709183

theorem football_group_stage_teams :
  let num_games := 6 * 4 * 10 in
  ∃ x : ℕ, x * (x - 1) = num_games ∧ x = 16 :=
by
  let num_games := 6 * 4 * 10
  existsi 16
  have h : 16 * 15 = num_games := by norm_num
  split
  assumption
  refl

end football_group_stage_teams_l709_709183


namespace max_value_f_l709_709416

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x - Real.tan x

theorem max_value_f : 
  ∃ x ∈ Set.Ioo 0 (Real.pi / 2), ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≤ f x ∧ f x = 3 * Real.sqrt 3 :=
by
  sorry

end max_value_f_l709_709416


namespace smallest_five_digit_multiple_of_18_l709_709423

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, n = 10008 ∧ (n ≥ 10000 ∧ n < 100000) ∧ n % 18 = 0 ∧ (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 18 = 0 → n ≤ m) := sorry

end smallest_five_digit_multiple_of_18_l709_709423


namespace increasing_interval_l709_709390

noncomputable def f (x : ℝ) : ℝ := Real.log (4 + 3 * x - x^2)

theorem increasing_interval : 
  ∃ a b : ℝ, a = -1 ∧ b = 3 / 2 ∧ ∀ x : Ioo a b, 4 + 3 * x - x^2 > 0 ∧ ∀ ⦃y : ℝ⦄, y ∈ Ioo a b → f' y > 0 :=
begin
  sorry
end

end increasing_interval_l709_709390


namespace range_of_a_l709_709927

variable (a : ℝ)

theorem range_of_a (h : ∀ x : ℝ, a * x^2 - a^2 * x - 2 ≤ 0) : a ∈ Icc (-2 : ℝ) 0 :=
sorry

end range_of_a_l709_709927


namespace amar_score_l709_709519

theorem amar_score (A B C : ℝ) (M : ℝ) (average_score : ℝ) 
  (hB: B = 0.36) (hC: C = 0.44) (hM: M = 900) (h_avg: (A * M / 100 + B * M + C * M) / 3 = average_score) :
  A = 64 := 
by 
  have h : (A * M / 100 + 0.36 * M + 0.44 * M) / 3 = average_score := by rwa [hB, hC]
  simp at h
  sorry

end amar_score_l709_709519


namespace find_radius_of_film_l709_709713

variables {h d t V_film R : ℝ}

-- Define the given conditions
def cylinder_height := 10
def cylinder_diameter := 5
def film_thickness := 0.2

-- Calculate the radius of the cylinder
def cylinder_radius := cylinder_diameter / 2

-- Volume of the cylindrical container
def V_cylinder := π * cylinder_radius^2 * cylinder_height

-- Volume of the circular film formed on water
def V_film := π * R^2 * film_thickness

-- The proof statement
theorem find_radius_of_film : R = Real.sqrt (312.5) :=
  sorry

end find_radius_of_film_l709_709713


namespace closest_to_sqrt_2_l709_709680

theorem closest_to_sqrt_2:
  let A := sqrt 3 * cos 14 + sin 14,
      B := sqrt 3 * cos 24 + sin 24,
      C := sqrt 3 * cos 64 + sin 64,
      D := sqrt 3 * cos 74 + sin 74
  in abs (D - sqrt 2) < abs (B - sqrt 2) ∧ abs (D - sqrt 2) < abs (A - sqrt 2) ∧ abs (D - sqrt 2) < abs (C - sqrt 2)
:= sorry

end closest_to_sqrt_2_l709_709680


namespace proj_a_onto_b_l709_709109

def vector (α : Type*) := (α × α)

noncomputable def proj_vector (a b : vector ℝ) : vector ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_squared := b.1 ^ 2 + b.2 ^ 2
  (dot_product / magnitude_squared * b.1, dot_product / magnitude_squared * b.2)

theorem proj_a_onto_b :
  let a := (-2 : ℝ, 3 : ℝ)
  let b := (0 : ℝ, 4 : ℝ)
  proj_vector a b = (0, 3) :=
by
  sorry

end proj_a_onto_b_l709_709109


namespace a3_eq_10_l709_709854

-- Definition of sequences and initial conditions
def seq (a : ℕ → ℕ) (S : ℕ → ℕ) := ∀ n : ℕ, a (n + 1) = 2 * S n + n
def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) := ∀ n : ℕ, S n = (finset.range(n + 1)).sum a
def initial_cond (a : ℕ → ℕ) := a 1 = 1

-- Proving that a_3 = 10 given the conditions
theorem a3_eq_10 {a S : ℕ → ℕ} (h1 : initial_cond a) (h2 : seq a S) (h3 : sum_seq a S) :
  a 3 = 10 :=
sorry

end a3_eq_10_l709_709854


namespace intersect_on_BC_l709_709972

-- Definitions to be used in the Lean statement
variables {A B C X Y K S T L : Type}
variables (ABC : IsAcuteTriangle A B C)
variables (distinct_XY : X ≠ Y)
variables (X_on_BC : OnSegment X B C)
variables (Y_on_BC : OnSegment Y B C)
variables (angle_CAX_eq_YAB : ∠ C A X = ∠ Y A B)
variables (K_foot_perpendicular : FootPerpendicular B A X K)
variables (S_foot_perpendicular : FootPerpendicular B A Y S)
variables (T_foot_perpendicular : FootPerpendicular C A X T)
variables (L_foot_perpendicular : FootPerpendicular C A Y L)

-- The theorem to be proved
theorem intersect_on_BC 
  (h1 : IsAcuteTriangle A B C)
  (h2 : X ≠ Y)
  (h3 : OnSegment X B C)
  (h4 : OnSegment Y B C)
  (h5 : ∠ C A X = ∠ Y A B)
  (h6 : FootPerpendicular B A X K)
  (h7 : FootPerpendicular B A Y S)
  (h8 : FootPerpendicular C A X T)
  (h9 : FootPerpendicular C A Y L) : 
  IntersectOnLine KL ST B C := 
sorry

end intersect_on_BC_l709_709972


namespace find_a_for_even_function_l709_709145

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, (x-1)^2 + a * x + sin (x + π / 2) = ((-x)-1)^2 + (-a * x) + sin (-x + π / 2)) →
  a = 2 :=
by
  sorry

end find_a_for_even_function_l709_709145


namespace modulus_eq_four_implies_a_l709_709170

theorem modulus_eq_four_implies_a (a : ℝ) :
  complex.abs ((a + 2 * complex.I) * (1 + complex.I)) = 4 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end modulus_eq_four_implies_a_l709_709170


namespace cans_recycling_proof_l709_709654

def aluminum_cans (n : ℕ) (recycle : ℕ → ℕ) : ℕ :=
  let rec helper (m : ℕ) :=
    if m < 2 then 0
    else let new_cans := recycle m
        in new_cans + helper new_cans
  helper n

def recycle (m : ℕ) : ℕ := m / 2

theorem cans_recycling_proof : aluminum_cans 128 recycle = 127 := by
  sorry

end cans_recycling_proof_l709_709654


namespace find_t_l709_709110

def vector (n : ℕ) := fin n → ℚ

noncomputable def vector_m : vector 2 := ![-2, 1]
noncomputable def vector_n (t : ℚ) : vector 2 := ![-1, t]
noncomputable def vector_k : vector 2 := ![1, -2]

noncomputable def sum_vectors (v1 v2 : vector 2) : vector 2 :=
  fun i => v1 i + v2 i

noncomputable def dot_product (v1 v2 : vector 2) : ℚ :=
  ∑ i, v1 i * v2 i

theorem find_t (t : ℚ) (h : dot_product (sum_vectors vector_m (vector_n t)) vector_k = 0) :
  t = -5 / 2 :=
sorry

end find_t_l709_709110


namespace polynomial_value_at_three_l709_709551

theorem polynomial_value_at_three (P : ℝ → ℝ)
  (hP : ∃ (b : ℕ → ℕ), (P x = ∑ i in finset.range (b.length), b i * x^i) ∧ ∀ i, 0 ≤ b i ∧ b i < 5) 
  (hP_sqrt5 : P (Real.sqrt 5) = 23 + 19 * Real.sqrt 5) : 
  P 3 = 132 :=
by sorry

end polynomial_value_at_three_l709_709551


namespace jill_travel_time_to_school_is_20_minutes_l709_709768

variables (dave_rate : ℕ) (dave_step : ℕ) (dave_time : ℕ)
variables (jill_rate : ℕ) (jill_step : ℕ)

def dave_distance : ℕ := dave_rate * dave_step * dave_time
def jill_time_to_school : ℕ := dave_distance dave_rate dave_step dave_time / (jill_rate * jill_step)

theorem jill_travel_time_to_school_is_20_minutes : 
  dave_rate = 85 → dave_step = 80 → dave_time = 18 → 
  jill_rate = 120 → jill_step = 50 → jill_time_to_school 85 80 18 120 50 = 20 :=
by
  intros
  unfold jill_time_to_school
  unfold dave_distance
  sorry

end jill_travel_time_to_school_is_20_minutes_l709_709768


namespace max_sum_sequence_l709_709447

noncomputable def sequence (n : ℕ) : ℤ := -n^2 + 10 * n + 11 

theorem max_sum_sequence :
  ∃ N, (10 ≤ N ∧ N ≤ 11) ∧ 
  (∑ i in Finset.range (N + 1), sequence i) = (∑ i in Finset.range (12), sequence i) :=
by
  sorry

end max_sum_sequence_l709_709447


namespace even_function_iff_a_eq_2_l709_709151

noncomputable def y (a x : ℝ) : ℝ :=
  (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_iff_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, y a x = y a (-x)) ↔ a = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end even_function_iff_a_eq_2_l709_709151


namespace equivalent_functions_l709_709774

theorem equivalent_functions :
  (∀ x: ℝ, x ≤ 0 → (x^2 - x + 2 = x^2 + |x| + 2)) ∧
  (∀ t: ℝ, t ≤ 0 → (t^2 - t + 2 = x^2 - x + 2)) ∧
  (∀ x: ℝ, x ≤ 0 → ((sqrt (-x))^2 + sqrt (x^4) + 2 = x^2 - x + 2)) :=
by
  sorry

end equivalent_functions_l709_709774


namespace prove_intersection_l709_709085

-- Defining the set M
def M : Set ℝ := { x | x^2 - 2 * x < 0 }

-- Defining the set N
def N : Set ℝ := { x | x ≥ 1 }

-- Defining the complement of N in ℝ
def complement_N : Set ℝ := { x | x < 1 }

-- The intersection M ∩ complement_N
def intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The statement to be proven
theorem prove_intersection : M ∩ complement_N = intersection :=
by
  sorry

end prove_intersection_l709_709085


namespace triangle_ABC_BC_length_l709_709201

noncomputable def AB : ℝ := 4 * Real.sqrt 3
noncomputable def angle_A : ℝ := Real.pi / 4  -- 45 degrees in radians
noncomputable def angle_C : ℝ := Real.pi / 3  -- 60 degrees in radians

theorem triangle_ABC_BC_length :
  let sin_A := Real.sin angle_A in
  let sin_C := Real.sin angle_C in
  sin_A = Real.sqrt 2 / 2 →
  sin_C = Real.sqrt 3 / 2 →
  BC = 4 * Real.sqrt 2 :=
  sorry

end triangle_ABC_BC_length_l709_709201


namespace pieces_from_rod_l709_709327

theorem pieces_from_rod (length_of_rod : ℝ) (length_of_piece : ℝ) 
  (h_rod : length_of_rod = 42.5) 
  (h_piece : length_of_piece = 0.85) :
  length_of_rod / length_of_piece = 50 :=
by
  rw [h_rod, h_piece]
  calc
    42.5 / 0.85 = 50 := by norm_num

end pieces_from_rod_l709_709327


namespace hyperbola_condition_l709_709093

theorem hyperbola_condition (k : ℝ) : (3 - k) * (k - 2) < 0 ↔ k < 2 ∨ k > 3 := by
  sorry

end hyperbola_condition_l709_709093


namespace wrench_turns_bolt_l709_709305

theorem wrench_turns_bolt (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (Real.sqrt 3 / Real.sqrt 2 < b / a) ∧ (b / a ≤ 3 - Real.sqrt 3) :=
sorry

end wrench_turns_bolt_l709_709305


namespace hyperbola_equation_is_correct_l709_709462

noncomputable theory

-- Definitions based on the problem's conditions
def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola_eccentricity : ℝ := Real.sqrt 5

def hyperbola_eqn (a b : ℝ) : Prop :=
  ∀ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1) ↔ (5 * x^2 - (5 / 4) * y^2 = 1)

theorem hyperbola_equation_is_correct :
  ∃ (a b : ℝ), 
    parabola_focus = (1, 0) ∧ 
    hyperbola_eccentricity = Real.sqrt 5 ∧ 
    hyperbola_eqn a b :=
sorry

end hyperbola_equation_is_correct_l709_709462


namespace min_value_of_dot_product_l709_709520

variables {V : Type*} [inner_product_space ℝ V]

def is_isosceles_trapezoid (A B C D : V) : Prop :=
  ∃ n : V, B - A = D - C ∧ ∥B - A∥ = ∥D - C∥

theorem min_value_of_dot_product
  (A B C D E F : V) 
  (AB_eq_2 : ∥B - A∥ = 2)
  (BC_eq_1 : ∥C - B∥ = 1)
  (angle_ABC : ∠B A C = real.pi / 3)
  (is_iso_trap : is_isosceles_trapezoid A B C D)
  (BE_eq_lambda_BC : ∃ (λ : ℝ), E - B = λ • (C - B))
  (DF_eq_one_fourth_DC : ∃ (λ : ℝ), F - D = (1 / (4 * λ)) • (C - D)) :

  ∃ (λ : ℝ), 1 / (2 * λ) + λ / 2 + 7/8 ≥ 15/8 :=
sorry

end min_value_of_dot_product_l709_709520


namespace angle_BDC_45_l709_709953

theorem angle_BDC_45 (A B C D : Type) [angle A = 75] [angle C = 60] 
  (AC CD : ℝ) (h1 : CD = 1/2 * AC) : angle BDC = 45 := 
sorry

end angle_BDC_45_l709_709953


namespace angle_between_b_and_c_is_90_degrees_l709_709554

noncomputable def unit_vector (v : V) : Prop :=
∥v∥ = 1

/--
Given:
1. a, b, c are unit vectors.
2. a × (b × c) = (c - b) / 2
3. ⟨a, b, c⟩ form an orthogonal set.
Prove: The angle between b and c is 90 degrees.
-/
theorem angle_between_b_and_c_is_90_degrees
  {V : Type*} [inner_product_space ℝ V]
  (a b c : V)
  (ha : unit_vector a)
  (hb : unit_vector b)
  (hc : unit_vector c)
  (orth_abc : ⟪a, b⟫ = 0 ∧ ⟪a, c⟫ = 0 ∧ ⟪b, c⟫ = 0)
  (eqn : a × (b × c) = (c - b) / 2) :
  angle b c = real.pi / 2 :=
sorry

end angle_between_b_and_c_is_90_degrees_l709_709554


namespace gcd_of_lengths_in_inches_l709_709044

theorem gcd_of_lengths_in_inches :
  let lengths_cm := [700, 385, 1295, 1545, 2663]
  let lengths_in := lengths_cm.map (λ cm, (cm : ℚ) / 2.54)
  let lengths_in_rounded := lengths_in.map (λ x, x.toInt)
  let gcd := Nat.gcd (lengths_in_rounded.head!) (lengths_in_rounded.tail.head!)
  List.foldr Nat.gcd gcd lengths_in_rounded.tail.tail == 2 :=
by
  sorry

end gcd_of_lengths_in_inches_l709_709044


namespace number_of_lines_with_negative_reciprocal_intercepts_l709_709057

-- Define the point (-2, 4)
def point : ℝ × ℝ := (-2, 4)

-- Define the condition that intercepts are negative reciprocals
def are_negative_reciprocals (a b : ℝ) : Prop :=
  a * b = -1

-- Define the proof problem: 
-- Number of lines through point (-2, 4) with intercepts negative reciprocals of each other
theorem number_of_lines_with_negative_reciprocal_intercepts :
  ∃ n : ℕ, n = 2 ∧ 
  ∀ (a b : ℝ), are_negative_reciprocals a b →
  (∃ m k : ℝ, (k * (-2) + m = 4) ∧ ((m ⁻¹ = a ∧ k = b) ∨ (k = a ∧ m ⁻¹ = b))) :=
sorry

end number_of_lines_with_negative_reciprocal_intercepts_l709_709057


namespace find_x_value_l709_709806

def cube_root (a : ℝ) : ℝ := a^(1/3)

theorem find_x_value :
  let y := cube_root 0.000343,
      x := 7 * y
  in x = 0.49 := by
  sorry

end find_x_value_l709_709806


namespace coefficient_properties_l709_709498

theorem coefficient_properties (f : ℕ → ℕ) (a : ℕ → ℤ) :
  (∀ x, (1 - x) ^ 2022 = ∑ i in range (2023), a i * x ^ i) →
  (a 0 = 1) ∧
  (∑ i in range (1, 2023), a i = -1) ∧
  (∑ i in range (2023), a i = 0) ∧
  (∑ i in range (2023), binom 2022 i = 2 ^ 2022) :=
by
  intro h
  sorry

end coefficient_properties_l709_709498


namespace remainder_415_pow_420_div_16_l709_709803

theorem remainder_415_pow_420_div_16 : 415^420 % 16 = 1 := by
  sorry

end remainder_415_pow_420_div_16_l709_709803


namespace find_period_and_monotonic_intervals_l709_709058

open Real

def period_and_monotonic_intervals (k : ℤ) : Prop :=
  let y := λ x : ℝ, 3 * tan (π / 6 - x / 4)
  let period := 4 * π
  let interval := (4 * k * π - 4 * π / 3, 4 * k * π + 8 * π / 3)
  let is_decreasing := ∀ x1 x2 : ℝ, x1 ∈ Ioo (interval.1) (interval.2) → x2 ∈ Ioo (interval.1) (interval.2) → x1 < x2 → y x1 > y x2
  (∀ x : ℝ, y (x + period) = y x) ∧ is_decreasing

-- Main theorem statement
theorem find_period_and_monotonic_intervals (k : ℤ) : 
  period_and_monotonic_intervals k :=
sorry

end find_period_and_monotonic_intervals_l709_709058


namespace constant_term_of_binomial_expansion_l709_709172

theorem constant_term_of_binomial_expansion :
  ∃ n : ℕ, ((∀ x : ℝ, (3 * real.sqrt x - 1 / real.sqrt x)^n).sum.coefficients = 64 ∧ (
    (n = 6) ∧
    (binomial.expansion.constant_term ((3 * real.sqrt x - 1 / real.sqrt x)^6) = -540))
  := by
  sorry

end constant_term_of_binomial_expansion_l709_709172


namespace remainder_of_9_6_plus_8_7_plus_7_8_mod_7_l709_709665

theorem remainder_of_9_6_plus_8_7_plus_7_8_mod_7 : (9^6 + 8^7 + 7^8) % 7 = 2 := 
by sorry

end remainder_of_9_6_plus_8_7_plus_7_8_mod_7_l709_709665


namespace black_shirts_in_pack_l709_709232

-- defining the conditions
variables (B : ℕ) -- the number of black shirts in each pack
variable (total_shirts : ℕ := 21)
variable (yellow_shirts_per_pack : ℕ := 2)
variable (black_packs : ℕ := 3)
variable (yellow_packs : ℕ := 3)

-- ensuring the conditions are met, the total shirts equals 21
def total_black_shirts := black_packs * B
def total_yellow_shirts := yellow_packs * yellow_shirts_per_pack

-- the proof problem
theorem black_shirts_in_pack : total_black_shirts + total_yellow_shirts = total_shirts → B = 5 := by
  sorry

end black_shirts_in_pack_l709_709232


namespace divide_square_into_quadrilaterals_l709_709079

theorem divide_square_into_quadrilaterals :
  ∃ quadrilaterals : list (set (ℝ × ℝ)),
    (∀ q ∈ quadrilaterals, inscribed q (circle_radius (sqrt 3 / 2))) ∧
    (sum_of_areas quadrilaterals = area_of_square 10) ∧
    (length quadrilaterals = 100) :=
by
  sorry

def inscribed (shape : set (ℝ × ℝ)) (circle : set (ℝ × ℝ)) : Prop :=
  ∀ (p ∈ shape), p ∈ circle

def circle_radius (r : ℝ) : set (ℝ × ℝ) :=
  {p | p.1 ^ 2 + p.2 ^ 2 = r ^ 2}

def area_of_square (side : ℝ) : ℝ :=
  side ^ 2

def sum_of_areas (shapes : list (set (ℝ × ℝ))) : ℝ :=
  shapes.foldr (λ s acc, acc + area_of_set s) 0

noncomputable def area_of_set (shape : set (ℝ × ℝ)) : ℝ :=
  sorry

end divide_square_into_quadrilaterals_l709_709079


namespace doug_initial_marbles_l709_709048

theorem doug_initial_marbles 
  (ed_marbles : ℕ)
  (doug_marbles : ℕ)
  (lost_marbles : ℕ)
  (ed_condition : ed_marbles = doug_marbles + 5)
  (lost_condition : lost_marbles = 3)
  (ed_value : ed_marbles = 27) :
  doug_marbles + lost_marbles = 25 :=
by
  sorry

end doug_initial_marbles_l709_709048


namespace sphere_volume_ratio_l709_709468

theorem sphere_volume_ratio (S₁ S₂ : ℝ) (r₁ r₂ : ℝ)
  (hS : S₁ / S₂ = 1 / 4)
  (hS₁ : S₁ = 4 * real.pi * r₁^2)
  (hS₂ : S₂ = 4 * real.pi * r₂^2) :
  (4 / 3 * real.pi * r₁^3) / (4 / 3 * real.pi * r₂^3) = 1 / 8 :=
by {
  sorry
}

end sphere_volume_ratio_l709_709468


namespace solution_set_x_fx_l709_709845

-- Define the necessary conditions for the function f
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)
def monodecreasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x1 x2, x1 < x2 -> x2 < 0 -> f(x1) ≥ f(x2)
def value_at_neg_two (f : ℝ → ℝ) : Prop := f(-2) = 0

-- Define the main theorem to prove the solution set of x * f(x) < 0
theorem solution_set_x_fx (f : ℝ → ℝ) 
  (hf_even : even_function f)
  (hf_mono : monodecreasing_on_neg f)
  (hf_neg_two : value_at_neg_two f) :
  { x : ℝ | x * f(x) < 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | 0 < x ∧ x < 2 } :=
by sorry

end solution_set_x_fx_l709_709845


namespace bags_total_on_next_day_l709_709648

def bags_on_monday : ℕ := 7
def additional_bags : ℕ := 5
def bags_on_next_day : ℕ := bags_on_monday + additional_bags

theorem bags_total_on_next_day : bags_on_next_day = 12 := by
  unfold bags_on_next_day
  unfold bags_on_monday
  unfold additional_bags
  sorry

end bags_total_on_next_day_l709_709648


namespace find_complex_number_l709_709075

noncomputable def z (a b : ℝ) : ℂ := a + b * complex.I

theorem find_complex_number (a b : ℝ) (ha : a ≤ 0) (h1 : 4 * b = -2 * a) (h2 : (a + 1)^2 + b^2 = 2) :
  z a b = -2 + complex.I :=
by
  sorry

end find_complex_number_l709_709075


namespace quadrilateral_rhombus_l709_709919

variables {V : Type*} [inner_product_space ℝ V]
variables {A B C D : V}

-- Definitions converted into lean constraints
def is_parallelogram (AB CD : V) : Prop := AB + CD = 0
def perpendicular (P Q : V) : Prop := inner_product_space.inner P Q = 0

-- Given the conditions, we need to prove the quadrilateral is a rhombus
theorem quadrilateral_rhombus (h1 : is_parallelogram (A - B) (C - D))
  (h2 : perpendicular ((A - B) + (D - A)) (A - C)) :
  -- Conclusion: ABCD is a rhombus, which implies the four sides are equal, i.e., the norm of vectors AB, BC, CD, and DA are equal
  ∃ r : ℝ, ∀ {U V W X : V}, (A = U) → (B = V) → (C = W) → (D = X) →
  (∥U - V∥ = r ∧ ∥V - W∥ = r ∧ ∥W - X∥ = r ∧ ∥X - U∥ = r)
:= 
sorry  -- proof omitted

end quadrilateral_rhombus_l709_709919


namespace find_roots_polynomial_equation_l709_709411

noncomputable def root1 : ℂ := 
  ((5 + complex.sqrt 109) / 6 + complex.sqrt ((5 + complex.sqrt 109) / 6)^2 - 4) / 2

noncomputable def root2 : ℂ := 
  ((5 + complex.sqrt 109) / 6 - complex.sqrt ((5 + complex.sqrt 109) / 6)^2 - 4) / 2

noncomputable def root3 : ℂ := 
  ((5 - complex.sqrt 109) / 6 + complex.sqrt ((5 - complex.sqrt 109) / 6)^2 - 4) / 2

noncomputable def root4 : ℂ := 
  ((5 - complex.sqrt 109) / 6 - complex.sqrt ((5 - complex.sqrt 109) / 6)^2 - 4) / 2

theorem find_roots_polynomial_equation : 
  (∃ x : ℂ, (3 * x^4 - 5 * x^3 - x^2 + 5 * x + 3 = 0) → 
    (x = root1 ∨ x = root2 ∨ x = root3 ∨ x = root4)) :=
sorry

end find_roots_polynomial_equation_l709_709411


namespace sasha_worked_2_hours_l709_709244

def sasha_hours_worked (questions_per_hour total_questions remaining_questions : ℕ) :=
  (total_questions - remaining_questions) / questions_per_hour

theorem sasha_worked_2_hours (questions_per_hour total_questions remaining_questions : ℕ) :
  questions_per_hour = 15 → total_questions = 60 → remaining_questions = 30 → sasha_hours_worked questions_per_hour total_questions remaining_questions = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end sasha_worked_2_hours_l709_709244


namespace sum_double_factorial_expr_cd_over_10_calc_l709_709771

-- Define double factorials for odd and even cases
def double_factorial (n : ℕ) : ℕ :=
  if n = 0 ∨ n = 1 then 1 else n * double_factorial (n - 2)

-- Prove the sum of the given expression equals 1005
theorem sum_double_factorial_expr :
  let expr := λ (j : ℕ), (Nat.factorial (2 * j)) / (double_factorial (2 * j - 1) * double_factorial (2 * j))
  ∑ j in Finset.range (1005 + 1), expr j = 1005 :=
by
  sorry

-- Prove the expression of 1005 as 2^c * d and the value of cd / 10
theorem cd_over_10_calc :
  (∃ c d : ℕ, (1005 = 2 ^ c * d) ∧ (d % 2 = 1) ∧ ((c * d) / 10 = 0)) :=
by
  use 0
  use 1005
  split
  · -- Prove 1005 = 2^0 * 1005
    exact (Nat.mul_one 1005).symm
  split
  · -- Prove d = 1005 is odd
    exact Nat.odd_iff_not_even.mpr (by simp [Nat.even_iff_two_dvd])
  · -- Prove (c * d) / 10 = 0
    simp
    sorry

end sum_double_factorial_expr_cd_over_10_calc_l709_709771


namespace remainder_when_dividing_l709_709574

theorem remainder_when_dividing (c d : ℕ) (p q : ℕ) :
  c = 60 * p + 47 ∧ d = 45 * q + 14 → (c + d) % 15 = 1 :=
by
  sorry

end remainder_when_dividing_l709_709574


namespace equal_tangents_angle_l709_709208

variables (A B C D P X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
          [MetricSpace D] [MetricSpace P] [MetricSpace X]

-- Axioms and specifications
axiom incircle (k : Type) : ∃ (I : Type) [InCircle I],
    QuadrilateralInCircle ABCD I
axiom lines_meet (AD BC : Type) : ∃ P : Type, meet AD BC P
axiom circumcircle_intersect (PAB PCD : Type) : ∃ X : Type, intersect PAB PCD X

theorem equal_tangents_angle (k : Type) (AD BC PAB PCD : Type)
  (A B C D P X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace P] [MetricSpace X] :
  incircle k →
  lines_meet AD BC →
  circumcircle_intersect PAB PCD →
  ∃ (tangent_l tangent_m : Type),
    Tangent X k tangent_l tangent_m ∧
    equal_angles_with_lines tangent_l tangent_m AX CX :=
sorry

end equal_tangents_angle_l709_709208


namespace shifted_parabola_eq_l709_709631

def original_parabola (x : ℝ) : ℝ := -2 * x^2

def shift_horizontal (x : ℝ) (h : ℝ) : ℝ := x - h

def shift_vertical (y : ℝ) (k : ℝ) : ℝ := y + k

theorem shifted_parabola_eq :
  ∀ (x : ℝ), 
    let h := 1 in 
    let k := 3 in 
    shift_vertical (original_parabola (shift_horizontal x h)) k 
    = -2 * (x - 1)^2 + 3 := 
by 
  intro x
  let h := 1
  let k := 3
  simp [original_parabola, shift_horizontal, shift_vertical]
  sorry

end shifted_parabola_eq_l709_709631


namespace red_car_rental_cost_l709_709575

/-- 
Lori owns a car-sharing company. There are three red cars and two white cars available to rent. 
Renting the white car costs $2 for every minute; renting the red car has an unknown cost per minute. 
All cars were rented for 3 hours in total, and Lori earned $2340. 
Prove that the cost to rent a red car is $3 per minute.
-/
theorem red_car_rental_cost :
  ∃ (R : ℝ), 
  let red_car_count := 3,
      white_car_count := 2,
      white_car_rate := 2,
      rent_duration := 3 * 60, -- in minutes
      total_earnings := 2340,
      white_car_earnings := white_car_count * white_car_rate * rent_duration,
      red_car_earnings := total_earnings - white_car_earnings,
      red_car_total_time := red_car_count * rent_duration,
      R := red_car_earnings / red_car_total_time 
  in
  R = 3 :=
begin
  sorry
end

end red_car_rental_cost_l709_709575


namespace range_of_f_intervals_of_increase_l709_709273

noncomputable def f (x ω : ℝ) := 4 * cos (ω * x - π / 6) * sin (ω * x) - 2 * cos (2 * ω * x + π)

theorem range_of_f (ω : ℝ) (hω : ω > 0) : 
  (set.range (f · ω) = set.Icc (-1 : ℝ) 3) :=
sorry

theorem intervals_of_increase (ω : ℝ) (hω : ω = 1) : 
  (set.Ioo (-(π/2)) π ⊂ 
  (set.Ioo (-(π/3)) (π/6)) ∪ 
  (set.Ioo ((2 * π) / 3) π)) :=
sorry

end range_of_f_intervals_of_increase_l709_709273


namespace p_functions_identification_l709_709504

noncomputable def is_p_function (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, 1 < x ∧ x < y → (f x / Real.log x) > (f y / Real.log y)

def f1 : ℝ → ℝ := λ x, 1
def f2 : ℝ → ℝ := λ x, x
def f3 : ℝ → ℝ := λ x, 1 / x
def f4 : ℝ → ℝ := λ x, Real.sqrt x

theorem p_functions_identification :
  is_p_function f1 ∧ is_p_function f3 ∧ ¬ is_p_function f2 ∧ ¬ is_p_function f4 :=
by
  sorry

end p_functions_identification_l709_709504


namespace even_function_iff_a_eq_2_l709_709155

noncomputable def y (a x : ℝ) : ℝ :=
  (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_iff_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, y a x = y a (-x)) ↔ a = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end even_function_iff_a_eq_2_l709_709155


namespace total_donuts_three_days_l709_709035

def donuts_on_Monday := 14

def donuts_on_Tuesday := donuts_on_Monday / 2

def donuts_on_Wednesday := 4 * donuts_on_Monday

def total_donuts := donuts_on_Monday + donuts_on_Tuesday + donuts_on_Wednesday

theorem total_donuts_three_days : total_donuts = 77 :=
  by
    sorry

end total_donuts_three_days_l709_709035


namespace mode_is_10_l709_709021

def mode_of_data_set : list ℕ := [9, 7, 10, 8, 10, 9, 10]

theorem mode_is_10 (l : list ℕ) (h : l = mode_of_data_set) : 
  l.count 10 > l.count 9 ∧ l.count 10 > l.count 8 ∧ l.count 10 > l.count 7 := by
  sorry

end mode_is_10_l709_709021


namespace magnitude_product_l709_709399

def c1 := 3 - 4 * complex.I
def c2 := 2 + 6 * complex.I
def r := 5

theorem magnitude_product (h1 : complex.abs c1 = 5)
                          (h2 : complex.abs c2 = 2 * real.sqrt 10)
                          (h3 : real.abs r = 5) :
    complex.abs (c1 * c2 * r) = 50 * real.sqrt 10 :=
by sorry

end magnitude_product_l709_709399


namespace solve_problem_l709_709859

noncomputable def f (x m : ℝ) : ℝ :=
  x^3 - 3*x + m

-- Conditions:
-- 1. The function is f(x) = x^3 - 3x + m
-- 2. The interval is [-3, 0]
-- 3. The sum of the maximum and minimum values of f(x) in [-3,0] is -14

def problem_statement : Prop :=
  ∃ m : ℝ, 
  (let f_max := f (-1) m in
   let f_min := f (-3) m in
   f_max + f_min = -14) → m = 1

theorem solve_problem : problem_statement := 
  sorry

end solve_problem_l709_709859


namespace coprime_bijective_function_exists_l709_709418

variables (a b n : ℕ)

def conditions := (nat.coprime a b) ∧ (n % 2 = 1) ∧ (a % 2 = 1)

/-- Necessary and sufficient conditions for the existence of a bijective function f: S → S
such that for any x ∈ S, x and f(x) are coprime, where S is defined by:
    S = {a + b * t | t = 0, 1, ..., n - 1} -/
theorem coprime_bijective_function_exists
  (a_pos : a > 0) (b_pos : b > 0) (n_ge_two : n ≥ 2) :
  ∃ (f : set (a + b * t for t in finset.range n) → set (a + b * t for t in finset.range n)),
    (∀ x ∈ (a + b * t for t in finset.range n), nat.coprime x (f x)) ↔ conditions a b n :=
sorry

end coprime_bijective_function_exists_l709_709418


namespace determine_a_l709_709161

def y (x : ℝ) (a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem determine_a (a : ℝ) : is_even (y _ a) → a = 2 :=
sorry

end determine_a_l709_709161


namespace sequence_general_term_l709_709446

theorem sequence_general_term (a : ℕ → ℝ) (h : ∀ n : ℕ, n > 0 →
  n / (∑ i in range (n + 1), (i + 1) * a (i + 1)) = 2 / (n + 2)) :
  ∀ n : ℕ, n > 0 → a n = (3 * n + 1) / (2 * n) :=
by
  sorry

end sequence_general_term_l709_709446


namespace complex_expression_l709_709229

noncomputable def z : ℂ := (1 + Complex.i) / Real.sqrt 2

theorem complex_expression:
  ((z^1 + z^4 + z^9) * (1/z^1 + 1/z^4 + 1/z^9) = 2) :=
sorry

end complex_expression_l709_709229


namespace ways_to_insert_signs_divisible_by_5_l709_709801

noncomputable def numberOfWays : ℕ :=
  816

theorem ways_to_insert_signs_divisible_by_5 :
  ∀ (f : ℤ → ℤ), (∃ signs : list (ℤ → ℤ), 
  (∀ (k ∈ signs), k = (+) ∨ k = (-)) ∧
  f = λ n, list.sum (list.map (λ (k : ℤ → ℤ) => k n) signs)) →
  (exists n : ℕ, n % 5 = 0) → numberOfWays =
  816 := by
  sorry

end ways_to_insert_signs_divisible_by_5_l709_709801


namespace problem1_values_problem2_lambda_Sn_problem3_arithmetic_seq_l709_709466

noncomputable def a₁ (n : ℕ) : ℤ := 2^n - 3 * n
noncomputable def a₂ (n : ℕ) : ℤ := 2^n - λ * n

noncomputable def M_n (a : ℕ → ℤ) (n : ℕ) : ℤ := finset.sup (finset.range n) a
noncomputable def m_n (a : ℕ → ℤ) (n : ℕ) : ℤ := finset.inf (finset.range n) a
noncomputable def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ := (M_n a n + m_n a n) / 2

theorem problem1_values :
  b_n a₁ 1 = -1 ∧ b_n a₁ 2 = -3 / 2 ∧ b_n a₁ 3 = -3 / 2 ∧ b_n a₁ 4 = 1 := sorry

theorem problem2_lambda_Sn (λ : ℤ) (n : ℕ) :
  (b_n a₂ 3 = -3 → λ = 4) ∧
  (λ = 4 → (∀(n ≥ 4), (finset.sum (finset.range n) (λ i, b_n a₂ i)) = 2^n - n^2 - 3 * n + 2)) := sorry

theorem problem3_arithmetic_seq (a : ℕ → ℤ) :
  (∀ n, b_n a (n + 1) - b_n a n = d) ↔ (∀ n, a (n + 1) - a n = d * 2) := sorry

end problem1_values_problem2_lambda_Sn_problem3_arithmetic_seq_l709_709466


namespace GP_perpendicular_to_BC_l709_709006

variables {P A B C D E F G : Point} -- define the points as variables

-- Given conditions
def square_BCEF (B C E F : Point) : Prop := square B C E F

def right_isosceles_ABP (A B P : Point) : Prop := right_isosceles_triangle A B P ∧ (right_angle A B P)

def right_isosceles_PCD (P C D : Point) : Prop := right_isosceles_triangle P C D ∧ (right_angle P C D)

def line_intersection_AF_DE (A F D E G : Point) : Prop := intersects (line_through A F) (line_through D E) G

-- Problem statement
theorem GP_perpendicular_to_BC 
  (h1 : square_BCEF B C E F) 
  (h2 : right_isosceles_ABP A B P) 
  (h3 : right_isosceles_PCD P C D) 
  (h4 : line_intersection_AF_DE A F D E G) : 
  perpendicular (line_through G P) (line_through B C) := 
sorry

end GP_perpendicular_to_BC_l709_709006


namespace indistinguishable_balls_into_boxes_l709_709887

theorem indistinguishable_balls_into_boxes : 
  ∃ n : ℕ, n = 3 ∧ (∀ (b : ℕ), (b = 5 → 
  ∃ (ways : ℕ), ways = 3 ∧ 
  (ways = 1 + 1 + 1 ∧ 
  ((∃ x y : ℕ, x + y = b ∧ (x = 5 ∧ y = 0)) ∧ 
   (∃ x y : ℕ, x + y = b ∧ (x = 4 ∧ y = 1)) ∧ 
   (∃ x y : ℕ, x + y = b ∧ (x = 3 ∧ y = 2)))))) := 
begin
  sorry
end

end indistinguishable_balls_into_boxes_l709_709887


namespace smallest_five_digit_multiple_of_18_l709_709422

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, n = 10008 ∧ (n ≥ 10000 ∧ n < 100000) ∧ n % 18 = 0 ∧ (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 18 = 0 → n ≤ m) := sorry

end smallest_five_digit_multiple_of_18_l709_709422


namespace cos_neg_seventy_nine_sixth_pi_l709_709045

theorem cos_neg_seventy_nine_sixth_pi : cos (-79 / 6 * real.pi) = -real.sqrt 3 / 2 :=
by
  sorry

end cos_neg_seventy_nine_sixth_pi_l709_709045


namespace find_C_grazing_duration_l709_709028

-- Define the known constants and variables
def units_grazing_A : ℕ := 10 * 7
def units_grazing_B : ℕ := 12 * 5
def total_rent : ℝ := 280
def C_rent_share : ℝ := 72

-- Define the units of grazing for C
def units_grazing_C (x : ℕ) : ℕ := 15 * x

-- Define the total units of grazing based on the units of grazing for A, B, and C
def total_units_grazing (x : ℕ) : ℕ := units_grazing_A + units_grazing_B + units_grazing_C x

-- Define the equation to be solved
def equation_for_C (x : ℕ) : Prop :=
    (C_rent_share = (units_grazing_C x : ℝ) / (total_units_grazing x : ℝ) * total_rent)

-- Statement to be proven
theorem find_C_grazing_duration : ∃ x : ℕ, equation_for_C x ∧ x = 3 :=
by
  -- This establishes the proof problem
  existsi 3
  split
  · unfold equation_for_C units_grazing_C total_units_grazing units_grazing_A units_grazing_B
    simp [C_rent_share, total_rent]
    field_simp
    sorry
  · rfl

end find_C_grazing_duration_l709_709028


namespace count_ordered_sets_eq_catalan_l709_709419

open BigOperators

/-- The number of ordered sets (a_1, a_2, ..., a_n) of n natural numbers such that 
    1 ≤ a_1 ≤ a_2 ≤ ... ≤ a_n and a_i ≤ i for all i = 1, 2, ..., n 
    is given by the Catalan number A_n = (2n choose n) / (n + 1) --/
theorem count_ordered_sets_eq_catalan (n : ℕ) :
  (∑ a : ℕ, (a ≤ n).toInt * (a ≤ n).toInt) =
    Nat.factorial (2 * n) / (Nat.factorial n * Nat.factorial (n + 1)) :=
  sorry

end count_ordered_sets_eq_catalan_l709_709419


namespace alice_finite_iterations_l709_709246

theorem alice_finite_iterations {n : ℕ} (a : Fin n → ℕ) :
  (∃ (steps : ℕ), ∀ (step : ℕ), step > steps → ∀ (i : Fin (n - 1)),
    ¬(a i > a (i + 1) ∧ (a i, a (i + 1)) = (a (i + 1) + 1, a i) ∨
    (a i, a (i + 1)) = (a i - 1, a i))) :=
sorry

end alice_finite_iterations_l709_709246


namespace not_all_roots_real_l709_709249

def polynomial_q (a_85 a_84 : ℝ) (a_n : ℕ → ℝ) :=
  ∑ i in (finset.range 86).filter (λ x, x ≥ 3 ∧ x ≤ 85),
    a_n i * x^i + 3 * x^2 + 2 * x + 1

theorem not_all_roots_real (a₃ a₄ ... a₈₅ : fin 83 → ℝ) : 
∃ x : ℂ, polynomial (a₃ 3) (a₄ 4) ... (a₈₅ 85) x = 0 ∧ x ∉ ℝ :=
begin
  sorry
end

end not_all_roots_real_l709_709249


namespace half_angle_in_first_quadrant_l709_709855

theorem half_angle_in_first_quadrant {α : ℝ} (h : 0 < α ∧ α < π / 2) : 
  0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end half_angle_in_first_quadrant_l709_709855


namespace apollonian_circle_distance_l709_709370

theorem apollonian_circle_distance :
  ∃ C : ℝ × ℝ, (dist C (-1, 0) / dist C (1, 0) = sqrt 3) →
  ∀ x y : ℝ, ∃ M : ℝ × ℝ, (M = (2, 0)) →
  (dist (2, 0) (x, y) / sqrt (1^2 + (-2)^2)) - sqrt 3 = 2 * sqrt 5 - sqrt 3 :=
begin
  sorry -- the actual proof should go here
end

end apollonian_circle_distance_l709_709370


namespace find_n_l709_709797

theorem find_n (n x y k : ℕ) (h_coprime : Nat.gcd x y = 1) (h_eq : 3^n = x^k + y^k) : n = 2 :=
sorry

end find_n_l709_709797


namespace simplify_fraction_l709_709250

theorem simplify_fraction :
  (1 : ℚ) / ((1 / (1 / 3 : ℚ) ^ 1) + (1 / (1 / 3 : ℚ) ^ 2) + (1 / (1 / 3 : ℚ) ^ 3) + (1 / (1 / 3 : ℚ) ^ 4)) = 1 / 120 := 
by 
  sorry

end simplify_fraction_l709_709250


namespace square_area_l709_709754

theorem square_area :
  ∀ (x : ℝ), (4 * x - 15 = 20 - 3 * x) → (let edge := 4 * x - 15 in edge ^ 2 = 25) :=
by
  intros x h
  have h1 : 4 * x - 15 = 20 - 3 * x := h
  sorry

end square_area_l709_709754


namespace area_of_triangle_l709_709174

def AB : ℝ := 3
def AC : ℝ := 4
def BC : ℝ := Real.sqrt 13
def area (AB AC BC : ℝ) : ℝ := 1 / 2 * AB * AC * (Real.sqrt (1 - (AB^2 + AC^2 - BC^2)^2 / (4 * AB^2 * AC^2)))

theorem area_of_triangle : area AB AC BC = 3 * Real.sqrt 3 := by
  sorry

end area_of_triangle_l709_709174


namespace sqrt_equality_l709_709912

theorem sqrt_equality (n : ℤ) (h : Real.sqrt (8 + n) = 9) : n = 73 :=
by
  sorry

end sqrt_equality_l709_709912


namespace num_valid_assignments_l709_709640

def Task : Type := {A, B, C, D}
def Worker : Type := {P, Q, R, S}

-- Conditions
def P_cannot_do_B (assignment : Worker → Task) : Prop :=
  assignment P ≠ B

def Q_cannot_do_B_or_C (assignment : Worker → Task) : Prop :=
  assignment Q ≠ B ∧ assignment Q ≠ C

def R_cannot_do_C_or_D (assignment : Worker → Task) : Prop :=
  assignment R ≠ C ∧ assignment R ≠ D

-- Prove the number of valid assignments
theorem num_valid_assignments :
  {assignment : Worker → Task // P_cannot_do_B assignment ∧ Q_cannot_do_B_or_C assignment ∧ R_cannot_do_C_or_D assignment}.to_finset.card = 4 :=
sorry

end num_valid_assignments_l709_709640


namespace arithmetic_geometric_progression_correct_l709_709766

theorem arithmetic_geometric_progression_correct :
  ∃ (a d b q : ℝ),
  -- Arithmetic progression terms
  let a1 := a,
      a2 := a + d,
      a3 := a + 2 * d,
      a4 := a + 3 * d,
  -- Geometric progression terms
      b1 := b,
      b2 := b * q,
      b3 := b * q^2,
      b4 := b * q^3,
  -- We need to show these terms meet the sum conditions:
  (a1 + b1 = 27 ∧
  a2 + b2 = 27 ∧
  a3 + b3 = 39 ∧
  a4 + b4 = 87) ∧

  -- Correct values from the solution:
  a = 24 ∧ d = -6 ∧ b = 3 ∧ q = 3
:=
begin
  -- Solution parameters
  use [24, -6, 3, 3],
  -- Arithmetic progression terms
  let a1 := 24,
      a2 := 24 - 6,
      a3 := 24 - 12,
      a4 := 24 - 18,
  -- Geometric progression terms
      b1 := 3,
      b2 := 3 * 3,
      b3 := 3 * 9,
      b4 := 3 * 27,

  -- Sum conditions
  simp [a1, a2, a3, a4, b1, b2, b3, b4],
  split; linarith,
  split; linarith,
  split; linarith,
  split; linarith,
end

end arithmetic_geometric_progression_correct_l709_709766


namespace graph_symmetry_l709_709844

theorem graph_symmetry 
  (f : ℝ → ℝ)
  (h1 : ∃ w > 0, ∀ x, f(x) = sin (w * x + π / 4))
  (h2 : ∃ p > 0, p = π ∧ ∀ x, f(x + p) = f(x)) :
  (∃ a, a = π / 8 ∧ ∀ x, f(a - x) = f(a + x)) :=
sorry

end graph_symmetry_l709_709844


namespace number_of_odd_digits_in_product_l709_709856

theorem number_of_odd_digits_in_product (Q : ℕ) : 
  (∃ R : ℕ, R = Q ∧ R = number_of_odd_digits (1111...11 * 9999...99)) ↔ Q = 12 :=
by
  sorry

end number_of_odd_digits_in_product_l709_709856


namespace probability_union_events_l709_709716

def fair_die_probability := 1 / 6
def event_A : set ℕ := {1, 3, 5}
def event_B : set ℕ := {1, 2, 3}

noncomputable def P (s : set ℕ) : ℚ :=
  s.to_finset.card / 6

theorem probability_union_events :
  P (event_A ∪ event_B) = 2 / 3 := by 
    sorry

end probability_union_events_l709_709716


namespace arithmetic_sequence_and_formula_l709_709994

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709994


namespace lattice_midpoint_l709_709517

theorem lattice_midpoint (points : Fin 5 → ℤ × ℤ) :
  ∃ (i j : Fin 5), i ≠ j ∧ 
  let (x1, y1) := points i 
  let (x2, y2) := points j
  (x1 + x2) % 2 = 0 ∧ (y1 + y2) % 2 = 0 := 
sorry

end lattice_midpoint_l709_709517


namespace gnollish_valid_sentences_count_l709_709612

/--
The Gnollish language consists of 4 words: "splargh," "glumph," "amr," and "bork."
A sentence is valid if "splargh" does not come directly before "glumph" or "bork."
Prove that there are 240 valid 4-word sentences in Gnollish.
-/
theorem gnollish_valid_sentences_count : 
  let words := ["splargh", "glumph", "amr", "bork"]
  let total_sentences := (words.length ^ 4)
  let invalid_conditions (w1 w2 : String) := 
    (w1 = "splargh" ∧ (w2 = "glumph" ∨ w2 = "bork"))
  let invalid_count : ℕ := 
    2 * words.length * words.length * (words.length - 1)
  let valid_sentences := total_sentences - invalid_count
  valid_sentences = 240 :=
by
  let words := ["splargh", "glumph", "amr", "bork"]
  let total_sentences := (words.length ^ 4)
  let invalid_conditions (w1 w2 : String) := 
    (w1 = "splargh" ∧ (w2 = "glumph" ∨ w2 = "bork"))
  let invalid_count : ℕ := 
    2 * words.length * words.length * (words.length - 1)
  let valid_sentences := total_sentences - invalid_count
  have : valid_sentences = 240 := by sorry
  exact this

end gnollish_valid_sentences_count_l709_709612


namespace stratified_sampling_l709_709499

-- Define the conditions
def total_students : ℕ := 16050
def vocational_students : ℕ := 4500
def undergraduate_students : ℕ := 9750
def graduate_students : ℕ := 1800
def vocational_students_sampled : ℕ := 60
def sampling_ratio := (vocational_students_sampled : ℚ) / vocational_students

-- State the theorem we need to prove
theorem stratified_sampling :
  (undergraduate_students_sampled = (sampling_ratio * undergraduate_students : ℚ).natCeil) ∧ 
  (graduate_students_sampled = (sampling_ratio * graduate_students : ℚ).natCeil) :=
  sorry

-- Definitions for the numbers of students to be sampled
def undergraduate_students_sampled : ℕ := 130
def graduate_students_sampled : ℕ := 24

end stratified_sampling_l709_709499


namespace percent_increase_over_six_months_l709_709331

variables (P : ℝ)

def new_profit (initial_profit change : ℝ) : ℝ := 
  initial_profit * (1 + change / 100)

theorem percent_increase_over_six_months 
  (initial_profit : ℝ)
  (P_April := new_profit initial_profit 40)
  (P_May := new_profit P_April (-20))
  (P_June := new_profit P_May 50)
  (P_July := new_profit P_June (-30))
  (P_Aug := new_profit P_July 25) :
  (P_Aug - initial_profit) / initial_profit * 100 = 47 := 
  sorry

end percent_increase_over_six_months_l709_709331


namespace problem_part_1_problem_part_2_l709_709561

theorem problem_part_1 (n : ℕ) (h₁ : n = 6) :
  A (6) = 6 :=
sorry

theorem problem_part_2 (n : ℕ) (h₁ : n ≥ 4) :
  A (n) = if n % 2 = 1 then ((n - 1) / 2) ^ 2 else (n^2 - 2 * n) / 4 :=
sorry

end problem_part_1_problem_part_2_l709_709561


namespace distinct_real_roots_equal_integer_roots_example_l709_709470

-- Part 1: Proving the equation always has two distinct real roots when n = m + 3
theorem distinct_real_roots (m n : ℝ) (h : n = m + 3) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_root (polynomial.C 1 * polynomial.X^2 + polynomial.C n * polynomial.X + polynomial.C (2 * m)) x1 ∧ is_root (polynomial.C 1 * polynomial.X^2 + polynomial.C n * polynomial.X + polynomial.C (2 * m)) x2 := 
by {
  -- Proof is omitted
  sorry
}

-- Part 2: Finding values of m and n for which the equation has two equal real roots that are integers
theorem equal_integer_roots_example : 
  ∃ (m n x : ℝ), n = 4 ∧ m = 2 ∧ x = -2 ∧ is_root (polynomial.C 1 * polynomial.X^2 + polynomial.C n * polynomial.X + polynomial.C (2 * m)) x :=
by {
  -- Proof is omitted
  sorry
}

end distinct_real_roots_equal_integer_roots_example_l709_709470


namespace participation_schemes_l709_709815

/-- There are four students {A, B, C, D} participating in four competitions {Mathematics, Writing, Science, English}.
    Student A cannot participate in the Writing competition. 
    We need to prove that there are 18 different ways to assign the students to the competitions. -/
theorem participation_schemes : ∀ (students : Fin 4 → Fin 4), (∀ (i : Fin 4), i ≠ 1 → students 0 ≠ 1) →
  (finset.univ.card = 4) →
  (∃! schemes : ℕ, schemes = 18) :=
by
  -- Given: Student A (students 0) cannot participate in Writing (1).
  -- Prove: The number of participation schemes is 18.
  sorry

end participation_schemes_l709_709815


namespace coeff_x3_expansion_range_x_value_l709_709698

-- The first part of the problem
theorem coeff_x3_expansion :
  let f (x : ℤ) := (1 - x)^5 + (1 - x)^6 + (1 - x)^7 + (1 - x)^8
  in (f(0) = 1) ∧ (f(1) = -121) := sorry

-- The second part of the problem
theorem range_x_value :
  ∀ x : ℝ,
  let f (x : ℝ) := (2 - x)^6
  in ∀ (second_term first_term third_term : ℝ),
    second_term ≤ first_term ∧ second_term ≥ third_term → 
    0 ≤ x ∧ x < (1 / 3) := sorry

end coeff_x3_expansion_range_x_value_l709_709698


namespace relationship_abc_l709_709720

-- Definition of the function f and its properties
variables {f : ℝ → ℝ} {a b c : ℝ}

-- Conditions
axiom even_func : ∀ x, f(x) = f(-x)
axiom periodic_func : ∀ x, f(x) = f(x + 2)
axiom monotonic_neg1_to_0 : ∀ x y, -1 ≤ x → x ≤ y → y ≤ 0 → f(x) ≥ f(y)

-- Definitions of a, b, and c
def a := f (real.sqrt 2)
def b := f 2
def c := f 3

-- Proof statement of the relationship between a, b, and c
theorem relationship_abc : b < a ∧ a < c :=
by 
  sorry 

end relationship_abc_l709_709720


namespace julia_total_cost_l709_709182

theorem julia_total_cost
  (snickers_cost : ℝ := 1.5)
  (mm_cost : ℝ := 2 * snickers_cost)
  (pepsi_cost : ℝ := 2 * mm_cost)
  (bread_cost : ℝ := 3 * pepsi_cost)
  (snickers_qty : ℕ := 2)
  (mm_qty : ℕ := 3)
  (pepsi_qty : ℕ := 4)
  (bread_qty : ℕ := 5)
  (money_given : ℝ := 5 * 20) :
  ((snickers_qty * snickers_cost) + (mm_qty * mm_cost) + (pepsi_qty * pepsi_cost) + (bread_qty * bread_cost)) > money_given := 
by
  sorry

end julia_total_cost_l709_709182


namespace math_problem_proof_l709_709831

open Real

/-- Proof of the problem given conditions and required answers. -/
theorem math_problem_proof :
  (∀ (A B M : Point), 
    (dist A B = 2) →
    (A = (M.1 * 2, 0)) →
    (B = (0, M.2 * 2)) →
    dist M (0, 0) = 1) ∧
  (∀ (x y : ℝ), 
    (x^2 + y^2 = 1) →
    -5 ≤ (3 * x - 4 * y) ∧ (3 * x - 4 * y) ≤ 5) ∧
  (∃ t λ : ℝ, 
    t ≠ 2/3 ∧ 
    t = 3/2 ∧ λ = 3/2 ∧
    (∀ (S : Point), 
      (S.1^2 + S.2^2 = 1) →
      dist S (0, t) = λ * dist S (0, 2/3))) :=
begin
  sorry
end

end math_problem_proof_l709_709831


namespace arithmetic_sequence_and_formula_l709_709992

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709992


namespace correct_calculation_l709_709676

theorem correct_calculation (a : ℝ) :
  2 * a^4 * 3 * a^5 = 6 * a^9 :=
by
  sorry

end correct_calculation_l709_709676


namespace greatest_root_of_f_l709_709795

noncomputable def f (x : ℝ) : ℝ := 15 * x^4 - 13 * x^2 + 2

theorem greatest_root_of_f :
  ∃ x : ℝ, f(x) = 0 ∧ ∀ y : ℝ, f(y) = 0 → y ≤ x ∧ x = (Real.sqrt 6) / 3 :=
sorry

end greatest_root_of_f_l709_709795


namespace distribute_balls_into_boxes_l709_709892

theorem distribute_balls_into_boxes :
  (∃ (ways : ℕ), ways = 3 ∧
  ∀ (a b : ℕ), a + b = 5 →
    (a = 5 ∧ b = 0) ∨ (a = 4 ∧ b = 1) ∨ (a = 3 ∧ b = 2)) :=
by
  use 3
  split
  case a => rfl
  case b => sorry

end distribute_balls_into_boxes_l709_709892


namespace solve_expr_l709_709315

theorem solve_expr (x : ℝ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end solve_expr_l709_709315


namespace earl_stuffing_rate_l709_709777

-- Conditions as definitions
def ellen_rate (E : ℝ) : ℝ := E / 1.5
def combined_rate (E : ℝ) (L : ℝ) : ℝ := E + L

-- Theorem statement: Prove that Earl can stuff 36 envelopes per minute.
theorem earl_stuffing_rate : ∃ E : ℝ, 
  (let L := ellen_rate E in 
  combined_rate E L = 60) ∧ 
  (ellen_rate E = E / 1.5) := 
begin
  use 36,
  split,
  { -- Show combined rate is 60 when E = 36
    let L := ellen_rate 36,
    unfold ellen_rate combined_rate,
    norm_num,
    sorry }, -- proof to be filled
  { -- Show Ellen's rate is consistent with L = E / 1.5
    unfold ellen_rate,
    norm_num,
    sorry } -- proof to be filled
end

end earl_stuffing_rate_l709_709777


namespace total_cost_l709_709962

theorem total_cost
  (permits_cost : ℕ)
  (contractor_hourly_rate : ℕ)
  (contractor_days : ℕ)
  (contractor_hours_per_day : ℕ)
  (inspector_discount : ℕ)
  (h_pc : permits_cost = 250)
  (h_chr : contractor_hourly_rate = 150)
  (h_cd : contractor_days = 3)
  (h_chpd : contractor_hours_per_day = 5)
  (h_id : inspector_discount = 80)
  (contractor_total_hours : ℕ := contractor_days * contractor_hours_per_day)
  (contractor_total_cost : ℕ := contractor_total_hours * contractor_hourly_rate)
  (inspector_cost : ℕ := contractor_total_cost - (contractor_total_cost * inspector_discount / 100))
  (total_cost : ℕ := permits_cost + contractor_total_cost + inspector_cost) :
  total_cost = 2950 :=
by
  sorry

end total_cost_l709_709962


namespace probability_P_S_mod_7_l709_709294

noncomputable def choose (n k : ℕ) : ℕ := nat.choose n k

theorem probability_P_S_mod_7 :
  let total_pairs := choose 100 2,
      valid_pairs := total_pairs - choose 86 2 in
  (valid_pairs : ℚ) / total_pairs = 1295 / 4950 := 
by
  sorry

end probability_P_S_mod_7_l709_709294


namespace points_earned_l709_709190

-- Define the given conditions
def points_per_enemy := 5
def total_enemies := 8
def enemies_remaining := 6

-- Calculate the number of enemies defeated
def enemies_defeated := total_enemies - enemies_remaining

-- Calculate the points earned based on the enemies defeated
theorem points_earned : enemies_defeated * points_per_enemy = 10 := by
  -- Insert mathematical operations
  sorry

end points_earned_l709_709190


namespace ab_ac_bc_all_real_l709_709228

theorem ab_ac_bc_all_real (a b c : ℝ) (h : a + b + c = 1) : ∃ x : ℝ, ab + ac + bc = x := by
  sorry

end ab_ac_bc_all_real_l709_709228


namespace find_scooters_l709_709721

variables (b t s : ℕ)

theorem find_scooters (h1 : b + t + s = 13) (h2 : 2 * b + 3 * t + 2 * s = 30) : s = 9 :=
sorry

end find_scooters_l709_709721


namespace magic_square_vector_sum_zero_l709_709515

-- Definitions for the problem
def magic_square (n : ℕ) (square : array (n × n) ℕ) : Prop :=
  (∀ i : fin n, (∑ j, square[i][j] = ∑ i, square[i][j]))

-- The proposition we aim to prove
theorem magic_square_vector_sum_zero (n : ℕ) (square : array (n × n) ℕ) 
  (h_square : magic_square n square)
  (h_elements : ∀ i j, square[i][j] ∈ set_of (fin (n*n)))
  (h_unique : set.card (set.univ (fin (n*n))) = n^2) :
  ∑ i j, if square[i][j] < square[i][j+1] then (square[i][j+1] - square[i][j]) else (square[i][j] - square[i][j+1]) = 0 :=
sorry

end magic_square_vector_sum_zero_l709_709515


namespace factorize_expression_l709_709052

theorem factorize_expression (a b : ℝ) : ab^2 - 2ab + a = a * (b-1)^2 := 
sorry

end factorize_expression_l709_709052


namespace trace_ellipse_l709_709265

-- Define the problem:
def complex_circle_ellipse (z : ℂ) (h : complex.abs z = 3) : Prop :=
  ∃ (x y : ℝ), (2*z + 1/z).re = x ∧ (2*z + 1/z).im = y ∧ (x^2) / (361 / 9) + (y^2) / (289 / 9) = 1

-- Statement to be proven:
theorem trace_ellipse (z : ℂ) (h : complex.abs z = 3) : complex_circle_ellipse z h :=
sorry

end trace_ellipse_l709_709265


namespace quiet_at_meeting_l709_709181

def Student (n : ℕ) := Fin n → Bool -- True represents talkative, False represents quiet
def is_friend (n : ℕ) (i j : Fin n) : Prop := sorry -- Define friendship relation

noncomputable def satisfies_condition (n : ℕ) (students : Student n) (friends : Fin n → Set (Fin n)) : Prop :=
  ∀ (i : Fin n), students i → (∃ j, j ∈ friends i ∧ ¬students j) ∧ (students i ∧ (friends i ∩ (λ j, ¬students j).to_set).card % 2 = 1)

theorem quiet_at_meeting (n : ℕ) :
  ∀ (students : Student n) (friends : Fin n → Set (Fin n)),
  satisfies_condition n students friends →
  ∃ M : Finset (Fin n), M.card ≥ n / 2 ∧ ∀ i ∈ M, students i → (friends i ∩ (λ j, j ∈ M ∧ ¬students j).to_set).card % 2 = 1 :=
begin
  sorry
end

end quiet_at_meeting_l709_709181


namespace part_i_part_ii_l709_709065

-- Defining m(P) and the size of the set P
def m (P : Set ℕ) [Finite P] (hP : ∀ p ∈ P, Nat.Prime p) : ℕ := sorry

-- Part (i)
theorem part_i (P : Set ℕ) [Finite P] (hPnonempty : P.Nonempty) (hP : ∀ p ∈ P, Nat.Prime p) :
  P.toFinset.card ≤ m P ∧ (P.toFinset.card = m P ↔ P.toFinset.min' hPnonempty > P.toFinset.card) := 
  sorry

-- Part (ii)
theorem part_ii (P : Set ℕ) [Finite P] (hP : ∀ p ∈ P, Nat.Prime p) :
  m P < (P.toFinset.card + 1) * (2 ^ P.toFinset.card - 1) := 
  sorry

end part_i_part_ii_l709_709065


namespace car_travel_distance_l709_709339

open Real

theorem car_travel_distance (distance_per_interval : ℝ) (time_per_interval : ℝ) (total_time : ℝ) (intervals_proof : total_time / time_per_interval = 80) : total_distance = 160 :=
by
  have one_interval : distance_per_interval = 2 := rfl 
  have interval_time : time_per_interval = 2.25 := rfl 
  have total_time_calc : total_time = 180 := rfl 
  have calculated_intervals : total_time / time_per_interval = 80 := intervals_proof
  have total_distance := distance_per_interval * (total_time / time_per_interval)
  sorry

end car_travel_distance_l709_709339


namespace smallest_possible_bob_number_l709_709743

theorem smallest_possible_bob_number : 
  let alices_number := 60
  let bobs_smallest_number := 30
  ∃ (bob_number : ℕ), (∀ p : ℕ, Prime p → p ∣ alices_number → p ∣ bob_number) ∧ bob_number = bobs_smallest_number :=
by
  sorry

end smallest_possible_bob_number_l709_709743


namespace student_observed_days_l709_709020

/-- Observations on weather for n days --/
def observations (n : ℕ) :=
  let clear_afternoons := 5 in
  let clear_mornings := 7 in
  let rainy_halfdays := 8 in
  let rainy_afternoons_clear_mornings := 
      ∃ r, r ≤ n ∧ r = 7 - (clear_mornings - (clear_afternoons - (n - rainy_halfdays))) in
  ∃ n, clear_afternoons + clear_mornings - rainy_halfdays = 4 ∧
       (n =  (7 - (clear_mornings - clear_afternoons)) + 
              (clear_afternoons - (7 - clear_mornings)) + 
              (clear_mornings - clear_afternoons))

theorem student_observed_days: ∃ n, observations n ∧ n = 8 :=
by
  sorry

end student_observed_days_l709_709020


namespace point_location_l709_709524

def point_in_fourth_quadrant (x : ℝ) : Prop :=
  let px := x^2 + 1
  let py := -2
  px > 0 ∧ py < 0

theorem point_location (x : ℝ) : point_in_fourth_quadrant x := by
  let px := x^2 + 1
  let py := -2
  have h1 : px > 0 := by nlinarith
  have h2 : py < 0 := by linarith
  exact ⟨h1, h2⟩

end point_location_l709_709524


namespace rhombus_side_length_l709_709284

theorem rhombus_side_length (V : ℝ) (r : ℝ) (a : ℝ) (A : ℝ) :
  V = 12 * Real.sqrt 3 → 
  r = 2.4 → 
  A = 4.8 * a → 
  V = 2.4 * a * (1/(2.4)) * (A) \ 
  a = 6 :=
sorry

end rhombus_side_length_l709_709284


namespace arithmetic_sequence_formula_max_sum_first_n_terms_l709_709449

-- Define the arithmetic sequence and conditions
variable {a : ℕ → ℤ} (a10 : a 10 = 24) (a25 : a 25 = -21)

-- Define the general formula for the sequence
theorem arithmetic_sequence_formula :
  ∃ a1 d, a n = a1 + (n - 1) * d ∧ a1 = 51 ∧ d = -3 :=
by
  -- Sorry to skip the proof (not providing exact solution steps)
  sorry

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (n : ℕ) : ℤ :=
  (n * (2 * 51 + (n - 1) * (-3))) / 2

-- Determine the value of n for which S_n is maximized
theorem max_sum_first_n_terms :
  ∃ n ∈ {16, 17}, ∀ m : ℕ, sum_first_n_terms m ≤ sum_first_n_terms n :=
by
  -- Sorry to skip the proof (not providing exact solution steps)
  sorry

end arithmetic_sequence_formula_max_sum_first_n_terms_l709_709449


namespace GDP_and_CPI_l709_709570

def PriceQuantityData :=
  { year2009_price_alpha := 5,
    year2009_quantity_alpha := 12,
    year2009_price_beta := 7,
    year2009_quantity_beta := 8,
    year2009_price_gamma := 9,
    year2009_quantity_gamma := 6,
    year2015_price_alpha := 6,
    year2015_quantity_alpha := 15,
    year2015_price_beta := 5,
    year2015_quantity_beta := 10,
    year2015_price_gamma := 10,
    year2015_quantity_gamma := 2 }

theorem GDP_and_CPI (data : PriceQuantityData):
  let nominal_GDP_2009 := data.year2009_price_alpha * data.year2009_quantity_alpha +
                          data.year2009_price_beta * data.year2009_quantity_beta +
                          data.year2009_price_gamma * data.year2009_quantity_gamma in
  let nominal_GDP_2015 := data.year2015_price_alpha * data.year2015_quantity_alpha +
                          data.year2015_price_beta * data.year2015_quantity_beta +
                          data.year2015_price_gamma * data.year2015_quantity_gamma in
  let real_GDP_2015 := data.year2009_price_alpha * data.year2015_quantity_alpha +
                       data.year2009_price_beta * data.year2015_quantity_beta +
                       data.year2009_price_gamma * data.year2015_quantity_gamma in
  let growth_rate_real_GDP := (real_GDP_2015 - nominal_GDP_2009) / nominal_GDP_2009.toReal in
  let basket_2015 := data.year2009_quantity_alpha * data.year2015_price_alpha +
                     data.year2009_quantity_beta * data.year2015_price_beta +
                     data.year2009_quantity_gamma * data.year2015_price_gamma in
  let CPI_2015 := (basket_2015.toReal / nominal_GDP_2009.toReal) * 100 in
  nominal_GDP_2009 = 170 ∧
  nominal_GDP_2015 = 160 ∧
  real_GDP_2015 = 163 ∧
  growth_rate_real_GDP = -0.0412 ∧
  CPI_2015 = 101.17 := 
by 
  simp only [PriceQuantityData, *, Int.toReal];
  sorry

end GDP_and_CPI_l709_709570


namespace album_count_l709_709368

theorem album_count (A B S : ℕ) (hA : A = 23) (hB : B = 9) (hS : S = 15) : 
  (A - S) + B = 17 :=
by
  -- Variables and conditions
  have Andrew_unique : ℕ := A - S
  have Bella_unique : ℕ := B
  -- Proof starts here
  sorry

end album_count_l709_709368


namespace determine_a_l709_709159

def y (x : ℝ) (a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem determine_a (a : ℝ) : is_even (y _ a) → a = 2 :=
sorry

end determine_a_l709_709159


namespace probability_of_f_le_0_l709_709865

def f (x : ℝ) := x^2 - 5 * x + 6

theorem probability_of_f_le_0 :
  let domain : set ℝ := set.Icc (-5 : ℝ) 5
  let favorable : set ℝ := set.Icc 2 3
  (favorable.card : ℝ) / (domain.card : ℝ) = 1 / 10 :=
sorry

end probability_of_f_le_0_l709_709865


namespace sphere_surface_area_l709_709591

-- Definitions of given conditions
def AB : ℝ := 1
def BC : ℝ := Real.sqrt 3
def AC : ℝ := 2
def Vmax : ℝ := (Real.sqrt 3) / 2

-- Theorem statement
theorem sphere_surface_area (AB BC AC Vmax : ℝ)
  (hAB : AB = 1) (hBC : BC = Real.sqrt 3) (hAC : AC = 2) (hVmax : Vmax = (Real.sqrt 3) / 2) :
  4 * Real.pi * (5 / 3) ^ 2 = 100 * Real.pi / 9 :=
by
  -- Proof would go here
  sorry

end sphere_surface_area_l709_709591


namespace season_cost_l709_709967

def first_half_cost (episodes_first_half : ℕ) (cost_per_episode_first_half : ℕ) : ℕ :=
  episodes_first_half * cost_per_episode_first_half

def second_half_cost (episodes_second_half : ℕ) (cost_per_episode_second_half : ℕ) : ℕ :=
  episodes_second_half * cost_per_episode_second_half

theorem season_cost (episodes_total : ℕ) (cost_per_episode_first_half : ℕ)
                    (increase_rate : ℚ) (episodes_first_half episodes_second_half : ℕ) :
  episodes_total = episodes_first_half + episodes_second_half →
  2 * episodes_first_half = episodes_total →
  let cost_per_episode_second_half := cost_per_episode_first_half + 
                                        (increase_rate * cost_per_episode_first_half).to_nat in
  first_half_cost episodes_first_half cost_per_episode_first_half +
  second_half_cost episodes_second_half cost_per_episode_second_half = 35200 :=
by sorry

end season_cost_l709_709967


namespace shape_is_plane_l709_709428

-- Define cylindrical coordinates
structure CylindricalCoord :=
  (r : ℝ) (theta : ℝ) (z : ℝ)

-- Define the condition
def condition (c : ℝ) (coord : CylindricalCoord) : Prop :=
  coord.z = c

-- The shape is described as a plane
def is_plane : Prop := ∀ (coord1 coord2 : CylindricalCoord), (coord1.z = coord2.z)

theorem shape_is_plane (c : ℝ) : 
  (∀ coord : CylindricalCoord, condition c coord) ↔ is_plane :=
by 
  sorry

end shape_is_plane_l709_709428


namespace point_D_coordinates_l709_709834

noncomputable def point := ℝ × ℝ

def A : point := (2, 3)
def B : point := (-1, 5)

def vector_sub (p1 p2 : point) : point := (p1.1 - p2.1, p1.2 - p2.2)
def scalar_mul (k : ℝ) (v : point) : point := (k * v.1, k * v.2)
def vector_add (p1 p2 : point) : point := (p1.1 + p2.1, p1.2 + p2.2)

def D : point := vector_add A (scalar_mul 3 (vector_sub B A))

theorem point_D_coordinates : D = (-7, 9) :=
by
  -- Proof goes here
  sorry

end point_D_coordinates_l709_709834


namespace find_diff_p_s_l709_709814

open Nat

noncomputable def p : ℕ := 42
noncomputable def q : ℕ := 24
noncomputable def r : ℕ := 168
noncomputable def s : ℕ := 18

def pq_cond := p * q + p + q = 1074
def qr_cond := q * r + q + r = 506
def rs_cond := r * s + r + s = 208
def product_cond := p * q * r * s = fact 12

theorem find_diff_p_s : pq_cond ∧ qr_cond ∧ rs_cond ∧ product_cond → p - s = 24 := by
  intros
  sorry

end find_diff_p_s_l709_709814


namespace Mike_catches_correct_l709_709543

def Joe_catches : ℕ := 23

def Derek_catches (J : ℕ) : ℕ := 2 * J - 4

def Tammy_catches (D : ℕ) : ℕ := D / 3 + 16

def Mike_catches (T : ℕ) : ℕ := (2 * T) * 6 / 5

theorem Mike_catches_correct : 
  let J := Joe_catches in
  let D := Derek_catches J in
  let T := Tammy_catches D in
  let M := Mike_catches T in
  M = 72 := by
  sorry

end Mike_catches_correct_l709_709543


namespace longest_side_triangle_l709_709025

open Real EuclideanSpace

noncomputable def dist (a b : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2))

theorem longest_side_triangle :
  let A := (2:ℝ, 2:ℝ)
  let B := (5:ℝ, 6:ℝ)
  let C := (8:ℝ, 2:ℝ)
  max (dist A B) (max (dist A C) (dist B C)) = 6 :=
by {
  -- Conditions (vertices of the triangle)
  let A := (2:ℝ, 2:ℝ),
  let B := (5:ℝ, 6:ℝ),
  let C := (8:ℝ, 2:ℝ),
  
  -- Distances between the points
  have dAB : dist A B = 5 := sorry,
  have dAC : dist A C = 6 := sorry,
  have dBC : dist B C = 5 := sorry,
  
  -- Maximum distance
  show max (dist A B) (max (dist A C) (dist B C)) = 6,
  rw [dAB, dAC, dBC],
  exact rfl,
}

end longest_side_triangle_l709_709025


namespace even_function_a_value_l709_709139

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, let y := (x - 1)^2 + a * x + sin(x + π / 2) in y = y) ↔ a = 2 :=
by
  let f := λ x, (x - 1)^2 + a * x + sin(x + π / 2)
  have h_even : ∀ x : ℝ, f(-x) = f(x) ↔ (a = 2) := sorry
  exact h_even

end even_function_a_value_l709_709139


namespace find_y_when_x_is_seven_l709_709460

theorem find_y_when_x_is_seven (k c : ℝ) 
  (h1 : 10 = 5 * k + c) 
  (h2 : 6 = k + c) : 
  let y := k * 7 + c in y = 12 := 
by {
  sorry
}

end find_y_when_x_is_seven_l709_709460


namespace find_a_l709_709444

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 < a^2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}
def C : Set ℝ := {x | 1 < x ∧ x < 2}

theorem find_a (a : ℝ) (h : A a ∩ B = C) : a = 2 ∨ a = -2 := by
  sorry

end find_a_l709_709444


namespace extreme_points_inequality_l709_709096

noncomputable def f (x : ℝ) (m : ℝ) := (1 / 2) * x^2 + m * Real.log (1 - x)

theorem extreme_points_inequality (m x1 x2 : ℝ) 
  (h_m1 : 0 < m) (h_m2 : m < 1 / 4)
  (h_x1 : 0 < x1) (h_x2: x1 < 1 / 2)
  (h_x3: x2 > 1 / 2) (h_x4: x2 < 1)
  (h_x5 : x1 < x2)
  (h_sum : x1 + x2 = 1)
  (h_prod : x1 * x2 = m)
  : (1 / 4) - (1 / 2) * Real.log 2 < (f x1 m) / x2 ∧ (f x1 m) / x2 < 0 :=
by
  sorry

end extreme_points_inequality_l709_709096


namespace problem_statement_l709_709483

-- Define the vectors and their properties using Lean 4
variables (a b c : ℝ × ℝ)

-- Conditions
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Given conditions
def vector_a := (3, -4)
def vector_b := (2, x)
def vector_c := (2, y)

variable (x y : ℝ)

-- Proof problem statement
theorem problem_statement (h1 : is_parallel vector_a vector_b) (h2 : is_perpendicular vector_a vector_c) :
  vector_b.1 * vector_c.1 + vector_b.2 * vector_c.2 = 0 ∧ 
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 180 ∧ cos θ = 0 ∧ θ = 90 :=
by
  sorry

end problem_statement_l709_709483


namespace monotonicity_of_f_min_value_diff_of_f_l709_709099

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x - (1 / x) + 2 * a * Real.log x

theorem monotonicity_of_f (a : ℝ) :
  (a >= -1 → ∀ x y, 0 < x → x < y → f x a < f y a) ∧
  (a < -1 → ((∀ x, 0 < x → x < -a - Real.sqrt(a^2 - 1) → f x a < f (-a - Real.sqrt(a^2 - 1)) a) ∧
             (∀ x, -a + Real.sqrt(a^2 - 1) < x → x < y → x < +∞ → f x a < f y a))) :=
sorry

theorem min_value_diff_of_f (a : ℝ) (x₁ x₂ : ℝ) (hx₂ : x₂ ∈ Set.Ici Real.exp 1) :
  f x₁ a - f x₂ a = 4 / Real.exp 1 :=
sorry

end monotonicity_of_f_min_value_diff_of_f_l709_709099


namespace rahul_batting_average_l709_709601

theorem rahul_batting_average:
  ∃ (A : ℝ), A = 46 ∧
  (∀ (R : ℝ), R = 138 → R = 54 * 4 - 78 → A = R / 3) ∧
  ∃ (n_matches : ℕ), n_matches = 3 :=
by
  sorry

end rahul_batting_average_l709_709601


namespace star_problem_l709_709770

-- Define the star operation
def star (X Y : ℝ) : ℝ := (X + Y) / 4

-- Define the specific problem to prove
theorem star_problem : star (star 3 11) 6 = 2.375 :=
by
  rw [star, star] -- expand both uses of star
  sorry -- proof to be completed

end star_problem_l709_709770


namespace exists_large_subset_free_of_arithmetic_progressions_l709_709659

open Finset

noncomputable def isFreeOfArithmeticProgressions (A : Finset ℕ) : Prop :=
  ∀ ⦃a b c : ℕ⦄, a ∈ A → b ∈ A → c ∈ A → a ≠ b → a ≠ c → b ≠ c → a + b ≠ 2 * c

theorem exists_large_subset_free_of_arithmetic_progressions :
  ∃ (A : Finset ℕ), A ⊆ range (3^8) ∧ A.card ≥ 256 ∧ isFreeOfArithmeticProgressions A :=
begin
  sorry
end

end exists_large_subset_free_of_arithmetic_progressions_l709_709659


namespace bob_fencing_needed_l709_709373

/-- Definitions and conditions -/
def length_rectangular_plot := 225
def width_rectangular_plot := 125
def side1_irregular := 75
def side2_irregular := 150
def side3_irregular := 45
def side4_irregular := 120
def gate1 := 3
def gate2 := 10
def gate3 := 4
def gate4 := 7

/-- Statement to prove -/
theorem bob_fencing_needed :
  let P_rect := 2 * (length_rectangular_plot + width_rectangular_plot),
      P_irregular := side1_irregular + side2_irregular + side3_irregular + side4_irregular,
      P_total := P_rect + P_irregular,
      Gates := gate1 + gate2 + gate3 + gate4
  in P_total - Gates = 1066 :=
by sorry

end bob_fencing_needed_l709_709373


namespace fill_675_cans_time_l709_709350

theorem fill_675_cans_time :
  (∀ (cans_per_batch : ℕ) (time_per_batch : ℕ) (total_cans : ℕ),
    cans_per_batch = 150 →
    time_per_batch = 8 →
    total_cans = 675 →
    total_cans / cans_per_batch * time_per_batch = 36) :=
begin
  intros cans_per_batch time_per_batch total_cans h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry
end

end fill_675_cans_time_l709_709350


namespace regular_polygon_interior_angle_of_108_has_5_sides_l709_709936

theorem regular_polygon_interior_angle_of_108_has_5_sides (interior_angle : ℝ) 
  (h : interior_angle = 108) : 
  let exterior_angle := 180 - interior_angle in
  let n := 360 / exterior_angle in
  n = 5 := 
by 
  unfold exterior_angle n
  rw [h, sub_eq_add_neg, add_neg_eq_sub]
  norm_num
sorry

end regular_polygon_interior_angle_of_108_has_5_sides_l709_709936


namespace total_trip_time_l709_709374

-- Define the points and times of travel
variables {A B C X Y : Point}
variables (t1 t2 t_total_stop : ℕ) (dist : ℝ)
variables (constant_speed : ℝ)

-- Define the time taken from X to B and from B to Y, and the stop time at B
def time_from_X_to_B := dist / constant_speed
def time_from_B_to_Y := dist / constant_speed
def stop_time_at_B := 5

-- The condition that the travel from some point X to B adding to B to Y with equal properties
axiom dist_condition_X : (distance X C) = (distance X A) + (distance X B)
axiom dist_condition_Y : (distance Y A) = (distance Y B) + (distance Y C)

-- The total time calculation considering all points and stopping time
def total_time (dist : ℝ) (speed : ℝ) : ℕ := 
  (dist / constant_speed) * 2 + 25 + stop_time_at_B

-- The final theorem stating the total trip time from A to C
theorem total_trip_time : total_time dist constant_speed = 180 :=
by 
  sorry

end total_trip_time_l709_709374


namespace ellipse_equation_parallelogram_area_const_l709_709832

-- Define the problem conditions
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def has_eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = Real.sqrt 2 / 2 ∧ (a^2 - b^2) / a^2 = e^2

def point_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x = b ∧ y = a / b

-- Define the statement for problem (1)
theorem ellipse_equation (a b : ℝ) (h_ecc : has_eccentricity a b (Real.sqrt 2 / 2)) (h_point : point_on_ellipse a b b (a / b)) :
  ∃ (a b : ℝ), a^2 = 8 ∧ b^2 = 4 :=
by
  sorry  -- Proof to be provided

-- Define the statement for problem (2)
theorem parallelogram_area_const (a b : ℝ) (h : is_ellipse a b) (P M N : ℝ × ℝ) (S : ℝ) :
  quadrilateral_OMPN_is_parallelogram O P M N → was_valid_quadrilateral O P M N → area_of_OMPN_quadrilateral P M N = 2*Real.sqrt 6 :=
by
  sorry  -- Proof to be provided

end ellipse_equation_parallelogram_area_const_l709_709832


namespace probability_abs_xi_less_1_point_96_l709_709851

noncomputable def standard_normal (ξ : ℝ → ℝ) :=
  ∃ μ σ : ℝ, μ = 0 ∧ σ = 1 ∧ (∀ x, ξ x = Real.exp (- (x - μ)^2 / (2 * σ^2)) / (σ * Real.sqrt (2 * Real.pi)))

theorem probability_abs_xi_less_1_point_96 (ξ : ℝ → ℝ)
  (Hξ : standard_normal ξ)
  (H₁ : ∃ p, p = 0.025 ∧ P (ξ < -1.96) = p):
  P (|ξ| < 1.96) = 0.95 :=
sorry

end probability_abs_xi_less_1_point_96_l709_709851


namespace even_function_iff_a_eq_2_l709_709153

noncomputable def y (a x : ℝ) : ℝ :=
  (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_iff_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, y a x = y a (-x)) ↔ a = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end even_function_iff_a_eq_2_l709_709153


namespace range_of_expression_l709_709529

variables {R : Type} [linear_ordered_field R] [inhabited R]
          {A B C : ℝ × ℝ} 
          (O : ℝ × ℝ := (0, 0))
          (λ μ : ℝ)

def on_circle (P : ℝ × ℝ) := P.1^2 + P.2^2 = 1

def vector_eq (A B C : ℝ × ℝ) (λ μ : ℝ) :=
  let OA := (A.1 - O.1, A.2 - O.2),
      OB := (B.1 - O.1, B.2 - O.2),
      OC := (C.1 - O.1, C.2 - O.2) in
  OC = (λ * OA.1 + μ * OB.1, λ * OA.2 + μ * OB.2)

theorem range_of_expression :
  on_circle A ∧ on_circle B ∧ on_circle C ∧
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  vector_eq A B C λ μ →
  ∃ δ > 0, (λ - 1)^2 + (μ - 3)^2 > δ ∧ (λ - 1)^2 + (μ - 3)^2 ∈ Ioi (1/2) :=
sorry

end range_of_expression_l709_709529


namespace AD_parallel_QC_l709_709224

noncomputable def P : Point := sorry
noncomputable def Γ : Circle := sorry
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry
noncomputable def Q : Point := sorry

-- Definitions provided conditions
axiom P_outside_Γ : ¬ (P ∈ Γ)

axiom tangent_PA_A : tangent P A Γ
axiom tangent_PB_B : tangent P B Γ

axiom A_B_lie_on_segment : lies_on_segment A B C
axiom circle_PBC_intersects_Γ_at_D : intersects (circle_through P B C) Γ D

axiom A_midpoint_PQ : midpoint A P Q

-- Statement to be proven
theorem AD_parallel_QC : parallel (line_through A D) (line_through Q C) := 
sorry

end AD_parallel_QC_l709_709224


namespace probability_ratio_l709_709786

noncomputable def choose (n k : ℕ) : ℕ := nat.choose n k

def p : ℚ := 10 / (choose 50 5)

def q : ℚ := ((choose 10 2) * (choose 5 3) * (choose 5 2)) / (choose 50 5)

theorem probability_ratio : q / p = 450 := 
by
  sorry

end probability_ratio_l709_709786


namespace k_value_solution_set_l709_709475

variable (a : ℝ) (k : ℝ)
variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom h1 : a > 0
axiom h2 : a ≠ 1
axiom h3 : ∀ x : ℝ, f x = k * a^x - a^(-x)
axiom h4 : ∀ x : ℝ, f (-x) = -f x -- f(x) is odd
axiom h5 : f 1 > 0

-- Part (1)
theorem k_value : k = 1 :=
by 
  have f_0 : f 0 = 0 := by sorry
  have k_minus_1_eq_0 : k - 1 = 0 := by sorry
  exact k_minus_1_eq_0

-- Part (2)
theorem solution_set : {x | f (x^2 + 2*x) + f (x - 4) > 0} = {x | x > 1 ∨ x < -4} := 
by 
  have a_gt_1 : a > 1 := by sorry
  have f_strictly_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry
  have f_inequality : ∀ x : ℝ, f (x^2 + 2*x) + f (x - 4) > 0 ↔ x > 1 ∨ x < -4 := by sorry
  exact f_inequality

end k_value_solution_set_l709_709475


namespace arithmetic_sequence_and_formula_l709_709993

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709993


namespace equilateral_triangle_rectangle_count_l709_709211

theorem equilateral_triangle_rectangle_count (A B C : Point) (hABC : equilateral_triangle A B C) :
  ∃ n : ℕ, n = 12 ∧ count_rectangles_two_vertices_shared A B C n :=
sorry

end equilateral_triangle_rectangle_count_l709_709211


namespace suitcase_combination_l709_709248

theorem suitcase_combination : 
  let A := 3
  let S := 9
  let T := 1
  let O := 4
  let R := 5
  let M := 2
  let N := 0
  let E := 8
  let STORM := S * 10^4 + T * 10^3 + O * 10^2 + R * 10 + M
  let TORN := T * 10^3 + O * 10^2 + R * 10 + N
  let TOSS := T * 10^3 + O * 10^2 + S * 10 + S
  let ASTERN := A * 10^5 + S * 10^4 + T * 10^3 + E * 10^2 + R * 10 + N
  in 
    A + STORM - TORN - TOSS = ASTERN :=
by
  sorry

end suitcase_combination_l709_709248


namespace minimal_moves_red_first_l709_709564

namespace chip_problem

/-- Given an integer k, and 4k chips (2k red, 2k blue), determine that 
    the smallest number of moves needed to arrange the first 2k chips 
    to be red is k. -/
theorem minimal_moves_red_first (k : ℕ) (h : k ≥ 1) :
  ∀ (seq : List (sum Unit Unit)), 
  (seq.length = 4 * k ∧ seq.countp (λ x, x = Sum.inl ()) = 2 * k ∧ seq.countp (λ x, x = Sum.inr ()) = 2 * k) →
  ∃ n, 
    (∀ s1 s2, 
      s1 ++ s2 = seq → 
      ∀ R B R' B', 
        s1 = R ++ B ∧ s2 = R' ++ B' → 
        (#moves_needed s1 s2 ≤ k)) ∧
    (n = k) :=  
sorry

end chip_problem

end minimal_moves_red_first_l709_709564


namespace number_of_true_propositions_l709_709846

-- Defining the function and stating necessary properties
def f (x : ℝ) : ℝ :=
  if x < 0 then e^x * (x + 1)
  else if x = 0 then 0
  else e^(-x) * (x - 1)

-- Stating the actual propositions
def proposition1 : Prop := ∀ x > 0, f x = -e^(-x) * (x - 1)
def proposition2 : Prop := ∃! x : ℝ, f x = 0
def proposition3 : Prop := ∀ x, (f x < 0 ↔ (x ∈ Iio (-1) ∪ Ioc 0 1))
def proposition4 : Prop := ∀ x1 x2 : ℝ, abs (f x1 - f x2) < 2

-- Stating the problem's equivalent proof problem
theorem number_of_true_propositions :
  (¬proposition1) ∧ (¬proposition2) ∧ proposition3 ∧ proposition4 ↔ (2 : ℕ) := sorry

end number_of_true_propositions_l709_709846


namespace smallest_n_divisibility_l709_709225

theorem smallest_n_divisibility :
  let a := Real.pi / 4032 in
  ∃ n : ℕ, n > 0 ∧ 4032 ∣ n^3 + n^2 ∧ n = 7 :=
by
  sorry

end smallest_n_divisibility_l709_709225


namespace unique_solution_log_eq_range_l709_709503

theorem unique_solution_log_eq_range (a : ℝ) :
  (∃! x : ℝ, Math.log (4 * x^2 + 4 * a * x) = Math.log (4 * x - a + 1))
  ↔ (a ≥ 1 / 5 ∧ a < 1) :=
by 
  sorry

end unique_solution_log_eq_range_l709_709503


namespace point_locus_is_minor_arc_l709_709451

noncomputable def isosceles_triangle := {A B C : Type*} [metric_space A] (B C : A) :
  ∃ (AB AC : A), AB = AC ∧ (∀ P : A, dist P B = dist P C → dist P B = dist P A) :=
by sorry 

theorem point_locus_is_minor_arc (A B C P : Type*) [metric_space A] (d : Type*) [metric_space d] [metric_space P]
  (isosceles_triangle : A → B → C → d)
  (condition : ∀ P : A, dist P B = dist P C → dist P B = dist P A)
  : isosceles_triangle =
    ∃ (AB AC : A), AB = AC ∧ (∀ P : A, dist P B = dist P C → dist P B = dist P A)
    =
    {P : A | dist (dist P B) = (dist P A) * (dist P C)} :=
  by sorry

end point_locus_is_minor_arc_l709_709451


namespace proof_probability_p_s_two_less_than_multiple_of_7_l709_709297

noncomputable def probability_p_s_two_less_than_multiple_of_7 : ℚ :=
  let possible_integers := {a : ℕ | 1 ≤ a ∧ a ≤ 100}
  let distinct_pairs := {p : ℕ × ℕ | p.1 ∈ possible_integers ∧ p.2 ∈ possible_integers ∧ p.1 ≠ p.2}
  let valid_pairs := {p : ℕ × ℕ | p ∈ distinct_pairs ∧ (let a := p.1, b := p.2 in
                        let s := a + b
                        let p := a * b
                        p + s ≡ 5 [MOD 7])}
  (fintype.card valid_pairs : ℚ) / (fintype.card distinct_pairs : ℚ)

theorem proof_probability_p_s_two_less_than_multiple_of_7 :
  probability_p_s_two_less_than_multiple_of_7 = 7 / 330 :=
sorry

end proof_probability_p_s_two_less_than_multiple_of_7_l709_709297


namespace total_time_to_school_and_back_l709_709708

-- Definition of the conditions
def speed_to_school : ℝ := 3 -- in km/hr
def speed_back_home : ℝ := 2 -- in km/hr
def distance : ℝ := 6 -- in km

-- Proof statement
theorem total_time_to_school_and_back : 
  (distance / speed_to_school) + (distance / speed_back_home) = 5 := 
by
  sorry

end total_time_to_school_and_back_l709_709708


namespace measure_of_angle_BPC_l709_709944

noncomputable def square_side_length := 6
noncomputable def angle_abe := 45 -- in degrees

-- Definitions of points and properties based on the given conditions
variables (A B C D E P Q : Type) -- Points in the geometric construction
variables (is_square : is_square A B C D square_side_length)
variables (is_right_triangle : is_right_triangle A B E)
variables (angle_ABE_eq_45 : ∠ A B E = angle_abe)
variables (intersect_AC_BE_at_P : line_segment AC = line_segment BE ∩ P)
variables (Q_on_BC : is_on_line Q B C)
variables (PQ_perpendicular_to_BC : is_perpendicular PQ BC)
variables (PQ_length_eq_x : length PQ = x)

theorem measure_of_angle_BPC : ∠ B P C = 90 := by sorry

end measure_of_angle_BPC_l709_709944


namespace inequality_proof_l709_709807

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
    (h4 : (3 / (a * b * c)) ≥ (a + b + c)) : 
    (1 / a + 1 / b + 1 / c) ≥ (a + b + c) :=
  sorry

end inequality_proof_l709_709807


namespace upper_bound_of_expression_l709_709066

theorem upper_bound_of_expression (n : ℤ) (h1 : ∀ (n : ℤ), 4 * n + 7 > 1 ∧ 4 * n + 7 < 111) :
  ∃ U, (∀ (n : ℤ), 4 * n + 7 < U) ∧ 
       (∀ (n : ℤ), 4 * n + 7 < U ↔ 4 * n + 7 < 111) ∧ 
       U = 111 :=
by
  sorry

end upper_bound_of_expression_l709_709066


namespace hair_cut_length_l709_709536

theorem hair_cut_length (original_length after_haircut : ℕ) (h1 : original_length = 18) (h2 : after_haircut = 9) :
  original_length - after_haircut = 9 :=
by
  sorry

end hair_cut_length_l709_709536


namespace balls_in_boxes_l709_709127

theorem balls_in_boxes : (2^7 = 128) := 
by
  -- number of balls
  let n : ℕ := 7
  -- number of boxes
  let b : ℕ := 2
  have h : b ^ n = 128 := by sorry
  exact h

end balls_in_boxes_l709_709127


namespace incorrect_statement_implies_m_eq_zero_l709_709848

theorem incorrect_statement_implies_m_eq_zero
  (m : ℝ)
  (y : ℝ → ℝ)
  (h : ∀ x, y x = m * x + 4 * m - 2)
  (intersects_y_axis_at : y 0 = -2) :
  m = 0 :=
sorry

end incorrect_statement_implies_m_eq_zero_l709_709848


namespace GDP_and_CPI_l709_709571

def PriceQuantityData :=
  { year2009_price_alpha := 5,
    year2009_quantity_alpha := 12,
    year2009_price_beta := 7,
    year2009_quantity_beta := 8,
    year2009_price_gamma := 9,
    year2009_quantity_gamma := 6,
    year2015_price_alpha := 6,
    year2015_quantity_alpha := 15,
    year2015_price_beta := 5,
    year2015_quantity_beta := 10,
    year2015_price_gamma := 10,
    year2015_quantity_gamma := 2 }

theorem GDP_and_CPI (data : PriceQuantityData):
  let nominal_GDP_2009 := data.year2009_price_alpha * data.year2009_quantity_alpha +
                          data.year2009_price_beta * data.year2009_quantity_beta +
                          data.year2009_price_gamma * data.year2009_quantity_gamma in
  let nominal_GDP_2015 := data.year2015_price_alpha * data.year2015_quantity_alpha +
                          data.year2015_price_beta * data.year2015_quantity_beta +
                          data.year2015_price_gamma * data.year2015_quantity_gamma in
  let real_GDP_2015 := data.year2009_price_alpha * data.year2015_quantity_alpha +
                       data.year2009_price_beta * data.year2015_quantity_beta +
                       data.year2009_price_gamma * data.year2015_quantity_gamma in
  let growth_rate_real_GDP := (real_GDP_2015 - nominal_GDP_2009) / nominal_GDP_2009.toReal in
  let basket_2015 := data.year2009_quantity_alpha * data.year2015_price_alpha +
                     data.year2009_quantity_beta * data.year2015_price_beta +
                     data.year2009_quantity_gamma * data.year2015_price_gamma in
  let CPI_2015 := (basket_2015.toReal / nominal_GDP_2009.toReal) * 100 in
  nominal_GDP_2009 = 170 ∧
  nominal_GDP_2015 = 160 ∧
  real_GDP_2015 = 163 ∧
  growth_rate_real_GDP = -0.0412 ∧
  CPI_2015 = 101.17 := 
by 
  simp only [PriceQuantityData, *, Int.toReal];
  sorry

end GDP_and_CPI_l709_709571


namespace grid_divisibility_l709_709811

theorem grid_divisibility (n : ℕ) (h_div : n % 7 = 0) (h_gt : n > 7) :
  ∃ m : ℕ, n * n = 7 * m ∧ (∃ arrangement : bool, arrangement = true) := sorry

end grid_divisibility_l709_709811


namespace inequality_solution_set_for_a_eq_8_inequality_range_of_a_l709_709869

-- Proof Problem (1)
theorem inequality_solution_set_for_a_eq_8:
  {x : ℝ} (a : ℝ) (ha : a = 8) : 
  (|2 * x - 1| - |x - 1| <= Real.log2 a) → (-3 <= x ∧ x <= 3) := by
  sorry

-- Proof Problem (2)
theorem inequality_range_of_a:
  {x : ℝ} : 
  (∃ x, |2 * x - 1| - |x - 1| <= Real.log2 a) ↔ (a ∈ Set.Ici (Real.sqrt 2 / 2)) := by
  sorry

end inequality_solution_set_for_a_eq_8_inequality_range_of_a_l709_709869


namespace find_a_for_even_function_l709_709144

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, (x-1)^2 + a * x + sin (x + π / 2) = ((-x)-1)^2 + (-a * x) + sin (-x + π / 2)) →
  a = 2 :=
by
  sorry

end find_a_for_even_function_l709_709144


namespace alpha_squared_plus_3alpha_plus_beta_equals_2023_l709_709900

-- Definitions and conditions
variables (α β : ℝ)
-- α and β are roots of the quadratic equation x² + 2x - 2025 = 0
def is_root_of_quadratic_1 : Prop := α^2 + 2 * α - 2025 = 0
def is_root_of_quadratic_2 : Prop := β^2 + 2 * β - 2025 = 0
-- Vieta's formula gives us α + β = -2
def sum_of_roots : Prop := α + β = -2

-- Theorem (statement) we want to prove
theorem alpha_squared_plus_3alpha_plus_beta_equals_2023 (h1 : is_root_of_quadratic_1 α)
                                                      (h2 : is_root_of_quadratic_2 β)
                                                      (h3 : sum_of_roots α β) :
                                                      α^2 + 3 * α + β = 2023 :=
by
  sorry

end alpha_squared_plus_3alpha_plus_beta_equals_2023_l709_709900


namespace f_sqrt2_l709_709566

def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 1 then log (x + 1) / log (1 / 2)
  else f (x - 1)

theorem f_sqrt2 : f (sqrt 2) = -1/2 :=
by
  sorry

end f_sqrt2_l709_709566


namespace fill_pipe_fraction_l709_709718

theorem fill_pipe_fraction (x : ℝ) (h : x = 1 / 2) : x = 1 / 2 :=
by
  sorry

end fill_pipe_fraction_l709_709718


namespace not_possible_to_construct_l709_709395

/-- The frame consists of 54 unit segments. -/
def frame_consists_of_54_units : Prop := sorry

/-- Each part of the construction set consists of three unit segments. -/
def part_is_three_units : Prop := sorry

/-- Each vertex of a cube is shared by three edges. -/
def vertex_shares_three_edges : Prop := sorry

/-- Six segments emerge from the center of the cube. -/
def center_has_six_segments : Prop := sorry

/-- It is not possible to construct the frame with exactly 18 parts. -/
theorem not_possible_to_construct
  (h1 : frame_consists_of_54_units)
  (h2 : part_is_three_units)
  (h3 : vertex_shares_three_edges)
  (h4 : center_has_six_segments) : 
  ¬ ∃ (parts : ℕ), parts = 18 :=
sorry

end not_possible_to_construct_l709_709395


namespace common_element_in_all_sets_l709_709173

variables {α : Type*} (sets : fin 21 → set α)

-- Condition: Each set has exactly 55 elements
def has_exactly_55_elements : Prop :=
  ∀ i : fin 21, (sets i).card = 55

-- Condition: Any four sets have exactly one common element
def any_four_sets_have_one_common_element : Prop :=
  ∀ (i j k l : fin 21), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
  (sets i ∩ sets j ∩ sets k ∩ sets l).card = 1

theorem common_element_in_all_sets :
  has_exactly_55_elements sets →
  any_four_sets_have_one_common_element sets →
  ∃ a : α, ∀ i : fin 21, a ∈ sets i :=
sorry

end common_element_in_all_sets_l709_709173


namespace angle_ABC_measure_l709_709493

theorem angle_ABC_measure : 
  ∀ (angle_CBD angle_ABD angle_sum : ℝ), 
    angle_CBD = 90 ∧ 
    angle_ABD = 60 ∧ 
    angle_sum = 200 →
    ∃ angle_ABC : ℝ, angle_ABC = 50 := 
by
  intros angle_CBD angle_ABD angle_sum h 
  obtain ⟨hCBD, hABD, hSum⟩ := h 
  use 50 
  have h_eq : angle_ABC + angle_ABD + angle_CBD = 200 := by rw [hSum]
  have h_values : 50 + 60 + 90 = 200 := by norm_num
  exact ⟨refl 50, h_eq.symm ▸ h_values⟩

end angle_ABC_measure_l709_709493


namespace distance_A_A_l709_709652

-- Definitions according to conditions
def point_A : ℝ × ℝ := (1, -3)
def point_A' : ℝ × ℝ := (1, 3)

-- Distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The proof statement
theorem distance_A_A' : distance point_A point_A' = 6 := 
by 
  -- We skip the proof with sorry
  sorry

end distance_A_A_l709_709652


namespace solve_for_x_l709_709130

noncomputable def f (x : ℝ) : ℝ := x^3

noncomputable def f_prime (x : ℝ) : ℝ := 3

theorem solve_for_x (x : ℝ) (h : f_prime x = 3) : x = 1 ∨ x = -1 :=
by
  sorry

end solve_for_x_l709_709130


namespace frustum_smaller_cone_height_l709_709719

theorem frustum_smaller_cone_height (H frustum_height radius1 radius2 : ℝ) 
  (h : ℝ) (h_eq : h = 30 - 18) : 
  radius1 = 6 → radius2 = 10 → frustum_height = 18 → H = 30 → h = 12 := 
by
  intros
  sorry

end frustum_smaller_cone_height_l709_709719


namespace geometric_log_sum_equals_five_l709_709092

noncomputable def geometric_sequence_sum (a : ℝ) (r : ℝ) (h : 0 < a ∧ 0 < r ∧ (a * r) * (a * r^3) = 9) : ℝ :=
  let a₁ := a,
      a₂ := a * r,
      a₃ := a * r^2,
      a₄ := a * r^3,
      a₅ := a * r^4
  in (Real.log (a₁ * a₂ * a₃ * a₄ * a₅) / Real.log 3)

theorem geometric_log_sum_equals_five (a r : ℝ) (h : 0 < a ∧ 0 < r ∧ (a * r) * (a * r^3) = 9) :
  geometric_sequence_sum a r h = 5 :=
sorry

end geometric_log_sum_equals_five_l709_709092


namespace imaginary_part_of_z_l709_709502

theorem imaginary_part_of_z (z : ℂ) (h : z * (3 - 4 * complex.I) = 1) : complex.im z = 4 / 25 :=
by
  sorry

end imaginary_part_of_z_l709_709502


namespace deltaH_relationship_l709_709435

/- Define the conditions -/
variables (a b c d : ℝ)
variables (H1 : a ∈ ℝ)
variables (H2 : b ∈ ℝ)
variables (H3 : c ∈ ℝ)
variables (H4 : d ∈ ℝ)

/- Define the statements -/
def condition1 : Prop := H1
def condition2 : Prop := H2
def condition3 : Prop := H3
def condition4 : Prop := H4

/- The theorem to prove the relationship -/
theorem deltaH_relationship (H1 : condition1) (H2 : condition2) (H3 : condition3) (H4 : condition4) : 
  2 * a = b ∧ b < 0 :=
by
  sorry

end deltaH_relationship_l709_709435


namespace min_value_of_expression_l709_709131

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) : 
  x + 2 * y ≥ 9 + 4 * Real.sqrt 2 := 
sorry

end min_value_of_expression_l709_709131


namespace evaluate_fractions_l709_709086

theorem evaluate_fractions (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 := 
by
  sorry

end evaluate_fractions_l709_709086


namespace count_eq_58_l709_709116

-- Definitions based on given conditions
def isThreeDigit (n : Nat) : Prop := n >= 100 ∧ n <= 999
def digitSum (n : Nat) : Nat := 
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 + d2 + d3
def isPerfectSquare (n : Nat) : Prop := ∃ k : Nat, k * k = n

-- Condition for digit sum being one of the specific perfect squares
def isPerfectSquareDigitSum (n : Nat) : Prop :=
  isThreeDigit n ∧ (digitSum n = 1 ∨ digitSum n = 4 ∨ digitSum n = 9 ∨ digitSum n = 16 ∨ digitSum n = 25)

-- Total count of such numbers
def countPerfectSquareDigitSumNumbers : Finset Nat :=
  (Finset.range 1000).filter isPerfectSquareDigitSum

-- Lean statement to prove the count is 58
theorem count_eq_58 : countPerfectSquareDigitSumNumbers.card = 58 := sorry

end count_eq_58_l709_709116


namespace balls_in_boxes_l709_709126

theorem balls_in_boxes : (2^7 = 128) := 
by
  -- number of balls
  let n : ℕ := 7
  -- number of boxes
  let b : ℕ := 2
  have h : b ^ n = 128 := by sorry
  exact h

end balls_in_boxes_l709_709126


namespace ab_eq_cd_l709_709973

theorem ab_eq_cd
  (A B C D K P : Point) 
  (h_convex : convex_quadrilateral A B C D)
  (h_common : common_point (ray_through A B) (ray_through D C) K)
  (h_bisector : on_bisector P (angle_bisector A K D))
  (h_bp_bisects_ac : bisects_segment (line_through B P) (segment A C))
  (h_cp_bisects_bd : bisects_segment (line_through C P) (segment B D)) :
  dist A B = dist C D := 
sorry

end ab_eq_cd_l709_709973


namespace sum_of_first_n_terms_b_l709_709948

def sum_first_n_terms_b (n : ℕ) : ℕ :=
  (1 to n).sum (λ k, (k + 1)^3 - k^3)

theorem sum_of_first_n_terms_b (n : ℕ) : 
  sum_first_n_terms_b n = n^3 + 4 * n^2 + 3 * n := 
by sorry

end sum_of_first_n_terms_b_l709_709948


namespace vertical_asymptote_count_l709_709391

def f (x : ℝ) : ℝ := (x + 3) / (x^2 - 2 * x - 15)

theorem vertical_asymptote_count : 
  let num_asymptotes := { x | x ∈ ({5} : Set ℝ) } in
  num_asymptotes.card = 1 :=
by
  sorry

end vertical_asymptote_count_l709_709391


namespace area_triangle_COD_l709_709765

noncomputable def area_of_triangle (t s : ℝ) : ℝ := 
  1 / 2 * abs (5 + 2 * s + 7 * t)

theorem area_triangle_COD (t s : ℝ) : 
  ∃ (C : ℝ × ℝ) (D : ℝ × ℝ), 
    C = (3 + 5 * t, 2 + 4 * t) ∧ 
    D = (2 + 5 * s, 3 + 4 * s) ∧ 
    area_of_triangle t s = 1 / 2 * abs (5 + 2 * s + 7 * t) :=
by
  sorry

end area_triangle_COD_l709_709765


namespace find_x_l709_709459

theorem find_x (x : ℝ) (h : x^2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 :=
sorry

end find_x_l709_709459


namespace domain_of_f_l709_709620

open Set Function

-- Define the function f
def f (x : ℝ) : ℝ := log (x^2 + 3*x + 2)

-- Define the domain of the function
def domain_f : Set ℝ := {x | x < -2 ∨ x > -1}

-- Theorem to prove the domain of f
theorem domain_of_f :
  ∀ x, x ∈ domain_f ↔ (x^2 + 3*x + 2 > 0) := by
  intro x
  dsimp [domain_f]
  split -- splitting the logical or
  · intro hx
    -- when x < -2
    calc x^2 + 3*x + 2
         = (x + 2) * (x + 1) : by ring
     ... > 0                 : sorry
  · intro hx
    -- when x > -1
    calc x^2 + 3*x + 2
         = (x + 2) * (x + 1) : by ring
     ... > 0                 : sorry


end domain_of_f_l709_709620


namespace simplify_fraction_l709_709251

theorem simplify_fraction :
  (1 : ℚ) / ((1 / (1 / 3 : ℚ) ^ 1) + (1 / (1 / 3 : ℚ) ^ 2) + (1 / (1 / 3 : ℚ) ^ 3) + (1 / (1 / 3 : ℚ) ^ 4)) = 1 / 120 := 
by 
  sorry

end simplify_fraction_l709_709251


namespace train_time_original_l709_709735

theorem train_time_original (D : ℝ) (T : ℝ) 
  (h1 : D = 48 * T) 
  (h2 : D = 60 * (2/3)) : T = 5 / 6 := 
by
  sorry

end train_time_original_l709_709735


namespace math_problem_l709_709697

open Real

noncomputable def first_problem : Prop :=
  (log 2) ^ 2 + log 5 * log 20 + (sqrt 2016) ^ 0 + 0.027 ^ (2/3) * (1/3) ^ (-2) = 102

noncomputable def second_problem (α : ℝ) : Prop :=
  (3 * tan α) / (tan α - 2) = -1 → 7 / ((sin α) ^ 2 + sin α * cos α + (cos α) ^ 2) = 5

theorem math_problem:
  first_problem ∧ ∀ α, second_problem α :=
by
  constructor
  · unfold first_problem
    sorry
  · intro α
    unfold second_problem
    sorry

end math_problem_l709_709697


namespace white_pairs_coincide_l709_709776

def num_red : Nat := 4
def num_blue : Nat := 4
def num_green : Nat := 2
def num_white : Nat := 6
def red_pairs : Nat := 3
def blue_pairs : Nat := 2
def green_pairs : Nat := 1 
def red_white_pairs : Nat := 2
def green_blue_pairs : Nat := 1

theorem white_pairs_coincide :
  (num_red = 4) ∧ 
  (num_blue = 4) ∧ 
  (num_green = 2) ∧ 
  (num_white = 6) ∧ 
  (red_pairs = 3) ∧ 
  (blue_pairs = 2) ∧ 
  (green_pairs = 1) ∧ 
  (red_white_pairs = 2) ∧ 
  (green_blue_pairs = 1) → 
  4 = 4 :=
by
  sorry

end white_pairs_coincide_l709_709776


namespace find_k_l709_709550

variables {A B C M N A' B' C' : Type*}
variables [improper_semi_det A B C]
variables {k : ℝ}

def midpoint (X Y : Type*) := sorry

def median_extension (X Y M Z : Type*) (k : ℝ) := X + k * (Y + Z - 2 * X)

noncomputable def triangle_equilateral (X Y Z : Type*) := sorry

theorem find_k 
(ABC_triang : triangle A B C)
(pos_k : k > 0)
(median_AM : M = midpoint B C)
(median_BN : N = midpoint A C)
(line_AA' : A' = median_extension A B M C k)
(line_BB' : B' = median_extension B C N A k)
(line_CC' : C' = median_extension C A M B k)
(equil_A'B'C' : triangle_equilateral A' B' C') : 
k = 1 / real.sqrt 3 := sorry

end find_k_l709_709550


namespace average_of_first_17_even_numbers_starting_from_100_l709_709660

def average_first_17_even_from_100 : ℤ := 116

theorem average_of_first_17_even_numbers_starting_from_100 :
  let seq := (λ n, 100 + 2 * n) in
  let terms := (list.range 17).map seq in
  let sum := terms.sum in
  let average := sum / 17 in
  average = average_first_17_even_from_100 :=
by
  sorry

end average_of_first_17_even_numbers_starting_from_100_l709_709660


namespace compare_negatives_l709_709383

theorem compare_negatives : -1 > -2 := 
by 
  sorry

end compare_negatives_l709_709383


namespace odd_vs_even_digits_count_l709_709318

noncomputable def is_odd_digit (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9
noncomputable def is_even_digit (n : ℕ) : Prop := n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

noncomputable def count_only_odd_digits_nat (m : ℕ) : ℕ :=
  if m < 1 then 0 else
  let rec count_odd (n : ℕ) : ℕ :=
    match n with
    | 0     => 1
    | n+1 => 5 * count_odd n
  in count_odd (Nat.log10 m)

noncomputable def count_only_even_digits_nat (m : ℕ) : ℕ :=
  if m < 1 then 0 else
  let rec count_even (n : ℕ) : ℕ :=
    match n with
    | 0     => 1
    | n+1 => 4 * 5^n + ((count_even n) * if n = 0 then 4 else 1)
  in count_even (Nat.log10 m)

theorem odd_vs_even_digits_count (m : ℕ) (h : 1 ≤ m ∧ m ≤ 60000) :
  let odd_count := count_only_odd_digits_nat m
  let even_count := count_only_even_digits_nat m
  odd_count - even_count = 780 :=
sorry

end odd_vs_even_digits_count_l709_709318


namespace min_value_fraction_ineq_l709_709432

-- Define the conditions and statement to be proved
theorem min_value_fraction_ineq (x : ℝ) (hx : x > 4) : 
  ∃ M, M = 4 * Real.sqrt 5 ∧ ∀ y : ℝ, y > 4 → (y + 16) / Real.sqrt (y - 4) ≥ M := 
sorry

end min_value_fraction_ineq_l709_709432


namespace solve_trig_eq_l709_709693

noncomputable def solution1 : ℝ := Real.arctan 0.973
noncomputable def solution2 : ℝ := Real.arctan (-0.59)

theorem solve_trig_eq (x : ℝ) (k : ℤ) (hx : cos x ≠ 0) :
  5.22 * sin x ^ 2 - 2 * sin x * cos x = 3 * cos x ^ 2 ↔
  x = solution1 + k * Real.pi ∨ x = solution2 + k * Real.pi := sorry

end solve_trig_eq_l709_709693


namespace maria_earnings_correct_l709_709584

/-- Maria's earnings calculation over three days -/
def maria_total_earnings : ℕ :=
  let first_day := (30 * 2) + (20 * 3) in
  let second_day := ((30 * 2) * 2) + ((20 * 3) * 2) in
  let third_day := ((30 * 2) * 0.1 * 2) + (16 * 3) in
  first_day + second_day + third_day
  
theorem maria_earnings_correct : 
  maria_total_earnings = 420 :=
by
  -- Formal proof steps would go here
  sorry

end maria_earnings_correct_l709_709584


namespace sum_a_k_less_than_1_l709_709610

theorem sum_a_k_less_than_1 (a : ℕ → ℚ)
  (h1 : a 1 = 1/2)
  (h2 : ∀ k : ℕ, k ≥ 2 → 2 * k * a k = (2 * k - 3) * a (k - 1)) :
  ∀ n : ℕ, ∑ k in Finset.range n.succ, a (k + 1) < 1 :=
by
  sorry

end sum_a_k_less_than_1_l709_709610


namespace greatest_C_l709_709215

theorem greatest_C (α : ℝ) (hα : 0 < α) : 
  (∃ C : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x * y + y * z + z * x = α → 
    (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ C * (x / z + z / x + 2)) 
    ∧ (∀ C' : ℝ, C' > C → ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y + y * z + z * x = α ∧ 
    (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) < C' * (x / z + z / x + 2)))) :=
  let C := 8 in
  exists.intro C sorry

end greatest_C_l709_709215


namespace integral_of_one_plus_sin_l709_709283

theorem integral_of_one_plus_sin : ∫ x in 0..(Real.pi / 2), (1 + Real.sin x) = Real.pi / 2 + 1 :=
by
  sorry

end integral_of_one_plus_sin_l709_709283


namespace ellipse_parabola_intersection_distance_l709_709081

noncomputable def ellipse := {
  center: (Float, Float),
  eccentricity: Float,
  right_focus: (Float, Float)
}

noncomputable def parabola := {
  equation: (Float, Float) -> Bool,
  directrix: Float
}

noncomputable def distance_between_points (A B : (Float, Float)) : Float :=
  (Math.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))

theorem ellipse_parabola_intersection_distance :
  let E := ellipse ⟨(0,0), 1/2, (2,0)⟩ in
  let C := parabola (λ x y => y^2 = 8 * x) (-2) in
  let A := ⟨-2, 3⟩ in
  let B := ⟨-2, -3⟩ in
  distance_between_points A B = 6 :=
by {
  -- Proof goes here
  sorry
}

end ellipse_parabola_intersection_distance_l709_709081


namespace equation_of_tangent_circle_l709_709841

theorem equation_of_tangent_circle 
(x y : ℝ) :
  (∃ r : ℝ, r = |1 + 2 * 2| / sqrt (1^2 + 2^2) ∧ (x - 1)^2 + (y - 2)^2 = r^2) ↔
  (x - 1)^2 + (y - 2)^2 = 5 :=
by 
  sorry

end equation_of_tangent_circle_l709_709841


namespace optimal_selling_price_and_max_profit_l709_709003

def profit_function (x : ℕ) : ℤ :=
  (100 - x) * (x - 40)

theorem optimal_selling_price_and_max_profit :
  (∃ x y : ℕ, x = 70 ∧ y = 900 ∧ profit_function x = y) :=
by
  let x := 70
  let y := 900
  have h1 : profit_function x = y :=
    by sorry
  use x, y
  exact ⟨rfl, rfl, h1⟩

end optimal_selling_price_and_max_profit_l709_709003


namespace yuna_initial_marbles_l709_709322

theorem yuna_initial_marbles (M : ℕ) :
  (M - 12 + 5) / 2 + 3 = 17 → M = 35 := by
  sorry

end yuna_initial_marbles_l709_709322


namespace boy_running_time_l709_709880

theorem boy_running_time :
  let side_length := 60
  let speed1 := 9 * 1000 / 3600       -- 9 km/h to m/s
  let speed2 := 6 * 1000 / 3600       -- 6 km/h to m/s
  let speed3 := 8 * 1000 / 3600       -- 8 km/h to m/s
  let speed4 := 7 * 1000 / 3600       -- 7 km/h to m/s
  let hurdle_time := 5 * 3 * 4        -- 3 hurdles per side, 4 sides
  let time1 := side_length / speed1
  let time2 := side_length / speed2
  let time3 := side_length / speed3
  let time4 := side_length / speed4
  let total_time := time1 + time2 + time3 + time4 + hurdle_time
  total_time = 177.86 := by
{
  -- actual proof would be provided here
  sorry
}

end boy_running_time_l709_709880


namespace middle_school_soccer_league_l709_709198

theorem middle_school_soccer_league (n : ℕ) (h : (n * (n - 1)) / 2 = 36) : n = 9 := 
  sorry

end middle_school_soccer_league_l709_709198


namespace two_digit_numbers_count_l709_709641

def card1 := {1, 2}
def card2 := {3, 4}

theorem two_digit_numbers_count : 
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ card1 ∪ card2, ∀ y ∈ card1 ∪ card2, x ≠ y → 
  ∃ d1 d2 : ℕ, (d1 ∈ card1 ∨ d1 ∈ card2) ∧ (d2 ∈ card1 ∨ d2 ∈ card2) ∧ 
  (10 * d1 + d2 = x * 10 + y) → n = 8 :=
by
  sorry

end two_digit_numbers_count_l709_709641


namespace evaluate_g_four_times_l709_709429

def g (z : ℂ) : ℂ :=
if (∃ x : ℝ, z = x) then -z^3 - 1 else z^3 + 1

theorem evaluate_g_four_times : g (g (g (g (2 + I)))) = (-(64555 : ℂ) + 70232 * I)^3 + 1 := 
by
sorry

end evaluate_g_four_times_l709_709429


namespace problem1_problem2_problem3_l709_709979

-- Definitions
def has_real_root (f : ℝ → ℝ) := ∃ x : ℝ, f x = x
def derivative_condition (f : ℝ → ℝ) := ∀ x : ℝ, 0 < deriv f x ∧ deriv f x < 1
def M := { f : ℝ → ℝ | has_real_root f ∧ derivative_condition f }

-- Problem 1: Prove that if f ∈ M, then f(x) - x = 0 has exactly one real root
theorem problem1 (f : ℝ → ℝ) (hf : f ∈ M) : ∀ x1 x2 : ℝ, f x1 = x1 ∧ f x2 = x2 → x1 = x2 := by
  sorry

-- Problem 2: Show that g(x) = (x / 2) - (ln x / 3) + 3 (for x > 1) is in M
def g (x : ℝ) : ℝ := (x / 2) - (Math.log x / 3) + 3
theorem problem2 : g ∈ M := by
  sorry

-- Problem 3: If f ∈ M, prove |f(α) - f(β)| < 2 for α, β in domain with |α - 2012| < 1 and |β - 2012| < 1
theorem problem3 (f : ℝ → ℝ) (hf : f ∈ M) (α β : ℝ)
  (hα : |α - 2012| < 1) (hβ : |β - 2012| < 1) : |f α - f β| < 2 := by
  sorry

end problem1_problem2_problem3_l709_709979


namespace ratio_red_to_black_l709_709180

-- Definitions derived from conditions
def red_cars : ℕ := 28
def black_cars : ℕ := 75

-- The statement proving the ratio of red cars to black cars
theorem ratio_red_to_black : red_cars = 28 → black_cars = 75 → red_cars / black_cars = 28 / 75 := 
by
  intros h_red h_black
  rw [h_red, h_black]
  sorry  -- Proof steps go here

end ratio_red_to_black_l709_709180


namespace sasha_added_number_divisible_l709_709971

noncomputable def digit_sum (digits : List ℕ) : ℕ :=
digits.foldl (+) 0

theorem sasha_added_number_divisible (d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 : ℕ) 
  (h_distinct : list.nodup [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10])
  (h_range : ∀ d ∈ [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10], d ≤ 9) 
  (h_not_all_digits : {x : ℕ | x ≤ 9} ⊆ {d1, d2, d3, d4, d5, d6, d7, d8, d9, d10}.to_finset) :
  ∃ d11 : ℕ, d11 ∈ (Finset.range 10).eraseₓ {d1, d2, d3, d4, d5, d6, d7, d8, d9, d10}.erase_lists ∧
  (digit_sum [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10] + d11) % 9 = 0 :=
by
  sorry

end sasha_added_number_divisible_l709_709971


namespace find_a_for_even_function_l709_709147

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, (x-1)^2 + a * x + sin (x + π / 2) = ((-x)-1)^2 + (-a * x) + sin (-x + π / 2)) →
  a = 2 :=
by
  sorry

end find_a_for_even_function_l709_709147


namespace Maria_soap_cost_l709_709579
-- Import the entire Mathlib library
  
theorem Maria_soap_cost (soap_last_months : ℕ) (cost_per_bar : ℝ) (months_in_year : ℕ):
  (soap_last_months = 2) -> 
  (cost_per_bar = 8.00) ->
  (months_in_year = 12) -> 
  (months_in_year / soap_last_months * cost_per_bar = 48.00) := 
by
  intros h_soap_last h_cost h_year
  sorry

end Maria_soap_cost_l709_709579


namespace geometric_and_arithmetic_sequences_l709_709838

noncomputable def geometric_ratio_q : ℕ → ℕ := λ n, 2

def a_sequence (n : ℕ) : ℕ := 3 * 2^(n-1)

def b_sequence (n : ℕ) : ℕ := (n + 3) - 3 * 2^(n-1)

def sum_b_sequence (n : ℕ) : ℕ :=
  (n * (n + 7)) / 2 - 3 * 2^n + 3

theorem geometric_and_arithmetic_sequences (n : ℕ) :
  (a_sequence 1 = 3) ∧
  (a_sequence 4 = 24) ∧
  (∀ k : ℕ, a_sequence k + b_sequence k = 4 + (k-1) * 1) ∧
  (sum_b_sequence n = ((n * (n + 7)) / 2 - 3 * 2^n + 3)) :=
by {
  sorry
}

end geometric_and_arithmetic_sequences_l709_709838


namespace value_of_f_one_is_two_l709_709553

noncomputable def S := { x : ℝ // x ≠ 0 }

variables (f : S → S)
  (h1 : ∀ x : S, f ⟨1 / (x : ℝ), by simpa using x.prop⟩ = x^2 * f x)
  (h2 : ∀ x y : S, x + y ≠ 0 → f ⟨1 / (x + y : ℝ), by simpa using add_ne_zero x.prop y.prop⟩ = 2 + f ⟨1 / (x : ℝ), by simpa using x.prop⟩ + f ⟨1 / (y : ℝ), by simpa using y.prop⟩)

theorem value_of_f_one_is_two : f (1 : S) = 2 :=
sorry

end value_of_f_one_is_two_l709_709553


namespace geo_seq_a4_a8_sum_equal_six_l709_709076

variable {a : ℕ → ℝ} [∀ n, 0 < a n]

theorem geo_seq_a4_a8_sum_equal_six 
  (h1 : a 6 * a 10 + a 3 * a 5 = 26)
  (h2 : a 5 * a 7 = 5) :
  a 4 + a 8 = 6 := 
sorry

end geo_seq_a4_a8_sum_equal_six_l709_709076


namespace arithmetic_sequence_and_formula_l709_709985

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709985


namespace find_p_and_min_cost_day_find_q_range_l709_709356

def startup_capital : ℝ := 100000
def experimental_cost (p : ℕ) (x : ℕ) : ℝ := p * x + 280
def total_cost_30_days (p : ℕ) : ℝ := 30 * (p + 280) + (30 * 29 / 2) * p
def sponsorship (q : ℝ) (x: ℕ) : ℝ := - q * (x ^ 2) + 50000
def average_daily_cost (startup_capital : ℝ) (p : ℕ) (x : ℕ) : ℝ := (startup_capital + x * 280 + (x * (x - 1) / 2) * p) / x
def average_daily_actual_cost (startup_capital : ℝ) (p : ℕ) (x : ℕ) (q : ℝ) : ℝ := (startup_capital + x * (p + 280) + (x * (x - 1) / 2) * p - (- q * (x ^ 2) + 50000)) / x

theorem find_p_and_min_cost_day (p : ℕ) :
  (total_cost_30_days p = 17700) → p = 20 ∧ average_daily_cost startup_capital 20 100 = 2290 := sorry

theorem find_q_range (q : ℝ) :
  (0 < q ∧ q <= 10 ∧ let x := (sqrt (50000 / (10 + q))) in x >= 50 ∧ average_daily_actual_cost startup_capital 20 x q = 2290) → (0 < q ∧ q <= 10) := sorry

end find_p_and_min_cost_day_find_q_range_l709_709356


namespace roots_of_equation_l709_709061

theorem roots_of_equation :
  ∀ x, 3 * sqrt x + 3 * x^(-1/2) = 7 ↔ x = ( (7 + sqrt 13) / 6 )^2 ∨ x = ( (7 - sqrt 13) / 6 )^2 :=
by sorry

end roots_of_equation_l709_709061


namespace range_of_a_l709_709923

theorem range_of_a {a : ℝ} (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, (a*x + 1)/(sqrt (a*x^2 - 4*a*x + 2)) (f x)) :
  (0 ≤ a) ∧ (a < 1/2) :=
sorry

end range_of_a_l709_709923


namespace multiplication_identity_l709_709910

theorem multiplication_identity (a b x y : ℝ) (h : a * b = x) (k : b / 100 = y) : 
  (a / 100) * y = 1.71 :=
by
  have h_subst : y = 1.46539 := k
  rw [←h_subst]
  have h_div : (a / 100) = 299.42163 := sorry
  rw [h_div]
  exact sorry

end multiplication_identity_l709_709910


namespace parallelogram_satisfies_l709_709032

-- Define the concept of axis-symmetric shapes
def is_axis_symmetric (shape : Type) : Prop := sorry

-- Define the concept of center-symmetric shapes
def is_center_symmetric (shape : Type) : Prop := sorry

-- Define the given shapes
inductive Shape
| EquilateralTriangle
| Parallelogram
| Rectangle
| Rhombus

-- Predicate indicating the given problem's criteria
def satisfies_criteria (shape : Shape) : Prop :=
is_center_symmetric shape ∧ ¬ is_axis_symmetric shape

-- The theorem stating the parallelogram satisfies the criteria
theorem parallelogram_satisfies : satisfies_criteria Shape.Parallelogram :=
sorry

end parallelogram_satisfies_l709_709032


namespace probability_P_S_mod_7_l709_709295

noncomputable def choose (n k : ℕ) : ℕ := nat.choose n k

theorem probability_P_S_mod_7 :
  let total_pairs := choose 100 2,
      valid_pairs := total_pairs - choose 86 2 in
  (valid_pairs : ℚ) / total_pairs = 1295 / 4950 := 
by
  sorry

end probability_P_S_mod_7_l709_709295


namespace B_is_power_function_l709_709747

-- Definition of a power function
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Functions given in the problem statement
def A (x : ℝ) : ℝ := 2 * x ^ 2
def B (x : ℝ) : ℝ := x ^ (-2)
def C (x : ℝ) : ℝ := x ^ 2 + x
def D (x : ℝ) : ℝ := -x ^ (-1)

-- Proof statement
theorem B_is_power_function : is_power_function B :=
sorry

end B_is_power_function_l709_709747


namespace largest_multiple_of_three_l709_709687

theorem largest_multiple_of_three (n : ℕ) (h : 3 * n + (3 * n + 3) + (3 * n + 6) = 117) : 3 * n + 6 = 42 :=
by
  sorry

end largest_multiple_of_three_l709_709687


namespace value_multiplied_by_15_l709_709278

theorem value_multiplied_by_15 (x : ℝ) (h : 3.6 * x = 10.08) : x * 15 = 42 :=
sorry

end value_multiplied_by_15_l709_709278


namespace solve_for_m_l709_709821

theorem solve_for_m (m α : ℝ) (h1 : Real.tan α = m / 3) (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
by
  sorry

end solve_for_m_l709_709821


namespace a_recurrence_l709_709352

-- Definition of an n-level-good set
def n_level_good (n : ℕ) (A : set ℕ) : Prop :=
  A ≠ ∅ ∧ A ⊆ {1, 2, ..., n} ∧ |A| ≤ finset.min' (A \ {0}) sorry

-- The number of n-level-good sets
def a : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 2
| (n + 3) := a (n + 2) + a (n + 1) + 1

-- Theorem statement to prove the recurrence relation for a
theorem a_recurrence (n : ℕ) : n > 0 → a (n + 2) = a (n + 1) + a n + 1 :=
begin
  sorry
end

end a_recurrence_l709_709352


namespace xiao_ming_incorrect_l709_709672

theorem xiao_ming_incorrect : ∃ (a b : ℚ), a > 0 ∧ b < 0 ∧ a > b ∧ (1/a) > (1/b) :=
by
  use (1 : ℚ),
  use (-1 : ℚ),
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  norm_num

end xiao_ming_incorrect_l709_709672


namespace square_perimeter_l709_709357

theorem square_perimeter (s : ℝ)
  (h1 : ∃ (s : ℝ), 4 * s = s * 1 + s / 4 * 1 + s * 1 + s / 4 * 1)
  (h2 : ∃ (P : ℝ), P = 4 * s)
  : (5/2) * s = 40 → 4 * s = 64 :=
by
  intro h
  sorry

end square_perimeter_l709_709357


namespace find_a_if_y_is_even_l709_709167

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem find_a_if_y_is_even (a : ℝ) (h : ∀ x : ℝ, y x a = y (-x) a) : a = 2 :=
by
  sorry

end find_a_if_y_is_even_l709_709167


namespace contrapositive_of_square_sum_zero_l709_709632

theorem contrapositive_of_square_sum_zero (a b : ℝ) :
  (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0 :=
by
  sorry

end contrapositive_of_square_sum_zero_l709_709632


namespace particle_position_after_180_moves_l709_709725

noncomputable def cis (θ : ℝ) : ℂ := complex.of_real (real.cos θ) + complex.I * complex.of_real (real.sin θ)

def move (z_n : ℂ) (ω : ℂ) : ℂ := ω * z_n + 8

theorem particle_position_after_180_moves :
  let ω := cis (real.pi / 3)
  let z_0 := (8 : ℂ)
  z_180 = 8 :=
  by
    let z_180 : ℂ := ω^180 * z_0 + 8 * ((1 - ω^180) / (1 - ω))
    have ω_6_eq_1 : ω^6 = 1 := by sorry -- This will use the fact that ω = cis (π / 3)
    have ω_180_eq_1 : ω^180 = (ω^6)^30 := by sorry -- With ω^6 = 1, ω^180 reduces to ((ω^6)^30)
    calc
      z_180 = 8 + 0 : by sorry -- Substituting ω_180 into equation and simplifying

end particle_position_after_180_moves_l709_709725


namespace diagonals_perpendicular_l709_709518

open Complex

variables {A B C D M N P Q : ℝ × ℝ}

-- Given: A quadrilateral ABCD with midpoints M, N, P, and Q
-- Conditions: Segments MP and NQ are equal in length
axiom M_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
axiom N_midpoint : N = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
axiom P_midpoint : P = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
axiom Q_midpoint : Q = ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
axiom MP_NQ_equal : dist M P = dist N Q

-- To Prove: AC is perpendicular to BD
theorem diagonals_perpendicular :
  (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0 :=
begin
  sorry
end

end diagonals_perpendicular_l709_709518


namespace total_cost_of_season_l709_709969

/-- 
John starts watching a TV show. He pays $1000 per episode for the first half of the season.
The second half of the season had episodes that cost 120% more expensive. 
If there are 22 episodes in total, we need to prove that the entire season cost $35,200.
-/
theorem total_cost_of_season (episodes : ℕ) (cost1 : ℕ) (cost_increase_pct : ℕ) (num_episodes : ℕ):
  episodes = 22 →
  cost1 = 1000 →
  cost_increase_pct = 120 →
  num_episodes = episodes / 2 →
  let cost2 := cost1 + (cost1 * cost_increase_pct / 100) in
  (num_episodes * cost1 + num_episodes * cost2) = 35200 :=
by
  intros h1 h2 h3 h4
  let cost2 := cost1 + (cost1 * cost_increase_pct / 100)
  sorry

end total_cost_of_season_l709_709969


namespace value_of_P2016_l709_709479

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 2
  else
    have H : 0 < n := nat.succ_pos (nat.pred n)
    have hf : n.pred + 1 = n := by simp [nat.succ_pred_eq_of_pos H]
    1 - 1 / sequence (n.pred + 1)

def product_sequence (n : ℕ) : ℝ :=
  if n = 0 then 1
  else (list.range n).map (λ n, sequence (n + 1)).prod

lemma sequence_periodicity {n : ℕ} (h : n >= 1) : sequence (n + 3) = sequence n :=
  by sorry

theorem value_of_P2016 : product_sequence 2016 = 1 :=
  by sorry

end value_of_P2016_l709_709479


namespace circles_satisfying_three_centers_similarity_l709_709241

-- Definitions of the problem conditions
def lines_parallel (l1 l2 : Line) : Prop := ∀ P Q : Point, P ∈ l1 → Q ∈ l2 → l1 ∥ l2

def circle_tangent_to_lines (S : Circle) (l1 l2 : Line) : Prop := 
  ∃ P Q : Point, P ∈ S ∧ Q ∈ S ∧ P ∈ l1 ∧ Q ∈ l2

def points_collinear (M O N : Point) : Prop := collinear M O N

-- Main statement of the theorem
theorem circles_satisfying_three_centers_similarity
  (l1 l2 : Line) (S1 : Circle) (M O N : Point) :
  lines_parallel l1 l2 →
  circle_tangent_to_lines S1 l1 l2 →
  points_collinear M O N →
  ∃ (S : Circle), circle_tangent_to_lines S l1 l2 ∧ ∃! N' : Point, N' = N :=
sorry

end circles_satisfying_three_centers_similarity_l709_709241


namespace find_a_b_l709_709909

noncomputable def y (a b x : ℝ) := a * (Real.ln x) + b * x^2 + x

theorem find_a_b (a b : ℝ) (f : ℝ → ℝ) (h₁ : f = y a b)
    (h_ext1 : ∀ f, Deriv f 1 = 0)
    (h_ext2 : ∀ f, Deriv f 2 = 0) :
  a = -2 / 3 ∧ b = -1 / 6 :=
sorry

end find_a_b_l709_709909


namespace binomial_problem_l709_709820

theorem binomial_problem
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ) 
  (H1 : ∀ (x : ℤ), (1 - 2 * x)^7 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7)
  (H2 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -1)
  (H3 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 - a_7 = 3^7) :
  (a_0 + a_2 + a_4 + a_6)^2 - (a_1 + a_3 + a_5 + a_7)^2 = -2187 := 
begin
  sorry
end

end binomial_problem_l709_709820


namespace lowest_possible_sum_l709_709975

noncomputable def f : ℕ → ℕ :=
λ n, if n ≤ 1997 then 1999 - n else if n = 1998 then 2 else if n = 1999 then 1 else 0

theorem lowest_possible_sum :
  (∀ n ≥ 1, f(n + f(n)) = 1) ∧ f(1998) = 2 →
  ∑ i in finset.range 2000, f(i + 1) = 1997003 :=
by
  sorry

end lowest_possible_sum_l709_709975


namespace main_theorem_l709_709073

variables {Plane : Type} [Nonempty Plane]
variables (α β γ : Plane) (m n : α → Prop)

-- Define m and n as lines in these planes
variable (line_m : ∀ {p : Plane}, ∀ {x y}, m x → m y → line x y)
variable (line_n : ∀ {p : Plane}, ∀ {x y}, n x → n y → line x y)

-- Proposition 1
def proposition1 (h1 : ∀ {x}, m x → α x) (h2 : ∀ {x}, m x → β x) : Prop :=
  (∀ x, α x → β x → False) → (∀ x y, line_m _ _ → β x → β y → False)

-- Proposition 4
def proposition4 (h1 : ∀ {x}, α x → β x) (h2 : ∃ p, α p ∧ γ p ∧ m p) (h3 : ∃ p, β p ∧ γ p ∧ n p) : Prop :=
  ∀ x y, line_m x y → line_n x y

theorem main_theorem :
  (∀ {x}, m x → α x) → (∀ {x}, m x → β x) →
  (∀ {x}, α x → β x → False) → (∀ x y, line_m _ _ → β x → β y → False) ∧
  (∀ {x}, α x → β x) → (∃ p, α p ∧ γ p ∧ m p) → (∃ p, β p ∧ γ p ∧ n p) →
  ∀ x y, line_m x y → line_n x y :=
begin
  sorry
end

end main_theorem_l709_709073


namespace max_t_eq_one_l709_709555

theorem max_t_eq_one {x y : ℝ} (hx : x > 0) (hy : y > 0) : 
  max (min x (y / (x^2 + y^2))) 1 = 1 :=
sorry

end max_t_eq_one_l709_709555


namespace probability_of_forming_triangle_l709_709257

-- Definitions for the problem conditions
def is_positive_integer (n : ℕ) : Prop := n > 0
def is_valid_combination (a b c : ℕ) : Prop := is_positive_integer a ∧ is_positive_integer b ∧ is_positive_integer c ∧ a + b + c = 6
def can_form_triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ a + c > b

-- The statement to prove the probability
theorem probability_of_forming_triangle : 
  let combinable_lengths := { (a, b, c) | is_valid_combination a b c } in 
  let triangle_forming_lengths := { (a, b, c) ∈ combinable_lengths | can_form_triangle a b c } in 
  (triangle_forming_lengths.card : ℚ) / combinable_lengths.card = 1 / 10 :=
by sorry

end probability_of_forming_triangle_l709_709257


namespace total_mowing_time_with_breaks_l709_709234

-- Condition Definitions
def lawn_section1_length := 60
def lawn_section1_width := 90
def lawn_section2_length := 30
def lawn_section2_width := 60
def swath_width_in_inches := 30
def overlap_in_inches := 6
def walking_speed := 4500  -- in feet per hour
def break_time_per_hour := 10 / 60  -- 10 minutes in hour format

-- Statement to verify the total mowing time including breaks
theorem total_mowing_time_with_breaks :
  let effective_swath_width := (swath_width_in_inches - overlap_in_inches) / 12 -- in feet
  let total_lawn_area := (lawn_section1_length * lawn_section1_width) + (lawn_section2_length * lawn_section2_width)
  let total_strips := lawn_section1_width / effective_swath_width
  let total_length_of_strips := total_strips * (lawn_section1_length + lawn_section2_length)
  let mowing_time_without_breaks := total_length_of_strips / walking_speed
  let total_break_time := (mowing_time_without_breaks / 1) * break_time_per_hour -- breaks every hour
  let total_time := mowing_time_without_breaks + total_break_time
  total_time ≈ (2.13 : ℝ) := 
by
  -- Calculation steps can be filled in here
  sorry

end total_mowing_time_with_breaks_l709_709234


namespace trigonometric_identity_1_l709_709334

theorem trigonometric_identity_1 :
  ( (Real.sqrt 3 * Real.sin (-1200 * Real.pi / 180)) / (Real.tan (11 * Real.pi / 3)) 
  - Real.cos (585 * Real.pi / 180) * Real.tan (-37 * Real.pi / 4) = (Real.sqrt 3 / 2) - (Real.sqrt 2 / 2) ) :=
by
  sorry

end trigonometric_identity_1_l709_709334


namespace find_b_for_line_passing_through_point_l709_709063

theorem find_b_for_line_passing_through_point :
  ∃ b : ℚ, 2 * b * 6 + (3 * b - 2) * (-10) = 5 * b + 6 ∧ b = 14 / 23 :=
by {
  have h : 2 * (14 / 23) * 6 + (3 * (14 / 23) - 2) * (-10) = 5 * (14 / 23) + 6 := sorry,
  exact ⟨14 / 23, h⟩
}

end find_b_for_line_passing_through_point_l709_709063


namespace least_number_to_subtract_l709_709667

theorem least_number_to_subtract (n m : ℕ) (h : n = 56783421) (d : m = 569) : (n % m) = 56783421 % 569 := 
by sorry

end least_number_to_subtract_l709_709667


namespace vertical_asymptote_values_l709_709809

def has_one_vertical_asymptote (f : ℝ → ℝ) : Prop :=
  ∃ a b ∈ {6, -4}, a ≠ b ∧ (∀ x ≠ a, abs (f x) → ∞) 

theorem vertical_asymptote_values (c : ℝ) :
  c = -48 ∨ c = -8 → has_one_vertical_asymptote (λ x, (x^2 + 2 * x + c) / (x^2 - 2 * x - 24)) :=
by
  sorry

end vertical_asymptote_values_l709_709809


namespace P_iff_q_l709_709903

variables (a b c: ℝ)

def P : Prop := a * c < 0
def q : Prop := ∃ α β : ℝ, α * β < 0 ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0

theorem P_iff_q : P a c ↔ q a b c := 
sorry

end P_iff_q_l709_709903


namespace find_distance_AB_l709_709197

-- Definition of the parametric equation of curve C1
def C1 (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 3 + 3 * Real.sin α)

-- Condition: Given point P satisfies O₂P = 2 * O₂M, derive parametric equation of C2
def C2 (α : ℝ) : ℝ × ℝ := (6 * Real.cos α, 6 + 6 * Real.sin α)

-- In polar coordinates, the intersections of the ray θ = π/3 with C1 and C2
def polar_C1 (θ : ℝ) : ℝ := 6 * Real.sin θ
def polar_C2 (θ : ℝ) : ℝ := 12 * Real.sin θ

-- Prove distance |AB| is 3sqrt(3) when θ = π/3
theorem find_distance_AB : 
  polar_C2 (Real.pi / 3) - polar_C1 (Real.pi / 3) = 3 * Real.sqrt 3 := 
sorry

end find_distance_AB_l709_709197


namespace no_roots_probability_l709_709091

noncomputable theory

open probability_theory

variables (σ : ℝ)
variables (ξ : ℝ) [normal_distribution : probability_distribution (normal ξ 1 σ^2)]

-- Statement of the problem:
theorem no_roots_probability : 
  P (λ ξ, no_roots (quadratic_of_x_plus_c 2 1 ξ)) = 1 / 2 := sorry

end no_roots_probability_l709_709091


namespace b_arithmetic_a_formula_l709_709998

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l709_709998


namespace nested_fraction_is_21_over_55_l709_709049

noncomputable def nested_fraction : ℚ := 1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))))

theorem nested_fraction_is_21_over_55 : nested_fraction = 21 / 55 :=
by {
  have innermost := 3 - (1 / 3),
  have step1 := 3 - (1 / (3 - (1 / 3))),
  have step2 := 3 - (1 / (3 - (1 / (3 - (1 / 3))))),
  have step3 := 3 - (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))),
  have ans : innermost = 8 / 3 := by library_search,
  have step1_ans : step1 = 21 / 8 := by library_search,
  have step2_ans : step2 = 8 / 21 := by library_search,
  have step3_ans : step3 = 55 / 21 := by library_search,
  exact (by {
    simp only [nested_fraction, step3_ans, inv_div, mul_one, one_mul, div_div_eq_div_mul];
    norm_num
  })
}

end nested_fraction_is_21_over_55_l709_709049


namespace ratio_of_areas_l709_709082

structure Point :=
(x : ℝ) (y : ℝ)

structure Parallelogram :=
(A B C D : Point)

def midpoint (p1 p2 : Point) : Point :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2 }

def DF_ratio (D F A : Point) : Prop :=
F.y = D.y + (A.y - D.y) / 4

def area_triangle (P Q R : Point) : ℝ :=
(abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))) / 2

def area_quadrilateral (P Q R S : Point) : ℝ :=
area_triangle P Q R + area_triangle P R S

noncomputable def ratio_areas (p : Parallelogram) (E F : Point) (hE : E = midpoint p.B p.D) (hF : DF_ratio p.D F p.A) : ℝ :=
area_triangle p.D F E / area_quadrilateral p.A B E F

theorem ratio_of_areas (p : Parallelogram) (E F : Point) (hE : E = midpoint p.B p.D) (hF : DF_ratio p.D F p.A) :
  ratio_areas p E F hE hF = 1 / 11 :=
sorry

end ratio_of_areas_l709_709082


namespace num_pos_factors_of_150_mul_15_l709_709490

theorem num_pos_factors_of_150_mul_15 : 
  ∃ (count : ℕ), count = ∑ factor in ({1, 2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150} : Finset ℕ), 
  if factor % 15 = 0 then 1 else 0 ∧ count = 4 :=
by
  sorry

end num_pos_factors_of_150_mul_15_l709_709490


namespace probability_sum_7_l709_709589

-- Define the probability function
def P (k : ℕ) (hk : k ∈ {1,2,3,4,5,6}) : ℚ :=
  k / 21

-- Define the event of rolling a sum of 7
def event_sum_7 (x y : ℕ) (hx : x ∈ {1,2,3,4,5,6}) (hy : y ∈ {1,2,3,4,5,6}) : Prop :=
  x + y = 7

-- Calculate the probability of a specific pair
def pair_probability (x y : ℕ) (hx : x ∈ {1,2,3,4,5,6}) (hy : y ∈ {1,2,3,4,5,6}) : ℚ :=
  P x hx * P y hy

-- Calculate the total probability of the event_sum_7
noncomputable def total_probability_sum_7 : ℚ :=
  (pair_probability 1 6 (by simp) (by simp)) +
  (pair_probability 2 5 (by simp) (by simp)) +
  (pair_probability 3 4 (by simp) (by simp)) +
  (pair_probability 4 3 (by simp) (by simp)) +
  (pair_probability 5 2 (by simp) (by simp)) +
  (pair_probability 6 1 (by simp) (by simp))

-- Main theorem statement
theorem probability_sum_7 : total_probability_sum_7 ≈ 0.13 := sorry

end probability_sum_7_l709_709589


namespace inversion_number_partially_reversed_l709_709430

-- Definition of array and inversion number conditions
def is_distinct (s : List ℕ) : Prop :=
  s.nodup

def inversion_number (s : List ℕ) : ℕ :=
  s.enum.map (fun ⟨i, x⟩ => s.drop (i + 1).countp (fun y => x > y)).sum

-- Specification
theorem inversion_number_partially_reversed {a : List ℕ} (h₁ : a.length = 8) 
  (h₂ : a.nodup) (h₃ : inversion_number a = 2) :
  inversion_number [a.getLast!, a.get ⟨7, by simp [h₁]⟩, a.get ⟨6, by simp [h₁]⟩,
                   a.get ⟨5, by simp [h₁]⟩, a.get ⟨4, by simp [h₁]⟩,
                   a.get ⟨3, by simp [h₁]⟩, a.get ⟨2, by simp [h₁]⟩] ≥ 19 := 
by
  sorry

end inversion_number_partially_reversed_l709_709430


namespace postage_arrangements_num_ways_to_arrange_postage_l709_709775

def stamp_denominations := Set (Fin 13)  -- Denominations ranging from 1 cent to 12 cents.
def total_stamps := 2  -- Two identical stamps per denomination.
def target_sum := 15  -- The total amount of postage to be arranged.

theorem postage_arrangements (stamps : List (Fin 13)) (h : ∀ s ∈ stamps, s ≠ 0) :
  stamps.length = 3 ∨ stamps.length = 4 →
  (∀ s ∈ stamps, stamps.count s ≤ total_stamps) →
  stamps.sum = target_sum →
  @Multiset.distinct stamps = false →
  stamps.erase_all stamps.length >= 0 :=
by
  sorry

-- Number of arrangements respecting the conditions
theorem num_ways_to_arrange_postage : 
  ∃ (arrangements : Nat), arrangements = 213 :=
by
  exists 213
  sorry

end postage_arrangements_num_ways_to_arrange_postage_l709_709775


namespace pizzaCostPerSlice_l709_709539

/-- Define the constants and parameters for the problem --/
def largePizzaCost : ℝ := 10.00
def numberOfSlices : ℕ := 8
def firstToppingCost : ℝ := 2.00
def secondThirdToppingCost : ℝ := 1.00
def otherToppingCost : ℝ := 0.50
def toppings : List String := ["pepperoni", "sausage", "ham", "olives", "mushrooms", "bell peppers", "pineapple"]

/-- Calculate the total number of toppings --/
def numberOfToppings : ℕ := toppings.length

/-- Calculate the total cost of the pizza including all toppings --/
noncomputable def totalPizzaCost : ℝ :=
  largePizzaCost + 
  firstToppingCost + 
  2 * secondThirdToppingCost + 
  (numberOfToppings - 3) * otherToppingCost

/-- Calculate the cost per slice --/
noncomputable def costPerSlice : ℝ := totalPizzaCost / numberOfSlices

/-- Proof statement: The cost per slice is $2.00 --/
theorem pizzaCostPerSlice : costPerSlice = 2 := by
  sorry

end pizzaCostPerSlice_l709_709539


namespace eccentricity_of_hyperbola_l709_709867

variables (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
def hyperbola := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}
def circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}

theorem eccentricity_of_hyperbola (h_area : ∃ p1 p2 p3 p4, 
  p1 ∈ circle ∧ p2 ∈ circle ∧ p3 ∈ circle ∧ p4 ∈ circle ∧ 
  (p1.2 = b / a * p1.1 ∨ p1.2 = -b / a * p1.1) ∧ 
  (p2.2 = b / a * p2.1 ∨ p2.2 = -b / a * p2.1) ∧ 
  (p3.2 = b / a * p3.1 ∨ p3.2 = -b / a * p3.1) ∧ 
  (p4.2 = b / a * p4.1 ∨ p4.2 = -b / a * p4.1) ∧ 
  ((p1.1 - p2.1) * (p3.2 - p1.2) = ab)) : 
  (eccentricity : ℝ) :=
sorry

end eccentricity_of_hyperbola_l709_709867


namespace sandy_correct_sums_l709_709602

variables (x y : ℕ)

theorem sandy_correct_sums :
  (x + y = 30) →
  (3 * x - 2 * y = 50) →
  x = 22 :=
by
  intro h1 h2
  -- Proof will be filled in here
  sorry

end sandy_correct_sums_l709_709602


namespace complement_union_l709_709482

open Set

variable (U A B : Set ℕ)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom A_def : A = {1, 2, 3}
axiom B_def : B = {3, 4}

theorem complement_union (C_U : Set ℕ) (h : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4}) :
  C_U = U \ (A ∪ B) :=
by
  rw [h, hA, hB] -- Rewrite using the definitions of U, A, and B
  simp [compl_eq_univ_diff] -- Simplify using the complement definition
  exact rfl -- Both sides are definitionally equal
  sorry

end complement_union_l709_709482


namespace probability_two_girls_l709_709069

/--
From a group of 5 students consisting of 2 boys and 3 girls, 2 representatives 
are randomly selected (with each student having an equal chance of being selected). 
Prove that the probability that both representatives are girls is 3/10.
-/
theorem probability_two_girls (total_students boys girls : ℕ) : 
  total_students = 5 ∧ boys = 2 ∧ girls = 3 → 
  let total_outcomes := Nat.choose 5 2 in
  let favorable_outcomes := Nat.choose 3 2 in
  favorable_outcomes.toRational / total_outcomes.toRational = 3 / 10 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  unfold total_outcomes favorable_outcomes
  sorry

end probability_two_girls_l709_709069


namespace remainder_sum_l709_709907

theorem remainder_sum (n : ℤ) : ((7 - n) + (n + 3)) % 7 = 3 :=
sorry

end remainder_sum_l709_709907


namespace target_hit_probability_l709_709393

-- Define the probabilities given in the problem
def prob_A_hits : ℚ := 9 / 10
def prob_B_hits : ℚ := 8 / 9

-- The required probability that at least one hits the target
def prob_target_hit : ℚ := 89 / 90

-- Theorem stating that the probability calculated matches the expected outcome
theorem target_hit_probability :
  1 - ((1 - prob_A_hits) * (1 - prob_B_hits)) = prob_target_hit :=
by
  sorry

end target_hit_probability_l709_709393


namespace count_unique_three_digit_integers_l709_709488

-- Definitions for the problem
def digit_set : multiset ℕ := {2, 3, 5, 5, 6, 6, 6}

def is_valid_digit (n : ℕ) : Prop := n ∈ digit_set

def count_digit (n : ℕ) : ℕ := multiset.count n digit_set

def unique_three_digit_integers (s : multiset ℕ) :=
  (s.card = 3) ∧ (∀ x ∈ s, is_valid_digit x) ∧ (∀ x, s.count x ≤ count_digit x)

-- Main theorem statement
theorem count_unique_three_digit_integers : 
  (multiplicities : finset (multiset ℕ)) = finset.univ.filter unique_three_digit_integers →
  (multiplicities.card = 44) :=
by
  sorry

end count_unique_three_digit_integers_l709_709488


namespace regular_polygon_interior_angle_of_108_has_5_sides_l709_709937

theorem regular_polygon_interior_angle_of_108_has_5_sides (interior_angle : ℝ) 
  (h : interior_angle = 108) : 
  let exterior_angle := 180 - interior_angle in
  let n := 360 / exterior_angle in
  n = 5 := 
by 
  unfold exterior_angle n
  rw [h, sub_eq_add_neg, add_neg_eq_sub]
  norm_num
sorry

end regular_polygon_interior_angle_of_108_has_5_sides_l709_709937


namespace vector_magnitude_sub_eq_sqrt2_l709_709090

open Real
open ComplexConjugate

namespace VectorMath

def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def angle_cos (a b : ℝ × ℝ) :=
  ((a.1 * b.1 + a.2 * b.2) / (magnitude a * magnitude b))

noncomputable def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - 2 * b.1, a.2 - 2 * b.2)

theorem vector_magnitude_sub_eq_sqrt2
  (a b : ℝ × ℝ)
  (h1 : angle_cos a b = cos (π / 4))
  (h2 : a = (-1, 1))
  (h3 : magnitude b = 1) :
  magnitude (vector_sub a b) = sqrt 2 := by
  sorry

end VectorMath

end vector_magnitude_sub_eq_sqrt2_l709_709090


namespace distribute_balls_into_boxes_l709_709894

theorem distribute_balls_into_boxes :
  (∃ (ways : ℕ), ways = 3 ∧
  ∀ (a b : ℕ), a + b = 5 →
    (a = 5 ∧ b = 0) ∨ (a = 4 ∧ b = 1) ∨ (a = 3 ∧ b = 2)) :=
by
  use 3
  split
  case a => rfl
  case b => sorry

end distribute_balls_into_boxes_l709_709894


namespace sugar_required_in_new_recipe_l709_709281

theorem sugar_required_in_new_recipe
  (ratio_flour_water_sugar : ℕ × ℕ × ℕ)
  (double_ratio_flour_water : (ℕ → ℕ))
  (half_ratio_flour_sugar : (ℕ → ℕ))
  (new_water_cups : ℕ) :
  ratio_flour_water_sugar = (7, 2, 1) →
  double_ratio_flour_water 7 = 14 → 
  double_ratio_flour_water 2 = 4 →
  half_ratio_flour_sugar 7 = 7 →
  half_ratio_flour_sugar 1 = 2 →
  new_water_cups = 2 →
  (∃ sugar_cups : ℕ, sugar_cups = 1) :=
by
  sorry

end sugar_required_in_new_recipe_l709_709281


namespace area_OCD_invariant_l709_709547

-- Define the problem setup
variables {ω : Type} [MetricSpace ω] [Circle ω]
variables {A B C D E P F G : ω} 
variables {O : ω}

-- Assumptions and conditions
axiom diameter_AB : diameter A B = ω
axiom points_on_circle : C ∈ ω ∧ D ∈ ω
axiom perp_CD_AB : E = (CD ∩ AB) ∧ CD ⊥ AB
axiom point_P_on_CD : P ∈ CD ∧ P ≠ E
axiom intersection_AP_BP : F ∈ ω ∧ G ∈ ω ∧ (AP ∩ ω = F) ∧ (BP ∩ ω = G)
axiom circumcenter_O : O = circumcenter(E F G)

-- The goal is to prove the invariance of the area of triangle OCD
theorem area_OCD_invariant : ∀ P : ω, P ∈ CD ∧ P ≠ E → 
  area(Δ O C D) = area(Δ O C D) :=
by sorry

end area_OCD_invariant_l709_709547


namespace fg_of_2_eq_15_l709_709913

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2_eq_15 : f (g 2) = 15 :=
by
  -- The detailed proof would go here
  sorry

end fg_of_2_eq_15_l709_709913


namespace major_premise_is_false_l709_709301

-- Define the major premise
def major_premise (a : ℝ) : Prop := a^2 > 0

-- Define the minor premise
def minor_premise (a : ℝ) := true

-- Define the conclusion based on the premises
def conclusion (a : ℝ) : Prop := a^2 > 0

-- Show that the major premise is false by finding a counterexample
theorem major_premise_is_false : ¬ ∀ a : ℝ, major_premise a := by
  sorry

end major_premise_is_false_l709_709301


namespace square_area_l709_709755

theorem square_area :
  ∀ (x : ℝ), (4 * x - 15 = 20 - 3 * x) → (let edge := 4 * x - 15 in edge ^ 2 = 25) :=
by
  intros x h
  have h1 : 4 * x - 15 = 20 - 3 * x := h
  sorry

end square_area_l709_709755


namespace tangent_line_at_radius_l709_709004

theorem tangent_line_at_radius
  (d : ℝ) 
  (diameter : ℝ)
  (radius : ℝ := diameter / 2)
  (h_diameter : diameter = 10) :
  d = radius ↔ ∃ line : ℝ → ℝ, (∀ (x : ℝ), dist (center, line x) = radius) := 
begin
  sorry
end

end tangent_line_at_radius_l709_709004


namespace probability_f_ineq_l709_709864

noncomputable def f (a x : ℝ) : ℝ := log a x + log (1 / a) 8

def valid_a (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1

def possible_a : set ℝ :=
  {1/4, 1/3, 1/2, 3, 4, 5, 6, 7}

def f_ineq (a : ℝ) : Prop :=
  f a (3 * a + 1) > f a (2 * a) ∧ f a (2 * a) > 0

def valid_a_condition (a : ℝ) : Prop :=
  a > 1/2 ∧ a > 1

def count_valid_a : ℕ :=
  { a ∈ possible_a | valid_a_condition a }.to_finset.card

def count_possible_a : ℕ :=
  possible_a.to_finset.card

theorem probability_f_ineq :
  (count_valid_a.to_rat / count_possible_a.to_rat) = 3 / 8 :=
sorry

end probability_f_ineq_l709_709864


namespace find_g_of_polynomial_l709_709789

variable (x : ℝ)

theorem find_g_of_polynomial :
  ∃ g : ℝ → ℝ, (4 * x^4 + 8 * x^3 + g x = 2 * x^4 - 5 * x^3 + 7 * x + 4) → (g x = -2 * x^4 - 13 * x^3 + 7 * x + 4) :=
sorry

end find_g_of_polynomial_l709_709789


namespace find_real_numbers_l709_709054

noncomputable def find_xy (n : ℕ) (h : n > 3) : ℝ × ℝ :=
  let k := Real.sqrt ((2 * n - 7) / (8 * n + 4))
  let x := Real.sqrt (1 + (7 + 4 * k - 4 * k^2) / (4 - 8 * k)) ^ 2
  let y := Real.sqrt (1 + (7 - 4 * k - 4 * k^2) / (4 + 8 * k)) ^ 2
  (x, y)

theorem find_real_numbers (n : ℕ) (h : n > 3) :
  let (x, y) := find_xy n h
  x ≥ y ∧ y ≥ 1 ∧ Real.sqrt(x - 1) + Real.sqrt(y - 1) = n
  ∧ Real.sqrt(x + 1) + Real.sqrt(y + 1) = n + 1 :=
by
  let (x, y) := find_xy n h
  sorry

end find_real_numbers_l709_709054


namespace sqrt_algebraic_expression_l709_709168

theorem sqrt_algebraic_expression (x : ℝ) :
    let a := 2000 * x + 2015
    let b := 2000 * x + 2016
    let c := 2000 * x + 2017
    sqrt (a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a) = sqrt 3 := by
  sorry

end sqrt_algebraic_expression_l709_709168


namespace circle_diameter_segments_l709_709932

-- Define the radius of the circle
def radius : ℝ := 7

-- Define the lengths of the segments into which the diameter is divided by K
def length_segments (a b : ℝ) : Prop :=
  a = 7 - sqrt 13 ∧ b = 7 + sqrt 13

-- Problem statement proving the lengths of the segments
theorem circle_diameter_segments (CH_length : ℝ) 
  (r : ℝ) (h_radius : r = radius) (h_CH_length : CH_length = 12) 
  (CD_perpendicular_AB : True) (CH_cuts_AB_at_K : True) : 
  ∃ a b : ℝ, length_segments a b :=
by
  have h_segments : length_segments (7 - sqrt 13) (7 + sqrt 13) := sorry
  exists (7 - sqrt 13), (7 + sqrt 13), h_segments

end circle_diameter_segments_l709_709932


namespace factorial_expression_l709_709780

theorem factorial_expression (h : ℤ) (n m : ℕ) :
  (∏ i in range 1 (10 + 1), i * ∏ j in range 1 (7 + 1), j * ∏ k in range 1 (3 + 1), k) /
  (∏ l in range 1 (9 + 1), l * ∏ x in range 1 (8 + 1), x) = 7.5 :=
by
  sorry

end factorial_expression_l709_709780


namespace sum_of_cubes_formula_l709_709392

theorem sum_of_cubes_formula (n : ℕ) : 
  (∑ k in Finset.range (n + 1), k^3) = (n^2 * (n + 1)^2) / 4 := 
by
  sorry

end sum_of_cubes_formula_l709_709392


namespace parallel_under_perpendicular_to_plane_l709_709445

open Plane Line

variable (α : Plane) (a b c : Line)

-- Conditions for conclusion
axiom a_perp_alpha : a ⊥ α
axiom b_perp_alpha : b ⊥ α

theorem parallel_under_perpendicular_to_plane : a ∥ b :=
by {
  -- Mathematical proof that a ∥ b given a_perp_alpha and b_perp_alpha
  sorry
}

end parallel_under_perpendicular_to_plane_l709_709445


namespace total_replenish_cost_correct_l709_709238

noncomputable def change_in_quantity (day1 day7 : ℕ) : ℕ :=
  day1 - day7

noncomputable def cost_to_replenish (change_in_quantity price_per_unit : ℝ) : ℝ :=
  change_in_quantity * price_per_unit

noncomputable def total_replenish_cost : ℝ :=
  let baking_powder_change := change_in_quantity 12 6 in
  let flour_change := change_in_quantity (6 * 2.20462) (3.5 * 2.20462) in
  let sugar_change := change_in_quantity 20 15 in
  let chocolate_chips_change := change_in_quantity (5000 * 0.00220462) (1500 * 0.00220462) in
  let baking_powder_cost := cost_to_replenish baking_powder_change 3.00 in
  let flour_cost := cost_to_replenish flour_change 1.50 in
  let sugar_cost := cost_to_replenish sugar_change 0.50 in
  let chocolate_chips_cost := cost_to_replenish (3500 : ℝ) 0.015 in
  baking_powder_cost + flour_cost + sugar_cost + chocolate_chips_cost

theorem total_replenish_cost_correct : total_replenish_cost = 81.27 :=
by
  sorry

end total_replenish_cost_correct_l709_709238


namespace percentage_of_girls_passed_l709_709940

theorem percentage_of_girls_passed
  (total_candidates : ℕ = 2000)
  (num_girls : ℕ = 900)
  (num_boys : ℕ = total_candidates - num_girls)
  (percentage_boys_passed : ℝ = 0.30)
  (total_candidates_failed_percentage : ℝ = 0.691) :
  let num_boys_passed := percentage_boys_passed * num_boys
      total_candidates_failed := total_candidates_failed_percentage * total_candidates
      total_candidates_passed := total_candidates - total_candidates_failed
      num_girls_passed := total_candidates_passed - num_boys_passed
      percentage_girls_passed := (num_girls_passed / num_girls) * 100
  in percentage_girls_passed = 32 := 
sorry

end percentage_of_girls_passed_l709_709940


namespace sum_fib_2019_eq_fib_2021_minus_1_l709_709286

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

def sum_fib : ℕ → ℕ
| 0 => 0
| n + 1 => sum_fib n + fib (n + 1)

theorem sum_fib_2019_eq_fib_2021_minus_1 : sum_fib 2019 = fib 2021 - 1 := 
by sorry -- proof here

end sum_fib_2019_eq_fib_2021_minus_1_l709_709286


namespace unit_digit_of_power_of_two_l709_709195

theorem unit_digit_of_power_of_two (n : ℕ) :
  (2 ^ 2023) % 10 = 8 := 
by
  sorry

end unit_digit_of_power_of_two_l709_709195


namespace mutually_exclusive_complementary_l709_709184

variable (A B C D : Type) [ProbabilitySpace A]
variable {P : Set A → ℝ}

-- Conditions
axiom prob_A : P(A) = 0.2
axiom prob_B : P(B) = 0.1
axiom prob_C : P(C) = 0.3
axiom prob_D : P(D) = 0.4
axiom mutually_exclusive : P(A ∩ B) = 0 ∧ P(A ∩ C) = 0 ∧ P(A ∩ D) = 0 ∧ P(B ∩ C) = 0 ∧ P(B ∩ D) = 0 ∧ P(C ∩ D) = 0
axiom total_probability : P(A ∪ B ∪ C ∪ D) = 1

-- Proof Problem
theorem mutually_exclusive_complementary :
  P(A ∩ (B ∪ C ∪ D)) = 0 ∧ P(A ∪ (B ∪ C ∪ D)) = 1 :=
by
  sorry

end mutually_exclusive_complementary_l709_709184


namespace correct_average_wrong_reading_l709_709685

theorem correct_average_wrong_reading
  (initial_average : ℕ) (list_length : ℕ) (wrong_number : ℕ) (correct_number : ℕ) (correct_average : ℕ) 
  (h1 : initial_average = 18)
  (h2 : list_length = 10)
  (h3 : wrong_number = 26)
  (h4 : correct_number = 66)
  (h5 : correct_average = 22) :
  correct_average = ((initial_average * list_length) - wrong_number + correct_number) / list_length :=
sorry

end correct_average_wrong_reading_l709_709685


namespace regular_hexagon_fits_unit_cube_l709_709382

theorem regular_hexagon_fits_unit_cube : 
  let hexagon_side := 2 / 3,
  let cube_diagonal_half := (Real.sqrt 2) / 2 in
  hexagon_side < cube_diagonal_half :=
by
  let hexagon_side := 2 / 3
  let cube_diagonal_half := (Real.sqrt 2) / 2
  show hexagon_side < cube_diagonal_half
  sorry

end regular_hexagon_fits_unit_cube_l709_709382


namespace conjugate_of_z_is_4_plus_3i_l709_709827
noncomputable theory

def complex_number_z (z : ℂ) : Prop :=
  (1 + complex.I)/(1 - complex.I) * z = 3 + 4 * complex.I

theorem conjugate_of_z_is_4_plus_3i (z : ℂ) (h : complex_number_z z) : complex.conj z = 4 + 3 * complex.I :=
by sorry

end conjugate_of_z_is_4_plus_3i_l709_709827


namespace smallest_term_seq_l709_709421

def is_even (n : Nat) : Prop :=
  n % 2 = 0

def next_term (a_n : ℕ) : ℕ :=
if is_even a_n then a_n / 2 else a_n + 7

def sequence (a_1 : ℕ) : ℕ → ℕ 
| 0     := a_1
| (n+1) := next_term (sequence n)

theorem smallest_term_seq (a_1 : ℕ) (h : a_1 = 2014 ^ (2015 ^ 2016)) : ∃ n, sequence a_1 n = 5 := by
  sorry

end smallest_term_seq_l709_709421


namespace range_of_a_l709_709611

open Real

theorem range_of_a (e : ℝ) (m : ℝ) : 0 < m ∧ e = Real.exp 1 → ∃ a : ℝ, 
(∀ x : ℝ, x + a * (2 * x + 2 * m - 4 * e * x) * (log(x + m) - log x) = 0) 
→ (a ∈ Ioi (1 / (2 * e))) := sorry

end range_of_a_l709_709611


namespace bob_picks_3_same_color_probability_l709_709337

noncomputable def probability_bob_picks_3_same_color : ℚ :=
  let total_ways_alice_picks := (nat.choose 9 3)
  let favorable_ways_bob_picks :=
    3 * (nat.choose 3 2) * 6
  favorable_ways_bob_picks / total_ways_alice_picks

theorem bob_picks_3_same_color_probability :
  probability_bob_picks_3_same_color = 9 / 14 :=
  by
  sorry

end bob_picks_3_same_color_probability_l709_709337


namespace total_cost_l709_709964

def permit_cost : Int := 250
def contractor_hourly_rate : Int := 150
def contractor_days : Int := 3
def contractor_hours_per_day : Int := 5
def inspector_discount_rate : Float := 0.80

theorem total_cost : Int :=
  let total_hours := contractor_days * contractor_hours_per_day
  let contractor_total_cost := total_hours * contractor_hourly_rate
  let inspector_hourly_rate := contractor_hourly_rate - (inspector_discount_rate * contractor_hourly_rate)
  let inspector_total_cost := total_hours * Int.ofFloat inspector_hourly_rate
  permit_cost + contractor_total_cost + inspector_total_cost

example : total_cost = 2950 := by
  sorry

end total_cost_l709_709964


namespace digits_in_decimal_representation_l709_709400

theorem digits_in_decimal_representation (h₁ : 8 = 2^3) (h₂ : 1250 = 5^3 * 2) : 
  ∃ n : ℕ, n = 2 ∧ 
  ∃ d : ℚ, d = (2^8 * 5^2 : ℚ) / (8^3 * 1250) ∧ d = (0.05 : ℚ) :=
by
  -- Definitions and assumptions according to the conditions
  have h₃ : 8^3 = 2^9, by rw [h₁, pow_mul],
  have h₄ : 1250 = 5^3 * 2, by exact h₂,
  have h₅ : 8^3 * 1250 = 2^{10} * 5^3, by rw [h₃, h₄, mul_assoc],
  --
  -- Placeholder for the actual proof
  sorry

end digits_in_decimal_representation_l709_709400


namespace total_bananas_in_collection_l709_709614

theorem total_bananas_in_collection (g b T : ℕ) (h₀ : g = 196) (h₁ : b = 2) (h₂ : T = 392) : g * b = T :=
by
  sorry

end total_bananas_in_collection_l709_709614


namespace exists_color_with_points_distance_x_l709_709288

open Classical -- To use the law of excluded middle and some other classical logic tools.

theorem exists_color_with_points_distance_x :
  (∃ color : Prop, ∀ (x : ℝ), x > 0 → ∃ (p q : ℝ × ℝ), p ≠ q ∧ (p - q).norm = x ∧ color p ∧ color q) :=
sorry

end exists_color_with_points_distance_x_l709_709288


namespace molecular_weight_N2O5_l709_709664

variable {x : ℕ}

theorem molecular_weight_N2O5 (hx : 10 * 108 = 1080) : (108 * x = 1080 * x / 10) :=
by
  sorry

end molecular_weight_N2O5_l709_709664


namespace complex_multiplication_identity_l709_709700

-- Given definitions for the problem
def i : ℂ := complex.I

-- The main statement of the math proof problem
theorem complex_multiplication_identity : i * (2 - i) = 1 + 2 * i :=
by sorry

end complex_multiplication_identity_l709_709700


namespace contestants_order_l709_709189

variables (G E H F : ℕ) -- Scores of the participants, given that they are nonnegative

theorem contestants_order (h1 : E + G = F + H) (h2 : F + E = H + G) (h3 : G > E + F) : 
  G ≥ E ∧ G ≥ H ∧ G ≥ F ∧ E = H ∧ E ≥ F :=
by {
  sorry
}

end contestants_order_l709_709189


namespace solve_system_of_equations_l709_709255

theorem solve_system_of_equations :
  ∃ (x y z : ℝ), 
    (2 * y + x - x^2 - y^2 = 0) ∧ 
    (z - x + y - y * (x + z) = 0) ∧ 
    (-2 * y + z - y^2 - z^2 = 0) ∧ 
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 0 ∧ z = 1)) :=
by
  sorry

end solve_system_of_equations_l709_709255


namespace train_length_l709_709931

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h_speed : speed_kmph = 60) 
  (h_time : time_sec = 7.199424046076314) 
  (h_length : length_m = 120)
  : speed_kmph * (1000 / 3600) * time_sec = length_m :=
by 
  sorry

end train_length_l709_709931


namespace badminton_allocation_methods_l709_709243

theorem badminton_allocation_methods :
  let countries := ["China", "Japan", "Korea"]
  let referees_per_country := 2
  let total_referees := 3 * referees_per_country
  let courts := 3
  let pairs_per_court := total_referees / courts
  let ways_to_assign_court_pairs := 3.factorial
  let ways_to_choose_referees_for_pairs := (referees_per_country.factorial ^ courts)
  let possible_allocations := ways_to_assign_court_pairs * ways_to_choose_referees_for_pairs
  possible_allocations = 48 := sorry

end badminton_allocation_methods_l709_709243


namespace vector_AD_l709_709438

variables {A B C D : Type} [add_comm_group A] [vector_space ℚ A] 
variables (AB AC AD BD DC BC : A)

-- Conditions
variables (h1 : BD = (3 : ℚ) • DC)
variables (h2 : BC = AC - AB)
variables (h3 : AD = AB + BD)

-- Goal
theorem vector_AD : AD = (1 / 4 : ℚ) • AB + (3 / 4 : ℚ) • AC :=
sorry

end vector_AD_l709_709438


namespace chocolates_initial_count_l709_709111

theorem chocolates_initial_count : 
  ∀ (chocolates_first_day chocolates_second_day chocolates_third_day chocolates_fourth_day chocolates_fifth_day initial_chocolates : ℕ),
  chocolates_first_day = 4 →
  chocolates_second_day = 2 * chocolates_first_day - 3 →
  chocolates_third_day = chocolates_first_day - 2 →
  chocolates_fourth_day = chocolates_third_day - 1 →
  chocolates_fifth_day = 12 →
  initial_chocolates = chocolates_first_day + chocolates_second_day + chocolates_third_day + chocolates_fourth_day + chocolates_fifth_day →
  initial_chocolates = 24 :=
by {
  -- the proof will go here,
  sorry
}

end chocolates_initial_count_l709_709111


namespace minimal_polynomial_l709_709059

theorem minimal_polynomial (x : ℝ) (h₁ : x = 2 + real.sqrt 3 ∨ x = 2 + real.sqrt 5 ∨ 
                                x = 2 - real.sqrt 3 ∨ x = 2 - real.sqrt 5) :
    (x - (2 + real.sqrt 3)) * (x - (2 - real.sqrt 3)) * 
    (x - (2 + real.sqrt 5)) * (x - (2 - real.sqrt 5)) = 
    x^4 - 8*x^3 + 16*x^2 - 1 :=
sorry

end minimal_polynomial_l709_709059


namespace count_arithmetic_sequence_l709_709885

theorem count_arithmetic_sequence :
  ∃ n, 195 - (n - 1) * 3 = 12 ∧ n = 62 :=
by {
  sorry
}

end count_arithmetic_sequence_l709_709885


namespace cake_icing_l709_709009

/-- Define the cake conditions -/
structure Cake :=
  (dimension : ℕ)
  (small_cube_dimension : ℕ)
  (total_cubes : ℕ)
  (iced_faces : ℕ)

/-- Define the main theorem to prove the number of smaller cubes with icing on exactly two sides -/
theorem cake_icing (c : Cake) : 
  c.dimension = 5 ∧ c.small_cube_dimension = 1 ∧ c.total_cubes = 125 ∧ c.iced_faces = 4 →
  ∃ n, n = 20 :=
by
  sorry

end cake_icing_l709_709009


namespace some_base_value_l709_709928

noncomputable def some_base (x y : ℝ) (h1 : x * y = 1) (h2 : (some_base : ℝ) → (some_base ^ (x + y))^2 / (some_base ^ (x - y))^2 = 2401) : ℝ :=
  7

theorem some_base_value (x y : ℝ) (h1 : x * y = 1) (h2 : ∀ some_base : ℝ, (some_base ^ (x + y))^2 / (some_base ^ (x - y))^2 = 2401) : some_base x y h1 h2 = 7 :=
by
  sorry

end some_base_value_l709_709928


namespace parallelogram_height_area_l709_709724

-- Define the given conditions
variables (AB BC AE : ℝ)
variables (angleA : ℝ)

-- Specify the values based on the problem
def parallelogram_conditions : Prop :=
  AB = 18 ∧ BC = 10 ∧ AE = 2 ∧ angleA = 150

-- Define the height and area to prove
def height (EF : ℝ) : Prop :=
  EF = 2 * Real.sqrt 3

def area (A : ℝ) : Prop :=
  A = 36 * Real.sqrt 3

-- Prove the height dropped and the area
theorem parallelogram_height_area (AB BC AE angleA EF A : ℝ) 
  (cond : parallelogram_conditions AB BC AE angleA) :
  height EF ∧ area A :=
by
  sorry

end parallelogram_height_area_l709_709724


namespace sum_of_ninety_degree_angles_l709_709372

-- Definitions based on problem conditions
def ninety_degree_angles_in_rectangle := 4
def ninety_degree_angles_in_square := 4

-- Theorem statement to be proven
theorem sum_of_ninety_degree_angles : ninety_degree_angles_in_rectangle + ninety_degree_angles_in_square = 8 :=
by
  unfold ninety_degree_angles_in_rectangle
  unfold ninety_degree_angles_in_square
  exact eq.refl 8

end sum_of_ninety_degree_angles_l709_709372


namespace coordinate_sum_condition_l709_709467

open Function

theorem coordinate_sum_condition :
  (∃ (g : ℝ → ℝ), g 6 = 5 ∧
    (∃ y : ℝ, 4 * y = g (3 * 2) + 4 ∧ y = 9 / 4 ∧ 2 + y = 17 / 4)) :=
by
  sorry

end coordinate_sum_condition_l709_709467


namespace add_decimals_l709_709377

theorem add_decimals : 5.763 + 2.489 = 8.252 := 
by
  sorry

end add_decimals_l709_709377


namespace sin_alpha_eq_63_over_65_l709_709837

open Real

variables {α β : ℝ}

theorem sin_alpha_eq_63_over_65
  (h1 : tan β = 4 / 3)
  (h2 : sin (α + β) = 5 / 13)
  (h3 : 0 < α ∧ α < π)
  (h4 : 0 < β ∧ β < π) :
  sin α = 63 / 65 := 
by
  sorry

end sin_alpha_eq_63_over_65_l709_709837


namespace billboard_area_l709_709355

theorem billboard_area {L W : ℕ} (h_perimeter : 2 * L + 2 * W = 44) (h_width : W = 9) : L * 9 = 117 :=
by
  have h1 : 2 * L + 2 * 9 = 44, from (congr_arg (fun W => 2 * L + 2 * W) h_width).trans h_perimeter
  have h2 : 2 * L + 18 = 44, from (congr_arg (fun x => x + 18) (h1.trans (Eq.refl (2 * L + 18))))
  have h3 : 2 * L = 26, from (Nat.sub_eq_of_eq_add rfl h2)
  have h4 : L = 13, from Nat.eq_of_mul_eq_mul_right (by norm_num) h3
  exact Nat.mul_eq_mul_right (Eq.refl 13) (by norm_num).symm

end billboard_area_l709_709355


namespace ab2_over_bc2_eq_l709_709371

variables {Point Vector : Type} [AdditiveGroup Vector] [NormedSpace ℝ Vector]

-- Points and Vectors definition
variables (A A2 B B2 C C2 B1 : Point)
variables (VAA2 VBB2 VCC2 : Vector)

-- Conditions
def condition1 := VAA2 + VBB2 + VCC2 = (0 : Vector)
def condition2 := (‖(A, B1).vector_to_point (B1, C).vector_to_point‖ : ℝ) = 1/4

-- Problem to prove
theorem ab2_over_bc2_eq : condition1 → condition2 → (‖(A, B2).vector_to_point (B2, C2).vector_to_point‖ : ℝ) = 4/3 :=
by
  intros h1 h2
  sorry

end ab2_over_bc2_eq_l709_709371


namespace range_of_fx_in_interval_analytic_form_of_fx_l709_709839

-- Definition of even function and specific form for x >= 0
def even_function_on_R (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x = f (-x)

def specific_form_nonneg (f : ℝ → ℝ) :=
  ∀ x : ℝ, x ≥ 0 → f x = x^2 - 4 * x

-- Proof statement for Question 1: Range of f(x) in [0, 5]
theorem range_of_fx_in_interval (f : ℝ → ℝ)
  (h_even : even_function_on_R f)
  (h_form : specific_form_nonneg f) :
  set.range (λ x, f x) ∩ set.Icc 0 5 = set.Icc (-4) 5 := sorry

-- Proof statement for Question 2: Analytic form of f(x)
theorem analytic_form_of_fx (f : ℝ → ℝ)
  (h_even : even_function_on_R f)
  (h_form : specific_form_nonneg f) :
  ∀ x : ℝ, f x = x^2 - 4 * (abs x) := sorry

end range_of_fx_in_interval_analytic_form_of_fx_l709_709839


namespace zero_points_range_number_and_values_of_zero_points_l709_709862

noncomputable def f (x b : ℝ) : ℝ := 2^x * (2^x - 2) + b

theorem zero_points_range (b : ℝ) :
  (∃ x : ℝ, f x b = 0) ↔ b ∈ set.Iic 1 := sorry

theorem number_and_values_of_zero_points (b : ℝ) (h : b ∈ set.Iic 1) :
  if b = 1 then ∃! x, f x b = 0 ∧ x = 0
  else if 0 < b < 1 then 
    (∃! x1, f x1 b = 0 ∧ x1 = Real.log (1 + Real.sqrt (1 - b)) / Real.log 2) ∧
    (∃! x2, f x2 b = 0 ∧ x2 = Real.log (1 - Real.sqrt (1 - b)) / Real.log 2)
  else if b ≤ 0 then 
    ∃! x, f x b = 0 ∧ x = Real.log (1 + Real.sqrt (1 - b)) / Real.log 2
  else ¬∃ x, f x b = 0 := sorry

end zero_points_range_number_and_values_of_zero_points_l709_709862


namespace train_speed_is_72_kmph_l709_709736

-- Define the given conditions in Lean
def crossesMan (L V : ℝ) : Prop := L = 19 * V
def crossesPlatform (L V : ℝ) : Prop := L + 220 = 30 * V

-- The main theorem which states that the speed of the train is 72 km/h under given conditions
theorem train_speed_is_72_kmph (L V : ℝ) (h1 : crossesMan L V) (h2 : crossesPlatform L V) :
  (V * 3.6) = 72 := by
  -- We will provide a full proof here later
  sorry

end train_speed_is_72_kmph_l709_709736


namespace partition_inequality_l709_709102

noncomputable def p : ℕ → ℕ := sorry -- Details of p omitted as we focus on the properties given

def partition_condition (n : ℕ) :=
  p n = if n = 4 then 5 else sorry -- Providing partition info for n = 4, leave others unspecified

theorem partition_inequality (n : ℕ) (h : n > 1) :
  p(n+1) - 2 * p(n) + p(n-1) ≥ 0 :=
sorry -- Proof omitted

end partition_inequality_l709_709102


namespace polynomial_division_quotient_l709_709310

theorem polynomial_division_quotient :
  (poly_div : ∀ (x : ℝ), (9 * x^4 + 18 * x^3 + 8 * x^2 - 7 * x + 4) / (3 * x + 5) =
                    3 * x^3 + 3 * x^2 - x - 2 / 3) :=
by
  sorry

end polynomial_division_quotient_l709_709310


namespace solve_arithmetic_series_l709_709379

theorem solve_arithmetic_series : 
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 338 :=
by sorry

end solve_arithmetic_series_l709_709379


namespace find_m_n_and_coefficient_find_coefficient_x2_l709_709852

theorem find_m_n_and_coefficient (m n : ℕ) (h1 : (∑ i in range (n + 1), choose n i) = 256) 
                                 (h2 : (choose n 2) * m^2 = 112) (m_pos : 0 < m) :
                                 m = 2 ∧ n = 8 :=
by sorry

theorem find_coefficient_x2 (n : ℕ) (m : ℕ) (h1 : m = 2) (h2 : n = 8) :
                                (choose n 4) * (2 ^ 4) - (choose n 2) * (2 ^ 2) = 1008 :=
by sorry

end find_m_n_and_coefficient_find_coefficient_x2_l709_709852


namespace mutually_exclusive_but_not_complementary_l709_709046

-- Definitions for the problem conditions
inductive Card
| red | black | white | blue

inductive Person
| A | B | C | D

open Card Person

-- The statement of the proof
theorem mutually_exclusive_but_not_complementary : 
  (∃ (f : Person → Card), (f A = red) ∧ (f B ≠ red)) ∧ (∃ (f : Person → Card), (f B = red) ∧ (f A ≠ red)) :=
sorry

end mutually_exclusive_but_not_complementary_l709_709046


namespace num_terms_divisible_by_101_l709_709112

theorem num_terms_divisible_by_101 (a : ℕ → ℕ) (n : ℕ) (h_seq : ∀ n, a n = 10^n + 1) (h_n : n = 2018) :
  (finset.filter (λ n, 101 ∣ a n) (finset.range (n + 1))).card = 505 :=
sorry

end num_terms_divisible_by_101_l709_709112


namespace complex_abs_problem_l709_709222

noncomputable def my_complex_problem (z w : ℂ) :=
    (complex.abs z = 2) ∧ 
    (complex.abs w = 4) ∧ 
    (complex.abs (z + w) = 5) → 
    complex.abs ((1 / z) + (1 / w)) = (5 / 8)

theorem complex_abs_problem (z w : ℂ) :
    my_complex_problem z w :=
sorry

end complex_abs_problem_l709_709222


namespace prove_incorrect_propositions_l709_709749

noncomputable def P : Event → ℝ := sorry

def complementary (A B : Event) : Prop := A ∩ B = ∅ ∧ P(A) + P(B) = 1
def mutually_exclusive (A B : Event) : Prop := A ∩ B = ∅
def pairwise_mutually_exclusive (A B C : Event) : Prop := mutually_exclusive A B ∧ mutually_exclusive B C ∧ mutually_exclusive A C

def proposition1 : Prop := ∀ A B : Event, complementary A B → mutually_exclusive A B
def proposition2 : Prop := ∀ A B : Event, P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
def proposition3 : Prop := ∀ A B C : Event, pairwise_mutually_exclusive A B C → P(A) + P(B) + P(C) = P(A ∪ B ∪ C)
def proposition4 : Prop := ∀ A B : Event, P(A) + P(B) = 1 → complementary A B

def number_of_incorrect_propositions : ℕ := 
  if ¬ proposition1 then 1 else 0 +
  if ¬ proposition2 then 1 else 0 +
  if ¬ proposition3 then 1 else 0 +
  if ¬ proposition4 then 1 else 0

theorem prove_incorrect_propositions :
  number_of_incorrect_propositions = 1 := 
sorry

end prove_incorrect_propositions_l709_709749


namespace percent_increase_after_five_squares_near_107_4_l709_709734

def initial_side_length : ℝ := 3
def side_length_multiplier : ℝ := 1.2
def perimeter (side_length : ℝ) : ℝ := 4 * side_length

def side_length_nth_square (n : ℕ) : ℝ :=
  initial_side_length * (side_length_multiplier ^ n)

def perimeter_nth_square (n : ℕ) : ℝ :=
  perimeter (side_length_nth_square n)

def percent_increase (initial : ℝ) (final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem percent_increase_after_five_squares_near_107_4 :
  abs (percent_increase (perimeter (side_length_nth_square 0))
                        (perimeter (side_length_nth_square 4)) - 107.4) < 0.1 :=
by 
  sorry

end percent_increase_after_five_squares_near_107_4_l709_709734


namespace part1_l709_709450

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def focus1 : (ℝ × ℝ) := (-1, 0)
def focus2 : (ℝ × ℝ) := (1, 0)

def line_through (A B : ℝ × ℝ) : ℝ → ℝ := 
  if A.1 = B.1 then λ x, B.2  -- vertical line case
  else λ x, (B.2 - A.2) / (B.1 - A.1) * (x - A.1) + A.2  

def intersects (l : ℝ → ℝ) (C : ℝ → ℝ → Prop) : Prop := 
  ∃ x1 y1 x2 y2, C x1 y1 ∧ C x2 y2 ∧ l x1 = y1 ∧ l x2 = y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2)

theorem part1 (A P Q : ℝ × ℝ) (hA : A = (0, 2)) (hP : ellipse P.1 P.2) (hQ : ellipse Q.1 Q.2)
  (hl : line_through A focus2) (hintersect : intersects (line_through A focus2) ellipse)
  (hf2P : ∃ y, P = (1, y)) (hf2Q : ∃ y, Q = (1, y)) :
  2 * real.sqrt 2 + 2 * real.sqrt 2 = 4 * real.sqrt 2 :=
by
  sorry

end part1_l709_709450


namespace value_of_f_neg_five_half_add_f_zero_eq_neg_two_l709_709847

def f (x : ℝ) : ℝ :=
  if x > 0 ∧ x < 1 then 1 / x
  else if x = 0 then 0
  else if x > 1 ∨ x < -1 then f (mod x 2)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic (f : ℝ → ℝ) := ∀ x, f (x + 2) = f x

theorem value_of_f_neg_five_half_add_f_zero_eq_neg_two
  (h_period : periodic f) (h_odd : is_odd f)
  (h1 : ∀ x, 0 < x ∧ x < 1 → f x = 1 / x)
  (h2 : f 0 = 0) :
  f (-5 / 2) + f 0 = -2 := by
  sorry

end value_of_f_neg_five_half_add_f_zero_eq_neg_two_l709_709847


namespace horner_method_not_in_l709_709657

-- Definitions of the conditions from the problem
def polynomial := λ x : ℕ, 7 * x ^ 3 + 3 * x ^ 2 - 5 * x + 11
def x_val : ℕ := 23

-- The Horner's method computation sequence
def horner_step (coeffs : List ℕ) (x : ℕ) : List ℕ :=
  coeffs.scanl (λ acc c, acc * x + c) 0

-- Prove the equivalence, namely that 85169 is not part of intermediate results.
theorem horner_method_not_in (f : ℕ → ℕ) (x : ℕ) :
  let coeffs := [7, 3, -5, 11] in
  let horner_intermediate := horner_step coeffs x in
  f = (λ x, 7*x^3 + 3*x^2 - 5*x + 11) → x = 23 →
  ∀ v ∈ horner_intermediate, v ≠ 85169 :=
by
  intro coeffs
  intro horner_intermediate
  sorry

end horner_method_not_in_l709_709657


namespace distance_between_joe_and_pete_l709_709542
noncomputable theory

-- Defining the speeds of Joe and Pete, and the time they run
def joe_speed := 0.266666666667 -- km/min
def pete_speed := joe_speed / 2 -- km/min
def time := 40 -- min

-- Defining the distances they each run
def joe_distance := joe_speed * time
def pete_distance := pete_speed * time

-- The total distance is the sum of their individual distances
def total_distance := joe_distance + pete_distance

-- The goal is to prove that the total distance is 16 km
theorem distance_between_joe_and_pete : total_distance = 16 := 
by
  sorry

end distance_between_joe_and_pete_l709_709542


namespace max_a_for_inequality_l709_709103

theorem max_a_for_inequality : ∀ (x : ℝ) (a : ℝ), x > 0 → (x * exp x - a * (x + 1) ≥ log x) ↔ a ≤ 1 := 
by 
  sorry

end max_a_for_inequality_l709_709103


namespace set_A_enum_l709_709783

def A : Set ℤ := {z | ∃ x : ℕ, 6 / (x - 2) = z ∧ 6 % (x - 2) = 0}

theorem set_A_enum : A = {-3, -6, 6, 3, 2, 1} := by
  sorry

end set_A_enum_l709_709783


namespace angle_A_in_quadrilateral_l709_709794

noncomputable def degree_measure_A (A B C D : ℝ) := A

theorem angle_A_in_quadrilateral 
  (A B C D : ℝ)
  (hA : A = 3 * B)
  (hC : A = 4 * C)
  (hD : A = 6 * D)
  (sum_angles : A + B + C + D = 360) :
  degree_measure_A A B C D = 206 :=
by
  sorry

end angle_A_in_quadrilateral_l709_709794


namespace sqrt_of_4_eq_2_l709_709311

theorem sqrt_of_4_eq_2 : sqrt 4 = 2 := 
sorry

end sqrt_of_4_eq_2_l709_709311


namespace ellipse_equation_l709_709367

theorem ellipse_equation :
  ∃ a b h k : ℝ, 0 < a ∧ 0 < b ∧ h = 8 ∧ k = 5 ∧ (a, b, h, k) = (9, Real.sqrt 97, 8, 5) ∧ 
    (∀ (x y : ℝ), 
      (x, y) = (17, 5) → 
      ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ∧ 
      ((x - 8)^2 / 81 + (y - 5)^2 / 97 = 1)) ∧
    (∃ c d : ℝ, (8, 1) = (8, d) ∧ (8, 9) = (c, 9) ∧ 
      (sqrt( (8 - 8)^2 + (1 - 9)^2 ) = 8) ∧ 
      (c, d) = (8, 1) ∧
      (h = 8 ∧ k = 5) ∧ 
      c = 4 ∧ 
      b = sqrt( 97 - 16)) := sorry

end ellipse_equation_l709_709367


namespace floor_square_root_property_l709_709791

theorem floor_square_root_property (n : ℕ) (h : 0 < n) :
  (⌊2 * real.sqrt n⌋ = ⌊real.sqrt (n - 1) + real.sqrt (n + 1)⌋ + 1) ↔ (∃ m : ℕ, m > 0 ∧ n = m * m) := sorry

end floor_square_root_property_l709_709791


namespace find_number_l709_709711

theorem find_number (n x : ℝ) (hx : x = 0.8999999999999999) (h : n / x = 0.01) : n = 0.008999999999999999 := by
  sorry

end find_number_l709_709711


namespace find_k_for_minimum_value_l709_709410

theorem find_k_for_minimum_value :
  ∃ (k : ℝ), (∀ (x y : ℝ), 9 * x^2 - 6 * k * x * y + (3 * k^2 + 1) * y^2 - 6 * x - 6 * y + 7 ≥ 1)
  ∧ (∃ (x y : ℝ), 9 * x^2 - 6 * k * x * y + (3 * k^2 + 1) * y^2 - 6 * x - 6 * y + 7 = 1)
  ∧ k = 3 :=
sorry

end find_k_for_minimum_value_l709_709410


namespace solve_x_value_l709_709897
-- Import the necessary libraries

-- Define the problem and the main theorem
theorem solve_x_value (x : ℝ) (h : 3 / x^2 = x / 27) : x = 3 * Real.sqrt 3 :=
by
  sorry

end solve_x_value_l709_709897


namespace arithmetic_sequence_and_formula_l709_709989

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709989


namespace incorrect_statements_l709_709764

theorem incorrect_statements :
  ¬ ((sqrt (-4) * sqrt (-16)) = sqrt ((-4) * (-16))) ∧
  (sqrt ((-4) * (-16)) = sqrt 64) ∧
  (sqrt 64 = 8) :=
by
  sorry

end incorrect_statements_l709_709764


namespace find_multiplier_l709_709723

theorem find_multiplier (x y : ℝ) (hx : x = 0.42857142857142855) (hx_nonzero : x ≠ 0) (h_eq : (x * y) / 7 = x^2) : y = 3 :=
sorry

end find_multiplier_l709_709723


namespace order_a_c_b_l709_709439

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log 8 / Real.log 5

theorem order_a_c_b : a > c ∧ c > b := 
by {
  sorry
}

end order_a_c_b_l709_709439


namespace coefficient_of_x_squared_in_expansion_l709_709526

-- Define the general term for the binomial expansion
def binomial_term (r : ℕ) : ℤ :=
  (-1)^r * Int.ofNat (Nat.choose 6 r) * 3^(6-r)

-- Define the exponent of x for a given term r
def exponent_of_x (r : ℕ) : ℤ :=
  18 - 4 * r

-- Statement to prove the coefficient of x^2
theorem coefficient_of_x_squared_in_expansion :
  let r := 4 in
  exponent_of_x r = 2 →
  binomial_term r = 135 :=
by
  -- Prove the statement using the given conditions (the proof is omitted)
  intro h
  sorry

end coefficient_of_x_squared_in_expansion_l709_709526


namespace compound_weight_l709_709384

noncomputable def weightB : ℝ := 275
noncomputable def ratioAtoB : ℝ := 2 / 10

theorem compound_weight (weightA weightB total_weight : ℝ) 
  (h1 : ratioAtoB = 2 / 10) 
  (h2 : weightB = 275) 
  (h3 : weightA = weightB * (2 / 10)) 
  (h4 : total_weight = weightA + weightB) : 
  total_weight = 330 := 
by sorry

end compound_weight_l709_709384


namespace least_positive_integer_x_l709_709663

theorem least_positive_integer_x
  (x : ℕ)
  (h : ((2 * x) ^ 2 + 2 * 37 * (2 * x) + 37 ^ 2) % 47 = 0) :
  x = 5 :=
begin
  sorry
end

end least_positive_integer_x_l709_709663


namespace problem_1_problem_2_l709_709857

-- Define the transformation matrix M and its inverse
def matrix (k : ℝ) : Matrix 2 2 ℝ := ![![k, 1], ![0, 2]]

-- Condition: matrix M should transform parallelogram ABCD
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (-3, -1)
def D : ℝ × ℝ := (1, -1)

-- We can represent transformed points under the matrix transformation
def transform (M : Matrix 2 2 ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  ((M 0 0) * x + (M 0 1) * y, (M 1 0) * x + (M 1 1) * y)

def A' (k : ℝ) : ℝ × ℝ := transform (matrix k) A
def B' (k : ℝ) : ℝ × ℝ := transform (matrix k) B
def C' (k : ℝ) : ℝ × ℝ := transform (matrix k) C
def D' (k : ℝ) : ℝ × ℝ := transform (matrix k) D

-- Questions:
-- 1. Prove that if the quadrilateral A'B'C'D' is a rhombus, then k = -1.
theorem problem_1 (k : ℝ) (h_neg : k < 0) (h_rhombus : (|A' k - B' k| = |B' k - C' k|)) : k = -1 := sorry

-- 2. Prove the inverse of the given matrix.
theorem problem_2 (k : ℝ) (h_neg : k < 0) (h_k : k = -1) : 
  inverse (matrix k) = ![![(-1 : ℝ), (1/2)], ![0, (1/2)]] := sorry

end problem_1_problem_2_l709_709857


namespace probability_both_pick_red_l709_709193

theorem probability_both_pick_red (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) 
  (p_xiaojun_red : ℚ) (p_xiaojing_red_after : ℚ) 
  (total_balls = 5) (red_balls = 2) (yellow_balls = 3) :
  p_xiaojun_red = red_balls / total_balls ∧ 
  p_xiaojing_red_after = (red_balls - 1) / (total_balls - 1) ∧ 
  ∃ p_both_red, p_both_red = p_xiaojun_red * p_xiaojing_red_after := by
  sorry

end probability_both_pick_red_l709_709193


namespace season_cost_l709_709968

def first_half_cost (episodes_first_half : ℕ) (cost_per_episode_first_half : ℕ) : ℕ :=
  episodes_first_half * cost_per_episode_first_half

def second_half_cost (episodes_second_half : ℕ) (cost_per_episode_second_half : ℕ) : ℕ :=
  episodes_second_half * cost_per_episode_second_half

theorem season_cost (episodes_total : ℕ) (cost_per_episode_first_half : ℕ)
                    (increase_rate : ℚ) (episodes_first_half episodes_second_half : ℕ) :
  episodes_total = episodes_first_half + episodes_second_half →
  2 * episodes_first_half = episodes_total →
  let cost_per_episode_second_half := cost_per_episode_first_half + 
                                        (increase_rate * cost_per_episode_first_half).to_nat in
  first_half_cost episodes_first_half cost_per_episode_first_half +
  second_half_cost episodes_second_half cost_per_episode_second_half = 35200 :=
by sorry

end season_cost_l709_709968


namespace wonder_nominal_GDP_2009_wonder_nominal_GDP_2015_wonder_real_GDP_2015_wonder_growth_rate_real_GDP_wonder_CPI_2015_l709_709572

noncomputable theory

def nominal_GDP_2009 : ℕ := 5 * 12 + 7 * 8 + 9 * 6
def nominal_GDP_2015 : ℕ := 6 * 15 + 5 * 10 + 10 * 2
def real_GDP_2015 : ℕ := 5 * 15 + 7 * 10 + 9 * 2

def growth_rate_real_GDP : ℚ :=
  (163 - 170) / 170

def CPI_2015 : ℚ :=
  ( (12 * 6 + 8 * 5 + 6 * 10) / (12 * 5 + 8 * 7 + 6 * 9)) * 100

theorem wonder_nominal_GDP_2009:
  nominal_GDP_2009 = 170 := by
  sorry

theorem wonder_nominal_GDP_2015:
  nominal_GDP_2015 = 160 := by
  sorry

theorem wonder_real_GDP_2015:
  real_GDP_2015 = 163 := by
  sorry

theorem wonder_growth_rate_real_GDP:
  growth_rate_real_GDP = -0.0412 := by
  -- 4.12% as a decimal
  sorry

theorem wonder_CPI_2015:
  CPI_2015 = 101.17 := by
  sorry

end wonder_nominal_GDP_2009_wonder_nominal_GDP_2015_wonder_real_GDP_2015_wonder_growth_rate_real_GDP_wonder_CPI_2015_l709_709572


namespace factorization_identity_l709_709402

theorem factorization_identity (a : ℝ) : (a + 3) * (a - 7) + 25 = (a - 2) ^ 2 :=
by
  sorry

end factorization_identity_l709_709402


namespace molecular_weight_of_oxygen_part_l709_709417

-- Define the known variables as constants
def atomic_weight_oxygen : ℝ := 16.00
def num_oxygen_atoms : ℕ := 2
def molecular_weight_compound : ℝ := 88.00

-- Define the problem as a theorem
theorem molecular_weight_of_oxygen_part :
  16.00 * 2 = 32.00 :=
by
  -- The proof will be filled in here
  sorry

end molecular_weight_of_oxygen_part_l709_709417


namespace max_value_cos_sin_l709_709849

-- Given constants with their conditions
variables (a b : ℝ)
-- Maximum value condition
hypothesis h1 : |a| + b = 1
-- Minimum value condition
hypothesis h2 : -|a| + b = -7

-- Prove the maximum value of a cosine x plus b sine x
theorem max_value_cos_sin : (∃ x, |a * cos x + b * sin x| = 5) :=
by
  -- reasoning and simplifications will be applied here
  sorry

end max_value_cos_sin_l709_709849


namespace relationship_between_x_y_z_l709_709825

noncomputable def x := Real.sqrt 0.82
noncomputable def y := Real.sin 1
noncomputable def z := Real.log 7 / Real.log 3

theorem relationship_between_x_y_z : y < z ∧ z < x := 
by sorry

end relationship_between_x_y_z_l709_709825


namespace recommended_calorie_intake_l709_709714

-- Conditions from problem
def total_calories : ℕ := 40
def fraction_eaten : ℚ := 3 / 4
def excess_calories : ℕ := 5
def calories_eaten : ℤ := (fraction_eaten * total_calories : ℚ).to_rat -- Note : ℚ.to_rat converts ℚ to ℤ

-- Theorem statement
theorem recommended_calorie_intake : 
  (calories_eaten - excess_calories : ℤ) = 25 :=
  by sorry

end recommended_calorie_intake_l709_709714


namespace range_of_m_for_extremum_l709_709506

-- Define the function and the condition
def has_extremum (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, deriv f x = 0

-- The function y = e^x + m * x
def y (m : ℝ) : ℝ → ℝ := λ x, Real.exp x + m * x

-- Statement to prove
theorem range_of_m_for_extremum (m : ℝ) : has_extremum (y m) ↔ m < 0 := by
  sorry

end range_of_m_for_extremum_l709_709506


namespace infinite_triplets_gcd_one_l709_709598

theorem infinite_triplets_gcd_one :
  ∀ (m n : ℕ), m > 0 → n > 0 →
  let a := m^2 + m * n + n^2,
      b := m^2 - m * n,
      c := n^2 - m * n
  in
  Nat.gcd a (Nat.gcd b c) = 1 ∧ a^2 = b^2 + c^2 + b * c := by
 sorry

end infinite_triplets_gcd_one_l709_709598


namespace cost_of_soap_per_year_l709_709578

-- Conditions:
def duration_of_soap (bar: Nat) : Nat := 2
def cost_per_bar (bar: Nat) : Real := 8.0
def months_in_year : Nat := 12

-- Derived quantity
def bars_needed (months: Nat) (duration: Nat): Nat := months / duration

-- Theorem statement:
theorem cost_of_soap_per_year : 
  let n := bars_needed months_in_year (duration_of_soap 1)
  n * (cost_per_bar 1) = 48.0 := 
  by 
    -- Skipping proof
    sorry

end cost_of_soap_per_year_l709_709578


namespace triangle_bc_length_l709_709200

theorem triangle_bc_length
  (A B C : Type)
  [InnerProductSpace ℝ A]
  (a b c : A)
  (AB AC BC : A)
  (h1 : dist A B = 3)
  (h2 : dist A C = 4)
  (h3 : ∠ B A C = π / 2)
  (h4 : dist B C = dist ((1/2 : ℝ) • B + (1/2 : ℝ) • C) A) :
  dist B C = 5 :=
by
  sorry

end triangle_bc_length_l709_709200


namespace pure_imaginary_solution_l709_709501

-- Given conditions and problem statement
variable (a : ℝ)

-- Main theorem statement 
theorem pure_imaginary_solution (h : (a - complex.i) * (1 + complex.i) = (0 : ℝ) + (b : ℂ).im * complex.i) : a = -1 :=
by
  sorry

end pure_imaginary_solution_l709_709501


namespace common_difference_of_arithmetic_sequence_l709_709918

theorem common_difference_of_arithmetic_sequence :
  ∀ (a1 a5 : ℝ), a1 = 25 ∧ a5 = 105 →
  ∃ d : ℝ, ∃ a2 a3 a4 : ℝ, a2 = a1 + d ∧ a3 = a1 + 2*d ∧ a4 = a1 + 3*d ∧ a5 = a1 + 4*d ∧ d = 20 :=
begin
  intros a1 a5 h,
  cases h with h1 h5,
  existsi (105 - 25) / 4,
  existsi (25 + (105 - 25) / 4),
  existsi (25 + 2 * (105 - 25) / 4),
  existsi (25 + 3 * (105 - 25) / 4),
  split, exact h1,
  split, exact h5,
  split, refl,
  split, refl,
  split, refl,
  refl,
end

end common_difference_of_arithmetic_sequence_l709_709918


namespace find_original_numbers_l709_709636

theorem find_original_numbers (x y : ℕ) (hx : x + y = 2022) 
  (hy : (x - 5) / 10 + 10 * y + 1 = 2252) : x = 1815 ∧ y = 207 :=
by sorry

end find_original_numbers_l709_709636


namespace find_m_l709_709486

def vector_a (m : ℝ) : (ℝ × ℝ) := (m, 3)
def vector_b : (ℝ × ℝ) := (Real.sqrt 3, 1)
def angle_between_vectors := Real.pi / 6 -- 30 degrees in radians

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def cos_theta (m : ℝ) : ℝ :=
  dot_product (vector_a m) vector_b / (magnitude (vector_a m) * magnitude vector_b)

-- Theorem statement
theorem find_m (m : ℝ) : cos_theta m = Real.sqrt 3 / 2 → m = Real.sqrt 3 :=
by
  sorry

end find_m_l709_709486


namespace find_sin_beta_l709_709819

variables {α β : ℝ}

theorem find_sin_beta (
  h1 : π / 2 < α ∧ α < π,
  h2 : 0 < β ∧ β < π / 2,
  h3 : Real.tan α = -3 / 4,
  h4 : Real.cos (β - α) = 5 / 13
) : Real.sin β = 63 / 65 :=
sorry

end find_sin_beta_l709_709819


namespace b_arithmetic_a_formula_l709_709999

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l709_709999


namespace digits_of_2_pow_100_l709_709824

theorem digits_of_2_pow_100 (N : ℕ) (hN : N = Nat.log10 (2^100) + 1) : 
  ∃ k : ℕ, k = 29 ∧ k ≤ N ∧ N ≤ k + 5 :=
by
  use 29
  split
  · rfl
  · sorry

end digits_of_2_pow_100_l709_709824


namespace one_triple_consists_of_integers_l709_709704

theorem one_triple_consists_of_integers 
  (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ)
  (H : ∀ m n : ℤ, (x₁ * m + y₁ * n + z₁).even ∨ (x₂ * m + y₂ * n + z₂).even) :
  (∃ a b c : ℤ, x₁ = a ∧ y₁ = b ∧ z₁ = c) ∨ (∃ a b c : ℤ, x₂ = a ∧ y₂ = b ∧ z₂ = c) :=
by sorry

end one_triple_consists_of_integers_l709_709704


namespace even_function_iff_a_eq_2_l709_709150

noncomputable def y (a x : ℝ) : ℝ :=
  (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_iff_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, y a x = y a (-x)) ↔ a = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end even_function_iff_a_eq_2_l709_709150


namespace sum_of_digits_of_palindrome_sum_l709_709388

def is_palindrome (n : ℕ) := 
  let digits := Int.toDigits 10 n in
  digits = digits.reverse

noncomputable def sum_digits (n : ℕ) : ℕ := 
  (Int.toDigits 10 n).sum

theorem sum_of_digits_of_palindrome_sum : 
  let palindromes := {n | is_palindrome n ∧ n >= 100000 ∧ n < 1000000 ∧ (n % 10000) / 1000 = 5} in
  sum_digits (palindromes.toFinset.sum id) = 18 :=
sorry

end sum_of_digits_of_palindrome_sum_l709_709388


namespace maria_total_earnings_l709_709581

-- Definitions of the conditions
def day1_tulips := 30
def day1_roses := 20
def day2_tulips := 2 * day1_tulips
def day2_roses := 2 * day1_roses
def day3_tulips := day2_tulips / 10
def day3_roses := 16
def tulip_price := 2
def rose_price := 3

-- Definition of the total earnings calculation
noncomputable def total_earnings : ℤ :=
  let total_tulips := day1_tulips + day2_tulips + day3_tulips
  let total_roses := day1_roses + day2_roses + day3_roses
  (total_tulips * tulip_price) + (total_roses * rose_price)

-- The proof statement
theorem maria_total_earnings : total_earnings = 420 := by
  sorry

end maria_total_earnings_l709_709581


namespace no_good_subset_with_405_elements_l709_709022

def is_good_subset (s : Set ℕ) : Prop :=
  ∀ x ∈ s, ((s.erase x).sum id) % 10 = x % 10

example : ∃ s : Finset ℕ, s.card = 400 ∧ is_good_subset (s : Set ℕ) :=
by {
  -- We can construct the specific subset as described in the solution.
  let s₀ := (Finset.range 201).map ⟨λ n, 10 * (n + 1), sorry⟩,
  let s₅ := (Finset.range 200).map ⟨λ n, 5 + 10 * n, sorry⟩,
  let s := s₀ ∪ s₅,
  use s,
  sorry  -- Further proof required to show s is good and has 400 elements.
}

theorem no_good_subset_with_405_elements : ¬ ∃ s : Finset ℕ, s.card = 405 ∧ is_good_subset (s : Set ℕ) :=
by {
  -- Use the reasoning given in the solution to show this is impossible.
  sorry
}

end no_good_subset_with_405_elements_l709_709022


namespace solve_for_h_l709_709790

theorem solve_for_h (h : ℝ[X]) :
  12 * X ^ 4 + 5 * X ^ 3 + h = -3 * X ^ 3 + 4 * X ^ 2 - 7 * X + 2 →
  h = -12 * X ^ 4 - 8 * X ^ 3 + 4 * X ^ 2 - 7 * X + 2 :=
by
  intro hyp
  sorry

end solve_for_h_l709_709790


namespace radius_of_semi_circle_l709_709013

variable (r w l : ℝ)

def rectangle_inscribed_semi_circle (w l : ℝ) := 
  l = 3*w ∧ 
  2*l + 2*w = 126 ∧ 
  (∃ r, l = 2*r)

theorem radius_of_semi_circle :
  (∃ w l r, rectangle_inscribed_semi_circle w l ∧ l = 2*r) → r = 23.625 :=
by
  sorry

end radius_of_semi_circle_l709_709013


namespace factorize_expression_l709_709051

theorem factorize_expression (a b : ℝ) : ab^2 - 2ab + a = a * (b-1)^2 := 
sorry

end factorize_expression_l709_709051


namespace solve_for_x_l709_709314

-- declare an existential quantifier to encapsulate the condition and the answer.
theorem solve_for_x : ∃ x : ℝ, x + (x + 2) + (x + 4) = 24 ∧ x = 6 := 
by 
  -- begin sorry to skip the proof part
  sorry

end solve_for_x_l709_709314


namespace range_of_m_l709_709808

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, (y = (m - 2) * x + m - 1 → (x ≥ 0 ∨ y ≥ 0))) ↔ (1 ≤ m ∧ m < 2) :=
by sorry

end range_of_m_l709_709808


namespace find_n_gadgets_l709_709935

-- Definitions reflecting conditions
def production_rate_gizmos_3_hours := 450 / (150 * 3) -- gizmos per hour per worker
def production_rate_gadgets_3_hours := 600 / (150 * 3) -- gadgets per hour per worker
def production_rate_gizmos_4_hours := 400 / (100 * 4) -- gizmos per hour per worker
def production_rate_gadgets_4_hours := 800 / (100 * 4) -- gadgets per hour per worker

-- Definitions for worker count and hours
def workers_3_hours := 150
def hours_3 := 3
def gizmos_3_hours := 450
def gadgets_3_hours := 600

def workers_4_hours := 100
def hours_4 := 4
def gizmos_4_hours := 400
def gadgets_4_hours := 800

def workers_5_hours := 75
def hours_5 := 5
def expected_gizmos_5_hours := 225

-- Main statement to prove
theorem find_n_gadgets :
  let gizmo_rate := production_rate_gizmos_4_hours in
  let gadget_rate := production_rate_gadgets_4_hours in
  (workers_5_hours * hours_5 * gadget_rate = 750) := sorry

end find_n_gadgets_l709_709935


namespace range_of_a_l709_709476

noncomputable def f (x a : ℝ) := x^2 + 2 * x - a
noncomputable def g (x : ℝ) := 2 * x + 2 * Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2, (1/e) ≤ x1 ∧ x1 < x2 ∧ x2 ≤ e ∧ f x1 a = g x1 ∧ f x2 a = g x2) ↔ 
  1 < a ∧ a ≤ (1/(e^2)) + 2 := 
sorry

end range_of_a_l709_709476


namespace baseball_games_played_l709_709002
-- Import necessary libraries

-- Define the conditions and state the main theorem
theorem baseball_games_played (P : ℕ) (L : ℕ) (h1 : P = 5 + L) (h2 : P = 2 * L) : P = 10 :=
by 
  sorry

end baseball_games_played_l709_709002


namespace mode_of_dataset_is_2_l709_709527

def dataset := [2, 3, 2, 2, 2, 5, 4]

theorem mode_of_dataset_is_2 : multiset.mode (dataset : multiset ℕ) = 2 := 
by
  -- proof goes here
  sorry

end mode_of_dataset_is_2_l709_709527


namespace sum_remainder_l709_709496

theorem sum_remainder (n : ℤ) : ((9 - n) + (n + 4)) % 9 = 4 := 
by 
  sorry

end sum_remainder_l709_709496


namespace sin_plus_cos_gt_one_iff_first_quadrant_l709_709565

variable (α : ℝ)

def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < π / 2

theorem sin_plus_cos_gt_one_iff_first_quadrant :
  (sin α + cos α > 1) ↔ is_first_quadrant α :=
sorry

end sin_plus_cos_gt_one_iff_first_quadrant_l709_709565


namespace tommy_total_balloons_l709_709650

-- Define the conditions from part (a)
def original_balloons : Nat := 26
def additional_balloons : Nat := 34

-- Define the proof problem from part (c)
theorem tommy_total_balloons : original_balloons + additional_balloons = 60 := by
  -- Skip the actual proof
  sorry

end tommy_total_balloons_l709_709650


namespace support_staff_bonus_petty_cash_l709_709514

theorem support_staff_bonus_petty_cash :
  ∀ (num_staff : ℕ)
    (num_admin : ℕ)
    (num_junior : ℕ)
    (num_support : ℕ)
    (daily_bonus_admin : ℕ)
    (daily_bonus_junior : ℕ)
    (daily_bonus_support : ℕ)
    (days : ℕ)
    (given_amount : ℕ)
    (petty_cash_budget : ℕ),
    num_staff = 30 →
    num_admin = 10 →
    num_junior = 10 →
    num_support = 10 →
    daily_bonus_admin = 100 →
    daily_bonus_junior = 120 →
    daily_bonus_support = 80 →
    days = 30 →
    given_amount = 85000 →
    petty_cash_budget = 25000 →
    10 * 80 * 30 - (90000 - 85000) = 5000 :=
begin
  intros num_staff num_admin num_junior num_support daily_bonus_admin daily_bonus_junior daily_bonus_support days given_amount petty_cash_budget,
  sorry
end

end support_staff_bonus_petty_cash_l709_709514


namespace case_a_not_partitionable_case_b_partitionable_l709_709645

-- Define the sum of the first n natural numbers
def sum_nat (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the proposition for partitionable sums
def partitionable (s : Set ℕ) : Prop :=
  ∃ (s1 s2 : Set ℕ), s1 ∪ s2 = s ∧ s1 ∩ s2 = ∅ ∧ sum_nat s1.card = sum_nat s2.card

theorem case_a_not_partitionable : ¬ partitionable {1, 2, ..., 30} := by
  sorry
  
theorem case_b_partitionable : partitionable {1, 2, ..., 31} := by
  sorry

end case_a_not_partitionable_case_b_partitionable_l709_709645


namespace election_valid_votes_l709_709328

-- Define the given conditions
def total_votes : ℕ := 560000
def invalid_vote_percentage : ℝ := 0.15
def valid_vote_percentage : ℝ := 1 - invalid_vote_percentage
def candidate_a_vote_percentage : ℝ := 0.55

-- The expected number of valid votes polled in favor of candidate A
def expected_valid_votes_for_candidate_a : ℕ := 261800

-- The theorem statement
theorem election_valid_votes :
  let valid_votes := (total_votes : ℝ) * valid_vote_percentage in
  let valid_votes_for_candidate_a := valid_votes * candidate_a_vote_percentage in
  valid_votes_for_candidate_a = (expected_valid_votes_for_candidate_a : ℝ) :=
by sorry

end election_valid_votes_l709_709328


namespace division_of_polynomial_l709_709763

theorem division_of_polynomial (a : ℤ) : (-28 * a^3) / (7 * a) = -4 * a^2 := by
  sorry

end division_of_polynomial_l709_709763


namespace no_possible_values_of_m_l709_709959

-- Define the right triangle and its properties
structure RightTriangle :=
  (a b : ℝ) -- coordinates of legs along x and y axes respectively
  (a_pos : a > 0)
  (b_pos : b > 0)
  (m : ℝ)  -- slope of the line parallel to the hypotenuse
  (hypotenuse_parallel : m = -b / a)
  (median_hypotenuse_line : ∃ k : ℕ, k = 2 ∧ -2 * b / a = 2)  -- median to hypotenuse aligns with y = 2x + 1
  (median_leg_line : ∃ l : ℕ, l = 5 ∧ -b / (2 * a) = 5) -- median to leg aligns with y = 5x + 2

theorem no_possible_values_of_m (Δ : RightTriangle) : Δ.median_hypotenuse_line = False ∧ Δ.median_leg_line = False :=
  sorry

end no_possible_values_of_m_l709_709959


namespace unique_polynomial_exists_l709_709562

noncomputable theory

open Polynomial

theorem unique_polynomial_exists 
  (m : ℕ → Polynomial ℤ) (a : ℕ → Polynomial ℤ)
  (n : ℕ) 
  (pairwise_coprime : ∀ i j, i < n → j < n → i ≠ j → (m i).isCoprime (m j)) :
  ∃! p : Polynomial ℤ, 
    (∀ i, i < n → p % m i = a i % m i) ∧ 
    (p.degree < (Finset.range n).sum (λ i, (m i).degree)) := 
sorry

end unique_polynomial_exists_l709_709562


namespace probability_both_pick_red_l709_709194

theorem probability_both_pick_red (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) 
  (p_xiaojun_red : ℚ) (p_xiaojing_red_after : ℚ) 
  (total_balls = 5) (red_balls = 2) (yellow_balls = 3) :
  p_xiaojun_red = red_balls / total_balls ∧ 
  p_xiaojing_red_after = (red_balls - 1) / (total_balls - 1) ∧ 
  ∃ p_both_red, p_both_red = p_xiaojun_red * p_xiaojing_red_after := by
  sorry

end probability_both_pick_red_l709_709194


namespace magician_earnings_l709_709722

theorem magician_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (decks_remaining : ℕ) (money_earned : ℕ) : 
    price_per_deck = 7 →
    initial_decks = 16 →
    decks_remaining = 8 →
    money_earned = (initial_decks - decks_remaining) * price_per_deck →
    money_earned = 56 :=
by
  intros hp hi hd he
  rw [hp, hi, hd] at he
  exact he

end magician_earnings_l709_709722


namespace monotonic_increasing_interval_l709_709829

noncomputable def f (x : ℝ) : ℝ := sorry

theorem monotonic_increasing_interval :
  (∀ x Δx : ℝ, 0 < x → 0 < Δx → 
  (f (x + Δx) - f x) / Δx = (2 / (Real.sqrt (x + Δx) + Real.sqrt x)) - (1 / (x^2 + x * Δx))) →
  ∀ x : ℝ, 1 < x → (∃ ε > 0, ∀ y, x < y ∧ y < x + ε → f y > f x) :=
by
  intro hyp
  sorry

end monotonic_increasing_interval_l709_709829


namespace Frank_seeds_per_orange_l709_709816

noncomputable def Betty_oranges := 15
noncomputable def Bill_oranges := 12
noncomputable def total_oranges := Betty_oranges + Bill_oranges
noncomputable def Frank_oranges := 3 * total_oranges
noncomputable def oranges_per_tree := 5
noncomputable def Philip_oranges := 810
noncomputable def number_of_trees := Philip_oranges / oranges_per_tree
noncomputable def seeds_per_orange := number_of_trees / Frank_oranges

theorem Frank_seeds_per_orange :
  seeds_per_orange = 2 :=
by
  sorry

end Frank_seeds_per_orange_l709_709816


namespace coeff_of_x5_in_expansion_l709_709464

theorem coeff_of_x5_in_expansion (a : ℝ) :
  let coeff := (binom 8 5 : ℝ) - a * (binom 8 4 : ℝ)
  in coeff = -84 → a = 2 :=
by
  have h_binom_8_5 : (binom 8 5 : ℝ) = 56 := by norm_num
  have h_binom_8_4 : (binom 8 4 : ℝ) = 70 := by norm_num
  sorry

end coeff_of_x5_in_expansion_l709_709464


namespace length_of_BD_l709_709532

-- Given conditions
variables {A B C D E : Point}
variables (triangle_ABC : Triangle A B C)
variables (is_isosceles : AB = AC)
variables (midpoint_D : is_midpoint D B C)
variables (perpendicular_DE_AC : DE ⊥ AC)
variables (CE_len : CE = 15)

-- Statement to prove
theorem length_of_BD :
  length BD = 7.5 :=
sorry

end length_of_BD_l709_709532


namespace probability_of_two_mathematicians_living_contemporarily_l709_709298

noncomputable def probability_of_contemporary_lifespan : ℚ :=
  let total_area := 500 * 500
  let triangle_area := 0.5 * 380 * 380
  let non_overlap_area := 2 * triangle_area
  let overlap_area := total_area - non_overlap_area
  overlap_area / total_area

theorem probability_of_two_mathematicians_living_contemporarily :
  probability_of_contemporary_lifespan = 2232 / 5000 :=
by
  -- The actual proof would go here
  sorry

end probability_of_two_mathematicians_living_contemporarily_l709_709298


namespace angle_MAN_l709_709952

theorem angle_MAN (A B C L M N : Type) [Inhabited A] 
  (angle_BAC : Float) (h1 : angle_BAC = 50)
  (h2 : ∃ (AL : A) (between_BC : A → Prop), AL = angle_bisector A B C ∧ between_BC L)
  (h3 : ∃ (MA_eq_ML : Float) (NA_eq_NL : Float),
      ∃ (AM : A) (AN : A) (k : Inhabited A), (eq AM AN = angle_bisector A B C) ∧ 
      (∃ (M : A), AM = AL ∧ MA_eq_ML = 1.0) ∧ (∃ (N : A), AN = AL ∧ NA_eq_NL = 1.0)) :
  ∃ (angle_MAN : Float), angle_MAN = 65 := 
  sorry

end angle_MAN_l709_709952


namespace CauchySchwarz_l709_709600

theorem CauchySchwarz' (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 := by
  sorry

end CauchySchwarz_l709_709600


namespace projection_calculation_l709_709484

open Real EuclideanGeometry 

variables (a b: Vector3) (h_angle: angle a b = π/3) (h_norm_a: ‖a‖ = 2) (h_norm_b: ‖b‖ = 6)

theorem projection_calculation :
  ∥a∥ = 2 ∧ ∥b∥ = 6 ∧ angle a b = π/3 → 
  (a • (2 • a - b) / ∥a∥ = 1) :=
by
  sorry

end projection_calculation_l709_709484


namespace determine_a_l709_709158

def y (x : ℝ) (a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem determine_a (a : ℝ) : is_even (y _ a) → a = 2 :=
sorry

end determine_a_l709_709158


namespace product_of_values_l709_709802

theorem product_of_values :
  let x_vals := (λ x : ℝ, ∣(16 / x^2) - 4 ∣ = 3) in
  (∀ x : ℝ, x_vals x → x = sqrt (16 / 7) ∨ x = -sqrt (16 / 7) ∨ x = 4 ∨ x = -4) ∧
  (∀ (x1 x2 x3 x4 : ℝ), 
    x1 = sqrt (16 / 7) → x2 = -sqrt (16 / 7) → x3 = 4 → x4 = -4 →
    (x1 * x2 * x3 * x4) = -256 / 7) :=
by
  sorry

end product_of_values_l709_709802


namespace solve_for_y_l709_709133

theorem solve_for_y (y : ℝ) (h_pos : y > 0) (h_eq : y^2 = 1024) : y = 32 := 
by
  sorry

end solve_for_y_l709_709133


namespace pizza_cost_per_slice_l709_709541

theorem pizza_cost_per_slice :
  let pizza_cost := 10
  let first_topping_cost := 2
  let next_two_toppings_cost := 2
  let remaining_toppings_cost := 2
  let total_cost := pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  let slices := 8
  total_cost / slices = 2 := by
  let pizza_cost := 10
  let first_topping_cost := 2
  let next_two_toppings_cost := 2
  let remaining_toppings_cost := 2
  let total_cost := pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  let slices := 8
  have h : total_cost = 16 := by
    -- calculations to show total_cost = 16 can be provided here
    sorry
  have hslices : slices = 8 := rfl
  calc
    total_cost / slices = 16 / 8 : by rw [h, hslices]
                  ... = 2         : by norm_num

end pizza_cost_per_slice_l709_709541


namespace remainder_cd_42_l709_709965

theorem remainder_cd_42 (c d : ℕ) (p q : ℕ) (hc : c = 84 * p + 76) (hd : d = 126 * q + 117) : 
  (c + d) % 42 = 25 :=
by
  sorry

end remainder_cd_42_l709_709965


namespace time_to_drive_to_work_l709_709769

theorem time_to_drive_to_work {D : ℝ} (h1 : ∃ T : ℝ, D = 40 * T)
  (h2 : ∃ T : ℝ, T + 1.25 = (0.80 * D) / 14) : 
  ∃ T : ℝ, T ≈ 58.33 / 60 :=
by 
  -- proof steps go here
  sorry

end time_to_drive_to_work_l709_709769


namespace farmer_revenue_correct_l709_709717

-- Define the conditions
def average_bacon : ℕ := 20
def price_per_pound : ℕ := 6
def size_factor : ℕ := 1 / 2

-- Calculate the bacon from the runt pig
def bacon_from_runt := average_bacon * size_factor

-- Calculate the revenue from selling the bacon
def revenue := bacon_from_runt * price_per_pound

-- Lean 4 Statement to prove
theorem farmer_revenue_correct :
  revenue = 60 :=
sorry

end farmer_revenue_correct_l709_709717


namespace range_of_a_l709_709870

theorem range_of_a (a : ℝ) : (∃ (x : ℤ), x > 1 ∧ x ≤ a) → ∃ (x : ℤ), (x = 2 ∨ x = 3 ∨ x = 4) ∧ 4 ≤ a ∧ a < 5 :=
by
  sorry

end range_of_a_l709_709870


namespace rectangle_perimeter_is_36_l709_709011

theorem rectangle_perimeter_is_36 (a b : ℕ) (h : a ≠ b) (h1 : a * b = 2 * (2 * a + 2 * b) - 8) : 2 * (a + b) = 36 :=
  sorry

end rectangle_perimeter_is_36_l709_709011


namespace proof_correct_hyperbola_params_l709_709842

noncomputable def hyperbola_params (a b : ℝ) (F1 F2 : ℝ × ℝ) :=
  ∃ (P : ℝ × ℝ), (a > 0 ∧ b > 0) ∧
  (|PF2 - P| = a ∧ (PF1 - P) ⟂ (PF2 - P)) ∧
  (|PF1 - P| = 3 * a) ∧
  (let e := (c/a) in e = (sqrt(10) / 2)) ∧
  (∀ x, x ≠ 0 → (y = - (sqrt(6) / 2) * x) is an asymptote of the hyperbola)

theorem proof_correct_hyperbola_params :
  ∀ (a b : ℝ) (F1 F2 : ℝ × ℝ), hyperbola_params a b F1 F2 := sorry

end proof_correct_hyperbola_params_l709_709842


namespace three_digit_numbers_with_perfect_square_digit_sum_l709_709119

noncomputable def count_three_digit_numbers_with_perfect_square_digit_sum : ℕ :=
  let valid_digits := Finset.range 10
  let perfect_squares := [1, 4, 9, 16, 25]
  let three_digit_numbers := Finset.Icc 100 999
  (three_digit_numbers.filter (λ n, perfect_squares.contains (digit_sum n))).card
where
  digit_sum (n : ℕ) : ℕ := 
    let d1 := n / 100
    let d2 := (n / 10) % 10
    let d3 := n % 10
    d1 + d2 + d3

theorem three_digit_numbers_with_perfect_square_digit_sum :
  count_three_digit_numbers_with_perfect_square_digit_sum = 51 :=
sorry

end three_digit_numbers_with_perfect_square_digit_sum_l709_709119


namespace derivative_y_l709_709055

noncomputable def k : ℝ := Real.cot (Real.sin (1 / 3))

def y (x : ℝ) : ℝ := (k * (Real.sin (17 * x))^2) / (17 * (Real.cos (34 * x)))

theorem derivative_y (x : ℝ) : deriv y x = (k * Real.tan (34 * x)) / (Real.cos (34 * x)) :=
by
  sorry

end derivative_y_l709_709055


namespace first_inlet_filling_time_l709_709642

variable {T : ℝ}

-- Define the rates based on the conditions
def first_inlet_rate (T : ℝ) : ℝ := 1 / T
def second_inlet_rate (T : ℝ) : ℝ := 1 / (2 * T)
def combined_inlets_rate (T : ℝ) : ℝ := first_inlet_rate T + second_inlet_rate T
def outlet_rate : ℝ := 1 / 2

-- Define the work done in the first hour
def work_done_first_hour (T : ℝ) : ℝ := combined_inlets_rate T * 1

-- Define the combined rate when the outlet is opened
def combined_rate_with_outlet (T : ℝ) : ℝ := combined_inlets_rate T - outlet_rate

-- Define the remaining work done after the outlet is opened
def remaining_work_done (T : ℝ) : ℝ := 1 - work_done_first_hour T

-- Define the equation to solve
def equation_to_solve (T : ℝ) : Prop :=
  combined_rate_with_outlet T = remaining_work_done T

theorem first_inlet_filling_time : T = 2 :=
by
  have eqn : equation_to_solve T := sorry
  exact sorry

end first_inlet_filling_time_l709_709642


namespace unit_vector_in_direction_of_AB_l709_709833

-- Define the points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)

-- Define the vector AB
def AB := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of the vector AB
def mag_AB := Real.sqrt (AB.1^2 + AB.2^2)

-- Define the unit vector in the direction of AB
def unit_AB := (AB.1 / mag_AB, AB.2 / mag_AB)

-- Provide the statement for the problem to prove
theorem unit_vector_in_direction_of_AB :
  unit_AB = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2) :=
sorry

end unit_vector_in_direction_of_AB_l709_709833


namespace even_function_a_value_l709_709142

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, let y := (x - 1)^2 + a * x + sin(x + π / 2) in y = y) ↔ a = 2 :=
by
  let f := λ x, (x - 1)^2 + a * x + sin(x + π / 2)
  have h_even : ∀ x : ℝ, f(-x) = f(x) ↔ (a = 2) := sorry
  exact h_even

end even_function_a_value_l709_709142


namespace find_a_if_y_is_even_l709_709164

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem find_a_if_y_is_even (a : ℝ) (h : ∀ x : ℝ, y x a = y (-x) a) : a = 2 :=
by
  sorry

end find_a_if_y_is_even_l709_709164


namespace average_weight_whole_class_l709_709688

def sectionA_students : Nat := 36
def sectionB_students : Nat := 44
def avg_weight_sectionA : Float := 40.0 
def avg_weight_sectionB : Float := 35.0
def total_weight_sectionA := avg_weight_sectionA * Float.ofNat sectionA_students
def total_weight_sectionB := avg_weight_sectionB * Float.ofNat sectionB_students
def total_students := sectionA_students + sectionB_students
def total_weight := total_weight_sectionA + total_weight_sectionB
def avg_weight_class := total_weight / Float.ofNat total_students

theorem average_weight_whole_class :
  avg_weight_class = 37.25 := by
  sorry

end average_weight_whole_class_l709_709688


namespace arithmetic_sequence_and_formula_l709_709984

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709984


namespace find_number_l709_709353

-- Definitions based on conditions
def sum : ℕ := 555 + 445
def difference : ℕ := 555 - 445
def quotient : ℕ := 2 * difference
def remainder : ℕ := 70
def divisor : ℕ := sum

-- Statement to be proved
theorem find_number : (divisor * quotient + remainder) = 220070 := by
  sorry

end find_number_l709_709353


namespace num_teacher_volunteers_l709_709235

theorem num_teacher_volunteers (total_needed volunteers_from_classes extra_needed teacher_volunteers : ℕ)
  (h1 : teacher_volunteers + extra_needed + volunteers_from_classes = total_needed) 
  (h2 : total_needed = 50)
  (h3 : volunteers_from_classes = 6 * 5)
  (h4 : extra_needed = 7) :
  teacher_volunteers = 13 :=
by
  sorry

end num_teacher_volunteers_l709_709235


namespace bead_arrangement_probability_l709_709068

noncomputable def totalArrangements (red white blue green : ℕ) : ℕ :=
  let n := red + white + blue + green
  Nat.factorial n / (Nat.factorial red * Nat.factorial white * Nat.factorial blue * Nat.factorial green)

noncomputable def validArrangements := 0.05 * 12600

def probability_no_adjacent_beads_same (r w b g : ℕ) : Prop :=
  let total := totalArrangements r w b g
  validArrangements / total = 0.05

theorem bead_arrangement_probability :
  probability_no_adjacent_beads_same 4 3 2 1 := 
  sorry

end bead_arrangement_probability_l709_709068


namespace anna_gold_cost_per_gram_l709_709818

noncomputable def cost_per_gram_of_anna_gold : ℕ → ℕ → ℕ → ℕ → ℕ :=
  λ gary_grams gary_cost_per_gram anna_grams combined_cost,
    (combined_cost - gary_grams * gary_cost_per_gram) / anna_grams

theorem anna_gold_cost_per_gram :
  cost_per_gram_of_anna_gold 30 15 50 1450 = 20 :=
by
  unfold cost_per_gram_of_anna_gold
  simp
  refl

end anna_gold_cost_per_gram_l709_709818


namespace evaporation_period_days_l709_709345

theorem evaporation_period_days
    (initial_water : ℝ)
    (daily_evaporation : ℝ)
    (evaporation_percentage : ℝ)
    (total_evaporated_water : ℝ)
    (number_of_days : ℝ) :
    initial_water = 10 ∧
    daily_evaporation = 0.06 ∧
    evaporation_percentage = 0.12 ∧
    total_evaporated_water = initial_water * evaporation_percentage ∧
    number_of_days = total_evaporated_water / daily_evaporation →
    number_of_days = 20 :=
by
  sorry

end evaporation_period_days_l709_709345


namespace grid_divisible_by_rectangles_l709_709813

theorem grid_divisible_by_rectangles (n : ℕ) :
  (∃ m : ℕ, n * n = 7 * m) ↔ (∃ k : ℕ, n = 7 * k ∧ k > 1) :=
by
  sorry

end grid_divisible_by_rectangles_l709_709813


namespace problem_a2_sub_b2_problem_a_mul_b_l709_709917

theorem problem_a2_sub_b2 {a b : ℝ} (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 :=
sorry

theorem problem_a_mul_b {a b : ℝ} (h1 : a + b = 8) (h2 : a - b = 4) : a * b = 12 :=
sorry

end problem_a2_sub_b2_problem_a_mul_b_l709_709917


namespace marble_ratio_l709_709911

-- Definitions and assumptions from the conditions
def my_marbles : ℕ := 16
def total_marbles : ℕ := 63
def transfer_amount : ℕ := 2

-- After transferring marbles to my brother
def my_marbles_after_transfer := my_marbles - transfer_amount
def brother_marbles (B : ℕ) := B + transfer_amount

-- Friend's marbles
def friend_marbles (F : ℕ) := F = 3 * my_marbles_after_transfer

-- Prove the ratio of marbles after transfer
theorem marble_ratio (B F : ℕ) (hf : F = 3 * my_marbles_after_transfer) (h_total : my_marbles + B + F = total_marbles)
  (h_multiple : ∃ M : ℕ, my_marbles_after_transfer = M * brother_marbles B) :
  (my_marbles_after_transfer : ℚ) / (brother_marbles B : ℚ) = 2 / 1 :=
by
  sorry

end marble_ratio_l709_709911


namespace true_propositions_are_one_l709_709095

def problem_conditions := 
  let cond1 := ∀ a b c d : ℝ, (a = b) → (c = d) → (a = d)
  let cond2 := ∀ a b : ℝ, (a + b = 90) → (a < 90 ∧ b < 90)
  let cond3 := ∀ a b : ℝ, (a = b) → ∀ l1 l2 : ℝ, (l1 ∥ l2) 
  let cond4 := ∀ a b : ℝ, ∀ l1 l2 : ℝ, (l1 ⊥ l2) → (a = b)
  cond1 ∧ cond2 ∧ cond3 ∧ cond4

theorem true_propositions_are_one (cond1 cond2 cond3 cond4 : problem_conditions) :
  (1) :=
by
  sorry

end true_propositions_are_one_l709_709095


namespace ratio_of_smallest_to_middle_piece_l709_709730

variables (V_A V_B : ℝ)

-- Given conditions
def cone_sliced_three_pieces_parallel_base_same_height (h r : ℝ) : Prop :=
  let V_A := (19 / 3) * π * r^2 * h
  let V_B := (8 / 3) * π * r^2 * (2 * h)
  true

-- The goal to prove
theorem ratio_of_smallest_to_middle_piece 
  (h r : ℝ) 
  (H : cone_sliced_three_pieces_parallel_base_same_height h r) :
  V_A / V_B = 19 / 8 :=
sorry

end ratio_of_smallest_to_middle_piece_l709_709730


namespace increasing_sequence_nec_but_not_suf_l709_709531

theorem increasing_sequence_nec_but_not_suf (a : ℕ → ℝ) :
  (∀ n, abs (a (n + 1)) > a n) → (∀ n, a (n + 1) > a n) ↔ 
  ∃ (n : ℕ), ¬ (abs (a (n + 1)) > a n) ∧ (a (n + 1) > a n) :=
sorry

end increasing_sequence_nec_but_not_suf_l709_709531


namespace problem_statement_l709_709898

theorem problem_statement (x : ℤ) (h : 5 * x + 9 ≡ 3 [MOD 16]) : 3 * x + 8 ≡ 14 [MOD 16] :=
sorry

end problem_statement_l709_709898


namespace even_function_a_value_l709_709143

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, let y := (x - 1)^2 + a * x + sin(x + π / 2) in y = y) ↔ a = 2 :=
by
  let f := λ x, (x - 1)^2 + a * x + sin(x + π / 2)
  have h_even : ∀ x : ℝ, f(-x) = f(x) ↔ (a = 2) := sorry
  exact h_even

end even_function_a_value_l709_709143


namespace red_apples_percentage_of_second_tree_l709_709038

theorem red_apples_percentage_of_second_tree
    (total_red_apples : ℕ)
    (red_apples_first_tree : ℕ)
    (apples_per_tree : ℕ) :
    total_red_apples = 18 →
    red_apples_first_tree = 8 →
    apples_per_tree = 20 →
    let red_apples_second_tree := total_red_apples - red_apples_first_tree in
    let second_tree_red_percentage := (red_apples_second_tree : ℚ) / apples_per_tree * 100 in
    second_tree_red_percentage = 50 := 
by 
  sorry

end red_apples_percentage_of_second_tree_l709_709038


namespace abs_condition_l709_709313

theorem abs_condition (x : ℝ) : |2 * x - 7| ≤ 0 ↔ x = 7 / 2 := 
by
  sorry

end abs_condition_l709_709313


namespace even_sum_probability_l709_709050

theorem even_sum_probability :
  let P_even1 := 1 / 2,
      P_odd1 := 1 / 2,
      P_even2 := 1 / 3,
      P_odd2 := 2 / 3 in
  (P_even1 * P_even2 + P_odd1 * P_odd2) = 1 / 2 :=
by
  sorry

end even_sum_probability_l709_709050


namespace distance_focus_directrix_l709_709619

-- Given conditions
def parabola_eqn (x y : ℝ) : Prop := y^2 = 5 * x

-- Proposition to prove
theorem distance_focus_directrix : 
  let a := (5 : ℝ) / 4 in
  let focus := (a, 0) in
  let directrix := -a in
  dist (focus.fst) directrix = 5 / 2 :=
by
  sorry

end distance_focus_directrix_l709_709619


namespace data_set_average_l709_709078

theorem data_set_average 
  (x : ℝ) 
  (h_mode : Multiset.mode {6, x, 3, 3, 5, 1} = {3, 6}) :
  (6 + x + 3 + 3 + 5 + 1) / 6 = 4 :=
by
  sorry

end data_set_average_l709_709078


namespace dishes_with_lentils_l709_709361

theorem dishes_with_lentils (total_dishes: ℕ) (beans_and_lentils: ℕ) (beans_and_seitan: ℕ) 
    (only_one_kind: ℕ) (only_beans: ℕ) (only_seitan: ℕ) (only_lentils: ℕ) :
    total_dishes = 10 →
    beans_and_lentils = 2 →
    beans_and_seitan = 2 →
    only_one_kind = (total_dishes - beans_and_lentils - beans_and_seitan) →
    only_beans = (only_one_kind / 2) →
    only_beans = 3 * only_seitan →
    only_lentils = (only_one_kind - only_beans - only_seitan) →
    total_dishes_with_lentils = beans_and_lentils + only_lentils →
    total_dishes_with_lentils = 4 :=
begin
  intros,
  sorry
end

end dishes_with_lentils_l709_709361


namespace increasing_interval_implication_l709_709505

theorem increasing_interval_implication (a : ℝ) :
  (∀ x ∈ Set.Ioo (1 / 2) 2, (1 / x + 2 * a * x > 0)) → a > -1 / 8 :=
by
  intro h
  sorry

end increasing_interval_implication_l709_709505


namespace count_valid_matrices_l709_709882

/-- We define a configuration of a 6x6 matrix with entries of 1 and -1 such that the sum of each row
    and each column is 0, and then ensure the number of such configurations is 12101. 
    -/
noncomputable def valid_matrices : Nat :=
  12101

theorem count_valid_matrices : ∃ (M : Matrix (Fin 6) (Fin 6) ℤ), (∀ i, (∑ j, M i j) = 0) ∧ (∀ j, (∑ i, M i j) = 0) ∧ (∀ i j, M i j = 1 ∨ M i j = -1) ∧ valid_matrices = 12101 :=
by
  sorry

end count_valid_matrices_l709_709882


namespace intersection_M_N_l709_709084

open Set Real

def M := {x : ℝ | x^2 < 4}
def N := {x : ℝ | ∃ α : ℝ, x = sin α}
def IntersectSet := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = IntersectSet := by
  sorry

end intersection_M_N_l709_709084


namespace midpoint_P_BD_l709_709216

variables (ω1 ω2 ω3 : Type) [circle ω1] [circle ω2] [circle ω3]
variables (P Q A B C D : Point)
variables (hω1ω2_intersect : intersect ω1 ω2 P Q)
variables (h_tangent_P_A : tangent_at ω1 P A)
variables (h_tangent_P_B : tangent_at ω2 P B)
variables (h_C_reflection : reflection P A C)
variables (h_D_second_intersect : second_intersect (line_through P B) (circumcircle P Q C) D)

theorem midpoint_P_BD :
  midpoint P B D :=
sorry

end midpoint_P_BD_l709_709216


namespace find_abc_l709_709873

noncomputable def A : Polynomial ℝ :=
  10 * Polynomial.X ^ 2 - 6 * Polynomial.X * Polynomial.Y + 3 * Polynomial.Y ^ 2

noncomputable def B (a b c : ℝ) : Polynomial ℝ :=
  a * Polynomial.X ^ 2 + b * Polynomial.X * Polynomial.Y + c * Polynomial.Y ^ 2

noncomputable def C : Polynomial ℝ :=
  15 * Polynomial.Y ^ 4 - 36 * Polynomial.X * Polynomial.Y ^ 3 + 62 * Polynomial.X ^ 2 * Polynomial.Y ^ 2 - 20 * Polynomial.X ^ 3 * Polynomial.Y

theorem find_abc (a b c : ℝ) : (A * B a b c = C) → a = 0 ∧ b = -2 ∧ c = 5 :=
by
  sorry

end find_abc_l709_709873


namespace three_digit_numbers_with_perfect_square_digit_sum_l709_709118

noncomputable def count_three_digit_numbers_with_perfect_square_digit_sum : ℕ :=
  let valid_digits := Finset.range 10
  let perfect_squares := [1, 4, 9, 16, 25]
  let three_digit_numbers := Finset.Icc 100 999
  (three_digit_numbers.filter (λ n, perfect_squares.contains (digit_sum n))).card
where
  digit_sum (n : ℕ) : ℕ := 
    let d1 := n / 100
    let d2 := (n / 10) % 10
    let d3 := n % 10
    d1 + d2 + d3

theorem three_digit_numbers_with_perfect_square_digit_sum :
  count_three_digit_numbers_with_perfect_square_digit_sum = 51 :=
sorry

end three_digit_numbers_with_perfect_square_digit_sum_l709_709118


namespace even_function_a_value_l709_709138

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, let y := (x - 1)^2 + a * x + sin(x + π / 2) in y = y) ↔ a = 2 :=
by
  let f := λ x, (x - 1)^2 + a * x + sin(x + π / 2)
  have h_even : ∀ x : ℝ, f(-x) = f(x) ↔ (a = 2) := sorry
  exact h_even

end even_function_a_value_l709_709138


namespace ratio_of_averages_is_one_l709_709741

-- Define the setup and necessary variables
variables {x : ℕ → ℝ} {A A' : ℝ}

-- Given conditions
def avg_50_scores (x : ℕ → ℝ) : ℝ :=
  (∑ i in finset.range 50, x i) / 50

def new_avg (x : ℕ → ℝ) (A : ℝ) : ℝ :=
  ((∑ i in finset.range 50, x i) + A) / 51

-- Statement to prove the ratio
theorem ratio_of_averages_is_one (x : ℕ → ℝ) (A : ℝ) (hA : avg_50_scores x = A) :
  let A' := new_avg x A in A' / A = 1 :=
by
  -- Proof would go here
  sorry

end ratio_of_averages_is_one_l709_709741


namespace vec_eqn_solution_l709_709878

theorem vec_eqn_solution :
  ∀ m : ℝ, let a : ℝ × ℝ := (1, -2) 
           let b : ℝ × ℝ := (m, 4) 
           (a.1 * b.2 = a.2 * b.1) → 2 • a - b = (4, -8) :=
by
  intro m a b h_parallel
  sorry

end vec_eqn_solution_l709_709878


namespace area_AKD_l709_709616

variables (A B C D O K : Type)
variables (AO OC : ℝ) (BK KC : ℕ) 

-- Conditions
variables (angle_ACB_eq_90 : ∃ O, ∀ (A B C D : Type), ∠ACB = 90)
variables (AO_eq_2 : AO = 2)
variables (OC_eq_3 : OC = 3)
variables (BK_KC_ratio : BK / KC = 1 / 2)
variables (triangle_AKD_equilateral : equilateral_triangle A K D)

-- Prove the area
theorem area_AKD : ∃ area, area = 7 * (sqrt 3) / 3 :=
by
  sorry

end area_AKD_l709_709616


namespace part1_part2_l709_709860

-- Define the function f and the absolute value operator
def abs (x : ℝ) : ℝ := if x ≥ 0 then x else -x

def f (x a : ℝ) : ℝ := abs (x - 2) + abs (2 * x + a)

-- Part (Ⅰ)
theorem part1 (x : ℝ) : f x 1 ≥ 5 ↔ (x ≤ -4 / 3 ∨ x ≥ 2) := 
  sorry

-- Part (Ⅱ)
theorem part2 (a : ℝ) (x0 : ℝ) (h : f x0 a + abs (x0 - 2) < 3) : -7 < a ∧ a < -1 := 
  sorry

end part1_part2_l709_709860


namespace num_valid_arrays_l709_709884

def valid_array (A : Matrix (Fin 6) (Fin 6) ℤ) : Prop :=
  (∀ i, ∑ j, A i j = 0) ∧ (∀ j, ∑ i, A i j = 0) ∧ (∀ i j, A i j = 1 ∨ A i j = -1)

theorem num_valid_arrays : 
  ∃ (n : ℕ), n = 160400 ∧ ∀ A : Matrix (Fin 6) (Fin 6) ℤ, valid_array A → A ∈ (Matrix (λ (i j : Fin 6), {x : ℤ // x = 1 ∨ x = -1}) (Fin 6) (Fin 6)) :=
sorry

end num_valid_arrays_l709_709884


namespace smallest_five_digit_multiple_of_18_l709_709424

def is_multiple_of (x : ℕ) (k : ℕ) : Prop := ∃ n : ℕ, x = k * n

theorem smallest_five_digit_multiple_of_18 : 
  ∃ x : ℕ, 
    (10000 ≤ x ∧ x < 100000) ∧ 
    is_multiple_of x 18 ∧ 
    (∀ y : ℕ, (10000 ≤ y ∧ y < 100000) ∧ is_multiple_of y 18 → x ≤ y) :=
begin
  use 10008,
  -- The details of the proof are omitted.
  sorry
end

end smallest_five_digit_multiple_of_18_l709_709424


namespace inverse_47_mod_48_l709_709788

theorem inverse_47_mod_48 : ∃ x, x < 48 ∧ x > 0 ∧ 47 * x % 48 = 1 :=
sorry

end inverse_47_mod_48_l709_709788


namespace solution_sqrt_eq_l709_709792

theorem solution_sqrt_eq (x : ℝ) : x = 81 ∨ x = 4 ↔ sqrt x = 18 / (11 - sqrt x) := by
  sorry

end solution_sqrt_eq_l709_709792


namespace weight_of_replaced_person_l709_709613

theorem weight_of_replaced_person 
  (average_increase : ℝ) 
  (num_persons : ℕ) 
  (new_person_weight : ℝ)
  (h1 : average_increase = 2.5)
  (h2 : num_persons = 8)
  (h3 : new_person_weight = 86) : 
  let weight_of_replaced := new_person_weight - (average_increase * num_persons) in 
  weight_of_replaced = 66 :=
by
  dsimp
  rw [h1, h2, h3]
  norm_num
  done

end weight_of_replaced_person_l709_709613


namespace find_A_l709_709171

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) 
(h_div9 : (A + 1 + 5 + B + 9 + 4) % 9 = 0) 
(h_div11 : (A + 5 + 9 - (1 + B + 4)) % 11 = 0) : A = 5 :=
by sorry

end find_A_l709_709171


namespace eval_expr_l709_709781

-- Define the variables a, b, c as given in the problem
def a : ℝ := 12
def b : ℝ := 14
def c : ℝ := 19

-- Define the fractions
def f1 : ℝ := 1 / 14 - 1 / 19
def f2 : ℝ := 1 / 19 - 1 / 10
def f3 : ℝ := 1 / 10 - 1 / 14

-- Define the main expression
def expr : ℝ :=
  (144 * f1 + 196 * f2 + 361 * f3) /
  (12 * f1 + 14 * f2 + 19 * f3)

-- Define S
def S : ℝ := a + b + c

-- State the theorem
theorem eval_expr : expr = S := by
  sorry

end eval_expr_l709_709781


namespace no_two_positive_roots_l709_709525

theorem no_two_positive_roots (a : Fin 2022 → ℕ)
  (h : ∀ i : Fin 2022, 2 + i < 2024)
  (distinct : Function.Injective a)
  (coeffs : ∀ i : Fin 2022, (a i) ∈ {n | 2 ≤ n ∧ n ≤ 2023}) :
  ¬ ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ (x₁ ^ 2022 - ∑ i in Finset.range 2022, (a ⟨i, Fin.is_lt (i, 2022)⟩) * x₁ ^ (2021 - i)) = 2023 ∧ (x₂ ^ 2022 - ∑ i in Finset.range 2022, (a ⟨i, Fin.is_lt (i, 2022)⟩) * x₂ ^ (2021 - i)) = 2023 := sorry

end no_two_positive_roots_l709_709525


namespace central_angle_is_two_length_of_chord_l709_709262

-- Define the conditions
constant r : ℝ
constant θ : ℝ
constant l : ℝ

axiom h1 : (1 / 2) * r^2 * θ = 1
axiom h2 : 2 * r + r * θ = 4

-- Prove the central angle in radians is 2
theorem central_angle_is_two : θ = 2 :=
sorry

-- Prove the length of the chord AB is 2 * sin(1)
theorem length_of_chord : l = 2 * sin 1 :=
sorry

end central_angle_is_two_length_of_chord_l709_709262


namespace find_f_of_2_l709_709457

noncomputable def f : ℝ → ℝ :=
  λ x, if h : ∃ y, y^5 = x then log 10 (classical.some h) else 0

axiom f_property : ∀ x, f(x^5) = log 10 x

theorem find_f_of_2 : f 2 = (1 / 5) * log 10 2 := by
  sorry

end find_f_of_2_l709_709457


namespace find_distinct_pairs_l709_709043

theorem find_distinct_pairs (n k : ℕ) (h_nk : n ≠ k) : 
  (∃ s : ℕ, nat.divisors_count (s * n) = nat.divisors_count (s * k)) ↔ (¬ n ∣ k ∧ ¬ k ∣ n) :=
by sorry

end find_distinct_pairs_l709_709043


namespace minimum_construction_cost_l709_709739

/-- Define the area constraint -/
def area_constraint (x : ℝ) : Prop := x * (12 / x) = 12

/-- Define the construction cost -/
def construction_cost (x : ℝ) : ℝ :=
  900 * (x + 16 / x) + 5800

/-- Define the domain constraint -/
def domain_constraint (x a : ℝ) : Prop := 0 < x ∧ x ≤ a

/-- Prove the minimum construction cost -/
theorem minimum_construction_cost (a : ℝ) (h_a : 0 < a) :
  ∃ x : ℝ, domain_constraint x a ∧
    let y := construction_cost x in
    (a ≥ 4 → x = 4 ∧ y = 13000) ∧
    (0 < a < 4 → x = a ∧ y = 900 * (a + 16 / a) + 5800) :=
sorry

end minimum_construction_cost_l709_709739


namespace merchant_profit_percentage_l709_709921

noncomputable def cost_price_of_one_article (C : ℝ) : Prop := ∃ S : ℝ, 20 * C = 16 * S

theorem merchant_profit_percentage (C S : ℝ) (h : cost_price_of_one_article C) : 
  100 * ((S - C) / C) = 25 :=
by 
  sorry

end merchant_profit_percentage_l709_709921


namespace distribute_balls_in_boxes_l709_709125

theorem distribute_balls_in_boxes (balls boxes : ℕ) (h_balls : balls = 7) (h_boxes : boxes = 2) : (boxes ^ balls) = 128 :=
by
  simp [h_balls, h_boxes]
  sorry

end distribute_balls_in_boxes_l709_709125


namespace omega_range_l709_709861

def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 3) - Real.sqrt 3

theorem omega_range (ω : ℝ) : (∃ x1 x2 x3 ∈ set.Icc 0 (Real.pi / 2), 
  f ω x1 = 0 ∧ f ω x2 = 0 ∧ f ω x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  4 ≤ ω ∧ ω < 14 / 3 :=
sorry

end omega_range_l709_709861


namespace total_selling_price_l709_709024

theorem total_selling_price (profit_per_meter cost_price_per_meter meters : ℕ)
  (h_profit : profit_per_meter = 20)
  (h_cost : cost_price_per_meter = 85)
  (h_meters : meters = 85) :
  (cost_price_per_meter + profit_per_meter) * meters = 8925 :=
by
  sorry

end total_selling_price_l709_709024


namespace usual_time_of_journey_l709_709303

/-- The usual time for the cab to cover the journey given that it is 6 minutes late when traveling
at 5/6th of its usual speed. -/
theorem usual_time_of_journey (S T : ℝ) (h1 : S > 0) :
  S * T = (5/6 * S) * (T + 1/10) → T = 1/2 :=
by
  intro h
  have h2 : S ≠ 0 := by linarith
  calc
    S * T = (5 / 6) * S * (T + 1 / 10) : h
    _ = (5 / 6) * S * T + (5 / 6) * S * (1 / 10) : by rw [←mul_add]
    _ = (5 / 6) * S * T + (5 / 60) * S : by ring
  linarith
  sorry

end usual_time_of_journey_l709_709303


namespace exists_triplet_sum_gt_20_l709_709386

theorem exists_triplet_sum_gt_20 :
  ∃ (i : ℕ), i < 12 ∧ ( (i + 1 ≤ 12 → (i + 2 ≤ 12 → (i + 3 ≤ 12 → (∑ j in finset.range 3, (1 + i + j)) > 20))) ∨
                 (i + 2 = 12 → (∑ j in finset.range 3, (1 + i + j % 12)) > 20) ∨
                 (i + 1 = 12 → (∑ j in finset.range 3, (1 + i + j % 12)) > 20) ∨
                 (i = 11 → (∑ j in finset.range 3, (1 + (i + j) % 12)) > 20))) := sorry

end exists_triplet_sum_gt_20_l709_709386


namespace tire_profit_per_tire_l709_709341

/-- 
  Given the cost and revenue conditions for producing and selling type A and type B tires, 
  prove that the profit per tire for type A is $6.97 and for type B is $9.60.
-/
theorem tire_profit_per_tire 
  (batch_cost_A1 : ℕ) (cost_per_tire_A1 : ℕ) (batch_cost_A2 : ℕ) 
  (cost_per_tire_A2 : ℕ) (batch_cost_A3 : ℕ) (cost_per_tire_A3 : ℕ)
  (batch_cost_B : ℕ) (cost_per_tire_B : ℕ)
  (price_A1 : ℕ) (price_A2 : ℕ) (price_A3 : ℕ)
  (price_B : ℕ)
  (num_A : ℕ) (num_B : ℕ) :
  batch_cost_A1 = 22500 ∧ cost_per_tire_A1 = 8 ∧
  batch_cost_A2 = 20000 ∧ cost_per_tire_A2 = 7 ∧
  batch_cost_A3 = 18000 ∧ cost_per_tire_A3 = 6 ∧
  batch_cost_B = 24000 ∧ cost_per_tire_B = 7 ∧
  price_A1 = 20 ∧ price_A2 = 18 ∧ price_A3 = 16 ∧
  price_B = 19 ∧
  num_A = 15000 ∧ num_B = 10000 →
  let cost_A1 := batch_cost_A1 + cost_per_tire_A1 * 5000 in
  let cost_A2 := batch_cost_A2 + cost_per_tire_A2 * 5000 in
  let cost_A3 := batch_cost_A3 + cost_per_tire_A3 * 5000 in
  let cost_B := batch_cost_B + cost_per_tire_B * num_B in
  let revenue_A1 := price_A1 * 5000 in
  let revenue_A2 := price_A2 * 5000 in
  let revenue_A3 := price_A3 * 5000 in
  let revenue_B := price_B * num_B in
  let profit_A := (revenue_A1 + revenue_A2 + revenue_A3) - (cost_A1 + cost_A2 + cost_A3) in
  let profit_B := revenue_B - cost_B in
  (profit_A.to_rat / num_A.to_rat) = 6.97 ∧ (profit_B.to_rat / num_B.to_rat) = 9.60 :=
sorry

end tire_profit_per_tire_l709_709341


namespace count_integers_congruent_to_7_mod_13_l709_709113

theorem count_integers_congruent_to_7_mod_13 : 
  (∃ (n : ℕ), ∀ x, (1 ≤ x ∧ x < 500 ∧ x % 13 = 7) → x = 7 + 13 * n ∧ n < 38) :=
sorry

end count_integers_congruent_to_7_mod_13_l709_709113


namespace total_caps_l709_709001

-- Definitions for the conditions
def week1_caps : ℕ := 320
def week2_caps : ℕ := 400
def week3_caps : ℕ := 300
def average_caps := (week1_caps + week2_caps + week3_caps) / 3
def first_half_week4 := average_caps + (average_caps / 10)
def second_half_week4 := first_half_week4 + (first_half_week4 * 3 / 10)

-- Proof statement
theorem total_caps :
  let total_weeks_1_to_3 := week1_caps + week2_caps + week3_caps
  let total_week4 := first_half_week4.toNat + second_half_week4.toNat 
  total_weeks_1_to_3 + total_week4 = 1880 :=
by 
  sorry

end total_caps_l709_709001


namespace specific_gravity_equal_five_halves_l709_709289

noncomputable theory

-- Let q be the weight of the sphere and v be its volume
variables (q v : ℝ)

-- Given conditions
def specific_weight (q v : ℝ) := q / v

def apparent_weight_fully_submerged (q v : ℝ) := q - v

def apparent_weight_half_submerged (q v : ℝ) := q - v / 2

-- The given critical condition
axiom critical_condition (q v : ℝ) : apparent_weight_half_submerged q v = 2 * apparent_weight_fully_submerged q v

-- To be proven: the specific gravity of the material equals 5 / 2
theorem specific_gravity_equal_five_halves (q v : ℝ) (h : critical_condition q v) : specific_weight q v = 5 / 2 :=
sorry

end specific_gravity_equal_five_halves_l709_709289


namespace clothes_add_percentage_l709_709321

theorem clothes_add_percentage (W : ℝ) (hW: W > 0) :
  let new_weight := 0.86 * W,
      final_weight := 0.8772 * W in
  ∃ C : ℝ, final_weight = new_weight * (1 + C) ∧ C = 0.02 :=
begin
  sorry
end

end clothes_add_percentage_l709_709321


namespace odd_degree_palindromic_root_l709_709596

-- Define the polynomial
def P (x : ℝ) (a b : ℝ) (k : ℕ) : ℝ :=
  a * x^(2*k + 1) + b * x^(2*k) + ... + b * x + a

-- State the theorem
theorem odd_degree_palindromic_root (a b : ℝ) (k : ℕ) :
  P (-1) a b k = 0 :=
  sorry

end odd_degree_palindromic_root_l709_709596


namespace circle_through_points_l709_709267

noncomputable def circle_eq (x y : ℝ) : ℝ := (x - 1)^2 + (y - 1)^2 - 4

theorem circle_through_points
  (A B : ℝ × ℝ)
  (line : ℝ × ℝ → Prop)
  (hA : A = (1, -1))
  (hB : B = (-1, 1))
  (hline : line = λ p : ℝ × ℝ, p.1 + p.2 - 2 = 0)
  (hA_on_circle : circle_eq A.1 A.2 = 0)
  (hB_on_circle : circle_eq B.1 B.2 = 0)
  (center : ℝ × ℝ)
  (hcenter_on_line : line center) :
  ∃ center_radius : ℝ × ℝ,
    center_radius = (1, 1) ∧ (circle_eq = λ x y, (x - 1)^2 + (y - 1)^2 - 4) :=
sorry

end circle_through_points_l709_709267


namespace original_price_l709_709362

variable (a : ℝ)

-- Given the price after a 20% discount is a yuan per unit,
-- Prove that the original price per unit was (5/4) * a yuan.
theorem original_price (h : a > 0) : (a / (4 / 5)) = (5 / 4) * a :=
by sorry

end original_price_l709_709362


namespace calculate_expression_l709_709762

theorem calculate_expression :
  ( (1 / 2) ^ (-2) + abs (-3) + (2 - real.sqrt 3) ^ 0 - real.sqrt 2 * real.cos (real.pi / 4) ) = 7 :=
by
  -- sorry, proof skipped as per the guidelines
  sorry

end calculate_expression_l709_709762


namespace max_parallel_equidistant_lines_l709_709523

theorem max_parallel_equidistant_lines (n : ℕ) (h_lines : ∀ i j : ℕ, i ≠ j → (lines i) ∥ (lines j)) 
  (h_equidistant : ∀ i j : ℕ, i ≠ j → distance (lines i) (lines j) = d) : n ≤ 3 :=
sorry

end max_parallel_equidistant_lines_l709_709523


namespace selection_schemes_correct_l709_709245

noncomputable def total_selection_schemes : ℕ :=
  let total_people := 6
  let foreign_lang_choices := 4
  let remaining_choices := total_people - 1
  let remaining_competitions := 3
  foreign_lang_choices * remaining_choices * (remaining_choices - 1) * (remaining_choices - 2)

theorem selection_schemes_correct :
  total_selection_schemes = 240 :=
by
  unfold total_selection_schemes
  norm_num
  sorry

end selection_schemes_correct_l709_709245


namespace even_function_a_value_l709_709141

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, let y := (x - 1)^2 + a * x + sin(x + π / 2) in y = y) ↔ a = 2 :=
by
  let f := λ x, (x - 1)^2 + a * x + sin(x + π / 2)
  have h_even : ∀ x : ℝ, f(-x) = f(x) ↔ (a = 2) := sorry
  exact h_even

end even_function_a_value_l709_709141


namespace min_value_of_function_l709_709277

theorem min_value_of_function (x : ℝ) (h : x > 0) : (∃ y : ℝ, y = x^2 + 3 * x + 1 ∧ ∀ z, z = x^2 + 3 * x + 1 → y ≤ z) → y = 5 :=
by
  sorry

end min_value_of_function_l709_709277


namespace solution_set_impossible_l709_709635

theorem solution_set_impossible (a b : ℝ) (h : b ≠ 0) : 
  ¬(∀ x : ℝ, x ∈ set.Iio (-b / a) ↔ a * x > b) :=
sorry

end solution_set_impossible_l709_709635


namespace minimized_point_is_orthocenter_l709_709548

noncomputable def isosceles_triangle_min_condition (A B C P M N : Point) : Prop :=
  is_isosceles_triangle A B C ∧
  inside_triangle P A B C ∧
  is_circle_intersection C A P M N ∧
  minimization_condition (mn_length M N + bp_length B P + cp_length C P)

theorem minimized_point_is_orthocenter 
  {A B C P M N : Point}
  (h : isosceles_triangle_min_condition A B C P M N) :
  is_orthocenter P A B C :=
sorry

end minimized_point_is_orthocenter_l709_709548


namespace cost_of_soap_per_year_l709_709577

-- Conditions:
def duration_of_soap (bar: Nat) : Nat := 2
def cost_per_bar (bar: Nat) : Real := 8.0
def months_in_year : Nat := 12

-- Derived quantity
def bars_needed (months: Nat) (duration: Nat): Nat := months / duration

-- Theorem statement:
theorem cost_of_soap_per_year : 
  let n := bars_needed months_in_year (duration_of_soap 1)
  n * (cost_per_bar 1) = 48.0 := 
  by 
    -- Skipping proof
    sorry

end cost_of_soap_per_year_l709_709577


namespace arithmetic_sequence_and_formula_l709_709990

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709990


namespace find_a_perpendicular_lines_l709_709312

theorem find_a_perpendicular_lines (a : ℚ) :
  (let line1 := 3 * a + 9
       line2 := 2 * a + 9
   in 3 * (2 * a + 9) = -1) →
  a = -14/3 :=
by sorry

end find_a_perpendicular_lines_l709_709312


namespace min_value_expression_l709_709914

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
    (x + 1 / y) ^ 2 + (y + 1 / (2 * x)) ^ 2 ≥ 3 + 2 * Real.sqrt 2 := 
sorry

end min_value_expression_l709_709914


namespace number_of_integer_values_n_l709_709772

theorem number_of_integer_values_n : 
  let p := λ n : ℤ, ∃ k : ℤ, (8000 : ℤ) * 2^n * 5^(-n) = k  in
  (card {n : ℤ | p n}) = 10 :=
by
  sorry

end number_of_integer_values_n_l709_709772


namespace intersection_of_P_and_Q_l709_709874

def P : Set ℝ := {x | 1 ≤ x}
def Q : Set ℝ := {x | x < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_of_P_and_Q_l709_709874


namespace wonder_nominal_GDP_2009_wonder_nominal_GDP_2015_wonder_real_GDP_2015_wonder_growth_rate_real_GDP_wonder_CPI_2015_l709_709573

noncomputable theory

def nominal_GDP_2009 : ℕ := 5 * 12 + 7 * 8 + 9 * 6
def nominal_GDP_2015 : ℕ := 6 * 15 + 5 * 10 + 10 * 2
def real_GDP_2015 : ℕ := 5 * 15 + 7 * 10 + 9 * 2

def growth_rate_real_GDP : ℚ :=
  (163 - 170) / 170

def CPI_2015 : ℚ :=
  ( (12 * 6 + 8 * 5 + 6 * 10) / (12 * 5 + 8 * 7 + 6 * 9)) * 100

theorem wonder_nominal_GDP_2009:
  nominal_GDP_2009 = 170 := by
  sorry

theorem wonder_nominal_GDP_2015:
  nominal_GDP_2015 = 160 := by
  sorry

theorem wonder_real_GDP_2015:
  real_GDP_2015 = 163 := by
  sorry

theorem wonder_growth_rate_real_GDP:
  growth_rate_real_GDP = -0.0412 := by
  -- 4.12% as a decimal
  sorry

theorem wonder_CPI_2015:
  CPI_2015 = 101.17 := by
  sorry

end wonder_nominal_GDP_2009_wonder_nominal_GDP_2015_wonder_real_GDP_2015_wonder_growth_rate_real_GDP_wonder_CPI_2015_l709_709573


namespace probability_of_two_non_defective_pens_l709_709684

-- Definitions for conditions from the problem
def total_pens : ℕ := 16
def defective_pens : ℕ := 3
def selected_pens : ℕ := 2
def non_defective_pens : ℕ := total_pens - defective_pens

-- Function to calculate probability of drawing non-defective pens
noncomputable def probability_no_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  (non_defective_pens / total_pens) * ((non_defective_pens - 1) / (total_pens - 1))

-- Theorem stating the correct answer
theorem probability_of_two_non_defective_pens : 
  probability_no_defective total_pens defective_pens selected_pens = 13 / 20 :=
by
  sorry

end probability_of_two_non_defective_pens_l709_709684


namespace gyuri_seungyeon_work_together_l709_709487

variable {D J : Type} [DivisionRing D] [MulAction D J] [MulOneClass D]

-- Gyuri's work rate: she takes 5 days to do 1/3 of the job
def gyuri_work_rate : D := (1 / 3) / 5

-- Seungyeon's work rate: she takes 2 days to do 1/5 of the job
def seungyeon_work_rate : D := (1 / 5) / 2

-- Combined work rate in one day
def combined_work_rate : D := gyuri_work_rate + seungyeon_work_rate

-- Time to complete the job together
def time_to_complete_job := 1 / combined_work_rate

theorem gyuri_seungyeon_work_together (h1 : gyuri_work_rate = 1 / 15) (h2 : seungyeon_work_rate = 1 / 10) 
(h3 : combined_work_rate = 1 / 15 + 1 / 10) (h4 : time_to_complete_job = 6) :
  time_to_complete_job = 6 := by
  -- proof steps would go here
  sorry

end gyuri_seungyeon_work_together_l709_709487


namespace least_positive_integer_with_four_prime_factors_l709_709413

-- Define a function that returns the number of prime factors (not necessarily distinct)
def num_prime_factors (n : ℕ) : ℕ :=
  (Multiset.card (Multiset.filter (λ x, nat.prime x) (Multiset.of_list (n.divisors))))

theorem least_positive_integer_with_four_prime_factors :
  ∃ n > 0, num_prime_factors n = 4 ∧ num_prime_factors (n+1) = 4 ∧
  ∀ m, m > 0 → num_prime_factors m = 4 ∧ num_prime_factors (m+1) = 4 → m ≥ 1155 :=
begin
  sorry
end

end least_positive_integer_with_four_prime_factors_l709_709413


namespace logarithmic_arithmetic_sequence_l709_709902

theorem logarithmic_arithmetic_sequence (x : Real) :
  (Log 10 (2)) < (Log 10 (2 ^ x + 1)) < (Log 10 (2 ^ x + 5)) ∧
  (2 * Log 10 (2 ^ x + 1) = Log 10 (2) + Log 10 (2 ^ x + 5)) →
  x = Real.log 3 / Real.log 2 :=
by
  sorry

end logarithmic_arithmetic_sequence_l709_709902


namespace total_interest_paid_l709_709545

-- Define the problem as a theorem in Lean 4
theorem total_interest_paid
  (initial_investment : ℝ)
  (interest_6_months : ℝ)
  (interest_10_months : ℝ)
  (interest_18_months : ℝ)
  (total_interest : ℝ) :
  initial_investment = 10000 ∧ 
  interest_6_months = 0.02 * initial_investment ∧
  interest_10_months = 0.03 * (initial_investment + interest_6_months) ∧
  interest_18_months = 0.04 * (initial_investment + interest_6_months + interest_10_months) ∧
  total_interest = interest_6_months + interest_10_months + interest_18_months →
  total_interest = 926.24 :=
by
  sorry

end total_interest_paid_l709_709545


namespace bus_driver_hours_worked_last_week_l709_709709

-- Definitions for given conditions
def regular_rate : ℝ := 12
def passenger_rate : ℝ := 0.50
def overtime_rate_1 : ℝ := 1.5 * regular_rate
def overtime_rate_2 : ℝ := 2 * regular_rate
def total_compensation : ℝ := 1280
def total_passengers : ℝ := 350
def earnings_from_passengers : ℝ := total_passengers * passenger_rate
def earnings_from_hourly_rate : ℝ := total_compensation - earnings_from_passengers
def regular_hours : ℝ := 40
def first_tier_overtime_hours : ℝ := 5

-- Theorem to prove the number of hours worked is 67
theorem bus_driver_hours_worked_last_week :
  ∃ (total_hours : ℝ),
    total_hours = 67 ∧
    earnings_from_passengers = total_passengers * passenger_rate ∧
    earnings_from_hourly_rate = total_compensation - earnings_from_passengers ∧
    (∃ (overtime_hours : ℝ),
      (overtime_hours = regular_hours + first_tier_overtime_hours + (earnings_from_hourly_rate - (regular_hours * regular_rate) - (first_tier_overtime_hours * overtime_rate_1)) / overtime_rate_2) ∧
      total_hours = regular_hours + first_tier_overtime_hours + (earnings_from_hourly_rate - (regular_hours * regular_rate) - (first_tier_overtime_hours * overtime_rate_1)) / overtime_rate_2 )
  :=
sorry

end bus_driver_hours_worked_last_week_l709_709709


namespace problem_a2_sum_eq_364_l709_709492

noncomputable def a (n : ℕ) : ℕ :=
  (polynomial.repr ((1 + X + X^2) ^ 6)).coeff n

theorem problem_a2_sum_eq_364 :
  (a 2) + (a 4) + (a 6) + (a 8) + (a 10) + (a 12) = 364 :=
sorry

end problem_a2_sum_eq_364_l709_709492


namespace ratio_of_time_l709_709544

-- Definitions for the conditions
def t_first : ℕ := 2
def t_total : ℕ := 10
def t_other_two : ℕ := t_total - t_first

-- Statement to be proven
theorem ratio_of_time (t_first t_total t_other_two : ℕ) 
  (h_first : t_first = 2) (h_total : t_total = 10) (h_other_two : t_other_two = t_total - t_first) : 
  t_other_two / t_first = 4 :=
by
  rw [h_first, h_total, h_other_two]
  -- Complete the proof here
  sorry

end ratio_of_time_l709_709544


namespace Kiera_dried_fruit_l709_709205

theorem Kiera_dried_fruit : ∃ F : ℕ, (∀ p : ℕ, (p = 2) → (16 + F) % p = 0) ∧ F = 2 :=
by {
  use 2,
  split,
  { 
    intros p hp, 
    rw hp, 
    norm_num 
  },
  refl
}

end Kiera_dried_fruit_l709_709205


namespace cos_theta_planes_l709_709556

noncomputable def cos_theta {α : Type*} [linear_ordered_field α] (n1 n2 : α × α × α) : α :=
(n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3) / 
((Real.sqrt (n1.1^2 + n1.2^2 + n1.3^2)) * (Real.sqrt (n2.1^2 + n2.2^2 + n2.3^2)))

theorem cos_theta_planes : 
  cos_theta (3, -2, 1) (4, 1, -3) = 7 / Real.sqrt 364 :=
by
  sorry

end cos_theta_planes_l709_709556


namespace expression_value_l709_709906

variable (m n : ℝ)

theorem expression_value (h : m - n = 1) : (m - n)^2 - 2 * m + 2 * n = -1 :=
by
  sorry

end expression_value_l709_709906


namespace abs_nested_expression_l709_709908

theorem abs_nested_expression (x : ℝ) (h : x < -4) : |2 - |2 + 2 * x|| = 4 :=
sorry

end abs_nested_expression_l709_709908


namespace isosceles_triangle_relation_l709_709696

variables {A B C D : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]

-- Define the points and segments in the isosceles triangle
variable (triangle_ABC : IsoscelesTriangle A B C)
variable (point_D : PointOnBase D B C)

-- Define the distances
def AD := dist A D
def BD := dist B D
def CD := dist C D

-- The theorem to be proven
theorem isosceles_triangle_relation :
  2 * AD ^ 2 > BD ^ 2 + CD ^ 2 := 
sorry

end isosceles_triangle_relation_l709_709696


namespace fourth_term_geometric_sequence_eq_neg_24_l709_709624

def first_term := (x : ℝ) : ℝ := x
def second_term := (x : ℝ) : ℝ := 3 * x + 3
def third_term := (x : ℝ) : ℝ := 6 * x + 6

theorem fourth_term_geometric_sequence_eq_neg_24 (x : ℝ) (hx : first_term x = x ∧ second_term x = 3 * x + 3 ∧ third_term x = 6 * x + 6) : 
  (x ≠ 0 → ∃ r : ℝ, r = (second_term x / first_term x) ∧ 6 * x + 6 * r = -24) :=
begin
  sorry
end

end fourth_term_geometric_sequence_eq_neg_24_l709_709624


namespace count_eq_58_l709_709115

-- Definitions based on given conditions
def isThreeDigit (n : Nat) : Prop := n >= 100 ∧ n <= 999
def digitSum (n : Nat) : Nat := 
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 + d2 + d3
def isPerfectSquare (n : Nat) : Prop := ∃ k : Nat, k * k = n

-- Condition for digit sum being one of the specific perfect squares
def isPerfectSquareDigitSum (n : Nat) : Prop :=
  isThreeDigit n ∧ (digitSum n = 1 ∨ digitSum n = 4 ∨ digitSum n = 9 ∨ digitSum n = 16 ∨ digitSum n = 25)

-- Total count of such numbers
def countPerfectSquareDigitSumNumbers : Finset Nat :=
  (Finset.range 1000).filter isPerfectSquareDigitSum

-- Lean statement to prove the count is 58
theorem count_eq_58 : countPerfectSquareDigitSumNumbers.card = 58 := sorry

end count_eq_58_l709_709115


namespace maria_earnings_correct_l709_709583

/-- Maria's earnings calculation over three days -/
def maria_total_earnings : ℕ :=
  let first_day := (30 * 2) + (20 * 3) in
  let second_day := ((30 * 2) * 2) + ((20 * 3) * 2) in
  let third_day := ((30 * 2) * 0.1 * 2) + (16 * 3) in
  first_day + second_day + third_day
  
theorem maria_earnings_correct : 
  maria_total_earnings = 420 :=
by
  -- Formal proof steps would go here
  sorry

end maria_earnings_correct_l709_709583


namespace find_ellipse_intersection_l709_709034

noncomputable def ellipse_point_of_intersection : Prop :=
  let F1 := (0 : ℝ, 2 : ℝ)
  let F2 := (3 : ℝ, 0 : ℝ)
  ∃ P : ℝ × ℝ, (P ≠ (0, 0) ∧ P.2 = 0 ∧ (dist P F1 + dist P F2 = dist (0, 0) F1 + dist (0, 0) F2)) ∧ P = (15 / 4, 0)

theorem find_ellipse_intersection : ellipse_point_of_intersection :=
sorry

end find_ellipse_intersection_l709_709034


namespace track_length_l709_709760

theorem track_length (x : ℝ) : 
  (∃ B S : ℝ, B + S = x ∧ S = (x / 2 - 75) ∧ B = 75 ∧ S + 100 = x / 2 + 25 ∧ B = x / 2 - 50 ∧ B / S = (x / 2 - 50) / 100) → 
  x = 220 :=
by
  sorry

end track_length_l709_709760


namespace problem_intersection_l709_709480

noncomputable def A (x : ℝ) : Prop := 1 < x ∧ x < 4
noncomputable def B (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem problem_intersection : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

end problem_intersection_l709_709480


namespace match_expression_l709_709317

theorem match_expression (y : ℝ) (hy : y > 0) : 
  3^y * 3^y + 3^y * 3^y = 2 * 3^(2 * y) :=
by
  sorry

end match_expression_l709_709317


namespace probability_B_more_points_than_A_l709_709397

open Nat

theorem probability_B_more_points_than_A 
  (teams : Finset ℕ)
  (h_teams_card : teams.card = 8)
  (h_play_conditions : ∀ (game : ℕ × ℕ), game ∈ teams ×ˢ teams → game.1 ≠ game.2)
  (h_first_game_B_win : ∃ game : ℕ × ℕ, game.1 = 1 ∧ game.2 = 0 ∧ game ∈ teams ×ˢ teams)
  (equal_chance : ∀ game : ℕ × ℕ, game ∈ teams ×ˢ teams → (game.1, game.2 ∈ teams → (0.5)))
  :
  let p := Rat.mk 793 2048 in
  (p = \frac{793}{2048}):
  true :=
sorry

end probability_B_more_points_than_A_l709_709397


namespace geometric_sequence_properties_l709_709946

theorem geometric_sequence_properties (a : ℕ → ℝ) (q : ℝ) :
  a 1 = 1 / 2 ∧ a 4 = -4 → q = -2 ∧ (∀ n, a n = 1 / 2 * q ^ (n - 1)) :=
by
  intro h
  sorry

end geometric_sequence_properties_l709_709946


namespace probability_both_red_l709_709192

-- Define the conditions
def initial_red_balls : ℕ := 2
def initial_yellow_balls : ℕ := 3
def total_balls : ℕ := initial_red_balls + initial_yellow_balls

-- Define the event that both Xiaojun and Xiaojing pick red balls
def both_pick_red_event : Prop :=
  let total_outcomes := (total_balls * (total_balls - 1)) in
  let favorable_outcomes := (initial_red_balls * (initial_red_balls - 1)) in
  (favorable_outcomes / total_outcomes = 1 / 10)

-- The statement to be proven
theorem probability_both_red : both_pick_red_event :=
by
  sorry

end probability_both_red_l709_709192


namespace Maria_soap_cost_l709_709580
-- Import the entire Mathlib library
  
theorem Maria_soap_cost (soap_last_months : ℕ) (cost_per_bar : ℝ) (months_in_year : ℕ):
  (soap_last_months = 2) -> 
  (cost_per_bar = 8.00) ->
  (months_in_year = 12) -> 
  (months_in_year / soap_last_months * cost_per_bar = 48.00) := 
by
  intros h_soap_last h_cost h_year
  sorry

end Maria_soap_cost_l709_709580


namespace vector_angle_120_degrees_l709_709875

-- Define the vector a
def vec_a : (ℝ × ℝ × ℝ) := (1, 0, -1)

-- Candidate vector
def vec_d : (ℝ × ℝ × ℝ) := (-1, 1, 0)

-- Define the dot product of two vectors
def dot_prod (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Establish the Lean theorem to be proved
theorem vector_angle_120_degrees :
  let cos_theta := dot_prod vec_a vec_d / (magnitude vec_a * magnitude vec_d) in
  cos_theta = -1 / 2 :=
sorry

end vector_angle_120_degrees_l709_709875


namespace percentage_conversion_l709_709000

theorem percentage_conversion (p : ℚ) (n : ℚ) (h : p = 1/3) : (33 + p) / 100 * n = 12 :=
by
  have p_val : p = 1 / 3 := h
  rw p_val
  norm_num
  sorry

end percentage_conversion_l709_709000


namespace spatial_analogy_properties_l709_709375

theorem spatial_analogy_properties :
  (∀ (x y z : Line), perpendicular x z → perpendicular y z → parallel x y) →
  (∀ (a b : Line), perpendicular_to_plane a P → perpendicular_to_plane b P → parallel a b) ∧ 
  (∀ (α β : Plane), perpendicular α l → perpendicular β l → parallel α β) := 
by sorry

end spatial_analogy_properties_l709_709375


namespace g_diff_l709_709559

def g (n : ℤ) : ℤ := (1 / 4 : ℚ) * n * (n + 1) * (n + 2) * (n + 3) -- Define g(n) as per the condition

theorem g_diff (r : ℤ) : g(r) - g(r-1) = r * (r + 1) * (r + 2) := by
  sorry

end g_diff_l709_709559


namespace find_cosine_integer_l709_709796

theorem find_cosine_integer (n : ℕ) (h1 : 0 ≤ n ∧ n ≤ 360) (h2 : cos (n * real.pi / 180) = cos (321 * real.pi / 180)) : n = 39 ∨ n = 321 :=
sorry

end find_cosine_integer_l709_709796


namespace y_value_is_32_l709_709137

-- Define the conditions
variables (y : ℝ) (hy_pos : y > 0) (hy_eq : y^2 = 1024)

-- State the theorem
theorem y_value_is_32 : y = 32 :=
by
  -- The proof will be written here
  sorry

end y_value_is_32_l709_709137


namespace find_ab_l709_709557

noncomputable def quadratic_has_complex_conjugate_roots (a b : ℝ) : Prop :=
  ∃ x y : ℝ, let z := x + y * complex.I in
             let conj_z := x - y * complex.I in
             (z + conj_z = -(6 + a * complex.I) ∧
              z * conj_z = 15 + b * complex.I)

theorem find_ab (a b : ℝ) (h : quadratic_has_complex_conjugate_roots a b) : a = 0 ∧ b = 0 :=
sorry

end find_ab_l709_709557


namespace ball_maximum_height_l709_709335
-- Import necessary libraries

-- Define the height function
def ball_height (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 20

-- Proposition asserting that the maximum height of the ball is 145 meters
theorem ball_maximum_height : ∃ t : ℝ, ball_height t = 145 :=
  sorry

end ball_maximum_height_l709_709335


namespace vectors_not_parallel_dot_product_value_l709_709456

noncomputable def α := sorry  -- α is an element of (π/2, π)
def sin_α := √5 / 5
def cos_α := -√(1 - (√5 / 5)^2)
def vec_a := (cos_α, sin_α)
def cos_2α := 1 - 2 * sin_α ^ 2
def sin_2α := 2 * sin_α * cos_α
def vec_b := (cos_2α, sin_2α)

theorem vectors_not_parallel : vec_a.1 * vec_b.2 - vec_a.2 * vec_b.1 ≠ 0 := sorry

theorem dot_product_value : vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = -2 * √5 / 5 := sorry

end vectors_not_parallel_dot_product_value_l709_709456


namespace median_CM_HLM_l709_709954

variable {A B C H L M : Type} 
variable [linear_ordered_field A] [metric_space B] [metric_space C] 
variable [add_comm_group H] 
variable (triangle_ABC : set (A × B × C))
variable (triangle_HLM: set (H × L × M))
variable (altitude_AH_ABC : (∃: A, A ⊥ B) )
variable (bisector_BL_ABC : (∃: B, ∃: L))
variable (median_CM_ABC : (∃: C, ∃: M))
variable (altitude_AH_HLM : (∃: A, A ⊥ L))
variable (bisector_BL_HLM : (∃: B, ∃: L))

theorem median_CM_HLM : median_CM_ABC → median_CM_HLM :=
by
  sorry

end median_CM_HLM_l709_709954


namespace max_distance_is_3_l709_709497

noncomputable def maximum_value_of_distance (z : ℂ) : ℝ :=
  if |z + 1 - complex.I| = 1 then |z - 1 - complex.I| else 0

theorem max_distance_is_3 (z : ℂ) (hz : |z + 1 - complex.I| = 1) : maximum_value_of_distance z = 3 :=
  sorry

end max_distance_is_3_l709_709497


namespace minimum_colors_needed_is_four_l709_709586

/-- 
Misha painted all the integers in several colors such that any two numbers
whose difference is a prime number are painted in different colors.
Prove that the minimum number of colors needed is exactly 4.
-/
theorem minimum_colors_needed_is_four :
  ∀ (coloring : ℤ → ℕ), 
    (∀ (a b : ℤ), prime (abs (a - b)) → coloring a ≠ coloring b) → 
    ∃ (n : ℕ), n = 4 ∧ ∀ (coloring' : ℤ → ℕ), 
    (∀ (a b : ℤ), prime (abs (a - b)) → coloring' a ≠ coloring' b) → (∀ m, m < n → ∃ (numbers : fin n → ℤ), 
    (∀ i j, i ≠ j → coloring' (numbers i) ≠ coloring' (numbers j)) 
    ) := 
sorry

end minimum_colors_needed_is_four_l709_709586


namespace problem_correct_statements_l709_709365

noncomputable def euclidean_algorithm_subtractions (a b : ℕ) : ℕ :=
if a = b then 0
else if a > b then 1 + euclidean_algorithm_subtractions (a - b) b
else euclidean_algorithm_subtractions b a

def swap_values {α : Type} (A B : α) : α × α :=
let X := A in
let A := B in
let B := X in
(A, B)

noncomputable def horner_method (coeffs : List ℤ) (x : ℤ) : List ℤ :=
coeffs.scanl (λ acc a => acc * x + a) 0 |>.tail

theorem problem_correct_statements :
  (euclidean_algorithm_subtractions 295 85 = 12) ∧
  (swap_values 5 8 = (8, 5)) ∧
  (horner_method [12, 35, -8, 79, 6, 5, 3] (-4) !! 2 ≠ -57) ∧
  ∀ (s : Finset ℤ) (a : ℤ), 
    s.Nonempty → a ≠ 0 → 
    let new_set := s.image (λ x => x - a) in
    s.mean = new_set.mean ∧ s.variance = new_set.variance :=
by
  sorry

end problem_correct_statements_l709_709365


namespace upward_distance_to_fred_l709_709010

-- Define Denis's location
def denisLocation : ℝ × ℝ := (8, -24)

-- Define Eliza's location
def elizaLocation : ℝ × ℝ := (-3, 18)

-- Define Fred's location
def fredLocation : ℝ × ℝ := (5 / 2, 5)

-- Define the midpoint function
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Calculate the midpoint of Denis's and Eliza's locations
def meetingPoint : ℝ × ℝ := midpoint denisLocation elizaLocation

-- Define the vertical distance function
def verticalDistance (A B : ℝ × ℝ) : ℝ :=
  abs (B.2 - A.2)

-- Proof statement: the vertical distance walked upwards by Denis and Eliza to reach Fred is 8 units
theorem upward_distance_to_fred : verticalDistance meetingPoint fredLocation = 8 :=
by
  sorry

end upward_distance_to_fred_l709_709010


namespace rect_garden_width_l709_709686

theorem rect_garden_width (w l : ℝ) (h1 : l = 3 * w) (h2 : l * w = 768) : w = 16 := by
  sorry

end rect_garden_width_l709_709686


namespace concurrence_of_lines_l709_709080

theorem concurrence_of_lines
    {A B C D E K M : Type}
    (h1 : (BD / DC) = 3)
    (h2 : (AE / EC) = 1.5)
    (h3 : (BK / KA) = 2) :
    ∃ M, collinear A D M ∧ collinear B E M ∧ collinear C K M :=
sorry

end concurrence_of_lines_l709_709080


namespace indistinguishable_balls_into_boxes_l709_709886

theorem indistinguishable_balls_into_boxes : 
  ∃ n : ℕ, n = 3 ∧ (∀ (b : ℕ), (b = 5 → 
  ∃ (ways : ℕ), ways = 3 ∧ 
  (ways = 1 + 1 + 1 ∧ 
  ((∃ x y : ℕ, x + y = b ∧ (x = 5 ∧ y = 0)) ∧ 
   (∃ x y : ℕ, x + y = b ∧ (x = 4 ∧ y = 1)) ∧ 
   (∃ x y : ℕ, x + y = b ∧ (x = 3 ∧ y = 2)))))) := 
begin
  sorry
end

end indistinguishable_balls_into_boxes_l709_709886


namespace quadrilateral_parallelogram_l709_709279

theorem quadrilateral_parallelogram
  (Q : Type)
  [quadrilateral Q]
  (P₁ P₂ : Q → Q)
  (O : Q)
  (r₁ r₂ : ℝ)
  (h1 : ∀ p₁ p₂ ∈ Q, projections P₁ p₁ lie_on_circle O r₁)
  (h2 : ∀ p₁ p₂ ∈ Q, projections P₂ p₂ lie_on_circle O r₂)
  (h3 : r₁ ≠ r₂) :
  is_parallelogram Q := sorry

end quadrilateral_parallelogram_l709_709279


namespace find_m_l709_709287

-- Definitions of the problem's conditions
def is_singleton (A : set ℝ) : Prop :=
  ∃ x, ∀ y, y ∈ A ↔ y = x

def quadratic_eq_singleton (m : ℝ) : Prop :=
  is_singleton {x | x^2 - 4 * x + m = 0}

-- Statement of the problem
theorem find_m (m : ℝ) : quadratic_eq_singleton m → m = 4 := by
  sorry

end find_m_l709_709287


namespace polar_coordinate_conversion_l709_709521

theorem polar_coordinate_conversion :
  ∃ (r θ : ℝ), (r = 2) ∧ (θ = 11 * Real.pi / 8) ∧ 
    ∀ (r1 θ1 : ℝ), (r1 = -2) ∧ (θ1 = 3 * Real.pi / 8) →
      (abs r1 = r) ∧ (θ1 + Real.pi = θ) :=
by
  sorry

end polar_coordinate_conversion_l709_709521


namespace machine_fill_time_l709_709348

theorem machine_fill_time (filled_cans : ℕ) (time_per_batch : ℕ) (total_cans : ℕ) (expected_time : ℕ)
  (h1 : filled_cans = 150)
  (h2 : time_per_batch = 8)
  (h3 : total_cans = 675)
  (h4 : expected_time = 36) :
  (total_cans / filled_cans) * time_per_batch = expected_time :=
by 
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end machine_fill_time_l709_709348


namespace greatest_prime_factor_of_249_l709_709307

/-- Problem: What is the greatest prime factor of 249? -/
theorem greatest_prime_factor_of_249 : ∃ p : ℕ, p = 19 ∧ (∀ q, nat.prime q ∧ q ∣ 249 → q ≤ p) :=
by
  sorry

end greatest_prime_factor_of_249_l709_709307


namespace find_softball_players_l709_709934

def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def football_players : ℕ := 18
def total_players : ℕ := 59

theorem find_softball_players :
  let C := cricket_players
  let H := hockey_players
  let F := football_players
  let T := total_players
  S = T - (C + H + F) :=
by
  let C := cricket_players
  let H := hockey_players
  let F := football_players
  let T := total_players
  show S = T - (C + H + F)
  sorry

end find_softball_players_l709_709934


namespace number_of_incorrect_statements_l709_709938

variable {A B C a b c : ℝ}

-- Given conditions rephrased in Lean
def is_obtuse_triangle (A B C : ℝ) : Prop := A + B + C = π ∧ (A > π / 2 ∨ B > π / 2 ∨ C > π / 2)
def is_acute_triangle (A B C : ℝ) : Prop := A + B + C = π ∧ A < π / 2 ∧ B < π / 2 ∧ C < π / 2

def condition1 (A B C : ℝ) : Prop := is_obtuse_triangle A B C → tan A + tan B + tan C > 0
def condition2 (A B C : ℝ) : Prop := is_acute_triangle A B C → cos A + cos B > sin A + sin B
def condition3 (A B : ℝ) : Prop := A < B → cos (sin A) < cos (tan B)
def condition4 (A B C : ℝ) : Prop := sin B = 2/5 ∧ tan C = 3/4 → A > C ∧ C > B

-- Complete proof problem statement in Lean
theorem number_of_incorrect_statements :
  ∃ A B C, ¬ condition1 A B C ∧ ¬ condition2 A B C ∧ ¬ condition3 A B ∧ condition4 A B C :=
sorry

end number_of_incorrect_statements_l709_709938


namespace smallest_number_l709_709030

theorem smallest_number (a b c d : ℝ) (h1 : a = sqrt 2) (h2 : b = 0) (h3 : c = -1) (h4 : d = 2) :
  (∀ x ∈ {a, b, c, d}, c ≤ x) :=
by {
  -- sorry to skip the proof
  sorry
}

end smallest_number_l709_709030


namespace volume_prism_l709_709691

noncomputable def volume_inclined_triangular_prism (S d : ℝ) : ℝ :=
  1 / 2 * S * d

theorem volume_prism (S d : ℝ) : volume_inclined_triangular_prism S d = 1 / 2 * S * d :=
by
  rw volume_inclined_triangular_prism
  sorry

end volume_prism_l709_709691


namespace water_speed_l709_709012

theorem water_speed (v : ℝ) (h1 : 4 - v > 0) (h2 : 6 * (4 - v) = 12) : v = 2 :=
by
  -- proof steps
  sorry

end water_speed_l709_709012


namespace area_of_bounded_region_l709_709275

-- Define the equation as a condition.
def eqn (x y : ℝ) : Prop := y^2 + 2 * x * y + 40 * |x| = 400

-- State the theorem.
theorem area_of_bounded_region :
  let bounded_region_area := 800 in
  ∃ (vertices : list (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ vertices → eqn x y) ∧ 
    (bounded_region_area = 800) :=
by -- proof goes here
  sorry

end area_of_bounded_region_l709_709275


namespace triangle_ABC_is_isosceles_l709_709593

open EuclideanGeometry

-- Define the problem context
variables {A B C P Q : Point}

-- Translate the conditions into Lean definitions
axiom triangle_ABC : Triangle A B C
axiom point_P_on_BC : P ∈ line (B, C)
axiom angle_PAB_45 : ∠ P A B = 45
axiom Q_on_perpendicular_bisector : ¬Colinear A P Q ∧ (Q ∈ line (A, C) ∧ dist A Q = dist P Q)
axiom PQ_perpendicular_BC : Perpendicular (segment P Q) (line (B, C))

-- Prove the desired theorem
theorem triangle_ABC_is_isosceles : Isosceles_triangle A B C :=
by
  sorry

end triangle_ABC_is_isosceles_l709_709593


namespace speed_of_faster_train_l709_709299

variables (L t f : ℝ)

-- Given conditions
def length_of_each_train := L = 150
def time_to_cross := t = 18
def speed_factor := f = 3

-- Proving the speed of the faster train is 12.5 m/s
theorem speed_of_faster_train 
  (hL : length_of_each_train L)
  (ht : time_to_cross t)
  (hf : speed_factor f) :
  let v := (150 / (18 * 4)) in
  3 * v = 12.5 :=
by
  sorry

end speed_of_faster_train_l709_709299


namespace b_arithmetic_a_formula_l709_709995

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l709_709995


namespace age_difference_between_mandy_and_sister_l709_709576

def mandy_age : ℕ := 3
def brother_age (mandy_age : ℕ) : ℕ := 4 * mandy_age
def sister_age (brother_age : ℕ) : ℕ := brother_age - 5

theorem age_difference_between_mandy_and_sister : 
  let mandy := mandy_age in
  let brother := brother_age mandy in
  let sister := sister_age brother in
  (sister - mandy) = 4 := 
by 
  sorry

end age_difference_between_mandy_and_sister_l709_709576


namespace complex_number_in_second_quadrant_l709_709843

theorem complex_number_in_second_quadrant (z : ℂ) (h : (1 - complex.I) * z = 2 + 3 * complex.I) :
  z.re < 0 ∧ z.im > 0 :=
by
  have : z = (2 + 3 * complex.I) / (1 - complex.I) := by sorry
  rw this
  rw complex.div_eq_mul_inv
  rw complex.inv_def
  rw complex.mul_assoc
  rw complex.mul_conj
  rw complex.of_real_mul_re
  rw complex.of_real_mul_im
  rw complex.I_mul_I
  rw add_sub_cancel
  rw complex.of_real_one
  sorry

end complex_number_in_second_quadrant_l709_709843


namespace problem_1_problem_2_l709_709699

-- Declaration of the function f and its properties
variable {f : ℝ → ℝ}
variable (h_decreasing : ∀ x y : ℝ, x < y → f(x) > f(y))
variable (h_functional_eq : ∀ x y : ℝ, f(x * y) = f(x) + f(y))

-- Problem 1: Proof that f(1) = 0
theorem problem_1 : f(1) = 0 :=
by
  -- Add necessary content here
  sorry

-- Problem 2: Proof that x > 2 if f(2x - 3) < 0
theorem problem_2 (x : ℝ) (h_ineq : f (2 * x - 3) < 0) : x > 2 :=
by
  -- Add necessary content here
  sorry

end problem_1_problem_2_l709_709699


namespace numeral_in_150th_place_l709_709670

theorem numeral_in_150th_place : 
  let s := "384615"
  ∃ n : ℕ, s.length = 6 ∧ (150 % 6 = 0) ∧ n = 6 ∧ (s.get (n-1) = '5') →
  (s[(150 % 6)-1] = '5') :=
by
  sorry

end numeral_in_150th_place_l709_709670


namespace problem_l709_709703

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x + c

def f_prime (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + 3 * b

theorem problem (a b c : ℝ) (h1 : f_prime a b 2 = 0) (h2 : f_prime a b 1 = -3) :
  a = -1 ∧ b = 0 ∧ (let f_min := f (-1) 0 c 2 
                   let f_max := 0 
                   f_max - f_min = 4) :=
by
  sorry

end problem_l709_709703


namespace sin_C_and_area_of_triangle_l709_709533

open Real

noncomputable section

theorem sin_C_and_area_of_triangle 
  (A B C : ℝ)
  (cos_A : Real := sqrt 3 / 3)
  (a b c : ℝ := (3 * sqrt 2)) 
  (cosA : cos A = sqrt 3 / 3)
  -- angles in radians, use radians for the angles when proving
  (side_c : c = sqrt 3)
  (side_a : a = 3 * sqrt 2) :
  (sin C = 1 / 3) ∧ (1 / 2 * a * b * sin C = 5 * sqrt 6 / 3) :=
by
  sorry

end sin_C_and_area_of_triangle_l709_709533


namespace catch_up_time_l709_709290

noncomputable def speed_ratios (v : ℝ) : Prop :=
  let a_speed := (4 / 5) * v
  let b_speed := (2 / 5) * v
  a_speed = 2 * b_speed

theorem catch_up_time (v t : ℝ) (a_speed b_speed : ℝ)
  (h1 : a_speed = (4 / 5) * v)
  (h2 : b_speed = (2 / 5) * v)
  (h3 : a_speed = 2 * b_speed) :
  (t = 11) := by
  sorry

end catch_up_time_l709_709290


namespace find_b_for_perpendicular_lines_l709_709276

theorem find_b_for_perpendicular_lines 
  (b : ℚ) : 
  (∀ (x y : ℚ), 2*x - 3*y + 6 = 0 → b*x - 3*y + 6 = 0 → (2/3) * (b/3) = -1) → b = -9/2 :=
by
  intro h
  sorry

end find_b_for_perpendicular_lines_l709_709276


namespace number_of_teams_l709_709587

theorem number_of_teams :
  let boys := 7 in
  let girls := 10 in
  let team_boys := 4 in
  let team_girls := 4 in
  (nat.choose boys team_boys) * (nat.choose girls team_girls) = 7350 :=
by
  let boys := 7;
  let girls := 10;
  let team_boys := 4;
  let team_girls := 4;
  have h1 : nat.choose boys team_boys = 35 := by sorry;
  have h2 : nat.choose girls team_girls = 210 := by sorry;
  calc (nat.choose boys team_boys) * (nat.choose girls team_girls)
      = 35 * 210 : by rw [h1, h2]
  ... = 7350 : by norm_num

end number_of_teams_l709_709587


namespace y_value_is_32_l709_709136

-- Define the conditions
variables (y : ℝ) (hy_pos : y > 0) (hy_eq : y^2 = 1024)

-- State the theorem
theorem y_value_is_32 : y = 32 :=
by
  -- The proof will be written here
  sorry

end y_value_is_32_l709_709136


namespace car_actual_time_l709_709710

-- Definitions and assumptions
variable (S T : ℝ)  -- S is actual speed, T is actual time

-- The car is 15 minutes late running at 4/5 of its actual speed
axiom h1 : (5/4:ℝ * (T + 15)) = T

-- Goal: to prove that the actual time T is 60 minutes
theorem car_actual_time (T : ℝ) (h1 : (5/4 * T + 15) = T) : T = 60 :=
by {
  -- proof steps would go here
  sorry
}

end car_actual_time_l709_709710


namespace arithmetic_sequence_and_formula_l709_709988

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709988


namespace Indians_drink_tea_is_zero_l709_709179

-- Definitions based on given conditions and questions
variable (total_people : Nat)
variable (total_drink_tea : Nat)
variable (total_drink_coffee : Nat)
variable (answer_do_you_drink_coffee : Nat)
variable (answer_are_you_a_turk : Nat)
variable (answer_is_it_raining : Nat)
variable (Indians_drink_tea : Nat)
variable (Indians_drink_coffee : Nat)
variable (Turks_drink_coffee : Nat)
variable (Turks_drink_tea : Nat)

-- The given facts and conditions
axiom hx1 : total_people = 55
axiom hx2 : answer_do_you_drink_coffee = 44
axiom hx3 : answer_are_you_a_turk = 33
axiom hx4 : answer_is_it_raining = 22
axiom hx5 : Indians_drink_tea + Indians_drink_coffee + Turks_drink_coffee + Turks_drink_tea = total_people
axiom hx6 : Indians_drink_coffee + Turks_drink_coffee = answer_do_you_drink_coffee
axiom hx7 : Indians_drink_coffee + Turks_drink_tea = answer_are_you_a_turk
axiom hx8 : Indians_drink_tea + Turks_drink_coffee = answer_is_it_raining

-- Prove that the number of Indians drinking tea is 0
theorem Indians_drink_tea_is_zero : Indians_drink_tea = 0 :=
by {
    sorry
}

end Indians_drink_tea_is_zero_l709_709179


namespace kiana_and_her_siblings_age_sum_l709_709546

theorem kiana_and_her_siblings_age_sum :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 256 ∧ a + b + c = 38 :=
by
sorry

end kiana_and_her_siblings_age_sum_l709_709546


namespace indistinguishable_balls_boxes_l709_709890

noncomputable def ways_to_distribute (balls : ℕ) (boxes : ℕ) : ℕ :=
  if boxes = 2 then
    if balls = 5 then
      3
    else
      sorry  -- Not needed for this problem
  else 
    sorry  -- Not needed for this problem

theorem indistinguishable_balls_boxes
  (balls : ℕ) (boxes : ℕ) : 
  boxes = 2 → balls = 5 → ways_to_distribute balls boxes = 3 :=
by
  intros h1 h2
  simp [ways_to_distribute, h1, h2]
  sorry

end indistinguishable_balls_boxes_l709_709890


namespace arc_length_of_octagon_side_l709_709015

-- Define the conditions
def is_regular_octagon (side_length : ℝ) (angle_subtended : ℝ) := side_length = 5 ∧ angle_subtended = 2 * Real.pi / 8

-- Define the property to be proved
theorem arc_length_of_octagon_side :
  ∀ (side_length : ℝ) (angle_subtended : ℝ), 
    is_regular_octagon side_length angle_subtended →
    (angle_subtended / (2 * Real.pi)) * (2 * Real.pi * side_length) = 5 * Real.pi / 4 :=
by
  intros side_length angle_subtended h
  unfold is_regular_octagon at h
  sorry

end arc_length_of_octagon_side_l709_709015


namespace sufficient_but_not_necessary_l709_709904

theorem sufficient_but_not_necessary (a b : ℝ) : (a > b ∧ b > 0) → (a^2 > b^2) ∧ ¬((a^2 > b^2) → (a > b ∧ b > 0)) :=
by
  sorry

end sufficient_but_not_necessary_l709_709904


namespace min_value_is_five_l709_709454

noncomputable def min_value (x y : ℝ) : ℝ :=
  if x + 3 * y = 5 * x * y then 3 * x + 4 * y else 0

theorem min_value_is_five {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : min_value x y = 5 :=
by
  sorry

end min_value_is_five_l709_709454


namespace grid_divisible_by_rectangles_l709_709812

theorem grid_divisible_by_rectangles (n : ℕ) :
  (∃ m : ℕ, n * n = 7 * m) ↔ (∃ k : ℕ, n = 7 * k ∧ k > 1) :=
by
  sorry

end grid_divisible_by_rectangles_l709_709812


namespace triangular_pyramid_correct_statements_l709_709448

variables (P A B C : Point)
variables (hPA_perp_PB : ⊥ PA PB) (hPB_perp_PC : ⊥ PB PC) (hPC_perp_PA : ⊥ PC PA)

theorem triangular_pyramid_correct_statements :
  (⊥ PA (BC) ∧ ⊥ PB (AC) ∧ ⊥ PC (AB)) ∧
  (perpendicular_height_to_orthocenter P A B C) ∧
  (midline_intersections P A B C) := 
sorry

end triangular_pyramid_correct_statements_l709_709448


namespace least_n_for_perfect_square_l709_709414

theorem least_n_for_perfect_square (n : ℕ) :
  (∀ m : ℕ, 2^8 + 2^11 + 2^n = m * m) → n = 12 := sorry

end least_n_for_perfect_square_l709_709414


namespace min_value_proof_l709_709071

noncomputable def min_value : ℝ :=
  7 + 4 * Real.sqrt 3

theorem min_value_proof {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  ∀ z, z = (3 / x + 4 / y) → z ≥ min_value :=
begin
  sorry
end

end min_value_proof_l709_709071


namespace max_irrationals_2x2019_grid_l709_709706

open Set

theorem max_irrationals_2x2019_grid :
  ∃ (a b : Fin 2019 → ℝ),
    (∀ i j : Fin 2019, i ≠ j → a i ≠ a j) ∧      -- Top row distinct
    (∃ σ : Equiv.Perm (Fin 2019),                 -- Permutation σ exists
      (∀ i : Fin 2019, a i + b (σ i) ∈ ℚ) ∧       -- Column sums are rational
      (∃ n, (∀ i, a i ∈ ℚ ∨ n ≤ i) ∧ n = 2016))   -- Maximum 2016 irrationals in top row
:= sorry

end max_irrationals_2x2019_grid_l709_709706


namespace total_fish_1148_l709_709259

def Jerk_Tuna_tuna := 144
def Jerk_Tuna_mackerel := 80

def Tall_Tuna_tuna := 2 * Jerk_Tuna_tuna
def Tall_Tuna_mackerel := Jerk_Tuna_mackerel + (0.3 * Jerk_Tuna_mackerel)

def Swell_Tuna_tuna := Tall_Tuna_tuna + (0.5 * Tall_Tuna_tuna)
def Swell_Tuna_mackerel := Jerk_Tuna_mackerel + (0.25 * Jerk_Tuna_mackerel)

def Jerk_Tuna_fish := Jerk_Tuna_tuna + Jerk_Tuna_mackerel
def Tall_Tuna_fish := Tall_Tuna_tuna + Tall_Tuna_mackerel
def Swell_Tuna_fish := Swell_Tuna_tuna + Swell_Tuna_mackerel

def total_fish := Jerk_Tuna_fish + Tall_Tuna_fish + Swell_Tuna_fish

theorem total_fish_1148 : total_fish = 1148 := by {
  sorry
}

end total_fish_1148_l709_709259


namespace average_of_remaining_two_numbers_l709_709263

theorem average_of_remaining_two_numbers (S S3 : ℝ) (h_avg5 : S / 5 = 8) (h_avg3 : S3 / 3 = 4) : S / 5 = 8 ∧ S3 / 3 = 4 → (S - S3) / 2 = 14 :=
by 
  sorry

end average_of_remaining_two_numbers_l709_709263


namespace average_age_l709_709778

theorem average_age (Devin_age Eden_age mom_age : ℕ)
  (h1 : Devin_age = 12)
  (h2 : Eden_age = 2 * Devin_age)
  (h3 : mom_age = 2 * Eden_age) :
  (Devin_age + Eden_age + mom_age) / 3 = 28 := by
  sorry

end average_age_l709_709778


namespace find_a_for_even_function_l709_709149

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, (x-1)^2 + a * x + sin (x + π / 2) = ((-x)-1)^2 + (-a * x) + sin (-x + π / 2)) →
  a = 2 :=
by
  sorry

end find_a_for_even_function_l709_709149


namespace angle_equivalence_l709_709826

-- Definitions of points, circle, and angles
variables {O A B C D E : Type} [Inhabited O] [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
variables {C_ O_ : Circle O} -- Circle C centered at O
variables {ray_A_ BC : Ray A B} -- Ray from A intersecting the circle at B and C
variables {ray_A_ DE : Ray A D} -- Ray from A intersecting the circle at D and E

-- Angles defined from points
variable {angle_CA_ED : Angle A C E}
variable {angle_COE : Angle O C E}
variable {angle_BOD : Angle O B D}

-- Proof statement
theorem angle_equivalence : 
  ∀ (C_ : Circle O) (A B C D E : Point) (ray_A_BC ray_A_DE : Ray A)
  (angle_CA_ED : Angle A C E) (angle_COE : Angle O C E) (angle_BOD : Angle O B D),
  (angle_CA_ED = (angle_COE - angle_BOD) / 2) :=
by
  sorry

end angle_equivalence_l709_709826


namespace total_appetizers_l709_709036

theorem total_appetizers (hotdogs cheese_pops chicken_nuggets mini_quiches stuffed_mushrooms total_portions : Nat)
  (h1 : hotdogs = 60)
  (h2 : cheese_pops = 40)
  (h3 : chicken_nuggets = 80)
  (h4 : mini_quiches = 100)
  (h5 : stuffed_mushrooms = 50)
  (h6 : total_portions = hotdogs + cheese_pops + chicken_nuggets + mini_quiches + stuffed_mushrooms) :
  total_portions = 330 :=
by sorry

end total_appetizers_l709_709036


namespace ratio_new_to_original_l709_709715

axiom jewels_original : ℕ
axiom jewels_stolen : ℕ := 3
axiom jewels_taken_back : jewels_stolen * 2
axiom jewels_remaining : jewels_original - jewels_stolen + jewels_taken_back = 24
axiom jewels_new := jewels_taken_back

theorem ratio_new_to_original (J : ℕ) (H : J - 3 + 6 = 24) :
  (6 : ℚ) / (J : ℚ) = 2 / 7 := sorry

end ratio_new_to_original_l709_709715


namespace line_through_center_parallel_to_l1_l709_709074

/-!
# Problem Statement
Given a circle C: (x - 1)^2 + (y + 2)^2 = 5 and a line l1: 2x - 3y + 6 = 0,
prove that the equation of the line l that is parallel to l1 and passes through
the center of circle C is 2x - 3y - 8 = 0.
-/

def circle : ℝ → ℝ → Prop :=
  λ x y, (x - 1)^2 + (y + 2)^2 = 5

def line_l1 : ℝ → ℝ → Prop :=
  λ x y, 2 * x - 3 * y + 6 = 0

theorem line_through_center_parallel_to_l1 :
  ∃ (l : ℝ → ℝ → Prop), (∀ x y, line_l1 x y → (∃ ky, l x y ↔ y = ky)) ∧ 
  (∀ x y, circle x y → l x y) ∧
  (l = λ x y, 2 * x - 3 * y - 8 = 0) :=
by 
  sorry

end line_through_center_parallel_to_l1_l709_709074


namespace find_value_of_expression_l709_709920

theorem find_value_of_expression (m n : ℝ) (h : |m - n - 5| + (2 * m + n - 4)^2 = 0) : 3 * m + n = 7 := 
sorry

end find_value_of_expression_l709_709920


namespace sum_of_lowest_scores_l709_709252

noncomputable def test_scores_sum (scores : List ℕ) := scores.sum

theorem sum_of_lowest_scores (scores : List ℕ) (Hlength : scores.length = 6) 
  (Hmean : (∑ x in scores, x) / 6 = 92)
  (Hmedian : (scores.nth_le 2 (by simp)) + (scores.nth_le 3 (by simp)) = 186)
  (Hmode : (∃ n, scores.count n > 1 ∧ n = 94)) :
  (scores.nth_le 0 (by simp)) + (scores.nth_le 1 (by simp)) = 178 := 
sorry

end sum_of_lowest_scores_l709_709252


namespace intervals_of_decrease_tangent_line_at_minus_two_l709_709097

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + 1

theorem intervals_of_decrease :
  (∀ x, x < -1 ∨ x > 3 → f' x < 0) :=
sorry

theorem tangent_line_at_minus_two :
  ∃ k y₀, k = 15 ∧ y₀ = 3 ∧ ∀ x y, y - y₀ = k * (x + 2) ↔ y = -15 * x + 27 :=
sorry

end intervals_of_decrease_tangent_line_at_minus_two_l709_709097


namespace problem_a_problem_b_problem_c_l709_709508

open Finset

-- Part (a)
theorem problem_a : (card {s : finset (fin 10) | s.card = 2 ∧ (∃ i j, i ≠ j ∧
  (s = {i, j} ∧ (i + 5) % 10 = j ∨ (j + 5) % 10 = i))}) / (choose 10 2) = 1 / 9 := sorry

-- Part (b)
theorem problem_b : (card {s : finset (fin 10) | s.card = 3 ∧ (∃ i j k, 
  (s = {i, j, k}) ∧ (i ≠ j) ∧ (j ≠ k) ∧ (k ≠ i) ∧ 
  (i + 5) % 10 ≠ j ∧ (j + 5) % 10 ≠ k ∧ (k + 5) % 10 ≠ i ∧ 
  (((i + 5) % 10 = k ∨ (k + 5) % 10 = i ∨ (j + 5) % 10 = i) ∨ 
  ((i - 5) % 10 = j ∨ (j - 5) % 10 = k ∨ (k - 5) % 10 = i)) )}) / 
  (choose 10 3) = 1 / 3 := sorry

-- Part (c)
theorem problem_c : (card {s : finset (fin 10) | s.card = 4 ∧ (∃ i j k l, 
  i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧ (s = {i, j, k, l} ∧ 
  (({i, j} ∪ {k, l} ⊆ {0, 5}) ∨ ({i, k} ∪ {j, l} ⊆ {1, 6}) ∨ 
  ({i, l} ∪ {j, k} ⊆ {2, 7}) ∨ ({i, j} ∪ {k, l} ⊆ {3, 8}) ∨ 
  ({i, k} ∪ {j, l} ⊆ {4, 9})))}) / (choose 10 4) = 1 / 21 := sorry

end problem_a_problem_b_problem_c_l709_709508


namespace day_of_week_l709_709260

theorem day_of_week (h : ∀ d, (d - 20) % 7 = 0 → day_of_week d = "Wednesday") : day_of_week 5 = "Tuesday" := 
by
  sorry

end day_of_week_l709_709260


namespace additional_workers_needed_l709_709396

theorem additional_workers_needed
  (people_total : ℕ) -- total people who can complete the painting
  (hours_total : ℕ) -- total hours taken by the total people to complete the painting
  (people_initial : ℕ) -- initial people working
  (hours_initial : ℕ) -- initial hours worked by the initial people
  (hours_remaining : ℕ) -- remaining hours for completion)
  (hrs_eq_four : hours_total = 4)
  (people_eq_eight : people_total = 8)
  (people_initial_eq_six : people_initial = 6)
  (hours_initial_eq_two : hours_initial = 2)
  (hours_remaining_eq_two : hours_remaining = 2) : -- remaining hours which are 2
  (required_additional_workers : ℕ) -- number of required additional workers
  (required_additional_workers = (people_total * hours_total - people_initial * hours_initial) / hours_remaining - people_initial) :=
begin
  -- Sorry proof term will be inserted here
  sorry
end

end additional_workers_needed_l709_709396


namespace solve_for_y_l709_709132

theorem solve_for_y (y : ℝ) (h_pos : y > 0) (h_eq : y^2 = 1024) : y = 32 := 
by
  sorry

end solve_for_y_l709_709132


namespace minimum_N_l709_709977

open Set Nat Function

theorem minimum_N (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ N : ℕ,
  (∃ S : Set ℤ, (∀ x : ℤ, ∃ y ∈ S, (y % m = x % m)) ∧ ((S.card : ℕ) = N) ) ∧
  (∃ A ⊆ S, (A ≠ ∅) ∧ (n ∣ A.sum id)) ∧
  N = (let d := gcd m n in
       let a := m / d in
       let b := n / d in
       b * d - (a * d * (d - 1) / 2)) :=
sorry

end minimum_N_l709_709977


namespace true_inverse_of_original_l709_709031

-- Define the original proposition
def original_proposition (x y : ℝ) : Prop := x > y → x > |y|

-- Define the inverse of the original proposition
def inverse_proposition (x y : ℝ) : Prop := x ≤ |y| → x ≤ y

-- State the theorem
theorem true_inverse_of_original : ∀ (x y : ℝ), inverse_proposition x y :=
by
  sorry

end true_inverse_of_original_l709_709031


namespace period_of_f_mono_increasing_intervals_max_value_of_f_l709_709568
noncomputable theory
open Real

def f (x : ℝ) : ℝ := 2 * cos x * (cos x + sqrt 3 * sin x)

theorem period_of_f : ∀ x : ℝ, f (x + π) = f x :=
by sorry

theorem mono_increasing_intervals : ∀ k : ℤ, ∀ x y : ℝ, (k * π - π / 3) < x ∧ y < (k * π + π / 6) ∧ x < y → f x < f y :=
by sorry

theorem max_value_of_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f x ≤ 3 :=
by sorry

end period_of_f_mono_increasing_intervals_max_value_of_f_l709_709568


namespace height_of_house_proof_l709_709758

/-- height_of_house_proof -/
theorem height_of_house_proof
  (tree1_height tree2_height house_shadow shadow1_length shadow2_length: ℝ)
  (ratio : ℝ) 
  (h1 : tree1_height = 25)
  (h2 : tree2_height = 15)
  (shadow1_eq : shadow1_length = 30)
  (shadow2_eq : shadow2_length = 18)
  (h_ratio1 : shadow1_length / tree1_height = ratio)
  (h_ratio2 : shadow2_length / tree2_height = ratio)
  (house_shadow_eq : house_shadow = 60)
:
  let house_height := (house_shadow / (shadow1_length / tree1_height)) * tree1_height in
  house_height = 50 :=
by
  -- Sorry to halt proof details.
  sorry

end height_of_house_proof_l709_709758


namespace range_of_a_l709_709169

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (1/2)^(x^2 - 2*a*x) < 2^(3*x + a^2)) → a > 3/4 :=
sorry

end range_of_a_l709_709169


namespace curve_symmetric_reflection_l709_709268

theorem curve_symmetric_reflection (f : ℝ → ℝ → ℝ) :
  (∀ x y, f x y = 0 ↔ f (y + 3) (x - 3) = 0) → 
  (∀ x y, (x - y - 3 = 0) → (f (y + 3) (x - 3) = 0)) :=
sorry

end curve_symmetric_reflection_l709_709268


namespace main_problem_l709_709974

open EuclideanGeometry

-- Define the conditions given in the problem
structure ProblemConditions where
  A B C D O E F G A1 B1 C1 D1 K L : Point
  ω : circle
  cond1 : CyclicQuadrilateral ω O A B C D
  cond2 : Intersect AB CD E
  cond3 : Intersect AD BC F
  cond4 : Intersect AC BD G
  cond5 : OnRay G A A1 ∧ OnRay G B B1 ∧ OnRay G C C1 ∧ OnRay G D D1
  cond6 : (GA_1_ratio : (dist G A1) / (dist G A) = (dist G B1) / (dist G B)) = ((dist G C1) / (dist G C) = (dist G D1) / (dist G D))
  cond7 : OnCircumcircle ω A1 B1 C1 D1 O
  cond8 : Intersect A1B1 C1D1 K
  cond9 : Intersect A1D1 B1C1 L

-- Define what needs to be proven
def ProblemResult (cond : ProblemConditions) : Prop :=
  let invertedCircle := inversionAboutCircle ω (circleThroughPts cond.A1 cond.B1 cond.C1 cond.D1)
  let lineThroughMidpoints := lineThroughMidpoints (midpoint cond.K cond.E) (midpoint cond.L cond.F)
  ∃ invCircle : Line, 
      invertedCircle = invCircle ∧
      ∀ M, M ∈ lineThroughMidpoints → M ∈ invCircle
⟩ 

-- The main theorem statement
theorem main_problem (cond : ProblemConditions) : ProblemResult cond :=
  sorry

end main_problem_l709_709974


namespace _l709_709733

-- Defining that ABCD is a square
structure Square (A B C D : Point) : Prop :=
(is_square : ∃ x : ℕ, distance A B = x ∧ distance B C = x ∧ distance C D = x ∧ distance D A = x)

-- Definition of the Figure F
def Figure_F (A B C D : Point) (div : Π i j : ℕ, (i, j) ∈ Finset.range x × Finset.range x → Rectangle) : Set Rectangle :=
{ r | ∃ i j, (i, j) ∈ Finset.range x × Finset.range x ∧ r = div i j ∧ r ∩ diagonal A C ≠ ∅ }

-- Main theorem statement
example
  (A B C D : Point)
  (x : ℕ) -- side length of the square, and coordinate range
  (div : Π i j : ℕ, (i, j) ∈ Finset.range x × Finset.range x → Rectangle)
  (h1 : Square A B C D)
  (h2 : ∀ i j : ℕ, (i, j) ∈ Finset.range x × Finset.range x → side_lengths (div i j) = (1, 1)) :
  divides_area_exactly (diagonal A C) (Figure_F A B C D div) :=
by
  sorry

end _l709_709733


namespace maria_total_earnings_l709_709582

-- Definitions of the conditions
def day1_tulips := 30
def day1_roses := 20
def day2_tulips := 2 * day1_tulips
def day2_roses := 2 * day1_roses
def day3_tulips := day2_tulips / 10
def day3_roses := 16
def tulip_price := 2
def rose_price := 3

-- Definition of the total earnings calculation
noncomputable def total_earnings : ℤ :=
  let total_tulips := day1_tulips + day2_tulips + day3_tulips
  let total_roses := day1_roses + day2_roses + day3_roses
  (total_tulips * tulip_price) + (total_roses * rose_price)

-- The proof statement
theorem maria_total_earnings : total_earnings = 420 := by
  sorry

end maria_total_earnings_l709_709582


namespace groupD_same_function_l709_709746

def f (x : ℝ) : ℝ := log2 (2^x)
def g (x : ℝ) : ℝ := (x^3)^(1/3)

theorem groupD_same_function : ∀ x : ℝ, f x = g x :=
by
  assume x : ℝ
  -- Since (2^x) is always greater than zero for any real x,
  -- log2(2^x) translates to x because logarithm base 2 of 2^x is x
  -- Also, x^3 raised to the 1/3 power is just x
  sorry -- Proof to be completed

end groupD_same_function_l709_709746


namespace intersection_point_l709_709333

-- Define the line with its parametric equations
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (-1 - 3 * t, -2 + 2 * t, 3 - 2 * t)

-- Define the plane equation
def plane (x y z : ℝ) : Prop :=
  x + 3 * y - 5 * z + 9 = 0

-- Statement of proof
theorem intersection_point :
  ∃ t : ℝ, plane (line t).1 (line t).2 (line t).3 ∧ line t = (-4, 0, 1) :=
sorry

end intersection_point_l709_709333


namespace find_minimum_value_of_quadratic_l709_709426

theorem find_minimum_value_of_quadratic :
  ∀ (x : ℝ), (x = 5/2) -> (∀ y, y = 3 * x ^ 2 - 15 * x + 7 -> ∀ z, z ≥ y) := 
sorry

end find_minimum_value_of_quadratic_l709_709426


namespace jenny_kenny_time_to_see_each_other_again_l709_709203

theorem jenny_kenny_time_to_see_each_other_again :
  ∀ (d_path parallel_lines_speed kenny_speed jenny_speed building_diameter radius t a b : ℝ),
    d_path = 300 ∧ parallel_lines_speed = (4, 1) ∧ building_diameter = 200 ∧ radius = building_diameter / 2 ∧
    t = 240 / 5 ∧ (a, b) = (240, 5) →
    t * b = a ∧ (t = a / b) → (t = 48) ∧ (a + b = 245) :=
by
  intros
  sorry

end jenny_kenny_time_to_see_each_other_again_l709_709203


namespace dishes_with_lentils_l709_709360

theorem dishes_with_lentils (total_dishes: ℕ) (beans_and_lentils: ℕ) (beans_and_seitan: ℕ) 
    (only_one_kind: ℕ) (only_beans: ℕ) (only_seitan: ℕ) (only_lentils: ℕ) :
    total_dishes = 10 →
    beans_and_lentils = 2 →
    beans_and_seitan = 2 →
    only_one_kind = (total_dishes - beans_and_lentils - beans_and_seitan) →
    only_beans = (only_one_kind / 2) →
    only_beans = 3 * only_seitan →
    only_lentils = (only_one_kind - only_beans - only_seitan) →
    total_dishes_with_lentils = beans_and_lentils + only_lentils →
    total_dishes_with_lentils = 4 :=
begin
  intros,
  sorry
end

end dishes_with_lentils_l709_709360


namespace smallest_sum_k_column_largest_sum_k_column_l709_709219

-- Definitions and assumptions
variables (k n : ℕ)
assume (h : k ≤ n)

-- Smallest Sum
theorem smallest_sum_k_column (h : k ≤ n) : 
  let table := matrix (fin n) (fin n) ℕ in 
  -- Assuming a specific table definition that follows the desired properties
  (table : list (list ℕ) := list.range (n^2)) in 
  (∑ (i : fin n), table[i, k] = k * (n * (n + 1)) / 2) :=
sorry

-- Largest Sum
theorem largest_sum_k_column (h : k ≤ n) : 
  let table := matrix (fin n) (fin n) ℕ in 
  -- Assuming a specific table definition that follows the desired properties
  (table : list (list ℕ) := list.range (n^2).reverse) in 
  (∑ (i : fin n), table[i, k] = (n * ((n - 1)^2 + k * (n + 1))) / 2) :=
sorry

end smallest_sum_k_column_largest_sum_k_column_l709_709219


namespace second_smallest_number_in_set_is_4_l709_709638

variable {y x : ℕ}
variable {S : Finset ℕ} -- The set S containing 6 numbers

-- Conditions
def condition_one : S.card = 6 := sorry
def condition_two : 4 ∈ S := sorry
def condition_three : y ∈ S := sorry
def condition_four : 710 ∈ S := sorry
def condition_five : range (S) = 12 := sorry
def condition_six : |(x.max' S) - (x.min' S)| = 13 := sorry

-- Theorem to prove
theorem second_smallest_number_in_set_is_4 (h1 : condition_one) (h2 : condition_two) (h3 : condition_three) 
     (h4 : condition_four) (h5 : condition_five) (h6 : condition_six) :
  (S.erase (S.min' (finset_nonempty_of_card_eq_succ h1))).min' (finset_nonempty_of_card_eq_succ (Finset.card_erase_lt (S.min' _))) = 4 :=
sorry

end second_smallest_number_in_set_is_4_l709_709638


namespace original_number_of_members_l709_709516

-- Define the initial conditions
variables (x y : ℕ)

-- First condition: if five 9-year-old members leave
def condition1 : Prop := x * y - 45 = (y + 1) * (x - 5)

-- Second condition: if five 17-year-old members join
def condition2 : Prop := x * y + 85 = (y + 1) * (x + 5)

-- The theorem to be proven
theorem original_number_of_members (h1 : condition1 x y) (h2 : condition2 x y) : x = 20 :=
by sorry

end original_number_of_members_l709_709516


namespace overall_avg_is_60_l709_709187

-- Define the number of students and average marks for each class
def classA_students : ℕ := 30
def classA_avg_marks : ℕ := 40

def classB_students : ℕ := 50
def classB_avg_marks : ℕ := 70

def classC_students : ℕ := 25
def classC_avg_marks : ℕ := 55

def classD_students : ℕ := 45
def classD_avg_marks : ℕ := 65

-- Calculate the total number of students
def total_students : ℕ := 
  classA_students + classB_students + classC_students + classD_students

-- Calculate the total marks for each class
def total_marks_A : ℕ := classA_students * classA_avg_marks
def total_marks_B : ℕ := classB_students * classB_avg_marks
def total_marks_C : ℕ := classC_students * classC_avg_marks
def total_marks_D : ℕ := classD_students * classD_avg_marks

-- Calculate the combined total marks of all classes
def combined_total_marks : ℕ := 
  total_marks_A + total_marks_B + total_marks_C + total_marks_D

-- Calculate the overall average marks
def overall_avg_marks : ℕ := combined_total_marks / total_students

-- Prove that the overall average marks is 60
theorem overall_avg_is_60 : overall_avg_marks = 60 := by
  sorry -- Proof will be written here

end overall_avg_is_60_l709_709187


namespace statement1_statement2_statement3_statement4_l709_709366

-- Statement 1: even function implies b = 2
theorem statement1 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x ∈ (set.Icc (2 * a - 1) (a + 4)), f x = a * x² + (2 * a + b) * x + 2)
  (h2 : ∀ x ∈ (set.Icc (2 * a - 1) (a + 4)), f (-x) = f x) : b = 2 := sorry

-- Statement 2: both odd and even implies always zero
theorem statement2 (f : ℝ → ℝ) (h1 : ∀ x, f x = sqrt (2008 - x²) + sqrt (x² - 2008))
  (h2 : ∀ x, f (-x) = -f x) (h3 : ∀ x, f (-x) = f x) : ∀ x, f x = 0 := sorry

-- Statement 3: defining odd extension for specific function
theorem statement3 (f : ℝ → ℝ) (h1 : ∀ x, f x = x * (1 + abs x))
  (h2 : ∀ x ∈ set.Ici 0, f x = x * (1 + x))
  (h3 : ∀ x ∈ set.Iic 0, f (-x) = -f x) : ∀ x, f x = x * (1 + abs x) := sorry

-- Statement 4: non-zero constant function with functional property implies contradiction
theorem statement4 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x * y) = x * f y + y * f x)
  (h2 : ∀ x, f x ≠ 0) : false := sorry

#check statement1
#check statement2
#check statement3
#check statement4

end statement1_statement2_statement3_statement4_l709_709366


namespace find_a_from_chord_and_circle_l709_709478

theorem find_a_from_chord_and_circle 
  (a : ℝ) 
  (line_eq: ∀ x y: ℝ, x - y + 3 = 0)
  (circle_eq: ∀ x y: ℝ, (x - a)^2 + (y - 2)^2 = 4)
  (chord_length: 2 * Real.sqrt 2):
  a = 1 ∨ a = -3 := by
  sorry

end find_a_from_chord_and_circle_l709_709478


namespace length_AE_is_root113div3_l709_709513

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := { x := 0, y := 4 }
def B : Point := { x := 8, y := 0 }
def C : Point := { x := 6, y := 3 }
def D : Point := { x := 3, y := 0 }

noncomputable def intersection (p1 p2 p3 p4 : Point) : Point := sorry -- Function to calculate intersection

noncomputable def distance (p1 p2 : Point) : ℝ := 
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def E : Point := intersection A B C D

theorem length_AE_is_root113div3 : distance A E = real.sqrt 113 / 3 := 
by
  sorry

end length_AE_is_root113div3_l709_709513


namespace num_integers_satisfying_inequality_l709_709431

theorem num_integers_satisfying_inequality : 
  {n : ℤ | (n >= 0) ∧ (sqrt n ≤ sqrt (3 * n - 9)) ∧ (sqrt (3 * n - 9) < sqrt (n + 8))}.to_finset.card = 4 := 
by
  sorry

end num_integers_satisfying_inequality_l709_709431


namespace angle_ACD_eq_angle_BCM_l709_709535

-- Define some points and the objects involving them
variables (A B C D M : Point)

-- Definitions of the conditions expressed in Lean-like definitions
def is_parallelogram (P1 P2 P3 P4 : Point) : Prop :=
  opposite_sides_parallel P1 P2 P3 P4 ∧ opposite_sides_equal P1 P2 P3 P4

def angle (P Q R : Point) : Angle := -- Definition of the angle at a point
sorry

def equal_angles (α β : Angle) : Prop :=
α = β

-- The main theorem statement
theorem angle_ACD_eq_angle_BCM 
  (h1 : is_parallelogram A B M D)
  (h2 : equal_angles (angle C B M) (angle C D M)) :
  equal_angles (angle A C D) (angle B C M) :=
sorry

end angle_ACD_eq_angle_BCM_l709_709535


namespace f_monotonically_increasing_iff_g_monotonicity_intervals_l709_709100

variable (a : ℝ)

def f (x : ℝ) := x * Real.log (1 + x) - a * (x + 1)

theorem f_monotonically_increasing_iff :
  (∀ x ≥ 1, ∀ y ≥ 1, f x ≤ f y) ↔ a ≤ (1 / 2) + Real.log 2 :=
sorry

def g (x : ℝ) := f''' x - (a * x) / (x + 1)

theorem g_monotonicity_intervals :
  (∀ x > a - 2, g' x > 0) ∧ (∀ x < a - 2, g' x < 0) ↔ a > 1 ∨
  (∀ x > -1, g' x > 0) ↔ a ≤ 1 :=
sorry

end f_monotonically_increasing_iff_g_monotonicity_intervals_l709_709100


namespace center_of_circle_from_diameter_l709_709621

theorem center_of_circle_from_diameter (x1 y1 x2 y2 : ℝ) 
  (h1 : x1 = 3) (h2 : y1 = -3) (h3 : x2 = 13) (h4 : y2 = 17) :
  (x1 + x2) / 2 = 8 ∧ (y1 + y2) / 2 = 7 :=
by
  sorry

end center_of_circle_from_diameter_l709_709621


namespace equal_segments_l709_709223

-- Given definitions
variables (A B C P K L M S : Type)
variable [In : ∀ x : Type, Inhabited x]

def Triangle (A B C : Type) := ∃ (α β γ : Type), α + β + γ = 180
def Circumcircle (A B C : Type) := True    -- Placeholder definition for the circumcircle
def Intersection (l1 l2 : Type) (Γ : Type) (P : Type) := True  -- Placeholder for intersection definition
def Tangent (Γ : Type) (C : Type) (S : Type) := True  -- Placeholder for tangent
def SC_eq_SP (S : Type) (C : Type) (P : Type) := S = C ∧ C = P  -- Notation: SC = SP

-- Problem statement
theorem equal_segments
  (h1 : Triangle A B C)                          -- Condition: ABC is a triangle
  (h2 : Circumcircle A B C)                      -- Condition: Γ is the circumcircle of triangle ABC
  (h3 : Intersection (AP) (Γ) K)                 -- Condition: K is an intersection of line AP with Γ
  (h4 : Intersection (BP) (Γ) L)                 -- Condition: L is an intersection of line BP with Γ
  (h5 : Intersection (CP) (Γ) M)                 -- Condition: M is an intersection of line CP with Γ
  (h6 : Tangent Γ C S)                           -- Condition: Tangent to Γ at C intersects AB at S
  (h7 : SC_eq_SP S C P)                          -- Condition: SC = SP
  : MK = ML := sorry                             -- Prove that MK = ML

end equal_segments_l709_709223


namespace car_travel_difference_l709_709026

variable (d_t : ℕ) (t_t : ℕ) (t_c : ℝ) (Δv : ℕ)

theorem car_travel_difference (h1 : d_t = 296) (h2 : t_t = 8) (h3 : t_c = 5.5) (h4 : Δv = 18) : 
  let v_t := (d_t : ℝ) / (t_t : ℝ) in
  let v_c := v_t + (Δv : ℝ) in
  let d_c := v_c * t_c in
  d_c - (d_t : ℝ) = 6.5 :=
by
  sorry

end car_travel_difference_l709_709026


namespace cyclist_arrives_first_l709_709712

-- Definitions based on given conditions
def speed_cyclist (v : ℕ) := v
def speed_motorist (v : ℕ) := 5 * v

def distance_total (d : ℕ) := d
def distance_half (d : ℕ) := d / 2

def time_motorist_first_half (d v : ℕ) : ℕ := distance_half d / speed_motorist v

def remaining_distance_cyclist (d v : ℕ) := d - v * time_motorist_first_half d v

def speed_motorist_walking (v : ℕ) := v / 2

def time_motorist_second_half (d v : ℕ) := distance_half d / speed_motorist_walking v
def time_cyclist_remaining (d v : ℕ) : ℕ := remaining_distance_cyclist d v / speed_cyclist v

-- Comparison to prove cyclist arrives first
theorem cyclist_arrives_first (d v : ℕ) (hv : 0 < v) (hd : 0 < d) :
  time_cyclist_remaining d v < time_motorist_second_half d v :=
by sorry

end cyclist_arrives_first_l709_709712


namespace find_pairs_l709_709690

theorem find_pairs (x y : ℕ) (h1 : x < y) (h2 : ∑ i in Finset.range (y - x - 1), (x + 1 + i) = 1999) :
  (x = 1998 ∧ y = 2000) ∨ (x = 998 ∧ y = 1001) :=
by {
  sorry
}

end find_pairs_l709_709690


namespace three_digit_numbers_with_perfect_square_digit_sum_l709_709120

noncomputable def count_three_digit_numbers_with_perfect_square_digit_sum : ℕ :=
  let valid_digits := Finset.range 10
  let perfect_squares := [1, 4, 9, 16, 25]
  let three_digit_numbers := Finset.Icc 100 999
  (three_digit_numbers.filter (λ n, perfect_squares.contains (digit_sum n))).card
where
  digit_sum (n : ℕ) : ℕ := 
    let d1 := n / 100
    let d2 := (n / 10) % 10
    let d3 := n % 10
    d1 + d2 + d3

theorem three_digit_numbers_with_perfect_square_digit_sum :
  count_three_digit_numbers_with_perfect_square_digit_sum = 51 :=
sorry

end three_digit_numbers_with_perfect_square_digit_sum_l709_709120


namespace f_2017_equals_minus_one_half_l709_709461

noncomputable def f (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x < 0 then 2^x + Real.log2 (-x) else sorry -- Placeholder for other cases

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry -- Placeholder for odd property

lemma period_4 (x : ℝ) : f (x + 4) = f x := sorry -- Placeholder for period property

-- Prove that f(2017) = -1/2
theorem f_2017_equals_minus_one_half : f 2017 = -1 / 2 := by
  have h1 : f 2017 = f 1 := by sorry -- Using periodicity
  have h2 : f 1 = -f (-1) := by apply odd_function
  have h3 : f (-1) = 1/2 := by
    have h4 : -2 ≤ (-1) ∧ (-1) < 0 := by sorry -- Bounds check
    rw [f, if_pos h4] -- Simplifying definition
    norm_num
    rw [Real.log2_one]
  rw [h2, h3]
  norm_num

end f_2017_equals_minus_one_half_l709_709461


namespace standard_deviation_of_data_set_l709_709615

theorem standard_deviation_of_data_set 
  {x : ℝ} 
  (h1 : x ≠ 5) 
  (h2 : (2 / 3) * ((x + 2) / 2) = 2) : 
  (let data := [10, 5, x, 2, 2, 1] in 
    stddev data = 3) := 
sorry

end standard_deviation_of_data_set_l709_709615


namespace crosses_overlap_l709_709707

theorem crosses_overlap (n : ℕ) (hn : n = 10^9) (r_field : ℝ) (hr_field : r_field = 1000)
  (l_branch : ℕ → ℝ) (hl_branch : ∀ i, l_branch i ≥ 1) :
  ∃ i j, i ≠ j ∧ overlap (cross i) (cross j) := sorry

noncomputable def cross (i : ℕ) : set ℝ := sorry

noncomputable def overlap (c1 c2 : set ℝ) : Prop := sorry

end crosses_overlap_l709_709707


namespace problem_I_problem_II_l709_709098

noncomputable def f (x : ℝ) : ℝ := (1/4) * x^2 + (1/2) * x - (3/4)

def a_n (n : ℕ) : ℝ := 2 * n + 1

def b_n (n : ℕ) : ℝ := 1 / (a_n n + 1)^2

def T_n (n : ℕ) : ℝ := ∑ i in range (n + 1), b_n i

theorem problem_I : ∀ n : ℕ, a_n n = 2 * n + 1 :=
by sorry

theorem problem_II : ∀ n : ℕ, T_n n < 1 / 6 :=
by sorry

end problem_I_problem_II_l709_709098


namespace geometric_sequence_properties_l709_709896

theorem geometric_sequence_properties (a b c : ℝ) (r : ℝ) (h : r ≠ 0)
  (h1 : a = r * (-1))
  (h2 : b = r * a)
  (h3 : c = r * b)
  (h4 : -9 = r * c) :
  b = -3 ∧ a * c = 9 :=
by sorry

end geometric_sequence_properties_l709_709896


namespace total_cost_l709_709963

def permit_cost : Int := 250
def contractor_hourly_rate : Int := 150
def contractor_days : Int := 3
def contractor_hours_per_day : Int := 5
def inspector_discount_rate : Float := 0.80

theorem total_cost : Int :=
  let total_hours := contractor_days * contractor_hours_per_day
  let contractor_total_cost := total_hours * contractor_hourly_rate
  let inspector_hourly_rate := contractor_hourly_rate - (inspector_discount_rate * contractor_hourly_rate)
  let inspector_total_cost := total_hours * Int.ofFloat inspector_hourly_rate
  permit_cost + contractor_total_cost + inspector_total_cost

example : total_cost = 2950 := by
  sorry

end total_cost_l709_709963


namespace mask_production_decrease_l709_709929

theorem mask_production_decrease (x : ℝ) : 
  (1 : ℝ) * (1 - x)^2 = 0.64 → 100 * (1 - x)^2 = 64 :=
by
  intro h
  sorry

end mask_production_decrease_l709_709929


namespace no_such_number_exists_l709_709622

-- Definitions for conditions
def base_5_digit_number (x : ℕ) : Prop := 
  ∀ n, 0 ≤ n ∧ n < 2023 → x / 5^n % 5 < 5

def odd_plus_one (n m : ℕ) : Prop :=
  (∀ k < 1012, (n / 5^(2*k) % 25 / 5 = m / 5^(2*k) % 25 / 5 + 1)) ∧
  (∀ k < 1011, (n / 5^(2*k+1) % 25 / 5 = m / 5^(2*k+1) % 25 / 5 - 1))

def has_two_prime_factors_that_differ_by_two (x : ℕ) : Prop :=
  ∃ u v, u * v = x ∧ Prime u ∧ Prime v ∧ v = u + 2

-- Combined conditions for the hypothesized number x
def hypothesized_number (x : ℕ) : Prop := 
  base_5_digit_number x ∧
  odd_plus_one x x ∧
  has_two_prime_factors_that_differ_by_two x

-- The proof statement that the hypothesized number cannot exist
theorem no_such_number_exists : ¬ ∃ x, hypothesized_number x :=
by
  sorry

end no_such_number_exists_l709_709622


namespace intersection_A_B_l709_709569

universe u

variable (x : ℝ)

def U := ℝ

def A := { x : ℝ | (x - 1) / (4 - x) ≥ 0 }

def B := { x : ℝ | real.log x / real.log 2 ≤ 2 }

theorem intersection_A_B :
  { x : ℝ | (x - 1) / (4 - x) ≥ 0 } ∩ { x : ℝ | real.log x / real.log 2 ≤ 2 } = { x : ℝ | 1 ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_A_B_l709_709569


namespace optionB_correct_l709_709678

theorem optionB_correct : ((-1:ℤ)^3 = -1^3) := by
  calculate sorry
  sorry

end optionB_correct_l709_709678


namespace profit_percentage_correct_l709_709185

def initial_cost : ℝ := 100
def initial_profit_rate : ℝ := 1.70
def food_cost_percentage : ℝ := 0.65
def labor_cost_percentage : ℝ := 0.25
def overhead_cost_percentage : ℝ := 0.10

def food_increase_rate : ℝ := 0.14
def labor_increase_rate : ℝ := 0.05
def overhead_decrease_rate : ℝ := 0.08
def sp_increase_rate : ℝ := 0.07

def initial_profit := initial_profit_rate * initial_cost
def initial_sp := initial_cost + initial_profit

def new_food_cost := initial_cost * food_cost_percentage * (1 + food_increase_rate)
def new_labor_cost := initial_cost * labor_cost_percentage * (1 + labor_increase_rate)
def new_overhead_cost := initial_cost * overhead_cost_percentage * (1 - overhead_decrease_rate)

def new_total_cost := new_food_cost + new_labor_cost + new_overhead_cost
def new_sp := initial_sp * (1 + sp_increase_rate)
def new_profit := new_sp - new_total_cost

def percentage_new_sp_profit := (new_profit / new_sp) * 100

theorem profit_percentage_correct : percentage_new_sp_profit ≈ 62.07 := by
  sorry

end profit_percentage_correct_l709_709185


namespace three_digit_numbers_with_perfect_square_sum_l709_709122

theorem three_digit_numbers_with_perfect_square_sum :
  ∃ (n : ℕ), n = 58 ∧ ∀ (a b c : ℕ), (100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c < 1000) →
  ∃ (s : ℕ), a + b + c = s ∧ s ∈ {1, 4, 9, 16, 25} :=
by sorry

end three_digit_numbers_with_perfect_square_sum_l709_709122


namespace salty_cookies_initial_at_least_34_l709_709594

variable {S : ℕ}  -- S will represent the initial number of salty cookies

-- Conditions from the problem
def sweet_cookies_initial := 8
def sweet_cookies_ate := 20
def salty_cookies_ate := 34
def more_salty_than_sweet := 14

theorem salty_cookies_initial_at_least_34 :
  8 = sweet_cookies_initial ∧
  20 = sweet_cookies_ate ∧
  34 = salty_cookies_ate ∧
  salty_cookies_ate = sweet_cookies_ate + more_salty_than_sweet
  → S ≥ 34 :=
by sorry

end salty_cookies_initial_at_least_34_l709_709594


namespace minimize_travel_expense_l709_709240

-- Definitions of the points and distances
def X : ℝ := 0
def Y : ℝ := 4500
def Z : ℝ := 4000

-- Distance between Y and Z using Pythagorean theorem
def distance_YZ : ℝ := Real.sqrt ((Y - X)^2 - (Z - X)^2) -- √(4500^2 - 4000^2) = 2062

-- Total travel distance
def total_distance : ℝ := Y + distance_YZ + Z -- 4500 + 2062 + 4000 = 10562

-- Costs
def bus_cost_per_km : ℝ := 0.20
def airplane_booking_fee : ℝ := 120.00
def airplane_cost_per_km : ℝ := 0.12

-- Total cost calculations
def bus_total_cost : ℝ := bus_cost_per_km * total_distance -- $0.20 * 10562
def airplane_total_cost : ℝ := airplane_booking_fee + airplane_cost_per_km * total_distance -- $120 + $0.12 * 10562

-- Proof statement
theorem minimize_travel_expense :
  total_distance = 10562 ∧ airplane_total_cost = 1387.44 := by
  sorry

end minimize_travel_expense_l709_709240


namespace omega_is_half_intervals_of_decrease_l709_709567

noncomputable def f (ω x : ℝ) : ℝ :=
  sin (ω * x) * cos (ω * x) - sqrt 3 * (cos (ω * x))^2 + sqrt 3 / 2

def g (x φ : ℝ) : ℝ := cos (2 * x - φ)

-- Given conditions
variable (ω : ℝ) (hω_pos : 0 < ω)
variable (h_dist : (let T := 2 * pi in (T / 2)^2 + 4 = pi^2 + 4))
variable (φ : ℝ) (hφ_range : 0 < φ ∧ φ < pi / 2)
variable (h_odd : (∀ x : ℝ, f ω (x + φ) = -f ω x))

-- Questions to be proven
theorem omega_is_half : ω = 1 / 2 :=
  sorry

theorem intervals_of_decrease :
  ∀ I ∈ [Set.Icc (pi / 6) (2 * pi / 3), Set.Icc (7 * pi / 6) (5 * pi / 3)],
    ∀ x ∈ I, deriv (g x φ) < 0 :=
  sorry

end omega_is_half_intervals_of_decrease_l709_709567


namespace coeff_x3_in_expansion_l709_709264

theorem coeff_x3_in_expansion (x : ℝ) : 
  let f := x^2 - x + 2
  in (coeff_in_binomial_expansion f 5 3) = -200 := 
by
  -- Definitions of terms and binomial expansions
  let f := x ^ 2 - x + 2
  let n := 5
  let k := 3
  sorry -- Omit proof steps, focus on correctness of goal

end coeff_x3_in_expansion_l709_709264


namespace count_valid_matrices_l709_709881

/-- We define a configuration of a 6x6 matrix with entries of 1 and -1 such that the sum of each row
    and each column is 0, and then ensure the number of such configurations is 12101. 
    -/
noncomputable def valid_matrices : Nat :=
  12101

theorem count_valid_matrices : ∃ (M : Matrix (Fin 6) (Fin 6) ℤ), (∀ i, (∑ j, M i j) = 0) ∧ (∀ j, (∑ i, M i j) = 0) ∧ (∀ i j, M i j = 1 ∨ M i j = -1) ∧ valid_matrices = 12101 :=
by
  sorry

end count_valid_matrices_l709_709881


namespace find_a_l709_709926

open Real

-- Define the given points (3, a) and (-2, 0)
def point1 (a : ℝ) := (3, a : ℝ)
def point2 := (-2, 0 : ℝ)

-- Define the slope function of a line passing through two points
def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

-- Define the perpendicular condition, where the product of the slopes equals -1
def perpendicular_slope_condition (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- lambda functions to specify the exact slopes
def slope1 (a : ℝ) : ℝ := slope (3 : ℝ) a (-2 : ℝ) 0
def slope2 : ℝ := (1 / 2)

-- Define the final theorem that matches the problem's conditions and solution
theorem find_a (a : ℝ) : perpendicular_slope_condition (slope1 a) slope2 → a = -10 :=
begin
  -- We state the proof obligation
  assume h : perpendicular_slope_condition (slope1 a) slope2,
  -- We use the steps from the solution in our reasoning towards the statement
  sorry
end

end find_a_l709_709926


namespace three_digit_numbers_with_perfect_square_sum_l709_709121

theorem three_digit_numbers_with_perfect_square_sum :
  ∃ (n : ℕ), n = 58 ∧ ∀ (a b c : ℕ), (100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c < 1000) →
  ∃ (s : ℕ), a + b + c = s ∧ s ∈ {1, 4, 9, 16, 25} :=
by sorry

end three_digit_numbers_with_perfect_square_sum_l709_709121


namespace max_value_triangle_l709_709176

-- Conditions from the problem
variables {a b c : ℝ} (A B C : ℝ)
def in_triangle_ABC : Prop := 
  a = 2 * b * c * Real.sin A ∧
  b^2 + c^2 = a^2 + 2 * b * c * Real.cos A ∧
  ∃ (h : ℝ), h = a / 2

-- Main statement to prove
theorem max_value_triangle {A B C : ℝ} (h1 : in_triangle_ABC A B C) :
  ∃ (x : ℝ), (x = 2 * Real.sqrt 2 ∧ x = max (frac (c) (b) + frac (b) (c))) :=
sorry

end max_value_triangle_l709_709176


namespace sequence_product_zero_l709_709528

theorem sequence_product_zero :
  (∃ (e f g : ℤ),
    (∀ (n : ℕ), 
      let b_n := e * (Int.floor (Real.sqrt (n + f))) + g in 
      b_n = 2 * (Int.floor (Real.sqrt (n + f)))) ∧ 
    e * f * g = 0) :=
begin
  sorry
end

end sequence_product_zero_l709_709528


namespace sum_pqrs_eq_3150_l709_709220

theorem sum_pqrs_eq_3150
  (p q r s : ℝ)
  (h1 : p ≠ q) (h2 : p ≠ r) (h3 : p ≠ s) (h4 : q ≠ r) (h5 : q ≠ s) (h6 : r ≠ s)
  (hroots1 : ∀ x : ℝ, x^2 - 14*p*x - 15*q = 0 → (x = r ∨ x = s))
  (hroots2 : ∀ x : ℝ, x^2 - 14*r*x - 15*s = 0 → (x = p ∨ x = q)) :
  p + q + r + s = 3150 :=
by
  sorry

end sum_pqrs_eq_3150_l709_709220


namespace smallest_positive_period_of_f_l709_709420

-- Definition of the function
def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)

-- Period formula for the sine function
def period (ω : ℝ) : ℝ := 2 * π / ω

-- Angular frequency of the function
def ω : ℝ := 2

-- Proof statement
theorem smallest_positive_period_of_f : period ω = π := by
  -- The proof will be placed here
  sorry

end smallest_positive_period_of_f_l709_709420


namespace cube_root_of_110592000_l709_709385

theorem cube_root_of_110592000 : real.cbrt 110592000 = 480 := by
  sorry

end cube_root_of_110592000_l709_709385


namespace find_a_l709_709481

noncomputable def A : Set ℝ := {0, 1, 2}

noncomputable def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 3}

theorem find_a (a : ℝ) (h : A ∩ (B a) = {1}) : a = -1 :=
by
  have h₁ : 1 ∈ A := by simp [A]
  have h₂ : 1 ∈ B a := by simp [B, h]
  simp at h₂
  cases h₂
  case one =>
    simp [A] at h₁
    sorry
  case two =>
    simp [A] at h₁
    sorry

end find_a_l709_709481


namespace value_of_a_l709_709639

theorem value_of_a 
  (a b c d e : ℤ)
  (h1 : a + 4 = b + 2)
  (h2 : a + 2 = b)
  (h3 : a + c = 146)
  (he : e = 79)
  (h4 : e = d + 2)
  (h5 : d = c + 2)
  (h6 : c = b + 2) :
  a = 71 :=
by
  sorry

end value_of_a_l709_709639


namespace circle_tangency_l709_709293

theorem circle_tangency
  (R r : ℝ) (hR_gt_r : R > r) (internal_tangency : Bool) :
  let sgn := if internal_tangency then -1 else 1;
      r0 := (4 * R * r * (R - (sgn * r))) / (R + (sgn * r))^2 in
  r0 = (4 * R * r * (R - (sgn * r))) / (R + (sgn * r))^2 :=
by 
  sorry

end circle_tangency_l709_709293


namespace intervals_of_monotonic_increase_max_and_min_values_g_l709_709472

noncomputable def f (x : ℝ) : ℝ :=
  2 * sqrt 3 * sin (x + π/4) * cos (x + π/4) + sin (2*x)

noncomputable def g (x : ℝ) : ℝ :=
  f (x + π/6)

theorem intervals_of_monotonic_increase :
  ∀ k ∈ ℤ, (∀ x ∈ Icc (-5*π/12 + k*π) (π/12 + k*π), f' x > 0)
  sorry

theorem max_and_min_values_g :
  ∀ x ∈ Icc 0 (π/2), g (x) ∈ Icc (-2) (sqrt 3)
  sorry

end intervals_of_monotonic_increase_max_and_min_values_g_l709_709472


namespace rectangle_area_l709_709407

theorem rectangle_area (P : ℕ) (a : ℕ) (b : ℕ) (h₁ : P = 2 * (a + b)) (h₂ : P = 40) (h₃ : a = 5) : a * b = 75 :=
by
  sorry

end rectangle_area_l709_709407


namespace diameter_of_tripled_volume_sphere_l709_709617

-- Define the volume of a sphere with radius r
def sphere_volume (r : ℝ) : ℝ :=
  (4 / 3) * π * r ^ 3

-- Define the diameter function based on the radius
def sphere_diameter (r : ℝ) : ℝ :=
  2 * r

-- Problem setup based on provided conditions
theorem diameter_of_tripled_volume_sphere :
  let r₁ := 6
  let v₁ := sphere_volume r₁
  let v₂ := 3 * v₁
  let r₂ := (v₂ * (3 / (4 * π))) ^ (1 / 3)
  let d := sphere_diameter r₂ 
  let a := 18
  let b := 12
  d = a * (b ^ (1 / 3)) →
  a + b = 30 :=
by
  sorry

end diameter_of_tripled_volume_sphere_l709_709617


namespace largest_base_4_number_with_four_digits_is_255_l709_709662

theorem largest_base_4_number_with_four_digits_is_255 :
  let n := 3 * 4^3 + 3 * 4^2 + 3 * 4^1 + 3 * 4^0 in
  n = 255 := by
sorry

end largest_base_4_number_with_four_digits_is_255_l709_709662


namespace area_of_rhombus_l709_709644

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 9) (h2 : d2 = 14) : 
  (d1 * d2) / 2 = 63 :=
by 
  -- Substitute the given diagonal lengths
  rw [h1, h2],
  -- Calculate the product and division
  norm_num,
  -- Result should be 63
  sorry

end area_of_rhombus_l709_709644


namespace problem_sufficiency_not_necessity_l709_709949

variable {a : ℕ → ℕ}

def p (n : ℕ) : Prop := a (n + 1) + a n = 2^(n + 1) + 2^n

def q : Prop := ∃ r : ℕ, ∀ n : ℕ, a (n + 1) - 2^(n + 1) = r * (a n - 2^n)

theorem problem_sufficiency_not_necessity 
  (h₀ : a 1 = 1)
  (h₁ : ∀ n, p n) : q ∧ ¬(∀ n, q → p n) :=
sorry

end problem_sufficiency_not_necessity_l709_709949


namespace parallel_vectors_k_value_l709_709877

theorem parallel_vectors_k_value (k : ℝ) :
  (∀ a b : ℝ, a * (k - 6) = b * k → a = 1 ∧ b = 9) → k = -3 / 4 :=
by
  intro h
  have h' := h 1 9
  simpa using h' sorry

end parallel_vectors_k_value_l709_709877


namespace pizza_cost_per_slice_l709_709540

theorem pizza_cost_per_slice :
  let pizza_cost := 10
  let first_topping_cost := 2
  let next_two_toppings_cost := 2
  let remaining_toppings_cost := 2
  let total_cost := pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  let slices := 8
  total_cost / slices = 2 := by
  let pizza_cost := 10
  let first_topping_cost := 2
  let next_two_toppings_cost := 2
  let remaining_toppings_cost := 2
  let total_cost := pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  let slices := 8
  have h : total_cost = 16 := by
    -- calculations to show total_cost = 16 can be provided here
    sorry
  have hslices : slices = 8 := rfl
  calc
    total_cost / slices = 16 / 8 : by rw [h, hslices]
                  ... = 2         : by norm_num

end pizza_cost_per_slice_l709_709540


namespace count_eq_58_l709_709117

-- Definitions based on given conditions
def isThreeDigit (n : Nat) : Prop := n >= 100 ∧ n <= 999
def digitSum (n : Nat) : Nat := 
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 + d2 + d3
def isPerfectSquare (n : Nat) : Prop := ∃ k : Nat, k * k = n

-- Condition for digit sum being one of the specific perfect squares
def isPerfectSquareDigitSum (n : Nat) : Prop :=
  isThreeDigit n ∧ (digitSum n = 1 ∨ digitSum n = 4 ∨ digitSum n = 9 ∨ digitSum n = 16 ∨ digitSum n = 25)

-- Total count of such numbers
def countPerfectSquareDigitSumNumbers : Finset Nat :=
  (Finset.range 1000).filter isPerfectSquareDigitSum

-- Lean statement to prove the count is 58
theorem count_eq_58 : countPerfectSquareDigitSumNumbers.card = 58 := sorry

end count_eq_58_l709_709117


namespace sum_of_real_roots_exists_solution_in_interval_l709_709325

-- Part (a)
theorem sum_of_real_roots :
  (∑ x : ℝ in finset.univ.filter (λ x, x^2 + 18 * x + 30 = 2 * real.sqrt (x^2 + 18 * x + 45)), x) = -18 :=
sorry

-- Part (b)
theorem exists_solution_in_interval :
  ∃ x : ℝ, 0 < x ∧ x < 5 ∧ √ (5 - (√ (5 - x))) = x :=
sorry

end sum_of_real_roots_exists_solution_in_interval_l709_709325


namespace total_snowballs_l709_709039

theorem total_snowballs (Lc : ℕ) (Ch : ℕ) (Pt : ℕ)
  (h1 : Ch = Lc + 31)
  (h2 : Lc = 19)
  (h3 : Pt = 47) : 
  Ch + Lc + Pt = 116 := by
  sorry

end total_snowballs_l709_709039


namespace equilateral_triangle_rectangles_l709_709214

theorem equilateral_triangle_rectangles {A B C: Point} (h : equilateral_triangle A B C) : 
  num_rectangles_share_two_vertices A B C = 3 :=
sorry

end equilateral_triangle_rectangles_l709_709214


namespace eric_required_bike_speed_l709_709398

noncomputable def required_average_speed 
  (total_goal_time : ℝ)
  (swim_speed : ℝ) (swim_distance : ℝ)
  (run_speed : ℝ) (run_distance : ℝ)
  (bike_distance : ℝ) : ℝ :=
  let t_swim := swim_distance / swim_speed in
  let t_run := run_distance / run_speed in
  let t_total := t_swim + t_run in
  let t_bike := total_goal_time - t_total in
  bike_distance / t_bike

theorem eric_required_bike_speed
  (total_goal_time : ℝ := 2.5)
  (swim_speed : ℝ := 2)
  (swim_distance : ℝ := 0.25)
  (run_speed : ℝ := 5)
  (run_distance : ℝ := 3)
  (bike_distance : ℝ := 20) :
  required_average_speed total_goal_time swim_speed swim_distance run_speed run_distance bike_distance = 11.27 :=
sorry

end eric_required_bike_speed_l709_709398


namespace area_of_shaded_region_l709_709403

theorem area_of_shaded_region :
  let ABCD_area := 36
  let EFGH_area := 1 * 3
  let IJKL_area := 2 * 4
  let MNOP_area := 3 * 1
  let shaded_area := ABCD_area - (EFGH_area + IJKL_area + MNOP_area)
  shaded_area = 22 :=
by
  let ABCD_area := 36
  let EFGH_area := 1 * 3
  let IJKL_area := 2 * 4
  let MNOP_area := 3 * 1
  let shaded_area := ABCD_area - (EFGH_area + IJKL_area + MNOP_area)
  sorry

end area_of_shaded_region_l709_709403


namespace ratio_of_x_and_y_l709_709221

theorem ratio_of_x_and_y (x y : ℝ) (h : (x - y) / (x + y) = 4) : x / y = -5 / 3 :=
by sorry

end ratio_of_x_and_y_l709_709221


namespace process_end_after_two_draws_exactly_two_white_balls_l709_709178

noncomputable def prob_end_after_two_draws (n : ℕ) (red_balls white_balls blue_balls : ℕ) : ℝ :=
((red_balls + white_balls).choose 1 / n.choose 1) * (blue_balls.choose 1 / n.choose 1)

noncomputable def prob_two_white_balls (n : ℕ) (red_balls white_balls blue_balls : ℕ) : ℝ :=
(((red_balls / n) * (white_balls / n) * (white_balls / n) * 3) 
+ ((white_balls / n) * (white_balls / n) * (blue_balls / n)))

theorem process_end_after_two_draws :
  prob_end_after_two_draws 10 5 3 2 = 4 / 25 :=
begin
  sorry
end

theorem exactly_two_white_balls :
  prob_two_white_balls 10 5 3 2 = 153 / 1000 :=
begin
  sorry
end

end process_end_after_two_draws_exactly_two_white_balls_l709_709178


namespace inequality_holds_l709_709242

theorem inequality_holds (a : ℝ) : 3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 :=
by
  sorry

end inequality_holds_l709_709242


namespace total_cost_of_season_l709_709970

/-- 
John starts watching a TV show. He pays $1000 per episode for the first half of the season.
The second half of the season had episodes that cost 120% more expensive. 
If there are 22 episodes in total, we need to prove that the entire season cost $35,200.
-/
theorem total_cost_of_season (episodes : ℕ) (cost1 : ℕ) (cost_increase_pct : ℕ) (num_episodes : ℕ):
  episodes = 22 →
  cost1 = 1000 →
  cost_increase_pct = 120 →
  num_episodes = episodes / 2 →
  let cost2 := cost1 + (cost1 * cost_increase_pct / 100) in
  (num_episodes * cost1 + num_episodes * cost2) = 35200 :=
by
  intros h1 h2 h3 h4
  let cost2 := cost1 + (cost1 * cost_increase_pct / 100)
  sorry

end total_cost_of_season_l709_709970


namespace a_n_formula_b_n_sum_l709_709106

noncomputable def a_sequence : ℕ → ℕ
| 1 => 2
| n+1 => 2 * ((n + 1) / n)^2 * (a_sequence n)

def b_sequence (n : ℕ) : ℤ :=
3 * (Int.log2 ((a_sequence n) / (n^2))) - 26

def T (n : ℕ) : ℕ :=
if n ≤ 8 then (49 * n - 3 * n^2) / 2
else (3 * n^2 - 49 * n + 400) / 2

theorem a_n_formula (n : ℕ) (h : n > 0) : a_sequence n = n^2 * 2^n :=
sorry

theorem b_n_sum (n : ℕ) : 
let T_n := if n ≤ 8 then (49 * n - 3 * n^2) / 2 else (3 * n^2 - 49 * n + 400) / 2
in (∀ n : ℕ, T n = T_n) :=
sorry

end a_n_formula_b_n_sum_l709_709106


namespace rhombus_area_l709_709793

-- Define the problem and conditions
variables (PQRS : Type) [metric_space PQRS] [inner_product_space ℝ PQRS]
variables (P Q R S : PQRS)
variable (circumradius_PQR : ℝ)
variable (circumradius_PSR : ℝ)

-- Given conditions
axiom circumradius_PQR_eq_15 : circumradius_PQR = 15
axiom circumradius_PSR_eq_30 : circumradius_PSR = 30

-- Prove that the area of the rhombus PQRS is 400
theorem rhombus_area
  (h : (triangle P Q R).circumradius = circumradius_PQR)
  (h' : (triangle P S R).circumradius = circumradius_PSR) :
  let d1 := 20, d2 := 40 in (1 / 2 * d1 * d2) = 400 := sorry

end rhombus_area_l709_709793


namespace find_p_l709_709674

open LinearMap

-- Define the vectors a, b, and the result p as constant vectors in ℝ^3
def a : ℝ^3 := ⟨1, -1, 2⟩
def b : ℝ^3 := ⟨-1, 2, 1⟩
def p : ℝ^3 := ⟨0, 1 / 2, 3 / 2⟩

-- Helper function to determine collinearity by a scalar multiple
def collinear (u v : ℝ^3) : Prop := ∃ k : ℝ, u = k • v

-- Statement of the problem where p is the given vector satisfying specified conditions
theorem find_p (v : ℝ^3) 
  (h1 : ∃ t : ℝ, p = a + t • (b - a))
  (h2 : collinear a p) 
  (h3 : collinear b p) :
  p = ⟨0, 1 / 2, 3 / 2⟩ := 
sorry

end find_p_l709_709674


namespace max_m_n_l709_709549

theorem max_m_n (m n: ℕ) (h: m + 3*n - 5 = 2 * Nat.lcm m n - 11 * Nat.gcd m n) : 
  m + n ≤ 70 :=
sorry

end max_m_n_l709_709549


namespace rectangle_perimeter_l709_709266

theorem rectangle_perimeter (d m n : ℝ) :
  2 * sqrt 2 * d * cos ((m - n) * real.pi / (4 * (m + n))) = 
  let x := d in 
  let ratio := m / n in
  let angle_division := (m / ratio + n / ratio) * real.pi / 2 in
  2 * sqrt 2 * x * cos (angle_division / (2 * x * ratio)) :=
sorry

end rectangle_perimeter_l709_709266


namespace total_distance_run_l709_709363

def track_meters : ℕ := 9
def laps_already_run : ℕ := 6
def laps_to_run : ℕ := 5

theorem total_distance_run :
  (laps_already_run * track_meters) + (laps_to_run * track_meters) = 99 := by
  sorry

end total_distance_run_l709_709363


namespace b_arithmetic_a_formula_l709_709996

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l709_709996


namespace is_orthocenter_l709_709227

variables (A B C O P Q : Type)
variables [T : Triangle ABC] [H1 : Circumcenter O ABC] 
variables [H2 : PointOnHalfLine P AC] [H3 : PointOnHalfLine Q BA]
variables [Simil : Similarity ABC BPQ]
variables [Angle1 : Angle BPQ = Angle ABC] [Angle2 : Angle PQB = Angle BCA]

theorem is_orthocenter : Orthocenter O BPQ :=
by
  sorry

end is_orthocenter_l709_709227


namespace solve_quadratic_eq_l709_709606

theorem solve_quadratic_eq (x : ℝ) : x ^ 2 + 2 * x - 5 = 0 → (x = -1 + Real.sqrt 6 ∨ x = -1 - Real.sqrt 6) :=
by 
  intro h
  sorry

end solve_quadratic_eq_l709_709606


namespace simplest_square_root_l709_709319

theorem simplest_square_root (A B C D : Real) 
    (hA : A = Real.sqrt 0.1) 
    (hB : B = 1 / 2) 
    (hC : C = Real.sqrt 30) 
    (hD : D = Real.sqrt 18) : 
    C = Real.sqrt 30 := 
by 
    sorry

end simplest_square_root_l709_709319


namespace find_a_if_y_is_even_l709_709166

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem find_a_if_y_is_even (a : ℝ) (h : ∀ x : ℝ, y x a = y (-x) a) : a = 2 :=
by
  sorry

end find_a_if_y_is_even_l709_709166


namespace finite_diff_function_count_l709_709563

open Int

noncomputable def delta {a p e : ℕ} (f : ZMod a → ZMod (p^e)) : ZMod a → ZMod (p^e) :=
  λ n => f (n+1) - f n

noncomputable def D {a p e : ℕ} (f : ZMod a → ZMod (p^e)) (k : ℕ) : ZMod a → ZMod (p^e) :=
  match k with
  | 0 => f
  | (k+1) => delta (D f k)

theorem finite_diff_function_count
  (p a e : ℕ)
  (h_prime : Prime p)
  (h_a : a ≥ 2)
  (h_e : e ≥ 1) :
  ∃ (f : ZMod a → ZMod (p^e)), (∃ k ≥ 1, D f k = f) ↔ Finset.card (Finset.univ : Finset (ZMod a → ZMod (p^e))) = p^(a - p^((nat_trailing_zeros p) a)).
Proof
  sorry

end finite_diff_function_count_l709_709563


namespace unique_positive_integer_n_l709_709053

theorem unique_positive_integer_n (n x : ℕ) (hx : x > 0) (hn : n = 2 ^ (2 * x - 1) - 5 * x - 3 ∧ n = (2 ^ (x-1) - 1) * (2 ^ x + 1)) : n = 2015 := by
  sorry

end unique_positive_integer_n_l709_709053


namespace johns_last_month_savings_l709_709329

theorem johns_last_month_savings (earnings rent dishwasher left_over : ℝ) 
  (h1 : rent = 0.40 * earnings) 
  (h2 : dishwasher = 0.70 * rent) 
  (h3 : left_over = earnings - rent - dishwasher) :
  left_over = 0.32 * earnings :=
by 
  sorry

end johns_last_month_savings_l709_709329


namespace pizzaCostPerSlice_l709_709538

/-- Define the constants and parameters for the problem --/
def largePizzaCost : ℝ := 10.00
def numberOfSlices : ℕ := 8
def firstToppingCost : ℝ := 2.00
def secondThirdToppingCost : ℝ := 1.00
def otherToppingCost : ℝ := 0.50
def toppings : List String := ["pepperoni", "sausage", "ham", "olives", "mushrooms", "bell peppers", "pineapple"]

/-- Calculate the total number of toppings --/
def numberOfToppings : ℕ := toppings.length

/-- Calculate the total cost of the pizza including all toppings --/
noncomputable def totalPizzaCost : ℝ :=
  largePizzaCost + 
  firstToppingCost + 
  2 * secondThirdToppingCost + 
  (numberOfToppings - 3) * otherToppingCost

/-- Calculate the cost per slice --/
noncomputable def costPerSlice : ℝ := totalPizzaCost / numberOfSlices

/-- Proof statement: The cost per slice is $2.00 --/
theorem pizzaCostPerSlice : costPerSlice = 2 := by
  sorry

end pizzaCostPerSlice_l709_709538


namespace gamma_minus_alpha_eq_4pi_over_3_l709_709441

theorem gamma_minus_alpha_eq_4pi_over_3 (α β γ : ℝ) (h₀ : 0 < α) (h₁ : α < β) (h₂ : β < γ) (h₃ : γ < 2 * π) 
  (h : ∀ x : ℝ, cos (x + α) + cos (x + β) + cos (x + γ) = 0) : γ - α = 4 * π / 3 := 
sorry

end gamma_minus_alpha_eq_4pi_over_3_l709_709441


namespace angle_DCE_invariant_l709_709976

variables {A B P C D E : Point}
variables {h : Semicircle}
variables [InsideDiameter AB P]
variables [PerpendicularThrough P AB C h]
variables [InscribedCirclesTouch AB PC h]
variables [PointsContact D E AB]

theorem angle_DCE_invariant (h : Semicircle) 
  (AB : Diameter h) (P : Point)
  (H1 : InsideDiameter AB P)
  (H2 : PerpendicularThrough P AB C h)
  (H3 : InscribedCirclesTouch AB PC h)
  (H4 : PointsContact D E AB)
  : ∠DCE = π/4 :=
sorry

end angle_DCE_invariant_l709_709976


namespace bus_speed_excluding_stoppages_l709_709401

variable (v : ℝ)

-- Given conditions
def speed_including_stoppages := 45 -- kmph
def stoppage_time_ratio := 1/6 -- 10 minutes per hour is 1/6 of the time

-- Prove that the speed excluding stoppages is 54 kmph
theorem bus_speed_excluding_stoppages (h1 : speed_including_stoppages = 45) 
                                      (h2 : stoppage_time_ratio = 1/6) : 
                                      v = 54 := by
  sorry

end bus_speed_excluding_stoppages_l709_709401


namespace equilateral_triangle_rectangle_count_l709_709212

theorem equilateral_triangle_rectangle_count (A B C : Point) (hABC : equilateral_triangle A B C) :
  ∃ n : ℕ, n = 12 ∧ count_rectangles_two_vertices_shared A B C n :=
sorry

end equilateral_triangle_rectangle_count_l709_709212


namespace find_a_values_l709_709868

theorem find_a_values (a : ℝ) :
  (∀ x : ℤ, (ax-1)*(x+2a-1) > 0 → ∃ x₁ x₂ x₃ : ℤ, x < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ (ax₁-1)*(x₁+2a-1) > 0 ∧ (ax₂-1)*(x₂+2a-1) > 0 ∧ (ax₃-1)*(x₃+2a-1) > 0) ↔ (a = -1 ∨ a = -1/2) := 
sorry

end find_a_values_l709_709868


namespace even_function_a_value_l709_709140

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, let y := (x - 1)^2 + a * x + sin(x + π / 2) in y = y) ↔ a = 2 :=
by
  let f := λ x, (x - 1)^2 + a * x + sin(x + π / 2)
  have h_even : ∀ x : ℝ, f(-x) = f(x) ↔ (a = 2) := sorry
  exact h_even

end even_function_a_value_l709_709140


namespace probability_of_two_green_apples_l709_709537

theorem probability_of_two_green_apples (total_apples green_apples choose_apples : ℕ)
  (h_total : total_apples = 8)
  (h_green : green_apples = 4)
  (h_choose : choose_apples = 2) 
: (Nat.choose green_apples choose_apples : ℚ) / (Nat.choose total_apples choose_apples) = 3 / 14 := 
by
  -- This part we would provide a proof, but for now we will use sorry
  sorry

end probability_of_two_green_apples_l709_709537


namespace maximum_eccentricity_of_intersecting_ellipse_l709_709453

noncomputable def ellipse_max_eccentricity (F₁ F₂ : ℝ × ℝ) (l : ℝ → ℝ) : ℝ :=
  let e := (λ a b c, c / a) 
  let a_min := Real.sqrt 10 / 2
  e a_min (Real.sqrt (10 - a_min^2)) 1

theorem maximum_eccentricity_of_intersecting_ellipse :
  ∀ (F₁ F₂ : ℝ × ℝ) 
    (hf₁ : F₁ = (-1, 0))
    (hf₂ : F₂ = (1, 0))
    (l : ℝ → ℝ)
    (hl : l = (λ x, x + 2)),
  ellipse_max_eccentricity F₁ F₂ l = Real.sqrt 10 / 5 := 
  sorry

end maximum_eccentricity_of_intersecting_ellipse_l709_709453


namespace find_foci_l709_709626

def is_focus_of_hyperbola (h k f_x f_y : ℝ) :=
  ∃ a b : ℝ, 
    ( a = sqrt 3 ∧
      b = sqrt 6 ∧
      (y - k) ^ 2 / a ^ 2 - (x - h) ^ 2 / b ^ 2 = 1 ∧
      (f_x, f_y) = (-5, k + sqrt (a ^ 2 + b ^ 2)) 
    ) ∨ (
      (f_x, f_y) = (-5, k - sqrt (a ^ 2 + b ^ 2))
    )

theorem find_foci : is_focus_of_hyperbola (-5) 4 (-5) 7 ∨ is_focus_of_hyperbola (-5) 4 (-5) 1 :=
sorry

end find_foci_l709_709626


namespace min_bailing_rate_l709_709047

theorem min_bailing_rate
  (distance_to_shore : ℝ)
  (water_intake_rate : ℝ)
  (boat_capacity : ℝ)
  (rowing_speed : ℝ)
  (min_bailing_rate : ℝ)
  (h_distance_to_shore : distance_to_shore = 2)
  (h_water_intake_rate : water_intake_rate = 15)
  (h_boat_capacity : boat_capacity = 50)
  (h_rowing_speed : rowing_speed = 5) :
  min_bailing_rate ≥ 13 :=
begin
  sorry
end

end min_bailing_rate_l709_709047


namespace log_eq_solution_l709_709405

theorem log_eq_solution :
  (∀ (x : ℝ), \log_x 625 = \log_3 81) → (x = 5) :=
by
  sorry

end log_eq_solution_l709_709405


namespace log_sin_sq_plus_cos_sq_eq_zero_l709_709745

theorem log_sin_sq_plus_cos_sq_eq_zero (a : ℝ) (alpha : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  log a (sin alpha ^ 2 + cos alpha ^ 2) = 0 :=
sorry

end log_sin_sq_plus_cos_sq_eq_zero_l709_709745


namespace math_problem_equivalent_l709_709853

noncomputable def M (n : ℕ) : ℤ := (Sum of coefficients of (3 * x^2 + sqrt x)^n)
noncomputable def N (n : ℕ) : ℤ := (Sum of coefficients of (3 * x^2 - sqrt x)^(n + 5))
noncomputable def P (n : ℕ) : ℤ := (Sum of coefficients of (x + 1)^n)

def problem_conditions (n : ℕ) : Prop := M n + N n - P n = 2016

def term_max_binomial_coefficient (n r : ℕ) : Prop :=
  (General term of (2 * x^2 - 1 / x^2)^(2 * n)) = -8064

def term_max_absolute_value_coeff (n r : ℕ) : Prop :=
  (General term of (2 * x^2 - 1 / x^2)^(2 * n)) = -15360 * x^8

theorem math_problem_equivalent :
  ∀ n : ℕ, problem_conditions n → (term_max_binomial_coefficient n 5) ∧ (term_max_absolute_value_coeff n 3) :=
by
  sorry

end math_problem_equivalent_l709_709853


namespace total_amount_of_money_if_all_cookies_sold_equals_1255_50_l709_709067

-- Define the conditions
def number_cookies_Clementine : ℕ := 72
def number_cookies_Jake : ℕ := 5 * number_cookies_Clementine / 2
def number_cookies_Tory : ℕ := (number_cookies_Jake + number_cookies_Clementine) / 2
def number_cookies_Spencer : ℕ := 3 * (number_cookies_Jake + number_cookies_Tory) / 2
def price_per_cookie : ℝ := 1.50

-- Total number of cookies
def total_cookies : ℕ :=
  number_cookies_Clementine + number_cookies_Jake + number_cookies_Tory + number_cookies_Spencer

-- Proof statement
theorem total_amount_of_money_if_all_cookies_sold_equals_1255_50 :
  (total_cookies * price_per_cookie : ℝ) = 1255.50 := by
  sorry

end total_amount_of_money_if_all_cookies_sold_equals_1255_50_l709_709067


namespace intersection_of_A_and_B_l709_709083

noncomputable def setA : Set ℝ := {x | 2 ^ x > 4 }
noncomputable def setB : Set ℝ := {x | abs (x - 1) < 3 }

theorem intersection_of_A_and_B : (setA ∩ setB) = { x | 2 < x ∧ x < 4 } :=
by
  sorry

end intersection_of_A_and_B_l709_709083


namespace cyclic_quadrilateral_identity_cyclic_quadrilateral_inequality_l709_709740

-- Define the properties of the quadrilateral
variables (R a b c d S : ℝ)
variables (ABCD : Prop) (is_cyclic : ABCD → Prop) (has_circumradius : is_cyclic ABCD → R = Circumradius ℝ)
variables (side_lengths : is_cyclic ABCD → (a, b, c, d) = SideLengths ℝ)
variables (area : is_cyclic ABCD → S = Area ℝ)

-- Proof problem 1
theorem cyclic_quadrilateral_identity (h : is_cyclic ABCD) :
  16 * R^2 * S^2 = (a * b + c * d) * (a * c + b * d) * (a * d + b * c) :=
sorry

-- Proof problem 2
theorem cyclic_quadrilateral_inequality (h : is_cyclic ABCD) :
  R * S * Real.sqrt 2 ≥ (a * b * c * d)^(3/4) ↔ (is_square : is_square ℝ) :=
sorry

end cyclic_quadrilateral_identity_cyclic_quadrilateral_inequality_l709_709740


namespace hospital_university_library_ages_l709_709196

-- Definitions based on conditions
variables (H L : ℕ)
def G : ℕ := 25
def U : ℕ := H - 10

-- Conditions
def condition_library : Prop := L = U + 20
def condition_sum_ages : Prop := H + G = 5 * U
def condition_future_age : Prop := G + 5 = (2 * (H + 5)) / 3

-- Statement to prove
theorem hospital_university_library_ages :
  condition_library H L ∧ condition_sum_ages H ∧ condition_future_age H :=
by
  sorry

end hospital_university_library_ages_l709_709196


namespace final_bill_is_correct_l709_709960

def Alicia_order := [7.50, 4.00, 5.00]
def Brant_order := [10.00, 4.50, 6.00]
def Josh_order := [8.50, 4.00, 3.50]
def Yvette_order := [9.00, 4.50, 6.00]

def discount_rate := 0.10
def sales_tax_rate := 0.08
def tip_rate := 0.20

noncomputable def calculate_final_bill : Float :=
  let subtotal := (Alicia_order.sum + Brant_order.sum + Josh_order.sum + Yvette_order.sum)
  let discount := discount_rate * subtotal
  let discounted_total := subtotal - discount
  let sales_tax := sales_tax_rate * discounted_total
  let pre_tax_and_discount_total := subtotal
  let tip := tip_rate * pre_tax_and_discount_total
  discounted_total + sales_tax + tip

theorem final_bill_is_correct : calculate_final_bill = 84.97 := by
  sorry

end final_bill_is_correct_l709_709960


namespace sum_exclude_multiples_of_2_and_5_l709_709077

theorem sum_exclude_multiples_of_2_and_5 (n : ℕ) (hn : n > 0) : 
  ∑ k in (finset.range (10 * n)).filter (λ x, ¬ (x % 2 = 0 ∨ x % 5 = 0)), k = 25 * n^2 :=
by
  -- The proof is omitted
  sorry

end sum_exclude_multiples_of_2_and_5_l709_709077


namespace probability_both_red_l709_709191

-- Define the conditions
def initial_red_balls : ℕ := 2
def initial_yellow_balls : ℕ := 3
def total_balls : ℕ := initial_red_balls + initial_yellow_balls

-- Define the event that both Xiaojun and Xiaojing pick red balls
def both_pick_red_event : Prop :=
  let total_outcomes := (total_balls * (total_balls - 1)) in
  let favorable_outcomes := (initial_red_balls * (initial_red_balls - 1)) in
  (favorable_outcomes / total_outcomes = 1 / 10)

-- The statement to be proven
theorem probability_both_red : both_pick_red_event :=
by
  sorry

end probability_both_red_l709_709191


namespace proof_condition1_proof_condition2_proof_condition3_proof_condition4_l709_709346

-- Definition of total doctors
def total_doctors : ℕ := 20
def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8

-- Define the specific doctors A and B
def doctor_A : ℕ := 1
def doctor_B : ℕ := 1

-- Conditions from the problem statement:
def condition1 (A B : ℕ) : Prop := A = 1 ∧ B = 1
def condition2 (A B : ℕ) : Prop := A = 0 ∧ B = 0
def condition3 (A B : ℕ) : Prop := A + B ≥ 1
def condition4 : Prop := ∃ (x y : ℕ), 1 ≤ x + y ∧ x + y ≤ 5

-- Lean statements for proof problems:
theorem proof_condition1 : ∀ (A B : ℕ), condition1 A B → ∑ k in (finset.range total_doctors).powerset, (if condition1 A B then 1 else 0) = 816 :=
by simp; exact sorry

theorem proof_condition2 : ∀ (A B : ℕ), condition2 A B → ∑ k in (finset.range total_doctors).powerset, (if condition2 A B then 1 else 0) = 8568 :=
by simp; exact sorry

theorem proof_condition3 : ∀ (A B : ℕ), condition3 A B → ∑ k in (finset.range total_doctors).powerset, (if condition3 A B then 1 else 0) = 6936 :=
by simp; exact sorry

theorem proof_condition4 : condition4 → ∑ k in (finset.range total_doctors).powerset, (if condition4 then 1 else 0) = 14656 :=
by simp; exact sorry

end proof_condition1_proof_condition2_proof_condition3_proof_condition4_l709_709346


namespace commission_sales_amount_l709_709751

/-
An agent gets a commission of 4% on the sales of cloth. 
If on a certain day, he gets Rs. 12.50 as commission,
prove that the worth of the cloth sold through him on that day is Rs. 312.50.
-/

theorem commission_sales_amount (commission rate : ℝ) (total_sales : ℝ) (commission_rate : ℝ) 
  (h_commission_rate : commission_rate = 4) (h_commission_received : commission = 12.50) :
  total_sales = 312.50 :=
by
  have rate_fraction : ℝ := commission_rate / 100
  have h_total_sales_eq : total_sales = commission / rate_fraction
  rw [h_commission_rate, h_commission_received]
  sorry

end commission_sales_amount_l709_709751


namespace range_of_a_l709_709924
noncomputable theory

def domain_is_real (a : ℝ) : Prop :=
  ∀ x : ℝ, ax^2 - 4ax + 2 > 0

theorem range_of_a (a : ℝ) : domain_is_real a ↔ a ∈ set.Ico 0 (1/2) := 
begin
  sorry
end

end range_of_a_l709_709924


namespace no_floating_point_reciprocal_floating_point_interval_floating_point_logarithmic_range_l709_709008

-- Problem 1
theorem no_floating_point_reciprocal (f : ℝ → ℝ) (h : ∀ x, f x = 1 / x) : ¬ ∃ x_0 : ℝ, f (x_0 + 1) = f x_0 + f 1 :=
sorry

-- Problem 2
theorem floating_point_interval (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2^x) : ∃ x_0 ∈ Ioo 0 1, f (x_0 + 1) = f x_0 + f 1 :=
sorry

-- Problem 3
theorem floating_point_logarithmic_range (a : ℝ) (h : ∃ x_0 ∈ Set.Ioi 0, log (a / ((x_0 + 1)^2 + 1)) = log (a / (x_0^2 + 1)) + log (a / 2)) : 3 - Real.sqrt 5 ≤ a ∧ a < 2 :=
sorry

end no_floating_point_reciprocal_floating_point_interval_floating_point_logarithmic_range_l709_709008


namespace part_one_part_two_l709_709951

-- Given conditions
variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h₁ : ∀ A C : ℝ, B = π - (A + C))
variable (h₂ : (1 / Real.tan A + 1 / Real.tan C = 1 / Real.sin B))

-- Part 1: Proving b^2 = ac
theorem part_one (h_b_squared_eq_ac : b^2 = a * c) : Prop := sorry

-- Given specific condition for part 2
variable (h_b_val : b = 2)

-- Part 2: Proving the area of triangle when B is at its maximum value
theorem part_two (h_area_eq_sqrt3 : 1 / 2 * a * c * Real.sin B = sqrt 3) : Prop := sorry

end part_one_part_two_l709_709951


namespace john_takes_away_pink_hats_l709_709939

-- Define initial conditions
def initial_pink_hats := 26
def initial_green_hats := 15
def initial_yellow_hats := 24
def pink_hats_taken_by_Carl := 4
def remaining_hats := 43

-- Define the main theorem
theorem john_takes_away_pink_hats :
  ∃ (P : ℕ), 
    22 - P + (15 - 2 * P) + 24 = 43 ∧ 
    P = 6 :=
by
  have h1 : (initial_pink_hats - pink_hats_taken_by_Carl) = 22, by
    simp [initial_pink_hats, pink_hats_taken_by_Carl]
  use 6
  split
  · 
    -- show the total remaining hats equation holds
    simp [h1]
    norm_num 
  · 
    -- show that P = 6
    refl

end john_takes_away_pink_hats_l709_709939


namespace initial_speed_of_wheel_l709_709027

theorem initial_speed_of_wheel 
  (circumference: ℝ) 
  (increase_in_speed: ℝ) 
  (time_decrease: ℝ) 
  (initial_speed: ℝ) 
  (circumference = 12) 
  (increase_in_speed = 6) 
  (time_decrease = (1 / 3) / 3600) 
  (circumference_in_miles := circumference / 5280) 
  (initial_time := circumference_in_miles / initial_speed) 
  (increased_speed := initial_speed + increase_in_speed) 
  (new_time := initial_time - time_decrease) :
  initial_speed = 12 :=
by
  sorry -- Proof omitted

end initial_speed_of_wheel_l709_709027


namespace arithmetic_sequence_and_formula_l709_709983

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709983


namespace color_cube_color_octahedron_l709_709823

theorem color_cube (colors : Fin 6) : ∃ (ways : Nat), ways = 30 :=
  sorry

theorem color_octahedron (colors : Fin 8) : ∃ (ways : Nat), ways = 1680 :=
  sorry

end color_cube_color_octahedron_l709_709823


namespace min_value_proof_l709_709850

noncomputable def min_value (x y : ℝ) : ℝ :=
x^3 + y^3 - x^2 - y^2

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 15 * x - y = 22) : 
min_value x y ≥ 1 := by
  sorry

end min_value_proof_l709_709850


namespace problem_statement_l709_709748

variables {Point : Type*} [EuclideanGeometry Point]

-- Definitions for lines and planes in Euclidean space
variable {l m : Line Point}
variable {α β : Plane Point}

-- Predicate stating that a line is perpendicular to a plane
def line_perpendicular_plane (l : Line Point) (P : Plane Point) : Prop := sorry

-- Predicate stating that a line is parallel to a plane
def line_parallel_plane (l : Line Point) (P : Plane Point) : Prop := sorry

-- Predicate stating that a plane is perpendicular to another plane
def plane_perpendicular_plane (P Q : Plane Point) : Prop := sorry

-- Predicate stating that a plane is parallel to another plane
def plane_parallel_plane (P Q : Plane Point) : Prop := sorry

-- The problem statement to prove
theorem problem_statement
  (h1 : line_perpendicular_plane l β)
  (h2 : plane_parallel_plane α β) :
  line_perpendicular_plane l α :=
sorry

end problem_statement_l709_709748


namespace equal_areas_condition_l709_709209

-- Definitions for the conditions
variables {A B C D P : Type*}
variable [ordered_ring A]

-- Dummy types for points
structure Point (A : Type*) :=
  (x y : A)

-- Definitions for midpoint and diagonals intersection
def is_midpoint (P A B : Point ℝ) : Prop :=
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

def intersect_at_midpoint (A B C D : Point ℝ) : Prop :=
  ∃ M : Point ℝ, is_midpoint M A C ∧ is_midpoint M B D

-- Definition for interior point
def is_interior (P A B C D : Point ℝ) : Prop :=
  sorry -- Definition to be filled in with appropriate geometric constraints

-- Main statement
theorem equal_areas_condition (A B C D P : Point ℝ) 
  (h_convex : is_interior P A B C D) :
  (∀ (a b c d : Point ℝ), intersect_at_midpoint A B C D ∧ is_midpoint P A C) =
  (∀ (a b c d : Point ℝ), intersect_at_midpoint A B C D ∧ is_midpoint P B D) :=
sorry

end equal_areas_condition_l709_709209


namespace pages_left_l709_709394

theorem pages_left (books : ℕ → ℕ)
  (h_books : ∀ n, books n ∈ {
    120, 150, 80, 200, 90, 180, 75, 190, 110, 160,
    130, 170, 100, 140, 210 })
  (misplaced : finset ℕ)
  (h_misplaced : misplaced = {2, 6, 10, 15})
  : finset.sum (finset.range 16 \ misplaced) books = 1305 :=
by
  sorry

end pages_left_l709_709394


namespace length_of_opposite_leg_l709_709186

noncomputable def hypotenuse_length : Real := 18

noncomputable def angle_deg : Real := 30

theorem length_of_opposite_leg (h : Real) (angle : Real) (condition1 : h = hypotenuse_length) (condition2 : angle = angle_deg) : 
 ∃ x : Real, 2 * x = h ∧ angle = 30 → x = 9 := 
by
  sorry

end length_of_opposite_leg_l709_709186


namespace indistinguishable_balls_boxes_l709_709889

noncomputable def ways_to_distribute (balls : ℕ) (boxes : ℕ) : ℕ :=
  if boxes = 2 then
    if balls = 5 then
      3
    else
      sorry  -- Not needed for this problem
  else 
    sorry  -- Not needed for this problem

theorem indistinguishable_balls_boxes
  (balls : ℕ) (boxes : ℕ) : 
  boxes = 2 → balls = 5 → ways_to_distribute balls boxes = 3 :=
by
  intros h1 h2
  simp [ways_to_distribute, h1, h2]
  sorry

end indistinguishable_balls_boxes_l709_709889


namespace collinear_P_S_Q_l709_709757

variables 
  {A B C D E F P Q S : Type*}
  [Incircle (Hex A B C D E F)]
  (hP : Intersects AC BF P)
  (hQ : Intersects CE DF Q)
  (hS : Intersects AD BE S)

theorem collinear_P_S_Q : Collinear P S Q :=
by sorry

end collinear_P_S_Q_l709_709757


namespace min_value_my_function_l709_709628

noncomputable def my_function (x : ℝ) : ℝ :=
  abs (x - 1) + 2 * abs (x - 2) + 3 * abs (x - 3) + 4 * abs (x - 4)

theorem min_value_my_function :
  ∃ (x : ℝ), my_function x = 8 ∧ (∀ y : ℝ, my_function y ≥ 8) :=
sorry

end min_value_my_function_l709_709628


namespace exist_two_non_congruent_quadrilaterals_with_equal_perimeters_and_areas_l709_709599

theorem exist_two_non_congruent_quadrilaterals_with_equal_perimeters_and_areas :
  ∃ (A B C D E : Point) (circle : Circle),
  (A ∈ circle.points ∧ B ∈ circle.points ∧ C ∈ circle.points ∧ D ∈ circle.points ∧ E ∈ circle.points) ∧
  (∀ (p : Point), p ∈ circle.points → cyclic_quadrilateral A B C p) ∧
  side_length A B ≠ side_length B C ∧ side_length B C ≠ side_length C D ∧ side_length C D ≠ side_length D A ∧ 
  side_length C D > side_length D A ∧ 
  side_length C E = side_length D A ∧ side_length E A = side_length C D ∧
  quadrilateral_perimeter A B C D = quadrilateral_perimeter A B C E ∧
  quadrilateral_area A B C D = quadrilateral_area A B C E ∧
  ¬congruent_quadrilateral A B C D A B C E :=
sorry

end exist_two_non_congruent_quadrilaterals_with_equal_perimeters_and_areas_l709_709599


namespace climb_staircase_in_93_ways_l709_709726

noncomputable def numberOfWaysToClimbStaircase (n : ℕ) (minSteps : ℕ) : ℕ :=
  ∑ k in finset.range (minSteps.succ) \ finset.range (minSteps - 1), nat.choose (n - 1) (k - 1)

theorem climb_staircase_in_93_ways :
  numberOfWaysToClimbStaircase 9 6 = 93 :=
by
  sorry

end climb_staircase_in_93_ways_l709_709726


namespace line_passes_through_fixed_point_tangent_implies_m_eq_neg1_l709_709871

open Real

noncomputable def line_eq (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y + 1 - m = 0

def circle1_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

def passes_through_fixed_point (x y : ℝ) : Prop :=
  x = 1 ∧ y = 1

theorem line_passes_through_fixed_point (m : ℝ) :
  line_eq m 1 1 :=
by
  simp [line_eq, passes_through_fixed_point]
  linarith

theorem tangent_implies_m_eq_neg1 (m : ℝ) (h : ∃ x y : ℝ, line_eq m x y ∧ circle1_eq x y) :
  m = -1 :=
by
  sorry

end line_passes_through_fixed_point_tangent_implies_m_eq_neg1_l709_709871


namespace tetrahedron_faces_property_l709_709750

theorem tetrahedron_faces_property :
  ∃ (tetrahedron : Type) 
  (faces : tetrahedron → Set Point) 
  (properties : (Set Point → Prop) → tetrahedron → Prop), 
  (properties (λ face, is_right_triangle face) tetrahedron) ∧ 
  ¬ (∀ face in faces tetrahedron, is_right_triangle face) ∧ 
  (∀ face in faces tetrahedron, is_equilateral_triangle face) ∧
  (∀ face in faces tetrahedron, is_isosceles_triangle face → is_equilateral_triangle face) ∧
  (∃ face in faces tetrahedron, is_obtuse_triangle face) := sorry

end tetrahedron_faces_property_l709_709750


namespace even_function_iff_a_eq_2_l709_709152

noncomputable def y (a x : ℝ) : ℝ :=
  (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_iff_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, y a x = y a (-x)) ↔ a = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end even_function_iff_a_eq_2_l709_709152


namespace unique_square_side_length_l709_709830

theorem unique_square_side_length (a : ℝ) (h : ∃! square, ∀ (p : ℝ × ℝ), p ∈ square.vertices → p.snd = p.fst ^ 3 + a * p.fst) :
  ∃ (s : ℝ), s = real.sqrt (real.sqrt 72) :=
sorry

end unique_square_side_length_l709_709830


namespace curve_symmetric_reflection_l709_709269

theorem curve_symmetric_reflection (f : ℝ → ℝ → ℝ) :
  (∀ x y, f x y = 0 ↔ f (y + 3) (x - 3) = 0) → 
  (∀ x y, (x - y - 3 = 0) → (f (y + 3) (x - 3) = 0)) :=
sorry

end curve_symmetric_reflection_l709_709269


namespace opposite_numbers_abs_eq_l709_709679

theorem opposite_numbers_abs_eq (a : ℚ) : abs a = abs (-a) :=
by
  sorry

end opposite_numbers_abs_eq_l709_709679


namespace mutually_exclusive_not_opposite_l709_709817

-- Define the total number of balls and their colors
def total_balls := [{color := "red"}, {color := "red"}, {color := "black"}, {color := "black"}]

-- Define the draw event as drawing two balls from the total balls
def draw_event := -- a function that takes two balls from the total balls (implementation detail for the draw_event is not specified as we are focusing on the statement)

-- Define the mutually exclusive but not opposite events
def mutually_exclusive_events (event1 event2 : bool) : Prop :=
event1 ≠ event2

theorem mutually_exclusive_not_opposite :
  let events := [("Exactly one black ball", draw_event "one_black"),
                 ("Exactly two black balls", draw_event "two_black")]
  mutate-events events.goal events.includes := (event1, event2) = (draw_event "one_black" ≠ draw_event "two_black")


end mutually_exclusive_not_opposite_l709_709817


namespace dive_score_correct_l709_709510

variable (scores : List ℝ) (degree_of_difficulty : ℝ) (performance_multiplier : ℝ) (synchronicity_factor : ℝ)

noncomputable def point_value_of_dive (scores : List ℝ) (degree_of_difficulty : ℝ) (performance_multiplier : ℝ) (synchronicity_factor : ℝ) : ℝ :=
  let filtered_scores := scores.erase (List.maximum scores).get_or_else 0 |>.erase (List.minimum scores).get_or_else 0
  let sum_scores := filtered_scores.foldl (λ x y => x + y) 0
  sum_scores * degree_of_difficulty * performance_multiplier * synchronicity_factor

theorem dive_score_correct :
  point_value_of_dive [7.5, 8.0, 9.0, 6.0, 8.8, 9.5] 3.2 1.2 0.95 = 121.4784 :=
by
  sorry

end dive_score_correct_l709_709510


namespace hotel_flat_fee_l709_709347

theorem hotel_flat_fee
  (f n : ℝ)
  (h1 : f + 3 * n = 195)
  (h2 : f + 7 * n = 380) :
  f = 56.25 :=
by sorry

end hotel_flat_fee_l709_709347


namespace total_three_digit_integers_from_set_l709_709489

theorem total_three_digit_integers_from_set 
  (S : Finset ℕ) 
  (hS : S = {4, 5, 5, 5, 6, 6, 7}) :
  (∃ n : ℕ, n = 43 ∧ ∀ x ∈ S, x ∈ {x // (∃! (x - n), x ∈ {4, 5, 5, 5, 6, 6, 7}})) :=
  sorry

end total_three_digit_integers_from_set_l709_709489


namespace indistinguishable_balls_into_boxes_l709_709888

theorem indistinguishable_balls_into_boxes : 
  ∃ n : ℕ, n = 3 ∧ (∀ (b : ℕ), (b = 5 → 
  ∃ (ways : ℕ), ways = 3 ∧ 
  (ways = 1 + 1 + 1 ∧ 
  ((∃ x y : ℕ, x + y = b ∧ (x = 5 ∧ y = 0)) ∧ 
   (∃ x y : ℕ, x + y = b ∧ (x = 4 ∧ y = 1)) ∧ 
   (∃ x y : ℕ, x + y = b ∧ (x = 3 ∧ y = 2)))))) := 
begin
  sorry
end

end indistinguishable_balls_into_boxes_l709_709888


namespace rectangle_vertex_area_y_value_l709_709729

theorem rectangle_vertex_area_y_value (y : ℕ) (hy : 0 ≤ y) :
  let A := (0, y)
  let B := (10, y)
  let C := (0, 4)
  let D := (10, 4)
  10 * (y - 4) = 90 → y = 13 :=
by
  sorry

end rectangle_vertex_area_y_value_l709_709729


namespace sqrt_170569_sqrt_175561_l709_709787

theorem sqrt_170569 : Nat.sqrt 170569 = 413 := 
by 
  sorry 

theorem sqrt_175561 : Nat.sqrt 175561 = 419 := 
by 
  sorry

end sqrt_170569_sqrt_175561_l709_709787


namespace total_cost_l709_709961

theorem total_cost
  (permits_cost : ℕ)
  (contractor_hourly_rate : ℕ)
  (contractor_days : ℕ)
  (contractor_hours_per_day : ℕ)
  (inspector_discount : ℕ)
  (h_pc : permits_cost = 250)
  (h_chr : contractor_hourly_rate = 150)
  (h_cd : contractor_days = 3)
  (h_chpd : contractor_hours_per_day = 5)
  (h_id : inspector_discount = 80)
  (contractor_total_hours : ℕ := contractor_days * contractor_hours_per_day)
  (contractor_total_cost : ℕ := contractor_total_hours * contractor_hourly_rate)
  (inspector_cost : ℕ := contractor_total_cost - (contractor_total_cost * inspector_discount / 100))
  (total_cost : ℕ := permits_cost + contractor_total_cost + inspector_cost) :
  total_cost = 2950 :=
by
  sorry

end total_cost_l709_709961


namespace distinct_natural_numbers_circles_sum_equal_impossible_l709_709957

theorem distinct_natural_numbers_circles_sum_equal_impossible :
  ¬∃ (f : ℕ → ℕ) (distinct : ∀ i j, i ≠ j → f i ≠ f j) (equal_sum : ∀ i j k, (f i + f j + f k = f (i+1) + f (j+1) + f (k+1))),
  true :=
  sorry

end distinct_natural_numbers_circles_sum_equal_impossible_l709_709957


namespace distribute_balls_in_boxes_l709_709124

theorem distribute_balls_in_boxes (balls boxes : ℕ) (h_balls : balls = 7) (h_boxes : boxes = 2) : (boxes ^ balls) = 128 :=
by
  simp [h_balls, h_boxes]
  sorry

end distribute_balls_in_boxes_l709_709124


namespace arithmetic_sequence_and_formula_l709_709981

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709981


namespace geometric_progression_sum_l709_709062

theorem geometric_progression_sum :
  ∀ (b q a d : ℝ),
    b = a →
    b * q ^ 3 = a + 3 * d →
    b * q ^ 7 = a + 7 * d →
    3 * a + 10 * d = 148 / 9 →
    b * (1 + q + q^2 + q^3) = 700 / 27 :=
by
  intros b q a d h1 h2 h3 h4
  sorry

end geometric_progression_sum_l709_709062


namespace problem_1_problem_2_l709_709107

-- Proof Problem 1: Prove A ∩ B = {x | -3 ≤ x ≤ -2} given m = -3
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2 * m - 1 ≤ x ∧ x ≤ m + 1}

theorem problem_1 : B (-3) ∩ A = {x | -3 ≤ x ∧ x ≤ -2} := sorry

-- Proof Problem 2: Prove m ≥ -1 given B ⊆ A
theorem problem_2 (m : ℝ) : (B m ⊆ A) → m ≥ -1 := sorry

end problem_1_problem_2_l709_709107


namespace find_a_if_y_is_even_l709_709165

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem find_a_if_y_is_even (a : ℝ) (h : ∀ x : ℝ, y x a = y (-x) a) : a = 2 :=
by
  sorry

end find_a_if_y_is_even_l709_709165


namespace daily_guinea_fowl_loss_l709_709732

theorem daily_guinea_fowl_loss :
  ∀ (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
    (daily_chicken_loss daily_turkey_loss : ℕ)
    (days total_birds_after_week : ℕ),
    initial_chickens = 300 →
    initial_turkeys = 200 →
    initial_guinea_fowls = 80 →
    daily_chicken_loss = 20 →
    daily_turkey_loss = 8 →
    days = 7 →
    total_birds_after_week = 349 →
    (total_birds_after_week -  
     ((initial_chickens - daily_chicken_loss * days) + 
      (initial_turkeys - daily_turkey_loss * days)) = 
     initial_guinea_fowls - (5 * days)) :=
by 
  intros initial_chickens initial_turkeys initial_guinea_fowls 
         daily_chicken_loss daily_turkey_loss 
         days total_birds_after_week 
         h_chickens h_turkeys h_guinea_fowls 
         h_daily_chickens h_daily_turkeys h_days 
         h_total_birds_after_week;
  rw [h_chickens, h_turkeys, h_guinea_fowls,
      h_daily_chickens, h_daily_turkeys,
      h_days, h_total_birds_after_week];
  sorry

end daily_guinea_fowl_loss_l709_709732


namespace max_sum_at_16_l709_709230

noncomputable def f (x : ℝ) : ℝ := 8 * Real.log x + 15 * x - x^2

def a (n : ℕ) : ℝ := f n

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

theorem max_sum_at_16 : ∃ k : ℕ, S k = Finset.univ.argmax (λ n, S n) = 16 := sorry

end max_sum_at_16_l709_709230


namespace greatest_digit_sum_base9_less_than_3000_eq_24_l709_709661

def greatest_digit_sum_base9 (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else let q := n / 9, r := n % 9 in r + greatest_digit_sum_base9 q

theorem greatest_digit_sum_base9_less_than_3000_eq_24 :
  (∀ n : ℕ, n > 0 ∧ n < 3000 → greatest_digit_sum_base9 n ≤ 24) ∧ 
  (∃ n : ℕ, n > 0 ∧ n < 3000 ∧ greatest_digit_sum_base9 n = 24) :=
begin
  -- Proof of the theorem
  sorry
end

end greatest_digit_sum_base9_less_than_3000_eq_24_l709_709661


namespace logo_enlargement_l709_709742

/-- Alexia's logo scaling problem -/
theorem logo_enlargement (original_width original_height new_width : ℕ) (h1 : original_width = 3) (h2 : original_height = 2) (h3 : new_width = 12) :
  let scaling_factor := new_width / original_width in
  new_height = original_height * scaling_factor :=
begin
  let scaling_factor := new_width / original_width,
  have scaling_factor_eq : scaling_factor = 4, by {
    simp [scaling_factor, h1, h3],
  },
  have new_height : ℕ := original_height * scaling_factor,
  rw [h2, scaling_factor_eq],
  exact 8,
end

end logo_enlargement_l709_709742


namespace ellipse_standard_eq_l709_709455

theorem ellipse_standard_eq (F1 F2 : ℝ × ℝ): 
  F1 = (-1, 0) ∧ F2 = (1, 0) ∧ 
  (∃ A B : ℝ × ℝ, A = (1, 3 / 2) ∧ B = (1, -3 / 2) ∧ dist A B = 3) →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = 4 ∧ b^2 = 3 ∧ ∀ x y : ℝ, 
  (x, y) ∈ ({p | p.1^2 / a^2 + p.2^2 / b^2 = 1} : set (ℝ × ℝ)) :=
begin
  sorry
end

end ellipse_standard_eq_l709_709455


namespace anicka_savings_l709_709756

theorem anicka_savings (x y : ℕ) (h1 : x + y = 290) (h2 : (1/4 : ℚ) * (2 * y) = (1/3 : ℚ) * x) : 2 * y + x = 406 :=
by
  sorry

end anicka_savings_l709_709756


namespace distribute_balls_into_boxes_l709_709893

theorem distribute_balls_into_boxes :
  (∃ (ways : ℕ), ways = 3 ∧
  ∀ (a b : ℕ), a + b = 5 →
    (a = 5 ∧ b = 0) ∨ (a = 4 ∧ b = 1) ∨ (a = 3 ∧ b = 2)) :=
by
  use 3
  split
  case a => rfl
  case b => sorry

end distribute_balls_into_boxes_l709_709893


namespace reduced_price_per_kg_l709_709014

theorem reduced_price_per_kg {P R : ℝ} (H1 : R = 0.75 * P) (H2 : 1100 = 1100 / P * P) (H3 : 1100 = (1100 / P + 5) * R) : R = 55 :=
by sorry

end reduced_price_per_kg_l709_709014


namespace find_a_l709_709452

noncomputable def distance_point_to_line (a y : ℝ) :=
  |3 * a - 4 * y - 2| / real.sqrt (3 ^ 2 + (-4) ^ 2)

theorem find_a (a : ℝ) (h : distance_point_to_line a 6 = 4) :
  a = 2 ∨ a = 46 / 3 :=
sorry

end find_a_l709_709452


namespace calculate_expression_l709_709761

theorem calculate_expression : 
  (π - 3.14) ^ 0 - 8 ^ (2 / 3) + (1 / 5) ^ 2 * (Real.logb 2 32) + 5 ^ (Real.logb 5 3) = 1 / 5 :=
by
  sorry

end calculate_expression_l709_709761


namespace machine_fill_time_l709_709349

theorem machine_fill_time (filled_cans : ℕ) (time_per_batch : ℕ) (total_cans : ℕ) (expected_time : ℕ)
  (h1 : filled_cans = 150)
  (h2 : time_per_batch = 8)
  (h3 : total_cans = 675)
  (h4 : expected_time = 36) :
  (total_cans / filled_cans) * time_per_batch = expected_time :=
by 
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end machine_fill_time_l709_709349


namespace exists_city_reachable_from_all_l709_709509

theorem exists_city_reachable_from_all (n : ℕ) (G : Fin n → Fin n → Prop) : 
  (∀ i j, G i j ∨ G j i) → (∃ c, ∀ j, G c j) := 
by
  induction n with
  | zero =>
      trivial
  | succ n ih =>
      sorry

end exists_city_reachable_from_all_l709_709509


namespace a_n_formula_reciprocal_sum_formula_l709_709042

section geometric_series_and_sequences

-- Definitions/conditions
def b (n : ℕ) : ℕ := 4
def S (n : ℕ) : ℕ := 7
def a : ℕ → ℕ 
| 0     := 1
| (n+1) := a n + n + 1

-- Theorem statements
/-
  Ⅰ) Prove the general formula for the sequence {a_n}, specifically:
      a_n = n(n+1)/2 for all n ∈ ℕ
-/
theorem a_n_formula (n : ℕ) : a n = n * (n + 1) / 2 :=
by
  sorry

/-
  Ⅱ) Prove the sum of the first n terms of the sequence {1/a_n}, specifically:
      sum from i = 1 to n of (1/a_i) = 2n / (n+1) for all n ∈ ℕ
-/
def reciprocal_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, 1 / (a (k+1) : ℚ))

theorem reciprocal_sum_formula (n : ℕ) : reciprocal_sum n = 2 * n / (n + 1) :=
by
  sorry

end geometric_series_and_sequences

end a_n_formula_reciprocal_sum_formula_l709_709042


namespace least_product_of_primes_greater_than_50_l709_709056

open Nat

theorem least_product_of_primes_greater_than_50 :
  ∃ p1 p2 : ℕ, (Prime p1) ∧ (Prime p2) ∧ (p1 > 50) ∧ (p2 > 50) ∧ (p1 ≠ p2) ∧ (∀ q1 q2 : ℕ, (Prime q1) ∧ (Prime q2) ∧ (q1 > 50) ∧ (q2 > 50) ∧ (q1 ≠ q2) → p1 * p2 ≤ q1 * q2) ∧ p1 * p2 = 3127 :=
sorry

end least_product_of_primes_greater_than_50_l709_709056


namespace problem_statement_l709_709258

noncomputable def omega : ℂ := sorry  -- primitive 2007th root of unity

theorem problem_statement :
  (2^2007 - 1) * (∑ j in finset.range 2000, 1 / (2 - omega ^ j)) = 2005 * 2^2006 + 1 :=
sorry

end problem_statement_l709_709258


namespace number_of_meat_lovers_pizzas_is_9_l709_709595

-- Define the conditions in Lean
variable buy1get1free_triple_cheese : Bool := true
variable buy2get1free_meat_lovers : Bool := true
variable price_per_pizza : Int := 5
variable total_cost : Int := 55
variable num_triple_cheese_pizzas : Int := 10

-- Define the number of meat lovers pizzas purchased
def number_of_meat_lovers_pizzas (buy1get1free_tc : Bool) (buy2get1free_ml : Bool) (price : Int) (total : Int) (num_tc : Int) : Int :=
  if buy1get1free_tc ∧ buy2get1free_ml then
    let cost_triple_cheese := (num_tc / 2) * price
    let remaining_cost := total - cost_triple_cheese
    let sets_of_3_meat_lovers := remaining_cost / (2 * price)
    sets_of_3_meat_lovers * 3
  else
    0

-- Statement to prove
theorem number_of_meat_lovers_pizzas_is_9
  (buy1get1free_tc : Bool)
  (buy2get1free_ml : Bool)
  (price : Int)
  (total : Int)
  (num_tc : Int) :
  number_of_meat_lovers_pizzas buy1get1free_tc buy2get1free_ml price total num_tc = 9 := by
  sorry

end number_of_meat_lovers_pizzas_is_9_l709_709595


namespace equifacial_tetrahedron_faces_acute_l709_709023

structure EquifacialTetrahedron (V : Type) [normedAddCommGroup V] [innerProductSpace ℝ V] :=
  (A B C D : V)
  (AB_eq_CD : dist A B = dist C D)
  (BC_eq_AD : dist B C = dist A D)
  (AC_eq_BD : dist A C = dist B D)
  (ABC_congruent : congruent (triangle A B C) (triangle B C D))

theorem equifacial_tetrahedron_faces_acute
  {V : Type} [normedAddCommGroup V] [innerProductSpace ℝ V]
  (T : EquifacialTetrahedron V) :
  ∀ (face : triangle V), face ∈ {triangle T.A T.B T.C, triangle T.A T.C T.D, triangle T.A T.B T.D, triangle T.B T.C T.D} →
  acute_triangle face :=
by
  sorry

end equifacial_tetrahedron_faces_acute_l709_709023


namespace topmost_triangle_values_l709_709943

-- Given values
def W1 : ℕ := 12
def W3 : ℕ := 3

-- Modulo 5 conditions for the gray triangles
def is_valid_5 (n : ℕ) : Prop := n % 5 = 0

-- Specify the mathematical equivalence and requirements for each gray triangle
def G1 (W2 W4 : ℕ) : Prop := is_valid_5 (W1 + W2 + W4)
def G2 (W2 W5 : ℕ) : Prop := is_valid_5 (W2 + W3 + W5)
def G3 (W4 W5 W6 : ℕ) : Prop := is_valid_5 (W4 + W5 + W6)

-- The main problem statement
theorem topmost_triangle_values (W2 W4 W5 W6 : ℕ) (h1: G1 W2 W4) (h2: G2 W2 W5) (h3: G3 W4 W5 W6) :
  W6 = 1 ∨ W6 = 4 ∨ W6 = 2 ∨ W6 = 0 ∨ W6 = 3 :=
begin
  sorry
end

end topmost_triangle_values_l709_709943


namespace first_calculation_problem_second_simplification_problem_l709_709381

-- Define the mathematical problem as propositions
theorem first_calculation_problem :
  16^ (1 / 2 : ℝ) + (1 / 81 : ℝ)^(-0.25) - (-1 / 2 : ℝ)^0 = 6 :=
by
  -- Calculation to be proved
  sorry
  
theorem second_simplification_problem (a b : ℝ) :
  (2 * a^(1 / 4) * b^(-1 / 3)) * (-3 * a^(-1 / 2) * b^(2 / 3)) / (-1 / 4 * a^(-1 / 4) * b^(-2 / 3)) = 24 * b^(-1 / 3) :=
by
  -- Simplification to be proved
  sorry

end first_calculation_problem_second_simplification_problem_l709_709381


namespace arrangement_count_l709_709037

def students := ["A", "B", "C", "D", "E", "F"]
def grades := ["First", "Second", "Third"]

-- Define the function to check if the arrangement is valid
def is_valid_arrangement (arrangement : List (String × String)) : Bool :=
  (("A", "First") ∈ arrangement) ∧
  (("B", "Third") ∉ arrangement) ∧
  (("C", "Third") ∉ arrangement) ∧
  (arrangement.length = 6) ∧
  ((arrangement.filter (λ p, p.snd = "First")).length = 2) ∧
  ((arrangement.filter (λ p, p.snd = "Second")).length = 2) ∧
  ((arrangement.filter (λ p, p.snd = "Third")).length = 2)

-- Define the proof problem
theorem arrangement_count : 
  ∃ (s : Finset (List (String × String))),
  (∀ a ∈ s, is_valid_arrangement a) ∧ s.card = 9 := 
sorry

end arrangement_count_l709_709037


namespace decorations_given_to_friend_l709_709236

-- Definitions of the given conditions
def boxes : ℕ := 6
def decorations_per_box : ℕ := 25
def used_decorations : ℕ := 58
def neighbor_decorations : ℕ := 75

-- The statement of the proof problem
theorem decorations_given_to_friend : 
  (boxes * decorations_per_box) - used_decorations - neighbor_decorations = 17 := 
by 
  sorry

end decorations_given_to_friend_l709_709236


namespace probability_prime_or_multiple_of_three_l709_709673

open Nat

def is_fair_die (n : ℕ) : Prop :=
  n = 8

def range_of_die (n : ℕ) (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ n

def is_prime_or_multiple_of_three (x : ℕ) : Prop :=
  Prime x ∨ x % 3 = 0

theorem probability_prime_or_multiple_of_three :
  is_fair_die 8 →
  (∀ x, range_of_die 8 x → is_prime_or_multiple_of_three x → x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 6 ∨ x = 7) →
  (5 / 8 : ℚ) =  (5 / 8 : ℚ) := by
  intros
  sorry

end probability_prime_or_multiple_of_three_l709_709673


namespace cos_angle_AND_l709_709681

structure RegularTetrahedron (V : Type) [AddCommGroup V] [Module ℝ V] :=
(A B C D : V)
(eqAB : ∥A - B∥ = ∥B - C∥)
(eqBC : ∥B - C∥ = ∥C - D∥)
(eqCD : ∥C - D∥ = ∥D - A∥)
(eqDA : ∥D - A∥ = ∥A - B∥)
(equalEdges : ∀ (X Y : V), X ∈ [A, B, C, D] -> Y ∈ [A, B, C, D] -> ∥X - Y∥ = ∥A - B∥)

def midpoint {V : Type} [AddCommGroup V] [Module ℝ V] (X Y : V) : V :=
  (X + Y) / 2

theorem cos_angle_AND (V : Type) [InnerProductSpace ℝ V] (tetra : RegularTetrahedron V) :
  let N := midpoint tetra.B tetra.C in
  ∃ (cosAND : ℝ), cosAND = 1 / 3 := by
  sorry

end cos_angle_AND_l709_709681


namespace tan_phi_sqrt_three_l709_709836

theorem tan_phi_sqrt_three (ϕ : ℝ)
  (h1 : sin (π / 2 + ϕ) = 1 / 2)
  (h2 : 0 < ϕ ∧ ϕ < π) :
  tan ϕ = sqrt 3 :=
sorry

end tan_phi_sqrt_three_l709_709836


namespace length_of_AB_l709_709682

-- Definitions based on given conditions:
variables (AB BC CD DE AE AC : ℕ)
variables (h1 : BC = 3 * CD) (h2 : DE = 8) (h3 : AC = 11) (h4 : AE = 21)

-- The theorem stating the length of AB given the conditions.
theorem length_of_AB (AB BC CD DE AE AC : ℕ)
  (h1 : BC = 3 * CD) (h2 : DE = 8) (h3 : AC = 11) (h4 : AE = 21) : AB = 5 := by
  sorry

end length_of_AB_l709_709682


namespace linda_savings_l709_709231

theorem linda_savings (S : ℝ) 
  (h1 : ∃ f : ℝ, f = 0.9 * 1/2 * S) -- She spent half of her savings on furniture with a 10% discount
  (h2 : ∃ t : ℝ, t = 1/2 * S * 1.05) -- The rest of her savings, spent on TV, had a 5% sales tax applied
  (h3 : 1/2 * S * 1.05 = 300) -- The total cost of the TV after tax was $300
  : S = 571.42 := 
sorry

end linda_savings_l709_709231


namespace pencils_per_pack_l709_709784

def packs := 28
def rows := 42
def pencils_per_row := 16

theorem pencils_per_pack (total_pencils : ℕ) : total_pencils = rows * pencils_per_row → total_pencils / packs = 24 :=
by
  sorry

end pencils_per_pack_l709_709784


namespace mod_calculation_l709_709040

theorem mod_calculation : (44 ^ 1234 + 99 ^ 567) % 7 = 3 :=
by
  have h1 : 44 % 7 = 2 := by norm_num
  have h2 : 99 % 7 = 1 := by norm_num
  have h3 : (2 ^ 1234) % 7 = 2 := by 
    calc 
      (2 ^ 1234) % 7 = (2 ^ (3 * 411 + 1)) % 7     : by rw [1234, mul_add, mul_comm 3, mul_one]
                ... = (((2 ^ 3) ^ 411) * 2) % 7  : by ring_exp
                ... = (1 ^ 411 * 2) % 7         : by rw pow_mod_eq_one_iff; norm_num
                ... = 2 % 7                     : by ring
                ... = 2                         : by norm_num 
  have h4 : (1 ^ 567) % 7 = 1 := by norm_num
  calc 
    (44 ^ 1234 + 99 ^ 567) % 7 
        = ((2 ^ 1234) + (1 ^ 567)) % 7  : by rw [← h1, ← h2]
    ... = (2 + 1) % 7                     : by rw [h3, h4]
    ... = 3                               : by norm_num 

end mod_calculation_l709_709040


namespace thursday_loaves_baked_l709_709261

theorem thursday_loaves_baked (wednesday friday saturday sunday monday : ℕ) (p1 : wednesday = 5) (p2 : friday = 10) (p3 : saturday = 14) (p4 : sunday = 19) (p5 : monday = 25) : 
  ∃ thursday : ℕ, thursday = 11 := 
by 
  sorry

end thursday_loaves_baked_l709_709261


namespace even_function_iff_a_eq_2_l709_709154

noncomputable def y (a x : ℝ) : ℝ :=
  (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_iff_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, y a x = y a (-x)) ↔ a = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end even_function_iff_a_eq_2_l709_709154


namespace PT_value_l709_709653

-- Define the basic geometric setup
variables (P Q R S T U : Point)
variables (PQ QR RP PS PT SU TU : ℝ)

-- Define the triangle PQR with given side lengths
axiom h1 : distance P Q = 5
axiom h2 : distance Q R = 6
axiom h3 : distance R P = 7

-- Define points S and T on ray PQ
axiom h4 : PS > 5
axiom h5 : PT > PS

-- Point U is such that it intersects the circumcircles of triangles PRS and QRT, with distances given
axiom h6 : distance S U = 3
axiom h7 : distance T U = 8

-- Desired statement to prove
theorem PT_value : PT = (6 + 46 * Real.sqrt 2) / 5 := sorry

end PT_value_l709_709653


namespace find_a_for_even_function_l709_709146

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, (x-1)^2 + a * x + sin (x + π / 2) = ((-x)-1)^2 + (-a * x) + sin (-x + π / 2)) →
  a = 2 :=
by
  sorry

end find_a_for_even_function_l709_709146


namespace polar_eq_C2_PQ_distance_eq_sqrt3_l709_709941

noncomputable def polar_equation_C2 (α : ℝ) : ℝ × ℝ :=
  let x := 3 + 3 * Math.cos α
  let y := 2 * Math.sin α
  (x / 3, y / 2)

theorem polar_eq_C2 : ∀ θ : ℝ, (∃ α : ℝ, polar_equation_C2 α = (2 * Math.cos θ, Math.sin α)) ↔
  Math.sqrt ((2 * Math.cos θ) - 1)^2 + (Math.sin α)^2 = 1 := by
  sorry

noncomputable def distance_between_intersections : ℝ :=
  let center_C2 := (1, 0)
  let line_C3 := (λ x y : ℝ, x - Math.sqrt 3 * y - 2 = 0)
  let d := |1 - 2| / 2
  2 * Math.sqrt 1 - d^2

theorem PQ_distance_eq_sqrt3 : distance_between_intersections = Math.sqrt 3 := by
  sorry

end polar_eq_C2_PQ_distance_eq_sqrt3_l709_709941


namespace expected_value_ξ_l709_709017

-- Define the problem conditions
def total_boys := 5
def total_girls := 2
def total_selected := 3

-- Define the random variable representing the number of girls among the selected volunteers
noncomputable def ξ_experiment : Finset ℕ := {0, 1, 2}

-- Define the probability mass function for ξ
def P (k : ℕ) : ℚ :=
  match k with
  | 0 => 2 / 7
  | 1 => 4 / 7
  | 2 => 1 / 7
  | _ => 0

-- Define the expected value formula
noncomputable def E (P : ℕ → ℚ) : ℚ :=
  ∑ k in ξ_experiment, k * P k

-- Statement of the theorem to prove
theorem expected_value_ξ : E P = 6 / 7 := by
  sorry

end expected_value_ξ_l709_709017


namespace total_earning_l709_709326

theorem total_earning (days_a days_b days_c : ℕ) (wage_ratio_a wage_ratio_b wage_ratio_c daily_wage_c total : ℕ)
  (h_ratio : wage_ratio_a = 3 ∧ wage_ratio_b = 4 ∧ wage_ratio_c = 5)
  (h_days : days_a = 6 ∧ days_b = 9 ∧ days_c = 4)
  (h_daily_wage_c : daily_wage_c = 125)
  (h_total : total = ((wage_ratio_a * (daily_wage_c / wage_ratio_c) * days_a) +
                     (wage_ratio_b * (daily_wage_c / wage_ratio_c) * days_b) +
                     (daily_wage_c * days_c))) : total = 1850 := by
  sorry

end total_earning_l709_709326


namespace coaching_start_day_l709_709369

def is_non_leap_year (year : ℕ) : Prop :=
  ¬ (year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0))

def days_in_month (month : ℕ) (is_leap_year : Bool) : ℕ :=
  if month = 1 then 31
  else if month = 2 then if is_leap_year then 29 else 28
  else if month = 3 then 31
  else if month = 4 then 30
  else if month = 5 then 31
  else if month = 6 then 30
  else if month = 7 then 31
  else if month = 8 then 31
  else if month = 9 then 30
  else if month = 10 then 31
  else if month = 11 then 30
  else 31

def total_days_from_start (day : ℕ) (month : ℕ) (is_leap_year : Bool) : ℕ :=
  let d := List.range (month - 1) |>.map (λ m => days_in_month (m + 1) is_leap_year)
  d.foldl (.+.) day

theorem coaching_start_day {year : ℕ} (h : is_non_leap_year year) : Σ' m d, total_days_from_start d m false = 245 + 4 :=
begin
  use [1, 2], -- month and day
  sorry -- proof omitted
end

end coaching_start_day_l709_709369


namespace find_x_coordinate_l709_709608

noncomputable def point_on_curve_and_line : Prop :=
  ∃ (x : ℝ), (x y : ℝ), y = 4 ∧  (sqrt((x-5)^2 + y^2) - sqrt((x+5)^2 + y^2)) = 6

theorem find_x_coordinate :
  (S : Type) [topL : topological_space S],
  (P : S) [group G].
  if P 
  (P : ℝ × ℝ) (h1 : sqrt((P.fst - 5)^2 + P.snd^2) - sqrt((P.fst + 5)^2 + P.snd^2) = 6)
    (h2 : P.snd = 4) : P.fst = -3 * sqrt(2) := 
by sorry

end find_x_coordinate_l709_709608


namespace prove_ff1_eq_0_l709_709701

def f (x : Int) : Int :=
  if x = -1 then 0
  else if x = 0 then 1
  else if x = 1 then -1
  else 0 -- default case which ideally should not be needed for this problem

theorem prove_ff1_eq_0 : f (f 1) = 0 :=
by
  have h1 : f 1 = -1 := by simp [f]
  have h2 : f (-1) = 0 := by simp [f]
  simp [h1, h2]
  sorry

end prove_ff1_eq_0_l709_709701


namespace computation_correct_l709_709041

theorem computation_correct :
  let a := 7^2 in
  let b := 4^2 in
  let c := 2 * 5 in
  let d := 3^3 in
  a - b + c - d = 16 :=
by
  let a := 49
  let b := 16
  let c := 10
  let d := 27
  sorry

end computation_correct_l709_709041


namespace vertex_of_parabola_y_axis_intersection_l709_709872

noncomputable def parabola : ℝ → ℝ := λ x, x^2 + 6 * x - 1

theorem vertex_of_parabola :
  (∃ v : ℝ × ℝ, v = (-3, -10)) ∧ (∃ axis : ℝ, axis = -3) := 
by sorry

theorem y_axis_intersection :
  ∃ p : ℝ × ℝ, p = (0, -1) := 
by sorry

end vertex_of_parabola_y_axis_intersection_l709_709872


namespace megan_eggs_l709_709233

theorem megan_eggs (E : ℕ) : 
  let total_eggs := 2 * E in
  let used_eggs := 2 + 4 in
  let remaining_after_cooking := total_eggs - used_eggs in
  let remaining_after_aunt := remaining_after_cooking / 2 in
  let meals_eggs := 3 * 3 in
  remaining_after_aunt - meals_eggs = 0 →
  total_eggs = 24 :=
by
  sorry

end megan_eggs_l709_709233


namespace division_quotient_is_correct_l709_709309

noncomputable def polynomial_division_quotient : Polynomial ℚ :=
  Polynomial.div (Polynomial.C 8 * Polynomial.X ^ 3 + 
                  Polynomial.C 16 * Polynomial.X ^ 2 + 
                  Polynomial.C (-7) * Polynomial.X + 
                  Polynomial.C 4) 
                 (Polynomial.C 2 * Polynomial.X + Polynomial.C 5)

theorem division_quotient_is_correct :
  polynomial_division_quotient =
    Polynomial.C 4 * Polynomial.X ^ 2 +
    Polynomial.C (-2) * Polynomial.X +
    Polynomial.C (3 / 2) :=
by
  sorry

end division_quotient_is_correct_l709_709309


namespace ceiling_calculation_l709_709779

theorem ceiling_calculation : ⌈4 * (7 - 1 / 3)⌉ = 27 := 
by {
  sorry
}

end ceiling_calculation_l709_709779


namespace remaining_flight_time_calculation_l709_709728

def plane_speed_in_still_air : ℝ := 450
def fuel_consumption_rate : ℝ := 9.5
def headwind : ℝ := 30
def remaining_fuel : ℝ := 6.3333

theorem remaining_flight_time_calculation :
  (remaining_fuel / fuel_consumption_rate * 60 = 40) :=
by
  have effective_speed := plane_speed_in_still_air - headwind
  let remaining_time := remaining_fuel / fuel_consumption_rate
  have minutes_remaining := remaining_time * 60
  show minutes_remaining = 40
  sorry

end remaining_flight_time_calculation_l709_709728


namespace coefficient_x2y2_expansion_l709_709408

theorem coefficient_x2y2_expansion 
  : (finset.card (finset.filter (λ (n : ℕ), n = 2) (finset.range 9))) * 
    (finset.card (finset.filter (λ (n : ℕ), n = 2) (finset.range 5))) = 168 := 
by
  sorry -- skipping the proof

end coefficient_x2y2_expansion_l709_709408


namespace positive_integer_divides_reversed_l709_709406

-- Definitions
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_special_form (n : ℕ) : Prop :=
  -- Specific forms are handled here
  n = 2178 ∨ n = 1089

-- Main theorem statement
theorem positive_integer_divides_reversed
  (n : ℕ)
  (h1 : n > 0)
  (h2 : ∀ k, 1 ≤ k ∧ k ≤ 9 → (n ∣ n.reverse_digits k)) :
  is_palindrome n ∨ is_special_form n :=
sorry

end positive_integer_divides_reversed_l709_709406


namespace max_value_l709_709840

noncomputable def f (x: ℝ) : ℝ := 4 / (4 ^ x + 2)

def a (n : ℕ) : ℝ :=
if n = 1 then 0
else ∑ k in Finset.range (n - 1), f (↑k / n)

def S (n : ℕ) : ℝ := (n - 1 : ℝ) * n / 2

theorem max_value : 
  ∃ n : ℕ, 
  let max_val := (a (n + 1)) / (2 * S n + a 6) in
  max_val = 2 / 7 ∧ (∀ m, (a (m + 1)) / (2 * S m + a 6) ≤ max_val) :=
sorry

end max_value_l709_709840


namespace solve_for_y_l709_709134

theorem solve_for_y (y : ℝ) (h_pos : y > 0) (h_eq : y^2 = 1024) : y = 32 := 
by
  sorry

end solve_for_y_l709_709134


namespace minimum_value_of_f_l709_709473

noncomputable def f (x : ℝ) : ℝ :=
  -cos (2 * x) - 8 * sin x + 9

theorem minimum_value_of_f :
  ∃ x : ℝ, sin x ∈ set.Icc (-1 : ℝ) 1 ∧ f x = 2 :=
sorry

end minimum_value_of_f_l709_709473


namespace total_ages_l709_709434

variable (Frank : ℕ) (Gabriel : ℕ)
variables (h1 : Frank = 10) (h2 : Gabriel = Frank - 3)

theorem total_ages (hF : Frank = 10) (hG : Gabriel = Frank - 3) : Frank + Gabriel = 17 :=
by
  rw [hF, hG]
  norm_num
  sorry

end total_ages_l709_709434


namespace continued_fraction_alternating_l709_709782

theorem continued_fraction_alternating : 
  let y := 3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + ...))))))
  in y = (3 + Real.sqrt 69) / 2 :=
by
  -- Proof is omitted
  sorry

end continued_fraction_alternating_l709_709782


namespace area_of_BEJK_l709_709387

/-- Problem setup: defining the points and conditions. -/
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (4, 0)
def D : ℝ × ℝ := (4, 3)
def E : ℝ × ℝ := (0, 1.5)
def G : ℝ × ℝ := (4, 1.5)
def J : ℝ × ℝ := (0, 3)
def K : ℝ × ℝ := (8/3, 2)

/-- To be proven: the area of quadrilateral BEJK is 4/3. -/
theorem area_of_BEJK : 
  (let area := 1/2 * |(B.1 * (E.2 - K.2) + E.1 * (J.2 - B.2) + J.1 * (K.2 - E.2) + K.1 * (B.2 - J.2))| 
  in area = 4/3) :=
sorry

end area_of_BEJK_l709_709387


namespace symmetric_curve_wrt_line_l709_709271

theorem symmetric_curve_wrt_line {f : ℝ → ℝ → ℝ} :
  (∀ x y : ℝ, f x y = 0 → f (y + 3) (x - 3) = 0) := by
  sorry

end symmetric_curve_wrt_line_l709_709271


namespace tangent_circle_radius_l709_709512

theorem tangent_circle_radius (O A B C : ℝ) (r1 r2 : ℝ) :
  (O = 5) →
  (abs (A - B) = 8) →
  (C = (2 * A + B) / 3) →
  r1 = 8 / 9 ∨ r2 = 32 / 9 :=
sorry

end tangent_circle_radius_l709_709512


namespace chosen_number_600_l709_709033

theorem chosen_number_600 (f : Fin 372 → ℕ) (h_subset : ∀ x, f x ∈ Finset.range 1200.succ)
  (h_diff : ∀ i j, i ≠ j → ¬ (abs (f i - f j) = 4 ∨ abs (f i - f j) = 5 ∨ abs (f i - f j) = 9)) :
  ∃ i, f i = 600 :=
by
  sorry

end chosen_number_600_l709_709033


namespace points_on_circle_l709_709658

theorem points_on_circle {P : set (ℝ × ℝ)} (hP : P.card = 2021)
  (h_collinear : ∀ (A B C : ℝ × ℝ), {A, B, C} ⊆ P → ¬ collinear {A, B, C})
  (h_concyclic : ∀ E, E ⊆ P → E.card = 5 → ∃ F, F ⊆ E ∧ F.card ≥ 4 ∧ concyclic F) :
  ∃ Γ : set (ℝ × ℝ), ∃ P' ⊆ P, P'.card ≥ 2020 ∧ ∀ (Q ∈ P'), Q ∈ Γ :=
sorry

end points_on_circle_l709_709658


namespace hammers_ordered_in_october_l709_709206

theorem hammers_ordered_in_october
  (ordered_in_june : Nat)
  (ordered_in_july : Nat)
  (ordered_in_august : Nat)
  (ordered_in_september : Nat)
  (pattern_increase : ∀ n : Nat, ordered_in_june + n = ordered_in_july ∧ ordered_in_july + (n + 1) = ordered_in_august ∧ ordered_in_august + (n + 2) = ordered_in_september) :
  ordered_in_september + 4 = 13 :=
by
  -- Proof omitted
  sorry

end hammers_ordered_in_october_l709_709206


namespace projection_correct_l709_709060

def projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_ab := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let dot_bb := b.1 * b.1 + b.2 * b.2 + b.3 * b.3
  let scalar := dot_ab / dot_bb
  (scalar * b.1, scalar * b.2, scalar * b.3)

theorem projection_correct :
  let a : ℝ × ℝ × ℝ := (4, -1, 5)
  let b : ℝ × ℝ × ℝ := (3, -2, 4)
  projection a b = (102 / 29, -68 / 29, 136 / 29) :=
by
  intros
  sorry

end projection_correct_l709_709060


namespace angle_AFD_90_l709_709956

-- Conditions
variables {A B C D E F : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]

-- Define the points and conditions
axiom point_in_triangle (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] : Type*
axiom midpoint (P Q : Type*) [metric_space P] [metric_space Q] : Type* → Type*

-- Given conditions
axiom D_inside_ABC (A B C D : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] : set.point_in_triangle A B C → Prop
axiom E_outside_ABC (A B C E : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space E] : set.set.direct_diff_half_plane A C B E → Prop
axiom DE_equal (B C D : Type*) [metric_space B] [metric_space C] [metric_space D] : ∀ (P : set.point_in_triangle B C), eq.dist B D C D → Prop
axiom angle_BDC_120 (B C D : Type*) [metric_space B] [metric_space C] [metric_space D] : ∠BDC = 120 → Prop
axiom AE_equal_CE (A C E : Type*) [metric_space A] [metric_space C] [metric_space E] : ∀ (P : set.direct_diff_half_plane C E), eq.dist A E C E → Prop
axiom angle_AEC_60 (A C E : Type*) [metric_space A] [metric_space C] [metric_space E] : ∠AEC = 60 → Prop
axiom F_is_midpoint_BE (B E F : Type*) [metric_space B] [metric_space E] [metric_space F] : midpoint B E F

-- Prove statement: ∠ A F D = 90°
theorem angle_AFD_90 (A B C D E F : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] 
  (P1: D_inside_ABC A B C D) 
  (P2: E_outside_ABC A B C E) 
  (P3: DE_equal B C D) 
  (P4: angle_BDC_120 B C D) 
  (P5: AE_equal_CE A B C E) 
  (P6: angle_AEC_60 A B E) 
  (P7: F_is_midpoint_BE B E F) : 
  ∠ A F D = 90 :=
sorry

end angle_AFD_90_l709_709956


namespace find_usual_time_l709_709304

-- Definitions based on conditions
def usual_speed : ℝ := sorry
def usual_time : ℝ := sorry
def distance : ℝ := usual_speed * usual_time

def first_half_speed : ℝ := (3 / 4) * usual_speed
def second_half_speed : ℝ := (9 / 10) * usual_speed

-- Conditions based on the problem statement
def time_first_half : ℝ := (1 / 2) * distance / first_half_speed
def time_second_half : ℝ := (1 / 2) * distance / second_half_speed
def total_slow_time : ℝ := time_first_half + time_second_half
def delay : ℝ := 7.5

axiom usual_time_condition : total_slow_time = usual_time + delay

-- Proof statement to prove usual_time is 67.5 minutes
theorem find_usual_time : usual_time = 67.5 :=
by {
  sorry
}

end find_usual_time_l709_709304


namespace problem_a2_sum_eq_364_l709_709491

noncomputable def a (n : ℕ) : ℕ :=
  (polynomial.repr ((1 + X + X^2) ^ 6)).coeff n

theorem problem_a2_sum_eq_364 :
  (a 2) + (a 4) + (a 6) + (a 8) + (a 10) + (a 12) = 364 :=
sorry

end problem_a2_sum_eq_364_l709_709491


namespace numeral_in_150th_place_l709_709671

theorem numeral_in_150th_place : 
  let s := "384615"
  ∃ n : ℕ, s.length = 6 ∧ (150 % 6 = 0) ∧ n = 6 ∧ (s.get (n-1) = '5') →
  (s[(150 % 6)-1] = '5') :=
by
  sorry

end numeral_in_150th_place_l709_709671


namespace modular_inverse_unique_l709_709799

theorem modular_inverse_unique (a : ℤ) (h : 0 ≤ a ∧ a < 23) : (5 * a ≡ 1 [MOD 23]) → a = 14 :=
by
  intro h1,
  sorry

end modular_inverse_unique_l709_709799


namespace posters_cost_is_13_l709_709320

noncomputable def posters_cost (P : ℝ) : Prop :=
  let total_money := 40
  let notebooks_cost := 3 * 4
  let bookmarks_cost := 2 * 2
  let remaining_money := total_money - 14
  let posters_total_cost := remaining_money - (notebooks_cost + bookmarks_cost)
  posters_total_cost / 2 = P

theorem posters_cost_is_13 : posters_cost 13 := by
  let P := 13
  let total_money := 40
  let notebooks_cost := 3 * 4
  let bookmarks_cost := 2 * 2
  let remaining_money := total_money - 14
  let posters_total_cost := remaining_money - (notebooks_cost + bookmarks_cost)
  have h1 : posters_total_cost / 2 = P
  sorry

end posters_cost_is_13_l709_709320


namespace corrected_mean_is_30_point_5_l709_709330

-- Definitions and conditions from the problem
def original_mean : ℕ → ℚ := λ n, 30
def number_of_observations : ℕ := 50
def incorrect_observation : ℕ := 23
def correct_observation : ℕ := 48

-- Calculate the corrected new mean given the above conditions
theorem corrected_mean_is_30_point_5 :
  let original_sum := original_mean number_of_observations * number_of_observations,
      difference := correct_observation - incorrect_observation,
      corrected_sum := original_sum + difference in
  corrected_sum / number_of_observations = 30.5 := by
  sorry

end corrected_mean_is_30_point_5_l709_709330


namespace sum_of_three_consecutive_cubes_divisible_by_9_l709_709603

theorem sum_of_three_consecutive_cubes_divisible_by_9 (n : ℕ) : 
  (n^3 + (n + 1)^3 + (n + 2)^3) % 9 = 0 := 
by
  sorry

end sum_of_three_consecutive_cubes_divisible_by_9_l709_709603


namespace paul_money_last_weeks_l709_709692

theorem paul_money_last_weeks (a b c: ℕ) (h1: a = 68) (h2: b = 13) (h3: c = 9) : 
  (a + b) / c = 9 := 
by 
  sorry

end paul_money_last_weeks_l709_709692


namespace vector_proof_l709_709436

def a : ℝ × ℝ × ℝ := (1, 0, 1)
def b : ℝ × ℝ × ℝ := (-1, 2, -3)
def c : ℝ × ℝ × ℝ := (2, -4, 6)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let factor := (dot_product u v) / (dot_product u u)
  (factor * u.1, factor * u.2, factor * u.3)

theorem vector_proof : 
  (dot_product a b ≠ 0) ∧ 
  (∃ k : ℝ, c = (-2 : ℝ) • b) ∧ 
  (dot_product a c > 0 ∧ c ≠ (dot_product a c / dot_product a a) • a) ∧ 
  (projection c a = (4, 0, 4)) :=
by 
  sorry

end vector_proof_l709_709436


namespace length_of_BC_l709_709364

noncomputable def length_BC (A B C : ℝ × ℝ) : ℝ := 
  abs (B.1 - C.1)

def parabola (x : ℝ) : ℝ := x^2 + 1

theorem length_of_BC :
  let A := (0, 1)
  let yA := parabola A.1
  ∀ a : ℝ, 
    B = (a, parabola a) →
    C = (-a, parabola (-a)) →
    B.2 = parabola a →
    C.2 = parabola (-a) →
    (B.2 = C.2) →
    (parabola a - yA) * length_BC A B C / 2 = 128 →
    length_BC A B C = 8 * real.cbrt 2 :=
by 
  sorry

end length_of_BC_l709_709364


namespace geom_mean_mult_5_l709_709188

open_locale big_operators

-- Define the problem conditions
variables {b : ℕ → ℝ} (h : ∀ i, 0 < b i) -- b is an array of positive numbers

-- Define the geometric mean of the original set
noncomputable def geom_mean (b : ℕ → ℝ) (h : ∀ i, 0 < b i) : ℝ :=
  (∏ i in finset.range 8, b i) ^ (1 / 8 : ℝ)

-- Define the geometric mean after multiplying each number by 5
noncomputable def geom_mean_after_mul (b : ℕ → ℝ) (h : ∀ i, 0 < b i) : ℝ :=
  (∏ i in finset.range 8, 5 * b i) ^ (1 / 8 : ℝ)

-- The theorem we want to prove
theorem geom_mean_mult_5 (b : ℕ → ℝ) (h : ∀ i, 0 < b i) :
  geom_mean_after_mul b h = 5 * geom_mean b h :=
by sorry

end geom_mean_mult_5_l709_709188


namespace arithmetic_sequence_max_sum_l709_709623

-- Condition: first term is 23
def a1 : ℤ := 23

-- Condition: common difference is -2
def d : ℤ := -2

-- Sum of the first n terms of the arithmetic sequence
def Sn (n : ℕ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Problem Statement: Prove the maximum value of Sn(n)
theorem arithmetic_sequence_max_sum : ∃ n : ℕ, Sn n = 144 :=
sorry

end arithmetic_sequence_max_sum_l709_709623


namespace volume_of_pyramid_in_cube_l709_709695

structure Cube :=
(side_length : ℝ)

noncomputable def base_triangle_area (side_length : ℝ) : ℝ :=
(1/2) * side_length * side_length

noncomputable def pyramid_volume (triangle_area : ℝ) (height : ℝ) : ℝ :=
(1/3) * triangle_area * height

theorem volume_of_pyramid_in_cube (c : Cube) (h : c.side_length = 2) : 
  pyramid_volume (base_triangle_area c.side_length) c.side_length = 4/3 :=
by {
  sorry
}

end volume_of_pyramid_in_cube_l709_709695


namespace part_I_part_II_part_III_l709_709442

-- Definition for complex number |z| = 1 case
def is_unit_norm (z : ℂ) : Prop := ‖z‖ = 1

-- Part (Ⅰ)
theorem part_I {a : ℝ} (z : ℂ) : is_unit_norm z → z = a + complex.i → a = 0 :=
by sorry

-- Part (Ⅱ)
theorem part_II {a : ℝ} (z : ℂ) : (z / (1 + complex.i)).im = 0 → z = a + complex.i → a = 1 :=
by sorry

-- Part (Ⅲ)
theorem part_III {a b : ℝ} (z : ℂ) :
  (∀ z, z = a + complex.i → is_root (λ x, x^2 + b*x + 2) z) →
  (a = 1 ∧ b = -2) ∨ (a = -1 ∧ b = 2) :=
by sorry

end part_I_part_II_part_III_l709_709442


namespace weighted_binomial_sum_l709_709605

theorem weighted_binomial_sum (n : ℕ) :
  (∑ k in finset.range n.succ, (k + 1) * nat.choose n (k + 1)) = n * 2 ^ (n - 1) :=
by {
  sorry
}

end weighted_binomial_sum_l709_709605


namespace find_a_for_even_function_l709_709148

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, (x-1)^2 + a * x + sin (x + π / 2) = ((-x)-1)^2 + (-a * x) + sin (-x + π / 2)) →
  a = 2 :=
by
  sorry

end find_a_for_even_function_l709_709148


namespace opponents_total_score_l709_709019

-- Defining the problem conditions
def games_played : Nat := 8
def scores : List Nat := [2, 4, 5, 7, 8, 10, 11, 13]
def lost_by_two (score: Nat) : Prop := ∃ opponent_score, opponent_score = score + 2
def won_by_triple (score: Nat) : Prop := ∃ opponent_score, score = opponent_score * 3

-- The main theorem statement
theorem opponents_total_score (h_lost_games: ∃ s₁ s₂ s₃, s₁ ∈ scores ∧ s₂ ∈ scores ∧ s₃ ∈ scores ∧ lost_by_two s₁ ∧ lost_by_two s₂ ∧ lost_by_two s₃)
                              (h_won_games: ∃ remaining_scores, ∀ s ∈ scores \ {s₁, s₂, s₃}, won_by_triple s):
  ∑ s in scores.map (λ x, if lost_by_two x then x + 2 else (x / 3) ) = 42 := 
sorry


end opponents_total_score_l709_709019


namespace square_area_proof_l709_709753

   theorem square_area_proof (x : ℝ) (h1 : 4 * x - 15 = 20 - 3 * x) :
     (20 - 3 * x) * (4 * x - 15) = 25 :=
   by
     sorry
   
end square_area_proof_l709_709753


namespace centroid_divides_l709_709597

-- Definitions based on the given conditions
structure EquilateralTriangle (α : Type) [OrderedRing α] :=
(a b c : α)
(sides_eq : a = b ∧ b = c)
(angles_eq : ∀ A B C, A + B + C = 180 ∧ A = 60 ∧ B = 60 ∧ C = 60)

noncomputable def centroid_divides_angle_bisectors (α : Type) [OrderedRing α] (T : EquilateralTriangle α) : Prop :=
  ∃ G : α, ∀ A B C, 
  let centroid := G in
  let (AG, BG, CG) := (2 * G, G * 1) in
  AG = 2 * BG ∧ BG = 2 * CG

theorem centroid_divides (α : Type) [OrderedRing α] (T : EquilateralTriangle α) :
  centroid_divides_angle_bisectors α T :=
sorry

end centroid_divides_l709_709597


namespace selling_price_is_correct_l709_709433

def wholesale_cost : ℝ := 24.35
def gross_profit_percentage : ℝ := 0.15

def gross_profit : ℝ := gross_profit_percentage * wholesale_cost
def selling_price : ℝ := wholesale_cost + gross_profit

theorem selling_price_is_correct :
  selling_price = 28.00 :=
by
  sorry

end selling_price_is_correct_l709_709433


namespace distance_from_pole_to_line_l709_709530

/-- Definition of the line in polar coordinates -/
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Definition of the pole in Cartesian coordinates -/
def pole_cartesian : ℝ × ℝ := (0, 0)

/-- Convert the line from polar to Cartesian -/
def line_cartesian (x y : ℝ) : Prop := x = 2

/-- The distance function between a point and a line in Cartesian coordinates -/
def distance_to_line (p : ℝ × ℝ) : ℝ := abs (p.1 - 2)

/-- Prove that the distance from the pole to the line is 2 -/
theorem distance_from_pole_to_line : distance_to_line pole_cartesian = 2 := by
  sorry

end distance_from_pole_to_line_l709_709530


namespace equation_solution_l709_709254

noncomputable def solveEquation (x : ℂ) : Prop :=
  (x - 4)^4 + (x - 6)^4 = 16

theorem equation_solution :
  ∀ x : ℂ, solveEquation x ↔
  x = (5 + complex.I * complex.sqrt 7) ∨
  x = (5 - complex.I * complex.sqrt 7) ∨
  x = 4 ∨
  x = 6 :=
by
  sorry

end equation_solution_l709_709254


namespace log_eq_solution_l709_709404

theorem log_eq_solution :
  (∀ (x : ℝ), \log_x 625 = \log_3 81) → (x = 5) :=
by
  sorry

end log_eq_solution_l709_709404


namespace number_of_possible_scores_l709_709511

theorem number_of_possible_scores :
  let scores : Finset ℕ := {70, 85, 88, 90, 98, 100}
  let valid_arrangements := ( 
    (scores.powerset.filter (λ s, s.card = 4)) -- selecting 4 out of 6
  ).filter (λ s, (s.min' sorry < s.erase (s.min' sorry).min' sorry) ∧ 
                  (s.erase (s.min' sorry).erase (s.erase (s.min' sorry).min' sorry).max' sorry < s.max' sorry)).card +
  ( 
    (scores.powerset.filter (λ s, s.card = 3)) -- selecting 3 out of 6
  ).filter (λ s, let list := s.toList.sort (<=) in 
                  (list.nthLe 0 sorry < list.nthLe 1 sorry) ∧ 
                  (list.nthLe 1 sorry < list.nthLe 2 sorry) ∧ 
                  (list.nthLe 2 sorry < s.max sorry)).card
  in
  valid_arrangements = 35 := 
by sorry

end number_of_possible_scores_l709_709511


namespace find_a_if_y_is_even_l709_709162

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem find_a_if_y_is_even (a : ℝ) (h : ∀ x : ℝ, y x a = y (-x) a) : a = 2 :=
by
  sorry

end find_a_if_y_is_even_l709_709162


namespace right_triangle_BFC_l709_709016

variables {A B C D E F : Type}
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry E] [EuclideanGeometry F]

-- Declare that A, B, C, D are points in the trapezoid
variable {AB : line A B}
variable {CD : line C D}
variable {AD : line A D}
variable {BC : line B C}
variable {h : ℝ} -- height of the trapezoid
variable {AB_base : ℝ} -- base AB
variable {CD_base : ℝ} -- base CD

-- Conditions of the problem
axiom parallel_AB_CD : parallel AB CD
axiom perpendicular_AD_AB : perpendicular AD AB
axiom height_eq_sum_of_bases : h = AB_base + CD_base

-- The point E is the midpoint of BC, and F is where the perpendicular bisector intersects AD
variable {E : midpoint B C}
variable {F : intersection (perpendicular_bisector B C) AD}

-- The main theorem to be proved
theorem right_triangle_BFC : angle B F C = 90 :=
by sorry

end right_triangle_BFC_l709_709016


namespace range_of_f_l709_709773

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.arcsin x

theorem range_of_f : Set.Icc (-Real.sin 1 - Real.pi / 2) (Real.sin 1 + Real.pi / 2) 
  = {y : ℝ | ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = y} :=
by 
  sorry

end range_of_f_l709_709773


namespace range_of_m_l709_709458

theorem range_of_m (m : ℝ) :
  ((1 - 2 + m) < 0) ∧ (∀ (x y : ℝ), (2 * x - y + m) ≠ (-(1 / m) * x + y - 1)) →
  m ∈ set.Ioo (-∞) (-2) ∪ set.Ioo (-2) 1 :=
by
  intro h
  sorry

end range_of_m_l709_709458


namespace dishes_with_lentils_l709_709358

theorem dishes_with_lentils :
  ∀ (total_dishes beans_and_lentils beans_and_seitan single_protein_dishes dishes_with_only_beans dishes_with_only_seitan : ℕ),
    total_dishes = 10 →
    beans_and_lentils = 2 →
    beans_and_seitan = 2 →
    single_protein_dishes = total_dishes - beans_and_lentils - beans_and_seitan →
    dishes_with_only_beans = single_protein_dishes / 2 →
    dishes_with_only_beans = 3 * dishes_with_only_seitan →
    beans_and_lentils = 2 :=
begin
  intros,
  sorry,
end

end dishes_with_lentils_l709_709358


namespace find_x_l709_709282

theorem find_x : ∃ x : ℝ, (x^3 - (0.1 : ℝ)^3) / (x^2 + 0.066 + (0.1 : ℝ)^2) = 0.5599999999999999 ∧ x ≈ 0.8 := by
  sorry

end find_x_l709_709282


namespace problem_statement_l709_709088

variable (a b : ℝ)
variable (h : IsRoot (3 * X^2 + 2 * X - 2) a)
variable (h' : IsRoot (3 * X^2 + 2 * X - 2) b)

theorem problem_statement : (2 * a / (a^2 - b^2) - 1 / (a - b)) = -3 / 2 := by
  sorry

end problem_statement_l709_709088


namespace determine_a_l709_709157

def y (x : ℝ) (a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem determine_a (a : ℝ) : is_even (y _ a) → a = 2 :=
sorry

end determine_a_l709_709157


namespace quadrilateral_area_96_97_98_99_l709_709207

noncomputable def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![-3/2, 1/2; -1/2, -1/2]

def P (n : ℕ) : ℝ × ℝ := sorry -- Define recursively based on the transformation.

def area_of_quadrilateral (P : ℕ → ℝ × ℝ) (i j k l : ℕ) : ℝ :=
  abs (P i.1 * P j.2 + P j.1 * P k.2 + P k.1 * P l.2 + P l.1 * P i.2
    - (P i.2 * P j.1 + P j.2 * P k.1 + P k.2 * P l.1 + P l.2 * P i.1)) / 2

theorem quadrilateral_area_96_97_98_99 :
  area_of_quadrilateral P 96 97 98 99 = 8 := sorry

end quadrilateral_area_96_97_98_99_l709_709207


namespace weigh_load_on_balance_scales_l709_709895

variables {W : ℝ} {n m : ℕ} 
variables (w : fin n → ℝ) (w' : fin m → ℝ)

-- Conditions
def initial_balance := W = (finset.univ.sum w : ℝ)
def rearranged_balance := (finset.univ.sum w : ℝ) = (finset.univ.sum w' : ℝ)

-- Proof statement
theorem weigh_load_on_balance_scales (h1 : initial_balance w) (h2 : rearranged_balance w w') :
  W = (finset.univ.sum w' : ℝ) :=
by sorry

end weigh_load_on_balance_scales_l709_709895


namespace monotonic_range_a_l709_709863

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (3 * x + a / x - 2) / Real.log 2

theorem monotonic_range_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x ∧ 1 ≤ y ∧ x ≤ y → f x a ≤ f y a) ↔ (-1 < a ∧ a ≤ 3) :=
sorry

end monotonic_range_a_l709_709863


namespace segment_exists_no_red_line_intersect_l709_709237

theorem segment_exists_no_red_line_intersect (P : Finset (Fin 30 → ℝ × ℝ))
  (hP : ∀ (p₁ p₂ p₃ : P), ¬(p₁.1 = p₂.1 ∧ p₂.1 = p₃.1))
  (L : Fin 7 → set (ℝ × ℝ))
  (hL : ∀ i, ∀ p ∈ P, ¬p ∈ L i) :
  ∃ (p₁ p₂ ∈ P), ∀ i, ¬∃ (x : Fin 30 → ℝ × ℝ), segment (p₁.1, p₂.1) ⊆ L i :=
by sorry

end segment_exists_no_red_line_intersect_l709_709237


namespace rotation_matrix_135_degrees_l709_709415

theorem rotation_matrix_135_degrees :
  let θ := 135 * Real.pi / 180 in
  let cos_theta := Real.cos θ in
  let sin_theta := Real.sin θ in
  cos_theta = - 1 / Real.sqrt 2 ∧ sin_theta = 1 / Real.sqrt 2 →
  (Matrix.vec3 (λ _ _, Facing.leaf ⟨Real.cos θ, -Real.sin θ⟩)
     (λ _ _, Facing.leaf ⟨Real.sin θ, Real.cos θ⟩)) =
  (Matrix.vec3 (λ _ _, Facing.leaf ⟨-1 / Real.sqrt 2, -1 / Real.sqrt 2⟩)
     (λ _ _, Facing.leaf ⟨1 / Real.sqrt 2, -1 / Real.sqrt 2⟩)) :=
by
  sorry

end rotation_matrix_135_degrees_l709_709415


namespace parallel_lines_distance_max_triangle_area_l709_709108

noncomputable def line1 (m : ℝ) : ℝ × ℝ × ℝ := (m, -2 * m, -6)
noncomputable def line2 (m : ℝ) : ℝ × ℝ × ℝ := (3 - m, m, m^2 - 3 * m)

-- Proof of Distance
theorem parallel_lines_distance (m : ℝ) (h1 : line1 m = (1, -2, 1)) (h2 : line2 m = (1, -2, 6)) :
  ∥-6 - 1∥ / real.sqrt (1^2 + (-2)^2) = real.sqrt 5 := sorry

-- Proof of Maximum Triangle Area
theorem max_triangle_area (m : ℝ) (h: 0 < m ∧ m < 3) (h3 : line2 3 / 2 = (2, 2, -3)) :
  ∃ m = 3/2, (3 - (3/2)) * (3/2) = (3 - (3/2)) * (3/2) := sorry

end parallel_lines_distance_max_triangle_area_l709_709108


namespace emma_age_proof_l709_709955

theorem emma_age_proof (Inez Zack Jose Emma : ℕ)
  (hJose : Jose = 20)
  (hZack : Zack = Jose + 4)
  (hInez : Inez = Zack - 12)
  (hEmma : Emma = Jose + 5) :
  Emma = 25 :=
by
  sorry

end emma_age_proof_l709_709955


namespace candy_store_sampling_l709_709930

theorem candy_store_sampling (total_customers caught_sampling not_caught_percent : ℝ)
  (h1 : caught_sampling = 0.22 * total_customers)
  (h2 : not_caught_percent = 0.1) :
  let total_sampling_percent := caught_sampling / (1 - not_caught_percent * total_customers) in
  total_sampling_percent = 0.2444 := 
by
  sorry

end candy_store_sampling_l709_709930


namespace arithmetic_sequence_and_formula_l709_709987

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l709_709987


namespace sum_of_elements_in_A_l709_709274

def floor (x : Real) : Int := Int.floor x

def g (x : Real) : Int := floor x + floor (2 * x)

noncomputable def A : Set Int := {y | ∃ (x : Real), 0 ≤ x ∧ x ≤ 1 ∧ g x = y}

theorem sum_of_elements_in_A : (∑ y in A, y) = 4 := 
  by 
    sorry

end sum_of_elements_in_A_l709_709274


namespace check_location_determination_l709_709029

theorem check_location_determination 
  (A : string) (B : string) (C : string) (D : string)
  (coordinates_A : A = "East longitude 118°, north latitude 40°")
  (description_B : B = "Second Ring Road in Beijing")
  (direction_C : C = "Northeast 45°")
  (venue_D : D = "Row 2 at Red Star Cinema") :
  A = "East longitude 118°, north latitude 40°" :=
by sorry

end check_location_determination_l709_709029


namespace square_fence_perimeter_l709_709253

theorem square_fence_perimeter (post_count : ℕ) (post_width_inch : ℕ) (gap_length_ft : ℕ)
  (h_posts : post_count = 16) (h_post_width : post_width_inch = 4) (h_gap_length : gap_length_ft = 6) :
  let post_width_ft := post_width_inch / 12 in
  let side_posts := (post_count - 4) / 4 + 1 in
  let side_length := (side_posts - 1) * gap_length_ft + side_posts * post_width_ft in
  4 * side_length = 77 + 1 / 3 :=
by
  sorry

end square_fence_perimeter_l709_709253


namespace dishes_with_lentils_l709_709359

theorem dishes_with_lentils :
  ∀ (total_dishes beans_and_lentils beans_and_seitan single_protein_dishes dishes_with_only_beans dishes_with_only_seitan : ℕ),
    total_dishes = 10 →
    beans_and_lentils = 2 →
    beans_and_seitan = 2 →
    single_protein_dishes = total_dishes - beans_and_lentils - beans_and_seitan →
    dishes_with_only_beans = single_protein_dishes / 2 →
    dishes_with_only_beans = 3 * dishes_with_only_seitan →
    beans_and_lentils = 2 :=
begin
  intros,
  sorry,
end

end dishes_with_lentils_l709_709359


namespace determine_a_l709_709156

def y (x : ℝ) (a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem determine_a (a : ℝ) : is_even (y _ a) → a = 2 :=
sorry

end determine_a_l709_709156


namespace symmetric_curve_wrt_line_l709_709270

theorem symmetric_curve_wrt_line {f : ℝ → ℝ → ℝ} :
  (∀ x y : ℝ, f x y = 0 → f (y + 3) (x - 3) = 0) := by
  sorry

end symmetric_curve_wrt_line_l709_709270


namespace supercomputer_transformation_stops_l709_709592

def transformation_rule (n : ℕ) : ℕ :=
  let A : ℕ := n / 100
  let B : ℕ := n % 100
  2 * A + 8 * B

theorem supercomputer_transformation_stops (n : ℕ) :
  let start := (10^900 - 1) / 9 -- 111...111 with 900 ones
  (n = start) → (∀ m, transformation_rule m < 100 → false) :=
by
  sorry

end supercomputer_transformation_stops_l709_709592


namespace tangent_addition_l709_709916

open Real

theorem tangent_addition (x : ℝ) (h : tan x = 3) :
  tan (x + π / 6) = - (5 * (sqrt 3 + 3)) / 3 := by
  -- Providing a brief outline of the proof steps is not necessary for the statement
  sorry

end tangent_addition_l709_709916


namespace consecutive_even_numbers_count_l709_709731

theorem consecutive_even_numbers_count (smallest: Int) (positive_range: Nat) : 
((smallest == -12) → (positive_range == 20) → 
∃ n : Nat, n = ((22 - (-12)) / 2 + 1) ∧ n == 18) :=
begin
  intros h_smallest h_positive_range,
  have h_largest : 22 = 2 + 20,
  { linarith [h_positive_range], },
  have h_num_terms : ∀ a b d : Int, (b - a) / d + 1 = n,
  { intros, sorry },
  use 18,
  split,
  { rw h_num_terms, linarith, },
  { refl, }
end

end consecutive_even_numbers_count_l709_709731


namespace proof_problem_l709_709129

def ceil : ℝ → ℤ := λ x, int.ceil x

def min_value_statement := (∀ x : ℝ, ceil x - x > 0)
def max_value_statement := (∀ x : ℝ, ceil x - x ≤ 1)
def existing_x_0_2_statement := ∃ x, ceil x - x = 0.2

def problem_conditions := ceil 5 = 6 ∧ ceil (-1.8) = -1

theorem proof_problem (x : ℝ) (h : problem_conditions) : (ceil 0 = 1) ∧ existing_x_0_2_statement :=
by {
  sorry
}

end proof_problem_l709_709129


namespace evaluate_g_at_6_l709_709560

def g (x : ℝ) := 3 * x^4 - 19 * x^3 + 31 * x^2 - 27 * x - 72

theorem evaluate_g_at_6 : g 6 = 666 := by
  sorry

end evaluate_g_at_6_l709_709560


namespace airplane_altitude_l709_709744

-- Defining the given conditions
def dist_AB : ℝ := 12
def angle_Alice : ℝ := 45
def angle_Bob : ℝ := 30

-- Main statement to prove that the altitude of the airplane is 4 miles
theorem airplane_altitude : ∃ h : ℝ, h = 4 ∧ (h + 2 * h = dist_AB) := 
by 
    -- Declaring a variable for altitude
    have h : ℝ := 4
    -- Assuming the conditions and deriving the fact
    use h
    split 
    { 
        -- Proving h equals 4
        exact rfl 
    }
    { 
        -- Proving that 3 times h equals 12 miles
        calc
        h + 2 * h = 3 * h : by ring
        _ = 12     : by rw [← dist_AB, three_mul h]
    }

end airplane_altitude_l709_709744


namespace find_AC_l709_709070

theorem find_AC
  (A B C D M N : Type*)
  (c m n : ℝ)
  (h1 : AB = c)
  (h2 : AM = m)
  (h3 : AN = n)
  (h4 : is_perpendicular (AD) (BC))
  (h5 : circle_center_radius (D) (DA) (intersects_side AB M) (intersects_side AC N)) :
  AC = (m * c) / n :=
sorry

end find_AC_l709_709070


namespace smallest_in_list_l709_709634

theorem smallest_in_list : 
  let l := [0.40, 0.25, 0.37, 0.05, 0.81] in
    l.min = 0.05 :=
by
  sorry

end smallest_in_list_l709_709634


namespace polynomial_roots_l709_709629

-- The statement that we need to prove
theorem polynomial_roots (a b : ℚ) (h : (2 + Real.sqrt 3) ^ 3 + 4 * (2 + Real.sqrt 3) ^ 2 + a * (2 + Real.sqrt 3) + b = 0) :
  ((Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C (4 : ℚ) * Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b) = Polynomial.X ^ 3 + Polynomial.C (4 : ℚ) * Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b) →
  (2 - Real.sqrt 3) ^ 3 + 4 * (2 - Real.sqrt 3) ^ 2 + a * (2 - Real.sqrt 3) + b = 0 ∧ -8 ^ 3 + 4 * (-8) ^ 2 + a * (-8) + b = 0 := sorry

end polynomial_roots_l709_709629


namespace grandmother_ratio_l709_709879

noncomputable def Grace_Age := 60
noncomputable def Mother_Age := 80

theorem grandmother_ratio :
  ∃ GM, Grace_Age = (3 / 8 : Rat) * GM ∧ GM / Mother_Age = 2 :=
by
  sorry

end grandmother_ratio_l709_709879


namespace smallest_next_divisor_221_l709_709966

structure Conditions (m : ℕ) :=
  (m_even : m % 2 = 0)
  (m_4digit : 1000 ≤ m ∧ m < 10000)
  (m_div_221 : 221 ∣ m)

theorem smallest_next_divisor_221 (m : ℕ) (h : Conditions m) : ∃ k, k > 221 ∧ k ∣ m ∧ k = 289 := by
  sorry

end smallest_next_divisor_221_l709_709966


namespace triangle_angle_proof_l709_709199

-- Definitions corresponding to conditions
variable (A B C D E : Point)
variable (ABC : ∠ B A C = 2 * 45)
variable (hAB : AB = AD)
variable (hCB : CB = CE)
variable (hgamma : ∠ D B E = γ)

-- Target theorem statement
theorem triangle_angle_proof (A B C D E : Point)
    (ABC : ∠ B A C = 90)
    (hAB : AB = AD)
    (hCB : CB = CE)
    (hgamma : ∠ D B E = γ) :
    γ = 45 := 
sorry

end triangle_angle_proof_l709_709199


namespace Tina_trip_distance_l709_709291

-- Define the total distance of Tina's trip
variable (d : ℝ)

-- Define the conditions
def highway_distance := d / 2
def city_distance := 30
def rural_distance := d / 4

-- Lean 4 statement to prove the total distance is 120
theorem Tina_trip_distance : 
  highway_distance d + city_distance + rural_distance d = d → d = 120 :=
by
  intro h
  sorry

end Tina_trip_distance_l709_709291


namespace grid_divisibility_l709_709810

theorem grid_divisibility (n : ℕ) (h_div : n % 7 = 0) (h_gt : n > 7) :
  ∃ m : ℕ, n * n = 7 * m ∧ (∃ arrangement : bool, arrangement = true) := sorry

end grid_divisibility_l709_709810


namespace find_angle_ACB_l709_709828

theorem find_angle_ACB
    (convex_quadrilateral : Prop)
    (angle_BAC : ℝ)
    (angle_CAD : ℝ)
    (angle_ADB : ℝ)
    (angle_BDC : ℝ)
    (h1 : convex_quadrilateral)
    (h2 : angle_BAC = 20)
    (h3 : angle_CAD = 60)
    (h4 : angle_ADB = 50)
    (h5 : angle_BDC = 10)
    : ∃ angle_ACB : ℝ, angle_ACB = 80 :=
by
  -- Here use sorry to skip the proof.
  sorry

end find_angle_ACB_l709_709828


namespace problem_alpha_beta_l_l709_709089

variables (m n l : Line) (α β : Plane)
-- m and n are skew lines
axiom skew_lines : ¬∃ (P : Point), P ∈ m ∧ P ∈ n
-- α and β are different planes
axiom diff_planes : α ≠ β
-- α is parallel to m and n
axiom alpha_parallel_m : α ∥ m
axiom alpha_parallel_n : α ∥ n
-- l is perpendicular to m and n
axiom l_perpendicular_m : l ⊥ m
axiom l_perpendicular_n : l ⊥ n
-- l is parallel to β
axiom l_parallel_beta : l ∥ β

theorem problem_alpha_beta_l :
  α ⊥ β ∧ l ⊥ α :=
by
  sorry

end problem_alpha_beta_l_l709_709089


namespace maximize_profit_l709_709343

theorem maximize_profit (n : ℕ) (h1 : n > 0) : 
  let a_n := λ n, 3 * n - 1 in
  let T_n := λ n, (3 * n^2 + n) / 2 in
  let S_n := λ n, 21 * n - (3 * n^2 + n) / 2 - 90 in
  (∀ m : ℕ, m > 0 → S_n n ≥ S_n m) → n = 7 := 
by
  sorry

end maximize_profit_l709_709343


namespace tan_alpha_does_not_exist_l709_709702

theorem tan_alpha_does_not_exist (P : ℝ × ℝ) (α : ℝ) (hα : P = (0, -4) ∧ -- Condition
    ∃ k : ℤ, α = k * π + 3/2 * π) : -- α passes through P(0, -4) means α=270° or odd multiples of 270°
  ¬ ∃ t : ℝ, tan α = t := -- tanα does not exist
by {
  sorry
}

end tan_alpha_does_not_exist_l709_709702


namespace perpendicular_vectors_magnitude_l709_709485

-- Definition of the vectors and perpendicular condition
def a : ℝ × ℝ := (x, sqrt 3)
def b : ℝ × ℝ := (3, -sqrt 3)
def perp (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0
def magnitude (a : ℝ × ℝ) : ℝ := sqrt (a.1 * a.1 + a.2 * a.2)

-- Lean 4 statement of the problem
theorem perpendicular_vectors_magnitude :
  ∃ x : ℝ, perp (x, sqrt 3) (3, -sqrt 3) ∧ magnitude (x, sqrt 3) = 2 :=
by
  sorry

end perpendicular_vectors_magnitude_l709_709485


namespace triangle_area_ratio_l709_709651

noncomputable def decagon_area_ratio (A B C : Point) (area_decagon : ℝ) : ℝ :=
  let area_triangle : ℝ := 1/5 * area_decagon
  in area_triangle / area_decagon

theorem triangle_area_ratio {A B C : Point} {area_decagon : ℝ} (regular_decagon : RegularDecagon)
  (h1 : A ∈ regular_decagon.vertices) (h2 : B ∈ regular_decagon.vertices) (h3 : C ∈ regular_decagon.vertices)
  (h4 : regular_decagon.center = O) (h5 : dist O A = dist O B)
  (h6 : dist O B = dist O C) : decagon_area_ratio A B C area_decagon = 1/5 :=
by
  sorry

end triangle_area_ratio_l709_709651


namespace triangle_area_correct_l709_709738

-- Mathematical Definitions
def R := 2 -- radius of the circle
def α := 60 * Real.pi / 180 -- first angle in radians
def β := 45 * Real.pi / 180 -- second angle in radians
def γ := 75 * Real.pi / 180 -- third angle in radians

noncomputable def triangle_area := 
  let area₁ := 1/2 * R * R * Real.sin (2 * α)
  let area₂ := 1/2 * R * R * Real.sin (2 * β)
  let area₃ := 1/2 * R * R * Real.sin (2 * γ)
  area₁ + area₂ + area₃

-- Theorem Statement
theorem triangle_area_correct :
  triangle_area = 3 + Real.sqrt 3 :=
by {
  sorry
}

end triangle_area_correct_l709_709738


namespace probability_absolute_difference_two_l709_709643

def tetrahedral_dice_faces := {1, 2, 3, 4}

noncomputable def favorable_pairs : ℕ := 4

noncomputable def total_pairs : ℕ := 16

theorem probability_absolute_difference_two :
  (favorable_pairs / (total_pairs : ℚ)) = 1 / 4 := by
{ sorry }

end probability_absolute_difference_two_l709_709643


namespace f_le_g_l709_709822

def f (n : ℕ) : ℚ := (List.range n).sum' (λ i, 1 / ((i + 1) : ℚ)^3) + 1
def g (n : ℕ) : ℚ := (1 / 2) * (3 - 1 / (n : ℚ)^2)

theorem f_le_g (n : ℕ) (h : 0 < n) : f n ≤ g n :=
by sorry

end f_le_g_l709_709822


namespace trajectory_of_M_is_ellipse_line_tangent_to_fixed_circle_l709_709094

noncomputable def fixed_point_Q : ℝ × ℝ := (real.sqrt 3, 0)

noncomputable def circle_N (x y : ℝ) : Prop :=
  (x + real.sqrt 3) ^ 2 + y ^ 2 = 24

noncomputable def trajectory_C (x y : ℝ) : Prop :=
  x ^ 2 / 6 + y ^ 2 / 3 = 1

noncomputable def fixed_circle_E (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = 2

theorem trajectory_of_M_is_ellipse :
  ∀ (P : ℝ × ℝ) (M : ℝ × ℝ),
    circle_N P.1 P.2 →
    (∃ Q : ℝ × ℝ, Q = fixed_point_Q ∧ dist M P = dist M Q) →
    trajectory_C M.1 M.2 :=
sorry

theorem line_tangent_to_fixed_circle :
  ∀ (l : ℝ → ℝ) (A B : ℝ × ℝ),
    (trajectory_C A.1 A.2 ∧ trajectory_C B.1 B.2) →
    (l A.1 = A.2 ∧ l B.1 = B.2) →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    (∃ (x y : ℝ), fixed_circle_E x y ∧ ∀ t : ℝ, l t = y + x * t) :=
sorry

end trajectory_of_M_is_ellipse_line_tangent_to_fixed_circle_l709_709094


namespace max_value_of_y_l709_709798

open Real

noncomputable def y (x : ℝ) : ℝ := 
  (sin (π / 4 + x) - sin (π / 4 - x)) * sin (π / 3 + x)

theorem max_value_of_y : 
  ∃ x : ℝ, (∀ x, y x ≤ 3 * sqrt 2 / 4) ∧ (∀ k : ℤ, x = k * π + π / 3 → y x = 3 * sqrt 2 / 4) :=
sorry

end max_value_of_y_l709_709798


namespace part1_real_values_part2_imaginary_values_l709_709427

namespace ComplexNumberProblem

-- Definitions of conditions for part 1
def imaginaryZero (x : ℝ) : Prop :=
  x^2 + 3*x + 2 = 0

def realPositive (x : ℝ) : Prop :=
  x^2 - 2*x - 2 > 0

-- Definition of question for part 1
def realValues (x : ℝ) : Prop :=
  x = -1 ∨ x = -2

-- Proof problem for part 1
theorem part1_real_values (x : ℝ) (h1 : imaginaryZero x) (h2 : realPositive x) : realValues x :=
by
  have h : realValues x := sorry
  exact h

-- Definitions of conditions for part 2
def realPartOne (x : ℝ) : Prop :=
  x^2 - 2*x - 2 = 1

def imaginaryNonZero (x : ℝ) : Prop :=
  x^2 + 3*x + 2 ≠ 0

-- Definition of question for part 2
def imaginaryValues (x : ℝ) : Prop :=
  x = 3

-- Proof problem for part 2
theorem part2_imaginary_values (x : ℝ) (h1 : realPartOne x) (h2 : imaginaryNonZero x) : imaginaryValues x :=
by
  have h : imaginaryValues x := sorry
  exact h

end ComplexNumberProblem

end part1_real_values_part2_imaginary_values_l709_709427


namespace determine_g_l709_709440

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

def is_linear (g : ℝ → ℝ) : Prop := ∃ (a b : ℝ), g = λ x, a * x + b

theorem determine_g (g : ℝ → ℝ) (h_linear : is_linear g) (h_fg : ∀ x : ℝ, f (g x) = 4 * x^2) :
  g = λ x, 2 * x + 1 ∨ g = λ x, -2 * x + 1 :=
  sorry

end determine_g_l709_709440


namespace number_of_valid_m_l709_709958

def is_right_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  (Qx - Px) * (Qx - Px) + (Qy - Py) * (Qy - Py) + (Rx - Qx) * (Rx - Qx) + (Ry - Qy) * (Ry - Qy) ==
  (Px - Rx) * (Px - Rx) + (Py - Ry) * (Py - Ry) + 2 * ((Qx - Px) * (Rx - Qx) + (Qy - Py) * (Ry - Qy))

def legs_parallel_to_axes (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  Px = Qx ∨ Px = Rx ∨ Qx = Rx ∧ Py = Qy ∨ Py = Ry ∨ Qy = Ry

def medians_condition (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  let M_PQ := ((Px + Qx) / 2, (Py + Qy) / 2);
  let M_PR := ((Px + Rx) / 2, (Py + Ry) / 2);
  (M_PQ.2 = 3 * M_PQ.1 + 1) ∧ (M_PR.2 = 2)

theorem number_of_valid_m (a b c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (P := (a, b)) (Q := (a, b+2*c)) (R := (a-2*d, b)) :
  is_right_triangle P Q R →
  legs_parallel_to_axes P Q R →
  medians_condition P Q R →
  ∃ m, m = 1 :=
sorry

end number_of_valid_m_l709_709958


namespace chocolates_problem_l709_709534

theorem chocolates_problem 
  (Ingrid_starts : ℕ)
  (Jin_starts : ℕ = 0)
  (Brian_starts : ℕ = 0)
  (Ingrid_gives_to_Jin : ℕ := Ingrid_starts / 3)
  (Jin_gives_to_Brian : ℕ := 8)
  (Jin_eats_half : ℕ := (Ingrid_gives_to_Jin - Jin_gives_to_Brian) / 2)
  (Jin_ends_up : ℕ := 5) :
  Ingrid_starts = 54 :=
by
  sorry

end chocolates_problem_l709_709534


namespace dot_product_eq_l709_709378

def vec1 : ℝ^4 := ![4, -5, 6, -7]
def vec2 : ℝ^4 := ![-6, 3, -2, 8]

theorem dot_product_eq : vec1.dot vec2 = -107 := by
  sorry

end dot_product_eq_l709_709378


namespace find_x_in_diagram_l709_709945

def triangle_angle_sum (A B C: ℝ) : Prop :=
  A + B + C = 180

theorem find_x_in_diagram 
  (angle_ABC : ℝ) (h_ABC : angle_ABC = 45)
  (angle_ACB : ℝ) (h_ACB : angle_ACB = 90)
  (angle_CDE : ℝ) (h_CDE : angle_CDE = 72) : x = 153 :=
by
  let angle_BAC := 180 - angle_ABC - angle_ACB
  have h_BAC : angle_BAC = 45 := by
    sorry
  let angle_ADE := 180 - angle_CDE
  have h_ADE : angle_ADE = 108 := by
    sorry
  let angle_AED := 180 - angle_ADE - angle_BAC
  have h_AED : angle_AED = 27 := by
    sorry
  let angle_DEB := 180 - angle_AED
  show angle_DEB = x ≧ 153/
  sorry

end find_x_in_diagram_l709_709945


namespace children_got_off_bus_l709_709705

theorem children_got_off_bus :
  ∀ (initial_children final_children new_children off_children : ℕ),
    initial_children = 21 → final_children = 16 → new_children = 5 →
    initial_children - off_children + new_children = final_children →
    off_children = 10 :=
by
  intro initial_children final_children new_children off_children
  intros h_init h_final h_new h_eq
  sorry

end children_got_off_bus_l709_709705


namespace value_of_a_plus_b_l709_709477

noncomputable def inequality_solution (a b : ℝ) (x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 < 0

theorem value_of_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < b → inequality_solution a b x) →
  (a = 1) → (b = 2) →
  a + b = 3 :=
by
  intros h₁ h₂ h₃
  rw [h₂, h₃]
  exact rfl

end value_of_a_plus_b_l709_709477


namespace polynomial_is_perfect_cube_l709_709409

theorem polynomial_is_perfect_cube (p q n : ℚ) :
  (∃ a : ℚ, x^3 + p * x^2 + q * x + n = (x + a)^3) ↔ (q = p^2 / 3 ∧ n = p^3 / 27) :=
by sorry

end polynomial_is_perfect_cube_l709_709409


namespace times_faster_l709_709689

variable (x y : ℝ)

theorem times_faster (h1 : ∃ k : ℝ, x = k * y)
                    (h2 : y = 1 / 80)
                    (h3 : x + y = 1 / 20) : 
                    ∃ k : ℝ, k = 3 :=
by
  obtain ⟨k, hk⟩ := h1
  rw [hk, h2] at h3
  have h4 : (k * (1 / 80) + (1 / 80)) = 1 / 20 := h3
  have h5 : (k + 1) * (1 / 80) = 1 / 20 := by simp [h4]
  have h6 : (k + 1) = 80 / 20 := by linarith
  have h7 : (k + 1) = 4 := by norm_num [h6]
  have h8 : k = 3 := by linarith [h7]
  exact ⟨k, h8.symm⟩

end times_faster_l709_709689


namespace smallest_five_digit_multiple_of_18_l709_709425

def is_multiple_of (x : ℕ) (k : ℕ) : Prop := ∃ n : ℕ, x = k * n

theorem smallest_five_digit_multiple_of_18 : 
  ∃ x : ℕ, 
    (10000 ≤ x ∧ x < 100000) ∧ 
    is_multiple_of x 18 ∧ 
    (∀ y : ℕ, (10000 ≤ y ∧ y < 100000) ∧ is_multiple_of y 18 → x ≤ y) :=
begin
  use 10008,
  -- The details of the proof are omitted.
  sorry
end

end smallest_five_digit_multiple_of_18_l709_709425


namespace part1_part2_l709_709437

variables (a b : ℝ × ℝ) (m : ℝ)

def vector_a : ℝ × ℝ := (1, 0)
def vector_b : ℝ × ℝ := (2, 1)
def v1 : ℝ × ℝ := (1 / 2 * vector_a.1 + vector_b.1, 1 / 2 * vector_a.2 + vector_b.2)
def v2 : ℝ × ℝ := (vector_a.1 + m * vector_b.1, vector_a.2 + m * vector_b.2)

theorem part1 :
  (v1 • v2 = 0) → m = -5 / 12 := by sorry

theorem part2 :
  (v1 • v2 > 0) → (∀ m, m > -5 / 12 ∧ m ≠ 2) := by sorry

end part1_part2_l709_709437


namespace complex_expression_simplified_l709_709604

-- Prove that 3(2 + i) - 2i(3 - i) = 4 - 3i given that i^2 = -1

noncomputable def simplify_complex_expression : Prop :=
  let i := Complex.I in
  (3 * (2 + i) - 2 * i * (3 - i)) = (4 - 3 * i)

theorem complex_expression_simplified : simplify_complex_expression :=
begin
  sorry
end

end complex_expression_simplified_l709_709604


namespace solve_expr_l709_709316

theorem solve_expr (x : ℝ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end solve_expr_l709_709316


namespace red_pieces_count_l709_709646

-- Define the conditions
def total_pieces : ℕ := 3409
def blue_pieces : ℕ := 3264

-- Prove the number of red pieces
theorem red_pieces_count : total_pieces - blue_pieces = 145 :=
by sorry

end red_pieces_count_l709_709646


namespace b_arithmetic_a_formula_l709_709997

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l709_709997


namespace inequality_solution_range_of_a_l709_709101

noncomputable def f (x : ℝ) : ℝ := abs (x + 2) - 2 * (abs (x - 1))

-- Statement for problem (1)
theorem inequality_solution (x : ℝ) : 
  f(x) ≥ -2 ↔ (-2/3) ≤ x ∧ x ≤ 6 := 
sorry

-- Statement for problem (2)
theorem range_of_a (x a : ℝ) : 
  (∀ x, f(x) ≤ x - a) ↔ a ≤ -2 := 
sorry

end inequality_solution_range_of_a_l709_709101


namespace solution_set_of_gx_lt_0_l709_709465

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := f_inv (1 - x) - f_inv (1 + x)

theorem solution_set_of_gx_lt_0 : { x : ℝ | g x < 0 } = Set.Ioo 0 1 := by
  sorry

end solution_set_of_gx_lt_0_l709_709465


namespace ratio_of_efficiencies_l709_709323

-- Definitions of efficiencies
def efficiency (time : ℕ) : ℚ := 1 / time

-- Conditions:
def E_C : ℚ := efficiency 20
def E_D : ℚ := efficiency 30
def E_A : ℚ := efficiency 18
def E_B : ℚ := 1 / 36 -- Placeholder for efficiency of B to complete the statement

-- The proof goal
theorem ratio_of_efficiencies (h1 : E_A + E_B = E_C + E_D) : E_A / E_B = 2 :=
by
  -- Placeholder to structure the format, the proof will be constructed here
  sorry

end ratio_of_efficiencies_l709_709323


namespace log_x_y_eq_sqrt_3_l709_709607

variable (x y z : ℝ)
variable (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
variable (h1 : x ^ (Real.log z / Real.log y) = 2)
variable (h2 : y ^ (Real.log x / Real.log y) = 4)
variable (h3 : z ^ (Real.log y / Real.log x) = 8)

theorem log_x_y_eq_sqrt_3 : Real.log y / Real.log x = Real.sqrt 3 :=
by
  sorry

end log_x_y_eq_sqrt_3_l709_709607


namespace modular_inverse_unique_l709_709800

theorem modular_inverse_unique (a : ℤ) (h : 0 ≤ a ∧ a < 23) : (5 * a ≡ 1 [MOD 23]) → a = 14 :=
by
  intro h1,
  sorry

end modular_inverse_unique_l709_709800


namespace digit_at_150_is_five_l709_709668

theorem digit_at_150_is_five : 
  ∀ (n : ℕ), 
  (0 < n) → 
  (n = 150) → 
  (decimal_repr (5 / 13) = "0.384615") → 
  (repeats_every_six_digit : ("0.384615" / 6 = 25)) → -- repeats every 6 digits
  (nth_digit n (decimal_repr (5 / 13)) = 5) := 
by
  intros
  sorry

end digit_at_150_is_five_l709_709668


namespace not_basic_logic_structure_l709_709677

def SequenceStructure : Prop := true
def ConditionStructure : Prop := true
def LoopStructure : Prop := true
def DecisionStructure : Prop := true

theorem not_basic_logic_structure : ¬ (SequenceStructure ∨ ConditionStructure ∨ LoopStructure) -> DecisionStructure := by
  sorry

end not_basic_logic_structure_l709_709677


namespace smallest_d_in_range_l709_709805

theorem smallest_d_in_range (d : ℝ) : (∃ x : ℝ, x^2 + 5 * x + d = 5) ↔ d ≤ 45 / 4 := 
sorry

end smallest_d_in_range_l709_709805


namespace problem_statement_cubic_number_l709_709500

noncomputable def smallest_cubic_number_divisible_by (n : ℕ) : ℕ :=
  if h : n > 0 then (finset.range (n * n)).filter (λ m, m % n = 0 ∧ ∃ k, m = k^3).min' 
     ((finset.range (n * n)).filter_nonempty 
     ⟨n^3, by simp [h, pow_succ, mul_assoc, show 0 < 2 by norm_num, zero_lt_iff_ne_zero.2 h]⟩)
  else 0

theorem problem_statement_cubic_number : smallest_cubic_number_divisible_by 810 = 729000 := 
sorry

end problem_statement_cubic_number_l709_709500


namespace angle_OMA_half_diff_C_B_l709_709005

theorem angle_OMA_half_diff_C_B 
  (O A B C M : Point)
  (h_circumscribed : is_circumscribed O A B C)
  (h_midpoint_arc : midpoint_arc_not_containing O A B C M) :
  angle O M A = (angle C - angle B) / 2 := 
sorry

end angle_OMA_half_diff_C_B_l709_709005


namespace part_I_part_II_l709_709866

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)

theorem part_I (x : ℝ) : (f x > 5) ↔ (x < -3 ∨ x > 2) :=
  sorry

theorem part_II (a : ℝ) : (∀ x, f x < a ↔ false) ↔ (a ≤ 3) :=
  sorry

end part_I_part_II_l709_709866


namespace shannon_bracelets_l709_709247

theorem shannon_bracelets (total_stones : ℕ) (stones_per_bracelet : ℕ) (h1 : total_stones = 48) (h2 : stones_per_bracelet = 8) : total_stones / stones_per_bracelet = 6 :=
by
  rw [h1, h2]
  norm_num
  sorry

end shannon_bracelets_l709_709247


namespace johns_total_earnings_l709_709947

noncomputable def hourly_wage (extra_earnings : ℝ) (extra_hours : ℕ) : ℝ :=
  extra_earnings / extra_hours

def total_earnings (hours_week1 : ℕ) (hours_week2 : ℕ) (wage : ℝ) : ℝ :=
  (hours_week1 + hours_week2) * wage

theorem johns_total_earnings
  (hours_week2 : ℕ)
  (hours_week1 : ℕ)
  (extra_earnings : ℝ)
  (wage : ℝ)
  (H1 : hours_week2 = 30)
  (H2 : hours_week1 = 20)
  (H3 : extra_earnings = 102.75)
  (H4 : wage = hourly_wage extra_earnings 10) :
  total_earnings hours_week1 hours_week2 wage = 513.75 :=
by
  rw [H1, H2, H3, H4, hourly_wage, total_earnings]
  simp
  sorry

end johns_total_earnings_l709_709947


namespace english_but_not_french_l709_709585

theorem english_but_not_french (total_students : ℕ)
  (both_classes : ℕ)
  (english_french_ratio : ℕ → ℕ)
  (total_students = 30)
  (both_classes = 2)
  (english_french_ratio = λ x, 3 * (x + 2) - 2)
  : (english_french_ratio 6 - 2) = 20 :=
by
  sorry

end english_but_not_french_l709_709585


namespace remainder_division_l709_709210

noncomputable def Q (x : ℝ) : ℝ := sorry -- Placeholder definition for Q(x)

theorem remainder_division (Q : ℝ → ℝ) (h1 : Q 19 = 16) (h2 : Q 15 = 8) :
  ∃ (c d : ℝ), c = 2 ∧ d = -22 ∧ ∀ x, Q x = (x - 15) * (x - 19) * (R x) + c * x + d :=
sorry -- Proof to be completed

end remainder_division_l709_709210
