import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.Complex.Residue
import Mathlib.Analysis.Integral.IntervalIntegral
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace xn_plus_inv_xn_l250_250750

theorem xn_plus_inv_xn (θ : ℝ) (x : ℝ) (n : ℕ) (h₀ : 0 < θ) (h₁ : θ < π / 2)
  (h₂ : x + 1 / x = -2 * Real.sin θ) (hn_pos : 0 < n) :
  x ^ n + x⁻¹ ^ n = -2 * Real.sin (n * θ) := by
  sorry

end xn_plus_inv_xn_l250_250750


namespace function_C_is_quadratic_l250_250511

def isQuadratic (f : ℝ → ℝ) :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

def function_C (x : ℝ) : ℝ := (x + 1)^2 - 5

theorem function_C_is_quadratic : isQuadratic function_C :=
by
  sorry

end function_C_is_quadratic_l250_250511


namespace probability_all_even_before_odd_l250_250925

/-- Prove that in a fair 8-sided die rolled repeatedly until an odd number appears,
    the probability that each even number (2, 4, 6, 8) appears at least once before
    the first occurrence of any odd number is 1/70. -/
theorem probability_all_even_before_odd :
  let die_faces := {1, 2, 3, 4, 5, 6, 7, 8} in
  let even_faces := {2, 4, 6, 8} in
  let odd_faces := {1, 3, 5, 7} in
  let prob_even := 1 / 2 in
  let prob_odd := 1 / 2 in
  probability
    (λ ω, (∀ e ∈ even_faces, e ∈ ω) ∧ (∀ o ∈ odd_faces, o ∉ ω))
    (repeat die_faces)
  = 1 / 70 := sorry

end probability_all_even_before_odd_l250_250925


namespace goods_train_speed_correct_l250_250900

noncomputable def goods_train_speed 
  (man_train_speed : ℚ) 
  (pass_time_seconds : ℚ) 
  (goods_train_length_meters : ℚ) : ℚ :=
  let relative_speed_kmh := man_train_speed + (goods_train_length_meters / pass_time_seconds) * 3.6 in
  relative_speed_kmh - man_train_speed

theorem goods_train_speed_correct :
  goods_train_speed 30 9 280 = 82 := 
  sorry

end goods_train_speed_correct_l250_250900


namespace sum_first_60_terms_l250_250870

theorem sum_first_60_terms (a : ℕ → ℝ) (h : ∀ n, a (n + 1) + (-1)^n * a n = 2 * n - 1) :
  (∑ n in finset.range 60, a n) = 1830 := 
sorry

end sum_first_60_terms_l250_250870


namespace number_of_elements_in_M_intersect_N_l250_250421

noncomputable theory

def M : Set (ℝ × ℝ) := {p | ∃ k k' : ℤ, p = (k, k') ∧ tan (π * k') = 0 ∧ sin (π * k) = 0}
def N : Set (ℝ × ℝ) := {p | p.fst^2 + p.snd^2 ≤ 2}

theorem number_of_elements_in_M_intersect_N : 
  ∃ n : ℕ, n = 9 ∧ ∀ p : ℝ × ℝ, p ∈ M ∩ N → n = 9 :=
sorry

end number_of_elements_in_M_intersect_N_l250_250421


namespace domain_of_func_l250_250045

open Set

def f (x : ℝ) : ℝ := log 2 (1 - x^2)

theorem domain_of_func : {x : ℝ | 1 - x^2 > 0} = Ioo (-1 : ℝ) (1 : ℝ) := by
  sorry

end domain_of_func_l250_250045


namespace number_of_perfect_square_factors_of_180_l250_250303

theorem number_of_perfect_square_factors_of_180 :
  let prime_factors := (2, 2, 1); 
  let conditions (p1 p2 p3 : ℕ) := 
    p1 ∈ {0, 2} ∧ 
    p2 ∈ {0, 2} ∧ 
    p3 ∈ {0}
  in Σ' x, x ∈ { (p1, p2, p3) | conditions p1 p2 p3 } = 4 :=
begin
  sorry
end

end number_of_perfect_square_factors_of_180_l250_250303


namespace bbq_guests_l250_250960

theorem bbq_guests (burger_cook_time total_cook_time guests_half : ℕ) 
  (total_burgers_per_set : ℕ) 
  (total_burgers : ℕ) 
  (h_burger_time : burger_cook_time = 8)
  (h_burgers_per_set : total_burgers_per_set = 5)
  (h_total_time : total_cook_time = 72)
  (h_total_burgers : total_burgers = 45)
  (h_guests_half : guests_half * 2 = guests_half + guests_half) :
  let G := guests_half * 2,
  (3 * G / 2) = total_burgers →
  G = 30 :=
begin
  sorry
end

end bbq_guests_l250_250960


namespace count_integers_in_interval_l250_250737

theorem count_integers_in_interval :
  {n : ℤ | -5 < n ∧ n < 3}.finite ∧ {n : ℤ | -5 < n ∧ n < 3}.to_finset.card = 7 := by
sorry

end count_integers_in_interval_l250_250737


namespace factorize_xcube_minus_x_l250_250993

theorem factorize_xcube_minus_x (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by 
  sorry

end factorize_xcube_minus_x_l250_250993


namespace find_n_modulo_23_l250_250500

theorem find_n_modulo_23 :
  ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ 123456 % 23 = n :=
by
  use 9
  split
  · linarith
  split
  · linarith
  · sorry

end find_n_modulo_23_l250_250500


namespace six_digit_divisible_by_72_l250_250140

theorem six_digit_divisible_by_72 (n m : ℕ) (h1 : n = 920160 ∨ n = 120168) :
  (∃ (x y : ℕ), 10 * x + y = 2016 ∧ (10^5 * x + n * 10 + m) % 72 = 0) :=
by
  sorry

end six_digit_divisible_by_72_l250_250140


namespace max_min_f_when_m_eq_1_interval_of_monotonic_increasing_l250_250812

-- Defining the function f(x) with the parameter m
def f (m : ℝ) (x : ℝ) : ℝ := -((1:ℝ)/3) * x^3 + x^2 + (m^2 - 1) * x

-- First problem: proving the maximum and minimum values when m = 1
theorem max_min_f_when_m_eq_1 :
  ∃ (x_max x_min : ℝ), x_max = -3 ∧ x_min = 0 ∧ 
  ∀ x ∈ Set.Icc (-3 : ℝ) 2, f 1 x ≤ 18 ∧ f 1 x ≥ 0 :=
sorry

-- Second problem: proving the interval of monotonic increase
theorem interval_of_monotonic_increasing (m: ℝ) (hm : 0 < m) :
  ∀ (x : ℝ), 1 - m < x ∧ x < m + 1 → 
  ∀ y ∈ {y | f m y ≤ f m (y + 1)}, monotone_on (f m) (Set.Ioo (1 - m) (m + 1)) :=
sorry

end max_min_f_when_m_eq_1_interval_of_monotonic_increasing_l250_250812


namespace custom_op_evaluation_l250_250799

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem custom_op_evaluation : custom_op 4 2 = 26 := 
by 
  sorry

end custom_op_evaluation_l250_250799


namespace geometric_sequence_a4_a5_l250_250356

open BigOperators

theorem geometric_sequence_a4_a5 (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 9) : 
  a 4 + a 5 = 27 ∨ a 4 + a 5 = -27 :=
sorry

end geometric_sequence_a4_a5_l250_250356


namespace max_abs_sum_sqrt2_l250_250329

theorem max_abs_sum_sqrt2 (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (a : ℝ), (a = |x| + |y| ∧ a ≤ 2 * Real.sqrt 2) ∧ 
             ∀ (x y : ℝ), x^2 + y^2 = 4 → (|x| + |y|) ≤ 2 * Real.sqrt 2 :=
sorry

end max_abs_sum_sqrt2_l250_250329


namespace range_of_k_l250_250459

variable (f : ℝ → ℝ := λ x, x * abs x)
variable (x k : ℝ)

theorem range_of_k (h1 : x ∈ set.Ici 1) (h2 : ∃ x ∈ set.Ici 1, f (x - 2 * k) - k < 0) :
  k ∈ set.Ioi (1/4) := 
sorry

end range_of_k_l250_250459


namespace parabola_y_range_l250_250797

theorem parabola_y_range
  (x y : ℝ)
  (M_on_C : x^2 = 8 * y)
  (F : ℝ × ℝ)
  (F_focus : F = (0, 2))
  (circle_intersects_directrix : F.2 + y > 4) :
  y > 2 :=
by
  sorry

end parabola_y_range_l250_250797


namespace standard_equation_of_ellipse_l250_250048

theorem standard_equation_of_ellipse (a b: ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b)
    (h₃ : ∀ x y : ℝ, (x + 2 * y - 4 = 0) → (x = 4 ∨ y = 2)) :
    (a = 4 ∧ b = 2) → (∀ x y : ℝ, (x, y) ∈ (C : ℝ × ℝ) ↔ x^2 / 16 + y^2 / 4 = 1) :=
    sorry

end standard_equation_of_ellipse_l250_250048


namespace trig_identity_l250_250321

variable (α : Real)
variable (h : Real.tan α = 2)

theorem trig_identity :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := by
  sorry

end trig_identity_l250_250321


namespace ellipse_slope_l250_250237

theorem ellipse_slope (a b : ℝ) (ha : a > b) (hb : b > 0)
  (e : ℝ) (he : e = (Real.sqrt 2) / 2)
  (p : ℝ × ℝ) (hp : p = (1, Real.sqrt 2 / 2))
  (A B E F : ℝ × ℝ)
  (hA : A = (Real.sqrt 2, 0))
  (hB : B = (0, 1))
  (hE : E = ((2 * Real.sqrt 2 - Real.sqrt 2) / (1 + 2 * (a/b)^2), -(2 * Real.sqrt 2) / (1 + 2 * (a/b)^2)))
  (hF : F = (4 * (a/b) / (1 + 2 * (a/b)^2), (1 - 2 * (a/b)^2) / (1 + 2 * (a/b)^2)))
  (hk : (b/a)^2 = 1 - (b/a)^2) :
  let kEF := (E.snd - F.snd) / (E.fst - F.fst) in
  kEF = Real.sqrt 2 / 2 := 
sorry

end ellipse_slope_l250_250237


namespace joe_money_left_l250_250381

theorem joe_money_left (starting_amount : ℕ) (num_notebooks : ℕ) (cost_per_notebook : ℕ) (num_books : ℕ) (cost_per_book : ℕ)
  (h_starting_amount : starting_amount = 56)
  (h_num_notebooks : num_notebooks = 7)
  (h_cost_per_notebook : cost_per_notebook = 4)
  (h_num_books : num_books = 2)
  (h_cost_per_book : cost_per_book = 7) : 
  starting_amount - (num_notebooks * cost_per_notebook + num_books * cost_per_book) = 14 :=
by
  rw [h_starting_amount, h_num_notebooks, h_cost_per_notebook, h_num_books, h_cost_per_book]
  -- sorry lõpetab ajutiselt
  norm_num  
  -- sorry 

end joe_money_left_l250_250381


namespace min_value_func_min_solution_set_l250_250530

-- Proof Problem (1):
def min_value_func (x : ℝ) (h : 0 < x ∧ x < π / 2) : ℝ :=
  1 / (Real.sin x) ^ 2 + 4 / (Real.cos x) ^ 2

theorem min_value_func_min (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (∃ y, y ∈ set.Ioo (0 : ℝ) (π / 2) ∧ min_value_func y ⟨0, 1⟩ = y) → 9 :=
sorry

-- Proof Problem (2):
variables
  (a b c α β : ℝ)
  (h0 : 0 < α ∧ α < β)
  (h1 : ∀ x, (α < x ∧ x < β) → ax^2 + bx + c > 0)

theorem solution_set (a b c α β : ℝ)
  (h0 : 0 < α ∧ α < β)
  (h1 : ∀ x, (α < x ∧ x < β) → ax^2 + bx + c > 0) :
  ∀ x, (x < 1 / β ∨ 1 / α < x) → cx^2 + bx + a < 0 :=
sorry

end min_value_func_min_solution_set_l250_250530


namespace convex_ngon_no_tessellation_example_of_tessellating_pentagon_l250_250874

-- Part (a): Formalize the proof problem in Lean 4 statement
theorem convex_ngon_no_tessellation (n : ℕ) (h_convex: convex_polygon M)
  (h_ngon: M.is_ngon n) (h_ge7: n ≥ 7) : ¬ tessellates M :=
sorry

-- Part (b): Provide an example in Lean 4 statement 
theorem example_of_tessellating_pentagon (M : polygon) 
  (h_convex: convex M) (h_pentagon: M.is_pentagon) 
  (h_non_parallel_sides: pairwise_non_parallel_sides M) : tessellates M :=
sorry

end convex_ngon_no_tessellation_example_of_tessellating_pentagon_l250_250874


namespace incorrect_statements_l250_250898

-- Define the propositions (conditions)
def StatementA : Prop :=
  ∀ (P : polygon), (∀ s1 s2 : side P, s1.length = s2.length) → regular_polygon P

def StatementB : Prop :=
  ∀ (P : regular_polygon), ∀ s1 s2 : side P, s1.length = s2.length

def StatementC : Prop :=
  ∀ (T : equilateral_triangle), regular_triangle T

def StatementD : Prop :=
  ∀ (P : polygon), (∀ a1 a2 : angle P, a1.measure = a2.measure) → ¬(regular_polygon P)

-- The theorem that we need to prove
theorem incorrect_statements :
  StatementA ∧ StatementB ∧ StatementC ∧ ¬StatementD := 
by
  sorry

end incorrect_statements_l250_250898


namespace ed_vs_combined_marbles_difference_l250_250979

def initial_marbles_doug (marbles_ed : ℕ) : ℕ := marbles_ed - 10
def initial_marbles_frank (marbles_doug : ℕ) : ℕ := marbles_doug + 15
def marbles_after_loss (marbles : ℕ) (loss : ℕ) : ℕ := marbles - loss

def final_combined_marbles_after_losses (marbles_doug : ℕ) (marbles_frank : ℕ) : ℕ :=
  let marbles_doug_loss := marbles_after_loss marbles_doug 11
  let marbles_frank_loss := marbles_after_loss marbles_frank 7
  marbles_doug_loss + marbles_frank_loss

theorem ed_vs_combined_marbles_difference : 
  ∀ (marbles_ed : ℕ), marbles_ed = 45 → 
  ∀ (marbles_doug : ℕ), marbles_doug = initial_marbles_doug marbles_ed → 
  ∀ (marbles_frank : ℕ), marbles_frank = initial_marbles_frank marbles_doug →
  marbles_ed - final_combined_marbles_after_losses marbles_doug marbles_frank = -22 :=
by
  intros marbles_ed h1 marbles_doug h2 marbles_frank h3
  rw [h1, h2, h3]
  unfold initial_marbles_doug initial_marbles_frank final_combined_marbles_after_losses marbles_after_loss
  linarith

end ed_vs_combined_marbles_difference_l250_250979


namespace positive_square_factors_of_180_l250_250289

theorem positive_square_factors_of_180 :
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  factors = 4 :=
by
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  show factors = 4
  sorry

end positive_square_factors_of_180_l250_250289


namespace factorization_correct_l250_250991

theorem factorization_correct (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end factorization_correct_l250_250991


namespace largest_heptagon_angle_l250_250923

-- Definitions made directly from the conditions in the problem
def heptagon_interior_angles (x : ℝ) : list ℝ := [
  x + 2,
  2 * x + 3,
  3 * x - 1,
  4 * x + 4,
  5 * x - 2,
  6 * x + 1,
  7 * x]

def heptagon_angle_sum : ℝ := 900

-- The formal statement of the math proof problem
theorem largest_heptagon_angle (x : ℝ) (h_sum : list.sum (heptagon_interior_angles x) = heptagon_angle_sum) :
  7 * x = 6251 / 28 :=
sorry

end largest_heptagon_angle_l250_250923


namespace tucker_boxes_l250_250485

def tissues_per_box := 160
def used_tissues := 210
def left_tissues := 270

def total_tissues := used_tissues + left_tissues

theorem tucker_boxes : total_tissues = tissues_per_box * 3 :=
by
  sorry

end tucker_boxes_l250_250485


namespace sum_floor_div_eq_floor_of_real_and_int_l250_250840

theorem sum_floor_div_eq_floor_of_real_and_int (x : ℝ) (p : ℕ) (h : 2 ≤ p) :
    (∑ k in Finset.range p, ⌊(x + k) / p⌋) = ⌊x⌋ :=
sorry

end sum_floor_div_eq_floor_of_real_and_int_l250_250840


namespace triangle_sides_angles_l250_250401

open Real

variables {a b c : ℝ} {α β γ : ℝ}

theorem triangle_sides_angles
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (angles_sum : α + β + γ = π)
  (condition : 3 * α + 2 * β = π) :
  a^2 + b * c - c^2 = 0 :=
sorry

end triangle_sides_angles_l250_250401


namespace domain_of_sqrt_function_l250_250046

noncomputable def domain (f : ℝ → ℝ) (dom : Set ℝ) := ∀ (x : ℝ), x ∈ dom → ∃ y : ℝ, f x = y

theorem domain_of_sqrt_function :
  domain (λ x : ℝ, (Real.sqrt (x - 2)) * (Real.sqrt (x + 2))) {x | 2 ≤ x} := 
begin
  sorry
end

end domain_of_sqrt_function_l250_250046


namespace intersection_eq_set_l250_250001

def M : Set ℤ := { x | -4 < (x : Int) ∧ x < 2 }
def N : Set Int := { x | (x : ℝ) ^ 2 < 4 }
def intersection := M ∩ N

theorem intersection_eq_set : intersection = {-1, 0, 1} := 
sorry

end intersection_eq_set_l250_250001


namespace smallest_square_area_l250_250501

theorem smallest_square_area (r : ℝ) (h : r = 5) : ∃ s : ℝ, s^2 = 100 :=
by
  have diameter := 2 * r
  have side_length := diameter
  use side_length
  have area := side_length^2
  have hs : side_length = 10
  rw [(h : r = 5)] at hs
  exact hs
  sorry

end smallest_square_area_l250_250501


namespace angle_bah_eq_angle_oac_om_eq_ah_div_two_angle_o1ho2_eq_80_l250_250905

-- Definitions based on the given conditions
variables {A B C O H: Type*} {triangle : Type*}
variables (circumcenter orthocenter: triangle → O) (is_midpoint : B → C → A → Prop)

-- Given angles
variable (angle_BAC: angle A B C = 60)
variable (angle_ABC: angle A B C = 80)

-- Proof problem statements
theorem angle_bah_eq_angle_oac (BAH OAC: Type*) (H : orthocenter H = orthocenter) (O : circumcenter O = circumcenter) 
(BAH_eq : angle BAH = (90 - angle ABC)) : angle BAH = angle OAC := 
sorry

theorem om_eq_ah_div_two (M : is_midpoint B C M) :  distance O M = (distance A H) / 2 := 
sorry

theorem angle_o1ho2_eq_80 (O1 O2 : Type*) (AC_perp_bisector_OO1 : perp_bisector AC O O1) 
(BC_perp_bisector_OO2 : perp_bisector BC O O2) 
: angle O1 H O2 = 80 :=
sorry

end angle_bah_eq_angle_oac_om_eq_ah_div_two_angle_o1ho2_eq_80_l250_250905


namespace tickets_sold_l250_250940

theorem tickets_sold : 
  ∀ (x y : ℕ), 
    12 * x + 8 * y = 3320 
    ∧ y = x + 140 
    → x + y = 360 :=
by 
  intros x y h,
  cases h with h1 h2,
  sorry

end tickets_sold_l250_250940


namespace range_of_m_l250_250340

theorem range_of_m (m : ℝ) :
  (∀ (x y : ℝ), (x, y) = (0, 0) → |2 * x + y + m| < 3) ∧ 
  (∀ (x y : ℝ), (x, y) = (-1, 1) → |2 * x + y + m| < 3) → 
  -2 < m ∧ m < 3 :=
by
  intro h
  have h1 := h (0 : ℝ) (0 : ℝ) (by refl)
  have h2 := h (-1 : ℝ) (1 : ℝ) (by refl)
  sorry

end range_of_m_l250_250340


namespace ethan_days_worked_per_week_l250_250188

-- Define the conditions
def hourly_wage : ℕ := 18
def hours_per_day : ℕ := 8
def total_earnings : ℕ := 3600
def weeks_worked : ℕ := 5

-- Compute derived values
def daily_earnings : ℕ := hourly_wage * hours_per_day
def weekly_earnings : ℕ := total_earnings / weeks_worked

-- Define the proposition to be proved
theorem ethan_days_worked_per_week : ∃ d: ℕ, d * daily_earnings = weekly_earnings ∧ d = 5 :=
by
  use 5
  simp [daily_earnings, weekly_earnings]
  sorry

end ethan_days_worked_per_week_l250_250188


namespace algorithm_must_have_sequential_structure_l250_250093

-- Definitions and conditions
def sequential_structure (alg : Type) : Prop := 
  -- (Elaboration of what it means for an algorithm to contain a sequential structure)
  sorry

def algorithm (alg : Type) : Prop := 
  -- (Elaboration of what constitutes an algorithm)
  sorry

-- Theorem statement
theorem algorithm_must_have_sequential_structure (alg : Type) [algorithm alg] : sequential_structure alg :=
sorry

end algorithm_must_have_sequential_structure_l250_250093


namespace exists_nonneg_a_l250_250666

noncomputable def lg : ℝ → ℝ := Math.log -- Using the natural log for simplicity

variable {x a : ℝ}

def condition_p (a : ℝ) : Set ℝ := { x | abs (x - 1) > a }
def condition_q : Set ℝ := { x | lg (x^2 - 3 * x + 3) > 0 }
def set_A (a : ℝ) : Set ℝ := { x | x < 1 - a ∨ x > 1 + a }
def set_B : Set ℝ := { x | x < 1 ∨ x > 2 }

theorem exists_nonneg_a (h : ∀ x, condition_p a x → condition_q x) :
  ∃ a : ℝ, 0 ≤ a ∧ (∀ x, condition_p a x → condition_q x) ∧ ¬ (∀ x, condition_q x → condition_p a x) := {
  sorry 
}

end exists_nonneg_a_l250_250666


namespace height_around_145_83_l250_250931

def height_prediction_model (x : ℝ) : ℝ := 7.19 * x + 73.93

theorem height_around_145_83 (x : ℝ) (H : x = 10) :
  height_prediction_model x ≈ 145.83 :=
by {
  rw [height_prediction_model, H],
  norm_num,
}

end height_around_145_83_l250_250931


namespace greatest_possible_fourth_term_l250_250478

theorem greatest_possible_fourth_term {a d : ℕ} (h : 5 * a + 10 * d = 60) : a + 3 * (12 - a) ≤ 34 :=
by 
  sorry

end greatest_possible_fourth_term_l250_250478


namespace first_player_wins_if_picks_1_match_first_l250_250487

theorem first_player_wins_if_picks_1_match_first :
  ∃ n (h1 : n = 100) (h2 : ∀ k, 1 ≤ k ∧ k ≤ 10), 
    (∃ m, m % 11 = 10 ∧ m + k = n) → 1 = 1 :=
by
  sorry

end first_player_wins_if_picks_1_match_first_l250_250487


namespace smallest_number_in_systematic_sampling_l250_250830

theorem smallest_number_in_systematic_sampling (n m total classes: ℕ)
    (h1 : n = 4) 
    (h2 : m = 24)
    (h3 : total = 48)
    (h4 : classes = 4) 
    : let x := (total / m) in 
    (x + (x + (total / n)) + (x + 2 * (total / n)) + (x + 3 * (total / n)) = total) → x = 3 :=
by
  sorry

end smallest_number_in_systematic_sampling_l250_250830


namespace not_monotonic_subinterval_l250_250338

theorem not_monotonic_subinterval (k : ℝ) : 
  (∃ a b : ℝ, a < b ∧ (∀ x ∈ Ioo (k-1) (k+1), deriv (λ x, 2 * x ^ 2 - log x) x < 0 ↔ x < a) 
    ∧ (∀ x ∈ Ioo (k-1) (k+1), deriv (λ x, 2 * x ^ 2 - log x) x > 0 ↔ x > b)) ↔ (1 ≤ k ∧ k < 3 / 2) :=
sorry

end not_monotonic_subinterval_l250_250338


namespace four_digit_integers_divisible_by_11_and_5_l250_250316

theorem four_digit_integers_divisible_by_11_and_5 :
  ∃ count, count = 163 ∧ ∀ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (n % 55 = 0) → ∃ k, 1 ≤ k ∧ k ≤ 163 ∧ n = 1045 + (k - 1) * 55 :=
by
  use 163
  split
  · rfl
  · intros n hn1 hn2 hn3
    have hn4 : 1045 ≤ n ∧ n ≤ 9955 := sorry
    obtain ⟨k, hk1, hk2⟩ : ∃ k, 1 ≤ k ∧ k ≤ 163 ∧ n = 1045 + (k - 1) * 55 := sorry
    exact ⟨k, hk1, hk2⟩

end four_digit_integers_divisible_by_11_and_5_l250_250316


namespace maximum_value_of_sum_l250_250234

open Real

theorem maximum_value_of_sum (n : ℕ) (n_ge_two : 2 ≤ n) 
  (a : Fin n → ℝ) (a_nonneg : ∀ i, 0 ≤ a i) 
  (sum_a : (∑ i, a i) = 1) : 
  (∑ (i : Fin n), ∑ (j : Fin n) in finset.range(n), 
    if i < j then (j.val - i.val) * a i * a j else 0) ≤ (n - 1) / 4 := 
sorry

end maximum_value_of_sum_l250_250234


namespace find_y_l250_250629

theorem find_y (y : ℝ) : log y 64 = log 4 256 + 1 → y = 2^(6/5) :=
by
  sorry

end find_y_l250_250629


namespace candy_bars_to_buy_l250_250134

variable (x : ℕ)

theorem candy_bars_to_buy (h1 : 25 * x + 2 * 75 + 50 = 11 * 25) : x = 3 :=
by
  sorry

end candy_bars_to_buy_l250_250134


namespace perimeter_ABCD_l250_250795

structure Point where
  x : ℝ
  y : ℝ

def dist (P1 P2 : Point) : ℝ :=
  Real.sqrt ((P2.x - P1.x)^2 + (P2.y - P1.y)^2)

def A : Point := { x := 1, y := 0 }
def B : Point := { x := 3, y := 4 }
def C : Point := { x := 6, y := 3 }
def D : Point := { x := 8, y := 1 }

theorem perimeter_ABCD :
  let a := 7
  let b := 2
  a + b = 9 := by
  sorry

end perimeter_ABCD_l250_250795


namespace value_of_expression_when_x_is_2_l250_250893

theorem value_of_expression_when_x_is_2 : 
  (3 * 2 + 4) ^ 2 = 100 := 
by
  sorry

end value_of_expression_when_x_is_2_l250_250893


namespace collinear_vectors_l250_250282

theorem collinear_vectors {n : ℝ} :
  let a := (1, 2)
  let b := (n, 3)
  let c := (4, -1)
  ∃ k : ℝ, b - a = k • c → n = -3 :=
by
  let a := (1 : ℝ, 2 : ℝ)
  let b := (n, 3 : ℝ)
  let c := (4 : ℝ, -1 : ℝ)
  intro h
  cases h with k hk
  rw [sub_eq_zero] at hk
  sorry

end collinear_vectors_l250_250282


namespace find_smallest_n_l250_250011

def smallest_n (k : ℕ) : ℕ :=
  if even k then 100 * (k / 2 + 1) else 100 * (k / 2 + 1) + 1

theorem find_smallest_n (k : ℕ) : ∃ n, 
  ((k = 2 * (k / 2)) → (n = 100 * (k / 2 + 1))) ∧ 
  ((k = 2 * (k / 2) + 1) → (n = 100 * (k / 2 + 1) + 1)) :=
begin
  use smallest_n k,
  split,
  {
    intro h,
    unfold smallest_n,
    rw if_pos,
    exact nat.even_two_mul_div k,
    exact nat.even_iff.mpr h,
  },
  {
    intro h,
    unfold smallest_n,
    rw if_neg,
    exact nat.odd_iff.mpr h,
  }
end

end find_smallest_n_l250_250011


namespace complex_expression_value_l250_250405

noncomputable def x := complex.cos (real.pi / 9) + complex.sin (real.pi / 9) * complex.I

theorem complex_expression_value :
  (2 * x + x^3) * (2 * x^3 + x^9) * (2 * x^6 + x^18) * (2 * x^9 + x^27) * (2 * x^12 + x^36) * (2 * x^15 + x^45) = 549 := 
sorry

end complex_expression_value_l250_250405


namespace sum_of_a_l250_250410

theorem sum_of_a (a_1 a_2 a_3 a_4 : ℚ) 
  (h : {a_1 * a_2, a_1 * a_3, a_1 * a_4, a_2 * a_3, a_2 * a_4, a_3 * a_4} = 
       {-24, -2, -3/2, -1/8, 1, 3}) :
  a_1 + a_2 + a_3 + a_4 = 9/4 ∨ a_1 + a_2 + a_3 + a_4 = -(9/4) :=
sorry

end sum_of_a_l250_250410


namespace alternating_sum_2_to_100_l250_250173

theorem alternating_sum_2_to_100 : 
  ∑ i in Finset.range 50, (if i % 2 = 0 then (2 * (i+1)) else -(2 * (i+1) + 1)) = 51 :=
by
  sorry

end alternating_sum_2_to_100_l250_250173


namespace number_with_7_or_9_in_three_digits_l250_250740

/-- 
The number of three-digit whole numbers with at least one digit being 7 or at least one digit being 9 
--/
theorem number_with_7_or_9_in_three_digits :
  let total_three_digit_numbers := 900 in
  let without_7_or_9_hundreds := 7 in
  let without_7_or_9_tens_ones := 8 in
  let without_7_or_9 := without_7_or_9_hundreds * without_7_or_9_tens_ones * without_7_or_9_tens_ones in
  total_three_digit_numbers - without_7_or_9 = 452 := 
by
  sorry

end number_with_7_or_9_in_three_digits_l250_250740


namespace percentage_palindromes_with_seven_l250_250122

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s == s.reverse

def in_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 2000

def contains_seven (n : ℕ) : Prop :=
  '7' ∈ n.toString.data

def num_palindromes_in_range : ℕ :=
  (Finset.filter (λ n, is_palindrome n) (Finset.filter in_range (Finset.range 2001))).card

def num_palindromes_with_seven : ℕ :=
  (Finset.filter (λ n, contains_seven n) (Finset.filter (λ n, is_palindrome n) (Finset.filter in_range (Finset.range 2001)))).card

theorem percentage_palindromes_with_seven : 
  (num_palindromes_with_seven * 100) / num_palindromes_in_range = 19 := by
  sorry

end percentage_palindromes_with_seven_l250_250122


namespace equation_of_line_with_midpoint_l250_250683

-- Constants and assumptions in the problem
def ellipse (x y : ℝ) := x^2 / 4 + y^2 / 3 = 1
def midpoint (x1 y1 x2 y2 : ℝ) := (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1

-- The main theorem that needs to be proved
theorem equation_of_line_with_midpoint (x1 y1 x2 y2 : ℝ)
    (h1 : ellipse x1 y1)
    (h2 : ellipse x2 y2)
    (h_mid : midpoint x1 y1 x2 y2) :
    ∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = 3 ∧ b = -4 ∧ c = 7 :=
by
  sorry

end equation_of_line_with_midpoint_l250_250683


namespace equilateral_triangle_perimeter_l250_250676

theorem equilateral_triangle_perimeter (p_ADC : ℝ) (h_ratio : ∀ s1 s2 : ℝ, s1 / s2 = 1 / 2) :
  p_ADC = 9 + 3 * Real.sqrt 3 → (3 * (2 * (3 + Real.sqrt 3)) = 18 + 6 * Real.sqrt 3) :=
by
  intro h
  have h1 : 3 * (2 * (3 + Real.sqrt 3)) = 18 + 6 * Real.sqrt 3 := sorry
  exact h1

end equilateral_triangle_perimeter_l250_250676


namespace initial_population_approx_l250_250916

theorem initial_population_approx : 
  ∃ P : ℝ, (0.76 * P ≈ 3553) ∧ P ≈ 4678 :=
by
  use 3553 / 0.76
  split
  · exact rfl
  · exact rfl

end initial_population_approx_l250_250916


namespace choose_elements_sum_eq_binom_l250_250443

open Finset
open Fintype

theorem choose_elements_sum_eq_binom {n : ℕ} 
  (S : Fin n → Finset (Fin (2 * n))) 
  (hS_nonempty : ∀ i, (S i).Nonempty)
  (hS_sum : ∑ i in univ, ∑ x in S i, (x : ℕ) = (2 * n + 1) * n / 2) :
  ∃ f : Fin n → Fin (2 * n), (∀ i, f i ∈ S i) ∧ ∑ i in univ, (f i : ℕ) = (2 * n - 1) * n / 2 :=
  sorry

end choose_elements_sum_eq_binom_l250_250443


namespace num_mappings_l250_250229

open Finset

theorem num_mappings (A B : Finset ℤ) (hA : A = {-1, 0}) (hB : B = {1, 2}) :
  ∃ (f : A → B), A.card * B.card = 4 :=
by
  rw [hA, card_insert_of_not_mem, card_singleton, add_comm, add_one, hB, card_insert_of_not_mem, card_singleton, add_comm, add_one]
  exact ⟨_, rfl⟩

end num_mappings_l250_250229


namespace pow_three_not_sum_of_two_squares_l250_250019

theorem pow_three_not_sum_of_two_squares (k : ℕ) (hk : 0 < k) : 
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = 3^k :=
by
  sorry

end pow_three_not_sum_of_two_squares_l250_250019


namespace cyclic_quadrilateral_BXMY_l250_250015

variable {α : Type*} [EuclideanGeometry α]

theorem cyclic_quadrilateral_BXMY 
  (A B C M P Q X Y : α)
  (hM_midpoint : M = midpoint A C)
  (hP_on_AM : P ∈ segment A M)
  (hQ_on_CM : Q ∈ segment C M)
  (hPQ_eq_half_AC : dist P Q = dist A C / 2)
  (hX_circumcircle_ABQ : X ∈ circumcircle A B Q ∧ X ∈ line B C)
  (hY_circumcircle_BCP : Y ∈ circumcircle B C P ∧ Y ∈ line A B)
  (hX_not_B : X ≠ B)
  (hY_not_B : Y ≠ B) : 
  cyclic B X M Y :=
by
  sorry

end cyclic_quadrilateral_BXMY_l250_250015


namespace compute_sum_series_l250_250606

noncomputable def sum_series : ℚ :=
  ∑' (n : ℕ) in set.Ici 3, ∑' (k : ℕ) in set.Icc 2 (n - 1), (k ^ 2) / (3 ^ (n + k))

theorem compute_sum_series : sum_series = 65609 / 1024 := by
  sorry

end compute_sum_series_l250_250606


namespace integer_modulus_solution_l250_250081

theorem integer_modulus_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ 58294 % 23 = n := 
begin
  use 12,
  split,
  { -- prove 0 <= 12
    exact le_refl 12,
  },
  split,
  { -- prove 12 < 23
    exact lt_of_lt_of_le (of_nat_lt.2 dec_trivial) (nat.cast_le.2 (dec_trivial : 12 ≤ 23)),
  },
  { -- prove 58294 % 23 = 12
    exact int.mod_eq_of_lt (by norm_num) (by norm_num),
  },
end

end integer_modulus_solution_l250_250081


namespace cans_in_case_bulk_warehouse_l250_250538

-- Definitions based on conditions
def price_per_case_bulk_warehouse : ℝ := 12.00
def num_cans_grocery_store : ℝ := 12
def price_for_12_cans_grocery_store : ℝ := 6.00
def extra_cost_per_can_grocery_store : ℝ := 0.25

-- Proof statement
theorem cans_in_case_bulk_warehouse :
  let price_per_can_grocery_store := price_for_12_cans_grocery_store / num_cans_grocery_store in
  let price_per_can_bulk_warehouse := price_per_can_grocery_store - extra_cost_per_can_grocery_store in
  price_per_case_bulk_warehouse / price_per_can_bulk_warehouse = 48 :=
by
  sorry

end cans_in_case_bulk_warehouse_l250_250538


namespace m_n_sum_l250_250332

theorem m_n_sum (m n : ℝ) (h : ∀ x : ℝ, x^2 + m * x + 6 = (x - 2) * (x - n)) : m + n = -2 :=
by
  sorry

end m_n_sum_l250_250332


namespace largest_multiple_of_11_gt_neg_150_l250_250887

-- Largest multiple of 11 whose negation is greater than -150
theorem largest_multiple_of_11_gt_neg_150 : 
  ∃ (m : ℕ), 143 = 11 * m ∧ - (11 * m) > -150 :=
by {
  use 13,
  split,
  { -- Proof of 143 = 11 * 13
    sorry },
  { -- Proof of -143 > -150
    sorry }
}

end largest_multiple_of_11_gt_neg_150_l250_250887


namespace solve_for_x_l250_250031

theorem solve_for_x (x : ℝ) : 3^(2 * x + 1) = 81 → x = 3 / 2 :=
by
  intro h
  -- Proof steps would go here
  sorry

end solve_for_x_l250_250031


namespace evaluate_expression_l250_250623

theorem evaluate_expression (x : ℝ) (hx : 0 < x) : 
  sqrt (x / (1 - (3 * x - 2) / (2 * x))) = sqrt (2 * x^2 / (2 - x)) :=
by
  sorry

end evaluate_expression_l250_250623


namespace count_correct_propositions_l250_250247

-- Definitions of lines and planes
variable (a b : Type) [AffineSpace ℝ a] [AffineSpace ℝ b]
variable (M : Set (Type)) [AffineSpace ℝ M]

-- Definitions of propositions
def prop1 := a ∥ M ∧ b ∥ M → a ∥ b
def prop2 := b ⊆ M ∧ ¬(a ⊆ M) ∧ a ∥ b → a ∥ M
def prop3 := a ⟂ b ∧ b ⊆ M → a ⟂ M
def prop4 := a ⟂ M ∧ a ⟂ b → b ∥ M

-- Theorem to be proven
theorem count_correct_propositions : (if prop1 then 1 else 0) + (if prop2 then 1 else 0) + (if prop3 then 1 else 0) + (if prop4 then 1 else 0) = 1 :=
by sorry

end count_correct_propositions_l250_250247


namespace correct_statements_l250_250024

-- Definitions of the vector operations.
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

 -- Define the vectors given in the conditions.
def e1 : Vector2D := ⟨2, -3⟩
def e2 : Vector2D := ⟨-1, 3 / 2⟩

def a : Vector2D := ⟨-1, 1⟩
def b : Vector2D := ⟨2, 3⟩

-- Function to calculate the dot product of two vectors.
def dot_product (u v : Vector2D) : ℝ :=
  u.x * v.x + u.y * v.y

-- Function to calculate the projection of one vector onto another.
def projection (u v : Vector2D) : Vector2D :=
  let scalar := (dot_product u v) / (u.x * u.x + u.y * u.y)
  ⟨scalar * u.x, scalar * u.y⟩

-- The statement to be proven in Lean.

theorem correct_statements : 
  (determinant (λ e1 e2 : Vector2D, e1.x * e2.y - e2.x * e1.y) = 0 → False) ∧ -- A is incorrect
  (forall G : Vector2D, triangle_has_centroid G → (GA + GB + GC = Vector2D.zero)) ∧ -- B is correct
  (dot_product a b = 0 → (a = Vector2D.zero ∨ b = Vector2D.zero) = False) ∧ -- C is incorrect
  (projection b a = ⟨a.x / 2, a.y / 2⟩) -- D is correct
:= sorry

end correct_statements_l250_250024


namespace second_solution_alcohol_percentage_l250_250535

theorem second_solution_alcohol_percentage (x : ℝ) :
  let s1_volume := 8 -- volume of the first solution in liters
  let s1_concentration := 0.25 -- concentration of the first solution
  let s2_volume := 2 -- volume of the second solution in liters
  let new_solution_volume := s1_volume + s2_volume -- total volume
  let new_solution_concentration := 0.224 -- concentration of the new solution
  let s1_alcohol := s1_volume * s1_concentration -- pure alcohol in the first solution
  let s2_alcohol := s2_volume * (x / 100) -- pure alcohol in the second solution
  let total_alcohol := s1_alcohol + s2_alcohol -- total pure alcohol in the new solution
  new_solution_concentration * new_solution_volume = total_alcohol
  ↔ x = 12 :=
by
  let s1_volume := 8
  let s1_concentration := 0.25
  let s2_volume := 2
  let new_solution_volume := s1_volume + s2_volume
  let new_solution_concentration := 0.224
  let s1_alcohol := s1_volume * s1_concentration
  let s2_alcohol := s2_volume * (x / 100)
  let total_alcohol := s1_alcohol + s2_alcohol
  show new_solution_concentration * new_solution_volume = total_alcohol ↔ x = 12
  sorry

end second_solution_alcohol_percentage_l250_250535


namespace triangle_side_length_l250_250345

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 6) (h2 : c = 4) (h3 : sin (B / 2) = sqrt 3 / 3) : 
  ∃ b : ℝ, b = 6 :=
by
  sorry

end triangle_side_length_l250_250345


namespace number_of_possible_multisets_l250_250451

theorem number_of_possible_multisets :
  ∀ (b0 b1 b2 b3 b4 b5 b6 : ℤ),
  let p := λ x : ℤ, b6 * x^6 + b5 * x^5 + b4 * x^4 + b3 * x^3 + b2 * x^2 + b1 * x + b0,
  let q := λ x : ℤ, b0 * x^6 + b1 * x^5 + b2 * x^4 + b3 * x^3 + b4 * x^2 + b5 * x + b6,
  (∀ s : ℤ, p s = 0 → q s = 0) →
  (∀ s : ℤ, p s = 0 → (s = 1 ∨ s = -1 ∨ s = 2 ∨ s = -2)) →
  (∃ T : Multiset ℤ, (T.card = 6 ∧ (∀ x ∈ T, x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2) ∧ T.ndupcount 1 + T.ndupcount (-1) + T.ndupcount 2 + T.ndupcount (-2) = 6) ∧ T.nodupcount(6) = 78) :=
sorry

end number_of_possible_multisets_l250_250451


namespace find_a5_l250_250230

-- Define the problem conditions within Lean
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions of the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def positive_terms (a : ℕ → ℝ) := ∀ n, 0 < a n
def condition1 (a : ℕ → ℝ) := a 1 * a 3 = 4
def condition2 (a : ℕ → ℝ) := a 7 * a 9 = 25

-- Proposition to prove
theorem find_a5 :
  geometric_sequence a q →
  positive_terms a →
  condition1 a →
  condition2 a →
  a 5 = Real.sqrt 10 :=
by
  sorry

end find_a5_l250_250230


namespace centroid_vector_sum_zero_projection_vector_l250_250021

/-- If point G is the centroid of triangle ABC,
    then the sum of the vectors from G to the vertices of the triangle is zero. -/
theorem centroid_vector_sum_zero
  {A B C G : ℝ × ℝ}
  (hG : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)):
  (G.1 - A.1, G.2 - A.2) + (G.1 - B.1, G.2 - B.2) + (G.1 - C.1, G.2 - C.2) = (0, 0) :=
by
  sorry

/-- The projection of vector b onto vector a is equal to (1/2) * a
    when a = (-1, 1) and b = (2, 3). -/
theorem projection_vector
  (a b : ℝ × ℝ)
  (ha : a = (-1, 1))
  (hb : b = (2, 3)):
  let proj_ab := ((((a.1 * b.1 + a.2 * b.2) / (a.1 * a.1 + a.2 * a.2)) * a.1,
                    ((a.1 * b.1 + a.2 * b.2) / (a.1 * a.1 + a.2 * a.2)) * a.2))
  in proj_ab = ((1 / 2) * a.1, (1 / 2) * a.2) :=
by
  sorry

end centroid_vector_sum_zero_projection_vector_l250_250021


namespace greatest_possible_d_l250_250553

noncomputable def point_2d_units_away_origin (d : ℝ) : Prop :=
  2 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d + 5)^2)

theorem greatest_possible_d : 
  ∃ d : ℝ, point_2d_units_away_origin d ∧ d = (5 + Real.sqrt 244) / 3 :=
sorry

end greatest_possible_d_l250_250553


namespace triangle_midpoints_equilateral_not_imply_original_equilateral_l250_250858

theorem triangle_midpoints_equilateral_not_imply_original_equilateral
  (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (midpoint : A × B → A) (altitude_base1 : A → B) (altitude_base2 : A → C)
  (abc_midpoints_equilateral : midway_point (altitude_base1 A) (altitude_base2 A) = midpoint (altitude_base1 A, altitude_base2 A))
  (equilateral_midpoints : ∀ A B C : Type, equilateral_triangle(midtpoints(A,B,C))) :
  ¬ equilateral_triangle A B C :=
begin
  sorry,
end

end triangle_midpoints_equilateral_not_imply_original_equilateral_l250_250858


namespace polygon_cover_impossible_l250_250430

-- Given conditions setup
variables {P P₁ P₂ : Type} [polygon P] [polygon P₁] [polygon P₂]
variables {O₁ O₂ : Point}
variables {k : ℝ} (h : 0 < k ∧ k < 1)

-- Statement of the problem
theorem polygon_cover_impossible (h₁ : homothety O₁ P k P₁) (h₂ : homothety O₂ P k P₂) :
  ¬ covers P [P₁, P₂] :=
sorry

end polygon_cover_impossible_l250_250430


namespace find_unit_prices_and_rental_schemes_l250_250347

def unit_prices (x : ℕ) (y : ℕ) : Prop :=
  x = 300 ∧ y = 220

def rental_schemes (m : ℕ) (n : ℕ) : Prop :=
  m + n = 30 ∧ m ≥ 10 ∧ m ≤ 12 ∧ 300 * m + 220 * n ≤ 7600

theorem find_unit_prices_and_rental_schemes :
  (∃ (x y : ℕ), unit_prices x y) ∧ (∃ (m n : ℕ), rental_schemes m n) :=
by
  split
  · use 300, 220
    exact And.intro rfl rfl
  · use 10, 20
    split
    · exact Nat.add_comm 20 10
    · split
      · exact Nat.le_refl 10
      · split
        · exact Nat.le_succ 11
        · exact Nat.le_succ_of_le (Nat.le_succ_of_le 11) -- This covers up to 12
        · sorry -- Add detailed reasoning for the total cost calculation
          q₁ -- This implies the possible rental schemes as per the provided answer to check these combinations 10, 11, 12 with 20, 19, 18 for m and n respectively.

end find_unit_prices_and_rental_schemes_l250_250347


namespace probability_of_drawing_red_ball_l250_250771

theorem probability_of_drawing_red_ball (total_balls red_balls white_balls: ℕ) 
    (h1 : total_balls = 5) 
    (h2 : red_balls = 2) 
    (h3 : white_balls = 3) : 
    (red_balls : ℚ) / total_balls = 2 / 5 := 
by 
    sorry

end probability_of_drawing_red_ball_l250_250771


namespace productivity_increase_l250_250872

/-- 
The original workday is 8 hours. 
During the first 6 hours, productivity is at the planned level (1 unit/hour). 
For the next 2 hours, productivity falls by 25% (0.75 units/hour). 
The workday is extended by 1 hour (now 9 hours). 
During the first 6 hours of the extended shift, productivity remains at the planned level (1 unit/hour). 
For the remaining 3 hours of the extended shift, productivity falls by 30% (0.7 units/hour). 
Prove that the overall productivity for the shift increased by 8% as a result of extending the workday.
-/
theorem productivity_increase
  (planned_productivity : ℝ)
  (initial_work_hours : ℝ)
  (initial_productivity_drop : ℝ)
  (extended_work_hours : ℝ)
  (extended_productivity_drop : ℝ)
  (initial_total_work : ℝ)
  (extended_total_work : ℝ)
  (percentage_increase : ℝ) :
  planned_productivity = 1 →
  initial_work_hours = 8 →
  initial_productivity_drop = 0.25 →
  extended_work_hours = 9 →
  extended_productivity_drop = 0.30 →
  initial_total_work = 7.5 →
  extended_total_work = 8.1 →
  percentage_increase = 8 →
  ((extended_total_work - initial_total_work) / initial_total_work * 100) = percentage_increase :=
sorry

end productivity_increase_l250_250872


namespace max_min_f_when_m_eq_1_interval_of_monotonic_increasing_l250_250810

-- Defining the function f(x) with the parameter m
def f (m : ℝ) (x : ℝ) : ℝ := -((1:ℝ)/3) * x^3 + x^2 + (m^2 - 1) * x

-- First problem: proving the maximum and minimum values when m = 1
theorem max_min_f_when_m_eq_1 :
  ∃ (x_max x_min : ℝ), x_max = -3 ∧ x_min = 0 ∧ 
  ∀ x ∈ Set.Icc (-3 : ℝ) 2, f 1 x ≤ 18 ∧ f 1 x ≥ 0 :=
sorry

-- Second problem: proving the interval of monotonic increase
theorem interval_of_monotonic_increasing (m: ℝ) (hm : 0 < m) :
  ∀ (x : ℝ), 1 - m < x ∧ x < m + 1 → 
  ∀ y ∈ {y | f m y ≤ f m (y + 1)}, monotone_on (f m) (Set.Ioo (1 - m) (m + 1)) :=
sorry

end max_min_f_when_m_eq_1_interval_of_monotonic_increasing_l250_250810


namespace maximize_volume_l250_250496

-- Define the problem-specific constants
def bar_length : ℝ := 0.18
def length_to_width_ratio : ℝ := 2

-- Function to define volume of the rectangle frame
def volume (length width height : ℝ) : ℝ := length * width * height

theorem maximize_volume :
  ∃ (length width height : ℝ), 
  (length / width = length_to_width_ratio) ∧ 
  (2 * (length + width) = bar_length) ∧ 
  ((length = 2) ∧ (height = 1.5)) :=
sorry

end maximize_volume_l250_250496


namespace arithmetic_mean_inequality_l250_250409

theorem arithmetic_mean_inequality (n : ℕ) (a : ℕ → ℝ)
  (h : ∀ i, 1 ≤ i → i ≤ n → 0 ≤ a i) :
  (∑ i in Finset.range n, (∑ j in Finset.range (i + 1), a j) ^ 2 / (i + 1) ^ 2) ≤
  4 * (∑ i in Finset.range n, (a i) ^ 2) :=
by sorry

end arithmetic_mean_inequality_l250_250409


namespace correct_statement_is_D_l250_250094

-- Definitions of statements
def statementA : Prop :=
  ∀ (event : Prop) (trials : ℕ), 
  frequency(event, trials) exists independently of the number of trials

def statementB : Prop :=
  ∀ (event : Prop), 
  (probability(event) = 0) → ¬ (event)

def statementC : Prop :=
  ∀ (event : Prop), 
  is_random_before_experiment(probability(event))

def statementD : Prop :=
  ∀ (trials : ℕ) (outcomes : list ℕ), 
  (sum (outcomes) = trials) → (sum (frequencies(outcomes)) = 1)

-- Declaration of the proof problem
theorem correct_statement_is_D : statementD :=
sorry

end correct_statement_is_D_l250_250094


namespace starting_player_wins_l250_250876

-- Define the game setup and conditions
structure GameSetup :=
  (pile1 : ℕ := 100)
  (pile2 : ℕ := 200)
  (pile3 : ℕ := 300)

def is_good_position (piles : (ℕ × ℕ × ℕ)) : Prop :=
  ∃ (n m : ℕ) (a b c : ℕ),
  0 ≤ n ∧ n < m ∧
  piles.1 = 2^n * a ∧ 
  piles.2 = 2^n * b ∧ 
  piles.3 = 2^m * c ∧
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1

-- Define the game rule on making a move
def make_move (piles : (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) :=
  (sorry, sorry, sorry) -- implementation of move rule is omitted

theorem starting_player_wins (setup : GameSetup) : 
  ∃ piles : (ℕ × ℕ × ℕ),
  setup.pile1 = piles.1 ∧ 
  setup.pile2 = piles.2 ∧ 
  setup.pile3 = piles.3 ∧
  is_good_position piles →
  ∀ (moves : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ)),
  ∃ winner : ℕ,
  winner = 1 :=
by
  sorry

end starting_player_wins_l250_250876


namespace volume_fraction_is_correct_l250_250936

noncomputable def volume_fraction_of_cubes (L W H: ℕ) (cube_side: ℕ) : ℚ :=
  let cubes_fit_long_dim := L / cube_side
  let cubes_fit_wide_dim := W / cube_side
  let cubes_fit_high_dim := H / cube_side
  let cube_volume := cube_side ^ 3
  let total_cubes_volume := cubes_fit_long_dim * cubes_fit_wide_dim * cubes_fit_high_dim * cube_volume
  let box_volume := L * W * H
  total_cubes_volume / box_volume

theorem volume_fraction_is_correct :
  volume_fraction_of_cubes 8 7 12 4 = 57.14 := by
  let expected_volume_fraction : ℚ := 4 / 7
  have : 57.14 = (4 / 7 : ℚ) * 100 by sorry
  rw this
  sorry

end volume_fraction_is_correct_l250_250936


namespace operation_correct_l250_250179

def operation (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem operation_correct :
  operation 4 2 = 18 :=
by
  show 2 * 4 + 5 * 2 = 18
  sorry

end operation_correct_l250_250179


namespace sin_square_necessary_but_not_sufficient_l250_250909

theorem sin_square_necessary_but_not_sufficient (α β : ℝ) :
  (sin α + cos β = 0) → (sin α + cos β = 0 ∨ sin α - cos β = 0) →
  sin^2 α + sin^2 β = 1 := 
by
  sorry -- Proof goes here

end sin_square_necessary_but_not_sufficient_l250_250909


namespace hexagon_area_semicircle_l250_250524

theorem hexagon_area_semicircle (d : ℝ) (h : d = 1) :
  let a := (1/√13) in
  let hexagon_area := (3 * Real.sqrt 3) / 26 in
  hexagon_area = (3 * Real.sqrt 3 / 26) := by
sorry

end hexagon_area_semicircle_l250_250524


namespace find_max_z_l250_250758

theorem find_max_z :
  ∃ (x y : ℝ), abs x + abs y ≤ 4 ∧ 2 * x + y ≤ 4 ∧ (2 * x - y) = (20 / 3) :=
by
  sorry

end find_max_z_l250_250758


namespace perfect_square_factors_of_180_l250_250304

theorem perfect_square_factors_of_180 {n : ℕ} (h1 : n = 180) : 
  (∃ f : ℕ → ℕ, (∀ p, (p = 2 ∨ p = 3 ∨ p = 5 → 
    (f p = 0 ∨ (p = 2 ∨ p = 3) ∧ f p = 2 ∧ p ≠ 5))) ∧ 2^(f 2) * 3^(f 3) * 5^(f 5) = n ∧  ((f 2 + f 3 + f 5) % 2 = 0)) → 
  4 :=
begin
  sorry
end

end perfect_square_factors_of_180_l250_250304


namespace point_coordinates_l250_250362

-- Define the complex number represented by 3 + 4i
def z : ℂ := 3 + 4 * complex.I

-- Define the point corresponding to the complex number after division by i
def point_after_division (z : ℂ) : ℂ := z / complex.I

-- State the theorem we want to prove
theorem point_coordinates :
  let coordinate := point_after_division z in
  coordinate = 4 - 3 * complex.I :=
by { sorry }

end point_coordinates_l250_250362


namespace lock_code_difference_l250_250014

theorem lock_code_difference :
  ∃ A B C D, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
             (A = 4 ∧ B = 2 * C ∧ C = D) ∨
             (A = 9 ∧ B = 3 * C ∧ C = D) ∧
             (A * 100 + B * 10 + C - (D * 100 + (2 * D) * 10 + D)) = 541 :=
sorry

end lock_code_difference_l250_250014


namespace monotonically_increasing_function_l250_250265

theorem monotonically_increasing_function (f : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Ioc 1 2, f x = (1 / 2) ^ (x^2 - 2 * m * x)) →
  monotone_on f (Ioc 1 2) →
  m ≥ 2 :=
by
  intros h1 h2
  sorry

end monotonically_increasing_function_l250_250265


namespace cesar_watched_fraction_l250_250932

theorem cesar_watched_fraction
  (total_seasons : ℕ) (episodes_per_season : ℕ) (remaining_episodes : ℕ)
  (h1 : total_seasons = 12)
  (h2 : episodes_per_season = 20)
  (h3 : remaining_episodes = 160) :
  (total_seasons * episodes_per_season - remaining_episodes) / (total_seasons * episodes_per_season) = 1 / 3 := 
sorry

end cesar_watched_fraction_l250_250932


namespace max_value_proof_l250_250253

noncomputable def max_val_x2_plus_y2_minus_xy (a b c : ℝ → ℝ) (λ : ℝ) : ℝ :=
  let x := (c λ) * (a λ)
  let y := (c λ) * (b λ) in
  x^2 + y^2 - x * y

theorem max_value_proof (a b : ℝ → ℝ) (λ : ℝ) (h1 : ∀ λ, (0 < λ ∧ λ < 1))
  (h2 : a ⬝ b = 3) (h3 : measure_theory.angle a b = π / 3) 
  (h4 : ∀ λ, c λ = λ * a λ + (1 - λ) * b λ) :
  max_val_x2_plus_y2_minus_xy a b c = 27 / 8 := 
sorry

end max_value_proof_l250_250253


namespace problem_statement_l250_250274

open Set

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}

theorem problem_statement :
  {y : ℤ | ∃ x ∈ A, y = |x + 1|} = {0, 1, 2, 3} :=
by
  sorry

end problem_statement_l250_250274


namespace hyperbola_eccentricity_l250_250232

-- Definition of a hyperbola with conditions on a and b
def hyperbola (x y a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Lean theorem statement for proving the eccentricity is 2 given the conditions
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : ∀ (x y : ℝ), hyperbola x y a b h₁ h₂) 
  (dist_eq : ∀ (x y : ℝ), x = a → y = 0 → abs (b * x - a * y) / real.sqrt (a ^ 2 + b ^ 2) = b / 2) :
  ∀ e : ℝ, e = sqrt (a^2 + b^2) / a → e = 2 :=
by
  -- Proof omitted
  sorry

end hyperbola_eccentricity_l250_250232


namespace hammer_order_prediction_l250_250392

/-- Given the historic ordering data of three types of hammers over four months, 
    prove that the total number of hammers ordered in the following month (October) 
    with a 7% seasonal increase is 32. -/
theorem hammer_order_prediction :
  let claw_hammers := [3, 4, 6, 9]
  let ball_peen_hammers := [2, 3, 7, 11]
  let sledgehammers := [1, 2, 3, 4]
  let pattern_increase (orders : List ℕ) := List.length orders - 1 * (orders[List.length orders - 1] - orders[List.length orders - 2])
  let next_calculation := λ recent_orders increase => recent_orders.last! + increase
  let october_claw_hammers := next_calculation claw_hammers 4  -- 4 being the expected pattern increase
  let october_ball_peen_hammers := next_calculation ball_peen_hammers 1  -- 1 being the expected pattern increase
  let october_sledgehammers := next_calculation sledgehammers 1  -- 1 being the expected pattern increase
  let total_hammers_before_increase := october_claw_hammers + october_ball_peen_hammers + october_sledgehammers
  let total_with_increase := Float.ceil ((total_hammers_before_increase : Float) * 1.07)
  total_with_increase = 32 := 
sorry

end hammer_order_prediction_l250_250392


namespace not_all_less_than_two_l250_250248

theorem not_all_less_than_two {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a + 1/b < 2 ∧ b + 1/c < 2 ∧ c + 1/a < 2) :=
sorry

end not_all_less_than_two_l250_250248


namespace equality_of_counts_l250_250698

def periodic_function (n m : ℕ) (f : ℤ × ℤ → ℕ) : Prop :=
  ∀ (i j : ℤ), f(i, j) = f(i + n, j) ∧ f(i, j) = f(i, j + m)

def count_a (n m : ℕ) (f : ℤ × ℤ → ℕ) : ℕ :=
  (finset.filter
    (λ p : ℕ × ℕ, f (p.1, p.2) = f (p.1 + 1, p.2) ∧ f (p.1, p.2) = f (p.1, p.2 + 1))
    (finset.product (finset.range n) (finset.range m))).card

def count_b (n m : ℕ) (f : ℤ × ℤ → ℕ) : ℕ :=
  (finset.filter
    (λ p : ℕ × ℕ, f (p.1, p.2) = f (p.1 - 1, p.2) ∧ f (p.1, p.2) = f (p.1, p.2 - 1))
    (finset.product (finset.range n) (finset.range m))).card

theorem equality_of_counts (n m : ℕ) (f : ℤ × ℤ → ℕ) 
  (h_periodic : periodic_function n m f) : count_a n m f = count_b n m f :=
by sorry

end equality_of_counts_l250_250698


namespace percentage_of_palindromes_with_seven_is_10_percent_l250_250125

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def has_seven (n : ℕ) : Prop :=
  ('7' ∈ n.to_string)

def count_eligible_palindromes : ℕ :=
  let candidates := finset.filter (λ n, 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n) (finset.range 2000)
  finset.card candidates

def count_palindromes_with_seven : ℕ :=
  let qualified := finset.filter (λ n, 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n ∧ has_seven n) (finset.range 2000)
  finset.card qualified

theorem percentage_of_palindromes_with_seven_is_10_percent :
  100 * count_palindromes_with_seven / count_eligible_palindromes = 10 :=
by sorry

end percentage_of_palindromes_with_seven_is_10_percent_l250_250125


namespace how_many_integers_satisfy_l250_250709

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end how_many_integers_satisfy_l250_250709


namespace value_of_expression_when_x_is_2_l250_250892

theorem value_of_expression_when_x_is_2 : 
  (3 * 2 + 4) ^ 2 = 100 := 
by
  sorry

end value_of_expression_when_x_is_2_l250_250892


namespace count_integers_satisfy_inequality_l250_250731

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end count_integers_satisfy_inequality_l250_250731


namespace complex_expr_evaluation_l250_250625

def complex_expr : ℤ :=
  2 * (3 * (2 * (3 * (2 * (3 * (2 + 1) * 2) + 2) * 2) + 2) * 2) + 2

theorem complex_expr_evaluation : complex_expr = 5498 := by
  sorry

end complex_expr_evaluation_l250_250625


namespace fraction_book_read_l250_250918

theorem fraction_book_read (read_pages : ℚ) (h : read_pages = 3/7) :
  (1 - read_pages = 4/7) ∧ (read_pages / (1 - read_pages) = 3/4) :=
by
  sorry

end fraction_book_read_l250_250918


namespace find_n_l250_250250

-- Define the conditions
variables (n : ℕ) (p : ℕ) (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ)

-- Conditions
def conditions (pn : ℕ) : Prop :=
  (pn = p * n) ∧
  (n > 0) ∧
  (Nat.Prime p) ∧
  (1 = d1) ∧
  (pn = d8) ∧
  (d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 < d6 ∧ d6 < d7 ∧ d7 < d8) ∧
  (d_17 = d_1 + d_2 + d_3 * (d_3 + d_4 + 13 * p)) -- missing a good way to describe the index condition correctly.

-- Target Statement
theorem find_n (pn : ℕ) (h : conditions pn) : n = 2021 :=
by {
  sorry
}

end find_n_l250_250250


namespace original_weight_of_beef_l250_250519

theorem original_weight_of_beef (w_after : ℝ) (loss_percentage : ℝ) (w_before : ℝ) : 
  (w_after = 550) → (loss_percentage = 0.35) → (w_after = 550) → (w_before = 846.15) :=
by
  intros
  sorry

end original_weight_of_beef_l250_250519


namespace derivative_at_one_l250_250824

theorem derivative_at_one :
  (∀ (x : ℝ), x > 0 → 
    let f := λ y, y^2 - 2 / y + Real.log y
    in (derivative (derivative f) 1) = -5) :=
begin
  intros x hx,
  let f := λ y: ℝ, y^2 - 2 / y + Real.log y,
  have hf_deriv : ∀ x, times_cont_diff ℝ 2 f x := by sorry,
  exact calc
    (derivative (derivative f) 1) = -5 : sorry,
end

end derivative_at_one_l250_250824


namespace count_integers_satisfy_inequality_l250_250728

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end count_integers_satisfy_inequality_l250_250728


namespace correct_conclusions_l250_250647

noncomputable def f1 (x : ℝ) : ℝ := 2^x - 1
noncomputable def f2 (x : ℝ) : ℝ := x^3
noncomputable def f3 (x : ℝ) : ℝ := x
noncomputable def f4 (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2

theorem correct_conclusions :
  ((∀ x, 0 < x ∧ x < 1 → f4 x > f1 x ∧ f4 x > f2 x ∧ f4 x > f3 x) ∧
  (∀ x, x > 1 → f4 x < f1 x ∧ f4 x < f2 x ∧ f4 x < f3 x)) ∧
  (∀ x, ¬(f3 x > f1 x ∧ f3 x > f2 x ∧ f3 x > f4 x) ∧
        ¬(f3 x < f1 x ∧ f3 x < f2 x ∧ f3 x < f4 x)) ∧
  (∃ x, x > 0 ∧ ∀ y, y > x → f1 y > f2 y ∧ f1 y > f3 y ∧ f1 y > f4 y) := by
  sorry

end correct_conclusions_l250_250647


namespace translate_parabola_l250_250866

theorem translate_parabola :
  ∀ x : ℝ, ∃ (dx dy : ℝ), 
  (dx = -2) ∧ 
  (dy = 1) ∧ 
  ((x + dx)^2 + dy = (x + 2)^2 + 1) :=
by
  intros x
  use [-2, 1]
  split
  . rfl
  split
  . rfl
  calc
    ((x + -2)^2 + 1) = (x - 2)^2 + 1 : by rw sub_eq_add_neg
                  ... = (x + 2)^2 + 1 : sorry

end translate_parabola_l250_250866


namespace perfect_square_factors_of_180_l250_250309

theorem perfect_square_factors_of_180 {n : ℕ} (h1 : n = 180) : 
  (∃ f : ℕ → ℕ, (∀ p, (p = 2 ∨ p = 3 ∨ p = 5 → 
    (f p = 0 ∨ (p = 2 ∨ p = 3) ∧ f p = 2 ∧ p ≠ 5))) ∧ 2^(f 2) * 3^(f 3) * 5^(f 5) = n ∧  ((f 2 + f 3 + f 5) % 2 = 0)) → 
  4 :=
begin
  sorry
end

end perfect_square_factors_of_180_l250_250309


namespace problem_statement_l250_250333

open Int

theorem problem_statement (a b : ℤ) (h : (18 ^ a) * 9 ^ (3 * a - 1) = (2 ^ 6) * (3 ^ b)) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : a = 6 := 
by 
  sorry

end problem_statement_l250_250333


namespace number_of_integers_satisfying_ineq_l250_250707

theorem number_of_integers_satisfying_ineq : 
  (finset.card {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset) = 7 := 
sorry

end number_of_integers_satisfying_ineq_l250_250707


namespace perimeter_of_EFGH_l250_250773

theorem perimeter_of_EFGH :
  ∀ (E F G H : Type) [metric_space E] [metric_space F] [metric_space G] [metric_space H]
  (EF : ℝ) (FG : ℝ) (GH : ℝ) (EH : ℝ)
  (angle_E : ∠E = 90) (EG_perp_GH : EG ⊥ GH) (EF24 : EF = 24) (FG10 : FG = 10) (GH26 : GH = 26),
  EF + FG + GH + EH = 112 :=
by
  sorry

end perimeter_of_EFGH_l250_250773


namespace multiple_of_3_b_multiple_of_3_a_minus_b_multiple_of_3_a_minus_c_multiple_of_3_c_minus_b_l250_250846

variable (a b c : ℕ)

-- Define the conditions as hypotheses
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k, n = 3 * k
def is_multiple_of_12 (n : ℕ) : Prop := ∃ k, n = 12 * k
def is_multiple_of_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

-- Hypotheses
axiom ha : is_multiple_of_3 a
axiom hb : is_multiple_of_12 b
axiom hc : is_multiple_of_9 c

-- Statements to be proved
theorem multiple_of_3_b : is_multiple_of_3 b := sorry
theorem multiple_of_3_a_minus_b : is_multiple_of_3 (a - b) := sorry
theorem multiple_of_3_a_minus_c : is_multiple_of_3 (a - c) := sorry
theorem multiple_of_3_c_minus_b : is_multiple_of_3 (c - b) := sorry

end multiple_of_3_b_multiple_of_3_a_minus_b_multiple_of_3_a_minus_c_multiple_of_3_c_minus_b_l250_250846


namespace new_median_after_adding_nine_l250_250921

-- Let's define the conditions
variables {a b c d e f : ℕ}

-- Here we define the main hypothesis and the conclusion
theorem new_median_after_adding_nine 
  (hmean : (a + b + c + d + e + f) / 6 = 4.5)
  (hmedian : c = 5 ∧ d = 5) 
  (hmode : (∀ x ∈ ({a, b, c, d, e, f} : multiset ℕ), x = 4 → ({a, b, c, d, e, f} : multiset ℕ).count 4 > ({a, b, c, d, e, f} : multiset ℕ).count x)) :
  multiset.median ({a, b, c, d, e, f, 9} : multiset ℕ) = 5.0 :=
by
  sorry

end new_median_after_adding_nine_l250_250921


namespace find_k_l250_250115

-- Define the line passing through a and b
variables {V : Type*} [AddCommGroup V] [Module ℝ V] {a b : V}

-- Assume a and b are distinct
variables (h : a ≠ b)

-- Define the parameterized line through a and b
def line_param (t : ℝ) : V := a + t • (b - a)

-- The vector on the line we are interested in
def target_vector (k : ℝ) : V := k • a + (5 / 6) • b

-- The theorem to prove
theorem find_k (k : ℝ) : target_vector k = line_param (5 / 6) → k = 5 / 6 :=
by
  sorry

end find_k_l250_250115


namespace how_many_integers_satisfy_l250_250708

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end how_many_integers_satisfy_l250_250708


namespace barbara_typing_time_l250_250589

theorem barbara_typing_time :
  let original_speed := 212
  let reduction := 40
  let num_words := 3440
  let reduced_speed := original_speed - reduction
  let time := num_words / reduced_speed
  time = 20 :=
by
  sorry

end barbara_typing_time_l250_250589


namespace max_abs_sum_x_y_l250_250326

theorem max_abs_sum_x_y (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * real.sqrt 2 :=
sorry

end max_abs_sum_x_y_l250_250326


namespace fifteenth_prime_is_47_l250_250983

theorem fifteenth_prime_is_47 (h : (Nat.nth_prime 5) = 11) : (Nat.nth_prime 15) = 47 := sorry

end fifteenth_prime_is_47_l250_250983


namespace max_min_values_monotonic_increasing_interval_l250_250809

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  - (1 / 3) * x^3 + x^2 + (m^2 - 1) * x

theorem max_min_values (m : ℝ) (h : m = 1) :
  (∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m ≤ 18) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m = 18) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m ≥ 0) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m = 0) := sorry

theorem monotonic_increasing_interval (m : ℝ) (h : m > 0) :
  ∀ x ∈ Set.Ioo (1 - m : ℝ) (m + 1 : ℝ), 0 < deriv (λ x, f x m) x := sorry

end max_min_values_monotonic_increasing_interval_l250_250809


namespace tan_four_fifths_alpha_l250_250245

theorem tan_four_fifths_alpha 
  (α : ℝ) 
  (hα1 : 0 < α) 
  (hα2 : α < π / 2) 
  (h_eq : 2 * sqrt 3 * (cos α) ^ 2 - sin (2 * α) + 2 - sqrt 3 = 0) : 
  tan (4 / 5 * α) = sqrt 3 := 
sorry

end tan_four_fifths_alpha_l250_250245


namespace general_pattern_specific_case_l250_250529

-- Define the general form of the pattern
theorem general_pattern (a : ℕ) (n : ℕ) :
  (a - 1) * (∑ i in finset.range (n + 1), a ^ (n - i)) = a ^ (n + 1) - 1 :=
sorry

-- Calculate the specific case
theorem specific_case :
  (4 - 1) * (∑ i in finset.range 2013, 4 ^ (2012 - i)) = 4 ^ 2013 - 1 :=
sorry

end general_pattern_specific_case_l250_250529


namespace sine_function_strictly_increasing_l250_250269

-- Given initial conditions and definitions
def theta : ℝ := π / 3
def f (x : ℝ) : ℝ := Real.sin (π * x + theta)
def is_strictly_increasing_interval (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Problem statement
theorem sine_function_strictly_increasing (k : ℤ) : 
  is_strictly_increasing_interval (2 * k - 5 / 6) (2 * k + 1 / 6) := 
sorry

end sine_function_strictly_increasing_l250_250269


namespace residue_at_zero_l250_250216

noncomputable def f (z : ℂ) : ℂ := Complex.exp (1 / z^2) * Complex.cos z

theorem residue_at_zero : Complex.residue f 0 = 0 := sorry

end residue_at_zero_l250_250216


namespace compare_numbers_l250_250059

theorem compare_numbers : 7^0.3 > 0.3^7 ∧ 0.3^7 > log 3 0.7 := 
by 
  sorry

end compare_numbers_l250_250059


namespace prime_arithmetic_sequence_l250_250018

theorem prime_arithmetic_sequence {p1 p2 p3 d : ℕ} 
  (hp1 : Nat.Prime p1) 
  (hp2 : Nat.Prime p2) 
  (hp3 : Nat.Prime p3)
  (h3_p1 : 3 < p1)
  (h3_p2 : 3 < p2)
  (h3_p3 : 3 < p3)
  (h_seq1 : p2 = p1 + d)
  (h_seq2 : p3 = p1 + 2 * d) : 
  d % 6 = 0 :=
by sorry

end prime_arithmetic_sequence_l250_250018


namespace paint_cube_faces_l250_250155

noncomputable def paintCubeDistinctWays : ℕ :=
  let colors := { "blue", "green", "red" }
  let num_faces := 6
  let pairs := num_faces / 2
  -- Here we encode the question of counting distinct colorings considering rotations.
  6 -- This should correspond to the rigorous mathematical proof in actual practice.
  
theorem paint_cube_faces :
  ∃ (paintCubeDistinctWays : ℕ), 
    paintCubeDistinctWays = 6 := 
by 
  use 6
  -- The rigorous proof of this statement would be fleshed out in practice.
  sorry

end paint_cube_faces_l250_250155


namespace number_of_integers_satisfying_ineq_l250_250704

theorem number_of_integers_satisfying_ineq : 
  (finset.card {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset) = 7 := 
sorry

end number_of_integers_satisfying_ineq_l250_250704


namespace solve_star_op_eq_l250_250614

def star_op (a b : ℕ) : ℕ :=
  if a < b then b * b else b * b * b

theorem solve_star_op_eq :
  ∃ x : ℕ, 5 * star_op 5 x = 64 ∧ (x = 4 ∨ x = 8) :=
sorry

end solve_star_op_eq_l250_250614


namespace chef_used_apples_l250_250541

theorem chef_used_apples (initial_apples remaining_apples used_apples : ℕ) 
  (h1 : initial_apples = 40) 
  (h2 : remaining_apples = 39) 
  (h3 : used_apples = initial_apples - remaining_apples) : 
  used_apples = 1 := 
  sorry

end chef_used_apples_l250_250541


namespace systematic_sampling_interval_papers_l250_250945

-- Define the mathematical problem and prove the statement
theorem systematic_sampling_interval_papers (n m : ℕ) (start end_ : ℕ) :
  n = 1000 → m = 50 → start = 850 → end_ = 949 →
  let interval := end_ - start + 1 in
  ∀ k, k = n / m →
  interval / k = 5 :=
by
  intros
  sorry

end systematic_sampling_interval_papers_l250_250945


namespace necklace_painting_probability_l250_250516

def probability_all_beads_painted 
    (n k : ℕ) 
    (necklace : Fin n → Bool) 
    (paint_bead : Fin n → Fin n → Bool)
    (prob : ℚ) 
    : Prop :=
    ∀ (fin_n : Fin n), 
    necklace fin_n = false ∧
    ((Fintype.card { s : Finset (Fin n) // s.card = k }) = 252) →
    (1 - (150 / 252) = prob) → 
    prob = 17 / 42

theorem necklace_painting_probability : 
    probability_all_beads_painted 10 5 (λ _, false) (λ i j, true) 17/42 :=
sorry

end necklace_painting_probability_l250_250516


namespace linear_equation_check_l250_250509

theorem linear_equation_check : 
  (∃ a b : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x + b = 1)) ∧ 
  ¬ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, a * x + b * y = 3)) ∧ 
  ¬ (∀ x : ℝ, x^2 - 2 * x = 0) ∧ 
  ¬ (∀ x : ℝ, x - 1 / x = 0) := 
sorry

end linear_equation_check_l250_250509


namespace other_x_intercept_l250_250144

def foci1 := (0, -3)
def foci2 := (4, 0)
def x_intercept1 := (0, 0)

theorem other_x_intercept :
  (∃ x : ℝ, (|x - 4| + |-3| * x = 7)) → x = 11 / 4 := by
  sorry

end other_x_intercept_l250_250144


namespace percentile_80_example_l250_250061

variable (D : List ℝ) (P : ℝ)

def isInOrder (l : List ℝ) : Prop := 
  ∀ (i j : ℕ), i < j → i < l.length → j < l.length → l.nthLe i (by linarith) ≤ l.nthLe j (by linarith)

def find80thPercentile (l : List ℝ) : ℝ :=
  let sorted_l := l.qsort (≤)
  let position := Nat.ceil (0.8 * sorted_l.length)
  sorted_l.nthLe (position - 1) (by sorry) -- placeholder for proof that position - 1 is within bounds

theorem percentile_80_example :
  let D := [110.2, 109.7, 110.8, 109.1, 108.9, 108.6, 109.8, 109.6, 109.9, 111.2, 110.6, 111.7]
  isInOrder D.qsort (≤) →
  find80thPercentile D = 110.8 :=
by
  intros
  sorry -- proof steps

end percentile_80_example_l250_250061


namespace irreducible_polynomial_l250_250659

def f (x : ℤ) (a : List ℤ) : ℤ := (a.foldl (λ acc ai → acc * (x - ai)) 1) - 1

theorem irreducible_polynomial (a : List ℤ) (h : a.Nodup) (h_len : a.length = n) :
  Irreducible (f x a) :=
sorry

end irreducible_polynomial_l250_250659


namespace art_museum_survey_l250_250521

theorem art_museum_survey (V E : ℕ) 
  (h1 : ∀ (x : ℕ), x = 140 → ¬ (x ≤ E))
  (h2 : E = (3 / 4) * V)
  (h3 : V = E + 140) :
  V = 560 := by
  sorry

end art_museum_survey_l250_250521


namespace regular_tetrahedron_distance_relation_l250_250784

theorem regular_tetrahedron_distance_relation
  (T : Type*)
  [metric_space T]
  (equilateral_triangle_distance_relation : ∀ (V₁ V₂ V₃ : T) (center : T),
      (equilateral_triangle V₁ V₂ V₃ → 
       dist center V₁ = 2 * dist center (midpoint V₂ V₃))) :
  ∀ (V₁ V₂ V₃ V₄ : T) (center : T), 
    (regular_tetrahedron V₁ V₂ V₃ V₄ → 
     dist center V₁ = 3 * dist center (centroid V₂ V₃ V₄)) :=
begin
  sorry
end

end regular_tetrahedron_distance_relation_l250_250784


namespace polygon_interior_angle_sum_l250_250066

theorem polygon_interior_angle_sum (n : ℕ) (h : (n - 2) * 180 = 1800) : n = 12 :=
by sorry

end polygon_interior_angle_sum_l250_250066


namespace max_value_fraction_l250_250673

theorem max_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∀ z, z = (x / (2 * x + y) + y / (x + 2 * y)) → z ≤ (2 / 3) :=
by
  sorry

end max_value_fraction_l250_250673


namespace volume_of_region_l250_250642

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y - z| 

theorem volume_of_region : let region := {p : ℝ × ℝ × ℝ | let (x, y, z) := p in f x y z ≤ 6} in
  let volume := 22.5 in
  sorry

end volume_of_region_l250_250642


namespace binary_representation_l250_250060

theorem binary_representation (n : ℕ) (h1 : n % 17 = 0) (h2 : (nat.popcount n = 3)) :
  let zcount := nat.bits n - 3 in
  (6 ≤ zcount) ∧ (zcount = 7 → even n) :=
by
  let zcount := nat.bits n - 3
  have zero_count_six_or_more : 6 ≤ zcount := sorry
  have zero_count_seven_even : zcount = 7 → even n := sorry
  exact ⟨zero_count_six_or_more, zero_count_seven_even⟩

end binary_representation_l250_250060


namespace find_k_l250_250499

theorem find_k : ∃ k : ℤ, 0 ≤ k ∧ k < 17 ∧ (-175 ≡ k [MOD 17]) ∧ k = 12 :=
by 
  use 12
  sorry

end find_k_l250_250499


namespace melted_mixture_weight_l250_250518

theorem melted_mixture_weight (Z C : ℝ) (ratio : 9 / 11 = Z / C) (zinc_weight : Z = 28.8) : Z + C = 64 :=
by
  sorry

end melted_mixture_weight_l250_250518


namespace sin_square_necessary_but_not_sufficient_l250_250910

theorem sin_square_necessary_but_not_sufficient (α β : ℝ) :
  (sin α + cos β = 0) → (sin α + cos β = 0 ∨ sin α - cos β = 0) →
  sin^2 α + sin^2 β = 1 := 
by
  sorry -- Proof goes here

end sin_square_necessary_but_not_sufficient_l250_250910


namespace Nina_second_distance_l250_250829

theorem Nina_second_distance 
  (total_distance : ℝ) 
  (first_run : ℝ) 
  (second_same_run : ℝ)
  (run_twice : first_run = 0.08 ∧ second_same_run = 0.08)
  (total : total_distance = 0.83)
  : (total_distance - (first_run + second_same_run)) = 0.67 := by
  sorry

end Nina_second_distance_l250_250829


namespace approximate_number_of_fish_in_pond_l250_250763

theorem approximate_number_of_fish_in_pond :
  (∃ N : ℕ, 
  (∃ tagged1 tagged2 : ℕ, tagged1 = 50 ∧ tagged2 = 10) ∧
  (∃ caught1 caught2 : ℕ, caught1 = 50 ∧ caught2 = 50) ∧
  ((tagged2 : ℝ) / caught2 = (tagged1 : ℝ) / (N : ℝ)) ∧
  N = 250) :=
sorry

end approximate_number_of_fish_in_pond_l250_250763


namespace find_sheets_used_l250_250438

variable (x y : ℕ) -- define variables for x and y
variable (h₁ : 82 - x = y) -- 82 - x = number of sheets left
variable (h₂ : y = x - 6) -- number of sheets left = number of sheets used - 6

theorem find_sheets_used (h₁ : 82 - x = x - 6) : x = 44 := 
by
  sorry

end find_sheets_used_l250_250438


namespace floor_5_7_l250_250196

theorem floor_5_7 : Int.floor 5.7 = 5 :=
by
  sorry

end floor_5_7_l250_250196


namespace positive_square_factors_count_l250_250739

theorem positive_square_factors_count (a b c : ℕ) (h1 : a = 12) (h2 : b = 10) (h3 : c = 9) :
  ∃ n : ℕ, n = 210 ∧
    (∀ k ∈ {2^i | i ∈ {0,2,4,6,8,10,12}}, k ∣ 2^a ∧ ∀ j ∈ {7^i | i ∈ {0,2,4,6,8,10}}, j ∣ 7^b ∧
    ∀ l ∈ {11^i | i ∈ {0,2,4,6,8}}, l ∣ 11^c ) :=
sorry

end positive_square_factors_count_l250_250739


namespace measure_of_angle_Q_l250_250882

theorem measure_of_angle_Q (y : ℝ)
  (h1 : ∃ P Q R : Point, inscribed_circle_triangle P Q R)
  (h2 : arc_measures P Q R = [y + 60, 2y + 40, 3y - 10])
  (h3 : (y + 60) + (2y + 40) + (3y - 10) = 360) :
  interior_angle Q = 62.5 := by
  -- Proof omitted
  sorry

end measure_of_angle_Q_l250_250882


namespace f_increasing_on_neg1_0_l250_250508

noncomputable def f (x : ℝ) := x * Real.exp (-x)

theorem f_increasing_on_neg1_0 : ∀ x y, -1 ≤ x ∧ x ≤ 0 ∧ x ≤ y ∧ y ≤ 0 → f(x) ≤ f(y) := by
  intro x y h
  let ⟨hx1, hx2, hxy1, hxy2⟩ := h
  sorry

end f_increasing_on_neg1_0_l250_250508


namespace sixth_term_sequence_l250_250620

theorem sixth_term_sequence (a : ℕ → ℕ) (h₁ : a 0 = 3) (h₂ : ∀ n, a (n + 1) = (a n)^2) : 
  a 5 = 1853020188851841 := 
by {
  sorry
}

end sixth_term_sequence_l250_250620


namespace double_sum_eq_fraction_l250_250609

theorem double_sum_eq_fraction :
  (∀ (n : ℕ), n ≥ 3 → (∀ (k : ℕ), k ≥ 2 ∧ k < n → 
  (∑ n in filter (λ n, n ≥ 3) (range (n+1)), 
  ∑ k in filter (λ k, k ≥ 2 ∧ k < n) (range (n)) (k^2 / 3^(n+k))) = 9 / 128)) :=
by
  exact sorry

end double_sum_eq_fraction_l250_250609


namespace ABCD_Expression_Mod_12_Zero_l250_250875

theorem ABCD_Expression_Mod_12_Zero (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) 
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : a ∈ {1, 5, 7, 11}) (h8 : b ∈ {1, 5, 7, 11}) 
  (h9 : c ∈ {1, 5, 7, 11}) (h10 : d ∈ {1, 5, 7, 11}) :
  ((a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d) * (a * b * c * d)⁻¹) % 12 = 0 := 
by
  sorry

end ABCD_Expression_Mod_12_Zero_l250_250875


namespace angle_alpha_second_or_third_l250_250257

noncomputable def alpha_quadrant (m : ℝ) (h : m ≠ 0) : string :=
  if (-√3, m) == (P : ℝ × ℝ) ∧ sin (angle P) = (√3 / 4) * m then
    if m > 0 then "second" else "third"
  else
    "undefined"

theorem angle_alpha_second_or_third (m : ℝ) (h : m ≠ 0) (P : ℝ × ℝ) (angle : (ℝ × ℝ) → ℝ) :
  P = (-√3, m) → sin (angle P) = (√3 / 4) * m → (alpha_quadrant m h = "second" ∨ alpha_quadrant m h = "third") :=
by
  assume hP : P = (-√3, m)
  assume hsin : sin (angle P) = (√3 / 4) * m
  sorry

end angle_alpha_second_or_third_l250_250257


namespace factorize_xcube_minus_x_l250_250992

theorem factorize_xcube_minus_x (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by 
  sorry

end factorize_xcube_minus_x_l250_250992


namespace count_integers_satisfy_inequality_l250_250729

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end count_integers_satisfy_inequality_l250_250729


namespace club_truncator_more_wins_than_losses_l250_250965

def probabilities := (2/5: ℝ, 1/5: ℝ, 2/5: ℝ)

def total_matches := 7

def total_outcomes := 3^total_matches

def possible_same_wins_losses := 140 + 210 + 42 + 1

def probability_same_wins_losses := possible_same_wins_losses / total_outcomes

def probability_not_same_wins_losses := 1 - probability_same_wins_losses

def probability_more_wins_than_losses := (probability_not_same_wins_losses / 2 : ℝ)

/- This theorem captures the equivalent proof problem -/
theorem club_truncator_more_wins_than_losses :
  probability_more_wins_than_losses = 897 / 2187 := 
by
  sorry

end club_truncator_more_wins_than_losses_l250_250965


namespace monotonic_interval_of_f_l250_250468

def f (x : ℝ) : ℝ := -6 / x - 5 * Real.log x

theorem monotonic_interval_of_f :
  ∃ a b : ℝ, (0 < a) ∧ (b = 6 / 5) ∧ (∀ x : ℝ, a < x ∧ x < b → deriv f x > 0) :=
by {
  sorry
}

end monotonic_interval_of_f_l250_250468


namespace concyclic_points_in_triangle_l250_250358

theorem concyclic_points_in_triangle (A B C H D E F : Point) 
  (hABC : acute_triangle A B C)
  (h_s1 : semicircle_diameter A B)
  (h_s2 : semicircle_diameter A C)
  (h_AH_perp_BC : perpendicular A H B C)
  (h_intersect_H : line_segment_intersect A H B C H)
  (h_D_on_BC : point_on_line D B C)
  (h_not_endpoint_D : D ≠ B ∧ D ≠ C)
  (h_DE_parallel_AC : parallel_line_segment D E A C)
  (h_DF_parallel_AB : parallel_line_segment D F A B)
  (h_E_on_semicircle_AC : point_on_semicircle E A C)
  (h_F_on_semicircle_AB : point_on_semicircle F A B) :
    concyclic_points D E F H :=
  sorry

end concyclic_points_in_triangle_l250_250358


namespace find_x_value_l250_250904

-- Definitions based on the conditions
def varies_inversely_as_square (k : ℝ) (x y : ℝ) : Prop := x = k / y^2

def given_condition (k : ℝ) : Prop := 1 = k / 3^2

-- The main proof problem to solve
theorem find_x_value (k : ℝ) (y : ℝ) (h1 : varies_inversely_as_square k 1 3) (h2 : y = 9) : 
  varies_inversely_as_square k (1/9) y :=
sorry

end find_x_value_l250_250904


namespace divisors_square_less_than_four_n_l250_250816

theorem divisors_square_less_than_four_n (n : ℕ) (k : ℕ) (h : k = ∑ d in (Finset.range (n + 1)).filter (n % · = 0), 1): 
  k^2 < 4 * n := 
by 
  sorry

end divisors_square_less_than_four_n_l250_250816


namespace james_coffee_weekdays_l250_250786

theorem james_coffee_weekdays :
  ∃ (c d : ℕ) (k : ℤ), (c + d = 5) ∧ 
                      (3 * c + 2 * d + 10 = k / 3) ∧ 
                      (k % 3 = 0) ∧ 
                      c = 2 :=
by 
  sorry

end james_coffee_weekdays_l250_250786


namespace tangent_slope_at_A_l250_250064

open Real

theorem tangent_slope_at_A (x y : ℝ) (h : y = exp (-x)) : 
  deriv (λ x, exp (-x)) 0 = -1 := by
sorry

end tangent_slope_at_A_l250_250064


namespace subtraction_of_decimals_l250_250162

theorem subtraction_of_decimals : (3.75 - 0.48) = 3.27 :=
by
  sorry

end subtraction_of_decimals_l250_250162


namespace barbara_typing_time_l250_250590

theorem barbara_typing_time :
  let original_speed := 212
  let reduction := 40
  let num_words := 3440
  let reduced_speed := original_speed - reduction
  let time := num_words / reduced_speed
  time = 20 :=
by
  sorry

end barbara_typing_time_l250_250590


namespace electricity_average_l250_250547

-- Define the daily electricity consumptions
def electricity_consumptions : List ℕ := [110, 101, 121, 119, 114]

-- Define the function to calculate the average
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Formalize the proof problem
theorem electricity_average :
  average electricity_consumptions = 113 :=
  sorry

end electricity_average_l250_250547


namespace roller_coaster_ticket_cost_l250_250869

theorem roller_coaster_ticket_cost (total_tickets : ℕ) (rides : ℕ) (cost_per_ride : ℕ) 
  (h1 : total_tickets = 10) (h2 : rides = 2) : cost_per_ride = total_tickets / rides :=
by 
  dsimp only [cost_per_ride, total_tickets, rides] at *
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul rfl

#print roller_coaster_ticket_cost -- confirming the structure

end roller_coaster_ticket_cost_l250_250869


namespace ellipse_equation_and_lambda_range_l250_250950

-- Define the conditions for the ellipse (C)
def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def center_at_origin (C : set (ℝ × ℝ)) : Prop :=
  ∀ p ∈ C, p.1 = 0 ∧ p.2 = 0

def foci_on_x_axis (C : set (ℝ × ℝ)) : Prop :=
  ∀ p ∈ C, (∃ e : ℝ, e > 0 ∧ p = (e, 0) ∨ p = (-e, 0))

def focal_length (C : set (ℝ × ℝ)) (f : ℝ) : Prop :=
  ∃ c : ℝ, 2 * c = f

def same_eccentricity (C1 C2 : set (ℝ × ℝ)) : Prop :=
  ∃ e : ℝ, ∀ p ∈ C1, ∃ q ∈ C2, (p.1^2 + p.2^2) * e = (q.1^2 + q.2^2)

def line (k m : ℝ) : set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + m}

def intersects_at_two_points (l C : set (ℝ × ℝ)) : Prop :=
  finset.card (l ∩ C) = 2

def vector_eq (A B Q O : ℝ × ℝ) (λ : ℝ) : Prop :=
  (A.1 + B.1, A.2 + B.2) = (λ * Q.1, λ * Q.2)

-- The main statement for the problem
theorem ellipse_equation_and_lambda_range :
  ∀ (a b : ℝ) (C : set (ℝ × ℝ)), 
    ellipse a b = C ∧ a = sqrt 2 ∧ b = 1 ∧ center_at_origin C ∧ foci_on_x_axis C ∧ focal_length C 2 ∧
    same_eccentricity C ({p | p.1^2 + p.2^2 / 2 = 1}) →
    C = {p | p.1^2 / 2 + p.2^2 = 1} ∧ 
    (∀ (k m : ℝ) (l : set (ℝ × ℝ)), line k m = l ∧ intersects_at_two_points l C →
      ∀ (A B Q O : ℝ × ℝ) (λ : ℝ), vector_eq A B Q O λ →
        -2 < λ ∧ λ < 2 ∧ λ ≠ 0) :=
begin
  sorry
end

end ellipse_equation_and_lambda_range_l250_250950


namespace floor_5_7_eq_5_l250_250194

theorem floor_5_7_eq_5 : Int.floor 5.7 = 5 := by
  sorry

end floor_5_7_eq_5_l250_250194


namespace concyclic_if_and_only_if_equal_angles_l250_250342

-- Basic definitions: points, triangles, midpoints, reflections, intersections, angles, concyclic
variable {A B C M P Q D E : Point}

-- Conditions
def is_midpoint (M : Point) (A B : Point) : Prop :=
  M = midpoint A B

def is_reflection (Q P M : Point) : Prop :=
  reflect Q P M

def is_intersection (D P B C : Point) : Prop :=
  D = intersection (line P B) (line B C)

def is_intersection2 (E P A C : Point) : Prop :=
  E = intersection (line P A) (line A C)

def is_concyclic (A B D E : Point) : Prop := 
  concyclic {A, B, D, E}

def angle_equal (A C P Q B : Point) : Prop := 
  ∠ A C P = ∠ Q C B

-- The theorem statement
theorem concyclic_if_and_only_if_equal_angles 
  (h_midpoint : is_midpoint M A B)
  (h_reflection : is_reflection Q P M)
  (h_intersection1 : is_intersection D P B C)
  (h_intersection2 : is_intersection2 E P A C):
  (is_concyclic A B D E ↔ angle_equal A C P Q B) :=
begin
  sorry
end

end concyclic_if_and_only_if_equal_angles_l250_250342


namespace fraction_traveled_by_foot_l250_250137

theorem fraction_traveled_by_foot :
  ∀ D : ℕ, D = 90 →
  let distance_by_bus := (2 / 3 : ℝ) * D
  let distance_by_car := 12
  (D - (distance_by_bus + distance_by_car)) = 18 →
  (D - (distance_by_bus + distance_by_car)) / D = 1 / 5 := 
begin
  intros D hD,
  rw ←hD,
  let distance_by_bus := (2 / 3 : ℝ) * 90,
  let distance_by_car := 12,
  sorry
end

end fraction_traveled_by_foot_l250_250137


namespace onions_remaining_l250_250073

theorem onions_remaining (initial_onions sold_onions remaining_onions : ℕ) (h_initial : initial_onions = 98) (h_sold : sold_onions = 65) : remaining_onions = 33 := 
by
  have h : remaining_onions = initial_onions - sold_onions,
  { sorry },
  have h_val : remaining_onions = 98 - 65,
  { rw [h_initial, h_sold, h], },
  have h_eq : 98 - 65 = 33,
  { sorry },
  rw [h_val, h_eq],
  sorry

end onions_remaining_l250_250073


namespace train_pass_bridge_time_l250_250567

noncomputable def totalDistance (trainLength bridgeLength : ℕ) : ℕ :=
  trainLength + bridgeLength

noncomputable def speedInMPerSecond (speedInKmPerHour : ℕ) : ℝ :=
  (speedInKmPerHour * 1000) / 3600

noncomputable def timeToPass (totalDistance : ℕ) (speedInMPerSecond : ℝ) : ℝ :=
  totalDistance / speedInMPerSecond

theorem train_pass_bridge_time
  (trainLength : ℕ) (bridgeLength : ℕ) (speedInKmPerHour : ℕ)
  (h_train : trainLength = 300)
  (h_bridge : bridgeLength = 115)
  (h_speed : speedInKmPerHour = 35) :
  timeToPass (totalDistance trainLength bridgeLength) (speedInMPerSecond speedInKmPerHour) = 42.7 :=
by
  sorry

end train_pass_bridge_time_l250_250567


namespace tangent_line_passes_through_homothety_center_tangent_line_passes_through_homothety_center_equal_radii_l250_250450

theorem tangent_line_passes_through_homothety_center
  (O1 O2 O3 T1 T2 : Point)
  (tangent1 : Tangent T1 O1 O3)
  (tangent2 : Tangent T2 O2 O3)
  (homothety_center_exists : HomothetyCenter O1 O2) :
  (Line_through T1 T2) ∧ (Passes_through_homothety_center (Line_through T1 T2) homothety_center_exists) :=
sorry

-- Special case when the radii of O1 and O2 are equal
theorem tangent_line_passes_through_homothety_center_equal_radii
  (O1 O2 O3 T1 T2 : Point)
  (tangent1 : Tangent T1 O1 O3)
  (tangent2 : Tangent T2 O2 O3)
  (equal_radii : Radius O1 = Radius O2)
  (homothety_center_exists : HomothetyCenter O1 O2) :
  (Line_through T1 T2) ∧ (Passes_through_homothety_center (Line_through T1 T2) homothety_center_exists) :=
sorry

end tangent_line_passes_through_homothety_center_tangent_line_passes_through_homothety_center_equal_radii_l250_250450


namespace smallest_positive_period_monotonically_increasing_intervals_max_min_on_interval_l250_250689

noncomputable def f (x : ℝ) : ℝ :=
  (Float.sqrt 3 * Real.cos x - Real.sin x) * Real.sin x

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
sorry

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x ∈ (Set.Icc (k * π - π / 3) (k * π + π / 6)), 
  ∃ a b, f a ≤ f x ∧ f x ≤ f b :=
sorry

theorem max_min_on_interval :
  ∀ x ∈ (Set.Icc 0 (π / 4)), 0 ≤ f x ∧ f x ≤ 0.5 :=
sorry

end smallest_positive_period_monotonically_increasing_intervals_max_min_on_interval_l250_250689


namespace find_4_oplus_2_l250_250801

def operation (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem find_4_oplus_2 : operation 4 2 = 26 :=
by
  sorry

end find_4_oplus_2_l250_250801


namespace word_PROOF_arrangements_l250_250975

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem word_PROOF_arrangements : 
  let n := 5 in  -- total number of letters
  let k := 2 in  -- number of times 'O' repeats
  factorial n / factorial k = 60 :=
by
  let n := 5
  let k := 2
  have h1 : factorial n = 120 := 
    by sorry  -- This would use the definition of factorial applied to 5
  have h2 : factorial k = 2 :=
    by sorry  -- This would use the definition of factorial applied to 2
  have h3 : 120 / 2 = 60 := 
    by sorry  -- Arithmetic division step
  show factorial n / factorial k = 60 from
  by 
    rw [h1, h2, h3]
    sorry  -- Combine the previous steps’ results

end word_PROOF_arrangements_l250_250975


namespace atomic_weight_of_carbon_is_12_01_l250_250109

-- Define the atomic weight of oxygen as a constant
def atomic_weight_oxygen := 16.00

-- Define the molecular weight of the compound as a constant
def molecular_weight_compound := 28.00

-- Define the atomic weight of carbon as a variable
def atomic_weight_carbon (C : ℝ) : Prop :=
  C + atomic_weight_oxygen = molecular_weight_compound

-- Statement proving the atomic weight of carbon
theorem atomic_weight_of_carbon_is_12_01 : atomic_weight_carbon 12.01 :=
by
  unfold atomic_weight_carbon
  sorry

end atomic_weight_of_carbon_is_12_01_l250_250109


namespace blue_balls_balance_l250_250010

variables {R B O P : ℝ}

-- Given conditions
def cond1 : 4 * R = 8 * B := sorry
def cond2 : 3 * O = 7 * B := sorry
def cond3 : 8 * B = 6 * P := sorry

-- Proof problem: proving equal balance of 5 red balls, 3 orange balls, and 4 purple balls
theorem blue_balls_balance : 5 * R + 3 * O + 4 * P = (67 / 3) * B :=
by
  sorry

end blue_balls_balance_l250_250010


namespace cubes_intersected_by_diagonal_l250_250536

theorem cubes_intersected_by_diagonal (a b c : ℕ) (hab : nat.gcd a b = 6) (hbc : nat.gcd b c = 3) (hca : nat.gcd c a = 75) (habc : nat.gcd (nat.gcd a b) c = 3) :
  a + b + c - (nat.gcd a b + nat.gcd b c + nat.gcd c a) + nat.gcd (nat.gcd a b) c = 768 :=
by
  -- Prove the specific numerical case for a = 150, b = 324, c = 375
  let a := 150
  let b := 324
  let c := 375
  have h1 : nat.gcd 150 324 = 6 := hab
  have h2 : nat.gcd 324 375 = 3 := hbc
  have h3 : nat.gcd 375 150 = 75 := hca
  have h4 : nat.gcd (nat.gcd 150 324) 375 = 3 := habc
  sorry

end cubes_intersected_by_diagonal_l250_250536


namespace arithmetic_sequence_ratio_l250_250696

variable {a_n b_n : ℕ → ℕ}
variable {S_n T_n : ℕ → ℕ}

-- Given two arithmetic sequences a_n and b_n, their sums of the first n terms are S_n and T_n respectively.
-- Given that S_n / T_n = (2n + 2) / (n + 3).
-- Prove that a_10 / b_10 = 20 / 11.

theorem arithmetic_sequence_ratio (h : ∀ n, S_n n / T_n n = (2 * n + 2) / (n + 3)) : (a_n 10) / (b_n 10) = 20 / 11 := 
by
  sorry

end arithmetic_sequence_ratio_l250_250696


namespace any_integer_amount_purchasable_amount_over_mn_minus_two_payable_l250_250956
open Int

variable (m n : ℕ) (h : Nat.gcd m n = 1)

theorem any_integer_amount_purchasable (x : ℤ) : 
  ∃ (a b : ℤ), a * n + b * m = x :=
by sorry

theorem amount_over_mn_minus_two_payable (k : ℤ) (hk : k > m * n - 2) : 
  ∃ (a b : ℤ), a * n + b * m = k :=
by sorry

end any_integer_amount_purchasable_amount_over_mn_minus_two_payable_l250_250956


namespace positive_integer_solutions_sum_10_l250_250184

theorem positive_integer_solutions_sum_10 :
  ∃ (n : ℕ), (∑ k in finset.filter (λ (k : ℕ × ℕ × ℕ), (k.1 + k.2.1 + k.2.2 = 10) ∧ k.1 > 0 ∧ k.2.1 > 0 ∧ k.2.2 > 0) 
  ({(x, y, z) | x, y, z ∈ finset.range 11}) finset.univ, 1) = n ∧ n = 36 :=
sorry

end positive_integer_solutions_sum_10_l250_250184


namespace computer_operations_correct_l250_250919

-- Define the rate of operations per second
def operations_per_second : ℝ := 4 * 10^8

-- Define the total number of seconds the computer operates
def total_seconds : ℝ := 6 * 10^5

-- Define the expected total number of operations
def expected_operations : ℝ := 2.4 * 10^14

-- Theorem stating the total number of operations is as expected
theorem computer_operations_correct :
  operations_per_second * total_seconds = expected_operations :=
by
  sorry

end computer_operations_correct_l250_250919


namespace total_cost_correct_l250_250101

def sandwich_cost : ℝ := 2.45
def soda_cost : ℝ := 0.87
def sandwich_quantity : ℕ := 2
def soda_quantity : ℕ := 4

theorem total_cost_correct :
  sandwich_quantity * sandwich_cost + soda_quantity * soda_cost = 8.38 := 
  by
    sorry

end total_cost_correct_l250_250101


namespace common_chord_length_l250_250493

-- Define the radii and distance between circle centers
def R : ℝ := 13
def r : ℝ := 5
def d : ℝ := 12

-- State the theorem to prove the length of the common chord is 10
theorem common_chord_length : 2 * R * real.sin (real.acos ((R^2 + d^2 - r^2) / (2 * R * d))) = 10 := 
by 
  sorry

end common_chord_length_l250_250493


namespace ratio_of_sums_l250_250624

theorem ratio_of_sums :
  let numerator_sequence := λ (n : ℕ), 4 + (n - 1) * 4 in
  let denominator_sequence := λ (n : ℕ), 5 + (n - 1) * 5 in
  let n_num := 15 in  -- Calculated from 60 = 4 + (n-1) * 4
  let n_den := 15 in  -- Calculated from 75 = 5 + (n-1) * 5
  let S_num := (n_num * (4 + 60)) / 2 in
  let S_den := (n_den * (5 + 75)) / 2 in
  S_num / S_den = 4 / 5 :=
by {
  sorry
}

end ratio_of_sums_l250_250624


namespace largest_non_sum_of_elements_of_A_n_l250_250395

theorem largest_non_sum_of_elements_of_A_n
  (n : ℕ) (h : n ≥ 2) :
  ∃ m : ℕ, m = (n - 2) * 2^n + 1 ∧
  (∀ s : multiset ℕ, (∀ x ∈ s, x ∈ {2^n - 2^k | k : ℤ, 0 ≤ k ∧ k < n}) → s.sum ≠ m) :=
sorry

end largest_non_sum_of_elements_of_A_n_l250_250395


namespace wine_limit_l250_250112

noncomputable def wine_in_container (a_0 : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a_0
  else (9 / 10) * wine_in_container a_0 (n - 1) + (1 / 10)

theorem wine_limit (a_0 : ℝ) (h : a_0 = 1) : 
  filter.tendsto (λ n, wine_in_container a_0 n) filter.at_top (𝓝 (1 / 2)) :=
sorry

end wine_limit_l250_250112


namespace max_flight_time_correct_l250_250651

def V0 : ℝ := 10  -- initial velocity in m/s
def g : ℝ := 10   -- acceleration due to gravity in m/s^2

-- Range of projectile
def range (α : ℝ) : ℝ := (V0 ^ 2 * Real.sin(2 * α)) / g

-- Time of flight of projectile
def time_of_flight (α : ℝ) : ℝ := (2 * V0 * Real.sin(α)) / g

-- Specify the angle condition
def angle_condition (α : ℝ) : Prop := Real.sin(2 * α) >= 0.96

-- The flight time
def max_flight_time : ℝ := 1.6

theorem max_flight_time_correct (α : ℝ) (h : angle_condition α) : 
  time_of_flight α = max_flight_time :=
sorry

end max_flight_time_correct_l250_250651


namespace arithmetic_sequence_eighth_term_is_l250_250083

noncomputable def nth_term_of_arithmetic_sequence
  (n k : ℕ) (a l : ℚ)
  (h1 : n = 25)
  (h2 : a = 7)
  (h3 : l = 98) :
  ℚ :=
  let d := (l - a) / (n - 1) in
  a + (k - 1) * d

theorem arithmetic_sequence_eighth_term_is
  (h1 : nth_term_of_arithmetic_sequence 25 8 7 98 25 rfl 7 rfl 98 rfl = 343 / 12) :
  nth_term_of_arithmetic_sequence 25 8 7 98 = 343 / 12 :=
  by
    sorry

end arithmetic_sequence_eighth_term_is_l250_250083


namespace correct_statements_B_and_C_l250_250514

theorem correct_statements_B_and_C :
  (let A := (1 : ℝ, -3 : ℝ),
       B := (1 : ℝ, 3 : ℝ),
       line_eq_C := λ x, x + 1) in
  (
    -- Condition for Statement B
    (A.1 = B.1) ∧
    (let θ := 90 in θ = 90) ∧
    -- Condition for Statement C
    (line_eq_C 1 = 2) ∧
    (line_eq_C 3 = 4)
  ) :=
begin
  sorry
end

end correct_statements_B_and_C_l250_250514


namespace factorize_xcube_minus_x_l250_250994

theorem factorize_xcube_minus_x (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by 
  sorry

end factorize_xcube_minus_x_l250_250994


namespace center_lies_on_axis_of_symmetry_l250_250832

-- Define what it means for a figure to be bounded
def is_bounded (F : Type) [metric_space F] : Prop :=
  ∃ R : ℝ, ∀ (x : F), dist x 0 ≤ R

-- Define what it means for a figure to have an axis of symmetry
def has_axis_of_symmetry (F : Type) (axis : ℝ) : Prop :=
  ∀ (x : F), ∃ (y : F), y ≠ x ∧ ∃ (p : ℝ × ℝ), x = (p.1, p.2) ∧ y = (p.1, -p.2)

-- Define what it means for a figure to have a center of symmetry
def has_center_of_symmetry (F : Type) (center : ℝ × ℝ) : Prop :=
  ∀ (x : F), ∃ (y : F), y ≠ x ∧ ∃ (p1 p2 : ℝ × ℝ), x = p1 ∧ y = (2*center.1 - p1.1, 2*center.2 - p1.2)

-- Define the proposition stating the center of symmetry lies on the axis of symmetry
theorem center_lies_on_axis_of_symmetry
  {F : Type} [metric_space F] (axis : ℝ) (center : ℝ × ℝ)
  (h1 : is_bounded F) (h2 : has_axis_of_symmetry F axis) (h3 : has_center_of_symmetry F center) :
  center.2 = 0 :=
begin
  sorry
end

end center_lies_on_axis_of_symmetry_l250_250832


namespace minimum_distance_l250_250056

open Real

/-- Definition of the curve y = e^x + 1 -/
def curve (x : ℝ) : ℝ := exp x + 1

/-- Definition of the line x - y - 2 = 0 -/
def line (x y : ℝ) : Prop := x - y - 2 = 0

theorem minimum_distance : ∃ P : ℝ × ℝ, curve P.1 = P.2 ∧ 
  (∀ Q : ℝ × ℝ, curve Q.1 = Q.2 → dist P.1 (P.2) (line Q.1 Q.2) >= dist P.1 P.2 ) ∧ 
  dist P.1 P.2 line P.1  P.2 = 2 * sqrt 2 := 
sorry

end minimum_distance_l250_250056


namespace lines_AT_and_PQ_are_perpendicular_l250_250920

open_locale real

variables {A B C D P Q R T : Type*} [circle_diameter AB A B]
variables [points_on_circle C D A B]
variables [AC_lt_BC_or_AC_lt_AD AC BC AD]
variables [P_on_segment_BC P BC]
variables [angle_CAP_eq_angle_ABC CAP ABC]
variables [perpendicular_from_C_to_AB_intersects_BD Q C AB BD]
variables [lines_PQ_AD_intersect_R PQ AD R]
variables [lines_PQ_CD_intersect_T PQ CD T]
variables [AR_eq_RQ AR RQ]

theorem lines_AT_and_PQ_are_perpendicular 
  (cir Diameter AB A B)
  (points Circle C D A B)
  (h : AC < BC ∨ AC < AD)
  (P Segment BC)
  (CAP ABC : ∠CAP = ∠ABC)
  (CQ_perpendicular_AB BD)
  (PQ_AD_R : PQ ∩ AD = R)
  (PQ_CD_T : PQ ∩ CD = T)
  (isosceles_triangle AR_eq_RQ : AR = RQ):
  ∠(AT, PQ) = 90 :=
sorry

end lines_AT_and_PQ_are_perpendicular_l250_250920


namespace triangle_a_eq_5_over_3_triangle_b_plus_c_eq_4_l250_250762

theorem triangle_a_eq_5_over_3
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B))
  (h2 : b = Real.sqrt 5 * Real.sin B) :
  a = 5 / 3 := sorry

theorem triangle_b_plus_c_eq_4
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B))
  (h2 : a = Real.sqrt 6)
  (h3 : 1 / 2 * b * c * Real.sin A = Real.sqrt 5 / 2) :
  b + c = 4 := sorry

end triangle_a_eq_5_over_3_triangle_b_plus_c_eq_4_l250_250762


namespace number_of_digits_in_value_l250_250617

theorem number_of_digits_in_value :
  let expr := (2^15 * 5^11)
  digits expr = 12 :=
by
  have h : expr = 16 * 10^11 := by sorry
  exact h

end number_of_digits_in_value_l250_250617


namespace math_problem_equivalence_l250_250165

theorem math_problem_equivalence :
  (-3 : ℚ) / (-1 - 3 / 4) * (3 / 4) / (3 / 7) = 3 := 
by 
  sorry

end math_problem_equivalence_l250_250165


namespace find_sum_of_money_l250_250564

theorem find_sum_of_money 
  (R : ℝ)        -- original rate of interest
  (P : ℝ)        -- sum of money
  (h : P * 12 * (R + 7.5) / 100 - P * 12 * R / 100 = 630) : 
  P = 700 := 
begin
  sorry
end

end find_sum_of_money_l250_250564


namespace arithmetic_sequence_common_difference_l250_250360

theorem arithmetic_sequence_common_difference
  (a_n : ℕ → ℝ)  -- the arithmetic sequence
  (a1 : ℝ)   -- the first term of the sequence
  (d : ℝ)    -- the common difference
  (h1 : a_n 7 = 8)  -- the 7th term is 8
  (h2 : 7 * a1 + (7 * 6 / 2) * d = 42)  -- the sum of the first 7 terms is 42
  : d = 2 / 3 :=
begin
  sorry,
end

end arithmetic_sequence_common_difference_l250_250360


namespace ball_total_distance_l250_250138

-- Definitions based on conditions identified in step a)
def initial_height : ℝ := 20
def bounce_ratio : ℚ := 2 / 3

def total_distance (h : ℝ) (r : ℚ) (bounces : ℕ) : ℝ :=
  sorry -- This will compute the total distance based on the formula.

-- The statement to prove that the total distance is 68 meters to the nearest meter.
theorem ball_total_distance (h : ℝ) (r : ℚ) (bounces : ℕ) : 
  total_distance h r bounces ≈ 68 ∧ h = 20 ∧ r = (2 / 3) ∧ bounces = 4 :=
by
  sorry

end ball_total_distance_l250_250138


namespace find_six_quotients_l250_250209

def is_5twos_3ones (n: ℕ) : Prop :=
  n.digits 10 = [2, 2, 2, 2, 2, 1, 1, 1]

def divides_by_7 (n: ℕ) : Prop :=
  n % 7 = 0

theorem find_six_quotients:
  ∃ n₁ n₂ n₃ n₄ n₅: ℕ, 
    n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₂ ≠ n₃ ∧ n₁ ≠ n₄ ∧ n₂ ≠ n₄ ∧ n₃ ≠ n₄ ∧ n₁ ≠ n₅ ∧ n₂ ≠ n₅ ∧ n₃ ≠ n₅ ∧ n₄ ≠ n₅ ∧
    is_5twos_3ones n₁ ∧ is_5twos_3ones n₂ ∧ is_5twos_3ones n₃ ∧ is_5twos_3ones n₄ ∧ is_5twos_3ones n₅ ∧
    divides_by_7 n₁ ∧ divides_by_7 n₂ ∧ divides_by_7 n₃ ∧ divides_by_7 n₄ ∧ divides_by_7 n₅ ∧
    n₁ / 7 = 1744603 ∧ n₂ / 7 = 3031603 ∧ n₃ / 7 = 3160303 ∧ n₄ / 7 = 3017446 ∧ n₅ / 7 = 3030316 :=
sorry

end find_six_quotients_l250_250209


namespace inequality_example_l250_250831

variable {a b c : ℝ} -- Declare a, b, c as real numbers

theorem inequality_example
  (ha : 0 < a)  -- Condition: a is positive
  (hb : 0 < b)  -- Condition: b is positive
  (hc : 0 < c) :  -- Condition: c is positive
  (ab * (a + b) + ac * (a + c) + bc * (b + c)) / (abc) ≥ 6 := 
sorry  -- Proof is skipped

end inequality_example_l250_250831


namespace how_many_integers_satisfy_l250_250710

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end how_many_integers_satisfy_l250_250710


namespace Find_AC_l250_250768

noncomputable def length_of_AC
  (ABC : Triangle)
  (A B C H X Y : Point)
  (r : Real)
  (rpos : r > 0)
  (AB : Real) (AX : Real) (AY : Real)
  (h : Line)
  (circ : Circle)
  (right_angle_at_A : angle A B C = 90)
  (altitude_AH : Line)
  (altitude_condition : ⊥_  altitude_AH B C)
  (circle_condition : passes_through circ A /\ passes_through circ H)
  (circle_intersections : intersects circ h X/\ intersects circ h Y)
  (AX_cond : length AX = 5)
  (AY_cond : length AY = 6)
  (AB_cond : length AB = 9) : Real :=
  let AC := (9 * 6) / 4
  AC

theorem Find_AC 
  (ABC : Triangle)
  (A B C H X Y : Point)
  (AB AX AY : Real)
  (h : Line)
  (circ : Circle)
  (right_angle_at_A : angle A B C = 90)
  (altitude_AH : Line)
  (altitude_condition : ⊥_  altitude_AH B C)
  (circle_condition : passes_through circ A /\ passes_through circ H)
  (circle_intersections : intersects circ h X /\ intersects circ h Y)
  (AX_cond : length AX = 5)
  (AY_cond : length AY = 6)
  (AB_cond : length AB = 9)
 : length (side AC of ABC) = 13.5 := sorry 

end Find_AC_l250_250768


namespace sum_series_upto_9_l250_250181

open Nat

noncomputable def series_sum_to (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n+1), (1 : ℝ) / (2 : ℝ) ^ i

theorem sum_series_upto_9 : series_sum_to 9 = 2 - 2 ^ (-9 : ℝ) :=
by
  sorry

end sum_series_upto_9_l250_250181


namespace profit_per_meter_correct_l250_250566

noncomputable def total_selling_price := 6788
noncomputable def num_meters := 78
noncomputable def cost_price_per_meter := 58.02564102564102
noncomputable def total_cost_price := 4526 -- rounded total
noncomputable def total_profit := 2262 -- calculated total profit
noncomputable def profit_per_meter := 29

theorem profit_per_meter_correct :
  (total_selling_price - total_cost_price) / num_meters = profit_per_meter :=
by
  sorry

end profit_per_meter_correct_l250_250566


namespace solve_logarithm_eq_l250_250843

theorem solve_logarithm_eq (x : ℝ) (h : log 7 x - 3 * log 7 2 = 1) : x = 56 := 
by
  sorry

end solve_logarithm_eq_l250_250843


namespace cylindrical_to_rectangular_conversion_l250_250613

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular 6 (5 * Real.pi / 4) (-3) = (-3 * Real.sqrt 2, -3 * Real.sqrt 2, -3) :=
by
  sorry

end cylindrical_to_rectangular_conversion_l250_250613


namespace find_value_of_question_mark_l250_250520

theorem find_value_of_question_mark (x : ℕ) (h : sqrt x / 13 = 4) : x = 2704 := 
by 
  sorry

end find_value_of_question_mark_l250_250520


namespace extra_flowers_l250_250906

theorem extra_flowers (tulips roses flowers_used : ℕ) (h_tulips : tulips = 4) (h_roses : roses = 11) (h_used : flowers_used = 11) :
  tulips + roses - flowers_used = 4 :=
by
  rw [h_tulips, h_roses, h_used]
  norm_num
  done

#eval extra_flowers 4 11 11 (by rfl) (by rfl) (by rfl) 

end extra_flowers_l250_250906


namespace joe_money_left_l250_250380

theorem joe_money_left (starting_amount : ℕ) (num_notebooks : ℕ) (cost_per_notebook : ℕ) (num_books : ℕ) (cost_per_book : ℕ)
  (h_starting_amount : starting_amount = 56)
  (h_num_notebooks : num_notebooks = 7)
  (h_cost_per_notebook : cost_per_notebook = 4)
  (h_num_books : num_books = 2)
  (h_cost_per_book : cost_per_book = 7) : 
  starting_amount - (num_notebooks * cost_per_notebook + num_books * cost_per_book) = 14 :=
by
  rw [h_starting_amount, h_num_notebooks, h_cost_per_notebook, h_num_books, h_cost_per_book]
  -- sorry lõpetab ajutiselt
  norm_num  
  -- sorry 

end joe_money_left_l250_250380


namespace carol_points_loss_l250_250168

theorem carol_points_loss 
  (first_round_points : ℕ) (second_round_points : ℕ) (end_game_points : ℕ) 
  (h1 : first_round_points = 17) 
  (h2 : second_round_points = 6) 
  (h3 : end_game_points = 7) : 
  (first_round_points + second_round_points - end_game_points = 16) :=
by 
  sorry

end carol_points_loss_l250_250168


namespace sum_telescope_l250_250957

open BigOperators

theorem sum_telescope :
  ∑ n in Finset.range 99 \ Finset.singleton 0, (3 : ℚ) / ((2 * (n + 2) - 3) * (2 * (n + 2) + 1)) = 300 / 201 :=
by
  sorry

end sum_telescope_l250_250957


namespace max_abs_sum_of_squares_eq_2_sqrt_2_l250_250325

theorem max_abs_sum_of_squares_eq_2_sqrt_2 (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_of_squares_eq_2_sqrt_2_l250_250325


namespace evaluate_expression_at_x_eq_2_l250_250889

theorem evaluate_expression_at_x_eq_2 :
  (3 * 2 + 4)^2 = 100 := by
  sorry

end evaluate_expression_at_x_eq_2_l250_250889


namespace problem_theorem_l250_250668

variables {V : Type*} [inner_product_space ℝ V]

-- Conditions of the problem
variables (A B C O N : V)
variables (h1 : dist O A = dist O B)
variables (h2 : dist O B = dist O C)
variables (h3 : (N -ᵥ A : V) + (N -ᵥ B) + (N -ᵥ C) = 0)

-- Translate the conditions to definitions in Lean
def is_circumcenter (O : V) (A B C : V) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

def is_centroid (N : V) (A B C : V) : Prop :=
  (N -ᵥ A : V) + (N -ᵥ B) + (N -ᵥ C) = 0

-- Statement of the theorem to prove
theorem problem_theorem (A B C O N : V)
  (h1 : dist O A = dist O B)
  (h2 : dist O B = dist O C)
  (h3 : (N -ᵥ A : V) + (N -ᵥ B) + (N -ᵥ C) = 0) :
  is_circumcenter O A B C ∧ is_centroid N A B C :=
by
  sorry

end problem_theorem_l250_250668


namespace team_A_total_victory_prob_l250_250446

variables (p_w5 p_wo5 : ℚ) (P_total : ℚ)

-- Conditions
def team_A_fifth_set_win_prob : p_w5 = 1 / 2 := sorry
def team_A_other_sets_win_prob : p_wo5 = 2 / 3 := sorry
def sets_are_independent : true := sorry
def match_victory_condition : ∀ n, n ≥ 3 → true := sorry

-- Question translated as a theorem to prove
theorem team_A_total_victory_prob (h1 : team_A_fifth_set_win_prob p_w5) (h2 : team_A_other_sets_win_prob p_wo5)
  (h3 : sets_are_independent) (h4 : match_victory_condition) : 
  P_total = 20 / 27 :=
sorry

end team_A_total_victory_prob_l250_250446


namespace proof_problem_l250_250189

noncomputable def eval_expr : ℝ :=
  (real.sqrt ((-5 / 3) ^ 2)) + ((27 / 64) ^ (-1:ℤ / 3:ℤ)) - (real.pi ^ 0) + (real.log_base (1 / 2) 2)

theorem proof_problem : eval_expr = 1 := 
  sorry

end proof_problem_l250_250189


namespace curve_is_circle_l250_250210

theorem curve_is_circle (r : ℝ) (θ : ℝ) (h : r = 3) : 
  ∃ (c : ℝ) (p : ℝ × ℝ), c = 3 ∧ p = (3 * Real.cos θ, 3 * Real.sin θ) := 
sorry

end curve_is_circle_l250_250210


namespace total_cost_of_supplies_l250_250148

variable (E P M : ℝ)

open Real

theorem total_cost_of_supplies (h1 : E + 3 * P + 2 * M = 240)
                                (h2 : 2 * E + 4 * M + 5 * P = 440)
                                : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_supplies_l250_250148


namespace sampling_is_systematic_l250_250133

-- Definitions corresponding to conditions
def audience_rows : ℕ := 25
def seats_per_row : ℕ := 20
def total_students : ℕ := audience_rows * seats_per_row
def students_kept_back : ℕ := 25
def seat_number_kept_back : ℕ := 15

-- Mathematical equivalent proof problem
theorem sampling_is_systematic :
  ∀ (audience_rows seats_per_row total_students students_kept_back seat_number_kept_back : ℕ),
  audience_rows = 25 →
  seats_per_row = 20 →
  total_students = audience_rows * seats_per_row →
  students_kept_back = 25 →
  seat_number_kept_back = 15 →
  (∃ (n : ℕ), ∀ i, seat_number_kept_back = (1 + i * n) ∧ n = 20) →
  "Systematic sampling method" := 
by
  sorry


end sampling_is_systematic_l250_250133


namespace seed_mixture_Y_is_25_percent_ryegrass_l250_250025

variables (X Y : ℝ) (R : ℝ)

def proportion_X_is_40_percent_ryegrass : Prop :=
  X = 40 / 100

def proportion_Y_contains_percent_ryegrass (R : ℝ) : Prop :=
  100 - R = 75 / 100 * 100

def mixture_contains_30_percent_ryegrass (X Y R : ℝ) : Prop :=
  (1/3) * (40 / 100) * 100 + (2/3) * (R / 100) * 100 = 30

def weight_of_mixture_is_33_percent_X (X Y : ℝ) : Prop :=
  X / (X + Y) = 1 / 3

theorem seed_mixture_Y_is_25_percent_ryegrass
  (X Y : ℝ) (R : ℝ) 
  (h1 : proportion_X_is_40_percent_ryegrass X)
  (h2 : proportion_Y_contains_percent_ryegrass R)
  (h3 : weight_of_mixture_is_33_percent_X X Y)
  (h4 : mixture_contains_30_percent_ryegrass X Y R) :
  R = 25 :=
sorry

end seed_mixture_Y_is_25_percent_ryegrass_l250_250025


namespace clothing_loss_l250_250136

theorem clothing_loss
  (a : ℝ)
  (h1 : ∃ x y : ℝ, x * 1.25 = a ∧ y * 0.75 = a ∧ x + y - 2 * a = -8) :
  a = 60 :=
sorry

end clothing_loss_l250_250136


namespace avg_speed_comparison_l250_250605

variable (u v : ℝ) (D : ℝ)

noncomputable def avg_speed_C : ℝ := 
  (4 / ((1 / u) + (1 / v)))

noncomputable def avg_speed_D : ℝ := 
  ((3 * u + 3 * v) / 2)

theorem avg_speed_comparison (h1 : D > 0) (h2 : u > 0) (h3 : v > 0) : 
  avg_speed_C u v D ≤ avg_speed_D u v :=
by
  sorry

end avg_speed_comparison_l250_250605


namespace glen_hannah_distance_l250_250700

theorem glen_hannah_distance :
  ∀ (D : ℕ) (T : ℕ), let glen_speed := 37 in
                    let hannah_speed := 15 in
                    let total_time := 5 in
                    (2 * T = total_time) →
                    D = (glen_speed * T) + (hannah_speed * T) →
                    D = 130 :=
by
  intros D T glen_speed hannah_speed total_time h1 h2
  have glen_speed : ℕ := 37
  have hannah_speed : ℕ := 15
  have total_time : ℕ := 5
  rw [←nat.cast_mul, h1]
  rw [←nat.cast_add, ←mul_add, add_comm, mul_comm] at h2
  have h3: T = 2.5 := sorry
  rw [←nat.cast_mul, h3, mul_comm] at h2
  exact sorry

end glen_hannah_distance_l250_250700


namespace compute_sum_series_l250_250607

noncomputable def sum_series : ℚ :=
  ∑' (n : ℕ) in set.Ici 3, ∑' (k : ℕ) in set.Icc 2 (n - 1), (k ^ 2) / (3 ^ (n + k))

theorem compute_sum_series : sum_series = 65609 / 1024 := by
  sorry

end compute_sum_series_l250_250607


namespace total_working_days_l250_250930

-- Define the variables
variables (a b c x : ℕ)

-- Define the conditions
def condition1 : Prop := b + c = 8
def condition2 : Prop := a + c = 15
def condition3 : Prop := a + b = 9
def condition4 : Prop := x = a + b + c

-- Prove the total number of working days
theorem total_working_days (h1 : condition1) (h2 : condition2) (h3 : condition3) : x = 16 :=
by
  -- Explicitly bind variables (a b c x) to avoid confusion in further context
  let a := 8
  let b := 1
  let c := 7
  rw [condition4] at *,
  -- Use given conditions to solve for x
  calc
  x = a + b + c : by rw [condition4]
  _ = 8 + 1 + 7 : by norm_num
  _ = 16 : by norm_num

end total_working_days_l250_250930


namespace mixture_kerosene_l250_250013

theorem mixture_kerosene (x : ℝ) (h₁ : 0.25 * x + 1.2 = 0.27 * (x + 4)) : x = 6 :=
sorry

end mixture_kerosene_l250_250013


namespace count_integers_satisfying_inequality_l250_250724

theorem count_integers_satisfying_inequality : 
  ∃ (s : Finset ℤ), (∀ n ∈ s, (n - 3) * (n + 5) < 0) ∧ s.card = 7 :=
begin
  sorry
end

end count_integers_satisfying_inequality_l250_250724


namespace find_n_squares_l250_250972

theorem find_n_squares (n : ℤ) : 
  (∃ a : ℤ, n^2 + 6 * n + 24 = a^2) ↔ n = 4 ∨ n = -2 ∨ n = -4 ∨ n = -10 :=
by
  sorry

end find_n_squares_l250_250972


namespace find_a5_plus_a7_l250_250755

variable {a : ℕ → ℕ}

-- Assume a is a geometric sequence with common ratio q and first term a1.
def geometric_sequence (a : ℕ → ℕ) (a_1 : ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a_1 * q ^ n

-- Given conditions of the problem:
def conditions (a : ℕ → ℕ) : Prop :=
  a 2 + a 4 = 20 ∧ a 3 + a 5 = 40

-- The objective is to prove a_5 + a_7 = 160
theorem find_a5_plus_a7 (a : ℕ → ℕ) (a_1 q : ℕ) (h_geo : geometric_sequence a a_1 q) (h_cond : conditions a) : a 5 + a 7 = 160 :=
  sorry

end find_a5_plus_a7_l250_250755


namespace card_pair_probability_l250_250546

theorem card_pair_probability :
  let deck_size := 40
  let matching_pair_removed := 2
  let remaining_cards := deck_size - matching_pair_removed
  let cards_per_number := 4
  let removed_number := 1
  let remaining_cards_per_number := cards_per_number - (if removed_number = 1 then 1 else 0)
  let total_ways := (remaining_cards * (remaining_cards - 1)) / 2
  let favor_cases_with_3 := binom 3 2 -- choosing from 3 cards
  let favor_cases_with_4 := binom 4 2 * 9 -- choosing from 4 cards in 9 sets
  let total_favor_cases := favor_cases_with_3 + favor_cases_with_4
  let probability := total_favor_cases / total_ways
  let (m, n) := (57, 703)
  in rel_prime m n ∧ m + n = 760 :=
by
  sorry

end card_pair_probability_l250_250546


namespace rational_function_q_l250_250856

theorem rational_function_q (q : ℝ → ℝ) (a : ℝ) 
  (h₁ : ∀ x, q(x) = a * (x + 2) * (x - 3)) 
  (h₂ : q(1) = -10) : 
  q(x) = (5 / 3) * x^2 - (5 / 3) * x - 10 := 
sorry

end rational_function_q_l250_250856


namespace increasing_function_afx_gt_x_l250_250805

-- Part (1)
theorem increasing_function (f : ℝ → ℝ) (x : ℝ) (h₁ : 0 < x ∧ x < 1 ∨ 1 < x) :
  (∀ x, f(x) = (x^2 - 1) / (Real.log x)) → (∀ x, 0 < x ∧ x < 1 ∨ 1 < x → ∃ ε > 0,  f'(x) > 0) :=
by
  sorry

-- Part (2)
theorem afx_gt_x (f : ℝ → ℝ) (a : ℝ) (x : ℝ) (h₂ : 0 < x ∧ x ≠ 1) :
  (∀ x, f(x) = (x^2 - 1) / (Real.log x)) → (∀ a ≥ 1/2, af(x) > x) :=
by
  sorry

end increasing_function_afx_gt_x_l250_250805


namespace symmetry_axis_l250_250853

noncomputable def f (x : ℝ) : ℝ := sin (x - (π / 4))

theorem symmetry_axis : ∃ k : ℤ, k = -1 ∧ ∀ x : ℝ, f(x) = f(2 * (k * π + (3 * π) / 4) - x) := 
by
  use -1
  sorry

end symmetry_axis_l250_250853


namespace number_of_perfect_square_factors_of_180_l250_250295

-- Define the prime factorization of 180
def prime_factorization_180 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Define what it means to be a factor of 180
def is_factor_of_180 (n : ℕ) : Prop :=
  ∀ p k, (p, k) ∈ prime_factorization_180 → ∃ m, n = p ^ m ∧ m ≤ k

-- Define what it means to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p m, ∃ k, n = p ^ m ∧ even m

-- Function to count factors of 180 that are perfect squares
def count_perfect_square_factors_of_180 : ℕ :=
  List.length [d | d ← List.range 181, is_factor_of_180 d ∧ is_perfect_square d]

-- Main theorem
theorem number_of_perfect_square_factors_of_180 : count_perfect_square_factors_of_180 = 4 := 
by sorry

end number_of_perfect_square_factors_of_180_l250_250295


namespace sqrt_17_irrational_l250_250180

theorem sqrt_17_irrational (h : ∀ (x : ℚ), x^2 - 17 = 0 → x ∈ ℤ) : ¬ ∃ (q : ℚ), q^2 = 17 :=
by
  intro ⟨q, hq⟩
  have hq' : q^2 - 17 = 0 := by rw [hq]; ring
  have : q ∈ ℤ := h q hq'
  have q_square_integral : q^2 ∈ ℤ := by exact_mod_cast (this : q ∈ ℤ) ^ 2
  have : (17 : ℤ) = q^2 := by exact_mod_cast hq
  cases this
  case inl =>
    exact this
  case inr =>
    contradiction

/-
The above statement sets up the theorem that asserts the irrationality of sqrt(17). 
It relies on the condition (Theorem 61013) that any rational root of a polynomial with integer coefficients must be an integer.
-/

end sqrt_17_irrational_l250_250180


namespace Megan_needs_fifteen_folders_l250_250825

def files_initial : ℝ := 93.0
def files_added : ℝ := 21.0
def files_per_folder : ℝ := 8.0
def folders_needed : ℝ := real.ceil ((files_initial + files_added) / files_per_folder)

theorem Megan_needs_fifteen_folders:
  folders_needed = 15 :=
by
  sorry

end Megan_needs_fifteen_folders_l250_250825


namespace TotalMarks_l250_250947

def AmayaMarks (Arts Maths Music SocialStudies : ℕ) : Prop :=
  Maths = Arts - 20 ∧
  Maths = (9 * Arts) / 10 ∧
  Music = 70 ∧
  Music + 10 = SocialStudies

theorem TotalMarks (Arts Maths Music SocialStudies : ℕ) : 
  AmayaMarks Arts Maths Music SocialStudies → 
  (Arts + Maths + Music + SocialStudies = 530) :=
by
  sorry

end TotalMarks_l250_250947


namespace natasha_average_speed_climbing_l250_250901

theorem natasha_average_speed_climbing :
  ∀ D : ℝ,
    (total_time = 3 + 2) →
    (total_distance = 2 * D) →
    (average_speed = total_distance / total_time) →
    (average_speed = 3) →
    (D = 7.5) →
    (climb_speed = D / 3) →
    (climb_speed = 2.5) :=
by
  intros D total_time_eq total_distance_eq average_speed_eq average_speed_is_3 D_is_7_5 climb_speed_eq
  sorry

end natasha_average_speed_climbing_l250_250901


namespace number_of_perfect_square_factors_of_180_l250_250294

-- Define the prime factorization of 180
def prime_factorization_180 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Define what it means to be a factor of 180
def is_factor_of_180 (n : ℕ) : Prop :=
  ∀ p k, (p, k) ∈ prime_factorization_180 → ∃ m, n = p ^ m ∧ m ≤ k

-- Define what it means to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p m, ∃ k, n = p ^ m ∧ even m

-- Function to count factors of 180 that are perfect squares
def count_perfect_square_factors_of_180 : ℕ :=
  List.length [d | d ← List.range 181, is_factor_of_180 d ∧ is_perfect_square d]

-- Main theorem
theorem number_of_perfect_square_factors_of_180 : count_perfect_square_factors_of_180 = 4 := 
by sorry

end number_of_perfect_square_factors_of_180_l250_250294


namespace problem_f2017_eq_neg_sin_l250_250226

noncomputable def f (x : ℝ) : ℝ := Real.cos x
noncomputable def f1 (x : ℝ) : ℝ := deriv f x
noncomputable def fn : ℕ+ → (ℝ → ℝ)
| 1 := f1
| (nat.succ pn) := λ x, deriv (fn pn) x

theorem problem_f2017_eq_neg_sin (x : ℝ) : fn ⟨2017, by norm_num⟩ x = -Real.sin x :=
  sorry

end problem_f2017_eq_neg_sin_l250_250226


namespace compute_t_triangle_area_l250_250881

theorem compute_t_triangle_area :
  let A := (0 : ℝ, 10 : ℝ) in
  let B := (4 : ℝ, 0 : ℝ) in
  let C := (10 : ℝ, 0 : ℝ) in
  (∃ t : ℝ, 
    ∃ T : ℝ × ℝ, 
    ∃ U : ℝ × ℝ, 
    T = (4 - (2 / 5) * t, t) ∧
    U = (10 - t, t) ∧
    1 / 2 * ((10 - t) - (4 - (2 / 5) * t)) * (10 - t) = 18 ∧
    t = 10 - real.sqrt 20) :=
  sorry

end compute_t_triangle_area_l250_250881


namespace floor_of_5_point_7_l250_250203

theorem floor_of_5_point_7 : Int.floor 5.7 = 5 := by
  sorry

end floor_of_5_point_7_l250_250203


namespace schoolchildren_total_collected_l250_250062

-- Define the given conditions as Lean statements
def fourthGradeCabbage : ℕ := 18
def sixthGradeOnions : ℕ := 7
def sixthGradeCucumbers : ℕ := fourthGradeCabbage / 2
def fourthGradeCarrots : ℕ := sixthGradeOnions
def fifthGradeCucumbers : ℕ := 8

-- Total vegetables collected by all students
def totalFourthGrade := fourthGradeCabbage + fourthGradeCarrots
def totalSixthGrade := sixthGradeOnions + sixthGradeCucumbers

-- Proof statement to show the total collection is equal to 49 centners
theorem schoolchildren_total_collected : 
  totalFourthGrade + fifthGradeCucumbers + totalSixthGrade = 49 :=
by
  unfold totalFourthGrade
  unfold totalSixthGrade
  simp only [fourthGradeCabbage, sixthGradeOnions, fourthGradeCarrots, 
    sixthGradeCucumbers, fifthGradeCucumbers]
  sorry

end schoolchildren_total_collected_l250_250062


namespace sum_of_digits_succ_2080_l250_250398

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_succ_2080 (m : ℕ) (h : sum_of_digits m = 2080) :
  sum_of_digits (m + 1) = 2081 ∨ sum_of_digits (m + 1) = 2090 :=
sorry

end sum_of_digits_succ_2080_l250_250398


namespace count_integers_n_satisfying_inequality_l250_250718

theorem count_integers_n_satisfying_inequality :
  {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset.card = 7 :=
by sorry

end count_integers_n_satisfying_inequality_l250_250718


namespace num_perfect_square_factors_of_180_l250_250314

theorem num_perfect_square_factors_of_180 (n : ℕ) (h : n = 180) :
  ∃ k : ℕ, k = 4 ∧ ∀ d : ℕ, d ∣ n → ∃ a b c : ℕ, d = 2^a * 3^b * 5^c ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 :=
by
  use 4
  sorry

end num_perfect_square_factors_of_180_l250_250314


namespace probability_even_digit_pi_probability_one_chinese_one_foreign_l250_250068

-- Part 1: Probability of selecting an even digit from the decimal part of π
theorem probability_even_digit_pi : 
  let digits := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      even_digits := [0, 2, 4, 6, 8] in
  (List.length even_digits : ℚ) / (List.length digits : ℚ) = 1 / 2 :=
by
  sorry

-- Part 2: Probability of randomly selecting exactly one Chinese and one foreign mathematician
theorem probability_one_chinese_one_foreign :
  let chinese := ["Zu Chongzhi", "Liu Hui"],
      foreign := ["Viète", "Euler"],
      total_combinations := 12,  -- calculated using the method of combinations
      favorable_combinations := 8 in
  (favorable_combinations : ℚ) / (total_combinations : ℚ) = 2 / 3 :=
by
  sorry

end probability_even_digit_pi_probability_one_chinese_one_foreign_l250_250068


namespace problem_part_1_problem_part_2_l250_250646

noncomputable def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (log x / log 2) / (log a / log 2) + (log (2 * a - x) / log 2) / (log a / log 2) = 1 / (log 2 / log (a^2 - 1))

noncomputable def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, (log x / log 2) / (log a / log 2) + (log (2 * a - x) / log 2) / (log a / log 2) = 1 / (log 2 / log (a^2 - 1))

theorem problem_part_1 (a : ℝ) :
  a > 1 ∧ a ≠ real.sqrt 2 → has_solution a := sorry

theorem problem_part_2 (a : ℝ) :
  a = 2 → has_exactly_one_solution a := sorry

end problem_part_1_problem_part_2_l250_250646


namespace factorize_cubic_l250_250998

theorem factorize_cubic (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end factorize_cubic_l250_250998


namespace floor_5_7_eq_5_l250_250193

theorem floor_5_7_eq_5 : Int.floor 5.7 = 5 := by
  sorry

end floor_5_7_eq_5_l250_250193


namespace discount_ratios_l250_250585

-- Define the discount conditions
def discount_first_5 (g: ℕ) : ℝ :=
  if g > 0 then min g 5 * 0.06 else 0

def discount_next_10 (g: ℕ) : ℝ :=
  if g > 5 then min (g - 5) 10 * 0.12 else 0

def discount_after_15 (g: ℕ) : ℝ :=
  if g > 15 then (g - 15) * 0.20 else 0

-- Define the total discount calculation
def total_discount (g: ℕ) : ℝ :=
  discount_first_5 g + discount_next_10 g + discount_after_15 g

-- Defining the discount for each individual
def kim_gallons : ℕ := 18
def isabella_gallons : ℕ := 24
def elijah_gallons : ℕ := 35

def kim_discount : ℝ := total_discount kim_gallons
def isabella_discount : ℝ := total_discount isabella_gallons
def elijah_discount : ℝ := total_discount elijah_gallons

-- Theorem statement for discount and their ratios
theorem discount_ratios :
  kim_discount = 2.10 ∧ 
  isabella_discount = 3.30 ∧ 
  elijah_discount = 5.50 ∧ 
  ratio (kim_discount, isabella_discount, elijah_discount) = (1, 1.571, 2.619) :=
by {
  sorry
}

end discount_ratios_l250_250585


namespace charlie_feathers_needed_l250_250962

theorem charlie_feathers_needed:
  let sets := 2 in
  let feathers_per_set := 900 in
  let feathers_collected := 387 in
  let total_feathers_needed := sets * feathers_per_set in
  total_feathers_needed - feathers_collected = 1413 :=
by
  sorry

end charlie_feathers_needed_l250_250962


namespace sum_first_eight_terms_geometric_sequence_l250_250958

noncomputable def sum_of_geometric_sequence (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_eight_terms_geometric_sequence :
  sum_of_geometric_sequence (1/2) (1/3) 8 = 9840 / 6561 :=
by
  sorry

end sum_first_eight_terms_geometric_sequence_l250_250958


namespace problem_statement_l250_250175

theorem problem_statement :
  ∃ (A B C : ℤ), 
    (let x := 2 + √3 / (2 + √3 / (2 + ...))
     ∧ (B % 4 ≠ 0)
     ∧ |A| + |B| + |C| = 120) := by
  sorry

end problem_statement_l250_250175


namespace necessary_not_sufficient_condition_l250_250908

open Function

theorem necessary_not_sufficient_condition (α β : ℝ) : 
  (sin α + cos β = 0) → (sin^2 α + sin^2 β = 1) ∧ (¬(sin^2 α + sin^2 β = 1 → sin α + cos β = 0)) :=
by
  sorry

end necessary_not_sufficient_condition_l250_250908


namespace original_numbers_replacement_l250_250971

theorem original_numbers_replacement (A B C D E F : ℕ) 
  (h1 : A + D + F = 6)
  (h2 : B + E + D = 14)
  (h3 : C + F + A = 6)
  (h4 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ D ≠ E ∧ D ≠ F ∧ E ≠ F)
  (h5 : {A, B, C, D, E, F} = {1, 2, 3, 4, 5, 6})
  : A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4 :=
by
  sorry

end original_numbers_replacement_l250_250971


namespace min_value_inequality_l250_250817

theorem min_value_inequality (x : Fin 100 → ℝ) :
  (∀ i, 0 < x i) ∧ (∑ i, (x i)^2 = 1) →
  (∑ i, (x i)/(1 - (x i)^2)) ≥ (3 * Real.sqrt 3)/2 :=
by 
  sorry

end min_value_inequality_l250_250817


namespace ball_returns_to_dana_after_13_throws_l250_250877

noncomputable def num_girls : ℕ := 13

def next_throw (current : ℕ) : ℕ :=
  (current + 5) % num_girls

def throws_to_return_to_dana : ℕ :=
  Nat.find (λ n, (Nat.iterate next_throw n 1) = 1)

theorem ball_returns_to_dana_after_13_throws : throws_to_return_to_dana = 13 := 
by
  sorry

end ball_returns_to_dana_after_13_throws_l250_250877


namespace fifteenth_prime_is_47_l250_250982

theorem fifteenth_prime_is_47 (h : (Nat.nth_prime 5) = 11) : (Nat.nth_prime 15) = 47 := sorry

end fifteenth_prime_is_47_l250_250982


namespace count_integers_n_satisfying_inequality_l250_250719

theorem count_integers_n_satisfying_inequality :
  {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset.card = 7 :=
by sorry

end count_integers_n_satisfying_inequality_l250_250719


namespace ababab_divisible_by_7_l250_250007

theorem ababab_divisible_by_7 (a b : ℕ) (ha : a < 10) (hb : b < 10) : (101010 * a + 10101 * b) % 7 = 0 :=
by sorry

end ababab_divisible_by_7_l250_250007


namespace volume_of_region_l250_250641

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y - z| 

theorem volume_of_region : let region := {p : ℝ × ℝ × ℝ | let (x, y, z) := p in f x y z ≤ 6} in
  let volume := 22.5 in
  sorry

end volume_of_region_l250_250641


namespace factorize_cubic_l250_250996

theorem factorize_cubic (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end factorize_cubic_l250_250996


namespace parallel_ac_bm_l250_250791

/-- Given a triangle ABC with angle A being 90 degrees, and D as the orthogonal projection
of A onto BC. Let E be the midpoint of AD and F be the midpoint of AC. M is the circumcenter
of triangle BEF. Then AC is parallel to BM. -/
theorem parallel_ac_bm (A B C D E F M : Point)
  (h_triangle : Triangle A B C)
  (h_right_angle : Angle A = 90)
  (h_orthogonal_projection : OrthogonalProjection A BC D)
  (h_midpoint_E : Midpoint E A D)
  (h_midpoint_F : Midpoint F A C)
  (h_circumcenter : Circumcenter M B E F) : Parallel AC BM :=
sorry

end parallel_ac_bm_l250_250791


namespace valid_rod_pairs_l250_250790

theorem valid_rod_pairs :
  let rods := (1:50).toFinset   -- Set of rods from 1 to 50 cm
  let fixed_rods := {8, 12, 20} -- Rods already chosen
  let remaining_rods := rods \ fixed_rods
  -- Define a valid pair
  let valid_pair (d e : ℕ) := d + e > (8 + 12 + 20) ∧ 8 + 12 + 20 + d > e ∧ 8 + 12 + 20 + e > d ∧
                              ∀ (a b c d e : ℕ), a + b + c + d > e
in
  (remaining_rods.pairs satisfying valid_pair).card = 135 := sorry

end valid_rod_pairs_l250_250790


namespace number_of_perfect_square_factors_of_180_l250_250298

theorem number_of_perfect_square_factors_of_180 :
  let prime_factors := (2, 2, 1); 
  let conditions (p1 p2 p3 : ℕ) := 
    p1 ∈ {0, 2} ∧ 
    p2 ∈ {0, 2} ∧ 
    p3 ∈ {0}
  in Σ' x, x ∈ { (p1, p2, p3) | conditions p1 p2 p3 } = 4 :=
begin
  sorry
end

end number_of_perfect_square_factors_of_180_l250_250298


namespace min_distance_pm_l250_250428

theorem min_distance_pm : 
  (∀ (P M : ℝ × ℝ), 
     (1 ≤ P.1^2 + P.2^2 ∧ P.1^2 + P.2^2 ≤ 9 ∧ 
      P.1 ≥ 0 ∧ P.2 ≥ 0 ∧ 
      (M.1 + 5)^2 + (M.2 + 5)^2 = 1) → 
     (∃ d, ∀ (d' : ℝ),
       d' = real.sqrt((P.1 + 5)^2 + (P.2 + 5)^2) → d' = real.sqrt 61 - 1)) :=
begin
  sorry,
end

end min_distance_pm_l250_250428


namespace tangent_line_range_of_a_l250_250756

open Real

theorem tangent_line_range_of_a (a : ℝ) (h_a : 0 < a) :
  (∃ (x1 x2 : ℝ), (f : ℝ → ℝ) (g : ℝ → ℝ), 
    f = λ x, x^2 + 1 ∧ g = λ x, a * exp x + 1 ∧
    (∃ (m : ℝ), (∃ (b : ℝ), 
       (f x1 = g x2 ∧ f' x1 = g' x2) ∧ 
       f' x1 = m ∧ g' x2 = m ∧
       (f x1 = m * x1 + b) ∧ (g x2 = m * x2 + b)))) ↔ 
  (a ∈ set.Ioc 0 (4 / (exp 2))) :=
sorry

end tangent_line_range_of_a_l250_250756


namespace find_expression_value_l250_250281

theorem find_expression_value (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (2 * x^3 / y^2) + (2 * y^3 / x^2) + y = 342.777... :=
by
  sorry

end find_expression_value_l250_250281


namespace floretta_balloons_l250_250008

theorem floretta_balloons :
  let packs_own := 3
  let packs_neighbor := 2
  let balloons_per_pack := 6
  let milly_extra := 7
  let total_packs := packs_own + packs_neighbor
  let total_balloons := total_packs * balloons_per_pack
  let balloons_each := total_balloons / 2
  let floretta_left := balloons_each - milly_extra
  floretta_left = 8 := 
by 
  -- let variables and calculations be done
  let packs_own := 3
  let packs_neighbor := 2
  let balloons_per_pack := 6
  let milly_extra := 7
  let total_packs := packs_own + packs_neighbor
  let total_balloons := total_packs * balloons_per_pack
  let balloons_each := total_balloons / 2
  let floretta_left := balloons_each - milly_extra
  -- provide the conclusion
  show floretta_left = 8 from sorry

end floretta_balloons_l250_250008


namespace irrational_pi_l250_250512

theorem irrational_pi :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (π = a / b)) :=
sorry

end irrational_pi_l250_250512


namespace paving_cost_correct_l250_250902

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 400
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost_correct :
  cost = 8250 := by
  sorry

end paving_cost_correct_l250_250902


namespace find_polynomials_g_l250_250032

-- Assume f(x) = x^2
def f (x : ℝ) : ℝ := x ^ 2

-- Define the condition that f(g(x)) = 9x^2 - 6x + 1
def condition (g : ℝ → ℝ) : Prop := ∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1

-- Prove that the possible polynomials for g(x) are 3x - 1 or -3x + 1
theorem find_polynomials_g (g : ℝ → ℝ) (h : condition g) :
  (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) :=
sorry

end find_polynomials_g_l250_250032


namespace rectangle_width_is_4_l250_250463

-- Definitions of conditions
variable (w : ℝ) -- width of the rectangle
def length := w + 2 -- length of the rectangle
def perimeter := 2 * w + 2 * (w + 2) -- perimeter of the rectangle, using given conditions

-- The theorem to be proved
theorem rectangle_width_is_4 (h : perimeter = 20) : w = 4 :=
by {
  sorry -- To be proved
}

end rectangle_width_is_4_l250_250463


namespace expression_equivalence_l250_250035

def algebraicExpression : String := "5 - 4a"
def wordExpression : String := "the difference of 5 and 4 times a"

theorem expression_equivalence : algebraicExpression = wordExpression := 
sorry

end expression_equivalence_l250_250035


namespace third_term_binomial_expansion_l250_250067

theorem third_term_binomial_expansion :
  ∀ (x : ℝ), 
  (nat.choose 4 2) * x^2 * 2^2 = 24 * x^2 := by
    sorry

end third_term_binomial_expansion_l250_250067


namespace max_flight_time_correct_l250_250650

def V0 : ℝ := 10  -- initial velocity in m/s
def g : ℝ := 10   -- acceleration due to gravity in m/s^2

-- Range of projectile
def range (α : ℝ) : ℝ := (V0 ^ 2 * Real.sin(2 * α)) / g

-- Time of flight of projectile
def time_of_flight (α : ℝ) : ℝ := (2 * V0 * Real.sin(α)) / g

-- Specify the angle condition
def angle_condition (α : ℝ) : Prop := Real.sin(2 * α) >= 0.96

-- The flight time
def max_flight_time : ℝ := 1.6

theorem max_flight_time_correct (α : ℝ) (h : angle_condition α) : 
  time_of_flight α = max_flight_time :=
sorry

end max_flight_time_correct_l250_250650


namespace count_integers_in_interval_l250_250733

theorem count_integers_in_interval :
  {n : ℤ | -5 < n ∧ n < 3}.finite ∧ {n : ℤ | -5 < n ∧ n < 3}.to_finset.card = 7 := by
sorry

end count_integers_in_interval_l250_250733


namespace A_1988_eq_3314_l250_250652

def is_kept (n : ℕ) : Prop :=
  ¬ (n % 3 = 0 ∨ n % 4 = 0) ∨ n % 5 = 0

def A (n : ℕ) : ℕ :=
  (List.range (n * 5)).filter is_kept ![n - 1]

theorem A_1988_eq_3314 : A 1988 = 3314 :=
  sorry

end A_1988_eq_3314_l250_250652


namespace find_cost_price_l250_250883

theorem find_cost_price (C : ℝ) (h1 : 0.88 * C + 1500 = 1.12 * C) : C = 6250 := 
by
  sorry

end find_cost_price_l250_250883


namespace percent_less_than_weighted_mean_plus_pooled_std_l250_250539

variable (X : Type) (p : ℝ) (m1 m2 d1 d2 : ℝ) [probability_space X]

def weighted_mean (p : ℝ) (m1 m2 : ℝ) : ℝ :=
  p * m1 + (1 - p) * m2

def pooled_standard_deviation (p : ℝ) (d1 d2 m1 m2 : ℝ) : ℝ :=
  real.sqrt (p * d1 ^ 2 + (1 - p) * d2 ^ 2 + p * (1 - p) * (m1 - m2) ^ 2)

axiom bimodal_distribution (X : Type) : Prop

axiom combined_bimodal_distribution_within_one_std (p : ℝ) (m1 m2 d1 d2 : ℝ) :
  bimodal_distribution X → 
  ∃ m d, m = weighted_mean p m1 m2 ∧
  d = pooled_standard_deviation p d1 d2 m1 m2 ∧
  84% of the distribution lies within one standard deviation of m

theorem percent_less_than_weighted_mean_plus_pooled_std (X : Type) (p : ℝ) (m1 m2 d1 d2 : ℝ) :
  bimodal_distribution X →
  ∃ m d, m = weighted_mean p m1 m2 ∧ d = pooled_standard_deviation p d1 d2 m1 m2 →
  ∀ x ∈ X, x < (m + d) → probability_space.probability (event_of (λ y, y < x) X) = 0.84 :=
by sorry

end percent_less_than_weighted_mean_plus_pooled_std_l250_250539


namespace bacon_calories_percentage_l250_250597

theorem bacon_calories_percentage (total_calories : ℕ) (bacon_strip_calories : ℕ) (num_strips : ℕ)
    (h1 : total_calories = 1250) (h2 : bacon_strip_calories = 125) (h3 : num_strips = 2) :
    (bacon_strip_calories * num_strips * 100) / total_calories = 20 := by
  sorry

end bacon_calories_percentage_l250_250597


namespace relationship_between_B_and_C_after_transfer_l250_250946

-- Let A be the capacity of container A.
variable (A : ℝ)

-- Initial quantities in containers B and C.
def B_initial := 0.375 * A
def C_initial := 0.625 * A

-- Quantities after transferring 158 liters from C to B.
def B_new := B_initial + 158
def C_new := C_initial - 158

theorem relationship_between_B_and_C_after_transfer :
  (B_new / C_new) = ((0.375 * A + 158) / (0.625 * A - 158)) :=
by
  -- We can skip the proof as instructed.
  sorry

end relationship_between_B_and_C_after_transfer_l250_250946


namespace vector_addition_l250_250699

variables {V : Type*} [AddCommGroup V]

variables (A B C : V)
variables (a b : V)

-- Definitions based on given conditions
def AB : V := a
def BC : V := b

-- Theorem statement
theorem vector_addition :
  (AB + BC) = (a + b) :=
by sorry

end vector_addition_l250_250699


namespace rate_of_simple_interest_l250_250561

-- Define the principal amount and time
variables (P : ℝ) (R : ℝ) (T : ℝ := 12)

-- Define the condition that the sum becomes 9/6 of itself in 12 years (T)
def simple_interest_condition (P : ℝ) (R : ℝ) (T : ℝ) : Prop :=
  (9 / 6) * P - P = P * R * T

-- Define the main theorem stating the rate R is 1/24
theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ := 12) (h : simple_interest_condition P R T) : 
  R = 1 / 24 := 
sorry

end rate_of_simple_interest_l250_250561


namespace area_of_transformed_region_l250_250937

open_locale complex_conjugate

-- Define the regular octagon centered at the origin
def octagon_centered_at_origin (z : ℂ) : Prop :=
  abs (z) = 1

-- Define the region R outside the octagon
def region_R (z : ℂ) : Prop :=
  ¬ octagon_centered_at_origin z

-- Define the transformation set T
def set_T : set ℂ :=
  {w | ∃ z ∈ region_R, w = 1 / z}

-- The condition that opposite pairs of sides are two units apart
def octave_side_distance_condition (s : ℝ) : Prop :=
  s = 2

-- Define the area 8 + pi
def area_of_T : ℝ :=
  8 + Real.pi

-- Objective: Prove the area of T is 8 + pi
theorem area_of_transformed_region :
  ∃ s : ℝ, octave_side_distance_condition s ∧
           (∃ R T, T = (1 : ℂ)) --
           ∃ regions : set ℂ, regions = set_T ∧
           (area_of_T = 8 + Real.pi) := sorry

end area_of_transformed_region_l250_250937


namespace curve_eq_circle_l250_250212

theorem curve_eq_circle (r θ : ℝ) : (∀ θ : ℝ, r = 3) ↔ ∃ c : ℝ, c = 0 ∧ ∀ z : ℝ, (r = real.sqrt ((3 - c)^2 + θ^2)) := sorry

end curve_eq_circle_l250_250212


namespace inequality_solution_l250_250532

theorem inequality_solution (a b : ℝ) :
  (∀ x : ℝ, (-1/2 < x ∧ x < 2) → (ax^2 + bx + 2 > 0)) →
  a + b = 1 :=
by
  sorry

end inequality_solution_l250_250532


namespace molecular_weight_of_compound_l250_250504

constant atomic_weight_N : ℝ := 14.01
constant atomic_weight_H : ℝ := 1.01
constant atomic_weight_Br : ℝ := 79.90

-- Define the number of atoms of each type
constant n_N : ℕ := 1
constant n_H : ℕ := 4
constant n_Br : ℕ := 1

-- Define the calculation of molecular weight
def molecular_weight (n_N n_H n_Br : ℕ) (weight_N weight_H weight_Br : ℝ) : ℝ :=
  (n_N * weight_N) + (n_H * weight_H) + (n_Br * weight_Br)

-- State the theorem
theorem molecular_weight_of_compound : molecular_weight n_N n_H n_Br atomic_weight_N atomic_weight_H atomic_weight_Br = 97.95 := by
  sorry

end molecular_weight_of_compound_l250_250504


namespace a_4_value_l250_250660

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then -x^2 + x else 2 * f (x - 1)

def a_n (n : ℕ) [fact (0 < n)] : ℝ :=
(2 : ℝ)^(n - 3)

theorem a_4_value : a_n 4 = 2 := by sorry

end a_4_value_l250_250660


namespace count_integers_in_interval_l250_250732

theorem count_integers_in_interval :
  {n : ℤ | -5 < n ∧ n < 3}.finite ∧ {n : ℤ | -5 < n ∧ n < 3}.to_finset.card = 7 := by
sorry

end count_integers_in_interval_l250_250732


namespace number_of_perfect_square_factors_of_180_l250_250302

theorem number_of_perfect_square_factors_of_180 :
  let prime_factors := (2, 2, 1); 
  let conditions (p1 p2 p3 : ℕ) := 
    p1 ∈ {0, 2} ∧ 
    p2 ∈ {0, 2} ∧ 
    p3 ∈ {0}
  in Σ' x, x ∈ { (p1, p2, p3) | conditions p1 p2 p3 } = 4 :=
begin
  sorry
end

end number_of_perfect_square_factors_of_180_l250_250302


namespace consecutive_integer_sum_l250_250088

theorem consecutive_integer_sum (a b c : ℕ) 
  (h1 : b = a + 2) 
  (h2 : c = a + 4) 
  (h3 : a + c = 140) 
  (h4 : b - a = 2) : a + b + c = 210 := 
sorry

end consecutive_integer_sum_l250_250088


namespace sin_alpha_value_l250_250680

theorem sin_alpha_value (x y : ℝ) (hx : x < 0) (hy : y = -√3 * x) (α : ℝ) :
    sin α = √3 / 2 :=
  sorry

end sin_alpha_value_l250_250680


namespace parallelogram_area_l250_250130

theorem parallelogram_area
  (b : ℝ) (h : ℝ) (s : ℝ) (θ : ℝ)
  (hb : b = 10) (hs : s = 6) (hθ : θ = 30) (h_h : h = s * real.sin (θ * real.pi / 180)) :
  b * h = 30 := by
  sorry

end parallelogram_area_l250_250130


namespace popsicle_total_l250_250777

def popsicle_count (g c b : Nat) : Nat :=
  g + c + b

theorem popsicle_total : 
  let g := 2
  let c := 13
  let b := 2
  popsicle_count g c b = 17 := by
  sorry

end popsicle_total_l250_250777


namespace exists_convex_1990_sided_polygon_with_given_properties_l250_250436

noncomputable def is_polygon (n : ℕ) (sides : list ℕ) (angles : list ℝ) : Prop :=
  sides.length = n ∧ angles.length = n ∧ (∀ θ ∈ angles, θ = (n-2).to_rat / n * real.pi)

def square_side_lengths_up_to (n : ℕ) : list ℕ :=
  list.map (λ m, m ^ 2) (list.range n)

def is_permutation {α : Type*} [decidable_eq α] (l₁ l₂ : list α) : Prop :=
  l₁ ~ l₂

theorem exists_convex_1990_sided_polygon_with_given_properties :
  ∃ (sides : list ℕ) (angles : list ℝ),
    is_polygon 1990 sides angles ∧ is_permutation sides (square_side_lengths_up_to 1990) :=
sorry

end exists_convex_1990_sided_polygon_with_given_properties_l250_250436


namespace exists_equilateral_triangle_in_different_discs_equilateral_triangle_side_length_gt_96_l250_250986

-- Conditions
def is_center_of_disc (x y : ℤ) : Prop :=
  ∃ r : ℝ, r = 1/1000

def points_in_different_discs (p1 p2 p3 : ℤ × ℤ) : Prop :=
  is_center_of_disc p1.1 p1.2 ∧ is_center_of_disc p2.1 p2.2 ∧ is_center_of_disc p3.1 p3.2 ∧
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

def equilateral_triangle (p1 p2 p3 : ℤ × ℤ) : Prop :=
  let d1 := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2
  let d2 := (p2.1 - p3.1)^2 + (p2.2 - p3.2)^2
  let d3 := (p3.1 - p1.1)^2 + (p3.2 - p1.2)^2
  d1 = d2 ∧ d2 = d3

-- Statements
theorem exists_equilateral_triangle_in_different_discs :
  ∃ (p1 p2 p3 : ℤ × ℤ), equilateral_triangle p1 p2 p3 ∧ points_in_different_discs p1 p2 p3 :=
sorry

theorem equilateral_triangle_side_length_gt_96 :
  ∀ (p1 p2 p3 : ℤ × ℤ), equilateral_triangle p1 p2 p3 ∧ points_in_different_discs p1 p2 p3 →
  let d := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  d > 96 :=
sorry

end exists_equilateral_triangle_in_different_discs_equilateral_triangle_side_length_gt_96_l250_250986


namespace cube_volume_accommodates_cone_l250_250005

def cone_height : ℝ := 15
def cone_base_diameter : ℝ := 8
def cube_side : ℝ := cone_height

theorem cube_volume_accommodates_cone :
  let volume := cube_side ^ 3 in
  volume = 3375 := 
by
  sorry

end cube_volume_accommodates_cone_l250_250005


namespace find_special_number_l250_250070

-- Let's define the conditions first
def is_valid_integer (n : ℕ) : Prop :=
  100 < n ∧ n < 1100

def reversed (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  units * 100 + tens * 10 + hundreds

-- Now, let's state the theorem
theorem find_special_number :
  ∃ X : ℕ, (∀ n : ℕ, is_valid_integer n → reversed n = n + X) ∧ X = 99 :=
by
  use 99
  intros n hn
  sorry

end find_special_number_l250_250070


namespace trigonometric_identity_l250_250104

theorem trigonometric_identity :
  (Real.sin (18 * Real.pi / 180) * Real.sin (78 * Real.pi / 180)) -
  (Real.cos (162 * Real.pi / 180) * Real.cos (78 * Real.pi / 180)) = 1 / 2 := by
  sorry

end trigonometric_identity_l250_250104


namespace imaginary_part_of_z_mul_i_l250_250052

-- Define complex number (4 - 8i)
def z : ℂ := 4 - 8 * complex.I

-- Define the multiplication of z with i (which is complex.I in Lean)
def zi : ℂ := z * complex.I

-- Prove that the imaginary part of z * i is equal to 4
theorem imaginary_part_of_z_mul_i : complex.im zi = 4 :=
sorry

end imaginary_part_of_z_mul_i_l250_250052


namespace michael_left_money_l250_250418

def michael_initial_money : Nat := 100
def michael_spent_on_snacks : Nat := 25
def michael_spent_on_rides : Nat := 3 * michael_spent_on_snacks
def michael_spent_on_games : Nat := 15
def total_expenditure : Nat := michael_spent_on_snacks + michael_spent_on_rides + michael_spent_on_games
def michael_money_left : Nat := michael_initial_money - total_expenditure

theorem michael_left_money : michael_money_left = 15 := by
  sorry

end michael_left_money_l250_250418


namespace necessary_not_sufficient_condition_l250_250907

open Function

theorem necessary_not_sufficient_condition (α β : ℝ) : 
  (sin α + cos β = 0) → (sin^2 α + sin^2 β = 1) ∧ (¬(sin^2 α + sin^2 β = 1 → sin α + cos β = 0)) :=
by
  sorry

end necessary_not_sufficient_condition_l250_250907


namespace evaluate_expression_at_x_eq_2_l250_250890

theorem evaluate_expression_at_x_eq_2 :
  (3 * 2 + 4)^2 = 100 := by
  sorry

end evaluate_expression_at_x_eq_2_l250_250890


namespace find_r6_plus_s6_l250_250000

-- Define the variables and the equation
variables (r s : ℝ)

-- Define the condition that r and s are the roots of the given quadratic equation
def roots_condition : Prop :=
  r * r - 2 * r * real.sqrt 5 + 2 = 0 ∧
  s * s - 2 * s * real.sqrt 5 + 2 = 0

-- State the problem as a theorem in Lean
theorem find_r6_plus_s6 (h : roots_condition r s) : r^6 + s^6 = 3904 :=
sorry

end find_r6_plus_s6_l250_250000


namespace find_triangle_angles_and_area_l250_250782

variable (a b c : ℝ) (A B C : ℝ)

def triangle_condition_1 : Prop :=
  b = 1 ∧ c = sqrt 3 ∧ B = Real.pi / 6

def angles_case_1 : Prop :=
  a = 1 → A = Real.pi / 6 ∧ C = 2 * Real.pi / 3
  
def angles_case_2 : Prop :=
  a = 2 → A = Real.pi / 2 ∧ C = Real.pi / 3

def area_case_1 (area : ℝ) : Prop :=
  a = 1 → area = (sqrt 3) / 4

def area_case_2 (area : ℝ) : Prop :=
  a = 2 → area = (sqrt 3) / 2

theorem find_triangle_angles_and_area (area : ℝ) :
  triangle_condition_1 a b c B →
  (angles_case_1 a A C ∧ angles_case_2 a A C) ∧
  (area_case_1 a area ∧ area_case_2 a area) :=
  by
    sorry

end find_triangle_angles_and_area_l250_250782


namespace floor_of_5_point_7_l250_250204

theorem floor_of_5_point_7 : Int.floor 5.7 = 5 := by
  sorry

end floor_of_5_point_7_l250_250204


namespace intersecting_lines_solution_l250_250857

theorem intersecting_lines_solution (a b : ℝ) :
  (∃ (a b : ℝ), 
    ((a^2 + 1) * 2 - 2 * b * (-3) = 4) ∧ 
    ((1 - a) * 2 + b * (-3) = 9)) →
  (a, b) = (4, -5) ∨ (a, b) = (-2, -1) :=
by
  sorry

end intersecting_lines_solution_l250_250857


namespace gibi_percentage_is_59_l250_250770

-- Define the conditions
def max_score := 700
def avg_score := 490
def jigi_percent := 55
def mike_percent := 99
def lizzy_percent := 67

def jigi_score := (jigi_percent * max_score) / 100
def mike_score := (mike_percent * max_score) / 100
def lizzy_score := (lizzy_percent * max_score) / 100

def total_score := 4 * avg_score
def gibi_score := total_score - (jigi_score + mike_score + lizzy_score)

def gibi_percent := (gibi_score * 100) / max_score

-- The proof goal
theorem gibi_percentage_is_59 : gibi_percent = 59 := by
  sorry

end gibi_percentage_is_59_l250_250770


namespace almond_butter_ratio_l250_250422

theorem almond_butter_ratio
  (peanut_cost almond_cost batch_extra almond_per_batch : ℝ)
  (h1 : almond_cost = 3 * peanut_cost)
  (h2 : peanut_cost = 3)
  (h3 : almond_per_batch = batch_extra)
  (h4 : batch_extra = 3) :
  almond_per_batch / almond_cost = 1 / 3 := sorry

end almond_butter_ratio_l250_250422


namespace amount_invested_is_8001_95_l250_250465

def market_value : ℝ := 110.86111111111111
def income : ℝ := 756
def rate : ℝ := 10.5
def brokerage_rate : ℝ := 0.25

theorem amount_invested_is_8001_95 :
  let FV := (income * 100) / rate in
  let actual_price_paid := market_value + (market_value * brokerage_rate) / 100 in
  let AI := (FV / 100) * actual_price_paid in
  AI = 8001.95 :=
by
  unfold FV
  unfold actual_price_paid
  unfold AI
  sorry

end amount_invested_is_8001_95_l250_250465


namespace find_f_3_l250_250821

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : 
  (∀ (x : ℝ), x ≠ 0 → 27 * f (-x) / x - x^2 * f (1 / x) = - 2 * x^2) →
  f 3 = 2 :=
sorry

end find_f_3_l250_250821


namespace num_perfect_square_factors_of_180_l250_250312

theorem num_perfect_square_factors_of_180 (n : ℕ) (h : n = 180) :
  ∃ k : ℕ, k = 4 ∧ ∀ d : ℕ, d ∣ n → ∃ a b c : ℕ, d = 2^a * 3^b * 5^c ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 :=
by
  use 4
  sorry

end num_perfect_square_factors_of_180_l250_250312


namespace probability_calculation_l250_250318

def satisfies_equation (p q : ℤ) : Prop := p * q - 5 * p - 3 * q = 6

def p_in_range (p : ℤ) : Prop := p ≥ 1 ∧ p ≤ 12

def suitable_ps : list ℤ := [2, 4, 6, 10]  -- identified suitable values of p 

def probability : ℚ := 1/3 -- expected probability

theorem probability_calculation :
  (finset.range 13).filter (λ p, (∃ q, satisfies_equation p q)).card.to_rat / 12 = probability := by
  sorry

end probability_calculation_l250_250318


namespace exp_division_rule_l250_250166

-- The theorem to prove the given problem
theorem exp_division_rule (x : ℝ) (hx : x ≠ 0) :
  x^10 / x^5 = x^5 :=
by sorry

end exp_division_rule_l250_250166


namespace Mike_s_Trip_Graph_Accurate_l250_250826

-- Represent Mike's travel narrative using types and properties
inductive Speed
| slow
| moderate
| fast

structure Segment where
  duration : ℝ
  speed : Speed

structure Journey where
  segments : List Segment
  break_time : ℝ

-- Given conditions translated to Lean structures
def MikeJourney := Journey.mk [
  Segment.mk 1 Speed.slow,     -- initial city traffic
  Segment.mk 1 Speed.moderate, -- suburban area (to mall)
  Segment.mk 1 Speed.fast,     -- highway (to mall)
  Segment.mk 1 Speed.moderate, -- suburban area (return trip)
  Segment.mk 1 Speed.fast,     -- highway (return trip)
  Segment.mk 1 Speed.slow      -- final city traffic
] 1.5                           -- shopping and lunch break (1.5 hours)

-- Definition of a graph property
def correctlyRepresents (j : Journey) (g : Type) : Prop :=
  -- This is a placeholder for detailed graph properties matching Mike's journey pattern, 
  -- such as gradual, moderate, and steep slopes, and the horizontal line for mall visit.
  sorry

-- The proposition to be proven
theorem Mike_s_Trip_Graph_Accurate : correctlyRepresents MikeJourney Graph.C :=
  sorry

end Mike_s_Trip_Graph_Accurate_l250_250826


namespace probability_angle_in_quarter_pi_l250_250926

/-- Probability problem involving rolling two fair dice and angles between vectors. -/
theorem probability_angle_in_quarter_pi (m n : ℕ) (h₁ : 1 ≤ m ∧ m ≤ 6) (h₂ : 1 ≤ n ∧ n ≤ 6) :
  let prob := ((finset.card {p : ℕ × ℕ | p.fst > p.snd ∧ 1 ≤ p.fst ∧ p.fst ≤ 6 ∧ 1 ≤ p.snd ∧ p.snd ≤ 6}).to_real / 
               (finset.card {p : ℕ × ℕ | 1 ≤ p.fst ∧ p.fst ≤ 6 ∧ 1 ≤ p.snd ∧ p.snd ≤ 6}).to_real) in
  prob = 5 / 12 :=
by
  sorry

end probability_angle_in_quarter_pi_l250_250926


namespace exist_arithmetic_progression_of_good_numbers_l250_250584

def sum_of_cubes_of_digits (n : ℕ) : ℕ :=
  n.to_digits ℕ 10 |>.map (λ d, d^3) |>.sum

def is_good (n : ℕ) : Prop :=
  ∃ k, (λ f k => Nat.iterate f k n = 1) sum_of_cubes_of_digits k

theorem exist_arithmetic_progression_of_good_numbers :
  ∃ a d, ∀ k, k < 1402 → is_good (a + k * d) :=
by
  sorry

end exist_arithmetic_progression_of_good_numbers_l250_250584


namespace number_of_perfect_square_factors_of_180_l250_250293

-- Define the prime factorization of 180
def prime_factorization_180 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Define what it means to be a factor of 180
def is_factor_of_180 (n : ℕ) : Prop :=
  ∀ p k, (p, k) ∈ prime_factorization_180 → ∃ m, n = p ^ m ∧ m ≤ k

-- Define what it means to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p m, ∃ k, n = p ^ m ∧ even m

-- Function to count factors of 180 that are perfect squares
def count_perfect_square_factors_of_180 : ℕ :=
  List.length [d | d ← List.range 181, is_factor_of_180 d ∧ is_perfect_square d]

-- Main theorem
theorem number_of_perfect_square_factors_of_180 : count_perfect_square_factors_of_180 = 4 := 
by sorry

end number_of_perfect_square_factors_of_180_l250_250293


namespace find_polynomials_l250_250631

-- Define our polynomial P(x)
def polynomial_condition (P : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, (x-1) * P.eval (x+1) - (x+2) * P.eval x = 0

-- State the theorem
theorem find_polynomials (P : Polynomial ℝ) :
  polynomial_condition P ↔ ∃ a : ℝ, P = Polynomial.C a * (Polynomial.X^3 - Polynomial.X) :=
by
  sorry

end find_polynomials_l250_250631


namespace cos_periodicity_even_function_property_l250_250634

theorem cos_periodicity_even_function_property (n : ℤ) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) (h_range : -180 ≤ n ∧ n ≤ 180) : n = 43 :=
by
  sorry

end cos_periodicity_even_function_property_l250_250634


namespace probability_min_diff_at_least_3_l250_250074

theorem probability_min_diff_at_least_3 :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let total_ways := (Finset.card (Finset.powersetLen 3 S) : ℚ) 
  let valid_sets := [({1, 4, 7} : Finset ℕ), ({2, 5, 8} : Finset ℕ), ({3, 6, 9} : Finset ℕ)].length
  (valid_sets / total_ways) = (1 / 28) := by
  sorry

end probability_min_diff_at_least_3_l250_250074


namespace max_abs_sum_sqrt2_l250_250331

theorem max_abs_sum_sqrt2 (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (a : ℝ), (a = |x| + |y| ∧ a ≤ 2 * Real.sqrt 2) ∧ 
             ∀ (x y : ℝ), x^2 + y^2 = 4 → (|x| + |y|) ≤ 2 * Real.sqrt 2 :=
sorry

end max_abs_sum_sqrt2_l250_250331


namespace parallel_lines_slope_l250_250054

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → a = 3 / 2 :=
by
  sorry

end parallel_lines_slope_l250_250054


namespace arithmetic_sequence_sum_l250_250677

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (h_arith : arithmetic_sequence a)
    (h_a2 : a 2 = 3)
    (h_a1_a6 : a 1 + a 6 = 12) : a 7 + a 8 + a 9 = 45 :=
by
  sorry

end arithmetic_sequence_sum_l250_250677


namespace projection_ratio_le_sqrt2_plus_one_l250_250114

theorem projection_ratio_le_sqrt2_plus_one 
  (polygon : Type) 
  (l : polygon → ℝ) 
  (convex : ∀ p1 p2 : polygon, ∃ p : polygon, p = p1 ∨ p = p2) 
  (l_divides_half : ∀ p : polygon, l p = 0 ∨ l p ≥ 0) 
  [metric_space polygon] 
  [normed_space ℝ polygon] 
  (perpendicular_projection : polygon → polygon) 
  (projection_is_divided : ∃ d : ℝ, d ≤ 1 + sqrt 2) :
  d_ratios (perpendicular_projection, polygon, l) ≤ 1 + sqrt 2 :=
sorry

end projection_ratio_le_sqrt2_plus_one_l250_250114


namespace limit_eq_one_third_derivative_l250_250002

noncomputable def lim_expression (f : ℝ → ℝ) (x : ℝ) : ℝ := (f (1 + x) - f 1) / (3 * x)

theorem limit_eq_one_third_derivative (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  tendsto (lim_expression f) (𝓝 0) (𝓝 (1/3 * deriv f 1)) :=
sorry

end limit_eq_one_third_derivative_l250_250002


namespace geometry_problem_l250_250412

theorem geometry_problem
  (A B C D E F P : Point)
  (h_ell : collinear {F, A, B})
  (h_AC : collinear {D, A, C})
  (h_BC : collinear {E, B, C})
  (h_bisector : is_bisector {P, A, C})
  (h_side : F ∉ line_through (A, B))
  (h_CD_CE : dist C D = dist C E)
  (h_interior : P ∈ interior (triangle A B C)) :
  (dist F B * dist F A + dist C P ^ 2 = dist C F ^ 2) ↔ (dist A D * dist B E = dist P D ^ 2) := 
by
  sorry

end geometry_problem_l250_250412


namespace decryption_probability_l250_250879

-- Definitions of the probabilities of individual decryption
def P_A1 : ℝ := 1/5
def P_A2 : ℝ := 1/3
def P_A3 : ℝ := 1/4

-- The independence of the events A1, A2, and A3 is assumed
axiom independent_events : ∀ (P_A1 P_A2 P_A3 : Prop), 
                          independent P_A1 P_A2 → independent P_A1 P_A3 → independent P_A2 P_A3

-- The main theorem stating the probability that the code will be decrypted
theorem decryption_probability : P_C = 3/5 :=
by 
  -- Introduce events and their complements
  let A1 := P_A1
  let A2 := P_A2
  let A3 := P_A3

  let not_A1 := 1 - A1
  let not_A2 := 1 - A2
  let not_A3 := 1 - A3

  -- Because the events are independent
  have : P(not_A1 ∩ not_A2 ∩ not_A3) = (not_A1) * (not_A2) * (not_A3),
    from by apply independent_events A1 A2 A3

  -- Calculate the complementary probability
  let P_D : ℝ := (1 - P_A1) * (1 - P_A2) * (1 - P_A3)
  let P_C : ℝ := 1 - P_D

  -- Simplify and conclude the proof with the correct probability
  sorry

end decryption_probability_l250_250879


namespace platform_length_l250_250943

theorem platform_length (speed_km_hr : ℝ) (time_man : ℝ) (time_platform : ℝ) (L : ℝ) (P : ℝ) :
  speed_km_hr = 54 → time_man = 20 → time_platform = 22 → 
  L = (speed_km_hr * (1000 / 3600)) * time_man →
  L + P = (speed_km_hr * (1000 / 3600)) * time_platform → 
  P = 30 := 
by
  intros hs ht1 ht2 hL hLP
  sorry

end platform_length_l250_250943


namespace Lucas_test_scores_l250_250769

theorem Lucas_test_scores (s1 s2 s3 s4 s5 s6 : ℕ) :
  (s1 = 82) ∧ (s2 = 77) ∧ (s3 = 90) ∧ (s4 = 68) ∧
  (s1 + s2 + s3 + s4 + s5 + s6 = 6 * 79) ∧
  (s1 < 95) ∧ (s2 < 95) ∧ (s3 < 95) ∧ (s4 < 95) ∧ (s5 < 95) ∧ (s6 < 95) ∧
  (s1 ≠ s2) ∧ (s1 ≠ s3) ∧ (s1 ≠ s4) ∧ (s1 ≠ s5) ∧ (s1 ≠ s6) ∧
  (s2 ≠ s3) ∧ (s2 ≠ s4) ∧ (s2 ≠ s5) ∧ (s2 ≠ s6) ∧
  (s3 ≠ s4) ∧ (s3 ≠ s5) ∧ (s3 ≠ s6) ∧
  (s4 ≠ s5) ∧ (s4 ≠ s6) ∧
  (s5 ≠ s6) →
  {s1, s2, s3, s4, s5, s6} = {93, 90, 82, 77, 68, 64} :=
by
  sorry

end Lucas_test_scores_l250_250769


namespace divisor_five_l250_250489

theorem divisor_five {D : ℝ} (h : 95 / D + 23 = 42) : D = 5 := by
  sorry

end divisor_five_l250_250489


namespace positive_square_factors_of_180_l250_250290

theorem positive_square_factors_of_180 :
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  factors = 4 :=
by
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  show factors = 4
  sorry

end positive_square_factors_of_180_l250_250290


namespace floor_5_7_eq_5_l250_250191

theorem floor_5_7_eq_5 : Int.floor 5.7 = 5 := by
  sorry

end floor_5_7_eq_5_l250_250191


namespace simplify_expression_l250_250028

variable (y : ℝ)

theorem simplify_expression : 
    4 * y + 8 * y^2 + 6 - (3 - 4 * y - 8 * y^2) = 16 * y^2 + 8 * y + 3 :=
begin
  sorry
end

end simplify_expression_l250_250028


namespace ratio_simple_to_compound_interest_l250_250063

theorem ratio_simple_to_compound_interest :
  let P₁ := 5250
  let R₁ := 4
  let T₁ := 2
  let SI := P₁ * R₁ * T₁ / 100
  let P₂ := 4000
  let R₂ := 10
  let T₂ := 2
  let CI := P₂ * ((1 + R₂ / 100) ^ T₂ - 1)
  SI / CI = 1 / 4
:= by {
  let P₁ := 5250
  let R₁ := 4
  let T₁ := 2
  let SI := P₁ * R₁ * T₁ / 100
  let P₂ := 4000
  let R₂ := 10
  let T₂ := 2
  let CI := P₂ * ((1 + R₂ / 100) ^ T₂ - 1)
  show SI / CI = 1 / 4, from sorry -- Proof to be filled in
}

end ratio_simple_to_compound_interest_l250_250063


namespace option_A_option_B_option_D_l250_250669

-- Given real numbers a, b, c such that a > b > 1 and c > 0,
-- prove the following inequalities.
variables {a b c : ℝ}

-- Assume the conditions
axiom H1 : a > b
axiom H2 : b > 1
axiom H3 : c > 0

-- Statements to prove
theorem option_A (H1: a > b) (H2: b > 1) (H3: c > 0) : a^2 - bc > b^2 - ac := sorry
theorem option_B (H1: a > b) (H2: b > 1) : a^3 > b^2 := sorry
theorem option_D (H1: a > b) (H2: b > 1) : a + (1/a) > b + (1/b) := sorry
  
end option_A_option_B_option_D_l250_250669


namespace gold_distribution_l250_250778

theorem gold_distribution :
  ∃ (d : ℚ), 
    (4 * (a1: ℚ) + 6 * d = 3) ∧ 
    (3 * (a1: ℚ) + 24 * d = 4) ∧
    d = 7 / 78 :=
by {
  sorry
}

end gold_distribution_l250_250778


namespace partitions_equiv_l250_250803

-- Definition of partitions into distinct integers
def a (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition or count function

-- Definition of partitions into odd integers
def b (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition or count function

-- Theorem stating that the number of partitions into distinct integers equals the number of partitions into odd integers
theorem partitions_equiv (n : ℕ) : a n = b n :=
sorry

end partitions_equiv_l250_250803


namespace AE_eq_BF_l250_250549

variables (O A B C D X Y K L M N E F : Point)
variables (l : Line)

-- Assume A and B are points where line l intersect OX and OY respectively
-- Assume C and D are points on line l such that CA = DB
-- Assume lines passing through C intersects OX at K and OY at L
-- Assume lines passing through D intersects OX at M and OY at N
-- Assume lines ML and KN intersect line l at points E and F.

def intersects (l : Line) (p q : Point) : Prop := p ∈ l ∧ q ∈ l
def collinear (p q r : Point) : Prop := ∃ l : Line, p ∈ l ∧ q ∈ l ∧ r ∈ l

axiom CA_eq_DB : CA = DB
axiom l_intersects_OX : intersects l OX A
axiom l_intersects_OY : intersects l OY B
axiom C_on_l : C ∈ l
axiom D_on_l : D ∈ l
axiom D_opposite_side_C : C ≠ D ∧ collinear O A B ∧ collinear O C D
axiom K_on_C : intersects (Line.mk C K) OX K
axiom L_on_C : intersects (Line.mk C L) OY L
axiom M_on_D : intersects (Line.mk D M) OX M
axiom N_on_D : intersects (Line.mk D N) OY N
axiom E_on_ML : intersects (Line.mk M L) l E
axiom F_on_KN : intersects (Line.mk K N) l F

theorem AE_eq_BF : AE = BF :=
by
  sorry

end AE_eq_BF_l250_250549


namespace erasers_pens_markers_cost_l250_250147

theorem erasers_pens_markers_cost 
  (E P M : ℝ)
  (h₁ : E + 3 * P + 2 * M = 240)
  (h₂ : 2 * E + 4 * M + 5 * P = 440) :
  3 * E + 4 * P + 6 * M = 520 :=
sorry

end erasers_pens_markers_cost_l250_250147


namespace george_worked_hours_l250_250653

-- Define the necessary variables and conditions
variable (M : ℕ) -- Hours George worked on Monday

-- Define conditions
def hourly_wage := 5
def hours_tuesday := 2
def total_earnings := (hourly_wage * M) + (hourly_wage * hours_tuesday)
def total_money := 45

-- The theorem to prove
theorem george_worked_hours : total_earnings = total_money → M = 7 :=
by
  intro h
  have h1 : total_earnings = (hourly_wage * M) + (hourly_wage * hours_tuesday)
      := rfl
  simp at h
  sorry

end george_worked_hours_l250_250653


namespace greatest_two_digit_prime_saturated_l250_250120

-- Definition of prime saturated number
def is_prime_saturated (d : ℕ) : Prop :=
  (∏ p in (nat.factorization d).keys.to_finset, p) < d.sqrt

-- The formal statement of the math proof problem
theorem greatest_two_digit_prime_saturated :
  ∀ d : ℕ, d = 98 → (10 ≤ d ∧ d < 100) → is_prime_saturated d ∧ 
  (∀ d' : ℕ, (10 ≤ d' ∧ d' < 100) → is_prime_saturated d' → d' ≤ 98) :=
by {
  intro d,
  intro h,
  intro h_d_range,
  rw h,
  constructor,
  { sorry }, -- Proof that 98 is prime saturated
  { sorry } -- Proof that there is no greater two-digit prime saturated integer
}

end greatest_two_digit_prime_saturated_l250_250120


namespace sqrt_ab_equals_sqrt_2_l250_250038

theorem sqrt_ab_equals_sqrt_2 
  (a b : ℝ)
  (h1 : a ^ 2 = 16 / 25)
  (h2 : b ^ 3 = 125 / 8) : 
  Real.sqrt (a * b) = Real.sqrt 2 := 
by 
  -- proof will go here
  sorry

end sqrt_ab_equals_sqrt_2_l250_250038


namespace trigonometric_translation_l250_250472

theorem trigonometric_translation (P P' : ℝ → ℝ) (S t : ℝ) (k : ℤ) :
  (P t = 1 ∧ P t = sin (2 * t)) ∧ (P' (t + S) = P t ∧ P' (t + S) = sin (2 * (t + S) - π / 3))
  → (t = k * π + π / 4 ∧ S = π / 6) :=
begin
  sorry,
end

end trigonometric_translation_l250_250472


namespace positive_square_factors_of_180_l250_250288

theorem positive_square_factors_of_180 :
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  factors = 4 :=
by
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  show factors = 4
  sorry

end positive_square_factors_of_180_l250_250288


namespace fifteenth_row_seats_l250_250355

theorem fifteenth_row_seats : 
  ∀ (a_1 d n : ℕ), a_1 = 5 → d = 2 → n = 15 → a_1 + (n - 1) * d = 33 :=
by
  intros a_1 d n h1 h2 h3
  rw [h1, h2, h3]
  sorry

end fifteenth_row_seats_l250_250355


namespace polynomial_solution_l250_250632

theorem polynomial_solution (p : polynomial ℝ) :
  (∀ x, (x - 16) * p.eval (2 * x) = (16 * x - 16) * p.eval x) →
  p = (X - 2) * (X - 4) * (X - 8) * (X - 16) :=
begin
  sorry,
end

end polynomial_solution_l250_250632


namespace barbara_typing_time_l250_250592

theorem barbara_typing_time:
  let original_speed := 212
  let speed_decrease := 40
  let document_length := 3440
  let new_speed := original_speed - speed_decrease
  (new_speed > 0) → 
  (document_length / new_speed = 20) :=
by
  intros
  sorry

end barbara_typing_time_l250_250592


namespace number_of_correct_propositions_l250_250261

theorem number_of_correct_propositions :
  let p_and_q_false_implies_both_false (P Q : Prop) := ¬(P ∧ Q) → (¬P ∧ ¬Q)
  let negation_of_power_inequality (a b : ℝ) := ¬(a > b → 2 ^ a > 2 ^ b - 1) = (a ≤ b → 2 ^ a ≤ 2 ^ b - 1)
  let negation_of_universal_inequality (x : ℝ) := ¬(∀ x : ℝ, x^2 + 1 ≥ 1) = ∃ x : ℝ, x^2 + 1 < 1
  let contrapositive_of_cosine_equality (x y : ℝ) := ¬(cos x = cos y → x = y) → ¬(x ≠ y → cos x ≠ cos y)
  (if p_and_q_false_implies_both_false P Q = false then 0 else if negation_of_power_inequality a b
   then 1 else if negation_of_universal_inequality x then 1 else if contrapositive_of_cosine_equality x y then 0 else 0) = 2 := sorry

end number_of_correct_propositions_l250_250261


namespace opposite_of_pi_l250_250863

theorem opposite_of_pi : -1 * Real.pi = -Real.pi := 
by sorry

end opposite_of_pi_l250_250863


namespace max_abs_sum_of_squares_eq_2_sqrt_2_l250_250324

theorem max_abs_sum_of_squares_eq_2_sqrt_2 (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_of_squares_eq_2_sqrt_2_l250_250324


namespace eccentricity_of_ellipse_l250_250776

-- Define given conditions and the statement to be proven
theorem eccentricity_of_ellipse {F1 F2 P O : Point}
  (dist_F1F2 : |F1 - F2| = 3)
  (dist_O_to_F1P_line : dist_from_point_to_line O (line_through F1 P) = 1) :
  eccentricity_of_ellipse = (sqrt 5 / 3) :=
sorry

end eccentricity_of_ellipse_l250_250776


namespace longest_flight_time_l250_250648

theorem longest_flight_time (V₀ g : ℝ) (hV₀ : V₀ = 10) (hg : g = 10) :
  ∃ τ : ℝ, τ = 1.6 ∧
    ∀ α : ℝ, 
      let l := (V₀^2 * sin (2 * α)) / g in
      (l ≥ 0.96 * (V₀^2 * sin (2 * (π/4))) / g) → 
      let t := (2 * V₀ * sin α) / g in
      t ≤ τ := 
sorry

end longest_flight_time_l250_250648


namespace count_valid_n_l250_250643

def s2 (n : ℕ) : ℕ := (Nat.toDigits 2 n).sum

theorem count_valid_n (N : ℕ := 500) :
  {n : ℕ | n > 0 ∧ n ≤ N ∧ s2(n) ≤ 2}.toFinset.card = 44 := by
  sorry

end count_valid_n_l250_250643


namespace area_of_triangle_DEF_l250_250759

variables {DE EF DF : ℝ}
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def herons_formula (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c in 
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_DEF (h1 : DE = 26) (h2 : EF = 26) (h3 : DF = 40) : 
  herons_formula DE EF DF = 332 :=
by
  /- Proof is omitted -/
  sorry

end area_of_triangle_DEF_l250_250759


namespace monotonically_increasing_interval_l250_250466

def f (x : ℝ) := -6 / x - 5 * Real.log x

theorem monotonically_increasing_interval : 
  ∀ x, 0 < x ∧ x < (6 / 5 : ℝ) → monotone_increasing_on (f : ℝ → ℝ) (Ioo 0 (6 / 5)) :=
begin
  sorry
end

end monotonically_increasing_interval_l250_250466


namespace sum_of_cubes_divisible_l250_250435

theorem sum_of_cubes_divisible (a b c : ℤ) (h : (a + b + c) % 3 = 0) : 
  (a^3 + b^3 + c^3) % 3 = 0 := 
by sorry

end sum_of_cubes_divisible_l250_250435


namespace negate_forall_implies_exists_l250_250057

theorem negate_forall_implies_exists {x : ℝ} :
  (¬ ∀ x ∈ set.Ioo 0 1, x ^ 2 - x < 0) ↔ (∃ x ∈ set.Ioo 0 1, x ^ 2 - x ≥ 0) :=
by
  sorry

end negate_forall_implies_exists_l250_250057


namespace factorize_xcube_minus_x_l250_250995

theorem factorize_xcube_minus_x (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by 
  sorry

end factorize_xcube_minus_x_l250_250995


namespace number_of_digits_of_9944_l250_250084

theorem number_of_digits_of_9944 : (number_of_digits 9944 = 4) := 
begin
  -- Here we add the essential mathematical content
  sorry
end

end number_of_digits_of_9944_l250_250084


namespace domain_of_g_l250_250183

def g (x : ℝ) : ℝ := 1 / ((x - 2)^2 + (x + 2)^2 + 1)

theorem domain_of_g : ∀ x: ℝ, ∃ y: ℝ, g x = y :=
by 
  intro x
  use g x
  trivial

end domain_of_g_l250_250183


namespace find_4_oplus_2_l250_250802

def operation (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem find_4_oplus_2 : operation 4 2 = 26 :=
by
  sorry

end find_4_oplus_2_l250_250802


namespace distance_between_points_on_sphere_l250_250255

noncomputable def sphere_distance (R : ℝ) (θ : ℝ) : ℝ :=
  2 * R^2 * (1 - Real.cos θ)

theorem distance_between_points_on_sphere (R : ℝ) 
  (hR_pos : 0 < R) 
  (A B : ℝ × ℝ × ℝ) 
  (spherical_dist : ℝ → ℝ → ℝ → ℝ := fun R θ φ => R * θ)
  (h : spherical_dist R (π / 3) 0 = π * R / 3) 
: Real.sqrt (sphere_distance R (π / 3)) = R :=
by
  have eqn : sphere_distance R (π / 3) = R^2 :=
    by
      calc
        sphere_distance R (π / 3)
          = 2 * R^2 * (1 - Real.cos (π / 3)) : rfl
      ... = 2 * R^2 * (1 - 1 / 2) : by rw [Real.cos_pi_div_three]
      ... = 2 * R^2 * (1 / 2) : by ring
      ... = R^2 : by ring
  rw [eqn]
  exact Real.sqrt_sq hR_pos


end distance_between_points_on_sphere_l250_250255


namespace number_of_perfect_square_factors_of_180_l250_250299

theorem number_of_perfect_square_factors_of_180 :
  let prime_factors := (2, 2, 1); 
  let conditions (p1 p2 p3 : ℕ) := 
    p1 ∈ {0, 2} ∧ 
    p2 ∈ {0, 2} ∧ 
    p3 ∈ {0}
  in Σ' x, x ∈ { (p1, p2, p3) | conditions p1 p2 p3 } = 4 :=
begin
  sorry
end

end number_of_perfect_square_factors_of_180_l250_250299


namespace curve_is_circle_l250_250211

theorem curve_is_circle (r : ℝ) (θ : ℝ) (h : r = 3) : 
  ∃ (c : ℝ) (p : ℝ × ℝ), c = 3 ∧ p = (3 * Real.cos θ, 3 * Real.sin θ) := 
sorry

end curve_is_circle_l250_250211


namespace problem_l250_250404

noncomputable def x : ℂ := complex.exp (complex.I * (2 * real.pi / 7))

theorem problem : ((2 * x + x^2) * (2 * x^2 + x^4) * (2 * x^3 + x^6) * (2 * x^4 + x^8) * (2 * x^5 + x^10) * (2 * x^6 + x^12)) = 43 := 
by {
  sorry
}

end problem_l250_250404


namespace typing_time_l250_250586

-- Definitions based on the problem conditions
def initial_typing_speed : ℕ := 212
def speed_decrease : ℕ := 40
def words_in_document : ℕ := 3440

-- Definition for Barbara's new typing speed
def new_typing_speed : ℕ := initial_typing_speed - speed_decrease

-- Lean proof statement: Proving the time to finish typing is 20 minutes
theorem typing_time :
  (words_in_document / new_typing_speed) = 20 :=
by sorry

end typing_time_l250_250586


namespace count_distinct_products_div_by_10_l250_250284

def distinct_products_div_by_10 (S : Set ℕ) (n : ℕ) : Prop :=
  S = {2, 3, 5, 7, 9} ∧ n = 8

theorem count_distinct_products_div_by_10 : ∃ n, distinct_products_div_by_10 {2, 3, 5, 7, 9} n :=
begin
  use 8,
  -- Proof to be provided
  sorry
end

end count_distinct_products_div_by_10_l250_250284


namespace product_value_l250_250749

theorem product_value (x : ℝ) (h : sqrt (8 + x) + sqrt (27 - x) = 9) : (8 + x) * (27 - x) = 529 :=
sorry

end product_value_l250_250749


namespace probability_pairs_less_than_5000_l250_250482

def city := ℕ

def distance (a b : city) : ℕ :=
  if (a, b) = (0, 1) then 3635 else
  if (a, b) = (0, 2) then 9957 else
  if (a, b) = (0, 3) then 6743 else
  if (a, b) = (1, 2) then 10550 else
  if (a, b) = (1, 3) then 6065 else
  if (a, b) = (2, 3) then 4844 else
  0

def pairs := [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def num_pairs_less_than_5000 : ℕ := 
  pairs.count (λ p, distance p.1 p.2 < 5000)

theorem probability_pairs_less_than_5000 :
  (num_pairs_less_than_5000 : ℚ) / (pairs.length : ℚ) = 1 / 3 :=
by
  sorry

end probability_pairs_less_than_5000_l250_250482


namespace find_second_interest_rate_l250_250565

/-- A total investment of $5,400 where $3,000 is invested at 8% interest rate.
The question is to prove that the second interest rate is equal to 0.10 --/
theorem find_second_interest_rate (total_investment : ℝ) (invested_at_8_percent : ℝ) 
    (second_investment : ℝ) (second_rate : ℝ) :
  total_investment = 5400 →
  invested_at_8_percent = 3000 →
  second_investment = total_investment - invested_at_8_percent →
  invested_at_8_percent * 0.08 = second_investment * second_rate →
  second_rate = 0.10 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  linarith
  sorry

end find_second_interest_rate_l250_250565


namespace geom_seq_42_l250_250231

variable {α : Type*} [Field α] [CharZero α]

noncomputable def a_n (n : ℕ) (a1 q : α) : α := a1 * q ^ n

theorem geom_seq_42 (a1 q : α) (h1 : a1 = 3) (h2 : a1 * (1 + q^2 + q^4) = 21) :
  a1 * (q^2 + q^4 + q^6) = 42 := 
by
  sorry

end geom_seq_42_l250_250231


namespace fraction_addition_target_l250_250089

open Rat

theorem fraction_addition_target (n : ℤ) : 
  (4 + n) / (7 + n) = 3 / 4 → 
  n = 5 := 
by
  intro h
  sorry

end fraction_addition_target_l250_250089


namespace percentage_passed_in_both_subjects_l250_250097

variable (F_H F_E F_HE P_HE : ℝ)

-- Defining the conditions
def F_H_condition : Prop := F_H = 0.20
def F_E_condition : Prop := F_E = 0.70
def F_HE_condition : Prop := F_HE = 0.10

-- Defining what we want to prove
def P_HE_correct : Prop := P_HE = 1.00 - (F_H + F_E - F_HE)

theorem percentage_passed_in_both_subjects (h1 : F_H_condition) (h2 : F_E_condition) (h3 : F_HE_condition) : P_HE_correct :=
sorry

end percentage_passed_in_both_subjects_l250_250097


namespace man_and_son_work_together_l250_250116

-- Define the rates at which the man and his son can complete the work
def man_work_rate := 1 / 5
def son_work_rate := 1 / 20

-- Define the combined work rate when they work together
def combined_work_rate := man_work_rate + son_work_rate

-- Define the total time taken to complete the work together
def days_to_complete_together := 1 / combined_work_rate

-- The theorem stating that they will complete the work in 4 days
theorem man_and_son_work_together : days_to_complete_together = 4 := by
  sorry

end man_and_son_work_together_l250_250116


namespace tan_alpha_eq_neg_5_div_12_l250_250670

noncomputable def α : ℝ :=
  sorry

def sin_α : ℝ := 5 / 13

def α_in_second_quadrant : Prop := α ∈ Set.Ioo (π / 2) π

theorem tan_alpha_eq_neg_5_div_12 (h1 : sin α = sin_α) (h2 : α_in_second_quadrant) :
  tan α = -5 / 12 :=
sorry

end tan_alpha_eq_neg_5_div_12_l250_250670


namespace rectangles_in_3x3_grid_l250_250176

theorem rectangles_in_3x3_grid : 
  let n := 3 in (finset.card ((finset.range n.succ).powerset.filter (λ s, s.card = 2))) ^ 2 = 9 :=
by
  sorry

end rectangles_in_3x3_grid_l250_250176


namespace daily_egg_count_per_female_emu_l250_250389

noncomputable def emus_per_pen : ℕ := 6
noncomputable def pens : ℕ := 4
noncomputable def total_eggs_per_week : ℕ := 84

theorem daily_egg_count_per_female_emu :
  (total_eggs_per_week / ((pens * emus_per_pen) / 2 * 7) = 1) :=
by
  sorry

end daily_egg_count_per_female_emu_l250_250389


namespace vehicle_refuel_probability_l250_250860

def truck_to_car_ratio := (3, 2)

def P_B1 := 3 / (3 + 2)  -- Probabilty of a truck approaching gas station
def P_B2 := 2 / (3 + 2)  -- Probabilty of a car approaching gas station

def P_A_B1 := 1 / 30  -- Conditional probability that a truck refuels
def P_A_B2 := 1 / 22.5 -- Conditional probability that a car refuels

def P_A := P_B1 * P_A_B1 + P_B2 * P_A_B2 -- Total probability calculation

theorem vehicle_refuel_probability :
  P_A = 0.0378 := by 
  sorry

end vehicle_refuel_probability_l250_250860


namespace count_two_digit_numbers_l250_250742

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_sum_perfect_square_or_prime (ds : ℕ) : Prop :=
  ds ∈ {1, 4, 9, 16, 2, 3, 5, 7, 11, 13, 17}

theorem count_two_digit_numbers :
  (∑ n in (Finset.range 90).filter (λ n, is_two_digit n ∧ is_sum_perfect_square_or_prime (digit_sum n)), 1) = 41 := 
sorry

end count_two_digit_numbers_l250_250742


namespace max_abs_sum_sqrt2_l250_250330

theorem max_abs_sum_sqrt2 (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (a : ℝ), (a = |x| + |y| ∧ a ≤ 2 * Real.sqrt 2) ∧ 
             ∀ (x y : ℝ), x^2 + y^2 = 4 → (|x| + |y|) ≤ 2 * Real.sqrt 2 :=
sorry

end max_abs_sum_sqrt2_l250_250330


namespace find_natural_number_l250_250207

theorem find_natural_number (n : ℕ) (k : ℤ) (h : 2^n + 3 = k^2) : n = 0 :=
sorry

end find_natural_number_l250_250207


namespace touch_all_on_l250_250351

-- Button state is either on or off
inductive ButtonState
| on
| off

-- Definition for a matrix of ButtonStates
def ButtonGrid (m n : ℕ) := array (array ButtonState n) m

-- Function to toggle the state of a button
def toggle (state : ButtonState) : ButtonState :=
  match state with
  | ButtonState.on => ButtonState.off
  | ButtonState.off => ButtonState.on

-- Function to toggle a button and its row and column
def touchButton (grid : ButtonGrid 40 50) (i j : ℕ) : ButtonGrid 40 50 :=
  array.mapIdx (λ rowIdx row =>
    if rowIdx == i then
      row.mapIdx (λ colIdx state => toggle state)
    else
      row.mapIdx (λ colIdx state => if colIdx == j then toggle state else state)
  ) grid

-- The initial state of the grid is all off
def initialGrid : ButtonGrid 40 50 :=
  array.mk (fun _ => array.mk (fun _ => ButtonState.off))

-- The target state of the grid is all on
def targetGrid : ButtonGrid 40 50 :=
  array.mk (fun _ => array.mk (fun _ => ButtonState.on))

-- Proof Problem:
-- Prove that the initial grid can be transformed to the target grid with exactly 40 * 50 touches
theorem touch_all_on :
  ∃ touches : list (ℕ × ℕ),
    (touches.length = 40 * 50) ∧
    (touches.foldl (λ grid (i, j) => touchButton grid i j) initialGrid = targetGrid) :=
sorry

end touch_all_on_l250_250351


namespace how_many_integers_satisfy_l250_250713

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end how_many_integers_satisfy_l250_250713


namespace count_solutions_cos2x_plus_3sin2x_eq_1_l250_250743

open Real

theorem count_solutions_cos2x_plus_3sin2x_eq_1 :
  ∀ x : ℝ, (-10 < x ∧ x < 45 → cos x ^ 2 + 3 * sin x ^ 2 = 1) → 
  ∃! n : ℕ, n = 18 := 
by
  intro x hEq
  sorry

end count_solutions_cos2x_plus_3sin2x_eq_1_l250_250743


namespace a_n_formula_Tn_formula_l250_250479

variables (a b : ℕ → ℝ)

-- Condition: Sum of first 5 terms S_5 = 45
axiom S5_eq_45 : (5 * a 1 + 10 * (a 2 - a 1)) = 45

-- Condition: Sum of first 6 terms S_6 = 60
axiom S6_eq_60 : (6 * a 1 + 15 * (a 2 - a 1)) = 60

-- Condition: Relationship between b and a
axiom bn_relation : ∀ n : ℕ, b (n + 1) - b n = a n

-- Condition: Initial value of b
axiom b1_eq_3 : b 1 = 3

-- Main proofs as Lean 4 statements
theorem a_n_formula : a = (λ n, 2 * n + 3) :=
by {
  -- Placeholder implementation
  sorry
}

theorem Tn_formula : ∀ n : ℕ, ∑ i in finset.range n, 1 / b i = (3 / 4 - 1 / (2 * (n + 1)) - 1 / (2 * (n + 2))) :=
by {
  -- Placeholder implementation
  sorry
}

end a_n_formula_Tn_formula_l250_250479


namespace shift_gives_f_l250_250075

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem shift_gives_f :
  (∀ x, f x = g (x + Real.pi / 3)) :=
  by
  sorry

end shift_gives_f_l250_250075


namespace count_eq_expression_l250_250738

theorem count_eq_expression (x : ℝ) (hx : x > 0) :
  let exprs := [2 * x^x, x^(x + 1), (x + 1)^x, 3 * x^x]
  in exprs.count (λ y, y = x^x + x^x) = 1 :=
sorry

end count_eq_expression_l250_250738


namespace smallest_n_contains_234_l250_250453

def decimal_seq_234 (x : ℚ) : Prop :=
  let str_repr := x.to_decimal 200 
  (∃ a b c : ℕ, str_repr[a] == '2' ∧ str_repr[a+1] == '3' ∧ str_repr[a+2] == '4')

theorem smallest_n_contains_234 :
  ∃ (n : ℕ), (∀ m : ℕ, m < n ∧ Nat.coprime m n → decimal_seq_234 (m / n)) ∧ n = 127 :=
by
  sorry

end smallest_n_contains_234_l250_250453


namespace commutative_op_l250_250820

variable {S : Type} (op : S → S → S)

-- Conditions
axiom cond1 : ∀ (a b : S), op a (op a b) = b
axiom cond2 : ∀ (a b : S), op (op a b) b = a

-- Proof problem statement
theorem commutative_op : ∀ (a b : S), op a b = op b a :=
by
  intros a b
  sorry

end commutative_op_l250_250820


namespace truckloads_per_mile_l250_250938

-- Define the given conditions
def total_miles : ℕ := 16
def miles_paved_day1 : ℕ := 4
def miles_paved_day2 : ℕ := 2 * miles_paved_day1 - 1
def remaining_miles : ℕ := total_miles - (miles_paved_day1 + miles_paved_day2)
def barrels_pitch_day3 : ℕ := 6
def barrels_per_truckload : ℕ := 1
def truckloads_needed : ℕ := barrels_pitch_day3 -- since each truckload uses 1 barrel of pitch

-- Question: How many truckloads of asphalt are needed to pave each mile of road?
theorem truckloads_per_mile : truckloads_needed / remaining_miles = 1.2 :=
by
  sorry

end truckloads_per_mile_l250_250938


namespace exists_hamiltonian_cycle_l250_250159

theorem exists_hamiltonian_cycle (P : Polyhedron) (F : Face P) (h : ∀ (G : Face P), G ≠ F → ∃ e ∈ edges F, e ∈ edges G) : 
  ∃ cycle : List (Vertex P), is_simple_closed_polygon cycle ∧ ∀ v : Vertex P, v ∈ cycle :=
sorry

end exists_hamiltonian_cycle_l250_250159


namespace sin_A_angle_B_and_projection_l250_250344

noncomputable def m (A B : ℝ) : ℝ × ℝ :=
  (Real.cos (A - B), Real.sin (A - B))

noncomputable def n (B : ℝ) : ℝ × ℝ :=
  (Real.cos B, -Real.sin B)

axiom m_dot_n (A B : ℝ) (h : m A B • n B = -3/5) : Real.cos A = -3/5

theorem sin_A (A B : ℝ) (h : m A B • n B = -3/5) : Real.sin A = 4/5 :=
by
    have cosA : Real.cos A = -3/5 := m_dot_n A B h
    have h1 : 1 - cosA ^ 2 = 1 - (-3/5) ^ 2 := rfl
    rw [Real.sin_eq_sqrt_one_sub_cos_sq]
    rw [h1]
    norm_num
    done

theorem angle_B_and_projection (a b : ℝ) (c : ℝ) (A B : ℝ)
  (ha : a = 4 * Real.sqrt 2) (hb : b = 5)
  (cos_A : Real.cos A = -3/5) (sin_A : Real.sin A = 4/5)
  (hb_sine_law : Real.sin B = b * sin_A / a)
  (h_B_cosine_law : c = 1) : 
  B = Real.pi / 4 ∧ c * Real.cos B = Real.sqrt 2 / 2 :=
by
  rw [ha, hb, sin_A]
  have sinB : Real.sin B = Real.sqrt 2 / 2 := by
    rw [hb_sine_law]
    norm_num
    done
  have B_eq : B = Real.pi / 4 := by
    sorry
  have proj : c * Real.cos B = Real.sqrt 2 / 2 := by
    rw [h_B_cosine_law]
    sorry
  exact ⟨B_eq, proj⟩
  done

end sin_A_angle_B_and_projection_l250_250344


namespace coexistent_pair_example_coexistent_pair_neg_coexistent_pair_find_a_l250_250423

section coexistent_rational_number_pairs

-- Definitions based on the problem conditions:
def coexistent_pair (a b : ℚ) : Prop := a - b = a * b + 1

-- Proof problem 1
theorem coexistent_pair_example : coexistent_pair 3 (1/2) :=
sorry

-- Proof problem 2
theorem coexistent_pair_neg (m n : ℚ) (h : coexistent_pair m n) :
  coexistent_pair (-n) (-m) :=
sorry

-- Proof problem 3
example : ∃ (p q : ℚ), coexistent_pair p q ∧ (p, q) ≠ (2, 1/3) ∧ (p, q) ≠ (5, 2/3) ∧ (p, q) ≠ (3, 1/2) :=
sorry

-- Proof problem 4
theorem coexistent_pair_find_a (a : ℚ) (h : coexistent_pair a 3) :
  a = -2 :=
sorry

end coexistent_rational_number_pairs

end coexistent_pair_example_coexistent_pair_neg_coexistent_pair_find_a_l250_250423


namespace arcsin_one_eq_pi_div_two_l250_250172

theorem arcsin_one_eq_pi_div_two :
  arcsin 1 = π / 2 :=
by
  have h : sin (π / 2) = 1 := sorry
  exact arcsin_eq_pi_div_two_of_sin_pi_div_two_eq_one h

end arcsin_one_eq_pi_div_two_l250_250172


namespace bacon_percentage_l250_250599

theorem bacon_percentage (total_calories : ℕ) (bacon_calories : ℕ) (strips_of_bacon : ℕ) :
  total_calories = 1250 →
  bacon_calories = 125 →
  strips_of_bacon = 2 →
  (strips_of_bacon * bacon_calories * 100 / total_calories) = 20 :=
by sorry

end bacon_percentage_l250_250599


namespace downstream_speed_is_28_l250_250551

-- Define the speed of the man in still water
def speed_in_still_water : ℝ := 24

-- Define the speed of the man rowing upstream
def speed_upstream : ℝ := 20

-- Define the speed of the stream
def speed_stream : ℝ := speed_in_still_water - speed_upstream

-- Define the speed of the man rowing downstream
def speed_downstream : ℝ := speed_in_still_water + speed_stream

-- The main theorem stating that the speed of the man rowing downstream is 28 kmph
theorem downstream_speed_is_28 : speed_downstream = 28 := by
  sorry

end downstream_speed_is_28_l250_250551


namespace regression_prediction_l250_250233

-- Define the linear regression model as a function
def linear_regression (x : ℝ) : ℝ :=
  7.19 * x + 73.93

-- State that using this model, the predicted height at age 10 is approximately 145.83
theorem regression_prediction :
  abs (linear_regression 10 - 145.83) < 0.01 :=
by 
  sorry

end regression_prediction_l250_250233


namespace find_k_l250_250051

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

theorem find_k (x_0 : ℝ) (k : ℕ) 
  (h_intersect : f x_0 = 0)
  (h_interval : x_0 ∈ (k : ℝ) + 1) 
  (h_f2 : f 2 < 0)
  (h_f3 : f 3 > 0) : k = 2 := 
sorry

end find_k_l250_250051


namespace factorize_cubic_l250_250997

theorem factorize_cubic (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end factorize_cubic_l250_250997


namespace find_uv_l250_250974

def mat_eqn (u v : ℝ) : Prop :=
  (3 + 8 * u = -3 * v) ∧ (-1 - 6 * u = 1 + 4 * v)

theorem find_uv : ∃ (u v : ℝ), mat_eqn u v ∧ u = -6/7 ∧ v = 5/7 := 
by
  sorry

end find_uv_l250_250974


namespace trajectory_of_P_l250_250240

-- Define points F1 and F2
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define the condition |PF2| - |PF1| = 4 for a moving point P
def condition (P : ℝ × ℝ) : Prop :=
  let PF1 := Real.sqrt ((P.1 + 4)^2 + P.2^2)
  let PF2 := Real.sqrt ((P.1 - 4)^2 + P.2^2)
  abs (PF2 - PF1) = 4

-- The target equation of the trajectory
def target_eq (P : ℝ × ℝ) : Prop :=
  P.1 * P.1 / 4 - P.2 * P.2 / 12 = 1 ∧ P.1 ≤ -2

theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, condition P → target_eq P := by
  sorry

end trajectory_of_P_l250_250240


namespace one_can_be_first_l250_250954

theorem one_can_be_first (n : ℕ) (h : n = 1993) (perm : List (Fin n)) :
  ∃ (steps : ℕ), (perform_transformations perm steps).head = 1 :=
sorry

end one_can_be_first_l250_250954


namespace triangle_inequalities_l250_250017

theorem triangle_inequalities (A B C M : Point) (a b c S : ℝ)
  (d_a d_b d_c h_a h_b h_c : ℝ)
  (h_tri : is_triangle A B C)
  (h_point : inside_triangle M A B C)
  (h_altitudes : min (h_a, h_b, h_c) ≤ d_a + d_b + d_c)
  (h_max_min : d_a + d_b + d_c ≤ max (h_a, h_b, h_c))
  (h_area : 2 * S = a * d_a + b * d_b + c * d_c)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0):
  (min (h_a, h_b, h_c) ≤ d_a + d_b + d_c ∧ d_a + d_b + d_c ≤ max (h_a, h_b, h_c))
  ∧
  (d_a * d_b * d_c ≤ (8 * S^3) / (27 * a * b * c)) := 
sorry

end triangle_inequalities_l250_250017


namespace exists_lucky_set_l250_250794

theorem exists_lucky_set (n : ℕ) (hn : n > 1) :
  ∃ (a : Fin n → ℕ), (∀ i j, i ≠ j → (a i ≠ a j)) ∧ (a 0 + a 1 + ... + (a (n - 1)) < n * 2^n) :=
sorry

end exists_lucky_set_l250_250794


namespace how_many_integers_satisfy_l250_250711

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end how_many_integers_satisfy_l250_250711


namespace range_of_a_l250_250675

noncomputable def f : ℝ → ℝ := sorry -- This is a placeholder for the even function f

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

def even_function (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ), f (-x) = f x

def condition1 : Prop :=
even_function f ∧ is_decreasing_on_nonneg f

def condition2 (a : ℝ) : Prop :=
∀ (x : ℝ), 0 < x → x ≤ real.sqrt 2 → f (-a*x + x^3 + 1) + f (a*x - x^3 - 1) ≥ 2 * f 1

theorem range_of_a (a : ℝ) :
  condition1 ∧ condition2 a → 2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l250_250675


namespace perfect_square_factors_of_180_l250_250308

theorem perfect_square_factors_of_180 {n : ℕ} (h1 : n = 180) : 
  (∃ f : ℕ → ℕ, (∀ p, (p = 2 ∨ p = 3 ∨ p = 5 → 
    (f p = 0 ∨ (p = 2 ∨ p = 3) ∧ f p = 2 ∧ p ≠ 5))) ∧ 2^(f 2) * 3^(f 3) * 5^(f 5) = n ∧  ((f 2 + f 3 + f 5) % 2 = 0)) → 
  4 :=
begin
  sorry
end

end perfect_square_factors_of_180_l250_250308


namespace solve_system_eqns_l250_250442

theorem solve_system_eqns :
  ∀ x y z : ℝ, 
  (x * y + 5 * y * z - 6 * x * z = -2 * z) ∧
  (2 * x * y + 9 * y * z - 9 * x * z = -12 * z) ∧
  (y * z - 2 * x * z = 6 * z) →
  x = -2 ∧ y = 2 ∧ z = 1 / 6 ∨
  y = 0 ∧ z = 0 ∨
  x = 0 ∧ z = 0 :=
by
  sorry

end solve_system_eqns_l250_250442


namespace volume_of_region_l250_250640

noncomputable def f (x y z : ℝ) : ℝ :=
|x + y + z| + |x + y - z| + |x - y + z| + |-x + y - z|

theorem volume_of_region : 
  ∀ x y z : ℝ, f x y z ≤ 6 ↔ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x ≤ 1.5 ∧ y ≤ 1.5 ∧ z ≤ 1.5 ∧ x + y + z ≤ 3 → 
  volume_of_region f 22.5 :=
begin
  cases h,
  sorry
end

end volume_of_region_l250_250640


namespace length_broken_line_LAM_area_quadrilateral_KLMN_l250_250363

variables {K L M N A : Type}
variables (KLMN : ConvexQuadrilateral K L M N)
variables (KM_perpendicular_MN LN_perpendicular_KL : ∀ {P Q R}, RightAngle (P Q R))
variables (KN_length : (distance K N) = 4 * Real.sqrt 3)
variables (angle_LAK_eq_angle_MAN : ∀ {A K L M}, (angle A K L) = (angle M A N))
variables (angle_MKN_eq_angle_KNL : (angle M K N) - (angle K N L) = 15 * (Math.pi / 180))
variables (LA_AM_ratio : ∀ {A M L}, (distance L A) / (distance A M) = 1 / (Real.sqrt 3))

theorem length_broken_line_LAM :
  (distance L A) + (distance A M) = Real.sqrt 6 * (Real.sqrt 3 + 1) := sorry

theorem area_quadrilateral_KLMN :
  area K L M N = 9 + 3 * Real.sqrt 3 := sorry

end length_broken_line_LAM_area_quadrilateral_KLMN_l250_250363


namespace find_beta_l250_250272

variables {m n p : ℤ} -- defining variables m, n, p as integers
variables {α β : ℤ} -- defining roots α and β as integers

theorem find_beta (h1: α = 3)
  (h2: ∀ x, x^2 - (m+n)*x + (m*n - p) = 0) -- defining the quadratic equation
  (h3: α + β = m + n)
  (h4: α * β = m * n - p)
  (h5: m ≠ n) (h6: n ≠ p) (h7: m ≠ p) : -- ensuring m, n, and p are distinct
  β = m + n - 3 := sorry

end find_beta_l250_250272


namespace monotonically_increasing_on_neg_infty_to_zero_l250_250575

-- Define the functions
def f_A (x : ℝ) : ℝ := - |x|
def f_B (x : ℝ) : ℝ := x^2 - 2
def f_C (x : ℝ) : ℝ := - (x - 1)
def f_D (x : ℝ) : ℝ := - x

-- Prove that f_A is monotonically increasing on (-∞, 0]
theorem monotonically_increasing_on_neg_infty_to_zero : 
  ∀ x y : ℝ, x ∈ set.Iic 0 → y ∈ set.Iic 0 → x ≤ y → f_A x ≤ f_A y :=
sorry

end monotonically_increasing_on_neg_infty_to_zero_l250_250575


namespace geometric_series_sum_l250_250603

theorem geometric_series_sum :
  (∑ k in finset.range 6, (1:ℚ) / (4 ^ (k + 1))) = 4 / 3 :=
by
  sorry

end geometric_series_sum_l250_250603


namespace tim_cantaloupes_l250_250223

theorem tim_cantaloupes (fred_cantaloupes : ℕ) (total_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : total_cantaloupes = 82) : total_cantaloupes - fred_cantaloupes = 44 :=
by {
  -- proof steps go here
  sorry
}

end tim_cantaloupes_l250_250223


namespace total_tickets_l250_250941

theorem total_tickets (O B : ℕ) (h1 : 12 * O + 8 * B = 3320) (h2 : B = O + 90) : O + B = 350 := by
  sorry

end total_tickets_l250_250941


namespace frustum_intersection_area_is_nine_l250_250039

-- Define the areas of the top and bottom bases
def topBaseArea : ℝ := 1
def bottomBaseArea : ℝ := 16

-- Define the height ratios
def heightRatio : ℝ := 2

-- Define the cut area function
def cutArea (A_top A_bottom : ℝ) (ratio : ℝ) : ℝ :=
  ((Math.sqrt A_top + ratio * Math.sqrt A_bottom) / (1 + ratio)) ^ 2

-- Main theorem: the area of the intersection is 9 given the conditions
theorem frustum_intersection_area_is_nine :
  cutArea topBaseArea bottomBaseArea heightRatio = 9 :=
  sorry

end frustum_intersection_area_is_nine_l250_250039


namespace fraction_equivalence_l250_250627

theorem fraction_equivalence :
  ( (3 / 7 + 2 / 3) / (5 / 11 + 3 / 8) ) = (119 / 90) :=
by
  sorry

end fraction_equivalence_l250_250627


namespace min_positive_period_pi_monotonic_decreasing_on_interval_l250_250690

noncomputable def f (x : ℝ) : ℝ := Math.sin (2 * x) + Real.sqrt 3 * Math.cos (2 * x) + 1

theorem min_positive_period_pi : ∀ x, f (x + π) = f x := sorry

theorem monotonic_decreasing_on_interval : ∀ x, x ∈ Ioo (π / 6) (π / 2) → f' x < 0 :=
sorry

end min_positive_period_pi_monotonic_decreasing_on_interval_l250_250690


namespace typing_time_l250_250588

-- Definitions based on the problem conditions
def initial_typing_speed : ℕ := 212
def speed_decrease : ℕ := 40
def words_in_document : ℕ := 3440

-- Definition for Barbara's new typing speed
def new_typing_speed : ℕ := initial_typing_speed - speed_decrease

-- Lean proof statement: Proving the time to finish typing is 20 minutes
theorem typing_time :
  (words_in_document / new_typing_speed) = 20 :=
by sorry

end typing_time_l250_250588


namespace floor_5_7_l250_250197

theorem floor_5_7 : Int.floor 5.7 = 5 :=
by
  sorry

end floor_5_7_l250_250197


namespace DF_perp_FG_l250_250366

variables {A B C M D E F G N : Type} 
variables (triangle_ABC : Triangle A B C)
variables [AcuteTriangle triangle_ABC] (H1 : B < AC)
variables (M : Midpoint B C)
variables (D : MidpointArc BAC)
variables (E : MidpointArc BC)
variables (F : IncircleTouchpoint A B C AB)
variables (G : Intersection AE BC)
variables (N : (segment EF : Line A B) → Perpendicular NB AB)

theorem DF_perp_FG (triangle_ABC : Triangle A B C)
  [AcuteTriangle triangle_ABC]
  (H1 : B < AC)
  (M : Midpoint B C)
  (D : MidpointArc BAC)
  (E : MidpointArc BC)
  (F : IncircleTouchpoint A B C AB)
  (G : Intersection AE BC)
  (N : (segment EF : Line A B) → Perpendicular NB AB):
  Perpendicular DF FG :=
by
  sorry

end DF_perp_FG_l250_250366


namespace area_of_rectangle_l250_250555

theorem area_of_rectangle (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 90) : w * l = 379.6875 :=
by
  sorry

end area_of_rectangle_l250_250555


namespace difference_value_l250_250454

theorem difference_value (x : ℝ) (h : x = -10.0) : 2 * x - (-8) = -12.0 :=
by {
  rw h,
  norm_num,
  sorry
}

end difference_value_l250_250454


namespace sum_x_components_of_solutions_l250_250003

theorem sum_x_components_of_solutions :
  let solutions := { (x : ℂ, y : ℂ, z : ℂ) | x + y * z = 9 ∧ y + x * z = 13 ∧ z + x * y = 12 } in
  (∑ p in solutions, p.1) = -31 :=
by
  -- proof to be added
  sorry

end sum_x_components_of_solutions_l250_250003


namespace standard_equation_of_ellipse_OT_bisects_PQ_minimized_TF_PQ_l250_250665

-- Definitions of the given conditions
def is_ellipse (C : ℝ → ℝ → Prop) (a b : ℝ) : Prop :=
∀ (x y : ℝ), C x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)

def focal_length (a b : ℝ) : ℝ := 2 * real.sqrt (a^2 - b^2)

def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
dist A B = dist B C ∧ dist B C = dist C A

-- Initial ellipses C and its conditions
variable {C : ℝ → ℝ → Prop} {a b : ℝ}
variable (cond1 : is_ellipse C a b)
variable (cond2 : focal_length a b = 4)
variable (cond3 : equilateral_triangle (0, b) (0, -b) (a, 0))

-- Questions proofs
theorem standard_equation_of_ellipse :
  C = (λ x y, x^2 / 6 + y^2 / 2 = 1) :=
sorry

variables (F : ℝ × ℝ) (T : ℝ × ℝ)
variable (cond4 : F = (-2, 0))
variable (cond5 : T = (-3, _))
variable (OT : ℝ × ℝ → ℝ × ℝ → Prop)
variable (PQ : ℝ × ℝ → ℝ × ℝ → Prop)
variable (O : ℝ × ℝ := (0, 0))

theorem OT_bisects_PQ :
  OT O T → PQ T _
:= sorry

theorem minimized_TF_PQ :
  ∃ t, T = (-3, t) ∧ t ∈ {-1, 1} :=
sorry

end standard_equation_of_ellipse_OT_bisects_PQ_minimized_TF_PQ_l250_250665


namespace monotonically_increasing_interval_l250_250467

def f (x : ℝ) := -6 / x - 5 * Real.log x

theorem monotonically_increasing_interval : 
  ∀ x, 0 < x ∧ x < (6 / 5 : ℝ) → monotone_increasing_on (f : ℝ → ℝ) (Ioo 0 (6 / 5)) :=
begin
  sorry
end

end monotonically_increasing_interval_l250_250467


namespace semicircular_window_perimeter_correct_l250_250939

noncomputable def semicircular_window_perimeter (diameter : ℝ) : ℝ :=
  (real.pi * diameter) / 2 + diameter

theorem semicircular_window_perimeter_correct :
  semicircular_window_perimeter 63 ≈ 162.12 :=
by
  sorry

end semicircular_window_perimeter_correct_l250_250939


namespace expansion_properties_l250_250684

theorem expansion_properties (n : ℕ) (r : ℕ) :
  ((∑ i in range n, binom n i * (2 : ℕ) ^ (n - i) * (3 : ℕ) ^ i * x ^ (n - i - (2 / 3) * i)) =
    (2 : ℕ) ^ n * (3 / x^( (2) / 3))^n * x^n) →
  (binom n 2 / binom n 1 = 5 / 2 → n = 6) ∧
  (∃ r : ℕ, 6 - (4 / 3) * r = 2 ∧ binom 6 r * 2^(6 - r) * 3^r = 4320) ∧
  ( ∃ r : ℕ,
    r = 4 ∧
    (binom 6 r * 2 ^ (6 - r) * 3^r ≥ binom 6 (r - 1) * 2 ^ (6 - (r - 1)) * 3 ^ (r - 1)) ∧
    binom 6 r * 2 ^ (6 - r) * 3^r ≥ binom 6 (r + 1) * 2 ^ (6 - (r + 1)) * 3 ^ (r + 1) ∧
    binom 6 r * 2 ^ (6 - r) * 3^r * x ^ (6 - (4/3) * r) = 4860 * x ^ (2/3))
  sorry

end expansion_properties_l250_250684


namespace barbara_typing_time_l250_250594

theorem barbara_typing_time:
  let original_speed := 212
  let speed_decrease := 40
  let document_length := 3440
  let new_speed := original_speed - speed_decrease
  (new_speed > 0) → 
  (document_length / new_speed = 20) :=
by
  intros
  sorry

end barbara_typing_time_l250_250594


namespace area_difference_l250_250171

theorem area_difference (X Y Z : Type) [metric_space X] [metric_space Y] [metric_space Z]
  (rX rY rZ : ℝ) (hX : rX = 1) (hY : rY = 1) (hZ : rZ = 2)
  (tangent_XY : ∃ p : X, ∀ q ∈ X, q ≠ p → dist p q = 2 * rX)
  (tangent_XZ : ∃ p : X, ∀ q ∈ Z, q ≠ p → dist p q = 2 * rZ)
  (tangent_YZ : ∃ p : Y, ∀ q ∈ Z, q ≠ p → dist p q = 2 * rZ) :
  let A := π * rZ^2 in
  let B := 0 in
  A - B = 4 * π := 
begin
  sorry
end

end area_difference_l250_250171


namespace largest_divisor_of_consecutive_even_product_l250_250396

theorem largest_divisor_of_consecutive_even_product :
  ∀ (n : ℕ), ∃ k : ℤ, k = 24 ∧ 
  (2 * n) * (2 * n + 2) * (2 * n + 4) % k = 0 :=
by
  sorry

end largest_divisor_of_consecutive_even_product_l250_250396


namespace card_arrangement_l250_250071

theorem card_arrangement :
  ∀ (A B C D E F : Type), ∃ (arrangements : Finset (Fin 6 → Fin 6)),
  (∀ (a : Fin 6 → Fin 6), a ∈ arrangements → (a 0 = A) ∧ (a 5 = F)) ∧ arrangements.card = 24 :=
by
  sorry

end card_arrangement_l250_250071


namespace count_integers_n_satisfying_inequality_l250_250714

theorem count_integers_n_satisfying_inequality :
  {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset.card = 7 :=
by sorry

end count_integers_n_satisfying_inequality_l250_250714


namespace mark_friend_contribution_l250_250417

theorem mark_friend_contribution:
  let F := 40 - 30 in -- this represents Mark's friend's contribution
  (0.20 * 200 = 40) ∧ (F = 40 - 30) ∧ (F + 30 = 40) → F = 10 := by
sorry

end mark_friend_contribution_l250_250417


namespace complex_calculation_l250_250601

theorem complex_calculation :
  (7 - 3 * Complex.i) - 3 * (2 + 4 * Complex.i) + 2 * Complex.i * (3 - 5 * Complex.i) = 11 - 9 * Complex.i :=
by
  sorry

end complex_calculation_l250_250601


namespace conjugate_of_product_l250_250259

open Complex

theorem conjugate_of_product : 
  let z := (1 + Complex.i) * (2 - Complex.i) in
  conj z = 3 - Complex.i :=
by
  let z := (1 + Complex.i) * (2 - Complex.i)
  show conj z = 3 - Complex.i
  sorry

end conjugate_of_product_l250_250259


namespace x_add_inv_ge_two_l250_250433

theorem x_add_inv_ge_two (x : ℝ) (hx : x > 0) : x + (1 / x) ≥ 2 :=
by
  sorry

end x_add_inv_ge_two_l250_250433


namespace find_a4_plus_a6_l250_250678

variable {a : ℕ → ℝ}

-- Geometric sequence definition
def is_geometric_seq (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Conditions for the problem
axiom seq_geometric : is_geometric_seq a
axiom seq_positive : ∀ n : ℕ, n > 0 → a n > 0
axiom given_equation : a 3 * a 5 + 2 * a 4 * a 6 + a 5 * a 7 = 81

-- The problem to prove
theorem find_a4_plus_a6 : a 4 + a 6 = 9 :=
sorry

end find_a4_plus_a6_l250_250678


namespace max_min_f_when_m_eq_1_interval_of_monotonic_increasing_l250_250811

-- Defining the function f(x) with the parameter m
def f (m : ℝ) (x : ℝ) : ℝ := -((1:ℝ)/3) * x^3 + x^2 + (m^2 - 1) * x

-- First problem: proving the maximum and minimum values when m = 1
theorem max_min_f_when_m_eq_1 :
  ∃ (x_max x_min : ℝ), x_max = -3 ∧ x_min = 0 ∧ 
  ∀ x ∈ Set.Icc (-3 : ℝ) 2, f 1 x ≤ 18 ∧ f 1 x ≥ 0 :=
sorry

-- Second problem: proving the interval of monotonic increase
theorem interval_of_monotonic_increasing (m: ℝ) (hm : 0 < m) :
  ∀ (x : ℝ), 1 - m < x ∧ x < m + 1 → 
  ∀ y ∈ {y | f m y ≤ f m (y + 1)}, monotone_on (f m) (Set.Ioo (1 - m) (m + 1)) :=
sorry

end max_min_f_when_m_eq_1_interval_of_monotonic_increasing_l250_250811


namespace count_integers_in_interval_l250_250735

theorem count_integers_in_interval :
  {n : ℤ | -5 < n ∧ n < 3}.finite ∧ {n : ℤ | -5 < n ∧ n < 3}.to_finset.card = 7 := by
sorry

end count_integers_in_interval_l250_250735


namespace probability_two_red_balls_l250_250917

theorem probability_two_red_balls (R B G : ℕ) (hR : R = 5) (hB : B = 6) (hG : G = 4) :
  ((R * (R - 1)) / ((R + B + G) * (R + B + G - 1)) : ℚ) = 2 / 21 :=
by
  rw [hR, hB, hG]
  norm_num
  sorry

end probability_two_red_balls_l250_250917


namespace fifteenth_prime_is_correct_l250_250980

-- Definitions
def fifth_prime := 11
def fifteenth_prime := 47

-- The statement to be proven
theorem fifteenth_prime_is_correct : nat.prime fifteenth_prime ∧ fifteenth_prime = 47 :=
by
  sorry

end fifteenth_prime_is_correct_l250_250980


namespace count_scalene_triangles_natural_sides_l250_250578

theorem count_scalene_triangles_natural_sides :
  let scalene_triangles := { S : ℕ × ℕ × ℕ | 
    let a := S.1, 
        b := S.2.1, 
        c := S.2.2 in 
    a < b ∧ b < c ∧ 
    a + c = 2 * b ∧ 
    a + b + c ≤ 30 ∧ 
    a > 0 ∧ b > 0 ∧ c > 0 } in
  scalene_triangles.finite.count = 20 :=
by 
  sorry

end count_scalene_triangles_natural_sides_l250_250578


namespace range_of_a_l250_250757

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≥ 0) → -√3 ≤ a ∧ a ≤ √3 := 
by {
  sorry
}

end range_of_a_l250_250757


namespace triangle_ABC_is_acute_l250_250780

theorem triangle_ABC_is_acute (A B C : ℝ) (a b c : ℝ) 
  (h1: a^2 + b^2 >= c^2) (h2: b^2 + c^2 >= a^2) (h3: c^2 + a^2 >= b^2)
  (h4: (Real.sin A + Real.sin B) / (Real.sin B + Real.sin C) = 9 / 11)
  (h5: (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 11 / 10) : 
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2 :=
sorry

end triangle_ABC_is_acute_l250_250780


namespace sum_underlined_positive_l250_250533

-- Definitions for conditions
def cond1 (nums : List ℤ) : Prop := nums.length = 100

def cond2 (num : ℤ) : Prop := num > 0 → num ∈ nums

def sum_with_next_positive (nums : List ℤ) : List ℤ :=
  nums.zip nums.tail |>.filter (fun (x : ℤ × ℤ) => x.fst + x.snd > 0) |>.map Prod.fst

def sum_with_next_2_positive (nums : List ℤ) : List ℤ :=
  nums.zip (nums.tail.zip nums.tail.tail) |>.filter (fun (x : ℤ × (ℤ × ℤ)) => x.fst + x.snd.fst + x.snd.snd > 0) |>.map Prod.fst

-- Theorem statement
theorem sum_underlined_positive (nums : List ℤ) (h1 : cond1 nums) (h2 : ∀ n, cond2 nums n) :
  (sum_with_next_positive nums ++ sum_with_next_2_positive nums).sum > 0 :=
by
  sorry

end sum_underlined_positive_l250_250533


namespace highest_score_is_174_l250_250099

noncomputable def batting_average (total_runs : ℕ) (innings : ℕ) : ℕ :=
  total_runs / innings

variable (H L : ℕ)

def condition1 := H - L = 140
def condition2 := H + L = 208
def total_runs := 60 * 46
def total_runs_excl_H_L := 58 * 44

theorem highest_score_is_174
  (h₁ : condition1)
  (h₂ : condition2)
  (tr : total_runs = 2760)
  (tr_excl : total_runs_excl_H_L = 2552) :
  H = 174 :=
by
  sorry

end highest_score_is_174_l250_250099


namespace arithmetic_seq_a7_l250_250361

structure arith_seq (a : ℕ → ℤ) : Prop :=
  (step : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)

theorem arithmetic_seq_a7
  {a : ℕ → ℤ}
  (h_seq : arith_seq a)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 10)
  : a 7 = 8 :=
by
  sorry

end arithmetic_seq_a7_l250_250361


namespace focus_of_parabola_l250_250452

theorem focus_of_parabola (m : ℝ) (m_nonzero : m ≠ 0) :
    ∃ (focus_x focus_y : ℝ), (focus_x, focus_y) = (m, 0) ∧
        ∀ (y : ℝ), (x = 1/(4*m) * y^2) := 
sorry

end focus_of_parabola_l250_250452


namespace money_left_eq_l250_250379

theorem money_left_eq :
  let initial_money := 56
  let notebooks := 7
  let cost_per_notebook := 4
  let books := 2
  let cost_per_book := 7
  let money_left := initial_money - (notebooks * cost_per_notebook + books * cost_per_book)
  money_left = 14 :=
by
  sorry

end money_left_eq_l250_250379


namespace matrix_own_inverse_l250_250185

theorem matrix_own_inverse : 
  ∃ (x y : ℚ), (∀ x y, 
  (λ M : Matrix (Fin 2) (Fin 2) ℚ, M * M = 1 → M = Matrix.of ![![4, -2], ![x, y]])) :=
begin
  use (15/2),
  use (-4),
  sorry
end

end matrix_own_inverse_l250_250185


namespace transform_A_correct_transform_D_correct_l250_250880

def sin_transform_A (x : ℝ) : ℝ := Real.sin (2 * (x - (Real.pi / 5)))
def sin_transform_D (x : ℝ) : ℝ := Real.sin (2 * (x - (Real.pi / 10)))

theorem transform_A_correct : ∀ (x : ℝ), sin_transform_A x = Real.sin (2 * x - (Real.pi / 5)) := 
by sorry

theorem transform_D_correct : ∀ (x : ℝ), sin_transform_D x = Real.sin (2 * x - (Real.pi / 5)) := 
by sorry

end transform_A_correct_transform_D_correct_l250_250880


namespace barbara_typing_time_l250_250593

theorem barbara_typing_time:
  let original_speed := 212
  let speed_decrease := 40
  let document_length := 3440
  let new_speed := original_speed - speed_decrease
  (new_speed > 0) → 
  (document_length / new_speed = 20) :=
by
  intros
  sorry

end barbara_typing_time_l250_250593


namespace sum_and_product_of_roots_l250_250628

theorem sum_and_product_of_roots (a b : ℝ) (h1 : a * a * a - 4 * a * a - a + 4 = 0)
  (h2 : b * b * b - 4 * b * b - b + 4 = 0) :
  a + b + a * b = -1 :=
sorry

end sum_and_product_of_roots_l250_250628


namespace total_brushing_time_in_hours_l250_250006

-- Define the conditions as Lean definitions
def brushing_duration : ℕ := 2   -- 2 minutes per brushing session
def brushing_times_per_day : ℕ := 3  -- brushes 3 times a day
def days : ℕ := 30  -- for 30 days

-- Define the calculation of total brushing time in hours
theorem total_brushing_time_in_hours : (brushing_duration * brushing_times_per_day * days) / 60 = 3 := 
by 
  -- Sorry to skip the proof
  sorry

end total_brushing_time_in_hours_l250_250006


namespace fish_population_after_changes_l250_250550

-- Definitions based on conditions
def initial_salmon_population : ℕ := 500
def initial_halibut_population : ℕ := 800
def initial_trout_population : ℕ := 700

def salmon_increase_factor : ℕ := 10
def salmon_loss_per_150 : ℕ := 25
def halibut_loss_percent : ℕ := 10
def trout_loss_percent : ℕ := 5

-- Final populations to prove
theorem fish_population_after_changes :
  ∃ (final_salmon_population final_halibut_population final_trout_population : ℕ),
    final_salmon_population = 4175 ∧
    final_halibut_population = 720 ∧
    final_trout_population = 665 :=
by {
  -- Calculating salmon population after migration
  let migrated_salmon_population := initial_salmon_population * salmon_increase_factor,
  let groups_of_150 := migrated_salmon_population / 150,
  let salmon_losses := groups_of_150 * salmon_loss_per_150,
  let final_salmon_population := migrated_salmon_population - salmon_losses,

  -- Calculating halibut population after environmental changes
  let halibut_losses := initial_halibut_population * halibut_loss_percent / 100,
  let final_halibut_population := initial_halibut_population - halibut_losses,

  -- Calculating trout population after environmental changes
  let trout_losses := initial_trout_population * trout_loss_percent / 100,
  let final_trout_population := initial_trout_population - trout_losses,

  -- Ensuring consistency with the given answers
  exact ⟨4175, 720, 665,
    by {
      unfold migrated_salmon_population groups_of_150 salmon_losses halibut_losses trout_losses,
      exact final_salmon_population = 4175
    },
    by {
      exact final_halibut_population = 720
    },
    by {
      exact final_trout_population = 665
    }
  ⟩
}

end fish_population_after_changes_l250_250550


namespace sum_of_digits_greatest_prime_factor_15999_l250_250976

theorem sum_of_digits_greatest_prime_factor_15999 : 
  ∃ p : ℕ, prime p ∧ p ∣ 15999 ∧ ∀ q : ℕ, prime q ∧ q ∣ 15999 → q ≤ p ∧ 
  (p.digits 10).sum = 17 := 
sorry

end sum_of_digits_greatest_prime_factor_15999_l250_250976


namespace triangle_area_relation_l250_250139

variables {Area : Type} [HasSqrt Area] [Mul Area]

-- Define the areas of the triangles
variables (a b c : Area)

-- We will define the property of being inscribed and parallel sides as a condition.
-- Since we don't have specific methods to describe geometric configurations directly,
-- we use a hypothesis representing the proportional area relationship derived from
-- parallel sides and similarity.
axiom inscribed_and_parallel
  (T_a T_b T_c : Type)
  (area_T_a : T_a → Area)
  (area_T_b : T_b → Area)
  (area_T_c : T_c → Area) :
  (area_T_a T_a = a) →
  (area_T_c T_c = c) →
  (a = b * sqrt a) →
  (b = sqrt (a * c))

theorem triangle_area_relation
  (T_a T_b T_c : Type)
  (area_T_a : T_a → Area)
  (area_T_b : T_b → Area)
  (area_T_c : T_c → Area) :
  (area_T_a T_a = a) →
  (area_T_c T_c = c) →
  b = sqrt (a * c) :=
by
  intros h1 h2
  exact inscribed_and_parallel T_a T_b T_c area_T_a area_T_b area_T_c h1 h2 bfl
#exit

end triangle_area_relation_l250_250139


namespace part_1_part_2_l250_250264

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x * Real.log x - m * x
noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + 1 - m
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - x - a * x^3
noncomputable def r (x : ℝ) : ℝ := (Real.log x - 1) / x^2
noncomputable def r' (x : ℝ) : ℝ := (3 - 2 * Real.log x) / x^3

theorem part_1 (x : ℝ) (m : ℝ) (h1 : f x m = -1) (h2 : f' x m = 0) :
  m = 1 ∧ (∀ y, y > 0 → y < x → f' y 1 < 0) ∧ (∀ y, y > x → f' y 1 > 0) :=
sorry

theorem part_2 (a : ℝ) :
  (a > 1 / (2 * Real.exp 3) → ∀ x, h x a ≠ 0) ∧
  (a ≤ 0 ∨ a = 1 / (2 * Real.exp 3) → ∃ x, h x a = 0 ∧ ∀ y, h y a = 0 → y = x) ∧
  (0 < a ∧ a < 1 / (2 * Real.exp 3) → ∃ x1 x2, x1 ≠ x2 ∧ h x1 a = 0 ∧ h x2 a = 0) :=
sorry

end part_1_part_2_l250_250264


namespace bacon_percentage_l250_250598

theorem bacon_percentage (total_calories : ℕ) (bacon_calories : ℕ) (strips_of_bacon : ℕ) :
  total_calories = 1250 →
  bacon_calories = 125 →
  strips_of_bacon = 2 →
  (strips_of_bacon * bacon_calories * 100 / total_calories) = 20 :=
by sorry

end bacon_percentage_l250_250598


namespace probability_ending_at_multiple_of_four_l250_250374

theorem probability_ending_at_multiple_of_four :
  let spinner_outcomes := ["move 2 spaces left", "move 2 spaces right", "move 1 space left", "move 1 space right"]
  let probabilities := [1/4, 1/4, 1/4, 1/4]
  let probability_starting_multiple_of_4 := 4 / 12
  let probability_ending_multiple_of_4_given_start := (1 / 4)
  (probability_starting_multiple_of_4 * probability_ending_multiple_of_4_given_start) = 1 / 12 :=
by {
  let spinner_outcomes := ["move 2 spaces left", "move 2 spaces right", "move 1 space left", "move 1 space right"],
  let probabilities := [1/4, 1/4, 1/4, 1/4],
  let probability_starting_multiple_of_4 := 4 / 12,
  let probability_ending_multiple_of_4_given_start := 1 / 4,
  show (probability_starting_multiple_of_4 * probability_ending_multiple_of_4_given_start) = 1 / 12,
  sorry
}

end probability_ending_at_multiple_of_four_l250_250374


namespace nurses_count_l250_250100

theorem nurses_count (total : ℕ) (ratio_doc : ℕ) (ratio_nurse : ℕ) (nurses : ℕ) : 
  total = 200 → 
  ratio_doc = 4 → 
  ratio_nurse = 6 → 
  nurses = (ratio_nurse * total / (ratio_doc + ratio_nurse)) → 
  nurses = 120 := 
by 
  intros h_total h_ratio_doc h_ratio_nurse h_calc
  rw [h_total, h_ratio_doc, h_ratio_nurse] at h_calc
  simp at h_calc
  exact h_calc

end nurses_count_l250_250100


namespace smallest_non_representable_number_l250_250217

theorem smallest_non_representable_number :
  ∀ n : ℕ, (∀ a b c d : ℕ, n = (2^a - 2^b) / (2^c - 2^d) → n < 11) ∧
           (∀ a b c d : ℕ, 11 ≠ (2^a - 2^b) / (2^c - 2^d)) :=
sorry

end smallest_non_representable_number_l250_250217


namespace smallest_four_digit_number_conditions_l250_250087

theorem smallest_four_digit_number_conditions :
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ (n % 9 = 0) ∧ 
           ((let digits := List.ofDigit n in
             (List.length digits = 4 ∧
              (2 = (List.filter (λ d, d % 2 = 0) digits).length) ∧
              (2 = (List.filter (λ d, d % 2 = 1) digits).length)))) ∧
           (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ (m % 9 = 0) ∧ 
              ((let digits_m := List.ofDigit m in 
                (List.length digits_m = 4 ∧
                 (2 = (List.filter (λ d, d % 2 = 0) digits_m).length) ∧
                 (2 = (List.filter (λ d, d % 2 = 1) digits_m).length))) → n ≤ m)) :=
  ∃ n, n = 1089
sorry

end smallest_four_digit_number_conditions_l250_250087


namespace max_abs_sum_of_squares_eq_2_sqrt_2_l250_250323

theorem max_abs_sum_of_squares_eq_2_sqrt_2 (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_of_squares_eq_2_sqrt_2_l250_250323


namespace find_f_minus_2_l250_250806

namespace MathProof

def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 5

theorem find_f_minus_2 (a b c : ℝ) (h : f a b c 2 = 3) : f a b c (-2) = -13 := 
by
  sorry

end MathProof

end find_f_minus_2_l250_250806


namespace prob_insurance_A_or_B_prob_exactly_one_no_insurance_out_of_three_l250_250161

noncomputable def P_A : ℝ := 0.5
noncomputable def P_B_not_A : ℝ := 0.3
noncomputable def P_B : ℝ := 0.6  -- given from solution step
noncomputable def P_C : ℝ := 1 - (1 - P_A) * (1 - P_B)
noncomputable def P_D : ℝ := (1 - P_A) * (1 - P_B)
noncomputable def P_E : ℝ := 3 * P_D * (P_C ^ 2)

theorem prob_insurance_A_or_B :
  P_C = 0.8 :=
by
  sorry

theorem prob_exactly_one_no_insurance_out_of_three :
  P_E = 0.384 :=
by
  sorry

end prob_insurance_A_or_B_prob_exactly_one_no_insurance_out_of_three_l250_250161


namespace range_of_m_l250_250041

theorem range_of_m :
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → (x < -1 ∨ x > 3)) ↔ (0 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l250_250041


namespace library_visitors_total_l250_250788

theorem library_visitors_total
  (visitors_monday : ℕ)
  (visitors_tuesday : ℕ)
  (average_visitors_remaining_days : ℕ)
  (remaining_days : ℕ)
  (total_visitors : ℕ)
  (hmonday : visitors_monday = 50)
  (htuesday : visitors_tuesday = 2 * visitors_monday)
  (haverage : average_visitors_remaining_days = 20)
  (hremaining_days : remaining_days = 5)
  (htotal : total_visitors =
    visitors_monday + visitors_tuesday + remaining_days * average_visitors_remaining_days) :
  total_visitors = 250 :=
by
  -- here goes the proof, marked as sorry for now
  sorry

end library_visitors_total_l250_250788


namespace mike_picked_peaches_l250_250419

def initial_peaches : ℕ := 34
def total_peaches : ℕ := 86

theorem mike_picked_peaches : total_peaches - initial_peaches = 52 :=
by
  sorry

end mike_picked_peaches_l250_250419


namespace inequality_proof_l250_250525

theorem inequality_proof (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
    (h_sum : a + b + c + d = 8) :
    (a^3 / (a^2 + b + c) + b^3 / (b^2 + c + d) + c^3 / (c^2 + d + a) + d^3 / (d^2 + a + b)) ≥ 4 :=
by
  sorry

end inequality_proof_l250_250525


namespace find_missing_pair_sum_l250_250828

theorem find_missing_pair_sum (a b c d : ℕ) (h : a ≤ b ∧ b ≤ c ∧ c ≤ d) 
  (sums : multiset ℕ)
  (hs : sums = {a + b, a + c, a + d, b + c, b + d, c + d})
  (known_sums : multiset ℕ)
  (hks : known_sums = {270, 360, 390, 500, 620}) :
  ∃ x : ℕ, x ∉ known_sums ∧ x ∈ sums ∧ x = 530 :=
by
  sorry

end find_missing_pair_sum_l250_250828


namespace div_by_9_implies_not_div_by_9_l250_250915

/-- If 9 divides 10^n + 1, then it also divides 10^(n+1) + 1 -/
theorem div_by_9_implies:
  ∀ n: ℕ, (9 ∣ (10^n + 1)) → (9 ∣ (10^(n + 1) + 1)) :=
by
  intro n
  intro h
  sorry

/-- 9 does not divide 10^1 + 1 -/
theorem not_div_by_9:
  ¬(9 ∣ (10^1 + 1)) :=
by 
  sorry

end div_by_9_implies_not_div_by_9_l250_250915


namespace joe_money_left_l250_250382

theorem joe_money_left (starting_amount : ℕ) (num_notebooks : ℕ) (cost_per_notebook : ℕ) (num_books : ℕ) (cost_per_book : ℕ)
  (h_starting_amount : starting_amount = 56)
  (h_num_notebooks : num_notebooks = 7)
  (h_cost_per_notebook : cost_per_notebook = 4)
  (h_num_books : num_books = 2)
  (h_cost_per_book : cost_per_book = 7) : 
  starting_amount - (num_notebooks * cost_per_notebook + num_books * cost_per_book) = 14 :=
by
  rw [h_starting_amount, h_num_notebooks, h_cost_per_notebook, h_num_books, h_cost_per_book]
  -- sorry lõpetab ajutiselt
  norm_num  
  -- sorry 

end joe_money_left_l250_250382


namespace analogy_for_parallelepiped_l250_250091

theorem analogy_for_parallelepiped (F : Type) (figures : F → Prop) (triangle parallelogram trapezoid rectangle : F) 
    (hp : figures parallelepiped) 
    (ht : figures triangle) 
    (hpp : figures parallelogram) 
    (htr : figures trapezoid) 
    (hr : figures rectangle) 
    : parallelogram = most_suitable_figure ↔ parallelogram = parallelogram :=
by
  sorry

end analogy_for_parallelepiped_l250_250091


namespace greatest_whole_number_satisfying_inequality_l250_250214

theorem greatest_whole_number_satisfying_inequality :
  ∃ x : ℕ, (∀ y : ℕ, y < 1 → y ≤ x) ∧ 4 * x - 3 < 2 - x :=
sorry

end greatest_whole_number_satisfying_inequality_l250_250214


namespace max_min_values_monotonic_increasing_interval_l250_250808

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  - (1 / 3) * x^3 + x^2 + (m^2 - 1) * x

theorem max_min_values (m : ℝ) (h : m = 1) :
  (∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m ≤ 18) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m = 18) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m ≥ 0) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m = 0) := sorry

theorem monotonic_increasing_interval (m : ℝ) (h : m > 0) :
  ∀ x ∈ Set.Ioo (1 - m : ℝ) (m + 1 : ℝ), 0 < deriv (λ x, f x m) x := sorry

end max_min_values_monotonic_increasing_interval_l250_250808


namespace stars_per_classmate_is_correct_l250_250026

-- Define the given conditions
def total_stars : ℕ := 45
def num_classmates : ℕ := 9

-- Define the expected number of stars per classmate
def stars_per_classmate : ℕ := 5

-- Prove that the number of stars per classmate is 5 given the conditions
theorem stars_per_classmate_is_correct :
  total_stars / num_classmates = stars_per_classmate :=
sorry

end stars_per_classmate_is_correct_l250_250026


namespace quadratic_function_expression_l250_250662

theorem quadratic_function_expression :
  ∃ (f : ℝ → ℝ), (f 4 = 3) ∧ (f 1 = 0) ∧ (f 3 = 0) ∧ (∀ x : ℝ, f (2 - x) = f (2 + x)) ∧ (f = λ x, x^2 - 4 * x + 3) :=
by
  sorry

end quadratic_function_expression_l250_250662


namespace money_left_eq_l250_250377

theorem money_left_eq :
  let initial_money := 56
  let notebooks := 7
  let cost_per_notebook := 4
  let books := 2
  let cost_per_book := 7
  let money_left := initial_money - (notebooks * cost_per_notebook + books * cost_per_book)
  money_left = 14 :=
by
  sorry

end money_left_eq_l250_250377


namespace sum_reciprocal_S_l250_250823

def euler_totient (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ k, Nat.coprime k n).card

def S (n : ℕ) : Prop :=
  2 * n % (euler_totient n) = 0

noncomputable def reciprocal_sum (s : Finset ℕ) : ℚ :=
  ∑ n in s.filter S, (1 : ℚ) / n

theorem sum_reciprocal_S : reciprocal_sum (Finset.range 100) = 10 / 3 := 
by
  sorry

end sum_reciprocal_S_l250_250823


namespace max_min_values_monotonic_increasing_interval_l250_250807

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  - (1 / 3) * x^3 + x^2 + (m^2 - 1) * x

theorem max_min_values (m : ℝ) (h : m = 1) :
  (∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m ≤ 18) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m = 18) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m ≥ 0) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x m = 0) := sorry

theorem monotonic_increasing_interval (m : ℝ) (h : m > 0) :
  ∀ x ∈ Set.Ioo (1 - m : ℝ) (m + 1 : ℝ), 0 < deriv (λ x, f x m) x := sorry

end max_min_values_monotonic_increasing_interval_l250_250807


namespace count_integers_satisfy_inequality_l250_250727

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end count_integers_satisfy_inequality_l250_250727


namespace equivalent_solution_l250_250047

theorem equivalent_solution (c x : ℤ) 
    (h1 : 3 * x + 9 = 6)
    (h2 : c * x - 15 = -5)
    (hx : x = -1) :
    c = -10 :=
sorry

end equivalent_solution_l250_250047


namespace each_persons_share_l250_250903

def total_bill : ℝ := 211.00
def number_of_people : ℕ := 5
def tip_rate : ℝ := 0.15

theorem each_persons_share :
  (total_bill * (1 + tip_rate)) / number_of_people = 48.53 := 
by sorry

end each_persons_share_l250_250903


namespace count_integers_satisfying_inequality_l250_250722

theorem count_integers_satisfying_inequality : 
  ∃ (s : Finset ℤ), (∀ n ∈ s, (n - 3) * (n + 5) < 0) ∧ s.card = 7 :=
begin
  sorry
end

end count_integers_satisfying_inequality_l250_250722


namespace hyperbola_other_asymptote_l250_250012

theorem hyperbola_other_asymptote (c : ℝ) (k : ℝ) (h₁ : c = 3) (h₂ : k = 12):
  (∀ x : ℝ, y = 4x) → (∀ x : ℝ, y = -4x + 24) :=
by
  sorry

end hyperbola_other_asymptote_l250_250012


namespace digital_earth_incorrect_statement_l250_250513

def statement_A : Prop :=
  "The digital Earth refers to the technology system that digitizes the entire Earth's information and is managed by computer networks."

def statement_B : Prop :=
  "The digital Earth is the same as geographic information technology."

def statement_C : Prop :=
  "The digital Earth uses digital means to uniformly address Earth's issues."

def statement_D : Prop :=
  "The digital Earth is the comprehensive application of various technologies such as RS, GIS, GPS, etc."

theorem digital_earth_incorrect_statement : ¬ statement_B :=
by
  -- Proof goes here
  sorry

end digital_earth_incorrect_statement_l250_250513


namespace opposite_of_pi_l250_250864

theorem opposite_of_pi : -1 * Real.pi = -Real.pi := 
by sorry

end opposite_of_pi_l250_250864


namespace two_digit_numbers_with_digit_sum_nine_count_l250_250317

/-- Proof problem: determining the number of two-digit numbers with a digit sum equal to 9. -/

theorem two_digit_numbers_with_digit_sum_nine_count :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (let d1 := n / 10 in let d2 := n % 10 in d1 + d2 = 9)}.card = 9 :=
by
  -- Here you would insert the proof, if it was required.
  sorry

end two_digit_numbers_with_digit_sum_nine_count_l250_250317


namespace truth_teller_liar_island_l250_250424

-- Definitions
def K_claims_all_liars (K M P : Prop) := ¬K ∧ ¬M ∧ ¬P
def M_claims_one_truth_teller (K M P : Prop) := (K ∧ ¬M ∧ ¬P) ∨ (¬K ∧ M ∧ ¬P) ∨ (¬K ∧ ¬M ∧ P)

-- The main theorem
theorem truth_teller_liar_island (K M P : Prop) :
  K_claims_all_liars K M P →
  M_claims_one_truth_teller K M P →
  ¬K ∧ M ∧ ¬P :=
begin
  intros hK hM,
  sorry
end

end truth_teller_liar_island_l250_250424


namespace find_x_l250_250043

variable (n : ℝ) (x : ℝ)

theorem find_x (h1 : n = 15.0) (h2 : 3 * n - x = 40) : x = 5.0 :=
by
  sorry

end find_x_l250_250043


namespace incorrect_statement_l250_250354

-- Define the number of students in the class
def num_students_class : Nat := 400

-- Define the number of students selected for the survey
def num_students_survey : Nat := 40

-- Define Statement A
def statementA : Prop := num_students_survey = 40

-- Define Statement B
def statementB : Prop := True

-- Define Statement C
def statementC : Prop := True

-- Define Statement D (which is said to be incorrect)
def statementD : Prop := ∀ student, student ∈ fin num_students_class → student

-- The theorem to be proven
theorem incorrect_statement : ¬ statementD := by
  -- Placeholder for the proof
  sorry

end incorrect_statement_l250_250354


namespace min_max_values_l250_250049

def function_y (x : ℝ) : ℝ := -x^3 + 3 * x + 2

theorem min_max_values : ∃ (xmin xmax : ℝ), function_y xmin = -1 ∧ function_y xmax = 4 :=
by
  use [-1, 1]
  sorry

end min_max_values_l250_250049


namespace find_a_l250_250461

noncomputable def quadratic_func (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def vertex (a b c : ℤ) : Prop :=
  ∀ x y, (x, y) = (2 : ℝ, 5 : ℝ) → y = quadratic_func a b c x

def passes_through_point (a b c : ℤ) : Prop :=
  ∀ x y, (x, y) = (3 : ℝ, 4 : ℝ) → y = quadratic_func a b c x

theorem find_a (a b c : ℤ) (h1 : vertex a b c) (h2 : passes_through_point a b c) : a = -1 :=
  by sorry

end find_a_l250_250461


namespace yura_picture_dimensions_l250_250517

theorem yura_picture_dimensions (l w : ℕ) (h_frame : (l + 2) * (w + 2) - l * w = l * w) :
    (l = 3 ∧ w = 10) ∨ (l = 4 ∧ w = 6) :=
by {
  sorry
}

end yura_picture_dimensions_l250_250517


namespace reflection_sum_eq_eight_l250_250552

theorem reflection_sum_eq_eight :
  let midpoint := (1 + 5) / 2, (3 + 1) / 2
  let m := (- (5 - 1) / (1 - 3))
  let b := -1 / m
  let fold_line x := 2 * x - 4

  let p := 16 - 2 * q
  let q := 8 + p
  (p, q) == (0, 8) :=
  (p + q = 8) :=
begin
  sorry
end

end reflection_sum_eq_eight_l250_250552


namespace sum_of_first_2019_terms_l250_250065

def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * a n

def S : ℕ → ℕ
| 0     := 0
| (n+1) := S n + a n

theorem sum_of_first_2019_terms : S 2019 = 2 ^ 2019 - 1 :=
by
  sorry

end sum_of_first_2019_terms_l250_250065


namespace king_rook_non_attacking_positions_l250_250744

def chessboard : Type := {i : ℕ // 1 ≤ i ∧ i ≤ 8} × {j : ℕ // 1 ≤ j ∧ j ≤ 8}

def places_king (pos : chessboard) : Prop := sorry
def places_rook (pos_king pos_rook : chessboard) : Prop := sorry
def non_attacking (pos_king pos_rook : chessboard) : Prop := sorry

theorem king_rook_non_attacking_positions :
  ∃ (count : ℕ), count = 2468 ∧
  (∑ pos_king in finset.univ, (∑ pos_rook in finset.univ.filter (λ pos_rook, non_attacking pos_king pos_rook), 1)) = count :=
begin
  sorry
end

end king_rook_non_attacking_positions_l250_250744


namespace n_prime_or_power_of_two_l250_250792

theorem n_prime_or_power_of_two (n : ℕ) (h1 : n > 6) 
  (a : ℕ → ℕ) (hk : ∀ i j, 0 ≤ i → i < j → j < n → Nat.gcd n (a i) = 1 → a(j) - a(i) = a(1) - a(0)) :
  Nat.Prime n ∨ ∃ k : ℕ, n = 2^k :=
  sorry

end n_prime_or_power_of_two_l250_250792


namespace profit_function_relationship_maximize_profit_has_price_115_equal_profit_at_120_l250_250913

open Real

-- Definitions based on conditions
def cost : ℝ := 100

def units_sold_initial (price : ℝ) : ℝ := 
  if price = 120 then 300 else 0

def units_sold_func (price : ℝ) : ℝ :=
  if price > 120 then units_sold_initial 120 - 10 * (price - 120)
  else units_sold_initial 120 + 30 * (120 - price)

def profit_func (price : ℝ) : ℝ :=
  price * units_sold_func price - cost * units_sold_func price

-- Given profit functions (answers identified in solution)
def profit_func1 (x : ℝ) : ℝ := -10 * x^2 + 2500 * x - 150000

def profit_func2 (x : ℝ) : ℝ := -30 * x^2 + 6900 * x - 390000

-- Proof problem statements
theorem profit_function_relationship (x : ℝ) (h1 : x > 120 ∨ 100 < x < 120) :
  (x > 120 → profit_func x = profit_func1 x) ∧ (100 < x < 120 → profit_func x = profit_func2 x) :=
  sorry

theorem maximize_profit_has_price_115 :
  ∃ (x : ℝ), 100 < x ∧ x < 120 ∧ profit_func2 x = 6750 :=
  sorry

theorem equal_profit_at_120 :
  profit_func1 120 = profit_func2 120 :=
  sorry

end profit_function_relationship_maximize_profit_has_price_115_equal_profit_at_120_l250_250913


namespace problem_part1_problem_part2_l250_250693

-- Vector definitions
structure planar_vector :=
  (x : ℝ)
  (y : ℝ)

-- Dot product of two vectors
def dot_product (v1 v2 : planar_vector) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Magnitude of a vector
def magnitude (v : planar_vector) : ℝ :=
  real.sqrt (v.x ^ 2 + v.y ^ 2)

-- Definitions of the planar vectors
def vec_a (α : ℝ) : planar_vector :=
  ⟨real.cos α, real.sin α⟩

def vec_b : planar_vector :=
  ⟨-1/2, real.sqrt(3)/2⟩

-- Theorem statement
theorem problem_part1 (α : ℝ) (hα : 0 ≤ α ∧ α ≤ 2 * real.pi) (h_mag : magnitude (vec_a α) = magnitude vec_b) : 
  dot_product (vec_a α + vec_b) (vec_a α - vec_b) = 0 := sorry

theorem problem_part2 (α : ℝ) (hα : 0 ≤ α ∧ α ≤ 2 * real.pi) 
  (h_eq_mag : magnitude (planar_vector.mk (real.sqrt 3 * (vec_a α).x + vec_b.x) (real.sqrt 3 * (vec_a α).y + vec_b.y)) = 
               magnitude (planar_vector.mk ((vec_a α).x - real.sqrt 3 * vec_b.x) ((vec_a α).y - real.sqrt 3 * vec_b.y))) : 
  α = real.pi / 6 ∨ α = 7 * real.pi / 6 := sorry

end problem_part1_problem_part2_l250_250693


namespace barbara_typing_time_l250_250591

theorem barbara_typing_time :
  let original_speed := 212
  let reduction := 40
  let num_words := 3440
  let reduced_speed := original_speed - reduction
  let time := num_words / reduced_speed
  time = 20 :=
by
  sorry

end barbara_typing_time_l250_250591


namespace lassis_from_mangoes_l250_250961

theorem lassis_from_mangoes (mangoes lassis mangoes' lassis' : ℕ) 
  (h1 : lassis = (8 * mangoes) / 3)
  (h2 : mangoes = 15) :
  lassis = 40 :=
by
  sorry

end lassis_from_mangoes_l250_250961


namespace octahedron_to_tetrahedron_faces_correct_l250_250090

def octahedron := sorry
def tetrahedron := sorry

variables (faces_octahedron : list (list ℝ))
          (tetra_faces : list (list ℝ))
          (A_face : ℝ)
          (V_octahedron : ℝ)
          (V_tetrahedron : ℝ)
          (SA_octahedron : ℝ)
          (SA_tetrahedron : ℝ)

-- Define properties of the octahedron
def properties_octahedron := {
  faces := faces_octahedron,
  edges := 12,
  vertices := 6,
  face_count := 8,
}

-- Define properties of the tetrahedron
def properties_tetrahedron := {
  faces := tetra_faces,
  edges := 6,
  vertices := 4,
  face_count := 4,
}

def is_non_adjacent_non_opposite (fs : list (list ℝ)) : Prop :=
-- A predicate to check if the given faces are non-adjacent and non-opposite.
sorry

noncomputable def resulting_tetrahedron_faces :=
  [ [A_face, A_face, A_face], [A_face, A_face, A_face], [A_face, A_face, A_face], [A_face, A_face, A_face] ]

theorem octahedron_to_tetrahedron_faces_correct :
  is_non_adjacent_non_opposite faces_octahedron →
  faces_octahedron = [ [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, -1, -1], [0, 1, -1], [1, 0, -1], [2, 3, 5] ] →
  tetra_faces = resulting_tetrahedron_faces →
  (V_tetrahedron = 2 * V_octahedron) ∧ (SA_tetrahedron = 2 * SA_octahedron) :=
sorry

end octahedron_to_tetrahedron_faces_correct_l250_250090


namespace emily_purchased_9_wall_prints_l250_250621

/-
  Given the following conditions:
  - cost_of_each_pair_of_curtains = 30
  - num_of_pairs_of_curtains = 2
  - installation_cost = 50
  - cost_of_each_wall_print = 15
  - total_order_cost = 245

  Prove that Emily purchased 9 wall prints
-/
noncomputable def num_wall_prints_purchased 
  (cost_of_each_pair_of_curtains : ℝ) 
  (num_of_pairs_of_curtains : ℝ) 
  (installation_cost : ℝ) 
  (cost_of_each_wall_print : ℝ) 
  (total_order_cost : ℝ) 
  : ℝ :=
  (total_order_cost - (num_of_pairs_of_curtains * cost_of_each_pair_of_curtains + installation_cost)) / cost_of_each_wall_print

theorem emily_purchased_9_wall_prints
  (cost_of_each_pair_of_curtains : ℝ := 30) 
  (num_of_pairs_of_curtains : ℝ := 2) 
  (installation_cost : ℝ := 50) 
  (cost_of_each_wall_print : ℝ := 15) 
  (total_order_cost : ℝ := 245) :
  num_wall_prints_purchased cost_of_each_pair_of_curtains num_of_pairs_of_curtains installation_cost cost_of_each_wall_print total_order_cost = 9 :=
sorry

end emily_purchased_9_wall_prints_l250_250621


namespace point_on_same_side_as_l250_250897

def f (x y : ℝ) : ℝ := 2 * x - y + 1

theorem point_on_same_side_as (x1 y1 : ℝ) (h : f 1 2 > 0) : f 1 0 > 0 := sorry

end point_on_same_side_as_l250_250897


namespace area_PQRS_l250_250429

structure Rectangle :=
  (length : ℝ)
  (breadth : ℝ)

structure EquilateralTriangle :=
  (side_length : ℝ)

noncomputable def area_of_quadrilateral_PQRS (W X Y Z P Q R S : ℝ) 
  (rect : Rectangle) (eq_triangle_WXP eq_triangle_XQY eq_triangle_YRZ eq_triangle_ZSW : EquilateralTriangle)
  (h1: rect.length = 8) (h2: rect.breadth = 6)
  (h3: eq_triangle_WXP.side_length = 6) 
  (h4: eq_triangle_XQY.side_length = 8)
  (h5: eq_triangle_YRZ.side_length = 8)
  (h6: eq_triangle_ZSW.side_length = 6) : ℝ :=
  82 * Real.sqrt 3

theorem area_PQRS (W X Y Z P Q R S : ℝ)
  (rect : Rectangle) (eq_triangle_WXP eq_triangle_XQY eq_triangle_YRZ eq_triangle_ZSW : EquilateralTriangle)
  (h1: rect.length = 8) (h2: rect.breadth = 6)
  (h3: eq_triangle_WXP.side_length = 6) 
  (h4: eq_triangle_XQY.side_length = 8)
  (h5: eq_triangle_YRZ.side_length = 8)
  (h6: eq_triangle_ZSW.side_length = 6) :
  area_of_quadrilateral_PQRS W X Y Z P Q R S rect eq_triangle_WXP eq_triangle_XQY eq_triangle_YRZ eq_triangle_ZSW h1 h2 h3 h4 h5 h6 = 82 * Real.sqrt 3 :=
sorry

end area_PQRS_l250_250429


namespace number_of_perfect_square_factors_of_180_l250_250300

theorem number_of_perfect_square_factors_of_180 :
  let prime_factors := (2, 2, 1); 
  let conditions (p1 p2 p3 : ℕ) := 
    p1 ∈ {0, 2} ∧ 
    p2 ∈ {0, 2} ∧ 
    p3 ∈ {0}
  in Σ' x, x ∈ { (p1, p2, p3) | conditions p1 p2 p3 } = 4 :=
begin
  sorry
end

end number_of_perfect_square_factors_of_180_l250_250300


namespace number_of_perfect_square_factors_of_180_l250_250292

-- Define the prime factorization of 180
def prime_factorization_180 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Define what it means to be a factor of 180
def is_factor_of_180 (n : ℕ) : Prop :=
  ∀ p k, (p, k) ∈ prime_factorization_180 → ∃ m, n = p ^ m ∧ m ≤ k

-- Define what it means to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p m, ∃ k, n = p ^ m ∧ even m

-- Function to count factors of 180 that are perfect squares
def count_perfect_square_factors_of_180 : ℕ :=
  List.length [d | d ← List.range 181, is_factor_of_180 d ∧ is_perfect_square d]

-- Main theorem
theorem number_of_perfect_square_factors_of_180 : count_perfect_square_factors_of_180 = 4 := 
by sorry

end number_of_perfect_square_factors_of_180_l250_250292


namespace num_perfect_square_factors_of_180_l250_250315

theorem num_perfect_square_factors_of_180 (n : ℕ) (h : n = 180) :
  ∃ k : ℕ, k = 4 ∧ ∀ d : ℕ, d ∣ n → ∃ a b c : ℕ, d = 2^a * 3^b * 5^c ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 :=
by
  use 4
  sorry

end num_perfect_square_factors_of_180_l250_250315


namespace cos_angle_CA_CB_l250_250779

def A := (1 : ℝ, 2, -1)
def B := (2 : ℝ, 0, 0)
def C := (0 : ℝ, 1, 3)

def vec_CA := (1 - 0, 2 - 1, -1 - 3)
def vec_CB := (2 - 0, 0 - 1, 0 - 3)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

noncomputable def cos_angle (v w : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product v w) / (magnitude v * magnitude w)

theorem cos_angle_CA_CB : cos_angle vec_CA vec_CB = 13 * Real.sqrt 7 / 42 := by
  sorry

end cos_angle_CA_CB_l250_250779


namespace tan_double_angle_of_A_tan_double_angle_difference_l250_250371

variable {A B : ℝ}

-- Given conditions
def cos_A_is_four_fifths : Prop := Real.cos A = 4 / 5
def tan_B_is_two : Prop := Real.tan B = 2

-- Proof statements
theorem tan_double_angle_of_A : cos_A_is_four_fifths → Real.tan (2 * A) = 24 / 7 := 
by sorry

theorem tan_double_angle_difference : cos_A_is_four_fifths → tan_B_is_two → Real.tan (2 * A - 2 * B) = -4 / 3 :=
by sorry

end tan_double_angle_of_A_tan_double_angle_difference_l250_250371


namespace max_ab_l250_250671

-- Given constants and conditions
variables {a b : ℝ}

-- Hypotheses
hypothesis h1 : a > 0
hypothesis h2 : b > 0
hypothesis h3 : 3 * a + 2 * b = 1

-- Main statement to be proved
theorem max_ab : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 3 * a + 2 * b = 1 ∧ ab = 1 / 24 :=
by {
  sorry
}

end max_ab_l250_250671


namespace swap_values_correct_l250_250258

variables (a b : ℕ) (c : ℕ)

theorem swap_values_correct :
  let a₀ := a in
  let b₀ := b in
  let c := a₀ in
  let a := b₀ in
  let b := c in
  a₀ = b ∧ b₀ = a :=
by
  sorry

end swap_values_correct_l250_250258


namespace probability_same_color_l250_250582

open_locale big_operators

-- Definitions for the problem's conditions
def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles
def number_of_draws : ℕ := 4

-- Definitions for the probabilities
def P_all_red : ℚ := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) *
                      ((red_marbles - 2) / (total_marbles - 2)) * ((red_marbles - 3) / (total_marbles - 3))

def P_all_white : ℚ := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) *
                       ((white_marbles - 2) / (total_marbles - 2)) * ((white_marbles - 3) / (total_marbles - 3))

def P_all_blue : ℚ := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) *
                      ((blue_marbles - 2) / (total_marbles - 2)) * ((blue_marbles - 3) / (total_marbles - 3))

def P_same_color : ℚ := P_all_red + P_all_white + P_all_blue

-- Theorem statement to prove the probability of drawing four marbles of the same color
theorem probability_same_color :
  P_same_color = 55 / 3060 :=
sorry

end probability_same_color_l250_250582


namespace line_exists_l250_250682

open Real

variables {a b c k x y : ℝ}
variable {l : ℝ → ℝ}

-- Define the ellipse C with given conditions
def ellipse_C (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

lemma foci_condition (h1 : a > b) (h2 : b > 0) (h3 : (F₁ F₂ : ℝ × ℝ), F₁ = (-sqrt 3, 0) ∧ F₂ = (sqrt 3, 0)) :
  a^2 = 4 ∧ b^2 = 1 :=
sorry

-- Define the existence of line l that intersects the ellipse at points D and E such that |AD| = |AE|
theorem line_exists (h1 : a > b) (h2 : b > 0) (h3 : point_A = (1, 0))
  (h4 : ∃ (l : ℝ → ℝ), ∃ D E, l x = y ∧ ellipse_C D.1 D.2 ∧ ellipse_C E.1 E.2 ∧ |A - D| = |A - E|)
  {k : ℝ} :
  (k > sqrt 5 / 5) ∨ (k < - sqrt 5 / 5) :=
sorry

end line_exists_l250_250682


namespace longest_flight_time_l250_250649

theorem longest_flight_time (V₀ g : ℝ) (hV₀ : V₀ = 10) (hg : g = 10) :
  ∃ τ : ℝ, τ = 1.6 ∧
    ∀ α : ℝ, 
      let l := (V₀^2 * sin (2 * α)) / g in
      (l ≥ 0.96 * (V₀^2 * sin (2 * (π/4))) / g) → 
      let t := (2 * V₀ * sin α) / g in
      t ≤ τ := 
sorry

end longest_flight_time_l250_250649


namespace sum_possible_k_l250_250365

theorem sum_possible_k (j k : ℕ) (hjk : 1 / (j : ℝ) + 1 / (k : ℝ) = 1 / 2) : k ∈ {6, 4, 3} → k ∈ {6, 4, 3} ∧ {6, 4, 3}.sum = 13 :=
by 
  sorry

end sum_possible_k_l250_250365


namespace greatest_value_of_x_l250_250503

theorem greatest_value_of_x : ∀ x : ℝ, 4*x^2 + 6*x + 3 = 5 → x ≤ 1/2 :=
by
  intro x
  intro h
  sorry

end greatest_value_of_x_l250_250503


namespace percentage_calculation_l250_250110

variable (percentage : Type) [LinearOrderedField percentage]

def percentage_answered_first_correctly (A B Anotinclusion answered_atleast_one : percentage) : Prop :=
  A = 65

-- Given conditions
variable {B : percentage} (hB : B = 55)
variable {Anotinclusion : percentage} (hAnotinclusion : Anotinclusion = 40)
variable {answered_atleast_one : percentage} (h_answered_atleast_one : answered_atleast_one = 80)

-- The theorem
theorem percentage_calculation :
  ∃ A : percentage, percentage_answered_first_correctly A B Anotinclusion answered_atleast_one :=
by 
  -- Assuming the conditions
  assume hB : B = 55
  assume hAnotinclusion : Anotinclusion = 40
  assume h_answered_atleast_one : answered_atleast_one = 80 

  -- Calculate A following the given conditions
  let A := 80 - 15

  -- Proving A = 65 
  use A
  show A = 65 from sorry

end percentage_calculation_l250_250110


namespace distance_from_origin_to_line_l250_250268

theorem distance_from_origin_to_line (O : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (hO : O = (0, 0)) (hline : ∀ x y, line x y ↔ (x + y + 2 = 0)) :
  ∃ d : ℝ, d = Real.sqrt 2 ∧ (∀ x y, line x y → ∥(x, y) - O∥ = d) := sorry

end distance_from_origin_to_line_l250_250268


namespace lines_parallel_a_eq_3_l250_250416

theorem lines_parallel_a_eq_3
  (a : ℝ)
  (l1 : ℝ → ℝ → Prop := λ x y, a * x + 3 * y + 4 = 0)
  (l2 : ℝ → ℝ → Prop := λ x y, x + (a - 2) * y + a^2 - 5 = 0)
  (parallel : ∀ x1 y1 x2 y2, l1 x1 y1 → l2 x2 y2 → -a / 3 = -1 / (a - 2)) :
  a = 3 :=
by
  sorry

end lines_parallel_a_eq_3_l250_250416


namespace floor_5_7_l250_250199

theorem floor_5_7 : Int.floor 5.7 = 5 :=
by
  sorry

end floor_5_7_l250_250199


namespace fraction_division_l250_250082

variable {x : ℝ}
variable (hx : x ≠ 0)

theorem fraction_division (hx : x ≠ 0) : (3 / 8) / (5 * x / 12) = 9 / (10 * x) := 
by
  sorry

end fraction_division_l250_250082


namespace not_all_inequalities_true_l250_250432

theorem not_all_inequalities_true (a b c : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : 0 < b ∧ b < 1) (h₂ : 0 < c ∧ c < 1) :
  ¬(a * (1 - b) > 1 / 4 ∧ b * (1 - c) > 1 / 4 ∧ c * (1 - a) > 1 / 4) :=
  sorry

end not_all_inequalities_true_l250_250432


namespace future_value_compound_interest_l250_250058

theorem future_value_compound_interest
  (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ)
  (hP : P = 2500) (hr : r = 0.20) (hn : n = 1) (ht : t = 2) :
  P * (1 + r / n) ^ (n * t) = 3600 := 
by {
  rw [hP, hr, hn, ht],
  norm_num,
}

end future_value_compound_interest_l250_250058


namespace find_side_a_from_triangle_conditions_l250_250761

-- Define the variables.
variables (A : ℝ) (b : ℝ) (area : ℝ) (a : ℝ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  A = 60 * Real.pi / 180 ∧ -- Convert 60 degrees to radians for Lean.
  b = 1 ∧
  area = Real.sqrt 3

-- The theorem we want to prove.
theorem find_side_a_from_triangle_conditions :
  conditions A b area → a = Real.sqrt 13 :=
by
  sorry

end find_side_a_from_triangle_conditions_l250_250761


namespace leading_digit_same_l250_250220

theorem leading_digit_same (n : ℕ) (hn : 0 < n) (h : leading_digit (2^n) = leading_digit (5^n)) : leading_digit (2^n) = 3 :=
sorry

noncomputable def leading_digit (x : ℕ) : ℕ :=
if x = 0 then 0 else x / 10^(x.log10)


end leading_digit_same_l250_250220


namespace intersection_M_N_l250_250276

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = {0, 1} := 
by
  sorry

end intersection_M_N_l250_250276


namespace john_new_cards_l250_250789

def cards_per_page : ℕ := 3
def old_cards : ℕ := 16
def pages_used : ℕ := 8

theorem john_new_cards : (pages_used * cards_per_page) - old_cards = 8 := by
  sorry

end john_new_cards_l250_250789


namespace minimize_EF_length_l250_250426

-- Define the unit cube
structure UnitCube := 
  (A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ)
  (AC : ℝ)
  (B1D1 : ℝ)

-- Define the segments AE, B1F, and the needed lengths
def AE_length (l : ℝ) := l
def B1F_length (ml : ℝ) := ml

-- Prove the minimal length of segment EF
theorem minimize_EF_length : 
  ∀ (m : ℝ) (h : m > 0), 
  ∃ l : ℝ, EF_length (AE_length l) (B1F_length (m * l)) = 1 + l^2 * (m^2 + 1) - l * real.sqrt 2 * (m + 1) ->
  l = (real.sqrt 2 / 2) * (m + 1) / (m^2 + 1) :=
by
  intros m h
  -- Definitions and statements for using in proof
  sorry

end minimize_EF_length_l250_250426


namespace expected_value_of_biased_die_l250_250963

-- Definitions for probabilities
def prob1 : ℚ := 1 / 15
def prob2 : ℚ := 1 / 15
def prob3 : ℚ := 1 / 15
def prob4 : ℚ := 1 / 15
def prob5 : ℚ := 1 / 5
def prob6 : ℚ := 3 / 5

-- Definition for expected value
def expected_value : ℚ := (prob1 * 1) + (prob2 * 2) + (prob3 * 3) + (prob4 * 4) + (prob5 * 5) + (prob6 * 6)

theorem expected_value_of_biased_die : expected_value = 16 / 3 :=
by sorry

end expected_value_of_biased_die_l250_250963


namespace factorization_correct_l250_250990

theorem factorization_correct (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end factorization_correct_l250_250990


namespace sum_difference_of_consecutive_integers_l250_250522

theorem sum_difference_of_consecutive_integers (n : ℤ) :
  let set1 := [(n-3), (n-2), (n-1), n, (n+1), (n+2), (n+3)]
  let set2 := [(n+1), (n+2), (n+3), (n+4), (n+5), (n+6), (n+7)]
  let S1 := set1.sum
  let S2 := set2.sum
  S2 - S1 = 28 :=
by
  let set1 := [(n-3), (n-2), (n-1), n, (n+1), (n+2), (n+3)]
  let set2 := [(n+1), (n+2), (n+3), (n+4), (n+5), (n+6), (n+7)]
  let S1 := set1.sum
  let S2 := set2.sum
  have hS1 : S1 = (n-3) + (n-2) + (n-1) + n + (n+1) + (n+2) + (n+3) := by sorry
  have hS2 : S2 = (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) := by sorry
  have h_diff : S2 - S1 = 28 := by sorry
  exact h_diff

end sum_difference_of_consecutive_integers_l250_250522


namespace area_triangle_PTW_in_regular_octagon_l250_250557

theorem area_triangle_PTW_in_regular_octagon (a : ℝ) (h : a = 3) :
  let side_length := a,
      octagon := regular_polygon 8 side_length,
      triangle_PTW := triangle_in_octagon_PTW octagon
  in triangle_PTW.area = 9 / 2 := 
by
  sorry

end area_triangle_PTW_in_regular_octagon_l250_250557


namespace cost_price_l250_250580

theorem cost_price (MP : ℝ) (SP : ℝ) (C : ℝ) 
  (h1 : MP = 87.5) 
  (h2 : SP = 0.95 * MP) 
  (h3 : SP = 1.25 * C) : 
  C = 66.5 := 
by
  sorry

end cost_price_l250_250580


namespace greatest_five_digit_divisible_by_13_l250_250935

def is_five_digit (n : ℕ) := n ≥ 10000 ∧ n < 100000

def is_divisible_by_13 (n : ℕ) := n % 13 = 0

def distinct_digits (A B C : ℕ) := A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ A < 10 ∧ B < 10 ∧ C < 10

theorem greatest_five_digit_divisible_by_13 :
  ∃ (A B C : ℕ), is_five_digit (10001 * A + 1010 * B + 100 * C) ∧
                 distinct_digits A B C ∧
                 is_divisible_by_13 (10001 * A + 1010 * B + 100 * C) ∧
                 ∀ (A' B' C' : ℕ), 
                   is_five_digit (10001 * A' + 1010 * B' + 100 * C') ∧
                   distinct_digits A' B' C' ∧
                   is_divisible_by_13 (10001 * A' + 1010 * B' + 100 * C') →
                   10001 * A + 1010 * B + 100 * C ≥ 10001 * A' + 1010 * B' + 100 * C' :=
exists.intro 8 (exists.intro 3 (exists.intro 6 (by sorry)))

end greatest_five_digit_divisible_by_13_l250_250935


namespace fifteenth_prime_is_correct_l250_250981

-- Definitions
def fifth_prime := 11
def fifteenth_prime := 47

-- The statement to be proven
theorem fifteenth_prime_is_correct : nat.prime fifteenth_prime ∧ fifteenth_prime = 47 :=
by
  sorry

end fifteenth_prime_is_correct_l250_250981


namespace part_a_l250_250105

theorem part_a (x y : ℝ) (hx : 1 > x ∧ x ≥ 0) (hy : 1 > y ∧ y ≥ 0) : 
  ⌊5 * x⌋ + ⌊5 * y⌋ ≥ ⌊3 * x + y⌋ + ⌊3 * y + x⌋ := sorry

end part_a_l250_250105


namespace constant_length_of_perpendicular_l250_250852

-- Constants and definitions used in the problem
variables {A : Point} {B C : Point}
variable {BC_length : ℝ}

-- Conditions of the problem
axiom segment_length (BC : Segment) : length BC = BC_length
axiom slides_along_angle (A B C : Point) : (B,C) slides along angle with vertex A

-- Definition of the midpoint and the perpendicular segment
noncomputable def midpoint (B C : Point) : Point := midpoint B C
noncomputable def perpendicular_from_midpoint (B C : Point) : Line := perpendicular_from (midpoint B C) (B, C)

-- Goal: Prove the length from midpoint to the intersection with the angle bisector is constant
theorem constant_length_of_perpendicular (A B C : Point)
  (hBC : length (Segment.mk B C) = BC_length)
  (hslides : slides_along_angle (Angle.mk A B C)) :
  ∃ (L : ℝ), ∀ (t : ℝ), length (perpendicular_from_midpoint (B C) ∩ angle_bisector A B C) = L :=
by
  sorry

end constant_length_of_perpendicular_l250_250852


namespace product_of_two_distinct_elements_of_S_count_l250_250397

-- Let S be the set of all positive integer divisors of 72000
def S := {d : ℕ | d > 0 ∧ d ∣ 72000}

-- Define the product of two distinct elements of S
def product_of_distinct_elements (d1 d2 : ℕ) : ℕ :=
  if d1 ∈ S ∧ d2 ∈ S ∧ d1 ≠ d2 then d1 * d2 else 1

-- The theorem to prove
theorem product_of_two_distinct_elements_of_S_count : 
  (finset.univ : finset {d : ℕ | d ∣ 72000}).card.filter (λ (d : ℕ), ∃ (a b : ℕ) (ha : a ∈ S) (hb : b ∈ S), a ≠ b ∧ d = a * b) = 377 :=
sorry 

end product_of_two_distinct_elements_of_S_count_l250_250397


namespace cost_to_fill_tank_is_45_l250_250042

noncomputable def cost_to_fill_tank : ℕ :=
  let F := 45 in
  let total_distance := 2000 in
  let distance_per_tank := 500 in
  let num_refills := total_distance / distance_per_tank in
  let fuel_cost := num_refills * F in
  let food_cost := (3 / 5) * fuel_cost in
  let total_spent := fuel_cost + food_cost in
  if total_spent = 288 then F
  else sorry

theorem cost_to_fill_tank_is_45 :
  ∃ F : ℕ, (total_distance := 2000) (distance_per_tank := 500) (num_refills := total_distance / distance_per_tank)
  (fuel_cost := num_refills * F) (food_cost := (3 / 5) * fuel_cost)
  (total_spent := fuel_cost + food_cost),
  total_spent = 288 → F = 45 :=
begin
  use 45,
  intro h,
  let total_distance := 2000,
  let distance_per_tank := 500,
  let num_refills := total_distance / distance_per_tank,
  let fuel_cost := num_refills * 45,
  let food_cost := (3 / 5) * fuel_cost,
  let total_spent := fuel_cost + food_cost,
  have : total_spent = 288, by assumption,
  exact eq.symm h
end

end cost_to_fill_tank_is_45_l250_250042


namespace incorrect_statements_l250_250239

/-- Definitions for lines and planes and their relations. -/
variables {a b : Set ℝ^3} {α β : Set ℝ^3}

def parallel (l₁ l₂ : Set ℝ^3) : Prop := ∀ p₁ ∈ l₁, ∀ p₂ ∈ l₂, ∃ v, p₁ + v = p₂

def contained_in (l : Set ℝ^3) (p : Set ℝ^3) : Prop := ∀ pt ∈ l, pt ∈ p

def skew (l₁ l₂ : Set ℝ^3) : Prop := ¬(parallel l₁ l₂) ∧ ∀ pt₁ ∈ l₁, ∀ pt₂ ∈ l₂, pt₁ ≠ pt₂

def conditions : Prop :=
  (parallel a b ∧ contained_in b α) ∧
  (parallel a α ∧ parallel b α) ∧
  (skew a b ∧ contained_in a α ∧ contained_in b β ∧ parallel a β ∧ parallel b α) ∧
  (parallel a α ∧ parallel a β)

/-- The theorem stating the incorrect statements are A, B, and D. -/
theorem incorrect_statements : conditions → ([false, false, true, false] = [A, B, C, D]) :=
by
  sorry

end incorrect_statements_l250_250239


namespace product_equality_l250_250474

theorem product_equality : (2.05 * 4.1 = 20.5 * 0.41) :=
by
  sorry

end product_equality_l250_250474


namespace geometric_series_first_term_l250_250150

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by 
  sorry

end geometric_series_first_term_l250_250150


namespace find_diameter_of_tin_l250_250851

noncomputable def diameter_of_cylinder (V h : ℝ) : ℝ :=
  2 * real.sqrt (V / (π * h))

theorem find_diameter_of_tin : 
  diameter_of_cylinder 20 5 = (4 / real.sqrt π) :=
by
  sorry

end find_diameter_of_tin_l250_250851


namespace sum_of_digits_repeating_decimal_of_one_over_99_squared_l250_250457

theorem sum_of_digits_repeating_decimal_of_one_over_99_squared : 
    let n := 198 in
    let seq := 0 :: 0 :: 0 :: 1 :: 0 :: 2 :: 0 :: 3 :: 0 :: 4 :: 0 :: 5 :: 0 :: 6 :: 0 :: 7 :: 0 :: 8 :: 0 :: 9 :: ... (up to 99) :: 99 :: (with modulo 100 wrap around to 0 0)
    ∑ i in finset.range n, seq.nth i = 883 :=
sorry

end sum_of_digits_repeating_decimal_of_one_over_99_squared_l250_250457


namespace floor_5_7_eq_5_l250_250190

theorem floor_5_7_eq_5 : Int.floor 5.7 = 5 := by
  sorry

end floor_5_7_eq_5_l250_250190


namespace proof_problem_l250_250399

-- Definitions
def U : Set ℕ := {x | x < 7 ∧ x > 0}
def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {2, 3, 4, 5}

-- The equality proof statement
theorem proof_problem :
  (A ∩ B = {2, 5}) ∧
  ({x | x ∈ U ∧ ¬ (x ∈ A)} = {3, 4, 6}) ∧
  (A ∪ {x | x ∈ U ∧ ¬ (x ∈ B)} = {1, 2, 5, 6}) :=
by
  sorry

end proof_problem_l250_250399


namespace mode_and_median_of_data_set_l250_250559

-- Define the dataset
def data_set : List ℕ := [2, 3, 4, 4, 4, 5, 5]

-- Define a function to extract the mode
def mode (l : List ℕ) : ℕ :=
  l.foldr (λ x acc, if l.count x > l.count acc then x else acc) (l.head!)

-- Define a function to extract the median
def median (l : List ℕ) : ℕ :=
  let sorted := l.insertionSort (≤)
  sorted.get! (sorted.length / 2)

-- Prove that the mode and median are both 4
theorem mode_and_median_of_data_set :
  mode data_set = 4 ∧ median data_set = 4 := by
  sorry

end mode_and_median_of_data_set_l250_250559


namespace basketball_games_count_l250_250827

noncomputable def tokens_per_game : ℕ := 3
noncomputable def total_tokens : ℕ := 18
noncomputable def air_hockey_games : ℕ := 2
noncomputable def air_hockey_tokens := air_hockey_games * tokens_per_game
noncomputable def remaining_tokens := total_tokens - air_hockey_tokens

theorem basketball_games_count :
  (remaining_tokens / tokens_per_game) = 4 := by
  sorry

end basketball_games_count_l250_250827


namespace pugs_cleaning_time_l250_250515

theorem pugs_cleaning_time : 
  (∀ (p t: ℕ), 15 * 12 = p * t ↔ 15 * 12 = 4 * 45) :=
by
  sorry

end pugs_cleaning_time_l250_250515


namespace find_mass_plate_l250_250635

-- Define the region D in Cartesian coordinates
def regionD (x y : ℝ) : Prop :=
  1 ≤ (x^2 / 16 + y^2) ∧ (x^2 / 16 + y^2) ≤ 3 ∧ y ≥ x / 4 ∧ x ≥ 0

-- Define the surface density μ
def surfaceDensity (x y : ℝ) : ℝ :=
  x / (y^5)

-- Define the integral for mass
def mass (m : ℝ) : Prop :=
  m = ∫ x in 0..(2*√3), ∫ y in (x / 4)..(sqrt(3 - x^2 / 16)), surfaceDensity x y

-- Prove that the mass is equal to 4
theorem find_mass_plate : ∃ (m : ℝ), mass m ∧ m = 4 :=
by
  sorry

end find_mass_plate_l250_250635


namespace pentagon_perimeter_l250_250888

/-!
# Problem Statement

Given the side lengths of the pentagon \(ABCDE\):
\(AB = 1\)
\(BC = \sqrt{3}\)
\(CD = \sqrt{4} = 2\)
\(DE = \sqrt{5}\)
And calculating \(EA\) as \( \sqrt{13} \)

We need to prove that the perimeter of the pentagon \(ABCDE\) is \(1 + \sqrt{3} + 2 + \sqrt{5} + \sqrt{13} = 3 + \sqrt{3} + \sqrt{5} + \sqrt{13}\)

-/

noncomputable def perimeter_pentagon (AB BC CD DE EA : ℝ) : ℝ :=
  AB + BC + CD + DE + EA

theorem pentagon_perimeter (h1 : AB = 1) (h2 : BC = sqrt 3) (h3 : CD = sqrt 4) (h4 : DE = sqrt 5) (h5 : EA = sqrt 13) :
  perimeter_pentagon AB BC CD DE EA = 3 + sqrt 3 + sqrt 5 + sqrt 13 :=
by
  sorry

end pentagon_perimeter_l250_250888


namespace inverse_modulus_l250_250243

theorem inverse_modulus (h1 : 17⁻¹ ≡ 53 [MOD 89]) (h2 : 72 ≡ -17 [MOD 89]) : 72⁻¹ ≡ 36 [MOD 89] :=
sorry

end inverse_modulus_l250_250243


namespace highest_elevation_l250_250131

   noncomputable def elevation (t : ℝ) : ℝ := 240 * t - 24 * t^2

   theorem highest_elevation : ∃ t : ℝ, elevation t = 600 ∧ ∀ x : ℝ, elevation x ≤ 600 := 
   sorry
   
end highest_elevation_l250_250131


namespace count_valid_triangles_l250_250576

/-- 
Define the problem constraints: scalene triangles with side lengths a, b, c, 
where a < b < c, a + c = 2b, and a + b + c ≤ 30.
-/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + c = 2 * b ∧ a + b + c ≤ 30

/-- 
Statement of the problem: Prove that there are 20 distinct triangles satisfying the above constraints. 
-/
theorem count_valid_triangles : ∃ n, n = 20 ∧ (∀ {a b c : ℕ}, is_valid_triangle a b c → n = 20) :=
sorry

end count_valid_triangles_l250_250576


namespace rectangle_area_ratio_l250_250663

theorem rectangle_area_ratio (a : ℝ) (a_pos : 0 < a) :
  let l := a * Real.sqrt 2 in
  let w := a in
  let area_square := a^2 in
  let area_rectangle := l * w in
  area_rectangle / area_square = Real.sqrt 2 :=
by
  -- proof steps here
  sorry

end rectangle_area_ratio_l250_250663


namespace q_zero_iff_arithmetic_l250_250679

-- Definitions of the terms and conditions
variables (A B q : ℝ) (hA : A ≠ 0)
def Sn (n : ℕ) : ℝ := A * n^2 + B * n + q
def arithmetic_sequence (an : ℕ → ℝ) : Prop := ∃ d a1, ∀ n, an n = a1 + n * d

-- The proof statement we need to show
theorem q_zero_iff_arithmetic (an : ℕ → ℝ) :
  (q = 0) ↔ (∃ a1 d, ∀ n, Sn A B 0 n = (d / 2) * n^2 + (a1 - d / 2) * n) :=
sorry

end q_zero_iff_arithmetic_l250_250679


namespace s₁₇_value_l250_250470

theorem s₁₇_value (s : Fin 1008 → ℝ) (k : ℝ) 
  (h_sum : (Finset.univ.sum (λ i, s i)) = 2016^2) 
  (h_fraction_eq : ∀ i : Fin 1008, (s i) / (s i + (2 * (i : ℝ) + 1)) = k) :
  s ⟨16, by decide⟩ = 132 :=
by
  sorry

end s₁₇_value_l250_250470


namespace number_of_integers_with_three_divisors_l250_250859

def has_exactly_three_positive_divisors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p * p

theorem number_of_integers_with_three_divisors (n : ℕ) :
  n = 2012 → Nat.card { x : ℕ | x ≤ n ∧ has_exactly_three_positive_divisors x } = 14 :=
by
  sorry

end number_of_integers_with_three_divisors_l250_250859


namespace xy_sum_l250_250834

theorem xy_sum (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y + 20) : x + y = 12 + 2 * Real.sqrt 6 ∨ x + y = 12 - 2 * Real.sqrt 6 :=
by
  sorry

end xy_sum_l250_250834


namespace perfect_square_factors_of_180_l250_250305

theorem perfect_square_factors_of_180 {n : ℕ} (h1 : n = 180) : 
  (∃ f : ℕ → ℕ, (∀ p, (p = 2 ∨ p = 3 ∨ p = 5 → 
    (f p = 0 ∨ (p = 2 ∨ p = 3) ∧ f p = 2 ∧ p ≠ 5))) ∧ 2^(f 2) * 3^(f 3) * 5^(f 5) = n ∧  ((f 2 + f 3 + f 5) % 2 = 0)) → 
  4 :=
begin
  sorry
end

end perfect_square_factors_of_180_l250_250305


namespace train_stops_12_minutes_per_hour_l250_250987

theorem train_stops_12_minutes_per_hour
  (speed_excluding_stoppages : ℕ)
  (speed_including_stoppages : ℕ)
  (H1 : speed_excluding_stoppages = 45)
  (H2 : speed_including_stoppages = 36) :
  (60 * (speed_excluding_stoppages - speed_including_stoppages)) / speed_excluding_stoppages = 12 :=
by
  rw [H1, H2]
  norm_num

end train_stops_12_minutes_per_hour_l250_250987


namespace find_value_of_dot_product_l250_250004

open Real

variables {V : Type*} [InnerProductSpace ℝ V]
variables (a b c : V) (k : ℝ)

theorem find_value_of_dot_product 
  (h₁ : a = k • b) 
  (h₂ : ⟪a, c⟫ = 0) 
  : ⟪c, a + 2 • b⟫ = 0 :=
sorry

end find_value_of_dot_product_l250_250004


namespace volume_of_region_l250_250639

noncomputable def f (x y z : ℝ) : ℝ :=
|x + y + z| + |x + y - z| + |x - y + z| + |-x + y - z|

theorem volume_of_region : 
  ∀ x y z : ℝ, f x y z ≤ 6 ↔ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x ≤ 1.5 ∧ y ≤ 1.5 ∧ z ≤ 1.5 ∧ x + y + z ≤ 3 → 
  volume_of_region f 22.5 :=
begin
  cases h,
  sorry
end

end volume_of_region_l250_250639


namespace hexagon_perimeter_is_27_l250_250494

-- Given conditions as definitions
def triangle1_perimeter : ℕ := 12
def triangle2_perimeter : ℕ := 15

def side_length1 (perimeter : ℕ) : ℕ := perimeter / 3
def side_length2 (perimeter : ℕ) : ℕ := perimeter / 3

-- Actual theorem statement
theorem hexagon_perimeter_is_27
  (h1 : side_length1 triangle1_perimeter = 4)
  (h2 : side_length2 triangle2_perimeter = 5) :
  let a := side_length1 triangle1_perimeter in
  let b := side_length1 triangle1_perimeter in
  let c := side_length1 triangle1_perimeter in
  let d := side_length2 triangle2_perimeter in
  let e := side_length2 triangle2_perimeter in
  let f := side_length2 triangle2_perimeter in
  a + b + c + d + e + f = 27 := by
  sorry

end hexagon_perimeter_is_27_l250_250494


namespace triangle_to_initial_position_l250_250568

-- Definitions for triangle vertices
structure Point where
  x : Int
  y : Int

def p1 : Point := { x := 0, y := 0 }
def p2 : Point := { x := 6, y := 0 }
def p3 : Point := { x := 0, y := 4 }

-- Definitions for transformations
def rotate90 (p : Point) : Point := { x := -p.y, y := p.x }
def rotate180 (p : Point) : Point := { x := -p.x, y := -p.y }
def rotate270 (p : Point) : Point := { x := p.y, y := -p.x }
def reflect_y_eq_x (p : Point) : Point := { x := p.y, y := p.x }
def reflect_y_eq_neg_x (p : Point) : Point := { x := -p.y, y := -p.x }

-- Definitions for combination of transformations
-- This part defines how to combine transformations, e.g., as a sequence of three transformations.
def transform (fs : List (Point → Point)) (p : Point) : Point :=
  fs.foldl (fun acc f => f acc) p

-- The total number of valid sequences that return the triangle to its original position
def valid_sequences_count : Int := 6

-- Lean 4 statement
theorem triangle_to_initial_position : valid_sequences_count = 6 := by
  sorry

end triangle_to_initial_position_l250_250568


namespace part1_part2_part3_l250_250262

namespace ProofProblems

open Real

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  log 4 (4 ^ x + 1) + k * x

noncomputable def h (x : ℝ) (m : ℝ) : ℝ :=
  4 ^ (f x - 1/2) + m * 2 ^ x - 1

theorem part1 (k : ℝ) :
  (∀ x : ℝ, log 4 (4 ^ (-x) + 1) - k * x = log 4 (4 ^ x + 1) + k * x) ↔ k = -1/2 :=
by sorry

theorem part2 (a : ℝ) : 
  (¬∃ x : ℝ, log 4 (4 ^ x + 1) - 1/2 * x = 1/2 * x + a) ↔ a ≤ 0 :=
by sorry

theorem part3 :
  (∃ m : ℝ, ∀ x : ℝ, 0 ≤ x ∧ x ≤ log 2 3 → h x m = 0) ↔ m = -1 :=
by sorry

end ProofProblems

end part1_part2_part3_l250_250262


namespace findAngleA_l250_250343

-- Conditions as definitions
def triangleSideLengths (a b c : ℝ) : Prop := true -- assumes valid triangle sides
def angleEqualityCondition (a b : ℝ) (A B : ℝ) : Prop := 
  a * Real.sin B = sqrt(3) * b * Real.cos A

-- Questions turned into statements to prove
theorem findAngleA (a b : ℝ) (A B : ℝ) 
  (h1 : angleEqualityCondition a b A B) : 
  A = π / 3 :=
sorry

-- More definitions for the second part, including specific values and the cosine rule
def cosineRule (a b c : ℝ) (C : ℝ) : Prop :=
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos C

def triangleArea (a b C : ℝ) : ℝ := 
  1 / 2 * a * b * Real.sin C

-- Second question using specific values
noncomputable def findTriangleArea (a b C c : ℝ) 
  (ha : a = sqrt 7)
  (hb : b = 2) 
  (hC : C = π / 4)
  (hcos : cosineRule a b c (π / 3)) :
  triangleArea b c (π / 4) = 3 * sqrt 3 / 2 :=
sorry

end findAngleA_l250_250343


namespace sum_binom_coeffs_l250_250408

open BigOperators

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_binom_coeffs :
  let a : ℕ → ℕ := λ i, binom 16 i
  ∑ i in Finset.range 9, (i + 1) * a (i + 1) = 262144 :=
by
  let a : ℕ → ℕ := λ i, binom 16 i
  sorry

end sum_binom_coeffs_l250_250408


namespace complex_problem_l250_250337

-- Define the complex number z
def z : ℂ := 1 + complex.i

-- State the theorem with the given condition and desired outcome
theorem complex_problem : (1 + z) * z = 1 + 3 * complex.i := 
  sorry

end complex_problem_l250_250337


namespace find_f_2008_l250_250804

noncomputable def f (x : ℝ) : ℝ := Real.cos x

noncomputable def f_n (n : ℕ) : (ℝ → ℝ) :=
match n with
| 0     => f
| (n+1) => (deriv (f_n n))

theorem find_f_2008 (x : ℝ) : (f_n 2008) x = Real.cos x := by
  sorry

end find_f_2008_l250_250804


namespace simplify_expression_l250_250029

variable (y : ℝ)

theorem simplify_expression : 
    4 * y + 8 * y^2 + 6 - (3 - 4 * y - 8 * y^2) = 16 * y^2 + 8 * y + 3 :=
begin
  sorry
end

end simplify_expression_l250_250029


namespace count_integers_satisfy_inequality_l250_250726

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end count_integers_satisfy_inequality_l250_250726


namespace rita_bought_4_pounds_l250_250837

-- Define the conditions
def card_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def amount_left : ℝ := 35.68

-- Define the theorem to prove the number of pounds of coffee bought is 4
theorem rita_bought_4_pounds :
  (card_amount - amount_left) / cost_per_pound = 4 := by sorry

end rita_bought_4_pounds_l250_250837


namespace problem_part1_problem_part2_l250_250746

theorem problem_part1 (a a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (2 * x - a) ^ 7 = a₀ + a₁ * (x + 1) + a₂ * (x + 1) ^ 2 + a₃ * (x + 1) ^ 3 + a₄ * (x + 1) ^ 4 + a₅ * (x + 1) ^ 5 + a₆ * (x + 1) ^ 6 + a₇ * (x + 1) ^ 7)
  → a₄ = -560
  → a = -1 :=
by { sorry }

theorem problem_part2 (a a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (a = -1)
  → (2 * 0 - a) ^ 7 = a₀ + a₁ * 0 + a₂ * 0 ^ 2 + a₃ * 0 ^ 3 + a₄ * 0 ^ 4 + a₅ * 0 ^ 5 + a₆ * 0 ^ 6 + a₇ * 0 ^ 7
  → ∑ i in finset.range 8 \ finset.singleton 0, abs ([a₁, a₂, a₃, a₄, a₅, a₆, a₇][i]) = 2186 :=
by { sorry }

end problem_part1_problem_part2_l250_250746


namespace percent_palindromes_with_7_l250_250129

-- Definition: A number is a palindrome if it reads the same forward and backward
def is_palindrome (n : ℕ) : Prop := 
  let digits := repr n in
  digits = reverse digits

-- Definition: Palindromes between 1000 and 2000
def palindromes_1000_to_2000 : Set ℕ := 
  {n | 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n}

-- Definition: Palindromes between 1000 and 2000 that contain at least one 7
def palindromes_with_7_1000_to_2000 : Set ℕ := 
  {n | n ∈ palindromes_1000_to_2000 ∧ '7' ∈ (repr n).toList}

-- The proof statement to be proved
theorem percent_palindromes_with_7 : 
  let total := palindromes_1000_to_2000.toFinset.card in
  let count_with_7 := palindromes_with_7_1000_to_2000.toFinset.card in
  100 * count_with_7 / total = 190 :=
sorry

end percent_palindromes_with_7_l250_250129


namespace centroid_vector_sum_zero_projection_vector_l250_250022

/-- If point G is the centroid of triangle ABC,
    then the sum of the vectors from G to the vertices of the triangle is zero. -/
theorem centroid_vector_sum_zero
  {A B C G : ℝ × ℝ}
  (hG : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)):
  (G.1 - A.1, G.2 - A.2) + (G.1 - B.1, G.2 - B.2) + (G.1 - C.1, G.2 - C.2) = (0, 0) :=
by
  sorry

/-- The projection of vector b onto vector a is equal to (1/2) * a
    when a = (-1, 1) and b = (2, 3). -/
theorem projection_vector
  (a b : ℝ × ℝ)
  (ha : a = (-1, 1))
  (hb : b = (2, 3)):
  let proj_ab := ((((a.1 * b.1 + a.2 * b.2) / (a.1 * a.1 + a.2 * a.2)) * a.1,
                    ((a.1 * b.1 + a.2 * b.2) / (a.1 * a.1 + a.2 * a.2)) * a.2))
  in proj_ab = ((1 / 2) * a.1, (1 / 2) * a.2) :=
by
  sorry

end centroid_vector_sum_zero_projection_vector_l250_250022


namespace value_of_expression_when_x_is_2_l250_250894

theorem value_of_expression_when_x_is_2 : 
  (3 * 2 + 4) ^ 2 = 100 := 
by
  sorry

end value_of_expression_when_x_is_2_l250_250894


namespace money_left_eq_l250_250378

theorem money_left_eq :
  let initial_money := 56
  let notebooks := 7
  let cost_per_notebook := 4
  let books := 2
  let cost_per_book := 7
  let money_left := initial_money - (notebooks * cost_per_notebook + books * cost_per_book)
  money_left = 14 :=
by
  sorry

end money_left_eq_l250_250378


namespace min_value_fraction_l250_250636

theorem min_value_fraction {x : ℝ} (h : x > 8) : 
    ∃ c : ℝ, (∀ y : ℝ, y = (x^2) / ((x - 8)^2) → c ≤ y) ∧ c = 1 := 
sorry

end min_value_fraction_l250_250636


namespace triangle_segment_BL_length_l250_250491

noncomputable def length_BL : Float := 22.5

theorem triangle_segment_BL_length
  (A B C : Point)
  (ω : Circle)
  (M L : Point)
  (angle_A_right : angle A B C = 90)
  (side_BC : dist B C = 25)
  (area_ABC : area A B C = 150)
  (AB_gt_AC : dist A B > dist A C)
  (inscribed_circle : is_inscribed_in ω (Triangle.mk A B C))
  (tangent_M : is_tangency_point M AC ω)
  (BM_meets_L : meets_second_time (line_through B M) ω L) : 
  dist B L = length_BL := 
  sorry

end triangle_segment_BL_length_l250_250491


namespace ratio_QA_AR_l250_250543

theorem ratio_QA_AR {P Q R O C A B K L : Point} (hC : touches_circle_in_triangle P Q R O C A B)
  (hBO_CO : lines_intersect K L P Q R O B C) (hKQ : KQ = 3) (hQR : QR = 16) (hLR : LR = 1) :
  QA / AR = 9 / 7 :=
by
  sorry

end ratio_QA_AR_l250_250543


namespace smallest_prime_factor_1729_l250_250507

theorem smallest_prime_factor_1729 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ 1729 → p ≤ q :=
by
  exist 7
  sorry

end smallest_prime_factor_1729_l250_250507


namespace find_m_l250_250270

noncomputable def polar_to_cartesian_eq_c (rho theta : ℝ) : ℝ := 
  rho^2 = 4 * rho * cos theta

noncomputable def parametric_eq_l (t m : ℝ) : (ℝ × ℝ) := 
  (sqrt 2 / 2 * t + m, sqrt 2 / 2 * t)

theorem find_m (m : ℝ) (t : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (polar_to_cartesian_eq_c (sqrt (A.1^2 + A.2^2)) (atan (A.2 / A.1)))
    ∧ (parametric_eq_l t m = A ∨ parametric_eq_l t m = B)
    ∧ dist A B = sqrt 14) 
  → (m = 1 ∨ m = 3) :=
sorry

end find_m_l250_250270


namespace part1_max_min_part2_monotonic_increasing_intervals_l250_250814

/-- Definition of the function f(x) -/
def f (x : ℝ) (m : ℝ) : ℝ := - (1 / 3) * x ^ 3 + x ^ 2 + (m ^ 2 - 1) * x

/-- Part 1: Prove maximum and minimum values when m = 1 on the interval [-3, 2] -/
theorem part1_max_min (x : ℝ) : x ∈ Icc (-3 : ℝ) 2 → 
  (f x 1 ≤ 18 ∧ (∀ y, y ∈ Icc (-3 : ℝ) 2 → f y 1 ≤ f x 1) → (18 ≤ f x 1)) ∧
  ((0:ℝ) ≤ f x 1 ∧ (∀ y, y ∈ Icc (-3 : ℝ) 2 → f x 1 ≤ f y 1) → (f x 1 ≤ 0)) := 
sorry

/-- Part 2: Prove the function f(x) is monotonically increasing on the interval (1 - m, m + 1) -/
theorem part2_monotonic_increasing_intervals (x m : ℝ) : m > 0 → 
  x ∈ Ioo (1 - m) (m + 1) → 
  (∀ x1 x2, x1 < x2 → f x1 m ≤ f x2 m) := 
sorry 

end part1_max_min_part2_monotonic_increasing_intervals_l250_250814


namespace find_a_l250_250685

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 2 then 2 * real.exp (x - 1) else real.log (x^2 - a) / real.log 3

theorem find_a (a : ℝ) : 
  (f (f 1 a) a) = 2 ↔ a = -5 := by
  sorry

end find_a_l250_250685


namespace total_miles_run_l250_250095

theorem total_miles_run :
  ∀ (Xavier Katie Cole Lily Joe : ℝ),
  Xavier = 84 ∧
  Katie = (1 / 3) * Xavier ∧
  Cole = (1 / 4) * Katie ∧
  Lily = 5 * Cole ∧
  Joe = (1 / 2) * Lily ∧
  Lily = Joe - 0.15 * Joe →
  Xavier + Katie + Cole + Lily + Joe = 168.875 :=
begin
  sorry
end

end total_miles_run_l250_250095


namespace positive_square_factors_of_180_l250_250286

theorem positive_square_factors_of_180 :
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  factors = 4 :=
by
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  show factors = 4
  sorry

end positive_square_factors_of_180_l250_250286


namespace tangent_line_at_origin_l250_250414

noncomputable def f (a x : ℝ) : ℝ := x^3 + (a-1) * x^2 + a * x

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - f (x)

theorem tangent_line_at_origin (a : ℝ) (h : is_odd_function (f a)) :
  let f' := λ x, 3*x^2 + 1 in 
  (f' 0 = 1) → 
  (∀ x, f a x = x^3 + x) → 
  (f' 0 = 1) → 
  (∀ x : ℝ, x = 0 → y = 0 ∧ f' 0 * x = f a x - 0): 
    y = x :=
sorry

end tangent_line_at_origin_l250_250414


namespace trigonometric_quadrant_l250_250748

theorem trigonometric_quadrant (θ : Real) (h : sin θ * cos θ > 0) :
  θ ∈ {θ : Real | (0 < sin θ ∧ 0 < cos θ) ∨ (sin θ < 0 ∧ cos θ < 0)} :=
sorry

end trigonometric_quadrant_l250_250748


namespace count_valid_triangles_l250_250577

/-- 
Define the problem constraints: scalene triangles with side lengths a, b, c, 
where a < b < c, a + c = 2b, and a + b + c ≤ 30.
-/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + c = 2 * b ∧ a + b + c ≤ 30

/-- 
Statement of the problem: Prove that there are 20 distinct triangles satisfying the above constraints. 
-/
theorem count_valid_triangles : ∃ n, n = 20 ∧ (∀ {a b c : ℕ}, is_valid_triangle a b c → n = 20) :=
sorry

end count_valid_triangles_l250_250577


namespace find_a_c_l250_250868

theorem find_a_c (a c : ℝ) (h1 : a + c = 35) (h2 : a < c)
  (h3 : ∀ x : ℝ, a * x^2 + 30 * x + c = 0 → ∃! x, a * x^2 + 30 * x + c = 0) :
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) :=
by
  sorry

end find_a_c_l250_250868


namespace sequence_equality_l250_250368

open Nat

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  ∀ n, a (n + 2) = 2 * a (n + 1) - a n + 2

theorem sequence_equality (a : ℕ → ℤ) (h : sequence a) : 
  ∀ n, a n = n^2 - 2 * n + 2 :=
by
  sorry

end sequence_equality_l250_250368


namespace john_loses_probability_eq_3_over_5_l250_250867

-- Definitions used directly from the conditions in a)
def probability_win := 2 / 5
def probability_lose := 1 - probability_win

-- The theorem statement
theorem john_loses_probability_eq_3_over_5 : 
  probability_lose = 3 / 5 := 
by
  sorry -- proof is to be filled in later

end john_loses_probability_eq_3_over_5_l250_250867


namespace find_third_number_l250_250534

theorem find_third_number (x : ℝ) : 3 + 33 + x + 3.33 = 369.63 → x = 330.30 :=
by
  intros h
  sorry

end find_third_number_l250_250534


namespace sugar_solution_l250_250540

theorem sugar_solution (V x : ℝ) (h1 : V > 0) (h2 : 0.1 * (V - x) + 0.5 * x = 0.2 * V) : x / V = 1 / 4 :=
by sorry

end sugar_solution_l250_250540


namespace greatest_possible_x_l250_250886

theorem greatest_possible_x (x : ℕ) (h : x^3 < 15) : x ≤ 2 := by
  sorry

end greatest_possible_x_l250_250886


namespace sin_value_l250_250654

theorem sin_value (α : ℝ) (hα : 0 < α ∧ α < π) 
  (hcos : Real.cos (α + Real.pi / 6) = -3 / 5) : 
  Real.sin (2 * α + Real.pi / 12) = -17 * Real.sqrt 2 / 50 := 
sorry

end sin_value_l250_250654


namespace triangle_non_existence_triangle_existence_l250_250526

-- Definition of the triangle inequality theorem for a triangle with given sides.
def triangle_exists (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_non_existence (h : ¬ triangle_exists 2 3 7) : true := by
  sorry

theorem triangle_existence (h : triangle_exists 5 5 5) : true := by
  sorry

end triangle_non_existence_triangle_existence_l250_250526


namespace hexagon_circle_area_ratio_l250_250556

theorem hexagon_circle_area_ratio (r : ℝ) (h : r = 6) :
  let A_hexagon := 6 * (sqrt 3 / 4 * r^2)
  let A_circle := π * r^2
  (A_hexagon / A_circle = 3 * sqrt 3 / (8 * π)) :=
by sorry

end hexagon_circle_area_ratio_l250_250556


namespace find_b_l250_250595

noncomputable def cycle_period_condition (a b c d : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) : Prop :=
  (∃ f : ℝ → ℝ, f = (λ x, a * Real.sin (b * x + c) + d) ∧ ∀ x, f (x + 2 * Real.pi / b) = f x)

theorem find_b (a b c d : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d)
  (h : cycle_period_condition a b c d h_a h_b h_c h_d) :
  b = 5 :=
sorry

end find_b_l250_250595


namespace count_integers_in_interval_l250_250734

theorem count_integers_in_interval :
  {n : ℤ | -5 < n ∧ n < 3}.finite ∧ {n : ℤ | -5 < n ∧ n < 3}.to_finset.card = 7 := by
sorry

end count_integers_in_interval_l250_250734


namespace sum_of_consecutive_odds_divisible_by_count_l250_250527

theorem sum_of_consecutive_odds_divisible_by_count (n : ℕ) :
  let seq := list.range (2 * n + 1).map (λ i, 2 * i + 3)
  in (list.sum seq) % (2 * n + 1) = 0 :=
by
  sorry

end sum_of_consecutive_odds_divisible_by_count_l250_250527


namespace find_k_find_Q_find_min_f_l250_250141

-- Defining the conditions given in the problem
def P (k : ℝ) (x : ℝ) : ℝ := 1 + k / x
def Q (x : ℝ) : ℝ := 125 - abs (x - 25)
def f (k : ℝ) (x : ℝ) : ℝ := P k x * Q x

-- Lean 4 statement for problem (1): finding k
theorem find_k (k : ℝ) (h : f k 10 = 121) (Q10_eq : Q 10 = 110) :
  k = 1 := 
sorry

-- Lean 4 statement for problem (2): finding Q(x)
theorem find_Q (Q_table_values : ∀ x ∈ {10, 20, 25, 30}, Q x ∈ {110, 120, 125, 120}) :
  Q = λ x, 125 - abs (x - 25) := 
sorry

-- Lean 4 statement for problem (3): finding minimum value of f(x)
theorem find_min_f (Q_expr : ∀ x, Q x = 125 - abs (x - 25)) : 
  ∃ x ∈ set.Ico (1 : ℝ) (31 : ℝ), f 1 x = 121 :=
sorry

end find_k_find_Q_find_min_f_l250_250141


namespace number_of_cuboids_intersected_diagonal_l250_250531

theorem number_of_cuboids_intersected_diagonal :
  -- Conditions 
  (∀ cuboids : set (set ℕ), ∀ length width height : ℕ, length = 2 ∧ width = 3 ∧ height = 5 ->
  (∃ cube : set ℕ, ∀ side : ℕ, side = 90 -> 
  -- Assertion
  (∃ n : ℕ, n = 66))) := sorry

end number_of_cuboids_intersected_diagonal_l250_250531


namespace count_integers_n_satisfying_inequality_l250_250717

theorem count_integers_n_satisfying_inequality :
  {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset.card = 7 :=
by sorry

end count_integers_n_satisfying_inequality_l250_250717


namespace positive_square_factors_of_180_l250_250287

theorem positive_square_factors_of_180 :
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  factors = 4 :=
by
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  show factors = 4
  sorry

end positive_square_factors_of_180_l250_250287


namespace finite_nonempty_set_of_positive_integers_l250_250182

theorem finite_nonempty_set_of_positive_integers (S : Set ℕ) (hS1 : S.Nonempty) (hS2 : S.Finite)
  (hS3 : ∀ i j ∈ S, ((i + j) / (Nat.gcd i j)) ∈ S) :
  (∃ n : ℕ, S = {n} ∨ (n > 2 ∧ S = {n, n * (n - 1)})) :=
by
  sorry

end finite_nonempty_set_of_positive_integers_l250_250182


namespace T_point_sequence_example_obtuse_triangle_example_vector_inequality_l250_250359

noncomputable def Point (n : ℕ) (a_n : ℝ) : ℝ × ℝ := (n, a_n)

def b_n (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (a (n+1)) - a n

def T_point_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, b_n (n+1) a > b_n n a

def specific_sequence (n : ℕ) : ℝ :=
  1 / n

theorem T_point_sequence_example :
  T_point_seq specific_sequence :=
  sorry

def obtuse_triangle (a : ℕ → ℝ) (k : ℕ) : Prop :=
  (a (k+2) - a (k+1)) * (a k - a (k+1)) < 0

theorem obtuse_triangle_example (k : ℕ) :
  T_point_seq specific_sequence →
  1 / 2 - 1 > 0 →
  obtuse_triangle specific_sequence k :=
  sorry

theorem vector_inequality (m n p q : ℕ) (hm : 1 ≤ m) (hn : m < n) (hp : n < p) (hq : p < q) (h_eq : m + q = n + p) :
  T_point_seq specific_sequence →
  ∑ i in finrange (q-p), (b_n (p+i) specific_sequence) > ∑ i in finrange (n-m), (b_n (m+i) specific_sequence) :=
  sorry

end T_point_sequence_example_obtuse_triangle_example_vector_inequality_l250_250359


namespace triangle_exists_l250_250177

theorem triangle_exists (x : ℕ) (hx : x > 0) :
  (3 * x + 10 > x * x) ∧ (x * x + 10 > 3 * x) ∧ (x * x + 3 * x > 10) ↔ (x = 3 ∨ x = 4) :=
by
  sorry

end triangle_exists_l250_250177


namespace probability_three_cards_same_l250_250490

-- Let A, B, C each send a card to D or E, with each choice being equally likely.
noncomputable def probability_all_send_same (p : ℕ) : ℚ :=
  if h : p = 1 then 1 else sorry

theorem probability_three_cards_same :
  probability_all_send_same 2 = 1 / 4 :=
by
  sorry

end probability_three_cards_same_l250_250490


namespace average_weight_l250_250040

def weights (A B C : ℝ) : Prop :=
  (A + B + C = 135) ∧
  (B + C = 86) ∧
  (B = 31)

theorem average_weight (A B C : ℝ) (h : weights A B C) :
  (A + B) / 2 = 40 :=
by
  sorry

end average_weight_l250_250040


namespace find_constant_A_l250_250845

theorem find_constant_A :
  ∀ (x : ℝ)
  (A B C D : ℝ),
      (
        (1 : ℝ) / (x^4 - 20 * x^3 + 147 * x^2 - 490 * x + 588) = 
        (A / (x + 3)) + (B / (x - 4)) + (C / ((x - 4)^2)) + (D / (x - 7))
      ) →
      A = - (1 / 490) := 
by 
  intro x A B C D h
  sorry

end find_constant_A_l250_250845


namespace transformed_function_equivalence_l250_250050

-- Define the original function
def original_function (x : ℝ) : ℝ := 2 * x + 1

-- Define the transformation involving shifting 2 units to the right
def transformed_function (x : ℝ) : ℝ := original_function (x - 2)

-- The theorem we want to prove
theorem transformed_function_equivalence : 
  ∀ x : ℝ, transformed_function x = 2 * x - 3 :=
by
  sorry

end transformed_function_equivalence_l250_250050


namespace triangles_hyperbola_parallel_l250_250573

variable (a b c a1 b1 c1 : ℝ)

-- Defining the property that all vertices lie on the hyperbola y = 1/x
def on_hyperbola (x : ℝ) (y : ℝ) : Prop := y = 1 / x

-- Defining the parallelism condition for line segments
def parallel (slope1 slope2 : ℝ) : Prop := slope1 = slope2

theorem triangles_hyperbola_parallel
  (H1A : on_hyperbola a (1 / a))
  (H1B : on_hyperbola b (1 / b))
  (H1C : on_hyperbola c (1 / c))
  (H2A : on_hyperbola a1 (1 / a1))
  (H2B : on_hyperbola b1 (1 / b1))
  (H2C : on_hyperbola c1 (1 / c1))
  (H_AB_parallel_A1B1 : parallel ((b - a) / (a * b * (a - b))) ((b1 - a1) / (a1 * b1 * (a1 - b1))))
  (H_BC_parallel_B1C1 : parallel ((c - b) / (b * c * (b - c))) ((c1 - b1) / (b1 * c1 * (b1 - c1)))) :
  parallel ((c1 - a) / (a * c1 * (a - c1))) ((c - a1) / (a1 * c * (a1 - c))) :=
sorry

end triangles_hyperbola_parallel_l250_250573


namespace lines_intersection_l250_250055

theorem lines_intersection (a b : ℝ) : 
  (2 : ℝ) = (1/3 : ℝ) * (1 : ℝ) + a →
  (1 : ℝ) = (1/3 : ℝ) * (2 : ℝ) + b →
  a + b = 2 := 
by
  intros h₁ h₂
  sorry

end lines_intersection_l250_250055


namespace floor_5_7_l250_250198

theorem floor_5_7 : Int.floor 5.7 = 5 :=
by
  sorry

end floor_5_7_l250_250198


namespace hexagon_fills_ground_l250_250092

theorem hexagon_fills_ground : 
  let n := 6
  let interior_angle := (n - 2) * 180 / n
  interior_angle * 3 = 360 :=
by
  let n := 6
  let interior_angle := (n - 2) * 180 / n
  have h : interior_angle = 120 := by sorry
  calc
    interior_angle * 3 = 120 * 3 : by rw h
                ...  = 360       : by norm_num

end hexagon_fills_ground_l250_250092


namespace sylvie_turtle_weight_l250_250445

theorem sylvie_turtle_weight
  (ounces_per_half_pound : ℕ)
  (ounce_per_half_pound_needs : ounces_per_half_pound = 1)
  (ounces_per_jar : ℕ)
  (ounces_per_jar_value : ounces_per_jar = 15)
  (dollars_per_jar : ℕ)
  (dollars_per_jar_value : dollars_per_jar = 2)
  (total_cost : ℕ)
  (total_cost_value : total_cost = 8) :
  30 = ((total_cost / dollars_per_jar) * ounces_per_jar) * (1 : ℚ) / ounces_per_half_pound * (1/2 : ℚ) := 
by 
  rw [total_cost_value, dollars_per_jar_value, ounces_per_jar_value, ounce_per_half_pound_needs],
  norm_num,
so sorry

end sylvie_turtle_weight_l250_250445


namespace largest_square_side_length_l250_250554

theorem largest_square_side_length (smallest_square_side next_square_side : ℕ) (h1 : smallest_square_side = 1) 
(h2 : next_square_side = smallest_square_side + 6) :
  ∃ x : ℕ, x = 7 :=
by
  existsi 7
  sorry

end largest_square_side_length_l250_250554


namespace workshop_workers_count_l250_250098

theorem workshop_workers_count 
  (W : ℕ) (N : ℕ) 
  (average_all : W * 8000)
  (average_technicians : 7 * 12000) 
  (average_rest : N * 6000) 
  (total_salary : W * 8000 = 7 * 12000 + N * 6000)
  (total_workers : W = 7 + N) : 
  W = 21 := 
sorry

end workshop_workers_count_l250_250098


namespace symmetric_point_x_axis_l250_250667

theorem symmetric_point_x_axis (P Q : ℝ × ℝ) (hP : P = (-1, 2)) (hQ : Q = (P.1, -P.2)) : Q = (-1, -2) :=
sorry

end symmetric_point_x_axis_l250_250667


namespace min_a_plus_b_l250_250280

theorem min_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 2 / b = 2) : a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
begin
  sorry
end

end min_a_plus_b_l250_250280


namespace line_symmetric_fixed_point_l250_250336

theorem line_symmetric_fixed_point (k : ℝ) :
  let l1 := λ (x : ℝ), (k - 1) * x + 2
  let l2 := λ (x : ℝ), (-1) * (x - 1) / (k - 1) + 3 - 2 / (k - 1)
  (∀ x, l1 x = (k - 1) * x + 2) →
  (∀ x, l1 x = 2 → x = 0) →
  (∀ x, l2 x = 3 - x) →
  l2 1 = 1 :=
by
  sorry

end line_symmetric_fixed_point_l250_250336


namespace area_of_triangle_ABC_find_AC_given_angle_B_l250_250143

open Real

variables {A B C O P T K : Point} {ω Ω : Circle}

-- Given conditions
axiom hABC_acute : AcuteTriangle A B C
axiom hCircumcircle : Circle ω ∧ IsCircumcircleOf ω A B C ∧ CenterOfCircle ω O
axiom hCircleAOC : Circle Ω ∧ PassesThroughPoints Ω A O C
axiom hP : ∃ P, IntersectsSegment Ω B C P
axiom hTangents : TangentToCircle ω A T ∧ TangentToCircle ω C T
axiom hTP_intersect : IntersectAtSegment T P C A K
axiom hAreas : AreaOfTriangle A P K = 6 ∧ AreaOfTriangle C P K = 4

-- Questions
theorem area_of_triangle_ABC : AreaOfTriangle A B C = 25 := by
  sorry

theorem find_AC_given_angle_B :
  ∀ β : ℝ, β = arctan (7 / 5) → 
  ∃ AC : ℝ, TakesValueForAngle A B C β AC := by 
  sorry

end area_of_triangle_ABC_find_AC_given_angle_B_l250_250143


namespace largest_remainder_two_digit_sum_l250_250085

theorem largest_remainder_two_digit_sum (n : ℕ) (h1 : 10 ≤ n) (h2 : n < 100) : 
  let s := (n / 10) + (n % 10) in (∃ (k r : ℕ), (n = k * s + r) ∧ (r < s)) → r ≤ 15 :=
by
  sorry

end largest_remainder_two_digit_sum_l250_250085


namespace pascal_row_with_ratio_456_exists_at_98_l250_250350

theorem pascal_row_with_ratio_456_exists_at_98 :
  ∃ n, ∃ r, 0 ≤ r ∧ r + 2 ≤ n ∧ 
  ((Nat.choose n r : ℚ) / Nat.choose n (r + 1) = 4 / 5) ∧
  ((Nat.choose n (r + 1) : ℚ) / Nat.choose n (r + 2) = 5 / 6) ∧ 
  n = 98 := by
  sorry

end pascal_row_with_ratio_456_exists_at_98_l250_250350


namespace rita_bought_four_pounds_l250_250835

def initial_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def left_amount : ℝ := 35.68

theorem rita_bought_four_pounds :
  (initial_amount - left_amount) / cost_per_pound = 4 :=
by
  sorry

end rita_bought_four_pounds_l250_250835


namespace paint_cube_faces_l250_250156

noncomputable def paintCubeDistinctWays : ℕ :=
  let colors := { "blue", "green", "red" }
  let num_faces := 6
  let pairs := num_faces / 2
  -- Here we encode the question of counting distinct colorings considering rotations.
  6 -- This should correspond to the rigorous mathematical proof in actual practice.
  
theorem paint_cube_faces :
  ∃ (paintCubeDistinctWays : ℕ), 
    paintCubeDistinctWays = 6 := 
by 
  use 6
  -- The rigorous proof of this statement would be fleshed out in practice.
  sorry

end paint_cube_faces_l250_250156


namespace length_of_BC_l250_250968

-- Definitions of the triangle and its properties
def Triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180 ∧ 0 < α ∧ α < 180 ∧ 0 < β ∧ β < 180 ∧ 0 < γ ∧ γ < 180

-- Conditions
def right_triangle {A B C : Type} (angle_BAC : ℝ) (angle_ABC : ℝ) (AC : ℝ) : Prop :=
  Triangle angle_BAC angle_ABC (180 - angle_BAC - angle_ABC) ∧
  angle_BAC = 90 ∧ angle_ABC = 60 ∧ AC = 6 * Real.sqrt 3

-- The theorem to prove
theorem length_of_BC (A B C : Type) (angle_BAC angle_ABC AC : ℝ) 
  (h : right_triangle angle_BAC angle_ABC AC) : 
  let BC := Real.sqrt 3 * AC in 
  BC = 18 :=
by
  sorry

end length_of_BC_l250_250968


namespace sum_not_zero_l250_250476

theorem sum_not_zero (a b c d : ℝ) (h1 : a * b * c - d = 1) (h2 : b * c * d - a = 2) 
  (h3 : c * d * a - b = 3) (h4 : d * a * b - c = -6) : a + b + c + d ≠ 0 :=
sorry

end sum_not_zero_l250_250476


namespace two_digit_numbers_with_ones_greater_than_tens_l250_250741

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def ones_digit_greater_than_tens_digit (n : ℕ) : Prop := (n / 10) < (n % 10)

-- Define the problem
theorem two_digit_numbers_with_ones_greater_than_tens : 
  { n : ℕ | is_two_digit n ∧ ones_digit_greater_than_tens_digit n }.to_finset.card = 36 :=
by
  sorry

end two_digit_numbers_with_ones_greater_than_tens_l250_250741


namespace polynomial_relation_l250_250271

def M (m : ℚ) : ℚ := 5 * m^2 - 8 * m + 1
def N (m : ℚ) : ℚ := 4 * m^2 - 8 * m - 1

theorem polynomial_relation (m : ℚ) : M m > N m := by
  sorry

end polynomial_relation_l250_250271


namespace area_triangle_ADE_l250_250145

variables {ABC : Type} [triangle ABC]
variables (A B C D E : point)

-- Conditions of the problem
variables [is_equilateral ABC]
variables [area ABC = 27 * sqrt 3]
variables [trisecting_rays (∠ BAC) [D E]]

-- The statement we need to prove
theorem area_triangle_ADE : area (triangle A D E) = 3 * sqrt 3 := 
sorry

end area_triangle_ADE_l250_250145


namespace number_of_dogs_is_approximately_138_l250_250353

noncomputable def dog_ratio : ℝ := 4.5
noncomputable def bunny_ratio : ℝ := 9.8
noncomputable def parrot_ratio : ℝ := 12.2
noncomputable def total_animals : ℝ := 815
noncomputable def total_ratio : ℝ := dog_ratio + bunny_ratio + parrot_ratio
noncomputable def ratio_unit : ℝ := total_animals / total_ratio
noncomputable def num_dogs : ℝ := dog_ratio * ratio_unit

theorem number_of_dogs_is_approximately_138 :
  num_dogs ≈ 138 := sorry

end number_of_dogs_is_approximately_138_l250_250353


namespace irrationals_make_f_zero_l250_250657

def f (x : ℝ) : ℤ :=
  if x > 0 then 1 else if x = 0 then 0 else -1

def g (x : ℝ) : ℤ :=
  if irrational x then 0 else 1

theorem irrationals_make_f_zero (a : ℝ) (h : f (g a) = 0) : irrational a :=
by
  sorry

end irrationals_make_f_zero_l250_250657


namespace daily_savings_amount_l250_250970

theorem daily_savings_amount (total_savings : ℕ) (days : ℕ) (daily_savings : ℕ)
  (h1 : total_savings = 12410)
  (h2 : days = 365)
  (h3 : total_savings = daily_savings * days) :
  daily_savings = 34 :=
sorry

end daily_savings_amount_l250_250970


namespace probability_multiple_of_4_l250_250053

theorem probability_multiple_of_4 :
  let M := {m : ℤ | 0 ≤ m ∧ m ≤ 7}
  let N := {n : ℤ | 2 ≤ n ∧ n ≤ 9}
  let pairs := {p : ℤ × ℤ | (p.1 ∈ M) ∧ (p.2 ∈ N)}
  let valid_pairs := {p : ℤ × ℤ | p ∈ pairs ∧ (2 * p.1 * p.2) % 4 = 0}
  (valid_pairs.to_finset.card : ℚ) / (pairs.to_finset.card : ℚ) = 3 / 4 :=
by
  let M := {m : ℤ | 0 ≤ m ∧ m ≤ 7}
  let N := {n : ℤ | 2 ≤ n ∧ n ≤ 9}
  let pairs := {p : ℤ × ℤ | (p.1 ∈ M) ∧ (p.2 ∈ N)}
  let valid_pairs := {p : ℤ × ℤ | p ∈ pairs ∧ (2 * p.1 * p.2) % 4 = 0}
  have total_pairs : pairs.to_finset.card = 64 := sorry
  have valid_pairs_count : valid_pairs.to_finset.card = 48 := sorry
  show (valid_pairs.to_finset.card : ℚ) / (pairs.to_finset.card : ℚ) = 3 / 4,
  from sorry

end probability_multiple_of_4_l250_250053


namespace side_length_of_square_l250_250036

theorem side_length_of_square :
  ∀ (d1 d2 : ℝ) (s : ℝ),
    d1 = 16 →
    d2 = 8 →
    (d1 * d2) / 2 = s^2 →
    s = 8 :=
by
  intros d1 d2 s h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  norm_num at h3
  linarith

end side_length_of_square_l250_250036


namespace binomial_largest_coefficient_l250_250256

theorem binomial_largest_coefficient :
  ∃ (k : ℕ), (k ≤ 6) ∧ (∀ (m : ℕ), m ≤ 6 → (binomial 6 k) * (2 ^ k) ≥ (binomial 6 m) * (2 ^ m)) ∧ 
  (binomial 6 k) * (2 ^ k) = 160 := by
  sorry

end binomial_largest_coefficient_l250_250256


namespace arranging_12225_l250_250914

def digits := [1, 2, 2, 2, 5]

def is_four_digit_multiple_of_5 (n : ℕ) : Prop :=
  999 < n ∧ n < 10000 ∧ (n % 5 = 0)

def count_valid_permutations : ℕ :=
  ((list.permutations digits).filter (λ perm, let n := perm.take 4 in is_four_digit_multiple_of_5 (nat.digits 10 n))).length

theorem arranging_12225 : count_valid_permutations = 4 := by
  sorry

end arranging_12225_l250_250914


namespace part_a_part_b_l250_250488

-- Condition: Definition of t_n (number of ways to tile 1 × n plot)
def t_n (n : Nat) : Nat := sorry  -- Definition of t_n, placeholder here

-- Part (a) statement
theorem part_a (n : Nat) (h : n > 1) : t_n (2 * n + 1) = t_n n * (t_n (n - 1) + t_n (n + 1)) := sorry

-- Additional binomial coefficient definition for part (b)
def binom (m r : Nat) : Nat :=
if h : 0 ≤ r ∧ r ≤ m then (Nat.factorial m) / ((Nat.factorial r) * (Nat.factorial (m - r)))
else 0

-- Part (b) statement
theorem part_b (n : Nat) (h : n > 0) : t_n n = ∑ d in Finset.range (n / 2 + 1), binom (n - d) d * 2^((n-2*d) : Nat) := sorry

end part_a_part_b_l250_250488


namespace find_P_coordinates_find_minor_arc_length_line_MN_through_fixed_point_l250_250681

-- Circle definition
def Circle (radius : ℝ) (center : ℝ × ℝ) := 
  { p : ℝ × ℝ // (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

-- Problem definitions
def circle_O : Circle 2 (0, 0) := ⟨(0, 0), by norm_num⟩
def line_l (P : ℝ × ℝ) := P.1 = 4

-- Given conditions
def P_on_line_l (P : ℝ × ℝ) : Prop := line_l P
def distance_P_to_tangency (P C : ℝ × ℝ) : Prop :=
  real.dist P C = 2 * real.sqrt 3 ∧
  tangent_to_circle O P C

-- Goals to prove
theorem find_P_coordinates (P : ℝ × ℝ) 
  (h1 : P_on_line_l P)
  (h2 : ∃ C, distance_P_to_tangency P C) :
  P = (4, 0) :=
sorry

theorem find_minor_arc_length (P : ℝ × ℝ) 
  (h1 : P = (4, 0)) :
  minor_arc_length P circle_O = (4 * real.pi) / 3 :=
sorry

theorem line_MN_through_fixed_point (P A B M N : ℝ × ℝ)
  (h1 : A = (-2, 0) ∧ B = (2, 0))
  (h2 : ∃ Q, Q = (1, 0))
  (h3 : intersect_circle circle_O P A M)
  (h4 : intersect_circle circle_O P B N) :
  line_through_points M N (1, 0) :=
sorry

end find_P_coordinates_find_minor_arc_length_line_MN_through_fixed_point_l250_250681


namespace opposite_of_pi_is_neg_pi_l250_250862

-- Definition that the opposite of a number x is -1 * x
def opposite (x : ℝ) : ℝ := -1 * x

-- Theorem stating that the opposite of π is -π
theorem opposite_of_pi_is_neg_pi : opposite π = -π := 
  sorry

end opposite_of_pi_is_neg_pi_l250_250862


namespace number_of_perfect_square_factors_of_180_l250_250297

-- Define the prime factorization of 180
def prime_factorization_180 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Define what it means to be a factor of 180
def is_factor_of_180 (n : ℕ) : Prop :=
  ∀ p k, (p, k) ∈ prime_factorization_180 → ∃ m, n = p ^ m ∧ m ≤ k

-- Define what it means to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p m, ∃ k, n = p ^ m ∧ even m

-- Function to count factors of 180 that are perfect squares
def count_perfect_square_factors_of_180 : ℕ :=
  List.length [d | d ← List.range 181, is_factor_of_180 d ∧ is_perfect_square d]

-- Main theorem
theorem number_of_perfect_square_factors_of_180 : count_perfect_square_factors_of_180 = 4 := 
by sorry

end number_of_perfect_square_factors_of_180_l250_250297


namespace A1O_bisects_KL_l250_250279

-- Define the points and midpoints
variables {A B C A_1 B_1 C_1 K L O : Type*}

-- Define the geometric conditions
def is_midpoint (P Q R : Type*) : Prop := sorry
def is_perpendicular_foot (P Q R : Type*) : Prop := sorry
def nine_point_circle_center (P Q R S : Type*) : Prop := sorry
def bisects_segment (P Q R : Type*) : Prop := sorry

-- Given Conditions
axiom h1 : is_midpoint A_1 B C
axiom h2 : is_midpoint B_1 C A
axiom h3 : is_midpoint C_1 A B
axiom h4 : is_perpendicular_foot K B A_1 C_1
axiom h5 : is_perpendicular_foot L C A_1 B_1
axiom h6 : nine_point_circle_center O A B C

-- The proof statement
theorem A1O_bisects_KL : bisects_segment A_1 O K L := 
sorry

end A1O_bisects_KL_l250_250279


namespace percentage_of_palindromes_with_seven_is_10_percent_l250_250124

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def has_seven (n : ℕ) : Prop :=
  ('7' ∈ n.to_string)

def count_eligible_palindromes : ℕ :=
  let candidates := finset.filter (λ n, 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n) (finset.range 2000)
  finset.card candidates

def count_palindromes_with_seven : ℕ :=
  let qualified := finset.filter (λ n, 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n ∧ has_seven n) (finset.range 2000)
  finset.card qualified

theorem percentage_of_palindromes_with_seven_is_10_percent :
  100 * count_palindromes_with_seven / count_eligible_palindromes = 10 :=
by sorry

end percentage_of_palindromes_with_seven_is_10_percent_l250_250124


namespace part1_max_min_part2_monotonic_increasing_intervals_l250_250815

/-- Definition of the function f(x) -/
def f (x : ℝ) (m : ℝ) : ℝ := - (1 / 3) * x ^ 3 + x ^ 2 + (m ^ 2 - 1) * x

/-- Part 1: Prove maximum and minimum values when m = 1 on the interval [-3, 2] -/
theorem part1_max_min (x : ℝ) : x ∈ Icc (-3 : ℝ) 2 → 
  (f x 1 ≤ 18 ∧ (∀ y, y ∈ Icc (-3 : ℝ) 2 → f y 1 ≤ f x 1) → (18 ≤ f x 1)) ∧
  ((0:ℝ) ≤ f x 1 ∧ (∀ y, y ∈ Icc (-3 : ℝ) 2 → f x 1 ≤ f y 1) → (f x 1 ≤ 0)) := 
sorry

/-- Part 2: Prove the function f(x) is monotonically increasing on the interval (1 - m, m + 1) -/
theorem part2_monotonic_increasing_intervals (x m : ℝ) : m > 0 → 
  x ∈ Ioo (1 - m) (m + 1) → 
  (∀ x1 x2, x1 < x2 → f x1 m ≤ f x2 m) := 
sorry 

end part1_max_min_part2_monotonic_increasing_intervals_l250_250815


namespace left_handed_jazz_lovers_count_l250_250764

variables (C : Type) [Fintype C] (L J R : Finset C)
variables (hC : Fintype.card C = 25)
          (hL : L.card = 10)
          (hJ : J.card = 18)
          (hRJ' : (R ∩ Jᶜ).card = 4)
          (hDisj : ∀ x : C, x ∈ L ↔ ¬ x ∈ R)

theorem left_handed_jazz_lovers_count : (L ∩ J).card = 7 :=
by { sorry }

end left_handed_jazz_lovers_count_l250_250764


namespace intersection_of_M_and_N_l250_250278

noncomputable def M : Set (ℝ × ℝ) := 
  {p | ∃ λ : ℝ, p = (1 + 3 * λ, 2 + 4 * λ)}

noncomputable def N : Set (ℝ × ℝ) := 
  {p | ∃ λ : ℝ, p = (-2 + 4 * λ, -2 + 5 * λ)}

theorem intersection_of_M_and_N :
  M ∩ N = {(-2, -2)} :=
sorry

end intersection_of_M_and_N_l250_250278


namespace stockholm_to_uppsala_distance_l250_250044

/-- The real distance between the two cities, Stockholm and Uppsala, given the map distance
    and the scale. -/
theorem stockholm_to_uppsala_distance
    (map_distance : ℝ) (scale_factor : ℝ) (real_distance : ℝ) :
    map_distance = 35 → scale_factor = 10 → real_distance = map_distance * scale_factor → real_distance = 350 := 
by
  intros h_map_distance h_scale_factor h_real_distance
  rw [h_map_distance, h_scale_factor] at h_real_distance
  exact h_real_distance

end stockholm_to_uppsala_distance_l250_250044


namespace morse_code_symbol_count_up_to_5_is_62_l250_250349

-- Define the total number of Morse code symbols that can be represented
-- using sequences up to 5 dots and/or dashes.
def morse_code_symbol_count_up_to_5 : ℕ :=
  let sequences_length_1 := 2^1 in
  let sequences_length_2 := 2^2 in
  let sequences_length_3 := 2^3 in
  let sequences_length_4 := 2^4 in
  let sequences_length_5 := 2^5 in
  sequences_length_1 + sequences_length_2 + sequences_length_3 + sequences_length_4 + sequences_length_5

-- The theorem we wish to prove
theorem morse_code_symbol_count_up_to_5_is_62 : 
  morse_code_symbol_count_up_to_5 = 62 := by
  sorry

end morse_code_symbol_count_up_to_5_is_62_l250_250349


namespace sequence_formula_l250_250611

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1 else if n = 1 then 1 else
  let rec aux (m : ℕ) : ℕ :=
    if m = 2 then (√(sequence 2-2 * sequence 2-1) - √(sequence 2-1 * sequence 2-2)) / (2 * sequence 2 -1)
    else aux (m - 1)
  in aux n

theorem sequence_formula (n : ℕ) :
  (if n = 0 then sequence n = 1
  else if n = 1 then sequence n = 1
  else sequence n = ∏ i in range (n + 1).filter (λ i, i ≠ 0), (2 ^ i - 1) ^ 2) :=
sorry

end sequence_formula_l250_250611


namespace solve_eq1_solve_eq2_solve_eq3_l250_250441

-- Proof problem 1: Prove that x = 3.9 is the solution to 4x + x = 19.5
theorem solve_eq1 : ∀ x : ℝ, 4 * x + x = 19.5 ↔ x = 3.9 :=
begin
  sorry
end

-- Proof problem 2: Prove that x = 4 is the solution to 26.4 - 3x = 14.4
theorem solve_eq2 : ∀ x : ℝ, 26.4 - 3 * x = 14.4 ↔ x = 4 :=
begin
  sorry
end

-- Proof problem 3: Prove that x = 0.9 is the solution to 2x - 1 = 0.8
theorem solve_eq3 : ∀ x : ℝ, 2 * x - 1 = 0.8 ↔ x = 0.9 :=
begin
  sorry
end

end solve_eq1_solve_eq2_solve_eq3_l250_250441


namespace parking_ways_l250_250933

theorem parking_ways (n k : ℕ) (h_eq : n = 8) (h_car : k = 4) :
  (∃ num_ways : ℕ, num_ways = (Nat.choose 5 4 * 4! * 1!)) → 
  (num_ways = 120) := 
by
  intros
  sorry

end parking_ways_l250_250933


namespace total_number_of_coins_l250_250537

theorem total_number_of_coins (num_5c : Nat) (num_10c : Nat) (h1 : num_5c = 16) (h2 : num_10c = 16) : num_5c + num_10c = 32 := by
  sorry

end total_number_of_coins_l250_250537


namespace volume_of_doubled_cylinder_l250_250545

-- Given definitions: 
def original_radius : ℝ := 8
def original_height : ℝ := 15
def new_radius := 2 * original_radius
def new_height := 2 * original_height

-- Volume of a cylinder formula
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- Proof problem: The volume of the new cylinder with doubled dimensions
theorem volume_of_doubled_cylinder :
  volume new_radius new_height = 7680 * π :=
by
  sorry

end volume_of_doubled_cylinder_l250_250545


namespace no_convex_27_gon_with_distinct_integer_angles_l250_250978

theorem no_convex_27_gon_with_distinct_integer_angles :
  ¬ ∃ (polygon : Finset ℕ), (polygon.card = 27) ∧ 
  (∀ (a : ℕ) (b : ℕ), a ∈ polygon ∧ b ∈ polygon ∧ a ≠ b → a ≠ b) ∧ 
  (∀ (a : ℕ), a ∈ polygon → a ≥ 1 ∧ a ≤ 27) ∧ 
  (∑ x in polygon, x = 360) :=
by
  -- The proof would inductively verify the calculations leading to the impossibility.
  sorry

end no_convex_27_gon_with_distinct_integer_angles_l250_250978


namespace integer_range_l250_250973

theorem integer_range (n : ℤ) : 
  (∃ (k : ℤ), k ≥ 2 ∧ ∃ (x : Fin k → ℕ), 
    (∀ i, 1 ≤ x i) ∧
    (∑ i in Finset.range (k.toNat), x i = 2019) ∧ 
    (∑ i in Finset.range (k.toNat - 1), x i * x (i+1) = n)) → 
  (2018 ≤ n ∧ n ≤ 1009 * 1010) :=
sorry

end integer_range_l250_250973


namespace intersection_A_B_l250_250275

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {1, 3} := 
by 
  sorry

end intersection_A_B_l250_250275


namespace find_x_value_l250_250775

noncomputable def x_value (x y z : ℝ) : Prop :=
  (26 = (z + x) / 2) ∧
  (z = 52 - x) ∧
  (52 - x = (26 + y) / 2) ∧
  (y = 78 - 2 * x) ∧
  (78 - 2 * x = (8 + (52 - x)) / 2) ∧
  (x = 32)

theorem find_x_value : ∃ x y z : ℝ, x_value x y z :=
by
  use 32  -- x
  use 14  -- y derived from 78 - 2x where x = 32 leads to y = 14
  use 20  -- z derived from 52 - x where x = 32 leads to z = 20
  unfold x_value
  simp
  sorry

end find_x_value_l250_250775


namespace no_nat_n_exists_l250_250186

theorem no_nat_n_exists (n : ℕ) : ¬ ∃ n, ∃ k, n ^ 2012 - 1 = 2 ^ k := by
  sorry

end no_nat_n_exists_l250_250186


namespace fill_6x6_with_tetris_pieces_l250_250102

-- Definitions for Tetris pieces (represented as internal notation or lists)
-- Definition of each Tetris piece (can be expanded if necessary for proofs)
inductive TetrisPiece
| O | I | T | S | Z | L | J
-- If we assume these shapes can be represented appropriately in Lean.

-- The 6x6 grid using Fin 6 for constraint bounds
def grid : Matrix (Fin 6) (Fin 6) (Option TetrisPiece) := sorry

-- Proof statement that grid can be filled with the given pieces
theorem fill_6x6_with_tetris_pieces : 
  ∃ (g : Matrix (Fin 6) (Fin 6) (Option TetrisPiece)),
    (∀ i j, g i j ≠ none) ∧ -- Every cell is filled
    (∀ p : TetrisPiece, ∃ i j, g i j = some p) -- Each piece is used at least once
:= 
begin
  sorry
end

end fill_6x6_with_tetris_pieces_l250_250102


namespace number_of_integers_satisfying_ineq_l250_250702

theorem number_of_integers_satisfying_ineq : 
  (finset.card {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset) = 7 := 
sorry

end number_of_integers_satisfying_ineq_l250_250702


namespace geometric_series_first_term_l250_250153

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by {
  sorry
}

end geometric_series_first_term_l250_250153


namespace circle_equation_center_on_y_axis_radius_one_semantics_l250_250542

theorem circle_equation_center_on_y_axis_radius_one_semantics
  (b : ℝ)
  (h1 : (0, b) ∈ set_of (λ p : ℝ × ℝ, p.1 = 0))
  (h2 : (1, 2) ∈ set_of (λ p : ℝ × ℝ, (p.1-0)^2 + (p.2-b)^2 = 1)) :
  (∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1-0)^2 + (p.2-b)^2 = 1) ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1^2 + (p.2-2)^2 = 1)) :=
sorry

end circle_equation_center_on_y_axis_radius_one_semantics_l250_250542


namespace joe_money_left_l250_250383

theorem joe_money_left (initial_money : ℕ) (notebook_count : ℕ) (book_count : ℕ)
    (notebook_price : ℕ) (book_price : ℕ) (total_spent : ℕ) : 
    initial_money = 56 → notebook_count = 7 → book_count = 2 → notebook_price = 4 → book_price = 7 →
    total_spent = notebook_count * notebook_price + book_count * book_price →
    (initial_money - total_spent) = 14 := 
by 
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end joe_money_left_l250_250383


namespace find_angles_of_triangle_l250_250854

theorem find_angles_of_triangle
  {α β γ : ℝ} 
  (C H L M P : Type*) 
  [Triangle ABC]
  (CH : H = C) 
  (CL : L = C)
  (circumcircle : Circle P)
  (height_extension : Line CH)
  (bisector : Line CL)
  (CH_extension_point: P ∈ circumcircle)
  (CP_length: CP = 2 * CH)
  (CM_length: CM = (9 / 4) * CL)
  (angles_conditions : α + β + γ = π)
  (right_angle_condition: γ = π/2) :
  α = 0.5 * Real.arcsin(4 / 5) ∧ β = π/2 - α ∧ γ = π/2 :=
by
  sorry

end find_angles_of_triangle_l250_250854


namespace smallest_positive_period_f_intervals_monotonic_decrease_range_of_g_l250_250688

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * cos (π / 2 - 2 * x) + 2 * cos x ^ 2 - 1
noncomputable def g (x : ℝ) : ℝ := f (x + π / 12) + 1

theorem smallest_positive_period_f : period f = π := sorry

theorem intervals_monotonic_decrease :
  ∀ k : ℤ, ∃ a b : ℝ, (a = π / 6 + k * π ∧ b = 2 * π / 3 + k * π) ∧
    ∀ x : ℝ, a ≤ x ∧ x ≤ b ∨ b < a ∨ a > b → f' x ≤ 0 := sorry

theorem range_of_g (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) :
  - sqrt 3 + 1 ≤ g x ∧ g x ≤ 3 := sorry

end smallest_positive_period_f_intervals_monotonic_decrease_range_of_g_l250_250688


namespace emma_additional_miles_l250_250984

theorem emma_additional_miles :
  ∀ (initial_distance : ℝ) (initial_speed : ℝ) (additional_speed : ℝ) (desired_avg_speed : ℝ) (total_distance : ℝ) (additional_distance : ℝ),
    initial_distance = 20 →
    initial_speed = 40 →
    additional_speed = 70 →
    desired_avg_speed = 60 →
    total_distance = initial_distance + additional_distance →
    (total_distance / ((initial_distance / initial_speed) + (additional_distance / additional_speed))) = desired_avg_speed →
    additional_distance = 70 :=
by
  intros initial_distance initial_speed additional_speed desired_avg_speed total_distance additional_distance
  intros h1 h2 h3 h4 h5 h6
  sorry

end emma_additional_miles_l250_250984


namespace sum_of_first_60_terms_l250_250477

theorem sum_of_first_60_terms :
  (∑ n in finset.range 60, (rec { a : ℕ // True } 0 (λ (n : ℕ) (t : a ≠ (\sum a)))),
  sorry :=

end sum_of_first_60_terms_l250_250477


namespace inscribed_circle_radius_in_sector_l250_250839

theorem inscribed_circle_radius_in_sector
  (radius : ℝ)
  (sector_fraction : ℝ)
  (r : ℝ) :
  radius = 4 →
  sector_fraction = 1/3 →
  r = 2 * Real.sqrt 3 - 2 →
  true := by
sorry

end inscribed_circle_radius_in_sector_l250_250839


namespace part1_part2_l250_250218

-- Define the sequences and conditions
def S_n (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  1/4 * (a n)^2 + 1/2 * (a n) + 1/4

def geom_seq (a b : ℕ → ℕ) (q : ℚ) : Prop :=
  ∀ n, n > 0 → (a n + b n = 1 * q^(n-1))

-- Problem Statements
theorem part1 (a : ℕ → ℕ) :
  (∀ n, n > 0 → S_n a n = 1/4 * (a n)^2 + 1/2 * (a n) + 1/4) →
  (∀ n, n > 1 → a n = 2*n - 1) :=
sorry

theorem part2 (a b : ℕ → ℕ) (q : ℚ) :
  (∀ n, n > 0 → S_n a n = 1/4 * (a n)^2 + 1/2 * (a n) + 1/4) →
  (∀ n, n > 1 → a n = 2*n - 1) →
  geom_seq a b q →
  (∀ n, q = 1 → ∑ i in finset.range n, b i = -n^2 + n) ∧
  (∀ n, q ≠ 1 → ∑ i in finset.range n, b i = -n^2 + (1 - q^n) / (1 - q)) :=
sorry

end part1_part2_l250_250218


namespace polynomial_roots_product_l250_250444

theorem polynomial_roots_product :
  let p := - (\sin (3 * π / 7) + \sin (5 * π / 7) + \sin (π / 7)),
      q := \sin (3 * π / 7) * \sin (5 * π / 7) + \sin (5 * π / 7) * \sin (π / 7) + \sin (π / 7) * \sin (3 * π / 7),
      r := - \sin (3 * π / 7) * \sin (5 * π / 7) * \sin (π / 7)
  in p * q * r = 0.725 :=
sorry

end polynomial_roots_product_l250_250444


namespace number_of_tables_in_lunchroom_l250_250106

def num_tables (students_per_table total_students : ℕ) : ℕ :=
  total_students / students_per_table

theorem number_of_tables_in_lunchroom (h1 : 6) (h2 : 204) :
  num_tables h1 h2 = 34 :=
by
  sorry

end number_of_tables_in_lunchroom_l250_250106


namespace number_of_integers_satisfying_ineq_l250_250703

theorem number_of_integers_satisfying_ineq : 
  (finset.card {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset) = 7 := 
sorry

end number_of_integers_satisfying_ineq_l250_250703


namespace smallest_positive_integer_n_l250_250967

noncomputable def satisfies_condition (n : ℕ) : Prop :=
  ∑ k in Finset.range (n + 1), Real.log (1 + 1 / 2 ^ (2 ^ k)) / Real.log 2 ≥
    1 + Real.log 1553 / 1554 / Real.log 2

theorem smallest_positive_integer_n : ∃ n : ℕ, satisfies_condition n ∧ ∀ m : ℕ, satisfies_condition m → n ≤ m :=
  sorry

end smallest_positive_integer_n_l250_250967


namespace count_integers_satisfying_inequality_l250_250720

theorem count_integers_satisfying_inequality : 
  ∃ (s : Finset ℤ), (∀ n ∈ s, (n - 3) * (n + 5) < 0) ∧ s.card = 7 :=
begin
  sorry
end

end count_integers_satisfying_inequality_l250_250720


namespace probability_median_five_l250_250949

theorem probability_median_five {S : Finset ℕ} (hS : S = {1, 2, 3, 4, 5, 6, 7, 8}) :
  let n := 8
  let k := 5
  let total_ways := Nat.choose n k
  let ways_median_5 := Nat.choose 4 2 * Nat.choose 3 2
  (ways_median_5 : ℚ) / (total_ways : ℚ) = (9 : ℚ) / (28 : ℚ) :=
by
  sorry

end probability_median_five_l250_250949


namespace nested_radical_simplification_l250_250027

theorem nested_radical_simplification : 
  (√ (∛ (∜ (1 / 4096)))) = 1 / (√ 2) := 
sorry

end nested_radical_simplification_l250_250027


namespace value_of_y_l250_250335

theorem value_of_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3 * y) = 8) : y = 1 :=
by
  sorry

end value_of_y_l250_250335


namespace exists_disjoint_nonempty_subsets_with_equal_sum_l250_250498

theorem exists_disjoint_nonempty_subsets_with_equal_sum :
  ∀ (A : Finset ℕ), (A.card = 11) → (∀ a ∈ A, 1 ≤ a ∧ a ≤ 100) →
  ∃ (B C : Finset ℕ), B ≠ ∅ ∧ C ≠ ∅ ∧ B ∩ C = ∅ ∧ (B ∪ C ⊆ A) ∧ (B.sum id = C.sum id) :=
by
  sorry

end exists_disjoint_nonempty_subsets_with_equal_sum_l250_250498


namespace max_min_f_l250_250460

def f (x : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 2 then 2 * x + 6
  else if -1 ≤ x ∧ x ≤ 1 then x + 7
  else 0

theorem max_min_f :
  (∀ x, x ∈ set.Ioc 1 2 → f x ≤ 10) ∧ 
  (∃ x, x ∈ set.Ioc 1 2 ∧ f x = 10) ∧ 
  (∀ x, x ∈ set.Ioc 1 2 → f x ≥ 8) ∧
  (∃ x, x ∈ set.Ioc 1 2 ∧ f x = 8) ∧ 
  (∀ x, x ∈ set.Icc (-1) 1 → f x ≤ 8) ∧ 
  (∃ x, x ∈ set.Icc (-1) 1 ∧ f x = 8) ∧ 
  (∀ x, x ∈ set.Icc (-1) 1 → f x ≥ 6) ∧
  (∃ x, x ∈ set.Icc (-1) 1 ∧ f x = 6) :=
by
  sorry

end max_min_f_l250_250460


namespace solve_equations_l250_250844

theorem solve_equations :
  (∀ x : ℝ, x^2 - 2 * x - 15 = 0 ↔ x = 5 ∨ x = -3) ∧
  (∀ x : ℝ, 2 * x^2 + 3 * x - 1 = 0 ↔ x = (-3 + Real.sqrt 17) / 4 ∨ x = (-3 - Real.sqrt 17) / 4) :=
by
  sorry

end solve_equations_l250_250844


namespace dot_product_eq_neg20_l250_250225

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (-5, 5)

def dot_product (x y : ℝ × ℝ) : ℝ :=
x.1 * y.1 + x.2 * y.2

theorem dot_product_eq_neg20 :
  dot_product a b = -20 :=
by
  sorry

end dot_product_eq_neg20_l250_250225


namespace connie_marble_problem_l250_250174

theorem connie_marble_problem :
  ∀ (initial_marbles remaining_marbles marbles_given : ℕ),
  initial_marbles = 776 →
  remaining_marbles = 593 →
  marbles_given = initial_marbles - remaining_marbles →
  marbles_given = 183 :=
by
  intros initial_marbles remaining_marbles marbles_given h_initial h_remaining h_calc
  rw [h_initial, h_remaining] at h_calc
  exact h_calc.sorry

end connie_marble_problem_l250_250174


namespace multiples_count_18_l250_250285

theorem multiples_count_18 : 
  (finset.filter (λ x, x % 15 = 0 ∧ x % 8 ≠ 0) (finset.range 301)).card = 18 :=
by
  sorry

end multiples_count_18_l250_250285


namespace exists_x_in_interval_l250_250241

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b + 9 / x

theorem exists_x_in_interval (a b : ℝ) : ∃ x ∈ set.Icc 1 9, |f a b x| ≥ 2 :=
begin
  sorry
end

end exists_x_in_interval_l250_250241


namespace count_integers_satisfying_inequality_l250_250725

theorem count_integers_satisfying_inequality : 
  ∃ (s : Finset ℤ), (∀ n ∈ s, (n - 3) * (n + 5) < 0) ∧ s.card = 7 :=
begin
  sorry
end

end count_integers_satisfying_inequality_l250_250725


namespace number_of_paths_3x3_l250_250616

-- Definition of the problem conditions
def grid_moves (n m : ℕ) : ℕ := Nat.choose (n + m) n

-- Lean statement for the proof problem
theorem number_of_paths_3x3 : grid_moves 3 3 = 20 := by
  sorry

end number_of_paths_3x3_l250_250616


namespace cell_growth_at_end_of_9_days_l250_250154

theorem cell_growth_at_end_of_9_days 
    (initial_cells : ℕ) 
    (triples_every_three_days : ℕ) 
    (total_days : ℕ) 
    (cycles : ℕ) 
    (h_initial_cells : initial_cells = 5) 
    (h_triples_every_three_days : triples_every_three_days = 3) 
    (h_total_days : total_days = 9) 
    (h_cycles : cycles = total_days / 3):
  initial_cells * triples_every_three_days^cycles = 135 := 
by 
  rw [h_initial_cells, h_triples_every_three_days, h_total_days, h_cycles];
  norm_num;
  sorry

end cell_growth_at_end_of_9_days_l250_250154


namespace max_diff_sum_seq_is_10_l250_250273

def seq (n : ℕ) : ℤ := -(n : ℤ)^2 + 12 * (n : ℤ) - 32

def sum_seq (n : ℕ) : ℤ := (Fin.partialSum seq n).val

theorem max_diff_sum_seq_is_10 :
  ∃ (n m : ℕ), n > m ∧ (sum_seq n - sum_seq m) = 10 :=
by
  sorry

end max_diff_sum_seq_is_10_l250_250273


namespace probability_product_divisible_by_four_l250_250079

theorem probability_product_divisible_by_four :
  (∀ (d : fin 8 → fin 6), let product := ∏ (i : fin 8), (d i).val + 1 in (product % 4 = 0) → 63 / 64 = 1) :=
by sorry

end probability_product_divisible_by_four_l250_250079


namespace estimated_excellent_students_is_152_l250_250942

noncomputable def estimate_excellent_students
  (total_students : ℕ)
  (sampled_students : ℕ)
  (sampled_excellent : ℕ) : ℕ :=
let sample_rate := (sampled_excellent : ℚ) / sampled_students in
let estimate := total_students * sample_rate in
estimate.natAbs

theorem estimated_excellent_students_is_152 :
  estimate_excellent_students 380 50 20 = 152 :=
by
  unfold estimate_excellent_students
  -- Insert detailed steps if needed, but for now, we use sorry to indicate the proof is not provided
  sorry

end estimated_excellent_students_is_152_l250_250942


namespace count_scalene_triangles_natural_sides_l250_250579

theorem count_scalene_triangles_natural_sides :
  let scalene_triangles := { S : ℕ × ℕ × ℕ | 
    let a := S.1, 
        b := S.2.1, 
        c := S.2.2 in 
    a < b ∧ b < c ∧ 
    a + c = 2 * b ∧ 
    a + b + c ≤ 30 ∧ 
    a > 0 ∧ b > 0 ∧ c > 0 } in
  scalene_triangles.finite.count = 20 :=
by 
  sorry

end count_scalene_triangles_natural_sides_l250_250579


namespace probability_of_six_each_color_in_urn_after_five_iterations_l250_250951

-- Define an event that represents drawing a ball from the urn
inductive Ball : Type
| red : Ball
| blue : Ball

open Ball

def initial_urn : List Ball := [red, red, blue]

def nth_urn (n : ℕ) : List Ball
| 0 => initial_urn
| n + 1 => 
  let urn := nth_urn n
  let red_count := urn.count red
  let blue_count := urn.count blue
  let new_ball := classical.some (nat.find (λ x, urn.count x > 0))
  if new_ball = red then urn ++ [red, red] else urn ++ [blue, blue]

noncomputable def probability_six_each_color : ℚ := 
  let total_ways := 10 -- Combinations of selecting 3 red and 2 blue balls from 5 draws
                      -- binomial coefficient calculated as (5 C 3)
  let probability_each_sequence := (2 / 3) * (3 / 4) * (4 / 5) * (1 / 6) * (2 / 7) -- Example for sequence RRRBB
  total_ways * probability_each_sequence

theorem probability_of_six_each_color_in_urn_after_five_iterations : 
  probability_six_each_color = 16 / 63 :=
by 
  sorry

end probability_of_six_each_color_in_urn_after_five_iterations_l250_250951


namespace perp_BR_CR_l250_250369

theorem perp_BR_CR
  (A B C I R P Q : Type)
  [euclidean_geometry_type A B C I]
  (h1 : center_incircle I (triangle A B C))
  (h2 : dist A B = dist A C)
  (h3 : circle_center_radius (circle A) A (dist A B))
  (h4 : circle_center_radius (circle I) I (dist I B))
  (h5 : ∃ Γ : Type, circle (Γ) ∧ passes_through B I (Γ) 
                    ∧ intersects Γ (circle A) P ∧ intersects Γ (circle I) Q 
                  ∧ P ≠ B ∧ Q ≠ B)
  (h6 : ∃ R : Type, intersects_at (line IP) (line BQ) R):
  ∠B R C = 90 :=
by
  sorry

end perp_BR_CR_l250_250369


namespace y_coord_of_B_l250_250833

theorem y_coord_of_B (A B C D : (ℝ × ℝ)) (y : ℝ) 
  (hA : A = (0, 0)) (hB : B = (8, y)) (hC : C = (8, 16)) (hD : D = (0, 16)) 
  (symmetry : ∀ (p : ℝ × ℝ), p = (p.1, 16 - p.2) → (p ∈ {A, B, C, D} ∨ p ∈ {A, B, C, D})) 
  (area_eq : 8 * y = 72) : 
  y = 9 :=
by
  sorry

end y_coord_of_B_l250_250833


namespace part1_max_min_part2_monotonic_increasing_intervals_l250_250813

/-- Definition of the function f(x) -/
def f (x : ℝ) (m : ℝ) : ℝ := - (1 / 3) * x ^ 3 + x ^ 2 + (m ^ 2 - 1) * x

/-- Part 1: Prove maximum and minimum values when m = 1 on the interval [-3, 2] -/
theorem part1_max_min (x : ℝ) : x ∈ Icc (-3 : ℝ) 2 → 
  (f x 1 ≤ 18 ∧ (∀ y, y ∈ Icc (-3 : ℝ) 2 → f y 1 ≤ f x 1) → (18 ≤ f x 1)) ∧
  ((0:ℝ) ≤ f x 1 ∧ (∀ y, y ∈ Icc (-3 : ℝ) 2 → f x 1 ≤ f y 1) → (f x 1 ≤ 0)) := 
sorry

/-- Part 2: Prove the function f(x) is monotonically increasing on the interval (1 - m, m + 1) -/
theorem part2_monotonic_increasing_intervals (x m : ℝ) : m > 0 → 
  x ∈ Ioo (1 - m) (m + 1) → 
  (∀ x1 x2, x1 < x2 → f x1 m ≤ f x2 m) := 
sorry 

end part1_max_min_part2_monotonic_increasing_intervals_l250_250813


namespace M_gt_N_l250_250798

open Real

noncomputable def M (a : ℝ) := a
def N (x : ℝ) := log (x^2 + x) / log 0.5

theorem M_gt_N (a x : ℝ) (ha : 2 < a ∧ a < 3) (hx1 : -1 < x) (hx2 : x ≠ 0) (hx3 : x^2 + x > 0) : M a > N x := 
by {
  rw [M, N],
  sorry
}

end M_gt_N_l250_250798


namespace cube_painting_distinct_ways_l250_250157

theorem cube_painting_distinct_ways :
  ∃ (n : ℕ), n = 6 ∧ 
  ∀ (cubes : ℕ → list (list color)), 
    (∀ (c : ℕ), let faces := cubes c in 
      faces.length = 6 ∧ 
      (∀ (f1 f2 : ℕ), 
        (f1 ≠ f2 → 
        (faces.nth_le f1 (by sorry) == faces.nth_le f2 (by sorry) ↔
         opposite_faces f1 f2))) ∧ -- each pair of opposite faces has the same color
      (rotational_symmetry_preserved cubes)) :=
sorry

end cube_painting_distinct_ways_l250_250157


namespace find_teddy_dogs_l250_250447

-- Definitions from the conditions
def teddy_cats := 8
def ben_dogs (teddy_dogs : ℕ) := teddy_dogs + 9
def dave_cats (teddy_cats : ℕ) := teddy_cats + 13
def dave_dogs (teddy_dogs : ℕ) := teddy_dogs - 5
def total_pets (teddy_dogs teddy_cats : ℕ) := teddy_dogs + teddy_cats + (ben_dogs teddy_dogs) + (dave_dogs teddy_dogs) + (dave_cats teddy_cats)

-- Theorem statement
theorem find_teddy_dogs (teddy_dogs : ℕ) (teddy_cats : ℕ) (hd : total_pets teddy_dogs teddy_cats = 54) :
  teddy_dogs = 7 := sorry

end find_teddy_dogs_l250_250447


namespace solve_rational_equation_l250_250440

theorem solve_rational_equation (x : ℝ) : 
  (7 * x + 3) / (3 * x^2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔ 
  (x = (-1 + real.sqrt 10) / 3 ∨ x = (-1 - real.sqrt 10) / 3) :=
by
  sorry

end solve_rational_equation_l250_250440


namespace factorize_cubic_l250_250999

theorem factorize_cubic (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end factorize_cubic_l250_250999


namespace max_erasers_l250_250142

theorem max_erasers (p n e : ℕ) (h₁ : p ≥ 1) (h₂ : n ≥ 1) (h₃ : e ≥ 1) (h₄ : 3 * p + 4 * n + 8 * e = 60) :
  e ≤ 5 :=
sorry

end max_erasers_l250_250142


namespace ratio_volume_John_to_Sarah_l250_250386

-- Define the dimensions of the cylindrical containers.
def diameter_John := 10
def height_John := 15
def diameter_Sarah := 15
def height_Sarah := 10

-- Calculate the radius from the diameter.
def radius_John := diameter_John / 2
def radius_Sarah := diameter_Sarah / 2

-- Define the volume calculations for John's and Sarah's containers.
def volume_John := Real.pi * (radius_John ^ 2) * height_John
def volume_Sarah := Real.pi * (radius_Sarah ^ 2) * height_Sarah

-- Prove the ratio of the volumes is 2:3.
theorem ratio_volume_John_to_Sarah : volume_John / volume_Sarah = 2 / 3 := by
  sorry

end ratio_volume_John_to_Sarah_l250_250386


namespace problem1_problem2_l250_250267

noncomputable def f (a x : ℝ) := a * x - (Real.log x) / x - a

theorem problem1 (a : ℝ) (h : deriv (f a) 1 = 0) : a = 1 := by
  sorry

theorem problem2 (a : ℝ) (h : ∀ x, 1 < x ∧ x < Real.exp 1 → f a x ≤ 0) : a ≤ 1 / (Real.exp 1 * (Real.exp 1 - 1)) := by
  sorry

end problem1_problem2_l250_250267


namespace systematic_sampling_fourth_group_l250_250544

theorem systematic_sampling_fourth_group 
  (n k : ℕ) (group_size : ℕ) 
  (groups : ℕ → set ℕ) 
  (sampled_num_group2 : ℕ) 
  (h1 : n = 100)
  (h2 : k = 5)
  (h3 : group_size = n / k)
  (h4 : groups 1 = {i | 1 ≤ i ∧ i ≤ 20})
  (h5 : groups 2 = {i | 21 ≤ i ∧ i ≤ 40})
  (h6 : groups 3 = {i | 41 ≤ i ∧ i ≤ 60})
  (h7 : groups 4 = {i | 61 ≤ i ∧ i ≤ 80})
  (h8 : groups 5 = {i | 81 ≤ i ∧ i ≤ 100})
  (h9 : sampled_num_group2 = 24) :
  {i | 64 ∈ groups 4} :=
by
  sorry

end systematic_sampling_fourth_group_l250_250544


namespace two_digit_numbers_three_digit_numbers_infinite_sequence_exists_l250_250103

-- Part a
theorem two_digit_numbers (A : ℕ) : (A^2 % 100 = A % 100) ↔ (A = 25 ∨ A = 76) :=
sorry

-- Part b
theorem three_digit_numbers (A : ℕ) : (A^2 % 1000 = A % 1000) ↔ (A = 625 ∨ A = 376) :=
sorry

-- Part c
theorem infinite_sequence_exists : ∃ (a : ℕ → ℕ), (∀ n : ℕ, a 1 = 6 ∧ (nat.pow 10 n) + a n (nat.pow 10 n) ≠ 1) :=
sorry

end two_digit_numbers_three_digit_numbers_infinite_sequence_exists_l250_250103


namespace how_many_integers_satisfy_l250_250712

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end how_many_integers_satisfy_l250_250712


namespace determine_a_l250_250339

-- Define the function f(x) = a^x
def f (a : ℝ) (x : ℝ) : ℝ := a ^ x

-- Define the condition a > 0 and a ≠ 1
def valid_a (a : ℝ) : Prop := a > 0 ∧ a ≠ 1

-- The function's inverse passes through (3, -1)
def inverse_passes_through (a : ℝ) : Prop := ∃ f_inv : ℝ → ℝ, (∀ y : ℝ, f_inv (f a y) = y) ∧ f_inv 3 = -1

-- The proof statement
theorem determine_a (a : ℝ) (h1 : valid_a a) (h2 : inverse_passes_through a) : a = 1 / 3 :=
sorry

end determine_a_l250_250339


namespace number_of_valid_pairs_l250_250252

theorem number_of_valid_pairs : 
  ∃ (n : ℕ), n = 1995003 ∧ (∃ b c : ℤ, c < 2000 ∧ b > 2 ∧ (∀ x : ℂ, x^2 - (b:ℝ) * x + (c:ℝ) = 0 → x.re > 1)) := 
sorry

end number_of_valid_pairs_l250_250252


namespace end_with_one_stone_l250_250205

def edge_cells (i j : fin 4) : Prop :=
  (i = 0 ∨ i = 3 ∨ j = 0 ∨ j = 3)

-- Initialize the board with 15 stones and 1 empty cell.
def initial_board (empty_cell : fin 4 × fin 4) : matrix (fin 4) (fin 4) bool :=
  λ x y => if (x, y) = empty_cell then false else true

-- Verify the stone jumping rules and the board's configuration.
def valid_move (board : matrix (fin 4) (fin 4) bool) (start dest : fin 4 × fin 4) : Prop :=
  true -- Placeholder for actual move validation rules

-- Define the board transformation rules according to valid moves.
def board_transform (board : matrix (fin 4) (fin 4) bool) (move : fin 4 × fin 4 × fin 4 × fin 4) : matrix (fin 4) (fin 4) bool :=
  board -- Placeholder for actual transformation logic

-- Define a function to count the remaining stones on the board.
def count_stones (board : matrix (fin 4) (fin 4) bool) : nat :=
  finset.card (finset.filter id (finset.univ.image (λ ij : fin 4 × fin 4 => board ij.1 ij.2)))

-- The main theorem to prove: starting with an empty edge cell, it is possible to end with exactly one stone.
theorem end_with_one_stone (empty_cell : fin 4 × fin 4) (h_edge : edge_cells empty_cell.1 empty_cell.2) :
  ∃ moves : list (fin 4 × fin 4 × fin 4 × fin 4), 
    let board := list.foldl board_transform (initial_board empty_cell) moves 
    in count_stones board = 1 :=
sorry

end end_with_one_stone_l250_250205


namespace curve_eq_circle_l250_250213

theorem curve_eq_circle (r θ : ℝ) : (∀ θ : ℝ, r = 3) ↔ ∃ c : ℝ, c = 0 ∧ ∀ z : ℝ, (r = real.sqrt ((3 - c)^2 + θ^2)) := sorry

end curve_eq_circle_l250_250213


namespace allan_balloons_l250_250574

def initial_balloons : ℕ := 5
def additional_balloons : ℕ := 3
def total_balloons : ℕ := initial_balloons + additional_balloons

theorem allan_balloons :
  total_balloons = 8 :=
sorry

end allan_balloons_l250_250574


namespace number_of_integers_satisfying_ineq_l250_250706

theorem number_of_integers_satisfying_ineq : 
  (finset.card {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset) = 7 := 
sorry

end number_of_integers_satisfying_ineq_l250_250706


namespace arithmetic_geom_sequence_a2_l250_250664

theorem arithmetic_geom_sequence_a2 :
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n+1) = a n + 2) →  -- Arithmetic sequence with common difference of 2
    a 1 * a 4 = a 3 ^ 2 →  -- Geometric sequence property for a_1, a_3, a_4
    a 2 = -6 :=             -- The value of a_2
by
  intros a h_arith h_geom
  sorry

end arithmetic_geom_sequence_a2_l250_250664


namespace recoloring_process_terminates_l250_250955

-- Define the initial setup and recolor function
def set_of_distances (color : ℝ → Prop) : set ℝ :=
  {d | ∃ x y : ℝ, color x ∧ color y ∧ d = |x - y|}

def recolor (R : set ℝ) (B : set ℝ) : set ℝ × set ℝ :=
  let D := set_of_distances (λ x, x ∈ R) in
  (D, {x | x ∉ D})

-- Define the iterative process
def iterate_recolor (R : set ℝ) (B : set ℝ) (n : ℕ) : set ℝ × set ℝ :=
  nat.rec (R, B) (λ _ prev, recolor prev.1 prev.2) n

-- The proof goal: After a finite number of steps, all numbers will be red
theorem recoloring_process_terminates : 
  ∀ R B : set ℝ, ∃ n : ℕ, ∀ x : ℝ, x ∈ (iterate_recolor R B n).1 :=
sorry

end recoloring_process_terminates_l250_250955


namespace percentage_of_palindromes_with_seven_is_10_percent_l250_250126

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def has_seven (n : ℕ) : Prop :=
  ('7' ∈ n.to_string)

def count_eligible_palindromes : ℕ :=
  let candidates := finset.filter (λ n, 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n) (finset.range 2000)
  finset.card candidates

def count_palindromes_with_seven : ℕ :=
  let qualified := finset.filter (λ n, 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n ∧ has_seven n) (finset.range 2000)
  finset.card qualified

theorem percentage_of_palindromes_with_seven_is_10_percent :
  100 * count_palindromes_with_seven / count_eligible_palindromes = 10 :=
by sorry

end percentage_of_palindromes_with_seven_is_10_percent_l250_250126


namespace popsicle_melting_ratio_l250_250108

theorem popsicle_melting_ratio (S : ℝ) (r : ℝ) (h : r^5 = 32) : r = 2 :=
by
  sorry

end popsicle_melting_ratio_l250_250108


namespace minimum_distance_l250_250752

theorem minimum_distance (m n : ℕ) (h : m * n - m - n = 3) : 
  ∃ d, d = abs (m + n) / real.sqrt 2 ∧ d = 3 * real.sqrt 2 :=
by {
  sorry
}

end minimum_distance_l250_250752


namespace circumscribed_circle_eq_l250_250695

theorem circumscribed_circle_eq (A B C : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (4, 2)) (hC : C = (2, -2)) :
  ∃ D E F : ℝ, (D = -6 ∧ E = 0 ∧ F = 4) ∧ 
  (∀ P ∈ {A, B, C}, P.1^2 + P.2^2 + D * P.1 + E * P.2 + F = 0) :=
by {
  use [-6, 0, 4],
  split,
  { exact ⟨rfl, rfl, rfl⟩ },
  { intros P hP,
    cases hP,
    { rw hA at hP,
      simp [hP] },
    cases hP,
    { rw hB at hP,
      simp [hP] },
    { rw hC at hP,
      simp [hP] }
  }
}

end circumscribed_circle_eq_l250_250695


namespace marbles_in_jar_is_144_l250_250113

noncomputable def marbleCount (M : ℕ) : Prop :=
  M / 16 - M / 18 = 1

theorem marbles_in_jar_is_144 : ∃ M : ℕ, marbleCount M ∧ M = 144 :=
by
  use 144
  unfold marbleCount
  sorry

end marbles_in_jar_is_144_l250_250113


namespace number_of_drawn_cards_l250_250224

def total_cards : ℕ := 52
def face_cards : ℕ := 12
def probability : ℚ := 12 / 52

theorem number_of_drawn_cards (n : ℕ) :
  52.cards_in_pack ∧ 0.23076923076923078.probability_of_faceCard ∧ 12.face_cards → n = 52 :=
by
  sorry

end number_of_drawn_cards_l250_250224


namespace exist_pair_lcm_gcd_l250_250462

theorem exist_pair_lcm_gcd (a b: ℤ) : 
  ∃ a b : ℤ, Int.lcm a b - Int.gcd a b = 19 := 
sorry

end exist_pair_lcm_gcd_l250_250462


namespace cyclic_trapezoid_area_l250_250924

theorem cyclic_trapezoid_area (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha_lt_hb : a < b) :
  (a * 2 * ⟨Real.sin (45 / 180 * Real.pi)⟩ + b) / 2 = sqrt 2 * a / 2 → 
  let t₁ := (a + b) * (a - sqrt (2 * a^2 - b^2)) / 4 in
  let t₂ := (a + b) * (a + sqrt (2 * a^2 - b^2)) / 4 in
  ∃ t, t = t₁ ∨ t = t₂ := sorry

end cyclic_trapezoid_area_l250_250924


namespace lassis_from_mangoes_l250_250604

def ratio (lassis mangoes : ℕ) : Prop := lassis = 11 * mangoes / 2

theorem lassis_from_mangoes (mangoes : ℕ) (h : mangoes = 10) : ratio 55 mangoes :=
by
  rw [h]
  unfold ratio
  sorry

end lassis_from_mangoes_l250_250604


namespace count_integers_satisfying_inequality_l250_250721

theorem count_integers_satisfying_inequality : 
  ∃ (s : Finset ℤ), (∀ n ∈ s, (n - 3) * (n + 5) < 0) ∧ s.card = 7 :=
begin
  sorry
end

end count_integers_satisfying_inequality_l250_250721


namespace who_has_winning_strategy_l250_250952
open Nat

/-- The definition of a winning strategy for Andrew in the pebble game. -/
def Andrew_winning_strategy (a b : ℕ) : Prop :=
  a = 1 ∨ b = 1 ∨ ∃ x, a + b = 2^x + 1 ∨ (a = 2^x + 1 ∧ b < a) ∨ (b = 2^x + 1 ∧ a < b)

/-- Theorem stating whether Andrew or Barry has a winning strategy.
Andrew has a winning strategy if:
    1. a = 1, or
    2. b = 1, or
    3. ∃ x, a + b = 2^x + 1, or
    4. a = 2^x + 1 and b < a, or
    5. b = 2^x + 1 and a < b.
Otherwise, Barry has a winning strategy. -/
theorem who_has_winning_strategy (a b : ℕ) : 
  (Andrew_winning_strategy a b → (¬ (Barry_has_winning_strategy a b))) ∧ 
  ((¬ (Andrew_winning_strategy a b)) → Barry_has_winning_strategy a b) :=
sorry

end who_has_winning_strategy_l250_250952


namespace sin_cos_quad_sum_l250_250320

theorem sin_cos_quad_sum (θ : ℝ) (h : cos (2 * θ) = 1 / 3) : sin θ ^ 4 + cos θ ^ 4 = 5 / 9 :=
by
  -- The proof here would use trigonometric identities and algebraic manipulation.
  sorry

end sin_cos_quad_sum_l250_250320


namespace joe_money_left_l250_250385

theorem joe_money_left (initial_money : ℕ) (notebook_count : ℕ) (book_count : ℕ)
    (notebook_price : ℕ) (book_price : ℕ) (total_spent : ℕ) : 
    initial_money = 56 → notebook_count = 7 → book_count = 2 → notebook_price = 4 → book_price = 7 →
    total_spent = notebook_count * notebook_price + book_count * book_price →
    (initial_money - total_spent) = 14 := 
by 
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end joe_money_left_l250_250385


namespace gcd_gx_x_l250_250672

theorem gcd_gx_x (x : ℕ) (h : 2520 ∣ x) : 
  Nat.gcd ((4*x + 5) * (5*x + 2) * (11*x + 8) * (3*x + 7)) x = 280 := 
sorry

end gcd_gx_x_l250_250672


namespace max_additional_birds_l250_250626

def bird1_weight := 2.5
def bird2_weight := 3.5
def new_bird_weight := 2.8
def new_bird_count := 4
def fence_capacity := 20
def additional_bird_weight := 3

theorem max_additional_birds : 
  let current_weight := bird1_weight + bird2_weight + new_bird_weight * new_bird_count in
  let remaining_capacity := fence_capacity - current_weight in
  remaining_capacity < additional_bird_weight → 0 = 0 :=
by
  sorry

end max_additional_birds_l250_250626


namespace func_eq_x_or_neg_x_l250_250206

theorem func_eq_x_or_neg_x (f : ℚ → ℚ) 
    (h : ∀ x y : ℚ, f (f x + f y) = f (f x) + y) :
  f = (λ x, x) ∨ f = (λ x, -x) := 
by 
  sorry

end func_eq_x_or_neg_x_l250_250206


namespace number_of_increasing_6_digit_numbers_l250_250796

open Function Nat

theorem number_of_increasing_6_digit_numbers : 
  let M := 5004
  M = card { digits : Vector ℕ 6 | (∀ i, digits.get i ≤ digits.get (i+1)) ∧ (∀ i, digits.get i < 10) } := by sorry

end number_of_increasing_6_digit_numbers_l250_250796


namespace largest_angle_of_triangle_l250_250781

noncomputable def largest_internal_angle (a b c : ℝ) (h1 : a = 5) (h2 : b = 3) (h3 : c = 7) : ℝ :=
  let cosC : ℝ := (a^2 + b^2 - c^2) / (2 * a * b)
  if a < b then
    let temp := a
    a := b
    b := temp
  if a < c then
    let temp := a
    a := c
    c := temp
  if b < c then
    let temp := b
    b := c
    c := temp
  if h : cosC = -1 / 2 then 
    acos (-1 / 2)
  else
    0 -- return 0 for any other case to make it compilable

theorem largest_angle_of_triangle : largest_internal_angle 5 3 7 5 3 7 = 2 * pi / 3 :=
  sorry

end largest_angle_of_triangle_l250_250781


namespace problem_one_problem_two_l250_250687

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (a + Real.log x) / x

theorem problem_one (a : ℝ) 
  (h1 : ∀ e : ℝ, e > 0 → Deriv f e = -1/e^2)
  (h2 : (e:ℝ) > 0 → ∃ x : ℝ, f(x) is_extremum_in (0, 1)) : 0 < a ∧ a < 1 := 
sorry

theorem problem_two (x : ℝ) (h1 : x > 1) (a : ℝ) 
  (h2 : f x a / (Real.exp 1 + 1) > 2 * Real.exp (x - 1) / ((x + 1) * (x * Real.exp x + 1))) := 
sorry

end problem_one_problem_two_l250_250687


namespace suki_ninth_place_l250_250352

def position (name : String) : ℕ

variable
  {Jonah Aaron Suki Theo Mila Lana : String}
  (p_jonah_ahead_aaron : position "Jonah" = position "Aaron" - 5)
  (p_theo_behind_suki : position "Theo" = position "Suki" + 1)
  (p_mila_behind_aaron : position "Mila" = position "Aaron" + 3)
  (p_suki_behind_lana : position "Suki" = position "Lana" + 3)
  (p_theo_ahead_jonah : position "Theo" = position "Jonah" + 4)
  (p_lana_position : position "Lana" = 7)

theorem suki_ninth_place : position "Suki" = 9 :=
by
  sorry

end suki_ninth_place_l250_250352


namespace cyclic_hexagon_similar_triangles_l250_250818

noncomputable def cyclic_hexagon (A B C D E F : Type*) : Prop := sorry
noncomputable def inter (A B C D : Type*) : Type* := sorry

theorem cyclic_hexagon_similar_triangles
  (A B C D E F O Z X Y : Type*)
  (h_cyclic : cyclic_hexagon A B C D E F)
  (h_eq1 : AB = CD)
  (h_eq2 : CD = EF)
  (h_inter1 : Z = inter A C B D)
  (h_inter2 : X = inter C E D F)
  (h_inter3 : Y = inter E A F B) :
  triangle_similar XYZ BDF :=
begin
  sorry
end

end cyclic_hexagon_similar_triangles_l250_250818


namespace batsman_average_l250_250096

theorem batsman_average (A : ℕ) (H : (16 * A + 82) / 17 = A + 3) : (A + 3 = 34) :=
sorry

end batsman_average_l250_250096


namespace eval_log_example_problem_l250_250622

noncomputable def log_example_problem : ℝ := log 2 (32 * sqrt 8)

theorem eval_log_example_problem :
  (32 = 2^5) →
  (sqrt 8 = 2^(3/2)) →
  log_example_problem = 13 / 2 :=
by
  intro h1 h2
  sorry

end eval_log_example_problem_l250_250622


namespace simplified_product_sequence_l250_250030

theorem simplified_product_sequence :
  (∏ n in finset.range 1009, (5 * (n + 1) + 5) / (5 * (n + 1))) = 1010 := 
by sorry

end simplified_product_sequence_l250_250030


namespace imaginary_part_of_ratio_l250_250697

noncomputable def z1 : ℂ := 1 - 2 * I
noncomputable def z2 : ℂ := -1 - 2 * I  -- Symmetric about the imaginary axis

theorem imaginary_part_of_ratio 
  (z1_z2_symmetric : z2 = conj (z1) * (-1)) :
  complex.im (z2 / z1) = -4 / 5 := by
  rw [z2, z1] at z1_z2_symmetric
  rw [complex.div_eq_mul_inv, complex.mul_conj, complex.inv_apply]
  sorry

end imaginary_part_of_ratio_l250_250697


namespace sin_arccos_l250_250966

theorem sin_arccos (adj hyp : ℝ) (h_adj : adj = 3) (h_hyp : hyp = 5) :
  sin (arccos (adj / hyp)) = 4 / 5 :=
by
  sorry

end sin_arccos_l250_250966


namespace rita_bought_4_pounds_l250_250838

-- Define the conditions
def card_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def amount_left : ℝ := 35.68

-- Define the theorem to prove the number of pounds of coffee bought is 4
theorem rita_bought_4_pounds :
  (card_amount - amount_left) / cost_per_pound = 4 := by sorry

end rita_bought_4_pounds_l250_250838


namespace number_of_integers_satisfying_ineq_l250_250705

theorem number_of_integers_satisfying_ineq : 
  (finset.card {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset) = 7 := 
sorry

end number_of_integers_satisfying_ineq_l250_250705


namespace perfect_square_factors_of_180_l250_250307

theorem perfect_square_factors_of_180 {n : ℕ} (h1 : n = 180) : 
  (∃ f : ℕ → ℕ, (∀ p, (p = 2 ∨ p = 3 ∨ p = 5 → 
    (f p = 0 ∨ (p = 2 ∨ p = 3) ∧ f p = 2 ∧ p ≠ 5))) ∧ 2^(f 2) * 3^(f 3) * 5^(f 5) = n ∧  ((f 2 + f 3 + f 5) % 2 = 0)) → 
  4 :=
begin
  sorry
end

end perfect_square_factors_of_180_l250_250307


namespace all_two_digit_numbers_in_tape_l250_250572

theorem all_two_digit_numbers_in_tape (tape: List ℕ) (h1: ∀ n, 1 ≤ n ∧ n ≤ 1000000 → n ∈ tape) (h2: ∀ i, 0 ≤ i ∧ i < tape.length - 1 → (tape.nth i).isSome ∧ (tape.nth (i+1)).isSome):
  ∀ ab ∈ finset.Ico 10 100, ∃ i, (0 ≤ i ∧ i < tape.length - 1) ∧ (tape.nth i).isSome ∧ (tape.nth (i+1)).isSome ∧ 
    (tape.nth i).get! = ab / 10 ∧ (tape.nth (i+1)).get! = ab % 10 :=
sorry

end all_two_digit_numbers_in_tape_l250_250572


namespace find_m_l250_250694

-- Definitions based on conditions
def Point (α : Type) := α × α

def A : Point ℝ := (2, -3)
def B : Point ℝ := (4, 3)
def C (m : ℝ) : Point ℝ := (5, m)

-- The collinearity condition
def collinear (p1 p2 p3 : Point ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

-- The proof problem
theorem find_m (m : ℝ) : collinear A B (C m) → m = 6 :=
by
  sorry

end find_m_l250_250694


namespace factorization_correct_l250_250989

theorem factorization_correct (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end factorization_correct_l250_250989


namespace integral_problem_l250_250163

theorem integral_problem :
  ∫ x in 0..1, (Real.sqrt (1 - x^2) + 3 * x) = (Real.pi / 4) + (3 / 2) := by
  sorry

end integral_problem_l250_250163


namespace vectors_are_perpendicular_l250_250283

def vector_a := (2: ℝ, -1: ℝ)
def vector_b := (1: ℝ, -3: ℝ)

theorem vectors_are_perpendicular :
  let a := vector_a
  let b := vector_b
  let a_minus_b := (a.1 - b.1, a.2 - b.2)
  a.1 * a_minus_b.1 + a.2 * a_minus_b.2 = 0 :=
by
  sorry

end vectors_are_perpendicular_l250_250283


namespace gcf_of_294_and_108_l250_250502

theorem gcf_of_294_and_108 : Nat.gcd 294 108 = 6 :=
by
  -- We are given numbers 294 and 108
  -- Their prime factorizations are 294 = 2 * 3 * 7^2 and 108 = 2^2 * 3^3
  -- The minimum power of the common prime factors are 2^1 and 3^1
  -- Thus, the GCF by multiplying these factors is 2^1 * 3^1 = 6
  sorry

end gcf_of_294_and_108_l250_250502


namespace fuel_cost_rounded_l250_250948

def odometer_start : ℕ := 52214
def odometer_end : ℕ := 52235
def fuel_efficiency : ℝ := 32 -- miles per gallon
def fuel_price : ℝ := 3.89 -- dollars per gallon

-- Prove that the cost of the fuel used during the trip rounds to $2.55
theorem fuel_cost_rounded :
  let total_miles := odometer_end - odometer_start,
      gallons_used := total_miles / fuel_efficiency,
      total_cost := gallons_used * fuel_price
  in round(total_cost * 100) = 255 :=
by
  sorry

end fuel_cost_rounded_l250_250948


namespace slope_CD_eq_one_l250_250969

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 8*y + 40 = 0

-- The problem statement
theorem slope_CD_eq_one :
  ∃ (C D : ℝ × ℝ),
    circle1 C.1 C.2 ∧ circle2 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 D.1 D.2 ∧
    let slope := (D.2 - C.2) / (D.1 - C.1) in
    slope = 1 :=
sorry

end slope_CD_eq_one_l250_250969


namespace milly_needs_boas_l250_250420

def num_fl_boas (total_flamingos: ℕ) (tail_feathers_per_flamingo: ℕ) (pluckable_percentage: ℚ) (feathers_per_boa: ℕ) : ℕ :=
  total_flamingos * (tail_feathers_per_flamingo * pluckable_percentage) / feathers_per_boa

theorem milly_needs_boas :
  num_fl_boas 480 20 0.25 200 = 12 :=
by
  -- Calculation steps would go here
  sorry

end milly_needs_boas_l250_250420


namespace B_pow_200_eq_identity_l250_250393

open Matrix

noncomputable def B : Matrix (Fin 4) (Fin 4) ℝ := ![
  ![0, 0, 0, 1],
  ![1, 0, 0, 0],
  ![0, 1, 0, 0],
  ![0, 0, 1, 0]
]

theorem B_pow_200_eq_identity : B ^ 200 = (1 : Matrix (Fin 4) (Fin 4) ℝ) :=
by
  sorry

end B_pow_200_eq_identity_l250_250393


namespace sector_to_cone_volume_l250_250927

noncomputable def volume_of_cone_from_sector (r : ℝ) (theta : ℝ) : ℝ :=
  let circumference := theta * r in
  let base_radius := circumference / (2 * Real.pi) in
  let height := Real.sqrt (r^2 - base_radius^2) in
  (1 / 3) * Real.pi * base_radius^2 * height

noncomputable def volume_of_given_cone : ℝ := 
  volume_of_cone_from_sector 6 (Real.pi)

theorem sector_to_cone_volume :
  volume_of_given_cone = 9 * Real.pi * Real.sqrt 3 :=
  sorry

end sector_to_cone_volume_l250_250927


namespace factorization_correct_l250_250988

theorem factorization_correct (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end factorization_correct_l250_250988


namespace simplify_log_expression_l250_250841

theorem simplify_log_expression (a b c d x y : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hx : x > 0) (hy : y > 0) :
  log (a^2 / b) + log (b / c) + log (c / d^2) - log (a^2 * y / (d^2 * x)) = log (x / y) :=
by {
  sorry
}

end simplify_log_expression_l250_250841


namespace no_square_free_decomposition_l250_250135

theorem no_square_free_decomposition (L : ℕ) (n : ℕ) (m : ℕ) (a : ℕ) (b : ℕ) :
  L = 9 → n = 14 → (∀ i, 1 ≤ i ∧ i ≤ n → (a i ≥ 2 ∧ b i ≥ 2)) → 
  ∑ i in Finset.range n, (a i * b i) = L * L →
  ∃ i, a i = b i :=
by
  sorry

end no_square_free_decomposition_l250_250135


namespace find_special_four_digit_square_l250_250633

theorem find_special_four_digit_square :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ ∃ (a b c d : ℕ), 
    n = 1000 * a + 100 * b + 10 * c + d ∧
    n = 8281 ∧
    a = c ∧
    b + 1 = d ∧
    n = (91 : ℕ) ^ 2 :=
by
  sorry

end find_special_four_digit_square_l250_250633


namespace angle_OP_XY_is_zero_l250_250346
-- Import the complete mathematical library

-- Define the problem in Lean 4
def triangle_configuration (X Y Z V H O P : Type)
  (angle_X : ℝ) (angle_Y : ℝ) (XY : ℝ) 
  (YV VZ : ℝ) (YH HX : ℝ)
  (midpoint_XY : (X × Y) → O)
  (midpoint_VH : (V × H) → P)
  (angle_between_OP_XY : (O × P) → (X × Y) → ℝ) : Prop :=
  angle_X = 40 ∧ 
  angle_Y = 52 ∧ 
  XY = 12 ∧ 
  (YV = 2 ∧ VZ = 2) ∧ 
  (YH = 3 ∧ HX = 3) ∧ 
  (∀ x y, midpoint_XY (x, y) = O) ∧ 
  (∀ v h, midpoint_VH (v, h) = P) ∧ 
  angle_between_OP_XY (O, P) (X, Y) = 0

-- Theorem stating the degree measure of the acute angle formed by lines OP and XY
theorem angle_OP_XY_is_zero 
  (X Y Z V H O P : Type)
  (angle_X : ℝ) (angle_Y : ℝ) (XY : ℝ) 
  (YV VZ : ℝ) (YH HX : ℝ)
  (midpoint_XY : (X × Y) → O)
  (midpoint_VH : (V × H) → P)
  (angle_between_OP_XY : (O × P) → (X × Y) → ℝ) :
  triangle_configuration X Y Z V H O P angle_X angle_Y XY YV VZ YH HX midpoint_XY midpoint_VH angle_between_OP_XY →
  angle_between_OP_XY (O, P) (X, Y) = 0 :=
by
  -- Proof is omitted.
  sorry

end angle_OP_XY_is_zero_l250_250346


namespace molecular_weight_of_compound_is_correct_l250_250505

noncomputable def molecular_weight (nC nH nN nO : ℕ) (wC wH wN wO : ℝ) :=
  nC * wC + nH * wH + nN * wN + nO * wO

theorem molecular_weight_of_compound_is_correct :
  molecular_weight 8 18 2 4 12.01 1.008 14.01 16.00 = 206.244 :=
by
  sorry

end molecular_weight_of_compound_is_correct_l250_250505


namespace find_2012th_term_l250_250558

noncomputable def sequence_term (n k : ℕ) : ℚ :=
if h : k ≤ n then (⟨n - k + 1, k⟩ : ℚ) else 0

theorem find_2012th_term :
  let term_2012 := 5 / 59 in
  (∑ i in finset.range 63, i + 1) + 5 = 2012 ∧ sequence_term 63 59 = term_2012 :=
by
  let term_2012 := 5 / 59
  have h_sum : (∑ i in finset.range 63, i + 1) + 5 = 2012 := sorry
  have h_term : sequence_term 63 59 = term_2012 := sorry
  exact ⟨h_sum, h_term⟩

end find_2012th_term_l250_250558


namespace all_are_rational_l250_250618

theorem all_are_rational :
    ∀ (x : ℚ), x = real.sqrt 4 ∨ 
               x = real.cbrt 0.064 ∨ 
               x = (real.root 5 (1 / 32)) ∨ 
               x = (real.cbrt (-8) * real.sqrt ((0.25:ℝ)⁻¹)) → 
               x ∈ ℚ :=
by sorry

end all_are_rational_l250_250618


namespace solve_triangle_proof_problem_l250_250370

noncomputable def triangle_proof_problem : Prop :=
  ∃ (p q r s : ℕ),
    p + q = 66 ∧
    r = 41 ∧
    s = 2 ∧
    AB = 7 ∧
    BC = 8 ∧
    CA = 9 ∧
    DF = 3 ∧
    EF = 8 ∧
    BE = (p + q * sqrt r) / s ∧
    p + q + r + s = 114

theorem solve_triangle_proof_problem : triangle_proof_problem :=
by
  sorry

end solve_triangle_proof_problem_l250_250370


namespace numbers_of_form_10001_are_composite_l250_250431

theorem numbers_of_form_10001_are_composite : ∀ k : ℕ, ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ (p * q = 1 + ∑ i in Finset.range (k + 1), 10 ^ (4 * i)) :=
by
sorry

end numbers_of_form_10001_are_composite_l250_250431


namespace number_of_perfect_square_factors_of_180_l250_250296

-- Define the prime factorization of 180
def prime_factorization_180 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Define what it means to be a factor of 180
def is_factor_of_180 (n : ℕ) : Prop :=
  ∀ p k, (p, k) ∈ prime_factorization_180 → ∃ m, n = p ^ m ∧ m ≤ k

-- Define what it means to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p m, ∃ k, n = p ^ m ∧ even m

-- Function to count factors of 180 that are perfect squares
def count_perfect_square_factors_of_180 : ℕ :=
  List.length [d | d ← List.range 181, is_factor_of_180 d ∧ is_perfect_square d]

-- Main theorem
theorem number_of_perfect_square_factors_of_180 : count_perfect_square_factors_of_180 = 4 := 
by sorry

end number_of_perfect_square_factors_of_180_l250_250296


namespace valid_configuration_l250_250016

variables (P Q R S : Type) [has_dist P] [has_dist Q] [has_dist R] [has_dist S]
variables (a b c : ℝ)

-- Given conditions
axiom collinear_points : collinear P Q R S
axiom PQ_eq_a : dist P Q = a
axiom PR_eq_b : dist P R = b
axiom PS_eq_c : dist P S = c

-- Rotations creating a valid triangle (points P and S coincide)
axiom rotation_valid_triangle : ∃ (P' S' : P), P' = P ∧ S' = S ∧
                               (∃ (Q' R' : P), dist P' Q' = dist P Q ∧ dist R' S' = dist R S ∧ 
                               triangle_inequality P' Q' R' S')

-- Angle at Q less than 120 degrees
axiom angle_at_Q_lt_120 : ∃ θ, θ < 120 ∧ angle_at Q θ < 120

-- Proof challenges
theorem valid_configuration :
  2 * b > c ∧ c > a ∧ c > b ∧ angle_at Q θ < 120 :=
by
  sorry

end valid_configuration_l250_250016


namespace range_of_a_l250_250263

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x - Real.exp (-x) + Real.log (x + Real.sqrt (x^2 + 1))

theorem range_of_a
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f (x^2 + 2) + f (-2 * a * x) ≥ 0) :
  -3/2 ≤ a ∧ a ≤ Real.sqrt 2 :=
sorry

end range_of_a_l250_250263


namespace monotonic_interval_of_f_l250_250469

def f (x : ℝ) : ℝ := -6 / x - 5 * Real.log x

theorem monotonic_interval_of_f :
  ∃ a b : ℝ, (0 < a) ∧ (b = 6 / 5) ∧ (∀ x : ℝ, a < x ∧ x < b → deriv f x > 0) :=
by {
  sorry
}

end monotonic_interval_of_f_l250_250469


namespace lowest_selling_price_l250_250922

/-- Define the variables and constants -/
def production_cost_per_component := 80
def shipping_cost_per_component := 7
def fixed_costs_per_month := 16500
def components_per_month := 150

/-- Define the total variable cost -/
def total_variable_cost (production_cost_per_component shipping_cost_per_component : ℕ) (components_per_month : ℕ) :=
  (production_cost_per_component + shipping_cost_per_component) * components_per_month

/-- Define the total cost -/
def total_cost (variable_cost fixed_costs_per_month : ℕ) :=
  variable_cost + fixed_costs_per_month

/-- Define the lowest price per component -/
def lowest_price_per_component (total_cost components_per_month : ℕ) :=
  total_cost / components_per_month

/-- The main theorem to prove the lowest selling price required to cover all costs -/
theorem lowest_selling_price (production_cost shipping_cost fixed_costs components : ℕ)
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 7)
  (h3 : fixed_costs = 16500)
  (h4 : components = 150) :
  lowest_price_per_component (total_cost (total_variable_cost production_cost shipping_cost components) fixed_costs) components = 197 :=
by
  sorry

end lowest_selling_price_l250_250922


namespace percent_palindromes_with_7_l250_250128

-- Definition: A number is a palindrome if it reads the same forward and backward
def is_palindrome (n : ℕ) : Prop := 
  let digits := repr n in
  digits = reverse digits

-- Definition: Palindromes between 1000 and 2000
def palindromes_1000_to_2000 : Set ℕ := 
  {n | 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n}

-- Definition: Palindromes between 1000 and 2000 that contain at least one 7
def palindromes_with_7_1000_to_2000 : Set ℕ := 
  {n | n ∈ palindromes_1000_to_2000 ∧ '7' ∈ (repr n).toList}

-- The proof statement to be proved
theorem percent_palindromes_with_7 : 
  let total := palindromes_1000_to_2000.toFinset.card in
  let count_with_7 := palindromes_with_7_1000_to_2000.toFinset.card in
  100 * count_with_7 / total = 190 :=
sorry

end percent_palindromes_with_7_l250_250128


namespace max_value_expr_l250_250249

variable (x y z : ℝ)

theorem max_value_expr (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (∃ a, ∀ x y z, (a = (x*y + y*z) / (x^2 + y^2 + z^2)) ∧ a ≤ (Real.sqrt 2) / 2) ∧
  (∃ x' y' z', (x' > 0) ∧ (y' > 0) ∧ (z' > 0) ∧ ((x'*y' + y'*z') / (x'^2 + y'^2 + z'^2) = (Real.sqrt 2) / 2)) :=
by
  sorry

end max_value_expr_l250_250249


namespace jess_correct_answer_l250_250376

theorem jess_correct_answer :
  (∃ y : ℤ, (y - 11) / 5 = 31) → (∃ y : ℤ, y = 166 ∧ ((y - 5) / 11 : ℝ) = 14.6) := 
by
  intro h1
  cases h1 with y h1
  have h2 : y = 166 := by
    linarith
  use y
  split
  · exact h2
  · rw h2
    norm_num
    sorry

end jess_correct_answer_l250_250376


namespace sequence_is_arithmetic_l250_250911

theorem sequence_is_arithmetic 
  (a_n : ℕ → ℤ) 
  (h : ∀ n : ℕ, a_n n = n + 1) 
  : ∀ n : ℕ, a_n (n + 1) - a_n n = 1 :=
by
  sorry

end sequence_is_arithmetic_l250_250911


namespace number_of_true_propositions_l250_250475

-- Defining the propositions as functions
def P (x : ℝ) := x > 1 → x^2 > 1
def negation_P (x : ℝ) := x ≤ 1 → x^2 ≤ 1
def converse_P (x : ℝ) := x^2 > 1 → x > 1
def contrapositive_P (x : ℝ) := x ≤ 1 → x^2 ≤ 1

-- Counting the number of true propositions
def true_propositions_count : ℕ :=
  (if ∀ x, P x then 1 else 0) + 
  (if ∀ x, negation_P x then 1 else 0) + 
  (if ∀ x, converse_P x then 1 else 0) + 
  (if ∀ x, contrapositive_P x then 1 else 0)

-- The theorem statement
theorem number_of_true_propositions : true_propositions_count = 3 :=
by
  sorry

end number_of_true_propositions_l250_250475


namespace smallest_integer_l250_250169

theorem smallest_integer : 
  ∃ n > 1, (∀ k ∈ { (3, 5, 3/5, 2/3), (5, 7, 5/7, 2/5), (7, 9, 7/9, 2/7), (9, 11, 9/11, 2/9) }, 
            let (a, b, q, fp) := k in
            n / q = (n / q).floor + fp) ∧ 
           n = 316 := sorry

end smallest_integer_l250_250169


namespace count_original_scissors_l250_250486

def originalScissors (addedScissors totalScissors : ℕ) : ℕ := totalScissors - addedScissors

theorem count_original_scissors :
  ∃ (originalScissorsCount : ℕ), originalScissorsCount = originalScissors 13 52 := 
  sorry

end count_original_scissors_l250_250486


namespace Jamie_needs_to_climb_40_rungs_l250_250787

-- Define the conditions
def height_of_new_tree : ℕ := 20
def rungs_climbed_previous : ℕ := 12
def height_of_previous_tree : ℕ := 6
def rungs_per_foot := rungs_climbed_previous / height_of_previous_tree

-- Define the theorem
theorem Jamie_needs_to_climb_40_rungs :
  height_of_new_tree * rungs_per_foot = 40 :=
by
  -- Proof placeholder
  sorry

end Jamie_needs_to_climb_40_rungs_l250_250787


namespace circle_center_is_B_l250_250473

-- Definitions for points A, B, C, D on a line
variables {A B C D : Point}
variables (h_line : collinear A B C D)
variables (h_AB_BC: distance A B = distance B C)

-- Definitions for perpendiculars and intersections
variables {P Q K L : Point}
variables (h_perpendicular_B : perpendicular_at B A D P Q)
variables (h_perpendicular_C : perpendicular_at C B D K L)

-- The theorem stating that B is the center of the circle passing through points P, K, L, Q
theorem circle_center_is_B :
  circle_center P K L Q = B :=
sorry

end circle_center_is_B_l250_250473


namespace irrational_roots_of_odd_quadratic_l250_250434

theorem irrational_roots_of_odd_quadratic (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ p q : ℤ, q ≠ 0 ∧ gcd p q = 1 ∧ p * p = a * (p / q) * (p / q) + b * (p / q) + c := sorry

end irrational_roots_of_odd_quadratic_l250_250434


namespace not_enough_funds_to_buy_two_books_l250_250885

def storybook_cost : ℝ := 25.5
def sufficient_funds (amount : ℝ) : Prop := amount >= 50

theorem not_enough_funds_to_buy_two_books : ¬ sufficient_funds (2 * storybook_cost) :=
by
  sorry

end not_enough_funds_to_buy_two_books_l250_250885


namespace tan_angle_BCD_max_value_l250_250492

theorem tan_angle_BCD_max_value :
  ∀ (A B C D : Point) (angle_A : ∠A = 90) (AB AC AD BC DC : ℝ),
  AB = 6 →
  AC = 12 →
  AD = 3 →
  BC = 3 * Real.sqrt 3 →
  DC = 9 →
  ∃ tan_BCD, tan_BCD = Real.sqrt 3 / 3 :=
sorry

end tan_angle_BCD_max_value_l250_250492


namespace find_x0_l250_250411

def f (x : ℝ) : ℝ := x * (2018 + Real.log x)

noncomputable def f_prime (x : ℝ) : ℝ := (f x).derivative

theorem find_x0 (x0 : ℝ) (h : f_prime x0 = 2019) : x0 = 1 := by
  -- computation of f'(x) = 2018 + Real.log x + 1
  sorry

end find_x0_l250_250411


namespace arrange_books_l250_250745

theorem arrange_books (math_books english_books : Fin 4 → ℕ) :
  -- There are 2 groups of books to arrange
  (2.factorial * (4.factorial * 4.factorial)) = 1152 := by
  -- Skip the proof
  sorry

end arrange_books_l250_250745


namespace problem_equivalent_proof_l250_250899

noncomputable def Points := {A B C D : Type*}

def is_on_same_side (A B C D : Points) : Prop :=
  sorry -- Predicate for points C and D being on the same side of line AB

def similar_triangles_condition (A B C D : Points) : Prop :=
  AC * BD = AD * BC

def angle_condition (A B C D : Points) : Prop :=
  ∠ ADB = 90 + ∠ ACB

-- Main statement
theorem problem_equivalent_proof (A B C D : Points)
  (H1 : is_on_same_side A B C D)
  (H2 : similar_triangles_condition A B C D)
  (H3 : angle_condition A B C D) :
  (AB * CD) / (AC * BD) = sqrt 2 ∧
  orthogonal_circles A C D B C D :=
sorry

end problem_equivalent_proof_l250_250899


namespace similar_triangle_side_length_l250_250455

theorem similar_triangle_side_length
  (A_1 A_2 : ℕ)
  (area_diff : A_1 - A_2 = 32)
  (area_ratio : A_1 = 9 * A_2)
  (side_small_triangle : ℕ)
  (side_small_triangle_eq : side_small_triangle = 5)
  (side_ratio : ∃ r : ℕ, r = 3) :
  ∃ side_large_triangle : ℕ, side_large_triangle = side_small_triangle * 3 := by
sorry

end similar_triangle_side_length_l250_250455


namespace workshop_total_workers_l250_250357

theorem workshop_total_workers
  (avg_salary_per_head : ℕ)
  (num_technicians num_managers num_apprentices total_workers : ℕ)
  (avg_tech_salary avg_mgr_salary avg_appr_salary : ℕ) 
  (h1 : avg_salary_per_head = 700)
  (h2 : num_technicians = 5)
  (h3 : num_managers = 3)
  (h4 : avg_tech_salary = 800)
  (h5 : avg_mgr_salary = 1200)
  (h6 : avg_appr_salary = 650)
  (h7 : total_workers = num_technicians + num_managers + num_apprentices)
  : total_workers = 48 := 
sorry

end workshop_total_workers_l250_250357


namespace problem_solution_l250_250686

-- Definitions as identified from conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * sin x + 4

-- Given conditions from the problem statement
axiom cond1 (a b : ℝ) : f (Real.log (Real.log 10 / Real.log 2) / Real.log 10) a b = 5
axiom cond2 : Real.log (Real.log 10 / Real.log 2) / Real.log 10 + Real.log (Real.log 2) = 0

-- The proof goal (no proof required, so we use sorry)
theorem problem_solution (a b : ℝ) : f (Real.log (Real.log 2)) a b = 3 := 
by
  sorry

end problem_solution_l250_250686


namespace inequality_log_l250_250228

variable (a b c : ℝ)
variable (h1 : 1 < a)
variable (h2 : 1 < b)
variable (h3 : 1 < c)

theorem inequality_log (a b c : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c) : 
  2 * ( (Real.log a / Real.log b) / (a + b) + (Real.log b / Real.log c) / (b + c) + (Real.log c / Real.log a) / (c + a) ) 
  ≥ 9 / (a + b + c) := 
sorry

end inequality_log_l250_250228


namespace part1_part2_l250_250822

-- Definitions and conditions
variables {A B C D P Q : Type}
variables [rect : rectangle A B C D] -- Rectangle ABCD with area 2
variables [point_P_on_CD : P ∈ [C, D]] -- P is a point on side CD

-- Part 1: Prove AB ≥ 2BC
theorem part1 (h_area : (rect.area = 2)) :
  rect.side A B ≥ 2 * rect.side B C :=
sorry

-- Part 2: Prove AQ · BQ = 1
theorem part2 (h_area : (rect.area = 2)) (h_incircle: incircle (triangle P A B) Q) :
  (segment A Q).length * (segment B Q).length = 1 :=
sorry

end part1_part2_l250_250822


namespace sin_2alpha_plus_sin_squared_l250_250246

theorem sin_2alpha_plus_sin_squared (α : ℝ) (h : Real.tan α = 1 / 2) : Real.sin (2 * α) + Real.sin α ^ 2 = 1 :=
sorry

end sin_2alpha_plus_sin_squared_l250_250246


namespace cube_painting_distinct_ways_l250_250158

theorem cube_painting_distinct_ways :
  ∃ (n : ℕ), n = 6 ∧ 
  ∀ (cubes : ℕ → list (list color)), 
    (∀ (c : ℕ), let faces := cubes c in 
      faces.length = 6 ∧ 
      (∀ (f1 f2 : ℕ), 
        (f1 ≠ f2 → 
        (faces.nth_le f1 (by sorry) == faces.nth_le f2 (by sorry) ↔
         opposite_faces f1 f2))) ∧ -- each pair of opposite faces has the same color
      (rotational_symmetry_preserved cubes)) :=
sorry

end cube_painting_distinct_ways_l250_250158


namespace largest_prime_2023_digits_l250_250403

theorem largest_prime_2023_digits:
  let q := -- Define q to be the largest prime with 2023 digits
  -- Sorry for the actual definition which would be complex
  q^2 ≡ 1 [MOD 15] :=
sorry

end largest_prime_2023_digits_l250_250403


namespace sum_of_possible_values_of_CD_l250_250178

variable (AB CD : ℝ) (angleA : ℝ) (parallelABCD : Prop)
variable (isArithmeticProgression : Prop) (maxSideAB : Prop)

-- Given conditions
axiom given_conditions :
  AB = 20 ∧ angleA = 90 ∧ parallelABCD ∧
  isArithmeticProgression ∧ maxSideAB

-- Statement of the problem
theorem sum_of_possible_values_of_CD : 
  given_conditions AB CD angleA parallelABCD isArithmeticProgression maxSideAB →
  CD = 18 :=
by
  sorry

end sum_of_possible_values_of_CD_l250_250178


namespace count_integers_satisfying_inequality_l250_250723

theorem count_integers_satisfying_inequality : 
  ∃ (s : Finset ℤ), (∀ n ∈ s, (n - 3) * (n + 5) < 0) ∧ s.card = 7 :=
begin
  sorry
end

end count_integers_satisfying_inequality_l250_250723


namespace count_integers_satisfy_inequality_l250_250730

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end count_integers_satisfy_inequality_l250_250730


namespace complexExpression_evaluation_l250_250912

-- Define the problem
noncomputable def complexExpression := 
  (complex.cos (72 * real.pi / 180) + complex.sin (72 * real.pi / 180) * complex.I) *
  (complex.cos (41 * real.pi / 180) + complex.sin (41 * real.pi / 180) * complex.I) ^ 2 /
  (complex.cos (19 * real.pi / 180) + complex.sin (19 * real.pi / 180) * complex.I)

-- Define the expected answer
noncomputable def expectedAnswer := - real.sqrt 2 / 2 + (real.sqrt 2 / 2) * complex.I

-- The statement saying that the expression equals the expected answer
theorem complexExpression_evaluation :
  complexExpression = expectedAnswer :=
by
  -- This is where the proof would go, but we will add 'sorry' to skip it
  sorry

end complexExpression_evaluation_l250_250912


namespace general_formula_a_Tn_bound_l250_250236

-- Define the sequence a_n and its general formula
def seq_a : ℕ → ℝ
| 0       := 1
| (n+1)   := 2*(n+1)    -- covers the case n >= 1

-- Define sum of the sequence terms S_n
def S (n : ℕ) : ℝ :=
∑ i in finset.range n, seq_a i

-- Define b_n in terms of a_n
def seq_b (n : ℕ) : ℝ :=
1 / (seq_a n * seq_a (n + 1))

-- Define the partial sum T_n
def T (n : ℕ) : ℝ :=
∑ i in finset.range n, seq_b i

-- Formalize the recurrence relation condition
axiom recurrence_relation (n : ℕ) : n ≠ 0 → seq_a (n + 1) = (2 * S n + 2) / n

-- Prove the general formula for seq_a
theorem general_formula_a (n : ℕ) : seq_a n = 
if n = 0 then 1 else 2 * n :=
sorry

-- Prove T_n < 3/8 for all natural numbers n
theorem Tn_bound (n : ℕ) : T n < 3/8 :=
sorry

end general_formula_a_Tn_bound_l250_250236


namespace double_sum_eq_fraction_l250_250608

theorem double_sum_eq_fraction :
  (∀ (n : ℕ), n ≥ 3 → (∀ (k : ℕ), k ≥ 2 ∧ k < n → 
  (∑ n in filter (λ n, n ≥ 3) (range (n+1)), 
  ∑ k in filter (λ k, k ≥ 2 ∧ k < n) (range (n)) (k^2 / 3^(n+k))) = 9 / 128)) :=
by
  exact sorry

end double_sum_eq_fraction_l250_250608


namespace floor_of_5_point_7_l250_250200

theorem floor_of_5_point_7 : Int.floor 5.7 = 5 := by
  sorry

end floor_of_5_point_7_l250_250200


namespace odd_prime_mod_four_l250_250413

-- Define the group G with order 1 + p
variable {G : Type*} [Group G]

-- Define p as an odd prime number
variable {p : ℕ}
hypothesis (hp : Nat.Prime p) (odd_p : p % 2 = 1)

-- Define the order of G to be 1 + p
hypothesis (order_G : Card G = 1 + p)

-- Define that p divides the number of automorphisms of G
variable {α : G ≃* G}
hypothesis (div_p_aut : p ∣ Card (G ≃* G))

-- The target statement
theorem odd_prime_mod_four (p_unique : ∀ p, p ≠ 2) : p % 4 = 3 := 
sorry

end odd_prime_mod_four_l250_250413


namespace televisions_sold_this_black_friday_l250_250009

theorem televisions_sold_this_black_friday 
  (T : ℕ) 
  (h1 : ∀ (n : ℕ), n = 3 → (T + (50 * n) = 477)) 
  : T = 327 := 
sorry

end televisions_sold_this_black_friday_l250_250009


namespace membership_change_l250_250569

theorem membership_change (original : ℝ) (fall_increase : ℝ) (spring_decrease : ℝ): 
  fall_increase = 4 → spring_decrease = 19 → 
  let increased := original + (fall_increase / 100) * original in 
  let decreased := increased - (spring_decrease / 100) * increased in
  (original - decreased) / original * 100 = 15.76 :=
by
  intros h1 h2
  let increased := original + (fall_increase / 100) * original
  let decreased := increased - (spring_decrease / 100) * increased
  have h3 : increased = original + (4 / 100) * original, from sorry
  have h4 : decreased = increased - (19 / 100) * increased, from sorry
  have h5 : (original - decreased) / original * 100 = 15.76, from sorry
  exact h5

end membership_change_l250_250569


namespace value_of_x_l250_250341

theorem value_of_x (x : ℝ) (h : x = 88 * 1.2) : x = 105.6 :=
by
  sorry

end value_of_x_l250_250341


namespace value_of_a_l250_250754
noncomputable def find_a (a b c : ℝ) : ℝ :=
if 2 * b = a + c ∧ (a * c) * (b * c) = ((a * b) ^ 2) ∧ a + b + c = 6 then a else 0

theorem value_of_a (a b c : ℝ) :
  (2 * b = a + c) ∧ ((a * c) * (b * c) = (a * b) ^ 2) ∧ (a + b + c = 6) ∧ (a ≠ c) ∧ (a ≠ b) ∧ (b ≠ c) → a = 4 :=
by sorry

end value_of_a_l250_250754


namespace erasers_pens_markers_cost_l250_250146

theorem erasers_pens_markers_cost 
  (E P M : ℝ)
  (h₁ : E + 3 * P + 2 * M = 240)
  (h₂ : 2 * E + 4 * M + 5 * P = 440) :
  3 * E + 4 * P + 6 * M = 520 :=
sorry

end erasers_pens_markers_cost_l250_250146


namespace total_vegetables_l250_250069

variable (C R T D : ℕ)

def cucumbers : ℕ := 58
def cucumbers_carrots_relation : Prop := C = R + 24
def cucumbers_tomatoes_relation : Prop := C = T - 49
def radishes_carrots_relation : Prop := D = R

theorem total_vegetables :
  cucumbers = 58 →
  cucumbers_carrots_relation →
  cucumbers_tomatoes_relation →
  radishes_carrots_relation →
  (C + R + T + D = 233) :=
by
  intros hC hCR hCT hRD
  sorry

end total_vegetables_l250_250069


namespace sum_of_products_not_zero_l250_250772

theorem sum_of_products_not_zero (n : ℕ) (h_odd : n % 2 = 1) (a : Fin n → Fin n → ℤ)
  (h_a : ∀ i j, a i j = 1 ∨ a i j = -1) :
  let p := λ i, ∏ j, a i j
      q := λ j, ∏ i, a i j in
  ∑ i, p i + ∑ j, q j ≠ 0 :=
by
  sorry

end sum_of_products_not_zero_l250_250772


namespace find_m_values_for_one_real_solution_l250_250645

theorem find_m_values_for_one_real_solution :
  ∀ m : ℝ,
  (∃ x : ℝ, (3 * x + 4) * (x - 8) = -50 + m * x) ∧ 
  (∀ x₁ x₂ : ℝ, (3 * x₁ + 4) * (x₁ - 8) = -50 + m * x₁ → (3 * x₂ + 4) * (x₂ - 8) = -50 + m * x₂ → x₁ = x₂)
  ↔ m = -20 + 6 * real.sqrt 6 ∨ m = -20 - 6 * real.sqrt 6 :=
by
  sorry

end find_m_values_for_one_real_solution_l250_250645


namespace recycling_problem_l250_250497

theorem recycling_problem 
  (n V P : ℕ) (h1 : n = 9) (h2 : V = 20) (h3 : P = 4) :
  let F := (P * n) - V in F = 16 :=
by
  sorry

end recycling_problem_l250_250497


namespace typing_time_l250_250587

-- Definitions based on the problem conditions
def initial_typing_speed : ℕ := 212
def speed_decrease : ℕ := 40
def words_in_document : ℕ := 3440

-- Definition for Barbara's new typing speed
def new_typing_speed : ℕ := initial_typing_speed - speed_decrease

-- Lean proof statement: Proving the time to finish typing is 20 minutes
theorem typing_time :
  (words_in_document / new_typing_speed) = 20 :=
by sorry

end typing_time_l250_250587


namespace bruce_three_times_son_in_six_years_l250_250600

-- Define the current ages of Bruce and his son
def bruce_age : ℕ := 36
def son_age : ℕ := 8

-- Define the statement to be proved
theorem bruce_three_times_son_in_six_years :
  ∃ (x : ℕ), x = 6 ∧ ∀ t, (t = x) → (bruce_age + t = 3 * (son_age + t)) :=
by
  sorry

end bruce_three_times_son_in_six_years_l250_250600


namespace dividend_correct_l250_250523

-- Given constants for the problem
def divisor := 19
def quotient := 7
def remainder := 6

-- Dividend formula
def dividend := (divisor * quotient) + remainder

-- The proof problem statement
theorem dividend_correct : dividend = 139 := by
  sorry

end dividend_correct_l250_250523


namespace total_cost_of_supplies_l250_250149

variable (E P M : ℝ)

open Real

theorem total_cost_of_supplies (h1 : E + 3 * P + 2 * M = 240)
                                (h2 : 2 * E + 4 * M + 5 * P = 440)
                                : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_supplies_l250_250149


namespace floor_add_r_eq_12_2_l250_250208

-- Define r which satisfies the condition
def r (n : ℤ) : ℝ := n + 0.2

-- The main theorem that we need to prove
theorem floor_add_r_eq_12_2 : ∃ r : ℝ, (⌊r⌋ : ℝ) + r = 12.2 ∧ r = 6.2 :=
by
  -- We can simply set r = 6.2 and show that it satisfies the conditions
  use 6.2
  split
  -- Proof that floor of 6.2 plus 6.2 equals 12.2
  { sorry }
  -- Proof that r equals 6.2
  { refl }

end floor_add_r_eq_12_2_l250_250208


namespace abs_diff_of_two_numbers_l250_250480

-- Define the two numbers x and y
variables {x y : ℝ}

-- Given conditions: Sum and Product
def sum_condition := x + y = 30
def product_condition := x * y = 221

-- Proof statement
theorem abs_diff_of_two_numbers :
  sum_condition → product_condition → |x - y| = 4 :=
by
  -- Proof steps omitted
  sorry

end abs_diff_of_two_numbers_l250_250480


namespace count_integers_in_interval_l250_250736

theorem count_integers_in_interval :
  {n : ℤ | -5 < n ∧ n < 3}.finite ∧ {n : ℤ | -5 < n ∧ n < 3}.to_finset.card = 7 := by
sorry

end count_integers_in_interval_l250_250736


namespace w_remaining_time_correct_l250_250427

-- Defining the given constants
def departure (townA townB villageC : Type) : Prop :=
  -- Initial travel time until meeting
  let meeting_time := 12 in
  -- Xiao Zhang's travel time to Village C post meeting
  let zh_travel_post_meeting := 6 in
  
  -- Xiao Wang's remaining travel time
  let expected_remaining_travel := 2 in

  -- Assume speeds and town distances proportional to meeting times
  let sz_ratio := 3/2 in
  
  -- From the speed ratio and meeting time, compute total travel time for Xiao Wang
  let wz_total_travel := meeting_time * 2 in
  let wz_remaining_travel := wz_total_travel - meeting_time - zh_travel_post_meeting in
  
  wz_remaining_travel = expected_remaining_travel

-- Now, stating formally in Lean:
theorem w_remaining_time_correct (townA townB villageC : Type) : 
  departure townA townB villageC :=
  sorry

end w_remaining_time_correct_l250_250427


namespace number_of_ways_reseating_l250_250076

def S : ℕ → ℕ
| 0       := 0
| 1       := 0
| 2       := 1
| (n + 3) := S (n + 2) + S (n + 1)

theorem number_of_ways_reseating : S 12 = 89 := by
  sorry

end number_of_ways_reseating_l250_250076


namespace ice_cream_flavor_variations_l250_250528

variables (r t a b c s : ℕ) -- Declaring all used variables as natural numbers
variable (f : Finset (ℕ × ℕ × ℕ × ℕ × ℕ)) -- Declaring a set of 5-tuples of natural numbers

-- Defining the required proof problem with all conditions and the result
theorem ice_cream_flavor_variations :
  let f := {x : ℕ × ℕ × ℕ × ℕ × ℕ | 
    let (r, t, a, b, c) := x in
    r + t + 6*a + b + c = 62 ∧ 
    r ≥ t ∧ 
    a % 6 = 0 ∧ 
    b ≤ 5 ∧ 
    c ≤ 1} in
  f.card = 2016 :=
sorry

end ice_cream_flavor_variations_l250_250528


namespace minimize_sum_of_distances_l250_250610

noncomputable def point {α : Type*} [division_ring α] := (α × α)

def distance {α : Type*} [field α] (p1 p2 : point α) : α :=
real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def sum_of_distances {α : Type*} [field α] (P A B C D : point α) : α :=
distance P A + distance P B + distance P C + distance P D

theorem minimize_sum_of_distances : 
  ∀ (α : Type*) [field α],
  let A : point α := (2, 4),
      B : point α := (0, 0),
      C : point α := (6, 0),
      D : point α := (6, 3),
      P : point α := (4, 2) in
  (∀ Q : point α, sum_of_distances Q A B C D ≥ sum_of_distances P A B C D) :=
by
  intros
  sorry

end minimize_sum_of_distances_l250_250610


namespace base_5_conversion_correct_l250_250612

def base_5_to_base_10 : ℕ := 2 * 5^2 + 4 * 5^1 + 2 * 5^0

theorem base_5_conversion_correct : base_5_to_base_10 = 72 :=
by {
  -- Proof (not required in the problem statement)
  sorry
}

end base_5_conversion_correct_l250_250612


namespace custom_op_evaluation_l250_250800

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem custom_op_evaluation : custom_op 4 2 = 26 := 
by 
  sorry

end custom_op_evaluation_l250_250800


namespace geometric_series_first_term_l250_250152

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by {
  sorry
}

end geometric_series_first_term_l250_250152


namespace parabola_line_intersection_l250_250078

theorem parabola_line_intersection :
  let a := (3 + Real.sqrt 11) / 2
  let b := (3 - Real.sqrt 11) / 2
  let p1 := (a, (9 + Real.sqrt 11) / 2)
  let p2 := (b, (9 - Real.sqrt 11) / 2)
  (3 * a^2 - 9 * a + 4 = (9 + Real.sqrt 11) / 2) ∧
  (-a^2 + 3 * a + 6 = (9 + Real.sqrt 11) / 2) ∧
  ((9 + Real.sqrt 11) / 2 = a + 3) ∧
  (3 * b^2 - 9 * b + 4 = (9 - Real.sqrt 11) / 2) ∧
  (-b^2 + 3 * b + 6 = (9 - Real.sqrt 11) / 2) ∧
  ((9 - Real.sqrt 11) / 2 = b + 3) :=
by
  sorry

end parabola_line_intersection_l250_250078


namespace cos_squared_eq_half_cos_double_l250_250615

theorem cos_squared_eq_half_cos_double (θ : Real) : 
  ∃ c d : Real, 
    (∀ θ : Real, cos θ ^ 2 = c * cos (2 * θ) + d * cos θ) ∧ 
    c = 1 / 2 ∧ 
    d = 0 := 
by 
  use 1 / 2, 0
  intro θ
  rw [cos_sq_to_cos_double]
  exact ⟨rfl, rfl⟩

end cos_squared_eq_half_cos_double_l250_250615


namespace magnitude_b_when_perpendicular_l250_250655

-- Define the vectors a and b and the condition that a is perpendicular to b
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (2, m)
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Main theorem to prove
theorem magnitude_b_when_perpendicular (m : ℝ) (h : perpendicular a (b m)) : magnitude (b m) = real.sqrt 5 :=
by
  sorry

end magnitude_b_when_perpendicular_l250_250655


namespace cryptarithmetic_proof_l250_250364

theorem cryptarithmetic_proof (A B C D : ℕ) 
  (h1 : A * B = 6) 
  (h2 : C = 2) 
  (h3 : A + B + D = 13) 
  (h4 : A + B + C = D) : 
  D = 6 :=
by
  sorry

end cryptarithmetic_proof_l250_250364


namespace rita_bought_four_pounds_l250_250836

def initial_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def left_amount : ℝ := 35.68

theorem rita_bought_four_pounds :
  (initial_amount - left_amount) / cost_per_pound = 4 :=
by
  sorry

end rita_bought_four_pounds_l250_250836


namespace num_perfect_square_factors_of_180_l250_250310

theorem num_perfect_square_factors_of_180 (n : ℕ) (h : n = 180) :
  ∃ k : ℕ, k = 4 ∧ ∀ d : ℕ, d ∣ n → ∃ a b c : ℕ, d = 2^a * 3^b * 5^c ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 :=
by
  use 4
  sorry

end num_perfect_square_factors_of_180_l250_250310


namespace range_of_a_for_inequality_l250_250219

theorem range_of_a_for_inequality (a : ℝ) : (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ (a ≥ -2) :=
sorry

end range_of_a_for_inequality_l250_250219


namespace integral_of_quadratic_has_minimum_value_l250_250254

theorem integral_of_quadratic_has_minimum_value :
  ∃ m : ℝ, (∀ x : ℝ, x^2 + 2 * x + m ≥ -1) ∧ (∫ x in (1:ℝ)..(2:ℝ), x^2 + 2 * x = (16 / 3:ℝ)) :=
by sorry

end integral_of_quadratic_has_minimum_value_l250_250254


namespace f_1982_l250_250458

-- Define the function f and the essential properties and conditions
def f : ℕ → ℕ := sorry

axiom f_nonneg (n : ℕ) : f n ≥ 0
axiom f_add_property (m n : ℕ) : f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1
axiom f_2 : f 2 = 0
axiom f_3_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

-- Statement of the theorem we want to prove
theorem f_1982 : f 1982 = 660 := 
  by sorry

end f_1982_l250_250458


namespace divisors_pairing_l250_250020

def is_not_perfect_square (n : ℕ) : Prop :=
  ¬∃ m : ℕ, m * m = n

theorem divisors_pairing (n : ℕ) (h₁ : n > 0) (h₂ : is_not_perfect_square n) : 
  ∃ (pairs : (ℕ × ℕ) → Prop), (∀ d ∈ (finset.filter (λ x, n % x = 0) (finset.range (n + 1))), 
    ∃ d' ∈ (finset.filter (λ x, n % x = 0) (finset.range (n + 1))), (pairs (d, d') ∧ (d ∣ d' ∨ d' ∣ d))) :=
sorry

end divisors_pairing_l250_250020


namespace class_speeds_relationship_l250_250964

theorem class_speeds_relationship (x : ℝ) (hx : 0 < x) :
    (15 / (1.2 * x)) = ((15 / x) - (1 / 2)) :=
sorry

end class_speeds_relationship_l250_250964


namespace average_increased_by_three_l250_250107

def original_average (A : ℝ) := A / 16
def final_average (total_runs : ℝ) := total_runs / 17
def increment_in_average := 18 - 15

theorem average_increased_by_three 
  (A : ℝ) (H : 16 * A + 66 = 17 * 18) :
  final_average (16 * A + 66) - original_average (16 * A) = increment_in_average :=
by
  sorry

end average_increased_by_three_l250_250107


namespace arithmetic_seq_problem_l250_250774

variable {a : Nat → ℝ}  -- a_n represents the value at index n
variable {d : ℝ} -- The common difference in the arithmetic sequence

-- Define the general term of the arithmetic sequence
def arithmeticSeq (a : Nat → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
  ∀ n : Nat, a n = a1 + n * d

-- The main proof problem
theorem arithmetic_seq_problem
  (a1 : ℝ)
  (d : ℝ)
  (a : Nat → ℝ)
  (h_arithmetic: arithmeticSeq a a1 d)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : 
  a 7 - (1 / 2) * a 8 = 8 := 
  by
  sorry

end arithmetic_seq_problem_l250_250774


namespace largest_prime_2023_digits_l250_250402

theorem largest_prime_2023_digits:
  let q := -- Define q to be the largest prime with 2023 digits
  -- Sorry for the actual definition which would be complex
  q^2 ≡ 1 [MOD 15] :=
sorry

end largest_prime_2023_digits_l250_250402


namespace floor_of_5_point_7_l250_250201

theorem floor_of_5_point_7 : Int.floor 5.7 = 5 := by
  sorry

end floor_of_5_point_7_l250_250201


namespace correct_statements_l250_250023

-- Definitions of the vector operations.
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

 -- Define the vectors given in the conditions.
def e1 : Vector2D := ⟨2, -3⟩
def e2 : Vector2D := ⟨-1, 3 / 2⟩

def a : Vector2D := ⟨-1, 1⟩
def b : Vector2D := ⟨2, 3⟩

-- Function to calculate the dot product of two vectors.
def dot_product (u v : Vector2D) : ℝ :=
  u.x * v.x + u.y * v.y

-- Function to calculate the projection of one vector onto another.
def projection (u v : Vector2D) : Vector2D :=
  let scalar := (dot_product u v) / (u.x * u.x + u.y * u.y)
  ⟨scalar * u.x, scalar * u.y⟩

-- The statement to be proven in Lean.

theorem correct_statements : 
  (determinant (λ e1 e2 : Vector2D, e1.x * e2.y - e2.x * e1.y) = 0 → False) ∧ -- A is incorrect
  (forall G : Vector2D, triangle_has_centroid G → (GA + GB + GC = Vector2D.zero)) ∧ -- B is correct
  (dot_product a b = 0 → (a = Vector2D.zero ∨ b = Vector2D.zero) = False) ∧ -- C is incorrect
  (projection b a = ⟨a.x / 2, a.y / 2⟩) -- D is correct
:= sorry

end correct_statements_l250_250023


namespace number_of_perfect_square_factors_of_180_l250_250301

theorem number_of_perfect_square_factors_of_180 :
  let prime_factors := (2, 2, 1); 
  let conditions (p1 p2 p3 : ℕ) := 
    p1 ∈ {0, 2} ∧ 
    p2 ∈ {0, 2} ∧ 
    p3 ∈ {0}
  in Σ' x, x ∈ { (p1, p2, p3) | conditions p1 p2 p3 } = 4 :=
begin
  sorry
end

end number_of_perfect_square_factors_of_180_l250_250301


namespace finite_points_coverable_by_non_overlapping_circles_l250_250953

theorem finite_points_coverable_by_non_overlapping_circles 
  (P : Finset (ℝ × ℝ)) :
  ∃ (C : Finset (ℝ × ℝ × ℝ)), 
    (∀ c ∈ C, ∃ p ∈ P, 
      let ⟨cx, cy, r⟩ := c in 
      (p.1 - cx)^2 + (p.2 - cy)^2 ≤ r^2) ∧ 
    (∀ c1 c2 ∈ C, c1 ≠ c2 → 
      let ⟨cx1, cy1, r1⟩ := c1,
          ⟨cx2, cy2, r2⟩ := c2
      in (cx1 - cx2)^2 + (cy1 - cy2)^2 ≥ (r1 + r2)^2) ∧ 
    (C.sum (λ ⟨_, _, r⟩, 2 * r) < P.card) ∧ 
    (∀ c1 c2 ∈ C, c1 ≠ c2 → 
      let ⟨cx1, cy1⟩ := (c1.1, c1.2),
          ⟨cx2, cy2⟩ := (c2.1, c2.2)
      in (cx1 - cx2)^2 + (cy1 - cy2)^2 > 1) :=
sorry

end finite_points_coverable_by_non_overlapping_circles_l250_250953


namespace eval_expression_l250_250658

theorem eval_expression (x : ℝ) (h : x = Real.sqrt 2 + 1) : (x + 1) / (x - 1) = 1 + Real.sqrt 2 := 
by
  sorry

end eval_expression_l250_250658


namespace remaining_halves_cover_one_third_l250_250072

theorem remaining_halves_cover_one_third (a b : ℝ) (covering_segments : list (ℝ × ℝ)) :
  (∀ seg ∈ covering_segments, let (a₁, b₁) := seg in
    (a₁ ≤ b₁) ∧ (a₁ ≤ a) ∧ (b ≤ b₁) ∧ 
    ((a₁ ≤ (a₁ + b₁) / 2 ∧ (a₁ + b₁) / 2 ≤ b₁) ∨ ((a₁ + b₁) / 2 ≤ b₁ ∧ a₁ ≤ (a₁ + b₁) / 2))) →
  ∃ segments_with_removed_halves : list (ℝ × ℝ), 
  (∀ seg ∈ segments_with_removed_halves, let (a₁, b₁) := seg in 
    (a₁ ≤ b₁) ∧ (a ≤ a₁) ∧ (b₁ ≤ b)) ∧
  sum (segments_with_removed_halves.map (λ seg, let (a₁, b₁) := seg in b₁ - a₁)) ≥ (b - a) / 3 :=
sorry

end remaining_halves_cover_one_third_l250_250072


namespace initial_money_of_fox_l250_250850

theorem initial_money_of_fox :
  ∃ a₀ : ℤ, (∀ n : ℕ, n > 0 → ∃ a_n : ℤ, a_n = 2 * (a_n - 1) - 2^{2019}) ∧ (a_{2019} = 0) ∧ (a₀ = 2^{2019} - 1) :=
sorry

end initial_money_of_fox_l250_250850


namespace total_cost_food_l250_250388

theorem total_cost_food
  (beef_pounds : ℕ)
  (beef_cost_per_pound : ℕ)
  (chicken_pounds : ℕ)
  (chicken_cost_per_pound : ℕ)
  (h_beef : beef_pounds = 1000)
  (h_beef_cost : beef_cost_per_pound = 8)
  (h_chicken : chicken_pounds = 2 * beef_pounds)
  (h_chicken_cost : chicken_cost_per_pound = 3) :
  (beef_pounds * beef_cost_per_pound + chicken_pounds * chicken_cost_per_pound = 14000) :=
by
  sorry

end total_cost_food_l250_250388


namespace exists_continuous_function_takes_value_exactly_3_times_l250_250977

theorem exists_continuous_function_takes_value_exactly_3_times :
  ∃ (f : ℝ → ℝ), continuous f ∧ (∀ y : ℝ, ∃ (x1 x2 x3 : ℝ), 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f x1 = y ∧ f x2 = y ∧ f x3 = y) :=
sorry

end exists_continuous_function_takes_value_exactly_3_times_l250_250977


namespace passage_is_deductive_reasoning_l250_250471

-- Define the conditions from the problem
def passage : Prop := 
  "When names are not correct, language will not be used correctly, " ++
  "when language is not used correctly, things will not be done successfully; " ++
  "when things are not done successfully, rituals and music will not flourish; " ++
  "when rituals and music do not flourish, punishments will not be proper; " ++
  "and when punishments are not proper, the people will not know where to put their hands and feet. " ++
  "Therefore, when names are not correct, the people will not know where to put their hands and feet."

-- Define the type of reasoning
def analogical_reasoning : Prop := ∀ {A B : Type} (P : A → Prop) (Q : B → Prop), 
  (∃ a : A, ∃ b : B, P a ∧ Q b ∧ similar a b)

def inductive_reasoning : Prop := ∀ {A : Type} (P : ℕ → A → Prop), 
  (P 0 a ∧ (∀ n, P n a → P (n+1) a) → ∀ n, P n a)

def deductive_reasoning : Prop := ∀ {A : Prop}, 
  (A → ∃ B, B)

def syllogism : Prop := ∀ {A B C : Prop}, 
  (A → B) → (B → C) → (A → C)

-- We are to prove that the reasoning in the passage is deductive.
theorem passage_is_deductive_reasoning : deductive_reasoning :=
by
  sorry

end passage_is_deductive_reasoning_l250_250471


namespace intersection_points_area_l250_250425

noncomputable def C (x : ℝ) : ℝ := (Real.log x)^2

noncomputable def L (α : ℝ) (x : ℝ) : ℝ :=
  (2 * Real.log α / α) * x - (Real.log α)^2

noncomputable def n (α : ℝ) : ℕ :=
  if α < 1 then 0 else if α = 1 then 1 else 2

noncomputable def S (α : ℝ) : ℝ :=
  2 - 2 * α - (1 / 2) * α * (Real.log α)^2 + 2 * α * Real.log α

theorem intersection_points (α : ℝ) (h : 0 < α) : n α = if α < 1 then 0 else if α = 1 then 1 else 2 := by
  sorry

theorem area (α : ℝ) (h : 0 < α ∧ α < 1) : S α = 2 - 2 * α - (1 / 2) * α * (Real.log α)^2 + 2 * α * Real.log α := by
  sorry

end intersection_points_area_l250_250425


namespace cake_icing_cubes_l250_250437

-- Define the dimensions and properties of the cake and smaller cubes
def cake_cuboid := { length := 4, width := 4, height := 4 }
def small_cuboid := { length := 1, width := 1, height := 1 }

-- Define the properties of icing on the cake
def has_icing (x y z : ℕ) : Bool :=
  (z = cake_cuboid.height ∨ y = 0 ∨ y = cake_cuboid.width - 1 ∨ x = 0 ∨ x = cake_cuboid.length - 1) &&
  (z ≠ 0)

-- Define a function that counts cubes with icing on exactly two sides
def count_cubes_with_two_sides_iced : ℕ :=
  let is_iced_side (x y z : ℕ) :=
    has_icing x y z && ((has_icing (x + 1) y z).toNat + (has_icing (x - 1) y z).toNat +
    (has_icing x (y + 1) z).toNat + (has_icing x (y - 1) z).toNat + (has_icing x y (z + 1)).toNat + (has_icing x y (z - 1)).toNat = 2)
  List.sum (List.map
    (fun x => List.sum (List.map
      (fun y => List.sum (List.map
        (fun z => is_iced_side x y z)
        (List.range cake_cuboid.height)))
      (List.range cake_cuboid.width)))
    (List.range cake_cuboid.length))
  
theorem cake_icing_cubes :
  count_cubes_with_two_sides_iced = 20 :=
by
  sorry

end cake_icing_cubes_l250_250437


namespace bacon_calories_percentage_l250_250596

theorem bacon_calories_percentage (total_calories : ℕ) (bacon_strip_calories : ℕ) (num_strips : ℕ)
    (h1 : total_calories = 1250) (h2 : bacon_strip_calories = 125) (h3 : num_strips = 2) :
    (bacon_strip_calories * num_strips * 100) / total_calories = 20 := by
  sorry

end bacon_calories_percentage_l250_250596


namespace opposite_of_pi_is_neg_pi_l250_250861

-- Definition that the opposite of a number x is -1 * x
def opposite (x : ℝ) : ℝ := -1 * x

-- Theorem stating that the opposite of π is -π
theorem opposite_of_pi_is_neg_pi : opposite π = -π := 
  sorry

end opposite_of_pi_is_neg_pi_l250_250861


namespace solve_equation_l250_250439

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2 → 
    (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (6 - 9 * x) / (x - 2) + 2 = 0) →
  (1:ℝ/3 (4/3)) :=
sorry

end solve_equation_l250_250439


namespace toothpicks_150th_stage_l250_250855

-- Define the arithmetic sequence parameters
def first_term : ℕ := 4
def common_difference : ℕ := 4

-- Define the term number we are interested in
def stage_number : ℕ := 150

-- The total number of toothpicks in the nth stage of an arithmetic sequence
def num_toothpicks (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

-- Theorem stating the number of toothpicks in the 150th stage
theorem toothpicks_150th_stage : num_toothpicks first_term common_difference stage_number = 600 :=
by
  sorry

end toothpicks_150th_stage_l250_250855


namespace complex_root_modulus_one_iff_divisible_by_six_l250_250407

theorem complex_root_modulus_one_iff_divisible_by_six (n : ℕ) :
  (∃ z : ℂ, (z^(n+1) - z^n - 1 = 0) ∧ (|z| = 1)) ↔ 6 ∣ (n+2) :=
sorry

end complex_root_modulus_one_iff_divisible_by_six_l250_250407


namespace proof_of_equivalence_l250_250260

-- Definitions of the curve C and the line l, and the polar coordinate system
def curve_C_parametric (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 + 2 * Real.sin α)
def line_l_parametric (t : ℝ) : ℝ × ℝ := (3 + Real.sqrt 2 * t, Real.sqrt 2 * t)

-- Question (1): Polar equation of curve C and Cartesian equation of line l
def polar_equation_curve_C : Prop :=
  ∀ (ρ θ : ℝ), (∃ α : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin α ∧ θ = 2 * Real.sin α) ↔ ρ = 4 * Real.sin θ

def standard_equation_line_l : Prop :=
  ∀ (x y : ℝ), (∃ t : ℝ, x = 3 + Real.sqrt 2 * t ∧ y = Real.sqrt 2 * t) ↔ x - y - 3 = 0

-- Question (2): Minimum distance between points A on curve C and B on line l
def minimum_distance_curve_C_line_l : ℝ :=
  (5 * Real.sqrt 2) / 2 - 2

theorem proof_of_equivalence : polar_equation_curve_C ∧ standard_equation_line_l ∧ (minimum_distance_curve_C_line_l = (5 * Real.sqrt 2 / 2 - 2)) :=
  by
  sorry

end proof_of_equivalence_l250_250260


namespace perfect_square_factors_of_180_l250_250306

theorem perfect_square_factors_of_180 {n : ℕ} (h1 : n = 180) : 
  (∃ f : ℕ → ℕ, (∀ p, (p = 2 ∨ p = 3 ∨ p = 5 → 
    (f p = 0 ∨ (p = 2 ∨ p = 3) ∧ f p = 2 ∧ p ≠ 5))) ∧ 2^(f 2) * 3^(f 3) * 5^(f 5) = n ∧  ((f 2 + f 3 + f 5) % 2 = 0)) → 
  4 :=
begin
  sorry
end

end perfect_square_factors_of_180_l250_250306


namespace simple_interest_rate_l250_250562

theorem simple_interest_rate :
  ∀ (P : ℝ) (R : ℝ),
    (9 / 6) * P = P + (P * R * 12) / 100 →
    R = 100 / 24 :=
by {
  intros P R H,
  sorry
}

end simple_interest_rate_l250_250562


namespace sequence_sum_value_l250_250235

-- Definition of the sequence sum S_n
def S (n : ℕ) : ℤ :=
  let seq_sum := fun (k : ℕ) => if even k then (-1) ^ (k + 1) * (4 * k - 3) else (-1) ^ (k + 1) * (4 * k - 3)
  (Finset.range (n + 1)).sum seq_sum

-- The Lean theorem statement
theorem sequence_sum_value : S 15 + S 22 - S 31 = -76 :=
by
  sorry

end sequence_sum_value_l250_250235


namespace calculate_car_wheels_l250_250390

-- Define the conditions
variables (w_c w_b w_t w_tr w_rs : ℕ)
variables (n : ℕ) (c b t tr rs : ℕ)

-- Specific conditions
def cars_wheels : Prop := c = 2 ∧ 2 * w_c + w_b + w_t + w_tr + w_rs = 25
def bikes_wheels : Prop := b = 2 ∧ 2 * w_b = 4
def trash_can_wheels : Prop := t = 1 ∧ w_t = 2
def tricycle_wheels : Prop := tr = 1 ∧ w_tr = 3
def roller_skates_wheels : Prop := rs = 1 ∧ 4 = w_rs
def total_wheels : Prop := n = 25

-- Main proof problem
theorem calculate_car_wheels : cars_wheels w_c w_b w_t w_tr w_rs c 2 → 
  bikes_wheels 2 2 2 2 w_b w_t w_tr w_rs n → 
  trash_can_wheels 2 w_t w_c w_b w_tr w_rs n → 
  tricycle_wheels 3 w_tr w_b w_c w_rs n → 
  roller_skates_wheels 4 w_rs w_c w_b w_tr w_t n → 
  total_wheels 25 → 
    w_c = 6 :=
by sorry

end calculate_car_wheels_l250_250390


namespace problem_1_problem_2_problem_3_l250_250674

variables {a b : EuclideanGeometry.Vector ℝ} (θ : ℝ) (hθ : θ = 120 * Real.pi / 180)
          (ha : ∥a∥ = 4) (hb : ∥b∥ = 2)

-- Define the conditions for double angle formula and magnitude
def cos_angle : Real :=
  Real.cos θ

theorem problem_1 : a • b = -4 :=
  by sorry

theorem problem_2 : (a + b) • (a - 2 • b) = 12 :=
  by sorry

theorem problem_3 : ∥a + b∥ = 2 * Real.sqrt 3 :=
  by sorry

end problem_1_problem_2_problem_3_l250_250674


namespace daffodil_stamps_count_l250_250849

theorem daffodil_stamps_count (r d : ℕ) (h1 : r = 2) (h2 : r = d) : d = 2 := by
  sorry

end daffodil_stamps_count_l250_250849


namespace greene_family_total_spent_l250_250448

def adm_cost : ℕ := 45

def food_cost : ℕ := adm_cost - 13

def total_cost : ℕ := adm_cost + food_cost

theorem greene_family_total_spent : total_cost = 77 := 
by 
  sorry

end greene_family_total_spent_l250_250448


namespace PA_PB_product_eq_l250_250692

noncomputable def x_l (t : ℝ) := 2 + (real.sqrt 2 / 2) * t
noncomputable def y_l (t : ℝ) := (real.sqrt 2 / 2) * t

noncomputable def x_C (θ : ℝ) := 4 * real.cos θ
noncomputable def y_C (θ : ℝ) := 2 * real.sqrt 3 * real.sin θ

def line_eq {x y : ℝ} : Prop := x - y - 2 = 0
def curve_eq {x y : ℝ} : Prop := x^2 / 16 + y^2 / 12 = 1

def P : ℝ × ℝ := (2, 0)

def intersects (A B : ℝ × ℝ) (l_eq : ∀ t, (x_l t, y_l t) = A ∨ (x_l t, y_l t) = B) 
(C_eq : ∀ θ, (x_C θ, y_C θ) = A ∨ (x_C θ, y_C θ) = B) : Prop :=
l_eq ∧ C_eq

theorem PA_PB_product_eq 
(A B : ℝ × ℝ)
(H_A : intersects A B (λ t, line_eq (x_l t) (y_l t)) (λ θ, curve_eq (x_C θ) (y_C θ))) :
(|dist P A| * |dist P B| = 48 / 7) :=
sorry

end PA_PB_product_eq_l250_250692


namespace total_number_of_coins_is_15_l250_250117

theorem total_number_of_coins_is_15 (x : ℕ) (h : 1*x + 5*x + 10*x + 25*x + 50*x = 273) : 5 * x = 15 :=
by {
  -- Proof omitted
  sorry
}

end total_number_of_coins_is_15_l250_250117


namespace trail_length_l250_250077

theorem trail_length (v_Q : ℝ) (v_P : ℝ) (d_P d_Q : ℝ) 
  (h_vP: v_P = 1.25 * v_Q) 
  (h_dP: d_P = 20) 
  (h_meet: d_P / v_P = d_Q / v_Q) :
  d_P + d_Q = 36 :=
sorry

end trail_length_l250_250077


namespace maximized_area_conditions_l250_250783

theorem maximized_area_conditions (O A M N : Point) (φ ψ β : ℝ) :
  angle O A M = φ → angle O A N = ψ → angle M A N = β → |M A| = |A N| → φ + ψ + β < 180 → 
  φ < 90 - β / 2 ∧ ψ < 90 - β / 2 :=
by
  sorry

end maximized_area_conditions_l250_250783


namespace floor_of_5_point_7_l250_250202

theorem floor_of_5_point_7 : Int.floor 5.7 = 5 := by
  sorry

end floor_of_5_point_7_l250_250202


namespace possible_integer_roots_of_polynomial_l250_250934

theorem possible_integer_roots_of_polynomial (b1 b2 : ℤ) :
  ∃ s : set ℤ, s = {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30} ∧
    ∀ x : ℤ, (x ∈ s) ↔ (∀ P : ℤ → ℤ, P = (λ x, x^3 + b2 * x^2 + b1 * x - 30) → P x = 0) :=
by
  sorry

end possible_integer_roots_of_polynomial_l250_250934


namespace min_expression_value_l250_250244

-- Definitions
def valid_x (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 5
def valid_y (y : ℝ) : Prop := -5 ≤ y ∧ y ≤ -3
def expression_value (x y : ℝ) : ℝ := (x + y) / x

-- Theorem statement
theorem min_expression_value : (∀ x y : ℝ, valid_x x → valid_y y → expression_value x y ≥ (2/5)) ∧ 
                              (∃ x y : ℝ, valid_x x ∧ valid_y y ∧ expression_value x y = (2/5)) := 
by
  sorry

end min_expression_value_l250_250244


namespace irrational_sum_floor_eq_iff_l250_250394

theorem irrational_sum_floor_eq_iff (a b c d : ℝ) (h_irr_a : ¬ ∃ (q : ℚ), a = q) 
                                     (h_irr_b : ¬ ∃ (q : ℚ), b = q) 
                                     (h_irr_c : ¬ ∃ (q : ℚ), c = q) 
                                     (h_irr_d : ¬ ∃ (q : ℚ), d = q) 
                                     (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
                                     (h_pos_c : 0 < c) (h_pos_d : 0 < d)
                                     (h_sum_ab : a + b = 1) :
  (c + d = 1) ↔ (∀ (n : ℕ), ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) :=
sorry

end irrational_sum_floor_eq_iff_l250_250394


namespace tabitha_gave_mom_8_l250_250033

theorem tabitha_gave_mom_8 {initial_amount spent_amount remaining_amount_after_spending : ℝ} (h1 : initial_amount = 25)
  (h2 : spent_amount = 5 * 0.50)
  (h3 : remaining_amount_after_spending = 6)
  (h4 : let invested_and_leftover_amount := 2 * remaining_amount_after_spending in true)
  (h5 : let remaining_amount_after_giving_to_mom := invested_and_leftover_amount - spent_amount in true) :
  ∃ M : ℝ, M + remaining_amount_after_giving_to_mom = 25 ∧ M = 8 :=
by
  -- Proof not required
  sorry

end tabitha_gave_mom_8_l250_250033


namespace floor_5_7_eq_5_l250_250192

theorem floor_5_7_eq_5 : Int.floor 5.7 = 5 := by
  sorry

end floor_5_7_eq_5_l250_250192


namespace complex_sum_l250_250873

theorem complex_sum (a b c d e f : ℝ) : 
  b = 5 → 
  e = -2 * a - c → 
  a + c + e = 3 → 
  b + d + f = 2 → 
  d + 3 * f = -3 :=
by
  intros hb he hae hbd
  subst hb
  subst he
  have h1 : a + c + (-2 * a - c) = 3 := by rw [←hae]
  simp [sub_eq_add_neg, add_comm, add_left_comm] at h1
  obtain ha : a = -3 := by linarith
  have h2 : 5 + d + f = 2 := by rw [hbd]
  simp at h2
  obtain hf : d + f = -3 := by linarith
  have h3 : d + 3 * (-3 - d) = -3 := by 
    simp [add_assoc]
    linarith
  simp at h3
  assumption

end complex_sum_l250_250873


namespace sin_pi_minus_alpha_l250_250367

theorem sin_pi_minus_alpha  (α : ℝ) (hα : sin (Real.pi - α) = sin α) (x y : ℝ) (hP : (x, y) = (Real.sqrt 3, 1)) :
  sin (Real.pi - α) = 1 / 2 :=
by
  sorry

end sin_pi_minus_alpha_l250_250367


namespace circle_radius_zero_l250_250637

theorem circle_radius_zero :
  ∀ (x y : ℝ),
    (4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0) →
    ((x - 1)^2 + (y - 2)^2 = 0) → 
    0 = 0 :=
by
  intros x y h_eq h_circle
  sorry

end circle_radius_zero_l250_250637


namespace A_wins_in_finite_moves_l250_250238

/-- Two players, A and B, play on the coordinate plane. A marks lattice points
where B cannot move. B moves the piece from (x, y) to (x+1, y) or (x, y+1) up to m times 
(1 ≤ m ≤ k). A wins if B cannot move anymore. Prove that A has a winning strategy for 
any positive integer k, ensuring B is trapped in a finite number of moves. -/
theorem A_wins_in_finite_moves (k : ℕ) (hk : k > 0) : 
  ∃ (finite_moves : ℕ), ∀ (B_moves : (ℕ × ℕ) → ℕ), finite_moves < ∞ := 
begin
  sorry -- proof goes here
end

end A_wins_in_finite_moves_l250_250238


namespace evaluate_expression_at_x_eq_2_l250_250891

theorem evaluate_expression_at_x_eq_2 :
  (3 * 2 + 4)^2 = 100 := by
  sorry

end evaluate_expression_at_x_eq_2_l250_250891


namespace smallest_period_of_f_l250_250638

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x + Real.cos x) ^ 2 + 1

theorem smallest_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by
  sorry

end smallest_period_of_f_l250_250638


namespace _l250_250581

noncomputable def area_of_large_triangle : ℝ := (1 / 2) * 10 * 10

noncomputable def area_of_small_triangle (n : ℝ) : ℝ := area_of_large_triangle / n

noncomputable theorem shaded_area_of_partitioned_triangle : ℝ :=
  let large_triangle_area := area_of_large_triangle
  let small_triangle_area := area_of_small_triangle 8
  let shaded_area := 3 * small_triangle_area
  shaded_area

example : shaded_area_of_partitioned_triangle = 18.75 := by
  sorry

end _l250_250581


namespace percentage_palindromes_with_seven_l250_250123

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s == s.reverse

def in_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 2000

def contains_seven (n : ℕ) : Prop :=
  '7' ∈ n.toString.data

def num_palindromes_in_range : ℕ :=
  (Finset.filter (λ n, is_palindrome n) (Finset.filter in_range (Finset.range 2001))).card

def num_palindromes_with_seven : ℕ :=
  (Finset.filter (λ n, contains_seven n) (Finset.filter (λ n, is_palindrome n) (Finset.filter in_range (Finset.range 2001)))).card

theorem percentage_palindromes_with_seven : 
  (num_palindromes_with_seven * 100) / num_palindromes_in_range = 19 := by
  sorry

end percentage_palindromes_with_seven_l250_250123


namespace joe_money_left_l250_250384

theorem joe_money_left (initial_money : ℕ) (notebook_count : ℕ) (book_count : ℕ)
    (notebook_price : ℕ) (book_price : ℕ) (total_spent : ℕ) : 
    initial_money = 56 → notebook_count = 7 → book_count = 2 → notebook_price = 4 → book_price = 7 →
    total_spent = notebook_count * notebook_price + book_count * book_price →
    (initial_money - total_spent) = 14 := 
by 
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end joe_money_left_l250_250384


namespace sum_of_numbers_l250_250481

variable (x y S : ℝ)
variable (H1 : x + y = S)
variable (H2 : x * y = 375)
variable (H3 : (1 / x) + (1 / y) = 0.10666666666666667)

theorem sum_of_numbers (H1 : x + y = S) (H2 : x * y = 375) (H3 : (1 / x) + (1 / y) = 0.10666666666666667) : S = 40 :=
by {
  sorry
}

end sum_of_numbers_l250_250481


namespace AD_plus_BC_eq_AB_plus_CD_l250_250765

-- Definitions based on conditions.
variables {A B C D : Type} [ConvexQuadrilateral A B C D]
variables (AB_parallel_CD : Parallel A B C D)
variables (AC_perpendicular_BD : Perpendicular A B C D)
namespace GeometryProve

-- Statement to prove the equality.
theorem AD_plus_BC_eq_AB_plus_CD
  (h_parallel : AB_parallel_CD)
  (h_perpendicular : AC_perpendicular_BD) :
  AD + BC = AB + CD :=
sorry

end GeometryProve

end AD_plus_BC_eq_AB_plus_CD_l250_250765


namespace max_inscribed_cylinder_volume_percentage_l250_250959

-- Define the geometric quantities.
def R : ℝ := sorry  -- radius of the base of the cone
def M : ℝ := sorry  -- height of the cone
def π : ℝ := Real.pi  -- value of pi

-- Define the volume function for the cylinder and the cone.
def V_cylinder (r m : ℝ) : ℝ := π * r^2 * m
def V_cone : ℝ := (1/3) * π * R^2 * M

-- Define the function for the percentage volume of the largest inscribed cylinder to the volume of the cone.
def percentage_volume (r : ℝ) : ℝ := 
  let m : ℝ := M * (R - r) / R
  let V_cyl : ℝ := V_cylinder r m
  (V_cyl / V_cone) * 100

-- The maximum volume of the inscribed cylinder is 44 4/9 % of the volume of the cone.
theorem max_inscribed_cylinder_volume_percentage : percentage_volume (2/3 * R) = 400 / 9 := sorry

end max_inscribed_cylinder_volume_percentage_l250_250959


namespace relationship_between_D_and_A_l250_250753

variables (A B C D : Prop)

def sufficient_not_necessary (P Q : Prop) : Prop := (P → Q) ∧ ¬ (Q → P)
def necessary_not_sufficient (P Q : Prop) : Prop := (Q → P) ∧ ¬ (P → Q)
def necessary_and_sufficient (P Q : Prop) : Prop := (P ↔ Q)

-- Conditions
axiom h1 : sufficient_not_necessary A B
axiom h2 : necessary_not_sufficient C B
axiom h3 : necessary_and_sufficient D C

-- Proof Goal
theorem relationship_between_D_and_A : necessary_not_sufficient D A :=
by
  sorry

end relationship_between_D_and_A_l250_250753


namespace max_abs_sum_x_y_l250_250328

theorem max_abs_sum_x_y (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * real.sqrt 2 :=
sorry

end max_abs_sum_x_y_l250_250328


namespace find_x_l250_250691

def log_sqrt_x34 (x : ℝ) := Real.log (2 * x + 23) / Real.log (Real.sqrt (x + 34))
def log_x4_sq (x : ℝ) := Real.log (x + 34) / Real.log ((x + 4) * (x + 4))
def log_sqrt_2x23 (x : ℝ) := Real.log (-x - 4) / Real.log (Real.sqrt (2 * x + 23))

theorem find_x : ∃ x : ℝ, 
  2 * x + 23 > 0 ∧ x + 34 > 0 ∧ -x - 4 > 0 ∧ 
  (log_sqrt_x34 x = log_x4_sq x ∧ log_sqrt_x34 x = log_sqrt_2x23 x + 1 ∨
   log_x4_sq x = log_sqrt_2x23 x ∧ log_sqrt_2x23 x = log_sqrt_x34 x + 1 ∨
   log_sqrt_2x23 x = log_sqrt_x34 x ∧ log_sqrt_x34 x = log_x4_sq x + 1) 
   ∧ x = -9 := 
by
  sorry

end find_x_l250_250691


namespace simple_interest_rate_l250_250563

theorem simple_interest_rate :
  ∀ (P : ℝ) (R : ℝ),
    (9 / 6) * P = P + (P * R * 12) / 100 →
    R = 100 / 24 :=
by {
  intros P R H,
  sorry
}

end simple_interest_rate_l250_250563


namespace coeff_sum_eq_twenty_l250_250319

theorem coeff_sum_eq_twenty 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h : ((2 * x - 3) ^ 5) = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) :
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ = 20 :=
by
  sorry

end coeff_sum_eq_twenty_l250_250319


namespace exponential_equality_l250_250221

theorem exponential_equality (x : ℝ) (hx : 2^x * 8^(3 * x) = 16^(5 * x)) : x = 0 :=
by
  sorry

end exponential_equality_l250_250221


namespace area_ratio_l250_250944

theorem area_ratio (
    ABCD : CyclicQuadrilateral,
    center_in_ABCD : CenterInsideCircumcircle ABCD,
    shortest_side : ∃t : ℝ, √(4 - t^2) = ABCD.shortest_side,
    longest_side : ∃t : ℝ, t = ABCD.longest_side ∧ √2 < t ∧ t < 2,
    tangents_intersections : ∀ A B A' B' C' D',
        TangentIntersection ABCD A B A' ∧
        TangentIntersection ABCD B C B' ∧
        TangentIntersection ABCD C D C' ∧
        TangentIntersection ABCD D A D'
) : 2 ≤ area_ratio ABCD.toTangents ABCD := 
sorry

end area_ratio_l250_250944


namespace problem1_problem2_l250_250167

-- Problem 1: Prove \(\sqrt{6} \times \sqrt{3} - 6\sqrt{\frac{1}{2}} = 0\)
theorem problem1 : sqrt 6 * sqrt 3 - 6 * sqrt (1 / 2) = 0 := by
  sorry

-- Problem 2: Prove \(\frac{\sqrt{20} + \sqrt{5}}{\sqrt{5}} = 3\)
theorem problem2 : (sqrt 20 + sqrt 5) / sqrt 5 = 3 := by
  sorry

end problem1_problem2_l250_250167


namespace max_abs_sum_x_y_l250_250327

theorem max_abs_sum_x_y (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * real.sqrt 2 :=
sorry

end max_abs_sum_x_y_l250_250327


namespace shoe_length_size_15_l250_250118

theorem shoe_length_size_15:
  ∀ (L : ℝ), 
  (∀ n : ℕ, 8 ≤ n ∧ n ≤ 17) ∧
  ∀ n : ℕ, (L + (n - 8) * (1/4)) = 
    if n = 17 then
      L * (1 + 0.10)
    else 
      L + (n - 8) * (1/4) ->
  L + (15 - 8) * (1/4) = 24.25 :=
by
  sorry

end shoe_length_size_15_l250_250118


namespace weightlifter_total_weight_l250_250570

theorem weightlifter_total_weight (w : ℕ) (h : ℕ) (w_lift : w = 7) (h_hands : h = 2) :
  2 * w = 14 :=
by
  rw [← h_hands, ← w_lift]
  show 2 * 7 = 14
  exact rfl

end weightlifter_total_weight_l250_250570


namespace village_X_population_l250_250884

/-- 
Village X has a population that is decreasing at the rate of 1,200 per year.
Village Y has a population of 42,000 that is increasing at a rate of 800 per year.
In 17 years, the population of the two villages will be equal.
Prove that the current population of Village X is 76,000.
--/
theorem village_X_population (P_x : ℤ) :
  (P_x - 1,200 * 17 = 42,000 + 800 * 17) → P_x = 76,000 :=
by
  intro h
  sorry

end village_X_population_l250_250884


namespace period_and_phase_shift_l250_250506

theorem period_and_phase_shift (b c : ℝ) (h₁ : b = 3) (h₂ : c = π / 4) :
  (∀ x, y = sin (b * x + c) → (period y = 2 * π / b ∧ phase_shift y = -c / b)) :=
by
  sorry

end period_and_phase_shift_l250_250506


namespace ratio_of_work_capacity_l250_250449

theorem ratio_of_work_capacity (work_rate_A work_rate_B : ℝ)
  (hA : work_rate_A = 1 / 45)
  (hAB : work_rate_A + work_rate_B = 1 / 18) :
  work_rate_A⁻¹ / work_rate_B⁻¹ = 3 / 2 :=
by
  sorry

end ratio_of_work_capacity_l250_250449


namespace count_integers_n_satisfying_inequality_l250_250715

theorem count_integers_n_satisfying_inequality :
  {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset.card = 7 :=
by sorry

end count_integers_n_satisfying_inequality_l250_250715


namespace non_adjacent_plate_arrangement_l250_250119

noncomputable def count_non_adjacent_arrangements : ℕ :=
  let total_arrangements := (15.factorial / (6.factorial * 3.factorial * 3.factorial * 2.factorial * (1.factorial * 15)));
  let green_adjacent := (13.factorial / (6.factorial * 3.factorial * 2.factorial * 2.factorial * (1.factorial * 13)));
  let orange_adjacent := (14.factorial / (6.factorial * 3.factorial * 3.factorial * 1.factorial * (1.factorial * 14)));
  let both_adjacent := (12.factorial / (6.factorial * 3.factorial * 2.factorial * (1.factorial * 12)));
  total_arrangements - (green_adjacent + orange_adjacent - both_adjacent)

theorem non_adjacent_plate_arrangement :
  count_non_adjacent_arrangements = (15.factorial / (6.factorial * 3.factorial * 3.factorial * 2.factorial * 1.factorial * 15)) - ((13.factorial / (6.factorial * 3.factorial * 2.factorial * 2.factorial * 1.factorial * 13)) + (14.factorial / (6.factorial * 3.factorial * 3.factorial * 1.factorial * 1.factorial * 14)) - (12.factorial / (6.factorial * 3.factorial * 2.factorial * 1.factorial * 12))) := 
by sorry

end non_adjacent_plate_arrangement_l250_250119


namespace small_mold_radius_l250_250928

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2.0 / 3.0) * real.pi * r^3

theorem small_mold_radius :
  let large_bowl_radius := 2.0
  let num_small_molds := 64.0
  let total_volume_small_molds := hemisphere_volume large_bowl_radius
  ∃ (small_mold_radius : ℝ), 
    num_small_molds * hemisphere_volume small_mold_radius = total_volume_small_molds ∧
    small_mold_radius = 1.0 / 2.0 :=
by
  sorry

end small_mold_radius_l250_250928


namespace percentage_increase_in_length_l250_250464

-- Definitions and conditions
variables (L B : ℝ) -- Original length and breadth
variables (x : ℝ) -- Percentage increase in length
variable (new_length : ℝ) -- New length
variable (new_breadth : ℝ) -- New breadth
variable (new_area : ℝ) -- New area

-- Condition definitions
def original_length := L
def original_breadth := B
def percentage_increase_length := x / 100
def increased_length := original_length * (1 + percentage_increase_length)
def increased_breadth := original_breadth * 1.15
def original_area := original_length * original_breadth
def expected_new_area := original_area * 1.2075

-- Proof statement
theorem percentage_increase_in_length 
  (h1 : new_length = increased_length)
  (h2 : new_breadth = increased_breadth)
  (h3 : new_area = new_length * new_breadth)
  (h4 : new_area = expected_new_area) :
  x = 5 :=
sorry

end percentage_increase_in_length_l250_250464


namespace remainder_of_prime_division_l250_250251

theorem remainder_of_prime_division
  (p : ℕ) (hp : Nat.Prime p)
  (r : ℕ) (hr : r = p % 210) 
  (hcomp : ¬ Nat.Prime r)
  (hsum : ∃ a b : ℕ, r = a^2 + b^2) : 
  r = 169 := 
sorry

end remainder_of_prime_division_l250_250251


namespace trajectory_of_circle_minimum_S_value_l250_250661

noncomputable def trajectory_equation (x y : ℝ) := y^2 = 4 * x

theorem trajectory_of_circle (x y : ℝ) (E : ℝ × ℝ) (hE : E = (2, 0)) (length_PQ : ℝ) (hP : length_PQ = 4) :
  trajectory_equation x y :=
by sorry

noncomputable def minimization_S (A B : ℝ × ℝ) (OA OB : ℝ × ℝ) (F : ℝ × ℝ) (S : ℝ) :=
  (OA.1 * OB.1 + OA.2 * OB.2 = -4) →
  F = (1, 0) →
  S = ((1 / 2) * |(F.1 - A.1) ∗ (F.2 - A.2)| + 
      |(A.1 - B.1) * (A.2 - B.2)|) →
  S ≥ 4 * sqrt 3

theorem minimum_S_value (A B : ℝ × ℝ) (OA OB : ℝ × ℝ) (F : ℝ × ℝ) (S : ℝ) 
  (H : OA.1 * OB.1 + OA.2 * OB.2 = -4) (hF : F = (1, 0)) :
  minimization_S A B OA OB F S  :=
by sorry

end trajectory_of_circle_minimum_S_value_l250_250661


namespace division_of_15_by_neg_5_l250_250164

theorem division_of_15_by_neg_5 : 15 / (-5) = -3 :=
by
  sorry

end division_of_15_by_neg_5_l250_250164


namespace geometric_series_first_term_l250_250151

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by 
  sorry

end geometric_series_first_term_l250_250151


namespace count_integers_n_satisfying_inequality_l250_250716

theorem count_integers_n_satisfying_inequality :
  {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset.card = 7 :=
by sorry

end count_integers_n_satisfying_inequality_l250_250716


namespace max_friday_13ths_in_non_leap_year_l250_250701

def is_friday (d : ℕ) : Prop := d % 7 = 5

def day_of_the_week (day : ℕ) := day % 7

def calculate_mod7_dates_in_non_leap_year : list ℕ :=
  [13, 44, 72, 103, 133, 164, 194, 225, 256, 286, 317, 347].map day_of_the_week

def find_friday_13ths (start_day : ℕ) : list ℕ :=
  calculate_mod7_dates_in_non_leap_year.filter (λ d, is_friday (d + start_day))

theorem max_friday_13ths_in_non_leap_year : 3 = find_friday_13ths 5

end max_friday_13ths_in_non_leap_year_l250_250701


namespace max_area_triangle_ABC_l250_250372

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
and given the conditions a = 6 and sqrt(7) * b * cos(A) = 3 * a * sin(B),
prove that the maximum area of the triangle is 9 * sqrt(7). -/
theorem max_area_triangle_ABC (A B C : ℝ) (a b c : ℝ)
  (ha : a = 6)
  (hconds : sqrt(7) * b * Real.cos A = 3 * a * Real.sin B):
  (1 / 2 * b * c * Real.sin A) ≤ 9 * sqrt(7) :=
sorry

end max_area_triangle_ABC_l250_250372


namespace sylvester_sharadek_claim_l250_250187

-- Define the conditions: number of questions allowed, presence of the word in the dictionary, and dictionary size
def questions_allowed : ℕ := 20
def word_in_dictionary : Prop := true
def dictionary_size : ℕ := 1000000

-- Define the proposition that Dr. Sylvester Sharadek can always guess the word in 20 questions.
theorem sylvester_sharadek_claim : 
  word_in_dictionary → dictionary_size ≤ 1000000 → questions_allowed ≥ 20 → true :=
by
  intros _ _ _
  exact true.intro

end sylvester_sharadek_claim_l250_250187


namespace f_is_odd_f_monotonic_increasing_intervals_l250_250266

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  simp [f]
  sorry -- Proof needed

theorem f_monotonic_increasing_intervals :
  ∀ x : ℝ, (x < -real.sqrt (1 / 3) → deriv f x > 0) ∧ (x > real.sqrt (1 / 3) → deriv f x > 0) :=
by
  intro x
  split
  · intro h
    simp [f, deriv]
    linarith -- Derivation and proof needed
  · intro h
    simp [f, deriv]
    linarith -- Derivation and proof needed
  sorry -- Further details needed

end f_is_odd_f_monotonic_increasing_intervals_l250_250266


namespace percent_of_300_is_66_l250_250895

theorem percent_of_300_is_66 (Part Whole : ℝ) (h1 : Part = 66) (h2 : Whole = 300) : (Part / Whole) * 100 = 22 := 
by
  rw [h1, h2]
  norm_num
  sorry

end percent_of_300_is_66_l250_250895


namespace sequence_product_mod_six_l250_250215

theorem sequence_product_mod_six : 
  (∏ n in Finset.range 16, 7 + 10 * n) % 6 = 1 := by
  sorry

end sequence_product_mod_six_l250_250215


namespace rate_of_simple_interest_l250_250560

-- Define the principal amount and time
variables (P : ℝ) (R : ℝ) (T : ℝ := 12)

-- Define the condition that the sum becomes 9/6 of itself in 12 years (T)
def simple_interest_condition (P : ℝ) (R : ℝ) (T : ℝ) : Prop :=
  (9 / 6) * P - P = P * R * T

-- Define the main theorem stating the rate R is 1/24
theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ := 12) (h : simple_interest_condition P R T) : 
  R = 1 / 24 := 
sorry

end rate_of_simple_interest_l250_250560


namespace extremum_identity_l250_250415

noncomputable def f (x : ℝ) := x * Real.sin x

theorem extremum_identity (x₀ : ℝ) (h_extremum : ∀ x, deriv f x₀ = 0) :
  (1 + x₀^2) * (1 + Real.cos (2 * x₀)) = 2 :=
by
  have hf : deriv f x₀ = Real.sin x₀ + x₀ * Real.cos x₀ := 
    calc
      deriv f x₀ = deriv (λ x, x * Real.sin x) x₀ : by rfl
      ... = Real.sin x₀ + x₀ * Real.cos x₀ : by simp[deriv_mul, deriv_sin]
  sorry

end extremum_identity_l250_250415


namespace kindergarten_library_models_l250_250848

theorem kindergarten_library_models
  (paid : ℕ)
  (reduced_price : ℕ)
  (models_total_gt_5 : ℕ)
  (bought : ℕ) 
  (condition : paid = 570 ∧ reduced_price = 95 ∧ models_total_gt_5 > 5 ∧ bought = 3 * (2 : ℕ)) :
  exists x : ℕ, bought / 3 = x ∧ x = 2 :=
by
  sorry

end kindergarten_library_models_l250_250848


namespace area_of_D_n_l250_250793

def floor (x : ℝ) : ℝ := Real.floor x

noncomputable def D_n (n : ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x ≥ 0 ∧ (x / (n + 1/2) ≤ y ∧ y ≤ floor (x + 1) - x)}

noncomputable def area_D_n (n : ℝ) : ℝ := sorry  -- Function to compute the area (not defined here)

theorem area_of_D_n (n : ℝ) (h : 0 < n) : 
  area_D_n n = (1 / 2) * ((n + 3/2) / (n + 1/2)) :=
sorry

end area_of_D_n_l250_250793


namespace acute_angled_inequality_l250_250222

variables {A B C S : Type} [EuclideanGeometry S] (r : ℝ)
variables (SA SB SC : ℝ) (acute_angled_triangle : Prop)

def is_acute_angled (ABC : Triangle) : Prop :=
  ABC.angle A B C < π / 2 ∧ ABC.angle B C A < π / 2 ∧ ABC.angle C A B < π / 2

theorem acute_angled_inequality (ABC : Triangle) (S : Point S) (r : ℝ)
  (h₁ : centroid ABC = S)
  (h₂ : ∀ x ∈ {A, B, C}, circumradius ABC = r)
  (h₃ : ∀ x ∈ {A, B, C}, dist S x = SA ∨ dist S x = SB ∨ dist S x = SC) :
  SA^2 + SB^2 + SC^2 > 8 * r^2 / 3 ↔ is_acute_angled ABC :=
sorry

end acute_angled_inequality_l250_250222


namespace variance_calculation_l250_250842

section
variables (scores : List ℝ) (n : ℝ)
def mean (l : List ℝ) : ℝ := (l.sum) / n
def variance (l : List ℝ) (m : ℝ) : ℝ := (l.map (λ x => (x - m) ^ 2)).sum / n

theorem variance_calculation :
  ∀ (scores : List ℝ), scores = [8, 5, 2, 5, 6, 4] →
  let n := 6 in
  variance scores (mean scores) = 10 / 3 :=
begin
  sorry -- This should be replaced by the complete proof
end
end

end variance_calculation_l250_250842


namespace min_length_M_inter_N_l250_250277

variable (m n : ℝ)

def M : Set ℝ := { x | m ≤ x ∧ x ≤ m + 3 / 4 }
def N : Set ℝ := { x | n - 1 / 3 ≤ x ∧ x ≤ n }
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem min_length_M_inter_N : M ⊆ P → N ⊆ P → (∃ m n, 
  let I := { x | (m ≤ x ∧ x ≤ m + 3 / 4) ∧ (n - 1 / 3 ≤ x ∧ x ≤ n) } in
  (Sup I - Inf I) = 1 / 12) :=
by
  sorry

end min_length_M_inter_N_l250_250277


namespace Jackson_game_time_l250_250785

/-- Jackson's grade increases by 15 points for every hour he spends studying, 
    and his grade is 45 points, prove that he spends 9 hours playing video 
    games when he spends 3 hours studying and 1/3 of his study time on 
    playing video games. -/
theorem Jackson_game_time (S G : ℕ) (h1 : 15 * S = 45) (h2 : G = 3 * S) : G = 9 :=
by
  sorry

end Jackson_game_time_l250_250785


namespace sin_of_cos_AMB_angle_B_projection_AB_BC_l250_250760

noncomputable def SineLaw (a A b B : ℝ) : Prop := (a / sin A) = (b / sin B)
noncomputable def CosineLaw (a b c A : ℝ) : Prop := a^2 = b^2 + c^2 - 2 * b * c * cos A

theorem sin_of_cos_AMB (A B : ℝ) (h : cos (A - B) * cos B - sin (A - B) * sin B = -3/5) : 
sin A = 4/5 := 
by sorry

theorem angle_B_projection_AB_BC (a b : ℝ) (h1 : a = 4 * sqrt 2) (h2 : b = 5) : 
B = pi / 4 ∧ (a * b * cos \ (pi / 4)) / (b) = -sqrt 2 / 2 := 
by sorry

end sin_of_cos_AMB_angle_B_projection_AB_BC_l250_250760


namespace equal_angles_YB1Z_XB1Z_l250_250111

theorem equal_angles_YB1Z_XB1Z
  (A B C I C₁ A₁ B₁ X Y Z : Point)
  (h1 : OnCircle I A₁)
  (h2 : OnCircle I B₁)
  (h3 : OnCircle I C₁)
  (h4 : Touches A B C₁)
  (h5 : Touches B C A₁)
  (h6 : Touches C A B₁)
  (h7 : Intersect AI A₁C₁ X)
  (h8 : Intersect CI A₁C₁ Y)
  (h9 : Intersect B₁I A₁C₁ Z) :
  ∠ Y B₁ Z = ∠ X B₁ Z := 
by
  sorry

end equal_angles_YB1Z_XB1Z_l250_250111


namespace new_video_card_cost_l250_250387

def initial_pc_cost : ℕ := 1200
def old_video_card_sale : ℕ := 300
def total_spent : ℕ := 1400

theorem new_video_card_cost : 
  ∃ (C_new : ℕ), C_new = total_spent - (initial_pc_cost - old_video_card_sale) := 
begin
  use 500,
  sorry
end

end new_video_card_cost_l250_250387


namespace smile_region_area_correct_l250_250132

noncomputable def smile_region_area : ℝ :=
(15 - 4 * real.sqrt 2) * real.pi - 4

theorem smile_region_area_correct :
  ∀ (A B C D E F : Type) (r : ℝ) (center_C : C = A)
    (radius_2 : r = 2)
    (point_D_on_semicircle : D ∈ (circle.center C, r/2))
    (CD_perpendicular_AB : line_perpendicular_to CD AB)
    (extend_to_E : line_extend_to BD E)
    (extend_to_F : line_extend_to AD F) 
    (arc_center_B : arc_center AE B)
    (arc_center_A : arc_center BF A)
    (arc_center_D : arc_center EF D),
  {
    have area_smile : real,
    exact smile_region_area,
    sorry
  } 

end smile_region_area_correct_l250_250132


namespace positive_square_factors_of_180_l250_250291

theorem positive_square_factors_of_180 :
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  factors = 4 :=
by
  let prime_factorization_180 := (2^2, 3^2, 5^1)
  let num_even_exponents (exp : ℕ) := if exp = 1 then 1 else 2 in
  let factors := num_even_exponents 2 * num_even_exponents 2 * num_even_exponents 1 in
  show factors = 4
  sorry

end positive_square_factors_of_180_l250_250291


namespace floor_5_7_l250_250195

theorem floor_5_7 : Int.floor 5.7 = 5 :=
by
  sorry

end floor_5_7_l250_250195


namespace trees_distance_l250_250766

theorem trees_distance (num_trees : ℕ) (yard_length : ℕ) (trees_at_end : Prop) (tree_count : num_trees = 26) (yard_size : yard_length = 800) : 
  (yard_length / (num_trees - 1)) = 32 := 
by
  sorry

end trees_distance_l250_250766


namespace segments_equal_l250_250373

-- Import the necessary library
variable {Circle : Type} [MetricSpace Circle]

def midpoint (P Q : Circle) : Circle := sorry
-- Placeholder for the midpoint function

variables (A B C D M N O P : Circle)

-- Conditions as hypotheses in Lean
axiom rectangle_in_circle : ∃ rect : set Circle,
  isRectangle rect ∧ intersects rect Circle = {A, B, C, D}

axiom midpoint_chords : midpoint A B = M ∧ midpoint C D = N
                      ∧ midpoint A C = O ∧ midpoint B D = P

-- Theorem to prove the segments are equal
theorem segments_equal : ∀ (MN OR : set Circle),
  MN = line_segment M N ∧ OR = line_segment O R → MN = OR := 
by { 
  -- This section is to be filled with a full proof
  sorry
  }

end segments_equal_l250_250373


namespace max_value_of_a_l250_250400

theorem max_value_of_a (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h1 : a < 3 * b) (h2 : b < 2 * c) (h3 : c < 5 * d) (h4 : d < 150) : a ≤ 4460 :=
by
  sorry

end max_value_of_a_l250_250400


namespace find_m_l250_250747

theorem find_m (x n m : ℝ) (h : (x + n)^2 = x^2 + 4*x + m) : m = 4 :=
sorry

end find_m_l250_250747


namespace polar_equation_of_curve_C_l250_250034

theorem polar_equation_of_curve_C (x y : ℝ) (h : x^2 + y^2 - 2*x = 0) :
  ∃ (ρ θ : ℝ), (x = ρ * cos θ ∧ y = ρ * sin θ) → ρ = 2 * cos θ :=
begin
  sorry
end

end polar_equation_of_curve_C_l250_250034


namespace total_travel_time_correct_l250_250391

-- Define the conditions
def highway_distance : ℕ := 100 -- miles
def mountain_distance : ℕ := 15 -- miles
def break_time : ℕ := 30 -- minutes
def time_on_mountain_road : ℕ := 45 -- minutes
def speed_ratio : ℕ := 5

-- Define the speeds using the given conditions.
def mountain_speed := mountain_distance / time_on_mountain_road -- miles per minute
def highway_speed := speed_ratio * mountain_speed -- miles per minute

-- Prove that total trip time equals 240 minutes
def total_trip_time : ℕ := 2 * (time_on_mountain_road + (highway_distance / highway_speed)) + break_time

theorem total_travel_time_correct : total_trip_time = 240 := 
by
  -- to be proved
  sorry

end total_travel_time_correct_l250_250391


namespace part1_part2_l250_250227

-- Condition definitions
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x
def g (x : ℝ) : ℝ := Real.log x
def h (x : ℝ) (a : ℝ) : ℝ := f x a + g x

-- Theorem statement for part 1
theorem part1 (a : ℝ) : (∀ x : ℝ, x > 0 → f x a ≥ g x) ↔ a ≤ 1 :=
sorry

-- Theorem statement for part 2
theorem part2 (a x1 x2 m : ℝ) (hx1 : 0 < x1 ∧ x1 < 1/2) 
  (hx1_crit : 2 * x1 - a + 1/x1 = 0) 
  (hx2_crit : 2 * x2 - a + 1/x2 = 0) 
  (hx2_pos : x2 > 1) : h x1 a - h x2 a > m ↔ m ≤ 3/4 - Real.log 2 :=
sorry

end part1_part2_l250_250227


namespace intersection_empty_l250_250242

open Set

def setA : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ 2 * x - y = 0}
def setB : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2 * x - 3}

theorem intersection_empty : setA ∩ setB = ∅ := by
  sorry

end intersection_empty_l250_250242


namespace equation_nec_not_suff_for_x_eq_1_l250_250456

theorem equation_nec_not_suff_for_x_eq_1 : 
  (∀ x, (x^2 + 2 * x - 3 = 0) → (x = -3 ∨ x = 1)) → ((x = 1) ↔ (x^2 + 2 * x - 3 = 0 := leafy_necessary (x = 1)) ∧ ¬ (x^2 + 2 * x - 3 = 0 = by_eq [x = 1])) :=
sorry

end equation_nec_not_suff_for_x_eq_1_l250_250456


namespace find_number_l250_250602

theorem find_number (x : ℕ) : x * 9999 = 4691130840 → x = 469200 :=
by
  intros h
  sorry

end find_number_l250_250602


namespace derivative_at_1_l250_250656

noncomputable def f (a x : ℝ) : ℝ := a^x * x^a

theorem derivative_at_1 (a : ℝ) : deriv (λ x : ℝ, f a x) 1 = a * log a + a^2 := 
by 
  sorry

end derivative_at_1_l250_250656


namespace marcus_brought_30_peanut_butter_cookies_l250_250375

/-- Jenny brought in 40 peanut butter cookies. -/
def jenny_peanut_butter_cookies := 40

/-- Jenny brought in 50 chocolate chip cookies. -/
def jenny_chocolate_chip_cookies := 50

/-- Marcus brought in 20 lemon cookies. -/
def marcus_lemon_cookies := 20

/-- The total number of non-peanut butter cookies is the sum of chocolate chip and lemon cookies. -/
def non_peanut_butter_cookies := jenny_chocolate_chip_cookies + marcus_lemon_cookies

/-- The total number of peanut butter cookies is Jenny's plus Marcus'. -/
def total_peanut_butter_cookies (marcus_peanut_butter_cookies : ℕ) := jenny_peanut_butter_cookies + marcus_peanut_butter_cookies

/-- If Renee has a 50% chance of picking a peanut butter cookie, the number of peanut butter cookies must equal the number of non-peanut butter cookies. -/
theorem marcus_brought_30_peanut_butter_cookies (x : ℕ) : total_peanut_butter_cookies x = non_peanut_butter_cookies → x = 30 :=
by
  sorry

end marcus_brought_30_peanut_butter_cookies_l250_250375


namespace num_perfect_square_factors_of_180_l250_250313

theorem num_perfect_square_factors_of_180 (n : ℕ) (h : n = 180) :
  ∃ k : ℕ, k = 4 ∧ ∀ d : ℕ, d ∣ n → ∃ a b c : ℕ, d = 2^a * 3^b * 5^c ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 :=
by
  use 4
  sorry

end num_perfect_square_factors_of_180_l250_250313


namespace gray_region_area_l250_250170

theorem gray_region_area
  (center_C : ℝ × ℝ) (r_C : ℝ)
  (center_D : ℝ × ℝ) (r_D : ℝ)
  (C_center : center_C = (3, 5)) (C_radius : r_C = 5)
  (D_center : center_D = (13, 5)) (D_radius : r_D = 5) :
  let rect_area := 10 * 5
  let semi_circle_area := 12.5 * π
  rect_area - 2 * semi_circle_area = 50 - 25 * π := 
by 
  sorry

end gray_region_area_l250_250170


namespace certain_event_l250_250510

theorem certain_event (event_heads : Prop) (event_rain : Prop) (event_water : Prop) (event_draw : Prop) :
  (¬event_heads) → (¬event_rain) → event_water → (¬event_draw) → event_water :=
begin
  intros h_not_heads h_not_rain h_water h_not_draw,
  exact h_water,
end

end certain_event_l250_250510


namespace find_s_l250_250619

theorem find_s 
  (a b c x s z : ℕ)
  (h1 : a + b = x)
  (h2 : x + c = s)
  (h3 : s + a = z)
  (h4 : b + c + z = 16) : 
  s = 8 := 
sorry

end find_s_l250_250619


namespace rectangle_dimensions_l250_250080

theorem rectangle_dimensions 
  (total_wire_length : ℝ)
  (total_wire_length = 240)
  (r1 r2 r3 : ℝ)
  (h_ratio : r1 = 3 ∧ r2 = 2 ∧ r3 = 1)
  : 
  ∃ (length width height : ℝ), 
    length = 30 ∧ width = 20 ∧ height = 10 :=
sorry

end rectangle_dimensions_l250_250080


namespace floor_of_neg_3_point_7_l250_250985

def floor_function (x : ℝ) : ℤ := Int.floor x

theorem floor_of_neg_3_point_7 : floor_function (-3.7) = -4 :=
by
  sorry

end floor_of_neg_3_point_7_l250_250985


namespace min_circles_ge_max_disjoint_l250_250406

variable (M : Type) [Polygon M]
variable (r : ℝ) (r_pos : r = 1)

def min_circles_covering (M : Type) [Polygon M] (r : ℝ) : ℕ := sorry -- definition to be provided
def max_disjoint_circles (M : Type) [Polygon M] (r : ℝ) : ℕ := sorry -- definition to be provided

theorem min_circles_ge_max_disjoint (r_pos : r = 1):
  ∀ (M : Type) [Polygon M], min_circles_covering M r ≥ max_disjoint_circles M r :=
sorry

end min_circles_ge_max_disjoint_l250_250406


namespace sqrt_ax3_eq_negx_sqrt_ax_l250_250322

variable (a x : ℝ)
variable (ha : a < 0) (hx : x < 0)

theorem sqrt_ax3_eq_negx_sqrt_ax : Real.sqrt (a * x^3) = -x * Real.sqrt (a * x) := by
  sorry

end sqrt_ax3_eq_negx_sqrt_ax_l250_250322


namespace diameter_count_le_n_l250_250767

section DiameterProof

variables {Point : Type*} (P : set Point) (n : ℕ)
variable [finite P]

def segment_length (p1 p2 : Point) := real
def is_diameter (d : real) (p1 p2 : Point) := segment_length p1 p2 = d

theorem diameter_count_le_n (h_n_ge_3 : 3 ≤ n)
  (h_points : fintype P)
  (h_P_card : fintype.card P = n)
  (d : real)
  (h_d : ∀ p1 p2 ∈ P, segment_length p1 p2 ≤ d)
  (diameters : set (Point × Point))
  (h_diameters : ∀ (p1 p2 : Point), (p1, p2) ∈ diameters ↔ p1 ∈ P ∧ p2 ∈ P ∧ is_diameter d p1 p2) :
  diameters.size ≤ n := 
sorry

end DiameterProof

end diameter_count_le_n_l250_250767


namespace concurrency_of_cevians_l250_250571

variables (A B C : Point) (A' B' C' : Point)

-- Assumptions based on the given conditions
variables (acute_angled_triangle : is_acute_angled_triangle A B C)
          (A'_center_square_on_BC_AB_AC : is_center_of_square_with_vertices_on_sides A' B C A C)
          (B'_center_square_on_CA_AB_BC : is_center_of_square_with_vertices_on_sides B' C A A B)
          (C'_center_square_on_AB_BC_CA : is_center_of_square_with_vertices_on_sides C' A B B C)

theorem concurrency_of_cevians :
  are_concurrent (line_through A A') (line_through B B') (line_through C C') := 
sorry

end concurrency_of_cevians_l250_250571


namespace remainder_expression_l250_250896

theorem remainder_expression (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v) (h3 : v < y) : 
  (x + 3 * u * y) % y = v := 
by
  sorry

end remainder_expression_l250_250896


namespace solve_for_x_l250_250334

theorem solve_for_x (x y : ℤ) 
  (h1 : 3 ^ x * 4 ^ y = 19683)
  (h2 : x - y = 9) :
  x = 9 :=
by {
  sorry
}

end solve_for_x_l250_250334


namespace combined_width_approximately_8_l250_250878

noncomputable def C1 := 352 / 7
noncomputable def C2 := 528 / 7
noncomputable def C3 := 704 / 7

noncomputable def r1 := C1 / (2 * Real.pi)
noncomputable def r2 := C2 / (2 * Real.pi)
noncomputable def r3 := C3 / (2 * Real.pi)

noncomputable def W1 := r2 - r1
noncomputable def W2 := r3 - r2

noncomputable def combined_width := W1 + W2

theorem combined_width_approximately_8 :
  |combined_width - 8| < 1 :=
by
  sorry

end combined_width_approximately_8_l250_250878


namespace percent_increase_l250_250160

theorem percent_increase (P : ℝ) :
  let P1 := 1.25 * P in
  let P2 := 1.44 * P1 in
  ((P2 - P) / P) * 100 = 80 := 
by
  let P1 := 1.25 * P
  let P2 := 1.44 * P1
  have h : ((P2 - P) / P) * 100 = 80 := sorry
  exact h

end percent_increase_l250_250160


namespace total_problems_l250_250583

variable (Martha Jenna Mark Angela Total : ℕ)

-- Conditions
def martha_problems := 2
def jenna_problems := 4 * martha_problems - 2
def mark_problems := jenna_problems / 2
def angela_problems := 9

-- Proof statement
theorem total_problems :
  Total = Martha + Jenna + Mark + Angela →
  Total = 20 :=
by
  have Martha := martha_problems
  have Jenna := jenna_problems
  have Mark := mark_problems
  have Angela := angela_problems
  calc
    Total = Martha + Jenna + Mark + Angela : by sorry
    ...   = 2 + 6 + 3 + 9 : by sorry
    ...   = 20 : by sorry

end total_problems_l250_250583


namespace radius_of_sector_is_12_l250_250037

noncomputable def sector_radius : ℝ :=
  let θ := 40
  let area := 50.28571428571428
  let pi := Real.pi
  let formula := λ (θ : ℝ) (pi : ℝ) (r : ℝ), (θ / 360) * pi * r^2
  sqrt (area * 360 / (θ * pi))

theorem radius_of_sector_is_12 :
  sector_radius = 12 := by
  sorry

end radius_of_sector_is_12_l250_250037


namespace expected_value_bounds_l250_250484

def prob_red_second_box : ℚ := 5 / 9

def prob_red_nth_box (n : ℕ) : ℚ := 
  1 / 2 + 1 / 2 * (1 / 3)^n

theorem expected_value_bounds (n : ℕ) : 
  let E_X := 3 / 2 + 1 / 2 * (1 / 3)^(n-1) in 
  3 / 2 < E_X ∧ E_X ≤ 2 :=
by
  sorry

end expected_value_bounds_l250_250484


namespace odd_number_probability_l250_250483

theorem odd_number_probability :
  let cards := finset.range 10 in
  let odd_cards := {n ∈ cards | n % 2 = 1}.card in
  (odd_cards : ℚ) / cards.card = 5 / 9 :=
by
  sorry

end odd_number_probability_l250_250483


namespace diameter_of_circle_l250_250495

theorem diameter_of_circle (r : ℝ) :
  ∃ (d : ℝ), 
  (let l1 := 3 + 4 in 
   let l2 := 6 + 2 in
   let m1 := l1 / 2 in
   let m2 := l2 / 2 in
   ∃ radius : ℝ,
   radius^2 = m1^2 + r^2 ∧
   radius^2 = m2^2 + r^2 ∧
   d = 2 * radius) ∧
  d = real.sqrt 65 :=
begin
  sorry
end

end diameter_of_circle_l250_250495


namespace slope_angle_of_line_l250_250871

theorem slope_angle_of_line {A B : ℝ × ℝ} (hA : A = (-3, 5)) (hB : B = (1, 1)) :
  ∃ θ : ℝ, θ = 135 ∧ tan θ = (1 - 5) / (1 - (-3)) :=
by
  use 135
  split
  repeat {sorry}

end slope_angle_of_line_l250_250871


namespace range_of_a_opposite_sides_l250_250865

theorem range_of_a_opposite_sides {a : ℝ} (h : (0 + 0 - a) * (1 + 1 - a) < 0) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_opposite_sides_l250_250865


namespace line_perpendicularity_l250_250644

theorem line_perpendicularity (k b : ℝ) :
  ¬(∃ k b : ℝ, ∀ y, x = k * y + b → is_perpendicular_to_y_axis x) ∧
  (∃ k b : ℝ, k = 0 ∧ (∀ y, x = b → is_perpendicular_to_x_axis x)) :=
by
  sorry

end line_perpendicularity_l250_250644


namespace age_difference_l250_250548

def A := 10
def B := 8
def C := B / 2
def total_age (A B C : ℕ) : Prop := A + B + C = 22

theorem age_difference (A B C : ℕ) (hB : B = 8) (hC : B = 2 * C) (h_total : total_age A B C) : A - B = 2 := by
  sorry

end age_difference_l250_250548


namespace least_multiple_of_33_gt_500_is_528_l250_250086

-- We define the conditions given in the problem:
def is_multiple_of_33 (n : ℕ) : Prop := ∃ k : ℕ, n = 33 * k
def greater_than_500 (n : ℕ) : Prop := n > 500
def least_positive_multiple_of_33_greater_than_500 (n : ℕ) : Prop :=
  is_multiple_of_33(n) ∧ greater_than_500(n) ∧ ∀ m : ℕ, is_multiple_of_33(m) ∧ greater_than_500(m) → m ≥ n

-- We state the theorem using the above definitions:
theorem least_multiple_of_33_gt_500_is_528 : least_positive_multiple_of_33_greater_than_500 528 :=
by
  sorry

end least_multiple_of_33_gt_500_is_528_l250_250086


namespace inequality_proof_l250_250751

theorem inequality_proof (x y : ℝ) (h1 : x - y > -x) (h2 : x + y > y) : x > 0 ∧ y < 2x := by
  sorry

end inequality_proof_l250_250751


namespace percent_palindromes_with_7_l250_250127

-- Definition: A number is a palindrome if it reads the same forward and backward
def is_palindrome (n : ℕ) : Prop := 
  let digits := repr n in
  digits = reverse digits

-- Definition: Palindromes between 1000 and 2000
def palindromes_1000_to_2000 : Set ℕ := 
  {n | 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n}

-- Definition: Palindromes between 1000 and 2000 that contain at least one 7
def palindromes_with_7_1000_to_2000 : Set ℕ := 
  {n | n ∈ palindromes_1000_to_2000 ∧ '7' ∈ (repr n).toList}

-- The proof statement to be proved
theorem percent_palindromes_with_7 : 
  let total := palindromes_1000_to_2000.toFinset.card in
  let count_with_7 := palindromes_with_7_1000_to_2000.toFinset.card in
  100 * count_with_7 / total = 190 :=
sorry

end percent_palindromes_with_7_l250_250127


namespace trapezoid_circle_tangent_CT_l250_250819

theorem trapezoid_circle_tangent_CT :
  ∀ (A B C D T : Type) [convex_trapezoid A B C D]
    (angle_DAB_90 : ∠DAB = 90)
    (angle_ABC_90 : ∠ABC = 90)
    (DA_eq_2 : DA = 2)
    (AB_eq_3 : AB = 3)
    (BC_eq_8 : BC = 8)
    (circle_passing_through_A_and_tangent_to_CD : circle_passing_through A ω ∧ tangent_in_T CD ω)
    (center_of_omega_lies_on_BC : center ω ∈ line BC),
  segment_CT (C, T) = 4 * (√5) - (√7) :=
by
  sorry

end trapezoid_circle_tangent_CT_l250_250819


namespace unique_solution_l250_250630

def is_prime (n : ℕ) : Prop := Nat.Prime n

def eq_triple (m p q : ℕ) : Prop :=
  2 ^ m * p ^ 2 + 1 = q ^ 5

theorem unique_solution (m p q : ℕ) (h1 : m > 0) (h2 : is_prime p) (h3 : is_prime q) :
  eq_triple m p q ↔ (m, p, q) = (1, 11, 3) := by
  sorry

end unique_solution_l250_250630


namespace machine_stops_at_n_l250_250929

-- Define the events and probability measures
section

variables {n : ℕ}

-- Probability of stopping after exactly n integers
noncomputable def probability_stops (n : ℕ) : ℚ :=
  (Nat.factorial (n - 1) * (2^n - n - 1)) / (∏ r in finset.range n, (2^r.succ - 1))

-- Theorem statement
theorem machine_stops_at_n (n : ℕ) : 
  probability_stops n = (Nat.factorial (n - 1) * (2^n - n - 1)) / (∏ r in finset.range n, (2^r.succ - 1)) := 
sorry

end

end machine_stops_at_n_l250_250929


namespace percentage_palindromes_with_seven_l250_250121

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s == s.reverse

def in_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 2000

def contains_seven (n : ℕ) : Prop :=
  '7' ∈ n.toString.data

def num_palindromes_in_range : ℕ :=
  (Finset.filter (λ n, is_palindrome n) (Finset.filter in_range (Finset.range 2001))).card

def num_palindromes_with_seven : ℕ :=
  (Finset.filter (λ n, contains_seven n) (Finset.filter (λ n, is_palindrome n) (Finset.filter in_range (Finset.range 2001)))).card

theorem percentage_palindromes_with_seven : 
  (num_palindromes_with_seven * 100) / num_palindromes_in_range = 19 := by
  sorry

end percentage_palindromes_with_seven_l250_250121


namespace num_perfect_square_factors_of_180_l250_250311

theorem num_perfect_square_factors_of_180 (n : ℕ) (h : n = 180) :
  ∃ k : ℕ, k = 4 ∧ ∀ d : ℕ, d ∣ n → ∃ a b c : ℕ, d = 2^a * 3^b * 5^c ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 :=
by
  use 4
  sorry

end num_perfect_square_factors_of_180_l250_250311


namespace fibonacci_problem_l250_250847

theorem fibonacci_problem 
  (F : ℕ → ℕ)
  (h1 : F 1 = 1)
  (h2 : F 2 = 1)
  (h3 : ∀ n ≥ 3, F n = F (n - 1) + F (n - 2))
  (a b c : ℕ)
  (h4 : F c = 2 * F b - F a)
  (h5 : F c - F a = F a)
  (h6 : a + c = 1700) :
  a = 849 := 
sorry

end fibonacci_problem_l250_250847


namespace prob_same_team_l250_250348

/-- Given three teams: red, blue, and yellow,
    and students A and B each randomly choosing one team,
    the probability that students A and B choose the same team
    is 1/3. -/
theorem prob_same_team (teams : Finset String) (A_choice B_choice : String)
  (h_teams : teams = {"red", "blue", "yellow"})
  (h_A_choice : A_choice ∈ teams) (h_B_choice : B_choice ∈ teams) :
  ↑(teams.card) = 3 →
  (Finset.filter (λ (t : String), t = A_choice ∧ t = B_choice) teams).card / teams.card = 1 / 3 :=
by
  sorry

end prob_same_team_l250_250348
