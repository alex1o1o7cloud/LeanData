import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Arithmetic.Basic
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Permutation
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Polytope
import Mathlib.Probability.Distributions.Poisson
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.MetricSpace.Basic
import algebra.big_operators.basic
import combinatorics.pigeonhole
import data.real.basic

namespace max_value_fraction_ratio_tangent_line_through_point_l245_245270

theorem max_value_fraction_ratio (x y : ℝ) (h : x^2 + y^2 - 2*x - 2 = 0) :
  ∃ (max_value : ℝ), max_value = 2 + sqrt 6 :=
sorry

theorem tangent_line_through_point (x y : ℝ) (h : x^2 + y^2 - 2*x - 2 = 0) :
  ∃ (m : ℝ), m ≠ 0 ∧ (x, y) = (0, sqrt 2) → ∀ (x' y' : ℝ), y' = m * x' + sqrt 2 → x' - sqrt 2 * y' + 2 = 0 :=
sorry

end max_value_fraction_ratio_tangent_line_through_point_l245_245270


namespace length_of_PQ_l245_245348

theorem length_of_PQ
  (k : ℝ) -- height of the trapezoid
  (PQ RU : ℝ) -- sides of trapezoid PQRU
  (A1 : ℝ := (PQ * k) / 2) -- area of triangle PQR
  (A2 : ℝ := (RU * k) / 2) -- area of triangle PUR
  (ratio_A1_A2 : A1 / A2 = 5 / 2) -- given ratio of areas
  (sum_PQ_RU : PQ + RU = 180) -- given sum of PQ and RU
  : PQ = 900 / 7 :=
by
  sorry

end length_of_PQ_l245_245348


namespace choir_members_unique_l245_245901

theorem choir_members_unique (n : ℕ) :
  (n % 10 = 6) ∧ 
  (n % 11 = 6) ∧ 
  (150 ≤ n) ∧ 
  (n ≤ 300) → 
  n = 226 := 
by
  sorry

end choir_members_unique_l245_245901


namespace integral_value_l245_245918

noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.exp x

theorem integral_value :
  ∫ x in 0..1, f x = Real.exp 1 := by
  sorry

end integral_value_l245_245918


namespace graph_passes_through_quadrants_l245_245897

def linear_function (x : ℝ) : ℝ := -5 * x + 5

theorem graph_passes_through_quadrants :
  (∃ x y : ℝ, linear_function x = y ∧ x > 0 ∧ y > 0) ∧  -- Quadrant I
  (∃ x y : ℝ, linear_function x = y ∧ x < 0 ∧ y > 0) ∧  -- Quadrant II
  (∃ x y : ℝ, linear_function x = y ∧ x > 0 ∧ y < 0)    -- Quadrant IV
  :=
by
  sorry

end graph_passes_through_quadrants_l245_245897


namespace coprime_lin_comb_exists_l245_245756

theorem coprime_lin_comb_exists (a b n : ℕ) 
  (h_coprime : Nat.coprime a b) 
  (h_n : n > a * b) 
  : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ n = a * x + b * y := 
by 
  sorry

end coprime_lin_comb_exists_l245_245756


namespace log_expression_eval_l245_245187

theorem log_expression_eval :
  log 4 (64 * real.cbrt 16 * real.root 32 5) = 3.967 :=
by
  have h1 : 64 = 4^3 := sorry
  have h2 : real.cbrt 16 = 4^(2/3) := sorry
  have h3 : real.root 32 5 = 4^0.3 := sorry
  have h4 : log 4 (4^(3 + 2/3 + 0.3)) = 3.967 := sorry
  have h5 : log 4 (4^(3.967)) = 3.967 := by
    rw [log_pow (by norm_num : 4 > 0) (by norm_num : 3.967 > 0)]
  rw [← h1, ← h2, ← h3] at h5
  assumption

end log_expression_eval_l245_245187


namespace pie_piece_problem_l245_245044

theorem pie_piece_problem :
  ∃ (ΠИPΟΓ ΚУСΟΚ: Nat), 
    (∃ (Π И Р Ο Γ К У С Ο К : Fin 10), 
       ΠИPΟΓ = 10000 * Π + 1000 * И + 100 * Р + 10 * Ο + Γ ∧ ΚУСΟΚ = 10000 * К + 1000 * У + 100 * С + 10 * Ο + К ∧
       7 * ΚУСΟΚ = ΠИPΟΓ ∧ 
       ∀ (Π' И' Р' Ο' Γ' К' У' С' ΟК': Fin 10), 
         ΠИPΟΓ' = 10000 * Π' + 1000 * И' + 100 * Р' + 10 * О' + Γ' → 
         ΚУСΟΚ' = 10000 * К' + 1000 * У' + 100 * С' + 10 * О' + К' → 
         7 * ΚУСΟК' = ΠИPΟΓ' → 
         (ΚУСΟК ≤ ΚУСΟΚ')) := sorry

end pie_piece_problem_l245_245044


namespace smallest_possible_positive_difference_l245_245505

def Vovochka_sum (a b : Nat) : Nat :=
  let ha := a / 100
  let ta := (a / 10) % 10
  let ua := a % 10
  let hb := b / 100
  let tb := (b / 10) % 10
  let ub := b % 10
  1000 * (ha + hb) + 100 * (ta + tb) + (ua + ub)

def correct_sum (a b : Nat) : Nat :=
  a + b

def difference (a b : Nat) : Nat :=
  abs (Vovochka_sum a b - correct_sum a b)

theorem smallest_possible_positive_difference :
  ∀ a b : Nat, (∃ a b : Nat, a > 0 ∧ b > 0 ∧ difference a b = 1800) :=
by
  sorry

end smallest_possible_positive_difference_l245_245505


namespace Jirina_number_l245_245357

theorem Jirina_number (a b c d : ℕ) (h_abcd : 1000 * a + 100 * b + 10 * c + d = 1468) :
  (1000 * a + 100 * b + 10 * c + d) +
  (1000 * a + 100 * d + 10 * c + b) = 3332 ∧ 
  (1000 * a + 100 * b + 10 * c + d)+
  (1000 * c + 100 * b + 10 * a + d) = 7886 :=
by
  sorry

end Jirina_number_l245_245357


namespace investment_amount_l245_245182

def monthly_interest_payment : ℝ := 240
def annual_interest_rate : ℝ := 0.09

theorem investment_amount : 
  let annual_interest_payment := monthly_interest_payment * 12,
      principal := annual_interest_payment / annual_interest_rate
  in principal = 32000 :=
by
  let annual_interest_payment := 240 * 12
  let principal := annual_interest_payment / 0.09
  show principal = 32000
  sorry

end investment_amount_l245_245182


namespace inequality_ABC_l245_245994

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_ABC_l245_245994


namespace total_cost_after_discounts_l245_245356

-- Definition of initial conditions
def packs_red := 5
def packs_yellow := 4
def packs_blue := 3

def balls_per_pack_red := 18
def balls_per_pack_yellow := 15
def balls_per_pack_blue := 12

def price_per_ball_red := 1.50
def price_per_ball_yellow := 1.25
def price_per_ball_blue := 1.00

def discount_red := 0.10
def discount_blue := 0.05

-- Proof problem statement
theorem total_cost_after_discounts :
  let total_balls_red := packs_red * balls_per_pack_red
  let total_balls_yellow := packs_yellow * balls_per_pack_yellow
  let total_balls_blue := packs_blue * balls_per_pack_blue

  let cost_red := total_balls_red * price_per_ball_red
  let cost_yellow := total_balls_yellow * price_per_ball_yellow
  let cost_blue := total_balls_blue * price_per_ball_blue

  let final_cost_red := cost_red * (1 - discount_red)
  let final_cost_blue := cost_blue * (1 - discount_blue)

  final_cost_red = 121.50 ∧
  cost_yellow = 75.00 ∧
  final_cost_blue = 34.20 :=
by
  sorry

end total_cost_after_discounts_l245_245356


namespace largest_m_for_Q_divisible_by_pow2_l245_245013

theorem largest_m_for_Q_divisible_by_pow2 :
  let Q := (List.range' 1 50).map (λ n => 2 * n) in
  let product_Q := Q.foldr (· * ·) 1 in
  ∃ m : ℕ, (2^m) ∣ product_Q ∧ ∀ k : ℕ, (2^k) ∣ product_Q → k ≤ 97 :=
by
  sorry

end largest_m_for_Q_divisible_by_pow2_l245_245013


namespace largest_k_three_13_l245_245233

def largest_k (n : ℕ) : ℕ := 
  let s := 2 * 3 ^ n
  let upper_bound := 3 ^ (n / 2)
  list.max ({
    k ∈ (multiset.range (upper_bound + 1)).filter (λ k, s % k = 0 && k * (2 * ((s div k) - 1) - k + 1) = 2 * 3 ^ 13)
  }).get_or_else 0

theorem largest_k_three_13 : largest_k 13 = 1458 := sorry

end largest_k_three_13_l245_245233


namespace solve_for_y_l245_245056

noncomputable def g (y : ℝ) : ℝ := (30 * y + (30 * y + 27)^(1/3))^(1/3)

theorem solve_for_y :
  (∃ y : ℝ, g y = 15) ↔ (∃ y : ℝ, y = 1674 / 15) :=
by
  sorry

end solve_for_y_l245_245056


namespace rectangle_area_k_value_l245_245475

theorem rectangle_area_k_value (x d : ℝ) (h1 : d = Real.sqrt 13 * x) (h2 : ∃ (l w : ℝ), l = 3 * x ∧ w = 2 * x) :
  ∃ k : ℝ, (3 * x) * (2 * x) = k * d^2 ∧ k = 6 / 13 := 
by {
  -- Defined conditions for Lean to solve
  obtain ⟨l, w, hl, hw⟩ := h2,
  -- Establish the desired result based on conditions
  use (6 / 13), 
  rw [hl, hw],
  split,
  { -- Proof placeholder for area calculation
    sorry },
  { -- Proof placeholder for k evaluation
    refl }
}

end rectangle_area_k_value_l245_245475


namespace find_percentage_decrease_l245_245463

-- Define the measures of two complementary angles
def angles_complementary (a b : ℝ) : Prop := a + b = 90

-- Given variables
variable (small_angle large_angle : ℝ)

-- Given conditions
def ratio_of_angles (small_angle large_angle : ℝ) : Prop := small_angle / large_angle = 1 / 2

def increased_small_angle (small_angle : ℝ) : ℝ := small_angle * 1.2

noncomputable def new_large_angle (small_angle large_angle : ℝ) : ℝ :=
  90 - increased_small_angle small_angle

def percentage_decrease (original new : ℝ) : ℝ :=
  ((original - new) / original) * 100

-- The theorem we need to prove
theorem find_percentage_decrease
  (h1 : ratio_of_angles small_angle large_angle)
  (h2 : angles_complementary small_angle large_angle) :
  percentage_decrease large_angle (new_large_angle small_angle large_angle) = 10 :=
sorry

end find_percentage_decrease_l245_245463


namespace water_level_after_ice_melts_l245_245247

-- Definitions for water density and ice density
def ρвода : ℝ := 1000  -- density of water in kg/m^3
def ρльда : ℝ := 917   -- density of ice in kg/m^3

-- Conditions for conservation of mass and volume displacement
theorem water_level_after_ice_melts (V : ℝ) (W : ℝ) (ρвода ρльда: ℝ) 
  (Hρвода : ρвода = 1000) (Hρльда : ρльда = 917)
  (Hmass : V * ρвода = W * ρльда) 
  (Hvol : V * ρвода = W * ρльда) : 
  V = W := 
sorry

end water_level_after_ice_melts_l245_245247


namespace croissants_left_l245_245026

theorem croissants_left (total_croissants : ℕ) (neighbors : ℕ)
    (h1 : total_croissants = 59) (h2 : neighbors = 8) : (59 % 8) = 3 :=
by { rw [h1, h2], norm_num }

end croissants_left_l245_245026


namespace lisa_total_spoons_l245_245846

def number_of_baby_spoons (num_children num_spoons_per_child : Nat) : Nat :=
  num_children * num_spoons_per_child

def number_of_decorative_spoons : Nat := 2

def number_of_old_spoons (baby_spoons decorative_spoons : Nat) : Nat :=
  baby_spoons + decorative_spoons
  
def number_of_new_spoons (large_spoons teaspoons : Nat) : Nat :=
  large_spoons + teaspoons

def total_number_of_spoons (old_spoons new_spoons : Nat) : Nat :=
  old_spoons + new_spoons

theorem lisa_total_spoons
  (children : Nat)
  (spoons_per_child : Nat)
  (large_spoons : Nat)
  (teaspoons : Nat)
  (children_eq : children = 4)
  (spoons_per_child_eq : spoons_per_child = 3)
  (large_spoons_eq : large_spoons = 10)
  (teaspoons_eq : teaspoons = 15)
  : total_number_of_spoons (number_of_old_spoons (number_of_baby_spoons children spoons_per_child) number_of_decorative_spoons) (number_of_new_spoons large_spoons teaspoons) = 39 :=
by
  sorry

end lisa_total_spoons_l245_245846


namespace clayton_first_game_points_l245_245198

theorem clayton_first_game_points :
  ∃ (P : ℕ), 
    let P₂ := 14, 
        P₃ := 6, 
        P₄ := (P + P₂ + P₃) / 3 in
    P + P₂ + P₃ + P₄ = 40 ∧ P = 10 :=
by
  exists 10
  let P := 10
  let P₂ := 14
  let P₃ := 6
  let P₄ := (P + P₂ + P₃) / 3
  have h1 : P + P₂ + P₃ + P₄ = 40 := by
    calc
      P + P₂ + P₃ + P₄
      = 10 + 14 + 6 + (10 + 14 + 6) / 3       : by rfl
      ... = 10 + 14 + 6 + 10                  : by simp [P₄, P₂, P₃]
      ... = 40                                : by norm_num
  have h2 : P = 10 := by rfl
  exact ⟨h1, h2⟩

end clayton_first_game_points_l245_245198


namespace josh_paths_to_top_center_l245_245369

/-- Define the grid --/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Josh's initial position --/
def start_pos : (ℕ × ℕ) := (0, 0)

/-- Define a function representing Josh's movement possibilities --/
def move_right (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1, pos.2 + 1)

def move_left_up (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1 + 1, pos.2 - 1)

/-- Define the goal position --/
def goal_pos (n : ℕ) : (ℕ × ℕ) :=
  (n - 1, 1)

/-- Theorem stating the required proof --/
theorem josh_paths_to_top_center {n : ℕ} (h : n > 0) : 
  let g := Grid.mk n 3 in
  ∃ (paths : ℕ), paths = 2^(n - 1) := 
  sorry

end josh_paths_to_top_center_l245_245369


namespace volume_of_region_within_larger_sphere_not_within_smaller_l245_245924

theorem volume_of_region_within_larger_sphere_not_within_smaller 
  (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 7) : 
  let V_small := (4 / 3) * π * r_small^3,
      V_large := (4 / 3) * π * r_large^3 in
  V_large - V_small = 372 * π :=
by
  sorry

end volume_of_region_within_larger_sphere_not_within_smaller_l245_245924


namespace cannot_form_right_triangle_l245_245178

theorem cannot_form_right_triangle (a b c : Nat) (h₁ : (a, b, c) = (5, 12, 13) → a^2 + b^2 = c^2)
                                   (h₂ : (a, b, c) = (6, 8, 10) → a^2 + b^2 = c^2)
                                   (h₃ : (a, b, c) = (7, 24, 25) → a^2 + b^2 = c^2)
                                   (h₄ : (a, b, c) = (4, 6, 8)) : a^2 + b^2 ≠ c^2 :=
by {
  rw [h₄];
  have : ¬(4^2 + 6^2 = 8^2), by sorry;
  exact this
}

end cannot_form_right_triangle_l245_245178


namespace circle_passing_origin_l245_245158

theorem circle_passing_origin (a b r : ℝ) :
  ((a^2 + b^2 = r^2) ↔ (∃ (x y : ℝ), (x-a)^2 + (y-b)^2 = r^2 ∧ x = 0 ∧ y = 0)) :=
by
  sorry

end circle_passing_origin_l245_245158


namespace solution_range_of_a_l245_245242

noncomputable def satisfiesEquation (x a : ℝ) : Prop :=
  3^x = a^2 + 2*a

theorem solution_range_of_a (a : ℝ) :
  (∃ x, x ∈ Iic (1 : ℝ) ∧ satisfiesEquation x a) ↔ a ∈ set.Icc (-3 : ℝ) (-2) ∪ set.Ioc (0 : ℝ) (1) :=
by sorry

end solution_range_of_a_l245_245242


namespace problem_statement_l245_245815

variables {A B O C D P Q : Point}

-- Definitions based on the problem conditions
def semicircle_diameter (A B O : Point) : Prop := 
  -- O is the midpoint of AB
  O = midpoint A B ∧ 
  -- AB is the diameter of the semicircle
  (A ≠ B) ∧ 
  -- C and D are on the arc AB
  on_arc A B C ∧ on_arc A B D

def circumcenter_OAC (O A C P : Point) : Prop :=
  -- P is the circumcenter of triangle OAC
  circumcenter O A C P 

def circumcenter_OBD (O B D Q : Point) : Prop :=
  -- Q is the circumcenter of triangle OBD
  circumcenter O B D Q

-- The theorem statement
theorem problem_statement (h₁ : semicircle_diameter A B O)
                          (h₂ : circumcenter_OAC O A C P)
                          (h₃ : circumcenter_OBD O B D Q) :
  CP * CQ = DP * DQ :=
sorry

end problem_statement_l245_245815


namespace problem_statement_l245_245283

noncomputable def necessary_but_not_sufficient_condition (x y : ℝ) (hx : x > 0) : Prop :=
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|)

theorem problem_statement
  (x y : ℝ)
  (hx : x > 0)
  : necessary_but_not_sufficient_condition x y hx :=
sorry

end problem_statement_l245_245283


namespace midpoint_I_A_is_midpoint_KaH_l245_245051

/-- Prove that the midpoint \(I_{A}\) of \(BC\) is also the midpoint of the segment \([K_{A}H]\) under given conditions. -/
theorem midpoint_I_A_is_midpoint_KaH
  (A H O I_A K_A G : Point)
  (hG : Homothety(G, -1/2))
  (hKa : Homothety(K_A, 2))
  (hG_maps_A_to_I_A : hG.map A = I_A)
  (hG_maps_H_to_O : hG.map H = O)
  (vec_eq_HA : Vector(H, A) = -2 * Vector(O, I_A))
  (hKa_maps_O_to_A : hKa.map O = A)
  (hKa_maps_I_A : ∃ X : Point, hKa.map I_A = X ∧ Vector(A, X) = 2 * Vector(O, I_A))
  (vec_eq_AX_AH : ∀ X, Vector(A, X) = Vector(A, H) -> Point.eq X H) :
  is_midpoint I_A K_A H :=
sorry

end midpoint_I_A_is_midpoint_KaH_l245_245051


namespace intersecting_line_eq_l245_245895

theorem intersecting_line_eq (x y : ℝ) (λ : ℝ)
  (h1 : 2 * x - y + 4 = 0)
  (h2 : x - y + 5 = 0)
  (h3 : (2 + λ) * x - (1 + λ) * y + 4 + 5 * λ = 0)
  (perpendicular : 2 + λ = -2 * (1 + λ)) :
  2 * x + y - 8 = 0 :=
by
  sorry

end intersecting_line_eq_l245_245895


namespace no_integer_solutions_l245_245455

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^1988 + y^1988 + z^1988 = 7^1990 := by
  sorry

end no_integer_solutions_l245_245455


namespace max_value_fraction_ratio_tangent_line_through_point_l245_245272

theorem max_value_fraction_ratio (x y : ℝ) (h : x^2 + y^2 - 2*x - 2 = 0) :
  ∃ (max_value : ℝ), max_value = 2 + sqrt 6 :=
sorry

theorem tangent_line_through_point (x y : ℝ) (h : x^2 + y^2 - 2*x - 2 = 0) :
  ∃ (m : ℝ), m ≠ 0 ∧ (x, y) = (0, sqrt 2) → ∀ (x' y' : ℝ), y' = m * x' + sqrt 2 → x' - sqrt 2 * y' + 2 = 0 :=
sorry

end max_value_fraction_ratio_tangent_line_through_point_l245_245272


namespace length_of_train_l245_245170

-- Definitions of given conditions
def time_to_cross_pole : ℝ := 4.499640028797696
def speed_km_per_hour : ℝ := 72
def speed_m_per_sec : ℝ := 72 * (1000 / 3600)

-- The theorem to prove
theorem length_of_train :
  (speed_m_per_sec * time_to_cross_pole) = 89.99280057595392 :=
by
  sorry

end length_of_train_l245_245170


namespace find_f_l245_245764

-- Define the symmetry condition
def symmetric_wrt_origin (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y → g (-x) = -y

-- Define functions f and g
noncomputable def f : ℝ → ℝ := sorry  -- To be defined as the expression in the proof
noncomputable def g : ℝ → ℝ := λ x, 3 - 2 * x

theorem find_f:
  symmetric_wrt_origin f g → (∀ x, f x = -2 * x - 3) :=
by
  intro h
  -- Proof to be filled in
  sorry

end find_f_l245_245764


namespace chickens_to_goats_ratio_l245_245485

variables {Goats Chickens Ducks Pigs : ℕ}

noncomputable def goats := 66
noncomputable def ducks := (1 / 2) * (goats + Chickens)
noncomputable def pigs := (1 / 3) * ducks
noncomputable def goats_eq_pigs_plus_33 := goats = Pigs + 33

theorem chickens_to_goats_ratio :
  (∀ (Chickens : ℕ),
    ducks = (1 / 2) * (goats + Chickens) →
    pigs = (1 / 3) * ducks →
    goats = pigs + 33 →
    (Chickens / goats = 2 : 1)) :=
by
  sorry

end chickens_to_goats_ratio_l245_245485


namespace conic_intersection_result_l245_245688

noncomputable def conic_intersection_value : ℝ :=
  let ellipse : (ℝ × ℝ) → Prop := fun p => (p.1^2 / 25) + (p.2^2 / 16) = 1
  let hyperbola : (ℝ × ℝ) → Prop := fun p => (p.1^2 / 4) - (p.2^2 / 5) = 1
  let F₁ : ℝ × ℝ := (5, 0)
  let F₂ : ℝ × ℝ := (-5, 0)
  let P : ℝ × ℝ := (x, y)
  let PF1 := dist P F₁
  let PF2 := dist P F₂
  21

theorem conic_intersection_result :
  let ellipse : (ℝ × ℝ) → Prop := fun p => (p.1^2 / 25) + (p.2^2 / 16) = 1
  let hyperbola : (ℝ × ℝ) → Prop := fun p => (p.1^2 / 4) - (p.2^2 / 5) = 1
  let F₁ : ℝ × ℝ := (5, 0)
  let F₂ : ℝ × ℝ := (-5, 0)
  let P : ℝ × ℝ := (x, y)
  let PF1 := dist P F₁
  let PF2 := dist P F₂
  21 :=
sorry

end conic_intersection_result_l245_245688


namespace four_digit_numbers_count_even_numbers_greater_than_3000_count_l245_245102

open Nat

def digits : List ℕ := [0, 1, 2, 3, 4]
def even_digits : List ℕ := [0, 2, 4]

theorem four_digit_numbers_count : 
  (∃ count, count = digits.erase 0 |>.permutations.length * 4 = 96) := 
sorry

theorem even_numbers_greater_than_3000_count :
  (∃ count, (count = 84) ∧ (∀ n, n ∈ mk_five_digit_numbers digits even_digits → n ≥ 3000)) :=
sorry

-- Helper definitions for completeness, they should be properly defined for the complete proof 
def mk_five_digit_numbers (digs even_digs : List ℕ) : List ℕ := -- construct the five digit numbers
sorry

def mk_four_digit_numbers (digs even_digs : List ℕ) : List ℕ := -- construct the four digit numbers
sorry

end four_digit_numbers_count_even_numbers_greater_than_3000_count_l245_245102


namespace point_B_value_l245_245415

theorem point_B_value :
  ∃ B : ℝ, (|B + 1| = 4) ∧ (B = 3 ∨ B = -5) := 
by
  sorry

end point_B_value_l245_245415


namespace mary_mortgage_payment_l245_245027

theorem mary_mortgage_payment :
  let a1 := 400
  let r := 2
  let n := 11
  let sum_geom_series (a1 r : ℕ) (n : ℕ) : ℕ := (a1 * (1 - r^n)) / (1 - r)
  sum_geom_series a1 r n = 819400 :=
by
  let a1 := 400
  let r := 2
  let n := 11
  let sum_geom_series (a1 r : ℕ) (n : ℕ) : ℕ := (a1 * (1 - r^n)) / (1 - r)
  have h : sum_geom_series a1 r n = 819400 := sorry
  exact h

end mary_mortgage_payment_l245_245027


namespace train_pass_time_approx_l245_245168

-- Define the constants given in the problem
def train_length : ℝ := 300  -- in meters
def train_speed_kmph : ℝ := 68  -- in km/hour
def man_speed_kmph : ℝ := 8  -- in km/hour

-- Convert speed from km/h to m/s
def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

-- Calculate the relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (train_speed_kmph - man_speed_kmph)

-- Calculate the time to pass the man using the formula: Time = Distance / Speed
def time_to_pass_man : ℝ := train_length / relative_speed_mps

-- The theorem we aim to prove
theorem train_pass_time_approx : abs (time_to_pass_man - 18) < 0.1 := by
  sorry

end train_pass_time_approx_l245_245168


namespace unusual_digits_exists_l245_245559

def is_unusual (n : ℕ) : Prop :=
  let len := n.digits.count;
  let high_power := 10 ^ len;
  (n^3 % high_power = n) ∧ (n^2 % high_power ≠ n)

theorem unusual_digits_exists :
  ∃ n1 n2 : ℕ, (n1 ≥ 10^99 ∧ n1 < 10^100 ∧ is_unusual n1) ∧ 
             (n2 ≥ 10^99 ∧ n2 < 10^100 ∧ is_unusual n2) ∧
             (n1 ≠ n2) :=
by
  let n1 := 10^100 - 1;
  let n2 := (10^100 / 2) - 1;
  use n1, n2;
  sorry

end unusual_digits_exists_l245_245559


namespace radian_measure_of_neg_300_degrees_l245_245515

theorem radian_measure_of_neg_300_degrees : (-300 : ℝ) * (Real.pi / 180) = -5 * Real.pi / 3 :=
by
  sorry

end radian_measure_of_neg_300_degrees_l245_245515


namespace find_a_l245_245728

theorem find_a (a : ℝ) (h : ∃ (x y : ℝ), x = 1 ∧ y = 2 ∧ 3 * x + a * y - 5 = 0) : a = 1 :=
by
  cases h with x hx
  cases hx with y hy
  cases hy with hx1 hy1
  cases hy1 with hy2 hy3
  simp at hy3
  have h1 : x = 1 := hx1
  have h2 : y = 2 := hy2
  subst h1
  subst h2
  simp at hy3
  linarith

end find_a_l245_245728


namespace extreme_values_max_min_values_on_interval_l245_245720

def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x + 8

theorem extreme_values :
  (f 1 = 13) ∧ (f 2 = 12) :=
by
  -- Proof of extreme values
  sorry

theorem max_min_values_on_interval :
  (∀ x ∈ set.Icc (-1 : ℝ) 3, f x ≤ 14) ∧ (∃ x ∈ set.Icc (-1 : ℝ) 3, f x = 14) ∧
  (∀ x ∈ set.Icc (-1 : ℝ) 3, -15 ≤ f x) ∧ (∃ x ∈ set.Icc (-1 : ℝ) 3, f x = -15) :=
by
  -- Proof of maximum and minimum values on [-1, 3]
  sorry

end extreme_values_max_min_values_on_interval_l245_245720


namespace complex_sum_l245_245837

-- Define the given condition as a hypothesis
variables {z : ℂ} (h : z^2 + z + 1 = 0)

-- Define the statement to prove
theorem complex_sum (h : z^2 + z + 1 = 0) : z^96 + z^97 + z^98 + z^99 + z^100 + z^101 = 0 :=
sorry

end complex_sum_l245_245837


namespace lisa_total_spoons_l245_245843

def num_children := 4
def spoons_per_child := 3
def decorative_spoons := 2
def large_spoons := 10
def teaspoons := 15

def total_spoons := num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

theorem lisa_total_spoons : total_spoons = 39 := by
  sorry

end lisa_total_spoons_l245_245843


namespace fuel_tank_oil_quantity_l245_245782

theorem fuel_tank_oil_quantity (t : ℝ) (Q : ℝ) : (Q = 40 - 0.2 * t) :=
begin
  sorry
end

end fuel_tank_oil_quantity_l245_245782


namespace part1_part2_l245_245681

noncomputable def f₀ (x : ℝ) : ℝ := 1

noncomputable def f (a : ℝ) : ℕ → (ℝ → ℝ)
| 0     := f₀
| (n+1) := λ x, x * f a n x + f a n (a * x)

theorem part1 (a : ℝ) (n : ℕ) : 
  ∀ x : ℝ, f a n x = x^n * f a n (1 / x) :=
begin
  sorry
end

theorem part2 (a : ℝ) : 
  ∀ n : ℕ, ∀ x : ℝ, 
    f a n x = 1 + ∑ j in finset.range n, 
    (finset.prod (finset.range j) (λ k, (a^(n - k) - 1)) / 
    finset.prod (finset.range j) (λ k, (a^(j - k) - 1))) * x ^ j :=
begin
  sorry
end

end part1_part2_l245_245681


namespace youngest_person_age_l245_245085

noncomputable def avg_age_seven_people := 30
noncomputable def avg_age_six_people_when_youngest_born := 25
noncomputable def num_people := 7
noncomputable def num_people_minus_one := 6

theorem youngest_person_age :
  let total_age_seven_people := num_people * avg_age_seven_people
  let total_age_six_people := num_people_minus_one * avg_age_six_people_when_youngest_born
  total_age_seven_people - total_age_six_people = 60 :=
by
  let total_age_seven_people := num_people * avg_age_seven_people
  let total_age_six_people := num_people_minus_one * avg_age_six_people_when_youngest_born
  sorry

end youngest_person_age_l245_245085


namespace distance_from_A_l245_245872

variable {R a : ℝ}

theorem distance_from_A (h1 : R > 0)
                        (h2 : ∀ x : ℝ, x > 0 → 
                           (x^2 + (2*R - x)^2 + x*(2*R - x)) / 2 = a^2)
                        (h3 : PC = x * (2*R - x)) : 
  (∃ x : ℝ, x = R + sqrt(2*a^2 - 3*R^2) ∨ x = R - sqrt(2*a^2 - 3*R^2)) ∧
  (3/2*R^2 ≤ a^2 ∧ a^2 < 2*R^2) :=
by
  sorry

end distance_from_A_l245_245872


namespace abs_neg_two_l245_245885

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end abs_neg_two_l245_245885


namespace problem_1_problem_2_l245_245304

-- Definition of conditions
def gcd (n m : ℕ) : ℕ := Nat.gcd n m
def lcm (n m : ℕ) : ℕ := Nat.lcm n m

-- Problem 1: Proving number of pairs (n, m) == 0 when n ∧ m = 50 and n ∨ m = 75.
theorem problem_1 : (Finset.card {(n, m) | gcd n m = 50 ∧ lcm n m = 75 : Finset (ℕ × ℕ)}) = 0 :=
by sorry

-- Problem 2: Proving number of pairs (n, m) == 6 when n ∧ m = 50 and n ∨ m = 600.
theorem problem_2 : (Finset.card {(n, m) | gcd n m = 50 ∧ lcm n m = 600 : Finset (ℕ × ℕ)}) = 6 :=
by sorry

end problem_1_problem_2_l245_245304


namespace summer_camp_activity_l245_245790

theorem summer_camp_activity :
  ∃ (a b c d e f : ℕ), 
  a + b + c + d + 3 * e + 4 * f = 12 ∧ 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧
  f = 1 := by
  sorry

end summer_camp_activity_l245_245790


namespace area_of_triangle_l245_245039

theorem area_of_triangle (a b : ℝ) 
  (hypotenuse : ℝ) (median : ℝ)
  (h_side : hypotenuse = 2)
  (h_median : median = 1)
  (h_sum : a + b = 1 + Real.sqrt 3) 
  (h_pythagorean :(a^2 + b^2 = 4)): 
  (1/2 * a * b) = (Real.sqrt 3 / 2) := 
sorry

end area_of_triangle_l245_245039


namespace inequality_ABC_l245_245997

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_ABC_l245_245997


namespace intersection_unique_point_l245_245005

def f (x : ℝ) : ℝ := x^3 + 6 * x^2 + 16 * x + 28

theorem intersection_unique_point :
  ∃ a : ℝ, f a = a ∧ a = -4 := sorry

end intersection_unique_point_l245_245005


namespace triangle_inequality_inequality_l245_245840

variable {a b c : ℝ}
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (triangle_ineq : a + b > c)

theorem triangle_inequality_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) (triangle_ineq : a + b > c) :
  a^3 + b^3 + 3 * a * b * c > c^3 :=
sorry

end triangle_inequality_inequality_l245_245840


namespace hyperbola_slope_l245_245566

theorem hyperbola_slope (e p : ℝ) (h_e_pos : e > 0) (h_p_pos : p > 0) :
  let C1 := { P : ℝ × ℝ | (∃ (a b : ℝ) (x y : ℝ), P = (x, y) ∧ a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1 ∧ b^2 = a^2 * (e^2 - 1)) },
      l := { l : ℝ × ℝ | (∃ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ C1 ∧ (x2, y2) ∈ C1 ∧ ∀ M : ℝ × ℝ, M = ((x1 + x2) / 2, (y1 + y2) / 2) ∨ (y1 - y2) / (x1 - x2) = l) } in
  ∀ A B : ℝ × ℝ, A ∈ C1 ∧ B ∈ C1 ∧ let M := ((A.1 + B.1)/2, (A.2 + B.2)/2) in (M.1 = p/2 ∧ M.2 = p ∧ sqrt((M.1 - 0)^2 + (M.2 - p)^2) = p)
  → (∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ slope A B = (e^2 - 1) / 2) :=
sorry

end hyperbola_slope_l245_245566


namespace average_speed_l245_245531

-- Define the distances and speeds for the two parts of the trip.
def distance1 : ℝ := 180
def speed1 : ℝ := 60
def distance2 : ℝ := 120
def speed2 : ℝ := 40

-- Calculate the total distance.
def total_distance : ℝ := distance1 + distance2

-- Calculate the time for each part of the trip.
def time1 : ℝ := distance1 / speed1
def time2 : ℝ := distance2 / speed2

-- Calculate the total time.
def total_time : ℝ := time1 + time2

-- Prove that the average speed for the entire trip is 50 miles per hour.
theorem average_speed : (total_distance / total_time) = 50 := by
  sorry

end average_speed_l245_245531


namespace problem1_problem2_problem3_l245_245732

-- Problem 1
theorem problem1 (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d) (h_a1 : a 1 = 15) (h_d_ne_zero : d ≠ 0)
  (h_geo : a 7 ^ 2 = a 4 * a 8) : d = -2 :=
by
  sorry

-- Problem 2
theorem problem2 (b : ℕ → ℝ) (n : ℕ) (h_b : ∀ n, b n = |(15 + 2 * (n - 1) - 21)|) :
  let S_n := ∑ i in Finset.range n, b i
  in if n ≤ 4 then S_n = 7n - n^2 else S_n = n^2 - 7n + 24 :=
by
  sorry

-- Problem 3
theorem problem3 (b : ℕ → ℝ) (n : ℕ) (h_b : ∀ n, b n = Real.exp (15 - (n - 1) - 1)) :
  let T_n := ∏ i in Finset.range n, b i
  in T_n is maximized when n = 15 ∨ n = 16 :=
by
  sorry

end problem1_problem2_problem3_l245_245732


namespace solution_l245_245628

noncomputable def problem : Prop := 
  - (Real.sin (133 * Real.pi / 180)) * (Real.cos (197 * Real.pi / 180)) -
  (Real.cos (47 * Real.pi / 180)) * (Real.cos (73 * Real.pi / 180)) = 1 / 2

theorem solution : problem :=
by
  sorry

end solution_l245_245628


namespace coeff_x60_expanded_polynomial_l245_245210

theorem coeff_x60_expanded_polynomial :
  let P := (∏ i in finset.range 1 13, (X^i - i : Polynomial ℤ))
  ∑ i, coeff (finset.range P) (monomial 60 1) = 156 :=
by sorry

end coeff_x60_expanded_polynomial_l245_245210


namespace circumcenter_on_median_l245_245400

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def altitude (A B C : Point) (p : Point) : Line := sorry
noncomputable def median (A B C : Point) (A' : Point) : Line := sorry
noncomputable def intersection (O : Point) (l m : Line) : Point := sorry

theorem circumcenter_on_median 
  {A B C O H P Q G : Point}
  (hAcute : acute_angled_triangle A B C)
  (hO : circumcenter A B C = O)
  (hH : orthocenter A B C = H)
  (hP : intersection O (altitude B A C) (line_through O A) = P)
  (hQ : intersection O (altitude C A B) (line_through O A) = Q)
  (hG : circumcenter P Q H = G) :
  G ∈ median A B C (midpoint B C) :=
sorry

end circumcenter_on_median_l245_245400


namespace Rahul_can_complete_work_alone_in_3_days_l245_245043

variable (R : ℝ) -- R represents the number of days Rahul takes to complete the work alone
variable (total_money : ℝ) -- total money they receive together
variable (rahul_share : ℝ) -- Rahul's share of the money
variable (rajesh_days : ℝ) -- number of days Rajesh takes to complete the work
variable (rajesh_money : ℝ) -- Rajesh's share of the money

-- The conditions
axiom H1 (total_money_eq : total_money = 105)
axiom H2 (rahul_share_eq : rahul_share = 42)
axiom H3 (rajesh_days_eq : rajesh_days = 2)
axiom H4 (money_sharing : total_money - rahul_share = rajesh_money)

-- The ratio of their earnings should be equal to the ratio of their work rates
axiom H5 (earnings_ratio : rahul_share / rajesh_money = (1 / R) / (1 / rajesh_days))

theorem Rahul_can_complete_work_alone_in_3_days : R = 3 :=
by
  sorry

end Rahul_can_complete_work_alone_in_3_days_l245_245043


namespace functional_equation_unique_solutions_l245_245647

theorem functional_equation_unique_solutions (f : ℝ⁺ → ℝ⁺)
    (cond : ∀ x y : ℝ⁺, f(x^y) = f(x)^(f(y))) :
    (∀ x : ℝ⁺, f(x) = 1) ∨ (∀ x : ℝ⁺, f(x) = x) :=
by
  sorry

end functional_equation_unique_solutions_l245_245647


namespace range_of_a_for_inequality_l245_245673

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ p q : ℝ, (0 < p ∧ p < 1) → (0 < q ∧ q < 1) → p ≠ q → (f a p - f a q) / (p - q) > 1) ↔ 3 ≤ a :=
sorry

end range_of_a_for_inequality_l245_245673


namespace principal_sum_l245_245134

/-- Given compound interest and simple interest for two years and 
    the time period of two years, prove the principal sum -/
theorem principal_sum (CI SI : ℝ) (T : ℝ) (hCI : CI = 11730) (hSI : SI = 10200) (hT : T = 2) :
  ∃ P : ℝ, P = 17000 :=
by
  use 17000
  -- Remaining proof can be filled here
  sorry

end principal_sum_l245_245134


namespace proof_statements_l245_245122

theorem proof_statements :
  (∀ (P : Type) (a b c d e f : P) (A1 A2 : P → Prop)
   (parallelogram : P → P → P → P → Prop)
   (median : P → P → P → P → Prop)
   (central_angle : P → P → P → ℝ)
   (on_circle : P → P → Prop)
   (regular_polygon : P → Prop)
   (even_side_reg_poly : P → Prop),
   (∃ line : P,
     (parallelogram a b c d → 
       (A1 line → A2 line)) ∧
     (A1 line = (∃ e f, (e ≠ f ∧ (parallelogram a b c d → central_angle e f O = 70 ∧ 
      (on_circle c f) → false ∧ (on_circle c f) → true ∧ (on_circle c f) → A2 line)))) ∧
     (A1 line = (∃ g h, (regular_polygon g → even_side_reg_poly g → false ∧
      (median a b c d = A2 line))) ∧
      (A1 line → ∃ h, (median a b c d → 
        (half_median_equals_this_side a b = A2 line))))) :=
  sorry

end proof_statements_l245_245122


namespace Number_of_Values_A_inter_B_inter_C_l245_245426

theorem Number_of_Values_A_inter_B_inter_C (A B C : Finset ℕ) :
  |A| = 92 →
  |B| = 35 →
  |C| = 63 →
  |A ∩ B| = 16 →
  |A ∩ C| = 51 →
  |B ∩ C| = 19 →
  (∃ n : ℕ, 7 ≤ n ∧ n ≤ 16) → 
  (∃ num_values : ℕ, num_values = 10) :=
by
  intros hA hB hC hAB hAC hBC _,
  use 10,
  sorry

end Number_of_Values_A_inter_B_inter_C_l245_245426


namespace twenty_second_entry_l245_245240

def r_9 (n : ℕ) : ℕ := n % 9

noncomputable def sequence_of_interest : List ℕ :=
List.filter (λ n, r_9 (7 * n) ≤ 4) (List.range 1000)

theorem twenty_second_entry :
  sequence_of_interest.nth 21 = some 39 :=
begin
  sorry
end

end twenty_second_entry_l245_245240


namespace paperclips_exceed_200_at_friday_l245_245856

def paperclips_on_day (n : ℕ) : ℕ :=
  3 * 4^n

theorem paperclips_exceed_200_at_friday : 
  ∃ n : ℕ, n = 4 ∧ paperclips_on_day n > 200 :=
by
  sorry

end paperclips_exceed_200_at_friday_l245_245856


namespace inequality_hold_l245_245969

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end inequality_hold_l245_245969


namespace shelves_in_closet_l245_245351

theorem shelves_in_closet (cans_per_row : ℕ) (rows_per_shelf : ℕ) (cans_per_closet : ℕ) : 
  cans_per_row = 12 → rows_per_shelf = 4 → cans_per_closet = 480 → 
  cans_per_closet / (cans_per_row * rows_per_shelf) = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end shelves_in_closet_l245_245351


namespace smallest_positive_difference_l245_245502

-- Define the method used by Vovochka to sum numbers.
def sum_without_carrying (a b : ℕ) : ℕ :=
  let ha := a / 100 
  let ta := (a % 100) / 10 
  let ua := a % 10
  let hb := b / 100 
  let tb := (b % 100) / 10 
  let ub := b % 10
  (ha + hb) * 1000 + (ta + tb) * 100 + (ua + ub)

-- Define the correct method to sum numbers.
def correct_sum (a b : ℕ) : ℕ :=
  a + b

-- The smallest positive difference between the sum without carrying and the correct sum.
theorem smallest_positive_difference (a b : ℕ) (h : 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000) :
  (sum_without_carrying a b - correct_sum a b > 0) →
  (∀ c d, 100 ≤ c ∧ c < 1000 ∧ 100 ≤ d ∧ d < 1000 →
    (sum_without_carrying c d - correct_sum c d > 0 → sum_without_carrying c d - correct_sum c d ≥ 1800 )) :=
begin
  sorry
end

end smallest_positive_difference_l245_245502


namespace sequence_property_l245_245685

noncomputable def U : ℕ → ℕ
| 0       => 0  -- This definition is added to ensure U 1 corresponds to U_1 = 1
| (n + 1) => U n + (n + 1)

theorem sequence_property (n : ℕ) : U n + U (n + 1) = (n + 1) * (n + 1) :=
  sorry

end sequence_property_l245_245685


namespace larger_number_l245_245899

noncomputable def hcf (a b : ℕ) := Nat.gcd a b
noncomputable def lcm (a b : ℕ) := (a * b) / (hcf a b)

theorem larger_number
  (A B : ℕ)
  (h : hcf A B = 60)
  (h1 : ∃ (m n : ℕ), lcm A B = 60 * 11 * 15 ∧ A = 60 * m ∧ B = 60 * n ∧ Nat.coprime m n ∧ m * n = 11 * 15) :
  max A B = 900 :=
by
  sorry -- proof omitted

end larger_number_l245_245899


namespace mia_days_not_worked_l245_245859

theorem mia_days_not_worked :
  ∃ (y : ℤ), (∃ (x : ℤ), 
  x + y = 30 ∧ 80 * x - 40 * y = 1600) ∧ y = 20 :=
by
  sorry

end mia_days_not_worked_l245_245859


namespace total_pastries_l245_245194

-- Defining the initial conditions
def Grace_pastries : ℕ := 30
def Calvin_pastries : ℕ := Grace_pastries - 5
def Phoebe_pastries : ℕ := Grace_pastries - 5
def Frank_pastries : ℕ := Calvin_pastries - 8

-- The theorem we want to prove
theorem total_pastries : 
  Calvin_pastries + Phoebe_pastries + Frank_pastries + Grace_pastries = 97 := by
  sorry

end total_pastries_l245_245194


namespace find_exponent_of_five_in_8_factorial_l245_245328

theorem find_exponent_of_five_in_8_factorial (i k p : ℕ) (h1 : 8! = 2^i * 3^k * 5^1 * 7^p) (h2 : i + k + 1 + p = 11) : 1 = 1 :=
by 
-- The proof is omitted as we are only required to state the theorem.
sorry

end find_exponent_of_five_in_8_factorial_l245_245328


namespace maximize_profit_l245_245606

def p (t : ℕ) : ℝ :=
  if t ≥ 1 ∧ t ≤ 5 then 10 + 2 * t else
  if t > 5 ∧ t ≤ 10 then 20 else
  if t > 10 ∧ t ≤ 16 then 40 - 2 * t else 0

def q (t : ℕ) : ℝ :=
  - (1 / 8) * (t - 8) ^ 2 + 12

def profit (t : ℕ) : ℝ :=
  p t - q t

theorem maximize_profit :
  ∃ t_max ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      profit t_max = 9.125 ∧ 
      ∀ t ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, profit t ≤ 9.125 :=
sorry

end maximize_profit_l245_245606


namespace quadratic_equal_roots_iff_l245_245243

theorem quadratic_equal_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 - k * x + 9 = 0 ∧ x^2 - k * x + 9 = 0 ∧ x = x) ↔ k^2 = 36 :=
by
  sorry

end quadratic_equal_roots_iff_l245_245243


namespace find_a_l245_245709

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h : f = fun x => a * x ^ 3 - 3 * x) (h1 : f (-1) = 4) : a = -1 :=
by
  sorry

end find_a_l245_245709


namespace sum_of_two_numbers_l245_245893

theorem sum_of_two_numbers :
  ∀ (A B : ℚ), (A - B = 8) → (1 / 4 * (A + B) = 6) → (A = 16) → (A + B = 24) :=
by
  intros A B h1 h2 h3
  sorry

end sum_of_two_numbers_l245_245893


namespace nat_with_1234_divisors_is_not_perfect_square_l245_245197

theorem nat_with_1234_divisors_is_not_perfect_square (n : ℕ) (h : Nat.divisors n = 1234) 
    (h1 : ∀ m : ℕ, n = m^2 → false) : n \not\perfect_square :=
by
  sorry

end nat_with_1234_divisors_is_not_perfect_square_l245_245197


namespace sabrina_total_leaves_l245_245046

-- Definitions based on conditions
def basil_leaves := 12
def twice_the_sage_leaves (sages : ℕ) := 2 * sages = basil_leaves
def five_fewer_than_verbena (sages verbenas : ℕ) := sages + 5 = verbenas

-- Statement to prove
theorem sabrina_total_leaves (sages verbenas : ℕ) 
    (h1 : twice_the_sage_leaves sages) 
    (h2 : five_fewer_than_verbena sages verbenas) :
    basil_leaves + sages + verbenas = 29 :=
sorry

end sabrina_total_leaves_l245_245046


namespace pieces_of_candy_l245_245086

def total_items : ℝ := 3554
def secret_eggs : ℝ := 145.0

theorem pieces_of_candy : (total_items - secret_eggs) = 3409 :=
by 
  sorry

end pieces_of_candy_l245_245086


namespace initial_books_l245_245808

theorem initial_books (B : ℕ) (h : B + 5 = 7) : B = 2 :=
by sorry

end initial_books_l245_245808


namespace collinear_bisectors_l245_245554

theorem collinear_bisectors
  (A B C D P Q M_AC M_BD M_PQ : Type*)
  [convex_quadrilateral A B C D]
  [intersects_opposite_sides A B C D P Q]
  [external_angle_bisectors A B C D P Q l_A l_B l_C l_D l_P l_Q]
  (l_A l_C : Type*) (l_B l_D : Type*) (l_P l_Q : Type*)
  [intersection_point l_A l_C M_AC]
  [intersection_point l_B l_D M_BD]
  [intersection_point l_P l_Q M_PQ]
  : collinear M_AC M_BD M_PQ :=
sorry

end collinear_bisectors_l245_245554


namespace prove_a_eq_1_l245_245838

variables {a b c d k m : ℕ}
variables (h_odd_a : a%2 = 1) 
          (h_odd_b : b%2 = 1) 
          (h_odd_c : c%2 = 1) 
          (h_odd_d : d%2 = 1)
          (h_a_pos : 0 < a) 
          (h_ineq1 : a < b) 
          (h_ineq2 : b < c) 
          (h_ineq3 : c < d)
          (h_eqn1 : a * d = b * c)
          (h_eqn2 : a + d = 2^k) 
          (h_eqn3 : b + c = 2^m)

theorem prove_a_eq_1 
  (h_odd_a : a%2 = 1) 
  (h_odd_b : b%2 = 1) 
  (h_odd_c : c%2 = 1) 
  (h_odd_d : d%2 = 1)
  (h_a_pos : 0 < a) 
  (h_ineq1 : a < b) 
  (h_ineq2 : b < c) 
  (h_ineq3 : c < d)
  (h_eqn1 : a * d = b * c)
  (h_eqn2 : a + d = 2^k) 
  (h_eqn3 : b + c = 2^m) :
  a = 1 := by
  sorry

end prove_a_eq_1_l245_245838


namespace number_of_divisors_20_l245_245075

def prime_factorization_20 : List (ℕ × ℕ) := [(2, 2), (5, 1)]

theorem number_of_divisors_20 : number_of_divisors 20 = 6 :=
by
  -- Formal proof that the number of positive divisors of 20 is 6 given the prime factorization
  sorry

end number_of_divisors_20_l245_245075


namespace chicken_pot_pie_pieces_l245_245626

theorem chicken_pot_pie_pieces (customers_shepherd: ℕ) (shepherds_pie_pieces: ℕ) 
  (customers_chicken: ℕ) (total_pies: ℕ) (sold_shepherd_pies: ℕ) 
  (sold_chicken_pies: ℕ) (pieces_per_pie: ℕ) :
  customers_shepherd = 52 → 
  shepherds_pie_pieces = 4 → 
  customers_chicken = 80 → 
  total_pies = 29 → 
  sold_shepherd_pies = customers_shepherd / shepherds_pie_pieces → 
  sold_chicken_pies = total_pies - sold_shepherd_pies → 
  pieces_per_pie = customers_chicken / sold_chicken_pies → 
  pieces_per_pie = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  sorry

end chicken_pot_pie_pieces_l245_245626


namespace domain_of_function_l245_245894

def dom_f := {x : ℝ | x > 0}

theorem domain_of_function : ∀ x : ℝ, f x = x / sqrt (exp x - 1) → x ∈ dom_f :=
by
  intros x hx
  simp [f] at hx
  sorry

end domain_of_function_l245_245894


namespace binary_to_octal_correct_l245_245631

def binary_to_decimal (b : ℕ) : ℕ := sorry -- Define conversion from binary to decimal
def decimal_to_octal (d : ℕ) : ℕ := sorry -- Define conversion from decimal to octal

theorem binary_to_octal_correct : 
  (binary_to_decimal 0b1010101 = 85) → 
  (decimal_to_octal 85 = 125) → 
  decimal_to_octal (binary_to_decimal 0b1010101) = 125 :=
begin
  intros h1 h2,
  rw h1,
  rw h2,
end

end binary_to_octal_correct_l245_245631


namespace cos_2alpha_eq_63_over_65_l245_245251

theorem cos_2alpha_eq_63_over_65 
  (α β : ℝ) 
  (h_alpha_beta_range : α ∈ Ioo 0 (π / 2) ∧ β ∈ Ioo 0 (π / 2))
  (hcos_sum : Real.cos (α + β) = 5 / 13)
  (hsin_diff : Real.sin (α - β) = -4 / 5) : 
  Real.cos (2 * α) = 63 / 65 :=
sorry

end cos_2alpha_eq_63_over_65_l245_245251


namespace josh_paths_to_center_square_l245_245376

-- Definition of the problem's conditions based on given movements and grid size
def num_paths (n : Nat) : Nat :=
  2^(n-1)

-- Main statement
theorem josh_paths_to_center_square (n : Nat) : ∃ p : Nat, p = num_paths n :=
by
  exists num_paths n
  sorry

end josh_paths_to_center_square_l245_245376


namespace inequality_ABC_l245_245999

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_ABC_l245_245999


namespace price_of_turban_l245_245130

variable (T : ℝ)
variable (rs90 : ℝ := 90)
variable (rs50 : ℝ := 50)
variable (proratedSalary9Months : ℝ := 0.75 * rs90)

theorem price_of_turban : rs50 + T = proratedSalary9Months → T = 17.5 := by
  intro h
  calc
    T = proratedSalary9Months - rs50 := by linarith
    _ = 17.5 := by norm_num

end price_of_turban_l245_245130


namespace polynomial_simplification_l245_245429

variable (x : ℝ)

theorem polynomial_simplification : 
  ((3 * x - 2) * (5 * x ^ 12 + 3 * x ^ 11 - 4 * x ^ 9 + x ^ 8)) = 
  (15 * x ^ 13 - x ^ 12 - 6 * x ^ 11 - 12 * x ^ 10 + 11 * x ^ 9 - 2 * x ^ 8) := by
  sorry

end polynomial_simplification_l245_245429


namespace solve_for_n_l245_245761

theorem solve_for_n (n : ℕ) (h : 2 * n - 5 = 1) : n = 3 :=
by
  sorry

end solve_for_n_l245_245761


namespace Lisa_total_spoons_l245_245850

def total_spoons (children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) (large_spoons : ℕ) (teaspoons : ℕ) : ℕ := 
  (children * spoons_per_child) + decorative_spoons + (large_spoons + teaspoons)

theorem Lisa_total_spoons :
  (total_spoons 4 3 2 10 15) = 39 :=
by
  sorry

end Lisa_total_spoons_l245_245850


namespace proj_not_square_area_one_proj_square_area_reciprocal_2019_l245_245456

theorem proj_not_square_area_one (T : Tetrahedron) :
  (∃ P : Plane, orthogonal_projection T P = Trapezoid 1) →
  ¬ ∃ P' : Plane, orthogonal_projection T P' = Square 1 :=
by sorry

theorem proj_square_area_reciprocal_2019 (T : Tetrahedron) :
  (∃ P : Plane, orthogonal_projection T P = Trapezoid 1) →
  ∃ P' : Plane, orthogonal_projection T P' = Square (1 / 2019) :=
by sorry

end proj_not_square_area_one_proj_square_area_reciprocal_2019_l245_245456


namespace graph_condition_l245_245320

/-- Given conditions -/
variables (f : ℝ → ℝ)
variables (hinv : ∃ g : ℝ → ℝ, function.left_inverse f g ∧ function.right_inverse f g)
variables (hg : ∀ x, x ≠ 2 → tan (π * x / 6) - f x ≠ sqrt 3 - 1/3)
variables (h2 : tan (π * 2 / 6) - f 2 = sqrt 3 - 1/3)

/-- Required proof -/
theorem graph_condition :
  let g := classical.some hinv in
  function.left_inverse f g ∧ function.right_inverse f g →
  ∃ x y, (y = g x - π / 2) ∧ (x = 1/3) ∧ (y = 2 - π / 2) :=
by
  sorry

end graph_condition_l245_245320


namespace part1_part2_l245_245483

-- Define the condition
def summerCamp (n : ℕ) (is_positive : n > 0) : Prop :=
  ∃ arrangement : List (List ℕ),
    arrangement.length = (3 * n * (3 * n - 1)) / 6 ∧
    ∀ (i j : ℕ), i ≠ j → (∃! day ∈ arrangement, {i, j} ⊆ set.day)

-- Part 1
theorem part1 (n : ℕ) (h : n = 3) : ∃ arrangement, summerCamp n (sorry) :=
  sorry

-- Part 2
theorem part2 (n : ℕ) (is_positive : n > 0) (h : summerCamp n is_positive) : n % 2 = 1 :=
  sorry

end part1_part2_l245_245483


namespace af_b_lt_bf_a_l245_245286

variable {f : ℝ → ℝ}
variable {a b : ℝ}

theorem af_b_lt_bf_a (h1 : ∀ x y, 0 < x → 0 < y → x < y → f x > f y)
                    (h2 : ∀ x, 0 < x → f x > 0)
                    (h3 : 0 < a)
                    (h4 : 0 < b)
                    (h5 : a < b) :
  a * f b < b * f a :=
sorry

end af_b_lt_bf_a_l245_245286


namespace sqrt_div_sqrt_l245_245221

theorem sqrt_div_sqrt (x y : ℝ) (h : ( (1/3)^2 + (1/4)^2 ) / ( (1/5)^2 + (1/6)^2 + 1/600 ) = 25 * x / (73 * y)) :
  sqrt x / sqrt y = 147 / 43 :=
by
  sorry

end sqrt_div_sqrt_l245_245221


namespace all_two_term_sums_are_integers_l245_245512

theorem all_two_term_sums_are_integers (real_numbers : Fin 13 → ℝ)
  (pairwise_sums : Fin 78 → ℝ)
  (sum_is_integer : ∀ i : Fin 78, i < 67 → ∃ (a b : Fin 13), pairwise_sums i = real_numbers a + real_numbers b ∧ (∃ (n : ℤ), pairwise_sums i = n)) :
  ∀ i : Fin 78, ∃ (a b : Fin 13), pairwise_sums i = real_numbers a + real_numbers b ∧ (∃ (n : ℤ), pairwise_sums i = n) :=
begin
  sorry,
end

end all_two_term_sums_are_integers_l245_245512


namespace proof_statement_l245_245011

variables (A B C D E F M : Type*) 
variables [has_coe_to_fun A B C D E F M (λ _, ℝ)]
variables (P Q R S : Proj ℝ)
variables [h : triangle ABC (points A B C)] 

-- Assume P, Q, R, and S are some points satisfying the below properties
axiom circle_through_A_B : is_circle_through (points A B) P
axiom P_intersects_AC : segment_intersects (segment A C) P D
axiom P_intersects_BC : segment_intersects (segment B C) P E
axiom line_AB_DE_intersect : line_intersects (line A B) (line D E) F
axiom line_BD_CF_intersect : line_intersects (line B D) (line C F) M

-- We need to prove MF = MC if and only if MB * MD = MC^2
theorem proof_statement : MF = MC ↔ MB * MD = MC * MC :=
    sorry

end proof_statement_l245_245011


namespace max_elements_diff_not_prime_l245_245943

-- Define helper functions and the set
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def difference_not_prime (s : set ℕ) : Prop :=
  ∀ (a b ∈ s), a ≠ b → ¬is_prime (abs (a - b))

theorem max_elements_diff_not_prime :
  ∃ s : set ℕ, s ⊆ { n : ℕ | 1 ≤ n ∧ n ≤ 1985 } ∧
              difference_not_prime s ∧
              s.card = 330 := sorry

end max_elements_diff_not_prime_l245_245943


namespace problem_statement_l245_245708

-- Define the function and properties
variable (f : ℝ → ℝ)

-- Assume f is an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(-x)

-- Assume f is monotonically increasing on [0, +∞)
def monotonically_increasing_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f(x) ≤ f(y)

-- The theorem to prove the problem statement
theorem problem_statement (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_mono : monotonically_increasing_nonneg f) 
  : f(-2) > f(1) :=
by sorry

end problem_statement_l245_245708


namespace infinite_solutions_l245_245690

theorem infinite_solutions (x y z : ℕ) (h : x^3 + y^4 = z^5) : 
  ∃ f : ℕ → ℕ × ℕ × ℕ, ∀ n, (f n).1^3 + (f n).2^4 = (f n).3^5 :=
sorry

end infinite_solutions_l245_245690


namespace translate_graph_x_squared_l245_245765

theorem translate_graph_x_squared:
  (∀ x, E x (x^2 - 2 * x + 1) = E (x-1) x^2)
:= sorry 

end translate_graph_x_squared_l245_245765


namespace symmedian_proof_l245_245818

-- Definitions for the conditions
variables {A B C A₁ C₁ B₀ Q : Type*}
variables [triangle ABC] [altitude AA₁ ABC] [altitude CC₁ ABC] [circumcircle ABC]
variables [common_point (altitude_from_B B (circumcircle ABC)) B₀]
variables [circumcircle_intersection A₁ B₀ C₁ Q (circumcircle ABC)]

-- Proof Statement
theorem symmedian_proof :
  symmedian B Q ABC :=
sorry

end symmedian_proof_l245_245818


namespace wilson_theorem_l245_245139

theorem wilson_theorem (p : ℕ) [Fact p.Prime] : (factorial (p - 1)) % p = p - 1 :=
sorry

end wilson_theorem_l245_245139


namespace area_of_triangle_formed_by_tangents_l245_245731

-- Definitions and conditions
variables {r R : ℝ} (r_pos : 0 < r) (R_pos : 0 < R)

-- Given conditions
def circles_non_intersecting (O1 O2 : ℝ × ℝ) : Prop :=
  dist O1 O2 > (r + R)

def tangents_perpendicular 
  (O1 O2 : ℝ × ℝ) 
  (A B C D: ℝ × ℝ) 
  (internal_tangent1 : line_segment O1 D = line_segment O2 D) 
  (internal_tangent2 : line_segment O1 C = line_segment O2 C) 
  : Prop :=
  ⟪(C - O1), (D - O1)⟫ = 0

-- Statement to prove
theorem area_of_triangle_formed_by_tangents 
  (O1 O2 A B C D : ℝ × ℝ)
  (non_intersecting : circles_non_intersecting O1 O2)
  (perpendicular_tangents : tangents_perpendicular O1 O2 A B C D
                            (internal_tangent1 : line_segment O1 D = line_segment O2 D)
                            (internal_tangent2 : line_segment O1 C = line_segment O2 C)) 
  : area_of_triangle A B D = r * R := 
sorry

end area_of_triangle_formed_by_tangents_l245_245731


namespace hemisphere_surface_area_l245_245913

-- Define the condition of the problem
def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2
def base_area_of_hemisphere : ℝ := 3

-- The proof problem statement
theorem hemisphere_surface_area : 
  ∃ (r : ℝ), (Real.pi * r^2 = 3) → (2 * Real.pi * r^2 + Real.pi * r^2 = 9) := 
by 
  sorry

end hemisphere_surface_area_l245_245913


namespace max_inscribed_triangle_area_l245_245514

theorem max_inscribed_triangle_area (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ A, A = (3 * Real.sqrt 3 / 4) * a * b := 
sorry

end max_inscribed_triangle_area_l245_245514


namespace each_person_shared_l245_245482

noncomputable def total_shared_cost_per_person : ℝ :=
let total_meal_cost := 35.25 + 42.50 + 61.25 in
let service_charge := 0.15 * total_meal_cost in
let total_bill_service := total_meal_cost + service_charge in
let share_before_tip := total_bill_service / 3 in
let tip_per_person := 0.10 * share_before_tip in
share_before_tip + tip_per_person

theorem each_person_shared : total_shared_cost_per_person = 58.61 := sorry

end each_person_shared_l245_245482


namespace lines_concurrent_l245_245008

noncomputable def angles :=
  list.set [108, 216]

structure polygon (n : ℕ) :=
(vertices : fin n → point)
(equal_sides : ∀ i j, abs (vertices i - vertices (i + 1)) = abs (vertices j - vertices (j + 1)))
(angles : ∀ i, (if odd i then 108 else 216) ∈ angles)

-- Define the specific polygon P
def P : polygon 20 := {
  vertices := sorry, -- we have to define the vertices properly with a specific arrangement
  equal_sides := sorry, -- all sides should be equal
  angles := sorry -- angles should alternate as described
}

-- Define collinearity of points
def collinear (a b c : point) : Prop :=
∃ k, b = a + k * (c - a)

-- Define concurrency of lines
def concurrent (l₁ l₂ l₃ l₄ l₅ : line) : Prop :=
∃ P, P ∈ l₁ ∧ P ∈ l₂ ∧ P ∈ l₃ ∧ P ∈ l₄ ∧ P ∈ l₅

theorem lines_concurrent :
  concurrent (line_through (P.vertices 1) (P.vertices 7))
             (line_through (P.vertices 3) (P.vertices 9))
             (line_through (P.vertices 4) (P.vertices 12))
             (line_through (P.vertices 5) (P.vertices 15))
             (line_through (P.vertices 6) (P.vertices 18)) :=
sorry -- proof goes here

end lines_concurrent_l245_245008


namespace josh_walk_ways_l245_245366

theorem josh_walk_ways (n : ℕ) :
  let grid_rows := n
      grid_columns := 3
      start_position := (0, 0)  -- (row, column) starting from bottom left
  in grid_rows > 0 →
      let center_square (k : ℕ) := (k, 1) -- center square of k-th row
  in ∃ ways : ℕ, ways = 2^(n-1) ∧
                ways = count_paths_to_center_topmost n  -- custom function representation
sorry

end josh_walk_ways_l245_245366


namespace find_b_l245_245073

theorem find_b (b : ℝ) :
  let slope_line1 := - (2 / 3)
  let slope_line2 := - (b / 3)
  slope_line1 * slope_line2 = -1 → b = - (9 / 2) :=
by
  intro h
  have : (-(2 / 3)) * (-(b / 3)) = -1 := h
  sorry

end find_b_l245_245073


namespace fuel_tank_oil_quantity_l245_245781

theorem fuel_tank_oil_quantity (t : ℝ) (Q : ℝ) : (Q = 40 - 0.2 * t) :=
begin
  sorry
end

end fuel_tank_oil_quantity_l245_245781


namespace find_lended_interest_rate_equivalent_interest_rate_l245_245578

variable (Principal BorrowedRate T LendedRate AnnualGain : ℝ)
variable BSI LSI AnnualInterestInterestRate : ℝ

noncomputable def calc_borrowed_interest : ℝ := Principal * BorrowedRate * T / 100

noncomputable def calc_lended_interest_rate : ℝ := 
  (Principal * LendedRate / 100) / T

theorem find_lended_interest_rate (calc_borrowed_interest = 200) -- Rs. 200 per year
  (AnnualGain = 100) :
  let AnnualInterestInterestRate = calc_borrowed_interest + AnnualGain in
  (AnnualInterestInterestRate * T) = BSI  := sorry

theorem equivalent_interest_rate (LendedRate = 6) :
  calc_lended_interest_rate = LSI := sorry

end find_lended_interest_rate_equivalent_interest_rate_l245_245578


namespace simplify_root_product_l245_245054

theorem simplify_root_product : 
  (625:ℝ)^(1/4) * (125:ℝ)^(1/3) = 25 :=
by
  have h₁ : (625:ℝ) = 5^4 := by norm_num
  have h₂ : (125:ℝ) = 5^3 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end simplify_root_product_l245_245054


namespace red_balls_in_box_l245_245792

theorem red_balls_in_box {n : ℕ} (h : n = 6) (p : (∃ (r : ℕ), r / 6 = 1 / 3)) : ∃ r, r = 2 :=
by
  sorry

end red_balls_in_box_l245_245792


namespace find_sample_size_l245_245546

def ratio_A : ℚ := 2
def ratio_B : ℚ := 3
def ratio_C : ℚ := 5
def total_ratio : ℚ := ratio_A + ratio_B + ratio_C
def proportion_B : ℚ := ratio_B / total_ratio
def units_B_sampled : ℚ := 24

theorem find_sample_size : ∃ n : ℚ, proportion_B = units_B_sampled / n ∧ n = 80 :=
by
  sorry

end find_sample_size_l245_245546


namespace division_of_cubics_l245_245216

theorem division_of_cubics (c d : ℕ) (h1 : c = 7) (h2 : d = 3) : 
  (c^3 + d^3) / (c^2 - c * d + d^2) = 10 := by
  sorry

end division_of_cubics_l245_245216


namespace sum_a_eq_9_l245_245488

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sum_a_eq_9 (a2 a3 a4 a5 a6 a7 : ℤ) 
  (h1 : 0 ≤ a2 ∧ a2 < 2) (h2 : 0 ≤ a3 ∧ a3 < 3) (h3 : 0 ≤ a4 ∧ a4 < 4)
  (h4 : 0 ≤ a5 ∧ a5 < 5) (h5 : 0 ≤ a6 ∧ a6 < 6) (h6 : 0 ≤ a7 ∧ a7 < 7)
  (h_eq : (5 : ℚ) / 7 = (a2 : ℚ) / factorial 2 + (a3 : ℚ) / factorial 3 + (a4 : ℚ) / factorial 4 + 
                         (a5 : ℚ) / factorial 5 + (a6 : ℚ) / factorial 6 + (a7 : ℚ) / factorial 7) :
  a2 + a3 + a4 + a5 + a6 + a7 = 9 := 
sorry

end sum_a_eq_9_l245_245488


namespace probability_sum_22_l245_245149

def twenty_faced_die_1 := { n : ℕ // n = 0 ∨ (1 ≤ n ∧ n ≤ 18) }
def twenty_faced_die_2 := { n : ℕ // n = 0 ∨ (2 ≤ n ∧ n ≤ 9) ∨ (11 ≤ n ∧ n ≤ 20) }

def valid_sum_22 (x y : ℕ) : Prop := x + y = 22

def valid_outcomes : set (twenty_faced_die_1 × twenty_faced_die_2) := 
{(x, y) | valid_sum_22 x y}

theorem probability_sum_22 : 
  ((set.card valid_outcomes).toReal / 20.0 / 20.0 = 1 / 40)
:= 
by 
  sorry

end probability_sum_22_l245_245149


namespace abs_add_lt_abs_add_l245_245253

open Real

theorem abs_add_lt_abs_add {a b : ℝ} (h : a * b < 0) : abs (a + b) < abs a + abs b := 
  sorry

end abs_add_lt_abs_add_l245_245253


namespace smallest_possible_positive_difference_l245_245503

def Vovochka_sum (a b : Nat) : Nat :=
  let ha := a / 100
  let ta := (a / 10) % 10
  let ua := a % 10
  let hb := b / 100
  let tb := (b / 10) % 10
  let ub := b % 10
  1000 * (ha + hb) + 100 * (ta + tb) + (ua + ub)

def correct_sum (a b : Nat) : Nat :=
  a + b

def difference (a b : Nat) : Nat :=
  abs (Vovochka_sum a b - correct_sum a b)

theorem smallest_possible_positive_difference :
  ∀ a b : Nat, (∃ a b : Nat, a > 0 ∧ b > 0 ∧ difference a b = 1800) :=
by
  sorry

end smallest_possible_positive_difference_l245_245503


namespace exists_two_unusual_numbers_l245_245560

noncomputable def is_unusual (n : ℕ) : Prop :=
  (n ^ 3 % 10 ^ 100 = n) ∧ (n ^ 2 % 10 ^ 100 ≠ n)

theorem exists_two_unusual_numbers :
  ∃ n1 n2 : ℕ, (is_unusual n1) ∧ (is_unusual n2) ∧ (n1 ≠ n2) ∧ (n1 >= 10 ^ 99) ∧ (n1 < 10 ^ 100) ∧ (n2 >= 10 ^ 99) ∧ (n2 < 10 ^ 100) :=
begin
  sorry
end

end exists_two_unusual_numbers_l245_245560


namespace initial_position_of_M_l245_245040

theorem initial_position_of_M :
  ∃ x : ℤ, (x + 7) - 4 = 0 ∧ x = -3 :=
by sorry

end initial_position_of_M_l245_245040


namespace inequality_proof_l245_245977

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end inequality_proof_l245_245977


namespace equation1_solution_equation2_solution_l245_245434

theorem equation1_solution (x : ℝ) : (x - 4)^2 - 9 = 0 ↔ (x = 7 ∨ x = 1) := 
sorry

theorem equation2_solution (x : ℝ) : (x + 1)^3 = -27 ↔ (x = -4) := 
sorry

end equation1_solution_equation2_solution_l245_245434


namespace carols_remaining_miles_l245_245621

noncomputable def distance_to_college : ℝ := 220
noncomputable def car_efficiency : ℝ := 20
noncomputable def gas_tank_capacity : ℝ := 16

theorem carols_remaining_miles :
  let gallons_used := distance_to_college / car_efficiency in
  let gallons_remaining := gas_tank_capacity - gallons_used in
  let miles_remaining := gallons_remaining * car_efficiency in
  miles_remaining = 100 :=
by
  sorry

end carols_remaining_miles_l245_245621


namespace train_length_l245_245171

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (conversion_factor : ℝ) (speed_ms : ℝ) (distance_m : ℝ) 
  (h1 : speed_kmh = 36) 
  (h2 : time_s = 28)
  (h3 : conversion_factor = 1000 / 3600) -- convert km/hr to m/s
  (h4 : speed_ms = speed_kmh * conversion_factor)
  (h5 : distance_m = speed_ms * time_s) :
  distance_m = 280 := 
by
  sorry

end train_length_l245_245171


namespace cost_of_cherries_l245_245176

theorem cost_of_cherries (total_spent amount_for_grapes amount_for_cherries : ℝ)
  (h1 : total_spent = 21.93)
  (h2 : amount_for_grapes = 12.08)
  (h3 : amount_for_cherries = total_spent - amount_for_grapes) :
  amount_for_cherries = 9.85 :=
sorry

end cost_of_cherries_l245_245176


namespace bobby_initial_candy_l245_245612

-- Define the constants and variables according to the problem conditions
def initial_candy : ℕ := sorry
def additional_candy : ℕ := 36
def chocolate : ℕ := 16
def more_candy_than_chocolate : ℕ := 58

-- Prove that Bobby ate 38 pieces of candy initially
theorem bobby_initial_candy : initial_candy = 38 :=
by
  have h1 : initial_candy + additional_candy = chocolate + more_candy_than_chocolate := sorry
  have h2 : initial_candy + 36 = 16 + 58 := h1
  have h3 : initial_candy + 36 = 74 := by norm_num
  have h4 : initial_candy = 74 - 36 := by linarith
  have h5 : initial_candy = 38 := by norm_num
  exact h5

end bobby_initial_candy_l245_245612


namespace greatest_prime_factor_391_l245_245110

theorem greatest_prime_factor_391 : ∃ p, prime p ∧ p ∣ 391 ∧ ∀ q, prime q ∧ q ∣ 391 → q ∣ p := by
  sorry

end greatest_prime_factor_391_l245_245110


namespace sum_of_all_three_digit_numbers_divisible_by_sum_of_digits_l245_245947

/-- The sum of all three-digit numbers divisible by the sum of their digits -/
theorem sum_of_all_three_digit_numbers_divisible_by_sum_of_digits : 
    ∃ S : ℕ, 
    S = ∑ n in finset.filter (λ n, 
        n ∈ [100, 101, ..., 999] ∧ 
        (let a := n / 100,
             b := (n / 10) % 10,
             c := n % 10 in
         (a + b + c) ∣ n)), 
    finset.range 900) :=
sorry

end sum_of_all_three_digit_numbers_divisible_by_sum_of_digits_l245_245947


namespace find_angle_EHG_l245_245345

noncomputable def angle_EHG (angle_EFG : ℝ) (angle_GHE : ℝ) : ℝ := angle_GHE - angle_EFG
 
theorem find_angle_EHG : 
  ∀ (EF GH : Prop) (angle_EFG angle_GHE : ℝ), (EF ∧ GH) → 
    EF ∧ GH ∧ angle_EFG = 50 ∧ angle_GHE = 80 → angle_EHG angle_EFG angle_GHE = 30 := 
by 
  intros EF GH angle_EFG angle_GHE h1 h2
  sorry

end find_angle_EHG_l245_245345


namespace hyperbola_proof_m_value_proof_l245_245258

noncomputable def hyperbola_eq := 
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (a * sqrt 3 = b ∧ (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1))) → 
  (a = sqrt (9/2) ∧ b = 3) ∧ (∀ x y : ℝ, (x^2 / (9/2) - y^2 / 9 = 1))

noncomputable def find_m := 
  ∀ (m : ℝ), 
  (∀ x y : ℝ, (x^2 / (9/2) - y^2 / 9 = 1) ∧ (x - y + m = 0)) ∧ 
  (let A B : (ℝ × ℝ) := ((x1, y1), (x2, y2)) in
    let mid := ((x1 + x2)/2, (y1 + y2)/2) in
    (fst mid)^2 + (snd mid)^2 = 5) → 
  (m = 1 ∨ m = -1)

-- Sorry we are not providing proofs
theorem hyperbola_proof : hyperbola_eq := sorry

theorem m_value_proof : find_m := sorry

end hyperbola_proof_m_value_proof_l245_245258


namespace find_a_such_that_f_is_odd_l245_245762

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^2 * x - 1
  else if x > 0 then x + a
  else 0

theorem find_a_such_that_f_is_odd :
  ∃ a : ℝ, (∀ x : ℝ, f a (-x) = -f a x) ↔ a = -1 :=
by
  sorry

end find_a_such_that_f_is_odd_l245_245762


namespace inequality_abc_l245_245282

theorem inequality_abc (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / (b ^ (1/2 : ℝ)) + b / (a ^ (1/2 : ℝ)) ≥ a ^ (1/2 : ℝ) + b ^ (1/2 : ℝ) :=
by { sorry }

end inequality_abc_l245_245282


namespace num_divisible_by_both_digits_l245_245627

theorem num_divisible_by_both_digits : 
  ∃ n, n = 14 ∧ ∀ (d : ℕ), (d ≥ 10 ∧ d < 100) → 
      (∀ a b, (d = 10 * a + b) → d % a = 0 ∧ d % b = 0 → (a = b ∨ a * 2 = b ∨ a * 5 = b)) :=
sorry

end num_divisible_by_both_digits_l245_245627


namespace fraction_equality_l245_245824

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 5 * b) / (b + 5 * a) = 2) : a / b = 3 / 5 :=
by
  sorry

end fraction_equality_l245_245824


namespace place_value_ratio_56439_2071_l245_245800

theorem place_value_ratio_56439_2071 :
  let n := 56439.2071
  let digit_6_place_value := 1000
  let digit_2_place_value := 0.1
  digit_6_place_value / digit_2_place_value = 10000 :=
by
  sorry

end place_value_ratio_56439_2071_l245_245800


namespace simplify_expression_l245_245430

theorem simplify_expression : 
  (1 / (1 / (1 / 2)^0 + 1 / (1 / 2)^1 + 1 / (1 / 2)^2 + 1 / (1 / 2)^3)) = 1 / 15 :=
by 
  sorry

end simplify_expression_l245_245430


namespace toms_total_money_l245_245095

def quarter_value : ℕ := 25 -- cents
def dime_value : ℕ := 10 -- cents
def nickel_value : ℕ := 5 -- cents
def penny_value : ℕ := 1 -- cent

def quarters : ℕ := 10
def dimes : ℕ := 3
def nickels : ℕ := 4
def pennies : ℕ := 200

def total_in_cents : ℕ := (quarters * quarter_value) + (dimes * dime_value) + (nickels * nickel_value) + (pennies * penny_value)

def total_in_dollars : ℝ := total_in_cents / 100

theorem toms_total_money : total_in_dollars = 5 := by
  sorry

end toms_total_money_l245_245095


namespace remainder_of_polynomial_l245_245115

-- Definitions
def polynomial : ℚ[X] := 5 * X^8 - X^7 + 3 * X^6 - 5 * X^4 + 6 * X^3 - 7
def divisor : ℚ[X] := 3 * X - 6
def evaluation_point : ℚ := 2
def remainder := polynomial.eval evaluation_point

-- Theorem statement
theorem remainder_of_polynomial :
  remainder = 1305 :=
sorry

end remainder_of_polynomial_l245_245115


namespace hyperbola_properties_l245_245331

theorem hyperbola_properties :
  let h := -3
  let k := 0
  let a := 5
  let c := Real.sqrt 50
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ h + k + a + b = 7 :=
by
  sorry

end hyperbola_properties_l245_245331


namespace angle_between_vectors_l245_245677

-- Define the conditions as constants.
variables (a b : ℝ^3)
variable (θ : ℝ)

-- Assume the conditions given in the problem.
axiom norm_a : ∥a∥ = 3
axiom norm_b : ∥b∥ = 4
axiom dot_product_eq : (a + b) • (a + 3 * b) = 33

-- State the theorem.
theorem angle_between_vectors : θ = 120 * (π / 180) := 
sorry

end angle_between_vectors_l245_245677


namespace unusual_numbers_exist_l245_245563

noncomputable def n1 : ℕ := 10 ^ 100 - 1
noncomputable def n2 : ℕ := 10 ^ 100 / 2 - 1

theorem unusual_numbers_exist : 
  (n1 ^ 3 % 10 ^ 100 = n1 ∧ n1 ^ 2 % 10 ^ 100 ≠ n1) ∧ 
  (n2 ^ 3 % 10 ^ 100 = n2 ∧ n2 ^ 2 % 10 ^ 100 ≠ n2) :=
by
  sorry

end unusual_numbers_exist_l245_245563


namespace integral_value_l245_245964

noncomputable def definite_integral : ℝ :=
  ∫ x in 0..Real.arcsin (Real.sqrt (3 / 7)), 
    (Real.tan x)^2 / (3 * (Real.sin x)^2 + 4 * (Real.cos x)^2 - 7)

theorem integral_value :
  definite_integral = -Real.sqrt 3 / 8 + 3 * Real.sqrt 3 * Real.pi / 32 :=
by
  sorry

end integral_value_l245_245964


namespace minimum_black_edges_l245_245639

-- Define the dimensions and faces of the rectangular prism
def dimensions := (2, 2, 1)

-- Define the condition of edge colors and the necessity for at least 2 black edges on each face
structure RectangularPrism :=
  (edges : Fin 12 → Bool)  -- 12 edges, each either black (true) or red (false)
  (face1_has_two_black_edges : count_black_edges (Fin 4.map edges) ≥ 2) -- bottom face
  (face2_has_two_black_edges : count_black_edges (Fin 4.map edges) ≥ 2) -- top face
  (face3_has_two_black_edges : count_black_edges (Fin 4.map edges) ≥ 2) -- front face
  (face4_has_two_black_edges : count_black_edges (Fin 4.map edges) ≥ 2) -- back face
  (face5_has_two_black_edges : count_black_edges (Fin 4.map edges) ≥ 2) -- left face
  (face6_has_two_black_edges : count_black_edges (Fin 4.map edges) ≥ 2) -- right face

-- Function to count black edges on a face
def count_black_edges (edges : Fin 4 → Bool) : Nat :=
  edges.toList.count (λ x => x)

-- Proposition to prove the minimum number of black edges is 8
theorem minimum_black_edges : ∃ (rp : RectangularPrism), (rp.edges.toList.count (λ x => x) = 8) := by
  sorry

end minimum_black_edges_l245_245639


namespace real_solutions_l245_245228

theorem real_solutions (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 4) :
  ( ( (x - 1)^3 * (x - 2)^2 * (x - 3)^3 * (x - 4)^2 / ( (x - 2) * (x - 4)^2 * (x - 2) ) = 2 ) ↔ 
  ( x = 2 + real.sqrt(4 + real.cbrt(2)) ∨ x = 2 - real.sqrt(4 + real.cbrt(2)) )) :=
sorry

end real_solutions_l245_245228


namespace log_eq_solution_l245_245479

noncomputable def log_eq_solution_set : Set ℝ :=
  { x : ℝ | log 2 (x - 1) = 2 - log 2 (x + 1) }

theorem log_eq_solution :
  log_eq_solution_set = { real.sqrt 5 } :=
by
s
orry

end log_eq_solution_l245_245479


namespace find_r_values_l245_245395

noncomputable def r (a : list ℝ) (i : ℕ) (h : i < a.length) : ℝ :=
a.nth_le i h / Real.sqrt (list.sum (a.map (λ x, x^2)))

theorem find_r_values (n : ℕ) (a : list ℝ) (h_length : a.length = n) 
  (h_nonzero : ∀ i, i < n → a.nth i ≠ 0)
  (r : list ℝ) 
  (h_r : ∀ (i : ℕ) (hi : i < n), r.nth_le i hi = r a i hi) :
  ∀ (x : list ℝ), x.length = a.length →
    (list.sum (list.zip_with (λ r xk, r * (xk - a.nth_le x.index_of xk (by simp))), r, x))
    ≤ (Real.sqrt (list.sum (x.map (λ xk, xk^2))) - Real.sqrt (list.sum (a.map (λ ak, ak^2)))) :=
by
  sorry

end find_r_values_l245_245395


namespace necessary_but_not_sufficient_l245_245694

variables (a b c : ℝ)

-- Definitions of the propositions
def p := a > b
def q := a * c^2 > b * c^2

-- The theorem stating the necessary but not sufficient condition relationship
theorem necessary_but_not_sufficient :
  (∀ (c : ℝ), q → c^2 > 0 → p) ∧ ¬(∀ (c : ℝ), p → q) :=
by
  sorry

end necessary_but_not_sufficient_l245_245694


namespace correct_answers_l245_245949

open Real

noncomputable def problem_statements := 
  (∀ x : ℝ, x ≠ 0 → x < 0 → (∃ y, y = (x^2 + 1) / x ∧ y < 2)) ∧
  (∀ a b : ℝ, ab > 0 → (b / a + a / b ≥ 2)) ∧
  (∀ x : ℝ, (x^2 + 2 + 1 / (x^2 + 2) ≠ 2)) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → (1 / sqrt (a * b) + (a + b) / 2 ≥ 2))

theorem correct_answers : problem_statements :=
by
  sorry

end correct_answers_l245_245949


namespace sabrina_total_leaves_l245_245048

theorem sabrina_total_leaves (num_basil num_sage num_verbena : ℕ)
  (h1 : num_basil = 2 * num_sage)
  (h2 : num_sage = num_verbena - 5)
  (h3 : num_basil = 12) :
  num_sage + num_verbena + num_basil = 29 :=
by
  sorry

end sabrina_total_leaves_l245_245048


namespace max_AMC_expression_l245_245401

theorem max_AMC_expression (A M C : ℕ) (h : A + M + C = 15) : A * M * C + A * M + M * C + C * A ≤ 200 :=
by
  sorry

end max_AMC_expression_l245_245401


namespace share_of_annual_gain_l245_245126

theorem share_of_annual_gain (x : ℕ) (gain : ℕ) (hx : gain = 18300) (h : a_investment = x * 12) (h2 : b_investment = 2 * x * 6) (h3 : c_investment = 3 * x * 4) : 
  let total_investment := a_investment + b_investment + c_investment in
  (a_investment / total_investment) * gain = 6100 := 
by
  sorry

end share_of_annual_gain_l245_245126


namespace range_of_a_l245_245721

theorem range_of_a (a : ℝ) : 
  0 < 1 - 2 * a ∧ 1 - 2 * a < 1 ∧ 0 < a ∧ a < 1 ∧ 1 - 2 * a ≥ 1 / 3 → 0 < a ∧ a ≤ 1 / 3 := 
by {
  sorry,
}

end range_of_a_l245_245721


namespace range_of_x_l245_245693

noncomputable 
def proposition_p (x : ℝ) : Prop := 6 - 3 * x ≥ 0

noncomputable 
def proposition_q (x : ℝ) : Prop := 1 / (x + 1) < 0

theorem range_of_x (x : ℝ) : proposition_p x ∧ ¬proposition_q x → x ∈ Set.Icc (-1 : ℝ) (2 : ℝ) := by
  sorry

end range_of_x_l245_245693


namespace fifth_number_in_10th_row_of_lattice_l245_245571

theorem fifth_number_in_10th_row_of_lattice : 
  (10th row contains 10 * 7 - 6 + 5) := by
    sorry

end fifth_number_in_10th_row_of_lattice_l245_245571


namespace area_of_BCD_l245_245389

variables (a b c x y : ℝ)

-- Conditions
axiom h1 : x = (1 / 2) * a * b
axiom h2 : y = (1 / 2) * b * c

-- Conclusion to prove
theorem area_of_BCD (a b c x y : ℝ) (h1 : x = (1 / 2) * a * b) (h2 : y = (1 / 2) * b * c) : 
  (1 / 2) * b * c = y :=
sorry

end area_of_BCD_l245_245389


namespace complementary_angle_decrease_l245_245460

theorem complementary_angle_decrease (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90)
  (h2 : angle1 / 3 = angle2 / 6) (h3 : ∃ x: ℝ, x = 0.2) :
  let new_angle1 := angle1 * 1.2 in 
  let new_angle2 := 90 - new_angle1 in
  (new_angle2 - angle2) / angle2 = -0.10 := sorry

end complementary_angle_decrease_l245_245460


namespace largest_r_l245_245428

theorem largest_r (a : ℕ → ℕ) (h : ∀ n, 0 < a n ∧ a n ≤ a (n + 2) ∧ a (n + 2) ≤ Int.sqrt (a n ^ 2 + 2 * a (n + 1))) :
  ∃ M, ∀ n ≥ M, a (n + 2) = a n :=
sorry

end largest_r_l245_245428


namespace most_accurate_significant_value_l245_245340

theorem most_accurate_significant_value (K : ℝ) (err : ℝ) 
  (hK : K = 3.71729) (herr : err = 0.00247) : 
  ∃ (accurate_K : ℝ), accurate_K = 3.7 ∧ 
    (3.71482 ≤ K + err ∧ 3.71976 ≥ K - err) :=
by
  use 3.7
  split
  · rfl
  · sorry

end most_accurate_significant_value_l245_245340


namespace equilateral_triangle_area_l245_245757

theorem equilateral_triangle_area (s b : ℝ)
  (h : (sqrt 3 / 4) * s^2 = 6 + b * sqrt 3) : b = 1 :=
by
  sorry

end equilateral_triangle_area_l245_245757


namespace probability_real_roots_discrete_probability_real_roots_continuous_l245_245730

-- Define the quadratic equation
def quadratic_eq (a b : ℝ) : ℝ :=
  2 * a * (0 : ℝ) - b^2 + 4

-- Define the condition for real roots
def has_real_roots (a b : ℝ) : Prop :=
  a^2 + b^2 ≥ 4

-- Discrete case: given a ∈ {-1, 0, 1} and b ∈ {-3, -2, -1, 0, 1}
theorem probability_real_roots_discrete :
  let a_set := {-1, 0, 1}
  let b_set := {-3, -2, -1, 0, 1}
  ∃ prob : ℚ, 
    prob = 2 / 5 ∧ 
    ∀ a ∈ a_set, ∀ b ∈ b_set, 
      (has_real_roots a b ↔ ((a = -1 ∧ b ∈ {-3, -2}) ∨ (a = 0 ∧ b ∈ {-3, -2}) ∨ (a = 1 ∧ b ∈ {-3, -2})) ↔ b_set) :=
sorry

-- Continuous case: given a ∈ [-2, 2] and b ∈ [-2, 2]
theorem probability_real_roots_continuous :
  let interval := (-2 : ℝ) .. (2 : ℝ)
  ∃ prob : ℝ, 
    prob = 1 - Real.pi / 4 ∧ 
    ∀ (a b : ℝ), 
      a ∈ interval ∧ b ∈ interval ∧ has_real_roots a b :=
sorry

end probability_real_roots_discrete_probability_real_roots_continuous_l245_245730


namespace max_black_cells_on_board_l245_245613

-- Define the game and the strategy conditions
def borya_vova_game := true

-- Define the board setup and the strategies
constant board : Type
constant initialize_board : board → board
constant turn_borya : board → board
constant turn_vova : board → board

-- Assume board is initially white with 8×8 grid
axiom is_initial_white_8x8 : initialize_board = board

-- Define the conditions of Borya's and Vova's moves.
axiom borya_move : ∀ (b : board), turn_borya b = board
axiom vova_move : ∀ (b : board), turn_vova b = board

theorem max_black_cells_on_board : ∀ (b : board), borya_vova_game → board = initialize_board →
  (∃ (max_black_cells : ℕ), max_black_cells = 25) :=
by
  sorry

end max_black_cells_on_board_l245_245613


namespace number_of_selection_methods_l245_245166

theorem number_of_selection_methods : 
  let male_select := Nat.choose 5 2 in
  let female_select := Nat.choose 6 2 in
  let pairings := Nat.factorial 2 in
  male_select * female_select * pairings = 300 :=
by 
  -- Definitions and calculations
  let male_select := Nat.choose 5 2
  let female_select := Nat.choose 6 2
  let pairings := Nat.factorial 2
  have h_male_select : male_select = 10 := by simp [male_select]
  have h_female_select : female_select = 15 := by simp [female_select]
  have h_pairings : pairings = 2 := by simp [pairings]
  calc
    male_select * female_select * pairings
        = 10 * 15 * 2 : by rw [h_male_select, h_female_select, h_pairings]
    ... = 300 : by norm_num

end number_of_selection_methods_l245_245166


namespace range_of_a_l245_245763

open Real

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a + b * cos x + c * sin x

theorem range_of_a (a b c : ℝ) :
  (∀ x, x ∈ Icc 0 (π/2) → abs (f a b c x) ≤ 2) →
  f a b c 0 = 1 →
  f a b c (π/2) = 1 →
  a ∈ Icc (-sqrt 2) (4 + 3 * sqrt 2) :=
by sorry

end range_of_a_l245_245763


namespace pascal_triangle_sum_l245_245206

theorem pascal_triangle_sum :
  (\sum i in Finset.filter (λ x, x % 2 = 0) (Finset.range 501), (502 - i) / 502) -
  (\sum i in Finset.filter (λ x, x % 2 = 0) (Finset.range 501), (501 - i) / 501) = 0 :=
by sorry

end pascal_triangle_sum_l245_245206


namespace painted_cubes_even_faces_l245_245438

theorem painted_cubes_even_faces :
  let L := 6 -- length of the block
  let W := 2 -- width of the block
  let H := 2 -- height of the block
  let total_cubes := 24 -- the block is cut into 24 1-inch cubes
  let cubes_even_faces := 12 -- the number of 1-inch cubes with even number of blue faces
  -- each cube has a total of 6 faces,
  -- we need to count how many cubes have an even number of painted faces.
  L * W * H = total_cubes → 
  cubes_even_faces = 12 := sorry

end painted_cubes_even_faces_l245_245438


namespace option_B_option_D_l245_245274

noncomputable def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- The maximum value of (y + 1) / (x + 1) is 2 + sqrt(6)
theorem option_B (x y : ℝ) (h : curve_C x y) :
  ∃ k, (y + 1) / (x + 1) = k ∧ k = 2 + Real.sqrt 6 :=
sorry

-- A tangent line through the point (0, √2) on curve C has the equation x - √2 * y + 2 = 0
theorem option_D (h : curve_C 0 (Real.sqrt 2)) :
  ∃ a b c, a * 0 + b * Real.sqrt 2 + c = 0 ∧ c = 2 ∧ a = 1 ∧ b = - Real.sqrt 2 :=
sorry

end option_B_option_D_l245_245274


namespace invalid_votes_count_l245_245184

def total_polled_votes := 850
def candidate1_votes (V : ℕ) := 0.20 * V
def candidate2_votes (V : ℕ) := 0.80 * V
def vote_difference := 500

theorem invalid_votes_count (V : ℕ) 
  (h1 : 0.8 * V - 0.2 * V = vote_difference) 
  (h_total_polled_votes : total_polled_votes = 850) : 
  total_polled_votes - V = 17 :=
  sorry

end invalid_votes_count_l245_245184


namespace complex_number_sum_equals_one_l245_245825

variable {a b c d : ℝ}
variable {ω : ℂ}

theorem complex_number_sum_equals_one
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1) 
  (hd : d ≠ -1) 
  (hω : ω^4 = 1) 
  (hω_ne : ω ≠ 1)
  (h_eq : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω)
  : (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 :=
by sorry

end complex_number_sum_equals_one_l245_245825


namespace seth_boxes_l245_245878

-- Define the conditions as hypotheses
variables
  (boxes_market : ℕ)
  (boxes_farm : ℕ)
  (total_initial : ℕ)
  (after_giving_to_mother : ℕ)
  (after_donations : ℕ)
  (final_boxes : ℕ)

-- Conditions
def initial_conditions :=
  boxes_market = 3 ∧
  boxes_farm = 2 * boxes_market ∧
  total_initial = boxes_market + boxes_farm ∧
  after_giving_to_mother = total_initial - 1 ∧
  after_donations = after_giving_to_mother - after_giving_to_mother / 4 ∧
  final_boxes = after_donations - 5 * 0

-- Question: How many boxes did Seth buy in the first place?
theorem seth_boxes
  (h : initial_conditions) : total_initial = 9 :=
sorry

end seth_boxes_l245_245878


namespace eating_time_correct_l245_245410

noncomputable def time_to_eat_cereal (pounds : ℚ) : ℚ :=
  pounds / (1/15 + 1/45)

theorem eating_time_correct :
  time_to_eat_cereal 5 = 56.25 :=
by
  -- Setup the known facts
  let fat_rate := (1 : ℚ) / 15
  let thin_rate := (1 : ℚ) / 45
  let combined_rate := fat_rate + thin_rate
  -- Calculate time to eat 5 pounds
  let time := 5 / combined_rate
  -- Assert and prove the final result
  have h : time = 56.25, by sorry
  exact h

end eating_time_correct_l245_245410


namespace slope_of_tangent_line_at_1_l245_245080

noncomputable def y (x : ℝ) : ℝ := 3 * Real.log x - 1 / x

def derivativeOfY (x : ℝ) : ℝ := (3 / x) + (1 / x^2)

theorem slope_of_tangent_line_at_1 :
  derivativeOfY 1 = 4 := by
  sorry

end slope_of_tangent_line_at_1_l245_245080


namespace possibleStartingCities_l245_245741

-- Define the cities as an inductive type
inductive City 
| StPetersburg
| Tver
| Yaroslavl
| NizhnyNovgorod
| Moscow
| Kazan

open City

-- Define the tickets as a list of pairs of cities
def tickets : List (City × City) := [
  (StPetersburg, Tver),
  (Yaroslavl, NizhnyNovgorod),
  (Moscow, Kazan),
  (NizhnyNovgorod, Kazan),
  (Moscow, Tver),
  (Moscow, NizhnyNovgorod)
]

-- Define the property that checks if a path visits each city exactly once
def validJourney (path : List City) : Prop := 
  path.nodup ∧ path.length = 6 ∧ 
  ∀ (a b : City), (a, b) ∈ tickets ∨ (b, a) ∈ tickets → 
                   a ∈ path.tail ∧ b ∈ path

-- State the theorem that the journey can only start from St. Petersburg or Yaroslavl
theorem possibleStartingCities :
  ∃ path, validJourney path ∧ 
  (path.head = StPetersburg ∨ path.head = Yaroslavl) := 
sorry

end possibleStartingCities_l245_245741


namespace max_value_of_function_l245_245235

theorem max_value_of_function : 
  ∃ x : ℝ, (∀ y : ℝ, f(y) ≤ f(x)) ∧ f(x) = 4 / 3 :=
by
  let f : ℝ → ℝ := λ x, 1 / (1 - x * (1 - x))
  use 1 / 2
  sorry

end max_value_of_function_l245_245235


namespace find_percentage_decrease_l245_245465

-- Define the measures of two complementary angles
def angles_complementary (a b : ℝ) : Prop := a + b = 90

-- Given variables
variable (small_angle large_angle : ℝ)

-- Given conditions
def ratio_of_angles (small_angle large_angle : ℝ) : Prop := small_angle / large_angle = 1 / 2

def increased_small_angle (small_angle : ℝ) : ℝ := small_angle * 1.2

noncomputable def new_large_angle (small_angle large_angle : ℝ) : ℝ :=
  90 - increased_small_angle small_angle

def percentage_decrease (original new : ℝ) : ℝ :=
  ((original - new) / original) * 100

-- The theorem we need to prove
theorem find_percentage_decrease
  (h1 : ratio_of_angles small_angle large_angle)
  (h2 : angles_complementary small_angle large_angle) :
  percentage_decrease large_angle (new_large_angle small_angle large_angle) = 10 :=
sorry

end find_percentage_decrease_l245_245465


namespace toms_total_money_l245_245097

def quarter_value : ℕ := 25 -- cents
def dime_value : ℕ := 10 -- cents
def nickel_value : ℕ := 5 -- cents
def penny_value : ℕ := 1 -- cent

def quarters : ℕ := 10
def dimes : ℕ := 3
def nickels : ℕ := 4
def pennies : ℕ := 200

def total_in_cents : ℕ := (quarters * quarter_value) + (dimes * dime_value) + (nickels * nickel_value) + (pennies * penny_value)

def total_in_dollars : ℝ := total_in_cents / 100

theorem toms_total_money : total_in_dollars = 5 := by
  sorry

end toms_total_money_l245_245097


namespace coat_price_reduction_l245_245550

noncomputable def reduced_price_with_tax (price : ℝ) (reduction : ℝ) (tax : ℝ) : ℝ :=
  let reduced_price := price * (1 - reduction)
  let taxed_price := reduced_price * (1 + tax)
  taxed_price

theorem coat_price_reduction : 
  let initial_price : ℝ := 500
  let first_month_reduction : ℝ := 0.1
  let first_month_tax : ℝ := 0.05
  let second_month_reduction : ℝ := 0.15
  let second_month_tax : ℝ := 0.08
  let third_month_reduction : ℝ := 0.2
  let third_month_tax : ℝ := 0.06
  let price_after_first_month := reduced_price_with_tax initial_price first_month_reduction first_month_tax
  let price_after_second_month := reduced_price_with_tax price_after_first_month second_month_reduction second_month_tax
  let price_after_third_month := reduced_price_with_tax price_after_second_month third_month_reduction third_month_tax
  let total_percent_reduction := (initial_price - price_after_third_month) / initial_price * 100
  price_after_third_month ≈ 367.824 ∧ total_percent_reduction ≈ 26.44 := 
  by
    sorry

end coat_price_reduction_l245_245550


namespace correct_option_D_l245_245597

variables {a b : Line} {α : Plane}

def skew_lines (a b : Line) : Prop := ¬ (∃ p, p ∈ a ∧ p ∈ b) ∧ ¬ parallel a b
def perpendicular (a b : Line) : Prop := ∃ p, p ∈ a ∧ p ∈ b ∧ angle a b = 90
def perpendicular (a : Line) (α : Plane) : Prop := ∀ l : Line, l ∈ α → perpendicular a l
def passes_through (α : Plane) (a : Line) : Prop := ∃ p, p ∈ α ∧ p ∈ a

theorem correct_option_D :
  skew_lines a b ∧ ¬ perpendicular a b →
  ∀ α, passes_through α a → ¬ perpendicular b α :=
sorry

end correct_option_D_l245_245597


namespace sabrina_total_leaves_l245_245045

-- Definitions based on conditions
def basil_leaves := 12
def twice_the_sage_leaves (sages : ℕ) := 2 * sages = basil_leaves
def five_fewer_than_verbena (sages verbenas : ℕ) := sages + 5 = verbenas

-- Statement to prove
theorem sabrina_total_leaves (sages verbenas : ℕ) 
    (h1 : twice_the_sage_leaves sages) 
    (h2 : five_fewer_than_verbena sages verbenas) :
    basil_leaves + sages + verbenas = 29 :=
sorry

end sabrina_total_leaves_l245_245045


namespace is_isosceles_l245_245063

variables {α : Type*} [linear_ordered_field α]

structure Triangle (V : Type*) [add_comm_group V] [module α V] :=
(A B C : V)
(altitudes : Prop)

def is_altitude {V : Type*} [add_comm_group V] [module α V] (A B C D: V) : Prop :=
∃ l : α, D = A + l • (C - B)

lemma altitudes_intersect_at 
  {V : Type*} [add_comm_group V] [module α V] 
  {A B C M : V} 
  (h1: is_altitude B A C M) 
  (h2: is_altitude C A B M) :
  ∃ (BM CM : α), BM = CM → BM = CM :=
sorry

theorem is_isosceles
  {V : Type*} [add_comm_group V] [module α V]
  {A B C M : V}
  (h1: is_altitude B A C M) 
  (h2: is_altitude C A B M)
  (h3: ∃ (BM CM : α), BM = CM → BM = CM):
  ∃ (a b: α), a = b :=
sorry

end is_isosceles_l245_245063


namespace relationship_between_MH_and_MK_l245_245347

theorem relationship_between_MH_and_MK
  (L BC : Line ℝ)
  (E H K M D : Point ℝ)
  (L_not_perpendicular : ¬ is_perpendicular L BC)
  (H_on_L : H ∈ L)
  (K_on_L : K ∈ L)
  (BH_perp_BC : perp (Line.through B H) BC)
  (CK_perp_BC : perp (Line.through C K) BC)
  (M_midpoint : M = midpoint B C)
  (BD_3DC : ∃ D, dist B D = 3 * dist D C) :
  dist M H = dist M K :=
sorry

end relationship_between_MH_and_MK_l245_245347


namespace cube_volume_l245_245776

-- Definition of the problem conditions
def space_diagonal (s : ℝ) : ℝ :=
  s * sqrt 3

-- The fact we need to prove
theorem cube_volume (d : ℝ) (h : d = 5 * sqrt 3) :
  ∃ s : ℝ, space_diagonal s = d ∧ s^3 = 125 :=
by
  sorry

end cube_volume_l245_245776


namespace max_rect_area_l245_245580

theorem max_rect_area (l w : ℤ) (h1 : 2 * l + 2 * w = 40) (h2 : 0 < l) (h3 : 0 < w) : 
  l * w ≤ 100 :=
by sorry

end max_rect_area_l245_245580


namespace u_n_eq_n_squared_S_n_formula_l245_245966

-- Define u_n as the sum of the first n odd numbers
def u_n (n : ℕ) : ℕ := List.sum (List.map (λ i, 2 * i + 1) (List.range n))

-- State the theorem u_n = n^2
theorem u_n_eq_n_squared (n : ℕ) : u_n n = n^2 := 
by sorry

-- Define S_n as the sum of cubes of the first n natural numbers
def S_n (n : ℕ) : ℕ := List.sum (List.map (λ i, (i + 1)^3) (List.range n))

-- State the theorem for S_n
theorem S_n_formula (n : ℕ) : S_n n = (n * (n + 1) / 2)^2 :=
by sorry

end u_n_eq_n_squared_S_n_formula_l245_245966


namespace cross_time_is_27_seconds_l245_245101

noncomputable def time_to_cross_trains
  (length1 length2 : ℕ) (speed1_kmh speed2_kmh : ℕ) : ℕ :=
let relative_speed := (speed1_kmh + speed2_kmh) * 5 / 18 in  -- converting to m/s
let total_length := length1 + length2 in
total_length / relative_speed

def length1 := 250
def length2 := 500
def speed1 := 60
def speed2 := 40

theorem cross_time_is_27_seconds :
  time_to_cross_trains length1 length2 speed1 speed2 = 27 := 
by
  sorry

end cross_time_is_27_seconds_l245_245101


namespace triangle_area_change_l245_245766

theorem triangle_area_change (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let A_original := (B * H) / 2
  let H_new := H * 0.60
  let B_new := B * 1.40
  let A_new := (B_new * H_new) / 2
  (A_new = A_original * 0.84) :=
by
  sorry

end triangle_area_change_l245_245766


namespace percentage_problem_l245_245753

variable (x : ℝ)

theorem percentage_problem (h : 0.4 * x = 160) : 240 / x = 0.6 :=
by sorry

end percentage_problem_l245_245753


namespace carol_additional_miles_l245_245622

theorem carol_additional_miles (distance_home : ℕ) (mpg : ℕ) (gallons : ℕ) 
  (h_home : distance_home = 220) (h_mpg : mpg = 20) (h_gallons : gallons = 16) :
  let total_distance := mpg * gallons
  in total_distance - distance_home = 100 :=
by
  sorry

end carol_additional_miles_l245_245622


namespace water_level_at_rim_l245_245248

-- Defining the densities of water and ice
def ρ_воды : ℝ := 1000 -- Density of water in kg/m^3
def ρ_льда : ℝ := 917 -- Density of ice in kg/m^3

-- Defining volumes
variables (V W U : ℝ)

-- Conservation of mass assuming all water converted to ice and vice versa
axiom conservation_of_mass : V * ρ_воды = W * ρ_льда

-- Relating the volumes based on conservation of mass
lemma volume_relation : W = V * (ρ_воды / ρ_льда) :=
by sorry

-- Using Archimedes' principle for floating ice
lemma archimedes_principle : U = V :=
by sorry

theorem water_level_at_rim (initial_volume_filled_to_brim : V) :
  W = V :=
begin
  apply archimedes_principle,
  apply volume_relation,
  assumption
end

end water_level_at_rim_l245_245248


namespace knight_count_l245_245035

theorem knight_count (K L : ℕ) (h1 : K + L = 15) 
  (h2 : ∀ k, k < K → (∃ l, l < L ∧ l = 6)) 
  (h3 : ∀ l, l < L → (K > 7)) : K = 9 :=
by 
  sorry

end knight_count_l245_245035


namespace greatest_prime_factor_391_l245_245112

theorem greatest_prime_factor_391 : ∃ p, Prime p ∧ p ∣ 391 ∧ ∀ q, Prime q ∧ q ∣ 391 → q ≤ p :=
by
  sorry

end greatest_prime_factor_391_l245_245112


namespace inequality_proof_l245_245981

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end inequality_proof_l245_245981


namespace min_cost_per_ounce_l245_245154

theorem min_cost_per_ounce 
  (cost_40 : ℝ := 200) (cost_90 : ℝ := 400)
  (percentage_40 : ℝ := 0.4) (percentage_90 : ℝ := 0.9)
  (desired_percentage : ℝ := 0.5) :
  (∀ (x y : ℝ), 0.4 * x + 0.9 * y = 0.5 * (x + y) → 200 * x + 400 * y / (x + y) = 240) :=
sorry

end min_cost_per_ounce_l245_245154


namespace number_of_ways_to_choose_lineup_l245_245418

   -- Define the problem conditions
   def num_players : ℕ := 13
   def positions : ℕ := 6

   -- Define the theorem to prove
   theorem number_of_ways_to_choose_lineup : 
     (Finset.range num_players).card.factorial.div ((Finset.range (num_players - positions)).card.factorial) = 1_027_680 := by
     sorry
   
end number_of_ways_to_choose_lineup_l245_245418


namespace single_elimination_tournament_l245_245861

theorem single_elimination_tournament (T : ℕ) (hT : T = 32) : 
  ∃ n : ℕ, (n = T - 1) ∧ (n = 31) :=
by {
  existsi 31,
  split,
  { rw hT,
    exact nat.sub_one 32 },
  { refl }
}

end single_elimination_tournament_l245_245861


namespace exists_point_in_half_subsets_l245_245640

theorem exists_point_in_half_subsets 
  (S : Finset (Set ℝ))
  (h1 : ∀ s ∈ S, ∃ a b c d : ℝ, s = Icc a b ∪ Icc c d)
  (h2 : ∀ s1 s2 s3 ∈ S, ∃ p : ℝ, p ∈ s1 ∧ p ∈ s2 ∧ p ∈ s3) : 
  ∃ p : ℝ, (∃ T ⊆ S, T.card ≥ S.card / 2 ∧ ∀ s ∈ T, p ∈ s) := 
sorry

end exists_point_in_half_subsets_l245_245640


namespace solve_quadratic_1_solve_quadratic_2_l245_245058

-- Define the first problem
theorem solve_quadratic_1 (x : ℝ) : 3 * x^2 - 4 * x = 2 * x → x = 0 ∨ x = 2 := by
  -- Proof step will go here
  sorry

-- Define the second problem
theorem solve_quadratic_2 (x : ℝ) : x * (x + 8) = 16 → x = -4 + 4 * Real.sqrt 2 ∨ x = -4 - 4 * Real.sqrt 2 := by
  -- Proof step will go here
  sorry

end solve_quadratic_1_solve_quadratic_2_l245_245058


namespace jenny_sweets_division_l245_245002

theorem jenny_sweets_division :
  ∃ n : ℕ, n ∣ 30 ∧ n ≠ 5 ∧ n ≠ 12 :=
by
  existsi 6
  repeat {split}
  · exact dvd_refl 30
  · exact ne_of_gt (by norm_num)
  · exact ne_of_gt (by norm_num)
  sorry

end jenny_sweets_division_l245_245002


namespace ratio_current_to_past_l245_245625

-- Conditions
def current_posters : ℕ := 22
def posters_after_summer (p : ℕ) : ℕ := p + 6
def posters_two_years_ago : ℕ := 14

-- Proof problem statement
theorem ratio_current_to_past (h₁ : current_posters = 22) (h₂ : posters_two_years_ago = 14) : 
  (current_posters / Nat.gcd current_posters posters_two_years_ago) = 11 ∧ 
  (posters_two_years_ago / Nat.gcd current_posters posters_two_years_ago) = 7 :=
by
  sorry

end ratio_current_to_past_l245_245625


namespace arithmetic_sequence_sum_l245_245391

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := n * (a + (a + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum:
  ∀ (a₃ a₅ : ℝ), a₃ = 5 → a₅ = 9 →
  ∃ (d a₁ a₇ : ℝ), arithmetic_sequence a₁ d 3 = a₃ ∧ arithmetic_sequence a₁ d 5 = a₅ ∧ a₁ + a₇ = 14 ∧
  sum_arithmetic_sequence a₁ d 7 = 49 :=
by
  intros a₃ a₅ h₁ h₂
  sorry

end arithmetic_sequence_sum_l245_245391


namespace percentage_decrease_in_larger_angle_l245_245468

-- Define the angles and conditions
def angles_complementary (A B : ℝ) : Prop := A + B = 90
def angle_ratio (A B : ℝ) : Prop := A / B = 1 / 2
def small_angle_increase (A A' : ℝ) : Prop := A' = A * 1.2
def large_angle_new (A' B' : ℝ) : Prop := A' + B' = 90

-- Prove percentage decrease in larger angle
theorem percentage_decrease_in_larger_angle (A B A' B' : ℝ) 
    (h1 : angles_complementary A B)
    (h2 : angle_ratio A B)
    (h3 : small_angle_increase A A')
    (h4 : large_angle_new A' B')
    : (B - B') / B * 100 = 10 :=
sorry

end percentage_decrease_in_larger_angle_l245_245468


namespace simplify_expression_l245_245190

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ((x + y) ^ 2 - (x - y) ^ 2) / (4 * x * y) = 1 := 
by sorry

end simplify_expression_l245_245190


namespace area_of_isosceles_trapezoid_l245_245601

noncomputable def trapezoid_area (b1 b2 h : ℝ) : ℝ := 0.5 * (b1 + b2) * h

theorem area_of_isosceles_trapezoid :
  let long_base := 20
  let base_angle := Real.arcsin 0.6
  let height := 9
  ∃ (short_base : ℝ) (leg : ℝ),
  Real.sin base_angle = 0.6 ∧
  height = leg * 0.6 ∧
  long_base = short_base + 2 * leg * 0.8 ∧
  trapezoid_area long_base short_base height = 100 :=
by
  let long_base := 20
  let base_angle := Real.arcsin 0.6
  let height := 9
  let leg := 20 * 0.5555555555555556 -- approximately 11.11
  let short_base := 20 * 0.1111111111111111 -- approximately 2.22
  use short_base, leg
  split,
  { exact Real.sin_arcsin_of_leighy 0.6 (by norm_num) (by norm_num) },
  split,
  { sorry }, -- height computation
  split,
  { sorry }, -- base computation
  { sorry } -- area computation


end area_of_isosceles_trapezoid_l245_245601


namespace correct_options_l245_245277

-- Given condition
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- Option B assertion
def option_B (x y : ℝ) : Prop := (x^2 + y^2 - 4)/((x - 1)^2 + y^2 + 1) ≤ 2 + Real.sqrt 6

-- Option D assertion
def option_D (x y : ℝ) : Prop := x - Real.sqrt 2 * y + 2 = 0

-- Theorem to prove both options B and D are correct under the given condition
theorem correct_options {x y : ℝ} (h : curve x y) : option_B x y ∧ option_D x y := by
  sorry

end correct_options_l245_245277


namespace greater_than_neg2_by_1_l245_245121

theorem greater_than_neg2_by_1 : -2 + 1 = -1 := by
  sorry

end greater_than_neg2_by_1_l245_245121


namespace albert_new_percentage_increase_l245_245755

variable {E P : ℝ}

theorem albert_new_percentage_increase 
  (h₁ : E * 1.14 = 678) 
  (h₂ : E * (1 + P) = 683.95) :
  P ≈ 0.14957983193 :=
by
  sorry

end albert_new_percentage_increase_l245_245755


namespace inversely_proportional_y_l245_245830

theorem inversely_proportional_y (k : ℚ) (x y : ℚ) (hx_neg_10 : x = -10) (hy_5 : y = 5) (hprop : y * x = k) (hx_neg_4 : x = -4) : 
  y = 25 / 2 := 
by
  sorry

end inversely_proportional_y_l245_245830


namespace f_f_neg2_eq_3_l245_245896

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then log x / log 2 else x * (x - 2)

theorem f_f_neg2_eq_3 : f (f (-2)) = 3 :=
by
  sorry

end f_f_neg2_eq_3_l245_245896


namespace find_percentage_decrease_l245_245466

-- Define the measures of two complementary angles
def angles_complementary (a b : ℝ) : Prop := a + b = 90

-- Given variables
variable (small_angle large_angle : ℝ)

-- Given conditions
def ratio_of_angles (small_angle large_angle : ℝ) : Prop := small_angle / large_angle = 1 / 2

def increased_small_angle (small_angle : ℝ) : ℝ := small_angle * 1.2

noncomputable def new_large_angle (small_angle large_angle : ℝ) : ℝ :=
  90 - increased_small_angle small_angle

def percentage_decrease (original new : ℝ) : ℝ :=
  ((original - new) / original) * 100

-- The theorem we need to prove
theorem find_percentage_decrease
  (h1 : ratio_of_angles small_angle large_angle)
  (h2 : angles_complementary small_angle large_angle) :
  percentage_decrease large_angle (new_large_angle small_angle large_angle) = 10 :=
sorry

end find_percentage_decrease_l245_245466


namespace inequality_solution_set_l245_245229

theorem inequality_solution_set :
  ∀ x : ℝ, (1 / (x^2 + 1) > 5 / x + 21 / 10) ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) :=
by
  sorry

end inequality_solution_set_l245_245229


namespace calculator_transform_implication_l245_245543

noncomputable def transform (x n S : ℕ) : Prop :=
  (S > x^n + 1)

theorem calculator_transform_implication (x n S : ℕ) (hx : 0 < x) (hn : 0 < n) (hS : 0 < S) 
  (h_transform: transform x n S) : S > x^n + x - 1 := by
  sorry

end calculator_transform_implication_l245_245543


namespace smallest_positive_integer_modulo_l245_245116

theorem smallest_positive_integer_modulo {n : ℕ} (h : 19 * n ≡ 546 [MOD 13]) : n = 11 := by
  sorry

end smallest_positive_integer_modulo_l245_245116


namespace octagon_area_l245_245941

-- Define the given condition: the circle has an area of 400π square units
def circle_area := 400 * Real.pi

-- Define the calculation of the radius from the circle's area
def radius (ca : ℝ) : ℝ := Real.sqrt (ca / Real.pi)

-- Define the formula to compute the area of an isosceles triangle with given base angle and radius
def octagon_triangle_area (r : ℝ) (angle : ℝ) : ℝ := (1/2) * r * r * Real.sin angle

-- Translation of the problem statement into a theorem in Lean 4
theorem octagon_area {ca : ℝ} (h : ca = circle_area) : 
  let r := radius ca in
  let tri_angle := Real.pi / 4 in
  let one_triangle_area := octagon_triangle_area r tri_angle in
  8 * one_triangle_area = 800 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l245_245941


namespace flag_arrangement_division_l245_245926

noncomputable def flag_arrangement_modulo : ℕ :=
  let num_blue_flags := 9
  let num_red_flags := 8
  let num_slots := num_blue_flags + 1
  let initial_arrangements := (num_slots.choose num_red_flags) * (num_blue_flags + 1)
  let invalid_cases := (num_blue_flags.choose num_red_flags) * 2
  let M := initial_arrangements - invalid_cases
  M % 1000

theorem flag_arrangement_division (M : ℕ) (num_blue_flags num_red_flags : ℕ) :
  num_blue_flags = 9 → num_red_flags = 8 → M = flag_arrangement_modulo → M % 1000 = 432 :=
by
  intros _ _ hM
  rw [hM]
  trivial

end flag_arrangement_division_l245_245926


namespace response_rate_increase_approx_l245_245958

theorem response_rate_increase_approx :
  let original_customers := 80
  let original_respondents := 7
  let redesigned_customers := 63
  let redesigned_respondents := 9
  let original_response_rate := (original_respondents : ℝ) / original_customers * 100
  let redesigned_response_rate := (redesigned_respondents : ℝ) / redesigned_customers * 100
  let percentage_increase := (redesigned_response_rate - original_response_rate) / original_response_rate * 100
  abs (percentage_increase - 63.24) < 0.01 := by
  sorry

end response_rate_increase_approx_l245_245958


namespace find_b_l245_245481

theorem find_b (a b c : ℕ) (h1 : a + b + c = 99) (h2 : a + 6 = b - 6) (h3 : b - 6 = 5 * c) : b = 51 :=
sorry

end find_b_l245_245481


namespace lucius_weekly_earnings_l245_245853

-- Define constants
def price_french_fries := 12
def price_poutine := 8
def price_onion_rings := 6

-- Define total portions sold over the week
def total_french_fries := 75
def total_poutine := 50
def total_onion_rings := 60

-- Define total sales amounts
def total_sales_french_fries := total_french_fries * price_french_fries
def total_sales_poutine := total_poutine * price_poutine
def total_sales_onion_rings := total_onion_rings * price_onion_rings

-- Calculate total sales for the week
def total_sales := total_sales_french_fries + total_sales_poutine + total_sales_onion_rings

-- Define the cost range for ingredients per day
def min_daily_cost := 8
def max_daily_cost := 15

-- Calculate the average daily cost of ingredients
def avg_daily_cost := (min_daily_cost + max_daily_cost) / 2

-- Define the number of days in a week
def number_of_days := 7

-- Calculate the total cost of ingredients for the week
def total_cost_ingredients := avg_daily_cost * number_of_days

-- Define the tax rate
def tax_rate := 0.10

-- Calculate the total tax
def total_tax := total_sales * tax_rate

-- Calculate Lucius's total earnings after tax and cost of ingredients
def total_earnings := total_sales - total_tax - total_cost_ingredients

-- Lean statement for the proof problem
theorem lucius_weekly_earnings : total_earnings = 1413.50 := by
  sorry

end lucius_weekly_earnings_l245_245853


namespace percent_married_employees_l245_245604

def total_employees : ℕ := sorry
def percent_women : ℝ := 0.61
def percent_married_women : ℝ := 0.7704918032786885
def single_men_fraction : ℝ := (2 : ℝ) / 3
def married_men_fraction : ℝ := 1 - single_men_fraction

theorem percent_married_employees :
  let E := total_employees in
  let women := percent_women * E in
  let men := (1 - percent_women) * E in
  let married_women := percent_married_women * women in
  let married_men := married_men_fraction * men in
  let percent_married := 100 * (married_women + married_men) / E in
  percent_married = 60.020016 :=
sorry

end percent_married_employees_l245_245604


namespace minimize_sum_of_ratios_l245_245656

theorem minimize_sum_of_ratios
  (a b c PD PE PF : ℝ)
  (P_area : ℝ)
  (h1 : P_area = (1/2) * (a * PD + b * PE + c * PF))
  (P_is_incenter : PD = PE ∧ PE = PF) :
  ∃ P : Type, (Point_in_triangle P)
  ((BC PD) + (CA PE) + (AB PF)) = (a + b + c)^2 / (2 * P_area) :=
by
  sorry

end minimize_sum_of_ratios_l245_245656


namespace not_prime_infinite_n_l245_245423

theorem not_prime_infinite_n (a : ℤ) : ∃ᶠ (n : ℕ) in at_top, ¬ prime (a ^ 2^n + 2^n) :=
sorry

end not_prime_infinite_n_l245_245423


namespace sum_of_even_factors_of_720_l245_245946

-- Given conditions
def n : ℕ := 720

def prime_factorization_of_n : Prop :=
  (2 ^ 4) * (3 ^ 2) * (5 ^ 1) = n

def even_factors_form (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1

-- Main statement to prove
theorem sum_of_even_factors_of_720 : 
  prime_factorization_of_n ∧ (∀ a b c, even_factors_form a b c) →
  ∑ (a in [1,2,3,4]), ∑ (b in [0,1,2]), ∑ (c in [0,1]), 2^a * 3^b * 5^c = 2340 :=
by sorry

end sum_of_even_factors_of_720_l245_245946


namespace find_angle_ABC_l245_245668

/-- Given conditions for angles at point B --/
variable (CBD ABC ABD : ℝ)
axiom angle_CBD_is_right : CBD = 90
axiom sum_of_angles_around_B : ABC + ABD + CBD = 200
axiom angle_ABD : ABD = 70

/-- Prove that the measure of angle ABC is 40 degrees --/
theorem find_angle_ABC : ABC = 40 := by
  sorry

end find_angle_ABC_l245_245668


namespace find_amount_l245_245142

-- Let A be the certain amount.
variable (A x : ℝ)

-- Given conditions
def condition1 (x : ℝ) := 0.65 * x = 0.20 * A
def condition2 (x : ℝ) := x = 150

-- Goal
theorem find_amount (A x : ℝ) (h1 : condition1 A x) (h2 : condition2 x) : A = 487.5 := 
by 
  sorry

end find_amount_l245_245142


namespace swap_columns_produce_B_l245_245007

variables {n : ℕ} (A : Matrix (Fin n) (Fin n) (Fin n))

def K (B : Matrix (Fin n) (Fin n) (Fin n)) : ℕ :=
  Fintype.card {ij : (Fin n) × (Fin n) // B ij.1 ij.2 = ij.2}

theorem swap_columns_produce_B (A : Matrix (Fin n) (Fin n) (Fin n)) :
  ∃ (B : Matrix (Fin n) (Fin n) (Fin n)), (∀ i, ∃ j, B i j = A i j) ∧ K B ≤ n :=
sorry

end swap_columns_produce_B_l245_245007


namespace rhombus_perimeter_l245_245892

def diagonal1 : ℝ := 24
def diagonal2 : ℝ := 16
def answer : ℝ := 16 * Real.sqrt 13

theorem rhombus_perimeter :
  let d1 := diagonal1 / 2
  let d2 := diagonal2 / 2
  let side := Real.sqrt (d1^2 + d2^2)
  4 * side = answer :=
by
  sorry

end rhombus_perimeter_l245_245892


namespace rotate270_neg8_add_2i_l245_245539

/-- The complex number after a 270-degree counter-clockwise rotation around the origin -/
theorem rotate270_neg8_add_2i : 
  let z := (-8 + 2 * Complex.i : Complex)
  let rotation := (-Complex.i : Complex)
  rotation * z = -2 + 8 * Complex.i :=
by
  sorry

end rotate270_neg8_add_2i_l245_245539


namespace total_money_l245_245090

def value_of_quarters (count: ℕ) : ℝ := count * 0.25
def value_of_dimes (count: ℕ) : ℝ := count * 0.10
def value_of_nickels (count: ℕ) : ℝ := count * 0.05
def value_of_pennies (count: ℕ) : ℝ := count * 0.01

theorem total_money (q d n p : ℕ) :
  q = 10 → d = 3 → n = 4 → p = 200 →
  value_of_quarters q + value_of_dimes d + value_of_nickels n + value_of_pennies p = 5.00 :=
by {
  intros,
  sorry
}

end total_money_l245_245090


namespace solve_equation_l245_245057

theorem solve_equation :
  ∀ x : ℝ, 
    (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 4)) ↔ 
      (x = (-3 + Real.sqrt 37) / 2 ∨ x = (-3 - Real.sqrt 37) / 2) := 
by
  intro x
  sorry

end solve_equation_l245_245057


namespace error_percentage_approx_111_l245_245160

theorem error_percentage_approx_111 {y : ℕ} (hy : 0 < y) : 
    let correct_result := y + 8
    let erroneous_result := y - 12
    let error := |correct_result - erroneous_result|
    let error_percentage := (error * 100) / correct_result
    in error_percentage ≈ 111 :=
by
    let correct_result := y + 8
    let erroneous_result := y - 12
    let error := |correct_result - erroneous_result|
    let error_percentage := (error * 100) / correct_result
    have h : error = 20 := by sorry
    have h_correct_result : correct_result = y + 8 := rfl
    have h_error_percentage : error_percentage ≈ 111 :=
        have h_pos : y = 10 := by sorry
        sorry
    exact h_error_percentage

end error_percentage_approx_111_l245_245160


namespace josh_paths_l245_245362

theorem josh_paths (n : ℕ) (h : n > 0) : 
  let start := (0, 0)
  let end := (n - 1, 1)
  -- the number of distinct paths from start to end is 2^(n-1)
  (if n = 1 then 1 else 2^(n-1)) = 2^(n-1) :=
by
  sorry

end josh_paths_l245_245362


namespace hemisphere_surface_area_l245_245910

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (hπ : π = Real.pi) (h : π * r^2 = 3) :
    2 * π * r^2 + 3 = 9 :=
by
  sorry

end hemisphere_surface_area_l245_245910


namespace find_number_l245_245519

theorem find_number (x : ℝ) (h : 0.8 * x = (2/5 : ℝ) * 25 + 22) : x = 40 :=
by
  sorry

end find_number_l245_245519


namespace bacteria_growth_hours_bacteria_count_after_24_hours_l245_245889

theorem bacteria_growth_hours (initial : ℕ) (rate : ℕ) (target : ℕ) (period : ℕ) 
  (h_initial : initial = 200) (h_rate : rate = 4) (h_target : target = 819200) 
  (h_period : period = 12) : 
  ∃ t : ℕ, (initial * rate^(t / period) = target) ∧ t = 72 :=
begin
  sorry
end

theorem bacteria_count_after_24_hours (initial : ℕ) (rate : ℕ)
  (h_initial : initial = 200) (h_rate : rate = 4) :
  initial * rate^2 = 3200 :=
begin
  sorry
end

end bacteria_growth_hours_bacteria_count_after_24_hours_l245_245889


namespace symmetric_circle_eq_l245_245445

/-- The equation of the circle symmetric to the circle (x-2)^2+(y-3)^2=1 with respect to the line
    x+y-1=0 is (x+2)^2+(y+1)^2=1. -/
theorem symmetric_circle_eq :
  ∀ (x y : ℝ), let c : ℝ × ℝ := (2, 3),
                   r : ℝ := 1,
                   l (x y : ℝ) : Prop := x + y - 1 = 0,
                   c' : ℝ × ℝ := (-2, -1)
  in (x - (c.fst))^2 + (y - (c.snd))^2 = r^2 →
     l c.fst c.snd →
     (x - (c'.fst))^2 + (y - (c'.snd))^2 = r^2
:= sorry

end symmetric_circle_eq_l245_245445


namespace function_relation_l245_245450

theorem function_relation (f : ℝ → ℝ)
  (h_diff : ∀ x, differentiable_at ℝ f x)
  (h_symm : ∀ x, f x = f (2 - x))
  (h_ineq : ∀ x, x ≠ 1 → (x - 1) * (f' x) < 0) :
  let a := f (Real.tan (5 * Real.pi / 4))
  let b := f (Real.log 2 / Real.log 3)
  let c := f (0.2 ^ (-3))
  in c < b ∧ b < a := by
  sorry

end function_relation_l245_245450


namespace total_first_class_equipment_l245_245607

theorem total_first_class_equipment (x y : ℕ) (hx : x < y)
  (h1 : 0.36 * x - 0.6 * y = 26)
  (h2 : 32 * x > 25 * y) :
  5 * y + 30 = 60 :=
by
  let z := 3 * x - 5 * y;
  have hz : z = 130, from sorry;
  have y_val : y = 4, from sorry;
  have total_first_class := 5 * y + 30;
  have correct_answer : total_first_class = 60, by sorry;
  exact correct_answer

end total_first_class_equipment_l245_245607


namespace solve_quadratic_equation_l245_245652

noncomputable def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

theorem solve_quadratic_equation :
  let x1 := -2
      x2 := 11 in
  quadratic_eq 1 (-9) (-22) x1 ∧ quadratic_eq 1 (-9) (-22) x2 ∧ x1 < x2 :=
by
  sorry

end solve_quadratic_equation_l245_245652


namespace necessary_and_sufficient_condition_for_inscription_l245_245935

section ConvexTetrahedralAngleInscription

variable (α β γ δ α' α'' β' β'' γ' γ'' δ' δ'' : Real)
variable (l : Line) -- Rotation axis

-- Conditions
def conditions (α β γ δ : Real) : Prop := 
  α' + α'' = α ∧
  β' + β'' = β ∧
  γ' + γ'' = γ ∧
  δ' + δ'' = δ ∧
  α' = δ'' ∧
  α'' = β' ∧
  γ' = β'' ∧
  γ'' = δ'

theorem necessary_and_sufficient_condition_for_inscription 
  (h : conditions α β γ δ) : 
  α + γ = β + δ :=
by
  sorry

end ConvexTetrahedralAngleInscription

end necessary_and_sufficient_condition_for_inscription_l245_245935


namespace inequality_proof_l245_245989

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l245_245989


namespace part_a_part_b_l245_245404

variable {R : Type*} [ring R] (h : ∀ x : R, ∃ y z : R, (idempotent y) ∧ (idempotent z) ∧ x = y * z)

theorem part_a (h1 : ∀ a b : R, ab = 1 → a = 1 ∧ b = 1) : 
  ∀ a b : R, ab = 1 → a = 1 ∧ b = 1 := sorry

theorem part_b : ∀ x : R, x^2 = x := sorry

end part_a_part_b_l245_245404


namespace solution_set_of_inequality_l245_245883

variable {f : ℝ → ℝ}

-- Conditions
axiom odd_fn : ∀ x, f(-x) = -f(x)
axiom increasing_fn : ∀ {a b : ℝ}, 0 < a → a < b → f(a) < f(b)
axiom f_at_2 : f(2) = 0

-- Theorem
theorem solution_set_of_inequality :
  {x : ℝ | (f(x) - f(-x)) / x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 2} :=
sorry

end solution_set_of_inequality_l245_245883


namespace correct_propositions_l245_245214

theorem correct_propositions :
  let circle1 := (x : ℝ, y : ℝ) -> (x + 2)^2 + (y + 1)^2 = 4
  let line1 := (x : ℝ, y : ℝ) -> x - 2 * y = 0
  let circle2 := (θ : ℝ) -> (x : ℝ, y : ℝ) -> (x - cos θ)^2 + (y - sin θ)^2 = 1
  let line2 := (k : ℝ) -> (x : ℝ, y : ℝ) -> y = k * x
  let tetrahedron_edges := all (eq (sqrt 2))
  let sphere_volume := sqrt 3 / 2 * π
  (¬ ∃ x y, circle1 x y ∧ line1 x y ∧ (4 = 2)) ∧
  (∀ k θ, ∃ x y, line2 k x y ∧ circle2 θ x y) ∧
  (∀ a, (a = 2 ↔ ∀ x y, ¬ (ax + 2y = 0 ∧ x + y = 1))) ∧
  (∀ θ, tetrahedron_edges θ →  ∃ r : ℝ, r = sqrt 3 / 2 → sphere_volume = (4 / 3) * π * r^3)
:= sorry

end correct_propositions_l245_245214


namespace area_of_regular_inscribed_octagon_l245_245937

theorem area_of_regular_inscribed_octagon (r : ℝ) (h : π * r^2 = 400 * π) : 
  (8 * (1/2) * r^2 * sin(π/4)) = 800 * sqrt 2 :=
begin
  sorry
end

end area_of_regular_inscribed_octagon_l245_245937


namespace equilateral_if_centroid_eq_circumcenter_l245_245424

theorem equilateral_if_centroid_eq_circumcenter (ABC : Triangle) (G : Point) :
  is_centroid G ABC ∧ is_circumcenter G ABC → is_equilateral ABC := 
sorry

end equilateral_if_centroid_eq_circumcenter_l245_245424


namespace inequality_proof_l245_245982

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end inequality_proof_l245_245982


namespace correct_propositions_l245_245598

-- Define the conditions as propositions
def proposition1 (a b c : ℝ) : Prop := (a * c^2 < b * c^2) → (a < b)
def proposition2 (m : ℝ) : Prop := m > 0 → ∃ x : ℝ, x^2 + x - m = 0
def proposition3 (quad_has_circumscribed_circle : Prop) : Prop := ∀ q : quad_has_circumscribed_circle, q
def proposition4 (a b : ℝ) : Prop := (a * b ≠ 0) → (a ≠ 0)

-- The proof problem
theorem correct_propositions {a b c : ℝ} {m : ℝ} (quad_has_circumscribed_circle : Prop) : 
  ¬proposition1 a b c ∧ proposition2 m ∧ ¬proposition3 quad_has_circumscribed_circle ∧ proposition4 a b :=
by
  sorry

end correct_propositions_l245_245598


namespace angle_ratio_l245_245339

theorem angle_ratio (A B C : ℝ) (hA : A = 60) (hB : B = 80) (h_sum : A + B + C = 180) : B / C = 2 := by
  sorry

end angle_ratio_l245_245339


namespace x_pow4_plus_one_over_x_pow4_eq_two_l245_245768

theorem x_pow4_plus_one_over_x_pow4_eq_two (x : ℝ) (h : x^2 + x⁻² = 2) : 
    x^4 + x⁻⁴ = 2 := 
by 
  sorry

end x_pow4_plus_one_over_x_pow4_eq_two_l245_245768


namespace intersection_of_medians_x_coord_l245_245967

def parabola (x : ℝ) : ℝ := x^2 - 4 * x - 1

theorem intersection_of_medians_x_coord (x_a x_b : ℝ) (y : ℝ) :
  (parabola x_a = y) ∧ (parabola x_b = y) ∧ (parabola 5 = parabola 5) → 
  (2 : ℝ) < ((5 + 4) / 3) :=
sorry

end intersection_of_medians_x_coord_l245_245967


namespace graph_k_linked_of_2k_connected_and_edge_count_l245_245390

open GraphTheory

variables {V : Type*} [Fintype V]

def is_k_linked (G : SimpleGraph V) (k : ℕ) : Prop :=
  ∀ (X : Finset V), X.card ≤ k → ∃ (Y : Finset V), Y.card = k ∧ G.linked X Y

theorem graph_k_linked_of_2k_connected_and_edge_count {G : SimpleGraph V} (k : ℕ) 
  (h1 : G.2k_connected) (h2 : G.edge_count ≥ 8 * k) : G.is_k_linked k := 
by
  sorry

end graph_k_linked_of_2k_connected_and_edge_count_l245_245390


namespace abs_lt_implies_neg_lt_counterexample_neg_lt_not_implies_abs_lt_l245_245023

theorem abs_lt_implies_neg_lt {a b : ℝ} (ha : a > 0) (hb : b ∈ ℝ) : (|a| < b) → (-a < b) :=
by 
  intro h₀,
  have h₁ : -b < a ∧ a < b := abs_lt.mp h₀,
  exact h₁.right

theorem counterexample_neg_lt_not_implies_abs_lt : ∃ (a b : ℝ), -a < b ∧ ¬(|a| < b) :=
by 
  use [1, 0],
  constructor
  . exact neg_one_lt_zero
  . change |(1)| < 0,
    linarith

end abs_lt_implies_neg_lt_counterexample_neg_lt_not_implies_abs_lt_l245_245023


namespace total_cost_of_calls_l245_245244

-- Define the necessary constants and conditions
def call_duration_dad : Nat := 45
def call_duration_brother : Nat := 31
def call_duration_cousin : Nat := 20
def call_duration_grandparents : Nat := 15

def local_rate_per_min_peak : Float := 5 * 1.5
def local_rate_per_min_off_peak : Float := 5
def national_rate_per_min_peak : Float := 10 * 1.5
def national_rate_per_min_off_peak : Float := 10
def international_rate_per_min_brother_peak : Float := 25 * 1.2
def international_rate_per_min_brother_off_peak : Float := 25
def international_rate_per_min_grandparents_peak : Float := 30 * 1.2
def international_rate_per_min_grandparents_off_peak : Float := 30

def conversion_rate_brother_currency : Float := 1.1
def conversion_rate_grandparents_currency : Float := 0.85

-- Time zone and peak hour conditions
def brother_offset_hours : Int := 8
def grandparents_offset_hours : Int := -6
def peak_start_hour : Int := 18
def peak_end_hour : Int := 21

-- Correct answer (total cost in USD)
def correct_total_cost : Float := 17.72

theorem total_cost_of_calls :
  let cost_dad := (call_duration_dad * local_rate_per_min_peak) / 100
  let cost_brother := ((call_duration_brother * international_rate_per_min_brother_off_peak) / conversion_rate_brother_currency) / 100
  let cost_cousin := (call_duration_cousin * national_rate_per_min_off_peak) / 100
  let cost_grandparents := ((call_duration_grandparents * international_rate_per_min_grandparents_off_peak) / conversion_rate_grandparents_currency) / 100
  let total_cost := cost_dad + cost_brother + cost_cousin + cost_grandparents
  total_cost = correct_total_cost := 
by
  sorry

end total_cost_of_calls_l245_245244


namespace find_a_l245_245318

def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem find_a (a b : ℤ) (h1 : b = a + 1)
  (h2 : ∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) : a = 2 :=
by
  sorry

end find_a_l245_245318


namespace locus_of_R_proof_l245_245665

section LocusOfPointR

variable (x y x₁ y₁ : ℝ)

-- Conditions
def is_parabola (y x : ℝ) : Prop := y^2 = 2 * x
def directrix : ℝ := -1 / 2
def focus : (ℝ × ℝ) := (1 / 2, 0)
def vertex : (ℝ × ℝ) := (0, 0)

-- Definition of point P on parabola
def point_on_parabola (P : ℝ × ℝ) : Prop := is_parabola P.2 P.1

-- Locus of R
def locus_of_R (R : ℝ × ℝ) : Prop := 
  ∃ x₁ y₁ : ℝ, 
    is_parabola y₁ x₁ ∧ 
    R.1 = x₁ / (2 * (1 + x₁)) ∧ 
    R.2 = y₁ / (2 * (1 + x₁))

-- The proof problem
theorem locus_of_R_proof : locus_of_R (x, y) → y^2 = -2 * x^2 + x := sorry

end LocusOfPointR

end locus_of_R_proof_l245_245665


namespace slope_l3_proof_l245_245025

theorem slope_l3_proof :
  ∃ (m : ℚ),
    let l1 := λ x y : ℚ, 4 * x - 3 * y = 2,
        A := (0 : ℚ, -2 : ℚ),
        l2 := λ y : ℚ, y = 3,
        B := (11 / 4 : ℚ, 3 : ℚ),
        C := (131 / 20 : ℚ, 3 : ℚ),
        area_ABC := 12 in
      l1 A.1 A.2 ∧ -- A is on l1
      l2 B.2 ∧ -- B is on l2
      B = (11 / 4, 3) ∧ -- Coordinates of B
      C = (131 / 20, 3) ∧ -- Coordinates of C
      ∃ (l3_slope : ℚ), l3_slope = (C.2 - A.2) / (C.1 - A.1) ∧
      area_ABC = 12 ∧
      l3_slope = 100 / 131 :=
by
  sorry

end slope_l3_proof_l245_245025


namespace arithmetic_sequence_problem_l245_245886

noncomputable def a : ℕ → ℚ := sorry

def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, (n > 0 → a (n + 1) - a n = a n - a (n - 1))

def sum_of_odd_indexed_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.filter (λ i, i % 2 = 1) (finset.range n), a i

theorem arithmetic_sequence_problem (h_arith : is_arithmetic_sequence a) 
  (h_sum : sum_of_odd_indexed_terms a 22 = 10) : 
  a 10 = 10 / 21 :=
sorry

end arithmetic_sequence_problem_l245_245886


namespace hexagon_rectangle_intersection_l245_245831

-- Define point S on the circumscribed circle of triangle PQR
variable {P Q R S : Point} 
variable (S_on_circle : S ∈ circumscribed_circle P Q R)
variable {A B C D E F : Point}
variable (hexagon_circular : inscribed_in_circle [A, B, C, D, E, F])

-- Define the condition of perpendicular feet
def collinearity_condition (S : Point) (P Q R : Point) : Prop :=
  ∃ l, collinear [foot_perpendicular S P Q, foot_perpendicular S Q R, foot_perpendicular S R P]

-- Define the proposed lines l(A, BDF), l(B, ACE), l(D, ABF), and l(E, ABC)
def l (S : Point) (P Q R : Point) : Line :=
  let P' := foot_perpendicular S P Q in
  let Q' := foot_perpendicular S Q R in
  let R' := foot_perpendicular S R P in
  Line_through P' Q' R'

-- Lean 4 statement translating the problem
theorem hexagon_rectangle_intersection :
  (collinearity_condition A B D F
  ∧ collinearity_condition B A C E
  ∧ collinearity_condition D A B F
  ∧ collinearity_condition E A B C)
  ↔ rectangle C D E F :=
sorry

end hexagon_rectangle_intersection_l245_245831


namespace cos_B_l245_245770

theorem cos_B (A B C : ℝ) (hC : C = 90) (htanA : tan A = sqrt 3 / 3) : 
  cos B = 1 / 2 :=
sorry

end cos_B_l245_245770


namespace center_number_is_thirteen_l245_245864

theorem center_number_is_thirteen :
  ∀ (arr : Array (Array ℕ)),
    (arr.length = 3 ∧ arr.all (λ row => row.length = 3)) →
    ({5, 6, 7, 8, 9, 10, 11, 12, 13} ⊆ arr.flatten) →
    (∀ i j, (i + 1 < 3 → arr[i][j].succ = arr[i + 1][j]) ∧
            (j + 1 < 3 → arr[i][j].succ = arr[i][j + 1])) →
    (arr[0][0] + arr[2][0] + arr[0][2] + arr[2][2] = 32) →
    (arr[1][1] = 13) :=
by
    intros arr hlength hflatten hconsecutive hsum
    sorry

end center_number_is_thirteen_l245_245864


namespace range_of_f_l245_245226

def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((1 - x) / (1 + x))

theorem range_of_f : 
  (∃ x : ℝ, f x = -3*Real.pi/4) ∧ (∃ x : ℝ, f x = Real.pi/4) ∧ (∀ y : ℝ, (y ≠ -3*Real.pi/4 ∧ y ≠ Real.pi/4) → ¬∃ x : ℝ,  f x = y) := 
by 
  sorry

end range_of_f_l245_245226


namespace acute_triangle_problem_l245_245722

-- Define the vectors involved in the function f
def m (x : ℝ) : ℝ × ℝ := (Real.sin x + Real.cos x, Real.sqrt 3 * Real.cos x)
def n (x : ℝ) : ℝ × ℝ := (Real.cos x - Real.sin x, 2 * Real.sin x)

-- Define the function f
def f (x : ℝ) : ℝ := (m x).fst * (n x).fst + (m x).snd * (n x).snd

-- Prove the stated conditions and results
theorem acute_triangle_problem {A B C a b c : ℝ}
  (ha : a = Real.sqrt 3)
  (hA : 0 < A ∧ A < Real.pi / 2)
  (hB : 0 < B ∧ B < Real.pi / 2)
  (hC : 0 < C ∧ C < Real.pi / 2)
  (habc : a < b + c ∧ b < a + c ∧ c < a + b)
  (hA' : f A = 1)
  (ha_sine : a = 2 * Real.sin A) :
  A = Real.pi / 3 ∧ 3 < c + b ∧ c + b ≤ 2 * Real.sqrt 3 := by
  sorry

end acute_triangle_problem_l245_245722


namespace sum_distances_in_tetrahedron_l245_245443

-- Definitions of equilateral triangle properties
def equilateral_triangle := Type
variables {T : equilateral_triangle}
def height_of_equilateral_triangle (T : equilateral_triangle) : ℝ := sorry
def sum_distances_to_sides (P : T) (sides : set T) : ℝ := sorry

-- Definitions of regular tetrahedron properties
def regular_tetrahedron := Type
variables {Tetra : regular_tetrahedron}
def height_of_regular_tetrahedron (Tetra : regular_tetrahedron) : ℝ := sorry
def sum_distances_to_faces (P : Tetra) (faces : set Tetra) : ℝ := sorry

theorem sum_distances_in_tetrahedron (tetra : regular_tetrahedron)
  (h_eq_tri : ∀ (tri : equilateral_triangle) (point : tri), 
    sum_distances_to_sides point tri = height_of_equilateral_triangle tri)
  (point : tetra) :
  sum_distances_to_faces point tetra = height_of_regular_tetrahedron tetra := sorry

end sum_distances_in_tetrahedron_l245_245443


namespace students_in_canteen_l245_245084

theorem students_in_canteen (total_students : ℕ) (absent_fraction present_fraction : ℚ) 
  (h_absent_fraction : absent_fraction = 1/10)
  (h_present_fraction : present_fraction = 3/4)
  (h_total_students : total_students = 40) :
  let absent_students := (absent_fraction * total_students : ℚ).toNat,
      present_students := total_students - absent_students,
      classroom_students := (present_fraction * present_students : ℚ).toNat,
      canteen_students := present_students - classroom_students
  in canteen_students = 9 := by
  sorry

end students_in_canteen_l245_245084


namespace probability_of_rectangle_area_l245_245034

theorem probability_of_rectangle_area (AC CB : ℝ) (h1 : 0 ≤ AC) (h2 : AC ≤ 10) (h3 : CB = 10 - AC) : 
  let event := {x : ℝ | 1 ≤ x ∧ x ≤ 9} in
  let total := {x : ℝ | 0 ≤ x ∧ x ≤ 10} in
  (set.finite (λ x ∈ event, true) / set.finite (λ x ∈ total, true)) = 4 / 5 := 
by sorry

end probability_of_rectangle_area_l245_245034


namespace parallelogram_length_TZ_proof_l245_245014

/-- 
Given a parallelogram WXYZ with properties: 
  ∠WXY = 100° 
  WY = 20 
  XZ = 15 
  Extend ZY through Y to V so that YV = 6 
  Line XV intersects WZ at T.
Prove that the length of TZ is 4.3 
  -/
theorem parallelogram_length_TZ_proof (W X Y Z V T : Point) 
  (angle_WXY : Angle)
  (WY XZ YV TZ : Length) 
  (parallelogram_WXYZ : Parallelogram W X Y Z)
  (h_angle : angle_WXY = 100)
  (h_WY : WY = 20)
  (h_XZ : XZ = 15)
  (extend_ZY_V : Extend_Line_through Y Z V)
  (h_YV : YV = 6)
  (line_intersects : Intersects XV WZ T) :
  TZ = 4.3 :=
sorry

end parallelogram_length_TZ_proof_l245_245014


namespace lisa_total_spoons_l245_245845

def num_children := 4
def spoons_per_child := 3
def decorative_spoons := 2
def large_spoons := 10
def teaspoons := 15

def total_spoons := num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

theorem lisa_total_spoons : total_spoons = 39 := by
  sorry

end lisa_total_spoons_l245_245845


namespace angle_AOB_is_pi_div_4_l245_245890

noncomputable def A : ℂ := 2 + I
noncomputable def B : ℂ := 10 / (3 + I)
def angle_AOB : ℝ := Real.arctan (1 / 2) - Real.arctan (-2)

theorem angle_AOB_is_pi_div_4 : angle_AOB = π / 4 := sorry

end angle_AOB_is_pi_div_4_l245_245890


namespace inequality_proof_l245_245987

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l245_245987


namespace jenny_hours_left_l245_245354

theorem jenny_hours_left
  (hours_research : ℕ)
  (hours_proposal : ℕ)
  (hours_total : ℕ)
  (h1 : hours_research = 10)
  (h2 : hours_proposal = 2)
  (h3 : hours_total = 20) :
  (hours_total - (hours_research + hours_proposal) = 8) :=
by
  sorry

end jenny_hours_left_l245_245354


namespace largest_gcd_for_13_integers_sum_2142_l245_245933

theorem largest_gcd_for_13_integers_sum_2142 :
  ∃ (S : Finset ℕ) (d : ℕ), 
  (S.card = 13 ∧ S.sum id = 2142 ∧ S.pairwise (λ a b, Nat.coprime a b) ∧ 
  ∀ x ∈ S, ∃ k : ℕ, x = d * k ∧ Nat.gcd k d = 1) 
  → d = 21 :=
by
  sorry

end largest_gcd_for_13_integers_sum_2142_l245_245933


namespace josh_path_count_l245_245379

theorem josh_path_count (n : ℕ) : 
  let count_ways (steps: ℕ) := 2^steps in
  count_ways (n-1) = 2^(n-1) :=
by
  sorry

end josh_path_count_l245_245379


namespace club_officer_selection_count_l245_245869

theorem club_officer_selection_count (n : ℕ) (h : n = 12) : 
  (∏ i in (finset.range 5), (n - i)) = 95040 :=
by
  rw h
  have : (finset.range 5) = {0, 1, 2, 3, 4} := rfl
  rw this
  simp
  norm_num
  sorry

end club_officer_selection_count_l245_245869


namespace unusual_digits_exists_l245_245558

def is_unusual (n : ℕ) : Prop :=
  let len := n.digits.count;
  let high_power := 10 ^ len;
  (n^3 % high_power = n) ∧ (n^2 % high_power ≠ n)

theorem unusual_digits_exists :
  ∃ n1 n2 : ℕ, (n1 ≥ 10^99 ∧ n1 < 10^100 ∧ is_unusual n1) ∧ 
             (n2 ≥ 10^99 ∧ n2 < 10^100 ∧ is_unusual n2) ∧
             (n1 ≠ n2) :=
by
  let n1 := 10^100 - 1;
  let n2 := (10^100 / 2) - 1;
  use n1, n2;
  sorry

end unusual_digits_exists_l245_245558


namespace jenny_hours_left_l245_245355

theorem jenny_hours_left
  (hours_research : ℕ)
  (hours_proposal : ℕ)
  (hours_total : ℕ)
  (h1 : hours_research = 10)
  (h2 : hours_proposal = 2)
  (h3 : hours_total = 20) :
  (hours_total - (hours_research + hours_proposal) = 8) :=
by
  sorry

end jenny_hours_left_l245_245355


namespace complementary_angle_decrease_l245_245462

theorem complementary_angle_decrease (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90)
  (h2 : angle1 / 3 = angle2 / 6) (h3 : ∃ x: ℝ, x = 0.2) :
  let new_angle1 := angle1 * 1.2 in 
  let new_angle2 := 90 - new_angle1 in
  (new_angle2 - angle2) / angle2 = -0.10 := sorry

end complementary_angle_decrease_l245_245462


namespace bella_steps_to_meet_ella_l245_245185

-- Given conditions
variables (b t : ℝ) (house_distance : ℝ := 15840) (bella_step_length : ℝ := 2)

-- Condition: Bella and Ella's initial speeds and the time delay
variables (bella_speed : ℝ := b) (ella_speed : ℝ := 5 * b) (time_delay : ℝ := 5)

-- Calculations based on the given conditions
-- Distance Bella covers in 5 minutes
def initial_distance_covered := time_delay * bella_speed

-- Remaining distance they need to cover together
def remaining_distance := house_distance - initial_distance_covered

-- Time taken for Bella and Ella to meet after the 5-minute delay
def time_to_meet := remaining_distance / (bella_speed + ella_speed)

-- Total time Bella walks
def total_time_bella_walks := time_delay + time_to_meet

-- Distance Bella covers in total
def total_distance_bella_covers := bella_speed * total_time_bella_walks

-- Number of steps Bella will take
def steps_bella_takes := total_distance_bella_covers / bella_step_length

-- Actual proof statement
theorem bella_steps_to_meet_ella (h : b = 528) : steps_bella_takes b = 2112 :=
by
  sorry

end bella_steps_to_meet_ella_l245_245185


namespace exponents_eq_sum_l245_245754

theorem exponents_eq_sum (x y : ℝ) (h : 5^(x+1) * 4^(y-1) = 25^x * 64^y) : x + y = 1 / 2 := by
  sorry

end exponents_eq_sum_l245_245754


namespace angleKMT_is_120_degrees_l245_245416

def rightTriangle (A B C : Type) (angleBCA : ℝ) [inhabited A] [inhabited B] [inhabited C] := angleBCA = 90

def externalRightTriangles (A B C T K : Type) 
  (angleATB : ℝ) (angleAKC : ℝ) (angleABT : ℝ) (angleACK : ℝ)
  [inhabited A] [inhabited B] [inhabited C] 
  [inhabited T] [inhabited K] :=
  angleATB = 90 ∧ angleAKC = 90 ∧ angleABT = 60 ∧ angleACK = 60

def pointM (B C M : Type) (BM CM : ℝ) [inhabited B] [inhabited C] [inhabited M] := BM = CM

theorem angleKMT_is_120_degrees
  (A B C T K M : Type) 
  [inhabited A] [inhabited B] [inhabited C] 
  [inhabited T] [inhabited K] [inhabited M] 
  (angleBCA : ℝ) (angleATB : ℝ) (angleAKC : ℝ) (angleABT : ℝ) (angleACK : ℝ)
  (BM CM : ℝ):
  rightTriangle A B C angleBCA →
  externalRightTriangles A B C T K angleATB angleAKC angleABT angleACK →
  pointM B C M BM CM →
  ∠KMT M T K = 120 :=
by
  sorry

end angleKMT_is_120_degrees_l245_245416


namespace find_original_price_l245_245959

variable (P : ℝ) (reduced_price : ℝ) (original_qty : ℝ) (new_qty : ℝ)

def original_price (P : ℝ) : Prop :=
  -- condition: reduced price is 15/16 of the original price
  let reduced_price := (15 / 16) * P in
  -- condition: original quantity bought for Rs. 120
  let original_qty := 120 / P in
  -- condition: new quantity bought for Rs. 120 after price reduction
  let new_qty := original_qty + 1 in
  -- equation relating original and new quantities for Rs. 120
  (new_qty * reduced_price = 120) ∧ (P = 8)

theorem find_original_price (h : original_price P) : P = 8 := by
  sorry

end find_original_price_l245_245959


namespace sum_of_roots_l245_245237

theorem sum_of_roots (p : Polynomial ℝ) (h : p = 3 * (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 - Polynomial.C 16 * Polynomial.X - Polynomial.C 4)) :
  p.roots.Sum = 3 :=
by 
  sorry

end sum_of_roots_l245_245237


namespace problem1_problem2_problem3_l245_245188

-- Definition and proofs to verify the results
theorem problem1 : 0.78 * 7 - 39 / 50 + 4 * (39 / 50) = 7.8 :=
by sorry

theorem problem2 : 12.5 * 8 / (12.5 * 8) = 64 :=
by sorry

theorem problem3 :
  let term := λ n : Nat, (88 - n * 10 - 1/8) * 1/8
  in (term 0) + (term 1) + (term 2) + (term 3) + (term 4) + (term 5) + (term 6) + (term 7) = 52 + 7/8 :=
by sorry

end problem1_problem2_problem3_l245_245188


namespace inequality_ABC_l245_245992

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_ABC_l245_245992


namespace sum_sin_squared_eq_15_l245_245629

noncomputable def sum_sin_squared : ℝ :=
  ∑ k in finset.range 30, real.sin (6 * k * real.pi / 180) ^ 2

theorem sum_sin_squared_eq_15 : sum_sin_squared = 15 :=
sorry

end sum_sin_squared_eq_15_l245_245629


namespace m_plus_n_eq_neg_one_l245_245751

theorem m_plus_n_eq_neg_one (m n : ℝ) (h : sqrt (m - 2) + (n + 3) ^ 2 = 0) : m + n = -1 := 
sorry

end m_plus_n_eq_neg_one_l245_245751


namespace product_of_all_n_satisfying_quadratic_l245_245657

theorem product_of_all_n_satisfying_quadratic :
  (∃ n : ℕ, n^2 - 40 * n + 399 = 3) ∧
  (∀ p : ℕ, Prime p → ((∃ n : ℕ, n^2 - 40 * n + 399 = p) → p = 3)) →
  ∃ n1 n2 : ℕ, (n1^2 - 40 * n1 + 399 = 3) ∧ (n2^2 - 40 * n2 + 399 = 3) ∧ n1 ≠ n2 ∧ (n1 * n2 = 396) :=
by
  sorry

end product_of_all_n_satisfying_quadratic_l245_245657


namespace problem1_problem2_l245_245537

-- Define the sets of balls and boxes
inductive Ball
| ball1 | ball2 | ball3 | ball4

inductive Box
| boxA | boxB | boxC

-- Define the arrangements for the first problem
def arrangements_condition1 (arrangement : Ball → Box) : Prop :=
  (arrangement Ball.ball3 = Box.boxB) ∧
  (∃ b1 b2 b3 : Box, b1 ≠ b2 ∧ b2 ≠ b3 ∧ b3 ≠ b1 ∧ 
    ∃ (f : Ball → Box), 
      (f Ball.ball1 = b1) ∧ (f Ball.ball2 = b2) ∧ (f Ball.ball3 = Box.boxB) ∧ (f Ball.ball4 = b3))

-- Define the proof statement for the first problem
theorem problem1 : ∃ n : ℕ, (∀ arrangement : Ball → Box, arrangements_condition1 arrangement → n = 7) :=
sorry

-- Define the arrangements for the second problem
def arrangements_condition2 (arrangement : Ball → Box) : Prop :=
  (arrangement Ball.ball1 ≠ Box.boxA) ∧
  (arrangement Ball.ball2 ≠ Box.boxB)

-- Define the proof statement for the second problem
theorem problem2 : ∃ n : ℕ, (∀ arrangement : Ball → Box, arrangements_condition2 arrangement → n = 36) :=
sorry

end problem1_problem2_l245_245537


namespace eccentricity_of_hyperbola_l245_245576

variables (a b : ℝ) (ha : a > 0) (hb : b > 0)

def parabola := ∀ x y : ℝ, y^2 = 8 * x

def hyperbola := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def asymptote_distance := ∃ f_x f_y a_vec, 
  (f_x, f_y) = (4, 0) ∧ -- Focus of the parabola
  ∀ k : ℝ, a_vec = (1, k) ∧
  sqrt (a_vec.1^2 - a_vec.2^2) = √3

theorem eccentricity_of_hyperbola (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_parabola : parabola) 
  (h_hyperbola : hyperbola) 
  (h_distance : asymptote_distance) : 
  let c := sqrt (a^2 + b^2) in
  (c / a) = 2 := 
sorry

end eccentricity_of_hyperbola_l245_245576


namespace circle_equation_l245_245679

theorem circle_equation 
  (C : ℝ × ℝ)
  (hC1 : 2 * C.1 + 3 * C.2 + 1 = 0)
  (hCenter_eq : ∃ C : ℝ × ℝ, 2 * C.1 + 3 * C.2 + 1 = 0 ∧ (C.1 - 4)^2 + (C.2 + 3)^2 = 25)
  (hRadius : (4 - 0)^2 + (-3 - 0)^2 = 25)
  (hCond1 : (1 - 0)^2 + (1 - 0)^2 ≠ 0) :
  ∃ (center : ℝ × ℝ), (center.1 - 4)^2 + (center.2 + 3)^2 = 25 := 
by
sory ဆိုရီး

end circle_equation_l245_245679


namespace cone_aperture_angle_l245_245584

-- Definitions as conditions derived from problem statement
def sphere_inscribed_in_cone : Prop := -- precise mathematical definition omitted for brevity
sorry

def surface_area_relation (A_sphere A_cone : ℝ) : Prop :=
A_sphere = (2 / 3) * A_cone

def aperture_angle (θ : ℝ) : Prop :=
∃ (r a : ℝ), sphere_inscribed_in_cone ∧ surface_area_relation (4 * r^2 * π) (r * π * a) ∧ 
θ = 2 * arcsin (r / a)

-- Main theorem we aim to prove
theorem cone_aperture_angle :
  ∃ θ : ℝ, aperture_angle θ ∧ (θ = 60 ∨ θ = 38.93) :=
begin
  sorry
end

end cone_aperture_angle_l245_245584


namespace union_of_A_and_B_l245_245405

/-- Let the universal set U = ℝ, and let the sets A = {x | x^2 - x - 2 = 0}
and B = {y | ∃ x, x ∈ A ∧ y = x + 3}. We want to prove that A ∪ B = {-1, 2, 5}.
-/
theorem union_of_A_and_B (U : Set ℝ) (A B : Set ℝ) (A_def : ∀ x, x ∈ A ↔ x^2 - x - 2 = 0)
  (B_def : ∀ y, y ∈ B ↔ ∃ x, x ∈ A ∧ y = x + 3) :
  A ∪ B = {-1, 2, 5} :=
sorry

end union_of_A_and_B_l245_245405


namespace Xiaoli_can_finish_typing_l245_245577

theorem Xiaoli_can_finish_typing :
  52 * 6 ≥ 300 :=
by {
  calc
    52 * 6 = 312 : by norm_num
    ... ≥ 300 : by norm_num
}

end Xiaoli_can_finish_typing_l245_245577


namespace sin_cos_double_angle_identity_l245_245669

theorem sin_cos_double_angle_identity (α : ℝ) 
  (h1 : Real.sin α = 1/3) 
  (h2 : α ∈ Set.Ioc (π/2) π) : 
  Real.sin (2*α) + Real.cos (2*α) = (7 - 4 * Real.sqrt 2) / 9 := 
by
  sorry

end sin_cos_double_angle_identity_l245_245669


namespace circumscribed_sphere_radius_l245_245908

-- Define the conditions
variables (a : ℝ) (h₁ : a > 0) 

theorem circumscribed_sphere_radius (h_angle : ∀ {P A M : ℝ}, angle P A M = 60) :
  ∃ R, R = (2 * a / 3) :=
by sorry

end circumscribed_sphere_radius_l245_245908


namespace hemisphere_surface_area_l245_245915

-- Define the condition of the problem
def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2
def base_area_of_hemisphere : ℝ := 3

-- The proof problem statement
theorem hemisphere_surface_area : 
  ∃ (r : ℝ), (Real.pi * r^2 = 3) → (2 * Real.pi * r^2 + Real.pi * r^2 = 9) := 
by 
  sorry

end hemisphere_surface_area_l245_245915


namespace cosine_angle_eq_slope_midpoint_eq_l245_245263

-- Given conditions for the ellipse and the origin
def ellipse (x y : ℝ) : Prop := (y^2 / 9) + (x^2 / 8) = 1
def origin : ℝ × ℝ := (0, 0)

-- Declaration for Part (1)
theorem cosine_angle_eq (P F1 F2 : ℝ × ℝ) (hP : ellipse P.1 P.2) (h_eq : dist P F1 = dist P F2) :
  cos (angle F1 P F2) = 7 / 9 := 
sorry

-- Declaration for Part (2)
theorem slope_midpoint_eq (A B : ℝ × ℝ) (M : ℝ × ℝ := midpoint A B)
  (hA : ellipse A.1 A.2) (hB : ellipse B.1 B.2) 
  (l : line := { x - y + 1 = 0 }) 
  (h_l : intersects l C A B M) :
  slope (line_through origin M) = -9 / 8 := 
sorry

end cosine_angle_eq_slope_midpoint_eq_l245_245263


namespace largest_number_using_three_3s_and_three_8s_l245_245114

theorem largest_number_using_three_3s_and_three_8s :
  ∃ L : ℕ, L = 3 ^ (3 ^ (3 ^ (8 ^ (8 ^ 8)))) ∧ 
  (∀ n, n ∈ {e | ∃ (a b c d e f : ℕ),
     a + b + c + d + e + f = 6 ∧
     (a = 3 ∨ a = 8) ∧ 
     (b = 3 ∨ b = 8) ∧ 
     (c = 3 ∨ c = 8) ∧ 
     (d = 3 ∨ d = 8) ∧ 
     (e = 3 ∨ e = 8) ∧ 
     (f = 3 ∨ f = 8) ∧ 
     e = some_expression_using_operations_plus_minus_times_div_pow a b c d e f} → n ≤ L) := sorry

end largest_number_using_three_3s_and_three_8s_l245_245114


namespace proof_y_value_l245_245224

noncomputable def y_value {A B C D O : Type} [metric_space O] [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  (AO BO CO DO BD AC : ℝ) (angle_AOC_eq_angle_BOD : angle A O C = angle B O D) : ℝ :=
  let φ := angle A O C in
  have cos_φ : real.cos φ = -2/15, from by sorry,
  let y := real.sqrt (5^2 + 12^2 - 2 * 5 * 12 * cos_φ) in
  y

theorem proof_y_value {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space ℝ]
  (AO BO CO DO BD : ℝ) (angle_AOC_eq_angle_BOD : angle A O C = angle B O D) (h_AO : AO = 5) (h_BO : BO = 3) (h_CO : CO = 12) (h_DO : DO = 5) (h_BD : BD = 6) :
  let y := y_value AO BO CO DO BD 0 angle_AOC_eq_angle_BOD in
  y = 3 * real.sqrt 67 :=
begin
  sorry
end

end proof_y_value_l245_245224


namespace josh_path_count_l245_245382

theorem josh_path_count (n : ℕ) : 
  let count_ways (steps: ℕ) := 2^steps in
  count_ways (n-1) = 2^(n-1) :=
by
  sorry

end josh_path_count_l245_245382


namespace students_taking_geometry_or_history_not_both_l245_245491

theorem students_taking_geometry_or_history_not_both :
  ∀ (students_both : ℕ) (students_geometry : ℕ) (students_history_only : ℕ), 
  students_both = 20 →
  students_geometry = 40 →
  students_history_only = 15 →
  (students_geometry - students_both) + students_history_only = 35 :=
by
  intros students_both students_geometry students_history_only h_both h_geom h_hist
  simp [h_both, h_geom, h_hist]
  sorry

end students_taking_geometry_or_history_not_both_l245_245491


namespace total_yield_l245_245788

-- Definitions based on conditions
def initial_yield : ℕ := 20
def growth_rate_first : ℝ := 0.15
def growth_rate_second : ℝ := 0.20
def diminishing_rate : ℝ := 0.03
def loss_rate : ℝ := 0.05

-- Harvest times (days)
def harvest_time_total : ℕ := 60
def first_harvest_time : ℕ := 15
def second_harvest_time : ℕ := 20
def third_harvest_time : ℕ := 25

-- Helper function to calculate the yield after growth and loss
def calculate_yield (previous_yield : ℕ) (growth_rate : ℝ) : ℕ :=
  let increased_yield := previous_yield * (1 + growth_rate)
  let total_yield := increased_yield * (1 - loss_rate)
  total_yield.to_int

theorem total_yield : initial_yield + ⌊initial_yield * (1 + growth_rate_first) * (1 - loss_rate)⌋.floor +
  ⌊(⌊initial_yield * (1 + growth_rate_first) * (1 - loss_rate)⌋.floor * (1 + growth_rate_second) * (1 - loss_rate))⌋.floor +
  ⌊(⌊(⌊initial_yield * (1 + growth_rate_first) * (1 - loss_rate)⌋.floor * (1 + growth_rate_second) * (1 - loss_rate))⌋.floor *
  (1 + growth_rate_second - diminishing_rate) * (1 - loss_rate))⌋.floor = 69 := 
by {
  sorry
}

end total_yield_l245_245788


namespace greatest_two_values_l245_245664

theorem greatest_two_values :
  let a := 16^(1/4)
  let b := 27^(1/3)
  let c := 25^(1/2)
  let d := 32^(1/5)
  (a = 2) ∧ (b = 3) ∧ (c = 5) ∧ (d = 2) →
  (c > b) ∧ (b > a) ∧ (b > d) → 
  (∀ x ∈ {a, b, c, d}, x ≤ c) ∧ (∀ x ∈ {a, b, d}, x ≤ b) :=
by
  intros ha hb hc hd h0 h1;
  split;
  · intro;
    finish
  · split;
    intro;
    finish

end greatest_two_values_l245_245664


namespace inequality_proof_l245_245979

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end inequality_proof_l245_245979


namespace cosine_angle_eq_slope_midpoint_eq_l245_245264

-- Given conditions for the ellipse and the origin
def ellipse (x y : ℝ) : Prop := (y^2 / 9) + (x^2 / 8) = 1
def origin : ℝ × ℝ := (0, 0)

-- Declaration for Part (1)
theorem cosine_angle_eq (P F1 F2 : ℝ × ℝ) (hP : ellipse P.1 P.2) (h_eq : dist P F1 = dist P F2) :
  cos (angle F1 P F2) = 7 / 9 := 
sorry

-- Declaration for Part (2)
theorem slope_midpoint_eq (A B : ℝ × ℝ) (M : ℝ × ℝ := midpoint A B)
  (hA : ellipse A.1 A.2) (hB : ellipse B.1 B.2) 
  (l : line := { x - y + 1 = 0 }) 
  (h_l : intersects l C A B M) :
  slope (line_through origin M) = -9 / 8 := 
sorry

end cosine_angle_eq_slope_midpoint_eq_l245_245264


namespace product_has_trailing_zeros_l245_245903

theorem product_has_trailing_zeros (a b : ℕ) (h1 : a = 350) (h2 : b = 60) :
  ∃ (n : ℕ), (10^n ∣ a * b) ∧ n = 3 :=
by
  sorry

end product_has_trailing_zeros_l245_245903


namespace bead_no_full_rotation_l245_245141

def bead_exists_with_full_rotation : Prop :=
  ∃ (positions : ℕ → ℝ) (moves : list ℕ), 
  (∀ n, positions (n + 2009) = positions n + 2 * Float.pi) →
  (∀ n, moves.contains n → positions n = (positions (n - 1) + positions (n + 1)) / 2) →
  false

theorem bead_no_full_rotation : ¬ bead_exists_with_full_rotation := 
  by
  sorry

end bead_no_full_rotation_l245_245141


namespace smallest_positive_difference_l245_245499

-- Define the method used by Vovochka to sum numbers.
def sum_without_carrying (a b : ℕ) : ℕ :=
  let ha := a / 100 
  let ta := (a % 100) / 10 
  let ua := a % 10
  let hb := b / 100 
  let tb := (b % 100) / 10 
  let ub := b % 10
  (ha + hb) * 1000 + (ta + tb) * 100 + (ua + ub)

-- Define the correct method to sum numbers.
def correct_sum (a b : ℕ) : ℕ :=
  a + b

-- The smallest positive difference between the sum without carrying and the correct sum.
theorem smallest_positive_difference (a b : ℕ) (h : 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000) :
  (sum_without_carrying a b - correct_sum a b > 0) →
  (∀ c d, 100 ≤ c ∧ c < 1000 ∧ 100 ≤ d ∧ d < 1000 →
    (sum_without_carrying c d - correct_sum c d > 0 → sum_without_carrying c d - correct_sum c d ≥ 1800 )) :=
begin
  sorry
end

end smallest_positive_difference_l245_245499


namespace pipes_fill_together_in_40_minutes_l245_245417

def faster_pipe_time (slower_pipe_time : ℕ) : ℝ :=
  slower_pipe_time / 3

def combined_rate (faster_time slower_time : ℝ) : ℝ :=
  (1 / faster_time) + (1 / slower_time)

def total_time_to_fill_together (combined_rate : ℝ) : ℝ :=
  1 / combined_rate

theorem pipes_fill_together_in_40_minutes (slower_pipe_time : ℝ) (h : slower_pipe_time = 160) :
  total_time_to_fill_together (combined_rate (faster_pipe_time slower_pipe_time) slower_pipe_time) = 40 := by
  rw [h, faster_pipe_time, combined_rate, total_time_to_fill_together]
  norm_num
  sorry

end pipes_fill_together_in_40_minutes_l245_245417


namespace tom_original_portion_l245_245930

-- Variables representing the original amounts Tom, Uma, and Vicky have
variables (t u v : ℝ)

-- Conditions given in the problem
axiom (c1 : t + u + v = 2000)
axiom (c2 : t - 200 + 3 * u + 3 * v = 3500)

-- The theorem stating the goal to prove
theorem tom_original_portion : t = 1150 :=
by {
  sorry
}

end tom_original_portion_l245_245930


namespace ratio_of_areas_l245_245876

-- Definition of sides and given condition
variables {a b c d : ℝ}
-- Given condition in the problem.
axiom condition : a / c = 3 / 5 ∧ b / d = 3 / 5

-- Statement of the theorem to be proved in Lean 4
theorem ratio_of_areas (h : a / c = 3 / 5) (h' : b / d = 3 / 5) : (a * b) / (c * d) = 9 / 25 :=
by sorry

end ratio_of_areas_l245_245876


namespace distinct_terms_count_l245_245204

theorem distinct_terms_count
  (x y z w p q r s t : Prop)
  (h1 : ¬(x = y ∨ x = z ∨ x = w ∨ y = z ∨ y = w ∨ z = w))
  (h2 : ¬(p = q ∨ p = r ∨ p = s ∨ p = t ∨ q = r ∨ q = s ∨ q = t ∨ r = s ∨ r = t ∨ s = t)) :
  ∃ (n : ℕ), n = 20 := by
  sorry

end distinct_terms_count_l245_245204


namespace fraction_shaded_l245_245583

theorem fraction_shaded (s r : ℝ) (h : s^2 = 3 * r^2) :
    (1/2 * π * r^2) / (1/4 * π * s^2) = 2/3 := 
  sorry

end fraction_shaded_l245_245583


namespace evaluate_expression_l245_245119

theorem evaluate_expression (a b : ℝ) (h1 : a = 4) (h2 : b = -1) : -2 * a ^ 2 - 3 * b ^ 2 + 2 * a * b = -43 :=
by
  sorry

end evaluate_expression_l245_245119


namespace min_weighings_to_order_four_stones_l245_245605

theorem min_weighings_to_order_four_stones : ∀ (A B C D : ℝ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D → ∃ n, n = 5 :=
by sorry

end min_weighings_to_order_four_stones_l245_245605


namespace josh_walk_ways_l245_245367

theorem josh_walk_ways (n : ℕ) :
  let grid_rows := n
      grid_columns := 3
      start_position := (0, 0)  -- (row, column) starting from bottom left
  in grid_rows > 0 →
      let center_square (k : ℕ) := (k, 1) -- center square of k-th row
  in ∃ ways : ℕ, ways = 2^(n-1) ∧
                ways = count_paths_to_center_topmost n  -- custom function representation
sorry

end josh_walk_ways_l245_245367


namespace complex_third_quadrant_l245_245675

-- Define the imaginary unit i.
def i : ℂ := Complex.I 

-- Define the complex number z = i * (1 + i).
def z : ℂ := i * (1 + i)

-- Prove that z lies in the third quadrant.
theorem complex_third_quadrant : z.re < 0 ∧ z.im < 0 := 
by
  sorry

end complex_third_quadrant_l245_245675


namespace rth_term_arithmetic_progression_l245_245663

-- Define the sum of the first n terms of the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^3

-- Define the r-th term of the arithmetic progression
def a (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem stating the r-th term of the arithmetic progression
theorem rth_term_arithmetic_progression (r : ℕ) : a r = 12 * r^2 - 12 * r + 9 := by
  sorry

end rth_term_arithmetic_progression_l245_245663


namespace intersection_angle_range_l245_245321

theorem intersection_angle_range (k b : Real)
  (hk : k > -2/3)
  (hL1L2_inter : ∃ M : Real × Real,
    M.1 > 0 ∧ M.2 > 0 ∧ 
    M.2 = k * M.1 - b ∧ 
    2 * M.1 + 3 * M.2 - 6 = 0) :
  arctan (-2/3) < arctan k ∧ arctan k < π / 2 :=
by
  sorry

end intersection_angle_range_l245_245321


namespace smallest_sum_of_squares_l245_245071

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 145) : x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l245_245071


namespace ball_distance_l245_245540

/-- Define a ball with a diameter of 6 inches rolling along a path consisting of three semicircular arcs with specified radii. Prove that the distance the center of the ball travels along this path is 267π inches. -/
theorem ball_distance (d : ℝ) (R₁ R₂ R₃ : ℝ) (h₁ : d = 6) (h₂ : R₁ = 120) (h₃ : R₂ = 50) (h₄ : R₃ = 100) :
  let r := d / 2,
      arc₁ := R₁ - r,
      arc₂ := R₂ + r,
      arc₃ := R₃ - r in
  (1/2 * 2 * π * arc₁) + (1/2 * 2 * π * arc₂) + (1/2 * 2 * π * arc₃) = 267 * π :=
by {
  sorry
}

end ball_distance_l245_245540


namespace carols_remaining_miles_l245_245620

noncomputable def distance_to_college : ℝ := 220
noncomputable def car_efficiency : ℝ := 20
noncomputable def gas_tank_capacity : ℝ := 16

theorem carols_remaining_miles :
  let gallons_used := distance_to_college / car_efficiency in
  let gallons_remaining := gas_tank_capacity - gallons_used in
  let miles_remaining := gallons_remaining * car_efficiency in
  miles_remaining = 100 :=
by
  sorry

end carols_remaining_miles_l245_245620


namespace josh_paths_l245_245361

theorem josh_paths (n : ℕ) (h : n > 0) : 
  let start := (0, 0)
  let end := (n - 1, 1)
  -- the number of distinct paths from start to end is 2^(n-1)
  (if n = 1 then 1 else 2^(n-1)) = 2^(n-1) :=
by
  sorry

end josh_paths_l245_245361


namespace total_gas_needed_l245_245743

noncomputable def gas_needed : ℕ :=
  let large_lawn_cuts := 4 + 12 + 2 in  -- Calculating total cuts for the large lawn
  let small_lawn_cuts := 8 + 8 - 2 in  -- Calculating total cuts for the small lawn
  let large_lawn_gas := (large_lawn_cuts / 3) * 2 in  -- Gas for large lawn mower
  let small_lawn_gas := (small_lawn_cuts / 2) in  -- Gas for small lawn mower
  let trimmer_gas := (6 / 3) * 1 in  -- Gas for trimmer
  let leaf_blower_gas := 2 * 0.5 in  -- Gas for leaf blower
  large_lawn_gas + small_lawn_gas + trimmer_gas + leaf_blower_gas

theorem total_gas_needed : gas_needed = 22 := by
  sorry

end total_gas_needed_l245_245743


namespace interval_where_f_is_decreasing_l245_245718

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem interval_where_f_is_decreasing :
  { x : ℝ | f' x < 0 } = set.Iio 1 ∪ {1} :=
by
  sorry

end interval_where_f_is_decreasing_l245_245718


namespace range_of_a_max_value_of_z_l245_245841

variable (a b : ℝ)

-- Definition of the assumptions
def condition1 := (2 * a + b = 9)
def condition2 := (|9 - b| + |a| < 3)
def condition3 := (a > 0)
def condition4 := (b > 0)
def z := a^2 * b

-- Statement for problem (i)
theorem range_of_a (h1 : condition1 a b) (h2 : condition2 a b) : -1 < a ∧ a < 1 := sorry

-- Statement for problem (ii)
theorem max_value_of_z (h1 : condition1 a b) (h2 : condition3 a) (h3 : condition4 b) : 
  z a b = 27 := sorry

end range_of_a_max_value_of_z_l245_245841


namespace not_all_pairs_ab_minus_one_perfect_square_l245_245680

theorem not_all_pairs_ab_minus_one_perfect_square (d : ℕ) (hd : d ≠ 2 ∧ d ≠ 5 ∧ d ≠ 13) :
  ∃ a b ∈ ({2, 5, 13, d} : Finset ℕ), a ≠ b ∧ ¬ (∃ k : ℕ, a * b - 1 = k * k) :=
by
  sorry

end not_all_pairs_ab_minus_one_perfect_square_l245_245680


namespace range_of_a_l245_245397

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) (interval : set ℝ) : Prop :=
  ∀ x y ∈ interval, x < y → f y ≤ f x

theorem range_of_a {f : ℝ → ℝ} :
  (is_odd f) →
  (is_monotonically_decreasing f {x | -2 < x ∧ x < 2}) →
  (∀ a, a > 1/2 ∧ a < 5/2 → f(2-a) + f(2*a-3) < 0) :=
by
  sorry

end range_of_a_l245_245397


namespace solve_system1_solve_system2_l245_245435

-- Define System (1) and prove its solution
theorem solve_system1 (x y : ℝ) (h1 : x = 5 - y) (h2 : x - 3 * y = 1) : x = 4 ∧ y = 1 := by
  sorry

-- Define System (2) and prove its solution
theorem solve_system2 (x y : ℝ) (h1 : x - 2 * y = 6) (h2 : 2 * x + 3 * y = -2) : x = 2 ∧ y = -2 := by
  sorry

end solve_system1_solve_system2_l245_245435


namespace find_x_l245_245538

noncomputable def eq_num (x : ℝ) : Prop :=
  9 - 3 / (1 / 3) + x = 3

theorem find_x : ∃ x : ℝ, eq_num x ∧ x = 3 := 
by
  sorry

end find_x_l245_245538


namespace total_money_found_l245_245093

def value_of_quarters (n_quarters : ℕ) : ℝ := n_quarters * 0.25
def value_of_dimes (n_dimes : ℕ) : ℝ := n_dimes * 0.10
def value_of_nickels (n_nickels : ℕ) : ℝ := n_nickels * 0.05
def value_of_pennies (n_pennies : ℕ) : ℝ := n_pennies * 0.01

theorem total_money_found (n_quarters n_dimes n_nickels n_pennies : ℕ) :
  n_quarters = 10 →
  n_dimes = 3 →
  n_nickels = 4 →
  n_pennies = 200 →
  value_of_quarters n_quarters + value_of_dimes n_dimes + value_of_nickels n_nickels + value_of_pennies n_pennies = 5.00 := 
by
  intros h_quarters h_dimes h_nickels h_pennies
  sorry

end total_money_found_l245_245093


namespace simplify_expression_l245_245963

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end simplify_expression_l245_245963


namespace triangle_is_isosceles_l245_245771

theorem triangle_is_isosceles (A B C : ℝ)
  (h : Real.log (Real.sin A) - Real.log (Real.cos B) - Real.log (Real.sin C) = Real.log 2) :
  ∃ a b c : ℝ, a = b ∨ b = c ∨ a = c := 
sorry

end triangle_is_isosceles_l245_245771


namespace quadratic_root_q_value_l245_245704

noncomputable def quadratic_root_condition (p q : ℝ) : Prop :=
  ∃ (r : ℂ), r = 2 + 3 * Complex.I ∧ (r.conj = 2 - 3 * Complex.I) ∧ ∀ (x : ℂ),
  x^2 + ↑p * x + ↑q = 0 ↔ (x = r ∨ x = r.conj)

theorem quadratic_root_q_value (p q : ℝ) (h : quadratic_root_condition p q) : q = 13 :=
sorry

end quadratic_root_q_value_l245_245704


namespace percentage_decrease_in_larger_angle_l245_245470

-- Define the angles and conditions
def angles_complementary (A B : ℝ) : Prop := A + B = 90
def angle_ratio (A B : ℝ) : Prop := A / B = 1 / 2
def small_angle_increase (A A' : ℝ) : Prop := A' = A * 1.2
def large_angle_new (A' B' : ℝ) : Prop := A' + B' = 90

-- Prove percentage decrease in larger angle
theorem percentage_decrease_in_larger_angle (A B A' B' : ℝ) 
    (h1 : angles_complementary A B)
    (h2 : angle_ratio A B)
    (h3 : small_angle_increase A A')
    (h4 : large_angle_new A' B')
    : (B - B') / B * 100 = 10 :=
sorry

end percentage_decrease_in_larger_angle_l245_245470


namespace area_of_regular_inscribed_octagon_l245_245938

theorem area_of_regular_inscribed_octagon (r : ℝ) (h : π * r^2 = 400 * π) : 
  (8 * (1/2) * r^2 * sin(π/4)) = 800 * sqrt 2 :=
begin
  sorry
end

end area_of_regular_inscribed_octagon_l245_245938


namespace inequality_hold_l245_245971

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end inequality_hold_l245_245971


namespace value_of_a_l245_245719

def f (x : ℝ) : ℝ :=
if x < 0 then -2 / x else real.log x / real.log 2

theorem value_of_a (a : ℝ) (h : f a = 2) : a = -1 ∨ a = 4 := by
  sorry

end value_of_a_l245_245719


namespace determine_value_of_k_l245_245207

def param_line1 (r k : ℝ) : ℝ × ℝ × ℝ :=
  (2 + r, -1 - 2 * k * r, 3 + k * r)

def param_line2 (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + 3 * t, 2 - t, 1 + 2 * t)

noncomputable def k_coplanar_condition (k : ℝ) : Prop :=
  ∃ u v w : ℝ, u * (1 : ℝ × ℝ × ℝ) + v * (-2, 3, 3 - 2 * k) = w * (3, -1, 2)

theorem determine_value_of_k 
  (k : ℝ) : k_coplanar_condition k → k = 3 / 2 :=
sorry

end determine_value_of_k_l245_245207


namespace right_footed_throwers_count_l245_245870

theorem right_footed_throwers_count 
  (total_players : Nat)
  (thrower_percentage : Nat)
  (wide_receivers_percentage : Nat)
  (wide_receivers_to_right_footed_ratio : Nat)
  (right_footed_throwers_ratio : Nat)
  (team_consists_of : ∀{p : Nat}, p = total_players -> thrower_percentage = 60)
  (total_throwers : Nat)
  (throwers_are_wide_receivers_or_right_footed : total_throwers = (wide_receivers_percentage / (wide_receivers_to_right_footed_ratio + right_footed_throwers_ratio * 2)) * thrower_percentage)
  : Prop :=
  (total_players = 100) -> 
  (thrower_percentage = 60) -> 
  (wide_receivers_percentage + right_footed_throwers_ratio = 100/active thrower_percentage) -> 
  (total_throwers = total_players) -> 
  right_footed_throwers_ratio = 24

end right_footed_throwers_count_l245_245870


namespace integral_solution_l245_245136

noncomputable def integral_of_function := ∫ (x : ℝ in Set.Icc a b), (1 - 6 * x) * Real.exp (2 * x)

theorem integral_solution :
  ∀ a b : ℝ, ∫ x in Set.Icc a b, (1 - 6 * x) * Real.exp (2 * x) = ((2 - 3 * x) * Real.exp (2 * x) + C) :=
by
  exact sorry

end integral_solution_l245_245136


namespace students_distribution_l245_245217

theorem students_distribution (students villages : ℕ) (h_students : students = 4) (h_villages : villages = 3) :
  ∃ schemes : ℕ, schemes = 36 := 
sorry

end students_distribution_l245_245217


namespace inequality_hold_l245_245973

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end inequality_hold_l245_245973


namespace boxes_per_case_l245_245049

-- Define the conditions
def total_boxes : ℕ := 54
def total_cases : ℕ := 9

-- Define the result we want to prove
theorem boxes_per_case : total_boxes / total_cases = 6 := 
by sorry

end boxes_per_case_l245_245049


namespace inequality_ABC_l245_245993

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_ABC_l245_245993


namespace find_f_prime_at_1_l245_245725

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x * (f' 0) - 2 * Real.exp (2 * x)

theorem find_f_prime_at_1 : derivative f 1 = 9 - 4 * Real.exp (2) := by
  sorry

end find_f_prime_at_1_l245_245725


namespace probability_at_least_3_calls_in_4_minutes_l245_245219

/-
  Given conditions:
  - The switchboard receives, on average, 90 calls per hour.
  - The number of calls follows a Poisson distribution.
  - We are considering the number of calls received in a time interval of 4 minutes.
-/

def lambda : ℝ := (90 : ℝ) / 60 * 4

def poisson_dist := Poisson ℝ λ

def P_at_least_3_calls_in_4_minutes : ℝ :=
  1 - (poisson_dist.probability 0 + poisson_dist.probability 1 + poisson_dist.probability 2)

theorem probability_at_least_3_calls_in_4_minutes :
  P_at_least_3_calls_in_4_minutes = 0.938 :=
sorry

end probability_at_least_3_calls_in_4_minutes_l245_245219


namespace partial_fraction_sum_zero_l245_245616

theorem partial_fraction_sum_zero (A B C D E F : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_sum_zero_l245_245616


namespace stephen_hawking_philosophical_implications_l245_245447

/-- Stephen Hawking's statements -/
def stephen_hawking_statement_1 := "The universe was not created by God"
def stephen_hawking_statement_2 := "Modern science can explain the origin of the universe"

/-- Definitions implied by Hawking's statements -/
def unity_of_world_lies_in_materiality := "The unity of the world lies in its materiality"
def thought_and_existence_identical := "Thought and existence are identical"

/-- Combined implication of Stephen Hawking's statements -/
def correct_philosophical_implications := [unity_of_world_lies_in_materiality, thought_and_existence_identical]

/-- Theorem: The correct philosophical implications of Stephen Hawking's statements are ① and ②. -/
theorem stephen_hawking_philosophical_implications :
  (stephen_hawking_statement_1 = "The universe was not created by God") →
  (stephen_hawking_statement_2 = "Modern science can explain the origin of the universe") →
  correct_philosophical_implications = ["The unity of the world lies in its materiality", "Thought and existence are identical"] :=
by
  sorry

end stephen_hawking_philosophical_implications_l245_245447


namespace max_value_fraction_ratio_tangent_line_through_point_l245_245271

theorem max_value_fraction_ratio (x y : ℝ) (h : x^2 + y^2 - 2*x - 2 = 0) :
  ∃ (max_value : ℝ), max_value = 2 + sqrt 6 :=
sorry

theorem tangent_line_through_point (x y : ℝ) (h : x^2 + y^2 - 2*x - 2 = 0) :
  ∃ (m : ℝ), m ≠ 0 ∧ (x, y) = (0, sqrt 2) → ∀ (x' y' : ℝ), y' = m * x' + sqrt 2 → x' - sqrt 2 * y' + 2 = 0 :=
sorry

end max_value_fraction_ratio_tangent_line_through_point_l245_245271


namespace find_sum_m_n_d_l245_245332

noncomputable def area_of_region (r : ℝ) (chord_len : ℝ) (dist_center : ℝ) : ℝ :=
  let sector_area := (chord_len^2 / 2) * π
  let triangle_area := r^2 / 2
  sector_area - triangle_area

theorem find_sum_m_n_d : 
  ∀ (r chord_len dist_center : ℝ), 
  r = 45 → chord_len = 84 → dist_center = 15 →
  area_of_region r chord_len dist_center = 506.25 * π - 1012.5
  → let m := 506.25 in
     let n := 1012.5 in
     let d := 1 in
     m + n + d = 1519.75 :=
by intros r chord_len dist_center hr hchord hdist heq;
   repeat { rw heq };
   sorry -- Proof omitted

end find_sum_m_n_d_l245_245332


namespace lisa_need_add_pure_juice_l245_245852

theorem lisa_need_add_pure_juice
  (x : ℝ) 
  (total_volume : ℝ := 2)
  (initial_pure_juice_fraction : ℝ := 0.10)
  (desired_pure_juice_fraction : ℝ := 0.25) 
  (added_pure_juice : ℝ := x) 
  (initial_pure_juice_amount : ℝ := total_volume * initial_pure_juice_fraction)
  (final_pure_juice_amount : ℝ := initial_pure_juice_amount + added_pure_juice)
  (final_volume : ℝ := total_volume + added_pure_juice) :
  (final_pure_juice_amount / final_volume) = desired_pure_juice_fraction → x = 0.4 :=
by
  intro h
  sorry

end lisa_need_add_pure_juice_l245_245852


namespace collinear_K_L_M_l245_245686

variables {α : Type} [euclidean_geometry α]

/-- Given conditions -/
variables {A B C O H M U : α}
variables {K L : α}

-- Triangle ABC is acute and inscribed in the circle O
variable (h1 : acute_triangle_in_circle A B C O)

-- H is the orthocenter of triangle ABC
variable (h2 : orthocenter H A B C)

-- M is the midpoint of BC
variable (h3 : midpoint M B C)

-- U is a point on BC such that angle BAM equals angle CAU
variable (h4 : ∃ U, angle_eq (angle B A M) (angle C A U) ∧ point_on_line U B C)

-- K is the projection of H onto the tangent at A to the circle O
variable (h5 : projection K H (tangent_at A O))

-- L is the projection of H onto AU
variable (h6 : projection L H (line A U))

-- Prove that points K, L, and M are collinear
theorem collinear_K_L_M : collinear K L M :=
sorry

end collinear_K_L_M_l245_245686


namespace find_smaller_circle_radius_l245_245548

noncomputable def radius_of_smaller_circle (A₁ A₂ A1_plus_A2 : ℝ) (r_large_small r_radius : ℝ) : Prop :=
  r_large_small = 4 ∧ 
  2 * (A1_plus_A2) = A₁ + A1_plus_A2 ∧
  A1_plus_A2 = (real.pi * (2 * r_large_small)^2) ∧
  A₁ = real.pi * r_radius^2 ∧
  A₁ + A₂ = A1_plus_A2

theorem find_smaller_circle_radius {A₁ A₂ A1_plus_A2 : ℝ} (r_large_small r_radius : ℝ) 
  (h1 : radius_of_smaller_circle A₁ A₂ A1_plus_A2 r_large_small r_radius) : r_radius = 4 :=
begin
  sorry
end

end find_smaller_circle_radius_l245_245548


namespace intersection_of_sets_l245_245734

noncomputable def set_A := {x : ℝ | Real.log x ≥ 0}
noncomputable def set_B := {x : ℝ | x^2 < 9}

theorem intersection_of_sets :
  set_A ∩ set_B = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by {
  sorry
}

end intersection_of_sets_l245_245734


namespace minimum_accumulation_l245_245029

/-- Masha puts a bill of 50 or 100 rubles into her piggy bank every week. -/
def puts_bill (n : ℕ) : Prop := 
  n = 50 ∨ n = 100

/-- Every 4 weeks, she gives her sister the smallest denomination bill. -/
def gives_smallest_bill (bills : List ℕ) : ℕ := 
  bills.min'.get_or_else 0

/-- At the end of the year, she had given her sister 1250 rubles. -/
def total_given (years_weeks : ℕ) (initials : List ℕ -> List ℕ): ℕ := 
  (List.range (years_weeks / 4)).sumBy (λ i => gives_smallest_bill (initials (List.range (4 * i))))

theorem minimum_accumulation (initials : List ℕ -> List ℕ):
  (List.range 52).All (λ i, (initials [i]).All (puts_bill)) → 
  total_given 52 initials = 1250 → 
  (List.range 52).sumBy (λ i, sums [initials [i]]) - total_given 52 initials ≥ 3750 := 
  sorry

end minimum_accumulation_l245_245029


namespace coefficient_of_y_in_third_equation_l245_245290

variables (x y z : ℝ)

theorem coefficient_of_y_in_third_equation
  (h1 : 6 * x - 5 * y + 3 * z = 22)
  (h2 : 4 * x + 8 * y - 11 * z = 7)
  (h3 : 5 * x - y + 2 * z = 2)
  (h4 : x + y + z = 10) :
  ∃ (a : ℝ), 5 * x + a * y + 2 * z = 2 ∧ a = -1 :=
by {
  use -1,
  split,
  { exact h3, },
  { refl, }
}

end coefficient_of_y_in_third_equation_l245_245290


namespace problem1_range_of_f_problem2_range_of_m_l245_245291

noncomputable def f (x : ℝ) : ℝ :=
  (Real.log x / Real.log 2 - 2) * (Real.log x / Real.log 4 - 1/2)

theorem problem1_range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc 1 4 = Set.Icc (-1/8 : ℝ) 1 :=
sorry

theorem problem2_range_of_m :
  ∀ x, x ∈ Set.Icc 4 16 → f x > (m : ℝ) * (Real.log x / Real.log 4) ↔ m < 0 :=
sorry

end problem1_range_of_f_problem2_range_of_m_l245_245291


namespace rabbit_fur_genetics_l245_245795

theorem rabbit_fur_genetics :
  ∀ (X Y : ℕ) (Z : String),
  (
    (dominant_gene : String),
    (recessive_gene : String),
    (heterozygous : Bool),
    (total_genes : ℕ),
    (genes_per_oocyte : ℕ),
    (types_of_nucleotides : ℕ),
    (meiotic_phase : String)
  ),
  dominant_gene = "long_fur" ∧
  recessive_gene = "short_fur" ∧
  heterozygous = true ∧
  total_genes = 20 ∧
  genes_per_oocyte = 4 ∧
  types_of_nucleotides = 4 ∧
  meiotic_phase = "late in the first meiotic division" →
  X = 5 ∧ 
  Y = 4 ∧ 
  Z = "late in the first meiotic division"
:= sorry

end rabbit_fur_genetics_l245_245795


namespace correct_option_l245_245523

universe u
variable {a m b : Type u} [CommRing a] [CommRing m] [CommRing b]

def option_A (a : ℕ) : Prop := 2 * a^2 + a^3 = 3 * a^5
def option_B (a : ℕ) : Prop := a^3 / a = a
def option_C (m : ℕ) : Prop := (-m^2)^3 = -m^6
def option_D (a b : ℕ) : Prop := (-2 * a * b)^2 = 4 * a^2 * b^2

theorem correct_option : option_D a b :=
sorry

end correct_option_l245_245523


namespace certain_number_l245_245313

theorem certain_number (x : ℝ) (h : x = 0.1) : 0.12 / x * 2 = 2.4 :=
by 
  rw [h]
  norm_num
  sorry

end certain_number_l245_245313


namespace student_matches_teacher_sequence_l245_245167

-- The permutation of the numbers 1 to 30
def permutation (l : List ℕ) : Prop := l.perm (List.range 30).map (λ x, x + 1)

-- definition to calculate points
def points (teacher : List ℕ) (student : List ℕ) : ℕ :=
  List.foldl (λ acc i, if teacher.nth i = student.nth i then acc + 1 else acc) 0 (List.range 30)

-- the main theorem
theorem student_matches_teacher_sequence :
  ∀ (teacher : List ℕ) (students : List (List ℕ)),
    permutation teacher →
    (∀ s, s ∈ students → permutation s) →
    (∀ i j, i ≠ j → points teacher (students.nthLe i sorry) ≠ points teacher (students.nthLe j sorry)) →
    ∃ s, s ∈ students ∧ points teacher s = 30 :=
by
  intro teacher students h_teacher h_students h_unique
  sorry

end student_matches_teacher_sequence_l245_245167


namespace angle_BXY_42_l245_245344

theorem angle_BXY_42 (AB CD : Line) (AXE CYX : ℝ)
    (h_parallel : Parallel AB CD)
    (h_angle : AXE = 4 * CYX - 126) :
  CYX = 42 :=
by
  sorry

end angle_BXY_42_l245_245344


namespace problem_l245_245692

open Real

def p (x : ℝ) : Prop := 2*x^2 + 2*x + 1/2 < 0

def q (x y : ℝ) : Prop := (x^2)/4 - (y^2)/12 = 1 ∧ x ≥ 2

def x0_condition (x0 : ℝ) : Prop := sin x0 - cos x0 = sqrt 2

theorem problem (h1 : ∀ x : ℝ, ¬ p x)
               (h2 : ∃ x y : ℝ, q x y)
               (h3 : ∃ x0 : ℝ, x0_condition x0) :
               ∀ x : ℝ, ¬ ¬ p x := 
sorry

end problem_l245_245692


namespace percentage_flags_both_colors_l245_245128

variable (F : ℕ) (h_even : Even F) (C : ℕ) (h_children : C = F / 2)
variable (h_all_used : F = 2 * C)
variable (h_blue : 0.6 * C) (h_red : 0.6 * C)

theorem percentage_flags_both_colors (F : ℕ) (h_even : Even F) (C : ℕ) (h_children : C = F / 2)
  (h_all_used : F = 2 * C) (h_blue : 0.6 * C) (h_red : 0.6 * C) : 
  ∃ X, X = 20 / 100 * C := 
by
  sorry

end percentage_flags_both_colors_l245_245128


namespace delegates_seating_probability_l245_245440

open classical
noncomputable def factorial : ℤ → ℤ
| 0       := 1
| (nat.succ n) := (n+1) * factorial n

def configurations_count (total_seats : ℤ) (delegates_each_country : ℤ) : ℤ :=
  (factorial total_seats) / (2 ^ delegates_each_country)

theorem delegates_seating_probability (m n : ℤ) (rel_prime : m.gcd n = 1) :
  m = 14968800 ∧ n = 32  →
  ∃ (P : ℚ), P = (m / n) ∧ m + n = 14968833 :=
by
  sorry

end delegates_seating_probability_l245_245440


namespace smallest_sum_of_squares_l245_245072

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 145) : x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l245_245072


namespace find_g_2002_l245_245284

variable {f : ℝ → ℝ}

noncomputable def g (x : ℝ) : ℝ := f(x) + 1 - x

theorem find_g_2002 (h1 : f(1) = 1)
                    (h2 : ∀ x : ℝ, f(x + 5) ≥ f(x) + 5)
                    (h3 : ∀ x : ℝ, f(x + 1) ≤ f(x) + 1) :
                    g 2002 = 1 :=
by
  sorry

end find_g_2002_l245_245284


namespace knights_in_room_l245_245037

noncomputable def number_of_knights (n : ℕ) : ℕ :=
  if n = 15 then 9 else 0

theorem knights_in_room : ∀ (n : ℕ), 
  (n = 15 ∧ 
  (∀ (k l : ℕ), k + l = n ∧ k ≥ 8 ∧ l ≥ 6 → k = 9)) :=
begin
  intro n,
  split,
  { -- prove n = 15
    sorry,
  },
  { -- prove the number of knights k is 9 when conditions are met
    intros k l h,
    sorry
  }
end

end knights_in_room_l245_245037


namespace percentage_decrease_in_larger_angle_l245_245467

-- Define the angles and conditions
def angles_complementary (A B : ℝ) : Prop := A + B = 90
def angle_ratio (A B : ℝ) : Prop := A / B = 1 / 2
def small_angle_increase (A A' : ℝ) : Prop := A' = A * 1.2
def large_angle_new (A' B' : ℝ) : Prop := A' + B' = 90

-- Prove percentage decrease in larger angle
theorem percentage_decrease_in_larger_angle (A B A' B' : ℝ) 
    (h1 : angles_complementary A B)
    (h2 : angle_ratio A B)
    (h3 : small_angle_increase A A')
    (h4 : large_angle_new A' B')
    : (B - B') / B * 100 = 10 :=
sorry

end percentage_decrease_in_larger_angle_l245_245467


namespace toms_total_money_l245_245096

def quarter_value : ℕ := 25 -- cents
def dime_value : ℕ := 10 -- cents
def nickel_value : ℕ := 5 -- cents
def penny_value : ℕ := 1 -- cent

def quarters : ℕ := 10
def dimes : ℕ := 3
def nickels : ℕ := 4
def pennies : ℕ := 200

def total_in_cents : ℕ := (quarters * quarter_value) + (dimes * dime_value) + (nickels * nickel_value) + (pennies * penny_value)

def total_in_dollars : ℝ := total_in_cents / 100

theorem toms_total_money : total_in_dollars = 5 := by
  sorry

end toms_total_money_l245_245096


namespace exists_two_unusual_numbers_l245_245562

noncomputable def is_unusual (n : ℕ) : Prop :=
  (n ^ 3 % 10 ^ 100 = n) ∧ (n ^ 2 % 10 ^ 100 ≠ n)

theorem exists_two_unusual_numbers :
  ∃ n1 n2 : ℕ, (is_unusual n1) ∧ (is_unusual n2) ∧ (n1 ≠ n2) ∧ (n1 >= 10 ^ 99) ∧ (n1 < 10 ^ 100) ∧ (n2 >= 10 ^ 99) ∧ (n2 < 10 ^ 100) :=
begin
  sorry
end

end exists_two_unusual_numbers_l245_245562


namespace median_to_AB_perpendicular_bisector_to_AB_l245_245691

open Real

namespace MedianAndPerpendicularBisector

def point (x y : ℝ) := (x, y)
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
def line_eq (m b : ℝ) (x y : ℝ) := y = m * x + b

theorem median_to_AB :
  let A := point 1 2
  let B := point (-1) 4
  let M := midpoint A B
  let m := slope A M
  ∃ b, line_eq m b M.1 M.2 → ∀ x y, line_eq m b x y → y = -x + 3 :=
by
  sorry

theorem perpendicular_bisector_to_AB :
  let A := point 1 2
  let B := point (-1) 4
  let k_AB := slope A B
  let k_perp := -1 / k_AB
  let M := midpoint A B
  ∃ b, line_eq k_perp b M.1 M.2 → ∀ x y, line_eq k_perp b x y → y = x + 3 :=
by
  sorry

end MedianAndPerpendicularBisector

end median_to_AB_perpendicular_bisector_to_AB_l245_245691


namespace xiao_ming_incorrect_steps_l245_245609

-- Definitions based on conditions from part a.
variable (x : ℝ)

-- Mathematical definitions related to the problem
def inequality_original := (x + 5) / 2 - 1 < (3 * x + 2) / 2
def step_1 := (x + 5) - 2 < 3 * x + 2
def step_2 := x - 3 * x < 2 - 5 + 1
def step_3 := -2 * x < -2
def step_4 := x > 1

-- Proof problem: Verify that the final solution from the inequality proves the correct step, and identify incorrect ones.
theorem xiao_ming_incorrect_steps : 
  inequality_original → step_1 ∧ step_4 :=
  by sorry

end xiao_ming_incorrect_steps_l245_245609


namespace hyperbola_equation_of_focus_and_asymptote_l245_245074

theorem hyperbola_equation_of_focus_and_asymptote :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (2 * a) ^ 2 + (2 * b) ^ 2 = 25 ∧ b / a = 2 ∧ 
  (∀ x y : ℝ, (y = 2 * x + 10) → (x = -5) ∧ (y = 0)) ∧ 
  (∀ x y : ℝ, (x ^ 2 / 5 - y ^ 2 / 20 = 1)) :=
by
  sorry

end hyperbola_equation_of_focus_and_asymptote_l245_245074


namespace evaluate_expression_l245_245643

theorem evaluate_expression : (1 / (1 - 1 / (3 + 1 / 4))) = (13 / 9) :=
by
  sorry

end evaluate_expression_l245_245643


namespace prime_in_repeated_61_list_l245_245634

-- Define the sequence of numbers in the problem, forming numbers by repeating "61"
def is_in_list (n : ℕ) : Prop := ∃ k : ℕ, n = 61 * ((10^(2*k) - 1) / 99)

-- Define the primality predicate
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the list of numbers starting from 61, excluding non-multiples of it
noncomputable def repeated_61 : List ℕ := List.map (λ k => 61 * (10^(2*k) / 99)) (List.range 100)

-- The main theorem
theorem prime_in_repeated_61_list : List.countp is_prime repeated_61 = 1 := by
  sorry

end prime_in_repeated_61_list_l245_245634


namespace greatest_prime_factor_391_l245_245109

theorem greatest_prime_factor_391 : ∃ p, prime p ∧ p ∣ 391 ∧ ∀ q, prime q ∧ q ∣ 391 → q ∣ p := by
  sorry

end greatest_prime_factor_391_l245_245109


namespace oil_quantity_relationship_l245_245785

variable (Q : ℝ) (t : ℝ)

-- Initial quantity of oil in the tank
def initial_quantity := 40

-- Flow rate of oil out of the tank
def flow_rate := 0.2

-- Function relationship between remaining oil quantity Q and time t
theorem oil_quantity_relationship : Q = initial_quantity - flow_rate * t :=
sorry

end oil_quantity_relationship_l245_245785


namespace field_area_is_36_square_meters_l245_245030

theorem field_area_is_36_square_meters (side_length : ℕ) (h : side_length = 6) : side_length * side_length = 36 :=
by
  sorry

end field_area_is_36_square_meters_l245_245030


namespace knight_count_l245_245036

theorem knight_count (K L : ℕ) (h1 : K + L = 15) 
  (h2 : ∀ k, k < K → (∃ l, l < L ∧ l = 6)) 
  (h3 : ∀ l, l < L → (K > 7)) : K = 9 :=
by 
  sorry

end knight_count_l245_245036


namespace smallest_integer_cubing_y_eq_350_l245_245590

def y : ℕ := 2^3 * 3^5 * 4^5 * 5^4 * 6^3 * 7^5 * 8^2

theorem smallest_integer_cubing_y_eq_350 : ∃ z : ℕ, z * y = (2^23) * (3^9) * (5^6) * (7^6) → z = 350 :=
by
  sorry

end smallest_integer_cubing_y_eq_350_l245_245590


namespace geometric_sum_eq_l245_245697

variable {R : Type*} [LinearOrderedField R]

def geometric_seq (a : ℕ → R) := ∃ q : R, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sum_eq {a : ℕ → R} (h : geometric_seq a) (h2 : a 2 = 2) (h5 : a 5 = 1 / 4) :
  a 1 * a 2 + a 2 * a 3 + ∑ i in Finset.range (n + 1), a i * a (i + 1) = 
  (512 / 15) * (1 - 1 / 16 ^ n) :=
sorry

end geometric_sum_eq_l245_245697


namespace tangent_normal_equations_l245_245965

noncomputable theory
open Real

def x (t : ℝ) := t * (t * cos t - 2 * sin t)
def y (t : ℝ) := t * (t * sin t + 2 * cos t)
def t₀ := π / 4
def x₀ := x t₀
def y₀ := y t₀

def tangentLine (x y : ℝ) := y = -x + (π^2 * sqrt 2 / 16)
def normalLine (x y : ℝ) := y = x + (π * sqrt 2 / 2)

theorem tangent_normal_equations :
  tangentLine x₀ y₀ ∧ normalLine x₀ y₀ :=
  sorry

end tangent_normal_equations_l245_245965


namespace max_ab_5sqrt2_l245_245078

noncomputable def max_ab {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : 2 * real.sqrt (a^2 + b^2) = 10) : ℝ :=
  a + b

theorem max_ab_5sqrt2 {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : 2 * real.sqrt (a^2 + b^2) = 10) :
  max_ab h_pos_a h_pos_b h = 5 * real.sqrt 2 :=
sorry

end max_ab_5sqrt2_l245_245078


namespace distance_PF_l245_245535

-- Definitions for the given conditions
structure Rectangle :=
  (EF GH: ℝ)
  (interior_point : ℝ × ℝ)
  (PE : ℝ)
  (PH : ℝ)
  (PG : ℝ)

-- The theorem to prove PF equals 12 under the given conditions
theorem distance_PF 
  (r : Rectangle)
  (hPE : r.PE = 5)
  (hPH : r.PH = 12)
  (hPG : r.PG = 13) :
  ∃ PF, PF = 12 := 
sorry

end distance_PF_l245_245535


namespace find_x_y_l245_245280

-- Expressing the number 141x28y3
def num := 14100000 + x * 100000 + 280000 + y * 1000 + 3

-- Conditions
theorem find_x_y (x y : ℕ) 
  (h1: 99 ∣ num)
  (h2: 0 ≤ x ∧ x ≤ 9) 
  (h3: 0 ≤ y ∧ y ≤ 9) : 
  x = 4 ∧ y = 4 := 
sorry

end find_x_y_l245_245280


namespace binary_to_base4_l245_245513

theorem binary_to_base4 (n : ℕ) : n = 10111011002 → nat.base_repr n 4 = "23230" := by
  intros h
  rw h
  sorry

end binary_to_base4_l245_245513


namespace octagon_area_l245_245942

-- Define the given condition: the circle has an area of 400π square units
def circle_area := 400 * Real.pi

-- Define the calculation of the radius from the circle's area
def radius (ca : ℝ) : ℝ := Real.sqrt (ca / Real.pi)

-- Define the formula to compute the area of an isosceles triangle with given base angle and radius
def octagon_triangle_area (r : ℝ) (angle : ℝ) : ℝ := (1/2) * r * r * Real.sin angle

-- Translation of the problem statement into a theorem in Lean 4
theorem octagon_area {ca : ℝ} (h : ca = circle_area) : 
  let r := radius ca in
  let tri_angle := Real.pi / 4 in
  let one_triangle_area := octagon_triangle_area r tri_angle in
  8 * one_triangle_area = 800 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l245_245942


namespace percentage_decrease_in_larger_angle_l245_245469

-- Define the angles and conditions
def angles_complementary (A B : ℝ) : Prop := A + B = 90
def angle_ratio (A B : ℝ) : Prop := A / B = 1 / 2
def small_angle_increase (A A' : ℝ) : Prop := A' = A * 1.2
def large_angle_new (A' B' : ℝ) : Prop := A' + B' = 90

-- Prove percentage decrease in larger angle
theorem percentage_decrease_in_larger_angle (A B A' B' : ℝ) 
    (h1 : angles_complementary A B)
    (h2 : angle_ratio A B)
    (h3 : small_angle_increase A A')
    (h4 : large_angle_new A' B')
    : (B - B') / B * 100 = 10 :=
sorry

end percentage_decrease_in_larger_angle_l245_245469


namespace initial_plan_days_l245_245148

-- Define the given conditions in Lean
variables (D : ℕ) -- Initial planned days for completing the job
variables (P : ℕ) -- Number of people initially hired
variables (Q : ℕ) -- Number of people fired
variables (W1 : ℚ) -- Portion of the work done before firing people
variables (D1 : ℕ) -- Days taken to complete W1 portion of work
variables (W2 : ℚ) -- Remaining portion of the work done after firing people
variables (D2 : ℕ) -- Days taken to complete W2 portion of work

-- Conditions from the problem
axiom h1 : P = 10
axiom h2 : Q = 2
axiom h3 : W1 = 1 / 4
axiom h4 : D1 = 20
axiom h5 : W2 = 3 / 4
axiom h6 : D2 = 75

-- The main theorem that proves the total initially planned days were 80
theorem initial_plan_days : D = 80 :=
sorry

end initial_plan_days_l245_245148


namespace intersect_tetrahedral_angle_by_plane_l245_245421

noncomputable theory

open Geometry

def tetrahedral_angle_intersection (A B C D S : Point) (a b : Line) (α : Plane) : Prop :=
  convex_tetrahedral_angle A B C D S ∧
  intersects_line (face A S B) (face C S D) a S ∧
  intersects_line (face A S D) (face B S C) b S ∧
  plane_passes_through_lines α a b ∧
  (∀ (P : Point), P ∈ α ∧ P ∈ segment (face A S B) → 
    intersects α (tetrahedral_angle A B C D S) ∧ 
    quadrilateral (cross_section α (tetrahedral_angle A B C D S))).quadrilateral_parallelogram

theorem intersect_tetrahedral_angle_by_plane :
  ∀ (A B C D S : Point) (a b : Line) (α : Plane),
  tetrahedral_angle_intersection A B C D S a b α →
  ∃ (P : Parallelogram), (cross_section α (tetrahedral_angle A B C D S)) = P :=
by
  intros A B C D S a b α H
  sorry

end intersect_tetrahedral_angle_by_plane_l245_245421


namespace quadratic_polynomial_prime_values_l245_245151

-- Let f be a quadratic polynomial with integer coefficients
def is_quadratic_polynomial (f : ℤ → ℤ) : Prop := 
  ∃ a b c : ℤ, ∀ x : ℤ, f(x) = a * x^2 + b * x + c

-- Check if a number is prime
def is_prime (p : ℤ) : Prop := 
  p > 1 ∧ (∀ d : ℤ, d ∣ p → d = 1 ∨ d = p)

-- The main theorem
theorem quadratic_polynomial_prime_values 
  (f : ℤ → ℤ) (n : ℤ)
  (h_quad : is_quadratic_polynomial f)
  (h1 : is_prime (f (n - 1)))
  (h2 : is_prime (f n))
  (h3 : is_prime (f (n + 1))) : 
  ∃ m : ℤ, m ≠ n - 1 ∧ m ≠ n ∧ m ≠ n + 1 ∧ is_prime (f m) :=
by 
  sorry

end quadratic_polynomial_prime_values_l245_245151


namespace total_money_l245_245089

def value_of_quarters (count: ℕ) : ℝ := count * 0.25
def value_of_dimes (count: ℕ) : ℝ := count * 0.10
def value_of_nickels (count: ℕ) : ℝ := count * 0.05
def value_of_pennies (count: ℕ) : ℝ := count * 0.01

theorem total_money (q d n p : ℕ) :
  q = 10 → d = 3 → n = 4 → p = 200 →
  value_of_quarters q + value_of_dimes d + value_of_nickels n + value_of_pennies p = 5.00 :=
by {
  intros,
  sorry
}

end total_money_l245_245089


namespace students_less_than_D_l245_245920

theorem students_less_than_D (total_students : ℕ) (bombed_ratio : ℚ) (no_show_ratio : ℚ) (passed_students : ℕ) : 
  total_students = 180 → 
  bombed_ratio = 1/4 → 
  no_show_ratio = 1/3 → 
  passed_students = 70 → 
  let bombed := (bombed_ratio * total_students) in
  let remaining_after_bombed := total_students - bombed in
  let no_show := (no_show_ratio * remaining_after_bombed) in
  let remaining_after_no_show := remaining_after_bombed - no_show in
  let less_than_D := remaining_after_no_show - passed_students in
  less_than_D = 20 :=
by
  intros
  simp only [bombed, remaining_after_bombed, no_show, remaining_after_no_show, less_than_D]
  sorry

end students_less_than_D_l245_245920


namespace divide_triangle_equal_area_l245_245260

def point := ℝ × ℝ

-- Define the vertices of the triangle
def A : point := (0, 4)
def B : point := (0, 0)
def C : point := (10, 0)

-- Calculate the area of the triangle ABC
def triangle_area (A B C : point) : ℝ :=
  1/2 * real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Prove that the vertical line x = d divides the triangle into two regions of equal area
theorem divide_triangle_equal_area (d : ℝ) :
  triangle_area A B C = 20 → d = 5 :=
  sorry

end divide_triangle_equal_area_l245_245260


namespace benzene_description_l245_245526

theorem benzene_description : 
  ¬ (
    (B : (benzene_is_aromatic_hydrocarbon : Prop) ∧
         (benzene_exists_in_two_isomeric_forms : Prop) ∧
         (benzene_undergoes_substitution_reactions : Prop) ∧
         (benzene_forms_three_products_with_C6H4Cl2 : Prop))
  ) :=
begin
  have benzene_is_aromatic_hydrocarbon : Prop := true,
  have benzene_exists_in_two_isomeric_forms : Prop := false,
  have benzene_undergoes_substitution_reactions : Prop := true,
  have benzene_forms_three_products_with_C6H4Cl2 : Prop := true,
  sorry -- Proof not needed per instructions
end

end benzene_description_l245_245526


namespace smallest_positive_difference_exists_l245_245509

-- Definition of how Vovochka sums two three-digit numbers
def vovochkaSum (x y : ℕ) : ℕ :=
  let h1 := (x / 100) + (y / 100)
  let t1 := ((x / 10) % 10) + ((y / 10) % 10)
  let u1 := (x % 10) + (y % 10)
  h1 * 1000 + t1 * 100 + u1

-- Definition for correct sum of two numbers
def correctSum (x y : ℕ) : ℕ := x + y

-- Function to find difference
def difference (x y : ℕ) : ℕ :=
  abs (vovochkaSum x y - correctSum x y)

-- Proof the smallest positive difference between Vovochka's sum and the correct sum
theorem smallest_positive_difference_exists :
  ∃ x y : ℕ, (x < 1000) ∧ (y < 1000) ∧ difference x y > 0 ∧ 
  (∀ a b : ℕ, (a < 1000) ∧ (b < 1000) ∧ difference a b > 0 → difference x y ≤ difference a b) :=
sorry

end smallest_positive_difference_exists_l245_245509


namespace find_f_log2_20_l245_245449

noncomputable def f : ℝ → ℝ
| x => if x ∈ (-1:ℝ, 0:ℝ) then 2^x + (1/5) else sorry

axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom periodic_function : ∀ x : ℝ, f(x - 2) = f(x + 2)

theorem find_f_log2_20 : f (Real.logb 2 20) = -1 := by
  sorry

end find_f_log2_20_l245_245449


namespace initial_number_of_men_l245_245887

variable (M : ℕ) (A : ℕ)
variable (change_in_age: ℕ := 16)
variable (age_increment: ℕ := 2)

theorem initial_number_of_men :
  ((A + age_increment) * M = A * M + change_in_age) → M = 8 :=
by
  intros h_1
  sorry

end initial_number_of_men_l245_245887


namespace josh_paths_to_top_center_l245_245372

/-- Define the grid --/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Josh's initial position --/
def start_pos : (ℕ × ℕ) := (0, 0)

/-- Define a function representing Josh's movement possibilities --/
def move_right (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1, pos.2 + 1)

def move_left_up (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1 + 1, pos.2 - 1)

/-- Define the goal position --/
def goal_pos (n : ℕ) : (ℕ × ℕ) :=
  (n - 1, 1)

/-- Theorem stating the required proof --/
theorem josh_paths_to_top_center {n : ℕ} (h : n > 0) : 
  let g := Grid.mk n 3 in
  ∃ (paths : ℕ), paths = 2^(n - 1) := 
  sorry

end josh_paths_to_top_center_l245_245372


namespace layla_total_fish_food_l245_245814

theorem layla_total_fish_food : 
  let goldfish_count := 2
      goldfish_food := 1
      swordtail_count := 3
      swordtail_food := 2
      guppy_count := 8
      guppy_food := 0.5
  in goldfish_count * goldfish_food + swordtail_count * swordtail_food + guppy_count * guppy_food = 12 := 
by
  let goldfish_count := 2
  let goldfish_food := 1
  let swordtail_count := 3
  let swordtail_food := 2
  let guppy_count := 8
  let guppy_food := 0.5
  calc
    goldfish_count * goldfish_food + swordtail_count * swordtail_food + guppy_count * guppy_food
          = 2 * 1 + 3 * 2 + 8 * 0.5 : by rfl
      ... = 2 + 6 + 4 : by norm_num
      ... = 12 : by norm_num

end layla_total_fish_food_l245_245814


namespace problem_1_problem_2_l245_245341

-- Definition of the problem and required constants
def curve (x : ℝ) : ℝ := x^2 - 8 * x + 2

def D : ℝ × ℝ := (2, 1 / 2)
def C := (4, 1.5)
def circle_C (x y : ℝ) : Prop :=
  (x - 4) ^ 2 + (y - 1.5) ^ 2 = 16.25

-- Proof statements
theorem problem_1 :
  let points := [(0, curve 0), (4 - sqrt 14, curve (4 - sqrt 14)), (4 + sqrt 14, curve (4 + sqrt 14))] in
  ∀ (p : ℝ × ℝ), p ∈ points → circle_C p.1 p.2 :=
sorry

theorem problem_2 :
  let P := (163 / 33, -35 / 33) in
  let on_line : ℝ × ℝ → Prop := fun P => P.1 - P.2 - 6 = 0 in
  on_line P ∧ (∀ Q : ℝ × ℝ, on_line Q → ((Q.1 - 4)^2 + (Q.2 - 1.5)^2)^0.5 + ((Q.1 - 2)^2 + (Q.2 - 1/2)^2)^0.5 ≥ ((P.1 - 4)^2 + (P.2 - 1.5)^2)^0.5 + ((P.1 - 2)^2 + (P.2 - 1/2)^2)^0.5) :=
sorry

end problem_1_problem_2_l245_245341


namespace option_C_l245_245596

theorem option_C (a b c : ℝ) (h₀ : a > b) (h₁ : b > c) (h₂ : c > 0) :
  (b + c) / (a + c) > b / a :=
sorry

end option_C_l245_245596


namespace solution_set_of_inequality_l245_245437

noncomputable def f : ℝ → ℝ := sorry

axiom deriv_f : ∀ x : ℝ, deriv f x = f' x 

axiom f_derivative_condition : ∀ x : ℝ, f x + (deriv f x) > 1

axiom f_at_zero : f 0 = 2016

theorem solution_set_of_inequality (x : ℝ) : 
  ( e^x * f x > e^x + 2015 ) ↔ ( x > 0 ) := by sorry

end solution_set_of_inequality_l245_245437


namespace inequality_hold_l245_245671

theorem inequality_hold (a b : ℝ) (h : a > b) : (1 / 2) ^ a < (1 / 2) ^ b := 
sorry

end inequality_hold_l245_245671


namespace magic_square_y_value_l245_245630

theorem magic_square_y_value :
  ∃ y : ℕ, (∃ a b c d e : ℕ,
    (y + 5 + c = 104 + a + c) ∧
    (a = y - 99) ∧
    (y + a + e = 104 + b + e) ∧
    (b = 2y - 203) ∧
    (y + 23 + 104 = 5 + a + b)) ∧
    y = 212 :=
by
  sorry

end magic_square_y_value_l245_245630


namespace complementary_angle_decrease_l245_245459

theorem complementary_angle_decrease (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90)
  (h2 : angle1 / 3 = angle2 / 6) (h3 : ∃ x: ℝ, x = 0.2) :
  let new_angle1 := angle1 * 1.2 in 
  let new_angle2 := 90 - new_angle1 in
  (new_angle2 - angle2) / angle2 = -0.10 := sorry

end complementary_angle_decrease_l245_245459


namespace trig_values_same_terminal_side_l245_245917

-- Statement: The trigonometric function values of angles with the same terminal side are equal.
theorem trig_values_same_terminal_side (θ₁ θ₂ : ℝ) (h : ∃ k : ℤ, θ₂ = θ₁ + 2 * k * π) :
  (∀ f : ℝ -> ℝ, f θ₁ = f θ₂) :=
by
  sorry

end trig_values_same_terminal_side_l245_245917


namespace problem_solution_l245_245594

def is_coprime_pair (x y : ℕ) : Prop := ∃ a b, a * x + b * y = 1

def all_pairwise_coprime (lst : List ℕ) : Prop :=
  ∀ i j, i < lst.length → j < lst.length → i ≠ j → is_coprime_pair (lst.nthLe i (by linarith)) (lst.nthLe j (by linarith))

def gcd (x y : ℕ) : ℕ := Nat.gcd x y

def circle_neighbors_sum (lst : List ℕ) (i : ℕ) : ℕ :=
  let len := lst.length
  lst.nthLe i (by linarith) + gcd (lst.nthLe ((i+1) % len) (by linarith)) (lst.nthLe ((i+len-1) % len) (by linarith))

theorem problem_solution : ∀ (lst : List ℕ), lst.length = 100 → 
  all_pairwise_coprime lst →
  ∃ (updated_lst : List ℕ), all_pairwise_coprime updated_lst ∧ 
  (∀ i, updated_lst.nthLe i (by linarith) = circle_neighbors_sum lst i) :=
sorry

end problem_solution_l245_245594


namespace value_of_a_plus_b_l245_245293

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if 0 ≤ x then Real.sqrt x + 3 else a * x + b

theorem value_of_a_plus_b (a b : ℝ) 
  (h1 : ∀ x1 : ℝ, x1 ≠ 0 → ∃ x2 : ℝ, x1 ≠ x2 ∧ f x1 a b = f x2 a b)
  (h2 : f (2 * a) a b = f (3 * b) a b) :
  a + b = - (Real.sqrt 6) / 2 + 3 :=
by
  sorry

end value_of_a_plus_b_l245_245293


namespace determinant_computation_l245_245748

theorem determinant_computation :
  let A := Matrix.of ![![2, 3], ![2, 2]]
  det (A ^ 2 - 3 • A) = 10 :=
by
  let A := Matrix.of ![![2, 3], ![2, 2]]
  sorry

end determinant_computation_l245_245748


namespace terminal_zeros_factorial_305_l245_245902

theorem terminal_zeros_factorial_305 (n : ℕ) (h : n = 305) : 
  let zeros := ∑ k in ({5, 25, 125, 625} : Finset ℕ), n / k in
  zeros = 75 :=
by
  sorry

end terminal_zeros_factorial_305_l245_245902


namespace square_difference_l245_245022

noncomputable def x : ℝ := 100^50 - 100^(-50)
noncomputable def y : ℝ := 100^50 + 100^(-50)

theorem square_difference :
  x^2 - y^2 = -4 := by
  sorry

end square_difference_l245_245022


namespace value_of_a_l245_245707

theorem value_of_a (a : ℝ) : (-2)^2 + 3*(-2) + a = 0 → a = 2 :=
by {
  sorry
}

end value_of_a_l245_245707


namespace area_of_regular_inscribed_octagon_l245_245939

theorem area_of_regular_inscribed_octagon (r : ℝ) (h : π * r^2 = 400 * π) : 
  (8 * (1/2) * r^2 * sin(π/4)) = 800 * sqrt 2 :=
begin
  sorry
end

end area_of_regular_inscribed_octagon_l245_245939


namespace first_digit_after_decimal_l245_245326

theorem first_digit_after_decimal :
  let s := ∑ n in Finset.range 1004, if n % 2 = 0 then \frac{1}{(2 * n + 1) * (2 * n + 2)} else -\frac{1}{(2 * n + 1) * (2 * n + 2)}
  floor (10 * s) % 10 = 4 :=
by 
  -- Declaration of the series sum
  let s := ∑ n in Finset.range 1004, if n % 2 = 0 then \frac{1}{(2 * n + 1) * (2 * n + 2)} else -\frac{1}{(2 * n + 1) * (2 * n + 2)}
  sorry

end first_digit_after_decimal_l245_245326


namespace find_m_l245_245646

theorem find_m (m : ℕ) (h : m * (Nat.factorial m) + 2 * (Nat.factorial m) = 5040) : m = 5 :=
by
  sorry

end find_m_l245_245646


namespace problem1_solution_problem2_solution_l245_245189

noncomputable def problem1 : Real :=
  2^(-1/2) + ((-4)^0) / Real.sqrt 2 + 1 / (Real.sqrt 2 - 1) - Real.sqrt ((1 - Real.sqrt 5)^0)

theorem problem1_solution : problem1 = 2 * Real.sqrt 2 :=
by
  sorry

noncomputable def problem2 : Real :=
  Real.log 2 / Real.log 2 * Real.log (1/16) / Real.log 3 * Real.log (1/9) / Real.log 5

theorem problem2_solution : problem2 = 8 * (Real.log 2 / Real.log 5) :=
by
  sorry

end problem1_solution_problem2_solution_l245_245189


namespace initial_crayons_count_l245_245087

variable (x : ℕ) -- x represents the initial number of crayons

theorem initial_crayons_count (h1 : x + 3 = 12) : x = 9 := 
by sorry

end initial_crayons_count_l245_245087


namespace age_difference_l245_245082

theorem age_difference 
  (a b : ℕ) 
  (h1 : 0 ≤ a ∧ a < 10) 
  (h2 : 0 ≤ b ∧ b < 10) 
  (h3 : 10 * a + b + 5 = 3 * (10 * b + a + 5)) : 
  (10 * a + b) - (10 * b + a) = 63 := 
by
  sorry

end age_difference_l245_245082


namespace inequality_proof_l245_245832

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b * c) + b / (a * c) + c / (a * b) ≥ 2 / a + 2 / b - 2 / c := 
  sorry

end inequality_proof_l245_245832


namespace compute_x_l245_245201

noncomputable def geometric_series_sum (a r : ℝ) : ℝ := a / (1 - r)

theorem compute_x :
  (geometric_series_sum 1 (1/2)) * (geometric_series_sum 1 (-(1/2))) = geometric_series_sum 1 (1/x) :=
by
  have sum1 : geometric_series_sum 1 (1/2) = 2 := by sorry
  have sum2 : geometric_series_sum 1 (-(1/2)) = 2/3 := by sorry
  have rhs_sum : geometric_series_sum 1 (1/x) = 4/3 := by sorry
  show x = 4 := by sorry

end compute_x_l245_245201


namespace max_cardinality_A_l245_245301

open Set

theorem max_cardinality_A (P : Finset ℕ) (A : Finset ℕ)
  (hP : P = {n ∈ Finset.range 2013 | 1 ≤ n})
  (hA : A ⊆ P ∧ ∀ a b ∈ A, a ≠ b → (a - b) % 101 ≠ 0) : A.card ≤ 101 :=
sorry

end max_cardinality_A_l245_245301


namespace factorize_expression_l245_245954

variables (a b x : ℝ)

theorem factorize_expression :
    5 * a * (x^2 - 1) - 5 * b * (x^2 - 1) = 5 * (x + 1) * (x - 1) * (a - b) := 
by
  sorry

end factorize_expression_l245_245954


namespace heaven_remaining_money_l245_245742

theorem heaven_remaining_money (total_money : ℕ) (notebook_price : ℕ) (notebook_count : ℕ)
    (eraser_price : ℕ) (eraser_count : ℕ) (highlighter_spent : ℕ) :
    total_money = 100 ∧ notebook_price = 5 ∧ notebook_count = 4 ∧ eraser_price = 4 ∧ eraser_count = 10 ∧ highlighter_spent = 30 →
    total_money - (notebook_price * notebook_count + eraser_price * eraser_count + highlighter_spent) = 10 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8
  cases h8 with h9 h10
  have h_notebooks : notebook_count * notebook_price = 20 := by
    rw [← h3]
    exact (by norm_num : 4 * 5 = 20)
  have h_erasers : eraser_count * eraser_price = 40 := by
    rw [← h5]
    exact (by norm_num : 10 * 4 = 40)
  have h_spent : 20 + 40 + highlighter_spent = 90 := by
    rw [← h6]
    exact (by norm_num : 20 + 40 + 30 = 90)
  have h_total_spent : total_money - (20 + 40 + highlighter_spent) = 10 := by
    rw [← h1, ← h_spent]
    exact (by norm_num : 100 - 90 = 10)
  exact h_total_spent

end heaven_remaining_money_l245_245742


namespace solve_quadratic_equation_l245_245654

noncomputable def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

theorem solve_quadratic_equation :
  let x1 := -2
      x2 := 11 in
  quadratic_eq 1 (-9) (-22) x1 ∧ quadratic_eq 1 (-9) (-22) x2 ∧ x1 < x2 :=
by
  sorry

end solve_quadratic_equation_l245_245654


namespace verify_value_of_sum_l245_245919

noncomputable def value_of_sum (a b c d e f : ℕ) (values : Finset ℕ) : ℕ :=
if h : a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ e ∈ values ∧ f ∈ values ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
        d ≠ e ∧ d ≠ f ∧
        e ≠ f ∧
        a + b = c ∧
        b + c = d ∧
        c + e = f
then a + c + f
else 0

theorem verify_value_of_sum :
  ∃ (a b c d e f : ℕ) (values : Finset ℕ),
  values = {4, 12, 15, 27, 31, 39} ∧
  a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ e ∈ values ∧ f ∈ values ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a + b = c ∧
  b + c = d ∧
  c + e = f ∧
  value_of_sum a b c d e f values = 73 :=
by
  sorry

end verify_value_of_sum_l245_245919


namespace value_set_of_angle_third_quadrant_l245_245325

theorem value_set_of_angle_third_quadrant (α k : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + 3 * π / 2) :
  {d : ℝ | d = (sin (α / 2)) / |sin (α / 2)| + (cos (α / 2)) / |cos (α / 2)|} = {0} :=
by
  sorry

end value_set_of_angle_third_quadrant_l245_245325


namespace sector_angle_l245_245442

noncomputable def central_angle_of_sector (r l : ℝ) : ℝ := l / r

theorem sector_angle (r l : ℝ) (h1 : 2 * r + l = 6) (h2 : (1 / 2) * l * r = 2) :
  central_angle_of_sector r l = 1 ∨ central_angle_of_sector r l = 4 :=
by
  sorry

end sector_angle_l245_245442


namespace balance_balls_l245_245866

open Real

variables (G B Y W : ℝ)

-- Conditions
def condition1 := (4 * G = 8 * B)
def condition2 := (3 * Y = 6 * B)
def condition3 := (8 * B = 6 * W)

-- Theorem statement
theorem balance_balls 
  (h1 : condition1 G B) 
  (h2 : condition2 Y B) 
  (h3 : condition3 B W) :
  ∃ (B_needed : ℝ), B_needed = 5 * G + 3 * Y + 4 * W ∧ B_needed = 64 / 3 * B :=
sorry

end balance_balls_l245_245866


namespace inequality_ABC_l245_245995

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_ABC_l245_245995


namespace seq_50th_term_eq_327_l245_245648

theorem seq_50th_term_eq_327 : 
  let n := 50
  let binary_representation : List Nat := [1, 1, 0, 0, 1, 0] -- 50 in binary
  let powers_of_3 := [5, 4, 1] -- Positions of 1s in the binary representation 
  let term := List.sum (powers_of_3.map (λ k => 3^k))
  term = 327 := by
  sorry

end seq_50th_term_eq_327_l245_245648


namespace value_of_b_l245_245752

theorem value_of_b (b : ℚ) (h : b + b / 4 = 3) : b = 12 / 5 := by
  sorry

end value_of_b_l245_245752


namespace distance_between_planes_l245_245651

open Real

theorem distance_between_planes : 
  let P1 : ℝ × ℝ × ℝ → Prop := λ (p : ℝ × ℝ × ℝ), p.1 + 2 * p.2 - 2 * p.3 + 1 = 0
  let P2 : ℝ × ℝ × ℝ → Prop := λ (p : ℝ × ℝ × ℝ), 2 * p.1 + 4 * p.2 - 4 * p.3 + 8 = 0
  ∀ (p1 p2 : ℝ × ℝ × ℝ), 
    P1 p1 → P2 p2 → 
    let n := (1, 2, -2) in
    let d := abs (n.1 * p1.1 + n.2 * p1.2 + n.3 * p1.3 + 4) / sqrt (n.1^2 + n.2^2 + n.3^2)
    d = 1 := 
by
  intro P1 P2 p1 p2 hP1 hP2
  let n := (1 : ℝ, 2 : ℝ, -2 : ℝ)
  let d := abs (n.1 * p1.1 + n.2 * p1.2 + n.3 * p1.3 + 4) / sqrt (n.1^2 + n.2^2 + n.3^2)
  have : d = 1 := sorry
  exact this

end distance_between_planes_l245_245651


namespace johnny_hours_second_job_l245_245810

theorem johnny_hours_second_job (x : ℕ) (h_eq : 5 * (69 + 10 * x) = 445) : x = 2 :=
by 
  -- The proof will go here, but we skip it as per the instructions
  sorry

end johnny_hours_second_job_l245_245810


namespace investment_period_two_years_l245_245600

theorem investment_period_two_years
  (P : ℝ) (r : ℝ) (A : ℝ) (n : ℕ) (hP : P = 6000) (hr : r = 0.10) (hA : A = 7260) (hn : n = 1) : 
  ∃ t : ℝ, t = 2 ∧ A = P * (1 + r / n) ^ (n * t) :=
by
  sorry

end investment_period_two_years_l245_245600


namespace cone_volume_correct_l245_245458

noncomputable def cone_volume (R : ℝ) : ℝ :=
  (1 / 3) * π * R^2 * (R * (sqrt 2) / 2)

theorem cone_volume_correct (R : ℝ) (h1 : R > 0)
  (h2: ∀ BA BC : ℝ, BA = BC) : cone_volume R = (π * R^3 * sqrt 2) / 6 :=
by
  sorry

end cone_volume_correct_l245_245458


namespace floor_expression_bounds_l245_245227

theorem floor_expression_bounds (x : ℝ) (h : ⌊x * ⌊x / 2⌋⌋ = 12) : 
  4.9 ≤ x ∧ x < 5.1 :=
sorry

end floor_expression_bounds_l245_245227


namespace stratified_sampling_correct_l245_245147

variable (totalEmployees seniorEmployees intermediateEmployees juniorEmployees sampleSize : ℕ)

def stratifiedSampling (totalEmployees seniorEmployees intermediateEmployees juniorEmployees sampleSize : ℕ) : Prop :=
  totalEmployees = 150 ∧ seniorEmployees = 15 ∧ intermediateEmployees = 45 ∧ juniorEmployees = 90 ∧ sampleSize = 30 →

  let proportionalSample := sampleSize * 1 / totalEmployees in
  
  proportionalSample * seniorEmployees = 3 ∧
  proportionalSample * intermediateEmployees = 9 ∧
  proportionalSample * juniorEmployees = 18

theorem stratified_sampling_correct :
  stratifiedSampling 150 15 45 90 30 :=
by
  sorry

end stratified_sampling_correct_l245_245147


namespace inequality_ABC_l245_245996

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_ABC_l245_245996


namespace leos_current_weight_l245_245315

theorem leos_current_weight (L K : ℝ) 
  (h1 : L + 10 = 1.5 * K) 
  (h2 : L + K = 180) : L = 104 := 
by 
  sorry

end leos_current_weight_l245_245315


namespace positive_difference_between_sums_l245_245003

def sum_nat (n : ℕ) : ℕ := n * (n + 1) / 2

def rounded_sum_to_25 (n : ℕ) : ℕ :=
  let intervals := [((0, 12), 0), ((13, 37), 25), ((38, 62), 50), ((63, 87), 75),
                    ((88, 112), 100), ((113, 137), 125), ((138, 162), 150), 
                    ((163, 187), 175), ((188, 200), 200)]
  intervals.foldl (λ acc (range_val : (ℕ × ℕ) × ℕ),
    let (range, val) := range_val in acc + (val * (range.2 - range.1 + 1))) 0

theorem positive_difference_between_sums : 
  abs (sum_nat 200 - rounded_sum_to_25 200) = 1625 := 
by
  sorry

end positive_difference_between_sums_l245_245003


namespace range_of_k_n_as_function_of_m_l245_245717

noncomputable def equation_circle (x y : ℝ) := x ^ 2 + (y - 4) ^ 2 = 4
def line (k x : ℝ) := k * x

theorem range_of_k (k : ℝ) :
  ( (k ^ 2 > 3) ↔ k ∈ (-∞, -sqrt 3) ∪ (√3, ∞) ) :=
sorry

theorem n_as_function_of_m (m n k : ℝ) 
  (h_line_n : n = k * m)
  (h_circle : equation_circle m n )
  (h_magic : 2 / (m ^ 2 + n ^ 2) = 
             1 / ((1 + k ^ 2) * m ^ 2) + 
             1 / ((1 + k ^ 2) * (8k/(1 + k^2)))^2 )
  ( h_k : k^2 > 3 )
  : n = (sqrt (15 * m ^ 2 + 180))/5 :=
sorry

end range_of_k_n_as_function_of_m_l245_245717


namespace josh_walk_ways_l245_245365

theorem josh_walk_ways (n : ℕ) :
  let grid_rows := n
      grid_columns := 3
      start_position := (0, 0)  -- (row, column) starting from bottom left
  in grid_rows > 0 →
      let center_square (k : ℕ) := (k, 1) -- center square of k-th row
  in ∃ ways : ℕ, ways = 2^(n-1) ∧
                ways = count_paths_to_center_topmost n  -- custom function representation
sorry

end josh_walk_ways_l245_245365


namespace find_triple_column_matrix_l245_245234

-- Define a 2x2 matrix type alias for simplicity
def Matrix2x2 := Matrix (Fin 2) (Fin 2) ℝ

-- Define the matrix M
def M : Matrix2x2 :=
  ![![1, 0], ![0, 3]]

-- Define the transformation condition
def triples_second_column (A : Matrix2x2) (B : Matrix2x2) : Prop :=
  B = ⟨λ i j, if j = 1 then 3 * A i j else A i j⟩ -- B(fst,0) = A(fst,0) and B(snd,0) = A(snd,0)
                                                 -- B(fst,1) = 3 * A(fst,1) and B(snd,1) = 3 * A(snd,1)

-- The theorem statement
theorem find_triple_column_matrix (A : Matrix2x2) :
  let B := matrix.mul M A in triples_second_column A B :=
sorry

end find_triple_column_matrix_l245_245234


namespace probability_top_card_face_card_of_hearts_l245_245588

-- Definitions based on conditions
def total_cards := 52
def face_cards_per_suit := 3
def suits := {"Hearts", "Diamonds", "Clubs", "Spades"}
def red_suits := {"Hearts", "Diamonds"}
def black_suits := {"Clubs", "Spades"}
def face_cards := {"Jack", "Queen", "King"}

-- The final theorem to prove
theorem probability_top_card_face_card_of_hearts :
  let num_favorable_outcomes := face_cards_per_suit
  let total_outcomes := total_cards
  num_favorable_outcomes.toReal / total_outcomes.toReal = 3 / 52 :=
by
  sorry  -- Proof skipped for now

end probability_top_card_face_card_of_hearts_l245_245588


namespace range_PF1_dot_PF2_max_area_incircle_l245_245262

-- Define the ellipse C
structure Ellipse (a b : ℝ) (a_gt_b_gt_0 : a > b ∧ b > 0) :=
  (focal_distance : ℝ)
  (foci_distance_condition : focal_distance = 2 * sqrt 3)
  (point_on_ellipse : ℝ × ℝ)
  (point_condition : point_on_ellipse = (sqrt 3, -1 / 2))
  (ellipse_condition :
    ∀ x y, (x, y) = point_on_ellipse → 
    (x^2 / (a^2)) + (y^2 / (b^2)) = 1)

-- Define the problem conditions
def ellipse_C := Ellipse
  2 1
  (by norm_num)
  (2 * sqrt 3)
  (by norm_num)
  (sqrt 3, -1/2)
  (by norm_num)

-- Proof for part (1)
theorem range_PF1_dot_PF2 : 
  ∃ foci : ℝ × ℝ, (foci = (0, 0)) → 
  (∃ P : ℝ × ℝ, P.1^2 / (2^2) + P.2^2 / (1^2) = 1 → 
  (∃ range : set ℝ, range = {x | x ≥ -2 ∧ x ≤ 1})) :=
sorry

-- Proof for part (2)
theorem max_area_incircle :
  ∃ incircle_area : ℝ, incircle_area = π / 4 :=
sorry

end range_PF1_dot_PF2_max_area_incircle_l245_245262


namespace set_A_not_right_triangle_set_B_is_right_triangle_set_C_is_right_triangle_set_D_is_right_triangle_l245_245181

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem set_A_not_right_triangle :
  ¬ is_right_triangle 4 6 8 :=
by
  simp [is_right_triangle]
  norm_num
  sorry

theorem set_B_is_right_triangle :
  is_right_triangle 5 12 13 :=
by
  simp [is_right_triangle]
  norm_num
  sorry

theorem set_C_is_right_triangle :
  is_right_triangle 6 8 10 :=
by
  simp [is_right_triangle]
  norm_num
  sorry

theorem set_D_is_right_triangle :
  is_right_triangle 7 24 25 :=
by
  simp [is_right_triangle]
  norm_num
  sorry

end set_A_not_right_triangle_set_B_is_right_triangle_set_C_is_right_triangle_set_D_is_right_triangle_l245_245181


namespace oil_quantity_relationship_l245_245783

variable (Q : ℝ) (t : ℝ)

-- Initial quantity of oil in the tank
def initial_quantity := 40

-- Flow rate of oil out of the tank
def flow_rate := 0.2

-- Function relationship between remaining oil quantity Q and time t
theorem oil_quantity_relationship : Q = initial_quantity - flow_rate * t :=
sorry

end oil_quantity_relationship_l245_245783


namespace problem_solution_l245_245532

noncomputable def problem_limit_seq : Real :=
  Real.lim (fun n =>  
    (Real.cbrt (n^2 + Real.cos n) + Real.sqrt (3 * n^2 + 2)) / Real.cbrt (n^6 + 1))

theorem problem_solution : problem_limit_seq = 0 := 
by 
  sorry

end problem_solution_l245_245532


namespace probability_red_purple_not_same_bed_l245_245793

def colors : Set String := {"red", "yellow", "white", "purple"}

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_red_purple_not_same_bed : 
  let total_ways := C 4 2
  let unwanted_ways := 2
  let desired_ways := total_ways - unwanted_ways
  let probability := (desired_ways : ℚ) / total_ways
  probability = 2 / 3 := by
  sorry

end probability_red_purple_not_same_bed_l245_245793


namespace field_trip_buses_needed_l245_245060

def fifth_graders : Nat := 109
def sixth_graders : Nat := 115
def seventh_graders : Nat := 118
def teachers_per_grade : Nat := 4
def parents_per_grade : Nat := 2
def total_grades : Nat := 3
def seats_per_bus : Nat := 72

def total_students : Nat := fifth_graders + sixth_graders + seventh_graders
def total_chaperones : Nat := (teachers_per_grade + parents_per_grade) * total_grades
def total_people : Nat := total_students + total_chaperones
def buses_needed : Nat := (total_people + seats_per_bus - 1) / seats_per_bus  -- ceiling division

theorem field_trip_buses_needed : buses_needed = 5 := by
  sorry

end field_trip_buses_needed_l245_245060


namespace green_tea_price_decrease_l245_245891

def percentage_change (old_price new_price : ℚ) : ℚ :=
  ((new_price - old_price) / old_price) * 100

theorem green_tea_price_decrease
  (C : ℚ)
  (h1 : C > 0)
  (july_coffee_price : ℚ := 2 * C)
  (mixture_price : ℚ := 3.45)
  (july_green_tea_price : ℚ := 0.3)
  (old_green_tea_price : ℚ := C)
  (equal_mixture : ℚ := (1.5 * july_green_tea_price) + (1.5 * july_coffee_price)) :
  mixture_price = equal_mixture →
  percentage_change old_green_tea_price july_green_tea_price = -70 :=
by
  sorry

end green_tea_price_decrease_l245_245891


namespace perpendicular_vectors_m_value_l245_245737

theorem perpendicular_vectors_m_value
  (a : ℝ × ℝ := (1, 2))
  (b : ℝ × ℝ)
  (h_perpendicular : (a.1 * b.1 + a.2 * b.2) = 0) :
  b = (-2, 1) :=
by
  sorry

end perpendicular_vectors_m_value_l245_245737


namespace sum_smallest_largest_l245_245064

theorem sum_smallest_largest (n a : ℕ) (h_even_n : n % 2 = 0) (y x : ℕ)
  (h_y : y = a + n - 1)
  (h_x : x = (a + 3 * (n / 3 - 1)) * (n / 3)) : 
  2 * y = a + (a + 2 * (n - 1)) :=
by
  sorry

end sum_smallest_largest_l245_245064


namespace rhombus_inscribed_circle_tangency_points_division_l245_245146

theorem rhombus_inscribed_circle_tangency_points_division 
  (acute_angle : ℝ)
  (h : acute_angle = 37) :
  ∃ angles : list ℝ, 
    angles = [143, 37, 143, 37] ∧
    sum angles = 360 :=
by
  exists [143, 37, 143, 37]
  split
  sorry -- Proof of equality
  sorry -- Proof of sum

end rhombus_inscribed_circle_tangency_points_division_l245_245146


namespace volunteer_activities_arrangement_l245_245549

theorem volunteer_activities_arrangement :
  let num_people := 6
  let num_activities := 2
  let max_per_activity := 4
  (nat.choose 6 4) * 2 + (nat.choose 6 3) = 50 :=
by
  let num_people := 6
  let num_activities := 2
  let max_per_activity := 4
  sorry

end volunteer_activities_arrangement_l245_245549


namespace amount_over_budget_l245_245611

-- Define the prices of each item
def cost_necklace_A : ℕ := 34
def cost_necklace_B : ℕ := 42
def cost_necklace_C : ℕ := 50
def cost_first_book := cost_necklace_A + 20
def cost_second_book := cost_necklace_C - 10

-- Define Bob's budget
def budget : ℕ := 100

-- Define the total cost
def total_cost := cost_necklace_A + cost_necklace_B + cost_necklace_C + cost_first_book + cost_second_book

-- Prove the amount over budget
theorem amount_over_budget : total_cost - budget = 120 := by
  sorry

end amount_over_budget_l245_245611


namespace percentage_increase_equiv_l245_245962

theorem percentage_increase_equiv {P : ℝ} : 
  (P * (1 + 0.08) * (1 + 0.08)) = (P * 1.1664) :=
by
  sorry

end percentage_increase_equiv_l245_245962


namespace maximum_sum_of_permuted_integers_with_unique_digits_l245_245175

open Nat

variables {a b c : ℕ}
variables {A B C : ℕ}

def is_6_digit (n : ℕ) : Prop := n >= 100000 ∧ n < 1000000

def unique_digits (n : ℕ) : Prop := 
  (List.ofDigits (digs 10 n)).Nodup ∧ (List.ofDigits (digs 10 n)).length = 6

def valid_integer (n A : ℕ) : Prop :=
  is_6_digit n ∧ unique_digits n ∧ A ∈ (List.ofDigits (digs 10 n))

def permutation_of_each_other (n m : ℕ) : Prop := 
  List.Permutation (List.ofDigits (digs 10 n)) (List.ofDigits (digs 10 m))

theorem maximum_sum_of_permuted_integers_with_unique_digits : 
  ∀ (a b c : ℕ) (A B C : ℕ),
  valid_integer a A → valid_integer b B → valid_integer c C →
  permutation_of_each_other a b → permutation_of_each_other b c → permutation_of_each_other a c → 
  a + b + c ≤ 1436649 :=
by 
  sorry

end maximum_sum_of_permuted_integers_with_unique_digits_l245_245175


namespace spherical_to_rectangular_coordinates_l245_245632

open Real

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 5 → θ = 7 * π / 4 → φ = π / 3 →
    (let x := ρ * sin φ * cos θ in
     let y := ρ * sin φ * sin θ in
     let z := ρ * cos φ in
     (x, y, z) = (-5 * sqrt 6 / 4, -5 * sqrt 6 / 4, 2.5)) :=
by
  intros ρ θ φ hρ hθ hφ
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coordinates_l245_245632


namespace weight_of_new_person_l245_245441

-- Definitions
variable (W : ℝ) -- total weight of original 15 people
variable (x : ℝ) -- weight of the new person
variable (n : ℕ) (avr_increase : ℝ) (original_person_weight : ℝ)
variable (total_increase : ℝ) -- total weight increase

-- Given constants
axiom n_value : n = 15
axiom avg_increase_value : avr_increase = 8
axiom original_person_weight_value : original_person_weight = 45
axiom total_increase_value : total_increase = n * avr_increase

-- Equation stating the condition
axiom weight_replace : W - original_person_weight + x = W + total_increase

-- Theorem (problem translated)
theorem weight_of_new_person : x = 165 := by
  sorry

end weight_of_new_person_l245_245441


namespace problem1_problem2_l245_245295

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + a*x - Real.log x

theorem problem1 (a : ℝ) : (∀ x > 0, (deriv (λ x, f x a)) x ≤ 0) → a ≤ 2 * Real.sqrt 2 := sorry

theorem problem2 (a : ℝ) : (∃ x y ∈ Ioo (0:ℝ) 3, (deriv (λ x, f x a)) x = 0 ∧ (deriv (λ x, f x a)) y = 0 ∧ x ≠ y) → 2 * Real.sqrt 2 < a ∧ a < 19 / 3 := sorry

end problem1_problem2_l245_245295


namespace max_possible_b_l245_245702

def max_b (a b : ℤ) : Prop :=
  (quadroot : ℤ -> Prop) (quadroot x = x^2 + a * x + b = 0) ∧ quadroot (a + b)

theorem max_possible_b (a b : ℤ) (h1 : a + b = quadroot) (h2 : quadroot (a + b)) : b ≤ 9 :=
sorry

end max_possible_b_l245_245702


namespace g_two_val_l245_245518

def g (x : ℕ) : ℕ := x^2 - 3*x + 1

theorem g_two_val : g 2 = -1 := by
  -- calculations and proof steps will be here
  sorry

end g_two_val_l245_245518


namespace probability_at_least_8_stay_l245_245863

def number_of_people := 9
def certain_stay := 5
def unsure_stay_probability := 1/3

theorem probability_at_least_8_stay : 
    (calc ∑ i in (finset.range 5), 
        if i = 3 then 4 * 1/27 * 2/3 else 
        if i = 4 then 1/81 else 0) = 1/9 :=
sorry

end probability_at_least_8_stay_l245_245863


namespace gary_profit_l245_245666

theorem gary_profit :
  let total_flour := 8 -- pounds
  let cost_flour := 4 -- dollars
  let large_cakes_flour := 5 -- pounds
  let small_cakes_flour := 3 -- pounds
  let flour_per_large_cake := 0.75 -- pounds per large cake
  let flour_per_small_cake := 0.25 -- pounds per small cake
  let cost_additional_large := 1.5 -- dollars per large cake
  let cost_additional_small := 0.75 -- dollars per small cake
  let cost_baking_equipment := 10 -- dollars
  let revenue_per_large := 6.5 -- dollars per large cake
  let revenue_per_small := 2.5 -- dollars per small cake
  let num_large_cakes := 6 -- (from calculation: ⌊5 / 0.75⌋)
  let num_small_cakes := 12 -- (from calculation: 3 / 0.25)
  let cost_additional_ingredients := num_large_cakes * cost_additional_large + num_small_cakes * cost_additional_small
  let total_revenue := num_large_cakes * revenue_per_large + num_small_cakes * revenue_per_small
  let total_cost := cost_flour + cost_baking_equipment + cost_additional_ingredients
  let profit := total_revenue - total_cost
  profit = 37 := by
  sorry

end gary_profit_l245_245666


namespace complex_number_is_rational_l245_245010

/-- 
  Let z = x + yi be a complex number with x and y 
  rational and with |z| = 1. Prove that the number 
  |z^(2n) - 1| is rational for every integer n. 
-/
theorem complex_number_is_rational
  (x y : ℚ)
  (z : ℂ)
  (n : ℤ)
  (h1 : z = x + y * complex.i)
  (h2 : complex.abs z = 1) :
  ∃ r : ℚ, complex.abs (z ^ (2 * ↑n) - 1) = r :=
sorry

end complex_number_is_rational_l245_245010


namespace probability_perfect_square_sum_l245_245492

theorem probability_perfect_square_sum (D1 D2 : ℕ) (h1 : 1 ≤ D1 ∧ D1 ≤ 8) (h2 : 1 ≤ D2 ∧ D2 ≤ 8) :
  (finset.filter (λ s, s = 4 ∨ s = 9 ∨ s = 16) (finset.image (λ x, fst x + snd x) (finset.product (finset.range 8) (finset.range 8)))).card / 64 = 3 / 16 :=
by
  sorry

end probability_perfect_square_sum_l245_245492


namespace inequality_proof_l245_245984

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l245_245984


namespace ellipse_equation_and_dot_product_range_l245_245689

theorem ellipse_equation_and_dot_product_range
  (a b : ℝ)
  (h_ab : a > b)
  (h_b : b > 0)
  (h_ellipse : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x, y) ∈ E)
  (focal_distance : real.dist (2, 0) (-2, 0) = 4)
  (chord_length : ∀ y : ℝ, (∃ l : ℝ, (2, l) ∈ E ∧ l = sqrt 2 ∧ (2, l) ∈ E))
  :
  (∃ a b : ℝ, (h_ab : a = 2 * sqrt 2) 
             ∧ (h_b : b = 2) 
             ∧ (h_ellipse : ∀ x y : ℝ, (x^2 / 8) + (y^2 / 4) = 1 → (x, y) ∈ E)
             ∧ (∀ (F M N : ℝ × ℝ), (F = (-2, 0) ∨ F = (2, 0))
                    ∧ (M ∈ E) 
                    ∧ (N ∈ E)
                    ∧ ((∃ x : ℝ, M = (x, sqrt 2 ∨ N = (x, - sqrt 2)))
                    → -4 ≤ (FM.1 * FN.1 + FM.2 * FN.2 ∧ FM.1 * FN.1 + FM.2 * FN.2 ≤ 14))) :=
begin
  sorry
end

end ellipse_equation_and_dot_product_range_l245_245689


namespace minimum_value_of_sum_of_squares_l245_245323

theorem minimum_value_of_sum_of_squares (x y z : ℝ) (h : 4 * x + 3 * y + 12 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 169 :=
by
  sorry

end minimum_value_of_sum_of_squares_l245_245323


namespace jason_fishes_on_day_12_l245_245807

def initial_fish_count : ℕ := 10

def fish_on_day (n : ℕ) : ℕ :=
  if n = 0 then initial_fish_count else
  (match n with
  | 1 => 10 * 3
  | 2 => 30 * 3
  | 3 => 90 * 3
  | 4 => 270 * 3 * 3 / 5 -- removes fish according to rule
  | 5 => (270 * 3 * 3 / 5) * 3
  | 6 => ((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7 -- removes fish according to rule
  | 7 => (((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3
  | 8 => ((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25
  | 9 => (((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3
  | 10 => ((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)
  | 11 => (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3
  | 12 => (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3 + (3 * (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3) + 5
  | _ => 0
  )
 
theorem jason_fishes_on_day_12 : fish_on_day 12 = 1220045 := 
  by sorry

end jason_fishes_on_day_12_l245_245807


namespace triangle_angle_equality_l245_245769

theorem triangle_angle_equality
  (A B C P Q : Type)
  [has_angle A B C 50]
  [has_angle A C B 50]
  [has_angle P C A 10]
  [has_angle Q B C 10]
  [has_angle P A C 20]
  [has_angle Q C B 20] :
  BP = BQ :=
by
  sorry

end triangle_angle_equality_l245_245769


namespace minimal_colors_l245_245333

def complete_graph (n : ℕ) := Type

noncomputable def color_edges (G : complete_graph 2015) := ℕ → ℕ → ℕ

theorem minimal_colors (G : complete_graph 2015) (color : color_edges G) :
  (∀ {u v w : ℕ} (h1 : u ≠ v) (h2 : v ≠ w) (h3 : w ≠ u), color u v ≠ color v w ∧ color u v ≠ color u w ∧ color u w ≠ color v w) →
  ∃ C: ℕ, C = 2015 := 
sorry

end minimal_colors_l245_245333


namespace count_solutions_three_x_plus_four_y_eq_1000_l245_245637

theorem count_solutions_three_x_plus_four_y_eq_1000 :
  { n : ℕ | ∃ x y : ℕ, 3 * x + 4 * y = 1000 ∧ x > 0 ∧ y > 0 }.card = 84 := 
sorry

end count_solutions_three_x_plus_four_y_eq_1000_l245_245637


namespace find_m_l245_245302

-- Definitions for the system of equations and the condition
def system_of_equations (x y m : ℝ) :=
  2 * x + 6 * y = 25 ∧ 6 * x + 2 * y = -11 ∧ x - y = m - 1

-- Statement to prove
theorem find_m (x y m : ℝ) (h : system_of_equations x y m) : m = -8 :=
  sorry

end find_m_l245_245302


namespace fifth_inequality_l245_245412

theorem fifth_inequality :
  1 + (1 / 2^2) + (1 / 3^2) + (1 / 4^2) + (1 / 5^2) + (1 / 6^2) < 11 / 6 :=
sorry

end fifth_inequality_l245_245412


namespace volume_scaled_l245_245916

-- Define the original surface area S and the original volume V
variables {S V r : ℝ}

-- Given Conditions: The surface area of a sphere is increased to 4 times its original size.
def sphere_surface_area_scaled (S' : ℝ) : Prop := S' = 4 * S

def radius_scaled (r' : ℝ) : Prop := r' = 2 * r

-- Question: Proving the volume will be 8 times the original volume.
theorem volume_scaled (S' V' r' : ℝ)
  (h1 : sphere_surface_area_scaled S')
  (h2 : radius_scaled r')
  (h3 : S = 4 * π * r^2)
  (h4 : V = 4/3 * π * r^3) :
  V' = 8 * V :=
by
  sorry

end volume_scaled_l245_245916


namespace num_pos_divisors_of_8400_and_7560_l245_245746

def num_pos_divisors_common (n m : ℕ) : ℕ :=
  let g := Nat.gcd n m
  in (g.factors.map fun p => p.2 + 1).foldl (· * ·) 1

theorem num_pos_divisors_of_8400_and_7560 : num_pos_divisors_common 8400 7560 = 32 := by
  sorry

end num_pos_divisors_of_8400_and_7560_l245_245746


namespace equation_of_line_AB_l245_245257

noncomputable def circle : set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 5 = 0}

def midpoint : ℝ × ℝ := (3, 1)

theorem equation_of_line_AB :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (set.mem (x, y) circle) → (set.mem (x, y) midpoint) → (y = m*x + b)) ∧
    (∃ (x y : ℝ), y = -x + 4) :=
sorry

end equation_of_line_AB_l245_245257


namespace period_of_f_minimum_value_zero_then_a_eq_one_maximum_value_of_f_axis_of_symmetry_l245_245298

noncomputable def f (x a : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2 + a

theorem period_of_f : ∀ a : ℝ, ∀ x : ℝ, f (x + π) a = f x a := 
by sorry

theorem minimum_value_zero_then_a_eq_one : (∀ x : ℝ, f x a ≥ 0) → a = 1 := 
by sorry

theorem maximum_value_of_f : a = 1 → (∀ x : ℝ, f x 1 ≤ 4) :=
by sorry

theorem axis_of_symmetry : a = 1 → ∃ k : ℤ, ∀ x : ℝ, 2 * x + π / 6 = k * π + π / 2 ↔ f x 1 = f 0 1 :=
by sorry

end period_of_f_minimum_value_zero_then_a_eq_one_maximum_value_of_f_axis_of_symmetry_l245_245298


namespace arithmetic_sequence_common_difference_l245_245713

theorem arithmetic_sequence_common_difference {a : ℕ → ℝ} (h₁ : a 1 = 2) (h₂ : a 2 + a 4 = a 6) : ∃ d : ℝ, d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l245_245713


namespace probability_real_roots_correct_l245_245161

/-- Given a polynomial with a coefficient chosen randomly and uniformly --/
def probability_real_roots : ℝ :=
  let interval_length := (4 - (-3) : ℝ),
  let valid_interval_length := ((-1 - (-3)) + (4 - 3) : ℝ),
  valid_interval_length / interval_length

theorem probability_real_roots_correct (a : ℝ) (h : -3 ≤ a ∧ a ≤ 4) : 
  let P := λ x : ℝ, x^3 + a * x^2 + a * x + 1,
  (probability_real_roots = 3 / 7) :=
by
  sorry

end probability_real_roots_correct_l245_245161


namespace number_of_distinct_tables_l245_245834

-- Lean formalization of the problem conditions
def is_permutation (l : List ℕ) (s : Finset ℕ) : Prop := 
  ∃ p : List ℕ, p ~ l ∧ p.toFinset = s

def is_derangement (f : ℕ → ℕ) (s : Finset ℕ) : Prop :=
  ∀ x ∈ s, f x ≠ x

-- Define the set of interest
def set4 : Finset ℕ := {1, 2, 3, 4}

-- The main theorem
theorem number_of_distinct_tables : 
  (∀ a : List ℕ, is_permutation a set4 →
    ∀ f : ℕ → ℕ, is_derangement f set4 → 
      True) → 
  216 := 
sorry

end number_of_distinct_tables_l245_245834


namespace find_m_value_l245_245573

theorem find_m_value (m : ℚ) :
  (m - 10) / -10 = (5 - m) / -8 → m = 65 / 9 :=
by
  sorry

end find_m_value_l245_245573


namespace probability_f_ge_0_l245_245294

def f (x : ℝ) : ℝ := -x^2 + 3*x

theorem probability_f_ge_0 : 
  let interval := Set.Icc (-1 : ℝ) (5 : ℝ)
  let pos_interval := Set.Icc (0 : ℝ) (3 : ℝ)
  (∃ P : ℝ, P = (pos_interval.measure) / (interval.measure) ∧ P = 1 / 2) :=
sorry

end probability_f_ge_0_l245_245294


namespace inequality_proof_l245_245978

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end inequality_proof_l245_245978


namespace smallest_positive_difference_exists_l245_245507

-- Definition of how Vovochka sums two three-digit numbers
def vovochkaSum (x y : ℕ) : ℕ :=
  let h1 := (x / 100) + (y / 100)
  let t1 := ((x / 10) % 10) + ((y / 10) % 10)
  let u1 := (x % 10) + (y % 10)
  h1 * 1000 + t1 * 100 + u1

-- Definition for correct sum of two numbers
def correctSum (x y : ℕ) : ℕ := x + y

-- Function to find difference
def difference (x y : ℕ) : ℕ :=
  abs (vovochkaSum x y - correctSum x y)

-- Proof the smallest positive difference between Vovochka's sum and the correct sum
theorem smallest_positive_difference_exists :
  ∃ x y : ℕ, (x < 1000) ∧ (y < 1000) ∧ difference x y > 0 ∧ 
  (∀ a b : ℕ, (a < 1000) ∧ (b < 1000) ∧ difference a b > 0 → difference x y ≤ difference a b) :=
sorry

end smallest_positive_difference_exists_l245_245507


namespace ratio_of_art_to_math_books_l245_245811

-- The conditions provided
def total_budget : ℝ := 500
def price_math_book : ℝ := 20
def num_math_books : ℕ := 4
def num_art_books : ℕ := num_math_books
def price_art_book : ℝ := 20
def num_science_books : ℕ := num_math_books + 6
def price_science_book : ℝ := 10
def cost_music_books : ℝ := 160

-- Desired proof statement
theorem ratio_of_art_to_math_books : num_art_books / num_math_books = 1 :=
by
  sorry

end ratio_of_art_to_math_books_l245_245811


namespace tangent_line_through_point_l245_245208

noncomputable def parabola (x : ℝ) : ℝ := x^2 + x + 1
noncomputable def tangent_line (x0 : ℝ) : (ℝ → ℝ) := λ x, (2 * x0 + 1) * (x - x0) + parabola x0

theorem tangent_line_through_point (x0 : ℝ) (h : 0 - parabola x0 = (2 * x0 + 1) * (-1 - x0)) :
  ∃ x : ℝ, (x - tangent_line x0 (-1) = 0) = (-1, 0) :=
begin
  sorry
end

end tangent_line_through_point_l245_245208


namespace negation_of_P_l245_245299

open Real

theorem negation_of_P :
  (¬ (∀ x : ℝ, x > sin x)) ↔ (∃ x : ℝ, x ≤ sin x) :=
by
  sorry

end negation_of_P_l245_245299


namespace sum_of_lengths_of_sides_and_diagonals_of_15gon_l245_245163

theorem sum_of_lengths_of_sides_and_diagonals_of_15gon (a b c d : ℤ) :
  ∃ (a b c d : ℤ), 
    ∀ (radius : ℝ), 
      radius = 15 →
      ∃ (sum : ℝ), 
        sum = (15 * 2 * radius * (Real.sin (Float.pi / 15)))
        + ∑ (n from 1 to 14), 2 * radius * (Real.sin ((n * Float.pi) / 15) / 2) ∧
        sum = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5 := sorry

end sum_of_lengths_of_sides_and_diagonals_of_15gon_l245_245163


namespace train_passengers_l245_245427

theorem train_passengers (total_passengers : ℕ) 
    (men_percentage women_percentage men_business_percentage : ℕ) 
    (women_first_percentage : ℕ)
    (H_total : total_passengers = 300)
    (H_men_percentage : men_percentage = 70)
    (H_women_percentage : women_percentage = 30)
    (H_men_business_percentage : men_business_percentage = 20)
    (H_women_first_percentage : women_first_percentage = 15) :
     let men := total_passengers * men_percentage / 100 in
     let women := total_passengers * women_percentage / 100 in
     let men_in_business := men * men_business_percentage / 100 in
     let women_in_first := women * women_first_percentage / 100 in
     men_in_business = 42 ∧ women_in_first = 14 :=
by sorry

end train_passengers_l245_245427


namespace bisect_angle_l245_245582

noncomputable def gamma {α : Type*} [EuclideanSpace α] (R : ℝ) : Semicircle α := sorry
noncomputable def l {α : Type*} [EuclideanSpace α] : Line α := sorry
noncomputable def C {α : Type*} [EuclideanSpace α] : α := sorry
noncomputable def D {α : Type*} [EuclideanSpace α] : α := sorry
noncomputable def B {α : Type*} [EuclideanSpace α] : α := sorry
noncomputable def A {α : Type*} [EuclideanSpace α] : α := sorry
noncomputable def O {α : Type*} [EuclideanSpace α] : α := sorry
noncomputable def E {α : Type*} [EuclideanSpace α] : α := sorry
noncomputable def F {α : Type*} [EuclideanSpace α] : α := sorry

theorem bisect_angle {α : Type*} [EuclideanSpace α] 
  (Γ : Semicircle α) (C D l : Line α) (B A O E F : α)
  (h1 : C ∈ Γ) (h2 : D ∈ Γ)
  (h3 : ∃ t₁ t₂ : α, B = tangent_point Γ C t₁ ∧ A = tangent_point Γ D t₂)
  (h4 : O ∈ segment B A)
  (h5 : E ∈ line_intersection (line_through A C) (line_through B D))
  (h6 : orthogonal (line_through E F) l) :
  is_angle_bisector F (angle_between C F D) :=
sorry

end bisect_angle_l245_245582


namespace ratio_of_toys_l245_245055

theorem ratio_of_toys (total_toys : ℕ) (num_friends : ℕ) (toys_D : ℕ) 
  (h1 : total_toys = 118) 
  (h2 : num_friends = 4) 
  (h3 : toys_D = total_toys / num_friends) : 
  (toys_D / total_toys : ℚ) = 1 / 4 :=
by
  sorry

end ratio_of_toys_l245_245055


namespace all_punctures_flat_time_l245_245173

theorem all_punctures_flat_time :
  let rate1 := 1 / 9
  let rate2 := 1 / 6
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  let time := 1 / combined_rate
  abs (time - 2.77) < 0.01 :=
by
  let rate1 := 1 / 9
  let rate2 := 1 / 6
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  let time := 1 / combined_rate
  sorry

end all_punctures_flat_time_l245_245173


namespace custom_op_evaluation_l245_245310

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x

theorem custom_op_evaluation : custom_op 6 4 - custom_op 4 6 = -6 :=
by
  sorry

end custom_op_evaluation_l245_245310


namespace part_a_part_b_part_c_l245_245152

-- Part a
def can_ratings_increase_after_first_migration (QA_before : ℚ) (QB_before : ℚ) (QA_after : ℚ) (QB_after : ℚ) : Prop :=
  QA_before < QA_after ∧ QB_before < QB_after

-- Part b
def can_ratings_increase_after_second_migration (QA_after_first : ℚ) (QB_after_first : ℚ) (QA_after_second : ℚ) (QB_after_second : ℚ) : Prop :=
  QA_after_second ≤ QA_after_first ∨ QB_after_second ≤ QB_after_first

-- Part c
def can_all_ratings_increase_after_reversed_migration (QA_before : ℚ) (QB_before : ℚ) (QC_before : ℚ) (QA_after_first : ℚ) (QB_after_first : ℚ) (QC_after_first : ℚ)
  (QA_after_second : ℚ) (QB_after_second : ℚ) (QC_after_second : ℚ) : Prop :=
  QA_before < QA_after_first ∧ QB_before < QB_after_first ∧ QC_before < QC_after_first ∧
  QA_after_first < QA_after_second ∧ QB_after_first < QB_after_second ∧ QC_after_first <= QC_after_second


-- Lean statements
theorem part_a (QA_before QA_after QB_before QB_after : ℚ) (Q_moved : ℚ) 
  (h : QA_before < QA_after ∧ QA_after < Q_moved ∧ QB_before < QB_after ∧ QB_after < Q_moved) : 
  can_ratings_increase_after_first_migration QA_before QB_before QA_after QB_after := 
by sorry

theorem part_b (QA_after_first QB_after_first QA_after_second QB_after_second : ℚ):
  ¬ can_ratings_increase_after_second_migration QA_after_first QB_after_first QA_after_second QB_after_second := 
by sorry

theorem part_c (QA_before QB_before QC_before QA_after_first QB_after_first QC_after_first
  QA_after_second QB_after_second QC_after_second: ℚ)
  (h: QA_before < QA_after_first ∧ QB_before < QB_after_first ∧ QC_before < QC_after_first ∧ 
      QA_after_first < QA_after_second ∧ QB_after_first < QB_after_second) :
   can_all_ratings_increase_after_reversed_migration QA_before QB_before QC_before QA_after_first QB_after_first QC_after_first QA_after_second QB_after_second QC_after_second :=
by sorry

end part_a_part_b_part_c_l245_245152


namespace find_larger_number_of_two_l245_245898

theorem find_larger_number_of_two (A B : ℕ) (hcf lcm : ℕ) (factor1 factor2 : ℕ)
  (h_hcf : hcf = 23)
  (h_factor1 : factor1 = 13)
  (h_factor2 : factor2 = 16)
  (h_lcm : lcm = hcf * factor1 * factor2)
  (h_A : A = hcf * m ∧ m = factor1)
  (h_B : B = hcf * n ∧ n = factor2):
  max A B = 368 := by
  sorry

end find_larger_number_of_two_l245_245898


namespace type_a_price_min_bundles_l245_245145

theorem type_a_price (x : ℝ) (h1 : 1.5 * x) (h2 : (300 / x) - (300 / (1.5 * x)) = 5) : x = 20 := 
by 
  sorry

theorem min_bundles (m : ℕ) (h1 : m + (100 - m) = 100) (h2 : 20 * m + 30 * (100 - m) ≤ 2400) : m ≥ 60 := 
by 
  sorry

end type_a_price_min_bundles_l245_245145


namespace inequality_hold_l245_245974

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end inequality_hold_l245_245974


namespace is_hexagonal_number_2016_l245_245745

theorem is_hexagonal_number_2016 :
  ∃ (n : ℕ), 2 * n^2 - n = 2016 :=
sorry

end is_hexagonal_number_2016_l245_245745


namespace statement_A_is_incorrect_statement_B_is_incorrect_l245_245952

theorem statement_A_is_incorrect:
  ∃ a b c : ℝ, (b^2 - 4 * a * c ≤ 0) ∧ (∃ x : ℝ, a * x^2 + b * x + c < 0) := sorry

theorem statement_B_is_incorrect:
  ∃ a b c : ℝ, (a > c) ∧ (ab^2 ≤ cb^2) := sorry

end statement_A_is_incorrect_statement_B_is_incorrect_l245_245952


namespace inequality_proof_l245_245976

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end inequality_proof_l245_245976


namespace object_travel_distance_in_one_hour_l245_245131

/-- If an object travels at 3 feet per second, then it travels 10800 feet in one hour. -/
theorem object_travel_distance_in_one_hour
  (speed : ℕ) (seconds_in_minute : ℕ) (minutes_in_hour : ℕ)
  (h_speed : speed = 3)
  (h_seconds_in_minute : seconds_in_minute = 60)
  (h_minutes_in_hour : minutes_in_hour = 60) :
  (speed * (seconds_in_minute * minutes_in_hour) = 10800) :=
by
  sorry

end object_travel_distance_in_one_hour_l245_245131


namespace transformed_graph_l245_245436

noncomputable def f (x : ℝ) : ℝ := sin (x - π / 3)

noncomputable def g (x : ℝ) : ℝ := f (x / 2)

noncomputable def h (x : ℝ) : ℝ := g (x - π / 3)

theorem transformed_graph :
  h x = sin (x / 2 - π / 2) :=
by
  unfold h g f
  sorry

end transformed_graph_l245_245436


namespace monotonic_increasing_intervals_l245_245453

noncomputable def f (x : ℝ) : ℝ := (Real.cos (x - Real.pi / 6))^2

theorem monotonic_increasing_intervals (k : ℤ) : 
  ∃ t : Set ℝ, t = Set.Ioo (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi) ∧ 
    ∀ x y, x ∈ t → y ∈ t → x ≤ y → f x ≤ f y :=
sorry

end monotonic_increasing_intervals_l245_245453


namespace total_cleaning_validation_l245_245408

-- Define the cleaning frequencies and their vacations
def Michael_bath_week := 2
def Michael_shower_week := 1
def Michael_vacation_weeks := 3

def Angela_shower_day := 1
def Angela_vacation_weeks := 2

def Lucy_bath_week := 3
def Lucy_shower_week := 2
def Lucy_alter_weeks := 4
def Lucy_alter_shower_day := 1
def Lucy_alter_bath_week := 1

def weeks_year := 52
def days_week := 7

-- Calculate Michael's total cleaning times in a year
def Michael_total := (Michael_bath_week * weeks_year) + (Michael_shower_week * weeks_year)
def Michael_vacation_reduction := Michael_vacation_weeks * (Michael_bath_week + Michael_shower_week)
def Michael_cleaning_times := Michael_total - Michael_vacation_reduction

-- Calculate Angela's total cleaning times in a year
def Angela_total := (Angela_shower_day * days_week * weeks_year)
def Angela_vacation_reduction := Angela_vacation_weeks * (Angela_shower_day * days_week)
def Angela_cleaning_times := Angela_total - Angela_vacation_reduction

-- Calculate Lucy's total cleaning times in a year
def Lucy_baths_total := Lucy_bath_week * weeks_year
def Lucy_showers_total := Lucy_shower_week * weeks_year
def Lucy_alter_showers := Lucy_alter_shower_day * days_week * Lucy_alter_weeks
def Lucy_alter_baths_reduction := (Lucy_bath_week - Lucy_alter_bath_week) * Lucy_alter_weeks
def Lucy_cleaning_times := Lucy_baths_total + Lucy_showers_total + Lucy_alter_showers - Lucy_alter_baths_reduction

-- Calculate the total times they clean themselves in 52 weeks
def total_cleaning_times := Michael_cleaning_times + Angela_cleaning_times + Lucy_cleaning_times

-- The proof statement
theorem total_cleaning_validation : total_cleaning_times = 777 :=
by simp [Michael_cleaning_times, Angela_cleaning_times, Lucy_cleaning_times, total_cleaning_times]; sorry

end total_cleaning_validation_l245_245408


namespace value_of_product_l245_245516

theorem value_of_product : (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049 = 243) :=
by sorry

end value_of_product_l245_245516


namespace similar_triangles_AC_length_l245_245805

open Classical

noncomputable def length_AC (BC FG GH : ℝ) : ℝ := (BC * GH) / FG

theorem similar_triangles_AC_length 
  (h_sim : ∀ (A B C F G H : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace F] [MetricSpace G] [MetricSpace H],
            SimilarTriangles A B C F G H)
  (BC_val : BC = 24)
  (FG_val : FG = 16)
  (GH_val : GH = 15) :
  (length_AC BC FG GH).round = 22.5 :=
by
  sorry

end similar_triangles_AC_length_l245_245805


namespace number_machine_output_l245_245575

def number_machine (n : ℕ) : ℕ :=
  let step1 := n * 3
  let step2 := step1 + 20
  let step3 := step2 / 2
  let step4 := step3 ^ 2
  let step5 := step4 - 45
  step5

theorem number_machine_output : number_machine 90 = 20980 := by
  sorry

end number_machine_output_l245_245575


namespace cos_angle_F1PF2_slope_line_OM_l245_245265

-- Part 1: Prove that the cosine value of angle F₁PF₂ is 7/9
theorem cos_angle_F1PF2 (x y : ℝ) (h : (x^2 / 8) + (y^2 / 9) = 1)
  (dPF1 : dist (x, y) (sqrt(1), 0) = 3) (dPF2 : dist (x, y) (-sqrt(1), 0) = 3) :
  cos (angle ((sqrt(1), 0)) ((x, y)) ((-sqrt(1), 0))) = 7 / 9 := 
sorry

-- Part 2: Prove that the slope of the line OM is -9/8
theorem slope_line_OM (x1 y1 x2 y2 : ℝ)
  (h1 : (x1^2 / 8) + (y1^2 / 9) = 1) (h2 : (x2^2 / 8) + (y2^2 / 9) = 1)
  (line_intersect : y1 = x1 + 1 ∧ y2 = x2 + 1) :
  ((x1 + x2) / 2, (y1 + y2) / 2).snd / ((x1 + x2) / 2) = -9 / 8 :=
sorry

end cos_angle_F1PF2_slope_line_OM_l245_245265


namespace find_alex_min_total_sum_l245_245812

noncomputable def kelvin_dice : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

noncomputable def kelvin_possible_sums : List ℕ :=
  (List.product kelvin_dice kelvin_dice).map (λ pair => pair.fst + pair.snd)

def alex_dice_possible_total_dots (a b : ℕ) (a_list b_list : List ℕ) : Prop :=
  a_list.sum = a ∧ b_list.sum = b ∧ a ≠ b

def same_distribution (l1 l2 : List ℕ) : Prop :=
  ∀ s, l1.count s = l2.count s

theorem find_alex_min_total_sum (a b : ℕ) (a_list b_list : List ℕ) :
  alex_dice_possible_total_dots a b a_list b_list →
  same_distribution kelvin_possible_sums
    ((List.product a_list b_list).map (λ pair => pair.fst + pair.snd)) →
  ∃ min_val : ℕ, min_val ∈ {24, 28, 32} ∧ min_val = min a b :=
sorry

end find_alex_min_total_sum_l245_245812


namespace total_games_played_l245_245541

theorem total_games_played (games_won games_lost : ℕ) (h₁ : games_won = 45) (h₂ : games_lost = 17) : 
  games_won + games_lost = 62 :=
by
  rw [h₁, h₂]
  norm_num
  done

end total_games_played_l245_245541


namespace maximum_abc_value_l245_245205

theorem maximum_abc_value:
  (∀ (a b c : ℝ), (0 < a ∧ a < 3) ∧ (0 < b ∧ b < 3) ∧ (0 < c ∧ c < 3) ∧ (∀ x : ℝ, (x^4 + a * x^3 + b * x^2 + c * x + 1) ≠ 0) → (abc ≤ 18.75)) :=
sorry

end maximum_abc_value_l245_245205


namespace total_money_found_l245_245094

def value_of_quarters (n_quarters : ℕ) : ℝ := n_quarters * 0.25
def value_of_dimes (n_dimes : ℕ) : ℝ := n_dimes * 0.10
def value_of_nickels (n_nickels : ℕ) : ℝ := n_nickels * 0.05
def value_of_pennies (n_pennies : ℕ) : ℝ := n_pennies * 0.01

theorem total_money_found (n_quarters n_dimes n_nickels n_pennies : ℕ) :
  n_quarters = 10 →
  n_dimes = 3 →
  n_nickels = 4 →
  n_pennies = 200 →
  value_of_quarters n_quarters + value_of_dimes n_dimes + value_of_nickels n_nickels + value_of_pennies n_pennies = 5.00 := 
by
  intros h_quarters h_dimes h_nickels h_pennies
  sorry

end total_money_found_l245_245094


namespace proof_sum_subset_l245_245733

noncomputable def sum_of_sums_of_subsets (n : ℕ) : ℕ :=
  ∑ k in finset.range (n + 1), k * 2^(n - 1)

theorem proof_sum_subset (n : ℕ) (hn : 0 < n) :
  sum_of_sums_of_subsets n = n * (n + 1) * 2^(n - 2) :=
by
  sorry

end proof_sum_subset_l245_245733


namespace find_a10_l245_245342

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Given conditions
variables (a : ℕ → ℤ) (h_arith_seq : arithmetic_sequence a)
variables (h_a2 : a 2 = 2) (h_a6 : a 6 = 10)

-- Goal to prove
theorem find_a10 : a 10 = 18 :=
by
  sorry

end find_a10_l245_245342


namespace sum_of_x_and_y_l245_245334

-- Define the given angles
def angle_A : ℝ := 34
def angle_B : ℝ := 74
def angle_C : ℝ := 32

-- State the theorem
theorem sum_of_x_and_y (x y : ℝ) :
  (680 - x - y) = 720 → (x + y = 40) :=
by
  intro h
  sorry

end sum_of_x_and_y_l245_245334


namespace max_f5_in_interval_l245_245724

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x

noncomputable def f_iter (n : ℕ) : ℝ → ℝ
| 0 => id
| (n + 1) => f ∘ f_iter n

theorem max_f5_in_interval :
  ∃ x ∈ set.Icc 1 2, f_iter 5 x = 3^32 - 1 :=
sorry

end max_f5_in_interval_l245_245724


namespace total_money_found_l245_245092

def value_of_quarters (n_quarters : ℕ) : ℝ := n_quarters * 0.25
def value_of_dimes (n_dimes : ℕ) : ℝ := n_dimes * 0.10
def value_of_nickels (n_nickels : ℕ) : ℝ := n_nickels * 0.05
def value_of_pennies (n_pennies : ℕ) : ℝ := n_pennies * 0.01

theorem total_money_found (n_quarters n_dimes n_nickels n_pennies : ℕ) :
  n_quarters = 10 →
  n_dimes = 3 →
  n_nickels = 4 →
  n_pennies = 200 →
  value_of_quarters n_quarters + value_of_dimes n_dimes + value_of_nickels n_nickels + value_of_pennies n_pennies = 5.00 := 
by
  intros h_quarters h_dimes h_nickels h_pennies
  sorry

end total_money_found_l245_245092


namespace largest_real_part_of_z2_l245_245595

def z1 : ℂ := -2
def z2 : ℂ := -real.sqrt 3 + complex.I
def z3 : ℂ := -real.sqrt 2 + real.sqrt 2 * complex.I
def z4 : ℂ := 2 * complex.I
def z5 : ℂ := -1 + real.sqrt 3 * complex.I

theorem largest_real_part_of_z2:
  ∀ (z : ℂ), z ∈ {z1, z2, z3, z4, z5} →
  z.re^5 ≤ (z2^5).re :=
sorry

end largest_real_part_of_z2_l245_245595


namespace greatest_prime_factor_391_l245_245107

theorem greatest_prime_factor_391 : 
  greatestPrimeFactor 391 = 23 :=
sorry

end greatest_prime_factor_391_l245_245107


namespace problem_part_I_problem_part_II_l245_245254

section
variable (a b x : ℝ)

def f (x : ℝ) := (a * x + b) * Real.exp (-2 * x)

def g (x : ℝ) := f a b x + x * Real.log x

theorem problem_part_I
  (h₀ : f a b 0 = 1)
  (h₁ : deriv (f a b) 0 = -1) :
  a = 1 ∧ b = 1 :=
sorry

theorem problem_part_II
  (h : 0 < x ∧ x < 1) :
  2 * Real.exp (-2) - Real.exp (-1) < g 1 1 x 
  ∧ g 1 1 x < 1 :=
sorry

end

end problem_part_I_problem_part_II_l245_245254


namespace students_not_enrolled_in_course_l245_245774

def total_students : ℕ := 150
def french_students : ℕ := 61
def german_students : ℕ := 32
def spanish_students : ℕ := 45
def french_and_german : ℕ := 15
def french_and_spanish : ℕ := 12
def german_and_spanish : ℕ := 10
def all_three_courses : ℕ := 5

theorem students_not_enrolled_in_course : total_students - 
    (french_students + german_students + spanish_students - 
     french_and_german - french_and_spanish - german_and_spanish + 
     all_three_courses) = 44 := by
  sorry

end students_not_enrolled_in_course_l245_245774


namespace unit_vectors_perpendicular_sum_of_squares_l245_245750

variable {V : Type*} [InnerProductSpace ℝ V]

theorem unit_vectors_perpendicular_sum_of_squares 
  (a b c : V) 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = 1) 
  (hc : ∥c∥ = 1) 
  (hab : ⟪a, b⟫ = 0) 
  (hac : ⟪a, c⟫ = 0) 
  (hbc : ⟪b, c⟫ = 0) :
  ∥a - b∥^2 + ∥a - c∥^2 + ∥b - c∥^2 = 6 :=
sorry

end unit_vectors_perpendicular_sum_of_squares_l245_245750


namespace calories_difference_l245_245614

theorem calories_difference
  (calories_squirrel : ℕ := 300)
  (squirrels_per_hour : ℕ := 6)
  (calories_rabbit : ℕ := 800)
  (rabbits_per_hour : ℕ := 2) :
  ((squirrels_per_hour * calories_squirrel) - (rabbits_per_hour * calories_rabbit)) = 200 :=
by
  sorry

end calories_difference_l245_245614


namespace product_of_k_values_l245_245402

theorem product_of_k_values (a b c k : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_eq : a / (1 + b) = k ∧ b / (1 + c) = k ∧ c / (1 + a) = k) : k = -1 :=
by
  sorry

end product_of_k_values_l245_245402


namespace area_bounds_l245_245330

variables {A B C P E F : Type} [plane_geometry : Geometry A B C P E F]
variables (H₁ : IsOnLineSegment P B C)
variables (H₂ : ParallelLine PE BA)
variables (H₃ : ParallelLine PF CA)
variables (H₄ : area (triangle A B C) = 1)

theorem area_bounds : 
  max (area (triangle B P F)) (max (area (triangle P C E)) (area (quadrilateral P E A F))) ≥ (4 : ℝ) / 9 :=
sorry

end area_bounds_l245_245330


namespace find_a_l245_245016

-- Define the conditions
def a_ge_zero (a : ℝ) : Prop := a ≥ 0

def f (a x : ℝ) : ℝ := a * real.sqrt (1 - x^2) + real.sqrt (1 + x) - real.sqrt (1 - x)

def t_def (x : ℝ) : ℝ := real.sqrt (1 + x) - real.sqrt (1 - x)

noncomputable def m (a t : ℝ) : ℝ :=
  - (1/2) * a * t ^ 2 + t + a

-- Define the maximum function g(a)
noncomputable def g (a : ℝ) : ℝ :=
  if a > real.sqrt 2 / 2 then (1 / (2 * a)) + a
  else real.sqrt 2

-- Statement to prove
theorem find_a (a : ℝ) : a_ge_zero a → g a = g (1 / a) → a = 1 :=
by sorry

end find_a_l245_245016


namespace angle_VYZ_34_l245_245346

variables (P Q R S X Y T V Z : Point)
variables (line_PQ line_RS line_UV line_TY line_ZV : Line)

-- Given conditions:
axiom PQ_parallel_RS : parallel line_PQ line_RS
axiom angle_PXT_146 : angle P X T = 146
axiom line_UV_intersects_PQ_at_X : intersects line_UV line_PQ X
axiom line_UV_intersects_RS_at_Y : intersects line_UV line_RS Y
axiom line_TY_crosses_RS_at_Y_meets_UV_at_T : on_line line_TY Y ∧ intersects line_TY line_UV T
axiom Z_on_TY : on_segment line_TY Z
axiom ZV_parallel_PQ : parallel line_ZV line_PQ
axiom ZV_intersects_UV_at_V : intersects line_ZV line_UV V

theorem angle_VYZ_34 : 
  ∀ (P Q R S X Y T V Z : Point) 
    (line_PQ line_RS line_UV line_TY line_ZV : Line),
    parallel line_PQ line_RS →
    angle P X T = 146 →
    intersects line_UV line_PQ X →
    intersects line_UV line_RS Y →
    on_line line_TY Y ∧ intersects line_TY line_UV T →
    on_segment line_TY Z →
    parallel line_ZV line_PQ →
    intersects line_ZV line_UV V →
    angle V Y Z = 34 := sorry

end angle_VYZ_34_l245_245346


namespace minimum_bailing_rate_l245_245123

-- Conditions as formal definitions.
def distance_from_shore : ℝ := 3
def intake_rate : ℝ := 20 -- gallons per minute
def sinking_threshold : ℝ := 120 -- gallons
def speed_first_half : ℝ := 6 -- miles per hour
def speed_second_half : ℝ := 3 -- miles per hour

-- Formal translation of the problem using definitions.
theorem minimum_bailing_rate : (distance_from_shore = 3) →
                             (intake_rate = 20) →
                             (sinking_threshold = 120) →
                             (speed_first_half = 6) →
                             (speed_second_half = 3) →
                             (∃ r : ℝ, 18 ≤ r) :=
by
  sorry

end minimum_bailing_rate_l245_245123


namespace unusual_numbers_exist_l245_245564

noncomputable def n1 : ℕ := 10 ^ 100 - 1
noncomputable def n2 : ℕ := 10 ^ 100 / 2 - 1

theorem unusual_numbers_exist : 
  (n1 ^ 3 % 10 ^ 100 = n1 ∧ n1 ^ 2 % 10 ^ 100 ≠ n1) ∧ 
  (n2 ^ 3 % 10 ^ 100 = n2 ∧ n2 ^ 2 % 10 ^ 100 ≠ n2) :=
by
  sorry

end unusual_numbers_exist_l245_245564


namespace triangle_def_area_l245_245329

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_def_area :
  ∀ (DE EF DF : ℝ) (hDE : DE = 17) (hEF : EF = 17) (hDF : DF = 26),
  area_of_triangle DE EF DF = 142 := by
  intros DE EF DF hDE hEF hDF
  sorry

end triangle_def_area_l245_245329


namespace cornflowers_count_l245_245028

theorem cornflowers_count
  (n k : ℕ)
  (total_flowers : 9 * n + 17 * k = 70)
  (equal_dandelions_daisies : 5 * n = 7 * k) :
  (9 * n - 20 - 14 = 2) ∧ (17 * k - 20 - 14 = 0) :=
by
  sorry

end cornflowers_count_l245_245028


namespace divide_into_three_equal_parts_l245_245050

variables {Shape : Type} [has_area Shape] [has_symmetry Shape]

-- The definition for "has_area" and "has_symmetry" would actually typically need to be defined, 
-- but let's assume they are part of Mathlib for simplicity.

-- Conditions of the problem
axiom symmetry_property (s : Shape) : has_symmetry s
axiom height_third (s : Shape) : ∃ h : ℝ, 0 < h ∧ (∃ h1 h2 h3 : Shape,
  have area_eq : has_area h1 ∧ has_area h2 ∧ has_area h3,
  have vertical_cut : can_be_cut_vertically s h1 h2,
  have horizontal_cut : can_be_cut_horizontally s h1 h2 h3))

-- The theorem to prove
theorem divide_into_three_equal_parts (s : Shape) [symmetry_property s] [height_third s]:
  ∃ h1 h2 h3 : Shape, can_be_cut_vertically s h1 h2 ∧ can_be_cut_horizontally s h1 h2 h3 ∧ 
  has_area h1 ∧ has_area h2 ∧ has_area h3 :=
sorry

end divide_into_three_equal_parts_l245_245050


namespace sufficient_but_not_necessary_condition_l245_245068

-- Define the lines
def line1 (m : ℝ) := { p : ℝ × ℝ | p.1 * m + (2 * m - 1) * p.2 + 2 = 0 }
def line2 (m : ℝ) := { p : ℝ × ℝ | 3 * p.1 + m * p.2 + 3 = 0 }

-- Define the condition for perpendicularity
def perpendicular (m : ℝ) := line1 m ⟂ line2 m

-- Statement that m = -1 is a sufficient but not necessary condition for the lines to be perpendicular
theorem sufficient_but_not_necessary_condition (m : ℝ) : (m = -1) → perpendicular m :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_l245_245068


namespace total_amount_spent_l245_245747

-- Define the prices of the CDs
def price_life_journey : ℕ := 100
def price_day_life : ℕ := 50
def price_when_rescind : ℕ := 85

-- Define the discounted price for The Life Journey CD
def discount_life_journey : ℕ := 20 -- 20% discount equivalent to $20
def discounted_price_life_journey : ℕ := price_life_journey - discount_life_journey

-- Define the number of CDs bought
def num_life_journey : ℕ := 3
def num_day_life : ℕ := 4
def num_when_rescind : ℕ := 2

-- Define the function to calculate money spent on each type with offers in consideration
def cost_life_journey : ℕ := num_life_journey * discounted_price_life_journey
def cost_day_life : ℕ := (num_day_life / 2) * price_day_life -- Buy one get one free offer
def cost_when_rescind : ℕ := num_when_rescind * price_when_rescind

-- Calculate the total cost
def total_cost := cost_life_journey + cost_day_life + cost_when_rescind

-- Define Lean theorem to prove the total cost
theorem total_amount_spent : total_cost = 510 :=
  by
    -- Skipping the actual proof as the prompt specifies
    sorry

end total_amount_spent_l245_245747


namespace count_valid_pairs_l245_245143

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 3 ∧ ∀ (m n : ℕ), m > n → n ≥ 4 → (m + n) ≤ 40 → (m - n)^2 = m + n → (m, n) ∈ [(10, 6), (15, 10), (21, 15)] := 
by {
  sorry 
}

end count_valid_pairs_l245_245143


namespace smallest_difference_exists_l245_245498

-- Define the custom addition method used by Vovochka
def vovochka_add (a b : Nat) : Nat := 
  let hundreds := (a / 100) + (b / 100)
  let tens := ((a % 100) / 10) + ((b % 100) / 10)
  let units := (a % 10) + (b % 10)
  hundreds * 1000 + tens * 100 + units

-- Define the standard addition
def std_add (a b : Nat) : Nat := a + b

-- Define the function to compute the difference
def difference (a b : Nat) : Nat :=
  abs (vovochka_add a b - std_add a b)

-- Define the claim
theorem smallest_difference_exists : ∃ a b : Nat, 
  a < 1000 ∧ b < 1000 ∧ a > 99 ∧ b > 99 ∧ 
  difference a b = 1800 := 
sorry

end smallest_difference_exists_l245_245498


namespace tangent_line_m_eq_1_monotonicity_intervals_extremum_values_l245_245727

-- Define the function f for arbitrary m
def f (m : ℝ) (x : ℝ) : ℝ := (m^2 * x) / (x^2 - m)

-- Prove the tangent line equation at point (0,0) when m = 1
theorem tangent_line_m_eq_1 : 
  let f1 := λ (x : ℝ), (1^2 * x) / (x^2 - 1)
  in ∀ (x y : ℝ), y = f1 x → x = 0 → y = 0 → x + y = 0 :=
sorry

-- Prove the monotonicity intervals for the function f based on m
theorem monotonicity_intervals (m : ℝ) (hm : m ≠ 0) :
  let f' := λ (x : ℝ), (m^2 * (-x^2 - m)) / ((x^2 - m)^2)
  in if m < 0 then 
      (∀ x, x < -sqrt (-m) → f' x < 0) ∧
      (∀ x, -sqrt (-m) < x ∧ x < sqrt (-m) → f' x > 0) ∧
      (∀ x, x > sqrt (-m) → f' x < 0)
    else if m > 0 then 
      (∀ x, f' x < 0)
    else false :=
sorry

-- Prove the conditions for extremum values
theorem extremum_values (m : ℝ) (hm : m ≠ 0) :
  (∃ x, let f' := (λ (x : ℝ), (m^2 * (-x^2 - m)) / ((x^2 - m)^2))
       in f' x = 0) → m < 0 :=
sorry

end tangent_line_m_eq_1_monotonicity_intervals_extremum_values_l245_245727


namespace inequality_hold_l245_245970

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end inequality_hold_l245_245970


namespace ab_value_l245_245387

variable (a b : ℝ)

theorem ab_value (h1 : a^5 * b^8 = 12) (h2 : a^8 * b^13 = 18) : a * b = 128 / 3 := 
by 
  sorry

end ab_value_l245_245387


namespace task_allocation_l245_245487

theorem task_allocation (s : Finset ℕ) (h₁ : s.card = 10) (h₂ : s.choose 4).card = 4 : 
  ∃ (a b c : Finset ℕ), a.card = 2 ∧ b.card = 1 ∧ c.card = 1 ∧ 
    (a ∪ b ∪ c = s.choose 4) ∧ 
    ((s.choose 4).choose 2).card * (s.choose 2).erase_fin (a ∪ b).card * 2 = 2520 :=
sorry

end task_allocation_l245_245487


namespace eq_lengths_l245_245796

variable (A B C D E F G : Type*)
variables  [metric_space A] [metric_space B] [metric_space C] 
variables [add_comm_group A] [module ℝ A] [normed_space ℝ A] 
variables [add_comm_group B] [module ℝ B] [normed_space ℝ B] 
variables [add_comm_group C] [module ℝ C] [normed_space ℝ C] 

variables (A B C : Type*) 
variables (BD CE : Prop) 
variables (ED : line A) 
variables (F G : A)

def acute_triangle (A B C : Type*) : Prop :=
  ∀ (α β γ : ℝ), (α + β + γ = π) ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2

def altitude (D E : Type*) : Prop :=
  ∀ (A B : Type*) (line : Type*), ∃ (h : A), A ⊥ B ∧ D ∈ line ∧ E ∈ line

def perp_to (F G : A) (ED : line A) : Prop :=
  ∀ (B C : Type*), (B,F) ⊥ ED ∧ (C,G) ⊥ ED

theorem eq_lengths (ABC acute : A → B → C → Prop) 
  (alt_D alt_E : altitude B D) (alt_E2 : altitude C E) 
  (line_DE : A → (line B C)) (EF DG : A → B) : Prop :=
  ABC → altitude B D → altitude C E → 
  F ∈ line_DE → G ∈ line_DE → 
  perp_to F (line_DE) → perp_to G (line_DE) → 
  EF = DG :=
sorry

end eq_lengths_l245_245796


namespace batsman_average_after_17th_inning_l245_245542

theorem batsman_average_after_17th_inning
  (A : ℕ)  -- average after the 16th inning
  (h1 : 16 * A + 300 = 17 * (A + 10)) :
  A + 10 = 140 :=
by
  sorry

end batsman_average_after_17th_inning_l245_245542


namespace total_students_class_is_63_l245_245409

def num_tables : ℕ := 6
def students_per_table : ℕ := 3
def girls_bathroom : ℕ := 4
def times_canteen : ℕ := 4
def group1_students : ℕ := 4
def group2_students : ℕ := 5
def group3_students : ℕ := 6
def germany_students : ℕ := 2
def france_students : ℕ := 4
def norway_students : ℕ := 3
def italy_students : ℕ := 1

def total_students_in_class : ℕ :=
  (num_tables * students_per_table) +
  girls_bathroom +
  (times_canteen * girls_bathroom) +
  (group1_students + group2_students + group3_students) +
  (germany_students + france_students + norway_students + italy_students)

theorem total_students_class_is_63 : total_students_in_class = 63 :=
  by
    sorry

end total_students_class_is_63_l245_245409


namespace mixed_periodic_fraction_l245_245873

noncomputable def mixed_periodic_decimal (n : ℕ) (h : n > 0) : Prop :=
  let num := 3 * n ^ 2 + 6 * n + 2
  let denom := n * (n + 1) * (n + 2)
  ∃ p : ℕ, p.prime ∧ p ≠ 2 ∧ p ≠ 5 ∧ p ∣ denom

theorem mixed_periodic_fraction (n : ℕ) (h : n > 0) : mixed_periodic_decimal n h :=
sorry

end mixed_periodic_fraction_l245_245873


namespace set_A_not_right_triangle_set_B_is_right_triangle_set_C_is_right_triangle_set_D_is_right_triangle_l245_245180

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem set_A_not_right_triangle :
  ¬ is_right_triangle 4 6 8 :=
by
  simp [is_right_triangle]
  norm_num
  sorry

theorem set_B_is_right_triangle :
  is_right_triangle 5 12 13 :=
by
  simp [is_right_triangle]
  norm_num
  sorry

theorem set_C_is_right_triangle :
  is_right_triangle 6 8 10 :=
by
  simp [is_right_triangle]
  norm_num
  sorry

theorem set_D_is_right_triangle :
  is_right_triangle 7 24 25 :=
by
  simp [is_right_triangle]
  norm_num
  sorry

end set_A_not_right_triangle_set_B_is_right_triangle_set_C_is_right_triangle_set_D_is_right_triangle_l245_245180


namespace product_of_possible_b_l245_245900

theorem product_of_possible_b (b : ℝ) (h1 : ∃ b, (y = 3) ∧ (y = 7) ∧ (x = 2) ∧ (x = b) ∧ 
                              ∃ side_length : ℝ, side_length = 4 ∧ (x = b) = (x = 2) + 4 ∨ (x = b) = (x = 2) - 4): 
  (∃ b1 b2 : ℝ, ((b1 = -2 ∧ b2 = 6) ∨ (b1 = 6 ∧ b2 = -2)) ∧ b1 * b2 = -12) := 
by 
  sorry

end product_of_possible_b_l245_245900


namespace constant_function_condition_l245_245230

theorem constant_function_condition (f : ℝ → ℝ) (h1 : ∀ x, differentiable ℝ f)
  (h2 : ∀ x, differentiable ℝ (deriv f))
  (h3 : ∀ x, deriv (deriv f x) * cos (f x) ≥ (deriv f x)^2 * sin (f x)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k :=
sorry

end constant_function_condition_l245_245230


namespace water_level_at_rim_l245_245249

-- Defining the densities of water and ice
def ρ_воды : ℝ := 1000 -- Density of water in kg/m^3
def ρ_льда : ℝ := 917 -- Density of ice in kg/m^3

-- Defining volumes
variables (V W U : ℝ)

-- Conservation of mass assuming all water converted to ice and vice versa
axiom conservation_of_mass : V * ρ_воды = W * ρ_льда

-- Relating the volumes based on conservation of mass
lemma volume_relation : W = V * (ρ_воды / ρ_льда) :=
by sorry

-- Using Archimedes' principle for floating ice
lemma archimedes_principle : U = V :=
by sorry

theorem water_level_at_rim (initial_volume_filled_to_brim : V) :
  W = V :=
begin
  apply archimedes_principle,
  apply volume_relation,
  assumption
end

end water_level_at_rim_l245_245249


namespace second_runner_stop_time_l245_245494

-- Definitions provided by the conditions
def pace_first := 8 -- pace of the first runner in minutes per mile
def pace_second := 7 -- pace of the second runner in minutes per mile
def time_elapsed := 56 -- time elapsed in minutes before the second runner stops
def distance_first := time_elapsed / pace_first -- distance covered by the first runner in miles
def distance_second := time_elapsed / pace_second -- distance covered by the second runner in miles
def distance_gap := distance_second - distance_first -- gap between the runners in miles

-- Statement of the proof problem
theorem second_runner_stop_time :
  8 = distance_gap * pace_first :=
by
sorry

end second_runner_stop_time_l245_245494


namespace oil_quantity_relationship_l245_245784

variable (Q : ℝ) (t : ℝ)

-- Initial quantity of oil in the tank
def initial_quantity := 40

-- Flow rate of oil out of the tank
def flow_rate := 0.2

-- Function relationship between remaining oil quantity Q and time t
theorem oil_quantity_relationship : Q = initial_quantity - flow_rate * t :=
sorry

end oil_quantity_relationship_l245_245784


namespace josh_paths_to_center_square_l245_245374

-- Definition of the problem's conditions based on given movements and grid size
def num_paths (n : Nat) : Nat :=
  2^(n-1)

-- Main statement
theorem josh_paths_to_center_square (n : Nat) : ∃ p : Nat, p = num_paths n :=
by
  exists num_paths n
  sorry

end josh_paths_to_center_square_l245_245374


namespace part1_part2_l245_245674

noncomputable def f (x : ℝ) : ℝ := abs (x + 20) - abs (16 - x)

theorem part1 (x : ℝ) : f x ≥ 0 ↔ x ≥ -2 := 
by sorry

theorem part2 (m : ℝ) (x_exists : ∃ x : ℝ, f x ≥ m) : m ≤ 36 := 
by sorry

end part1_part2_l245_245674


namespace quadrilateral_perpendicular_and_equal_l245_245794

/-- Given a quadrilateral ABCD with midpoints of sides M_A, M_B, M_C, M_D,
    and perpendicular lines drawn outward from each midpoint with length half of the respective side,
    resulting in points P, Q, R, S.
    Prove that PR ⊥ QS and PR = QS. --/
theorem quadrilateral_perpendicular_and_equal (
  A B C D M_A M_B M_C M_D P Q R S : ℝ×ℝ)
  (h_midA : M_A = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_midB : M_B = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (h_midC : M_C = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
  (h_midD : M_D = ((D.1 + A.1) / 2, (D.2 + A.2) / 2))
  (h_P : P = (M_A.1 + 1/2 * (B.2 - A.2), M_A.2 - 1/2 * (B.1 - A.1)))
  (h_Q : Q = (M_B.1 + 1/2 * (C.2 - B.2), M_B.2 - 1/2 * (C.1 - B.1)))
  (h_R : R = (M_C.1 + 1/2 * (D.2 - C.2), M_C.2 - 1/2 * (D.1 - C.1)))
  (h_S : S = (M_D.1 + 1/2 * (A.2 - D.2), M_D.2 - 1/2 * (A.1 - D.1)))
  : (PR ⟂ QS) ∧ (PR = QS) :=
by sorry

end quadrilateral_perpendicular_and_equal_l245_245794


namespace maximize_x3y4_correct_l245_245398

noncomputable def maximize_x3y4 : ℝ × ℝ :=
  let x := 160 / 7
  let y := 120 / 7
  (x, y)

theorem maximize_x3y4_correct :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 40 ∧ (x, y) = maximize_x3y4 ∧ 
  ∀ (x' y' : ℝ), 0 < x' ∧ 0 < y' ∧ x' + y' = 40 → x ^ 3 * y ^ 4 ≥ x' ^ 3 * y' ^ 4 :=
by
  sorry

end maximize_x3y4_correct_l245_245398


namespace inequality_proof_l245_245986

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l245_245986


namespace matrix_operation_l245_245394

variable {R : Type*} [CommRing R]
variable {M : Type*} [AddCommGroup M] [Module R M]
variable (v w : M)
variable (Mv : M) (Mw : M) (a b : R)

-- Conditions
def cond1 : Prop := Mv = ⟨2, -3⟩
def cond2 : Prop := Mw = ⟨-1, 4⟩

-- Goal
theorem matrix_operation : cond1 → cond2 → 
  M ⟨-3 * a + 2 * b, -3 * b - 6⟩ = ⟨-8, 17⟩ := by
  sorry

end matrix_operation_l245_245394


namespace unusual_digits_exists_l245_245557

def is_unusual (n : ℕ) : Prop :=
  let len := n.digits.count;
  let high_power := 10 ^ len;
  (n^3 % high_power = n) ∧ (n^2 % high_power ≠ n)

theorem unusual_digits_exists :
  ∃ n1 n2 : ℕ, (n1 ≥ 10^99 ∧ n1 < 10^100 ∧ is_unusual n1) ∧ 
             (n2 ≥ 10^99 ∧ n2 < 10^100 ∧ is_unusual n2) ∧
             (n1 ≠ n2) :=
by
  let n1 := 10^100 - 1;
  let n2 := (10^100 / 2) - 1;
  use n1, n2;
  sorry

end unusual_digits_exists_l245_245557


namespace hemisphere_surface_area_l245_245911

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (hπ : π = Real.pi) (h : π * r^2 = 3) :
    2 * π * r^2 + 3 = 9 :=
by
  sorry

end hemisphere_surface_area_l245_245911


namespace distinct_positive_integers_exists_l245_245303

theorem distinct_positive_integers_exists 
(n : ℕ)
(a b : ℕ)
(h1 : a ≠ b)
(h2 : b % a = 0)
(h3 : a > 10^(2 * n - 1) ∧ a < 10^(2 * n))
(h4 : b > 10^(2 * n - 1) ∧ b < 10^(2 * n))
(h5 : ∀ x y : ℕ, a = 10^n * x + y ∧ b = 10^n * y + x ∧ x < y ∧ x / 10^(n - 1) ≠ 0 ∧ y / 10^(n - 1) ≠ 0) :
a = (10^(2 * n) - 1) / 7 ∧ b = 6 * (10^(2 * n) - 1) / 7 := 
by
  sorry

end distinct_positive_integers_exists_l245_245303


namespace conjugate_complex_theorem_l245_245444

noncomputable def conjugate_of_fraction_complex : Prop :=
  let z := (1 + 2 * Complex.i) / (3 - 4 * Complex.i)
  Complex.conj z = -1/5 - 2/5 * Complex.i

theorem conjugate_complex_theorem : conjugate_of_fraction_complex := by
  sorry

end conjugate_complex_theorem_l245_245444


namespace find_value_of_ratio_l245_245403

theorem find_value_of_ratio (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x / y + y / x = 4) :
  (x + 2 * y) / (x - 2 * y) = Real.sqrt 33 / 3 := 
  sorry

end find_value_of_ratio_l245_245403


namespace conditions_iff_positive_l245_245698

theorem conditions_iff_positive (a b : ℝ) (h₁ : a + b > 0) (h₂ : ab > 0) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ ab > 0) :=
sorry

end conditions_iff_positive_l245_245698


namespace jenny_hours_left_l245_245353

theorem jenny_hours_left : 
  ∀ (total_hours research_hours proposal_hours : ℕ), 
  total_hours = 20 → 
  research_hours = 10 → 
  proposal_hours = 2 → 
  total_hours - (research_hours + proposal_hours) = 8 :=
by
  intros total_hours research_hours proposal_hours h_total h_research h_proposal
  rw [h_total, h_research, h_proposal]
  norm_num

end jenny_hours_left_l245_245353


namespace smallest_k_for_ball_arrangement_l245_245932

theorem smallest_k_for_ball_arrangement :
  ∃ k : ℕ, k = 6 ∧ ∀ (balls : ℤ) (bags : ℤ), 
  (balls = 5040) → 
  (bags = 2520) → 
  ∀ (c : ℕ), 
  (c = balls / k) →
  ∀ (b : ℕ → ℕ → Prop),
  (∀ (i j: ℕ), i ≠ j → b i j) →
  ∀ (arrange : ℕ → Prop),
  (∀ (i j: ℕ), i ≠ j → arrange i j) :=
by
  sorry

end smallest_k_for_ball_arrangement_l245_245932


namespace series_sum_l245_245907

noncomputable def x : ℕ → ℕ 
| 0 := 25
| (n + 1) := (x n) ^ 2 + (x n)

theorem series_sum (s : ℕ → ℕ) (h₀ : s 1 = 25) 
                   (h₁ : ∀ k ≥ 2, s k = s (k - 1) ^ 2 + s (k - 1)) :
                   (∑' (n : ℕ), 1 / (s n + 1)) = 1 / 25 := 
by 
  sorry

end series_sum_l245_245907


namespace part_I_proof_part_II_proof_part_III_proof_l245_245739

noncomputable theory
open Real

variables {a b : ℝ^3} -- Assuming 3-dimensional vectors

-- Part I
def part_I (a b : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = sqrt 2) (angle_ab : Real.angle a b = π / 3) : Real :=
  dot_product a b

theorem part_I_proof (a b : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = sqrt 2) (angle_ab : Real.angle a b = π / 3) :
  part_I a b ha hb angle_ab = sqrt 2 / 2 := by
  sorry

-- Part II
def part_II (a b : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = sqrt 2) (norm_sum_ab : ‖a + b‖ = sqrt 5) : Real :=
  dot_product a b

theorem part_II_proof (a b : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = sqrt 2) (norm_sum_ab : ‖a + b‖ = sqrt 5) :
  part_II a b ha hb norm_sum_ab = 1 := by
  sorry

-- Part III
def part_III (a b : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = sqrt 2) (dot_diff_is_zero : dot_product a (a - b) = 0) : Real :=
  Real.angle a b

theorem part_III_proof (a b : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = sqrt 2) (dot_diff_is_zero : dot_product a (a - b) = 0) :
  part_III a b ha hb dot_diff_is_zero = π / 4 := by
  sorry

end part_I_proof_part_II_proof_part_III_proof_l245_245739


namespace pastries_total_l245_245196

-- We start by defining the conditions
def Calvin_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Phoebe_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Grace_pastries : ℕ := 30
def have_same_pastries (Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : Prop :=
  Calvin_pastries + 5 = Grace_pastries ∧ Phoebe_pastries + 5 = Grace_pastries

-- Total number of pastries held by Calvin, Phoebe, Frank, and Grace
def total_pastries (Frank_pastries Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : ℕ :=
  Frank_pastries + Calvin_pastries + Phoebe_pastries + Grace_pastries

-- The statement to prove
theorem pastries_total (Frank_pastries : ℕ) : 
  have_same_pastries (Calvin_pastries Frank_pastries Grace_pastries) (Phoebe_pastries Frank_pastries Grace_pastries) Grace_pastries → 
  Frank_pastries + Calvin_pastries Frank_pastries Grace_pastries + Phoebe_pastries Frank_pastries Grace_pastries + Grace_pastries = 97 :=
by
  sorry

end pastries_total_l245_245196


namespace triangle_cosine_sine_identity_l245_245327

theorem triangle_cosine_sine_identity
  (PQ PR QR : ℝ) (P Q R : ℝ)
  (h_PQ : PQ = 8)
  (h_PR : PR = 10)
  (h_QR : QR = 6)
  (h_sum_angles : P + Q + R = 180) 
  : (cos ((P - Q) / 2) / sin (R / 2)) - (sin ((P - Q) / 2) / cos (R / 2)) = (6:ℝ) / 5 := 
by
  sorry

end triangle_cosine_sine_identity_l245_245327


namespace correct_sentence_is_D_l245_245953

-- Define the sentences as strings
def sentence_A : String :=
  "Between any two adjacent integers on the number line, an infinite number of fractions can be inserted to fill the gaps on the number line; mathematicians once thought that with this approach, the entire number line was finally filled."

def sentence_B : String :=
  "With zero as the center, all integers are arranged from right to left at equal distances, and then connected with a horizontal line; this is what we call the 'number line'."

def sentence_C : String :=
  "The vast collection of books in the Beijing Library contains an enormous amount of information, but it is still finite, whereas the number pi contains infinite information, which is awe-inspiring."

def sentence_D : String :=
  "Pi is fundamentally the exact ratio of a circle's circumference to its diameter, but the infinite sequence it produces has the greatest uncertainty; we cannot help but be amazed and shaken by the marvel and mystery of nature."

-- Define the problem statement
theorem correct_sentence_is_D :
  sentence_D ≠ "" := by
  sorry

end correct_sentence_is_D_l245_245953


namespace batsman_average_after_17th_l245_245127

variable (A : ℝ)
variable (runs_17th_inning new_average : ℝ)

-- Conditions
def condition_1 : Prop := runs_17th_inning = 66
def condition_2 : Prop := new_average = A + 3

-- Main goal
theorem batsman_average_after_17th
  (h₁ : condition_1)
  (h₂ : condition_2)
  (h₃ : 16 * A + runs_17th_inning = 17 * new_average) : new_average = 18 := by
  sorry

end batsman_average_after_17th_l245_245127


namespace line_passes_through_fixed_point_points_distance_from_line_line_intersects_at_single_point_common_tangents_three_l245_245255

noncomputable def circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
noncomputable def line (m x y : ℝ) : Prop := (m + 1) * x + 2 * y - 1 + m = 0

theorem line_passes_through_fixed_point :
  ∀ m, line m (-1) 1 :=
sorry

theorem points_distance_from_line :
  ¬∃ p1 p2 p3, (circle p1.1 p1.2) ∧ (circle p2.1 p2.2) ∧ (circle p3.1 p3.2) ∧ 
               dist_from_line p1 = 1 ∧ dist_from_line p2 = 1 ∧ dist_from_line p3 = 1 :=
sorry

theorem line_intersects_at_single_point :
  ¬ ∃ p, circle p.1 p.2 ∧ line m p.1 p.2 :=
sorry

theorem common_tangents_three (a : ℝ) :
  (∀ t, tangent_circle (x^2 + y^2 - 2*x + 8*y + a = 0) t ∧ tangent_circle ((x+2)^2 + y^2 = 4) t) 
  ↔ a = 8 :=
sorry

end line_passes_through_fixed_point_points_distance_from_line_line_intersects_at_single_point_common_tangents_three_l245_245255


namespace sabrina_total_leaves_l245_245047

theorem sabrina_total_leaves (num_basil num_sage num_verbena : ℕ)
  (h1 : num_basil = 2 * num_sage)
  (h2 : num_sage = num_verbena - 5)
  (h3 : num_basil = 12) :
  num_sage + num_verbena + num_basil = 29 :=
by
  sorry

end sabrina_total_leaves_l245_245047


namespace coat_price_calculation_l245_245553

noncomputable def effective_price (initial_price : ℝ) 
  (reduction1 reduction2 reduction3 : ℝ) 
  (tax1 tax2 tax3 : ℝ) : ℝ :=
  let price_after_first_month := initial_price * (1 - reduction1 / 100) * (1 + tax1 / 100)
  let price_after_second_month := price_after_first_month * (1 - reduction2 / 100) * (1 + tax2 / 100)
  let price_after_third_month := price_after_second_month * (1 - reduction3 / 100) * (1 + tax3 / 100)
  price_after_third_month

noncomputable def total_percent_reduction (initial_price final_price : ℝ) : ℝ :=
  (initial_price - final_price) / initial_price * 100

theorem coat_price_calculation :
  let original_price := 500
  let price_final := effective_price original_price 10 15 20 5 8 6
  let reduction_percentage := total_percent_reduction original_price price_final
  price_final = 367.824 ∧ reduction_percentage = 26.44 :=
by
  sorry

end coat_price_calculation_l245_245553


namespace wives_identification_l245_245000

theorem wives_identification (Anna Betty Carol Dorothy MrBrown MrGreen MrWhite MrSmith : ℕ):
  Anna = 2 ∧ Betty = 3 ∧ Carol = 4 ∧ Dorothy = 5 ∧
  (MrBrown = Dorothy ∧ MrGreen = 2 * Carol ∧ MrWhite = 3 * Betty ∧ MrSmith = 4 * Anna) ∧
  (Anna + Betty + Carol + Dorothy + MrBrown + MrGreen + MrWhite + MrSmith = 44) →
  (
    Dorothy = 5 ∧
    Carol = 4 ∧
    Betty = 3 ∧
    Anna = 2 ∧
    MrBrown = 5 ∧
    MrGreen = 8 ∧
    MrWhite = 9 ∧
    MrSmith = 8
  ) :=
by
  intros
  sorry

end wives_identification_l245_245000


namespace find_u_exists_l245_245822

def vector_a : ℝ × ℝ × ℝ := (5, 0, -3)
def vector_b : ℝ × ℝ × ℝ := (2, 3, -1)
def vector_c : ℝ × ℝ × ℝ := (7, 9, -10)

noncomputable def cross_product (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x₁, y₁, z₁) := v₁
  let (x₂, y₂, z₂) := v₂
  (y₁ * z₂ - z₁ * y₂, z₁ * x₂ - x₁ * z₂, x₁ * y₂ - y₁ * x₂)

noncomputable def a_cross_b : ℝ × ℝ × ℝ := cross_product vector_a vector_b

theorem find_u_exists (s t u : ℝ) :
  vector_c = (s * vector_a.1 + t * vector_b.1 + u * a_cross_b.1,
              s * vector_a.2 + t * vector_b.2 + u * a_cross_b.2,
              s * vector_a.3 + t * vector_b.3 + u * a_cross_b.3) →
  u = -222 / 307 :=
by
  sorry

end find_u_exists_l245_245822


namespace N_is_perfect_square_l245_245819

def N (n : ℕ) : ℕ :=
  (10^(2*n+1) - 1) / 9 * 10 + 
  2 * (10^(n+1) - 1) / 9 + 25

theorem N_is_perfect_square (n : ℕ) : ∃ k, k^2 = N n :=
  sorry

end N_is_perfect_square_l245_245819


namespace postage_for_78_5g_l245_245120

noncomputable def postage (weight : ℝ) : ℝ :=
if weight ≤ 20 then 0.8
else if weight ≤ 40 then 1.6
else if weight ≤ 60 then 2.4
else if weight ≤ 80 then 3.2
else 0 -- This case is not used since weight is within 100g constraint

theorem postage_for_78_5g : postage 78.5 = 3.2 :=
by
  unfold postage
  split_ifs
  sorry

end postage_for_78_5g_l245_245120


namespace no_solution_exists_l245_245633

def product_of_digits (x : ℕ) : ℕ :=
  if x < 10 then x else (x / 10) * (x % 10)

theorem no_solution_exists :
  ¬ ∃ x : ℕ, product_of_digits x = x^2 - 10 * x - 22 :=
by
  sorry

end no_solution_exists_l245_245633


namespace num_shelves_l245_245157

noncomputable def number_of_shelves (movies : ℕ) (extra_movie_needed : ℕ) : ℕ :=
  let factors := [1, 3, 9]
  factors.filter (λ n, (movies + 1) % n = 0).head! + extra_movie_needed

theorem num_shelves (h_movies : ∃ n, number_of_shelves 9 1 = n) : number_of_shelves 9 1 = 4 :=
  by sorry

end num_shelves_l245_245157


namespace sum_of_digits_of_square_of_11111111_l245_245117

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_digits_of_square_of_11111111 :
  let X := 11111111 in digit_sum (X * X) = 64 :=
by
  let X := 11111111
  have : digit_sum (X * X) = 64 := sorry
  exact this

end sum_of_digits_of_square_of_11111111_l245_245117


namespace problem_proof_l245_245712

-- Define the sequence a_n being an increasing geometric sequence
def is_increasing_geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * 4 ∧ a n < a (n + 1)

-- Define the conditions for a_1 and a_3
def conditions (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 = 17 ∧ a 1 * a 3 = 16

-- General formula for the sequence
def general_formula (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = 4^(n-1)

-- Define b_n and the sum of the absolute values of the first n terms of b_n
def b_n (a : ℕ → ℝ) (n : ℕ) : ℝ := log (1/2 : ℝ) (a n) + 11

def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, |b_n a i|)

-- Combining everything in the final theorem statement 
theorem problem_proof (a : ℕ → ℝ) :
  is_increasing_geom_seq a →
  conditions a →
  general_formula a →
  ∀ n, (n ≤ 6 → T_n a n = 12 * n - n^2) ∧ (n ≥ 7 → T_n a n = n^2 - 12 * n + 72) :=
by 
  intros h_increasing h_conditions h_general n
  sorry

end problem_proof_l245_245712


namespace probability_of_white_ball_l245_245144

noncomputable def total_balls_initial := 18
noncomputable def white_balls_initial := 8
noncomputable def black_balls_initial := 10
noncomputable def balls_removed := 2
noncomputable def balls_left := total_balls_initial - balls_removed

-- Scenario probabilities calculations
noncomputable def prob_two_different_colors :=
  (white_balls_initial.toFloat / total_balls_initial) * (black_balls_initial.toFloat / (total_balls_initial - 1)) +
  (black_balls_initial.toFloat / total_balls_initial) * (white_balls_initial.toFloat / (total_balls_initial - 1))

noncomputable def prob_both_white :=
  (white_balls_initial.toFloat / total_balls_initial) * ((white_balls_initial - 1).toFloat / (total_balls_initial - 1))

noncomputable def prob_both_black :=
  (black_balls_initial.toFloat / total_balls_initial) * ((black_balls_initial - 1).toFloat / (total_balls_initial - 1))

noncomputable def prob_same_color := prob_both_white + prob_both_black

-- Conditional probabilities of drawing a white ball
noncomputable def prob_white_draw_different_colors :=
  (white_balls_initial - 1).toFloat / balls_left

noncomputable def prob_white_draw_both_white :=
  (white_balls_initial - 2).toFloat / balls_left

noncomputable def prob_white_draw_both_black :=
  white_balls_initial.toFloat / balls_left

noncomputable def overall_prob_white :=
  prob_two_different_colors * prob_white_draw_different_colors +
  prob_both_white * prob_white_draw_both_white +
  prob_both_black * prob_white_draw_both_black

theorem probability_of_white_ball :
  overall_prob_white = 37 / 98 := sorry

end probability_of_white_ball_l245_245144


namespace total_people_l245_245335

/-- In a group, 90 people are more than 30 years old, and the probability that
a randomly chosen person is less than 20 years old is 0.25. Prove that the total number
of people in the group is 120. -/
theorem total_people (T : ℕ) (L : ℕ)
  (h1 : L / T = 0.25)
  (h2 : L + 90 = T) :
  T = 120 :=
by
  sorry

end total_people_l245_245335


namespace remove_terms_sum_l245_245527

theorem remove_terms_sum :
  let s := (1/3 + 1/5 + 1/7 + 1/9 + 1/11 + 1/13 + 1/15 : ℚ)
  s = 16339/15015 →
  (1/13 + 1/15 = 2061/5005) →
  s - (1/13 + 1/15) = 3/2 :=
by
  intros s hs hremove
  have hrem : (s - (1/13 + 1/15 : ℚ) = 3/2) ↔ (16339/15015 - 2061/5005 = 3/2) := sorry
  exact hrem.mpr sorry

end remove_terms_sum_l245_245527


namespace greatest_integer_a_l245_245231

-- Define formal properties and state the main theorem.
theorem greatest_integer_a (a : ℤ) : (∀ x : ℝ, ¬(x^2 + (a:ℝ) * x + 15 = 0)) → (a ≤ 7) :=
by
  intro h
  sorry

end greatest_integer_a_l245_245231


namespace Lucia_first_day_messages_l245_245174

-- Definitions based on conditions
def messages_sent_by_Lucia_on_first_day : Nat
def messages_sent_by_Alina_on_first_day := messages_sent_by_Lucia_on_first_day - 20
def messages_sent_by_Lucia_on_second_day := (1 / 3) * messages_sent_by_Lucia_on_first_day
def messages_sent_by_Alina_on_second_day := 2 * (messages_sent_by_Lucia_on_first_day - 20)
def messages_sent_by_Lucia_on_third_day := messages_sent_by_Lucia_on_first_day
def messages_sent_by_Alina_on_third_day := messages_sent_by_Alina_on_first_day

def total_messages :=
  messages_sent_by_Lucia_on_first_day + messages_sent_by_Alina_on_first_day +
  messages_sent_by_Lucia_on_second_day + messages_sent_by_Alina_on_second_day +
  messages_sent_by_Lucia_on_third_day + messages_sent_by_Alina_on_third_day

theorem Lucia_first_day_messages :
  680 = total_messages → messages_sent_by_Lucia_on_first_day = 120 := by
  sorry

end Lucia_first_day_messages_l245_245174


namespace decaf_coffee_percentage_l245_245956

variable (initial_stock new_stock : ℕ)
variable (percent_decaf_initial percent_decaf_new : ℝ)

def decaf_percent (initial_stock new_stock : ℕ) (percent_decaf_initial percent_decaf_new : ℝ) : ℝ :=
  let decaf_initial := (percent_decaf_initial / 100) * initial_stock
  let decaf_new := (percent_decaf_new / 100) * new_stock
  let total_decaf := decaf_initial + decaf_new
  let total_stock := initial_stock + new_stock
  (total_decaf / total_stock) * 100

theorem decaf_coffee_percentage :
  decaf_percent 400 100 30 60 = 36 :=
by sorry

end decaf_coffee_percentage_l245_245956


namespace equation_of_line_passing_through_and_between_l245_245572

theorem equation_of_line_passing_through_and_between (L L1 L2 : Setₓ (ℝ × ℝ)) (x y : ℝ) :
  L = {p | 4 * p.1 - 5 * p.2 + 7 = 0} ∧
  ({p | 2 * p.1 - 5 * p.2 + 9 = 0} = L1 ∧ {p | 2 * p.1 - 5 * p.2 - 7 = 0} = L2) ∧
  (∃ (A B : ℝ × ℝ), A ∈ L ∧ B ∈ L ∧ (A.1 + B.1) / 2 = x ∧ (A.2 + B.2) / 2 = y ∧ (x, y) ∈ {p | p.1 - 4 * p.2 - 1 = 0} ∧ (2, 3) ∈ L) :=
sorry

end equation_of_line_passing_through_and_between_l245_245572


namespace min_distance_between_M_and_N_l245_245269

noncomputable def f (x : ℝ) := Real.sin x + (1 / 6) * x^3
noncomputable def g (x : ℝ) := x - 1

theorem min_distance_between_M_and_N :
  ∃ (x1 x2 : ℝ), x1 ≥ 0 ∧ x2 ≥ 0 ∧ f x1 = g x2 ∧ (x2 - x1 = 1) :=
sorry

end min_distance_between_M_and_N_l245_245269


namespace smallest_sum_of_squares_l245_245069

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 145) : 
  ∃ x y, x^2 - y^2 = 145 ∧ x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l245_245069


namespace triangle_inequality_inequality_l245_245839

variable {a b c : ℝ}
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (triangle_ineq : a + b > c)

theorem triangle_inequality_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) (triangle_ineq : a + b > c) :
  a^3 + b^3 + 3 * a * b * c > c^3 :=
sorry

end triangle_inequality_inequality_l245_245839


namespace exists_two_unusual_numbers_l245_245561

noncomputable def is_unusual (n : ℕ) : Prop :=
  (n ^ 3 % 10 ^ 100 = n) ∧ (n ^ 2 % 10 ^ 100 ≠ n)

theorem exists_two_unusual_numbers :
  ∃ n1 n2 : ℕ, (is_unusual n1) ∧ (is_unusual n2) ∧ (n1 ≠ n2) ∧ (n1 >= 10 ^ 99) ∧ (n1 < 10 ^ 100) ∧ (n2 >= 10 ^ 99) ∧ (n2 < 10 ^ 100) :=
begin
  sorry
end

end exists_two_unusual_numbers_l245_245561


namespace combination_sum_l245_245138

-- Definition of combination, also known as binomial coefficient
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem statement
theorem combination_sum :
  (combination 8 2) + (combination 8 3) = 84 :=
by
  sorry

end combination_sum_l245_245138


namespace geometric_locus_of_M_l245_245547

noncomputable def circle (center : Point) (radius : ℝ) : Set Point := sorry

axiom Point : Type

variables 
  (O N : Point) 
  (R : ℝ) 
  (circle_O : circle O R) 
  (A B : Point) 
  (chord_AB : A ∈ circle O R ∧ B ∈ circle O R) 
  (M : Point) 
  (tangent_N : tangent_at N (circle O R))
  (intersection_M : M = line_through A B ∩ tangent_N)

theorem geometric_locus_of_M :
  ∀ (M : Point), 
  (M = line_through A B ∩ tangent_at N (circle O R)) → 
  M ∈ perpendicular_line O N := 
sorry

end geometric_locus_of_M_l245_245547


namespace graph_passes_quadrants_l245_245729

theorem graph_passes_quadrants (k : ℝ) : 
  (∀ (x : ℝ), (-k^2 - 2) < 0 → (y : ℝ), y = (-k^2 - 2) * x → 
  ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) :=
by
  sorry

end graph_passes_quadrants_l245_245729


namespace ella_stamps_value_l245_245641

theorem ella_stamps_value :
  let total_stamps := 18
  let value_of_6_stamps := 18
  let consistent_value_per_stamp := value_of_6_stamps / 6
  total_stamps * consistent_value_per_stamp = 54 := by
  sorry

end ella_stamps_value_l245_245641


namespace greatest_prime_factor_391_l245_245108

theorem greatest_prime_factor_391 : ∃ p, prime p ∧ p ∣ 391 ∧ ∀ q, prime q ∧ q ∣ 391 → q ∣ p := by
  sorry

end greatest_prime_factor_391_l245_245108


namespace train_speed_l245_245589

/--A train leaves Delhi at 9 a.m. at a speed of 30 kmph.
Another train leaves at 3 p.m. on the same day and in the same direction.
The two trains meet 720 km away from Delhi.
Prove that the speed of the second train is 120 kmph.-/
theorem train_speed
  (speed_first_train speed_first_kmph : 30 = 30)
  (leave_first_train : Nat)
  (leave_first_9am : 9 = 9)
  (leave_second_train : Nat)
  (leave_second_3pm : 3 = 3)
  (distance_meeting_km : Nat)
  (distance_meeting_720km : 720 = 720) :
  ∃ speed_second_train, speed_second_train = 120 := 
sorry

end train_speed_l245_245589


namespace other_group_land_l245_245591

def total_land : ℕ := 900
def remaining_land : ℕ := 385
def lizzies_group_land : ℕ := 250

theorem other_group_land :
  total_land - remaining_land - lizzies_group_land = 265 :=
by
  sorry

end other_group_land_l245_245591


namespace midpoint_AB_is_C_l245_245250

section MidpointProof
variable {α : Type*} [linear_ordered_field α]

-- Definitions based on the given conditions
def point (a b : α) : α × α := (a, b)

def vec (a b c d : α) : α × α := (c - a, d - b)

def midpoint (p1 p2 : α × α) : α × α :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Coefficients of Points and Vectors
def A : α × α := point (-3) 2
def AB : α × α := vec (-3) 2 3 2 -- B is calculated as (3, 2)

-- The Proof Goal
theorem midpoint_AB_is_C :
  midpoint A (3, 2) = (0, 2) :=
by
  -- We skip the proof as per the instructions
  sorry

end MidpointProof

end midpoint_AB_is_C_l245_245250


namespace josh_paths_to_top_center_l245_245371

/-- Define the grid --/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Josh's initial position --/
def start_pos : (ℕ × ℕ) := (0, 0)

/-- Define a function representing Josh's movement possibilities --/
def move_right (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1, pos.2 + 1)

def move_left_up (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1 + 1, pos.2 - 1)

/-- Define the goal position --/
def goal_pos (n : ℕ) : (ℕ × ℕ) :=
  (n - 1, 1)

/-- Theorem stating the required proof --/
theorem josh_paths_to_top_center {n : ℕ} (h : n > 0) : 
  let g := Grid.mk n 3 in
  ∃ (paths : ℕ), paths = 2^(n - 1) := 
  sorry

end josh_paths_to_top_center_l245_245371


namespace calc_expr_solve_fractional_eq_l245_245534

-- Problem 1: Calculate the expression
theorem calc_expr : (-2)^2 - (64:ℝ)^(1/3) + (-3)^0 - (1/3)^0 = 0 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

-- Problem 2: Solve the fractional equation
theorem solve_fractional_eq (x : ℝ) (h : x ≠ -1) : 
  (x / (x + 1) = 5 / (2 * x + 2) - 1) ↔ x = 3 / 4 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

end calc_expr_solve_fractional_eq_l245_245534


namespace complementary_angle_percentage_decrease_l245_245474

theorem complementary_angle_percentage_decrease :
  ∀ (α β : ℝ), α + β = 90 → 6 * α = 3 * β → 
  let α' := 1.2 * α in
  let β' := 90 - α' in
  (100 * (β' / β)) = 90 :=
by
  intros α β h_sum h_ratio α' β'
  have α' := 1.2 * α
  have β' := 90 - α'
  sorry

end complementary_angle_percentage_decrease_l245_245474


namespace inequality_hold_l245_245972

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end inequality_hold_l245_245972


namespace ana_has_15_dresses_l245_245406

noncomputable def ana_dresses (x y : ℕ) : Prop :=
  y = x + 18 ∧ x + y = 48 → x = 15

-- Now state the theorem
theorem ana_has_15_dresses : ∃ (x y : ℕ), y = x + 18 ∧ x + y = 48 ∧ x = 15 :=
by
  use 15
  use 33  -- y = 15 + 18
  split
  { exact rfl }
  split
  { norm_num }
  { exact rfl }

end ana_has_15_dresses_l245_245406


namespace josh_walk_ways_l245_245363

theorem josh_walk_ways (n : ℕ) :
  let grid_rows := n
      grid_columns := 3
      start_position := (0, 0)  -- (row, column) starting from bottom left
  in grid_rows > 0 →
      let center_square (k : ℕ) := (k, 1) -- center square of k-th row
  in ∃ ways : ℕ, ways = 2^(n-1) ∧
                ways = count_paths_to_center_topmost n  -- custom function representation
sorry

end josh_walk_ways_l245_245363


namespace part1_part2_l245_245285

open Nat

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, r       => 0
| n+1, r+1   => binom n r + binom n (r + 1)

def S (n r : ℕ) : ℕ :=
(let num := n + 1 - 2 * r
let denom := n + 1 - r
if denom = 0 then 0 else num / denom * binom n r)

def sum_S (n : ℕ) : ℕ :=
∑ r in range (n / 2 + 1), S n r

theorem part1 (n r : ℕ) (h1 : r ≤ n) : Int (S n r) :=
by
  sorry

theorem part2 (n : ℕ) (h2 : n ≥ 9) : sum_S n < 2 ^ (n - 2) :=
by
  sorry

end part1_part2_l245_245285


namespace remaining_oil_quantity_check_remaining_oil_quantity_l245_245779

def initial_oil_quantity : Real := 40
def outflow_rate : Real := 0.2

theorem remaining_oil_quantity (t : Real) : Real :=
  initial_oil_quantity - outflow_rate * t

theorem check_remaining_oil_quantity (t : Real) : remaining_oil_quantity t = 40 - 0.2 * t := 
by 
  sorry

end remaining_oil_quantity_check_remaining_oil_quantity_l245_245779


namespace area_of_rhombus_l245_245936

-- Given definitions from problem conditions
def side_length := Real.sqrt 145
def diagonal_diff := 10

theorem area_of_rhombus : 
  ∃ d₁ d₂ : ℝ, (d₁ - d₂ = diagonal_diff ∧ 
  d₁ / 2 = side_length ∧ 
  d₂ / 2 = side_length ∧ 
  (1/2) * d₁ * d₂ = 208) :=
by
  sorry

end area_of_rhombus_l245_245936


namespace f_1997_leq_666_l245_245835

noncomputable def f : ℕ+ → ℕ := sorry

axiom f_mn_inequality : ∀ (m n : ℕ+), f (m + n) ≥ f m + f n
axiom f_two : f 2 = 0
axiom f_three_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

theorem f_1997_leq_666 : f 1997 ≤ 666 := sorry

end f_1997_leq_666_l245_245835


namespace locus_P_is_hyperbola_l245_245268

-- Definitions that directly appear in the conditions
variables {A B E F P : Point} {e : Ellipse A B E F} -- Ellipse with endpoints A, B and foci E, F
variable (P : Point) -- Point in space

-- Define the conditions
def is_ellipse (e : Ellipse A B E F) : Prop := e.foci = (E, F) ∧ e.major_axis_ends = (A, B)
def is_right_circular_cone (P : Point) (e : Ellipse A B E F) : Prop := 
  ∃ (DandelinSphere₁ DandelinSphere₂ : Sphere), 
    DandelinSphere₁.tangent_to_cone P ∧ DandelinSphere₂.tangent_to_cone P ∧ 
    DandelinSphere₁.tangent_to_plane e ∧ DandelinSphere₂.tangent_to_plane e ∧
    DandelinSphere₁.tangent_point_on_plane = E ∧ DandelinSphere₂.tangent_point_on_plane = F

-- The final answer we want to prove
theorem locus_P_is_hyperbola (e : Ellipse A B E F) (P : Point) :
  is_ellipse e →
  (is_right_circular_cone P e) →
  ∃ (h : Hyperbola), h.foci = (A, B) ∧ h.major_axis_ends = E ∨ h.major_axis_ends = F :=
by 
  intros h_is_ellipse h_cone_is_right_circular
  sorry

end locus_P_is_hyperbola_l245_245268


namespace find_ratio_l245_245388

noncomputable def problem_statement : Prop :=
  ∃ (AB BC : ℝ) (P : ℝ × ℝ), 
    AB * BC = 18 ∧ 
    (∃ (h1 h3 : ℝ), (1 / 2) * AB * h1 = 3 ∧ (1 / 2) * AB * h3 = 6) ∧ 
    (∃ (h2 h4 : ℝ), (1 / 2) * BC * h2 = 4 ∧ (1 / 2) * BC * h4 = 5) ∧ 
    2 * (h1 + h3) = 3 * (h2 + h4) ∧
    AB / BC = 2

theorem find_ratio : problem_statement :=
  sorry

end find_ratio_l245_245388


namespace area_of_triangle_l245_245650

-- Define the triangle DEF in the plane
structure Triangle :=
  (D E F : ℝ × ℝ)
  (right_angle : E.1 - D.1 = E.2 - F.2)
  (DF : (F.2 - D.2) = 8)
  (angle_EDF : ∃ θ : ℝ, θ = Real.arcsin((F.2 - D.2) / (f E)) out_f (DF ) = 2 ∧ θ = π / 6)

-- Theorem: The area of triangle DEF is 32√3
theorem area_of_triangle {DEF : Triangle} (h : DEF.angle_EDF = π / 6) :
  triangle_area DEF.D DEF.E DEF.F = 32 * Real.sqrt 3 :=
sorry

end area_of_triangle_l245_245650


namespace water_level_after_ice_melts_l245_245246

-- Definitions for water density and ice density
def ρвода : ℝ := 1000  -- density of water in kg/m^3
def ρльда : ℝ := 917   -- density of ice in kg/m^3

-- Conditions for conservation of mass and volume displacement
theorem water_level_after_ice_melts (V : ℝ) (W : ℝ) (ρвода ρльда: ℝ) 
  (Hρвода : ρвода = 1000) (Hρльда : ρльда = 917)
  (Hmass : V * ρвода = W * ρльда) 
  (Hvol : V * ρвода = W * ρльда) : 
  V = W := 
sorry

end water_level_after_ice_melts_l245_245246


namespace cube_vertex_selection_l245_245245

theorem cube_vertex_selection (V : Finset (Fin 8)) (hV : V.card = 4) :
  (∃ A B C D : Fin 8, A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
    A ≠ C ∧ B ≠ D ∧
    -- Rectangle condition
    ((A, B, C, D).rectangle) ∨
    -- Tetrahedron with specified triangle faces condition
    ((A, B, C, D).tetrahedron_with_isosceles_and_equilateral) ∨
    -- Tetrahedron with equilateral faces condition
    ((A, B, C, D).tetrahedron_with_equilateral_faces))
:=
sorry

end cube_vertex_selection_l245_245245


namespace modulus_Z1_correct_l245_245696

def Z1 : ℂ := 1 + complex.I * (-2)
def modulus_Z1 := complex.abs Z1

theorem modulus_Z1_correct : modulus_Z1 = real.sqrt 5 :=
by sorry

end modulus_Z1_correct_l245_245696


namespace max_n_for_sum_lt_zero_l245_245439

-- Define the arithmetic sequence and associated properties
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ k in finset.range n, a k

-- Problem statement
theorem max_n_for_sum_lt_zero
  (a : ℕ → ℚ)
  (h_arith : is_arithmetic_sequence a)
  (h_a1 : a 1 < 0)
  (h_a2015_a2016 : a 2015 + a 2016 > 0)
  (h_a2015_a2016_neg_product : a 2015 * a 2016 < 0) :
  ∃ n : ℕ, n = 4029 ∧ sum_of_first_n_terms a n < 0 ∧ ∀ m : ℕ, m > n → sum_of_first_n_terms a m ≥ 0 :=
sorry

end max_n_for_sum_lt_zero_l245_245439


namespace josh_paths_l245_245358

theorem josh_paths (n : ℕ) (h : n > 0) : 
  let start := (0, 0)
  let end := (n - 1, 1)
  -- the number of distinct paths from start to end is 2^(n-1)
  (if n = 1 then 1 else 2^(n-1)) = 2^(n-1) :=
by
  sorry

end josh_paths_l245_245358


namespace find_phi_and_omega_l245_245297

def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem find_phi_and_omega :
  ∃ φ ω, (0 ≤ φ ∧ φ ≤ π) ∧ 
         (ω > 0) ∧
         (∀ x, f ω φ x = f ω φ (-x)) ∧
         (∀ x, f ω φ (x + 3*π/4) = -f ω φ (x - 3*π/4)) ∧
         (∀ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ π/2) ∧ (0 ≤ x2 ∧ x2 ≤ π/2) ∧ x1 < x2 → f ω φ x1 < f ω φ x2)
  → φ = π / 2 ∧ (ω = 2 / 3 ∨ ω = 2) := 
sorry

end find_phi_and_omega_l245_245297


namespace greatestIntegerSum_modulo_l245_245393

noncomputable def greatestInteger (x : ℝ) : ℤ := Int.floor x

theorem greatestIntegerSum_modulo (
  A : ℝ := ∑ n in Finset.range (2021), greatestInteger (7^(n+1) / 8)
) : (A % 50) = 40 := 
by sorry

end greatestIntegerSum_modulo_l245_245393


namespace total_distance_biked_l245_245383

-- Definitions of the given conditions
def biking_time_to_park : ℕ := 15
def biking_time_return : ℕ := 25
def average_speed : ℚ := 6 -- miles per hour

-- Total biking time in minutes, then converted to hours
def total_biking_time_minutes : ℕ := biking_time_to_park + biking_time_return
def total_biking_time_hours : ℚ := total_biking_time_minutes / 60

-- Prove that the total distance biked is 4 miles
theorem total_distance_biked : total_biking_time_hours * average_speed = 4 := 
by
  -- proof will be here
  sorry

end total_distance_biked_l245_245383


namespace jenny_hours_left_l245_245352

theorem jenny_hours_left : 
  ∀ (total_hours research_hours proposal_hours : ℕ), 
  total_hours = 20 → 
  research_hours = 10 → 
  proposal_hours = 2 → 
  total_hours - (research_hours + proposal_hours) = 8 :=
by
  intros total_hours research_hours proposal_hours h_total h_research h_proposal
  rw [h_total, h_research, h_proposal]
  norm_num

end jenny_hours_left_l245_245352


namespace general_formulas_sum_b_n_l245_245281

variable {a : ℕ → ℤ} (b : ℕ → ℤ)

-- Conditions
def a_seq_condition : Prop :=
  a 1 = 3 ∧ a 4 = 12 ∧ ∀ n, a n = 3 * n

def b_seq_condition : Prop :=
  b 1 = 4 ∧ b 4 = 20 ∧ (∃ q, q = 2 ∧ ∀ n, b n - a n = q^(n-1))

-- General formulas for a_n and b_n
theorem general_formulas 
  (ha : a_seq_condition a)
  (hb : b_seq_condition b) :
  (∀ n, a n = 3 * n) ∧ (∀ n, b n = 3 * n + 2^(n-1)) := sorry

-- Sum of the first n terms of b_n
theorem sum_b_n 
  {n : ℕ}
  (ha : a_seq_condition a)
  (hb : b_seq_condition b) :
  (finset.range n).sum b = (3 * n * (n + 1)) / 2 + 2^n - 1 := sorry

end general_formulas_sum_b_n_l245_245281


namespace sequence_term_is_log2_3_l245_245300

theorem sequence_term_is_log2_3 :
  ∃ (n : ℕ), (0 < n) ∧ (∃ m : ℕ, m = 3 ∧ (log2 3 = log2 ((3 + m^2) / 4)) = n) :=
by
  sorry

end sequence_term_is_log2_3_l245_245300


namespace round_trip_time_l245_245411

variable (dist : ℝ)
variable (speed_to_work : ℝ)
variable (speed_to_home : ℝ)

theorem round_trip_time (h_dist : dist = 24) (h_speed_to_work : speed_to_work = 60) (h_speed_to_home : speed_to_home = 40) :
    (dist / speed_to_work + dist / speed_to_home) = 1 := 
by 
  sorry

end round_trip_time_l245_245411


namespace unique_sequence_to_q_l245_245511

def mediant (a b c d : ℕ) : ℕ × ℕ := (a + c, b + d)

def invariant (a b c d e f : ℕ) : Prop :=
  b * e - a * f = 1 ∧ d * e - c * f = 1 ∧ a * d - b * c = 1

theorem unique_sequence_to_q :
  ∀ (q : ℚ), 0 < q ∧ q < 1 →
  ∃! (seq : List (ℕ × ℕ)), 
    (seq.head = (0, 1) ∧ seq.nth 1 = (1, 2) ∧ seq.nth 2 = (1, 1)) ∧
    ∀ (n : ℕ), 
      match seq.nth (n + 2) with
      | some (a, b) :=
        invariant ((seq.nth n).getOrElse (0,1)).1 ((seq.nth n).getOrElse (0,1)).2
                  ((seq.nth (n + 1)).getOrElse (1,2)).1 ((seq.nth (n + 1)).getOrElse (1,2)).2
                  a b
      | none := True
    end ∧
    seq.nth (seq.length - 2) = some q :=
begin
  sorry
end

end unique_sequence_to_q_l245_245511


namespace div_add_example_l245_245104

theorem div_add_example : 150 / (10 / 2) + 5 = 35 := by
  sorry

end div_add_example_l245_245104


namespace num_students_in_class_l245_245773

-- Define the conditions
variables (S : ℕ) (num_boys : ℕ) (num_boys_under_6ft : ℕ)

-- Assume the conditions given in the problem
axiom two_thirds_boys : num_boys = (2 * S) / 3
axiom three_fourths_under_6ft : num_boys_under_6ft = (3 * num_boys) / 4
axiom nineteen_boys_under_6ft : num_boys_under_6ft = 19

-- The statement we want to prove
theorem num_students_in_class : S = 38 :=
by
  -- Proof omitted (insert proof here)
  sorry

end num_students_in_class_l245_245773


namespace unusual_numbers_exist_l245_245565

noncomputable def n1 : ℕ := 10 ^ 100 - 1
noncomputable def n2 : ℕ := 10 ^ 100 / 2 - 1

theorem unusual_numbers_exist : 
  (n1 ^ 3 % 10 ^ 100 = n1 ∧ n1 ^ 2 % 10 ^ 100 ≠ n1) ∧ 
  (n2 ^ 3 % 10 ^ 100 = n2 ∧ n2 ^ 2 % 10 ^ 100 ≠ n2) :=
by
  sorry

end unusual_numbers_exist_l245_245565


namespace inequality_proof_l245_245980

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end inequality_proof_l245_245980


namespace compute_100p_plus_q_l245_245019

-- Given constants p, q under the provided conditions,
-- prove the result: 100p + q = 430 / 3.
theorem compute_100p_plus_q (p q : ℚ) 
  (h1 : ∀ x : ℚ, (x + p) * (x + q) * (x + 20) = 0 → x ≠ -4)
  (h2 : ∀ x : ℚ, (x + 3 * p) * (x + 4) * (x + 10) = 0 → (x = -4 ∨ x ≠ -4)) :
  100 * p + q = 430 / 3 := 
by 
  sorry

end compute_100p_plus_q_l245_245019


namespace find_n_in_interval_l245_245661

theorem find_n_in_interval :
  ∃ n : ℕ, 1 ≤ n ∧ n < 1500 ∧
  (dec_eq_period_of_1_over_n (1 / n) 5) ∧
  (dec_eq_period_of_1_over_n (1 / (n + 7)) 4) ∧
  (301 ≤ n ∧ n ≤ 600) :=
by
  sorry

-- Definitions for the periodicity conditions of the decimal expansion
def dec_eq_period_of_1_over_n (x : ℚ) (p : ℕ) : Prop :=
  ∃ a b : ℚ, x = a + b * 10 ^ (-p)

end find_n_in_interval_l245_245661


namespace wendy_candy_in_each_box_l245_245103

variable (x : ℕ)

def brother_candy : ℕ := 6
def total_candy : ℕ := 12
def wendy_boxes : ℕ := 2 * x

theorem wendy_candy_in_each_box :
  2 * x + brother_candy = total_candy → x = 3 :=
by
  intro h
  sorry

end wendy_candy_in_each_box_l245_245103


namespace volume_of_region_within_larger_sphere_not_within_smaller_l245_245925

theorem volume_of_region_within_larger_sphere_not_within_smaller 
  (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 7) : 
  let V_small := (4 / 3) * π * r_small^3,
      V_large := (4 / 3) * π * r_large^3 in
  V_large - V_small = 372 * π :=
by
  sorry

end volume_of_region_within_larger_sphere_not_within_smaller_l245_245925


namespace min_dividend_l245_245944

variable (q: ℕ) (r: ℕ)

theorem min_dividend (hq : q = 6) (hr : r = 6) : ∃ d: ℕ, d = r + 1 ∧ q * d + r = 48 :=
by
  use r + 1
  split
  { rw hr },
  sorry

end min_dividend_l245_245944


namespace bryan_bookshelves_l245_245615

-- Definitions: 
def total_books : ℕ := 621
def books_per_shelf : ℕ := 27

-- Theorem statement
theorem bryan_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) (total_books = 621) (books_per_shelf = 27) : total_books / books_per_shelf = 23 :=
by 
  sorry

end bryan_bookshelves_l245_245615


namespace pump_without_leak_time_l245_245579

theorem pump_without_leak_time :
  ∃ T : ℝ, (1/T - 1/5.999999999999999 = 1/3) ∧ T = 2 :=
by 
  sorry

end pump_without_leak_time_l245_245579


namespace solve_for_a_l245_245700

noncomputable def i : ℂ := complex.I

theorem solve_for_a (a : ℝ) (h : ((a + i) ^ 2 * i).im = 0 ∧ ((a + i) ^ 2 * i).re > 0) : a = -1 :=
by {
  -- Proof here
  sorry
}

end solve_for_a_l245_245700


namespace radius_of_inscribed_circle_of_DEF_l245_245212

namespace Geometry

def radius_inscribed_circle (DE DF EF : ℝ) (h1 : DE = 30) (h2 : DF = 26) (h3 : EF = 28) : ℝ :=
  let s := (DE + DF + EF) / 2 in
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  K / s

theorem radius_of_inscribed_circle_of_DEF (r : ℝ) (h1 : DE = 30) (h2 : DF = 26) (h3 : EF = 28) :
  radius_inscribed_circle DE DF EF h1 h2 h3 = 8 * Real.sqrt 6 :=
by
  sorry

end Geometry

end radius_of_inscribed_circle_of_DEF_l245_245212


namespace number_of_intersections_l245_245209

-- Definitions of line and circle equations
def line (x y : ℝ) := 3 * x + 4 * y = 12
def circle (x y : ℝ) := x^2 + y^2 = 25

-- Main theorem
theorem number_of_intersections : 
  (∃ x y, line x y ∧ circle x y) ∧ (∃ a b, line a b ∧ circle a b ∧ (a ≠ x ∨ b ≠ y)) :=
sorry

end number_of_intersections_l245_245209


namespace complement_union_A_B_l245_245024

def U := {x : ℕ | x > 0 ∧ x < 8}
def A := {1, 3, 5, 7}
def B := {2, 4, 5}

theorem complement_union_A_B : {x ∈ U | x ∉ A ∪ B} = {6} :=
by
  sorry

end complement_union_A_B_l245_245024


namespace smallest_positive_difference_exists_l245_245508

-- Definition of how Vovochka sums two three-digit numbers
def vovochkaSum (x y : ℕ) : ℕ :=
  let h1 := (x / 100) + (y / 100)
  let t1 := ((x / 10) % 10) + ((y / 10) % 10)
  let u1 := (x % 10) + (y % 10)
  h1 * 1000 + t1 * 100 + u1

-- Definition for correct sum of two numbers
def correctSum (x y : ℕ) : ℕ := x + y

-- Function to find difference
def difference (x y : ℕ) : ℕ :=
  abs (vovochkaSum x y - correctSum x y)

-- Proof the smallest positive difference between Vovochka's sum and the correct sum
theorem smallest_positive_difference_exists :
  ∃ x y : ℕ, (x < 1000) ∧ (y < 1000) ∧ difference x y > 0 ∧ 
  (∀ a b : ℕ, (a < 1000) ∧ (b < 1000) ∧ difference a b > 0 → difference x y ≤ difference a b) :=
sorry

end smallest_positive_difference_exists_l245_245508


namespace parabola_expression_l245_245322

theorem parabola_expression 
  (a b : ℝ) 
  (h : 9 = a * (-2)^2 + b * (-2) + 5) : 
  2 * a - b + 6 = 8 :=
by
  sorry

end parabola_expression_l245_245322


namespace groupC_is_all_polyhedra_l245_245177

inductive GeometricBody
| TriangularPrism : GeometricBody
| QuadrangularPyramid : GeometricBody
| Sphere : GeometricBody
| Cone : GeometricBody
| Cube : GeometricBody
| TruncatedCone : GeometricBody
| HexagonalPyramid : GeometricBody
| Hemisphere : GeometricBody

def isPolyhedron : GeometricBody → Prop
| GeometricBody.TriangularPrism => true
| GeometricBody.QuadrangularPyramid => true
| GeometricBody.Sphere => false
| GeometricBody.Cone => false
| GeometricBody.Cube => true
| GeometricBody.TruncatedCone => false
| GeometricBody.HexagonalPyramid => true
| GeometricBody.Hemisphere => false

def groupA := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Sphere, GeometricBody.Cone]
def groupB := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Cube, GeometricBody.TruncatedCone]
def groupC := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Cube, GeometricBody.HexagonalPyramid]
def groupD := [GeometricBody.Cone, GeometricBody.TruncatedCone, GeometricBody.Sphere, GeometricBody.Hemisphere]

def allPolyhedra (group : List GeometricBody) : Prop :=
  ∀ b, b ∈ group → isPolyhedron b

theorem groupC_is_all_polyhedra : 
  allPolyhedra groupC ∧
  ¬ allPolyhedra groupA ∧
  ¬ allPolyhedra groupB ∧
  ¬ allPolyhedra groupD :=
by
  sorry

end groupC_is_all_polyhedra_l245_245177


namespace _l245_245533

noncomputable theory

variables {ι : Type*} [encodable ι] {ξ : ι → ℝ}
          (Sn : ℕ → ℝ) (S : ℝ)

-- Definitions
def orthogonal (ξ : ι → ℝ) := ∀ i j, i ≠ j → ∫ ξ i * ξ j = 0

def quadratic_mean (ξ : ι → ℝ) := ∑' n, ∫ (ξ n) ^ 2

-- Problem Statement

lemma part_a 
  (h_orth : orthogonal ξ)
  (h_sum : quadratic_mean ξ < ∞) :
  ∃ S, ∫ (S)^2 < ∞ ∧ tendsto (λ n, ∫ (Sn n - S)^2) at_top (𝓝 0) := 
sorry

lemma rademacher_menchoff_theorem 
  (h_orth : orthogonal ξ)
  (h_radc : ∀ (m n : ℕ), ∫ (max (finite (λ k, m < k ∧ k ≤ n + m)) (λ k, Sn k - Sn m)^2) ≤ (log n)^2 * ∑ i in finset.range n, ∫ (ξ (m + i + 1))^2)
  (h_rm_sum : ∑' n, quadratic_mean ξ * (log n)^2 < ∞) :
  ∀ᶠ n in at_top, Sn n = S := 
sorry

end _l245_245533


namespace find_q_l245_245261

variable (d : ℝ) (q : ℝ)
variable (h_d_nonzero : d ≠ 0)
variable (h_q_rational_positive : ∃ (r : ℚ), q = r ∧ 0 < q ∧ q < 1)

-- Given arithmetic sequence an and geometric sequence bn
def a_n (n : ℕ) : ℝ := d * n
def b_n (n : ℕ) : ℝ := d^2 * q^(n - 1)

-- Specified initial conditions
def a1 := d
def b1 := d^2

-- Sum of the squares of the first three terms of the arithmetic sequence
def sum_of_squares_an := a_n d 1 ^ 2 + a_n d 2 ^ 2 + a_n d 3 ^ 2

-- Sum of the first three terms of the geometric sequence
def sum_bn := b_n d q 1 + b_n d q 2 + b_n d q 3

-- The ratio of the sums
def ratio := sum_of_squares_an / sum_bn

-- The theorem we want to prove
theorem find_q (h_rat_eq : ratio = 14 / (1 + q + q^2)) : q = (Real.sqrt 5 - 1) / 2 :=
sorry

end find_q_l245_245261


namespace problem_first_three_digits_l245_245934

noncomputable def first_three_decimal_digits (x : ℝ) : ℕ :=
  let y := x - x.floor
  (((y * 1000).floor : ℕ) % 1000)

theorem problem_first_three_digits (n k : ℝ) (x : ℝ)
  (h1 : x = (n^100 + 1)^(5/3))
  (h2 : n = 10) :
  first_three_decimal_digits x = 666 := 
  sorry

end problem_first_three_digits_l245_245934


namespace exp_ln_of_five_l245_245517

theorem exp_ln_of_five : exp (real.log 5) = 5 :=
by
  sorry

end exp_ln_of_five_l245_245517


namespace lisa_total_spoons_l245_245848

def number_of_baby_spoons (num_children num_spoons_per_child : Nat) : Nat :=
  num_children * num_spoons_per_child

def number_of_decorative_spoons : Nat := 2

def number_of_old_spoons (baby_spoons decorative_spoons : Nat) : Nat :=
  baby_spoons + decorative_spoons
  
def number_of_new_spoons (large_spoons teaspoons : Nat) : Nat :=
  large_spoons + teaspoons

def total_number_of_spoons (old_spoons new_spoons : Nat) : Nat :=
  old_spoons + new_spoons

theorem lisa_total_spoons
  (children : Nat)
  (spoons_per_child : Nat)
  (large_spoons : Nat)
  (teaspoons : Nat)
  (children_eq : children = 4)
  (spoons_per_child_eq : spoons_per_child = 3)
  (large_spoons_eq : large_spoons = 10)
  (teaspoons_eq : teaspoons = 15)
  : total_number_of_spoons (number_of_old_spoons (number_of_baby_spoons children spoons_per_child) number_of_decorative_spoons) (number_of_new_spoons large_spoons teaspoons) = 39 :=
by
  sorry

end lisa_total_spoons_l245_245848


namespace comparison_of_abc_l245_245827

noncomputable def a : ℝ := Real.log 8 / Real.log 3  -- equivalent to log base 3 of 8
noncomputable def b : ℝ := 2 ^ 1.2
noncomputable def c : ℝ := 0.3 ^ 3.1

theorem comparison_of_abc : c < a ∧ a < b := by
  sorry

end comparison_of_abc_l245_245827


namespace mul_102_102_l245_245642

theorem mul_102_102 : 102 * 102 = 10404 := by
  sorry

end mul_102_102_l245_245642


namespace solve_for_x_l245_245881

noncomputable section

-- Define the determinant equation condition
def determinant_condition (x : ℝ) : Prop :=
  (∥Real.sqrt 3 * Real.cos x, Real.sin x∥.det.toReal = Real.cos x, Real.cos x∥.det.toReal = (Real.sqrt 3 / 2))

-- Define the interval condition
def interval_condition (x : ℝ) : Prop :=
  3 < x ∧ x < 4

-- Main theorem statement
theorem solve_for_x (x : ℝ) (hx₁ : interval_condition x) (hx₂ : determinant_condition x) : 
  x = 7 * Real.pi / 6 := 
sorry

end solve_for_x_l245_245881


namespace josh_walk_ways_l245_245364

theorem josh_walk_ways (n : ℕ) :
  let grid_rows := n
      grid_columns := 3
      start_position := (0, 0)  -- (row, column) starting from bottom left
  in grid_rows > 0 →
      let center_square (k : ℕ) := (k, 1) -- center square of k-th row
  in ∃ ways : ℕ, ways = 2^(n-1) ∧
                ways = count_paths_to_center_topmost n  -- custom function representation
sorry

end josh_walk_ways_l245_245364


namespace complementary_angle_percentage_decrease_l245_245472

theorem complementary_angle_percentage_decrease :
  ∀ (α β : ℝ), α + β = 90 → 6 * α = 3 * β → 
  let α' := 1.2 * α in
  let β' := 90 - α' in
  (100 * (β' / β)) = 90 :=
by
  intros α β h_sum h_ratio α' β'
  have α' := 1.2 * α
  have β' := 90 - α'
  sorry

end complementary_angle_percentage_decrease_l245_245472


namespace cost_of_one_lesson_l245_245860

-- Define the conditions
def total_cost_for_lessons : ℝ := 360
def total_hours_of_lessons : ℝ := 18
def duration_of_one_lesson : ℝ := 1.5

-- Define the theorem statement
theorem cost_of_one_lesson :
  (total_cost_for_lessons / total_hours_of_lessons) * duration_of_one_lesson = 30 := by
  -- Proof goes here
  sorry

end cost_of_one_lesson_l245_245860


namespace ratio_eval_l245_245644

universe u

def a : ℕ := 121
def b : ℕ := 123
def c : ℕ := 122

theorem ratio_eval : (2 ^ a * 3 ^ b) / (6 ^ c) = (3 / 2) := by
  sorry

end ratio_eval_l245_245644


namespace hcf_of_two_numbers_l245_245904

theorem hcf_of_two_numbers (A B : ℕ) (h1 : A * B = 4107) (h2 : A = 111) : (Nat.gcd A B) = 37 :=
by
  -- Given conditions
  have h3 : B = 37 := by
    -- Deduce B from given conditions
    sorry
  -- Prove hcf (gcd) is 37
  sorry

end hcf_of_two_numbers_l245_245904


namespace part1_part2_l245_245715

open Classical

noncomputable theory

variables {x y m k b : ℝ}
variables (A B C D O : ℝ × ℝ)
variables hCurve : ∀ x y, x^2 - y^2 = m
variables hXpos : ∀ x, x > 0
variables hConst : m ≠ 0
variables hLine : ∀ x y, y = k * x + b
variables hPoints : A = (0, 0) ∧ B = (x, y) ∧ C = (x, y) ∧ D = (0, 0)

theorem part1 (hEqual : dist A B = dist B C ∧ dist B C = dist C D) :
  area_triangle A O D = 9 * m / 8 := sorry

theorem part2 (hArea : area_triangle B O C = area_triangle A O D / 3) :
  dist A B = dist B C ∧ dist B C = dist C D := sorry

end part1_part2_l245_245715


namespace division_criterion_based_on_stroke_l245_245608

-- Definition of a drawable figure with a single stroke
def drawable_in_one_stroke (figure : Type) : Prop := sorry -- exact conditions can be detailed with figure representation

-- Example figures for the groups (types can be extended based on actual representation)
def Group1 := {fig1 : Type // drawable_in_one_stroke fig1}
def Group2 := {fig2 : Type // ¬drawable_in_one_stroke fig2}

-- Problem Statement:
theorem division_criterion_based_on_stroke (fig : Type) :
  (drawable_in_one_stroke fig ∨ ¬drawable_in_one_stroke fig) := by
  -- We state that every figure belongs to either Group1 or Group2
  sorry

end division_criterion_based_on_stroke_l245_245608


namespace car_daily_rental_cost_l245_245545

theorem car_daily_rental_cost 
  (x : ℝ)
  (cost_per_mile : ℝ)
  (budget : ℝ)
  (miles : ℕ)
  (h1 : cost_per_mile = 0.18)
  (h2 : budget = 75)
  (h3 : miles = 250)
  (h4 : x + (miles * cost_per_mile) = budget) : 
  x = 30 := 
sorry

end car_daily_rental_cost_l245_245545


namespace ratio_of_part_to_whole_l245_245868

def N : ℝ := 750

def part_of_one_third_of_two_fifth (N : ℝ) : ℝ :=
  (1/3) * (2/5) * N

theorem ratio_of_part_to_whole 
  (h1 : part_of_one_third_of_two_fifth N = 25)
  (h2 : 0.4 * N = 300) :
  (25 / part_of_one_third_of_two_fifth N) = 1 / 4 :=
by
  sorry

end ratio_of_part_to_whole_l245_245868


namespace island_knights_liars_two_people_l245_245414

def islanders_knights_and_liars (n : ℕ) : Prop :=
  ∃ (knight liar : ℕ),
    knight + liar = n ∧
    (∀ i : ℕ, 1 ≤ i → i ≤ n → 
      ((i % i = 0 → liar > 0 ∧ knight > 0) ∧ (i % i ≠ 0 → liar > 0)))

theorem island_knights_liars_two_people :
  islanders_knights_and_liars 2 :=
sorry

end island_knights_liars_two_people_l245_245414


namespace sum_of_two_digit_numbers_with_product_1540_l245_245454

theorem sum_of_two_digit_numbers_with_product_1540 : 
  ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ a * b = 1540 ∧ a + b = 97 :=
begin
  -- Proof goes here
  sorry
end

end sum_of_two_digit_numbers_with_product_1540_l245_245454


namespace find_a2018_l245_245684

-- Given Conditions
def initial_condition (a : ℕ → ℤ) : Prop :=
  a 1 = -1

def absolute_difference (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → abs (a n - a (n-1)) = 2^(n-1)

def subseq_decreasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2*n-1) > a (2*(n+1)-1)

def subseq_increasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2*n) < a (2*(n+1))

-- Theorem to Prove
theorem find_a2018 (a : ℕ → ℤ)
  (h1 : initial_condition a)
  (h2 : absolute_difference a)
  (h3 : subseq_decreasing a)
  (h4 : subseq_increasing a) :
  a 2018 = (2^2018 - 1) / 3 :=
sorry

end find_a2018_l245_245684


namespace find_k_l245_245736

open Real

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

theorem find_k (k : ℝ) :
  let a : vector := (2, 3)
  let b : vector := (1, 4)
  let c : vector := (k, 3)
  orthogonal (a.1 + b.1, a.2 + b.2) c → k = -7 :=
by
  intros
  sorry

end find_k_l245_245736


namespace chord_length_of_intersection_l245_245451

def P_0 : ℝ × ℝ := (-4, 0)
def line_param (t : ℝ) : ℝ × ℝ := (-4 + (sqrt 3) / 2 * t, 1 / 2 * t)
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 7

theorem chord_length_of_intersection : 
  (∃ t1 t2 : ℝ, (line_param t1).fst ^ 2 + (line_param t1).snd ^ 2 = 7 ∧ 
                 (line_param t2).fst ^ 2 + (line_param t2).snd ^ 2 = 7 ∧
                 t1 ≠ t2) →
  ∃ t1 t2 : ℝ, |(t2 - t1)| = 2 * sqrt 3 :=
sorry

end chord_length_of_intersection_l245_245451


namespace single_elimination_tournament_games_l245_245031

theorem single_elimination_tournament_games (n : ℕ) (h : n = 32) : (n - 1) = 31 :=
by
  rw [h]
  norm_num

end single_elimination_tournament_games_l245_245031


namespace points_in_rect_close_l245_245419

open Set

/-- Define the rectangle of dimensions 4 by 3 and the 6 points inside it --/
structure Rectangle := (width : ℝ) (height : ℝ)

noncomputable def rect := Rectangle.mk 4 3

/-- Define what it means for a set of points to be inside a rectangle --/
def inside_rect (rect : Rectangle) (points : Fin 6 → ℝ × ℝ) : Prop := 
  ∀ i, (0 ≤ (points i).fst ∧ (points i).fst ≤ rect.width) ∧ 
       (0 ≤ (points i).snd ∧ (points i).snd ≤ rect.height)

/-- Define the distance function between two points in a plane --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.fst - p2.fst) ^ 2 + (p1.snd - p2.snd) ^ 2)

theorem points_in_rect_close (points : Fin 6 → ℝ × ℝ) 
  (h : inside_rect rect points) : 
  ∃ (i j : Fin 6), i ≠ j ∧ distance (points i) (points j) ≤ real.sqrt 5 := 
sorry

end points_in_rect_close_l245_245419


namespace extreme_value_proof_l245_245317

noncomputable def extreme_value (x y : ℝ) := 4 * x + 3 * y 

theorem extreme_value_proof 
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (h : x + y = 5 * x * y) : 
  extreme_value x y = 3 :=
sorry

end extreme_value_proof_l245_245317


namespace exists_uncolored_subgrid_l245_245241

def grid_8x8 := Fin 8 × Fin 8

def two_by_two_colored (s : set (Fin 8 × Fin 8)) : Prop :=
  ∃ (x y : Fin 4),  ∀ (i j : Fin 2), (2 * x + i, 2 * y + j) ∈ s

theorem exists_uncolored_subgrid:
  ∀ (s : set (Fin 8 × Fin 8)),
  (∀ (x y : Fin 4), ∃ (i j : Fin 2), ((2 * x + i, 2 * y + j) ∈ s)) →
  ∃ (x y : Fin 4), ¬ two_by_two_colored (s \ { p : grid_8x8 | two_by_two_colored {p}}) :=
by
  sorry

end exists_uncolored_subgrid_l245_245241


namespace year_of_max_increase_l245_245448

def revenue : ℕ → ℝ
| 2010 := 100
| 2011 := 120
| 2012 := 115
| 2013 := 130
| 2014 := 140
| 2015 := 150
| 2016 := 145
| 2017 := 180
| 2018 := 175
| 2019 := 200
| _ := 0

theorem year_of_max_increase : ∃ y, y > 2010 ∧ (∀ x, x > 2010 → x ≠ y → (revenue x - revenue (x - 1)) ≤ (revenue y - revenue (y - 1))) ∧ y = 2017 :=
by
  sorry

end year_of_max_increase_l245_245448


namespace largest_possible_y_l245_245385

theorem largest_possible_y (x : ℝ) (hx : x > 10^(1.5)) :
  ∃ y : ℕ, ∀ y', (\log 10 x)^(y' - 1) = y' → (\log 10 x) > 1.5 → y' ≤ 4 := 
sorry

end largest_possible_y_l245_245385


namespace greatest_prime_factor_391_l245_245111

theorem greatest_prime_factor_391 : ∃ p, Prime p ∧ p ∣ 391 ∧ ∀ q, Prime q ∧ q ∣ 391 → q ≤ p :=
by
  sorry

end greatest_prime_factor_391_l245_245111


namespace find_other_number_l245_245081

-- Defining the two numbers and their properties
def sum_is_84 (a b : ℕ) : Prop := a + b = 84
def one_is_36 (a b : ℕ) : Prop := a = 36 ∨ b = 36
def other_is_48 (a b : ℕ) : Prop := a = 48 ∨ b = 48

-- The theorem statement
theorem find_other_number (a b : ℕ) (h1 : sum_is_84 a b) (h2 : one_is_36 a b) : other_is_48 a b :=
by {
  sorry
}

end find_other_number_l245_245081


namespace find_percentage_decrease_l245_245464

-- Define the measures of two complementary angles
def angles_complementary (a b : ℝ) : Prop := a + b = 90

-- Given variables
variable (small_angle large_angle : ℝ)

-- Given conditions
def ratio_of_angles (small_angle large_angle : ℝ) : Prop := small_angle / large_angle = 1 / 2

def increased_small_angle (small_angle : ℝ) : ℝ := small_angle * 1.2

noncomputable def new_large_angle (small_angle large_angle : ℝ) : ℝ :=
  90 - increased_small_angle small_angle

def percentage_decrease (original new : ℝ) : ℝ :=
  ((original - new) / original) * 100

-- The theorem we need to prove
theorem find_percentage_decrease
  (h1 : ratio_of_angles small_angle large_angle)
  (h2 : angles_complementary small_angle large_angle) :
  percentage_decrease large_angle (new_large_angle small_angle large_angle) = 10 :=
sorry

end find_percentage_decrease_l245_245464


namespace combined_sum_exterior_angles_pentagon_hexagon_l245_245556

theorem combined_sum_exterior_angles_pentagon_hexagon : 
  let pentagon_exterior_sum := 360
  let hexagon_exterior_sum := 360
  pentagon_exterior_sum + hexagon_exterior_sum = 720 := 
by {
  -- Definitions
  let pentagon_exterior_sum := 360
  let hexagon_exterior_sum := 360

  -- Proof
  have h_sum := pentagon_exterior_sum + hexagon_exterior_sum
  show h_sum = 720, from rfl
}

end combined_sum_exterior_angles_pentagon_hexagon_l245_245556


namespace total_savings_correct_l245_245083

def total_cost_without_discounts := 30600
def boxes_purchased := 36
def discount_rate := if boxes_purchased >= 30 then 0.15 else if boxes_purchased >= 20 then 0.10 else if boxes_purchased >= 10 then 0.05 else 0.0
def total_cost_with_discounts := total_cost_without_discounts * (1 - discount_rate)
def amount_saved := total_cost_without_discounts - total_cost_with_discounts

theorem total_savings_correct : 
  amount_saved = 5090 := 
by 
  sorry

end total_savings_correct_l245_245083


namespace A_receives_correct_amount_l245_245957

section

variables (rs : Type*) [field rs]

def total_capital (A_cap B_cap : rs) : rs :=
  A_cap + B_cap

def managing_profit (total_profit : rs) : rs :=
  0.10 * total_profit

def remaining_profit (total_profit manage_profit : rs) : rs :=
  total_profit - manage_profit

def ratio_A_B (A_cap B_cap : rs) : rs × rs :=
  (A_cap / 500, B_cap / 500)

def total_parts (ratios : rs × rs) : rs :=
  ratios.1 + ratios.2

def one_part_value (rem_pro total_parts : rs) : rs :=
  rem_pro / total_parts

def A_share (one_part : rs) (A_ratio : rs) : rs :=
  A_ratio * one_part

def total_money_received_by_A (managing_profit A_share : rs) : rs :=
  managing_profit + A_share

variables (A_cap B_cap total_profit : rs)
variables (A_cap_val : A_cap = 3500) (B_cap_val : B_cap = 1500) (total_profit_val : total_profit = 9600)

theorem A_receives_correct_amount :
  total_money_received_by_A
    (managing_profit total_profit)
    (A_share 
      (one_part_value 
        (remaining_profit total_profit (managing_profit total_profit)) 
        (total_parts (ratio_A_B A_cap B_cap)))
      (prod.fst (ratio_A_B A_cap B_cap)))
  = 7008 :=
sorry

end

end A_receives_correct_amount_l245_245957


namespace josh_paths_l245_245360

theorem josh_paths (n : ℕ) (h : n > 0) : 
  let start := (0, 0)
  let end := (n - 1, 1)
  -- the number of distinct paths from start to end is 2^(n-1)
  (if n = 1 then 1 else 2^(n-1)) = 2^(n-1) :=
by
  sorry

end josh_paths_l245_245360


namespace Lisa_total_spoons_l245_245849

def total_spoons (children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) (large_spoons : ℕ) (teaspoons : ℕ) : ℕ := 
  (children * spoons_per_child) + decorative_spoons + (large_spoons + teaspoons)

theorem Lisa_total_spoons :
  (total_spoons 4 3 2 10 15) = 39 :=
by
  sorry

end Lisa_total_spoons_l245_245849


namespace complementary_angle_percentage_decrease_l245_245471

theorem complementary_angle_percentage_decrease :
  ∀ (α β : ℝ), α + β = 90 → 6 * α = 3 * β → 
  let α' := 1.2 * α in
  let β' := 90 - α' in
  (100 * (β' / β)) = 90 :=
by
  intros α β h_sum h_ratio α' β'
  have α' := 1.2 * α
  have β' := 90 - α'
  sorry

end complementary_angle_percentage_decrease_l245_245471


namespace inequality_proof_l245_245991

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l245_245991


namespace concur_or_parallel_l245_245349

variables (A B C A₁ B₁ A₂ B₂ A₃ B₃ : Type)

open_locale classical

noncomputable def geometric_configuration :=
  ∃ (A B C : Type),
  ∃ (altitude_AA₁ : Type) (altitude_BB₁ : Type),
  ∃ (angle_bisector_AA₂ : Type) (angle_bisector_BB₂ : Type),
  ∃ (incenter : Type),
  ∃ (incircle_tangents : Type),
  ∃ (A₃ : Type) (B₃ : Type),
  altitude_AA₁ ∧ altitude_BB₁ ∧ angle_bisector_AA₂ ∧ angle_bisector_BB₂ ∧
  incircle_tangents ∧ A₃ ∧ B₃ ∧
  (line A₁ B₁ = line A₂ B₂ ∨ parallel (line A₁ B₁) (line A₂ B₂))

theorem concur_or_parallel (A B C A₁ B₁ A₂ B₂ A₃ B₃ : Type)
  (H : geometric_configuration A B C A₁ B₁ A₂ B₂ A₃ B₃) :
  (line A₁ B₁ = line A₂ B₂ ∨ parallel (line A₁ B₁) (line A₂ B₂)) ∧
  (line A₁ B₁ = line A₃ B₃ ∨ parallel (line A₁ B₁) (line A₃ B₃)) :=
sorry

end concur_or_parallel_l245_245349


namespace inequality_hold_l245_245975

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end inequality_hold_l245_245975


namespace geometric_sequence_sum_l245_245259

-- Define the geometric sequence properties
variables {a : ℕ → ℝ}
variable {r : ℝ} -- common ratio
variable {a1 : ℝ} -- first term
variable {n : ℕ} -- number of terms

-- Geometric sequence definition stating it is monotonically increasing
def is_geometric (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * r^n

-- Solve the problem
theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (n : ℕ)
  (h1 : is_geometric a r a1)
  (h2 : a 2 * a 6 = 16)
  (h3 : a 3 + a 5 = 10)
  (h_monotonic : ∀ n : ℕ, a n ≤ a (n + 1)):
  ∑ i in finset.range n, a i = 2^(n-1) - (1/2) :=
sorry

end geometric_sequence_sum_l245_245259


namespace A_scored_full_marks_l245_245786

/-- We have four students: A, B, C, D. Each made a statement about who scored full marks,
    only one of the statements is true, and only one student scored full marks. -/
constant A_full_marks : Prop
constant B_full_marks : Prop
constant C_full_marks : Prop
constant D_full_marks : Prop

/-- Only one of the students scored full marks. -/
axiom full_marks_unique : A_full_marks ∨ B_full_marks ∨ C_full_marks ∨ D_full_marks
  ∧ ¬ (A_full_marks ∧ B_full_marks ∨ A_full_marks ∧ C_full_marks ∨ A_full_marks ∧ D_full_marks
       ∨ B_full_marks ∧ C_full_marks ∨ B_full_marks ∧ D_full_marks ∨ C_full_marks ∧ D_full_marks)

/-- Students' statements: -/
constant A_statement : Prop := ¬ A_full_marks
constant B_statement : Prop := C_full_marks
constant C_statement : Prop := D_full_marks
constant D_statement : Prop := ¬ D_full_marks

/-- Only one of the students is telling the truth. -/
axiom statements_truth_condition : (A_statement ∧ ¬ B_statement ∧ ¬ C_statement ∧ ¬ D_statement)
  ∨ (¬ A_statement ∧ B_statement ∧ ¬ C_statement ∧ ¬ D_statement)
  ∨ (¬ A_statement ∧ ¬ B_statement ∧ C_statement ∧ ¬ D_statement)
  ∨ (¬ A_statement ∧ ¬ B_statement ∧ ¬ C_statement ∧ D_statement)

/-- Student A is the one who scored full marks. -/
theorem A_scored_full_marks : A_full_marks :=
by
  /- Proof goes here -/
  sorry

end A_scored_full_marks_l245_245786


namespace apples_eaten_l245_245183

-- Define the number of apples eaten by Anna on Tuesday
def apples_eaten_on_Tuesday : ℝ := 4

theorem apples_eaten (A : ℝ) (h1 : A = apples_eaten_on_Tuesday) 
                      (h2 : 2 * A = 2 * apples_eaten_on_Tuesday) 
                      (h3 : A / 2 = apples_eaten_on_Tuesday / 2) 
                      (h4 : A + (2 * A) + (A / 2) = 14) : 
  A = 4 :=
by {
  sorry
}

end apples_eaten_l245_245183


namespace book_chapters_pages_l245_245877

theorem book_chapters_pages :
  ∃ x : ℕ,
    (let chapter1 := 13 in
     let chapter2 := 13 + x in
     let chapter3 := 13 + 2 * x in
     let chapter4 := 13 + 3 * x in
     let chapter5 := 13 + 4 * x in
     chapter1 + chapter2 + chapter3 + chapter4 + chapter5 = 95) → 
     x = 3 :=
by
  sorry

end book_chapters_pages_l245_245877


namespace delta_slope_calculation_l245_245617

theorem delta_slope_calculation (Δx : ℝ) : 
    let x := 1
    let y := x^2 + 2
    let Δy := (1 + Δx)^2 + 2 - (1^2 + 2)
    y = 3 →
    Δx ≠ 0 →
    (Δy / Δx) = 2 + Δx :=
by 
  intros
  have h1: Δy = 2 * Δx + Δx^2 := by
    sorry
  rw h1
  field_simp
  ring
  done

end delta_slope_calculation_l245_245617


namespace cubes_sum_formula_l245_245833

theorem cubes_sum_formula (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 5) : a^3 + b^3 = 238 := 
by 
  sorry

end cubes_sum_formula_l245_245833


namespace hyperbola_eccentricity_range_l245_245308

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) : 
  ∃ e : ℝ, e = real.sqrt (1 + 1 / (a^2)) ∧ (1 < e ∧ e < real.sqrt 2) :=
by
  sorry

end hyperbola_eccentricity_range_l245_245308


namespace quadratic_intersections_l245_245829

variable {p x1 x2 : ℝ}

theorem quadratic_intersections (h_distinct: (4 * p^2 + 4 * p > 0))
  (h_distance: (abs (sqrt (4 * p^2 + 4 * p)) ≤ abs (2 * p - 3))):
  (2 * p * x1 + x2^2 + 3 * p > 0) ∧ (p ≤ 9 / 16) :=
by 
  sorry

end quadratic_intersections_l245_245829


namespace range_of_m_min_value_7a_4b_l245_245723

noncomputable def f (x m : ℝ) : ℝ := Real.sqrt (|x - 1| + |x + 1| - m)

theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f x m).dom → m ≤ 2 := sorry

theorem min_value_7a_4b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) 
  (h₃ : 2/(3*a + b) + 1/(a + 2*b) = 2) : 
  7*a + 4*b ≥ 9/2 := sorry


end range_of_m_min_value_7a_4b_l245_245723


namespace smallest_possible_positive_difference_l245_245504

def Vovochka_sum (a b : Nat) : Nat :=
  let ha := a / 100
  let ta := (a / 10) % 10
  let ua := a % 10
  let hb := b / 100
  let tb := (b / 10) % 10
  let ub := b % 10
  1000 * (ha + hb) + 100 * (ta + tb) + (ua + ub)

def correct_sum (a b : Nat) : Nat :=
  a + b

def difference (a b : Nat) : Nat :=
  abs (Vovochka_sum a b - correct_sum a b)

theorem smallest_possible_positive_difference :
  ∀ a b : Nat, (∃ a b : Nat, a > 0 ∧ b > 0 ∧ difference a b = 1800) :=
by
  sorry

end smallest_possible_positive_difference_l245_245504


namespace binomial_properties_l245_245287

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) := (2 * x - 1 / Real.sqrt x) ^ n

theorem binomial_properties (x : ℝ) (n : ℕ) (h1 : (binomial_expansion x n).terms.length = 7) :
  nat.succ n = 7 → 
  (2 ^ n = 64) ∧
  ((binomial_expansion 1 n).eval 1 = 1) ∧
  (∃ k, k = 4 ∧ binomial_coefficient (binomial_expansion x n) k = binomial_coefficient (binomial_expansion x n) (n - k)) ∧
  (∃ r_values : list ℕ, ∀ r ∈ r_values, (6 - (3 * r / 2)) ∈ ℤ ∧ r ∈ [0, 2, 4, 6] ∧ r_values.length = 4) :=
by { sorry }

end binomial_properties_l245_245287


namespace subset_of_size_3_count_l245_245305

open Finset

def odd_numbers : Finset ℕ := {1, 3, 5, 7, 9}

theorem subset_of_size_3_count : (odd_numbers.powerset.filter (λ s, s.card = 3)).card = 10 :=
by
  sorry

end subset_of_size_3_count_l245_245305


namespace total_boys_in_school_l245_245789

-- Define the total percentage of boys belonging to other communities
def percentage_other_communities := 100 - (44 + 28 + 10)

-- Total number of boys in the school, represented by a variable B
def total_boys (B : ℕ) : Prop :=
0.18 * (B : ℝ) = 117

-- The theorem states that the total number of boys B is 650
theorem total_boys_in_school : ∃ B : ℕ, total_boys B ∧ B = 650 :=
sorry

end total_boys_in_school_l245_245789


namespace painter_remaining_time_l245_245159

-- Define the initial conditions
def total_rooms : ℕ := 11
def hours_per_room : ℕ := 7
def painted_rooms : ℕ := 2

-- Define the remaining rooms to paint
def remaining_rooms : ℕ := total_rooms - painted_rooms

-- Define the proof problem: the remaining time to paint the rest of the rooms
def remaining_hours : ℕ := remaining_rooms * hours_per_room

theorem painter_remaining_time :
  remaining_hours = 63 :=
sorry

end painter_remaining_time_l245_245159


namespace find_n_l245_245100

theorem find_n (n : ℤ) (h_lcm : Nat.lcm n 16 = 48) (h_gcf : Nat.gcd n 16 = 8) : n = 24 := by
  sorry

end find_n_l245_245100


namespace right_triangle_vertex_in_subset_l245_245012

-- We start by defining the equilateral triangle ∆ABC.
structure EquilateralTriangle (A B C : Type) :=
(AB : Type)
(BC : Type)
(AC : Type)
(AB_eq : AB = BC)
(BC_eq : BC = AC)

-- Now, we define our main theorem.
theorem right_triangle_vertex_in_subset
  (A B C : Type)
  (E : EquilateralTriangle A B C)
  (S1 S2 : set A)
  (h_disjoint : S1 ∩ S2 = ∅)
  (h_union : S1 ∪ S2 = {x : A | x ∈ E.AB ∪ E.BC ∪ E.AC}) :
  ∃ (P Q R ∈ A), (P ∈ S1 ∨ P ∈ S2) ∧ (Q ∈ S1 ∨ Q ∈ S2) ∧ (R ∈ S1 ∨ R ∈ S2) ∧ is_right_triangle P Q R :=
sorry

end right_triangle_vertex_in_subset_l245_245012


namespace midpoint_distance_proof_l245_245867

variables (a b c d : ℝ) (m n : ℝ)
def initial_midpoint : Prop := (m = (a + c) / 2) ∧ (n = (b + d) / 2)

def moved_points : Prop :=
  let A' := (a + 3, b + 5)
  let B' := (c - 7, d - 4)
  ∃ m' n', m' = (A'.fst + B'.fst) / 2 ∧ n' = (A'.snd + B'.snd) / 2 ∧ (m' = m - 2) ∧ (n' = n + 1 / 2)

def distance_between_midpoints : ℝ :=
  real.sqrt ((m - 2 - m)^2 + ((n + 1 / 2) - n)^2)

theorem midpoint_distance_proof (h1 : initial_midpoint a b c d m n) (h2 : moved_points a b c d) :
  distance_between_midpoints a b c d m n = real.sqrt 17 / 2 :=
by sorry

end midpoint_distance_proof_l245_245867


namespace qualify_diameter_l245_245905

theorem qualify_diameter : ∃ (d : ℝ), 19.95 ≤ d ∧ d ≤ 20.02 ∧ d = 19.96 :=
by
  use 19.96
  split
  repeat' split
  exact le_of_lt (show 19.95 < 19.96 by norm_num)
  exact le_of_lt (show 19.96 < 20.02 by norm_num)
  rfl

end qualify_diameter_l245_245905


namespace min_people_in_TeamA_l245_245927

noncomputable def TeamProof (a b k : ℕ) : Prop :=
(b + 90 = 2 * (a - 90)) ∧ (a + k = 6 * (b - k))

theorem min_people_in_TeamA : ∃ a b k, TeamProof a b k ∧ a = 153 :=
begin
  sorry
end

end min_people_in_TeamA_l245_245927


namespace complementary_angle_decrease_l245_245461

theorem complementary_angle_decrease (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90)
  (h2 : angle1 / 3 = angle2 / 6) (h3 : ∃ x: ℝ, x = 0.2) :
  let new_angle1 := angle1 * 1.2 in 
  let new_angle2 := 90 - new_angle1 in
  (new_angle2 - angle2) / angle2 = -0.10 := sorry

end complementary_angle_decrease_l245_245461


namespace expression_equals_one_l245_245618

theorem expression_equals_one : 
  (Real.sqrt 6 / Real.sqrt 2) + abs (1 - Real.sqrt 3) - Real.sqrt 12 + (1 / 2)⁻¹ = 1 := 
by sorry

end expression_equals_one_l245_245618


namespace tea_or_coffee_indifference_l245_245218

open Classical

theorem tea_or_coffee_indifference : 
  (∀ (wants_tea wants_coffee : Prop), (wants_tea ∨ wants_coffee) → (¬ wants_tea → wants_coffee) ∧ (¬ wants_coffee → wants_tea)) → 
  (∀ (wants_tea wants_coffee : Prop), (wants_tea ∨ wants_coffee) → (¬ wants_tea → wants_coffee) ∧ (¬ wants_coffee → wants_tea)) :=
by
  sorry

end tea_or_coffee_indifference_l245_245218


namespace Laura_running_speed_l245_245813

theorem Laura_running_speed 
  (total_minutes : ℝ)
  (transition_minutes : ℝ)
  (bike_distance : ℝ)
  (bike_speed_expr : ℝ → ℝ)
  (run_distance : ℝ)
  (run_speed : ℝ)
  (total_time_in_hours : ℝ) :
  total_minutes = 150 →
  transition_minutes = 7 →
  bike_distance = 30 →
  bike_speed_expr = (λ x, 3 * x - 1) →
  run_distance = 10 →
  total_time_in_hours = (total_minutes - transition_minutes) / 60 →
  (∃ x : ℝ, (bike_distance / bike_speed_expr x + run_distance / x = total_time_in_hours) ∧ (run_speed = x)) →
  run_speed ≈ 8.54 :=
begin
  sorry
end

end Laura_running_speed_l245_245813


namespace probability_is_12_over_2907_l245_245153

noncomputable def probability_drawing_red_red_green : ℚ :=
  (3 / 19) * (2 / 18) * (4 / 17)

theorem probability_is_12_over_2907 :
  probability_drawing_red_red_green = 12 / 2907 :=
sorry

end probability_is_12_over_2907_l245_245153


namespace part1_part2_l245_245735

noncomputable def vector_a : ℝ^3 := sorry
noncomputable def vector_b : ℝ^3 := sorry
noncomputable def dot_product (u v : ℝ^3) : ℝ := u • v
noncomputable def magnitude (u : ℝ^3) : ℝ := real.sqrt (u • u)

axiom unit_vectors : magnitude vector_a = 1 ∧ magnitude vector_b = 1
axiom given_condition : (2 • vector_a - 3 • vector_b) • (2 • vector_a + vector_b) = 3

theorem part1 : dot_product vector_a vector_b = -1/2 :=
by sorry

theorem part2 : magnitude (2 • vector_a - vector_b) = real.sqrt 7 :=
by sorry

end part1_part2_l245_245735


namespace prisoner_strategy_l245_245004

def figure_strategy_exists : Prop :=
  ∃ (strategy : list (list ℕ → ℕ → ℕ)), 
    ∀ (captives : ℕ) (figures : list ℕ) (arrangement : list (list ℕ → ℕ → ℕ)),
    captives ≥ 3 → 
    (∀ i j, i ≠ j → arrangement[i].length = arrangement[j].length) →
    (∀ i, figures.length = captives) →
    ∃ (k : ℕ), 0 < k ∧ k ≤ captives ∧ arrangement[k] = figures[k]

theorem prisoner_strategy : figure_strategy_exists :=
sorry

end prisoner_strategy_l245_245004


namespace concurrency_of_ABC_l245_245875

variables {A B C H W S : Type} [Triangle A B C]

-- Condition variables
variable (wa : AngleBisector A B C)
variable (sb : Median B C A)
variable (hc : Altitude C A B)
variable (common_point : Concur B C (⊙H A))

-- Theorem statement
theorem concurrency_of_ABC (h1 : common_point A) :
  Concur wa sb hc :=
by
  sorry

end concurrency_of_ABC_l245_245875


namespace cannot_be_face_diagonals_l245_245521

theorem cannot_be_face_diagonals (a b c: ℕ) :
    ¬ ((5^2 + 7^2 = 9^2) ∨ (5^2 + 9^2 = 7^2) ∨ (7^2 + 9^2 = 5^2)) :=
by {
  -- To prove by contradiction, assume any of the equations is true
  intro h,
  -- Analyze each case
  cases h,
  { -- First case: 5^2 + 7^2 = 9^2
    have h₁ : 25 + 49 = 81, by simp,
    contradiction,
  },
  cases h,
  { -- Second case: 5^2 + 9^2 = 7^2
    have h₂ : 25 + 81 = 49, by simp,
    contradiction,
  },
  { -- Third case: 7^2 + 9^2 = 5^2
    have h₃ : 49 + 81 = 25, by simp,
    contradiction,
  }
}

end cannot_be_face_diagonals_l245_245521


namespace donut_area_proof_l245_245649

noncomputable def area_donut (r1 r2 : ℝ) (π : ℝ) : ℝ :=
  π * r2^2 - π * r1^2

theorem donut_area_proof :
  area_donut 7 10 real.pi = 51 * real.pi :=
by
  simp [area_donut]
  norm_num
  sorry

end donut_area_proof_l245_245649


namespace inequality_proof_l245_245015

noncomputable def a : ℝ := Real.logBase Real.pi Real.exp
noncomputable def b : ℝ := 2 ^ Real.cos (7 * Real.pi / 3)
noncomputable def c : ℝ := Real.logBase 3 (Real.sin (17 * Real.pi / 6))

theorem inequality_proof : b > a ∧ a > c := by
  sorry

end inequality_proof_l245_245015


namespace max_triplets_l245_245484

theorem max_triplets (n : ℕ) (h : n = 100) :
  ∃ N, (∀ (points : set (ℕ × ℕ)), points.card = n → 
  (∀ (A B C ∈ points), 
    (A.snd = B.snd → 
     B.fst = C.fst → 
     (A ≠ B ∧ B ≠ C ∧ A ≠ C)) → count_triplets points = N)) ∧ N = 8100 := sorry

end max_triplets_l245_245484


namespace find_C_l245_245567

theorem find_C :
  ∃ C, (A = 680) ∧ (A = B + 157) ∧ (B = C + 185) ∧ (C = 338) :=
by
  let A := 680
  let B := A - 157
  let C := B - 185
  use C
  split
  case a =>
    exact rfl
  case b =>
    exact rfl
  case c =>
    exact rfl
  case d =>
    sorry

end find_C_l245_245567


namespace countDistinctVals_9_l245_245695

noncomputable def countDistinctVals (x y z : ℕ) : ℕ :=
  if h : (x ≥ 100) ∧ (x ≤ 999) ∧ (y = (x % 100) * 10 + (x / 100) + (x / 10 % 10) * 100) ∧ (z = abs (x - y)) 
  then (λ (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9), 
         (finset.image (λ n, 90 * n) (finset.range 9)).card)
  else 0

theorem countDistinctVals_9 (x y z : ℕ) :
  (x ≥ 100) ∧ (x ≤ 999) ∧ (y = (x % 100) * 10 + (x / 100) + (x / 10 % 10) * 100) ∧ (z = abs (x - y)) 
  → countDistinctVals x y z = 9 :=
by
  sorry

end countDistinctVals_9_l245_245695


namespace josh_path_count_l245_245378

theorem josh_path_count (n : ℕ) : 
  let count_ways (steps: ℕ) := 2^steps in
  count_ways (n-1) = 2^(n-1) :=
by
  sorry

end josh_path_count_l245_245378


namespace two_A_minus_B_l245_245804

theorem two_A_minus_B (A B : ℝ) 
  (h1 : Real.tan (A - B - Real.pi) = 1 / 2) 
  (h2 : Real.tan (3 * Real.pi - B) = 1 / 7) : 
  2 * A - B = -3 * Real.pi / 4 :=
sorry

end two_A_minus_B_l245_245804


namespace josh_paths_to_top_center_l245_245368

/-- Define the grid --/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Josh's initial position --/
def start_pos : (ℕ × ℕ) := (0, 0)

/-- Define a function representing Josh's movement possibilities --/
def move_right (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1, pos.2 + 1)

def move_left_up (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1 + 1, pos.2 - 1)

/-- Define the goal position --/
def goal_pos (n : ℕ) : (ℕ × ℕ) :=
  (n - 1, 1)

/-- Theorem stating the required proof --/
theorem josh_paths_to_top_center {n : ℕ} (h : n > 0) : 
  let g := Grid.mk n 3 in
  ∃ (paths : ℕ), paths = 2^(n - 1) := 
  sorry

end josh_paths_to_top_center_l245_245368


namespace domain_of_sqrt_function_l245_245760

theorem domain_of_sqrt_function (m : ℝ) : (∀ x : ℝ, mx^2 + mx + 1 ≥ 0) ↔ 0 ≤ m ∧ m ≤ 4 := sorry

end domain_of_sqrt_function_l245_245760


namespace tg_sum_ctg_identity_l245_245874

theorem tg_sum_ctg_identity (α : ℝ) (n : ℕ) : 
  (∑ i in Finset.range (n + 1), (1 / 2^i) * Real.tan (α / 2^i)) = 
  (1 / 2^n) * Real.cot (α / 2^n) - 2 * Real.cot (2 * α) := sorry

end tg_sum_ctg_identity_l245_245874


namespace minimum_unhappiness_l245_245137

theorem minimum_unhappiness (n : ℕ) (h : n = 2017) : 
  ∃ (groups : list (list ℕ)), 
  groups.length = 15 ∧ 
  (∀ g ∈ groups, 1 ≤ g.length ∧ ∑ i in g, i = g.length * (g.head.getD 0 + g.length - 1) / 2) ∧ 
  (∑ g in groups, Real.ofRat (∑ i in g, i) / g.length = 1120.5) := 
by 
  sorry

end minimum_unhappiness_l245_245137


namespace abs_value_equality_l245_245192

theorem abs_value_equality : abs (-5 + 3) = 2 := by
  calc
    abs (-5 + 3) = abs (-2) : by rfl
    ...           = 2       : rfl

end abs_value_equality_l245_245192


namespace carol_additional_miles_l245_245623

theorem carol_additional_miles (distance_home : ℕ) (mpg : ℕ) (gallons : ℕ) 
  (h_home : distance_home = 220) (h_mpg : mpg = 20) (h_gallons : gallons = 16) :
  let total_distance := mpg * gallons
  in total_distance - distance_home = 100 :=
by
  sorry

end carol_additional_miles_l245_245623


namespace problem_part1_problem_part2_l245_245836

theorem problem_part1 
  (n : ℕ) (n_pos : 0 < n) 
  (f g : ℝ → ℝ) 
  (k : ℝ) 
  (a : Finₓ (n + 1) → ℝ) 
  (h_f : ∀ x, f x = ∑ i in Finₓ.range (n + 1), a i * x^(n - i.val)) 
  (h_g : ∀ x, g x = ∑ i in Finₓ.range (n + 1), (k / a i) * x^(n - i.val)) 
  (k_pos : 0 ≤ k) 
  (a_pos : ∀ i, 0 < a i) 
  (k_ge : k ≥ (1/4)) :
  (f (g 1)) * (g (f 1)) ≥ 4 * k :=
sorry

theorem problem_part2 
  (n : ℕ) (n_pos : 0 < n) 
  (f g : ℝ → ℝ) 
  (k : ℝ) 
  (a : Finₓ (n + 1) → ℝ) 
  (h_f : ∀ x, f x = ∑ i in Finₓ.range (n + 1), a i * x^(n - i.val)) 
  (h_g : ∀ x, g x = ∑ i in Finₓ.range (n + 1), (k / a i) * x^(n - i.val)) 
  (k_pos : 0 < k) 
  (k_le : k ≤ (1/4)) :
  ∃ (n' : ℕ) (a' : Finₓ (n' + 1) → ℝ), 
    (f (g 1)) * (g (f 1)) < 4 * k :=
sorry

end problem_part1_problem_part2_l245_245836


namespace solve_t_l245_245660

theorem solve_t :
  ∀ t : ℝ, sqrt (3 * sqrt (t - 3)) = real.sqrt (10 - t)^(1/4) → t = 37 / 10 :=
by sorry

end solve_t_l245_245660


namespace single_question_suffices_l245_245062

/-- We define a type for a person who can either be a human or a zombie -/
inductive Person
| human : Person
| zombie : Person

def response (p : Person) (baffirmeansyes : Bool) : Bool :=
  match p with
  | Person.human => baffirmeansyes
  | Person.zombie => !baffirmeansyes

def determine_identity (response : Bool) (baffirmeansyes : Bool) : Person :=
  if response = baffirmeansyes then Person.human else Person.zombie

theorem single_question_suffices (p : Person) (baffirmeansyes : Bool) :
  determine_identity (response p baffirmeansyes) baffirmeansyes = p := by
  cases p
  case human =>
    simp [determine_identity, response]
  case zombie =>
    simp [determine_identity, response]
    sorry   -- Proof content will go here

end single_question_suffices_l245_245062


namespace sum_of_valid_bs_l245_245659

def discriminant (a b c : ℤ) := b^2 - 4 * a * c

def has_rational_roots (a b c : ℤ) : Prop :=
  ∃ k : ℤ, discriminant a b c = k^2

def has_positive_root (a b c : ℤ) : Prop :=
  ∃ r : ℚ, 3 * r^2 + 7 * r + c = 0 ∧ r > 0

theorem sum_of_valid_bs : ∑ b in {1, 2, 3, 4}, 
  if (has_rational_roots 3 7 b ∧ has_positive_root 3 7 b) then b else 0 = 0 :=
by
  sorry

end sum_of_valid_bs_l245_245659


namespace max_distance_between_points_l245_245232

open EuclideanGeometry

noncomputable def maximum_distance_in_equilateral_triangle (T : Triangle ℝ) (h : T.is_equilateral ∧ T.side_length = 2) : ℝ := 
  2

theorem max_distance_between_points (T : Triangle ℝ) (h : T.is_equilateral ∧ T.side_length = 2) : 
  ∀ x y ∈ T, dist x y ≤ maximum_distance_in_equilateral_triangle T h := 
sorry

end max_distance_between_points_l245_245232


namespace total_money_l245_245091

def value_of_quarters (count: ℕ) : ℝ := count * 0.25
def value_of_dimes (count: ℕ) : ℝ := count * 0.10
def value_of_nickels (count: ℕ) : ℝ := count * 0.05
def value_of_pennies (count: ℕ) : ℝ := count * 0.01

theorem total_money (q d n p : ℕ) :
  q = 10 → d = 3 → n = 4 → p = 200 →
  value_of_quarters q + value_of_dimes d + value_of_nickels n + value_of_pennies p = 5.00 :=
by {
  intros,
  sorry
}

end total_money_l245_245091


namespace josh_paths_to_center_square_l245_245377

-- Definition of the problem's conditions based on given movements and grid size
def num_paths (n : Nat) : Nat :=
  2^(n-1)

-- Main statement
theorem josh_paths_to_center_square (n : Nat) : ∃ p : Nat, p = num_paths n :=
by
  exists num_paths n
  sorry

end josh_paths_to_center_square_l245_245377


namespace sequence_is_int_l245_245041

noncomputable theory
open_locale classical

-- Let a be an integer constant
variables {a : ℤ}

-- Define the sequence a_n with initial conditions and recurrence relation
def sequence (a : ℤ) (a1 a2 : ℤ) : ℕ → ℤ
| 0       := a1
| 1       := a2
| (n + 2) := (sequence n (n + 1))^2 / (sequence n) + a

-- Assume initial conditions are integers
variable (a1 a2 : ℤ)
-- Assume the fraction condition holds initially
variable (h : (a1^2 + a2^2 + a) / (a1 * a2) ∈ ℤ)

-- Prove that for all n, the sequence is an integer
theorem sequence_is_int (h1 : a1 ∈ ℤ) (h2 : a2 ∈ ℤ) (h3 : (a1^2 + a2^2 + a) / (a1 * a2) ∈ ℤ) :
  ∀ n : ℕ, (sequence a a1 a2 n) ∈ ℤ :=
sorry -- proof not required

end sequence_is_int_l245_245041


namespace volume_difference_spheres_l245_245923

theorem volume_difference_spheres {π : Real} :
  let r_small := 4
  let r_large := 7
  let V_small := (4 / 3) * π * r_small^3
  let V_large := (4 / 3) * π * r_large^3
  V_large - V_small = 372 * π :=
by
  let r_small := 4
  let r_large := 7
  let V_small := (4 / 3 : Real) * π * r_small^3
  let V_large := (4 / 3 : Real) * π * r_large^3
  calc
    V_large - V_small = (4 / 3) * π * (r_large^3 - r_small^3) : by ring
    ... = 372 * π : by sorry

end volume_difference_spheres_l245_245923


namespace number_of_members_l245_245858

def socks_cost : ℕ := 3
def shirt_cost : ℕ := socks_cost + 4
def cap_cost : ℝ := socks_cost / 2

def cost_per_member : ℝ := 2 * socks_cost + 2 * shirt_cost + cap_cost
def total_cost : ℝ := 3213

theorem number_of_members (n : ℕ) : n * cost_per_member = total_cost → n = 149 := by
  sorry

end number_of_members_l245_245858


namespace solve_system_l245_245059

theorem solve_system :
  (4 * (cos x) ^ 2 * (sin (x / 6)) ^ 2 + 4 * sin (x / 6) - 4 * (sin x) ^ 2 * sin (x / 6) + 1 = 0) ∧
  (sin (x / 4) = sqrt (cos y)) →
  (∃ m n : ℤ,
    (x = 11 * π + 24 * π * m ∧ y = π / 3 + 2 * π * n) ∨ 
    (x = 11 * π + 24 * π * m ∧ y = -π / 3 + 2 * π * n) ∨ 
    (x = -5 * π + 24 * π * m ∧ y = π / 3 + 2 * π * n) ∨ 
    (x = -5 * π + 24 * π * m ∧ y = -π / 3 + 2 * π * n)) :=
by
  sorry

end solve_system_l245_245059


namespace smallest_positive_difference_exists_l245_245510

-- Definition of how Vovochka sums two three-digit numbers
def vovochkaSum (x y : ℕ) : ℕ :=
  let h1 := (x / 100) + (y / 100)
  let t1 := ((x / 10) % 10) + ((y / 10) % 10)
  let u1 := (x % 10) + (y % 10)
  h1 * 1000 + t1 * 100 + u1

-- Definition for correct sum of two numbers
def correctSum (x y : ℕ) : ℕ := x + y

-- Function to find difference
def difference (x y : ℕ) : ℕ :=
  abs (vovochkaSum x y - correctSum x y)

-- Proof the smallest positive difference between Vovochka's sum and the correct sum
theorem smallest_positive_difference_exists :
  ∃ x y : ℕ, (x < 1000) ∧ (y < 1000) ∧ difference x y > 0 ∧ 
  (∀ a b : ℕ, (a < 1000) ∧ (b < 1000) ∧ difference a b > 0 → difference x y ≤ difference a b) :=
sorry

end smallest_positive_difference_exists_l245_245510


namespace factor_expression_eq_l245_245223

theorem factor_expression_eq (x : ℤ) : 75 * x + 50 = 25 * (3 * x + 2) :=
by
  -- The actual proof is omitted
  sorry

end factor_expression_eq_l245_245223


namespace calc_sqrt_div_l245_245191

theorem calc_sqrt_div : 
  (√3) / (√(1 / 3) + √(3 / 16)) = (12 / 7) :=
by
  sorry

end calc_sqrt_div_l245_245191


namespace f_f_of_one_ninth_l245_245292

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then Real.log x / Real.log 2 else 2 ^ x

theorem f_f_of_one_ninth :
  f (f (1 / 9)) = 1 / 9 :=
by
  sorry

end f_f_of_one_ninth_l245_245292


namespace walk_time_is_correct_l245_245955

noncomputable def time_to_walk_one_block := 
  let blocks := 18
  let bike_time_per_block := 20 -- seconds
  let additional_walk_time := 12 * 60 -- 12 minutes in seconds
  let walk_time := blocks * bike_time_per_block + additional_walk_time
  walk_time / blocks

theorem walk_time_is_correct : 
  let W := time_to_walk_one_block
  W = 60 := by
    sorry -- proof goes here

end walk_time_is_correct_l245_245955


namespace num_non_congruent_triangles_with_perimeter_18_l245_245211

theorem num_non_congruent_triangles_with_perimeter_18 :
  ∃ triangles : finset (ℤ × ℤ × ℤ),
  (∀ t ∈ triangles, let (a, b, c) := t in a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 18 ∧
    a + b > c ∧ a + c > b ∧ b + c > a) ∧
  triangles.card = 11 :=
begin
  sorry
end

end num_non_congruent_triangles_with_perimeter_18_l245_245211


namespace geometric_series_solution_l245_245200

/-- Given the infinite geometric series properties, prove the value of x such that
(1 + 1/2 + 1/4 + 1/8 + ...) * (1 - 1/2 + 1/4 - 1/8 + ...) = (1 + 1/x + 1/x^2 + 1/x^3 + ...).
-/
theorem geometric_series_solution :
  (∑' n : ℕ, (1 / 2) ^ n) * (∑' n : ℕ, (-1 / 2) ^ n) = ∑' n : ℕ, (1 : ℚ) / 4 ^ n →
  ∃ x : ℚ, (x = 4) :=
by
  intro h
  use 4
  sorry

end geometric_series_solution_l245_245200


namespace incenter_of_triangle_l245_245806

theorem incenter_of_triangle
  (A B C M O : Point)
  (angle_BMC_eq : ∠ B M C = 90 + 1/2 * ∠ B A C)
  (line_AM_contains_O : O ∈ line A M)
  (circumcenter_of_BMC : circumcenter (triangle B M C) = O) :
  incenter (triangle A B C) = M :=
sorry

end incenter_of_triangle_l245_245806


namespace solve_problem_l245_245017

-- Define the constants c and d
variables (c d : ℝ)

-- Define the conditions of the problem
def condition1 : Prop := 
  (∀ x : ℝ, (x + c) * (x + d) * (x + 15) = 0 ↔ x = -c ∨ x = -d ∨ x = -15) ∧
  -4 ≠ -c ∧ -4 ≠ -d ∧ -4 ≠ -15

def condition2 : Prop := 
  (∀ x : ℝ, (x + 3 * c) * (x + 4) * (x + 9) = 0 ↔ x = -4) ∧
  d ≠ -4 ∧ d ≠ -15

-- We need to prove this final result under the given conditions
theorem solve_problem (h1 : condition1 c d) (h2 : condition2 c d) : 100 * c + d = -291 := 
  sorry

end solve_problem_l245_245017


namespace meaningful_fraction_l245_245906

theorem meaningful_fraction {x : ℝ} : (x - 2) ≠ 0 ↔ x ≠ 2 :=
by
  sorry

end meaningful_fraction_l245_245906


namespace solution_set_inequality_l245_245396

noncomputable def f : ℝ → ℝ := sorry
noncomputable def derivative_f : ℝ → ℝ := sorry -- f' is the derivative of f

-- Conditions
axiom f_domain {x : ℝ} (h1 : 0 < x) : f x ≠ 0
axiom derivative_condition {x : ℝ} (h1 : 0 < x) : f x + x * derivative_f x > 0
axiom initial_value : f 1 = 2

-- Proof that the solution set of the inequality f(x) < 2/x is (0, 1)
theorem solution_set_inequality : ∀ x : ℝ, 0 < x ∧ x < 1 → f x < 2 / x := sorry

end solution_set_inequality_l245_245396


namespace first_player_winning_strategy_l245_245931
noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem first_player_winning_strategy (x1 y1 : ℕ)
    (h1 : x1 > 0) (h2 : y1 > 0) :
    (x1 / y1 = 1) ∨ 
    (x1 / y1 > golden_ratio) ∨ 
    (x1 / y1 < 1 / golden_ratio) :=
sorry

end first_player_winning_strategy_l245_245931


namespace square_area_divided_into_equal_rectangles_l245_245586

theorem square_area_divided_into_equal_rectangles (w : ℝ) (a : ℝ) (h : 5 = w) :
  (∃ s : ℝ, s * s = a ∧ s * s / 5 = a / 5) ↔ a = 400 :=
by
  sorry

end square_area_divided_into_equal_rectangles_l245_245586


namespace average_speed_second_bus_l245_245909

theorem average_speed_second_bus (x : ℝ) (h1 : x > 0) :
  (12 / x) - (12 / (1.2 * x)) = 3 / 60 :=
by
  sorry

end average_speed_second_bus_l245_245909


namespace find_a_l245_245020

theorem find_a (a : ℕ) (h_pos : 0 < a)
  (h_cube : ∀ n : ℕ, 0 < n → ∃ k : ℤ, 4 * ((a : ℤ) ^ n + 1) = k^3) :
  a = 1 :=
sorry

end find_a_l245_245020


namespace josh_path_count_l245_245380

theorem josh_path_count (n : ℕ) : 
  let count_ways (steps: ℕ) := 2^steps in
  count_ways (n-1) = 2^(n-1) :=
by
  sorry

end josh_path_count_l245_245380


namespace cost_of_one_shirt_l245_245314

-- Definitions based on the conditions given
variables (J S : ℝ)

-- First condition: 3 pairs of jeans and 2 shirts cost $69
def condition1 : Prop := 3 * J + 2 * S = 69

-- Second condition: 2 pairs of jeans and 3 shirts cost $61
def condition2 : Prop := 2 * J + 3 * S = 61

-- The theorem to prove that the cost of one shirt is $9
theorem cost_of_one_shirt (J S : ℝ) (h1 : condition1 J S) (h2 : condition2 J S) : S = 9 :=
by
  sorry

end cost_of_one_shirt_l245_245314


namespace charlie_cost_per_gb_l245_245865

noncomputable def total_data_usage (w1 w2 w3 w4 : ℕ) : ℕ := w1 + w2 + w3 + w4

noncomputable def data_over_limit (total_data usage_limit: ℕ) : ℕ :=
  if total_data > usage_limit then total_data - usage_limit else 0

noncomputable def cost_per_gb (extra_cost data_over_limit: ℕ) : ℕ :=
  if data_over_limit > 0 then extra_cost / data_over_limit else 0

theorem charlie_cost_per_gb :
  let D := 8
  let w1 := 2
  let w2 := 3
  let w3 := 5
  let w4 := 10
  let C := 120
  let total_data := total_data_usage w1 w2 w3 w4
  let data_over := data_over_limit total_data D
  C / data_over = 10 := by
  -- Sorry to skip the proof
  sorry

end charlie_cost_per_gb_l245_245865


namespace DogHeight_is_24_l245_245624

-- Define the given conditions as Lean definitions (variables and equations)
variable (CarterHeight DogHeight BettyHeight : ℝ)

-- Assume the conditions given in the problem
axiom h1 : CarterHeight = 2 * DogHeight
axiom h2 : BettyHeight + 12 = CarterHeight
axiom h3 : BettyHeight = 36

-- State the proposition (the height of Carter's dog)
theorem DogHeight_is_24 : DogHeight = 24 :=
by
  -- Proof goes here
  sorry

end DogHeight_is_24_l245_245624


namespace solution_set_inequality_l245_245670

theorem solution_set_inequality (a : ℕ) (h : ∀ x : ℝ, (a-2) * x > (a-2) → x < 1) : a = 0 ∨ a = 1 :=
by
  sorry

end solution_set_inequality_l245_245670


namespace smallest_positive_integer_cube_ends_in_632_l245_245658

theorem smallest_positive_integer_cube_ends_in_632 :
  ∃ n : ℕ, (n > 0) ∧ (n^3 % 1000 = 632) ∧ ∀ m : ℕ, (m > 0) ∧ (m^3 % 1000 = 632) → n ≤ m := 
sorry

end smallest_positive_integer_cube_ends_in_632_l245_245658


namespace proposition_B_l245_245667

-- Definitions of planes and lines
variable {Plane : Type}
variable {Line : Type}
variable (α β : Plane)
variable (m n : Line)

-- Definitions of parallel and perpendicular relationships
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (_perpendicular : Line → Line → Prop)

-- Theorem statement
theorem proposition_B (h1 : perpendicular m α) (h2 : parallel n α) : _perpendicular m n :=
sorry

end proposition_B_l245_245667


namespace graph_shift_right_l245_245929

noncomputable def function_shift : Prop :=
  ∀ (x : ℝ), (2 * sin (3 * x - (π / 5))) = (2 * sin (3 * (x - π / 15)))

theorem graph_shift_right : function_shift := 
by
  sorry

end graph_shift_right_l245_245929


namespace inequality_proof_l245_245990

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l245_245990


namespace melinda_textbook_prob_l245_245857

theorem melinda_textbook_prob :
  let total_ways := (Nat.choose 15 4) * (Nat.choose 11 5),
      favorable_ways_box_5 := (Nat.choose 11 1) * (Nat.choose 10 4),
      favorable_ways_box_6 := (Nat.choose 11 2) * (Nat.choose 9 4),
      favorable_outcomes := favorable_ways_box_5 + favorable_ways_box_6,
      probability := favorable_outcomes / total_ways in
  277 = 4 + 273 ∧ probability = 4 / 273 := by
  let total_ways := (Nat.choose 15 4) * (Nat.choose 11 5)
  let favorable_ways_box_5 := (Nat.choose 11 1) * (Nat.choose 10 4)
  let favorable_ways_box_6 := (Nat.choose 11 2) * (Nat.choose 9 4)
  let favorable_outcomes := favorable_ways_box_5 + favorable_ways_box_6
  let probability := favorable_outcomes / total_ways
  have H1 : 277 = 4 + 273 := sorry
  have H2 : probability = 4 / 273 := sorry
  exact ⟨H1, H2⟩

end melinda_textbook_prob_l245_245857


namespace stratified_sampling_l245_245581

def total_teachers := 300
def senior_teachers := 90
def intermediate_teachers := 150
def junior_teachers := 60
def sample_size := 60

theorem stratified_sampling :
  let prob := sample_size / total_teachers in
  let senior_sampled := senior_teachers * prob in
  let intermediate_sampled := intermediate_teachers * prob in
  let junior_sampled := junior_teachers * prob in
  senior_sampled = 18 ∧ intermediate_sampled = 30 ∧ junior_sampled = 12 :=
by
  have prob_calc : prob = 60 / 300 := by sorry
  have senior_calc : senior_sampled = 90 * (60 / 300) := by sorry
  have intermediate_calc : intermediate_sampled = 150 * (60 / 300) := by sorry
  have junior_calc : junior_sampled = 60 * (60 / 300) := by sorry
  have senior_val : 90 * (60 / 300) = 18 := by sorry
  have intermediate_val : 150 * (60 / 300) = 30 := by sorry
  have junior_val : 60 * (60 / 300) = 12 := by sorry
  exact ⟨senior_val, intermediate_val, junior_val⟩

end stratified_sampling_l245_245581


namespace balls_sold_l245_245413

theorem balls_sold
  (SP : ℕ) (CP_per_ball : ℕ) (loss_equal_cp_5_balls : ℕ) (n : ℕ) 
  (h1 : SP = 720)
  (h2 : CP_per_ball = 72)
  (h3 : loss_equal_cp_5_balls = CP_per_ball * 5) :
  n = 15 := by
    -- assert the cost price for n balls
    let CP_total := CP_per_ball * n
    -- assert the loss is defined as CP_total - SP
    let loss := CP_total - SP
    -- now assert the loss condition
    have h4 : loss = loss_equal_cp_5_balls := by
      have loss_value : loss = 
        CP_per_ball * n - SP := rfl
      have loss_cp5 : loss_equal_cp_5_balls = CP_per_ball * 5 := h3
      rw [h2, h1, loss_value, loss_cp5]
      sorry
    -- conclude the number of balls sold
    sorry

end balls_sold_l245_245413


namespace sum_of_first_13_terms_l245_245683

theorem sum_of_first_13_terms (m : ℤ) (a : ℕ → ℤ) 
  (h_seq_def : ∀ n : ℕ, (m + 3) * n - (2 * m + 4) * a n - m - 9 = 0) :
  ∑ n in finset.range 13, a (n + 1) = 39 :=
by
  sorry

end sum_of_first_13_terms_l245_245683


namespace sum_of_29_12_23_is_64_sixtyfour_is_two_to_six_l245_245480

theorem sum_of_29_12_23_is_64: 29 + 12 + 23 = 64 := sorry

theorem sixtyfour_is_two_to_six:
  64 = 2^6 := sorry

end sum_of_29_12_23_is_64_sixtyfour_is_two_to_six_l245_245480


namespace cube_root_two_irrational_l245_245536

theorem cube_root_two_irrational : ∛2 ∉ ℚ :=
sorry

end cube_root_two_irrational_l245_245536


namespace f_odd_f_periodic_f_def_on_interval_problem_solution_l245_245602

noncomputable def f : ℝ → ℝ := 
sorry

theorem f_odd (x : ℝ) : f (-x) = -f x := 
sorry

theorem f_periodic (x : ℝ) : f (x + 4) = f x := 
sorry

theorem f_def_on_interval (x : ℝ) (h : -2 < x ∧ x < 0) : f x = 2 ^ x :=
sorry

theorem problem_solution : f 2015 - f 2014 = 1 / 2 :=
sorry

end f_odd_f_periodic_f_def_on_interval_problem_solution_l245_245602


namespace greatest_prime_factor_391_l245_245106

theorem greatest_prime_factor_391 : 
  greatestPrimeFactor 391 = 23 :=
sorry

end greatest_prime_factor_391_l245_245106


namespace determine_weights_l245_245884

-- Definitions
variable {W : Type} [AddCommGroup W] [OrderedAddCommMonoid W]
variable (w : Fin 20 → W) -- List of weights for 20 people
variable (s : W) -- Total sum of weights
variable (lower upper : W) -- Lower and upper weight limits

-- Conditions
def weight_constraints : Prop :=
  (∀ i, lower ≤ w i ∧ w i ≤ upper) ∧ (Finset.univ.sum w = s)

-- Problem statement
theorem determine_weights (w : Fin 20 → ℝ) :
  weight_constraints w 60 90 3040 →
  ∃ w : Fin 20 → ℝ, weight_constraints w 60 90 3040 := by
  sorry

end determine_weights_l245_245884


namespace valid_triangle_set_invalid_triangle_set_1_invalid_triangle_set_2_invalid_triangle_set_3_l245_245951

theorem valid_triangle_set {a b c : ℕ} : (a = 5) ∧ (b = 6) ∧ (c = 10) →
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) :=
begin
  intros h,
  cases h with ha hbc,
  cases hbc with hb hc,
  rw [ha, hb, hc],
  exact ⟨by linarith, by linarith, by linarith⟩,
end

theorem invalid_triangle_set_1 {a b c : ℕ} : (a = 2) ∧ (b = 2) ∧ (c = 4) →
  ¬((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
begin
  intros h,
  cases h with ha hbc,
  cases hbc with hb hc,
  rw [ha, hb, hc],
  simp [not_and_distrib],
  right,
  linarith,
end

theorem invalid_triangle_set_2 {a b c : ℕ} : (a = 3) ∧ (b = 4) ∧ (c = 8) →
  ¬((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
begin
  intros h,
  cases h with ha hbc,
  cases hbc with hb hc,
  rw [ha, hb, hc],
  simp [not_and_distrib],
  left,
  linarith,
end

theorem invalid_triangle_set_3 {a b c : ℕ} : (a = 4) ∧ (b = 5) ∧ (c = 10) →
  ¬((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
begin
  intros h,
  cases h with ha hbc,
  cases hbc with hb hc,
  rw [ha, hb, hc],
  simp [not_and_distrib],
  left,
  linarith,
end

end valid_triangle_set_invalid_triangle_set_1_invalid_triangle_set_2_invalid_triangle_set_3_l245_245951


namespace lisa_total_spoons_l245_245844

def num_children := 4
def spoons_per_child := 3
def decorative_spoons := 2
def large_spoons := 10
def teaspoons := 15

def total_spoons := num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

theorem lisa_total_spoons : total_spoons = 39 := by
  sorry

end lisa_total_spoons_l245_245844


namespace sum_of_distances_geq_6006_l245_245816

noncomputable def rectangle_vertices (c : Circle) : Rectangle := sorry

theorem sum_of_distances_geq_6006
  (points : Fin 2002 → Point) 
  (c : Circle) 
  (h₁ : ∀ p, dist p c.center ≤ 1)
  (rect : Rectangle)
  (h₂ : Inscribes rect c) :
  ∃ M N P : Point,
    (∑ i, dist (points i) M) + (∑ i, dist (points i) N) + (∑ i, dist (points i) P) ≥ 6006 :=
sorry

end sum_of_distances_geq_6006_l245_245816


namespace train_length_180_meters_l245_245129

theorem train_length_180_meters 
  (s_k : ℝ) (t : ℝ)
  (h_speed : s_k = 72)
  (h_time : t = 9) :
  ∃ l : ℝ, l = 180 :=
by
  -- Conversion factor from km/hr to m/s
  let s_m := s_k * (1000 / 3600)
  have h_s_m : s_m = 20, from sorry

  -- Distance formula: Distance = Speed × Time
  let l := s_m * t
  have h_l : l = 180, from sorry

  -- The length of the train is 180 meters
  use l
  exact h_l

end train_length_180_meters_l245_245129


namespace candle_problem_l245_245099

-- Define the initial heights and burn rates of the candles
def heightA (t : ℝ) : ℝ := 12 - 2 * t
def heightB (t : ℝ) : ℝ := 15 - 3 * t

-- Lean theorem statement for the given problem
theorem candle_problem : ∃ t : ℝ, (heightA t = (1/3) * heightB t) ∧ t = 7 :=
by
  -- This is to keep the theorem statement valid without the proof
  sorry

end candle_problem_l245_245099


namespace greatest_value_of_a_b_c_l245_245662

section digit_proof
variables (a b c : ℕ) (n : ℕ)
  (A_n B_n C_n : ℕ)

-- Defining the conditions
def condition1 := (n > 0)
def condition2 := (a > 0 ∧ a < 10)
def condition3 := (b > 0 ∧ b < 10)
def condition4 := (c > 0 ∧ c < 10)
def definition_A_n := A_n = a * (1 + 8 + 8^2 + ... + 8^(n-1))
def definition_B_n := B_n = b * (1 + 6 + 6^2 + ... + 6^(n-1))
def definition_C_n := C_n = c * (1 + 10 + 10^2 + ... + 10^(3n-1))

-- Main proof statement
theorem greatest_value_of_a_b_c :
  ∃ a b c, (a > 0 ∧ a < 10) ∧ (b > 0 ∧ b < 10) ∧ (c > 0 ∧ c < 10) ∧
  (∃ n m, n ≠ m ∧ 
    c * (10^(3*n) - 1) / 9 - b * (6^n - 1) / 5 = a^3 * ((8^n - 1) / 7)^3 ∧ 
    c * (10^(3*m) - 1) / 9 - b * (6^m - 1) / 5 = a^3 * ((8^m - 1) / 7)^3) ∧
  (a + b + c = 21) :=
sorry
end digit_proof

end greatest_value_of_a_b_c_l245_245662


namespace josh_paths_to_center_square_l245_245375

-- Definition of the problem's conditions based on given movements and grid size
def num_paths (n : Nat) : Nat :=
  2^(n-1)

-- Main statement
theorem josh_paths_to_center_square (n : Nat) : ∃ p : Nat, p = num_paths n :=
by
  exists num_paths n
  sorry

end josh_paths_to_center_square_l245_245375


namespace perimeter_of_quadrilateral_l245_245478

theorem perimeter_of_quadrilateral (b m : ℝ) (ABC : Triangle) (D E : Point)
    (h1 : ABC.right_angle_at C)
    (h2 : ABC.shorter_leg = segment AC)
    (h3 : length segment AC = b)
    (h4 : D ∈ segment AB)
    (h5 : length segment BD = length segment BC)
    (h6 : E ∈ segment BC)
    (h7 : length segment DE = m)
    (h8 : length segment BE = m) :
    length perimeter (quadrilateral A D E C) = 2 * m + b := 
sorry

end perimeter_of_quadrilateral_l245_245478


namespace angle_bac_2_angle_abm_l245_245098

open EuclideanGeometry

noncomputable def triangle_properties (A B C M K: Point) :=
  BC < AC < AB ∧
  midpoint M A C ∧
  on_line K A B ∧
  CK = BC ∧
  BK = AC

theorem angle_bac_2_angle_abm {A B C M K : Point} :
  triangle_properties A B C M K → ∠BAC = 2 * ∠ABM :=
by
  intros
  sorry

end angle_bac_2_angle_abm_l245_245098


namespace problem_1_l245_245267

theorem problem_1 (f h g : ℝ → ℝ) (k b : ℝ) (D : set ℝ)
  (hf : ∀ x, f x = x^2 + x)
  (hh : ∀ x, h x = -x^2 + x)
  (hg : ∀ x, g x = k*x + b)
  (cond : ∀ x ∈ D, f x ≥ g x ∧ g x ≥ h x) :
  g = λ x, x := sorry

end problem_1_l245_245267


namespace find_d_squared_l245_245555

noncomputable def g (z : ℂ) (c d : ℝ) : ℂ := (c + d * complex.I) * z

theorem find_d_squared
  (c d : ℝ)
  (z : ℂ)
  (h1 : ∀ z : ℂ, complex.abs ((g z c d) - z) = complex.abs (g z c d))
  (h2 : complex.abs (c + d * complex.I) = 7) :
  d^2 = 195 / 4 := sorry

end find_d_squared_l245_245555


namespace loop_termination_l245_245052

variable (n α : ℕ)
def initial_l : ℕ := 0
def initial_m : ℕ := n - 1
def loop_n1 (l m : ℕ) : ℕ × ℕ := (l + 1, m / 2)

theorem loop_termination (β : ℕ) (hα : α = log2 (n-1) - log2 β):
  ∀ l m, 
  (l, m) = (initial_l, initial_m) → 
  (∀ k, k ≤ α → (l, m) = ((loop_n1^[k]) (initial_l, initial_m))) → 
  (∃ β, l = α ∧ m = β) :=
by
  sorry

end loop_termination_l245_245052


namespace prove_three_cell_corners_l245_245032

noncomputable def possible_three_cell_corners (x y : ℕ) :=
  3 * x + 4 * y = 22

theorem prove_three_cell_corners :
  ∃ x y : ℕ, possible_three_cell_corners x y ∧ (x = 2 ∨ x = 6) :=
begin
  sorry
end

end prove_three_cell_corners_l245_245032


namespace root_in_interval_l245_245311

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 3

theorem root_in_interval : ∃ x ∈ Ioo (1/2 : ℝ) 1, f x = 0 :=
by
  have h1 : f (1/2) < 0 := by sorry
  have h2 : f 1 > 0 := by sorry
  apply IntermediateValueTheorem
  exact h1
  exact h2
  sorry -- Provide the actual continuity of f(x)

end root_in_interval_l245_245311


namespace smallest_a1_l245_245828

noncomputable def sequence (a : ℕ → ℝ) := ∀ n > 1, a n = 13 * a (n - 1) - 2 * n

theorem smallest_a1 (a : ℕ → ℝ) (h : ∀ n > 1, a n = 13 * a (n - 1) - 2 * n) :
  (a 1 ≥ 3 / 8) := sorry

end smallest_a1_l245_245828


namespace contribution_is_6_l245_245239

-- Defining the earnings of each friend
def earning_1 : ℕ := 18
def earning_2 : ℕ := 22
def earning_3 : ℕ := 30
def earning_4 : ℕ := 35
def earning_5 : ℕ := 45

-- Defining the modified contribution for the highest earner
def modified_earning_5 : ℕ := 40

-- Calculate the total adjusted earnings
def total_earnings : ℕ := earning_1 + earning_2 + earning_3 + earning_4 + modified_earning_5

-- Calculate the equal share each friend should receive
def equal_share : ℕ := total_earnings / 5

-- Calculate the contribution needed from the friend who earned $35 to match the equal share
def contribution_from_earning_4 : ℕ := earning_4 - equal_share

-- Stating the proof problem
theorem contribution_is_6 : contribution_from_earning_4 = 6 := by
  sorry

end contribution_is_6_l245_245239


namespace seashells_initial_count_l245_245593

theorem seashells_initial_count (S : ℕ)
  (h1 : S - 70 = 2 * 55) : S = 180 :=
by
  sorry

end seashells_initial_count_l245_245593


namespace smallest_positive_difference_l245_245500

-- Define the method used by Vovochka to sum numbers.
def sum_without_carrying (a b : ℕ) : ℕ :=
  let ha := a / 100 
  let ta := (a % 100) / 10 
  let ua := a % 10
  let hb := b / 100 
  let tb := (b % 100) / 10 
  let ub := b % 10
  (ha + hb) * 1000 + (ta + tb) * 100 + (ua + ub)

-- Define the correct method to sum numbers.
def correct_sum (a b : ℕ) : ℕ :=
  a + b

-- The smallest positive difference between the sum without carrying and the correct sum.
theorem smallest_positive_difference (a b : ℕ) (h : 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000) :
  (sum_without_carrying a b - correct_sum a b > 0) →
  (∀ c d, 100 ≤ c ∧ c < 1000 ∧ 100 ≤ d ∧ d < 1000 →
    (sum_without_carrying c d - correct_sum c d > 0 → sum_without_carrying c d - correct_sum c d ≥ 1800 )) :=
begin
  sorry
end

end smallest_positive_difference_l245_245500


namespace teacher_age_is_45_l245_245065

def avg_age_of_students := 14
def num_students := 30
def avg_age_with_teacher := 15
def num_people_with_teacher := 31

def total_age_of_students := avg_age_of_students * num_students
def total_age_with_teacher := avg_age_with_teacher * num_people_with_teacher

theorem teacher_age_is_45 : (total_age_with_teacher - total_age_of_students = 45) :=
by
  sorry

end teacher_age_is_45_l245_245065


namespace cards_per_deck_l245_245407

theorem cards_per_deck (decks : ℕ) (cards_per_layer : ℕ) (layers : ℕ) 
  (h_decks : decks = 16) 
  (h_cards_per_layer : cards_per_layer = 26) 
  (h_layers : layers = 32) 
  (total_cards_used : ℕ := cards_per_layer * layers) 
  (number_of_cards_per_deck : ℕ := total_cards_used / decks) :
  number_of_cards_per_deck = 52 :=
by 
  sorry

end cards_per_deck_l245_245407


namespace initially_calculated_average_weight_l245_245888

-- Define the conditions
def num_boys : ℕ := 20
def correct_average_weight : ℝ := 58.7
def misread_weight : ℝ := 56
def correct_weight : ℝ := 62
def weight_difference : ℝ := correct_weight - misread_weight

-- State the goal
theorem initially_calculated_average_weight :
  let correct_total_weight := correct_average_weight * num_boys
  let initial_total_weight := correct_total_weight - weight_difference
  let initially_calculated_weight := initial_total_weight / num_boys
  initially_calculated_weight = 58.4 :=
by
  sorry

end initially_calculated_average_weight_l245_245888


namespace triangle_segment_equality_l245_245006

theorem triangle_segment_equality (A B C K L P M : Type) [triangle A B C]
(hK : is_on_segment K A B) (hL : is_on_segment L A C) (hBK_CL : segment_length B K = segment_length C L)
(hP : intersection_point CK BL P) (hM : parallel_through P (angle_bisector A B C) (intersection_segment AC M)) :
segment_length C M = segment_length A B :=
begin
  sorry
end

end triangle_segment_equality_l245_245006


namespace overlapping_circles_area_l245_245798

/-
Prove that the area of the overlapping region between two circles with centers 3 feet apart, 
where the smaller circle has a radius of 3 feet and the larger circle has a radius of 5 feet, 
is equal to 9 * real.arccos (-7 / 18) + 25 * real.arccos (5 / 6) - (1 / 2) * real.sqrt 275 square feet.
-/
theorem overlapping_circles_area :
  let r1 := 3
  let r2 := 5
  let d := 3
  r1^2 * real.arccos ((d^2 + r1^2 - r2^2)/(2 * d * r1)) + 
  r2^2 * real.arccos ((d^2 + r2^2 - r1^2)/(2 * d * r2)) - 
  (1/2) * real.sqrt ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
  = 9 * real.arccos (-7/18) + 25 * real.arccos (5/6) - (1/2) * real.sqrt 275 := 
by 
  sorry

end overlapping_circles_area_l245_245798


namespace unique_solution_inequality_l245_245767

theorem unique_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, -3 ≤ x^2 - 2 * a * x + a ∧ x^2 - 2 * a * x + a ≤ -2 → ∃! x : ℝ, x^2 - 2 * a * x + a = -2) ↔ (a = 2 ∨ a = -1) :=
sorry

end unique_solution_inequality_l245_245767


namespace percentage_of_non_union_employees_are_women_l245_245775

variable (E : ℝ) -- Total number of employees
-- Conditions given in the problem
def unionized_employees := 0.60 * E
def men_in_union := 0.70 * unionized_employees
def non_unionized_employees := E - unionized_employees
def women_in_non_union := 0.65 * non_unionized_employees

-- Statement to prove
theorem percentage_of_non_union_employees_are_women (h_unionized : 0.60 * E ≠ 0) : 
  0.65 * (E - 0.60 * E) / (E - 0.60 * E) * 100 = 65 := 
by
  sorry

end percentage_of_non_union_employees_are_women_l245_245775


namespace min_dist_sum_sq_l245_245803

open Real

variables {A B C P : Type} [metric_space P] [norm_space ℝ P]
variables {x : P}

-- Conditions:
def is_right_angle (p a b : P) : Prop := ∠a p b = π / 2
def PA : ℝ := 4
def PB : ℝ := 3
def PC : ℝ := 3

-- Proof problem (statement only):
theorem min_dist_sum_sq (A B C P : P)
  (h₁ : is_right_angle P A B)
  (h₂ : is_right_angle P B C)
  (h₃ : is_right_angle P C A)
  (dPA : dist P A = PA)
  (dPB : dist P B = PB)
  (dPC : dist P C = PC) :
  ∃ M : P, (dist M (line_through P A))^2 + (dist M (line_through P B))^2 + (dist M (line_through P C))^2 = 144 / 41 := 
sorry

end min_dist_sum_sq_l245_245803


namespace maximize_x3y4_correct_l245_245399

noncomputable def maximize_x3y4 : ℝ × ℝ :=
  let x := 160 / 7
  let y := 120 / 7
  (x, y)

theorem maximize_x3y4_correct :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 40 ∧ (x, y) = maximize_x3y4 ∧ 
  ∀ (x' y' : ℝ), 0 < x' ∧ 0 < y' ∧ x' + y' = 40 → x ^ 3 * y ^ 4 ≥ x' ^ 3 * y' ^ 4 :=
by
  sorry

end maximize_x3y4_correct_l245_245399


namespace period_of_f_range_of_f_in_interval_l245_245738

noncomputable def f (x : ℝ) (ω : ℝ) (λ : ℝ) := 
  let a: ℝ × ℝ := (cos (ω * x) - sin (ω * x), sin (ω * x))
  let b: ℝ × ℝ := (- cos (ω * x) - sin (ω * x), 2 * sqrt 3 * cos (ω * x))
  a.1 * b.1 + a.2 * b.2 + λ

theorem period_of_f (ω : ℝ) (λ : ℝ) (hω : ω ∈ set.Ioo (1/2) 1) :
  (∃ T > 0, ∀ x, f x ω λ = f (x + T) ω λ) ∧
  (∀ T > 0, (∀ x, f x ω λ = f (x + T) ω λ) → T = 6 * π / 5) := sorry

theorem range_of_f_in_interval (ω : ℝ) (λ : ℝ) 
  (hω : ω = 5 / 6) (hλ : f (π / 4) ω λ = 0) :
  set.range (λ x, f x ω λ) ∩ set.Icc 0 (3 * π / 5) =
  set.Icc (- 1 - sqrt 2) (2 - sqrt 2) := sorry

end period_of_f_range_of_f_in_interval_l245_245738


namespace train_stoppage_time_l245_245960

theorem train_stoppage_time 
  (speed_excl_stoppages : ℝ) (speed_incl_stoppages : ℝ) 
  (h1 : speed_excl_stoppages = 54) (h2 : speed_incl_stoppages = 40) : 
  (14 / 0.9 ≈ 15.56) :=
by
  have speed_reduction := speed_excl_stoppages - speed_incl_stoppages
  have time := speed_reduction / (speed_excl_stoppages / 60) 
  have h3 : speed_reduction = 14
  have h4 : time = 14 / (54 / 60)
  have h5 : (54 / 60) = 0.9
  exact sorry

end train_stoppage_time_l245_245960


namespace function_d_is_odd_l245_245522

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Given function
def f (x : ℝ) : ℝ := x^3

-- Proof statement
theorem function_d_is_odd : is_odd_function f := 
by sorry

end function_d_is_odd_l245_245522


namespace correct_options_l245_245278

-- Given condition
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- Option B assertion
def option_B (x y : ℝ) : Prop := (x^2 + y^2 - 4)/((x - 1)^2 + y^2 + 1) ≤ 2 + Real.sqrt 6

-- Option D assertion
def option_D (x y : ℝ) : Prop := x - Real.sqrt 2 * y + 2 = 0

-- Theorem to prove both options B and D are correct under the given condition
theorem correct_options {x y : ℝ} (h : curve x y) : option_B x y ∧ option_D x y := by
  sorry

end correct_options_l245_245278


namespace two_point_line_l245_245125

theorem two_point_line (k b : ℝ) (h_k : k ≠ 0) :
  (∀ (x y : ℝ), (y = k * x + b → (x, y) = (0, 0) ∨ (x, y) = (1, 1))) →
  (∀ (x y : ℝ), (y = k * x + b → (x, y) ≠ (2, 0))) :=
by
  sorry

end two_point_line_l245_245125


namespace remaining_oil_quantity_check_remaining_oil_quantity_l245_245777

def initial_oil_quantity : Real := 40
def outflow_rate : Real := 0.2

theorem remaining_oil_quantity (t : Real) : Real :=
  initial_oil_quantity - outflow_rate * t

theorem check_remaining_oil_quantity (t : Real) : remaining_oil_quantity t = 40 - 0.2 * t := 
by 
  sorry

end remaining_oil_quantity_check_remaining_oil_quantity_l245_245777


namespace myrtle_eggs_after_collection_l245_245862

def henA_eggs_per_day : ℕ := 3
def henB_eggs_per_day : ℕ := 4
def henC_eggs_per_day : ℕ := 2
def henD_eggs_per_day : ℕ := 5
def henE_eggs_per_day : ℕ := 3

def days_gone : ℕ := 12
def eggs_taken_by_neighbor : ℕ := 32

def eggs_dropped_day1 : ℕ := 3
def eggs_dropped_day2 : ℕ := 5
def eggs_dropped_day3 : ℕ := 2

theorem myrtle_eggs_after_collection :
  let total_eggs :=
    (henA_eggs_per_day * days_gone) +
    (henB_eggs_per_day * days_gone) +
    (henC_eggs_per_day * days_gone) +
    (henD_eggs_per_day * days_gone) +
    (henE_eggs_per_day * days_gone)
  let remaining_eggs_after_neighbor := total_eggs - eggs_taken_by_neighbor
  let total_dropped_eggs := eggs_dropped_day1 + eggs_dropped_day2 + eggs_dropped_day3
  let eggs_after_drops := remaining_eggs_after_neighbor - total_dropped_eggs
  eggs_after_drops = 162 := 
by 
  sorry

end myrtle_eggs_after_collection_l245_245862


namespace smallest_difference_exists_l245_245497

-- Define the custom addition method used by Vovochka
def vovochka_add (a b : Nat) : Nat := 
  let hundreds := (a / 100) + (b / 100)
  let tens := ((a % 100) / 10) + ((b % 100) / 10)
  let units := (a % 10) + (b % 10)
  hundreds * 1000 + tens * 100 + units

-- Define the standard addition
def std_add (a b : Nat) : Nat := a + b

-- Define the function to compute the difference
def difference (a b : Nat) : Nat :=
  abs (vovochka_add a b - std_add a b)

-- Define the claim
theorem smallest_difference_exists : ∃ a b : Nat, 
  a < 1000 ∧ b < 1000 ∧ a > 99 ∧ b > 99 ∧ 
  difference a b = 1800 := 
sorry

end smallest_difference_exists_l245_245497


namespace collinear_XYZ_l245_245820

open EuclideanGeometry

def collinear_points (X Y Z : Point) : Prop :=
  ∃ (l : Line), X ∈ l ∧ Y ∈ l ∧ Z ∈ l

theorem collinear_XYZ
  (d d' : Line) (Γ1 Γ2 : Circle) (X Y Z : Point)
  (h_parallel : are_parallel d d')
  (h_tangent_Γ1 : Γ1.Tangent X d)
  (h_between : between d Γ1 Γ2 d')
  (h_tangent_Γ2_X : Γ2.Tangent Z d')
  (h_tangent_Γ2_Γ1_Y : Γ2.Tangent Y Γ1) :
  collinear_points X Y Z :=
sorry

end collinear_XYZ_l245_245820


namespace comparison_l245_245710

noncomputable def f : ℝ → ℝ := sorry
-- f is an odd function on ℝ
axiom f_odd : ∀ x, f (-x) = -f x
-- f'' exists and f''(x) + f(x)/x > 0 for x ≠ 0
axiom f_double_prime_condition : ∀ x ≠ 0, (f '' x) + (f x)/x > 0

-- Define a, b, and c
noncomputable def a := (1/2) * f (1/2)
noncomputable def b := -2 * f (-2)
noncomputable def c := (Real.log (1/2)) * f (Real.log (1/2))

theorem comparison : b > c ∧ c > a := sorry

end comparison_l245_245710


namespace inscribed_quadrilateral_diagonal_difference_l245_245422

theorem inscribed_quadrilateral_diagonal_difference (a b c d e f : ℝ)
  (h_cyclic : is_cyclic_quadrilateral a b c d e f) :
  abs (e - f) ≤ abs (b - d) := sorry

end inscribed_quadrilateral_diagonal_difference_l245_245422


namespace length_AB_is_4_l245_245703

section HyperbolaProof

/-- Define the hyperbola -/
def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 16) - (y^2 / 8) = 1

/-- Define the line l given by x = 2√6 -/
def line_l (x : ℝ) : Prop :=
  x = 2 * Real.sqrt 6

/-- Define the condition for intersection points -/
def intersect_points (x y : ℝ) : Prop :=
  hyperbola x y ∧ line_l x

/-- Prove the length of the line segment AB is 4 -/
theorem length_AB_is_4 :
  ∀ y : ℝ, intersect_points (2 * Real.sqrt 6) y → |y| = 2 → length_AB = 4 :=
sorry

end HyperbolaProof

end length_AB_is_4_l245_245703


namespace line_AB_circle_relationship_l245_245256

noncomputable def circle (O : Point) (r : ℝ) : Set Point := {P | dist O P = r}

def pointA_position (O A : Point) (radius : ℝ) :=
  dist O A > radius

def pointB_position (O B : Point) (radius : ℝ) :=
  dist O B = radius

theorem line_AB_circle_relationship
  (O A B : Point)
  (radius : ℝ)
  (h_radius : radius = 2)
  (h_OA : dist O A = 3)
  (h_OB : dist O B = 2) :
  (line_through A B).intersect (circle O radius) ∨
  tangent (line_through A B) (circle O radius) :=
sorry

end line_AB_circle_relationship_l245_245256


namespace inequality_proof_l245_245988

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l245_245988


namespace union_area_of_square_and_circle_l245_245585

-- Define the basic entities
def side_length : ℝ := 8
def radius : ℝ := 8

-- Lean statement of the problem
theorem union_area_of_square_and_circle (s r : ℝ) (h_s : s = side_length) (h_r: r = radius) :
  let square_area := s^2,
      circle_area := π * r^2,
      overlapping_area := 1/4 * circle_area in
  square_area + circle_area - overlapping_area = 64 + 48 * π := by
  sorry

end union_area_of_square_and_circle_l245_245585


namespace face_opposite_green_is_purple_l245_245431

theorem face_opposite_green_is_purple
  (color : ℕ → char)
  (h_colors : color 0 = 'G' ∧ color 1 = 'R' ∧ color 2 = 'B' ∧ color 3 = 'Y' ∧ color 4 = 'P' ∧ color 5 = 'O') :
  cube_face_opposite (color 0) = color 4 :=
by
  sorry

end face_opposite_green_is_purple_l245_245431


namespace josh_path_count_l245_245381

theorem josh_path_count (n : ℕ) : 
  let count_ways (steps: ℕ) := 2^steps in
  count_ways (n-1) = 2^(n-1) :=
by
  sorry

end josh_path_count_l245_245381


namespace problem_statement_l245_245687

noncomputable def ellipse_equation (a b : ℝ) (ha : a > b) (hb : b > 0) : Prop :=
  ∃ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def check_eccentricity (a b : ℝ) (eccentricity : ℝ) : Prop :=
  eccentricity = (Real.sqrt (a^2 - b^2) / a)

noncomputable def point_on_ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def parallelogram_area_constant (a b : ℝ) : Prop :=
  ∀ (P M N : ℝ × ℝ), 
  let O := (0, 0) in
  point_on_ellipse a b P.1 P.2 → 
  point_on_ellipse a b M.1 M.2 → 
  point_on_ellipse a b N.1 N.2 → 
  (∃ k : ℝ, k ≠ 0 ∧ (M.2 - O.2) = k * (M.1 - O.1) ∧ (N.2 - P.2) = k * (N.1 - P.1)) →
  PSArea ((0, 0), P, M, N) = 2 * Real.sqrt 6

theorem problem_statement :
  ∀ a b : ℝ, a > b → b > 0 → 
  check_eccentricity a b (Real.sqrt 2 / 2) →
  point_on_ellipse a b b (a / b) →
  ellipse_equation a b →
  parallelogram_area_constant a b :=
by
  sorry

end problem_statement_l245_245687


namespace find_volume_of_sphere_l245_245162

noncomputable def volume_of_sphere (AB BC AA1 : ℝ) (hAB : AB = 2) (hBC : BC = 2) (hAA1 : AA1 = 2 * Real.sqrt 2) : ℝ :=
  let diagonal := Real.sqrt (AB^2 + BC^2 + AA1^2)
  let radius := diagonal / 2
  (4 * Real.pi * radius^3) / 3

theorem find_volume_of_sphere : volume_of_sphere 2 2 (2 * Real.sqrt 2) (by rfl) (by rfl) (by rfl) = (32 * Real.pi) / 3 :=
by
  sorry

end find_volume_of_sphere_l245_245162


namespace gran_age_indeterminate_l245_245740

theorem gran_age_indeterminate
(gran_age : ℤ) -- Let Gran's age be denoted by gran_age
(guess1 : ℤ := 75) -- The first grandchild guessed 75
(guess2 : ℤ := 78) -- The second grandchild guessed 78
(guess3 : ℤ := 81) -- The third grandchild guessed 81
-- One guess is mistaken by 1 year
(h1 : (abs (gran_age - guess1) = 1) ∨ (abs (gran_age - guess2) = 1) ∨ (abs (gran_age - guess3) = 1))
-- Another guess is mistaken by 2 years
(h2 : (abs (gran_age - guess1) = 2) ∨ (abs (gran_age - guess2) = 2) ∨ (abs (gran_age - guess3) = 2))
-- Another guess is mistaken by 4 years
(h3 : (abs (gran_age - guess1) = 4) ∨ (abs (gran_age - guess2) = 4) ∨ (abs (gran_age - guess3) = 4)) :
  False := sorry

end gran_age_indeterminate_l245_245740


namespace subspace_isometrically_isomorphic_to_X_of_finite_codimension_l245_245817

variables {U : Type*} [normed_space ℝ U]
variables {X : Type*} [finite_dimensional ℝ X] [normed_space ℝ X]
variables {V N : Type*} [submodule U V] [submodule U N]
variables (U X)

-- Main statement:
theorem subspace_isometrically_isomorphic_to_X_of_finite_codimension
  (hU : ∃ (W : submodule U), is_isometric X W)
  (hV : ∃ (N : submodule U), finite_dimensional ℝ N ∧ V + N = ⊤) :
  ∃ (W' : submodule V), is_isometric X W' := 
sorry

end subspace_isometrically_isomorphic_to_X_of_finite_codimension_l245_245817


namespace probability_of_at_least_one_completed_pass_l245_245787

theorem probability_of_at_least_one_completed_pass :
  let P_completed := (3 : ℝ) / 10,
      P_incomplete := 1 - P_completed,
      P_both_incomplete := P_incomplete * P_incomplete,
      P_at_least_one_completed := 1 - P_both_incomplete,
      percentage := P_at_least_one_completed * 100 in
  percentage = 51 := by
  let P_completed := (3 : ℝ) / 10,
      P_incomplete := 1 - P_completed,
      P_both_incomplete := P_incomplete * P_incomplete,
      P_at_least_one_completed := 1 - P_both_incomplete,
      percentage := P_at_least_one_completed * 100
  have h : percentage = 51 := by {
      -- Here you should perform the proof, but we skip it
      sorry
  }
  exact h

end probability_of_at_least_one_completed_pass_l245_245787


namespace octagon_area_l245_245940

-- Define the given condition: the circle has an area of 400π square units
def circle_area := 400 * Real.pi

-- Define the calculation of the radius from the circle's area
def radius (ca : ℝ) : ℝ := Real.sqrt (ca / Real.pi)

-- Define the formula to compute the area of an isosceles triangle with given base angle and radius
def octagon_triangle_area (r : ℝ) (angle : ℝ) : ℝ := (1/2) * r * r * Real.sin angle

-- Translation of the problem statement into a theorem in Lean 4
theorem octagon_area {ca : ℝ} (h : ca = circle_area) : 
  let r := radius ca in
  let tri_angle := Real.pi / 4 in
  let one_triangle_area := octagon_triangle_area r tri_angle in
  8 * one_triangle_area = 800 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l245_245940


namespace altitude_segment_length_l245_245493

theorem altitude_segment_length 
  {A B C D E : Type} 
  (BD DC AE y : ℝ) 
  (h1 : BD = 4) 
  (h2 : DC = 6) 
  (h3 : AE = 3) 
  (h4 : 3 / 4 = 9 / (y + 3)) : 
  y = 9 := 
by 
  sorry

end altitude_segment_length_l245_245493


namespace solve_geometry_problem_l245_245801

-- Polar coordinate system establishment
def polar_eq_circle (θ : ℝ) : ℝ := 3 * real.cos θ

-- Parametric equations of the circle
def param_eq_circle (θ : ℝ) : ℝ × ℝ :=
  (3/2 + 3/2 * real.cos θ, 3/2 * real.sin θ)

-- Line equation in Cartesian form
def line_eq (x y : ℝ) : Prop :=
  sqrt 3 * x - y + 2 * sqrt 3 = 0

-- Distance between point P and the line
def distance_to_line (θ : ℝ) : ℝ :=
  abs ((sqrt 3 * (3/2 + 3/2 * real.cos θ) - 3/2 * real.sin θ + 2 * sqrt 3) / sqrt (3^2 + (-1)^2))

-- Angle measure
def angle_ACP (θ : ℝ) : ℝ := if θ = π/3 ∨ θ = 4*π/3 then θ else 0

-- Main theorem
theorem solve_geometry_problem (θ : ℝ) (h_dist : distance_to_line θ = 7 * sqrt 3 / 4) :
  (param_eq_circle θ = (3/2 + 3/2 * real.cos θ, 3/2 * real.sin θ)) ∧ 
  (angle_ACP θ = π/3 ∨ angle_ACP θ = 2*π/3) :=
begin
  sorry,
end

end solve_geometry_problem_l245_245801


namespace smallest_integer_in_set_l245_245338

theorem smallest_integer_in_set (n : ℤ) (h : n+4 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) / 5)) : n ≥ 0 :=
by sorry

end smallest_integer_in_set_l245_245338


namespace min_value_of_trig_function_l245_245948

theorem min_value_of_trig_function :
  ∀ x : ℝ, 0 < x ∧ x < (π / 2) → (∃ c : ℝ, c = 4 ∧ 
  (c = min (λ x, (1 + cos (2 * x) + 8 * sin (2 * x)) / sin (2 * x)) {x | 0 < x ∧ x < (π / 2)})) :=
by
   sorry

end min_value_of_trig_function_l245_245948


namespace total_oranges_l245_245433

theorem total_oranges (initial_oranges : ℕ) (additional_oranges : ℕ) (weeks : ℕ) (multiplier : ℕ) :
  initial_oranges = 10 → additional_oranges = 5 → weeks = 2 → multiplier = 2 → 
  let first_week := initial_oranges + additional_oranges in
  let next_weeks := weeks * (multiplier * first_week) in
  first_week + next_weeks = 75 :=
begin
  intros h1 h2 h3 h4,
  let first_week := initial_oranges + additional_oranges,
  let next_weeks := weeks * (multiplier * first_week),
  have h_first : first_week = 15, 
  { rw [h1, h2], exact add_comm 10 5 },
  have h_next : next_weeks = 60, 
  { rw [h_first, h3, h4], exact mul_comm 2 30 },
  rw [h_first, h_next],
  exact add_comm 15 60,
end

end total_oranges_l245_245433


namespace chess_tournament_participants_l245_245316

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 378) : n = 28 :=
sorry

end chess_tournament_participants_l245_245316


namespace purely_imaginary_implication_l245_245759

theorem purely_imaginary_implication (x : ℝ) :
  (∃ (z : ℂ), z = complex.mk 0 (x - 1) ∧ x^2 - 3 * x + 2 = 0 ∧ x - 1 ≠ 0) → x = 2 :=
by
  intro h,
  cases h with z h_z,
  cases h_z with h1 h2,
  cases h2 with h3 h4,
  sorry

end purely_imaginary_implication_l245_245759


namespace moral_of_saying_l245_245203

/-!
  Comrade Mao Zedong said: "If you want to know the taste of a pear, you must change the pear and taste it yourself." 
  Prove that this emphasizes "Practice is the source of knowledge" (option C) over the other options.
-/

def question := "What is the moral of Comrade Mao Zedong's famous saying about tasting a pear?"

def options := ["Knowledge is the driving force behind the development of practice", 
                "Knowledge guides practice", 
                "Practice is the source of knowledge", 
                "Practice has social and historical characteristics"]

def correct_answer := "Practice is the source of knowledge"

theorem moral_of_saying : (question, options[2]) ∈ [("What is the moral of Comrade Mao Zedong's famous saying about tasting a pear?", 
                                                      "Practice is the source of knowledge")] := by 
  sorry

end moral_of_saying_l245_245203


namespace root_in_interval_l245_245476

def f (x : ℝ) : ℝ := x^2 - (2 / x)

theorem root_in_interval :
  f 1 < 0 → f (5/4) < 0 → f (3/2) > 0 → ∃ x, (5/4 < x ∧ x < 3/2 ∧ f x = 0) :=
by
  intros h1 h2 h3
  -- Use the intermediate value theorem to show that there is a root in the interval (5/4, 3/2)
  sorry

end root_in_interval_l245_245476


namespace symmedian_contains_OI_l245_245446

/-- Given points O, I, and K with specified properties, prove that line 
OI contains the symmedian of triangle AIK. -/
theorem symmedian_contains_OI
  (A B C K I O : Point) 
  (circumcircle : Circle) 
  (incircle : Circle) 
  (arc_midpoint : Point)
  (h_O_circum : IsCircumcenter O ⟨A, B, C⟩)
  (h_I_incenter : IsIncenter I ⟨A, B, C⟩)
  (h_K_arc : K = arc_midpoint ⟨B, C⟩ circumcircle ∧ OnAngleBisector K ⟨A, B, C⟩)
  (h_K_perp : OnPerpendicularFromIncenter K ⟨B, C⟩ I)
  : ContainsSymmedian (Line O I) ⟨A, I, K⟩ := 
sorry

end symmedian_contains_OI_l245_245446


namespace inequality_solution_set_l245_245150

def problem_statement (f : ℝ → ℝ) :=
  (∀ x, f (-x) + f x = 2 * x^2) ∧
  (continuous f) ∧
  (∀ x, x ≤ 0 → f' x < 2 * x)

theorem inequality_solution_set (f : ℝ → ℝ) (h : problem_statement f) :
  { x | f x + 1 ≥ f (1 - x) + 2 * x } = Iic (1 / 2) :=
by
  sorry

end inequality_solution_set_l245_245150


namespace Helen_needs_41_gallons_l245_245744

-- Definitions based on the conditions
def height : ℝ := 24
def diameter : ℝ := 8
def radius : ℝ := diameter / 2
def single_pole_area : ℝ := 2 * Real.pi * radius * height
def number_of_poles : ℕ := 20
def total_area : ℝ := number_of_poles * single_pole_area
def paint_coverage : ℝ := 300  -- square feet per gallon
def gallons_needed (area : ℝ) : ℝ := area / paint_coverage

-- The theorem to be proven
theorem Helen_needs_41_gallons : 
  ⌈gallons_needed total_area⌉ = 41 := by
  sorry

end Helen_needs_41_gallons_l245_245744


namespace smallest_positive_difference_l245_245501

-- Define the method used by Vovochka to sum numbers.
def sum_without_carrying (a b : ℕ) : ℕ :=
  let ha := a / 100 
  let ta := (a % 100) / 10 
  let ua := a % 10
  let hb := b / 100 
  let tb := (b % 100) / 10 
  let ub := b % 10
  (ha + hb) * 1000 + (ta + tb) * 100 + (ua + ub)

-- Define the correct method to sum numbers.
def correct_sum (a b : ℕ) : ℕ :=
  a + b

-- The smallest positive difference between the sum without carrying and the correct sum.
theorem smallest_positive_difference (a b : ℕ) (h : 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000) :
  (sum_without_carrying a b - correct_sum a b > 0) →
  (∀ c d, 100 ≤ c ∧ c < 1000 ∧ 100 ≤ d ∧ d < 1000 →
    (sum_without_carrying c d - correct_sum c d > 0 → sum_without_carrying c d - correct_sum c d ≥ 1800 )) :=
begin
  sorry
end

end smallest_positive_difference_l245_245501


namespace survey_respondents_l245_245132

theorem survey_respondents (X Y : ℕ) (hX : X = 150) (hRatio : X / Y = 5) : X + Y = 180 :=
by
  sorry

end survey_respondents_l245_245132


namespace red_envelope_grabs_l245_245772

/-- Given five people: A, B, C, D, and E, and four red envelopes with values
    {2, 2, 3, 4} where each person can grab at most one red envelope,
    prove that the number of ways in which both A and B can grab a red envelope is 36. --/
theorem red_envelope_grabs : 
  let people := ['A', 'B', 'C', 'D', 'E']
  let envelopes := [2, 2, 3, 4]
  let possible_grabs := {x : list (char × nat) | ∀(p : char × nat), p ∈ x → p.1 ∈ people ∧ p.2 ∈ envelopes ∧ (list.count x (p.1)) ≤ 1}
  let valid_grabs := filter (λ x, ('A', 2) ∈ x ∨ ('A', 3) ∈ x ∨ ('A', 4) ∈ x ∧ ('B', 2) ∈ x ∨ ('B', 3) ∈ x ∨ ('B', 4) ∈ x ∧ x.card = 4) possible_grabs
  valid_grabs.card = 36 :=
begin
  sorry
end

end red_envelope_grabs_l245_245772


namespace harmonic_binom_identity_l245_245042

theorem harmonic_binom_identity (n : Nat) (hn : n ≥ 1) :
    ∑ k in Finset.range (n + 1), (-1) ^ (k - 1) * Nat.choose n k * (∑ i in Finset.range (k + 1), (1 : ℚ) / (i + 1)) = 1 / n := 
sorry

end harmonic_binom_identity_l245_245042


namespace complementary_event_is_even_l245_245088

-- Defining the universal set for a fair cubic die.
def universal_set := {1, 2, 3, 4, 5, 6}

-- Defining event A.
def event_A := {n ∈ universal_set | n % 2 = 1}

-- Defining the complementary event of event A.
def complementary_event_A := {n ∈ universal_set | n % 2 = 0}

-- Statement of the problem
theorem complementary_event_is_even : complementary_event_A = {n ∈ universal_set | n % 2 = 0} := by
  sorry

end complementary_event_is_even_l245_245088


namespace compute_x_l245_245202

noncomputable def geometric_series_sum (a r : ℝ) : ℝ := a / (1 - r)

theorem compute_x :
  (geometric_series_sum 1 (1/2)) * (geometric_series_sum 1 (-(1/2))) = geometric_series_sum 1 (1/x) :=
by
  have sum1 : geometric_series_sum 1 (1/2) = 2 := by sorry
  have sum2 : geometric_series_sum 1 (-(1/2)) = 2/3 := by sorry
  have rhs_sum : geometric_series_sum 1 (1/x) = 4/3 := by sorry
  show x = 4 := by sorry

end compute_x_l245_245202


namespace discounted_price_l245_245079

theorem discounted_price (P : ℝ) (original_price : ℝ) (discount_rate : ℝ)
  (h1 : original_price = 975)
  (h2 : discount_rate = 0.20)
  (h3 : P = original_price - discount_rate * original_price) : 
  P = 780 := 
by
  sorry

end discounted_price_l245_245079


namespace quadrilateral_BF_eq_4_l245_245799

variables {A B C D E F : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables [has_dist A B] [has_dist B C] [has_dist C D] [has_dist D E] [has_dist E F] [has_dist F A]

/-- Quadrilateral ABCD has right angles at A and C, points E and F are on AC,
    DE and BF are perpendicular to AC, AE = 4, DE = 4, CE = 6.
    Prove that BF = 4. -/
theorem quadrilateral_BF_eq_4 
  (h1 : dist A C = 10) 
  (h2 : dist A E = 4) 
  (h3 : dist C E = 6) 
  (h4 : dist D E = 4) 
  (h5 : ∠A = π/2) 
  (h6 : ∠C = π/2) 
  (h7 : ∠B F A = π / 2) 
  (h8 : ∠D E A = π / 2) 
  : dist B F = 4 :=
sorry

end quadrilateral_BF_eq_4_l245_245799


namespace josh_paths_to_center_square_l245_245373

-- Definition of the problem's conditions based on given movements and grid size
def num_paths (n : Nat) : Nat :=
  2^(n-1)

-- Main statement
theorem josh_paths_to_center_square (n : Nat) : ∃ p : Nat, p = num_paths n :=
by
  exists num_paths n
  sorry

end josh_paths_to_center_square_l245_245373


namespace remainder_div_l245_245528

theorem remainder_div (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) : N % 15 = 10 := by
  sorry

end remainder_div_l245_245528


namespace sufficient_but_not_necessary_condition_l245_245307

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, (x = 1 → ((x - a) * (x - (a + 2)) ≤ 0)) ∧ 
       (∃ x, x ≠ 1 ∧ (x - a) * (x - (a + 2)) ≤ 0)) ↔ (a ∈ set.Icc (-1 : ℝ) (1 : ℝ)) := by
  sorry

end sufficient_but_not_necessary_condition_l245_245307


namespace sum_of_real_solutions_l245_245238

theorem sum_of_real_solutions :
  (∑ x in { x : ℝ | |x - 1| = 3 * |x + 1| }, x) = -5 / 2 :=
by
  sorry

end sum_of_real_solutions_l245_245238


namespace option_B_option_D_l245_245273

noncomputable def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- The maximum value of (y + 1) / (x + 1) is 2 + sqrt(6)
theorem option_B (x y : ℝ) (h : curve_C x y) :
  ∃ k, (y + 1) / (x + 1) = k ∧ k = 2 + Real.sqrt 6 :=
sorry

-- A tangent line through the point (0, √2) on curve C has the equation x - √2 * y + 2 = 0
theorem option_D (h : curve_C 0 (Real.sqrt 2)) :
  ∃ a b c, a * 0 + b * Real.sqrt 2 + c = 0 ∧ c = 2 ∧ a = 1 ∧ b = - Real.sqrt 2 :=
sorry

end option_B_option_D_l245_245273


namespace intersect_octahedron_area_l245_245164

theorem intersect_octahedron_area :
  ∀ (a b c : ℕ),
  let s := 2 in
  let A := (3 * Real.sqrt 3) / 2 * (Real.sqrt 3 / 2)^2 in
  a = 9 ∧ b = 3 ∧ c = 8 ∧ gcd a c = 1 ∧ ¬ (∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ b) →
  (a * Real.sqrt b) / c = A :=
by
  sorry

end intersect_octahedron_area_l245_245164


namespace josh_paths_to_top_center_l245_245370

/-- Define the grid --/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Josh's initial position --/
def start_pos : (ℕ × ℕ) := (0, 0)

/-- Define a function representing Josh's movement possibilities --/
def move_right (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1, pos.2 + 1)

def move_left_up (pos : ℕ × ℕ) : ℕ × ℕ :=
  (pos.1 + 1, pos.2 - 1)

/-- Define the goal position --/
def goal_pos (n : ℕ) : (ℕ × ℕ) :=
  (n - 1, 1)

/-- Theorem stating the required proof --/
theorem josh_paths_to_top_center {n : ℕ} (h : n > 0) : 
  let g := Grid.mk n 3 in
  ∃ (paths : ℕ), paths = 2^(n - 1) := 
  sorry

end josh_paths_to_top_center_l245_245370


namespace shampooing_time_l245_245001

theorem shampooing_time (h1 : ∀ t : ℝ, t * (1/3) + t * (1/6) = 1) : ℝ :=
begin
  have t := 2,
  have H : t * (1/2) = 1, {
    simp,
    ring,
  },
  exact t,
end

end shampooing_time_l245_245001


namespace total_profit_l245_245568

def share (P : ℝ) : ℝ :=
  0.10 * P + (4 / 9) * (0.90 * P)

theorem total_profit (P : ℝ) (h : share P = 4800) : P = 9600 := by
  sorry

end total_profit_l245_245568


namespace photo_area_with_frame_l245_245530

-- Define the areas and dimensions given in the conditions
def paper_length : ℕ := 12
def paper_width : ℕ := 8
def frame_width : ℕ := 2

-- Define the dimensions of the photo including the frame
def photo_length_with_frame : ℕ := paper_length + 2 * frame_width
def photo_width_with_frame : ℕ := paper_width + 2 * frame_width

-- The theorem statement proving the area of the wall photo including the frame
theorem photo_area_with_frame :
  (photo_length_with_frame * photo_width_with_frame) = 192 := by
  sorry

end photo_area_with_frame_l245_245530


namespace extremum_condition_l245_245823

noncomputable def y (a x : ℝ) : ℝ := Real.exp (a * x) + 3 * x

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, (y a x = 0) ∧ ∀ x' > x, y a x' < y a x) → a < -3 :=
by
  sorry

end extremum_condition_l245_245823


namespace cone_lateral_surface_area_correct_l245_245705

def cone_base_radius : ℝ := 1
def cone_height : ℝ := 2 * Real.sqrt 2
def lateral_surface_area (r h : ℝ) : ℝ := (Real.pi * r * Real.sqrt (r^2 + h^2))

theorem cone_lateral_surface_area_correct :
  lateral_surface_area cone_base_radius cone_height = 3 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_correct_l245_245705


namespace fraction_spent_on_clothes_l245_245574

-- Conditions
def salary : ℝ := 140000
def spent_on_food : ℝ := (1/5) * salary
def spent_on_rent : ℝ := (1/10) * salary
def remaining_amount : ℝ := 14000

-- Proof Statement
theorem fraction_spent_on_clothes : (salary - spent_on_food - spent_on_rent - remaining_amount) / salary = 0.6 :=
by
  -- We write the proof here.
  sorry

end fraction_spent_on_clothes_l245_245574


namespace tennis_tournament_64_matches_tennis_tournament_2011_matches_l245_245140

theorem tennis_tournament_64_matches (n : ℕ) (h : n = 64) :
  ∃ m : ℕ, m = 63 :=
by
  have h1 : n = 64 := h
  use 63
  sorry

theorem tennis_tournament_2011_matches (n : ℕ) (h : n = 2011) :
  ∃ m : ℕ, m = 2010 :=
by
  have h1 : n = 2011 := h
  use 2010
  sorry

end tennis_tournament_64_matches_tennis_tournament_2011_matches_l245_245140


namespace intersection_single_point_l245_245452

-- Definitions of the points and the setup
variables {A B C A₁ B₁ C₁ A₂ B₂ C₂ L: Type*}
variable [metric_space L]

-- Definitions of the triangle and the circumcircle
variables (triangle :  set (L × L)) -- ∀ A B C collinear
variables (circumcircle : set L)

-- Conditions stated as hypotheses
hypothesis (median_intersect : ∀ (A B C : L), 
  ∃ (A₁ B₁ C₁ : L), 
  (A₁ ∈ circumcircle) ∧ (B₁ ∈ circumcircle) ∧ (C₁ ∈ circumcircle))

hypothesis (parallel_intersect : ∀ (A B C : L), 
  ∃ (A₂ B₂ C₂ : L), 
  (A₂ ∈ circumcircle) ∧ (B₂ ∈ circumcircle) ∧ (C₂ ∈ circumcircle))

-- The theorem that needs proving
theorem intersection_single_point 
  (h1 : ∀ A B C, median_intersect A B C)
  (h2 : ∀ A B C, parallel_intersect A B C) :
  ∃ P: L, ∀ (A B C A₁ B₁ C₁ A₂ B₂ C₂ : L), P ∈ (line A₁ A₂) ∧ P ∈ (line B₁ B₂) ∧ P ∈ (line C₁ C₂) :=
sorry -- The proof of this theorem is omitted

end intersection_single_point_l245_245452


namespace pipe_A_fill_time_alone_l245_245569

theorem pipe_A_fill_time_alone
  (B_rate : ℝ := 1/15)
  (together_rate : ℝ := 1/10)
  (A_rate : ℝ := together_rate - B_rate) :
  (1 / A_rate) = 30 :=
by
  have hB : B_rate = 1 / 15 := rfl
  have htogether : together_rate = 1 / 10 := rfl
  have hA : A_rate = (1 / 10) - (1 / 15) := by
    calc
      A_rate = together_rate - B_rate   : by rfl
      ...    = (1 / 10) - (1 / 15)       : by rw [htogether, hB]
  have hTime : (1 / A_rate) = 30 := by
    calc
      1 / A_rate = 1 / ((1 / 10) - (1 / 15)) : by rw hA
      ...        = 30                         : by norm_num
  exact hTime

end pipe_A_fill_time_alone_l245_245569


namespace unique_triangle_determination_l245_245525

theorem unique_triangle_determination : 
  ∀ (isosceles_base_angle v_side : ℝ) 
    (right_leg right_angle : ℝ)
    (equilateral_radius equilateral_vertex_angle : ℝ)
    (isosceles_arm isosceles_altitude : ℝ)
    (scalene_side1 scalene_side2 scalene_opposite_angle : ℝ),
  ¬unique (isosceles_triangle isosceles_base_angle v_side) ∧
  unique (right_triangle right_leg right_angle) ∧
  unique (equilateral_triangle equilateral_radius equilateral_vertex_angle) ∧
  unique (isosceles_triangle_two_params isosceles_arm isosceles_altitude) ∧
  unique (scalene_triangle scalene_side1 scalene_side2 scalene_opposite_angle) := by
  sorry

-- Definitions to aid in the theorem
def isosceles_triangle (base_angle side_opposite_vertex_angle : ℝ) : Prop :=
  ∃ (a b : ℝ), a = b ∧ (angle_of a b = base_angle) ∧ (side_of a b = side_opposite_vertex_angle)

def right_triangle (leg opposite_angle : ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 + b^2 = leg^2 ∧ opposite_angle = π / 2

def equilateral_triangle (radius vertex_angle : ℝ) : Prop :=
  vertex_angle = π / 3 ∧ ∃ (r : ℝ), r = radius

def isosceles_triangle_two_params (arm altitude : ℝ) : Prop :=
  ∃ (a h : ℝ), a^2 = arm^2 + h^2 ∧ altitude = h

def scalene_triangle (side1 side2 opposite_angle : ℝ) : Prop :=
  ∃ (a b : ℝ), angle_of a b = opposite_angle ∧ side_of a b = side1 ∧ side_of b a = side2

def unique (t : Prop) : Prop :=
  ∃! x, t

end unique_triangle_determination_l245_245525


namespace proof_AC_length_l245_245336

noncomputable def AC_length 
  (A B C D : ℝ) 
  (AB : ℝ) (DC : ℝ) (AD : ℝ)
  (BC_perpendicular_to_AD : Prop) 
  (hAB : AB = 15) (hDC : DC = 25)
  (hAD : AD = 7) 
  : ℝ := 22.3

theorem proof_AC_length 
  (A B C D : ℝ) 
  (AB : ℝ) (DC : ℝ) (AD : ℝ)
  (BC_perpendicular_to_AD : Prop) 
  (hAB : AB = 15) (hDC : DC = 25)
  (hAD : AD = 7) 
  : AC_length A B C D AB DC AD BC_perpendicular_to_AD hAB hDC hAD = 22.3 :=
begin
  sorry
end

end proof_AC_length_l245_245336


namespace perfect_square_term_l245_245477

def seq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 4 * seq (n - 1) - seq (n - 2)

theorem perfect_square_term : ∀ n, (∃ k, seq n = k * k) ↔ n = 0 := by
  sorry

end perfect_square_term_l245_245477


namespace modulus_z_l245_245288

-- Define the complex number z
def z : ℂ := (sqrt 5 * complex.I) / (1 + 2 * complex.I)

-- State the problem: Prove |z| = 1 given z = (sqrt 5 * complex.I) / (1 + 2 * complex.I)
theorem modulus_z (z : ℂ) (hz : z = (sqrt 5 * complex.I) / (1 + 2 * complex.I)) : complex.abs z = 1 :=
sorry

end modulus_z_l245_245288


namespace line_intersects_circle_but_not_center_l245_245077

theorem line_intersects_circle_but_not_center :
  let A := 3
  let B := -4
  let C := -9
  let x1 := 0
  let y1 := 0
  let r := 2
  let d := |C| / (real.sqrt (A^2 + B^2))
  A * x1 + B * y1 + C = 0 → d < r := by
    -- Definitions of conditions
    let A := 3
    let B := -4
    let C := -9
    let x1 := 0
    let y1 := 0
    let r := 2
    let d := |C| / (real.sqrt (A^2 + B^2))

    -- Question to prove
    have h1 : A * x1 + B * y1 + C = 0 := by sorry
    have h2 : d < r := by sorry
    h2

end line_intersects_circle_but_not_center_l245_245077


namespace f_is_even_f_monotonic_increase_range_of_a_for_solutions_l245_245842

-- Define the function f(x) = x^2 - 2a|x|
def f (a x : ℝ) : ℝ := x^2 - 2 * a * |x|

-- Given a > 0
variable (a : ℝ) (ha : a > 0)

-- 1. Prove that f(x) is an even function.
theorem f_is_even : ∀ x : ℝ, f a x = f a (-x) := sorry

-- 2. Prove the interval of monotonic increase for f(x) when x > 0 is [a, +∞).
theorem f_monotonic_increase (x : ℝ) (hx : x > 0) : a ≤ x → ∃ c : ℝ, x ≤ c := sorry

-- 3. Prove the range of values for a for which the equation f(x) = -1 has solutions is a ≥ 1.
theorem range_of_a_for_solutions : (∃ x : ℝ, f a x = -1) ↔ 1 ≤ a := sorry

end f_is_even_f_monotonic_increase_range_of_a_for_solutions_l245_245842


namespace cos_angle_F1PF2_slope_line_OM_l245_245266

-- Part 1: Prove that the cosine value of angle F₁PF₂ is 7/9
theorem cos_angle_F1PF2 (x y : ℝ) (h : (x^2 / 8) + (y^2 / 9) = 1)
  (dPF1 : dist (x, y) (sqrt(1), 0) = 3) (dPF2 : dist (x, y) (-sqrt(1), 0) = 3) :
  cos (angle ((sqrt(1), 0)) ((x, y)) ((-sqrt(1), 0))) = 7 / 9 := 
sorry

-- Part 2: Prove that the slope of the line OM is -9/8
theorem slope_line_OM (x1 y1 x2 y2 : ℝ)
  (h1 : (x1^2 / 8) + (y1^2 / 9) = 1) (h2 : (x2^2 / 8) + (y2^2 / 9) = 1)
  (line_intersect : y1 = x1 + 1 ∧ y2 = x2 + 1) :
  ((x1 + x2) / 2, (y1 + y2) / 2).snd / ((x1 + x2) / 2) = -9 / 8 :=
sorry

end cos_angle_F1PF2_slope_line_OM_l245_245266


namespace right_triangle_altitudes_sum_21_l245_245215

theorem right_triangle_altitudes_sum_21 
  (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 15) (h_a : a = 9) (h_b : b = 12) : a + b = 21 :=
by {
  have h_leg1 : a = 9 := h_a,
  have h_leg2 : b = 12 := h_b,
  rw [h_leg1, h_leg2],
  sorry
}

end right_triangle_altitudes_sum_21_l245_245215


namespace annual_interest_rate_is_correct_l245_245809

theorem annual_interest_rate_is_correct :
  ∃ r : ℝ, r = 0.0583 ∧
  (200 * (1 + r)^2 = 224) :=
by
  sorry

end annual_interest_rate_is_correct_l245_245809


namespace coat_price_calculation_l245_245552

noncomputable def effective_price (initial_price : ℝ) 
  (reduction1 reduction2 reduction3 : ℝ) 
  (tax1 tax2 tax3 : ℝ) : ℝ :=
  let price_after_first_month := initial_price * (1 - reduction1 / 100) * (1 + tax1 / 100)
  let price_after_second_month := price_after_first_month * (1 - reduction2 / 100) * (1 + tax2 / 100)
  let price_after_third_month := price_after_second_month * (1 - reduction3 / 100) * (1 + tax3 / 100)
  price_after_third_month

noncomputable def total_percent_reduction (initial_price final_price : ℝ) : ℝ :=
  (initial_price - final_price) / initial_price * 100

theorem coat_price_calculation :
  let original_price := 500
  let price_final := effective_price original_price 10 15 20 5 8 6
  let reduction_percentage := total_percent_reduction original_price price_final
  price_final = 367.824 ∧ reduction_percentage = 26.44 :=
by
  sorry

end coat_price_calculation_l245_245552


namespace integer_root_of_polynomial_l245_245457

theorem integer_root_of_polynomial (d e : ℚ) (h1 : is_root (λ x : ℚ, x^3 + d * x + e) (3 - real.sqrt 5))
  (h2 : ∃ r : ℚ, (is_root (λ x, x^3 + d * x + e) r) ∧ r ≠ (3 - real.sqrt 5) ∧ r ≠ (3 + real.sqrt 5)) :
  ∃ r : ℤ, is_root (λ x : ℚ, x^3 + d * x + e) r ∧ r = -6 :=
by 
  sorry

end integer_root_of_polynomial_l245_245457


namespace seq_equiv_a_n_correctness_l245_245802

-- Define the sequence a_n
def seq_a : ℕ → ℕ
| 0       := 0
| (n + 1) := if n = 0 then 2 else 2^n

-- Define the sequence S_n as the sum of the first n terms of seq_a
def sum_S : ℕ → ℕ
| 0       := 0
| (n + 1) := seq_a (n + 1) + sum_S n

-- Define the common ratio for the geometric sequence S_n
def common_ratio := 2

-- Define the geometric sequence S_n with the first term S_1 = 2 and common ratio 2
def geom_S : ℕ → ℕ
| 0       := 0
| (n + 1) := 2 * common_ratio^n

-- Lean 4 statement to prove the equivalence
theorem seq_equiv (n: ℕ) : 
  sum_S n = geom_S n :=
by
  induction n with k ih,
  { simp [sum_S, geom_S, common_ratio], },
  { simp [sum_S, geom_S, ih], sorry }

theorem a_n_correctness (n : ℕ) :
  seq_a n = 
    if n = 1 then 2 else 2^(n - 1) :=
by
  induction n with k ih,
  { simp [seq_a], },
  { simp [seq_a] at *, 
    cases k,
    { simp [seq_a], },
    { simp [seq_a, ih], sorry, } }

end seq_equiv_a_n_correctness_l245_245802


namespace age_problem_l245_245156

theorem age_problem (F : ℝ) (M : ℝ) (Y : ℝ)
  (hF : F = 40.00000000000001)
  (hM : M = (2/5) * F)
  (hY : M + Y = (1/2) * (F + Y)) :
  Y = 8.000000000000002 :=
sorry

end age_problem_l245_245156


namespace arithmetic_seq_a7_l245_245797

theorem arithmetic_seq_a7 (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 4 + a 5 = 12) : a 7 = 10 :=
by
  sorry

end arithmetic_seq_a7_l245_245797


namespace exiting_arrangements_l245_245486

theorem exiting_arrangements :
  let number_of_chess_pieces := 9
  let number_of_exits := 6
  let possible_arrangements := (14.factorial) / (5.factorial)
  possible_arrangements = 
  ∑ n in (finset.range(9)), n + number_of_exits := 
sorry

end exiting_arrangements_l245_245486


namespace binomial_third_term_coeff_l245_245067

theorem binomial_third_term_coeff (a b : ℕ) :
  (∃ k : ℕ, (3*b + 2*a)^6 = (∑ i in finset.range 7, binom 6 i * (3 * b)^(6-i) * (2 * a)^i)) →
  (∃ k : ℕ, k = 2) → 
  (∃ c b_coeff : ℕ, c = 4860 ∧ b_coeff = 15 ∧ 
    binom 6 2 * (3*b)^4 * (2*a)^2 = c ∧ binom 6 2 = b_coeff) :=
by
  sorry

end binomial_third_term_coeff_l245_245067


namespace locus_of_midpoint_of_chord_l245_245701

theorem locus_of_midpoint_of_chord 
  (A B C : ℝ) (h_arith_seq : A - 2 * B + C = 0) 
  (h_passing_through : ∀ t : ℝ,  t*A + -2*B + C = 0) :
  ∀ (x y : ℝ), 
    (Ax + By + C = 0) → 
    (h_on_parabola : y = -2 * x ^ 2) 
    → y + 1 = -(2 * x - 1) ^ 2 :=
sorry

end locus_of_midpoint_of_chord_l245_245701


namespace number_of_two_digit_numbers_with_sum_143_l245_245635

theorem number_of_two_digit_numbers_with_sum_143 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (let a := n / 10, b := n % 10 in n + (10 * b + a) = 143)}.card = 6 :=
sorry

end number_of_two_digit_numbers_with_sum_143_l245_245635


namespace part_i_part_ii_l245_245726

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - 2 * a) + abs (x - a)

-- Part I: Prove solution to the inequality.
theorem part_i (x : ℝ) : f x 1 > 3 ↔ x ∈ {x | x < 0} ∪ {x | x > 3} :=
sorry

-- Part II: Prove the inequality for general a and b with condition for equality.
theorem part_ii (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  f b a ≥ f a a ∧ ((2 * a - b = 0 ∨ b - a = 0) ∨ (2 * a - b > 0 ∧ b - a > 0) ∨ (2 * a - b < 0 ∧ b - a < 0)) ↔ f b a = f a a :=
sorry

end part_i_part_ii_l245_245726


namespace product_value_l245_245118

theorem product_value :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 = 81 :=
  sorry

end product_value_l245_245118


namespace volume_difference_spheres_l245_245922

theorem volume_difference_spheres {π : Real} :
  let r_small := 4
  let r_large := 7
  let V_small := (4 / 3) * π * r_small^3
  let V_large := (4 / 3) * π * r_large^3
  V_large - V_small = 372 * π :=
by
  let r_small := 4
  let r_large := 7
  let V_small := (4 / 3 : Real) * π * r_small^3
  let V_large := (4 / 3 : Real) * π * r_large^3
  calc
    V_large - V_small = (4 / 3) * π * (r_large^3 - r_small^3) : by ring
    ... = 372 * π : by sorry

end volume_difference_spheres_l245_245922


namespace vertical_asymptotes_count_l245_245306

theorem vertical_asymptotes_count (x : ℝ) :
  let y := (x-2)/(x^2 + 8*x - 9)
  ∃ (n : ℕ), n = 2 ∧ (∀ ε : ℝ, ε > 0 → ( ( ε = 1 → (x = 1 ∨ x = -9)) 
    ∧ (ε ≠ 1 → (x^2 + 8*x - 9 ≠ 0))))
  sorry

end vertical_asymptotes_count_l245_245306


namespace least_x_l245_245961

noncomputable def is_odd_prime (n : ℕ) : Prop :=
  n > 1 ∧ Prime n ∧ n % 2 = 1

theorem least_x (x p : ℕ) (hp : Prime p) (hx : x > 0) (hodd_prime : is_odd_prime (x / (12 * p))) : x = 72 := 
  sorry

end least_x_l245_245961


namespace inequality_hold_l245_245968

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end inequality_hold_l245_245968


namespace number_of_sheep_l245_245076

def ratio_sheep_horses (S H : ℕ) : Prop := S / H = 3 / 7
def horse_food_per_day := 230 -- ounces
def total_food_per_day := 12880 -- ounces

theorem number_of_sheep (S H : ℕ) 
  (h1 : ratio_sheep_horses S H) 
  (h2 : H * horse_food_per_day = total_food_per_day) 
  : S = 24 :=
sorry

end number_of_sheep_l245_245076


namespace solve_quadratic_equation_l245_245653

noncomputable def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

theorem solve_quadratic_equation :
  let x1 := -2
      x2 := 11 in
  quadratic_eq 1 (-9) (-22) x1 ∧ quadratic_eq 1 (-9) (-22) x2 ∧ x1 < x2 :=
by
  sorry

end solve_quadratic_equation_l245_245653


namespace calc_ab_l245_245587

noncomputable theory

-- Define the side lengths
def side_length_smaller : ℝ := 4
def side_length_larger : ℝ := 3 * real.sqrt 2

-- Define the given relations
axiom a_plus_b : ∀ (a b : ℝ), a + b = 3 * real.sqrt 2
axiom right_triangle_condition : ∀ (a b : ℝ), real.sqrt (a ^ 2 + b ^ 2) = 2 * real.sqrt 6

-- Define the proposition to be proved
theorem calc_ab (a b : ℝ) : a * b = -3 :=
by
  -- The relations will be provided as assumptions
  have h1 : a + b = 3 * real.sqrt 2 := a_plus_b a b,
  have h2 : a^2 + b^2 = (2 * real.sqrt 6) ^ 2 := by rw [right_triangle_condition a b]; sorry,
  -- Needs to complete the proof (this line should be removed when actual proof is added)
  sorry

end calc_ab_l245_245587


namespace olivia_not_sold_bars_l245_245220

theorem olivia_not_sold_bars (cost_per_bar : ℕ) (total_bars : ℕ) (total_money_made : ℕ) :
  cost_per_bar = 3 →
  total_bars = 7 →
  total_money_made = 9 →
  total_bars - (total_money_made / cost_per_bar) = 4 :=
by
  intros h1 h2 h3
  sorry

end olivia_not_sold_bars_l245_245220


namespace soccer_tournament_probability_l245_245880

noncomputable def prob_teamA_more_points : ℚ :=
  (163 : ℚ) / 256

theorem soccer_tournament_probability :
  m + n = 419 ∧ prob_teamA_more_points = 163 / 256 := sorry

end soccer_tournament_probability_l245_245880


namespace sum_of_all_edges_ge_neg_10000_l245_245009

-- Define the graph with vertices and edges
def Graph := {vertices : Finset ℕ // vertices.card = 400}
def Edge := (ℕ × ℕ)

-- Define conditions
variable (G : Graph)

-- Each edge has a value of 1 or -1
def edge_value (e : Edge) : ℤ := 1 ∨ -1

-- Define a cuttlefish as the set of all edges incident to A and B including AB
def cuttlefish (A B : ℕ) : Finset Edge :=
  { e | e.fst = A ∨ e.snd = A ∨ e.fst = B ∨ e.snd = B }.toFinset

-- Sum of the values of edges in any cuttlefish is at least 1
def cuttlefish_sum_ge_one (A B : ℕ) : Prop :=
  (cuttlefish G.vertices A B).sum edge_value ≥ 1

-- Prove that the sum of the values of all edges is at least -10^4
theorem sum_of_all_edges_ge_neg_10000 : 
  (finset.univ : Finset Edge).sum edge_value ≥ -10000 := 
sorry

end sum_of_all_edges_ge_neg_10000_l245_245009


namespace inequality_always_true_l245_245520

theorem inequality_always_true (a : ℝ) (x : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  x^2 + (a - 4) * x + 4 - 2 * a > 0 → (x < 1 ∨ x > 3) :=
by {
  sorry
}

end inequality_always_true_l245_245520


namespace number_of_correct_statements_l245_245599

theorem number_of_correct_statements :
    let s1 := (round (1.804 : ℝ) * 100) / 100 = 1.80
    let s2 : ∀ a b c : ℝ, a + b + c = 0 → 
             let expr := (abs a / a) + (abs b / b) + (abs c / c) + (abs (a * b * c) / (a * b * c))
             expr = 0 ∨ expr = 1 ∨ expr = 2
    let s3 := ∀ (p1 p2 : Polynomial ℝ), p1.degree = 3 → p2.degree = 3 → (p1 + p2).degree = 3
    let s4 := let a := -8
              let b := -a-3
              a + b = -13
    in s1 ∧ ¬s2 ∧ ¬s3 ∧ ¬s4 :=
by
  let s1 := (Float.round (1.804: Float) * 100) / 100 = 1.80
  let s2 : ∀ a b c : ℝ, a + b + c = 0 → 
            let expr := (abs a / a) + (abs b / b) + (abs c / c) + (abs (a * b * c) / (a * b * c))
            expr = 0 ∨ expr = 1 ∨ expr = 2
  let s3 := ∀ (p1 p2 : Polynomial ℝ), p1.degree = 3 → p2.degree = 3 → (p1 + p2).degree = 3
  let s4 := let a := -8
            let b := -a - 3
            a + b = -13
  exact s1 ∧ ¬s2 ∧ ¬s3 ∧ ¬s4

end number_of_correct_statements_l245_245599


namespace pow_div_simplify_l245_245186

theorem pow_div_simplify : (((15^15) / (15^14))^3 * 3^3) / 3^3 = 3375 := by
  sorry

end pow_div_simplify_l245_245186


namespace complementary_angle_percentage_decrease_l245_245473

theorem complementary_angle_percentage_decrease :
  ∀ (α β : ℝ), α + β = 90 → 6 * α = 3 * β → 
  let α' := 1.2 * α in
  let β' := 90 - α' in
  (100 * (β' / β)) = 90 :=
by
  intros α β h_sum h_ratio α' β'
  have α' := 1.2 * α
  have β' := 90 - α'
  sorry

end complementary_angle_percentage_decrease_l245_245473


namespace order_of_numbers_l245_245638

theorem order_of_numbers : 
  let a := 0.7 ^ 6
  let b := 6 ^ 0.7
  let c := Real.log 6 / Real.log 0.7 -- since log_{0.7}(6) = log(6) / log(0.7)
  (c < a ∧ a < b) :=
by
  let a := 0.7 ^ 6
  let b := 6 ^ 0.7
  let c := Real.log 6 / Real.log 0.7
  sorry

end order_of_numbers_l245_245638


namespace magnitude_of_z_l245_245714

def z : ℂ := (1 + complex.I) / complex.I

theorem magnitude_of_z : complex.abs z = real.sqrt 2 := by
  sorry

end magnitude_of_z_l245_245714


namespace trapezoid_midline_segment_length_segment_MN_length_l245_245066

-- Definitions based on the conditions
def trapezoid (A B C D : Type) := sorry
def length (s : Type) (l : ℝ) := sorry

variable (A B C D : Type)
variable (a b p q : ℝ)
variable (h1 : a > b)
variable (AM MB DN NC : Type)

-- Mathematical equivalent structure of the proof problem
theorem trapezoid_midline_segment_length :
  length (trapezoid A B C D) A = a →
  length (trapezoid A B C D) B = a →
  length (trapezoid A B C D) C = b →
  length (trapezoid A B C D) D = b →
  trapezoid A B C D → 
  ((a > b) →
  (∃ (KL : ℝ), KL = (a - b) / 2)) := 
sorry

theorem segment_MN_length :
  length (trapezoid A B C D) A = a →
  length (trapezoid A B C D) B = a →
  length (trapezoid A B C D) C = b →
  length (trapezoid A B C D) D = b →
  AM : B = p*q →
  MB : B = p*q →
  DN : C = p*q →
  NC : C = p*q → 
  trapezoid A B C D → 
  (∃ (MN : ℝ), MN = (q * a + p * b) / (p + q)) := 
sorry

end trapezoid_midline_segment_length_segment_MN_length_l245_245066


namespace range_of_f_l245_245319

def f (x : ℝ) : ℝ := log (x + 1) / log 2  -- Defining the function f

theorem range_of_f :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1) :=
by
  sorry

end range_of_f_l245_245319


namespace coat_price_reduction_l245_245551

noncomputable def reduced_price_with_tax (price : ℝ) (reduction : ℝ) (tax : ℝ) : ℝ :=
  let reduced_price := price * (1 - reduction)
  let taxed_price := reduced_price * (1 + tax)
  taxed_price

theorem coat_price_reduction : 
  let initial_price : ℝ := 500
  let first_month_reduction : ℝ := 0.1
  let first_month_tax : ℝ := 0.05
  let second_month_reduction : ℝ := 0.15
  let second_month_tax : ℝ := 0.08
  let third_month_reduction : ℝ := 0.2
  let third_month_tax : ℝ := 0.06
  let price_after_first_month := reduced_price_with_tax initial_price first_month_reduction first_month_tax
  let price_after_second_month := reduced_price_with_tax price_after_first_month second_month_reduction second_month_tax
  let price_after_third_month := reduced_price_with_tax price_after_second_month third_month_reduction third_month_tax
  let total_percent_reduction := (initial_price - price_after_third_month) / initial_price * 100
  price_after_third_month ≈ 367.824 ∧ total_percent_reduction ≈ 26.44 := 
  by
    sorry

end coat_price_reduction_l245_245551


namespace probability_CE_l245_245420

/-- Given line segment AB, points A, B, C, and D on the segment, and point E as the midpoint of CD.
AB is 4 times AD and 5 times BC. Prove that the probability of a randomly selected point on AB falls between C and E is 1/4. --/
theorem probability_CE (A B C D E : Point) (AB AD BC : ℝ) (h1 : AB = 4 * AD) (h2 : AB = 5 * BC)
  (h3 : midpoint E C D) : probability (random_point_on_segment falls_between C E) = 1 / 4 :=
sorry

end probability_CE_l245_245420


namespace limit_value_l245_245296

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * real.log x + 8 * x

-- State the limit equality to be proved
theorem limit_value : (λ Δx : ℝ, (f (1 - 2 * Δx) - f 1) / Δx) →ₐ[ℝ] (-20) :=
by
  sorry

end limit_value_l245_245296


namespace max_value_of_expression_is_C_l245_245309

def max_expression_value (x : ℝ) (h : x ∈ Set.Icc (-Real.pi) (0 : ℝ)) : ℝ :=
  let y := (Real.tan (x + Real.pi / 4)) - (Real.tan (x + 3 * Real.pi / 4)) + Real.cos (x + Real.pi / 2)
  y

theorem max_value_of_expression_is_C :
  ∀ x ∈ Set.Icc (-Real.pi) (0 : ℝ), max_expression_value x _ <= Sorry
sorry

end max_value_of_expression_is_C_l245_245309


namespace knights_in_room_l245_245038

noncomputable def number_of_knights (n : ℕ) : ℕ :=
  if n = 15 then 9 else 0

theorem knights_in_room : ∀ (n : ℕ), 
  (n = 15 ∧ 
  (∀ (k l : ℕ), k + l = n ∧ k ≥ 8 ∧ l ≥ 6 → k = 9)) :=
begin
  intro n,
  split,
  { -- prove n = 15
    sorry,
  },
  { -- prove the number of knights k is 9 when conditions are met
    intros k l h,
    sorry
  }
end

end knights_in_room_l245_245038


namespace quadratic_roots_inverse_sum_l245_245124

theorem quadratic_roots_inverse_sum (a b c x1 x2 : ℝ) (h1 : a ≠ 0) 
  (h2 : a * x1^2 + b * x1 + c = 0) (h3 : a * x2^2 + b * x2 + c = 0) : 
  x1 ≠ 0 ∧ x2 ≠ 0 → x1^(-2) + x2^(-2) = (b^2 - 2 * a * c) / c^2 := 
by sorry

end quadratic_roots_inverse_sum_l245_245124


namespace smallest_sum_of_squares_l245_245070

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 145) : 
  ∃ x y, x^2 - y^2 = 145 ∧ x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l245_245070


namespace interest_correction_l245_245592

noncomputable def calculate_cents_credited (A : ℕ) (r : ℚ) (t : ℚ) : ℚ := 
  let P := A / (1 + r * t)
  let interest := A - P
  (interest - interest.to_int) * 100

theorem interest_correction (A : ℕ) (r : ℚ) (t : ℚ) : 
  A = 41248 → r = 0.04 → t = (3/12) → 
  calculate_cents_credited A r t = 8 := 
by 
  intros hA hr ht
  rw [hA, hr, ht]
  sorry

end interest_correction_l245_245592


namespace inequality_proof_l245_245983

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end inequality_proof_l245_245983


namespace arithmetic_sequence_property_l245_245343

variable {α : Type*} [LinearOrderedField α]

theorem arithmetic_sequence_property
  (a : ℕ → α) (h1 : a 1 + a 8 = 9) (h4 : a 4 = 3) : a 5 = 6 :=
by
  sorry

end arithmetic_sequence_property_l245_245343


namespace sequences_count_l245_245655

theorem sequences_count (a_n b_n c_n : ℕ → ℕ) :
  (a_n 1 = 1) ∧ (b_n 1 = 1) ∧ (c_n 1 = 1) ∧ 
  (∀ n : ℕ, a_n (n + 1) = a_n n + b_n n) ∧ 
  (∀ n : ℕ, b_n (n + 1) = a_n n + b_n n + c_n n) ∧ 
  (∀ n : ℕ, c_n (n + 1) = b_n n + c_n n) → 
  ∀ n : ℕ, a_n n + b_n n + c_n n = 
            (1/2 * ((1 + Real.sqrt 2)^(n+1) + (1 - Real.sqrt 2)^(n+1))) :=
by
  intro h
  sorry

end sequences_count_l245_245655


namespace contractor_absent_days_l245_245529

-- Definition of conditions
def total_days : ℕ := 30
def payment_per_work_day : ℝ := 25
def fine_per_absent_day : ℝ := 7.5
def total_payment : ℝ := 490

-- The proof statement
theorem contractor_absent_days : ∃ y : ℕ, (∃ x : ℕ, x + y = total_days ∧ payment_per_work_day * (x : ℝ) - fine_per_absent_day * (y : ℝ) = total_payment) ∧ y = 8 := 
by 
  sorry

end contractor_absent_days_l245_245529


namespace smallest_possible_positive_difference_l245_245506

def Vovochka_sum (a b : Nat) : Nat :=
  let ha := a / 100
  let ta := (a / 10) % 10
  let ua := a % 10
  let hb := b / 100
  let tb := (b / 10) % 10
  let ub := b % 10
  1000 * (ha + hb) + 100 * (ta + tb) + (ua + ub)

def correct_sum (a b : Nat) : Nat :=
  a + b

def difference (a b : Nat) : Nat :=
  abs (Vovochka_sum a b - correct_sum a b)

theorem smallest_possible_positive_difference :
  ∀ a b : Nat, (∃ a b : Nat, a > 0 ∧ b > 0 ∧ difference a b = 1800) :=
by
  sorry

end smallest_possible_positive_difference_l245_245506


namespace f_2019_equals_neg2_l245_245699

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 4) = f x)
variable (h_defined : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2)

theorem f_2019_equals_neg2 : f 2019 = -2 :=
by 
  sorry

end f_2019_equals_neg2_l245_245699


namespace larger_variance_means_larger_fluctuations_l245_245524

-- Definitions based on conditions
def data_set : Type := list ℝ

def mean (data : data_set) : ℝ := (data.sum / data.length)

def variance (data : data_set) : ℝ := 
  let m := mean data in
  (data.map (λ x, (x - m) ^ 2)).sum / data.length

-- A mathematically equivalent proof problem
theorem larger_variance_means_larger_fluctuations 
  (data1 data2 : data_set) (h1: variance data1 < variance data2) : 
  true := 
sorry

end larger_variance_means_larger_fluctuations_l245_245524


namespace Samia_walked_distance_approx_2point8_l245_245425

/-- Define constants for the given conditions --/
def bike_speed : ℝ := 20
def walk_speed : ℝ := 4
def total_time_minutes : ℝ := 50
def total_time_hours : ℝ := total_time_minutes / 60

/-- Define the total distance represented as 2x where x is the distance walked --/
def total_distance (x : ℝ) : ℝ := 2 * x

/-- Calculate the biking and walking times in hours --/
def biking_time (x : ℝ) : ℝ := x / bike_speed
def walking_time (x : ℝ) : ℝ := x / walk_speed

/-- Define the total calculated time from biking and walking --/
def total_calculated_time (x : ℝ) : ℝ := biking_time x + walking_time x

/-- Prove that the distance Samia walked is approximately 2.8 km, rounded to the nearest tenth --/
theorem Samia_walked_distance_approx_2point8 :
  ∃ (x : ℝ), 
  total_calculated_time x = total_time_hours 
  ∧ abs (x - 2.8) < 0.05 := 
sorry

end Samia_walked_distance_approx_2point8_l245_245425


namespace total_oranges_l245_245432

theorem total_oranges (initial_oranges : ℕ) (additional_oranges : ℕ) (weeks : ℕ) (multiplier : ℕ) :
  initial_oranges = 10 → additional_oranges = 5 → weeks = 2 → multiplier = 2 → 
  let first_week := initial_oranges + additional_oranges in
  let next_weeks := weeks * (multiplier * first_week) in
  first_week + next_weeks = 75 :=
begin
  intros h1 h2 h3 h4,
  let first_week := initial_oranges + additional_oranges,
  let next_weeks := weeks * (multiplier * first_week),
  have h_first : first_week = 15, 
  { rw [h1, h2], exact add_comm 10 5 },
  have h_next : next_weeks = 60, 
  { rw [h_first, h3, h4], exact mul_comm 2 30 },
  rw [h_first, h_next],
  exact add_comm 15 60,
end

end total_oranges_l245_245432


namespace find_lambda_l245_245252

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C M : V}
variable (λ : ℝ)

def is_centroid (M A B C : V) : Prop :=
  (A + B + C) / 3 = M

def satisfies_conditions (M A B C : V) (λ : ℝ) : Prop :=
  (A - M) + (B - M) + (C - M) = 0 ∧ (B - A) + (C - A) = λ * (A - M)

theorem find_lambda (h₁ : satisfies_conditions M A B C λ) : λ = 3 :=
by sorry

end find_lambda_l245_245252


namespace product_even_if_sum_odd_l245_245324

theorem product_even_if_sum_odd (a b : ℤ) (h : (a + b) % 2 = 1) : (a * b) % 2 = 0 :=
sorry

end product_even_if_sum_odd_l245_245324


namespace inequality_ABC_l245_245998

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_ABC_l245_245998


namespace max_a_plus_2b_l245_245711

theorem max_a_plus_2b (a b : ℝ) (h : a^2 + 2 * b^2 = 1) : a + 2 * b ≤ Real.sqrt 3 := 
sorry

end max_a_plus_2b_l245_245711


namespace platform_length_l245_245169

def speed_kmh_to_ms (s : ℕ) : ℕ := s * 1000 / 3600

theorem platform_length (L_t : ℕ) (S : ℕ) (T : ℕ) (h_lt : L_t = 360) (h_s : S = 45) (h_t : T = 43.2) : 
  ∃ L_p : ℕ, L_p = 180 := by
  let S_ms := speed_kmh_to_ms S
  have h_s_ms : S_ms = 12.5 := sorry
  let D := S_ms * T
  have h_d : D = 540 := sorry
  let L_p := D - L_t
  have h_lp : L_p = 180 := sorry
  exact ⟨L_p, h_lp⟩

end platform_length_l245_245169


namespace line_no_intersect_parabola_range_l245_245821

def parabola_eq (x : ℝ) : ℝ := x^2 + 4

def line_eq (m x : ℝ) : ℝ := m * (x - 10) + 6

theorem line_no_intersect_parabola_range (r s m : ℝ) :
  (m^2 - 40 * m + 8 = 0) →
  r < s →
  (∀ x, parabola_eq x ≠ line_eq m x) →
  r + s = 40 :=
by
  sorry

end line_no_intersect_parabola_range_l245_245821


namespace Kevin_speed_increase_l245_245384

noncomputable def d : ℝ := 600
noncomputable def v₁ : ℝ := 50
noncomputable def time_reduction : ℝ := 4

theorem Kevin_speed_increase :
  let t₁ := d / v₁,
      t₂ := t₁ - time_reduction,
      v₂ := d / t₂
  in v₂ - v₁ = 25 := by
  sorry

end Kevin_speed_increase_l245_245384


namespace find_arithmetic_sequence_elements_l245_245350

theorem find_arithmetic_sequence_elements :
  ∃ (a b c : ℤ), -1 < a ∧ a < b ∧ b < c ∧ c < 7 ∧
  (∃ d : ℤ, a = -1 + d ∧ b = -1 + 2 * d ∧ c = -1 + 3 * d ∧ 7 = -1 + 4 * d) :=
sorry

end find_arithmetic_sequence_elements_l245_245350


namespace area_change_l245_245289

variable (d x : ℝ)

-- Defining the area of the quadrilateral ACED as a function of x.
def area_ACED (d x : ℝ) : ℝ := (2 * d^2 + 4 * d * x - x^2) / (4 * Real.sqrt 3)

noncomputable def area_range (d x : ℝ) : Prop :=
  area_ACED d 0 = d^2 / (2 * Real.sqrt 3) ∧
  area_ACED d d = 5 * d^2 / (4 * Real.sqrt 3) ∧
  (∀ x, (0 ≤ x) ∧ (x ≤ d) → (d^2 / (2 * Real.sqrt 3) <= area_ACED d x) ∧ (area_ACED d x <= 5 * d^2 / (4 * Real.sqrt 3)))

theorem area_change (d : ℝ) : area_range d x :=
  sorry

end area_change_l245_245289


namespace purely_imaginary_condition_l245_245758

open Complex

noncomputable def complex_expression (a : ℝ) : ℂ := (a + I) / (1 + 2 * I)

theorem purely_imaginary_condition (a : ℝ) : Im (complex_expression a) ≠ 0 → Re (complex_expression a) = 0 → a = -2 :=
by
  sorry

end purely_imaginary_condition_l245_245758


namespace polynomial_simplification_l245_245879

-- Definition of the polynomials
def P1 := λ x : ℝ, 3 * x ^ 3 + 3 * x ^ 2 + 8 * x - 5
def P2 := λ x : ℝ, x ^ 3 + 6 * x ^ 2 + 2 * x - 15
def P3 := λ x : ℝ, 2 * x ^ 3 + x ^ 2 + 4 * x - 8
def simplifiedForm := λ x : ℝ, -4 * x ^ 2 + 2 * x + 18

-- The proof statement
theorem polynomial_simplification :
  ∀ x : ℝ, P1 x - P2 x - P3 x = simplifiedForm x :=
by
  sorry

end polynomial_simplification_l245_245879


namespace petya_vasya_eq_l245_245871

theorem petya_vasya_eq (a : ℕ) :
  ∃ M N : ℕ, 
  (∃ L1 L2 : list ℕ, 
     (L1 = list.range' a 20) ∧ 
     (L2 = list.range' (a + 1) 21) ∧ 
     M = L1.foldr (λ x acc, 10 * acc + x) 0 ∧ 
     N = L2.foldr (λ x acc, 10 * acc + x) 0) ∧ M = N :=
begin
  sorry
end

end petya_vasya_eq_l245_245871


namespace distinct_pairs_count_l245_245636

theorem distinct_pairs_count :
  (∃ (pairs : Finset (ℕ × ℕ)), 
    pairs.card = 20 ∧
    ∀ p ∈ pairs, 
      let x := p.1 in let y := p.2 in 
      0 < x ∧ x < y ∧ 42 = Nat.sqrt (1764) -> (Int.sqrt (x) + Int.sqrt (y))) :=
by
  sorry

end distinct_pairs_count_l245_245636


namespace hemisphere_surface_area_l245_245914

-- Define the condition of the problem
def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2
def base_area_of_hemisphere : ℝ := 3

-- The proof problem statement
theorem hemisphere_surface_area : 
  ∃ (r : ℝ), (Real.pi * r^2 = 3) → (2 * Real.pi * r^2 + Real.pi * r^2 = 9) := 
by 
  sorry

end hemisphere_surface_area_l245_245914


namespace smallest_difference_exists_l245_245496

-- Define the custom addition method used by Vovochka
def vovochka_add (a b : Nat) : Nat := 
  let hundreds := (a / 100) + (b / 100)
  let tens := ((a % 100) / 10) + ((b % 100) / 10)
  let units := (a % 10) + (b % 10)
  hundreds * 1000 + tens * 100 + units

-- Define the standard addition
def std_add (a b : Nat) : Nat := a + b

-- Define the function to compute the difference
def difference (a b : Nat) : Nat :=
  abs (vovochka_add a b - std_add a b)

-- Define the claim
theorem smallest_difference_exists : ∃ a b : Nat, 
  a < 1000 ∧ b < 1000 ∧ a > 99 ∧ b > 99 ∧ 
  difference a b = 1800 := 
sorry

end smallest_difference_exists_l245_245496


namespace tournament_independent_set_l245_245791
-- Define the conditions
variables {m : ℕ} (pairs1 pairs2 : fin m → ℕ × ℕ)

-- Define the main statement
theorem tournament_independent_set (h_pairs1 : ∀ i, pairs1 i.1 ≠ pairs1 i.2)
                                   (h_pairs2 : ∀ i, pairs2 i.1 ≠ pairs2 i.2)
                                   (h_distinct : ∀ i j, i ≠ j → pairs1 i ≠ pairs2 j)
                                   :
    ∃ (S : fin m → ℕ), (∀ i j, i ≠ j → S i ≠ S j) ∧
                       (∀ i, ∀ j, (pairs1 i).1 ≠ S j ∧ (pairs1 i).2 ≠ S j ∧
                                   (pairs2 i).1 ≠ S j ∧ (pairs2 i).2 ≠ S j) :=
sorry

end tournament_independent_set_l245_245791


namespace poles_needed_to_enclose_trapezoidal_plot_l245_245172

theorem poles_needed_to_enclose_trapezoidal_plot :
  (let side1 := 60; side2 := 30; side3 := 50; side4 := 40;
       distance1 := 5; distance2 := 4 in
   let poles_side1 := side1 / distance1 + 1 - 1;
       poles_side3 := side3 / distance1 + 1 - 1;
       poles_side2 := (side2 + (distance2 - 1)) / distance2 + 1 - 1;
       poles_side4 := (side4 + (distance2 - 1)) / distance2 + 1 - 1
   in
   poles_side1 + poles_side2 + poles_side3 + poles_side4 - 4 = 40) :=
by sorry

end poles_needed_to_enclose_trapezoidal_plot_l245_245172


namespace maximum_value_of_f_on_interval_max_value_interval_l245_245236

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x * Real.exp 1

open Real

theorem maximum_value_of_f_on_interval : 
  ∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), f x ≤ f 1 :=
sorry

example : f 1 = Real.exp 1 - 1 * Real.exp 1 := rfl

theorem max_value_interval :
  ∃ y ∈ set.Icc (0 : ℝ) (1 : ℝ), f y = Real.exp 1 - Real.exp 1 :=
begin
  use 1,
  split,
  norm_num, -- 1 ∈ [0,1]
  exact Real.exp 1 - Real.exp 1,
end

end maximum_value_of_f_on_interval_max_value_interval_l245_245236


namespace min_value_of_function_l245_245678

theorem min_value_of_function (x : ℝ) (h : x > 2) : (x + 1 / (x - 2)) ≥ 4 :=
  sorry

end min_value_of_function_l245_245678


namespace middle_three_sum_is_14_l245_245610

-- Given conditions
def green_cards : List ℕ := [1, 2, 3, 4, 5, 6, 7]
def yellow_cards : List ℕ := [4, 5, 6, 7, 8]

def alternates (lst : List ℕ) : Prop :=
  ∀ n, 0 < n → n < lst.length - 1 →
  (lst[n] ∈ green_cards ↔ lst[n+1] ∈ yellow_cards) ∧
  (lst[n] ∈ yellow_cards ↔ lst[n+1] ∈ green_cards)

def divisible_neighbors (lst : List ℕ) : Prop :=
  ∀ n, 0 < n → n < lst.length - 1 →
  (lst[n] ∈ green_cards →
    (lst[n-1] ∈ yellow_cards ∧ lst[n-1] % lst[n] = 0) ∧
    (lst[n+1] ∈ yellow_cards ∧ lst[n+1] % lst[n] = 0)) ∧
  (lst[n] ∈ yellow_cards →
    (lst[n-1] ∈ green_cards ∧ lst[n] % lst[n-1] = 0) ∧
    (lst[n+1] ∈ green_cards ∧ lst[n] % lst[n+1] = 0))

-- The list with cards satisfying both conditions
axiom valid_stack : List ℕ
  (alternates valid_stack) ∧ (divisible_neighbors valid_stack)

-- Middle three cards and their sum
def middle_three_sum : ℕ :=
  valid_stack[nat.div2 valid_stack.length - 1] +
  valid_stack[nat.div2 valid_stack.length] +
  valid_stack[nat.div2 valid_stack.length + 1]

-- Proof Problem Statement
theorem middle_three_sum_is_14 : middle_three_sum = 14 := sorry

end middle_three_sum_is_14_l245_245610


namespace expand_product_l245_245222

-- Define the problem
theorem expand_product (x : ℝ) : (x - 3) * (x + 4) = x^2 + x - 12 :=
by
  sorry

end expand_product_l245_245222


namespace number_of_rational_terms_l245_245682

noncomputable def a_n (n : ℕ) : ℝ :=
  1 / ((n + 1) * Real.sqrt n + n * Real.sqrt (n + 1))

def s_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), a_n k

def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem number_of_rational_terms :
  (Finset.range 2010).filter (λ n, is_rational (s_n n)).card = 43 :=
sorry

end number_of_rational_terms_l245_245682


namespace simplify_root_product_l245_245053

theorem simplify_root_product : 
  (625:ℝ)^(1/4) * (125:ℝ)^(1/3) = 25 :=
by
  have h₁ : (625:ℝ) = 5^4 := by norm_num
  have h₂ : (125:ℝ) = 5^3 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end simplify_root_product_l245_245053


namespace labor_budget_constraint_l245_245928

-- Define the conditions
def wage_per_carpenter : ℕ := 50
def wage_per_mason : ℕ := 40
def labor_budget : ℕ := 2000
def num_carpenters (x : ℕ) := x
def num_masons (y : ℕ) := y

-- The proof statement
theorem labor_budget_constraint (x y : ℕ) 
    (hx : wage_per_carpenter * num_carpenters x + wage_per_mason * num_masons y ≤ labor_budget) : 
    5 * x + 4 * y ≤ 200 := 
by sorry

end labor_budget_constraint_l245_245928


namespace cost_of_steel_ingot_l245_245489

theorem cost_of_steel_ingot :
  ∃ P : ℝ, 
    (∃ initial_weight : ℝ, initial_weight = 60) ∧
    (∃ weight_increase_percentage : ℝ, weight_increase_percentage = 0.6) ∧
    (∃ ingot_weight : ℝ, ingot_weight = 2) ∧
    (weight_needed = initial_weight * weight_increase_percentage) ∧
    (number_of_ingots = weight_needed / ingot_weight) ∧
    (number_of_ingots > 10) ∧
    (discount_percentage = 0.2) ∧
    (total_cost = 72) ∧
    (discounted_price_per_ingot = P * (1 - discount_percentage)) ∧
    (total_cost = discounted_price_per_ingot * number_of_ingots) ∧
    P = 5 := 
by
  sorry

end cost_of_steel_ingot_l245_245489


namespace sum_of_edges_of_pyramid_l245_245165

theorem sum_of_edges_of_pyramid :
  let s := 8 in
  let h := 10 in
  let side_edge_length := Real.sqrt ((s / 2 * Real.sqrt 2) ^ 2 + h ^ 2) in
  let total_length := 4 * s + 4 * side_edge_length in
  Int.round total_length = 78 :=
by
  let s := 8
  let h := 10
  let side_edge_length := Real.sqrt ((s / 2 * Real.sqrt 2) ^ 2 + h ^ 2)
  let total_length := 4 * s + 4 * side_edge_length
  have : Int.round total_length = 78 := sorry
  assumption

end sum_of_edges_of_pyramid_l245_245165


namespace part_a_part_b_l245_245225

theorem part_a (m : ℕ) : m = 1 ∨ m = 2 ∨ m = 4 → (3^m - 1) % (2^m) = 0 := by
  sorry

theorem part_b (m : ℕ) : m = 1 ∨ m = 2 ∨ m = 4 ∨ m = 6 ∨ m = 8 → (31^m - 1) % (2^m) = 0 := by
  sorry

end part_a_part_b_l245_245225


namespace area_of_triangle_AEF_l245_245603

theorem area_of_triangle_AEF
  (A B C D E F : Type)
  (area_ABCD : ℝ)
  (BE DF : ℝ)
  (area_ABCD_eq : area_ABCD = 56)
  (BE_eq : BE = 3)
  (DF_eq : DF = 2) :
  let area_AEF : ℝ := 25 in
  area_AEF = 25 :=
sorry

end area_of_triangle_AEF_l245_245603


namespace remaining_oil_quantity_check_remaining_oil_quantity_l245_245778

def initial_oil_quantity : Real := 40
def outflow_rate : Real := 0.2

theorem remaining_oil_quantity (t : Real) : Real :=
  initial_oil_quantity - outflow_rate * t

theorem check_remaining_oil_quantity (t : Real) : remaining_oil_quantity t = 40 - 0.2 * t := 
by 
  sorry

end remaining_oil_quantity_check_remaining_oil_quantity_l245_245778


namespace cannot_form_right_triangle_l245_245179

theorem cannot_form_right_triangle (a b c : Nat) (h₁ : (a, b, c) = (5, 12, 13) → a^2 + b^2 = c^2)
                                   (h₂ : (a, b, c) = (6, 8, 10) → a^2 + b^2 = c^2)
                                   (h₃ : (a, b, c) = (7, 24, 25) → a^2 + b^2 = c^2)
                                   (h₄ : (a, b, c) = (4, 6, 8)) : a^2 + b^2 ≠ c^2 :=
by {
  rw [h₄];
  have : ¬(4^2 + 6^2 = 8^2), by sorry;
  exact this
}

end cannot_form_right_triangle_l245_245179


namespace total_pastries_l245_245193

-- Defining the initial conditions
def Grace_pastries : ℕ := 30
def Calvin_pastries : ℕ := Grace_pastries - 5
def Phoebe_pastries : ℕ := Grace_pastries - 5
def Frank_pastries : ℕ := Calvin_pastries - 8

-- The theorem we want to prove
theorem total_pastries : 
  Calvin_pastries + Phoebe_pastries + Frank_pastries + Grace_pastries = 97 := by
  sorry

end total_pastries_l245_245193


namespace incorrect_reasoning_l245_245950

section IncorrectReasoning

variable {α : Type*} -- Universe declarations for sets
variables {l α : set α} {A : α}

theorem incorrect_reasoning
  (h1 : l ⊂ α) -- l is a proper subset of α
  (h2 : A ∈ l) -- A is in l
  : ¬(A ∉ α) := -- it is incorrect to say A is not in α
by
  sorry

end IncorrectReasoning

end incorrect_reasoning_l245_245950


namespace complex_mod_conjugate_division_l245_245312

def z : ℂ := 4 + 3 * complex.I

theorem complex_mod_conjugate_division :
  (Complex.conj z) / Complex.abs z = (4 / 5) - (3 / 5) * complex.I := 
sorry

end complex_mod_conjugate_division_l245_245312


namespace encode_key_5_decode_sucuri_encode_obmep_key_20_decode_2620138_key_20_key_sum_abc_is_52_l245_245135

-- Part a
theorem encode_key_5_decode_sucuri (key : ℕ) (encoded : List ℕ) :
  key = 5 ∧ encoded = [23, 25, 7, 25, 22, 13] → 
  (decode key encoded = "SUCURI") := sorry

-- Part b
theorem encode_obmep_key_20 (word : String) (key : ℕ) :
  word = "OBMEP" ∧ key = 20 → 
  (encode key word = [4, 17, 2, 20, 5]) := sorry

-- Part c
theorem decode_2620138_key_20 (seq : ℕ) (key : ℕ) :
  seq = 2620138 ∧ key = 20 → 
  (decode_number_sequence key seq = "GATO") := sorry

-- Part d
theorem key_sum_abc_is_52 (sum : ℕ) :
  sum = 52 → 
  find_key_sum_abc sum = 25 := sorry

end encode_key_5_decode_sucuri_encode_obmep_key_20_decode_2620138_key_20_key_sum_abc_is_52_l245_245135


namespace max_value_y_l245_245279

theorem max_value_y :
  ∀ x: ℝ, (0 ≤ x ∧ x ≤ 2) → 2^(2 * x - 1) - 3 * 2^x + 5 ≤ 5 / 2 := 
by
  intros x h
  sorry

end max_value_y_l245_245279


namespace volume_of_larger_cube_is_343_l245_245570

-- We will define the conditions first
def smaller_cube_side_length : ℤ := 1
def number_of_smaller_cubes : ℤ := 343
def volume_small_cube (l : ℤ) : ℤ := l^3
def diff_surface_area (l L : ℤ) : ℤ := (number_of_smaller_cubes * 6 * l^2) - (6 * L^2)

-- Main statement to prove the volume of the larger cube
theorem volume_of_larger_cube_is_343 :
  ∃ L, volume_small_cube smaller_cube_side_length * number_of_smaller_cubes = L^3 ∧
        diff_surface_area smaller_cube_side_length L = 1764 ∧
        volume_small_cube L = 343 :=
by
  sorry

end volume_of_larger_cube_is_343_l245_245570


namespace binomial_sum_of_coefficients_l245_245749

-- Given condition: for the third term in the expansion, the binomial coefficient is 15
def binomial_coefficient_condition (n : ℕ) := Nat.choose n 2 = 15

-- The goal: the sum of the coefficients of all terms in the expansion is 1/64
theorem binomial_sum_of_coefficients (n : ℕ) (h : binomial_coefficient_condition n) :
  (1:ℚ) / (2 : ℚ)^6 = 1 / 64 :=
by 
  have h₁ : n = 6 := by sorry -- Solve for n using the given condition.
  sorry -- Prove the sum of coefficients when x is 1.

end binomial_sum_of_coefficients_l245_245749


namespace lisa_total_spoons_l245_245847

def number_of_baby_spoons (num_children num_spoons_per_child : Nat) : Nat :=
  num_children * num_spoons_per_child

def number_of_decorative_spoons : Nat := 2

def number_of_old_spoons (baby_spoons decorative_spoons : Nat) : Nat :=
  baby_spoons + decorative_spoons
  
def number_of_new_spoons (large_spoons teaspoons : Nat) : Nat :=
  large_spoons + teaspoons

def total_number_of_spoons (old_spoons new_spoons : Nat) : Nat :=
  old_spoons + new_spoons

theorem lisa_total_spoons
  (children : Nat)
  (spoons_per_child : Nat)
  (large_spoons : Nat)
  (teaspoons : Nat)
  (children_eq : children = 4)
  (spoons_per_child_eq : spoons_per_child = 3)
  (large_spoons_eq : large_spoons = 10)
  (teaspoons_eq : teaspoons = 15)
  : total_number_of_spoons (number_of_old_spoons (number_of_baby_spoons children spoons_per_child) number_of_decorative_spoons) (number_of_new_spoons large_spoons teaspoons) = 39 :=
by
  sorry

end lisa_total_spoons_l245_245847


namespace smallest_b_l245_245882

theorem smallest_b
  (a b : ℕ)
  (h_pos : 0 < b)
  (h_diff : a - b = 8)
  (h_gcd : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) :
  b = 4 := sorry

end smallest_b_l245_245882


namespace smallest_difference_exists_l245_245495

-- Define the custom addition method used by Vovochka
def vovochka_add (a b : Nat) : Nat := 
  let hundreds := (a / 100) + (b / 100)
  let tens := ((a % 100) / 10) + ((b % 100) / 10)
  let units := (a % 10) + (b % 10)
  hundreds * 1000 + tens * 100 + units

-- Define the standard addition
def std_add (a b : Nat) : Nat := a + b

-- Define the function to compute the difference
def difference (a b : Nat) : Nat :=
  abs (vovochka_add a b - std_add a b)

-- Define the claim
theorem smallest_difference_exists : ∃ a b : Nat, 
  a < 1000 ∧ b < 1000 ∧ a > 99 ∧ b > 99 ∧ 
  difference a b = 1800 := 
sorry

end smallest_difference_exists_l245_245495


namespace distance_from_A_to_C_l245_245033

theorem distance_from_A_to_C :
  let A := (0 : ℂ)
  let C := (1170 + 1560*complex.I : ℂ)
  abs (C - A) = 1950 :=
by
  let A := (0 : ℂ)
  let C := (1170 + 1560 * complex.I : ℂ)
  have h : abs (C - A) = abs C, by simp
  have h_dist : abs C = 1950, sorry
  rw [h],
  exact h_dist

end distance_from_A_to_C_l245_245033


namespace pastries_total_l245_245195

-- We start by defining the conditions
def Calvin_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Phoebe_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Grace_pastries : ℕ := 30
def have_same_pastries (Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : Prop :=
  Calvin_pastries + 5 = Grace_pastries ∧ Phoebe_pastries + 5 = Grace_pastries

-- Total number of pastries held by Calvin, Phoebe, Frank, and Grace
def total_pastries (Frank_pastries Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : ℕ :=
  Frank_pastries + Calvin_pastries + Phoebe_pastries + Grace_pastries

-- The statement to prove
theorem pastries_total (Frank_pastries : ℕ) : 
  have_same_pastries (Calvin_pastries Frank_pastries Grace_pastries) (Phoebe_pastries Frank_pastries Grace_pastries) Grace_pastries → 
  Frank_pastries + Calvin_pastries Frank_pastries Grace_pastries + Phoebe_pastries Frank_pastries Grace_pastries + Grace_pastries = 97 :=
by
  sorry

end pastries_total_l245_245195


namespace inequality_proof_l245_245985

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l245_245985


namespace mayor_cup_num_teams_l245_245490

theorem mayor_cup_num_teams (x : ℕ) (h : x * (x - 1) / 2 = 21) : 
    ∃ x, x * (x - 1) / 2 = 21 := 
by
  sorry

end mayor_cup_num_teams_l245_245490


namespace angle_independent_of_line_g_l245_245386

-- Define a parallelogram as a structure
structure Parallelogram (A B C D : Type) :=
  (parallelogram_property : True) -- Simplification for the purposes of this example

-- Define a line through a point
structure LineThroughPoint (A : Type) :=
  (line_property : True) -- Simplification for the purposes of this example

-- Define a structure for excenters of a triangle
structure Excenter (A B X : Type) :=
  (excenter_property : True) -- Simplification for the poses of this example

-- Define angles
structure Angle (P Q R : Type) :=
  (angle_property : True) -- Simplification for the purposes of this example

-- The theorem statement
theorem angle_independent_of_line_g
  (A B C D X Y K L : Type) 
  (pgram: Parallelogram A B C D) 
  (line_g: LineThroughPoint A) 
  (intersects_BC: line_g→ X)
  (intersects_DC: line_g→ Y)
  (A_excenter_ABX: Excenter A B X → K)
  (A_excenter_ADY: Excenter A D Y → L)
  (angle_KCL : Angle K C L) :
  ∀ g, angle_KCL.angle_property := sorry

end angle_independent_of_line_g_l245_245386


namespace number_of_boys_l245_245337

-- Define variables for the number of girls and the difference in the number of girls and boys
variables (girls boys diff : ℕ)

-- Given conditions:
def number_of_girls : Prop := girls = 739
def difference_between_girls_and_boys : Prop := girls = boys + diff
def specific_difference : Prop := diff = 402

-- The statement we want to prove:
theorem number_of_boys (h1 : number_of_girls) (h2 : difference_between_girls_and_boys) (h3 : specific_difference) : boys = 337 :=
by sorry

end number_of_boys_l245_245337


namespace nissan_sales_l245_245544

noncomputable def numberOfNissans (totalCars : ℕ) (bmwPercent : ℚ) (fordPercent : ℚ) (chevyPercent : ℚ) : ℕ :=
  let totalPercent := bmwPercent + fordPercent + chevyPercent
  let nissanPercent := 1 - totalPercent
  totalCars * nissanPercent

theorem nissan_sales (totalCars : ℕ) (bmwPercent : ℚ) (fordPercent : ℚ) (chevyPercent : ℚ) :
  totalCars = 500 →
  bmwPercent = 0.18 →
  fordPercent = 0.20 →
  chevyPercent = 0.30 →
  numberOfNissans totalCars bmwPercent fordPercent chevyPercent = 160 :=
by
  intros h1 h2 h3 h4
  simp [numberOfNissans, h1, h2, h3, h4]
  sorry

end nissan_sales_l245_245544


namespace range_of_f_l245_245945

-- Definition of the function as given in the problem
def f (x : ℝ) : ℝ := (x^2 + 2 * x + 1) / (x + 2)

-- The statement to be proven
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y ∧ (x ≠ -2)) ↔ (y ∈ Set.Ioo (1 : ℝ) ∞ ∪ Set.Ioo (-(∞ : ℝ)) 1) :=
sorry

end range_of_f_l245_245945


namespace circle_passes_through_fixed_point_l245_245706

open Real

theorem circle_passes_through_fixed_point :
  ∀ (C : ℝ × ℝ → ℝ) (h1 : ∀ p : ℝ × ℝ, p.2^2 = 4 * p.1 → C p = dist (p.1, p.2) (1, 0)) (h2 : ∀ p : ℝ × ℝ, C p = dist p (-1 : ℝ, 0) → C p = dist (1, 0)),
  C (1, 0) = 0 :=
by
  sorry

end circle_passes_through_fixed_point_l245_245706


namespace eval_f_l245_245672

def f (x : ℚ) : ℚ :=
if x < 1/2 then 2 * x - 1 else f (x - 1) + 1

theorem eval_f :
  f (1/4) + f (7/6) = -1/6 :=
by
  -- put your proof here
  sorry


end eval_f_l245_245672


namespace hemisphere_surface_area_l245_245912

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (hπ : π = Real.pi) (h : π * r^2 = 3) :
    2 * π * r^2 + 3 = 9 :=
by
  sorry

end hemisphere_surface_area_l245_245912


namespace B_k_largest_at_45_l245_245645

def B_k (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.1)^k

theorem B_k_largest_at_45 : ∀ k : ℕ, k = 45 → ∀ m : ℕ, m ≠ 45 → B_k 45 > B_k m :=
by
  intro k h_k m h_m
  sorry

end B_k_largest_at_45_l245_245645


namespace greatest_prime_factor_391_l245_245113

theorem greatest_prime_factor_391 : ∃ p, Prime p ∧ p ∣ 391 ∧ ∀ q, Prime q ∧ q ∣ 391 → q ≤ p :=
by
  sorry

end greatest_prime_factor_391_l245_245113


namespace fuel_tank_oil_quantity_l245_245780

theorem fuel_tank_oil_quantity (t : ℝ) (Q : ℝ) : (Q = 40 - 0.2 * t) :=
begin
  sorry
end

end fuel_tank_oil_quantity_l245_245780


namespace four_ships_distances_satisfiable_l245_245619

def canPositionFourShipsWithDistances : Prop :=
  ∃ (a b c d : ℝ), 
    { |b - a|, |c - a|, |d - a|, |c - b|, |d - b|, |d - c| } = {1, 2, 3, 4, 5, 6}

theorem four_ships_distances_satisfiable : canPositionFourShipsWithDistances :=
  sorry

end four_ships_distances_satisfiable_l245_245619


namespace find_a_in_terms_of_x_l245_245061

theorem find_a_in_terms_of_x (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a^3 - b^3 = 22 * x^3) (h₃ : a - b = 2 * x) : 
  a = x * (1 + (Real.sqrt (40 / 3)) / 2) ∨ a = x * (1 - (Real.sqrt (40 / 3)) / 2) :=
by
  sorry

end find_a_in_terms_of_x_l245_245061


namespace original_strength_of_class_l245_245133

variable (N : ℕ) -- original strength of the class
variable (T : ℕ) -- total age of the original class
variable (avg_original : ℕ) -- average age of the original class
variable (new_students : ℕ) -- number of new students
variable (avg_new_students : ℕ) -- average age of new students
variable (age_decrease : ℕ) -- decrease in the average age

-- Given conditions
def conditions : Prop :=
  avg_original = 40 ∧
  new_students = 8 ∧
  avg_new_students = 32 ∧
  age_decrease = 4 ∧
  T = avg_original * N ∧
  T + new_students * avg_new_students = (avg_original - age_decrease) * (N + new_students)

-- Proof statement: the original strength of the class is 8
theorem original_strength_of_class (h : conditions) : N = 8 := by
  sorry

end original_strength_of_class_l245_245133


namespace option_B_option_D_l245_245275

noncomputable def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- The maximum value of (y + 1) / (x + 1) is 2 + sqrt(6)
theorem option_B (x y : ℝ) (h : curve_C x y) :
  ∃ k, (y + 1) / (x + 1) = k ∧ k = 2 + Real.sqrt 6 :=
sorry

-- A tangent line through the point (0, √2) on curve C has the equation x - √2 * y + 2 = 0
theorem option_D (h : curve_C 0 (Real.sqrt 2)) :
  ∃ a b c, a * 0 + b * Real.sqrt 2 + c = 0 ∧ c = 2 ∧ a = 1 ∧ b = - Real.sqrt 2 :=
sorry

end option_B_option_D_l245_245275


namespace range_of_a_l245_245716

theorem range_of_a (a : ℝ) :
  (∃ x ∈ set.Icc (-1 : ℝ) (0: ℝ), 2 * ( (1/4 : ℝ) ^ -x ) - ( (1/2 : ℝ) ^ -x ) + a = 0) ↔ a ∈ set.Icc (-1 : ℝ) (0 : ℝ) :=
sorry

end range_of_a_l245_245716


namespace correct_options_l245_245276

-- Given condition
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- Option B assertion
def option_B (x y : ℝ) : Prop := (x^2 + y^2 - 4)/((x - 1)^2 + y^2 + 1) ≤ 2 + Real.sqrt 6

-- Option D assertion
def option_D (x y : ℝ) : Prop := x - Real.sqrt 2 * y + 2 = 0

-- Theorem to prove both options B and D are correct under the given condition
theorem correct_options {x y : ℝ} (h : curve x y) : option_B x y ∧ option_D x y := by
  sorry

end correct_options_l245_245276


namespace abc_min_value_l245_245826

open Real

theorem abc_min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_sum : a + b + c = 1) (h_bound : a ≤ b ∧ b ≤ c ∧ c ≤ 3 * a) :
  3 * a * a * (1 - 4 * a) = (9/343) := 
sorry

end abc_min_value_l245_245826


namespace function_equality_l245_245021

theorem function_equality
  (f : ℕ+ → ℝ)
  (g : ℕ+ → ℝ)
  (h1 : f 1 = 1)
  (h2 : f 2 = 2)
  (h3 : ∀ n : ℕ+, f (n + 2) ≥ 2 * f (n + 1) - f n)
  (h4 : ∀ n : ℕ+, g n ≥ 1)
  (n : ℕ+) :
  f n = 2 + ∑ i in Finset.range (n - 2), ∏ j in Finset.range i.succ, g ⟨j, sorry⟩ :=
sorry

end function_equality_l245_245021


namespace bakery_children_count_l245_245921

def bakery_problem (initial_children girls_came_in boys_left : ℕ) : Prop :=
  initial_children + girls_came_in - boys_left = 78

theorem bakery_children_count : bakery_problem 85 24 31 :=
by
  unfold bakery_problem
  simp
  sorry

end bakery_children_count_l245_245921


namespace rice_less_than_beans_by_30_l245_245854

noncomputable def GB : ℝ := 60
noncomputable def S : ℝ := 50

theorem rice_less_than_beans_by_30 (R : ℝ) (x : ℝ) (h1 : R = 60 - x) (h2 : (2/3) * R + (4/5) * S + GB = 120) : 60 - R = 30 :=
by 
  -- Proof steps would go here, but they are not required for this task.
  sorry

end rice_less_than_beans_by_30_l245_245854


namespace Lisa_total_spoons_l245_245851

def total_spoons (children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) (large_spoons : ℕ) (teaspoons : ℕ) : ℕ := 
  (children * spoons_per_child) + decorative_spoons + (large_spoons + teaspoons)

theorem Lisa_total_spoons :
  (total_spoons 4 3 2 10 15) = 39 :=
by
  sorry

end Lisa_total_spoons_l245_245851


namespace range_of_a_l245_245676

theorem range_of_a (a x : ℝ) (p : 0.5 ≤ x ∧ x ≤ 1) (q : (x - a) * (x - a - 1) > 0) :
  (0 ≤ a ∧ a ≤ 0.5) :=
by 
  sorry

end range_of_a_l245_245676


namespace josh_paths_l245_245359

theorem josh_paths (n : ℕ) (h : n > 0) : 
  let start := (0, 0)
  let end := (n - 1, 1)
  -- the number of distinct paths from start to end is 2^(n-1)
  (if n = 1 then 1 else 2^(n-1)) = 2^(n-1) :=
by
  sorry

end josh_paths_l245_245359


namespace line_slope_point_l245_245155

theorem line_slope_point (m b : ℝ) (h_slope : m = -4) (h_point : ∃ x y : ℝ, (x, y) = (5, 2) ∧ y = m * x + b) : 
  m + b = 18 := by
  sorry

end line_slope_point_l245_245155


namespace greatest_prime_factor_391_l245_245105

theorem greatest_prime_factor_391 : 
  greatestPrimeFactor 391 = 23 :=
sorry

end greatest_prime_factor_391_l245_245105


namespace odd_function_f_neg_one_l245_245018

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 2 then 2^x else 0 -- Placeholder; actual implementation skipped for simplicity

theorem odd_function_f_neg_one :
  (∀ x, f (-x) = -f x) ∧ (∀ x, (0 < x ∧ x < 2) → f x = 2^x) → 
  f (-1) = -2 :=
by
  intros h
  let odd_property := h.1
  let condition_in_range := h.2
  sorry

end odd_function_f_neg_one_l245_245018


namespace floor_sum_l245_245392

variable (x : ℝ) (y : ℝ) 

def floor_eq : ℝ := Real.floor x

theorem floor_sum :
  Real.floor (2017 * 3 / 11) +
  Real.floor (2017 * 4 / 11) +
  Real.floor (2017 * 5 / 11) +
  Real.floor (2017 * 6 / 11) +
  Real.floor (2017 * 7 / 11) +
  Real.floor (2017 * 8 / 11) =
  6048 := sorry

end floor_sum_l245_245392


namespace determine_x_value_l245_245213

noncomputable def x : ℝ :=
  3 + 5 / (2 + 5 / x)

theorem determine_x_value :
  ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) ∧ x = (3 + Real.sqrt 39) / 2 :=
  sorry

end determine_x_value_l245_245213


namespace geometric_series_solution_l245_245199

/-- Given the infinite geometric series properties, prove the value of x such that
(1 + 1/2 + 1/4 + 1/8 + ...) * (1 - 1/2 + 1/4 - 1/8 + ...) = (1 + 1/x + 1/x^2 + 1/x^3 + ...).
-/
theorem geometric_series_solution :
  (∑' n : ℕ, (1 / 2) ^ n) * (∑' n : ℕ, (-1 / 2) ^ n) = ∑' n : ℕ, (1 : ℚ) / 4 ^ n →
  ∃ x : ℚ, (x = 4) :=
by
  intro h
  use 4
  sorry

end geometric_series_solution_l245_245199


namespace age_of_markus_great_grandson_l245_245855

variable (Markus Son Grandson GreatGrandson : ℝ)

-- Conditions
def markus_is_twice_the_age_of_his_son :: Markus = 2 * Son
def son_is_twice_the_age_of_grandson :: Son = 2 * Grandson
def grandson_is_3_5_times_the_age_of_greatGrandson :: Grandson = 3.5 * GreatGrandson
def sum_of_ages_is_140 :: Markus + Son + Grandson + GreatGrandson = 140

-- Statement to prove
theorem age_of_markus_great_grandson: 
  (Markus = 2 * Son) ∧ 
  (Son = 2 * Grandson) ∧ 
  (Grandson = 3.5 * GreatGrandson) ∧ 
  (Markus + Son + Grandson + GreatGrandson = 140) → 
  GreatGrandson ≈ 5.49 :=
by
  sorry

end age_of_markus_great_grandson_l245_245855
