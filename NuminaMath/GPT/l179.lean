import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Composition
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Init
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology.Basic

namespace train_speeds_and_distance_l179_179669

noncomputable def find_train_params (p t q : ℝ) : ℝ × ℝ × ℝ :=
  let z := 3 * p - q
  let x := (4 * p - 2 * q) / t
  let y := 2 * p / t
  (x, y, z)
  
theorem train_speeds_and_distance (p t q : ℝ) (hp : p > 0) (ht : t > 0) (hq : q > 0) :
  ∃ (x y z : ℝ), (x, y, z) = find_train_params p t q :=
by
  use ((4 * p - 2 * q) / t), (2 * p / t), (3 * p - q)
  simp [find_train_params]
  sorry

end train_speeds_and_distance_l179_179669


namespace probability_of_valid_quadrilateral_l179_179585

-- Define a regular octagon
def regular_octagon_sides : ℕ := 8

-- Total number of ways to choose 4 sides from 8 sides
def total_ways_choose_four_sides : ℕ := Nat.choose 8 4

-- Number of ways to choose 4 adjacent sides (invalid)
def invalid_adjacent_ways : ℕ := 8

-- Number of ways to choose 4 sides with 3 adjacent unchosen sides (invalid)
def invalid_three_adjacent_unchosen_ways : ℕ := 8 * 3

-- Total number of invalid ways
def total_invalid_ways : ℕ := invalid_adjacent_ways + invalid_three_adjacent_unchosen_ways

-- Total number of valid ways
def total_valid_ways : ℕ := total_ways_choose_four_sides - total_invalid_ways

-- Probability of forming a quadrilateral that contains the octagon
def probability_valid_quadrilateral : ℚ :=
  (total_valid_ways : ℚ) / (total_ways_choose_four_sides : ℚ)

-- Theorem statement
theorem probability_of_valid_quadrilateral :
  probability_valid_quadrilateral = 19 / 35 :=
by
  sorry

end probability_of_valid_quadrilateral_l179_179585


namespace range_of_a_intersection_nonempty_range_of_a_intersection_A_l179_179825

noncomputable def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a_intersection_nonempty (a : ℝ) : (A a ∩ B ≠ ∅) ↔ (a < -1 ∨ a > 2) :=
sorry

theorem range_of_a_intersection_A (a : ℝ) : (A a ∩ B = A a) ↔ (a < -4 ∨ a > 5) :=
sorry

end range_of_a_intersection_nonempty_range_of_a_intersection_A_l179_179825


namespace jessica_cut_2_roses_l179_179625

theorem jessica_cut_2_roses (initial_roses : ℕ) (new_roses : ℕ) (initial_roses_eq : initial_roses = 15) (new_roses_eq : new_roses = 17) :
  new_roses - initial_roses = 2 :=
by
  rw [initial_roses_eq, new_roses_eq]
  exact rfl

end jessica_cut_2_roses_l179_179625


namespace volume_of_EFGH_l179_179057

noncomputable def cube_edge_length : ℝ := 2

def is_cube (A B C D E F G H : ℝ × ℝ × ℝ) : Prop :=
  ∃ s, s = cube_edge_length ∧
    dist A B = s ∧ dist A D = s ∧ dist A E = s ∧
    dist B C = s ∧ dist B F = s ∧
    dist C D = s ∧ dist C G = s ∧
    dist D H = s ∧ dist E F = s ∧
    dist F G = s ∧ dist G H = s ∧
    dist E H = s

def volume_of_pyramid (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem volume_of_EFGH
  (A B C D E F G H : ℝ × ℝ × ℝ)
  (h_cube : is_cube A B C D E F G H)
  (edge_length_two : cube_edge_length = 2) :
  volume_of_pyramid 1 cube_edge_length = 2 / 3 :=
by
  -- Proof would go here
  sorry

end volume_of_EFGH_l179_179057


namespace compute_expression_l179_179726

theorem compute_expression : (-9 * 3 - (-7 * -4) + (-11 * -6) = 11) := by
  sorry

end compute_expression_l179_179726


namespace sum_g_from_3_infty_l179_179388

noncomputable def g (n : ℕ) : ℝ :=
\sum_{k : ℕ in (3 : ℕ) → nat_succ n, (1 : ℝ) / k^n 

theorem sum_g_from_3_infty :
  \sum_{n = 3}^\infty (g (n)) = \frac{1}{3} := by
  sorry

end sum_g_from_3_infty_l179_179388


namespace minimum_apples_collected_l179_179115

-- Anya, Vanya, Dania, Sanya, and Tanya each collected an integer percentage of the total number of apples,
-- with all these percentages distinct and greater than zero.
-- Prove that the minimum total number of apples is 20.

theorem minimum_apples_collected :
  ∃ (n : ℕ), (∀ (a v d s t : ℕ), 
    1 ≤ a ∧ 1 ≤ v ∧ 1 ≤ d ∧ 1 ≤ s ∧ 1 ≤ t ∧
    a ≠ v ∧ a ≠ d ∧ a ≠ s ∧ a ≠ t ∧ 
    v ≠ d ∧ v ≠ s ∧ v ≠ t ∧ 
    d ≠ s ∧ d ≠ t ∧ 
    s ≠ t ∧
    a + v + d + s + t = 100) →
  n ≥ 20 :=
by 
  sorry

end minimum_apples_collected_l179_179115


namespace simplify_expression_l179_179716

theorem simplify_expression (x : ℝ) : x + 3 - 4x - 5 + 6x + 7 - 8x - 9 = -5 * x - 4 := 
by {
  sorry
}

end simplify_expression_l179_179716


namespace exponent_of_5_in_30_factorial_l179_179981

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179981


namespace hyperbola_line_intersections_l179_179440

-- Define the hyperbola and line equations
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4
def line (x y k : ℝ) : Prop := y = k * (x - 1)

-- Conditions for intersecting the hyperbola at two points
def intersect_two_points (k : ℝ) : Prop := 
  k ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-1) ∨ 
  k ∈ Set.Ioo (-1) 1 ∨ 
  k ∈ Set.Ioo 1 (2 * Real.sqrt 3 / 3)

-- Conditions for intersecting the hyperbola at exactly one point
def intersect_one_point (k : ℝ) : Prop := 
  k = 1 ∨ 
  k = -1 ∨ 
  k = 2 * Real.sqrt 3 / 3 ∨ 
  k = -2 * Real.sqrt 3 / 3

-- Proof that k is in the appropriate ranges
theorem hyperbola_line_intersections (k : ℝ) :
  ((∃ x y : ℝ, hyperbola x y ∧ line x y k) 
  → (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ hyperbola x₁ y₁ ∧ line x₁ y₁ k ∧ hyperbola x₂ y₂ ∧ line x₂ y₂ k) 
  → intersect_two_points k))
  ∧ ((∃ x y : ℝ, hyperbola x y ∧ line x y k) 
  → (∃ x y : ℝ, (hyperbola x y ∧ line x y k ∧ (∀ x' y', hyperbola x' y' ∧ line x' y' k → (x' ≠ x ∨ y' ≠ y) = false)) 
  → intersect_one_point k)) := 
sorry

end hyperbola_line_intersections_l179_179440


namespace john_profit_percentage_is_50_l179_179525

noncomputable def profit_percentage
  (P : ℝ)  -- The sum of money John paid for purchasing 30 pens
  (recovered_amount : ℝ)  -- The amount John recovered when he sold 20 pens
  (condition : recovered_amount = P) -- Condition that John recovered the full amount P when he sold 20 pens
  : ℝ := 
  ((P / 20) - (P / 30)) / (P / 30) * 100

theorem john_profit_percentage_is_50
  (P : ℝ)
  (recovered_amount : ℝ)
  (condition : recovered_amount = P) :
  profit_percentage P recovered_amount condition = 50 := 
  by 
  sorry

end john_profit_percentage_is_50_l179_179525


namespace arrange_in_non_decreasing_order_l179_179131

def Psi : ℤ := (1 / 2 : ℚ) * (Finset.sum (Finset.range 1006) (λ i, 1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + (4 * i)))

def Omega : ℤ := Finset.sum (Finset.range 1007) (λ i, 1 - 2 + 3 - 4 + (2 * i))

def Theta : ℤ := Finset.sum (Finset.range 504) (λ i, 1 - 3 + 5 - 7 + (4 * i))

theorem arrange_in_non_decreasing_order : [Theta, Omega, Psi] = [(-1008 : ℤ), (-1007 : ℤ), (-1006 : ℤ)] :=
by
  have : Psi = -1006 := sorry
  have : Omega = -1007 := sorry
  have : Theta = -1008 := sorry
  rw [this, this, this]
  trivial

end arrange_in_non_decreasing_order_l179_179131


namespace number_of_white_balls_l179_179068

-- Defining the given conditions
def total_balls : ℕ := 60
def green_balls : ℕ := 10
def yellow_balls : ℕ := 7
def red_balls : ℕ := 15
def purple_balls : ℕ := 6
def prob_neither_red_nor_purple : ℚ := 0.65

-- Problem statement: How many white balls?
theorem number_of_white_balls : 
  let balls := total_balls 
  let red_and_purple := red_balls + purple_balls
  let prob_red_or_purple := 1 - prob_neither_red_nor_purple
  let balls_neither_red_nor_purple := prob_neither_red_nor_purple * balls
  let other_balls := green_balls + yellow_balls
  let white_balls := balls_neither_red_nor_purple - other_balls
  in white_balls = 22 := by
  sorry

end number_of_white_balls_l179_179068


namespace yellow_to_blue_apples_ratio_l179_179837

theorem yellow_to_blue_apples_ratio : 
  ∀ (Y : ℕ), 
    let b := 5 in
    let T := b + Y in
    let given_to_son := (1 / 5 : ℚ) * T in
    let remaining_apples := T - given_to_son in
    remaining_apples = 12 →
    Y / b = 2 :=
by
  sorry

end yellow_to_blue_apples_ratio_l179_179837


namespace exponent_of_5_in_30_factorial_l179_179948

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179948


namespace KHSO4_formed_l179_179382

-- Define the reaction condition and result using moles
def KOH_moles : ℕ := 2
def H2SO4_moles : ℕ := 2

-- The balanced chemical reaction in terms of moles
-- 1 mole of KOH reacts with 1 mole of H2SO4 to produce 
-- 1 mole of KHSO4
def react (koh : ℕ) (h2so4 : ℕ) : ℕ := 
  -- stoichiometry 1:1 ratio of KOH and H2SO4 to KHSO4
  if koh ≤ h2so4 then koh else h2so4

-- The proof statement that verifies the expected number of moles of KHSO4
theorem KHSO4_formed (koh : ℕ) (h2so4 : ℕ) (hrs : react koh h2so4 = koh) : 
  koh = KOH_moles → h2so4 = H2SO4_moles → react koh h2so4 = 2 := 
by
  intros 
  sorry

end KHSO4_formed_l179_179382


namespace hyperbola_range_l179_179847

theorem hyperbola_range (m : ℝ) : (∃ x y : ℝ, (x^2 / (|m| - 1) - y^2 / (m - 2) = 1)) → (-1 < m ∧ m < 1) ∨ (m > 2) := by
  sorry

end hyperbola_range_l179_179847


namespace inverse_modulo_l179_179408

theorem inverse_modulo (h : 11 * 43 ≡ 1 [MOD 47]) : ∃ x : ℤ, 36 * x ≡ 1 [MOD 47] ∧ 0 ≤ x ∧ x ≤ 46 :=
by 
  have h₁ : 36 ≡ -11 [MOD 47], by 
    calc 36 ≡ -11 [MOD 47] : by norm_num
  have h₂ : -11 * 43 ≡ 1 [MOD 47], by 
    calc (-11) * 43 ≡ (-1) * (11 * 43) [MOD 47] : by norm_num
                ... ≡ (-1) * 1 [MOD 47] : by rw h
                ... ≡ -1 [MOD 47]       : by norm_num
  have h₃ : 36 * 43 ≡ -(-1) [MOD 47] := by 
    calc 36 * 43 ≡ (-11) * 43 [MOD 47] : by rw h₁
             ... ≡ -1 [MOD 47] : h₂
             ... ≡ 1 [MOD 47]  : by norm_num
  use 43
  split
  exact h₃
  split
  linarith
  linarith

end inverse_modulo_l179_179408


namespace totalBooksOnShelves_l179_179053

-- Define the conditions
def numShelves : Nat := 150
def booksPerShelf : Nat := 15

-- Define the statement to be proved
theorem totalBooksOnShelves : numShelves * booksPerShelf = 2250 :=
by
  -- Skipping the proof
  sorry

end totalBooksOnShelves_l179_179053


namespace exponent_of_5_in_30_factorial_l179_179976

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179976


namespace semi_minor_axis_is_sqrt_21_l179_179870

-- Define the conditions
def center : ℝ × ℝ := (0, 0)
def focus : ℝ × ℝ := (0, -2)
def semi_major_endpoint : ℝ × ℝ := (0, 5)

-- Define the distances calculated from the conditions
def c : ℝ := ∥(0, 0) - (0, -2)∥
def a : ℝ := ∥(0, 0) - (0, 5)∥

-- State the theorem: the semi-minor axis of the ellipse is √21
theorem semi_minor_axis_is_sqrt_21 : ∥(0, 0) - (0, 5)∥^2 - ∥(0, 0) - (0, -2)∥^2 = 21 :=
by
  sorry

end semi_minor_axis_is_sqrt_21_l179_179870


namespace sin_45_eq_sqrt_two_over_two_l179_179246

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179246


namespace max_slope_tangent_eqn_l179_179761

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem max_slope_tangent_eqn (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) :
    (∃ m b, m = Real.sqrt 2 ∧ b = -Real.sqrt 2 * (Real.pi / 4) ∧ 
    (∀ y, y = m * x + b)) :=
sorry

end max_slope_tangent_eqn_l179_179761


namespace sin_45_eq_sqrt2_div_2_l179_179337

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179337


namespace sin_alpha_cos_3alpha_minus_beta_l179_179776

/-
  Given: 
  0 < α < π/3 
  -π/2 < β < -π/3
  cos(α + π/6) = √3/3
  sin(α - β + π/6) = 2/3

  Prove:
  sin(α) = (3√2 - √3)/6
  cos(3α - β) = -(2√10 + 2)/9
-/

variables (α β : ℝ)

axiom cond1 : 0 < α ∧ α < π / 3
axiom cond2 : -π / 2 < β ∧ β < -π / 3
axiom cond3 : cos (α + π / 6) = sqrt 3 / 3
axiom cond4 : sin (α - β + π / 6) = 2 / 3

theorem sin_alpha : sin α = (3 * sqrt 2 - sqrt 3) / 6 := by
  sorry

theorem cos_3alpha_minus_beta : cos (3 * α - β) = -(2 * sqrt 10 + 2) / 9 := by
  sorry

end sin_alpha_cos_3alpha_minus_beta_l179_179776


namespace length_DF_zero_l179_179529

noncomputable def triangle_ABC (AB BC AC : ℝ) : Prop :=
AB = 3 ∧ BC = 4 ∧ AC = 5

noncomputable def projection_B_to_AC {A B C D : Type*} [EuclideanGeometry A B C D] 
  (AB BC AC : ℝ) (D_proj : D) (h_triangle : triangle_ABC AB BC AC) : Prop :=
is_projection B AC D_proj

noncomputable def projection_D_to_BC {A B C D E : Type*} [EuclideanGeometry A B C D E]
  (D_proj E_proj : E) (h_proj : projection_B_to_AC 3 4 5 D_proj) : Prop :=
is_projection D_proj BC E_proj

noncomputable def projection_E_to_AC {A B C D E F : Type*} [EuclideanGeometry A B C D E F]
  (E_proj F_proj : F) (h_proj : projection_D_to_BC D_proj 4 5 E_proj) : Prop :=
is_projection E_proj AC F_proj

noncomputable def segment_length {A B : Type*} [EuclideanGeometry A B] (seg : Segment A B) : ℝ :=
length seg

theorem length_DF_zero 
  (DF : ℝ) 
  (h_triangle_ABC : triangle_ABC 3 4 5)
  (h_proj_BD : projection_B_to_AC 3 4 5 D)
  (h_proj_DE : projection_D_to_BC D 4 5 E)
  (h_proj_EF : projection_E_to_AC E 4 5 F): 
  segment_length (DF) = 0 := 
sorry

end length_DF_zero_l179_179529


namespace sin_45_eq_l179_179267

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179267


namespace extremum_at_neg3_l179_179439

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 + 5 * x^2 + a * x
def f_deriv (x : ℝ) : ℝ := 3 * x^2 + 10 * x + a

theorem extremum_at_neg3 (h : f_deriv a (-3) = 0) : a = 3 := 
  by
  sorry

end extremum_at_neg3_l179_179439


namespace range_of_lambda_l179_179411

noncomputable def vectors_basis {α : Type*} [field α] 
  (e1 e2 : α × α) (a b : α × α) (λ : α) : Prop :=
  ¬ collinear e1 e2 → 
  a = (e1.1 + (2 : α) * e2.1, e1.2 + (2 : α) * e2.2) → 
  b = (2 * e1.1 + λ * e2.1, 2 * e1.2 + λ * e2.2) → 
  ¬ collinear a b

theorem range_of_lambda (e1 e2 : ℝ × ℝ) (λ : ℝ):
  ¬ collinear e1 e2 → 
  λ ∉ set.Ioo 4 4 → 
  set.Ico λ ∞ :=
sorry

end range_of_lambda_l179_179411


namespace sin_45_eq_sqrt2_div_2_l179_179284

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179284


namespace prime_exponent_of_5_in_30_factorial_l179_179911

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179911


namespace exponent_of_5_in_30_factorial_l179_179996

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179996


namespace original_price_l179_179609

theorem original_price (P : ℝ) (final_price : ℝ) (percent_increase : ℝ) (h1 : final_price = 450) (h2 : percent_increase = 0.50) : 
  P + percent_increase * P = final_price → P = 300 :=
by
  sorry

end original_price_l179_179609


namespace sin_45_eq_sqrt2_div_2_l179_179326

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179326


namespace minimum_distance_f_g_l179_179406

open Real

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : Real := log x

theorem minimum_distance_f_g : 
  let M := (f (1/2)), N := (1/2, -log 2) in
  dist M N = (2 + log 2) / 5 * sqrt 5 := 
sorry

end minimum_distance_f_g_l179_179406


namespace triangle_area_solutions_l179_179711

theorem triangle_area_solutions (ABC BDE : ℝ) (k : ℝ) (h₁ : BDE = k^2) : 
  S >= 4 * k^2 ∧ (if S = 4 * k^2 then solutions = 1 else solutions = 2) :=
by
  sorry

end triangle_area_solutions_l179_179711


namespace arrange_in_non_increasing_order_l179_179121

noncomputable def Psi : ℤ := (1/2) * ∑ k in finset.range 503, -4
noncomputable def Omega : ℤ := ∑ k in finset.range 1007, -1
noncomputable def Theta : ℤ := ∑ k in finset.range 504, -2

theorem arrange_in_non_increasing_order :
  Theta ≤ Omega ∧ Omega ≤ Psi :=
begin
  -- Proof to be implemented
  sorry,
end

end arrange_in_non_increasing_order_l179_179121


namespace sin_45_degree_eq_sqrt2_div_2_l179_179154

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179154


namespace pentagon_area_is_13_l179_179737

-- Define the vertices of the pentagon
def v1 := (2, 1 : ℝ)
def v2 := (4, 3 : ℝ)
def v3 := (6, 1 : ℝ)
def v4 := (5, -2 : ℝ)
def v5 := (3, -2 : ℝ)

-- List of vertices
def vertices := [v1, v2, v3, v4, v5]

-- Shoelace Theorem function
def shoelace (vertices : List (ℝ × ℝ)) : ℝ :=
  let n := vertices.length
  (0, vertices ++ [vertices.head!]) |
    (i : Fin n) => 
      let (x1, y1) := vertices.head!
      let (x2, y2) := vertices.nth i.succ!
      .1 + x1 * y2 - y1 * x2

def area_of_pentagon : ℝ :=
  (1 / 2) * | shoelace vertices |

theorem pentagon_area_is_13 : area_of_pentagon = 13 := by
  sorry

end pentagon_area_is_13_l179_179737


namespace range_of_a_l179_179598

theorem range_of_a {a : ℝ} : (∀ x1 x2 : ℝ, 2 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → -x1^2 + 4*a*x1 ≤ -x2^2 + 4*a*x2)
  ∨ (∀ x1 x2 : ℝ, 2 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → -x1^2 + 4*a*x1 ≥ -x2^2 + 4*a*x2) ↔ (a ≤ 1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l179_179598


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179235

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179235


namespace find_base_of_log_l179_179362

theorem find_base_of_log : ∃ b : ℝ, log b 625 = -4/3 ∧ b = 1/125 :=
by
  sorry

end find_base_of_log_l179_179362


namespace area_greater_than_half_l179_179871

-- Definitions for the conditions
variable {A B C M K H : Type} -- Define the points

-- Assume the conditions are satisfied
variable (acute_triangle : ∀ (A B C : Type), acute_angle A B C) -- Assume ABC is an acute-angled triangle
variable (is_median_AM : median A M B C) -- AM is a median
variable (is_angle_bisector_BK : angle_bisector B K A C) -- BK is an angle bisector
variable (is_altitude_CH : altitude C H A B) -- CH is an altitude

-- Area predicate and the final question
variable (area_triangle_formed : area_triangle (intersection_points AM BK CH)) -- Define the area of the formed triangle

-- The final proof statement
theorem area_greater_than_half : area_triangle_formed > 0.499 * area_triangle A B C :=
sorry

end area_greater_than_half_l179_179871


namespace all_points_same_number_l179_179357

-- Given condition: each point P in the plane is assigned a real number p
variable (plane_points : Type) [MetricSpace plane_points]

-- Given condition: the number at the center of the incenter of a triangle is the arithmetic mean of the numbers at its vertices
variable (point_num : plane_points → ℝ)
variable (triangle : plane_points × plane_points × plane_points)
variable (incenter : plane_points × plane_points × plane_points → plane_points)
variable (arithmetic_mean : (ℝ × ℝ × ℝ) → ℝ)

-- Definition to express the given condition
def incenter_property (T : plane_points × plane_points × plane_points) : Prop :=
  point_num (incenter T) = arithmetic_mean (point_num T.1, point_num T.2, point_num T.3)

-- The main theorem to prove that all points in the plane are assigned the same number
theorem all_points_same_number :
  (∀ T : plane_points × plane_points × plane_points, incenter_property T) →
  ∀ x y : plane_points, point_num x = point_num y :=
sorry

end all_points_same_number_l179_179357


namespace log_base_frac_eq_l179_179750

theorem log_base_frac_eq : log (1/5) 25 = -2 := 
by {
  sorry
}

end log_base_frac_eq_l179_179750


namespace number_of_desired_numbers_l179_179832

-- Define a predicate for a four-digit number with the thousands digit 3
def isDesiredNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (n / 1000) % 10 = 3

-- Statement of the theorem
theorem number_of_desired_numbers : 
  ∃ k, k = 1000 ∧ (∀ n, isDesiredNumber n ↔ 3000 ≤ n ∧ n < 4000) := 
by
  -- Proof omitted, using sorry to skip the proof
  sorry

end number_of_desired_numbers_l179_179832


namespace great_dane_weight_l179_179682

def weight_problem (C P G : ℝ) : Prop :=
  (P = 3 * C) ∧ (G = 3 * P + 10) ∧ (C + P + G = 439)

theorem great_dane_weight : ∃ (C P G : ℝ), weight_problem C P G ∧ G = 307 :=
by
  sorry

end great_dane_weight_l179_179682


namespace intersection_point_minimizes_distance_l179_179014

noncomputable def point (α : Type u) [metric_space α] := α

variable {α : Type u} [metric_space α]

def minimize_sum_of_distances {A B C D : point α} (P : point α) :=
∀ (M : point α), dist P A + dist P B + dist P C + dist P D ≤ dist M A + dist M B + dist M C + dist M D

theorem intersection_point_minimizes_distance {A B C D P: point α}
  (h_convex_quadrilateral : convex {A, B, C, D})
  (h_diagonal_intersection : is_intersection_diagonals P A C B D) :
  minimize_sum_of_distances P :=
by
  sorry

end intersection_point_minimizes_distance_l179_179014


namespace factorize_ab_factorize_x_l179_179361

-- Problem 1: Factorization of a^3 b - 2 a^2 b^2 + a b^3
theorem factorize_ab (a b : ℤ) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = a * b * (a - b)^2 := 
by sorry

-- Problem 2: Factorization of (x^2 + 4)^2 - 16 x^2
theorem factorize_x (x : ℤ) : (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 :=
by sorry

end factorize_ab_factorize_x_l179_179361


namespace functional_expression_l179_179418

def f (x : ℝ) : ℝ :=
  2 * |Real.cos x|

theorem functional_expression :
  (∀ x : ℝ, f(-x) = f(x)) ∧ (∀ x : ℝ, 0 ≤ f(x) ∧ f(x) ≤ 2) :=
by
  sorry

end functional_expression_l179_179418


namespace sin_45_degree_eq_sqrt2_div_2_l179_179149

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179149


namespace exponent_of_5_in_30_factorial_l179_179956

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179956


namespace cone_cannot_have_rectangular_projection_l179_179855

def orthographic_projection (solid : Type) : Type := sorry

theorem cone_cannot_have_rectangular_projection :
  (∀ (solid : Type), orthographic_projection solid = Rectangle → solid ≠ Cone) :=
sorry

end cone_cannot_have_rectangular_projection_l179_179855


namespace max_segments_l179_179508

theorem max_segments (n : ℕ) (segments : list (line ℝ)) :
  ∀ (a b c : line ℝ), a ∉ segments ∨ b ∉ segments ∨ c ∉ segments ∨ ¬ are_parallel_to_same_plane [a, b, c] →
  (∀ (x y : line ℝ), x ∈ segments → y ∈ segments → x ≠ y → line_connecting_midpoints_perpendicular x y) →
  n ≤ 2 :=
sorry

noncomputable def line_connecting_midpoints_perpendicular (a b : line ℝ) : Prop :=
∃ l : line ℝ, l.is_midpoint_line a b ∧ l.is_perpendicular a ∧ l.is_perpendicular b

noncomputable def are_parallel_to_same_plane (lines : list (line ℝ)) : Prop :=
∃ (plane : plane ℝ), ∀ l ∈ lines, l.is_parallel plane

end max_segments_l179_179508


namespace sum_of_xyz_l179_179475

theorem sum_of_xyz (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 18) (hxz : x * z = 3) (hyz : y * z = 6) : x + y + z = 10 := 
sorry

end sum_of_xyz_l179_179475


namespace sin_of_cos_eq_l179_179410

theorem sin_of_cos_eq (theta : ℝ) (h1 : real.cos theta = -3/5) (h2 : real.tan theta > 0) :
  real.sin theta = -4/5 :=
begin
  sorry
end

end sin_of_cos_eq_l179_179410


namespace exists_non_monotonic_subinterval_l179_179849

noncomputable def f (x : ℝ) : ℝ := x^2 - (1 / 2) * log x + 1

theorem exists_non_monotonic_subinterval :
  ∀ k : ℝ, 1 ≤ k ∧ k < 3 / 2 ↔ ∃ (a b : ℝ), (a < b) ∧ (a > 0) ∧ (k - 1 < a) ∧ (b < k + 1) ∧ 
  ¬ monotone_on f (set.Icc a b) := 
by
  sorry

end exists_non_monotonic_subinterval_l179_179849


namespace number_of_outfits_l179_179583

theorem number_of_outfits (shirts pants ties : ℕ) (h_shirts : shirts = 7) (h_pants : pants = 5) (h_ties : ties = 4) : 
  let tie_options := ties + 1 in 
  shirts * pants * tie_options = 175 := 
by 
  rw [h_shirts, h_pants, h_ties]
  let tie_options := 5
  norm_num
  sorry

end number_of_outfits_l179_179583


namespace complex_z_1000_l179_179804

open Complex

theorem complex_z_1000 (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (Real.pi * 5 / 180)) :
  z^(1000 : ℕ) + (z^(1000 : ℕ))⁻¹ = 2 * Real.cos (Real.pi * 20 / 180) :=
sorry

end complex_z_1000_l179_179804


namespace finite_triples_with_small_prime_factors_l179_179732

-- Define positive integers and their properties
def is_positive_integer (n : ℕ) : Prop := n > 0

-- Define the property we are interested in
def all_prime_factors_smaller_than (x y : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p ∣ x → p < y

-- Define the main theorem 
theorem finite_triples_with_small_prime_factors :
  ¬ ∃ S : Set (ℕ × ℕ × ℕ), (∀ t ∈ S, let ⟨a, b, c⟩ := t in 
    is_positive_integer a ∧ is_positive_integer b ∧ is_positive_integer c ∧
    all_prime_factors_smaller_than (a.factorial + b.factorial + c.factorial) 2020) ∧ 
    S.Infinite :=
by
  sorry

end finite_triples_with_small_prime_factors_l179_179732


namespace derek_travel_distance_l179_179349

def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem derek_travel_distance : distance (-5, 7) (0, 0) + distance (0, 0) (3, 2) + distance (3, 2) (6, -5) = Real.sqrt 74 + Real.sqrt 13 + Real.sqrt 58 :=
by
  sorry

end derek_travel_distance_l179_179349


namespace find_n_l179_179813

noncomputable def log₂ (x : ℝ) := real.log x / real.log 2
noncomputable def log₃ (x : ℝ) := real.log x / real.log 3

theorem find_n (n : ℤ) :
  (∀ a b x, (2 ^ a = 3) → (3 ^ b = 2) → (∃ x_0, a ^ x_0 + x_0 - b = 0 ∧ (x_0:ℤ) = n)) → n = -1 :=
by
  sorry

end find_n_l179_179813


namespace gasoline_reduction_l179_179660

theorem gasoline_reduction (P Q : ℝ) (h₁ : 1.14 * (P * Q) = 1.20 * P * Q')
  (h₂ : Q' = 0.95 * Q) : (1 - Q' / Q) * 100 = 5 :=
by
  rw [h₂],
  field_simp,
  norm_num,
  exact eq.refl 5

end gasoline_reduction_l179_179660


namespace sin_45_deg_eq_l179_179309

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179309


namespace proof_problem_l179_179422

variables (x y k : ℝ)

-- Definitions and conditions
def circle_eqn := x^2 + y^2 = 4
def line_through_P := ∃ (L : ℝ → ℝ), L 1 = 2

-- Problem statement (I)
def line_L_eqn :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4 ∧
  ∥(A.1 - B.1, A.2 - B.2)∥ = 2 * real.sqrt 3 ∧
  (A = (1,2) ∨ B = (1,2)) ∧
  (line_through_P → (x = 1 ∨ 3 * x - 4 * y + 5 = 0))

-- Problem statement (II)
def trajectory_eqn :=
  ∀ (M N Q : ℝ × ℝ), 
    (M.1^2 + M.2^2 = 4 ∧ N = (0, M.2) ∧ Q = (M.1, 2 * M.2)),
    (Q.1^2 / 4 + Q.2^2 / 16 = 1)

-- Proving both statements
theorem proof_problem : line_L_eqn x y k ∧ trajectory_eqn x y :=
sorry

end proof_problem_l179_179422


namespace sin_45_eq_one_div_sqrt_two_l179_179223

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179223


namespace multiple_6_9_statements_false_l179_179584

theorem multiple_6_9_statements_false
    (a b : ℤ)
    (h₁ : ∃ m : ℤ, a = 6 * m)
    (h₂ : ∃ n : ℤ, b = 9 * n) :
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → ((a + b) % 2 = 0)) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 6 = 0) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 9 = 0) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 9 ≠ 0) :=
by
  sorry

end multiple_6_9_statements_false_l179_179584


namespace sin_45_eq_one_div_sqrt_two_l179_179218

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179218


namespace cistern_length_l179_179683

variable (L : ℝ) (width water_depth total_area : ℝ)

theorem cistern_length
  (h_width : width = 8)
  (h_water_depth : water_depth = 1.5)
  (h_total_area : total_area = 134) :
  11 * L + 24 = total_area → L = 10 :=
by
  intro h_eq
  have h_eq1 : 11 * L = 110 := by
    linarith
  have h_L : L = 10 := by
    linarith
  exact h_L

end cistern_length_l179_179683


namespace T_2023_odd_l179_179775

-- Define T_n as the count of specific quadruples
def T (n : ℕ) : ℕ := 
  ∑ a in finset.range (n + 1), 
  ∑ b in finset.range a, 
  ∑ x in finset.range (n / a + 1), 
  ∑ y in finset.range (n / b + 1), 
  if (a > b) ∧ (n = a * x + b * y) then 1 else 0

-- The main theorem to be proved
theorem T_2023_odd : ∃ k : ℕ, T 2023 = 2 * k + 1 := 
sorry

end T_2023_odd_l179_179775


namespace angle_C_eq_pi_div_3_l179_179478

variables {a b c : ℝ}

def vector_m := (a + c, b - a)
def vector_n := (a - c, b)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem angle_C_eq_pi_div_3
  (h : dot_product vector_m vector_n = 0) :
  ∠C = Real.pi / 3 := 
sorry

end angle_C_eq_pi_div_3_l179_179478


namespace exponent_of_5_in_30_factorial_l179_179950

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179950


namespace sin_45_eq_sqrt_two_over_two_l179_179250

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179250


namespace sin_45_eq_sqrt2_div_2_l179_179290

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179290


namespace exponent_of_5_in_30_factorial_l179_179991

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179991


namespace sin_45_degree_l179_179270

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179270


namespace decimal_to_binary_49_l179_179593

theorem decimal_to_binary_49 : nat.binary 49 = "110001" := by
  sorry

end decimal_to_binary_49_l179_179593


namespace exponent_of_5_in_30_factorial_l179_179900

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179900


namespace sin_45_degree_l179_179271

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179271


namespace James_flowers_per_day_l179_179520

-- Defining the friends' rates based on the problem's conditions
def FriendB_rate := 12
def FriendC_rate := FriendB_rate / 0.7
def FriendD_rate := FriendC_rate * 1.1
def FriendE_rate := FriendD_rate * 0.75
def FriendF_rate := FriendE_rate
def FriendG_rate := FriendF_rate * 0.7
def FriendA_rate := FriendB_rate * 1.15
def James_rate := FriendA_rate * 1.2

-- Prove that James plants 16.56 flowers in a day
theorem James_flowers_per_day : James_rate = 16.56 := by
  sorry

end James_flowers_per_day_l179_179520


namespace sin_45_deg_eq_l179_179301

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179301


namespace slope_of_line_is_3_over_2_x_intercept_of_line_is_minus_8_y_intercept_of_line_is_12_l179_179743

-- Defining the linear equation
def line (x y : ℝ) : Prop :=
  3 * x - 2 * y + 24 = 0

-- Proving the slope
theorem slope_of_line_is_3_over_2 : ∀ x y, line x y → (∃ m b, y = m * x + b ∧ m = 3 / 2) := by
  intros x y h
  use [3 / 2, 12]
  split
  · linarith
  · rfl

-- Proving the x-intercept
theorem x_intercept_of_line_is_minus_8 : ∃ y, line (-8) y ∧ y = 0 := by
  use 0
  split
  · linarith
  · rfl

-- Proving the y-intercept
theorem y_intercept_of_line_is_12 : ∃ x, line x 12 ∧ x = 0 := by
  use 0
  split
  · linarith
  · rfl

end slope_of_line_is_3_over_2_x_intercept_of_line_is_minus_8_y_intercept_of_line_is_12_l179_179743


namespace rebus_puzzle_verified_l179_179733

-- Defining the conditions
def A := 1
def B := 1
def C := 0
def D := 1
def F := 1
def L := 1
def M := 0
def N := 1
def P := 0
def Q := 1
def T := 1
def G := 8
def H := 1
def K := 4
def W := 4
def X := 1

noncomputable def verify_rebus_puzzle : Prop :=
  (A * B * 10 = 110) ∧
  (6 * G / (10 * H + 7) = 4) ∧
  (L + N * 10 = 20) ∧
  (12 - K = 8) ∧
  (101 + 10 * W + X = 142)

-- Lean statement to verify the problem
theorem rebus_puzzle_verified : verify_rebus_puzzle :=
by {
  -- Values are already defined and will be concluded by Lean
  sorry
}

end rebus_puzzle_verified_l179_179733


namespace sin_45_eq_1_div_sqrt_2_l179_179320

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179320


namespace sin_45_eq_1_div_sqrt_2_l179_179318

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179318


namespace cone_volume_surface_area_correct_l179_179074

noncomputable def cone_from_sector_volume_surface_area (r l : ℕ) (sector_degrees : ℕ) :
  (ℕ × ℕ) := by
  let sector_fraction := (Float.ofInt sector_degrees) / 360.0
  let circumference := 2 * Float.pi * (Float.ofInt r)
  let arc_length := sector_fraction * circumference
  let radius_base := arc_length / (2 * Float.pi)
  let height := Float.sqrt ((Float.ofInt l)^2 - radius_base^2)
  let volume := (1.0 / 3.0) * Float.pi * radius_base^2 * height
  let lateral_surface_area := Float.pi * radius_base * (Float.ofInt l)
  let volume_div_pi := volume / Float.pi
  let lateral_surface_area_div_pi := lateral_surface_area / Float.pi
  (Float.toInt volume_div_pi, Float.toInt lateral_surface_area_div_pi)

theorem cone_volume_surface_area_correct :
  cone_from_sector_volume_surface_area 16 16 270 = (384, 192) := by
  sorry

end cone_volume_surface_area_correct_l179_179074


namespace exponent_of_5_in_30_factorial_l179_179986

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179986


namespace average_of_first_16_even_numbers_l179_179044

theorem average_of_first_16_even_numbers : 
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30 + 32) / 16 = 17 := 
by sorry

end average_of_first_16_even_numbers_l179_179044


namespace sin_45_degree_l179_179272

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179272


namespace sin_45_degree_l179_179277

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179277


namespace carpet_width_is_correct_l179_179114

variable (length : ℝ) (coverage_percentage : ℝ) (total_area : ℝ) (carpet_area : ℝ) (width : ℝ)

noncomputable def living_room_floor_area : ℝ := 48
noncomputable def carpet_coverage : ℝ := 0.75
noncomputable def carpet_length : ℝ := 9
noncomputable def carpet_area := carpet_coverage * living_room_floor_area
noncomputable def carpet_width := carpet_area / carpet_length

theorem carpet_width_is_correct : carpet_width = 4 := by
  sorry

end carpet_width_is_correct_l179_179114


namespace ratio_of_inscribed_squares_l179_179340

-- Definitions of the conditions
def right_triangle_sides (a b c : ℕ) : Prop := a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2

def inscribed_square_1 (x : ℚ) : Prop := x = 18 / 7

def inscribed_square_2 (y : ℚ) : Prop := y = 32 / 7

-- Statement of the problem
theorem ratio_of_inscribed_squares (x y : ℚ) : right_triangle_sides 6 8 10 ∧ inscribed_square_1 x ∧ inscribed_square_2 y → (x / y) = 9 / 16 :=
by
  sorry

end ratio_of_inscribed_squares_l179_179340


namespace x0_range_l179_179820

noncomputable def f (x : ℝ) := (1 / 2) ^ x - Real.log x

theorem x0_range (x0 : ℝ) (h : f x0 > 1 / 2) : 0 < x0 ∧ x0 < 1 :=
by
  sorry

end x0_range_l179_179820


namespace sin_45_deg_l179_179199

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179199


namespace volume_of_prism_l179_179028

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 20) (h3 : c * a = 12) (h4 : a + b + c = 11) :
  a * b * c = 12 * Real.sqrt 15 :=
by
  sorry

end volume_of_prism_l179_179028


namespace min_value_of_S_l179_179610

theorem min_value_of_S (n : ℕ) (S : ℕ) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 6 ∧ n = ⌈1994 / 6⌉) → (∃ s : ℕ, s = 334) → (∃ p1 p2 : ℕ, p1 > 0 ∧ p2 > 0 ∧ p1 = p2) := sorry

end min_value_of_S_l179_179610


namespace find_AB_l179_179874

def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem find_AB 
  (A B C : ℝ) 
  (hC : A = 6) 
  (hT : ∃ θ, tan θ = 3 / 4) 
  (hRight : right_triangle 6 (4.5 : ℝ) B) 
  : B = 7.5 :=
by 
  sorry

end find_AB_l179_179874


namespace pretty_12_sum_div_12_l179_179142

def is_pretty_12 (n : ℕ) : Prop :=
  n % 12 = 0 ∧ (∃ d : ℕ, d = 12 ∧ ∀ m, 1 ≤ m ∧ m ≤ n → n % m = 0 → ∑ m = 12)

theorem pretty_12_sum_div_12 :
  let S := ∑ n in finset.filter (λ n, is_pretty_12 n ∧ n < 1000) (finset.range 1000), n
  S / 12 = 55.5 :=
by
  sorry

end pretty_12_sum_div_12_l179_179142


namespace exponent_of_5_in_30_factorial_l179_179965

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179965


namespace sum_of_coordinates_l179_179005

theorem sum_of_coordinates : 
  let points := {p : ℝ × ℝ | (abs (p.2 - 20) = 7) ∧ (sqrt ((p.1 - 10) ^ 2 + (p.2 - 20) ^ 2) = 10)} in
  ∑ p in points, (p.1 + p.2) = 120 :=
sorry

end sum_of_coordinates_l179_179005


namespace no_2000_digit_perfect_square_with_1999_digits_of_5_l179_179746

theorem no_2000_digit_perfect_square_with_1999_digits_of_5 :
  ¬ (∃ n : ℕ,
      (Nat.digits 10 n).length = 2000 ∧
      ∃ k : ℕ, n = k * k ∧
      (Nat.digits 10 n).count 5 ≥ 1999) :=
sorry

end no_2000_digit_perfect_square_with_1999_digits_of_5_l179_179746


namespace senior_junior_ratio_l179_179133

variable (S J : ℕ) (k : ℕ)

theorem senior_junior_ratio (h1 : S = k * J) 
                           (h2 : (1/8 : ℚ) * S + (3/4 : ℚ) * J = (1/3 : ℚ) * (S + J)) : 
                           k = 2 :=
by
  sorry

end senior_junior_ratio_l179_179133


namespace symmetric_points_origin_a_plus_b_l179_179414

theorem symmetric_points_origin_a_plus_b (a b : ℤ) 
  (h1 : a + 3 * b = 5)
  (h2 : a + 2 * b = -3) :
  a + b = -11 :=
by
  sorry

end symmetric_points_origin_a_plus_b_l179_179414


namespace exponent_of_5_in_30_factorial_l179_179919

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179919


namespace number_of_committees_l179_179773

theorem number_of_committees (n k : ℕ) (h_n : n = 8) (h_k : k = 3) : nat.choose n k = 56 :=
by
  simp [h_n, h_k]
  -- here your solution proof will go
  sorry

end number_of_committees_l179_179773


namespace evaluate_expr_l179_179359

def x := 2
def y := -1
def z := 3
def expr := 2 * x^2 + y^2 - z^2 + 3 * x * y

theorem evaluate_expr : expr = -6 :=
by sorry

end evaluate_expr_l179_179359


namespace sin_45_degree_eq_sqrt2_div_2_l179_179144

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179144


namespace exponent_of_5_in_30_factorial_l179_179896

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179896


namespace trains_meet_in_approximately_21_67_seconds_l179_179018

-- Define the lengths of the trains in meters
def length_train_a : ℝ := 150
def length_train_b : ℝ := 250

-- Define the initial distance apart in meters
def initial_distance_apart : ℝ := 900

-- Define the speeds of the trains in meters per second
def speed_train_a_kmh : ℝ := 120
def speed_train_b_kmh : ℝ := 96

def speed_train_a : ℝ := speed_train_a_kmh * (1000 / 3600)
def speed_train_b : ℝ := speed_train_b_kmh * (1000 / 3600)

-- Calculate the relative speed
def relative_speed : ℝ := speed_train_a + speed_train_b

-- Calculate the total distance to be covered
def total_distance : ℝ := length_train_a + length_train_b + initial_distance_apart

-- Calculate the time to meet
def time_to_meet : ℝ := total_distance / relative_speed

theorem trains_meet_in_approximately_21_67_seconds :
  abs (time_to_meet - 21.67) < 0.01 :=
sorry

end trains_meet_in_approximately_21_67_seconds_l179_179018


namespace exponent_of_5_in_30_factorial_l179_179975

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179975


namespace hyperbola_eccentricity_correct_l179_179409

noncomputable def hyperbola_eccentricity (a b : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (ha : a > 0) (hb : b > 0)
  (hP_on_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (hPF1PF2_sum : abs (dist P F1) + abs (dist P F2) = 3 * b)
  (hPF1PF2_prod : abs (dist P F1) * abs (dist P F2) = (9 / 4) * a * b) :
  ℝ := (5 / 3)

theorem hyperbola_eccentricity_correct :
  ∀ (a b : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ),
    a > 0 →
    b > 0 →
    (P.1^2 / a^2) - (P.2^2 / b^2) = 1 →
    abs (dist P F1) + abs (dist P F2) = 3 * b →
    abs (dist P F1) * abs (dist P F2) = (9 / 4) * a * b →
    hyperbola_eccentricity a b P F1 F2 = (5 / 3) :=
by
  intros a b P F1 F2 ha hb hP_on_hyperbola hPF1PF2_sum hPF1PF2_prod
  simp only [hyperbola_eccentricity]
  sorry

end hyperbola_eccentricity_correct_l179_179409


namespace sin_45_degree_l179_179164

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179164


namespace stratified_sampling_admin_staff_count_l179_179680

theorem stratified_sampling_admin_staff_count
  (total_staff : ℕ)
  (admin_staff : ℕ)
  (sample_size : ℕ)
  (h_total : total_staff = 160)
  (h_admin : admin_staff = 32)
  (h_sample : sample_size = 20) :
  admin_staff * sample_size / total_staff = 4 :=
by
  sorry

end stratified_sampling_admin_staff_count_l179_179680


namespace exponent_of_5_in_30_factorial_l179_179987

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179987


namespace mary_number_l179_179552

-- Definitions for conditions
def has_factor_150 (m : ℕ) : Prop := 150 ∣ m
def is_multiple_of_45 (m : ℕ) : Prop := 45 ∣ m
def in_range (m : ℕ) : Prop := 1000 < m ∧ m < 3000

-- Theorem stating that Mary's number is one of {1350, 1800, 2250, 2700} given the conditions
theorem mary_number 
  (m : ℕ) 
  (h1 : has_factor_150 m)
  (h2 : is_multiple_of_45 m)
  (h3 : in_range m) :
  m = 1350 ∨ m = 1800 ∨ m = 2250 ∨ m = 2700 :=
sorry

end mary_number_l179_179552


namespace sin_45_degree_l179_179273

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179273


namespace find_a_l179_179850

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a^2 - 1

theorem find_a (a : ℝ) (h : ∀ x ∈ (Set.Icc 1 2), f x a ≤ 16 ∧ ∃ y ∈ (Set.Icc 1 2), f y a = 16) : a = 3 ∨ a = -3 :=
by
  sorry

end find_a_l179_179850


namespace exponent_of_5_in_30_factorial_l179_179934

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179934


namespace problem_1_problem_2_l179_179395

-- Definitions of conditions
variables {a b : ℝ}
axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_sum : a + b = 1

-- The statements to prove
theorem problem_1 : 
  (1 / (a^2)) + (1 / (b^2)) ≥ 8 := 
sorry

theorem problem_2 : 
  (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 := 
sorry

end problem_1_problem_2_l179_179395


namespace intersection_point_ratios_l179_179564

variables {α β : ℝ}
variables {A B C D K L M N P : Type*}
variables [convex_quadrilateral A B C D]
variables (h1 : segment_ratio A B K α)
variables (h2 : segment_ratio D C M α)
variables (h3 : segment_ratio B C L β)
variables (h4 : segment_ratio A D N β)
variables (h5 : intersection_point K M L N P)

-- The theorem states that NP:PL = α and KP:PM = β given the above conditions
theorem intersection_point_ratios
  (h1 : AK / KB = α)
  (h2 : DM / MC = α)
  (h3 : BL / LC = β)
  (h4 : AN / ND = β)
  (h5 : ∃ P, is_intersection_point K M L N P) :
  NP / PL = α ∧ KP / PM = β :=
sorry

end intersection_point_ratios_l179_179564


namespace find_integer_solutions_xy_l179_179364

theorem find_integer_solutions_xy :
  ∀ (x y : ℕ), (x * y = x + y + 3) → (x, y) = (2, 5) ∨ (x, y) = (5, 2) ∨ (x, y) = (3, 3) := by
  intros x y h
  sorry

end find_integer_solutions_xy_l179_179364


namespace sin_45_degree_eq_sqrt2_div_2_l179_179145

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179145


namespace max_value_sin_cos_combination_l179_179377

theorem max_value_sin_cos_combination :
  ∀ x : ℝ, (5 * Real.sin x + 12 * Real.cos x) ≤ 13 :=
by
  intro x
  sorry

end max_value_sin_cos_combination_l179_179377


namespace arrange_in_order_l179_179118

noncomputable def Psi : ℤ := 1 / 2 * (1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + List.sum (List.of_fn (λ n => if n % 4 < 2 then 2 * n + 5 else -2 * n - 5) 503))
noncomputable def Omega : ℤ := List.sum (List.of_fn (λ n => if n % 2 == 0 then n + 1 else -n - 1) 1007)
noncomputable def Theta : ℤ := List.sum (List.range (2015 / 2) |>.map (λ n => if n % 2 == 0 then 2 * n + 1 else -2 * n - 1))

theorem arrange_in_order : Theta = -1008 ∧ Omega = -1007 ∧ Psi = -1006 → Theta ≤ Omega ∧ Omega ≤ Psi := sorry

end arrange_in_order_l179_179118


namespace intersection_of_lines_l179_179352

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 
  (5 * x - 3 * y = 20) ∧ (3 * x + 4 * y = 6) ∧ 
  x = 98 / 29 ∧ 
  y = 87 / 58 :=
by 
  sorry

end intersection_of_lines_l179_179352


namespace initial_tax_rate_is_46_l179_179487

variable (annual_income : ℝ)
variable (lowered_tax_rate : ℝ)
variable (savings : ℝ)

-- Conditions
def initial_income_tax (P : ℝ) : ℝ :=
  P / 100 * annual_income

def new_income_tax : ℝ :=
  lowered_tax_rate / 100 * annual_income

def tax_savings (P : ℝ) : ℝ :=
  initial_income_tax annual_income P - new_income_tax annual_income

-- The Lean 4 statement to prove
theorem initial_tax_rate_is_46 :
  (annual_income = 36000) → 
  (lowered_tax_rate = 32) → 
  (savings = 5040) → 
  tax_savings annual_income lowered_tax_rate savings P = 46 :=
sorry

end initial_tax_rate_is_46_l179_179487


namespace sin_45_deg_l179_179187

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179187


namespace fraction_identity_l179_179796

theorem fraction_identity (a b c : ℕ) (h : (a : ℚ) / (36 - a) + (b : ℚ) / (48 - b) + (c : ℚ) / (72 - c) = 9) : 
  4 / (36 - a) + 6 / (48 - b) + 9 / (72 - c) = 13 / 3 := 
by 
  sorry

end fraction_identity_l179_179796


namespace sin_45_degree_eq_sqrt2_div_2_l179_179146

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179146


namespace sin_45_eq_sqrt2_div_2_l179_179330

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179330


namespace a_10_eq_505_l179_179789

-- The sequence definition
def a (n : ℕ) : ℕ :=
  let start := (n * (n - 1)) / 2 + 1
  List.sum (List.range' start n)

-- Theorem that the 10th term of the sequence is 505
theorem a_10_eq_505 : a 10 = 505 := 
by
  sorry

end a_10_eq_505_l179_179789


namespace magnet_to_stuffed_ratio_l179_179135

theorem magnet_to_stuffed_ratio (magnet_cost stuffed_animal_cost : ℕ) (h1 : magnet_cost = 3) (h2 : stuffed_animal_cost = 6) :
  (magnet_cost : ℚ) / (2 * stuffed_animal_cost) = 1 / 4 :=
by
  -- Proof is skipped using sorry
  sorry

end magnet_to_stuffed_ratio_l179_179135


namespace regular_polygon_sides_l179_179477

theorem regular_polygon_sides (n : ℕ) (h₁ : n ≥ 3) (h₂ : 120 = 180 * (n - 2) / n) : n = 6 :=
by
  sorry

end regular_polygon_sides_l179_179477


namespace sin_45_degree_l179_179158

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179158


namespace sarah_trip_distance_l179_179570

theorem sarah_trip_distance (y : ℝ) 
    (h1 : y / 4 + 25 + y / 6 = y) : y = 300 / 7 := 
begin
  sorry
end

end sarah_trip_distance_l179_179570


namespace output_of_code_snippet_is_six_l179_179029

-- Define the variables and the condition
def a : ℕ := 3
def y : ℕ := if a < 10 then 2 * a else a * a 

-- The statement to be proved
theorem output_of_code_snippet_is_six :
  y = 6 :=
by
  sorry

end output_of_code_snippet_is_six_l179_179029


namespace bonferroni_inequalities_l179_179638

variable {n k : ℕ} {A : ℕ → Set ω} {P : Set ω → ℝ}
variable (S : ℕ → ℝ) (P_union : ℝ)
variable (h1 : k ≥ 1) (h2 : 2 * k ≤ n) (h3 : P_union = P (⋃ i in Finset.range n, A i))

theorem bonferroni_inequalities :
  (S 1 - S 2 + ... - S (2 * k)) ≤ P_union ∧ P_union ≤ (S 1 - S 2 + ... + S (2 * k - 1)) :=
sorry

end bonferroni_inequalities_l179_179638


namespace sin_45_eq_1_div_sqrt_2_l179_179314

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179314


namespace sin_45_eq_sqrt2_div_2_l179_179211

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179211


namespace RB_eq_RC_l179_179493

-- Definitions based on the given conditions
variable {A B C I M_A M_B M_C H_A H_B H_C : Point}
variable {l_b l_b' l_c l_c' : Line}
variable {P_B Q_B P_C Q_C R : Point}
variable {circumcircle : Triangle → Circle}

-- Given conditions
axiom triangle_ABC : Triangle := ⟨A, B, C⟩
axiom incenter_I : Incenter triangle_ABC = I
axiom midpoint_M_A : Midpoint B C = M_A
axiom midpoint_M_B : Midpoint C A = M_B
axiom midpoint_M_C : Midpoint A B = M_C
axiom feet_H_A : FootPerpendicular A B C = H_A
axiom feet_H_B : FootPerpendicular B C A = H_B
axiom feet_H_C : FootPerpendicular C A B = H_C
axiom tangent_l_b : TangentLine (circumcircle triangle_ABC) B = l_b
axiom reflected_l_b' : ReflectOverLine l_b (LineThrough B I) = l_b'
axiom intersection_P_B : IntersectLine (LineThrough M_A M_C) l_b = P_B
axiom intersection_Q_B : IntersectLine (LineThrough H_A H_C) l_b' = Q_B
axiom tangent_l_c : TangentLine (circumcircle triangle_ABC) C = l_c
axiom reflected_l_c' : ReflectOverLine l_c (LineThrough C I) = l_c'
axiom intersection_P_C : IntersectLine (LineThrough M_B M_C) l_c = P_C
axiom intersection_Q_C : IntersectLine (LineThrough H_B H_C) l_c' = Q_C
axiom intersection_R : IntersectLine (LineThrough P_B Q_B) (LineThrough P_C Q_C) = R

-- To prove
theorem RB_eq_RC : dist R B = dist R C := sorry

end RB_eq_RC_l179_179493


namespace largest_fraction_l179_179031

theorem largest_fraction
: (1 : ℚ / 5 * 2) < (1 : ℚ / 7 * 3) ∧
  (1 : ℚ / 9 * 4) < (1 : ℚ / 7 * 3) ∧
  (1 : ℚ / 11 * 5) < (1 : ℚ / 7 * 3) ∧
  (1 : ℚ / 13 * 6) < (1 : ℚ / 7 * 3):=
by
  sorry

end largest_fraction_l179_179031


namespace magic_triangle_max_sum_l179_179497

theorem magic_triangle_max_sum :
  ∀ (a b c d e f : ℕ), {11, 12, 13, 14, 15, 16} = {a, b, c, d, e, f} →
  (a + b + c) = (c + d + e) ∧ (c + d + e) = (e + f + a) →
  (a + b + c) ≤ 42 :=
begin
  sorry
end

end magic_triangle_max_sum_l179_179497


namespace center_square_is_six_l179_179862

def unique_digits_in_grid (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 → grid i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
          ∀ k l, (i ≠ k ∨ j ≠ l) → grid i j ≠ grid k l

def consecutive_adjacent (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 →
    ∀ ni nj, (|i - ni| = 1 ∧ j = nj) ∨ (|j - nj| = 1 ∧ i = ni) →
      (grid i j = grid ni nj + 1 ∨ grid i j = grid ni nj - 1)

def product_of_corners (grid : ℕ → ℕ → ℕ) : Prop :=
  grid 1 1 * grid 1 3 * grid 3 1 * grid 3 3 = 945

def is_center_six (grid : ℕ → ℕ → ℕ) : Prop :=
  grid 2 2 = 6

theorem center_square_is_six (grid : ℕ → ℕ → ℕ) :
  unique_digits_in_grid grid ∧ 
  consecutive_adjacent grid ∧ 
  product_of_corners grid → 
  is_center_six grid :=
by sorry

end center_square_is_six_l179_179862


namespace arrange_numbers_l179_179124

noncomputable def ψ : ℤ := (1 / 2) * (Finset.sum (Finset.range 1006) (λ n, if n % 4 = 0 ∨ n % 4 = 1 then n + 1 else -(n + 1)))

noncomputable def ω : ℤ := Finset.sum (Finset.range 1007) (λ n, if n % 2 = 0 then (2 * n + 1) else -(2 * n + 1))

noncomputable def θ : ℤ := Finset.sum (Finset.range 1008) (λ n, if n % 2 = 0 then (2 * n + 1) else (-(2 * n + 1)))

theorem arrange_numbers :
  θ <= ω ∧ ω <= ψ :=
  sorry

end arrange_numbers_l179_179124


namespace average_price_of_kept_fruits_l179_179132

def price_apple : ℕ := 40
def price_orange : ℕ := 60
def total_fruits : ℕ := 10
def average_initial_price : ℕ := 54
def fruits_put_back : ℕ := 6

theorem average_price_of_kept_fruits (A O : ℕ) 
  (h1 : A + O = total_fruits) 
  (h2 : price_apple * A + price_orange * O = average_initial_price * total_fruits) 
  (h3 : O >= fruits_put_back) : 
  let kept_fruits := A + (O - fruits_put_back) in
  let total_kept_price := price_apple * A + price_orange * (O - fruits_put_back) in
  total_kept_price / kept_fruits = 45 :=
sorry

end average_price_of_kept_fruits_l179_179132


namespace sin_45_degree_l179_179175

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179175


namespace sin_45_eq_sqrt2_div_2_l179_179295

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179295


namespace sqrt_eq_l179_179839

theorem sqrt_eq (x : ℝ) (h : real.sqrt 32 + real.sqrt x = real.sqrt 50) : x = 2 := 
  sorry

end sqrt_eq_l179_179839


namespace probability_both_girls_l179_179072

/-- A club has 12 members, 7 boys and 5 girls, and 2 members are chosen at random.
    Prove that the probability that both members chosen are girls is 5/33. -/
theorem probability_both_girls (total_members boys girls chosen_members : ℕ)
  (h_total: total_members = 12) (h_boys: boys = 7) (h_girls: girls = 5)
  (h_chosen: chosen_members = 2) :
  let total_ways := Nat.choose total_members chosen_members,
      girl_ways := Nat.choose girls chosen_members,
      probability := girl_ways / total_ways in
  probability = 5 / 33 :=
by
  sorry

end probability_both_girls_l179_179072


namespace sin_45_eq_sqrt2_div_2_l179_179203

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179203


namespace mangoes_total_l179_179666

def dozen := 12

def mangoes_in_one_box (dozens_per_box : ℕ) : ℕ :=
  dozens_per_box * dozen

def total_mangoes (boxes : ℕ) (dozens_per_box : ℕ) : ℕ :=
  boxes * mangoes_in_one_box dozens_per_box

theorem mangoes_total (dozens_per_box : ℕ) (boxes : ℕ) :
  dozens_per_box = 10 → boxes = 36 → total_mangoes boxes dozens_per_box = 4320 :=
by
  intros h1 h2
  simp [total_mangoes, mangoes_in_one_box, dozen, h1, h2]
  sorry

end mangoes_total_l179_179666


namespace ap_bq_cr_inequality_l179_179415

/-- Given points P, Q, R on the sides of triangle ABC with a specified ratio,
show that the sum of the fourth powers of the segments is at least a fraction 
of the sum of the fourth powers of the sides -/
theorem ap_bq_cr_inequality {A B C P Q R : ℝ} (a b c : ℝ) (λ : ℝ) 
  (hP : P ∈ open_segment ℝ B C)
  (hQ : Q ∈ open_segment ℝ C A)
  (hR : R ∈ open_segment ℝ A B)
  (h1 : (P - B)/(C - P) = λ)
  (h2 : (Q - C)/(A - Q) = λ)
  (h3 : (R - A)/(B - R) = λ) :
  (A - P)^4 + (B - Q)^4 + (C - R)^4 ≥ (9 / 16) * (a^4 + b^4 + c^4) :=
by
  sorry

end ap_bq_cr_inequality_l179_179415


namespace arrange_in_order_l179_179119

noncomputable def Psi : ℤ := 1 / 2 * (1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + List.sum (List.of_fn (λ n => if n % 4 < 2 then 2 * n + 5 else -2 * n - 5) 503))
noncomputable def Omega : ℤ := List.sum (List.of_fn (λ n => if n % 2 == 0 then n + 1 else -n - 1) 1007)
noncomputable def Theta : ℤ := List.sum (List.range (2015 / 2) |>.map (λ n => if n % 2 == 0 then 2 * n + 1 else -2 * n - 1))

theorem arrange_in_order : Theta = -1008 ∧ Omega = -1007 ∧ Psi = -1006 → Theta ≤ Omega ∧ Omega ≤ Psi := sorry

end arrange_in_order_l179_179119


namespace extremum_at_neg3_l179_179438

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 + 5 * x^2 + a * x
def f_deriv (x : ℝ) : ℝ := 3 * x^2 + 10 * x + a

theorem extremum_at_neg3 (h : f_deriv a (-3) = 0) : a = 3 := 
  by
  sorry

end extremum_at_neg3_l179_179438


namespace solve_for_x_l179_179579

theorem solve_for_x : 
  ∃ x : ℝ, 7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + 1 / 2 ∧ x = -5.375 :=
by
  sorry

end solve_for_x_l179_179579


namespace digit_150_in_sequence_l179_179846

theorem digit_150_in_sequence : 
  let seq := List.join (List.map repr (List.range' 100 (-1) 50)) -- sequence from 100 to 50 in reversed order
  seq.get! (150 - 1) = '2' := -- Lean indices start at 0
sorry

end digit_150_in_sequence_l179_179846


namespace hotel_cost_of_operations_l179_179690

theorem hotel_cost_of_operations : 
  ∃ (C : ℝ), (3 / 4) * C + 25 = C ∧ C = 100 :=
by
  existsi 100
  split
  { calc (3 / 4) * 100 + 25 = 75 + 25 : by ring
                         ... = 100 : by ring }
  { refl }

end hotel_cost_of_operations_l179_179690


namespace sin_45_eq_sqrt_two_over_two_l179_179247

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179247


namespace total_time_l179_179651

theorem total_time {minutes seconds : ℕ} (hmin : minutes = 3450) (hsec : seconds = 7523) :
  ∃ h m s : ℕ, h = 59 ∧ m = 35 ∧ s = 23 :=
by
  sorry

end total_time_l179_179651


namespace sin_45_deg_eq_l179_179298

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179298


namespace exponent_of_5_in_30_factorial_l179_179968

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179968


namespace sum_f_eq_338_l179_179672

def f (x : ℝ) : ℝ :=
  if (x % 6) < -1 then -(x + 2) ^ 2 
  else if (x % 6) < 3 then x % 6 
  else f (x - 6)

theorem sum_f_eq_338 : (∑ i in finset.range 2012, f (i + 1)) = 338 := by
  sorry

end sum_f_eq_338_l179_179672


namespace slopes_of_line_intersecting_ellipse_l179_179081

noncomputable def possible_slopes : Set ℝ := {m : ℝ | m ≤ -1/Real.sqrt 20 ∨ m ≥ 1/Real.sqrt 20}

theorem slopes_of_line_intersecting_ellipse (m : ℝ) (h : ∃ x y, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) : 
  m ∈ possible_slopes :=
sorry

end slopes_of_line_intersecting_ellipse_l179_179081


namespace largest_fraction_l179_179033

theorem largest_fraction :
  (∀ (x ∈ { (2 / 5), (3 / 7), (4 / 9), (5 / 11), (6 / 13) }), x ≤ (6 / 13)) ∧ 
  (6 / 13) ∈ { (2 / 5), (3 / 7), (4 / 9), (5 / 11), (6 / 13) } :=
by { sorry }

end largest_fraction_l179_179033


namespace problem_statement_l179_179707

def has_no_medium_divisor (n : ℕ) : Prop :=
  ∀ k, 1 < k ∧ k < 30 → ¬ k ∣ n

def valid_numbers_count : ℕ :=
  Nat.card {n | n ≤ 60 ∧ has_no_medium_divisor n}

theorem problem_statement : valid_numbers_count = 9 := 
by
  sorry

end problem_statement_l179_179707


namespace exponent_of_5_in_30_factorial_l179_179985

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179985


namespace total_company_pay_monthly_l179_179526

-- Define the given conditions
def hours_josh_works_daily : ℕ := 8
def days_josh_works_weekly : ℕ := 5
def weeks_josh_works_monthly : ℕ := 4
def hourly_rate_josh : ℕ := 9

-- Define Carl's working hours and rate based on the conditions
def hours_carl_works_daily : ℕ := hours_josh_works_daily - 2
def hourly_rate_carl : ℕ := hourly_rate_josh / 2

-- Calculate total hours worked monthly by Josh and Carl
def total_hours_josh_monthly : ℕ := hours_josh_works_daily * days_josh_works_weekly * weeks_josh_works_monthly
def total_hours_carl_monthly : ℕ := hours_carl_works_daily * days_josh_works_weekly * weeks_josh_works_monthly

-- Calculate monthly pay for Josh and Carl
def monthly_pay_josh : ℕ := total_hours_josh_monthly * hourly_rate_josh
def monthly_pay_carl : ℕ := total_hours_carl_monthly * hourly_rate_carl

-- Theorem to prove the total pay for both Josh and Carl in one month
theorem total_company_pay_monthly : monthly_pay_josh + monthly_pay_carl = 1980 := by
  sorry

end total_company_pay_monthly_l179_179526


namespace range_of_M_l179_179471

theorem range_of_M (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x + y + z = 30) (h2 : 3 * x + y - z = 50) :
  120 ≤ 5 * x + 4 * y + 2 * z ∧ 5 * x + 4 * y + 2 * z ≤ 130 :=
by
  -- We would start the proof here by using the given constraints
  sorry

end range_of_M_l179_179471


namespace applicantA_stability_l179_179710

noncomputable def applicantA_probs : ℕ → ℚ := λ n, 
  match n with
  | 1 => 1 / 5
  | 2 => 3 / 5
  | 3 => 1 / 5
  | _ => 0
  end

noncomputable def applicantB_probs : ℕ → ℚ := λ n, 
  match n with
  | 0 => 1 / 27
  | 1 => 2 / 9
  | 2 => 4 / 9
  | 3 => 8 / 27
  | _ => 0
  end

theorem applicantA_stability (E_X : ℚ) (D_X : ℚ) (E_Y : ℚ) (D_Y : ℚ) :
  E_X = 2 → E_Y = 2 → D_X = 2 / 5 → D_Y = 2 / 3 → D_X < D_Y → 
  "Applicant A has a higher probability of passing the interview due to greater stability in performance." := 
  by
  intro h1 h2 h3 h4 h5
  sorry

end applicantA_stability_l179_179710


namespace no_polynomial_transforms_set_l179_179720

theorem no_polynomial_transforms_set :
  ∀ (f : ℤ → ℤ),
    (∀ x y, ∃ k, f x - f y = k * (x - y)) →
    ¬ (∃ (g : ℤ → ℤ), (∀ x ∈ {2, 4, 7}, g x = f x) ∧
      g 2 ∈ {2, 6, 9} ∧ g 4 ∈ {2, 6, 9} ∧ g 7 ∈ {2, 6, 9}) :=
by
  sorry

end no_polynomial_transforms_set_l179_179720


namespace eval_powers_of_i_l179_179358

theorem eval_powers_of_i : (λ (i : ℂ), i^23 + i^47) i = -2 * i :=
by
  sorry

end eval_powers_of_i_l179_179358


namespace percentage_square_area_in_rectangle_l179_179097

-- Define the given conditions
def square_side_length : ℝ := s
def rectangle_width := (3 * s) / 2
def rectangle_length := (3 / 2) * rectangle_width

-- Define areas
def square_area := s^2
def rectangle_area := (rectangle_length * rectangle_width)

-- Define the ratio and percentage calculation
def area_ratio := square_area / rectangle_area
def percentage_occupied := area_ratio * 100

-- Statement to prove
theorem percentage_square_area_in_rectangle :
  percentage_occupied = (8 / 27) * 100 := 
sorry

end percentage_square_area_in_rectangle_l179_179097


namespace arrange_in_non_increasing_order_l179_179123

noncomputable def Psi : ℤ := (1/2) * ∑ k in finset.range 503, -4
noncomputable def Omega : ℤ := ∑ k in finset.range 1007, -1
noncomputable def Theta : ℤ := ∑ k in finset.range 504, -2

theorem arrange_in_non_increasing_order :
  Theta ≤ Omega ∧ Omega ≤ Psi :=
begin
  -- Proof to be implemented
  sorry,
end

end arrange_in_non_increasing_order_l179_179123


namespace no_equilateral_integer_coords_l179_179723

theorem no_equilateral_integer_coords (x1 y1 x2 y2 x3 y3 : ℤ) : 
  ¬ ((x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
     (x1 ≠ x3 ∨ y1 ≠ y3) ∧
     (x2 ≠ x3 ∨ y2 ≠ y3) ∧ 
     ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x1) ^ 2 + (y3 - y1) ^ 2 ∧ 
      (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x2) ^ 2 + (y3 - y2) ^ 2)) :=
by
  sorry

end no_equilateral_integer_coords_l179_179723


namespace nabla_exponent_l179_179735

def nabla (x y : ℕ) := x^3 - 2 * y

theorem nabla_exponent (a b c d : ℕ) :
  nabla (a^ (nabla b c)) (d^ (nabla d c)) = a ^ (nabla (nabla b c) (nabla b c)) - 2 * (d ^ (nabla d c)) :=
sorry

example : (5 ^ nabla 7 4) \nabla (2 ^ nabla 6 9) = 5 ^ 1005 - 2 ^ 199 := 
by {
  let a := 5,
  let b := 7,
  let c := 4,
  let d := 2,
  let e := 6,
  have h1 : nabla b c = 335,
  have h2 : nabla e 9 = 198,
  rw [←h1, ←h2],
  rw [a^335, d^198],
  exact nat.eq_phenomenal,
}

end nabla_exponent_l179_179735


namespace dividing_condition_l179_179757

variable (a b : ℝ)

def P (X : ℝ) := a * X^4 + b * X^2 + 1

theorem dividing_condition :
  (∀ X, (X - 1) ^ 2 ∣ P X) ↔ (a = 1 ∧ b = -2) :=
sorry

end dividing_condition_l179_179757


namespace exponent_of_5_in_30_factorial_l179_179984

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179984


namespace increase_in_selling_price_l179_179572

theorem increase_in_selling_price (current_profit_margin : ℝ) (current_selling_price_percentage : ℝ) : 
  current_profit_margin = 0.25 → current_selling_price_percentage = 1.25 → 
  ∃ (increase_in_price_percentage : ℝ), increase_in_price_percentage = 0.12 :=
by
  intros h1 h2
  use 0.12
  sorry

end increase_in_selling_price_l179_179572


namespace average_age_after_person_leaves_l179_179589

theorem average_age_after_person_leaves 
  (initial_people : ℕ) 
  (initial_average_age : ℕ) 
  (person_leaving_age : ℕ) 
  (remaining_people : ℕ) 
  (new_average_age : ℝ)
  (h1 : initial_people = 7) 
  (h2 : initial_average_age = 32) 
  (h3 : person_leaving_age = 22) 
  (h4 : remaining_people = 6) :
  new_average_age = 34 := 
by 
  sorry

end average_age_after_person_leaves_l179_179589


namespace positive_difference_sum_pos_evens_and_odds_l179_179645

def sum_first_n_positive_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_positive_odds (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_sum_pos_evens_and_odds :
  let sum_evens := sum_first_n_positive_evens 20
  let sum_odds := sum_first_n_positive_odds 15
  (sum_evens > sum_odds) ∧ (sum_evens - sum_odds = 195) :=
by
  let sum_evens := sum_first_n_positive_evens 20
  let sum_odds := sum_first_n_positive_odds 15
  have h1 : sum_evens = 420,
  { sorry }
  have h2 : sum_odds = 225,
  { sorry }
  have h3 : 420 - 225 = 195,
  { sorry }
  show (sum_evens > sum_odds) ∧ (sum_evens - sum_odds = 195),
  by
    simp [h1, h2, h3, sum_first_n_positive_evens, sum_first_n_positive_odds]
    exact ⟨by norm_num, by norm_num⟩


end positive_difference_sum_pos_evens_and_odds_l179_179645


namespace inequalities_cannot_hold_simultaneously_l179_179790

theorem inequalities_cannot_hold_simultaneously (n : ℕ) (a : fin n → ℝ) :
  (∀ i, 0 < a i ∧ a i < 1) →
  ¬ (∀ i, a i * (1 - a ((i + 1) % n)) > 1/4) := 
by 
  intros h1 h2 
  sorry

end inequalities_cannot_hold_simultaneously_l179_179790


namespace length_CD_l179_179604

-- Definitions of the edge lengths provided in the problem
def edge_lengths : Set ℕ := {7, 13, 18, 27, 36, 41}

-- Assumption that AB = 41
def AB := 41
def BC : ℕ := 13
def AC : ℕ := 36

-- Main theorem to prove that CD = 13
theorem length_CD (AB BC AC : ℕ) (edges : Set ℕ) (hAB : AB = 41) (hedges : edges = edge_lengths) :
  ∃ (CD : ℕ), CD ∈ edges ∧ CD = 13 :=
by
  sorry

end length_CD_l179_179604


namespace smallest_n_l179_179774

open Nat

namespace MathProof

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000)

theorem smallest_n :
  ∃ n, (∀ (s : Finset ℕ), (s ⊆ finset.range 2017) → 
  (finset.card s = n) → 
  ∃ t ⊆ s, (finset.card t = 5) ∧ (finset.sum (finset.map ⟨sum_of_digits, sorry⟩ t) = sum_of_digits (finset.sum t)))
  ∧ n = 110 := 
sorry

end MathProof

end smallest_n_l179_179774


namespace sum_of_integers_l179_179764

theorem sum_of_integers (n : ℕ) (h1 : n > 0) (h2 : 1.5 * n - 6.7 < 8.3) : ∑ i in finset.filter (λ i, i < 10) (finset.range 10), i = 45 :=
by
  sorry

end sum_of_integers_l179_179764


namespace sin_45_eq_l179_179263

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179263


namespace sequence_periodic_l179_179769

-- Define the function f
def f (N : ℕ) :=
  if h : N ≥ 2 then
    let primes := N.factors.group₁_finsupp.keys in
    let exponents := N.factors.group₁_finsupp.values in
    1 + (primes.zip exponents).map (λ p_e : ℕ × ℕ, p_e.1 * p_e.2).sum
  else
    N

-- Define the sequence x_n using f
def sequence (x0 : ℕ) : ℕ → ℕ
  | 0 => x0
  | n+1 => f (sequence n)

-- Prove that the sequence is eventually periodic with a fundamental period of 1 or 2
theorem sequence_periodic (x0 : ℕ) (hx0 : x0 ≥ 2) :
  ∃ p : ℕ, p = 1 ∨ p = 2 ∧ ∀ n m : ℕ, sequence x0 n = sequence x0 m → (n - m) % p = 0 :=
by
  sorry

end sequence_periodic_l179_179769


namespace find_smaller_number_l179_179637

variable (x y : ℕ)

theorem find_smaller_number (h1 : ∃ k : ℕ, x = 2 * k ∧ y = 5 * k) (h2 : x + y = 21) : x = 6 :=
by
  sorry

end find_smaller_number_l179_179637


namespace greatest_int_less_neg_22_3_l179_179642

theorem greatest_int_less_neg_22_3 : ∃ n : ℤ, n = -8 ∧ n < -22 / 3 ∧ ∀ m : ℤ, m < -22 / 3 → m ≤ n :=
by
  sorry

end greatest_int_less_neg_22_3_l179_179642


namespace probability_event1_probability_event2_l179_179504

-- Define the balls and the probability distribution
def balls : List ℕ := [1, 2, 3, 4]

-- Two integers x and y are drawn with replacement
def draws_with_replacement : List (ℕ × ℕ) := 
  (balls.product balls)

-- Probability Mass Function for equal probability draws
def pmf := ProbabilityMassFunction.ofFinsetUniform succeeds,
  λ s, s ∈ draws_with_replacement.toFinset

-- Define the events
def event1 (s : ℕ × ℕ) : Prop := (s.fst + s.snd = 5)
def event2 (s : ℕ × ℕ) : Prop := (2 * s.fst + abs (s.fst - s.snd) = 6)

-- Propositions to be proved
theorem probability_event1 : pmf.event (λ s, event1 s) = 1/4 :=
sorry

theorem probability_event2 : pmf.event (λ s, event2 s) = 1/8 :=
sorry

end probability_event1_probability_event2_l179_179504


namespace clearance_sale_gain_percent_l179_179664

theorem clearance_sale_gain_percent
    (SP: ℝ) (gain_percent: ℝ) (discount_percent: ℝ)
    (CP: ℝ) (DSP: ℝ) (gain: ℝ) (gain_percent_sale: ℝ):
    SP = 30 → 
    gain_percent = 15 → 
    discount_percent = 10 → 
    CP = SP / (1 + gain_percent / 100) → 
    DSP = SP * (1 - discount_percent / 100) → 
    gain = DSP - CP → 
    gain_percent_sale = (gain / CP) * 100 → 
    gain_percent_sale ≈ 3.49 :=
begin
  assume h1 h2 h3 h4 h5 h6 h7,
  sorry
end

end clearance_sale_gain_percent_l179_179664


namespace solve_eq1_solve_eq2_l179_179580

theorem solve_eq1 (y : ℝ) : 6 - 3 * y = 15 + 6 * y ↔ y = -1 := by
  sorry

theorem solve_eq2 (x : ℝ) : (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 2 ↔ x = 2 := by
  sorry

end solve_eq1_solve_eq2_l179_179580


namespace solve_exponential_equation_l179_179619

theorem solve_exponential_equation :
  ∃ x, (2:ℝ)^(2*x) - 8 * (2:ℝ)^x + 12 = 0 ↔ x = 1 ∨ x = 1 + Real.log 3 / Real.log 2 :=
by
  sorry

end solve_exponential_equation_l179_179619


namespace avg_weight_A_l179_179000

-- Define the conditions
def num_students_A : ℕ := 40
def num_students_B : ℕ := 20
def avg_weight_B : ℝ := 40
def avg_weight_whole_class : ℝ := 46.67

-- State the theorem using these definitions
theorem avg_weight_A :
  ∃ W_A : ℝ,
    (num_students_A * W_A + num_students_B * avg_weight_B = (num_students_A + num_students_B) * avg_weight_whole_class) ∧
    W_A = 50.005 :=
by
  sorry

end avg_weight_A_l179_179000


namespace sin_45_eq_sqrt_two_over_two_l179_179254

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179254


namespace prime_exponent_of_5_in_30_factorial_l179_179918

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179918


namespace average_annual_growth_rate_l179_179079

variable (a b : ℝ)

theorem average_annual_growth_rate :
  ∃ x : ℝ, (1 + x)^2 = (1 + a) * (1 + b) ∧ x = Real.sqrt ((1 + a) * (1 + b)) - 1 := by
  sorry

end average_annual_growth_rate_l179_179079


namespace log_base_frac_eq_l179_179749

theorem log_base_frac_eq : log (1/5) 25 = -2 := 
by {
  sorry
}

end log_base_frac_eq_l179_179749


namespace min_sum_of_inverses_l179_179542

theorem min_sum_of_inverses 
  (x y z p q r : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h_sum : x + y + z + p + q + r = 10) :
  (1 / x + 9 / y + 4 / z + 25 / p + 16 / q + 36 / r) = 44.1 :=
sorry

end min_sum_of_inverses_l179_179542


namespace probability_a1_divides_a2_and_a2_divides_a3_l179_179544

theorem probability_a1_divides_a2_and_a2_divides_a3 :
  let S := finset.filter (λ x, x > 0) (finset.Icc (1:ℕ) (30^7))
  ∃ (m n: ℕ), m.gcd n = 1 ∧ (∃ (a1 a2 a3 ∈ S), a1 ∣ a2 ∧ a2 ∣ a3) ∧ (mkRat m n = 147 / 3328) :=
begin
  sorry
end

end probability_a1_divides_a2_and_a2_divides_a3_l179_179544


namespace complex_z_power_l179_179802

theorem complex_z_power:
  ∀ (z : ℂ), (z + 1/z = 2 * Real.cos (5 * Real.pi / 180)) →
  z^1000 + (1/z)^1000 = 2 * Real.cos (20 * Real.pi / 180) :=
by
  sorry

end complex_z_power_l179_179802


namespace exponent_of_5_in_30_factorial_l179_179895

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179895


namespace find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7_l179_179763

def isOdd (n : ℕ) : Prop := n % 2 = 1
def isInRange (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 50
def hasRemainderTwo (n : ℕ) : Prop := n % 7 = 2

theorem find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7 :
  ∃ n : ℕ, isInRange n ∧ isOdd n ∧ hasRemainderTwo n ∧ n = 37 :=
by
  sorry

end find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7_l179_179763


namespace total_goals_is_50_l179_179747

def team_a_first_half_goals := 8
def team_b_first_half_goals := team_a_first_half_goals / 2
def team_c_first_half_goals := 2 * team_b_first_half_goals
def team_a_first_half_missed_penalty := 1
def team_c_first_half_missed_penalty := 2

def team_a_second_half_goals := team_c_first_half_goals
def team_b_second_half_goals := team_a_first_half_goals
def team_c_second_half_goals := team_b_second_half_goals + 3
def team_a_second_half_successful_penalty := 1
def team_b_second_half_successful_penalty := 2

def total_team_a_goals := team_a_first_half_goals + team_a_second_half_goals + team_a_second_half_successful_penalty
def total_team_b_goals := team_b_first_half_goals + team_b_second_half_goals + team_b_second_half_successful_penalty
def total_team_c_goals := team_c_first_half_goals + team_c_second_half_goals

def total_goals := total_team_a_goals + total_team_b_goals + total_team_c_goals

theorem total_goals_is_50 : total_goals = 50 := by
  unfold total_goals
  unfold total_team_a_goals total_team_b_goals total_team_c_goals
  unfold team_a_first_half_goals team_b_first_half_goals team_c_first_half_goals
  unfold team_a_second_half_goals team_b_second_half_goals team_c_second_half_goals
  unfold team_a_second_half_successful_penalty team_b_second_half_successful_penalty
  sorry

end total_goals_is_50_l179_179747


namespace shaded_region_perimeter_l179_179496

theorem shaded_region_perimeter (C : Real) (r : Real) (L : Real) (P : Real)
  (h0 : C = 48)
  (h1 : r = C / (2 * Real.pi))
  (h2 : L = (90 / 360) * C)
  (h3 : P = 3 * L) :
  P = 36 := by
  sorry

end shaded_region_perimeter_l179_179496


namespace range_a_for_decreasing_fn_l179_179814

def is_decreasing_on (f : ℝ → ℝ) (s : set ℝ) := 
  ∀ x y ∈ s, x ≤ y → f y ≤ f x

theorem range_a_for_decreasing_fn :
  ∀ a : ℝ, is_decreasing_on (λ x : ℝ, x^2 + 2 * (a - 1) * x + 2) (set.Iic 6) ↔ a ≤ -5 :=
by
  sorry

end range_a_for_decreasing_fn_l179_179814


namespace range_of_ab_l179_179430

noncomputable def f (a x : ℝ) : ℝ := log ((1 + a * x) / (1 - 2 * x))

theorem range_of_ab (a b : ℝ) (h1 : f a (-x) = - f a x) (h2 : ∀ x, -b < x ∧ x < b → (1 + a * x) / (1 - 2 * x) > 0)
  (h_a : a ≠ -2) (h_b_pos : 0 < b) (h_b_domain : b ≤ 1/2) : 1 < a^b ∧ a^b ≤ sqrt 2 := 
sorry

end range_of_ab_l179_179430


namespace geese_survived_first_year_l179_179559

variable (n h s1 d s2 : ℕ)

theorem geese_survived_first_year :
  let n := 60 in
  let h := 2 * n / 3 in
  let s1 := 3 * h / 4 in
  let d := 3 * s1 / 5 in
  let s2 := s1 - d in
  s2 = 12 := 
by
  sorry

end geese_survived_first_year_l179_179559


namespace sin_45_deg_eq_l179_179306

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179306


namespace exponent_of_5_in_30_factorial_l179_179897

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179897


namespace number_of_elements_in_A_l179_179549

def I : Set ℕ := {n | n ≤ 22}

def A : Set (ℕ × ℕ × ℕ × ℕ) := 
  {t | let a := t.1 in let b := t.2.1 in let c := t.2.2.1 in let d := t.2.2.2 in
       a ∈ I ∧ b ∈ I ∧ c ∈ I ∧ d ∈ I ∧ (a + d) % 23 = 1 ∧ (a * d - b * c) % 23 = 0}

theorem number_of_elements_in_A : (A.toFinset.card) = 552 := 
by
  sorry

end number_of_elements_in_A_l179_179549


namespace main_divisors_equality_implies_equal_l179_179020

theorem main_divisors_equality_implies_equal {a b : ℕ} [Nat.Prime a] [Nat.Prime b] (h_composite_a : ¬ Nat.Prime a) (h_composite_b : ¬ Nat.Prime b) 
  (main_divisors_coincide : ∀ (d : ℕ), d ∈ Nat.divisors a → d ∈ Nat.divisors b) : a = b := by
  sorry

end main_divisors_equality_implies_equal_l179_179020


namespace complex_z_power_l179_179801

theorem complex_z_power:
  ∀ (z : ℂ), (z + 1/z = 2 * Real.cos (5 * Real.pi / 180)) →
  z^1000 + (1/z)^1000 = 2 * Real.cos (20 * Real.pi / 180) :=
by
  sorry

end complex_z_power_l179_179801


namespace sin_45_eq_1_div_sqrt_2_l179_179319

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179319


namespace unique_prime_sum_diff_l179_179367

theorem unique_prime_sum_diff (p : ℕ) (primeP : Prime p)
  (hx : ∃ (x y : ℕ), Prime x ∧ Prime y ∧ p = x + y)
  (hz : ∃ (z w : ℕ), Prime z ∧ Prime w ∧ p = z - w) : p = 5 :=
sorry

end unique_prime_sum_diff_l179_179367


namespace counting_numbers_with_special_order_l179_179833

-- Define the set of digits allowed in the numbers
def digits := {2, 3, 4, 5, 6, 7, 8}

-- Define the condition that a number has three distinct digits in strictly increasing order.
def is_strictly_increasing (n : Nat) : Prop :=
  ∃ (a b c : Nat), n = 100 * a + 10 * b + c ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a < b ∧ b < c

-- Define the condition that a number has three distinct digits in strictly decreasing order.
def is_strictly_decreasing (n : Nat) : Prop :=
  ∃ (a b c : Nat), n = 100 * a + 10 * b + c ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a > b ∧ b > c

-- Define the main theorem as a Lean statement.
theorem counting_numbers_with_special_order : 
  (Finset.filter (λ n, 200 ≤ n ∧ n ≤ 899 ∧ (is_strictly_increasing n ∨ is_strictly_decreasing n))
   (Finset.range 1000)).card = 70 := 
sorry

end counting_numbers_with_special_order_l179_179833


namespace Brianchons_theorem_l179_179532

theorem Brianchons_theorem {A B C D E F O : Type} 
  [CircumscribedHexagon A B C D E F] 
  (h_AD : DiagonalPassesThrough A D O) 
  (h_BE : DiagonalPassesThrough B E O) 
  (h_CF : DiagonalPassesThrough C F O) :
  Concurrent AD BE CF O :=
sorry

end Brianchons_theorem_l179_179532


namespace perpendicular_lines_condition_l179_179054

theorem perpendicular_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, x + y = 0 ∧ x - ay = 0 → x = 0) ↔ (a = 1) := 
sorry

end perpendicular_lines_condition_l179_179054


namespace max_value_sin_cos_combination_l179_179375

theorem max_value_sin_cos_combination :
  ∀ x : ℝ, (5 * Real.sin x + 12 * Real.cos x) ≤ 13 :=
by
  intro x
  sorry

end max_value_sin_cos_combination_l179_179375


namespace sin_45_deg_eq_l179_179308

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179308


namespace appropriate_function_model_is_logarithmic_l179_179073

/-- After the adjustment, the initial profit increased rapidly, 
but the growth rate slowed down over time. -/
def profit_function_model (x y : ℝ) : Prop :=
∃ f : ℝ → ℝ, f x = y ∧ ∀ x, (f x).deriv > 0 ∧ (f x).deriv.deriv < 0

theorem appropriate_function_model_is_logarithmic :
  ∀ (x y : ℝ), profit_function_model x y → ∃ (a b : ℝ), a > 0 ∧ ∀ x, y = a * log x + b :=
by
  sorry

end appropriate_function_model_is_logarithmic_l179_179073


namespace sin_45_degree_l179_179180

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179180


namespace times_more_balloons_l179_179730

theorem times_more_balloons (dan_balloons : ℕ) (tim_balloons : ℕ) (H_dan : dan_balloons = 29) (H_tim : tim_balloons = 203) : tim_balloons / dan_balloons = 7 :=
by 
  rw [H_dan, H_tim]
  norm_num

end times_more_balloons_l179_179730


namespace angle_sum_straight_line_l179_179512

theorem angle_sum_straight_line (x : ℝ) (h : 4 * x + x = 180) : x = 36 :=
sorry

end angle_sum_straight_line_l179_179512


namespace sin_45_degree_l179_179278

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179278


namespace exponent_of_5_in_30_factorial_l179_179949

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179949


namespace curve_proportional_arc_length_l179_179760

def arc_length_in_polar (r : ℝ → ℝ) (φ : ℝ → ℝ) (t : ℝ) : ℝ :=
  let dr := (fun t : ℝ => r t * (deriv r t)) t
  let dφ := (fun t : ℝ => r t * (deriv φ t)) t
  (dr ^ 2 + r t ^ 2 * dφ ^ 2) ^ (1 / 2)

theorem curve_proportional_arc_length (O : Point) (k : ℝ) :
  ∃ C a : ℝ, ∀ r φ : ℝ → ℝ, arc_length_in_polar r φ t = r t * exp (a * φ (t + k)) :=
sorry

end curve_proportional_arc_length_l179_179760


namespace positional_relationship_between_b_and_alpha_l179_179479

-- Definitions representing the conditions
variables (α : Type) [Plane α]
variables (a b : Type) [Line a] [Line b]

-- Conditions
variable (perpendicular_a_alpha : Perpendicular a α)
variable (perpendicular_b_a : Perpendicular b a)

-- Theorem to prove the positional relationship
theorem positional_relationship_between_b_and_alpha
  (perpendicular_a_alpha : Perpendicular a α)
  (perpendicular_b_a : Perpendicular b a) :
  (Parallel b α) ∨ (Subset b α) :=
sorry

end positional_relationship_between_b_and_alpha_l179_179479


namespace time_to_count_envelopes_l179_179702

theorem time_to_count_envelopes (r : ℕ) : (r / 10 = 1) → (r * 60 / r = 60) ∧ (r * 90 / r = 90) :=
by sorry

end time_to_count_envelopes_l179_179702


namespace largest_integer_n_l179_179102

theorem largest_integer_n (n : ℕ) 
  (h_total_containers : 110)
  (h_min_oranges : 98)
  (h_max_oranges : 119) :
  ∃ n, ∀ (containers : ℕ → ℕ), (∀ i, 98 ≤ containers i ∧ containers i ≤ 119) → 
  (∃ i, n ≤ (card (λ x, containers x = i : finset ℕ))) :=
sorry

end largest_integer_n_l179_179102


namespace sin_45_degree_l179_179171

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179171


namespace range_f_min_value_l179_179812

-- Definitions based on conditions
def f (a : ℝ) (x : ℝ) := -a^(2*x) - 2*a^x + 1

-- a > 1
variable {a : ℝ} (ha : 1 < a)

-- minimum value of f(x) in the range x ∈ [-2,1] is -7
variable {x : ℝ} (hx : -2 ≤ x ∧ x ≤ 1)

theorem range_f (a : ℝ) (ha : 1 < a) : 
  set.range (f a) = set.Iio 1 := 
sorry

theorem min_value (a : ℝ) (ha : 1 < a) (hx : -2 ≤ x ∧ x ≤ 1) (hmin : ∃ x, -2 ≤ x ∧ x ≤ 1 ∧ f a x = -7) :
  a = 2 := 
sorry

end range_f_min_value_l179_179812


namespace exponent_of_5_in_30_factorial_l179_179941

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179941


namespace sin_45_eq_sqrt_two_over_two_l179_179251

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179251


namespace exponent_of_5_in_30_factorial_l179_179899

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179899


namespace exponent_of_5_in_30_factorial_l179_179945

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179945


namespace inverse_proposition_false_l179_179600

-- Define the original proposition: ∀ a b, if a = b then |a| = |b|
def original_proposition (a b : ℝ) : Prop := a = b → abs a = abs b

-- Define the inverse proposition: ∀ a b, if |a| = |b| then a = b
def inverse_proposition (a b : ℝ) : Prop := abs a = abs b → a = b

-- Prove that the inverse proposition is false
theorem inverse_proposition_false : ¬ (∀ a b : ℝ, inverse_proposition a b) := 
by {
  intro h,
  have h1 : inverse_proposition 1 (-1) := h 1 (-1),
  have h2 : abs 1 = abs (-1),
  { rfl },
  exact h1 h2,
  have h3 : 1 = -1,
  { sorry }
}

end inverse_proposition_false_l179_179600


namespace sin_45_degree_l179_179177

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179177


namespace inverse_proposition_false_l179_179603

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a = b → abs a = abs b

-- Define the inverse proposition
def inverse_proposition (a b : ℝ) : Prop :=
  abs a = abs b → a = b

-- The theorem to prove
theorem inverse_proposition_false : ∃ (a b : ℝ), abs a = abs b ∧ a ≠ b :=
sorry

end inverse_proposition_false_l179_179603


namespace product_evaluation_l179_179753

theorem product_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) = 5^32 - 4^32 :=
by 
sorry

end product_evaluation_l179_179753


namespace prime_exponent_of_5_in_30_factorial_l179_179906

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179906


namespace min_value_f_a_eq_1_range_of_a_for_solution_prod_of_powers_leq_one_l179_179424

open Real

-- Problem 1: Proving the minimum value of the function f(x) when a = 1 is 0
theorem min_value_f_a_eq_1 : 
  let f (x : ℝ) := exp x - x - 1 
  in ∀ x, f(x) ≥ 0 :=
begin
  sorry
end

-- Problem 2: Determine the range of values for a such that f(x) = 0 has a solution in (0, 2]
theorem range_of_a_for_solution : 
  let f (x : ℝ) (a : ℝ) := exp x - a*x - 1
  in ∀ a, ( ∃ x ∈ Ioo 0 2, f x a = 0 ) ↔ (a > 1 ∧ a ≤ (exp 2 - 1)/2) :=
begin
  sorry
end

-- Problem 3: Prove the inequality involving product of powers
theorem prod_of_powers_leq_one 
  (n : ℕ) (a b : ℕ → ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < b i) 
  (h_sum : ∑ i in finset.range(n+1), a i * b i ≤ ∑ i in finset.range(n+1), b i) :
  ∏ i in finset.range(n+1), (a i ^ (b i)) ≤ 1 :=
begin
  sorry
end

end min_value_f_a_eq_1_range_of_a_for_solution_prod_of_powers_leq_one_l179_179424


namespace tank_fraction_filled_l179_179684

-- A cubical tank with water level at 1 foot and occupying 16 cubic feet
variables (side_length : ℝ) (height : ℝ := 1) (volume_water : ℝ := 16)

-- Capacity of the tank when side length is known (side_length = √(base area) = √16 = 4 feet)
def tank_capacity (side_length : ℝ) : ℝ := side_length ^ 3

-- Let’s state the theorem we need to prove
theorem tank_fraction_filled (h : height = 1) (v : volume_water = 16) (s : side_length = 4) :
  (16 / tank_capacity side_length) = 1 / 4 :=
begin
  sorry -- Proof not required
end

end tank_fraction_filled_l179_179684


namespace exponent_of_5_in_30_factorial_l179_179952

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179952


namespace perfect_square_seq_n_l179_179344

def is_perfect_square (k : ℕ) : Prop := ∃ m : ℕ, m^2 = k

def sequence (n : ℕ) : ℕ :=
  Nat.recOn n 9 (λ n a_n, ((n + 5) * a_n + 22) / (n + 3))

theorem perfect_square_seq_n (n : ℕ) : is_perfect_square (sequence n) ↔ n = 1 ∨ n = 8 := 
by
  sorry

end perfect_square_seq_n_l179_179344


namespace ob_less_than_half_oa1_ob_less_than_or_equal_quarter_oa1_l179_179668

namespace GeometryProof

-- Definitions for points, lines, and angles
variables (Point Line : Type)
variables (O A1 A2 A3 A4 B : Point)
variables (l1 l2 l3 l4 : Line)

-- Conditions: Definitions for lines through O and points derived from conditions
axiom l1_through_O : (O ∈ l1)
axiom l2_through_O : (O ∈ l2)
axiom l3_through_O : (O ∈ l3)
axiom l4_through_O : (O ∈ l4)
axiom A1_on_l1 : (A1 ∈ l1)
axiom A1A2_parallel_l4 : ∀ (x : Point), (A1 ∈ x) → (x ∈ l4) → (A2 ∈ l2)
axiom A2A3_parallel_l1 : ∀ (y : Point), (A2 ∈ y) → (y ∈ l1) → (A3 ∈ l3)
axiom A3A4_parallel_l2 : ∀ (z : Point), (A3 ∈ z) → (z ∈ l2) → (A4 ∈ l4)
axiom A4B_parallel_l3 : ∀ (w : Point), (A4 ∈ w) → (w ∈ l3) → (B ∈ l1)

-- Propositions: OB length related to OA1 length
noncomputable def OA1_length : ℝ := sorry
noncomputable def OB_length : ℝ := sorry

-- Proof 1: OB < 1/2 * OA1
theorem ob_less_than_half_oa1 : OB_length < (1 / 2) * OA1_length :=
sorry

-- Proof 2: OB ≤ 1/4 * OA1 and cannot be improved
theorem ob_less_than_or_equal_quarter_oa1 : OB_length ≤ (1 / 4) * OA1_length :=
sorry

end GeometryProof

end ob_less_than_half_oa1_ob_less_than_or_equal_quarter_oa1_l179_179668


namespace joan_mortgage_payment_l179_179522

noncomputable def geometric_series_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (1 - r^n) / (1 - r)

theorem joan_mortgage_payment : 
  ∃ n : ℕ, geometric_series_sum 100 3 n = 109300 ∧ n = 7 :=
by
  sorry

end joan_mortgage_payment_l179_179522


namespace trigonometric_identity_example_l179_179618

theorem trigonometric_identity_example :
  sin (15 * π / 180) * cos (45 * π / 180) + sin (75 * π / 180) * sin (135 * π / 180) = sqrt 3 / 2 :=
  sorry

end trigonometric_identity_example_l179_179618


namespace log_equation_solution_l179_179575

theorem log_equation_solution (a b x : ℝ) (h : 5 * (Real.log x / Real.log b) ^ 2 + 2 * (Real.log x / Real.log a) ^ 2 = 10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)) :
    b = a ^ (1 + Real.sqrt 15 / 5) ∨ b = a ^ (1 - Real.sqrt 15 / 5) :=
sorry

end log_equation_solution_l179_179575


namespace sin_45_eq_l179_179258

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179258


namespace find_sin_theta_l179_179056

-- Define the square and the midpoints
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 0, y := 0}
def B : Point := {x := 0, y := 4}
def C : Point := {x := 4, y := 4}
def D : Point := {x := 4, y := 0}

def M : Point := {x := 2, y := 4}
def N : Point := {x := 4, y := 2}

-- Define the function to calculate distance between two points
def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Define the function to calculate the angle
def cosine_law (a b c : ℝ) : ℝ :=
  (a^2 + b^2 - c^2) / (2 * a * b)

-- state the theorem
theorem find_sin_theta : 
  let AM := dist A M,
      AN := dist A N,
      MN := dist M N,
      cos_theta := cosine_law AM AN MN,
      sin_squared_theta := 1 - cos_theta^2
  in Real.sqrt sin_squared_theta = 3 / 5 := sorry

end find_sin_theta_l179_179056


namespace chebyshev_inequality_less_than_two_chebyshev_inequality_more_or_equal_to_two_l179_179078

namespace ChebyshevInequalityProof

open MeasureTheory

-- Define the parameters and random variable
noncomputable def n : ℕ := 10
noncomputable def p : ℝ := 0.05

-- Define the expected value and variance
noncomputable def E_X : ℝ := n * p
noncomputable def D_X : ℝ := n * p * (1 - p)

theorem chebyshev_inequality_less_than_two :
  let ε := 2
  in P(X, |X - E_X| < ε) ≥ 0.88 := by
  -- Skip the proof
  sorry

theorem chebyshev_inequality_more_or_equal_to_two :
  let ε := 2
  in P(X, |X - E_X| >= ε) ≤ 0.12 := by
  -- Skip the proof
  sorry

end ChebyshevInequalityProof

end chebyshev_inequality_less_than_two_chebyshev_inequality_more_or_equal_to_two_l179_179078


namespace shaded_triangle_area_l179_179587

def square_area (s : ℝ) : ℝ := s * s

def is_midpoint (p1 p2 p_mid : ℝ) : Prop :=
  p_mid = (p1 + p2) / 2

theorem shaded_triangle_area :
  ∀ (s : ℝ), square_area s = 1 → 
  (∀ (a b c d e f : ℝ), 
    is_midpoint a b e → 
    is_midpoint b c f → 
    is_midpoint c d g → 
    is_midpoint d a h → 
    ∃ (area : ℝ), area = 3 / 32) :=
begin
  intros s h_area a b c d e f,
  intros h_mide_ab h_mide_bc h_mide_cd h_mide_da,
  use 3 / 32,
  sorry,
end

end shaded_triangle_area_l179_179587


namespace count_squares_in_H_l179_179094

def H : Set (ℤ × ℤ) := {p | 2 ≤ |p.1| ∧ |p.1| ≤ 8 ∧ 2 ≤ |p.2| ∧ |p.2| ≤ 8}

theorem count_squares_in_H : H.Finite ∧ (countSquaresInH H 5) = 294 := sorry

def countSquaresInH (S : Set (ℤ × ℤ)) (side : ℕ) : ℕ :=
  -- implementation to count squares, not provided for this example
  sorry

end count_squares_in_H_l179_179094


namespace sin_45_eq_sqrt2_div_2_l179_179329

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179329


namespace sin_45_degree_l179_179169

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179169


namespace deck_card_count_l179_179865

theorem deck_card_count (r n : ℕ) (h1 : n = 2 * r) (h2 : n + 4 = 3 * r) : r + n = 12 :=
by
  sorry

end deck_card_count_l179_179865


namespace hannah_bananas_l179_179830

theorem hannah_bananas (B : ℕ) (h1 : B / 4 = 15 / 3) : B = 20 :=
by
  sorry

end hannah_bananas_l179_179830


namespace sin_750_eq_one_half_l179_179353

theorem sin_750_eq_one_half :
  ∀ (θ: ℝ), (∀ n: ℤ, Real.sin (θ + n * 360) = Real.sin θ) → Real.sin 30 = 1 / 2 → Real.sin 750 = 1 / 2 :=
by 
  intros θ periodic_sine sin_30
  -- insert proof here
  sorry

end sin_750_eq_one_half_l179_179353


namespace sin_45_eq_sqrt2_div_2_l179_179335

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179335


namespace no_int_coords_equilateral_l179_179721

--- Define a structure for points with integer coordinates
structure Point :=
(x : ℤ)
(y : ℤ)

--- Definition of the distance squared between two points
def dist_squared (P Q : Point) : ℤ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

--- Statement that given three points with integer coordinates, they cannot form an equilateral triangle
theorem no_int_coords_equilateral (A B C : Point) :
  ¬ (dist_squared A B = dist_squared B C ∧ dist_squared B C = dist_squared C A ∧ dist_squared C A = dist_squared A B) :=
sorry

end no_int_coords_equilateral_l179_179721


namespace sin_45_degree_l179_179167

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179167


namespace slope_tangent_line_l179_179799

variable {f : ℝ → ℝ}

-- Assumption: f is differentiable
def differentiable_at (f : ℝ → ℝ) (x : ℝ) := ∃ f', ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ → |(f (x + h) - f x) / h - f'| < ε

-- Hypothesis: limit condition
axiom limit_condition : (∀ x, differentiable_at f (1 - x)) → (∀ ε > 0, ∃ δ > 0, ∀ Δx > 0, |Δx| < δ → |(f 1 - f (1 - Δx)) / (2 * Δx) + 1| < ε)

-- Theorem: the slope of the tangent line to the curve y = f(x) at (1, f(1)) is -2
theorem slope_tangent_line : differentiable_at f 1 → (∀ ε > 0, ∃ δ > 0, ∀ Δx > 0, |Δx| < δ → |(f 1 - f (1 - Δx)) / (2 * Δx) + 1| < ε) → deriv f 1 = -2 :=
by
    intro h_diff h_lim
    sorry

end slope_tangent_line_l179_179799


namespace minimum_tents_l179_179075

theorem minimum_tents (X Y : ℕ) (h1 : 10 * (X - 1) < 1.5 * Y) (h2 : 1.5 * Y < 10 * X) 
  (h3 : 10 * (X + 2) < 1.6 * Y) (h4 : 1.6 * Y < 10 * (X + 3)) : Y = 213 :=
sorry

end minimum_tents_l179_179075


namespace number_belongs_to_32nd_group_l179_179613

theorem number_belongs_to_32nd_group :
  ∀ (n : ℕ), (n * n ≥ 2009) → 2009 ∈ { 
    let m := (n - 1) * (n - 1) + 1 in 
    (finset.range (2 * n - 1)).map (λ x, 2 * x + m) 
  } :=
by
  sorry

end number_belongs_to_32nd_group_l179_179613


namespace usable_area_is_correct_l179_179091

variable (x : ℝ)

def total_field_area : ℝ := (x + 9) * (x + 7)
def flooded_area : ℝ := (2 * x - 2) * (x - 1)
def usable_area : ℝ := total_field_area x - flooded_area x

theorem usable_area_is_correct : usable_area x = -x^2 + 20 * x + 61 :=
by
  sorry

end usable_area_is_correct_l179_179091


namespace total_students_in_high_school_l179_179689

theorem total_students_in_high_school
    (senior_students : ℕ) (sample_size : ℕ) (freshman_sample : ℕ) (sophomore_sample : ℕ)
    (known_senior_students : senior_students = 600)
    (known_sample_size : sample_size = 90)
    (known_freshman_sample : freshman_sample = 27)
    (known_sophomore_sample : sophomore_sample = 33) :
    let senior_sample := sample_size - freshman_sample - sophomore_sample in
    senior_sample = 30 →
    let ratio_senior := senior_sample / 600 in
    let ratio_sample := sample_size / 1800 in
    ratio_senior = ratio_sample :=
by
  intros H_senior_sample H_ratio_senior H_ratio_sample
  have H1 : senior_sample = 30 := H_senior_sample
  have H2 : 600 * ratio_senior = sample_size * ratio_sample := sorry
  exact H2

end total_students_in_high_school_l179_179689


namespace parallel_vectors_l179_179058

theorem parallel_vectors (λ : ℝ) (h : (2, 6) = (λ, -1) ↔ True) : λ = -3 := by
  sorry

end parallel_vectors_l179_179058


namespace problem1_problem2_problem3_l179_179829

open Real EuclideanSpace

noncomputable def c (a : ℝ × ℝ) : (ℝ × ℝ) := 
  let x := (1 : ℝ)
  let y := (2 : ℝ)
  if a = (1, 2) then (2, 4) else (-2, -4)

theorem problem1 (a : ℝ × ℝ) (c : ℝ × ℝ) 
(h_a : a = (1, 2))
(h_c_norm : sqrt ((c.1 ^ 2) + (c.2 ^ 2)) = 2 * sqrt 5)
(h_parallel : ∃ k : ℝ, c = (k * a.1, k * a.2)) :
c = (2, 4) ∨ c = (-2, -4) :=
sorry

theorem problem2 (a : ℝ × ℝ) (b : ℝ × ℝ) (theta : ℝ)
(h_a : a = (1, 2))
(h_b_norm : sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = sqrt 5 / 2)
(h_perpendicular : (a.1 + 2 * b.1) * (2 * a.1 - b.1) + (a.2 + 2 * b.2) * (2 * a.2 - b.2) = 0) :
theta = π :=
sorry

theorem problem3 (a : ℝ × ℝ) (b : ℝ × ℝ) (λ : ℝ)
(h_a : a = (1, 2)) 
(h_b : b = (1, 1)) 
(h_acute : (a.1 * (a.1 + λ * b.1) + a.2 * (a.2 + λ * b.2)) > 0):
λ ∈ Icc (-5 / 3) 0 ∪ Ioc 0 ⊤ :=
sorry

end problem1_problem2_problem3_l179_179829


namespace exponent_of_5_in_30_fact_l179_179888

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179888


namespace sqrt_div_eq_five_over_two_l179_179360

variable (a b : ℝ)

def condition : Prop := 
  (1/3)^2 + (1/4)^2 / (1/5)^2 + (1/6)^2 = 25 * a / (61 * b)

theorem sqrt_div_eq_five_over_two (h : condition a b) : 
  sqrt(a) / sqrt(b) = 5 / 2 := 
sorry

end sqrt_div_eq_five_over_two_l179_179360


namespace count_whole_numbers_between_roots_l179_179836

theorem count_whole_numbers_between_roots :
  1 < real.root 4 10 ∧ real.root 3 1000 = 10 →
  ∃ n, n = 9 := 
by
  intro h
  sorry

end count_whole_numbers_between_roots_l179_179836


namespace segment_length_in_meters_l179_179083

-- Conditions
def inch_to_meters : ℝ := 500
def segment_length_in_inches : ℝ := 7.25

-- Theorem to prove
theorem segment_length_in_meters : segment_length_in_inches * inch_to_meters = 3625 := by
  sorry

end segment_length_in_meters_l179_179083


namespace sin_45_deg_eq_l179_179302

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179302


namespace probability_odd_product_l179_179627

open Nat

noncomputable def countOddSumOfDigits (n : ℕ) : ℕ :=
  (List.range (n + 1)).countp 
    (fun k => k % 2 = 1 ∧ digitSum k % 2 = 1)

theorem probability_odd_product :
  ∀ (a b c : ℕ),
    (1 ≤ a) ∧ (a ≤ 2020) ∧ 
    (1 ≤ b) ∧ (b ≤ 2020) ∧ 
    (1 ≤ c) ∧ (c ≤ 2020) ∧ 
    (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ 
    (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ 
    (digitSum a % 2 = 1) ∧ (digitSum b % 2 = 1) ∧ (digitSum c % 2 = 1) 
    → (505 / 2020) * (504 / 2019) * (503 / 2018) = 1 / 64 :=
  sorry

end probability_odd_product_l179_179627


namespace square_side_lengths_l179_179093

theorem square_side_lengths (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 120) :
  (x = 13 ∧ y = 7) ∨ (x = 7 ∧ y = 13) :=
by {
  -- skip proof
  sorry
}

end square_side_lengths_l179_179093


namespace sin_45_eq_1_div_sqrt_2_l179_179317

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179317


namespace cubic_polynomial_sum_l179_179077

noncomputable def q : ℚ → ℚ := sorry

theorem cubic_polynomial_sum :
  (∀ x : ℚ, ∃ a b c d : ℚ, q(x) = a * x^3 + b * x^2 + c * x + d) →
  q(3) = 2 →
  q(8) = 20 →
  q(16) = 12 →
  q(21) = 30 →
  (finset.range 21).sum (λ n, q (n + 2)) = 336 :=
begin
  sorry
end

end cubic_polynomial_sum_l179_179077


namespace addition_problem_l179_179502

theorem addition_problem (x y S : ℕ) 
    (h1 : x = S - 2000)
    (h2 : S = y + 6) :
    x = 6 ∧ y = 2000 ∧ S = 2006 :=
by
  -- The proof will go here
  sorry

end addition_problem_l179_179502


namespace sin_45_degree_eq_sqrt2_div_2_l179_179157

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179157


namespace find_weight_A_l179_179045

noncomputable def weight_of_A (a b c d e : ℕ) : Prop :=
  (a + b + c) / 3 = 84 ∧
  (a + b + c + d) / 4 = 80 ∧
  e = d + 5 ∧
  (b + c + d + e) / 4 = 79 →
  a = 77

theorem find_weight_A (a b c d e : ℕ) : weight_of_A a b c d e :=
by
  sorry

end find_weight_A_l179_179045


namespace number_of_ways_divisible_by_101_l179_179505

noncomputable def board_101_101 :=
  { board : Fin 101 × Fin 101 → ℕ // ∀ (cells : Fin 101 → Fin 101), 
    ∑ i in Finset.univ, board (i, cells i) % 101 = 0 }

theorem number_of_ways_divisible_by_101
  (board : board_101_101) :
  (∃ (select_cells : Fin 101 → Fin 101), ∑ i in Finset.univ, board.1 (i, select_cells i) % 101 = 0) → 
  (Finset.card { select_cells : Fin 101 → Fin 101 // 
    ∑ i in Finset.univ, board.1 (i, select_cells i) % 101 = 0 } % 101 = 0) :=
sorry

end number_of_ways_divisible_by_101_l179_179505


namespace sin_45_eq_l179_179264

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179264


namespace exponent_of_5_in_30_factorial_l179_179935

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179935


namespace exponent_of_5_in_30_factorial_l179_179892

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179892


namespace constant_term_expansion_l179_179369

theorem constant_term_expansion :
  (∃ T : ℤ, T = -80 ∧ (√x - 2/∛x)^5 = T) :=
sorry

end constant_term_expansion_l179_179369


namespace fill_tank_time_l179_179060

theorem fill_tank_time :
  let tank_capacity : ℤ := 10000 -- tank capacity in liters
  let initial_water : ℤ := tank_capacity / 2 -- half-full tank
  let fill_rate : ℚ := 1 / 2 -- fill rate in kiloliters per minute
  let drain_rate1 : ℚ := 1 / 4 -- first drain rate, kiloliters per minute
  let drain_rate2 : ℚ := 1 / 6 -- second drain rate, kiloliters per minute
  
  let net_flow_rate : ℚ := fill_rate - (drain_rate1 + drain_rate2) -- net flow rate
  let remaining_volume : ℚ := (tank_capacity / 2) / 1000 -- remaining volume in kiloliters
  
  -- Time to fill = remaining volume / net flow rate
  let time_to_fill : ℚ := remaining_volume / net_flow_rate in
  time_to_fill = 60 :=
by {
  sorry
}

end fill_tank_time_l179_179060


namespace smallest_abundant_even_gt_12_eq_18_l179_179462

noncomputable def proper_divisors_sum (n : ℕ) : ℕ :=
∑ d in Finset.filter (λ d, d < n ∧ n % d = 0) (Finset.range (n + 1)), d

def is_abundant (n : ℕ) : Prop := proper_divisors_sum n > n

def smallest_abundant_even_number_greater_than (k : ℕ) (m : ℕ) : ℕ :=
nat.find (λ n, n > m ∧ n % 2 = 0 ∧ is_abundant n)

theorem smallest_abundant_even_gt_12_eq_18 : smallest_abundant_even_number_greater_than 18 12 = 18 :=
by
  -- Proof skipped
  sorry

end smallest_abundant_even_gt_12_eq_18_l179_179462


namespace largest_fraction_l179_179032

theorem largest_fraction
: (1 : ℚ / 5 * 2) < (1 : ℚ / 7 * 3) ∧
  (1 : ℚ / 9 * 4) < (1 : ℚ / 7 * 3) ∧
  (1 : ℚ / 11 * 5) < (1 : ℚ / 7 * 3) ∧
  (1 : ℚ / 13 * 6) < (1 : ℚ / 7 * 3):=
by
  sorry

end largest_fraction_l179_179032


namespace sin_45_eq_sqrt_two_over_two_l179_179249

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179249


namespace catch_up_time_l179_179080

variable (x : ℝ) -- speed of the girl
variable (t : ℝ) -- time until the young man catches up with the girl

-- Declaring speeds based on given conditions
def speed_of_young_man : ℝ := 2 * x
def speed_of_tram : ℝ := 10 * x

-- Main theorem statement
theorem catch_up_time
  (h : 2 * x * t + (8 * speed_of_tram) = 8 * x + t * x) :
  t = 88 := 
sorry

end catch_up_time_l179_179080


namespace abcd_efgh_difference_l179_179387

-- Define the function f(wxyz) = 5^w * 3^x * 2^y * 7^z
def f (w x y z : ℕ) : ℕ := 5^w * 3^x * 2^y * 7^z

-- Prove that for a, b, c, d, e, f, g, h as given in the problem, the difference between abcd - efgh is 1
theorem abcd_efgh_difference (a b c d e f g h : ℕ) (h1 : f a b c d = 7 * f e f g h) (h2 : d = h + 1) : 
  (1000 * a + 100 * b + 10 * c + d) - (1000 * e + 100 * f + 10 * g + h) = 1 :=
by
  sorry

end abcd_efgh_difference_l179_179387


namespace probability_of_composite_in_range_50_l179_179665

-- Define the range of natural numbers from 1 to 50
def range_50 := { n : ℕ | 1 ≤ n ∧ n ≤ 50 }

-- Define the set of composite numbers within this range
def composite_numbers_50 := { 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 
                              24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 
                              39, 40, 42, 44, 45, 46, 48, 49, 50 }

-- Define the count of composite numbers in the specified range
def count_composite_numbers_50 := 34

-- Define the total count of natural numbers in the specified range
def total_natural_numbers_50 := 50

-- Define the probability of selecting a composite number
theorem probability_of_composite_in_range_50 : 
  (count_composite_numbers_50 : ℝ) / (total_natural_numbers_50 : ℝ) = 0.68 := 
by
  sorry

end probability_of_composite_in_range_50_l179_179665


namespace largest_three_digit_interesting_arbitrarily_large_interesting_l179_179719

def is_interesting (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≤ n → ∃ d : Finset ℕ, d ⊆ n.divisors ∧ m = ∑ x in d, x ∧ d.Pairwise (≠)

theorem largest_three_digit_interesting :
  ∃ (n : ℕ), is_interesting n ∧ n < 1000 ∧ ∀ m : ℕ, m < 1000 → is_interesting m → m ≤ n :=
  ⟨992, sorry⟩

theorem arbitrarily_large_interesting :
  ∀ N : ℕ, ∃ n : ℕ, is_interesting n ∧ n > N ∧ ¬ ∃ k : ℕ, n = 2^k :=
  sorry

end largest_three_digit_interesting_arbitrarily_large_interesting_l179_179719


namespace prime_exponent_of_5_in_30_factorial_l179_179915

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179915


namespace prime_exponent_of_5_in_30_factorial_l179_179910

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179910


namespace exponent_of_5_in_30_factorial_l179_179951

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179951


namespace sin_45_degree_l179_179178

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179178


namespace exponent_of_5_in_30_factorial_l179_179921

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179921


namespace exponent_of_5_in_30_factorial_l179_179994

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179994


namespace no_equilateral_integer_coords_l179_179724

theorem no_equilateral_integer_coords (x1 y1 x2 y2 x3 y3 : ℤ) : 
  ¬ ((x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
     (x1 ≠ x3 ∨ y1 ≠ y3) ∧
     (x2 ≠ x3 ∨ y2 ≠ y3) ∧ 
     ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x1) ^ 2 + (y3 - y1) ^ 2 ∧ 
      (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x2) ^ 2 + (y3 - y2) ^ 2)) :=
by
  sorry

end no_equilateral_integer_coords_l179_179724


namespace log_one_fifth_twentyfive_l179_179752

theorem log_one_fifth_twentyfive : logb (1/5) 25 = -2 := by
  -- defer the proof, which is not required
  sorry

end log_one_fifth_twentyfive_l179_179752


namespace leap_years_count_between_2000_and_4000_l179_179697

def is_leap_year (y : ℕ) : Prop :=
  ∃ r, r ∈ {200, 300, 600, 700} ∧ (y % 1000 = r)

def valid_year (y : ℕ) : Prop :=
  2000 < y ∧ y < 4000 ∧ y % 100 = 0

def count_leap_years (low high : ℕ) : ℕ :=
  (set.to_finset {y | valid_year y ∧ is_leap_year y}).card

theorem leap_years_count_between_2000_and_4000 :
  count_leap_years 2000 4000 = 8 :=
by
  sorry

end leap_years_count_between_2000_and_4000_l179_179697


namespace evaluate_expression_l179_179652

theorem evaluate_expression : 4 * (8 - 3) - 6 / 3 = 18 :=
by sorry

end evaluate_expression_l179_179652


namespace exponent_of_5_in_30_factorial_l179_179924

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179924


namespace ratio_of_perimeters_l179_179647

variable {s : ℝ}
def original_square_perimeter := 4 * s

def first_square_side_length := 2 * s
def first_square_perimeter := 4 * first_square_side_length

def second_square_side_length := 2 * s
def second_square_perimeter := 4 * second_square_side_length

theorem ratio_of_perimeters
  (h1 : first_square_side_length = 2 * s)
  (h2 : second_square_side_length = 2 * s):
  first_square_perimeter / second_square_perimeter = 1 :=
by sorry

end ratio_of_perimeters_l179_179647


namespace exponent_of_5_in_30_fact_l179_179887

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179887


namespace exponent_of_5_in_30_factorial_l179_179955

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179955


namespace irrational_dimensions_integer_Volume_SurfaceArea_l179_179518

-- Define the dimensions a, b, and c as specific irrational numbers
def a := Real.sqrt 2 - 1
def b := Real.sqrt 2 - 1
def c := 3 + 2 * Real.sqrt 2

-- Define the volume V and surface area A
def V := a * b * c
def A := 2 * (a * b + a * c + b * c)

-- Prove that the volume and surface area are integers
theorem irrational_dimensions_integer_Volume_SurfaceArea : 
  a = Real.sqrt 2 - 1 ∧ 
  b = Real.sqrt 2 - 1 ∧ 
  c = 3 + 2 * Real.sqrt 2 ∧ 
  V = 1 ∧ 
  A = 20 := 
by 
  -- Use the definitions directly
  have h_a_val : a = Real.sqrt 2 - 1 := rfl
  have h_b_val : b = Real.sqrt 2 - 1 := rfl
  have h_c_val : c = 3 + 2 * Real.sqrt 2 := rfl
  -- Compute V
  have h_V_val := calc 
    V = (Real.sqrt 2 - 1) * (Real.sqrt 2 - 1) * (3 + 2 * Real.sqrt 2) : by rw [V, h_a_val, h_b_val, h_c_val]
      ... = (3 - 2 * Real.sqrt 2) * (3 + 2 * Real.sqrt 2) : by rw [(Real.sqrt 2 - 1) * (Real.sqrt 2 - 1)]
      ... = 9 - 8 : by norm_num
      ... = 1 : by norm_num
  -- Compute A 
  have h_A_val := calc
    A = 2 * ((Real.sqrt 2 - 1) * (Real.sqrt 2 - 1) + 2 * (Real.sqrt 2 - 1) * (3 + 2 * Real.sqrt 2)) : by rw [A, h_a_val, h_b_val, h_c_val]
      ... = 2 * (3 - 2 * Real.sqrt 2 + 8 + 2 * Real.sqrt 2) : by norm_num
      ... = 2 * 10 : by norm_num
      ... = 20 : by norm_num
  -- Conclude the proof
  exact ⟨h_a_val, h_b_val, h_c_val, h_V_val, h_A_val⟩

end irrational_dimensions_integer_Volume_SurfaceArea_l179_179518


namespace coefficient_x4_in_binom_l179_179875

theorem coefficient_x4_in_binom (x : ℝ) : 
    (coeff (expand (1 - x) 5) 4) = -5 := 
sorry

end coefficient_x4_in_binom_l179_179875


namespace sin_45_degree_eq_sqrt2_div_2_l179_179150

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179150


namespace complex_z_1000_l179_179803

open Complex

theorem complex_z_1000 (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (Real.pi * 5 / 180)) :
  z^(1000 : ℕ) + (z^(1000 : ℕ))⁻¹ = 2 * Real.cos (Real.pi * 20 / 180) :=
sorry

end complex_z_1000_l179_179803


namespace no_real_roots_of_quadratic_composition_l179_179611

noncomputable def quadratic_poly (a b c : ℝ) : ℝ → ℝ := λ x, a*x^2 + b*x + c

theorem no_real_roots_of_quadratic_composition
  (a b c : ℝ)
  (h : ∀ x : ℝ, quadratic_poly a (b - 1) c x ≠ 0) :
  ∀ x : ℝ, quadratic_poly a b c (quadratic_poly a b c x) ≠ x :=
by
  sorry

end no_real_roots_of_quadratic_composition_l179_179611


namespace greatest_int_less_neg_22_3_l179_179643

theorem greatest_int_less_neg_22_3 : ∃ n : ℤ, n = -8 ∧ n < -22 / 3 ∧ ∀ m : ℤ, m < -22 / 3 → m ≤ n :=
by
  sorry

end greatest_int_less_neg_22_3_l179_179643


namespace volume_region_between_spheres_l179_179623

theorem volume_region_between_spheres (r_small r_large : ℝ)
    (h_small : r_small = 4) (h_large : r_large = 8) :
    (4 / 3) * π * (r_large ^ 3 - r_small ^ 3) = (1792 / 3) * π := 
by
  -- utilize the conditions
  rw [h_small, h_large]
  -- simplify the expression
  norm_num


end volume_region_between_spheres_l179_179623


namespace volume_of_revolved_triangle_l179_179701

-- Define the vertices of the triangle
structure Point where
  x : ℝ
  y : ℝ

def vertex1 : Point := ⟨1003, 0⟩
def vertex2 : Point := ⟨1004, 3⟩
def vertex3 : Point := ⟨1005, 1⟩

-- Define the triangle as a set of points in the xy-plane
def triangle := {p : Point | ∃ a b c : ℝ, a + b + c = 1 ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
                  p.x = a * vertex1.x + b * vertex2.x + c * vertex3.x ∧ 
                  p.y = a * vertex1.y + b * vertex2.y + c * vertex3.y}

-- Define the resulting solid obtained by revolving the triangle around the y-axis
def volume_of_solid (tri : Set Point) : ℝ :=
  2 * Math.pi * ∫ (p : Point) in tri, p.x

-- Define the main theorem
theorem volume_of_revolved_triangle :
  volume_of_solid triangle = 5020 * Math.pi :=
by
  sorry

end volume_of_revolved_triangle_l179_179701


namespace exponent_of_5_in_30_fact_l179_179890

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179890


namespace arrange_in_non_decreasing_order_l179_179130

def Psi : ℤ := (1 / 2 : ℚ) * (Finset.sum (Finset.range 1006) (λ i, 1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + (4 * i)))

def Omega : ℤ := Finset.sum (Finset.range 1007) (λ i, 1 - 2 + 3 - 4 + (2 * i))

def Theta : ℤ := Finset.sum (Finset.range 504) (λ i, 1 - 3 + 5 - 7 + (4 * i))

theorem arrange_in_non_decreasing_order : [Theta, Omega, Psi] = [(-1008 : ℤ), (-1007 : ℤ), (-1006 : ℤ)] :=
by
  have : Psi = -1006 := sorry
  have : Omega = -1007 := sorry
  have : Theta = -1008 := sorry
  rw [this, this, this]
  trivial

end arrange_in_non_decreasing_order_l179_179130


namespace sin_45_eq_sqrt_two_over_two_l179_179252

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179252


namespace product_symmetry_about_y_eq_x_l179_179488

theorem product_symmetry_about_y_eq_x 
  (z1 z2 : ℂ)
  (symmetry : ∀ (z1 z2 : ℂ), z1.re = z2.im ∧ z1.im = z2.re) 
  (hz1 : z1 = 3 + 2 * complex.I) 
  (hz2 : z2 = 2 + 3 * complex.I) :
  z1 * z2 = 13 * complex.I :=
by
  sorry

end product_symmetry_about_y_eq_x_l179_179488


namespace groups_needed_l179_179067

theorem groups_needed (h_camper_count : 36 > 0) (h_group_limit : 12 > 0) : 
  ∃ x : ℕ, x = 36 / 12 ∧ x = 3 := by
  sorry

end groups_needed_l179_179067


namespace bill_amount_each_person_shared_l179_179046

-- Conditions
def total_bill : ℝ := 139.00
def tip_percentage : ℝ := 0.10
def number_of_people : ℕ := 7

-- Calculation to prove the correctness of the split bill amount
def tip_amount : ℝ := total_bill * tip_percentage
def total_with_tip : ℝ := total_bill + tip_amount
def amount_each_person_pays : ℝ := total_with_tip / number_of_people

-- The statement to prove
theorem bill_amount_each_person_shared : amount_each_person_pays = 21.84 := by
  sorry  -- Proof goes here

end bill_amount_each_person_shared_l179_179046


namespace tan_theta_value_l179_179489

theorem tan_theta_value
  (x y : ℝ)
  (h₁ : x = -√3 / 2)
  (h₂ : y = 1 / 2) :
  real.tan (real.atan2 y x) = -√3 / 3 := by
  sorry

end tan_theta_value_l179_179489


namespace inner_ring_speed_minimum_train_distribution_l179_179624

theorem inner_ring_speed_minimum
  (l_inner : ℝ) (num_trains_inner : ℕ) (max_wait_inner : ℝ) (speed_min : ℝ) :
  l_inner = 30 →
  num_trains_inner = 9 →
  max_wait_inner = 10 →
  speed_min = 20 :=
by 
  sorry

theorem train_distribution
  (l_inner : ℝ) (speed_inner : ℝ) (speed_outer : ℝ) (total_trains : ℕ) (max_wait_diff : ℝ) (trains_inner : ℕ) (trains_outer : ℕ) :
  l_inner = 30 →
  speed_inner = 25 →
  speed_outer = 30 →
  total_trains = 18 →
  max_wait_diff = 1 →
  trains_inner = 10 →
  trains_outer = 8 :=
by 
  sorry

end inner_ring_speed_minimum_train_distribution_l179_179624


namespace problem_172_l179_179567

-- Definition to model points and intersections
variables {Point Line : Type}

-- The condition that perpendiculars from A1, B1, C1 to BC, CA, AB intersect at one point
def perpendiculars_intersect (p1 p2 p3 l1 l2 l3 : Point) : Prop :=
  ∃ P : Point, True -- A placeholder for actual geometrical construction

-- Theorems used from problem context directly
theorem problem_172 (A1 B1 C1 A B C : Point) (BC CA AB B1C1 C1A1 A1B1 : Line) 
  (h : perpendiculars_intersect A1 B1 C1 BC CA AB) : 
  perpendiculars_intersect A B C B1C1 C1A1 A1B1 :=
sorry

end problem_172_l179_179567


namespace no_int_coords_equilateral_l179_179722

--- Define a structure for points with integer coordinates
structure Point :=
(x : ℤ)
(y : ℤ)

--- Definition of the distance squared between two points
def dist_squared (P Q : Point) : ℤ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

--- Statement that given three points with integer coordinates, they cannot form an equilateral triangle
theorem no_int_coords_equilateral (A B C : Point) :
  ¬ (dist_squared A B = dist_squared B C ∧ dist_squared B C = dist_squared C A ∧ dist_squared C A = dist_squared A B) :=
sorry

end no_int_coords_equilateral_l179_179722


namespace median_of_mileages_l179_179011

theorem median_of_mileages (mileages : List ℝ) (h : mileages = [96, 112, 97, 108, 99, 104, 86, 98]) :
  median (sort mileages) = 98.5 := by
  sorry

-- Define median for a list in Lean 4.
noncomputable def median (lst : List ℝ) : ℝ :=
  if lst.length % 2 = 1 then
    -- odd length: return the middle element
    lst.nth (lst.length / 2)
  else
    -- even length: return the average of the two middle elements
    (lst.nth (lst.length / 2 - 1) + lst.nth (lst.length / 2)) / 2

end median_of_mileages_l179_179011


namespace exponent_of_5_in_30_factorial_l179_179931

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179931


namespace sin_45_eq_one_div_sqrt_two_l179_179214

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179214


namespace books_on_each_shelf_l179_179553

theorem books_on_each_shelf 
  (shelves_mystery : ℕ) 
  (shelves_picture : ℕ) 
  (total_books : ℕ) 
  (h_mystery : shelves_mystery = 8) 
  (h_picture : shelves_picture = 2) 
  (h_total : total_books = 70) 
  : total_books / (shelves_mystery + shelves_picture) = 7 := 
by
  rw [h_mystery, h_picture, h_total]
  norm_num
  sorry

end books_on_each_shelf_l179_179553


namespace florist_rose_count_l179_179688

theorem florist_rose_count :
  ∀ (initial sold picked : ℕ), initial = 37 → sold = 16 → picked = 19 → initial - sold + picked = 40 :=
by
  intros initial sold picked h_initial h_sold h_picked
  rw [h_initial, h_sold, h_picked]
  exact Nat.sub_add_cancel 16 16 19 -- assuming 37 ≥ 16

end florist_rose_count_l179_179688


namespace remainder_polynomial_l179_179541

noncomputable def p (x : ℝ) : ℝ := sorry
noncomputable def r (x : ℝ) : ℝ := x^2 + x

theorem remainder_polynomial (p : ℝ → ℝ) (r : ℝ → ℝ) :
  (p 2 = 6) ∧ (p 4 = 20) ∧ (p 6 = 42) →
  (r 2 = 2^2 + 2) ∧ (r 4 = 4^2 + 4) ∧ (r 6 = 6^2 + 6) :=
sorry

end remainder_polynomial_l179_179541


namespace aram_fraction_of_fine_l179_179040

theorem aram_fraction_of_fine
  (F : ℝ)
  (Joe_payment : ℝ := (1 / 4) * F + 3)
  (Peter_payment : ℝ := (1 / 3) * F - 3)
  (Aram_payment : ℝ := (1 / 2) * F - 4)
  (sum_payments_eq_F : Joe_payment + Peter_payment + Aram_payment = F):
  (Aram_payment / F) = (5 / 12) :=
by
  sorry

end aram_fraction_of_fine_l179_179040


namespace sin_45_deg_l179_179191

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179191


namespace sphere_radius_l179_179857

theorem sphere_radius (r : ℝ) (h : (4 * real.pi * r^3) / 3 = 4 * real.pi * r^2 ) : r = 3 :=
  sorry

end sphere_radius_l179_179857


namespace find_a_l179_179437

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 5 * x^2 + a * x

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, x = -3 ∧ ∀ x' : ℝ, f (f'':=f' (a:=a) : (x : ℝ) -> x^3 + 5x^2 + ax
  := 3x^2 + 10x + a) x' = 0 -> x = -3) → a = 3 :=
by
  sorry

end find_a_l179_179437


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179234

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179234


namespace exponent_of_5_in_30_factorial_l179_179998

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179998


namespace exponent_of_5_in_30_factorial_l179_179933

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179933


namespace baseball_cards_given_l179_179038

theorem baseball_cards_given
  (initial_cards : ℕ)
  (maria_take : ℕ)
  (peter_cards : ℕ)
  (paul_triples : ℕ)
  (final_cards : ℕ)
  (h1 : initial_cards = 15)
  (h2 : maria_take = (initial_cards + 1) / 2)
  (h3 : final_cards = 3 * (initial_cards - maria_take - peter_cards))
  (h4 : final_cards = 18) :
  peter_cards = 1 := 
sorry

end baseball_cards_given_l179_179038


namespace exponent_of_5_in_30_factorial_l179_179893

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179893


namespace num_polygon_sides_with_angle_divisible_by_9_l179_179540

noncomputable def count_polygon_sides_divisible_by_9 : Nat :=
  List.filter (λ n => (360 / n) % 9 = 0) (List.range (15 - 3 + 1)).map (λ x => x + 3)).length

theorem num_polygon_sides_with_angle_divisible_by_9 :
  count_polygon_sides_divisible_by_9 = 5 := by
  sorry

end num_polygon_sides_with_angle_divisible_by_9_l179_179540


namespace prime_exponent_of_5_in_30_factorial_l179_179908

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179908


namespace exponent_of_5_in_30_factorial_l179_179930

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179930


namespace range_of_a_l179_179421

def line_intersects_circle (a : ℝ) : Prop :=
  let distance_from_center_to_line := |1 - a| / Real.sqrt 2
  distance_from_center_to_line ≤ Real.sqrt 2

theorem range_of_a :
  {a : ℝ | line_intersects_circle a} = {a : ℝ | -1 ≤ a ∧ a ≤ 3} :=
by
  sorry

end range_of_a_l179_179421


namespace evaluation_of_expression_l179_179465

theorem evaluation_of_expression
  (a b x y m : ℤ)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = -1) :
  2023 * (a + b) + 3 * (|m|) - 2 * (x * y) = 1 :=
by
  -- skipping the proof
  sorry

end evaluation_of_expression_l179_179465


namespace vector_perpendicular_to_line_l179_179827

theorem vector_perpendicular_to_line :
  let a := (2, 3)
  let l : ℝ → ℝ → Prop := λ x y, 2 * x + 3 * y - 1 = 0
  let k := 3 / 2
  let m := -2 / 3
  - k * m = 1 :=
by
  sorry

end vector_perpendicular_to_line_l179_179827


namespace value_of_m_if_pure_imaginary_l179_179483

variable (m : ℝ)

def z : ℂ := (2 + 3 * complex.i) * (1 - m * complex.i)

-- Definition that z is pure imaginary means the real part is 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem value_of_m_if_pure_imaginary :
  is_pure_imaginary (z m) -> m = -2 / 3 :=
by sorry

end value_of_m_if_pure_imaginary_l179_179483


namespace math_expr_evaluation_l179_179649

theorem math_expr_evaluation :
  3 + 15 / 3 - 2^2 + 1 = 5 :=
by
  -- The proof will be filled here
  sorry

end math_expr_evaluation_l179_179649


namespace solution_l179_179736

def valid_r (r : ℝ) : Prop :=
  ∀ (s : ℝ), s > 0 → (4 * (r * s^2 + r^2 * s + 4 * s^2 + 4 * r * s)) / (r + s) ≤ 3 * r^2 * s

theorem solution :
  ∀ (r : ℝ), (r ∈ set.Ici ((2 + 2 * real.sqrt 13) / 3) → valid_r r) :=
begin
  intros r hr s hs,
  -- The proof steps will go here, but it's omitted and replaced with 'sorry'
  sorry,
end

end solution_l179_179736


namespace sin_45_degree_l179_179159

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179159


namespace exponent_of_5_in_30_factorial_l179_179927

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179927


namespace value_of_expression_l179_179468

noncomputable def largestNegativeInteger : Int := -1

theorem value_of_expression (a b x y : ℝ) (m : Int)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = largestNegativeInteger) :
  2023 * (a + b) + 3 * |m| - 2 * (x * y) = 1 :=
by
  sorry

end value_of_expression_l179_179468


namespace sin_45_eq_sqrt_two_over_two_l179_179245

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179245


namespace chess_tournament_points_l179_179869

theorem chess_tournament_points (boys girls : ℕ) (total_points : ℝ) 
  (total_matches : ℕ)
  (matches_among_boys points_among_boys : ℕ)
  (matches_among_girls points_among_girls : ℕ)
  (matches_between points_between : ℕ)
  (total_players : ℕ := boys + girls)
  (H1 : boys = 9) (H2 : girls = 3) (H3 : total_players = 12)
  (H4 : total_matches = total_players * (total_players - 1) / 2) 
  (H5 : total_points = total_matches) 
  (H6 : matches_among_boys = boys * (boys - 1) / 2) 
  (H7 : points_among_boys = matches_among_boys)
  (H8 : matches_among_girls = girls * (girls - 1) / 2) 
  (H9 : points_among_girls = matches_among_girls) 
  (H10 : matches_between = boys * girls) 
  (H11 : points_between = matches_between) :
  ¬ ∃ (P_B P_G : ℝ) (x : ℝ),
    P_B = points_among_boys + x ∧
    P_G = points_among_girls + (points_between - x) ∧
    P_B = P_G := by
  sorry

end chess_tournament_points_l179_179869


namespace perimeter_triangle_l179_179630

noncomputable def incenter (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry     -- Definition of incenter, skipping proof

def triangle_PQR (P Q R X Y : ℝ × ℝ) (PQ QR PR : ℝ) : Prop :=
  dist P Q = PQ ∧ dist Q R = QR ∧ dist P R = PR ∧ 
  ∃ I : ℝ × ℝ, incenter P Q R = I ∧ (side_parallel (I, X) (Q, R)) ∧ 
  line_through I X PQ ∧ line_through I Y PR

theorem perimeter_triangle (P Q R X Y : ℝ × ℝ) (PQ QR PR : ℝ) :
  triangle_PQR P Q R X Y PQ QR PR → PQ = 15 → QR = 30 → PR = 20 →
  dist P X + dist X Y + dist Y P = 35 := 
by
  intros hPQ hPQR
  sorry

end perimeter_triangle_l179_179630


namespace joint_probability_l179_179420

noncomputable def P (A B : Prop) : ℝ := sorry
def A : Prop := sorry
def B : Prop := sorry

axiom prob_A : P A true = 0.005
axiom prob_B_given_A : P B true = 0.99

theorem joint_probability :
  P A B = 0.00495 :=
by sorry

end joint_probability_l179_179420


namespace complement_of_M_in_U_l179_179443

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end complement_of_M_in_U_l179_179443


namespace tangent_line_equation_at_one_l179_179596

noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) * Real.exp x

theorem tangent_line_equation_at_one :
  (∃ (m b : ℝ), m = Real.exp 1 ∧ b = -2 * Real.exp 1 ∧ ∀ x y, y = f x → y = m * x + b) :=
by
  use Real.exp 1
  use -2 * Real.exp 1
  split
  { rfl }
  split
  { rfl }
  intros x y h
  apply h
  sorry

end tangent_line_equation_at_one_l179_179596


namespace sin_45_eq_1_div_sqrt_2_l179_179312

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179312


namespace parabola_ellipse_graph_l179_179706

theorem parabola_ellipse_graph (
  (m n : ℝ) 
  (m_ne_zero : m ≠ 0)
  (n_ne_zero : n ≠ 0)
):
  let eq1 := ∀ (x y : ℝ), m * x + n * y^2 = 0
  let eq2 := ∀ (x y : ℝ), m * x^2 + n * y^2 = 1
  figure = "A" 
  sorry

end parabola_ellipse_graph_l179_179706


namespace log_one_fifth_twentyfive_l179_179751

theorem log_one_fifth_twentyfive : logb (1/5) 25 = -2 := by
  -- defer the proof, which is not required
  sorry

end log_one_fifth_twentyfive_l179_179751


namespace cheese_sculpture_blocks_needed_l179_179064

/-
Given a block of cheese shaped as a right rectangular prism measuring 8 inches by 3 inches by 2 inches,
and a cheese sculpture in the shape of a cylinder with a height of 6 inches and a diameter of 5 inches,
prove that 3 whole blocks are needed to create the cheese sculpture.
-/

def volume_of_prism (l w h : ℝ) : ℝ := l * w * h

def volume_of_cylinder (r h : ℝ) : ℝ := Float.pi * r^2 * h

def number_of_whole_blocks_needed (cylinder_volume block_volume : ℝ) : ℕ :=
  (Float.ceil (cylinder_volume / block_volume)).toNat

theorem cheese_sculpture_blocks_needed :
  let block_volume := volume_of_prism 8 3 2
  let cylinder_radius := 5 / 2
  let cylinder_height := 6
  let cylinder_volume := volume_of_cylinder cylinder_radius cylinder_height
  number_of_whole_blocks_needed cylinder_volume block_volume = 3 :=
by
  sorry

end cheese_sculpture_blocks_needed_l179_179064


namespace express_A_using_roster_method_l179_179441

def A := {x : ℕ | ∃ (n : ℕ), 8 / (2 - x) = n }

theorem express_A_using_roster_method :
  A = {0, 1} :=
sorry

end express_A_using_roster_method_l179_179441


namespace min_value_expr_l179_179828

theorem min_value_expr 
  (x y : ℝ)
  (h_parallel : (3 - x) = 2 * y) :
  ∃ (c : ℝ), c = 6 * real.sqrt 3 + 2 ∧ ∀ (x y : ℝ), x + 2 * y = 3 → 
    (3^x + 9^y + 2) ≥ c :=
begin
  sorry
end

end min_value_expr_l179_179828


namespace exponent_of_5_in_30_factorial_l179_179973

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179973


namespace given_function_properties_l179_179808

-- Required Definitions and Hypotheses
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f x = f (x + p)

theorem given_function_properties 
  (f : ℝ → ℝ)
  (h1 : odd_function (λ x, f (2 * x + 1)))
  (h2 : periodic_function (λ x, f (2 * x + 1)) 2) :
  (∀ x, f (x + 1) = -f (-x + 1)) ∧ (periodic_function f 4) :=
by
  sorry

end given_function_properties_l179_179808


namespace exponent_of_5_in_30_factorial_l179_179960

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179960


namespace number_of_correct_answers_l179_179043

theorem number_of_correct_answers (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 110) : c = 34 :=
by
  -- placeholder for proof
  sorry

end number_of_correct_answers_l179_179043


namespace min_value_of_expression_l179_179535

open Real

theorem min_value_of_expression {a b c d e f : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
    (h_sum : a + b + c + d + e + f = 10) :
    (∃ x, x = 44.1 ∧ ∀ y, y = 1 / a + 4 / b + 9 / c + 16 / d + 25 / e + 36 / f → x ≤ y) :=
sorry

end min_value_of_expression_l179_179535


namespace largest_fermat_like_prime_below_300_l179_179084

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m ∈ finset.range (n-2).succ, m + 2 ∣ n → m + 2 = n

def is_fermat_like (n : ℕ) : Prop := ∃ (p : ℕ), nat.prime p ∧ n = 2^p + 1

def largest_fermat_like_prime_below (limit : ℕ) : ℕ :=
  finset.filter (λ n, is_prime n ∧ is_fermat_like n)
  (finset.range limit)
  .max' sorry  -- providing proof that it is non_empty as required, lean 4 statement build check

theorem largest_fermat_like_prime_below_300 : largest_fermat_like_prime_below 300 = 5 :=
by sorry

end largest_fermat_like_prime_below_300_l179_179084


namespace sin_45_eq_l179_179269

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179269


namespace determine_ordered_triple_l179_179818

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

def g (x : ℝ) : ℝ := -f (2 * x) + 3

theorem determine_ordered_triple : (-1, 2, 3) = (-1, 2, 3) := 
by
  sorry

end determine_ordered_triple_l179_179818


namespace players_either_left_handed_or_throwers_l179_179003

theorem players_either_left_handed_or_throwers (total_players throwers : ℕ) (h1 : total_players = 70) (h2 : throwers = 34) (h3 : ∀ n, n = total_players - throwers → 1 / 3 * n = n / 3) :
  ∃ n, n = 46 := 
sorry

end players_either_left_handed_or_throwers_l179_179003


namespace exponent_of_5_in_30_factorial_l179_179972

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179972


namespace proof_collinear_PQR_l179_179712

open EuclideanGeometry 

variable (O1 O2 O3 : Point)
variable (a b m n c d : Line)
variable (P Q R : Point)
variable (A B C O : Point)

noncomputable def proof_problem :=
  -- Conditions
  (distinct : O1 ≠ O2 ∧ O1 ≠ O3 ∧ O2 ≠ O3) 
  (non_overlapping : ¬ (incident O1 O2 ∧ ¬ incident O1 O3 ∧ ¬ incident O2 O3))
  (tangent_intersections: (intersection a b = P) ∧ (intersection m n = Q) ∧ (intersection c d = R))
  -- Question
  (collinear_PQR: collinear P Q R)

theorem proof_collinear_PQR (O1 O2 O3: Point) (a b m n c d: Line) (P Q R: Point)
  (distinct: O1 ≠ O2 ∧ O1 ≠ O3 ∧ O2 ≠ O3)
  (non_overlapping: ¬ (incident O1 O2 ∧ ¬ incident O1 O3 ∧ ¬ incident O2 O3))
  (tangent_intersections: (intersection a b = P) ∧ (intersection m n = Q) ∧ (intersection c d = R))
  : collinear P Q R :=
by
  sorry

end proof_collinear_PQR_l179_179712


namespace average_income_of_all_customers_l179_179100

theorem average_income_of_all_customers
  (n m : ℕ) 
  (a b : ℝ) 
  (customers_responded : n = 50) 
  (wealthiest_count : m = 10) 
  (other_customers_count : n - m = 40) 
  (wealthiest_avg_income : a = 55000) 
  (other_avg_income : b = 42500) : 
  (m * a + (n - m) * b) / n = 45000 := 
by
  -- transforming given conditions into useful expressions
  have h1 : m = 10 := by assumption
  have h2 : n = 50 := by assumption
  have h3 : n - m = 40 := by assumption
  have h4 : a = 55000 := by assumption
  have h5 : b = 42500 := by assumption
  sorry

end average_income_of_all_customers_l179_179100


namespace sin_45_degree_l179_179174

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179174


namespace sin_45_eq_1_div_sqrt_2_l179_179325

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179325


namespace original_strength_of_class_l179_179590

theorem original_strength_of_class :
  ∀ (x : ℕ),
  let avg_age := 40
  let new_students := 8
  let new_avg_age := 32
  let new_avg_decrease := 4 in
  (x + new_students) * (avg_age - new_avg_decrease) = avg_age * x + new_students * new_avg_age 
  → x = 8 :=
by
  intros x avg_age new_students new_avg_age new_avg_decrease h
  sorry

end original_strength_of_class_l179_179590


namespace find_a_and_theta_find_sin_alpha_plus_pi_over_3_l179_179417

noncomputable def f (a θ x : ℝ) : ℝ :=
  (a + 2 * Real.cos x ^ 2) * Real.cos (2 * x + θ)

theorem find_a_and_theta (a θ : ℝ) (h1 : f a θ (Real.pi / 4) = 0)
  (h2 : ∀ x, f a θ (-x) = -f a θ x) :
  a = -1 ∧ θ = Real.pi / 2 :=
sorry

theorem find_sin_alpha_plus_pi_over_3 (α θ : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : f (-1) (Real.pi / 2) (α / 4) = -2 / 5) :
  Real.sin (α + Real.pi / 3) = (4 - 3 * Real.sqrt 3) / 10 :=
sorry

end find_a_and_theta_find_sin_alpha_plus_pi_over_3_l179_179417


namespace triangle_not_identical_l179_179019

-- Define the initial setup and conditions
structure Triangle where
  A : Point
  B : Point
  C : Point

def isAcute (T : Triangle) : Prop :=
  acute_angle (angle T.A T.B T.C) ∧
  acute_angle (angle T.B T.C T.A) ∧ 
  acute_angle (angle T.C T.A T.B)

-- Define the midpoint and median cutting
def isMedianCut (T : Triangle) (D : Point) : Prop :=
  is_midpoint D T.A T.C ∧
  median_cut T.A T.B D ∧
  median_cut T.B T.C D

-- Define the reassembled triangle by Petya
def reassembles (T1 T2 T : Triangle) : Prop :=
  reassembled_from T1 T2 T ∧
  diff_config T1 T2 T

-- The statement to be proven
theorem triangle_not_identical (T : Triangle) (D : Point) :
  isAcute T → isMedianCut T D → ∃ T', reassembles (Triangle.mk T.A T.B D) (Triangle.mk T.B D T.C) T' → ¬ congruent T T' :=
by
  sorry

end triangle_not_identical_l179_179019


namespace maximum_value_of_f_l179_179372

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem maximum_value_of_f : ∃ x : ℝ, f x = 13 :=
by 
  sorry

end maximum_value_of_f_l179_179372


namespace intersection_height_of_poles_l179_179491

-- Definitions to translate the mathematical conditions
def pole1_height : ℝ := 20
def pole2_height : ℝ := 80
def pole_distance : ℝ := 100

-- The statement to be proved
theorem intersection_height_of_poles :
  let line1 := (λ x : ℝ, - (pole1_height / pole_distance) * x + pole1_height)
  let line2 := (λ x : ℝ, (pole2_height / pole_distance) * x)
  let x_intersection := pole1_height * pole_distance / (pole1_height + pole2_height)
  (line1 x_intersection) = (line2 x_intersection) → line1 x_intersection = 16 := 
by
  sorry

end intersection_height_of_poles_l179_179491


namespace exponent_of_5_in_30_factorial_l179_179937

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179937


namespace average_score_of_class_l179_179042

theorem average_score_of_class : 
  ∀ (total_students assigned_students make_up_students : ℕ)
    (assigned_avg_score make_up_avg_score : ℚ),
    total_students = 100 →
    assigned_students = 70 →
    make_up_students = total_students - assigned_students →
    assigned_avg_score = 60 →
    make_up_avg_score = 80 →
    (assigned_students * assigned_avg_score + make_up_students * make_up_avg_score) / total_students = 66 :=
by
  intro total_students assigned_students make_up_students assigned_avg_score make_up_avg_score
  intros h_total_students h_assigned_students h_make_up_students h_assigned_avg_score h_make_up_avg_score
  sorry

end average_score_of_class_l179_179042


namespace sin_45_eq_sqrt2_div_2_l179_179201

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179201


namespace area_of_region_enclosed_by_parabolas_l179_179139

-- Define the given parabolas
def parabola1 (y : ℝ) : ℝ := -3 * y^2
def parabola2 (y : ℝ) : ℝ := 1 - 4 * y^2

-- Define the integral representing the area between the parabolas
noncomputable def areaBetweenParabolas : ℝ :=
  2 * (∫ y in (0 : ℝ)..1, (parabola2 y - parabola1 y))

-- The statement to be proved
theorem area_of_region_enclosed_by_parabolas :
  areaBetweenParabolas = 4 / 3 := 
sorry

end area_of_region_enclosed_by_parabolas_l179_179139


namespace sin_45_eq_l179_179268

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179268


namespace part1_part2_l179_179844

-- Definitions for the conditions
structure Triangle :=
  (A B C : ℝ) -- Angles
  (a b c : ℝ) -- Sides opposite to angles A, B, and C respectively.

-- Hypothesis for part 1
def part1_conditions (T : Triangle) : Prop :=
  T.a * sin T.B - sqrt 3 * T.b * cos T.A = 0

-- Theorem for part 1
theorem part1 (T : Triangle) (h : part1_conditions T) : T.A = π / 3 :=
  sorry

-- Hypothesis for part 2
def part2_conditions (T : Triangle) : Prop :=
  T.a = sqrt 7 ∧ T.b = 2 ∧ T.A = π / 3

-- Theorem for part 2
theorem part2 (T : Triangle) (h : part2_conditions T) : 
  Triangle_area T = (3 * sqrt 3) / 2 :=
  sorry

end part1_part2_l179_179844


namespace points_lie_on_line_l179_179770

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (t + 1) / t
  let y := (t - 1) / t
  x + y = 2 := by
  sorry

end points_lie_on_line_l179_179770


namespace plate_729_driving_days_l179_179509

def plate (n : ℕ) : Prop := n >= 0 ∧ n <= 999

def monday (n : ℕ) : Prop := n % 2 = 1

def sum_digits (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 + d2 + d3

def tuesday (n : ℕ) : Prop := sum_digits n >= 11

def wednesday (n : ℕ) : Prop := n % 3 = 0

def thursday (n : ℕ) : Prop := sum_digits n <= 14

def count_digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100, (n / 10) % 10, n % 10)

def friday (n : ℕ) : Prop :=
  let (d1, d2, d3) := count_digits n
  d1 = d2 ∨ d2 = d3 ∨ d1 = d3

def saturday (n : ℕ) : Prop := n < 500

def sunday (n : ℕ) : Prop := 
  let (d1, d2, d3) := count_digits n
  d1 <= 5 ∧ d2 <= 5 ∧ d3 <= 5

def can_drive (n : ℕ) (day : String) : Prop :=
  plate n ∧ 
  (day = "Monday" → monday n) ∧ 
  (day = "Tuesday" → tuesday n) ∧ 
  (day = "Wednesday" → wednesday n) ∧ 
  (day = "Thursday" → thursday n) ∧ 
  (day = "Friday" → friday n) ∧ 
  (day = "Saturday" → saturday n) ∧ 
  (day = "Sunday" → sunday n)

theorem plate_729_driving_days :
  can_drive 729 "Monday" ∧
  can_drive 729 "Tuesday" ∧
  can_drive 729 "Wednesday" ∧
  ¬ can_drive 729 "Thursday" ∧
  ¬ can_drive 729 "Friday" ∧
  ¬ can_drive 729 "Saturday" ∧
  ¬ can_drive 729 "Sunday" :=
by
  sorry

end plate_729_driving_days_l179_179509


namespace exponent_of_5_in_30_factorial_l179_179943

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179943


namespace largest_number_l179_179444

theorem largest_number (a b c : ℕ) (h1 : c = a + 6) (h2 : b = (a + c) / 2) (h3 : a * b * c = 46332) : 
  c = 39 := 
sorry

end largest_number_l179_179444


namespace radius_of_circle_correct_l179_179494

noncomputable def radius_of_circle (area_of_square : ℝ) : ℝ :=
  let s := real.sqrt area_of_square
  let r := real.sqrt (s ^ 2 / 2)
  r

theorem radius_of_circle_correct :
  radius_of_circle 71.99999999999999 = 6 :=
by
  sorry

end radius_of_circle_correct_l179_179494


namespace calculate_nabla_l179_179469

def operation_nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

theorem calculate_nabla : (operation_nabla (operation_nabla 2 3) (operation_nabla 1 4)) = 1 :=
by
  sorry

end calculate_nabla_l179_179469


namespace probability_of_selecting_one_second_class_product_l179_179863

def total_products : ℕ := 100
def first_class_products : ℕ := 90
def second_class_products : ℕ := 10
def selected_products : ℕ := 3
def exactly_one_second_class_probability : ℚ :=
  (Nat.choose first_class_products 2 * Nat.choose second_class_products 1) / Nat.choose total_products selected_products

theorem probability_of_selecting_one_second_class_product :
  exactly_one_second_class_probability = 0.25 := 
  sorry

end probability_of_selecting_one_second_class_product_l179_179863


namespace sin_45_deg_eq_l179_179311

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179311


namespace subset_transitivity_complement_subset_l179_179108

theorem subset_transitivity {A B C : Set α} (h1 : A ⊆ B) (h2 : B ⊆ C) : A ⊆ C :=
by
  intros x hx
  apply h2
  apply h1
  assumption

theorem complement_subset {U : Set α} {A B : Set α} (h : A ⊆ B) : (U \ B) ⊆ (U \ A) :=
by
  intros x hx
  rw [Set.mem_diff, Set.mem_diff] at hx
  exact And.intro hx.1 (fun hA => hx.2 (h hA))

example (A B C : Set α) (U : Set α) :
  (A ⊆ B ∧ B ⊆ C → A ⊆ C) ∧ (A ⊆ B → (U \ B) ⊆ (U \ A)) :=
by
  exact ⟨subset_transitivity, complement_subset⟩

end subset_transitivity_complement_subset_l179_179108


namespace arrange_numbers_l179_179125

noncomputable def ψ : ℤ := (1 / 2) * (Finset.sum (Finset.range 1006) (λ n, if n % 4 = 0 ∨ n % 4 = 1 then n + 1 else -(n + 1)))

noncomputable def ω : ℤ := Finset.sum (Finset.range 1007) (λ n, if n % 2 = 0 then (2 * n + 1) else -(2 * n + 1))

noncomputable def θ : ℤ := Finset.sum (Finset.range 1008) (λ n, if n % 2 = 0 then (2 * n + 1) else (-(2 * n + 1)))

theorem arrange_numbers :
  θ <= ω ∧ ω <= ψ :=
  sorry

end arrange_numbers_l179_179125


namespace Judy_score_l179_179867

noncomputable def JudyScore (correct incorrect unanswered total : ℕ) : ℤ :=
  2 * correct - incorrect

theorem Judy_score
    (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) (total : ℕ)
    (H1 : correct = 15)
    (H2 : incorrect = 5)
    (H3 : unanswered = 10)
    (H4 : total = 30)
    (H5 : correct + incorrect + unanswered = total) :
    JudyScore correct incorrect unanswered total = 25 :=
by
  -- Definitions of the scoring system
  have ScoringSystem : JudyScore correct incorrect unanswered total = (2 * correct - incorrect),
  from rfl,
  
  -- Apply given values to get the final score
  rw [H1, H2, H3, H4, ScoringSystem],
  norm_num,
  exact rfl

end Judy_score_l179_179867


namespace sin_45_degree_l179_179162

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179162


namespace cover_n_by_tetromino_iff_divisible_4_l179_179670

-- Define the concept of a T-shaped tetromino
def T_tetromino := { 
  coords : List (ℕ × ℕ),
  cover_count : coords.length = 4 /- Each tetromino covers exactly 4 squares -/
}

-- Condition: We require an infinite stack of T-shaped tetrominoes.
def infinite_stack_of_T_tetrominoes := true

-- Condition: No two tetrominoes overlap.
def no_overlap (t1 t2 : T_tetromino) (board : Finset (Fin n × Fin n)) : Prop :=
  let t1_coords := t1.coords.to_finset
  let t2_coords := t2.coords.to_finset
  ∀ x, x ∈ t1_coords → x ∉ t2_coords

-- Condition: No tetromino extends off the board.
def on_board (tetromino : T_tetromino) (board : Finset (Fin n × Fin n)) : Prop :=
  ∀ (x, y) ∈ tetromino.coords, (x < n ∧ y < n)

-- Main theorem: Prove the board can be covered if and only if n is divisible by 4.
theorem cover_n_by_tetromino_iff_divisible_4 (n : ℕ) (board : Finset (Fin n × Fin n)) :
  (∃ tiling : List T_tetromino, ∀ t ∈ tiling, on_board t board ∧ (∀ t2 ∈ tiling, no_overlap t t2 board)) ↔ 4 ∣ n :=
sorry

end cover_n_by_tetromino_iff_divisible_4_l179_179670


namespace solve_trig_eq_l179_179039

theorem solve_trig_eq (x : ℝ) (l : ℤ) :
  cos (2 * x) ≠ 0 → cos (3 * x) ≠ 0 → 
  (sqrt 3 * (1 + (tan (2 * x)) * (tan (3 * x)))) = (tan (2 * x)) * (sec (3 * x)) ↔ 
  (∃ k : ℤ, x = (-1)^k * (π / 3) + π * k) :=
by
  -- Proof omitted
  sorry

end solve_trig_eq_l179_179039


namespace exponent_of_5_in_30_factorial_l179_179980

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179980


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179232

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179232


namespace card_arrangements_l179_179576

theorem card_arrangements (cards : Finset ℕ) (cards_eq : cards = {1, 2, 3, 4, 5, 6}) :
  (∀ s ∈ cards.powerset.filter (λ s, s.card = 5), (s.val.sort (≤) = s.val ∨ s.val.sort (≥) = s.val)) →
  cards.powerset.filter (λ s, s.card = 5).card * 2 = 52 :=
begin
  sorry
end

end card_arrangements_l179_179576


namespace sin_45_eq_sqrt2_div_2_l179_179285

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179285


namespace sarah_took_correct_amount_l179_179748

-- Definition of the conditions
def total_cookies : Nat := 150
def neighbors_count : Nat := 15
def correct_amount_per_neighbor : Nat := 10
def remaining_cookies : Nat := 8
def first_neighbors_count : Nat := 14
def last_neighbor : String := "Sarah"

-- Calculations based on conditions
def total_cookies_taken : Nat := total_cookies - remaining_cookies
def correct_cookies_taken : Nat := first_neighbors_count * correct_amount_per_neighbor
def extra_cookies_taken : Nat := total_cookies_taken - correct_cookies_taken
def sarah_cookies : Nat := correct_amount_per_neighbor + extra_cookies_taken

-- Proof statement: Sarah took 12 cookies
theorem sarah_took_correct_amount : sarah_cookies = 12 := by
  sorry

end sarah_took_correct_amount_l179_179748


namespace sin_45_eq_sqrt2_div_2_l179_179334

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179334


namespace exponent_of_5_in_30_factorial_l179_179982

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179982


namespace sin_45_degree_eq_sqrt2_div_2_l179_179152

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179152


namespace systematic_sampling_selection_l179_179001

theorem systematic_sampling_selection :
  ∃ (selected : List ℕ), 
  (∀ x ∈ selected, x ∈ [1..20]) ∧ 
  selected.length = 4 ∧ 
  (∀ i j, i < j → i < selected.length → j < selected.length → (selected[j] - selected[i]) = (selected[1] - selected[0])) ∧
  selected = [5, 10, 15, 20] :=
by
  sorry

end systematic_sampling_selection_l179_179001


namespace time_B_alone_correct_l179_179709

-- Defining the conditions
variable (m : ℕ) (A_efficiency : ℝ) (B_efficiency : ℝ)

-- Assume m > 20 to avoid division by zero and negative time
axiom m_gt_20 : m > 20

-- Efficiency of person A (works alone)
def efficiency_A : ℝ := 1 / m

-- Combined efficiency of A and B
def combined_efficiency : ℝ := 1 / 20

-- Efficiency of person B
def efficiency_B : ℝ := combined_efficiency - efficiency_A

-- Time taken by B to complete the project alone
def time_B_alone : ℝ := 1 / efficiency_B

-- The goal is to prove that time_B_alone equals the expected result
theorem time_B_alone_correct : time_B_alone = (20 * m) / (m - 20) := by
  sorry

end time_B_alone_correct_l179_179709


namespace exponent_of_5_in_30_factorial_l179_179891

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179891


namespace sin_45_eq_sqrt2_div_2_l179_179212

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179212


namespace AM_perp_MD_l179_179794

-- Define the points and the triangle
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

-- Define conditions as hypotheses
variables {A B C E F D M : Point} 
variables (tri : Triangle A B C)
variables (h_E : E ∈ line_segment A C)
variables (h_F : F ∈ line_segment A B)
variables (h_angles_equal : ∠ B E C = ∠ B F C)
variables (h_intersection : ∃ D, line_segment B E ∩ line_segment C F)
variables (h_circumcircle_intersection : ∃ M, (circumcircle (Triangle A E F)) ∩ (circumcircle (Triangle A B C)) = {A, M})

-- Define the statement that needs to be proven
theorem AM_perp_MD : 
  AM ⊥ MD :=
sorry

end AM_perp_MD_l179_179794


namespace sin_45_degree_l179_179183

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179183


namespace mary_principal_amount_l179_179551

theorem mary_principal_amount (t1 t2 t3 t4:ℕ) (P R:ℕ) :
  (t1 = 2) →
  (t2 = 260) →
  (t3 = 5) →
  (t4 = 350) →
  (P + 2 * P * R = t2) →
  (P + 5 * P * R = t4) →
  P = 200 :=
by
  intros
  sorry

end mary_principal_amount_l179_179551


namespace mini_marshmallows_count_l179_179555

theorem mini_marshmallows_count (total_marshmallows large_marshmallows : ℕ) (h1 : total_marshmallows = 18) (h2 : large_marshmallows = 8) :
  total_marshmallows - large_marshmallows = 10 :=
by 
  sorry

end mini_marshmallows_count_l179_179555


namespace sin_45_eq_sqrt2_div_2_l179_179291

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179291


namespace largest_unformable_sum_l179_179511

namespace Limonia

-- Definitions for our problem conditions
def coin_denoms (n : ℕ) : set ℕ := {3 * n - 1, 6 * n + 1, 6 * n + 4, 6 * n + 7}

-- The target property to prove
theorem largest_unformable_sum (n : ℕ) (hn : n > 0) :
  ∃ L, L = 6 * n^2 + 4 * n - 5 ∧ (∀ k : ℕ, ¬ (k ∈ coin_denoms n → k ≤ L)) :=
by
  sorry

end Limonia

end largest_unformable_sum_l179_179511


namespace sin_45_degree_l179_179184

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179184


namespace parallel_line_eq_l179_179371

theorem parallel_line_eq (x y : ℝ) (c : ℝ) :
  (∀ x y, x - 2 * y - 2 = 0 → x - 2 * y + c = 0) ∧ (x = 1 ∧ y = 0) → c = -1 :=
by
  sorry

end parallel_line_eq_l179_179371


namespace sin_45_deg_eq_l179_179305

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179305


namespace range_of_y0_l179_179545

-- Definition of a point on the parabola
def point_on_parabola (x0 y0 : ℝ) : Prop :=
  x0^2 = 8 * y0

-- Definition of the focus of the parabola
def focus_of_parabola : (ℝ × ℝ) :=
  (0, 2)

-- Definition of the distance between a point and the focus
def distance_from_focus (x0 y0 : ℝ) (F : ℝ × ℝ) : ℝ :=
  real.sqrt ((x0 - F.1)^2 + (y0 - F.2)^2)

theorem range_of_y0 (x0 y0 : ℝ) (F : ℝ × ℝ) 
  (h1 : point_on_parabola x0 y0) 
  (h2 : F = focus_of_parabola)
  (h3 : distance_from_focus x0 y0 F > 4)
  :
  y0 > 2
:=
sorry

end range_of_y0_l179_179545


namespace inverse_proportion_l179_179586

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x = 3) (h3 : y = 15) (h4 : y = -30) : x = -3 / 2 :=
by
  sorry

end inverse_proportion_l179_179586


namespace quadrilateral_properties_l179_179873

-- Definition of the conditions
variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (AB BC CD DA : ℝ)
variables (angle_ADC : ℝ)

-- Specific values given in the problem
def q_AB : AB = 1 := rfl
def q_BC : BC = 9 := rfl
def q_CD : CD = 8 := rfl
def q_DA : DA = 6 := rfl

-- The three statements to be proven
def diagonal_not_perpendicular (AB BC CD DA : ℝ) : Prop :=
  (1 * 1 + 8 * 8 ≠ 6 * 6 + 9 * 9)

def angle_ADC_strictly_less_than_90 (angle_ADC : ℝ) : Prop :=
  angle_ADC < 90

def not_isosceles_triangle_BCD (BC CD : ℝ) : Prop :=
  BC ≠ CD ∧ (BC ≠ sqrt (BC ^ 2 + CD ^ 2 - 2 * BC * CD * cos (π/3)))

theorem quadrilateral_properties (AB BC CD DA : ℝ) (angle_ADC : ℝ)
  (h1 : AB = 1) (h2 : BC = 9) (h3 : CD = 8) (h4 : DA = 6) :
  diagonal_not_perpendicular AB BC CD DA ∧
  angle_ADC_strictly_less_than_90 angle_ADC ∧
  not_isosceles_triangle_BCD BC CD :=
by
  sorry

end quadrilateral_properties_l179_179873


namespace base_of_numeral_system_l179_179639

theorem base_of_numeral_system (x : ℕ) : 3 * x^3 + 3 * x^2 + 6 * x + 2 = 1728 → x = 8 :=
by
  assume h : 3 * x^3 + 3 * x^2 + 6 * x + 2 = 1728
  sorry

end base_of_numeral_system_l179_179639


namespace joan_spent_14_half_dollars_on_thursday_l179_179560

def joan_spent_half_dollars (total_half_dollars_wednesday : ℕ)
                            (total_spent : ℝ)
                            (cost_per_half_dollar : ℝ) : ℕ :=
  (total_spent - total_half_dollars_wednesday * cost_per_half_dollar) / cost_per_half_dollar

theorem joan_spent_14_half_dollars_on_thursday : joan_spent_half_dollars 4 9 0.5 = 14 :=
by
  -- Define the necessary conditions:
  let total_half_dollars_wednesday := 4
  let total_spent := 9.0 : ℝ
  let cost_per_half_dollar := 0.5 : ℝ

  -- Derive the number of half-dollars spent on Thursday:
  let spent_on_thursday := (total_spent - total_half_dollars_wednesday * cost_per_half_dollar) / cost_per_half_dollar

  -- Assert that this value is 14:
  show spent_on_thursday = 14 from sorry

end joan_spent_14_half_dollars_on_thursday_l179_179560


namespace monthly_rent_of_shop_l179_179096

theorem monthly_rent_of_shop
  (length width : ℕ)
  (annual_rent_per_sq_ft : ℕ)
  (length_def : length = 18)
  (width_def : width = 22)
  (annual_rent_per_sq_ft_def : annual_rent_per_sq_ft = 68) :
  (18 * 22 * 68) / 12 = 2244 := 
by
  sorry

end monthly_rent_of_shop_l179_179096


namespace max_a_condition_l179_179463

theorem max_a_condition (a : ℝ) :
  (∀ x : ℝ, x < a → |x| > 2) ∧ (∃ x : ℝ, |x| > 2 ∧ ¬ (x < a)) →
  a ≤ -2 :=
by 
  sorry

end max_a_condition_l179_179463


namespace legs_per_dragon_l179_179111

variables (x y dragon_legs : ℕ)

-- Given Conditions
def total_heads : ℕ := 31
def total_legs : ℕ := 286
def creature_legs : ℕ := 34
def dragon_heads : ℕ := 3

-- Problem Statement
theorem legs_per_dragon : dragon_legs = 6 :=
  -- Proof Outline
  by
  -- Definitions and Conditions
  have h1 : creature_legs * x + dragon_legs * y = total_legs, by sorry
  have h2 : x + dragon_heads * y = total_heads, by sorry
  -- Isolate Variables and Solve
  have h3 : x = total_heads - dragon_heads * y, by sorry
  -- Substituting and simplifying the equations
  have h4 : creature_legs * (total_heads - dragon_heads * y) + dragon_legs * y = total_legs, by sorry
  -- After simplifying as per the solution
  have h5 : creature_legs * total_heads - 102 * y + dragon_legs * y = total_legs, by sorry
  have h6 : 768 = (dragon_legs - 102) * y, by sorry
  -- We need to find integer y so that the equation holds, proving dragon_legs == 6
  sorry

end legs_per_dragon_l179_179111


namespace points_enclosed_in_circle_l179_179678

open Set

variable (points : Set (ℝ × ℝ))
variable (radius : ℝ)
variable (h1 : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points → 
  ∃ (c : ℝ × ℝ), dist c A ≤ radius ∧ dist c B ≤ radius ∧ dist c C ≤ radius)

theorem points_enclosed_in_circle
  (h1 : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points →
    ∃ (c : ℝ × ℝ), dist c A ≤ 1 ∧ dist c B ≤ 1 ∧ dist c C ≤ 1) :
  ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ points → dist c p ≤ 1 :=
sorry

end points_enclosed_in_circle_l179_179678


namespace cos_double_angle_l179_179784

theorem cos_double_angle (α β : ℝ) (h1 : Real.sin (α - β) = 1 / 3) (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_double_angle_l179_179784


namespace sin_45_deg_eq_l179_179303

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179303


namespace cost_of_five_plastic_chairs_l179_179628

theorem cost_of_five_plastic_chairs (C T : ℕ) (h1 : 3 * C = T) (h2 : T + 2 * C = 55) : 5 * C = 55 :=
by {
  sorry
}

end cost_of_five_plastic_chairs_l179_179628


namespace monthly_profit_10000_yuan_impossible_daily_profit_15000_yuan_max_profit_at_65_yuan_max_profit_value_l179_179069

theorem monthly_profit_10000_yuan (x : ℝ) : 
  (40 + x - 30) * (600 - 10 * x) = 10000 ↔ (x = 40 ∨ x = 10) :=
sorry

theorem impossible_daily_profit_15000_yuan (x : ℝ) : 
  ¬ ∃ x, (40 + x - 30) * (600 - 10 * x) = 15000 :=
sorry

theorem max_profit_at_65_yuan (x : ℝ) : 
  let profit := (x - 30) * (600 - 10 * (x - 40)) in
  ∀ x, profit ≤ (65 - 30) * (600 - 10 * (65 - 40)) :=
sorry

theorem max_profit_value : 
  let x := 65 in
  (x - 30) * (600 - 10 * (x - 40)) = 12250 :=
sorry

end monthly_profit_10000_yuan_impossible_daily_profit_15000_yuan_max_profit_at_65_yuan_max_profit_value_l179_179069


namespace min_value_of_f_l179_179778

def f (x : ℕ) (hx : 0 < x) : ℚ := (x^2 + 33)/(x : ℚ)

theorem min_value_of_f : ∃ x : ℕ, 0 < x ∧ ∀ y : ℕ, 0 < y → f y (by assumption) ≥ (23/2 : ℚ) :=
sorry

end min_value_of_f_l179_179778


namespace total_votes_l179_179663

theorem total_votes (V : ℕ) (h1 : 0.60 * V - 0.40 * V = 1504) : V = 7520 :=
sorry

end total_votes_l179_179663


namespace number_of_common_tangents_l179_179793

-- Define circle M
def circleM (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0

-- Define circle N
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- The statement to prove: the number of common tangent lines is 2
theorem number_of_common_tangents : 
  ∀ x y : ℝ, 
    (circleM x y ∧ circleN x y → 2) :=
sorry

end number_of_common_tangents_l179_179793


namespace tetrahedra_different_dihedral_sums_l179_179574

open Real

noncomputable def sum_dihedral_angles(tetrahedron: Type) : Real := sorry

theorem tetrahedra_different_dihedral_sums :
  ∃ (T1 T2 : Type) (AB BC BD : Type) (M : Type),
    (AB = BC ∧ BC = BD ∧ BD = AB) ∧
    (is_orthogonal_to_plane AB BC BD) ∧
    (midpoint M CD BC) ∧
    (is_isosceles_triangle BMC) ∧
    (BM ⊥ CD) ∧
    (sum_dihedral_angles T1 ≠ sum_dihedral_angles T2) :=
  sorry

end tetrahedra_different_dihedral_sums_l179_179574


namespace bowling_average_decrease_l179_179082

theorem bowling_average_decrease 
  (original_average : ℚ) 
  (wickets_last_match : ℚ) 
  (runs_last_match : ℚ) 
  (original_wickets : ℚ) 
  (original_total_runs : ℚ := original_wickets * original_average) 
  (new_total_wickets : ℚ := original_wickets + wickets_last_match) 
  (new_total_runs : ℚ := original_total_runs + runs_last_match)
  (new_average : ℚ := new_total_runs / new_total_wickets) :
  original_wickets = 85 → original_average = 12.4 → wickets_last_match = 5 → runs_last_match = 26 → new_average = 12 →
  original_average - new_average = 0.4 := 
by 
  intros 
  sorry

end bowling_average_decrease_l179_179082


namespace sin_45_eq_sqrt2_div_2_l179_179294

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179294


namespace check_triangle_345_l179_179036

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem check_triangle_345 : satisfies_triangle_inequality 3 4 5 := by
  sorry

end check_triangle_345_l179_179036


namespace sin_45_deg_l179_179193

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179193


namespace min_selection_contains_coprimes_l179_179390

/-- There must be at least two numbers that are relatively prime in any selection of 12 numbers from the set {1, 2, ..., 100} -/
theorem min_selection_contains_coprimes :
  ∀ (s : Finset ℕ), (s ⊆ Finset.range 101) → Finset.card s = 12 → 
  ∃ a b ∈ s, Nat.gcd a b = 1 :=
begin
  intros s hs hcard,
  sorry
end

end min_selection_contains_coprimes_l179_179390


namespace range_of_m_l179_179797

theorem range_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : 0 < m) 
  (h4 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : m ≥ 4 :=
sorry

end range_of_m_l179_179797


namespace sin_45_degree_eq_sqrt2_div_2_l179_179147

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179147


namespace intersect_parabolas_l179_179635

theorem intersect_parabolas :
  ∀ (x y : ℝ),
    ((y = 2 * x^2 - 7 * x + 1 ∧ y = 8 * x^2 + 5 * x + 1) ↔ 
     ((x = -2 ∧ y = 23) ∨ (x = 0 ∧ y = 1))) :=
by sorry

end intersect_parabolas_l179_179635


namespace exponent_of_5_in_30_factorial_l179_179932

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179932


namespace sin_45_degree_l179_179176

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179176


namespace f_leq_2x_l179_179599

noncomputable def f : ℝ → ℝ := sorry
axiom f_nonneg {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : 0 ≤ f x
axiom f_one : f 1 = 1
axiom f_superadditive {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hxy : x + y ≤ 1) : f (x + y) ≥ f x + f y

-- The theorem statement to be proved
theorem f_leq_2x {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : f x ≤ 2 * x := sorry

end f_leq_2x_l179_179599


namespace sin_45_eq_1_div_sqrt_2_l179_179324

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179324


namespace combinations_toppings_l179_179557

theorem combinations_toppings (n k : ℕ) (h_n: n = 9) (h_k: k = 3) : (nat.choose n k) = 84 :=
by
  rw [h_n, h_k]
  have fact_nine : 9! = 362880 := by norm_num
  have fact_six : 6! = 720 := by norm_num
  have fact_three : 3! = 6 := by norm_num
  rw [nat.choose_eq_factorial_div_factorial, fact_nine, fact_three, fact_six]
  norm_num
  sorry

end combinations_toppings_l179_179557


namespace area_of_triangle_O_P1_P2_l179_179547

-- Define the hyperbola and its conditions.
def hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x^2 / 9 - y^2 / 16 = 1

-- Define the relationship condition of the vectors.
def vector_ratio (P1 P2 P : ℝ × ℝ) : Prop :=
  let x1 := P1.1 in let x2 := P2.1 in let y1 := P1.2 in let y2 := P2.2 in
  (P1.1 - P.1) * (P.1 - P2.1) = 3

-- Define the distance formula for points from O.
def distance_from_origin (P : ℝ × ℝ) : ℝ :=
  real.sqrt (P.1^2 + P.2^2)

-- Define the angle sine formula.
def sine_of_angle (a b : ℝ) : ℝ :=
  a / real.sqrt (a^2 + b^2)

-- Formalize the main theorem statement.
theorem area_of_triangle_O_P1_P2
 (P P1 P2 : ℝ × ℝ)
 (h1 : hyperbola P)
 (h2 : vector_ratio P1 P2 P)
 (h3 : distance_from_origin P1 * distance_from_origin P2 = 100 / 3)
 (h4 : sine_of_angle 24 7 = 24 / 25) :
  1/2 * distance_from_origin P1 * distance_from_origin P2 * 24 / 25 = 16 :=
by
  sorry

end area_of_triangle_O_P1_P2_l179_179547


namespace exponent_of_5_in_30_factorial_l179_179938

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179938


namespace prime_exponent_of_5_in_30_factorial_l179_179907

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179907


namespace problem1_problem2_l179_179718

-- Problem 1: Show that (a - b)^2 - b(b - 2a) = a^2 for all real numbers a and b.
theorem problem1 (a b : ℝ) : (a - b)^2 - b * (b - 2 * a) = a^2 := sorry

-- Problem 2: Show that (x^2 - 4 * x + 4) / (x^2 - x) ÷ (x + 1 - 3 / (x - 1)) = (x - 2) / (x * (x + 2)) for x ≠ 0, x ≠ 1, and x ≠ 2.
theorem problem2 (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) (hx2 : x ≠ 2) : 
  (x^2 - 4 * x + 4) / (x^2 - x) ÷ (x + 1 - 3 / (x - 1)) = (x - 2) / (x * (x + 2)) := sorry

end problem1_problem2_l179_179718


namespace max_f_value_l179_179379

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 13 :=
sorry

end max_f_value_l179_179379


namespace exponent_of_5_in_30_factorial_l179_179970

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179970


namespace calculate_third_discount_pct_l179_179677

variable (actual_price final_price first_discount second_discount third_discount_pct : ℝ)
variable (third_discount_pct_correct : Prop)

def third_discount_pct_def : Prop :=
  actual_price = 10000 ∧
  first_discount = 20 ∧
  second_discount = 10 ∧
  final_price = 6840 ∧
  third_discount_pct_correct = 5

axiom given_conditions : third_discount_pct_def actual_price final_price first_discount second_discount third_discount_pct_correct

theorem calculate_third_discount_pct : third_discount_pct_correct = 5 :=
by
  sorry

end calculate_third_discount_pct_l179_179677


namespace problem_statement_l179_179476

theorem problem_statement (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 2) : a^2 + b^2 = 29 := 
by
  sorry

end problem_statement_l179_179476


namespace exists_x0_log2_3_ge_1_l179_179824

theorem exists_x0_log2_3_ge_1 : ∃ x0 ∈ Set.Ici (1 : ℝ), Real.log 2 3 ^ x0 ≥ 1 :=
by
  have h : Real.log 2 3 > 1 := sorry
  use 1
  simp [h]

end exists_x0_log2_3_ge_1_l179_179824


namespace sin_45_degree_l179_179275

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179275


namespace sin_45_deg_l179_179188

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179188


namespace maximum_value_l179_179762

-- Definitions based on the conditions in the problem
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2

def interval : set ℝ := set.Icc (-2 : ℝ) (2 : ℝ)

theorem maximum_value : 
  ∃ x ∈ interval, (∀ y ∈ interval, f y ≤ f x) ∧ f x = 7 :=
by
  sorry

end maximum_value_l179_179762


namespace exponent_of_5_in_30_factorial_l179_179997

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179997


namespace distribution_ways_l179_179454

-- Definitions based on the conditions from part a)
def distinguishable_balls : ℕ := 6
def indistinguishable_boxes : ℕ := 4

-- The theorem to prove the question equals the correct answer given the conditions
theorem distribution_ways (n : ℕ) (k : ℕ) (h_n : n = distinguishable_balls) (h_k : k = indistinguishable_boxes) : 
  number_of_distributions n k = 262 := 
sorry

end distribution_ways_l179_179454


namespace how_many_months_to_buy_tv_l179_179051

-- Definitions based on given conditions
def monthly_income : ℕ := 30000
def food_expenses : ℕ := 15000
def utilities_expenses : ℕ := 5000
def other_expenses : ℕ := 2500

def total_expenses := food_expenses + utilities_expenses + other_expenses
def current_savings : ℕ := 10000
def tv_cost : ℕ := 25000
def monthly_savings := monthly_income - total_expenses

-- Theorem statement based on the problem
theorem how_many_months_to_buy_tv 
    (H_income : monthly_income = 30000)
    (H_food : food_expenses = 15000)
    (H_utilities : utilities_expenses = 5000)
    (H_other : other_expenses = 2500)
    (H_savings : current_savings = 10000)
    (H_tv_cost : tv_cost = 25000)
    : (tv_cost - current_savings) / monthly_savings = 2 :=
by
  sorry

end how_many_months_to_buy_tv_l179_179051


namespace Cedar_school_earnings_l179_179582

noncomputable def total_earnings_Cedar_school : ℝ :=
  let total_payment := 774
  let total_student_days := 6 * 4 + 5 * 6 + 3 * 10
  let daily_wage := total_payment / total_student_days
  let Cedar_student_days := 3 * 10
  daily_wage * Cedar_student_days

theorem Cedar_school_earnings :
  total_earnings_Cedar_school = 276.43 :=
by
  sorry

end Cedar_school_earnings_l179_179582


namespace sin_45_eq_l179_179265

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179265


namespace cost_of_Striploin_is_71_82_l179_179095

noncomputable def cost_of_NY_Striploin : ℝ :=
  let T := 140
  let tax := 0.10
  let W := 10
  let G := 41
  let meal_before_wine := (T - G - 10 + tax) / (1 + tax)
  meal_before_wine - W

theorem cost_of_Striploin_is_71_82 : cost_of_NY_Striploin ≈ 71.82 := by
  sorry

end cost_of_Striploin_is_71_82_l179_179095


namespace exponent_of_5_in_30_fact_l179_179881

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179881


namespace sin_45_eq_sqrt_two_over_two_l179_179243

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179243


namespace exists_points_on_curve_through_center_l179_179685

theorem exists_points_on_curve_through_center (Γ : set (ℝ × ℝ)) (O : ℝ × ℝ) (square : set (ℝ × ℝ))
  (hΓ : Γ ⊆ square) (hO : O = (0, 0)) (h_eq_area : ∃ A B, Γ divides square into regions of equal area) :
  ∃ A B ∈ Γ, line_through A B O :=
sorry

end exists_points_on_curve_through_center_l179_179685


namespace nth_equation_l179_179558

theorem nth_equation (n : ℕ) : (2 * n + 2) ^ 2 - (2 * n) ^ 2 = 4 * (2 * n + 1) :=
by
  sorry

end nth_equation_l179_179558


namespace log_mono_decreasing_iff_l179_179615

theorem log_mono_decreasing_iff
  (m : ℝ) :
  (-3 < m ∧ m < 0) ↔
  ∀ x y : ℝ, (x <= 1 ∧ y <= 1 ∧ x < y) → ln (m * x + 3) > ln (m * y + 3) :=
by sorry

end log_mono_decreasing_iff_l179_179615


namespace sin_45_degree_l179_179179

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179179


namespace sin_45_eq_l179_179261

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179261


namespace sin_45_degree_eq_sqrt2_div_2_l179_179155

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179155


namespace maximum_of_f_in_interval_l179_179605

noncomputable def f (x : ℝ) : ℝ := 1 / 3 * x ^ 3 - 2 * x ^ 2 + 3 * x - 2

theorem maximum_of_f_in_interval : ∃ x ∈ set.Icc 0 2, ∀ y ∈ set.Icc 0 2, f y ≤ f x ∧ f x = -2 / 3 :=
by
  sorry

end maximum_of_f_in_interval_l179_179605


namespace sin_45_degree_l179_179160

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179160


namespace exponent_of_5_in_30_factorial_l179_179958

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179958


namespace students_did_not_pass_l179_179562

theorem students_did_not_pass (total_students : ℕ) (pass_percentage : ℝ) 
  (total_students_eq : total_students = 804) (pass_percentage_eq : pass_percentage = 0.75) : 
  ∃ (students_did_not_pass : ℕ), students_did_not_pass = total_students - nat.floor (total_students * pass_percentage) :=
by
  sorry

end students_did_not_pass_l179_179562


namespace minimum_a_l179_179851

theorem minimum_a (a : ℝ) : (∀ x y : ℝ, 0 < x → 0 < y → (x + y) * (a / x + 4 / y) ≥ 16) → a ≥ 4 :=
by
  intros h
  -- We would provide a detailed mathematical proof here, but we use sorry for now.
  sorry

end minimum_a_l179_179851


namespace exponent_of_5_in_30_factorial_l179_179971

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179971


namespace eleven_pow_603_mod_500_eq_331_l179_179648

theorem eleven_pow_603_mod_500_eq_331 : 11^603 % 500 = 331 := by
  sorry

end eleven_pow_603_mod_500_eq_331_l179_179648


namespace annual_simple_interest_rate_l179_179674

noncomputable def principal : ℝ := 200
noncomputable def total_repayment : ℝ := 240
noncomputable def time_period : ℝ := 3

theorem annual_simple_interest_rate :
  let interest := total_repayment - principal,
      rate := (interest / (principal * time_period)) * 100
  in rate ≈ 7 :=
by
  sorry

end annual_simple_interest_rate_l179_179674


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179230

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179230


namespace sin_45_eq_one_div_sqrt_two_l179_179217

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179217


namespace value_of_expression_l179_179467

noncomputable def largestNegativeInteger : Int := -1

theorem value_of_expression (a b x y : ℝ) (m : Int)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = largestNegativeInteger) :
  2023 * (a + b) + 3 * |m| - 2 * (x * y) = 1 :=
by
  sorry

end value_of_expression_l179_179467


namespace sin_45_eq_sqrt2_div_2_l179_179333

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179333


namespace smallest_digit_is_one_l179_179474

-- Given a 4-digit integer x.
def four_digit_integer (x : ℕ) : Prop :=
  1000 ≤ x ∧ x < 10000

-- Define function for the product of digits of x.
def product_of_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := (x / 1000) % 10
  d1 * d2 * d3 * d4

-- Define function for the sum of digits of x.
def sum_of_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := (x / 1000) % 10
  d1 + d2 + d3 + d4

-- Assume p is a prime number.
def is_prime (p : ℕ) : Prop :=
  ¬ ∃ d, d ∣ p ∧ d ≠ 1 ∧ d ≠ p

-- Proof problem: Given conditions for T(x) and S(x),
-- prove that the smallest digit in x is 1.
theorem smallest_digit_is_one (x p k : ℕ) (h1 : four_digit_integer x)
  (h2 : is_prime p) (h3 : product_of_digits x = p^k)
  (h4 : sum_of_digits x = p^p - 5) : 
  ∃ d1 d2 d3 d4, d1 <= d2 ∧ d1 <= d3 ∧ d1 <= d4 ∧ d1 = 1 
  ∧ (d1 + d2 + d3 + d4 = p^p - 5) 
  ∧ (d1 * d2 * d3 * d4 = p^k) := 
sorry

end smallest_digit_is_one_l179_179474


namespace average_rate_of_interest_is_5133_percent_l179_179703

noncomputable def average_rate_of_interest (total : ℕ) (rate1 rate2 : ℚ) (fee : ℚ) (x : ℚ)
  (h1 : total = 3000)
  (h2 : rate1 = 0.05)
  (h3 : rate2 = 0.07)
  (h4 : fee = 18)
  (h5 : x = 1100)
  (h6 : 0.05 * (3000 - x) - 18 = 0.07 * x) :
  ℚ :=
(let interest1 := rate1 * (total - x) - fee in
 let interest2 := rate2 * x in
 let total_interest := interest1 + interest2 in
 total_interest / total * 100)

theorem average_rate_of_interest_is_5133_percent :
  average_rate_of_interest 3000 0.05 0.07 18 1100
    (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by simp [sub_eq_add_neg]; norm_num) = 5.133 :=
by sorry

end average_rate_of_interest_is_5133_percent_l179_179703


namespace trajectory_of_M_l179_179854

theorem trajectory_of_M
  (x y : ℝ)
  (h : Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) :
  (x^2 / 16) - (y^2 / 9) = 1 :=
sorry

end trajectory_of_M_l179_179854


namespace range_of_a_min_value_reciprocals_l179_179434

noncomputable def f (x a : ℝ) : ℝ := |x - 2| + |x - a^2|

theorem range_of_a (a : ℝ) : (∃ x : ℝ, f x a ≤ a) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

theorem min_value_reciprocals (m n a : ℝ) (h : m + 2 * n = a) (ha : a = 2) : (1/m + 1/n) ≥ (3/2 + Real.sqrt 2) := by
  sorry

end range_of_a_min_value_reciprocals_l179_179434


namespace total_students_in_class_l179_179499

theorem total_students_in_class :
  ∃ x, (10 * 90 + 15 * 80 + x * 60) / (10 + 15 + x) = 72 → 10 + 15 + x = 50 :=
by
  -- Providing an existence proof and required conditions
  use 25
  intro h
  sorry

end total_students_in_class_l179_179499


namespace f_comp_f_one_fourth_l179_179427

def f : ℝ → ℝ := 
λ x, if x > 0 then Real.log x / Real.log 2 else 1 / 3^x

theorem f_comp_f_one_fourth : f (f (1 / 4)) = 9 :=
  sorry

end f_comp_f_one_fourth_l179_179427


namespace hyeongjun_older_sister_age_l179_179461

-- Define the ages of Hyeongjun and his older sister
variables (H S : ℕ)

-- Conditions
def age_gap := S = H + 2
def sum_of_ages := H + S = 26

-- Theorem stating that the older sister's age is 14
theorem hyeongjun_older_sister_age (H S : ℕ) (h1 : age_gap H S) (h2 : sum_of_ages H S) : S = 14 := 
by 
  sorry

end hyeongjun_older_sister_age_l179_179461


namespace athlete_heartbeats_during_race_l179_179708

variable (heart_rate_per_min : ℕ) (pace_min_per_mile : ℕ) (distance_miles : ℕ)

def total_race_time (pace : ℕ) (distance : ℕ) : ℕ := pace * distance
def total_heartbeats (heart_rate : ℕ) (time : ℕ) : ℕ := heart_rate * time

theorem athlete_heartbeats_during_race : 
  heart_rate_per_min = 120 →
  pace_min_per_mile = 4 →
  distance_miles = 100 →
  total_heartbeats heart_rate_per_min (total_race_time pace_min_per_mile distance_miles) = 48000 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rw [total_race_time, total_heartbeats]
  rw [Nat.mul_comm 4 100]
  norm_num
  sorry

end athlete_heartbeats_during_race_l179_179708


namespace exponent_of_5_in_30_factorial_l179_179925

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179925


namespace sin_45_eq_one_div_sqrt_two_l179_179219

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179219


namespace sin_45_degree_eq_sqrt2_div_2_l179_179148

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179148


namespace K_N_D_collinear_l179_179517

-- Definitions and conditions
variable {Point : Type}
variable {A B C D N K : Point}

-- Conditions based on the problem statement
axiom rhombus_ABCD : ∀ P : Point, ∃ Q R S, true -- Stub for rhombus condition (to be defined properly)
axiom triangle_BCN_is_equilateral : equilateral_triangle B C N
axiom angle_bisector_BL_of_ABC_intersects_AC_at_K : is_angle_bisector B L (∠ A B N) ∧ lies_on_diagonal K A C

-- Lean implementation of the collinearity proof
theorem K_N_D_collinear 
    (rhombus_ABCD : ∀ P, ∃ Q R S, true)
    (equilateral_triangle B C N)
    (is_angle_bisector B L (∠ A B N))
    (lies_on_diagonal K A C) 
    : collinear {K, N, D} :=
sorry

end K_N_D_collinear_l179_179517


namespace time_without_moving_walkway_l179_179659

/--
Assume a person walks from one end to the other of a 90-meter long moving walkway at a constant rate in 30 seconds, assisted by the walkway. When this person reaches the end, they reverse direction and continue walking with the same speed, but this time it takes 120 seconds because the person is traveling against the direction of the moving walkway.

Prove that if the walkway were to stop moving, it would take this person 48 seconds to walk from one end of the walkway to the other.
-/
theorem time_without_moving_walkway : 
  ∀ (v_p v_w : ℝ),
  (v_p + v_w) * 30 = 90 →
  (v_p - v_w) * 120 = 90 →
  90 / v_p = 48 :=
by
  intros v_p v_w h1 h2
  have hpw := eq_of_sub_eq_zero (sub_eq_zero.mpr h1)
  have hmw := eq_of_sub_eq_zero (sub_eq_zero.mpr h2)
  sorry

end time_without_moving_walkway_l179_179659


namespace exponent_of_5_in_30_fact_l179_179879

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179879


namespace surface_area_of_cube_edge_8_l179_179650

-- Definition of surface area of a cube
def surface_area_of_cube (edge_length : ℕ) : ℕ :=
  6 * (edge_length * edge_length)

-- Theorem to prove the surface area for a cube with edge length of 8 cm is 384 cm²
theorem surface_area_of_cube_edge_8 : surface_area_of_cube 8 = 384 :=
by
  -- The proof will be inserted here. We use sorry to indicate the missing proof.
  sorry

end surface_area_of_cube_edge_8_l179_179650


namespace function_graph_intersection_l179_179447

theorem function_graph_intersection (f : ℝ → ℝ) :
  (∃ y : ℝ, f 1 = y) → (∃! y : ℝ, f 1 = y) :=
by
  sorry

end function_graph_intersection_l179_179447


namespace sin_45_eq_sqrt_two_over_two_l179_179255

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179255


namespace length_of_AP_l179_179495

variables {x : ℝ} (M B C P A : Point) (circle : Circle)
  (BC AB MP : Line)

-- Definitions of conditions
def is_midpoint_of_arc (M B C : Point) (circle : Circle) : Prop := sorry
def is_perpendicular (MP AB : Line) (P : Point) : Prop := sorry
def chord_length (BC : Line) (length : ℝ) : Prop := sorry
def segment_length (BP : Line) (length : ℝ) : Prop := sorry

-- Prove statement
theorem length_of_AP
  (h1 : is_midpoint_of_arc M B C circle)
  (h2 : is_perpendicular MP AB P)
  (h3 : chord_length BC (2 * x))
  (h4 : segment_length BP (3 * x)) :
  ∃AP : Line, segment_length AP (2 * x) :=
sorry

end length_of_AP_l179_179495


namespace sin_45_eq_sqrt2_div_2_l179_179209

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179209


namespace right_triangle_angle_l179_179507

theorem right_triangle_angle {A B C : ℝ} (hABC : A + B + C = 180) (hC : C = 90) (hA : A = 70) : B = 20 :=
sorry

end right_triangle_angle_l179_179507


namespace bug_returns_to_A_after_6_meters_l179_179530

def tetrahedron_bug_probability (Q : ℕ → ℚ) : Prop :=
  (∀ n, Q(n + 1) = 1 / 4 * Q(n) + 1 / 4 * (1 - Q(n))) ∧ Q(0) = 1

theorem bug_returns_to_A_after_6_meters (Q : ℕ → ℚ) :
  tetrahedron_bug_probability Q → Q 6 = 354 / 729 :=
by
  -- proof to be provided
  sorry

end bug_returns_to_A_after_6_meters_l179_179530


namespace largest_store_visits_l179_179004

theorem largest_store_visits (S : Finset (Fin 8)) (p : Finset ℕ) 
    (total_visits : ∑ x in p, x = 23) 
    (exact_two_stores_visited : (∑ x in (Finset.range 8).filter (λ x, x = 2), x) = 16) 
    (total_shoppers : p.card = 12) 
    (everyone_visits_at_least_one : ∀ x ∈ p, x ≥ 1) : 
    ∃ (max_visits : ℕ), max_visits = 7 :=
by
  sorry

end largest_store_visits_l179_179004


namespace strictly_increasing_condition_max_min_values_l179_179536

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) := -1/3 * x^3 + 1/2 * x^2 + 2 * a * x

-- Define the derivative of f
def f_prime (x : ℝ) (a : ℝ) := -x^2 + x + 2 * a

-- 1. Prove the condition for strictly increasing interval on (2/3, +∞)
theorem strictly_increasing_condition (a : ℝ) : 
  (∀ x ≥ (2/3 : ℝ), f_prime x a > 0) ↔ (a > -1/9) :=
by
  sorry -- Proof would be provided here.

-- 2. Prove maximum and minimum values when a = 1
theorem max_min_values (a : ℝ) (h : a = 1) :
  ∃ x_max x_min : ℝ, 
    (x_max ∈ set.Icc (1 : ℝ) 4 ∧ f x_max 1 = 10/3) ∧
    (x_min ∈ set.Icc (1 : ℝ) 4 ∧ f x_min 1 = -16/3) :=
by
  use 2, 4
  sorry -- Proof would be provided here.

end strictly_increasing_condition_max_min_values_l179_179536


namespace f_of_5_l179_179396

noncomputable def f : ℝ → ℝ := sorry

-- Hypothesis: ∀ x, f(10^x) = x
axiom f_property : ∀ x : ℝ, f(10^x) = x

-- Goal: f(5) = log10 5
theorem f_of_5 : f 5 = Real.log10 5 :=
by sorry

end f_of_5_l179_179396


namespace exponent_of_5_in_30_factorial_l179_179954

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179954


namespace speed_of_first_train_l179_179016

-- Define the conditions
def distance_pq := 110 -- km
def speed_q := 25 -- km/h
def meet_time := 10 -- hours from midnight
def start_p := 7 -- hours from midnight
def start_q := 8 -- hours from midnight

-- Define the total travel time for each train
def travel_time_p := meet_time - start_p -- hours
def travel_time_q := meet_time - start_q -- hours

-- Define the distance covered by each train
def distance_covered_p (V_p : ℕ) : ℕ := V_p * travel_time_p
def distance_covered_q := speed_q * travel_time_q

-- Theorem to prove the speed of the first train
theorem speed_of_first_train (V_p : ℕ) : distance_covered_p V_p + distance_covered_q = distance_pq → V_p = 20 :=
sorry

end speed_of_first_train_l179_179016


namespace chord_length_and_midpoint_properties_fixed_point_exists_P_l179_179402

noncomputable def F := (-2 : ℝ, 0 : ℝ)

def is_on_circle (G : ℝ × ℝ) :=
  (G.fst + 4)^2 + G.snd^2 = 16

def is_midpoint (G T : ℝ × ℝ) :=
  2 * G.fst = G.fst + T.fst ∧ 2 * G.snd = G.snd + T.snd

def line_intersects_x_4 (G : ℝ × ℝ) (T : ℝ × ℝ) :=
  T.fst = -4 ∧ ∃ k : ℝ, k * (-4 - G.fst) + G.snd = 0 ∧ T.snd = k * (-4 - G.fst) + G.snd

theorem chord_length_and_midpoint_properties :
  ∀ (G : ℝ × ℝ), is_on_circle G →
  ∃ T : ℝ × ℝ, line_intersects_x_4 G T ∧ is_midpoint G T →
  true := sorry

theorem fixed_point_exists_P :
  ∃ P : ℝ × ℝ, ∀ (G : ℝ × ℝ), is_on_circle G →
  real.sqrt ((G.fst - P.fst)^2 + (G.snd - P.snd)^2) = 
  2 * real.sqrt ((G.fst - F.fst)^2 + (G.snd - F.snd)^2) →
  P = (4, 0) := sorry

end chord_length_and_midpoint_properties_fixed_point_exists_P_l179_179402


namespace find_m_from_split_l179_179413

theorem find_m_from_split (m : ℕ) (h1 : m > 1) (h2 : m^2 - m + 1 = 211) : True :=
by
  -- This theorem states that under the conditions that m is a positive integer greater than 1
  -- and m^2 - m + 1 = 211, there exists an integer value for m that satisfies these conditions.
  trivial

end find_m_from_split_l179_179413


namespace sin_45_degree_l179_179173

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179173


namespace ticket_queue_orders_l179_179008

open Nat

def catalan (n : ℕ) : ℕ :=
  (1 / (n + 1)) * nat.choose (2 * n) n

theorem ticket_queue_orders (n : ℕ) : 
  ∃ order_count, order_count = catalan n := 
sorry

end ticket_queue_orders_l179_179008


namespace sin_45_degree_l179_179181

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179181


namespace determine_a_l179_179800

theorem determine_a (x : ℝ) (n : ℕ) (h : x > 0) (h_ineq : x + a / x^n ≥ n + 1) : a = n^n := by
  sorry

end determine_a_l179_179800


namespace phil_packs_duration_l179_179563

noncomputable def total_cards_left_after_fire : ℕ := 520
noncomputable def total_cards_initially : ℕ := total_cards_left_after_fire * 2
noncomputable def cards_per_pack : ℕ := 20
noncomputable def packs_bought_weeks : ℕ := total_cards_initially / cards_per_pack

theorem phil_packs_duration : packs_bought_weeks = 52 := by
  sorry

end phil_packs_duration_l179_179563


namespace winning_candidate_percentage_l179_179503

-- Definitions based on the conditions
def total_votes : ℕ := 1400
def majority_votes : ℕ := 280

-- The statement to prove
theorem winning_candidate_percentage : 
  ∃ P : ℕ, (P / 100 * total_votes - (100 - P) / 100 * total_votes = majority_votes) ∧ P = 60 :=
begin
  sorry
end

end winning_candidate_percentage_l179_179503


namespace exponent_of_5_in_30_factorial_l179_179903

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179903


namespace trig_shift_proof_l179_179485

def f_initial (x : ℝ) : ℝ := 2 * real.sin (2 * x + real.pi / 6)

def f_shifted (x : ℝ) : ℝ := 2 * real.sin (2 * (x - real.pi / 4) + real.pi / 6)

theorem trig_shift_proof :
  f_shifted = λ x, 2 * real.sin (2 * x - real.pi / 3) :=
sorry

end trig_shift_proof_l179_179485


namespace exponent_of_5_in_30_factorial_l179_179978

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179978


namespace sin_45_eq_sqrt_two_over_two_l179_179248

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179248


namespace geometric_sequence_sum_eight_terms_l179_179792

theorem geometric_sequence_sum_eight_terms
  (q : ℤ) (a1 : ℤ)
  (h1 : a1 + a1 * q ^ 3 = 18)
  (h2 : a1 * q + a1 * q ^ 2 = 12) :
  (finset.range 8).sum (λ n, a1 * q ^ n) = 510 := by
  sorry

end geometric_sequence_sum_eight_terms_l179_179792


namespace exponent_of_5_in_30_factorial_l179_179964

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179964


namespace largest_fraction_l179_179034

theorem largest_fraction :
  (∀ (x ∈ { (2 / 5), (3 / 7), (4 / 9), (5 / 11), (6 / 13) }), x ≤ (6 / 13)) ∧ 
  (6 / 13) ∈ { (2 / 5), (3 / 7), (4 / 9), (5 / 11), (6 / 13) } :=
by { sorry }

end largest_fraction_l179_179034


namespace linear_inequalities_solution_range_l179_179858

theorem linear_inequalities_solution_range (m : ℝ) :
  (∃ x : ℝ, x - 2 * m < 0 ∧ x + m > 2) ↔ m > 2 / 3 :=
by
  sorry

end linear_inequalities_solution_range_l179_179858


namespace exponent_of_5_in_30_factorial_l179_179901

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179901


namespace ana_can_find_weights_l179_179113

def weights : List ℕ := [1001, 1002, 1004, 1005]

structure Scale (α : Type) :=
  (weigh : α → α → Prop)

axiom scale_condition (s : Scale (List ℕ)) (a b c d : ℕ) :
  a ∈ weights → b ∈ weights → c ∈ weights → d ∈ weights → 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem ana_can_find_weights (s : Scale (List ℕ)) :
  (∃ (m1 m2 m3 m4 : Scale (List ℕ) → Prop), 
    (∀ (a b c d : ℕ), scale_condition s a b c d → 
       m1 s ∨ m2 s ∨ m3 s ∨ m4 s)) :=
by
  -- Proof is not required
  sorry

end ana_can_find_weights_l179_179113


namespace assign_colleagues_to_rooms_l179_179675

theorem assign_colleagues_to_rooms : 
  let rooms := 7
  let colleagues := 7
in (∑ (k : ℕ) in ({1, 3, 5} : Finset ℕ), 
    (choose rooms k) * 
    (if k = 1 then factorial colleagues
     else if k = 3 then (choose (colleagues - k + 3) 3) * factorial (colleagues - 3)
     else if k = 5 then (choose (colleagues - k + 2) 2) * factorial (colleagues - 2)
     else if k = 3 ∧ 3 then 
       ((choose rooms k) * (choose colleagues k) * factorial k *
        (1 / factorial 2 * choose (colleagues - k) 2) * factorial 2)
   else 0)) = 131460 := 
sorry

end assign_colleagues_to_rooms_l179_179675


namespace exponent_of_5_in_30_factorial_l179_179967

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179967


namespace find_integer_x_l179_179758

theorem find_integer_x : ∃ x : ℤ, x ≡ 2 [MOD 3] ∧ x ≡ 3 [MOD 4] ∧ x ≡ 1 [MOD 5] ∧ x ≡ 11 [MOD 60] := 
sorry

end find_integer_x_l179_179758


namespace cos_alpha_solution_l179_179777

open Real

theorem cos_alpha_solution
  (α : ℝ)
  (h1 : π < α)
  (h2 : α < 3 * π / 2)
  (h3 : tan α = 2) :
  cos α = -sqrt (1 / (1 + 2^2)) :=
by
  sorry

end cos_alpha_solution_l179_179777


namespace sin_45_degree_l179_179165

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179165


namespace sin_45_deg_l179_179197

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179197


namespace sin_45_deg_l179_179196

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179196


namespace minimum_first_prize_l179_179089

noncomputable def prize_values (C : ℤ) (x y z : ℕ) : Prop :=
  9 * C * x + 3 * C * y + C * z = 10800

theorem minimum_first_prize : ∃ (A : ℤ) (x y z : ℕ),
  (A = 2700) ∧
  (∀ B C : ℤ, A = 3 * B ∧ B = 3 * C) ∧
  (prize_values C x y z) ∧
  (x + y + z ≤ 20) ∧
  (z > 3 * y) ∧
  (y > 3 * x) :=
begin
  sorry
end

end minimum_first_prize_l179_179089


namespace sphere_surface_area_l179_179661

-- Given the radius of the sphere.
def radius : ℝ := 14

-- Total surface area of the sphere.
def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- The specific surface area to be proved.
theorem sphere_surface_area : surface_area radius ≈ 2463.01 := 
by sorry

end sphere_surface_area_l179_179661


namespace sqrt_x_minus_2_real_iff_x_ge_2_l179_179464

theorem sqrt_x_minus_2_real_iff_x_ge_2 (x : ℝ) : (∃ r : ℝ, r * r = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_x_minus_2_real_iff_x_ge_2_l179_179464


namespace find_x_l179_179385

theorem find_x (x : ℝ) (h : 3.5 * ( (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) ) = 2800.0000000000005) : x = 0.3 :=
sorry

end find_x_l179_179385


namespace area_of_triangle_l179_179568

theorem area_of_triangle (a b : ℝ) (γ : ℝ) : 
  (γ > 0 ∧ γ < π) → 
  (S_Δ : ℝ) (h₁ : γ ≠ π) (h₂ : γ > 0) : 
  S_Δ = (1/2 * a * b * sin γ) := sorry

end area_of_triangle_l179_179568


namespace find_omega_phi_find_cos_alpha_l179_179435

def f (x : ℝ) (ω φ : ℝ) : ℝ := real.sqrt 3 * real.sin (ω * x + φ)

-- Define hypotheses
variables (ω : ℝ) (φ : ℝ) (α : ℝ)

-- Given conditions
axiom ω_pos : ω > 0
axiom φ_range : -real.pi / 2 ≤ φ ∧ φ < real.pi / 2
axiom symmetry_condition : 2 * π / 3 + φ = k * real.pi + π / 2
axiom periodic_distance : 2 * real.pi / ω = real.pi
axiom f_alpha_cond : f (α / 2) ω φ = real.sqrt 3 / 4
axiom alpha_range : real.pi / 6 < α ∧ α < 2 * real.pi / 3

theorem find_omega_phi :
  ω = 2 ∧ φ = -real.pi / 6 :=
by sorry

theorem find_cos_alpha :
  cos (α + 3 * real.pi / 2) = (real.sqrt 3 + real.sqrt 15) / 8 :=
by sorry

end find_omega_phi_find_cos_alpha_l179_179435


namespace prime_factorization_sum_of_exponents_l179_179860

theorem prime_factorization_sum_of_exponents (i k m p q : ℕ) (h1 : 0 < i) (h2 : 0 < k) (h3 : 0 < m) (h4 : 0 < p) (h5 : 0 < q) :
  (∏ n in (finset.range 12).map (finset.nat nat.succ), n) = (2^i * 3^k * 5^m * 7^p * 11^q) →
  i + k + m + p + q = 28 :=
by
  sorry

end prime_factorization_sum_of_exponents_l179_179860


namespace find_f_2_l179_179597

variable (f : ℝ → ℝ)

def equation (x : ℝ) : Prop := f(x) + 2 * f(1 - x) = 5 * x^2 - 4 * x + 1

theorem find_f_2 (h : ∀ x, equation f x) : f 2 = 7 / 3 := 
by
  sorry

end find_f_2_l179_179597


namespace sin_45_deg_eq_l179_179307

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179307


namespace range_of_a_l179_179383

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∀ θ ∈ set.Icc 0 (real.pi / 2), 
    (x + 3 + 2 * real.sin θ * real.cos θ)^2 + (x + a * real.sin θ + a * real.cos θ)^2 ≥ 1 / 8) →
  (a ≥ 7 / 2 ∨ a ≤ real.sqrt 6) := 
by
  sorry

end range_of_a_l179_179383


namespace zero_of_ffx_l179_179426

def f (x : ℝ) : ℝ :=
if x ≤ 0 then -2 * Real.exp x else Real.log x

theorem zero_of_ffx : f (f Real.exp 1) = 0 := 
by sorry

end zero_of_ffx_l179_179426


namespace fixed_salary_correct_l179_179699

noncomputable def fixed_salary (F C S E : ℝ) := E = F + C * S

theorem fixed_salary_correct (C S E : ℝ) (hC : C = 0.05) (hS : S = 80000) (hE : E = 5000) : 
  ∃ F, fixed_salary F C S E ∧ F = 1000 :=
by
  use 1000
  rw [fixed_salary, hC, hS, hE]
  norm_num
  sorry

end fixed_salary_correct_l179_179699


namespace exponent_of_5_in_30_factorial_l179_179902

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179902


namespace solve_for_x_l179_179578

theorem solve_for_x (x : ℝ) (h : 2^(x - 3) = 16) : x = 7 := 
sorry

end solve_for_x_l179_179578


namespace total_distance_travelled_l179_179571

theorem total_distance_travelled
  (XZ XY : ℝ)
  (hXZ : XZ = 4000)
  (hXY : XY = 4500)
  (hXYZ : ∀ Z : ℝ, XZ^2 + Z^2 = XY^2) :
  XY + sqrt ((XY^2 - XZ^2) : ℝ) + XZ = 8500 + 2500 * sqrt (17 : ℝ) :=
by
  sorry

end total_distance_travelled_l179_179571


namespace circumference_of_jack_head_l179_179519

theorem circumference_of_jack_head (J C : ℝ) (h1 : (2 / 3) * C = 10) (h2 : (1 / 2) * J + 9 = 15) :
  J = 12 :=
by
  sorry

end circumference_of_jack_head_l179_179519


namespace sin_45_eq_sqrt2_div_2_l179_179205

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179205


namespace mid_reflect_sum_zero_l179_179026

theorem mid_reflect_sum_zero :
  let P1 := (3, -4) 
  let P2 := (-5, 2)
  let Mx := (P1.1 + P2.1) / 2
  let My := (P1.2 + P2.2) / 2
  let M := (Mx, My)
  let R := (M.1, -M.2)
  R.1 + R.2 = 0 := 
by
  let P1 := (3, -4)
  let P2 := (-5, 2)
  let Mx := (P1.1 + P2.1) / 2
  let My := (P1.2 + P2.2) / 2
  let M := (Mx, My)
  let R := (M.1, -M.2)
  show R.1 + R.2 = 0 from
  sorry

end mid_reflect_sum_zero_l179_179026


namespace cos_angle_value_l179_179756

noncomputable def cos_angle := Real.cos (19 * Real.pi / 4)

theorem cos_angle_value : cos_angle = -Real.sqrt 2 / 2 := by
  sorry

end cos_angle_value_l179_179756


namespace sin_45_degree_l179_179281

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179281


namespace find_a_l179_179436

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 5 * x^2 + a * x

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, x = -3 ∧ ∀ x' : ℝ, f (f'':=f' (a:=a) : (x : ℝ) -> x^3 + 5x^2 + ax
  := 3x^2 + 10x + a) x' = 0 -> x = -3) → a = 3 :=
by
  sorry

end find_a_l179_179436


namespace sin_45_eq_sqrt2_div_2_l179_179339

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179339


namespace sin_45_eq_sqrt_two_over_two_l179_179244

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179244


namespace triangle_inequality_l179_179566

variable {A B C O : Type*}

-- Assuming point O lies on the segment AB, but does not coincide with A and B
variable [InnerProductSpace ℝ A]

-- Coordinates of the points
variable (a b c o : A)

-- The assumptions that O lies on AB
variable (h_oc : o = a + ((1:ℝ) - x) • (b - a) + x • c)

-- Prove the desired inequality
theorem triangle_inequality (hO : o ∈ ConvexHull ℝ {a, b})
  (hO₁ : o ≠ a) (hO₂ : o ≠ b) :
  ∥o - c∥ * ∥b - a∥ < ∥o - a∥ * ∥b - c∥ + ∥o - b∥ * ∥a - c∥ :=
sorry

end triangle_inequality_l179_179566


namespace smallest_positive_period_l179_179744

-- Define the function and omega
def y (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)
def omega : ℝ := 2

-- State the theorem about the smallest positive period
theorem smallest_positive_period : (2 * Real.pi / omega = Real.pi) :=
by
  simp [omega]
  sorry

end smallest_positive_period_l179_179744


namespace exponent_of_5_in_30_factorial_l179_179961

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179961


namespace largest_prime_factor_of_145_187_221_299_169_l179_179035

theorem largest_prime_factor_of_145_187_221_299_169 :
  max (max (max (max (nat.prime_factor 145) (nat.prime_factor 187)) (nat.prime_factor 221)) (nat.prime_factor 299)) (nat.prime_factor 169) = 29 :=
by
  -- This is where the proof would go
  sorry

end largest_prime_factor_of_145_187_221_299_169_l179_179035


namespace identical_digits_satisfy_l179_179606

theorem identical_digits_satisfy (n : ℕ) (hn : n ≥ 2) (x y z : ℕ) :
  (∃ (x y z : ℕ),
     (∃ (x y z : ℕ), 
         x = 3 ∧ y = 2 ∧ z = 1) ∨
     (∃ (x y z : ℕ), 
         x = 6 ∧ y = 8 ∧ z = 4) ∨
     (∃ (x y z : ℕ), 
         x = 8 ∧ y = 3 ∧ z = 7)) :=
by sorry

end identical_digits_satisfy_l179_179606


namespace pencil_distribution_l179_179772

/-- The statement of our theorem which encapsulates the problem conditions and required proof. -/
theorem pencil_distribution (pencils friends : ℕ) (h_pencils : pencils = 7) (h_friends : friends = 4) :
  (∃ distribution : friends → ℕ,
  (∀ f, 1 ≤ distribution f) ∧ (∑ f in finset.range friends, distribution f = pencils)) ↔ (52) := 
sorry

end pencil_distribution_l179_179772


namespace num_factors_of_90_multiple_of_6_l179_179451

def is_factor (m n : ℕ) : Prop := n % m = 0
def is_multiple_of (m n : ℕ) : Prop := n % m = 0

theorem num_factors_of_90_multiple_of_6 : 
  ∃ (count : ℕ), count = 4 ∧ ∀ x, is_factor x 90 → is_multiple_of 6 x → x > 0 :=
sorry

end num_factors_of_90_multiple_of_6_l179_179451


namespace find_b_l179_179393

theorem find_b (a b : ℝ) (h1 : (1 + a * x)^5 = (1 : ℝ) + 10 * x + b * x^2 + ∑ k in Ico 3 6, (some_coeff k) * x^k)
  (h2 : 5 * a = 10) : b = 40 :=
sorry

end find_b_l179_179393


namespace find_x_l179_179473

theorem find_x (b x : ℝ) (h₀ : b > 1) (h₁ : x > 0) :
  (4 * x) ^ (Real.log b 4) + (5 * x) ^ (Real.log b 5) = 9 * x ↔ x = 1 :=
by
  sorry

end find_x_l179_179473


namespace regular_polygon_sides_l179_179092

theorem regular_polygon_sides (n : ℕ) (h1 : 2 ≤ n) (h2 : (n - 2) * 180 / n = 120) : n = 6 :=
by
  sorry

end regular_polygon_sides_l179_179092


namespace sin_45_eq_l179_179266

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179266


namespace sin_45_eq_one_div_sqrt_two_l179_179221

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179221


namespace probability_units_digit_and_even_sum_l179_179386

theorem probability_units_digit_and_even_sum :
  let a_range := {1, 2, ..., 200}
  let b_range := {1, 2, ..., 200}
  (∀ a ∈ a_range, b ∈ b_range,
  ∃ (count : ℕ),
    count = (3 ^ a + 7 ^ b) % 10 = 3 ∧ (a + b) % 2 = 0) →
  (count.to_real / (200 * 200).to_real) = 1 / 8 :=
begin
  sorry
end

end probability_units_digit_and_even_sum_l179_179386


namespace max_dot_product_is_correct_l179_179492

open Real

-- We define the given conditions as hypotheses in Lean.
variables (A B C P : Type) [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace P]
variables {AB AC AP : ℝ} {θ : ℝ}
hypothesis h1 : AB = 3
hypothesis h2 : AC = 4
hypothesis h3 : θ = 60
hypothesis h4 : AP = 2

-- Define a function that computes the dot product maximum.
def maximum_dot_product : ℝ :=
  let AD := (|B - C|) / 2 in
  let BC := sqrt ((AB^2 + AC^2) - (2 * AB * AC * cos θ)) in
  let max_val := 10 + 2 * sqrt (AD^2) in
  max_val

-- State the theorem in Lean.
theorem max_dot_product_is_correct :
  maximum_dot_product = 10 + 2 * sqrt 37 := sorry

end max_dot_product_is_correct_l179_179492


namespace board_coloring_even_conditions_l179_179363

-- Define the condition for the board coloring problem.
def board_coloring_condition (m n : ℤ) : Prop :=
  ∃ (colors : ℤ × ℤ → bool), ∀ (i j : ℤ), 
    0 ≤ i ∧ i < m → 0 ≤ j ∧ j < n →
    let same_color_neighbors := 
      (if colors (i, j) = colors (i + 1, j) then 1 else 0) + 
      (if colors (i, j) = colors (i - 1, j) then 1 else 0) + 
      (if colors (i, j) = colors (i, j + 1) then 1 else 0) + 
      (if colors (i, j) = colors (i, j - 1) then 1 else 0) 
    in same_color_neighbors % 2 = 0

-- Code the theorem statement.
theorem board_coloring_even_conditions (m n : ℤ) (h : 0 < m ∧ 0 < n) : 
  (board_coloring_condition m n) ↔ (2 ∣ m * n) :=
by {
  sorry
}

end board_coloring_even_conditions_l179_179363


namespace find_f_5_l179_179397

noncomputable def f : ℤ → ℤ :=
  λ n, if h : ∃ x : ℤ, 2 * x + 1 = n then some ⟨x, by sorry⟩ ^ 2 else 0

theorem find_f_5 : f 5 = 4 := by
  sorry

end find_f_5_l179_179397


namespace problem_1_problem_2_problem_3_l179_179767

noncomputable def P_sequence (seq : List ℝ) : Prop :=
    ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ seq.length → 
    |seq.get ⟨i - 1, sorry⟩ - seq.get ⟨i, sorry⟩| ≤ |seq.get ⟨j - 1, sorry⟩ - seq.get ⟨j, sorry⟩|

theorem problem_1 {a b : ℕ → ℝ} :
  (∀ n, 1 ≤ n ∧ n ≤ 4 → a n = n ^ 2) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 5 → b n = (1 / 2)^n) →
  P_sequence [a 1, a 2, a 3, a 4] ∧ P_sequence [b 1, b 2, b 3, b 4, b 5] :=
sorry

theorem problem_2 (a : ℕ → ℝ) 
  (hP : P_sequence (List.ofFn (λ n, a (10 - n + 1))))
  (hdist : ∀ i j : ℕ, 1 ≤ i < j ≤ 10 → a i ≠ a j)
  (hstart : a 1 = 20)
  (hend : a 10 = 2) : 
  ∀ n, 1 ≤ n ∧ n ≤ 10 → a n = -2 * n + 22 :=
sorry

theorem problem_3 (a : ℕ → ℝ) 
  (hP : P_sequence (List.ofFn (λ n, a n))) 
  (hperm : ∀ n, 1 ≤ n ≤ m → ∃ k, a k = n)
  (hsum : (List.ofFn (λ n, |a n - a (n + 1)|)).sum = m + 1) :
  m = 4 :=
sorry

end problem_1_problem_2_problem_3_l179_179767


namespace sin_45_deg_l179_179195

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179195


namespace find_center_and_radius_find_chord_length_l179_179405

def circle_equation (x y : ℝ) := x^2 + y^2 - 4 * x - 4 * y + 4

def line_equation (x y : ℝ) := x + 2 * y - 4

noncomputable def distance_to_line (px py : ℝ) (a b c : ℝ) :=
  (a * px + b * py + c) / real.sqrt (a^2 + b^2)

theorem find_center_and_radius :
  let center := (2, 2) in
  let radius := 2 in
  (∀ x y, circle_equation x y = (x - center.1)^2 + (y - center.2)^2 - radius^2) ∧
  ∀ x y, circle_equation x y = 0 → (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

theorem find_chord_length :
  let center := (2, 2) in
  let radius := 2 in
  let chord_length := 2 * real.sqrt (radius^2 - (distance_to_line center.1 center.2 1 2 (-4))^2) in
  chord_length = 8 * real.sqrt 5 / 5 :=
sorry

end find_center_and_radius_find_chord_length_l179_179405


namespace sin_45_degree_l179_179283

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179283


namespace independent_set_exists_l179_179501

theorem independent_set_exists (g : fin 8 → fin 8 → Prop)
  (row_condition : ∀ r : fin 8, fintype.card {c : fin 8 // g r c} = 3)
  (col_condition : ∀ c : fin 8, fintype.card {r : fin 8 // g r c} = 3) :
  ∃ (S : fin 8 → fin 8), function.injective S ∧ ∀ r, g r (S r) :=
by
  sorry

end independent_set_exists_l179_179501


namespace veridux_female_employees_l179_179713

theorem veridux_female_employees :
  let total_employees := 250
  let total_managers := 40
  let male_associates := 160
  let female_managers := 40
  let total_associates := total_employees - total_managers
  let female_associates := total_associates - male_associates
  let female_employees := female_managers + female_associates
  female_employees = 90 :=
by
  let total_employees := 250
  let total_managers := 40
  let male_associates := 160
  let female_managers := 40
  let total_associates := total_employees - total_managers
  let female_associates := total_associates - male_associates
  let female_employees := female_managers + female_associates
  have : female_employees = 90 := by
    unfold total_employees total_managers male_associates female_managers total_associates female_associates female_employees
    sorry
  exact this

end veridux_female_employees_l179_179713


namespace max_min_sum_function_l179_179425

theorem max_min_sum_function (g : ℝ → ℝ) (f : ℝ → ℝ) (M N : ℝ) (h1 : ∀ x ∈ set.Icc (-3 : ℝ) 3, f x = g x + 2)
    (h2 : ∀ x : ℝ, g (-x) = -g x)
    (h3 : ∃ a ∈ set.Icc (-3 : ℝ) 3, is_max_on f (set.Icc (-3) 3) a)
    (h4 : ∃ b ∈ set.Icc (-3 : ℝ) 3, is_min_on f (set.Icc (-3) 3) b) 
    (h5 : M = Sup (set.Icc (-3 : ℝ) 3) f)
    (h6 : N = Inf (set.Icc (-3 : ℝ) 3) f) 
:
  M + N = 4 := 
sorry

end max_min_sum_function_l179_179425


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179241

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179241


namespace exponent_of_5_in_30_factorial_l179_179962

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179962


namespace sin_45_degree_eq_sqrt2_div_2_l179_179153

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179153


namespace bea_has_max_profit_l179_179137

theorem bea_has_max_profit : 
  let price_bea := 25
  let price_dawn := 28
  let price_carla := 35
  let sold_bea := 10
  let sold_dawn := 8
  let sold_carla := 6
  let cost_bea := 10
  let cost_dawn := 12
  let cost_carla := 15
  let profit_bea := (price_bea * sold_bea) - (cost_bea * sold_bea)
  let profit_dawn := (price_dawn * sold_dawn) - (cost_dawn * sold_dawn)
  let profit_carla := (price_carla * sold_carla) - (cost_carla * sold_carla)
  profit_bea = 150 ∧ profit_dawn = 128 ∧ profit_carla = 120 ∧ ∀ p, p ∈ [profit_bea, profit_dawn, profit_carla] → p ≤ 150 :=
by
  sorry

end bea_has_max_profit_l179_179137


namespace hydroxide_mass_percentage_l179_179510

theorem hydroxide_mass_percentage (Ba_percent : ℚ) (hBa : Ba_percent = 80.12) : 
  100 - Ba_percent = 19.88 :=
by
  rw [hBa]
  norm_num
  sorry

end hydroxide_mass_percentage_l179_179510


namespace length_of_train_is_110_l179_179700

/-- Define the speed of the train in km/hr -/
def speed_train_kmh : ℝ := 80

/-- Define the speed of the man in km/hr (negative because it's in the opposite direction) -/
def speed_man_kmh : ℝ := -8

/-- Define the time taken for the train to pass the man in seconds -/
def time_seconds : ℝ := 4.499640028797696

/-- Convert km/hr to m/s -/
def kmh_to_ms (speed: ℝ) : ℝ := speed * 1000 / 3600

/-- Define the relative speed in m/s -/
def relative_speed_ms : ℝ := kmh_to_ms (speed_train_kmh - speed_man_kmh)

/-- Define the length of the train. -/
noncomputable def length_of_train : ℝ := relative_speed_ms * time_seconds

/-- Prove that the length of the train is approximately 110 meters -/
theorem length_of_train_is_110 : length_of_train ≈ 110 :=
by
  sorry

end length_of_train_is_110_l179_179700


namespace exponent_of_5_in_30_factorial_l179_179977

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179977


namespace sin_45_eq_sqrt2_div_2_l179_179292

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179292


namespace problem_1_problem_2_l179_179433

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1)

theorem problem_1 (m : ℝ) (h : m > 0) :
  (set_of (λ x : ℝ, f (x + 1/2) ≤ 2 * m + 1) = set.Icc (-2 : ℝ) (2 : ℝ)) →
  m = 3 / 2 :=
begin
  sorry
end

theorem problem_2 (a : ℝ) :
  (∀ x y : ℝ, f x ≤ 2^y + a / 2^y + abs(2 * x + 3)) →
  a = 4 :=
begin
  sorry
end

end problem_1_problem_2_l179_179433


namespace molecular_weight_of_10_moles_l179_179644

-- Define the molecular weight of a compound as a constant
def molecular_weight (compound : Type) : ℝ := 840

-- Prove that the molecular weight of 10 moles of the compound is the same as the molecular weight of 1 mole of the compound
theorem molecular_weight_of_10_moles (compound : Type) :
  molecular_weight compound = 840 :=
by
  -- Proof
  sorry

end molecular_weight_of_10_moles_l179_179644


namespace exponent_of_5_in_30_factorial_l179_179988

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179988


namespace exponent_of_5_in_30_factorial_l179_179939

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179939


namespace only_monotonic_positive_function_is_reciprocal_l179_179365

theorem only_monotonic_positive_function_is_reciprocal (f : ℝ → ℝ)
  (h1 : ∀ x y > 0, f(x * y) * f(f(y) / x) = 1)
  (h2 : ∀ x > 0, f(x) > 0)
  (h3 : strict_mono f ∨ strict_anti f) :
  ∀ x > 0, f(x) = 1 / x :=
by
  sorry

end only_monotonic_positive_function_is_reciprocal_l179_179365


namespace sin_45_eq_l179_179257

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179257


namespace concentric_circle_radius_ratio_l179_179632

theorem concentric_circle_radius_ratio (r AC BC AB : ℝ) (h1 : r = 20 / 14)
  (H1 : AC = 2 * 7 * r)
  (H2 : BC = AB):
  7 * r = 70 / 3 :=
by
  have H : r = 10 / 3 := by sorry
  rw [H]
  norm_num
  sorry

end concentric_circle_radius_ratio_l179_179632


namespace train_length_l179_179047

/-- Conditions:
a) Two trains run on parallel lines in the same direction.
b) The speeds of the trains are 48 km/h and 36 km/h respectively.
c) The faster train takes 36 seconds to pass the slower train completely.
-/
theorem train_length (L : ℝ) : 
  let speed_faster := 48 * 1000 / 3600,
      speed_slower := 36 * 1000 / 3600,
      relative_speed := speed_faster - speed_slower,
      time := 36,
      distance := relative_speed * time in
  distance / 2 = L → 
  L = 60 :=
by
  intro h
  sorry

end train_length_l179_179047


namespace bike_tire_repairs_l179_179521

theorem bike_tire_repairs (x : ℕ) (h1 : ∀ n : ℕ, (20 - 5) * n + 500 + 2000 - 4000 = 3000 → n = 300) :
  ∃ x : ℕ, (20 - 5) * x + 500 + 2000 - 4000 = 3000 :=
begin
  use 300,
  apply h1,
end

end bike_tire_repairs_l179_179521


namespace range_of_a_l179_179442

theorem range_of_a (A : Set ℝ) (a : ℝ) : 
  (A = {y | ∃ x, y = Real.sqrt (a*x^2 + 2*(a-1)*x - 4)} ∧ A = Ici 0) ↔ a ∈ Ici 0 :=
by
  sorry

end range_of_a_l179_179442


namespace chord_product_equality_l179_179543

theorem chord_product_equality {A B C D : Point} (h : cyclic A B C D) :
  (dist A C) * (dist B D) = (dist A B) * (dist D C) + (dist A D) * (dist B C) :=
sorry

end chord_product_equality_l179_179543


namespace sin_45_eq_sqrt2_div_2_l179_179328

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179328


namespace binom_coeff_sum_poly_coeff_sum_l179_179055

-- Part 1
theorem binom_coeff_sum {n : ℕ} (h : 2^n = 64) : 
  let k := 6 in
  (k = n) →
  let term_rplus1 (r : ℕ) := ↑((nat.choose k r) * (-1) ^ r * 5 ^ (k - r) : ℤ) * ((5 : ℤ) ^ (6 - 4 * r / 3) : ℤ) in
  term_rplus1 3 = -2500 :=
by 
  intros _ h; sorry

-- Part 2
theorem poly_coeff_sum {a b c d e f g : ℤ} (h : (1 - x) * (2 * x + 1)^3 = a + b * x + c * x^2 + d * x^3 + e * x^4 + f * x^5 + g * x^6) :
  (c + e + g = -2) ∧ (d = -4) :=
by 
  sorry

end binom_coeff_sum_poly_coeff_sum_l179_179055


namespace distinguishable_balls_indistinguishable_boxes_l179_179457

theorem distinguishable_balls_indistinguishable_boxes :
  ∃ (f : Finset (Multiset ℕ)), ∀ (n : ℕ) (m : ℕ) 
  (hn : n = 6) (hm : m = 4), Multiset.card f = 257 ∧ 
  (∀ x ∈ f, Multiset.sum x = n ∧ Multiset.card x ≤ m) 
  := sorry

end distinguishable_balls_indistinguishable_boxes_l179_179457


namespace greatest_integer_less_than_neg22_div_3_l179_179640

def greatest_integer_less_than (x : ℝ) : ℤ :=
  int.floor x

theorem greatest_integer_less_than_neg22_div_3 : greatest_integer_less_than (-22 / 3) = -8 := by
  sorry

end greatest_integer_less_than_neg22_div_3_l179_179640


namespace taxi_company_charges_l179_179070

theorem taxi_company_charges
  (X : ℝ)  -- charge for the first 1/5 of a mile
  (C : ℝ)  -- charge for each additional 1/5 of a mile
  (total_charge : ℝ)  -- total charge for an 8-mile ride
  (remaining_distance_miles : ℝ)  -- remaining miles after the first 1/5 mile
  (remaining_increments : ℝ)  -- remaining 1/5 mile increments
  (charge_increments : ℝ)  -- total charge for remaining increments
  (X_val : X = 2.50)
  (C_val : C = 0.40)
  (total_charge_val : total_charge = 18.10)
  (remaining_distance_miles_val : remaining_distance_miles = 7.8)
  (remaining_increments_val : remaining_increments = remaining_distance_miles * 5)
  (charge_increments_val : charge_increments = remaining_increments * C)
  (proof_1: charge_increments = 15.60)
  (proof_2: total_charge - charge_increments = X) : X = 2.50 := 
by
  sorry

end taxi_company_charges_l179_179070


namespace sin_45_eq_one_div_sqrt_two_l179_179224

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179224


namespace sin_45_deg_l179_179192

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179192


namespace input_statement_is_INPUT_l179_179110

-- Define the type for statements
inductive Statement
| PRINT
| INPUT
| IF
| END

-- Define roles for the types of statements
def isOutput (s : Statement) : Prop := s = Statement.PRINT
def isInput (s : Statement) : Prop := s = Statement.INPUT
def isConditional (s : Statement) : Prop := s = Statement.IF
def isTermination (s : Statement) : Prop := s = Statement.END

-- Theorem to prove INPUT is the input statement
theorem input_statement_is_INPUT :
  isInput Statement.INPUT := by
  -- Proof to be provided
  sorry

end input_statement_is_INPUT_l179_179110


namespace discriminant_zero_geometric_progression_l179_179343

variable (a b c : ℝ)

theorem discriminant_zero_geometric_progression
  (h : b^2 = 4 * a * c) : (b / (2 * a)) = (2 * c / b) :=
by
  sorry

end discriminant_zero_geometric_progression_l179_179343


namespace find_annual_interest_rate_l179_179136

/-- 
  Given:
  - Principal P = 10000
  - Interest I = 450
  - Time period T = 0.75 years

  Prove that the annual interest rate is 0.08.
-/
theorem find_annual_interest_rate (P I : ℝ) (T : ℝ) (hP : P = 10000) (hI : I = 450) (hT : T = 0.75) : 
  (I / (P * T) / T) = 0.08 :=
by
  sorry

end find_annual_interest_rate_l179_179136


namespace exponent_of_5_in_30_fact_l179_179884

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179884


namespace find_coyote_speed_l179_179346

noncomputable def coyote_speed (c : ℕ) : Prop :=
  let coyote_distance := c * 1 / 1 in
  let darrel_chase_time := 30 / 1 in
  coyote_distance + c = darrel_chase_time

theorem find_coyote_speed : {c : ℕ // coyote_speed c} :=
  begin
    use 15,
    have h : (15 * 1) + 15 = 30, by norm_num,
    exact h,
  end

end find_coyote_speed_l179_179346


namespace find_selling_price_l179_179009

noncomputable theory

-- Define constants and conditions
def cost_price : ℝ := 30
def point1 : ℝ × ℝ := (35, 550)
def point2 : ℝ × ℝ := (40, 500)
def max_selling_price : ℝ := 60
def target_profit : ℝ := 80

-- Prove the linear function relationship and the required selling price
theorem find_selling_price : 
  ∃ (k b x : ℝ), 
  (y = k * x + b) ∧  -- Linear relationship
  (k = (point2.2 - point1.2) / (point2.1 - point1.1)) ∧ -- Compute slope k
  (b = point1.2 - k * point1.1) ∧ -- Compute intercept b
  (x ≤ max_selling_price) ∧      -- Selling price constraint
  ((x - cost_price) * (k * x + b) = target_profit)  -- Profit equation
  :=
begin
  -- Definitions for linear relationship
  let k := (point2.2 - point1.2) / (point2.1 - point1.1),
  let b := point1.2 - k * point1.1,
  let y := λ x, k * x + b,

  -- Define target price solving the profit equation
  existsi k,
  existsi b,
  existsi ((cost_price + target_profit) / (k + 900 - cost_price)): ℝ,
  split,
  { exact y },
  split,
  { exact k },
  split,
  { exact b },
  split,
  { linarith },
  { sorry }
end

end find_selling_price_l179_179009


namespace quadrilateral_is_rhombus_l179_179049

noncomputable def is_rhombus (A B C D : Type) [points : add_group A] [has_scalar ℝ A] : Prop :=
∃ p q r s O : A, 
  (O = (p + r)/2) ∧
  (O = (q + s)/2) ∧
  (incircle_radius p q O = incircle_radius q r O) ∧ 
  (incircle_radius q r O = incircle_radius r s O) ∧ 
  (incircle_radius r s O = incircle_radius s p O) ∧
  (∥p - q∥ = ∥q - r∥) ∧ 
  (∥q - r∥ = ∥r - s∥) ∧ 
  (∥r - s∥ = ∥s - p∥)

theorem quadrilateral_is_rhombus (A B C D : Type) [points : add_group A] [has_scalar ℝ A] 
  (O : A) (r : ℝ) 
  (incircle_radius : ∀ (X Y Z : A), ℝ) : is_rhombus A B C D → 
  (∃ p q r s : A, 
    O = (p + r)/2 ∧ 
    O = (q + s)/2 ∧
    incircle_radius p q O = r ∧
    incircle_radius q r O = r ∧
    incircle_radius r s O = r ∧
    incircle_radius s p O = r :=
sorry

end quadrilateral_is_rhombus_l179_179049


namespace inclination_angle_range_l179_179486

theorem inclination_angle_range (a : ℝ) (θ : ℝ) :
  (∃ (x y : ℝ), x + a * y + real.sqrt 3 * a = 0 ∧ 2 * x + 3 * y - 6 = 0 ∧ x > 0 ∧ y > 0) →
  θ = real.arctan a →
  θ ∈ set.Ioo (real.pi / 6) (real.pi / 2) :=
by
  sorry

end inclination_angle_range_l179_179486


namespace ratio_q_t_eqn_l179_179341

-- Conditions right from the problem description
variable (side_length : ℝ) -- side length of the square dartboard
variable (t : ℝ) -- area of the triangular region
variable (q : ℝ) -- area of the quadrilateral region

-- Assuming specific values for the side_length and calculation formulas for t and q (details derived from conditions, not solution steps directly)
def is_dartboard_well_defined (side_length t q : ℝ) : Prop :=
  t = (sqrt (2 + sqrt 2)) / 4 ∧
  q = (4 - sqrt (2 + sqrt 2)) / 4

-- Lean statement proposition
theorem ratio_q_t_eqn : side_length = 2 → is_dartboard_well_defined side_length t q → 
  q / t = (4 - sqrt (2 + sqrt 2)) / (sqrt (2 + sqrt 2)) :=
by { sorry }

end ratio_q_t_eqn_l179_179341


namespace perimeter_triangle_formed_by_parallel_lines_l179_179013

-- Defining the side lengths of the triangle ABC
def AB := 150
def BC := 270
def AC := 210

-- Defining the lengths of the segments formed by intersections with lines parallel to the sides of ABC
def length_lA := 65
def length_lB := 60
def length_lC := 20

-- The perimeter of the triangle formed by the intersection of the lines
theorem perimeter_triangle_formed_by_parallel_lines :
  let perimeter : ℝ := 5.71 + 20 + 83.33 + 65 + 91 + 60 + 5.71
  perimeter = 330.75 := by
  sorry

end perimeter_triangle_formed_by_parallel_lines_l179_179013


namespace find_cd_sum_l179_179534

namespace LatticePointProblem

def lattice_points (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 40 ∧ 1 ≤ y ∧ y ≤ 40

def points_below_parabola (a : ℚ) (x y : ℕ) : Prop :=
  lattice_points x y ∧ y ≤ a * x ^ 2

theorem find_cd_sum : 
  ∃ (a_min a_max : ℚ), 
    (∀ (x y : ℕ), points_below_parabola a_min x y) ∧ 
    (∀ (x y : ℕ), points_below_parabola a_max x y) ∧ 
    a_max - a_min = 1 / 400 ∧ 
    let c := 1 in let d := 400 in 
    c + d = 401 :=
sorry

end LatticePointProblem

end find_cd_sum_l179_179534


namespace smallest_prime_dividing_2_pow_14_plus_7_pow_8_l179_179025

theorem smallest_prime_dividing_2_pow_14_plus_7_pow_8 :
  ∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → ¬ (p ∣ (2^14 + 7^8)) :=
by 
  sorry

end smallest_prime_dividing_2_pow_14_plus_7_pow_8_l179_179025


namespace share_pizza_l179_179859

variable (Yoojung_slices Minyoung_slices total_slices : ℕ)
variable (Y : ℕ)

theorem share_pizza :
  Yoojung_slices = Y ∧
  Minyoung_slices = Y + 2 ∧
  total_slices = 10 ∧
  Yoojung_slices + Minyoung_slices = total_slices →
  Y = 4 :=
by
  sorry

end share_pizza_l179_179859


namespace sin_45_deg_eq_l179_179310

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179310


namespace sin_45_eq_1_div_sqrt_2_l179_179322

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179322


namespace inverse_proposition_false_l179_179602

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a = b → abs a = abs b

-- Define the inverse proposition
def inverse_proposition (a b : ℝ) : Prop :=
  abs a = abs b → a = b

-- The theorem to prove
theorem inverse_proposition_false : ∃ (a b : ℝ), abs a = abs b ∧ a ≠ b :=
sorry

end inverse_proposition_false_l179_179602


namespace sin_45_deg_l179_179190

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179190


namespace exponent_of_5_in_30_factorial_l179_179940

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179940


namespace annual_interest_rate_l179_179745

-- Define the initial amount charged
def initial_amt : ℝ := 60

-- Define the final amount owed after a year
def final_amt : ℝ := 63.6

-- Define the time period 
def time : ℝ := 1

-- Calculate the interest
def interest : ℝ := final_amt - initial_amt

-- Define the principal amount
def principal : ℝ := initial_amt

-- Define the rate as the value we need to prove
def rate : ℝ := interest / (principal * time) * 100

-- Prove that the rate is 6%
theorem annual_interest_rate : rate = 6 :=
by
  -- placeholder for proof
  sorry

end annual_interest_rate_l179_179745


namespace ellipse_line_intersection_l179_179404

theorem ellipse_line_intersection {a b c : ℝ} (h1 : c^2 = a^2 - b^2)
  (x y : ℝ) (h2 : x^2 / (a * a) + y^2 / (b * b) = 1) (λ₁ λ₂ : ℝ) :
  ∃ (P A B : ℝ × ℝ),
    P.1 = 0 ∧ P.2 ≠ 0 ∧
    A.1 = λ₁ ∧ B.1 = λ₂ ∧
    (λ₁ + λ₂ = 2 * a) :=
begin
  sorry
end

end ellipse_line_intersection_l179_179404


namespace sin_45_eq_l179_179262

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179262


namespace radius_of_C1_l179_179725

theorem radius_of_C1
    (O X Y Z : Type)
    [metric_space O]
    [metric_space X]
    [metric_space Y]
    [metric_space Z]
    (C1 C2 : metric_space O)
    (hC1 : center O C1)
    (hC2 : center O C2)
    (hXY_meet : meet X Y C2)
    (XZ : ℝ)
    (hXZ : XZ = 15)
    (OZ : ℝ)
    (hOZ : OZ = 12)
    (YZ : ℝ)
    (hYZ : YZ = 9) :
    radius C1 = 3 * real.sqrt 7 :=
sorry

end radius_of_C1_l179_179725


namespace set_of_points_z_is_line_l179_179842

open Complex

theorem set_of_points_z_is_line (S : Set ℂ) :
  (∀ z : ℂ, (2 + 3 * Complex.I) * z ∈ ℝ ↔ z ∈ S) ↔ ∃ k : ℝ, ∀ z : ℂ, z.im = k * z.re :=
sorry

end set_of_points_z_is_line_l179_179842


namespace distinguishable_balls_indistinguishable_boxes_l179_179456

theorem distinguishable_balls_indistinguishable_boxes :
  ∃ (f : Finset (Multiset ℕ)), ∀ (n : ℕ) (m : ℕ) 
  (hn : n = 6) (hm : m = 4), Multiset.card f = 257 ∧ 
  (∀ x ∈ f, Multiset.sum x = n ∧ Multiset.card x ≤ m) 
  := sorry

end distinguishable_balls_indistinguishable_boxes_l179_179456


namespace trajectory_ellipse_distance_BD_l179_179786

-- Definition and setup for the conditions and given problem.
structure Circle where
  center : Point
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- (I) Prove the equation of the trajectory E of point M is x²/4 + y² = 1
theorem trajectory_ellipse :
  ∀ (Q : Point), 
  Q ∈ Circle (Point.mk (-sqrt 3) 0) 4 →
  let AQ_midpoint := Point.mk ((Q.x + sqrt 3) / 2) (Q.y / 2)
  let M := -- some computation based on AQ_midpoint and Q
  let E := M ∈ Ellipse (Point.mk 0 0) 2 1 → 
  E = Ellipse (Point.mk 0 0) 2 1 :=
  by
  sorry

-- (II) Prove the value of |BD| is sqrt(66)/3
theorem distance_BD :
  ∀ (k : ℝ),
  let l := TangentLineFromPoint A Circle (Point.mk 0 0) 1
  let (B D : Point) := (IntersectionPoints E l)
  let BD := distance B D
  BD = sqrt 66 / 3 :=
  by
  sorry 

end trajectory_ellipse_distance_BD_l179_179786


namespace mixed_oil_rate_per_litre_l179_179041

variables (volume1 : ℝ) (price1 : ℝ) (volume2 : ℝ) (price2 : ℝ)

def total_cost (v p : ℝ) : ℝ := v * p
def total_volume (v1 v2 : ℝ) : ℝ := v1 + v2

theorem mixed_oil_rate_per_litre (h1 : volume1 = 10) (h2 : price1 = 55) (h3 : volume2 = 5) (h4 : price2 = 66) :
  (total_cost volume1 price1 + total_cost volume2 price2) / total_volume volume1 volume2 = 58.67 := 
by
  sorry

end mixed_oil_rate_per_litre_l179_179041


namespace even_integer_squares_l179_179741

noncomputable def Q (x : ℤ) : ℤ := x^4 + 6 * x^3 + 11 * x^2 + 3 * x + 25

theorem even_integer_squares (x : ℤ) (hx : x % 2 = 0) :
  (∃ (a : ℤ), Q x = a ^ 2) → x = 8 :=
by
  sorry

end even_integer_squares_l179_179741


namespace sin_45_eq_sqrt2_div_2_l179_179207

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179207


namespace sin_45_deg_l179_179189

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179189


namespace linear_elimination_l179_179654

theorem linear_elimination (a b : ℤ) (x y : ℤ) :
  (a = 2) ∧ (b = -5) → 
  (a * (5 * x - 2 * y) + b * (2 * x + 3 * y) = 0) → 
  (10 * x - 4 * y + -10 * x - 15 * y = 8 + -45) :=
by
  sorry

end linear_elimination_l179_179654


namespace sin_45_eq_one_div_sqrt_two_l179_179216

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179216


namespace number_of_sets_B_l179_179531

open Set

theorem number_of_sets_B :
  ∃ B : Set ℕ, A ∪ B = {1, 2, 3, 4} ∧ card {B | A ∪ B = {1, 2, 3, 4}} = 4 :=
by
  let A := {x : ℕ | log 3 x < 1} ∩ {x | x ∈ {1, 2, 3, 4}}
  have hA : A = {1, 2} := sorry
  sorry

end number_of_sets_B_l179_179531


namespace sin_45_eq_l179_179259

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179259


namespace incorrect_statement_D_l179_179539

-- Definitions of lines and planes
variables (m n : Line) (α β : Plane)

-- Conditions
variables (h1 : m ∥ α) (h2 : α ∩ β = n)

-- Statement to prove
theorem incorrect_statement_D : ¬ (m ∥ n) :=
begin
  sorry
end

end incorrect_statement_D_l179_179539


namespace peacock_count_l179_179085

theorem peacock_count : ∃ n : ℕ, (∀ p : ℕ, 
  (p > 0) ∧ (∀ d, d ∈ finset.univ (fin 10) → digit_in_number p d) ∧ 
  (∀ t : ℕ, 2*t = p → (∀ d, d ∈ finset.univ (fin 10) → digit_in_number t d) → 
  (digit_count t = 10) ∧ (digit_count p = 10)) → p ∈ peacock_numbers) ∧ 
  n = 184320 :=
begin
  sorry
end

end peacock_count_l179_179085


namespace pyramid_volume_correct_l179_179090

noncomputable def pyramid_volume (a b e : ℕ) (area height : ℝ) : ℝ :=
  (1 / 3) * area * height

theorem pyramid_volume_correct :
  let a := 7
  let b := 9
  let e := 15
  let base_area := (a:ℝ) * (b:ℝ)
  let diagonal := real.sqrt ((a ^ 2) + (b ^ 2))
  let center_to_corner := diagonal / 2
  let height := real.sqrt ((e ^ 2) - (center_to_corner ^ 2))
  pyramid_volume a b e base_area height = 84 * real.sqrt 10 := 
by
  sorry

end pyramid_volume_correct_l179_179090


namespace modulus_of_z_l179_179811

noncomputable def z : ℂ := 1 / (1 + complex.i) - complex.i

theorem modulus_of_z : complex.abs z = real.sqrt 10 / 2 := by
  sorry

end modulus_of_z_l179_179811


namespace largest_k_14_eq_5_l179_179472

-- Define the factorial function
def fact : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Definition of divisible
def divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

-- Legendre's formula to count the power of a prime p in n!
def legendre (n p : ℕ) : ℕ :=
  if h : p > 1 then
    Nat.sum (fun i => n / p ^ i) (h.trans_le n)
  else
    0

-- Define the largest_k function
def largest_k (n : ℕ) : ℕ :=
  min (legendre n 2) (legendre n 3)

theorem largest_k_14_eq_5 : largest_k 14 = 5 := by
  -- Add the proof here
  sorry

end largest_k_14_eq_5_l179_179472


namespace monotonic_intervals_and_range_l179_179381

noncomputable def function_y (x : ℝ) : ℝ :=
  (1 / 4) ^ x - (1 / 2) ^ x + 1

theorem monotonic_intervals_and_range :
  ∀ x ∈ Icc (-3 : ℝ) 2,
  (∀ x1 x2 ∈ Icc (-3 : ℝ) 1, x1 < x2 → function_y x1 > function_y x2) ∧
  (∀ x1 x2 ∈ Icc (1 : ℝ) 2, x1 < x2 → function_y x1 < function_y x2) ∧
  range function_y = set.Icc (3 / 4) 57 :=
sorry

end monotonic_intervals_and_range_l179_179381


namespace western_region_area_scientific_notation_l179_179588

-- Definition for scientific notation
def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * 10^n

-- Condition provided in the problem
def million := 10^6
def western_region_area : ℝ := 6.4 * million

-- Statement of the proof 
theorem western_region_area_scientific_notation :
  scientific_notation 6.4 6 = western_region_area :=
by sorry

end western_region_area_scientific_notation_l179_179588


namespace no_term_is_prime_l179_179658

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def Q : ℕ := ∏ p in (Finset.filter Nat.prime (Finset.range 68)), p

theorem no_term_is_prime : ∀ n : ℕ, 2 ≤ n ∧ n ≤ 73 → ¬ is_prime (Q + n) :=
by
  sorry

end no_term_is_prime_l179_179658


namespace sin_45_deg_l179_179198

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179198


namespace distribution_ways_l179_179452

-- Definitions based on the conditions from part a)
def distinguishable_balls : ℕ := 6
def indistinguishable_boxes : ℕ := 4

-- The theorem to prove the question equals the correct answer given the conditions
theorem distribution_ways (n : ℕ) (k : ℕ) (h_n : n = distinguishable_balls) (h_k : k = indistinguishable_boxes) : 
  number_of_distributions n k = 262 := 
sorry

end distribution_ways_l179_179452


namespace weeks_jake_buys_papayas_l179_179872

theorem weeks_jake_buys_papayas
  (jake_papayas : ℕ)
  (brother_papayas : ℕ)
  (father_papayas : ℕ)
  (total_papayas : ℕ)
  (h1 : jake_papayas = 3)
  (h2 : brother_papayas = 5)
  (h3 : father_papayas = 4)
  (h4 : total_papayas = 48) :
  (total_papayas / (jake_papayas + brother_papayas + father_papayas) = 4) :=
by
  sorry

end weeks_jake_buys_papayas_l179_179872


namespace exponent_of_5_in_30_factorial_l179_179953

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179953


namespace arrange_in_non_increasing_order_l179_179122

noncomputable def Psi : ℤ := (1/2) * ∑ k in finset.range 503, -4
noncomputable def Omega : ℤ := ∑ k in finset.range 1007, -1
noncomputable def Theta : ℤ := ∑ k in finset.range 504, -2

theorem arrange_in_non_increasing_order :
  Theta ≤ Omega ∧ Omega ≤ Psi :=
begin
  -- Proof to be implemented
  sorry,
end

end arrange_in_non_increasing_order_l179_179122


namespace find_a_and_b_and_prove_inequality_l179_179819

-- Given definition of f(x)
def f (x : ℝ) (a b : ℝ) : ℝ := (a * (Real.log x) / (x + 1)) + (b / x)

-- Condition for the tangent line at (1, f(1))
def tangent_condition (a b : ℝ) : Prop := 
  let m := - (1 / 2) in -- slope of the tangent line
  f 1 a b = 1 ∧ m = (f' 1 a b) -- making sure the function values and derivatives match accordingly, this simplifies to given system

-- The inequality to prove for x > 0 and x ≠ 1
def inequality_condition (x : ℝ) : Prop :=
  x > 0 ∧ x ≠ 1

theorem find_a_and_b_and_prove_inequality (a b : ℝ) : 
  tangent_condition a b →
  ∀ x : ℝ, inequality_condition x → f x a b > (Real.log x / (x - 1)) :=
by
  intro h_condition -- assume the tangent condition holds
  intro x h_inequality -- assume the conditions on x
  sorry -- the actual proof will be filled in here.

end find_a_and_b_and_prove_inequality_l179_179819


namespace exponent_of_5_in_30_factorial_l179_179963

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179963


namespace max_value_sin_cos_combination_l179_179376

theorem max_value_sin_cos_combination :
  ∀ x : ℝ, (5 * Real.sin x + 12 * Real.cos x) ≤ 13 :=
by
  intro x
  sorry

end max_value_sin_cos_combination_l179_179376


namespace lines_intersect_at_point_l179_179691

def point := ℝ × ℝ

def line1 (t : ℝ) : point :=
  (1 - 3 * t, 1 + 4 * t)

def line2 (u : ℝ) : point :=
  (2 + 6 * u, -7 - u)

theorem lines_intersect_at_point :
  ∃ (t u : ℝ), line1 t = (5.5, -5) ∧ line2 u = (5.5, -5) :=
by
  use [-3/2, 1]
  simp [line1, line2]
  norm_num
  sorry

end lines_intersect_at_point_l179_179691


namespace find_imaginary_m_l179_179481

variable (m : ℝ)

theorem find_imaginary_m (h : (2 + 3 * Complex.I) * (1 - m * Complex.I)).re = 0 : m = -2 / 3 :=
sorry

end find_imaginary_m_l179_179481


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179231

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179231


namespace relay_order_count_correct_l179_179868

def relay_team_members : Type := { member : String // member ∈ ["Alex", "Jordan", "Blake", "Casey", "Dana"] }

def lap_order_valid (order : List relay_team_members) : Prop :=
  order.head = "Alex" ∧ order.last = "Jordan" ∧ 
  ∀ x ∈ ["Blake", "Casey", "Dana"], x ∈ order.tail.dropLast 3

theorem relay_order_count_correct :
  ∃ order : List relay_team_members, lap_order_valid order → order.tail.dropLast 3.permutations.length = 6 :=
by 
  sorry

end relay_order_count_correct_l179_179868


namespace sin_45_eq_sqrt2_div_2_l179_179206

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179206


namespace known_fourier_transform_fourier_transform_of_f_l179_179759

-- Definition: Given function f
def f (x : ℝ) : ℝ := 1 / (x^2 + 2*x + 2)

-- Given known Fourier transform formula
theorem known_fourier_transform :
  ∀ (p : ℝ), 
  fourier_transform (λ x : ℝ, 1 / (x^2 + 1)) p = sqrt (π / 2) * exp (-abs p) := sorry

-- The main theorem to be proved
theorem fourier_transform_of_f :
  ∀ (p : ℝ),
  fourier_transform f p = sqrt (π / 2) * exp (-abs p + complex.I * p) := sorry

end known_fourier_transform_fourier_transform_of_f_l179_179759


namespace sin_45_degree_l179_179274

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179274


namespace skew_lines_no_common_perpendicular_plane_l179_179840

noncomputable def check_condition (a b : Line) (α : Plane) : Prop :=
  ∃ (α : Plane), (skew a b) ∧ (¬ (perpendicular α a ∧ perpendicular α b))

theorem skew_lines_no_common_perpendicular_plane (a b : Line) :
  skew a b → ¬ (∃ α : Plane, perpendicular α a ∧ perpendicular α b) :=
by
  intro h
  exact check_condition a b (arbitrary Plane)
  sorry

end skew_lines_no_common_perpendicular_plane_l179_179840


namespace exponent_of_5_in_30_factorial_l179_179990

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179990


namespace gerbils_sold_eq_69_l179_179087

theorem gerbils_sold_eq_69 :
  ∀ (initial_gerbils left_gerbils : ℕ), initial_gerbils = 85 → left_gerbils = 16 → initial_gerbils - left_gerbils = 69 :=
by
  intros initial_gerbils left_gerbils
  intros h1 h2
  rw [h1, h2]
  exact Nat.sub_eq_of_eq_add (rfl)
  sorry

end gerbils_sold_eq_69_l179_179087


namespace average_percentage_reduction_l179_179679

theorem average_percentage_reduction (initial_price reduced_price : ℝ) (price_drop : ℝ) (reductions : ℕ) 
  (h1 : initial_price = 25) (h2 : reduced_price = 16) (h3 : reductions = 2) 
  (h4 : initial_price * (1 - price_drop)^reductions = reduced_price) : 
  price_drop = 0.20 :=
by
  have h5 : (1 - price_drop)^2 = 16 / 25 := by
    rw [h1, h2, mul_pow]
    simp
  have h6 : 1 - price_drop = 4 / 5 := by
    rw [pow_two, eq_div_iff]
    simp [sub_eq_iff_eq_add']
  have h7 : price_drop = 1 - 4 / 5 := by
    linarith
  exact h7.symm

end average_percentage_reduction_l179_179679


namespace systematic_sampling_of_bicycles_on_main_road_l179_179771

def is_systematic_sampling (sampling_method: Type) : Prop := 
  ∃ (fixed_sampling_interval: ℕ), ∀ (bicycles: List ℕ), 
  -- sampling applied with a fixed interval condition
  ∀ (bicycle_number: ℕ), bicycle_number ∈ bicycles → 
    (bicycle_number % fixed_sampling_interval = 0)

def six_digit_license_plate (bicycle: ℕ) : Prop :=
  100000 ≤ bicycle ∧ bicycle < 1000000

def samples_target_on_main_road (bicycle: ℕ) : Prop :=
  -- True for the sake of defining the condition; further context would be needed in a real setting
  true

theorem systematic_sampling_of_bicycles_on_main_road :
  ∀ (sampling_method: Type)
  (bicycles: List ℕ),
  (∀ (bicycle: ℕ), bicycle ∈ bicycles → six_digit_license_plate bicycle) →
  (∀ (bicycle: ℕ), bicycle ∈ bicycles → samples_target_on_main_road bicycle) →
  is_systematic_sampling sampling_method :=
begin
  sorry
end

end systematic_sampling_of_bicycles_on_main_road_l179_179771


namespace greatest_integer_less_than_neg22_div_3_l179_179641

def greatest_integer_less_than (x : ℝ) : ℤ :=
  int.floor x

theorem greatest_integer_less_than_neg22_div_3 : greatest_integer_less_than (-22 / 3) = -8 := by
  sorry

end greatest_integer_less_than_neg22_div_3_l179_179641


namespace prime_exponent_of_5_in_30_factorial_l179_179909

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179909


namespace problem_fraction_eq_l179_179021

theorem problem_fraction_eq (x : ℝ) :
  (x * (3 / 4) * (1 / 2) * 5060 = 759.0000000000001) ↔ (x = 0.4) :=
by
  sorry

end problem_fraction_eq_l179_179021


namespace tan_double_angle_eq_neg_4_over_3_l179_179795

theorem tan_double_angle_eq_neg_4_over_3
  (h_alpha : α ∈ Ioo π (bit0 π))
  (h_cos_alpha : Real.cos α = -Real.sqrt 5 / 5) :
  Real.tan (bit0 α) = -4 / 3 :=
sorry

end tan_double_angle_eq_neg_4_over_3_l179_179795


namespace exponent_of_5_in_30_factorial_l179_179989

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179989


namespace sin_45_eq_sqrt2_div_2_l179_179286

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179286


namespace problem_statement_l179_179429

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sin (x - π / 2) ^ 2
noncomputable def g (x : ℝ) : ℝ := f (x - π / 12) - 1

theorem problem_statement :
  (∀ x : ℝ, f x = 2 * cos x ^ 2) ∧
  (∀ x : ℝ, f x = 1 + cos (2 * x) ∧ even_fun : ∀ x : ℝ, f x = f (-x)) ∧
  g x = cos (2 * x - π / 6) ∧
  (∀ k : ℤ, axis_symm : ∀ x : ℝ, (2 * x - π / 6 = k * π) →  x = (k * π / 2 + π / 12)) :=
by
  sorry

end problem_statement_l179_179429


namespace min_value_f_l179_179655

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x)^2 / (Real.cos x * Real.sin x - (Real.sin x)^2)

theorem min_value_f :
  ∃ x : ℝ, 0 < x ∧ x < Real.pi / 4 ∧ f x = 4 := 
sorry

end min_value_f_l179_179655


namespace prime_exponent_of_5_in_30_factorial_l179_179914

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179914


namespace sin_45_eq_sqrt2_div_2_l179_179210

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179210


namespace log_mono_decreasing_iff_l179_179614

theorem log_mono_decreasing_iff
  (m : ℝ) :
  (-3 < m ∧ m < 0) ↔
  ∀ x y : ℝ, (x <= 1 ∧ y <= 1 ∧ x < y) → ln (m * x + 3) > ln (m * y + 3) :=
by sorry

end log_mono_decreasing_iff_l179_179614


namespace smarties_division_l179_179754

theorem smarties_division (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end smarties_division_l179_179754


namespace log_function_decreasing_on_interval_l179_179617

theorem log_function_decreasing_on_interval (m : ℝ) :
  (-3 < m ∧ m < 0) ↔ ∀ x : ℝ, x ≤ 1 → f'(x) < 0 :=
begin
  sorry
end

-- Definitions needed for the problem
noncomputable def f (x : ℝ) : ℝ := log (m * x + 3)

noncomputable def f' (x : ℝ) : ℝ := (differential (log (m * x + 3))).eval x

end log_function_decreasing_on_interval_l179_179617


namespace sin_45_deg_eq_l179_179299

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179299


namespace exponent_of_5_in_30_fact_l179_179878

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179878


namespace sin_45_degree_l179_179172

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179172


namespace non_prime_count_l179_179350

def sum_of_digits (n : Nat) : Nat :=
  n.digits.sum

def digits_greater_than_one (n : Nat) : Prop :=
  ∀ d ∈ n.digits, d > 1

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem non_prime_count : 
  {n : Nat // sum_of_digits n = 7 ∧ digits_greater_than_one n ∧ ¬is_prime n }.card = 5 :=
by
  sorry

end non_prime_count_l179_179350


namespace minimum_distance_to_capture_all_cyclists_l179_179007

theorem minimum_distance_to_capture_all_cyclists 
  (v1 v2 v3 : ℝ) (h1 : v1 > v2) (h2 : v2 > v3) :
  ∃ (d : ℝ), d = 75 :=
by
  have h : ∀ d, d ≥ 75, sorry
  use 75
  exact h 75

# Output should: minimum_distance_to_capture_all_cyclists

end minimum_distance_to_capture_all_cyclists_l179_179007


namespace range_fx_in_interval_l179_179816

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem range_fx_in_interval :
  ∀ x ∈ set.Icc (0 : ℝ) 3, 2 ≤ f x ∧ f x ≤ 6 := by
  sorry

end range_fx_in_interval_l179_179816


namespace shifted_function_expression_l179_179419

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x + Real.pi / 3)

theorem shifted_function_expression (ω : ℝ) (h : ℝ) (x : ℝ) (h_positive : ω > 0) (h_period : Real.pi = 2 * Real.pi / ω) :
  f ω (x + h) = Real.cos (2 * x) :=
by
  -- We assume h = π/12, ω = 2
  have ω_val : ω = 2 := by sorry
  have h_val : h = Real.pi / 12 := by sorry
  rw [ω_val, h_val]
  sorry

end shifted_function_expression_l179_179419


namespace arrangement_count_proof_l179_179621

noncomputable def num_arrangements : ℕ :=
  let num_ways_elderly := 2      -- 2 ways to arrange the two elderly people in the center
  let num_ways_females := 2      -- 2 ways to arrange the two female volunteers next to elderly people
  let num_ways_males := 24       -- 4! ways to arrange the 4 male volunteers in the remaining positions
  num_ways_elderly * num_ways_females * num_ways_males

theorem arrangement_count_proof : num_arrangements = 96 := by
  unfold num_arrangements
  norm_num
  sorry

end arrangement_count_proof_l179_179621


namespace sin_45_eq_sqrt2_div_2_l179_179327

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179327


namespace find_value_of_k_l179_179515

noncomputable def value_of_k (m n : ℝ) : ℝ :=
  let p := 0.4
  let point1 := (m, n)
  let point2 := (m + 2, n + p)
  let k := 5
  k

theorem find_value_of_k (m n : ℝ) : value_of_k m n = 5 :=
sorry

end find_value_of_k_l179_179515


namespace symmetry_of_f_l179_179432

noncomputable def f (x ω φ : ℝ) := Math.sin (ω * x + φ)

theorem symmetry_of_f (ω : ℝ) (φ : ℝ)
  (h1 : ω > 0)
  (h2 : abs φ < Real.pi / 2)
  (h3 : (2 * Real.pi) / ω = Real.pi)
  (h4 : ∀ x, f (x - Real.pi / 3) ω φ = Math.cos (ω * x)) :
  ∃ a, ∀ x, f x ω φ = f (2 * a - x) ω φ ∧ a = Real.pi / 12 ∧ f a ω φ = 0 :=
by
  sorry

end symmetry_of_f_l179_179432


namespace exists_multiple_with_zero_one_digits_of_length_at_most_n_l179_179573

theorem exists_multiple_with_zero_one_digits_of_length_at_most_n (n : ℕ) :
  ∃ k, k < 10^n ∧ (k % n = 0) ∧ (∀ d ∈ (finset.digits 10 k), d = 0 ∨ d = 1) :=
sorry

end exists_multiple_with_zero_one_digits_of_length_at_most_n_l179_179573


namespace integer_pairs_count_l179_179834

theorem integer_pairs_count :
  set.countable {p : ℤ × ℤ | let (x, y) := p in x / (y + 7) + y / (x + 7) = 1} = 14 := sorry

end integer_pairs_count_l179_179834


namespace sin_45_eq_1_div_sqrt_2_l179_179316

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179316


namespace exists_disk_of_radius_one_containing_1009_points_l179_179002

theorem exists_disk_of_radius_one_containing_1009_points
  (points : Fin 2017 → ℝ × ℝ)
  (h : ∀ (a b c : Fin 2017), (dist (points a) (points b) < 1) ∨ (dist (points b) (points c) < 1) ∨ (dist (points c) (points a) < 1)) :
  ∃ (center : ℝ × ℝ), ∃ (sub_points : Finset (Fin 2017)), sub_points.card ≥ 1009 ∧ ∀ p ∈ sub_points, dist (center) (points p) ≤ 1 :=
sorry

end exists_disk_of_radius_one_containing_1009_points_l179_179002


namespace sin_45_eq_sqrt2_div_2_l179_179202

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179202


namespace intersection_points_of_graph_and_line_l179_179450

theorem intersection_points_of_graph_and_line (f : ℝ → ℝ) :
  (∀ x : ℝ, f x ≠ my_special_value) → (∀ x₁ x₂ : ℝ, f x₁ = f x₂ → x₁ = x₂) →
  ∃! x : ℝ, x = 1 ∧ ∃ y : ℝ, y = f x :=
by
  sorry

end intersection_points_of_graph_and_line_l179_179450


namespace sin_45_eq_l179_179256

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179256


namespace exponent_of_5_in_30_factorial_l179_179894

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179894


namespace quadratic_roots_eq1_quadratic_roots_eq2_l179_179581

theorem quadratic_roots_eq1 :
  ∀ x : ℝ, (x^2 + 3 * x - 1 = 0) ↔ (x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) :=
by
  intros x
  sorry

theorem quadratic_roots_eq2 :
  ∀ x : ℝ, ((x + 2)^2 = (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intros x
  sorry

end quadratic_roots_eq1_quadratic_roots_eq2_l179_179581


namespace exponent_of_5_in_30_factorial_l179_179974

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179974


namespace solution_set_of_inequality_l179_179538

variable {f : ℝ → ℝ} 
variable [Differentiable ℝ f]
variable H : ∀ x < 0, 3 * f x + x * (deriv f) x < 0

theorem solution_set_of_inequality : 
  {x : ℝ | (x + 2016)^3 * f (x + 2016) + 8 * f (-2) < 0} = Ioo (-2018) (-2016) :=
by
  sorry

end solution_set_of_inequality_l179_179538


namespace distance_two_points_circle_l179_179810

theorem distance_two_points_circle {b : ℝ} :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁^2 + y₁^2 = 9 ∧ x₂^2 + y₂^2 = 9 ∧ 
    let l := y = x + b in 
    abs (y₁ - (x₁ + b)) = 1 ∧ abs (y₂ - (x₂ + b)) = 1) ↔ 
  (b ∈ set.Ioo (-4 * real.sqrt 2) (-2 * real.sqrt 2) ∨ b ∈ set.Ioo (2 * real.sqrt 2) (4 * real.sqrt 2)) :=
sorry

end distance_two_points_circle_l179_179810


namespace sin_45_degree_l179_179170

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179170


namespace exponent_of_5_in_30_factorial_l179_179944

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179944


namespace fraction_of_1840_to_1849_states_l179_179569

theorem fraction_of_1840_to_1849_states (total_states : ℕ) (states_from_1840_to_1849 : ℕ) :
  total_states = 33 → states_from_1840_to_1849 = 6 → 
  (states_from_1840_to_1849 : ℚ) / (total_states : ℚ) = 2 / 11 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end fraction_of_1840_to_1849_states_l179_179569


namespace exact_three_blue_probability_l179_179673

theorem exact_three_blue_probability :
  let total_ways := Nat.choose 15 4
  let ways_three_blue := Nat.choose 4 3
  let ways_one_non_blue := Nat.choose 11 1
  let successful_outcomes := ways_three_blue * ways_one_non_blue
  in (successful_outcomes : ℚ) / (total_ways : ℚ) = 44 / 1365 := 
by
  sorry

end exact_three_blue_probability_l179_179673


namespace sin_45_eq_sqrt_two_over_two_l179_179242

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179242


namespace sin_45_degree_eq_sqrt2_div_2_l179_179151

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179151


namespace prime_exponent_of_5_in_30_factorial_l179_179912

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179912


namespace find_second_derivative_at_1_l179_179780

-- Define the function f(x) and its second derivative
noncomputable def f (x : ℝ) := x * Real.exp x
noncomputable def f'' (x : ℝ) := (x + 2) * Real.exp x

-- State the theorem to be proved
theorem find_second_derivative_at_1 : f'' 1 = 2 * Real.exp 1 := by
  sorry

end find_second_derivative_at_1_l179_179780


namespace value_of_m_if_pure_imaginary_l179_179482

variable (m : ℝ)

def z : ℂ := (2 + 3 * complex.i) * (1 - m * complex.i)

-- Definition that z is pure imaginary means the real part is 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem value_of_m_if_pure_imaginary :
  is_pure_imaginary (z m) -> m = -2 / 3 :=
by sorry

end value_of_m_if_pure_imaginary_l179_179482


namespace number_of_possible_scenarios_l179_179622

theorem number_of_possible_scenarios 
  (subjects : ℕ) 
  (students : ℕ) 
  (h_subjects : subjects = 4) 
  (h_students : students = 3) : 
  (subjects ^ students) = 64 := 
by
  -- Provide proof here
  sorry

end number_of_possible_scenarios_l179_179622


namespace sin_45_eq_sqrt2_div_2_l179_179332

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179332


namespace six_hundred_sixes_not_square_l179_179565

theorem six_hundred_sixes_not_square : 
  ∀ (n : ℕ), (n = 66666666666666666666666666666666666666666666666666666666666 -- continued 600 times
  ∨ n = 66666666666666666666666666666666666666666666666666666666666 -- continued with some zeros
  ) → ¬ (∃ k : ℕ, k * k = n) := 
by
  sorry

end six_hundred_sixes_not_square_l179_179565


namespace calculate_expression_l179_179140

theorem calculate_expression : 4 + (-8) / (-4) - (-1) = 7 := 
by 
  sorry

end calculate_expression_l179_179140


namespace exponent_of_5_in_30_factorial_l179_179995

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179995


namespace last_digit_of_even_indexed_is_zero_l179_179728

def modified_fibonacci : ℕ → ℕ
| 1     := 3
| 2     := 4
| (n+3) := modified_fibonacci (n+2) + modified_fibonacci (n+1)

theorem last_digit_of_even_indexed_is_zero :
  ∃ n : ℕ, modified_fibonacci (2*n) % 10 = 0 :=
sorry

end last_digit_of_even_indexed_is_zero_l179_179728


namespace sin_45_eq_sqrt2_div_2_l179_179200

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179200


namespace day_of_week_june_1_2014_l179_179347

/--
  Given:
    - December 31, 2013, is a Tuesday (represented as 2).
    - The year 2014 is not a leap year.
    - Days of the week are numbered as follows:
      Monday = 1, Tuesday = 2, ..., Sunday = 7.
  Prove that June 1, 2014, is a Sunday.
-/
theorem day_of_week_june_1_2014 : 
  let dec_31_2013 := 2
      days_in_jan := 31
      days_in_feb := 28  -- 2014 is not a leap year
      days_in_mar := 31
      days_in_apr := 30
      days_in_may := 31
      june_1 := 1
      total_days := days_in_jan + days_in_feb + days_in_mar + days_in_apr + days_in_may + june_1
      weeks := total_days / 7
      extra_days := total_days % 7
  in (dec_31_2013 + extra_days) % 7 = 7 :=
by
  sorry

end day_of_week_june_1_2014_l179_179347


namespace find_polynomial_l179_179366

open Polynomial

-- Defining the properties of the polynomial P
def P (p : Polynomial ℤ) : Prop :=
  ∀ s t : ℝ, (P.eval s ∈ ℤ ∧ P.eval t ∈ ℤ) → P.eval (s * t) ∈ ℤ

-- The main theorem statement
theorem find_polynomial (P : Polynomial ℤ) :
  (∀ s t : ℝ, P.eval s ∈ ℤ ∧ P.eval t ∈ ℤ → P.eval (s * t) ∈ ℤ) →
  (∃ c : ℤ, ∃ d : ℕ, P = Polynomial.C c + Polynomial.X ^ d) :=
sorry

end find_polynomial_l179_179366


namespace ball_distribution_result_l179_179459

theorem ball_distribution_result :
  ∀ (balls boxes : ℕ), balls = 6 → boxes = 4 →
  let ways : ℕ :=
    1 +  -- (6,0,0,0)
    6 +  -- (5,1,0,0)
    15 + -- (4,2,0,0)
    15 + -- (4,1,1,0)
    10 + -- (3,3,0,0)
    60 + -- (3,2,1,0)
    20 + -- (3,1,1,1)
    15 + -- (2,2,2,0)
    15   -- (2,2,1,1)
  in ways = 157 :=
by
  intros balls boxes hb hb'
  rw [hb, hb']
  sorry

end ball_distribution_result_l179_179459


namespace greatest_value_l179_179022

theorem greatest_value (y : ℝ) (h : 4 * y^2 + 4 * y + 3 = 1) : (y + 1)^2 = 1/4 :=
sorry

end greatest_value_l179_179022


namespace find_imaginary_m_l179_179480

variable (m : ℝ)

theorem find_imaginary_m (h : (2 + 3 * Complex.I) * (1 - m * Complex.I)).re = 0 : m = -2 / 3 :=
sorry

end find_imaginary_m_l179_179480


namespace prime_exponent_of_5_in_30_factorial_l179_179913

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179913


namespace exponent_of_5_in_30_fact_l179_179885

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179885


namespace elongation_rate_significantly_increased_l179_179676

-- Definitions for the experimental data 
def x : List ℝ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def y : List ℝ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

def z : List ℝ := List.map₂ (λ a b => a - b) x y

-- Function to calculate mean
def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Function to calculate variance
def variance (l : List ℝ) (mean_l : ℝ) : ℝ :=
  (List.sum (List.map (λ x => (x - mean_l) ^ 2) l)) / l.length

-- Given problem encapsulated as a Lean theorem statement
theorem elongation_rate_significantly_increased :
  let mean_z := mean z in
  let var_z := variance z mean_z in
  mean_z = 11 ∧ var_z = 61 ∧ mean_z ≥ 2 * real.sqrt (var_z / 10) :=
by
  let mean_z := mean z
  let var_z := variance z mean_z
  have h_mean : mean_z = 11 := by sorry  -- Proof of mean calculation
  have h_var : var_z = 61 := by sorry   -- Proof of variance calculation
  have h_significant : mean_z ≥ 2 * real.sqrt (var_z / 10) := by sorry -- Proof of significance
  exact ⟨h_mean, h_var, h_significant⟩

end elongation_rate_significantly_increased_l179_179676


namespace exponent_of_5_in_30_factorial_l179_179936

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179936


namespace oil_leakage_problem_l179_179626

theorem oil_leakage_problem :
    let l_A := 25  -- Leakage rate of Pipe A (gallons/hour)
    let l_B := 37  -- Leakage rate of Pipe B (gallons/hour)
    let l_C := 55  -- Leakage rate of Pipe C (gallons/hour)
    let l_D := 41  -- Leakage rate of Pipe D (gallons/hour)
    let l_E := 30  -- Leakage rate of Pipe E (gallons/hour)

    let t_A := 10  -- Time taken to fix Pipe A (hours)
    let t_B := 7   -- Time taken to fix Pipe B (hours)
    let t_C := 12  -- Time taken to fix Pipe C (hours)
    let t_D := 9   -- Time taken to fix Pipe D (hours)
    let t_E := 14  -- Time taken to fix Pipe E (hours)

    let leak_A := l_A * t_A  -- Total leaked from Pipe A (gallons)
    let leak_B := l_B * t_B  -- Total leaked from Pipe B (gallons)
    let leak_C := l_C * t_C  -- Total leaked from Pipe C (gallons)
    let leak_D := l_D * t_D  -- Total leaked from Pipe D (gallons)
    let leak_E := l_E * t_E  -- Total leaked from Pipe E (gallons)
  
    let overall_total := leak_A + leak_B + leak_C + leak_D + leak_E
  
    leak_A = 250 ∧
    leak_B = 259 ∧
    leak_C = 660 ∧
    leak_D = 369 ∧
    leak_E = 420 ∧
    overall_total = 1958 :=
by
    sorry

end oil_leakage_problem_l179_179626


namespace distinct_flavors_count_l179_179577

theorem distinct_flavors_count : 
  let reds := 6
  let greens := 5
  (finset.filter (λ p : ℕ × ℕ, p.1 ≤ reds ∧ p.2 ≤ greens ∧ (p.1 ≠ 0 ∨ p.2 ≠ 0)) 
  ((finset.range (reds + 1)).product (finset.range (greens + 1)))))
  .card = 20 := 
by 
  sorry

end distinct_flavors_count_l179_179577


namespace exponent_of_5_in_30_factorial_l179_179983

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179983


namespace conic_section_type_l179_179037

theorem conic_section_type (x y : ℝ) : 
  (|y + 5| = real.sqrt ((x - 2)^2 + y^2)) → "P" := 
by
  sorry

end conic_section_type_l179_179037


namespace min_value_when_a_equals_1_range_of_a_for_f_geq_a_l179_179815

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem min_value_when_a_equals_1 : 
  ∃ x, f x 1 = 1 :=
by
  sorry

theorem range_of_a_for_f_geq_a (a : ℝ) :
  (∀ x, x ≥ -1 → f x a ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end min_value_when_a_equals_1_range_of_a_for_f_geq_a_l179_179815


namespace exponent_of_5_in_30_factorial_l179_179947

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179947


namespace exponent_of_5_in_30_factorial_l179_179966

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179966


namespace hyperbola_real_axis_length_l179_179823

theorem hyperbola_real_axis_length {a : ℝ} (h : 0 < a) :
  (let c := sqrt (a^2 + 4) in (2 - c)^2 + 1^2 = (2 + c)^2) ->
  2 * a = 2 :=
by
  sorry

end hyperbola_real_axis_length_l179_179823


namespace album_cost_l179_179138

-- Definitions for given conditions
def M (X : ℕ) : ℕ := X - 2
def K (X : ℕ) : ℕ := X - 34
def F (X : ℕ) : ℕ := X - 35

-- We need to prove that X = 35
theorem album_cost : ∃ X : ℕ, (M X) + (K X) + (F X) < X ∧ X = 35 :=
by
  sorry -- Proof not required.

end album_cost_l179_179138


namespace angle_QSP_l179_179671

theorem angle_QSP {P Q R S : Type*} (h : right_triangle P Q R)
  (hP : angle PQR P = 30)
  (hQ : angle PRQ Q = 60)
  (hR : angle bisector QS (angle QRP) S) : 
  angle QSP = 105 :=
by 
  -- Proof goes here, for now we use sorry
  sorry

end angle_QSP_l179_179671


namespace average_salary_increase_l179_179591

theorem average_salary_increase :
  ∀ (num_employees : ℕ) (avg_salary_per_employee manager_salary new_num_employees new_avg_salary : ℚ)
  (h1 : num_employees = 20)
  (h2 : avg_salary_per_employee = 1600)
  (h3 : manager_salary = 3700)
  (h4 : new_num_employees = num_employees + 1)
  (h5 : new_avg_salary = (num_employees * avg_salary_per_employee + manager_salary) / new_num_employees),
  new_avg_salary - avg_salary_per_employee = 100 :=
by
  intros
  rw [h1, h2, h3, h4, h5]
  sorry

end average_salary_increase_l179_179591


namespace exponent_of_5_in_30_factorial_l179_179942

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179942


namespace length_of_BC_equals_DE_l179_179403

-- Definitions of the required geometric elements and properties

structure Geometry (Point : Type) :=
  (O : Point) -- Center of the original semicircle
  (A : Point) -- An arbitrary point on the diameter
  (B C D E : Point) -- Points of intersection with semicircles

-- Predicates or relations that describe the properties and relationships
-- of the points generally and with specific constructs (e.g., semicircle, line, etc.).

-- Assuming length and midpoint definitions exist in the mathematical library

noncomputable def length_of_segment {Point : Type} [MetricSpace Point] (p1 p2 : Point) : ℝ := sorry
noncomputable def midpoint_of_segment {Point : Type} [MetricSpace Point] (p1 p2 : Point) : Point := sorry

axiom properties_of_intersections {Point : Type} [MetricSpace Point] [Geometry Point] :
  ∀ (geom : Geometry Point),
    let AB := length_of_segment geom.A geom.B,
        AE := length_of_segment geom.A geom.E,
        M_BE := midpoint_of_segment geom.B geom.E,
        M_CD := midpoint_of_segment geom.C geom.D in
    M_BE = M_CD →
    length_of_segment geom.B geom.C = length_of_segment geom.D geom.E

-- Theorem statement
theorem length_of_BC_equals_DE {Point : Type} [MetricSpace Point] (geom : Geometry Point):
  length_of_segment geom.B geom.C = length_of_segment geom.D geom.E :=
begin
  apply properties_of_intersections,
  sorry
end

end length_of_BC_equals_DE_l179_179403


namespace positive_difference_of_solutions_of_absolute_equation_l179_179023

theorem positive_difference_of_solutions_of_absolute_equation :
  (|x - 4| = 15) → (|19 - (-11)| = 30) :=
by
  sorry

end positive_difference_of_solutions_of_absolute_equation_l179_179023


namespace magnitude_of_a_add_b_k_value_if_parallel_l179_179445

def vector2d := (ℝ × ℝ)

-- Define the vectors
def a : vector2d := (3, 2)
def b : vector2d := (0, 2)
def c : vector2d := (4, 1)

-- Magnitude of a vector
def magnitude (v : vector2d) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Problem 1: Prove the magnitude of a + b is 5
theorem magnitude_of_a_add_b : magnitude (a.1 + b.1, a.2 + b.2) = 5 := 
  sorry

-- Problem 2: Prove k = 3 if a + k * c is parallel to 2 * a - b
theorem k_value_if_parallel : 
  ∀ k : ℝ, ((a.1 + k * c.1), (a.2 + k * c.2)) = ((2 * a.1 - b.1), (2 * a.2 - b.2)) → k = 3 := 
  sorry

end magnitude_of_a_add_b_k_value_if_parallel_l179_179445


namespace even_perfect_sum_of_two_cubes_unique_l179_179692

def is_perfect (n : ℕ) : Prop :=
  (∑ d in finset.filter (∣) (nat.divisors n) (λ d, d ≠ n)) = n

def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = a^3 + b^3

theorem even_perfect_sum_of_two_cubes_unique :
  ∀ n, is_perfect n ∧ even n ∧ is_sum_of_two_cubes n ↔ n = 28 :=
by
  sorry

end even_perfect_sum_of_two_cubes_unique_l179_179692


namespace number_of_balls_selected_is_three_l179_179065

-- Definitions of conditions
def total_balls : ℕ := 100
def odd_balls_selected : ℕ := 2
def even_balls_selected : ℕ := 1
def probability_first_ball_odd : ℚ := 2 / 3

-- The number of balls selected
def balls_selected := odd_balls_selected + even_balls_selected

-- Statement of the proof problem
theorem number_of_balls_selected_is_three 
(h1 : total_balls = 100)
(h2 : odd_balls_selected = 2)
(h3 : even_balls_selected = 1)
(h4 : probability_first_ball_odd = 2 / 3) :
  balls_selected = 3 :=
sorry

end number_of_balls_selected_is_three_l179_179065


namespace exponent_of_5_in_30_factorial_l179_179959

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179959


namespace sin_45_eq_1_div_sqrt_2_l179_179321

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179321


namespace ball_distribution_result_l179_179460

theorem ball_distribution_result :
  ∀ (balls boxes : ℕ), balls = 6 → boxes = 4 →
  let ways : ℕ :=
    1 +  -- (6,0,0,0)
    6 +  -- (5,1,0,0)
    15 + -- (4,2,0,0)
    15 + -- (4,1,1,0)
    10 + -- (3,3,0,0)
    60 + -- (3,2,1,0)
    20 + -- (3,1,1,1)
    15 + -- (2,2,2,0)
    15   -- (2,2,1,1)
  in ways = 157 :=
by
  intros balls boxes hb hb'
  rw [hb, hb']
  sorry

end ball_distribution_result_l179_179460


namespace sqrt_sum_inequality_l179_179785

theorem sqrt_sum_inequality (a b c : ℝ) (h1 : 1 ≤ a) (h2 : 1 ≤ b) (h3 : 1 ≤ c)
  (h4 : a + b + c = 9) : 
  sqrt (a * b + b * c + c * a) ≤ sqrt a + sqrt b + sqrt c :=
  sorry

end sqrt_sum_inequality_l179_179785


namespace exponent_of_5_in_30_factorial_l179_179926

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179926


namespace sequence_evaluation_l179_179348

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n ≤ 10 then 3 * n
  else 2 * (List.prod (List.map sequence (List.range (n - 1)))) - n

theorem sequence_evaluation :
  List.prod (List.map sequence (List.range 2011)) - (List.sum (List.map (λ i, (sequence i)^2) (List.range 2011))) = -407396 :=
sorry

end sequence_evaluation_l179_179348


namespace sin_45_eq_1_div_sqrt_2_l179_179315

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179315


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179238

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179238


namespace exponent_of_5_in_30_factorial_l179_179898

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179898


namespace quadratic_inequality_min_value_l179_179399

noncomputable def min_value (a b: ℝ) : ℝ := 2 * a^2 + b^2

theorem quadratic_inequality_min_value
  (a b: ℝ) (hx: ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (x0: ℝ) (hx0: a * x0^2 + 2 * x0 + b = 0) :
  a > b → min_value a b = 2 * Real.sqrt 2 := 
sorry

end quadratic_inequality_min_value_l179_179399


namespace ellipse_coeff_sum_l179_179112

theorem ellipse_coeff_sum :
  ∃ (A B C D E F : ℤ), let gcd_val := Int.gcdA (|A| ∣ |B| ∣ |C| ∣ |D| ∣ |E| ∣ |F|) in
  gcd_val = 1 ∧
  (∀ (x y : ℝ) (t : ℝ),
    x = (3 * (Real.sin t + 2)) / (3 - Real.cos t) ∧
    y = (4 * (Real.cos t - 1)) / (3 - Real.cos t) →
    A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0) ∧
  |A| + |B| + |C| + |D| + |E| + |F| = 265 := sorry

end ellipse_coeff_sum_l179_179112


namespace problem_259_problem_260_l179_179059

theorem problem_259 (x a b : ℝ) (h : x ^ 3 = a * x ^ 2 + b * x) (hx : x ≠ 0) : x ^ 2 = a * x + b :=
by sorry

theorem problem_260 (x a b : ℝ) (h : x ^ 4 = a * x ^ 2 + b) : 
  x ^ 2 = (a + Real.sqrt (a ^ 2 + 4 * b)) / 2 ∨ x ^ 2 = (a - Real.sqrt (a ^ 2 + 4 * b)) / 2 :=
by sorry

end problem_259_problem_260_l179_179059


namespace yeast_population_correct_l179_179062

noncomputable def yeast_population_estimation 
    (count_per_small_square : ℕ)
    (dimension_large_square : ℝ)
    (dilution_factor : ℝ)
    (thickness : ℝ)
    (total_volume : ℝ) 
    : ℝ :=
    (count_per_small_square:ℝ) / ((dimension_large_square * dimension_large_square * thickness) / 400) * dilution_factor * total_volume

theorem yeast_population_correct:
    yeast_population_estimation 5 1 10 0.1 10 = 2 * 10^9 :=
by
    sorry

end yeast_population_correct_l179_179062


namespace proof_pythagorean_triple_5_12_13_l179_179657

theorem proof_pythagorean_triple_5_12_13 : (5^2 + 12^2 = 13^2) :=
by
  calc
    5^2 + 12^2 = 25 + 144 := by rfl
    ... = 169 := by rfl
    ... = 13^2 := by rfl

end proof_pythagorean_triple_5_12_13_l179_179657


namespace log_function_decreasing_on_interval_l179_179616

theorem log_function_decreasing_on_interval (m : ℝ) :
  (-3 < m ∧ m < 0) ↔ ∀ x : ℝ, x ≤ 1 → f'(x) < 0 :=
begin
  sorry
end

-- Definitions needed for the problem
noncomputable def f (x : ℝ) : ℝ := log (m * x + 3)

noncomputable def f' (x : ℝ) : ℝ := (differential (log (m * x + 3))).eval x

end log_function_decreasing_on_interval_l179_179616


namespace sin_45_eq_sqrt2_div_2_l179_179208

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179208


namespace exponent_of_5_in_30_fact_l179_179883

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179883


namespace inequality_proof_l179_179548

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c ≤ 3) : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l179_179548


namespace maximum_value_of_f_l179_179374

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem maximum_value_of_f : ∃ x : ℝ, f x = 13 :=
by 
  sorry

end maximum_value_of_f_l179_179374


namespace xiao_li_payment_l179_179355

-- Define the discount rules
def discount (amount : ℝ) : ℝ :=
  if amount ≤ 100 then amount
  else if amount ≤ 300 then amount * 0.9
  else (amount - 300) * 0.8 + 300 * 0.9

-- Original prices of Xiao Mei's purchases
def original_price_first_purchase : ℝ := 94.5
def original_price_second_purchase : ℝ := 282.8

-- Calculate the combined price considering the rules, proving it is equivalent to either 358.4 or 366.8 yuan
theorem xiao_li_payment :
  let combined_price := discount (original_price_first_purchase + original_price_second_purchase) in
  combined_price = 358.4 ∨ combined_price = 366.8 :=
by
  let combined_original_price := original_price_first_purchase + original_price_second_purchase
  have h1 : discount combined_original_price = (316 + 94.5 - 300) * 0.8 + 300 * 0.9, 
  sorry
  have h2 : discount combined_original_price = (316 + 105 - 300) * 0.8 + 300 * 0.9, 
  sorry
  have h3 : (316 + 94.5 - 300) * 0.8 + 300 * 0.9 = 358.4, 
  sorry
  have h4 : (316 + 105 - 300) * 0.8 + 300 * 0.9 = 366.8, 
  sorry
  exact Or.inl h1 ⬝ h3 ∨ Or.inr h2 ⬝ h4

end xiao_li_payment_l179_179355


namespace sin_45_degree_l179_179161

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179161


namespace probability_of_valid_pair_correct_l179_179490

def is_multiple_of_126 (x : ℕ) : Prop :=
  x % 126 = 0

def valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ is_multiple_of_126 (a * b)

noncomputable def probability_of_valid_pair : ℚ :=
  let elements := [6, 14, 18, 35, 42, 49, 54]
  let num_pairs := (elements.length choose 2).toNat
  let valid_pairs := elements.pairs.filter (λ (p : ℕ × ℕ), valid_pair p.1 p.2)
  (valid_pairs.length : ℚ) / num_pairs

theorem probability_of_valid_pair_correct :
  probability_of_valid_pair = 5 / 21 := 
sorry

end probability_of_valid_pair_correct_l179_179490


namespace cos_double_angle_l179_179783

theorem cos_double_angle (α β : ℝ) (h1 : Real.sin (α - β) = 1 / 3) (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_double_angle_l179_179783


namespace cos_double_angle_value_l179_179470

theorem cos_double_angle_value (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_value_l179_179470


namespace sin_45_eq_sqrt2_div_2_l179_179293

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179293


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179229

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179229


namespace sum_outside_layers_is_neg1_l179_179076

theorem sum_outside_layers_is_neg1 :
  ∃ (numbers : ℕ → ℕ → ℕ → ℤ), (∀ i j, (∑ k in finset.range 10, numbers i j k) = 0) ∧
                            (∀ i k, (∑ j in finset.range 10, numbers i j k) = 0) ∧
                            (∀ j k, (∑ i in finset.range 10, numbers i j k) = 0) ∧
                            numbers 1 1 1 = 1 ∧
                            (∑ i in finset.range 10, ∑ j in finset.range 10, ∑ k in finset.range 10, 
                               if (i = 1 ∨ j = 1 ∨ k = 1) then 0 else numbers i j k) = -1 :=
begin
  sorry
end

end sum_outside_layers_is_neg1_l179_179076


namespace existence_of_magical_polynomials_l179_179693

-- Assume the definition of a magical polynomial
def is_magical_polynomial (P : (Fin n → ℕ) → ℕ) : Prop :=
  (∀ (x : Fin n → ℕ), P x ∈ ℕ) ∧ ∀ n, function.bijective P

-- Definition of Catalan number
def catalan (n : ℕ) : ℕ := (Nat.binomial (2 * n) n) / (n + 1)

noncomputable def polynomials_count (n : ℕ) : ℕ := 
  factorial n * (catalan n - catalan (n - 1))

theorem existence_of_magical_polynomials (n : ℕ) (h : n > 0) :
  ∃ P : (Fin n → ℕ) → ℕ, is_magical_polynomial P ∧ polynomials_count n ≥ P :=
sorry

end existence_of_magical_polynomials_l179_179693


namespace mrs_wilsborough_vip_tickets_l179_179556

theorem mrs_wilsborough_vip_tickets:
  let S := 500 -- Initial savings
  let PVIP := 100 -- Price per VIP ticket
  let preg := 50 -- Price per regular ticket
  let nreg := 3 -- Number of regular tickets
  let R := 150 -- Remaining savings after purchase
  
  -- The total amount spent on tickets is S - R
  S - R = PVIP * 2 + preg * nreg := 
by sorry

end mrs_wilsborough_vip_tickets_l179_179556


namespace sin_45_eq_1_div_sqrt_2_l179_179323

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179323


namespace shaded_squares_percentage_l179_179653

theorem shaded_squares_percentage : 
  let grid_size := 6
  let total_squares := grid_size * grid_size
  let shaded_squares := total_squares / 2
  (shaded_squares / total_squares) * 100 = 50 :=
by
  /- Definitions and conditions -/
  let grid_size := 6
  let total_squares := grid_size * grid_size
  let shaded_squares := total_squares / 2

  /- Required proof statement -/
  have percentage_shaded : (shaded_squares / total_squares) * 100 = 50 := sorry

  /- Return the proof -/
  exact percentage_shaded

end shaded_squares_percentage_l179_179653


namespace largest_even_number_below_300_l179_179704

/-- 
  Proving that the largest even number less than or equal to 300,
  which can be formed using three digits from 0 to 9, is 298.
-/
theorem largest_even_number_below_300 : ∃ n, (nat.even n ∧ n ≤ 300 ∧ 
  100 ≤ n ∧ n % 10 ∈ {0, 2, 4, 6, 8} ∧ 
  ∀ m, (nat.even m → m ≤ 300 → 100 ≤ m → m % 10 ∈ {0, 2, 4, 6, 8} → m ≤ n)) ∧ n = 298 := 
sorry

end largest_even_number_below_300_l179_179704


namespace exponent_of_5_in_30_factorial_l179_179904

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l179_179904


namespace cute_polynomial_zero_l179_179143

open Polynomial

def is_prime (n : ℕ) : Prop := nat.prime n
def is_composite (n : ℕ) : Prop := ¬is_prime n ∧ n > 1

def is_cute_subset (s : set ℕ) : Prop :=
  ∃ p q, s = {p, q} ∧ is_prime p ∧ is_composite q ∨ is_prime q ∧ is_composite p

theorem cute_polynomial_zero (f : Polynomial ℤ) :
  (∀ s : set ℕ, is_cute_subset s → is_cute_subset (s.image (λ p, f.eval p))) →
  f = 0 :=
begin
  sorry
end

end cute_polynomial_zero_l179_179143


namespace slant_asymptote_sum_l179_179742

theorem slant_asymptote_sum (y : ℝ → ℝ) (m b : ℝ) (h : ∀ x : ℝ, y x = (3 * x^2 + 4 * x - 8) / (x^2 - 4 * x + 3)) :
  y = λ x, m * x + b → m + b = 3 :=
by
  sorry

end slant_asymptote_sum_l179_179742


namespace sin_45_eq_sqrt_two_over_two_l179_179253

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l179_179253


namespace plane_second_trace_line_solutions_l179_179401

noncomputable def num_solutions_second_trace_line
  (first_trace_line : Line)
  (angle_with_projection_plane : ℝ)
  (intersection_outside_paper : Prop) : ℕ :=
2

theorem plane_second_trace_line_solutions
  (first_trace_line : Line)
  (angle_with_projection_plane : ℝ)
  (intersection_outside_paper : Prop) :
  num_solutions_second_trace_line first_trace_line angle_with_projection_plane intersection_outside_paper = 2 := by
sorry

end plane_second_trace_line_solutions_l179_179401


namespace fred_walking_speed_l179_179389

/-- 
Fred and Sam are standing 55 miles apart and they start walking in a straight line toward each other
at the same time. Fred walks at a certain speed and Sam walks at a constant speed of 5 miles per hour.
Sam has walked 25 miles when they meet.
-/
theorem fred_walking_speed
  (initial_distance : ℕ) 
  (sam_speed : ℕ)
  (sam_distance : ℕ) 
  (meeting_time : ℕ)
  (fred_distance : ℕ) 
  (fred_speed : ℕ)
  (h_initial_distance : initial_distance = 55)
  (h_sam_speed : sam_speed = 5)
  (h_sam_distance : sam_distance = 25)
  (h_meeting_time : meeting_time = 5)
  (h_fred_distance : fred_distance = 30)
  (h_fred_speed : fred_speed = 6)
  : fred_speed = fred_distance / meeting_time :=
by sorry

end fred_walking_speed_l179_179389


namespace problem1_problem2_problem3_l179_179620

-- Definition for problem condition (1)
def condition1 (arrangements : ℕ) : Prop :=
  arrangements = 4320

-- Theorem to prove for condition (1)
theorem problem1 : ∃ (arrangements : ℕ), condition1 arrangements :=
by {
  existsi 4320,
  simp,
  sorry,  -- Proof would go here
}

-- Definition for problem condition (2)
def condition2 (arrangements : ℕ) : Prop :=
  arrangements = 30240

-- Theorem to prove for condition (2)
theorem problem2 : ∃ (arrangements : ℕ), condition2 arrangements :=
by {
  existsi 30240,
  simp,
  sorry,  -- Proof would go here
}

-- Definition for problem condition (3)
def condition3 (arrangements : ℕ) : Prop :=
  arrangements = 6720

-- Theorem to prove for condition (3)
theorem problem3 : ∃ (arrangements : ℕ), condition3 arrangements :=
by {
  existsi 6720,
  simp,
  sorry,  -- Proof would go here
}

end problem1_problem2_problem3_l179_179620


namespace exponent_of_5_in_30_fact_l179_179880

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179880


namespace number_of_sequences_l179_179391

theorem number_of_sequences :
  let letters := Finset.singleton 'P' ∪ Finset.singleton 'Q' ∪ Finset.singleton 'R' ∪ Finset.singleton 'S'
  let numbers := Finset.range 10
  (∑ l1 in letters, ∑ l2 in (letters \ {l1}), ∑ n1 in numbers, ∑ n2 in (numbers \ {n1}),
    if l1 = 'Q' ∧ l2 = 'Q' then 0 else if n1 = 0 ∧ n2 = 0 then 0 else 1) = 5832 := by
  sorry

end number_of_sequences_l179_179391


namespace sin_45_degree_l179_179282

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179282


namespace complement_intersection_l179_179550

open Set

-- Definitions of U, A, and B
def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- Proof statement
theorem complement_intersection : 
  ((U \ A) ∩ (U \ B)) = ({0, 2, 4} : Set ℕ) :=
by sorry

end complement_intersection_l179_179550


namespace inverse_proposition_false_l179_179601

-- Define the original proposition: ∀ a b, if a = b then |a| = |b|
def original_proposition (a b : ℝ) : Prop := a = b → abs a = abs b

-- Define the inverse proposition: ∀ a b, if |a| = |b| then a = b
def inverse_proposition (a b : ℝ) : Prop := abs a = abs b → a = b

-- Prove that the inverse proposition is false
theorem inverse_proposition_false : ¬ (∀ a b : ℝ, inverse_proposition a b) := 
by {
  intro h,
  have h1 : inverse_proposition 1 (-1) := h 1 (-1),
  have h2 : abs 1 = abs (-1),
  { rfl },
  exact h1 h2,
  have h3 : 1 = -1,
  { sorry }
}

end inverse_proposition_false_l179_179601


namespace sin_45_eq_sqrt2_div_2_l179_179331

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179331


namespace sin_45_eq_sqrt2_div_2_l179_179297

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179297


namespace sin_45_degree_l179_179280

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179280


namespace angle_relation_l179_179048

def circles_tangent (Gamma1 Gamma2 : Circle) (A : Point) : Prop :=
  Gamma1.TangentInternallyTo Gamma2 A

def points_on_circle (P : Point) (Gamma : Circle) : Prop :=
  P.OnCircle Gamma

def tangent_points (P : Point) (Gamma1 : Circle) : (Point × Point) :=
  let X := P.Tangent Gamma1
  let Y := P.Tangent Gamma1
  (X, Y)

def tangent_intersections (X Y : Point) (Gamma2 : Circle) : (Point × Point) :=
  let Q := X.IntersectAgain Gamma2
  let R := Y.IntersectAgain Gamma2
  (Q, R)

theorem angle_relation
  (Gamma1 Gamma2 : Circle) (A P X Y Q R : Point)
  (h_tangent : circles_tangent Gamma1 Gamma2 A)
  (h_P_on_Gamma2 : points_on_circle P Gamma2)
  (h_tangents : (X, Y) = tangent_points P Gamma1)
  (h_intersections : (Q, R) = tangent_intersections X Y Gamma2) :
  angle Q A R = 2 * angle X A Y := 
sorry

end angle_relation_l179_179048


namespace crafts_sold_l179_179134

theorem crafts_sold (x : ℕ) 
  (h1 : ∃ (n : ℕ), 12 * n = x * 12)
  (h2 : x * 12 + 7 - 18 = 25):
  x = 3 :=
by
  sorry

end crafts_sold_l179_179134


namespace circle_equation_l179_179370

theorem circle_equation (a : ℝ) (h : a = 1) :
  (∀ (C : ℝ × ℝ), C = (a, a) →
  (∀ (r : ℝ), r = dist C (1, 0) →
  r = 1 → ((x - a) ^ 2 + (y - a) ^ 2 = r ^ 2))) :=
by
  sorry

end circle_equation_l179_179370


namespace sin_45_eq_sqrt2_div_2_l179_179287

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179287


namespace proof_goal_l179_179546

noncomputable def exp_value (k m n : ℕ) : ℤ :=
  (6^k - k^6 + 2^m - 4^m + n^3 - 3^n : ℤ)

theorem proof_goal (k m n : ℕ) (h_k : 18^k ∣ 624938) (h_m : 24^m ∣ 819304) (h_n : n = 2 * k + m) :
  exp_value k m n = 0 := by
  sorry

end proof_goal_l179_179546


namespace root_equation_satisfies_expr_l179_179798

theorem root_equation_satisfies_expr (a : ℝ) (h : 2 * a ^ 2 - 7 * a - 1 = 0) :
  a * (2 * a - 7) + 5 = 6 :=
by
  sorry

end root_equation_satisfies_expr_l179_179798


namespace domain_of_f_l179_179351

def f (x : ℝ) : ℝ := (2 * x - 3) / (x^2 - 5 * x + 6)

theorem domain_of_f :
  {x : ℝ | x ≠ 2 ∧ x ≠ 3} = {x | ∃ (r : ℝ), x = r ∧ (r < 2 ∨ (r > 2 ∧ r < 3) ∨ r > 3)} :=
by
  sorry

end domain_of_f_l179_179351


namespace range_of_f_inequality_proof_l179_179822

noncomputable def f (x : ℝ) := real.sqrt (x^2 - 2*x + 1) + 2 * real.sqrt (4 - 4*x + x^2)

theorem range_of_f : set.range f = set.Ici 1 := sorry

theorem inequality_proof (m : ℝ) (h : ∃ x : ℝ, f x - m < 0) : 3 * m + 2 / (m - 1) > 7 :=
by sorry

end range_of_f_inequality_proof_l179_179822


namespace profit_function_correct_maximize_profit_units_l179_179687

-- Define Total Cost Function
def G (x : ℝ) : ℝ := 2.8 + x

-- Define Revenue Function
def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x
  else 11

-- Define Profit Function
def f (x : ℝ) : ℝ :=
  R(x) - G(x)

-- Statement to prove analytical expression of profit function is as derived
theorem profit_function_correct :
  ∀ x : ℝ,
    (0 ≤ x ∧ x ≤ 5 → f(x) = -0.4 * x^2 + 3.2 * x - 2.8) ∧
    (x > 5 → f(x) = 8.2 - x) :=
by
  sorry

-- Maximize the profit
theorem maximize_profit_units :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 5) ∧
  (∀ y : ℝ, (0 ≤ y ∧ y ≤ 5) → f(y) ≤ f(x)) ∧ x = 4 :=
by
  sorry

end profit_function_correct_maximize_profit_units_l179_179687


namespace geometric_sequence_of_an_min_value_of_sum_bn_increasing_sequence_of_cn_l179_179431

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := log x / log k
noncomputable def a_n (n : ℕ) (k : ℝ) : ℝ := k^(2 * (n : ℝ) + 2)
noncomputable def b_n (n : ℕ) (k : ℝ) : ℝ := a_n n k + f (a_n n k) k
noncomputable def c_n (n : ℕ) (k : ℝ) : ℝ := a_n n k * log (a_n n k)

theorem geometric_sequence_of_an 
  (k : ℝ) (hk1 : k > 0) (hk2 : k ≠ 1) :
  ∀ n : ℕ, a_n (n + 1) k = a_n n k * (k ^ 2) :=
sorry

theorem min_value_of_sum_bn 
  (k : ℝ) : 
  ∀ n : ℕ, k = 1 / (Real.sqrt 2) →
  S_n n k = (n^2 + 3 * n + 1 / 2 - 1 / (2^(n + 1))) := 
sorry

theorem increasing_sequence_of_cn
  (k : ℝ) (hk1 : 0 < k) (hk2 : k < Real.sqrt 6 / 3 ∨ 1 < k) :
  ∀ n : ℕ, c_n (n + 1) k > c_n n k :=
sorry

end geometric_sequence_of_an_min_value_of_sum_bn_increasing_sequence_of_cn_l179_179431


namespace parallelogram_perpendicular_distances_l179_179392

-- Necessary noncomputable treatment
noncomputable theory

-- Definitions for Points and Line in the plane
variables {Point Line : Type}

-- Function to get perpendicular distances from a point to a line
variable (perpendicular_distance : Point → Line → ℝ)

-- Variables for points A, B, C, D and line e
variables (A B C D : Point) (e : Line)

-- Assumption that A A1 is the shortest and C C1 is the longest distance
variable (shortest longest : ℝ)
hypothesis (h1 : shortest = perpendicular_distance A e)
hypothesis (h2 : longest = perpendicular_distance C e)
  
-- Define the sums of distances
def sum_long_short : ℝ := shortest + longest
def sum_other_two : ℝ := perpendicular_distance B e + perpendicular_distance D e

-- Main theorem to prove
theorem parallelogram_perpendicular_distances :
  sum_long_short perpendicular_distance A B C D e = sum_other_two perpendicular_distance A B C D e :=
by {
  sorry
}

end parallelogram_perpendicular_distances_l179_179392


namespace ball_distribution_result_l179_179458

theorem ball_distribution_result :
  ∀ (balls boxes : ℕ), balls = 6 → boxes = 4 →
  let ways : ℕ :=
    1 +  -- (6,0,0,0)
    6 +  -- (5,1,0,0)
    15 + -- (4,2,0,0)
    15 + -- (4,1,1,0)
    10 + -- (3,3,0,0)
    60 + -- (3,2,1,0)
    20 + -- (3,1,1,1)
    15 + -- (2,2,2,0)
    15   -- (2,2,1,1)
  in ways = 157 :=
by
  intros balls boxes hb hb'
  rw [hb, hb']
  sorry

end ball_distribution_result_l179_179458


namespace chess_tournament_ordering_chess_tournament_ordering_l179_179631

theorem chess_tournament_ordering :
  ∃ P : Fin 20 → Fin 20, (∀ i : Fin 19, P i beats P (i+1)) :=
sorry

-- Definitions (as assumptions based on the problem statement) needed for the theorem
variables {Player : Type} -- Define the type of players
variable (beats : Player → Player → Prop) -- Define the "beats" relation

variables (players : Fin 20 → Player) -- A labeling of the 20 players
variable (draws : set (Player×Player)) -- All draws between players

-- Tournament Conditions based on problem description
axiom each_player_plays_once :
  ∀ (p1 p2 : Player), (p1 ≠ p2 → p1 beats p2 ∨ p2 beats p1 ∨ (p1, p2) ∈ draws)

axiom each_draw_condition :
  ∀ (p1 p2 : Player), (p1, p2) ∈ draws → ∀ p3 : Player, p3 ≠ p1 → p3 ≠ p2 
    → p3 beats p1 ∨ p3 beats p2

axiom at_least_two_draws :
  ∃ (p1 p2 p3 p4 : Player), (p1, p2) ∈ draws ∧ (p3, p4) ∈ draws ∧ (p1, p2) ≠ (p3, p4)

-- Formalizing the actual proof problem using the assumptions
theorem chess_tournament_ordering :
  ∃ P : Fin 20 → Fin 20, (∀ i : Fin 19, P i beats P (i+1)) :=
sorry

end chess_tournament_ordering_chess_tournament_ordering_l179_179631


namespace zuminglish_words_mod_1000_l179_179861

def a : ℕ → ℕ 
| 1 := 0
| (n+1) := 2 * (a n + c n)

def b : ℕ → ℕ 
| 1 := 0
| (n+1) := a n

def c : ℕ → ℕ 
| 1 := 0
| (n+1) := 2 * b n

def N (n : ℕ) : ℕ := a n + b n + c n

theorem zuminglish_words_mod_1000 :
  (N 12) % 1000 = 202 :=
sorry -- Proof goes here

end zuminglish_words_mod_1000_l179_179861


namespace smallest_four_digit_palindrome_divisible_by_8_l179_179024

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem smallest_four_digit_palindrome_divisible_by_8 : ∃ (n : ℕ), is_palindrome n ∧ is_four_digit n ∧ divisible_by_8 n ∧ n = 4004 := by
  sorry

end smallest_four_digit_palindrome_divisible_by_8_l179_179024


namespace exponent_of_5_in_30_factorial_l179_179922

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179922


namespace sin_45_degree_l179_179168

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179168


namespace sum_of_numbers_with_at_most_three_prime_factors_l179_179765

-- Define the set of primes
def primes : Set ℕ := {2, 3, 5, 7}

-- Define a function to check the number of prime factors within the given set
def count_prime_factors (n : ℕ) : ℕ :=
  primes.toList.foldr (λ p acc => acc + (nat.factorization n).count p) 0

-- Define a function to sum all positive integers with at most three prime factors from the set
def sum_at_most_three_prime_factors : ℕ :=
  (Finset.range 350).filter (λ n => count_prime_factors n ≤ 3 ∧ n ≠ 0).sum id

-- The theorem to prove the correctness of the solution
theorem sum_of_numbers_with_at_most_three_prime_factors :
  sum_at_most_three_prime_factors = 1932 :=
begin
  -- The proof goes here
  sorry
end

end sum_of_numbers_with_at_most_three_prime_factors_l179_179765


namespace function_graph_intersection_l179_179448

theorem function_graph_intersection (f : ℝ → ℝ) :
  (∃ y : ℝ, f 1 = y) → (∃! y : ℝ, f 1 = y) :=
by
  sorry

end function_graph_intersection_l179_179448


namespace count_divisible_by_five_l179_179537

def f (x : ℤ) : ℤ := x ^ 2 + 5 * x + 6

def S : Set ℤ := { x | 0 ≤ x ∧ x ≤ 30 }

theorem count_divisible_by_five : 
  (S.filter (λ s => f s % 5 = 0)).card = 12 :=
by sorry

end count_divisible_by_five_l179_179537


namespace evaluate_expression_l179_179727

theorem evaluate_expression :
  (0.25) ^ (-0.5) + 8 ^ (2 / 3) - 2 * (log 25 / log 5) = 2 :=
by
  sorry

end evaluate_expression_l179_179727


namespace find_geometric_sequence_preserving_functions_l179_179734

-- Define the geometric sequence preserving function property
def is_geometric_sequence {α : Type*} [OrderedRing α] (a : ℕ → α) : Prop :=
∀ n : ℕ, a n * a (n + 2) = (a (n + 1)) ^ 2

def geometricSequencePreserving (f : ℝ → ℝ) : Prop :=
∀ (a : ℕ → ℝ), is_geometric_sequence a → is_geometric_sequence (λ n, f (a n))

-- Define the functions in question
def f1 (x : ℝ) : ℝ := x^2
def f2 (x : ℝ) : ℝ := x^2 + 1
def f3 (x : ℝ) : ℝ := real.sqrt (abs x)
def f4 (x : ℝ) : ℝ := real.log (abs x)

-- Theorem stating which functions are geometric sequence preserving
theorem find_geometric_sequence_preserving_functions :
  (geometricSequencePreserving f1) ∧
  ¬(geometricSequencePreserving f2) ∧
  (geometricSequencePreserving f3) ∧
  ¬(geometricSequencePreserving f4) :=
sorry

end find_geometric_sequence_preserving_functions_l179_179734


namespace problem_1_problem_2_l179_179500

open Real

variables {A B C a b c : ℝ}

theorem problem_1 (h1 : ∀ {A B C : ℝ},  ∃ {A B C : ℝ}, 
 (A + B + C = π) ∧ 
 0 < A ∧ A < π / 2 ∧
 0 < B ∧ B < π / 2 ∧ 
 0 < C ∧ C < π / 2 )
 (h2 : ∀ {A : ℝ}, ∃ {B : ℝ}, 2 * sin(A + C) / (cos 2B) = 1) :
  B = π / 3 :=
by
  sorry

theorem problem_2 (h3: sin A * sin C= sin^2 B):
  ∀ {a b c : ℝ}, 
  (a * c = b^2) ∧
    (b^2 = a^2 + c^2 - 2 * a * c * cos ((π / 3))): (a - c = 0)= 
by
  sorry

end problem_1_problem_2_l179_179500


namespace largest_prime_divisor_of_13_plus_14_factorial_is_13_l179_179739

theorem largest_prime_divisor_of_13_plus_14_factorial_is_13 : 
  (∀ p : ℕ, prime p ∧ p ∣ (13! + 14!) → p ≤ 13) ∧ (∃ p : ℕ, prime p ∧ p ∣ (13! + 14!) ∧ p = 13) :=
by 
  sorry

end largest_prime_divisor_of_13_plus_14_factorial_is_13_l179_179739


namespace specimen_exchange_l179_179063

theorem specimen_exchange (x : ℕ) (h : x * (x - 1) = 110) : x * (x - 1) = 110 := by
  exact h

end specimen_exchange_l179_179063


namespace problem_l179_179843

variable (a b : ℝ)

theorem problem (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 :=
by
  sorry

end problem_l179_179843


namespace probability_exactly_one_l179_179866

noncomputable def total_people : ℕ := 4000
noncomputable def frac_play_instrument : ℚ := 7 / 10
noncomputable def frac_play_two_or_more : ℚ := 5 / 8

noncomputable def num_play_instrument : ℕ := frac_play_instrument * total_people
noncomputable def num_play_two_or_more : ℕ := frac_play_two_or_more * num_play_instrument
noncomputable def num_play_exactly_one : ℕ := num_play_instrument - num_play_two_or_more

noncomputable def probability_play_exactly_one : ℚ := num_play_exactly_one / total_people

theorem probability_exactly_one (h : probability_play_exactly_one = 0.2625) :
  probability_play_exactly_one = 2625/10000 := 
sorry

end probability_exactly_one_l179_179866


namespace sin_45_degree_l179_179182

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179182


namespace rectangular_prism_diagonal_length_l179_179696

theorem rectangular_prism_diagonal_length :
  ∀ (a b c : ℝ), a = 12 → b = 15 → c = 8 → (sqrt (a^2 + b^2 + c^2) = sqrt 433) :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  sorry

end rectangular_prism_diagonal_length_l179_179696


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179228

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179228


namespace minimum_f_zero_iff_t_is_2sqrt2_l179_179428

noncomputable def f (x t : ℝ) : ℝ := 4 * x^4 - 6 * t * x^3 + (2 * t + 6) * x^2 - 3 * t * x + 1

theorem minimum_f_zero_iff_t_is_2sqrt2 :
  (∀ x > 0, f x t ≥ 0) ∧ (∃ x > 0, f x t = 0) ↔ t = 2 * Real.sqrt 2 := 
sorry

end minimum_f_zero_iff_t_is_2sqrt2_l179_179428


namespace prism_faces_vertices_l179_179694

theorem prism_faces_vertices {L E F V : ℕ} (hE : E = 21) (hEdges : E = 3 * L) 
    (hF : F = L + 2) (hV : V = L) : F = 9 ∧ V = 7 :=
by
  sorry

end prism_faces_vertices_l179_179694


namespace sin_45_eq_one_div_sqrt_two_l179_179227

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179227


namespace exists_indices_for_sequences_l179_179050

theorem exists_indices_for_sequences 
  (a b c : ℕ → ℕ) :
  ∃ (p q : ℕ), p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
sorry

end exists_indices_for_sequences_l179_179050


namespace fraction_identity_l179_179838

theorem fraction_identity (a b : ℝ) (h : a / b = 5 / 2) : (a + 2 * b) / (a - b) = 3 :=
by sorry

end fraction_identity_l179_179838


namespace exponent_of_5_in_30_factorial_l179_179957

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l179_179957


namespace csc_sum_geq_12_csc_sum_eq_12_iff_l179_179806

variable {α β γ : ℝ} (h_sum : α + β + γ = Real.pi)
variable (h_triangle : 0 < α ∧ α < Real.pi ∧ 0 < β ∧ β < Real.pi ∧ 0 < γ ∧ γ < Real.pi)

theorem csc_sum_geq_12 (α β γ : ℝ) (h_sum : α + β + γ = Real.pi)
(h_triangle : 0 < α ∧ α < Real.pi ∧ 0 < β ∧ β < Real.pi ∧ 0 < γ ∧ γ < Real.pi) :
  (Real.csc (α / 2))^2 + (Real.csc (β / 2))^2 + (Real.csc (γ / 2))^2 ≥ 12 :=
sorry

theorem csc_sum_eq_12_iff (α β γ : ℝ) (h_sum : α + β + γ = Real.pi)
(h_triangle : 0 < α ∧ α < Real.pi ∧ 0 < β ∧ β < Real.pi ∧ 0 < γ ∧ γ < Real.pi) :
  ((Real.csc (α / 2))^2 + (Real.csc (β / 2))^2 + (Real.csc (γ / 2))^2 = 12) ↔
  (α = Real.pi / 3 ∧ β = Real.pi / 3 ∧ γ = Real.pi / 3) :=
sorry

end csc_sum_geq_12_csc_sum_eq_12_iff_l179_179806


namespace exponent_of_5_in_30_factorial_l179_179992

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179992


namespace avg_y_is_58_5_l179_179400

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := 1.5 * x + 45

-- Given values of x
def x_values : List ℝ := [1, 7, 5, 13, 19]

-- Calculate the average of the given x values
def avg_x : ℝ := (x_values.sum / x_values.length)

-- Target average y value
def avg_y : ℝ := linear_regression avg_x

-- The proof statement
theorem avg_y_is_58_5 : avg_y = 58.5 :=
by
  sorry

end avg_y_is_58_5_l179_179400


namespace exponent_of_5_in_30_factorial_l179_179999

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179999


namespace cos_thirteen_pi_over_three_l179_179755

theorem cos_thirteen_pi_over_three : Real.cos (13 * Real.pi / 3) = 1 / 2 := 
by
  sorry

end cos_thirteen_pi_over_three_l179_179755


namespace tan_theta_eq_sqrt3_div_3_l179_179446

theorem tan_theta_eq_sqrt3_div_3
  (θ : ℝ)
  (h : (Real.cos θ * Real.sqrt 3 + Real.sin θ) = 2) :
  Real.tan θ = Real.sqrt 3 / 3 := by
  sorry

end tan_theta_eq_sqrt3_div_3_l179_179446


namespace sin_45_degree_l179_179279

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179279


namespace sample_and_size_correct_l179_179681

structure SchoolSurvey :=
  (students_selected : ℕ)
  (classes_selected : ℕ)

def survey_sample (survey : SchoolSurvey) : String :=
  "the physical condition of " ++ toString survey.students_selected ++ " students"

def survey_sample_size (survey : SchoolSurvey) : ℕ :=
  survey.students_selected

theorem sample_and_size_correct (survey : SchoolSurvey)
  (h_selected : survey.students_selected = 190)
  (h_classes : survey.classes_selected = 19) :
  survey_sample survey = "the physical condition of 190 students" ∧ 
  survey_sample_size survey = 190 :=
by
  sorry

end sample_and_size_correct_l179_179681


namespace area_difference_l179_179876

def area_triangle (base height : ℕ) : ℝ :=
  0.5 * base * height

variables (AB AE BC : ℕ)
variables (ADE BDC ABD : ℝ)

axiom right_angles : ∀ A B E C : Type, ∠(EAB) = 90 ∧ ∠(ABC) = 90

theorem area_difference :
  AB = 5 →
  BC = 7 →
  AE = 10 →
  area_triangle AB AE = ADE + ABD →
  area_triangle AB BC = BDC + ABD →
  ADE - BDC = 7.5 :=
by
  intros hAB hBC hAE hA1 hA2
  sorry

end area_difference_l179_179876


namespace cannot_fill_box_exactly_l179_179066

def box_length : ℝ := 70
def box_width : ℝ := 40
def box_height : ℝ := 25
def cube_side : ℝ := 4.5

theorem cannot_fill_box_exactly : 
  ¬ (∃ n : ℕ, n * cube_side^3 = box_length * box_width * box_height ∧
               (∃ x y z : ℕ, x * cube_side = box_length ∧ 
                             y * cube_side = box_width ∧ 
                             z * cube_side = box_height)) :=
by sorry

end cannot_fill_box_exactly_l179_179066


namespace problem_A_problem_C_problem_D_l179_179779

noncomputable def f (x : ℝ) : ℝ := (2 - x) ^ 8

theorem problem_A :
  let a := (λ n : ℕ, (nat.choose 8 n) * (-1) ^ n * 2 ^ (8 - n)) in
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 1) :=
by
  sorry

theorem problem_C :
  (f (-1) % 5 = 1) :=
by
  sorry

theorem problem_D :
  let a := (λ n : ℕ, (nat.choose 8 n) * (-1) ^ n * 2 ^ (8 - n)) in
  ((1 * a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 + 5 * a 5 + 6 * a 6 + 7 * a 7 + 8 * a 8) = -8) :=
by
  sorry


end problem_A_problem_C_problem_D_l179_179779


namespace part1_part2_part3_part4_l179_179107

section
variables {x : ℝ} {f : ℝ → ℝ}

-- 1. If y = cos x, then y' = -sin x
theorem part1 (h : f = cos x) : deriv f x = -sin x :=
sorry

-- 2. If y = -1 / sqrt x, then y' = 1 / (2x sqrt x)
theorem part2 (h : f = -1 / sqrt x) : deriv f x = 1 / (2 * x * sqrt x) :=
sorry

-- 3. If f(x) = 1 / x^2, then f'(3) = -2 / 27
theorem part3 (h : f = λ x, 1 / x^2) : deriv f 3 = -2 / 27 :=
sorry

-- 4. If y = 3, then y' = 0
theorem part4 (h : f = λ x, 3) : deriv f x = 0 :=
sorry
end

end part1_part2_part3_part4_l179_179107


namespace sin_45_eq_sqrt2_div_2_l179_179213

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179213


namespace union_sets_l179_179805

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2^a}

theorem union_sets : A ∪ B = {0, 1, 2, 4} := by
  sorry

end union_sets_l179_179805


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179233

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179233


namespace nth_term_arithmetic_sequence_l179_179768

variable (n r : ℕ)

def S (n : ℕ) : ℕ := 4 * n + 5 * n^2

theorem nth_term_arithmetic_sequence :
  (S r) - (S (r-1)) = 10 * r - 1 :=
by
  sorry

end nth_term_arithmetic_sequence_l179_179768


namespace prime_exponent_of_5_in_30_factorial_l179_179916

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179916


namespace sin_45_eq_sqrt2_div_2_l179_179336

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179336


namespace arrange_in_order_l179_179117

noncomputable def Psi : ℤ := 1 / 2 * (1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + List.sum (List.of_fn (λ n => if n % 4 < 2 then 2 * n + 5 else -2 * n - 5) 503))
noncomputable def Omega : ℤ := List.sum (List.of_fn (λ n => if n % 2 == 0 then n + 1 else -n - 1) 1007)
noncomputable def Theta : ℤ := List.sum (List.range (2015 / 2) |>.map (λ n => if n % 2 == 0 then 2 * n + 1 else -2 * n - 1))

theorem arrange_in_order : Theta = -1008 ∧ Omega = -1007 ∧ Psi = -1006 → Theta ≤ Omega ∧ Omega ≤ Psi := sorry

end arrange_in_order_l179_179117


namespace expected_greetings_l179_179356

theorem expected_greetings :
  let p1 := 1       -- Probability 1
  let p2 := 0.8     -- Probability 0.8
  let p3 := 0.5     -- Probability 0.5
  let p4 := 0       -- Probability 0
  let n1 := 8       -- Number of colleagues with probability 1
  let n2 := 15      -- Number of colleagues with probability 0.8
  let n3 := 14      -- Number of colleagues with probability 0.5
  let n4 := 3       -- Number of colleagues with probability 0
  p1 * n1 + p2 * n2 + p3 * n3 + p4 * n4 = 27 :=
by
  sorry

end expected_greetings_l179_179356


namespace perpendicular_lines_a_value_parallel_lines_distance_l179_179407

theorem perpendicular_lines_a_value (a b : ℝ) (h_b : b = 0) 
    (h_perp : ∀ (l₁ l₂ : Prop), l₁ = (λ (x : ℝ), a * x + 1 = 0) 
                              → l₂ = (λ (x : ℝ), (a - 2) * x + a = 0) 
                              → l₁ ⊥ l₂) : 
    a = 2 :=
sorry

theorem parallel_lines_distance (a b : ℝ) (h_b : b = 3) 
    (h_par : ∀ (l₁ l₂ : Prop), l₁ = (λ (x y : ℝ), a * x + 3 * y + 1 = 0) 
                              → l₂ = (λ (x y : ℝ), (a - 2) * x + y + a = 0) 
                              → l₁ ∥ l₂)
    (h_a : a = 3) : 
  distance_parallel_lines (λ (x y : ℝ), 3 * x + 3 * y + 1 = 0) 
                          (λ (x y : ℝ), x + y + 3 = 0) 
                          = (4 * real.sqrt 2 / 3) :=
sorry

noncomputable def distance_parallel_lines 
  (l₁ l₂ : ℝ → ℝ → Prop) : ℝ :=
  let C₁ := 1 
  let C₂ := 9 
  let A := 3 
  let B := 3 
  abs (C₁ - C₂) / real.sqrt (A ^ 2 + B ^ 2)

end perpendicular_lines_a_value_parallel_lines_distance_l179_179407


namespace log_sum_result_l179_179141

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

theorem log_sum_result :
  log_b 0.5 0.125 + log_b 2 (log_b 3 (log_b 4 64)) = 3 :=
by
  -- Definitions using conditions
  let a1 : ℝ := 2⁻¹
  let a2 : ℝ := 2⁻³
  let a3 : ℝ := 4⁻¹
  let a4 : ℝ := 4³
  
  -- Given the math problem, stating the evaluated expressions
  have h1 : log_b a1 a2 = 3 := sorry
  have h2 : log_b a3 a4 = 3 := sorry
  have h3 : log_b 3 3 = 1 := sorry
  have h4 : log_b 2 1 = 0 := sorry
  
  -- Assembling the solution
  calc
    log_b 0.5 0.125 + log_b 2 (log_b 3 (log_b 4 64)) = log_b a1 a2 + log_b 2 (log_b 3 3) : by sorry
    ... = 3 + log_b 2 1 : by sorry
    ... = 3 + 0 : by sorry
    ... = 3 : by sorry

end log_sum_result_l179_179141


namespace solve_star_eq_l179_179061

noncomputable def star (a b : ℤ) : ℤ := if a = b then 2 else sorry

axiom star_assoc : ∀ (a b c : ℤ), star a (star b c) = (star a b) - c
axiom star_self_eq_two : ∀ (a : ℤ), star a a = 2

theorem solve_star_eq : ∀ (x : ℤ), star 100 (star 5 x) = 20 → x = 20 :=
by
  intro x hx
  sorry

end solve_star_eq_l179_179061


namespace range_of_a_l179_179484

noncomputable def f (a x : ℝ) := a * x - x^2 - Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 2*x₁*x₁ - a*x₁ + 1 = 0 ∧ 
  2*x₂*x₂ - a*x₂ + 1 = 0 ∧ f a x₁ + f a x₂ ≥ 4 + Real.log 2) ↔ 
  a ∈ Set.Ici (2 * Real.sqrt 3) := 
sorry

end range_of_a_l179_179484


namespace general_formula_is_correct_value_of_k_l179_179416

variables (a_n : ℕ → ℝ) (S_k : ℕ → ℝ)

-- Conditions
def decreasing_arithmetic_sequence (d : ℝ) : Prop := 
  ∀ n : ℕ, a_n (n+1) = a_n n + d

def initial_conditions (a1 a3 : ℝ) : Prop :=
  a1 + a3 = -2 ∧ a1 * a3 = -3

-- Definitions
def general_formula : ℕ → ℝ := λ n, -2 * n + 3

def sum_of_first_k_terms (k : ℕ) : ℝ := 
  k * a_n 1 + (k * (k - 1) / 2) * (-2)

-- Equivalence to be proved
theorem general_formula_is_correct (a1 a3 : ℝ) 
  (h1 : initial_conditions a1 a3) 
  (h2 : decreasing_arithmetic_sequence (-2)) :
  ∀ n : ℕ, a_n n = general_formula n := 
sorry

theorem value_of_k (h3 : S_k 7 = -35) :
  ∃ k : ℕ, S_k k = -35 ∧ k = 7 :=
sorry

end general_formula_is_correct_value_of_k_l179_179416


namespace exponent_of_5_in_30_fact_l179_179886

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179886


namespace arrange_in_order_l179_179116

noncomputable def Psi : ℤ := 1 / 2 * (1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + List.sum (List.of_fn (λ n => if n % 4 < 2 then 2 * n + 5 else -2 * n - 5) 503))
noncomputable def Omega : ℤ := List.sum (List.of_fn (λ n => if n % 2 == 0 then n + 1 else -n - 1) 1007)
noncomputable def Theta : ℤ := List.sum (List.range (2015 / 2) |>.map (λ n => if n % 2 == 0 then 2 * n + 1 else -2 * n - 1))

theorem arrange_in_order : Theta = -1008 ∧ Omega = -1007 ∧ Psi = -1006 → Theta ≤ Omega ∧ Omega ≤ Psi := sorry

end arrange_in_order_l179_179116


namespace coefficient_x10_in_expansion_l179_179738

theorem coefficient_x10_in_expansion :
  (algebra_map ℚ ℤ (binom 9 7) : ℚ) = 36 := by
  sorry

end coefficient_x10_in_expansion_l179_179738


namespace sin_45_degree_l179_179185

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l179_179185


namespace arrange_in_non_increasing_order_l179_179120

noncomputable def Psi : ℤ := (1/2) * ∑ k in finset.range 503, -4
noncomputable def Omega : ℤ := ∑ k in finset.range 1007, -1
noncomputable def Theta : ℤ := ∑ k in finset.range 504, -2

theorem arrange_in_non_increasing_order :
  Theta ≤ Omega ∧ Omega ≤ Psi :=
begin
  -- Proof to be implemented
  sorry,
end

end arrange_in_non_increasing_order_l179_179120


namespace sum_of_first_9_multiples_of_5_l179_179027

theorem sum_of_first_9_multiples_of_5 : (∑ i in finset.range 1 (9 + 1), (5 * i)) = 225 := 
by sorry

end sum_of_first_9_multiples_of_5_l179_179027


namespace officer_location_fuel_consumption_l179_179506

def police_patrol_distances : List ℤ := [+15, -4, +13, -10, -12, +3, -13, -17]

def fuel_consumption_rate : ℚ := 0.4

-- Total displacement from the given distances
def total_displacement : ℤ := police_patrol_distances.sum

-- Absolute value of all distances + returning to starting point
def total_distance_traveled : ℤ :=
  police_patrol_distances.map (λ x => |x|).sum + |total_displacement|

-- Fuel consumption for the patrol, including the return trip
def total_fuel_consumed : ℚ := total_distance_traveled * fuel_consumption_rate

-- Proof that the total displacement is -25 kilometers
theorem officer_location : total_displacement = -25 := 
by
  sorry

-- Proof that the total fuel consumption for the patrol (including the return trip) is 44.8 liters
theorem fuel_consumption : total_fuel_consumed = 44.8 :=
by
  sorry

end officer_location_fuel_consumption_l179_179506


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179236

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179236


namespace curtain_additional_material_l179_179345

theorem curtain_additional_material
  (room_height_feet : ℕ)
  (curtain_length_inches : ℕ)
  (height_conversion_factor : ℕ)
  (desired_length : ℕ)
  (h_room_height_conversion : room_height_feet * height_conversion_factor = 96)
  (h_desired_length : desired_length = 101) :
  curtain_length_inches = desired_length - (room_height_feet * height_conversion_factor) :=
by
  sorry

end curtain_additional_material_l179_179345


namespace initial_children_in_camp_l179_179513

theorem initial_children_in_camp 
  (C : ℕ)
  (h1 : 0.80 * C = 4 / 5 * C)
  (h2 : 0.20 * C = 1 / 5 * C)
  (h3 : 0.80 * C + 50 = 4 / 5 * C + 50)
  (h4 : 0.20 * C = 1 / 5 * C)
  (h5 : 0.20 * C = 0.10 * (C + 50)) :
  C = 50 := 
sorry

end initial_children_in_camp_l179_179513


namespace max_value_sin_cos_expression_l179_179740

theorem max_value_sin_cos_expression (θ₁ θ₂ θ₃ θ₄ : ℝ) : 
  ∃ θ₁ θ₂ θ₃ θ₄, 
    (cos θ₁ * sin θ₂ + 
     cos θ₂ * sin θ₃ + 
     cos θ₃ * sin θ₄ + 
     cos θ₄ * sin θ₁) ≤ 2 :=
by {
  sorry
}

end max_value_sin_cos_expression_l179_179740


namespace graduation_problem_l179_179629

def valid_xs : List ℕ :=
  [10, 12, 15, 18, 20, 24, 30]

noncomputable def sum_valid_xs (l : List ℕ) : ℕ :=
  l.foldr (λ x sum => x + sum) 0

theorem graduation_problem :
  sum_valid_xs valid_xs = 129 :=
by
  sorry

end graduation_problem_l179_179629


namespace true_root_30_40_l179_179342

noncomputable def u (x : ℝ) : ℝ := Real.sqrt (x + 15)
noncomputable def original_eqn (x : ℝ) : Prop := u x - 3 / (u x) = 4

theorem true_root_30_40 : ∃ (x : ℝ), 30 < x ∧ x < 40 ∧ original_eqn x :=
by
  sorry

end true_root_30_40_l179_179342


namespace travel_cost_Piravena_l179_179498

def right_angled_triangle (D E F: Type) :=
  ∃ (DE: D → E → ℝ) (EF: E → F → ℝ) (FD: F → D → ℝ),
  DE.val = 4000 ∧ EF.val = 4500 ∧ FD.val = 5000 ∧
  DE.val^2 + FD.val^2 = EF.val^2

def bus_fare (distance: ℝ): ℝ :=
  distance * 0.20

def airplane_fare (distance: ℝ) (booking_fee: ℝ): ℝ :=
  distance * 0.12 + booking_fee

theorem travel_cost_Piravena (D E F: Type)
  (h_triangle: right_angled_triangle D E F)
  (DE := 4000 : ℝ)
  (EF := 4500 : ℝ)
  (bus_rate := 0.20 : ℝ)
  (flight_rate := 0.12 : ℝ)
  (bus_cost := bus_fare DE)
  (flight_cost := airplane_fare EF 150):
  bus_cost + flight_cost = 1490 :=
sorry

end travel_cost_Piravena_l179_179498


namespace adam_beth_distance_l179_179103

theorem adam_beth_distance (t : ℝ) (h1 : 0 ≤ t) :
  let adam_distance := 10 * t,
      beth_distance := 12 * t in
  sqrt (adam_distance^2 + beth_distance^2) = 130 → t ≈ 8.32 :=
by
  let adam_distance := 10 * t
  let beth_distance := 12 * t
  assume h : sqrt (adam_distance^2 + beth_distance^2) = 130
  sorry

end adam_beth_distance_l179_179103


namespace max_f_value_l179_179378

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 13 :=
sorry

end max_f_value_l179_179378


namespace set_intersection_A_B_l179_179826

def A := {x : ℝ | 2 * x - x^2 > 0}
def B := {x : ℝ | x > 1}
def I := {x : ℝ | 1 < x ∧ x < 2}

theorem set_intersection_A_B :
  A ∩ B = I :=
sorry

end set_intersection_A_B_l179_179826


namespace value_of_a_l179_179821

theorem value_of_a (a : ℝ) (h_neg : a < 0) (h_f : ∀ (x : ℝ), (0 < x ∧ x ≤ 1) → 
  (x + 4 * a / x - a < 0)) : a ≤ -1 / 3 := 
sorry

end value_of_a_l179_179821


namespace median_angle_division_l179_179729

theorem median_angle_division (a b : ℝ) (h : a < b) :
  ∃ (C : ℝ), ∃ (α : ℝ), 0 < α ∧ α < 60 ∧ (cos α = a / (2 * b)) := 
sorry

end median_angle_division_l179_179729


namespace exponent_of_5_in_30_fact_l179_179882

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179882


namespace sin_45_degree_l179_179163

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179163


namespace unique_triangle_determined_by_ASA_l179_179030

def angle_A : ℝ := 50
def angle_B : ℝ := 30
def side_AB : ℝ := 10

theorem unique_triangle_determined_by_ASA :
  ∃! (ABC : Triangle), 
  ABC.angle_A = angle_A ∧ ABC.angle_B = angle_B ∧ ABC.side_AB = side_AB :=
by sorry

end unique_triangle_determined_by_ASA_l179_179030


namespace exponent_of_5_in_30_factorial_l179_179923

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179923


namespace irrational_count_in_set_l179_179607

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_count_in_set :
  let s := {22 / 7: ℝ, Real.sqrt 5, Real.sqrt 2 + 1, 2 * Real.pi, Real.sqrt 3 ^ 0, |(-3): ℝ|, 0.313113111..}
  finset.count (λ x, is_irrational x) s = 4 :=
by
  let s := {22 / 7: ℝ, Real.sqrt 5, Real.sqrt 2 + 1, 2 * Real.pi, Real.sqrt 3 ^ 0, |(-3): ℝ|, 0.313113111..}
  sorry

end irrational_count_in_set_l179_179607


namespace sin_45_deg_eq_l179_179304

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179304


namespace sin_order_l179_179608

-- Definitions of sine values at specific points
def sin1 := sin 1
def sin2 := sin 2
def sin3 := sin 3
def sin4 := sin 4

-- The theorem stating the order of these sine values
theorem sin_order : sin 4 < sin 3 ∧ sin 3 < sin 1 ∧ sin 1 < sin 2 :=
by 
  sorry

end sin_order_l179_179608


namespace find_base_of_log_l179_179423

theorem find_base_of_log (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : ∃ x : ℝ, (f : ℝ → ℝ := λ x, 2 + real.log x / real.log a) x = 3 ∧ (inverse_f : ℝ → ℝ := λ y, inverse (λ x, 2 + real.log x / real.log a) y) 3 = 4) :
  a = 2 := 
by {
  sorry
}

end find_base_of_log_l179_179423


namespace intersection_points_of_graph_and_line_l179_179449

theorem intersection_points_of_graph_and_line (f : ℝ → ℝ) :
  (∀ x : ℝ, f x ≠ my_special_value) → (∀ x₁ x₂ : ℝ, f x₁ = f x₂ → x₁ = x₂) →
  ∃! x : ℝ, x = 1 ∧ ∃ y : ℝ, y = f x :=
by
  sorry

end intersection_points_of_graph_and_line_l179_179449


namespace prime_square_implies_equal_l179_179088

theorem prime_square_implies_equal (p : ℕ) (hp : Nat.Prime p) (hp_gt_2 : p > 2)
  (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ (p-1)/2) (hy : 1 ≤ y ∧ y ≤ (p-1)/2)
  (h_square: ∃ k : ℕ, x * (p - x) * y * (p - y) = k ^ 2) : x = y :=
sorry

end prime_square_implies_equal_l179_179088


namespace solution_set_of_inequality_l179_179384

theorem solution_set_of_inequality :
  {x : ℝ | 1 / x < 1 / 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solution_set_of_inequality_l179_179384


namespace prime_exponent_of_5_in_30_factorial_l179_179917

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179917


namespace arrange_in_non_decreasing_order_l179_179128

def Psi : ℤ := (1 / 2 : ℚ) * (Finset.sum (Finset.range 1006) (λ i, 1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + (4 * i)))

def Omega : ℤ := Finset.sum (Finset.range 1007) (λ i, 1 - 2 + 3 - 4 + (2 * i))

def Theta : ℤ := Finset.sum (Finset.range 504) (λ i, 1 - 3 + 5 - 7 + (4 * i))

theorem arrange_in_non_decreasing_order : [Theta, Omega, Psi] = [(-1008 : ℤ), (-1007 : ℤ), (-1006 : ℤ)] :=
by
  have : Psi = -1006 := sorry
  have : Omega = -1007 := sorry
  have : Theta = -1008 := sorry
  rw [this, this, this]
  trivial

end arrange_in_non_decreasing_order_l179_179128


namespace sin_45_deg_l179_179194

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179194


namespace sin_45_eq_one_div_sqrt_two_l179_179225

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179225


namespace sin_45_eq_one_div_sqrt_two_l179_179222

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179222


namespace evaluation_of_expression_l179_179466

theorem evaluation_of_expression
  (a b x y m : ℤ)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = -1) :
  2023 * (a + b) + 3 * (|m|) - 2 * (x * y) = 1 :=
by
  -- skipping the proof
  sorry

end evaluation_of_expression_l179_179466


namespace max_f_value_l179_179380

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 13 :=
sorry

end max_f_value_l179_179380


namespace hazel_salmon_caught_l179_179831

-- Define the conditions
def father_salmon_caught : Nat := 27
def total_salmon_caught : Nat := 51

-- Define the main statement to be proved
theorem hazel_salmon_caught : total_salmon_caught - father_salmon_caught = 24 := by
  sorry

end hazel_salmon_caught_l179_179831


namespace width_of_path_l179_179695

theorem width_of_path
  (a b x : ℝ)
  (h : 2 * (a + b) + 24 = 2 * (a + b + 4 * x)) : 
  x = 3 := 
begin
  sorry
end

end width_of_path_l179_179695


namespace amaya_last_part_time_l179_179705

variables
  (t1 t2 tR1 tR2 t_total t_last: ℕ)
  (H1: t1 = 35)
  (H2: tR1 = 5)
  (H3: t2 = 45)
  (H4: tR2 = 15)
  (H5: t_total = 120)

-- The total time spent on rewinds is H6.
def total_rewind_time := tR1 + tR2

-- The total time spent watching before the last part is H7.
def before_last_part_time := t1 + t2

-- The time Amaya watched the last part of the movie uninterrupted is:
def uninterrupted_time : ℕ :=
  t_total - total_rewind_time - before_last_part_time

theorem amaya_last_part_time :
  uninterrupted_time = 20 :=
by
  rw [uninterrupted_time, total_rewind_time, before_last_part_time, H1, H2, H3, H4, H5]
  simp
  sorry

end amaya_last_part_time_l179_179705


namespace distinguishable_balls_indistinguishable_boxes_l179_179455

theorem distinguishable_balls_indistinguishable_boxes :
  ∃ (f : Finset (Multiset ℕ)), ∀ (n : ℕ) (m : ℕ) 
  (hn : n = 6) (hm : m = 4), Multiset.card f = 257 ∧ 
  (∀ x ∈ f, Multiset.sum x = n ∧ Multiset.card x ≤ m) 
  := sorry

end distinguishable_balls_indistinguishable_boxes_l179_179455


namespace growth_factor_condition_l179_179561

open BigOperators

theorem growth_factor_condition {n : ℕ} (h : ∏ i in Finset.range n, (i + 2) / (i + 1) = 50) : n = 49 := by
  sorry

end growth_factor_condition_l179_179561


namespace college_girls_count_l179_179864

theorem college_girls_count 
  (B G : ℕ)
  (h1 : B / G = 8 / 5)
  (h2 : B + G = 455) : 
  G = 175 := 
sorry

end college_girls_count_l179_179864


namespace arrange_numbers_l179_179127

noncomputable def ψ : ℤ := (1 / 2) * (Finset.sum (Finset.range 1006) (λ n, if n % 4 = 0 ∨ n % 4 = 1 then n + 1 else -(n + 1)))

noncomputable def ω : ℤ := Finset.sum (Finset.range 1007) (λ n, if n % 2 = 0 then (2 * n + 1) else -(2 * n + 1))

noncomputable def θ : ℤ := Finset.sum (Finset.range 1008) (λ n, if n % 2 = 0 then (2 * n + 1) else (-(2 * n + 1)))

theorem arrange_numbers :
  θ <= ω ∧ ω <= ψ :=
  sorry

end arrange_numbers_l179_179127


namespace tricolor_partition_l179_179514

theorem tricolor_partition (n : ℕ) (h : n ≥ 3)
  (coloring : Fin n → ℕ)
  (coloring_conditions : (∀ i, coloring i ∈ {0, 1, 2}) 
    ∧ (∀ i : Fin n, coloring i ≠ coloring ((i + 1) % n))
    ∧ (∃ i : Fin n, coloring i = 0) 
    ∧ (∃ i : Fin n, coloring i = 1) 
    ∧ (∃ i : Fin n, coloring i = 2)) :
  ∃ (triangles : List (Fin n × Fin n × Fin n)),
    (∀ (t : Fin n × Fin n × Fin n), t ∈ triangles → (t.1 ≠ t.2 ∧ t.2 ≠ t.3 ∧ t.1 ≠ t.3)) ∧
    (∀ (t : Fin n × Fin n × Fin n), t ∈ triangles → (coloring t.1 ≠ coloring t.2 ∧ coloring t.2 ≠ coloring t.3 ∧ coloring t.1 ≠ coloring t.3)) ∧
    (∀ (v : Fin n), ∃ t ∈ triangles, v = t.1 ∨ v = t.2 ∨ v = t.3) ∧
    (∀ (t1 t2 : Fin n × Fin n × Fin n), t1 ∈ triangles → t2 ∈ triangles → t1 ≠ t2 → Disjoint (Finset.product (Finset.singleton t1.1) (Finset.singleton t1.2) ∪ Finset.product (Finset.singleton t1.2) (Finset.singleton t1.3) ∪ Finset.product (Finset.singleton t1.3) (Finset.singleton t1.1)) (Finset.product (Finset.singleton t2.1) (Finset.singleton t2.2) ∪ Finset.product (Finset.singleton t2.2) (Finset.singleton t2.3) ∪ Finset.product (Finset.singleton t2.3) (Finset.singleton t2.1))))
:= sorry

end tricolor_partition_l179_179514


namespace total_number_of_seashells_l179_179012

-- Defining the conditions
def number_of_broken_seashells : Nat := 4
def number_of_unbroken_seashells : Nat := 3

-- Stating the theorem to be proven
theorem total_number_of_seashells : number_of_broken_seashells + number_of_unbroken_seashells = 7 :=
by
  rw [number_of_broken_seashells, number_of_unbroken_seashells]
  exact rfl

end total_number_of_seashells_l179_179012


namespace number_of_bus_routes_l179_179667

theorem number_of_bus_routes (V R : ℕ) 
    (h1 : ∀ (s₁ s₂ : ℕ), s₁ ≠ s₂ → ∃ r, r ≠ 0 ∧ some_route s₁ r ∧ some_route s₂ r)
    (h2 : ∀ (r₁ r₂ : ℕ), r₁ ≠ r₂ → ∃! s, some_route s r₁ ∧ some_route s r₂)
    (h3 : ∀ r, ∃! s₁ s₂ s₃, some_route s₁ r ∧ some_route s₂ r ∧ some_route s₃ r)
    (h4 : R > 1) : 
  R = 7 :=
by
  sorry

def some_route : ℕ → ℕ → Prop := 
by 
  sorry

noncomputable def number_of_routes (V : ℕ) : ℕ :=
by
  sorry

end number_of_bus_routes_l179_179667


namespace proof_problem_l179_179354

-- Definitions
variable (T : Type) (Sam : T)
variable (solves_all : T → Prop) (passes : T → Prop)

-- Given condition (Dr. Evans's statement)
axiom dr_evans_statement : ∀ x : T, solves_all x → passes x

-- Statement to be proven
theorem proof_problem : ¬ (passes Sam) → ¬ (solves_all Sam) :=
  by sorry

end proof_problem_l179_179354


namespace min_congestion_route_l179_179086

theorem min_congestion_route :
  ∀ (P_AC P_CD P_DB P_CF P_FB P_AE P_EF : ℝ),
  P_AC = 9/10 → P_CD = 14/15 → P_DB = 5/6 → P_CF = 2/5 → 
  P_FB = 7/8 → P_AE = 4/5 → P_EF = 4/5 →
  let P_no_congestion_ACD = P_AC * P_CD * P_DB in
  let P_no_congestion_ACF = P_AC * P_CF * P_FB in
  let P_no_congestion_AEF = P_AE * P_EF * P_FB in
  let P_congestion_ACD = 1 - P_no_congestion_ACD in
  let P_congestion_ACF = 1 - P_no_congestion_ACF in
  let P_congestion_AEF = 1 - P_no_congestion_AEF in
  P_congestion_ACD ≤ P_congestion_ACF ∧ P_congestion_ACD ≤ P_congestion_AEF :=
by
  intros
  sorry

end min_congestion_route_l179_179086


namespace skier_total_time_l179_179010

variable (t1 t2 t3 : ℝ)

-- Conditions
def condition1 : Prop := t1 + t2 = 40.5
def condition2 : Prop := t2 + t3 = 37.5
def condition3 : Prop := 1 / t2 = 2 / (t1 + t3)

-- Theorem to prove total time is 58.5 minutes
theorem skier_total_time (h1 : condition1 t1 t2) (h2 : condition2 t2 t3) (h3 : condition3 t1 t2 t3) : t1 + t2 + t3 = 58.5 := 
by
  sorry

end skier_total_time_l179_179010


namespace train_passes_platform_in_approx_22_seconds_l179_179101

noncomputable def time_to_pass_platform (train_speed_kmh : ℕ) (time_to_pass_man : ℕ) (platform_length_m : ℝ) : ℕ := 
let train_speed_ms := (train_speed_kmh * 1000) / 3600 in
let train_length_m := time_to_pass_man * train_speed_ms in
let total_distance_m := train_length_m + platform_length_m in
let time_to_pass_platform_s := total_distance_m / train_speed_ms in
Nat.round time_to_pass_platform_s

theorem train_passes_platform_in_approx_22_seconds :
  time_to_pass_platform 54 20 30.0024  = 22 :=
by
  sorry

end train_passes_platform_in_approx_22_seconds_l179_179101


namespace exponent_of_5_in_30_factorial_l179_179946

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l179_179946


namespace larger_ball_radius_l179_179612

theorem larger_ball_radius :
  let V_small := (4 / 3) * Real.pi * 1^3,
      V_total := 12 * V_small,
      r := Real.cbrt (V_total * 3 / (4 * Real.pi)) 
  in r = Real.cbrt 12 :=
by
  let V_small := (4 / 3) * Real.pi * 1^3
  let V_total := 12 * V_small
  let r := Real.cbrt (V_total * 3 / (4 * Real.pi))
  sorry

end larger_ball_radius_l179_179612


namespace books_sold_on_wednesday_l179_179524

theorem books_sold_on_wednesday
  (stock : ℕ)
  (sold_Mon : ℕ)
  (sold_Tue : ℕ)
  (sold_Thu : ℕ)
  (sold_Fri : ℕ)
  (percent_not_sold : ℚ)
  (h_stock : stock = 1100)
  (h_sold_Mon : sold_Mon = 75)
  (h_sold_Tue : sold_Tue = 50)
  (h_sold_Thu : sold_Thu = 78)
  (h_sold_Fri : sold_Fri = 135)
  (h_percent_not_sold : percent_not_sold = 63.45) : 
  let total_books_not_sold := (percent_not_sold / 100) * stock in
  let total_books_sold := stock - total_books_not_sold in
  let books_sold_excluding_Wednesday := sold_Mon + sold_Tue + sold_Thu + sold_Fri in
  let sold_Wed := total_books_sold - books_sold_excluding_Wednesday in
  sold_Wed = 64 := 
by 
  sorry

end books_sold_on_wednesday_l179_179524


namespace find_y_l179_179853

-- Defining the parameters
def num_steps_to_zero : ℕ := 0
def num_steps_to_twenty_five : ℕ := 5
def distance_to_twenty_five : ℕ := 25
def num_steps_to_y : ℕ := 4

-- Conclusion we want to prove
theorem find_y (steps_eq_dist: num_steps_to_twenty_five = 5)
               (dist_eq_25: distance_to_twenty_five = 25)
               (steps_to_y: num_steps_to_y = 4)
               : (num_steps_to_y * (distance_to_twenty_five / num_steps_to_twenty_five)) = 20 :=
by simp [steps_eq_dist, dist_eq_25, steps_to_y]; norm_num;
   sorry

end find_y_l179_179853


namespace range_of_a_l179_179817

noncomputable def f (a b x : ℝ) : ℝ := x^4 + a * x^3 + 2 * x^2 + b

def derivative_f (a b x : ℝ) : ℝ := 4*x^3 + 3*a*x^2 + 4*x

theorem range_of_a (a b : ℝ) (h : ∀ x ∈ ℝ, derivative_f a b x = 0 → x = 0) :
  -8/3 ≤ a ∧ a ≤ 8/3 :=
sorry

end range_of_a_l179_179817


namespace smallest_possible_bob_number_l179_179104

theorem smallest_possible_bob_number (b : ℕ) (h : ∀ p: ℕ, p.prime → p ∣ 30 → p ∣ b) : b ≥ 30 :=
sorry

end smallest_possible_bob_number_l179_179104


namespace cistern_fill_time_l179_179071

theorem cistern_fill_time (hA : ∀ C : ℝ, 0 < C → ∀ t : ℝ, 0 < t → C / t = C / 10) 
                          (hB : ∀ C : ℝ, 0 < C → ∀ t : ℝ, 0 < t → C / t = -(C / 15)) :
  ∀ C : ℝ, 0 < C → ∃ t : ℝ, t = 30 := 
by 
  sorry

end cistern_fill_time_l179_179071


namespace triangle_is_right_l179_179516

theorem triangle_is_right 
  (A B C : ℝ) 
  (h : sin A ^ 2 = sin B ^ 2 + sin C ^ 2) 
  (h_triangle : A + B + C = 180) :
  A = 90 :=
sorry

end triangle_is_right_l179_179516


namespace exponent_of_5_in_30_factorial_l179_179993

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179993


namespace arrange_in_non_decreasing_order_l179_179129

def Psi : ℤ := (1 / 2 : ℚ) * (Finset.sum (Finset.range 1006) (λ i, 1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + (4 * i)))

def Omega : ℤ := Finset.sum (Finset.range 1007) (λ i, 1 - 2 + 3 - 4 + (2 * i))

def Theta : ℤ := Finset.sum (Finset.range 504) (λ i, 1 - 3 + 5 - 7 + (4 * i))

theorem arrange_in_non_decreasing_order : [Theta, Omega, Psi] = [(-1008 : ℤ), (-1007 : ℤ), (-1006 : ℤ)] :=
by
  have : Psi = -1006 := sorry
  have : Omega = -1007 := sorry
  have : Theta = -1008 := sorry
  rw [this, this, this]
  trivial

end arrange_in_non_decreasing_order_l179_179129


namespace correct_propositions_l179_179109

-- Definitions of polyhedra and their structural characteristics:
def isPrism (P : Polyhedron) : Prop :=
  ∃ (F1 F2 : Face), F1 ∈ P.faces ∧ F2 ∈ P.faces ∧ F1 ≠ F2 ∧ F1.is_parallel_to F2 ∧
  ∀ (F : Face), F ∈ P.faces ∧ F ≠ F1 ∧ F ≠ F2 → F.is_parallelogram

def isQuadrangularPyramid (Q : Polyhedron) : Prop :=
  ∃ (F1 F2 F3 F4 : Face), Q.is_pyramid ∧
  F1 ∈ Q.lateral_faces ∧ F2 ∈ Q.lateral_faces ∧ F3 ∈ Q.lateral_faces ∧ F4 ∈ Q.lateral_faces ∧
  F1.is_right_angled_triangle ∧ F2.is_right_angled_triangle ∧ 
  F3.is_right_angled_triangle ∧ F4.is_right_angled_triangle

def isTruncatedPrism (T : Polyhedron) : Prop :=
  ∃ (F1 F2 : Face), F1 ∈ T.faces ∧ F2 ∈ T.faces ∧ F1 ≠ F2 ∧ F1.is_parallel_to F2 ∧
  ∀ (F : Face), F ∈ T.faces ∧ F ≠ F1 ∧ F ≠ F2 → F.is_trapezoid

def isTriangularPyramid (T : Polyhedron) : Prop :=
  T.is_pyramid ∧ T.base.is_triangle

-- The Lean theorem statement
theorem correct_propositions :
  (¬ ∀ P, isPrism P) ∧ (∀ Q, isQuadrangularPyramid Q) ∧ 
  (¬ ∀ T, isTruncatedPrism T) ∧ (∀ T, isTriangularPyramid T)
:= by sorry

end correct_propositions_l179_179109


namespace no_show_last_year_l179_179527

theorem no_show_last_year (signed_up_last_year : ℕ) (runners_this_year : ℕ) (twice_last_year : runners_this_year = 2 * (signed_up_last_year - no_show)) :
    signed_up_last_year = 200 ∧ runners_this_year = 320 →
    no_show = 40 :=
by
  intro h
  cases h with h1 h2
  have : 2 * (signed_up_last_year - no_show) = runners_this_year := h2.symm
  rw [←h1, ←h2] at this
  linarith

end no_show_last_year_l179_179527


namespace cos_double_angle_l179_179782

variables {α β : ℝ}

theorem cos_double_angle (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_l179_179782


namespace sin_45_eq_one_div_sqrt_two_l179_179220

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179220


namespace alternating_sum_binom_l179_179717

open BigOperators

theorem alternating_sum_binom (n : ℕ) (h : n = 100) : 
  ∑ k in finset.range (n.div 2 + 1), (nat.choose n (2 * k) * (-1)^k) = -2^50 := 
by
  sorry

end alternating_sum_binom_l179_179717


namespace exponent_of_5_in_30_fact_l179_179877

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179877


namespace hyperbola_focus_y_axis_l179_179848

theorem hyperbola_focus_y_axis (m : ℝ) :
  (∀ x y : ℝ, (m + 1) * x^2 + (2 - m) * y^2 = 1) → m < -1 :=
sorry

end hyperbola_focus_y_axis_l179_179848


namespace prime_exponent_of_5_in_30_factorial_l179_179905

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l179_179905


namespace container_volume_ratio_l179_179105

variables (A B C : ℝ)

theorem container_volume_ratio (h1 : (2 / 3) * A = (1 / 2) * B) (h2 : (1 / 2) * B = (3 / 5) * C) :
  A / C = 6 / 5 :=
sorry

end container_volume_ratio_l179_179105


namespace sin_45_eq_sqrt2_div_2_l179_179204

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l179_179204


namespace quadratic_inequality_solution_range_l179_179852

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x, 1 < x ∧ x < 4 ∧ x^2 - 4 * x - 2 - a > 0) → a < -2 :=
sorry

end quadratic_inequality_solution_range_l179_179852


namespace set_equality_l179_179533

def M : Set ℝ := {x | x^2 - x > 0}

def N : Set ℝ := {x | 1 / x < 1}

theorem set_equality : M = N := 
by
  sorry

end set_equality_l179_179533


namespace sin_45_eq_one_div_sqrt_two_l179_179215

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179215


namespace maximum_value_of_f_l179_179373

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem maximum_value_of_f : ∃ x : ℝ, f x = 13 :=
by 
  sorry

end maximum_value_of_f_l179_179373


namespace symmetrical_coordinates_y_axis_l179_179845

/-- Q (a, b) is symmetrical to P (4, -5) about the y-axis -/
theorem symmetrical_coordinates_y_axis (a b : ℝ) : 
  (b = -5) ∧ (a = -4) ↔ (P = (4, -5) ∧ P = (-a, b)) :=
begin
  sorry
end

end symmetrical_coordinates_y_axis_l179_179845


namespace darry_steps_l179_179731

theorem darry_steps (f_steps : ℕ) (f_times : ℕ) (s_steps : ℕ) (s_times : ℕ) (no_other_steps : ℕ)
  (hf : f_steps = 11)
  (hf_times : f_times = 10)
  (hs : s_steps = 6)
  (hs_times : s_times = 7)
  (h_no_other : no_other_steps = 0) :
  (f_steps * f_times + s_steps * s_times + no_other_steps = 152) :=
by
  sorry

end darry_steps_l179_179731


namespace arrange_numbers_l179_179126

noncomputable def ψ : ℤ := (1 / 2) * (Finset.sum (Finset.range 1006) (λ n, if n % 4 = 0 ∨ n % 4 = 1 then n + 1 else -(n + 1)))

noncomputable def ω : ℤ := Finset.sum (Finset.range 1007) (λ n, if n % 2 = 0 then (2 * n + 1) else -(2 * n + 1))

noncomputable def θ : ℤ := Finset.sum (Finset.range 1008) (λ n, if n % 2 = 0 then (2 * n + 1) else (-(2 * n + 1)))

theorem arrange_numbers :
  θ <= ω ∧ ω <= ψ :=
  sorry

end arrange_numbers_l179_179126


namespace sin_45_eq_l179_179260

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l179_179260


namespace equal_cylinder_volumes_l179_179633

theorem equal_cylinder_volumes (x : ℝ) (hx : x > 0) :
  π * (5 + x) ^ 2 * 4 = π * 25 * (4 + x) → x = 35 / 4 :=
by
  sorry

end equal_cylinder_volumes_l179_179633


namespace cos_product_value_l179_179394

open Real

theorem cos_product_value (α : ℝ) (h : sin α = 1 / 3) : 
  cos (π / 4 + α) * cos (π / 4 - α) = 7 / 18 :=
by
  sorry

end cos_product_value_l179_179394


namespace exponent_of_5_in_30_factorial_l179_179920

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179920


namespace solve_equation_l179_179766

theorem solve_equation :
  ∀ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) → x = 105 :=
by
  intros x h
  sorry

end solve_equation_l179_179766


namespace number_of_functions_l179_179052

def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem number_of_functions (A = {1, 2, 3, 4, 5, 6, 7, 8}) :
  (∃ (f : ℕ → ℕ), (∀ (i j : ℕ), i < j → f i < f j) ∧ (f 1 ∈ A) ∧ (f 2 ∈ A) ∧ (f 3 ∈ A)) →
  (finset.card (choose A 3) * 8^5 = nat.binom 8 3 * 8^5) :=
by
  sorry

end number_of_functions_l179_179052


namespace compute_delta_y2_l179_179554

-- Define the increment in x and the corresponding increment in y based on the problem's conditions
def delta_x1 := 2
def delta_y1 := 5
def delta_x2 := 8

-- Define the linear relationship (slope)
def slope := delta_y1 / delta_x1

-- The target increment in y for the given change in x
def delta_y2 := slope * delta_x2

-- The theorem that states the resulting increase in y
theorem compute_delta_y2 (h1 : delta_y1 = 5) (h2 : delta_x1 = 2) (h3 : delta_x2 = 8) : delta_y2 = 20 :=
  by
  -- Proof is skipped here, the statement itself should be enough.
  sorry

end compute_delta_y2_l179_179554


namespace probability_triangle_formed_l179_179698

/-- 
Given a rod of length l, broken into three parts at random points (x, y),
prove that the probability of forming a triangle with these parts is 1/4.
-/
theorem probability_triangle_formed (l : ℝ) :
  let x := choose (0, l) in let y := choose (x, l) in
  -- conditions for forming a triangle
  (x < l ∧ x + (l - y) > l / 2 ∧ (l - x) ≥ y) →
  -- probability of forming the triangle
  probability (form_triangle x (y - x) (l - y)) = 1/4 :=
sorry

end probability_triangle_formed_l179_179698


namespace sin_45_degree_l179_179276

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l179_179276


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179240

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179240


namespace square_sum_inverse_eq_23_l179_179662

theorem square_sum_inverse_eq_23 {x : ℝ} (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 :=
by
  sorry

end square_sum_inverse_eq_23_l179_179662


namespace coupons_used_l179_179098

theorem coupons_used
  (initial_books : ℝ)
  (sold_books : ℝ)
  (coupons_per_book : ℝ)
  (remaining_books := initial_books - sold_books)
  (total_coupons := remaining_books * coupons_per_book) :
  initial_books = 40.0 →
  sold_books = 20.0 →
  coupons_per_book = 4.0 →
  total_coupons = 80.0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end coupons_used_l179_179098


namespace sin_45_degree_eq_sqrt2_div_2_l179_179156

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l179_179156


namespace sin_45_eq_sqrt2_div_2_l179_179296

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179296


namespace find_years_l179_179099

def sum_interest_years (P R : ℝ) (T : ℝ) : Prop :=
  (P * (R + 5) / 100 * T = P * R / 100 * T + 300) ∧ P = 600

theorem find_years {R : ℝ} {T : ℝ} (h1 : sum_interest_years 600 R T) : T = 10 :=
by
  -- proof omitted
  sorry

end find_years_l179_179099


namespace sin_45_eq_sqrt2_div_2_l179_179338

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l179_179338


namespace exponent_of_5_in_30_factorial_l179_179929

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179929


namespace possible_values_of_k_l179_179856

theorem possible_values_of_k (N : ℕ) (K : ℕ) 
  (h1 : N < 50)
  (h2 : K = 2 * N - 1) 
  (h3 : ∑ i in finset.range (N + 1), (2 * i + 1) = N^2) : 
  (K = 1 ∨ K = 3 ∨ K = 5 ∨ K = 7 ∨ K = 9 ∨ K = 11 ∨ K = 13 ∨ K = 15 ∨ 
   K = 17 ∨ K = 19 ∨ K = 21 ∨ K = 23 ∨ K = 25 ∨ K = 27 ∨ K = 29 ∨ 
   K = 31 ∨ K = 33 ∨ K = 35 ∨ K = 37 ∨ K = 39 ∨ K = 41 ∨ K = 43 ∨ 
   K = 45 ∨ K = 47 ∨ K = 49 ∨ K = 51 ∨ K = 53 ∨ K = 55 ∨ K = 57 ∨ 
   K = 59 ∨ K = 61 ∨ K = 63 ∨ K = 65 ∨ K = 67 ∨ K = 69 ∨ K = 71 ∨ 
   K = 73 ∨ K = 75 ∨ K = 77 ∨ K = 79 ∨ K = 81 ∨ K = 83 ∨ K = 85 ∨ 
   K = 87 ∨ K = 89 ∨ K = 91 ∨ K = 93 ∨ K = 95 ∨ K = 97) :=
sorry

end possible_values_of_k_l179_179856


namespace max_ab_min_inv_a_plus_4_div_b_l179_179412

theorem max_ab (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_sum : a + 4 * b = 4) : 
  ab ≤ 1 :=
by
  sorry

theorem min_inv_a_plus_4_div_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_sum : a + 4 * b = 4) :
  1 / a + 4 / b ≥ 25 / 4 :=
by
  sorry

end max_ab_min_inv_a_plus_4_div_b_l179_179412


namespace Bernie_postcards_l179_179714

/-- 
Bernie has a collection of 18 unique postcards. 
He sells 6 postcards at $12 each, 3 postcards at $15 each, and 2 postcards at $10 each. 
He spends 70% of the earned money to buy new postcards at $8 each. 
He buys another 5 postcards at $6 each using the remaining money, 
but can only afford to buy 1 more postcard. 
Finally, we want to know how many postcards Bernie has after all his transactions.
-/
theorem Bernie_postcards :
  Bernie's final postcard count = 19 :=
sorry

end Bernie_postcards_l179_179714


namespace prime_factors_difference_l179_179646

theorem prime_factors_difference (n : ℤ) (h₁ : n = 180181) : ∃ p q : ℤ, Prime p ∧ Prime q ∧ p > q ∧ n % p = 0 ∧ n % q = 0 ∧ (p - q) = 2 :=
by
  sorry

end prime_factors_difference_l179_179646


namespace exponent_of_5_in_30_factorial_l179_179928

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l179_179928


namespace exponent_of_5_in_30_factorial_l179_179979

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l179_179979


namespace find_intersection_l179_179809

-- Definitions given in the conditions
def linear_function (k b x : ℝ) := k*x + b
def proportional_function (x : ℝ) := (1 / 2) * x

-- Theorem statements to prove the questions with given conditions
theorem find_intersection (a : ℝ) (k b : ℝ) :
  linear_function k b (-1) = -5 →
  proportional_function 2 = a →
  linear_function k b 2 = a →
  a = 1 ∧ k = 2 ∧ b = -3 :=
by {
  intros h₁ h₂ h₃,
  rw [linear_function, proportional_function] at *,
  have h₄ : -k + b = -5, by linarith,
  have h₅ : 1 = 2 * k + b, by linarith,
  have : k = 2, by linarith,
  have : b = -3, by linarith,
  subst_vars,
  split; refl,
}


end find_intersection_l179_179809


namespace sin_45_eq_sqrt2_div_2_l179_179289

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179289


namespace sin_45_eq_one_div_sqrt_two_l179_179226

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l179_179226


namespace composite_function_value_l179_179398

def f (x : ℝ) : ℝ :=
  if x > 0 then real.log x / real.log 2 else 3^x + 1

theorem composite_function_value : f (f (1 / 4)) = 10 / 9 :=
by
  sorry

end composite_function_value_l179_179398


namespace ratio_volumes_l179_179523

-- Define conditions
def diameter_john : ℝ := 8 -- John's can diameter in cm
def height_john : ℝ := 16 -- John's can height in cm
def diameter_maria : ℝ := 16 -- Maria's can diameter in cm
def height_maria : ℝ := 8 -- Maria's can height in cm

-- Volume of a cylinder
def volume (d h : ℝ) : ℝ := 
  let r := d / 2
  Math.pi * r^2 * h

-- Define volumes based on given conditions
def volume_john : ℝ := volume diameter_john height_john
def volume_maria : ℝ := volume diameter_maria height_maria

-- Theorem stating the required ratio
theorem ratio_volumes : volume_john / volume_maria = 1 / 2 :=
by
  -- Mathematical proof here
  sorry

end ratio_volumes_l179_179523


namespace exponent_of_5_in_30_fact_l179_179889

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l179_179889


namespace problem_l179_179841

theorem problem (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = 1) 
  (h3 : a + c + d = 16) 
  (h4 : b + c + d = 9) : 
  a * b + c * d = 734 / 9 := 
by 
  sorry

end problem_l179_179841


namespace ellipse_equation_l179_179791

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0)
    (eccentricity : ℝ := (√3) / 3)
    (perimeter : ℝ := 4 * √3)
    (h3 : eccentricity = (√3) / 3)
    (h4 : perimeter = 4 * √3) :
    (∃ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 / 2 = 1}) :=
sorry

end ellipse_equation_l179_179791


namespace circle_cover_polygon_l179_179787

noncomputable def midpoint (A B: ℝ) : ℝ := (A + B) / 2

theorem circle_cover_polygon (P : Set ℝ) (A B O : ℝ) (h_P_closed : is_closed P) (h_perimeter : P.perimeter = 1)
    (h_midpoint : O = midpoint A B) :
    ∀ M ∈ P, abs (O - M) ≤ (1 / 4) :=
by
  sorry

end circle_cover_polygon_l179_179787


namespace sin_45_eq_1_div_sqrt_2_l179_179313

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l179_179313


namespace circle_tangent_problem_l179_179595

theorem circle_tangent_problem : ∃ a b : ℝ, (10 * (Real.sqrt 2 - 1) * (Real.sqrt 2 - 1) = a - b * Real.sqrt 2) ∧ (a + b = 50) :=
by
  use 30, 20
  split
  sorry
  sorry

end circle_tangent_problem_l179_179595


namespace sin_45_deg_eq_l179_179300

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l179_179300


namespace sin_45_degree_l179_179166

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l179_179166


namespace minimum_balls_l179_179686

/-- Given that tennis balls are stored in big boxes containing 25 balls each 
    and small boxes containing 20 balls each, and the least number of balls 
    that can be left unboxed is 5, prove that the least number of 
    freshly manufactured balls is 105.
-/
theorem minimum_balls (B S : ℕ) : 
  ∃ (n : ℕ), 25 * B + 20 * S = n ∧ n % 25 = 5 ∧ n % 20 = 5 ∧ n = 105 := 
sorry

end minimum_balls_l179_179686


namespace intersection_point_l179_179528

noncomputable def f : ℝ → ℝ := λ x, (x^2 - 8*x + 12) / (2*x - 6)

noncomputable def g : ℝ → ℝ := λ x, (-2*x - 4 + 29 / (x - 3))

theorem intersection_point :
  ∃ y : ℝ, ¬(14 = -3) ∧ f 14 = g 14 ∧ y = f 14 :=
begin
  sorry
end

end intersection_point_l179_179528


namespace option_D_correct_l179_179656

variable (a : ℚ)

theorem option_D_correct : -3 * (a - 1) = 3 - 3 * a :=
by
  calc
    -3 * (a - 1) = -3 * a + 3       : by ring
             ... = 3 - 3 * a        : by ring

end option_D_correct_l179_179656


namespace allan_initial_balloons_l179_179106

theorem allan_initial_balloons (jake_balloons allan_bought_more allan_total_balloons : ℕ) 
  (h1 : jake_balloons = 4)
  (h2 : allan_bought_more = 3)
  (h3 : allan_total_balloons = 8) :
  ∃ (allan_initial_balloons : ℕ), allan_total_balloons = allan_initial_balloons + allan_bought_more ∧ allan_initial_balloons = 5 := 
by
  sorry

end allan_initial_balloons_l179_179106


namespace height_difference_correct_l179_179634

-- Diameter of each cylindrical can
def diameter : ℝ := 12

-- Total height calculation for Crate A with side-by-side packing
def crateA_total_height : ℝ := 15 * diameter

-- Vertical center-to-center distance between staggered rows in Crate B
def staggered_distance : ℝ := 6 * Real.sqrt 3

-- Total height calculation for Crate B with staggered packing
def crateB_total_height : ℝ := 12 + 12 * staggered_distance

-- Difference in heights of the two packing methods
def difference_in_heights : ℝ := crateA_total_height - crateB_total_height

-- The positive difference should be 43.3 cm
theorem height_difference_correct : Abs.abs (difference_in_heights) = 43.3 := by
  sorry

end height_difference_correct_l179_179634


namespace ellipse_equation_range_of_k_l179_179807

-- (I) Statement for part I
theorem ellipse_equation (h_e : (x : ℝ)^2 - (y : ℝ)^2 / 3 = 1)
                         (P : ℝ × ℝ) (hP : P = (1, 3 / 2))
                         (ecc_h : ℝ := 2)
                         (ecc_e : ℝ := 1 / ecc_h)
                         (ha : (a : ℝ) = 2 * (c : ℝ))
                         (hb : (b : ℝ) = sqrt (a^2 - c^2))
                         (eq_ellipse : (x : ℝ)^2 / (4 * c^2) + (y : ℝ)^2 / (3 * c^2) = 1)
                         (h_c : (c : ℝ)^2 = 1) :
  (x : ℝ)^2 / 4 + (y : ℝ)^2 / 3 = 1 := sorry

-- (II) Statement for part II
theorem range_of_k (ellipse_eq : (x : ℝ)^2 / 4 + (y : ℝ)^2 / 3 = 1)
                   (P : ℝ × ℝ) (hP : P = (1 / 5, 0))
                   (k : ℝ) (h_k : k ≠ 0)
                   (M N : ℝ × ℝ) (hM : ∃ x y, M = (x, y) ∧ ((x : ℝ)^2 / 4 + (y : ℝ)^2 / 3 = 1))
                   (hN : ∃ x y, N = (x, y) ∧ ((x : ℝ)^2 / 4 + (y : ℝ)^2 / 3 = 1))
                   (h_intersect : ∃ y m, ∃! x1 x2, y = k * x + m ∧ (x1 : ℝ)^2 / 4 + (y(x1) : ℝ)^2 / 3 = 1 ∧ (x2 : ℝ)^2 / 4 + (y(x2) : ℝ)^2 / 3 = 1)
                   (h_bisector : ∀ x y, y = -1 / k * (x - 1 / 5)) :
  k^2 > 1 / 7 → ((k < -sqrt (7) / 7) ∨ (k > sqrt (7) / 7)) := sorry

end ellipse_equation_range_of_k_l179_179807


namespace Moe_has_least_amount_of_money_l179_179715

variables (Money : Type) [LinearOrder Money]
variables (Bo Coe Flo Jo Moe Zoe : Money)
variables (Bo_lt_Flo : Bo < Flo) (Jo_lt_Flo : Jo < Flo)
variables (Moe_lt_Bo : Moe < Bo) (Moe_lt_Coe : Moe < Coe)
variables (Moe_lt_Jo : Moe < Jo) (Jo_lt_Bo : Jo < Bo)
variables (Moe_lt_Zoe : Moe < Zoe) (Zoe_lt_Jo : Zoe < Jo)

theorem Moe_has_least_amount_of_money : ∀ x, x ≠ Moe → Moe < x := by
  sorry

end Moe_has_least_amount_of_money_l179_179715


namespace max_surface_area_of_rectangular_solid_l179_179788

theorem max_surface_area_of_rectangular_solid {r a b c : ℝ} (h_sphere : 4 * π * r^2 = 4 * π)
  (h_diagonal : a^2 + b^2 + c^2 = (2 * r)^2) :
  2 * (a * b + a * c + b * c) ≤ 8 :=
by
  sorry

end max_surface_area_of_rectangular_solid_l179_179788


namespace find_solution_l179_179368

noncomputable def equation (x : ℝ) :=
  real.sqrt x + 3 * real.sqrt (x^2 + 9 * x) + real.sqrt (x + 9) = 45 - 3 * x

theorem find_solution : (∃ x : ℝ, equation x ∧ x = 400 / 49) :=
begin
  sorry -- Proof is omitted
end

end find_solution_l179_179368


namespace length_of_AP_l179_179594

variables (X Y Z A P Q S T M : Type) [metric_space X] [metric_space Y] [metric_space Z]
  [metric_space A] [metric_space P] [metric_space Q] [metric_space S] [metric_space T] [metric_space M]

-- lengths of sides of triangle XYZ
variable (XY : ℝ) (YZ : ℝ) (XZ : ℝ)

-- conditions: lengths of sides of triangle XYZ
def triangle_XYZ := (XY = 2) ∧ (YZ = 3) ∧ (XZ = 4)

-- lines parallel to the sides of triangle XYZ
def lines_parallel := (parallel AMB XYZ) ∧ (parallel PMQ XY) ∧ (parallel SMT YZ)

-- lengths of AP, QS, and BT are equal
def equal_segments (x : ℝ) := (dist A P = x) ∧ (dist Q S = x) ∧ (dist B T = x)

-- proof that the length of AP is 12/13
theorem length_of_AP : 
    (triangle_XYZ XY YZ XZ) ∧ (lines_parallel AMB PMQ SMT XY YZ XZ) ∧ 
    (∃ x, equal_segments x) → 
    ∃ x, x = 12 / 13 :=
sorry

end length_of_AP_l179_179594


namespace time_to_fill_pipe_A_l179_179636

-- Definitions based on the conditions
def rate_B := 39.99999999999999 / 5  -- Pipe B fills rate in liters per minute
def rate_C := 14                      -- Pipe C emptying rate in liters per minute
def total_volume := 39.99999999999999 -- Volume of the cistern in liters
def emptying_time := 60                -- Time to empty the cistern in minutes with all pipes open

-- Hypothesis based on given conditions
theorem time_to_fill_pipe_A :
  ∃ rate_A : ℝ, (rate_A + rate_B - rate_C = - (total_volume / emptying_time)) ∧ 
               (total_volume / rate_A = 7.5) :=
begin
  -- Statements translated into conditions
  let rate_B := 39.99999999999999 / 5,
  let rate_C := 14,
  let total_volume := 39.99999999999999,
  let emptying_time := 60,
  
  -- The solution expression showing that rate_A should be 5.3333333333333335
  use 5.3333333333333335,
  split,
  { -- First part of the equivalency: rate_A = - (total_volume / emptying_time) + 6
    calc
      5.3333333333333335 + rate_B - rate_C = 5.3333333333333335 + 8 - 14 : by norm_num
      ... = - (total_volume / emptying_time) : by norm_num
  },
  { -- Second part of the time calculation: total_volume / rate_A = 7.5
    calc
      total_volume / 5.3333333333333335 = 7.5 : by norm_num
  }
end

end time_to_fill_pipe_A_l179_179636


namespace medians_form_right_triangle_l179_179015

open Real

-- Define the medians and perpendicular condition
variables {a b c m_a m_b m_c : ℝ}

-- Define the given condition
axiom medians_perpendicular : m_a ⊥ m_b

-- The theorem to prove
theorem medians_form_right_triangle (h : m_a ⊥ m_b) : m_a^2 + m_b^2 = m_c^2 :=
sorry

end medians_form_right_triangle_l179_179015


namespace sin_45_eq_sqrt2_div_2_l179_179288

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l179_179288


namespace num_pos_three_digit_ints_with_0_units_div_25_eq_9_l179_179835

theorem num_pos_three_digit_ints_with_0_units_div_25_eq_9 :
  {n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ n % 100 = 0}.card = 9 :=
sorry

end num_pos_three_digit_ints_with_0_units_div_25_eq_9_l179_179835


namespace probability_sum_divisible_by_3_l179_179006

theorem probability_sum_divisible_by_3: 
  (let face_values := {1, 2, 3, 4, 5, 6},
       prob := λ r, (1 : ℚ) / 6,
       sum := λ x y z, (x + y + z) % 3,
       is_div_3 := λ n, n % 3 = 0,
       all_possible_outcomes := { (x, y, z) | x ∈ face_values ∧ y ∈ face_values ∧ z ∈ face_values },
       favorable_outcomes := { (x, y, z) | (x, y, z) ∈ all_possible_outcomes ∧ is_div_3 (sum x y z) }
  in (favorable_outcomes.card : ℚ) / (all_possible_outcomes.card : ℚ)) = 1 / 3 :=
by
  sorry

end probability_sum_divisible_by_3_l179_179006


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179239

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179239


namespace sin_45_deg_l179_179186

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l179_179186


namespace distribution_ways_l179_179453

-- Definitions based on the conditions from part a)
def distinguishable_balls : ℕ := 6
def indistinguishable_boxes : ℕ := 4

-- The theorem to prove the question equals the correct answer given the conditions
theorem distribution_ways (n : ℕ) (k : ℕ) (h_n : n = distinguishable_balls) (h_k : k = indistinguishable_boxes) : 
  number_of_distributions n k = 262 := 
sorry

end distribution_ways_l179_179453


namespace cos_double_angle_l179_179781

variables {α β : ℝ}

theorem cos_double_angle (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_l179_179781


namespace trains_crossing_time_l179_179017

-- Definitions for conditions
def length_first_train : ℝ := 140
def length_second_train : ℝ := 150
def speed_first_train_kmh : ℝ := 60
def speed_second_train_kmh : ℝ := 40
def kmhr_to_mps (v : ℝ) : ℝ := (v * 1000) / 3600

-- The main statement
theorem trains_crossing_time :
  let total_distance := length_first_train + length_second_train,
      relative_speed_kmh := speed_first_train_kmh + speed_second_train_kmh,
      relative_speed_mps := kmhr_to_mps relative_speed_kmh,
      crossing_time := total_distance / relative_speed_mps
  in
    crossing_time ≈ 10.44 :=
by
  sorry

end trains_crossing_time_l179_179017


namespace exponent_of_5_in_30_factorial_l179_179969

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l179_179969


namespace sin_45_deg_eq_one_div_sqrt_two_l179_179237

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l179_179237


namespace find_m_eq_5_l179_179592

-- Definitions for the problem conditions
def f (x m : ℝ) := 2 * x + m

theorem find_m_eq_5 (m : ℝ) (a b : ℝ) :
  (a = f 0 m) ∧ (b = f m m) ∧ ((b - a) = (m - 0 + 5)) → m = 5 :=
by
  sorry

end find_m_eq_5_l179_179592
