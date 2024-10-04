import Data.Real.Basic
import Mathlib
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Analysis.Calculus.Area
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.Calculus.LinearAlgebra
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.Convex.Function
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Denumerable
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Rat
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Geometry.Euclidean.Circumradius
import Mathlib.Init.Data.Real.Basic
import Mathlib.Integration
import Mathlib.LinearAlgebra.Matrix
import Mathlib.NumberTheory.ArithmeticFunctions
import Mathlib.Order.Monotone
import Mathlib.Probability.Basic
import Mathlib.Probability.Classical
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Algebra.Polynomial
import Mathlib.Topology.Basic
import Mathlib.Topology.MetricSpace.Basic

namespace percent_black_population_midwest_l518_518372

theorem percent_black_population_midwest :
  let NE := 6
  let MW := 12
  let South := 18
  let West := 3
  let total_black := NE + MW + South + West
  MW / total_black * 100 = 31 :=
by
  let NE := 6
  let MW := 12
  let South := 18
  let West := 3
  let total_black := NE + MW + South + West
  calc
    MW / total_black * 100
    = 12 / 39 * 100 : by sorry
    ... = 31 : by sorry

end percent_black_population_midwest_l518_518372


namespace moles_of_H2O_formed_l518_518828

-- Define the initial conditions
def molesNaOH : ℕ := 2
def molesHCl : ℕ := 2

-- Balanced chemical equation behavior definition
def reaction (x y : ℕ) : ℕ := min x y

-- Statement of the problem to prove
theorem moles_of_H2O_formed :
  reaction molesNaOH molesHCl = 2 := by
  sorry

end moles_of_H2O_formed_l518_518828


namespace distribute_balls_in_boxes_l518_518510

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l518_518510


namespace find_supplementary_angle_l518_518369

def A := 45
def supplementary_angle (A S : ℕ) := A + S = 180
def complementary_angle (A C : ℕ) := A + C = 90
def thrice_complementary (S C : ℕ) := S = 3 * C

theorem find_supplementary_angle : 
  ∀ (A S C : ℕ), 
    A = 45 → 
    supplementary_angle A S →
    complementary_angle A C →
    thrice_complementary S C → 
    S = 135 :=
by
  intros A S C hA hSupp hComp hThrice
  have h1 : A = 45 := by assumption
  have h2 : A + S = 180 := by assumption
  have h3 : A + C = 90 := by assumption
  have h4 : S = 3 * C := by assumption
  sorry

end find_supplementary_angle_l518_518369


namespace part_a_first_part_a_second_l518_518740

open Real

-- Define the setup using the given conditions
variables {C A B P PA₁ PB₁ PC₁ : Point} (hTangencyA : Tangent A P) (hTangencyB : Tangent B P) (hAngleVertexC : Vertex C)
variables (hPerpPA₁ : Perpendicular PA₁ (Line P B C)) (hPerpPB₁ : Perpendicular PB₁ (Line P C A)) (hPerpPC₁ : Perpendicular PC₁ (Line P A B))

-- State the proof goals:
theorem part_a_first (hSameCircle: OnCircle P A B)
: (PC₁^2 = PA₁ * PB₁) := sorry

theorem part_a_second (hSameCircle: OnCircle P A B)
: (PA₁ / PB₁ = PA^2 / PB^2) := sorry

end part_a_first_part_a_second_l518_518740


namespace a_k_bound_l518_518857

variable {a : ℕ → ℝ}

axiom h1 : ∀ k, a k - 2 * a (k + 1) + a (k + 2) ≥ 0
axiom h2 : ∀ k, ∑ i in Finset.range k.succ, a i ≤ 1

theorem a_k_bound (k : ℕ) : 0 ≤ a k - a (k + 1) ∧ a k - a (k + 1) < 2 / (k : ℝ)^2 :=
by
  sorry

end a_k_bound_l518_518857


namespace james_baked_multiple_l518_518793

theorem james_baked_multiple (x : ℕ) (h1 : 115 ≠ 0) (h2 : 1380 = 115 * x) : x = 12 :=
sorry

end james_baked_multiple_l518_518793


namespace part_c_part_b_l518_518109

-- Definitions
def f : ℝ → ℝ := sorry
def g (x : ℝ) := (f' x)

-- Conditions
axiom dom_f : ∀ x : ℝ, x ∈ domain f
axiom dom_g : ∀ x : ℝ, x ∈ domain g
axiom even_f : ∀ x : ℝ, f (3 / 2 - 2 * x) = f (3 / 2 + 2 * x)
axiom even_g : ∀ x : ℝ, g (2 + x) = g (2 - x)

-- Theorems to prove
theorem part_c : f (-1) = f 4 := by sorry
theorem part_b : g (-1 / 2) = 0 := by sorry

end part_c_part_b_l518_518109


namespace equilateral_triangle_area_ratio_l518_518301

theorem equilateral_triangle_area_ratio (ABC : Triangle) (h : is_equilateral ABC) :
  let ratios := { r : ℚ | ∃ e : Line, passes_through_centroid ABC e ∧ divides_triangle ABC e r} in
  lower_bound ratios = 4 / 5 ∧ upper_bound ratios = 5 / 4 :=
sorry

end equilateral_triangle_area_ratio_l518_518301


namespace measure_XP_Y_given_conditions_l518_518177

-- Definitions of the problem components
def angle (A B C : Type) : Type := sorry -- Remains to be defined

noncomputable def PX_tangent_semicircle_RUX : Prop := sorry
noncomputable def PY_tangent_semicircle_TZV : Prop := sorry
noncomputable def RTV_straight_line : Prop := sorry
noncomputable def arc_RU_70_deg : Prop := sorry
noncomputable def arc_VZ_45_deg : Prop := sorry

-- Restating the problem in terms of Lean
theorem measure_XP_Y_given_conditions :
  PX_tangent_semicircle_RUX →
  PY_tangent_semicircle_TZV →
  RTV_straight_line →
  arc_RU_70_deg →
  arc_VZ_45_deg →
  angle XP Y = 115 :=
by
  intros
  sorry

end measure_XP_Y_given_conditions_l518_518177


namespace sphere_radius_l518_518765

theorem sphere_radius
    (shadow_sphere : ℝ)
    (cone_height : ℝ)
    (shadow_cone : ℝ)
    (tan_theta_cone : cone_height / shadow_cone = 3 / 5) :
    (shadow_sphere * 3 / 5 = 12) :=
by {
    -- define the conditions
    let radius_sphere := 20 * (3 / 5),
    -- prove the radius is 12 meters
    have h : radius_sphere = 12,
    from calc
        radius_sphere = 20 * (3 / 5) : by sorry
        ... = 12 : by sorry,
    exact h,
}

end sphere_radius_l518_518765


namespace geometric_sequence_general_term_l518_518609

theorem geometric_sequence_general_term (a₁ a_n S_n : ℕ → ℕ) (b_n : ℕ → ℚ) (c_n : ℕ → ℝ) (n : ℕ) :
  (∀ n, a_n n = a₁ * 2 ^ (n - 1)) → 
  (∀ n, S_n n = a₁ * (2 ^ n - 1)) →
  (∀ n, b_n n = (S_n n + 1) / (a_n n)) →
  (∀ n, b_n n = b_n (n + 1)) →
  (∀ n, c_n n = Real.log (a_n n) / Real.log 2) →
  (∀ n, a_n n = 2 ^ (n - 1) ∧ 
         sum (λ k, (c_n k / a_n k)) (range n) = 2 - (n + 1) / 2 ^ (n - 1)) :=
by sorry

end geometric_sequence_general_term_l518_518609


namespace evaluate_fraction_expression_l518_518035

theorem evaluate_fraction_expression :
  ( (1 / 5 - 1 / 6) / (1 / 3 - 1 / 4) ) = 2 / 5 :=
by
  sorry

end evaluate_fraction_expression_l518_518035


namespace angle_XDE_eq_angle_EDY_l518_518968

theorem angle_XDE_eq_angle_EDY
    (ABC : Triangle)
    (h_eq : ABC.AB = ABC.AC)
    (circumscribed : ABC.inscribedInCircle ω)
    (D_on_BC : D ∈ line(ABC.B, ABC.C))
    (BD_neq_DC : ABC.BD ≠ ABC.DC)
    (AD_intersects_E : A ∣ ω ∧ AD ∩ ω = E ∧ E ≠ A)
    (F_on_omega : F ∣ ω ∧ F ≠ E)
    (angle_DFE_90 : ∡[D, F, E] = 90°)
    (FE_intersects_AB_at_X : X ∈ ray(ABC.AB) ∧ line(F, E) ∩ ray(ABC.AB) = X)
    (FE_intersects_AC_at_Y : Y ∈ ray(ABC.AC) ∧ line(F, E) ∩ ray(ABC.AC) = Y) :
        ∠[X, D, E] = ∠[E, D, Y] :=
sorry

end angle_XDE_eq_angle_EDY_l518_518968


namespace sufficient_condition_for_inequality_l518_518821

theorem sufficient_condition_for_inequality (m : ℝ) : (m ≥ 2) → (∀ x : ℝ, x^2 - 2 * x + m ≥ 0) :=
by
  sorry

end sufficient_condition_for_inequality_l518_518821


namespace train_speed_144_kmph_l518_518755

theorem train_speed_144_kmph (d : ℝ) (t : ℝ) (h_d : d = 400) (h_t : t = 10) :
  (d / 1000) / (t / 3600) = 144 := 
by
  -- assumptions
  rw [h_d, h_t]
  have : (400/1000) / (10/3600) = 0.4 / (10/3600) := by sorry
  have : 0.4 / (10 / 3600) = 0.4 / (1 / 360) := by sorry
  have : 0.4 / (1 / 360) = 0.4 * 360 := by sorry
  have : 0.4 * 360 = 144 := by sorry
  rw this

end train_speed_144_kmph_l518_518755


namespace part_a_part_b_l518_518593

variables {α : Type} [normed_group α]

-- Definitions and assumptions
variables {A B C A' B' C' : α} -- vertices of the triangle and feet of perpendiculars
variables {R r d x : ℝ} -- radius of circumcircle, radius of incircle, distance OI, and given length

-- Hypotheses for part (a)
variables (hABC_sim_A'B'C' : similar (triangle A B C) (triangle A' B' C')) (hx_eq : A'.dist C = x) 

theorem part_a (h_sim : similar (triangle A B C) (triangle A' B' C')) (hcircum : ∀ x, circumradius A B C = R) (hincircum : ∀ x, inradius A B C = r) :
  A'.dist C = 2 * inradius A B C := sorry

-- Hypotheses for part (b)
variables (hA'_B'_C'_collinear : collinear (set_univ) {A', B', C'}) (_dist : ∀ ABC, dist (A,B,C) = x)

theorem part_b (hB'C_collinear : collinear (set {A', B', C'}) A B C) (hcircum : ∀ x, circumradius A B C = R) (hOI : distance (circumcenter A B C) (incenter A B C) = d) :
  (A'.dist C = circumradius A B C + dist O I) ∨ (A'.dist C = circumradius A B C - dist O I) := sorry

end part_a_part_b_l518_518593


namespace range_of_function_l518_518696

theorem range_of_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = (1/3)^(x^2 - 1) ↔ y ∈ set.Ioc 0 3 :=
sorry

end range_of_function_l518_518696


namespace length_AB_l518_518571

-- Define the vertices
variables {A B C D: Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Define the lengths of the sides of the triangles
variables {BD AC BC AB CD : ℝ}

-- Given conditions
def is_isosceles (ABC : Triangle) : Prop := AC = BC
def is_isosceles' (CBD : Triangle) : Prop := CD = BC
def perimeter_CBD (CBD : Triangle) : Prop := BD + BC + CD = 24
def perimeter_ABC (ABC : Triangle) : Prop := AB + AC + BC = 25
def length_BD : Prop := BD = 10

-- The proof statement
theorem length_AB {ABC CBD: Triangle} 
  (h_isosceles_ABC : is_isosceles ABC) 
  (h_isosceles_CBD : is_isosceles' CBD) 
  (h_perimeter_CBD: perimeter_CBD CBD)
  (h_perimeter_ABC: perimeter_ABC ABC)
  (h_length_BD: length_BD) 
  : AB = 11 := 
sorry

end length_AB_l518_518571


namespace ball_box_problem_l518_518496

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l518_518496


namespace traffic_light_probability_l518_518797

theorem traffic_light_probability :
  let P_A := 25 / 60,
      P_B := 35 / 60,
      P_C := 45 / 60 in
  (P_A * P_B * P_C) = 35 / 192 :=
by
  let P_A := 25 / 60,
  let P_B := 35 / 60,
  let P_C := 45 / 60,
  have h : P_A * P_B * P_C = 35 / 192 := sorry,
  exact h

end traffic_light_probability_l518_518797


namespace odd_divisor_probability_of_factorial_25_l518_518269

theorem odd_divisor_probability_of_factorial_25 : 
  let n := 25!
  let total_factors := 23 * 11 * 7 * 4 * 3 * 2 * 2 * 2 * 2 
  let odd_factors := 11 * 7 * 4 * 3 * 2 * 2 * 2 * 2 
  (odd_factors / total_factors) = (1 / 23) :=
by
  sorry

end odd_divisor_probability_of_factorial_25_l518_518269


namespace BPB1Q_is_parallelogram_l518_518905

-- Given conditions
variable {α : Type*} [EuclideanGeometry α]

variable (N A B C K M Q P B1 : Point α)
variable (innerCircle outerCircle : Circle α)

-- Conditions based on the problem description
axiom tangent_inner_outer : innerCircle.tangent outerCircle N
axiom touch_inner_K : innerCircle.tangentAt K (Segment B A)
axiom touch_inner_M : innerCircle.tangentAt M (Segment B C)
axiom mid_arc_AB : outerCircle.midpoint_arc_not_in N (Segment B A) Q
axiom mid_arc_BC : outerCircle.midpoint_arc_not_in N (Segment B C) P
axiom circumcircle_BQK : (Triangle B Q K).circumcircle.exists B1
axiom circumcircle_BPM : (Triangle B P M).circumcircle.exists B1

-- Theorem to be proved
theorem BPB1Q_is_parallelogram :
  Parallelogram (Segment B P) (Segment B1 Q) :=
sorry

end BPB1Q_is_parallelogram_l518_518905


namespace exists_n_coprime_sum_of_primes_l518_518219

def sum_of_primes (n : ℕ) : ℕ := (Finset.filter Nat.prime (Finset.range n)).sum

theorem exists_n_coprime_sum_of_primes :
  ∃ n : ℕ, n > 10 ^ 2018 ∧ Nat.coprime (sum_of_primes n) n :=
by
  sorry

end exists_n_coprime_sum_of_primes_l518_518219


namespace functional_eq_solution_l518_518044

theorem functional_eq_solution (f : ℝ → ℝ) :
  (∀ a b : ℝ, f (a * f b) = a * b) →
  (f = fun x => x ∨ f = fun x => -x) :=
by
  sorry

end functional_eq_solution_l518_518044


namespace coprime_sum_primes_l518_518221

def S (n : ℕ) : ℕ :=
  (Finset.filter Nat.prime (Finset.range n)).sum id

theorem coprime_sum_primes (n : ℕ) (h : 10^2018 < n) : ∃ m : ℕ, 10^2018 < m ∧ Nat.coprime (S m) m :=
by
  sorry

end coprime_sum_primes_l518_518221


namespace part_one_part_two_part_three_l518_518204

-- Part 1: The set of possible values for a
theorem part_one (a : ℝ) (h₁ : ∀ x, ax^2 + x - 1 = 0 → x = 1) (h₂ : b = 1) :
  a ∈ ({-1/4, 0} : set ℝ) :=
sorry

-- Part 2: Solve the inequality with respect to x
theorem part_two (a b : ℝ) :
  ∀ x, ax^2 + x - b < (a-1)x^2 + (b+2)x - 2b ↔
  (b < 1 ∧ b < x ∧ x < 1) ∨
  (b = 1 ∧ false) ∨
  (b > 1 ∧ 1 < x ∧ x < b) :=
sorry

-- Part 3: Prove the maximum value of 1/a - 1/b
theorem part_three (a b : ℝ) (h₁ : a > 0) (h₂ : b > 1) (t : ℝ) (h₃ : t > 0)
  (h₄ : ∀ x, ax^2 + x - b > 0 → -2 - t < x ∧ x < -2 + t) :
  (1/a - 1/b) ≤ 1/2 :=
sorry

end part_one_part_two_part_three_l518_518204


namespace intersecting_absolute_value_functions_l518_518897

theorem intersecting_absolute_value_functions (a b c d : ℝ) (h1 : -|2 - a| + b = 5) (h2 : -|8 - a| + b = 3) (h3 : |2 - c| + d = 5) (h4 : |8 - c| + d = 3) (ha : 2 < a) (h8a : a < 8) (hc : 2 < c) (h8c : c < 8) : a + c = 10 :=
sorry

end intersecting_absolute_value_functions_l518_518897


namespace area_of_square_is_26_l518_518303

open Real

def P := (2 : ℝ, 3 : ℝ)
def Q := (-3 : ℝ, 4 : ℝ)
def R := (-2 : ℝ, -1 : ℝ)
def S := (3 : ℝ, 0 : ℝ)

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

noncomputable def side_length : ℝ :=
  distance P Q

noncomputable def area_of_square : ℝ :=
  side_length ^ 2

theorem area_of_square_is_26 :
  area_of_square = 26 := by
  sorry

end area_of_square_is_26_l518_518303


namespace marble_weight_l518_518975

noncomputable def base_length := 2
noncomputable def base_width := 2
noncomputable def height := 8
noncomputable def density := 2700
noncomputable def base_area := base_length * base_width
noncomputable def volume := base_area * height
noncomputable def weight := density * volume

theorem marble_weight :
  weight = 86400 :=
by
  -- Skipping the actual proof steps
  sorry

end marble_weight_l518_518975


namespace pen_cost_is_4_l518_518348

variable (penCost pencilCost : ℝ)

-- Conditions
def totalCost := penCost + pencilCost = 6
def costRelation := penCost = 2 * pencilCost

-- Theorem to be proved
theorem pen_cost_is_4 (h1 : totalCost) (h2 : costRelation) : penCost = 4 :=
by
  rw [totalCost, costRelation] at h1
  sorry

end pen_cost_is_4_l518_518348


namespace drew_got_wrong_19_l518_518164

theorem drew_got_wrong_19 :
  ∃ (D_wrong C_wrong : ℕ), 
    (20 + D_wrong = 52) ∧
    (14 + C_wrong = 52) ∧
    (C_wrong = 2 * D_wrong) ∧
    D_wrong = 19 :=
by
  sorry

end drew_got_wrong_19_l518_518164


namespace cube_root_of_8_l518_518250

theorem cube_root_of_8 : ∃ (x : ℝ), x^3 = 8 ∧ x = 2 :=
by {
  use 2,
  split,
  { norm_num },  -- Prove 2^3 = 8
  { refl }      -- Prove x = 2
}

end cube_root_of_8_l518_518250


namespace cosine_angle_BKD_l518_518162

noncomputable def rectangular_solid (D K G F B : Type) (DK KG : ℝ) (angleDKG angleFKB : ℝ) : Prop :=
  angleDKG = 60 ∧ DK = KG ∧ angleFKB = 45
  
-- Prove that the cosine of ∠BKD is √2/2 given the specified conditions
theorem cosine_angle_BKD (D K G F B : Type) (DK KG : ℝ) (angleDKG angleFKB : ℝ) 
  (h : rectangular_solid D K G F B DK KG angleDKG angleFKB) :
  (cos (angle_BKD D K G F B DK KG angleDKG angleFKB) = √2 / 2) :=
sorry

end cosine_angle_BKD_l518_518162


namespace eds_weight_l518_518362

variable (Al Ben Carl Ed : ℕ)

def weight_conditions : Prop :=
  Carl = 175 ∧ Ben = Carl - 16 ∧ Al = Ben + 25 ∧ Ed = Al - 38

theorem eds_weight (h : weight_conditions Al Ben Carl Ed) : Ed = 146 :=
by
  -- Conditions
  have h1 : Carl = 175    := h.1
  have h2 : Ben = Carl - 16 := h.2.1
  have h3 : Al = Ben + 25   := h.2.2.1
  have h4 : Ed = Al - 38    := h.2.2.2
  -- Proof itself is omitted, sorry placeholder
  sorry

end eds_weight_l518_518362


namespace evaluate_m_l518_518819

theorem evaluate_m (m : ℕ) : 2 ^ m = (64 : ℝ) ^ (1 / 3) → m = 2 :=
by
  sorry

end evaluate_m_l518_518819


namespace solution_set_l518_518429

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set (f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2))
  (hA : f 0 = -1) (hB : f 3 = 1) :
  { x : ℝ | |f(x + 1)| < 1 } = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_l518_518429


namespace reception_desks_arrangement_l518_518244

theorem reception_desks_arrangement :
  let workers := {A, B, C, D, E, F}
  let desks := {desk1, desk2}
  (∃ (d1 d2 : set string), 
    d1 ∪ d2 = workers ∧ d1 ≠ d2 ∧ d1.card ≥ 2 ∧ d2.card ≥ 2 ∧ 
    A ∈ d1 ∧ B ∈ d2) ∨ 
  (∃ (d1 d2 : set string), 
    d1 ∪ d2 = workers ∧ d1 ≠ d2 ∧ d1.card ≥ 2 ∧ d2.card ≥ 2 ∧
    A ∈ d2 ∧ B ∈ d1) ∧ 
  (d1 ∩ d2 = ∅) →
  28 :=
by
  sorry

end reception_desks_arrangement_l518_518244


namespace value_fraction_eq_three_l518_518845

namespace Problem

variable {R : Type} [Field R]

theorem value_fraction_eq_three (a b c : R) (h : a / 2 = b / 3 ∧ b / 3 = c / 4) :
  (a + b + c) / (2 * a + b - c) = 3 := by
  sorry

end Problem

end value_fraction_eq_three_l518_518845


namespace number_ways_one_ball_correct_number_ways_unlimited_balls_correct_l518_518803

-- Defining the problems: Number of ways to place k identical balls into n urns with the respective conditions.

variable (n k : ℕ)

-- Problem 1: At most one ball in each urn.
def number_ways_one_ball : ℕ := (nat.choose n k)

theorem number_ways_one_ball_correct :
  number_ways_one_ball n k = nat.choose n k := by
  sorry

-- Problem 2: Unlimited number of balls in each urn.
def number_ways_unlimited_balls : ℕ := (nat.choose (n + k - 1) (n - 1))

theorem number_ways_unlimited_balls_correct :
  number_ways_unlimited_balls n k = nat.choose (n + k - 1) (n - 1) := by
  sorry

end number_ways_one_ball_correct_number_ways_unlimited_balls_correct_l518_518803


namespace distribute_balls_in_boxes_l518_518514

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l518_518514


namespace Angie_necessities_amount_l518_518370

noncomputable def Angie_salary : ℕ := 80
noncomputable def Angie_left_over : ℕ := 18
noncomputable def Angie_taxes : ℕ := 20
noncomputable def Angie_expenses : ℕ := Angie_salary - Angie_left_over
noncomputable def Angie_necessities : ℕ := Angie_expenses - Angie_taxes

theorem Angie_necessities_amount :
  Angie_necessities = 42 :=
by
  unfold Angie_necessities
  unfold Angie_expenses
  sorry

end Angie_necessities_amount_l518_518370


namespace four_f_one_greater_f_two_l518_518871

variables {f : ℝ → ℝ}

-- Assumptions
axiom f_double_prime_defined : ∀ x : ℝ, x > 0 → differentiable_at ℝ (f'') x
axiom inequality_holds : ∀ x : ℝ, x > 0 → x * f'' x < 2 * f x

-- Main statement to prove
theorem four_f_one_greater_f_two : 4 * f 1 > f 2 :=
by
  sorry

end four_f_one_greater_f_two_l518_518871


namespace sum_of_counts_2015_2016_l518_518580

theorem sum_of_counts_2015_2016 :
  ∀ (initial_circle : list ℕ) 
    (opposite_positions : list (ℕ × ℕ))
    (operation : list ℕ → list ℕ)
    (sufficient_operations : list ℕ → list ℕ),
  (initial_circle = [1, 2])
  ∧ (∀ positions, positions ∈ opposite_positions → (positions = (1, 2) ∨ positions = (2, 1)))
  ∧ (operation = λ circle, 
    list.map (λ (pair : ℕ × ℕ), pair.fst + pair.snd)
      (list.zip circle (circle.rotate 1)))
  ∧ (sufficient_operations = λ circle, nat.iterate operation circle 2015)
  → (∑ i in sufficient_operations [1, 2], i) = 2016 := 
sorry

end sum_of_counts_2015_2016_l518_518580


namespace max_sqrt_expr_l518_518415

theorem max_sqrt_expr (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 23) :
  sqrt (x + 35) + sqrt (23 - x) + 2 * sqrt x ≤ 15 :=
sorry

end max_sqrt_expr_l518_518415


namespace bug_traverses_36_tiles_l518_518382

-- Define the dimensions of the rectangle and the bug's problem setup
def width : ℕ := 12
def length : ℕ := 25

-- Define the function to calculate the number of tiles traversed by the bug
def tiles_traversed (w l : ℕ) : ℕ :=
  w + l - Nat.gcd w l

-- Prove the number of tiles traversed by the bug is 36
theorem bug_traverses_36_tiles : tiles_traversed width length = 36 :=
by
  -- This part will be proven; currently, we add sorry
  sorry

end bug_traverses_36_tiles_l518_518382


namespace arithmetic_sequence_max_sum_l518_518094

theorem arithmetic_sequence_max_sum :
  ∃ n : ℕ, (n = 4) ∧ 
  (∀ k : ℕ, S_k = (k / 2) * (2 * a₁ + (k - 1) * d)) ∧
  (S_n ≤ S_n) ∧ (S_4 = 26) :=
sorry

end arithmetic_sequence_max_sum_l518_518094


namespace string_length_correct_l518_518336

-- Let h be the height of the post
def h : ℝ := 18

-- Let c be the circumference of the post
def c : ℝ := 6

-- Let n be the number of loops
def n : ℕ := 6

-- Let vertical_step be the height covered by one loop
def vertical_step : ℝ := h / n

-- Let horizontal_step be the horizontal distance covered by one loop, which equals the circumference
def horizontal_step : ℝ := c

-- Let loop_length be the length of the string for one loop
def loop_length : ℝ := real.sqrt (vertical_step^2 + horizontal_step^2)

-- Let total_length be the total length of the string
def total_length : ℝ := n * loop_length

theorem string_length_correct : total_length = 18 * real.sqrt 5 := 
by
  sorry

end string_length_correct_l518_518336


namespace quadratic_inequality_solution_set_l518_518150

theorem quadratic_inequality_solution_set (a b c : ℝ) (h1 : a < 0)
  (h2 : -1 + 2 = b / a) (h3 : -1 * 2 = c / a) :
  (b = a) ∧ (c = -2 * a) :=
by
  sorry

end quadratic_inequality_solution_set_l518_518150


namespace island_liars_l518_518649

theorem island_liars (n : ℕ) (h₁ : n = 450) (h₂ : ∀ (i : ℕ), i < 450 → 
  ∃ (a : bool),  (if a then (i + 1) % 450 else (i + 2) % 450) = "liar"):
    (n = 150 ∨ n = 450) :=
sorry

end island_liars_l518_518649


namespace find_integer_l518_518547

-- Define f as a function from integers to integers
noncomputable def f : ℤ → ℤ
| 0       := 15
| (n + 1) := f n - (n + 1)

-- State the given conditions
variables (k : ℤ) (h1 : f 6 = 4)

-- State the theorem to prove 
theorem find_integer (k : ℤ) (h1 : f 6 = 4) : ∃ k, f k = 15 :=
  sorry

end find_integer_l518_518547


namespace cos_theta_value_projection_value_l518_518911

noncomputable def vec_a : (ℝ × ℝ) := (3, 1)
noncomputable def vec_b : (ℝ × ℝ) := (-2, 4)

theorem cos_theta_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / (magnitude_a * magnitude_b) = - Real.sqrt 2 / 10 :=
by 
  sorry

theorem projection_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  cos_theta = - Real.sqrt 2 / 10 →
  magnitude_a * cos_theta = - Real.sqrt 5 / 5 :=
by 
  sorry

end cos_theta_value_projection_value_l518_518911


namespace alex_blueberry_pies_l518_518363

-- Definitions based on given conditions:
def total_pies : ℕ := 30
def ratio (a b c : ℕ) : Prop := (a : ℚ) / b = 2 / 3 ∧ (b : ℚ) / c = 3 / 5

-- Statement to prove the number of blueberry pies
theorem alex_blueberry_pies :
  ∃ (a b c : ℕ), ratio a b c ∧ a + b + c = total_pies ∧ b = 9 :=
by
  sorry

end alex_blueberry_pies_l518_518363


namespace exists_n_coprime_sum_of_primes_l518_518220

def sum_of_primes (n : ℕ) : ℕ := (Finset.filter Nat.prime (Finset.range n)).sum

theorem exists_n_coprime_sum_of_primes :
  ∃ n : ℕ, n > 10 ^ 2018 ∧ Nat.coprime (sum_of_primes n) n :=
by
  sorry

end exists_n_coprime_sum_of_primes_l518_518220


namespace even_function_implications_l518_518107

variables {R : Type*} [LinearOrderedField R]

/-- Given conditions
  * The function f(x) and its derivative f'(x) both have a domain of ℝ.
  * Let g(x) = f'(x).
  * f(3/2 - 2x) is an even function.
  * g(2 + x) is an even function.
-/
def condition (f : R → R) (g : R → R) (h : ∀ x, g x = f' x) : Prop :=
  (∀ x, f (3 / 2 - 2 * x) = f (3 / 2 + 2 * x)) ∧
  (∀ x, g (2 + x) = g (2 - x))

-- Main theorem statement
theorem even_function_implications (f : R → R) (g : R → R) (h : ∀ x, g x = f' x) 
  (H : condition f g h) : 
  g (-1 / 2) = 0 ∧ f (-1) = f 4 := 
  sorry

end even_function_implications_l518_518107


namespace midpoint_of_F_on_AB_l518_518980

-- Definitions
variables {A B C D E F : Type}
variables [abc_isosceles : isosceles_triangle A B C C]
variables (D_on_AC : D ∈ segment A C)
variables (E_on_BC : E ∈ segment B C)
variables (F_on_AB : F ∈ segment A B)
variables (angle_bisectors : bisector (∠ D E B) (∠ A D E) F)

-- Hypothesis and goal
theorem midpoint_of_F_on_AB :
  is_midpoint F A B :=
begin
  sorry
end

end midpoint_of_F_on_AB_l518_518980


namespace service_provider_ways_l518_518977

theorem service_provider_ways:
  let children := 4
  let providers := 25
  (finset.range providers).card.choose children = 4 :=
  25 * 24 * 23 * 22 = 303600 := by
  sorry

end service_provider_ways_l518_518977


namespace koi_fish_in_pond_l518_518328

theorem koi_fish_in_pond : ∃ k : ℤ, 2 * k - 14 = 64 ∧ k = 39 := by
  use 39
  split
  · sorry
  · rfl

end koi_fish_in_pond_l518_518328


namespace factorize_quadratic_find_m_find_perimeter_l518_518224

-- Part 1: Factorization problem
theorem factorize_quadratic (a : ℝ) : a^2 - 6 * a + 5 = (a - 1) * (a - 5) :=
by sorry

-- Part 2: Solving for m and finding perimeter

-- Assuming the given equation
def equation_condition (a b m c : ℝ) :=
  a^2 + b^2 - 12 * a - 6 * b + 45 + |(1/2) * m - c| = 0

-- Condition requirement for a, b, m
def condition_on_abm (a b m : ℝ) := 
  2^a * 4^b = 8^m

-- Proving the value of m
theorem find_m (a b m : ℝ) (h : condition_on_abm a b m) : m = 4 :=
by sorry

-- Triange perimeter. Assumes a = 6, b = 3, c is odd
def perimeter_ABC (a b c : ℝ) : Prop :=
  c % 2 = 1 → ∃ (p : ℝ), p = a + b + c

-- Proving the perimeter is 14 or 16 for odd c
theorem find_perimeter (a b : ℝ) (ha : a = 6) (hb : b = 3) (h : ∃ (c : ℝ), (perimeter_ABC a b c)) : ∃ (p : ℝ), p = 14 ∨ p = 16 :=
by sorry

end factorize_quadratic_find_m_find_perimeter_l518_518224


namespace koi_fish_in_pond_l518_518327

theorem koi_fish_in_pond:
  ∃ k : ℕ, 2 * k - 14 = 64 ∧ k = 39 := sorry

end koi_fish_in_pond_l518_518327


namespace ellipse_hyperbola_proof_l518_518859

noncomputable def ellipse_and_hyperbola_condition (a b : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧ (a^2 - b^2 = 5) ∧ (a^2 = 11 * b^2)

theorem ellipse_hyperbola_proof : 
  ∀ (a b : ℝ), ellipse_and_hyperbola_condition a b → b^2 = 0.5 :=
by
  intros a b h
  sorry

end ellipse_hyperbola_proof_l518_518859


namespace find_x_l518_518729

theorem find_x :
  ∃ x : ℕ, 3005 - x + 10 = 2705 ∧ x = 310 :=
by {
  use 310,
  split,
  {
    /- This part is proving that 3005 - 310 + 10 = 2705 -/
    norm_num,
  },
  {
    /- This part is proving that the x used is indeed 310 -/
    refl,
  }
}

end find_x_l518_518729


namespace marbles_remaining_correct_l518_518809

-- Define the number of marbles Chris has
def marbles_chris : ℕ := 12

-- Define the number of marbles Ryan has
def marbles_ryan : ℕ := 28

-- Define the total number of marbles in the pile
def total_marbles : ℕ := marbles_chris + marbles_ryan

-- Define the number of marbles each person takes away from the pile
def marbles_taken_each : ℕ := total_marbles / 4

-- Define the total number of marbles taken away
def total_marbles_taken : ℕ := 2 * marbles_taken_each

-- Define the number of marbles remaining in the pile
def marbles_remaining : ℕ := total_marbles - total_marbles_taken

theorem marbles_remaining_correct : marbles_remaining = 20 := by
  sorry

end marbles_remaining_correct_l518_518809


namespace domain_of_sqrt_log5_sin_l518_518253

theorem domain_of_sqrt_log5_sin (x : ℝ) :
  (-π / 2 ≤ x ∧ x ≤ π / 2) →
  (1 - 2 * Real.sin x > 0) →
  (Real.log 5 (1 - 2 * Real.sin x) ≥ 0) →
  (-π / 2 ≤ x ∧ x ≤ 0) :=
by
  sorry

end domain_of_sqrt_log5_sin_l518_518253


namespace triangle_area_proof_l518_518684

noncomputable def cos_fun1 (x : ℝ) : ℝ := 2 * Real.cos (3 * x) + 1
noncomputable def cos_fun2 (x : ℝ) : ℝ := - Real.cos (2 * x)

theorem triangle_area_proof :
  let P := (5 * Real.pi, cos_fun1 (5 * Real.pi))
  let Q := (9 * Real.pi / 2, cos_fun2 (9 * Real.pi / 2))
  let m := (Q.snd - P.snd) / (Q.fst - P.fst)
  let y_intercept := P.snd - m * P.fst
  let y_intercept_point := (0, y_intercept)
  let x_intercept := -y_intercept / m
  let x_intercept_point := (x_intercept, 0)
  let base := x_intercept
  let height := y_intercept
  17 * Real.pi / 4 ≤ P.fst ∧ P.fst ≤ 21 * Real.pi / 4 ∧
  17 * Real.pi / 4 ≤ Q.fst ∧ Q.fst ≤ 21 * Real.pi / 4 ∧
  (P.fst = 5 * Real.pi ∧ Q.fst = 9 * Real.pi / 2) →
  1/2 * base * height = 361 * Real.pi / 8 :=
by
  sorry

end triangle_area_proof_l518_518684


namespace find_3087th_letter_l518_518815

def sequence : List Char := ['X', 'Y', 'Z', 'Z', 'Y', 'X', 'Z', 'X', 'Y']

def sequence_length : Nat := 9

theorem find_3087th_letter :
  let n := 3087
  let index := (n - 1) % sequence_length
  sequence.get? index = some 'Y' :=
by
  let n := 3087
  let index := (n - 1) % sequence_length
  show sequence.get? index = some 'Y'
  sorry

end find_3087th_letter_l518_518815


namespace ways_to_divide_8_friends_l518_518920

theorem ways_to_divide_8_friends : 
  ∃ (ways : ℕ), ways = ((3^8) - (3 * 2^8) + (3 * 1^8)) ∧ ways = 5796 :=
by 
  exists 5796
  sorry

end ways_to_divide_8_friends_l518_518920


namespace linear_equation_solution_l518_518143

theorem linear_equation_solution (a b : ℤ) (x y : ℤ) (h1 : x = 2) (h2 : y = -1) (h3 : a * x + b * y = -1) : 
  1 + 2 * a - b = 0 :=
by
  sorry

end linear_equation_solution_l518_518143


namespace sum_of_first_100_terms_l518_518114

noncomputable theory

open_locale classical

/-- Given conditions -/
variables {f : ℝ → ℝ} {a : ℕ → ℝ}
variable (d : ℝ)

-- Conditions
axiom fx_monotonic : ∀ x y : ℝ, (-1 < x ∧ x < y) → f(x) ≤ f(y)
axiom fy_symmetric : ∀ x : ℝ, f(x - 2) = f(-x - 2)
axiom a_arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d
axiom f_a_50_eq_f_a_51 : f (a 50) = f (a 51)
axiom d_nonzero : d ≠ 0

-- Main Theorem
theorem sum_of_first_100_terms :
  let s := (100 * (a 1 + a 100)) / 2 in
  s = -100 :=
begin
  sorry,
end

end sum_of_first_100_terms_l518_518114


namespace fraction_of_total_money_l518_518978

variable (Max Leevi Nolan Ollie : ℚ)

-- Condition: Each of Max, Leevi, and Nolan gave Ollie the same amount of money
variable (x : ℚ) (h1 : Max / 6 = x) (h2 : Leevi / 3 = x) (h3 : Nolan / 2 = x)

-- Proving that the fraction of the group's (Max, Leevi, Nolan, Ollie) total money possessed by Ollie is 3/11.
theorem fraction_of_total_money (h4 : Max + Leevi + Nolan + Ollie = Max + Leevi + Nolan + 3 * x) : 
  x / (Max + Leevi + Nolan + x) = 3 / 11 := 
by
  sorry

end fraction_of_total_money_l518_518978


namespace solution_of_sqrt_equation_l518_518241

theorem solution_of_sqrt_equation (m n : ℕ) (h_cond : (sqrt (7 + sqrt 48 : ℝ) = m + sqrt n)) : m^2 + n^2 = 13 :=
by
  sorry

end solution_of_sqrt_equation_l518_518241


namespace travel_time_without_paddles_l518_518076

variables (A B : Type) (v v_r S : ℝ)
noncomputable def time_to_travel (distance velocity : ℝ) := distance / velocity

-- Condition: The travel time from A to B is 3 times the travel time from B to A
axiom travel_condition : (time_to_travel S (v + v_r)) = 3 * (time_to_travel S (v - v_r))

-- Condition: We are considering travel from B to A by canoe without paddles
noncomputable def time_without_paddles := time_to_travel S v_r

-- Proving that without paddles it takes 3 times longer than usual (using canoes with paddles)
theorem travel_time_without_paddles :
  time_without_paddles S v_r = 3 * (time_to_travel S (v - v_r)) :=
sorry

end travel_time_without_paddles_l518_518076


namespace const_ratio_AB_MN_is_quarter_l518_518443

-- Define the ellipse and associated parameters
def a : ℝ := 2
def b : ℝ := Real.sqrt 3
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1

-- Statement of the proof problem
theorem const_ratio_AB_MN_is_quarter (x₁ y₁ x₂ y₂ k : ℝ) 
  (hA : ellipse x₁ y₁)
  (hB : ellipse x₂ y₂)
  (hF2 : x₂ = 1 ∨ (y₂ = k * (x₂ - 1)))
  (hMN_parallel_AB : (M N : ℝ × ℝ) 
    (hM : ellipse M.1 M.2)
    (hN : ellipse N.1 N.2)
    (MN_parallel_AB : (M.2 - N.2) / (M.1 - N.1) = (y₂ - y₁) / (x₂ - x₁))) : 
  ∃ m n : ℝ, (m * (1 + m^2)^0.5 / (n^2 * (1 + m^2)) = 1 / 4).
Proof
  sorry

end const_ratio_AB_MN_is_quarter_l518_518443


namespace cost_per_rose_l518_518584

theorem cost_per_rose (P : ℝ) (h1 : 5 * 12 = 60) (h2 : 0.8 * 60 * P = 288) : P = 6 :=
by
  -- Proof goes here
  sorry

end cost_per_rose_l518_518584


namespace even_function_implications_l518_518105

variables {R : Type*} [LinearOrderedField R]

/-- Given conditions
  * The function f(x) and its derivative f'(x) both have a domain of ℝ.
  * Let g(x) = f'(x).
  * f(3/2 - 2x) is an even function.
  * g(2 + x) is an even function.
-/
def condition (f : R → R) (g : R → R) (h : ∀ x, g x = f' x) : Prop :=
  (∀ x, f (3 / 2 - 2 * x) = f (3 / 2 + 2 * x)) ∧
  (∀ x, g (2 + x) = g (2 - x))

-- Main theorem statement
theorem even_function_implications (f : R → R) (g : R → R) (h : ∀ x, g x = f' x) 
  (H : condition f g h) : 
  g (-1 / 2) = 0 ∧ f (-1) = f 4 := 
  sorry

end even_function_implications_l518_518105


namespace exists_four_distinct_numbers_with_equal_half_sum_l518_518430

theorem exists_four_distinct_numbers_with_equal_half_sum (S : Finset ℕ) (h_card : S.card = 10) (h_range : ∀ x ∈ S, x ≤ 23) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S) ∧ (a + b = c + d) :=
by
  sorry

end exists_four_distinct_numbers_with_equal_half_sum_l518_518430


namespace sequence_general_formula_l518_518425

/--
A sequence a_n is defined such that the first term a_1 = 3 and the recursive formula 
a_{n+1} = (3 * a_n - 4) / (a_n - 2).

We aim to prove that the general term of the sequence is given by:
a_n = ( (-2)^(n+2) - 1 ) / ( (-2)^n - 1 )
-/
theorem sequence_general_formula (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 3)
  (hr : ∀ n, a (n + 1) = (3 * a n - 4) / (a n - 2)) :
  a n = ( (-2:ℝ)^(n+2) - 1 ) / ( (-2:ℝ)^n - 1) :=
sorry

end sequence_general_formula_l518_518425


namespace f_f_1_l518_518848

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x else log 2 (x - 1)

theorem f_f_1 : f (f 1) = 0 := by
  -- Proof goes here
  sorry

end f_f_1_l518_518848


namespace every_subsegment_contains_point_in_H_l518_518484

-- Definitions and conditions
variable {Point : Type}
variable (A B : Point)
variable (H : Set Point)
variable (on_segment : Point → Point → Point → Prop)
variable (dist : Point → Point → ℕ)

-- Conditions as definitions in Lean 4
def A_in_H : A ∈ H := sorry
def B_in_H : B ∈ H := sorry

-- Z condition
def Z_condition (X Y Z : Point) : Prop := 
  on_segment XY Z ∧ dist Y Z = 3 * dist X Z

def condition_b_allows (X Y : Point) (hX : X ∈ H) (hY : Y ∈ H) : { Z // Z_condition X Y Z } :=
  sorry -- nonconstructive definition that specifies there exists such a Z by axiom

-- The proof statement
theorem every_subsegment_contains_point_in_H :
  ∀ P Q : Point, on_segment AB P → on_segment AB Q → P ≠ Q →
  ∃ R : Point, on_segment PQ R ∧ R ∈ H :=
  sorry

end every_subsegment_contains_point_in_H_l518_518484


namespace sum_of_a_for_unique_solution_l518_518061

theorem sum_of_a_for_unique_solution (a : ℝ) (h : (a + 12)^2 - 384 = 0) : 
  let a1 := -12 + 16 * Real.sqrt 6
  let a2 := -12 - 16 * Real.sqrt 6
  a1 + a2 = -24 := 
by
  sorry

end sum_of_a_for_unique_solution_l518_518061


namespace proper_subset_with_even_count_l518_518902

def M : set ℕ := {2, 0, 11}

theorem proper_subset_with_even_count :
  ( {A : set ℕ // A ⊂ M ∧ ∃ x, x ∈ A ∧ (x % 2 = 0) }.to_finset.card = 5) :=
by
  sorry

end proper_subset_with_even_count_l518_518902


namespace exists_one_friend_l518_518028

open Finset

variable {E : Type} [DecidableEq E]

-- A finite set of people
variable {S : Finset E}

-- The number of friends function
variable (friends : E → Finset E)

-- Conditions
noncomputable def condition_1 := ∀ x y ∈ S, x ≠ y → friends x ≠ friends y → (friends x ∩ friends y) = ∅

theorem exists_one_friend (h1 : condition_1 friends S) (h2 : S.nonempty) :
  ∃ x ∈ S, (friends x).card = 1 :=
sorry

end exists_one_friend_l518_518028


namespace menu_count_l518_518777

-- Define the days of the week
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday

-- Dessert options
inductive Dessert
| Cookies | Pie | Jelly

-- Define the constraints and requirements
structure Menu :=
(dessert : Day → Dessert)
(no_consecutive : ∀ d, dessert d ≠ dessert (prev d))
(wednesday_pie : dessert Day.Wednesday = Dessert.Pie)

-- Define the previous day function to handle cyclic days
def prev : Day → Day
| Day.Monday => Day.Friday
| Day.Tuesday => Day.Monday
| Day.Wednesday => Day.Tuesday
| Day.Thursday => Day.Wednesday
| Day.Friday => Day.Thursday

-- Noncomputable function returning the count of valid menus, given the conditions
noncomputable def count_valid_menus : Nat :=
3 * 2 * 1 * 2 * 2 -- Compute based on conditions: 3 (Monday) * 2 (Tuesday) * 1 (Wednesday) * 2 (Thursday) * 2 (Friday)

-- Proof placeholder
theorem menu_count : count_valid_menus = 24 :=
by 
  sorry

end menu_count_l518_518777


namespace maximum_value_sin_theta_minus_pi_div_2_l518_518137

theorem maximum_value_sin_theta_minus_pi_div_2 :
  ∀ (m θ : ℝ), 
    m ∈ set.Icc (-1 : ℝ) 0 →
    θ ∈ set.Icc (-(real.pi / 2)) (real.pi / 2) →
    (1 + m) * m + (2 * real.cos θ - 4) * (-4) ≤ 10 →
    real.sin (θ - real.pi / 2) ≤ -real.sqrt 3 / 2 :=
begin
  intros m θ hm hθ hineq,
  -- steps to prove the theorem
  sorry
end

end maximum_value_sin_theta_minus_pi_div_2_l518_518137


namespace find_parabola_equation_l518_518052

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, 
    (y = a * x ^ 2 + b * x + c) ∧ 
    (y = (x - 3) ^ 2 - 2) ∧
    (a * (4 - 3) ^ 2 - 2 = 2)

theorem find_parabola_equation :
  ∃ (a b c : ℝ), parabola_equation a b c ∧ a = 4 ∧ b = -24 ∧ c = 34 :=
sorry

end find_parabola_equation_l518_518052


namespace original_population_l518_518565

def population_initial (P : ℝ) : Prop :=
  let after_other_factors := 0.8 * P
  let after_bombardment := 0.92 * after_other_factors
  let after_fear := 0.85 * after_bombardment
  let active_population := 0.9 * after_fear
  active_population = 3553

theorem original_population : ∃ P : ℝ, population_initial P ∧ P ≈ 6328 := 
sorry

end original_population_l518_518565


namespace cyclist_speed_north_l518_518293

theorem cyclist_speed_north (v : ℝ) :
  (∀ d t : ℝ, d = 50 ∧ t = 1 ∧ 40 * t + v * t = d) → v = 10 :=
by
  sorry

end cyclist_speed_north_l518_518293


namespace Max_S5_card_Max_Sn_card_l518_518994

def Sn (n : ℕ) := 
  Σ s : Finₙ -> ℤ, ∃ z : Finₙ -> ℤ, (∀ i, (s i) ∈ {0, 1}) ∧
  (∀ (x y : Finₙ -> ℤ), (x = y → ∑ i : Finₙ, x i * y i % 2 = 1) ∧ 
                          (x ≠ y → ∑ i : Finₙ, x i * y i % 2 = 0))

theorem Max_S5_card : ∀ (n : ℕ), n = 5 → ∃ S : Fin₅ -> ℑ, |S| ≤ 5 :=
  sorry

theorem Max_Sn_card : ∀ {n : ℕ} (h : n ≥ 6), 
  ∃ S : Finₙ -> ℤ, |S| = n :=
  sorry

end Max_S5_card_Max_Sn_card_l518_518994


namespace compute_M_l518_518995

open Matrix

variable {R : Type*} [Semiring R]
variable {n m : Type*} [Fintype n] [DecidableEq n] [Fintype m] [DecidableEq m]

def M : Matrix n m R := sorry
def u : m → R := sorry
def v : m → R := sorry
def w : m → R := sorry

axiom hM_u : M.mul_vec u = ![2, -2] 
axiom hM_v : M.mul_vec v = ![3, 1]
axiom hM_w : M.mul_vec w = ![-1, 4]

theorem compute_M (M : Matrix n m R) (u v w : m → R) :
  M.mul_vec (3 • u - v + 2 • w) = ![1, 1] :=
by
  sorry

end compute_M_l518_518995


namespace count_distinct_digits_in_range_2300_2500_l518_518492

open Finset

-- Define the condition: a 4-digit integer with distinct digits
def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.len = 4 ∧ digits.nodup

-- Total count of integers with distinct digits between 2300 and 2500.
theorem count_distinct_digits_in_range_2300_2500 : {n // n ∈ Icc 2300 2500 ∧ distinct_digits n}.card = 84 := 
by
  sorry

end count_distinct_digits_in_range_2300_2500_l518_518492


namespace men_can_complete_jobs_in_given_time_l518_518540

theorem men_can_complete_jobs_in_given_time (m n d k : ℕ) :
  let total_time := (m * d + n * k) / (m + n) in
  total_time * (m + n) = m * d + n * k :=
by sorry

end men_can_complete_jobs_in_given_time_l518_518540


namespace sum_of_digits_of_A_eq_959_l518_518831

noncomputable def A := 2^63 * 4^25 * 5^106 - 2^22 * 4^44 * 5^105 - 1

theorem sum_of_digits_of_A_eq_959 : 
  let digit_sum (n : Nat) : Nat := n.digits.sum
in digit_sum A = 959 :=
by 
  -- The proof steps would go here
  sorry

end sum_of_digits_of_A_eq_959_l518_518831


namespace equalize_milk_in_10_jars_l518_518038

theorem equalize_milk_in_10_jars (a : Fin 10 → ℝ) 
  (h : ∀ i, 0 ≤ a i ∧ a i ≤ 10) :
  ∃ n ≤ 10, ∀ (moves : Fin n → Fin 10 → ℝ → Fed 10 → Prop), 
    ∀ jar_amount : Fin n → Fin 10 → ℝ, 
    (∀ k : Fin n, move k jar_amount) →
    ∀ i j : Fin 10, jar_amount n i = jar_amount n j := 
sorry

end equalize_milk_in_10_jars_l518_518038


namespace koi_fish_in_pond_l518_518326

theorem koi_fish_in_pond:
  ∃ k : ℕ, 2 * k - 14 = 64 ∧ k = 39 := sorry

end koi_fish_in_pond_l518_518326


namespace correct_average_l518_518319

theorem correct_average (s : ℕ → ℕ) (H0 : (∑ i in Finset.range 10, s i) = 402)
    (H1 : s 0 = s 0' + 16) (H2 : s 1 = 13) (H3 : s 1' = 31) : 
    (∑ i in Finset.range 10, if i = 0 then s 0' else if i = 1 then s 1' else s i) / 10 = 40.4 :=
by
  sorry

end correct_average_l518_518319


namespace pictures_on_front_l518_518588

-- Conditions
variable (total_pictures : ℕ)
variable (pictures_on_back : ℕ)

-- Proof obligation
theorem pictures_on_front (h1 : total_pictures = 15) (h2 : pictures_on_back = 9) : total_pictures - pictures_on_back = 6 :=
sorry

end pictures_on_front_l518_518588


namespace find_number_l518_518937

variable (a b x : ℕ)

theorem find_number
    (h1 : x * a = 7 * b)
    (h2 : x * a = 20)
    (h3 : 7 * b = 20) :
    x = 1 :=
sorry

end find_number_l518_518937


namespace travel_time_tripled_l518_518066

variable {S v v_r : ℝ}

-- Conditions of the problem
def condition1 (t1 t2 : ℝ) : Prop :=
  t1 = 3 * t2

def condition2 (t1 t2 : ℝ) : Prop :=
  t1 = S / (v + v_r) ∧ t2 = S / (v - v_r)

def stationary_solution : Prop :=
  v = 2 * v_r

-- Conclusion: Time taken to travel from B to A without paddles is 3 times longer than usual
theorem travel_time_tripled (t_no_paddle t2 : ℝ) (h1 : condition1 t_no_paddle t2) (h2 : condition2 t_no_paddle t2) (h3 : stationary_solution) :
  t_no_paddle = 3 * t2 :=
sorry

end travel_time_tripled_l518_518066


namespace find_missing_number_l518_518747

theorem find_missing_number (x : ℚ) (h : (476 + 424) * 2 - x * 476 * 424 = 2704) : 
  x = -1 / 223 :=
by
  sorry

end find_missing_number_l518_518747


namespace count_of_non_conditional_problems_l518_518887

def problem1_needs_conditional_statement (x : ℝ) : Prop := False

def problem2_needs_conditional_statement : Prop := False

def problem3_needs_conditional_statement (a b c : ℝ) : Prop :=
  a ≠ b ∨ b ≠ c ∨ a ≠ c

def problem4_needs_conditional_statement (x : ℝ) : Prop :=
  (x ≥ 0 ∨ x < 0)

def number_of_problems_without_conditional_statements : ℕ := 2

theorem count_of_non_conditional_problems :
  (cond1 : ∀ x, ¬ problem1_needs_conditional_statement x) ->
  (cond2 : ¬ problem2_needs_conditional_statement) ->
  (cond3 : ∀ a b c, problem3_needs_conditional_statement a b c) ->
  (cond4 : ∀ x, problem4_needs_conditional_statement x) ->
  ({num : ℕ | num = 4 - (Nat.card {n | problem1_needs_conditional_statement n ∨ problem2_needs_conditional_statement ∨ problem3_needs_conditional_statement n n n ∨ problem4_needs_conditional_statement n })} = number_of_problems_without_conditional_statements) :=
by 
  intros _ _ _ _ 
  sorry

end count_of_non_conditional_problems_l518_518887


namespace blue_pen_cost_l518_518208

theorem blue_pen_cost :
  ∃ x : ℝ, (10 * x) + (15 * 2 * x) = 4 ∧ x = 0.1 :=
begin
  use 0.1,
  split,
  { calc (10 * 0.1) + (15 * 2 * 0.1)
            = 1 + (15 * 0.2) : by norm_num
        ... = 1 + 3         : by norm_num
        ... = 4             : by norm_num },
  { refl }
end

end blue_pen_cost_l518_518208


namespace analogy_sphere_from_circle_l518_518801

-- Noncomputable because mathematical existence is described, not computational.
noncomputable def determine_circle (A B C : Point) (h_non_collinear : ¬Collinear A B C) : Circle :=
sorry

-- Noncomputable because mathematical existence is described, not computational.
noncomputable def determine_sphere (A B C D : Point) (h_non_coplanar : ¬Coplanar A B C D) : Sphere :=
sorry

theorem analogy_sphere_from_circle
  (A B C : Point) (h_non_collinear : ¬Collinear A B C)
  (D : Point) (h_non_coplanar : ¬Coplanar A B C D) :
  Sphere :=
begin
  let circle := determine_circle A B C h_non_collinear,
  let sphere := determine_sphere A B C D h_non_coplanar,
  exact sphere
end

end analogy_sphere_from_circle_l518_518801


namespace travel_time_tripled_l518_518064

variable {S v v_r : ℝ}

-- Conditions of the problem
def condition1 (t1 t2 : ℝ) : Prop :=
  t1 = 3 * t2

def condition2 (t1 t2 : ℝ) : Prop :=
  t1 = S / (v + v_r) ∧ t2 = S / (v - v_r)

def stationary_solution : Prop :=
  v = 2 * v_r

-- Conclusion: Time taken to travel from B to A without paddles is 3 times longer than usual
theorem travel_time_tripled (t_no_paddle t2 : ℝ) (h1 : condition1 t_no_paddle t2) (h2 : condition2 t_no_paddle t2) (h3 : stationary_solution) :
  t_no_paddle = 3 * t2 :=
sorry

end travel_time_tripled_l518_518064


namespace sum_exponents_in_binary_representation_of_2023_l518_518820

  theorem sum_exponents_in_binary_representation_of_2023 : ∃ (s : Finset ℕ), (∑ x in s, 2^x = 2023) ∧ (∑ x in s, x = 48) :=
  by
    sorry
  
end sum_exponents_in_binary_representation_of_2023_l518_518820


namespace five_letter_words_with_one_consonant_l518_518139

theorem five_letter_words_with_one_consonant :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F']
  let vowels := ['A', 'E']
  let consonants := ['B', 'C', 'D', 'F']
  let total_words := (letters.length : ℕ)^5
  let vowel_only_words := (vowels.length : ℕ)^5
  total_words - vowel_only_words = 7744 :=
by
  sorry

end five_letter_words_with_one_consonant_l518_518139


namespace part_a_part_b_l518_518357

def balanced (S : set ℕ) : Prop :=
∀ a ∈ S, ∃ b ∈ S, b ≠ a ∧ (a + b) / 2 ∈ S

theorem part_a (k : ℕ) (h : k > 1) : 
  ∀ S : set ℕ, n = 2 ^ k → (|S| > 3 * n / 4) → balanced S :=
sorry

theorem part_b (k : ℕ) (h : k > 1) :
  ∃ S : set ℕ, n = 2 ^ k → (|S| > 2 * n / 3) ∧ not (balanced S) :=
sorry

end part_a_part_b_l518_518357


namespace angle_equal_l518_518906

theorem angle_equal (C1 C2 : Circle) (O1 O2 : Point)
  (A : Point) (h1 : A ∈ C1 ∧ A ∈ C2) 
  (P1 P2 Q1 Q2 : Point)
  (h2 : IsTangentToCircle P1 C1 ∧ IsTangentToCircle P1 C2 ∧ 
        IsTangentToCircle Q1 C1 ∧ IsTangentToCircle Q1 C2 ∧
        IsTangentToCircle P2 C1 ∧ IsTangentToCircle P2 C2 ∧ 
        IsTangentToCircle Q2 C1 ∧ IsTangentToCircle Q2 C2) 
  (M1 : Point) (M2 : Point)
  (h3 : IsMidpoint M1 P1 Q1 ∧ IsMidpoint M2 P2 Q2) :
  angle O1 A O2 = angle M1 A M2 := 
sorry

end angle_equal_l518_518906


namespace find_a_l518_518942

theorem find_a : ∃ a : ℝ, (∀ (x y : ℝ), (3 * x + y + a = 0) → (x^2 + y^2 + 2 * x - 4 * y = 0) → a = 1) :=
by
  let center_x : ℝ := -1
  let center_y : ℝ := 2
  have line_eqn : ∀ a : ℝ, 3 * center_x + center_y + a = 0
  have circle_eqn : ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → (x, y) = (center_x, center_y)
  sorry

end find_a_l518_518942


namespace line_passes_through_circle_center_l518_518941

theorem line_passes_through_circle_center
  (a : ℝ)
  (h_line : ∀ (x y : ℝ), 3 * x + y + a = 0 → (x, y) = (-1, 2))
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0 → (x, y) = (-1, 2)) :
  a = 1 :=
by
  sorry

end line_passes_through_circle_center_l518_518941


namespace greatest_number_of_elements_in_S_l518_518781

theorem greatest_number_of_elements_in_S (S : Set ℕ) (n : ℕ) (N : ℕ) (hn : Nat.Prime n)
  (h1 : 1 ∈ S) (h2 : 2310 ∈ S) (hsize: S.size = n + 1)
  (hmean : ∀ x ∈ S, (S.erase x).sum % n = 0) :
  S.size = 20 :=
sorry

end greatest_number_of_elements_in_S_l518_518781


namespace balls_in_boxes_l518_518533

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l518_518533


namespace slow_speed_distance_l518_518543

theorem slow_speed_distance (D : ℝ) (h : (D + 20) / 14 = D / 10) : D = 50 := by
  sorry

end slow_speed_distance_l518_518543


namespace property_depreciation_rate_l518_518774

noncomputable def initial_value : ℝ := 25599.08977777778
noncomputable def final_value : ℝ := 21093
noncomputable def annual_depreciation_rate : ℝ := 0.063

theorem property_depreciation_rate :
  final_value = initial_value * (1 - annual_depreciation_rate)^3 :=
sorry

end property_depreciation_rate_l518_518774


namespace area_of_region_l518_518406

def fractional_part (x : ℝ) : ℝ := x - floor x

theorem area_of_region : 
  let S := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ 50 * fractional_part p.1 ≥ floor p.1 + 2 * floor p.2} in
  measure_theory.measure_space.volume (set.univ ∩ S) = 0.49 :=
by
  sorry

end area_of_region_l518_518406


namespace percentile_65th_is_38_l518_518156

def data_set : List ℕ := [25, 29, 30, 32, 37, 38, 40, 42]

def N : ℕ := data_set.length

def P : ℕ := 65

def position : ℝ := N * (P / 100.0)

def percentile_value : ℕ := 
  let pos := position
  if pos.ceil ≤ N then data_set.nth (pos.ceil.to_nat - 1)
  else data_set.head!

theorem percentile_65th_is_38 :
  percentile_value = 38 := by
  sorry

end percentile_65th_is_38_l518_518156


namespace largest_n_for_factorable_polynomial_l518_518413

theorem largest_n_for_factorable_polynomial : ∃ n, 
  (∀ A B : ℤ, (6 * B + A = n) → (A * B = 144)) ∧ 
  (∀ n', (∀ A B : ℤ, (6 * B + A = n') → (A * B = 144)) → n' ≤ n) ∧ 
  (n = 865) :=
by
  sorry

end largest_n_for_factorable_polynomial_l518_518413


namespace dino_hourly_rate_third_gig_l518_518036

def dino_hours_first_gig : ℕ := 20
def dino_rate_first_gig : ℕ := 10
def dino_hours_second_gig : ℕ := 30
def dino_rate_second_gig : ℕ := 20
def dino_hours_third_gig : ℕ := 5
def dino_expenses : ℕ := 500
def dino_leftover : ℕ := 500

theorem dino_hourly_rate_third_gig :
  let total_earnings := dino_expenses + dino_leftover,
      earnings_first_gig := dino_hours_first_gig * dino_rate_first_gig,
      earnings_second_gig := dino_hours_second_gig * dino_rate_second_gig,
      earnings_first_two_gigs := earnings_first_gig + earnings_second_gig,
      earnings_third_gig := total_earnings - earnings_first_two_gigs,
      rate_third_gig := earnings_third_gig / dino_hours_third_gig
  in rate_third_gig = 40 := by
  sorry

end dino_hourly_rate_third_gig_l518_518036


namespace square_division_distinct_10_squares_l518_518320

theorem square_division_distinct_10_squares :
  ∃ f : ℕ → set (set (ℝ × ℝ)), 
      (∀ n : ℕ, n < 8 → card (f n) = 10) ∧
      (∀ n m : ℕ, n < 8 → m < 8 → n ≠ m → ∀ s ∈ f n, s ∉ f m) :=
sorry

end square_division_distinct_10_squares_l518_518320


namespace exam_marks_count_l518_518769

theorem exam_marks_count (c w u : ℕ) (h : c + w + u = 100) : 
  ∃ n : ℕ, n = 501 ∧ (∀ M, (∃ c w, c + w ≤ 100 ∧ M = 4 * c - w) ↔ M ∈ Icc (-100) 400) :=
by
  sorry

end exam_marks_count_l518_518769


namespace cubic_kilometers_to_cubic_meters_l518_518918

theorem cubic_kilometers_to_cubic_meters :
  (5 : ℝ) * (1000 : ℝ)^3 = 5_000_000_000 :=
by
  sorry

end cubic_kilometers_to_cubic_meters_l518_518918


namespace conjugate_in_first_quadrant_l518_518436

def point_in_quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else if z.re > 0 ∧ z.im < 0 then 4
  else 0 -- This handles the degenerate case where a point lies on the axes.

theorem conjugate_in_first_quadrant :
  let z := (3 - 4 * complex.I) / (1 + 2 * complex.I)
  in point_in_quadrant (conj z) = 1 :=
by
  sorry

end conjugate_in_first_quadrant_l518_518436


namespace geometric_sequence_sum_l518_518617

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h1 : a 1 + a 2 + a 3 = 7) 
  (h2 : a 2 + a 3 + a 4 = 14) 
  (geom_seq : ∃ q, ∀ n, a (n + 1) = q * a n ∧ q = 2) :
  a 4 + a 5 + a 6 = 56 := 
by
  sorry

end geometric_sequence_sum_l518_518617


namespace find_a_l518_518100

theorem find_a (a : ℝ) (h : (↑(a : ℂ) - complex.I) / (3 + complex.I)).re = 1 / 2 : a = 2 :=
sorry

end find_a_l518_518100


namespace sum_triple_l518_518278

theorem sum_triple {a b c d T : ℝ} (h : a + b + c + d = T) : 3 * (T + 4) = 3T + 12 :=
by
  sorry

end sum_triple_l518_518278


namespace isosceles_triangle_perimeter_20_l518_518428

-- Declaring the variables and assumptions
variables {x y : ℝ}
def valid_pair (x y : ℝ) := abs (4 - x) + real.sqrt (y - 8) = 0
def is_isosceles_triangle (a b c : ℝ) := a = b ∨ b = c ∨ a = c
def valid_triangle (a b c : ℝ) := a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem statement
theorem isosceles_triangle_perimeter_20 (hx : valid_pair x y) :
  is_isosceles_triangle x y (x + y - x) →
  valid_triangle x y (x + y - x) →
  2 * y + 4 = 20 :=
sorry

end isosceles_triangle_perimeter_20_l518_518428


namespace reciprocal_roots_l518_518302

theorem reciprocal_roots (a b : ℝ) (h : a ≠ 0) :
  ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + a = 0) ∧ (a * x2^2 + b * x2 + a = 0) → x1 = 1 / x2 ∧ x2 = 1 / x1 :=
by
  intros x1 x2 hroots
  have hsum : x1 + x2 = -b / a := by sorry
  have hprod : x1 * x2 = 1 := by sorry
  sorry

end reciprocal_roots_l518_518302


namespace f_100_eq_101_l518_518929

theorem f_100_eq_101 (f : ℕ → ℕ) (h1 : ∀ n, f(n + 1) = f(n) + 1) (h2 : f(1) = 2) : 
  f(100) = 101 := 
by 
  sorry

end f_100_eq_101_l518_518929


namespace evaporation_period_l518_518756

theorem evaporation_period (
  initial_water ounces : ℝ,
  daily_evaporation_rate ounces_per_day : ℝ,
  total_evaporation_percentage : ℝ,
  total_evaporation_percent_amount : ℝ,
  number_of_days : ℝ
) :
  initial_water = 10 →
  daily_evaporation_rate = 0.00008 →
  total_evaporation_percentage = 0.04 / 100 →
  total_evaporation_percent_amount = initial_water * total_evaporation_percentage →
  number_of_days = total_evaporation_percent_amount / daily_evaporation_rate →
  number_of_days = 500 :=
by
  sorry

end evaporation_period_l518_518756


namespace find_bk_l518_518653

theorem find_bk
  (A B C D : ℝ)
  (BC : ℝ) (hBC : BC = 3)
  (AB CD : ℝ) (hAB_CD : AB = 2 * CD)
  (BK : ℝ) (hBK : BK = 2) :
  ∃ x a : ℝ, (x = BK) ∧ (AB = 2 * CD) ∧ ((2 * a + x) * (3 - x) = x * (a + 3 - x)) :=
by
  sorry

end find_bk_l518_518653


namespace islanders_liars_l518_518637

theorem islanders_liars (n : ℕ) (h : n = 450) : (∃ L : ℕ, (L = 150 ∨ L = 450)) :=
sorry

end islanders_liars_l518_518637


namespace how_fast_is_a_l518_518541

variable (a b : ℝ) (k : ℝ)

theorem how_fast_is_a (h1 : a = k * b) (h2 : a + b = 1 / 30) (h3 : a = 1 / 40) : k = 3 := sorry

end how_fast_is_a_l518_518541


namespace every_positive_integer_sum_of_distinct_powers_of_3_4_7_l518_518216

theorem every_positive_integer_sum_of_distinct_powers_of_3_4_7 :
  ∀ n : ℕ, n > 0 →
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  ∃ (i j k : ℕ), n = 3^i + 4^j + 7^k :=
by
  sorry

end every_positive_integer_sum_of_distinct_powers_of_3_4_7_l518_518216


namespace tangent_line_tangent_to_circle_min_2_pow_a_2_pow_b_l518_518472

open Real

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := -(1 / sqrt b) * exp (sqrt a * x)

-- Condition that tangent line to graph at x = 0 is tangent to the circle x^2 + y^2 = 1
theorem tangent_line_tangent_to_circle_min_2_pow_a_2_pow_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (tangent_condition : a + b = 1) : (2:ℝ)^a + (2:ℝ)^b = 2 * sqrt 2 :=
by
  sorry

end tangent_line_tangent_to_circle_min_2_pow_a_2_pow_b_l518_518472


namespace probability_select_STRONG_l518_518971

theorem probability_select_STRONG : 
  let TRAIN := ["T", "R", "A", "I", "N"]
  let SHIELD := ["S", "H", "I", "E", "L", "D"]
  let GROW := ["G", "R", "O", "W"]
  let STRONG := ["S", "T", "R", "O", "N", "G"]
  let p_train := 1 / Real.binom 5 3 -- Probability of selecting S, T, and R from TRAIN
  let p_shield := 3 / Real.binom 6 4 -- Probability of selecting H, I, and E from SHIELD
  let p_grow := 1 / Real.binom 4 2   -- Probability of selecting G and O from GROW
  let total_probability := p_train * p_shield * p_grow
  total_probability = 1 / 300 :=
begin
  sorry
end

end probability_select_STRONG_l518_518971


namespace solution_set_for_a_2_value_of_a_for_right_triangle_l518_518896

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := abs (x + 1) - abs (a * x - 3)

-- Statement for part 1
theorem solution_set_for_a_2 :
  { x : ℝ | f 2 x > 1 } = { x : ℝ | 1 < x ∧ x < 3 } :=
sorry

-- Statement for part 2
theorem value_of_a_for_right_triangle :
  (∀ x : ℝ, f a x = (if x ≤ -1 then (a - 1) * x - 4 else if x ≤ 3 / a then (a + 1) * x - 2 else (1 - a) * x + 4))
  → (∃ (a : ℝ), a = sqrt 2) :=
sorry

end solution_set_for_a_2_value_of_a_for_right_triangle_l518_518896


namespace probability_range_l518_518945

noncomputable def X : Type := sorry
noncomputable def P : X → Prop := sorry

axiom normal_distribution (X : Type) (μ : ℝ) (σ2 : ℝ) : Prop := sorry
axiom prob (P : X → Prop) (x : ℝ) : ℝ := sorry

theorem probability_range {X : Type} (H : normal_distribution X 1 4) (H1 : prob (λ x, x ≤ 0) = 0.1) :
  prob (λ x, 0 < x ∧ x < 2) = 0.8 :=
sorry

end probability_range_l518_518945


namespace num_students_even_l518_518235

theorem num_students_even (n : ℕ) (f : Fin n → ℤ → ℤ) :
  (∀ i, f i (-1) ∨ f i 1) →
  n % 2 = 0 :=
by
  sorry

end num_students_even_l518_518235


namespace sum_of_digits_M_l518_518270

noncomputable def M : ℕ := nat.sqrt (36^49 * 49^36)

theorem sum_of_digits_M :
  (nat.digits 10 M).sum = 37 := by
  sorry

end sum_of_digits_M_l518_518270


namespace dan_present_age_l518_518031

-- Let x be Dan's present age
variable (x : ℤ)

-- Condition: Dan's age after 18 years will be 8 times his age 3 years ago
def condition (x : ℤ) : Prop :=
  x + 18 = 8 * (x - 3)

-- The goal is to prove that Dan's present age is 6
theorem dan_present_age (x : ℤ) (h : condition x) : x = 6 :=
by
  sorry

end dan_present_age_l518_518031


namespace items_purchased_total_profit_l518_518337

-- Definitions based on conditions given in part (a)
def total_cost := 6000
def cost_A := 22
def cost_B := 30
def sell_A := 29
def sell_B := 40

-- Proven answers from the solution (part (b))
def items_A := 150
def items_B := 90
def profit := 1950

-- Lean theorem statements (problems to be proved)
theorem items_purchased : (22 * items_A + 30 * (items_A / 2 + 15) = total_cost) → 
                          (items_A = 150) ∧ (items_B = 90) := sorry

theorem total_profit : (items_A = 150) → (items_B = 90) → 
                       ((items_A * (sell_A - cost_A) + items_B * (sell_B - cost_B)) = profit) := sorry

end items_purchased_total_profit_l518_518337


namespace vector_equation_l518_518192

variables {α β γ : ℝ}
variables {A B C X M : EuclideanSpace ℝ (fin 3)}
variables (α β γ : ℝ)
variables (A B C X M : EuclideanSpace ℝ (fin 3))

noncomputable def barycentric_coordinates (X : EuclideanSpace ℝ (fin 3)) (α β γ : ℝ) : Prop :=
  X = α • A + β • B + γ • C ∧ α + β + γ = 1

noncomputable def centroid (A B C : EuclideanSpace ℝ (fin 3)) : EuclideanSpace ℝ (fin 3) :=
  (1/3 : ℝ) • A + (1/3 : ℝ) • B + (1/3 : ℝ) • C

theorem vector_equation (α β γ : ℝ)
  (A B C X : EuclideanSpace ℝ (fin 3))
  (h1 : barycentric_coordinates X α β γ)
  (M := centroid A B C) :
  3 • (X - M) = (α - β) • (B - A) + (β - γ) • (C - B) + (γ - α) • (A - C) :=
sorry

end vector_equation_l518_518192


namespace problem_statement_l518_518134

variable {a b : ℝ}

theorem problem_statement : ({a^2, 0, -1} = {a, b, 0}) → a^2014 + b^2014 = 2 := by
  sorry

end problem_statement_l518_518134


namespace units_digit_G_n_for_n_eq_3_l518_518728

def G (n : ℕ) : ℕ := 2 ^ 2 ^ 2 ^ n + 1

theorem units_digit_G_n_for_n_eq_3 : (G 3) % 10 = 7 := 
by 
  sorry

end units_digit_G_n_for_n_eq_3_l518_518728


namespace complex_pure_imaginary_l518_518551

noncomputable def imaginaryPart (z : ℂ) : ℂ :=
Complex.im z * Complex.I

theorem complex_pure_imaginary (m : ℝ) (z : ℂ) (hz : z = (1 - m * Complex.I) / (2 + Complex.I)) (hz_imag : imaginaryPart z = z) : m = 2 :=
by
  sorry

end complex_pure_imaginary_l518_518551


namespace tan_theta_in_terms_of_x_l518_518199

-- Define the theorem
theorem tan_theta_in_terms_of_x (theta : ℝ) (x : ℝ) (h : 0 < x)
  (h0 : 0 < θ ∧ θ < π / 2)
  (h_cos_half_theta : cos (θ / 2) = sqrt ((x - 1) / (2 * x))) :
  tan θ = -x * sqrt (1 - 1 / x^2) :=
begin
  sorry
end

end tan_theta_in_terms_of_x_l518_518199


namespace irrational_count_l518_518004

def is_irrational (x : ℝ) : Prop :=
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

theorem irrational_count : 
  let numbers := [ (16 / 3 : ℝ), real.sqrt 3, real.pi, 0, (-1.6 : ℝ), real.sqrt 6 ] in
  (list.filter is_irrational numbers).length = 3 :=
by
  sorry

end irrational_count_l518_518004


namespace max_rooks_on_chessboard_l518_518721

theorem max_rooks_on_chessboard : 
  ∃ (n : ℕ), n = 10 ∧
  ∀ (rooks : ℕ → ℕ → Prop), 
    (∀ i j, rooks i j → (i < 8) ∧ (j < 8)) ∧
    (∀ i j, rooks i j → (∀ k, k ≠ i → ¬rooks k j) ∧ (∀ l, l ≠ j → ¬rooks i l)) → 
    (finset.card (finset.filter (λ x, rooks x.1 x.2) 
      (finset.univ : finset (ℕ × ℕ))) ≤ n) :=
sorry

end max_rooks_on_chessboard_l518_518721


namespace beth_coins_sold_l518_518799

def initial_coins : ℕ := 250
def additional_coins : ℕ := 75
def percentage_sold : ℚ := 60 / 100
def total_coins : ℕ := initial_coins + additional_coins
def coins_sold : ℚ := percentage_sold * total_coins

theorem beth_coins_sold : coins_sold = 195 :=
by
  -- Sorry is used to skip the proof as requested
  sorry

end beth_coins_sold_l518_518799


namespace range_of_a_l518_518483

noncomputable def proposition_p (a : ℝ) : Prop :=
∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

noncomputable def proposition_q (a : ℝ) : Prop :=
∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0

theorem range_of_a (a : ℝ) (h : proposition_p a ∧ proposition_q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l518_518483


namespace units_digit_of_product_l518_518727

theorem units_digit_of_product : 
  (3 ^ 401 * 7 ^ 402 * 23 ^ 403) % 10 = 9 := 
by
  sorry

end units_digit_of_product_l518_518727


namespace balls_in_boxes_l518_518505

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l518_518505


namespace rocket_momentum_l518_518296

theorem rocket_momentum (m F d p : ℝ)
  (h1 : p = sqrt (2 * d * m * F))
  (h2 : 0 < m) (h3 : 0 < d) (h4 : 0 < F) :
  let p' := 3 * p in
  p' = 3 * p :=
by
  sorry

end rocket_momentum_l518_518296


namespace find_function_expression_l518_518479

theorem find_function_expression
  (A ω φ : ℝ) (x : ℝ)
  (h1 : A > 0)
  (h2 : ω > 0)
  (h3 : abs φ < (π / 2))
  (h4 : (2 * π) / ω = (2 * π) / 3)
  (h5 : -2 = -A)
  (h6 : A * sin (ω * (5 * π / 9) + φ) = 0) :
  y = 2 * sin (3 * x + π / 3) :=
sorry

end find_function_expression_l518_518479


namespace distance_traveled_l518_518545

theorem distance_traveled
  (D : ℝ) (T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 14 * T)
  : D = 50 := sorry

end distance_traveled_l518_518545


namespace max_sum_a_b_c_d_e_f_g_l518_518687

theorem max_sum_a_b_c_d_e_f_g (a b c d e f g : ℕ)
  (h1 : a + b + c = 2)
  (h2 : b + c + d = 2)
  (h3 : c + d + e = 2)
  (h4 : d + e + f = 2)
  (h5 : e + f + g = 2) :
  a + b + c + d + e + f + g ≤ 6 := 
sorry

end max_sum_a_b_c_d_e_f_g_l518_518687


namespace triangle_right_angle_min_side_l518_518698

theorem triangle_right_angle_min_side (s : ℕ) (h1 : 7.5 + s > 12) (h2 : 7.5 + 12 > s) (h3 : 12 + s > 7.5) (h4 : 7.5^2 + 12^2 = s^2) : s = 15 :=
by sorry

end triangle_right_angle_min_side_l518_518698


namespace expression_simplification_l518_518932

theorem expression_simplification (x : ℝ) (h : x < -2) : 1 - |1 + x| = -2 - x := 
by
  sorry

end expression_simplification_l518_518932


namespace arrange_numbers_in_triangle_l518_518967

theorem arrange_numbers_in_triangle :
  ∃ (a b c d e f g h i : ℕ),
  {a, b, c, d, e, f, g, h, i} = {2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024} ∧
  (a + b + c = d + e + f ∧ d + e + f = g + h + i) := 
sorry

end arrange_numbers_in_triangle_l518_518967


namespace balls_in_boxes_l518_518503

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l518_518503


namespace find_b_l518_518262

noncomputable def func (x a b : ℝ) := (1 / 12) * x^2 + a * x + b

theorem  find_b (a b : ℝ) (x1 x2 : ℝ):
    (func x1 a b = 0) →
    (func x2 a b = 0) →
    (b = (x1 * x2) / 12) →
    ((3 - x1) = (x2 - 3)) →
    (b = -6) :=
by
    sorry

end find_b_l518_518262


namespace product_divisible_by_60_l518_518651

open Nat

theorem product_divisible_by_60 (S : Finset ℕ) (h_card : S.card = 10) (h_sum : S.sum id = 62) :
  60 ∣ S.prod id :=
  sorry

end product_divisible_by_60_l518_518651


namespace koi_fish_in_pond_l518_518329

theorem koi_fish_in_pond : ∃ k : ℤ, 2 * k - 14 = 64 ∧ k = 39 := by
  use 39
  split
  · sorry
  · rfl

end koi_fish_in_pond_l518_518329


namespace part_a_part_b_part_c_l518_518833

-- Definition of ord_2
def ord_2 (i : ℤ) : ℕ :=
  if h : i ≠ 0 then
    @nat.find (λ n, 2 ^ n ∣ i ∧ ¬ (2 ^ (n + 1)) ∣ i) (nat.exists_pow_dvd_and_dvd_succ h)
  else 0

-- Given the condition and the questions, we state the proofs as follows:

-- Part (a): For which positive integers \( n \) is \( ord_2(3^n - 1) = 1 \) ?
theorem part_a (n : ℕ) (h : n > 0) : ord_2 (3^n - 1) = 1 ↔ n % 2 = 1 :=
sorry

-- Part (b): For which positive integers \( n \) is \( ord_2(3^n - 1) = 2 \) ?
theorem part_b (n : ℕ) (h : n > 0) : ord_2 (3^n - 1) ≠ 2 :=
sorry

-- Part (c): For which positive integers \( n \) is \( ord_2(3^n - 1) = 3 \) ?
theorem part_c (n : ℕ) (h : n > 0) : ord_2 (3^n - 1) = 3 ↔ n % 4 = 2 :=
sorry

end part_a_part_b_part_c_l518_518833


namespace sum_xyz_l518_518146

variables (x y z : ℤ)

theorem sum_xyz (h1 : y = 3 * x) (h2 : z = 3 * y - x) : x + y + z = 12 * x :=
by 
  -- skip the proof
  sorry

end sum_xyz_l518_518146


namespace largest_integer_in_set_A_l518_518485

theorem largest_integer_in_set_A (x : ℝ) (h : |x - 55| ≤ 11 / 2) : x ≤ 60 :=
begin
    sorry
end

end largest_integer_in_set_A_l518_518485


namespace slope_of_line_MN_constant_1_over_m_plus_1_over_n_l518_518466

noncomputable def ellipse_equation := ∀ x y : ℝ, (x^2) / 8 + (y^2) / 4 = 1
def midpoint_condition := ∀ x1 y1 x2 y2 : ℝ,  ((x1 + x2) / 2 = 1) ∧ ((y1 + y2) / 2 = 1)

theorem slope_of_line_MN : 
  ∀ x1 y1 x2 y2 : ℝ, ellipse_equation x1 y1 → ellipse_equation x2 y2 → midpoint_condition x1 y1 x2 y2 → 
  (y1 - y2) / (x1 - x2) = -1 / 2 := 
  sorry

theorem constant_1_over_m_plus_1_over_n : 
  ∀ k : ℝ, 
  let m := 4 * sqrt 2 * (1 + k^2) / (1 + 2 * k^2),
      n := 4 * sqrt 2 * (1 + k^2) / (k^2 + 2) in 
  (1 / m) + (1 / n) = (3 * sqrt 2) / 8 := 
  sorry

end slope_of_line_MN_constant_1_over_m_plus_1_over_n_l518_518466


namespace distance_between_bus_stops_l518_518251

theorem distance_between_bus_stops (d : ℕ) (unit : String) 
  (h: d = 3000 ∧ unit = "meters") : unit = "C" := 
by 
  sorry

end distance_between_bus_stops_l518_518251


namespace linear_inequality_solution_l518_518463

theorem linear_inequality_solution (a b : ℝ)
  (h₁ : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ (x < -3 ∨ x > 1)) :
  ∀ x : ℝ, a * x + b < 0 ↔ x < 3 / 2 :=
by
  sorry

end linear_inequality_solution_l518_518463


namespace find_a_curve_condition_l518_518341

theorem find_a_curve_condition (a : ℝ) (C : ℝ → ℝ) (M A B : ℝ × ℝ) :
  (∀ x, C x = 2 * x ^ 3 + a * x + a) →
  C (-1) = 0 →
  (∀ t, let y' := 6 * t ^ 2 + a in
       y' = ((2 * t ^ 3 + a * t + a) / (t + 1))) →
  let y'_0 := 6 * (0 : ℝ) ^ 2 + a in
  let y'_-3_2 := 6 * ((-3/2 : ℝ) ^ 2) + a in
  y'_0 + y'_-3_2 = 0 →
  a = -27 / 4 :=
begin
  -- sorry will be replaced with the solution steps
  sorry
end

end find_a_curve_condition_l518_518341


namespace matrix_product_l518_518022

theorem matrix_product :
  let matrices := λ n, (1 + n) * (1 + n) in
  let product := ∏ n in (range 1 51), matrices (2 * n) in
  product = matrix [[1, 2550], [0, 1]] :=
by
  sorry

end matrix_product_l518_518022


namespace selection_methods_for_charity_event_l518_518657

theorem selection_methods_for_charity_event : 
  let total_selections := Nat.choose 10 4
  let selections_excluding_A_B := Nat.choose 8 4
  total_selections - selections_excluding_A_B = 140 :=
by
  let total_selections := Nat.choose 10 4
  let selections_excluding_A_B := Nat.choose 8 4
  have h1 : total_selections = 210 := by sorry
  have h2 : selections_excluding_A_B = 70 := by sorry
  calc 
    total_selections - selections_excluding_A_B
      = 210 - 70 : by rw [h1, h2]
  ... = 140 : by norm_num

end selection_methods_for_charity_event_l518_518657


namespace math_problem_l518_518115

variable (f g : ℝ → ℝ)
variable (a b x : ℝ)
variable (h_has_derivative_f : ∀ x, Differentiable ℝ f)
variable (h_has_derivative_g : ∀ x, Differentiable ℝ g)
variable (h_deriv_ineq : ∀ x, deriv f x > deriv g x)
variable (h_interval : x ∈ Ioo a b)

theorem math_problem :
  (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) :=
sorry

end math_problem_l518_518115


namespace remainder_x_squared_l518_518434

theorem remainder_x_squared (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 7 * x ≡ 14 [ZMOD 20]) : 
  (x^2 ≡ 4 [ZMOD 20]) :=
sorry

end remainder_x_squared_l518_518434


namespace hyperbola_asymptote_b_l518_518550

theorem hyperbola_asymptote_b {b : ℝ} (hb : b > 0) :
  (∀ x y : ℝ, x^2 - (y^2 / b^2) = 1 → (y = 2 * x)) → b = 2 := by
  sorry

end hyperbola_asymptote_b_l518_518550


namespace lite_soda_bottles_l518_518762

theorem lite_soda_bottles (regular diet total : ℕ) (h_reg : regular = 49) (h_diet : diet = 40) (h_total : total = 89) :
  (total = regular + diet) → ∃ (lite : ℕ), lite = 0 :=
by
  intro h
  use 0
  rw [h_reg, h_diet, h_total] at h
  have h49 : regular = 49 := h_reg
  have h40 : diet = 40 := h_diet
  have h89 : total = 89 := h_total
  have h_calculated_total : 49 + 40 = 89 := by norm_num
  have h_actual_total : regular + diet = total := h
  rw [←h_actual_total, h_calculated_total] at h
  exact h

end lite_soda_bottles_l518_518762


namespace exists_equal_submatrix_of_size_l518_518299

structure Matrix (α : Type*) (n : ℕ) := 
  (data : Fin n → Fin n → α)

def isWellGroomed (A : Matrix ℕ n) : Prop :=
  ∀ i1 i2 j1 j2 : Fin n, i1 ≠ i2 → j1 ≠ j2 → 
    ¬((A.data i1 j1 = 1) ∧ (A.data i1 j2 = 0) ∧ 
      (A.data i2 j1 = 0) ∧ (A.data i2 j2 = 1))

theorem exists_equal_submatrix_of_size (n : ℕ) (A : Matrix ℕ n) (h : isWellGroomed A) :
  ∃ c > (0:ℝ), ∃ m, m ≥ (c * n) ∧ ∃ (subA : Matrix ℕ m), 
  (∀ i j : Fin m, subA.data i j = 0) ∨ (∀ i j : Fin m, subA.data i j = 1) :=
sorry

end exists_equal_submatrix_of_size_l518_518299


namespace coefficient_x_squared_in_binomial_expansion_l518_518573

theorem coefficient_x_squared_in_binomial_expansion :
  let C := Nat.choose in
  ((x : ℚ) - (1 / (4 * x)))^6 = ∑ r in Finset.range 7, (-(1 / 4))^r * (C 6 r) * x^(6 - 2 * r) →
  ∑ r in Finset.range 3, (-(1 / 4))^r * (C 6 r) * x^(6 - 2 * r) = 15 / 16 * x^2 ∧
  ∀ r ∈ Finset.range 6, x^(6-2*r) = 2 → r = 2 ∧
  (-(1 / 4))^2 * (C 6 2) * x^2 = 1 / 16 * 15 * x^2 : by sorry

end coefficient_x_squared_in_binomial_expansion_l518_518573


namespace area_of_intersection_l518_518128

def setA : set (ℝ × ℝ) := { p | let (x, y) := p in (y - x) * (y - 1/x) ≥ 0 }
def setB : set (ℝ × ℝ) := { p | let (x, y) := p in (x - 1)^2 + (y - 1)^2 ≤ 1 }

theorem area_of_intersection : measure_theory.measure_space.volume (setA ∩ setB) = (real.pi / 2) :=
sorry

end area_of_intersection_l518_518128


namespace find_a_and_x_l518_518351

theorem find_a_and_x (a x : ℝ) (ha1 : x = (2 * a - 1)^2) (ha2 : x = (-a + 2)^2) : a = -1 ∧ x = 9 := 
by
  sorry

end find_a_and_x_l518_518351


namespace triangle_tangent_condition_l518_518170

theorem triangle_tangent_condition (a b c A B C : ℝ)
    (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : A + B + C = π)
    (h5 : tan C / tan A + tan C / tan B = 1) :
    (a^2 + b^2) / c^2 = 3 :=
by 
  sorry

end triangle_tangent_condition_l518_518170


namespace product_labels_l518_518685

def f2 (x : ℝ) : ℝ := x^3 - 3 * x

def domain_f3 : Set ℝ := {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3}

def g4 (x : ℝ) : ℝ := Real.tan x

def h5 (x : ℝ) : ℝ := 3 / x

-- Define a predicate for invertibility (injective functions over their domains).
def is_invertible {α β : Type*} (f : α → β) : Prop :=
  ∀ y1 y2, f y1 = f y2 → y1 = y2

-- Define the conditions of the problem.
def invertible_f2 : ¬is_invertible f2 := by
  intros h
  have h1 : f2 1 = f2 (-1), by { simp [f2] }
  contradiction

noncomputable def invertible_f3 : is_invertible (λ x, x) := by
  intros y1 y2 h
  simp at h; assumption

def invertible_g4 : ¬is_invertible g4 := by
  intros h
  have h1 : ∃ x1 x2, x1 ≠ x2 ∧ g4 x1 = g4 x2, 
  from Exists.intro (π / 4) (5 * π / 4)
    (by simp [g4, Real.tan_periodic])
  cases h1 with x1 h1
  cases h1 with x2 h1
  cases h1 with ne h1
  contradiction

def invertible_h5 : is_invertible h5 := by
  intros y1 y2 hy
  rw [h5, h5] at hy
  exact (mul_right_inj' (by norm_num : (3 : ℝ) ≠ 0)).mp (eq.symm hy)

theorem product_labels : 3 * 5 = 15 :=
  sorry

end product_labels_l518_518685


namespace work_needed_to_stretch_spring_l518_518921

theorem work_needed_to_stretch_spring :
  ∀ (F : ℝ → ℝ) (k x : ℝ),
  -- Conditions (given)
  (∀ x, F x = k * x) ∧
  F 0.03 = 24 →
  -- Conclusion (to prove)
  (∫ x in 0..0.18, F x) = 12.96 := by
  -- Applying Hooke's Law
  sorry

end work_needed_to_stretch_spring_l518_518921


namespace cone_height_ratio_l518_518352

noncomputable def cone_ratio : ℚ :=
  let r : ℝ := 10
  let original_height := 40
  let new_volume : ℝ := 400 * real.pi
  let new_height := (3 * new_volume) / (real.pi * r^2)
  new_height / original_height

theorem cone_height_ratio :
  cone_ratio = 3 / 10 :=
by
  sorry

end cone_height_ratio_l518_518352


namespace hyperbola_eq_l518_518457

theorem hyperbola_eq : 
  ∀ (a b : ℝ), 
    (a > 0) → (b > 0) →
    (∀ x y : ℝ, x^2 + y^2 - 6*x + 5 = 0 → (∃ m₁ m₂ : ℝ, y = m₁*x ∨ y = m₂*x) ∧ 
    (sqrt(a^2 + b^2) = 3) →
    (3*b / sqrt(a^2 + b^2) = 2)) →
    (a = sqrt(5) → b = 2) → 
    (∀ x y : ℝ, ((x/a)^2 - (y/b)^2 = 1) = (x^2 / 5 - y^2 / 4 = 1)) :=
by
  intro a b ha hb circle_tangent asymptote_focus a_val b_val
  sorry

end hyperbola_eq_l518_518457


namespace probability_single_trial_l518_518153

open Real

theorem probability_single_trial :
  ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (1 - p)^4 = 16 / 81 ∧ p = 1 / 3 :=
by
  -- The proof steps have been skipped.
  sorry

end probability_single_trial_l518_518153


namespace sin_pi_over_six_l518_518701

theorem sin_pi_over_six : Real.sin (Real.pi / 6) = 1 / 2 := 
by 
  sorry

end sin_pi_over_six_l518_518701


namespace simplify_cube_root_l518_518231

theorem simplify_cube_root :
  (∛(72^3 + 108^3 + 144^3) = 36 * ∛99) :=
by
  sorry

end simplify_cube_root_l518_518231


namespace problem_statement_l518_518619

theorem problem_statement (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 1 ≤ (i : ℕ) → (i : ℕ) ≤ n → 0 < a ⟨i, sorry⟩) :
  (∑ i in Finset.range n, (a ⟨i, sorry⟩ ^ 2 / a ⟨(i + 1) % n, sorry⟩)) ≥ ∑ i in Finset.range n, (a ⟨i, sorry⟩) :=
begin
  sorry,
end

end problem_statement_l518_518619


namespace hyperbola_correct_l518_518908

-- Define the fixed points F1 and F2
def F1 : ℝ × ℝ := (5, 0)
def F2 : ℝ × ℝ := (-5, 0)

-- Define the hyperbola equation to prove
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the property of the absolute value of difference of distances given in the problem
def abs_diff_dist (P : ℝ × ℝ) : Prop :=
  |((P.1 - F1.1)^2 + (P.2 - F1.2)^2)^0.5 - ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)^0.5| = 6

-- The main theorem statement to prove: Given the conditions, the hyperbola equation holds
theorem hyperbola_correct :
  ∀ (P : ℝ × ℝ), abs_diff_dist P → hyperbola_equation P.1 P.2 :=
by
  sorry

end hyperbola_correct_l518_518908


namespace log_order_magnitude_l518_518694

-- Definitions based on conditions
def log_4_3 := Real.log 3 / Real.log 4
def log_3_4 := Real.log 4 / Real.log 3
def log_4_3_inverse := Real.log (3 / 4) / Real.log (4 / 3)

-- Prove the required order of magnitude
theorem log_order_magnitude :
  0 < log_4_3 ∧ log_4_3 < 1 ∧ 
  log_3_4 > 1 ∧ 
  log_4_3_inverse < 0 →
  log_3_4 > log_4_3 ∧ log_4_3 > log_4_3_inverse := 
by {
  sorry  -- Proof omitted
}

end log_order_magnitude_l518_518694


namespace num_ways_to_put_5_balls_into_4_boxes_l518_518528

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l518_518528


namespace width_of_boxes_l518_518331

theorem width_of_boxes
  (total_volume : ℝ)
  (total_payment : ℝ)
  (cost_per_box : ℝ)
  (h1 : total_volume = 1.08 * 10^6)
  (h2 : total_payment = 120)
  (h3 : cost_per_box = 0.2) :
  (∃ w : ℝ, w = (total_volume / (total_payment / cost_per_box))^(1/3)) :=
by {
  sorry
}

end width_of_boxes_l518_518331


namespace balls_in_boxes_l518_518537

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l518_518537


namespace triplet_zero_solution_l518_518047

theorem triplet_zero_solution (x y z : ℝ) 
  (h1 : x^3 + y = z^2) 
  (h2 : y^3 + z = x^2) 
  (h3 : z^3 + x = y^2) :
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end triplet_zero_solution_l518_518047


namespace trig_identity_l518_518926

open Real

theorem trig_identity (θ : ℝ) (h : tan θ = 2) :
  ((sin θ + cos θ) * cos (2 * θ)) / sin θ = -9 / 10 :=
sorry

end trig_identity_l518_518926


namespace solve_quadratic_inequality_l518_518234

theorem solve_quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, (a * x ^ 2 - (2 * a + 1) * x + 2 > 0 ↔
    if a = 0 then
      x < 2
    else if a > 0 then
      if a >= 1 / 2 then
        x < 1 / a ∨ x > 2
      else
        x < 2 ∨ x > 1 / a
    else
      x > 1 / a ∧ x < 2)) :=
sorry

end solve_quadratic_inequality_l518_518234


namespace rhombus_AD_eq_2x_y_15_rhombus_BD_eq_3x_5y_13_l518_518575

section RhombusEquations

variables 
  (A C P : Point)
  (AD_line BD_line : Line)
  (hA : A.x = -4 ∧ A.y = 7)
  (hC : C.x = 2 ∧ C.y = -3)
  (hP : P.x = 3 ∧ P.y = -1)
  (hBC : passes_through_line P (line_through (point_midpoint A C) AD_line))

theorem rhombus_AD_eq_2x_y_15 (hR : is_rhombus A C) : 
  line_eq AD_line 2 -1 15 :=
sorry

theorem rhombus_BD_eq_3x_5y_13 (hR : is_rhombus A C) : 
  line_eq BD_line 3 (-5) 13 :=
sorry

end RhombusEquations

end rhombus_AD_eq_2x_y_15_rhombus_BD_eq_3x_5y_13_l518_518575


namespace solution_set_inequality_l518_518878

def f (x : ℝ) : ℝ :=
if x >= 0 then 2^x - 2 else 2^(-x) - 2

theorem solution_set_inequality :
  ∀ x : ℝ, (f x = f (-x)) → (∀ x, x >= 0 → f x = 2^x - 2) →
    (∀ x, f (x - 1) ≤ 2 ↔ -1 ≤ x ∧ x ≤ 3) :=
by
  sorry

end solution_set_inequality_l518_518878


namespace sequence_properties_l518_518439

theorem sequence_properties (S : ℕ → ℝ) (a : ℕ → ℝ) :
  S 2 = 4 →
  (∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1) →
  a 1 = 1 ∧ S 5 = 121 :=
by
  intros hS2 ha
  sorry

end sequence_properties_l518_518439


namespace proof_problem_l518_518913

noncomputable def question (a b c d m : ℚ) : ℚ :=
  2 * a + 2 * b + (a + b - 3 * (c * d)) - m

def condition1 (m : ℚ) : Prop :=
  abs (m + 1) = 4

def condition2 (a b : ℚ) : Prop :=
  a = -b

def condition3 (c d : ℚ) : Prop :=
  c * d = 1

theorem proof_problem (a b c d m : ℚ) :
  condition1 m → condition2 a b → condition3 c d →
  (question a b c d m = 2 ∨ question a b c d m = -6) :=
by
  sorry

end proof_problem_l518_518913


namespace arithmetic_sequence_sum_l518_518804

open Nat

theorem arithmetic_sequence_sum :
  let a1 := -45
  let d := 2
  let an := -1
  let n := ((an - a1) / d) + 1
  let sum := (n * (a1 + an)) / 2
  sum = -529 := by
  let a1 := -45
  let d := 2
  let an := -1
  let n := ((an - a1) / d) + 1
  let sum := (n * (a1 + an)) / 2
  simp [a1, d, an, n, sum]
  sorry

end arithmetic_sequence_sum_l518_518804


namespace total_collection_in_rupees_l518_518763

def paise_to_rupees (paise: ℕ) : ℚ := paise / 100

theorem total_collection_in_rupees :
  let members := 37
  let contribution_per_member := 37
  paise_to_rupees (members * contribution_per_member) = 13.69 :=
by let members := 37
   let contribution_per_member := 37
   have total_collection_in_paise : ℕ := members * contribution_per_member
   have total_collection_in_rupees : ℚ := paise_to_rupees total_collection_in_paise
   have expected_total : ℚ := 13.69
   show paise_to_rupees total_collection_in_paise = expected_total
   sorry

end total_collection_in_rupees_l518_518763


namespace number_of_liars_l518_518643

constant islanders : Type
constant knight : islanders → Prop
constant liar : islanders → Prop
constant sits_at_table : islanders → Prop
constant right_of : islanders → islanders

axiom A1 : ∀ x : islanders, sits_at_table x → (knight x ∨ liar x)
axiom A2 : (∃ n : ℕ, n = 450 ∧ (λ x, sits_at_table x))
axiom A3 : ∀ x : islanders, sits_at_table x →
  (liar (right_of x) ∧ ¬ liar (right_of (right_of x))) ∨ 
  (¬ liar (right_of x) ∧ liar (right_of (right_of x)))

theorem number_of_liars : 
  (∃ n, ∃ m, (n = 450) ∨ (m = 150)) :=
sorry

end number_of_liars_l518_518643


namespace correct_answers_l518_518111

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
noncomputable def g (x : ℝ) : ℝ := f' x

-- Conditions
axiom f_domain : ∀ x, f x ∈ ℝ
axiom f'_domain : ∀ x, f' x ∈ ℝ
axiom g_def : ∀ x, g x = f' x
axiom f_even : ∀ x, f (3 / 2 - 2 * x) = f (3 / 2 + 2 * x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- Proof Problem
theorem correct_answers : f (-1) = f 4 ∧ g (-1 / 2) = 0 :=
  by
    sorry

end correct_answers_l518_518111


namespace paint_cost_of_cube_l518_518935

def cube_side_length : ℝ := 10
def paint_cost_per_quart : ℝ := 3.20
def coverage_per_quart : ℝ := 1200
def number_of_faces : ℕ := 6

theorem paint_cost_of_cube : 
  (number_of_faces * (cube_side_length^2) / coverage_per_quart) * paint_cost_per_quart = 3.20 :=
by 
  sorry

end paint_cost_of_cube_l518_518935


namespace circumscribed_circle_radius_of_rectangle_l518_518274

theorem circumscribed_circle_radius_of_rectangle 
  (a b : ℝ) 
  (h1: a = 1) 
  (angle_between_diagonals : ℝ) 
  (h2: angle_between_diagonals = 60) : 
  ∃ R, R = 1 :=
by 
  sorry

end circumscribed_circle_radius_of_rectangle_l518_518274


namespace find_larger_number_l518_518746

-- Given conditions
def HCF : ℕ := 23
def factor1 : ℕ := 16
def factor2 : ℕ := 17

-- Definition of LCM
def LCM : ℕ := HCF * factor1 * factor2

-- Definition of the two numbers
def A : ℕ := HCF * factor1
def B : ℕ := HCF * factor2

-- Proof problem statement
theorem find_larger_number (HCF : ℕ) (factor1 : ℕ) (factor2 : ℕ) (h_HCF : HCF = 23) (h_factor1 : factor1 = 16) (h_factor2 : factor2 = 17) :
  max (HCF * factor1) (HCF * factor2) = 391 :=
by {
  rw [h_HCF, h_factor1, h_factor2],
  norm_num,
  sorry
}

end find_larger_number_l518_518746


namespace tan_identity_cos_tan_l518_518098

theorem tan_identity_cos_tan (α : ℝ) (h : (1 + tan α) / (1 - tan α) = 2016) :
  (1 / cos (2 * α)) + tan (2 * α) = 2016 :=
by
  sorry

end tan_identity_cos_tan_l518_518098


namespace min_value_ineq_l518_518875

theorem min_value_ineq (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 3 * y = 1) :
  (1 / x) + (1 / (3 * y)) ≥ 4 :=
  sorry

end min_value_ineq_l518_518875


namespace volumes_equal_l518_518702

-- Define the conditions for V1
def V1_region (x y : ℝ) : Prop :=
  (x^2 = 4 * y ∨ x^2 = -4 * y) ∧ (-4 ≤ x ∧ x ≤ 4)

-- Define the conditions for V2
def V2_region (x y : ℝ) : Prop :=
  (x^2 * y^2 ≤ 16) ∧ (x^2 + (y - 2)^2 ≥ 4) ∧ (x^2 + (y + 2)^2 ≥ 4)

-- To be proved: V1 = V2
theorem volumes_equal : 
  let V1 := solid_of_revolution V1_region
  let V2 := solid_of_revolution V2_region
  V1 = V2 :=
by sorry

end volumes_equal_l518_518702


namespace circumcenter_proof_l518_518280

noncomputable def circumcenter_of_triangle 
  (Z Z1 Z2 Z3 : ℂ) 
  (h1 : |Z - Z1| = |Z - Z2|)
  (h2 : |Z - Z2| = |Z - Z3|) : Prop :=
Z = circumcenter(Z1, Z2, Z3)

theorem circumcenter_proof
  (Z Z1 Z2 Z3 : ℂ)
  (h1 : |Z - Z1| = |Z - Z2|)
  (h2 : |Z - Z2| = |Z - Z3|) : circumcenter_of_triangle Z Z1 Z2 Z3 h1 h2 :=
sorry

end circumcenter_proof_l518_518280


namespace balls_in_boxes_l518_518508

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l518_518508


namespace equation_of_line_passing_through_center_length_of_chord_with_slope_one_l518_518850

section GeometricProofs

-- Define the circle with center (1, 0) and radius 3
def Circle (O : Point) (r : ℝ) : Set Point :=
  { P | dist O P = r }

-- Define a specific instance of the circle with center (1, 0) and radius 3
def circleC : Set Point := Circle (1, 0) 3

-- Define point P
def P : Point := (2, 2)

-- Define the line passing through P and a variable point Q
def Line (P Q : Point) : Set Point :=
  { R | collinear P Q R }

-- Lean proof statement to verify the equation of line l
theorem equation_of_line_passing_through_center : 
  ∀ (l : Set Point), l = Line P (1,0) → 
  ∃ (a b c : ℝ), l = { (x, y) | a * x + b * y + c = 0 } ∧ a = 2 ∧ b = -1 ∧ c = -2 :=
sorry

-- Define the slope 45 degrees line passing through P
def line_with_slope_one : Set Point := { (x, y) | x - y = 0 }

-- Lean proof statement to verify the length of chord AB
theorem length_of_chord_with_slope_one : 
  ∃ (A B : Point), A ∈ circleC ∧ B ∈ circleC ∧ A ≠ B ∧ collinear P A B → 
  dist A B = sqrt 34 :=
sorry

end GeometricProofs

end equation_of_line_passing_through_center_length_of_chord_with_slope_one_l518_518850


namespace biker_bob_initial_west_distance_l518_518018

noncomputable def initial_distance_west (distance_AB : ℝ) (north1 : ℝ) (east : ℝ) (north2 : ℝ) : ℝ :=
  let x := Real.sqrt ((distance_AB)^2 - (north1 + north2)^2)
  x

theorem biker_bob_initial_west_distance :
  let distance_AB := 20.615528128088304
  let north1 := 5
  let east := 5
  let north2 := 15
  initial_distance_west distance_AB north1 east north2 ≈ 5.067 :=
by
  sorry

end biker_bob_initial_west_distance_l518_518018


namespace molecular_weight_of_1_mole_l518_518725

theorem molecular_weight_of_1_mole (W_5 : ℝ) (W_1 : ℝ) (h : 5 * W_1 = W_5) (hW5 : W_5 = 490) : W_1 = 490 :=
by
  sorry

end molecular_weight_of_1_mole_l518_518725


namespace program_output_l518_518758

theorem program_output (x : ℝ) (h : x = Real.sqrt 3 - 2) : 
  let y := Real.sqrt (x ^ 2) - 2 in
  y = -Real.sqrt 3 :=
by
  have hx : x = Real.sqrt 3 - 2 := h
  let y := Real.sqrt (x ^ 2) - 2
  have hsq : x ^ 2 = (Real.sqrt 3 - 2) ^ 2 := by rw hx
  have hsimp : (Real.sqrt 3 - 2) ^ 2 = 3 - 4 * Real.sqrt 3 + 4 := by sorry
  have hsimp_sqrt : Real.sqrt (3 - 4 * Real.sqrt 3 + 4) = 2 - Real.sqrt 3 := by sorry
  show y = -Real.sqrt 3 from sorry

end program_output_l518_518758


namespace num_ways_to_put_5_balls_into_4_boxes_l518_518529

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l518_518529


namespace area_of_figure_XYZ_is_50pi_l518_518713

theorem area_of_figure_XYZ_is_50pi (r : ℝ) (θ1 θ2 θ3 : ℝ) (h1 : r = 10) (h2 : θ1 = 90) (h3 : θ2 = 60) (h4 : θ3 = 30) : 
  let area := (θ1 / 360 * r^2 * Real.pi) + (θ2 / 360 * r^2 * Real.pi) + (θ3 / 360 * r^2 * Real.pi)
  in area = 50 * Real.pi :=
by
  sorry

end area_of_figure_XYZ_is_50pi_l518_518713


namespace island_liars_l518_518650

theorem island_liars (n : ℕ) (h₁ : n = 450) (h₂ : ∀ (i : ℕ), i < 450 → 
  ∃ (a : bool),  (if a then (i + 1) % 450 else (i + 2) % 450) = "liar"):
    (n = 150 ∨ n = 450) :=
sorry

end island_liars_l518_518650


namespace intersection_empty_l518_518604

open Set Int

noncomputable def floor (x : ℝ) : ℤ := Real.floor x

def A : Set ℝ := {x | floor x ^ 2 - 2 * floor x = 3}
def B : Set ℝ := {x | 2 ^ x > 8}

theorem intersection_empty : A ∩ B = ∅ := 
sorry

end intersection_empty_l518_518604


namespace solve_a_l518_518822

theorem solve_a (a : ℝ) :
  (∀ n : ℕ+, 4 * (floor (a * n)) = n + floor (a * (floor (a * n)))) → a = 2 + Real.sqrt 3 := 
by 
  sorry

end solve_a_l518_518822


namespace candy_bar_split_l518_518706
noncomputable def split (total: ℝ) (people: ℝ): ℝ := total / people

theorem candy_bar_split: split 5.0 3.0 = 1.67 :=
by
  sorry

end candy_bar_split_l518_518706


namespace jose_share_of_profit_l518_518288

/-- Given:
- Tom invested Rs. 30,000 for 12 months.
- Jose invested Rs. 45,000 for 10 months.
- Total profit after one year is Rs. 45,000.

We need to prove that Jose's share of the profit is Rs. 25,000.
-/
theorem jose_share_of_profit :
  let tom_investment_months := 30000 * 12,
      jose_investment_months := 45000 * 10,
      total_investment_months := tom_investment_months + jose_investment_months,
      total_profit := 45000,
      jose_share := (jose_investment_months / total_investment_months) * total_profit 
  in
  jose_share = 25000 :=
by
  sorry

end jose_share_of_profit_l518_518288


namespace not_correct_figure_l518_518435

-- Definitions based on conditions
def condition1 (x y : ℝ) : Prop :=
  abs x + abs y ≤ (3 / 2) * real.sqrt (2 * (x^2 + y^2))

def condition2 (x y : ℝ) : Prop :=
  real.sqrt (2 * (x^2 + y^2)) ≤ 3 * max (abs x) (abs y)

-- Representation of figures
inductive Figure
| I | II | III | IV

-- Define the statement to prove
theorem not_correct_figure :
  ¬ (∃ fig : Figure, ∀ x y : ℝ, condition1 x y ∧ condition2 x y ↔ fig = Figure.I ∨ fig = Figure.II ∨ fig = Figure.III ∨ fig = Figure.IV) :=
begin
  sorry
end

end not_correct_figure_l518_518435


namespace no_valid_solution_for_equation_l518_518662

theorem no_valid_solution_for_equation (x : ℝ) (hx : x ≠ 2) : 
  x + (5 / (x - 2)) = 2 + (5 / (x - 2)) → false :=
begin
  assume h,
  have h_eq : x = 2, {
    ring_nf at h,
    exact h
  },
  have : false,
  { exact hx h_eq },
  exact this
end

end no_valid_solution_for_equation_l518_518662


namespace horner_v4_at_2_l518_518023

def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner (x : ℝ) : List ℝ → ℝ
| []      := 0
| (c::cs) := c + x * horner x cs

theorem horner_v4_at_2 :
  horner 2 [-12, 60, -160, 240, -192, 64] = 240 :=
by
  sorry

end horner_v4_at_2_l518_518023


namespace overtime_pay_rate_ratio_l518_518768

noncomputable def regular_pay_rate : ℕ := 3
noncomputable def regular_hours : ℕ := 40
noncomputable def total_pay : ℕ := 180
noncomputable def overtime_hours : ℕ := 10

theorem overtime_pay_rate_ratio : 
  (total_pay - (regular_hours * regular_pay_rate)) / overtime_hours / regular_pay_rate = 2 := by
  sorry

end overtime_pay_rate_ratio_l518_518768


namespace probability_red_half_red_balls_taken_out_is_six_l518_518171
noncomputable theory

-- Define the initial number of balls.
def initial_red_balls := 10
def initial_yellow_balls := 2
def initial_blue_balls := 8
def total_balls := initial_red_balls + initial_yellow_balls + initial_blue_balls -- This should be 20

-- Part 1: Probability of drawing a red ball
def probability_red : ℚ := initial_red_balls / total_balls

-- Part 2: After some red balls are replaced with yellow balls
def replaced_red_balls (x : ℕ) := initial_red_balls - x
def replaced_yellow_balls (x : ℕ) := initial_yellow_balls + x

-- Given the probability of drawing a yellow ball is 2/5 after replacement
def probability_yellow_after_replacement (x : ℕ) : Prop :=
  (replaced_yellow_balls x : ℚ) / total_balls = 2 / 5

-- The number of red balls taken out (x) is 6
def number_of_red_balls_taken_out : ℕ :=
  ∃ x : ℕ, probability_yellow_after_replacement x ∧ x = 6

-- Prove that the probability of drawing a red ball initially is 1/2
theorem probability_red_half : probability_red = 1 / 2 := sorry

-- Prove that the number of red balls taken out to make the probability of drawing a yellow ball 2/5 is 6
theorem red_balls_taken_out_is_six : number_of_red_balls_taken_out := sorry

end probability_red_half_red_balls_taken_out_is_six_l518_518171


namespace possible_values_l518_518867

theorem possible_values
  (x : ℝ)
  (h : 2 * real.cos x - 5 * real.sin x = 2) :
  real.sin x + 2 * real.cos x = 2 ∨ real.sin x + 2 * real.cos x = -62 / 29 :=
begin
  sorry
end

end possible_values_l518_518867


namespace influence_function_l518_518055

-- Definitions based on the provided conditions
def bending_moment (x y : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ y then -(1 - y) * x
else if h : y ≤ x ∧ x ≤ 1 then -y * (1 - x)
else 0

def F : (ℝ → ℝ) := λ x, 1 / (E_x * I_x : ℝ)

-- Boundary condition
def G (x y : ℝ) := ∫ z in 0..1, (bending_moment x z) * (bending_moment z y) * (F z)

theorem influence_function (x y : ℝ) :
  G(x, y) = ∫ z in 0..1, (bending_moment x z) * (bending_moment z y) * (F z) :=
sorry

end influence_function_l518_518055


namespace distance_to_angle_bisector_l518_518552

theorem distance_to_angle_bisector 
  (P : ℝ × ℝ) 
  (h_hyperbola : P.1^2 - P.2^2 = 9) 
  (h_distance_to_line_neg_x : abs (P.1 + P.2) = 2016 * Real.sqrt 2) : 
  abs (P.1 - P.2) / Real.sqrt 2 = 448 :=
sorry

end distance_to_angle_bisector_l518_518552


namespace perfect_square_condition_l518_518601

theorem perfect_square_condition (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
    (gcd_xyz : Nat.gcd (Nat.gcd x y) z = 1)
    (hx_dvd : x ∣ y * z * (x + y + z))
    (hy_dvd : y ∣ x * z * (x + y + z))
    (hz_dvd : z ∣ x * y * (x + y + z))
    (sum_dvd : x + y + z ∣ x * y * z) :
  ∃ m : ℕ, m * m = x * y * z * (x + y + z) := sorry

end perfect_square_condition_l518_518601


namespace exponent_combination_l518_518928

theorem exponent_combination (a : ℝ) (m n : ℕ) (h₁ : a^m = 3) (h₂ : a^n = 4) :
  a^(2 * m + 3 * n) = 576 :=
by
  sorry

end exponent_combination_l518_518928


namespace sector_area_l518_518103

def sector (α R : ℝ) : ℝ := (1 / 2) * α * R^2

theorem sector_area (α R S : ℝ) (hα : α = 2 * Real.pi / 3) (hR : R = Real.sqrt 3) (hS : S = Real.pi) : 
  sector α R = S :=
by
  sorry

end sector_area_l518_518103


namespace complement_intersection_l518_518866

open Set

variable (R : Type) [LinearOrderedField R]

def A : Set R := {x | |x| < 1}
def B : Set R := {y | ∃ x, y = 2^x + 1}
def complement_A : Set R := {x | x ≤ -1 ∨ x ≥ 1}

theorem complement_intersection (x : R) : 
  x ∈ (complement_A R) ∩ B R ↔ x > 1 :=
by
  sorry

end complement_intersection_l518_518866


namespace seq_product_l518_518780

noncomputable def seq (n : ℕ) : ℝ :=
  if n = 0 then 2 / 3 else 1 + (seq (n - 1) - 1)^2

theorem seq_product : (∏ n : ℕ, seq n) = 3 / 4 := 
  sorry

end seq_product_l518_518780


namespace total_cupcakes_l518_518323

-- Definitions of initial conditions
def cupcakes_initial : ℕ := 42
def cupcakes_sold : ℕ := 22
def cupcakes_made_after : ℕ := 39

-- Proof statement: Total number of cupcakes Robin would have
theorem total_cupcakes : 
  (cupcakes_initial - cupcakes_sold + cupcakes_made_after) = 59 := by
    sorry

end total_cupcakes_l518_518323


namespace find_even_smallest_period_l518_518007

-- Define the functions
def f_A (x : ℝ) : ℝ := Real.sin (2 * x)
def f_B (x : ℝ) : ℝ := Real.cos x
def f_C (x : ℝ) : ℝ := Real.tan x
def f_D (x : ℝ) : ℝ := abs (Real.tan x)

-- Define the properties to check
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f x = f (x + p)

-- State the main theorem
theorem find_even_smallest_period :
  (is_even f_A ∧ has_period f_A π) ∨
  (is_even f_B ∧ has_period f_B π) ∨
  (is_even f_C ∧ has_period f_C π) ∨
  (is_even f_D ∧ has_period f_D π ∧
   ∀ (f : ℝ → ℝ), is_even f → has_period f π → f = f_D) :=
begin
  -- Solution omitted
  sorry
end

end find_even_smallest_period_l518_518007


namespace difference_area_octagon_shaded_l518_518015

-- Definitions based on the given conditions
def radius : ℝ := 10
def pi_value : ℝ := 3.14

-- Lean statement for the given proof problem
theorem difference_area_octagon_shaded :
  ∃ S_octagon S_shaded, 
    10^2 * pi_value = 314 ∧
    (20 / 2^0.5)^2 = 200 ∧
    S_octagon = 200 - 114 ∧ -- transposed to reverse engineering step
    S_shaded = 28 ∧ -- needs refinement
    S_octagon - S_shaded = 86 :=
sorry

end difference_area_octagon_shaded_l518_518015


namespace probability_larry_wins_l518_518591

noncomputable def P_larry_wins_game : ℝ :=
  let p_hit := (1 : ℝ) / 3
  let p_miss := (2 : ℝ) / 3
  let r := p_miss^3
  (p_hit / (1 - r))

theorem probability_larry_wins :
  P_larry_wins_game = 9 / 19 :=
by
  -- Proof is omitted, but the outline and logic are given in the problem statement
  sorry

end probability_larry_wins_l518_518591


namespace necessary_and_sufficient_condition_l518_518091

-- Sum of the first n terms of the sequence
noncomputable def S_n (n : ℕ) (c : ℤ) : ℤ := (n + 1) * (n + 1) + c

-- The nth term of the sequence
noncomputable def a_n (n : ℕ) (c : ℤ) : ℤ := S_n n c - (S_n (n - 1) c)

-- Define the sequence being arithmetic
noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) - a n = d

theorem necessary_and_sufficient_condition (c : ℤ) :
  (∀ n ≥ 1, a_n n c - a_n (n-1) c = 2) ↔ (c = -1) :=
by
  sorry

end necessary_and_sufficient_condition_l518_518091


namespace three_points_area_leq_quarter_l518_518955

theorem three_points_area_leq_quarter {A B C : ℝ×ℝ} (hArea : (triangle_area A B C) = 1)
  (P1 P2 P3 P4 P5 : ℝ×ℝ) (hPoints : inside_triangle A B C P1 ∧ inside_triangle A B C P2 ∧ 
  inside_triangle A B C P3 ∧ inside_triangle A B C P4 ∧ inside_triangle A B C P5) :
  ∃ (X Y Z : ℝ×ℝ), {X, Y, Z} ⊆ {P1, P2, P3, P4, P5} ∧ (triangle_area X Y Z) ≤ (1 / 4) := 
sorry

end three_points_area_leq_quarter_l518_518955


namespace differences_occur_10_times_l518_518599

variable (a : Fin 45 → Nat)

theorem differences_occur_10_times 
    (h : ∀ i j : Fin 44, i < j → a i < a j)
    (h_lt_125 : ∀ i : Fin 44, a i < 125) :
    ∃ i : Fin 43, ∃ j : Fin 43, i ≠ j ∧ (a (i + 1) - a i) = (a (j + 1) - a j) ∧ 
    (∃ k : Nat, k ≥ 10 ∧ (a (j + 1) - a j) = (a (k + 1) - a k)) :=
sorry

end differences_occur_10_times_l518_518599


namespace dice_probability_l518_518716

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def p_one_of_twelve_is_1 : ℝ :=
  let p := 1 / 6
  let q := 5 / 6
  let n := 12
  let k := 1
  (binomial n k : ℝ) * (p : ℝ)^k * (q : ℝ)^(n-k)

theorem dice_probability : abs (p_one_of_twelve_is_1 - 0.261) < 0.001 := sorry

end dice_probability_l518_518716


namespace balls_in_boxes_l518_518507

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l518_518507


namespace Jimin_addition_l518_518586

theorem Jimin_addition (x : ℕ) (h : 96 / x = 6) : 34 + x = 50 := 
by
  sorry

end Jimin_addition_l518_518586


namespace quadrilateral_similarity_condition_l518_518689

variable (A B C D K L M N P Q R S : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variable [metric_space K] [metric_space L] [metric_space M] [metric_space N]
variable [metric_space P] [metric_space Q] [metric_space R] [metric_space S]
variable (mid_point : ∀ {X Y : Type} [metric_space X] [metric_space Y], X → Y → (X → Y → Type))

-- Definitions for midpoints
def is_midpoint_of {X Y : Type} [metric_space X] [metric_space Y] (P : X → Y → Type) (x₁ x₂ x₃ : X) :=
  (P x₁ x₂ = x₃)

-- The statement
theorem quadrilateral_similarity_condition : 
  (∀ {X Y : Type} [metric_space X] [metric_space Y], 
    (is_midpoint_of mid_point A B K) ∧ (is_midpoint_of mid_point B C L) ∧ 
    (is_midpoint_of mid_point C D M) ∧ (is_midpoint_of mid_point D A N) ∧ 
    (is_midpoint_of mid_point K L P) ∧ (is_midpoint_of mid_point L M Q) ∧ 
    (is_midpoint_of mid_point M N R) ∧ (is_midpoint_of mid_point N K S)) 
  →
  ((∃ (a b e f : ℝ), e = a * sqrt 2 ∧ f = b * sqrt 2) ↔ 
    similar P Q R S K L M N) := 
sorry

end quadrilateral_similarity_condition_l518_518689


namespace part_a_l518_518178

variable {a b x : ℝ}

-- Conditions described
-- Rectangle setup with AE and BF on AB such that AE = BF < BE and EF = x
def rectangle_ABCD (a b x : ℝ) := 
  a > 0 ∧ b > 0 ∧ 0 < x

-- Points G and H on CD such that CG = DH = EF = x
-- Define the ratio
def ratio_AE_AB (a x : ℝ) := 
  let q := (a - x) / (2 * a) in 
  q > 1/4 ∧ q < 1/2

-- Full setup implies ratio constraint, using implication 
theorem part_a {a b x : ℝ} (h : rectangle_ABCD a b x) : 
  ratio_AE_AB a x := by
  sorry

end part_a_l518_518178


namespace num_ways_to_put_5_balls_into_4_boxes_l518_518527

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l518_518527


namespace arc_length_polar_l518_518020

theorem arc_length_polar :
  (∫ φ in (0:Real)..(π/3), 
   sqrt ((5 * exp (5 * φ / 12))^2 + (deriv (λ φ, 5 * exp (5 * φ / 12)) φ)^2)) 
  = 13 * (exp (5 * π / 36) - 1) :=
by
  sorry

end arc_length_polar_l518_518020


namespace find_other_digits_l518_518692

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem find_other_digits (n : ℕ) (h : ℕ) :
  tens_digit n = h →
  h = 1 →
  is_divisible_by_9 n →
  ∃ m : ℕ, m < 9 ∧ n = 10 * ((n / 10) / 10) * 10 + h * 10 + m ∧ (∃ k : ℕ, k * 9 = h + m + (n / 100)) :=
sorry

end find_other_digits_l518_518692


namespace ball_box_problem_l518_518499

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l518_518499


namespace systematic_sampling_max_label_l518_518844

theorem systematic_sampling_max_label (N n : ℕ) (k : ℕ) (j : ℕ) (sample : Finset ℕ)
  (hN : N = 80) 
  (hn : n = 5) 
  (hk : k = N / n)
  (hmod : j = (10 % k) + 1)
  (hin_sample : 10 ∈ sample)
  (h_sample : sample.card = n ∧ (∀ i : ℕ, i ∈ sample → (10 + (i - j) * k) ∈ sample)) :
  (sample.max id) = 74 :=
by
  -- Statement but no proof
  sorry

end systematic_sampling_max_label_l518_518844


namespace systematic_sampling_methods_l518_518008

-- Definitions for sampling methods ①, ②, ④
def sampling_method_1 : Prop :=
  ∀ (l : ℕ), (l ≤ 15 ∧ l + 5 ≤ 15 ∧ l + 10 ≤ 15 ∨
              l ≤ 15 ∧ l + 5 ≤ 20 ∧ l + 10 ≤ 20) → True

def sampling_method_2 : Prop :=
  ∀ (t : ℕ), (t % 5 = 0) → True

def sampling_method_3 : Prop :=
  ∀ (n : ℕ), (n > 0) → True

def sampling_method_4 : Prop :=
  ∀ (row : ℕ) (seat : ℕ), (seat = 12) → True

-- Equivalence Proof Statement
theorem systematic_sampling_methods :
  sampling_method_1 ∧ sampling_method_2 ∧ sampling_method_4 :=
by sorry

end systematic_sampling_methods_l518_518008


namespace double_root_divisors_l518_518350

theorem double_root_divisors (b_4 b_3 b_2 b_1 s : ℤ)
  (P : ℤ → ℤ) 
  (hP : P = λ x, x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + 72)
  (hs : (x - s)^2 ∣ P x) :
  s = -6 ∨ s = -3 ∨ s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 6 :=
sorry

end double_root_divisors_l518_518350


namespace compute_k_plus_m_l518_518202

theorem compute_k_plus_m :
  ∃ k m : ℝ, 
    (∀ (x y z : ℝ), x^3 - 9 * x^2 + k * x - m = 0 -> x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 9 ∧ 
    (x = 1 ∨ y = 1 ∨ z = 1) ∧ (x = 3 ∨ y = 3 ∨ z = 3) ∧ (x = 5 ∨ y = 5 ∨ z = 5)) →
    k + m = 38 :=
by
  sorry

end compute_k_plus_m_l518_518202


namespace intersection_A_B_union_complement_A_B_l518_518865

noncomputable theory

-- Define set A
def set_A : Set ℝ := {x | x^2 + 2 * x - 8 ≤ 0}

-- Define set B
def set_B : Set ℝ := {x | 3^x ≥ 1 / 3}

-- Define the complements in ℝ
def complement_in_R (S : Set ℝ) : Set ℝ := {x | x ∉ S}

-- Statement for the intersection
theorem intersection_A_B : (set_A ∩ set_B) = {x | -1 ≤ x ∧ x ≤ 2} := by
  sorry -- To be completed

-- Statement for the union of complements and B
theorem union_complement_A_B : (complement_in_R set_A ∪ set_B) = 
  {x | x < -4} ∪ {x | -1 ≤ x} := by
  sorry -- To be completed

end intersection_A_B_union_complement_A_B_l518_518865


namespace arithmetic_log_sum_l518_518442

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 1 * a 9 = a 2 * a 8 ∧ 
  a 2 * a 8 = 4 ∧ 
  ∀ n, a n > 0

theorem arithmetic_log_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  (Real.logBase 2 (a 1) + Real.logBase 2 (a 2) + Real.logBase 2 (a 3) + Real.logBase 2 (a 4) + Real.logBase 2 (a 5) +
   Real.logBase 2 (a 6) + Real.logBase 2 (a 7) + Real.logBase 2 (a 8) + Real.logBase 2 (a 9) = 9) :=
sorry

end arithmetic_log_sum_l518_518442


namespace number_of_liars_l518_518645

constant islanders : Type
constant knight : islanders → Prop
constant liar : islanders → Prop
constant sits_at_table : islanders → Prop
constant right_of : islanders → islanders

axiom A1 : ∀ x : islanders, sits_at_table x → (knight x ∨ liar x)
axiom A2 : (∃ n : ℕ, n = 450 ∧ (λ x, sits_at_table x))
axiom A3 : ∀ x : islanders, sits_at_table x →
  (liar (right_of x) ∧ ¬ liar (right_of (right_of x))) ∨ 
  (¬ liar (right_of x) ∧ liar (right_of (right_of x)))

theorem number_of_liars : 
  (∃ n, ∃ m, (n = 450) ∨ (m = 150)) :=
sorry

end number_of_liars_l518_518645


namespace nicholas_bottle_caps_l518_518631

theorem nicholas_bottle_caps (N : ℕ) (h : N + 85 = 93) : N = 8 :=
by
  sorry

end nicholas_bottle_caps_l518_518631


namespace min_elements_in_symmetricDifference_arithmetic_sequence_l518_518420

open Set

noncomputable def A (n : ℕ) (a : ℕ → ℕ) : Set ℕ :=
  {a i | 1 ≤ i ∧ i ≤ n}

noncomputable def B (n : ℕ) (a : ℕ → ℕ) : Set ℕ :=
  {a i + 2 * a j | ∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i ≠ j}

noncomputable def symmetricDifference {α : Type*} (S T : Set α) : Set α :=
  (S ∪ T) \ (S ∩ T)

theorem min_elements_in_symmetricDifference_arithmetic_sequence:
  ∀ (a : ℕ → ℕ) (n : ℕ) (d : ℕ), 2 < n →
  (∀ i j, i < j → a i < a j) →
  (∀ i, a i = a 1 + (i - 1) * d) →
  (if n ≥ 4 then Finset.card (symmetricDifference (A n a) (B n a)) = 2 * n
   else Finset.card (symmetricDifference (A n a) (B n a)) = 5) := 
  sorry

end min_elements_in_symmetricDifference_arithmetic_sequence_l518_518420


namespace find_digits_in_equation_l518_518409

theorem find_digits_in_equation :
  ∃ (x y z : ℕ), 
    x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    y ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    z ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    (10 * x + 5) * (300 + 10 * y + z) = 7850 ∧ 
    x = 2 ∧ 
    y = 1 ∧ 
    z = 4 :=
by {
  sorry
}

end find_digits_in_equation_l518_518409


namespace maximal_meeting_days_l518_518976

open Function

-- Definition of the problem conditions.
def is_valid_swap (perm : List Nat) (initial_adj : List (Nat × Nat)) : Prop :=
  ∀ (i : Nat) (j : Nat), (j = i + 1 ∨ i = j + 1) → (i, j) ∉ initial_adj → (swap perm i j).perm = perm

noncomputable def initial_neighbours (N : Nat) : List (Nat × Nat) :=
  List.zip (List.range N) ((List.range N).rotate 1)

def max_meeting_days (N : Nat) : Nat :=
  N

-- Statement that needs to be proved.
theorem maximal_meeting_days (N : Nat) : ∃ (days : Nat), days = max_meeting_days N := 
  sorry -- Proof of the theorem.

end maximal_meeting_days_l518_518976


namespace estimated_value_of_n_l518_518561

-- Definitions from the conditions of the problem
def total_balls (n : ℕ) : ℕ := n + 18 + 9
def probability_of_yellow (n : ℕ) : ℚ := 18 / total_balls n

-- The theorem stating what we need to prove
theorem estimated_value_of_n : ∃ n : ℕ, probability_of_yellow n = 0.30 ∧ n = 42 :=
by {
  sorry
}

end estimated_value_of_n_l518_518561


namespace max_unique_numbers_l518_518395

theorem max_unique_numbers (students : ℕ) (numbers_per_student : ℕ) (min_students_per_number : ℕ) :
  students = 10 → numbers_per_student = 5 → min_students_per_number = 3 →
  ∃ (max_unique_numbers : ℕ), max_unique_numbers = 16 ∧
    ∀ (total_numbers : ℕ), total_numbers = students * numbers_per_student →
                           ∀ (unique_numbers : ℕ), unique_numbers <= total_numbers / min_students_per_number →
                                                     unique_numbers <= max_unique_numbers :=
by {
  intros h_students h_numbers_per_student h_min_students_per_number,
  use 16,
  split,
  {
    refl, -- max_unique_numbers = 16 is the correct answer
  },
  {
    intros total_numbers h_total_numbers unique_numbers h_unique_numbers,
    sorry
  }
}

end max_unique_numbers_l518_518395


namespace max_citizens_l518_518211

theorem max_citizens (n : ℕ) (h : Nat.choose n 4 < Nat.choose n 2) : n ≤ 5 :=
by
  have h₁ : Nat.choose n 4 = n * (n-1) * (n-2) * (n-3) / 24 := sorry
  have h₂ : Nat.choose n 2 = n * (n-1) / 2 := sorry
  have h₃ : n * (n-1) * (n-2) * (n-3) / 24 < n * (n-1) / 2 := by 
    rw [h₁, h₂]
    exact h
  let d := λ n => n * (n-1)
  have h₄ : ∀ n, d (n-2) * (n-3) / 12 < 1 := by
    intro n
    simp [d, mul_assoc, div_lt_iff, zero_lt_bit0, zero_lt_bit0, zero_lt_bit0]
    exact (d (n-2) * (n-3))/12 < 1
  sorry

end max_citizens_l518_518211


namespace sum_g_2024_l518_518459

def f (x : ℝ) := sorry
def f' (x : ℝ) := sorry
def g (x : ℝ) := f' x

-- Symmetry conditions
axiom f_symmetric : ∀ x : ℝ, f (x + 3) = -f (-x + 3)
axiom g_even_on_transformation : ∀ x : ℝ, g (2 * x + 3 / 2) = g (-2 * x + 3 / 2)
axiom g_values : g 1 = 2 ∧ g 3 = -3

theorem sum_g_2024 : ∑ k in Finset.range 2024, g (k : ℝ) = 678 := by
  sorry

end sum_g_2024_l518_518459


namespace fraction_product_eq_six_l518_518718

theorem fraction_product_eq_six : (2/5) * (3/4) * (1/6) * (120 : ℚ) = 6 := by
  sorry

end fraction_product_eq_six_l518_518718


namespace angle_in_fourth_quadrant_l518_518924

theorem angle_in_fourth_quadrant (α : ℝ) (h : 0 < α ∧ α < 90) : 270 < 360 - α ∧ 360 - α < 360 :=
by
  sorry

end angle_in_fourth_quadrant_l518_518924


namespace sequence_is_geometric_iff_b_eq_1_and_k_eq_1_l518_518610

-- Definitions and conditions
def b1 (b : ℕ) : ℕ := b
def b2 (b k : ℕ) : ℕ := k * b
def b (b k : ℕ) : ℕ → ℕ
| 0       := b1 b
| 1       := b2 b k
| (n + 2) := b n * b (n + 1)

-- Geometric progression definition
def is_geometric_progression (b : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, b(n+1) = r * b n

-- Main theorem statement
theorem sequence_is_geometric_iff_b_eq_1_and_k_eq_1 (b k : ℕ) :
  (is_geometric_progression (b b k) ↔ (k = 1 ∧ b = 1)) :=
sorry

end sequence_is_geometric_iff_b_eq_1_and_k_eq_1_l518_518610


namespace med_circumcenters_circle_l518_518268

variables {A B C: Type*} [metric_space A] [metric_space B] [metric_space C]
variables [triangle ABC]  -- consider ABC as a triangle in metric space

-- The medians
def medians (A B C : triangle ABC) : set point := {
  AA1: point, BB1: point, CC1: point | 
  AA1 = midpoint(deleted_point, B, C) ∧
  BB1 = midpoint(deleted_point, A, C) ∧
  CC1 = midpoint(deleted_point, A, B)
}

-- Define the centroid of the triangle
def centroid (A B C : triangle ABC) : point := {
  M: point | M = (A + B + C) / 3
}

-- Define circumcenters of triangles formed by medians
def circumcenters (A M AA1 BB1 CC1: point) : set point := {
  A_plus: point, B_minus: point, C_plus: point, 
  A_minus: point, B_plus: point, C_minus: point | 
  A_plus = circumcenter(BB1, M, C) ∧
  B_minus = circumcenter(C, M, AA1) ∧ 
  C_plus = circumcenter(AA1, M, B) ∧ 
  A_minus = circumcenter(B, M, CC1) ∧ 
  B_plus = circumcenter(CC1, M, A) ∧
  C_minus = circumcenter(A, M, BB1)
}

-- Statement of the theorem
theorem med_circumcenters_circle (
  A B C : point,
  h_medians : medians A B C,
  h_centroid : centroid A B C,
  h_circumcenters : circumcenters A M AA1 BB1 CC1
): ∃ O R, Circle({A_plus, B_minus, C_plus, A_minus, B_plus, C_minus} O R) :=
begin
  sorry
end

end med_circumcenters_circle_l518_518268


namespace measure_segment_PQ_l518_518574

variables {P Q R S : Type} [segment : P → Q → R → Type]

def parallel (a b c d : P → Prop) : Prop := ∃ m, (a b = m * c d)

axiom angle_condition (Q S : ℝ) : ∃ beta : ℝ, S = 3 * beta ∧ Q = beta
axiom segment_lengths (PS RS : ℝ) : PS = c ∧ RS = d

theorem measure_segment_PQ (P Q R S : Type) [segment : P → Q → R → Type] 
  (PQ_parallel_RS : parallel PQ RS) 
  (angle_condition : ∃ beta : ℝ, ∠S = 3 * beta ∧ ∠Q = beta)
  (segment_lengths : PS = c ∧ RS = d) :
  PQ = c + d :=
by
  sorry

end measure_segment_PQ_l518_518574


namespace function_properties_l518_518437

/-- Definition of the function y = f(x) with given conditions and period --/
def periodic_function (x : ℝ) : ℝ := 
  if h : 1 ≤ x ∧ x ≤ 4 then 2 * (x - 2) ^ 2 - 5
  else if h : -1 ≤ x ∧ x < 1 then -3 * x
  else 0 -- placeholder for full periodic extension

/-- Lean 4 theorem expressing that f(x) matches the given function under the aforementioned conditions --/
theorem function_properties :
  (∀ x : ℝ, periodic_function (x + 5) = periodic_function x) ∧
  (∀ x : ℝ, x ∈ Icc (-1) 1 → periodic_function x = -3 * x) ∧
  (∀ x : ℝ, x ∈ Icc 1 4 → periodic_function(x) = 2 * (x - 2) ^ 2 - 5) :=
by
  sorry

end function_properties_l518_518437


namespace volume_cone_maximum_volume_prism_l518_518355

noncomputable def volume_of_cone (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

noncomputable def maximum_volume_rectangular_prism (a b c : ℝ) : ℝ :=
  a * b * c

theorem volume_cone (h : ℝ) (r : ℝ) (h_cond : h = 1) (r_cond : r = 0.5) :
  volume_of_cone r h = 0.5236 := 
by
  sorry

theorem maximum_volume_prism (a b c : ℝ) (area_cond : 2 * (a * b + b * c + c * a) = 1) :
  maximum_volume_rectangular_prism a b c = 0.1925 := 
by 
  sorry

end volume_cone_maximum_volume_prism_l518_518355


namespace length_of_chord_AB_l518_518469

-- Definitions and conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + y^2 = 1
def left_focus : (ℝ × ℝ) := (-2 * Real.sqrt 2, 0)
def inclination_angle : ℝ := Real.pi / 6
def line_through_focus (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x + 2 * Real.sqrt 2)

-- The proof statement
theorem length_of_chord_AB : 
  ∀ x1 x2 y1 y2 : ℝ, 
  ellipse x1 y1 → ellipse x2 y2 → 
  line_through_focus x1 y1 → line_through_focus x2 y2 → 
  ∥(x1 - x2, y1 - y2)∥ = 2 :=
by
  sorry

end length_of_chord_AB_l518_518469


namespace infinite_solutions_if_b_eq_neg_12_l518_518840

theorem infinite_solutions_if_b_eq_neg_12 (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by
  split
  { intro h,
    specialize h 0,
    simp at h,
    linarith },
  { intro h,
    intro x,
    rw h,
    simp }

end infinite_solutions_if_b_eq_neg_12_l518_518840


namespace num_true_statements_is_two_l518_518240

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem num_true_statements_is_two :
  let s1 := reciprocal 4 + reciprocal 8 = reciprocal 12
  let s2 := reciprocal 8 - reciprocal 5 = reciprocal 3
  let s3 := reciprocal 3 * reciprocal 9 = reciprocal 27
  let s4 := reciprocal 15 / reciprocal 3 = reciprocal 5
  (if s1 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0) = 2 :=
by
  sorry

end num_true_statements_is_two_l518_518240


namespace difference_of_smallest_integers_l518_518050

theorem difference_of_smallest_integers : 
    let m := Nat.lcm (List.range' 2 12) in
    2 * m - m = 360360 := by
  sorry

end difference_of_smallest_integers_l518_518050


namespace find_possible_k_values_l518_518255

-- Define the equation with a given k
def equation (x k : ℂ) : Prop := (x / (x + 1) + x / (x + 2) = k * x)

-- The main statement: find all k such that the equation 
-- has exactly two complex roots
theorem find_possible_k_values (k : ℂ) : 
  (∃! x : ℂ, equation x k) → k = 0 ∨ k = 3 / 2 ∨ k = 2 * complex.I ∨ k = -2 * complex.I :=
sorry -- proof is omitted

end find_possible_k_values_l518_518255


namespace equation_of_tangent_line_at_point_l518_518890

-- Define the function f(x) = 2 / x
def f (x : ℝ) : ℝ := 2 / x

-- Define the point of interest (1, 2)
def point : ℝ × ℝ := (1, 2)

-- Define the derivative of the function
noncomputable def f_prime (x : ℝ) : ℝ := -2 / x^2

-- Define the slope of the tangent line at the point (1, 2)
def slope : ℝ := f_prime 1

-- Define the equation of the tangent line
def tangent_line (x y: ℝ) : Prop := 2 * x + y - 4 = 0

-- The statement to prove
theorem equation_of_tangent_line_at_point :
  tangent_line 1 2 := by
  sorry

end equation_of_tangent_line_at_point_l518_518890


namespace number_of_valid_consecutive_sum_sets_l518_518417

-- Definition of what it means to be a set of consecutive integers summing to 225
def sum_of_consecutive_integers (n a : ℕ) : Prop :=
  ∃ k : ℕ, (k = (n * (2 * a + n - 1)) / 2) ∧ (k = 225)

-- Prove that there are exactly 4 sets of two or more consecutive positive integers that sum to 225
theorem number_of_valid_consecutive_sum_sets : 
  ∃ (sets : Finset (ℕ × ℕ)), 
    (∀ (n a : ℕ), (n, a) ∈ sets ↔ sum_of_consecutive_integers n a) ∧ 
    (2 ≤ n) ∧ 
    sets.card = 4 := sorry

end number_of_valid_consecutive_sum_sets_l518_518417


namespace least_value_b_l518_518165

-- Defining the conditions of the problem
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

variables (a b c : ℕ)

-- Conditions
axiom angle_sum : a + b + c = 180
axiom primes : is_prime a ∧ is_prime b ∧ is_prime c
axiom order : a > b ∧ b > c

-- The statement to be proved
theorem least_value_b (h : a + b + c = 180) (hp : is_prime a ∧ is_prime b ∧ is_prime c) (ho : a > b ∧ b > c) : b = 5 :=
sorry

end least_value_b_l518_518165


namespace find_angle_l518_518449

noncomputable def vector := ℝ → ℝ

variable (a b : vector)
variable (non_zero_a : (∥a 0∥ ≠ 0))
variable (non_zero_b : (∥b 0∥ ≠ 0))
variable (condition1 : ∥a 0∥ = 1 / 2 * ∥b 0∥)
variable (condition2 : dot_product (λ x, sqrt 3 * a x - b x) (a) = 0)

theorem find_angle :
  let θ := angle a b
  θ = 30 := sorry

end find_angle_l518_518449


namespace inradius_bounds_l518_518658

variables {a b c t : ℝ} {m_a m_b m_c : ℝ} {ρ : ℝ}
hypothesis h1 : a ≤ b ∧ b ≤ c
hypothesis h2 : m_a ≥ m_b ∧ m_b ≥ m_c
hypothesis h3 : t = 1/2 * a * m_a
hypothesis h4 : t = 1/2 * c * m_c
hypothesis h5 : 2 * t = ρ * (a + b + c)

theorem inradius_bounds (h1 : a ≤ b ∧ b ≤ c) (h2 : m_a ≥ m_b ∧ m_b ≥ m_c) (h3 : t = 1/2 * a * m_a) (h4 : t = 1/2 * c * m_c) (h5 : 2 * t = ρ * (a + b + c)) :
  ρ / 3 ≤ m_c / 3 ∧ ρ / 3 ≤ m_c / 3 :=
by
  sorry

end inradius_bounds_l518_518658


namespace B_finishes_work_in_4_days_l518_518312

-- Define the work rates of A and B
def work_rate_A : ℚ := 1 / 5
def work_rate_B : ℚ := 1 / 10

-- Combined work rate when A and B work together
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Work done by A and B in 2 days
def work_done_in_2_days : ℚ := 2 * combined_work_rate

-- Remaining work after 2 days
def remaining_work : ℚ := 1 - work_done_in_2_days

-- Time B needs to finish the remaining work
def time_for_B_to_finish_remaining_work : ℚ := remaining_work / work_rate_B

theorem B_finishes_work_in_4_days : time_for_B_to_finish_remaining_work = 4 := by
  sorry

end B_finishes_work_in_4_days_l518_518312


namespace probability_same_color_l518_518154

-- Define the total combinations function
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- The given values from the problem
def whiteBalls := 2
def blackBalls := 3
def totalBalls := whiteBalls + blackBalls
def drawnBalls := 2

-- Calculate combinations
def comb_white_2 := comb whiteBalls drawnBalls
def comb_black_2 := comb blackBalls drawnBalls
def comb_total_2 := comb totalBalls drawnBalls

-- The correct answer given in the solution
def correct_probability := 2 / 5

-- Statement for the proof in Lean
theorem probability_same_color : (comb_white_2 + comb_black_2) / comb_total_2 = correct_probability := by
  sorry

end probability_same_color_l518_518154


namespace islanders_liars_l518_518636

theorem islanders_liars (n : ℕ) (h : n = 450) : (∃ L : ℕ, (L = 150 ∨ L = 450)) :=
sorry

end islanders_liars_l518_518636


namespace balls_in_boxes_l518_518531

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l518_518531


namespace CowAdditionalSpots_l518_518229

/-- Ruth's cow has 16 spots on its left side and three times that number plus some additional 
spots on its right side. The cow has a total of 71 spots. 
Prove the number of additional spots on the right side is 7. -/
theorem CowAdditionalSpots : 
  ∀ (leftSpots rightSpots totalSpots additionalSpots : ℕ), 
    leftSpots = 16 → 
    rightSpots = 3 * leftSpots + additionalSpots → 
    totalSpots = leftSpots + rightSpots → 
    totalSpots = 71 → 
    additionalSpots = 7 := 
by 
  intros leftSpots rightSpots totalSpots additionalSpots
  intros h_left h_right h_total h_71
  have eq1 : 48 = 3 * leftSpots := by simp [h_left]
  have eq2 : rightSpots = 48 + additionalSpots := by simp [h_right, eq1]
  have eq3 : 71 = leftSpots + rightSpots := by simp [h_71]
  rw [h_total, eq2, h_left] at eq3
  have eq4 : 71 = 16 + 48 + additionalSpots := by simp [eq1]
  have : additionalSpots = 7 := by linarith
  exact this

end CowAdditionalSpots_l518_518229


namespace shift_right_graph_l518_518286

theorem shift_right_graph (x : ℝ) :
  (3 : ℝ)^(x+1) = (3 : ℝ)^((x+1) - 1) :=
by 
  -- Here we prove that shifting the graph of y = 3^(x+1) to right by 1 unit 
  -- gives the graph of y = 3^x
  sorry

end shift_right_graph_l518_518286


namespace area_sum_geq_l518_518184

variable (A B C A1 B1 C1 : Type)
variable [Triangle ABC]
variable (circumcircle : Circle (A, B, C))
variable (medianA : Line A1 B1)
variable (medianB : Line B C1)
variable (medianC : Line C A1)
variable (tABC tA1BC tB1CA tC1AB : ℝ)

-- Defining the conditions
-- Assume the medians are such that they intersect at the circumcircle
axiom medians_intersect_at_circumcircle (medians_intersect_circumcircle :
  (medianA.Int_circumcircle = A1) ∧ (medianB.Int_circumcircle = B1) ∧ (medianC.Int_circumcircle = C1))

-- Triangle area definitions
def t_ABC : ℝ := tABC
def t_A1BC : ℝ := tA1BC
def t_B1CA : ℝ := tB1CA
def t_C1AB : ℝ := tC1AB

-- The proof statement
theorem area_sum_geq (h : medians_intersect_at_circumcircle A B C medianA medianB medianC t_ABC t_A1BC t_B1CA t_C1AB) :
  (t_A1BC + t_B1CA + t_C1AB) ≥ t_ABC :=
sorry

end area_sum_geq_l518_518184


namespace ball_box_problem_l518_518501

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l518_518501


namespace domain_of_f_l518_518390

noncomputable def f (x : ℝ) : ℝ := (x^4 - 9*x^2 + 20) / (x^2 - 9)

theorem domain_of_f :
  ∀ x : ℝ, x ∈ (Iio (-3) ∪ Ioo (-3) 3 ∪ Ioi 3) ↔ (f x) ≠ 0 := by
  sorry

end domain_of_f_l518_518390


namespace function_properties_range_condition_l518_518889

noncomputable section

def f (A ω φ x : ℝ) : ℝ := A * sin (ω * x + φ) + 1

theorem function_properties (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) (hφ : -π/2 < φ ∧ φ < π/2)
  (h_max : f A ω φ (π / 3) = 3)
  (h_axes_distance : T := π; ω = 2 * π / T) :
  f A ω φ x = 2 * sin (2 * x - π / 6) + 1 :=
sorry

theorem range_condition (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) (hφ : -π/2 < φ ∧ φ < π/2)
  (h_max : f A ω φ (π / 3) = 3)
  (h_axes_distance : T := π; ω = 2 * π / T)
  (x : ℝ) (h_interval : 0 ≤ x ∧ x ≤ π / 2) :
  0 ≤ f 2 2 (-π / 6) x ∧ f 2 2 (-π / 6) x ≤ 3 :=
sorry

end function_properties_range_condition_l518_518889


namespace range_of_m_l518_518864

-- Define points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (2, -2)

-- Define the line equation as a predicate
def line_l (m : ℝ) (p : ℝ × ℝ) : Prop := p.1 + m * p.2 + m = 0

-- Define the condition for the line intersecting the segment AB
def intersects_segment_AB (m : ℝ) : Prop :=
  let P : ℝ × ℝ := (0, -1)
  let k_PA := (P.2 - A.2) / (P.1 - A.1) -- Slope of PA
  let k_PB := (P.2 - B.2) / (P.1 - B.1) -- Slope of PB
  (k_PA <= -1 / m) ∧ (-1 / m <= k_PB)

-- State the theorem
theorem range_of_m : ∀ (m : ℝ), intersects_segment_AB m → (1/2 ≤ m ∧ m ≤ 2) :=
by sorry

end range_of_m_l518_518864


namespace ratio_area_III_IV_l518_518226

theorem ratio_area_III_IV 
  (perimeter_I : ℤ)
  (perimeter_II : ℤ)
  (perimeter_IV : ℤ)
  (side_III_is_three_times_side_I : ℤ)
  (h1 : perimeter_I = 16)
  (h2 : perimeter_II = 20)
  (h3 : perimeter_IV = 32)
  (h4 : side_III_is_three_times_side_I = 3 * (perimeter_I / 4)) :
  (3 * (perimeter_I / 4))^2 / (perimeter_IV / 4)^2 = 9 / 4 :=
by
  sorry

end ratio_area_III_IV_l518_518226


namespace min_value_of_expression_l518_518936

theorem min_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 5) : 
  ∃ (c : ℝ), c = 1/2 ∧ ∀ x y : ℝ, (x > 0) → (y > 0) → (x + y = 5) → (\frac{1}{x + 1} + \frac{1}{y + 2} ≥ c) :=
sorry

end min_value_of_expression_l518_518936


namespace sequence_1994th_term_mod_1000_eq_63_l518_518273

def is_not_divisible_by_three (n : ℤ) : Prop := n % 3 ≠ 0

def sequence_term (n : ℤ) (h : is_not_divisible_by_three n) : ℤ :=
  n^2 - 1

def sequence : ℕ → ℤ
| 0       := 3
| (n + 1) := if h : is_not_divisible_by_three (3 * ↑n + 2)
             then sequence_term (3 * ↑n + 2) h
             else sequence_term (3 * ↑n + 4) (by { dsimp [is_not_divisible_by_three], exact_mod_cast dec_trivial })  -- since (3k+1) and (3k+2) are not divisible by 3

theorem sequence_1994th_term_mod_1000_eq_63 :
  (sequence 1993 % 1000) = 63 :=  -- 1994th term is sequence at index 1993
  sorry

end sequence_1994th_term_mod_1000_eq_63_l518_518273


namespace moving_circle_trajectory_line_through_point_l518_518172

-- Definitions based on given conditions
def circle_m := {x : ℝ × ℝ // (x.1 + 1)^2 + x.2^2 = 49 / 4}
def circle_n := {x : ℝ × ℝ // (x.1 - 1)^2 + x.2^2 = 1 / 4}

-- Part Ⅰ
theorem moving_circle_trajectory (P : ℝ × ℝ) (internally_tangent : ∃ r : ℝ, 0 < r ∧ dist P (-1, 0) = 7/2 - r ∧ dist P (1, 0) = r + 1/2) :
  ellipse_trajectory P :=
  sorry

-- Part Ⅱ
theorem line_through_point (A B l : ℝ × ℝ) (intersect_ellipse : A ≠ B ∧ line_through (1, 0) A B l) 
(h1 : inner_product A B = -2) :
  line_eq l (1, 0) :=
  sorry

end moving_circle_trajectory_line_through_point_l518_518172


namespace diagonal_reaches_another_corner_in_grid_l518_518854

-- Define the grid dimensions as given:
def width := 200
def height := 101

-- Define the starting and ending points as corner cells
def starting_point := (1, 1)
def target_point := (101, 1)

-- Define a proof problem to show that a diagonal line will reach the target point.
theorem diagonal_reaches_another_corner_in_grid :
  ∃ (path : list (ℕ × ℕ)), (starting_point = (1, 1)) ∧ (path.head = some starting_point) ∧ (path.last = some target_point) ∧
  (∀ p ∈ path, 1 ≤ p.1 ∧ p.1 ≤ height ∧ 1 ≤ p.2 ∧ p.2 ≤ width) :=
sorry

end diagonal_reaches_another_corner_in_grid_l518_518854


namespace reinforcement_after_days_l518_518738

-- Define initial conditions as constants
def initial_men : ℕ := 2000
def initial_days : ℕ := 54
def days_passed : ℕ := 15
def remaining_days_after_reinforcement : ℕ := 30

-- Define the proof statement in Lean
theorem reinforcement_after_days (initial_provisions men_remaining : ℕ) : 
    men_remaining = initial_men →
    initial_provisions = initial_men * initial_days →
    let provisions_left_after_15_days := initial_men * (initial_days - days_passed) in
    let provisions_remaining := provisions_left_after_15_days in
    (provisions_remaining = (initial_men + reinforcements) * remaining_days_after_reinforcement) →
    let reinforcements := (initial_men * (initial_days - days_passed) - initial_men * remaining_days_after_reinforcement) / remaining_days_after_reinforcement in
    reinforcements = 600 :=
by
  intros
  rw [← H1, ← H0] at H
  sorry  -- Proof steps not required

end reinforcement_after_days_l518_518738


namespace arithmetic_problem_l518_518380

theorem arithmetic_problem : 72 * 1313 - 32 * 1313 = 52520 := by
  sorry

end arithmetic_problem_l518_518380


namespace ball_box_problem_l518_518500

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l518_518500


namespace ordered_pair_satisfies_l518_518034

theorem ordered_pair_satisfies :
  ∃ (x y : ℚ), 
    7 * x - 50 * y = 3 ∧ 
    3 * y - x = 5 ∧ 
    x = -241 / 29 ∧ 
    y = -32 / 29 :=
by
  use -241 / 29, -32 / 29
  split
  -- Prove the first condition: 7x - 50y = 3
  { field_simp,
    norm_num },
  split
  -- Prove the second condition: 3y - x = 5
  { field_simp,
    norm_num },
  -- Prove x = -241 / 29
  { refl },
  -- Prove y = -32 / 29
  { refl }

end ordered_pair_satisfies_l518_518034


namespace sum_of_powers_of_i_l518_518400

-- Define the cyclical behavior of the imaginary unit i
def i : ℂ := Complex.I

-- State the cyclical property of i
lemma i_pow_four : i^4 = 1 := by
  -- This is a well-known fact about the imaginary unit
  calc
  i^4 = (i*i)*(i*i) : by ring
    ...  = -1 * -1 : by rw [Complex.I_mul_I, Complex.I_mul_I]
    ...  = 1 : by ring

-- Lean statement asserting the main theorem to prove
theorem sum_of_powers_of_i : 
  i^300 + i^301 + i^302 + i^303 + i^304 + i^305 + i^306 + i^307 = 0 := by
  sorry

end sum_of_powers_of_i_l518_518400


namespace smallest_n_exists_sat_l518_518981

-- Define m
def m : ℕ := 30030

-- Define a set M of positive divisors of m with exactly 2 prime factors
def is_two_prime_factor_divisor (x : ℕ) : Prop :=
  x.factors.to_finset.size = 2 ∧ x ∣ m

-- The set M
def M : finset ℕ := (finset.arange m.succ).filter is_two_prime_factor_divisor

-- Statement of the proof problem
theorem smallest_n_exists_sat : ∃ n, (∀ s : finset ℕ, s ⊆ M → s.card = n → ∃ (a b c : ℕ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a * b * c = m) ↔ n = 11 :=
sorry

end smallest_n_exists_sat_l518_518981


namespace distance_between_skew_lines_BE_and_A_l518_518953

noncomputable def dist_between_skew_lines (AB BC B'B : ℝ) : ℝ :=
  let r := (-5, 0, 6)
  let n := (-12, 15, -20)
  let dot_product := r.1 * n.1 + r.2 * n.2 + r.3 * n.3
  let magnitude_n := Real.sqrt (n.1 ^ 2 + n.2 ^ 2 + n.3 ^ 2)
  (Real.abs dot_product) / magnitude_n

theorem distance_between_skew_lines_BE_and_A'C'_is_correct :
  ∀ (AB BC B'B : ℝ), AB = 5 → BC = 4 → B'B = 6 →
  dist_between_skew_lines AB BC B'B = 60 / Real.sqrt 769 :=
by
  intros AB BC B'B hAB hBC hB'B
  sorry

end distance_between_skew_lines_BE_and_A_l518_518953


namespace polynomial_value_at_one_l518_518832

theorem polynomial_value_at_one
  (P : Polynomial ℚ)
  (h_deg : P.degree = 4)
  (h_leading : P.leadingCoeff = 1)
  (h_root : P.eval (Real.sqrt 3 + Real.sqrt 7) = 0) :
  P.eval 1 = 9 :=
by 
  -- The following is a placeholder for the proof steps
  sorry

end polynomial_value_at_one_l518_518832


namespace possible_values_of_K_l518_518948

theorem possible_values_of_K (K M : ℕ) (h : K * (K + 1) = M^2) (hM : M < 100) : K = 8 ∨ K = 35 :=
by sorry

end possible_values_of_K_l518_518948


namespace circumradius_triangle_AQR_l518_518996

-- Let \(\Gamma\) be a circle with radius 17.
-- Let \(\omega\) be a circle with radius 7, internally tangent to \(\Gamma\) at point \(P\).
axiom circle_Gamma : Circle 
axiom Gamma_radius : radius circle_Gamma = 17
axiom circle_omega : Circle 
axiom omega_radius : radius circle_omega = 7
axiom tangency_point : Point
axiom internal_tangency : tangent_at circle_omega circle_Gamma tangency_point

-- Chord \(AB\) of \(\Gamma\) is tangent to \(\omega\) at point \(Q\).
axiom point_A : Point
axiom point_B : Point
axiom point_Q : Point
axiom chord_tangent : tangent_to_chord circle_Gamma circle_omega point_A point_B point_Q

-- Line \(PQ\) intersects \(\Gamma\) at points \(P\) and \(R\) with \(R \neq P\).
axiom point_R : Point
axiom PQ_intersects_Gamma : intersects_at_points line_PQ circle_Gamma point (tangency_point, point_R)
axiom R_not_P : point_R ≠ tangency_point

-- \(\frac{AQ}{BQ} = 3\).
axiom AQ_BQ_ratio : ratio (distance point_A point_Q) (distance point_B point_Q) = 3

-- Prove that the circumradius of triangle \(AQR\) is \(\sqrt{170}\).
theorem circumradius_triangle_AQR :
  circumradius (triangle AQR) = sqrt 170 :=
sorry

end circumradius_triangle_AQR_l518_518996


namespace coprime_sum_primes_l518_518222

def S (n : ℕ) : ℕ :=
  (Finset.filter Nat.prime (Finset.range n)).sum id

theorem coprime_sum_primes (n : ℕ) (h : 10^2018 < n) : ∃ m : ℕ, 10^2018 < m ∧ Nat.coprime (S m) m :=
by
  sorry

end coprime_sum_primes_l518_518222


namespace find_value_of_a_l518_518860

def circle_eq (x y : ℝ) (a : ℝ) : Prop := x^2 + y^2 + 6 * y - a = 0

def center_of_circle (a : ℝ) : ℝ × ℝ := (0, -3)

def distance_from_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ := 
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

def radius_of_circle (a : ℝ) : ℝ := Real.sqrt (a + 9)

theorem find_value_of_a (a : ℝ) :
  (∃ x y : ℝ, circle_eq x y a) ∧ distance_from_point_to_line (center_of_circle a) 1 (-1) (-1) = 1/2 * radius_of_circle a →
  a = -1 := by
  sorry

end find_value_of_a_l518_518860


namespace range_of_k_l518_518383

noncomputable def P (k : ℝ) : set ℝ := {y : ℝ | y = k}
noncomputable def Q (a : ℝ) (x : ℝ) : set ℝ := {y : ℝ | y = a^x + 1}

theorem range_of_k (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : 
  (∃ k : ℝ, P k ∩ Q a = ∅) ↔ (k ∈ set.Iic 1) :=
sorry

end range_of_k_l518_518383


namespace student_grouping_l518_518006

-- Definitions given in conditions
variable {Student : Type}
variable (students : Finset Student)
variable (acquainted : Student → Student → Prop)
variable [DecidableRel acquainted]

/- Ensure students set is of size 49 and each student knows at least 25 others -/
axiom students_size : students.card = 49
axiom min_acquaintances : ∀ s ∈ students, (students.filter (acquainted s)).card ≥ 25

/-- Given these conditions, we need to prove the following: -/
theorem student_grouping :
  ∃ (groups : Finset (Finset Student)), 
    (∀ g ∈ groups, g.card = 2 ∨ g.card = 3) ∧
    (students = groups.bUnion id) ∧ 
    (∀ g ∈ groups, ∀ x y ∈ g, acquainted x y) :=
by
  sorry

end student_grouping_l518_518006


namespace students_number_l518_518784

theorem students_number (x a o : ℕ)
  (h1 : o = 3 * a + 3)
  (h2 : a = 2 * x + 6)
  (h3 : o = 7 * x - 5) :
  x = 26 :=
by sorry

end students_number_l518_518784


namespace add_and_round_l518_518361

theorem add_and_round: ∀ (a b : ℝ), a = 91.234 ∧ b = 42.7689 → Float.round ((a + b) * 100) / 100 = 134.00 := by
  sorry

end add_and_round_l518_518361


namespace distance_to_focus_F2_l518_518258

noncomputable def ellipse_foci_distance
  (x y : ℝ)
  (a b : ℝ) 
  (h_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1) 
  (a2 : a^2 = 9) 
  (b2 : b^2 = 2) 
  (F1 P : ℝ) 
  (h_P_on_ellipse : F1 = 3) 
  (h_PF1 : F1 = 4) 
: ℝ :=
  2

-- theorem to prove the problem statement
theorem distance_to_focus_F2
  (x y : ℝ)
  (a b : ℝ)
  (h_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1)
  (a2 : a^2 = 9)
  (b2 : b^2 = 2)
  (F1 P : ℝ)
  (h_P_on_ellipse : F1 = 3)
  (h_PF1 : F1 = 4)
: F2 = 2 :=
by
  sorry

end distance_to_focus_F2_l518_518258


namespace angle_A_measure_find_a_l518_518576

theorem angle_A_measure (a b c : ℝ) (A B C : ℝ) (h1 : (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) :
  A = π / 3 :=
by
  -- proof steps are omitted
  sorry

theorem find_a (a b c : ℝ) (A : ℝ) (h2 : 2 * c = 3 * b) (area : ℝ) (h3 : area = 6 * Real.sqrt 3)
  (h4 : A = π / 3) :
  a = 2 * Real.sqrt 21 / 3 :=
by
  -- proof steps are omitted
  sorry

end angle_A_measure_find_a_l518_518576


namespace polygon_sides_sum_l518_518556

theorem polygon_sides_sum (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by {
  sorry,
}

end polygon_sides_sum_l518_518556


namespace find_xy_l518_518421

theorem find_xy (x y : ℝ) (k : ℤ) :
  3 * Real.sin x - 4 * Real.cos x = 4 * y^2 + 4 * y + 6 ↔
  (x = -Real.arccos (-4/5) + (2 * k + 1) * Real.pi ∧ y = -1/2) := by
  sorry

end find_xy_l518_518421


namespace time_to_travel_from_B_to_A_without_paddles_l518_518071

-- Variables definition 
variables (v v_r S : ℝ)
-- Assume conditions
def condition_1 (t₁ t₂ : ℝ) (v v_r S : ℝ) := t₁ = 3 * t₂
def t₁ (S v v_r : ℝ) := S / (v + v_r)
def t₂ (S v v_r : ℝ) := S / (v - v_r)

theorem time_to_travel_from_B_to_A_without_paddles
  (v v_r S : ℝ)
  (h1 : v = 2 * v_r)
  (h2 : t₁ S v v_r = 3 * t₂ S v v_r) :
  let t_no_paddle := S / v_r in
  t_no_paddle = 3 * t₂ S v v_r :=
sorry

end time_to_travel_from_B_to_A_without_paddles_l518_518071


namespace sum_of_largest_odd_divisors_l518_518194

def largestOddDivisor (n : ℕ) : ℕ :=
  n / (2^ (n.totient 2))  -- equivalent to continually dividing by 2 until odd

theorem sum_of_largest_odd_divisors :
  (∑ k in (finset.range (220) \ finset.range (111)), largestOddDivisor k) = 12045 := by
  sorry

end sum_of_largest_odd_divisors_l518_518194


namespace an_gt_bn_l518_518598

theorem an_gt_bn (a b : ℕ → ℕ) (h₁ : a 1 = 2013) (h₂ : ∀ n, a (n + 1) = 2013^(a n))
                            (h₃ : b 1 = 1) (h₄ : ∀ n, b (n + 1) = 2013^(2012 * (b n))) :
  ∀ n, a n > b n := 
sorry

end an_gt_bn_l518_518598


namespace max_sum_dist_l518_518291

-- Define the equilateral triangle ABC with side length 1.
variables (A B C : Point)
-- Define the sequence of points {X_i}.
variables (X : ℕ → Point)

-- Define the center of equilateral triangle
def center (A B C : Point) : Point := sorry

-- Define the distance |P Q| for two points P and Q.
def dist (P Q : Point) : ℝ := sorry

-- Define the condition that the angle between three points is 90 degrees.
def right_angle (P Q R : Point) : Prop := sorry 

-- Define the conditions
def conditions (A B C : Point) (X : ℕ → Point) :=
  -- Equilateral triangle with side length 1
  dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1 ∧
  -- X_0 is the center of ABC
  X 0 = center A B C ∧
  -- X_{2i+1} lies on segment AB and X_{2i+2} lies on segment AC
  (∀ i, X (2 * i + 1) ∈ segment A B ∧ X (2 * i + 2) ∈ segment A C) ∧
  -- ∠ X_i X_{i+1} X_{i+2} = 90°
  (∀ i, right_angle (X i) (X (i+1)) (X (i+2))) ∧
  -- X_{i+2} lies in triangle A X_i X_{i+1}
  (∀ i > 0, X (i + 2) ∈ triangle A (X i) (X (i + 1)))

-- The theorem to prove the maximum sum of distances is sqrt(3)/3
theorem max_sum_dist (A B C : Point) (X : ℕ → Point) 
  (h : conditions A B C X) : ∑' i, dist (X i) (X (i+1)) = real.sqrt (6) / 3 := 
sorry

end max_sum_dist_l518_518291


namespace ants_on_vertices_l518_518398

theorem ants_on_vertices : 
  let vertices := [A, B, C, D, E, F, G, H] 
  let moves := [\(A, [B, D, E]), \(B, [A, C, F]), \(C, [B, D, G]), \(D, [A, C, H]),
                \(E, [A, F, H]), \(F, [B, E, G]), \(G, [C, F, H]), \(H, [D, E, G])]
  in
  (∀ (v1 v2 : vertices), v1 ≠ v2 → Prob (\(v1', v2') ∈ moves, v1' ≠ v2') = 1) :=
sorry

end ants_on_vertices_l518_518398


namespace star_comm_star_assoc_star_id_exists_star_not_dist_add_l518_518385

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Statement 1: Commutativity
theorem star_comm : ∀ x y : ℝ, star x y = star y x := 
by sorry

-- Statement 2: Associativity
theorem star_assoc : ∀ x y z : ℝ, star (star x y) z = star x (star y z) := 
by sorry

-- Statement 3: Identity Element
theorem star_id_exists : ∃ e : ℝ, ∀ x : ℝ, star x e = x := 
by sorry

-- Statement 4: Distributivity Over Addition
theorem star_not_dist_add : ∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z := 
by sorry

end star_comm_star_assoc_star_id_exists_star_not_dist_add_l518_518385


namespace smaller_angle_at_3_30_l518_518140

theorem smaller_angle_at_3_30 : 
  ∀ (hour_marks : ℕ) (deg_per_hour_mark : ℕ), 
  hour_marks = 12 → deg_per_hour_mark = 30 → 
  let minute_hand_pos := 6 * deg_per_hour_mark,
      hour_hand_pos := 3 * deg_per_hour_mark + deg_per_hour_mark / 2 in
  abs (minute_hand_pos - hour_hand_pos) = 75 :=
by
  intros hour_marks deg_per_hour_mark h1 h2 
  let minute_hand_pos := 6 * deg_per_hour_mark
  let hour_hand_pos := 3 * deg_per_hour_mark + deg_per_hour_mark / 2
  sorry

end smaller_angle_at_3_30_l518_518140


namespace prob_draw_correct_l518_518456

-- Given conditions
def prob_A_wins : ℝ := 0.40
def prob_A_not_lose : ℝ := 0.90

-- Definition to be proved
def prob_draw : ℝ := prob_A_not_lose - prob_A_wins

theorem prob_draw_correct : prob_draw = 0.50 := by
  sorry

end prob_draw_correct_l518_518456


namespace tangent_line_at_a_minus_1_f_monotonically_increasing_l518_518478

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

theorem tangent_line_at_a_minus_1 :
  let tangent_equation := λ x y, real.log 2 * x + y - real.log 2 = 0 in
  tangent_equation 1 (f 1 (-1)) :=
begin
  -- proof
  sorry
end

theorem f_monotonically_increasing (a : ℝ) (h : a >= 1/2) :
  ∀ x > 0, 0 ≤ - (x + 1) * real.log (1 + x) + x + a * x ^ 2 :=
begin
  -- proof
  sorry
end

end tangent_line_at_a_minus_1_f_monotonically_increasing_l518_518478


namespace breakable_iff_composite_l518_518345

-- Definitions directly from the problem conditions
def is_breakable (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ), a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ a + b = n ∧ (x / a : ℚ) + (y / b : ℚ) = 1

def is_composite (n : ℕ) : Prop :=
  ∃ (s t : ℕ), s > 1 ∧ t > 1 ∧ n = s * t

-- The proof statement
theorem breakable_iff_composite (n : ℕ) : is_breakable n ↔ is_composite n := sorry

end breakable_iff_composite_l518_518345


namespace fraction_of_raisins_cost_is_one_fourth_l518_518236

def total_cost_of_mixture (raisin_pounds almond_pounds cashew_pounds : ℝ) 
  (raisin_price_per_pound almond_cost_multiple cashew_cost_multiple : ℝ) : ℝ :=
  (raisin_pounds * raisin_price_per_pound) +
  (almond_pounds * almond_cost_multiple * raisin_price_per_pound) +
  (cashew_pounds * cashew_cost_multiple * raisin_price_per_pound)

def fraction_cost_of_raisins (raisin_pounds almond_pounds cashew_pounds : ℝ) 
  (raisin_price_per_pound almond_cost_multiple cashew_cost_multiple : ℝ) : ℝ :=
  (raisin_pounds * raisin_price_per_pound) / 
  total_cost_of_mixture raisin_pounds almond_pounds cashew_pounds 
                       raisin_price_per_pound almond_cost_multiple cashew_cost_multiple

theorem fraction_of_raisins_cost_is_one_fourth :
  fraction_cost_of_raisins 4 3 2 4.50 2 3 = 1 / 4 := by
  sorry

end fraction_of_raisins_cost_is_one_fourth_l518_518236


namespace ratio_of_areas_l518_518717

-- Given conditions
variables (R : ℝ)
def O : Type := ℝ
def largerCircle := { p : ℝ × ℝ // (p.1)^2 + (p.2)^2 = R^2 }
def Z : { p : ℝ × ℝ // (p.1)^2 + (p.2)^2 = (3/4 * R)^2 } := sorry
def P : { p : ℝ × ℝ // (p.1)^2 + (p.2)^2 = R^2 } := sorry

-- Proposition to prove
theorem ratio_of_areas (R : ℝ) (hR : 0 < R) :
  let OZ := (3/4 : ℝ) * R,
      area_OZ := π * (OZ)^2,
      area_OP := π * R^2
  in (area_OZ / area_OP) = (9 / 16) :=
sorry

end ratio_of_areas_l518_518717


namespace mountaineering_team_problem_l518_518652

structure Climber :=
  (total_students : ℕ)
  (advanced_climbers : ℕ)
  (intermediate_climbers : ℕ)
  (beginners : ℕ)

structure Experience :=
  (advanced_points : ℕ)
  (intermediate_points : ℕ)
  (beginner_points : ℕ)

structure TeamComposition :=
  (advanced_needed : ℕ)
  (intermediate_needed : ℕ)
  (beginners_needed : ℕ)
  (max_experience : ℕ)

def team_count (students : Climber) (xp : Experience) (comp : TeamComposition) : ℕ :=
  let total_experience := comp.advanced_needed * xp.advanced_points +
                          comp.intermediate_needed * xp.intermediate_points +
                          comp.beginners_needed * xp.beginner_points
  let max_teams_from_advanced := students.advanced_climbers / comp.advanced_needed
  let max_teams_from_intermediate := students.intermediate_climbers / comp.intermediate_needed
  let max_teams_from_beginners := students.beginners / comp.beginners_needed
  if total_experience ≤ comp.max_experience then
    min (max_teams_from_advanced) $ min (max_teams_from_intermediate) (max_teams_from_beginners)
  else 0

def problem : Prop :=
  team_count
    ⟨172, 45, 70, 57⟩
    ⟨80, 50, 30⟩
    ⟨5, 8, 5, 1000⟩ = 8

-- Let's declare the theorem now:
theorem mountaineering_team_problem : problem := sorry

end mountaineering_team_problem_l518_518652


namespace complex_number_b_l518_518206

theorem complex_number_b 
  (b : ℝ) 
  (z : ℂ)
  (h1 : z * (1 + complex.I) = 1 - b * complex.I)
  (h2 : complex.abs z = real.sqrt 2) : 
  b = real.sqrt 3 ∨ b = -real.sqrt 3 :=
sorry

end complex_number_b_l518_518206


namespace ellipse_problem_l518_518097

variable {a b c : ℝ}
variable {x y : ℝ}

-- Condition: Ellipse equation and a > b > 0
def ellipse_eq := x^2 / a^2 + y^2 / b^2 = 1

-- Condition: Focus of parabola coincides with left focus of ellipse
def parabola_focus_eq := c = sqrt 3 ∧ a^2 - b^2 = 3

-- Condition: Line l passing through left focus F1 intersects ellipse C
def line_intersection_with_ellipse (m : ℝ) := 
  ∃ (x₁ x₂ y₁ y₂ : ℝ), x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧ x₂^2 / a^2 + y₂^2 / b^2 = 1

-- Condition: Line l is tangent to a circle with radius as eccentricity of ellipse
def tangent_cond := 
  let e := c / a in (b * c) / sqrt(b^2 + c^2) = c / a

-- Conclusion: Equation of ellipse
def ellipse_conclusion := a = 2 ∧ b = 1 ∧ ellipse_eq 

-- Existence of fixed point M and constant value
def fixed_point_and_const_val (M : ℝ × ℝ) (k : ℝ) :=
  M = (-9 * sqrt 3 / 8, 0) ∧ ∃ (x₁ x₂ y₁ y₂ : ℝ),
  let x_sum := -8 * sqrt 3 * k^2 / (1 + 4 * k^2) in
  let x_prod := (12 * k^2 - 4) / (1 + 4 * k^2) in
  let const_val := (4 * M.1^2 + 8 * sqrt 3 * M.1 + 11) * k^2 + M.1^2 - 4 / (1 + 4 * k^2) in
  const_val = -13 / 64

-- Main statement combining all conditions and conclusions
theorem ellipse_problem :
  ellipse_eq ∧ parabola_focus_eq ∧ line_intersection_with_ellipse ∧ tangent_cond → ellipse_conclusion ∧ ∃ M, fixed_point_and_const_val M :=
by
  -- Proof goes here (placeholder for the actual proof steps)
  sorry

end ellipse_problem_l518_518097


namespace infinite_solutions_if_b_eq_neg_12_l518_518841

theorem infinite_solutions_if_b_eq_neg_12 (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by
  split
  { intro h,
    specialize h 0,
    simp at h,
    linarith },
  { intro h,
    intro x,
    rw h,
    simp }

end infinite_solutions_if_b_eq_neg_12_l518_518841


namespace five_cubic_km_to_cubic_meters_l518_518916

theorem five_cubic_km_to_cubic_meters (km_to_m : 1 = 1000) : 
  5 * (1000 ^ 3) = 5000000000 := 
by
  sorry

end five_cubic_km_to_cubic_meters_l518_518916


namespace max_sum_value_l518_518431

theorem max_sum_value (a : Fin 2002 → ℝ) (h1 : ∀ k, 0 ≤ a k ∧ a k ≤ 1)
  (h2 : a 0 = a 2002) (h3 : a 1 = a 2003) :
  ∑ k in Finset.range 2002, (a k - a (k + 1) * a (k + 2)) ≤ 1001 := 
sorry

end max_sum_value_l518_518431


namespace winning_candidate_percentage_l518_518166

def percentage_votes (totalVotes majorityVotes : ℕ) : ℕ :=
  let majorityPercentage := (majorityVotes / totalVotes) * 100
  50 + majorityPercentage

theorem winning_candidate_percentage :
  ∀ (totalVotes majorityVotes : ℕ), totalVotes = 6900 → majorityVotes = 1380 →
  percentage_votes totalVotes majorityVotes = 70 := 
by 
  intros totalVotes majorityVotes h1 h2
  rw [h1, h2]
  unfold percentage_votes
  have : (1380 / 6900 : ℕ) * 100 = 20 := by sorry
  rw this
  norm_num

end winning_candidate_percentage_l518_518166


namespace dx_perpendicular_to_ay_l518_518180

open EuclideanGeometry

/-- In the triangle ABC with ∠A = 90°, 
let X be the foot of the perpendicular from A to BC, 
D be the reflection of A in B,
and Y be the midpoint of XC.
Then, DX is perpendicular to AY. -/
theorem dx_perpendicular_to_ay
  (A B C X D Y : Point)
  (h_triangle : Triangle ABC)
  (h_angle_A : ∠ B A C = 90°)
  (h_perp : Perpendicular A X BC)
  (h_reflection : Reflection A B D)
  (h_midpoint : Midpoint Y X C) :
  Perpendicular D X A Y := 
sorry

end dx_perpendicular_to_ay_l518_518180


namespace decrease_of_negative_five_l518_518569

-- Definition: Positive and negative numbers as explained
def increase (n: ℤ) : Prop := n > 0
def decrease (n: ℤ) : Prop := n < 0

-- Conditions
def condition : Prop := increase 17

-- Theorem stating the solution
theorem decrease_of_negative_five (h : condition) : decrease (-5) ∧ -5 = -5 :=
by
  sorry

end decrease_of_negative_five_l518_518569


namespace Bobby_candy_l518_518019

theorem Bobby_candy (initial_candy remaining_candy1 remaining_candy2 : ℕ)
  (H1 : initial_candy = 21)
  (H2 : remaining_candy1 = initial_candy - 5)
  (H3 : remaining_candy2 = remaining_candy1 - 9):
  remaining_candy2 = 7 :=
by
  sorry

end Bobby_candy_l518_518019


namespace sum_of_largest_odd_divisors_l518_518195

def largestOddDivisor (n : ℕ) : ℕ :=
  n / (2^ (n.totient 2))  -- equivalent to continually dividing by 2 until odd

theorem sum_of_largest_odd_divisors :
  (∑ k in (finset.range (220) \ finset.range (111)), largestOddDivisor k) = 12045 := by
  sorry

end sum_of_largest_odd_divisors_l518_518195


namespace ways_to_distribute_balls_in_boxes_l518_518519

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l518_518519


namespace find_integer_solutions_l518_518046

theorem find_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + 1 / (x: ℝ)) * (1 + 1 / (y: ℝ)) * (1 + 1 / (z: ℝ)) = 2 ↔ (x = 2 ∧ y = 4 ∧ z = 15) ∨ (x = 2 ∧ y = 5 ∧ z = 9) ∨ (x = 2 ∧ y = 6 ∧ z = 7) ∨ (x = 3 ∧ y = 3 ∧ z = 8) ∨ (x = 3 ∧ y = 4 ∧ z = 5) := sorry

end find_integer_solutions_l518_518046


namespace num_ways_to_put_5_balls_into_4_boxes_l518_518526

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l518_518526


namespace find_angle_KAN_l518_518966

noncomputable def is_isosceles_triangle (A B C : Type) [decidable_eq A] [decidable_eq B] [decidable_eq C]
  (KA IA : ℝ)
  (angle_K : ℝ) (angle_I : ℝ) : Prop :=
angle_K = 30 ∧ angle_I = 30 ∧ KA = IA

noncomputable def perpendicular (K N : Type) (KI : ℝ) : Prop :=
N = KI * ⟨1, 0, 0⟩

noncomputable def congruent_length (A N K I : Type) (KI : ℝ) : Prop :=
(KI ≠ 0) ∧ distance A N = KI

noncomputable def triangle_angle (KA N : ℝ) : Prop :=
KA = 90 ∨ KA = 30

theorem find_angle_KAN (A K I N : Type) [decidable_eq A] [decidable_eq K] [decidable_eq I] [decidable_eq N]
  (angle_K angle_I : ℝ) (KA IA KI : ℝ) :
  is_isosceles_triangle K I A KA IA angle_K angle_I →
  perpendicular K N KI →
  congruent_length A N K I KI →
  triangle_angle KA :=
begin
    intros h1 h2 h3,
    sorry -- Proof goes here
end

end find_angle_KAN_l518_518966


namespace cone_ratio_l518_518116

-- Define the generatrix length l
def l := 1

-- Define the radius of the base of the cone r
variable (r : ℝ)

-- Define the condition: area of the unfolded side surface is 1/3 of the circle's area
def condition (r : ℝ) : Prop :=
  (π * r^2) / 3

-- Prove the required relation
theorem cone_ratio (r : ℝ) (h : π * r^2 / 3 = π * r^2 / 3) : 1 / r = 3 :=
by
  sorry

end cone_ratio_l518_518116


namespace g_x_squared_minus_3_l518_518992

theorem g_x_squared_minus_3 (g : ℝ → ℝ)
  (h : ∀ x : ℝ, g (x^2 - 1) = x^4 - 4 * x^2 + 4) :
  ∀ x : ℝ, g (x^2 - 3) = x^4 - 6 * x^2 + 11 :=
by
  sorry

end g_x_squared_minus_3_l518_518992


namespace percentage_decrease_after_two_reductions_price_increase_for_profit_maximum_profit_l518_518333

-- Definition of conditions
def original_price := 50
def profit_per_kg := 10
def daily_sales := 500
def max_price_increase := 8
def sales_decrease_per_yuan := 20

-- Part 1: Prove the percentage decrease
theorem percentage_decrease_after_two_reductions : 
  ∀ x : ℝ, original_price * ((1 - x / 100) * (1 - x / 100)) = 32 → x = 20 :=
by
  sorry

-- Part 2: Prove price increase for daily profit of 6000 yuan
theorem price_increase_for_profit :
  ∀ m : ℝ, 0 < m ∧ m ≤ max_price_increase → (profit_per_kg + m) * (daily_sales - sales_decrease_per_yuan * m) = 6000 → m = 5 :=
by
  sorry

-- Part 3: Prove maximum profit calculation
theorem maximum_profit :
  ∀ m : ℝ, daily_profit := (profit_per_kg + m) * (daily_sales - sales_decrease_per_yuan * m) → (0 < m ∧ m ≤ max_price_increase) → (profit_per_kg + 7.5) * (daily_sales - sales_decrease_per_yuan * 7.5) = 6125 :=
by
  sorry

end percentage_decrease_after_two_reductions_price_increase_for_profit_maximum_profit_l518_518333


namespace total_pears_l518_518187

def jason_pears : Nat := 46
def keith_pears : Nat := 47
def mike_pears : Nat := 12

theorem total_pears : jason_pears + keith_pears + mike_pears = 105 := by
  sorry

end total_pears_l518_518187


namespace volume_behavior_l518_518455

def tetrahedron_volume (x : ℝ) (h : 0 < x ∧ x < sqrt 3) : ℝ := 
  (1/3) * (sqrt 3 / 4) * (height_from_apex_to_base x)

theorem volume_behavior (x : ℝ) (h : 0 < x ∧ x < sqrt 3) : 
  ¬(∀ y z, y < z → F(y) < F(z)) ∧ (∃ m, ∀ y, F(y) ≤ F(m)) :=
sorry

end volume_behavior_l518_518455


namespace smallest_a1_value_exists_l518_518618
noncomputable def sequence_of_positives (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < a n

noncomputable def satisfies_recurrence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 1 → a n = 7 * a (n - 1) - n

noncomputable def smallest_initial_value (a : ℕ → ℝ) : ℝ :=
  if h : sequence_of_positives a ∧ satisfies_recurrence a
  then a 1
  else 0  -- Default value if the properties are not satisfied

theorem smallest_a1_value_exists :
  ∃ a : ℕ → ℝ, sequence_of_positives a ∧ satisfies_recurrence a ∧ smallest_initial_value a = 13 / 36 :=
sorry

end smallest_a1_value_exists_l518_518618


namespace tan_theta_eq_1_over_3_l518_518364

noncomputable def unit_circle_point (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := Real.sin θ
  (x^2 + y^2 = 1) ∧ (θ = Real.arccos ((4*x + 3*y) / 5))

theorem tan_theta_eq_1_over_3 (θ : ℝ) (h : unit_circle_point θ) : Real.tan θ = 1 / 3 := 
by
  sorry

end tan_theta_eq_1_over_3_l518_518364


namespace min_eq_sum_sqrt_l518_518829

theorem min_eq_sum_sqrt {a b c w : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < w) (hle_a : a ≤ 1) (hle_b : b ≤ 1) (hle_c : c ≤ 1) :
  (a = w^2 / (1 + (w^2 + 1)^2)) →
  (b = w^2 / (1 + w^2)) →
  (c = 1 / (1 + w^2)) →
  min (sqrt ((ab + 1) / (abc)))
      (min (sqrt ((bc + 1) / (abc))) (sqrt ((ca + 1) / (abc)))) =
  sqrt ((1 - a) / a) + sqrt ((1 - b) / b) + sqrt ((1 - c) / c) :=
begin
  sorry
end

end min_eq_sum_sqrt_l518_518829


namespace ω_range_l518_518125

noncomputable def f (ω φ x : ℝ) : ℝ := 
  sin (ω * x + 2 * φ) - 2 * sin φ * cos (ω * x + φ)

theorem ω_range (ω : ℝ) (φ : ℝ) (hω : ω > 0) (h_mono : ∀ x1 x2 : ℝ, π < x1 → x1 < x2 → x2 < (3 / 2) * π → f ω φ x1 > f ω φ x2) : 
  (1 / 2) ≤ ω ∧ ω ≤ 1 :=
sorry

end ω_range_l518_518125


namespace express_h_l518_518121

variable (a b S h : ℝ)
variable (h_formula : S = 1/2 * (a + b) * h)
variable (h_nonzero : a + b ≠ 0)

theorem express_h : h = 2 * S / (a + b) := 
by 
  sorry

end express_h_l518_518121


namespace tangent_slope_at_one_l518_518699

def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem tangent_slope_at_one : f' 1 = 2 := by
  sorry

end tangent_slope_at_one_l518_518699


namespace quadrilateral_bc_eq_3_l518_518852

/-- Given a convex quadrilateral ABCD with side AD = 3, diagonals AC and BD intersecting at point E.
  The areas of triangles ABE and DCE are both 1, and the area of ABCD does not exceed 4.
  Prove that the side BC equals 3. -/ 
theorem quadrilateral_bc_eq_3 {A B C D E : Type*} [metric_space A] [metric_space B]
  [metric_space C] [metric_space D] [metric_space E]
  (h1 : convex A B C D) (h2 : dist A D = 3) (h3 : intersection (segment A C) (segment B D) = E)
  (h4 : area (triangle A B E) = 1) (h5 : area (triangle D C E) = 1)
  (h6 : area (quadrilateral A B C D) ≤ 4) :
  dist B C = 3 :=
sorry

end quadrilateral_bc_eq_3_l518_518852


namespace average_of_P_and_R_l518_518249

theorem average_of_P_and_R (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (Q + R) / 2 = 5250)
  (h3 : P = 3000)
  : (P + R) / 2 = 6200 := by
  sorry

end average_of_P_and_R_l518_518249


namespace johns_number_l518_518587

theorem johns_number (n : ℕ) 
  (h1 : 125 ∣ n) 
  (h2 : 30 ∣ n) 
  (h3 : 800 ≤ n ∧ n ≤ 2000) : 
  n = 1500 :=
sorry

end johns_number_l518_518587


namespace probability_two_different_colors_l518_518155

-- Conditions
def blue_chips := 6
def red_chips := 5
def yellow_chips := 4
def total_chips := blue_chips + red_chips + yellow_chips -- 15

-- Proof problem: Prove that the probability of drawing two chips of different colors is 148/225
theorem probability_two_different_colors : 
  (blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips) +
  (red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips) +
  (yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips) = 
  148 / 225 := by
  sorry

end probability_two_different_colors_l518_518155


namespace complement_intersection_l518_518080

variable (U A B : Set ℕ)

-- Given conditions
def U := {1, 2, 3, 4}
def A := {1, 3, 4}
def B := {2, 3, 4}

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 2} :=
by sorry

end complement_intersection_l518_518080


namespace jenga_game_players_l518_518585

def jenga_game (players: ℕ) (initial_blocks: ℕ) (remaining_blocks: ℕ) (rounds: ℕ) (blocks_before_jess: ℕ): Prop :=
  (initial_blocks - remaining_blocks) = (rounds * players + 1) ∧ initial_blocks = 54 ∧ remaining_blocks = 28 ∧ rounds = 5 ∧ blocks_before_jess = 28

theorem jenga_game_players : ∃ P, jenga_game P 54 28 5 28 ∧ P = 5 :=
by
  use 5
  unfold jenga_game
  split
  . calc
    54 - 28 = 26      : by rfl
    ... = 5 * 5 + 1   : by rfl
  . constructor
    . rfl
    . constructor
      . rfl
      . rfl

end jenga_game_players_l518_518585


namespace enclosed_area_l518_518205

noncomputable def g (x : ℝ) := 1 - sqrt (4 - x^2)

theorem enclosed_area : 
    (let 
        integral_area := 4 * (∫ x in 0..1, sqrt (3 - x^2))
     in
        integral_area = 1 + sqrt 3 + (real.pi / 2)) :=
sorry

end enclosed_area_l518_518205


namespace rectangles_odd_area_l518_518680

theorem rectangles_odd_area (n : ℕ) (Hn : n = 9)
  (lengths widths : Fin n → ℕ)
  (Hlengths : ∀ i, lengths i ∈ {x | ∃ j, x = j ∧ even x ∨ x = 2 * j + 1})
  (Hwidths : ∀ i, widths i ∈ {x | ∃ j, x = j ∧ even x ∨ x = 2 * j + 1}) :
  ∃ k : ℕ, k = 4 ∧ (subset {i | odd (lengths i * widths i)} {0,...,8} ∧ card (subset {i | odd (lengths i * widths i)} {0,...,8}) = k) :=
by
  -- Placeholder for proof
  sorry


end rectangles_odd_area_l518_518680


namespace liars_at_table_l518_518640

open Set

noncomputable def number_of_liars : Set ℕ :=
  {n | ∃ (knights, liars : ℕ), knights + liars = 450 ∧
                                (∀ i : ℕ, i < 450 → (liars + ((i + 1) % 450) + ((i + 2) % 450) = 1)) }

theorem liars_at_table : number_of_liars = {150, 450} := 
  sorry

end liars_at_table_l518_518640


namespace combined_window_savings_zero_l518_518783

theorem combined_window_savings_zero :
  let price_per_window := 100
  let offer := λ (n : ℕ), (n + (n / 3)) 
  let dave_windows := 9
  let doug_windows := 10
  let total_windows := dave_windows + doug_windows
  let cost d := price_per_window * (d + (d / 3))
  let dave_cost_with_offer := cost 6 + 3 * price_per_window
  let doug_cost_with_offer := cost 7 + 1 * price_per_window
  let combined_cost_with_offer := price_per_window * (13)
  let separate_savings := (dave_cost_with_offer + doug_cost_with_offer - cost 19)
  let together_savings := (1300 - combined_cost_with_offer)
  separate_savings == together_savings :=
by
  unfold price_per_window offer dave_windows doug_windows total_windows cost dave_cost_with_offer doug_cost_with_offer combined_cost_with_offer separate_savings together_savings
  sorry

end combined_window_savings_zero_l518_518783


namespace john_drive_after_lunch_l518_518188

theorem john_drive_after_lunch (total_distance : ℝ) (speed : ℝ) (time_before_lunch : ℝ)
  (time_after_lunch : ℝ) (distance_before_lunch : ℝ) (distance_after_lunch : ℝ):
  total_distance = 225 → speed = 45 → time_before_lunch = 2 → 
  distance_before_lunch = speed * time_before_lunch →
  distance_after_lunch = total_distance - distance_before_lunch →
  time_after_lunch = distance_after_lunch / speed →
  time_after_lunch = 3 :=
begin
  sorry
end

end john_drive_after_lunch_l518_518188


namespace mean_median_difference_is_correct_l518_518634

noncomputable def mean_median_difference (scores : List ℕ) (percentages : List ℚ) : ℚ := sorry

theorem mean_median_difference_is_correct :
  mean_median_difference [60, 75, 85, 90, 100] [15/100, 20/100, 25/100, 30/100, 10/100] = 2.75 :=
sorry

end mean_median_difference_is_correct_l518_518634


namespace zoo_feed_days_l518_518949

theorem zoo_feed_days :
  (3 * 25 + 2 * 20 + 5 * 15 + 4 * 10 > 0) → -- total daily consumption is positive
  let total_meat : ℕ := 1200 in
  let daily_consumption : ℕ := (3 * 25 + 2 * 20 + 5 * 15 + 4 * 10) in
  total_meat / daily_consumption = 5 :=
by
  intro h
  let total_meat := 1200
  let daily_consumption := (3 * 25 + 2 * 20 + 5 * 15 + 4 * 10)
  have h_pos : daily_consumption > 0 := h
  sorry

end zoo_feed_days_l518_518949


namespace prime_count_in_range_l518_518493

theorem prime_count_in_range : 
  (∃! p, prime p ∧ 200 < p ∧ p < 220) → 
  (finset.card (finset.filter prime (finset.Icc 201 219)) = 1) :=
  by
  intro h
  sorry

end prime_count_in_range_l518_518493


namespace exist_positive_integers_l518_518594

variable {V : Type} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem exist_positive_integers (A B C O : V) (h : ∃ α β γ : ℝ, α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧ α • A + β • B + γ • C = O) :
  ∃ (p q r : ℕ+), ‖p • (A - O) + q • (B - O) + r • (C - O)‖ < 1 / 2007 :=
sorry

end exist_positive_integers_l518_518594


namespace find_plaid_shirts_l518_518666

def total_shirts : ℕ := 5
def total_pants : ℕ := 24
def total_items : ℕ := total_shirts + total_pants
def neither_plaid_nor_purple : ℕ := 21
def total_plaid_or_purple : ℕ := total_items - neither_plaid_nor_purple
def purple_pants : ℕ := 5
def plaid_shirts (p : ℕ) : Prop := total_plaid_or_purple - purple_pants = p

theorem find_plaid_shirts : plaid_shirts 3 := by
  unfold plaid_shirts
  repeat { sorry }

end find_plaid_shirts_l518_518666


namespace value_of_t_plus_k_l518_518119

noncomputable def f (x t : ℝ) : ℝ := x^3 + (t - 1) * x^2 - 1

theorem value_of_t_plus_k (k t : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∀ x, f x t = 2 * x - 1)
  (h3 : ∃ x₁ x₂, f x₁ t = 2 * x₁ - 1 ∧ f x₂ t = 2 * x₂ - 1) :
  t + k = 7 :=
sorry

end value_of_t_plus_k_l518_518119


namespace min_distance_crawling_ants_l518_518788

theorem min_distance_crawling_ants (a v : ℝ) :
    ∃ t : ℝ, (let dist := λ t : ℝ, real.sqrt (8 * v^2 * t^2 - 8 * v * a * t + 3 * a^2)
             in dist t) = a * real.sqrt (6 / 5) :=
begin
  -- We'd provide the proof here, now we only ensure the statement is correct.
  sorry,
end

end min_distance_crawling_ants_l518_518788


namespace find_x_l518_518488

-- Definitions based directly on conditions
def vec_a : ℝ × ℝ := (2, 4)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 3)
def vec_c (x : ℝ) : ℝ × ℝ := (2 - x, 1)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The mathematically equivalent proof problem statement
theorem find_x (x : ℝ) : dot_product (vec_c x) (vec_b x) = 0 → (x = -1 ∨ x = 3) :=
by
  -- Placeholder for the proof
  sorry

end find_x_l518_518488


namespace solution1_solution2_l518_518665

noncomputable def problem1 (z : ℂ) (ω : ℂ) : Prop :=
  (z.im ≠ 0) ∧ (z.re = 0) ∧ (ω = z + z⁻¹) ∧ (ω.re = ω) ∧ (-1 < ω.re) ∧ (ω.re < 2) ∧ (abs z = 1)

theorem solution1 {z ω : ℂ} : problem1 z ω :=
sorry

noncomputable def problem2 (z : ℂ) (u : ℂ) : Prop :=
  (z.im ≠ 0) ∧ (z.re = 0) ∧ (abs z = 1) ∧ (u = (1 - z) / (1 + z)) ∧ (u.re = 0)

theorem solution2 {z u : ℂ} : problem2 z u :=
sorry

end solution1_solution2_l518_518665


namespace cyclic_quadrilateral_proof_l518_518675

variable (AB BC CD DA P : Point)
variable (AC BD : Line)
variable (circ : Circle)
variable (h1 : Inscribed quadrilateral circ ABCD)
variable (h2 : Intersect AC BD P)
variable (dist1 : distance P AB = 4)
variable (dist2 : distance P BC = √3)
variable (dist3 : distance P CD = 8/√19)
variable (dist4 : distance P DA = 8*√(3/19))
variable (AC_length : length AC = 10)

theorem cyclic_quadrilateral_proof (h1 h2 : True) (dist1 dist2 dist3 dist4 AC_length) :
  AP / PC = 4 / 1 ∧ length BD = 35 / (√19) :=
by sorry

end cyclic_quadrilateral_proof_l518_518675


namespace speed_calculation_l518_518349

def distance : ℝ := 300
def time : ℝ := 4
def expected_speed : ℝ := 75

theorem speed_calculation : (distance / time = expected_speed) :=
by 
  have speed := distance / time
  show speed = expected_speed
  sorry

end speed_calculation_l518_518349


namespace difference_of_roots_l518_518408

theorem difference_of_roots (p : ℝ) :
  let a : ℝ := 1,
      b : ℝ := -p,
      c : ℝ := (p^2 - 5) / 4,
      discriminant := b^2 - 4 * a * c,
      r1 := (-b + real.sqrt discriminant) / (2 * a),
      r2 := (-b - real.sqrt discriminant) / (2 * a) in
  abs (r1 - r2) = real.sqrt 5 :=
by
  -- Proof can be provided here
  sorry

end difference_of_roots_l518_518408


namespace internal_bisector_iff_ratio_l518_518982

-- Given a triangle ABC, with a line Delta passing through A intersecting BC at point I
-- Prove that Delta is the internal angle bisector of ∠CAB if and only if the ratio AB/AC = IB/IC

variables {A B C I : Type} (AB AC IB IC : ℝ) (Delta : Set ℝ)

-- The assumption of Delta being the internal bisector of angle BAC
def is_internal_bisector (A B C I : Type) (AB AC IB IC : ℝ) (Delta : Set ℝ) : Prop :=
  (Δ ⊆ Set.mk {x | x ∈ Points & ∃ y, y ∈ Lines & x ∈ y}) -- simplified assumption

-- The condition AB/AC = IB/IC
def ratio_condition (AB AC IB IC : ℝ) : Prop :=
  AB / AC = IB / IC

-- The theorem statement combining both conditions
theorem internal_bisector_iff_ratio (A B C I : Type) (AB AC IB IC : ℝ) (Delta : Set ℝ) :
  is_internal_bisector A B C I AB AC IB IC Delta ↔ ratio_condition AB AC IB IC :=
sorry

end internal_bisector_iff_ratio_l518_518982


namespace cos_square_minus_sin_square_pi_div_12_l518_518751

theorem cos_square_minus_sin_square_pi_div_12 : 
  (Real.cos (Float.pi / 12))^2 - (Real.sin (Float.pi / 12))^2 = Real.cos (Float.pi / 6) := 
by
  sorry

end cos_square_minus_sin_square_pi_div_12_l518_518751


namespace pairs_at_max_distance_le_card_S_l518_518027

variable (S : Finset (ℝ × ℝ)) (d : ℝ)

def is_maximum_distance (a b : ℝ × ℝ) : Prop :=
  (a, b) ∈ S ×ˢ S ∧ dist a b = d

theorem pairs_at_max_distance_le_card_S 
  (h_nonempty : S.nonempty)
  (h_finite : S.finite)
  (h_max_dist : ∀ a b ∈ S, dist a b ≤ d) :
  (Finset.card {p : (ℝ × ℝ) × (ℝ × ℝ) | is_maximum_distance S d p.1 p.2 }) ≤ Finset.card S := 
  sorry

end pairs_at_max_distance_le_card_S_l518_518027


namespace find_a_l518_518943

theorem find_a : ∃ a : ℝ, (∀ (x y : ℝ), (3 * x + y + a = 0) → (x^2 + y^2 + 2 * x - 4 * y = 0) → a = 1) :=
by
  let center_x : ℝ := -1
  let center_y : ℝ := 2
  have line_eqn : ∀ a : ℝ, 3 * center_x + center_y + a = 0
  have circle_eqn : ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → (x, y) = (center_x, center_y)
  sorry

end find_a_l518_518943


namespace sequence_arrows_520_to_523_l518_518546

theorem sequence_arrows_520_to_523 :
  ∀ (n : ℕ), n % 5 = 0 → (n + 3) % 5 = 3 ∧ (n + 1) % 5 = 1 ∧  (n + 2) % 5 = 2 → 
  arrow_seq n (n + 3) = [0, 1, 2, 3] :=
by
  intros n hmod hseq
  sorry

end sequence_arrows_520_to_523_l518_518546


namespace min_value_y_l518_518085

noncomputable def minimum_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) : ℝ :=
  (1 / a) + (4 / b)

theorem min_value_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) : 
  minimum_y a b ha hb h = 9 / 2 :=
begin
  sorry
end

end min_value_y_l518_518085


namespace exists_rectangle_with_same_color_vertices_l518_518567

-- Define the strip S_n
def strip (n : ℤ) : set (ℝ × ℝ) := 
  { p | p.1 ≥ n ∧ p.1 < n + 1 }

-- Assume a coloring function for the strips
variable (color : ℤ → Prop) -- color is true for red, false for blue

-- Assume a and b are distinct positive integers
variables (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_neq : a ≠ b)

-- Prove there exists a rectangle with vertices of the same color
theorem exists_rectangle_with_same_color_vertices :
  ∃ (x y : ℤ), (color x = color y) ∧ (color (x + a) = color (y + b)) :=
sorry

end exists_rectangle_with_same_color_vertices_l518_518567


namespace original_number_l518_518157

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000

def permutations_sum (a b c : ℕ) : ℕ :=
  let abc := 100 * a + 10 * b + c
  let acb := 100 * a + 10 * c + b
  let bac := 100 * b + 10 * a + c
  let bca := 100 * b + 10 * c + a
  let cab := 100 * c + 10 * a + b
  let cba := 100 * c + 10 * b + a
  abc + acb + bac + bca + cab + cba

theorem original_number (abc : ℕ) (a b c : ℕ) :
  is_three_digit abc →
  abc = 100 * a + 10 * b + c →
  permutations_sum a b c = 3194 →
  abc = 358 :=
by
  sorry

end original_number_l518_518157


namespace farmer_can_transfer_l518_518761

structure State where
  farmer : Bool
  wolf : Bool
  goat : Bool
  cabbage : Bool
  deriving DecidableEq

inductive Trip
| take_goat
| take_wolf
| take_cabbage
| return_alone
| return_with_goat

def initialState : State := { farmer := true, wolf := true, goat := true, cabbage := true }
def finalState : State := { farmer := false, wolf := false, goat := false, cabbage := false }

def move : State → Trip → State
| { farmer := true, wolf, goat, cabbage }, Trip.take_goat => { farmer := false, wolf, goat := false, cabbage }
| { farmer := false, wolf, goat := false, cabbage }, Trip.return_alone => { farmer := true, wolf, goat := false, cabbage }
| { farmer := true, wolf, goat := false, cabbage }, Trip.take_wolf => { farmer := false, wolf := false, goat := false, cabbage }
| { farmer := false, wolf := false, goat := false, cabbage }, Trip.return_with_goat => { farmer := true, wolf := false, goat, cabbage }
| { farmer := true, wolf := false, goat, cabbage }, Trip.take_cabbage => { farmer := false, wolf := false, goat, cabbage := false }
| { farmer := false, wolf := false, goat, cabbage := false }, Trip.return_alone => { farmer := true, wolf := false, goat, cabbage := false }
| { farmer := true, wolf := false, goat, cabbage := false }, Trip.take_goat => { farmer := false, wolf := false, goat := false, cabbage := false }
| st, _ => st

def is_safe (s : State) : Prop :=
  (s.goat != s.wolf ∨ s.farmer == s.goat) ∧
  (s.goat != s.cabbage ∨ s.farmer == s.goat)

theorem farmer_can_transfer :
  ∃ (trips : List Trip),
  ∃ (states : List State),
  List.length trips = 7 ∧
  states.head = initialState ∧
  states.last = finalState ∧
  (∀ i : Fin 6, states.get? i.succ = some (move (states.get! i) (trips.get! i))) ∧
  (∀ s ∈ states, is_safe s) :=
begin
  sorry
end

end farmer_can_transfer_l518_518761


namespace fred_initial_sheets_l518_518843

theorem fred_initial_sheets (X : ℕ) (h1 : X + 307 - 156 = 363) : X = 212 :=
by
  sorry

end fred_initial_sheets_l518_518843


namespace solve_equation_l518_518737

theorem solve_equation (x : ℝ) (h : x + real.sqrt (x^2 - x) = 2) : x = 4 / 3 :=
sorry

end solve_equation_l518_518737


namespace part1_a2_part1_a3_part2_arithmetic_part2_first_term_part2_common_diff_part3_general_l518_518207

open Nat

-- Define the sequence a_n
def a : ℕ → ℤ
| 0     := 0
| 1     := 2
| (n+2) := 2 * a (n+1) - a n + 2

-- Prove that a_2 = 6
theorem part1_a2 : a 2 = 6 := sorry

-- Prove that a_3 = 12
theorem part1_a3 : a 3 = 12 := sorry

-- Prove that the sequence {a_n - a_{n-1}} is arithmetic
theorem part2_arithmetic (n : ℕ) : (a (n + 2) - a (n + 1)) - (a (n + 1) - a n) = 2 := sorry

-- The first term and common difference verification
theorem part2_first_term : (a 1 - a 0) = 2 := sorry
theorem part2_common_diff (n : ℕ) : (a (n+1) - a n) - (a n - a (n-1)) = 2 := sorry

-- General formula for the sequence
theorem part3_general (n : ℕ) : a n = n^2 + n := sorry

end part1_a2_part1_a3_part2_arithmetic_part2_first_term_part2_common_diff_part3_general_l518_518207


namespace problem_statement_l518_518169

open EuclideanGeometry

variable {A B C D E F O : Point}
variable {CD : Line}
variable {E : FootOfPerpendicular (LineThroughPoints O CD)}
variable {F : IntersectionPoint (ParallelLineThroughPoint CD E) (LineThroughPoints A B)}

-- Definitions for isosceles triangle and mentioned lines
def is_isosceles_triangle (A B C : Point) : Prop := dist A B = dist B C
def is_angle_bisector (CD : Line) (A B C : Point) : Prop := is_angle_bisector (∠ C) CD
def is_circumcenter (O : Point) (A B C : Point) : Prop := O = circumcenter A B C
def is_perpendicular (L1 L2 : Line) : Prop := is_perpendicular L1 L2
def is_parallel (L1 L2 : Line) : Prop := is_parallel L1 L2

-- Main proof goal
theorem problem_statement 
  (h1 : is_isosceles_triangle A B C)
  (h2 : is_angle_bisector CD A B C)
  (h3 : is_circumcenter O A B C)
  (h4 : E = FootOfPerpendicular (LineThroughPoints O CD))
  (h5 : F = IntersectionPoint (ParallelLineThroughPoint CD E) (LineThroughPoints A B)) :
  dist B E = dist F D := by
  sorry

end problem_statement_l518_518169


namespace ways_to_distribute_balls_in_boxes_l518_518522

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l518_518522


namespace interval_of_monotonic_increase_l518_518126

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x - π / 6) + 1 / 2

theorem interval_of_monotonic_increase (ω : ℝ) (a β α : ℝ) (hω : ω = 2 / 3)
  (hfa : f ω a = -1 / 2) (hfβ : f ω β = 1 / 2) (hαβ : |α - β| = 3 * π / 4):
    ∃ (k : ℤ), ∀ (x : ℝ), 3 * k * π - π / 2 ≤ x ∧ x ≤ 3 * k * π + π := 
sorry

end interval_of_monotonic_increase_l518_518126


namespace triangle_area_l518_518049

-- Define the points
def A : ℝ × ℝ := (-4, 4)
def B : ℝ × ℝ := (-8, 0)
def C : ℝ × ℝ := (0, 8)

-- Function to calculate the area of a triangle given three points
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

theorem triangle_area : area_of_triangle A B C = 16 := by
  sorry

end triangle_area_l518_518049


namespace find_y_value_l518_518572
-- Import the necessary Lean library

-- Define the conditions and the target theorem
theorem find_y_value (h : 6 * y + 3 * y + y + 4 * y = 360) : y = 180 / 7 :=
by
  sorry

end find_y_value_l518_518572


namespace tic_tac_toe_ways_l518_518952

theorem tic_tac_toe_ways (board : matrix (fin 4) (fin 4) char)
  (win_by_Azar : ∃ (x : fin 4) (y : fin 4), board x y = 'X' 
    ∧ (∀ i : fin 4, board (x + i) y = 'X' 
      ∨ board x (y + i) = 'X' 
      ∨ (∀ j : fin 4, board (x + i) (y + i) = 'X' 
        ∨ board (x - i) (y + i) = 'X'))) :
  (number_of_ways : ℕ) := 
  number_of_ways = 2200 := sorry

end tic_tac_toe_ways_l518_518952


namespace odd_number_expression_l518_518608

theorem odd_number_expression (o n : ℤ) (ho : o % 2 = 1) : (o^2 + n * o + 1) % 2 = 1 ↔ n % 2 = 1 := by
  sorry

end odd_number_expression_l518_518608


namespace all_ai_are_one_l518_518271

theorem all_ai_are_one :
  ∀ (a : Fin 100 → ℕ),
    (11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 = 
    (Finset.range 100).sum (λ i, a i * 10^i)) →
    (Finset.range 100).sum a ≤ 100 →
    (∀ i, a i = 1) :=
by
  sorry

end all_ai_are_one_l518_518271


namespace max_citizens_l518_518210

theorem max_citizens (n : ℕ) (h : Nat.choose n 4 < Nat.choose n 2) : n ≤ 5 :=
by
  have h₁ : Nat.choose n 4 = n * (n-1) * (n-2) * (n-3) / 24 := sorry
  have h₂ : Nat.choose n 2 = n * (n-1) / 2 := sorry
  have h₃ : n * (n-1) * (n-2) * (n-3) / 24 < n * (n-1) / 2 := by 
    rw [h₁, h₂]
    exact h
  let d := λ n => n * (n-1)
  have h₄ : ∀ n, d (n-2) * (n-3) / 12 < 1 := by
    intro n
    simp [d, mul_assoc, div_lt_iff, zero_lt_bit0, zero_lt_bit0, zero_lt_bit0]
    exact (d (n-2) * (n-3))/12 < 1
  sorry

end max_citizens_l518_518210


namespace min_value_of_squares_l518_518612

theorem min_value_of_squares (a b t : ℝ) (h : a + b = t) : (a^2 + b^2) ≥ t^2 / 2 := 
by
  sorry

end min_value_of_squares_l518_518612


namespace num_correct_propositions_l518_518611

-- Definitions of lines, planes, and relationships
variables {Line Plane : Type}

-- Defining predicates for parallelism and perpendicularity
variables (a b c : Line) (M : Plane)

-- Proposition 1: If a ∥ M and b ∥ M, then a ∥ b
def prop1 : Prop := (a ∥ M) ∧ (b ∥ M) → (a ∥ b)

-- Proposition 2: If b ⊂ M and a ∥ b, then a ∥ M
def prop2 : Prop := (b ⊂ M) ∧ (a ∥ b) → (a ∥ M)

-- Proposition 3: If a ⊥ c and b ⊥ c, then a ∥ b
def prop3 : Prop := (a ⊥ c) ∧ (b ⊥ c) → (a ∥ b)

-- Proposition 4: If a ⊥ M and b ⊥ M, then a ∥ b
def prop4 : Prop := (a ⊥ M) ∧ (b ⊥ M) → (a ∥ b)

-- Main theorem: The number of correct propositions is 1
theorem num_correct_propositions : 
  (prop1 a b M = false) ∧ 
  (prop2 a b M = false) ∧ 
  (prop3 a b c = false) ∧ 
  (prop4 a b M = true) → 
  (1 = 1)
:= by
  intros h,
  sorry

end num_correct_propositions_l518_518611


namespace smallest_b_factor_2020_l518_518057

theorem smallest_b_factor_2020 :
  ∃ b : ℕ, b > 0 ∧
  (∃ r s : ℕ, r > s ∧ r * s = 2020 ∧ b = r + s) ∧
  (∀ c : ℕ, c > 0 → (∃ r s : ℕ, r > s ∧ r * s = 2020 ∧ c = r + s) → b ≤ c) ∧
  b = 121 :=
sorry

end smallest_b_factor_2020_l518_518057


namespace telescoping_log_sum_l518_518379

noncomputable def sum_log_identity (k n : ℕ) : ℝ :=
  ∑ k in Finset.range 1 n.succ, (Real.log (1 + 1 / k) / Real.log 3) * 
  (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k + 1))

theorem telescoping_log_sum :
  sum_log_identity 1 128 = 1 - 1 / Real.log (129) / Real.log 3 :=
by
  sorry

end telescoping_log_sum_l518_518379


namespace value_of_expression_l518_518144

theorem value_of_expression (a b : ℝ) (h1 : ∃ x : ℝ, x^2 + 3 * x - 5 = 0)
  (h2 : ∃ y : ℝ, y^2 + 3 * y - 5 = 0)
  (h3 : a ≠ b)
  (h4 : ∀ r : ℝ, r^2 + 3 * r - 5 = 0 → r = a ∨ r = b) : a^2 + 3 * a * b + a - 2 * b = -4 :=
by
  sorry

end value_of_expression_l518_518144


namespace probability_AC_lt_11_l518_518010

noncomputable def prob_AC_less_11 (α : ℝ) : ℝ :=
  if h : 0 < α ∧ α < π / 2 then
    let β := Real.arctan (4 / (3 * Real.sqrt 63))
    in if hβ : β < π / 2 then 
         β / (π / 2)
       else 0
  else 0

-- Proof to be provided
theorem probability_AC_lt_11 :
  ∀ α : ℝ, 0 < α ∧ α < π / 2 → prob_AC_less_11 α = 1 / 3 :=
sorry

end probability_AC_lt_11_l518_518010


namespace dice_probability_area_gt_circumference_l518_518346

theorem dice_probability_area_gt_circumference :
  let dice_roll := finset.range(2, 17)
  let diameter := ∑ (x ∈ dice_roll), finset.card (finset.filter (λ p : ℕ × ℕ, p.1 + p.2 = x) (finset.product (finset.range 1 9) (finset.range 1 9)))
  (∑ (d in diameter.filter (λ d, 4 < d)), finset.card (finset.product (finset.range 1 9) (finset.range 1 9))) / (8 * 8) = 27 / 32 :=
sorry

end dice_probability_area_gt_circumference_l518_518346


namespace OC_linear_combo_l518_518962

variables (OA OB OC : EuclideanSpace ℝ (Fin 2))

-- We specify that ∥OA∥ = 1, ∥OB∥ = 1, and ∥OC∥ = 2
axiom norm_OA : ∥OA∥ = 1
axiom norm_OB : ∥OB∥ = 1
axiom norm_OC : ∥OC∥ = 2

-- We specify the angles
axiom tan_AOC : Real.tan (angle OA OC) = 3
axiom angle_BOC : angle OB OC = Real.pi / 4

-- Define the assertion to be proven
theorem OC_linear_combo : 
  ∃ m n : ℝ, OC = m • OA + n • OB ∧ m = (2 + Real.sqrt 5) / 2 ∧ n = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end OC_linear_combo_l518_518962


namespace sum_of_ai_powers_l518_518922

theorem sum_of_ai_powers :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ),
  (∀ x : ℝ, (1 + x) * (1 - 2 * x)^8 = 
            a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + 
            a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
            a_7 * x^7 + a_8 * x^8 + a_9 * x^9) →
  a_1 * 2 + a_2 * 2^2 + a_3 * 2^3 + 
  a_4 * 2^4 + a_5 * 2^5 + a_6 * 2^6 + 
  a_7 * 2^7 + a_8 * 2^8 + a_9 * 2^9 = 3^9 - 1 :=
by
  sorry

end sum_of_ai_powers_l518_518922


namespace probability_top_face_odd_l518_518754

theorem probability_top_face_odd (n : ℕ) (h : 1 ≤ n ∧ n ≤ 12) :
  let total_faces := 12
      total_dots := 78
      odds := finset.filter (λ x, x % 2 = 1) (finset.range (total_faces + 1))
      evens := finset.filter (λ x, x % 2 = 0) (finset.range (total_faces + 1))
      odd_prob := ∑ x in odds, (1 - (x / total_dots))
      even_prob := ∑ x in evens, (x / total_dots)
      combined_prob := (odd_prob + even_prob) / total_faces
  in combined_prob = (85 / 156) :=
by
  sorry

end probability_top_face_odd_l518_518754


namespace range_f_l518_518391

noncomputable def f (x : ℝ) := Real.logb 3 (8^x + 1)

theorem range_f :
  ∀ y, y > 0 ↔ ∃ x : ℝ, f x = y :=
begin
  sorry
end

end range_f_l518_518391


namespace problem_1_problem_2_l518_518444

/-- Given an ellipse defined by (x^2)/4 + y^2 = 1, a line parallel to the x-axis intersects the ellipse at points A and B, and ∠AOB = 90°. Then the area of △AOB is 4/5. -/
theorem problem_1 (O A B : Real × Real) (hO : O = (0, 0)) 
  (hA : ∃ x1 y1 : Real, A = (x1, y1) ∧ (x1 < 0) ∧ (x1 ≠ 0) ∧ (y1 > 0) ∧ (x1 ^ 2) / 4 + y1 ^ 2 = 1)
  (hB : ∃ x2 y2 : Real, B = (x2, y2) ∧ x2 = -x1 ∧ y2 = y1)
  (hangle : (x1, y1) • (-x1, y1) = 0) :
  let △OAB := 1 / 2 * |(O.1 * (A.2 - B.2) + A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2))| in
  △OAB = 4 / 5 := sorry

/-- Given an ellipse defined by (x^2)/4 + y^2 = 1, and a line always tangent to the circle of radius r, the value of r is 2√5/5. -/
theorem problem_2 (r : Real) 
  (hr : ∃ k m : Real, ∀ x y : Real, x ^ 2 + y ^ 2 = r ^ 2 ∧ (4 * k ^ 2 + 1) * x ^ 2 + 8 * k * m * x + 4 * m ^ 2 - 4 = 0 
  ∧ (5 * m ^ 2 - 4 * k ^ 2 - 4 = 0) ∧ r > 0 ∧ (4 * k ^ 2 = 5 * m ^ 2 - 4) 
  ∧ m ^ 2 ≥ 4 / 5 ∧ m ^ 2 > 3 / 4) :
  r = 2 * real.sqrt(5.0) / 5 := sorry

end problem_1_problem_2_l518_518444


namespace volume_OABC_is_l518_518289

noncomputable def volume_tetrahedron_ABC (a b c : ℝ) (hx : a^2 + b^2 = 36) (hy : b^2 + c^2 = 25) (hz : c^2 + a^2 = 16) : ℝ :=
  1 / 6 * a * b * c

theorem volume_OABC_is (a b c : ℝ) (hx : a^2 + b^2 = 36) (hy : b^2 + c^2 = 25) (hz : c^2 + a^2 = 16) :
  volume_tetrahedron_ABC a b c hx hy hz = (5 / 6) * Real.sqrt 30.375 :=
by
  sorry

end volume_OABC_is_l518_518289


namespace first_1000_digits_6_sqrt35_1999_first_1000_digits_6_sqrt37_1999_first_1000_digits_6_sqrt37_2000_l518_518053

noncomputable def first1000DigitsEq (x : Real) (n : ℕ) (d : ℕ) : Prop :=
  let y := x - Real.floor x
  d = (1 : ℕ) / 10 ^ 1000

theorem first_1000_digits_6_sqrt35_1999 :
  first1000DigitsEq ((6 + Real.sqrt 35) ^ 1999) 1999 9 := 
by
  sorry

theorem first_1000_digits_6_sqrt37_1999 :
  first1000DigitsEq ((6 + Real.sqrt 37) ^ 1999) 1999 0 := 
by
  sorry

theorem first_1000_digits_6_sqrt37_2000 :
  first1000DigitsEq ((6 + Real.sqrt 37) ^ 2000) 2000 9 := 
by
  sorry

end first_1000_digits_6_sqrt35_1999_first_1000_digits_6_sqrt37_1999_first_1000_digits_6_sqrt37_2000_l518_518053


namespace units_digit_of_expression_l518_518058

-- Conditions
def A := 17 + Real.sqrt 210
def B := 17 - Real.sqrt 210

-- Theorem statement
theorem units_digit_of_expression : 
  Nat.unitsDigit ((A^20 + A^83) + (B^20 + B^83)) = 8 := by
  sorry

end units_digit_of_expression_l518_518058


namespace sum_of_roots_of_cubic_l518_518201

def f (x: ℝ) : ℝ := x^3 - 2 * x + 4

theorem sum_of_roots_of_cubic :
  (∃ z1 z2 z3 : ℝ, 15625 * z1 ^ 3 - 50 * z1 - 12 = 0 ∧
                   15625 * z2 ^ 3 - 50 * z2 - 12 = 0 ∧
                   15625 * z3 ^ 3 - 50 * z3 - 12 = 0 ∧
                   z1 + z2 + z3 = 0) :=
by {
  -- Cubic equations given by:
  have h1 : 15625 * z1 ^ 3 - 50 * z1 - 12 = 0 := sorry,
  have h2 : 15625 * z2 ^ 3 - 50 * z2 - 12 = 0 := sorry,
  have h3 : 15625 * z3 ^ 3 - 50 * z3 - 12 = 0 := sorry,
  use [z1, z2, z3],
  split,
  exact h1,
  split,
  exact h2,
  split,
  exact h3,
  -- Sum of roots
  sorry
}

end sum_of_roots_of_cubic_l518_518201


namespace compute_expression_l518_518381

theorem compute_expression : 7^2 - 2 * 5 + 4^2 / 2 = 47 := by
  sorry

end compute_expression_l518_518381


namespace non_coincident_angles_l518_518367

theorem non_coincident_angles : ¬ ∃ k : ℤ, 1050 - (-300) = k * 360 := by
  sorry

end non_coincident_angles_l518_518367


namespace line_transformation_l518_518032

theorem line_transformation (x : ℝ) :
  let y := 2*x - 1
  let transformed_y := 2*(x+1) - 1 - 2
  transformed_y = y := by
  unfold y transformed_y
  sorry

end line_transformation_l518_518032


namespace find_even_increasing_function_l518_518033

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f (x)

def is_monotonic_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y

def fA (x : ℝ) : ℝ := 2^x + x
def fB (x : ℝ) : ℝ := if x < 0 then -x^2 - x else -x^2 + x
def fC (x : ℝ) : ℝ := -x * abs x
def fD (x : ℝ) : ℝ := log 3 (x^2 - 4)

theorem find_even_increasing_function :
  (is_even_function fD ∧ is_monotonic_increasing_on fD 2 4) ∧
  ¬(is_even_function fA ∧ is_monotonic_increasing_on fA 2 4) ∧
  ¬(is_even_function fB ∧ is_monotonic_increasing_on fB 2 4) ∧
  ¬(is_even_function fC ∧ is_monotonic_increasing_on fC 2 4) :=
by {
  sorry
}

end find_even_increasing_function_l518_518033


namespace ratio_ab_bd_l518_518748

-- Definitions based on the given conditions
def ab : ℝ := 4
def bc : ℝ := 8
def cd : ℝ := 5
def bd : ℝ := bc + cd

-- Theorem statement
theorem ratio_ab_bd :
  ((ab / bd) = (4 / 13)) :=
by
  -- Proof goes here
  sorry

end ratio_ab_bd_l518_518748


namespace limit_sin2_exp_ax_bx_1_eq_half_l518_518238

theorem limit_sin2_exp_ax_bx_1_eq_half (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (sin x)^2 / (exp (a * x) - b * x - 1) → 1/2) ↔ (a, b) = (2, 2) ∨ (a, b) = (-2, -2) := 
sorry

end limit_sin2_exp_ax_bx_1_eq_half_l518_518238


namespace stratified_sampling_l518_518779

theorem stratified_sampling (students_class1 students_class2 : ℕ)
  (total_students selected_total : ℕ)
  (h_class1 : students_class1 = 54)
  (h_class2 : students_class2 = 42)
  (h_total : total_students = students_class1 + students_class2)
  (h_selected_total : selected_total = 16) :
  let selection_probability := selected_total / total_students in
  let selected_class1 := students_class1 * selection_probability in
  let selected_class2 := students_class2 * selection_probability in
  selected_class1 = 9 ∧ selected_class2 = 7 :=
by {
  have h1 : total_students = 54 + 42, from h_total,
  have h2 : selection_probability = 16 / total_students, by sorry,
  have h3 : selected_class1 = 54 * (1/6), by sorry,
  have h4 : selected_class2 = 42 * (1/6), by sorry,
  exact and.intro h3 h4
}

end stratified_sampling_l518_518779


namespace max_citizens_l518_518212

theorem max_citizens (n : ℕ) : (nat.choose n 4 < nat.choose n 2 ↔ n ≤ 5) :=
by
  -- We state the conditions given by the problem
  sorry

end max_citizens_l518_518212


namespace find_a_5_l518_518130

def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ ∀ n, 2 * a (n + 1) = a n

theorem find_a_5 (a : ℕ → ℝ) (h : sequence a) : a 5 = 1 / 16 :=
  sorry

end find_a_5_l518_518130


namespace gotham_street_termite_ridden_not_collapsing_l518_518633

def fraction_termite_ridden := 1 / 3
def fraction_collapsing_given_termite_ridden := 4 / 7
def fraction_not_collapsing := 3 / 21

theorem gotham_street_termite_ridden_not_collapsing
  (h1: fraction_termite_ridden = 1 / 3)
  (h2: fraction_collapsing_given_termite_ridden = 4 / 7) :
  fraction_termite_ridden * (1 - fraction_collapsing_given_termite_ridden) = fraction_not_collapsing :=
sorry

end gotham_street_termite_ridden_not_collapsing_l518_518633


namespace part_I_part_II_l518_518123

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem part_I (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1) :
  ∃ x ∈ Set.Ioo m (m + 1), ∀ y ∈ Set.Ioo m (m + 1), f y ≤ f x := sorry

theorem part_II (x : ℝ) (h : 1 < x) :
  (x + 1) * (x + Real.exp (-x)) * f x > 2 * (1 + 1 / Real.exp 1) := sorry

end part_I_part_II_l518_518123


namespace inequality_proof_l518_518082

-- Let x and y be real numbers such that x > y
variables {x y : ℝ} (hx : x > y)

-- We need to prove -2x < -2y
theorem inequality_proof (hx : x > y) : -2 * x < -2 * y :=
sorry

end inequality_proof_l518_518082


namespace smallest_number_in_set_l518_518176

theorem smallest_number_in_set : ∀ (a ∈ {-2, 0, 1, 2}), -2 ≤ a :=
by
  intros a ha
  fin_cases ha <;> simp
  exact le_refl (-2)
  exact neg_nonpos.mpr (by norm_num)
  exact neg_le_zero.mpr (by norm_num)
  exact neg_le_zero.mpr (by norm_num)

end smallest_number_in_set_l518_518176


namespace setC_is_pythagorean_triple_l518_518368

/-- A structure representing a Pythagorean triple -/
structure PythagoreanTriple where
  a : ℕ
  b : ℕ
  c : ℕ
  h : a^2 + b^2 = c^2

/-- We define the specific numbers in set C -/
def setC : PythagoreanTriple := { a := 6, b := 8, c := 10, h := by {
  calc
    6^2 + 8^2 = 36 + 64   : by norm_num
          ... = 100      : by norm_num
          ... = 10^2      : by norm_num
} }

/-- Proposition that set C forms a Pythagorean triple -/
theorem setC_is_pythagorean_triple : ∃ (t : PythagoreanTriple), t.a = 6 ∧ t.b = 8 ∧ t.c = 10 :=
begin
  use setC,
  split,
  { refl },
  split,
  { refl },
  { refl },
end

#check setC_is_pythagorean_triple

end setC_is_pythagorean_triple_l518_518368


namespace sum_of_digits_2_2010_5_2012_6_l518_518306

theorem sum_of_digits_2_2010_5_2012_6 :
  let n := 2 ^ 2010 * 5 ^ 2012 * 6
  digitSum n = 6 :=
by
  sorry

end sum_of_digits_2_2010_5_2012_6_l518_518306


namespace cylinder_volume_increase_l518_518309

theorem cylinder_volume_increase (r h : ℝ) : 
  let V := π * r^2 * h
  let new_h := 3 * h
  let new_r := 3 * r
  let new_V := π * (new_r)^2 * new_h
  new_V = 27 * V :=
by {
  let V := π * r^2 * h,
  let new_h := 3 * h,
  let new_r := 3 * r,
  let new_V := π * (new_r)^2 * new_h,
  show new_V = 27 * V,
  sorry
}

end cylinder_volume_increase_l518_518309


namespace marbles_remaining_l518_518807

theorem marbles_remaining (c r : ℕ) (hc : c = 12) (hr : r = 28) : 
  let total_marbles := c + r in
  let each_take := total_marbles / 4 in
  let total_taken := 2 * each_take in
  total_marbles - total_taken = 20 :=
by
  sorry

end marbles_remaining_l518_518807


namespace subsets_count_of_intersection_l518_518486

-- Define the universal set U and the set M
def U : Set Char := {'a', 'b', 'c', 'd', 'e'}
def M : Set Char := {'a', 'b', 'c'}

-- Complement of N with respect to U
def complement (N : Set Char) : Set Char := U \ N

-- Define a proof statement to show that the number of subsets of M ∩ N is 4
theorem subsets_count_of_intersection (N : Set Char) (h : M ∩ complement N = {'b'}) :
  ∃ (n : Nat), finset.card (finset.powerset (M ∩ N).to_finset) = n ∧ n = 4 :=
by
  sorry

end subsets_count_of_intersection_l518_518486


namespace example_tight_function_ln_correct_statements_l518_518934

def tight_function (f : ℝ → ℝ) : Prop := 
  ∀ x1 x2, f x1 = f x2 → x1 = x2

def is_monotonic (f : ℝ → ℝ) : Prop := 
  ∀ x1 x2, x1 < x2 → f x1 ≤ f x2

theorem example_tight_function_ln : tight_function (λ x : ℝ, Real.log x) :=
sorry

theorem correct_statements :
  (∀ a : ℝ, a < 0 → tight_function (λ x : ℝ, (x^2 + 2*x + a) / x)) ∧
  (∀ f : ℝ → ℝ, tight_function f → ∀ x1 x2, x1 ≠ x2 → f x1 ≠ f x2) :=
sorry

end example_tight_function_ln_correct_statements_l518_518934


namespace time_to_travel_from_B_to_A_without_paddles_l518_518072

-- Variables definition 
variables (v v_r S : ℝ)
-- Assume conditions
def condition_1 (t₁ t₂ : ℝ) (v v_r S : ℝ) := t₁ = 3 * t₂
def t₁ (S v v_r : ℝ) := S / (v + v_r)
def t₂ (S v v_r : ℝ) := S / (v - v_r)

theorem time_to_travel_from_B_to_A_without_paddles
  (v v_r S : ℝ)
  (h1 : v = 2 * v_r)
  (h2 : t₁ S v v_r = 3 * t₂ S v v_r) :
  let t_no_paddle := S / v_r in
  t_no_paddle = 3 * t₂ S v v_r :=
sorry

end time_to_travel_from_B_to_A_without_paddles_l518_518072


namespace even_function_implications_l518_518106

variables {R : Type*} [LinearOrderedField R]

/-- Given conditions
  * The function f(x) and its derivative f'(x) both have a domain of ℝ.
  * Let g(x) = f'(x).
  * f(3/2 - 2x) is an even function.
  * g(2 + x) is an even function.
-/
def condition (f : R → R) (g : R → R) (h : ∀ x, g x = f' x) : Prop :=
  (∀ x, f (3 / 2 - 2 * x) = f (3 / 2 + 2 * x)) ∧
  (∀ x, g (2 + x) = g (2 - x))

-- Main theorem statement
theorem even_function_implications (f : R → R) (g : R → R) (h : ∀ x, g x = f' x) 
  (H : condition f g h) : 
  g (-1 / 2) = 0 ∧ f (-1) = f 4 := 
  sorry

end even_function_implications_l518_518106


namespace a_add_b_perp_a_sub_b_l518_518910

variables {α β : ℝ}
def a : ℝ × ℝ := (Real.cos α, Real.sin α)
def b : ℝ × ℝ := (Real.cos β, Real.sin β)

theorem a_add_b_perp_a_sub_b : (a.1 + b.1, a.2 + b.2) ⬝ (a.1 - b.1, a.2 - b.2) = 0 := by
  sorry

end a_add_b_perp_a_sub_b_l518_518910


namespace angle_between_planes_cosine_l518_518964

variables (a b c : ℝ)

-- The cosine of the angle between the planes BB₁D and ABC₁
theorem angle_between_planes_cosine :
  let normal_BB1D := (b, a, 0) in
  let normal_ABC1 := (b * c, 0, -a * b) in
  let dot_product := normal_BB1D.1 * normal_ABC1.1 + normal_BB1D.2 * normal_ABC1.2 + normal_BB1D.3 * normal_ABC1.3 in
  let magnitude_BB1D := real.sqrt (normal_BB1D.1^2 + normal_BB1D.2^2 + normal_BB1D.3^2) in
  let magnitude_ABC1 := real.sqrt (normal_ABC1.1^2 + normal_ABC1.2^2 + normal_ABC1.3^2) in
  cos (θ : ℝ) = (dot_product / (magnitude_BB1D * magnitude_ABC1)) →
  cos θ = (a * c) / (real.sqrt (a^2 + b^2) * real.sqrt (b^2 + c^2)) :=
sorry

end angle_between_planes_cosine_l518_518964


namespace larger_number_is_391_l518_518745

-- Define the H.C.F and factors
def HCF := 23
def factor1 := 13
def factor2 := 17
def LCM := HCF * factor1 * factor2

-- Define the two numbers based on the factors
def number1 := HCF * factor1
def number2 := HCF * factor2

-- Theorem statement
theorem larger_number_is_391 : max number1 number2 = 391 := 
by
  sorry

end larger_number_is_391_l518_518745


namespace intersection_unique_point_x_coordinate_l518_518553

theorem intersection_unique_point_x_coordinate (a b : ℝ) (h : a ≠ b) : 
  (∃ x y : ℝ, y = x^2 + 2*a*x + 6*b ∧ y = x^2 + 2*b*x + 6*a) → ∃ x : ℝ, x = 3 :=
by
  sorry

end intersection_unique_point_x_coordinate_l518_518553


namespace sequences_and_sum_l518_518450

theorem sequences_and_sum (a_n : ℕ → ℤ) (b_n : ℕ → ℤ) (S_n : ℕ → ℤ) (T_n : ℕ → ℤ) 
  (d q : ℤ) :
  (∀ n, a_n = 2 + (n - 1) * d) → 
  (∀ n, b_n = 2 * q^(n-1)) → 
  (a_n 1 = 2) →
  (b_n 1 = 2) →
  (a_n 4 + b_n 4 = 27) →
  (S_n 4 - b_n 4 = 10) →
  (∀ n, S_n n = n * (2 + (n - 1) * d) / 2) →

  (∀ n, a_n = 3 * n - 1) ∧ 
  (∀ n, b_n = 2^n) ∧ 
  (∀ n, T_n n = -8 + 6 * (2^n - n * 2^n)) :=
begin
  
  -- Proof will be inserted here
  sorry
end

end sequences_and_sum_l518_518450


namespace distinct_pairs_product_neg8_l518_518394

theorem distinct_pairs_product_neg8 : 
  {p : ℤ × ℤ // p.1 * p.2 = -8}.to_finset.card = 4 :=
sorry

end distinct_pairs_product_neg8_l518_518394


namespace sequence_expression_l518_518131

noncomputable def a_n (n : ℕ) : ℤ :=
if n = 1 then -1 else 1 - 2^n

def S_n (a_n : ℕ → ℤ) (n : ℕ) : ℤ :=
2 * a_n n + n

theorem sequence_expression :
  ∀ n : ℕ, n > 0 → (a_n n = 1 - 2^n) :=
by
  intro n hn
  sorry

end sequence_expression_l518_518131


namespace eugene_used_six_boxes_of_toothpicks_l518_518040

theorem eugene_used_six_boxes_of_toothpicks
  (cards_in_deck : ℕ)
  (cards_not_used : ℕ)
  (toothpicks_per_card : ℕ)
  (toothpicks_per_box : ℕ)
  (cards_in_deck = 52)
  (cards_not_used = 16)
  (toothpicks_per_card = 75)
  (toothpicks_per_box = 450) :
  (52 - 16) * 75 / 450 = 6 := 
sorry

end eugene_used_six_boxes_of_toothpicks_l518_518040


namespace deltoid_angle_A_and_C_l518_518802

-- Define the basic geometric constructs and assumptions
structure Deltoid where
  A B C D E F O K : Point
  circumscribed_circle_radius : ℝ
  inscribed_circle_radius : ℝ
  AC_symmetry_line : Line
  right_angle_at_B : ∠ B = 90

-- Problem statement within Lean 4 context
theorem deltoid_angle_A_and_C (d : Deltoid) 
(h1 : d.circumscribed_circle_radius = 1) 
(h2 : d.inscribed_circle_radius = r)
(h3 : LineContains d.K d.A ∧ LineContains d.K d.C)
(h4 : TouchesCircleAt d.inscribed_circle d.AB d.E)
(h5 : TouchesCircleAt d.inscribed_circle d.BC d.F)
(h6 : AngleSymmetry d.AC_symmetry_line d.right_angle_at_B) :
∃ α γ : ℝ, α ≈ 38.1733 ∧ γ ≈ 141.8267 := by
  sorry

end deltoid_angle_A_and_C_l518_518802


namespace euler_formula_quadrant_l518_518399

theorem euler_formula_quadrant
  (θ : Real)
  (hθ : θ = 2 * Real.pi / 3) :
  let z := Complex.exp (Complex.I * θ)
  in z.re < 0 ∧ z.im > 0 :=
by
  sorry

end euler_formula_quadrant_l518_518399


namespace measure_of_AB_l518_518736

-- Define the segments AB and CD being parallel and other given conditions.
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (AB CD : Segment A B) (AD CD : Segment) (angle_BCD : ℝ) (a b : ℝ)

-- Define angle measure conditions and lengths
  (h_parallel : AB ∥ CD)
  (h_angle_D : 3 * ∠B = ∠D)
  (h_AD_length : AD = a)
  (h_CD_length : CD = 2 * b)
  (h_angle_ABC : ∠ABC = 90)

-- The final statement to prove
theorem measure_of_AB (h_parallel : AB ∥ CD)
                      (h_angle_D : 3 * ∠B = ∠D)
                      (h_AD_length : AD = a)
                      (h_CD_length : CD = 2 * b)
                      (h_angle_ABC : ∠ABC = 90) :
  length AB = a + b := 
sorry

end measure_of_AB_l518_518736


namespace trigonometric_problem_l518_518423

open Real

noncomputable def problem1 (α : ℝ) : Prop :=
  2 * sin α = 2 * (sin (α / 2))^2 - 1

noncomputable def problem2 (β : ℝ) : Prop :=
  3 * (tan β)^2 - 2 * tan β = 1

theorem trigonometric_problem (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : π / 2 < β ∧ β < π)
  (h1 : problem1 α) (h2 : problem2 β) :
  sin (2 * α) + cos (2 * α) = -1 / 5 ∧ α + β = 7 * π / 4 :=
  sorry

end trigonometric_problem_l518_518423


namespace import_tax_l518_518730

theorem import_tax (total_value : ℝ) (tax_rate : ℝ) (excess_limit : ℝ) (correct_tax : ℝ)
  (h1 : total_value = 2560) (h2 : tax_rate = 0.07) (h3 : excess_limit = 1000) : 
  correct_tax = tax_rate * (total_value - excess_limit) :=
by
  sorry

end import_tax_l518_518730


namespace simplify_scientific_notation_l518_518660

theorem simplify_scientific_notation :
  (12 * 10^10) / (6 * 10^2) = 2 * 10^8 := 
sorry

end simplify_scientific_notation_l518_518660


namespace munificence_of_quadratic_l518_518401

def p (x : ℝ) : ℝ := x^2 - 2 * x - 1

def interval := Set.Icc (-1 : ℝ) (1 : ℝ)

def munificence (f : ℝ → ℝ) (s : Set ℝ) : ℝ :=
  s.sup (λ x, abs (f x))

theorem munificence_of_quadratic :
  munificence p interval = 2 :=
sorry

end munificence_of_quadratic_l518_518401


namespace domain_f_a_neg1_range_b_a_neg_half_l518_518124

noncomputable def f (x a : ℝ) := log 2 (1 + 2^x + a * (4^x + 1))
noncomputable def h (x a : ℝ) := f x a - x

theorem domain_f_a_neg1 :
  ∀ x : ℝ, f x (-1) ∈ real.log 2 (1 + 2^x - 4^x + 1) ↑((-) 1) ↔ x < 0 :=
sorry

theorem range_b_a_neg_half :
  ∀ b : ℝ, (∀ x ∈ Icc 0 1, h x (-1/2) ≠ b) ↔ b < -2 ∨ b > 0 :=
sorry

end domain_f_a_neg1_range_b_a_neg_half_l518_518124


namespace hyperbola_equation_determined_l518_518884

noncomputable def parabola_focus (p : ℝ) : Point ℝ := ⟨p / 2, 0⟩

noncomputable def hyperbola_eccentricity (a b c : ℝ) : ℝ := c / a

theorem hyperbola_equation_determined 
  (a b c : ℝ)
  (h1 : c = 2)
  (h2 : a = b)
  (h3 : hyperbola_eccentricity a b c = real.sqrt 2)
  (h4 : c^2 = a^2 + b^2) :
  ∃ (h : real), (\frac {x^{2}}{2} - \frac {y^{2}}{2} = 1) := 
sorry

end hyperbola_equation_determined_l518_518884


namespace ellipse_and_lines_slope_constant_l518_518096

noncomputable theory

-- Condition Definitions
def ellipse_focus_y_axis (C : set (ℝ × ℝ)) := 
  ∃ f : ℝ, ∀ {x y}, (x, y) ∈ C → y = f

def major_axis_length (C : set (ℝ × ℝ)) := 
  ∃ a b, a > b ∧ 2 * a = 4

def eccentricity (C : set (ℝ × ℝ)) := 
  ∃ c a, c / a = sqrt 2 / 2

def point_p (P : ℝ × ℝ) :=
  P.1 = 1 ∧ 0 < P.2
 
def complementary_lines_intersect_ellipse (C : set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧
  ∀ k, (∃ m, A = (m * (cos k) + P.1, m * (sin k) + P.2) ∧ 
          B = (m * (cos (π - k)) + P.1, m * (sin (π - k)) + P.2)) ∧
      A ∈ C ∧ B ∈ C

-- Proof Problem
theorem ellipse_and_lines_slope_constant (C : set (ℝ × ℝ)) (P : ℝ × ℝ) :
  ellipse_focus_y_axis C →
  major_axis_length C →
  eccentricity C →
  point_p P →
  complementary_lines_intersect_ellipse C P →
  C = { (x, y) | y^2 / 4 + x^2 / 2 = 1 } ∧
  ∃ A B : ℝ × ℝ, A ≠ B ∧ 
  (A ∈ C ∧ B ∈ C) →
  (∃ k, k = sqrt 2 ∧ (y_B - y_A) / (x_B - x_A) = k) :=
sorry

end ellipse_and_lines_slope_constant_l518_518096


namespace power_sum_positive_l518_518880

theorem power_sum_positive 
    (a b c : ℝ) 
    (h1 : a * b * c > 0)
    (h2 : a + b + c > 0)
    (n : ℕ):
    a ^ n + b ^ n + c ^ n > 0 :=
by
  sorry

end power_sum_positive_l518_518880


namespace instant_noodles_price_reduction_l518_518285

-- Define the conditions
variable (W : ℝ) (P : ℝ) (W_new : ℝ := 1.25 * W)
-- Define the hypothesis that price per bag remains unchanged
variable (price_per_bag_unchanged : P)

-- Statement to prove that the effective price per unit weight decreases by 20%
theorem instant_noodles_price_reduction :
  (P / W_new) = 0.8 * (P / W) :=
by
  sorry

end instant_noodles_price_reduction_l518_518285


namespace find_x_l518_518267

-- Define the mean of three numbers
def mean_three (a b c : ℕ) : ℚ := (a + b + c) / 3

-- Define the mean of two numbers
def mean_two (x y : ℕ) : ℚ := (x + y) / 2

-- Main theorem: value of x that satisfies the given condition
theorem find_x : 
  (mean_three 6 9 18) = (mean_two x 15) → x = 7 :=
by
  sorry

end find_x_l518_518267


namespace phi_value_l518_518471

noncomputable def f (x φ : ℝ) := Real.sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h1 : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|) (h2 : f (π / 3) φ > f (π / 2) φ) : φ = π / 6 :=
by
  sorry

end phi_value_l518_518471


namespace parallelogram_area_base_32_height_22_l518_518315

theorem parallelogram_area_base_32_height_22
    (base : ℝ) (height : ℝ) (h1 : base = 32) (h2 : height = 22) : 
    base * height = 704 :=
by
  rw [h1, h2]
  norm_num -- This step will actually perform the multiplication if used in the actual proof
  sorry

end parallelogram_area_base_32_height_22_l518_518315


namespace no_adjacent_stand_up_probability_l518_518243

noncomputable def coin_flip_prob_adjacent_people_stand_up : ℚ :=
  123 / 1024

theorem no_adjacent_stand_up_probability :
  let num_people := 10
  let total_outcomes := 2^num_people
  (123 : ℚ) / total_outcomes = coin_flip_prob_adjacent_people_stand_up :=
by
  sorry

end no_adjacent_stand_up_probability_l518_518243


namespace circumcenter_is_correct_l518_518559

noncomputable def circumcenter_equation 
(z z1 z2 z3 : ℂ) 
(h : |z - z1| = |z - z2| ∧ |z - z2| = |z - z3|) : Prop :=
z = (|z1|^2 * (z2 - z3) + |z2|^2 * (z3 - z1) + |z3|^2 * (z1 - z2)) / 
    (conj z1 * (z2 - z3) + conj z2 * (z3 - z1) + conj z3 * (z1 - z2))

theorem circumcenter_is_correct 
(z z1 z2 z3 : ℂ) 
(h : |z - z1| = |z - z2| ∧ |z - z2| = |z - z3|) : 
  circumcenter_equation z z1 z2 z3 h :=
sorry

end circumcenter_is_correct_l518_518559


namespace num_ways_to_put_5_balls_into_4_boxes_l518_518524

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l518_518524


namespace problem_statement_l518_518944

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def is_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x > f y

theorem problem_statement {f : ℝ → ℝ} :
  is_odd_function f →
  is_decreasing_on_interval f 3 7 →
  (∀ x, 3 ≤ x ∧ x ≤ 7 → f x ≤ 4) →
  (∃ c, c ∈ [-7, -3] ∧ ∀ x, x ∈ [-7, -3] → f x ≥ f c)
  ∧ (f (min [-7, -3]) = -4) :=
by
  sorry

end problem_statement_l518_518944


namespace sum_of_first_eleven_terms_l518_518886

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_first_eleven_terms 
  (h_arith : is_arithmetic_sequence a)
  (h_S : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_condition : 2 * a 7 - a 8 = 5) :
  S 11 = 55 :=
sorry

end sum_of_first_eleven_terms_l518_518886


namespace sand_exchange_impossible_l518_518749

/-- Given initial conditions for g and p, the goal is to determine if 
the banker can have at least 2 kg of each type of sand in the end. -/
theorem sand_exchange_impossible (g p : ℕ) (G P : ℕ) 
  (initial_g : g = 1001) (initial_p : p = 1001) 
  (initial_G : G = 1) (initial_P : P = 1)
  (exchange_rule : ∀ x y : ℚ, x * p = y * g) 
  (decrement_rule : ∀ k, 1 ≤ k ∧ k ≤ 2000 → 
    (g = 1001 - k ∨ p = 1001 - k)) :
  ¬(G ≥ 2 ∧ P ≥ 2) :=
by
  -- Add a placeholder to skip the proof
  sorry

end sand_exchange_impossible_l518_518749


namespace balls_in_boxes_l518_518504

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l518_518504


namespace find_union_l518_518904

noncomputable def A (p : ℝ) : Set ℝ := {x | 3 * x^2 + p * x - 7 = 0}
noncomputable def B (q : ℝ) : Set ℝ := {x | 3 * x^2 - 7 * x + q = 0}
noncomputable def intersection := Set.singleton (-1 / 3)
noncomputable def union_set := Set.insert 7 (Set.insert (8 / 3) (Set.singleton (-1 / 3)))

theorem find_union (p q : ℝ) (hA : A p = {x | x = -1 / 3 ∨ x = 7})
    (hB : B q = {x | x = -1 / 3 ∨ x = 8 / 3})
    (hIntersection : A p ∩ B q = intersection):
    A p ∪ B q = union_set := by
  sorry

end find_union_l518_518904


namespace GM_parallel_HK_lean_l518_518595

variables {α : Type} [linear_ordered_field α] [euclidean_space α α]
variables (A B C D E F G H K M : euclidean_space α α) 

-- Definitions based on geometric conditions
def triangle_ABC (A B C : euclidean_space α α) : Prop := ∃ A B C, A ≠ B ∧ B ≠ C ∧ C ≠ A

def foot_of_altitude (A B C P : euclidean_space α α) : Prop := ∃ P, collinear A P B ∧ collinear C P B

def orthocenter (A B C H : euclidean_space α α) : Prop := ∃ H, foot_of_altitude A B C H ∧ foot_of_altitude B C A H ∧ foot_of_altitude C A B H

def line_intersection (P Q R : euclidean_space α α) : euclidean_space α α := sorry 

def circumcircle (A B C K : euclidean_space α α) : Prop := ∃ K, ∥K - (A + B + C) / 3∥ = ∥A - (A + B + C) / 3∥

def AK_diameter (A K : euclidean_space α α) : Prop := circumcircle A B C K ∧ ∥K - A∥ = 2 * (∥A - (A + B + C) / 3∥)

def intersection_on_circle (A K BC M : euclidean_space α α) : Prop := ∃ M, AK_diameter A K ∧ M ∈ line_segment BC

def GM_parallel_HK (G M H K : euclidean_space α α) : Prop := ∥G - M∥ / ∥H - K∥ = ∥K - H∥ / ∥M - G∥

-- The final theorem to be proven in Lean 4
theorem GM_parallel_HK_lean 
  (A B C D E F G H K M : euclidean_space α α)
  (hABC : triangle_ABC A B C)
  (hD : foot_of_altitude A B C D)
  (hE : foot_of_altitude B A C E)
  (hF : foot_of_altitude C A B F)
  (hH : orthocenter A B C H)
  (hG : line_intersection E F A B = G)
  (hK : circumcircle A B C K ∧ AK_diameter A K)
  (hM : intersection_on_circle A K (line_segment B C) M) :
  GM_parallel_HK G M H K :=
sorry

end GM_parallel_HK_lean_l518_518595


namespace area_ratio_of_concentric_circles_is_16_over_25_l518_518292

noncomputable def area_ratio_of_concentric_circles (C1 C2: ℝ) (r1 r2: ℝ) 
  (arc_length1_eq_arc_length2: (45 / 360 * C1) = (36 / 360 * C2)) : C1 / C2 = 4 / 5 := by
  sorry

theorem area_ratio_of_concentric_circles_is_16_over_25 
  (C1 C2: ℝ) (r1 r2: ℝ) 
  (arc_length1_eq_arc_length2: (45 / 360 * C1) = (36 / 360 * C2)) :
  (π * r1^2) / (π * r2^2) = (16 / 25) := by
  have circumference_ratio : C1 / C2 = 4 / 5 := area_ratio_of_concentric_circles C1 C2 r1 r2 arc_length1_eq_arc_length2
  have radius_ratio : r1 / r2 = 4 / 5 := by
    -- You would derive this from C1 / C2 = 4 / 5
    sorry
  have area_ratio : (π * r1^2) / (π * r2^2) = (r1 / r2) ^ 2 := by
    -- Square the ratio of the radii to get the ratio of the areas
    sorry
  rw radius_ratio at area_ratio
  exact area_ratio

end area_ratio_of_concentric_circles_is_16_over_25_l518_518292


namespace max_cardinality_T_l518_518985

-- Define the set {1, 2, ..., 2023}
def set_universe : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2023}

-- Define the property that no two elements differ by 5 or 8
def no_diff_5_8 (S : Set ℕ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≠ y → x - y ≠ 5 ∧ x - y ≠ -5 ∧ x - y ≠ 8 ∧ x - y ≠ -8

-- Statement of the problem, where we are to find the maximum cardinality of T
theorem max_cardinality_T : ∃ T : Set ℕ, T ⊆ set_universe ∧ no_diff_5_8 T ∧ T.card = 777 :=
sorry

end max_cardinality_T_l518_518985


namespace solve_quadratic_l518_518663

theorem solve_quadratic :
  ∀ (x : ℝ), x^2 - 4 * x + 4 = 0 → x = 2 :=
by
  intros x h,
  sorry

end solve_quadratic_l518_518663


namespace inequality_always_true_l518_518263

theorem inequality_always_true (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end inequality_always_true_l518_518263


namespace regular_polyhedron_of_congruent_faces_and_equal_dihedral_angles_l518_518654

def regular_polygon (P : Type) [polygon P] : Prop :=
  ∃ (n : ℕ), is_regular n P

def convex_polyhedron (P : Type) [polyhedron P] : Prop :=
  convex P

def equal_dihedral_angles (P : Type) [polyhedron P] : Prop :=
  ∃ (θ : ℝ), ∀ (e₁ e₂ : face P), is_dihedral_angle θ e₁ e₂

theorem regular_polyhedron_of_congruent_faces_and_equal_dihedral_angles (P : Type) [polyhedron P] :
  (∀ (F₁ F₂ : face P), congruent_faces F₁ F₂) → (equal_dihedral_angles P) → is_regular P :=
by
  intros congruent_faces equal_dihedral_angles
  sorry

end regular_polyhedron_of_congruent_faces_and_equal_dihedral_angles_l518_518654


namespace evaluate_expression_l518_518041

theorem evaluate_expression : 
  (Int.ceil ((Int.floor ((15 / 8 : Rat) ^ 2) : Rat) - (19 / 5 : Rat) : Rat) : Int) = 0 :=
sorry

end evaluate_expression_l518_518041


namespace perfectTupleCount_l518_518030

/-- A tuple of length 5 consisting of positive integers from 1 to 5 is perfect if no 
three distinct elements of the tuple form an arithmetic progression. -/
def isPerfect (a : Fin 5 → ℕ) : Prop :=
  ∀ i j k : Fin 5, i ≠ j → j ≠ k → k ≠ i →
  ¬(2 * a j = a i + a k)

/-- The set of 5-tuples of positive integers at most 5. -/
def validTuples : List (Fin 5 → ℕ) :=
  List.replicateM 5 [1, 2, 3, 4, 5]

/-- The count of perfect 5-tuples within the validTuples set. -/
def countPerfectTuples : ℕ :=
  (validTuples.filter isPerfect).length

/-- Theorem stating the number of perfect 5-tuples. -/
theorem perfectTupleCount : countPerfectTuples = 780 :=
  by sorry

end perfectTupleCount_l518_518030


namespace problem_statement_l518_518366

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

def candidate_function (x : ℝ) : ℝ :=
  x * |x|

theorem problem_statement : is_odd_function candidate_function ∧ is_increasing_function candidate_function :=
by
  sorry

end problem_statement_l518_518366


namespace factor_expression_l518_518402

variable (x : ℤ)

theorem factor_expression : 63 * x - 21 = 21 * (3 * x - 1) := 
by 
  sorry

end factor_expression_l518_518402


namespace sum_of_solutions_equation_lean_l518_518830

noncomputable def sum_of_positive_integer_solutions : ℕ := 4450

theorem sum_of_solutions_equation_lean :
  (∑ y in (Finset.filter (λ y, 0 < y) (Finset.range 2001)), if 2 * sin (Real.pi * y) * (sin (Real.pi * y) - sin (Real.pi * (1000 / y))) = Real.cos (2 * Real.pi * y) - 1 then y else 0) = sum_of_positive_integer_solutions :=
sorry

end sum_of_solutions_equation_lean_l518_518830


namespace number_of_liars_l518_518646

constant islanders : Type
constant knight : islanders → Prop
constant liar : islanders → Prop
constant sits_at_table : islanders → Prop
constant right_of : islanders → islanders

axiom A1 : ∀ x : islanders, sits_at_table x → (knight x ∨ liar x)
axiom A2 : (∃ n : ℕ, n = 450 ∧ (λ x, sits_at_table x))
axiom A3 : ∀ x : islanders, sits_at_table x →
  (liar (right_of x) ∧ ¬ liar (right_of (right_of x))) ∨ 
  (¬ liar (right_of x) ∧ liar (right_of (right_of x)))

theorem number_of_liars : 
  (∃ n, ∃ m, (n = 450) ∨ (m = 150)) :=
sorry

end number_of_liars_l518_518646


namespace probability_is_three_fifths_l518_518481

-- Define the set of numbers
def numbers : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define a function to check if the sum of two numbers is odd
def is_odd_sum (a b : ℕ) : Prop := (a + b) % 2 = 1

-- Define the probability function to calculate the probability
def probability_of_odd_sum (s : Finset ℕ) : ℚ :=
  let pairs := s.powerset.filter (λ t, t.card = 2)
  let odd_sum_pairs := pairs.filter (λ t, is_odd_sum (t.to_list).head! (t.to_list).tail.head!)
  odd_sum_pairs.card / pairs.card

-- Theorem to prove the probability is 3/5
theorem probability_is_three_fifths : probability_of_odd_sum numbers = 3 / 5 :=
sorry

end probability_is_three_fifths_l518_518481


namespace mean_score_l518_518969

theorem mean_score (quiz_scores : List ℕ) (exam_scores : List ℕ) :
  quiz_scores = [99, 95, 93, 87, 90] →
  exam_scores = [88, 92] →
  (List.sum quiz_scores + List.sum exam_scores : ℚ) / (quiz_scores.length + exam_scores.length) = 644 / 7 := by
  intros hq he
  rw [hq, he]
  simp [List.sum, List.length]
  norm_num
  -- This is where the proof would be completed
  sorry

end mean_score_l518_518969


namespace num_points_on_ellipse_l518_518265

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 9 = 1

def line (x y : ℝ) : Prop :=
  x / 3 + y / 4 = 1

theorem num_points_on_ellipse (A B : ℝ × ℝ) (A_intersects : ellipse A.1 A.2 ∧ line A.1 A.2)
                             (B_intersects : ellipse B.1 B.2 ∧ line B.1 B.2) :
  ∃ (P : ℝ × ℝ) (hP : ellipse P.1 P.2) (area_eq : area_of_triangle P A B = 3), ∃! (P : ℝ × ℝ), (hP ∧ area_eq) :=
sorry

end num_points_on_ellipse_l518_518265


namespace integer_root_pairs_l518_518405

theorem integer_root_pairs (p q : ℤ) :
  (∀ x, x^2 + p * x + q = 0 → x ∈ ℤ) ∧ 
  (∀ y, y^2 + q * y + p = 0 → y ∈ ℤ) ↔
  (∃ t : ℤ, (p = 0 ∧ q = -t^2) ∨ (p = -t^2 ∧ q = 0)) ∨ 
  (p = 4 ∧ q = 4) ∨ 
  (∃ t : ℤ, p = t ∧ q = -1 - t) ∨ 
  (p = 5 ∧ q = 6) ∨ 
  (p = 6 ∧ q = 5) :=
sorry

end integer_root_pairs_l518_518405


namespace hugo_first_roll_prob_l518_518161

-- Definitions for conditions
def die_roll := {i : ℕ // 1 ≤ i ∧ i ≤ 8}
def is_max (rolls: list die_roll) (max_roll: die_roll) :=
  max_roll ∈ rolls ∧ ∀ r ∈ rolls, r ≤ max_roll

def hugo_wins (hugo_roll: die_roll) (other_rolls: list die_roll) :=
  (hugo_roll > 4 ∧ is_max (hugo_roll :: other_rolls) hugo_roll) ∧
  ¬(∃ tie_roll, tie_roll > 4 ∧ tie_roll = hugo_roll ∧ tie_roll ∈ other_rolls)

-- Statement to prove
theorem hugo_first_roll_prob (hugo_roll: die_roll)
  (other_rolls: list die_roll) (hw: hugo_wins hugo_roll other_rolls) :
  ∃ (prob: ℚ), prob = 625 / 921 := 
sorry

end hugo_first_roll_prob_l518_518161


namespace line_passes_through_circle_center_l518_518940

theorem line_passes_through_circle_center
  (a : ℝ)
  (h_line : ∀ (x y : ℝ), 3 * x + y + a = 0 → (x, y) = (-1, 2))
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0 → (x, y) = (-1, 2)) :
  a = 1 :=
by
  sorry

end line_passes_through_circle_center_l518_518940


namespace conference_games_l518_518667

/-- 
Two divisions of 8 teams each, where each team plays 21 games within its division 
and 8 games against the teams of the other division. 
Prove total number of scheduled conference games is 232.
-/
theorem conference_games (div_teams : ℕ) (intra_div_games : ℕ) (inter_div_games : ℕ) (total_teams : ℕ) :
  div_teams = 8 →
  intra_div_games = 21 →
  inter_div_games = 8 →
  total_teams = 2 * div_teams →
  (total_teams * (intra_div_games + inter_div_games)) / 2 = 232 :=
by
  intros
  sorry


end conference_games_l518_518667


namespace quadratic_completion_l518_518790

theorem quadratic_completion :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) ↔ ((x - 2)^2 = 3) :=
by
  sorry

end quadratic_completion_l518_518790


namespace valid_outfit_combinations_correct_l518_518538

-- Define the problem conditions
def num_colors : ℕ := 8
def items : list string := ["shirt", "pants", "hat", "socks"]

-- Define the function to compute the valid outfit combinations
noncomputable def valid_outfit_combinations : ℕ :=
  let total_combinations := num_colors ^ items.length in
  let invalid_combinations := (
    (binom 4 2 * num_colors * 7 * 6) +
    (binom 4 3 * num_colors * 7) +
    num_colors +
    (binom 4 2 * binom 2 2 * num_colors * 7)
  ) in
  total_combinations - invalid_combinations

-- The theorem we aim to prove
theorem valid_outfit_combinations_correct : valid_outfit_combinations = 1512 :=
by sorry

end valid_outfit_combinations_correct_l518_518538


namespace intersecting_digit_l518_518715

noncomputable def power_of_five := [3125]
noncomputable def power_of_two := [1024, 2048, 4096, 8192]

theorem intersecting_digit:
  ∀ m n, (10 ≤ m ∧ m ≤ 13) ∧ (n = 5) →
  let third_digit_powers_of_two := power_of_two.map (fun x => (x / 10) % 10) in
  let third_digit_power_of_five := (3125 / 10) % 10 in
  third_digit_power_of_five = 2 ∧ 2 ∈ third_digit_powers_of_two → 2
:= 
by 
  sorry

end intersecting_digit_l518_518715


namespace det_matrix_is_zero_l518_518810

noncomputable section

-- Define variables
variables {a b c : ℝ}

-- Define the matrix
def mat : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, cos (a + b + c), cos (a + c)],
    ![cos (a + b + c), 1, cos (b + c)],
    ![cos (a + c), cos (b + c), 1]]

-- State the main theorem
theorem det_matrix_is_zero : Matrix.det mat = 0 :=
  sorry

end det_matrix_is_zero_l518_518810


namespace quadratic_inequality_l518_518655

theorem quadratic_inequality (n : ℕ) (x : Fin (n + 1) → ℝ)
  (h₁ : x 0 = 0) (h₂ : x n = 0) :
  ∑ k in Finset.range n, (x (k + 1) - x k)^2 ≥ 4 * (Real.sin (Real.pi / (2 * n)))^2 * ∑ k in Finset.range (n + 1), (x k)^2 :=
by
  sorry

end quadratic_inequality_l518_518655


namespace unoccupied_volume_l518_518710

/--
Given:
1. Three congruent cones, each with a radius of 8 cm and a height of 8 cm.
2. The cones are enclosed within a cylinder such that the bases of two cones are at each base of the cylinder, and one cone is inverted in the middle touching the other two cones at their vertices.
3. The height of the cylinder is 16 cm.

Prove:
The volume of the cylinder not occupied by the cones is 512π cubic cm.
-/
theorem unoccupied_volume 
  (r h : ℝ) 
  (hr : r = 8) 
  (hh_cone : h = 8) 
  (hh_cyl : h_cyl = 16) 
  : (π * r^2 * h_cyl) - (3 * (1/3 * π * r^2 * h)) = 512 * π := 
by 
  sorry

end unoccupied_volume_l518_518710


namespace quadruplets_babies_l518_518017

variable (a b c d : ℝ)

-- Defining the given conditions
def conditions : Prop :=
  (b = 5 * c) ∧
  (a = 2 * b) ∧
  (d = 1/2 * c) ∧
  (2 * a + 3 * b + 4 * c + 5 * d = 1500)

-- The theorem to be proved
theorem quadruplets_babies : conditions a b c d → 4 * c ≈ 145 := 
by
  intros h
  sorry

end quadruplets_babies_l518_518017


namespace right_triangle_largest_angle_l518_518954

theorem right_triangle_largest_angle (a b : ℝ) (h : a^2 / b^2 = 64) (h_rt : a^2 + b^2 > 0) :
  a.bif (a > b) then 90 else sorry
  (if (b > a) then 0 else sorry)

end right_triangle_largest_angle_l518_518954


namespace sum_of_gcd_3n_plus_5_n_l518_518013

theorem sum_of_gcd_3n_plus_5_n (n: ℕ) (h : 0 < n):
  ∃ s, s = {d ∈ finset.range 6 | ∃ n, gcd (3*n + 5) n = d}.sum ∧ s = 6 :=
by
  sorry

end sum_of_gcd_3n_plus_5_n_l518_518013


namespace largest_x_acd_over_b_l518_518817

noncomputable def x := (14: ℚ) * (-8 + 16 * Real.sqrt 2)⁻¹

theorem largest_x
  (a b c d : ℤ)
  (x = (a + b * Real.sqrt c) / d)
  (h : (7 * x) / 8 + 2 = 4 / x) :
  x = (-8 + 16 * Real.sqrt 2) / 7 :=
by
  sorry

theorem acd_over_b (a b c d : ℤ)
  (hx : x = (-8 + 16 * Real.sqrt 2) / 7)
  (ha : a = -8) (hb : b = 16) (hc : c = 2) (hd : d = 7) :
  (a * c * d) / b = -7 :=
by
  sorry

end largest_x_acd_over_b_l518_518817


namespace marble_prism_weight_l518_518972

def height : ℝ := 8
def base_side_length : ℝ := 2
def density : ℝ := 2700

def volume (h : ℝ) (s : ℝ) : ℝ := s * s * h
def weight (v : ℝ) (d : ℝ) : ℝ := v * d

theorem marble_prism_weight : weight (volume height base_side_length) density = 86400 := by
  sorry

end marble_prism_weight_l518_518972


namespace part_c_part_b_l518_518110

-- Definitions
def f : ℝ → ℝ := sorry
def g (x : ℝ) := (f' x)

-- Conditions
axiom dom_f : ∀ x : ℝ, x ∈ domain f
axiom dom_g : ∀ x : ℝ, x ∈ domain g
axiom even_f : ∀ x : ℝ, f (3 / 2 - 2 * x) = f (3 / 2 + 2 * x)
axiom even_g : ∀ x : ℝ, g (2 + x) = g (2 - x)

-- Theorems to prove
theorem part_c : f (-1) = f 4 := by sorry
theorem part_b : g (-1 / 2) = 0 := by sorry

end part_c_part_b_l518_518110


namespace inscribed_sphere_radius_formula_l518_518275

-- Define the regular quadrilateral pyramid and its relevant properties
variables (a b : ℝ)

-- Define the function that calculates the radius of the inscribed sphere
def inscribed_sphere_radius (a b : ℝ) : ℝ :=
  (a * real.sqrt(2 * (b^2 - a^2 / 2))) / (2 * (1 + real.sqrt(4 * b^2 - a^2)))

-- The main statement to prove
theorem inscribed_sphere_radius_formula (a b : ℝ) : 
  inscribed_sphere_radius a b = (a * real.sqrt(2 * (b^2 - a^2 / 2))) / (2 * (1 + real.sqrt(4 * b^2 - a^2))) :=
by {
  sorry, -- proof to be provided
}

end inscribed_sphere_radius_formula_l518_518275


namespace findAngleB_findArea_l518_518577

-- Definitions based on the conditions
def inTriangleABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- \( (2a - c)\cos B = b\cos C \)
  (2 * a - c) * Real.cos B = b * Real.cos C

def areaOfTriangle (a b c : ℝ) (B : ℝ) : ℝ :=
  -- \( \frac{1}{2} \cdot a \cdot c \cdot \sin B \)
  (1/2) * a * c * Real.sin B

-- Statement to prove the measure of angle B
theorem findAngleB (A B C a b c : ℝ) (h : inTriangleABC A B C a b c) : B = Real.pi / 3 := sorry

-- Statement to find the area of the triangle with given sides and angle
theorem findArea (a c : ℝ) (h : a = 3) (h1 : c = 2) (B : ℝ) (h2 : B = Real.pi / 3) : areaOfTriangle a 0 c B = (3 * Real.sqrt 3) / 2 := sorry

end findAngleB_findArea_l518_518577


namespace remainder_ab_mod_n_l518_518203

theorem remainder_ab_mod_n (n : ℕ) (a c : ℤ) (h1 : a * c ≡ 1 [ZMOD n]) (h2 : b = a * c) :
    (a * b % n) = (a % n) :=
  by
  sorry

end remainder_ab_mod_n_l518_518203


namespace ways_to_distribute_balls_in_boxes_l518_518520

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l518_518520


namespace greatest_integer_less_than_or_equal_to_frac_l518_518719

theorem greatest_integer_less_than_or_equal_to_frac (a b c d : ℝ)
  (ha : a = 4^100) (hb : b = 3^100) (hc : c = 4^95) (hd : d = 3^95) :
  ⌊(a + b) / (c + d)⌋ = 1023 := 
by
  sorry

end greatest_integer_less_than_or_equal_to_frac_l518_518719


namespace line_minimizes_area_l518_518766

theorem line_minimizes_area (P A B : Point) (O : Point) (l : Line) 
    (hx : ∃ a > 0, ∃ b > 0, l = { x | x / a + y / b = 1 })
    (cond : l.contains P ∧ P.x = 2 ∧ P.y = 1 ∧ O.x = 0 ∧ O.y = 0)
    (intersects_ax: l.intersects_axis_positive A B)
    (area_minimizes: Area (triangle A B O) minimized) : 
    l.equation = (λ p, p.x + 2 * p.y - 4 = 0) := 
sorry

end line_minimizes_area_l518_518766


namespace equal_focal_distances_l518_518946

theorem equal_focal_distances (k : ℝ) (hk : 0 < k ∧ k < 5) :
  let a₁ := 4
      b₁ := real.sqrt (5 - k)
      a₂ := real.sqrt (16 - k)
      b₂ := real.sqrt 5
      c₁ := real.sqrt (a₁ ^ 2 + b₁ ^ 2)
      c₂ := real.sqrt (a₂ ^ 2 + b₂ ^ 2)
  in c₁ = c₂ :=
by sorry

end equal_focal_distances_l518_518946


namespace general_formula_seq_l518_518900

variable {a : ℝ}
variable {n : ℕ}

/-- Definition of the sequence and its recursive relationship --/
def seq (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := (2 * seq a n) / (seq a n + 1)

theorem general_formula_seq (a : ℝ) (n : ℕ) (h : a > 0) :
  seq a n = (2^(n-1) * a) / ((2^(n-1) - 1) * a + 1) :=
sorry

end general_formula_seq_l518_518900


namespace marbles_remaining_l518_518806

theorem marbles_remaining (c r : ℕ) (hc : c = 12) (hr : r = 28) : 
  let total_marbles := c + r in
  let each_take := total_marbles / 4 in
  let total_taken := 2 * each_take in
  total_marbles - total_taken = 20 :=
by
  sorry

end marbles_remaining_l518_518806


namespace parabola_slope_intersection_l518_518482

theorem parabola_slope_intersection (k : ℝ) (hk : 0 < k) :
  (∃ (F P A B M N : Point) (C : Parabola),
    C.equation = (λ p : Point, p.y^2 = 4 * p.x) ∧
    F = Focus C ∧
    P = ⟨-1, 0⟩ ∧
    (∀ l : Line, l.slope = k ∧ l.contains P ∧
    ∃ (A B : Point), l.intersects C at A ∧ l.intersects C at B ∧
    (LineThrough A F).intersects C again at M ∧
    (LineThrough B F).intersects C again at N ∧
    (|distance A F| / |distance F M| + 
     |distance B F| / |distance F N| = 18)) ∧
    k = sqrt 5 / 5)
sorry

end parabola_slope_intersection_l518_518482


namespace cubic_kilometers_to_cubic_meters_l518_518917

theorem cubic_kilometers_to_cubic_meters :
  (5 : ℝ) * (1000 : ℝ)^3 = 5_000_000_000 :=
by
  sorry

end cubic_kilometers_to_cubic_meters_l518_518917


namespace number_of_boys_in_other_communities_l518_518744

-- Definitions from conditions
def total_boys : ℕ := 700
def percentage_muslims : ℕ := 44
def percentage_hindus : ℕ := 28
def percentage_sikhs : ℕ := 10

-- Proof statement
theorem number_of_boys_in_other_communities : 
  (700 * (100 - (44 + 28 + 10)) / 100) = 126 := 
by
  sorry

end number_of_boys_in_other_communities_l518_518744


namespace six_digit_multiples_of_27_with_3_6_9_only_l518_518919

theorem six_digit_multiples_of_27_with_3_6_9_only : ∃ n, n = 51 ∧
  (∀ x, (6-digit x) ∧ (all_digits_are_3_6_or_9 x) ∧ (multiple_of_27 x) → count x = n) :=
sorry

end six_digit_multiples_of_27_with_3_6_9_only_l518_518919


namespace solution_l518_518603

-- Given conditions in the problem
def F (x : ℤ) : ℤ := sorry -- Placeholder for the polynomial with integer coefficients
variables (a : ℕ → ℤ) (m : ℕ)

-- Given that: ∀ n, ∃ k, F(n) is divisible by a(k) for some k in {1, 2, ..., m}
axiom forall_n_exists_k : ∀ n : ℤ, ∃ k : ℕ, k < m ∧ a k ∣ F n

-- Desired conclusion: ∃ k, ∀ n, F(n) is divisible by a(k)
theorem solution : ∃ k : ℕ, k < m ∧ (∀ n : ℤ, a k ∣ F n) :=
sorry

end solution_l518_518603


namespace only_1996_is_leap_l518_518371

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0))

def is_leap_year_1996 := is_leap_year 1996
def is_leap_year_1998 := is_leap_year 1998
def is_leap_year_2010 := is_leap_year 2010
def is_leap_year_2100 := is_leap_year 2100

theorem only_1996_is_leap : 
  is_leap_year_1996 ∧ ¬is_leap_year_1998 ∧ ¬is_leap_year_2010 ∧ ¬is_leap_year_2100 :=
by 
  -- proof will be added here later
  sorry

end only_1996_is_leap_l518_518371


namespace max_area_of_AOBC_l518_518179

-- Definitions for the geometric context
variables {P Q O C A B : Point}
variables (l : ℝ) [DecidableEq Point]

-- Conditions: ∠ POQ is a right angle and points A, B on OP, OQ respectively.
def right_angle (P O Q : Point) : Prop := ∠ POQ = 90°
def on_segment (A O P : Point) : Prop := A ∈ segment O P
def on_segment (B O Q : Point) : Prop := B ∈ segment O Q
def moving_point (C O : Point) : Prop := C ∈ (interior_angle POQ)

-- Condition: Perimeter BC + CA equals a constant l
def perimeter (B C A : Point) (l : ℝ) : Prop := distance B C + distance C A = l

theorem max_area_of_AOBC
  (h_right_angle : right_angle P O Q)
  (on_seg_A : on_segment A O P)
  (on_seg_B : on_segment B O Q)
  (move_C : moving_point C O)
  (h_perimeter : perimeter B C A l) :
  ∃ (max_area_config : A = B ∧ B = C ∧ 
                       OA = OB = OC = l / (4 * sin (22.5))) :
  quadrilateral_area A O B C =
    max (quadrilateral_area A O B C) :=
sorry

end max_area_of_AOBC_l518_518179


namespace part1_solution_part2_solution_l518_518621

noncomputable def f (ω x : ℝ) : ℝ := 2 * sqrt 3 * sin (2 * ω * x + π / 3) - 4 * (cos (ω * x))^2 + 3

theorem part1_solution (ω : ℝ) (h0 : 0 < ω) (h2 : ω < 2)
  (h_symm : ∀ x : ℝ, (2 * ω * x + π / 3 = k * π + π / 2) ↔ (x = π / 6)) :
  ω = 1 ∧ (∀ x : ℝ, f ω x ≥ -1) :=
sorry

theorem part2_solution (a b c A B C : ℝ)
  (h_a : a = 1)
  (h_area : 1 / 2 * b * c * sin A = sqrt 3 / 4)
  (h_fA : f 1 A = 2)
  (h2A : 2 * A + π / 6 = 5 * π / 6)
  (h_hs : a = b / c ∧ A = π / 3 ∧ B + C = π - A) :
  a + b + c = 3 :=
sorry

end part1_solution_part2_solution_l518_518621


namespace cardinal_sets_eq_l518_518432

open Set

variables (A B : Set Nat) [Finite A] [Finite B] (pA : PowerSet A)

def F (f : pA → B) : Prop :=
  ∀ (X Y : pA), f (X ∩ Y) = min (f X) (f Y)

def G (g : pA → B) : Prop :=
  ∀ (X Y : pA), g (X ∪ Y) = max (g X) (g Y)

theorem cardinal_sets_eq (A B : Set Nat) [Finite A] [Finite B] (pA : PowerSet A) :
  (card {f : (pA → B) // F f}) = (card {g : (pA → B) // G g}) ∧
  (card {f : (pA → B) // F f}) = ∑ i in (range (card B + 1)), i ^ (card A) := by
  sorry

end cardinal_sets_eq_l518_518432


namespace guards_catch_prisoners_l518_518393

/-- There are two guards (G1 and G2) and two prisoners (P1 and P2) in a connected cell network with doors.
    Guards and prisoners move through adjacent cells via doors.
    Initially, guards and prisoners are placed as shown in the initial diagram.
    Guards move first, followed by prisoners, alternating turns.
    If a guard catches a prisoner, both leave the game.
    The remaining pair continues the game.
    We prove that guards will inevitably catch the prisoners. -/
theorem guards_catch_prisoners (cells : set char) (adj : char → set char)
  (G1_start P1_start G2_start P2_start : char)
  (G1_moves P1_moves G2_moves P2_moves : list char)
  (H_adj : ∀ c1 c2, c2 ∈ adj c1 → c1 ∈ cells ∧ c2 ∈ cells)
  (G1_movement : G1_moves = move_through_adj_cells G1_start adj)
  (P1_movement : P1_moves = move_through_adj_cells P1_start adj)
  (G2_movement : G2_moves = move_through_adj_cells G2_start adj)
  (P2_movement : P2_moves = move_through_adj_cells P2_start adj) :
  ∃ G1_p G2_p P1_p P2_p, 
    (G1_moves.last = some G1_p ∧ P1_moves.last = some P1_p ∧ G1_p = P1_p) ∨
    (G2_moves.last = some G2_p ∧ P2_moves.last = some P2_p ∧ G2_p = P2_p) :=
begin
  sorry -- The proof logic would be constructed here
end

end guards_catch_prisoners_l518_518393


namespace product_of_8_65_and_0_3_is_2_60_l518_518304

open Real

def product_rounded_to_two_decimals (x y : ℝ) : ℝ :=
  (x * y * 100).round / 100

theorem product_of_8_65_and_0_3_is_2_60 :
  product_rounded_to_two_decimals 8.65 0.3 = 2.60 :=
by
  sorry

end product_of_8_65_and_0_3_is_2_60_l518_518304


namespace solution_is_correct_l518_518060

noncomputable def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else digit_sum (n / 10 + n % 10)

noncomputable def repeated_digit_sum (k : ℕ) : ℕ :=
  if k < 10 then k else repeated_digit_sum (digit_sum k)

def count_digit (digit : ℕ) (s : List ℕ) : ℕ :=
  s.count digit

def generated_numbers : List ℕ :=
  List.map repeated_digit_sum (List.range' 1 1992)

def problem_solution : ℕ × ℕ × ℕ :=
  (count_digit 1 generated_numbers, count_digit 9 generated_numbers, count_digit 0 generated_numbers)

theorem solution_is_correct :
  problem_solution = (222, 221, 0) :=
sorry

end solution_is_correct_l518_518060


namespace least_integer_with_eight_divisors_l518_518720

def num_divisors (n : ℕ) : ℕ :=
  ((0 : ℕ), n).range.count (λ d, n % d = 0)

theorem least_integer_with_eight_divisors : ∃ k : ℕ, (num_divisors k = 8) ∧ (∀ m : ℕ, num_divisors m = 8 → 24 ≤ m → 24 = m) :=
begin
  sorry
end

end least_integer_with_eight_divisors_l518_518720


namespace count_elements_in_B_l518_518079

def A : set ℕ := {2, 3, 4}

def B : set ℕ := {x | ∃ m n ∈ A, m ≠ n ∧ x = m * n}

theorem count_elements_in_B : B.to_finset.card = 3 := by
  sorry

end count_elements_in_B_l518_518079


namespace ellipse_equation_line_equation_l518_518445

-- Conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity (a c : ℝ) : Prop :=
  c / a = sqrt 3 / 3

def min_distance (a c : ℝ) : Prop :=
  a - c = sqrt 3 - 1

-- Part (1): Equation of the ellipse
theorem ellipse_equation (a b c : ℝ) (h1 : ellipse a b) (h2 : eccentricity a c) (h3 : min_distance a c) :
  a = sqrt 3 ∧ c = 1 ∧ b = sqrt 2 :=
sorry

-- Part (2): Equation of the line
def line_intersects_ellipse (a b m : ℝ) (x y : ℝ) (d : ℝ) : Prop :=
  y = x + m ∧ |2 * x * x - 6 * y + m^2 - 6| = d

theorem line_equation (m : ℝ) :
  (m = 1 ∨ m = -1) → (x y : ℝ) (line_intersects_ellipse a b m x y (8 * sqrt 3 / 5)) →
  (y = x + 1) ∨ (y = x - 1) :=
sorry


end ellipse_equation_line_equation_l518_518445


namespace pa_pb_pc_greater_than_half_perimeter_l518_518597

theorem pa_pb_pc_greater_than_half_perimeter (A B C P : Point) (hP : P ∉ line A B ∧ P ∉ line B C ∧ P ∉ line C A):
  let d := dist in
  d P A + d P B + d P C > (d A B + d B C + d C A) / 2 :=
sorry

end pa_pb_pc_greater_than_half_perimeter_l518_518597


namespace base_8_addition_and_conversion_l518_518360

def add_in_base_8 (a b : Nat) : Nat := 
  -- function to add numbers in base 8
  sorry

def convert_base8_to_decimal (n : Nat) : Nat :=
  -- function to convert base 8 to decimal
  sorry

def convert_decimal_to_base16 (n : Nat) : String :=
  -- function to convert decimal to base 16
  sorry

theorem base_8_addition_and_conversion :
  let sum_base8 := add_in_base_8 537 246 in
  sum_base8 = 1005 ∧
  convert_decimal_to_base16 (convert_base8_to_decimal sum_base8) = "205" :=
by
  sorry

end base_8_addition_and_conversion_l518_518360


namespace tractors_in_first_scenario_l518_518186

theorem tractors_in_first_scenario (T : ℕ) (h1 : T * 12 = 15 * 6.4) : T = 8 :=
by
  sorry

end tractors_in_first_scenario_l518_518186


namespace point_on_y_axis_point_equal_distance_to_axes_l518_518959

theorem point_on_y_axis (a : ℝ) (P : ℝ × ℝ) (hP : P = (2 + a, 3 * a - 6)) 
  (h : P.1 = 0) : a = -2 ∧ P = (0, -12) := 
by
  sorry

theorem point_equal_distance_to_axes (a : ℝ) (P : ℝ × ℝ) (hP : P = (2 + a, 3 * a - 6)) 
  (h : ∥P.1∥ = ∥P.2∥) : (a = 4 ∧ P = (6, 6)) ∨ (a = 1 ∧ P = (3, -3)) := 
by
  sorry

end point_on_y_axis_point_equal_distance_to_axes_l518_518959


namespace problem1_problem2_l518_518753

-- Problem 1
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) : (b^2 / a) + (a^2 / b) ≥ a + b :=
by 
  sorry

-- Problem 2
theorem problem2 (a b : ℝ) (ha : a > 0) (hb : b > 0) : (a + b) / 2 ≤ sqrt ((a^2 + b^2) / 2) :=
by
  sorry

end problem1_problem2_l518_518753


namespace prob_double_of_three_l518_518760

def prob (d : ℕ) := Real.log10(d * d + 1) - Real.log10(d * d)

theorem prob_double_of_three :
  let double_prob_three := 2 * (prob 3)
  ∑ d in ({6,7,8,9} : Finset ℕ), prob d = double_prob_three :=
by
  sorry

end prob_double_of_three_l518_518760


namespace proof_l518_518043

noncomputable def problem_statement : Prop :=
  let a := Real.arccos (4 / 5)
  let b := Real.arcsin (1 / 2)
  ∃ (a : ℝ), ∃ (b : ℝ),
    Real.cos a = 4 / 5 ∧
    Real.sin b = 1 / 2 ∧
    Real.cos (a - b) = (4 * Real.sqrt 3 + 3) / 10

theorem proof : problem_statement :=
begin
  sorry
end

end proof_l518_518043


namespace evaluate_parity_of_expression_l518_518200

theorem evaluate_parity_of_expression (a b c d : ℕ) (ha : odd a) (hb : odd b) (hd : even d) :
  (odd c → even (3^a + (b-1)^2 * c - (2^d - c))) ∧ (even c → odd (3^a + (b-1)^2 * c - (2^d - c))) :=
by
  sorry

end evaluate_parity_of_expression_l518_518200


namespace limit_sum_perimeters_areas_of_isosceles_triangles_l518_518791

theorem limit_sum_perimeters_areas_of_isosceles_triangles (b s h : ℝ) : 
  ∃ P A : ℝ, 
    (P = 2*(b + 2*s)) ∧ 
    (A = (2/3)*b*h) :=
  sorry

end limit_sum_perimeters_areas_of_isosceles_triangles_l518_518791


namespace find_a_value_l518_518117

variable (a : ℝ)
def complex_eq_condition : Prop :=
  let z := (a - complex.I) * (1 - complex.I)
  z.re = z.im

theorem find_a_value (h: complex_eq_condition a) : a = 1 := by
  sorry

end find_a_value_l518_518117


namespace infinite_divisible_269_l518_518624

theorem infinite_divisible_269 (a : ℕ → ℤ) (h₀ : a 0 = 2) (h₁ : a 1 = 15) 
  (h_recur : ∀ n : ℕ, a (n + 2) = 15 * a (n + 1) + 16 * a n) :
  ∃ infinitely_many k: ℕ, 269 ∣ a k :=
by
  sorry

end infinite_divisible_269_l518_518624


namespace canoe_no_paddle_time_l518_518069

-- All conditions needed for the problem
variables {S v v_r : ℝ}
variables (time_pa time_pb : ℝ)

-- Condition that time taken from A to B is 3 times the time taken from B to A
def condition1 : Prop := time_pa = 3 * time_pb

-- Define time taken from A to B (downstream) and B to A (upstream)
def time_pa_def : time_pa = S / (v + v_r) := sorry
def time_pb_def : time_pb = S / (v - v_r) := sorry

-- Main theorem stating the problem to prove
theorem canoe_no_paddle_time :
  condition1 →
  ∃ (t_no_paddle : ℝ), t_no_paddle = 3 * time_pb :=
begin
  intro h1,
  sorry
end

end canoe_no_paddle_time_l518_518069


namespace acute_triangle_l518_518307

variable (A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ 0 < A ∧ 0 < B ∧ 0 < C

def each_angle_less_than_sum_of_others (A B C : ℝ) : Prop :=
  A < B + C ∧ B < A + C ∧ C < A + B

theorem acute_triangle (A B C : ℝ) 
  (h1 : is_triangle A B C) 
  (h2 : each_angle_less_than_sum_of_others A B C) : 
  A < 90 ∧ B < 90 ∧ C < 90 := 
sorry

end acute_triangle_l518_518307


namespace clocks_resynchronize_after_days_l518_518794

/-- Arthur's clock gains 15 minutes per day. -/
def arthurs_clock_gain_per_day : ℕ := 15

/-- Oleg's clock gains 12 minutes per day. -/
def olegs_clock_gain_per_day : ℕ := 12

/-- The clocks display time in a 12-hour format, which is equivalent to 720 minutes. -/
def twelve_hour_format_in_minutes : ℕ := 720

/-- 
  After how many days will this situation first repeat given the 
  conditions of gain in Arthur's and Oleg's clocks and the 12-hour format.
-/
theorem clocks_resynchronize_after_days :
  ∃ (N : ℕ), N * arthurs_clock_gain_per_day % twelve_hour_format_in_minutes = 0 ∧
             N * olegs_clock_gain_per_day % twelve_hour_format_in_minutes = 0 ∧
             N = 240 :=
by
  sorry

end clocks_resynchronize_after_days_l518_518794


namespace proposition_1_proposition_2_proposition_3_l518_518090

def seq_a : ℕ → ℕ 
| 1 := 3
| 2 := 5
| (n + 3) := seq_a (n + 2) + 2^(n + 1)

def b_n (n : ℕ) : ℚ := 1 / (seq_a n * seq_a (n + 1))

def f (x : ℕ) : ℚ := 2^(x - 1)

def T_n (n : ℕ) : ℚ := (Finset.range n).sum (λ i, b_n (i + 1) * f (i + 1))

theorem proposition_1 (n : ℕ) : seq_a n = 2^n + 1 := sorry

theorem proposition_2 (n : ℕ) : 1 ≤ n → T_n n < 1/6 := sorry

theorem proposition_3 (a : ℚ) (h : 0 < a) : 
  (∀ n, T_n n < 1/6) ∧ 
  (∀ m ∈ Ioo 0 (1/6 : ℚ), ∃ n0 : ℕ, ∀ n ≥ n0, T_n n > m) ↔ a = 2 := 
sorry

end proposition_1_proposition_2_proposition_3_l518_518090


namespace max_bats_purchase_l518_518354

variables (B C : ℝ) (X : ℝ)

-- Given conditions
def condition1 := 2 * B + 4 * C = 200
def condition2 := B + 6 * C = 220

-- Function to calculate maximum number of sets
def max_sets (B C X : ℝ) : ℝ := ⌊ X / (B + C) ⌋

-- Lean statement for the proof problem
theorem max_bats_purchase (B C X: ℝ) (h1: 2 * B + 4 * C = 200) (h2: B + 6 * C = 220) :
  max_sets B C X = ⌊ X / (B + C) ⌋ :=
by
  sorry

end max_bats_purchase_l518_518354


namespace islanders_liars_l518_518638

theorem islanders_liars (n : ℕ) (h : n = 450) : (∃ L : ℕ, (L = 150 ∨ L = 450)) :=
sorry

end islanders_liars_l518_518638


namespace partition_n_iff_multiple_of_3_l518_518045

def is_arithmetic_mean (a b c d : ℕ) : Prop :=
  d = (a + b + c) / 3

def M (n : ℕ) : set ℕ := { i | 1 ≤ i ∧ i ≤ 4 * n }

def is_partitioned (n : ℕ) (partition: list (set ℕ)) : Prop :=
  (∀ k ∈ partition, ∃ a b c d, k = {a, b, c, d} ∧ is_arithmetic_mean a b c d) ∧
  (⋃ k ∈ partition, k) = M n ∧
  partition.pairwise disjoint

theorem partition_n_iff_multiple_of_3 (n : ℕ) :
  ∃ partition: list (set ℕ), is_partitioned n partition ↔ n % 3 = 0 :=
sorry

end partition_n_iff_multiple_of_3_l518_518045


namespace sq_sum_ge_one_third_l518_518752

theorem sq_sum_ge_one_third (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry

end sq_sum_ge_one_third_l518_518752


namespace trigonometric_identity_l518_518700

-- Define the angles in degrees
noncomputable def angle1 := 20 * Real.pi / 180

-- Define the expression
def expr := (Real.cos angle1) * (Real.cos angle1) - (Real.sin angle1) * (Real.sin angle1)

-- State the theorem
theorem trigonometric_identity : expr = 1 / 2 := by
  sorry

end trigonometric_identity_l518_518700


namespace probability_wife_alive_for_10_years_l518_518695

variable (P : String → ℝ)

theorem probability_wife_alive_for_10_years :
  (P "Man" = 1/4) →
  (P "Neither" = 0.5) →
  (∃ P_W, P_W = 1/3) :=
by
  intros hMan hNeither
  use 1/3
  sorry

end probability_wife_alive_for_10_years_l518_518695


namespace functional_eq_solution_l518_518605

theorem functional_eq_solution (f : ℤ → ℤ) (h : ∀ x y : ℤ, x ≠ 0 →
  x * f (2 * f y - x) + y^2 * f (2 * x - f y) = (f x ^ 2) / x + f (y * f y)) :
  (∀ x: ℤ, f x = 0) ∨ (∀ x : ℤ, f x = x^2) :=
sorry

end functional_eq_solution_l518_518605


namespace problem_statement_l518_518353

theorem problem_statement : 
  let m := 377
  let n := 4096
  m.gcd n = 1 →
  (m + n) = 4473 := 
by {
  intros hmgn,
  sorry
}

end problem_statement_l518_518353


namespace Margie_distance_on_25_dollars_l518_518630

theorem Margie_distance_on_25_dollars
  (miles_per_gallon : ℝ)
  (cost_per_gallon : ℝ)
  (amount_spent : ℝ) :
  miles_per_gallon = 40 →
  cost_per_gallon = 5 →
  amount_spent = 25 →
  (amount_spent / cost_per_gallon) * miles_per_gallon = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end Margie_distance_on_25_dollars_l518_518630


namespace adjusted_price_tv_in_2009_l518_518397

def adjusted_price_iter (init_price : ℝ) (D p r : ℝ) (years : ℕ) : ℝ :=
  nat.rec_on years init_price
    (λ n price, (price - D) * (1 - p / 100) * (1 + r / 100))

theorem adjusted_price_tv_in_2009 (P_2001 P_2009 : ℝ) (D p r : ℝ) :
  P_2001 = 1950 → D = 35 → P_2009 = adjusted_price_iter P_2001 D p r 8 → 
  P_2009 = ( ( ( ( ( ( ( ( (P_2001 - D) * (1 - p / 100) * (1 + r / 100)) - D) * (1 - p / 100) * (1 + r / 100)) - D) * (1 - p / 100) * (1 + r / 100)) - D) * (1 - p / 100) * (1 + r / 100)) - D) * (1 - p / 100) * (1 + r / 100)) - D) * (1 - p / 100) * (1 + r / 100)) := sorry

end adjusted_price_tv_in_2009_l518_518397


namespace area_of_quadrilateral_l518_518707

open Complex

theorem area_of_quadrilateral : 
  ∃ (z : ℂ) (x y : ℤ), 
    z = x + y * I ∧ 
    z * (conj z)^3 + (conj z) * z^3 + 100 = 450 ∧ 
    z ∈ {4 + 3 * I, 4 - 3 * I, -4 + 3 * I, -4 - 3 * I}
    → quadrilateral_area {4 + 3 * I, 4 - 3 * I, -4 + 3 * I, -4 - 3 * I} = 48 :=
sorry

def quadrilateral_area (s: set ℂ) : ℝ := sorry

end area_of_quadrilateral_l518_518707


namespace minimize_r3_locus_maximize_r3_locus_l518_518087

-- Definitions and conditions
structure Pentagon :=
  (vertices : Fin 5 → ℝ × ℝ) -- five vertices
  (side_length : ℝ)
  (regular : ∀ i, dist (vertices i) (vertices (i + 1) % 5) = side_length)

def distances (M : ℝ × ℝ) (p : Pentagon) : Fin 5 → ℝ :=
  fun i => dist M (p.vertices i)

def sorted_distances (M : ℝ × ℝ) (p : Pentagon) : Vector ℝ (5) :=
  (insertionSort (distances M p)).val

def r3 (M : ℝ × ℝ) (p : Pentagon) : ℝ :=
  sorted_distances M p 2

-- Statement (a): Locus minimizing r3
theorem minimize_r3_locus (p : Pentagon) (h : p.side_length = 1) :
  let minimal_locus (x y : ℝ) : Prop := ∃ i, 0 ≤ i < 5 ∧ (x, y) = midpoint (p.vertices i) (p.vertices ((i + 2) % 5)) in
  ∀ M : ℝ × ℝ, minimal_locus M.1 M.2 → r3 M p = 0.8090 := sorry

-- Statement (b): Locus maximizing r3
theorem maximize_r3_locus (p : Pentagon) (h : p.side_length = 1) :
  let maximal_locus (x y : ℝ) : Prop := ∃ i, 0 ≤ i < 5 ∧ (x, y) = midpoint (p.vertices i) (p.vertices ((i + 1) % 5)) in
  ∀ M : ℝ × ℝ, maximal_locus M.1 M.2 → r3 M p = 1.5590 := sorry

end minimize_r3_locus_maximize_r3_locus_l518_518087


namespace range_of_p_l518_518438

theorem range_of_p 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, S n = (-1 : ℝ)^n * a n + 1/(2^n) + n - 3)
  (h2 : ∀ n : ℕ, (a (n + 1) - p) * (a n - p) < 0) :
  -3/4 < p ∧ p < 11/4 :=
sorry

end range_of_p_l518_518438


namespace part_i_monotonic_intervals_part_ii_l518_518892

noncomputable def f (x a : ℝ) : ℝ := x * log x - a * (x - 1)^2 - x + 1

theorem part_i_monotonic_intervals (a : ℝ) (x : ℝ) :
  a = 0 →
  (∀ x, x ∈ Ioo 0 1 → deriv (f x 0) x < 0) ∧
  (∀ x, x ∈ Ioi 1 → deriv (f x 0) x > 0) ∧
  f 1 0 = 0 ∧
  (∀ y, y ∈ Ioi 0 → f y 0 ≥ 0) :=
by
  intros 
  sorry

theorem part_ii (x a : ℝ) :
  x > 1 → a ≥ 1/2 → f x a < 0 :=
by
  intros
  sorry

end part_i_monotonic_intervals_part_ii_l518_518892


namespace avg_difference_l518_518673

theorem avg_difference : 
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 80 + 15) / 3
  avg1 - avg2 = 5 :=
by
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 80 + 15) / 3
  show avg1 - avg2 = 5
  sorry

end avg_difference_l518_518673


namespace vertex_coordinates_max_value_of_n_range_of_m_l518_518089

theorem vertex_coordinates (m : ℝ) (h : m = 3) : 
  (∃ (x_vertex y_vertex : ℝ), y_vertex = 6 ∧ x_vertex = 2 ∧ y = -x^2 + 4*x + 2) := 
sorry

theorem max_value_of_n (m : ℝ) (h : m = 3) (n : ℝ) : 
  (n = 2 + Real.sqrt 3 ∨ n = -Real.sqrt 3) → 
  (∀ x, n ≤ x ∧ x ≤ n + 2 → y = -x^2 + 4*x + 2 → y ≤ 3) := 
sorry

theorem range_of_m : 
  (∀ y, ∃ x₁ x₂ : ℝ, y = -x₁^2 + 4*x₁ + m - 1 ∧ y = -x₂^2 + 4*x₂ + m - 1 ∧ Real.abs y = 3) → 
  (-6 < m ∧ m < 0) := 
sorry

end vertex_coordinates_max_value_of_n_range_of_m_l518_518089


namespace bob_paid_more_than_alice_l518_518003

-- conditions
def total_slices := 12
def plain_pizza_cost := 12
def pepperoni_additional_cost := 3
def mushrooms_additional_cost := 2
def total_cost := 12 + 3 + 2

def cost_per_slice := total_cost / total_slices

def bob_slices := 4 + 2
def charlie_slices := 4 + 1
def alice_slices := total_slices - bob_slices - charlie_slices

def bob_cost := bob_slices * cost_per_slice
def alice_cost := alice_slices * cost_per_slice

-- proof statement
theorem bob_paid_more_than_alice : bob_cost - alice_cost = 4.26 := 
by sorry

end bob_paid_more_than_alice_l518_518003


namespace coordinates_of_B_l518_518458

def initial_coords : ℤ × ℤ := (-5, 1)

def move_right (p : ℤ × ℤ) (units : ℤ) : ℤ × ℤ := (p.1 + units, p.2)

def move_up (p : ℤ × ℤ) (units : ℤ) : ℤ × ℤ := (p.1, p.2 + units)

theorem coordinates_of_B' : 
  let B' := move_up (move_right initial_coords 4) 2 in B' = (-1, 3) :=
by 
  sorry

end coordinates_of_B_l518_518458


namespace slow_speed_distance_l518_518542

theorem slow_speed_distance (D : ℝ) (h : (D + 20) / 14 = D / 10) : D = 50 := by
  sorry

end slow_speed_distance_l518_518542


namespace solve_for_x_l518_518232
-- Lean 4 Statement

theorem solve_for_x (x : ℝ) (h : 2^(3 * x) = Real.sqrt 32) : x = 5 / 6 := 
sorry

end solve_for_x_l518_518232


namespace binom_congruent_mod_2n_l518_518189

theorem binom_congruent_mod_2n (n : ℕ) (hn : n > 0) :
  ∃ (f : Fin (2^(n-1)) → Fin (2^n)), 
    (∀ k : Fin (2^(n-1)), f k ∈ {k | ∃ m : ℕ, m ≥ 1 ∧ 2 * m - 1 = k % 2^n})
    ∧ (∀ i j : Fin (2^(n-1)), i ≠ j → f i ≠ f j)
    ∧ (∀ k : Fin (2^(n-1)), ((2^n - 1).choose k.val) % 2^n = f k.val) :=
by
  sorry

end binom_congruent_mod_2n_l518_518189


namespace product_AF_CF_eq_20_l518_518182

-- Defining the problem with all given conditions.
variables {A B C P F : Point}
variables (h1 : dist P A = dist P C)
variables (h2 : ∠APC = 2 * ∠ABC)
variables (h3 : ∃O, Circle O A B C)
variables (h4 : dist P C = 5)
variables (h5 : dist P F = 4)

-- Theorem to prove the product AF * CF.
theorem product_AF_CF_eq_20 : dist A F * dist C F = 20 :=
by
  sorry

end product_AF_CF_eq_20_l518_518182


namespace max_n_for_factorable_polynomial_l518_518412

theorem max_n_for_factorable_polynomial :
  ∃ A B : ℤ, AB = 144 ∧ (A + 6 * B = 865) :=
begin
  sorry
end

end max_n_for_factorable_polynomial_l518_518412


namespace range_of_a_l518_518477

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ set.Icc 0 1 → (a * x^3 - 3 * x + 1) ≥ 0) → (a ≥ 4) := 
by 
  sorry

end range_of_a_l518_518477


namespace distribute_balls_in_boxes_l518_518513

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l518_518513


namespace arrow_hits_apple_l518_518628

noncomputable def time_to_hit (L V0 : ℝ) (α β : ℝ) : ℝ :=
  (L / V0) * (Real.sin β / Real.sin (α + β))

theorem arrow_hits_apple (g : ℝ) (L V0 : ℝ) (α β : ℝ) (h : (L / V0) * (Real.sin β / Real.sin (α + β)) = 3 / 4) 
  : time_to_hit L V0 α β = 3 / 4 := 
  by
  sorry

end arrow_hits_apple_l518_518628


namespace correct_answers_l518_518113

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
noncomputable def g (x : ℝ) : ℝ := f' x

-- Conditions
axiom f_domain : ∀ x, f x ∈ ℝ
axiom f'_domain : ∀ x, f' x ∈ ℝ
axiom g_def : ∀ x, g x = f' x
axiom f_even : ∀ x, f (3 / 2 - 2 * x) = f (3 / 2 + 2 * x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- Proof Problem
theorem correct_answers : f (-1) = f 4 ∧ g (-1 / 2) = 0 :=
  by
    sorry

end correct_answers_l518_518113


namespace find_monic_polynomial_l518_518999

-- Define the original polynomial
def original_polynomial := (x : ℚ) : ℚ := 
  x^3 - 4 * x^2 + 5 * x + 2

-- State the problem
theorem find_monic_polynomial :
  let p q r : ℚ := roots_of original_polynomial in
  let new_polynomial := 
    x^3 - 12 * x^2 + 45 * x + 54 in
  ∀ x : ℚ, (x - 3 * p) * (x - 3 * q) * (x - 3 * r) = new_polynomial :=
begin
  sorry
end

end find_monic_polynomial_l518_518999


namespace XY_distance_l518_518193

variables {A B C D X Y P Q : Type*}
variables [metric_space A]

-- Let \(A, B, C, D\) be points on a circle
variable (circle : Set A)
variable (on_circle : ∀ (x : A), x ∈ circle → x ∈ circle)

-- Conditions
variable (d_AB : ℝ)
variable (d_CD : ℝ)
variable (d_AP : ℝ)
variable (d_CQ : ℝ)
variable (d_PQ : ℝ)
variable (d_XY : ℝ)

-- Given Conditions
axiom h_AB : d_AB = 13
axiom h_CD : d_CD = 23
axiom h_AP : d_AP = 7
axiom h_CQ : d_CQ = 9
axiom h_PQ : d_PQ = 31

-- Question to Prove
theorem XY_distance :
  d_XY = 35.72 :=
sorry

end XY_distance_l518_518193


namespace balls_in_boxes_l518_518509

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l518_518509


namespace angle_difference_l518_518016

noncomputable def triangle_angles {A B C C1 C2 : ℝ} : Prop :=
  A = B - 15 ∧ C1 = 90 - A ∧ C2 = 90 - B

theorem angle_difference {A B C C1 C2 : ℝ} (h : triangle_angles A B C C1 C2) :
  C1 - C2 = 15 := by
  sorry

end angle_difference_l518_518016


namespace bisection_method_example_l518_518731

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 4

theorem bisection_method_example :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0) →
  (∃ x : ℝ, (1 / 2) < x ∧ x < 1 ∧ f x = 0) :=
by
  sorry

end bisection_method_example_l518_518731


namespace second_ball_probability_l518_518086

-- Definitions and conditions
def red_balls := 3
def white_balls := 2
def black_balls := 5
def total_balls := red_balls + white_balls + black_balls

def first_ball_white_condition : Prop := (white_balls / total_balls) = (2 / 10)
def second_ball_red_given_first_white (first_ball_white : Prop) : Prop :=
  (first_ball_white → (red_balls / (total_balls - 1)) = (1 / 3))

-- Mathematical equivalence proof problem statement in Lean
theorem second_ball_probability : 
  first_ball_white_condition ∧ second_ball_red_given_first_white first_ball_white_condition :=
by
  sorry

end second_ball_probability_l518_518086


namespace balls_in_boxes_l518_518532

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l518_518532


namespace ball_box_problem_l518_518498

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l518_518498


namespace prod_inequality_geometric_mean_l518_518433

theorem prod_inequality_geometric_mean (n : ℕ) (a b c d e : ℕ → ℝ) 
    (ha : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 1 < a i)
    (hb : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 1 < b i)
    (hc : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 1 < c i)
    (hd : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 1 < d i)
    (he : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 1 < e i) : 
  let A := (∑ i in Finset.range n, a i) / n,
      B := (∑ i in Finset.range n, b i) / n,
      C := (∑ i in Finset.range n, c i) / n,
      D := (∑ i in Finset.range n, d i) / n,
      E := (∑ i in Finset.range n, e i) / n
  in 
  (∏ i in Finset.range n, (a i * b i * c i * d i * e i + 1) / (a i * b i * c i * d i * e i - 1))
  ≥ ((A * B * C * D * E + 1) / (A * B * C * D * E - 1)) ^ n := 
sorry

end prod_inequality_geometric_mean_l518_518433


namespace minimum_value_of_polynomial_l518_518555

def polynomial (a b : ℝ) : ℝ := 2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999

theorem minimum_value_of_polynomial : ∃ (a b : ℝ), polynomial a b = 1947 :=
by
  sorry

end minimum_value_of_polynomial_l518_518555


namespace number_of_lines_passing_through_P_l518_518868

theorem number_of_lines_passing_through_P 
  (a b : Line) (P : Point) (ha : skew_lines a b)
  (θ : Angle) (hθ : θ = 60) :
  ∃! l : Line, passes_through l P ∧ angle l a = θ ∧ angle l b = θ :=
begin
  -- Proof goes here
  sorry
end

end number_of_lines_passing_through_P_l518_518868


namespace distance_traveled_l518_518544

theorem distance_traveled
  (D : ℝ) (T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 14 * T)
  : D = 50 := sorry

end distance_traveled_l518_518544


namespace largest_number_of_consecutive_integers_sum_21_l518_518691

theorem largest_number_of_consecutive_integers_sum_21 :
  ∀ (n : ℕ), (∃ (a : ℕ), (finset.range n).sum (λ i, a + i) = 21) → n ≤ 6 :=
by
  sorry

end largest_number_of_consecutive_integers_sum_21_l518_518691


namespace price_decrease_in_may_l518_518951

theorem price_decrease_in_may :
  let P0 : ℝ := 100 in
  let P1 := P0 * 1.15 in
  let P2 := P1 * 0.90 in
  let P3 := P2 * 1.20 in
  let P4 := P3 * 0.85 in
  let P5 := P4 * (1 - y / 100) in
  P5 = P0 → y = 5 :=
by
  sorry

end price_decrease_in_may_l518_518951


namespace trapezoid_ratio_l518_518359

theorem trapezoid_ratio
  (A B C D E O: Point) (R: ℝ) 
  (h_circ: Circle O R)
  (h_trap: IsTrapezoid A B C D)
  (h_long_base: AD > BC)
  (h_perpendicular: ∠COE = 90° ∧ IntersectCircle AD E)
  (h_arc_ratio: ArcLength BC / ArcLength CDE = 1 / 2)
  (h_radius_height: R = Height A B C D): 
  AD / BC = √(4 * √3 - 3) := sorry

end trapezoid_ratio_l518_518359


namespace range_of_curvature_l518_518907

noncomputable def curvature (f : ℝ → ℝ) (f' : ℝ → ℝ) (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := f x₁
  let y₂ := f x₂
  let k₁ := f' x₁
  let k₂ := f' x₂
  let dist_MN := Real.sqrt ((x₂ - x₁) ^ 2 + (y₂ - y₁) ^ 2)
  |k₁ - k₂| / dist_MN

theorem range_of_curvature :
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → x₁ + x₂ = 2 →
  let f := (λ x : ℝ, x ^ 3 + 2)
  let f' := (λ x : ℝ, 3 * x ^ 2)
  let φ := curvature f f' x₁ x₂
  0 < φ ∧ φ < 3 * Real.sqrt (10) / 5 := by
  begin
    intros,
    sorry
  end

end range_of_curvature_l518_518907


namespace M_gt_N_l518_518984

def M (a : ℝ) : ℝ := 2 * a * (a - 2)
def N (a : ℝ) : ℝ := (a + 1) * (a - 3)

theorem M_gt_N (a : ℝ) : M(a) > N(a) := sorry

end M_gt_N_l518_518984


namespace max_union_subset_l518_518147

noncomputable def is_union_pair (a b : ℕ) : Prop :=
  ¬ (Nat.gcd a b = 1) ∧ ¬ (a % b = 0 ∨ b % a = 0)

theorem max_union_subset :
  ∀ (A : Finset ℕ), 
    (∀ a b ∈ A, a ≠ b → is_union_pair a b) ∧ A ⊆ Finset.range 2018 → 
    A.card ≤ 504 :=
by
  sorry

end max_union_subset_l518_518147


namespace polar_eq_AF2_correct_distance_focus_correct_l518_518464

def conic_curve (alpha : ℝ) : ℝ × ℝ :=
  (2 * Real.cos alpha, Real.sqrt 3 * Real.sin alpha)

def point_A : ℝ × ℝ := (0, Real.sqrt 3)

def foci_F1_F2 : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, 0), (1, 0))

-- Polar equation of line AF2 rewritten
def polar_eq_AF2 (r θ : ℝ) : Prop :=
  r * Real.sin θ = (Real.sqrt 3) - r * (Real.cos θ) * (Real.sqrt 3)

theorem polar_eq_AF2_correct (r θ : ℝ) :
  conic_curve θ = (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ) →
  point_A = (0, Real.sqrt 3) →
  foci_F1_F2 = ((-1, 0), (1, 0)) →
  polar_eq_AF2 r θ := sorry

theorem distance_focus_correct (M N : ℝ × ℝ) :
  conic_curve θ = (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ) →
  point_A = (0, Real.sqrt 3) →
  foci_F1_F2 = ((-1, 0), (1, 0)) →
  foci_F1_F2.fst.1 = M.1 →
  foci_F1_F2.fst.2 = N.2 →
  abs (dist M foci_F1_F2.fst - dist N foci_F1_F2.fst) = (12 * Real.sqrt 3) / 13 := sorry

end polar_eq_AF2_correct_distance_focus_correct_l518_518464


namespace smallest_product_of_digits_5678_l518_518215

theorem smallest_product_of_digits_5678 : 
  ∃ (a b c d : ℕ), (a ∈ {5, 6, 7, 8}) ∧ (b ∈ {5, 6, 7, 8}) ∧ (c ∈ {5, 6, 7, 8}) ∧ (d ∈ {5, 6, 7, 8}) 
  ∧ (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) 
  ∧ ∃ (x y : ℕ), (x = 10 * a + b) ∧ (y = 10 * c + d) ∧ (x * y = 3876) :=
by
  sorry

end smallest_product_of_digits_5678_l518_518215


namespace velocity_at_t_10_time_to_reach_max_height_max_height_l518_518686

-- Define the height function H(t)
def H (t : ℝ) : ℝ := 200 * t - 4.9 * t^2

-- Define the velocity function v(t) as the derivative of H(t)
def v (t : ℝ) : ℝ := 200 - 9.8 * t

-- Theorem: The velocity of the body at t = 10 seconds
theorem velocity_at_t_10 : v 10 = 102 := by
  sorry

-- Theorem: The time to reach maximum height
theorem time_to_reach_max_height : (∃ t : ℝ, v t = 0 ∧ t = 200 / 9.8) := by
  sorry

-- Theorem: The maximum height the body will reach
theorem max_height : H (200 / 9.8) = 2040.425 := by
  sorry

end velocity_at_t_10_time_to_reach_max_height_max_height_l518_518686


namespace copies_per_minute_first_machine_l518_518339

-- Define the problem conditions
variable (x : Nat) -- x denotes the number of copies the first machine makes per minute
def second_machine_rate := 55 -- The second machine makes 55 copies per minute
def total_copies := 2550 -- Total copies made by both machines in half an hour (30 minutes)
def time_minutes := 30 -- Half an hour is 30 minutes

-- Specify the condition to solve for x
theorem copies_per_minute_first_machine : 
  30 * x + 30 * second_machine_rate = total_copies → x = 30 :=
by 
  intros h,
  sorry

end copies_per_minute_first_machine_l518_518339


namespace ratio_of_weights_l518_518297

noncomputable def tyler_weight (sam_weight : ℝ) : ℝ := sam_weight + 25
noncomputable def ratio_of_peter_to_tyler (peter_weight tyler_weight : ℝ) : ℝ := peter_weight / tyler_weight

theorem ratio_of_weights (sam_weight : ℝ) (peter_weight : ℝ) (h_sam : sam_weight = 105) (h_peter : peter_weight = 65) :
  ratio_of_peter_to_tyler peter_weight (tyler_weight sam_weight) = 0.5 := by
  -- We use the conditions to derive the information
  sorry

end ratio_of_weights_l518_518297


namespace no_statistics_increase_l518_518374

def scores := [39, 42, 47, 51, 51, 55, 55, 55, 59, 62, 66, 76]
def new_score := 39
def new_scores := scores ++ [new_score]

def range (l : List ℕ) : ℕ := l.maximum - l.minimum
def median (l : List ℕ) : ℕ := 
  let sorted := l.qsort (· ≤ ·)
  if sorted.length % 2 = 0 
  then (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2
  else sorted.get (sorted.length / 2)
def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / (l.length : ℚ)
def mode (l : List ℕ) : ℕ := l.group_by id (· == ·).max_by (λ g => g.length).head!
def midrange (l : List ℕ) : ℚ := (l.maximum + l.minimum) / 2

theorem no_statistics_increase :
  range new_scores = range scores ∧
  median new_scores = median scores ∧
  mean new_scores < mean scores ∧
  mode new_scores = mode scores ∧
  midrange new_scores = midrange scores :=
by
  sorry

end no_statistics_increase_l518_518374


namespace subset_P_Q_l518_518625

-- Definitions of the sets P and Q
def P : Set ℝ := {x | x^2 - 3 * x + 2 < 0}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- Statement to prove P ⊆ Q
theorem subset_P_Q : P ⊆ Q :=
sorry

end subset_P_Q_l518_518625


namespace wax_needed_l518_518214

theorem wax_needed (current_wax total_wax : ℕ) (hc : current_wax = 11) (ht : total_wax = 492) : 
  total_wax - current_wax = 481 :=
by {
  rw [hc, ht],
  exact rfl,
}

end wax_needed_l518_518214


namespace cube_same_color_probability_l518_518340

theorem cube_same_color_probability : 
∀ (red blue : Prop) 
  (faces : fintype (fin 6)) 
  (paint : faces → bool), 
  (1 - (1/2)^6 = 63/64) :=
by 
  sorry 

end cube_same_color_probability_l518_518340


namespace problem_l518_518470

def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 + 1 else x

theorem problem : f 4 - f (-2) = -1 :=
by sorry

end problem_l518_518470


namespace math_quiz_l518_518159

theorem math_quiz (x : ℕ) : 
  (∃ x ≥ 14, (∃ y : ℕ, 16 = x + y + 1) → (6 * x - 2 * y ≥ 75)) → 
  x ≥ 14 :=
by
  sorry

end math_quiz_l518_518159


namespace paint_wall_together_l518_518741

theorem paint_wall_together (h_work_rate : ℚ) (g_work_rate : ℚ) (combined_time : ℚ) :
  h_work_rate = 1 / 3 ∧ g_work_rate = 1 / 6 ∧ combined_time = 1 / (h_work_rate + g_work_rate) → combined_time = 2 :=
begin
  intros h,
  cases h with h_hr h_g_rt,
  cases h_g_rt with g_hr ct,
  rw [ct, h_hr, g_hr],
  norm_num,
end

end paint_wall_together_l518_518741


namespace sum_of_differences_l518_518163

theorem sum_of_differences (a b : ℝ) (n : ℕ) (seq : ℕ → ℝ) (h1 : seq 1 = a) (hn : seq n = b) :
  (∑ i in Finset.range (n - 1), (seq (i + 2) - seq (i + 1))) = b - a :=
by 
  sorry

end sum_of_differences_l518_518163


namespace problem_l518_518460

noncomputable def f : ℝ → ℝ :=
  if h : 0 < x ∧ x < 1 then 4^x
  else if h' : x < 0 then -f (-x)
  else f (x - 2 * ⌊(x + 1) / 2⌋)

theorem problem :
  (f (-5 / 2) + f 1) = -2 :=
sorry

end problem_l518_518460


namespace ending_number_l518_518705

theorem ending_number (num_even_integers : ℕ) (first_even_after_25 : ℕ) : ℕ :=
let even_count := num_even_integers - 1,
    ending_number := first_even_after_25 + even_count * 2 in
ending_number
    
example : ending_number 35 26 = 94 :=
by 
  unfold ending_number 
  simp
  sorry

end ending_number_l518_518705


namespace fill_bathtub_with_hot_water_in_192_seconds_l518_518148

theorem fill_bathtub_with_hot_water_in_192_seconds
  (t_cold : ℕ) (t_both : ℕ)
  (vcold : ℚ := 1 / t_cold) (vhot : ℚ := 1 / t_hot) 
  (t_hot_proof : ℚ := 1 / vhot)
  (combined_rate : ℚ := vcold + vhot := 1 / t_both) : t_hot_proof = 192 :=
by
  sorry

end fill_bathtub_with_hot_water_in_192_seconds_l518_518148


namespace complex_simplification_l518_518661

variables (a b c d e f g : ℤ)

theorem complex_simplification :
  (7 - 4 * complex.I) - (2 + 6 * complex.I) + (3 - 3 * complex.I) = 8 - 13 * complex.I :=
by
  sorry

end complex_simplification_l518_518661


namespace max_selection_no_two_diff_17_l518_518056

theorem max_selection_no_two_diff_17 {n : ℕ} (h : n = 2013) : 
  ∃ S : Finset ℕ, S.card = 1010 ∧ (∀ a b ∈ S, a ≠ b → abs (a - b) ≠ 17) :=
by
  sorry

end max_selection_no_two_diff_17_l518_518056


namespace balls_in_boxes_l518_518506

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l518_518506


namespace probability_red_tile_l518_518759

theorem probability_red_tile : 
  let red_tiles := setOf (λ n, ∃ k : ℕ, n = 3 + 7 * k ∧ n ≤ 98 ∧ 1 ≤ n)
  let total_tiles := {n | 1 ≤ n ∧ n ≤ 98}
  red_tiles.card / total_tiles.card = 1 / 7 :=
by
  sorry

end probability_red_tile_l518_518759


namespace tan_beta_minus_2alpha_l518_518925

theorem tan_beta_minus_2alpha (alpha beta : ℝ) (h1 : Real.tan alpha = 2) (h2 : Real.tan (beta - alpha) = 3) : 
  Real.tan (beta - 2 * alpha) = 1 / 7 := 
sorry

end tan_beta_minus_2alpha_l518_518925


namespace max_f_and_sin_alpha_l518_518427

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * Real.cos x

theorem max_f_and_sin_alpha :
  (∀ x : ℝ, f x ≤ Real.sqrt 5) ∧ (∃ α : ℝ, (α + Real.arccos (1 / Real.sqrt 5) = π / 2 + 2 * π * some_integer) ∧ (f α = Real.sqrt 5) ∧ (Real.sin α = 1 / Real.sqrt 5)) :=
by
  sorry

end max_f_and_sin_alpha_l518_518427


namespace gcd_lcm_sum_eq_90_l518_518602

def gcd_three (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem gcd_lcm_sum_eq_90 : 
  let A := gcd_three 18 36 72
  let B := lcm_three 18 36 72
  A + B = 90 :=
by
  let A := gcd_three 18 36 72
  let B := lcm_three 18 36 72
  sorry

end gcd_lcm_sum_eq_90_l518_518602


namespace median_length_is_five_l518_518767

theorem median_length_is_five :
  let l := [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7] in
  multiset.median (multiset.ofList l) = 5 :=
by
  sorry

end median_length_is_five_l518_518767


namespace balloons_allan_brought_l518_518365

theorem balloons_allan_brought:
  ∀ (initial_balloons: ℕ) (additional_balloons: ℕ), initial_balloons = 5 → additional_balloons = 3 →
  initial_balloons + additional_balloons = 8 :=
by
  intros initial_balloons additional_balloons h_initial h_additional
  rw [h_initial, h_additional]
  rfl

end balloons_allan_brought_l518_518365


namespace product_equals_permutation_l518_518325

-- Definitions and conditions
def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Given product sequence
def product_seq (n k : ℕ) : ℕ :=
  (List.range' (n - k + 1) k).foldr (λ x y => x * y) 1

-- Problem statement: The product of numbers from 18 to 9 is equivalent to A_{18}^{10}
theorem product_equals_permutation :
  product_seq 18 10 = perm 18 10 :=
by
  sorry

end product_equals_permutation_l518_518325


namespace tan_alpha_value_l518_518142

theorem tan_alpha_value (α : ℝ) (h : 2 * sin (2 * α) = 1 - cos (2 * α)) : tan α = 2 ∨ tan α = 0 := by
  sorry

end tan_alpha_value_l518_518142


namespace quadratic_function_expression_l518_518461

/-!
# Problem Statement
Prove that, given the vertex of the quadratic function at \(A(-1, 4)\) and the point \(B(2, -5)\) on its graph, the expression of the quadratic function is \(y = -x^2 - 2x + 3\).
-/

theorem quadratic_function_expression (a b c : ℝ) (x y : ℝ) (h1 : (∀ x, y = a * (x + 1)^2 + 4)) (h2 : y = -x^2 - 2x + 3) :
  y = -x^2 - 2x + 3 := 
sorry

end quadratic_function_expression_l518_518461


namespace distribute_balls_in_boxes_l518_518516

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l518_518516


namespace islanders_liars_l518_518635

theorem islanders_liars (n : ℕ) (h : n = 450) : (∃ L : ℕ, (L = 150 ∨ L = 450)) :=
sorry

end islanders_liars_l518_518635


namespace perfect_square_divisible_by_12_l518_518230

theorem perfect_square_divisible_by_12 (k : ℤ) : 12 ∣ (k^2 * (k^2 - 1)) :=
by sorry

end perfect_square_divisible_by_12_l518_518230


namespace number_of_B_students_l518_518160

/- Define the assumptions of the problem -/
variable (x : ℝ)  -- the number of students who earn a B

/- Express the number of students getting each grade in terms of x -/
def number_of_A (x : ℝ) := 0.6 * x
def number_of_C (x : ℝ) := 1.3 * x
def number_of_D (x : ℝ) := 0.8 * x
def total_students (x : ℝ) := number_of_A x + x + number_of_C x + number_of_D x

/- Prove that x = 14 for the total number of students being 50 -/
theorem number_of_B_students : total_students x = 50 → x = 14 :=
by 
  sorry

end number_of_B_students_l518_518160


namespace original_price_of_wand_l518_518138

theorem original_price_of_wand (x : ℝ) (h : x / 8 = 12) : x = 96 :=
by
  sorry

end original_price_of_wand_l518_518138


namespace fraction_calculation_l518_518805

theorem fraction_calculation : 
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 :=
by sorry

end fraction_calculation_l518_518805


namespace total_pencils_l518_518314

theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) (h1 : pencils_per_child = 2) (h2 : num_children = 8) : (pencils_per_child * num_children) = 16 :=
by
  rw [h1, h2]
  norm_num

end total_pencils_l518_518314


namespace local_minimum_of_function_l518_518870

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

theorem local_minimum_of_function : 
  (∃ a, a = 1 ∧ ∀ ε > 0, f a ≤ f (a + ε) ∧ f a ≤ f (a - ε)) := sorry

end local_minimum_of_function_l518_518870


namespace area_of_abs_eq_sum_l518_518376

theorem area_of_abs_eq_sum (x y : ℝ) (h : |x| + |3 * y| = 12) : enclosed_area x y = 96 :=
sorry

end area_of_abs_eq_sum_l518_518376


namespace find_foci_coordinates_l518_518824

def hyperbola := ∀ (x y : ℝ), 
  x^2 / 3 - y^2 / 4 = 1

theorem find_foci_coordinates :
  let a := Real.sqrt 3
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  ∀ f : ℝ × ℝ, f ∈ ({(-c, 0), (c, 0)} : Set (ℝ × ℝ)) :=
by
  let a := Real.sqrt 3
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  intro f
  simp [a, b, c]
  split
  · intro h
    apply Or.inl
    injection h with h₁ h₂
    cases h₁
    cases h₂
    simp
  · intro h
    apply Or.inr
    injection h with h₁ h₂
    cases h₁
    cases h₂
    simp

end find_foci_coordinates_l518_518824


namespace marble_weight_l518_518974

noncomputable def base_length := 2
noncomputable def base_width := 2
noncomputable def height := 8
noncomputable def density := 2700
noncomputable def base_area := base_length * base_width
noncomputable def volume := base_area * height
noncomputable def weight := density * volume

theorem marble_weight :
  weight = 86400 :=
by
  -- Skipping the actual proof steps
  sorry

end marble_weight_l518_518974


namespace determine_negative_range_for_a_l518_518818

noncomputable def determine_range_a (a : ℝ) : Prop :=
  ∀ x : ℝ, sin x ^ 2 + a * cos x + a ^ 2 ≥ 1 + cos x

theorem determine_negative_range_for_a : 
  ∀ a : ℝ, (a < 0 ∧ determine_range_a a) → a ≤ -2 :=
by 
  intros a h,
  let H1 := h.1,
  let H2 := h.2,
  sorry

end determine_negative_range_for_a_l518_518818


namespace number_of_valid_arrangements_l518_518957

-- Definitions based on conditions
def digits : List ℕ := [1, 2, 2, 5, 0]

def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def valid_arrangement (arr : List ℕ) : Prop :=
  arr.length = 5 ∧
  is_multiple_of_5 (arr.reverse.head) ∧
  arr.reverse.head ∈ [0, 5] ∧
  arr.perm digits

-- Main statement to prove the number of valid arrangements
theorem number_of_valid_arrangements : (finset.univ.filter (λ arr, arr.length = 5 ∧ valid_arrangement (arr.to_list))).card = 21 :=
sorry

end number_of_valid_arrangements_l518_518957


namespace pen_cost_is_4_l518_518347

variable (penCost pencilCost : ℝ)

-- Conditions
def totalCost := penCost + pencilCost = 6
def costRelation := penCost = 2 * pencilCost

-- Theorem to be proved
theorem pen_cost_is_4 (h1 : totalCost) (h2 : costRelation) : penCost = 4 :=
by
  rw [totalCost, costRelation] at h1
  sorry

end pen_cost_is_4_l518_518347


namespace range_PA_dot_PD_l518_518167

structure EquilateralTriangle where
  A B C : ℝ^2 
  side_length : ℝ
  eq_sides : (A - B).norm = side_length ∧ (B - C).norm = side_length ∧ (C - A).norm = side_length

structure PointOnCircumcircle (A B C D O : ℝ^2) where
  circumcenter : O = ((A + B + C) / 3)
  circumradius : ∀ {P}, P ∈ {A, B, C} → (P - O).norm = 1
  on_circumcircle : (D - O).norm = 1
  AD_eq_sqrt2 : (A - D).norm = sqrt 2

theorem range_PA_dot_PD :
  ∀ (ABC : EquilateralTriangle) (P : ℝ^2),
  let ⟨A, B, C, side_len, eq_sides⟩ := ABC in
  let O := ((A + B + C) / 3) in
  ∀ (D : ℝ^2) (HD : PointOnCircumcircle A B C D O),
  P ∈ {A, B, C} →
  ∃ min_val max_val,
    min_val = -((2 + sqrt 3) / 8) ∧
    max_val = (3 + sqrt 3) / 2 ∧
    (min_val ≤ (P - A) ⬝ (P - D) ∧ (P - A) ⬝ (P - D) ≤ max_val) := 
sorry

end range_PA_dot_PD_l518_518167


namespace judge_guilty_cases_l518_518342

theorem judge_guilty_cases :
  let total_cases := 27
  let dismissed_cases := 3
  let remaining_cases := total_cases - dismissed_cases
  let innocent_cases := 3 * remaining_cases / 4
  let delayed_rulings := 2
  remaining_cases - innocent_cases - delayed_rulings = 4 :=
by
  let total_cases := 27
  let dismissed_cases := 3
  let remaining_cases := total_cases - dismissed_cases
  let innocent_cases := 3 * remaining_cases / 4
  let delayed_rulings := 2
  show remaining_cases - innocent_cases - delayed_rulings = 4
  sorry

end judge_guilty_cases_l518_518342


namespace intersection_condition_l518_518549

def A (m : ℤ) : set ℤ := {1, m^2}
def B : set ℤ := {2, 4}

theorem intersection_condition (m : ℤ) (h : m = 2) : A m ∩ B = {4} ↔ m = 2 :=
by {
  -- set variables
  unfold A B,
  sorry   -- proof of the theorem goes here
}

end intersection_condition_l518_518549


namespace monotonic_intervals_inequation_l518_518894

-- Part I: Monotonic intervals of the function f(x) when b = 0
theorem monotonic_intervals (a : ℝ) :
  (∀ x > 0, (a ≤ 0 → (∀ x > 0, deriv (λ x => 2 * a * x - 1 - 2 * log x) x < 0))
    ∧ (a > 0 → ((∀ x > 0, x < 1/a → deriv (λ x => 2 * a * x - 1 - 2 * log x) x < 0) 
    ∧ (∀ x > 0, x > 1/a → deriv (λ x => 2 * a * x - 1 - 2 * log x) x > 0)))) := sorry

-- Part II: inequation e^x ln(y + 1) > e^y ln(x + 1) when x > y > e - 1
theorem inequation (x y : ℝ) (h1: x > y) (h2: y > real.exp 1 - 1) : 
  real.exp x * real.log (y + 1) > real.exp y * real.log (x + 1) := sorry

end monotonic_intervals_inequation_l518_518894


namespace no_positive_product_circle_l518_518321

theorem no_positive_product_circle (n m : ℕ) (hn : n = 100) (hm : m = 101)
  (h_total : n + m = 201): 
  ¬ ∃ (l : list ℤ), (l.length = 201 ∧ (∀ i, i < 201 → l.nth i = some (if i < 100 then 1 else -1)) ∧ ∀ j, 0 ≤ j ∧ j < 201 → (l.nth j).get_or_else 1 * (l.nth ((j + 1) % 201)).get_or_else 1 * (l.nth ((j + 2) % 201)).get_or_else 1 > 0) :=
by 
  sorry

end no_positive_product_circle_l518_518321


namespace root_in_interval_l518_518410

open Real

def f (x : ℝ) := log x + x - 4

theorem root_in_interval : ∃ c ∈ Ioo 2 3, f c = 0 := by
  have f_continuous := Real.continuous_on_log.add continuous_id.sub continuous_const
  have f_strictly_increasing := by sorry -- This needs a formal proof in Lean but is stated as a condition
  have f_2 := log 2 + 2 - 4
  have f_3 := log 3 + 3 - 4
  have f_2_lt_0 : f 2 < 0 := by sorry -- Needs numerical approximation/log value precision
  have f_3_gt_0 : f 3 > 0 := by sorry -- Needs numerical approximation/log value precision
  apply Exists.intro 2 3
  split
  show 2 < c ∧ c < 3 from sorry
  show f c = 0 from sorry

end root_in_interval_l518_518410


namespace number_of_liars_l518_518644

constant islanders : Type
constant knight : islanders → Prop
constant liar : islanders → Prop
constant sits_at_table : islanders → Prop
constant right_of : islanders → islanders

axiom A1 : ∀ x : islanders, sits_at_table x → (knight x ∨ liar x)
axiom A2 : (∃ n : ℕ, n = 450 ∧ (λ x, sits_at_table x))
axiom A3 : ∀ x : islanders, sits_at_table x →
  (liar (right_of x) ∧ ¬ liar (right_of (right_of x))) ∨ 
  (¬ liar (right_of x) ∧ liar (right_of (right_of x)))

theorem number_of_liars : 
  (∃ n, ∃ m, (n = 450) ∨ (m = 150)) :=
sorry

end number_of_liars_l518_518644


namespace compare_values_l518_518846

noncomputable def a : ℝ := (1/2)^(1/2)
noncomputable def b : ℝ := Real.log 2015 / Real.log 2014
noncomputable def c : ℝ := Real.log 2 / Real.log 4

theorem compare_values : b > a ∧ a > c :=
by
  sorry

end compare_values_l518_518846


namespace travel_time_without_paddles_l518_518078

variables (A B : Type) (v v_r S : ℝ)
noncomputable def time_to_travel (distance velocity : ℝ) := distance / velocity

-- Condition: The travel time from A to B is 3 times the travel time from B to A
axiom travel_condition : (time_to_travel S (v + v_r)) = 3 * (time_to_travel S (v - v_r))

-- Condition: We are considering travel from B to A by canoe without paddles
noncomputable def time_without_paddles := time_to_travel S v_r

-- Proving that without paddles it takes 3 times longer than usual (using canoes with paddles)
theorem travel_time_without_paddles :
  time_without_paddles S v_r = 3 * (time_to_travel S (v - v_r)) :=
sorry

end travel_time_without_paddles_l518_518078


namespace num_ways_to_put_5_balls_into_4_boxes_l518_518530

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l518_518530


namespace tan_diff_of_angle_l518_518120

noncomputable theory

open Complex

theorem tan_diff_of_angle 
  (θ : ℝ) 
  (h1 : cos θ = 3 / 5)
  (h2 : sin θ ≠ 4 / 5)
  (h3 : sin θ ^ 2 + cos θ ^ 2 = 1) : 
  tan (θ - π / 4) = 7 :=
sorry

end tan_diff_of_angle_l518_518120


namespace sin_double_alpha_zero_l518_518622

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem sin_double_alpha_zero (α : ℝ) (h : f α = 1) : Real.sin (2 * α) = 0 :=
by 
  -- Proof would go here, but we're using sorry
  sorry

end sin_double_alpha_zero_l518_518622


namespace double_sum_series_l518_518403

theorem double_sum_series :
  ∑' (i : ℕ) in (finset.Icc 1 (finset.univ : finset ℕ)), ∑' (j : ℕ) in (finset.Icc 1 (finset.univ : finset ℕ)), 
  (1 : ℝ) / (i^2 * j + 2 * i * j + i * j^2) = 7 / 4 :=
by
  sorry

end double_sum_series_l518_518403


namespace tan_double_angle_identity_l518_518539

theorem tan_double_angle_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) : 
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := 
  sorry

end tan_double_angle_identity_l518_518539


namespace sin_double_angle_second_quadrant_l518_518101

theorem sin_double_angle_second_quadrant (α : ℝ) (h1 : Real.cos α = -3/5) (h2 : α ∈ Set.Ioo (π / 2) π) :
    Real.sin (2 * α) = -24 / 25 := by
  sorry

end sin_double_angle_second_quadrant_l518_518101


namespace log_base_2_iff_l518_518324

open Function

theorem log_base_2_iff (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : a > b ↔ log 2 a > log 2 b :=
by

-- The proof would be filled here

sorry

end log_base_2_iff_l518_518324


namespace sum_abs_f_inequality_l518_518623

theorem sum_abs_f_inequality
  (f : ℚ → ℚ)
  (h : ∀ (m n : ℚ), |f (m + n) - f m| ≤ m / n)
  : ∀ k : ℕ, 0 < k → ∑ i in finset.range k, |f (2^k) - f (2^(i+1))| ≤ k * (k - 1) / 2 := 
by
  sorry

end sum_abs_f_inequality_l518_518623


namespace correct_answers_l518_518112

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
noncomputable def g (x : ℝ) : ℝ := f' x

-- Conditions
axiom f_domain : ∀ x, f x ∈ ℝ
axiom f'_domain : ∀ x, f' x ∈ ℝ
axiom g_def : ∀ x, g x = f' x
axiom f_even : ∀ x, f (3 / 2 - 2 * x) = f (3 / 2 + 2 * x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- Proof Problem
theorem correct_answers : f (-1) = f 4 ∧ g (-1 / 2) = 0 :=
  by
    sorry

end correct_answers_l518_518112


namespace train_is_late_l518_518000

theorem train_is_late (S : ℝ) (T : ℝ) (T' : ℝ) (h1 : T = 2) (h2 : T' = T * 5 / 4) :
  (T' - T) * 60 = 30 :=
by
  sorry

end train_is_late_l518_518000


namespace time_needed_n_l518_518772

variable (n : Nat)
variable (d : Nat := n - 1)
variable (s : ℚ := 2 / 3 * (d))
variable (time_third_mile : ℚ := 3)
noncomputable def time_needed (n : Nat) : ℚ := (3 * (n - 1)) / 2

theorem time_needed_n: 
  (∀ (n : Nat), n > 2 → time_needed n = (3 * (n - 1)) / 2) :=
by
  intros n hn
  sorry

end time_needed_n_l518_518772


namespace valid_assignments_l518_518914

-- Given the digits from 1 to 7
def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7]

-- Hexagon with center G and vertices A, B, C, D, E, F
variables (A B C D E F G : ℕ)

-- Conditions
def is_valid_assignment (A B C D E F G : ℕ) : Prop :=
  G ∈ {1, 4, 7} ∧
  (Set.toFinset [A, B, C, D, E, F, G] = Set.toFinset digits) ∧
  ((A + G + C) = (B + G + D)) ∧ ((B + G + D) = (C + G + E))

-- The total ways to assign the digits given the conditions
theorem valid_assignments : (∃ (A B C D E F G : ℕ),
  is_valid_assignment A B C D E F G) →
  (3 * 3! * 2^3 = 144) :=
by sorry

end valid_assignments_l518_518914


namespace triangle_proof_l518_518627

noncomputable def triangle_math_proof (A B C : ℝ) (AA1 BB1 CC1 : ℝ) : Prop :=
  AA1 = 2 * Real.sin (B + A / 2) ∧
  BB1 = 2 * Real.sin (C + B / 2) ∧
  CC1 = 2 * Real.sin (A + C / 2) ∧
  (Real.sin A + Real.sin B + Real.sin C) ≠ 0 ∧
  ∀ x, x = (AA1 * Real.cos (A / 2) + BB1 * Real.cos (B / 2) + CC1 * Real.cos (C / 2)) / (Real.sin A + Real.sin B + Real.sin C) → x = 2

theorem triangle_proof (A B C AA1 BB1 CC1 : ℝ) (h : triangle_math_proof A B C AA1 BB1 CC1) :
  (AA1 * Real.cos (A / 2) + BB1 * Real.cos (B / 2) + CC1 * Real.cos (C / 2)) /
  (Real.sin A + Real.sin B + Real.sin C) = 2 := by
  sorry

end triangle_proof_l518_518627


namespace math_problem_l518_518218

noncomputable def proof_problem (a b c : ℝ) : Prop :=
  (0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) →
  (1 / (a * (a + 1)) + 1 / (b * (b + 1)) + 1 / (c * (c + 1)) ≥  3 / 2)

theorem math_problem (a b c : ℝ) :
  proof_problem a b c :=
by
  sorry

end math_problem_l518_518218


namespace not_exists_k_good_l518_518419

theorem not_exists_k_good (k : ℕ) (hk : k > 0) : 
  ¬∀ n : ℕ, ∃ a : Fin k → ℕ, n = (Finset.univ.sum (λ (i : Fin k), a i ^ (2^(i+1)))) :=
by sorry

end not_exists_k_good_l518_518419


namespace max_annual_profit_l518_518882

noncomputable def annual_sales_volume (x : ℝ) : ℝ := - (1 / 3) * x^2 + 2 * x + 21

noncomputable def annual_sales_profit (x : ℝ) : ℝ := (- (1 / 3) * x^3 + 4 * x^2 + 9 * x - 126)

theorem max_annual_profit :
  ∀ x : ℝ, (x > 6) →
  (annual_sales_volume x) = - (1 / 3) * x^2 + 2 * x + 21 →
  (annual_sales_volume 10 = 23 / 3) →
  (21 - annual_sales_volume x = (1 / 3) * (x^2 - 6 * x)) →
    (annual_sales_profit x = - (1 / 3) * x^3 + 4 * x^2 + 9 * x - 126) ∧
    ∃ x_max : ℝ, 
      (annual_sales_profit x_max = 36) ∧
      x_max = 9 :=
by
  sorry

end max_annual_profit_l518_518882


namespace min_value_least_l518_518133

noncomputable def min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : ℝ :=
  Inf {v | v = 1/x + 4/y}

theorem min_value_least (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) :
  min_value x y hx hy h = 9 / 4 := sorry

end min_value_least_l518_518133


namespace area_of_triangle_with_sides_l518_518678

-- Given conditions from the problem
variable {a b c : ℝ}

axiom roots_of_polynomial : 
  ∀ {a b c : ℝ}, 
  (Poly : x^3 - 3*x^2 + 4*x - (1/2) = 0) → 
  (a + b + c = 3) ∧ (a*b*c = - (1/2)) ∧ 
  ((x - a) * (x - b) * (x - c) = Poly)

-- The Lean definition of the proof problem
theorem area_of_triangle_with_sides {a b c : ℝ} 
  (h_root_conditions : ∀ {a b c : ℝ}, x^3 - 3*x^2 + 4*x - (1/2) = 0 ∧ (roots_of_polynomial)) :
  area_of_triangle_with_sides a b c = (√3 / 2) := 
sorry

end area_of_triangle_with_sides_l518_518678


namespace savings_time_l518_518227

theorem savings_time (total_amount : ℕ) (monthly_saving_per_person : ℕ) (combined_saving : ℕ)
  (months_in_year : ℕ) (total_months: ℕ) (years : ℕ)
  (h_total_amount : total_amount = 108000)
  (h_monthly_saving_per_person : monthly_saving_per_person = 1500)
  (h_combined_saving : combined_saving = 2 * monthly_saving_per_person)
  (h_total_months : total_months = total_amount / combined_saving)
  (h_months_in_year : months_in_year = 12)
  (h_years : years = total_months / months_in_year) :
  years = 3 :=
by
  rewrite [h_total_amount, h_monthly_saving_per_person, h_combined_saving, h_total_months, h_months_in_year, h_years]
  sorry

end savings_time_l518_518227


namespace min_BD_plus_CD_l518_518579

/-- Given a right triangle ABC with right angle at A, sides AB and AC being equal to 4, 
and a point D such that AD = sqrt(2), the minimum value of BD + CD is 2 * sqrt(10). -/
theorem min_BD_plus_CD 
  (A B C D : EuclideanSpace ℝ (Fin 2))
  (hA : A = ⟨0, 0⟩)
  (hB : B = ⟨4, 0⟩)
  (hC : C = ⟨0, 4⟩)
  (hAB : dist A B = 4)
  (hAC : dist A C = 4)
  (hBAC : ∠ A B C = real.pi / 2)
  (hAD : dist A D = real.sqrt 2) :
  ∃ (x y : ℝ), min (dist B D + dist C D) = 2 * real.sqrt 10 :=
sorry

end min_BD_plus_CD_l518_518579


namespace verify_trajectory_l518_518938

def circle1_center (a : ℝ) : ℝ × ℝ :=
  (a / 2, -1)

def circle2_center : (ℝ × ℝ) :=
  (0, 0)

def is_symmetric (p1 p2 : ℝ × ℝ) (line : ℝ → ℝ) : Prop :=
  ∃ p3 : ℝ × ℝ, p1 = (fst p3, line (fst p3)) ∧ p2 = (fst p3, line (fst p3))

def trajectory_eq (x y : ℝ) : Prop :=
  y^2 + 4*x - 4*y + 8 = 0

noncomputable def center_trajectory (a : ℝ) : Prop :=
  trajectory_eq (2*a + 1/2) (-1 - a)

theorem verify_trajectory (a : ℝ) :
  circle1_center a = (a / 2, -1) →
  is_symmetric (a / 2, -1) (0, 0) (λ x, x - 1) →
  a = 2 →
  trajectory_eq (fst (2*a +1/2, -1 - a)) (snd (2*a +1/2, -1 - a)) :=
by
  intros h1 h2 h3
  rw h1
  simp only [circle1_center, trajectory_eq, is_symmetric] at *
  sorry

end verify_trajectory_l518_518938


namespace other_solution_of_quadratic_l518_518874

theorem other_solution_of_quadratic (x : ℚ) (h₁ : 81 * 2/9 * 2/9 + 220 = 196 * 2/9 - 15) (h₂ : 81*x^2 - 196*x + 235 = 0) : x = 2/9 ∨ x = 5/9 :=
by
  sorry

end other_solution_of_quadratic_l518_518874


namespace gimbap_total_cost_l518_518795

theorem gimbap_total_cost :
  let basic_gimbap_cost := 2000
  let tuna_gimbap_cost := 3500
  let red_pepper_gimbap_cost := 3000
  let beef_gimbap_cost := 4000
  let nude_gimbap_cost := 3500
  let cost_of_two gimbaps := (tuna_gimbap_cost * 2) + (beef_gimbap_cost * 2) + (nude_gimbap_cost * 2)
  cost_of_two gimbaps = 22000 := 
by 
  sorry

end gimbap_total_cost_l518_518795


namespace max_profit_when_18_pieces_l518_518279

noncomputable def C (x : ℝ) := 20 + 2 * x + 0.5 * x^2

noncomputable def profit (x : ℝ) := 20 * x - C x

theorem max_profit_when_18_pieces :
  let x := 18 in
  profit x = 142 :=
by
  sorry

end max_profit_when_18_pieces_l518_518279


namespace sequence_a_formula_l518_518965

noncomputable def sequence_a (n : ℕ) : ℕ :=
if n = 1 then 1 else 4 * sequence_a (n - 1) + 2

theorem sequence_a_formula (n : ℕ) (hn: n ≥ 1) : sequence_a n = (3 * n - 1) * 2 ^ (n - 2) :=
by sorry

end sequence_a_formula_l518_518965


namespace product_negative_probability_l518_518135

def SetA : Set Int := {0, 1, -3, 6, -8, -10}
def SetB : Set Int := {-1, 2, -4, 7, 6, -9}

noncomputable def probability_negative_product : ℚ :=
  let positiveA := {1, 6}
  let negativeA := {-3, -8, -10}
  let positiveB := {2, 7, 6}
  let negativeB := {-1, -4, -9}
  let ways_to_select_negative_product :=
    (positiveA.card * negativeB.card) + (negativeA.card * positiveB.card)
  let total_pairs := SetA.card * SetB.card
  (ways_to_select_negative_product : ℚ) / total_pairs

theorem product_negative_probability :
  probability_negative_product = 5 / 12 := by
  sorry

end product_negative_probability_l518_518135


namespace monotonic_increasing_interval_l518_518690

noncomputable def log_base_1_div_3 (t : ℝ) := Real.log t / Real.log (1/3)

def quadratic (x : ℝ) := 4 + 3 * x - x^2

theorem monotonic_increasing_interval :
  ∃ (a b : ℝ), (∀ x, a < x ∧ x < b → (log_base_1_div_3 (quadratic x)) < (log_base_1_div_3 (quadratic (x + ε))) ∧
               ((-1 : ℝ) < x ∧ x < 4) ∧ (quadratic x > 0)) ↔ (a, b) = (3 / 2, 4) :=
by
  sorry

end monotonic_increasing_interval_l518_518690


namespace molecular_weight_one_mole_l518_518722

theorem molecular_weight_one_mole {compound : Type} (moles : ℕ) (total_weight : ℝ) 
  (h_moles : moles = 5) (h_total_weight : total_weight = 490) :
  total_weight / moles = 98 := 
by {
    rw [h_moles, h_total_weight],
    norm_num,
    sorry
  }

end molecular_weight_one_mole_l518_518722


namespace point_on_x_axis_l518_518548

theorem point_on_x_axis (m : ℝ) (h : (m, m - 1).snd = 0) : m = 1 :=
by
  sorry

end point_on_x_axis_l518_518548


namespace lim_S_eq_one_l518_518901

section math_problem

open Real

noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := a n / 2 + 2^n * (a n)^2

noncomputable def S (n : ℕ) : ℝ :=
∑ i in finset.range (n + 1), 1 / (2^i * a i + 1)

theorem lim_S_eq_one : tendsto S at_top (nhds 1) :=
sorry

end math_problem

end lim_S_eq_one_l518_518901


namespace rectangle_side_length_l518_518566

theorem rectangle_side_length (ABCD : Type) [rectangle ABCD] (P : ABCD → ABCD → ABCD) (B C D : ABCD) 
  (BP : ℝ) (CP : ℝ) (tan_APD : ℝ) :
  (BP = 12) → (CP = 4) → (tan_APD = 2) → (∃ (AB : ℝ), AB = 12) :=
by
  assume hBP : BP = 12,
  assume hCP : CP = 4,
  assume htan_APD : tan_APD = 2,
  use 12,
  sorry

end rectangle_side_length_l518_518566


namespace blue_marbles_difference_l518_518294

variable (a b : ℕ)
variables (number_of_green_marbles_in_jar1 number_of_green_marbles_in_jar2 : ℕ)
variables (total_green_marbles : ℕ) (total_marbles : ℕ)
variable (number_of_blue_marbles_in_jar1 number_of_blue_marbles_in_jar2 : ℕ)

noncomputable def number_of_green_marbles_in_jar1 := 3 * a
noncomputable def number_of_green_marbles_in_jar2 := b
noncomputable def total_green_marbles := 110

noncomputable def number_of_blue_marbles_in_jar1 := 7 * a
noncomputable def number_of_blue_marbles_in_jar2 := 8 * b

noncomputable def total_marbles := 10 * a

theorem blue_marbles_difference :
  10 * a = 9 * b → 
  number_of_green_marbles_in_jar1 + number_of_green_marbles_in_jar2 = total_green_marbles →
  number_of_blue_marbles_in_jar2 - number_of_blue_marbles_in_jar1 = 51 :=
by
  intro h1 h2
  sorry

end blue_marbles_difference_l518_518294


namespace area_of_triangle_AOB_l518_518899

-- Define the parametric equations of curve C
def parametric_curve (α : ℝ) : ℝ × ℝ :=
  (1 + Real.sqrt 5 * Real.cos α, 2 + Real.sqrt 5 * Real.sin α)

-- Define the polar coordinates transformation
def polar_coordinates (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the lines l1 and l2 in polar coordinates
def line_l1 (θ : ℝ) := θ = π / 6
def line_l2 (θ : ℝ) := θ = π / 3

-- Define the points A and B in polar coordinates, where l1 and l2 intersect curve C
def point_A (ρ : ℝ) := ρ = 2 * Real.cos (π / 6) + 4 * Real.sin (π / 6)
def point_B (ρ : ℝ) := ρ = 2 * Real.cos (π / 3) + 4 * Real.sin (π / 3)

-- Statement to prove the area of triangle AOB
theorem area_of_triangle_AOB :
  let |OA| := Real.sqrt 3 + 2 in
  let |OB| := 1 + 2 * Real.sqrt 3 in
  let θOAB := π / 3 - π / 6 in
  0.5 * |OA| * |OB| * Real.sin θOAB = (8 + 5 * Real.sqrt 3) / 4 :=
sorry

end area_of_triangle_AOB_l518_518899


namespace sum_of_decimals_l518_518001

-- Definitions from conditions
def decimal_1 : ℝ := 0.45
def decimal_2 : ℝ := 0.003
def fraction_1 : ℝ := 1 / 4
def decimal_from_fraction_1 : ℝ := 0.25

-- Main theorem statement
theorem sum_of_decimals : (decimal_1 + decimal_2 + decimal_from_fraction_1) = 0.703 := by
  have fraction_to_decimal : fraction_1 = decimal_from_fraction_1 := by
    calc 
      fraction_1 
          = 1 / 4       : rfl
      ... = 0.25        : by norm_num
    
  rw fraction_to_decimal,
  calc 
    (decimal_1 + decimal_2 + decimal_from_fraction_1)
        = (0.45 + 0.003 + 0.25)     : by refl
    ... = 0.703                     : by norm_num

end sum_of_decimals_l518_518001


namespace balls_in_boxes_l518_518534

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l518_518534


namespace find_f_prime_2_l518_518474

theorem find_f_prime_2 (a : ℝ) (f' : ℝ → ℝ) 
    (h1 : f' 1 = -5)
    (h2 : ∀ x, f' x = 3 * a * x^2 + 2 * f' 2 * x) : f' 2 = -4 := by
    sorry

end find_f_prime_2_l518_518474


namespace part1_solution_set_a_eq_2_part2_range_of_a_l518_518893

set_option pp.explicit true 

def f (x a : ℝ) : ℝ := |x - a| - |2 * x - 1|

theorem part1_solution_set_a_eq_2 :
  ∀ x : ℝ, -4 ≤ x ∧ x ≤ 2 ↔ f x 2 + 3 ≥ 0 :=
by sorry

theorem part2_range_of_a :
  ∀ a : ℝ, ∀ x : ℝ, x ∈ set.Icc 1 3 → f x a ≤ 3 ↔ -3 ≤ a ∧ a ≤ 5 :=
by sorry

end part1_solution_set_a_eq_2_part2_range_of_a_l518_518893


namespace length_of_tangent_segment_l518_518811

noncomputable def circle_center : ℝ × ℝ := (3, 6)
noncomputable def radius : ℝ := real.sqrt 10
noncomputable def origin : ℝ × ℝ := (0, 0)

def point_on_circle (p : ℝ × ℝ) : Prop :=
  let (h, k) := circle_center in
  let R := radius in
  (p.1 - h)^2 + (p.2 - k)^2 = R^2

def points : list (ℝ × ℝ) := [(2, 3), (4, 6), (3, 9)]
def all_points_on_circle : Prop := ∀ p ∈ points, point_on_circle p

theorem length_of_tangent_segment :
  all_points_on_circle →
  let OT := real.sqrt (10 + 3 * real.sqrt 5) in
  ¬ ∃ T, (T = OT /\
  let (h, k) := circle_center in
  let R := radius in
  (0 - h)^2 + (0 - k)^2 = OT^2 + R^2) :=
by 
  intros h k R p all_points
  let OT := real.sqrt (10 + 3 * real.sqrt 5)
  sorry

end length_of_tangent_segment_l518_518811


namespace ashley_friends_ages_correct_sum_l518_518670

noncomputable def ashley_friends_ages_sum : Prop :=
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
                   (1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9) ∧
                   (a * b = 36) ∧ (c * d = 30) ∧ (a + b + c + d = 24)

theorem ashley_friends_ages_correct_sum : ashley_friends_ages_sum := sorry

end ashley_friends_ages_correct_sum_l518_518670


namespace sum_f_values_l518_518426

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2 / x) + 1

theorem sum_f_values : 
  f (-7) + f (-5) + f (-3) + f (-1) + f (3) + f (5) + f (7) + f (9) = 8 := 
by
  sorry

end sum_f_values_l518_518426


namespace maximize_probability_sum_is_15_l518_518298

def initial_list : List ℤ := [-1, 0, 1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16]

def valid_pairs (lst : List ℤ) : List (ℤ × ℤ) :=
  (lst.product lst).filter (λ ⟨x, y⟩ => x < y ∧ x + y = 15)

def remove_one_element (lst : List ℤ) (x : ℤ) : List ℤ :=
  lst.erase x

theorem maximize_probability_sum_is_15 :
  (List.length (valid_pairs (remove_one_element initial_list 8))
   = List.maximum (List.map (λ x => List.length (valid_pairs (remove_one_element initial_list x))) initial_list)) :=
sorry

end maximize_probability_sum_is_15_l518_518298


namespace finite_coin_set_exists_l518_518332

theorem finite_coin_set_exists (n : ℕ → ℕ) (h : ∀ i j, i < j → n i < n j) :
  ∃ N : ℕ, ∀ M : ℕ, (∃ k : ℕ, ∃ a : ℕ → ℕ, (∑ i in finset.range k, a i * n i) = M) →
    (∃ k' : ℕ, ∃ b : ℕ → ℕ, (∑ i in finset.range k', b i * (n i) (finset.range (N + 1))) = M) :=
begin
  sorry
end

end finite_coin_set_exists_l518_518332


namespace find_x_plus_y_l518_518837

def vectors_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2, k * b.3)

theorem find_x_plus_y (x y : ℝ) :
  vectors_parallel (2 - x, -1, y) (-1, x, -1) → x + y = 2 :=
by
  sorry

end find_x_plus_y_l518_518837


namespace building_height_l518_518583

theorem building_height (h : ℕ) 
 (P1 : True)
 (P2 : ∀ (n : ℕ), n < 10 → height n = h)
 (P3 : ∀ (n : ℕ), n ≥ 10 ∧ n < 20 → height n = h + 3)
 (P4 : (Σ n, height n) = 270) : 
 h = 12 := 
by
  sorry

end building_height_l518_518583


namespace distinct_values_2_l518_518927

theorem distinct_values_2
  (a b : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0):
  ∃ (s : Finset ℝ), s.card = 2 ∧
    ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → 
    (d := (a / |a|) + (b / |b|) + (a * b / |a * b|)),
    d ∈ s :=
by sorry

end distinct_values_2_l518_518927


namespace mina_numbers_l518_518209

theorem mina_numbers (a b : ℤ) (h1 : 3 * a + 4 * b = 140) (h2 : a = 20 ∨ b = 20) : a = 20 ∧ b = 20 :=
by
  sorry

end mina_numbers_l518_518209


namespace domain_of_func_l518_518051

def func (x : ℝ) : ℝ := (x + 1) / (x^2 + 6*x + 8)

theorem domain_of_func :
  {x : ℝ | (x ≠ -4) ∧ (x ≠ -2)} = set.univ \ ({-4} ∪ {-2}) :=
by
  sorry

end domain_of_func_l518_518051


namespace molecular_weight_of_1_mole_l518_518724

theorem molecular_weight_of_1_mole (W_5 : ℝ) (W_1 : ℝ) (h : 5 * W_1 = W_5) (hW5 : W_5 = 490) : W_1 = 490 :=
by
  sorry

end molecular_weight_of_1_mole_l518_518724


namespace vector_magnitude_l518_518881

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_magnitude (a b : V) (h_angle : real.angle a b = real.pi / 3)
  (h_a_mag : ∥a∥ = 2) (h_combined_mag : ∥a + (2 : ℝ) • b∥ = 2 * real.sqrt 7) : 
  ∥b∥ = 2 :=
sorry

end vector_magnitude_l518_518881


namespace distance_focus_parabola_to_line_l518_518252

theorem distance_focus_parabola_to_line :
  let focus : ℝ × ℝ := (1, 0)
  let distance (p : ℝ × ℝ) (A B C : ℝ) : ℝ := |A * p.1 + B * p.2 + C| / Real.sqrt (A^2 + B^2)
  distance focus 1 (-Real.sqrt 3) 0 = 1 / 2 :=
by
  sorry

end distance_focus_parabola_to_line_l518_518252


namespace general_formula_smallest_integer_m_l518_518118

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {b : ℕ → ℝ} {T : ℕ → ℝ}

/-- Conditions -/
def conditions : Prop :=
  (S 1 = (a 1 : ℝ)) ∧ -- sum of first term is just the first term
  (∀ n, n ≥ 2 → a n > 0) ∧ -- a_n > 0 for n ≥ 2
  (∀ n, a (n + 1) = sqrt (S (n + 1)) + sqrt (S n)) -- recurrence relation a_(n+1)

/-- Part 1: General formula for sequence a_n -/
theorem general_formula (h : conditions) : 
  ∀ n, n ≥ 2 → a n = 2 * n - 3 :=
sorry

/-- Part 2: Smallest integer value of m for sequence b_n -/
theorem smallest_integer_m (h : conditions) :
  let b := λ n, a n / 2^n in
  ∃ m : ℕ, (∀ n, T n < m) ∧ m = 2 :=
sorry

end general_formula_smallest_integer_m_l518_518118


namespace angle_at_6_25_l518_518021

noncomputable def angle_between_clock_hands (h : ℕ) (m : ℕ) : ℝ :=
  let hour_angle := (h % 12) * 30 + m / 2
  let minute_angle := m * 6
  let angle := abs (hour_angle - minute_angle)
  if angle > 180 then 360 - angle else angle

theorem angle_at_6_25 : angle_between_clock_hands 6 25 = 42.5 :=
by
  sorry

end angle_at_6_25_l518_518021


namespace owner_bike_speed_l518_518358

-- Define the initial conditions
def thief_speed : ℝ := 45 -- kmph
def discovery_time : ℝ := 0.5 -- hours
def overtake_time : ℝ := 4 -- hours

-- The proof statement
theorem owner_bike_speed : 
  ∃ speed_owner: ℝ, 
    let head_start_distance := thief_speed * discovery_time in
    let pursue_time := overtake_time - discovery_time in
    let distance_thief := thief_speed * pursue_time in
    let total_distance := head_start_distance + distance_thief in
    total_distance / pursue_time = 51.43 := 
sorry

end owner_bike_speed_l518_518358


namespace find_trapezoid_height_l518_518672

-- Define the conditions of the problem
variables {a b h : ℝ} (α : ℝ)

-- The area of an isosceles trapezoid is 32
def trapezoid_area (a b h : ℝ) : Prop :=
  (a + b) * h / 2 = 32

-- The cotangent of the angle between a diagonal and the base is 2
def angle_cotangent (α : ℝ) : Prop :=
  Real.cot α = 2

-- The height of the trapezoid needs to be found
def trapezoid_height (h : ℝ) : Prop :=
  h = 4

-- The theorem to prove
theorem find_trapezoid_height (A: ℝ) (α : ℝ) (h : ℝ) (a b : ℝ) :
  trapezoid_area a b h → angle_cotangent α → trapezoid_height h :=
sorry

end find_trapezoid_height_l518_518672


namespace time_to_traverse_nth_mile_l518_518771

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 3) : 
  let t : ℝ := Real.cbrt (27 * (n - 1)^2 / 4) in
  t = Real.cbrt (27 * (n - 1)^2 / 4) := by sorry

end time_to_traverse_nth_mile_l518_518771


namespace infinite_solutions_b_value_l518_518839

-- Given condition for the equation to hold
def equation_condition (x b : ℤ) : Prop :=
  4 * (3 * x - b) = 3 * (4 * x + 16)

-- The statement we need to prove: b = -12
theorem infinite_solutions_b_value :
  (∀ x : ℤ, equation_condition x b) → b = -12 :=
sorry

end infinite_solutions_b_value_l518_518839


namespace travel_time_tripled_l518_518065

variable {S v v_r : ℝ}

-- Conditions of the problem
def condition1 (t1 t2 : ℝ) : Prop :=
  t1 = 3 * t2

def condition2 (t1 t2 : ℝ) : Prop :=
  t1 = S / (v + v_r) ∧ t2 = S / (v - v_r)

def stationary_solution : Prop :=
  v = 2 * v_r

-- Conclusion: Time taken to travel from B to A without paddles is 3 times longer than usual
theorem travel_time_tripled (t_no_paddle t2 : ℝ) (h1 : condition1 t_no_paddle t2) (h2 : condition2 t_no_paddle t2) (h3 : stationary_solution) :
  t_no_paddle = 3 * t2 :=
sorry

end travel_time_tripled_l518_518065


namespace impossible_to_equalize_numbers_l518_518757

theorem impossible_to_equalize_numbers (nums : Fin 6 → ℤ) :
  ¬ (∃ n : ℤ, ∀ i : Fin 6, nums i = n) :=
sorry

end impossible_to_equalize_numbers_l518_518757


namespace no_real_roots_of_quadratic_l518_518276

theorem no_real_roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = 1 ∧ c = 1) :
  (b^2 - 4 * a * c < 0) → ¬∃ x : ℝ, a * x^2 + b * x + c = 0 := by
  sorry

end no_real_roots_of_quadratic_l518_518276


namespace max_score_lowest_team_l518_518158

theorem max_score_lowest_team {n : ℕ} (h : n ≥ 4) :
    ∃ a : ℕ, (∀ k, k < n → a + k) ∧ ((∑ i in finset.range n, a + i) = n * a + (n * (n - 1)) / 2) 
    ∧ (∑ i in finset.range n, a + i) ≤ (3 * n * (n - 1)) / 2 ∧ a = n - 1 :=
by sorry

end max_score_lowest_team_l518_518158


namespace perpendicular_line_eqn_l518_518582

noncomputable def line_intersection_with_x_axis (a b c : ℝ) (h : b ≠ 0) : ℝ := (-c / b)

theorem perpendicular_line_eqn (x y : ℝ) (h : 2 * x - y - 4 = 0) :
  x + 2 * y - 2 = 0 :=
by {
  have slope_l : ℝ := 2,
  have slope_perpendicular : ℝ := -1 / 2,
  have M_x_coord : ℝ := line_intersection_with_x_axis 2 (-1) (-4) (ne_of_lt (by norm_num)),
  have M : ℝ × ℝ := (M_x_coord, 0),
  sorry
}

end perpendicular_line_eqn_l518_518582


namespace exists_gcd_property_l518_518659

theorem exists_gcd_property (C : ℝ) (H N : ℕ) (hH : H ≥ 3) (hN : N ≥ real.exp (C * H)) (T : finset ℕ) (hT : T.card = ceiling ((C * H * N) / real.log N)) :
  ∃ (A : finset ℕ), A.card = H ∧ (∀ x y ∈ A, nat.gcd x y = nat.gcd (A.to_list.head) x) :=
sorry

end exists_gcd_property_l518_518659


namespace arithmetic_mean_after_removal_l518_518248

theorem arithmetic_mean_after_removal
  (mean_original : ℝ)
  (n_original : ℕ)
  (sum_original : ℝ)
  (a b : ℝ)
  (n_remaining : ℕ)
  (sum_remaining : ℝ)
  (mean_remaining : ℝ)
  (H1 : mean_original = 50)
  (H2 : n_original = 100)
  (H3 : sum_original = mean_original * n_original)
  (H4 : a = 60)
  (H5 : b = 70)
  (H6 : n_remaining = n_original - 2)
  (H7 : sum_remaining = sum_original - (a + b))
  (H8 : mean_remaining = sum_remaining / n_remaining) :
  mean_remaining ≈ 49.7 :=
by
  sorry

end arithmetic_mean_after_removal_l518_518248


namespace pyramid_height_proof_l518_518775

noncomputable def height_of_pyramid (side_length : ℝ) (height_to_vertex : ℝ) : ℝ :=
let half_diagonal := side_length * Real.sqrt 2 / 2 in
Real.sqrt (height_to_vertex^2 - half_diagonal^2)

theorem pyramid_height_proof 
  (side_length : ℝ) (height_to_vertex : ℝ)
  (side_length_eq : side_length = 10)
  (height_to_vertex_eq : height_to_vertex = 12) :
  height_of_pyramid side_length height_to_vertex = Real.sqrt 94 :=
by
  rw [side_length_eq, height_to_vertex_eq]
  unfold height_of_pyramid
  simp
  linarith
  sorry

end pyramid_height_proof_l518_518775


namespace first_equation_value_l518_518912

theorem first_equation_value (x y : ℝ) (V : ℝ) 
  (h1 : x + |x| + y = V) 
  (h2 : x + |y| - y = 6) 
  (h3 : x + y = 12) : 
  V = 18 := 
by
  sorry

end first_equation_value_l518_518912


namespace height_of_room_is_12_l518_518676

noncomputable def roomHeight (totalCost Rs : ℕ) (length width doorHeight doorWidth windowHeight windowWidth costPerSqFt numWindows : ℕ) : ℕ :=
let perimeter := 2 * (length + width)
let totalWallArea := perimeter * height
let doorArea := doorHeight * doorWidth
let windowArea := windowHeight * windowWidth
let totalWindowArea := numWindows * windowArea
let netArea := totalWallArea - (doorArea + totalWindowArea)
let totalCostEq := netArea * costPerSqFt
totalCostEq

theorem height_of_room_is_12 : roomHeight 2718 3 25 15 6 3 4 3 3 1 12 :=
by
  sorry

end height_of_room_is_12_l518_518676


namespace incorrect_statement_D_l518_518311

/-
Define the conditions for the problem:
- A prism intersected by a plane.
- The intersection of a sphere and a plane when the plane is less than the radius.
- The intersection of a plane parallel to the base of a circular cone.
- The geometric solid formed by rotating a right triangle around one of its sides.
- The incorrectness of statement D.
-/

noncomputable def intersect_prism_with_plane (prism : Type) (plane : Type) : Prop := sorry

noncomputable def sphere_intersection (sphere_radius : ℝ) (distance_to_plane : ℝ) : Type := sorry

noncomputable def cone_intersection (cone : Type) (plane : Type) : Type := sorry

noncomputable def rotation_result (triangle : Type) (side : Type) : Type := sorry

theorem incorrect_statement_D :
  ¬(rotation_result RightTriangle Side = Cone) :=
sorry

end incorrect_statement_D_l518_518311


namespace johnny_laps_per_minute_approx_l518_518589

noncomputable def johnny_laps_per_minute (L T : ℝ) : ℝ := L / T

theorem johnny_laps_per_minute_approx :
  ∀ (L T : ℝ), L = 10 → T = 3.333 → johnny_laps_per_minute L T ≈ 3 := by
  intros L T hL hT
  rw [hL, hT]
  sorry

end johnny_laps_per_minute_approx_l518_518589


namespace canoe_no_paddle_time_l518_518070

-- All conditions needed for the problem
variables {S v v_r : ℝ}
variables (time_pa time_pb : ℝ)

-- Condition that time taken from A to B is 3 times the time taken from B to A
def condition1 : Prop := time_pa = 3 * time_pb

-- Define time taken from A to B (downstream) and B to A (upstream)
def time_pa_def : time_pa = S / (v + v_r) := sorry
def time_pb_def : time_pb = S / (v - v_r) := sorry

-- Main theorem stating the problem to prove
theorem canoe_no_paddle_time :
  condition1 →
  ∃ (t_no_paddle : ℝ), t_no_paddle = 3 * time_pb :=
begin
  intro h1,
  sorry
end

end canoe_no_paddle_time_l518_518070


namespace sqrt_arith_progression_impossible_l518_518997

theorem sqrt_arith_progression_impossible (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hneab : a ≠ b) (hnebc : b ≠ c) (hneca : c ≠ a) :
  ¬ ∃ d : ℝ, (d = (Real.sqrt b - Real.sqrt a)) ∧ (d = (Real.sqrt c - Real.sqrt b)) :=
sorry

end sqrt_arith_progression_impossible_l518_518997


namespace largest_fraction_l518_518480

theorem largest_fraction (x y z w : ℝ) (hx : 0 < x) (hxy : x < y) (hyz : y < z) (hzw : z < w) :
  max (max (max (max ((x + y) / (z + w)) ((x + w) / (y + z))) ((y + z) / (x + w))) ((y + w) / (x + z))) ((z + w) / (x + y)) = (z + w) / (x + y) :=
by sorry

end largest_fraction_l518_518480


namespace num_possible_matrices_l518_518677

theorem num_possible_matrices : 
  (∑ (M : Matrix (Fin 4) (Fin 4) ℕ), 
    ((∀ i j, 1 ≤ M i j ∧ M i j ≤ 16) ∧ 
    (∀ i, StrictIncreasingRow (λ j, M i j)) ∧ 
    (∀ j, StrictIncreasingCol (λ i, M i j)) ∧ 
    ((M 1 1 = 8 ∧ M 1 2 = 9) ∨ (M 1 2 = 8 ∧ M 1 1 = 9)))
   .to_nat) = 2450 :=
sorry

end num_possible_matrices_l518_518677


namespace sufficient_condition_l518_518059

variable (a : ℝ)

theorem sufficient_condition (h : ∀ x : ℝ, -1 ≤ x → x ≤ 2 → x^2 - a ≥ 0) : a ≤ -1 := 
sorry

end sufficient_condition_l518_518059


namespace math_problem_l518_518317

theorem math_problem :
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 5000 := 
by
  sorry

end math_problem_l518_518317


namespace correct_statements_about_quadratic_function_l518_518836

def quadratic_function (x a : ℝ) : ℝ := x^2 - 4 * x + a

theorem correct_statements_about_quadratic_function (a b x : ℝ) :
  (∀ x < 1, (quadratic_function x a) decreases as x increases) ∧
  (if (∃ x : ℝ, quadratic_function x a = 0) then a ≤ 4) ∧
  (∀ x, a = 3 → (1 < x ∧ x < 3) → quadratic_function x 3 > 0) ∧
  (quadratic_function 2013 a = b → quadratic_function (-2009) a = b) :=
by sorry

end correct_statements_about_quadratic_function_l518_518836


namespace terminal_zeros_product_l518_518388

theorem terminal_zeros_product (h₁ : 75 = 5^2 * 3) (h₂ : 360 = 2^3 * 3^2 * 5) :
  nat_trailing_zeros (75 * 360) = 3 := sorry

end terminal_zeros_product_l518_518388


namespace water_fraction_after_replacements_l518_518330

-- Initially given conditions
def radiator_capacity : ℚ := 20
def initial_water_fraction : ℚ := 1
def antifreeze_quarts : ℚ := 5
def replacements : ℕ := 5

-- Derived condition
def water_remain_fraction : ℚ := 3 / 4

-- Statement of the problem
theorem water_fraction_after_replacements :
  (water_remain_fraction ^ replacements) = 243 / 1024 :=
by
  -- Proof goes here
  sorry

end water_fraction_after_replacements_l518_518330


namespace determine_range_of_a_l518_518473

def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

theorem determine_range_of_a (a : ℝ) (h : ∃ x₀ > 0, f a x₀ = 0 ∧ ∀ x,. x ≠ x₀ → f a x ≠ 0) :
  a < -2 :=
sorry

end determine_range_of_a_l518_518473


namespace columns_have_zero_entries_l518_518190

variables {𝔽 : Type*} [field 𝔽] [fintype 𝔽] (A : matrix (fin m) (fin m) 𝔽) [fintype (fin m)]

-- Define symmetric matrix over a two-element field
def is_symmetric (A : matrix (fin m) (fin m) 𝔽) : Prop :=
  Aᵀ = A

-- Define matrix with all diagonal entries zero
def diag_zero (A : matrix (fin m) (fin m) 𝔽) : Prop :=
  ∀ i, A i i = 0

theorem columns_have_zero_entries (n : ℕ) (h1 : is_symmetric A) (h2 : diag_zero A) :
  ∀ j, ∃ i, (A ^ n) i j = 0 :=
sorry

end columns_have_zero_entries_l518_518190


namespace relationship_between_x_and_y_l518_518869

theorem relationship_between_x_and_y (a b : ℝ) : 
  let x := a^2 + b^2 + 20,
      y := 4 * (2 * b - a)
  in x ≥ y :=
by
  -- Define x and y
  let x := a^2 + b^2 + 20
  let y := 4 * (2 * b - a)

  -- We need to show that x ≥ y
  sorry

end relationship_between_x_and_y_l518_518869


namespace relationship_between_lines_l518_518863

-- Define the type for a line and a plane
structure Line where
  -- some properties (to be defined as needed, omitted for brevity)

structure Plane where
  -- some properties (to be defined as needed, omitted for brevity)

-- Define parallelism between a line and a plane
def parallel_line_plane (m : Line) (α : Plane) : Prop := sorry

-- Define line within a plane
def line_within_plane (n : Line) (α : Plane) : Prop := sorry

-- Define parallelism between two lines
def parallel_lines (m n : Line) : Prop := sorry

-- Define skewness between two lines
def skew_lines (m n : Line) : Prop := sorry

-- The mathematically equivalent proof problem
theorem relationship_between_lines (m n : Line) (α : Plane)
  (h1 : parallel_line_plane m α)
  (h2 : line_within_plane n α) :
  parallel_lines m n ∨ skew_lines m n := 
sorry

end relationship_between_lines_l518_518863


namespace problem_1_problem_2_l518_518855

def sequence_an (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  a 1 = 8 ∧ (∀ n ≥ 2, a n = 3 * S (n - 1) + 8)

def sum_Sn (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = ∑ i in range n, a (i + 1)

def sequence_bn (b : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, b n = log2 (a n)

def sequence_cn (c : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, c n = 1 / (b n * b (n + 1))

def sum_Tn (T : ℕ → ℝ) (c : ℕ → ℝ) :=
  ∀ n, T n = ∑ i in range n, c (i + 1)

theorem problem_1 (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) :
  sequence_an a S →
  sum_Sn S a →
  (∀ n, a n = 2^(2n + 1)) →
  sequence_bn b a →
  ∀ n, b n = 2 * n + 1 := 
sorry

theorem problem_2 (b : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, b n = 2 * n + 1) →
  sequence_cn c b →
  sum_Tn T c →
  ∀ n, T n = n / (6 * n + 9) :=
sorry

end problem_1_problem_2_l518_518855


namespace inequality_solution_set_l518_518664

theorem inequality_solution_set (a : ℝ) : 
    (a = 0 → (∃ x : ℝ, x > 1 ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a < 0 → (∃ x : ℝ, (x < 2/a ∨ x > 1) ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (0 < a ∧ a < 2 → (∃ x : ℝ, (1 < x ∧ x < 2/a) ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a = 2 → ¬(∃ x : ℝ, ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a > 2 → (∃ x : ℝ, (2/a < x ∧ x < 1) ∧ ax^2 - (a + 2) * x + 2 < 0)) :=
by sorry

end inequality_solution_set_l518_518664


namespace rotating_AB_area_t_l518_518853

variables {A B P : Point}

-- Condition: segment AB has unit length
axiom unit_length (A B : Point) : dist A B = 1

-- Condition: Point P is outside of segment AB
axiom point_outside (A B P : Point) : ¬ collinear A B P 

-- Given area of the annulus
variable (t : ℝ)

-- Define the function for annulus area based on projection
def annulus_area (X T B : Point) (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then x^2 * π 
else (2 * x - 1) * π

-- Main theorem: areas and symmetry define X
theorem rotating_AB_area_t (X : Point) :
  let T := perpendicular_projection X A B in
  (annulus_area X T B (dist T B) = t) ↔ 
  (X lies on the perpendicular from P to A B) 
  ∨ (X is the reflection of P across the perpendicular bisector of A B) := 
by sorry

end rotating_AB_area_t_l518_518853


namespace segment_intersection_l518_518581

theorem segment_intersection 
  (n : ℕ) (h_n_pos : 0 < n) 
  (segments : Fin 4n → ℝ × ℝ × ℝ × ℝ) 
  (h_segment_len : ∀ i, let ⟨x1, y1, x2, y2⟩ := segments i in (x2 - x1)^2 + (y2 - y1)^2 = 1)
  (h_within_circle : ∀ i, let ⟨x1, y1, x2, y2⟩ := segments i in x1^2 + y1^2 ≤ n^2 ∧ x2^2 + y2^2 ≤ n^2) 
  : ∀ i, ∃ j ≠ i, (let ⟨x1_i, y1_i, x2_i, y2_i⟩ := segments i, ⟨x1_j, y1_j, x2_j, y2_j⟩ := segments j in
    ((x2_i - x1_i) * (x2_j - x1_j) + (y2_i - y1_i) * (y2_j - y1_j) = 0 ∨ 
     (x2_i - x1_i) * (y2_j - y1_j) = (x2_j - x1_j) * (y2_i - y1_i)) ∧
    (∃ k₁ k₂, k₁ ≠ k₂ ∧ line_segment_intersects (segments j) (segments k₁) ∧ line_segment_intersects (segments j) (segments k₂))) := 
sorry

noncomputable def line_segment_intersects : (ℝ × ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ × ℝ) → Prop
| ⟨x1, y1, x2, y2⟩ ⟨x3, y3, x4, y4⟩ := 
  ∃ t1 t2, 0 < t1 < 1 ∧ 0 < t2 < 1 ∧ 
   (x1 * (1 - t1) + x2 * t1 = x3 * (1 - t2) + x4 * t2) ∧ 
   (y1 * (1 - t1) + y2 * t1 = y3 * (1 - t2) + y4 * t2)         

end segment_intersection_l518_518581


namespace range_of_k_for_local_minimum_at_1_l518_518876

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  (x-2)*Real.exp x - (k/2)*x^2 + k*x

theorem range_of_k_for_local_minimum_at_1 (k : ℝ) (h : k > 0) (h_min : ∃ δ > 0, ∀ ε > 0, x : ℝ, abs (x - 1) < δ → (f k).deriv x < (f k).deriv 1 + ε) :
  0 < k ∧ k < Real.exp 1 :=
sorry

end range_of_k_for_local_minimum_at_1_l518_518876


namespace problem_statement_l518_518447

variable {n : ℕ}
variable {x : Fin n → ℝ}

def mean (data : Fin n → ℝ) : ℝ := (∑ i, data i) / n
def stddev (data : Fin n → ℝ) : ℝ := Real.sqrt ((∑ i, (data i - mean data) ^ 2) / n)
def median (data : Fin n → ℝ) : ℝ := sorry
def range (data : Fin n → ℝ) : ℝ := (Finset.sup Finset.univ (data)) - (Finset.inf Finset.univ (data))

variable {dataA : Fin n → ℝ}
variable {dataB : Fin n → ℝ} := λ i, 3 * dataA i - 1

theorem problem_statement (mean_A stddev_A median_A range_A : ℝ) (mean_B stddev_B median_B range_B : ℝ)
  (h1 : mean dataA = mean_A) (h2 : stddev dataA = stddev_A)
  (h3 : median dataA = median_A) (h4 : range dataA = range_A)
  (h5 : mean dataB = mean_B) (h6 : stddev dataB = stddev_B)
  (h7 : median dataB = median_B) (h8 : range dataB = range_B) :
  stddev_A = 3 * stddev_B ∧ range_A = 3 * range_B := sorry

end problem_statement_l518_518447


namespace intersections_concyclic_or_collinear_l518_518861

-- Given four circles S1, S2, S3, S4

variables {S1 S2 S3 S4 : Type*} [circle S1] [circle S2] [circle S3] [circle S4]
variables {A1 A2 B1 B2 C1 C2 D1 D2 : Type*} 
  [point A1] [point A2] [point B1] [point B2]
  [point C1] [point C2] [point D1] [point D2]

-- Defining the intersections based on the conditions given
variables (h₁ : point_on_circle A1 S1) (h₂ : point_on_circle A1 S2)
variables (h₃ : point_on_circle A2 S1) (h₄ : point_on_circle A2 S2)
variables (h₅ : point_on_circle B1 S2) (h₆ : point_on_circle B1 S3)
variables (h₇ : point_on_circle B2 S2) (h₈ : point_on_circle B2 S3)
variables (h₉ : point_on_circle C1 S3) (h₁₀ : point_on_circle C1 S4)
variables (h₁₁ : point_on_circle C2 S3) (h₁₂ : point_on_circle C2 S4)
variables (h₁₃ : point_on_circle D1 S4) (h₁₄ : point_on_circle D1 S1)
variables (h₁₅ : point_on_circle D2 S4) (h₁₆ : point_on_circle D2 S1)

-- Given: A1, B1, C1, D1 are concyclic
variables (hconcyc : concyclic A1 B1 C1 D1)

-- The theorem to prove: A2, B2, C2, D2 are concyclic or collinear
theorem intersections_concyclic_or_collinear :
  concyclic A2 B2 C2 D2 ∨ collinear A2 B2 C2 D2 := sorry

end intersections_concyclic_or_collinear_l518_518861


namespace max_citizens_l518_518213

theorem max_citizens (n : ℕ) : (nat.choose n 4 < nat.choose n 2 ↔ n ≤ 5) :=
by
  -- We state the conditions given by the problem
  sorry

end max_citizens_l518_518213


namespace liars_at_table_l518_518642

open Set

noncomputable def number_of_liars : Set ℕ :=
  {n | ∃ (knights, liars : ℕ), knights + liars = 450 ∧
                                (∀ i : ℕ, i < 450 → (liars + ((i + 1) % 450) + ((i + 2) % 450) = 1)) }

theorem liars_at_table : number_of_liars = {150, 450} := 
  sorry

end liars_at_table_l518_518642


namespace transform_equation_to_polynomial_l518_518308

variable (x y : ℝ)

theorem transform_equation_to_polynomial (h : (x^2 + 2) / (x + 1) = y) :
    (x^2 + 2) / (x + 1) + (5 * (x + 1)) / (x^2 + 2) = 6 → y^2 - 6 * y + 5 = 0 :=
by
  intro h_eq
  sorry

end transform_equation_to_polynomial_l518_518308


namespace ratio_of_area_pentagon_to_squares_l518_518714

open Real

structure Point :=
(x : ℝ)
(y : ℝ)

structure Square :=
(A : Point)
(B : Point)
(C : Point)
(D : Point)
(side_length : ℝ)

def midpoint (p1 p2 : Point) : Point :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2 }

noncomputable def area_triangle (p1 p2 p3 : Point) : ℝ :=
abs ((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2)

noncomputable def area_pentagon (A J I C B : Point) : ℝ :=
area_triangle A J I + area_triangle A I B +
area_triangle A B C + area_triangle A C J
  
noncomputable def area_square (s : Square) : ℝ :=
s.side_length ^ 2

noncomputable def total_area_squares (s1 s2 s3 : Square) : ℝ :=
area_square s1 + area_square s2 + area_square s3

theorem ratio_of_area_pentagon_to_squares
  (s1 s2 s3 : Square)
  (A J I C B : Point)
  (h1 : s1.side_length = 2)
  (h2 : s2.side_length = 2)
  (h3 : s3.side_length = 2)
  (h_midpt_C : C = midpoint s3.G s3.H)
  (h_midpt_D : s1.D = midpoint s2.E s2.F)
  (h_adj_AB_EF : s1.B = s2.F)
  (h_adj_GH_IJ : s2.H = s3.I) :
  (area_pentagon A J I C B) / total_area_squares s1 s2 s3 = 1 / 6 :=
sorry

end ratio_of_area_pentagon_to_squares_l518_518714


namespace points_collinear_l518_518300

open Set

theorem points_collinear (S : Finset Point)
  (h : ∀ (A B : Point), A ∈ S → B ∈ S → A ≠ B → ∃ C ∈ S, C ≠ A ∧ C ≠ B ∧ Collinear ({A, B, C} : Set Point)) :
  ∃ l : Line, ∀ P ∈ S, P ∈ l :=
sorry

end points_collinear_l518_518300


namespace part1_solution_part2_solution_l518_518909

-- Definitions of the lines
def l1 (a : ℝ) : ℝ × ℝ × ℝ :=
  (2 * a + 1, a + 2, 3)

def l2 (a : ℝ) : ℝ × ℝ × ℝ :=
  (a - 1, -2, 2)

-- Parallel lines condition
def parallel_lines (a : ℝ) : Prop :=
  let (A1, B1, C1) := l1 a
  let (A2, B2, C2) := l2 a
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ (A1 / B1) = (A2 / B2)

-- Perpendicular lines condition
def perpendicular_lines (a : ℝ) : Prop :=
  let (A1, B1, C1) := l1 a
  let (A2, B2, C2) := l2 a
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ (A1 * A2 + B1 * B2 = 0)

-- Statement for part 1
theorem part1_solution (a : ℝ) : parallel_lines a ↔ a = 0 :=
  sorry

-- Statement for part 2
theorem part2_solution (a : ℝ) : perpendicular_lines a ↔ (a = -1 ∨ a = 5 / 2) :=
  sorry


end part1_solution_part2_solution_l518_518909


namespace area_of_triangle_find_b_l518_518151

variables {α : Type*} [linear_ordered_field α] [decidable_eq α] [mul_one_class α]

-- Define the given conditions.
def sin_half_B_eq_sqrt5_div_5 (B : α) : Prop := sin (B / 2) = real.sqrt 5 / 5
def dot_product_eq_6 {V : Type*} [inner_product_space α V] (BA BC : V) : Prop := ⟪BA, BC⟫ = 6
def sum_c_a_eq_8 (c a : α) : Prop := c + a = 8

-- Problem 1: Calculate the area of triangle ABC.
theorem area_of_triangle 
  (A B C : α) 
  (a b c : α) 
  (BA BC : real) 
  (h1 : sin_half_B_eq_sqrt5_div_5 B)
  (h2 : dot_product_eq_6 BA BC) :
  real := 
  sorry

-- Problem 2: Given c + a = 8, find the value of b.
theorem find_b
  (A B C : α) 
  (a b c : α)
  (BA BC : real)
  (h1 : sin_half_B_eq_sqrt5_div_5 B)
  (h2 : dot_product_eq_6 BA BC)
  (h3 : sum_c_a_eq_8 c a) :
  b = 4 * real.sqrt 2 :=
  sorry

end area_of_triangle_find_b_l518_518151


namespace ac_bd_leq_8_l518_518451

theorem ac_bd_leq_8 (a b c d : ℝ) (h1 : a^2 + b^2 = 4) (h2 : c^2 + d^2 = 16) : ac + bd ≤ 8 :=
sorry

end ac_bd_leq_8_l518_518451


namespace gcd_sum_l518_518012

theorem gcd_sum : ∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (3 * n + 5) n}, d = 6 :=
by
  sorry

end gcd_sum_l518_518012


namespace students_in_at_least_one_group_l518_518560

-- Define the number of students in each of the groups
def st_group : ℕ := 65
def speech_group : ℕ := 35
def both_groups : ℕ := 20

-- Theorem stating the number of students who joined at least one of the interest groups
theorem students_in_at_least_one_group 
  (st_group : ℕ) 
  (speech_group : ℕ) 
  (both_groups : ℕ) 
  : st_group = 65 → speech_group = 35 → both_groups = 20 → (st_group + speech_group - both_groups = 80) := 
by {
  intros,
  rw [a, b, c],
  done,
}

end students_in_at_least_one_group_l518_518560


namespace sum_of_first_1730_terms_l518_518697

noncomputable section

def a : ℕ → ℝ
| 0 => 1
| 1 => 2
| n + 2 => a (n + 1) / a n

def sum_seq : ℕ → ℝ 
| 0 => a 0
| n => sum_seq (n - 1) + a n

theorem sum_of_first_1730_terms : sum_seq 1729 = 2019 := 
by 
  sorry

end sum_of_first_1730_terms_l518_518697


namespace number_of_roses_l518_518283

noncomputable def total_flowers := 100
noncomputable def tulips := 40
noncomputable def daisies := 35
noncomputable def not_roses_percent := 0.75
noncomputable def not_roses_number := tulips + daisies
noncomputable def roses := 0.25 * total_flowers

theorem number_of_roses :
  not_roses_percent * total_flowers = not_roses_number →
  25% (total_flowers) = roses →
  roses = 25 :=
by
  -- Given that total_flowers * 0.75 = not_roses_number and 0.25 * total_flowers = roses
  intros
  sorry

end number_of_roses_l518_518283


namespace marbles_remaining_correct_l518_518808

-- Define the number of marbles Chris has
def marbles_chris : ℕ := 12

-- Define the number of marbles Ryan has
def marbles_ryan : ℕ := 28

-- Define the total number of marbles in the pile
def total_marbles : ℕ := marbles_chris + marbles_ryan

-- Define the number of marbles each person takes away from the pile
def marbles_taken_each : ℕ := total_marbles / 4

-- Define the total number of marbles taken away
def total_marbles_taken : ℕ := 2 * marbles_taken_each

-- Define the number of marbles remaining in the pile
def marbles_remaining : ℕ := total_marbles - total_marbles_taken

theorem marbles_remaining_correct : marbles_remaining = 20 := by
  sorry

end marbles_remaining_correct_l518_518808


namespace num_mountain_numbers_l518_518813

def is_mountain_number (n : ℕ) : Prop :=
  let d1 := n / 1000 in
  let d2 := (n % 1000) / 100 in
  let d3 := (n % 100) / 10 in
  let d4 := n % 10 in
  1000 ≤ n ∧ n < 10000 ∧
  d1 < d2 ∧ d2 = d3 ∧ d3 > d4 ∧ d1 ≠ d2 ∧ d2 ≠ 0

theorem num_mountain_numbers : { n : ℕ // is_mountain_number n }.card = 44 :=
sorry

end num_mountain_numbers_l518_518813


namespace problem_equivalence_l518_518895

noncomputable def f (a b x : ℝ) : ℝ := a ^ x + b

theorem problem_equivalence (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : f a b 0 = -2) (h4 : f a b 2 = 0) :
    a = Real.sqrt 3 ∧ b = -3 ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 4, (-8 / 3 : ℝ) ≤ f a b x ∧ f a b x ≤ 6) :=
sorry

end problem_equivalence_l518_518895


namespace time_to_travel_from_B_to_A_without_paddles_l518_518074

-- Variables definition 
variables (v v_r S : ℝ)
-- Assume conditions
def condition_1 (t₁ t₂ : ℝ) (v v_r S : ℝ) := t₁ = 3 * t₂
def t₁ (S v v_r : ℝ) := S / (v + v_r)
def t₂ (S v v_r : ℝ) := S / (v - v_r)

theorem time_to_travel_from_B_to_A_without_paddles
  (v v_r S : ℝ)
  (h1 : v = 2 * v_r)
  (h2 : t₁ S v v_r = 3 * t₂ S v v_r) :
  let t_no_paddle := S / v_r in
  t_no_paddle = 3 * t₂ S v v_r :=
sorry

end time_to_travel_from_B_to_A_without_paddles_l518_518074


namespace area_increase_l518_518554

theorem area_increase (l w : ℝ) :
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_original := l * w
  let A_new := l_new * w_new
  ((A_new - A_original) / A_original) * 100 = 56 := 
by
  sorry

end area_increase_l518_518554


namespace travel_time_without_paddles_l518_518075

variables (A B : Type) (v v_r S : ℝ)
noncomputable def time_to_travel (distance velocity : ℝ) := distance / velocity

-- Condition: The travel time from A to B is 3 times the travel time from B to A
axiom travel_condition : (time_to_travel S (v + v_r)) = 3 * (time_to_travel S (v - v_r))

-- Condition: We are considering travel from B to A by canoe without paddles
noncomputable def time_without_paddles := time_to_travel S v_r

-- Proving that without paddles it takes 3 times longer than usual (using canoes with paddles)
theorem travel_time_without_paddles :
  time_without_paddles S v_r = 3 * (time_to_travel S (v - v_r)) :=
sorry

end travel_time_without_paddles_l518_518075


namespace largest_n_for_factorable_polynomial_l518_518414

theorem largest_n_for_factorable_polynomial : ∃ n, 
  (∀ A B : ℤ, (6 * B + A = n) → (A * B = 144)) ∧ 
  (∀ n', (∀ A B : ℤ, (6 * B + A = n') → (A * B = 144)) → n' ≤ n) ∧ 
  (n = 865) :=
by
  sorry

end largest_n_for_factorable_polynomial_l518_518414


namespace ways_to_distribute_balls_in_boxes_l518_518523

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l518_518523


namespace greatest_number_of_symmetry_lines_l518_518734

-- Defining the figures and their properties
def right_triangle_has_symmetry_lines (n : ℕ) : Prop :=
  n = 0 ∨ n = 1

def parallelogram_has_symmetry_lines (n : ℕ) : Prop :=
  n = 0

def regular_pentagon_has_symmetry_lines (n : ℕ) : Prop :=
  n = 5

def isosceles_trapezoid_has_symmetry_lines (n : ℕ) : Prop :=
  n = 1

def square_has_symmetry_lines (n : ℕ) : Prop :=
  n = 4

-- Theorem to prove
theorem greatest_number_of_symmetry_lines :
  (∀ n, right_triangle_has_symmetry_lines n → n ≤ 5) ∧
  (∀ n, parallelogram_has_symmetry_lines n → n ≤ 5) ∧
  (∀ n, regular_pentagon_has_symmetry_lines n → n = 5) ∧
  (∀ n, isosceles_trapezoid_has_symmetry_lines n → n ≤ 5) ∧
  (∀ n, square_has_symmetry_lines n → n ≤ 5) ∧
  (\forall n, (right_triangle_has_symmetry_lines n ∨ parallelogram_has_symmetry_lines n ∨ regular_pentagon_has_symmetry_lines n ∨ isosceles_trapezoid_has_symmetry_lines n ∨ square_has_symmetry_lines n) → n ≤ 5) ∧
  (\forall n, right_triangle_has_symmetry_lines n ∨ parallelogram_has_symmetry_lines n ∨ regular_pentagon_has_symmetry_lines n ∨ isosceles_trapezoid_has_symmetry_lines n ∨ square_has_symmetry_lines n → 
  n = 5) := sorry

end greatest_number_of_symmetry_lines_l518_518734


namespace all_values_equal_l518_518796

-- Definitions and conditions
def grid_value (a : ℤ → ℤ → ℕ) : Prop :=
∀ i j : ℤ, 
  (a i j = (a (i-1) j + a (i+1) j + a i (j-1) + a i (j+1)) / 4)

theorem all_values_equal (a : ℤ → ℤ → ℕ) (h : grid_value a) :
  ∃ c : ℕ, ∀ i j : ℤ, a i j = c :=
begin
  sorry
end

end all_values_equal_l518_518796


namespace circle_rotation_creates_solid_l518_518782

/-- When a circle (2D surface) is rotated around the line where its diameter lies, 
    the resulting shape is a solid (3D space). -/
theorem circle_rotation_creates_solid :
  ∀ (R : ℝ), ∃ (sphere : Type), rotation_around_diameter (circle R) = sphere := sorry

end circle_rotation_creates_solid_l518_518782


namespace triangle_AC_greater_than_4_l518_518093

-- Define the triangle and points as the given conditions
variable {A B C M N : Type}
variable [linear_ordered_field A]
variable [linear_ordered_field B]
variable [linear_ordered_field C]
variable [linear_ordered_field M]
variable [linear_ordered_field N]

-- Define the segments and equality conditions
variable (t : Triangle A B C)
variable (M_on_AB : M ∈ segment A B)
variable (N_on_BC : N ∈ segment B C)
variable (MN_parallel_AC : MN ∥ AC)
variable (BN_eq_1 : BN = 1)
variable (MN_eq_2 : MN = 2)
variable (AM_eq_3 : AM = 3)

-- Prove the required inequality
theorem triangle_AC_greater_than_4 (t : Triangle A B C)
    (M_on_AB : M ∈ segment A B) 
    (N_on_BC : N ∈ segment B C)
    (MN_parallel_AC : MN ∥ AC)
    (BN_eq_1 : BN = 1)
    (MN_eq_2 : MN = 2)
    (AM_eq_3 : AM = 3) : AC > 4 :=
by
    sorry

end triangle_AC_greater_than_4_l518_518093


namespace problem_1_l518_518122

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem problem_1 :
  (∀ x : ℝ, f x = Real.sin (2 * x + π / 6) + 1/2) ∧
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 3) → (∃ c,  f' c > 0)) ∧
  (Sup (f '' (set.Icc (-π / 6) (π / 3))) = 3/2) ∧
  (Inf (f '' (set.Icc (-π / 6) (π / 3))) = 0) :=
by sorry

end problem_1_l518_518122


namespace ryan_bread_slices_l518_518656

theorem ryan_bread_slices 
  (num_pb_people : ℕ)
  (pb_sandwiches_per_person : ℕ)
  (num_tuna_people : ℕ)
  (tuna_sandwiches_per_person : ℕ)
  (num_turkey_people : ℕ)
  (turkey_sandwiches_per_person : ℕ)
  (slices_per_pb_sandwich : ℕ)
  (slices_per_tuna_sandwich : ℕ)
  (slices_per_turkey_sandwich : ℝ)
  (h1 : num_pb_people = 4)
  (h2 : pb_sandwiches_per_person = 2)
  (h3 : num_tuna_people = 3)
  (h4 : tuna_sandwiches_per_person = 3)
  (h5 : num_turkey_people = 2)
  (h6 : turkey_sandwiches_per_person = 1)
  (h7 : slices_per_pb_sandwich = 2)
  (h8 : slices_per_tuna_sandwich = 3)
  (h9 : slices_per_turkey_sandwich = 1.5) : 
  (num_pb_people * pb_sandwiches_per_person * slices_per_pb_sandwich 
  + num_tuna_people * tuna_sandwiches_per_person * slices_per_tuna_sandwich 
  + (num_turkey_people * turkey_sandwiches_per_person : ℝ) * slices_per_turkey_sandwich) = 46 :=
by
  sorry

end ryan_bread_slices_l518_518656


namespace max_probability_l518_518991
open Classical

noncomputable theory
open Set

constant Ω : Type
constant P : Set Ω → ℝ      -- Probability measure

variables (A B C : Set Ω)

-- Conditions
def pairwise_independent : Prop :=
  P (A ∩ B) = P A * P B ∧
  P (B ∩ C) = P B * P C ∧
  P (A ∩ C) = P A * P C

def equal_probabilities : Prop :=
  P A = P B ∧ P B = P C

def disjoint_intersection : Prop :=
  A ∩ B ∩ C = ∅

-- The statement to prove
theorem max_probability (h1 : pairwise_independent A B C) (h2 : equal_probabilities A B C) (h3 : disjoint_intersection A B C) :
  P A ≤ 1 / 2 :=
sorry

end max_probability_l518_518991


namespace quadratic_inequality_solution_l518_518842

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 50 * x + 575 ≤ 25} = set.Icc (25 - 5 * real.sqrt 3) (25 + 5 * real.sqrt 3) :=
by {
  sorry
}

end quadratic_inequality_solution_l518_518842


namespace correct_N_l518_518827

open Matrix

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![ -3 ,  4 , 1 ],
    ![ 5 , -7 , 0 ],
    ![ 0 ,  0 , 1 ]]

noncomputable def N : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![ -7 , -4 , 7 ],
    ![ 25 , 9 , -25 ],
    ![  0 ,  0 ,  1 ]]

theorem correct_N :
  N ⬝ A = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by 
  sorry

end correct_N_l518_518827


namespace contains_spatial_quadrilateral_l518_518453

noncomputable theory

variables {V : Type} [fintype V] [nonempty V] [inhabited V]

-- Definitions based on the conditions
def n (q : ℕ) : ℕ := q^2 + q + 1
def l (q : ℕ) : ℕ := nat.ceil ((1 / 2 : ℝ) * q * (q + 1)^2) + 1
def q_min (q : ℕ) : Prop := q ≥ 2

-- The graph with n vertices and at least l edges
variables (G : simple_graph V)
variables {q : ℕ} (hq : q ≥ 2)
variables [fintype V] (v : V) (hl : G.edge_set.card ≥ l q)
variables (hn : fintype.card V = n q)
variables (h_non_coplanar : ∀ (p1 p2 p3 p4 : V), p1 ≠ p2 → p1 ≠ p3 → p1 ≠ p4 → p2 ≠ p3 → p2 ≠ p4 → p3 ≠ p4 → 
  ¬ collinear G {p1, p2, p3, p4})
variables (h_connected : ∀ (v : V), (G.neighbor_finset v).nonempty)
variables (h_high_degree : ∃ v : V, G.degree v ≥ q + 2)

-- The proof statement
theorem contains_spatial_quadrilateral :
  ∃ (p1 p2 p3 p4 : V), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ 
  G.adj p1 p2 ∧ G.adj p2 p3 ∧ G.adj p3 p4 ∧ G.adj p4 p1 :=
sorry

end contains_spatial_quadrilateral_l518_518453


namespace y_intercept_point_of_line_l518_518048

theorem y_intercept_point_of_line (x y : ℝ) (h : 6 * x + 10 * y = 40) : (0, 4) = (0, y) :=
by
  have hx : x = 0 := by sorry -- We assume we are at the intercept
  have hy : y = 4 := by sorry -- Simplification (hand wave for now)
  exact hx ▸ hy ▸ rfl

end y_intercept_point_of_line_l518_518048


namespace symmetry_y_axis_l518_518261

def f (x : ℝ) : ℝ := Real.sin (x + 5 * Real.pi / 2)

theorem symmetry_y_axis (x : ℝ) : f (-x) = f x := by
  sorry

end symmetry_y_axis_l518_518261


namespace point_A_coordinates_l518_518260

variable {a : ℝ}
variable {f : ℝ → ℝ}

theorem point_A_coordinates (h1 : a > 0) (h2 : a ≠ 1) (hf : ∀ x, f x = a^(x - 1)) :
  f 1 = 1 :=
by
  sorry

end point_A_coordinates_l518_518260


namespace solution_set_l518_518888

def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

theorem solution_set : { x : ℝ | f x > 1 } = Set.Ioo (2/3) 2 :=
by
  sorry

end solution_set_l518_518888


namespace area_of_bounded_region_l518_518389

-- Define the condition for the region
def condition (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ (50 * fract y) ≥ floor y + floor x

-- Define the function to compute the area under the given conditions
noncomputable def region_area : ℝ :=
∫ x in (0:ℝ)..(some upper_bound_of_x), ∫ y in (0:ℝ)..(some upper_bound_of_y), if condition x y then 1 else 0

-- The area of the region should be 25
theorem area_of_bounded_region : region_area = 25 :=
by
  -- Formal steps of the proof would go here
  sorry

end area_of_bounded_region_l518_518389


namespace thomas_path_exists_l518_518563

open Classical

theorem thomas_path_exists (cities : Finset α) (roads : Finset (α × α)) (M Z : α) :
  cities.card = 15 ∧ roads.card = 20 ∧ (M ≠ Z) ∧ (∀ r, r ∈ roads → r.fst ∈ cities ∧ r.snd ∈ cities) →
  (∃ path : list (α × α), path.length = 15 ∧ 
    path.head = (M, some_city) ∧ 
    path.last = (some_city', Z) ∧
    ∀ r, r ∈ roads → list.count (λ x, x = r) path = 1) := 
sorry

end thomas_path_exists_l518_518563


namespace solve_p_n_k_l518_518387

theorem solve_p_n_k :
    ∀ (p n k : ℕ), p.prime → 0 < p → 0 < n → 0 < k →
    144 + p^n = k^2 →
    (p = 5 ∧ n = 2 ∧ k = 13) ∨ 
    (p = 2 ∧ n = 8 ∧ k = 20) ∨ 
    (p = 3 ∧ n = 4 ∧ k = 15) := by
  sorry

end solve_p_n_k_l518_518387


namespace travel_time_without_paddles_l518_518077

variables (A B : Type) (v v_r S : ℝ)
noncomputable def time_to_travel (distance velocity : ℝ) := distance / velocity

-- Condition: The travel time from A to B is 3 times the travel time from B to A
axiom travel_condition : (time_to_travel S (v + v_r)) = 3 * (time_to_travel S (v - v_r))

-- Condition: We are considering travel from B to A by canoe without paddles
noncomputable def time_without_paddles := time_to_travel S v_r

-- Proving that without paddles it takes 3 times longer than usual (using canoes with paddles)
theorem travel_time_without_paddles :
  time_without_paddles S v_r = 3 * (time_to_travel S (v - v_r)) :=
sorry

end travel_time_without_paddles_l518_518077


namespace min_value_fraction_l518_518136

theorem min_value_fraction (x y : ℝ) 
  (h1 : x - 1 ≥ 0)
  (h2 : x - y + 1 ≤ 0)
  (h3 : x + y - 4 ≤ 0) : 
  ∃ a, (∀ x y, (x - 1 ≥ 0) ∧ (x - y + 1 ≤ 0) ∧ (x + y - 4 ≤ 0) → (x / (y + 1)) ≥ a) ∧ 
      (a = 1 / 4) :=
sorry

end min_value_fraction_l518_518136


namespace work_efficiency_ratio_l518_518764

-- Define the problem conditions and the ratio we need to prove.
theorem work_efficiency_ratio :
  (∃ (a b : ℝ), b = 1 / 18 ∧ (a + b) = 1 / 12 ∧ (a / b) = 1 / 2) :=
by {
  -- Definitions and variables can be listed if necessary
  -- a : ℝ
  -- b : ℝ
  -- Assume conditions
  sorry
}

end work_efficiency_ratio_l518_518764


namespace car_a_has_higher_avg_speed_l518_518378

-- Definitions of the conditions for Car A
def distance_car_a : ℕ := 120
def speed_segment_1_car_a : ℕ := 60
def distance_segment_1_car_a : ℕ := 40
def speed_segment_2_car_a : ℕ := 40
def distance_segment_2_car_a : ℕ := 40
def speed_segment_3_car_a : ℕ := 80
def distance_segment_3_car_a : ℕ := distance_car_a - distance_segment_1_car_a - distance_segment_2_car_a

-- Definitions of the conditions for Car B
def distance_car_b : ℕ := 120
def time_segment_1_car_b : ℕ := 1
def speed_segment_1_car_b : ℕ := 60
def time_segment_2_car_b : ℕ := 1
def speed_segment_2_car_b : ℕ := 40
def total_time_car_b : ℕ := 3
def distance_segment_1_car_b := speed_segment_1_car_b * time_segment_1_car_b
def distance_segment_2_car_b := speed_segment_2_car_b * time_segment_2_car_b
def time_segment_3_car_b := total_time_car_b - time_segment_1_car_b - time_segment_2_car_b
def distance_segment_3_car_b := distance_car_b - distance_segment_1_car_b - distance_segment_2_car_b
def speed_segment_3_car_b := distance_segment_3_car_b / time_segment_3_car_b

-- Total Time for Car A
def time_car_a := distance_segment_1_car_a / speed_segment_1_car_a
                + distance_segment_2_car_a / speed_segment_2_car_a
                + distance_segment_3_car_a / speed_segment_3_car_a

-- Average Speed for Car A
def avg_speed_car_a := distance_car_a / time_car_a

-- Total Time for Car B
def time_car_b := total_time_car_b

-- Average Speed for Car B
def avg_speed_car_b := distance_car_b / time_car_b

-- Proof that Car A has a higher average speed than Car B
theorem car_a_has_higher_avg_speed : avg_speed_car_a > avg_speed_car_b := by sorry

end car_a_has_higher_avg_speed_l518_518378


namespace max_value_ln_over_x_l518_518266

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_ln_over_x : ∀ x : ℝ, x > 0 → ∃ y, (y = (Real.log x) / x) ∧ (∀ a : ℝ, a > 0 → (Real.log a) / a ≤ y) :=
by
  intro x hx
  use (1 / Real.exp 1)
  split
  · -- Show that 1/e is the value of the function at x=e
    rw [Real.exp_eq_exp (1:ℝ)]
    have heq : x = Real.exp 1, from sorry
    rw [heq]
    simp
  · -- Show that this is the maximum value
    intro a ha
    refine sorry

end max_value_ln_over_x_l518_518266


namespace largest_sector_area_l518_518037

/-- Given a circle with a radius of 3 cm, divided into four sectors 
    with the ratio of their central angles being 2:3:3:4,
    the area of the largest sector is \(\frac{4\pi}{3} \text{cm}^2\). -/
theorem largest_sector_area :
  let r := 3
  let ratio := [2, 3, 3, 4]
  let angle_sum : ℝ := 360
  let total_ratio : ℝ := ratio.sum
  let angles := ratio.map (λ x, (x / total_ratio) * angle_sum)
  let max_angle := angles.max'
  let θ := max_angle
  let area := λ r θ, (θ / 360) * π * r^2
  θ = 120 → area r θ = (4 * π) / 3 :=
  by
  sorry

end largest_sector_area_l518_518037


namespace no_real_or_imaginary_values_satisfy_l518_518256

open Real

theorem no_real_or_imaginary_values_satisfy :
  ∀ x : ℂ, sqrt (49 - x^2) + 7 ≠ 0 :=
by
  intro x
  sorry

end no_real_or_imaginary_values_satisfy_l518_518256


namespace MQ_and_AB_properties_l518_518446

noncomputable theory

-- Define the circle
def M : ℝ → ℝ → Prop := λ x y, x^2 + (y - 2)^2 = 1

-- Conditions for point Q, |AB| and properties
def is_point_on_x_axis (q : ℝ → Prop) := ∀ y, y = 0 -> q y
def AB_length : ℝ := (4 * real.sqrt 2) / 3
def QA_tangent_to_M (q a : ℝ → ℝ → Prop) := is_tangent q a M
def QB_tangent_to_M (q b : ℝ → ℝ → Prop) := is_tangent q b M

-- Define the proposition to be proved
theorem MQ_and_AB_properties :
  ∀ q : ℝ → ℝ → Prop,
  is_point_on_x_axis q →
  (∃ a b : ℝ → ℝ → Prop, QA_tangent_to_M q a ∧ QB_tangent_to_M q b ∧ |AB| = AB_length) →
  ∃ mq : ℝ, mq = 3 ∧
  (∃ l : ℝ → ℝ → Prop, (∃ x : ℝ, l x 0 = 2x + real.sqrt 5 * l x 0 - 2 * real.sqrt 5 = 0) ∨
                     (∃ x : ℝ, l x 0 = 2x - real.sqrt 5 * l x 0 + 2 * real.sqrt 5 = 0)) ∧
  (∃ ab : ℝ → ℝ → Prop, ab (0, real.sqrt 5) = (0, 3/2)) :=
begin
  sorry
end

end MQ_and_AB_properties_l518_518446


namespace total_weight_of_nuts_l518_518334

theorem total_weight_of_nuts :
  let almonds_g := 140
  let pecans_lb := 0.56
  let grams_per_pound := 453.592
  let kilograms_per_gram := 1 / 1000
  let total_weight_g := almonds_g + pecans_lb * grams_per_pound
  let total_weight_kg := total_weight_g * kilograms_per_gram
  total_weight_kg = 0.3936112 :=
by
  let almonds_g := 140
  let pecans_lb := 0.56
  let grams_per_pound := 453.592
  let kilograms_per_gram := 1 / 1000
  let total_weight_g := almonds_g + pecans_lb * grams_per_pound
  let total_weight_kg := total_weight_g * kilograms_per_gram
  show total_weight_kg = 0.3936112
  sorry

end total_weight_of_nuts_l518_518334


namespace graph_intersects_self_24_times_l518_518770

noncomputable def x (t : ℝ) : ℝ := cos t + t / 2 + sin (2 * t)
noncomputable def y (t : ℝ) : ℝ := sin t

theorem graph_intersects_self_24_times :
  (∃ k_values : Finset ℕ, (∀ k ∈ k_values, 1 ≤ x (k * Real.pi) ∧ x (k * Real.pi) ≤ 80)
    ∧ k_values.card = 24) :=
sorry

end graph_intersects_self_24_times_l518_518770


namespace parabola_focus_properties_l518_518983

-- Definitions of the conditions
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y
def focus_distance (p m : ℝ) : ℝ := p / 2 + 4

-- Main proof problem statement
theorem parabola_focus_properties (m p : ℝ) (h1 : m > 0) (h2 : p > 0)
                                   (h3 : parabola p m 4)
                                   (h4 : focus_distance p m = 5) :
  p = 2 ∧ m = 4 ∧ ∃ k : ℝ, (∀ x : ℝ, k*x - 4 = 2*x - 4) :=
sorry

end parabola_focus_properties_l518_518983


namespace arith_seq_a4a6_equals_4_l518_518568

variable (a : ℕ → ℝ) (d : ℝ)
variable (h2 : a 2 = a 1 + d)
variable (h4 : a 4 = a 1 + 3 * d)
variable (h6 : a 6 = a 1 + 5 * d)
variable (h8 : a 8 = a 1 + 7 * d)
variable (h10 : a 10 = a 1 + 9 * d)
variable (condition : (a 2)^2 + 2 * a 2 * a 8 + a 6 * a 10 = 16)

theorem arith_seq_a4a6_equals_4 : a 4 * a 6 = 4 := by
  sorry

end arith_seq_a4a6_equals_4_l518_518568


namespace odd_f_periodic_f_period_4_f_expr_in_2_4_sum_f_periods_l518_518615

noncomputable def f : ℝ → ℝ
| x => if 0 ≤ x ∧ x ≤ 2 then 2 * x - x^2 else sorry

theorem odd_f (x : ℝ) : f (-x) = -f x := sorry

theorem periodic_f_period_4 (x : ℝ) : f (x + 4) = f x := sorry

theorem f_expr_in_2_4 (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4) : f x = x^2 - 6x + 8 := sorry

theorem sum_f_periods : (finset.range 2012).sum (λ x, f x) = 0 :=
begin
  sorry
end

end odd_f_periodic_f_period_4_f_expr_in_2_4_sum_f_periods_l518_518615


namespace sum_of_arithmetic_progressions_l518_518986

/-- The sequences a_n and b_n are arithmetic progressions with the given conditions. -/
theorem sum_of_arithmetic_progressions 
  (a b : ℕ → ℕ) 
  (d_a d_b : ℕ) 
  (h_a1 : a 1 = 10) 
  (h_b1 : b 1 = 90) 
  (h_a50_b50 : a 50 + b 50 = 200) 
  (h_a_n : ∀ n, a n = 10 + (n - 1) * d_a)
  (h_b_n : ∀ n, b n = 90 + (n - 1) * d_b) :
  ∑ n in Finset.range 50, a (n + 1) + b (n + 1) = 7500 :=
by
  sorry

end sum_of_arithmetic_progressions_l518_518986


namespace pi_times_positive_difference_of_volumes_l518_518009

noncomputable def cylinder_volume (radius height : ℝ) : ℝ :=
  π * radius^2 * height

def amy_height : ℝ := 9
def amy_circumference : ℝ := 7
def amy_radius : ℝ := amy_circumference / (2 * π)
def amy_volume : ℝ := cylinder_volume amy_radius amy_height

def belinda_height : ℝ := 10
def belinda_circumference : ℝ := 5
def belinda_radius : ℝ := belinda_circumference / (2 * π)
def belinda_volume : ℝ := cylinder_volume belinda_radius belinda_height

def volume_difference : ℝ := abs (amy_volume - belinda_volume)

theorem pi_times_positive_difference_of_volumes : 
  π * volume_difference = 191 / 4 :=
by
  sorry

end pi_times_positive_difference_of_volumes_l518_518009


namespace max_possible_value_ratio_l518_518679

-- Definitions of the essential geometric elements
noncomputable def triangle (A B C : Type) : Type := sorry -- Placeholder for triangle definition
noncomputable def excircle (A B C : Type) (α : Type) (β : Type) (γ : Type) : Type := sorry -- Placeholder for excircle definition

-- Definitions for points where excircle touches the triangle sides
constants {A B C A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ : Type}
constants (triangle_ABC : triangle A B C)
          (excircle_A : excircle A B C A₁ B₁ C₁)
          (excircle_B : excircle A B C A₂ B₂ C₂)
          (excircle_C : excircle A B C A₃ B₃ C₃)

-- Definition to compute the sum of the perimeters of given triangles
noncomputable def perimeter_sum (A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ : Type) : ℝ := sorry -- Placeholder for perimeter sum definition

-- Definition to compute the circumradius R
noncomputable def circumradius (triangle_ABC : triangle A B C) : ℝ := sorry -- Placeholder for circumradius definition

-- Maximum value of the ratio k
noncomputable def k_max : ℝ := 9 + (9 * real.sqrt 3) / 2

-- The proof problem statement
theorem max_possible_value_ratio :
  let R := circumradius triangle_ABC
  let k := (perimeter_sum A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃) / R
  in k ≤ k_max :=
sorry

end max_possible_value_ratio_l518_518679


namespace find_k_for_polynomial_division_l518_518407

theorem find_k_for_polynomial_division :
  ∃ (k : ℚ), (∀ x : ℚ, (3 * x ^ 4 + k * x ^ 3 + 5 * x ^ 2 - 15 * x + 55)
  % (3 * x + 5) = 20) :=
begin
  let k := - 101 / 30,
  use k,
  sorry
end

end find_k_for_polynomial_division_l518_518407


namespace g_f_neg4_eq_12_l518_518988

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 - 8

-- Define the assumption that g(f(4)) = 12
axiom g : ℝ → ℝ
axiom g_f4 : g (f 4) = 12

-- The theorem to prove that g(f(-4)) = 12
theorem g_f_neg4_eq_12 : g (f (-4)) = 12 :=
sorry -- proof placeholder

end g_f_neg4_eq_12_l518_518988


namespace value_at_2015_l518_518885

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Conditions of the problem
axiom even_f : ∀ x, f(-x) = f(x)
axiom periodicity_f : ∀ x, f(x + 2) * f(x) = 1
axiom positivity_f : ∀ x, f(x) > 0

-- The theorem to prove
theorem value_at_2015 : f 2015 = 1 :=
by sorry

end value_at_2015_l518_518885


namespace exists_irrational_term_l518_518197

noncomputable def sequence (a1 : ℝ) : ℕ → ℝ
| 0     := a1
| (n+1) := real.sqrt (sequence n + 1)

theorem exists_irrational_term (a1 : ℝ) (h : 0 < a1) :
  ∃ n : ℕ, irrational (sequence a1 n) :=
sorry

end exists_irrational_term_l518_518197


namespace find_m_l518_518102

noncomputable def f (a x : ℝ) : ℝ := log a x

theorem find_m (a : ℝ) (h₁ : f a 2 = 4) (h₂ : ∀ m, f a m = 16) : 16 = 16 :=
by
  sorry

end find_m_l518_518102


namespace ellipse_total_distance_l518_518789

theorem ellipse_total_distance (a b : ℝ) (h₁ : a = 3) (h₂ : b = sqrt 5) :
  let ellipse_eqn := ∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1 in
  2 * (2 * a) = 12 :=
by
  intro ellipse_eqn
  simp [ellipse_eqn, h₁, h₂]
  sorry

end ellipse_total_distance_l518_518789


namespace maximum_reduced_price_l518_518688

theorem maximum_reduced_price (marked_price : ℝ) (cost_price : ℝ) (reduced_price : ℝ) 
    (h1 : marked_price = 240) 
    (h2 : marked_price = cost_price * 1.6) 
    (h3 : reduced_price - cost_price ≥ cost_price * 0.1) : 
    reduced_price ≤ 165 :=
sorry

end maximum_reduced_price_l518_518688


namespace min_alterations_for_unique_sums_l518_518026

def matrix_4x4 : Matrix (Fin 4) (Fin 4) ℕ :=
  !![
    [5, 11, 3, 6],
    [12, 2, 7, 4],
    [8, 6, 9, 2],
    [10, 3, 1, 11]
  ]

theorem min_alterations_for_unique_sums (A : Matrix (Fin 4) (Fin 4) ℕ) (h : A = matrix_4x4) :
  (∃ S : Fin 4 × Fin 4 → ℕ, S ≠ A ∧ (
    let row_sums := (fun i => ∑ j, S i j),
        col_sums := (fun j => ∑ i, S i j),
        diag_sum := ∑ i, S i i in
    pairwise ((≠) on row_sums)
    ∧ pairwise ((≠) on col_sums)
    ∧ (diag_sum ≠ row_sums)
    ∧ (diag_sum ≠ col_sums)
    ∧ (∑ (i, j) in (Finset.univ.product Finset.univ), (S i j != A i j).to_nat = 4)
  )), 
sorry

end min_alterations_for_unique_sums_l518_518026


namespace four_digit_number_8802_l518_518404

theorem four_digit_number_8802 (x : ℕ) (a b c d : ℕ) (h1 : 1000 ≤ x ∧ x ≤ 9999)
  (h2 : x = 1000 * a + 100 * b + 10 * c + d)
  (h3 : a ≠ 0)  -- since a 4-digit number cannot start with 0
  (h4 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) : 
  x + 8802 = 1099 + 8802 :=
by
  sorry

end four_digit_number_8802_l518_518404


namespace num_correct_propositions_l518_518196

-- Definitions for the conditions.
def m_perpendicular_n (m n : Line) : Prop := ∃ p, p ∈ m ∧ p ∈ n ∧ p.perpendicular
def m_perpendicular_alpha (m : Line) (α : Plane) : Prop := ∃ p, p ∈ m ∧ p ∈ α ∧ p.perpendicular
def n_parallel_alpha (n : Line) (α : Plane) : Prop := ∃ q r, q ∈ n ∧ r ∈ α ∧ r.parallel q
def alpha_parallel_beta (α β : Plane) : Prop := ∀ p q, p ∈ α ∧ q ∈ β → p.parallel q
def m_in_alpha (m : Line) (α : Plane) : Prop := ∀ p, p ∈ m → p ∈ α
def m_parallel_beta (m : Line) (β : Plane) : Prop := ∃ p q, p ∈ m ∧ q ∈ β ∧ q.parallel p
def proposition1 (m n : Line) (α β : Plane) : Prop := m_perpendicular_n m n ∧ m_perpendicular_alpha m α ∧ n_parallel_alpha n β → α.parallel β
def proposition2 (m n : Line) (α : Plane) : Prop := m_perpendicular_alpha m α ∧ n_parallel_alpha n α → m_perpendicular_n m n
def proposition3 (m : Line) (α β : Plane) : Prop := alpha_parallel_beta α β ∧ m_in_alpha m α → m_parallel_beta m β

-- The proof question.
theorem num_correct_propositions (m n : Line) (α β : Plane) : 
    (¬ proposition1 m n α β) ∧
    (proposition2 m n α) ∧
    (proposition3 m α β) →
    (2 = 2) :=
begin
  sorry
end

end num_correct_propositions_l518_518196


namespace arrange_numbers_l518_518792

theorem arrange_numbers :
  ∃ (A B C D E F : ℕ), 
    (A ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    (B ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    (C ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    (D ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    (E ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    (F ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    {A, B, C, D, E, F} = {1, 2, 3, 4, 5, 6} ∧
    A + D + E = 15 ∧
    7 + C + E = 15 ∧
    9 + C + A = 15 ∧
    A + 8 + F = 15 ∧
    7 + D + F = 15 ∧
    9 + D + B = 15 ∧
    A = 4 ∧
    B = 1 ∧
    C = 2 ∧
    D = 5 ∧
    E = 6 ∧
    F = 3 := 
by sorry

end arrange_numbers_l518_518792


namespace more_cost_effective_scheme_a_l518_518590

/-- Regular ticket price per person -/
def ticket_price (a : ℝ) := a

/-- Total cost under Scheme A for a group of size x -/
def scheme_a_cost (a : ℝ) (x : ℕ) : ℝ :=
  if x < 2 then x * a else (0.75 * (x - 2) * a + 2 * a)

/-- Total cost under Scheme B for a group of size x -/
def scheme_b_cost (a : ℝ) (x : ℕ) : ℝ :=
  0.80 * x * a

/-- Proof statement for the problem -/
theorem more_cost_effective_scheme_a (a : ℝ) : 
  let x := 18 in
  scheme_a_cost a x < scheme_b_cost a x ∧ 
  scheme_b_cost a x - scheme_a_cost a x = 0.4 * a :=
by
  sorry

end more_cost_effective_scheme_a_l518_518590


namespace find_true_discount_l518_518674

-- Definitions of conditions
def bankersGain : ℝ := 8.4
def rateOfInterest : ℝ := 12
def time : ℝ := 1

-- Define the true discount based on the conditions
def trueDiscount (BG R T : ℝ) : ℝ := (BG * 100) / (R * T)

-- The theorem statement
theorem find_true_discount :
  trueDiscount bankersGain rateOfInterest time = 70 := 
  sorry

end find_true_discount_l518_518674


namespace diagonals_in_23_sided_polygon_one_vertex_unconnected_l518_518491

theorem diagonals_in_23_sided_polygon_one_vertex_unconnected :
  ∀ (n : ℕ), n = 23 → (∀ v : Fin n, (n * (n - 3) / 2) - (n - 3) = 210) :=
by
  intro n
  assume h : n = 23
  intro v
  have h1 : n * (n - 3) / 2 = 23 * 20 / 2 := by rw [h]
  have h2 : 23 * 20 / 2 - 20 = 210 := by norm_num
  rw [h1]
  exact h2

end diagonals_in_23_sided_polygon_one_vertex_unconnected_l518_518491


namespace line_circle_intersection_l518_518750

theorem line_circle_intersection (b : ℝ) : (|b| < 2) →
  (∃ x y : ℝ, y = sqrt 3 * x + b ∧ x^2 + y^2 - 4 * y = 0) :=
sorry

end line_circle_intersection_l518_518750


namespace find_exponent_l518_518418

theorem find_exponent (y : ℝ) (exponent : ℝ) :
  (12^1 * 6^exponent / 432 = y) → (y = 36) → (exponent = 3) :=
by 
  intros h₁ h₂ 
  sorry

end find_exponent_l518_518418


namespace exists_cubic_polynomial_with_cubed_roots_l518_518129

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

-- Statement that we need to prove
theorem exists_cubic_polynomial_with_cubed_roots :
  ∃ (b c d : ℝ), ∀ (x : ℝ),
  (f x = 0) → (x^3 = y → x^3^3 + b * x^3^2 + c * x^3 + d = 0) :=
sorry

end exists_cubic_polynomial_with_cubed_roots_l518_518129


namespace find_side_b_l518_518183

theorem find_side_b
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b * Real.sin A = 3 * c * Real.sin B)
  (h2 : a = 3)
  (h3 : Real.cos B = 2 / 3) :
  b = Real.sqrt 6 :=
by
  sorry

end find_side_b_l518_518183


namespace solve_inequalities_l518_518392

theorem solve_inequalities (x : ℝ) :
  (3 * x^2 - x > 4) ∧ (x < 3) ↔ (1 < x ∧ x < 3) := 
by 
  sorry

end solve_inequalities_l518_518392


namespace proof_sin_945_l518_518024

noncomputable def sin_945_eq_neg_sqrt2_div_2 : Prop :=
  sin (945 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem proof_sin_945 : sin_945_eq_neg_sqrt2_div_2 :=
  by 
  sorry

end proof_sin_945_l518_518024


namespace find_x_l518_518873

theorem find_x (x : ℕ) (h : x + 1 = 6) : x = 5 :=
sorry

end find_x_l518_518873


namespace DS_eq_2PL_l518_518335

-- Definitions and assumptions: O, P, Q, R are points such that 
-- there is a circle centered at O inscribed in ∠QPR touching PR at L.
-- A tangent to the circle, parallel to PO, intersects PQ at S and LP at D.

variables {O P Q R L S D : Type} [metric_space O] [normed_group O]
          (circle : O → Type) [circ_centered_at : ∀ (x : O), x ∈ center circle O] 
          (O_PQ : line PQ (P)) : 
          (O_PR : line PR (P)) :
          (O_L : L ∈ O_PR) :
          (PQ_S : S ∈ PQ) :
          (LP_D : D ∈ LP) :
          (parallel_tangent : (parallel PO (tangent circle)) : Prop)

theorem DS_eq_2PL (h₁ : touches circle PR L)
                  (h₂ : parallel PO (tangent circle))
                  (h₃ : tangent circle PQ S)
                  (h₄ : tangent circle LP D) :
  DS = 2 * PL :=
begin
  sorry
end

end DS_eq_2PL_l518_518335


namespace similar_triangle_shortest_side_l518_518776

theorem similar_triangle_shortest_side {a b c : ℝ} (h₁ : a = 24) (h₂ : b = 32) (h₃ : c = 80) :
  let hypotenuse₁ := Real.sqrt (a ^ 2 + b ^ 2)
  let scale_factor := c / hypotenuse₁
  let shortest_side₂ := scale_factor * a
  shortest_side₂ = 48 :=
by
  sorry

end similar_triangle_shortest_side_l518_518776


namespace range_x2_plus_y2_l518_518947

open Real

theorem range_x2_plus_y2 (x y : ℝ) (h : x^2 - 2 * x * y + 5 * y^2 = 4) :
  3 - sqrt 5 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 3 + sqrt 5 :=
sorry

end range_x2_plus_y2_l518_518947


namespace mutually_orthogonal_vectors_l518_518198

noncomputable def vector_addition (a b c : Vector ℝ) (p q r : ℝ) : Vector ℝ :=
  p * (a.cross b) + q * (b.cross c) + r * (c.cross a)

noncomputable def scalar_triple_product (a b c : Vector ℝ) : ℝ :=
  a.dot (b.cross c)

variable (a b c : Vector ℝ)
variable (p q r : ℝ)

theorem mutually_orthogonal_vectors (h1 : a.dot b = 0)
                                    (h2 : b.dot c = 0)
                                    (h3 : c.dot a = 0)
                                    (ha : a.norm = 2)
                                    (hb : b.norm = 3)
                                    (hc : c.norm = 4)
                                    (h4 : a = vector_addition a b c p q r)
                                    (h5 : scalar_triple_product a b c = 24) :
  p + q + r = 1 / 6 :=
sorry

end mutually_orthogonal_vectors_l518_518198


namespace isosceles_triangle_area_48_l518_518246

noncomputable def isosceles_triangle_area (b h s : ℝ) : ℝ :=
  (1 / 2) * (2 * b) * h

theorem isosceles_triangle_area_48 :
  ∀ (b s : ℝ),
  b ^ 2 + 8 ^ 2 = s ^ 2 ∧ s + b = 16 →
  isosceles_triangle_area b 8 s = 48 :=
by
  intros b s h
  unfold isosceles_triangle_area
  sorry

end isosceles_triangle_area_48_l518_518246


namespace projection_range_AE_AC_l518_518879

variable (A B C D E : Type) [InnerProductSpace ℝ A]
variables (AB AC : ℝ) (angleBAC : ℝ)
variable (x : ℝ) (h1 : AB = 2) (h2 : AC = 3) (h3 : angleBAC = real.Angle.pi / 3)
variable (h4 : x ∈ Ioo 0 1) (h5 : ∥D - C∥ = 2 * ∥E - B∥)
variable (h6 : E = x • D + (1 - x) • B)
variable [Nontrivial A]

theorem projection_range_AE_AC (A B C D E : A) (x : ℝ) (h1 : AB = 2) (h2 : AC = 3) (h3 : angleBAC = real.Angle.pi / 3)
  (h4 : x ∈ Ioo 0 1) (h5 : ∥D - C∥ = 2 * ∥E - B∥) (h6 : E = x • D + (1 - x) • B) : 
  ∃ L U, L = 1 ∧ U = 7 ∧ ∀ (p : ℝ), p = ((E - A) • (C - A)) / ∥C - A∥ → L < p ∧ p < U := sorry

end projection_range_AE_AC_l518_518879


namespace find_y_value_l518_518933

theorem find_y_value (x y : ℝ) 
    (h1 : x^2 + 3 * x + 6 = y - 2) 
    (h2 : x = -5) : 
    y = 18 := 
  by 
  sorry

end find_y_value_l518_518933


namespace infinite_solutions_b_value_l518_518838

-- Given condition for the equation to hold
def equation_condition (x b : ℤ) : Prop :=
  4 * (3 * x - b) = 3 * (4 * x + 16)

-- The statement we need to prove: b = -12
theorem infinite_solutions_b_value :
  (∀ x : ℤ, equation_condition x b) → b = -12 :=
sorry

end infinite_solutions_b_value_l518_518838


namespace island_liars_l518_518648

theorem island_liars (n : ℕ) (h₁ : n = 450) (h₂ : ∀ (i : ℕ), i < 450 → 
  ∃ (a : bool),  (if a then (i + 1) % 450 else (i + 2) % 450) = "liar"):
    (n = 150 ∨ n = 450) :=
sorry

end island_liars_l518_518648


namespace p_is_necessary_but_not_sufficient_for_q_l518_518104

variables {f : ℝ → ℝ} {x0 : ℝ}

-- Assumptions based on the conditions
namespace problem

def derivative_exists : Prop := ∃ f', ∀ x, deriv f x = f'

def p : Prop := deriv f x0 = 0
def q : Prop := ∀ x, (f x > f x0 ∨ f x < f x0)

theorem p_is_necessary_but_not_sufficient_for_q
  (h_deriv_exists : derivative_exists)
  (h_p : p) 
  : (p → q) ∧ (¬ (q → p)) := 
sorry

end problem

end p_is_necessary_but_not_sufficient_for_q_l518_518104


namespace question1_question2_l518_518475

open Classical

variable {R : Type*} [LinearOrderedField R]

def f (x a : R) := x^2 - 2 * a * x + 5

-- Define the statement equivalent to Question 1
theorem question1 (h1 : ∀ x ∈ [[1 : R], 2], f x (1 : R) = 1):
  ∀ (a : R), (1: R) < a → (∀ x : R, (x ∈ [[1: R], a]) → f x a = 1) → a = 2 :=
sorry

-- Define the statement equivalent to Question 2
theorem question2 (h1 : ∀ x : R, x < (2 : R) → f x (3 : R) = f x (2 : R))
  (h2 : ∀ x : R, ((x ∈ [[1: R], 2])) → f x 1 ≤ 0) :
  ∃ a : R, 1 < a ∧ 3 ≤ a :=
sorry

end question1_question2_l518_518475


namespace log_101600_l518_518742

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_101600 (h : log_base_10 102 = 0.3010) : log_base_10 101600 = 2.3010 :=
by
  sorry

end log_101600_l518_518742


namespace balls_in_boxes_l518_518535

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l518_518535


namespace lambda_range_for_obtuse_angle_l518_518424

theorem lambda_range_for_obtuse_angle {λ : ℝ} (h : (1 : ℝ) * λ + (-1 : ℝ) * (1 : ℝ) < 0) : λ < 1 :=
sorry

end lambda_range_for_obtuse_angle_l518_518424


namespace coefficient_of_c_l518_518786

theorem coefficient_of_c (f c : ℝ) (h₁ : f = (9/5) * c + 32)
                         (h₂ : f + 25 = (9/5) * (c + 13.88888888888889) + 32) :
  (5/9) = (9/5) := sorry

end coefficient_of_c_l518_518786


namespace marble_prism_weight_l518_518973

def height : ℝ := 8
def base_side_length : ℝ := 2
def density : ℝ := 2700

def volume (h : ℝ) (s : ℝ) : ℝ := s * s * h
def weight (v : ℝ) (d : ℝ) : ℝ := v * d

theorem marble_prism_weight : weight (volume height base_side_length) density = 86400 := by
  sorry

end marble_prism_weight_l518_518973


namespace fishing_ratio_l518_518800

variables (B C : ℝ)
variable (brian_per_trip : ℝ)
variable (chris_per_trip : ℝ)

-- Given conditions
def conditions : Prop :=
  C = 10 ∧
  brian_per_trip = 400 ∧
  chris_per_trip = 400 * (5 / 3) ∧
  B * brian_per_trip + 10 * chris_per_trip = 13600

-- The ratio of the number of times Brian goes fishing to the number of times Chris goes fishing
def ratio_correct : Prop :=
  B / C = 26 / 15

theorem fishing_ratio (h : conditions B C brian_per_trip chris_per_trip) : ratio_correct B C :=
by
  sorry

end fishing_ratio_l518_518800


namespace balanced_ternary_8_digits_nonnegative_count_l518_518141

theorem balanced_ternary_8_digits_nonnegative_count :
  let a_i : Fin 8 → ℤ := λ i, {x : ℤ | x = -1 ∨ x = 0 ∨ x = 1}
  in (Finset.sum (Finset.range 8) (λ i, a_i i * 3^i)).nonneg.card = 3281 :=
sorry

end balanced_ternary_8_digits_nonnegative_count_l518_518141


namespace quadratic_inequality_solution_range_l518_518062

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, a*x^2 + 2*a*x - 4 < 0) ↔ -4 < a ∧ a < 0 := 
by
  sorry

end quadratic_inequality_solution_range_l518_518062


namespace problem_a_problem_b_l518_518322

theorem problem_a (p : ℕ) (hp : Nat.Prime p) : 
  (∃ x : ℕ, (7^(p-1) - 1) = p * x^2) ↔ p = 3 := 
by
  sorry

theorem problem_b (p : ℕ) (hp : Nat.Prime p) : 
  ¬ ∃ x : ℕ, (11^(p-1) - 1) = p * x^2 := 
by
  sorry

end problem_a_problem_b_l518_518322


namespace num_ways_to_put_5_balls_into_4_boxes_l518_518525

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l518_518525


namespace domain_of_function_l518_518254

theorem domain_of_function : 
  (∀ y : ℝ, y = (1 / (sqrt (6 - x - x^2))) → x > -3 ∧ x < 2) :=
sorry

end domain_of_function_l518_518254


namespace lines_perpendicular_to_same_plane_are_parallel_l518_518448

variables {Point Line Plane : Type*}
variables [MetricSpace Point] [LinearOrder Line]

def line_parallel_to_plane (a : Line) (M : Plane) : Prop := sorry -- Define the formal condition
def line_perpendicular_to_plane (a : Line) (M : Plane) : Prop := sorry -- Define the formal condition
def lines_parallel (a b : Line) : Prop := sorry -- Define the formal condition

theorem lines_perpendicular_to_same_plane_are_parallel 
  (a b : Line) (M : Plane) 
  (h₁ : line_perpendicular_to_plane a M) 
  (h₂ : line_perpendicular_to_plane b M) : 
  lines_parallel a b :=
sorry

end lines_perpendicular_to_same_plane_are_parallel_l518_518448


namespace lattice_points_count_l518_518494

theorem lattice_points_count :
  let x1 := 5
      y1 := 23
      x2 := 60
      y2 := 353
      delta_x := x2 - x1 -- 55
      delta_y := y2 - y1 -- 330
      gcd_delta := Nat.gcd delta_y delta_x -- greatest common divisor
  in gcd_delta + 1 = 56 :=
by
  sorry

end lattice_points_count_l518_518494


namespace find_a_l518_518939

noncomputable def f (a x : ℝ) : ℝ := Real.log (Real.sqrt (1 + a * x ^ 2) - x)

theorem find_a (a : ℝ) :
  (∀ (x : ℝ), f a (-x) = -f a x) ↔ a = 1 :=
by
  sorry

end find_a_l518_518939


namespace odd_function_a_minus_b_l518_518847

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_a_minus_b
  (a b : ℝ)
  (h : is_odd_function (λ x => 2 * x ^ 3 + a * x ^ 2 + b - 1)) :
  a - b = -1 :=
sorry

end odd_function_a_minus_b_l518_518847


namespace part1_part2_l518_518467

-- Definitions for the ellipse and points
def ellipse_equation (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1

def point_A : ℝ × ℝ := (1, 1/2)
def point_B : ℝ × ℝ := (1, 2)

-- Midpoint condition
def midpoint (M N A : ℝ × ℝ) : Prop := (M.1 + N.1) / 2 = A.1 ∧ (M.2 + N.2) / 2 = A.2

-- Line intersecting ellipse
def line_intersects_ellipse (f : ℝ → ℝ) : Prop := 
  ∃ (x1 x2 : ℝ), ellipse_equation x1 (f x1) ∧ ellipse_equation x2 (f x2)

-- Definitions for the slope of the line and the maximum area
def slope (M N : ℝ × ℝ) : ℝ := (M.2 - N.2) / (M.1 - N.1)
def triangle_area (B P Q : ℝ × ℝ) : ℝ := abs (B.1 * (P.2 - Q.2) + P.1 * (Q.2 - B.2) + Q.1 * (B.2 - P.2)) / 2

-- Theorems
theorem part1 (M N : ℝ × ℝ) (hM : ellipse_equation M.1 M.2) (hN : ellipse_equation N.1 N.2) 
    (hA : midpoint M N point_A) : slope M N = -1 :=
sorry

theorem part2 (t : ℝ) (ht : t ≠ 0) (hineq : 0 < t^2 ∧ t^2 < 9) : 
    ∃ (P Q : ℝ × ℝ), line_intersects_ellipse (λ x, 2 * x + t) ∧ triangle_area point_B P Q = sqrt 2 / 2 :=
sorry

end part1_part2_l518_518467


namespace max_min_not_roots_F_l518_518487

-- Definitions
variable {F G : ℝ → ℝ}
variable hF : ∃ (a b c : ℝ), distinct_roots F [a, b, c] ∧ cubic_polynomial F
variable hG : ∃ (d e f : ℝ), distinct_roots G [d, e, f] ∧ cubic_polynomial G
variable hDistinct : F ≠ G
variable hRootsListed : ∀ x, (F x = 0 ∨ G x = 0 ∨ F x = G x) → x ∈ {a, b, c, d, e, f, g, h} -- and the set has exactly 8 elements

-- Statement to prove that the largest and smallest numbers listed among roots cannot both be roots of \( F(x) \).
theorem max_min_not_roots_F :
  ¬ (∀ x, x ∈ {a, b} → F x = 0) :=
sorry

end max_min_not_roots_F_l518_518487


namespace reeya_fifth_score_l518_518225

theorem reeya_fifth_score
  (s1 s2 s3 s4 avg: ℝ)
  (h1: s1 = 65)
  (h2: s2 = 67)
  (h3: s3 = 76)
  (h4: s4 = 82)
  (h_avg: avg = 75) :
  ∃ s5, s1 + s2 + s3 + s4 + s5 = 5 * avg ∧ s5 = 85 :=
by
  use 85
  sorry

end reeya_fifth_score_l518_518225


namespace total_number_of_flowers_l518_518281

noncomputable def number_of_roses : Nat := 34
noncomputable def number_of_lilies : Nat := number_of_roses + 13
noncomputable def number_of_tulips : Nat := number_of_lilies - 23

theorem total_number_of_flowers : 
  number_of_roses + number_of_lilies + number_of_tulips = 105 := 
by 
  -- We can use calculations from previous steps without proving them.
  have h1 : number_of_roses = 34 := rfl
  have h2 : number_of_lilies = 47 := by rw [number_of_lilies, h1]; refl
  have h3 : number_of_tulips = 24 := by rw [number_of_tulips, h2]; refl
  rw [h1, h2, h3]
  rfl

end total_number_of_flowers_l518_518281


namespace angle_between_tangents_l518_518191

-- Definition of the regular hexagon inscribed in a circle
def regular_hexagon (O : Point) (A B C D E F : Point) (r : ℝ) : Prop :=
  circle O r ∧
  (rotation_angle A B C D E F = 120) ∧
  (seg_length A B = r) ∧ (seg_length B C = r) ∧ (seg_length C D = r) ∧ 
  (seg_length D E = r) ∧ (seg_length E F = r) ∧ (seg_length F A = r)

-- Tangents and points definition
def tangent_points (O A D P Q : Point) : Prop :=
  tangent_circle O A P ∧ tangent_circle O D Q ∧ minor_arc_tangency PQ EF

-- Main theorem statement
theorem angle_between_tangents (O A B C D E F P Q : Point) (r : ℝ) :
  regular_hexagon O A B C D E F r →
  tangent_points O A D P Q →
  angle_between PB QC = 30 := sorry

end angle_between_tangents_l518_518191


namespace find_b_l518_518277

variable (a b : ℝ) (k : ℝ)
axiom h1 : a^2 * sqrt b = k
axiom h2 : a = 3
axiom h3 : b = 36
axiom h4 : a * b = 90

-- Goal: prove b = 28.
theorem find_b : b = 28 :=
by sorry

end find_b_l518_518277


namespace coeff_x_term_l518_518816

variables (x : ℚ)

-- Theorem to prove the coefficient of the x term in the expansion of (x + 1/x)² * (1 + x)⁵ is 20
theorem coeff_x_term : coeff (expand_polynomial (x + 1/x)^2 * (1 + x)^5) (degree_single 1) = 20 := 
by
  -- other hints and steps here
  sorry

end coeff_x_term_l518_518816


namespace arithmetic_sequence_sum_l518_518960

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S_9 : ℝ)
  (h1 : a 1 + a 4 + a 7 = 15)
  (h2 : a 3 + a 6 + a 9 = 3)
  (h_arith : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) :
  S_9 = 27 :=
by
  sorry

end arithmetic_sequence_sum_l518_518960


namespace library_books_l518_518693

theorem library_books (N x y : ℕ) (h1 : x = N / 17) (h2 : y = x + 2000)
    (h3 : y = (N - 2 * 2000) / 15 + (14 * (N - 2000) / 17)): 
  N = 544000 := 
sorry

end library_books_l518_518693


namespace cyclic_quadrilateral_l518_518284

theorem cyclic_quadrilateral 
  (P A B K L M N : Point)
  (c₁ c₂ : Circle)
  (h₁ : P ∈ c₁ ∧ P ∈ c₂)
  (h₂ : P ∈ segment A B)
  (h₃ : A ∈ c₁ ∧ B ∈ c₁)
  (h₄ : A ∈ c₂ ∧ B ∈ c₂)
  (h₅ : segment K M ∈ c₁)
  (h₆ : segment L N ∈ c₂)
  (h₇ : P = intersection (line_segment A B) (line_segment L N))
  (h₈ : PA * PB = PL * PN)
  (h₉ : PA * PB = PM * PK) : Cyclic (quadrilateral K L M N) := 
by 
  sorry

end cyclic_quadrilateral_l518_518284


namespace monotonicity_intervals_local_minimum_value_range_l518_518891

noncomputable def f (k x : ℝ) : ℝ := k * x^3 - 3 * x^2 + 1

theorem monotonicity_intervals (k : ℝ) (h : k ≥ 0) :
  if k = 0 then 
    (∃ I1 I2 : set ℝ, I1 = set.Iic 0 ∧ I2 = set.Ici 0 ∧ ∀ x ∈ I1, ∀ y ∈ I1, x ≤ y → f k x ≤ f k y ∧ ∀ x ∈ I2, ∀ y ∈ I2, x ≤ y → f k x ≥ f k y)
  else 
    (∃ I1 I2 I3 : set ℝ, I1 = set.Iic 0 ∧ I2 = set.Icc 0 (2/k) ∧ I3 = set.Ici (2/k) ∧ 
    (∀ x ∈ I1, ∀ y ∈ I1, x ≤ y → f k x ≤ f k y) ∧ 
    (∀ x ∈ I2, ∀ y ∈ I2, x ≤ y → f k x ≥ f k y) ∧ 
    (∀ x ∈ I3, ∀ y ∈ I3, x ≤ y → f k x ≤ f k y)) :=
sorry

theorem local_minimum_value_range (k : ℝ) (h : k > 0) :
  (∀ x : ℝ, x = (2 / k) → (f k x > 0 ↔ k > 2)) :=
sorry

end monotonicity_intervals_local_minimum_value_range_l518_518891


namespace sum_of_ratios_lt_l518_518877

variable {n : ℕ}
variable (a : Fin n → ℝ)

theorem sum_of_ratios_lt (h₁ : ∀ i : Fin n, 1 < a i) 
                         (h₂ : ∀ i : Fin (n-1), abs (a i.succ - a i) < 1) :
  (Finset.univ.sum (λ i : Fin (n-1), (a i) / (a i.succ))) + (a (Fin.last n) / (a 0)) < 2 * n - 1 :=
sorry

end sum_of_ratios_lt_l518_518877


namespace excircle_radius_is_6_l518_518441

noncomputable def triangleIncircleExcircle
  (ABC : Triangle)
  (C1 B1 A1 : Point)
  (r : ℝ)
  (D E G : Point)
  (CE : ℝ)
  (CB1 : ℝ) : ℝ :=
  if h : (ABC.hasIncircle inscribed r C1 B1 A1) ∧ (E.isOnExcircle ABC r D G) ∧ CE = 6 ∧ CB1 = 1 then 
    6 
  else
    0

theorem excircle_radius_is_6
  (ABC : Triangle) :
  ∃ (C1 B1 A1 D E G : Point) (r CE CB1 : ℝ),
  r = 1 ∧
  CE = 6 ∧
  CB1 = 1 ∧
  (ABC.hasIncircle inscribed r C1 B1 A1) ∧
  (E.isOnExcircle ABC r D G) ∧
  (E.excircleRadius ABC r = 6) :=
begin
  sorry,
end

end excircle_radius_is_6_l518_518441


namespace geometric_sequence_ratio_28_l518_518168

noncomputable def geometric_sequence_sum_ratio (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ) :=
  S 6 / S 3 = 28

theorem geometric_sequence_ratio_28 (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h_GS : ∀ n, S n = a1 * (1 - q^n) / (1 - q)) 
  (h_increasing : ∀ n m, n < m → a1 * q^n < a1 * q^m) 
  (h_mean : 2 * 6 * a1 * q^6 = a1 * q^7 + a1 * q^8) : 
  geometric_sequence_sum_ratio a1 q S := 
by {
  -- Proof should be completed here
  sorry
}

end geometric_sequence_ratio_28_l518_518168


namespace min_weighings_to_separate_l518_518005

-- Definition of the problem parameters
def n : ℕ := 2000
def n_half : ℕ := n / 2
def mass_aluminum : ℝ := 10.0
def mass_duralumin : ℝ := 9.9

-- There are 1000 aluminum balls and 1000 duralumin balls.
axiom aluminum_balls : Fin n_half → ℝ 
axiom duralumin_balls : Fin n_half → ℝ 

-- The aluminum balls each have a mass of 10 grams.
axiom aluminum_mass : ∀ i : Fin n_half, aluminum_balls i = mass_aluminum
-- The duralumin balls each have a mass of 9.9 grams.
axiom duralumin_mass : ∀ i : Fin n_half, duralumin_balls i = mass_duralumin

-- Given 2000 balls such that 1000 are aluminum and 1000 are duralumin,
-- Proves that it is possible to separate the balls into two piles of equal number
-- but different masses using one weighing.
theorem min_weighings_to_separate : {A1 : Fin n_half → ℝ // ∀ i, A1 i = mass_aluminum} → 
                                    {A2 : Fin n_half → ℝ // ∀ i, A2 i = mass_aluminum} → 
                                    {D1 : Fin n_half → ℝ // ∀ i, D1 i = mass_duralumin} → 
                                    {D2 : Fin n_half → ℝ // ∀ i, D2 i = mass_duralumin} → 
                                    (A1, D1, A2, D2) → 
                                    ∃ L R : Fin (n / 3) → ℝ, 
                                       (∑ i, L i) ≠ (∑ i, R i) :=
sorry

end min_weighings_to_separate_l518_518005


namespace find_d_l518_518613

def f (x : ℝ) (c : ℝ) : ℝ := 4 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 2
def composed (x : ℝ) (c : ℝ) (d : ℝ) : ℝ := 12 * x + d

theorem find_d (c d : ℝ) : (∀ x, f (g x c) c = composed x d) → d = 11 :=
by
  intros h
  -- proof goes here
  sorry

end find_d_l518_518613


namespace count_x_satisfying_ggx_eq_4_l518_518683

noncomputable def g (x : ℝ) : ℝ :=
  if x = -2 then 4 else
  if x = 2 then 4 else
  if x = 4 then 4 else
  if x = -1 then -2 else
  if x = 0 then 2 else
  if x = 1 then 4 else
  0 -- Assuming g(x) = 0 for others for simplicity

theorem count_x_satisfying_ggx_eq_4 :
  let solutions := {x : ℝ | g (g x) = 4} in
  solutions = {x : ℝ | x = -1 ∨ x = 0 ∨ x = 1} ∧ solutions.card = 3 := 
sorry

end count_x_satisfying_ggx_eq_4_l518_518683


namespace volume_of_prism_l518_518247

theorem volume_of_prism (x y z : ℝ) (h1 : x * y = 100) (h2 : z = 10) (h3 : x * z = 50) (h4 : y * z = 40):
  x * y * z = 200 :=
by
  sorry

end volume_of_prism_l518_518247


namespace cyclic_quad_tangent_line_incenter_l518_518596

variable {A B C D : Type}
variable [IsCyclicQuadrilateral A B C D]

def is_parallel (l1 l2 : Type) : Prop := sorry
def is_tangent_to_inscribed_circle (l : Type) (Δ : Triangle) : Prop := sorry
def passes_through_incenter (l : Type) (Δ : Triangle) : Prop := sorry

theorem cyclic_quad_tangent_line_incenter 
    (l : Type) (BD : Type)
    (h_parallel : is_parallel l BD)
    (h_tangent_ABC : is_tangent_to_inscribed_circle l (Triangle.mk A B C))
    (h_tangent_CDA : is_tangent_to_inscribed_circle l (Triangle.mk C D A)) :
  passes_through_incenter l (Triangle.mk B C D) ∨ passes_through_incenter l (Triangle.mk D A B) :=
sorry

end cyclic_quad_tangent_line_incenter_l518_518596


namespace area_PQRS_eq_area_ABCD_l518_518851

variable {Point : Type*}
variable (M A B C D P Q R S : Point)
variable (mid_AB mid_BC mid_CD mid_DA : Point)
variable (area : Point → Point → Point → Point → ℝ)

-- Definitions for the midpoints and symmetry conditions
def midpoint (X Y : Point) : Point := sorry
def symmetric_point (O P : Point) : Point := sorry

-- Conditions
axiom convex_quadrilateral (A B C D : Point) : Prop := sorry
axiom M_inside_ABCD (M A B C D : Point) : Prop := sorry
axiom symmetric_points (M mid_AB mid_BC mid_CD mid_DA P Q R S : Point) : Prop :=
  symmetric_point mid_AB M = P ∧
  symmetric_point mid_BC M = Q ∧
  symmetric_point mid_CD M = R ∧
  symmetric_point mid_DA M = S

-- Theorem to prove that the area of PQRS is equal to the area of ABCD
theorem area_PQRS_eq_area_ABCD
  (h_convex : convex_quadrilateral A B C D)
  (h_M_inside : M_inside_ABCD M A B C D)
  (h_symm : symmetric_points M (midpoint A B) (midpoint B C) (midpoint C D) (midpoint D A) P Q R S):
  area P Q R S = area A B C D :=
sorry

end area_PQRS_eq_area_ABCD_l518_518851


namespace seven_divides_n_l518_518979

theorem seven_divides_n (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 3^n + 4^n) : 7 ∣ n :=
sorry

end seven_divides_n_l518_518979


namespace arithmetic_sequence_a10_l518_518956

noncomputable theory

open_locale classical

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_a10 (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_a2 : a 2 = 2) (h_a3 : a 3 = 4) :
  a 10 = 18 :=
sorry

end arithmetic_sequence_a10_l518_518956


namespace sin_angle_AOB_l518_518961

-- Definitions for the problem
variables {A B C D O : Type}
variables [is_point A] [is_point B] [is_point C] [is_point D] [is_point O]
variables [is_convex_quadrilateral A B C D] -- A definition indicating ABCD is a convex quadrilateral
variables (angle_ABC : angle A B C = 60)
variables (angle_BAD : angle B A D = 90)
variables (angle_BCD : angle B C D = 90)
variables (AB : length A B = 2)
variables (CD : length C D = 1)
variables (intersect_diagonals : intersection (diagonal A C) (diagonal B D) = O)

-- Lean 4 statement we need to prove
theorem sin_angle_AOB :
  sin (angle A O B) = (15 + 6 * sqrt 3) / 26 :=
sorry

end sin_angle_AOB_l518_518961


namespace find_premium_l518_518344

-- Definitions based on conditions
variables (N P : ℕ)

-- Conditions
def condition1 : Prop := N * (100 + P) = 14400
def condition2 : Prop := N * 6 = 720

-- Theorem stating the proof problem
theorem find_premium (h1 : condition1 N P) (h2 : condition2 N P) : P = 20 :=
by
  -- Sorry to skip the actual proof
  sorry

end find_premium_l518_518344


namespace product_of_integers_l518_518264

theorem product_of_integers (x y : ℤ) (h1 : Int.gcd x y = 5) (h2 : Int.lcm x y = 60) : x * y = 300 :=
by
  sorry

end product_of_integers_l518_518264


namespace person_speed_in_kmph_l518_518739

-- Define the distance in meters
def distance_meters : ℕ := 300

-- Define the time in minutes
def time_minutes : ℕ := 4

-- Function to convert distance from meters to kilometers
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000

-- Function to convert time from minutes to hours
def minutes_to_hours (min : ℕ) : ℚ := min / 60

-- Define the expected speed in km/h
def expected_speed : ℚ := 4.5

-- Proof statement
theorem person_speed_in_kmph : 
  meters_to_kilometers distance_meters / minutes_to_hours time_minutes = expected_speed :=
by 
  -- This is where the steps to verify the theorem would be located, currently omitted for the sake of the statement.
  sorry

end person_speed_in_kmph_l518_518739


namespace simple_interest_years_l518_518835

theorem simple_interest_years (SI P : ℝ) (R : ℝ) (T : ℝ) 
  (hSI : SI = 200) 
  (hP : P = 1600) 
  (hR : R = 3.125) : 
  T = 4 :=
by 
  sorry

end simple_interest_years_l518_518835


namespace sufficient_but_not_necessary_l518_518081

noncomputable def problem_statement (a : ℝ) : Prop :=
(a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2)

theorem sufficient_but_not_necessary (a : ℝ) : problem_statement a := 
sorry

end sufficient_but_not_necessary_l518_518081


namespace cost_of_each_shirt_l518_518237

-- Setting up the constants based on the conditions
constant S : ℝ
constant num_shirts : ℝ := 2
constant trousers_cost : ℝ := 63
constant num_additional_items : ℝ := 4
constant additional_item_cost : ℝ := 40
constant total_budget : ℝ := 260

-- Define the equivalence of the total cost to the given budget
theorem cost_of_each_shirt : S = 18.50 :=
by
  have total_cost := (num_shirts * S) + trousers_cost + (num_additional_items * additional_item_cost)
  have budget_constraint : total_cost = total_budget
  rw [←budget_constraint] at *
  sorry

end cost_of_each_shirt_l518_518237


namespace root_in_interval_3_4_l518_518732

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 11

theorem root_in_interval_3_4 : ∃ (c : ℝ), 3 < c ∧ c < 4 ∧ f c = 0 :=
by
  let a := 3
  let b := 4
  have fa : f a < 0 := by sorry
  have fb : f b > 0 := by sorry
  exact exists_Ioo_zero_of_Icc fa fb

end root_in_interval_3_4_l518_518732


namespace canoe_no_paddle_time_l518_518067

-- All conditions needed for the problem
variables {S v v_r : ℝ}
variables (time_pa time_pb : ℝ)

-- Condition that time taken from A to B is 3 times the time taken from B to A
def condition1 : Prop := time_pa = 3 * time_pb

-- Define time taken from A to B (downstream) and B to A (upstream)
def time_pa_def : time_pa = S / (v + v_r) := sorry
def time_pb_def : time_pb = S / (v - v_r) := sorry

-- Main theorem stating the problem to prove
theorem canoe_no_paddle_time :
  condition1 →
  ∃ (t_no_paddle : ℝ), t_no_paddle = 3 * time_pb :=
begin
  intro h1,
  sorry
end

end canoe_no_paddle_time_l518_518067


namespace balls_in_boxes_l518_518536

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l518_518536


namespace cups_of_rice_morning_l518_518228

variable (cupsMorning : Nat) -- Number of cups of rice Robbie eats in the morning
variable (cupsAfternoon : Nat := 2) -- Cups of rice in the afternoon
variable (cupsEvening : Nat := 5) -- Cups of rice in the evening
variable (fatPerCup : Nat := 10) -- Fat in grams per cup of rice
variable (weeklyFatIntake : Nat := 700) -- Total fat in grams per week

theorem cups_of_rice_morning :
  ((cupsMorning + cupsAfternoon + cupsEvening) * fatPerCup) = (weeklyFatIntake / 7) → cupsMorning = 3 :=
  by
    sorry

end cups_of_rice_morning_l518_518228


namespace interior_angle_solution_l518_518290

noncomputable def interior_angle_of_inscribed_triangle (x : ℝ) (h : (2 * x + 40) + (x + 80) + (3 * x - 50) = 360) : ℝ :=
  (1 / 2) * (x + 80)

theorem interior_angle_solution (x : ℝ) (h : (2 * x + 40) + (x + 80) + (3 * x - 50) = 360) :
  interior_angle_of_inscribed_triangle x h = 64 :=
sorry

end interior_angle_solution_l518_518290


namespace find_B_area_cond1_area_cond2_area_cond3_l518_518578

variable (A B C : ℝ) -- Angles of the triangle
variable (a b c : ℝ) -- Sides opposite A, B, C respectively

-- Given conditions
axiom (h1 : a = 2 * Real.sqrt 3)
axiom (h2 : a^2 + c^2 - Real.sqrt 3 * a * c = b^2)

-- Condition 1
axiom (cond1 : b = 3)
-- Condition 2
axiom (cond2 : Real.cos A = 4 / 5)
-- Condition 3
axiom (cond3 : a + b + c = 4 + 2 * Real.sqrt 3)

-- Part (I): Angle B
theorem find_B (a b c : ℝ) (h1 : a = 2 * Real.sqrt 3) (h2 : a^2 + c^2 - Real.sqrt 3 * a * c = b^2) : B = Real.pi / 6 := sorry

-- Part (II): Area of the triangle under different conditions
theorem area_cond1 (a b c : ℝ) (h1 : a = 2 * Real.sqrt 3) (h2 : a^2 + c^2 - Real.sqrt 3 * a * c = b^2) (cond1 : b = 3) :
  Area_of_triangle = sorry := sorry -- Area calculation given b = 3

theorem area_cond2 (a b c : ℝ) (h1 : a = 2 * Real.sqrt 3) (h2 : a^2 + c^2 - Real.sqrt 3 * a * c = b^2) (cond2 : Real.cos A = 4 / 5) :
  Area_of_triangle = (3 * Real.sqrt 3 + 4) / 2 := sorry -- Area calculation given cos A = 4 / 5

theorem area_cond3 (a b c : ℝ) (h1 : a = 2 * Real.sqrt 3) (h2 : a^2 + c^2 - Real.sqrt 3 * a * c = b^2) (cond3 : a + b + c = 4 + 2 * Real.sqrt 3) :
  Area_of_triangle = Real.sqrt 3 := sorry -- Area calculation given perimeter is 4 + 2 * Real.sqrt 3

end find_B_area_cond1_area_cond2_area_cond3_l518_518578


namespace ways_to_distribute_balls_in_boxes_l518_518518

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l518_518518


namespace triangle_area_l518_518557

theorem triangle_area (A B C : ℝ) (a b c : ℝ) (S : ℝ) (hA : A = π / 3) (ha : a = sqrt 3) (hc : c = 1) :
  S = (sqrt 3 / 2) :=
sorry

end triangle_area_l518_518557


namespace units_digit_G1000_l518_518812

def Gn (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_G1000 : (Gn 1000) % 10 = 2 :=
by sorry

end units_digit_G1000_l518_518812


namespace ellipse_eccentricity_and_existence_of_line_l518_518468

noncomputable def ellipse_equation_t (x y t : ℝ) : Prop :=
  x^2 / t + y^2 = 1

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  x^2 = 2 * real.sqrt 2 * y

noncomputable def perpendicular_tangent_condition (x1 x2 : ℝ) : Prop :=
  (real.sqrt 2 / 2 * x1) * (real.sqrt 2 / 2 * x2) = -1

theorem ellipse_eccentricity_and_existence_of_line :
  ∃ (t : ℝ) (e : ℝ), ellipse_equation_t x y t → e = real.sqrt 3 / 2 ∧ t = 4 ∧ 
  ∃ (k : ℝ), (k = - real.sqrt 2 / 4) ∧ 
  ∀ (l : ℝ → ℝ → Prop), parabola_equation x y → 
  (perpendicular_tangent_condition x1 x2 ∧ x1 * x2 = -2) →
  (l y (x - 2), parabola_equation x y) → 
  y = - real.sqrt 2 / 4 * (x - 2) :=
sorry

end ellipse_eccentricity_and_existence_of_line_l518_518468


namespace five_cubic_km_to_cubic_meters_l518_518915

theorem five_cubic_km_to_cubic_meters (km_to_m : 1 = 1000) : 
  5 * (1000 ^ 3) = 5000000000 := 
by
  sorry

end five_cubic_km_to_cubic_meters_l518_518915


namespace exists_replacement_of_cos_with_sin_l518_518217

theorem exists_replacement_of_cos_with_sin (k : ℕ) (h : k > 10) :
  ∃ f1 : ℝ → ℝ, (∀ x : ℝ, ∃ n : ℕ, n ≤ k ∧ 
    f1 x = (∏ i in finset.Icc 1 k, if i = n then sin (2^i * x) else cos (2^i * x)) ∧ 
    |f1 x| ≤ 3 / 2^(k + 1)) :=
by
  sorry

end exists_replacement_of_cos_with_sin_l518_518217


namespace math_problem_l518_518823

theorem math_problem (x : ℝ) : 
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1/2 ∧ (x^2 + x^3 - 2 * x^4) / (x + x^2 - 2 * x^3) ≥ -1 ↔ 
  x ∈ Set.Icc (-1 : ℝ) (-1/2) ∪ Set.Ioc (-1/2 : ℝ) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1 := 
by 
  sorry

end math_problem_l518_518823


namespace range_of_a_l518_518872

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def strictly_increasing_on_nonnegative (f : ℝ → ℝ) : Prop :=
∀ x1 x2, (0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (m n : ℝ) (h_even : is_even_function f)
  (h_strict : strictly_increasing_on_nonnegative f)
  (h_m : m = 1/2) (h_f : ∀ x, m ≤ x ∧ x ≤ n → f (a * x + 1) ≤ f 2) :
  a ≤ 2 :=
sorry

end range_of_a_l518_518872


namespace possible_values_a_l518_518814

-- Define the problem statement
theorem possible_values_a :
  (∃ a b c : ℤ, ∀ x : ℝ, (x - a) * (x - 5) + 3 = (x + b) * (x + c)) → (a = 1 ∨ a = 9) :=
by 
  -- Variable declaration and theorem body will be placed here
  sorry

end possible_values_a_l518_518814


namespace domain_shift_l518_518883

theorem domain_shift (f : ℝ → ℝ) (h : ∀ (x : ℝ), (-2 < x ∧ x < 2) → (f (x + 2) = f x)) :
  ∀ (y : ℝ), (3 < y ∧ y < 7) ↔ (y - 3 < 4 ∧ y - 3 > -2) :=
by
  sorry

end domain_shift_l518_518883


namespace BO_perpendicular_to_DE_l518_518570

-- Given a pentagon ABCDE with diagonals AD and CE intersecting at O, 
-- which is the center of the inscribed circle.

variable (A B C D E O : Type*) [IsPentagon A B C D E]
variable (AD CE : Line) (circ_center : CenterOfInscribedCircle A B C D E O)
variable (intersect : Intersect AD CE = O)

-- Prove that segment BO is perpendicular to side DE.

theorem BO_perpendicular_to_DE :
  ∀ (BO DE : Segment), (BO.IntersectCenter DE.IntersectCenter O) → Perpendicular BO DE :=
sorry

end BO_perpendicular_to_DE_l518_518570


namespace find_triples_l518_518600

theorem find_triples (k : ℕ) (hk : 0 < k) :
  ∃ (a b c : ℕ), 
    (0 < a ∧ 0 < b ∧ 0 < c) ∧
    (a + b + c = 3 * k + 1) ∧ 
    (a * b + b * c + c * a = 3 * k^2 + 2 * k) ∧ 
    (a = k + 1 ∧ b = k ∧ c = k) :=
by
  sorry

end find_triples_l518_518600


namespace at_least_12_married_l518_518562

-- Definitions of conditions
def total_men : ℕ := 100
def men_with_TV : ℕ := 75
def men_with_radio : ℕ := 85
def men_with_AC : ℕ := 70
def men_with_TV_radio_AC_and_married : ℕ := 12

-- Statement to prove
theorem at_least_12_married : ∃ m : ℕ, m ≥ 12 :=
by
  use men_with_TV_radio_AC_and_married
  show men_with_TV_radio_AC_and_married ≥ 12
  from Nat.le_refl 12

end at_least_12_married_l518_518562


namespace sum_of_extreme_values_l518_518614

noncomputable def f (x : ℝ) : ℝ :=
  abs (x - 3) + abs (x - 5) - abs (2 * x - 8)

theorem sum_of_extreme_values :
  let domain := {x : ℝ | 3 ≤ x ∧ x ≤ 10} in
  (let max_val := @Real.sup (f '' domain) (image_nonempty_of_mem ⟨3, by simp⟩)
   let min_val := @Real.inf (f '' domain) (image_nonempty_of_mem ⟨3, by simp⟩))
  (max_val + min_val = 2) :=
by
  let domain := {x : ℝ | 3 ≤ x ∧ x ≤ 10};
  let max_val := @Real.sup (f '' domain) (image_nonempty_of_mem ⟨3, by simp⟩);
  let min_val := @Real.inf (f '' domain) (image_nonempty_of_mem ⟨3, by simp⟩);
  sorry

end sum_of_extreme_values_l518_518614


namespace spring_square_l518_518174

/-- Different characters represent different digits, and same characters represent same digits. -/
def Spring (n : ℕ) : Prop :=
  n = 3201

theorem spring_square :
  ∃ n : ℕ, Spring n ∧ n * n = 10246401 :=
by
  exists 3201
  constructor
  · exact rfl
  · exact rfl

end spring_square_l518_518174


namespace gcd_sum_l518_518011

theorem gcd_sum : ∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (3 * n + 5) n}, d = 6 :=
by
  sorry

end gcd_sum_l518_518011


namespace circumcircle_of_triangle_tangent_to_gamma_l518_518132

-- Given definitions
variable (A B C I D E F : Type)
variable (Γ : Type)

-- Circumcircle and incenter
def triangle_with_circumcircle_incenter (triangle : Type) :=
  ∃ (circumcircle : Type) (incenter : Type), true

-- Line intersecting with segments AI, BI, CI
def line_intersects_ai_bi_ci (line : Type) (AI BI CI : Type) (D E F : Type) :=
  ∃ (inter_pointD : Type) (inter_pointE : Type) (inter_pointF : Type), true

-- Perpendicular bisectors forming a triangle
def perp_bisectors_form_triangle (x y z : Type) :=
  ∃ (perp_bisector_x : Type) (perp_bisector_y : Type) (perp_bisector_z : Type), true

-- The final proof problem statement
theorem circumcircle_of_triangle_tangent_to_gamma :
  ∀ (ABC : Type) (Γ : Type) (I : Type) (l : Type)
    (D : Type) (E : Type) (F : Type)
    (x : Type) (y : Type) (z : Type),
    triangle_with_circumcircle_incenter ABC →
    line_intersects_ai_bi_ci l ABC D E F →
    perp_bisectors_form_triangle x y z →
    ∃ (Γ' : Type), tangent Γ Γ' := sorry

end circumcircle_of_triangle_tangent_to_gamma_l518_518132


namespace at_least_one_weight_greater_than_35_l518_518039

theorem at_least_one_weight_greater_than_35
  (weights : Fin 11 → ℕ)
  (distinct : Function.Injective weights)
  (heavy_side_more_weights : ∀ (s₁ s₂ : Finset (Fin 11)), 
                              s₁.card ≠ s₂.card → (∑ x in s₁, weights x) > (∑ x in s₂, weights x)) :
  ∃ w : Fin 11, weights w > 35 :=
sorry

end at_least_one_weight_greater_than_35_l518_518039


namespace total_workers_l518_518181

-- Define the conditions for each workshop
def avg_salary_A := 8000
def avg_salary_B := 9000
def avg_salary_C := 10000

def tech_salary_A := 20000
def rest_salary_A := 6000
def tech_count_A := 7

def tech_salary_B := 25000
def rest_salary_B := 5000
def tech_count_B := 10

def tech_salary_C := 30000
def rest_salary_C := 7000
def tech_count_C := 15

-- Prove the total number of workers in all three workshops is 214
theorem total_workers :
    let A_nt := (8000 * (tech_count_A + A_nt) - 20000 * tech_count_A) / (6000 - 8000),
        B_nt := (9000 * (tech_count_B + B_nt) - 25000 * tech_count_B) / (5000 - 9000),
        C_nt := (10000 * (tech_count_C + C_nt) - 30000 * tech_count_C) / (7000 - 10000),
        A := tech_count_A + A_nt,
        B := tech_count_B + B_nt,
        C := tech_count_C + C_nt
    in A + B + C = 214 := by
        sorry

end total_workers_l518_518181


namespace read_as_three_million_thirty_six_thousand_l518_518245

def distance_from_guangzhou_to_shenyang : ℕ := 3036000

theorem read_as_three_million_thirty_six_thousand :
  (read_number 3036000) = "three million thirty-six thousand" :=
sorry

end read_as_three_million_thirty_six_thousand_l518_518245


namespace function_behavior_on_interval_l518_518682

def behavior_of_function (y : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ x ∈ interval, (∃ x1 ∈ interval, ∃ x2 ∈ interval, 
  (x1 < x2 ∧ ∀ x ∈ Ioo(2,3), y' x < 0 ∧ ∀ x ∈ Ioo(3,4), y' x > 0))

theorem function_behavior_on_interval : 
  behavior_of_function (λ x : ℝ, x^2 - 6*x + 10) (Ioo 2 4) := 
sorry

end function_behavior_on_interval_l518_518682


namespace vector_parallel_implies_values_l518_518607

variables {Real : Type*} [LinearOrderedField Real]

def vector_parallel (a b : Real × Real × Real) : Prop :=
  ∃ λ : Real, a = (λ * b.1, λ * b.2, λ * b.3)

noncomputable def m_val (λ : Real) : Real := 3 * λ
noncomputable def n_val (λ : Real) : Real := 2 / λ

theorem vector_parallel_implies_values (λ : Real)
  (hλ : λ = 1 / 4) 
  (ha : (m_val λ, -1, 2) = (m, -1, 2))
  (hb : (3, -4, n_val λ) = (3, -4, n))
  (h_parallel : vector_parallel (m, -1, 2) (3, -4, n)) :
  m = 3 / 4 ∧ n = 8 :=
by
  sorry

end vector_parallel_implies_values_l518_518607


namespace fractions_are_integers_l518_518620

theorem fractions_are_integers (a b c : ℤ) (h : (ab / c + ac / b + bc / a) ∈ ℤ) : 
  (ab / c) ∈ ℤ ∧ (ac / b) ∈ ℤ ∧ (bc / a) ∈ ℤ := 
sorry

end fractions_are_integers_l518_518620


namespace part1_part2_l518_518476

-- Part (1)
def f_2 (x : ℝ) : ℝ := |x + 2| + |x + 1 / 2|

theorem part1 (x : ℝ) : (f_2 x > 3) ↔ (x < -11 / 4 ∨ x > 1 / 4) := sorry

-- Part (2)
def f (x a : ℝ) : ℝ := |x + a| + |x + 1 / a|

theorem part2 (m a : ℝ) (ha : 0 < a) : f m a + f (-1 / m) a ≥ 4 := sorry

end part1_part2_l518_518476


namespace coefficient_a2_in_expansion_l518_518632

theorem coefficient_a2_in_expansion (n : ℕ) : 
  let f := (1 + x + x^2) in 
  ∃ a : ℚ → ℚ, a = (λ n : ℕ, (1 + n) * n / 2) → 
  (f ^ n).coeff 2 = a n :=
by
  sorry

end coefficient_a2_in_expansion_l518_518632


namespace tom_saves_money_l518_518287

-- Defining the cost of a normal doctor's visit
def normal_doctor_cost : ℕ := 200

-- Defining the discount percentage for the discount clinic
def discount_percentage : ℕ := 70

-- Defining the cost reduction based on the discount percentage
def discount_amount (cost percentage : ℕ) : ℕ := (percentage * cost) / 100

-- Defining the cost of a visit to the discount clinic
def discount_clinic_cost (normal_cost discount_amount : ℕ ) : ℕ := normal_cost - discount_amount

-- Defining the number of visits to the discount clinic
def discount_clinic_visits : ℕ := 2

-- Defining the total cost for the discount clinic visits
def total_discount_clinic_cost (visit_cost visits : ℕ) : ℕ := visits * visit_cost

-- The final cost savings calculation
def cost_savings (normal_cost total_discount_cost : ℕ) : ℕ := normal_cost - total_discount_cost

-- Proving the amount Tom saves by going to the discount clinic
theorem tom_saves_money : cost_savings normal_doctor_cost (total_discount_clinic_cost (discount_clinic_cost normal_doctor_cost (discount_amount normal_doctor_cost discount_percentage)) discount_clinic_visits) = 80 :=
by
  sorry

end tom_saves_money_l518_518287


namespace problem_statement_l518_518849

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - Real.pi * x

theorem problem_statement (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) : 
  ((deriv f x < 0) ∧ (f x < 0)) :=
by
  sorry

end problem_statement_l518_518849


namespace find_certain_number_l518_518930

theorem find_certain_number (h1 : 2994 / 14.5 = 173) (h2 : ∃ x, x / 1.45 = 17.3) : ∃ x, x = 25.085 :=
by
  -- Proof goes here
  sorry

end find_certain_number_l518_518930


namespace five_digit_numbers_divisible_l518_518616

theorem five_digit_numbers_divisible : 
  let n_range := (List.range (99999 - 10000 + 1)).map (λ x => x + 10000)
  let qs := List.filter (λ q => q % 2 = 0) (List.range (1999 - 200 + 1)).map (λ x => x + 200)
  let valid_q_r_pairs := List.foldl (λ acc q => acc + (List.range 50).count (λ r => (q - r) % 7 = 0)) 0 qs
  valid_q_r_pairs = 7200 := by
sorry

end five_digit_numbers_divisible_l518_518616


namespace max_n_for_factorable_polynomial_l518_518411

theorem max_n_for_factorable_polynomial :
  ∃ A B : ℤ, AB = 144 ∧ (A + 6 * B = 865) :=
begin
  sorry
end

end max_n_for_factorable_polynomial_l518_518411


namespace distribute_balls_in_boxes_l518_518512

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l518_518512


namespace parallelogram_angle_l518_518743

theorem parallelogram_angle (a b : ℝ) (h1 : a + b = 180) (h2 : a = b + 50) : b = 65 :=
by
  -- Proof would go here, but we're adding a placeholder
  sorry

end parallelogram_angle_l518_518743


namespace part1_a_half_part2_expression_part2_turning_points_l518_518989

noncomputable def f (a x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ a then (1/a) * x else (1/(1 - a)) * (1 - x)

def turning_point (a x0 : ℝ) : Prop :=
f a (f a x0) = x0 ∧ f a x0 ≠ x0

-- Part 1: When a = 1/2
theorem part1_a_half : 
  f (1/2) (f (1/2) (4/5)) = 4/5 := sorry

def calc_f_f_x (a x : ℝ) : ℝ :=
if a < x ∧ x < a^2 - a + 1 then
  (1/(1 - a)) * (1 - (1/(1 - a)) * (1 - x))
else
  (1/(a * (1 - a))) * (1 - x)

-- Part 2: Analytical expression for f(f(x))
theorem part2_expression (a : ℝ) (h : 0 < a ∧ a < 1) (x : ℝ) (hx : a < x ∧ x ≤ 1) :
  f (f a x) = calc_f_f_x a x := sorry

-- The turning points of f(x)
theorem part2_turning_points (a : ℝ) (h : 0 < a ∧ a < 1) :
  turning_point a (1 / (2 - a)) ∧ turning_point a (1 / (1 + a - a^2)) := sorry

end part1_a_half_part2_expression_part2_turning_points_l518_518989


namespace solve_for_x_l518_518384

def custom_mul (a b : ℤ) : ℤ := a * b + a + b

theorem solve_for_x (x : ℤ) :
  custom_mul 3 (3 * x - 1) = 27 → x = 7 / 3 := by
sorry

end solve_for_x_l518_518384


namespace square_diagonal_sqrt_24_l518_518858

theorem square_diagonal_sqrt_24 :
  ∀ (a b : ℕ), 
  a = 6 → b = 4 →
  let triangle_area := (1 / 2 : ℝ) * a * b,
      square_area := triangle_area,
      square_side := Real.sqrt square_area,
      square_diagonal := Real.sqrt 2 * square_side in
  square_diagonal = Real.sqrt 24 :=
by
  intros a b ha hb
  simp [ha, hb, Real.sqrt, triangle_area, square_area, square_side, square_diagonal]
  sorry

end square_diagonal_sqrt_24_l518_518858


namespace intersecting_line_l518_518465

theorem intersecting_line (m : ℝ) :
  ∃ b : ℝ, (b = 2) ∨ (b = -2) ∧ 
  (∀ x y : ℝ, 4 * x^2 + 5 * y^2 - 8 * m * x - 20 * m * y + 24 * m^2 - 20 = 0 → 
  (forall x y : ℝ, x = m + √5 * cos θ → y = 2 * (m + √5 * cos θ) + b)) :=
sorry

end intersecting_line_l518_518465


namespace ball_box_problem_l518_518502

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l518_518502


namespace intersection_x_sum_l518_518489

theorem intersection_x_sum :
  ∃ x : ℤ, (0 ≤ x ∧ x < 17) ∧ (4 * x + 3 ≡ 13 * x + 14 [ZMOD 17]) ∧ x = 5 :=
by
  sorry

end intersection_x_sum_l518_518489


namespace points_on_single_circle_l518_518092

theorem points_on_single_circle (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ i j : Fin n, ∃ f : ℝ × ℝ → ℝ × ℝ, (∀ p, f p ≠ p) ∧ f (points i) = points j ∧ 
        (∀ k : Fin n, ∃ p, points k = f p)) :
  ∃ (O : ℝ × ℝ) (r : ℝ), ∀ i : Fin n, dist (points i) O = r := sorry

end points_on_single_circle_l518_518092


namespace greatest_least_S_T_l518_518826

theorem greatest_least_S_T (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) (triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  4 ≤ (a + b + c)^2 / (b * c) ∧ (a + b + c)^2 / (b * c) ≤ 9 :=
by sorry

end greatest_least_S_T_l518_518826


namespace days_to_use_up_one_bag_l518_518373

def rice_kg : ℕ := 11410
def bags : ℕ := 3260
def rice_per_day : ℚ := 0.25
def rice_per_bag : ℚ := rice_kg / bags

theorem days_to_use_up_one_bag : (rice_per_bag / rice_per_day) = 14 := by
  sorry

end days_to_use_up_one_bag_l518_518373


namespace part1_part2_part3_l518_518088

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (3 + k) * x + 3
noncomputable def g (k : ℝ) (m : ℝ) (x : ℝ) : ℝ := f k x - m * x

theorem part1 (k : ℝ) (hk : k ≠ 0) (hf2 : f k 2 = 3) :
  f k x = -x^2 + 2x + 3 :=
sorry

theorem part2 (m : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → deriv (λ x, -x^2 + (2 - m) * x + 3) x ≥ 0 ∨ deriv (λ x, -x^2 + (2 - m) * x + 3) x ≤ 0) ↔
  (m ≤ -2 ∨ m ≥ 6) :=
sorry

theorem part3 (k : ℝ) :
  (∃ x ∈ set.Icc (-1 : ℝ) 4, ∀ y ∈ set.Icc (-1 : ℝ) 4, f k x ≤ f k y) ↔ (k = -1 ∨ k = -9) :=
sorry

end part1_part2_part3_l518_518088


namespace accuracy_of_estimate_increases_with_sample_size_l518_518310

theorem accuracy_of_estimate_increases_with_sample_size
  (frequency_distribution : Type) 
  (population_density_curve : Type) 
  (sample_size : ℕ) 
  (accuracy_of_estimate : frequency_distribution → ℕ → Prop):
  (∀ fd pd s, s > 0 → accuracy_of_estimate fd s) → 
  (∀ (s1 s2 : ℕ), s1 < s2 → 
    ∀ (fd: frequency_distribution), 
      accuracy_of_estimate fd s1 → 
      accuracy_of_estimate fd s2) := 
sorry

end accuracy_of_estimate_increases_with_sample_size_l518_518310


namespace general_formula_sequence_l518_518856

variable {a : ℕ → ℝ}

-- Definitions and assumptions
def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n - 2 * a (n + 1) + a (n + 2) = 0

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 2 = 4

-- The proof problem
theorem general_formula_sequence (a : ℕ → ℝ)
  (h1 : recurrence_relation a)
  (h2 : initial_conditions a) :
  ∀ n : ℕ, a n = 2 * n :=

sorry

end general_formula_sequence_l518_518856


namespace not_basic_event_l518_518703

def ball := ℕ -- defining ball as a natural number

def is_red (b : ball) : Prop := b < 2   -- defining red balls
def is_white (b : ball) : Prop := b ≥ 2 ∧ b < 4  -- defining white balls
def is_black (b : ball) : Prop := b ≥ 4 ∧ b < 6  -- defining black balls

def basic_events : set (set ball) := { 
  {0, 1},                      -- 2 red balls
  {2, 3},                      -- 2 white balls
  {4, 5},                      -- 2 black balls
  {0, 2}, {0, 3}, {1, 2}, {1, 3}, -- 1 red and 1 white ball
  {0, 4}, {0, 5}, {1, 4}, {1, 5}, -- 1 red and 1 black ball
  {2, 4}, {2, 5}, {3, 4}, {3, 5}  -- 1 white and 1 black ball
}

def at_least_one_red : set ball := {0, 1, 2, 3, 4, 5} -- incorrect event "at least 1 red ball"

theorem not_basic_event : at_least_one_red ∉ basic_events := 
by sorry

end not_basic_event_l518_518703


namespace distribute_balls_in_boxes_l518_518511

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l518_518511


namespace number_of_triangles_l518_518175

theorem number_of_triangles (A B C D E F G H I J K : Type) :
  ∃ S T : set Type, 
    S = {A, B, C, D, E, F, G} ∧ T = {H, I, J, K} ∧
    (∑ (x ∈ S) ,(∑ (y ∈ S \ {x}), (T \ {A}).card) + 
      ∑ (y ∈ T),(S.card) + 
      ∑ (x ∈ T),(∑ (y ∈ T \ {x}), (S \ {A}).card) = 120 :=
by
  sorry

end number_of_triangles_l518_518175


namespace find_distance_to_bus_stand_l518_518318

-- Definitions based on the given conditions
def walk_time_slow (D : ℝ) : ℝ := D / 3
def bus_time_slow (D : ℝ) : ℝ := walk_time_slow D - 0.2

def walk_time_fast (D : ℝ) : ℝ := D / 6
def bus_time_fast (D : ℝ) : ℝ := walk_time_fast D + 1 / 6

-- The proof statement
theorem find_distance_to_bus_stand (D : ℝ) : 
  bus_time_slow D = bus_time_fast D → 
  D = 2.2 :=
begin
  sorry
end

end find_distance_to_bus_stand_l518_518318


namespace number_of_integer_roots_of_6561_l518_518422

theorem number_of_integer_roots_of_6561 : 
  let n := 6561
  let a := 8
  ∃ (k : ℕ) (h : k ∈ {1, 2, 4, 8}),  
  (∃ (m : ℕ), 3^a^1/k.toNat = m)
   = 3 := sorry

end number_of_integer_roots_of_6561_l518_518422


namespace h_increasing_l518_518735

def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

noncomputable def h : ℝ → ℝ := λ x, -2 / (x + 1)

theorem h_increasing : is_increasing_on h {x | x ∈ set.Iio 0 \ {-1}} :=
begin
  sorry
end

end h_increasing_l518_518735


namespace probability_passing_through_C_without_D_l518_518669

/--
The adjacent map is part of a city; the small rectangles are blocks, and the paths in between 
are streets. Each morning, a student walks from intersection A to intersection B, always 
walking along streets shown, and always going east or south. For variety, at each intersection 
where he has a choice, he chooses with probability 1/2 whether to go east or south. 
Prove that the probability that the student goes through C while avoiding intersection D is 12/21.
-/
theorem probability_passing_through_C_without_D
  (A B C D : × interpolate the coordinates of A, B, C, and D correctly ×) : 
  let total_paths_from_A_to_B := 21
  let paths_through_C_avoiding_D := 12
  let passing_through_C_without_D_probability := paths_through_C_avoiding_D / total_paths_from_A_to_B
  in
  passing_through_C_without_D_probability = 12 / 21 :=
sorry

end probability_passing_through_C_without_D_l518_518669


namespace midpoint_locus_of_segments_on_skew_lines_l518_518295

noncomputable def midpoint_locus_circle_radius (d h : ℝ) (h_d : d > 0) (h_h : h > 0) : ℝ :=
  1 / 2 * real.sqrt (d^2 - h^2)

theorem midpoint_locus_of_segments_on_skew_lines
  (L₁ L₂ : Set Point) 
  (h_perpendicular : ∀ P ∈ L₁, Q ∈ L₂, P ≠ Q -> line_through_points P Q ∩ plane_parallel_to L₁ L₂ = Ø)
  (d h : ℝ) (h_d : d > 0) (h_h : h > 0) :
    ∃ (circle_radius : ℝ), 
      circle_radius = midpoint_locus_circle_radius d h h_d h_h :=
by
  use midpoint_locus_circle_radius d h h_d h_h
  sorry

end midpoint_locus_of_segments_on_skew_lines_l518_518295


namespace focus_of_parabola_l518_518054

theorem focus_of_parabola (a h k : ℝ) (hyp : (∀ x : ℝ, 2*(x - 3)^2 = a*(x - h)^2 + k)) :
  (h, k + 1 / (4 * a)) = (3, 1 / 8) :=
by
  have ha : a = 2 := sorry
  have hh : h = 3 := sorry
  have hk : k = 0 := sorry
  rw [ha, hh, hk, ←mul_assoc],
  exact ⟨rfl, (by field_simp)⟩

end focus_of_parabola_l518_518054


namespace find_a3_l518_518095

variable (a_n : ℕ → ℤ) (a1 a4 a5 : ℤ)
variable (d : ℤ := -2)

-- Conditions
axiom h1 : ∀ n : ℕ, a_n (n + 1) = a_n n + d
axiom h2 : a4 = a1 + 3 * d
axiom h3 : a5 = a1 + 4 * d
axiom h4 : a4 * a4 = a1 * a5

-- Question to prove
theorem find_a3 : (a_n 3) = 5 := by
  sorry

end find_a3_l518_518095


namespace triangle_isosceles_or_right_l518_518558

theorem triangle_isosceles_or_right 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : b^2 * tan A = a^2 * tan B) 
  (h2 : a < b + c) 
  (h3 : b < a + c) 
  (h4 : c < a + b) : 
  (A = B) ∨ (A + B = π / 2) := 
sorry

end triangle_isosceles_or_right_l518_518558


namespace probability_of_pick_l518_518931

def alphabet := Set.univ -- Universe of all letters in the alphabet.

def unique_letters_in_mathematics : Finset Char := 
  {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

noncomputable def P : ℚ :=
  (unique_letters_in_mathematics.card : ℚ) / (alphabet.card : ℚ)

theorem probability_of_pick (h : alphabet.card = 26):
  P = 4 / 13 := by
  sorry

end probability_of_pick_l518_518931


namespace square_of_1017_l518_518025

theorem square_of_1017 : 1017^2 = 1034289 :=
by
  sorry

end square_of_1017_l518_518025


namespace digits_998_to_1000_are_116_l518_518002

noncomputable def reach_digit_sequence (n : ℕ) : ℕ :=
  let one_digit_count := 1 in
  let two_digit_count := 10 in
  let three_digit_count := 100 in
  let four_digit_count := (n - (one_digit_count * 1 + two_digit_count * 2 + three_digit_count * 3)) / 4 in
  let total_digits := one_digit_count * 1 + two_digit_count * 2 + three_digit_count * 3 + four_digit_count * 4 in
  if total_digits >= n then
    let remaining_digits := n - (one_digit_count * 1 + two_digit_count * 2 + three_digit_count * 3) in
    let first_number_with_four_digits := 1000 in
    let number_series_offset := remaining_digits / 4 in
    first_number_with_four_digits + number_series_offset
  else
    0

theorem digits_998_to_1000_are_116 :
  let final_number := reach_digit_sequence 1000,
     digit998 := final_number / 1000 % 10,
     digit999 := final_number / 100 % 10,
     digit1000 := final_number / 10 % 10
  in [digit998, digit999, digit1000] = [1, 1, 6] :=
by
  sorry

end digits_998_to_1000_are_116_l518_518002


namespace component_probability_l518_518711

theorem component_probability (p : ℝ) 
  (h : (1 - p)^3 = 0.001) : 
  p = 0.9 :=
sorry

end component_probability_l518_518711


namespace countThreeDigitMultiplesOf25Not40_l518_518495

def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def isMultipleOf25 (n : ℕ) : Prop := n % 25 = 0
def isMultipleOf40 (n : ℕ) : Prop := n % 40 = 0

theorem countThreeDigitMultiplesOf25Not40 : 
  (∑ n in (Finset.filter (λ n, isThreeDigit n ∧ isMultipleOf25 n ∧ ¬ isMultipleOf40 n) (Finset.range 1000)), 1) = 32 := 
sorry

end countThreeDigitMultiplesOf25Not40_l518_518495


namespace sum_of_gcd_3n_plus_5_n_l518_518014

theorem sum_of_gcd_3n_plus_5_n (n: ℕ) (h : 0 < n):
  ∃ s, s = {d ∈ finset.range 6 | ∃ n, gcd (3*n + 5) n = d}.sum ∧ s = 6 :=
by
  sorry

end sum_of_gcd_3n_plus_5_n_l518_518014


namespace real_part_of_z_l518_518990

-- Definitions based on given conditions
variables {z : ℂ} (ω : ℝ)

-- The hypothesis that z is a complex number and ω = z + 1/z is real
def condition1 : Prop := (z + (1/z : ℂ)).re = ω
def condition2 : Prop := -1 < ω ∧ ω < 2

-- The proof goal
theorem real_part_of_z (h1 : condition1 ω) (h2 : condition2 ω) : -1/2 < z.re ∧ z.re < 1 := by
  sorry

end real_part_of_z_l518_518990


namespace min_num_triangles_with_area_at_most_one_fourth_l518_518564

theorem min_num_triangles_with_area_at_most_one_fourth (A B C D : Point)
  (h_rect : area (rect A B C D) = 1)
  (P Q R S T : Point)
  (h_non_collinear : ∀ (X Y Z : Point), ¬ collinear X Y Z)
  (h_in_rect : ∀ (X : Point), X = P ∨ X = Q ∨ X = R ∨ X = S ∨ X = T → in_rectangle X (rect A B C D)) :
  ∃ (triangles : set (Triangle)), 
    (∀ (T ∈ triangles), area T ≤ 1/4) ∧ (card triangles = 2) :=
sorry

end min_num_triangles_with_area_at_most_one_fourth_l518_518564


namespace perimeter_parallelogram_ADEF_l518_518950

theorem perimeter_parallelogram_ADEF 
  (A B C D E F : Type) [ordered_ring A]
  [euclidean_geometry A]
  {AB AC BC AD DE EF AF : ℝ}
  (hD : ∃ (D E F : A), 
    line_through D E ∧ 
    line_through E F ∧ line_through F D)
  (h_triangle : triangle A B C)
  (h_AB_AC : AB = 26 ∧ AC = 26)
  (h_BC : BC = 24)
  (h_parallel_DE_AC : line_parallel DE AC)
  (h_parallel_EF_AB : line_parallel EF AB) 
  : calculate_perimeter A D E F = 52 :=
sorry

end perimeter_parallelogram_ADEF_l518_518950


namespace remainder_when_dividing_polynomial_by_x_minus_3_l518_518305

noncomputable def P (x : ℤ) : ℤ := 
  2 * x^8 - 3 * x^7 + 4 * x^6 - x^4 + 6 * x^3 - 5 * x^2 + 18 * x - 20

theorem remainder_when_dividing_polynomial_by_x_minus_3 :
  P 3 = 17547 :=
by
  sorry

end remainder_when_dividing_polynomial_by_x_minus_3_l518_518305


namespace lines_perpendicular_l518_518862

/-- Given two lines l1: 3x + 4y + 1 = 0 and l2: 4x - 3y + 2 = 0, 
    prove that the lines are perpendicular. -/
theorem lines_perpendicular :
  ∀ (x y : ℝ), (3 * x + 4 * y + 1 = 0) → (4 * x - 3 * y + 2 = 0) → (- (3 / 4) * (4 / 3) = -1) :=
by
  intro x y h₁ h₂
  sorry

end lines_perpendicular_l518_518862


namespace intersection_of_sets_l518_518903

def M (x : ℝ) : Prop := (x - 2) / (x - 3) < 0
def N (x : ℝ) : Prop := Real.log (x - 2) / Real.log (1 / 2) ≥ 1 

theorem intersection_of_sets : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 2 < x ∧ x ≤ 5 / 2} := by
  sorry

end intersection_of_sets_l518_518903


namespace Apollonian_Circle_Range_l518_518671

def range_of_m := Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 21 / 2)

theorem Apollonian_Circle_Range :
  ∃ P : ℝ × ℝ, ∃ m > 0, ((P.1 - 2) ^ 2 + (P.2 - m) ^ 2 = 1 / 4) ∧ 
            (Real.sqrt ((P.1 + 1) ^ 2 + P.2 ^ 2) = 2 * Real.sqrt ((P.1 - 2) ^ 2 + P.2 ^ 2)) →
            m ∈ range_of_m :=
  sorry

end Apollonian_Circle_Range_l518_518671


namespace g_eq_g_inv_l518_518386

-- Define the function g
def g (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

-- Define the inverse function of g
noncomputable def g_inv (y : ℝ) : ℝ := (5 + Real.sqrt (1 + 8 * y)) / 4 -- simplified to handle the principal value

theorem g_eq_g_inv (x : ℝ) : g x = g_inv x → x = 1 := by
  -- Placeholder for proof
  sorry

end g_eq_g_inv_l518_518386


namespace participants_in_sports_activities_l518_518282

theorem participants_in_sports_activities:
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = 3 ∧
  let a := 10 * x + 6
  let b := 10 * y + 6
  let c := 10 * z + 6
  a + b + c = 48 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a = 6 ∧ b = 16 ∧ c = 26 ∨ a = 6 ∧ b = 26 ∧ c = 16 ∨ a = 16 ∧ b = 6 ∧ c = 26 ∨ a = 16 ∧ b = 26 ∧ c = 6 ∨ a = 26 ∧ b = 6 ∧ c = 16 ∨ a = 26 ∧ b = 16 ∧ c = 6)
  :=
by {
  sorry
}

end participants_in_sports_activities_l518_518282


namespace students_total_l518_518709

def num_girls : ℕ := 11
def num_boys : ℕ := num_girls + 5

theorem students_total : num_girls + num_boys = 27 := by
  sorry

end students_total_l518_518709


namespace evaluate_expression_l518_518042

theorem evaluate_expression : 
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = (137 / 52) :=
by
  -- We need to evaluate from the innermost part to the outermost,
  -- as noted in the problem statement and solution steps.
  sorry

end evaluate_expression_l518_518042


namespace sum_first_2010_terms_l518_518440

section PeriodicSequence

def periodic (T : ℕ) (a : ℕ → ℕ) := ∀ m : ℕ, a (m + T) = a m

noncomputable def x_sequence (a : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then 1 else if n = 2 then a else |x_sequence a (n - 1) - x_sequence a (n - 2)|

theorem sum_first_2010_terms (a : ℝ) (h : a ≠ 0) (n : ℕ) :
  periodic 3 (λ n, x_sequence a n) →
  (∑ i in range 2010, x_sequence a i) = 1340 :=
sorry

end PeriodicSequence

end sum_first_2010_terms_l518_518440


namespace ways_to_distribute_balls_in_boxes_l518_518521

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l518_518521


namespace equilateral_triangle_volume_and_diagonals_l518_518626

-- Define the problem conditions
variables (a : ℝ)
variables (ABC : Type) [equilateral_triangle ABC a]

-- Folded hexagonal configurations with vertices described
variables (D1 D2 E1 E2 F1 F2 : Type)
variables (G H J K L M : Type)

-- Traces folding and conditions for hexagon coincidences
variables [fold_hexagons ABC D1 D2 E1 E2 F1 F2 G H J K L M]
variables (D E F : Type) -- Points after folding where D1==D2, E1==E2, and F1==F2
variables [coincide D1 D2 D] [coincide E1 E2 E] [coincide F1 F2 F]
variables [coplanar G H J K L M]

-- Main theorem statement
theorem equilateral_triangle_volume_and_diagonals : 
  (volume_of_convex_solid ABC a = (23 * real.sqrt 2 * a^3) / 12) ∧ 
  (internal_diagonals_count ABC a = 12) :=
begin
  sorry
end

end equilateral_triangle_volume_and_diagonals_l518_518626


namespace living_space_increase_l518_518958

theorem living_space_increase (a b x : ℝ) (h₁ : a = 10) (h₂ : b = 12.1) : a * (1 + x) ^ 2 = b :=
sorry

end living_space_increase_l518_518958


namespace problem_statement_l518_518998

theorem problem_statement (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + 2 * c) + b / (c + 2 * a) + c / (a + 2 * b) > 1 / 2) :=
by
  sorry

end problem_statement_l518_518998


namespace island_liars_l518_518647

theorem island_liars (n : ℕ) (h₁ : n = 450) (h₂ : ∀ (i : ℕ), i < 450 → 
  ∃ (a : bool),  (if a then (i + 1) % 450 else (i + 2) % 450) = "liar"):
    (n = 150 ∨ n = 450) :=
sorry

end island_liars_l518_518647


namespace find_d_e_l518_518963

variable {a b c d e : ℝ}
variable (ma ma' : ℝ)
variable (r1 r2 r3 d1 d2 : ℝ) -- row sums and diagonal sums
variable (sums_equal : r1 = r2 ∧ r2 = r3 ∧ r3 = d1 ∧ d1 = d2)

-- The conditions based on the magic square properties
def top_row_sum : ℝ := 30 + e + 15
def middle_row_sum : ℝ := 10 + c + d
def bottom_row_sum : ℝ := a + 35 + b

-- Conditions for rows and diagonals having equal sums
axiom all_sums_equal : top_row_sum = r1 ∧ middle_row_sum = r2 ∧ bottom_row_sum = r3
axiom diagonal1_sum : 30 + c + b = d1
axiom diagonal2_sum : 15 + c + a = d2

theorem find_d_e : d + e = 47.5 := by
  sorry

end find_d_e_l518_518963


namespace maximum_figures_per_shelf_l518_518787

theorem maximum_figures_per_shelf
  (figures_shelf_1 : ℕ)
  (figures_shelf_2 : ℕ)
  (figures_shelf_3 : ℕ)
  (additional_shelves : ℕ)
  (max_figures_per_shelf : ℕ)
  (total_figures : ℕ)
  (total_shelves : ℕ)
  (H1 : figures_shelf_1 = 9)
  (H2 : figures_shelf_2 = 14)
  (H3 : figures_shelf_3 = 7)
  (H4 : additional_shelves = 2)
  (H5 : max_figures_per_shelf = 11)
  (H6 : total_figures = figures_shelf_1 + figures_shelf_2 + figures_shelf_3)
  (H7 : total_shelves = 3 + additional_shelves)
  (H8 : ∃ d, d ∈ ({x : ℕ | x ∣ total_figures} ∩ {y : ℕ | y ≤ max_figures_per_shelf}))
  : ∃ d, d ∈ ({x : ℕ | x ∣ total_figures} ∩ {y : ℕ | y ≤ max_figures_per_shelf}) ∧ d = 6 := sorry

end maximum_figures_per_shelf_l518_518787


namespace determine_a_l518_518452

noncomputable def find_a (a : ℝ) : Prop :=
  ∃ (f g : ℝ → ℝ),
  (∀ x, f x = a^x * g x) ∧
  (∀ x, g x ≠ 0) ∧
  (∀ x, (f' x) * (g x) < (f x) * (g' x)) ∧
  (f 1 / g 1 + f (-1) / g (-1) = 5 / 2) ∧
  (a = 1/2)

theorem determine_a : find_a (1/2) :=
sorry

end determine_a_l518_518452


namespace max_value_of_function_l518_518923

theorem max_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 3 / 2) : 
  ∃ x_max : ℝ, (∀ x' : ℝ, 0 < x' → x' < 3 / 2 → x'(3 - 2 * x') ≤ x_max) ∧ x_max = 9 / 8 :=
sorry

end max_value_of_function_l518_518923


namespace company_match_percentage_l518_518490

theorem company_match_percentage (total_contribution : ℝ) (holly_contribution_per_paycheck : ℝ) (total_paychecks : ℕ) (total_contribution_one_year : ℝ) : 
  let holly_contribution := holly_contribution_per_paycheck * total_paychecks
  let company_contribution := total_contribution_one_year - holly_contribution
  (company_contribution / holly_contribution) * 100 = 6 :=
by
  let holly_contribution := holly_contribution_per_paycheck * total_paychecks
  let company_contribution := total_contribution_one_year - holly_contribution
  have h : holly_contribution = 2600 := by sorry
  have c : company_contribution = 156 := by sorry
  exact sorry

end company_match_percentage_l518_518490


namespace find_point_B_find_line_BC_l518_518185

-- Define the coordinates of point A
def point_A : ℝ × ℝ := (2, -1)

-- Define the equation of the median on side AB
def median_AB (x y : ℝ) : Prop := x + 3 * y = 6

-- Define the equation of the internal angle bisector of ∠ABC
def bisector_BC (x y : ℝ) : Prop := x - y = -1

-- Prove the coordinates of point B
theorem find_point_B :
  (a b : ℝ) →
  (median_AB ((a + 2) / 2) ((b - 1) / 2)) →
  (a - b = -1) →
  a = 5 / 2 ∧ b = 7 / 2 :=
sorry

-- Define the line equation BC
def line_BC (x y : ℝ) : Prop := x - 9 * y + 29 = 0

-- Prove the equation of the line containing side BC
theorem find_line_BC :
  (x0 y0 : ℝ) →
  bisector_BC x0 y0 →
  (x0, y0) = (-2, 3) →
  line_BC x0 y0 :=
sorry

end find_point_B_find_line_BC_l518_518185


namespace sqrt_inequality_l518_518223

theorem sqrt_inequality : (Real.sqrt 6 + Real.sqrt 7) > (2 * Real.sqrt 2 + Real.sqrt 5) :=
by {
  sorry
}

end sqrt_inequality_l518_518223


namespace points_per_member_l518_518785

theorem points_per_member
  (total_members : ℕ)
  (members_didnt_show : ℕ)
  (total_points : ℕ)
  (H1 : total_members = 14)
  (H2 : members_didnt_show = 7)
  (H3 : total_points = 35) :
  total_points / (total_members - members_didnt_show) = 5 :=
by
  sorry

end points_per_member_l518_518785


namespace prob_A_hits_B_misses_prob_equal_hits_after_two_shots_l518_518242

open ProbabilityTheory

-- Definitions for conditions
def prob_A : ℚ := 3 / 4
def prob_B : ℚ := 4 / 5

-- Theorem statement for Part (I)
theorem prob_A_hits_B_misses :
  prob_A * (1 - prob_B) = 3 / 20 :=
by
  sorry

-- Theorem statement for Part (II)
theorem prob_equal_hits_after_two_shots :
  (prob_A^2 * prob_B^2 + 2 * (prob_A * (1 - prob_A)) * (prob_B * (1 - prob_B)) + (1 - prob_A)^2 * (1 - prob_B)^2) =
  193 / 400 :=
by
  sorry

end prob_A_hits_B_misses_prob_equal_hits_after_two_shots_l518_518242


namespace painting_cost_is_correct_l518_518257

-- Definitions of radii
def large_circle_radius : ℝ := 2
def small_circle_radius : ℝ := 1

-- Definitions of painting costs
def cost_per_square_meter_dark_gray : ℝ := 30
def cost_per_square_meter_medium_gray : ℝ := 20
def cost_per_square_meter_light_gray : ℝ := 10

-- Areas calculated in the problem
def area_large_circle := π * large_circle_radius ^ 2
def area_one_small_circle := π * small_circle_radius ^ 2
def area_four_small_circles := 4 * area_one_small_circle

def area_light_gray := (π - 2)
def area_medium_gray := 3 * π + 2
def area_dark_gray := area_light_gray -- A_dark was found equivalent to A_light

-- Calculating total cost
def total_cost : ℝ :=
  cost_per_square_meter_dark_gray * area_dark_gray +
  cost_per_square_meter_medium_gray * area_medium_gray +
  cost_per_square_meter_light_gray * area_light_gray

-- The theorem to prove
theorem painting_cost_is_correct :
  total_cost = 100 * π - 40 :=
by
  sorry

end painting_cost_is_correct_l518_518257


namespace find_number_of_employees_l518_518708

theorem find_number_of_employees (E : ℕ) :
  (0.99 * E) - 149.99999999999986 = 0.98 * E → E = 15000 :=
by
  sorry

end find_number_of_employees_l518_518708


namespace trapezium_area_l518_518316

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  1/2 * (a + b) * h = 247 := 
by
  rw [ha, hb, hh]
  calc
    (1/2) * (20 + 18) * 13 = (1/2) * 38 * 13 := by rfl
                         ... = 19 * 13     := by norm_num
                         ... = 247         := by norm_num

end trapezium_area_l518_518316


namespace equation_solutions_l518_518993

noncomputable def num_solutions (m : ℕ) : ℕ :=
  m * (m - 1) + 1

theorem equation_solutions (m : ℕ) (h_pos : 0 < m) :
  ∃ n, n = num_solutions m ∧ ∀ x ∈ set.Icc (1 : ℝ) m, x^2 - ⌊x^2⌋ = fract x ^ 2 :=
sorry

end equation_solutions_l518_518993


namespace collinear_vectors_result_magnitude_sum_vectors_result_l518_518099

-- Definitions for given vectors and conditions
def vec_a (θ : ℝ) : ℝ × ℝ := (Real.sin (θ - Real.pi), 1)
def vec_b (θ : ℝ) : ℝ × ℝ := (Real.sin (Real.pi / 2 - θ), -1 / 2)

-- Condition for being collinear, leading to first question result
theorem collinear_vectors_result (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2) 
(h_collinear : (vec_a θ).fst * (vec_b θ).snd = (vec_b θ).fst * (vec_a θ).snd) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 3 := sorry

-- Condition for magnitude of sum of vectors, leading to second question result
theorem magnitude_sum_vectors_result (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2)
(h_magnitude : (vec_a θ).fst + (vec_b θ).fst)^2 + (vec_a θ).snd + (vec_b θ).snd^2 = 1) :
  Real.sin θ + Real.cos θ = Real.sqrt(5) / 2 := sorry

end collinear_vectors_result_magnitude_sum_vectors_result_l518_518099


namespace sum_tensor_A_B_l518_518834

def A : Set ℕ := {4, 5, 6}
def B : Set ℕ := {1, 2, 3}

def tensor (A B : Set ℕ) : Set ℕ := {x | ∃ (m ∈ A) (n ∈ B), x = m - n}

theorem sum_tensor_A_B :
  (∑ x in (tensor A B), x) = 15 :=
sorry

end sum_tensor_A_B_l518_518834


namespace M_minus_m_l518_518778

theorem M_minus_m :
  ∀ (n G R : ℕ),
    1000 ≤ n → n ≤ 1200 →
    ⌈0.7 * n⌉ ≤ G → G ≤ ⌊0.75 * n⌋ →
    ⌈0.35 * n⌉ ≤ R → R ≤ ⌊0.45 * n⌋ →
    let m := G + R - 1200 in
    let M := G + R - 1000 in
    M - m = 190 :=
by
  intros n G R hnle hng hGle hG hnRle hR
  let m := G + R - 1200
  let M := G + R - 1000
  sorry

end M_minus_m_l518_518778


namespace program_selection_count_l518_518356

-- Definitions based on the conditions
def courses := {English, Algebra, Geometry, History, Art, Latin, Science}

def is_math (course : courses) : Prop := 
course = Algebra ∨ course = Geometry

def is_science (course : courses) : Prop := 
course = Science

-- State the proof problem
theorem program_selection_count : 
  ∃ c : finset courses, 
    c.card = 4 ∧ 
    English ∈ c ∧ 
    (∃ m ∈ c, is_math m) ∧ 
    (∃ s ∈ c, is_science s) ∧ 
    finset.power_set_len 4 courses.card = 19 :=
sorry

end program_selection_count_l518_518356


namespace focus_of_ellipse_l518_518798

theorem focus_of_ellipse :
  let center_x := 2,
      center_y := (0 + 8) / 2,
      a := 4, -- semi-major axis
      b := 2, -- semi-minor axis
      distance_to_focus := Real.sqrt (a^2 - b^2),
      focus_y := center_y + distance_to_focus
  in
  (center_x, focus_y) = (2, 4 + 2 * Real.sqrt 3) :=
by
  let center_x := 2
  let center_y := (0 + 8) / 2
  let a := 4 -- semi-major axis
  let b := 2 -- semi-minor axis
  let distance_to_focus := Real.sqrt (a^2 - b^2)
  let focus_y := center_y + distance_to_focus
  show (center_x, focus_y) = (2, 4 + 2 * Real.sqrt 3)
  sorry

end focus_of_ellipse_l518_518798


namespace exists_tetrahedron_with_volume_1_l518_518084

section Problem

variables (points : Set (ℝ × ℝ × ℝ))
variables (h_points_card : Fintype.card points = 1996)
variables (h_non_coplanar : ∀ (p1 p2 p3 p4 : ℝ × ℝ × ℝ) (h_p1 : p1 ∈ points) (h_p2 : p2 ∈ points)
                                        (h_p3 : p3 ∈ points) (h_p4 : p4 ∈ points), ¬ are_coplanar {p1, p2, p3, p4})
variables (h_tetra_volume_lt_0037 : ∀ (p1 p2 p3 p4 : ℝ × ℝ × ℝ) (h_p1 : p1 ∈ points) (h_p2 : p2 ∈ points)
                                                 (h_p3 : p3 ∈ points) (h_p4 : p4 ∈ points), tetra_volume {p1, p2, p3, p4} < 0.037)

theorem exists_tetrahedron_with_volume_1 (points : Set (ℝ × ℝ × ℝ)) :
  ∃ (A B C D : ℝ × ℝ × ℝ), (A ∈ points) ∧ (B ∈ points) ∧ (C ∈ points) ∧ (D ∈ points) ∧ 
                          ∀ (P ∈ points), in_tetrahedron P {A, B, C, D} → tetra_volume {A, B, C, D} = 1 :=
  by 
  -- Given conditions
  have non_coplanar := h_non_coplanar,
  have tetra_volume_lt_0037 := h_tetra_volume_lt_0037,
  sorry

end Problem

noncomputable def are_coplanar (points : Set (ℝ × ℝ × ℝ)) : Prop := sorry
noncomputable def tetra_volume (tetra : Set (ℝ × ℝ × ℝ)) : ℝ := sorry
noncomputable def in_tetrahedron (p : ℝ × ℝ × ℝ) (tetra : Set (ℝ × ℝ × ℝ)) : Prop := sorry

end exists_tetrahedron_with_volume_1_l518_518084


namespace bob_measurements_l518_518375

theorem bob_measurements (flour_needed : ℚ) (flour_cup : ℚ) (milk_needed : ℚ) (milk_cup : ℚ) :
  flour_needed = 15/4 → flour_cup = 1/3 → milk_needed = 3/2 → milk_cup = 1/2 →
  (⌈flour_needed / flour_cup⌉ + ⌊milk_needed / milk_cup⌋) = 15 :=
begin
  intros hfn hfc hmn hmc,
  rw [hfn, hfc, hmn, hmc],
  simp,
  -- proof steps would go here
  sorry
end

end bob_measurements_l518_518375


namespace correct_statements_l518_518987

variables {d : ℝ} {S : ℕ → ℝ} {a : ℕ → ℝ}

axiom arithmetic_sequence (n : ℕ) : S n = n * a 1 + (n * (n - 1) / 2) * d

theorem correct_statements (h1 : S 6 = S 12) :
  (S 18 = 0) ∧ (d > 0 → a 6 + a 12 < 0) ∧ (d < 0 → |a 6| > |a 12|) :=
sorry

end correct_statements_l518_518987


namespace impossible_daisy_exchange_l518_518704

theorem impossible_daisy_exchange : 
  ∀ (n : ℕ) (girls : Fin n → ℕ) (h_n : n = 33) (h_girls : ∀ i : Fin n, girls i = i + 1),
  ¬ (∃ daisies : Fin n → ℕ, (∀ i : Fin n, daisies (i + 2) = girls i)) := 
by
  intros n girls h_n h_girls
  sorry

end impossible_daisy_exchange_l518_518704


namespace square_prism_volume_l518_518454

-- Given conditions for the math problem
variable (a r : ℝ)
axiom height_prism : 2
axiom surface_area_sphere : 4 * Real.pi * r^2 = 6 * Real.pi

-- Definition for the diagonal of the prism using given conditions
def space_diagonal := 2 * r
def base_diagonal := a * Real.sqrt 2

-- The relationship from the problem statement:
axiom relation : (space_diagonal) ^ 2 = (base_diagonal) ^ 2 + (height_prism) ^ 2

-- Solve for the volume of the prism
def prism_volume := a^2 * height_prism

-- The theorem to prove: volume of the prism is 2
theorem square_prism_volume : prism_volume = 2 :=
by
  -- Lean code to justify the theorem omitted, so we use sorry
  sorry

end square_prism_volume_l518_518454


namespace travel_time_tripled_l518_518063

variable {S v v_r : ℝ}

-- Conditions of the problem
def condition1 (t1 t2 : ℝ) : Prop :=
  t1 = 3 * t2

def condition2 (t1 t2 : ℝ) : Prop :=
  t1 = S / (v + v_r) ∧ t2 = S / (v - v_r)

def stationary_solution : Prop :=
  v = 2 * v_r

-- Conclusion: Time taken to travel from B to A without paddles is 3 times longer than usual
theorem travel_time_tripled (t_no_paddle t2 : ℝ) (h1 : condition1 t_no_paddle t2) (h2 : condition2 t_no_paddle t2) (h3 : stationary_solution) :
  t_no_paddle = 3 * t2 :=
sorry

end travel_time_tripled_l518_518063


namespace three_non_collinear_points_determine_one_plane_l518_518712

theorem three_non_collinear_points_determine_one_plane 
  (A B C : Point) (h1 : ¬Collinear A B C) : 
  ∃! P : Plane, A ∈ P ∧ B ∈ P ∧ C ∈ P :=
sorry

end three_non_collinear_points_determine_one_plane_l518_518712


namespace area_of_rhombus_l518_518149

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 10) : 
  1 / 2 * d1 * d2 = 30 :=
by 
  rw [h1, h2]
  norm_num

end area_of_rhombus_l518_518149


namespace rectangle_same_color_exists_l518_518396

def color := ℕ -- We use ℕ as a stand-in for three colors {0, 1, 2}

def same_color_rectangle_exists (coloring : (Fin 4) → (Fin 82) → color) : Prop :=
  ∃ (i j : Fin 4) (k l : Fin 82), i ≠ j ∧ k ≠ l ∧
    coloring i k = coloring i l ∧
    coloring j k = coloring j l ∧
    coloring i k = coloring j k

theorem rectangle_same_color_exists :
  ∀ (coloring : (Fin 4) → (Fin 82) → color),
  same_color_rectangle_exists coloring :=
by
  sorry

end rectangle_same_color_exists_l518_518396


namespace line_intersects_circle_l518_518127

theorem line_intersects_circle (a : ℝ) :
  let l := λ x : ℝ, k * x + 1
  let circle := λ x y : ℝ, x^2 + y^2 - 2 * a * x + a^2 - 2 * a - 4 
  (∀ k : ℝ, ∃ x y : ℝ, y = l x ∧ circle x y = 0) ↔ -1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end line_intersects_circle_l518_518127


namespace part_c_part_b_l518_518108

-- Definitions
def f : ℝ → ℝ := sorry
def g (x : ℝ) := (f' x)

-- Conditions
axiom dom_f : ∀ x : ℝ, x ∈ domain f
axiom dom_g : ∀ x : ℝ, x ∈ domain g
axiom even_f : ∀ x : ℝ, f (3 / 2 - 2 * x) = f (3 / 2 + 2 * x)
axiom even_g : ∀ x : ℝ, g (2 + x) = g (2 - x)

-- Theorems to prove
theorem part_c : f (-1) = f 4 := by sorry
theorem part_b : g (-1 / 2) = 0 := by sorry

end part_c_part_b_l518_518108


namespace range_of_expression_l518_518173

theorem range_of_expression (x : ℝ) (h1 : 1 - 3 * x ≥ 0) (h2 : 2 * x ≠ 0) : x ≤ 1 / 3 ∧ x ≠ 0 := by
  sorry

end range_of_expression_l518_518173


namespace ways_to_distribute_balls_in_boxes_l518_518517

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l518_518517


namespace rowing_time_to_place_and_back_l518_518343

-- Define the conditions
def rowing_speed_still_water := 4 -- in kmph
def river_current_speed := 2 -- in kmph
def distance_to_place := 2.25 -- in km

-- Calculate the downstream and upstream speeds
def downstream_speed := rowing_speed_still_water + river_current_speed -- in kmph
def upstream_speed := rowing_speed_still_water - river_current_speed -- in kmph

-- Convert speeds to km/min
def downstream_speed_min := downstream_speed / 60 -- in km/min
def upstream_speed_min := upstream_speed / 60 -- in km/min

-- Calculate the time taken to row downstream and upstream in minutes
def time_downstream := distance_to_place / downstream_speed_min
def time_upstream := distance_to_place / upstream_speed_min

-- Total time to row to the place and back
def total_time := time_downstream + time_upstream

-- The theorem statement
theorem rowing_time_to_place_and_back : total_time = 90 :=
sorry

end rowing_time_to_place_and_back_l518_518343


namespace manufacturing_percentage_l518_518668

theorem manufacturing_percentage (a b : ℕ) (h1 : a = 108) (h2 : b = 360) : (a / b : ℚ) * 100 = 30 :=
by
  sorry

end manufacturing_percentage_l518_518668


namespace range_of_m_l518_518898

noncomputable def line (k : ℝ) : ℝ := 𝑓 x = k * x + 2

noncomputable def ellipse (x y m : ℝ) : Prop := x^2 + (y^2) / m = 1

theorem range_of_m (k : ℝ) (m : ℝ) (h : ∀ x y : ℝ, ellipse x y m → y = k * x + 2) : m ≥ 4 :=
by
  sorry

end range_of_m_l518_518898


namespace rotation_center_l518_518259

noncomputable def f (z : ℂ) : ℂ := 2 * ((1 + complex.I) * z + (-4 - 6 * complex.I))

theorem rotation_center (c : ℂ) (h : f c = c) : 
  c = -16/5 + (28/5) * complex.I := 
by
  sorry

end rotation_center_l518_518259


namespace range_of_function_l518_518726

theorem range_of_function :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x + 2)) ↔ (y ∈ Set.Iio 1 ∨ y ∈ Set.Ioi 1) := 
sorry

end range_of_function_l518_518726


namespace molecular_weight_one_mole_l518_518723

theorem molecular_weight_one_mole {compound : Type} (moles : ℕ) (total_weight : ℝ) 
  (h_moles : moles = 5) (h_total_weight : total_weight = 490) :
  total_weight / moles = 98 := 
by {
    rw [h_moles, h_total_weight],
    norm_num,
    sorry
  }

end molecular_weight_one_mole_l518_518723


namespace integral_f_eq_l518_518145

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 2 * (2 - Real.pi) * x + Real.sin (2 * x)

theorem integral_f_eq : (∫ x in 0..1, f x) = 17/6 - Real.pi - (1/2) * Real.cos 2 :=
by
  sorry

end integral_f_eq_l518_518145


namespace sum_of_possible_values_sum_x_values_l518_518233

theorem sum_of_possible_values (x : ℝ) (h : 3^(x^2 + 6*x + 9) = 27^(x + 3)) : x = 0 ∨ x = -3 :=
begin
  sorry
end

theorem sum_x_values (h1 : ∀ x, 3^(x^2 + 6*x + 9) = 27^(x + 3) → (x = 0 ∨ x = -3)) 
: ∃ s, s = (0 + (-3)) :=
begin
  use -3,
  { intros x h,
    have hl := h1 x h,
    cases hl,
    { simp [hl] },
    { simp [hl] },
  },
  simp
end

end sum_of_possible_values_sum_x_values_l518_518233


namespace leo_mira_difference_l518_518592

/--
Leo writes down the whole numbers from 1 to 50.
Mira copies Leo's list, but she makes the following changes:
- each occurrence of the digit '2' is replaced by '1'
- each occurrence of digit '3' is replaced by '0'
Prove that the difference between the sum of the numbers Leo writes 
and the sum Mira writes after these digit replacements is 420.
-/
theorem leo_mira_difference : 
  let leo_sum := (List.range 50).map (· + 1) |>.sum in
  let mira_transform (n : ℕ) : ℕ := 
    let digits := n.digits 10 in
    digits.map (λ d => if d = 2 then 1 else if d = 3 then 0 else d) 
      |>.foldl (λ acc d => 10 * acc + d) 0 in
  let mira_sum := (List.range 50).map (· + 1) |>.map mira_transform |>.sum in
  leo_sum - mira_sum = 420 :=
by
  sorry

end leo_mira_difference_l518_518592


namespace inequality_identification_l518_518733

theorem inequality_identification (A B C D : Prop) (h_A : A = (0 < 19)) 
  (h_B : B = (∃ x : ℝ, x - 2 ≠ x)) -- using a trivially true condition for non-redundancy
  (h_C : C = (∃ x y : ℝ, 2 * x + 3 * y = -1)) 
  (h_D : D = (∃ y : ℝ, y^2 = y^2)) -- using a trivially true condition for non-redundancy
  : A := 
by 
  rw [h_A]
  exact lt_irrefl 19
  sorry

end inequality_identification_l518_518733


namespace find_angle_A_find_perimeter_l518_518152

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : sqrt 3 * b * cos A = a * cos B)
variable (h2 : a = sqrt 2)
variable (h3 : c / a = sin A / sin B)

-- Part 1: Prove A = π/3
theorem find_angle_A : A = π / 3 :=
by
  sorry

-- Part 2: Prove the perimeter is 3 sqrt 2
theorem find_perimeter : a + b + c = 3 * sqrt 2 :=
by
  sorry

end find_angle_A_find_perimeter_l518_518152


namespace frustum_lateral_surface_area_is_correct_l518_518681

noncomputable def slant_height (h R r : ℝ) : ℝ :=
  Real.sqrt (h^2 + (R - r)^2)

noncomputable def height_of_cone (R : ℝ) : ℝ :=
  R / Real.tan (Real.pi / 6)

noncomputable def lateral_surface_area_frustum (R r h : ℝ) : ℝ :=
  let s := slant_height h R r
  let H := height_of_cone R
  let H_small := (r / R) * H
  let s_full := Real.sqrt (H^2 + R^2)
  let s_small := Real.sqrt (H_small^2 + r^2)
  Real.pi * (R * s_full - r * s_small)

theorem frustum_lateral_surface_area_is_correct : 
  lateral_surface_area_frustum 8 4 6 = 96 * Real.pi :=
by {
  rw [lateral_surface_area_frustum],
  rw [slant_height, height_of_cone],
  --- Continue the proof steps, or simply add:
  sorry
}

end frustum_lateral_surface_area_is_correct_l518_518681


namespace garden_length_l518_518313

theorem garden_length (w l : ℝ) (h1: l = 2 * w) (h2 : 2 * l + 2 * w = 180) : l = 60 := 
by
  sorry

end garden_length_l518_518313


namespace projection_norm_square_l518_518606

variables (v w : Vec ℝ) (hp : proj v w)
variables (q : proj p v)
variables (h1 : ∥p∥ / ∥v∥ = 3 / 7)

theorem projection_norm_square (v w : Vec ℝ) 
  (hp : proj v w) 
  (q : proj p v) 
  (h1 : ∥p∥ / ∥v∥ = 3 / 7) 
: ∥q∥ / ∥v∥ = 9 / 49 := 
  sorry

end projection_norm_square_l518_518606


namespace find_parallel_line_through_point_l518_518825

noncomputable def line_equation_passing_through_point_and_parallel
(P : ℝ × ℝ) (a b c : ℝ) (m : ℝ) : Prop :=
let (x1, y1) := P in
a * x1 + b * y1 + m = 0 ∧ a = 2 ∧ b = 1 ∧ c = -3

theorem find_parallel_line_through_point (P : ℝ × ℝ)
(h : P = (2, -1)) : ∃ m : ℝ, line_equation_passing_through_point_and_parallel P 2 1 m :=
begin
  use -3,
  simp [line_equation_passing_through_point_and_parallel, h],
  sorry
end

end find_parallel_line_through_point_l518_518825


namespace distance_to_school_l518_518970

-- Definitions based on conditions
def normal_drive_time : ℝ := 1 / 2
def extended_drive_time : ℝ := 5 / 6
def speed_decrease : ℝ := 6

-- Proof that the distance to school is 7.5 miles given the conditions
theorem distance_to_school (v d : ℝ) 
    (h1 : d = v * normal_drive_time) 
    (h2 : d = (v - speed_decrease) * extended_drive_time) :
  d = 7.5 := 
sorry

end distance_to_school_l518_518970


namespace arccos_range_l518_518083

noncomputable def range_arccos (x : ℝ) : ℝ := Real.arccos x

theorem arccos_range (α : ℝ) (x : ℝ) (h1 : x = Real.sin α) (h2 : -Real.pi / 6 ≤ α ∧ α ≤ 5 * Real.pi / 6) :
  (0 ≤ range_arccos x ∧ range_arccos x ≤ (2 / 3) * Real.pi) :=
by
  have hα_range : -1 / 2 ≤ Real.sin α ∧ Real.sin α ≤ 1 := sorry
  have hx_range : -1 / 2 ≤ x ∧ x ≤ 1 := sorry
  have h_arccos : 0 ≤ Real.arccos x ∧ Real.arccos x ≤ 2 * Real.pi / 3 := sorry
  exact h_arccos

end arccos_range_l518_518083


namespace arithmetic_expression_value_l518_518377

def mixed_to_frac (a b c : ℕ) : ℚ := a + b / c

theorem arithmetic_expression_value :
  ( ( (mixed_to_frac 5 4 45 - mixed_to_frac 4 1 6) / mixed_to_frac 5 8 15 ) / 
    ( (mixed_to_frac 4 2 3 + 3 / 4) * mixed_to_frac 3 9 13 ) * mixed_to_frac 34 2 7 + 
    (3 / 10 / (1 / 100) / 70) + 2 / 7 ) = 1 :=
by
  -- We need to convert the mixed numbers to fractions using mixed_to_frac
  -- Then, we simplify step-by-step as in the problem solution, but for now we just use sorry
  sorry

end arithmetic_expression_value_l518_518377


namespace liars_at_table_l518_518639

open Set

noncomputable def number_of_liars : Set ℕ :=
  {n | ∃ (knights, liars : ℕ), knights + liars = 450 ∧
                                (∀ i : ℕ, i < 450 → (liars + ((i + 1) % 450) + ((i + 2) % 450) = 1)) }

theorem liars_at_table : number_of_liars = {150, 450} := 
  sorry

end liars_at_table_l518_518639


namespace find_plane_Q_l518_518629

variables {x y z : ℝ}

/-- Line M is the intersection of the planes P1 and P2 -/
def line_M (x y z : ℝ) : Prop :=
  (x - y + 2 * z = 1) ∧ (2 * x + 3 * y - z = 4)

/-- Plane Q contains line M and has a given distance from the point (1, 2, 3). -/
def plane_Q (A B C D : ℝ) : Prop :=
  ∀ {x y z : ℝ}, line_M x y z → A * x + B * y + C * z + D = 0

noncomputable def distance_to_point (A B C D : ℝ) (x y z : ℝ) : ℝ :=
  abs (A * x + B * y + C * z + D) / sqrt (A^2 + B^2 + C^2)

theorem find_plane_Q : 
  ∃ (A B C D : ℝ), plane_Q A B C D ∧ distance_to_point A B C D 1 2 3 = 3 / sqrt 14 ∧
  (A, B, C, D) = (10, -32, 25, -39) ∧ A > 0 ∧ Int.gcd (A.natAbs, B.natAbs, C.natAbs, D.natAbs) = 1 :=
begin
  sorry,
end

end find_plane_Q_l518_518629


namespace liars_at_table_l518_518641

open Set

noncomputable def number_of_liars : Set ℕ :=
  {n | ∃ (knights, liars : ℕ), knights + liars = 450 ∧
                                (∀ i : ℕ, i < 450 → (liars + ((i + 1) % 450) + ((i + 2) % 450) = 1)) }

theorem liars_at_table : number_of_liars = {150, 450} := 
  sorry

end liars_at_table_l518_518641


namespace quadratic_no_real_roots_min_k_l518_518029

theorem quadratic_no_real_roots_min_k :
  ∀ (k : ℤ), 
    (∀ x : ℝ, 3*x*(k*x-5) - 2*x^2 + 8 ≠ 0) ↔ 
    (k ≥ 3) := 
by 
  sorry

end quadratic_no_real_roots_min_k_l518_518029


namespace at_least_25_circled_not_crossed_l518_518272

-- Definitions based on the problem conditions
def grid (m n : ℕ) := ℕ → ℕ → ℕ

def crossed_out (g : grid 10 10) (r : fin 10) : set (fin 10) :=
{c | g r c ∈ {min1 g r, min2 g r, min3 g r, min4 g r, min5 g r}}

def circled (g : grid 10 10) (c : fin 10) : set (fin 10) :=
{r | g r c ∈ {max1 g c, max2 g c, max3 g c, max4 g c, max5 g c}}

-- The theorem statement we want to prove
theorem at_least_25_circled_not_crossed (g : grid 10 10) :
  ∃ S : set (fin 10 × fin 10), S.card ≥ 25 ∧
    ∀ (rc : fin 10 × fin 10), rc ∈ S → (rc.snd ∈ circled g rc.fst ∧ rc.snd ∉ crossed_out g rc.fst) :=
sorry

end at_least_25_circled_not_crossed_l518_518272


namespace nine_sided_polygon_diagonals_l518_518416

theorem nine_sided_polygon_diagonals : ∀ (n : ℕ), n = 9 → (n * (n - 3)) / 2 = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end nine_sided_polygon_diagonals_l518_518416


namespace part_a_part_b_l518_518338

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

def in_unit_square (A : ℝ × ℝ) : Prop :=
  0 ≤ A.1 ∧ A.1 ≤ 1 ∧ 0 ≤ A.2 ∧ A.2 ≤ 1

def convex_polygon_in_square (n : ℕ) (vertices : Fin n → ℝ × ℝ) : Prop :=
  (∀ i, in_unit_square (vertices i)) ∧
  -- Polygon must be convex (using some convexity condition here, e.g., all internal angles are less than 180 degrees)
  -- Simplifying, assume vertices form a cycle
  List.chain (λ v₁ v₂, v₁ ≠ v₂) true (List.ofFn vertices)

theorem part_a (n : ℕ) (hn : n ≥ 3) (vertices : Fin n → ℝ × ℝ) (h_convex: convex_polygon_in_square n vertices) :
  ∃ (i j k : Fin n), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ area_triangle (vertices i) (vertices j) (vertices k) ≤ 8 / n^2 := 
  sorry

theorem part_b (n : ℕ) (hn : n ≥ 3) (vertices : Fin n → ℝ × ℝ) (h_convex: convex_polygon_in_square n vertices) :
  ∃ (i j k : Fin n), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ area_triangle (vertices i) (vertices j) (vertices k) ≤ 16 * Real.pi / n^3 := 
  sorry

end part_a_part_b_l518_518338


namespace distribute_balls_in_boxes_l518_518515

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l518_518515


namespace sum_of_both_sequences_l518_518462

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

theorem sum_of_both_sequences (a : ℕ → ℝ) (n : ℕ) (a_1 : ℝ)
  (h_arith: is_arithmetic_sequence a)
  (h_geom: is_geometric_sequence a) :
  (∑ i in finset.range n, a i) = n * a_1 :=
sorry

end sum_of_both_sequences_l518_518462


namespace ball_box_problem_l518_518497

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l518_518497


namespace canoe_no_paddle_time_l518_518068

-- All conditions needed for the problem
variables {S v v_r : ℝ}
variables (time_pa time_pb : ℝ)

-- Condition that time taken from A to B is 3 times the time taken from B to A
def condition1 : Prop := time_pa = 3 * time_pb

-- Define time taken from A to B (downstream) and B to A (upstream)
def time_pa_def : time_pa = S / (v + v_r) := sorry
def time_pb_def : time_pb = S / (v - v_r) := sorry

-- Main theorem stating the problem to prove
theorem canoe_no_paddle_time :
  condition1 →
  ∃ (t_no_paddle : ℝ), t_no_paddle = 3 * time_pb :=
begin
  intro h1,
  sorry
end

end canoe_no_paddle_time_l518_518068


namespace find_radius_l518_518773

noncomputable def radius_probability (d : ℝ) : Prop :=
  π * d^2 = 1 / 3

theorem find_radius (d : ℝ) : radius_probability d → d = 1 / Real.sqrt (3 * π) :=
by
  sorry

end find_radius_l518_518773


namespace time_to_travel_from_B_to_A_without_paddles_l518_518073

-- Variables definition 
variables (v v_r S : ℝ)
-- Assume conditions
def condition_1 (t₁ t₂ : ℝ) (v v_r S : ℝ) := t₁ = 3 * t₂
def t₁ (S v v_r : ℝ) := S / (v + v_r)
def t₂ (S v v_r : ℝ) := S / (v - v_r)

theorem time_to_travel_from_B_to_A_without_paddles
  (v v_r S : ℝ)
  (h1 : v = 2 * v_r)
  (h2 : t₁ S v v_r = 3 * t₂ S v v_r) :
  let t_no_paddle := S / v_r in
  t_no_paddle = 3 * t₂ S v v_r :=
sorry

end time_to_travel_from_B_to_A_without_paddles_l518_518073


namespace find_y_in_terms_of_z_l518_518239

def g (t : ℝ) (ht : t ≠ 1) : ℝ := 2 * t / (1 - t)

theorem find_y_in_terms_of_z (y z : ℝ) (hy : y ≠ 1) (hz : z = g y hy) : y = z / (2 + z) :=
by sorry

end find_y_in_terms_of_z_l518_518239
