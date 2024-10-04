import Analysis.Calculus.Deriv.Basic
import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.LinearEquiv.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Extrema
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Integral.Integrable
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Integral
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Combinatorics
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Cast
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.TriangleProperties
import Mathlib.Init.Data.Int.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.Basic

namespace range_of_a_l42_42524

theorem range_of_a (a : ℝ) : (let z := (1 - complex.i) * (a + complex.i) in (z.re < 0 ∧ z.im > 0)) → a < -1 :=
sorry

end range_of_a_l42_42524


namespace cubic_inches_in_one_cubic_foot_l42_42111

theorem cubic_inches_in_one_cubic_foot (h : 1.foot = 12.inches) : (1.foot)^3 = 1728 * (1.inches)^3 :=
by
  rw [h]
  calc (12.foot)^3 = 12^3 * (1.inches)^3 : sorry

end cubic_inches_in_one_cubic_foot_l42_42111


namespace parabola_equation_line_AB_equation_l42_42189

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42189


namespace M_inter_N_is_empty_l42_42124

def M : set (ℂ) :=
  {z : ℂ | ∃ (t : ℝ), t ≠ -1 ∧ t ≠ 0 ∧ z = ↑(t / (1 + t)) + complex.I * ↑((1 + t) / t)}

def N : set (ℂ) :=
  {z : ℂ | ∃ (t : ℝ), t ≤ 1 ∧ z = ↑sqrt(2) * (↑(cos (real.arcsin t)) + complex.I * ↑(cos (real.arccos t)))}

theorem M_inter_N_is_empty : M ∩ N = ∅ := 
sorry

end M_inter_N_is_empty_l42_42124


namespace A_beats_B_by_this_amount_l42_42577

noncomputable def speed_of_A : ℝ := 1000 / 119

noncomputable def distance_A_travels_in_6_seconds : ℝ := speed_of_A * 6

theorem A_beats_B_by_this_amount :
  distance_A_travels_in_6_seconds ≈ 50.418 :=
by
  -- We do not complete the proof here, only stating it as a theorem.
  sorry

end A_beats_B_by_this_amount_l42_42577


namespace shelter_service_count_l42_42296

def total_cans : ℕ := 1800
def cans_per_person : ℕ := 10
def num_shelters : ℕ := 6

theorem shelter_service_count :
  let total_people := total_cans / cans_per_person in
  let people_per_shelter := total_people / num_shelters in
  people_per_shelter = 30 :=
by
  sorry

end shelter_service_count_l42_42296


namespace factors_of_48_are_multiples_of_6_l42_42937

theorem factors_of_48_are_multiples_of_6 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d, d ∣ 48 → (6 ∣ d ↔ d = 6 ∨ d = 12 ∨ d = 24 ∨ d = 48) := 
by { sorry }

end factors_of_48_are_multiples_of_6_l42_42937


namespace probability_white_ball_l42_42584

theorem probability_white_ball :
  let total_balls := 3 + 2 + 1 in
  let white_balls := 2 in
  (white_balls / total_balls : ℚ) = 1 / 3 :=
by
  let total_balls := 3 + 2 + 1
  let white_balls := 2
  have h1 : total_balls = 6 := by rfl
  have h2 : white_balls = 2 := by rfl
  calc
    (2 / 6 : ℚ) = 1 / 3 : by norm_num

end probability_white_ball_l42_42584


namespace exists_even_not_prime_n_minus_three_prime_l42_42393

theorem exists_even_not_prime_n_minus_three_prime :
  ∃ n : ℕ, even n ∧ ¬ (prime n) ∧ prime (n - 3) :=
by
  use 14
  sorry

end exists_even_not_prime_n_minus_three_prime_l42_42393


namespace problem_f_2004_l42_42911

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_f_2004 (a α b β : ℝ) 
  (h_non_zero : a ≠ 0 ∧ b ≠ 0 ∧ α ≠ 0 ∧ β ≠ 0) 
  (h_condition : f 2003 a α b β = 6) : 
  f 2004 a α b β = 2 := 
by
  sorry

end problem_f_2004_l42_42911


namespace closest_whole_number_l42_42849

theorem closest_whole_number :
  let expr := (10 ^ 3000 + 10 ^ 3004) / (10 ^ 3002 + 10 ^ 3002) in
  abs (expr - 50) < 1 :=
by
  let expr := (10 ^ 3000 + 10 ^ 3004) / (10 ^ 3002 + 10 ^ 3002)
  calc
    expr = (10 ^ 3000 * (1 + 10 ^ 4)) / (10 ^ 3002 * 2) : by sorry
    ... = (10 ^ 3000 * 10001) / (10 ^ 3002 * 2) : by sorry
    ... = 10001 / 200 : by sorry
    ... = 50.005 : by sorry
  exact sorry

end closest_whole_number_l42_42849


namespace max_p_q_r_of_B_squared_eq_identity_l42_42620

theorem max_p_q_r_of_B_squared_eq_identity :
  ∃ (p q r : ℤ) (B : Matrix (Fin 2) (Fin 2) ℤ) (inv_scalar : ℚ),
  B = inv_scalar • Matrix.of ![
    ![-5, 2 * p],
    ![3 * q, r]
  ] ∧
  (B^2 = Matrix.identity 2) ∧
  inv_scalar = (1 / 7 : ℚ) ∧
  (∀ p' q' r' : ℤ, B' = (1 / 7 : ℚ) • Matrix.of ![
    ![-5, 2 * p'],
    ![3 * q', r']
  ] ∧ B'^2 = Matrix.identity 2 → p + q + r ≥ p' + q' + r') :=
begin
  sorry

end max_p_q_r_of_B_squared_eq_identity_l42_42620


namespace truck_capacity_solution_l42_42357

variable (x y : ℝ)

theorem truck_capacity_solution (h1 : 3 * x + 4 * y = 22) (h2 : 2 * x + 6 * y = 23) :
  x + y = 6.5 := sorry

end truck_capacity_solution_l42_42357


namespace part_one_part_two_l42_42233

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42233


namespace avg_distributive_laws_l42_42169

def averaged_with (a b : ℝ) : ℝ := (a + b) / 2

theorem avg_distributive_laws (x y z : ℝ) :
  (x @ (y + z) ≠ (x @ y) + (x @ z)) ∧
  (x + 2 * (y @ z) = (x + 2 * y) @ (x + 2 * z)) ∧
  (x @ (y @ z) = (x @ y) @ (x @ z)) :=
by
{
  sorry
}

end avg_distributive_laws_l42_42169


namespace equation_of_parabola_equation_of_line_AB_l42_42260

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42260


namespace parabola_equation_line_AB_equation_l42_42220

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42220


namespace Brandy_can_safely_drink_20_mg_more_l42_42679

variable (maximum_caffeine_per_day : ℕ := 500)
variable (caffeine_per_drink : ℕ := 120)
variable (number_of_drinks : ℕ := 4)
variable (caffeine_consumed : ℕ := caffeine_per_drink * number_of_drinks)

theorem Brandy_can_safely_drink_20_mg_more :
    caffeine_consumed = caffeine_per_drink * number_of_drinks →
    (maximum_caffeine_per_day - caffeine_consumed) = 20 :=
by
  intros h1
  rw [h1]
  sorry

end Brandy_can_safely_drink_20_mg_more_l42_42679


namespace hyperbola_ratio_l42_42040

theorem hyperbola_ratio {a b : ℝ} (h : a > b)
  (eq_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (angle_asymptotes : angle_between_asymptotes a b = 45) :
  a / b = Real.sqrt 2 + 1 := 
sorry

end hyperbola_ratio_l42_42040


namespace find_m_minus_n_l42_42064

-- Define line equations, parallelism, and perpendicularity
def line1 (x y : ℝ) : Prop := 3 * x - 6 * y + 1 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := x - m * y + 2 = 0
def line3 (x y : ℝ) (n : ℝ) : Prop := n * x + y + 3 = 0

def parallel (m1 m2 : ℝ) : Prop := m1 = m2
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_m_minus_n (m n : ℝ) (h_parallel : parallel (1/2) (1/m)) (h_perpendicular: perpendicular (1/2) (-1/n)) : m - n = 0 :=
sorry

end find_m_minus_n_l42_42064


namespace infinite_solutions_l42_42870

theorem infinite_solutions (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by
  sorry

end infinite_solutions_l42_42870


namespace condition_b_l42_42948

theorem condition_b (b : ℝ) : sqrt ((3 - b)^2) = 3 - b → b ≤ 3 :=
by
  intro h
  -- steps of proof would go here
  sorry

end condition_b_l42_42948


namespace find_parabola_equation_find_line_AB_equation_l42_42241

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42241


namespace rotation_equivalence_l42_42825

variable {Point : Type} [Equiv.Point] {A B C : Point}

noncomputable def effectiveCWRotation (deg : ℕ) : ℕ :=
  deg % 360

theorem rotation_equivalence (A B C : Point) (y : ℕ) (h₁: effectiveCWRotation 480 = effectiveCWRotation 120)
  (h₂: y < 360)
  (h₃: Point.rotate A B (effectiveCWRotation 120) = C)
  (h₄: Point.rotate A B y = C)
  : y = 240 :=
sorry

end rotation_equivalence_l42_42825


namespace original_function_l42_42961

def f (x : ℝ) : ℝ := 
sorry

theorem original_function :
  (∀ x, (sin (x - π/4) = sin ((2 * (x - π / 3))))) ↔ (∀ x, f x = sin (x/2 + π/12)) :=
sorry

end original_function_l42_42961


namespace problem1_problem2_l42_42270

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42270


namespace coefficient_a5_l42_42120

theorem coefficient_a5 (a a1 a2 a3 a4 a5 a6 : ℝ) (h :  (∀ x : ℝ, x^6 = a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6)) :
  a5 = 6 :=
sorry

end coefficient_a5_l42_42120


namespace compare_abc_l42_42898

noncomputable def a : ℝ := Real.sin (145 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (52 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (47 * Real.pi / 180)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l42_42898


namespace estate_value_l42_42627

theorem estate_value (E : ℝ) (x : ℝ) (y: ℝ) (z: ℝ) 
  (h1 : 9 * x = 3 / 4 * E) 
  (h2 : z = 8 * x) 
  (h3 : y = 600) 
  (h4 : E = z + 9 * x + y):
  E = 1440 := 
sorry

end estate_value_l42_42627


namespace transformation_correct_l42_42977

noncomputable def original_function : ℝ → ℝ :=
  λ x, sin (x / 2 + π / 12)

theorem transformation_correct :
  ∀ (x : ℝ), (shift_right (shorten_abscissas f) (π / 3)) x = sin (x - π / 4) →
  f = original_function :=
by
  -- Assume the shifted and shortened function equals the given sine transformation
  intro x
  sorry

end transformation_correct_l42_42977


namespace translated_sine_function_l42_42338

theorem translated_sine_function (x : ℝ) :
  let f := λ x, sin (2 * x + π / 5)
  let g := λ x, sin (2 * x)
  (∀ x, g x = (f ∘ (λ x, x - π / 10)) x)
  → ∀ x, g x = sin (2 * x) :=
by sorry

end translated_sine_function_l42_42338


namespace points_on_same_side_parabola_1_points_on_same_side_parabola_3_points_on_same_side_parabola_5_l42_42813

def parabola_1 (x : ℝ) : ℝ :=
  2 * x^2 + 4 * x

def parabola_2 (x : ℝ) : ℝ :=
  (x^2 / 2) - x - (3 / 2)

def parabola_3 (x : ℝ) : ℝ :=
  -x^2 + 2 * x - 1

def parabola_4 (x : ℝ) : ℝ :=
  -x^2 - 4 * x - 3

def parabola_5 (x : ℝ) : ℝ :=
  -x^2 + 3

def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (0, 2)

theorem points_on_same_side_parabola_1 : 
  (parabola_1 A.1 < A.2) ∧ (parabola_1 B.1 < B.2) ∨ (parabola_1 A.1 > A.2) ∧ (parabola_1 B.1 > B.2) := 
sorry

theorem points_on_same_side_parabola_3 : 
  (parabola_3 A.1 < A.2) ∧ (parabola_3 B.1 < B.2) ∨ (parabola_3 A.1 > A.2) ∧ (parabola_3 B.1 > B.2) := 
sorry

theorem points_on_same_side_parabola_5 : 
  (parabola_5 A.1 < A.2) ∧ (parabola_5 B.1 < B.2) ∨ (parabola_5 A.1 > A.2) ∧ (parabola_5 B.1 > B.2) := 
sorry

end points_on_same_side_parabola_1_points_on_same_side_parabola_3_points_on_same_side_parabola_5_l42_42813


namespace number_of_rabbits_l42_42135

-- Given conditions
variable (r c : ℕ)
variable (cond1 : r + c = 51)
variable (cond2 : 4 * r = 3 * (2 * c) + 4)

-- To prove
theorem number_of_rabbits : r = 31 :=
sorry

end number_of_rabbits_l42_42135


namespace problem1_problem2_l42_42502

theorem problem1
  (f : ℝ → ℝ)
  (cond1 : ∀ x y : ℝ, f (x + y) = f x + f y + 1/2)
  (cond2 : ∀ x : ℝ, 0 < x → f x > f 0) :
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :=
by
  sorry

theorem problem2
  (f : ℝ → ℝ)
  (F : ℝ → ℝ)
  (k : ℝ)
  (cond1 : ∀ x y : ℝ, f (x + y) = f x + f y + 1/2)
  (cond2 : ∀ x : ℝ, 0 < x → f x > f 0)
  (cond3 : F = fun x => f (max (-x) (2*x - x^2)) + f (-k) + 1)
  (hF : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ F x1 = 0 ∧ F x2 = 0 ∧ F x3 = 0) :
  0 < (∑ x in {x1, x2, x3}, x) + (∏ x in {x1, x2, x3}, x) ∧ (∑ x in {x1, x2, x3}, x) + (∏ x in {x1, x2, x3}, x) < 2 :=
by
  sorry

end problem1_problem2_l42_42502


namespace expand_expression_l42_42027

theorem expand_expression (x : ℝ) : 24 * (3 * x + 4 - 2) = 72 * x + 48 :=
by 
  sorry

end expand_expression_l42_42027


namespace possible_ordered_pairs_l42_42317

theorem possible_ordered_pairs (f m : ℕ) (h1 : f + m = 7) (h2 : f ≤ 7) (h3 : m ≤ 7) :
    (f, m) ∈ {(0, 7), (2, 7), (4, 7), (5, 7), (6, 7), (7, 7)} ↔
        ∃ women men : ℕ, women + men = 7 ∧ 
        ((women = 0 ∧ men = 7) ∨ 
         (women = 1 ∧ men = 6 ∧ f = 2 ∧ m = 7) ∨ 
         (women = 2 ∧ men = 5 ∧ ((f = 4 ∧ m = 7) ∨ (f = 4 ∧ m = 7))) ∨ 
         (women = 3 ∧ men = 4 ∧ ((f = 6 ∧ m = 7) ∨ (f = 6 ∧ m = 7))) ∨ 
         (women = 4 ∧ men = 3 ∧ f = 7 ∧ m = 7) ∨ 
         (women = 5 ∧ men = 2 ∧ f = 7 ∧ m = 7) ∨ 
         (women = 6 ∧ men = 1 ∧ f = 7 ∧ m = 7) ∨ 
         (women = 7 ∧ men = 0 ∧ f = 7 ∧ m = 7)) := sorry

end possible_ordered_pairs_l42_42317


namespace value_of_expression_l42_42172

theorem value_of_expression (a b : ℝ) (h1 : 3 * a^2 + 9 * a - 21 = 0) (h2 : 3 * b^2 + 9 * b - 21 = 0) :
  (3 * a - 4) * (5 * b - 6) = -27 :=
by
  -- The proof is omitted, place 'sorry' to indicate it.
  sorry

end value_of_expression_l42_42172


namespace inequality_holds_iff_l42_42864

theorem inequality_holds_iff (a : ℝ) (h_a_pos : 0 < a) :
  (∀ n : ℕ, 0 < n → 
    ∑ k in finset.range n, (a^k / ((1 + 3^k) * (1 + 3^(k+1)))) < 1 / 8) ↔ (a < 3) :=
by
  sorry

end inequality_holds_iff_l42_42864


namespace sum_first_eight_super_nice_numbers_l42_42458

def isPrime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isSuperNice (n : ℕ) : Prop :=
  (∃ p q r : ℕ, isPrime p ∧ isPrime q ∧ isPrime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ n = p * q * r)
  ∨ (∃ p : ℕ, isPrime p ∧ n = p ^ 4)

def firstEightSuperNiceSum : ℕ := 
  16 + 30 + 42 + 66 + 70 + 81 + 105 + 110

theorem sum_first_eight_super_nice_numbers : 
  ∃ l : List ℕ, l.length = 8 ∧ (∀ x ∈ l, isSuperNice x) ∧ l.sum = 520 :=
by
  let l := [16, 30, 42, 66, 70, 81, 105, 110]
  have H_len : l.length = 8 := by decide
  have H_supernice : ∀ x ∈ l, isSuperNice x :=
    by
      intros x H
      fin_cases H <;>
      simp [isSuperNice, isPrime] <;> sorry -- Proof of each item satisfying isSuperNice
  have H_sum : l.sum = 520 := by decide
  exact ⟨l, H_len, H_supernice, H_sum⟩

end sum_first_eight_super_nice_numbers_l42_42458


namespace passengers_got_on_in_Texas_l42_42429

theorem passengers_got_on_in_Texas (start_pax : ℕ) 
  (texas_depart_pax : ℕ) 
  (nc_depart_pax : ℕ) 
  (nc_board_pax : ℕ) 
  (virginia_total_people : ℕ) 
  (crew_members : ℕ) 
  (final_pax_virginia : ℕ) 
  (X : ℕ) :
  start_pax = 124 →
  texas_depart_pax = 58 →
  nc_depart_pax = 47 →
  nc_board_pax = 14 →
  virginia_total_people = 67 →
  crew_members = 10 →
  final_pax_virginia = virginia_total_people - crew_members →
  X + 33 = final_pax_virginia →
  X = 24 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end passengers_got_on_in_Texas_l42_42429


namespace parabola_equation_line_AB_equation_l42_42194

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42194


namespace find_shortest_side_length_l42_42129

noncomputable def length_of_shortest_side (B C : ℝ) (c : ℝ) (hB : B = 45) (hC : C = 60) (hc : c = 1) : ℝ :=
  let A := 180 - B - C in
  let b := (Real.sin (B * (Real.pi / 180))) / (Real.sin (C * (Real.pi / 180))) * c in
  b

theorem find_shortest_side_length (B C : ℝ) (c : ℝ) (hB : B = 45) (hC : C = 60) (hc : c = 1) :
  length_of_shortest_side B C c hB hC hc = sqrt 6 / 3 :=
sorry

end find_shortest_side_length_l42_42129


namespace cricket_game_remaining_overs_l42_42591

theorem cricket_game_remaining_overs
  (target_runs : ℕ)
  (initial_run_rate : ℝ)
  (initial_overs : ℕ)
  (required_run_rate : ℝ)
  (runs_scored_first_5_overs : ℤ)
  (remaining_runs : ℤ)
  (remaining_overs : ℕ): Prop :=
  target_runs = 200 ∧
  initial_run_rate = 2.1 ∧
  initial_overs = 5 ∧
  required_run_rate = 6.316666666666666 ∧
  runs_scored_first_5_overs = 11 ∧
  remaining_runs = target_runs - runs_scored_first_5_overs ∧
  remaining_overs = int.ceil (remaining_runs / required_run_rate) →
  remaining_overs = 30

end cricket_game_remaining_overs_l42_42591


namespace no_common_factor_l42_42318

open Polynomial

theorem no_common_factor (f g : ℤ[X]) : f = X^2 + X - 1 → g = X^2 + 2 * X → ∀ d : ℤ[X], d ∣ f ∧ d ∣ g → d = 1 :=
by
  intros h1 h2 d h_dv
  rw [h1, h2] at h_dv
  -- Proof steps would go here
  sorry

end no_common_factor_l42_42318


namespace simplify_expression_l42_42653

theorem simplify_expression (n : ℕ) : 
  (3^(n + 3) - 3 * 3^n) / (3 * 3^(n + 2)) = 8 / 3 := 
sorry

end simplify_expression_l42_42653


namespace original_function_l42_42997

def f (x : ℝ) : ℝ := sin (x - π / 4)

theorem original_function :
  ∃ g : ℝ → ℝ, (∀ x, f (2 * (x + π / 3)) = sin (x - π / 4)) ∧ g x = sin (x / 2 + π / 12) :=
sorry

end original_function_l42_42997


namespace min_value_of_function_range_of_x_l42_42081

noncomputable def func (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem min_value_of_function (a b : ℤ) (c : ℤ) (h1 : a > 0) (h2 : b > 2 * a)
  (h3 : func a b c (Int.sin x) = -4 ∨ func a b c (Int.sin x) = 2) :
  ∃ x, func a b c x = -17 / 4 :=
sorry

theorem range_of_x (a b : ℤ) (c : ℤ) (m : ℤ) (h1 : a = 1) (h2 : b = 3) (h3 : c = -2)
  (h4 : ∀ m ∈ [-4, 1], func a b c x ≥ (2 * x^2 - m * x - 14) ) :
  ∀ x, x ∈ [-2, 3] :=
sorry

end min_value_of_function_range_of_x_l42_42081


namespace fg_difference_l42_42521

def f (x : ℝ) : ℝ := x^2 - 4 * x + 7
def g (x : ℝ) : ℝ := x + 4

theorem fg_difference : f (g 3) - g (f 3) = 20 :=
by
  sorry

end fg_difference_l42_42521


namespace compare_y_values_l42_42535

variable (k : ℝ)
def quad (x : ℝ) : ℝ := -2 * x^2 + 4 * x + k

def x1 := -0.99
def x2 := 0.98
def x3 := 0.99

def y1 := quad k x1
def y2 := quad k x2
def y3 := quad k x3

theorem compare_y_values : y1 k < y2 k ∧ y2 k < y3 k := 
by
  -- carry out the necessary proof steps here
  sorry 

end compare_y_values_l42_42535


namespace theta_plus_2phi_l42_42897

variable (θ φ : ℝ)
variables (hθ : 0 < θ ∧ θ < π/2) (hφ : 0 < φ ∧ φ < π/2)
variables (h₁ : tan θ = 3 / 5) (h₂ : sin φ = 2 / Real.sqrt 5)

theorem theta_plus_2phi (θ φ : ℝ) (hθ : 0 < θ ∧ θ < π/2) (hφ : 0 < φ ∧ φ < π/2)
  (h₁ : tan θ = 3 / 5) (h₂ : sin φ = 2 / Real.sqrt 5) : θ + 2 * φ = π - Real.arctan (11 / 27) := 
sorry

end theta_plus_2phi_l42_42897


namespace june_first_2012_friday_l42_42851

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def days_in_month (month : ℕ) (is_leap_year : Prop) : ℕ :=
  if month = 2 then (if is_leap_year then 29 else 28) 
  else if month = 4 ∨ month = 6 ∨ month = 9 ∨ month = 11 then 30 
  else 31

theorem june_first_2012_friday :
  ∀ (start_day : ℕ), start_day = 3 → is_leap_year 2012 →
  let february_days := days_in_month 2 true,
      march_days := days_in_month 3 true,
      april_days := days_in_month 4 true,
      may_days := days_in_month 5 true,
      total_days := february_days - 1 + march_days + april_days + may_days + 1 in
      (total_days % 7 + start_day) % 7 = 5 :=
by
  intros start_day start_day_condition leap_year_2012
  let february_days := days_in_month 2 leap_year_2012 = 29
  let march_days := days_in_month 3 leap_year_2012 = 31
  let april_days := days_in_month 4 leap_year_2012 = 30
  let may_days := days_in_month 5 leap_year_2012 = 31
  let total_days := february_days - 1 + march_days + april_days + may_days + 1
  have rem_days := total_days % 7
  have week_day := (rem_days + start_day) % 7
  have friday := week_day = 5
  sorry

end june_first_2012_friday_l42_42851


namespace general_term_formula_sum_and_min_value_l42_42142

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (n : ℕ)

-- Conditions from the problem
axiom h1 : S 9 = -9
axiom h2 : S 10 = -5

-- General term for the sequence
def a_n := n - 6

-- Sequence sum function
def S_n := (n:ℤ) * (a 1 + a n) / 2

-- Prove general term
theorem general_term_formula : a  = a_n := 
by
  sorry

-- Prove the sum and its minimum value
theorem sum_and_min_value : 
  ∀ n, S n = (1/2 : ℚ) * (n - 11 / 2) ^ 2 - 121 / 8 ∧ (S n = -15 ↔ n = 5 ∨ n = 6) := 
by
  sorry

end general_term_formula_sum_and_min_value_l42_42142


namespace arithmetic_sequence_solution_l42_42166

-- Definitions of a, b, c, and d in terms of d and sequence difference
def is_in_arithmetic_sequence (a b c d : ℝ) (diff : ℝ) : Prop :=
  a + diff = b ∧ b + diff = c ∧ c + diff = d

-- Conditions
def pos_real_sequence (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

def product_condition (a b c d : ℝ) (prod : ℝ) : Prop :=
  a * b * c * d = prod

-- The resulting value of d
def d_value_as_fraction (d : ℝ) : Prop :=
  d = (3 + Real.sqrt 95) / (Real.sqrt 2)

-- Proof statement
theorem arithmetic_sequence_solution :
  ∃ a b c d : ℝ, pos_real_sequence a b c d ∧ 
                 is_in_arithmetic_sequence a b c d (Real.sqrt 2) ∧ 
                 product_condition a b c d 2021 ∧ 
                 d_value_as_fraction d :=
sorry

end arithmetic_sequence_solution_l42_42166


namespace problem_1_problem_2_l42_42204

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42204


namespace chandler_bike_purchase_weeks_l42_42826

theorem chandler_bike_purchase_weeks (bike_cost birthday_money weekly_earnings total_weeks : ℕ) 
  (h_bike_cost : bike_cost = 600)
  (h_birthday_money : birthday_money = 60 + 40 + 20 + 30)
  (h_weekly_earnings : weekly_earnings = 18)
  (h_total_weeks : total_weeks = 25) :
  birthday_money + weekly_earnings * total_weeks = bike_cost :=
by {
  sorry
}

end chandler_bike_purchase_weeks_l42_42826


namespace factorize_poly_l42_42728

theorem factorize_poly :
  (x : ℤ) → x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) :=
by
  sorry

end factorize_poly_l42_42728


namespace max_value_of_reciprocals_of_zeros_l42_42078

noncomputable def f (k : ℝ) : ℝ → ℝ :=
  λ x, if x ∈ Ioc 0 1 then k * x^2 + 2 * x - 1 else k * x + 1

theorem max_value_of_reciprocals_of_zeros (k : ℝ) (h : k < 0) (h_pos : k > -1) :
  let x1 := -1 / k in
  let x2 := 1 / (1 + real.sqrt (1 + k)) in
  1 / x1 + 1 / x2 ≤ 9 / 4 :=
begin
  sorry
end

end max_value_of_reciprocals_of_zeros_l42_42078


namespace new_winning_percentage_l42_42133

noncomputable def initial_matches : ℕ := 120
noncomputable def initial_win_percentage : ℝ := 0.28
noncomputable def additional_wins : ℕ := 60

theorem new_winning_percentage :
  let initial_wins := (initial_win_percentage * initial_matches).round
  let total_wins := initial_wins + additional_wins
  let total_matches := initial_matches + additional_wins
  (total_wins : ℝ) / total_matches * 100 ≈ 52.22 :=
by
  sorry

end new_winning_percentage_l42_42133


namespace number_of_valid_starting_lineups_l42_42331

-- Define the total number of players
def total_players : ℕ := 15

-- Define the specific players Leo, Max, and Neo
def Leo : ℕ := 1
def Max : ℕ := 2
def Neo : ℕ := 3

-- Define the number of players needed for the starting lineup
def starting_lineup_size : ℕ := 6

-- Define the combinations count function
def count_combinations (n k : ℕ) : ℕ := (Finset.range n).choose k

-- Calculating the number of valid lineups
def count_valid_lineups : ℕ :=
  count_combinations 12 5 +   -- Case 1: Leo starts, Max and Neo don't.
  count_combinations 12 5 +   -- Case 2: Max starts, Leo and Neo don't.
  count_combinations 12 5 +   -- Case 3: Neo starts, Leo and Max don't.
  count_combinations 12 6     -- Case 4: None of Leo, Max, or Neo start.

-- The theorem to prove the correct answer
theorem number_of_valid_starting_lineups : count_valid_lineups = 3300 :=
by
  -- This will eventually contain the detailed proof
  sorry

end number_of_valid_starting_lineups_l42_42331


namespace sequence_sum_l42_42005

def sum_sequence (n : ℕ) : ℤ :=
    if n % 4 = 1 then n 
    else if n % 4 = 2 then -n
    else if n % 4 = 3 then -n
    else n

theorem sequence_sum :
    (finset.range 2028).sum sum_sequence = 0 :=
  sorry

end sequence_sum_l42_42005


namespace prove_L_length_l42_42561

variable (L : ℝ) -- length of the cloth that the first group of men can color
variable (t₁ t₂ : ℝ) -- time in days for the first group and the second group respectively
variable (n₁ n₂ : ℕ) -- number of men in the first group and the second group respectively
variable (length₂ : ℝ) -- length of the cloth for the second group

-- Conditions:
-- 4 men can color a certain length L of cloth in 2 days
-- 6 men can color a 36 m long cloth in 1 day
def condition_one : Prop := n₁ = 4 ∧ t₁ = 2
def condition_two : Prop := n₂ = 6 ∧ length₂ = 36 ∧ t₂ = 1

-- Prove:
theorem prove_L_length (h1 : condition_one L t₁ n₁) (h2 : condition_two length₂ t₂ n₂) : L = 48 :=
sorry

end prove_L_length_l42_42561


namespace count_even_numbers_is_320_l42_42935

noncomputable def count_even_numbers_with_distinct_digits : Nat := 
  let unit_choices := 5  -- Choices for the unit digit (0, 2, 4, 6, 8)
  let hundreds_choices := 8  -- Choices for the hundreds digit (1 to 9, excluding the unit digit)
  let tens_choices := 8  -- Choices for the tens digit (0 to 9, excluding the hundreds and unit digit)
  unit_choices * hundreds_choices * tens_choices

theorem count_even_numbers_is_320 : count_even_numbers_with_distinct_digits = 320 := by
  sorry

end count_even_numbers_is_320_l42_42935


namespace infinite_solutions_l42_42871

theorem infinite_solutions (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by
  sorry

end infinite_solutions_l42_42871


namespace train_speed_is_25_kmph_l42_42426

noncomputable def train_speed_kmph (train_length_m : ℕ) (man_speed_kmph : ℕ) (cross_time_s : ℕ) : ℕ :=
  let man_speed_mps := (man_speed_kmph * 1000) / 3600
  let relative_speed_mps := train_length_m / cross_time_s
  let train_speed_mps := relative_speed_mps - man_speed_mps
  let train_speed_kmph := (train_speed_mps * 3600) / 1000
  train_speed_kmph

theorem train_speed_is_25_kmph : train_speed_kmph 270 2 36 = 25 := by
  sorry

end train_speed_is_25_kmph_l42_42426


namespace common_point_circles_l42_42165

-- Definitions of the points and necessary conditions.
variables {A B C O P D S : Point}
variables (h1 : is_circumcenter O A B C) 
variables (h2 : circle P passes_through A ∧ circle P passes_through O)
variables (h3 : parallel OP BC)
variables (h4 : angle DBA = angle DCA ∧ angle DCA = angle BAC)

-- Theorem statement.
theorem common_point_circles 
  (h1 : is_circumcenter O A B C) 
  (h2 : circle P passes_through A ∧ circle P passes_through O)
  (h3 : parallel OP BC)
  (h4 : angle DBA = angle DCA ∧ angle DCA = angle BAC) :
  ∃ S, S ∈ circle P ∧ S ∈ circle BCD ∧ S ∈ circle_with_diameter A D :=
sorry

end common_point_circles_l42_42165


namespace problem_1_problem_2_l42_42689

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h₀ : a = 1) (h₁ : ∀ x : ℝ, x^2 - 5 * a * x + 4 * a^2 < 0)
                                    (h₂ : ∀ x : ℝ, (x - 2) * (x - 5) < 0) :
  ∃ x : ℝ, 2 < x ∧ x < 4 :=
by sorry

-- Proof Problem 2
theorem problem_2 (p q : ℝ → Prop) (h₀ : ∀ x : ℝ, p x → q x) 
                                (p_def : ∀ (a : ℝ) (x : ℝ), 0 < a → p x ↔ a < x ∧ x < 4 * a) 
                                (q_def : ∀ x : ℝ, q x ↔ 2 < x ∧ x < 5) :
  ∃ a : ℝ, (5 / 4) ≤ a ∧ a ≤ 2 :=
by sorry

end problem_1_problem_2_l42_42689


namespace odd_terms_zero_l42_42017

variables {α : Type} [Field α]

def sequence (a : ℕ → α) : ℕ → α
| 1 := a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 
| n := (a 1 ^ n) + (a 2 ^ n) + (a 3 ^ n) + (a 4 ^ n) + (a 5 ^ n) + (a 6 ^ n) + (a 7 ^ n) + (a 8 ^ n)

theorem odd_terms_zero (a : ℕ → α) (ha_nonzero : ∃ i, i ∈ (finset.range 9) ∧ a i ≠ 0):
  (∀ n : ℕ, sequence a n = 0 ↔ odd n) :=
by
  sorry

end odd_terms_zero_l42_42017


namespace min_value_of_function_l42_42055

theorem min_value_of_function (x : ℝ) (h: x > 1) :
  ∃ t > 0, x = t + 1 ∧ (t + 3 / t + 3) = 3 + 2 * Real.sqrt 3 :=
sorry

end min_value_of_function_l42_42055


namespace value_of_a_l42_42913

noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - 3 * x - Real.log x + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem value_of_a (a x0 : ℝ) (h : f x0 a = 3) : a = 1 - Real.log 2 :=
by
  sorry

end value_of_a_l42_42913


namespace part1_part2_l42_42903

noncomputable def f (x : ℝ) : ℝ := (Real.exp (-x) - Real.exp x) / 2

theorem part1 (h_odd : ∀ x, f (-x) = -f x) (g : ℝ → ℝ) (h_even : ∀ x, g (-x) = g x)
  (h_g_def : ∀ x, g x = f x + Real.exp x) :
  ∀ x, f x = (Real.exp (-x) - Real.exp x) / 2 := sorry

theorem part2 : {x : ℝ | f x ≥ 3 / 4} = {x | x ≤ -Real.log 2} := sorry

end part1_part2_l42_42903


namespace dice_surface_sum_l42_42756

noncomputable def surface_sum_of_dice (n : ℕ) := 28175 + 2 * n

theorem dice_surface_sum (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6):
  ∃ s, s ∈ {28177, 28179, 28181, 28183, 28185, 28187} ∧ s = surface_sum_of_dice X := by
  use surface_sum_of_dice X
  have : 28175 + 2 * X ∈ {28177, 28179, 28181, 28183, 28185, 28187} := by
    interval_cases X
    all_goals simp [surface_sum_of_dice]
  exact ⟨this, rfl⟩
sorry

end dice_surface_sum_l42_42756


namespace option_C_correct_l42_42174

variables {m n l : Type}
variables [linear_order m] [linear_order n] [linear_order l]
variables (α : Type) [plane α]

-- Conditions
variables {m_perp_n : m ⊥ n}
variables {n_in_alpha : n ⊂ α}
variables {m_parallel_n : m ∥ n}

-- Conclusion
theorem option_C_correct (h1 : m ∥ n) (h2 : n ⊥ α) : m ⊥ α :=
sorry

end option_C_correct_l42_42174


namespace fraction_addition_l42_42945

theorem fraction_addition (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end fraction_addition_l42_42945


namespace initial_amount_calc_l42_42711

theorem initial_amount_calc 
  (M : ℝ)
  (H1 : M * 0.3675 = 350) :
  M = 952.38 :=
by
  sorry

end initial_amount_calc_l42_42711


namespace convex_pentagon_medians_not_collinear_non_convex_pentagon_medians_collinear_l42_42392

-- Definition for convex pentagon
structure ConvexPentagon (A B C D E : Point) : Prop :=
(convex : /* definition stating A, B, C, D, E form a convex pentagon */)

-- Definition for non-convex pentagon
structure NonConvexPentagon (A B C D E : Point) : Prop :=
(non_convex : /* definition stating A, B, C, D, E form a non-convex pentagon */)

-- Medians intersecting at centroids
def centroid (T : Triangle) : Point := /* definition for centroid */

-- Statement for convex pentagon
theorem convex_pentagon_medians_not_collinear (A B C D E : Point) 
  (h : ConvexPentagon A B C D E) : 
  ¬ collinear (centroid (Triangle A B E)) (centroid (Triangle A D E)) (centroid (Triangle A D C)) :=
sorry

-- Statement for non-convex pentagon
theorem non_convex_pentagon_medians_collinear (A B C D E : Point) 
  (h : NonConvexPentagon A B C D E) : 
  collinear (centroid (Triangle A B E)) (centroid (Triangle A D E)) (centroid (Triangle A D C)) :=
sorry

end convex_pentagon_medians_not_collinear_non_convex_pentagon_medians_collinear_l42_42392


namespace greatest_individual_score_l42_42771

theorem greatest_individual_score (points : ℕ) (players : Fin 12 → ℕ) 
  (h_team_size : ∑ i, 1 = 12)
  (h_team_points : ∑ i, players i = 100)
  (h_min_score : ∀ i, players i ≥ 7) : 
  ∃ i, players i = 23 :=
by
  sorry

end greatest_individual_score_l42_42771


namespace equation_of_parabola_equation_of_line_AB_l42_42252

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42252


namespace transformed_function_is_correct_l42_42972

noncomputable def f : ℝ → ℝ := λ x, sin (x / 2 + π / 12)

theorem transformed_function_is_correct :
  ∀ x : ℝ, sin (2 * (x - π / 3)) = sin (x - π / 4) :=
by
  intros x
  symmetry
  apply eq.symm
  calc
    sin (2 * (x - π / 3)) = sin (2x - 2 * π / 3) : by rw mul_sub
    _ = sin (x - π / 4) : sorry

end transformed_function_is_correct_l42_42972


namespace biloca_path_proof_l42_42046

def diagonal_length := 5 -- Length of one diagonal as deduced from Pipoca's path
def tile_width := 3 -- Width of one tile as deduced from Tonica's path
def tile_length := 4 -- Length of one tile as deduced from Cotinha's path

def Biloca_path_length : ℝ :=
  3 * diagonal_length + 4 * tile_width + 2 * tile_length

theorem biloca_path_proof :
  Biloca_path_length = 43 :=
by
  sorry

end biloca_path_proof_l42_42046


namespace determine_c_l42_42022

noncomputable def fib (n : ℕ) : ℕ :=
match n with
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem determine_c (c d : ℤ) (h1 : ∃ s : ℂ, s^2 - s - 1 = 0 ∧ (c : ℂ) * s^19 + (d : ℂ) * s^18 + 1 = 0) : 
  c = 1597 :=
by
  sorry

end determine_c_l42_42022


namespace perpendicular_lines_l42_42541

-- Definitions of lines m, n and planes α, β
variable (m n : Line)
variable (α β : Plane)

-- Conditions
variable (h1 : m ⊥ α)
variable (h2 : n ⊥ β)
variable (h3 : α ⊥ β)

-- Theorem statement
theorem perpendicular_lines (m n : Line) (α β : Plane) 
  (h1 : m ⊥ α) (h2 : n ⊥ β) (h3 : α ⊥ β) : m ⊥ n :=
sorry

end perpendicular_lines_l42_42541


namespace necessary_but_not_sufficient_condition_l42_42351

variable (m : ℝ)

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def quadratic_has_two_distinct_roots (a b c : ℝ) : Prop :=
  discriminant a b c > 0

def equation := x^2 - real.sqrt m * x + 1 = 0

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (m > 4 → m > 2) ∧ (m > 2 → ¬ (m > 4)) → 
  quadratic_has_two_distinct_roots 1 (-real.sqrt m) 1 :=
begin
  sorry
end

end necessary_but_not_sufficient_condition_l42_42351


namespace trigonometric_identity_l42_42171

theorem trigonometric_identity (a b : ℝ) 
    (h1 : (sin a / cos b) + (sin b / cos a) = 2)
    (h2 : (cos a / sin b) + (cos b / sin a) = 4) :
    (tan a / tan b) + (tan b / tan a) = 2 :=
sorry

end trigonometric_identity_l42_42171


namespace dice_sum_surface_l42_42763

theorem dice_sum_surface (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6) : 
  ∃ Y : ℕ, Y = 28175 + 2 * X ∧ (Y = 28177 ∨ Y = 28179 ∨ Y = 28181 ∨ Y = 28183 ∨ 
  Y = 28185 ∨ Y = 28187) :=
by
  sorry

end dice_sum_surface_l42_42763


namespace cubic_inches_in_one_cubic_foot_l42_42107

theorem cubic_inches_in_one_cubic_foot (h : 1.foot = 12.inches) : (1.foot)^3 = 1728 * (1.inches)^3 :=
by
  rw [h]
  calc (12.foot)^3 = 12^3 * (1.inches)^3 : sorry

end cubic_inches_in_one_cubic_foot_l42_42107


namespace b_2015_is_neg1_l42_42572

def sequence_b : ℕ → ℚ
| 1     := 2
| (n+1) := 1 / (1 - sequence_b n)

theorem b_2015_is_neg1 : 
  sequence_b 2015 = -1 :=
sorry

end b_2015_is_neg1_l42_42572


namespace sqrt_expression_to_int_sum_eq_117_l42_42573

theorem sqrt_expression_to_int_sum_eq_117 : 
  ∃ (a b c : ℕ), 
    (0 < c) ∧ 
    (\sqrt{5} + \frac{1}{\sqrt{5}} + \sqrt{7} + \frac{1}{\sqrt{7}} = \frac{a\sqrt{5} + b\sqrt{7}}{c}) ∧ 
    (a + b + c = 117) :=
sorry

end sqrt_expression_to_int_sum_eq_117_l42_42573


namespace find_number_l42_42718

theorem find_number (x : ℝ) (h : x / 3 = x - 4) : x = 6 := 
by 
  sorry

end find_number_l42_42718


namespace rabbit_calories_l42_42821

theorem rabbit_calories (C : ℕ) :
  (6 * 300 = 2 * C + 200) → C = 800 :=
by
  intro h
  sorry

end rabbit_calories_l42_42821


namespace tangent_line_at_point_l42_42341

theorem tangent_line_at_point (k a b : ℝ) :
  (∀ x, deriv (λ x: ℝ, x^2 + a / x + 1) 2 = k) ∧ 
  (2^2 + a / 2 + 1 = 3) ∧ 
  (3 = 2 * k + b) → 
  (b = -7) :=
by
  intros h
  sorry

end tangent_line_at_point_l42_42341


namespace steve_marbles_l42_42311

theorem steve_marbles {S : ℤ} (sam_initial : ℤ) (sally_initial : ℤ) :
  (sam_initial = 2 * S) →
  (sally_initial = 2 * S - 5) →
  (sam_initial - 3 - 3 = 8) →
  (S + 3 = 10) :=
begin
  intros h1 h2 h3,
  sorry
end

end steve_marbles_l42_42311


namespace theta_range_l42_42906

noncomputable def f (x θ : ℝ) : ℝ :=
  1 / (x - 2)^2 - 2 * x + Real.cos (2 * θ) - 3 * Real.sin θ + 2

def pos_for_all_x (θ : ℝ) : Prop :=
  ∀ (x : ℝ), x < 2 → f x θ > 0

theorem theta_range {θ : ℝ} (h1 : 0 < θ) (h2 : θ < Real.pi) :
  pos_for_all_x θ ↔ (0 < θ ∧ θ < Real.pi / 6) :=
begin
  sorry
end

end theta_range_l42_42906


namespace equality_of_ratios_l42_42873

-- Define the points and segments
variables {α : Type*} [OrderedRing α] {A B C D A1 B1 C1 D1 : α}

-- Define the pencil of lines
-- In Lean, we might represent conditions geometrically using affine spaces or vector spaces, but here it is simplified.
def pencil_lines (P Q R S : α) : Prop :=
  ∃ k l m n : α, P = k + l ∧ Q = k + m ∧ R = k + n ∧ S = k + n * l

theorem equality_of_ratios 
  (h_pencil_ABCD : pencil_lines A B C D) 
  (h_pencil_A1B1C1D1 : pencil_lines A1 B1 C1 D1) :
  (AC / CB) / (AD / DB) = (A1C1 / C1B1) / (A1D1 / D1B1) :=
sorry

end equality_of_ratios_l42_42873


namespace area_covered_even_inequality_odd_inequality_l42_42885

variable (n : ℕ)
variable (S : ℝ)
variable (S_i : (finset (fin n) → ℝ))
variable (M : (ℕ → ℝ))

def covers_plane (S : ℝ) (M : ℕ → ℝ) :=
  S = ∑ k in (finset.range n).filter (λ k, k % 2 = 0), M (k + 1) - ∑ k in (finset.range n).filter (λ k, k % 2 = 1), M (k + 1)

def even_condition (S : ℝ) (M : ℕ → ℝ) (m : ℕ) : Prop :=
  m % 2 = 0 → S ≥ ∑ k in (finset.range m).filter (λ k, k % 2 = 0), M (k + 1) - ∑ k in (finset.range m).filter (λ k, k % 2 = 1), M (k + 1)

def odd_condition (S : ℝ) (M : ℕ → ℝ) (m : ℕ) : Prop :=
  m % 2 = 1 → S ≤ ∑ k in (finset.range m).filter (λ k, k % 2 = 0), M (k + 1) - ∑ k in (finset.range m).filter (λ k, k % 2 = 1), M (k + 1)

theorem area_covered (n : ℕ) (S : ℝ) (S_i : (finset (fin n) → ℝ)) (M : (ℕ → ℝ)) :
  covers_plane S M := sorry

theorem even_inequality (n : ℕ) (S : ℝ) (S_i : (finset (fin n) → ℝ)) (M : (ℕ → ℝ)) (m : ℕ) :
  even_condition S M m := sorry

theorem odd_inequality (n : ℕ) (S : ℝ) (S_i : (finset (fin n) → ℝ)) (M : (ℕ → ℝ)) (m : ℕ) :
  odd_condition S M m := sorry

end area_covered_even_inequality_odd_inequality_l42_42885


namespace solve_digit_addition_problem_l42_42144

theorem solve_digit_addition_problem :
    ∃ N O E T W S I X : ℕ,
      N ≠ O ∧ N ≠ E ∧ N ≠ T ∧ N ≠ W ∧ N ≠ S ∧ N ≠ I ∧ N ≠ X ∧
      O ≠ E ∧ O ≠ T ∧ O ≠ W ∧ O ≠ S ∧ O ≠ I ∧ O ≠ X ∧
      E ≠ T ∧ E ≠ W ∧ E ≠ S ∧ E ≠ I ∧ E ≠ X ∧
      T ≠ W ∧ T ≠ S ∧ T ≠ I ∧ T ≠ X ∧
      W ≠ S ∧ W ≠ I ∧ W ≠ X ∧ S ≠ I ∧ S ≠ X ∧ I ≠ X ∧
      (N ∈ finset.range 8) ∧ (N + 2 ∈ finset.range 8) ∧
      (O ∈ finset.range 8) ∧ (O + 2 ∈ finset.range 8) ∧
      (E ∈ finset.range 8) ∧ (E + 2 ∈ finset.range 8) ∧
      (T ∈ finset.range 8) ∧ (T + 2 ∈ finset.range 8) ∧
      (W ∈ finset.range 8) ∧ (W + 2 ∈ finset.range 8) ∧
      (S ∈ finset.range 8) ∧ (S + 2 ∈ finset.range 8) ∧
      (I ∈ finset.range 8) ∧ (I + 2 ∈ finset.range 8) ∧
      (X ∈ finset.range 8) ∧ (X + 2 ∈ finset.range 8) ∧
      (100 * N +  10 * I + N * 1 + E = 1000 * N + 100 * I + N * 10 + E) ∧
      (E + N = 2526) :=
by
  sorry

end solve_digit_addition_problem_l42_42144


namespace wade_tips_l42_42372

/-- Wade has a hot dog food truck. 
     He makes $2.00 in tips per customer.
     On Friday he served 28 customers.
     He served three times that amount of customers on Saturday.
     On Sunday, he served 36 customers.
     Prove that Wade made $296 in tips between the 3 days. -/
theorem wade_tips : 
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let customers_sunday := 36
  let tips_friday := tips_per_customer * customers_friday
  let tips_saturday := tips_per_customer * customers_saturday
  let tips_sunday := tips_per_customer * customers_sunday
  let total_tips := tips_friday + tips_saturday + tips_sunday
  in total_tips = 296 := 
by
  sorry

end wade_tips_l42_42372


namespace song_assignments_l42_42493

/-
Mathematical equivalent proof problem in Lean:
Given four friends Amy, Beth, Jo, and Kim, and five different songs such that:
1. Every song is liked by exactly two of the friends.
2. For each trio of the friends, there is at least one song liked by exactly two of them but not by the third in the trio.
We need to prove that there are exactly 15 different ways to assign the songs.
-/

def girls := {Amy, Beth, Jo, Kim} : set String
def trios (g : set String) := {t | ∃ (x y z : String), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ t = {x, y, z}} (filter (λ s, set.card s = 3) (powerset g))
def likes (s : set String) := {pair | ∃ (x y : String), x ≠ y ∧ pair = (x, y) ∧ pair ∈ s}

theorem song_assignments : 
  ∃ f : Fin 5 → set (String × String), 
    (∀ i, ∃ x y : String, x ≠ y ∧ f i = ({x, y} : set (String × String))) ∧
    (∀ t ∈ trios girls, ∃ i, ∃ x y : String, x ≠ y ∧ {x, y} ∩ t ≠ ∅ ∧ f i = ({x, y} : set (String × String))) ∧
    set.card {f i | i : Fin 5} = 15 := 
sorry

end song_assignments_l42_42493


namespace identify_incorrect_propositions_l42_42551

-- Definitions for parallel lines and planes
def line := Type -- Define a line type
def plane := Type -- Define a plane type
def parallel_to (l1 l2 : line) : Prop := sorry -- Assume a definition for parallel lines
def parallel_to_plane (l : line) (pl : plane) : Prop := sorry -- Assume a definition for a line parallel to a plane
def contained_in (l : line) (pl : plane) : Prop := sorry -- Assume a definition for a line contained in a plane

theorem identify_incorrect_propositions (a b : line) (α : plane) :
  (parallel_to_plane a α ∧ parallel_to_plane b α → ¬parallel_to a b) ∧
  (parallel_to_plane a α ∧ contained_in b α → ¬parallel_to a b) ∧
  (parallel_to a b ∧ contained_in b α → ¬parallel_to_plane a α) ∧
  (parallel_to a b ∧ parallel_to_plane b α → ¬parallel_to_plane a α) :=
by
  sorry -- The proof is not required

end identify_incorrect_propositions_l42_42551


namespace new_releases_fraction_is_2_over_5_l42_42384

def fraction_new_releases (total_books : ℕ) (frac_historical_fiction : ℚ) (frac_new_historical_fiction : ℚ) (frac_new_non_historical_fiction : ℚ) : ℚ :=
  let num_historical_fiction := frac_historical_fiction * total_books
  let num_new_historical_fiction := frac_new_historical_fiction * num_historical_fiction
  let num_non_historical_fiction := total_books - num_historical_fiction
  let num_new_non_historical_fiction := frac_new_non_historical_fiction * num_non_historical_fiction
  let total_new_releases := num_new_historical_fiction + num_new_non_historical_fiction
  num_new_historical_fiction / total_new_releases

theorem new_releases_fraction_is_2_over_5 :
  ∀ (total_books : ℕ), total_books > 0 →
    fraction_new_releases total_books (40 / 100) (40 / 100) (40 / 100) = 2 / 5 :=
by 
  intro total_books h
  sorry

end new_releases_fraction_is_2_over_5_l42_42384


namespace percentage_decrease_in_demand_l42_42417

variable (P_old D_old : ℝ)

def P_new : ℝ := 1.30 * P_old
def I_old : ℝ := P_old * D_old
def I_new : ℝ := 1.10 * I_old
def D_new : ℝ := (1.10 / 1.30) * D_old

theorem percentage_decrease_in_demand :
    let d := 1 - (D_new / D_old) in
    d * 100 = 15.38 :=
by
  unfold D_new I_old I_new P_new
  sorry

end percentage_decrease_in_demand_l42_42417


namespace cubic_foot_to_cubic_inches_l42_42103

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end cubic_foot_to_cubic_inches_l42_42103


namespace simplify_and_evaluate_l42_42320

theorem simplify_and_evaluate (a : ℕ) (h : a = 2) : 
  (1 - (1 : ℚ) / (a + 1)) / (a / ((a * a) - 1)) = 1 := by
  sorry

end simplify_and_evaluate_l42_42320


namespace sum_infallible_correct_l42_42416

def is_infallible (n : ℕ) : Prop :=
  n ≥ 3 ∧ n ≤ 100 ∧ (
    (n % 2 = 1 ∧ 100 % n = 0) ∨
    (n % 2 = 0 ∧ 100 % (n / 2) = 0)
  )

def sum_infallible_integers : ℕ :=
  (Finset.filter is_infallible (Finset.range 101)).sum id

theorem sum_infallible_correct : sum_infallible_integers = 262 := sorry

end sum_infallible_correct_l42_42416


namespace transformation_correct_l42_42976

noncomputable def original_function : ℝ → ℝ :=
  λ x, sin (x / 2 + π / 12)

theorem transformation_correct :
  ∀ (x : ℝ), (shift_right (shorten_abscissas f) (π / 3)) x = sin (x - π / 4) →
  f = original_function :=
by
  -- Assume the shifted and shortened function equals the given sine transformation
  intro x
  sorry

end transformation_correct_l42_42976


namespace measure_angle_DAE_l42_42431

-- Definitions based on problem conditions
variables (A B C D E F : Type*) [metric_space A]

def equilateral_triangle (A B C : A) := 
  let side_length := dist A B in
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

def regular_pentagon (B C D E F : A) := 
  let side_length := dist B C in
  dist B C = side_length ∧ dist C D = side_length ∧ dist D E = side_length ∧ dist E F = side_length ∧ dist F B = side_length ∧ 
  ∀ (P Q R : A), (P = B ∨ P = C ∨ P = D ∨ P = E ∨ P = F) → (Q = B ∨ Q = C ∨ Q = D ∨ Q = E ∨ Q = F) → (R = B ∨ R = C ∨ R = D ∨ R = E ∨ R = F) → angle P Q R ∈ ({108, 72} : set real)

-- Theorem to prove based on conditions
theorem measure_angle_DAE {A B C D E F : A} (h_triangle : equilateral_triangle A B C) (h_pentagon: regular_pentagon B C D E F): 
  ∀ (u v w x y z : A), angle D A E = 12 := 
sorry

end measure_angle_DAE_l42_42431


namespace inequality_transform_l42_42117

theorem inequality_transform {a b : ℝ} (h : a < b) : -2 + 2 * a < -2 + 2 * b :=
sorry

end inequality_transform_l42_42117


namespace ratio_of_areas_l42_42163

noncomputable def S1 : Set (ℝ × ℝ) := 
  { p | log 10 (3 + p.1^2 + p.2^2) ≤ 1 + log 10 (p.1 + 2 * p.2) }

noncomputable def S2 : Set (ℝ × ℝ) := 
  { p | log 10 (4 + p.1^2 + p.2^2) ≤ 2 + log 10 (2 * p.1 + 3 * p.2) }

theorem ratio_of_areas : 
  let A1 := π * 122
  let A2 := π * 32496
  (A2 / A1 = 266) :=
by
  sorry

end ratio_of_areas_l42_42163


namespace sequence_sum_l42_42889

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 2
  else 2 * (sequence (n - 1)) - 1

theorem sequence_sum : (∑ i in Finset.range 10, sequence (i + 1)) = 1033 :=
begin
  sorry
end

end sequence_sum_l42_42889


namespace part1_eq_C_part2_line_AB_l42_42283

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42283


namespace smallest_guesses_to_find_two_digit_number_l42_42437

theorem smallest_guesses_to_find_two_digit_number : 
  ∃ n : ℕ, (∀ picked : ℕ, 10 ≤ picked ∧ picked ≤ 99 → ∃! guesses : list ℕ, list.length guesses ≤ n ∧ (∀ guess ∈ guesses, guess ∈ range 10 100) ∧ (guess_digits_match picked guess)) ∧ n = 10 :=
begin
  sorry
end

end smallest_guesses_to_find_two_digit_number_l42_42437


namespace polynomial_has_real_root_l42_42643

theorem polynomial_has_real_root (a b : ℝ) :
  ∃ x : ℝ, x^3 + a * x + b = 0 :=
sorry

end polynomial_has_real_root_l42_42643


namespace average_time_rounded_to_4_l42_42367

def train1_distance : ℝ := 200
def train1_speed : ℝ := 50
def train2_distance : ℝ := 240
def train2_speed : ℝ := 80

def time_train1 := train1_distance / train1_speed
def time_train2 := train2_distance / train2_speed

def total_time := time_train1 + time_train2
def average_time := total_time / 2
def rounded_average_time := Real.ceil (average_time + 0.5) - 1

theorem average_time_rounded_to_4 : rounded_average_time = 4 := 
by sorry

end average_time_rounded_to_4_l42_42367


namespace maximize_profit_l42_42843
-- Importing the entire necessary library

-- Definitions and conditions
def cost_price : ℕ := 40
def minimum_selling_price : ℕ := 44
def maximum_profit_margin : ℕ := 30
def sales_at_minimum_price : ℕ := 300
def price_increase_effect : ℕ := 10
def max_profit_price := 52
def max_profit := 2640

-- Function relationship between y and x
def sales_volume (x : ℕ) : ℕ := 300 - 10 * (x - 44)

-- Range of x
def valid_price (x : ℕ) : Prop := 44 ≤ x ∧ x ≤ 52

-- Statement of the problem
theorem maximize_profit (x : ℕ) (hx : valid_price x) : 
  sales_volume x = 300 - 10 * (x - 44) ∧
  44 ≤ x ∧ x ≤ 52 ∧
  x = 52 → 
  (x - cost_price) * (sales_volume x) = max_profit :=
sorry

end maximize_profit_l42_42843


namespace certain_event_C_union_D_l42_42125

variable {Ω : Type} -- Omega, the sample space
variable {P : Set Ω → Prop} -- P as the probability function predicates the events

-- Definitions of the events
variable {A B C D : Set Ω}

-- Conditions
def mutually_exclusive (A B : Set Ω) : Prop := ∀ x, x ∈ A → x ∉ B
def complementary (A C : Set Ω) : Prop := ∀ x, x ∈ C ↔ x ∉ A

-- Given conditions
axiom A_and_B_mutually_exclusive : mutually_exclusive A B
axiom C_is_complementary_to_A : complementary A C
axiom D_is_complementary_to_B : complementary B D

-- Theorem statement
theorem certain_event_C_union_D : ∀ x, x ∈ C ∪ D := by
  sorry

end certain_event_C_union_D_l42_42125


namespace quadratic_root_product_l42_42554

theorem quadratic_root_product (a b : ℚ) (h1 : ∀ x, x^2 + a*x + b = 0 → x = 5 - real.sqrt 3 ∨ x = 5 + real.sqrt 3 ) :
  a * b = -220 :=
by sorry

end quadratic_root_product_l42_42554


namespace find_x_l42_42028

noncomputable def n : ℚ := 25 / 3

theorem find_x : ∃ x : ℚ, 3 * n + 15 = 6 * n - x ∧ x = 10 :=
by
  use 10
  have h1 : 3 * n + 15 = 6 * n - 10, from sorry
  exact ⟨h1, rfl⟩

end find_x_l42_42028


namespace find_function_l42_42994

open Real

theorem find_function (f : ℝ → ℝ) :
  let g := λ x, f (x / 2);
  let h := λ x, g (x - π / 3);
  (∀ x, h x = sin (x - π / 4)) →
  ∀ x, f x = sin (x / 2 + π / 12) :=
by
  intros g h h_eq x
  sorry

end find_function_l42_42994


namespace arithmetic_sequence_nth_term_eq_49_l42_42588

theorem arithmetic_sequence_nth_term_eq_49 :
  ∃ (n : ℕ), 11 + (n - 1) * -3 = -49 ∧ n = 21 :=
by
  use 21
  sorry

end arithmetic_sequence_nth_term_eq_49_l42_42588


namespace sum_m_b_eq_neg_five_halves_l42_42485

theorem sum_m_b_eq_neg_five_halves : 
  let x1 := 1 / 2
  let y1 := -1
  let x2 := -1 / 2
  let y2 := 2
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m + b = -5 / 2 :=
by 
  sorry

end sum_m_b_eq_neg_five_halves_l42_42485


namespace x_plus_2y_equals_5_l42_42122

theorem x_plus_2y_equals_5 (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : (x + y) / 3 = 1.222222222222222) : x + 2 * y = 5 := 
by sorry

end x_plus_2y_equals_5_l42_42122


namespace find_other_solution_l42_42067

theorem find_other_solution (x₁ : ℚ) (x₂ : ℚ) 
  (h₁ : x₁ = 3 / 4) 
  (h₂ : 72 * x₁^2 + 39 * x₁ - 18 = 0) 
  (eq : 72 * x₂^2 + 39 * x₂ - 18 = 0 ∧ x₂ ≠ x₁) : 
  x₂ = -31 / 6 := 
sorry

end find_other_solution_l42_42067


namespace radiator_initial_fluid_l42_42411

theorem radiator_initial_fluid (x : ℝ)
  (h1 : (0.10 * x - 0.10 * 2.2857 + 0.80 * 2.2857) = 0.50 * x) :
  x = 4 :=
sorry

end radiator_initial_fluid_l42_42411


namespace coefficient_of_monomial_is_correct_l42_42670

def monomial_coefficient : ℝ :=
  -2 * Real.pi

theorem coefficient_of_monomial_is_correct : monomial_coefficient = -2 * Real.pi :=
  by
    sorry

end coefficient_of_monomial_is_correct_l42_42670


namespace dice_surface_sum_l42_42752

theorem dice_surface_sum :
  ∃ X : ℤ, 1 ≤ X ∧ X ≤ 6 ∧ 
  (28175 + 2 * X = 28177 ∨
   28175 + 2 * X = 28179 ∨
   28175 + 2 * X = 28181 ∨
   28175 + 2 * X = 28183 ∨
   28175 + 2 * X = 28185 ∨
   28175 + 2 * X = 28187) := sorry

end dice_surface_sum_l42_42752


namespace profit_percentage_is_50_l42_42647

noncomputable def purchase_price : ℕ := 12000
noncomputable def repair_cost : ℕ := 5000
noncomputable def transportation_charges : ℕ := 1000
noncomputable def selling_price : ℕ := 27000

noncomputable def total_cost := purchase_price + repair_cost + transportation_charges
noncomputable def profit := selling_price - total_cost
noncomputable def profit_percentage := (profit / total_cost.toFloat) * 100

theorem profit_percentage_is_50 :
  profit_percentage = 50 := 
by
  sorry

end profit_percentage_is_50_l42_42647


namespace ratio_M_N_l42_42552

variable (M Q P N : ℝ)

-- Conditions
axiom h1 : M = 0.40 * Q
axiom h2 : Q = 0.25 * P
axiom h3 : N = 0.60 * P

theorem ratio_M_N : M / N = 1 / 6 :=
by
  sorry

end ratio_M_N_l42_42552


namespace parabola_equation_line_AB_equation_l42_42218

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42218


namespace solution_l42_42603

noncomputable def problem (z : ℂ) (hz : 
  z = (↑(-3 + 4 * complex.I)) ^ 5 * (↑(5 - 12 * complex.I)) ^ 6 / (2 + complex.I)) : 
  ℂ := (conjugate z) / z
  
theorem solution (z : ℂ) (hz : 
  z = (↑(-3 + 4 * complex.I)) ^ 5 * (↑(5 - 12 * complex.I)) ^ 6 / (2 + complex.I)) :
  complex.abs (problem z hz) = 1 := 
by
  sorry

end solution_l42_42603


namespace original_function_l42_42957

def f (x : ℝ) : ℝ := 
sorry

theorem original_function :
  (∀ x, (sin (x - π/4) = sin ((2 * (x - π / 3))))) ↔ (∀ x, f x = sin (x/2 + π/12)) :=
sorry

end original_function_l42_42957


namespace max_range_product_l42_42327

variable (f g : ℝ → ℝ)
variable (x : ℝ)

noncomputable def range_f := set.Icc (-5 : ℝ) (3 : ℝ)
noncomputable def range_g := set.Icc (-2 : ℝ) (1 : ℝ)

theorem max_range_product :
  (∀ x, f x ∈ range_f) →
  (∀ x, g x ∈ range_g) →
  ∃ (a b : ℝ), (∀ x, f x * g x ∈ set.Icc a b) ∧ b = 10 := 
by
  intros hf hg
  refine ⟨-10, 10, _, rfl⟩
  sorry

end max_range_product_l42_42327


namespace find_a_range_l42_42529

noncomputable
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 0 then x * Real.exp x else a * x ^ 2 - 2 * x

theorem find_a_range (a : ℝ) : (∀ x : ℝ, -1 / Real.exp 1 ≤ f x a) → a ∈ Set.Ici (Real.exp 1) :=
  sorry

end find_a_range_l42_42529


namespace original_function_l42_42998

def f (x : ℝ) : ℝ := sin (x - π / 4)

theorem original_function :
  ∃ g : ℝ → ℝ, (∀ x, f (2 * (x + π / 3)) = sin (x - π / 4)) ∧ g x = sin (x / 2 + π / 12) :=
sorry

end original_function_l42_42998


namespace concurrency_of_lines_l42_42892

-- Point definitions (A, B, C, D) lie on a circle and arc midpoints (M_1, M_2, M_3, M_4)
variables (A B C D K I1 I2 I3 I4 M1 M2 M3 M4 : Type)
noncomputable theory

-- Definitions of points lying on a circle and their properties
def points_on_circle (A B C D : Type) : Prop := sorry
def arcs_midpoints (M1 M2 M3 M4 : Type) : Prop := sorry
def intersection_point (K : Type) (A C B D : Type) : Prop := sorry
def incenters (I1 I2 I3 I4 : Type) (ABK BCK CDK DKA : Type) : Prop := sorry

-- Given conditions in terms of Lean definitions
axiom given_conditions : 
  points_on_circle A B C D ∧
  arcs_midpoints M1 M2 M3 M4 ∧
  intersection_point K A C B D ∧
  incenters I1 I2 I3 I4 (A × B × K) (B × C × K) (C × D × K) (D × A × K)

-- The problem statement: proving that the lines M1I1, M2I2, M3I3, M4I4 are concurrent
theorem concurrency_of_lines :
  ∃ P : Type, 
    ∀ (M1I1 M2I2 M3I3 M4I4 : Type),
      (M1I1 = M1 × I1) ∧
      (M2I2 = M2 × I2) ∧
      (M3I3 = M3 × I3) ∧
      (M4I4 = M4 × I4) →
      (∃ P: Type, (P = M1I1 ∧ P = M2I2 ∧ P = M3I3 ∧ P = M4I4))
:= by sorry

end concurrency_of_lines_l42_42892


namespace smallest_number_l42_42737

theorem smallest_number (x : ℕ) (h1 : 2 * x = third) (h2 : 4 * x = second) (h3 : 7 * x = fourth) (h4 : (x + second + third + fourth) / 4 = 77) :
  x = 22 :=
by sorry

end smallest_number_l42_42737


namespace asymptotes_of_C1_l42_42534

-- Definition of hyperbola C1 and condition that C1 and C2 share same eccentricity
def hyperbola_C1 : Type := { x : Real // (x^2) / 4 - (y^2) / k = 1 }
def hyperbola_C2 : Type := { x : Real // (x^2) / k - (y^2) / 9 = 1 }

-- Given conditions
variables (C1 C2 : hyperbola_C1) (h : C1.eccentricity = C2.eccentricity)

-- Prove the asymptote equations for hyperbola C1
theorem asymptotes_of_C1 (k : Real) (h_k : k = 6) : 
  asymptotes C1 = {y : Real | y = ± (sqrt ℝ 6 / 2) * x} :=
by
  sorry

end asymptotes_of_C1_l42_42534


namespace combinations_of_toppings_l42_42651

open Nat

theorem combinations_of_toppings :
  (nat.choose 7 3) = 35 :=
by
  sorry

end combinations_of_toppings_l42_42651


namespace part_one_part_two_l42_42236

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42236


namespace find_parabola_equation_find_line_AB_equation_l42_42247

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42247


namespace number_whose_multiples_are_considered_for_calculating_the_average_l42_42735

theorem number_whose_multiples_are_considered_for_calculating_the_average
  (x : ℕ)
  (n : ℕ)
  (a : ℕ)
  (b : ℕ)
  (h1 : n = 10)
  (h2 : a = (x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x) / 7)
  (h3 : b = 2*n)
  (h4 : a^2 - b^2 = 0) :
  x = 5 := 
sorry

end number_whose_multiples_are_considered_for_calculating_the_average_l42_42735


namespace final_value_of_f_l42_42340

theorem final_value_of_f (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f(x-1) = (1 + f(x+1)) / (1 - f(x+1))) :
  f 1 * f 2 * f 3 * ... * f 2008 + 2008 = 2009 :=
sorry

end final_value_of_f_l42_42340


namespace extremum_a_neg3_monotonic_f_a_geq_1_l42_42076

noncomputable def f (a x : ℝ) : ℝ := Real.exp x * (x^2 + 2*x + a)

theorem extremum_a_neg3 :
  ∀ (a x : ℝ), a = -3 →
  (∃ (x₁ x₂ : ℝ), f a x₁ = 6 * Real.exp (-3) ∧ f a x₂ = -2 * Real.exp 1) :=
by
  sorry

theorem monotonic_f_a_geq_1 :
  ∀ (a : ℝ), (∀ x : ℝ, (f a) = (λ x, Real.exp x * (x^2 + 2*x + a))) →
  (∀ x : ℝ, 0 ≤ x^2 + 2*x + a) →
  1 ≤ a :=
by
  sorry

end extremum_a_neg3_monotonic_f_a_geq_1_l42_42076


namespace dice_sum_surface_l42_42762

theorem dice_sum_surface (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6) : 
  ∃ Y : ℕ, Y = 28175 + 2 * X ∧ (Y = 28177 ∨ Y = 28179 ∨ Y = 28181 ∨ Y = 28183 ∨ 
  Y = 28185 ∨ Y = 28187) :=
by
  sorry

end dice_sum_surface_l42_42762


namespace number_of_2_face_painted_cubes_l42_42382

-- Condition definitions based on the problem statement
def painted_faces (n : ℕ) (type : String) : ℕ :=
  if type = "corner" then 8
  else if type = "edge" then 12
  else if type = "face" then 24
  else if type = "inner" then 9
  else 0

-- The mathematical proof statement
theorem number_of_2_face_painted_cubes : painted_faces 27 "edge" = 12 :=
by
  sorry

end number_of_2_face_painted_cubes_l42_42382


namespace david_mean_score_l42_42457

def david_quiz_scores : List ℕ := [87, 94, 89, 90, 88]

def mean_score (scores : List ℕ) : ℝ :=
  (scores.foldl (· + ·) 0 : ℚ) / scores.length

theorem david_mean_score : mean_score david_quiz_scores = 90 := by
  sorry

end david_mean_score_l42_42457


namespace campaign_fliers_l42_42723

theorem campaign_fliers (total_fliers : ℕ) (fraction_morning : ℚ) (fraction_afternoon : ℚ) 
  (remaining_fliers_after_morning : ℕ) (remaining_fliers_after_afternoon : ℕ) :
  total_fliers = 1000 → fraction_morning = 1/5 → fraction_afternoon = 1/4 → 
  remaining_fliers_after_morning = total_fliers - total_fliers * fraction_morning → 
  remaining_fliers_after_afternoon = remaining_fliers_after_morning - remaining_fliers_after_morning * fraction_afternoon → 
  remaining_fliers_after_afternoon = 600 := 
by
  sorry

end campaign_fliers_l42_42723


namespace greatest_cookies_one_student_can_take_l42_42423

theorem greatest_cookies_one_student_can_take
  (total_students : ℕ) (mean_cookies_per_student : ℕ) (min_cookies_per_student : ℕ)
  (total_cookies : ℕ) (min_cookies_24_students : ℕ) :
  total_students = 25 →
  mean_cookies_per_student = 4 →
  min_cookies_per_student = 1 →
  total_cookies = total_students * mean_cookies_per_student →
  min_cookies_24_students = min_cookies_per_student * (total_students - 1) →
  (total_cookies - min_cookies_24_students) = 76 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h4, h5],
  norm_num [h1, h2, h3]
end

end greatest_cookies_one_student_can_take_l42_42423


namespace number_of_triples_l42_42034

theorem number_of_triples :
  { (a, b, c) : ℕ × ℕ × ℕ // a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b + b * c = 56 ∧ a * c + b * c = 30 }.card = 2 :=
by
  sorry

end number_of_triples_l42_42034


namespace find_rate_of_interest_l42_42389

variable (P : ℝ) (R : ℝ) (T : ℕ := 2)

-- Condition for Simple Interest (SI = Rs. 660 for 2 years)
def simple_interest :=
  P * R * ↑T / 100 = 660

-- Condition for Compound Interest (CI = Rs. 696.30 for 2 years)
def compound_interest :=
  P * ((1 + R / 100) ^ T - 1) = 696.30

-- We need to prove that R = 11
theorem find_rate_of_interest (P : ℝ) (h1 : simple_interest P R) (h2 : compound_interest P R) : 
  R = 11 := by
  sorry

end find_rate_of_interest_l42_42389


namespace ball_box_arrangements_l42_42303

theorem ball_box_arrangements :
  let balls := {A, B, C}
  let boxes := {1, 2, 3, 4}
  (∑ a in boxes, ∑ b in boxes, ∑ c in boxes, (if a = 1 ∨ b = 1 ∨ c = 1 then 1 else 0)) = 37 :=
by
  sorry

end ball_box_arrangements_l42_42303


namespace parabola_equation_line_AB_equation_l42_42190

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42190


namespace michael_total_payment_correct_l42_42298

variable (original_suit_price : ℕ := 430)
variable (suit_discount : ℕ := 100)
variable (suit_tax_rate : ℚ := 0.05)

variable (original_shoes_price : ℕ := 190)
variable (shoes_discount : ℕ := 30)
variable (shoes_tax_rate : ℚ := 0.07)

variable (original_dress_shirt_price : ℕ := 80)
variable (original_tie_price : ℕ := 50)
variable (combined_discount_rate : ℚ := 0.20)
variable (dress_shirt_tax_rate : ℚ := 0.06)
variable (tie_tax_rate : ℚ := 0.04)

def calculate_total_amount_paid : ℚ :=
  let discounted_suit_price := original_suit_price - suit_discount
  let suit_tax := discounted_suit_price * suit_tax_rate
  let discounted_shoes_price := original_shoes_price - shoes_discount
  let shoes_tax := discounted_shoes_price * shoes_tax_rate
  let combined_original_price := original_dress_shirt_price + original_tie_price
  let combined_discount := combined_discount_rate * combined_original_price
  let discounted_combined_price := combined_original_price - combined_discount
  let discounted_dress_shirt_price := (original_dress_shirt_price / combined_original_price) * discounted_combined_price
  let discounted_tie_price := (original_tie_price / combined_original_price) * discounted_combined_price
  let dress_shirt_tax := discounted_dress_shirt_price * dress_shirt_tax_rate
  let tie_tax := discounted_tie_price * tie_tax_rate
  discounted_suit_price + suit_tax + discounted_shoes_price + shoes_tax + discounted_dress_shirt_price + dress_shirt_tax + discounted_tie_price + tie_tax

theorem michael_total_payment_correct : calculate_total_amount_paid = 627.14 := by
  sorry

end michael_total_payment_correct_l42_42298


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42277

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42277


namespace readers_both_l42_42137

-- Definitions of the number of readers
def total_readers : ℕ := 150
def readers_science_fiction : ℕ := 120
def readers_literary_works : ℕ := 90

-- Statement of the proof problem
theorem readers_both :
  (readers_science_fiction + readers_literary_works - total_readers) = 60 :=
by
  -- Proof omitted
  sorry

end readers_both_l42_42137


namespace problem1_problem2_l42_42268

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42268


namespace dice_sum_surface_l42_42765

theorem dice_sum_surface (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6) : 
  ∃ Y : ℕ, Y = 28175 + 2 * X ∧ (Y = 28177 ∨ Y = 28179 ∨ Y = 28181 ∨ Y = 28183 ∨ 
  Y = 28185 ∨ Y = 28187) :=
by
  sorry

end dice_sum_surface_l42_42765


namespace categorize_numbers_l42_42852

def is_integer (x : ℚ) : Prop := x.denominator = 1

def is_positive_integer (x : ℚ) : Prop := is_integer x ∧ x > 0

def is_negative_integer (x : ℚ) : Prop := is_integer x ∧ x < 0

def is_positive_fraction (x : ℚ) : Prop := x > 0 ∧ x < 1

def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x > -1

theorem categorize_numbers (numbers : List ℚ) :
  (∀ x ∈ numbers, is_integer x → x ∈ {2020, 1, -1, -2021, 0}) ∧
  (∀ x ∈ numbers, is_positive_integer x → x ∈ {2020, 1}) ∧
  (∀ x ∈ numbers, is_negative_integer x → x ∈ {-1, -2021}) ∧
  (∀ x ∈ numbers, is_positive_fraction x → x ∈ {0.5, 1/10, 0.2}) ∧
  (∀ x ∈ numbers, is_negative_fraction x → x ∈ {-1/3, -0.75}) :=
sorry

end categorize_numbers_l42_42852


namespace parabola_equation_line_AB_equation_l42_42223

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42223


namespace ellipse_focus_perpendicular_sin_l42_42511

theorem ellipse_focus_perpendicular_sin (a b x y : ℝ) (h1 : a > b) (h2 : b > 0)
  (ellipse_eq : x^2 / a^2 + y^2 / b^2 = 1)
  (F1 F2 P : ℝ × ℝ)
  (focus1 : F1 = (-sqrt(a^2 - b^2), 0))
  (focus2 : F2 = (sqrt(a^2 - b^2), 0))
  (perpendicular : ∃ x₀, P = (x₀, sqrt(a^2 - b^2)) ∨ P = (x₀, -sqrt(a^2 - b^2)))
  (sin_angle : real.sin (real.angle_of_vectors (F1 - P) (F2 - P)) = 1 / 3) :
  a = sqrt(2) * b :=
sorry

end ellipse_focus_perpendicular_sin_l42_42511


namespace part1_part2_l42_42537

def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x - 4 ≤ 0}

-- Problem 1
theorem part1 (m : ℝ) : 
  (A ∩ B m = {x : ℝ | 1 ≤ x ∧ x ≤ 3}) → m = 3 :=
by sorry

-- Problem 2
theorem part2 (m : ℝ) : 
  (A ⊆ (B m)ᶜ) → (m < -3 ∨ m > 5) :=
by sorry

end part1_part2_l42_42537


namespace hyperbola_equation_area_of_triangle_PF1F2_l42_42905

noncomputable def ellipse := set_of (λ p : ℝ × ℝ, (p.1^2 / 35 + p.2^2 / 10 = 1))

def hyperbola_eq (x y a b: ℝ) := (x^2 / a^2 - y^2 / b^2 = 1)

def foci_of_ellipse : set (ℝ × ℝ) := {p | p.1 = 5 ∨ p.1 = -5 ∧ p.2 = 0}

def asymptotes_of_hyperbola (λ : ℝ) : set (ℝ × ℝ) := {p | p.2 = (4 / 3) * p.1 ∨ p.2 = -(4 / 3) * p.1}

def is_midpoint_on_y_axis (p q: ℝ × ℝ) : Prop := (p.1 + q.1) / 2 = 0

def area_of_triangle (f1 f2 p: ℝ × ℝ) : ℝ := (1/2) * |f1.1 - f2.1| * |p.2|

theorem hyperbola_equation
  (a b: ℝ)
  (F1 F2 : ℝ × ℝ)
  (f1_foci : F1 ∈ foci_of_ellipse)
  (f2_foci: F2 ∈ foci_of_ellipse)
  (asympt_b : asymptotes_of_hyperbola 1 (a, b))
  : hyperbola_eq a b 3 4 :=
by
  sorry

theorem area_of_triangle_PF1F2
  (P F1 F2: ℝ × ℝ)
  (F1_def: F1 = (-5, 0))
  (is_mid_pt: is_midpoint_on_y_axis P F1)
  (F1_F2_dist: |F1.1 - F2.1| = 10)
  (p_y: P.2 = 16 / 3)
  : area_of_triangle F1 F2 P = 80 / 3 :=
by
  sorry

end hyperbola_equation_area_of_triangle_PF1F2_l42_42905


namespace smallest_positive_period_symmetry_center_exists_range_of_f_l42_42920

def f (x : ℝ) : ℝ :=
  real.sqrt 3 * real.cos (2 * x - real.pi / 3) - 2 * real.sin x * real.cos x

theorem smallest_positive_period :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = real.pi :=
sorry

theorem symmetry_center_exists :
  ∃ k : ℤ, ∀ x : ℝ, 2 * x + real.pi / 3 = k * real.pi →
  ∃ c : ℝ, ∀ x : ℝ, f (2 * x + real.pi / 3) = f (c) ∧ c = (k * real.pi / 2 - real.pi / 6) :=
sorry

theorem range_of_f :
  ∀ x : ℝ, x ∈ (-real.pi / 4, real.pi / 4] →
  f x ∈ (-1 / 2, 1] :=
sorry

end smallest_positive_period_symmetry_center_exists_range_of_f_l42_42920


namespace relationship_among_abc_l42_42495

noncomputable def a := log 2 3 + log 2 (sqrt 3)
noncomputable def b := log 2 9 - log 2 (sqrt 3)
noncomputable def c := log 3 2

theorem relationship_among_abc : a = b ∧ a > c := by
  sorry

end relationship_among_abc_l42_42495


namespace aniyah_more_candles_l42_42435

theorem aniyah_more_candles (x : ℝ) (h1 : 4 + 4 * x = 14) : x = 2.5 :=
sorry

end aniyah_more_candles_l42_42435


namespace probability_diamond_spade_spade_l42_42706

-- Condition definitions for the problem
def standard_deck := {cards : Set ℝ // cards.cardinality = 52}
def diamonds : Set ℝ := {x : ℝ | x ∈ standard_deck.cards ∧ x.is_diamond}
def spades : Set ℝ := {x : ℝ | x ∈ standard_deck.cards ∧ x.is_spade}

noncomputable def probability (A B : Set ℝ) (deck : Set ℝ) := 
  ((A.cardinality : ℝ) / (deck.cardinality : ℝ)) *
  ((B.cardinality : ℝ) / ((deck.cardinality : ℝ) - 1)) *
  (((B - {A}) .cardinality : ℝ) / ((deck.cardinality : ℝ) - 2))

-- Problem: Prove the probability calculation
theorem probability_diamond_spade_spade :
  probability diamonds spades standard_deck.cards = 13 / 850 := sorry

end probability_diamond_spade_spade_l42_42706


namespace find_equation_of_C_find_equation_of_AB_l42_42208

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42208


namespace geometric_number_difference_is_452_l42_42445

noncomputable def base9_to_base10 (a b c : ℕ) : ℕ := a * 9^2 + b * 9^1 + c * 9^0

def is_geometric_number (a b c : ℕ) (r : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  b = int.nat_abs (a * r) ∧ c = int.nat_abs (a * r^2)

def largest_geometric_number : ℕ := 818
def smallest_geometric_number : ℕ := 175

def largest_base10 := base9_to_base10 8 1 8
def smallest_base10 := base9_to_base10 1 7 5

theorem geometric_number_difference_is_452 :
  largest_base10 - smallest_base10 = 452 :=
by
  have h1 : largest_base10 = 8 * 9^2 + 1 * 9^1 + 8 * 9^0 := by rfl
  have h2 : smallest_base10 = 1 * 9^2 + 7 * 9^1 + 5 * 9^0 := by rfl
  rw [h1, h2]
  norm_num
  sorry

end geometric_number_difference_is_452_l42_42445


namespace num_true_statements_l42_42170

variables (a b c p : Vector ℝ)
  (x y z : ℝ)

-- Conditions 
axiom non_zero_vectors : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

axiom parallel_ab : a ∥ b
axiom parallel_bc : b ∥ c

axiom dot_product_equality : a • b = b • c

-- The number of additional true propositions
noncomputable def num_definitely_true_statements := (1 : ℕ)

-- Statement to prove
theorem num_true_statements : num_definitely_true_statements = 1 :=
sorry

end num_true_statements_l42_42170


namespace Turner_Catapult_rides_l42_42705

def tickets_needed (rollercoaster_rides Ferris_wheel_rides Catapult_rides : ℕ) : ℕ :=
  4 * rollercoaster_rides + 1 * Ferris_wheel_rides + 4 * Catapult_rides

theorem Turner_Catapult_rides :
  ∀ (x : ℕ), tickets_needed 3 1 x = 21 → x = 2 := by
  intros x h
  sorry

end Turner_Catapult_rides_l42_42705


namespace circle_equation_standard_line_equation_standard_chord_length_l42_42593

noncomputable def circle_center := (0, 2)
noncomputable def circle_radius := 3
noncomputable def line_equation (x y : ℝ) : Prop := (√3) * x - y = 0

theorem circle_equation_standard :
  ∀ (x y : ℝ), (x - fst circle_center)^2 + (y - snd circle_center)^2 = circle_radius^2 :=
  sorry

theorem line_equation_standard (x y : ℝ) :
  line_equation x y :=
  sorry

theorem chord_length :
  distance (0, 2) ({p : ℝ × ℝ | line_equation p.1 p.2}) = 4 * √2 :=
  sorry

end circle_equation_standard_line_equation_standard_chord_length_l42_42593


namespace license_plates_count_l42_42113

-- Definitions based on conditions from the problem
def letters_count : ℕ := 26
def positions_count : ℕ := 3
def odd_digit_choices : ℕ := 5
def even_digit_choices : ℕ := 5

-- The main statement without proof
theorem license_plates_count :
  (letters_count ^ positions_count) * (positions_count * odd_digit_choices * even_digit_choices ^ 2) = 6_591_000 := by sorry

end license_plates_count_l42_42113


namespace dot_product_result_parallelism_condition_l42_42090

-- Definitions of the vectors
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 2)

-- 1. Prove the dot product result
theorem dot_product_result :
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  let a_minus_2b := (a.1 - 2 * b.1, a.2 - 2 * b.2)
  a_plus_b.1 * a_minus_2b.1 + a_plus_b.2 * a_minus_2b.2 = -14 :=
by
  sorry

-- 2. Prove parallelism condition
theorem parallelism_condition (k : ℝ) :
  let k_a_plus_b := (k * a.1 + b.1, k * a.2 + b.2)
  let a_minus_3b := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  k = -1/3 → k_a_plus_b.1 * a_minus_3b.2 = k_a_plus_b.2 * a_minus_3b.1 :=
by
  sorry

end dot_product_result_parallelism_condition_l42_42090


namespace proper_coloring_count_l42_42549

def proper_divisors : ℕ → list ℕ
| 1     := []
| n     := (list.range (n - 1)).filter (λ k, k ∣ n && k ≠ 0)

def is_colored_properly (colors : ℕ → ℕ) (n : ℕ) : Prop :=
∀ d ∈ proper_divisors n, colors d ≠ colors n

def count_valid_colorings : ℕ :=
let numbers := list.range' 1 10,
    all_colorings := (colors)),
  sorry).length

theorem proper_coloring_count : count_valid_colorings = 940 := sorry

end proper_coloring_count_l42_42549


namespace minimum_races_needed_l42_42738

-- Define the conditions as Lean definitions
def max_horses_per_race := 4
def total_horses := 30

-- Define the problem statement in Lean
theorem minimum_races_needed (y : ℕ) :
  (∀ (max_horses_per_race : ℕ) (total_horses : ℕ), max_horses_per_race = 4 → total_horses = 30 → y = 8): 
  y = 8 := by
  sorry

end minimum_races_needed_l42_42738


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42276

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42276


namespace equal_areas_of_triangles_l42_42578

theorem equal_areas_of_triangles
  (A B C H E D P Q T : Type)
  (hABC : ∀ (A B C : Type), non_equilateral_acute_triangle A B C)
  (hAltitudes : ∀ (BE CD : Type), altitude A B C H)
  (hBisector_intersects : ∀ (P Q : Type), angle_bisector_intersects A B C P Q)
  (hOrthocenter : ∀ (T : Type), orthocenter_of_triangle_hpq T H P Q)
  (DA_orthocenter : ∀ (T D A : Type), orthocenter_intersects_altitude T D A)
  (EA_orthocenter : ∀ (T E A : Type), orthocenter_intersects_altitude T E A) :
  area (triangle T D A) = area (triangle T E A) :=
by
  sorry

end equal_areas_of_triangles_l42_42578


namespace problem_1_problem_2_l42_42195

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42195


namespace complex_mul_example_l42_42747

theorem complex_mul_example : (1 + complex.i) * (2 + complex.i) * (3 + complex.i) = 10 * complex.i :=
by
  sorry

end complex_mul_example_l42_42747


namespace triangle_areas_sum_le_half_rectangle_l42_42594

variables {A B C D E M N K T : Type}
variables [H1 : ∃ (A B C D : ℝ), true] 
variables [H2 : ∃ (E : ℝ), E ≠ B ∧ E ≠ C]
variables [H3 : ∃ (M : ℝ), M ∈ segment A D]
variables [H4 : ∃ (N : ℝ), N ∈ segment M D]
variables [H5 : ∃ (K : ℝ), K ∈ segment A B]
variables [H6 : ∃ (T : ℝ), T ∈ segment C D]

theorem triangle_areas_sum_le_half_rectangle (h1 : A B C D : ℝ) (h2 : E : ℝ) (h3 : M : ℝ) (h4 : N : ℝ) (h5 : K : ℝ) (h6 : T : ℝ) : 
  (area_of_triangle A K M + area_of_triangle M E N + area_of_triangle N D T) ≤ (1 / 2) * area_of_rectangle A B C D :=
sorry

end triangle_areas_sum_le_half_rectangle_l42_42594


namespace dice_surface_sum_l42_42751

theorem dice_surface_sum :
  ∃ X : ℤ, 1 ≤ X ∧ X ≤ 6 ∧ 
  (28175 + 2 * X = 28177 ∨
   28175 + 2 * X = 28179 ∨
   28175 + 2 * X = 28181 ∨
   28175 + 2 * X = 28183 ∨
   28175 + 2 * X = 28185 ∨
   28175 + 2 * X = 28187) := sorry

end dice_surface_sum_l42_42751


namespace subset_property_l42_42721

theorem subset_property : {2} ⊆ {x | x ≤ 10} := 
by 
  sorry

end subset_property_l42_42721


namespace original_function_l42_42984

theorem original_function :
  ∃ f : ℝ → ℝ,
  (∀ x : ℝ, sin (x - π / 4) = f (2 * (x - π / 3))) →
  (∀ x : ℝ, f x = sin (x / 2 + π / 12)) :=
by
  sorry

end original_function_l42_42984


namespace product_is_two_l42_42443

theorem product_is_two : 
  ((10 : ℚ) * (1/5) * 4 * (1/16) * (1/2) * 8 = 2) :=
sorry

end product_is_two_l42_42443


namespace dihedral_angle_sine_example_l42_42595

-- Define geometric objects and properties
structure TrianglePyramid :=
  (A B C D : Type)
  (Angle_ACB ABD BAD : ℝ)
  (CA CB : ℝ)
  (projection_AD : Prop)

noncomputable def sine_dihedral_angle (P : TrianglePyramid) : ℝ :=
  if h : (P.Angle_ACB = 90) ∧ (P.Angle_ABD = 90) ∧ (P.CA = P.CB) ∧ (P.Angle_BAD = 30) ∧ P.projection_AD
  then (√6 / 3)
  else 0

-- Example of defining a specific instance of TrianglePyramid and proving the property
def example_pyramid : TrianglePyramid :=
  { A := ℝ,
    B := ℝ,
    C := ℝ,
    D := ℝ,
    Angle_ACB := 90,
    Angle_ABD := 90,
    Angle_BAD := 30,
    CA := 1,
    CB := 1,
    projection_AD := sorry -- Proof of projection being on AD, needs to be demonstrated.
  }

theorem dihedral_angle_sine_example :
  sine_dihedral_angle example_pyramid = √6 / 3 :=
sorry

end dihedral_angle_sine_example_l42_42595


namespace gwen_points_earned_l42_42745
-- Import the entirety of the Mathlib library

-- Define the conditions
def points_per_bag := 8
def total_bags := 4
def not_recycled := 2

-- Theorem statement to match the problem requirements
theorem gwen_points_earned : 
  points_per_bag = 8 ∧ total_bags = 4 ∧ not_recycled = 2 → 
  (total_bags - not_recycled) * points_per_bag = 16 := 
by
  intros h,
  let h_points_per_bag := h.1,
  let h_total_bags := h.2.1,
  let h_not_recycled := h.2.2,
  -- Proof skipped with sorry
  sorry

end gwen_points_earned_l42_42745


namespace range_of_a_l42_42749

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (|x - 5| + |x + 3| < a)) ↔ a ≤ 8 :=
by
  sorry

end range_of_a_l42_42749


namespace parabola_equation_line_AB_equation_l42_42224

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42224


namespace part1_eq_C_part2_line_AB_l42_42284

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42284


namespace axis_of_symmetry_l42_42951

def is_symmetry_axis (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, g(x) = g(2 * a - x)

theorem axis_of_symmetry (g : ℝ → ℝ)
  (h : ∀ x : ℝ, g(x) = g(3 + x)) :
  is_symmetry_axis g 1.5 :=
by
  sorry

end axis_of_symmetry_l42_42951


namespace correct_propositions_l42_42505

variable {a : ℕ → ℝ}
variable S : ℕ → ℝ

-- Condition: Sequence is arithmetic and points are collinear
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: Sum of first n terms
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a i

-- Proposition 1: Points are collinear
def points_collinear (a : ℕ → ℝ) : Prop :=
  is_arithmetic a → let S n := sum_first_n_terms a n in
  let p1 := (10 : ℝ, S 10 / 10) in
  let p2 := (100 : ℝ, S 100 / 100) in
  let p3 := (110 : ℝ, S 110 / 110) in
  collinear ℝ {p1, p2, p3}

-- Condition: Arithmetic sequence with specific initial conditions
def specific_arithmetic_conditions (a : ℕ → ℝ) : Prop :=
  is_arithmetic a ∧ a 0 = -11 ∧ (a 2 + a 6) = -6

-- Proposition 2: No maximum exists in sum sequence
def no_maximum_in_sum_sequence (a : ℕ → ℝ) : Prop :=
  specific_arithmetic_conditions a → ¬ ∃ n, ∀ m > n, S m ≤ S n 

-- Condition: Sequence is geometric
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Proposition 3: Geometric relationship in sums
def geometric_sum_relationship (a : ℕ → ℝ) : Prop :=
  is_geometric a →
  ∀ m : ℕ, S m ≠ 0 →
  let diff1 := S (2 * m) - S m in
  let diff2 := S (3 * m) - S (2 * m) in
  ∃ q : ℝ, (q ≠ 0 ∧ diff1 = S m * (q - 1) ∧ diff2 = diff1 * (q - 1))

-- Condition: Recursive relation for sums
def recursive_sum_relation (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  q ≠ 0 ∧ ∀ n: ℕ, S (n + 1) = a1 + q * S n

-- Proposition 4: Sequence is geometric
def sequence_geometric (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  recursive_sum_relation a a1 q → is_geometric a

theorem correct_propositions : 
  points_collinear a ∧ 
  ¬ no_maximum_in_sum_sequence a ∧ 
  ¬ geometric_sum_relationship a ∧ 
  sequence_geometric a :=
  sorry

end correct_propositions_l42_42505


namespace find_f_of_two_thirds_l42_42950

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x^2

noncomputable def f (z : ℝ) (x : ℝ) : ℝ := if x ≠ 0 then z / x^2 else 0

theorem find_f_of_two_thirds :
  ∀ (y x : ℝ), g(y) = 2 - 3 * y^2 → f (g(y)) x = (2 - 3 * x^2) / x^2 → f (2/3) x = 3.5 :=
by
  intros y x h1 h2
  sorry

end find_f_of_two_thirds_l42_42950


namespace part1_eq_C_part2_line_AB_l42_42290

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42290


namespace seq_100_eq_11_div_12_l42_42835

def seq (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 1 / 3
  else if n ≥ 3 then (2 - seq (n - 1)) / (3 * seq (n - 2) + 1)
  else 0 -- This line handles the case n < 1, but shouldn't ever be used in practice.

theorem seq_100_eq_11_div_12 : seq 100 = 11 / 12 :=
  sorry

end seq_100_eq_11_div_12_l42_42835


namespace equation_of_parabola_equation_of_line_AB_l42_42259

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42259


namespace no_int_solutions_a_b_l42_42152

theorem no_int_solutions_a_b :
  ¬ ∃ (a b : ℤ), a^2 + 1998 = b^2 :=
by
  sorry

end no_int_solutions_a_b_l42_42152


namespace find_angle_A_range_of_b_plus_c_l42_42131

-- Definitions from the conditions
variables {A B C a b c : ℝ}
variable h_parallel : (a, sqrt 3 * b) = (cos A, sin B) * λ -- implies vectors are parallel

-- First Part: Prove A = π/3
theorem find_angle_A (h_parallel : (a, sqrt 3 * b) = (cos A, sin B) * λ) : A = π / 3 := 
  sorry

-- Second Part: Prove the range when a = 2 and A = π / 3
theorem range_of_b_plus_c (h_parallel : (a, sqrt 3 * b) = (cos A, sin B) * λ) (h_a : a = 2) (h_A : A = π / 3) : 2 < b + c ∧ b + c ≤ 4 := 
  sorry

end find_angle_A_range_of_b_plus_c_l42_42131


namespace shooter_hits_target_l42_42138

-- Definitions based on conditions
def total_shots : ℕ := 25
def penalty_points (misses : ℕ) : ℝ := (misses : ℝ) / 2 * (2 + 0.5 * (misses - 1))

-- Main Theorem statement
theorem shooter_hits_target (misses hits : ℕ) (h₁ : misses + hits = total_shots) (h₂ : penalty_points misses = 7) : hits = 21 := by
  sorry

end shooter_hits_target_l42_42138


namespace linear_transformation_is_scalar_multiple_of_identity_l42_42563

open LinearAlgebra

variables {K : Type*} [Field K]
variables {V : Type*} [AddCommGroup V] [Module K V]
variables {T : End K V}
variables {n : ℕ}

/-- T is a scalar multiple of the identity -/
theorem linear_transformation_is_scalar_multiple_of_identity
  (eigenvectors : Fin (n + 1) → V)
  (eigenvalues : Fin (n + 1) → K)
  (lin_indep : LinearIndependent K (λ i : Fin n, eigenvectors i))
  (eig_eq : ∀ i, T (eigenvectors i) = eigenvalues i • eigenvectors i) :
  ∃ c : K, ∀ v : V, T v = c • v :=
begin
  sorry
end

end linear_transformation_is_scalar_multiple_of_identity_l42_42563


namespace sum_of_digits_A_plus_B_l42_42699

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the problem conditions
variables (A B : ℕ)
variables (A_pos : A > 0) (B_pos : B > 0)
variables (hA : sum_of_digits A = 19)
variables (hB : sum_of_digits B = 20)
variables (carryovers : nat.succ (nat.succ nat.zero) = 2) -- Represents the two carryovers

-- The statement we need to prove
theorem sum_of_digits_A_plus_B : sum_of_digits (A + B) = 21 :=
by sorry

end sum_of_digits_A_plus_B_l42_42699


namespace gcd_of_expression_l42_42838

theorem gcd_of_expression 
  (a b c d : ℕ) :
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (a - b) (c - d)) (a - c)) (b - d)) (a - d)) (b - c) = 12 :=
sorry

end gcd_of_expression_l42_42838


namespace floor_function_range_l42_42674

open Int

theorem floor_function_range : (set.image (fun x => 2 * (floor x) + 1) (set.Ico (-1 : ℝ) 3) = {-1, 1, 3, 5}) :=
by
  sorry

end floor_function_range_l42_42674


namespace tan_product_l42_42562

theorem tan_product (h : ∀ x : ℝ, (1 + Real.tan x) * (1 + Real.tan (30 - x)) = 2) :
  (List.prod (List.map (λ x, 1 + Real.tan x) (List.range' 1 15))) = 2^14 :=
by
  sorry

end tan_product_l42_42562


namespace find_first_number_l42_42332

theorem find_first_number (x : ℕ) : 
    (x + 32 + 53) / 3 = (21 + 47 + 22) / 3 + 3 ↔ x = 14 := by
  sorry

end find_first_number_l42_42332


namespace problem_1_problem_2_l42_42538

open Set

variable {U : Set ℝ := univ}

-- Define set A
def A (a : ℝ) : Set ℝ := { x | a - 1 ≤ x ∧ x ≤ 2 * a + 3 }

-- Define set B
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }

-- Question 1
theorem problem_1  : (U \ (A 1)) ∩ B = { x | -1 ≤ x ∧ x < 0 } := sorry

-- Question 2
theorem problem_2 : (∀ x, A a x → B x) ↔ (a < -4 ∨ (0 ≤ a ∧ a ≤ 1 / 2)) := sorry

end problem_1_problem_2_l42_42538


namespace dot_product_self_l42_42556

variable (w : ℝ^n) (h : ∥w∥ = 5)

theorem dot_product_self : w ⬝ w = 25 := sorry

end dot_product_self_l42_42556


namespace a4_eq_5_div_3_l42_42147

def sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 0 -- This is just a placeholder for n=0
  | 1 => 1
  | (n+2) => 1/(sequence (n+1)) + 1

theorem a4_eq_5_div_3 : sequence 4 = 5/3 := by
  sorry

end a4_eq_5_div_3_l42_42147


namespace maximum_point_f_l42_42044

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem maximum_point_f : is_maximum_point f Real.e := sorry

end maximum_point_f_l42_42044


namespace roof_difference_l42_42690

noncomputable def width := Real.sqrt 147
noncomputable def length := 4 * width
def area := 588
def approx_sqrt_3 : ℝ := 1.732
def difference := length - width

theorem roof_difference :
  3 * width * width = 147 ∧
  length = 4 * width ∧
  area = 588 ∧
  difference ≈ 21 * approx_sqrt_3 :=
by
  sorry

end roof_difference_l42_42690


namespace maple_syrup_jar_price_l42_42786

noncomputable def calculate_price (d h : ℝ) (price : ℝ) : ℝ :=
  (price * (π * (d / 2) ^ 2 * h)) / (π * (2) ^ 2 * 5)

theorem maple_syrup_jar_price :
  calculate_price 8 10 1.20 = 9.60 :=
by
  sorry

end maple_syrup_jar_price_l42_42786


namespace daniel_last_10_successful_shots_l42_42456

theorem daniel_last_10_successful_shots :
  ∀ (initial_attempts additional_attempts total_attempts initial_success_rate final_success_rate initial_successful additional_till_final_success),
  initial_attempts = 30 →
  additional_attempts = 10 →
  total_attempts = initial_attempts + additional_attempts →
  initial_success_rate = 0.60 →
  final_success_rate = 0.62 →
  initial_successful = initial_attempts * initial_success_rate →
  (total_attempts * final_success_rate).round = initial_successful + additional_till_final_success →
  additional_till_final_success = 7 :=
by
  intros initial_attempts additional_attempts total_attempts initial_success_rate final_success_rate initial_successful additional_till_final_success
  assume h1 : initial_attempts = 30
  assume h2 : additional_attempts = 10
  assume h3 : total_attempts = initial_attempts + additional_attempts
  assume h4 : initial_success_rate = 0.60
  assume h5 : final_success_rate = 0.62
  assume h6 : initial_successful = initial_attempts * initial_success_rate
  assume h7 : (total_attempts * final_success_rate).round = initial_successful + additional_till_final_success
  sorry

end daniel_last_10_successful_shots_l42_42456


namespace range_of_f_l42_42532

noncomputable def f (x : ℝ) : ℝ := 2^(x + 2) - 4^x

def domain_M (x : ℝ) : Prop := 1 < x ∧ x < 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, domain_M x ∧ f x = y) ↔ -32 < y ∧ y < 4 :=
sorry

end range_of_f_l42_42532


namespace sum_of_possible_values_l42_42806

-- Define the triangle's base and height
def triangle_base (x : ℝ) : ℝ := x - 2
def triangle_height (x : ℝ) : ℝ := x - 2

-- Define the parallelogram's base and height
def parallelogram_base (x : ℝ) : ℝ := x - 3
def parallelogram_height (x : ℝ) : ℝ := x + 4

-- Define the areas
def triangle_area (x : ℝ) : ℝ := 0.5 * (triangle_base x) * (triangle_height x)
def parallelogram_area (x : ℝ) : ℝ := (parallelogram_base x) * (parallelogram_height x)

-- Statement to prove
theorem sum_of_possible_values (x : ℝ) (h : parallelogram_area x = 3 * triangle_area x) : x = 8 ∨ x = 3 →
  (x = 8 ∨ x = 3) → 8 + 3 = 11 :=
by sorry

end sum_of_possible_values_l42_42806


namespace find_parabola_equation_find_line_AB_equation_l42_42249

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42249


namespace part1_eq_C_part2_line_AB_l42_42288

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42288


namespace unique_intersection_singleton_l42_42316

theorem unique_intersection_singleton {a : ℤ} :
  let A := {-4, 2 * a - 1, a^2}
  let B := {9, a - 5, 1 - a}
  A ∩ B = {9} → a = -3 :=
by
  intros A B h
  sorry

end unique_intersection_singleton_l42_42316


namespace largest_domain_of_g_l42_42339

noncomputable def domain_g : Set ℝ := {x | x ≠ 0 ∧ ∃ g : ℝ → ℝ, ∀ x ∈ Set.univ, g(x) + g(1/x) = x + 3}

theorem largest_domain_of_g : domain_g = {-1, 1} :=
by
  sorry

end largest_domain_of_g_l42_42339


namespace area_of_abs_5x_plus_abs_2y_eq_20_l42_42714

theorem area_of_abs_5x_plus_abs_2y_eq_20 :
  (∃ d1 d2 : ℝ, d1 = 8 ∧ d2 = 20 ∧ (1 / 2) * d1 * d2 = 80) := by
sory

end area_of_abs_5x_plus_abs_2y_eq_20_l42_42714


namespace necessary_but_not_sufficient_condition_l42_42926

-- Define the sets M and P
def M (x : ℝ) : Prop := x > 2
def P (x : ℝ) : Prop := x < 3

-- Statement of the problem
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (M x ∨ P x) → (x ∈ { y : ℝ | 2 < y ∧ y < 3 }) :=
sorry

end necessary_but_not_sufficient_condition_l42_42926


namespace line_intersects_circle_at_two_points_minimum_chord_length_line_l42_42500

noncomputable def circle (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 25

noncomputable def line_l (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Part I: Prove that no matter what real number m is, line l always intersects the circle at two points.
theorem line_intersects_circle_at_two_points (m : ℝ) : 
  ∃ x y : ℝ, circle x y ∧ line_l m x y :=
begin
  sorry
end

-- Part II: Find the equation of line l when the chord cut by circle C has the minimum length.
theorem minimum_chord_length_line :
  ∃ m : ℝ, ∀ x y : ℝ, line_l m x y ↔ 2 * x - y - 5 = 0 :=
begin
  sorry
end

end line_intersects_circle_at_two_points_minimum_chord_length_line_l42_42500


namespace correct_function_is_f3_l42_42812

-- Definitions of the functions
noncomputable def f1 (x : ℝ) := (1 / 2) ^ |x|
noncomputable def f2 (x : ℝ) := |x| - x ^ 2
noncomputable def f3 (x : ℝ) := |x| - 1
noncomputable def f4 (x : ℝ) := x - (1 / x)

-- Theorem statement
theorem correct_function_is_f3 :
  (∀ x, f1 (-x) = f1 x) ∧ (∀ x > 0, f1 x < f1 (x + 1)) ∨
  (∀ x, f2 (-x) = f2 x) ∧ (∀ x > 0, f2 x < f2 (x + 1)) ∨
  (∀ x, f3 (-x) = f3 x) ∧ (∀ x > 0, f3 x < f3 (x + 1)) ∨
  (∀ x, f4 (-x) = f4 x) ∧ (∀ x > 0, f4 x < f4 (x + 1)) → f3 = λ x, |x| - 1 := 
sorry

end correct_function_is_f3_l42_42812


namespace find_m_minus_n_l42_42877

theorem find_m_minus_n :
  let a := (2 : ℝ, 1 : ℝ)
  let b := (1 : ℝ, -2 : ℝ)
  ∃ m n : ℝ, m * a + n * b = (9, -8) → m - n = -3 :=
by
  sorry

end find_m_minus_n_l42_42877


namespace sphere_radius_in_cube_l42_42887

noncomputable def radius_of_sphere (a : ℝ) : ℝ :=
  (a * Real.sqrt 41) / 8

theorem sphere_radius_in_cube (a : ℝ) :
  ∀ (A B C D A₁ B₁ C₁ D₁ : ℝ × ℝ × ℝ),
    A = (0, 0, 0) →
    B = (a, 0, 0) →
    C = (a, a, 0) →
    D = (0, a, 0) →
    A₁ = (0, 0, a) →
    B₁ = (a, 0, a) →
    C₁ = (a, a, a) →
    D₁ = (0, a, a) →
    ∀ (M : ℝ × ℝ × ℝ),
      (dist M A = dist M C) →
      (dist M ((B₁.1 + C₁.1) / 2, (B₁.2 + C₁.2) / 2, (B₁.3 + C₁.3) / 2) =
       dist M ((C₁.1 + D₁.1) / 2, (C₁.2 + D₁.2) / 2, (C₁.3 + D₁.3) / 2)) →
      dist M ((C₁.1 + D₁.1) / 2, (C₁.2 + D₁.2) / 2, (C₁.3 + D₁.3) / 2) = radius_of_sphere a :=
sorry

end sphere_radius_in_cube_l42_42887


namespace solve_abcd_l42_42472

theorem solve_abcd : 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |4 * x^3 - d * x| ≤ 1) ∧ 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |4 * x^3 + a * x^2 + b * x + c| ≤ 1) →
  d = 3 ∧ b = -3 ∧ a = 0 ∧ c = 0 :=
by
  sorry

end solve_abcd_l42_42472


namespace quotient_correct_l42_42480

noncomputable def dividend : Polynomial ℤ := 9 * X^3 - 5 * X^2 + 8 * X - 12
noncomputable def divisor : Polynomial ℤ := X - 3
noncomputable def expectedQuotient : Polynomial ℤ := 9 * X^2 + 22 * X + 74

theorem quotient_correct : (dividend / divisor) = expectedQuotient := 
  sorry

end quotient_correct_l42_42480


namespace factorize_poly_l42_42725

theorem factorize_poly : 
  (x : ℤ) → (x^12 + x^6 + 1) = (x^4 - x^2 + 1) * (x^8 + x^4 + 1) :=
by
  sorry

end factorize_poly_l42_42725


namespace probability_event_occurring_exactly_once_l42_42148

theorem probability_event_occurring_exactly_once
  (P : ℝ)
  (h1 : ∀ n : ℕ, P ≥ 0 ∧ P ≤ 1) -- Probabilities are valid for all trials
  (h2 : (1 - (1 - P)^3) = 63 / 64) : -- Given condition for at least once
  (3 * P * (1 - P)^2 = 9 / 64) := 
by
  -- Here you would provide the proof steps using the conditions given.
  sorry

end probability_event_occurring_exactly_once_l42_42148


namespace inequality_smallest_p_l42_42483

/-- Given positive real numbers a and b, the smallest real constant p for which the inequality 
    √(ab) - 2ab/(a + b) ≤ p * ((a + b)/2 - √(ab)) holds is p = 1. -/
theorem inequality_smallest_p (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  let p : ℝ := 1 in
  (sqrt (a * b) - (2 * a * b) / (a + b) ≤ p * ((a + b) / 2 - sqrt (a * b))) :=
by
  let p := 1
  sorry

end inequality_smallest_p_l42_42483


namespace ratio_of_perimeters_l42_42800

theorem ratio_of_perimeters
  (s : ℝ)
  (h_pos : s > 0) : 
  let large_perimeter := 2 * (s + s / 2)
  let small_perimeter := 2 * (s + s / 4)
  in small_perimeter / large_perimeter = 5 / 6 := 
by
  sorry

end ratio_of_perimeters_l42_42800


namespace equation_of_parabola_equation_of_line_AB_l42_42256

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42256


namespace equation_of_parabola_equation_of_line_AB_l42_42251

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42251


namespace word_value_at_l42_42470

def letter_value (c : Char) : ℕ :=
  if 'A' ≤ c ∧ c ≤ 'Z' then c.toNat - 'A'.toNat + 1 else 0

def word_value (s : String) : ℕ :=
  let sum_values := s.toList.map letter_value |>.sum
  sum_values * s.length

theorem word_value_at : word_value "at" = 42 := by
  sorry

end word_value_at_l42_42470


namespace bricks_required_l42_42734

theorem bricks_required (courtyard_length courtyard_breadth : ℕ) (brick_length brick_breadth : ℕ) 
    (hl : courtyard_length = 25) (hb : courtyard_breadth = 15) 
    (hbl : brick_length = 20) (hbb : brick_breadth = 10) :
    (courtyard_length * 100 * courtyard_breadth * 100) / (brick_length * brick_breadth) = 18750 :=
by
  rw [hl, hb, hbl, hbb]
  norm_num
  sorry

end bricks_required_l42_42734


namespace optical_power_and_distance_proof_l42_42795

def point_light_source_distance : ℝ := 10 -- in cm
def direct_image_distance : ℝ := 20 -- in cm

def magnification (x y : ℝ) : ℝ := y / x

def focal_length (x : ℝ) (Γ : ℝ) : ℝ := Γ * x

def optical_power (f d : ℝ) : ℝ := (1 / d) - (1 / f)

def distance_between_source_and_image (x y d f : ℝ) : ℝ := 
  Real.sqrt ((x - y) ^ 2 + (d - f) ^ 2)

theorem optical_power_and_distance_proof :
  magnification point_light_source_distance direct_image_distance = 2 →
  focal_length point_light_source_distance (magnification point_light_source_distance direct_image_distance) = 20 →
  optical_power 20 point_light_source_distance = 5 ∧
  distance_between_source_and_image point_light_source_distance direct_image_distance point_light_source_distance 20 ≈ 14.1 :=
  by sorry

end optical_power_and_distance_proof_l42_42795


namespace circumscribed_circle_radius_l42_42581

-- Definitions of side lengths
def a : ℕ := 5
def b : ℕ := 12

-- Defining the hypotenuse based on the Pythagorean theorem
def hypotenuse (a b : ℕ) : ℕ := Nat.sqrt (a * a + b * b)

-- Radius of the circumscribed circle of a right triangle
def radius (hypotenuse : ℕ) : ℕ := hypotenuse / 2

-- Theorem: The radius of the circumscribed circle of the right triangle is 13 / 2 = 6.5
theorem circumscribed_circle_radius : 
  radius (hypotenuse a b) = 13 / 2 :=
by
  sorry

end circumscribed_circle_radius_l42_42581


namespace problem1_problem2_l42_42265

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42265


namespace marbles_steve_now_l42_42314
-- Import necessary libraries

-- Define the initial conditions as given in a)
def initial_conditions (sam steve sally : ℕ) := sam = 2 * steve ∧ sally = sam - 5 ∧ sam - 6 = 8

-- Define the proof problem statement
theorem marbles_steve_now (sam steve sally : ℕ) (h : initial_conditions sam steve sally) : steve + 3 = 10 :=
sorry

end marbles_steve_now_l42_42314


namespace sin_nine_pi_over_two_plus_theta_l42_42526

variable (θ : ℝ)

-- Conditions: Point A(4, -3) lies on the terminal side of angle θ
def terminal_point_on_angle (θ : ℝ) : Prop :=
  let x := 4
  let y := -3
  let hypotenuse := Real.sqrt ((x ^ 2) + (y ^ 2))
  hypotenuse = 5 ∧ Real.cos θ = x / hypotenuse

theorem sin_nine_pi_over_two_plus_theta (θ : ℝ) 
  (h : terminal_point_on_angle θ) : 
  Real.sin (9 * Real.pi / 2 + θ) = 4 / 5 :=
sorry

end sin_nine_pi_over_two_plus_theta_l42_42526


namespace no_real_solution_for_x_l42_42901

theorem no_real_solution_for_x
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/x = 1/3) :
  false :=
by
  sorry

end no_real_solution_for_x_l42_42901


namespace calculate_3_Delta_4_l42_42559

def Delta (p q : ℝ) : ℝ := (p^2 + q^2) / (1 + p * q)

theorem calculate_3_Delta_4 : Delta 3 4 = 25 / 13 := by
  sorry

end calculate_3_Delta_4_l42_42559


namespace minimum_value_of_a2b2_l42_42890

  -- Definitions and conditions derived from the problem statement
  variables (A B C : Type) [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
  
  variables (a b c : A) (cosC : B)
  
  def given_conditions (a b c : A) (cosC : B) : Prop :=
    (a + b)^2 = 10 + c^2 ∧ cosC = 2 / 3
  
  theorem minimum_value_of_a2b2 (a b c : A) (cosC : B)
    (h : given_conditions a b c cosC) :
    (∃ m : A, (∀ x y : A, (x + y)^2 = 10 + c^2 ∧ cosC = 2 / 3 → x^2 + y^2 ≥ m) ∧ m = 6) :=
  by
    sorry
  
end minimum_value_of_a2b2_l42_42890


namespace difference_in_dollars_l42_42930

-- Problem Statement
theorem difference_in_dollars (initial_dollars : ℕ) (additional_dollars : ℕ) :
  initial_dollars = 2 → additional_dollars = 13 → (additional_dollars + initial_dollars) - initial_dollars = 13 :=
by
  intros h1 h2
  simp [h1, h2]
  exact h2

end difference_in_dollars_l42_42930


namespace f_monotonic_increasing_g_no_zero_points_l42_42079

-- Define f(x) = 2^x + 2^{-x}
def f (x : ℝ) : ℝ := 2^x + 2^(-x)

-- f(x) is monotonically increasing on (0, +infty)
theorem f_monotonic_increasing (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) : f(x) < f(y) := sorry

-- Define g(x) piecewise
def g (x a : ℝ) : ℝ :=
  if x > 0 then f x + a else x^2 + 2*a*x + 1

-- g(x) has no zero points for the range of a ∈ [-2, 1)
theorem g_no_zero_points (a : ℝ) (h_a : -2 ≤ a ∧ a < 1) : ∀ x : ℝ, g x a ≠ 0 := sorry

end f_monotonic_increasing_g_no_zero_points_l42_42079


namespace arithmetic_sequence_general_term_arithmetic_sequence_log_sum_l42_42143

theorem arithmetic_sequence_general_term :
  (∃ (a₃ a₄ : ℝ) (S₇ : ℝ) (a n : ℝ → ℝ), a₃ + a₄ = 12 ∧ S₇ = 49 ∧ 
    a₃ = a 3 ∧ a₄ = a 4 ∧ S₇ = 7 * a 1 + 21) →
  (∀ (n : ℕ), a n = 2 * n - 1) :=
sorry

theorem arithmetic_sequence_log_sum :
  (∀ (a₃ a₄ : ℝ) (S₇ : ℝ),  a₃ + a₄ = 12 ∧ S₇ = 49) →
  (∃ (b : ℕ → ℤ), b n = Int.floor (Real.log10 (2 * n - 1))) →
  (∑ k in (finset.range 2000).map (λ n, n +1), b k = 5445) :=
sorry

end arithmetic_sequence_general_term_arithmetic_sequence_log_sum_l42_42143


namespace sara_pumpkins_l42_42315

-- Define the total number of pumpkins grown by Sara
def total_pumpkins : ℕ := 150

-- Define the percentage of pumpkins eaten by rabbits
def percentage_eaten : ℝ := 0.35

-- Calculate the number of pumpkins eaten by the rabbits 
def pumpkins_eaten : ℕ := (percentage_eaten * total_pumpkins).floor

-- Calculate the number of pumpkins left
def pumpkins_left : ℕ := total_pumpkins - pumpkins_eaten

-- Prove the number of pumpkins left is 98
theorem sara_pumpkins : pumpkins_left = 98 :=
by {
  -- Proof is omitted
  sorry
}

end sara_pumpkins_l42_42315


namespace dot_product_AM_AN_l42_42787

noncomputable def line1 : set (ℝ × ℝ) := { p | p.1 - p.2 + 1 = 0 }
noncomputable def line2 : set (ℝ × ℝ) := { p | 2 * p.1 + p.2 - 1 = 0 }
noncomputable def circleC : set (ℝ × ℝ) := { p | (p.1 - 2) ^ 2 + (p.2 - 3) ^ 2 = 1 }

def intersection_point : (ℝ × ℝ) :=
if h : ∃ p : ℝ × ℝ, p ∈ line1 ∧ p ∈ line2 then Classical.some h else (0, 1)

noncomputable def line_through_A_with_slope (k : ℝ) : set (ℝ × ℝ) := { p | p.2 = k * p.1 + 1 }

noncomputable def intersection_with_circle (k : ℝ) : set (ℝ × ℝ) :=
circleC ∩ line_through_A_with_slope k

theorem dot_product_AM_AN {k : ℝ} :
  let M := Classical.some (exists_pair_mem (intersection_with_circle k)),
      N := Classical.some (exists_pair_mem (intersection_with_circle k)) 
  in
  let A := intersection_point in
  let vector_AM := ((M.1 - A.1), (M.2 - A.2)),
      vector_AN := ((N.1 - A.1), (N.2 - A.2)) in
  vector_AM.1 * vector_AN.1 + vector_AM.2 * vector_AN.2 = 7 :=
by
  sorry

end dot_product_AM_AN_l42_42787


namespace no_two_digit_integer_satisfies_condition_l42_42032

theorem no_two_digit_integer_satisfies_condition :
  ¬ ∃ (N : ℕ), N / 10 ≠ 0 ∧ N / 100 = 0 ∧ 
  (let t := N / 10 in let u := N % 10 in 
   ∃ (k : ℕ), 11 * (t + u) = 2 * k^2) := 
sorry

end no_two_digit_integer_satisfies_condition_l42_42032


namespace expected_winnings_l42_42400

theorem expected_winnings :
  let p_heads : ℚ := 1 / 4
  let p_tails : ℚ := 1 / 2
  let p_edge : ℚ := 1 / 4
  let win_heads : ℚ := 1
  let win_tails : ℚ := 3
  let loss_edge : ℚ := -8
  (p_heads * win_heads + p_tails * win_tails + p_edge * loss_edge) = -0.25 := 
by sorry

end expected_winnings_l42_42400


namespace find_value_of_expression_l42_42415

variable (p q r s : ℝ)

def g (x : ℝ) : ℝ := p * x ^ 3 + q * x ^ 2 + r * x + s

-- We state the condition that g(1) = 1
axiom g_at_one : g p q r s 1 = 1

-- Now, we state the problem we need to prove:
theorem find_value_of_expression : 5 * p - 3 * q + 2 * r - s = 5 :=
by
  -- We skip the proof here
  exact sorry

end find_value_of_expression_l42_42415


namespace part_one_part_two_l42_42234

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42234


namespace molecular_weight_of_complex_compound_l42_42024

def molecular_weight (n : ℕ) (N_w : ℝ) (o : ℕ) (O_w : ℝ) (h : ℕ) (H_w : ℝ) (p : ℕ) (P_w : ℝ) : ℝ :=
  (n * N_w) + (o * O_w) + (h * H_w) + (p * P_w)

theorem molecular_weight_of_complex_compound :
  molecular_weight 2 14.01 5 16.00 3 1.01 1 30.97 = 142.02 :=
by
  sorry

end molecular_weight_of_complex_compound_l42_42024


namespace solution_set_of_inequality_l42_42463

theorem solution_set_of_inequality : {x : ℝ | 8 * x^2 + 6 * x ≤ 2} = { x : ℝ | -1 ≤ x ∧ x ≤ (1/4) } :=
sorry

end solution_set_of_inequality_l42_42463


namespace max_product_distinct_digits_l42_42624

/-- Four distinct digits and a maximum product such that E * F solves the given problem. -/
theorem max_product_distinct_digits :
  ∃ (E F G H : ℕ), E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H ∧
  E ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  F ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  G ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  H ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  E * F > 0 ∧
  G > H ∧
  (E * F) % (G - H) = 0 ∧
  E * F = 72 :=
sorry

end max_product_distinct_digits_l42_42624


namespace find_cos_theta_l42_42518

noncomputable def theta : ℝ := sorry
axiom theta_in_range : 0 < theta ∧ theta < π
axiom tan_theta : Real.tan θ = -3/2

theorem find_cos_theta : Real.cos θ = -2/Real.sqrt 13 := sorry

end find_cos_theta_l42_42518


namespace service_charge_percentage_l42_42302

-- Define the initial and final account balances
def initial_balance : ℝ := 400
def final_balance : ℝ := 307

-- Define the transactions amounts
def transaction1 : ℝ := 90
def transaction2 : ℝ := 60

-- Let x be the percentage of the service charge
def x : ℝ := 2

-- Define the service charge amounts
def service_charge1 := transaction1 * x / 100
def service_charge2 := transaction2 * x / 100

-- Define the total amount deducted, considering the second transaction was reversed without the service charge
def total_deducted := transaction1 + service_charge1 + service_charge2

-- The proof statement
theorem service_charge_percentage : x = 2 :=
by
  have h1 : total_deducted = initial_balance - final_balance := sorry
  have h2 : total_deducted = 93 := sorry
  have h3 : transaction1 + 1.5 * x = 93 := sorry
  have h4 : 1.5 * x = 3 := sorry
  have h5 : x = 2 := sorry
  exact h5

end service_charge_percentage_l42_42302


namespace parabola_equation_line_AB_equation_l42_42226

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42226


namespace total_valid_four_digit_numbers_l42_42408

-- Define the sets of digit conditions
def valid_digits (A B : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ A ≠ B

-- Define the ABAB format
def is_ABAB (n : ℕ) : Prop :=
  ∃ A B, valid_digits A B ∧ n = 101 * (10 * A + B)

-- Define the AABB format
def is_AABB (n : ℕ) : Prop :=
  ∃ A B, valid_digits A B ∧ n = 1100 * A + 11 * B

-- Define the ABBA format
def is_ABBA (n : ℕ) : Prop :=
  ∃ A B, valid_digits A B ∧ n = 1001 * A + 110 * B

-- Define divisibility by 7 or 101 but not both
def divisible_by_7_or_101_but_not_both (n : ℕ) : Prop :=
  (n % 7 = 0 ∨ n % 101 = 0) ∧ ¬((n % 7 = 0) ∧ (n % 101 = 0))

-- Define the set of valid four-digit numbers
def valid_four_digit_numbers : Finset ℕ :=
  Finset.filter (λ n, 1000 ≤ n ∧ n < 10000 ∧ 
                       (is_ABAB n ∨ is_AABB n ∨ is_ABBA n) ∧ 
                       divisible_by_7_or_101_but_not_both n) 
                (Finset.range 10000)

-- Statement to be proven
theorem total_valid_four_digit_numbers : valid_four_digit_numbers.card = 97 :=
sorry

end total_valid_four_digit_numbers_l42_42408


namespace log_eqn_solutions_l42_42322

theorem log_eqn_solutions :
  {x : ℝ // log 2 (x^3 - 18 * x^2 + 81 * x) = 4} = {1, 2, 8} :=
begin
  sorry
end

end log_eqn_solutions_l42_42322


namespace equation_of_parabola_equation_of_line_AB_l42_42253

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42253


namespace parabola_equation_line_AB_equation_l42_42186

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42186


namespace marcella_matching_pairs_l42_42388

theorem marcella_matching_pairs (P : ℕ) (L : ℕ) (H : P = 20) (H1 : L = 9) : (P - L) / 2 = 11 :=
by
  -- definition of P and L are given by 20 and 9 respectively
  -- proof is omitted for the statement focus
  sorry

end marcella_matching_pairs_l42_42388


namespace number_of_ways_to_group_dogs_l42_42452

theorem number_of_ways_to_group_dogs (dogs : Finset ℕ) (Fluffy Nipper : ℕ) (d1 d2 d3 : Finset ℕ) 
  (h_dogs_card : dogs.card = 12) 
  (h_disjoint : dogs = d1 ∪ d2 ∪ d3 ∧ d1 ∩ d2 = ∅ ∧ d1 ∩ d3 = ∅ ∧ d2 ∩ d3 = ∅ )
  (h_d1_card : d1.card = 4) 
  (h_d2_card : d2.card = 6) 
  (h_d3_card : d3.card = 2) 
  (h_Fluffy : Fluffy ∈ d1) 
  (h_Nipper : Nipper ∈ d2) : 
  ∃ (n : ℕ), n = 2520 :=
by 
  use 2520
  sorry

end number_of_ways_to_group_dogs_l42_42452


namespace exponent_multiplication_rule_l42_42440

theorem exponent_multiplication_rule :
  3000 * (3000 ^ 3000) = 3000 ^ 3001 := 
by {
  sorry
}

end exponent_multiplication_rule_l42_42440


namespace exists_commuting_matrices_in_S_l42_42608

def is_square (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

def matrix_square_entries (A : Matrix (Fin 2) (Fin 2) ℤ) : Prop := 
  ∀ (i j : Fin 2), is_square (A i j) ∧ A i j ≤ 200

def elements_commute (A B : Matrix (Fin 2) (Fin 2) ℤ) : Prop :=
  A * B = B * A

theorem exists_commuting_matrices_in_S
  (S : Finset (Matrix (Fin 2) (Fin 2) ℤ)) 
  (h1 : ∀ A ∈ S, matrix_square_entries A)
  (h2 : S.card > 50387) :
  ∃ A B ∈ S, A ≠ B ∧ elements_commute A B :=
sorry

end exists_commuting_matrices_in_S_l42_42608


namespace area_of_region_l42_42441

noncomputable def region_area : ℝ :=
  ∫ t in (2 * Real.pi / 3)..(4 * Real.pi / 3), 16 * (1 - cos t) ^ 2

theorem area_of_region :
  region_area = 16 * Real.pi :=
by
  -- proof steps will go here
  sorry

end area_of_region_l42_42441


namespace parabola_equation_line_AB_equation_l42_42227

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42227


namespace intersection_M_N_l42_42927

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

theorem intersection_M_N : M ∩ N = {0, 3} := by
  sorry

end intersection_M_N_l42_42927


namespace conclusion_1_conclusion_2_l42_42866

open Function

-- Conclusion ①
theorem conclusion_1 {f : ℝ → ℝ} (h : StrictMono f) :
  ∀ {x1 x2 : ℝ}, f x1 ≤ f x2 ↔ x1 ≤ x2 := 
by
  intros x1 x2
  exact h.le_iff_le

-- Conclusion ②
theorem conclusion_2 {f : ℝ → ℝ} (h : ∀ x, f x ^ 2 = f (-x) ^ 2) :
  ¬ (∀ x, f (-x) = f x ∨ f (-x) = -f x) :=
by
  sorry

end conclusion_1_conclusion_2_l42_42866


namespace part1_eq_C_part2_line_AB_l42_42287

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42287


namespace solution_set_of_inequality_l42_42337

noncomputable def f (x : ℝ) : ℝ :=
if x > 1 then x else -1

theorem solution_set_of_inequality :
  {x : ℝ | x * f x - x ≤ 2} = set.Icc (-1 : ℝ) 2 :=
by
  sorry

end solution_set_of_inequality_l42_42337


namespace remainder_x2040_minus_1_div_by_x9_minus_x7_plus_x5_minus_x3_plus_1_zero_l42_42036

theorem remainder_x2040_minus_1_div_by_x9_minus_x7_plus_x5_minus_x3_plus_1_zero :
  polynomial.eval (x^2040 - 1) (polynomial.X^9 - polynomial.X^7 + polynomial.X^5 - polynomial.X^3 + 1) % polynomial.X^9 - polynomial.X^7 + polynomial.X^5 - polynomial.X^3 + 1 = 0 := 
by sorry

end remainder_x2040_minus_1_div_by_x9_minus_x7_plus_x5_minus_x3_plus_1_zero_l42_42036


namespace inverse_proportional_ratios_l42_42656

variables {x y x1 x2 y1 y2 : ℝ}
variables (h_inv_prop : ∀ (x y : ℝ), x * y = 1) (hx1_ne : x1 ≠ 0) (hx2_ne : x2 ≠ 0) (hy1_ne : y1 ≠ 0) (hy2_ne : y2 ≠ 0)

theorem inverse_proportional_ratios 
  (h1 : x1 * y1 = x2 * y2)
  (h2 : (x1 / x2) = (3 / 4)) : 
  (y1 / y2) = (4 / 3) :=
by 

sorry

end inverse_proportional_ratios_l42_42656


namespace distance_from_two_eq_three_l42_42300

theorem distance_from_two_eq_three (x : ℝ) (h : |x - 2| = 3) : x = -1 ∨ x = 5 :=
sorry

end distance_from_two_eq_three_l42_42300


namespace parallel_vectors_k_value_l42_42092

theorem parallel_vectors_k_value :
  (∃ k : ℝ, let a := (2 : ℝ, -1 : ℝ) in
           let b := (k, 5 / 2) in
           a.1 * b.2 = a.2 * b.1) ↔ k = -5 :=
by
  sorry

end parallel_vectors_k_value_l42_42092


namespace problem_l42_42717

theorem problem (K : ℕ) : 16 ^ 3 * 8 ^ 3 = 2 ^ K → K = 21 := by
  sorry

end problem_l42_42717


namespace factor_check_l42_42301

theorem factor_check :
  ∃ (f : ℕ → ℕ) (x : ℝ), f 1 = (x^2 - 2 * x + 3) ∧ f 2 = 29 * 37 * x^4 + 2 * x^2 + 9 :=
by
  let f : ℕ → ℕ := sorry -- Define a sequence or function for the proof context
  let x : ℝ := sorry -- Define the variable x in our context
  have h₁ : f 1 = (x^2 - 2 * x + 3) := sorry -- Establish the first factor
  have h₂ : f 2 = 29 * 37 * x^4 + 2 * x^2 + 9 := sorry -- Establish the polynomial expression
  exact ⟨f, x, h₁, h₂⟩ -- Use existential quantifier to capture the required form

end factor_check_l42_42301


namespace problem1_problem2_l42_42264

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42264


namespace find_m_l42_42074

def determinant (a b c d : ℂ) : ℂ := a*d - b*c

theorem find_m (z m : ℂ) (h1 : determinant z complex.i m complex.i = 1 - 2*complex.i)
  (h2 : ∀ (reIm : ℝ → z = 0), (∃ (reIm = z = ⟪z.re⟫) : = 0) :
  m = 2 :=
begin
  sorry
end

end find_m_l42_42074


namespace midpoint_of_Harry_and_Sandy_l42_42096

def midpoint (p1 p2 : Prod ℝ ℝ) : Prod ℝ ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_of_Harry_and_Sandy :
  midpoint (2, -3) (-4, 5) = (-1, 1) :=
by
  sorry

end midpoint_of_Harry_and_Sandy_l42_42096


namespace find_function_l42_42990

open Real

theorem find_function (f : ℝ → ℝ) :
  let g := λ x, f (x / 2);
  let h := λ x, g (x - π / 3);
  (∀ x, h x = sin (x - π / 4)) →
  ∀ x, f x = sin (x / 2 + π / 12) :=
by
  intros g h h_eq x
  sorry

end find_function_l42_42990


namespace lowest_population_increase_l42_42141

-- Definitions based on conditions
def option_A : Prop := "Africa and North America have the lowest rate of natural population increase."
def option_B : Prop := "Asia and Africa have the lowest rate of natural population increase."
def option_C : Prop := "Europe and North America have the lowest rate of natural population increase."
def option_D : Prop := "Europe and South America have the lowest rate of natural population increase."

-- Fact based on the statement that Europe and parts of North America (like Canada) are developed countries
axiom developed_countries_low_growth : "Developed countries, like Europe and parts of North America, have low or negative natural population growth rates."

-- The proof goal based on the question and correct answer
theorem lowest_population_increase : option_C := by
  sorry

end lowest_population_increase_l42_42141


namespace ratio_proof_l42_42405

-- Define the given conditions
def base_AB : ℝ := 40
def ratio_AD_AB : ℝ := 4 / 3
def radius : ℝ := base_AB / 4 -- Radius is 10

-- Define the areas
def area_rectangle : ℝ := ratio_AD_AB * base_AB * base_AB
def semi_major_axis : ℝ := base_AB / 2
def semi_minor_axis : ℝ := radius
def area_semi_ellipses : ℝ := π * semi_major_axis * semi_minor_axis
def area_triangle : ℝ := (1 / 2) * base_AB * radius
def combined_area : ℝ := area_semi_ellipses + area_triangle

-- Final ratio calculation
def ratio : ℝ := area_rectangle / combined_area

-- Proof statement
theorem ratio_proof : ratio = 32 / (3 * (π + 1)) :=
by sorry

end ratio_proof_l42_42405


namespace determinant_inequality_l42_42460

open Real

def det (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (x : ℝ) :
  det 7 (x^2) 2 1 > det 3 (-2) 1 x ↔ -5/2 < x ∧ x < 1 :=
by
  sorry

end determinant_inequality_l42_42460


namespace explicit_formula_triangle_perimeter_l42_42828

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem explicit_formula :
  (∃ (ω : ℝ), ω > 0 ∧ (∃ (ϕ : ℝ), |ϕ| < Real.pi / 2 ∧ (∃ x, (1, -Real.sqrt 3) = (Real.cos ϕ, Real.sin ϕ))) 
  ∧ (∃ x1 x2, |f (x1) - f (x2)| = 2 ∧ |x1 - x2| = Real.pi / 2)) → 
  f = (λ x, Real.sin (2 * x - Real.pi / 3)) :=
begin
  intro h,
  sorry
end

theorem triangle_perimeter {a b c : ℝ} {C : ℝ} :
  (∃ A B : ℝ, 
    1 / 2 * A * B * Real.sin C = 5 * Real.sqrt 3 ∧ 
    c = 2 * Real.sqrt 5 ∧ 
    Real.cos C = f (Real.pi / 4)) → 
  ∃ P : ℝ, P = 6 * Real.sqrt 5 :=
begin
  intro h,
  obtain ⟨A, B, hab, hc, hcos⟩ := h,
  let perimeter := A + B + c,
  use perimeter,
  sorry
end

end explicit_formula_triangle_perimeter_l42_42828


namespace determine_angle_l42_42149

noncomputable def angle_in_eq_triangle (D E : Point) (A : Point) [H1 : equilateral_triangle A D E] : 
  angle D A E = 60 := sorry

def ab_equals_3ac (AB AC : ℝ) : Prop := AB = 3 * AC

theorem determine_angle (A B C D E : Point) 
  (h1 : ab_equals_3ac (dist A B) (dist A C))
  (h2 : equilateral_triangle A D E)
  (h3 : angle D A B = angle D A C) :
  angle C A B = 30 := sorry

end determine_angle_l42_42149


namespace transformed_function_is_correct_l42_42973

noncomputable def f : ℝ → ℝ := λ x, sin (x / 2 + π / 12)

theorem transformed_function_is_correct :
  ∀ x : ℝ, sin (2 * (x - π / 3)) = sin (x - π / 4) :=
by
  intros x
  symmetry
  apply eq.symm
  calc
    sin (2 * (x - π / 3)) = sin (2x - 2 * π / 3) : by rw mul_sub
    _ = sin (x - π / 4) : sorry

end transformed_function_is_correct_l42_42973


namespace modulus_and_quadrant_of_z_values_of_a_and_b_l42_42527

open Complex Real

-- Define the given complex number z
def z := (1 + I)^2 + 2 * (5 - I) / (3 + I)

-- The first part: Prove the modulus and quadrant of z
theorem modulus_and_quadrant_of_z : abs z = sqrt 10 ∧ (Re z > 0 ∧ Im z < 0) :=
sorry

-- The second part: Prove the values of a and b
theorem values_of_a_and_b (a b: ℝ) (h: z = 3 - I) : 
  (z * (z + a) = b + I) → a = -7 ∧ b = -13 :=
sorry

end modulus_and_quadrant_of_z_values_of_a_and_b_l42_42527


namespace prove_q_min_value_zero_l42_42530

noncomputable def find_q (p : ℝ) : ℝ :=
  -((-1/3)^3 + (-1/3)^2) -- This is the pre-computed smallest value of y when p = 0

theorem prove_q_min_value_zero (p : ℝ) : 
  ∃ q : ℝ, (∀ y : ℝ, (y = x^3 + x^2 + p * x + q ∧ y >= 0) → y = 0) ∧ q = -2/27 :=
by
  existsi q
  sorry

end prove_q_min_value_zero_l42_42530


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42274

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42274


namespace complete_the_square_l42_42710

-- Definition of the initial condition
def eq1 : Prop := ∀ x : ℝ, x^2 + 4 * x + 1 = 0

-- The goal is to prove if the initial condition holds, then the desired result holds.
theorem complete_the_square (x : ℝ) (h : x^2 + 4 * x + 1 = 0) : (x + 2)^2 = 3 := by
  sorry

end complete_the_square_l42_42710


namespace sandcastle_ratio_l42_42695

-- Definitions based on conditions in a)
def sandcastles_on_marks_beach : ℕ := 20
def towers_per_sandcastle_marks_beach : ℕ := 10
def towers_per_sandcastle_jeffs_beach : ℕ := 5
def total_combined_sandcastles_and_towers : ℕ := 580

-- The main statement to prove
theorem sandcastle_ratio : 
  ∃ (J : ℕ), 
  (sandcastles_on_marks_beach + (towers_per_sandcastle_marks_beach * sandcastles_on_marks_beach) + J + (towers_per_sandcastle_jeffs_beach * J) = total_combined_sandcastles_and_towers) ∧ 
  (J / sandcastles_on_marks_beach = 3) :=
by 
  sorry

end sandcastle_ratio_l42_42695


namespace ethan_expected_wins_l42_42468

-- Define the conditions
def P_win := 2 / 5
def P_tie := 2 / 5
def P_loss := 1 / 5

-- Define the adjusted probabilities
def adj_P_win := P_win / (P_win + P_loss)
def adj_P_loss := P_loss / (P_win + P_loss)

-- Define Ethan's expected number of wins before losing
def expected_wins_before_loss : ℚ := 2

-- The theorem to prove 
theorem ethan_expected_wins :
  ∃ E : ℚ, 
    E = (adj_P_win * (E + 1) + adj_P_loss * 0) ∧ 
    E = expected_wins_before_loss :=
by
  sorry

end ethan_expected_wins_l42_42468


namespace average_time_rounded_to_4_l42_42368

def train1_distance : ℝ := 200
def train1_speed : ℝ := 50
def train2_distance : ℝ := 240
def train2_speed : ℝ := 80

def time_train1 := train1_distance / train1_speed
def time_train2 := train2_distance / train2_speed

def total_time := time_train1 + time_train2
def average_time := total_time / 2
def rounded_average_time := Real.ceil (average_time + 0.5) - 1

theorem average_time_rounded_to_4 : rounded_average_time = 4 := 
by sorry

end average_time_rounded_to_4_l42_42368


namespace find_parabola_equation_find_line_AB_equation_l42_42244

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42244


namespace distinct_even_numbers_between_100_and_999_l42_42933

def count_distinct_even_numbers_between_100_and_999 : ℕ :=
  let possible_units_digits := 5 -- {0, 2, 4, 6, 8}
  let possible_hundreds_digits := 8 -- {1, 2, ..., 9} excluding the chosen units digit
  let possible_tens_digits := 8 -- {0, 1, 2, ..., 9} excluding the chosen units and hundreds digits
  possible_units_digits * possible_hundreds_digits * possible_tens_digits

theorem distinct_even_numbers_between_100_and_999 : count_distinct_even_numbers_between_100_and_999 = 320 :=
  by sorry

end distinct_even_numbers_between_100_and_999_l42_42933


namespace actual_distance_in_km_l42_42630

-- Given conditions
def scale_factor : ℕ := 200000
def map_distance_cm : ℚ := 3.5

-- Proof goal: the actual distance in kilometers
theorem actual_distance_in_km : (map_distance_cm * scale_factor) / 100000 = 7 := 
by
  sorry

end actual_distance_in_km_l42_42630


namespace distance_C1_to_plane_ABC1_AEB1_l42_42582

variable (A B C A1 B1 C1 E : Type)
variable (distance : Type → Type → ℝ)

-- condition: Right triangular prism ABC-A1B1C1 with base and lateral edges 2
variable (base_edge_length lateral_edge_length : ℝ)
variable (right_triangular_prism : (A B C A1 B1 C1 : Type) → Bool)

-- Midpoint condition
variable (midpoint: C C1 E → Prop)

-- Main theorem to be proved
theorem distance_C1_to_plane_ABC1_AEB1 
    (h1 : right_triangular_prism A B C A1 B1 C1)
    (h2: base_edge_length = 2)
    (h3: lateral_edge_length = 2)
    (h4: midpoint C C1 E) : 
    distance C1 (A B B1 E) = (Real.sqrt 2) / 2 := 
sorry

end distance_C1_to_plane_ABC1_AEB1_l42_42582


namespace equilateral_triangle_locus_l42_42061

noncomputable def point_locus (A B C P : Point) (triangle : is_equilateral_triangle A B C) :
  set Point := 
  {P | ∠APB = ∠BPC}

theorem equilateral_triangle_locus (A B C : Point) 
  (hABC: is_equilateral_triangle A B C)
  (P : Point) :
  P ∈ point_locus A B C P hABC ↔ (P lies on the angle bisector of ∠ABC ∨ 
                                  P lies on the minor arc between A and C on the circumcircle of △ABC ∨ 
                                  (P lies on the line AC ∧ P ≠ A ∧ P ≠ C)) :=
sorry

end equilateral_triangle_locus_l42_42061


namespace find_parabola_equation_find_line_AB_equation_l42_42246

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42246


namespace k_equals_3_l42_42909

-- Define the function
def f (x k : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + k^3 * x

-- Define the condition where the derivative at x = 0 equals 27
def f_prime_at_zero_is_27 (k : ℝ) : Prop :=
  deriv (f x k) 0 = 27

-- Define the goal to prove k = 3 
theorem k_equals_3 (k : ℝ) (h : f_prime_at_zero_is_27 k) : k = 3 :=
by
  sorry

end k_equals_3_l42_42909


namespace elberta_money_l42_42545

theorem elberta_money (g a e : ℕ) 
  (hg : g = 100)
  (ha : a = 2 * g / 5)
  (he : e = a + 5) : 
  e = 45 :=
by {
  rw [hg, ha, he],
  norm_num,
}

end elberta_money_l42_42545


namespace problem_1_problem_2_l42_42196

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42196


namespace find_parabola_equation_find_line_AB_equation_l42_42239

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42239


namespace minimal_pieces_correct_l42_42785

noncomputable def minimal_pieces (p q : ℕ) [Nat.coprime p q] : ℕ :=
  p + q - 1

theorem minimal_pieces_correct (p q : ℕ) [Nat.coprime p q] :
  minimal_pieces p q = p + q - 1 :=
by
  sorry

end minimal_pieces_correct_l42_42785


namespace stick_numbers_total_results_l42_42329

-- Definitions based on the given conditions
def cards : List ℕ := [1, 2, 3, 4]
def slots : List ℕ := [1, 2, 3, 4]

-- Main statement proving the total number of valid results
theorem stick_numbers_total_results :
  let valid_permutations := {f : ℕ → ℕ | ∀ i, f i ≠ i ∧ ∃ j, f 1 = j} -- Function from cards to slots with specific properties
  finset.card valid_permutations = 11 :=
sorry

end stick_numbers_total_results_l42_42329


namespace infinite_solutions_b_l42_42869

theorem infinite_solutions_b (x b : ℝ) : 
    (∀ x, 4 * (3 * x - b) = 3 * (4 * x + 16)) → b = -12 :=
by
  sorry

end infinite_solutions_b_l42_42869


namespace midpoints_on_nine_point_circle_l42_42000

open EuclideanGeometry

-- Define the triangle and excircle centers
variables {A B C O₁ O₂ O₃ : Point}
variable {triangle_ABC : Triangle A B C}
variable {excircle_center1 : CenterExcircle A B C O₁}
variable {excircle_center2 : CenterExcircle B C A O₂}
variable {excircle_center3 : CenterExcircle C A B O₃}

-- Define the midpoints of the segments connecting excircle centers
noncomputable def midpoint_O₁_O₂ : Point := midpoint O₁ O₂
noncomputable def midpoint_O₂_O₃ : Point := midpoint O₂ O₃
noncomputable def midpoint_O₃_O₁ : Point := midpoint O₃ O₁

-- Define the nine-point circle of triangle ABC
noncomputable def nine_point_circle : Circle := ninePointCircle triangle_ABC

-- The theorem statement
theorem midpoints_on_nine_point_circle :
  midpoint_O₁_O₂ ∈ nine_point_circle ∧
  midpoint_O₂_O₃ ∈ nine_point_circle ∧
  midpoint_O₃_O₁ ∈ nine_point_circle :=
sorry

end midpoints_on_nine_point_circle_l42_42000


namespace problem_statement_l42_42009

theorem problem_statement : 2009 * 20082008 - 2008 * 20092009 = 0 := by
  sorry

end problem_statement_l42_42009


namespace solved_fraction_equation_l42_42486

theorem solved_fraction_equation :
  ∀ (x : ℚ),
    x ≠ 2 →
    x ≠ 7 →
    x ≠ -5 →
    (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 4*x - 5) / (x^2 - 2*x - 35) →
    x = 55 / 13 := by
  sorry

end solved_fraction_equation_l42_42486


namespace greatest_whole_number_satisfies_inequality_l42_42475

theorem greatest_whole_number_satisfies_inequality : 
  ∃ (x : ℕ), (∀ (y : ℕ), (6 * y - 4 < 5 - 3 * y) → y ≤ x) ∧ x = 0 := 
sorry

end greatest_whole_number_satisfies_inequality_l42_42475


namespace infinite_composite_numbers_dividing_power_l42_42642

theorem infinite_composite_numbers_dividing_power (n : ℕ) : 
  let sequence := nat.rec (λ _, ℕ) 11 (λ i acc, 2^acc - 1)
  in ∀ i : ℕ, (sequence i ∣ 2^(sequence i) - 2) ∧ (i > 0 → ¬nat.prime (sequence i)) :=
by 
  sorry

end infinite_composite_numbers_dividing_power_l42_42642


namespace one_cubic_foot_is_1728_cubic_inches_l42_42098

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end one_cubic_foot_is_1728_cubic_inches_l42_42098


namespace problem1_problem2_l42_42267

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42267


namespace parabola_equation_line_AB_equation_l42_42221

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42221


namespace problem_1_problem_2_l42_42203

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42203


namespace distance_from_starting_point_l42_42791

namespace BobHexagonPath

-- Define the properties of the regular hexagon and Bob's walk
def side_length : ℝ := 3
def walk_distance : ℝ := 10

-- Prove that Bob is 1 km away from his starting point after walking 10 km
theorem distance_from_starting_point (h : regular_hexagon side_length) :
  distance (walk_perimeter h walk_distance) (0, 0) = 1 :=
sorry

-- Definitions and assumptions used in the proof (to make the statement compilable)
def regular_hexagon (a : ℝ) := true
def walk_perimeter {a : ℝ} (h : regular_hexagon a) (d : ℝ) : ℝ × ℝ := (1, 0)
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)
end BobHexagonPath

end distance_from_starting_point_l42_42791


namespace complex_isosceles_right_triangle_solution_l42_42548

theorem complex_isosceles_right_triangle_solution : 
  {z : ℂ | z ≠ 0 ∧ (∃ θ, z = Complex.exp (Complex.I * θ) ∧ θ = π / 2 ∨ θ = -π / 2)}.card = 2 :=
by
  sorry

end complex_isosceles_right_triangle_solution_l42_42548


namespace transformation_correct_l42_42978

noncomputable def original_function : ℝ → ℝ :=
  λ x, sin (x / 2 + π / 12)

theorem transformation_correct :
  ∀ (x : ℝ), (shift_right (shorten_abscissas f) (π / 3)) x = sin (x - π / 4) →
  f = original_function :=
by
  -- Assume the shifted and shortened function equals the given sine transformation
  intro x
  sorry

end transformation_correct_l42_42978


namespace circles_coincide_S_touches_midlines_S_l42_42177

noncomputable def triangle_incircle (ABC : Triangle) : Circle := sorry
noncomputable def homothety (c : ℝ) (p : Point) (S : Circle) : Circle := sorry
noncomputable def nagel_point (ABC : Triangle) : Point := sorry
noncomputable def centroid (ABC : Triangle) : Point := sorry
noncomputable def midlines_touching (S : Circle) (ABC : Triangle) : Prop := sorry
noncomputable def midpoints_touching (S : Circle) (ABC : Triangle) (N : Point) : Prop := sorry

def S' (ABC : Triangle) := homothety (1/2) (nagel_point ABC) (triangle_incircle ABC)
def S (ABC : Triangle) := homothety (-1/2) (centroid ABC) (triangle_incircle ABC)

theorem circles_coincide (ABC : Triangle) : S ABC = S' ABC := sorry

theorem S_touches_midlines (ABC : Triangle) : midlines_touching (S ABC) ABC := sorry

theorem S'_touches_lines (ABC : Triangle) : midpoints_touching (S' ABC) ABC (nagel_point ABC) := sorry

end circles_coincide_S_touches_midlines_S_l42_42177


namespace min_positive_period_of_f_l42_42568

-- Define the function y = 3 * sin^2(x).
def f (x : ℝ) : ℝ := 3 * (Real.sin x)^2

-- Statement of the proof problem
theorem min_positive_period_of_f : ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ ε > 0, ∃ M > 0, M < T → ∀ x : ℝ, f (x + M) ≠ f x) := 
sorry

end min_positive_period_of_f_l42_42568


namespace find_q_l42_42542

noncomputable def solution_condition (p q : ℝ) : Prop :=
  (p > 1) ∧ (q > 1) ∧ (1 / p + 1 / q = 1) ∧ (p * q = 9)

theorem find_q (p q : ℝ) (h : solution_condition p q) : 
  q = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_q_l42_42542


namespace A_union_B_eq_B_l42_42649

-- Define set A
def A : Set ℝ := {-1, 0, 1}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

-- The proof problem
theorem A_union_B_eq_B : A ∪ B = B := 
  sorry

end A_union_B_eq_B_l42_42649


namespace find_function_l42_42619

variable (p : Set ℚ) (f : ℚ → ℚ)

def satisfies_conditions (f : ℚ → ℚ) : Prop :=
  ∀ x ∈ p, f(x) + f(1 / x) = 1 ∧ f(2 * x) = 2 * f(f(x))

theorem find_function (hp : ∀ x, 0 < x → x ∈ p)
  (h1 : satisfies_conditions p f) :
  ∀ x ∈ p, f(x) = x / (x + 1) :=
by {
  sorry -- Proof goes here.
}

end find_function_l42_42619


namespace greatest_score_of_individual_player_l42_42768

theorem greatest_score_of_individual_player
  (n : ℕ) (total_score : ℕ) (p : ℕ)
  (h_n : n = 12) (h_total_score : total_score = 100) (h_min_score : p ≥ 7)
  (h_team_scores : ∀ (scores : vector ℕ n), (∀ i, scores.nth i ≥ p) → vector.sum scores = total_score) :
  ∃ (max_p : ℕ), max_p ≤ total_score ∧ (∀ (scores : vector ℕ n), (∀ i, scores.nth i ≥ p) → vector.sum scores = total_score → max_p = max (vector.to_list scores)) ∧ max_p = 23 :=
by
  sorry

end greatest_score_of_individual_player_l42_42768


namespace safe_assignment_correct_l42_42039

structure SafeAssignment :=
  (nickel : Nat)
  (silver : Nat)
  (bronze : Nat)
  (platinum : Nat)
  (gold : Nat)
  (distinct : ∀ (a b : Nat), a ≠ b -> SafeAssignment.nickel ≠ SafeAssignment.silver 
            ∧ SafeAssignment.silver ≠ SafeAssignment.bronze 
            ∧ SafeAssignment.bronze ≠ SafeAssignment.platinum 
            ∧ SafeAssignment.platinum ≠ SafeAssignment.gold 
            ∧ SafeAssignment.gold ≠ SafeAssignment.nickel)

noncomputable def gold_in_correct_safe (assign : SafeAssignment) : Prop :=
  let gold_safe := assign.gold;
  match gold_safe with
  | 1 => gold_safe = 2 ∨ gold_safe = 3
  | 2 => assign.silver = 1
  | 3 => assign.bronze ≠ 3
  | 4 => assign.nickel = gold_safe - 1
  | 5 => assign.platinum = gold_safe + 1
  | _ => false

noncomputable def correct_safe_assignment : SafeAssignment :=
  { nickel := 1, silver := 2, bronze := 3, platinum := 4, gold := 5, distinct := sorry } 

theorem safe_assignment_correct : gold_in_correct_safe correct_safe_assignment ∧ 
            correct_safe_assignment.nickel = 1 ∧ 
            correct_safe_assignment.silver = 2 ∧ 
            correct_safe_assignment.bronze = 3 ∧ 
            correct_safe_assignment.platinum = 4 ∧ 
            correct_safe_assignment.gold = 5 :=
by sorry

end safe_assignment_correct_l42_42039


namespace intersection_equals_l42_42925

-- Define Set A
def setA : Set ℝ := {x : ℝ | -x^2 + 2x + 3 > 0 }

-- Define Set B
def setB : Set ℝ := {x : ℝ | x < 2 }

-- Define the complement of Set B in ℝ
def complementB : Set ℝ := {x : ℝ | x ≥ 2 }

-- Define the intersection of Set A and the complement of Set B
def intersectionAComplementB : Set ℝ := {x : ℝ | x ∈ setA ∧ x ∈ complementB}

-- The proof statement
theorem intersection_equals : intersectionAComplementB = {x : ℝ | 2 ≤ x ∧ x < 3} := 
by 
  sorry

end intersection_equals_l42_42925


namespace find_x_such_that_g_inverse_of_x_is_neg2_l42_42916

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 3

theorem find_x_such_that_g_inverse_of_x_is_neg2 : g (-2) = -43 :=
by 
  rw [g]
  simp
  norm_num
  sorry

end find_x_such_that_g_inverse_of_x_is_neg2_l42_42916


namespace grunters_win_4_out_of_6_l42_42330

/-- The Grunters have a probability of winning any given game as 60% --/
def p : ℚ := 3 / 5

/-- The Grunters have a probability of losing any given game as 40% --/
def q : ℚ := 1 - p

/-- The binomial coefficient for choosing exactly 4 wins out of 6 games --/
def binomial_6_4 : ℚ := Nat.choose 6 4

/-- The probability that the Grunters win exactly 4 out of the 6 games --/
def prob_4_wins : ℚ := binomial_6_4 * (p ^ 4) * (q ^ 2)

/--
The probability that the Grunters win exactly 4 out of the 6 games is
exactly $\frac{4860}{15625}$.
--/
theorem grunters_win_4_out_of_6 : prob_4_wins = 4860 / 15625 := by
  sorry

end grunters_win_4_out_of_6_l42_42330


namespace students_only_english_l42_42574

theorem students_only_english (total_students B G_B : ℕ) (h_total : total_students = 45) 
    (h_B : B = 12) (h_G_B : G_B = 22) : 
    E (students enrolled only in English) = 23 :=
  have G : ℕ := G_B - B
  calc E : ℕ = total_students - (G + B)
              = 45 - (10 + 12) : by rw [h_total, h_B, h_G_B]; sorry

end students_only_english_l42_42574


namespace a2018_is_65_l42_42304

-- Define the sequence and the operations
def n₁ : ℕ := 5

def a (n : ℕ) : ℕ := n^2 + 1

def sum_of_digits (n : ℕ) : ℕ :=
to_digits 10 n |>.foldl (· + ·) 0

def seq_n (i : ℕ) : ℕ :=
if i = 1 then n₁
else sum_of_digits (seq_a (i - 1))

def seq_a (i : ℕ) : ℕ :=
a (seq_n i)

-- Define the theorem to prove
theorem a2018_is_65 : seq_a 2018 = 65 :=
sorry -- skipping the proof statement for now

end a2018_is_65_l42_42304


namespace die_condition_B_die_condition_D_l42_42874

open Real

structure DieRolls where
  rolls : List ℕ
  length_is_five : rolls.length = 5
  all_between_1_and_6 : ∀ r ∈ rolls, r ≥ 1 ∧ r ≤ 6

def median (l : List ℕ) : ℕ :=
  l.nth_le (l.length / 2) sorry  -- Assume sorted list.

def mode (l : List ℕ) : ℕ :=
  l.maximum sorry  -- Assume appropriate implementations.

def mean (l : List ℕ) : ℝ :=
  (l.sum.toReal) / (l.length.toReal)

def variance (l : List ℕ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x.toReal - m) ^ 2)).sum / (l.length.toReal)

theorem die_condition_B : ∀ (d : DieRolls),
  mean d.rolls = 3 ∧ mode d.rolls = 4 → ¬ (6 ∈ d.rolls) := by
  sorry

theorem die_condition_D : ∀ (d : DieRolls),
  mean d.rolls = 2 ∧ variance d.rolls = 2.4 → ¬ (6 ∈ d.rolls) := by
  sorry

end die_condition_B_die_condition_D_l42_42874


namespace probability_blue_or_purple_is_4_over_11_l42_42398

def total_jelly_beans : ℕ := 10 + 12 + 13 + 15 + 5
def blue_or_purple_jelly_beans : ℕ := 15 + 5
def probability_blue_or_purple : ℚ := blue_or_purple_jelly_beans / total_jelly_beans

theorem probability_blue_or_purple_is_4_over_11 :
  probability_blue_or_purple = 4 / 11 :=
sorry

end probability_blue_or_purple_is_4_over_11_l42_42398


namespace number_of_valid_polynomials_l42_42478

theorem number_of_valid_polynomials :
  let f := λ x: ℝ, 2 * x^4 - 6 * x^2 + 1
  let g := λ x: ℝ, 4 - 5 * x^2
  ∃ n: ℕ, (∀ p : ℝ → ℤ, (∀ x: ℝ, min (f x) (g x) ≤ p x) ∧ (∀ x: ℝ, p x ≤ max (f x) (g x)) → n = 4) :=
by 
  let f := λ x: ℝ, 2 * x^4 - 6 * x^2 + 1
  let g := λ x: ℝ, 4 - 5 * x^2
  use 4
  sorry

end number_of_valid_polynomials_l42_42478


namespace valid_fraction_l42_42811

theorem valid_fraction (x: ℝ) : x^2 + 1 ≠ 0 :=
by
  sorry

end valid_fraction_l42_42811


namespace find_inverse_sum_l42_42622

def f (x : ℝ) : ℝ := x * |x|^2

theorem find_inverse_sum :
  (∃ x : ℝ, f x = 8) ∧ (∃ y : ℝ, f y = -64) → 
  (∃ a b : ℝ, f a = 8 ∧ f b = -64 ∧ a + b = 6) :=
sorry

end find_inverse_sum_l42_42622


namespace original_function_l42_42987

theorem original_function :
  ∃ f : ℝ → ℝ,
  (∀ x : ℝ, sin (x - π / 4) = f (2 * (x - π / 3))) →
  (∀ x : ℝ, f x = sin (x / 2 + π / 12)) :=
by
  sorry

end original_function_l42_42987


namespace sales_function_maximize_profit_l42_42844

def y (x : ℝ) : ℝ := 300 - 10 * (x - 44)

theorem sales_function (x : ℝ) (h₁ : x ≥ 44) (h₂ : x ≤ 52) :
  y x = -10 * x + 740 :=
by 
  have : y x = 300 - 10 * (x - 44) := rfl
  rw [this, sub_mul, add_sub_cancel]
  norm_num

def profit (x : ℝ) : ℝ := (x - 40) * (300 - 10 * (x - 44))

theorem maximize_profit (x : ℝ) (h₁ : x = 52) :
  profit x = 2640 :=
by
  unfold profit
  rw [h₁, sub_self, zero_mul, add_zero, mul_comm, ← mul_assoc, mul_left_comm]
  norm_num

end sales_function_maximize_profit_l42_42844


namespace final_speed_correct_l42_42004

def initial_speed_kmph : ℝ := 108
def acceleration : ℝ := 3
def time : ℝ := 8
def conversion_factor_kmph_to_mps : ℝ := 1000 / 3600

-- Convert initial speed to meters per second
def initial_speed_mps : ℝ := initial_speed_kmph * conversion_factor_kmph_to_mps

-- Final speed calculation
def final_speed_mps : ℝ := initial_speed_mps + (acceleration * time)

-- Conversion factor from m/s to mph
def conversion_factor_mps_to_mph : ℝ := 2.23694

-- Convert final speed from meters per second to miles per hour
def final_speed_mph : ℝ := final_speed_mps * conversion_factor_mps_to_mph

-- Theorem stating our final result
theorem final_speed_correct :
  final_speed_mps = 54 :=
by {
  -- proof goes here
  sorry
}

end final_speed_correct_l42_42004


namespace circles_externally_tangent_l42_42540

noncomputable def distance (C1 C2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((C1.1 - C2.1)^2 + (C1.2 - C2.2)^2)

theorem circles_externally_tangent :
  (∃ C1 r1 C2 r2,   -- Exist centers C1, C2 and radii r1, r2
    (C1 = (3,4)) ∧  -- Center of the second circle
    (r1 = 4) ∧      -- Radius of the second circle
    (C2 = (0,0)) ∧  -- Center of the first circle
    (r2 = 1) ∧      -- Radius of the first circle
    distance C1 C2 = r1 + r2) :=     -- Distance between centers equals the sum of radii
by
  let C1 := (3, 4)
  let r1 := 4
  let C2 := (0, 0)
  let r2 := 1
  have h1 : distance C1 C2 = 5 := by {
    unfold distance,
    rw [←real.sqrt_eq_rpow, show ((3 - 0)^2 + (4 - 0)^2 : ℝ) = 25, by norm_num],
    norm_num,
  }
  have h2 : r1 + r2 = 5 := by norm_num
  exact ⟨C1, r1, C2, r2, rfl, rfl, rfl, rfl, h1.trans h2.symm⟩

end circles_externally_tangent_l42_42540


namespace product_ab_l42_42513

noncomputable def median_of_four_numbers (a b : ℕ) := 3
noncomputable def mean_of_four_numbers (a b : ℕ) := 4

theorem product_ab (a b : ℕ)
  (h1 : 1 + 2 + a + b = 4 * 4)
  (h2 : median_of_four_numbers a b = 3)
  (h3 : mean_of_four_numbers a b = 4) : (a * b = 36) :=
by sorry

end product_ab_l42_42513


namespace line_equation_l42_42854

noncomputable def line_through_points (A B : ℝ × ℝ) : ℝ → ℝ → Prop :=
λ x y, (B.1 - A.1) * (y - A.2) = (B.2 - A.2) * (x - A.1)

theorem line_equation (x y : ℝ) :
  line_through_points (0, 1) (2, 0) x y ↔ x + 2*y - 2 = 0 :=
sorry

end line_equation_l42_42854


namespace transformed_function_is_correct_l42_42975

noncomputable def f : ℝ → ℝ := λ x, sin (x / 2 + π / 12)

theorem transformed_function_is_correct :
  ∀ x : ℝ, sin (2 * (x - π / 3)) = sin (x - π / 4) :=
by
  intros x
  symmetry
  apply eq.symm
  calc
    sin (2 * (x - π / 3)) = sin (2x - 2 * π / 3) : by rw mul_sub
    _ = sin (x - π / 4) : sorry

end transformed_function_is_correct_l42_42975


namespace proof_line_eq_l42_42790

variable (a T : ℝ) (line : ℝ × ℝ → Prop)

def line_eq (point : ℝ × ℝ) : Prop := 
  point.2 = (-2 * T / a^2) * point.1 + (2 * T / a)

def correct_line_eq (point : ℝ × ℝ) : Prop :=
  -2 * T * point.1 + a^2 * point.2 + 2 * a * T = 0

theorem proof_line_eq :
  ∀ point : ℝ × ℝ, line_eq a T point ↔ correct_line_eq a T point :=
by
  sorry

end proof_line_eq_l42_42790


namespace find_equation_of_C_find_equation_of_AB_l42_42213

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42213


namespace maximum_diagonals_l42_42501

noncomputable def max_intersecting_diagonals (n : ℕ) (h : n ≥ 4) : ℕ :=
  n / 2

theorem maximum_diagonals (n : ℕ) (h : n ≥ 4) :
  ∃ (D : set (ℕ × ℕ)), (∀ (d1 d2 ∈ D), d1 ≠ d2 → d1.snd ≠ d2.snd → true) ∧ D.card = max_intersecting_diagonals n h :=
sorry

end maximum_diagonals_l42_42501


namespace joggers_problem_l42_42810

-- Define the variables for the number of joggers bought by Tyson, Alexander, and Christopher.
variables (T A C : ℕ)

-- Translate conditions to Lean definitions.
def condition1 : Prop := C = 20 * T
def condition2 : Prop := C = 80
def condition3 : Prop := C = A + 54

-- The statement we want to prove:
theorem joggers_problem (h1 : condition1 T A C) (h2 : condition2 T A C) (h3 : condition3 T A C) :
  A = T + 22 :=
by
  sorry

end joggers_problem_l42_42810


namespace simplify_combination_expression_l42_42499

-- Definitions of key components
variables (n k m : ℕ)

-- Given conditions
def conditions := (1 ≤ k) ∧ (k < m) ∧ (m ≤ n)

-- The equivalence we aim to prove
theorem simplify_combination_expression (H : conditions n k m) :
  ∑ i in finset.range (k+1), nat.choose k i * nat.choose n (m - i) = nat.choose (n + k) m := by sorry

end simplify_combination_expression_l42_42499


namespace find_equation_of_C_find_equation_of_AB_l42_42210

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42210


namespace min_value_of_f_l42_42033

-- Define the function
def f (x : Real) : Real := Real.exp x - x

-- Define the statement to verify the minimum value
theorem min_value_of_f : ∃ x : Real, (∀ y : Real, f x ≤ f y) ∧ f 0 = 1 := 
by
  sorry

end min_value_of_f_l42_42033


namespace solve_for_x2_plus_9y2_l42_42952

variable (x y : ℝ)

def condition1 : Prop := x + 3 * y = 3
def condition2 : Prop := x * y = -6

theorem solve_for_x2_plus_9y2 (h1 : condition1 x y) (h2 : condition2 x y) :
  x^2 + 9 * y^2 = 45 :=
by
  sorry

end solve_for_x2_plus_9y2_l42_42952


namespace number_of_observations_corrected_l42_42682

variables (n : ℕ) (mean1 mean2 wrong observation correct : ℝ)
hypotheses (h1 : mean1 = 30) (h2 : wrong = 23) (h3 : correct = 48) (h4 : mean2 = 30.5)

-- The original problem statement in Lean
theorem number_of_observations_corrected :
  let original_sum := n * mean1,
      corrected_sum := original_sum + (correct - wrong) in
  corrected_sum / n = mean2 → n = 50 :=
by
  sorry

end number_of_observations_corrected_l42_42682


namespace color_infinite_points_divisible_by_k_l42_42466

theorem color_infinite_points_divisible_by_k :
  (∀ n : ℤ, painted n = red ∨ painted n = blue) →
  ∃ c : color, ∀ k : ℕ, ∃∞ n : ℤ, painted n = c ∧ n % k = 0 :=
begin
  intros h,
  sorry
end

end color_infinite_points_divisible_by_k_l42_42466


namespace proportion_of_bike_riders_is_correct_l42_42696

-- Define the given conditions as constants
def total_students : ℕ := 92
def bus_riders : ℕ := 20
def walkers : ℕ := 27

-- Define the remaining students after bus riders and after walkers
def remaining_after_bus_riders : ℕ := total_students - bus_riders
def bike_riders : ℕ := remaining_after_bus_riders - walkers

-- Define the expected proportion
def expected_proportion : ℚ := 45 / 72

-- State the theorem to be proved
theorem proportion_of_bike_riders_is_correct :
  (↑bike_riders / ↑remaining_after_bus_riders : ℚ) = expected_proportion := 
by
  sorry

end proportion_of_bike_riders_is_correct_l42_42696


namespace parabola_equation_line_AB_equation_l42_42217

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42217


namespace find_original_function_l42_42963

theorem find_original_function
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₁ : ∀ x, g (x + (π / 3)) = sin (x - (π / 4)))
  (h₂ : ∀ x, f x = g (2 * x)) :
  f x = sin (x / 2 + π / 12) :=
sorry

end find_original_function_l42_42963


namespace remainder_T_2015_mod_8_l42_42042

-- Definitions based on the problem's conditions
def T (n : ℕ) : ℕ := sorry -- Placeholder definition of T based on sequences constraint

theorem remainder_T_2015_mod_8 : T(2015) % 8 = 3 := sorry

end remainder_T_2015_mod_8_l42_42042


namespace factors_180_count_l42_42112

theorem factors_180_count : 
  ∃ (n : ℕ), 180 = 2^2 * 3^2 * 5^1 ∧ n = 18 ∧ 
  ∀ p a b c, 
  180 = p^a * p^b * p^c →
  (a+1) * (b+1) * (c+1) = 18 :=
by {
  sorry
}

end factors_180_count_l42_42112


namespace revenue_increase_20_percent_l42_42739

variable (P Q : ℝ)

def original_revenue (P Q : ℝ) : ℝ := P * Q
def new_price (P : ℝ) : ℝ := P * 1.5
def new_quantity (Q : ℝ) : ℝ := Q * 0.8
def new_revenue (P Q : ℝ) : ℝ := (new_price P) * (new_quantity Q)

theorem revenue_increase_20_percent (P Q : ℝ) : 
  (new_revenue P Q) = 1.2 * (original_revenue P Q) := by
  sorry

end revenue_increase_20_percent_l42_42739


namespace lifeguard_swim_time_l42_42361

theorem lifeguard_swim_time
  (front_crawl_speed : ℕ) (breaststroke_speed : ℕ) (total_distance : ℕ) (total_time : ℕ) 
  (h1 : front_crawl_speed = 45) (h2 : breaststroke_speed = 35) (h3 : total_distance = 500)
  (h4 : total_time = 12) :
  ∃ t : ℕ, 45 * t + 35 * (12 - t) = 500 ∧ t = 8 :=
by
  use 8
  split
  any_goals {
    sorry
  }

end lifeguard_swim_time_l42_42361


namespace maximize_profit_l42_42842
-- Importing the entire necessary library

-- Definitions and conditions
def cost_price : ℕ := 40
def minimum_selling_price : ℕ := 44
def maximum_profit_margin : ℕ := 30
def sales_at_minimum_price : ℕ := 300
def price_increase_effect : ℕ := 10
def max_profit_price := 52
def max_profit := 2640

-- Function relationship between y and x
def sales_volume (x : ℕ) : ℕ := 300 - 10 * (x - 44)

-- Range of x
def valid_price (x : ℕ) : Prop := 44 ≤ x ∧ x ≤ 52

-- Statement of the problem
theorem maximize_profit (x : ℕ) (hx : valid_price x) : 
  sales_volume x = 300 - 10 * (x - 44) ∧
  44 ≤ x ∧ x ≤ 52 ∧
  x = 52 → 
  (x - cost_price) * (sales_volume x) = max_profit :=
sorry

end maximize_profit_l42_42842


namespace max_planes_from_10_points_l42_42716

theorem max_planes_from_10_points (h : ∀ (p1 p2 p3 : Point), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → 
                                   ¬ collinear {p1, p2, p3}) : 
  (number_of_planes 10 = 120) :=
  sorry

-- Definitions to make the theorem statement self-contained
structure Point := (x y z : ℝ)

def collinear (s : set Point) : Prop := 
  ∃ (a b c : Point), s = {a, b, c} ∧ (b.x - a.x) * (c.y - a.y) = (c.x - a.x) * (b.y - a.y)

noncomputable def number_of_planes (n : ℕ) : ℕ :=
  if h : 3 ≤ n then nat.choose n 3 else 0

end max_planes_from_10_points_l42_42716


namespace minimum_value_of_f_l42_42857

open Real

def f (x : ℝ) : ℝ := x^2 + 1/(x^2) + 1/(x^2 + 1/(x^2))

theorem minimum_value_of_f (x > 0) : ∃ y, y = 2.5 ∧ ∀ z > 0, f z ≥ y := by
  sorry

end minimum_value_of_f_l42_42857


namespace slope_of_line_l42_42128

noncomputable def slope (t : ℝ) : ℝ :=
(let x := -1 + t * Real.sin (40 * Real.pi / 180) in
 let y := 3 + t * Real.cos (40 * Real.pi / 180) in
  (y - 3) / (x + 1))

theorem slope_of_line :
  ∀ t : ℝ, slope t = Real.tan (50 * Real.pi / 180) :=
by
  intros t
  unfold slope
  sorry

end slope_of_line_l42_42128


namespace problem_statement_l42_42520

def positive_integer (n : ℕ) : Prop :=
  0 < n

def sequence_a (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, positive_integer n → a n > 0 ∧ 4 * (n + 1) * (a n)^2 - n * (a (n + 1))^2 = 0

def sequence_b (a : ℕ → ℝ) (b : ℕ → ℝ) (t : ℝ) : Prop :=
  ∀ n : ℕ, positive_integer n → b n = (a n)^2 / t^n

theorem problem_statement 
  {n : ℕ} (hn : positive_integer n) 
  (a : ℕ → ℝ) (ha : sequence_a a) 
  (b : ℕ → ℝ) (hb : sequence_b a b) 
  (geometric_seq : ∀ n, n ≥ 1 → a (n+1) / √(n+1) = 2 * a n / √n) 
  (t_12_or_4 : t = 12 ∨ t = 4) 
  (sum_sn : ∀ n, ∃ m, 8 * (a 1)^2 * S n - (a 1)^4 * n^2 = 16 * b m)
  : (∃ k : ℕ, positive_integer k ∧ a 1 = 2 * √(m / n) ∧ (√(m / n) = 1/2 * ↑k)) := 
sorry

end problem_statement_l42_42520


namespace unique_solution_l42_42052

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem unique_solution (x y : ℕ) :
  is_prime x →
  is_odd y →
  x^2 + y = 2007 →
  (x = 2 ∧ y = 2003) :=
by
  sorry

end unique_solution_l42_42052


namespace parabola_equation_line_AB_equation_l42_42187

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42187


namespace area_of_PQR_is_one_sixth_l42_42508

-- Definitions of the areas and triangles involved
def area_ABC : ℝ := 2

-- Medians AK, BL, CN and the points P, Q, R with given ratios
def ratio_AP_PK : ℝ := 1
def ratio_BQ_QL : ℝ := 1 / 2
def ratio_CR_RN : ℝ := 5 / 4

-- Function to calculate the area of triangle PQR
noncomputable def area_PQR : ℝ :=
  let area_AOB := area_ABC / 3
  let area_AOC := area_ABC / 3
  let area_BOC := area_ABC / 3
  let OP := (ratio_AP_PK / (1 + ratio_AP_PK)) * 6
  let OQ := (ratio_BQ_QL / (1 + ratio_BQ_QL)) * 2
  let OR := (ratio_CR_RN / (1 + ratio_CR_RN)) * 1
  let S_POQ := OP / 4 * OQ / 4 * area_AOB
  let S_POR := OP / 4 * OR / 6 * area_AOC
  let S_ROQ := OR / 6 * OQ / 4 * area_BOC
  (S_POQ + S_POR + S_ROQ) * (2 / 3)

-- Proof statement
theorem area_of_PQR_is_one_sixth : area_PQR = 1 / 6 := 
by 
  sorry

end area_of_PQR_is_one_sixth_l42_42508


namespace semicircle_containing_all_numbers_exists_l42_42766

variable (n : ℕ)

def has_semicircle_containing_all_numbers (n : ℕ) :=
  ∃ semicircle : ℤ → Prop, (∀ i ∈ semicircle, 1 ≤ i ∧ i ≤ n) ∧
                            (∀ i, 1 ≤ i ∧ i ≤ n ↔ i ∈ semicircle)

theorem semicircle_containing_all_numbers_exists (n : ℕ) (h1 : 2 * n > 0) :
  has_semicircle_containing_all_numbers n :=
by sorry

end semicircle_containing_all_numbers_exists_l42_42766


namespace sum_of_surface_points_l42_42759

theorem sum_of_surface_points
  (n : ℕ) (h_n : n = 2012) 
  (total_sum : ℕ) (h_total : total_sum = n * 21)
  (matching_points_sum : ℕ) (h_matching : matching_points_sum = (n - 1) * 7)
  (x : ℕ) (h_x_range : 1 ≤ x ∧ x ≤ 6) :
  (total_sum - matching_points_sum + 2 * x = 28177 ∨
   total_sum - matching_points_sum + 2 * x = 28179 ∨
   total_sum - matching_points_sum + 2 * x = 28181 ∨
   total_sum - matching_points_sum + 2 * x = 28183 ∨
   total_sum - matching_points_sum + 2 * x = 28185 ∨
   total_sum - matching_points_sum + 2 * x = 28187) :=
by sorry

end sum_of_surface_points_l42_42759


namespace vector_coefficients_exists_l42_42041

open Real EuclideanSpace

theorem vector_coefficients_exists {n : ℕ} (hn : n = 1989)
  (v : Fin n → EuclideanSpace ℝ 2)
  (h : ∀ k, ∥v k∥ ≤ 1):
  ∃ (ε : Fin n → ℤ), (∀ k, ε k = -1 ∨ ε k = 1) ∧ ∥∑ k, ε k • v k∥ ≤ sqrt 3 := 
sorry

end vector_coefficients_exists_l42_42041


namespace parabola_equation_line_AB_equation_l42_42225

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42225


namespace find_a_b_p_l42_42693

def V : set ℂ :=
 {complex.I * real.sqrt 2,
  -complex.I * real.sqrt 2,
  real.sqrt 2,
  -real.sqrt 2,
  (1 + complex.I) / (2 * real.sqrt 2),
  (-1 + complex.I) / (2 * real.sqrt 2),
  (1 - complex.I) / (2 * real.sqrt 2),
  (-1 - complex.I) / (2 * real.sqrt 2)}

def z (j : fin 16) : ℂ :=
choose_any_from V -- This is intentionally left abstract as the problem states random choice

def P : ℂ :=
finset.prod (finset.univ : finset (fin 16)) (λ j, z j)

theorem find_a_b_p (hP : P = 1) :
  ∃ a b p : ℕ, nat.prime p ∧ a = 1 ∧ b = 2 ∧ p = 2 ∧ a + b + p = 5 := by
  sorry

end find_a_b_p_l42_42693


namespace total_lines_to_write_l42_42379

theorem total_lines_to_write (lines_per_page pages_needed : ℕ) (h1 : lines_per_page = 30) (h2 : pages_needed = 5) : lines_per_page * pages_needed = 150 :=
by {
  sorry
}

end total_lines_to_write_l42_42379


namespace find_equation_of_C_find_equation_of_AB_l42_42216

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42216


namespace original_function_l42_42989

theorem original_function :
  ∃ f : ℝ → ℝ,
  (∀ x : ℝ, sin (x - π / 4) = f (2 * (x - π / 3))) →
  (∀ x : ℝ, f x = sin (x / 2 + π / 12)) :=
by
  sorry

end original_function_l42_42989


namespace greatest_individual_score_l42_42770

theorem greatest_individual_score (points : ℕ) (players : Fin 12 → ℕ) 
  (h_team_size : ∑ i, 1 = 12)
  (h_team_points : ∑ i, players i = 100)
  (h_min_score : ∀ i, players i ≥ 7) : 
  ∃ i, players i = 23 :=
by
  sorry

end greatest_individual_score_l42_42770


namespace john_cakes_bought_l42_42158

-- Conditions
def cake_price : ℕ := 12
def john_paid : ℕ := 18

-- Definition of the total cost
def total_cost : ℕ := 2 * john_paid

-- Calculate number of cakes
def num_cakes (total_cost cake_price : ℕ) : ℕ := total_cost / cake_price

-- Theorem to prove that the number of cakes John Smith bought is 3
theorem john_cakes_bought : num_cakes total_cost cake_price = 3 := by
  sorry

end john_cakes_bought_l42_42158


namespace jake_sausages_cost_l42_42155

theorem jake_sausages_cost :
  let package_weight := 2
  let num_packages := 3
  let cost_per_pound := 4
  let total_weight := package_weight * num_packages
  let total_cost := total_weight * cost_per_pound
  total_cost = 24 := by
  sorry

end jake_sausages_cost_l42_42155


namespace hyperbola_focal_length_l42_42083

theorem hyperbola_focal_length {b : ℝ} (hb : b > 0) 
  (hyp : ∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) 
  (ecc : ∀ c : ℝ, c / 2 = (√3 / 3) * b) :
  2 * 4 = 8 :=
by
  sorry

end hyperbola_focal_length_l42_42083


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42280

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42280


namespace f_2015_log_l42_42611

def B : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 2}

noncomputable def f (x : ℚ) (h : x ∈ B) : ℝ := sorry

theorem f_2015_log (h : 2015 ∈ B) (hf : ∀ x ∈ B, f x (by assumption) + f (2 - x⁻¹) (by assumption) = Real.log (|x|)) :
  f 2015 h = Real.log (2015 / 13) :=
sorry

end f_2015_log_l42_42611


namespace one_cubic_foot_is_1728_cubic_inches_l42_42101

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end one_cubic_foot_is_1728_cubic_inches_l42_42101


namespace abc_inequality_l42_42623

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (a / (1 + a * b))^2 + (b / (1 + b * c))^2 + (c / (1 + c * a))^2 ≥ 3 / 4 :=
by
  sorry

end abc_inequality_l42_42623


namespace angle_of_inclination_of_line_l42_42086

theorem angle_of_inclination_of_line {α : ℝ} (hα : 0 ≤ α ∧ α < 180) (h : tan α = sqrt 3) : α = 60 :=
by
  sorry

end angle_of_inclination_of_line_l42_42086


namespace average_time_of_two_trains_is_4_l42_42366

theorem average_time_of_two_trains_is_4:
  ∀ (d1 s1 d2 s2: ℝ), d1 = 200 → s1 = 50 → d2 = 240 → s2 = 80 → 
  real.to_nnreal ( (d1 / s1 + d2 / s2) / 2).to_nat = 4 :=
begin
  intros d1 s1 d2 s2 hd1 hs1 hd2 hs2,
  rw [hd1, hs1, hd2, hs2],
  norm_num,
  sorry
end

end average_time_of_two_trains_is_4_l42_42366


namespace exists_perpendicular_line_in_plane_l42_42118

theorem exists_perpendicular_line_in_plane (a : Plane) (b : Line) (b_in_a : b ∈ a) : 
  ∃ c : Line, c ∈ a ∧ c ⊥ b := 
sorry

end exists_perpendicular_line_in_plane_l42_42118


namespace dot_product_self_l42_42558

variable {ℝ} [InnerProductSpace ℝ]

theorem dot_product_self (w : ℝ^n) (norm_w : ‖w‖ = 5) : w ⬝ w = 25 := by
  sorry

end dot_product_self_l42_42558


namespace centroid_value_l42_42364

-- Define the points according to given vertices 
def P : ℝ × ℝ := (4, 9)
def Q : ℝ × ℝ := (2, -3)
def R : ℝ × ℝ := (7, 2)

-- Define the centroid S of triangle PQR
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Calculate 8x + y using the centroid coordinates of P, Q, R
def value_8x_plus_y (P Q R : ℝ × ℝ) : ℝ := 
  let (x, y) := centroid P Q R
  8 * x + y

-- The theorem to prove
theorem centroid_value : value_8x_plus_y P Q R = 37 + 1/3 := by
  sorry

end centroid_value_l42_42364


namespace determine_q_l42_42014

theorem determine_q (p q : ℝ) 
  (h : ∀ x : ℝ, (x + 3) * (x + p) = x^2 + q * x + 12) : 
  q = 7 :=
by
  sorry

end determine_q_l42_42014


namespace part_one_part_two_l42_42229

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42229


namespace sum_of_angles_satisfying_equation_l42_42860

-- Define the problem and conditions in Lean
def angle_satisfies_equation (x : ℝ) : Prop :=
(sin x)^3 + (cos x)^3 = 1 / (sin x) + 1 / (cos x)

-- Define the theorem to prove the required statement
theorem sum_of_angles_satisfying_equation :
  (∀ x, 0 ≤ x ∧ x ≤ 360 → angle_satisfies_equation x → (x = 45 ∨ x = 315) ) →
  45 + 315 = 360 :=
by
  sorry

end sum_of_angles_satisfying_equation_l42_42860


namespace find_parabola_equation_find_line_AB_equation_l42_42243

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42243


namespace etienne_diana_money_difference_l42_42326

theorem etienne_diana_money_difference :
  let initial_exchange_rate := 1.25
  let diana_dollars := 600
  let etienne_euros := 350
  let euro_appreciation := 0.08
  let new_exchange_rate := initial_exchange_rate * (1 + euro_appreciation)
  let etienne_dollars_after_appreciation := etienne_euros * new_exchange_rate
  in ((diana_dollars - etienne_dollars_after_appreciation) / diana_dollars * 100 = 21.25) :=
by
  sorry

end etienne_diana_money_difference_l42_42326


namespace solution_set_f_x_lt_0_l42_42567

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ ⦃x y⦄, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem solution_set_f_x_lt_0 (h_even : is_even f)
    (h_increasing : is_increasing_on_nonneg f)
    (h_f2 : f 2 = 0) : 
  { x : ℝ | f x < 0 } = set.Ioo (-2) 2 :=
by
  sorry

end solution_set_f_x_lt_0_l42_42567


namespace range_of_a_l42_42569

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x → (a * x + 1) * (real.exp x - a * real.exp x) ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l42_42569


namespace find_original_function_l42_42966

theorem find_original_function
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₁ : ∀ x, g (x + (π / 3)) = sin (x - (π / 4)))
  (h₂ : ∀ x, f x = g (2 * x)) :
  f x = sin (x / 2 + π / 12) :=
sorry

end find_original_function_l42_42966


namespace find_function_l42_42995

open Real

theorem find_function (f : ℝ → ℝ) :
  let g := λ x, f (x / 2);
  let h := λ x, g (x - π / 3);
  (∀ x, h x = sin (x - π / 4)) →
  ∀ x, f x = sin (x / 2 + π / 12) :=
by
  intros g h h_eq x
  sorry

end find_function_l42_42995


namespace find_a_extreme_value_at_2_l42_42912

noncomputable def f (x : ℝ) (a : ℝ) := (2 / 3) * x^3 + a * x^2

theorem find_a_extreme_value_at_2 (a : ℝ) :
  (∀ x : ℝ, x ≠ 2 -> 0 = 2 * x^2 + 2 * a * x) ->
  (2 * 2^2 + 2 * a * 2 = 0) ->
  a = -2 :=
by {
  sorry
}

end find_a_extreme_value_at_2_l42_42912


namespace unique_four_digit_minuend_l42_42636

theorem unique_four_digit_minuend :
  ∃ (a b c d e f g h i j : ℕ),
     a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
     d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
     e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
     f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
     g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
     h ≠ i ∧ h ≠ j ∧
     i ≠ j ∧
     a ∈ {0, 2, 4, 5, 6, 7, 8, 9} ∧ d = 1 ∧
     1000 * a + 100 * d + 10 * g + h - (100 * e + 10 * f + g) * (1000 * i + 100 * j + 10 * k + l) = 2016
       :=
sorry

end unique_four_digit_minuend_l42_42636


namespace transformed_function_is_correct_l42_42974

noncomputable def f : ℝ → ℝ := λ x, sin (x / 2 + π / 12)

theorem transformed_function_is_correct :
  ∀ x : ℝ, sin (2 * (x - π / 3)) = sin (x - π / 4) :=
by
  intros x
  symmetry
  apply eq.symm
  calc
    sin (2 * (x - π / 3)) = sin (2x - 2 * π / 3) : by rw mul_sub
    _ = sin (x - π / 4) : sorry

end transformed_function_is_correct_l42_42974


namespace find_N_l42_42352

variable (a b c N : ℕ)

theorem find_N (h1 : a + b + c = 90) (h2 : a - 7 = N) (h3 : b + 7 = N) (h4 : 5 * c = N) : N = 41 := 
by
  sorry

end find_N_l42_42352


namespace range_of_a_l42_42531

noncomputable def delta (a : ℝ) : ℝ := 4 - 8 * a

theorem range_of_a (a : ℝ) (h : ∀ y ∈ ℝ, ∃ x : ℝ, y = log (a * x^2 - 2 * x + 2)) :
  0 < a ∧ a ≤ 1 / 2 :=
sorry

end range_of_a_l42_42531


namespace ball_bounce_height_l42_42399

theorem ball_bounce_height :
  ∃ k : ℕ, 800 * (1 / 2 : ℝ)^k < 2 ∧ k ≥ 9 :=
by
  sorry

end ball_bounce_height_l42_42399


namespace sum_of_surface_points_l42_42760

theorem sum_of_surface_points
  (n : ℕ) (h_n : n = 2012) 
  (total_sum : ℕ) (h_total : total_sum = n * 21)
  (matching_points_sum : ℕ) (h_matching : matching_points_sum = (n - 1) * 7)
  (x : ℕ) (h_x_range : 1 ≤ x ∧ x ≤ 6) :
  (total_sum - matching_points_sum + 2 * x = 28177 ∨
   total_sum - matching_points_sum + 2 * x = 28179 ∨
   total_sum - matching_points_sum + 2 * x = 28181 ∨
   total_sum - matching_points_sum + 2 * x = 28183 ∨
   total_sum - matching_points_sum + 2 * x = 28185 ∨
   total_sum - matching_points_sum + 2 * x = 28187) :=
by sorry

end sum_of_surface_points_l42_42760


namespace part1_eq_C_part2_line_AB_l42_42291

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42291


namespace harry_friday_speed_l42_42932

def speed_monday := 10 -- Speed on Monday in meters per hour
def speed_tuesday := speed_monday - (0.3 * speed_monday) -- Speed on Tuesday in meters per hour
def speed_wednesday := speed_monday + (0.5 * speed_monday) -- Speed on Wednesday in meters per hour
def speed_thursday := speed_wednesday -- Speed on Thursday in meters per hour
def speed_friday := speed_thursday + (0.6 * speed_thursday) -- Speed on Friday in meters per hour

theorem harry_friday_speed : speed_friday = 24 := 
by 
  have h1: speed_friday = speed_thursday + (0.6 * speed_thursday) := rfl
  have h2: speed_thursday = speed_wednesday := rfl
  have h3: speed_wednesday = speed_monday + (0.5 * speed_monday) := rfl
  have h4: speed_tuesday = speed_monday - (0.3 * speed_monday) := rfl
  have h5: speed_monday = 10 := rfl
  sorry

end harry_friday_speed_l42_42932


namespace growth_calculation_correct_l42_42583

noncomputable def turnip_carrot_growth : Prop :=
  let melanie_week1_turnips := 139
  let melanie_week1_carrots := 45
  let melanie_week2_turnips := melanie_week1_turnips * 2
  let melanie_week2_carrots := (melanie_week1_carrots * 1.25).toNat
  let melanie_total_turnips := melanie_week1_turnips + melanie_week2_turnips
  let melanie_total_carrots := melanie_week1_carrots + melanie_week2_carrots
  let benny_week1_turnips := 113
  let benny_week1_carrots := 75
  let benny_week2_turnips := (benny_week1_turnips * 1.30).toNat
  let benny_week2_carrots := (benny_week1_carrots * 1.30).toNat
  let benny_total_turnips := benny_week1_turnips + benny_week2_turnips
  let benny_total_carrots := benny_week1_carrots + benny_week2_carrots
  let carol_week1_turnips := 195
  let carol_week1_carrots := 60
  let carol_week2_turnips := (carol_week1_turnips * 1.20).toNat
  let carol_week2_carrots := (carol_week1_carrots * 1.40).toNat
  let carol_total_turnips := carol_week1_turnips + carol_week2_turnips
  let carol_total_carrots := carol_week1_carrots + carol_week2_carrots
  melanie_total_turnips = 417 ∧ melanie_total_carrots = 101 ∧
  benny_total_turnips = 260 ∧ benny_total_carrots = 173 ∧
  carol_total_turnips = 429 ∧ carol_total_carrots = 144 ∧
  carol_total_turnips > melanie_total_turnips ∧ carol_total_turnips > benny_total_turnips ∧
  benny_total_carrots > melanie_total_carrots ∧ benny_total_carrots > carol_total_carrots

-- Statement of the problem
theorem growth_calculation_correct : turnip_carrot_growth :=
by
  sorry

end growth_calculation_correct_l42_42583


namespace kevin_total_hops_l42_42162

/-- Define the hop function for Kevin -/
def hop (remaining_distance : ℚ) : ℚ :=
  remaining_distance / 4

/-- Summing the series for five hops -/
def total_hops (start_distance : ℚ) (hops : ℕ) : ℚ :=
  let h0 := hop start_distance
  let h1 := hop (start_distance - h0)
  let h2 := hop (start_distance - h0 - h1)
  let h3 := hop (start_distance - h0 - h1 - h2)
  let h4 := hop (start_distance - h0 - h1 - h2 - h3)
  h0 + h1 + h2 + h3 + h4

/-- Final proof statement: after five hops from starting distance of 2, total distance hopped should be 1031769/2359296 -/
theorem kevin_total_hops :
  total_hops 2 5 = 1031769 / 2359296 :=
sorry

end kevin_total_hops_l42_42162


namespace part_one_part_two_l42_42232

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42232


namespace find_lambda_l42_42068

open Real

def unit_vector (v : ℝ × ℝ) : Prop := (v.1 ^ 2 + v.2 ^ 2 = 1)

noncomputable def compute_lambda (e1 e2 : ℝ × ℝ) (lambda : ℝ) : ℝ :=
  let a := (e1.1 + lambda * e2.1, e1.2 + lambda * e2.2)
  sqrt (a.1 ^ 2 + a.2 ^ 2)

theorem find_lambda (e1 e2 : ℝ × ℝ) (h₁ : unit_vector e1) (h₂ : unit_vector e2)
  (h₃ : acos ((e1.1 * e2.1 + e1.2 * e2.2) / (sqrt (e1.1 ^ 2 + e1.2 ^ 2) * sqrt (e2.1 ^ 2 + e2.2 ^ 2))) = π / 3)
  (h₄ : compute_lambda e1 e2 (-1 / 2) = sqrt 3 / 2) :
  ∃ λ : ℝ, λ = -1 / 2 := by
  sorry

end find_lambda_l42_42068


namespace max_value_of_expression_l42_42072

theorem max_value_of_expression (α : ℝ) (n : ℕ) 
  (hα : α > 0) (hn : n > 0) :
  (∃ (f : ℕ → ℕ), ∑ i in finset.range(n), f i = n ∧ 
    n.bounded (λ i, f i) ∧ 
    ∑ i in finset.range(n), α ^ (f i)) = max (n * α) (α ^ n) :=
by
  sorry

end max_value_of_expression_l42_42072


namespace emma_time_to_complete_remaining_task_l42_42742

-- Definitions for the conditions
def emma_rate := 1 / 6
def remaining_task := 5 / 12

-- Main statement to prove that Emma will take 2.5 hours to complete the remaining task
theorem emma_time_to_complete_remaining_task :
  (remaining_task / emma_rate) = 2.5 :=
by
  sorry

end emma_time_to_complete_remaining_task_l42_42742


namespace parallel_vectors_k_value_l42_42094

-- Defining vectors a and b.
def a : ℝ × ℝ := (2, -1)
def b (k : ℝ) : ℝ × ℝ := (k, 5 / 2)

-- Define the condition that vectors a and b are parallel.
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v1 = (c * v2.1, c * v2.2)

-- Main theorem to prove.
theorem parallel_vectors_k_value (k : ℝ) 
  (ha : a = (2, -1)) 
  (hb : b k = (k, 5 / 2))
  (h_parallel : are_parallel a (b k)) : 
  k = -5 :=
sorry

end parallel_vectors_k_value_l42_42094


namespace find_first_term_of_sequence_l42_42691

theorem find_first_term_of_sequence (a : ℕ → ℝ)
  (h_rec : ∀ n, a (n + 1) = 1 / (1 - a n))
  (h_a8 : a 8 = 2) :
  a 1 = 1 / 2 :=
sorry

end find_first_term_of_sequence_l42_42691


namespace extended_ohara_quadruple_l42_42359

theorem extended_ohara_quadruple (a b c x : ℕ) (h1 : a = 16) (h2 : b = 9) (h3 : c = 4) (h4 : ℝ) 
  (h5 : sqrt (a) + sqrt (b) + sqrt (c) = x) : x = 9 :=
by {
  have h_sqrt_a : sqrt a = 4, by rw [h1, sqrt_nat.eq], -- Since sqrt(16) = 4
  have h_sqrt_b : sqrt b = 3, by rw [h2, sqrt_nat.eq], -- Since sqrt(9) = 3
  have h_sqrt_c : sqrt c = 2, by rw [h3, sqrt_nat.eq], -- Since sqrt(4) = 2
  rw [h_sqrt_a, h_sqrt_b, h_sqrt_c] at h5,
  sorry
}

end extended_ohara_quadruple_l42_42359


namespace sum_first_11_terms_of_arithmetic_sequence_l42_42748

noncomputable def sum_arithmetic_sequence (n : ℕ) (a1 an : ℤ) : ℤ :=
  n * (a1 + an) / 2

theorem sum_first_11_terms_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : S n = sum_arithmetic_sequence n (a 1) (a n))
  (h2 : a 3 + a 6 + a 9 = 60) : S 11 = 220 :=
sorry

end sum_first_11_terms_of_arithmetic_sequence_l42_42748


namespace analyze_convergence_l42_42655

noncomputable theory

open_locale classical

-- Define an arithmetic sequence
def arithmetic_sequence (u₀ r : ℝ) (n : ℕ) : ℝ := u₀ + n * r

-- Define the problem in Lean 4 as a theorem statement
theorem analyze_convergence (u₀ r : ℝ) :
  (r = 0 → ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |arithmetic_sequence u₀ r n - u₀| < ε) ∧
  (r > 0 → ∀ A : ℝ, ∃ N : ℕ, ∀ n ≥ N, arithmetic_sequence u₀ r n ≥ A) ∧
  (r < 0 → ∀ A : ℝ, ∃ N : ℕ, ∀ n ≥ N, arithmetic_sequence u₀ r n ≤ A) :=
by
  -- Statements are added to define each case of sequence convergence
  sorry

end analyze_convergence_l42_42655


namespace min_value_abs_z_sub_i_l42_42073

noncomputable def complex_number (z : ℂ) : Prop :=
  abs (conj z + 4 - 2 * complex.i) = 1

theorem min_value_abs_z_sub_i (z : ℂ) (h : complex_number z) : abs (z - complex.i) = 4 := sorry

end min_value_abs_z_sub_i_l42_42073


namespace find_angle_l42_42498

variables {a b : ℝ × ℝ}
noncomputable def norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Conditions
def condition_1 : Prop := norm a = 1
def condition_2 : Prop := norm b = 2
def condition_3 : Prop := a + b = (1, real.sqrt 2)

-- Question
def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  real.arccos ( (a.1 * b.1 + a.2 * b.2) / (norm a * norm b) )

-- Theorem
theorem find_angle (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) :
  angle_between_vectors a b = 2 * real.pi / 3 :=
sorry

end find_angle_l42_42498


namespace problem_1_problem_2_l42_42205

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42205


namespace coin_toss_sum_l42_42407

theorem coin_toss_sum :
  (∀ (r : ℚ), r ≠ 0 ∧ r ≠ 1 → (5 * r * (1 - r)^4 = 10 * r^2 * (1 - r)^3) → r = 1 / 3) →
  ∀ (heads_prob_three : ℚ), heads_prob_three = (choose 5 3 * (1 / 3)^3 * (2 / 3)^2) →
  (∃ (i j : ℕ), heads_prob_three = (i : ℚ) / (j : ℚ) ∧ nat.coprime i j ∧ i + j = 283).
Proof.
  intros h h_three_heads.
  have r_def := h (1 / 3).
  use 40, 243.
  split,
  {
    calc heads_prob_three = (choose 5 3 * (1 / 3)^3 * (2 / 3)^2) : by rw h_three_heads
    ... = 40 / 243 : by norm_num,
  },
  split,
  {
    exact nat.coprime_of_fraction_reduced 40 243,
  },
  exact eq.refl 283.

end coin_toss_sum_l42_42407


namespace hope_numbers_sum_l42_42507

def seq_a : ℕ → ℝ
| n := Real.logBase (n+1) (n+2)

noncomputable def is_hope_number (k : ℕ) : Prop :=
  ∃ m : ℕ, k = 2^m - 2 ∧ 1 ≤ k ∧ k ≤ 2010

noncomputable def sum_hope_numbers (n : ℕ) : ℕ :=
  if is_hope_number n then ∑ k in Finset.range (2011), if is_hope_number k then k else 0 else 0

theorem hope_numbers_sum : sum_hope_numbers 1 = 2026 := sorry

end hope_numbers_sum_l42_42507


namespace azalea_paid_shearer_l42_42809

noncomputable def amount_paid_to_shearer (number_of_sheep wool_per_sheep price_per_pound profit : ℕ) : ℕ :=
  let total_wool := number_of_sheep * wool_per_sheep
  let total_revenue := total_wool * price_per_pound
  total_revenue - profit

theorem azalea_paid_shearer :
  let number_of_sheep := 200
  let wool_per_sheep := 10
  let price_per_pound := 20
  let profit := 38000
  amount_paid_to_shearer number_of_sheep wool_per_sheep price_per_pound profit = 2000 := 
by
  sorry

end azalea_paid_shearer_l42_42809


namespace find_equation_of_C_find_equation_of_AB_l42_42206

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42206


namespace one_cubic_foot_is_1728_cubic_inches_l42_42100

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end one_cubic_foot_is_1728_cubic_inches_l42_42100


namespace exponential_gt_of_gt_l42_42496

theorem exponential_gt_of_gt {a b : ℝ} (h : a > b) : 2^a > 2^b :=
sorry

end exponential_gt_of_gt_l42_42496


namespace find_original_function_l42_42964

theorem find_original_function
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₁ : ∀ x, g (x + (π / 3)) = sin (x - (π / 4)))
  (h₂ : ∀ x, f x = g (2 * x)) :
  f x = sin (x / 2 + π / 12) :=
sorry

end find_original_function_l42_42964


namespace line_intersects_circle_l42_42921

theorem line_intersects_circle (α : ℝ) (r : ℝ) (hα : true) (hr : r > 0) :
  (∃ x y : ℝ, (x * Real.cos α + y * Real.sin α = 1) ∧ (x^2 + y^2 = r^2)) → r > 1 :=
by
  sorry

end line_intersects_circle_l42_42921


namespace length_of_shorter_base_l42_42451

theorem length_of_shorter_base 
  (length_midpoints_segment : ℝ)
  (longer_base : ℝ)
  (property_of_trapezoid : length_midpoints_segment = (longer_base - shorter_base) / 2) : 
  ∃ shorter_base : ℝ, shorter_base = 92 :=
by
  let length_midpoints_segment := 5
  let longer_base := 102
  have property_of_trapezoid := (length_midpoints_segment = (longer_base - 92) / 2)
  use 92
  sorry

end length_of_shorter_base_l42_42451


namespace find_b_range_a_plus_c_l42_42514

variable {α : Type*} [LinearOrderedField α]
variable (a b c : α) (A B C : α)

-- Given conditions
def triangle_conditions (a b c A B C : α) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a*c + b*c + b*a) > (a^2 + b^2 + c^2) ∧
  (cos B / b + cos C / c = (2 * sqrt 3 * sin A) / (3 * sin C)) ∧
  (cos B + sqrt 3 * sin B = 2)

-- Proving part 1
theorem find_b (h : triangle_conditions a b c A B C) : 
  b = sqrt 3 / 2 := 
sorry

-- Additional condition for part 2
def additional_condition_B (B : α) : Prop :=
  cos B + sqrt 3 * sin B = 2

-- Proving part 2
theorem range_a_plus_c (h : triangle_conditions a b c A B C) 
  (h_B : additional_condition_B B) : 
  sqrt 3 / 2 < a + c ∧ a + c ≤ sqrt 3 := 
sorry

end find_b_range_a_plus_c_l42_42514


namespace tetrahedrons_perpendicular_edges_l42_42543

theorem tetrahedrons_perpendicular_edges
    (A B : Fin 4 → ℝ^3)
    (H_perp : ∀ (i j k l : Fin 4), (({i, j, k, l} = {Fin.mk 0, Fin.mk 1, Fin.mk 2, Fin.mk 3}) →
      (i ≠ j ∧ k ≠ l) → ((i, j) ≠ (3,4) ∨ (k, l) ≠ (1,2)) →
      (A j - A i) • (B l - B k) = 0))
    : (A 4 - A 3) • (B 2 - B 1) = 0 :=
by
  sorry

end tetrahedrons_perpendicular_edges_l42_42543


namespace volume_of_revolution_calc_l42_42007

noncomputable def volume_of_solid_of_revolution : ℝ :=
  π * ∫ x in 0..π, (Real.sin x)^2

theorem volume_of_revolution_calc :
  volume_of_solid_of_revolution = π^2 / 2 :=
by
  sorry

end volume_of_revolution_calc_l42_42007


namespace equal_squares_isosceles_or_right_unequal_squares_vertices_position_l42_42139

-- Define the triangle without obtuse angles and the inscribed squares
variables {triangle : Type} [IsTriangle triangle]
variables (a b c : ℝ) -- lengths of the sides of the triangle
variables (h_a h_b h_c : ℝ) -- heights from the vertices opposite these sides
variables (x y : ℝ) -- side lengths of the inscribed squares
variable (no_obtuse : ∀ (α β γ : ℝ), α + β + γ = π ∧ α < π/2 ∧ β < π/2 ∧ γ < π/2)

-- First proof problem: If the squares are equal, then the triangle is either isosceles or right-angled.
theorem equal_squares_isosceles_or_right (h : x = y) : (a = b ∨ a = c ∨ b = c) ∨ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
sorry

-- Second proof problem: If the squares are not equal, then two vertices of the larger square lie on a side
-- that is smaller than the side of the triangle on which two vertices of the smaller square lie.
theorem unequal_squares_vertices_position (h : x ≠ y) :
  (x > y → (∀ (S1 S2 : ℝ), S1 < S2 ∧ S1 ≠ S2 ∧ (x = S1 ∧ y = S2) ∧ (∃ (s1 s2 : ℝ), s1 ≤ s2 ∧ x ≥ s1 ∧ y ≤ s2))) ∧ 
  (y > x → (∀ (S1 S2 : ℝ), S2 < S1 ∧ S1 ≠ S2 ∧ (y = S1 ∧ x = S2) ∧ (∃ (s1 s2 : ℝ), s2 ≤ s1 ∧ y ≥ s2 ∧ x ≤ s1))) :=
sorry

end equal_squares_isosceles_or_right_unequal_squares_vertices_position_l42_42139


namespace num_coloring_l42_42550

-- Define the set of numbers to be colored
def numbers_to_color : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define the set of colors
inductive Color
| red
| green
| blue

-- Define proper divisors for the numbers in the list
def proper_divisors (n : ℕ) : List ℕ :=
  match n with
  | 2 => []
  | 3 => []
  | 4 => [2]
  | 5 => []
  | 6 => [2, 3]
  | 7 => []
  | 8 => [2, 4]
  | 9 => [3]
  | _ => []

-- The proof statement
theorem num_coloring (h : ∀ n ∈ numbers_to_color, ∀ d ∈ proper_divisors n, n ≠ d) :
  ∃ f : ℕ → Color, ∀ n ∈ numbers_to_color, ∀ d ∈ proper_divisors n, f n ≠ f d :=
  sorry

end num_coloring_l42_42550


namespace sum_of_surface_points_l42_42761

theorem sum_of_surface_points
  (n : ℕ) (h_n : n = 2012) 
  (total_sum : ℕ) (h_total : total_sum = n * 21)
  (matching_points_sum : ℕ) (h_matching : matching_points_sum = (n - 1) * 7)
  (x : ℕ) (h_x_range : 1 ≤ x ∧ x ≤ 6) :
  (total_sum - matching_points_sum + 2 * x = 28177 ∨
   total_sum - matching_points_sum + 2 * x = 28179 ∨
   total_sum - matching_points_sum + 2 * x = 28181 ∨
   total_sum - matching_points_sum + 2 * x = 28183 ∨
   total_sum - matching_points_sum + 2 * x = 28185 ∨
   total_sum - matching_points_sum + 2 * x = 28187) :=
by sorry

end sum_of_surface_points_l42_42761


namespace sum_of_squares_eq_zero_iff_all_zero_l42_42547

theorem sum_of_squares_eq_zero_iff_all_zero (a b c : ℝ) :
  a^2 + b^2 + c^2 = 0 ↔ a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end sum_of_squares_eq_zero_iff_all_zero_l42_42547


namespace find_equation_of_C_find_equation_of_AB_l42_42211

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42211


namespace steve_marbles_l42_42312

theorem steve_marbles {S : ℤ} (sam_initial : ℤ) (sally_initial : ℤ) :
  (sam_initial = 2 * S) →
  (sally_initial = 2 * S - 5) →
  (sam_initial - 3 - 3 = 8) →
  (S + 3 = 10) :=
begin
  intros h1 h2 h3,
  sorry
end

end steve_marbles_l42_42312


namespace inversely_proportional_y_ratio_l42_42659

variable {k : ℝ}
variable {x₁ x₂ y₁ y₂ : ℝ}
variable (h_inv_prop : ∀ (x y : ℝ), x * y = k)
variable (hx₁x₂ : x₁ ≠ 0 ∧ x₂ ≠ 0)
variable (hy₁y₂ : y₁ ≠ 0 ∧ y₂ ≠ 0)
variable (hx_ratio : x₁ / x₂ = 3 / 4)

theorem inversely_proportional_y_ratio :
  y₁ / y₂ = 4 / 3 :=
by
  sorry

end inversely_proportional_y_ratio_l42_42659


namespace range_of_f_l42_42082

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x ^ 2 else Real.cos x

theorem range_of_f : Set.range f = Set.Ici (-1) := 
by
  sorry

end range_of_f_l42_42082


namespace ellipseEquation_distanceRange_l42_42586

-- Definitions based on conditions
def isEllipse (x y a b : ℝ) : Prop := (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)
def foci (f1 f2 : ℝ × ℝ) : Prop := f1.1 = -f2.1 ∧ f1.2 = 0 ∧ f2.2 = 0
def pointOnEllipse (E : ℝ × ℝ) (c b : ℝ) : Prop := E = (-c, (real.sqrt 2 / 2) * b)
def minorAxis (B : ℝ × ℝ) (b : ℝ) : Prop := B = (0, b)
def oeCondition (E f1 B : ℝ × ℝ) : Prop := vector.of E = vector.of f1 + (real.sqrt 2 / 2) • vector.of B
def perimeterCondition (E f1 f2 : ℝ × ℝ) : Prop := (E.euclideanDistance f1) + (E.euclideanDistance f2) + (f1.euclideanDistance f2) = 2 * (real.sqrt 2 + 1)
def isoscelesTriangle (M P Q : ℝ × ℝ) : Prop := (M.euclideanDistance P) = (M.euclideanDistance Q)
def lineThroughF2 (l : ℝ → ℝ) (F2 : ℝ × ℝ) (k : ℝ) : Prop := ∀ x, l(x) = k * (x - F2.1) ∧ k ≠ 0 

-- Main problem statement in Lean
theorem ellipseEquation :
  ∀ (a b c : ℝ) (x y : ℝ),
  ∀ (F1 F2 E B : ℝ × ℝ),
  isEllipse x y a b →
  foci F1 F2 →
  pointOnEllipse E c b →
  minorAxis B b →
  oeCondition E F1 B →
  perimeterCondition E F1 F2 →
  a = real.sqrt 2 ∧ b = 1 ∧ c = 1 ∧ (x^2 / 2 + y^2 = 1) :=
sorry

theorem distanceRange :
  ∀ (M F2 P Q : ℝ × ℝ) (f : ℝ → ℝ) (m : ℝ),
  pointOnLineSegment M F2 →
  ∃ k, lineThroughF2 f F2 k →
  isoscelesTriangle M P Q →
  ∀ d, 0 < d ∧ d < 1 / 2 :=
sorry

end ellipseEquation_distanceRange_l42_42586


namespace valid_fillings_board_a_valid_choices_A_B_total_valid_fillings_n2020_l42_42161

-- (a) Valid fillings of a specific board configuration
theorem valid_fillings_board_a:
  ∃ X Y, ((X = 0 ∨ X = 1) ∧ (Y = 0 ∨ Y = 1)) ∧
          X ≠ 0 ∧ Y ≠ 1 /\
          (Y = 0 ↔ (X = 1)) ∧ (Y = 1 ↔ (X = 0)) :=
begin
  sorry,
end

-- (b) Counting valid choices for A and B with given conditions
theorem valid_choices_A_B:
  ∃ (A B : ℕ), (A = 0 ∨ A = 1) ∧ (B = 0 ∨ B = 1) ∧
               (A ≠ B ∨ (A = 0 ∧ B = 1)) :=
begin
  sorry,
end

-- (c) Number of valid fillings for a 2 x 2020 board
theorem total_valid_fillings_n2020:
  (4 * 3^2019) = 
    nat.card { f : fin 2 × fin 2020 → fin 2 //
      ∀ (i : fin 2) (j : fin 2020), ¬(f (i, j) = f (i, j.succ) ∧ f (i.succ, j) = f (i.succ, j.succ)) } :=
begin
  sorry,
end

end valid_fillings_board_a_valid_choices_A_B_total_valid_fillings_n2020_l42_42161


namespace number_of_integer_points_l42_42476

def equation_satisfied (x y : ℤ) : Prop :=
  abs x ≠ 0 ∧ abs y ≠ 0 ∧ (1 / (abs x : ℚ) + 1 / (abs y : ℚ) = 1 / 2017)

theorem number_of_integer_points :
  {⟨x, y⟩ : ℤ × ℤ | equation_satisfied x y}.toFinset.card = 12 := 
sorry

end number_of_integer_points_l42_42476


namespace points_same_side_l42_42815

def parabola_a (x : ℝ) : ℝ := 2 * x ^ 2 + 4 * x
def parabola_b (x : ℝ) : ℝ := (x ^ 2) / 2 - x - 3 / 2
def parabola_c (x : ℝ) : ℝ := - x ^ 2 + 2 * x - 1
def parabola_d (x : ℝ) : ℝ := - x ^ 2 - 4 * x - 3
def parabola_e (x : ℝ) : ℝ := - x ^ 2 + 3

def point_A : ℝ × ℝ := (-1, -1)
def point_B : ℝ × ℝ := (0, 2)

theorem points_same_side (x y : ℝ) :
  (parabola_a (-1) < -1) = (parabola_a 0 < 2) ∧
  (parabola_c (-1) < -1) = (parabola_c 0 < 2) ∧
  (parabola_e (-1) < -1) = (parabola_e 0 < 2) :=
by
  sorry

end points_same_side_l42_42815


namespace even_with_odd_sum_more_than_odd_with_even_sum_l42_42377

-- Define a function to calculate sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

-- Define what it means for a number to be 'interesting'
def is_even_with_odd_sum (n : ℕ) : Prop :=
  (n % 2 = 0) ∧ (sum_of_digits n % 2 = 1)

def is_odd_with_even_sum (n : ℕ) : Prop :=
  (n % 2 = 1) ∧ (sum_of_digits n % 2 = 0)

-- Count the interesting numbers in the given range
def count_evens_with_odd_sums (upper : ℕ) : ℕ :=
  (List.range (upper + 1)).filter is_even_with_odd_sum |>.length

def count_odds_with_even_sums (upper : ℕ) : ℕ :=
  (List.range (upper + 1)).filter is_odd_with_even_sum |>.length

-- The main theorem
theorem even_with_odd_sum_more_than_odd_with_even_sum : 
  count_evens_with_odd_sums 1000000 > count_odds_with_even_sums 1000000 :=
by {
  sorry -- Proof will show that the count is greater
}

end even_with_odd_sum_more_than_odd_with_even_sum_l42_42377


namespace parabola_equation_line_AB_equation_l42_42222

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42222


namespace calculate_cost_price_l42_42817

noncomputable def cost_price (P : ℝ) : ℝ :=
  let discount_price := 0.88 * P
  let total_tax := 0.175 * discount_price
  let final_price := discount_price + total_tax
  let profit_rate := 0.18
  assume (P : ℝ) (hp : final_price = 720) (C : ℝ),
  final_price = 0.88 * P + 0.175 * (0.88 * P) →
  P = 720 / 1.034 →
  0.18 * C = 0.88 * P - C →
  C = 612.77 / 1.18 →
  C ≈ 519.29

theorem calculate_cost_price (P : ℝ) (C : ℝ) (federal_tax_rate state_tax_rate local_tax_rate discount_rate profit_rate final_sale_price : ℝ)
  (h1 : federal_tax_rate = 0.10)
  (h2 : state_tax_rate = 0.05)
  (h3 : local_tax_rate = 0.025)
  (h4 : discount_rate = 0.12)
  (h5 : profit_rate = 0.18)
  (h6 : final_sale_price = 720) :
  C ≈ 519.29 :=
by sorry

end calculate_cost_price_l42_42817


namespace max_S_n_of_arithmetic_sequence_l42_42589

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := 
  ∀ n m : ℕ, a (n + 1) = a n + (a (m + 1) - a m)

def S (n : ℕ) : ℤ :=
  nat.sum (fun i => a (i + 1)) n

theorem max_S_n_of_arithmetic_sequence (a : ℕ → ℤ)
  (h_arith: is_arithmetic_sequence a)
  (h_a5_pos : a 5 > 0)
  (h_a4_a7_neg : a 4 + a 7 < 0):
  ∀ n, S 5 ≥ S n :=
sorry

end max_S_n_of_arithmetic_sequence_l42_42589


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42282

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42282


namespace people_behind_yuna_l42_42381

theorem people_behind_yuna (total_people : ℕ) (people_in_front : ℕ) (yuna : ℕ)
  (h1 : total_people = 7) (h2 : people_in_front = 2) (h3 : yuna = 1) :
  total_people - people_in_front - yuna = 4 :=
by
  sorry

end people_behind_yuna_l42_42381


namespace part_one_part_two_l42_42230

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42230


namespace oranges_distributed_l42_42778

theorem oranges_distributed :
  ∀ (total_students : ℕ) (initial_oranges : ℕ) (bad_oranges : ℕ),
  total_students = 12 →
  initial_oranges = 108 →
  bad_oranges = 36 →
  let good_oranges := initial_oranges - bad_oranges in
  (initial_oranges / total_students - good_oranges / total_students) = 3 :=
by
  intros total_students initial_oranges bad_oranges ts_eq io_eq bo_eq
  let good_oranges := initial_oranges - bad_oranges
  sorry

end oranges_distributed_l42_42778


namespace product_equals_answer_l42_42900

noncomputable def complex_product : ℂ := (2 + complex.I) * (1 - 3 * complex.I)

theorem product_equals_answer : complex_product = 5 - 5 * complex.I := by
  -- proof will be here
  sorry  -- Placeholder for the actual proof.

end product_equals_answer_l42_42900


namespace divisibility_theorem_l42_42178

variable (a b c : ℕ)

-- Condition 1: c is not divisible by the square of any prime number
def not_divisible_by_square_of_prime (c : ℕ) : Prop :=
  ∀ p : ℕ, Nat.prime p → ¬ (p^2 ∣ c)

-- Condition 2: b^2 * c is divisible by a^2
def divisible_condition (a b c : ℕ) : Prop :=
  (a^2 ∣ b^2 * c)

-- The main theorem stating that b is divisible by a given the conditions
theorem divisibility_theorem (h1 : not_divisible_by_square_of_prime c)
                             (h2 : divisible_condition a b c) :
  a ∣ b :=
sorry

end divisibility_theorem_l42_42178


namespace fractional_part_frustum_l42_42801

noncomputable def base_edge : ℝ := 24
noncomputable def original_altitude : ℝ := 18
noncomputable def smaller_altitude : ℝ := original_altitude / 3

noncomputable def volume_pyramid (base_edge : ℝ) (altitude : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * altitude

noncomputable def volume_original : ℝ := volume_pyramid base_edge original_altitude
noncomputable def similarity_ratio : ℝ := (smaller_altitude / original_altitude) ^ 3
noncomputable def volume_smaller : ℝ := similarity_ratio * volume_original
noncomputable def volume_frustum : ℝ := volume_original - volume_smaller

noncomputable def fractional_volume_frustum : ℝ := volume_frustum / volume_original

theorem fractional_part_frustum : fractional_volume_frustum = 26 / 27 := by
  sorry

end fractional_part_frustum_l42_42801


namespace arrangements_of_6_people_l42_42403

open Finset

theorem arrangements_of_6_people:
  ∃ n : ℕ, n = 50 ∧
  (let total_people := 6
     let activity1_max := 4
     let activity2_max := 4 in
     ((2 ≤ activity1_max ∧ 4 ≤ total_people - 2 ∧
       (choose total_people 2) * (choose 2 2) +
       (3 ≤ activity1_max ∧ 3 ≤ total_people - 3 ∧
       (choose total_people 3)) = n)) :=
begin
  use 50,
  split,
  { refl, }, -- Proof that n = 50
  { -- Proof that the arrangements meet the criteria
    sorry
  }
end

end arrangements_of_6_people_l42_42403


namespace harmonic_sum_divisible_by_prime_square_l42_42180

theorem harmonic_sum_divisible_by_prime_square {p : ℕ} (a b : ℤ) (hp : p ≥ 5) (hp_prime : Nat.Prime p)
  (h_eq : (1 : ℚ) + (1 / 2) + ... + (1 / (p - 1)) = a / b) : p^2 ∣ a :=
sorry

end harmonic_sum_divisible_by_prime_square_l42_42180


namespace evaluate_expression_l42_42182

open Complex

theorem evaluate_expression (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + a * b + b^2 = 0) :
  (a^6 + b^6) / (a + b)^6 = 18 :=
by
  sorry

end evaluate_expression_l42_42182


namespace correct_formula_relation_l42_42929

theorem correct_formula_relation :
  ∀ (x y : ℕ), (x = 1 ∧ y = 3) ∨ (x = 2 ∧ y = 11) ∨ (x = 3 ∧ y = 23) ∨ (x = 4 ∧ y = 39) ∨ (x = 5 ∧ y = 59) →
  y = 2 * x^2 + 2 * x - 1 :=
by
  intro x y h
  cases h
  case inl h1 =>
    rw [h1.left, h1.right]
    simp
    norm_num
  case inr h2 =>
    cases h2
    case inl h2_1 =>
      rw [h2_1.left, h2_1.right]
      simp
      norm_num
    case inr h2_2 =>
      cases h2_2
      case inl h2_2_1 =>
        rw [h2_2_1.left, h2_2_1.right]
        simp
        norm_num
      case inr h2_2_2 =>
        cases h2_2_2
        case inl h2_2_2_1 =>
          rw [h2_2_2_1.left, h2_2_2_1.right]
          simp
          norm_num
        case inr h2_2_2_2 =>
          rw [h2_2_2_2.left, h2_2_2_2.right]
          simp
          norm_num

end correct_formula_relation_l42_42929


namespace circle_definitions_l42_42342

-- Definition of a circle's radius
def is_radius (center : Point) (p : Point) (c : Circle) : Prop :=
  p ∈ c ∧ dist center p = radius c

-- Definition: center (O) and any point on the circle (P) are defining properties
theorem circle_definitions (center : Point) (p : Point) (c : Circle) :
  (is_radius center p c) ∧ 
  (center_of_circle c = center) ∧ 
  (size_of_circle c = radius c) :=
sorry

end circle_definitions_l42_42342


namespace total_fruit_salads_is_1800_l42_42434

def Alaya_fruit_salads := 200
def Angel_fruit_salads := 2 * Alaya_fruit_salads
def Betty_fruit_salads := 3 * Angel_fruit_salads
def Total_fruit_salads := Alaya_fruit_salads + Angel_fruit_salads + Betty_fruit_salads

theorem total_fruit_salads_is_1800 : Total_fruit_salads = 1800 := by
  sorry

end total_fruit_salads_is_1800_l42_42434


namespace problem_A_problem_B_problem_C_problem_D_problem_E_l42_42015

-- Definitions and assumptions based on the problem statement
def eqI (x y z : ℕ) := x + y + z = 45
def eqII (x y z w : ℕ) := x + y + z + w = 50
def consecutive_odd_integers (x y z : ℕ) := y = x + 2 ∧ z = x + 4
def multiples_of_five (x y z w : ℕ) := (∃ a b c d : ℕ, x = 5 * a ∧ y = 5 * b ∧ z = 5 * c ∧ w = 5 * d)
def consecutive_integers (x y z w : ℕ) := y = x + 1 ∧ z = x + 2 ∧ w = x + 3
def prime_integers (x y z : ℕ) := Prime x ∧ Prime y ∧ Prime z

-- Lean theorem statements
theorem problem_A : ∃ x y z : ℕ, eqI x y z ∧ consecutive_odd_integers x y z := 
sorry

theorem problem_B : ¬ (∃ x y z : ℕ, eqI x y z ∧ prime_integers x y z) := 
sorry

theorem problem_C : ¬ (∃ x y z w : ℕ, eqII x y z w ∧ consecutive_odd_integers x y z) :=
sorry

theorem problem_D : ∃ x y z w : ℕ, eqII x y z w ∧ multiples_of_five x y z w := 
sorry

theorem problem_E : ∃ x y z w : ℕ, eqII x y z w ∧ consecutive_integers x y z w := 
sorry

end problem_A_problem_B_problem_C_problem_D_problem_E_l42_42015


namespace probability_two_out_of_three_is_0_384_l42_42345

def probability_correct_forecast (p : ℝ) (k : ℕ) (n : ℕ) : ℝ := 
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem probability_two_out_of_three_is_0_384 (p : ℝ) (h : p = 0.8) : 
  probability_correct_forecast p 2 3 = 0.384 :=
by
  rw h
  sorry

end probability_two_out_of_three_is_0_384_l42_42345


namespace find_x_y_l42_42473

theorem find_x_y (x y : ℝ) : 
  (x - 12) ^ 2 + (y - 13) ^ 2 + (x - y) ^ 2 = 1 / 3 ↔ (x = 37 / 3 ∧ y = 38 / 3) :=
by
  sorry

end find_x_y_l42_42473


namespace problem1_problem2_l42_42878

-- Problem 1: If a is parallel to b, then x = 4
theorem problem1 (x : ℝ) (u v : ℝ × ℝ) : 
  let a := (1, 1)
  let b := (4, x)
  (a.1 / b.1 = a.2 / b.2) → x = 4 := 
by 
  intros a b h
  dsimp [a, b] at h
  sorry

-- Problem 2: If (u - 2 * v) is perpendicular to (u + v), then x = -6
theorem problem2 (x : ℝ) (a u v : ℝ × ℝ) : 
  let a := (1, 1)
  let b := (4, x)
  let u := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let v := (2 * a.1 + b.1, 2 * a.2 + b.2)
  ((u.1 - 2 * v.1) * (u.1 + v.1) + (u.2 - 2 * v.2) * (u.2 + v.2) = 0) → x = -6 := 
by 
  intros a b u v h
  dsimp [a, b, u, v] at h
  sorry

end problem1_problem2_l42_42878


namespace parabola_equation_line_AB_equation_l42_42188

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42188


namespace snooker_tournament_l42_42421

theorem snooker_tournament : 
  ∀ (V G : ℝ),
    V + G = 320 →
    40 * V + 15 * G = 7500 →
    V ≥ 80 →
    G ≥ 100 →
    G - V = 104 :=
by
  intros V G h1 h2 h3 h4
  sorry

end snooker_tournament_l42_42421


namespace new_circle_contains_center_prob_l42_42776

noncomputable def probability_new_circle_contains_center (R : ℝ) (C : set (ℝ × ℝ)) (O : ℝ × ℝ)
(center_in_C : ∀ x ∈ C, ∃ (r : ℝ), 0 ≤ r ∧ r ≤ R - dist O x ∧ ball x r ⊆ C)
: ℝ :=
let prob := ∫ x in 0..R, (1 - x / R) in prob / R

theorem new_circle_contains_center_prob :
  ∀ (R : ℝ) (C : set (ℝ × ℝ)) (O : ℝ × ℝ)
  (center_in_C : ∀ x ∈ C, ∃ (r : ℝ), 0 ≤ r ∧ r ≤ R - dist O x ∧ ball x r ⊆ C),
  probability_new_circle_contains_center R C O center_in_C = 1 / 4 :=
begin
  assume R C O h,
  rw probability_new_circle_contains_center,
  sorry
end

end new_circle_contains_center_prob_l42_42776


namespace number_of_true_propositions_l42_42698

/-
  Proposition 1: The inverse proposition of "If xy = 1, then x and y are reciprocals of each other."
  Proposition 2: The negation of "Similar triangles have equal perimeters."
  Proposition 3: The contrapositive of "If A ∪ B = B, then A ⊇ B."
  Prove that there are exactly 2 true propositions among the above.
-/

theorem number_of_true_propositions : 
  let prop1 := ∃ (x y : ℝ), (x * y = 1) ∧ (x ≠ 1 ∨ y ≠ 1)
  let prop2 := ∃ (Δ₁ Δ₂ : Type) (P₁ P₂ : Δ₁ → Δ₂ -> Prop) (T₁ T₂ : Set Δ₁), ¬(P₁ = P₂ → T₁.perimeter = T₂.perimeter)
  let prop3 := ∀ (A B : Set ℕ), (A ∪ B = B) → (B ⊆ A) 
  ( (prop1 ∧ prop2 ∧ ¬prop3) ∨ (prop1 ∧ ¬prop2 ∧ prop3) ∨ (¬prop1 ∧ prop2 ∧ prop3) ) :=
sorry

end number_of_true_propositions_l42_42698


namespace Q_is_nice_l42_42414

def is_nice (P : Polynomial ℝ) : Prop :=
  P.eval 0 = 1 ∧ (∀ n : ℕ, P.coeff n ≠ 0 → if even n then P.coeff n = 1 else P.coeff n = -1)

def rel_prime (m : ℕ) (n : ℕ) : Prop := Nat.gcd m n = 1

def Q (P : Polynomial ℝ) (m n : ℕ) : Polynomial ℝ :=
  P.comp (X ^ n) * ((X ^ (m * n) - 1) * (X - 1) / ((X ^ m - 1) * (X ^ n - 1)))

theorem Q_is_nice (P : Polynomial ℝ) (m n : ℕ) (hP : is_nice P) (hmn : rel_prime m n) : is_nice (Q P m n) :=
by
  sorry

end Q_is_nice_l42_42414


namespace nontrivial_solution_fraction_l42_42865

theorem nontrivial_solution_fraction (x y z : ℚ)
  (h₁ : x - 6 * y + 3 * z = 0)
  (h₂ : 3 * x - 6 * y - 2 * z = 0)
  (h₃ : x + 6 * y - 5 * z = 0)
  (hne : x ≠ 0) :
  (y * z) / (x^2) = 2 / 3 :=
by
  sorry

end nontrivial_solution_fraction_l42_42865


namespace problem_1_problem_2_l42_42202

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42202


namespace one_cubic_foot_is_1728_cubic_inches_l42_42099

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end one_cubic_foot_is_1728_cubic_inches_l42_42099


namespace all_integers_ge_33_are_good_l42_42570

def good_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin k → ℕ), (∀ i, 0 < a i) ∧ (∑ i, (a i) = n) ∧ (∑ i, (1 : ℚ) / (a i) = 1)

theorem all_integers_ge_33_are_good :
  (∀ n, 33 ≤ n → good_number n) := by {
  -- Suppose we already know all integers from 33 to 73 are good numbers
  have base_case : ∀ n, 33 ≤ n ∧ n ≤ 73 → good_number n := sorry,
  
  -- We need to prove that all integers greater than 73 are good numbers
  have inductive_step : ∀ n, (33 ≤ n → good_number n) → (33 ≤ n+1 → good_number (n+1)) := sorry,
  
  -- Apply induction to conclude the theorem
  sorry
}

end all_integers_ge_33_are_good_l42_42570


namespace seq_not_square_l42_42639

open Nat

theorem seq_not_square (n : ℕ) (r : ℕ) :
  (r = 11 ∨ r = 111 ∨ r = 1111 ∨ ∃ k : ℕ, r = k * 10^(n + 1) + 1) →
  (r % 4 = 3) →
  (¬ ∃ m : ℕ, r = m^2) :=
by
  intro h_seq h_mod
  intro h_square
  sorry

end seq_not_square_l42_42639


namespace cubic_sum_cubes_l42_42662

noncomputable def p := sorry -- placeholder for a root
noncomputable def q := sorry -- placeholder for a root
noncomputable def r := sorry -- placeholder for a root

-- Assuming the conditions are met
axiom h1 : p + q + r = 4
axiom h2 : p * q + p * r + q * r = 3
axiom h3 : p * q * r = -6

-- Statement of the problem
theorem cubic_sum_cubes : p^3 + q^3 + r^3 = 34 :=
by
  sorry

end cubic_sum_cubes_l42_42662


namespace exist_value_of_x_l42_42949

theorem exist_value_of_x : ∃ x : ℝ, tan (3 * x) + cot (2 * x) = 2 ∧ x = 30 * (real.pi / 180) :=
by
  sorry

end exist_value_of_x_l42_42949


namespace find_k_and_b_integral_evaluation_l42_42880

noncomputable def f (k b x : ℝ) := (k * x + b) / Real.exp x

-- Part (I)
theorem find_k_and_b :
  (∃ (k b : ℝ), 
    let f := f k b
    (f (0) = 1) ∧ (deriv f (0)) = 1) ∧ 
    (∀ k b, 
      (let f := f k b
        f (0) = 1) ∧ (deriv f (0)) = 1 → k = 2 ∧ b = 1) :=
by
  sorry

-- Part (II)
theorem integral_evaluation :
  ∫ x in (0 : ℝ)..1, (x / Real.exp x) = 1 - 2 / Real.exp 1 :=
by
  sorry

end find_k_and_b_integral_evaluation_l42_42880


namespace lowest_fraction_done_by_two_people_l42_42465

theorem lowest_fraction_done_by_two_people : 
  ∀ (rate_A rate_B rate_C : ℕ) (hA : rate_A = 3) (hB : rate_B = 4) (hC : rate_C = 6),
  let work_A := 1 / rate_A,
      work_B := 1 / rate_B,
      work_C := 1 / rate_C in
  (work_B + work_C = 5 / 12) :=
by
  intros rate_A rate_B rate_C hA hB hC
  sorry

end lowest_fraction_done_by_two_people_l42_42465


namespace quadratic_roots_sum_product_l42_42487

theorem quadratic_roots_sum_product {p q : ℝ} 
  (h1 : p / 3 = 10) 
  (h2 : q / 3 = 15) : 
  p + q = 75 := sorry

end quadratic_roots_sum_product_l42_42487


namespace eduardo_overall_score_l42_42847

theorem eduardo_overall_score :
  let p1 := 15
  let c1 := p1 * 0.85 -- number of problems and correct answers in the first test
  let p2 := 25
  let c2 := 18 -- number of correct answers in the second test
  let p3 := 20
  let c3 := p3 * 0.925 -- number of problems and correct answers in the third test
  let total_problems := p1 + p2 + p3
  let total_correct := round c1 + c2 + round c3
  let overall_percentage := (total_correct / total_problems) * 100
  round overall_percentage = 83 :=
by
  sorry

end eduardo_overall_score_l42_42847


namespace population_by_eighth_year_l42_42669

theorem population_by_eighth_year :
  (∃ a : ℝ, ∀ x : ℝ, (y x = a * log (3 : ℝ) (x + 1)) ∧ y 2 = 100) →
  y 8 = 200 :=
by
  sorry

end population_by_eighth_year_l42_42669


namespace integral_1_integral_2_integral_3_integral_4_integral_5_l42_42442
open Real

-- Integral 1
theorem integral_1 : ∫ (x : ℝ), sin x * cos x ^ 3 = -1 / 4 * cos x ^ 4 + C :=
by sorry

-- Integral 2
theorem integral_2 : ∫ (x : ℝ), 1 / ((1 + sqrt x) * sqrt x) = 2 * log (1 + sqrt x) + C :=
by sorry

-- Integral 3
theorem integral_3 : ∫ (x : ℝ), x ^ 2 * sqrt (x ^ 3 + 1) = 2 / 9 * (x ^ 3 + 1) ^ (3/2) + C :=
by sorry

-- Integral 4
theorem integral_4 : ∫ (x : ℝ), (exp (2 * x) - 3 * exp x) / exp x = exp x - 3 * x + C :=
by sorry

-- Integral 5
theorem integral_5 : ∫ (x : ℝ), (1 - x ^ 2) * exp x = - (x - 1) ^ 2 * exp x + C :=
by sorry

end integral_1_integral_2_integral_3_integral_4_integral_5_l42_42442


namespace find_z_l42_42054

open Complex

theorem find_z (z : ℂ) (h : z * (2 - I) = 5 * I) : z = -1 + 2 * I :=
sorry

end find_z_l42_42054


namespace oranges_less_per_student_l42_42780

def total_students : ℕ := 12
def total_oranges : ℕ := 108
def bad_oranges : ℕ := 36

theorem oranges_less_per_student :
  (total_oranges / total_students) - ((total_oranges - bad_oranges) / total_students) = 3 :=
by
  sorry

end oranges_less_per_student_l42_42780


namespace max_abs_sum_l42_42946

-- Define the condition for the ellipse equation
def ellipse_condition (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Prove that the largest possible value of |x| + |y| given the condition is 2√3
theorem max_abs_sum (x y : ℝ) (h : ellipse_condition x y) : |x| + |y| ≤ 2 * Real.sqrt 3 :=
sorry

end max_abs_sum_l42_42946


namespace binary_to_octal_l42_42454

theorem binary_to_octal (b : ℕ) (h : b = 0b11100) : nat.toDigits 8 b = [4, 3] :=
by {
  rw h,
  sorry
}

end binary_to_octal_l42_42454


namespace transformation_correct_l42_42981

noncomputable def original_function : ℝ → ℝ :=
  λ x, sin (x / 2 + π / 12)

theorem transformation_correct :
  ∀ (x : ℝ), (shift_right (shorten_abscissas f) (π / 3)) x = sin (x - π / 4) →
  f = original_function :=
by
  -- Assume the shifted and shortened function equals the given sine transformation
  intro x
  sorry

end transformation_correct_l42_42981


namespace find_abc_l42_42119

noncomputable def ab : ℝ := 24 * real.root 3 (4⁻¹ : ℝ)
noncomputable def ac : ℝ := 40 * real.root 3 (4⁻¹ : ℝ)
noncomputable def bc : ℝ := 15 * real.root 3 (4⁻¹ : ℝ)

theorem find_abc (a b c : ℝ) (h1 : ab = a * b) (h2 : ac = a * c) (h3 : bc = b * c) :
  a * b * c = 120 * real.root (3: ℝ) (3⁻¹* 8⁻¹ : ℝ) :=
by sorry

end find_abc_l42_42119


namespace integral_x_squared_l42_42353

theorem integral_x_squared:
  ∫ x in (0:ℝ)..(1:ℝ), x^2 = 1/3 :=
by
  sorry

end integral_x_squared_l42_42353


namespace sixth_employee_salary_l42_42684

-- We define the salaries of the five employees
def salaries : List ℝ := [1000, 2500, 3100, 1500, 2000]

-- The mean of the salaries of these 5 employees and another employee
def mean_salary : ℝ := 2291.67

-- The number of employees
def number_of_employees : ℝ := 6

-- The total salary of the first five employees
def total_salary_5 : ℝ := salaries.sum

-- The total salary based on the given mean and number of employees
def total_salary_all : ℝ := mean_salary * number_of_employees

-- The statement to prove: The salary of the sixth employee
theorem sixth_employee_salary :
  total_salary_all - total_salary_5 = 3650.02 := 
  sorry

end sixth_employee_salary_l42_42684


namespace decimal_to_percentage_l42_42401

theorem decimal_to_percentage (d : ℝ) (h : d = 0.04) : (d * 100) = 4 := 
by {
  -- Given that d = 0.04
  rw h,
  -- Show that 0.04 * 100 = 4
  norm_num,
}

end decimal_to_percentage_l42_42401


namespace dice_surface_sum_l42_42750

theorem dice_surface_sum :
  ∃ X : ℤ, 1 ≤ X ∧ X ≤ 6 ∧ 
  (28175 + 2 * X = 28177 ∨
   28175 + 2 * X = 28179 ∨
   28175 + 2 * X = 28181 ∨
   28175 + 2 * X = 28183 ∨
   28175 + 2 * X = 28185 ∨
   28175 + 2 * X = 28187) := sorry

end dice_surface_sum_l42_42750


namespace chase_travel_time_l42_42446

-- Define the necessary mathematical structures
variable (time : Type)

-- Conditions
variable (chase_time cameron_time danielle_time : time)
variable (relation1 : cameron_time = 2 * chase_time)
variable (relation2 : danielle_time = 3 * cameron_time)
variable (danielle_given_time : danielle_time = 30)

-- Theorem statement
theorem chase_travel_time : chase_time = 180 := by
  sorry

end chase_travel_time_l42_42446


namespace find_r_l42_42592

variable (a_n : ℕ → ℝ)
variable (r : ℝ)

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∃ a₁ : ℝ, ∀ n : ℕ, a_n n = a₁ * q ^ n

axiom sum_of_first_n_terms (n : ℕ) : (∑ i in finset.range (n + 1), a_n i) = 3^n + r

theorem find_r (a_n : ℕ → ℝ) (r : ℝ) : is_geometric_sequence a_n → (∀ n : ℕ, (∑ i in finset.range (n + 1), a_n i) = 3^n + r) → r = -1 := 
  by
  sorry

end find_r_l42_42592


namespace region_area_proof_l42_42001

noncomputable def region_area := 
  let region := {p : ℝ × ℝ | abs (p.1 - p.2^2 / 2) + p.1 + p.2^2 / 2 ≤ 2 - p.2}
  2 * (0.5 * (3 * (2 + 0.5)))

theorem region_area_proof : region_area = 15 / 2 :=
by
  sorry

end region_area_proof_l42_42001


namespace original_average_l42_42650

noncomputable theory
open_locale classical

-- Given conditions
variables (S : Finset ℝ) (n : ℕ) (A : ℝ)
-- Set S contains exactly 10 numbers
-- The new average of set S is 7 after increasing one element by 8

theorem original_average (h1 : S.card = 10) 
                         (h2 : (S.sum + 8) / 10 = 7) :
  (S.sum / 10) = 6.2 :=
sorry

end original_average_l42_42650


namespace V4_plus_2V2_eq_S1_plus_2S3_l42_42601

section
variables {n : ℕ} (S1 S3 V2 V4 : ℕ)

-- Definitions based on the conditions
def S1_def := (list.sum (list.range (2*n + 1)))
def S3_def := (list.sum (list.map (λ x, x^3) (list.range (2*n + 1))))
def V2_def := (list.sum (list.map (λ x, if x % 2 = 0 then x^2 else -x^2) (list.range (2*n))))
def V4_def := (list.sum (list.map (λ x, if x % 2 = 0 then x^4 else -x^4) (list.range (2*n))))

-- Proof goal
theorem V4_plus_2V2_eq_S1_plus_2S3 (h1 : S1 = S1_def n) 
                                  (h2 : S3 = S3_def n) 
                                  (h3 : V2 = V2_def n) 
                                  (h4 : V4 = V4_def n) :
  V4 + 2 * V2 = S1 + 2 * S3 := 
by sorry
end

end V4_plus_2V2_eq_S1_plus_2S3_l42_42601


namespace part1_eq_C_part2_line_AB_l42_42286

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42286


namespace power_function_through_point_l42_42904

-- Given the power function f(x) = x^n passes through the point (2, 8), prove that n = 3.
theorem power_function_through_point (n : ℝ) (f : ℝ → ℝ) (h : f = λ x, x^n) (h1 : f 2 = 8) : n = 3 := 
by sorry

end power_function_through_point_l42_42904


namespace speeds_of_cars_l42_42707

theorem speeds_of_cars (d_A d_B : ℝ) (v_A v_B : ℝ) (h1 : d_A = 300) (h2 : d_B = 250) (h3 : v_A = v_B + 5) (h4 : d_A / v_A = d_B / v_B) :
  v_B = 25 ∧ v_A = 30 :=
by
  sorry

end speeds_of_cars_l42_42707


namespace number_of_true_propositions_l42_42503

theorem number_of_true_propositions 
  (m : Line) (α β : Plane) (n : Line)
  (hmα : m ⟂ α) (hnβ : n ∈ β) : 
  (1: ℕ) + (if α || β then (m ⟂ n : Prop) else false) + (if α ⟂ β then (m || n : Prop) else false) + (if m || n then (α ⟂ β : Prop) else false) = 2 := 
sorry

end number_of_true_propositions_l42_42503


namespace mod_37_5_l42_42347

theorem mod_37_5 : 37 % 5 = 2 :=
by
  sorry

end mod_37_5_l42_42347


namespace solve_poly_eq_l42_42031

theorem solve_poly_eq (x : ℂ) : 
  (x^4 = 64) ↔ (x = 2 * real.sqrt 2 ∨ x = -2 * real.sqrt 2 ∨ x = 2 * real.sqrt 2 * complex.I ∨ x = -2 * real.sqrt 2 * complex.I) :=
by sorry

end solve_poly_eq_l42_42031


namespace equation_of_parabola_equation_of_line_AB_l42_42257

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42257


namespace smallest_n_contains_constant_term_l42_42482

theorem smallest_n_contains_constant_term :
  ∃ n : ℕ, (∀ x : ℝ, x ≠ 0 → (2 * x^3 + 1 / x^(1/2))^n = c ↔ n = 7) :=
by
  sorry

end smallest_n_contains_constant_term_l42_42482


namespace find_equation_of_C_find_equation_of_AB_l42_42207

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42207


namespace spider_paths_l42_42781

theorem spider_paths : (Nat.choose (7 + 3) 3) = 210 := 
by
  sorry

end spider_paths_l42_42781


namespace m_add_n_equals_19_l42_42895

theorem m_add_n_equals_19 (n m : ℕ) (A_n_m : ℕ) (C_n_m : ℕ) (h1 : A_n_m = 272) (h2 : C_n_m = 136) :
  m + n = 19 :=
by
  sorry

end m_add_n_equals_19_l42_42895


namespace solve_quadratic_l42_42323

theorem solve_quadratic (x : ℝ) (h_pos : x > 0) (h_eq : 6 * x^2 + 9 * x - 24 = 0) : x = 4 / 3 :=
by
  sorry

end solve_quadratic_l42_42323


namespace child_ticket_price_correct_l42_42380

-- Definitions based on conditions
def total_collected := 104
def price_adult := 6
def total_tickets := 21
def children_tickets := 11

-- Derived conditions
def adult_tickets := total_tickets - children_tickets
def total_revenue_child (C : ℕ) := children_tickets * C
def total_revenue_adult := adult_tickets * price_adult

-- Main statement to prove
theorem child_ticket_price_correct (C : ℕ) 
  (h1 : total_revenue_child C + total_revenue_adult = total_collected) : 
  C = 4 :=
by
  sorry

end child_ticket_price_correct_l42_42380


namespace specific_clothing_choice_probability_l42_42944

noncomputable def probability_of_specific_clothing_choice : ℚ :=
  let total_clothing := 4 + 5 + 6
  let total_ways_to_choose_3 := Nat.choose 15 3
  let ways_to_choose_specific_3 := 4 * 5 * 6
  let probability := ways_to_choose_specific_3 / total_ways_to_choose_3
  probability

theorem specific_clothing_choice_probability :
  probability_of_specific_clothing_choice = 24 / 91 :=
by
  -- proof here 
  sorry

end specific_clothing_choice_probability_l42_42944


namespace ecuadorian_number_count_l42_42412

def isEcuadorian (n : ℕ) : Prop :=
  let digits := (List.range 5).map (fun i => (n / 10^i) % 10)
  let a := digits.get! 4
  let b := digits.get! 3
  let c := digits.get! 2
  let d := digits.get! 1
  let e := digits.get! 0
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  a = b + c + d + e

theorem ecuadorian_number_count : (Finset.filter isEcuadorian (Finset.range 90000)).card = 168 :=
sorry

end ecuadorian_number_count_l42_42412


namespace simplify_expression_l42_42713

theorem simplify_expression : (3^3 * 3^(-4)) / (3^2 * 3^(-1)) = 1 / 9 := by
  sorry

end simplify_expression_l42_42713


namespace original_function_l42_42985

theorem original_function :
  ∃ f : ℝ → ℝ,
  (∀ x : ℝ, sin (x - π / 4) = f (2 * (x - π / 3))) →
  (∀ x : ℝ, f x = sin (x / 2 + π / 12)) :=
by
  sorry

end original_function_l42_42985


namespace smallest_multiple_9_11_13_l42_42481

theorem smallest_multiple_9_11_13 : ∃ n : ℕ, n > 0 ∧ (9 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) ∧ n = 1287 := 
 by {
   sorry
 }

end smallest_multiple_9_11_13_l42_42481


namespace arithmetic_seq_problem_l42_42896

theorem arithmetic_seq_problem (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, S n = n * (a 1 + a n) / 2) →
  S 19 = 57 →
  3 * (a 1 + 4 * d) - a 1 - (a 1 + 3 * d) = 3 :=
by
  sorry

end arithmetic_seq_problem_l42_42896


namespace solve_equation_l42_42075

theorem solve_equation (Y : ℝ) : (3.242 * 10 * Y) / 100 = 0.3242 * Y := 
by 
  sorry

end solve_equation_l42_42075


namespace angle_between_HK_and_median_is_15_l42_42633

variable (H K A B C : Type) [MetricSpace H] [MetricSpace K] [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (triangle_ABC : EquilateralTriangle A B C)
variable (segment_AB : Segment A B)
variable (triangle_AHB : RightTriangle A H B)

-- Definitions based on given conditions
def angle_HBA_60 (H B A : Type) [HasAngle H B A] : Angle H B A := 60
def angle_CAK_15 (C A K : Type) [HasAngle C A K] : Angle C A K := 15

-- Theorem statement
theorem angle_between_HK_and_median_is_15 :
  ∀ (H K A B C : Type) [MetricSpace H] [MetricSpace K] [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (triangle_ABC : EquilateralTriangle A B C)
    (segment_AB : Segment A B)
    (triangle_AHB : RightTriangle A H B)
    (angle_HBA_is_60 : angle_HBA_60 H B A = 60)
    (point_K_on_ray_BC_beyond_C : LiesOnRay K C B) 
    (angle_CAK_is_15 : angle_CAK_15 C A K = 15),
  angle_between_lines (Line HK) (Median_AHB H B) = 15 :=
  by sorry

end angle_between_HK_and_median_is_15_l42_42633


namespace eq1_passing_through_P_eq2_x_intercept_eq3_y_intercept_l42_42087

-- Definitions based on conditions
def line1 (x y : ℝ) : Prop := y = - (Real.sqrt 3 / 3) * x + 5

def angle_inclination_line1 : ℝ := 150
def angle_inclination_line_l : ℝ := 30
def slope_line_l : ℝ := Real.sqrt 3 / 3

def point_P : ℝ × ℝ := (3, -4)

def x_intercept_line_l : ℝ := -2
def y_intercept_line_l : ℝ := 3

-- Proof statements

theorem eq1_passing_through_P : 
  (∀ x y, (y = slope_line_l * (x - point_P.1) + point_P.2) → (Real.sqrt 3) * x - (3 * y) - (3 * Real.sqrt 3) - 12 = 0) := sorry

theorem eq2_x_intercept :
  (∀ x y, (x = x_intercept_line_l ∧ y = 0) → (y = slope_line_l * (x + 2)) → (Real.sqrt 3) * x - (3 * y) + (2 * Real.sqrt 3) = 0) := sorry

theorem eq3_y_intercept :
  (∀ x y, (∃ y, y = y_intercept_line_l) → (y = slope_line_l * x + 3) → (Real.sqrt 3) * x - (3 * y) + 9 = 0) := sorry

end eq1_passing_through_P_eq2_x_intercept_eq3_y_intercept_l42_42087


namespace jake_total_payment_l42_42156

-- Definitions based on conditions
def packages : ℕ := 3
def weight_per_package : ℕ := 2
def price_per_pound : ℕ := 4

-- Theorem to prove the total cost
theorem jake_total_payment : 
  let total_pounds := packages * weight_per_package in
  let total_cost := total_pounds * price_per_pound in
  total_cost = 24 :=
by
  sorry

end jake_total_payment_l42_42156


namespace tangent_line_equation_at_origin_l42_42474

open Real

def curve (x : ℝ) : ℝ := exp (-5 * x) + 2

noncomputable def tangent_line_at_point (x : ℝ) (y : ℝ) : Prop :=
  ∃ m b, y = m * x + b ∧ (∀ x, y = -5 * x + 3)

theorem tangent_line_equation_at_origin : tangent_line_at_point 0 3 :=
  sorry

end tangent_line_equation_at_origin_l42_42474


namespace num_sum_of_cubes_lt_300_l42_42938

def is_cube (n : ℤ) : Prop := ∃ k : ℕ, k^3 = n

def valid_cubes (n : ℤ) : Prop := 
  ∃ a b : ℕ, 1 ≤ a ∧ 1 ≤ b ∧ a ≤ 6 ∧ b ≤ 6 ∧ n = a^3 + b^3

theorem num_sum_of_cubes_lt_300 : 
  ( finset.filter (λ n, valid_cubes n) (finset.range 300)).card = 19 := 
by {
  sorry
}

end num_sum_of_cubes_lt_300_l42_42938


namespace tangent_line_equation_at_point_l42_42672

theorem tangent_line_equation_at_point : 
  (∀ (x y : ℝ), y = x^2 + 3 → ((x, y) = (1, 4)) → 2 * x - y + 2 = 0) :=
begin
  intros x y curve_eq point_eq,
  sorry
end

end tangent_line_equation_at_point_l42_42672


namespace no_30_cents_l42_42321

/-- Given six coins selected from nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total value of the six coins cannot be 30 cents or less. -/
theorem no_30_cents {n d q : ℕ} (h : n + d + q = 6) (hn : n * 5 + d * 10 + q * 25 <= 30) : false :=
by
  sorry

end no_30_cents_l42_42321


namespace mountain_loop_trail_l42_42635

variable {a b c d e : ℕ}

-- Conditions
def cond1 : Prop := a + b = 28
def cond2 : Prop := b + c = 28
def cond3 : Prop := d + e = 36
def cond4 : Prop := a + c = 30

theorem mountain_loop_trail : cond1 ∧ cond2 ∧ cond3 ∧ cond4 → a + b + c + d + e = 94 := by
  -- This is where the proof would go, but we omit it for now
  sorry

end mountain_loop_trail_l42_42635


namespace angle_bisector_length_l42_42140

theorem angle_bisector_length
  (a β γ : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < β) 
  (h3 : β < π) 
  (h4 : 0 < γ) 
  (h5 : γ < π) 
  (h6 : β + γ < π) : 
  let AM := a * sin γ * sin β / (sin (β + γ) * cos ((γ - β) / 2)) in
  AM = a * sin γ * sin β / (sin (β + γ) * cos ((γ - β) / 2)) :=
by simp [sin, cos]; sorry

end angle_bisector_length_l42_42140


namespace example_12142_divisible_by_13_l42_42003

theorem example_12142_divisible_by_13 :
  -- Given conditions
  (∀ n, (n = 99) → (nat.prime 3 ∧ nat.prime 11) ∧ (nat.factors 99 = [3, 3, 11])) ∧
  (∀ n, (n = 1001) → (nat.prime 7 ∧ nat.prime 11 ∧ nat.prime 13) ∧ (nat.factors 1001 = [7, 11, 13])) ∧
  (∀ n, (nat.divisible_by_11 n ↔ let (odds, evens) := (nat.sum_digits_positions n odd, nat.sum_digits_positions n even) in (odds - evens) % 11 = 0)) ∧
  (∀ n, (let (n3) := nat.alternating_sum_triplets n in (n % 1001 = n3) → (nat.divisible_7_11_13 n3 ↔ (n3 % 7 = 0 ∨ n3 % 11 = 0 ∨ n3 % 13 = 0))) → 
  -- Conclusion
  nat.divisible 12142 13 :=
sorry

end example_12142_divisible_by_13_l42_42003


namespace sum_sequence_tangent_ord_l42_42863

theorem sum_sequence_tangent_ord (n : ℕ) (hn : 0 < n) (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ x, y x = x^n * (1 - x)) ∧
  (∀ x, y' x = n * x^(n - 1) - (n + 1) * x^n) ∧
  a n = (n + 1) * 2^n ∧
  (∀ k ≤ n, S n = ∑ i in finset.range (n + 1), a i) →
  S n = n * 2^(n + 1) :=
sorry

end sum_sequence_tangent_ord_l42_42863


namespace part_one_part_two_l42_42235

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42235


namespace covering_height_l42_42827

variables {n : ℕ} (P : ℝ → ℝ → Prop) (f : ℝ → ℝ)
  (A : ℝ) (p : ℝ) (Q : ℝ × ℝ) (h : ℝ)

-- Define what makes a set of points a convex n-gon
def is_convex_ngon (P : ℝ → ℝ → Prop) : Prop :=
  ∀ x y z : ℝ × ℝ, P x → P y → P z →
    (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ a * x.1 + b * y.1 + c * z.1 = x.1 ∧
     a * x.2 + b * y.2 + c * z.2 = x.2)

-- Define minimum distance f(P) from a point to sides of the polygon
def min_distance_to_sides (P : ℝ × ℝ) (n : ℕ) : ℝ := sorry

-- Given a convex n-gon having perimeter p and area A,
-- show that the covering height h can be given by f(Q) where Q is the point that maximizes f.
theorem covering_height (P : ℝ × ℝ → Prop) (Q : ℝ × ℝ) (max_f : ∀ x, P x → f x ≤ f Q)
  (h : ℝ) (convex_ngon : is_convex_ngon P) 
  (p f : ℝ) (A : ℝ) :
  ∃ Q, h = f Q ∧ (∀ x, P x → h ≤ f x) ∧ (p * h ≤ 2 * A) := sorry

end covering_height_l42_42827


namespace train_speed_in_kmph_l42_42425

-- Definitions for the given problem conditions
def length_of_train : ℝ := 110
def length_of_bridge : ℝ := 240
def time_to_cross_bridge : ℝ := 20.99832013438925

-- Main theorem statement
theorem train_speed_in_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = 60.0084 := 
by
  sorry

end train_speed_in_kmph_l42_42425


namespace divides_x_by_5_l42_42306

theorem divides_x_by_5 (x y : ℤ) (hx1 : 1 < x) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : 2 * x^2 - 1 = y^15) : 5 ∣ x := by
  sorry

end divides_x_by_5_l42_42306


namespace factorize_poly_l42_42727

theorem factorize_poly :
  (x : ℤ) → x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) :=
by
  sorry

end factorize_poly_l42_42727


namespace m_interval_l42_42459

def x : ℕ → ℝ 
| 0       := 7
| (n + 1) := (x n ^ 2 + 6 * x n + 5) / (x n + 7)

def m := Inf { n : ℕ | x n ≤ 4 + 1 / 2^25 }

theorem m_interval : 271 ≤ m ∧ m ≤ 810 := 
by sorry

end m_interval_l42_42459


namespace minimum_degree_g_l42_42065

open Polynomial

theorem minimum_degree_g (f g h : Polynomial ℝ) 
  (h_eq : 5 • f + 2 • g = h)
  (deg_f : f.degree = 11)
  (deg_h : h.degree = 12) : 
  ∃ d : ℕ, g.degree = d ∧ d >= 12 := 
sorry

end minimum_degree_g_l42_42065


namespace chase_travel_time_l42_42447

-- Define the necessary mathematical structures
variable (time : Type)

-- Conditions
variable (chase_time cameron_time danielle_time : time)
variable (relation1 : cameron_time = 2 * chase_time)
variable (relation2 : danielle_time = 3 * cameron_time)
variable (danielle_given_time : danielle_time = 30)

-- Theorem statement
theorem chase_travel_time : chase_time = 180 := by
  sorry

end chase_travel_time_l42_42447


namespace weekly_earnings_l42_42546

theorem weekly_earnings :
  let hours_Monday := 2
  let minutes_Tuesday := 75
  let start_Thursday := (15, 10) -- 3:10 PM in (hour, minute) format
  let end_Thursday := (17, 45) -- 5:45 PM in (hour, minute) format
  let minutes_Saturday := 45

  let pay_rate_weekday := 4 -- \$4 per hour
  let pay_rate_weekend := 5 -- \$5 per hour

  -- Convert time to hours
  let hours_Tuesday := minutes_Tuesday / 60.0
  let Thursday_work_minutes := (end_Thursday.1 * 60 + end_Thursday.2) - (start_Thursday.1 * 60 + start_Thursday.2)
  let hours_Thursday := Thursday_work_minutes / 60.0
  let hours_Saturday := minutes_Saturday / 60.0

  -- Calculate earnings
  let earnings_Monday := hours_Monday * pay_rate_weekday
  let earnings_Tuesday := hours_Tuesday * pay_rate_weekday
  let earnings_Thursday := hours_Thursday * pay_rate_weekday
  let earnings_Saturday := hours_Saturday * pay_rate_weekend

  -- Total earnings
  let total_earnings := earnings_Monday + earnings_Tuesday + earnings_Thursday + earnings_Saturday

  total_earnings = 27.08 := by sorry

end weekly_earnings_l42_42546


namespace find_parabola_equation_find_line_AB_equation_l42_42240

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42240


namespace number_of_feet_on_branches_l42_42355

def number_of_birds : ℕ := 46
def feet_per_bird : ℕ := 2

theorem number_of_feet_on_branches : number_of_birds * feet_per_bird = 92 := 
by 
  sorry

end number_of_feet_on_branches_l42_42355


namespace alicia_total_deductions_in_cents_l42_42427

def Alicia_hourly_wage : ℝ := 25
def local_tax_rate : ℝ := 0.015
def retirement_contribution_rate : ℝ := 0.03

theorem alicia_total_deductions_in_cents :
  let wage_cents := Alicia_hourly_wage * 100
  let tax_deduction := wage_cents * local_tax_rate
  let after_tax_earnings := wage_cents - tax_deduction
  let retirement_contribution := after_tax_earnings * retirement_contribution_rate
  let total_deductions := tax_deduction + retirement_contribution
  total_deductions = 111 :=
by
  sorry

end alicia_total_deductions_in_cents_l42_42427


namespace calculate_kevin_training_time_l42_42600

theorem calculate_kevin_training_time : 
  ∀ (laps : ℕ) 
    (track_length : ℕ) 
    (run1_distance : ℕ) 
    (run1_speed : ℕ) 
    (walk_distance : ℕ) 
    (walk_speed : Real) 
    (run2_distance : ℕ) 
    (run2_speed : ℕ) 
    (minutes : ℕ) 
    (seconds : Real),
    laps = 8 →
    track_length = 500 →
    run1_distance = 200 →
    run1_speed = 3 →
    walk_distance = 100 →
    walk_speed = 1.5 →
    run2_distance = 200 →
    run2_speed = 4 →
    minutes = 24 →
    seconds = 27 →
    (∀ (t1 t2 t3 t_total t_training : Real),
      t1 = run1_distance / run1_speed →
      t2 = walk_distance / walk_speed →
      t3 = run2_distance / run2_speed →
      t_total = t1 + t2 + t3 →
      t_training = laps * t_total →
      t_training = (minutes * 60 + seconds)) := 
by
  intros laps track_length run1_distance run1_speed walk_distance walk_speed run2_distance run2_speed minutes seconds
  intros h_laps h_track_length h_run1_distance h_run1_speed h_walk_distance h_walk_speed h_run2_distance h_run2_speed h_minutes h_seconds
  intros t1 t2 t3 t_total t_training
  intros h_t1 h_t2 h_t3 h_t_total h_t_training
  sorry

end calculate_kevin_training_time_l42_42600


namespace remainder_of_x_squared_l42_42123

theorem remainder_of_x_squared (x : ℤ) (h1 : 5 * x ≡ 10 [MOD 25]) (h2 : 4 * x ≡ 20 [MOD 25]) : (x^2) % 25 = 0 := 
by
  sorry

end remainder_of_x_squared_l42_42123


namespace jake_total_payment_l42_42157

-- Definitions based on conditions
def packages : ℕ := 3
def weight_per_package : ℕ := 2
def price_per_pound : ℕ := 4

-- Theorem to prove the total cost
theorem jake_total_payment : 
  let total_pounds := packages * weight_per_package in
  let total_cost := total_pounds * price_per_pound in
  total_cost = 24 :=
by
  sorry

end jake_total_payment_l42_42157


namespace initial_number_of_peanuts_l42_42598

theorem initial_number_of_peanuts (x : ℕ) (h : x + 2 = 6) : x = 4 :=
sorry

end initial_number_of_peanuts_l42_42598


namespace problem1_problem2_l42_42271

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42271


namespace am_gm_inequality_l42_42641

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) :=
sorry

end am_gm_inequality_l42_42641


namespace coordinates_C_on_segment_AB_l42_42637

theorem coordinates_C_on_segment_AB :
  ∃ C : (ℝ × ℝ), 
  (C.1 = 2 ∧ C.2 = 6) ∧
  ∃ A B : (ℝ × ℝ), 
  (A = (-1, 0)) ∧ 
  (B = (3, 8)) ∧ 
  (∃ k : ℝ, (k = 3) ∧ dist (C) (A) = k * dist (C) (B)) :=
by
  sorry

end coordinates_C_on_segment_AB_l42_42637


namespace minimum_number_of_elements_l42_42062

theorem minimum_number_of_elements (n : ℕ) (h : n ≥ 2) 
  (a : Fin n.succ → ℕ) 
  (h₀ : a 0 = 0) 
  (h₁ : ∀ i j, i ≤ j → a i < a j) 
  (h₂ : a ⟨n, h⟩ = 2 * n - 1) :
  ∃ (S : Set ℕ), 
  S = {a i + a j | i j : Fin n.succ, i ≤ j} ∧ 
  S.card = 3 * n :=
by sorry

end minimum_number_of_elements_l42_42062


namespace changhee_avg_score_is_correct_l42_42449

variable (midterm_avg : ℝ) (midterm_subjects : ℕ) (final_avg : ℝ) (final_subjects : ℕ)

def total_points (avg : ℝ) (subjects : ℕ) := avg * subjects

def changhee_average_score (mid_avg final_avg : ℝ) (mid_subj final_subj : ℕ) :=
  (total_points mid_avg mid_subj + total_points final_avg final_subj) / (mid_subj + final_subj)

theorem changhee_avg_score_is_correct :
  midterm_avg = 83.1 → midterm_subjects = 10 →
  final_avg = 84 → final_subjects = 8 →
  changhee_average_score midterm_avg final_avg midterm_subjects final_subjects = 83.5 :=
by
  intros h1 h2 h3 h4
  simp [changhee_average_score, total_points, h1, h2, h3, h4]
  norm_num
  sorry

end changhee_avg_score_is_correct_l42_42449


namespace sum_of_valid_n_l42_42834

theorem sum_of_valid_n :
  let valid_n (n : ℕ) := (n > 0) ∧ (n ≤ 100) ∧ ∃ k : ℕ, (n! * (n+1)! / 2 = k^2),
  let ns := (Finset.filter valid_n (Finset.range 101)),
  let sum_n := (ns.sum id)
  in sum_n = 273 :=
by
  -- Define valid_n predicate
  let valid_n (n : ℕ) := (n > 0) ∧ (n ≤ 100) ∧ ∃ k : ℕ, (n! * (n+1)! / 2 = k^2),
  -- Define the set of valid ns
  let ns := (Finset.filter valid_n (Finset.range 101)),
  -- Calculate the sum of filtered ns
  let sum_n := (ns.sum id),
  -- Assert and prove the equality
  show sum_n = 273,
  sorry

end sum_of_valid_n_l42_42834


namespace total_failing_grades_l42_42575

theorem total_failing_grades (k : ℕ) (a : ℕ → ℕ) :
  (∀ n, n ≤ k → a n ≥ a (n + 1)) →
  (∑ i in Finset.range k, a i) = a 1 + a 2 + ... + a k := 
by
  sorry

end total_failing_grades_l42_42575


namespace molecular_weight_of_compound_l42_42374

def atomic_weight_Al : ℕ := 27
def atomic_weight_I : ℕ := 127
def atomic_weight_O : ℕ := 16

def num_Al : ℕ := 1
def num_I : ℕ := 3
def num_O : ℕ := 2

def molecular_weight (n_Al n_I n_O w_Al w_I w_O : ℕ) : ℕ :=
  (n_Al * w_Al) + (n_I * w_I) + (n_O * w_O)

theorem molecular_weight_of_compound :
  molecular_weight num_Al num_I num_O atomic_weight_Al atomic_weight_I atomic_weight_O = 440 := 
sorry

end molecular_weight_of_compound_l42_42374


namespace unique_a_for_intersection_l42_42089

def A (a : ℝ) : Set ℝ := {-4, 2 * a - 1, a^2}
def B (a : ℝ) : Set ℝ := {a - 5, 1 - a, 9}

theorem unique_a_for_intersection (a : ℝ) :
  (9 ∈ A a ∩ B a ∧ ∀ x, x ∈ A a ∩ B a → x = 9) ↔ a = -3 := by
  sorry

end unique_a_for_intersection_l42_42089


namespace total_three_digit_numbers_l42_42702

noncomputable def card1 := {0, 2}
noncomputable def card2 := {3, 4}
noncomputable def card3 := {5, 6}

theorem total_three_digit_numbers : 
  let possible_numbers := (card1.prod card2).prod card3 in
  possible_numbers.card = 40 := by
  sorry

end total_three_digit_numbers_l42_42702


namespace interval_of_monotonic_increase_transformed_function_correctness_max_min_g_on_interval_l42_42077

noncomputable def f (x : ℝ) : ℝ := -sqrt 3 * sin x * sin (x + π / 2) + cos x ^ 2 - 1 / 2
noncomputable def transformed_f (x : ℝ) : ℝ := cos (2 * x + π / 3)
noncomputable def g (x : ℝ) : ℝ := cos (x + π / 6)

theorem interval_of_monotonic_increase (k : ℤ) :
  ∀ x : ℝ, (k * π - 2 * π / 3 ≤ x ∧ x ≤ k * π - π / 6) → monotone f :=
sorry

theorem transformed_function_correctness :
  ∀ x : ℝ, f x = cos (2 * x + π / 3) :=
sorry

theorem max_min_g_on_interval :
  ∃ (x₁ x₂ : ℝ), (0 ≤ x₁ ∧ x₁ ≤ π ∧ g x₁ = sqrt 3 / 2) ∧ (0 ≤ x₂ ∧ x₂ ≤ π ∧ g x₂ = -1) :=
sorry

end interval_of_monotonic_increase_transformed_function_correctness_max_min_g_on_interval_l42_42077


namespace theater_total_cost_l42_42803

theorem theater_total_cost 
  (cost_orchestra : ℕ) (cost_balcony : ℕ)
  (total_tickets : ℕ) (ticket_difference : ℕ)
  (O B : ℕ)
  (h1 : cost_orchestra = 12)
  (h2 : cost_balcony = 8)
  (h3 : total_tickets = 360)
  (h4 : ticket_difference = 140)
  (h5 : O + B = total_tickets)
  (h6 : B = O + ticket_difference) :
  12 * O + 8 * B = 3320 :=
by
  sorry

end theater_total_cost_l42_42803


namespace arithmetic_mean_sqrt3_l42_42049

theorem arithmetic_mean_sqrt3 (a b : ℝ) (h1 : a = sqrt 3 + sqrt 2) (h2 : b = sqrt 3 - sqrt 2) : 
  (a + b) / 2 = sqrt 3 := 
by 
  sorry

end arithmetic_mean_sqrt3_l42_42049


namespace diagonal_passes_through_squares_l42_42703

-- Define the dimensions of the chessboard
def columns : ℕ := 1983
def rows : ℕ := 999

-- Assertion of the number of squares the diagonal passes through
theorem diagonal_passes_through_squares :
  let gcd_cols_rows := Nat.gcd columns rows in
  let intersections := columns + rows - gcd_cols_rows in
  let squares_crossed := intersections - 1 in
  squares_crossed = 2979 := 
by
  let gcd_cols_rows := Nat.gcd columns rows
  let intersections := columns + rows - gcd_cols_rows
  let squares_crossed := intersections - 1
  show squares_crossed = 2979 from sorry

end diagonal_passes_through_squares_l42_42703


namespace area_parallelogram_example_l42_42853

noncomputable def area_of_parallelogram (a b : ℝ) (theta : ℝ) : ℝ :=
  a * b * Real.sin theta

theorem area_parallelogram_example :
  area_of_parallelogram 12 22 (72 * Real.pi / 180) ≈ 251.0904 :=
by
  sorry

end area_parallelogram_example_l42_42853


namespace find_f_2017_l42_42919

noncomputable def f : ℕ → ℕ := sorry

axiom f_property (n : ℕ) : f(f(n)) + f(n) = 2n + 3
axiom f_at_0 : f(0) = 1
axiom f_at_2017 : f(2017) = 2018

theorem find_f_2017 : f(2017) = 2018 :=
 by exact f_at_2017

end find_f_2017_l42_42919


namespace problem1_problem2_l42_42263

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42263


namespace cubic_foot_to_cubic_inches_l42_42104

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end cubic_foot_to_cubic_inches_l42_42104


namespace find_original_function_l42_42962

theorem find_original_function
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₁ : ∀ x, g (x + (π / 3)) = sin (x - (π / 4)))
  (h₂ : ∀ x, f x = g (2 * x)) :
  f x = sin (x / 2 + π / 12) :=
sorry

end find_original_function_l42_42962


namespace compute_expression_l42_42613

variable {a b c : ℝ}
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

def x := b / c + c / b
def y := a / c + c / a
def z := a / b + b / a

theorem compute_expression : 2 * x^2 + 2 * y^2 + 2 * z^2 - x * y * z = 22 :=
by
  sorry

end compute_expression_l42_42613


namespace cubic_inches_in_one_cubic_foot_l42_42110

theorem cubic_inches_in_one_cubic_foot (h : 1.foot = 12.inches) : (1.foot)^3 = 1728 * (1.inches)^3 :=
by
  rw [h]
  calc (12.foot)^3 = 12^3 * (1.inches)^3 : sorry

end cubic_inches_in_one_cubic_foot_l42_42110


namespace tan_alpha_eq_l42_42517

variable (α : ℝ) (m : ℝ)

-- Given conditions
def conditions (α m : ℝ) : Prop :=
  sin α = m ∧ abs m < 1 ∧ (pi / 2 < α ∧ α < pi)

-- What needs to be proved
theorem tan_alpha_eq (α m : ℝ) (h : conditions α m) :
  tan α = - m / sqrt (1 - m^2) := 
  sorry

#check tan_alpha_eq

end tan_alpha_eq_l42_42517


namespace repeat_45_fraction_repeat_245_fraction_l42_42822

-- Define the repeating decimal 0.454545... == n / d
def repeating_45_equiv : Prop := ∃ n d : ℕ, (d ≠ 0) ∧ (0.45454545 = (n : ℚ) / (d : ℚ))

-- First problem statement: 0.4545... == 5 / 11
theorem repeat_45_fraction : 0.45454545 = (5 : ℚ) / (11 : ℚ) :=
by
  sorry

-- Define the repeating decimal 0.2454545... == n / d
def repeating_245_equiv : Prop := ∃ n d : ℕ, (d ≠ 0) ∧ (0.2454545 = (n : ℚ) / (d : ℚ))

-- Second problem statement: 0.2454545... == 27 / 110
theorem repeat_245_fraction : 0.2454545 = (27 : ℚ) / (110 : ℚ) :=
by
  sorry

end repeat_45_fraction_repeat_245_fraction_l42_42822


namespace equal_parts_not_rectangles_nor_triangles_l42_42021

theorem equal_parts_not_rectangles_nor_triangles :
  ∃ partition : list (set (ℝ × ℝ)),
    (∀ part ∈ partition, (measure_theory.measure_preserving (equiv.set_congr part)) ∧ (∀ part_shape, part_shape ≠ rect ∧ part_shape ≠ triangle)) ∧
    (∑ part ∈ partition, measure_theory.volume part = measure_theory.volume (set.univ : set (ℝ × ℝ))) ∧
    (∀ part ∈ partition, (∃ l u : ℝ, part.subset (set.Icc l u))) := by
  sorry

end equal_parts_not_rectangles_nor_triangles_l42_42021


namespace inversely_proportional_y_ratio_l42_42660

variable {k : ℝ}
variable {x₁ x₂ y₁ y₂ : ℝ}
variable (h_inv_prop : ∀ (x y : ℝ), x * y = k)
variable (hx₁x₂ : x₁ ≠ 0 ∧ x₂ ≠ 0)
variable (hy₁y₂ : y₁ ≠ 0 ∧ y₂ ≠ 0)
variable (hx_ratio : x₁ / x₂ = 3 / 4)

theorem inversely_proportional_y_ratio :
  y₁ / y₂ = 4 / 3 :=
by
  sorry

end inversely_proportional_y_ratio_l42_42660


namespace correct_quotient_given_incorrect_l42_42386

theorem correct_quotient_given_incorrect (dividend : ℕ) (incorrect_divisor : ℕ) (incorrect_quotient : ℕ) (correct_divisor : ℕ) :
  dividend = incorrect_divisor * incorrect_quotient →
  incorrect_divisor = 63 →
  incorrect_quotient = 24 →
  correct_divisor = 36 →
  dividend / correct_divisor = 42 :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  exact nat.div_eq_of_lt (by norm_num) (by norm_num),
}

end correct_quotient_given_incorrect_l42_42386


namespace find_parabola_equation_find_line_AB_equation_l42_42245

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42245


namespace JakeMowingEarnings_l42_42599

theorem JakeMowingEarnings :
  (∀ rate hours_mowing hours_planting (total_charge : ℝ),
      rate = 20 →
      hours_mowing = 1 →
      hours_planting = 2 →
      total_charge = 45 →
      (total_charge = hours_planting * rate + 5) →
      hours_mowing * rate = 20) :=
by
  intros rate hours_mowing hours_planting total_charge
  sorry

end JakeMowingEarnings_l42_42599


namespace graph_passes_through_point_l42_42675

noncomputable def exponential_shift (a : ℝ) (x : ℝ) := a^(x - 2)

theorem graph_passes_through_point (a : ℝ) (h : a > 0) (h1 : a ≠ 1) : exponential_shift a 2 = 1 :=
by
  unfold exponential_shift
  sorry

end graph_passes_through_point_l42_42675


namespace nonzero_terms_count_l42_42839

noncomputable def p (x : ℤ) : ℤ := 2 * x ^ 3 + 5 * x - 2
noncomputable def q (x : ℤ) : ℤ := 4 * x ^ 2 - x + 1
noncomputable def r (x : ℤ) : ℤ := 4 * (x ^ 3 - 3 * x ^ 2 + 2)

theorem nonzero_terms_count : ∀ (x : ℤ), 
  let expr := (p x * q x - r x)
  expr.degree = 5 ∧ expr.coeff 5 ≠ 0 ∧ expr.coeff 4 ≠ 0 ∧ expr.coeff 3 ≠ 0 ∧ expr.coeff 2 ≠ 0 ∧ expr.coeff 1 ≠ 0 ∧ expr.coeff 0 ≠ 0 :=
by 
  sorry

end nonzero_terms_count_l42_42839


namespace original_function_l42_42983

theorem original_function :
  ∃ f : ℝ → ℝ,
  (∀ x : ℝ, sin (x - π / 4) = f (2 * (x - π / 3))) →
  (∀ x : ℝ, f x = sin (x / 2 + π / 12)) :=
by
  sorry

end original_function_l42_42983


namespace sum_x1_x2_eq_five_l42_42560

theorem sum_x1_x2_eq_five {x1 x2 : ℝ} 
  (h1 : 2^x1 = 5 - x1)
  (h2 : x2 + Real.log x2 / Real.log 2 = 5) : 
  x1 + x2 = 5 := 
sorry

end sum_x1_x2_eq_five_l42_42560


namespace sum_of_digits_of_triangular_number_2010_l42_42807

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_triangular_number_2010 (N : ℕ)
  (h₁ : triangular_number N = 2010) :
  sum_of_digits N = 9 :=
sorry

end sum_of_digits_of_triangular_number_2010_l42_42807


namespace number_of_moles_of_NaCl_l42_42477

theorem number_of_moles_of_NaCl
  (moles_NaOH : ℕ)
  (moles_Cl2 : ℕ)
  (reaction : 2 * moles_NaOH + moles_Cl2 = 2 * moles_NaOH + 1) :
  2 * moles_Cl2 = 2 := by 
  sorry

end number_of_moles_of_NaCl_l42_42477


namespace continued_fraction_bound_l42_42325

theorem continued_fraction_bound
  (N : ℕ) (hN : N ≠ ∃ m : ℕ, m * m = N) 
  (a₀ : ℕ) (a : ℕ → ℕ) 
  (h_cf : ∃ n, ∀ i < n, (√N = [a₀, ⟨a₁, a₂, ..., a_n⟩ ])) 
  (h_a1 : a 1 ≠ 1) :
  ∀ i, a i ≤ 2 * a₀ := by
  sorry

end continued_fraction_bound_l42_42325


namespace inversion_of_point_l42_42019

-- Define the concept of inversion in a circle centered at O with radius R
def inversion_in_circle (O : Point) (R : ℝ) (A : Point) : Point := sorry

-- Define a circle with center O and radius R
structure Circle where
  center : Point
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the norm function for Point
def norm (P : Point) : ℝ :=
  sqrt (P.x^2 + P.y^2)

-- Proof statement for the problem
theorem inversion_of_point (O : Point) (R : ℝ) (A : Point) :
  let S := Circle.mk O R in
  let inversion_A := inversion_in_circle O R A in
  norm A ≠ 0 → norm inversion_A = R^2 / norm (O - A) := sorry

end inversion_of_point_l42_42019


namespace haley_albums_l42_42931

theorem haley_albums (total_pics : ℕ) (pics_in_one_album : ℕ) (pics_per_album_rest : ℕ) (albums_with_rest_pics : ℕ)
  (h1 : total_pics = 65)
  (h2 : pics_in_one_album = 17)
  (h3 : pics_per_album_rest = 8)
  (h4 : albums_with_rest_pics = (total_pics - pics_in_one_album) / pics_per_album_rest) :
  1 + albums_with_rest_pics = 7 :=
by
  -- assigning total_pics, pics_in_one_album, pics_per_album_rest to their values
  rw [h1, h2, h3] at h4
  -- proving division step
  have h5 : (65 - 17) / 8 = 6, by norm_num
  rw h5 at h4
  rw h4
  norm_num
  sorry

end haley_albums_l42_42931


namespace rational_root_count_l42_42796

open Polynomial

-- Define the polynomial with integer coefficients
def poly : Polynomial ℤ := 9 * X ^ 4 + a_3 * X ^ 3 + a_2 * X ^ 2 + a_1 * X + 15

-- State the theorem to prove the number of possible rational roots
theorem rational_root_count : (poly.roots_of_poly_with_integer_coefficients).length = 16 :=
by {
   -- Proof would go here
   sorry
}

end rational_root_count_l42_42796


namespace right_triangle_hypotenuse_l42_42677

noncomputable def log3_16 : ℝ := Real.log 16 / Real.log 3
noncomputable def log5_25 : ℝ := Real.log 25 / Real.log 5

theorem right_triangle_hypotenuse (h : ℝ) 
  (a := log3_16)
  (b := log5_25) 
  (hypotenuse := Real.sqrt (a^2 + b^2)) : 3^hypotenuse = 31.0024 := 
sorry

end right_triangle_hypotenuse_l42_42677


namespace angle_FAG_36_degrees_l42_42432

-- Definitions of relevant angles in the triangle and pentagon
def is_equilateral_triangle (A B C : Point) : Prop :=
  ∀ (P : Point), ∃ (Q R : Point), P = A ∨ P = B ∨ P = C ∧ ∠ A B C = 60

def is_regular_pentagon (B C D F G : Point) : Prop :=
  ∀ (P Q R S T : Point), P = B ∨ P = C ∨ P = D ∨ P = F ∨ P = G ∧ ∠ B C D = 108

variables {A B C D F G : Point}

theorem angle_FAG_36_degrees (h1: is_equilateral_triangle A B C)
(h2: is_regular_pentagon B C D F G)
(h3: side_shared BC ABC BCDGF) : ∠ FAG = 36 :=
sorry

end angle_FAG_36_degrees_l42_42432


namespace count_even_numbers_is_320_l42_42936

noncomputable def count_even_numbers_with_distinct_digits : Nat := 
  let unit_choices := 5  -- Choices for the unit digit (0, 2, 4, 6, 8)
  let hundreds_choices := 8  -- Choices for the hundreds digit (1 to 9, excluding the unit digit)
  let tens_choices := 8  -- Choices for the tens digit (0 to 9, excluding the hundreds and unit digit)
  unit_choices * hundreds_choices * tens_choices

theorem count_even_numbers_is_320 : count_even_numbers_with_distinct_digits = 320 := by
  sorry

end count_even_numbers_is_320_l42_42936


namespace M_inter_N_equals_l42_42088

namespace IntersectionProof

def M : set ℝ := {x | -1 < x ∧ x < 1}
def N : set ℝ := {x | ∃ y, y = real.sqrt (2 * x - 1)}

theorem M_inter_N_equals :
  (M ∩ N = {x | 1 / 2 ≤ x ∧ x < 1}) :=
by
  sorry

end IntersectionProof

end M_inter_N_equals_l42_42088


namespace elimination_method_equation_y_l42_42369

theorem elimination_method_equation_y (x y : ℝ)
    (h1 : 5 * x - 3 * y = -5)
    (h2 : 5 * x + 4 * y = -1) :
    7 * y = 4 :=
by
  -- Adding the required conditions as hypotheses and skipping the proof.
  sorry

end elimination_method_equation_y_l42_42369


namespace sqrt_cbrt_equiv_l42_42436

noncomputable def equiv_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  real.cbrt (x^2 * real.sqrt x)

theorem sqrt_cbrt_equiv (x : ℝ) (hx : 0 < x) : equiv_expression x hx = x^(5/6) :=
by
  sorry

end sqrt_cbrt_equiv_l42_42436


namespace Brandy_can_safely_drink_20_mg_more_l42_42678

variable (maximum_caffeine_per_day : ℕ := 500)
variable (caffeine_per_drink : ℕ := 120)
variable (number_of_drinks : ℕ := 4)
variable (caffeine_consumed : ℕ := caffeine_per_drink * number_of_drinks)

theorem Brandy_can_safely_drink_20_mg_more :
    caffeine_consumed = caffeine_per_drink * number_of_drinks →
    (maximum_caffeine_per_day - caffeine_consumed) = 20 :=
by
  intros h1
  rw [h1]
  sorry

end Brandy_can_safely_drink_20_mg_more_l42_42678


namespace speed_of_car_A_l42_42824

variables (V_A : ℝ) (T_A T_B V_B : ℝ) (D_A D_B : ℝ)

-- Given conditions:
def car_A_reaches_in_5_hours : Prop := T_A = 5
def car_B_speed : Prop := V_B = 100
def car_B_reaches_in_2_hours : Prop := T_B = 2
def distances_ratio : Prop := D_A / D_B = 2

-- Distances based on given conditions:
def distance_car_B : Prop := D_B = V_B * T_B
def distance_car_A : Prop := D_A = V_A * T_A

-- The conclusion to prove:
theorem speed_of_car_A
  (h1 : car_A_reaches_in_5_hours)
  (h2 : car_B_speed)
  (h3 : car_B_reaches_in_2_hours)
  (h4 : distances_ratio)
  (h5 : distance_car_B)
  (h6 : distance_car_A) :
  V_A = 80 := by
  sorry

end speed_of_car_A_l42_42824


namespace corn_pounds_l42_42832

theorem corn_pounds (c b : ℤ) (h1 : b + c = 20) (h2 : 75 * b + 99 * c = 1680) : c = 15 / 2 :=
by 
begin
  -- This is skipped.
  sorry
end

end corn_pounds_l42_42832


namespace radius_of_larger_circle_l42_42346

theorem radius_of_larger_circle (r : ℝ) (r_pos : r > 0)
    (ratio_condition : ∀ (rs : ℝ), rs = 3 * r)
    (diameter_condition : ∀ (ac : ℝ), ac = 6 * r)
    (chord_tangent_condition : ∀ (ab : ℝ), ab = 12) :
     (radius : ℝ) = 3 * r :=
by
  sorry

end radius_of_larger_circle_l42_42346


namespace largest_triangles_area_sum_ge_twice_polygon_area_l42_42744

variable {P : Type} [ConvexPolygon P] (b : ℕ → ℝ) (T : ℕ → ℝ)

def largest_triangle_area (b : ℕ → ℝ) (T : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, T i

noncomputable def polygon_area (P : Type) [ConvexPolygon P] : ℝ := sorry

theorem largest_triangles_area_sum_ge_twice_polygon_area (P : Type) [ConvexPolygon P] (n : ℕ) :
  largest_triangle_area b T n ≥ 2 * polygon_area P := sorry

end largest_triangles_area_sum_ge_twice_polygon_area_l42_42744


namespace total_pastries_and_bagels_l42_42356

theorem total_pastries_and_bagels (total_items bread_rolls croissants muffins cinnamon_rolls : ℕ)
  (H1 : total_items = 720)
  (H2 : bread_rolls = 240)
  (H3 : croissants = 75)
  (H4 : muffins = 145)
  (H5 : cinnamon_rolls = 110)
  (H6 : 2.5 * bread_rolls = 240) -- This condition simplifies to bread_rolls = 240, so it's essentially redundant with H2
  (H7 : 3 * bread_rolls / 5 : ℕ = 144) -- Fix the proportion based on bread rolls count
  : 75 + 145 + 110 + 144 = 474 := by
  -- This is the theorem statement to prove
  sorry

end total_pastries_and_bagels_l42_42356


namespace problem_1_problem_2_l42_42197

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42197


namespace rectangles_have_equal_diagonals_l42_42746

-- Define a structure for a rectangle
structure Rectangle (A B C D : Type) :=
  (is_rectangle : Prop)
  (equal_diagonals : A = C ∧ B = D)

-- Define a quadrilateral ABCD
variables (A B C D : Type)
variables (ABCD : Rectangle A B C D)

-- Define the theorem stating that the diagonals are equal, given that ABCD is a rectangle
theorem rectangles_have_equal_diagonals :
  (ABCD.is_rectangle → ABCD.equal_diagonals) :=
begin
  assume h,
  sorry
end

end rectangles_have_equal_diagonals_l42_42746


namespace sum_of_primes_count_l42_42585

/-- There are exactly four pairs of prime numbers whose sum is 58. -/
theorem sum_of_primes_count : 
  (finset.filter (λ p, nat.prime p ∧ nat.prime (58 - p)) (finset.range 59)).card = 4 := 
by sorry

end sum_of_primes_count_l42_42585


namespace octal_addition_correct_l42_42522

-- Define the octal addition problem
def octal_addition (a b : ℕ) (base : ℕ) : ℕ :=
  let sum := a + b
  sum

-- Given conditions
axiom decimal_base : ℕ := 10
axiom octal_base : ℕ := 8

-- The carryover rule in octal system properly manages overflow
axiom octal_carry_rule (x : ℕ) : x / octal_base = (x % octal_base)

-- Problem: Calculate 47_8 + 56_8 in octal system and prove it equals to 125_8
theorem octal_addition_correct : octal_addition 47 56 octal_base = 125 := 
  by sorry

end octal_addition_correct_l42_42522


namespace table_area_l42_42360

theorem table_area 
  (runner_area : ℝ)
  (cover_percentage : ℝ)
  (double_layer_area : ℝ)
  (triple_layer_area : ℝ)
  (total_area : ℝ)
  (H1 : runner_area = 204)
  (H2 : cover_percentage = 0.80)
  (H3 : double_layer_area = 24)
  (H4 : triple_layer_area = 20)
  (H5 : total_area = 175) :
  let single_layer_area := runner_area - 2 * double_layer_area - 3 * triple_layer_area,
      covered_area := single_layer_area + double_layer_area + triple_layer_area,
      table_area := covered_area / cover_percentage in
  table_area = total_area :=
by
  sorry

end table_area_l42_42360


namespace problem_1_problem_2_l42_42199

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42199


namespace part_one_part_two_l42_42494

variable (α : Real) (h : Real.tan α = 2)

theorem part_one (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6 / 11 := 
by
  sorry

theorem part_two (h : Real.tan α = 2) : 
  (1 / 4 * Real.sin α ^ 2 + 1 / 3 * Real.sin α * Real.cos α + 1 / 2 * Real.cos α ^ 2 + 1) = 43 / 30 := 
by
  sorry

end part_one_part_two_l42_42494


namespace factorize_problem_1_factorize_problem_2_l42_42471

-- Problem 1 Statement
theorem factorize_problem_1 (a m : ℝ) : 2 * a * m^2 - 8 * a = 2 * a * (m + 2) * (m - 2) := 
sorry

-- Problem 2 Statement
theorem factorize_problem_2 (x y : ℝ) : (x - y)^2 + 4 * (x * y) = (x + y)^2 := 
sorry

end factorize_problem_1_factorize_problem_2_l42_42471


namespace magnitude_of_c_is_correct_l42_42923

noncomputable def a : ℝ × ℝ := (2, 4)
noncomputable def b : ℝ × ℝ := (-1, 2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2
noncomputable def c : ℝ × ℝ := (a.1 - (dot_product a b) * b.1, a.2 - (dot_product a b) * b.2)

noncomputable def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt ((u.1 ^ 2) + (u.2 ^ 2))

theorem magnitude_of_c_is_correct :
  magnitude c = 8 * Real.sqrt 2 :=
by
  sorry

end magnitude_of_c_is_correct_l42_42923


namespace triangle_perimeter_l42_42858

def distance : (Real × Real) → (Real × Real) → Real :=
  λ ⟨x1, y1⟩ ⟨x2, y2⟩ => Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem triangle_perimeter : 
  let A := (3, 3)
  let B := (3, 10)
  let C := (8, 6)
  (distance A B) + (distance B C) + (distance C A) = 7 + Real.sqrt 41 + Real.sqrt 34 :=
by
  sorry

end triangle_perimeter_l42_42858


namespace TrapezoidTangentCircles_l42_42604

open EuclideanGeometry

variables {A B C D : Point}
variables {lineAB lineCD lineAD lineBC : Line}
variables {circleBC circleAD : Circle}

-- Definitions:
def trapezoid (AB CD : Line) (A B C D : Point) : Prop :=
  AB.is_parallel_to CD ∧ A ∈ AB ∧ B ∈ AB ∧ C ∈ CD ∧ D ∈ CD

def tangent_to_line (circle : Circle) (line : Line) : Prop :=
  ∃ P : Point, P ∈ circle ∧ P ∈ line ∧ P ∈ circle.tangent_line_at P

def midpoint (P Q R : Point) : Prop :=
  distance P R = distance R Q

-- Conditions:
def conditions (A B C D : Point) (lineAD lineBC : Line) (circleBC circleAD : Circle) : Prop :=
  trapezoid lineAB lineCD A B C D ∧
  circleBC = Circle.mk (LineSegment.mk B C) ∧
  tangent_to_line circleBC lineAD ∧
  circleAD = Circle.mk (LineSegment.mk A D)

-- Question:
def circleAD_tangent_to_lineBC (A B C D : Point) (lineAD lineBC : Line) (circleAD : Circle) : Prop :=
  tangent_to_line circleAD lineBC

theorem TrapezoidTangentCircles
  (A B C D : Point)
  (lineAB lineCD lineAD lineBC : Line)
  (circleBC circleAD : Circle)
  (h1: conditions A B C D lineAD lineBC circleBC circleAD) :
  circleAD_tangent_to_lineBC A B C D lineAD lineBC circleAD :=
sorry  -- proof goes here

end TrapezoidTangentCircles_l42_42604


namespace warehouse_capacity_l42_42409

theorem warehouse_capacity (total_bins : ℕ) (bins_20_tons : ℕ) (bins_15_tons : ℕ)
    (total_capacity : ℕ) (h1 : total_bins = 30) (h2 : bins_20_tons = 12) 
    (h3 : bins_15_tons = total_bins - bins_20_tons) 
    (h4 : total_capacity = (bins_20_tons * 20) + (bins_15_tons * 15)) : 
    total_capacity = 510 :=
by {
  sorry
}

end warehouse_capacity_l42_42409


namespace transformed_function_is_correct_l42_42970

noncomputable def f : ℝ → ℝ := λ x, sin (x / 2 + π / 12)

theorem transformed_function_is_correct :
  ∀ x : ℝ, sin (2 * (x - π / 3)) = sin (x - π / 4) :=
by
  intros x
  symmetry
  apply eq.symm
  calc
    sin (2 * (x - π / 3)) = sin (2x - 2 * π / 3) : by rw mul_sub
    _ = sin (x - π / 4) : sorry

end transformed_function_is_correct_l42_42970


namespace range_of_x_l42_42515

open Real

def p (x : ℝ) : Prop := log (x^2 - 2 * x - 2) ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4
def not_p (x : ℝ) : Prop := -1 < x ∧ x < 3
def not_q (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 4

theorem range_of_x (x : ℝ) :
  (¬ p x ∧ ¬ q x ∧ (p x ∨ q x)) →
  x ≤ -1 ∨ (0 < x ∧ x < 3) ∨ x ≥ 4 :=
sorry

end range_of_x_l42_42515


namespace find_parabola_equation_find_line_AB_equation_l42_42242

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42242


namespace largest_divisor_of_prime_squares_l42_42617

theorem largest_divisor_of_prime_squares (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : q < p) : 
  ∃ d : ℕ, ∀ p q : ℕ, Prime p → Prime q → q < p → d ∣ (p^2 - q^2) ∧ ∀ k : ℕ, (∀ p q : ℕ, Prime p → Prime q → q < p → k ∣ (p^2 - q^2)) → k ≤ d :=
by 
  use 2
  {
    sorry
  }

end largest_divisor_of_prime_squares_l42_42617


namespace dot_product_self_l42_42557

variable {ℝ} [InnerProductSpace ℝ]

theorem dot_product_self (w : ℝ^n) (norm_w : ‖w‖ = 5) : w ⬝ w = 25 := by
  sorry

end dot_product_self_l42_42557


namespace convert_cylindrical_to_rectangular_l42_42831

open Real

def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * cos θ, r * sin θ, z)

theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 3 (π / 4) (-2) = (3 * (sqrt 2 / 2), 3 * (sqrt 2 / 2), -2) :=
by
  -- Proof need to be provided
  sorry

end convert_cylindrical_to_rectangular_l42_42831


namespace probability_distribution_l42_42134

namespace StandardParts

-- Conditions
def totalParts := 10
def standardParts := 8
def selectedParts := 2

-- Hypergeometric distribution calculations
def P_X_0 := (Nat.choose 8 0 * Nat.choose 2 2) / Nat.choose 10 2
def P_X_1 := (Nat.choose 8 1 * Nat.choose 2 1) / Nat.choose 10 2
def P_X_2 := (Nat.choose 8 2 * Nat.choose 2 0) / Nat.choose 10 2

-- Proof
theorem probability_distribution :
  P_X_0 = 1 / 45 ∧ P_X_1 = 16 / 45 ∧ P_X_2 = 28 / 45 := by
  sorry

end StandardParts

end probability_distribution_l42_42134


namespace parallelogram_side_lengths_l42_42579

open Function

section ParallelogramSides

variables (ABCD: Type) [Parallelogram ABCD]
variables (A B C D : Point)
variables (K M : Point)
variables (hB K-BC : OnLineSegment K B C) -- Points K and M on BC
variables (hB M-BC : OnLineSegment M B C)
variables (hAngleBAM : IsAngleBisector A M B A) -- AM is the bisector of angle ∠BAD
variables (hAngleDKC : IsAngleBisector D K C D) -- DK is the bisector of angle ∠ADC
variables (hAB : Length A B = 3) -- AB = 3 cm
variables (hKM : Length K M = 2) -- KM = 2 cm

theorem parallelogram_side_lengths :
  (Length B C = 4 ∨ Length B C = 7) ∧ Length A B = 3 :=
begin
  sorry
end

end ParallelogramSides

end parallelogram_side_lengths_l42_42579


namespace point_P_on_number_line_l42_42299

variable (A : ℝ) (B : ℝ) (P : ℝ)

theorem point_P_on_number_line (hA : A = -1) (hB : B = 5) (hDist : abs (P - A) = abs (B - P)) : P = 2 := 
sorry

end point_P_on_number_line_l42_42299


namespace jake_sausages_cost_l42_42154

theorem jake_sausages_cost :
  let package_weight := 2
  let num_packages := 3
  let cost_per_pound := 4
  let total_weight := package_weight * num_packages
  let total_cost := total_weight * cost_per_pound
  total_cost = 24 := by
  sorry

end jake_sausages_cost_l42_42154


namespace part_one_part_two_l42_42231

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42231


namespace transformed_function_is_correct_l42_42971

noncomputable def f : ℝ → ℝ := λ x, sin (x / 2 + π / 12)

theorem transformed_function_is_correct :
  ∀ x : ℝ, sin (2 * (x - π / 3)) = sin (x - π / 4) :=
by
  intros x
  symmetry
  apply eq.symm
  calc
    sin (2 * (x - π / 3)) = sin (2x - 2 * π / 3) : by rw mul_sub
    _ = sin (x - π / 4) : sorry

end transformed_function_is_correct_l42_42971


namespace zach_needs_more_money_l42_42730

noncomputable def cost_of_bike : ℕ := 100
noncomputable def weekly_allowance : ℕ := 5
noncomputable def mowing_income : ℕ := 10
noncomputable def babysitting_rate_per_hour : ℕ := 7
noncomputable def initial_savings : ℕ := 65
noncomputable def hours_babysitting : ℕ := 2

theorem zach_needs_more_money : 
  cost_of_bike - (initial_savings + weekly_allowance + mowing_income + (babysitting_rate_per_hour * hours_babysitting)) = 6 :=
by
  sorry

end zach_needs_more_money_l42_42730


namespace part_one_part_two_l42_42876

-- Set definitions
def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

-- Intersection of B and C
def B_inter_C : Set ℕ := B ∩ C

-- Complement of B ∩ C in A
def neg_A_B_inter_C : Set ℤ := A \ (coe '' B_inter_C)

-- Proving the first part
theorem part_one : A ∪ (coe '' B_inter_C) = {x | -6 ≤ x ∧ x ≤ 6} :=
by {
  sorry
}

-- Proving the second part
theorem part_two : A ∩ neg_A_B_inter_C = {x | -6 ≤ x ∧ x ≤ 6 ∧ x ≠ 3} :=
by {
  sorry
}

end part_one_part_two_l42_42876


namespace integral_log_10_l42_42590

theorem integral_log_10 {a : ℝ} (h₁ : (∀ r : ℕ, r = 2 → ∑ k in finset.range (6), nat.choose 5 r * ((-1)^r) * 10 - 3 * r = 10))
  (h₂ : a = 10) :
  ∫ x in 1..a, x⁻¹ = real.log 10 :=
by
  sorry

end integral_log_10_l42_42590


namespace part1_eq_C_part2_line_AB_l42_42289

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42289


namespace amount_received_from_mom_l42_42862

-- Defining the problem conditions
def receives_from_dad : ℕ := 5
def spends : ℕ := 4
def has_more_from_mom_after_spending (M : ℕ) : Prop := 
  (receives_from_dad + M - spends = receives_from_dad + 2)

-- Lean theorem statement
theorem amount_received_from_mom (M : ℕ) (h : has_more_from_mom_after_spending M) : M = 6 := 
by
  sorry

end amount_received_from_mom_l42_42862


namespace nonnegative_fraction_interval_l42_42491

theorem nonnegative_fraction_interval : 
  ∀ x : ℝ, (0 ≤ x ∧ x < 3) ↔ (0 ≤ (x - 15 * x^2 + 36 * x^3) / (9 - x^3)) := by
sorry

end nonnegative_fraction_interval_l42_42491


namespace log_three_nine_l42_42006

theorem log_three_nine : log 3 9 = 2 :=
by sorry

end log_three_nine_l42_42006


namespace vector_addition_l42_42029

def v1 : ℝ × ℝ × ℝ := (4, -9, 2)
def v2 : ℝ × ℝ × ℝ := (-1, 16, 5)
def sum := (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

theorem vector_addition : sum = (3, 7, 7) :=
by
  sorry

end vector_addition_l42_42029


namespace roots_of_quadratic_l42_42037

theorem roots_of_quadratic {a b c : ℝ} (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  ∀ x, (x = a ∨ x = b ∨ x = c) ↔ x^2 - (a + b + c) * x + (a * b + b * c + c * a) = 0 :=
by
  sorry

end roots_of_quadratic_l42_42037


namespace sum_of_4_digit_numbers_with_remainder_2_div_13_l42_42375

theorem sum_of_4_digit_numbers_with_remainder_2_div_13 :
  (∑ n in finset.filter (λ x, x % 13 = 2) (finset.Icc 1000 9999), n) = 3813693 :=
by sorry

end sum_of_4_digit_numbers_with_remainder_2_div_13_l42_42375


namespace speed_first_part_proof_l42_42805

noncomputable def speed_first_part (x : ℝ) : ℝ :=
  let t1 := x / v  -- time for first part of journey
  let t2 := 2 * x / 20  -- time for second part of journey
  let t := 5 * x / 40  -- total time for the entire journey
  have h : t1 + t2 = t, from sorry  -- proof goes here
  v  -- solving v from the equations would be the following step

theorem speed_first_part_proof (x : ℝ) : speed_first_part x = 50 / 11 :=
sorry

end speed_first_part_proof_l42_42805


namespace number_of_figures_l42_42422

theorem number_of_figures (num_squares num_rectangles : ℕ) 
  (h1 : 8 * 8 / 4 = num_squares + num_rectangles) 
  (h2 : 2 * 54 + 4 * 8 = 8 * num_squares + 10 * num_rectangles) :
  num_squares = 10 ∧ num_rectangles = 6 :=
sorry

end number_of_figures_l42_42422


namespace find_function_l42_42991

open Real

theorem find_function (f : ℝ → ℝ) :
  let g := λ x, f (x / 2);
  let h := λ x, g (x - π / 3);
  (∀ x, h x = sin (x - π / 4)) →
  ∀ x, f x = sin (x / 2 + π / 12) :=
by
  intros g h h_eq x
  sorry

end find_function_l42_42991


namespace inradius_eq_common_difference_l42_42349

theorem inradius_eq_common_difference (a d : ℝ) (h_a : a > 0) (h_d : d > 0)
  (triangle_legs : (a-d) * (a-d) + a * a = (a+d) * (a+d)) :
  let s := (a-d + a + a+d) / 2 in
  let A := 1/2 * (a-d) * a in
  let r := A / s in
  r = d :=
by
  sorry

end inradius_eq_common_difference_l42_42349


namespace num_distinct_sums_of_two_positive_cubes_less_than_300_l42_42940

theorem num_distinct_sums_of_two_positive_cubes_less_than_300 :
  {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ n = a^3 + b^3 ∧ n < 300}.to_finset.card = 19 := 
by
  sorry

end num_distinct_sums_of_two_positive_cubes_less_than_300_l42_42940


namespace ad_eq_neg_one_l42_42071

theorem ad_eq_neg_one
  (a b c d : ℝ)
  (h1 : a * d = b * c)
  (h2 : ∀ x, ((ln (x + 2) - x) ≤ ln (b + 2) - b))
  (h3 : ln (b + 2) - b = c) :
  ad = -1 :=
by
  sorry

end ad_eq_neg_one_l42_42071


namespace range_of_values_for_a_l42_42910

-- Define the function
def f (a x : ℝ) := log a (2 * a - x)

-- Define the conditions
variables {a x : ℝ}
axiom increasing_on_interval (h : ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ < f a x₂)
axiom decreasing_y (h : 0 < x ∧ x < 1 → 2 * a - x > 0)

-- Final problem: Prove the correct range for a
theorem range_of_values_for_a : 0.5 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_values_for_a_l42_42910


namespace cubic_foot_to_cubic_inches_l42_42106

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end cubic_foot_to_cubic_inches_l42_42106


namespace largest_five_digit_congruent_to_31_modulo_26_l42_42715

theorem largest_five_digit_congruent_to_31_modulo_26 :
  ∃ x : ℕ, (10000 ≤ x ∧ x < 100000) ∧ x % 26 = 31 ∧ x = 99975 :=
by
  sorry

end largest_five_digit_congruent_to_31_modulo_26_l42_42715


namespace factorize_poly_l42_42726

theorem factorize_poly : 
  (x : ℤ) → (x^12 + x^6 + 1) = (x^4 - x^2 + 1) * (x^8 + x^4 + 1) :=
by
  sorry

end factorize_poly_l42_42726


namespace area_larger_semicircles_is_25_percent_greater_l42_42418

open Real

def rect_length := 12
def rect_width := 8

def radius_larger_semicircle := rect_length / 2
def radius_smaller_semicircle := rect_width / 2

def area_semicircle(radius: ℝ) := (1/2) * π * radius^2

def area_larger_semicircles := 2 * area_semicircle(radius_larger_semicircle)
def area_smaller_semicircles := 2 * area_semicircle(radius_smaller_semicircle)

def percent_increase (new: ℝ) (old: ℝ) := ((new - old) / old) * 100

theorem area_larger_semicircles_is_25_percent_greater :
  percent_increase area_larger_semicircles area_smaller_semicircles = 25 :=
by
  sorry

end area_larger_semicircles_is_25_percent_greater_l42_42418


namespace arrangement_seats_l42_42700

/-- A committee composed of seven women and three men sits in a row with the women in
indistinguishable rocking chairs and the men on indistinguishable stools. Prove that
the number of distinct ways to arrange the seats such that no two men sit next to each 
other is exactly 56. -/
theorem arrangement_seats (women men : ℕ) (indistinguishable_chairs indistinguishable_stools : ℕ) 
  (no_adjacent_men : Prop) : 
  women = 7 → men = 3 → indistinguishable_chairs = 7 → indistinguishable_stools = 3 →
  no_adjacent_men → combination (indistinguishable_chairs + 1) indistinguishable_stools = 56 :=
begin
  -- Given conditions
  assume hwomen : women = 7,
  assume hmen : men = 3,
  assume hchairs : indistinguishable_chairs = 7,
  assume hstools : indistinguishable_stools = 3,
  assume hno_adjacent : no_adjacent_men,
  -- Define the combination function
  let combination := λ (n k : ℕ), nat.choose n k,
  -- Prove the statement using the combination formula
  have hcomb : combination (7 + 1) 3 = 56, 
  { sorry },
  -- Conclude the proof
  exact hcomb,
end

end arrangement_seats_l42_42700


namespace root_expression_eq_l42_42612

theorem root_expression_eq (p q α β γ δ : ℝ) 
  (h1 : ∀ x, (x - α) * (x - β) = x^2 + p * x + 2)
  (h2 : ∀ x, (x - γ) * (x - δ) = x^2 + q * x + 2) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 4 + 2 * (p^2 - q^2) := 
sorry

end root_expression_eq_l42_42612


namespace correct_propositions_l42_42063

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def period_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

def symmetry_about_points (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x, f (x + k) = f (x - k)

theorem correct_propositions (h1: is_odd_function f) (h2 : ∀ x, f (x + 1) = f (x -1)) :
  period_2 f ∧ (∀ k : ℤ, symmetry_about_points f k) :=
by
  sorry

end correct_propositions_l42_42063


namespace initial_cake_pieces_l42_42492

-- Define the initial number of cake pieces
variable (X : ℝ)

-- Define the conditions as assumptions
def cake_conditions (X : ℝ) : Prop :=
  0.60 * X + 3 * 32 = X 

theorem initial_cake_pieces (X : ℝ) (h : cake_conditions X) : X = 240 := sorry

end initial_cake_pieces_l42_42492


namespace second_vote_difference_l42_42576

-- Define the total number of members
def total_members : ℕ := 300

-- Define the votes for and against in the initial vote
structure votes_initial :=
  (a : ℕ) (b : ℕ) (h : a + b = total_members) (rejected : b > a)

-- Define the votes for and against in the second vote
structure votes_second :=
  (a' : ℕ) (b' : ℕ) (h : a' + b' = total_members)

-- Define the margin and condition of passage by three times the margin
def margin (vi : votes_initial) : ℕ := vi.b - vi.a

def passage_by_margin (vi : votes_initial) (vs : votes_second) : Prop :=
  vs.a' - vs.b' = 3 * margin vi

-- Define the condition that a' is 7/6 times b
def proportion (vs : votes_second) (vi : votes_initial) : Prop :=
  vs.a' = (7 * vi.b) / 6

-- The final proof statement
theorem second_vote_difference (vi : votes_initial) (vs : votes_second)
  (h_margin : passage_by_margin vi vs)
  (h_proportion : proportion vs vi) :
  vs.a' - vi.a = 55 :=
by
  sorry  -- This is where the proof would go

end second_vote_difference_l42_42576


namespace find_a_and_b_find_set_A_l42_42056

noncomputable def f (x a b : ℝ) := 4 ^ x - a * 2 ^ x + b

theorem find_a_and_b (a b : ℝ)
  (h₁ : f 1 a b = -1)
  (h₂ : ∀ x, ∃ t > 0, f x a b = t ^ 2 - a * t + b) :
  a = 4 ∧ b = 3 :=
sorry

theorem find_set_A (a b : ℝ)
  (ha : a = 4) (hb : b = 3) :
  {x : ℝ | f x a b ≤ 35} = {x : ℝ | x ≤ 3} :=
sorry

end find_a_and_b_find_set_A_l42_42056


namespace mixed_oil_rate_l42_42121

theorem mixed_oil_rate
  (amt1 amt2 amt3 : ℕ)
  (price1 price2 price3 : ℕ)
  (total_cost total_volume : ℕ)
  (rate : real)
  (h1 : amt1 = 15)
  (h2 : price1 = 50)
  (h3 : amt2 = 8)
  (h4 : price2 = 75)
  (h5 : amt3 = 10)
  (h6 : price3 = 65)
  (h7 : total_cost = amt1 * price1 + amt2 * price2 + amt3 * price3)
  (h8 : total_volume = amt1 + amt2 + amt3)
  (h9 : rate = total_cost / total_volume) :
  rate ≈ 60.61 :=
by
  sorry

end mixed_oil_rate_l42_42121


namespace transformation_correct_l42_42980

noncomputable def original_function : ℝ → ℝ :=
  λ x, sin (x / 2 + π / 12)

theorem transformation_correct :
  ∀ (x : ℝ), (shift_right (shorten_abscissas f) (π / 3)) x = sin (x - π / 4) →
  f = original_function :=
by
  -- Assume the shifted and shortened function equals the given sine transformation
  intro x
  sorry

end transformation_correct_l42_42980


namespace set_intersection_nonempty_l42_42928

theorem set_intersection_nonempty {a : ℕ} (h : ({0, a} ∩ {1, 2} : Set ℕ) ≠ ∅) :
  a = 1 ∨ a = 2 := by
  sorry

end set_intersection_nonempty_l42_42928


namespace points_lie_on_circle_l42_42183

-- Define the geometrical entities
variables {A B C : Type} [Nonempty α] [MetricSpace α] [T2Space α]

-- Assume definitions for Thales' circles, altitudes, and points
def is_thales_circle (k : set α) (x y : α) : Prop := sorry
def is_altitude (m : set α) (from to : α) : Prop := sorry
def is_tangent (tangent_point : α) (circle_center : α) (circle : set α) : Prop := sorry

axiom A B C : α
axiom k₁ k₂ k₃ : set α
axiom m₁ m₂ m₃ : set α
axiom B₁ B₂ C₁ C₂ E : α

-- Given conditions
variable (h_Thales₁ : is_thales_circle k₁ B C)
variable (h_Thales₂ : is_thales_circle k₂ C A)
variable (h_Thales₃ : is_thales_circle k₃ A B)

variable (h_altitude₂ : is_altitude m₂ B C)
variable (h_altitude₃ : is_altitude m₃ C A)

variable (h_B₁_in_m₂ : B₁ ∈ m₂)
variable (h_B₂_in_m₂ : B₂ ∈ m₂)
variable (h_B₁_in_k₂ : B₁ ∈ k₂)
variable (h_B₂_in_k₂ : B₂ ∈ k₂)

variable (h_C₁_in_m₃ : C₁ ∈ m₃)
variable (h_C₂_in_m₃ : C₂ ∈ m₃)
variable (h_C₁_in_k₃ : C₁ ∈ k₃)
variable (h_C₂_in_k₃ : C₂ ∈ k₃)

variable (h_tangent_A_k₁_E : is_tangent E A k₁)

-- The proof goal
theorem points_lie_on_circle :
  dist A B₁ = dist A B₂ ∧
  dist A B₂ = dist A C₁ ∧
  dist A C₁ = dist A C₂ ∧
  dist A C₂ = dist A E :=
sorry

end points_lie_on_circle_l42_42183


namespace points_on_same_side_parabola_1_points_on_same_side_parabola_3_points_on_same_side_parabola_5_l42_42814

def parabola_1 (x : ℝ) : ℝ :=
  2 * x^2 + 4 * x

def parabola_2 (x : ℝ) : ℝ :=
  (x^2 / 2) - x - (3 / 2)

def parabola_3 (x : ℝ) : ℝ :=
  -x^2 + 2 * x - 1

def parabola_4 (x : ℝ) : ℝ :=
  -x^2 - 4 * x - 3

def parabola_5 (x : ℝ) : ℝ :=
  -x^2 + 3

def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (0, 2)

theorem points_on_same_side_parabola_1 : 
  (parabola_1 A.1 < A.2) ∧ (parabola_1 B.1 < B.2) ∨ (parabola_1 A.1 > A.2) ∧ (parabola_1 B.1 > B.2) := 
sorry

theorem points_on_same_side_parabola_3 : 
  (parabola_3 A.1 < A.2) ∧ (parabola_3 B.1 < B.2) ∨ (parabola_3 A.1 > A.2) ∧ (parabola_3 B.1 > B.2) := 
sorry

theorem points_on_same_side_parabola_5 : 
  (parabola_5 A.1 < A.2) ∧ (parabola_5 B.1 < B.2) ∨ (parabola_5 A.1 > A.2) ∧ (parabola_5 B.1 > B.2) := 
sorry

end points_on_same_side_parabola_1_points_on_same_side_parabola_3_points_on_same_side_parabola_5_l42_42814


namespace slope_range_l42_42893

noncomputable def point := (ℝ × ℝ)

def A : point := (-2, -1)
def B : point := (2, -3)
def P : point := (1, 5)

def slope (P1 P2 : point) : ℝ := (P2.2 - P1.2) / (P2.1 - P1.1)

def line_intersects_segment (P1 P2 L1 L2 : point) : Prop :=
  ∃ k, (P1.2 + k * (P2.2 - P1.2) = (P1.2 + k * (P2.2 - P1.2)) ∧ 
        P1.1 + k * (P2.1 - P1.1) = (P1.1 + k * (P2.1 - P1.1)))

theorem slope_range :
  (line_intersects_segment A B P P) →
  (slope P A ≤ -8 ∨ slope P A ≥ 2 ∨ slope P B ≤ -8 ∨ slope P B ≥ 2) :=
by
  sorry

end slope_range_l42_42893


namespace intervals_of_monotonicity_max_min_values_l42_42908

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 - 2

theorem intervals_of_monotonicity :
  (∀ x: ℝ, x < 0 → deriv f x < 0) ∧
  (∀ x: ℝ, 0 < x ∧ x < 2 → deriv f x > 0) ∧
  (∀ x: ℝ, x > 2 → deriv f x < 0) :=
sorry

theorem max_min_values :
  ∃ c d: ℝ, (c = Sup (set_of(continuous_on f (set.Icc (-2) 2))) ∧
              d = Inf (set_of(continuous_on f (set.Icc (-2) 2))) ∧
              c = 18 ∧ d = -2) :=
sorry

end intervals_of_monotonicity_max_min_values_l42_42908


namespace cos_75_deg_l42_42450

theorem cos_75_deg : Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_deg_l42_42450


namespace wolf_nobel_laureates_l42_42395

/-- 31 scientists that attended a certain workshop were Wolf Prize laureates,
and some of them were also Nobel Prize laureates. Of the scientists who attended
that workshop and had not received the Wolf Prize, the number of scientists who had
received the Nobel Prize was 3 more than the number of scientists who had not received
the Nobel Prize. In total, 50 scientists attended that workshop, and 25 of them were
Nobel Prize laureates. Prove that the number of Wolf Prize laureates who were also
Nobel Prize laureates is 3. -/
theorem wolf_nobel_laureates (W N total W' N' W_N : ℕ)  
  (hW : W = 31) (hN : N = 25) (htotal : total = 50) 
  (hW' : W' = total - W) (hN' : N' = total - N) 
  (hcondition : N' - W' = 3) :
  W_N = N - W' :=
by
  sorry

end wolf_nobel_laureates_l42_42395


namespace mean_score_approx_l42_42488

theorem mean_score_approx (mu sigma : ℝ) :
  (42 = mu - 5 * sigma) → 
  (67 = mu + 2.5 * sigma) → 
  mu ≈ 58.67 :=
by
  intros h1 h2
  sorry -- Proof goes here

end mean_score_approx_l42_42488


namespace bons_wins_probability_l42_42850

theorem bons_wins_probability : 
  let p : ℚ := 
    let not_six := 5 / 6 
    let six := 1 / 6 
    by calc
      p = (not_six * six) + (not_six * not_six * p) : by sorry
       ... = 5 / 11 : by sorry
  ) in
  p = 5 / 11 :=
sorry

end bons_wins_probability_l42_42850


namespace inverse_at_neg_two_l42_42918

def g (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_at_neg_two :
  g (-2) = -43 :=
by
  -- sorry here to skip the proof, as instructed.
  sorry

end inverse_at_neg_two_l42_42918


namespace savings_calculation_l42_42736

-- Define the conditions as given in the problem
def income_expenditure_ratio (income expenditure : ℝ) : Prop :=
  ∃ x : ℝ, income = 10 * x ∧ expenditure = 4 * x

def income_value : ℝ := 19000

-- The final statement for the savings, where we will prove the above question == answer
theorem savings_calculation (income expenditure savings : ℝ)
  (h_ratio : income_expenditure_ratio income expenditure)
  (h_income : income = income_value) : savings = 11400 :=
by
  sorry

end savings_calculation_l42_42736


namespace total_marks_l42_42740

-- Define the conditions
def average_marks : ℝ := 35
def number_of_candidates : ℕ := 120

-- Define the total marks as a goal to prove
theorem total_marks : number_of_candidates * average_marks = 4200 :=
by
  sorry

end total_marks_l42_42740


namespace smallest_divisor_of_7614_l42_42687

theorem smallest_divisor_of_7614 (h : Nat) (H_h_eq : h = 1) (n : Nat) (H_n_eq : n = (7600 + 10 * h + 4)) :
  ∃ d, d > 1 ∧ d ∣ n ∧ ∀ x, x > 1 ∧ x ∣ n → d ≤ x :=
by
  sorry

end smallest_divisor_of_7614_l42_42687


namespace numerator_divisible_by_prime_l42_42307

theorem numerator_divisible_by_prime (p : ℕ) (hp : prime p) (hp_gt : p > 2) :
  let m := Nat.num (∑ k in finset.range (p - 1), (1 / k : ℚ))
  p ∣ m :=
sorry

end numerator_divisible_by_prime_l42_42307


namespace intersection_A_compl_B_l42_42066

open Set

noncomputable def R : Set ℝ := univ

def A : Set ℕ := {1, 2, 3, 4, 5}

def B : Set ℝ := {x : ℝ | x * (4 - x) < 0}

def compl_B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

theorem intersection_A_compl_B :
  (A ∩ (compl_B : Set ℝ)) = ({1, 2, 3, 4} : Set ℕ) :=
sorry

end intersection_A_compl_B_l42_42066


namespace induction_formula_incremental_step_l42_42709

open Nat

theorem induction_formula (n : ℕ) : (∏ i in range n, (n + 1 + i)) = 2^n * ∏ i in range n, (2*i + 1) := sorry

theorem incremental_step (k : ℕ) :
  (((∏ i in range (k+1), (k + 2 + i))
  ) - (∏ i in range k, (k + 1 + i))) = (∏ i in range k, (k + 1 + i)) * (4*k + 1) := sorry

end induction_formula_incremental_step_l42_42709


namespace greatest_score_of_individual_player_l42_42769

theorem greatest_score_of_individual_player
  (n : ℕ) (total_score : ℕ) (p : ℕ)
  (h_n : n = 12) (h_total_score : total_score = 100) (h_min_score : p ≥ 7)
  (h_team_scores : ∀ (scores : vector ℕ n), (∀ i, scores.nth i ≥ p) → vector.sum scores = total_score) :
  ∃ (max_p : ℕ), max_p ≤ total_score ∧ (∀ (scores : vector ℕ n), (∀ i, scores.nth i ≥ p) → vector.sum scores = total_score → max_p = max (vector.to_list scores)) ∧ max_p = 23 :=
by
  sorry

end greatest_score_of_individual_player_l42_42769


namespace part_one_part_two_l42_42237

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42237


namespace transformation_correct_l42_42979

noncomputable def original_function : ℝ → ℝ :=
  λ x, sin (x / 2 + π / 12)

theorem transformation_correct :
  ∀ (x : ℝ), (shift_right (shorten_abscissas f) (π / 3)) x = sin (x - π / 4) →
  f = original_function :=
by
  -- Assume the shifted and shortened function equals the given sine transformation
  intro x
  sorry

end transformation_correct_l42_42979


namespace find_function_l42_42996

open Real

theorem find_function (f : ℝ → ℝ) :
  let g := λ x, f (x / 2);
  let h := λ x, g (x - π / 3);
  (∀ x, h x = sin (x - π / 4)) →
  ∀ x, f x = sin (x / 2 + π / 12) :=
by
  intros g h h_eq x
  sorry

end find_function_l42_42996


namespace caffeine_safe_amount_l42_42680

theorem caffeine_safe_amount (max_caffeine : ℕ) (caffeine_per_drink : ℕ) (num_drinks : ℕ) :
    max_caffeine = 500 →
    caffeine_per_drink = 120 →
    num_drinks = 4 →
    max_caffeine - (caffeine_per_drink * num_drinks) = 20 :=
by
  intros h_max h_caffeine h_num
  rw [h_max, h_caffeine, h_num]
  norm_num
  sorry

end caffeine_safe_amount_l42_42680


namespace original_function_l42_42956

def f (x : ℝ) : ℝ := 
sorry

theorem original_function :
  (∀ x, (sin (x - π/4) = sin ((2 * (x - π / 3))))) ↔ (∀ x, f x = sin (x/2 + π/12)) :=
sorry

end original_function_l42_42956


namespace sales_function_maximize_profit_l42_42845

def y (x : ℝ) : ℝ := 300 - 10 * (x - 44)

theorem sales_function (x : ℝ) (h₁ : x ≥ 44) (h₂ : x ≤ 52) :
  y x = -10 * x + 740 :=
by 
  have : y x = 300 - 10 * (x - 44) := rfl
  rw [this, sub_mul, add_sub_cancel]
  norm_num

def profit (x : ℝ) : ℝ := (x - 40) * (300 - 10 * (x - 44))

theorem maximize_profit (x : ℝ) (h₁ : x = 52) :
  profit x = 2640 :=
by
  unfold profit
  rw [h₁, sub_self, zero_mul, add_zero, mul_comm, ← mul_assoc, mul_left_comm]
  norm_num

end sales_function_maximize_profit_l42_42845


namespace parabola_equation_line_AB_equation_l42_42191

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42191


namespace quadratic_inequality_solution_l42_42350

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 3 * x - 4 < 0 } = { x : ℝ | -4 < x ∧ x < 1 } :=
by
  sorry

end quadratic_inequality_solution_l42_42350


namespace num_sum_of_cubes_lt_300_l42_42939

def is_cube (n : ℤ) : Prop := ∃ k : ℕ, k^3 = n

def valid_cubes (n : ℤ) : Prop := 
  ∃ a b : ℕ, 1 ≤ a ∧ 1 ≤ b ∧ a ≤ 6 ∧ b ≤ 6 ∧ n = a^3 + b^3

theorem num_sum_of_cubes_lt_300 : 
  ( finset.filter (λ n, valid_cubes n) (finset.range 300)).card = 19 := 
by {
  sorry
}

end num_sum_of_cubes_lt_300_l42_42939


namespace volume_removed_tetrahedra_l42_42455

theorem volume_removed_tetrahedra (cube_edge_length : ℝ) (total_removed_volume : ℝ) :
  cube_edge_length = 2 ∧ (∃ hexadecagon : ℕ, hexadecagon = 16) → total_removed_volume = -64 * real.sqrt 2 :=
begin
  intros h,
  sorry
end

end volume_removed_tetrahedra_l42_42455


namespace divides_b_n_minus_n_l42_42841

theorem divides_b_n_minus_n (a b : ℕ) (h_a : a > 0) (h_b : b > 0) :
  ∃ n : ℕ, n > 0 ∧ a ∣ (b^n - n) :=
by
  sorry

end divides_b_n_minus_n_l42_42841


namespace parabola_equation_line_AB_equation_l42_42184

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42184


namespace shorter_piece_length_correct_l42_42396

noncomputable def shorter_piece_length (total_length : ℝ) (ratio : ℝ) : ℝ := 
  total_length * ratio / (ratio + 1)

theorem shorter_piece_length_correct :
  shorter_piece_length 57.134 (3.25678 / 7.81945) = 16.790 :=
by
  sorry

end shorter_piece_length_correct_l42_42396


namespace triangle_AB_length_l42_42596

theorem triangle_AB_length (A B C : Type) [inst : real_triangle A B C]
  (right_angle_A : ∠A = π / 2)
  (BC_length : segment_length B C = 10)
  (tan_C_eq_3cos_B : tan (∠C) = 3 * cos (∠B)) :
  segment_length A B = 20 * real.sqrt 2 / 3 := sorry

end triangle_AB_length_l42_42596


namespace garden_area_increase_l42_42397

noncomputable def original_garden_length : ℝ := 60
noncomputable def original_garden_width : ℝ := 20
noncomputable def original_garden_area : ℝ := original_garden_length * original_garden_width
noncomputable def original_garden_perimeter : ℝ := 2 * (original_garden_length + original_garden_width)

noncomputable def circle_radius : ℝ := original_garden_perimeter / (2 * Real.pi)
noncomputable def circle_area : ℝ := Real.pi * (circle_radius ^ 2)

noncomputable def area_increase : ℝ := circle_area - original_garden_area

theorem garden_area_increase :
  area_increase = (6400 / Real.pi) - 1200 :=
by 
  sorry -- proof goes here

end garden_area_increase_l42_42397


namespace dawn_bananas_l42_42833

-- Definitions of the given conditions
def total_bananas : ℕ := 200
def lydia_bananas : ℕ := 60
def donna_bananas : ℕ := 40

-- Proof that Dawn has 100 bananas
theorem dawn_bananas : (total_bananas - donna_bananas) - lydia_bananas = 100 := by
  sorry

end dawn_bananas_l42_42833


namespace f_minimum_value_l42_42539

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3/2 * x), Real.sin (3/2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)

noncomputable def f (x t : ℝ) : ℝ :=
  dot_product (a x) (b x) - 2 * t * magnitude ⟨a x + b x⟩

noncomputable def g (t : ℝ) : ℝ :=
  if t < 0 then -1
  else if t ≤ 1 then -1 - 2 * t^2
  else 1 - 4 * t

theorem f_minimum_value (x t : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  ∃ y, f x t = g t :=
sorry

end f_minimum_value_l42_42539


namespace equation_of_parabola_equation_of_line_AB_l42_42255

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42255


namespace original_function_l42_42955

def f (x : ℝ) : ℝ := 
sorry

theorem original_function :
  (∀ x, (sin (x - π/4) = sin ((2 * (x - π / 3))))) ↔ (∀ x, f x = sin (x/2 + π/12)) :=
sorry

end original_function_l42_42955


namespace fraction_area_above_line_l42_42410

theorem fraction_area_above_line : 
    let square_vertices := [(2,1), (7,1), (7,6), (2,6)]
    let line_points := [(2,1), (7,3)]
    (∃ v in square_vertices, (∀ i < 4, v = square_vertices[i])) ∧ 
    (∃ p in line_points, (∀ j < 2, p = line_points[j])) →
    (fractional_area_above_line square_vertices line_points = 4 / 5) :=
begin
    sorry
end

end fraction_area_above_line_l42_42410


namespace part1_eq_C_part2_line_AB_l42_42292

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42292


namespace monomial_2023rd_l42_42629

theorem monomial_2023rd : ∀ (x : ℝ), (2 * 2023 + 1) / 2023 * x ^ 2023 = (4047 / 2023) * x ^ 2023 :=
by
  intro x
  sorry

end monomial_2023rd_l42_42629


namespace probability_of_drawing_3_one_color_and_1_another_l42_42775

open Nat

def combinations (n k : ℕ) : ℕ := nat.choose n k

theorem probability_of_drawing_3_one_color_and_1_another : 
  let total_balls := 12 + 8
  let choose_4_from_total := combinations total_balls 4
  let choose_3_black_1_white := combinations 12 3 * combinations 8 1
  let choose_1_black_3_white := combinations 12 1 * combinations 8 3
  let favorable_outcomes := choose_3_black_1_white + choose_1_black_3_white
  let numerator := favorable_outcomes
  let denominator := choose_4_from_total
  numerator / denominator = 1 / 3 := 
by 
  sorry

end probability_of_drawing_3_one_color_and_1_another_l42_42775


namespace number_of_solutions_abs_eq_l42_42344

theorem number_of_solutions_abs_eq (f : ℝ → ℝ) (g : ℝ → ℝ) : 
  (∀ x : ℝ, f x = |3 * x| ∧ g x = |x - 2| ∧ (f x + g x = 4) → 
  ∃! x1 x2 : ℝ, 
    ((0 < x1 ∧ x1 < 2 ∧ f x1 + g x1 = 4 ) ∨ 
    (x2 < 0 ∧ f x2 + g x2 = 4) ∧ x1 ≠ x2)) :=
by
  sorry

end number_of_solutions_abs_eq_l42_42344


namespace find_lambda_find_BC_coordinates_find_A_coordinates_l42_42899

noncomputable section

-- Define the vectors e1 and e2
variables (e1 e2 : ℝ × ℝ)
variables (e1_ne_zero : e1 ≠ (0, 0)) (e2_ne_zero : e2 ≠ (0, 0))
variables (e1_ne_e2_collinear : e1 ≠ e2 ∧ e1 ≠ (-e2))

-- Define the given vector equations
def vec_AB := (2 : ℝ) • e1 + e2
def vec_BE (λ : ℝ) := (-e1) + λ • e2
def vec_EC := (-2 : ℝ) • e1 + e2

-- Collinearity condition
variables (A E C : ℝ × ℝ)
variables (vec_AE_collinear : ∃ k : ℝ, (2 : ℝ) • e1 + e2 + ((-e1) + λ • e2) = k • ((-2 : ℝ) • e1 + e2))

-- Step (1): Prove that λ = -3/2
theorem find_lambda (λ : ℝ) (k : ℝ) :
  (∃ k : ℝ, (2 : ℝ) • e1 + e2 + (-e1) + λ • e2 = k • ((-2 : ℝ) • e1 + e2)) →
  λ = -3 / 2 :=
sorry

-- Step (2): Given coordinates of e1 and e2, find vec_BC
def e1_val := (2, 1) : ℝ × ℝ
def e2_val := (2, -2) : ℝ × ℝ

-- Define the new vector BC based on the coordinates of e1 and e2
def vec_BC := (-3 : ℝ) • e1_val - (1 / 2 : ℝ) • e2_val

-- Prove the coordinates of BC
theorem find_BC_coordinates :
  vec_BC = (-7, -2) :=
sorry

-- Step (3): Given point D and vec_BC, find the coordinates of A
variables (D : ℝ × ℝ) (D_val : D = (3, 5))
def vec_AD (A : ℝ × ℝ) := (D.1 - A.1, D.2 - A.2)

-- Prove the coordinates of A
theorem find_A_coordinates (A : ℝ × ℝ) :
  vec_AD A = (-7, -2) → A = (10, 7) :=
sorry

end find_lambda_find_BC_coordinates_find_A_coordinates_l42_42899


namespace geometric_sequence_n_value_l42_42136

theorem geometric_sequence_n_value
  (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 * a 2 * a 3 = 4)
  (h2 : a 4 * a 5 * a 6 = 12)
  (h3 : a (n-1) * a n * a (n+1) = 324)
  (h_geometric : ∃ r > 0, ∀ i, a (i+1) = a i * r) :
  n = 14 :=
sorry

end geometric_sequence_n_value_l42_42136


namespace abs_diff_p_q_l42_42618

theorem abs_diff_p_q (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 :=
by 
  sorry

end abs_diff_p_q_l42_42618


namespace cosine_product_identity_l42_42309

theorem cosine_product_identity (n : ℕ) (α : ℝ) : 
  (∀ α, sin α ≠ 0) → 
  (cos α * cos (2 * α) * cos (4 * α) * ... * cos (2^n * α) = sin (2^(n+1) * α) / (2^(n+1) * sin α)) :=
by sorry

end cosine_product_identity_l42_42309


namespace part1_eq_C_part2_line_AB_l42_42293

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42293


namespace range_of_a_l42_42519

def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

theorem range_of_a (a : ℝ) :
  (∀ x ≤ -1, deriv (λ x, f x a) x ≥ 0) ↔ a ≤ 3 :=
by
  sorry

end range_of_a_l42_42519


namespace mary_money_left_l42_42297

theorem mary_money_left (q : ℝ) : 
  let drinks_cost := 2 * q in
  let small_pizzas_cost := 2 * q in
  let large_pizza_cost := 4 * q in
  let total_spent := drinks_cost + small_pizzas_cost + large_pizza_cost in
  let initial_money := 50 in
  initial_money - total_spent = 50 - 8 * q :=
by
  sorry

end mary_money_left_l42_42297


namespace problem_1_problem_2_l42_42200

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42200


namespace construct_focus_of_parabola_l42_42888

theorem construct_focus_of_parabola
  (P : Type) [real_plane P]
  (parabola : P → Prop)
  (RaysFocus : ∀ {L X F : P}, (L || axis_parabola_par || ray_intersect_at_parabola : ∀ {M}, Line M → Line XF)
  (ParallelChordsAxis : ∀ {C1 C2 : chord P}, ParallelChordMidpoints → ∃ M1 M2 : P, 
    (ChordMidpoint C1 P) ∧ (ChordMidpoint C2 P) ∧ (M1 == M2 || axis_parabola_par || (ParallelChords : ∃ C1 C2 : Parallels)
  : ∃ (focus : P), parabola focus :=
begin
  sorry
end

end construct_focus_of_parabola_l42_42888


namespace find_a2_a4_a9_l42_42509

open Nat -- Natural numbers
open Function -- Functions

variables {a : ℕ → ℤ} -- Define the sequence a_n as a function from natural numbers to integers

-- Definition of an arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n, a (n + 1) = a n + d

-- Definition of the sum of the first n terms
def sum_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, a i

-- Given the initial conditions 
variable (h_arith_seq : arithmetic_seq a) -- a_n is an arithmetic sequence
variable (h_sum_9 : sum_n_terms a 9 = 54) -- S_9 = 54

-- Theorem to prove
theorem find_a2_a4_a9 : a 2 + a 4 + a 9 = 18 :=
by
  sorry

end find_a2_a4_a9_l42_42509


namespace find_f_of_half_l42_42614

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition_1 : f 1 = -1

axiom f_condition_2 : ∀ x y : ℝ, f (xy + f x) = x * f y + f x

theorem find_f_of_half : f (1 / 2) = -2 := by
  sorry

end find_f_of_half_l42_42614


namespace find_original_function_l42_42965

theorem find_original_function
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₁ : ∀ x, g (x + (π / 3)) = sin (x - (π / 4)))
  (h₂ : ∀ x, f x = g (2 * x)) :
  f x = sin (x / 2 + π / 12) :=
sorry

end find_original_function_l42_42965


namespace f_is_even_if_g_is_odd_l42_42616

variable (g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) :=
  ∀ x : ℝ, h (-x) = -h x

def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = f x

theorem f_is_even_if_g_is_odd (hg : is_odd g) :
  is_even (fun x => |g (x^4)|) :=
by
  sorry

end f_is_even_if_g_is_odd_l42_42616


namespace number_of_chips_per_day_l42_42625

def total_chips : ℕ := 100
def chips_first_day : ℕ := 10
def total_days : ℕ := 10
def days_remaining : ℕ := total_days - 1
def chips_remaining : ℕ := total_chips - chips_first_day

theorem number_of_chips_per_day : 
  chips_remaining / days_remaining = 10 :=
by 
  unfold chips_remaining days_remaining total_chips chips_first_day total_days
  sorry

end number_of_chips_per_day_l42_42625


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42281

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42281


namespace truck_capacity_is_correct_l42_42324

-- Definitions of conditions
def rate_per_person : ℕ := 250
def initial_workers : ℕ := 2
def additional_workers : ℕ := 6
def initial_hours : ℕ := 4
def total_hours : ℕ := 6

-- Definition to calculate the capacity of the truck
def truck_capacity (rate_per_person initial_workers additional_workers initial_hours total_hours : ℕ) : ℕ :=
  let rate_initial := initial_workers * rate_per_person in
  let blocks_initial := initial_hours * rate_initial in
  let total_workers := initial_workers + additional_workers in
  let rate_total := total_workers * rate_per_person in
  let remaining_hours := total_hours - initial_hours in
  let blocks_remaining := remaining_hours * rate_total in
  blocks_initial + blocks_remaining

-- Statement to be proven
theorem truck_capacity_is_correct :
  truck_capacity rate_per_person initial_workers additional_workers initial_hours total_hours = 6000 := 
sorry

end truck_capacity_is_correct_l42_42324


namespace number_of_valid_pairs_l42_42016

theorem number_of_valid_pairs :
  (∃ n, 2^2876 < 5^n ∧ 5^n < 2^2878) →
  (∃ v, v = 1234 ∧ 
    (∀ m:ℕ, 1 ≤ m ∧ m ≤ 2875 → 
      (∃ n, 5^n < 2^m ∧ 2^m < 2^(m+1) ∧ 2^(m+1) < 5^(n+1)))) :=
begin
  sorry
end

end number_of_valid_pairs_l42_42016


namespace problem_conditions_l42_42673

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then x / (1 + x)
else if -1 < x ∧ x < 0 then x / (1 - x)
else 0

theorem problem_conditions (a b : ℝ) (x : ℝ) :
  (∀ x : ℝ, -1 < x → x < 1 → f (-x) = -f x) ∧ 
  (∀ x : ℝ, 0 ≤ x → x < 1 → f x = (-a * x - b) / (1 + x)) ∧ 
  (f (1 / 2) = 1 / 3) →
  (a = -1) ∧ (b = 0) ∧
  (∀ x :  ℝ, -1 < x ∧ x < 1 → 
    (if 0 ≤ x ∧ x < 1 then f x = x / (1 + x) else if -1 < x ∧ x < 0 then f x = x / (1 - x) else True)) ∧ 
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2) ∧ 
  (∀ x : ℝ, f (x - 1) + f x > 0 → (1 / 2 < x ∧ x < 1)) :=
by
  sorry

end problem_conditions_l42_42673


namespace max_distance_from_ellipse_to_line_l42_42343

theorem max_distance_from_ellipse_to_line :
  let ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 4) = 1
  let line (x y : ℝ) := x + 2 * y - Real.sqrt 2 = 0
  ∃ (d : ℝ), (∀ (x y : ℝ), ellipse x y → line x y → d = Real.sqrt 10) :=
sorry

end max_distance_from_ellipse_to_line_l42_42343


namespace minimum_a_l42_42497

-- Definitions based on the problem conditions
def odd_function {f : ℝ → ℝ} := ∀ x : ℝ, f (-x) = -f x

def monotonic (f : ℝ → ℝ) := ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The minimum value of a such that f is defined as described and meets conditions
theorem minimum_a (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : odd_function f)
  (h_def : ∀ x > 0, f x = Real.exp x + a)
  (h_mono : monotonic f) :
  a = -1 := 
sorry

end minimum_a_l42_42497


namespace angle_bisector_theorem_l42_42605

open EuclideanGeometry

noncomputable def acute_angled_triangle {α : Type} [Field α] (A B C : Point α) :=
  (acute_angle (angle B A C)) ∧ 
  (acute_angle (angle A B C)) ∧ 
  (acute_angle (angle A C B))

-- Properties and definitions for the feet of altitudes
noncomputable def foot_of_altitude {α : Type} [Field α] (A B C A1 B1 C1 : Point α) :=
  (altitude A B C A1) ∧ 
  (altitude B A C B1) ∧
  (altitude C A B C1)

-- Properties for points K and M and their specific angles
noncomputable def specific_points_with_angle 
  {α : Type} [Field α] (A A1 C1 K M : Point α) :=
  (on_segment A1 C1 K) ∧ 
  (on_segment B1 C1 M) ∧
  (angle_equal (angle K A M) (angle A1 A C))

theorem angle_bisector_theorem 
  {α : Type} [Field α] (A B C A1 B1 C1 K M : Point α) 
  (h_acute : acute_angled_triangle A B C)
  (h_feet : foot_of_altitude A B C A1 B1 C1)
  (h_points : specific_points_with_angle A A1 C1 K M) :
  is_angle_bisector A K C1 K M :=
by
  sorry

end angle_bisector_theorem_l42_42605


namespace prob_selecting_green_ball_l42_42020

-- Definition of the number of red and green balls in each container
def containerI_red := 10
def containerI_green := 5
def containerII_red := 3
def containerII_green := 5
def containerIII_red := 2
def containerIII_green := 6
def containerIV_red := 4
def containerIV_green := 4

-- Total number of balls in each container
def total_balls_I := containerI_red + containerI_green
def total_balls_II := containerII_red + containerII_green
def total_balls_III := containerIII_red + containerIII_green
def total_balls_IV := containerIV_red + containerIV_green

-- Probability of selecting a green ball from each container
def prob_green_I := containerI_green / total_balls_I
def prob_green_II := containerII_green / total_balls_II
def prob_green_III := containerIII_green / total_balls_III
def prob_green_IV := containerIV_green / total_balls_IV

-- Probability of selecting any one container
def prob_select_container := (1:ℚ) / 4

-- Combined probability for a green ball from each container
def combined_prob_I := prob_select_container * prob_green_I 
def combined_prob_II := prob_select_container * prob_green_II 
def combined_prob_III := prob_select_container * prob_green_III 
def combined_prob_IV := prob_select_container * prob_green_IV 

-- Total probability of selecting a green ball
def total_prob_green := combined_prob_I + combined_prob_II + combined_prob_III + combined_prob_IV 

-- Theorem to prove
theorem prob_selecting_green_ball : total_prob_green = 53 / 96 :=
by sorry

end prob_selecting_green_ball_l42_42020


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42278

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42278


namespace inverse_proportional_ratios_l42_42658

variables {x y x1 x2 y1 y2 : ℝ}
variables (h_inv_prop : ∀ (x y : ℝ), x * y = 1) (hx1_ne : x1 ≠ 0) (hx2_ne : x2 ≠ 0) (hy1_ne : y1 ≠ 0) (hy2_ne : y2 ≠ 0)

theorem inverse_proportional_ratios 
  (h1 : x1 * y1 = x2 * y2)
  (h2 : (x1 / x2) = (3 / 4)) : 
  (y1 / y2) = (4 / 3) :=
by 

sorry

end inverse_proportional_ratios_l42_42658


namespace cubic_foot_to_cubic_inches_l42_42102

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end cubic_foot_to_cubic_inches_l42_42102


namespace f_2014_value_l42_42058

def f : ℝ → ℝ :=
sorry

lemma f_periodic (x : ℝ) : f (x + 2) = f (x - 2) :=
sorry

lemma f_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x < 4) : f x = x^2 :=
sorry

theorem f_2014_value : f 2014 = 4 :=
by
  -- Insert proof here
  sorry

end f_2014_value_l42_42058


namespace line_passes_through_point_a_line_intersects_circle_shortest_chord_line_eq_l42_42891

-- Definitions
def circleC (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

def lineL (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y = 7 * m + 4

def pointA (x y : ℝ) : Prop := (x = 3) ∧ (y = 1)

-- Proof statements
theorem line_passes_through_point_a (m : ℝ) : 
  lineL m 3 1 :=
by sorry

theorem line_intersects_circle (m : ℝ) : 
  ∃ x y : ℝ, lineL m x y ∧ circleC x y :=
by sorry

theorem shortest_chord_line_eq : 
  ∃ l : ℝ → ℝ → Prop, (∀ x y, l x y ↔ 2 * x - y - 5 = 0) ∧
    (∀ m, lineL m x y → ¬ perpendicular l (diameter (1, 2) (3, 1))) :=
by sorry

end line_passes_through_point_a_line_intersects_circle_shortest_chord_line_eq_l42_42891


namespace banker_gain_is_correct_l42_42334

-- Definitions
def TD : ℝ := 60.00000000000001
def r : ℝ := 0.12
def t : ℝ := 1
def FV : ℝ := (TD * (1 + r * t)) / (r * t)
def BD : ℝ := FV * r * t
def BG : ℝ := BD - TD

-- Theorem statement
theorem banker_gain_is_correct : BG = 7.2 :=
by 
  have hFV : FV = (TD * (1 + r * t)) / (r * t), from sorry
  have hBD : BD = FV * r * t, from sorry
  have hBG : BG = BD - TD, from sorry
  have hEquation1 : 60.00000000000001 * 1.12 = 67.2, from sorry
  have hEquation2 : 0.12 * 560 = 67.2, from sorry
  have hFinal : 67.2 - 60.00000000000001 = 7.2, from sorry
  exact hFinal

end banker_gain_is_correct_l42_42334


namespace part1_part2_l42_42080

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.sin (x + Real.pi / 3) - 1

theorem part1 : f (5 * Real.pi / 6) = -2 := by
  sorry

variables {A : ℝ} (hA1 : A > 0) (hA2 : A ≤ Real.pi / 3) (hFA : f A = 8 / 5)

theorem part2 (h : A > 0 ∧ A ≤ Real.pi / 3 ∧ f A = 8 / 5) : f (A + Real.pi / 4) = 6 / 5 :=
by
  sorry

end part1_part2_l42_42080


namespace max_prop_cond_prob_expected_val_l42_42391

-- Conditions
noncomputable def xi (i : Nat) : Nat := sorry -- Placeholder: defines the i.i.d. random variables

axiom prob_pos_neg : 
  ∀ i, (Prob {ξ i = 1}) = (Prob {ξ i = -1}) = (1 / 2)

noncomputable def S2n (n : Nat) : Nat :=
  finset.sum (finset.range (2 * n)) (λ i, xi i)

noncomputable def M2n (n : Nat) : Nat :=
  finset.max (finset.image (λ i, S2n i) (finset.range (2 * n)))

-- Statements to prove
theorem max_prop (n k : Nat) (h : k ≤ n) :
  (Prob {M2n n ≥ k ∧ S2n n = 0}) = (Prob {S2n n = 2 * k}) := sorry 

theorem cond_prob (n k : Nat) (h : k ≤ n) :
  (Prob.heavy_cond_prob {M2n n ≥ k} {S2n n = 0}) = 
  (binomial_coeff (2 * n) (n + k)) / (binomial_coeff (2 * n) n) := sorry

theorem expected_val (n k : Nat) (h : k ≤ n) :
  (expected_value (λ _, M2n n) {S2n n = 0}) = 
  (1 / 2) * ((1 / (Prob {S2n n = 0})) - 1) := sorry

end max_prop_cond_prob_expected_val_l42_42391


namespace line_form_l42_42789

-- Given vector equation for a line
def line_eq (x y : ℝ) : Prop :=
  (3 * (x - 4) + 7 * (y - 14)) = 0

-- Prove that the line can be written in the form y = mx + b
theorem line_form (x y : ℝ) (h : line_eq x y) :
  y = (-3/7) * x + (110/7) :=
sorry

end line_form_l42_42789


namespace necessary_but_not_sufficient_condition_l42_42882

def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) :=
by
  sorry

end necessary_but_not_sufficient_condition_l42_42882


namespace train_crossing_time_after_detachment_l42_42804

def initial_boggies : ℕ := 12
def boggy_length : ℝ := 15 
def crossing_time : ℝ := 9
def detached_boggies : ℕ := 1

theorem train_crossing_time_after_detachment :
  let original_length := initial_boggies * boggy_length in
  let speed := original_length / crossing_time in
  let new_length := (initial_boggies - detached_boggies) * boggy_length in
  new_length / speed = 8.25 :=
by
  sorry

end train_crossing_time_after_detachment_l42_42804


namespace polynomial_divisible_l42_42640

theorem polynomial_divisible (n : ℕ) (hn : n > 0) :
  ∀ x : ℝ, (x-1)^3 ∣ x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 :=
by
  sorry

end polynomial_divisible_l42_42640


namespace coefficient_of_x2y4_l42_42023

theorem coefficient_of_x2y4 : coefficient (expand (2 * x + y) ((x - 2 * y)^5)) (x^2 * y^4) = 80 :=
by sorry

end coefficient_of_x2y4_l42_42023


namespace original_function_l42_42958

def f (x : ℝ) : ℝ := 
sorry

theorem original_function :
  (∀ x, (sin (x - π/4) = sin ((2 * (x - π / 3))))) ↔ (∀ x, f x = sin (x/2 + π/12)) :=
sorry

end original_function_l42_42958


namespace triangle_area_is_correct_l42_42837

noncomputable def triangle_area (a b c: ℝ) (h: a = b) : ℝ :=
  (1 / 2) * c * real.sqrt (a ^ 2 - (c / 2) ^ 2)

theorem triangle_area_is_correct : triangle_area 7 7 5 (rfl : 7 = 7) = (5 * real.sqrt 42.75) / 2 :=
  sorry

end triangle_area_is_correct_l42_42837


namespace inverse_at_neg_two_l42_42917

def g (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_at_neg_two :
  g (-2) = -43 :=
by
  -- sorry here to skip the proof, as instructed.
  sorry

end inverse_at_neg_two_l42_42917


namespace find_x_value_l42_42840

theorem find_x_value :
  ∃ x : ℝ, 5^(-6) = (5^(75/x - 4)) / ((5^(40/x)) * (25^(25/x))) ∧ x = 19 / 2 :=
sorry

end find_x_value_l42_42840


namespace polygon_diagonals_eq_sum_sides_and_right_angles_l42_42378

-- Define the number of sides of the polygon
variables (n : ℕ)

-- Definition of the number of diagonals in a convex n-sided polygon
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Definition of the sum of interior angles of an n-sided polygon
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

-- Definition of equivalent right angles for interior angles
def num_right_angles (n : ℕ) : ℕ := 2 * (n - 2)

-- The proof statement: prove that the equation holds for n
theorem polygon_diagonals_eq_sum_sides_and_right_angles (h : 3 ≤ n) :
  num_diagonals n = n + num_right_angles n :=
sorry

end polygon_diagonals_eq_sum_sides_and_right_angles_l42_42378


namespace tangent_line_eq_l42_42855

theorem tangent_line_eq {f : ℝ → ℝ} (hf : ∀ x, f x = x - 2 * Real.log x) :
  ∃ m b, (m = -1) ∧ (b = 2) ∧ (∀ x, f x = m * x + b) :=
by
  sorry

end tangent_line_eq_l42_42855


namespace parabola_equation_line_AB_equation_l42_42192

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42192


namespace max_transformed_set_l42_42461

/-- Define maximum value of a set -/
def is_max (s : Set ℝ) (m : ℝ) : Prop :=
  m ∈ s ∧ ∀ x ∈ s, x ≤ m

/-- Define transformed set -/
def transformed_set (B : Set ℝ) : Set ℝ :=
  {y | ∃ x in B, y = -x⁻¹}

theorem max_transformed_set (B : Set ℝ) (a₀ : ℝ) (hB : B ≠ ∅)
  (hB_nonzero : ∀ x ∈ B, x ≠ 0)
  (ha₀_max : is_max B a₀)
  (ha₀_neg : a₀ < 0) :
  is_max (transformed_set B) (-a₀⁻¹) :=
sorry

end max_transformed_set_l42_42461


namespace biased_coin_game_comparison_l42_42774

theorem biased_coin_game_comparison :
  let prob_heads := 2/3
  let prob_tails := 1/3
  let prob_game_C := prob_heads^4 + prob_tails^4
  let prob_game_D := (prob_heads^3 * 2 * prob_heads * prob_tails) + (prob_tails^3 * 2 * prob_heads * prob_tails)
  prob_game_C - prob_game_D = 5/81 :=
by
  let prob_heads := 2/3
  let prob_tails := 1/3
  let prob_game_C := prob_heads^4 + prob_tails^4
  let prob_game_D := (prob_heads^3 * 2 * prob_heads * prob_tails) + (prob_tails^3 * 2 * prob_heads * prob_tails)
  have hC : prob_game_C = 17/81 := sorry -- detailed calculation omitted
  have hD : prob_game_D = 12/81 := sorry -- detailed calculation omitted
  calc
    prob_game_C - prob_game_D
        = 17/81 - 12/81 : by rw [hC, hD]
    ... = 5/81 : by norm_num

end biased_coin_game_comparison_l42_42774


namespace trig_exponential_system_solution_l42_42732

theorem trig_exponential_system_solution (k1 k2 : ℤ) :
  ∀ x y : ℝ, 
    9^(2 * Real.tan x + Real.cos y) = 3 
    → 9^(Real.cos y) - 81^(Real.tan x) = 2 
    → Real.cos x ≠ 0 
    → x = k1 * Real.pi ∧ y = ± (Real.pi / 3) + 2 * k2 * Real.pi :=
by
  sorry

end trig_exponential_system_solution_l42_42732


namespace area_TCN_half_area_ABC_l42_42580

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [Module ℝ α]
variables (A B C K₁ K₂ K₃ N T : affine.algebra.affine_space ℝ α)

-- Definitions of right triangle and points on specific lines
def right_triangle (A B C : α) : Prop := 
  ∀ (x : α), (x = A + B) ∧ ∥C - x∥ = ∥C - x∥

-- Conditions as given in the problem:
def condition_1 : right_triangle A B C := sorry
def condition_2 : affine.algebra.affine_intersection K₁ K₃ N B C := sorry
def condition_3 : affine.algebra.affine_intersection K₂ K₃ T A C := sorry

-- The theorem statement:
theorem area_TCN_half_area_ABC :
  right_triangle A B C →
  affine.algebra.affine_intersection K₁ K₃ N B C →
  affine.algebra.affine_intersection K₂ K₃ T A C →
  (area_triangle T C N) = (1 / 2) * (area_triangle A B C) := 
sorry

end area_TCN_half_area_ABC_l42_42580


namespace range_of_a_l42_42881

noncomputable def g (a x : ℝ) : ℝ := Real.log2 (a + 2^(-x))

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, 1 < x1 ∧ x1 < x2 ∧ x2 < 2 → (g a x1 - g a x2) / (x1 - x2) > -2) →
  a ∈ Set.Ici (-1/8 : ℝ) :=
by
  sorry

end range_of_a_l42_42881


namespace codys_grandmother_age_l42_42011

theorem codys_grandmother_age
  (cody_age : ℕ)
  (grandmother_multiplier : ℕ)
  (h_cody_age : cody_age = 14)
  (h_grandmother_multiplier : grandmother_multiplier = 6) :
  (cody_age * grandmother_multiplier = 84) :=
by
  sorry

end codys_grandmother_age_l42_42011


namespace polka_dot_blankets_ratio_l42_42848

theorem polka_dot_blankets_ratio (total_blankets : ℕ) (total_polka_dot_blankets : ℕ) (given_polka_dot_blankets : ℕ)
  (h_total_blankets : total_blankets = 24)
  (h_total_polka_dot_blankets : total_polka_dot_blankets = 10)
  (h_given_polka_dot_blankets : given_polka_dot_blankets = 2) :
  let before_birthday_polka_dot_blankets := total_polka_dot_blankets - given_polka_dot_blankets in
  before_birthday_polka_dot_blankets / total_blankets = 1 / 3 :=
by
  sorry

end polka_dot_blankets_ratio_l42_42848


namespace find_unknown_number_l42_42038

theorem find_unknown_number (x : ℝ) (h : (45 + 23 / x) * x = 4028) : x = 89 :=
sorry

end find_unknown_number_l42_42038


namespace original_function_l42_42999

def f (x : ℝ) : ℝ := sin (x - π / 4)

theorem original_function :
  ∃ g : ℝ → ℝ, (∀ x, f (2 * (x + π / 3)) = sin (x - π / 4)) ∧ g x = sin (x / 2 + π / 12) :=
sorry

end original_function_l42_42999


namespace problem1_problem2_l42_42269

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42269


namespace num_powers_of_2_not_4_under_500000_l42_42942

theorem num_powers_of_2_not_4_under_500000 : 
  {n : ℕ | (n < 500000) ∧ (∃ k, n = 2 ^ k) ∧ (∀ m, n ≠ 4 ^ m)}.card = 9 := by 
  sorry

end num_powers_of_2_not_4_under_500000_l42_42942


namespace range_of_m_l42_42047

theorem range_of_m (m : ℝ) : 
  ((0.8^1.2)^m < (1.2^0.8)^m) → m ∈ Set.Iio 0 :=
by 
  sorry

end range_of_m_l42_42047


namespace range_of_x_when_m_eq_4_range_of_m_given_conditions_l42_42883

-- Definitions of p and q
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0

-- Question 1: Given m = 4 and conditions p ∧ q being true, prove the range of x is 4 < x < 5
theorem range_of_x_when_m_eq_4 (x m : ℝ) (h_m : m = 4) (h : p x ∧ q x m) : 4 < x ∧ x < 5 := 
by
  sorry

-- Question 2: Given conditions ⟪¬q ⟫is a sufficient but not necessary condition for ⟪¬p ⟫and constraints, prove the range of m is 5/3 ≤ m ≤ 2
theorem range_of_m_given_conditions (m : ℝ) (h_sufficient : ∀ (x : ℝ), ¬q x m → ¬p x) (h_constraints : m > 0) : 5 / 3 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_x_when_m_eq_4_range_of_m_given_conditions_l42_42883


namespace min_moves_to_sort_children_l42_42666

theorem min_moves_to_sort_children : ∀ (children : list ℕ), children.length = 10 → 
  (∀ c, c ∈ children → ∃ n ∈ finset.range 1 11, c = n) → 
  ∀ initial_positions, (∃ perms, list.perm perms initial_positions ∧ 
  (∀ i, perms[i] > perms[i + 1] →
  (minimum_moves perms = 8))) :=
by
  sorry

end min_moves_to_sort_children_l42_42666


namespace remainder_of_division_l42_42859

variable (a : ℝ) (b : ℝ)

theorem remainder_of_division : a = 28 → b = 10.02 → ∃ r : ℝ, 0 ≤ r ∧ r < b ∧ ∃ q : ℤ, a = q * b + r ∧ r = 7.96 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end remainder_of_division_l42_42859


namespace find_n_l42_42060

noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), a i

theorem find_n 
  (a : ℕ → ℝ)
  (h_arith : ∀ n > 1, 2 * a n = a (n + 1) + a (n - 1))
  (h1 : Sn a 3 < Sn a 5)
  (h2 : Sn a 5 < Sn a 4) : 
  ∃ n > 1, Sn a (n - 1) * Sn a n < 0 ∧ n = 9 := 
sorry

end find_n_l42_42060


namespace vote_difference_60_l42_42722

theorem vote_difference_60
    (total_members : ℕ)
    (x y x' y' m : ℕ)
    (total_eq : x + y = total_members)
    (initial_defeat : y > x)
    (margin_defeat : y - x = m)
    (revote_pass_margin : x' - y' = 2 * m)
    (revote_total_eq : x' + y' = total_members)
    (revote_ratio : x' = 12 * y / 11) :
    x' - x = 60 :=
by
  sorry

end vote_difference_60_l42_42722


namespace normal_price_l42_42741

theorem normal_price (P : ℝ) 
  (discount1 discount2 sale_price : ℝ) 
  (h_discount1 : discount1 = 0.10) 
  (h_discount2 : discount2 = 0.20) 
  (h_sale_price : sale_price = 36) : 
  P = 50 := 
by 
  let discounted_once := P * (1 - discount1)
  have h_discounted_once : discounted_once = P * 0.90, by rw [h_discount1]
  let discounted_twice := discounted_once * (1 - discount2)
  have h_discounted_twice : discounted_twice = discounted_once * 0.80, by rw [h_discount2]
  have h_final_price : discounted_twice = sale_price, by rw [h_sale_price]
  sorry

end normal_price_l42_42741


namespace distinct_integers_real_roots_l42_42701

theorem distinct_integers_real_roots (a b c : ℤ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > b) (h5 : b > c) :
    (∃ x : ℝ, x^2 + 2 * a * x + 3 * (b + c) = 0) :=
sorry

end distinct_integers_real_roots_l42_42701


namespace dice_surface_sum_l42_42753

theorem dice_surface_sum :
  ∃ X : ℤ, 1 ≤ X ∧ X ≤ 6 ∧ 
  (28175 + 2 * X = 28177 ∨
   28175 + 2 * X = 28179 ∨
   28175 + 2 * X = 28181 ∨
   28175 + 2 * X = 28183 ∨
   28175 + 2 * X = 28185 ∨
   28175 + 2 * X = 28187) := sorry

end dice_surface_sum_l42_42753


namespace total_tips_l42_42370

def tips_per_customer := 2
def customers_friday := 28
def customers_saturday := 3 * customers_friday
def customers_sunday := 36

theorem total_tips : 
  (tips_per_customer * (customers_friday + customers_saturday + customers_sunday) = 296) :=
by
  sorry

end total_tips_l42_42370


namespace find_obtuse_angle_l42_42902

-- Define the conditions
def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180

-- Lean statement assuming the needed conditions
theorem find_obtuse_angle (α : ℝ) (h1 : is_obtuse α) (h2 : 4 * α = 360 + α) : α = 120 :=
by sorry

end find_obtuse_angle_l42_42902


namespace single_dog_barks_per_minute_l42_42406

theorem single_dog_barks_per_minute (x : ℕ) (h : 10 * 2 * x = 600) : x = 30 :=
by
  sorry

end single_dog_barks_per_minute_l42_42406


namespace cubic_inches_in_one_cubic_foot_l42_42108

theorem cubic_inches_in_one_cubic_foot (h : 1.foot = 12.inches) : (1.foot)^3 = 1728 * (1.inches)^3 :=
by
  rw [h]
  calc (12.foot)^3 = 12^3 * (1.inches)^3 : sorry

end cubic_inches_in_one_cubic_foot_l42_42108


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42275

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42275


namespace smallest_alpha_l42_42439

noncomputable def minimum_alpha : ℝ :=
  1 / 3

theorem smallest_alpha (C : set (ℝ × ℝ)) (h1 : C.nonempty) (h2 : is_closed C) (h3 : bounded C)
    (h4 : convex C) :
    ∃ α ∈ Icc (0 : ℝ) 1, (∀ (L : line ℝ), ∃ (Bα : set (ℝ × ℝ)) (hBA : is_band Bα),
    (parallel Bα L ∧ width Bα = α * distance_to_further_parallel L C ∧
    intersection_contains_point_of C Bα)) :=
begin
  use minimum_alpha,
  split,
  { norm_num, },
  { intros L,
    sorry
  }
end

end smallest_alpha_l42_42439


namespace sum_of_arithmetic_sequence_equality_l42_42175

theorem sum_of_arithmetic_sequence_equality:
  ∀ (n : ℕ), n ≠ 0 → (let s1 := 2 * n * (n + 3)
                     let s2 := n * (n + 16)
                     in s1 = s2 → n = 10) :=
by
  intros n hn
  let s1 := 2 * n * (n + 3)
  let s2 := n * (n + 16)
  intro h_eq
  have h_simp : 2 * n * (n + 3) = n * (n + 16) := h_eq
  have h_exp : 2 * n^2 + 6 * n = n^2 + 16 * n := by linarith
  have h_simp2 : n^2 - 10 * n = 0 := by linarith
  have : n * (n - 10) = 0 := by linarith
  cases Classical.em (n = 0) with hn0 hn0,
  { contradiction },
  { exact eq_of_mul_eq_zero_right this }
  sorry

end sum_of_arithmetic_sequence_equality_l42_42175


namespace find_x_when_perpendicular_l42_42091

def a : ℝ × ℝ := (1, -2)
def b (x: ℝ) : ℝ × ℝ := (x, 1)
def are_perpendicular (a b: ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x_when_perpendicular (x: ℝ) (h: are_perpendicular a (b x)) : x = 2 :=
by
  sorry

end find_x_when_perpendicular_l42_42091


namespace polynomial_solution_exists_unique_l42_42504

theorem polynomial_solution_exists_unique
  (n : ℕ) (h : 0 < n) :
  ∃ (f : ℝ[X]), f.degree = (2 * n + 1) ∧
  (∀ x, (polynomial.eval x (polynomial.derivative (f + 1))) mod (x - 1)^n = 0) ∧
  (∀ x, (polynomial.eval x (polynomial.derivative (f - 1))) mod (x + 1)^n = 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧
    f = (polynomial.C a * polynomial.X + polynomial.C b) * 
        (polynomial.X + polynomial.C 1)^n * 
        (polynomial.X - polynomial.C 1)^n - 
        polynomial.C (1 / (finset.range n).sum (λ i, 
        (nat.choose (n - 1) i) * (-1)^(n - 1 - i) / (2 * i + 1))) * 
        (finset.range n).sum (λ i, 
        (nat.choose (n - 1) i) * (-1)^(n - 1 - i) * 
        (polynomial.X^(2 * i + 1)))) :=
by sorry

end polynomial_solution_exists_unique_l42_42504


namespace Tom_age_problem_l42_42362

theorem Tom_age_problem 
  (T : ℝ) 
  (h1 : T = T1 + T2 + T3 + T4) 
  (h2 : T - 3 = 3 * (T - 3 - 3 - 3 - 3)) : 
  T / 3 = 5.5 :=
by 
  -- sorry here to skip the proof
  sorry

end Tom_age_problem_l42_42362


namespace evaluate_f_l42_42051

variable (f : ℕ+ → ℕ+ → ℕ+)

-- Conditions
axiom f_init : f 1 1 = 1
axiom f_pos : ∀ m n : ℕ+, f m n > 0
axiom f_recur_n : ∀ m n : ℕ+, f m (n + 1) = f m n + 2
axiom f_recur_m : ∀ m : ℕ+, f (m + 1) 1 = 2 * f m 1

-- Goals
theorem evaluate_f :
  f 1 5 = 9 ∧ f 5 1 = 16 ∧ f 5 6 = 26 := by
  sorry

end evaluate_f_l42_42051


namespace sale_price_after_discounts_l42_42348

def calculate_sale_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount => price * (1 - discount)) original_price

theorem sale_price_after_discounts :
  calculate_sale_price 500 [0.10, 0.15, 0.20, 0.25, 0.30] = 160.65 :=
by
  sorry

end sale_price_after_discounts_l42_42348


namespace odd_powers_sum_divisible_by_p_l42_42168

theorem odd_powers_sum_divisible_by_p
  (p : ℕ)
  (hp_prime : Prime p)
  (hp_gt_3 : 3 < p)
  (a b c d : ℕ)
  (h_sum : (a + b + c + d) % p = 0)
  (h_cube_sum : (a^3 + b^3 + c^3 + d^3) % p = 0)
  (n : ℕ)
  (hn_odd : n % 2 = 1 ) :
  (a^n + b^n + c^n + d^n) % p = 0 :=
sorry

end odd_powers_sum_divisible_by_p_l42_42168


namespace solve_for_x_l42_42719

theorem solve_for_x (x : ℤ) (h : 2 * x + 13 = 89) : x = 38 :=
by {
  sorry,
}

end solve_for_x_l42_42719


namespace part1_part2_case1_part2_case2_part2_case3_l42_42085

section Part1
variable (a b : ℝ)
theorem part1 (h : ∀ x : ℝ, ax^2 - b ≥ 2x - ax ∧ -2 ≤ x ∧ x ≤ -1) : 
  a = -1 ∧ b = 2 := by
  sorry
end Part1

section Part2
variable (a : ℝ)
theorem part2_case1 (h : a < -2 ∧ ∀ x : ℝ, (ax - 2)*(x + 1) ≥ 0) : 
  ∀ x, -1 ≤ x ∧ x ≤ 2 / a := by
  sorry

theorem part2_case2 (h : a = -2 ∧ ∀ x : ℝ, (ax - 2)*(x + 1) ≥ 0) : 
  ∀ x, x = -1 := by
  sorry

theorem part2_case3 (h : -2 < a ∧ a < 0 ∧ ∀ x : ℝ, (ax - 2)*(x + 1) ≥ 0) : 
  ∀ x, 2 / a ≤ x ∧ x ≤ -1 := by
  sorry
end Part2

end part1_part2_case1_part2_case2_part2_case3_l42_42085


namespace leak_empties_cistern_in_72_hours_l42_42402

theorem leak_empties_cistern_in_72_hours :
  let fill_rate := 1 / 8
  let combined_rate := 1 / 9
  let leak_rate := fill_rate - combined_rate
  let empty_time := 1 / leak_rate
  empty_time = 72 :=
by
  let fill_rate := 1 / 8
  let combined_rate := 1 / 9
  let leak_rate := fill_rate - combined_rate
  let empty_time := 1 / leak_rate
  have h : empty_time = 72
  exact h

end leak_empties_cistern_in_72_hours_l42_42402


namespace problem1_problem2_l42_42261

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42261


namespace derivative1_derivative2_derivative3_differential_eq4_derivative5_derivative6_l42_42867

-- Define the first problem 
theorem derivative1 : 
  ∀ x : ℝ, 
  (deriv^[3] (λ x, x^5 - 7 * x^3 + 2)) x = 60 * x^2 - 42 := 
sorry

-- Define the second problem
theorem derivative2 : 
  ∀ x : ℝ, x ≠ 0 → 
  (deriv^[5] (λ x, log x)) x = 24 / x^6 := 
sorry

-- Define the third problem, evaluation at x = -1
theorem derivative3 : 
  (deriv^[2] (λ x, arctan (2 * x))) (-1) = 8 / 25 := 
sorry

-- Define the fourth problem, proving the differential equation
theorem differential_eq4 : 
  ∀ φ : ℝ, 
  let y := λ φ, exp (-φ) * sin φ in 
  deriv^[2] y φ + 2 * (deriv y φ) + 2 * y φ = 0 := 
sorry

-- Define the fifth problem
theorem derivative5 : 
  ∀ x : ℝ, 
  (deriv^[24] (λ x, exp x * (x^2 - 1))) x = exp x * (x^2 + 48 * x + 551) := 
sorry

-- Define the sixth problem
theorem derivative6 : 
  ∀ (x : ℝ) (m k : ℕ), 
  (deriv^[k] (λ x, x^m)) x = 
  if k ≤ m then 
    (m.factorial / (m - k).factorial) * x^(m - k) 
  else 
    0 := 
sorry

end derivative1_derivative2_derivative3_differential_eq4_derivative5_derivative6_l42_42867


namespace max_different_ages_l42_42333

theorem max_different_ages (avg_age stddev : ℕ) (h_avg : avg_age = 30) (h_stddev : stddev = 8) : 
  (count_ages_in_range avg_age stddev) = 17 := 
by
  sorry

def count_ages_in_range (avg stddev : ℕ) : ℕ :=
  let min_age := avg - stddev
  let max_age := avg + stddev
  (max_age - min_age + 1)

end max_different_ages_l42_42333


namespace polynomial_evaluation_sum_l42_42602

theorem polynomial_evaluation_sum :
  let f := (x : ℤ) → x^5 + x^3 - x^2 - 1
  ∃ q1 q2 : Polynomial ℤ,
  q1.monic ∧ q2.monic ∧
  irreducible q1 ∧ irreducible q2 ∧
  f = q1 * q2 ∧
  q1.eval 2 + q2.eval 2 = 36 :=
by
  let f := (x : ℤ) → x^5 + x^3 - x^2 - 1
  sorry

end polynomial_evaluation_sum_l42_42602


namespace even_sum_probability_l42_42708

theorem even_sum_probability {A B : Finset ℕ} (hA : A = {3, 4, 5, 8}) (hB : B = {6, 7, 9}) :
  (let even_sums := { (a, b) ∈ A ×ˢ B | (a + b) % 2 = 0 } in
  (even_sums.card / (A.card * B.card) : ℚ)) = 1 / 2 :=
by
  sorry

end even_sum_probability_l42_42708


namespace exists_student_solved_one_problem_l42_42819

theorem exists_student_solved_one_problem (m n : ℕ) (hm : m > 1) (hn : n > 1) 
(h1 : ∀ i j, i ≠ j → (∃ k, k ∈ finset.range n ∧ (∃ l, l ∈ finset.range m ∧ "student i solved k problems and student j solved l problems")))
(h2 : ∀ i j, i ≠ j → (∃ k, k ∈ finset.range m ∧ (∃ l, l ∈ finset.range n ∧ "problem i was solved by k students and problem j was solved by l students"))): 
  ∃ s, "student s solved exactly one problem" := 
by
  sorry

end exists_student_solved_one_problem_l42_42819


namespace count_solutions_in_interval_l42_42943

theorem count_solutions_in_interval :
  ∃ n : ℕ, n = 2 ∧
           ∀ x : ℝ,
             0 ≤ x ∧ x ≤ 2 * Real.pi →
             (sin (Real.pi / 3 * cos x) = cos (Real.pi / 3 * sin x) ↔ n = 2) :=
begin
  sorry
end

end count_solutions_in_interval_l42_42943


namespace constant_ratio_l42_42510

-- Definitions and conditions
def ellipse_eq (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def passes_through (x0 y0 a b : ℝ) : Prop :=
  ellipse_eq a b x0 y0

def ecc (a b c e : ℝ) : Prop :=
  e = c / a ∧ e = sqrt 2 / 2 ∧ a^2 = b^2 + c^2

lemma find_ellipse (a b : ℝ) :
  (a > b) → (b > 0) → passes_through 0 (-1) a b →
  ecc a b (sqrt (a^2 - b^2)) (sqrt 2 / 2) →
  ellipse_eq a b 0 (-1) →
  ellipse_eq (sqrt 2) 1 0 (-1) :=
by sorry

theorem constant_ratio (k a b x1 y1 x2 y2 x0 y0 p : ℝ) :
  (k ≠ 0) → (a = sqrt 2 ∧ b = 1) →
  ellipse_eq a b x1 y1 →
  ellipse_eq a b x2 y2 →
  -- Calculations involving x0, y0, p, and points M, N, P
  let x0 := (x1 + x2) / 2 in
  let y0 := k * (x0 - 1) in
  let p := k^2 / (2 * k^2 + 1) in
  let PF := 1 - p in
  let MN := sqrt ((1 + k^2) * ((4 * k^2 / (2 * k^2 + 1))^2 - 4 * (2 * k^2 - 2) / (2 * k^2 + 1))) in
  |MN| / |PF| = 2 * sqrt 2 :=
by sorry

end constant_ratio_l42_42510


namespace intersection_eq_l42_42565

def M : Set ℝ := {-1, 1}
def N : Set ℝ := { x | 1 ≤ 2^x ∧ 2^x ≤ 4}

theorem intersection_eq : M ∩ N = {1} := by
  sorry

end intersection_eq_l42_42565


namespace solve_for_x_l42_42609

def spadesuit (a b : ℝ) : ℝ :=
  (a^2 - b^2) / (2 * b - 2 * a)

theorem solve_for_x :
  (∃ x : ℝ, spadesuit 3 x = -10) → (x = 17) :=
sorry

end solve_for_x_l42_42609


namespace parabola_equation_line_AB_equation_l42_42185

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42185


namespace fraction_subtraction_l42_42712

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) 
  = 9 / 20 := by
  sorry

end fraction_subtraction_l42_42712


namespace maximal_surprising_matches_l42_42607

theorem maximal_surprising_matches (N : ℕ) (hN_odd : N % 2 = 1) (hN_ge3 : 3 ≤ N) :
  let surprising_matches := (N - 1) * (3 * N - 1) / 8 in
  surprising_matches = n ↔ n = (N-1) * (3 * N - 1) / 8 :=
begin
  sorry
end

end maximal_surprising_matches_l42_42607


namespace cubic_inches_in_one_cubic_foot_l42_42109

theorem cubic_inches_in_one_cubic_foot (h : 1.foot = 12.inches) : (1.foot)^3 = 1728 * (1.inches)^3 :=
by
  rw [h]
  calc (12.foot)^3 = 12^3 * (1.inches)^3 : sorry

end cubic_inches_in_one_cubic_foot_l42_42109


namespace original_function_l42_42986

theorem original_function :
  ∃ f : ℝ → ℝ,
  (∀ x : ℝ, sin (x - π / 4) = f (2 * (x - π / 3))) →
  (∀ x : ℝ, f x = sin (x / 2 + π / 12)) :=
by
  sorry

end original_function_l42_42986


namespace jackson_calories_l42_42153
noncomputable theory

def calories_salad : ℕ := 50 + 2 * 50 + 30 + 60 + 210
def calories_pizza : ℕ := 600 + (1/3 : ℝ) * 600 + (1/4 : ℝ) * 600 + 400
def calories_eaten : ℝ := 3/8 * calories_salad + 2/7 * calories_pizza

theorem jackson_calories : floor calories_eaten = 554 := by
  have h1 : calories_salad = 450 := rfl
  have h2 : calories_pizza = 1350 := rfl
  have h3 : calories_eaten = 3/8 * 450 + 2/7 * 1350 := by rw [h1, h2]
  have h4 : 3/8 * 450 + 2/7 * 1350 = 554.46 := by norm_num
  have h5 : floor 554.46 = 554 := rfl
  rw [h3, h4, h5]
  exact rfl

end jackson_calories_l42_42153


namespace daniel_monday_sunday_time_difference_l42_42026

-- Define the total distance Daniel drives back from work every day
def total_distance : ℕ := 150

-- Define Daniel's constant speed on Sunday
variable (x : ℝ)

-- Define the time taken on Sunday
def T_sunday (x : ℝ) :=
  total_distance / x

-- Define the segment distances and speeds on Monday
def segment1_distance : ℕ := 32
def segment2_distance : ℕ := 50
def segment3_distance : ℕ := 40
def segment4_distance : ℕ := total_distance - segment1_distance - segment2_distance - segment3_distance

def segment1_speed (x : ℝ) := (2 * x)
def segment2_speed (x : ℝ) := (x / 3)
def segment3_speed (x : ℝ) := (3 * x / 2)
def segment4_speed (x : ℝ) := (x / 2)

-- Define the time taken on each segment on Monday
def T1_monday (x : ℝ) :=
  segment1_distance / segment1_speed x

def T2_monday (x : ℝ) :=
  segment2_distance / segment2_speed x

def T3_monday (x : ℝ) :=
  segment3_distance / segment3_speed x

def T4_monday (x : ℝ) :=
  segment4_distance / segment4_speed x

-- Define the total time taken on Monday
def T_monday (x : ℝ) :=
  T1_monday x + T2_monday x + T3_monday x + T4_monday x

-- Calculate the percent increase in time from Sunday to Monday
def percent_increase (T_monday : ℝ) (T_sunday : ℝ) :=
  ((T_monday - T_sunday) / T_sunday) * 100

theorem daniel_monday_sunday_time_difference (x : ℝ) (hx : x > 0) :
  percent_increase (T_monday x) (T_sunday x) ≈ 65.78 :=
by
  sorry

end daniel_monday_sunday_time_difference_l42_42026


namespace dice_surface_sum_l42_42754

noncomputable def surface_sum_of_dice (n : ℕ) := 28175 + 2 * n

theorem dice_surface_sum (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6):
  ∃ s, s ∈ {28177, 28179, 28181, 28183, 28185, 28187} ∧ s = surface_sum_of_dice X := by
  use surface_sum_of_dice X
  have : 28175 + 2 * X ∈ {28177, 28179, 28181, 28183, 28185, 28187} := by
    interval_cases X
    all_goals simp [surface_sum_of_dice]
  exact ⟨this, rfl⟩
sorry

end dice_surface_sum_l42_42754


namespace milk_left_after_three_operations_l42_42733

def milk_volume_after_n_operations
  (initial_volume : ℝ)
  (remove_replace_volume : ℝ)
  (n : ℕ) : ℝ :=
if n = 0 then initial_volume else
let rec compute (vol : ℝ) (i : ℕ) :=
  if i = 0 then vol else
  let vol_after_removal := vol - remove_replace_volume * (vol / initial_volume) in
  let new_volume := vol_after_removal + remove_replace_volume in
  compute new_volume (i - 1)
in compute initial_volume n

theorem milk_left_after_three_operations :
  milk_volume_after_n_operations 40 4 3 = 29.16 := 
sorry

end milk_left_after_three_operations_l42_42733


namespace simple_interest_rate_problem_l42_42802

noncomputable def simple_interest_rate (P : ℝ) (T : ℝ) (final_amount : ℝ) : ℝ :=
  (final_amount - P) * 100 / (P * T)

theorem simple_interest_rate_problem
  (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : T = 2)
  (h2 : final_amount = (7 / 6) * P)
  (h3 : simple_interest_rate P T final_amount = R) : 
  R = 100 / 12 := sorry

end simple_interest_rate_problem_l42_42802


namespace original_function_l42_42959

def f (x : ℝ) : ℝ := 
sorry

theorem original_function :
  (∀ x, (sin (x - π/4) = sin ((2 * (x - π / 3))))) ↔ (∀ x, f x = sin (x/2 + π/12)) :=
sorry

end original_function_l42_42959


namespace rational_iff_arithmetic_progression_l42_42686

theorem rational_iff_arithmetic_progression (x : ℝ) : 
  (∃ (i j k : ℤ), i < j ∧ j < k ∧ (x + i) + (x + k) = 2 * (x + j)) ↔ 
  (∃ n d : ℤ, d ≠ 0 ∧ x = n / d) := 
sorry

end rational_iff_arithmetic_progression_l42_42686


namespace maximum_of_x_plus_y_l42_42116

theorem maximum_of_x_plus_y (x y : ℝ) (h : 2^x + 2^y = 1) : x + y ≤ -2 := sorry

end maximum_of_x_plus_y_l42_42116


namespace sqrt2_sq_sub_sqrt9_plus_cbrt8_l42_42008

theorem sqrt2_sq_sub_sqrt9_plus_cbrt8 : (sqrt 2)^2 - sqrt 9 + (8^(1/3)) = 1 := by
  sorry

end sqrt2_sq_sub_sqrt9_plus_cbrt8_l42_42008


namespace problem1_problem2_l42_42394

def problem1_lhs : Real := sqrt 4 + 2 * Real.sin (Real.pi / 4) - (Real.pi - 3)^0 + abs (sqrt 2 - 2)

theorem problem1 : problem1_lhs = 3 := by
  sorry

def problem2_ineq1 (x : Real) : Prop := 2 * (x + 2) - x <= 5
def problem2_ineq2 (x : Real) : Prop := (4 * x + 1) / 3 > x - 1

theorem problem2 (x : Real) : problem2_ineq1 x ∧ problem2_ineq2 x ↔ -4 < x ∧ x <= 1 := by
  sorry

end problem1_problem2_l42_42394


namespace compare_logs_l42_42050

noncomputable def a := Real.log 2 / Real.log 3
noncomputable def b := Real.log 3 / Real.log 5
noncomputable def c := Real.log 5 / Real.log 8

theorem compare_logs : a < b ∧ b < c := by
  sorry

end compare_logs_l42_42050


namespace replace_movie_cost_l42_42160

def num_popular_action_movies := 20
def num_moderate_comedy_movies := 30
def num_unpopular_drama_movies := 10
def num_popular_comedy_movies := 15
def num_moderate_action_movies := 25

def trade_in_rate_action := 3
def trade_in_rate_comedy := 2
def trade_in_rate_drama := 1

def dvd_cost_popular := 12
def dvd_cost_moderate := 8
def dvd_cost_unpopular := 5

def johns_movie_cost : Nat :=
  let total_trade_in := 
    (num_popular_action_movies + num_moderate_action_movies) * trade_in_rate_action +
    (num_moderate_comedy_movies + num_popular_comedy_movies) * trade_in_rate_comedy +
    num_unpopular_drama_movies * trade_in_rate_drama
  let total_dvd_cost :=
    (num_popular_action_movies + num_popular_comedy_movies) * dvd_cost_popular +
    (num_moderate_comedy_movies + num_moderate_action_movies) * dvd_cost_moderate +
    num_unpopular_drama_movies * dvd_cost_unpopular
  total_dvd_cost - total_trade_in

theorem replace_movie_cost : johns_movie_cost = 675 := 
by
  sorry

end replace_movie_cost_l42_42160


namespace prime_power_condition_l42_42030

open Nat

theorem prime_power_condition (u v : ℕ) :
  (∃ p n : ℕ, p.Prime ∧ p^n = (u * v^3) / (u^2 + v^2)) ↔ ∃ k : ℕ, k ≥ 1 ∧ u = 2^k ∧ v = 2^k := by {
  sorry
}

end prime_power_condition_l42_42030


namespace officers_selection_l42_42512

theorem officers_selection :
  (∃ (A B C D : Type), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  → (Dave ≠ A)
  → (∃ (num_ways : ℕ), num_ways = 18) :=
begin
  sorry
end

end officers_selection_l42_42512


namespace curve_tangent_range_l42_42671

theorem curve_tangent_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + y + m = 0 → 
  let sym_y := (-3 * y + 2 * x + 1) / 5 in
  let sym_x := (2 * y + 4 * x - 1) / 5 in 
  x^2 + sym_y^2 + sym_y + m = 0)
  ↔ (-11 / 20 < m ∧ m < 1 / 4) :=
by
  sorry

end curve_tangent_range_l42_42671


namespace arrangement_non_adjacent_males_grouping_methods_selection_method_l42_42667

theorem arrangement_non_adjacent_males (m f : ℕ) (h1 : m = 3) (h2 : f = 5) :
  let total := m + f
  let ways := factorial f * choose (total - 1) m * factorial m 
  total = 8 → ways = 14400 := 
by
  intros
  rw [h1, h2]
  -- Proof of the calculation will go here
  sorry

theorem grouping_methods (n : ℕ) (h : n = 8) :
  let total_groups := choose n 2 * choose 6 2 * choose 4 2 * choose 2 2 / factorial 4 * factorial 4 
  total_groups = 2520 := 
by
  intros
  rw h
  -- Proof of the calculation will go here
  sorry

theorem selection_method (n m f : ℕ) (h_n : n = 8) (h_m : m = 3) (h_f : f = 5) :
  let total_selections := choose n 4 - choose f 4
  let ways := total_selections * factorial 4 
  total_selections >= 1 → ways = 1560 := 
by
  intros
  rw [h_n, h_m, h_f]
  -- Proof of the calculation will go here
  sorry

end arrangement_non_adjacent_males_grouping_methods_selection_method_l42_42667


namespace math_problem_l42_42319

variable (a : ℝ)

theorem math_problem (h : a^2 + 3 * a - 2 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - 1 / (2 - a)) / (2 / (a^2 - 2 * a)) = 1 := 
sorry

end math_problem_l42_42319


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42273

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42273


namespace DX_eq_DY_l42_42164

/-- Definitions and assumptions -/
variables (A B C D E F X Y : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace X] [MetricSpace Y]

def isosceles_right_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
  (dist A B = dist B C) ∧ (angle A B C = π / 2)

def midpoint (P Q M : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace M] :=
  dist P M = dist M Q

def equilateral_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
  dist A B = dist B C ∧ dist B C = dist C A

/-- Main theorem: Given the conditions, prove that DX = DY -/
theorem DX_eq_DY 
  (A B C D E F X Y : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace X] [MetricSpace Y]
  (h_triangle : isosceles_right_triangle A B C)
  (h_midD : midpoint C A D)
  (h_midE : midpoint B A E)
  (h_equilateral : equilateral_triangle D E F)
  (h_X : collinear B F X ∧ collinear F A C)
  (h_Y : collinear A F Y ∧ collinear F B D) :
  dist D X = dist D Y := 
sorry

end DX_eq_DY_l42_42164


namespace daily_evaporation_amount_l42_42784

-- Define the given conditions
def initial_water_volume : ℝ := 25
def evaporation_period  : ℕ := 10
def evaporation_rate    : ℝ := 1.6 / 100

-- Define the statement we need to prove
theorem daily_evaporation_amount :
  (evaporation_rate * initial_water_volume) / evaporation_period = 0.04 :=
by
  sorry

end daily_evaporation_amount_l42_42784


namespace incenter_coordinates_l42_42150

-- Define lengths of the sides of the triangle
def a : ℕ := 8
def b : ℕ := 10
def c : ℕ := 6

-- Define the incenter formula components
def sum_of_sides : ℕ := a + b + c
def x : ℚ := a / (sum_of_sides : ℚ)
def y : ℚ := b / (sum_of_sides : ℚ)
def z : ℚ := c / (sum_of_sides : ℚ)

-- Prove the result
theorem incenter_coordinates :
  (x, y, z) = (1 / 3, 5 / 12, 1 / 4) :=
by 
  -- Proof skipped
  sorry

end incenter_coordinates_l42_42150


namespace first_5_bags_l42_42665

-- Definitions based on given conditions
def total_bags : ℕ := 800
def selected_bags : ℕ := 60
def random_table_rows : list (list ℕ) :=
  [ [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 
     21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
    [63, 01, 63, 78, 59, 16, 95, 56, 67, 19, 98, 10, 50, 71, 75, 
     12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
    [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 
     15, 51, 00, 13, 42, 99, 66, 02, 79, 54] ]

noncomputable def read_random_numbers (row : list ℕ) (start_col : ℕ) : list ℕ :=
  (list.drop start_col row) ++ (list.take start_col row)

theorem first_5_bags :
  read_random_numbers (random_table_rows.head!) 6 |>.take 10 = [7, 85, 66, 7, 19, 9, 5, 7, 17, 5] →
    read_random_numbers (random_table_rows.nth 1).to_list 6 |>.take 10 = [67, 19, 9, 18, 10, 50, 71, 75, 1, 2] → 
    read_random_numbers (random_table_rows.nth 2).to_list 6 |>.take 10 = [56, 7, 78, 64, 56, 7, 82, 52] →
    random_select.is_correct
    sorry

end first_5_bags_l42_42665


namespace maximum_area_right_triangle_in_rectangle_l42_42798

theorem maximum_area_right_triangle_in_rectangle :
  ∃ (area : ℕ), 
  (∀ (a b : ℕ), a = 12 ∧ b = 5 → area = 1 / 2 * a * b) :=
by
  use 30
  sorry

end maximum_area_right_triangle_in_rectangle_l42_42798


namespace jerry_age_l42_42626

theorem jerry_age :
  ∃ J : ℕ, let M := 22 in M = 2 * J + 10 → J = 6 :=
by
  sorry

end jerry_age_l42_42626


namespace not_necessarily_divisor_of_132_l42_42328

theorem not_necessarily_divisor_of_132 (k : ℤ) :
  let n := k * (k + 1) * (k + 2) * (k + 3) in
  (11 ∣ n) -> ¬(132 ∣ n) :=
by
  intros n h11
  sorry

end not_necessarily_divisor_of_132_l42_42328


namespace digit_divisibility_l42_42782

theorem digit_divisibility :
  let M_values := {M : ℕ | M < 10 ∧ (428 * 10 + M) % 4 = 0} in
  M_values.card = 3 :=
by
  sorry

end digit_divisibility_l42_42782


namespace find_original_function_l42_42967

theorem find_original_function
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₁ : ∀ x, g (x + (π / 3)) = sin (x - (π / 4)))
  (h₂ : ∀ x, f x = g (2 * x)) :
  f x = sin (x / 2 + π / 12) :=
sorry

end find_original_function_l42_42967


namespace num_true_props_even_l42_42797

def proposition := Prop

def inverse (P : proposition) := sorry -- define what inverse means
def contrapositive (P : proposition) := sorry -- define what contrapositive means

theorem num_true_props_even (P : proposition) : 
  (if P then 1 else 0) +
  (if inverse P then 1 else 0) +
  (if ¬P then 1 else 0) +
  (if contrapositive P then 1 else 0) % 2 = 0 := 
sorry

end num_true_props_even_l42_42797


namespace distance_from_T_to_plane_ABC_l42_42638

variables {A B C T : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space T]
variables (TA TB TC : ℝ)

-- Given conditions
def orthogonal (TA TB TC : ℝ) : Prop :=
  TA = 15 ∧ TB = 15 ∧ TC = 9 ∧ 
  (TA ^ 2 + TB ^ 2 = 15 ^ 2 + 15 ^ 2) ∧ 
  (TA ^ 2 + TC ^ 2 = 15 ^ 2 + 9 ^ 2) ∧ 
  (TB ^ 2 + TC ^ 2 = 15 ^ 2 + 9 ^ 2)

-- Proof statement
theorem distance_from_T_to_plane_ABC 
  (h : orthogonal TA TB TC) : 
  (d : ℝ) = 90 / (sqrt 423) :=
sorry

end distance_from_T_to_plane_ABC_l42_42638


namespace original_price_of_article_l42_42793

theorem original_price_of_article (SP : ℝ) (profit_rate : ℝ) (P : ℝ) (h1 : SP = 550) (h2 : profit_rate = 0.10) (h3 : SP = P * (1 + profit_rate)) : P = 500 :=
by
  sorry

end original_price_of_article_l42_42793


namespace store_profit_in_february_l42_42792

variable (C : ℝ)

def initialSellingPrice := C * 1.20
def secondSellingPrice := initialSellingPrice C * 1.25
def finalSellingPrice := secondSellingPrice C * 0.88

theorem store_profit_in_february
  (initialSellingPrice_eq : initialSellingPrice C = C * 1.20)
  (secondSellingPrice_eq : secondSellingPrice C = initialSellingPrice C * 1.25)
  (finalSellingPrice_eq : finalSellingPrice C = secondSellingPrice C * 0.88)
  : finalSellingPrice C - C = 0.32 * C :=
sorry

end store_profit_in_february_l42_42792


namespace repeating_decimals_count_l42_42861

-- Definitions to represent the problem

def is_repeating_decimal (n : ℕ) : Prop :=
  let m := if even n then (n + 2) / 2 else n + 2 in
  ∃ p : ℕ, p > 1 ∧ p ∉ {2, 5} ∧ m ≠ pow 5 p

noncomputable def count_repeating_decimals : ℕ :=
  Nat.count (λ n, 1 ≤ n ∧ n ≤ 200 ∧ is_repeating_decimal n) (Finset.range 201)

theorem repeating_decimals_count : count_repeating_decimals = 192 := by
  sorry

end repeating_decimals_count_l42_42861


namespace find_equation_of_C_find_equation_of_AB_l42_42209

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42209


namespace problem_1_problem_2_l42_42198

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42198


namespace train_length_is_400_l42_42383

-- Conditions from a)
def train_speed_kmph : ℕ := 180
def crossing_time_sec : ℕ := 8

-- The corresponding length in meters
def length_of_train : ℕ := 400

-- The problem statement to prove
theorem train_length_is_400 :
  (train_speed_kmph * 1000 / 3600) * crossing_time_sec = length_of_train := by
  -- Proof is skipped as per the requirement
  sorry

end train_length_is_400_l42_42383


namespace max_value_expression_l42_42587

-- Define the points A, B, and C
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (1, 0)

-- Point D satisfies |CD| = 1
def is_on_unit_circle (D : ℝ × ℝ) : Prop :=
  (D.1 - 1)^2 + D.2^2 = 1

-- Define the vector addition and norm
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vector_norm (u : ℝ × ℝ) : ℝ := real.sqrt (u.1^2 + u.2^2)

-- Define the expression we want to maximize
def expression_to_maximize (D : ℝ × ℝ) : ℝ :=
  let OA := A
  let OB := B
  let OD := D
  vector_norm (vector_add (vector_add OA OB) OD)

-- Statement of the theorem
theorem max_value_expression : ∀ D : ℝ × ℝ, is_on_unit_circle D → expression_to_maximize D ≤ 6 :=
by sorry

end max_value_expression_l42_42587


namespace eliza_ironing_hours_l42_42467

theorem eliza_ironing_hours (h : ℕ) 
  (blouse_minutes : ℕ := 15) 
  (dress_minutes : ℕ := 20) 
  (hours_ironing_blouses : ℕ := h)
  (hours_ironing_dresses : ℕ := 3)
  (total_clothes : ℕ := 17) :
  ((60 / blouse_minutes) * hours_ironing_blouses) + ((60 / dress_minutes) * hours_ironing_dresses) = total_clothes →
  hours_ironing_blouses = 2 := 
sorry

end eliza_ironing_hours_l42_42467


namespace total_eBook_readers_l42_42159

-- We will define the given conditions as part of the state
variable (A J J' M M' : ℕ)

noncomputable def Anna_eBook_readers := 50
noncomputable def John_initial_eBook_readers := Anna_eBook_readers - 15
noncomputable def John_final_eBook_readers := John_initial_eBook_readers - 3
noncomputable def Mary_initial_eBook_readers := 2 * John_initial_eBook_readers
noncomputable def Mary_final_eBook_readers := Mary_initial_eBook_readers - 7

theorem total_eBook_readers :
  let A := Anna_eBook_readers in
  let J' := John_final_eBook_readers in
  let M' := Mary_final_eBook_readers in
  A + J' + M' = 145 :=
by
  sorry

end total_eBook_readers_l42_42159


namespace hundredth_position_value_l42_42453

def is_valid_seq_element (n : ℕ) : Prop :=
  ∃ (s : Finset ℕ), n = s.sum (λ i, 3^i)

def seq_up_to_n (n : ℕ) : List ℕ :=
  (List.range n).filter is_valid_seq_element

def nth_seq_element (k : ℕ) : ℕ :=
  (seq_up_to_n (k + 1)).get! k

theorem hundredth_position_value : nth_seq_element 99 = 981 := 
  sorry

end hundredth_position_value_l42_42453


namespace equal_chord_lengths_l42_42181

theorem equal_chord_lengths 
  (ABC : Triangle) 
  (acute_ABC : IsAcute ABC) 
  (D E F : Point)
  (AD : Altitude ABC A D)
  (BE : Altitude ABC B E)
  (CF : Altitude ABC C F)
  (BDF CDE : Triangle)
  (incircle_B : Incircle BDF)
  (incircle_C : Incircle CDE)
  (M N : Point)
  (tangent_M : TangentPoint incircle_B DF M)
  (tangent_N : TangentPoint incircle_C DE N)
  (P Q : Point)
  (secant_P : SecantPoint MN incircle_B P)
  (secant_Q : SecantPoint MN incircle_C Q)
  : MP = NQ := sorry

end equal_chord_lengths_l42_42181


namespace mult_xy_eq_200_over_3_l42_42688

def hash_op (a b : ℚ) : ℚ := a + a / b

def x : ℚ := hash_op 8 3

def y : ℚ := hash_op 5 4

theorem mult_xy_eq_200_over_3 : x * y = 200 / 3 := 
by 
  -- lean uses real division operator, and hash_op must remain rational
  sorry

end mult_xy_eq_200_over_3_l42_42688


namespace distinct_digit_sum_unique_count_l42_42829

theorem distinct_digit_sum_unique_count :
  {s | ∃ a b c d e : ℕ, 
             a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
             c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
             1 ≤ a ∧ a ≤ 5 ∧ 
             1 ≤ b ∧ b ≤ 5 ∧ 
             1 ≤ c ∧ c ≤ 5 ∧ 
             1 ≤ d ∧ d ≤ 5 ∧ 
             1 ≤ e ∧ e ≤ 5 ∧ 
             s = (10 * a + b) + (100 * c + 10 * d + e)
} .card = 30 := by
  sorry

end distinct_digit_sum_unique_count_l42_42829


namespace points_same_side_l42_42816

def parabola_a (x : ℝ) : ℝ := 2 * x ^ 2 + 4 * x
def parabola_b (x : ℝ) : ℝ := (x ^ 2) / 2 - x - 3 / 2
def parabola_c (x : ℝ) : ℝ := - x ^ 2 + 2 * x - 1
def parabola_d (x : ℝ) : ℝ := - x ^ 2 - 4 * x - 3
def parabola_e (x : ℝ) : ℝ := - x ^ 2 + 3

def point_A : ℝ × ℝ := (-1, -1)
def point_B : ℝ × ℝ := (0, 2)

theorem points_same_side (x y : ℝ) :
  (parabola_a (-1) < -1) = (parabola_a 0 < 2) ∧
  (parabola_c (-1) < -1) = (parabola_c 0 < 2) ∧
  (parabola_e (-1) < -1) = (parabola_e 0 < 2) :=
by
  sorry

end points_same_side_l42_42816


namespace part_one_part_two_l42_42238

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42238


namespace product_closest_value_l42_42376

theorem product_closest_value (a b : ℝ) (ha : a = 0.000321) (hb : b = 7912000) :
  abs ((a * b) - 2523) < min (abs ((a * b) - 2500)) (min (abs ((a * b) - 2700)) (min (abs ((a * b) - 3100)) (abs ((a * b) - 2000)))) := by
  sorry

end product_closest_value_l42_42376


namespace find_equation_of_C_find_equation_of_AB_l42_42214

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42214


namespace log4_30_proven_l42_42553

noncomputable def log4_30 (a c : ℝ) : ℝ :=
  if h1 : (log 4 30 = (1 / (2 * a))) then (1 / (2 * a)) else 0

theorem log4_30_proven (a c : ℝ) (ha : real.log 10 2 = a) (hc : real.log 10 5 = c) :
  log 4 30 = (1 / (2 * a)) :=
sorry

end log4_30_proven_l42_42553


namespace cara_total_amount_owed_l42_42448

-- Define the conditions
def principal : ℝ := 54
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the simple interest calculation
def interest (P R T : ℝ) : ℝ := P * R * T

-- Define the total amount owed calculation
def total_amount_owed (P R T : ℝ) : ℝ := P + (interest P R T)

-- The proof statement
theorem cara_total_amount_owed : total_amount_owed principal rate time = 56.70 := by
  sorry

end cara_total_amount_owed_l42_42448


namespace dice_sum_surface_l42_42764

theorem dice_sum_surface (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6) : 
  ∃ Y : ℕ, Y = 28175 + 2 * X ∧ (Y = 28177 ∨ Y = 28179 ∨ Y = 28181 ∨ Y = 28183 ∨ 
  Y = 28185 ∨ Y = 28187) :=
by
  sorry

end dice_sum_surface_l42_42764


namespace a_5_value_l42_42506

noncomputable def seq : ℕ → ℤ
| 0       => 1
| (n + 1) => (seq n) ^ 2 - 1

theorem a_5_value : seq 4 = -1 :=
by
  sorry

end a_5_value_l42_42506


namespace line_in_plane_l42_42564

/-- Given that a segment AB lies within a plane α, this implies that line AB is contained within plane α. -/
theorem line_in_plane (A B : Point)
  (α : Plane)
  (hA : A ∈ α)
  (hB : B ∈ α)
  (hSegmentAB : segment A B ⊆ α) :
  line A B ⊆ α :=
by
  sorry

end line_in_plane_l42_42564


namespace circumcenter_of_triangle_l42_42305

-- Given conditions
variables {P A B C O : Type} [metric_space P] [metric_space A] [metric_space B] [metric_space C] [metric_space O]
variable (hP : ¬(P ∈ plane A B C))          -- Point P is outside the plane of triangle ABC
variable (hPerp : is_perp P O (plane A B C)) -- PO is perpendicular to plane ABC
variable (hFoot : foot_perp P (plane A B C) = O) -- The foot of the perpendicular from P to the plane ABC is O
variable (hDist : dist P A = dist P B ∧ dist P B = dist P C) -- PA = PB = PC

-- Proof goal
theorem circumcenter_of_triangle :
  is_circumcenter O A B C :=
sorry

end circumcenter_of_triangle_l42_42305


namespace ratio_of_line_cutting_median_lines_l42_42788

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem ratio_of_line_cutting_median_lines (A B C P Q : ℝ × ℝ) 
    (hA : A = (1, 0)) (hB : B = (0, 1)) (hC : C = (0, 0)) 
    (h_mid_AB : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) 
    (h_mid_BC : Q = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) 
    (h_ratio : (Real.sqrt (P.1^2 + P.2^2) / Real.sqrt (Q.1^2 + Q.2^2)) = (Real.sqrt (Q.1^2 + Q.2^2) / Real.sqrt (P.1^2 + P.2^2))) :
  (P.1 / Q.1) = golden_ratio :=
by 
  sorry

end ratio_of_line_cutting_median_lines_l42_42788


namespace shopkeeper_gain_percent_l42_42648

variable (SP MP CP DSP Gain GainPercent : ℝ)

-- Given conditions
def conditions :=
  SP = 30 ∧
  MP = 30 ∧
  (SP = CP + 0.30 * CP) ∧
  (DSP = MP - 0.10 * MP)

-- Correct answer
def correct_answer :=
  GainPercent ≈ 16.98

-- Proof statement
theorem shopkeeper_gain_percent:
  conditions →
  (GainPercent = (Gain / CP) * 100) →
  (Gain = DSP - CP) →
  correct_answer :=
by
  intros h_conditions h_gain_percent h_gain
  sorry

end shopkeeper_gain_percent_l42_42648


namespace probability_two_a_geq_five_b_when_rolling_two_dice_l42_42363

theorem probability_two_a_geq_five_b_when_rolling_two_dice :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]]
  let successful := [(a, b) in outcomes | 2 * a >= 5 * b]
  (successful.length : ℝ) / (outcomes.length : ℝ) = 1 / 6 :=
sorry

end probability_two_a_geq_five_b_when_rolling_two_dice_l42_42363


namespace polynomial_division_l42_42479

/-- The polynomial division statement for Lean 4 -/
theorem polynomial_division :
  polynomial.coeff ((2 * X ^ 6 - 3 * X ^ 4 + X ^ 3 + 5) /ₘ (X - 2)) \
    0 = 2 ∧
  polynomial.coeff ((2 * X ^ 6 - 3 * X ^ 4 + X ^ 3 + 5) /ₘ (X - 2)) \
    1 = 4 ∧
  polynomial.coeff ((2 * X ^ 6 - 3 * X ^ 4 + X ^ 3 + 5) /ₘ (X - 2)) \
    2 = 8 ∧
  polynomial.coeff ((2 * X ^ 6 - 3 * X ^ 4 + X ^ 3 + 5) /ₘ (X - 2)) \
    3 = 13 ∧
  polynomial.coeff ((2 * X ^ 6 - 3 * X ^ 4 + X ^ 3 + 5) /ₘ (X - 2)) \
    4 = 27 ∧
  polynomial.coeff ((2 * X ^ 6 - 3 * X ^ 4 + X ^ 3 + 5) /ₘ (X - 2)) \
    5 = 54 :=
sorry

end polynomial_division_l42_42479


namespace scientific_notation_of_83_nm_l42_42628

theorem scientific_notation_of_83_nm :
  let nanometer : ℝ := 10 ^ (-9)
  let nm_83_in_meters : ℝ := 83 * nanometer
  nm_83_in_meters = 8.3 * 10 ^ (-8) :=
by
  sorry

end scientific_notation_of_83_nm_l42_42628


namespace beetle_cannot_visit_all_once_l42_42773

def is_checkerboard (c : ℕ × ℕ × ℕ) : bool :=
  (c.1 + c.2 + c.3) % 2 = 0

theorem beetle_cannot_visit_all_once :
  ∀ (path : list (ℕ × ℕ × ℕ)),
    path.head = some (2, 2, 2) →
    (∀ c ∈ path, 1 ≤ c.1 ∧ c.1 ≤ 3 ∧ 
                  1 ≤ c.2 ∧ c.2 ≤ 3 ∧ 
                  1 ≤ c.3 ∧ c.3 ≤ 3) →
    (∀ (c₁ c₂ : ℕ × ℕ × ℕ), (c₁, c₂) ∈ path.zip path.tail →
                              (abs (c₁.1 - c₂.1) + abs (c₁.2 - c₂.2) + abs (c₁.3 - c₂.3) = 1)) →
    (∀ c₁ c₂ ∈ path, c₁ = c₂ → ∃ (i j : ℕ), path.i = path.j) →
    ¬(list.nodup path ∧ path.length = 27) :=
begin
  sorry
end

end beetle_cannot_visit_all_once_l42_42773


namespace caffeine_safe_amount_l42_42681

theorem caffeine_safe_amount (max_caffeine : ℕ) (caffeine_per_drink : ℕ) (num_drinks : ℕ) :
    max_caffeine = 500 →
    caffeine_per_drink = 120 →
    num_drinks = 4 →
    max_caffeine - (caffeine_per_drink * num_drinks) = 20 :=
by
  intros h_max h_caffeine h_num
  rw [h_max, h_caffeine, h_num]
  norm_num
  sorry

end caffeine_safe_amount_l42_42681


namespace exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987_l42_42151

theorem exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987 :
  ∃ n : ℕ, n ^ n + (n + 1) ^ n ≡ 0 [MOD 1987] := sorry

end exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987_l42_42151


namespace problem1_problem2_l42_42444

-- Problem 1
theorem problem1 : 2 * Real.cos (30 * Real.pi / 180) - Real.tan (60 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) = 1 / 2 :=
by sorry

-- Problem 2
theorem problem2 : (-1) ^ 2023 + 2 * Real.sin (45 * Real.pi / 180) - Real.cos (30 * Real.pi / 180) + Real.sin (60 * Real.pi / 180) + (Real.tan (60 * Real.pi / 180)) ^ 2 = 2 + Real.sqrt 2 :=
by sorry

end problem1_problem2_l42_42444


namespace perpendicular_bisector_eqn_l42_42146

-- Definitions based on given conditions
def C₁ (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

theorem perpendicular_bisector_eqn {ρ θ : ℝ} :
  (∃ A B : ℝ × ℝ,
    A ∈ {p : ℝ × ℝ | ∃ ρ θ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ C₁ ρ θ} ∧
    B ∈ {p : ℝ × ℝ | ∃ ρ θ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ C₂ ρ θ}) →
  ρ * Real.sin θ + ρ * Real.cos θ = 1 :=
sorry

end perpendicular_bisector_eqn_l42_42146


namespace bc_line_equation_l42_42922

open Real

def parabola_equation (p x y : ℝ) : Prop := y^2 = 2 * p * x

def point_on_parabola (p : ℝ) (A : ℝ × ℝ) : Prop := parabola_equation p A.1 A.2

def vector_relation (A B C F : ℝ × ℝ) : Prop := 
  (B.1 - A.1, B.2 - A.2) + (C.1 - A.1, C.2 - A.2) = (F.1 - A.1, F.2 - A.2)

def midpoint (A F M : ℝ × ℝ) : Prop := M = ((A.1 + F.1) / 2, (A.2 + F.2) / 2)

def line_equation (m b x y : ℝ) : Prop := x = m * y + b

def mid_point_on_line (m b : ℝ) (M : ℝ × ℝ) : Prop := line_equation m (1 - m) M.1 M.2

theorem bc_line_equation :
  ∀ (A F M : ℝ × ℝ) (p : ℝ),
  p = 2 → 
  A = (1, 2) → 
  F = (1, 0) → 
  midpoint A F M → 
  M = (1, 1) → 
  point_on_parabola p A → 
  vector_relation A _ _ F → 
  ∃ m b, line_equation m b = λ x y, 2 * x - y - 1 = 0 :=
by
  sorry

end bc_line_equation_l42_42922


namespace bee_distance_l42_42772

noncomputable def omega : ℂ := Complex.exp (-(Real.pi * Complex.I) / 4)

def P : ℕ → ℂ
| 0     := 0
| (n+1) := P n + (n+2) * (omega^n)

theorem bee_distance :
  Complex.abs (P 10) = (Real.sqrt 109 * Real.sqrt (2 + Real.sqrt 2)) / 2 :=
by
  sorry

end bee_distance_l42_42772


namespace range_distance_PA_l42_42126

def ellipse_point (x y : ℝ) : Prop := (x^2 / 9) + y^2 = 1
def fixed_point_A := (2 : ℝ, 0 : ℝ)

def distance_PA (x y : ℝ) : ℝ :=
  real.sqrt ((x - 2)^2 + y^2)

theorem range_distance_PA :
  ∀ x y, ellipse_point x y → ∃ a b, a = real.sqrt 2 / 2 ∧ b = 5 ∧ a ≤ distance_PA x y ∧ distance_PA x y ≤ b :=
begin
  sorry,
end

end range_distance_PA_l42_42126


namespace transformation_correct_l42_42982

noncomputable def original_function : ℝ → ℝ :=
  λ x, sin (x / 2 + π / 12)

theorem transformation_correct :
  ∀ (x : ℝ), (shift_right (shorten_abscissas f) (π / 3)) x = sin (x - π / 4) →
  f = original_function :=
by
  -- Assume the shifted and shortened function equals the given sine transformation
  intro x
  sorry

end transformation_correct_l42_42982


namespace pencils_calculation_l42_42010

variable (C B D : ℕ)

theorem pencils_calculation : 
  (C = B + 5) ∧
  (B = 2 * D - 3) ∧
  (C = 20) →
  D = 9 :=
by sorry

end pencils_calculation_l42_42010


namespace wade_tips_l42_42373

/-- Wade has a hot dog food truck. 
     He makes $2.00 in tips per customer.
     On Friday he served 28 customers.
     He served three times that amount of customers on Saturday.
     On Sunday, he served 36 customers.
     Prove that Wade made $296 in tips between the 3 days. -/
theorem wade_tips : 
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let customers_sunday := 36
  let tips_friday := tips_per_customer * customers_friday
  let tips_saturday := tips_per_customer * customers_saturday
  let tips_sunday := tips_per_customer * customers_sunday
  let total_tips := tips_friday + tips_saturday + tips_sunday
  in total_tips = 296 := 
by
  sorry

end wade_tips_l42_42373


namespace find_equation_of_C_find_equation_of_AB_l42_42215

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42215


namespace cost_of_first_variety_l42_42597

theorem cost_of_first_variety
  (cost_second : ℝ)
  (cost_mixture : ℝ)
  (ratio : ℝ)
  (approx_equal : ℝ → ℝ → Prop) : 
  cost_second = 8.75 →
  cost_mixture = 7.50 →
  ratio = 5 / 12 →
  approx_equal (cost_mixture - (cost_second - cost_mixture) * ratio) 6.98 :=
by
  intros cost_second_8.75 cost_mixture_7.50 ratio_5_12
  sorry

end cost_of_first_variety_l42_42597


namespace pentagon_area_AEDCB_l42_42644

-- Define the coordinate setup for points
variables {R : Type*} [linear_ordered_field R] {A B C D E : aff_pts R}

-- Define the conditions explicitly
variables (ABCD_square : is_square A B C D)
variables (perpendicular_AE_DE : perpendicular AE DE)
variables (length_AE : dist A E = 10)
variables (length_DE : dist D E = 10)

-- Define the theorem
theorem pentagon_area_AEDCB : area_pentagon A E D C B = 150 :=
by sorry


end pentagon_area_AEDCB_l42_42644


namespace find_original_function_l42_42968

theorem find_original_function
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₁ : ∀ x, g (x + (π / 3)) = sin (x - (π / 4)))
  (h₂ : ∀ x, f x = g (2 * x)) :
  f x = sin (x / 2 + π / 12) :=
sorry

end find_original_function_l42_42968


namespace gear_proportions_l42_42872

variables {x y z w : ℕ} {ω_A ω_B ω_C ω_D : ℝ}

theorem gear_proportions (h : x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D) :
  (ω_A : ω_B : ω_C : ω_D) = (yzw : xzw : xyw : xyz) :=
  sorry

end gear_proportions_l42_42872


namespace original_function_l42_42960

def f (x : ℝ) : ℝ := 
sorry

theorem original_function :
  (∀ x, (sin (x - π/4) = sin ((2 * (x - π / 3))))) ↔ (∀ x, f x = sin (x/2 + π/12)) :=
sorry

end original_function_l42_42960


namespace phoenix_hike_distance_l42_42634

variable (a b c d : ℕ)

theorem phoenix_hike_distance
  (h1 : a + b = 24)
  (h2 : b + c = 30)
  (h3 : c + d = 32)
  (h4 : a + c = 28) :
  a + b + c + d = 56 :=
by
  sorry

end phoenix_hike_distance_l42_42634


namespace prism_surface_area_eq_volume_l42_42419

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

def edge_lengths (y : ℝ) := (log_base 5 y, log_base 6 y, log_base 10 y)

def surface_area (y : ℝ) : ℝ :=
  let (a, b, c) := edge_lengths y
  2 * (a * b + a * c + b * c)

def volume (y : ℝ) : ℝ :=
  let (a, b, c) := edge_lengths y
  a * b * c

theorem prism_surface_area_eq_volume (y : ℝ) (h : surface_area y = volume y) : y = 90000 := sorry

end prism_surface_area_eq_volume_l42_42419


namespace arithmetic_sequence_inequality_l42_42525

variable {a : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality 
  (h : is_arithmetic_sequence a d)
  (d_pos : d ≠ 0)
  (a_pos : ∀ n, a n > 0) :
  (a 1) * (a 8) < (a 4) * (a 5) := 
by
  sorry

end arithmetic_sequence_inequality_l42_42525


namespace sequence_noncongruent_modulo_l42_42692

theorem sequence_noncongruent_modulo 
  (a : ℕ → ℕ)
  (h0 : a 1 = 1)
  (h1 : ∀ n, a (n + 1) = a n + 2^(a n)) :
  ∀ (i j : ℕ), i ≠ j → i ≤ 32021 → j ≤ 32021 →
  (a i) % (3^2021) ≠ (a j) % (3^2021) := 
by
  sorry

end sequence_noncongruent_modulo_l42_42692


namespace compare_neg_fractions_l42_42012

theorem compare_neg_fractions : (- (3 : ℚ) / 5) > (- (3 : ℚ) / 4) := sorry

end compare_neg_fractions_l42_42012


namespace inversely_proportional_y_ratio_l42_42661

variable {k : ℝ}
variable {x₁ x₂ y₁ y₂ : ℝ}
variable (h_inv_prop : ∀ (x y : ℝ), x * y = k)
variable (hx₁x₂ : x₁ ≠ 0 ∧ x₂ ≠ 0)
variable (hy₁y₂ : y₁ ≠ 0 ∧ y₂ ≠ 0)
variable (hx_ratio : x₁ / x₂ = 3 / 4)

theorem inversely_proportional_y_ratio :
  y₁ / y₂ = 4 / 3 :=
by
  sorry

end inversely_proportional_y_ratio_l42_42661


namespace problem1_problem2_l42_42262

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42262


namespace dice_surface_sum_l42_42755

noncomputable def surface_sum_of_dice (n : ℕ) := 28175 + 2 * n

theorem dice_surface_sum (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6):
  ∃ s, s ∈ {28177, 28179, 28181, 28183, 28185, 28187} ∧ s = surface_sum_of_dice X := by
  use surface_sum_of_dice X
  have : 28175 + 2 * X ∈ {28177, 28179, 28181, 28183, 28185, 28187} := by
    interval_cases X
    all_goals simp [surface_sum_of_dice]
  exact ⟨this, rfl⟩
sorry

end dice_surface_sum_l42_42755


namespace one_cubic_foot_is_1728_cubic_inches_l42_42097

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end one_cubic_foot_is_1728_cubic_inches_l42_42097


namespace parallel_vectors_k_value_l42_42093

theorem parallel_vectors_k_value :
  (∃ k : ℝ, let a := (2 : ℝ, -1 : ℝ) in
           let b := (k, 5 / 2) in
           a.1 * b.2 = a.2 * b.1) ↔ k = -5 :=
by
  sorry

end parallel_vectors_k_value_l42_42093


namespace mn_equals_am_plus_bn_perimeter_opq_equals_ab_l42_42704

-- Definition of conditions
variables {A B C O M N P Q : Point}

-- Assume O is the intersection of the angle bisectors of triangle ABC
axiom angle_bisector_intersection (O : Point) : is_incenter O A B C

-- Lines through O are parallel to sides of triangle ABC
axiom lines_parallel_to_sides (O A B C M N P Q : Point) :
  (line_through O parallel_to (line_through A B) ∧ 
   line_through O parallel_to (line_through A C) ∧ 
   line_through O parallel_to (line_through B C) ∧
   intersection (line_through O parallel_to (line_through A C)) (line_through A B) = P ∧
   intersection (line_through O parallel_to (line_through B C)) (line_through A B) = Q ∧
   intersection (line_through O parallel_to (line_through A B)) (line_through A C) = M ∧
   intersection (line_through O parallel_to (line_through A B)) (line_through B C) = N)

-- Prove that MN = AM + BN
theorem mn_equals_am_plus_bn (O A B C M N P Q : Point)
  (h1 : is_incenter O A B C)
  (h2 : lines_parallel_to_sides O A B C M N P Q) :
  distance M N = distance A M + distance B N :=
sorry

-- Prove that the perimeter of triangle OPQ is equal to the length of segment AB
theorem perimeter_opq_equals_ab (O A B C M N P Q : Point)
  (h1 : is_incenter O A B C)
  (h2 : lines_parallel_to_sides O A B C M N P Q) :
  perimeter (triangle O P Q) = distance A B :=
sorry

end mn_equals_am_plus_bn_perimeter_opq_equals_ab_l42_42704


namespace value_of_fraction_l42_42947

theorem value_of_fraction (x y : ℤ) (h : x / y = 7 / 2) : (x - 2 * y) / y = 3 / 2 := by
  sorry

end value_of_fraction_l42_42947


namespace math_problem_l42_42070

noncomputable def chord_length 
  (l : ℝ → ℝ → Prop) 
  (C : ℝ → ℝ → Prop) : ℝ :=
  let center := (2 : ℝ, -4 : ℝ)
  let radius := 2 * Real.sqrt 5
  let dist := 
    let (x₀, y₀) := center
    Real.abs (1 * x₀ + 2 * y₀ + 1) / Real.sqrt (1^2 + 2^2)
  let r := radius
  in 2 * Real.sqrt (r^2 - dist^2)

theorem math_problem :
  ∀ (l : ℝ → ℝ → Prop) 
    (C : ℝ → ℝ → Prop),
  -- Conditions
  (l = λ x y, x + 2 * y + 1 = 0) ∧
  (C = λ x y, x^2 + y^2 - 4 * x + 8 * y = 0) →
  -- Prove chord length
  chord_length l C = 2 * Real.sqrt 15 :=
begin
  -- Proof not required
  sorry
end

end math_problem_l42_42070


namespace find_equation_of_C_find_equation_of_AB_l42_42212

-- Definitions and conditions for the proof
def parabola (p : ℝ) : Prop :=
  p > 0

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def point_D (p : ℝ) : ℝ × ℝ :=
  (p, 0)

def intersection_points (p : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) → Prop :=
  λ M N, ∃ (x1 y1 x2 y2 : ℝ),
    M = (x1, y1) ∧
    N = (x2, y2) ∧
    y1 ^ 2 = 2 * p * x1 ∧
    y2 ^ 2 = 2 * p * x2

def perpendicular_MD (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

def distance_MF (F M : ℝ × ℝ) : ℝ :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2

-- Lean theorem statements
theorem find_equation_of_C (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N) :
  ∃ C, ∀ x y, C y = x ↔ y^2 = 4 * x :=
by
  sorry

theorem find_equation_of_AB (p : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hp : p > 0)
  (hMF : distance_MF (focus_of_parabola p) M = 9)
  (hMD_perp : perpendicular_MD p M)
  (hMN : intersection_points p M N)
  (hAB : intersection_points p A B)
  (hmax_alpha_beta : true) :  -- assuming true, replace with the actual condition for maximum α-β
  ∃ AB, ∀ x y, AB y = x ↔ x - sqrt (2 : ℝ) * y - 4 = 0 :=
by
  sorry


end find_equation_of_C_find_equation_of_AB_l42_42212


namespace maximum_PM_minus_PN_l42_42794

noncomputable def x_squared_over_9_minus_y_squared_over_16_eq_1 (x y : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / 16) = 1

noncomputable def circle1 (x y : ℝ) : Prop :=
  (x + 5)^2 + y^2 = 4

noncomputable def circle2 (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 1

theorem maximum_PM_minus_PN :
  ∀ (P M N : ℝ × ℝ),
    x_squared_over_9_minus_y_squared_over_16_eq_1 P.1 P.2 →
    circle1 M.1 M.2 →
    circle2 N.1 N.2 →
    (|dist P M - dist P N| ≤ 9) := sorry

end maximum_PM_minus_PN_l42_42794


namespace arc_length_parametric_curve_l42_42823

noncomputable def arc_length : ℝ :=
  (1 / 3) * (Real.sqrt_real.to_coe.real.pow_coe 3 / 2 (π ^ 2 + 4)) - (8 / 3)

theorem arc_length_parametric_curve :
  ∀ (t : ℝ), 0 ≤ t ∧ t ≤ π →
  let x := (t^2 - 2) * Real.sin t + 2 * t * Real.cos t in
  let y := (2 - t^2) * Real.cos t + 2 * t * Real.sin t in
  let x' := t^2 * Real.cos t + 2 * t * Real.sin t in
  let y' := t^2 * Real.sin t - 2 * t * Real.cos t in
  let dl := λ t, t * Real.sqrt (t^2 + 4) in
  ∫ t in 0..π, dl t = arc_length := by
  sorry

end arc_length_parametric_curve_l42_42823


namespace equation_of_parabola_equation_of_line_AB_l42_42250

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42250


namespace factorize_poly_l42_42724

theorem factorize_poly : 
  (x : ℤ) → (x^12 + x^6 + 1) = (x^4 - x^2 + 1) * (x^8 + x^4 + 1) :=
by
  sorry

end factorize_poly_l42_42724


namespace part1_part2_l42_42924

open Set

noncomputable def A (a : ℝ) := {-4, 2 * a - 1, a ^ 2}
noncomputable def B (a : ℝ) := {a - 5, 1 - a, 9}

theorem part1 (a : ℝ) (h1: 9 ∈ A a ∩ B a) : a = 5 ∨ a = -3 :=
by sorry

theorem part2 (a : ℝ) (h2: {9} = A a ∩ B a) : a = -3 :=
by sorry

end part1_part2_l42_42924


namespace multiply_469111111_by_99999999_l42_42002

theorem multiply_469111111_by_99999999 :
  469111111 * 99999999 = 46911111053088889 :=
sorry

end multiply_469111111_by_99999999_l42_42002


namespace problem1_problem2_l42_42266

-- Problem 1: Equation of the parabola
theorem problem1 (p : ℝ) (h : p > 0) (F : ℝ) (D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hD : D = (p, 0)) (hL_perpendicular : true) (hMF_dist : dist M F = 3) :
  C = fun (x y : ℝ) => y^2 = 4 * x :=
sorry

-- Problem 2: Equation of line AB when the inclination difference reaches its maximum value
theorem problem2 (α β : ℝ) (M N A B : ℝ × ℝ) 
  (h_parabola : fun (x y : ℝ) => y^2 = 4 * x)
  (h_max_inclination_diff : true) :
  ∃ k b, ∀ (x y : ℝ), (x, y) ∈ AB ↔ y = k * x + b :=
sorry

end problem1_problem2_l42_42266


namespace polynomial_equivalence_l42_42035

noncomputable def P (x : ℝ) : ℝ := 
  c * (x - 1) * x * (x + 1) * (x + 2) * (x^2 + x + 1)

theorem polynomial_equivalence (x c : ℝ) :
  (x^3 + 3 * x^2 + 3 * x + 2) * P (x - 1) = (x^3 - 3 * x^2 + 3 * x - 2) * P x := by
sory

end polynomial_equivalence_l42_42035


namespace harry_worked_34_hours_l42_42846

noncomputable def Harry_hours_worked (x : ℝ) : ℝ := 34

theorem harry_worked_34_hours (x : ℝ)
  (H : ℝ) (James_hours : ℝ) (Harry_pay James_pay: ℝ) 
  (h1 : Harry_pay = 18 * x + 1.5 * x * (H - 18)) 
  (h2 : James_pay = 40 * x + 2 * x * (James_hours - 40)) 
  (h3 : James_hours = 41) 
  (h4 : Harry_pay = James_pay) : 
  H = Harry_hours_worked x :=
by
  sorry

end harry_worked_34_hours_l42_42846


namespace perpendicular_lines_l42_42720

section
variables {l1 l2 : Type} 

-- Condition A: Slope of l1 is 1, slope of l2 is 1
def condition_A (k1 k2 : ℝ) : Prop :=
  k1 = 1 ∧ k2 = 1

-- Condition B: Slope of l1 is -√3/3, l2 passes through points (2,0) and (3,√3)
def condition_B (k1 : ℝ) (A B : ℝ × ℝ) : Prop :=
  k1 = - (real.sqrt 3) / 3 ∧ A = (2,0) ∧ B = (3, real.sqrt 3)

-- Condition C: l1 passes through points (2,1) and (-4,-5), and l2 passes through (1,2) and (1,0)
def condition_C (P Q M N : ℝ × ℝ) : Prop :=
  P = (2,1) ∧ Q = (-4,-5) ∧ M = (-1,2) ∧ N = (1,0)
  
-- Condition D: Direction vectors
def condition_D (m : ℝ) : Prop :=
  (1, m) ∧ (1, -1 / m)

-- Perpendicular if and only if correct conditions are met (Option B, C, D)
theorem perpendicular_lines (k1 k2 : ℝ) (A B P Q M N : ℝ × ℝ) (m : ℝ) :
  (∃ k1 k2, condition_A k1 k2 ∧ ¬ (k1 * k2 = -1)) →
  (∃ k1 A B, condition_B k1 A B ∧ k1 * ((B.2 - A.2) / (B.1 - A.1)) = -1) →
  (∃ P Q M N, condition_C P Q M N ∧ (((Q.2 - P.2) / (Q.1 - P.1)) * ((N.2 - M.2) / (N.1 - M.1)) = -1)) →
  (∃ m, condition_D m ∧ (1 - (1 / m)) = 0) →
  true :=
by exactly sorry
end

end perpendicular_lines_l42_42720


namespace find_width_l42_42697

variable (L W : ℕ)

def perimeter (L W : ℕ) : ℕ := 2 * L + 2 * W

theorem find_width (h1 : perimeter L W = 46) (h2 : W = L + 7) : W = 15 :=
sorry

end find_width_l42_42697


namespace max_value_of_f_l42_42533

noncomputable def f (x : ℝ) : ℝ := 
  (1 / 2) * (Real.sin x) ^ 2 + 
  (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 
  (Real.cos x) ^ 2

def x_vals (k : ℤ) : ℝ := k * Real.pi + Real.pi / 6

theorem max_value_of_f (x : ℝ) (k : ℤ) : 
  f (x_vals k) = 5 / 4 := 
sorry

end max_value_of_f_l42_42533


namespace rearrange_to_divisible_by_eleven_l42_42632

theorem rearrange_to_divisible_by_eleven :
  ∀ (n : ℕ) (d : List ℕ),
    d.length = 20 ∧ 
    (d.count 1) = 10 ∧ 
    (d.count 2) = 10 ∧ 
    (∀ i j, i < j → d[i] ≠ d[j] → reverse_sublist i j d = {l : List ℕ | l.reverse}) →
    (∃ l : List ℕ, is_valid_rearrangement l d ∧ (list_to_nat l % 11 = 0)) := 
sorry

end rearrange_to_divisible_by_eleven_l42_42632


namespace no_subsequence_limit_l42_42663

open MeasureTheory

noncomputable theory

variables {f : ℕ → ℝ → ℝ} (h1 : ∀ m n, (∫ x in 0..1, f m x * f n x) = if n = m then 1 else 0)
          (h2 : ∃ C : ℝ, ∀ n x, x ∈ Icc (0 : ℝ) 1 → |f n x| ≤ C)

theorem no_subsequence_limit :
  ¬(∃ (nk : ℕ → ℕ), strictMono nk ∧ ∃ g : ℝ → ℝ, tendsto (λ k, f (nk k)) atTop (𝓝 g)) :=
sorry

end no_subsequence_limit_l42_42663


namespace distinct_even_numbers_between_100_and_999_l42_42934

def count_distinct_even_numbers_between_100_and_999 : ℕ :=
  let possible_units_digits := 5 -- {0, 2, 4, 6, 8}
  let possible_hundreds_digits := 8 -- {1, 2, ..., 9} excluding the chosen units digit
  let possible_tens_digits := 8 -- {0, 1, 2, ..., 9} excluding the chosen units and hundreds digits
  possible_units_digits * possible_hundreds_digits * possible_tens_digits

theorem distinct_even_numbers_between_100_and_999 : count_distinct_even_numbers_between_100_and_999 = 320 :=
  by sorry

end distinct_even_numbers_between_100_and_999_l42_42934


namespace number_of_possible_multisets_l42_42018

theorem number_of_possible_multisets
  (b : Fin 9 → ℤ)
  (roots_p : Multiset ℤ)
  (roots_q : Multiset ℤ)
  (h_roots_p : ∀ s ∈ roots_p, Polynomial.eval s (Polynomial.of_multiset (b 8 :: b 7 :: b 6 :: b 5 :: b 4 :: b 3 :: b 2 :: b 1 :: b 0 :: [])) = 0)
  (h_roots_q : ∀ s ∈ roots_q, Polynomial.eval s (Polynomial.of_multiset (b 0 :: b 1 :: b 2 :: b 3 :: b 4 :: b 5 :: b 6 :: b 7 :: b 8 :: [])) = 0)
  (h_same_roots : roots_p = roots_q) :
  roots_p = (Multiset.replicate 1 1) + (Multiset.replicate (8 - 1) (-1)) ∨
  roots_p = (Multiset.replicate 2 1) + (Multiset.replicate (8 - 2) (-1)) ∨
  roots_p = (Multiset.replicate 3 1) + (Multiset.replicate (8 - 3) (-1)) ∨
  roots_p = (Multiset.replicate 4 1) + (Multiset.replicate (8 - 4) (-1)) ∨
  roots_p = (Multiset.replicate 5 1) + (Multiset.replicate (8 - 5) (-1)) ∨
  roots_p = (Multiset.replicate 6 1) + (Multiset.replicate (8 - 6) (-1)) ∨
  roots_p = (Multiset.replicate 7 1) + (Multiset.replicate (8 - 7) (-1)) ∨
  roots_p = (Multiset.replicate 8 1) + (Multiset.replicate (8 - 8) (-1)) ∨
  roots_p = (Multiset.replicate 0 1) + (Multiset.replicate 8 (-1)) :=
sorry

end number_of_possible_multisets_l42_42018


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42279

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42279


namespace original_function_l42_42988

theorem original_function :
  ∃ f : ℝ → ℝ,
  (∀ x : ℝ, sin (x - π / 4) = f (2 * (x - π / 3))) →
  (∀ x : ℝ, f x = sin (x / 2 + π / 12)) :=
by
  sorry

end original_function_l42_42988


namespace trigonometric_identity_l42_42884

open Real

variable (α : ℝ)
variable (h1 : π < α)
variable (h2 : α < 2 * π)
variable (h3 : cos (α - 7 * π) = -3 / 5)

theorem trigonometric_identity :
  sin (3 * π + α) * tan (α - 7 * π / 2) = 3 / 5 :=
by
  sorry

end trigonometric_identity_l42_42884


namespace effective_annual_rate_proof_l42_42335

/-
Proof Problem:
Prove that the effective annual rate (EAR) corresponding to a nominal rate of 6% per annum 
payable half-yearly is 6.09%, given the conditions.
-/

namespace InterestRate

def nominal_rate : ℝ := 0.06
def compounding_periods : ℕ := 2
def time_in_years : ℝ := 1

def effective_annual_rate (i : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  (1 + i / n) ^ (n * t) - 1

theorem effective_annual_rate_proof :
  effective_annual_rate nominal_rate compounding_periods time_in_years = 0.0609 :=
by
  sorry

end InterestRate

end effective_annual_rate_proof_l42_42335


namespace bert_toy_phones_l42_42438

theorem bert_toy_phones (P : ℕ) (berts_price_per_phone : ℕ) (berts_earning : ℕ)
                        (torys_price_per_gun : ℕ) (torys_earning : ℕ) (tory_guns : ℕ)
                        (earnings_difference : ℕ)
                        (h1 : berts_price_per_phone = 18)
                        (h2 : torys_price_per_gun = 20)
                        (h3 : tory_guns = 7)
                        (h4 : torys_earning = tory_guns * torys_price_per_gun)
                        (h5 : berts_earning = torys_earning + earnings_difference)
                        (h6 : earnings_difference = 4)
                        (h7 : P = berts_earning / berts_price_per_phone) :
  P = 8 := by sorry

end bert_toy_phones_l42_42438


namespace union_of_sets_l42_42295

open Set

theorem union_of_sets :
  let U := {1, 2, 3, 4, 5, 6}
  let S := {1, 3, 5}
  let T := {2, 3, 4, 5}
  S ∪ T = {1, 2, 3, 4, 5} :=
by
  let U := {1, 2, 3, 4, 5, 6}
  let S := {1, 3, 5}
  let T := {2, 3, 4, 5}
  show S ∪ T = {1, 2, 3, 4, 5}
  sorry

end union_of_sets_l42_42295


namespace problem_1_problem_2_l42_42201

variable {p : ℝ}
variable {x y : ℝ}
variable {α β : ℝ}

/-- Equation of the given parabola -/
def parabola_eq (p : ℝ) : Prop := y^2 = 2 * p * x

/-- Point D on the parabola -/
def point_D (p : ℝ) : Prop := x = p ∧ y = 0

/-- |MF| given as 3 -/
def mf_eq_3 (p : ℝ) : Prop := ∃ x y, dist (x, y) (p, 0) = 3

/-- Maximum of α - β condition -/
def max_alpha_beta (α β : ℝ) : Prop := α - β = (π / 4) -- This is a general placeholder and should reflect the maximum condition

theorem problem_1 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p) : parabola_eq 2 := 
sorry

theorem problem_2 (h1: 0 < p)
    (h2 : parabola_eq p)
    (h3 : point_D p)
    (h4 : mf_eq_3 p)
    (h5 : max_alpha_beta α β) : 
    ∀ x y, parabola_eq 2 → (x - sqrt 2 * y - 4 = 0) :=
sorry

end problem_1_problem_2_l42_42201


namespace conditional_probability_correct_l42_42664

noncomputable def total_products : ℕ := 8
noncomputable def first_class_products : ℕ := 6
noncomputable def chosen_products : ℕ := 2

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def P_A : ℚ := 1 - (combination first_class_products chosen_products) / (combination total_products chosen_products)
noncomputable def P_AB : ℚ := (combination 2 1 * combination first_class_products 1) / (combination total_products chosen_products)

noncomputable def conditional_probability : ℚ := P_AB / P_A

theorem conditional_probability_correct :
  conditional_probability = 12 / 13 :=
  sorry

end conditional_probability_correct_l42_42664


namespace smallest_number_when_diminished_by_7_is_divisible_l42_42390

-- Variables for divisors
def divisor1 : Nat := 12
def divisor2 : Nat := 16
def divisor3 : Nat := 18
def divisor4 : Nat := 21
def divisor5 : Nat := 28

-- The smallest number x which, when diminished by 7, is divisible by the divisors.
theorem smallest_number_when_diminished_by_7_is_divisible (x : Nat) : 
  (x - 7) % divisor1 = 0 ∧ 
  (x - 7) % divisor2 = 0 ∧ 
  (x - 7) % divisor3 = 0 ∧ 
  (x - 7) % divisor4 = 0 ∧ 
  (x - 7) % divisor5 = 0 → 
  x = 1015 := 
sorry

end smallest_number_when_diminished_by_7_is_divisible_l42_42390


namespace find_other_number_l42_42433

theorem find_other_number (x y : ℤ) (h : 3 * x + 4 * y = 161) (h₁: x = 17 ∨ y = 17) : x = 31 ∨ y = 31 :=
by
  cases h₁
  case mp x17 =>
    have : 3 * 17 + 4 * y = 161 := by rw [h₁] at h; exact y
    have : 51 + 4 * y = 161 := by norm_num [h₁] at h; rw h at (51) (4 * y ); rw
    have : 4 * y = 110 := by exact ⟨110⟩; rw h at (61) (4 * y ); rw (110) at 110
    have : y = 110 / 4 := by norm_num [h₁]; rw [4 * y] (110) exact y
    have : y = 27.5 := by norm_num /4[27.5] exact

  case mp y17 =>
    have : 3 * x + 4 * 17 = 161 := by rw [h₁] exact 17
    have : 3 * x + 68 = 161 := by norm_num 17 exact 17
    have : 3 * x = 93 := by exact 93 rw  (93 68); rw;
    have : x = 93 / 3 := by norm_num  3 rw 93 exact x;
    have : x = 31 := by norm_num ⟨31⟩;

end find_other_number_l42_42433


namespace parallelogram_area_l42_42385

theorem parallelogram_area (base height : ℕ) (h_base : base = 36) (h_height : height = 24) : base * height = 864 := by
  sorry

end parallelogram_area_l42_42385


namespace F_value_1_F_value_2_l42_42606

open Function

-- Define the function F with given conditions
def F (x : ℝ) : ℝ := sorry

lemma F_increasing (x y : ℝ) (hx : 0 ≤ x) (hy : x ≤ y) (hxy : y ≤ 1) : F x ≤ F y :=
sorry

lemma F_condition_1 (x : ℝ) (hx : 0 ≤ x) (hxy : x ≤ 1) : F (x / 3) = F x / 2 :=
sorry

lemma F_condition_2 (x : ℝ) (hx : 0 ≤ x) (hxy : x ≤ 1) : F (1 - x) = 1 - F x :=
sorry

theorem F_value_1 : F (173 / 1993) = 3 / 16 :=
by
  apply F (173 / 1993)
  apply sorry

theorem F_value_2 : F (1 / 13) = 1 / 7 :=
by
  apply F (1 / 13)
  apply sorry

end F_value_1_F_value_2_l42_42606


namespace exists_increasing_sequences_l42_42652

theorem exists_increasing_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) :
  (∀ n : ℕ, a n < a (n + 1)) ∧ (∀ n : ℕ, b n < b (n + 1)) ∧
  (∀ n : ℕ, a n * (a n + 1) ∣ b n ^ 2 + 1) :=
sorry

end exists_increasing_sequences_l42_42652


namespace geometric_sequence_value_l42_42145

variable {a_n : ℕ → ℝ}

-- Condition: {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given: a_1 a_2 a_3 = -8
variable (a1 a2 a3 : ℝ) (h_seq : is_geometric_sequence a_n)
variable (h_cond : a1 * a2 * a3 = -8)

-- Prove: a2 = -2
theorem geometric_sequence_value : a2 = -2 :=
by
  -- Proof will be provided later
  sorry

end geometric_sequence_value_l42_42145


namespace option_B_is_irrational_l42_42428

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define the options
def option_A : ℝ := 5 / 7
def option_B : ℝ := real.of_cauchy (λ n, if (∃ k, n = 1 + k * (k + 1) / 2) then 1 else 0)
def option_C : ℝ := -2
def option_D : ℝ := 37 / 10

theorem option_B_is_irrational : is_irrational option_B := 
by {
  sorry
}

end option_B_is_irrational_l42_42428


namespace find_alpha_find_length_MN_l42_42886

section geometry_problem

variables {t α x y : ℝ}

-- Given Conditions
def line_l (t α : ℝ) : ℝ × ℝ := 
(-3 + t * cos α, sqrt 3 + t * sin α)

def curve_C (ρ : ℝ) : Prop :=
ρ = 2 * sqrt 3

def length_AB : ℝ := 2 * sqrt 3

-- Part (1): Determine the value of α
theorem find_alpha (h₁ : 0 ≤ α) (h₂ : α < π) (h₃ : α ≠ π / 2)
  (h₄ : ∀ A B, (A, B) ∈ set.range (λ t, line_l t α) ∧ curve_C ρ → 
               dist A B = length_AB) : 
  α = π / 6 :=
by sorry

-- Part (2): Find the length of |MN|
theorem find_length_MN 
  (h₁ : 0 ≤ α) (h₂ : α < π) (h₃ : α ≠ π / 2)
  (h₄ : ∀ A B, (A, B) ∈ set.range (λ t, line_l t α) ∧ curve_C ρ → 
               dist A B = length_AB) 
  (hα : α = π / 6) : 
  ∃ M N, dist M N = 4 :=
by sorry

end geometry_problem

end find_alpha_find_length_MN_l42_42886


namespace count_distinct_m_in_right_triangle_l42_42830

theorem count_distinct_m_in_right_triangle (k : ℝ) (hk : k > 0) :
  ∃! m : ℝ, (m = -3/8 ∨ m = -3/4) :=
by
  sorry

end count_distinct_m_in_right_triangle_l42_42830


namespace constant_term_expansion_l42_42462

/-- Define the binomial expansion general term -/
def general_term (n r : ℕ) (a b : ℚ) : ℚ := (a^(n-r)) * (b^r) * (nat.choose n r)

/-- Define the given binomial expansion -/
def binomial_expansion : ℚ := general_term 6 4 (√5 / 5) 1

theorem constant_term_expansion : 
  binomial_expansion = 3 := by
  sorry

end constant_term_expansion_l42_42462


namespace sum_of_surface_points_l42_42758

theorem sum_of_surface_points
  (n : ℕ) (h_n : n = 2012) 
  (total_sum : ℕ) (h_total : total_sum = n * 21)
  (matching_points_sum : ℕ) (h_matching : matching_points_sum = (n - 1) * 7)
  (x : ℕ) (h_x_range : 1 ≤ x ∧ x ≤ 6) :
  (total_sum - matching_points_sum + 2 * x = 28177 ∨
   total_sum - matching_points_sum + 2 * x = 28179 ∨
   total_sum - matching_points_sum + 2 * x = 28181 ∨
   total_sum - matching_points_sum + 2 * x = 28183 ∨
   total_sum - matching_points_sum + 2 * x = 28185 ∨
   total_sum - matching_points_sum + 2 * x = 28187) :=
by sorry

end sum_of_surface_points_l42_42758


namespace relationship_between_a_and_b_l42_42879

theorem relationship_between_a_and_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x : ℝ, |(2 * x + 2)| < a → |(x + 1)| < b) : b ≥ a / 2 :=
by
  -- The proof steps will be inserted here
  sorry

end relationship_between_a_and_b_l42_42879


namespace dice_surface_sum_l42_42757

noncomputable def surface_sum_of_dice (n : ℕ) := 28175 + 2 * n

theorem dice_surface_sum (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6):
  ∃ s, s ∈ {28177, 28179, 28181, 28183, 28185, 28187} ∧ s = surface_sum_of_dice X := by
  use surface_sum_of_dice X
  have : 28175 + 2 * X ∈ {28177, 28179, 28181, 28183, 28185, 28187} := by
    interval_cases X
    all_goals simp [surface_sum_of_dice]
  exact ⟨this, rfl⟩
sorry

end dice_surface_sum_l42_42757


namespace equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42272

variables (p : ℝ) (F : ℝ × ℝ) (D M N A B : ℝ × ℝ) (α β : ℝ)

-- We need noncomputable for sqrt and other real number operations
noncomputable theory

-- Part 1: Proving the equation of the parabola
theorem equation_of_parabola (hp_pos : p > 0)
  (hC : ∀ x y, y^2 = 2 * p * x ↔ lies_on_parabola_C (x, y))
  (hD : D = (p, 0))
  (hF : F = (p / 2, 0))
  (hMD_perpendicular : M.1 = D.1 ∧ M.2 ≠ D.2)
  (hMF_eq_3 : dist M F = 3) :
  ∀ x y, y^2 = 4 * x ↔ lies_on_parabola_C (x, y) := sorry

-- Part 2: Proving the equation of line AB
theorem equation_of_line_AB_max_alpha_minus_beta 
  (hD : D = (2 : ℝ, 0))
  (hF : F = (1 : ℝ, 0))
  (hαβ_max :
    (lMN_slope_eq : ∀ y1 y2 : ℝ, tan α = 4 / (y1 + y2))
    (eq_y_y4 : ∀ y2 y4 : ℝ, y2 * y4 = -8)
    (eq_y3_y1 : ∀ y1 y3 : ℝ, y3 = -8 / y1)
    (tan_beta : ∀ y3 y4 : ℝ, tan β = 4 / (y3 + y4))) :
  line_eq_AB = "x - sqrt(2) * y - 4 = 0" := sorry

end equation_of_parabola_equation_of_line_AB_max_alpha_minus_beta_l42_42272


namespace average_student_teacher_diff_l42_42799

def number_of_students : ℕ := 120
def number_of_teachers : ℕ := 5
def class_sizes : List ℕ := [40, 30, 20, 15, 15]

def average_students_per_teacher : ℕ := class_sizes.sum / number_of_teachers

def average_class_size_per_student : ℚ :=
  (40 * (40 / 120) + 30 * (30 / 120) + 20 * (20 / 120) + 15 * (15 / 120) + 15 * (15 / 120)) 
  / 1

theorem average_student_teacher_diff :
  let t := average_students_per_teacher in
  let s := average_class_size_per_student in
  (t : ℚ) - s = -3.92 :=
by 
  sorry

end average_student_teacher_diff_l42_42799


namespace marbles_steve_now_l42_42313
-- Import necessary libraries

-- Define the initial conditions as given in a)
def initial_conditions (sam steve sally : ℕ) := sam = 2 * steve ∧ sally = sam - 5 ∧ sam - 6 = 8

-- Define the proof problem statement
theorem marbles_steve_now (sam steve sally : ℕ) (h : initial_conditions sam steve sally) : steve + 3 = 10 :=
sorry

end marbles_steve_now_l42_42313


namespace print_shop_Y_charge_l42_42489

noncomputable def charge_shop_x_per_copy : ℝ := 1.25

def total_charge_x (n : ℕ) (charge_per_copy : ℝ) : ℝ :=
  n * charge_per_copy

def total_charge_y (n : ℕ) (total_x : ℝ) (extra_charge : ℝ) : ℝ :=
  total_x + extra_charge

def charge_per_copy_y (total_y : ℝ) (n : ℕ) : ℝ :=
  total_y / n

theorem print_shop_Y_charge (n : ℕ) (extra_charge : ℝ) : 
  charge_shop_x_per_copy = 1.25 → 
  n = 80 → 
  extra_charge = 120 → 
  charge_per_copy_y (total_charge_y n (total_charge_x n charge_shop_x_per_copy) extra_charge) n = 2.75 :=
by
  sorry

end print_shop_Y_charge_l42_42489


namespace volume_tetrahedron_proof_l42_42743

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def A1 : Point := { x := 1, y := 2, z := 0 }
def A2 : Point := { x := 1, y := -1, z := 2 }
def A3 : Point := { x := 0, y := 1, z := -1 }
def A4 : Point := { x := -3, y := 0, z := 1 }

noncomputable def volume_of_tetrahedron (A1 A2 A3 A4 : Point) : ℝ :=
  let v1 := (A2.x - A1.x, A2.y - A1.y, A2.z - A1.z)
  let v2 := (A3.x - A1.x, A3.y - A1.y, A3.z - A1.z)
  let v3 := (A4.x - A1.x, A4.y - A1.y, A4.z - A1.z)
  let scalar_triple_product := v1.1 * (v2.2 * v3.3 - v2.3 * v3.2) -
                               v1.2 * (v2.1 * v3.3 - v2.3 * v3.1) +
                               v1.3 * (v2.1 * v3.2 - v2.2 * v3.1)
  (1 / 6) * |scalar_triple_product|

theorem volume_tetrahedron_proof : volume_of_tetrahedron A1 A2 A3 A4 = 19 / 6 :=
by
  sorry

end volume_tetrahedron_proof_l42_42743


namespace rationalized_factor_sqrt_13_rationalized_factor_sqrt_7_plus_sqrt_5_rationalize_denominator_3_over_sqrt_15_rationalize_denominator_2_over_sqrt_5_plus_sqrt_3_series_simplification_l42_42645

theorem rationalized_factor_sqrt_13 :
  ∃ x : ℝ, (¬(√13 = 0)) ∧ (√13 * x = 13) :=
begin
  use √13,
  split,
  { exact real.sqrt_ne_zero (by norm_num), },
  { rw [mul_self_sqrt],
    exact rfl,
    exact le_of_lt (real.sqrt_pos.2 (by norm_num)),
    exact le_of_lt (by norm_num) }
end

theorem rationalized_factor_sqrt_7_plus_sqrt_5 :
  ∃ y : ℝ, (¬((√7 + √5) = 0)) ∧ (((√7 + √5) * y) = 2) :=
begin
  use (√7 - √5),
  split,
  { intro h,
    have : √7 + √5 > 0 := add_pos (real.sqrt_pos.2 (by norm_num)) (real.sqrt_pos.2 (by norm_num)),
    linarith, },
  { ring_nf,
    repeat { rw [← sqrt_mul]},
    norm_num }
end

theorem rationalize_denominator_3_over_sqrt_15 :
  ∃ z : ℝ, (3 / √15) = z :=
begin
  use (√15 / 5),
  field_simp [real.sq_sqrt, ne_of_gt (real.sqrt_pos.2 (by norm_num))],
  norm_num
end

theorem rationalize_denominator_2_over_sqrt_5_plus_sqrt_3 :
  ∃ w : ℝ, (2 / (√5 + √3)) = w :=
begin
  use ((√5 - √3) / 2),
  field_simp [real.sq_sqrt],
  norm_num
end

theorem series_simplification : 
  (∑ n in finset.range (2022), 1 / (real.sqrt (↑n + 1) + real.sqrt (↑n + 2))) = real.sqrt 2023 - 1 :=
begin
  sorry
end

end rationalized_factor_sqrt_13_rationalized_factor_sqrt_7_plus_sqrt_5_rationalize_denominator_3_over_sqrt_15_rationalize_denominator_2_over_sqrt_5_plus_sqrt_3_series_simplification_l42_42645


namespace num_integers_abs_le_10_point_3_l42_42676

theorem num_integers_abs_le_10_point_3 : 
  {x : Int | |x| ≤ 10.3}.card = 21 := 
sorry

end num_integers_abs_le_10_point_3_l42_42676


namespace total_tips_l42_42371

def tips_per_customer := 2
def customers_friday := 28
def customers_saturday := 3 * customers_friday
def customers_sunday := 36

theorem total_tips : 
  (tips_per_customer * (customers_friday + customers_saturday + customers_sunday) = 296) :=
by
  sorry

end total_tips_l42_42371


namespace radius_of_circle_l42_42571

-- Define the problem condition
def diameter_of_circle : ℕ := 14

-- State the problem as a theorem
theorem radius_of_circle (d : ℕ) (hd : d = diameter_of_circle) : d / 2 = 7 := by 
  sorry

end radius_of_circle_l42_42571


namespace distance_from_house_to_sidewalk_l42_42310

/--
Rich walks some distance from his house to the sidewalk, then 200 feet down the sidewalk to the end of the road.
Then he makes a left and walks double his total distance so far until he reaches the next intersection. Then he
walks half the total distance up to this point again to the end of his route, before turning around and walking
the same path all the way back home. Given that Rich walked 1980 feet, prove that the distance from Rich's
house to the sidewalk is approximately 111 feet.
-/
theorem distance_from_house_to_sidewalk (x : ℝ) : 7 * x + 1200 = 1980 → x ≈ 111 := by
  sorry

end distance_from_house_to_sidewalk_l42_42310


namespace find_function_l42_42992

open Real

theorem find_function (f : ℝ → ℝ) :
  let g := λ x, f (x / 2);
  let h := λ x, g (x - π / 3);
  (∀ x, h x = sin (x - π / 4)) →
  ∀ x, f x = sin (x / 2 + π / 12) :=
by
  intros g h h_eq x
  sorry

end find_function_l42_42992


namespace proof_exists_real_constant_l42_42308

noncomputable def exists_real_constant (c : ℝ) : Prop :=
  ∀ (x y : ℝ), ∃ (m n : ℤ), (m.gcd n = 1) ∧ 
  (real.sqrt ((x - m)^2 + (y - n)^2) < c * real.log (x^2 + y^2 + 2))

theorem proof_exists_real_constant : ∃ (c : ℝ), exists_real_constant c :=
sorry

end proof_exists_real_constant_l42_42308


namespace visible_product_divisible_by_48_l42_42430

-- We represent the eight-sided die as the set {1, 2, 3, 4, 5, 6, 7, 8}.
-- Q is the product of any seven numbers from this set.

theorem visible_product_divisible_by_48 
   (Q : ℕ)
   (H : ∃ (numbers : Finset ℕ), numbers ⊆ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ) ∧ numbers.card = 7 ∧ Q = numbers.prod id) :
   48 ∣ Q :=
by
  sorry

end visible_product_divisible_by_48_l42_42430


namespace exists_set_max_reciprocals_sum_l42_42013

-- Definitions from the conditions:
def no_three_in_arith_progression (S : Set ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (a + c) ≠ 2 * b

def reciprocals_sum (S : Set ℕ) : ℚ :=
  S.to_finset.sum (λ s, (1 : ℚ) / s)

-- Main problem statement
theorem exists_set_max_reciprocals_sum (n : ℕ) :
  ∃ S : Set ℕ, S.card = n ∧ no_three_in_arith_progression S ∧
  ∀ T : Set ℕ, T.card = n ∧ no_three_in_arith_progression T → reciprocals_sum S ≥ reciprocals_sum T :=
sorry

end exists_set_max_reciprocals_sum_l42_42013


namespace distinct_real_roots_of_quadratic_eq_l42_42059

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem distinct_real_roots_of_quadratic_eq (k : ℝ) :
  (quadratic_discriminant 1 2 k > 0) ↔ (k < 1) :=
by
  -- Applying the definition of quadratic_discriminant
  have h : quadratic_discriminant 1 2 k = 4 - 4 * k := by
    simp [quadratic_discriminant]
  rw [h]
  -- Rewriting the inequality and solving for k
  exact (sub_pos.trans (div_lt_iff zero_lt_four)).symm

end distinct_real_roots_of_quadratic_eq_l42_42059


namespace prove_finite_transformations_l42_42836

def transformations (a : List ℤ) : List ℤ :=
  match a with
  | [] => []
  | h::t => List.zipWith (+) a (t ++ [h])

def is_multiple_of_k (k : ℤ) (a : List ℤ) : Prop :=
  ∀ x ∈ a, k ∣ x

def finite_transformations (n k : ℕ) : Prop :=
  ∀ a : List ℤ, a.length = n → ∃ t : ℕ, is_multiple_of_k k (Nat.iterate transformations t a)

theorem prove_finite_transformations : 
  ∀ (p q : ℕ), 2 ≤ p → 2 ≤ q → finite_transformations (2 ^ p) (2 ^ q) :=
by
  sorry

end prove_finite_transformations_l42_42836


namespace parabola_equation_line_AB_equation_l42_42193

-- Conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x → p > 0
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def M N line_through_F (p x1 y1 x2 y2 : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ x1 ≠ x2 ∧ (focus p, (x1, y1)), (focus p, (x2, y2))

-- Question 1
theorem parabola_equation (p : ℝ) (hp : parabola p) : ∃ p', y^2 = 4x := by
  sorry

-- Question 2
theorem line_AB_equation (p x1 y1 x2 y2 : ℝ) (hD : point_D p = (p, 0)) 
  (hline : M N line_through_F p x1 y1 x2 y2)
  (h_perpendicular : ∀ md : ℝ × ℝ → (ℝ × ℝ), (md (p, 0)) = (0, 1))
  (hMF : |M_focus_dist : ℝ × ℝ → ℝ, M_focus_dist ((x1, y1), (focus p)) = 3) : 
  ∃ (α β : ℝ), α - β = (1 / (2 * ((h_perpendicular (x1, y1) + h_perpendicular (x2, y2))⁻¹))) → 
    eqn_line_AB : ℝ × ℝ → ℝ := 
    (λ m, x - sqrt 2 * y - 4 = 0) := by
  sorry

end parabola_equation_line_AB_equation_l42_42193


namespace cubic_foot_to_cubic_inches_l42_42105

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end cubic_foot_to_cubic_inches_l42_42105


namespace cookies_in_each_package_l42_42043

theorem cookies_in_each_package (children : ℕ) (cookies_per_child : ℕ) (packages : ℕ) 
  (h_children : children = 5) (h_cookies_per_child : cookies_per_child = 15) (h_packages : packages = 3) : 
  (children * cookies_per_child) / packages = 25 := 
by
  rw [h_children, h_cookies_per_child, h_packages]
  norm_num
  rfl

end cookies_in_each_package_l42_42043


namespace solve_for_x_l42_42516

theorem solve_for_x (x y : ℝ) (h1 : 3 * x + y = 75) (h2 : 2 * (3 * x + y) - y = 138) : x = 21 :=
  sorry

end solve_for_x_l42_42516


namespace part1_eq_C_part2_line_AB_l42_42285

variables {p : ℝ}

-- Conditions
def parabola (p : ℝ) (p_pos : p > 0) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def point_D (p : ℝ) := (p, 0)
def points_MN (p : ℝ) := ∃ (M N : ℝ × ℝ), parabola p ∧ (∃ y, MD ⊥ x-axis) ∧ |MF| = 3

-- Proof Targets
def equation_C (p : ℝ) : Prop := (parabola p) = ∀ (x y : ℝ), y^2 = 4x
def line_AB : Prop := ∃ x y : ℝ, x - sqrt 2 * y - 4 = 0

-- Statements
theorem part1_eq_C (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p/2, 0)) (pointD: (p, 0)) (line_MN: points_MN p):
  equation_C p  := 
sorry

theorem part2_line_AB (p : ℝ) (h : parabola p) (h_pos: p > 0)
  (focus_F: (p / 2, 0)) (pointD: (p, 0)) (line_MN: points_MN p) 
  (alpha beta : ℝ) (h_max: ∀ α β, α - β = max_value) :
  line_AB :=
sorry

end part1_eq_C_part2_line_AB_l42_42285


namespace largestK_l42_42856

noncomputable def findLargestK : Nat :=
  let b : Fin 2009 → ℝ; -- lengths of blue sides in sorted order
  let r : Fin 2009 → ℝ; -- lengths of red sides in sorted order
  let w : Fin 2009 → ℝ; -- lengths of white sides in sorted order
  1

theorem largestK (b r w : Fin 2009 → ℝ) (h_b_sorted : ∀ i j, i ≤ j → b i ≤ b j)
  (h_r_sorted : ∀ i j, i ≤ j → r i ≤ r j) (h_w_sorted : ∀ i j, i ≤ j → w i ≤ w j) : 
  findLargestK = 1 := by
  sorry -- Proof steps are not included, only the statement is required

end largestK_l42_42856


namespace equivalent_single_discount_l42_42413

noncomputable def original_price : ℝ := 50
noncomputable def first_discount : ℝ := 0.25
noncomputable def coupon_discount : ℝ := 0.10
noncomputable def final_price : ℝ := 33.75

theorem equivalent_single_discount :
  (1 - final_price / original_price) * 100 = 32.5 :=
by
  sorry

end equivalent_single_discount_l42_42413


namespace alice_stops_in_quadrant_A_l42_42631

/-- Alice runs exactly one mile (5280 feet) on a circular track of circumference 60 feet,
starting from point S and stops. Prove that she stops in quadrant A. -/
theorem alice_stops_in_quadrant_A
  (circumference : ℕ)
  (distance_traveled : ℕ)
  (num_quadrants : ℕ)
  (stopping_point : string)
  (start : string)
  (quadrant_A : string) :
  circumference = 60 →
  distance_traveled = 5280 →
  num_quadrants = 4 →
  start = "S" →
  stopping_point = "S" →
  stopping_point = quadrant_A :=
by
  sorry

end alice_stops_in_quadrant_A_l42_42631


namespace find_c_l42_42808

theorem find_c (a b c : ℤ) (h1 : a + b * c = 2017) (h2 : b + c * a = 8) :
  c = -6 ∨ c = 0 ∨ c = 2 ∨ c = 8 :=
by 
  sorry

end find_c_l42_42808


namespace angle_A_eq_pi_div_4_side_c_eq_4_l42_42523

-- Define the problem conditions
variable {A B C : ℝ} [Fact (0 < A ∧ A < π)] [Fact (0 < B ∧ B < π)] [Fact (0 < C ∧ C < π)]
variable {a b c : ℝ} (h_a : a = b) (h_b : b = c) (h_c : tan B = (tan C + 1) / (tan C - 1))

-- Part (1): Prove that A = π/4 given the conditions
theorem angle_A_eq_pi_div_4 (h_tan_B : tan B = (tan C + 1) / (tan C - 1)) : A = π / 4 :=
sorry

-- Additional variables for Part (2)
variable (h_cos_B : cos B = (3 * sqrt 10) / 10) (h_b_sqrt2 : b = sqrt 2)

-- Part (2): Prove that c = 4 given the additional conditions
theorem side_c_eq_4 (h_cos_B : cos B = (3 * sqrt 10) / 10) (h_b : b = sqrt 2) : c = 4 :=
sorry

end angle_A_eq_pi_div_4_side_c_eq_4_l42_42523


namespace seventieth_element_in_s_l42_42387

/-- Define the set of positive integers that when divided by 8 have a remainder of 5 -/
def s : set ℕ := { x | ∃ n : ℕ, x = 8 * n + 5 }

/-- Prove that the 70th element in this set is 557 -/
theorem seventieth_element_in_s : 
  ∃ (n : ℕ), set.to_finset s ∈ finset.range(1, 70) → n = 69 ∧ 8 * 69 + 5 = 557 :=
  sorry

end seventieth_element_in_s_l42_42387


namespace num_distinct_sums_of_two_positive_cubes_less_than_300_l42_42941

theorem num_distinct_sums_of_two_positive_cubes_less_than_300 :
  {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ n = a^3 + b^3 ∧ n < 300}.to_finset.card = 19 := 
by
  sorry

end num_distinct_sums_of_two_positive_cubes_less_than_300_l42_42941


namespace trigonometric_identity_l42_42731

variable (α : ℝ)

theorem trigonometric_identity :
  2 * (sin (3 * π - 2 * α))^2 * (cos (5 * π + 2 * α))^2 =
  (1 / 4) - (1 / 4) * sin ((5 * π) / 2 - 8 * α) :=
sorry

end trigonometric_identity_l42_42731


namespace division_remainder_l42_42025

open Polynomial

noncomputable def p := X^4 - 3*X^2 + 1
noncomputable def q := X^3 - X - 1
noncomputable def r := -2*X^2 + X + 1

theorem division_remainder : (p % q) = r := by
  sorry

end division_remainder_l42_42025


namespace number_of_integer_terms_l42_42053

open Real

def combinatorial_term (n k : ℕ) : ℕ := Nat.choose n k

def sequence_term (n : ℕ) : Real :=
  (combinatorial_term 200 n) * (6 : Real) ^ ((200 - n) / 3) * (2 : Real) ^ (- n / 2)

def is_integer (x : Real) : Prop :=
  ∃ z : ℤ, x = z

def count_integer_terms_in_sequence (n_max : ℕ) : ℕ :=
  (Finset.filter (λ n, is_integer (sequence_term n)) (Finset.range (n_max + 1))).card

theorem number_of_integer_terms :
  count_integer_terms_in_sequence 95 = 15 := 
sorry

end number_of_integer_terms_l42_42053


namespace integer_values_not_satisfying_inequality_l42_42490

theorem integer_values_not_satisfying_inequality :
  (∃ x : ℤ, ¬(3 * x^2 + 17 * x + 28 > 25)) ∧ (∃ x1 x2 : ℤ, x1 = -2 ∧ x2 = -1) ∧
  ∀ x : ℤ, (x = -2 ∨ x = -1) -> ¬(3 * x^2 + 17 * x + 28 > 25) :=
by
  sorry

end integer_values_not_satisfying_inequality_l42_42490


namespace jamie_workday_percent_l42_42464

theorem jamie_workday_percent
  (total_work_hours : ℕ)
  (first_meeting_minutes : ℕ)
  (second_meeting_multiplier : ℕ)
  (break_minutes : ℕ)
  (total_minutes_per_hour : ℕ)
  (total_work_minutes : ℕ)
  (first_meeting_duration : ℕ)
  (second_meeting_duration : ℕ)
  (total_meeting_time : ℕ)
  (percentage_spent : ℚ) :
  total_work_hours = 10 →
  first_meeting_minutes = 60 →
  second_meeting_multiplier = 2 →
  break_minutes = 30 →
  total_minutes_per_hour = 60 →
  total_work_minutes = total_work_hours * total_minutes_per_hour →
  first_meeting_duration = first_meeting_minutes →
  second_meeting_duration = second_meeting_multiplier * first_meeting_duration →
  total_meeting_time = first_meeting_duration + second_meeting_duration + break_minutes →
  percentage_spent = (total_meeting_time : ℚ) / (total_work_minutes : ℚ) * 100 →
  percentage_spent = 35 :=
sorry

end jamie_workday_percent_l42_42464


namespace number_of_arrangements_l42_42820

theorem number_of_arrangements (teachers students : ℕ) 
    (positions : ℕ) (exactly_two_students_between_teachers : ℕ) :
    teachers = 2 ∧ students = 4 ∧ positions = 6 ∧ exactly_two_students_between_teachers = 3 → 
    (3 * 2 * 24 = 144) := 
begin
  sorry
end

end number_of_arrangements_l42_42820


namespace max_area_quad_l42_42610

noncomputable def MaxAreaABCD : ℝ :=
  let x : ℝ := 3
  let θ : ℝ := Real.pi / 2
  let φ : ℝ := Real.pi
  let area_ABC := (1/2) * x * 3 * Real.sin θ
  let area_BCD := (1/2) * 3 * 5 * Real.sin (φ - θ)
  area_ABC + area_BCD

theorem max_area_quad (x : ℝ) (h : x > 0)
  (BC_eq_3 : True)
  (CD_eq_5 : True)
  (centroids_form_isosceles : True) :
  MaxAreaABCD = 12 := by
  sorry

end max_area_quad_l42_42610


namespace average_time_of_two_trains_is_4_l42_42365

theorem average_time_of_two_trains_is_4:
  ∀ (d1 s1 d2 s2: ℝ), d1 = 200 → s1 = 50 → d2 = 240 → s2 = 80 → 
  real.to_nnreal ( (d1 / s1 + d2 / s2) / 2).to_nat = 4 :=
begin
  intros d1 s1 d2 s2 hd1 hs1 hd2 hs2,
  rw [hd1, hs1, hd2, hs2],
  norm_num,
  sorry
end

end average_time_of_two_trains_is_4_l42_42365


namespace set_cardinality_leq_5050_l42_42167

theorem set_cardinality_leq_5050 {a : Fin 2023 → ℝ} 
  (h_nonneg : ∀ i, 0 ≤ a i) 
  (h_sum : (Finset.univ.sum a.to_fun) = 100) : 
  let A := {p : Fin 2023 × Fin 2023 | p.1 ≤ p.2 ∧ a p.1 * a p.2 ≥ 1} in
  Finset.card A ≤ 5050 := 
sorry

end set_cardinality_leq_5050_l42_42167


namespace oranges_less_per_student_l42_42779

def total_students : ℕ := 12
def total_oranges : ℕ := 108
def bad_oranges : ℕ := 36

theorem oranges_less_per_student :
  (total_oranges / total_students) - ((total_oranges - bad_oranges) / total_students) = 3 :=
by
  sorry

end oranges_less_per_student_l42_42779


namespace transformed_function_is_correct_l42_42969

noncomputable def f : ℝ → ℝ := λ x, sin (x / 2 + π / 12)

theorem transformed_function_is_correct :
  ∀ x : ℝ, sin (2 * (x - π / 3)) = sin (x - π / 4) :=
by
  intros x
  symmetry
  apply eq.symm
  calc
    sin (2 * (x - π / 3)) = sin (2x - 2 * π / 3) : by rw mul_sub
    _ = sin (x - π / 4) : sorry

end transformed_function_is_correct_l42_42969


namespace evaluate_expression_l42_42469

theorem evaluate_expression : (8^6 / 8^4) * 3^10 = 3783136 := by
  sorry

end evaluate_expression_l42_42469


namespace solve_for_x2_plus_9y2_l42_42953

variable (x y : ℝ)

def condition1 : Prop := x + 3 * y = 3
def condition2 : Prop := x * y = -6

theorem solve_for_x2_plus_9y2 (h1 : condition1 x y) (h2 : condition2 x y) :
  x^2 + 9 * y^2 = 45 :=
by
  sorry

end solve_for_x2_plus_9y2_l42_42953


namespace solve_for_x_l42_42566

theorem solve_for_x (x : ℝ) (h : x + 2 = 7) : x = 5 := 
by
  sorry

end solve_for_x_l42_42566


namespace dot_product_self_l42_42555

variable (w : ℝ^n) (h : ∥w∥ = 5)

theorem dot_product_self : w ⬝ w = 25 := sorry

end dot_product_self_l42_42555


namespace probability_log_floor_eq_zero_l42_42176

noncomputable def log_base2 (x : ℝ) : ℝ := Real.log x / Real.log 2

def is_in_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

theorem probability_log_floor_eq_zero (x : ℝ) (hx : is_in_interval x) :
  probability (λ x, (⌊log_base2 (5 * x)⌋ = ⌊log_base2 x⌋)) = 0.605 :=
sorry

end probability_log_floor_eq_zero_l42_42176


namespace thomas_probability_of_two_pairs_l42_42358

def number_of_ways_to_choose_five_socks := Nat.choose 12 5
def number_of_ways_to_choose_two_pairs_of_colors := Nat.choose 4 2
def number_of_ways_to_choose_one_color_for_single_sock := Nat.choose 2 1
def number_of_ways_to_choose_two_socks_from_three := Nat.choose 3 2
def number_of_ways_to_choose_one_sock_from_three := Nat.choose 3 1

theorem thomas_probability_of_two_pairs : 
  number_of_ways_to_choose_five_socks = 792 →
  number_of_ways_to_choose_two_pairs_of_colors = 6 →
  number_of_ways_to_choose_one_color_for_single_sock = 2 →
  number_of_ways_to_choose_two_socks_from_three = 3 →
  number_of_ways_to_choose_one_sock_from_three = 3 →
  6 * 2 * 3 * 3 * 3 = 324 →
  (324 : ℚ) / 792 = 9 / 22 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end thomas_probability_of_two_pairs_l42_42358


namespace max_value_a_in_fourth_quadrant_l42_42907

noncomputable def z (a : ℤ) : ℂ := (2 + (a : ℂ) * Complex.i) / (1 + 2 * Complex.i)

def in_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

theorem max_value_a_in_fourth_quadrant :
  ∃ a : ℤ, let z_val := z a in in_fourth_quadrant z_val ∧ ∀ a' : ℤ, (a' < a) → let z_val' := z a' in in_fourth_quadrant z_val' :=
sorry

end max_value_a_in_fourth_quadrant_l42_42907


namespace find_parabola_equation_find_line_AB_equation_l42_42248

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∃ C : ℝ → ℝ → Prop, 
  (∀ y x, C y x ↔ y^2 = 2 * p * x) ∧ 
  (∃ F : ℝ × ℝ, F = (p / 2, 0)) ∧
  (∃ D : ℝ × ℝ, D = (p, 0)) ∧
  (∃ M N : ℝ × ℝ, (∃ y1 y2, M = (p / 2 - y1, y1) ∧ N = (p / 2 - y2, y2) ∧ 
   (D.2 - y1) * (D.2 - y2) = -8)) ∧
  (MD : ℝ × ℝ, MD = (p, 0)) ∧
  (|MF| = 3)

theorem find_parabola_equation : ∃ p : ℝ, p > 0 → 
  parabola_equation p (λ p, (y^2 = 4 * x)) := sorry

noncomputable def line_AB_equation (p : ℝ) (hp : p > 0) (α β: ℝ) : Prop :=
  ∃ AB : ℝ → ℝ → Prop, 
  (∀ x y, AB x y ↔ x - sqrt 2 * y - 4 = 0) ∧
  α - β = (sqrt 2) / 4 

theorem find_line_AB_equation : ∃ p : ℝ, p > 0 → 
  line_AB_equation p (λ p, (x - sqrt 2 * y - 4 = 0) when α - β is maximized ) := sorry

end find_parabola_equation_find_line_AB_equation_l42_42248


namespace hyperbola_eccentricity_l42_42084

theorem hyperbola_eccentricity : 
  ∀ (a b : ℝ), 
  a > 0 → b > 0 →
  ∃ P : ℝ × ℝ, 
    (P.1 ≥ 0) ∧ 
    (P.2 = Real.sqrt P.1) ∧ 
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧ 
    (let slope := 1 / (2 * Real.sqrt P.1) in
     let tangentLine := λ x, slope * (x - P.1) + P.2 in
     tangentLine (-1) = 0) →
    let c := Real.sqrt(a^2 + b^2) in
    (c / a = (Real.sqrt 5 + 1) / 2) :=
by
  intros a b ha hb
  sorry

end hyperbola_eccentricity_l42_42084


namespace cos_double_angle_l42_42048

theorem cos_double_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : sin (π / 2 + α) = 1 / 3) : cos (2 * α) = -7 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l42_42048


namespace find_f_2018_l42_42057

-- Define the function f, its periodicity and even property
variable (f : ℝ → ℝ)

-- Conditions
axiom f_periodicity : ∀ x : ℝ, f (x + 4) = -f x
axiom f_symmetric : ∀ x : ℝ, f x = f (-x)
axiom f_at_two : f 2 = 2

-- Theorem stating the desired property
theorem find_f_2018 : f 2018 = 2 :=
  sorry

end find_f_2018_l42_42057


namespace range_of_a_l42_42894

open Real

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 2 ≠ 0
def q (a : ℝ) : Prop := ∀ x y : ℝ, 0 < x → 0 < y → x < y → log a x < log a y

theorem range_of_a (a : ℝ) : ¬ (p a ∧ q a) ∧ (p a ∨ q a) ↔ (-2 * sqrt 2 < a ∧ a ≤ 1) ∨ (a ≥ 2 * sqrt 2) :=
by
  sorry

end range_of_a_l42_42894


namespace transformed_curve_l42_42528

variables (x y x' y' : ℝ)

def original_curve := (x^2) / 4 - y^2 = 1
def transformation_x := x' = (1/2) * x
def transformation_y := y' = 2 * y

theorem transformed_curve : original_curve x y → transformation_x x x' → transformation_y y y' → x^2 - (y^2) / 4 = 1 := 
sorry

end transformed_curve_l42_42528


namespace abs_abc_eq_one_l42_42173

theorem abs_abc_eq_one 
  (a b c : ℝ)
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hca : c ≠ a)
  (h_eq : a + 1/b^2 = b + 1/c^2 ∧ b + 1/c^2 = c + 1/a^2) : 
  |a * b * c| = 1 := 
sorry

end abs_abc_eq_one_l42_42173


namespace factorial_sum_l42_42179

theorem factorial_sum (n : ℕ) (h : n ≥ 1) : 
  (∑ i in Finset.range (n + 1), (i + 1) * ((i + 1)!)) = (n + 1)! - 1 := by
  sorry

end factorial_sum_l42_42179


namespace triangle_area_l42_42132

noncomputable def area_of_ΔPQR (area_STU : ℝ) : ℝ :=
  if h : area_STU = 24 then 144 else 0

theorem triangle_area (P Q R S T U : Point) (area_STU : ℝ)
  (hS : S = midpoint Q R)
  (hT_PR : collinear P T R ∧ dist P T / dist T R = 1 / 3)
  (hU_PS : collinear P U S ∧ dist P U / dist U S = 2 /1)
  (h_area_STU : area_STU = 24) :
  area_of_ΔPQR area_STU = 144 :=
sorry

end triangle_area_l42_42132


namespace find_x_such_that_g_inverse_of_x_is_neg2_l42_42915

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 3

theorem find_x_such_that_g_inverse_of_x_is_neg2 : g (-2) = -43 :=
by 
  rw [g]
  simp
  norm_num
  sorry

end find_x_such_that_g_inverse_of_x_is_neg2_l42_42915


namespace A_equals_k_with_conditions_l42_42621

theorem A_equals_k_with_conditions (n m : ℕ) (h_n : 1 < n) (h_m : 1 < m) :
  ∃ k : ℤ, (1 : ℝ) < k ∧ (( (n + Real.sqrt (n^2 - 4)) / 2 ) ^ m = (k + Real.sqrt (k^2 - 4)) / 2) :=
sorry

end A_equals_k_with_conditions_l42_42621


namespace area_of_remaining_shape_l42_42875

/-- Define the initial 6x6 square grid with each cell of size 1 cm. -/
def initial_square_area : ℝ := 6 * 6

/-- Define the area of the combined dark gray triangles forming a 1x3 rectangle. -/
def dark_gray_area : ℝ := 1 * 3

/-- Define the area of the combined light gray triangles forming a 2x3 rectangle. -/
def light_gray_area : ℝ := 2 * 3

/-- Define the total area of the gray triangles cut out. -/
def total_gray_area : ℝ := dark_gray_area + light_gray_area

/-- Calculate the area of the remaining figure after cutting out the gray triangles. -/
def remaining_area : ℝ := initial_square_area - total_gray_area

/-- Proof that the area of the remaining shape is 27 square centimeters. -/
theorem area_of_remaining_shape : remaining_area = 27 := by
  sorry

end area_of_remaining_shape_l42_42875


namespace num_right_angled_triangles_l42_42536

open Complex

theorem num_right_angled_triangles (n : ℕ) (hn : n ≥ 2) :
  let A := {z : ℂ | (∀ n : ℕ, z^(2*n-1) = conj z) ∧ n ≥ 2 } in 
  if even n then 2 * n^2 = (number of right-angled triangles in A) 
  else 2 * n * (n - 1) = (number of right-angled triangles in A) :=
by
  sorry

end num_right_angled_triangles_l42_42536


namespace find_2x_2y_2z_l42_42954

theorem find_2x_2y_2z (x y z : ℝ) 
  (h1 : y + z = 10 - 2 * x)
  (h2 : x + z = -12 - 4 * y)
  (h3 : x + y = 5 - 2 * z) : 
  2 * x + 2 * y + 2 * z = 3 :=
by
  sorry

end find_2x_2y_2z_l42_42954


namespace initial_plank_count_l42_42818

def Bedroom := 8
def LivingRoom := 20
def Kitchen := 11
def DiningRoom := 13
def Hallway := 4
def GuestBedroom := Bedroom - 2
def Study := GuestBedroom + 3
def BedroomReplacements := 3
def LivingRoomReplacements := 2
def StudyReplacements := 1
def LeftoverPlanks := 7

def TotalPlanksUsed := 
  (Bedroom + BedroomReplacements) +
  (LivingRoom + LivingRoomReplacements) +
  (Kitchen) +
  (DiningRoom) +
  (GuestBedroom + BedroomReplacements) +
  (Hallway * 2) +
  (Study + StudyReplacements)

theorem initial_plank_count : 
  TotalPlanksUsed + LeftoverPlanks = 91 := 
by
  sorry

end initial_plank_count_l42_42818


namespace inverse_proportional_ratios_l42_42657

variables {x y x1 x2 y1 y2 : ℝ}
variables (h_inv_prop : ∀ (x y : ℝ), x * y = 1) (hx1_ne : x1 ≠ 0) (hx2_ne : x2 ≠ 0) (hy1_ne : y1 ≠ 0) (hy2_ne : y2 ≠ 0)

theorem inverse_proportional_ratios 
  (h1 : x1 * y1 = x2 * y2)
  (h2 : (x1 / x2) = (3 / 4)) : 
  (y1 / y2) = (4 / 3) :=
by 

sorry

end inverse_proportional_ratios_l42_42657


namespace equation_of_parabola_equation_of_line_AB_l42_42258

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42258


namespace commission_percentage_l42_42420

theorem commission_percentage (monthly_salary goal sales_amount commission_amount : ℝ)
  (h1 : monthly_salary = 1000)
  (h2 : goal = 5000)
  (h3 : sales_amount = 80000)
  (h4 : commission_amount = goal - monthly_salary)
  (h5 : commission_percentage = (commission_amount / sales_amount) * 100) :
  commission_percentage = 5 := 
by 
  sorry

end commission_percentage_l42_42420


namespace parabola_equation_line_AB_equation_l42_42219

-- Define the problem and conditions
noncomputable def parabola_C_eq (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p, 0)

-- The theorem proving the equation of parabola C is correct
theorem parabola_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ parabola_C_eq p hp = (x, 0) :=
sorry

-- Define the equation for line AB given the maximum alpha - beta difference
noncomputable def line_AB_max_diff (m : ℝ) : ℝ × ℝ × ℝ := (1, -√2, -4)

-- The theorem proving the equation of line AB is correct when alpha - beta reaches its maximum value
theorem line_AB_equation (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), (line_AB_max_diff p = (x, y, -4)) ∧ (x - √2 * y - 4 = 0) :=
sorry

end parabola_equation_line_AB_equation_l42_42219


namespace find_k_l42_42694

noncomputable def k := 5000

theorem find_k
  (num_students : Nat := 10001)
  (num_pair_students : Nat := num_students * (num_students - 1) / 2)
  (k_ := k)
  (sum_club_pairs : Nat := ∑ i in {1..num_students}, (2 * i + 1) * i / 2)
  (sum_societies : Nat := k_ * num_students) :
  sum_societies = num_pair_students :=
by
  sorry

end find_k_l42_42694


namespace factorize_poly_l42_42729

theorem factorize_poly :
  (x : ℤ) → x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) :=
by
  sorry

end factorize_poly_l42_42729


namespace largest_spherical_ball_radius_on_torus_l42_42424

noncomputable def radius_of_largest_spherical_ball_on_torus := 4

theorem largest_spherical_ball_radius_on_torus :
  ∀ (inner_radius outer_radius : ℝ) (center : ℝ × ℝ × ℝ) (ball_center_z : ℝ),
  inner_radius = 3 →
  outer_radius = 5 →
  center = (4, 0, 1) →
  ball_center_z = radius_of_largest_spherical_ball_on_torus →
  (4^2 + (ball_center_z - 1)^2) = (ball_center_z + 1)^2 :=
by
  intros inner_radius outer_radius center ball_center_z
  assume inner_radius_eq outer_radius_eq center_eq ball_center_z_eq
  simp only [inner_radius_eq, outer_radius_eq, center_eq, ←ball_center_z_eq]
  ring_nf
  sorry

#eval largest_spherical_ball_radius_on_torus 3 5 (4, 0, 1) 4

end largest_spherical_ball_radius_on_torus_l42_42424


namespace fair_coin_toss_probability_l42_42783

theorem fair_coin_toss_probability :
  let n := 10
  let a : ℕ → ℕ
  let b : ℕ → ℕ
  a 1 = 1 ∧ b 1 = 1 →
  (∀ n, a (n + 1) = a n + b n ∧ b (n + 1) = a n) →
  (∀ n, a 10 + b 10 = 144) →
  (∃ i j, i + j = 73 ∧ i / j = 9 / 64) :=
by
  let n := 10
  let a : ℕ → ℕ := sorry
  let b : ℕ → ℕ := sorry
  assume h1 : a 1 = 1 ∧ b 1 = 1
  assume h2 : ∀ n, a (n + 1) = a n + b n ∧ b (n + 1) = a n
  assume h3 : a 10 + b 10 = 144
  use 9
  use 64
  split
  · rfl
  · norm_num
  sorry

end fair_coin_toss_probability_l42_42783


namespace part_one_part_two_l42_42228

-- Definitions given in the problem
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def line_passing_through_focus (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = 0}

-- Given conditions rewritten in Lean
axiom M_and_N_intersect_C (p : ℝ) (p_pos : p > 0) : ∃ M N : ℝ × ℝ, M ∈ parabola p p_pos ∧ N ∈ parabola p p_pos ∧ M ∈ line_passing_through_focus p ∧ N ∈ line_passing_through_focus p
axiom MD_perpendicular_x_axis (MD : ℝ × ℝ) : MD.1 = 0
axiom MF_distance_three (M : ℝ × ℝ) (F : ℝ × ℝ) : dist M F = 3

-- Goals to prove:
-- Part (1)
theorem part_one (p : ℝ) (p_pos : p > 0) : parabola p p_pos = {xy : ℝ × ℝ | xy.2^2 = 4 * xy.1} :=
sorry

-- Part (2)
theorem part_two : ∃ p : ℝ, ∃ AB : Set (ℝ × ℝ), AB = {xy : ℝ × ℝ | xy.1 - sqrt 2 * xy.2 - 4 = 0} :=
sorry

end part_one_part_two_l42_42228


namespace oranges_distributed_l42_42777

theorem oranges_distributed :
  ∀ (total_students : ℕ) (initial_oranges : ℕ) (bad_oranges : ℕ),
  total_students = 12 →
  initial_oranges = 108 →
  bad_oranges = 36 →
  let good_oranges := initial_oranges - bad_oranges in
  (initial_oranges / total_students - good_oranges / total_students) = 3 :=
by
  intros total_students initial_oranges bad_oranges ts_eq io_eq bo_eq
  let good_oranges := initial_oranges - bad_oranges
  sorry

end oranges_distributed_l42_42777


namespace coins_are_5_and_10_l42_42114

-- Define the conditions
variables {a b : ℕ}

-- The condition that the sum of the two coin values is 15 kopecks
def sum_is_15 : a + b = 15 := by sorry

-- The condition that one of the coins is not a 5-kopek coin
def one_is_not_five : a ≠ 5 ∨ b ≠ 5 := by sorry

-- The theorem stating that the two coins are 5 kopecks and 10 kopecks
theorem coins_are_5_and_10 (h_sum : sum_is_15) (h_not_five : one_is_not_five) : 
  (a = 5 ∧ b = 10) ∨ (a = 10 ∧ b = 5) := 
by 
  sorry

end coins_are_5_and_10_l42_42114


namespace general_term_of_a_T_sequence_minimum_value_of_sequence_l42_42294

section Problem

variable {n : ℕ} (x : ℝ)
noncomputable def S (n : ℕ) : ℕ := n^2 + n
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 2 else 2 * n
noncomputable def b (n : ℕ) : ℝ := x^(n-1)
noncomputable def c (n : ℕ) : ℝ := a n * b n
noncomputable def T (n : ℕ) : ℝ := if x = 1 then (n^2 + n) else (2 - 2 * (n + 1) * x^n + 2 * n * x^(n+1)) / (1 - x)^2

theorem general_term_of_a (n : ℕ) : a n = if n = 1 then 2 else 2 * n := sorry

theorem T_sequence (n : ℕ) : 
  T n = 
    if x = 1 then (n^2 + n) 
    else (2 - 2 * (n + 1) * x^n + 2 * n * x^(n + 1)) / (1 - x)^2 := sorry

theorem minimum_value_of_sequence (n : ℕ) (hn : 0 < n): 
  (∀ (x : ℝ), x = 2 → ∃ k, ∀ n ≥ k, (nT (n+1) - 2 * n) / (T (n+2) - 2) ≥ (n^2) / (2 * (n + 1))) := sorry

end Problem

end general_term_of_a_T_sequence_minimum_value_of_sequence_l42_42294


namespace find_other_endpoint_l42_42683

theorem find_other_endpoint :
  ∀ (A B M : ℝ × ℝ),
  M = (2, 3) →
  A = (7, -4) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  B = (-3, 10) :=
by
  intros A B M hM1 hA hM2
  sorry

end find_other_endpoint_l42_42683


namespace scientific_notation_of_population_l42_42404

def population : ℤ := 229000

theorem scientific_notation_of_population : (229000 : ℤ) = 2.29 * (10 : ℝ) ^ 5 := 
by
sry

end scientific_notation_of_population_l42_42404


namespace find_a_l42_42115

theorem find_a (a : ℝ) (h : ∫ x in 1..a, (2 * x + 1 / x) = ln 3 + 8) : a = 3 :=
sorry

end find_a_l42_42115


namespace find_expression_for_f_l42_42127

noncomputable def f (x a b : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

-- Assuming a, b ∈ ℝ, f(x) is even, and range of f(x) is (-∞, 2]
theorem find_expression_for_f (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = f (-x) a b) (h2 : ∀ y : ℝ, ∃ x : ℝ, f x a b = y → y ≤ 2):
  f x a b = -x^2 + 2 :=
by 
  sorry

end find_expression_for_f_l42_42127


namespace find_k_l42_42045

def line_equation (k x y : ℝ) : Prop := 2 + 3 * k * x = -4 * y
def point_on_line : Prop := line_equation k (-1/3 : ℝ) (1 : ℝ)
def correct_answer : Prop := k = 6

theorem find_k (h : point_on_line) : correct_answer :=
sorry

end find_k_l42_42045


namespace infinite_solutions_b_l42_42868

theorem infinite_solutions_b (x b : ℝ) : 
    (∀ x, 4 * (3 * x - b) = 3 * (4 * x + 16)) → b = -12 :=
by
  sorry

end infinite_solutions_b_l42_42868


namespace line_equation_slope_intercept_l42_42336

theorem line_equation_slope_intercept (m b : ℝ) (h1 : m = -1) (h2 : b = -1) :
  ∀ x y : ℝ, y = m * x + b → x + y + 1 = 0 :=
by
  intros x y h
  sorry

end line_equation_slope_intercept_l42_42336


namespace solve_for_x_l42_42654

theorem solve_for_x (x : ℝ) (h : (3 * x - 17) / 4 = (x + 12) / 5) : x = 12.09 :=
by
  sorry

end solve_for_x_l42_42654


namespace correct_time_fraction_l42_42767

theorem correct_time_fraction : (3 / 4 : ℝ) * (3 / 4 : ℝ) = (9 / 16 : ℝ) :=
by
  sorry

end correct_time_fraction_l42_42767


namespace round_nearest_tenth_l42_42646

theorem round_nearest_tenth (x : ℝ) (h1 : x = 7.25) : 
  Float.round_nearest (x * 10) / 10 = 7.3 :=
by
  rw [h1]
  norm_num
  sorry

end round_nearest_tenth_l42_42646


namespace sum_of_angles_of_roots_l42_42484

theorem sum_of_angles_of_roots (z : ℂ) (θ_1 θ_2 θ_3 θ_4 θ_5 θ_6 : ℝ) :
  (∃ (k : ℤ), (z = complex.exp (complex.I * (θ_1 / 180 * real.pi))) ∧ (0 <= θ_1 ∧ θ_1 < 360) ) ∧
  (∃ (k : ℤ), (z = complex.exp (complex.I * (θ_2 / 180 * real.pi))) ∧ (0 <= θ_2 ∧ θ_2 < 360) ) ∧
  (∃ (k : ℤ), (z = complex.exp (complex.I * (θ_3 / 180 * real.pi))) ∧ (0 <= θ_3 ∧ θ_3 < 360) ) ∧
  (∃ (k : ℤ), (z = complex.exp (complex.I * (θ_4 / 180 * real.pi))) ∧ (0 <= θ_4 ∧ θ_4 < 360) ) ∧
  (∃ (k : ℤ), (z = complex.exp (complex.I * (θ_5 / 180 * real.pi))) ∧ (0 <= θ_5 ∧ θ_5 < 360) ) ∧
  (∃ (k : ℤ), (z = complex.exp (complex.I * (θ_6 / 180 * real.pi))) ∧ (0 <= θ_6 ∧ θ_6 < 360) ) ∧
  z ^ 6 = -1 - complex.I → θ_1 + θ_2 + θ_3 + θ_4 + θ_5 + θ_6 = 1125 := 
sorry

end sum_of_angles_of_roots_l42_42484


namespace parallel_vectors_k_value_l42_42095

-- Defining vectors a and b.
def a : ℝ × ℝ := (2, -1)
def b (k : ℝ) : ℝ × ℝ := (k, 5 / 2)

-- Define the condition that vectors a and b are parallel.
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v1 = (c * v2.1, c * v2.2)

-- Main theorem to prove.
theorem parallel_vectors_k_value (k : ℝ) 
  (ha : a = (2, -1)) 
  (hb : b k = (k, 5 / 2))
  (h_parallel : are_parallel a (b k)) : 
  k = -5 :=
sorry

end parallel_vectors_k_value_l42_42095


namespace find_function_l42_42993

open Real

theorem find_function (f : ℝ → ℝ) :
  let g := λ x, f (x / 2);
  let h := λ x, g (x - π / 3);
  (∀ x, h x = sin (x - π / 4)) →
  ∀ x, f x = sin (x / 2 + π / 12) :=
by
  intros g h h_eq x
  sorry

end find_function_l42_42993


namespace find_min_phi_l42_42069

noncomputable def f (x φ : Real) : Real :=
  sin (2 * x) * cos (2 * φ) + cos (2 * x) * sin (2 * φ)

theorem find_min_phi {φ : Real} (h_symm : ∀ x, f x φ = f (π / 3 - x) φ) (h_pos : φ > 0) : 
  φ = 5 * π / 12 :=
by
  sorry

end find_min_phi_l42_42069


namespace cos_angle_between_vectors_l42_42544

noncomputable def norm {E : Type*} [inner_product_space ℝ E] (v : E) := ∥v∥

variables {E : Type*} [inner_product_space ℝ E]
variables (u v : E)

-- Definitions based on conditions
axiom norm_u : norm u = 5
axiom norm_v : norm v = 10
axiom norm_u_plus_v : norm (u + v) = 12

theorem cos_angle_between_vectors :
  let φ := real.acos (inner_product_space.inner u v / (norm u * norm v)) in
  real.cos φ = 19 / 100 :=
by sorry

end cos_angle_between_vectors_l42_42544


namespace problem_1_solution_problem_2_solution_l42_42914

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x| - 2

theorem problem_1_solution (x : ℝ) : f(x) ≥ 0 ↔ x ≤ -3 ∨ x ≥ 1 := by
  sorry

theorem problem_2_solution (a : ℝ) : (∃ x : ℝ, f(x) ≤ |x| + a) → a ≥ -3 := by
  sorry

end problem_1_solution_problem_2_solution_l42_42914


namespace equation_of_parabola_equation_of_line_AB_l42_42254

-- Define the conditions and provide two theorems to prove
namespace Proof1

-- Given conditions
def parabola (p : ℝ) (h : p > 0) : Prop := ∃ x y : ℝ, y^2 = 2 * p * x
def focus_point (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)
def condition_MD_perp_x_axis (x y p : ℝ) (h : p > 0) : Prop := y^2 = 2 * p * x ∧ x = p

-- |MF| = 3
def condition_MF_dist (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop := 
  let dist := λ F M : ℝ × ℝ, (F.1 - M.1)^2 + (F.2 - M.2)^2
  ∃ F_dist : ℝ,  dist F M = F_dist ∧ F_dist = 9 -- (sqrt(9)^2)  

-- Targets to prove
theorem equation_of_parabola (p : ℝ) (h : p > 0) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  parabola p h ∧ focus_point p = F ∧ point_D p = (p, 0) ∧ condition_MD_perp_x_axis M.1 M.2 p h ∧ condition_MF_dist M F → 
  ∃ x y : ℝ, y^2 = 4 * x :=
sorry -- Proof to be completed

end Proof1

namespace Proof2

-- Given conditions from proof 1 hold

-- Additional conditions and angles
def slope_MN (y₁ y₂ : ℝ) : ℝ := 4 / (y₁ + y₂)
def condition_intersection (y₂ : ℝ) : ℝ := -8 / y₂
def condition_tangent (y₁ y₂ : ℝ) : ℝ := y₁ * y₂ / (-2 * (y₁ + y₂))

-- A line determined by intersection points
def line_AB (m : ℝ) : ℝ × ℝ → Prop := 
  λ (P : ℝ × ℝ), let x := P.1 in let y := P.2 in 4 * x - 4 * (Real.sqrt 2) * y - 16 = 0
 
-- Targets to prove
theorem equation_of_line_AB (p : ℝ) (h : p > 0) :
  -- Given that the maximal difference between the inclinations holds
  parabola p h ∧ ( ∀ M_n M : ℝ × ℝ, slope_MN M_n.2 M.2 ≠ 0) ∧
  ( ∀ y₂ : ℝ, condition_intersection y₂ ≠ 0 ) ∧ 
  ( ∀ y₁ y₂ : ℝ, condition_tangent y₁ y₂ ≠ 0 ) → 
  line_AB (Real.sqrt (2) / 2) (4, 4 * Real.sqrt (2), 16) := 
sorry -- Proof to be completed.

end Proof2

end equation_of_parabola_equation_of_line_AB_l42_42254


namespace f_is_even_if_g_is_odd_l42_42615

variable (g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) :=
  ∀ x : ℝ, h (-x) = -h x

def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = f x

theorem f_is_even_if_g_is_odd (hg : is_odd g) :
  is_even (fun x => |g (x^4)|) :=
by
  sorry

end f_is_even_if_g_is_odd_l42_42615


namespace william_taxable_land_percentage_l42_42354

theorem william_taxable_land_percentage :
  (total_tax_collected : ℝ) = 3840 →
  (tax_paid_by_william : ℝ) = 480 →
  (percent_of_cultivated_land_taxed : ℝ) = 0.75 →
  let total_taxable_income := total_tax_collected / percent_of_cultivated_land_taxed in
  let percentage_owned_by_william := (tax_paid_by_william / total_taxable_income) * 100 in
  percentage_owned_by_william = 9.375 :=
begin
  sorry
end

end william_taxable_land_percentage_l42_42354


namespace initial_students_count_l42_42668

theorem initial_students_count (n : ℕ) (W : ℝ) 
  (h1 : W = n * 28) 
  (h2 : W + 1 = (n + 1) * 27.1) : 
  n = 29 := by
  sorry

end initial_students_count_l42_42668


namespace daughter_age_l42_42685

-- Definitions for weights in kg
variables (M D G S : ℝ)

-- Grandchild's age in years
def grandchild_age : ℝ := 6

-- Conditions
axiom total_weight: M + D + G + S = 230
axiom daughter_grandchild_weight: D + G = 60
axiom son_in_law_weight: S = 2 * M
axiom grandchild_weight_proportion: G = M / 5

-- Proportionality of daughter's weight to age
axiom proportional_weight_age: ∃ k : ℝ, D = k * A

-- Given the grandchild's age
variable (A : ℝ)

-- Main statement: proving the daughter's age
theorem daughter_age : A ≈ 25.76 :=
by
  -- Skipping the actual proof and providing the statement as requested
  sorry

end daughter_age_l42_42685


namespace find_b_find_area_of_incircle_l42_42130

noncomputable theory

def a : ℝ := 4
def cos_A : ℝ := 3/4
def sin_B : ℝ := (5 * real.sqrt 7) / 16
axiom c_gt_4 : ∀ (c : ℝ), c > 4

-- Prove b = 5
theorem find_b (b : ℝ) (c : ℝ) (h_c : c > 4) :
  b = (a * sin_B / (real.sqrt (1 - cos_A^2))) :=
by sorry

-- Prove the area of the incircle
theorem find_area_of_incircle (c : ℝ) (h_c: c > 4) :
  let b := 5
  in let S := (1/2) * a * c * sin_B
  in let l := a + b + c
  in let r := (2 * S) / l
  in real.pi * r^2 = (7/4) * real.pi :=
by sorry

end find_b_find_area_of_incircle_l42_42130
