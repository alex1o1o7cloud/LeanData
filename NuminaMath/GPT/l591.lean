import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Logic.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Ring
import Real
import data.int.modeq
import data.nat.prime

namespace problem_solution_l591_591665

theorem problem_solution :
  (204^2 - 196^2) / 16 = 200 :=
by
  sorry

end problem_solution_l591_591665


namespace min_expression_value_l591_591215

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  a^2 + 4 * a * b + 8 * b^2 + 10 * b * c + 3 * c^2

theorem min_expression_value (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 3) :
  minimum_value a b c ≥ 27 :=
sorry

end min_expression_value_l591_591215


namespace log_product_eq_10_for_n_1023_l591_591105

theorem log_product_eq_10_for_n_1023 :
  (λ (n : ℕ), ∏ i in finset.range n, real.log ((i + 2) : real) / real.log ((i + 1) : real)) 1023 = 10 := 
sorry

end log_product_eq_10_for_n_1023_l591_591105


namespace sin_arccos_l591_591068

theorem sin_arccos :
  sin (arccos (8 / 17)) = 15 / 17 :=
sorry

end sin_arccos_l591_591068


namespace total_cakes_served_l591_591039

-- Define the conditions
def cakes_lunch_today := 5
def cakes_dinner_today := 6
def cakes_yesterday := 3

-- Define the theorem we want to prove
theorem total_cakes_served : (cakes_lunch_today + cakes_dinner_today + cakes_yesterday) = 14 :=
by
  -- The proof is not required, so we use sorry to skip it
  sorry

end total_cakes_served_l591_591039


namespace cost_of_fencing_per_meter_in_cents_l591_591994

-- Define the conditions
def length_ratio : ℕ := 3
def width_ratio : ℕ := 2
def total_area : ℕ := 3750
def total_fencing_cost_dollars : ℕ := 100

-- Define the theorem we need to prove
theorem cost_of_fencing_per_meter_in_cents : 
  let x := Math.sqrt(3750 / (length_ratio * width_ratio))
  let length := length_ratio * x
  let width := width_ratio * x
  let perimeter := 2 * (length + width)
  let cost_per_meter := total_fencing_cost_dollars / perimeter
  cost_per_meter * 100 = 40 := 
  sorry

end cost_of_fencing_per_meter_in_cents_l591_591994


namespace triangle_equilateral_l591_591562

-- Define a structure for a point
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a structure for a triangle
structure Triangle :=
  (A B C : Point)

-- Define medians of the triangle
def median (A B C : Point) : Point :=
  ⟨(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3⟩

-- Define centroid of the triangle
def centroid (T : Triangle) : Point :=
  median T.A T.B T.C

-- Define the problem statement
theorem triangle_equilateral (T : Triangle)
  (G : Point)
  (incircle_congruent : ∀ α β γ : Triangle,
    α.A = T.A → β.B = T.B → γ.C = T.C →
    α.C = γ.A → β.C = γ.B →
    α.B = G → β.A = G → γ.B = G →
    -- circles inscribed in α, β, γ are congruent
    true) :
  -- Prove that triangle ABC is equilateral
  (T.A.distance T.B = T.B.distance T.C) ∧ (T.B.distance T.C = T.C.distance T.A) :=
  sorry

end triangle_equilateral_l591_591562


namespace find_g_inverse_75_l591_591857

noncomputable def g (x : ℝ) : ℝ := 3 * x^3 - 6

theorem find_g_inverse_75 : g⁻¹ 75 = 3 := sorry

end find_g_inverse_75_l591_591857


namespace geom_series_sum_l591_591664

noncomputable def geom_sum (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1) 

theorem geom_series_sum (S : ℕ) (a r n : ℕ) (eq1 : a = 1) (eq2 : r = 3)
  (eq3 : 19683 = a * r^(n-1)) (S_eq : S = geom_sum a r n) : 
  S = 29524 :=
by
  sorry

end geom_series_sum_l591_591664


namespace tangent_line_eq_l591_591153

open Real

-- The function f(x) = x * log x
def f (x : ℝ) : ℝ := x * log x

-- The point (1, 0)
def p : ℝ × ℝ := (1, 0)

-- Lean statement of proving the tangent line equation
theorem tangent_line_eq (x y : ℝ) (h : y = f x) (hx : x = 1) (hy : y = 0) : y = x - 1 :=
by
  sorry

end tangent_line_eq_l591_591153


namespace min_goals_in_previous_three_matches_l591_591190

theorem min_goals_in_previous_three_matches 
  (score1 score2 score3 score4 : ℕ)
  (total_after_seven_matches : ℕ)
  (previous_three_goal_sum : ℕ) :
  score1 = 18 →
  score2 = 12 →
  score3 = 15 →
  score4 = 14 →
  total_after_seven_matches ≥ 100 →
  previous_three_goal_sum = total_after_seven_matches - (score1 + score2 + score3 + score4) →
  (previous_three_goal_sum / 3 : ℝ) < ((score1 + score2 + score3 + score4) / 4 : ℝ) →
  previous_three_goal_sum ≥ 41 :=
by
  sorry

end min_goals_in_previous_three_matches_l591_591190


namespace chair_cost_l591_591573

-- Define the conditions
def total_spent : ℕ := 56
def table_cost : ℕ := 34
def num_chairs : ℕ := 2

-- Define the statement we need to prove
theorem chair_cost :
  ∃ (chair_cost : ℕ), 2 * chair_cost + table_cost = total_spent ∧ chair_cost = 11 :=
by
  use 11
  split
  sorry -- proof goes here, skipped as per instructions

end chair_cost_l591_591573


namespace selling_price_l591_591018

-- Definitions for conditions
variables (CP SP_loss SP_profit : ℝ)
variable (h1 : SP_loss = 0.8 * CP)
variable (h2 : SP_profit = 1.05 * CP)
variable (h3 : SP_profit = 11.8125)

-- Theorem statement to prove
theorem selling_price (h1 : SP_loss = 0.8 * CP) (h2 : SP_profit = 1.05 * CP) (h3 : SP_profit = 11.8125) :
  SP_loss = 9 := 
sorry

end selling_price_l591_591018


namespace vector_EF_l591_591889

section VectorMath
variables {α : Type*} [AddGroup α] [VectorSpace ℝ α]
variables (a b c : α)

-- Definitions and assumptions
def vector_AB := a - (2 : ℝ) • b
def vector_CD := (3 : ℝ) • a - (4 : ℝ) • b + (2 : ℝ) • c
def midpoint_vector (x y : α) := (1 / 2 : ℝ) • (x + y)

-- Variables for midpoints E and F
variables (E F : α)

-- Given conditions in Lean syntax
axiom Midpoint_E : E = midpoint_vector a c
axiom Midpoint_F : F = midpoint_vector b (3 • a - 4 • b + 2 • c)

-- Goal statement for proof
theorem vector_EF :
  E - F = 2 • a - 3 • b + c :=
sorry -- Proof omitted
end VectorMath

end vector_EF_l591_591889


namespace sum_of_integers_l591_591490

theorem sum_of_integers (a b : ℤ) (h : (Int.sqrt (a - 2023) + |b + 2023| = 1)) : a + b = 1 ∨ a + b = -1 :=
by
  sorry

end sum_of_integers_l591_591490


namespace behavior_of_sequences_l591_591212

variable {x y : ℝ} (hxy : x ≠ y ∧ 0 < x ∧ 0 < y)

def A_seq : ℕ → ℝ
| 0       := (x + y) / 2
| (n + 1) := (A_seq n + H_seq n) / 2

def G_seq : ℕ → ℝ
| 0       := sqrt (x * y)
| (n + 1) := sqrt (A_seq n * H_seq n)

def H_seq : ℕ → ℝ
| 0       := 2 * x * y / (x + y)
| (n + 1) := 2 / ((1 / A_seq n) + (1 / H_seq n))

def Q_seq : ℕ → ℝ
| 0       := sqrt ((x^2 + y^2) / 2)
| (n + 1) := sqrt ((A_seq n)^2 + (H_seq n)^2 / 2)

theorem behavior_of_sequences :
  (∀ n, A_seq (n+1) < A_seq n) ∧
  (∀ n, G_seq (n+1) = G_seq n) ∧
  (∀ n, H_seq (n+1) > H_seq n)
:=
sorry

end behavior_of_sequences_l591_591212


namespace count_multiples_of_67_l591_591725

-- Define the sequence and structural properties of the triangular array
def triangular_array_entries : ℕ → ℕ → ℕ
| 0 k => 2 * k + 1 
| n k => (triangular_array_entries (n - 1) k) + (triangular_array_entries (n - 1) (k + 1))

-- The statement that we need to prove
theorem count_multiples_of_67 :
  let entries := λ i j => triangular_array_entries i j in
  (∑ i in finRange 34, ∑ j in finRange (34 - i), if entries i j % 67 == 0 then 1 else 0) = 17 :=
sorry

end count_multiples_of_67_l591_591725


namespace tallest_stack_is_b_l591_591652

def number_of_pieces_a : ℕ := 8
def number_of_pieces_b : ℕ := 11
def number_of_pieces_c : ℕ := 6

def height_per_piece_a : ℝ := 2
def height_per_piece_b : ℝ := 1.5
def height_per_piece_c : ℝ := 2.5

def total_height_a : ℝ := number_of_pieces_a * height_per_piece_a
def total_height_b : ℝ := number_of_pieces_b * height_per_piece_b
def total_height_c : ℝ := number_of_pieces_c * height_per_piece_c

theorem tallest_stack_is_b : (total_height_b = 16.5) ∧ (total_height_b > total_height_a) ∧ (total_height_b > total_height_c) := 
by
  sorry

end tallest_stack_is_b_l591_591652


namespace relationship_among_abc_l591_591110

noncomputable def a : ℝ := 2 ^ -1.2
noncomputable def b : ℝ := Real.log 6 / Real.log 3
noncomputable def c : ℝ := Real.log 10 / Real.log 5

theorem relationship_among_abc : b > c ∧ c > a := sorry

end relationship_among_abc_l591_591110


namespace simplify_and_evaluate_l591_591957

variable {x : ℝ}
hypothesis (hx : x = Real.sqrt 5 + 1)
def main_expr := (x + 1) / (x + 2) / (x - 2 + 3 / (x + 2))

theorem simplify_and_evaluate : main_expr = Real.sqrt 5 / 5 :=
by
  unfold main_expr
  subst hx
  sorry

end simplify_and_evaluate_l591_591957


namespace ellipse_slope_product_l591_591121

theorem ellipse_slope_product (x₀ y₀ : ℝ) (hp : x₀^2 / 4 + y₀^2 / 3 = 1) :
  (y₀ / (x₀ + 2)) * (y₀ / (x₀ - 2)) = -3 / 4 :=
by
  -- The proof is omitted.
  sorry

end ellipse_slope_product_l591_591121


namespace question1_question2_question3_l591_591575

-- Define the scores and relevant statistics for seventh and eighth grades
def seventh_grade_scores : List ℕ := [96, 86, 96, 86, 99, 96, 90, 100, 89, 82]
def eighth_grade_C_scores : List ℕ := [94, 90, 92]
def total_eighth_grade_students : ℕ := 800

def a := 40
def b := 93
def c := 96

-- Define given statistics from the table
def seventh_grade_mean := 92
def seventh_grade_variance := 34.6
def eighth_grade_mean := 91
def eighth_grade_median := 93
def eighth_grade_mode := 100
def eighth_grade_variance := 50.4

-- Proof for question 1
theorem question1 : (a = 40) ∧ (b = 93) ∧ (c = 96) :=
by sorry

-- Proof for question 2 (stability comparison)
theorem question2 : seventh_grade_variance < eighth_grade_variance :=
by sorry

-- Proof for question 3 (estimating number of excellent students)
theorem question3 : (7 / 10 : ℝ) * total_eighth_grade_students = 560 :=
by sorry

end question1_question2_question3_l591_591575


namespace train_length_l591_591353

theorem train_length (L V : ℝ) (h1 : L = V * 120) (h2 : L + 1000 = V * 220) : L = 1200 := 
by
  sorry

end train_length_l591_591353


namespace knights_and_liars_on_grid_l591_591319

/-- On a 3x3 grid, there are knights who always tell the truth and liars who always lie.
Each one stated: "Among my neighbors, exactly three are liars."
Neighbors are considered to be people located on cells that share a common side.
Prove that the number of liars on the grid is exactly 5. -/
theorem knights_and_liars_on_grid :
  let grid := Matrix (Fin 3) (Fin 3) Bool in
  ∃ liars : Fin 3 × Fin 3 → Bool,
    (∀ pos, 
      ((pos = ⟨0,0⟩ ∨ pos = ⟨0,2⟩ ∨ pos = ⟨2,0⟩ ∨ pos = ⟨2,2⟩) → liars pos) ∧ 
      ((0 < pos.1 ∧ pos.1 < 2) ∧ (0 < pos.2 ∧ pos.2 < 2) → ¬liars pos)) ∧
    (∑ i j, if liars (⟨i, j⟩ : Fin 3 × Fin 3) then 1 else 0) = 5 :=
by
  sorry

end knights_and_liars_on_grid_l591_591319


namespace number_of_division_games_l591_591331

theorem number_of_division_games (N M : ℕ) (h1 : N > 2 * M) (h2 : M > 5) (h3 : 4 * N + 5 * M = 100) :
  4 * N = 60 :=
by
  sorry

end number_of_division_games_l591_591331


namespace measure_angle_PML_l591_591893

variables (P M L Q R S : Type*)
variables (angle : Type*) 
variables (angleQRS angleRSQ angleMPL anglePML: angle)
variables (SR SQ PM PL: ℝ)

-- Given conditions
def isosceles_triangle (PM PL: ℝ) : Prop := PM = PL
def equal_angles (a b: angle) : Prop := a = b
def opposite_angles (a b: angle) : Prop := a = b

-- Given values from problem
def angleQRS_value : angle := 36
def angleRSQ_eq_angleQRS : angle := angleQRS_value
def angleMPL_eq_angleRSQ : angle := angleRSQ_eq_angleQRS

-- The goal to prove
def goal : angle := 72

theorem measure_angle_PML :
  isosceles_triangle PM PL →
  equal_angles angleQRS angleRSQ →
  equal_angles angleRSQ angleQRS_value →
  equal_angles angleMPL angleRSQ →
  equal_angles anglePML goal :=
begin
  intros hp h1 h2 h3,
  sorry -- Proof not required
end


end measure_angle_PML_l591_591893


namespace theta_solution_count_l591_591483

theorem theta_solution_count :
  let I := Set.Ioc 0 (2 * Real.pi),
      f (θ : ℝ) := 2 + 4 * Real.cos θ - 2 * Real.sin (2 * θ) in
  (Set.count (Set.filter (λ θ, f θ = 0) I) = 4) :=
by
  sorry

end theta_solution_count_l591_591483


namespace not_all_plane_figures_have_axis_of_symmetry_l591_591053

def plane_figure (fig : Type) : Prop := true /- Placeholder definition, can be any plane figure -/

def is_axisymmetric (fig : Type) (l : Type) : Prop :=
∀ (fold : fig → fig → Type), (fold fig l = fold fig l) -- This captures the idea of overlapping sides when folded

def has_axis_of_symmetry (fig : Type) : Prop :=
∃ l : Type, is_axisymmetric fig l

theorem not_all_plane_figures_have_axis_of_symmetry :
  ¬ (∀ f: Type, plane_figure f → has_axis_of_symmetry f) :=
sorry

end not_all_plane_figures_have_axis_of_symmetry_l591_591053


namespace min_steps_to_construct_altitudes_l591_591074

theorem min_steps_to_construct_altitudes (A B C : Point) : 
    ∃ steps : ℕ, steps = 7 ∧ (∀ (arc line : Line) (draw : arc ∪ line -> steps) 
    (not_count_steps : ∀ (align_compass align_straightedge : Action), align_compass ∨ align_straightedge -> False), 
    draw(arc ∪ line) = 7) :=
begin
  sorry,
end

end min_steps_to_construct_altitudes_l591_591074


namespace calculate_expression_l591_591368

theorem calculate_expression : (-3)^2 + 2017^0 - Real.sqrt(18) * Real.sin (Real.pi / 4) = 7 := by
  sorry

end calculate_expression_l591_591368


namespace chair_cost_l591_591571

/--
Nadine went to a garage sale and spent $56. She bought a table for $34 and 2 chairs.
Each chair cost the same amount.
Prove that one chair cost $11.
-/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) (total_cost : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  total_cost = 56 - 34 →
  total_cost / num_chairs = 11 :=
by
  sorry

end chair_cost_l591_591571


namespace find_k_l591_591134

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem find_k 
  (a b : ℝ → ℝ) 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = 1) 
  (hab : ∀ x, a x = b x) 
  (hinner : a • b = (1 + 4 * k^2) / (4 * k)) 
  (hk_pos : 0 < k) :
  k = 1/2 := 
sorry

end find_k_l591_591134


namespace coolant_system_l591_591972

noncomputable def initial_coolant : ℝ :=
  let x := 10.13 in 
  let antifreeze_initial := 0.30 * x in
  let y := x - 7.6 in
  let antifreeze_replaced := 0.80 * y in
  let antifreeze_final := 0.50 * x in
  x

theorem coolant_system :
  ∃ x: ℝ, let antifreeze_initial := 0.30 * x in
          let y := x - 7.6 in
          let antifreeze_replaced := 0.80 * y in
          let antifreeze_final := 0.50 * x in
          1.10 * x - 6.08 = 0.50 * x ∧ x = 10.13 :=
begin
  use 10.13,
  sorry
end

end coolant_system_l591_591972


namespace min_value_is_two_l591_591379

noncomputable def polynomial_min_value : ℝ :=
  let r1 := 1
  let r2 := 1
  (1 / (r1^5)) + (1 / (r2^5))

theorem min_value_is_two :
  (∃ (r1 r2 : ℝ), 
    (r1 + r2 = r1^2 + r2^2) ∧ 
    (r1 + r2 = r1^4 + r2^4) ∧ 
    (r1 + r2 = 2) ∧ 
    (1 / (r1^5)) + (1 / (r2^5)) = polynomial_min_value) :=
by
  use 1, 1
  split
  -- Proof steps will come here
  sorry

end min_value_is_two_l591_591379


namespace inequality_proof_l591_591824

open Real

noncomputable def f (t x : ℝ) : ℝ := t * x - (t - 1) * log x - t

theorem inequality_proof (t x : ℝ) (h_t : t ≤ 0) (h_x : x > 1) : 
  f t x < exp (x - 1) - 1 :=
sorry

end inequality_proof_l591_591824


namespace shanna_initial_tomato_plants_l591_591954

theorem shanna_initial_tomato_plants (T : ℕ) 
  (h1 : 56 = (T / 2) * 7 + 2 * 7 + 3 * 7) : 
  T = 6 :=
by sorry

end shanna_initial_tomato_plants_l591_591954


namespace curve_C_cartesian_line_l_cartesian_max_area_triangle_MAB_l591_591899

noncomputable def parametric_curve_C : ℝ → ℝ × ℝ := λ α, (4 * Real.cos α, 4 * Real.sin α)

theorem curve_C_cartesian :
  ∀ (α : ℝ), let point := parametric_curve_C α in (point.1 ^ 2 + point.2 ^ 2 = 16) := 
by
  intro α
  let point := parametric_curve_C α
  simp [parametric_curve_C]
  sorry

noncomputable def polar_line_l (ρ θ : ℝ) : ℝ := ρ * Real.cos (θ - Real.pi / 3) - 2

theorem line_l_cartesian :
  ∀ (ρ θ : ℝ), polar_line_l ρ θ = 0 → ρ * Real.cos θ + ρ * (Real.sin (θ)) * (Real.sqrt 3) - 4 = ρ * 2 :=
by
  intro ρ θ h
  simp [polar_line_l] at h
  sorry

noncomputable def intersection_points_AB : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((4, 0), (0, 4 * Real.sqrt 3 / 3))

noncomputable def maximum_area_triangle_MAB : ℝ := 8 * Real.sqrt 3

theorem max_area_triangle_MAB :
  ∀ (α : ℝ),
  let M := parametric_curve_C α,
      A := intersection_points_AB.1,
      B := intersection_points_AB.2,
      area := (A.1 * B.2 - A.2 * B.1) / 2 in
  area ≤ maximum_area_triangle_MAB :=
by
  intro α
  let M := parametric_curve_C α
  let A := intersection_points_AB.1
  let B := intersection_points_AB.2
  sorry

end curve_C_cartesian_line_l_cartesian_max_area_triangle_MAB_l591_591899


namespace sqrt_eighteen_simplifies_l591_591604

open Real

theorem sqrt_eighteen_simplifies :
  sqrt 18 = 3 * sqrt 2 :=
by
  sorry

end sqrt_eighteen_simplifies_l591_591604


namespace volume_of_regular_triangular_pyramid_l591_591421

theorem volume_of_regular_triangular_pyramid (r α : ℝ) : 
  ∃ V : ℝ, V = (r^3 * Real.sqrt 3 * (1 + Real.sqrt (1 + 4 * (Real.tan α)^2))^3) / (12 * (Real.tan α)^2) :=
begin
  use (r^3 * Real.sqrt 3 * (1 + Real.sqrt (1 + 4 * (Real.tan α)^2))^3) / (12 * (Real.tan α)^2),
  sorry
end

end volume_of_regular_triangular_pyramid_l591_591421


namespace integer_between_roots_l591_591361

theorem integer_between_roots (x : ℤ) (h1 : (sqrt 5 : ℝ) < ↑x) (h2 : ↑x < (sqrt 15 : ℝ)) : x = 3 :=
by
  sorry

end integer_between_roots_l591_591361


namespace range_of_m_l591_591496

-- Assume g(x) = x^2 - m is a positive function on (-∞, 0).
-- We want to prove that the range of m is (3/4, 1).
noncomputable def is_positive_function (f: ℝ → ℝ) (D: set ℝ) := 
  ∃ a b, a < b ∧ [a, b] ⊆ D ∧ (∀ x ∈ [a, b], f x ∈ [a, b])

theorem range_of_m (m: ℝ):
  is_positive_function (λ x, x^2 - m) (set.Iio 0) → m ∈ set.Ioo (3/4) 1 :=
sorry

end range_of_m_l591_591496


namespace greatest_divisor_for_balls_l591_591932

theorem greatest_divisor_for_balls (n : ℕ) (hn : n > 3) : 
  ∃ d : ℕ, d = n / 4 ∧ ∀ (a : fin n → ℤ), 
  ∃ S : finset (fin n → ℤ), S.card ≥ 4 ∧ ∀ s ∈ S.powerset, 
  ∃ sum_divisible : ℤ, ∑ (x : (fin n → ℤ)) in s, x % d = 0 := 
sorry

end greatest_divisor_for_balls_l591_591932


namespace point_in_third_quadrant_l591_591516

-- Define the quadrants
inductive Quadrant
| First
| Second
| Third
| Fourth

-- Define the point (-3, -2)
def point : ℤ × ℤ := (-3, -2)

-- Define the property for each quadrant
def isInQuadrant : Quadrant → ℤ × ℤ → Prop
| Quadrant.First := λ p, p.1 > 0 ∧ p.2 > 0
| Quadrant.Second := λ p, p.1 < 0 ∧ p.2 > 0
| Quadrant.Third := λ p, p.1 < 0 ∧ p.2 < 0
| Quadrant.Fourth := λ p, p.1 > 0 ∧ p.2 < 0

-- Theorem stating that the point (-3, -2) is in the third quadrant
theorem point_in_third_quadrant : isInQuadrant Quadrant.Third point := 
sorry

end point_in_third_quadrant_l591_591516


namespace factorize_polynomial_1_factorize_polynomial_2_triangle_shape_l591_591066

-- Proof for (1)
theorem factorize_polynomial_1 (a : ℝ) : 2*a^2 - 8*a + 8 = 2*(a - 2)^2 :=
by
  sorry

-- Proof for (2)
theorem factorize_polynomial_2 (x y : ℝ) : x^2 - y^2 + 3*x - 3*y = (x - y)*(x + y + 3) :=
by
  sorry

-- Proof for (3)
theorem triangle_shape (a b c : ℝ) (h : a^2 - ab - ac + bc = 0) : 
  (a = b ∨ a = c) :=
by
  sorry

end factorize_polynomial_1_factorize_polynomial_2_triangle_shape_l591_591066


namespace cosine_identity_l591_591671

theorem cosine_identity (θ : ℝ) : 
  cos (3 * π / 2 - θ) = sin (π + θ) ∧ cos (3 * π / 2 - θ) = cos (π / 2 + θ) :=
by
  sorry

end cosine_identity_l591_591671


namespace equation_of_line_AB_l591_591817

open Real

noncomputable def circle : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 4*x + 2*y - 3 = 0

def point_M := (4 : ℝ, -8 : ℝ)

def line_through_M_intersects_circle (k : ℝ) (A B : ℝ × ℝ) : Prop := 
  (circle A.1 A.2) ∧ (circle B.1 B.2) ∧ (A ≠ B) ∧ 
  ∃ k : ℝ, ∀ x y : ℝ, (x, y) ≠ point_M → 
  k * (x - point_M.1) = y - point_M.2 → 
  k * x - y - k * point_M.1 + point_M.2 = 0

def points_distance_eq_4 (A B : ℝ × ℝ) : Prop := 
  (sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = 4)

theorem equation_of_line_AB : 
  ∀ (A B : ℝ × ℝ), 
  line_through_M_intersects_circle (-45 / 28) A B → 
  points_distance_eq_4 A B → 
  (∀ x y : ℝ, (45 * x + 28 * y + 44 = 0) ∨ (x = 4)) := 
sorry

end equation_of_line_AB_l591_591817


namespace sin_alpha_sum_eq_zero_sin_squared_alpha_sum_eq_three_halves_sin_cubed_alpha_sum_eq_negative_three_fourths_sin_3_alpha_l591_591581

namespace TrigonometryProofs

theorem sin_alpha_sum_eq_zero (α : ℝ) :
  sin α + sin (α + 2 * π / 3) + sin (α - 2 * π / 3) = 0 := 
sorry

theorem sin_squared_alpha_sum_eq_three_halves (α : ℝ) :
  sin α^2 + sin (α + 2 * π / 3)^2 + sin (α - 2 * π / 3)^2 = 3 / 2 := 
sorry

theorem sin_cubed_alpha_sum_eq_negative_three_fourths_sin_3_alpha (α : ℝ) :
  sin α^3 + sin (α + 2 * π / 3)^3 + sin (α - 2 * π / 3)^3 = -3 / 4 * sin (3 * α) := 
sorry

end TrigonometryProofs

end sin_alpha_sum_eq_zero_sin_squared_alpha_sum_eq_three_halves_sin_cubed_alpha_sum_eq_negative_three_fourths_sin_3_alpha_l591_591581


namespace hyperbola_foci_condition_l591_591323

theorem hyperbola_foci_condition (m n : ℝ) (h : m * n > 0) :
    (m > 0 ∧ n > 0) ↔ ((∃ (x y : ℝ), m * x^2 - n * y^2 = 1) ∧ (∃ (x y : ℝ), m * x^2 - n * y^2 = 1)) :=
sorry

end hyperbola_foci_condition_l591_591323


namespace sum_of_first_seven_terms_geom_seq_l591_591882

noncomputable def pos_geom_seq (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0 ∧ a (n + 1) / a n = a 1

noncomputable def geom_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
a 1 * ((1 - (a (1+0)/ a 0)^n) / (1 - a (1+0)/ a 0))

theorem sum_of_first_seven_terms_geom_seq :
  ∃ (a : ℕ → ℝ), pos_geom_seq a ∧ a 1 = 1 ∧ geom_sum a 7 = 127 ∧ 
  (-a 3, a 2, a 4) form an arithmetic sequence :=
begin
  sorry -- Proof is not required
end

end sum_of_first_seven_terms_geom_seq_l591_591882


namespace mouse_cannot_eat_entire_cheese_l591_591342

-- Defining the conditions of the problem
structure Cheese :=
  (size : ℕ := 3)  -- The cube size is 3x3x3
  (central_cube_removed : Bool := true)  -- The central cube is removed

inductive CubeColor
| black
| white

structure Mouse :=
  (can_eat : CubeColor -> CubeColor -> Bool)
  (adjacency : Nat -> Nat -> Bool)

def cheese_problem (c : Cheese) (m : Mouse) : Bool := sorry

-- The main theorem: It is impossible for the mouse to eat the entire piece of cheese.
theorem mouse_cannot_eat_entire_cheese : ∀ (c : Cheese) (m : Mouse),
  cheese_problem c m = false := sorry

end mouse_cannot_eat_entire_cheese_l591_591342


namespace trigonometric_identity_l591_591132

theorem trigonometric_identity (α : ℝ) (h1 : cos α - sin α = (3 * real.sqrt 2) / 5) (h2 : π < α ∧ α < 3 * π / 2) :
  (sin (2 * α) * (1 + tan α)) / (1 - tan α) = -28 / 75 := 
by
  sorry


end trigonometric_identity_l591_591132


namespace julia_tuesday_kids_l591_591536

-- Definitions based on conditions
def kids_on_monday : ℕ := 11
def tuesday_more_than_monday : ℕ := 1

-- The main statement to be proved
theorem julia_tuesday_kids : (kids_on_monday + tuesday_more_than_monday) = 12 := by
  sorry

end julia_tuesday_kids_l591_591536


namespace range_of_a_for_empty_solution_set_l591_591471

theorem range_of_a_for_empty_solution_set :
  ∀ a : ℝ, (∀ x : ℝ, ¬ (|x - 3| + |x - 4| < a)) ↔ a ≤ 1 := 
sorry

end range_of_a_for_empty_solution_set_l591_591471


namespace range_of_f_l591_591112

def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 3 * (Real.sin x) + 3

theorem range_of_f :
  (∀ x, x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3) → 6 ≤ f x ∧ f x ≤ 49 / 8) :=
by
  intros x hx
  sorry

end range_of_f_l591_591112


namespace range_of_a_with_root_in_interval_l591_591149

theorem range_of_a_with_root_in_interval {a : ℝ} :
  (∃ x ∈ (0 : ℝ), (2 * Real.pi), (Real.sin (a * x) - a * Real.sin x = 0)) ↔ 
  a ∈ Set.Ioo (-(1/2 : ℝ)) (-(0 : ℝ)) ∪ Set.Ioo (0 : ℝ) (1/2) ∪ {0} := 
by
  sorry

end range_of_a_with_root_in_interval_l591_591149


namespace train_time_lost_l591_591352

-- Definitions 
def speed_car : ℝ := 120 -- speed of the car in km/h
def speed_train : ℝ := speed_car * 1.5 -- speed of the train in km/h (50% faster than the car)
def distance_A_to_B : ℝ := 75 -- distance between point A and B in km

-- Time calculations (in hours)
def time_car : ℝ := distance_A_to_B / speed_car
def time_train_without_stops : ℝ := distance_A_to_B / speed_train

-- Time lost by the train in hours
def time_lost_hours : ℝ := time_car - time_train_without_stops

-- Time lost by the train in minutes
def time_lost_minutes : ℝ := time_lost_hours * 60

-- Theorem to prove the time lost in minutes
theorem train_time_lost : time_lost_minutes = 12.5 :=
by
  sorry

end train_time_lost_l591_591352


namespace cubic_polynomial_inequality_l591_591950

theorem cubic_polynomial_inequality
  (A B C : ℝ)
  (h : ∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    A = -(a + b + c) ∧ B = ab + bc + ca ∧ C = -abc) :
  A^2 + B^2 + 18 * C > 0 :=
by
  sorry

end cubic_polynomial_inequality_l591_591950


namespace combined_average_score_correct_l591_591937

-- Define the given values based on the conditions.
def average_score_first_group : ℝ := 88
def average_score_second_group : ℝ := 76
def ratio_students : ℚ := 4 / 5

-- Define the combined average score that we need to prove.
def combined_average_score : ℝ := 81

-- Now, state the theorem that proves the combined average score given the conditions.
theorem combined_average_score_correct 
  (G1 : ℝ) (G2 : ℝ) (r : ℚ) (combined_avg : ℝ) 
  (h1 : G1 = 88) 
  (h2 : G2 = 76) 
  (h3 : r = 4 / 5) 
  (h4 : combined_avg = 81) 
  : 
  let g1 := (r.num * (G2 : ℚ)) / (r.denom : ℚ) in
  let g2 := G2 in
  let total_students := g1 + g2 in
  let total_score := G1 * g1 + G2 * g2 in
  combined_avg = total_score / total_students := 
by {
  sorry
}

end combined_average_score_correct_l591_591937


namespace unique_real_solution_of_quadratic_l591_591080

theorem unique_real_solution_of_quadratic :
  ∃ n : ℝ, n > 0 ∧ (∀ x : ℝ, 25 * x^2 + n * x + 4 = 0 → 
                         ∃! x : ℝ, 25 * x^2 + n * x + 4 = 0) :=
by {
  use 20,
  split,
  { norm_num, },
  { intros x h,
    sorry }
}

end unique_real_solution_of_quadratic_l591_591080


namespace find_q_sum_bn_l591_591801

open Real

noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (2 * (1 - (2 : ℝ) ^ (n + 1)) / (1 - (2 : ℝ))) - n)

-- Given part
axiom a_1 : ℝ := 2 -- First term
axiom S1 : ℝ := a_1 -- Sum of first term
axiom S2 : ℝ
axiom S3 : ℝ
/-
  S_2 and S_3 need to be defined in terms of a transition formula based on the given condition 
  typical for sum calculations in arithmetic progression. 
  These definitions might change in practice subject to how Sn is defined comprehensively.
-/

axiom h1 : 2 * (3 * S2) = 4 * S1 + 2 * S3 -- Given arithmetic progression property

-- Defining the sequences and claiming the results
theorem find_q : ∃ q : ℝ, q = 2 := by {
  -- Proof to be provided
  sorry
}

noncomputable def b_n (n : ℕ) : ℕ → ℝ
| 0 => 0
| n + 1 => n + a_1 * (2 ^ (n+1))

noncomputable def T_n (n : ℕ) : ℝ :=
  (n * (n + 1) / 2) + (2 ^ (n + 1) - 2)

theorem sum_bn (n : ℕ) : T_n n = (n * (n + 1) / 2) + (2 ^ (n + 1) - 2) := by {
  -- Proof to be provided
  sorry
}

end find_q_sum_bn_l591_591801


namespace radius_ratio_l591_591707

variable (VL VS rL rS : ℝ)
variable (hVL : VL = 432 * Real.pi)
variable (hVS : VS = 0.275 * VL)

theorem radius_ratio (h1 : (4 / 3) * Real.pi * rL^3 = VL)
                     (h2 : (4 / 3) * Real.pi * rS^3 = VS) :
  rS / rL = 2 / 3 := by
  sorry

end radius_ratio_l591_591707


namespace tetrahedral_dice_sum_6_probability_l591_591299

theorem tetrahedral_dice_sum_6_probability :
  let outcomes : Finset (ℕ × ℕ) := { (a, b) | a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4} }
  let favorable : Finset (ℕ × ℕ) := { (a, b) ∈ outcomes | a + b = 6 }
  let total_combinations : ℕ := Finset.card outcomes
  let favorable_combinations : ℕ := Finset.card favorable
  (favorable_combinations : ℚ) / total_combinations = 3 / 16 :=
by
  sorry

end tetrahedral_dice_sum_6_probability_l591_591299


namespace smallest_integer_solution_l591_591662

theorem smallest_integer_solution : ∃ y : ℤ, (10 + 3 * y ≤ -8 ∧ ∀ z : ℤ, (10 + 3 * z ≤ -8 → z ≥ y)) :=
begin
  use -6,
  split,
  { exact le_of_eq (by norm_num), },
  { intros z h,
    suffices : z ≤ -6, by linarith,
    have h1 := (sub_le_sub_iff_left 10).mpr h,
    norm_num at h1,
    exact (le_div_iff (by norm_num : 0 < 3)).mp h1,
  },
end

#eval smallest_integer_solution

end smallest_integer_solution_l591_591662


namespace area_triangle_ABC_a_plus_b_eq_112_l591_591539

variable (ABC : Type) [Triangle ABC]
variable (A B C D E T : ABC → Triangle) (circumcircle : ∀ (t : Triangle), TriangleC)
variable (BD AD : ℝ)

-- Conditions
variable (H1 : altitude_meeting A D BC)
variable (H2 : altitude_meeting B E AC)
variable (H3 : point_on_circumcircle T (circumcircle ABC))
variable (H4 : AT_parallel_BC T A BC)
variable (H5 : collinear_points D E T)
variable (H6 : BD = 3)
variable (H7 : AD = 4)

-- The equivalence proof problem
theorem area_triangle_ABC_a_plus_b_eq_112 : ∃ a b : ℕ, a + b = 112 ∧ 
  ∃ k : ℝ, k = a + real.sqrt b ∧ area (triangle ABC) = k := 
by {
  sorry
}

end area_triangle_ABC_a_plus_b_eq_112_l591_591539


namespace simplify_expr_l591_591486

variable (x : ℝ)

theorem simplify_expr (hx : 1 < x ∧ x < 2) : sqrt((x - 2) ^ 2) + x = 2 :=
by
  sorry

end simplify_expr_l591_591486


namespace distribute_cards_l591_591920

theorem distribute_cards (n : ℕ) (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : k > 0) :
  (∃ (f : Fin n → Fin p), ∀ i : Fin p, (∑ j in Finset.filter (λ x, f x = i) Finset.univ, (j.val : ℕ) + 1) = (k * (2 * p * k + 1))) ↔ (∃ k : ℕ, n = 2 * k * p) :=
by
  sorry

end distribute_cards_l591_591920


namespace find_x_equals_2017_l591_591090

open Real

theorem find_x_equals_2017 (x : ℝ) (h : x - floor (x / 2016) = 2016) : x = 2017 :=
sorry

end find_x_equals_2017_l591_591090


namespace sequence_difference_constant_l591_591321

theorem sequence_difference_constant :
  ∀ (x y : ℕ → ℕ), x 1 = 2 → y 1 = 1 →
  (∀ k, k > 1 → x k = 2 * x (k - 1) + 3 * y (k - 1)) →
  (∀ k, k > 1 → y k = x (k - 1) + 2 * y (k - 1)) →
  ∀ k, x k ^ 2 - 3 * y k ^ 2 = 1 :=
by
  -- Insert the proof steps here
  sorry

end sequence_difference_constant_l591_591321


namespace percent_students_elected_to_learn_from_home_l591_591961

theorem percent_students_elected_to_learn_from_home (H : ℕ) : 
  (100 - H) / 2 = 30 → H = 40 := 
by
  sorry

end percent_students_elected_to_learn_from_home_l591_591961


namespace perpendicular_CF_AB_l591_591803

open EuclideanGeometry

theorem perpendicular_CF_AB 
  {A B C M H E F : Point}
  (hA1 : IsAcuteTriangle A B C)
  (hM : Midpoint M B C)
  (hH : Perpendicular BH AC)
  (hE : PerpendicularLineThrough A PerpendicularTo AM ∧ LineIntersect BH AE E)
  (hF : PointOppositeRay AE F ∧ Distance AE AF) :
  Perpendicular CF AB :=
by
  sorry

end perpendicular_CF_AB_l591_591803


namespace simplify_expression_l591_591603

noncomputable def ω := (-1 + Complex.I * Real.sqrt 3) / 2
noncomputable def ω_bar := (-1 - Complex.I * Real.sqrt 3) / 2

-- Lean 4 statement
theorem simplify_expression : (ω ^ 9) + (ω_bar ^ 9) = 2 := 
by
  have h1 : ω ^ 3 = 1 := sorry
  have h2 : ω_bar ^ 3 = 1 := sorry
  have h3 : ω ^ 9 = (ω ^ 3) ^ 3 := by rw [←pow_mul, mul_comm, mul_one]; apply h1
  have h4 : ω_bar ^ 9 = (ω_bar ^ 3) ^ 3 := by rw [←pow_mul, mul_comm, mul_one]; apply h2
  rw [h3, h4]
  norm_num
  rw [h1, h2]
  norm_num
  done

end simplify_expression_l591_591603


namespace find_false_coin_bag_l591_591330

variable {n : ℕ} (h : n ≥ 2)
variable (x d : ℝ) (d_ne_zero : d ≠ 0)

-- Define weight of real coins
def real_weight : ℝ := x

-- Define weight of false coins
def false_weight : ℝ := x + d

-- Define weight of coins in first weighing
def first_weigh (L R : ℕ → ℝ) :=
  let L1 := (L 0) * real_weight in
  let R1 := (R 0) * real_weight in
  (R1 - L1, if L 1 then L1 + d else L1)

-- Define weight of coins in second weighing
def second_weigh (L R : ℕ → ℝ) :=
  let L2 := (L 0) * real_weight in
  let R2 := (R 0) * real_weight in
  (R2 - L2, if L 1 then L2 + d else L2)

-- Define solution
theorem find_false_coin_bag :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ 
  ( ∀ R1 L1 R2 L2 : ℝ, R1 - L1 ≠ R2 - L2 → 
    if R1 - L1 = (i - 1) * d ∧ R2 - L2 = (n - i + 1) * d then
      i = (n * (R1 - L1)) / ((R2 - L2) + (R1 - L1)) + 1
    else
      i = (n * (R1 - L1)) / ((R2 - L2) + (R1 - L1)) + 1) :=
by
  sorry

end find_false_coin_bag_l591_591330


namespace prove_smallest_number_l591_591306

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

lemma smallest_number_to_add (n : ℕ) (k : ℕ) (h: sum_of_digits n % k = r) : n % k = r →
  n % k = r → (k - r) = 7 :=
by
  sorry

theorem prove_smallest_number (n : ℕ) (k : ℕ) (r : ℕ) :
  (27452 % 9 = r) ∧ (9 - r = 7) :=
by
  sorry

end prove_smallest_number_l591_591306


namespace police_emergency_number_has_prime_divisor_gt_seven_l591_591022

theorem police_emergency_number_has_prime_divisor_gt_seven (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
by
  sorry

end police_emergency_number_has_prime_divisor_gt_seven_l591_591022


namespace derivative_of_f_tangent_line_at_pi_l591_591466

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / x

theorem derivative_of_f (x : ℝ) (h : x ≠ 0) : deriv f x = (x * Real.cos x - Real.sin x) / (x ^ 2) :=
  sorry

theorem tangent_line_at_pi : 
  let M := (Real.pi, 0)
  let slope := -1 / Real.pi
  let tangent_line (x : ℝ) : ℝ := -x / Real.pi + 1
  ∀ (x y : ℝ), (x, y) = M → y = tangent_line x :=
  sorry

end derivative_of_f_tangent_line_at_pi_l591_591466


namespace find_angle_ABC_l591_591576

noncomputable def parallelogram_conditions (A B C D E F : Type) (circumcircleABC : Set _) (DOnTangentE : Set _) (DOnTangentF : Set _) : Prop :=
  (∀ (ABCD_parallel : parallelogram A B C D), angle B < 90 ∧ distance AB < distance BC ∧
  ∈ circumcircleABC E ∧ ∈ circumcircleABC F ∧ tangent_of circle circumcircleABC D E DOnTangentE ∧
  tangent_of circle circumcircleABC D F DOnTangentF ∧ ∠ EDA = ∠ FDC)

theorem find_angle_ABC (A B C D E F : Type) (circumcircleABC : Set _) (DOnTangentE : Set _) (DOnTangentF : Set _) :
  parallelogram_conditions A B C D E F circumcircleABC DOnTangentE DOnTangentF → ∠ ABC = 60 :=
by
  sorry

end find_angle_ABC_l591_591576


namespace max_delta_success_ratio_l591_591187

theorem max_delta_success_ratio :
  ∀ (a b c d e f : ℕ),
    a * 170 < b * 210 ∧ c * 250 < d * 170 ∧ b + d + f = 600 ∧
    b ≠ 350 ∧ d ≠ 250 ∧ f ≠ 0 →
    (a + c + e : ℚ) / 600 ≤ 378 / 600 :=
by
  intros a b c d e f h,
  sorry

end max_delta_success_ratio_l591_591187


namespace chess_tournament_games_l591_591287

theorem chess_tournament_games (n : ℕ) (h : n = 5) : (n * (n - 1)) / 2 = 10 :=
by
  rw [h]
  exact rfl

end chess_tournament_games_l591_591287


namespace committee_selection_problem_l591_591953

theorem committee_selection_problem 
  (members : Finset ℕ) 
  (h1 : members.card = 5) 
  (A B : ℕ) 
  (h2 : A ∈ members) 
  (h3 : B ∈ members)
  (h4 : ∀ x, x ∈ members \ {A, B}) :
  ∃ (n : ℕ), n = 36 :=
by
  have ex_comm := members.filter (λ x, x ≠ A ∧ x ≠ B)
  have h_ex_comm := ex_comm.card
  have rem := members \ {A, B}
  have h_rem := rem.card
  have ex_aff_spt := (rem \ {ex_comm}).card
  have := 3 * 12
  use 36
  sorry

end committee_selection_problem_l591_591953


namespace unknown_number_correct_l591_591996

theorem unknown_number_correct (x : ℝ) (h : 38 + 2 * x^3 = 1250) : x = real.cbrt 606 :=
sorry

end unknown_number_correct_l591_591996


namespace sqrt3_rho1_eq_rho2_plus_rho3_area_of_OBAC_l591_591900

section problem_one
variables (ρ1 ρ2 ρ3 φ : ℝ)

def is_polar_eq := ρ1 = 2 * cos φ ∧ ρ2 = 2 * cos (φ + π / 6) ∧ ρ3 = 2 * cos (φ - π / 6)
def condition1 := is_polar_eq ρ1 ρ2 ρ3 φ

theorem sqrt3_rho1_eq_rho2_plus_rho3 (h : condition1) : sqrt 3 * ρ1 = ρ2 + ρ3 :=
sorry
end problem_one

section problem_two
variables (φ : ℝ) (t : ℝ → ℝ → ℝ)

def parametric_eq (x y : ℝ) := x = 2 - (sqrt 3 / 2) * t ∧ y = (1 / 2) * t
def parametric_line_through_BC := parametric_eq 2 0 -- parametric coordinates of B and C

theorem area_of_OBAC : 0 < t 0 0 → 0 < t 1 (sqrt 3) → 
  φ = π / 6 → 
  ρ1 = sqrt 3 →
  ρ2 = 1 →
  ρ3 = 2 →
  area = (3 * sqrt 3) / 4 :=
sorry
end problem_two

end sqrt3_rho1_eq_rho2_plus_rho3_area_of_OBAC_l591_591900


namespace greatest_mult_of_4_lessthan_5000_l591_591255

theorem greatest_mult_of_4_lessthan_5000 :
  ∃ x : ℕ, (0 < x) ∧ (x % 4 = 0) ∧ (x^3 < 5000) ∧ (∀ y : ℕ, (0 < y) ∧ (y % 4 = 0) ∧ (y^3 < 5000) → y ≤ x) := 
sorry

end greatest_mult_of_4_lessthan_5000_l591_591255


namespace triangle_expression_l591_591178

noncomputable theory
open Real

-- Define the triangle PQR with its sides
variables {P Q R : ℝ}
variables (PQ PR QR : ℝ)

-- Assume sides of the triangle
def triangle_sides := PQ = 7 ∧ PR = 6 ∧ QR = 8

-- Define the cosine and sine of angles given the sides
def expression := (cos ((P - Q)/2) / sin (R/2)) - (sin ((P - Q)/2) / cos (R/2))

-- Lean statement to be proved
theorem triangle_expression (h : triangle_sides) : expression PQ PR QR = 12 / 7 :=
sorry

end triangle_expression_l591_591178


namespace problem_conditions_l591_591151

noncomputable def f (x : ℝ) : ℝ := exp (-abs x) + cos (π * x)

theorem problem_conditions :
  let f := fun x : ℝ => exp (-abs x) + cos (π * x) in
  (∀ x, f x ≤ 2) ∧ (f 0 = 2) ∧ 
  (∑ x in (-(10 : ℝ), 10), f x = 0) ∧
  (∀ x, local_max f x > 1) :=
by
  sorry

end problem_conditions_l591_591151


namespace fraction_subtraction_l591_591666

theorem fraction_subtraction : (1 / 6 : ℚ) - (5 / 12) = -1 / 4 := 
by sorry

end fraction_subtraction_l591_591666


namespace equal_sum_partition_probability_l591_591583

-- Define the set of positive integers from 1 to 7
def nums : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define what it means to partition the set into two non-empty subsets
def valid_partition (s1 s2 : Finset ℕ) : Prop :=
  s1.nonempty ∧ s2.nonempty ∧ (s1 ∪ s2 = nums) ∧ (s1 ∩ s2 = ∅)

-- Define the condition where the sums of the two subsets are equal
def equal_sum_partition (s1 s2 : Finset ℕ) : Prop :=
  valid_partition s1 s2 ∧ (s1.sum id = s2.sum id)

-- Calculate the total number of valid partitions
def total_partitions : ℕ := Finset.card (Finset.powerset nums) - 1

-- Calculate the number of partitions with equal sums
def equal_sum_partitions : ℕ :=
  (Finset.powerset nums).count (λ s, equal_sum_partition s (nums \ s))

-- Define the probability
noncomputable def probability_equal_sum_partition : ℚ :=
  equal_sum_partitions / total_partitions

-- State the theorem to prove the probability is 4/63
theorem equal_sum_partition_probability : probability_equal_sum_partition = 4 / 63 :=
  sorry

end equal_sum_partition_probability_l591_591583


namespace lefty_jazz_non_basketball_l591_591616

-- Definitions
def total_members : ℕ := 30
def left_handed_members : ℕ := 12
def jazz_loving_members : ℕ := 20
def right_handed_non_jazz_non_basketball : ℕ := 5
def basketball_players : ℕ := 10
def left_handed_jazz_loving_basketball_players : ℕ := 3

-- Problem Statement: Prove the number of lefty jazz lovers who do not play basketball.
theorem lefty_jazz_non_basketball (x : ℕ) :
  (x + left_handed_jazz_loving_basketball_players) + (left_handed_members - x - left_handed_jazz_loving_basketball_players) + 
  (jazz_loving_members - x - left_handed_jazz_loving_basketball_players) + 
  right_handed_non_jazz_non_basketball + left_handed_jazz_loving_basketball_players = 
  total_members → x = 4 :=
by
  sorry

end lefty_jazz_non_basketball_l591_591616


namespace two_trains_distance_before_meeting_l591_591052

noncomputable def distance_one_hour_before_meeting (speed_A speed_B : ℕ) : ℕ :=
  speed_A + speed_B

theorem two_trains_distance_before_meeting (speed_A speed_B total_distance : ℕ) (h_speed_A : speed_A = 60) (h_speed_B : speed_B = 40) (h_total_distance : total_distance ≤ 250) :
  distance_one_hour_before_meeting speed_A speed_B = 100 :=
by
  sorry

end two_trains_distance_before_meeting_l591_591052


namespace exponential_inequality_l591_591855

theorem exponential_inequality (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^m < a^n) : ¬ (m < n) := 
sorry

end exponential_inequality_l591_591855


namespace circle_radius_of_complex_roots_l591_591732

theorem circle_radius_of_complex_roots (z : ℂ) (hz : (z - 1)^3 = 8 * z^3) : 
  ∃ r : ℝ, r = 1 / Real.sqrt 3 :=
by
  sorry

end circle_radius_of_complex_roots_l591_591732


namespace angle_AK_BC_eq_ninety_degrees_l591_591359

-- Define the conditions of our problem
variables (A B C D E K : Type)
  [triangle : triangle A B C]
  [acute_angled : acute_angled_triangle A B C]
  [circle_ω : circle (segment B C)]
  [intersects_AB : intersects circle_ω (segment A B) D]
  [intersects_AC : intersects circle_ω (segment A C) E]
  [tangent_D : tangent circle_ω D]
  [tangent_E : tangent circle_ω E]
  [tangents_intersect : tangents_intersect D E K]

-- Define theorem statement
theorem angle_AK_BC_eq_ninety_degrees : (angle (line A K) (line B C)) = 90 :=
  sorry

end angle_AK_BC_eq_ninety_degrees_l591_591359


namespace determine_ab_and_c_l591_591468

open Set

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^4 * Real.log x + b * x^4 - c

theorem determine_ab_and_c (a b c : ℝ) (h1 : f a b c 1 = -3 - c)
  (h2 : ∃ x > 0, ∀ x > 0, Deriv f x = 0 ∧ x = 1)
  (h3 : ∀ x > 0, f a b c x ≥ -2 * c ^ 2) :
  b = -3 ∧ a = 12 ∧ (c ∈ Iic (-1) ∪ Ici (3/2)) :=
by
  sorry

end determine_ab_and_c_l591_591468


namespace effect_on_revenue_percentage_l591_591998

theorem effect_on_revenue_percentage (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax := 0.84 * T
  let new_consumption := 1.15 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  let revenue_change := ((new_revenue - original_revenue) / original_revenue) * 100
in revenue_change = -3.4 := by
  sorry

end effect_on_revenue_percentage_l591_591998


namespace downstream_distance_l591_591020

variable (Vb Vs time_upstream distance_upstream distance_downstream speed_downstream : ℝ)

def upstream_condition := time_upstream = 5 ∧ distance_upstream = 45 ∧ Vs = 3

theorem downstream_distance 
  (h : upstream_condition Vb Vs time_upstream distance_upstream distance_downstream speed_downstream) :
  distance_downstream = 75 := 
by
  -- Proof would go here
  sorry

end downstream_distance_l591_591020


namespace maximal_guests_l591_591217

theorem maximal_guests (n : ℕ) (h_pos : 0 < n) :
  \(\forall \) guests, 
  (guests ≤ n^4 - n^3) ∧ 
  (∀ guest1 guest2, guest1 ≠ guest2 → order(guest1) ≠ order(guest2)) ∧
  ¬\(\exists \) S \(\subseteq \) guests, (S.card = n ∧ coincidesInThreeButDiffersInFourth(S)) :=
by sorry

end maximal_guests_l591_591217


namespace length_EX_collinear_AXG_l591_591554

def squares (A B C D E F G H : Point) : Prop := 
  square A B C D ∧ square E F G H ∧ side_length A B = 33 ∧ side_length E F = 12

def geometric_config (X D E A G H B C : Point) : Prop :=
  squares A B C D E F G H ∧
  collinear F E D C ∧
  between D E X ∧
  side_length D E = 18 ∧
  intersect HB DC X

theorem length_EX (A B C D E F G H X : Point) :
  geometric_config X D E A G H B C → side_length E X = 4 :=
by sorry

theorem collinear_AXG (A B C D E F G H X : Point) :
  geometric_config X D E A G H B C → side_length E X = 4 → collinear A X G :=
by sorry

end length_EX_collinear_AXG_l591_591554


namespace triangle_angle_A_triangle_perimeter_range_l591_591904

theorem triangle_angle_A
  {a b c : ℝ} (h : 2 * a * real.sin (angle A) = (2 * b + c) * real.sin (angle B) + (2 * c + b) * real.sin (angle C)) 
  (A B C : ℝ) (h₁ : A + B + C = π) (h₂ : a^2 = b^2 + c^2 - 2 * b * c * (real.cos A)) : 
  A = 2 * π / 3 :=
sorry

theorem triangle_perimeter_range
  {a : ℝ} (h : a = real.sqrt 3) {b c : ℝ} (A B C : ℝ) (l : ℝ)
  (h₁ : A + B + C = π) (h₂ : l = a + b + c) (h₃ : ∀ B, A + B = 2π / 3) :
  2 * real.sqrt 3 < l ∧ l ≤ real.sqrt 3 + 2 :=
sorry

end triangle_angle_A_triangle_perimeter_range_l591_591904


namespace intersect_ln_curve_l591_591386

theorem intersect_ln_curve (a b : ℝ) :
  (∃! x : ℝ, ln (1 + x^2) = a * x + b) ↔
  (a = 0 ∧ b = 0) ∨
  (|a| ≥ 1) ∨
  (0 < |a| ∧ |a| < 1 ∧ b ∉ Icc (ln ((2 / |a|) * (1 / |a| - sqrt (1 - (a^2)))) - (1 / |a| - sqrt (1 - a^2)))
                            (ln ((2 / |a|) * (1 / |a| + sqrt (1 - (a^2)))) - (1 / |a| + sqrt (1 - a^2)))) :=
by
  sorry

end intersect_ln_curve_l591_591386


namespace difference_of_sums_is_minus_2250_l591_591747

def sum_first_n_arith_seq (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def first_1500_odd_sum : ℕ :=
  sum_first_n_arith_seq 1 2 1500

def first_1500_even_minus_3_sum: ℕ :=
  sum_first_n_arith_seq (-1) 2 1500

theorem difference_of_sums_is_minus_2250 :
  first_1500_even_minus_3_sum - first_1500_odd_sum = -2250 :=
by {
  sorry
}

end difference_of_sums_is_minus_2250_l591_591747


namespace minimum_k_l591_591462

-- Function and tangent condition definitions
def f (x : ℝ) : ℝ := x * (Real.log x) + 2 * x - 1

def tangent_line (x y : ℝ) : Prop := 3 * x - y - 2 = 0

-- Minimum integer value problem
theorem minimum_k (k : ℤ) (x : ℝ) (hx : 0 < x) 
  (hf_tangent : ∀ x y, f(1) = y → tangent_line 1 y)
  (h_k : k > (f (x + 1)) / x) : 
  k = 5 :=
sorry

end minimum_k_l591_591462


namespace female_employees_l591_591876

theorem female_employees (total_employees male_employees : ℕ) 
  (advanced_degree_male_adv: ℝ) (advanced_degree_female_adv: ℝ) (prob: ℝ) 
  (h1 : total_employees = 450) 
  (h2 : male_employees = 300)
  (h3 : advanced_degree_male_adv = 0.10) 
  (h4 : advanced_degree_female_adv = 0.40)
  (h5 : prob = 0.4) : 
  ∃ F : ℕ, 0.10 * male_employees + (advanced_degree_female_adv * F + (1 - advanced_degree_female_adv) * F) / total_employees = prob ∧ F = 150 :=
by
  sorry

end female_employees_l591_591876


namespace right_triangle_square_side_length_l591_591587

variables (P Q R D E F G : Type) 
variables (nine twelve : ℝ) -- lengths in cm
variables (PR PQ : ℝ) -- sides of the right triangle PQR

noncomputable def hypotenuse (a b : ℝ) := (a ^ 2 + b ^ 2).sqrt
noncomputable def side_length_square (s : ℝ) := 540 / 111

theorem right_triangle_square_side_length:
  ∀ (PR PQ: ℝ),
  PR = 12 → PQ = 9 →
  hypotenuse PR PQ = 15 →
  (∃ s : ℝ, 1 / (s / 15) + 1 / (s / 7.2) = 1 / 12 + 1 / 7.2 - 1 / 6) →
  s = side_length_square (540 / 111)
  sorry

end right_triangle_square_side_length_l591_591587


namespace minimal_square_area_l591_591003

noncomputable def smallest_area_square : ℕ := 36

theorem minimal_square_area
  (r1_r2_l w2 : ℕ)
  (r1 : r1_r2_l = 4)
  (r2 : r1_r2_l = 5)
  (h1 : r1 = 2 ∧ r1_r2_l = 4)
  (h2 : r2 = 3 ∧ r1_r2_l = 5)
  (parallel_sides : true) -- sides are parallel to the sides of the square
  (no_overlap : true) -- rectangles do not overlap
  : (minimal_square_area 2 4 3 5 = some 36) := sorry

end minimal_square_area_l591_591003


namespace fifth_friend_payment_l591_591422

def contributions (a b c d e : ℕ) : Prop :=
  a + b + c + d + e = 120 ∧
  a = (1 / 3 : ℕ) * (b + c + d + e) ∧
  b = (1 / 4 : ℕ) * (a + c + d + e) ∧
  c = (1 / 5 : ℕ) * (a + b + d + e)

theorem fifth_friend_payment (a b c d e : ℕ) (h : contributions a b c d e) : e = 13 :=
sorry

end fifth_friend_payment_l591_591422


namespace monotonic_decreasing_interval_l591_591621

variable (x : ℝ)

def f (x : ℝ) : ℝ := x - log x 

noncomputable def f' (x : ℝ) := deriv f x

theorem monotonic_decreasing_interval : ( {x : ℝ | 0 < x ∧ x < 1} = { x | x ∈ Ioo 0 1}) ↔ ( ∀ x, f' x < 0 ) :=
sorry

end monotonic_decreasing_interval_l591_591621


namespace marbles_jack_gave_l591_591535

-- Definitions based on conditions
def initial_marbles : ℕ := 22
def final_marbles : ℕ := 42

-- Theorem stating that the difference between final and initial marbles Josh collected is the marbles Jack gave
theorem marbles_jack_gave :
  final_marbles - initial_marbles = 20 :=
  sorry

end marbles_jack_gave_l591_591535


namespace seashells_total_l591_591589

theorem seashells_total (s m : Nat) (hs : s = 18) (hm : m = 47) : s + m = 65 := 
by
  -- We are just specifying the theorem statement here
  sorry

end seashells_total_l591_591589


namespace police_emergency_number_has_prime_divisor_gt_7_l591_591026

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := sorry

end police_emergency_number_has_prime_divisor_gt_7_l591_591026


namespace police_emergency_number_has_prime_divisor_gt_7_l591_591029

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := sorry

end police_emergency_number_has_prime_divisor_gt_7_l591_591029


namespace extreme_values_b_equals_4_range_of_b_for_monotonic_in_delta_l591_591830

noncomputable def f (x b : ℝ) := (x^2 + b * x + b) * real.sqrt (1 - 2 * x)

theorem extreme_values_b_equals_4 : 
  let b := 4 in
  (∀ x, (f x b) ≥ 0) ∧ (∀ x, (f x b) ≤ 4) :=
by
  sorry

theorem range_of_b_for_monotonic_in_delta: 
  (∀ x, (0 < x) → (x < 1/3) → (∀ b, (f' x b) ≥ 0)) → b ≤ 1/9 :=
by
  sorry


end extreme_values_b_equals_4_range_of_b_for_monotonic_in_delta_l591_591830


namespace infinite_solutions_x3_y5_eq_z2_l591_591956

theorem infinite_solutions_x3_y5_eq_z2 :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x^3 + y^5 = z^2) ∧
  ∀ k : ℕ, (k ≥ 1) → (k^10 * x, k^6 * y, k^15 * z) satisfies x^3 + y^5 = z^2 :=
sorry

end infinite_solutions_x3_y5_eq_z2_l591_591956


namespace carrie_profit_l591_591752

def hours_per_day : ℕ := 4
def days : ℕ := 6
def pay_rate : ℝ := 35
def cost_of_supplies : ℝ := 150
def sales_tax_rate : ℝ := 0.07

theorem carrie_profit : let total_hours := (hours_per_day : ℝ) * (days : ℝ),
                            earnings := total_hours * pay_rate,
                            tax := earnings * sales_tax_rate,
                            earnings_after_tax := earnings - tax,
                            profit := earnings_after_tax - cost_of_supplies
                        in profit = 631.20 :=
by
  sorry

end carrie_profit_l591_591752


namespace group_meal_cost_l591_591057

-- Defining the conditions mentioned in the problem
def cost_per_adult_meal : ℕ := 3
def total_people : ℕ := 12
def kids : ℕ := 7
def adults : ℕ := total_people - kids

-- The total cost for the group's meals
def total_cost : ℕ := adults * cost_per_adult_meal

-- Statement to be proved
theorem group_meal_cost (cost_per_adult_meal : ℕ) (total_people : ℕ) (kids : ℕ) :
  total_cost cost_per_adult_meal total_people kids = 15 :=
  by sorry 

end group_meal_cost_l591_591057


namespace seashells_total_l591_591590

theorem seashells_total (s m : Nat) (hs : s = 18) (hm : m = 47) : s + m = 65 := 
by
  -- We are just specifying the theorem statement here
  sorry

end seashells_total_l591_591590


namespace smallest_even_number_of_consecutive_sum_750_l591_591282

theorem smallest_even_number_of_consecutive_sum_750 :
  (∃ n : ℕ, (∀ (i : ℕ), i < 25 → even (n + 2 * i)) ∧ ((∑ i in finset.range 25, (n + 2 * i)) = 750)) → 
  (∃ n : ℕ, (∀ (i : ℕ), i < 25 → even (n + 2 * i)) ∧ n = 6) :=
sorry

end smallest_even_number_of_consecutive_sum_750_l591_591282


namespace ambulance_ride_cost_correct_l591_591534

noncomputable def total_bill : ℝ := 12000
noncomputable def medication_percentage : ℝ := 0.40
noncomputable def imaging_tests_percentage : ℝ := 0.15
noncomputable def surgical_procedure_percentage : ℝ := 0.20
noncomputable def overnight_stays_percentage : ℝ := 0.25
noncomputable def food_cost : ℝ := 300
noncomputable def consultation_fee : ℝ := 80

noncomputable def ambulance_ride_cost := total_bill - (food_cost + consultation_fee)

theorem ambulance_ride_cost_correct :
  ambulance_ride_cost = 11620 :=
by
  sorry

end ambulance_ride_cost_correct_l591_591534


namespace inequality_proof_l591_591825

open Real

noncomputable def f (t x : ℝ) : ℝ := t * x - (t - 1) * log x - t

theorem inequality_proof (t x : ℝ) (h_t : t ≤ 0) (h_x : x > 1) : 
  f t x < exp (x - 1) - 1 :=
sorry

end inequality_proof_l591_591825


namespace highest_degree_horizontal_asymptote_l591_591072

-- Define the degree of a polynomial
def degree (p : Polynomial ℝ) : ℕ :=
  p.natDegree

-- The given polynomial
def denominator : Polynomial ℝ :=
  3 * X^6 - 2 * X^3 + X - 4

-- The condition from the problem statement
def denominator_degree : ℕ :=
  degree denominator

-- The proof statement
theorem highest_degree_horizontal_asymptote (p : Polynomial ℝ) :
  (degree denominator = 6) → (degree p ≤ 6) := 
  by
  intro h
  have h_deg := h
  sorry

end highest_degree_horizontal_asymptote_l591_591072


namespace range_of_a_l591_591503

open Real

theorem range_of_a
  (a : ℝ)
  (curve : ∀ θ : ℝ, ∃ p : ℝ × ℝ, p = (a + 2 * cos θ, a + 2 * sin θ))
  (distance_two_points : ∀ θ : ℝ, dist (0,0) (a + 2 * cos θ, a + 2 * sin θ) = 2) :
  (-2 * sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < 2 * sqrt 2) :=
sorry

end range_of_a_l591_591503


namespace train_passes_platform_in_450_seconds_l591_591004

theorem train_passes_platform_in_450_seconds :
  ∀ (length_train length_platform time_tree : ℕ),
  length_train = 2000 →
  length_platform = 2500 →
  time_tree = 200 →
  let speed := length_train / time_tree in
  let total_distance := length_train + length_platform in
  let time_platform := total_distance / speed in
  time_platform = 450 :=
by
  intros length_train length_platform time_tree h_train h_platform h_time_tree
  let speed := length_train / time_tree
  let total_distance := length_train + length_platform
  let time_platform := total_distance / speed
  simp [h_train, h_platform, h_time_tree] at *
  exact calc
    total_distance / speed
      = (2000 + 2500) / (2000 / 200) : by simp [speed]
  ... = 450 : by norm_num

end train_passes_platform_in_450_seconds_l591_591004


namespace cricket_initial_matches_l591_591504

theorem cricket_initial_matches (x : ℝ) :
  (0.28 * x + 60 = 0.52 * (x + 60)) → x = 120 :=
by
  sorry

end cricket_initial_matches_l591_591504


namespace trig_identity_solution_l591_591131

theorem trig_identity_solution
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h: sin α / cos α = cos β / (1 - sin β)) :
  2 * α - β = π / 2 := 
sorry

end trig_identity_solution_l591_591131


namespace smallest_m_div_18_l591_591269

noncomputable def smallest_multiple_18 : ℕ :=
  900

theorem smallest_m_div_18 : (∃ m: ℕ, (m % 18 = 0) ∧ (∀ d ∈ m.digits 10, d = 9 ∨ d = 0) ∧ ∀ k: ℕ, k % 18 = 0 → (∀ d ∈ k.digits 10, d = 9 ∨ d = 0) → m ≤ k) → 900 / 18 = 50 :=
by
  intro h
  sorry

end smallest_m_div_18_l591_591269


namespace hyperbola_equation_l591_591968

theorem hyperbola_equation (e : ℝ) (a : ℝ) (b : ℝ) :
  e = 2 → a = 2 → y^2 = 8 * x → ∃ (h : ℝ), 
  (e = 2 ∧ (1 : ℂ) * (h = sqrt (3 * (a ^ 2) + 4 ))) ( b = sqrt (3 * a ^ 2))  ∧
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ⇔ (x^2 / 4 - y^2 / 12 = 1)),
  
begin
  sorry
end

end hyperbola_equation_l591_591968


namespace friend_gives_amount_l591_591101

theorem friend_gives_amount :
  let earnings := [18, 23, 28, 35, 45] in
  let total := List.sum earnings in
  let equal_share := total / 5 in
  let friend_45 := 45 in
  friend_45 - equal_share = 15.2 :=
by
  -- Begin proof
  sorry
  -- End proof

end friend_gives_amount_l591_591101


namespace num_pairs_solution_l591_591848

theorem num_pairs_solution :
  let a := {a : ℝ // 0 < a},
      b := {b : ℤ // 3 ≤ b ∧ b ≤ 300} in
  (∃ a b, ((Real.log a) / (Real.log b)) ^ 3 = (Real.log (a ^ 3)) / (Real.log b)) →
  3 * 298 = 894 :=
by {
  sorry
}

end num_pairs_solution_l591_591848


namespace sum_of_ages_l591_591975

namespace AgeProblem

-- Definitions based on the given conditions
def age1 : ℕ := 13
def age2 : ℕ := 14
def age_diff (a b : ℕ) : ℕ := abs (a - b)

-- Theorem statement based on the proof needed
theorem sum_of_ages : age1 + age2 = 27 :=
sorry

end AgeProblem

end sum_of_ages_l591_591975


namespace inequality_nonneg_real_l591_591927

theorem inequality_nonneg_real (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ (2 / (1 + a * b)) ∧ ((1 / (1 + a^2)) + (1 / (1 + b^2)) = (2 / (1 + a * b)) ↔ a = b) :=
sorry

end inequality_nonneg_real_l591_591927


namespace households_with_bike_only_l591_591881

theorem households_with_bike_only (total_households : ℕ) (households_neither : ℕ) 
  (households_both : ℕ) (households_car_incl_both : ℕ) 
  (H1 : total_households = 90) 
  (H2 : households_neither = 11) 
  (H3 : households_both = 20) 
  (H4 : households_car_incl_both = 44) : 
  ∃ (households_bike_only : ℕ), households_bike_only = 35 := 
by 
  have h1 : total_households = 90 := H1
  have h2 : households_neither = 11 := H2
  have h3 : households_both = 20 := H3
  have h4 : households_car_incl_both = 44 := H4
  let households_at_least_one := total_households - households_neither
  have h5 : households_at_least_one = 79 := by
    rw [h1, h2]
    norm_num
  let households_only_car := households_car_incl_both - households_both
  have h6 : households_only_car = 24 := by
    rw [h4, h3]
    norm_num
  let households_bike_only := households_at_least_one - (households_only_car + households_both)
  have h7 : households_bike_only = 35 := by
    rw [h5, h6, h3]
    norm_num
  use households_bike_only
  exact h7

end households_with_bike_only_l591_591881


namespace bridge_must_hold_weight_l591_591532

def weight_of_full_can (soda_weight empty_can_weight : ℕ) : ℕ :=
  soda_weight + empty_can_weight

def total_weight_of_full_cans (num_full_cans weight_per_full_can : ℕ) : ℕ :=
  num_full_cans * weight_per_full_can

def total_weight_of_empty_cans (num_empty_cans empty_can_weight : ℕ) : ℕ :=
  num_empty_cans * empty_can_weight

theorem bridge_must_hold_weight :
  let num_full_cans := 6
  let soda_weight := 12
  let empty_can_weight := 2
  let num_empty_cans := 2
  let weight_per_full_can := weight_of_full_can soda_weight empty_can_weight
  let total_full_cans_weight := total_weight_of_full_cans num_full_cans weight_per_full_can
  let total_empty_cans_weight := total_weight_of_empty_cans num_empty_cans empty_can_weight
  total_full_cans_weight + total_empty_cans_weight = 88 := by
  sorry

end bridge_must_hold_weight_l591_591532


namespace infinite_geometric_series_common_ratio_l591_591736

theorem infinite_geometric_series_common_ratio 
  (a S r : ℝ) 
  (ha : a = 400) 
  (hS : S = 2500)
  (h_sum : S = a / (1 - r)) :
  r = 0.84 :=
by
  -- Proof will go here
  sorry

end infinite_geometric_series_common_ratio_l591_591736


namespace probability_route_X_is_8_over_11_l591_591894

-- Definitions for the graph paths and probabilities
def routes_from_A_to_B (X Y : Nat) : Nat := 2 + 6 + 3

def routes_passing_through_X (X Y : Nat) : Nat := 2 + 6

def probability_passing_through_X (total_routes passing_routes : Nat) : Rat :=
  (passing_routes : Rat) / (total_routes : Rat)

theorem probability_route_X_is_8_over_11 :
  let total_routes := routes_from_A_to_B 2 3
  let passing_routes := routes_passing_through_X 2 3
  probability_passing_through_X total_routes passing_routes = 8 / 11 :=
by
  -- Assumes correct route calculations from the conditions and aims to prove the probability value
  sorry

end probability_route_X_is_8_over_11_l591_591894


namespace numberOfFriends_l591_591228

def moorCandiesDivisors : ℕ := 2016

theorem numberOfFriends (n : ℕ) (h : 1 ≤ n) : 
  ∃ k, nat.factors moorCandiesDivisors = [2, 2, 2, 2, 2, 3, 3, 7] →
  ∃ d, d = 35 :=
sorry

end numberOfFriends_l591_591228


namespace math_problem_l591_591549

noncomputable def f (a : ℕ → ℤ) (x : ℤ) (n : ℕ) :=
  a (n + 1) * x^3 - a n * x^2 - a (n + 2) * x + 1

noncomputable def f_prime_at_extremum (a : ℕ → ℤ) (n : ℕ) :=
  3 * a(n+1) - 2 * a(n) - a(n+2)

noncomputable def a_seq (n : ℕ) : ℤ :=
  if n = 1 then 1 else if n = 2 then 2 else 2^(n-1)

noncomputable def b (n : ℕ) : ℤ :=
  Int.log2 (a_seq (n + 1))

theorem math_problem :
  (∀ n : ℕ, n > 0 → f_prime_at_extremum a_seq n = 0) →
  (⟦ ∑ k in Finset.range 2018, (2018 : ℤ) / (b k * b (k+1)) ⟧ = 2017) :=
by
  sorry

end math_problem_l591_591549


namespace sum_cis_is_cis_100_l591_591062

open Real

-- Definition of cis using Euler's formula
def cis (θ : ℝ) : ℂ := complex.exp (complex.I * θ * real.pi / 180)

-- Sum of angles in a specified range with given step
noncomputable def sum_cis_angles : ℂ :=
  ∑ i in finset.range 13, cis (40 + i * 10)

-- Prove that the sum of cis angles can be expressed in the form r * cis θ with θ = 100 degrees
theorem sum_cis_is_cis_100 :
  ∃ r : ℝ, r > 0 ∧ sum_cis_angles = r * cis 100 := 
begin
  sorry
end

end sum_cis_is_cis_100_l591_591062


namespace seating_arrangement_six_people_l591_591051

theorem seating_arrangement_six_people : 
  ∃ (n : ℕ), n = 216 ∧ 
  (∀ (a b c d e f : ℕ),
    -- Alice, Bob, and Carla indexing
    1 ≤ a ∧ a ≤ 6 ∧ 
    1 ≤ b ∧ b ≤ 6 ∧ 
    1 ≤ c ∧ c ≤ 6 ∧ 
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
    (a ≠ b + 1 ∧ a ≠ b - 1) ∧
    (a ≠ c + 1 ∧ a ≠ c - 1) ∧
    
    -- Derek, Eric, and Fiona indexing
    1 ≤ d ∧ d ≤ 6 ∧ 
    1 ≤ e ∧ e ≤ 6 ∧ 
    1 ≤ f ∧ f ≤ 6 ∧ 
    d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    (d ≠ e + 1 ∧ d ≠ e - 1) ∧
    (d ≠ f + 1 ∧ d ≠ f - 1)) -> 
  n = 216 := 
sorry

end seating_arrangement_six_people_l591_591051


namespace initial_boys_count_l591_591739

variable (q : ℕ) -- total number of children initially in the group
variable (b : ℕ) -- number of boys initially in the group

-- Initial condition: 60% of the group are boys initially
def initial_boys (q : ℕ) : ℕ := 6 * q / 10

-- Change after event: three boys leave, three girls join
def boys_after_event (b : ℕ) : ℕ := b - 3

-- After the event, the number of boys is 50% of the total group
def boys_percentage_after_event (b : ℕ) (q : ℕ) : Prop :=
  boys_after_event b = 5 * q / 10

theorem initial_boys_count :
  ∃ b q : ℕ, b = initial_boys q ∧ boys_percentage_after_event b q → b = 18 := 
sorry

end initial_boys_count_l591_591739


namespace midpoint_AM_point_X_l591_591578

noncomputable def midpoint (p1 p2 : Point) : Point := sorry

theorem midpoint_AM_point_X
  {A B C D M X : Point} (hMBC : M = midpoint B C)
  (hPerp : Perp D M B C)
  (hIntersect : Intersect (segment A M) (segment B D) = X)
  (hLength : length (segment A C) = 2 * length (segment B X)) :
  X = midpoint A M :=
by
  sorry

end midpoint_AM_point_X_l591_591578


namespace paint_house_l591_591167

theorem paint_house (n s h : ℕ) (h_pos : 0 < h)
    (rate_eq : ∀ (x : ℕ), 0 < x → ∃ t : ℕ, x * t = n * h) :
    (n + s) * (nh / (n + s)) = n * h := 
sorry

end paint_house_l591_591167


namespace line_equation_l591_591126

open Real

-- Define the points A, B, and C
def A : ℝ × ℝ := ⟨1, 4⟩
def B : ℝ × ℝ := ⟨3, 2⟩
def C : ℝ × ℝ := ⟨2, -1⟩

-- Definition for a line passing through point C
-- and having equal distance to points A and B
def is_line_equation (l : ℝ → ℝ → Prop) :=
  ∀ x y, (l x y ↔ (x + y - 1 = 0 ∨ x - 2 = 0))

-- Our main statement
theorem line_equation :
  ∃ l : ℝ → ℝ → Prop, is_line_equation l ∧ (l 2 (-1)) :=
by
  sorry  -- Proof goes here.

end line_equation_l591_591126


namespace tire_miles_usage_l591_591726

theorem tire_miles_usage (total_tires road_tires total_miles : ℕ) (h : total_tires = 6) (h1 : road_tires = 4) (h2 : total_miles = 40000) :
  (total_miles * road_tires) / total_tires = 26667 :=
by
  -- By the provided conditions
  have T : total_tires = 6 := h,
  have R : road_tires = 4 := h1,
  have M : total_miles = 40000 := h2,
  -- Now we proceed to the division
  calc
    (total_miles * road_tires) / total_tires
    = (40000 * 4) / 6 : by rw [M, R, T]
    = 160000 / 6 : by norm_num
    = 26666.66 : by norm_num
    = 26667 : by norm_num

end tire_miles_usage_l591_591726


namespace geometric_sequence_properties_l591_591797

noncomputable def geometric_sequence_sum 
  (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_properties 
  (a : ℝ) (a1 : ℝ = 1 / 16) (q ≠ 1) (S3_eq : geometric_sequence_sum a1 q 3 = 4 * geometric_sequence_sum a1 q 2 - 5 / 16) :
    (∀ n, a_n = 2^(n-5)) ∧ 
    ((0 < a ∧ a ≠ 1) → 
      ((a > 1 → ∀ n, T_n = -10 * log a 2) ∧ 
       (0 < a → a < 1 → ∀ n, T_n = -10 * log a 2))) :=
by
  sorry

end geometric_sequence_properties_l591_591797


namespace common_difference_is_neg4_max_sum_of_6_max_n_for_positive_S_l591_591630

noncomputable def first_term := 23
noncomputable def common_difference : Int := -4

theorem common_difference_is_neg4 (d : Int) 
  (h_pos_6 : (first_term + 5 * d) > 0) 
  (h_neg_7 : (first_term + 6 * d) < 0) : 
  d = -4 := 
sorry

theorem max_sum_of_6 : S_n = 78 :=
sorry

theorem max_n_for_positive_S (n : ℕ)
  (h_sn : S_n > 0) : 
  n <= 12 :=
sorry

end common_difference_is_neg4_max_sum_of_6_max_n_for_positive_S_l591_591630


namespace matrix_product_arithmetic_sequence_l591_591776

theorem matrix_product_arithmetic_sequence :
  (\begin
    have : Π (k : ℕ) (m : ℕ), \prod_{k = 1}^m \begin{pmatrix} 1 & 2k \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & (m * (m + 1)) \\ 0 & 1 \end{pmatrix},
    sorry,
these "k" and "m" are iterable and their range is 1~50. Thus, the final matrix is:
   ∏ (k = 1)^{50} = \begin{pmatrix} 1 & 2550 \\ 0 & 1 \end{pmatrix}
end sorry

end matrix_product_arithmetic_sequence_l591_591776


namespace probability_all_same_color_l591_591005

theorem probability_all_same_color :
  let total_marbles := 5 + 7 + 4
  let total_draws := nat.choose total_marbles 3
  let prob_all_red := (5 / total_marbles) * ((5 - 1) / (total_marbles - 1)) * ((5 - 2) / (total_marbles - 2))
  let prob_all_white := (7 / total_marbles) * ((7 - 1) / (total_marbles - 1)) * ((7 - 2) / (total_marbles - 2))
  let prob_all_green := (4 / total_marbles) * ((4 - 1) / (total_marbles - 1)) * ((4 - 2) / (total_marbles - 2))
  prob_all_red + prob_all_white + prob_all_green = 43 / 280 := sorry

end probability_all_same_color_l591_591005


namespace javier_first_throw_l591_591909

theorem javier_first_throw 
  (second third first : ℝ)
  (h1 : first = 2 * second)
  (h2 : first = (1 / 2) * third)
  (h3 : first + second + third = 1050) :
  first = 300 := 
by sorry

end javier_first_throw_l591_591909


namespace trig_identity_l591_591145

-- Define the conditions
def terminal_side_point (P : ℝ × ℝ) := P = (-3, 4)

-- Define the trigonometric functions based on the given conditions
def sin_alpha {P : ℝ × ℝ} (h : terminal_side_point P) : ℝ :=
  P.2 / Real.sqrt (P.1^2 + P.2^2)

def cos_alpha {P : ℝ × ℝ} (h : terminal_side_point P) : ℝ :=
  P.1 / Real.sqrt (P.1^2 + P.2^2)

-- Prove the mathematical problem
theorem trig_identity (P : ℝ × ℝ) (h : terminal_side_point P) :
  sin_alpha h + 2 * cos_alpha h = -2 / 5 :=
by
  -- We omit the proof for this example
  sorry

end trig_identity_l591_591145


namespace new_sandbox_volume_l591_591681

theorem new_sandbox_volume (L W H : ℝ) (h : L * W * H = 10) :
  2 * 2 * 2 * (L * W * H) = 80 :=
by {
  calc
  2 * 2 * 2 * (L * W * H) = 8 * (L * W * H) : by ring
  ... = 8 * 10 : by rw h
  ... = 80 : by norm_num
}

end new_sandbox_volume_l591_591681


namespace neg_p_equiv_l591_591472

theorem neg_p_equiv (p : Prop) : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end neg_p_equiv_l591_591472


namespace angle_between_unit_vectors_l591_591843

noncomputable def angle_between_vectors (a b : ℝ) (x : ℝ) : ℝ := sorry

theorem angle_between_unit_vectors
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (h : ∀ x : ℝ, ∥x • a + b∥ ≥ 1/2) :
  real.arccos (a ⬝ b) ∈ set.Icc (π / 6) (5 * π / 6) :=
sorry

end angle_between_unit_vectors_l591_591843


namespace factorization_l591_591402

theorem factorization (a b : ℤ) : a^2 * b - 2 * a * b + b = b * (a - 1)^2 := by
  sorry

end factorization_l591_591402


namespace maximum_earnings_l591_591227

-- Define Mary's working conditions
def max_hours_per_week := 45
def regular_hourly_rate := 8
def overtime_rate_increase := 0.25
def regular_hours_limit := 20

-- Calculate the earnings for the first 20 hours
def earnings_first_20_hours := regular_hours_limit * regular_hourly_rate

-- Calculate the overtime hourly rate
def overtime_hourly_rate := regular_hourly_rate * (1 + overtime_rate_increase)

-- Calculate the maximum number of overtime hours Mary can work
def max_overtime_hours := max_hours_per_week - regular_hours_limit

-- Calculate the earnings for the overtime hours
def earnings_overtime := max_overtime_hours * overtime_hourly_rate

-- Calculate the total maximum earnings
def total_max_earnings := earnings_first_20_hours + earnings_overtime

-- Statement to be proved: 
theorem maximum_earnings : total_max_earnings = 410 := by
  -- Sorry, skipping the proof
  sorry

end maximum_earnings_l591_591227


namespace handshakes_count_l591_591742

theorem handshakes_count (players_per_team referees teams : ℕ) (shakes_per_referee : ℕ)
  (players_per_team = 5) (teams = 2) (referees = 3) (shakes_per_referee = 3) :
  let inter_team_handshakes := players_per_team * players_per_team
  let total_players := players_per_team * teams
  let player_referee_handshakes := total_players * shakes_per_referee
  inter_team_handshakes + player_referee_handshakes = 55 :=
by
  intros
  have h1 : inter_team_handshakes = 25 := by
    unfold inter_team_handshakes
    rw [players_per_team]
    norm_num
  have h2 : player_referee_handshakes = 30 := by
    unfold player_referee_handshakes total_players
    rw [players_per_team, teams, shakes_per_referee]
    norm_num
  rw [h1, h2]
  norm_num
  sorry

end handshakes_count_l591_591742


namespace calculate_expression_l591_591746

theorem calculate_expression :
  (-1: ℤ) ^ 53 + 2 ^ (4 ^ 4 + 3 ^ 3 - 5 ^ 2) = -1 + 2 ^ 258 := 
by
  sorry

end calculate_expression_l591_591746


namespace math_problem_l591_591553

theorem math_problem (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p + q + r = 0) :
    (p^2 * q^2 / ((p^2 - q * r) * (q^2 - p * r)) +
    p^2 * r^2 / ((p^2 - q * r) * (r^2 - p * q)) +
    q^2 * r^2 / ((q^2 - p * r) * (r^2 - p * q))) = 1 :=
by
  sorry

end math_problem_l591_591553


namespace find_n_l591_591552

theorem find_n (n : ℕ) (composite_n : n > 1 ∧ ¬Prime n) : 
  ((∃ m : ℕ, ∀ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n → d + 1 ∣ m ∧ 1 < d + 1 ∧ d + 1 < m) ↔ 
    (n = 4 ∨ n = 8)) :=
by sorry

end find_n_l591_591552


namespace domain_of_f_l591_591774

def f (x : ℝ) : ℝ := Real.log (x^2 - 6)

theorem domain_of_f :
  {x : ℝ | x^2 - 6 > 0} = {x : ℝ | x < -Real.sqrt 6} ∪ {x : ℝ | x > Real.sqrt 6} := by
  sorry

end domain_of_f_l591_591774


namespace contractor_original_days_l591_591014

noncomputable def original_days (total_laborers absent_laborers working_laborers days_worked : ℝ) : ℝ :=
  (working_laborers * days_worked) / (total_laborers - absent_laborers)

-- Our conditions:
def total_laborers : ℝ := 21.67
def absent_laborers : ℝ := 5
def working_laborers : ℝ := 16.67
def days_worked : ℝ := 13

-- Our main theorem:
theorem contractor_original_days :
  original_days total_laborers absent_laborers working_laborers days_worked = 10 := 
by
  sorry

end contractor_original_days_l591_591014


namespace sum_of_distinct_sums_of_roots_l591_591565

theorem sum_of_distinct_sums_of_roots (h1 : ∃ a b : ℤ, a < 0 ∧ b < 0 ∧ a ≠ b ∧ a * b = 24) :
  (∑ r in { (a + b) | (∃ a b : ℤ, a < 0 ∧ b < 0 ∧ a ≠ b ∧ a * b = 24) }.to_finset, r) = 60 :=
by
  sorry

end sum_of_distinct_sums_of_roots_l591_591565


namespace perimeter_sum_of_cd_is_five_l591_591713

theorem perimeter_sum_of_cd_is_five :
  let a := (1, 0)
  let b := (5, 4)
  let c := (6, 3)
  let d := (6, 0)
  let dist (p1 p2 : ℝ × ℝ) := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let perimeter := dist a b + dist b c + dist c d + dist d a
  let (c1, d1) := (0, 2) in -- Placeholder, correct c and d found by analyzing perimeter expression
  c1 + d1 = 5 :=
by 
  sorry

end perimeter_sum_of_cd_is_five_l591_591713


namespace function_nested_value_l591_591856

def f (x : ℝ) : ℝ := 4 * x^2 - 6

theorem function_nested_value : f(f(2)) = 394 :=
by
  let f_in = f(2)
  calc
    f(f_in) = f(10) : by rfl
    ... = 394 : by rfl

-- skip the actual proof step with sorry
sorry

end function_nested_value_l591_591856


namespace sum_non_solutions_eq_neg_64_over_3_l591_591207

noncomputable def A : ℝ := 3
noncomputable def B : ℝ := 8
noncomputable def C : ℝ := 40 / 3
theorem sum_non_solutions_eq_neg_64_over_3 :
  let f (x : ℝ) := ((x + B) * (A * x + 40)) / ((x + C) * (x + 8))
  (∀ x, f x = 3) → -8 + (-40 / 3) = -64 / 3 := by
  intro _ 
  norm_num
  sorry

end sum_non_solutions_eq_neg_64_over_3_l591_591207


namespace total_gadgets_sold_after_15_days_l591_591918

/-- Kamil's selling pattern is an arithmetic sequence with the first term 2 gadgets 
and a common difference of 4 gadgets. We need to determine the total number of gadgets he sold after 15 days. -/
theorem total_gadgets_sold_after_15_days :
  let a1 := 2
  let d := 4
  ∀ n : ℕ, ∑ i in Finset.range 15, (a1 + i * d) = 450 :=
by
  sorry

end total_gadgets_sold_after_15_days_l591_591918


namespace fraction_zero_solution_l591_591868

theorem fraction_zero_solution (x : ℝ) (h : (|x| - 2) / (x - 2) = 0) : x = -2 :=
sorry

end fraction_zero_solution_l591_591868


namespace common_difference_is_one_l591_591195

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 1 + (n - 1) * d

-- Given conditions
def given_conditions : Prop :=
  a 1 = 2 ∧ (a 3 + a 5 = 10)

-- Proof statement
theorem common_difference_is_one (h_arith : is_arithmetic_sequence a d) (h_given : given_conditions) :
  d = 1 := 
sorry

end common_difference_is_one_l591_591195


namespace magnitude_quotient_l591_591061

open Complex

theorem magnitude_quotient : 
  abs ((1 + 2 * I) / (2 - I)) = 1 := 
by 
  sorry

end magnitude_quotient_l591_591061


namespace polynomial_evaluation_l591_591750

theorem polynomial_evaluation : 
  (x : ℤ) (h : x = 3) : x ^ 6 - 4 * x ^ 2 + 3 * x = 702 :=
by
  rw [h]
  sorry

end polynomial_evaluation_l591_591750


namespace incorrect_statements_for_quadratic_inequality_l591_591456

-- Definitions for given conditions and required expressions
def quadratic_solution_set_empty (a b c : ℝ) : Prop := 
  a > 0 ∧ (b^2 - 4 * a * c) ≤ 0

def minimum_value_expression (a b c : ℝ) (x0 : ℝ) : ℝ :=
  (a + 4 * c) / (b - a)

theorem incorrect_statements_for_quadratic_inequality (a b c x0 : ℝ) :
  (quadratic_solution_set_empty a b c ↔ M = ∅) → 
  (M = { x | x ≠ x0 } ∧ a < b → minimum_value_expression a 4 c = 2 - 2 * real.sqrt 2) :=
sorry

end incorrect_statements_for_quadratic_inequality_l591_591456


namespace tan_alpha_is_three_halves_l591_591430

theorem tan_alpha_is_three_halves (α : ℝ) (h : Real.tan (α - 5 * Real.pi / 4) = 1 / 5) : 
  Real.tan α = 3 / 2 :=
by
  sorry

end tan_alpha_is_three_halves_l591_591430


namespace least_k_for_expression_l591_591313

theorem least_k_for_expression (k : ℤ) (h : 0.00010101 * 10^k > 10) : k ≥ 5 := 
sorry

end least_k_for_expression_l591_591313


namespace arithmetic_sequence_sum_l591_591543

variable {α : Type} [LinearOrderedField α]

theorem arithmetic_sequence_sum (a : ℕ → α) (d a1 : α)
  (h_arith : ∀ n, a n = a1 + n * d)
  (h_condition : a 2 + a 5 + a 8 = 18) : 
  let S : ℕ → α := λ n, n * (2 * a1 + (n - 1) * d) / 2 in
  S 9 = 54 := 
by
  sorry

end arithmetic_sequence_sum_l591_591543


namespace highest_page_number_l591_591947

def plenty_digits : Set ℕ := {0, 1, 2, 3, 4, 6, 7, 8, 9}

def has_only_fifteen_fives (n : ℕ) : Prop :=
  (n.digits 10).count 5 ≤ 15

def no_consecutive_digits (n : ℕ) : Prop :=
  ∀ (d1 d2 : Nat) (h : d1 < (n.digits 10).length - 1), 
  (n.digits 10).get ⟨d1, h⟩ ≠ (n.digits 10).get ⟨d1 + 1, by linarith⟩

/-- How far can Pat number the pages of his scrapbook given the restrictions? -/
theorem highest_page_number :
  ∃ (n : ℕ), (n ≤ 74) ∧ ¬ ∃ (m : ℕ), m > n ∧ m ≤ 74 ∧
  has_only_fifteen_fives m ∧ no_consecutive_digits m :=
sorry

end highest_page_number_l591_591947


namespace fully_loaded_truck_weight_l591_591394

def empty_truck : ℕ := 12000
def weight_soda_crate : ℕ := 50
def num_soda_crates : ℕ := 20
def weight_dryer : ℕ := 3000
def num_dryers : ℕ := 3
def weight_fresh_produce : ℕ := 2 * (weight_soda_crate * num_soda_crates)

def total_loaded_truck_weight : ℕ :=
  empty_truck + (weight_soda_crate * num_soda_crates) + weight_fresh_produce + (weight_dryer * num_dryers)

theorem fully_loaded_truck_weight : total_loaded_truck_weight = 24000 := by
  sorry

end fully_loaded_truck_weight_l591_591394


namespace right_angle_triangle_congruence_l591_591307

theorem right_angle_triangle_congruence (A B C D : Prop) :
  (A ↔ (hypotenuse_and_one_leg : Two_Conditions)) ∧ 
  (B ↔ (two_acute_angles : Two_Conditions)) ∧ 
  (C ↔ (one_acute_angle_and_hypotenuse : Two_Conditions)) ∧ 
  (D ↔ (two_legs : Two_Conditions)) ∧ 
  ¬(congruence : Two_Conditions → Prop) B :=
sorry

end right_angle_triangle_congruence_l591_591307


namespace monotonic_interval_k1_f_eq_2g_single_root_p_x_two_distinct_zeros_l591_591470

noncomputable def f (k : ℝ) (x : ℝ) := log (k * x)
def g (x : ℝ) := log (x + 1)
def h (x : ℝ) := x / (x^2 + 1)
def p (m : ℝ) (x : ℝ) := h x + m*x / (1 + x)

-- Problem 1
theorem monotonic_interval_k1 : ∀ x : ℝ, 0 < x → (f 1 x + g x) is_monotonic on set.Ioi 0 :=
sorry

-- Problem 2
theorem f_eq_2g_single_root : ∀ k : ℝ, (∃! x : ℝ, f k x = 2 * g x) ↔ k < 0 ∨ k = 4 :=
sorry

-- Problem 3
theorem p_x_two_distinct_zeros : ∀ m : ℝ, (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ Ioo (-1) 1 ∧ x2 ∈ Ioo (-1) 1 ∧ p m x1 = 0 ∧ p m x2 = 0) ↔ m ∈ set.Ioo (-1) 0 ∪ {(-1 - real.sqrt 2) / 2} :=
sorry

end monotonic_interval_k1_f_eq_2g_single_root_p_x_two_distinct_zeros_l591_591470


namespace coefficient_x3_y4_in_expansion_l591_591657

theorem coefficient_x3_y4_in_expansion :
  let n := 7
  let a := 3
  let b := 4
  binom n a = 35 :=
by
  sorry

end coefficient_x3_y4_in_expansion_l591_591657


namespace find_b_l591_591756

def f (x : ℝ) : ℝ := 5 * x + 3

theorem find_b : ∃ b : ℝ, f b = -2 ∧ b = -1 := by
  have h : 5 * (-1 : ℝ) + 3 = -2 := by norm_num
  use -1
  simp [f, h]
  sorry

end find_b_l591_591756


namespace find_h_l591_591923

noncomputable def odot (a b : ℝ) : ℝ :=
  a + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + ...)))

theorem find_h : (9 + Real.sqrt (h^2 + Real.sqrt (h^2 + Real.sqrt (h^2 + ...)))) = 12 → h = Real.sqrt 6 :=
by
  sorry

end find_h_l591_591923


namespace radius_and_distance_of_sphere_l591_591081

noncomputable def CircumsphereRadiusAndDistance (R : ℝ) (a b c d e f : ℝ) : Prop :=
∃ (R' OS : ℝ), 
 R' = (1 / 3) * R ∧ 
 OS^2 = (16 / 9) * R^2 - (1 / 9) * (a^2 + b^2 + c^2 + d^2 + e^2 + f^2)

theorem radius_and_distance_of_sphere (R a b c d e f : ℝ) :
 CircumsphereRadiusAndDistance R a b c d e f :=
begin
  use (1 / 3) * R,
  use (4 / 3) * Real.sqrt (R^2 - (1 / 16) * (a^2 + b^2 + c^2 + d^2 + e^2 + f^2)),
  split,
  { 
    -- Proof of R' = (1 / 3) * R
    sorry 
  },
  { 
    -- Proof of OS^2 = (16 / 9) * R^2 - (1 / 9) * (a^2 + b^2 + c^2 + d^2 + e^2 + f^2)
    sorry 
  }
end

end radius_and_distance_of_sphere_l591_591081


namespace constant_term_expansion_l591_591407

theorem constant_term_expansion : 
  let T_r := λ r : ℕ, (-1)^r * Nat.choose 5 r * (λ x : ℝ, x^(r - 5)) in
  let expansion := λ x : ℝ, (x^2 + 2) * (∑ i in Finset.range 6, T_r i x) in
  expansion 1 = -12 :=
by
  sorry

end constant_term_expansion_l591_591407


namespace zero_point_in_interval_l591_591281

-- Definitions based on the conditions.
def f (x : ℝ) : ℝ := Real.log x + x - 3

-- Mathematical proof problem statement.
theorem zero_point_in_interval :
  (0 < 2) → (2 < 3) → monotone_on f (set.Ioi 0) →
  f 2 < 0 → f 3 > 0 → ∃ x, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  -- The conditions and goal are provided.
  -- Proof not required, hence using sorry.
  intro h₀ h₁ h₂ h₃ h₄
  sorry

end zero_point_in_interval_l591_591281


namespace curve_is_line_l591_591773

theorem curve_is_line : ∀ (r θ : ℝ), r = 2 / (2 * Real.sin θ - Real.cos θ) → ∃ m b, ∀ (x y : ℝ), x = r * Real.cos θ → y = r * Real.sin θ → y = m * x + b :=
by
  intros r θ h
  sorry

end curve_is_line_l591_591773


namespace millennium_run_time_l591_591279

theorem millennium_run_time (M A B : ℕ) (h1 : B = 100) (h2 : B = A + 10) (h3 : A = M - 30) : M = 120 := by
  sorry

end millennium_run_time_l591_591279


namespace sum_power_inequality_l591_591859

theorem sum_power_inequality
  (n : ℕ)
  (a b : Fin n → ℝ)
  (m : ℝ)
  (h_pos : ∀ i, a i > 0 ∧ b i > 0)
  (h_m : m > 0 ∨ m < -1) :
  (∑ i, (a i) ^ (m + 1) / (b i) ^ m) >=
  ((∑ i, a i) ^ (m + 1) / (∑ i, b i) ^ m) :=
sorry

end sum_power_inequality_l591_591859


namespace stddev_of_lengths_l591_591648

-- Define the lengths of the first five strands
def l1 := 20 
def l2 := 26 
def l3 := 22 
def l4 := 20 
def l5 := 22

-- Define the given average length
def avg := 25

-- Define the equation to find the length of the sixth strand
def l6 := 6 * avg - (l1 + l2 + l3 + l4 + l5)

-- Define the list of all six strand lengths
def lengths := [l1, l2, l3, l4, l5, l6]

-- Define the function to calculate the mean
def mean (lst : List ℝ) : ℝ := lst.sum / lst.length

-- Define the function to calculate the variance
def variance (lst : List ℝ) : ℝ := (lst.foldl (λ acc x => acc + (x - mean lst) ^ 2) 0) / lst.length

-- Define the function to calculate the standard deviation
def stddev (lst : List ℝ) : ℝ := Real.sqrt (variance lst)

-- The main statement to prove
theorem stddev_of_lengths : stddev lengths = 7 := by
  sorry

end stddev_of_lengths_l591_591648


namespace crabapple_sequences_count_l591_591364

theorem crabapple_sequences_count (n m : ℕ) (h1 : n = 12) (h2 : m = 5) :
  n^m = 248832 := by
  rw [h1, h2]
  norm_num

end crabapple_sequences_count_l591_591364


namespace polygon_sides_eq_six_l591_591283

theorem polygon_sides_eq_six
  (n : ℕ)
  (sum_interior_angles : ℕ → ℕ := λ n, (n - 2) * 180)
  (sum_exterior_angles_is_360 : ∀ n, 360 = 360) :
  (sum_interior_angles n = 2 * 360) → n = 6 :=
by
  -- Proof omitted
  intro h
  sorry

end polygon_sides_eq_six_l591_591283


namespace domain_of_fx_l591_591976

theorem domain_of_fx {x : ℝ} : (2 * x) / (x - 1) = (2 * x) / (x - 1) ↔ x ∈ {y : ℝ | y ≠ 1} :=
by
  sorry

end domain_of_fx_l591_591976


namespace percentage_increase_overtime_rate_l591_591333

noncomputable def regular_rate : ℝ := 14
noncomputable def regular_hours : ℝ := 40
noncomputable def total_hours_worked : ℝ := 57.88
noncomputable def total_earnings : ℝ := 998

noncomputable def overtime_hours : ℝ := total_hours_worked - regular_hours
noncomputable def regular_earnings : ℝ := regular_hours * regular_rate
noncomputable def overtime_earnings : ℝ := total_earnings - regular_earnings
noncomputable def overtime_rate : ℝ := overtime_earnings / overtime_hours

theorem percentage_increase_overtime_rate :
  (overtime_rate - regular_rate) / regular_rate * 100 ≈ 74.93 :=
by
  sorry

end percentage_increase_overtime_rate_l591_591333


namespace continuous_function_solution_l591_591390

theorem continuous_function_solution (n : ℕ) (hn : 0 < n) : 
  ∀ (f : ℝ → ℝ), (∀ x, 0 ≤ x ∧ x ≤ 1 → continuous_at f x) → 
  (f 0 = 0) → (f 1 = 0) → 
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ f (x + 1 / n) = f x :=
begin
  sorry,
end

end continuous_function_solution_l591_591390


namespace xyz_value_l591_591608

theorem xyz_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
    (hx : x * (y + z) = 162)
    (hy : y * (z + x) = 180)
    (hz : z * (x + y) = 198)
    (h_sum : x + y + z = 26) :
    x * y * z = 2294.67 :=
by
  sorry

end xyz_value_l591_591608


namespace minimum_moves_to_find_coin_l591_591638

/--
Consider a circle of 100 thimbles with a coin hidden under one of them. 
You can check four thimbles per move. After each move, the coin moves to a neighboring thimble.
Prove that the minimum number of moves needed to guarantee finding the coin is 33.
-/
theorem minimum_moves_to_find_coin 
  (N : ℕ) (hN : N = 100) (M : ℕ) (hM : M = 4) :
  ∃! k : ℕ, k = 33 :=
by sorry

end minimum_moves_to_find_coin_l591_591638


namespace parallel_lines_k_values_l591_591841

theorem parallel_lines_k_values :
  ∀ k : ℝ,
  let l1 := (k - 3) * x + (4 - k) * y + 1 = 0,
      l2 := 2 * (k - 3) * x - 2 * y + 3 = 0,
  (l1 = l2) ↔ (k = 3 ∨ k = 5) :=
by sorry

end parallel_lines_k_values_l591_591841


namespace max_mom_money_difference_l591_591943

theorem max_mom_money_difference:
  let tuesday_amount := 8 in
  let wednesday_amount := 5 * tuesday_amount in
  let thursday_amount := wednesday_amount + 9 in
  (thursday_amount - tuesday_amount = 41) :=
by
  sorry

end max_mom_money_difference_l591_591943


namespace salary_increase_l591_591613

theorem salary_increase 
  (A1 : ℝ) (Manager_salary : ℝ) (A1_val : A1 = 1300)
  (Manager_salary_val : Manager_salary = 3400) : 
  (Total1 : ℝ) (Total2 : ℝ) (A2 : ℝ) (Increase : ℝ) 
  (Total1_val : Total1 = 20 * A1)
  (Total2_val : Total2 = Total1 + Manager_salary)
  (A2_val : A2 = Total2 / 21)
  (Increase_val : Increase = A2 - A1) :
  Increase = 100 := 
by
  -- Skipping the proof here
  sorry

end salary_increase_l591_591613


namespace relationship_l591_591689

noncomputable def a : ℝ := 3 ^ 0.7
noncomputable def b : ℝ := log 3 2
noncomputable def c : ℝ := log (1 / 3) 10

theorem relationship : a > b ∧ b > c :=
by sorry

end relationship_l591_591689


namespace _l591_591878

noncomputable def geometry_problem : Prop :=
  let O : Type := Point
  ∃ (A B M P : Point) (AM BM AP OM : Real) (angle_AMB : Angle),
    Circle O
    ∧ Chord A B
    ∧ Diameter M
    ∧ IntersectsAt A B M
    ∧ angle_AMB = 60
    ∧ distance A M = 10
    ∧ distance B M = 4
    ∧ MidpointOf P A B
    ∧ PerpBisector O P
    ∧ Perpendicular O P
    ∧ IsoscelesTriangle A B O
    ∧ Pythagorean_theorem A B O M P
    -> distance O M = 6

#check geometry_problem

end _l591_591878


namespace equivalent_statements_l591_591308

variable (P Q : Prop)

theorem equivalent_statements :
  (P → Q) ↔ (P → Q) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) := by
  sorry

end equivalent_statements_l591_591308


namespace red_car_count_l591_591317

-- Define the ratio and the given number of black cars
def ratio_red_to_black (R B : ℕ) : Prop := R * 8 = B * 3

-- Define the given number of black cars
def black_cars : ℕ := 75

-- State the theorem we want to prove
theorem red_car_count : ∃ R : ℕ, ratio_red_to_black R black_cars ∧ R = 28 :=
by
  sorry

end red_car_count_l591_591317


namespace javier_first_throw_distance_l591_591911

-- Definitions based on the conditions
def distance_second_throw : ℕ := 150  -- As solved in the solution, the second throw is 150 meters

theorem javier_first_throw_distance
  (distance_second_throw : ℕ)
  (h_first_throw : 2 * distance_second_throw = 2 * 150)
  (h_third_throw : 4 * distance_second_throw = 4 * 150)
  (h_sum_throws : 2 * distance_second_throw + distance_second_throw + 4 * distance_second_throw = 1050) :
  2 * distance_second_throw = 300 :=
by
  -- Introduce variables for the throw distances
  let distance_first_throw := 2 * distance_second_throw
  let distance_third_throw := 4 * distance_second_throw
  -- Use the provided hypothesis and solve for the first throw distance
  have h_sum : distance_first_throw + distance_second_throw + distance_third_throw = 1050,
    from h_sum_throws
  sorry

end javier_first_throw_distance_l591_591911


namespace problem_statement_l591_591116

-- Definitions from the conditions
variable (a : ℕ → ℝ) (q : ℝ) (b : ℕ → ℝ)
variable (a1_gt_1 : a 1 > 1)
variable (q_gt_0 : q > 0)
variable (b_eq_log2_a : ∀ n, b n = Real.log (a n) / Real.log 2)
variable (b3_eq_2 : b 3 = 2)
variable (b5_eq_0 : b 5 = 0)
variable (a_geom : ∀ n, a (n + 1) = a n * q)

-- Proof problem: Prove the following equivalences
theorem problem_statement :
  (∀ n, b (n + 1) - b n = Real.log q / Real.log 2) ∧
  (∀ n, b 1 + (n - 1) * (Real.log q / Real.log 2) = b n) ∧
  (∀ n, ∑ i in Finset.range n, b 1 + i * (Real.log q / Real.log 2) = - (n - 1 : ℝ) * n / 2 + 9 * n / 2) ∧
  (∀ n, a n = 2 ^ (5 - n)) := 
by
  sorry

end problem_statement_l591_591116


namespace ratio_of_sector_FOG_l591_591945

theorem ratio_of_sector_FOG (O : Type*) [metric_space O] [normed_group O] [normed_space ℝ O]
(points E F G A B : O)
(hOE : angle A O E = real.pi / 3)
(hFOB : angle F O B = real.pi / 2)
(hAGO : angle A G O = real.pi / 9) :
  let sector_FOG := 10 * real.pi / 180 -- 10 degrees converted to radians
      ratio := sector_FOG / (2 * real.pi) -- Ratio of the sector angle to the full circle
  in ratio = 1 / 36 :=
sorry

end ratio_of_sector_FOG_l591_591945


namespace express_set_l591_591401

-- Define the condition and the set for which the problem is framed
def set_condition (x : ℕ) : Prop :=
  x - 1 ≤ 2

def expected_set : set ℕ := {0, 1, 2, 3}

-- The theorem statement
theorem express_set :
  {x ∈ ℕ | set_condition x} = expected_set :=
  sorry

end express_set_l591_591401


namespace percent_below_m_plus_d_l591_591682

variable {α : Type*} [Distrib α]

-- Define the conditions
def is_symmetric_about_mean (distribution : Set α) (m : α) : Prop := sorry -- precise definition of symmetry
def within_one_std_dev (distribution : Set α) (m : α) (d : α) : percent := 84

-- Define the question
def percent_less_than (distribution : Set α) (x : α) : percent := sorry -- function to calculate the percentage of distribution below x

-- The theorem equivalent to the problem statement
theorem percent_below_m_plus_d (distribution : Set α) (m d : α)
  (H_symm : is_symmetric_about_mean distribution m)
  (H_within_std : within_one_std_dev distribution m d = 84) :
  percent_less_than distribution (m + d) = 42 :=
sorry

end percent_below_m_plus_d_l591_591682


namespace stutterer_square_number_unique_l591_591017

-- Definitions based on problem conditions
def is_stutterer (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧ (n / 100 = (n % 1000) / 100) ∧ ((n % 1000) % 100 = n % 10 * 10 + n % 10)

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The theorem statement
theorem stutterer_square_number_unique : ∃ n, is_stutterer n ∧ is_square n ∧ n = 7744 :=
by
  sorry

end stutterer_square_number_unique_l591_591017


namespace transformation_impossible_l591_591378

-- Definitions for colors and sequences
inductive Color
| Red
| Blue
| Yellow

open Color

-- Initial and target sequences
def initial_sequence : List Color := List.cycle [Red, Blue] ++ [Yellow]
def target_sequence : List Color := List.cycle [Red, Blue] ++ [Yellow, Blue]

/-- Attempting to transform the initial sequence to the target sequence
    is impossible given the operations allowed. -/
theorem transformation_impossible (initial_sequence = List.cycle [Red, Blue] ++ [Yellow]) 
  (target_sequence = List.cycle [Red, Blue] ++ [Yellow, Blue]) :
  (∀ op : List Color → List Color, valid_operation op → op initial_sequence ≠ target_sequence) :=
by
  sorry

end transformation_impossible_l591_591378


namespace reflection_midpoint_sum_l591_591579

theorem reflection_midpoint_sum (A B : ℝ × ℝ) (hA : A = (3, 2)) (hB : B = (15, 18)) :
  let N := (λ p q : ℝ × ℝ, ((p.1 + q.1) / 2, (p.2 + q.2) / 2)) A B in
  let N' := (λ p q : ℝ × ℝ, ((-p.1 - q.1) / 2, (p.2 + q.2) / 2)) A B in
  N' = (-9, 10) → -9 + 10 = 1 :=
by intros; simp; sorry

end reflection_midpoint_sum_l591_591579


namespace find_a9_l591_591118

axiom positive_sequence (a : ℕ+ → ℝ) : Prop :=
∀ n : ℕ+, a n > 0

axiom functional_equation (a : ℕ+ → ℝ) : Prop :=
∀ p q : ℕ+, a (p + q) = a p * a q

axiom initial_condition (a : ℕ+ → ℝ) : a 2 = 4

theorem find_a9 (a : ℕ+ → ℝ)
  (h1 : positive_sequence a)
  (h2 : functional_equation a)
  (h3 : initial_condition a) : a 9 = 512 :=
sorry

end find_a9_l591_591118


namespace problem_statement_l591_591104

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem problem_statement :
  (∀ x : ℝ, f (x) = 0 → x = - Real.pi / 6) ∧ (∀ x : ℝ, f (x) = 4 * Real.cos (2 * x - Real.pi / 6)) := sorry

end problem_statement_l591_591104


namespace translate_parabola_l591_591294

theorem translate_parabola (x : ℝ) :
  let y := x^2 + 1 in
  let y_translated := (x - 3)^2 - 1 in
  y_translated = y := sorry

end translate_parabola_l591_591294


namespace fraction_of_girls_l591_591183

variable (T G B : ℕ) -- The total number of students, number of girls, and number of boys
variable (x : ℚ) -- The fraction of the number of girls

-- Definitions based on the given conditions
def fraction_condition : Prop := x * G = (1/6) * T
def ratio_condition : Prop := (B : ℚ) / (G : ℚ) = 2
def total_students : Prop := T = B + G

-- The statement we need to prove
theorem fraction_of_girls (h1 : fraction_condition T G x)
                          (h2 : ratio_condition B G)
                          (h3 : total_students T G B):
  x = 1/2 :=
by
  sorry

end fraction_of_girls_l591_591183


namespace digit_of_sequence_l591_591866

theorem digit_of_sequence (n: ℕ) (d: ℕ) : 
  (n = 110 → d = 1) := 
by {
  -- Set up range of numbers
  let digits := (List.range' 1 75).reverse.map (λ x, toString x).join,
  -- n-th digit of the concatenated string
  let target_digit := digits.nth (n - 1),
  have h1 : target_digit = some '1', sorry,
  injection h1 with d_eq,
  assumption 
}

end digit_of_sequence_l591_591866


namespace tangent_line_at_1_eq_x_minus_1_monotonic_intervals_range_of_a_l591_591461

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_1_eq_x_minus_1 :
  ∃ m b, m = 1 ∧ b = -1 ∧ (∀ x, f(x) = m * x + b) :=
sorry

theorem monotonic_intervals :
  (∀ x : ℝ, x ∈ set.Ioo 0 (1/Real.exp 1) → f'(x) < 0) ∧
  (∀ x : ℝ, x ∈ set.Ioo (1/Real.exp 1) +∞ → f'(x) > 0) :=
sorry

theorem range_of_a :
  (∀ x : ℝ, x ∈ set.Icc (1/Real.exp 1) (Real.exp 1) → f(x) ≤ a * x - 1) → 
  set.Ici (Real.exp 1 - 1) a :=
sorry

end tangent_line_at_1_eq_x_minus_1_monotonic_intervals_range_of_a_l591_591461


namespace find_a7_l591_591117

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n, a (n + 1) ≥ a n) ∧ 
  (∀ n, a (n + 2) = 3 * a (n + 1) - a n)

theorem find_a7
  (a : ℕ → ℤ)
  (h_seq : sequence a)
  (a6_eq : a 6 = 280) :
  a 7 = 733 :=
sorry

end find_a7_l591_591117


namespace license_plate_problem_l591_591164

noncomputable def license_plate_ways : ℕ :=
  let letters := 26
  let digits := 10
  let both_same := letters * digits * 1 * 1
  let digits_adj_same := letters * digits * 1 * letters
  let letters_adj_same := letters * digits * digits * 1
  digits_adj_same + letters_adj_same - both_same

theorem license_plate_problem :
  9100 = license_plate_ways :=
by
  -- Skipping the detailed proof for now
  sorry

end license_plate_problem_l591_591164


namespace total_running_time_l591_591480

def distance_per_side : ℝ := 50
def speed_9kmh : ℝ := 9 * 1000 / 3600
def speed_7kmh : ℝ := 7 * 1000 / 3600
def num_hurdles_each_side : ℕ := 2
def time_per_hurdle : ℝ := 5

def total_distance : ℝ := 4 * distance_per_side
def total_time_9kmh : ℝ := 2 * (distance_per_side / speed_9kmh)
def total_time_7kmh : ℝ := 2 * (distance_per_side / speed_7kmh)
def total_time_hurdles : ℝ := 4 * num_hurdles_each_side * time_per_hurdle

theorem total_running_time : 
  total_time_9kmh + total_time_7kmh + total_time_hurdles = 131.44 :=
by 
  sorry

end total_running_time_l591_591480


namespace probability_neither_nearsighted_l591_591645

-- Definitions based on problem conditions
def P_A : ℝ := 0.4
def P_not_A : ℝ := 1 - P_A
def event_B₁_not_nearsighted : Prop := true
def event_B₂_not_nearsighted : Prop := true

-- Independence assumption
variables (indep_B₁_B₂ : event_B₁_not_nearsighted) (event_B₂_not_nearsighted)

-- Theorem statement
theorem probability_neither_nearsighted (H1 : P_A = 0.4) (H2 : P_not_A = 0.6)
  (indep_B₁_B₂ : event_B₁_not_nearsighted ∧ event_B₂_not_nearsighted) :
  P_not_A * P_not_A = 0.36 :=
by
  -- Proof omitted
  sorry

end probability_neither_nearsighted_l591_591645


namespace police_emergency_number_has_prime_divisor_gt_seven_l591_591025

theorem police_emergency_number_has_prime_divisor_gt_seven (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
by
  sorry

end police_emergency_number_has_prime_divisor_gt_seven_l591_591025


namespace candy_in_each_bag_l591_591784

theorem candy_in_each_bag (total_candy : ℕ) (bags : ℕ) (h1 : total_candy = 16) (h2 : bags = 2) : total_candy / bags = 8 :=
by {
    sorry
}

end candy_in_each_bag_l591_591784


namespace prob_no_risk_factors_given_no_X_l591_591877

open ProbabilityTheory

def total_population : ℕ := 200

def prob_one_risk_factor (X Y Z : Prop) : ℝ := 0.07
def prob_two_risk_factors (X Y Z : Prop) : ℝ := 0.12
def prob_all_three_given_X_Y (X Y Z : Prop) : ℝ := 0.4

theorem prob_no_risk_factors_given_no_X (X Y Z : Prop) 
  (h_pop : total_population = 200)
  (h_one : prob_one_risk_factor X Y Z = 0.07)
  (h_two : prob_two_risk_factors X Y Z = 0.12)
  (h_conditional : prob_all_three_given_X_Y X Y Z = 0.4) :
  (70 / 122 : ℝ) = sorry :=
sorry

end prob_no_risk_factors_given_no_X_l591_591877


namespace police_emergency_number_has_prime_gt_7_l591_591030

theorem police_emergency_number_has_prime_gt_7 (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
sorry

end police_emergency_number_has_prime_gt_7_l591_591030


namespace unique_outstanding_participant_l591_591506

noncomputable def outstanding_participant {n : ℕ} (G : Fin n → Fin n → Prop) (a : Fin n) : Prop :=
  ∀ b : Fin n, b ≠ a → (G a b ∨ (∃ c : Fin n, G a c ∧ G c b))

theorem unique_outstanding_participant {n : ℕ} (h : 3 ≤ n) (G : Fin n → Fin n → Prop)
    (unique : ∃! a : Fin n, outstanding_participant G a) :
    ∃ a : Fin n, outstanding_participant G a ∧ (∀ b : Fin n, a ≠ b → G a b) :=
begin
  sorry
end

end unique_outstanding_participant_l591_591506


namespace smallest_distance_from_point_to_line_l591_591617

-- Define the line equation as a function
def line_eq (x y : ℝ) := y = (4 / 3) * x - 100

-- Define the point (0, 0)
def point : ℝ × ℝ := (0, 0)

-- Define the distance between a point (x0, y0) and the line Ax + By + C = 0
def distance_from_point_to_line (A B C x0 y0 : ℝ) : ℝ := 
  |A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)

-- Theorem to prove the smallest distance from (0,0) to y = (4 / 3) * x - 100 is 60
theorem smallest_distance_from_point_to_line : 
  distance_from_point_to_line (-4) 3 (-300) 0 0 = 60 := by
  sorry

end smallest_distance_from_point_to_line_l591_591617


namespace whipped_cream_needed_l591_591400

def total_days : ℕ := 15
def odd_days_count : ℕ := 8
def even_days_count : ℕ := 7

def pumpkin_pies_on_odd_days : ℕ := 3 * odd_days_count
def apple_pies_on_odd_days : ℕ := 2 * odd_days_count

def pumpkin_pies_on_even_days : ℕ := 2 * even_days_count
def apple_pies_on_even_days : ℕ := 4 * even_days_count

def total_pumpkin_pies_baked : ℕ := pumpkin_pies_on_odd_days + pumpkin_pies_on_even_days
def total_apple_pies_baked : ℕ := apple_pies_on_odd_days + apple_pies_on_even_days

def tiffany_pumpkin_pies_consumed : ℕ := 2
def tiffany_apple_pies_consumed : ℕ := 5

def remaining_pumpkin_pies : ℕ := total_pumpkin_pies_baked - tiffany_pumpkin_pies_consumed
def remaining_apple_pies : ℕ := total_apple_pies_baked - tiffany_apple_pies_consumed

def whipped_cream_for_pumpkin_pies : ℕ := 2 * remaining_pumpkin_pies
def whipped_cream_for_apple_pies : ℕ := remaining_apple_pies

def total_whipped_cream_needed : ℕ := whipped_cream_for_pumpkin_pies + whipped_cream_for_apple_pies

theorem whipped_cream_needed : total_whipped_cream_needed = 111 := by
  -- Proof omitted
  sorry

end whipped_cream_needed_l591_591400


namespace inequality_ab_bc_ca_max_l591_591685

theorem inequality_ab_bc_ca_max (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|))
  ≤ 1 + (1 / 3) * (a + b + c)^2 := sorry

end inequality_ab_bc_ca_max_l591_591685


namespace length_CD_squared_l591_591949

noncomputable def parabola (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 2

def midpoint (p1 p2 p_midpoint : ℝ × ℝ) : Prop :=
  p_midpoint.1 = (p1.1 + p2.1) / 2 ∧ p_midpoint.2 = (p1.2 + p2.2) / 2

def slope_of_tangent (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (fderiv ℝ f x) 1

theorem length_CD_squared:
  ∃ (x_C x_D : ℝ) (y_C y_D : ℝ),
    parabola x_C = y_C ∧
    parabola x_D = y_D ∧
    midpoint (x_C, y_C) (x_D, y_D) (0, 0) ∧
    slope_of_tangent parabola x_C = 10 ∧
    ((x_D - x_C)^2 + (y_D - y_C)^2) = 8 :=
by
  sorry

end length_CD_squared_l591_591949


namespace no_numbers_equal_7_times_sum_of_digits_l591_591162

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem no_numbers_equal_7_times_sum_of_digits :
  ∀ n : ℕ, n > 0 ∧ n ≤ 1000 → n = 7 * (digit_sum n) → false :=
by
  intros n h.bounds h.eq
  sorry

end no_numbers_equal_7_times_sum_of_digits_l591_591162


namespace right_triangle_hypotenuse_l591_591717

theorem right_triangle_hypotenuse (a b : ℝ) (h : ℝ) (ha : a = 24) (hb : b = 10) :
  h = Real.sqrt (a^2 + b^2) -> h = 26 :=
by {
  -- use the provided conditions to formalize the proof
  sorry
}

end right_triangle_hypotenuse_l591_591717


namespace number_of_wins_and_losses_l591_591292

theorem number_of_wins_and_losses (x y : ℕ) (h1 : x + y = 15) (h2 : 3 * x + y = 41) :
  x = 13 ∧ y = 2 :=
sorry

end number_of_wins_and_losses_l591_591292


namespace train_speed_including_stoppages_l591_591767

theorem train_speed_including_stoppages (speed_excl_stoppages : ℝ) (time_stoppages_per_hour : ℝ) :
  speed_excl_stoppages = 48 → time_stoppages_per_hour = 10 → (40 : ℝ) = speed_excl_stoppages * (50.0 / 60.0) :=
begin
  intros h_speed h_stoppages,
  rw [h_speed, h_stoppages],
  sorry
end

end train_speed_including_stoppages_l591_591767


namespace tangent_expression_equals_two_l591_591688

noncomputable def eval_tangent_expression : ℝ :=
  (1 + Real.tan (3 * Real.pi / 180)) * (1 + Real.tan (42 * Real.pi / 180))

theorem tangent_expression_equals_two :
  eval_tangent_expression = 2 :=
by sorry

end tangent_expression_equals_two_l591_591688


namespace hyperbola_sum_l591_591182

theorem hyperbola_sum (h k a c b : ℝ) 
  (h_def : h = 0) 
  (k_def : k = 3) 
  (a_def : a = 3) 
  (c_def : c = 5) 
  (hyperbola_relation : c^2 = a^2 + b^2) 
  (calc_b : b = real.sqrt (c^2 - a^2)) :
  h + k + a + b = 10 :=
by
  -- Definitions based on the problem 
  have h := h_def
  have k := k_def
  have a := a_def
  have c := c_def
  have b := calc_b
  -- Summing the defined values
  sorry

end hyperbola_sum_l591_591182


namespace part1_part2_l591_591653

-- Define the operation * on integers
def op (a b : ℤ) : ℤ := a^2 - b + a * b

-- Prove that 2 * 3 = 7 given the defined operation
theorem part1 : op 2 3 = 7 := 
sorry

-- Prove that (-2) * (op 2 (-3)) = 1 given the defined operation
theorem part2 : op (-2) (op 2 (-3)) = 1 := 
sorry

end part1_part2_l591_591653


namespace proposition_verification_l591_591980

-- Definitions and Propositions
def prop1 : Prop := (∀ x, x = 1 → x^2 - 3 * x + 2 = 0) ∧ (∃ x, x ≠ 1 ∧ x^2 - 3 * x + 2 = 0)
def prop2 : Prop := (∀ x, ¬ (x^2 - 3 * x + 2 = 0 → x = 1) → (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0))
def prop3 : Prop := ¬ (∃ x > 0, x^2 + x + 1 < 0) → (∀ x ≤ 0, x^2 + x + 1 ≥ 0)
def prop4 : Prop := ¬ (∃ p q : Prop, (p ∨ q) → ¬p ∧ ¬q)

-- Final theorem statement
theorem proposition_verification : prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 := by 
  sorry

end proposition_verification_l591_591980


namespace line_c_parallel_to_a_b_l591_591620

noncomputable def line (α : Type*) := α → α → Prop
noncomputable def parallel {α : Type*} (a b : line α) := ∀ p₁ p₂, a p₁ p₂ → b p₁ p₂
noncomputable def plane (α : Type*) := set (line α)

variable {α : Type*} [field α]

variables (a b c : line α) (α₁ β₁ : plane α)

axiom lines_parallel : parallel a b
axiom planes_intersect_along_line : ∃ c, ∀ (l : line α), α₁ l → β₁ l → c l

theorem line_c_parallel_to_a_b (h₁ : α₁ a) (h₂ : β₁ b)
  (h₃ : α₁ c) (h₄ : β₁ c) : parallel c a ∧ parallel c b :=
by 
  -- Add the assumptions that a is contained in plane α and b is contained in plane β
  assume h₁ : α₁ a,
  assume h₂ : β₁ b,
  -- intersecting condition 
  let ⟨c, h⟩ := planes_intersect_along_line,
  assume h₃ : α₁ c,
  assume h₄ : β₁ c,
  -- Proof that c is parallel to both a and b
  sorry

end line_c_parallel_to_a_b_l591_591620


namespace slope_of_line_l_l591_591175

theorem slope_of_line_l (l m : Line) (α : ℝ) (h1 : inclination_angle l = α) (h2 : inclination_angle m = 2 * α) 
(h3 : m = line_of_slope (-4 / 3)) : slope l = 2 :=
sorry

end slope_of_line_l_l591_591175


namespace vector_magnitude_l591_591113

theorem vector_magnitude (a b : EuclideanSpace ℝ (Fin 3)) (ha : ‖a‖ = 2) (hb : ‖b‖ = 1) (hab : (inner a b) = 1) :
  ‖2 • a - b‖ = Real.sqrt 13 :=
by
  sorry

end vector_magnitude_l591_591113


namespace seashells_total_l591_591592

theorem seashells_total (sam_seashells : ℕ) (mary_seashells : ℕ) (h1 : sam_seashells = 18) (h2 : mary_seashells = 47) : sam_seashells + mary_seashells = 65 :=
by
  rw [h1, h2]
  exact rfl

end seashells_total_l591_591592


namespace hyperbola_asymptote_y_eq_1_has_m_neg_3_l591_591863

theorem hyperbola_asymptote_y_eq_1_has_m_neg_3
    (m : ℝ)
    (h1 : ∀ x y, (x^2 / (2 * m)) - (y^2 / m) = 1)
    (h2 : ∀ x, 1 = (x^2 / (2 * m))): m = -3 :=
by
  sorry

end hyperbola_asymptote_y_eq_1_has_m_neg_3_l591_591863


namespace areas_of_triangle_and_parallelogram_are_equal_l591_591045

theorem areas_of_triangle_and_parallelogram_are_equal (b : ℝ) :
  let parallelogram_height := 100
  let triangle_height := 200
  let area_parallelogram := b * parallelogram_height
  let area_triangle := (1/2) * b * triangle_height
  area_parallelogram = area_triangle :=
by
  -- conditions
  let parallelogram_height := 100
  let triangle_height := 200
  let area_parallelogram := b * parallelogram_height
  let area_triangle := (1 / 2) * b * triangle_height
  -- relationship
  show area_parallelogram = area_triangle
  sorry

end areas_of_triangle_and_parallelogram_are_equal_l591_591045


namespace T_5_value_l591_591220

noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + (1 / y)^m

theorem T_5_value (y : ℝ) (h : y + 1 / y = 5) : T y 5 = 2525 := 
by {
  sorry
}

end T_5_value_l591_591220


namespace find_multiple_l591_591990

theorem find_multiple (x m : ℝ) (h₁ : 10 * x = m * x - 36) (h₂ : x = -4.5) : m = 2 :=
by
  sorry

end find_multiple_l591_591990


namespace third_square_placed_is_G_l591_591396

-- Definitions representing the squares
inductive Square
| A | B | C | D | E | F | G | H
deriving DecidableEq

-- Order of placement as a list of squares
variable (placement : List Square)
-- The specific condition from the problem statement
def third_square_is_G : Prop :=
  placement.length = 8 ∧
  placement.nth 0 = some Square.F ∧
  placement.nth 1 = some Square.H ∧
  placement.nth 2 = some Square.G ∧
  placement.nth 3 = some Square.D ∧
  placement.nth 4 = some Square.A ∧
  placement.nth 5 = some Square.B ∧
  placement.nth 6 = some Square.C ∧
  placement.nth 7 = some Square.E

-- The theorem we intend to prove
theorem third_square_placed_is_G (placement : List Square) :
  third_square_is_G placement → placement.nth 2 = some Square.G :=
by
  intro h
  dsimp [third_square_is_G] at h
  cases h with _ h1
  cases h1 with _ h2
  exact h2

#check third_square_placed_is_G

end third_square_placed_is_G_l591_591396


namespace sum_of_parabola_distances_l591_591423

def parabola_sum (n : ℕ) : ℝ :=
  1 / (n * (n + 1))

theorem sum_of_parabola_distances :
  (∑ n in Finset.range 2009, parabola_sum (n + 1)) = 2009 / 2010 := by
  sorry

end sum_of_parabola_distances_l591_591423


namespace parent_payment_per_year_l591_591703

noncomputable def former_salary : ℕ := 45000
noncomputable def raise_percentage : ℕ := 20
noncomputable def number_of_kids : ℕ := 9

theorem parent_payment_per_year : 
  (former_salary + (raise_percentage * former_salary / 100)) / number_of_kids = 6000 := by
  sorry

end parent_payment_per_year_l591_591703


namespace max_digit_sum_24hr_format_l591_591699

theorem max_digit_sum_24hr_format : 
  ∃ (h m : ℕ), (0 ≤ h ∧ h < 24) ∧ (0 ≤ m ∧ m < 60) ∧ 
               (let hd1 := h / 10, hd2 := h % 10, md1 := m / 10, md2 := m % 10 in
                (hd1 + hd2 + md1 + md2 = 24)) := 
by 
  existsi 19 -- hours
  existsi 59 -- minutes
  split
  -- hours range
  { split; norm_num },
  -- minutes range
  { split; norm_num },
  -- digit sum calculation
  {
    repeat { rw [nat.div_eq_of_lt, nat.mod_eq_of_lt], norm_num },
    sorry
  }

end max_digit_sum_24hr_format_l591_591699


namespace intersection_of_M_and_N_l591_591476

def U := {0, 1, 2, 3, 4}
def M := {0, 2, 3}
def complement_U_N := {1, 2, 4}
def N := U \ complement_U_N

theorem intersection_of_M_and_N : M ∩ N = {0, 3} :=
by sorry

end intersection_of_M_and_N_l591_591476


namespace david_older_than_scott_l591_591011

-- Define the ages of Richard, David, and Scott
variables (R D S : ℕ)

-- Given conditions
def richard_age_eq : Prop := R = D + 6
def richard_twice_scott : Prop := R + 8 = 2 * (S + 8)
def david_current_age : Prop := D = 14

-- Prove the statement
theorem david_older_than_scott (h1 : richard_age_eq R D) (h2 : richard_twice_scott R S) (h3 : david_current_age D) :
  D - S = 8 :=
  sorry

end david_older_than_scott_l591_591011


namespace zeros_in_expansion_of_squared_nines_l591_591484

theorem zeros_in_expansion_of_squared_nines (n : ℕ) (h : ∀ k, (repeat 9 k).val^2 contains (k-1) zeros) :
  (999999999:ℕ)^2 contains 8 zeros :=
by
  have h₉ : 999999999 = 10^9 - 1 := by sorry
  exact h 9

end zeros_in_expansion_of_squared_nines_l591_591484


namespace johns_work_hours_l591_591204

theorem johns_work_hours (h : ℕ) (h_positive : 0 < h) :
  (80 / h) = 10 ↔ h = 8 := by
  have h1 : 80 - h * 10 = 0 → h = 8 := sorry
  have h2 : ∀ h, 80 - h * 10 = 0 ↔ (10 * (h + 2)) = 100 := sorry
  exact (h2 h).mpr ((h1 h).mpr h_positive)

end johns_work_hours_l591_591204


namespace number_of_solutions_l591_591625

theorem number_of_solutions (x y : ℕ) (h : 4 * x + 7 * y = 600) : 
  ∃! n : ℕ, 
  (1 ≤ x ∧ 1 ≤ y ∧ 4 * x + 7 * y = 600 ∧ x = 1 + 7 * n ∧ y = 85 - 4 * n ∧ 0 ≤ n ∧ n ≤ 21) := 
sorry

end number_of_solutions_l591_591625


namespace parabola_ellipse_tangency_l591_591275

theorem parabola_ellipse_tangency :
  ∃ (a b : ℝ), (∀ x y, y = x^2 - 5 → (x^2 / a) + (y^2 / b) = 1) →
               (∃ x, y = x^2 - 5 ∧ (x^2 / a) + ((x^2 - 5)^2 / b) = 1) ∧
               a = 1/10 ∧ b = 1 :=
by
  sorry

end parabola_ellipse_tangency_l591_591275


namespace right_triangle_hypotenuse_l591_591619

noncomputable def s : ℝ := real.logb 3 5

-- Legs of the triangle
noncomputable def log3_125 : ℝ := real.logb 3 125
noncomputable def log3_48 : ℝ := real.logb 3 48

-- Hypotenuse calculation
theorem right_triangle_hypotenuse (s : ℝ) (log3_125 = 3 * s) (log3_48 = 6 - 4 * s) :
  let h := real.sqrt ((3 * s)^2 + (6 - 4 * s)^2) in
  3 ^ h = 5 ^ 13 :=
by
  sorry

end right_triangle_hypotenuse_l591_591619


namespace distance_hyperbola_vertices_l591_591410

noncomputable def distance_between_vertices (a : ℝ) : ℝ :=
2 * a

theorem distance_hyperbola_vertices :
  ∃ a : ℝ, a^2 = 64 ∧ distance_between_vertices a = 16 :=
by
  have h : a = sqrt 64 := by
    apply sqrt_eq
    rw [pow_two, nat.cast_comm]
  use 8
  rw h
  apply sqrt_pos
  norm_num
  use 8
  split
  exact rfl
  norm_num
  sorry

end distance_hyperbola_vertices_l591_591410


namespace dice_probability_l591_591106

-- Define the condition of rolling four fair four-sided dice.
def four_sided_dice_rolls : list (fin 4) := [1, 2, 3, 4]

-- Define the probability calculation
theorem dice_probability : 
  ∃ p : ℚ, p = 53 / 64 ∧ 
  (∀ (d1 d2 d3 d4 : fin 4), 
    d1 ∈ four_sided_dice_rolls → 
    d2 ∈ four_sided_dice_rolls → 
    d3 ∈ four_sided_dice_rolls → 
    d4 ∈ four_sided_dice_rolls → 
    (d1 + d2 + d3 = d4 ∨ 
     d1 + d2 + d4 = d3 ∨ 
     d1 + d3 + d4 = d2 ∨ 
     d2 + d3 + d4 = d1) → p = 53 / 64) := sorry

end dice_probability_l591_591106


namespace modulus_of_complex_root_l591_591818

theorem modulus_of_complex_root (z : ℂ) (h : z^2 + 1 = 0) : |z| = 1 :=
sorry

end modulus_of_complex_root_l591_591818


namespace imaginary_part_of_conjugate_of_square_l591_591813

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Given the complex number (1 + 2i)
def z : ℂ := 1 + 2 * i

-- Define the square of the complex number
def z_squared := z^2

-- Define the conjugate of the square of the complex number
def z_squared_conj := conj z_squared

-- Statement to prove
theorem imaginary_part_of_conjugate_of_square :
  z_squared_conj.im = -4 :=
by sorry

end imaginary_part_of_conjugate_of_square_l591_591813


namespace a6_is_4_l591_591798

noncomputable def a_sequence (n : ℕ) : ℝ :=
if n = 1 then 1
else if n = 2 then 2
else real.sqrt (3 * n - 2)

theorem a6_is_4 :
  (∀ n : ℕ, n ≥ 2 → 2 * a_sequence n ^ 2 = a_sequence (n + 1) ^ 2 + a_sequence (n - 1) ^ 2) →
  a_sequence 1 = 1 →
  a_sequence 2 = 2 →
  a_sequence 6 = 4 :=
begin
  intros h1 h2 h3,
  -- Proof is omitted
  sorry
end

end a6_is_4_l591_591798


namespace parent_payment_per_year_l591_591704

noncomputable def former_salary : ℕ := 45000
noncomputable def raise_percentage : ℕ := 20
noncomputable def number_of_kids : ℕ := 9

theorem parent_payment_per_year : 
  (former_salary + (raise_percentage * former_salary / 100)) / number_of_kids = 6000 := by
  sorry

end parent_payment_per_year_l591_591704


namespace negation_statement_l591_591168

theorem negation_statement (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) : x^2 - x ≠ 0 :=
by sorry

end negation_statement_l591_591168


namespace part1_part2_part3_l591_591820

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1)*x^2 - (m - 1)*x + (m - 1)

theorem part1 (m : ℝ) : (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 := 
sorry

theorem part2 (m : ℝ) (x : ℝ) : (f m x ≥ (m + 1) * x) ↔ 
  (m = -1 ∧ x ≥ 1) ∨ 
  (m > -1 ∧ (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1)) ∨ 
  (m < -1 ∧ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) := 
sorry

theorem part3 (m : ℝ) : (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) ↔
  m ≥ 1 := 
sorry

end part1_part2_part3_l591_591820


namespace deceased_member_income_l591_591880

theorem deceased_member_income
  (initial_income_4_members : ℕ)
  (initial_members : ℕ := 4)
  (initial_average_income : ℕ := 840)
  (final_income_3_members : ℕ)
  (remaining_members : ℕ := 3)
  (final_average_income : ℕ := 650)
  (total_income_initial : initial_income_4_members = initial_average_income * initial_members)
  (total_income_final : final_income_3_members = final_average_income * remaining_members)
  (income_deceased : ℕ) :
  income_deceased = initial_income_4_members - final_income_3_members :=
by
  -- sorry indicates this part of the proof is left as an exercise
  sorry

end deceased_member_income_l591_591880


namespace determine_ab_l591_591135

theorem determine_ab (a b : ℕ) (h1: a + b = 30) (h2: 2 * a * b + 14 * a = 5 * b + 290) : a * b = 104 := by
  -- the proof would be written here
  sorry

end determine_ab_l591_591135


namespace negation_of_prop_original_l591_591272

-- Definitions and conditions as per the problem
def prop_original : Prop :=
  ∃ x : ℝ, x^2 + x + 1 ≤ 0

def prop_negation : Prop :=
  ∀ x : ℝ, x^2 + x + 1 > 0

-- The theorem states the mathematical equivalence
theorem negation_of_prop_original : ¬ prop_original ↔ prop_negation := 
sorry

end negation_of_prop_original_l591_591272


namespace average_percent_score_l591_591873

theorem average_percent_score :
  let total_score :=
    100 * 10 + 95 * 20 + 85 * 40 + 75 * 30 + 65 * 25 + 55 * 15 + 45 * 10,
    number_of_students := 150
  in total_score / number_of_students = 76 :=
begin
  sorry
end

end average_percent_score_l591_591873


namespace base7_to_base10_l591_591757

theorem base7_to_base10 (a b c : ℕ) (h1 : a = 1) (h2 : b = 3) (h3 : c = 5) : 
  a * 7^2 + b * 7^1 + c * 7^0 = 75 := 
by {
  rw [h1, h2, h3],
  -- Show that 1 * 49 + 3 * 7 + 5 * 1 = 75
  norm_num,
  sorry
}

end base7_to_base10_l591_591757


namespace same_terminal_side_l591_591358

theorem same_terminal_side :
  ¬(angles_with_same_terminal_side (20 * π / 3) (87 * π / 9)) ∧
  ¬(angles_with_same_terminal_side (-π / 3) (22 * π / 3)) ∧
  ¬(angles_with_same_terminal_side (3 * π / 2) (-3 * π / 2)) ∧
  (angles_with_same_terminal_side (-7 * π / 9) (-25 * π / 9)) :=
sorry

end same_terminal_side_l591_591358


namespace rock_collecting_l591_591611

theorem rock_collecting :
  ∀ (S : ℕ), Sydney_total S = Conner_total S ∨ Conner_total S > Sydney_total S -> S ≤ 4 :=
begin
  intros S h,
  sorry
end

def Sydney_total (S : ℕ) : ℕ := 837 + 17 * S

def Conner_total (S : ℕ) : ℕ := 873 + 8 * S

end rock_collecting_l591_591611


namespace find_b_l591_591853

-- Define the conditions as given in the problem
def poly1 (x : ℝ) : ℝ := x^2 - 2 * x - 1
def poly2 (x a b : ℝ) : ℝ := a * x^3 + b * x^2 + 1

-- Define the problem statement using these conditions
theorem find_b (a b : ℤ) (h : ∀ x, poly1 x = 0 → poly2 x a b = 0) : b = -3 :=
sorry

end find_b_l591_591853


namespace exists_circle_through_consecutive_vertices_l591_591243

-- Define a convex polygon as a list of points
structure ConvexPolygon :=
(vertices : List (EuclideanSpace ℝ 2))
(convex : ∀ (P Q R : EuclideanSpace ℝ 2), P ∈ vertices → Q ∈ vertices → R ∈ vertices → ∃ a b : ℝ, a + b = 1 ∧ 0 ≤ a ∧ 0 ≤ b ∧ a • P + b • Q = R)

-- Define a predicate stating that a circle passes through a set of points
def passes_through (C : Set (EuclideanSpace ℝ 2)) (points : List (EuclideanSpace ℝ 2)) : Prop :=
∀ P ∈ points, P ∈ C

-- Define a predicate stating that a circle contains a set of points
def contains (C : Set (EuclideanSpace ℝ 2)) (points : List (EuclideanSpace ℝ 2)) : Prop :=
∀ P ∈ points, P ∈ C

-- The main theorem statement
theorem exists_circle_through_consecutive_vertices {P : ConvexPolygon} :
  ∃ (C : Set (EuclideanSpace ℝ 2)), 
    ∃ (i : ℕ), i < P.vertices.length - 2 ∧ 
    passes_through C [P.vertices.nthLe i (by linarith), P.vertices.nthLe (i+1) (by linarith), P.vertices.nthLe (i+2) (by linarith)] ∧
    contains C P.vertices :=
by sorry

end exists_circle_through_consecutive_vertices_l591_591243


namespace azalea_profit_l591_591730

def num_sheep : Nat := 200
def wool_per_sheep : Nat := 10
def price_per_pound : Nat := 20
def shearer_cost : Nat := 2000

theorem azalea_profit : 
  (num_sheep * wool_per_sheep * price_per_pound) - shearer_cost = 38000 := 
by
  sorry

end azalea_profit_l591_591730


namespace compound_interest_two_years_l591_591406

/-- Given the initial amount, and year-wise interest rates, 
     we want to find the amount in 2 years and prove it equals to a specific value. -/
theorem compound_interest_two_years 
  (P : ℝ) (R1 : ℝ) (R2 : ℝ) (T1 : ℝ) (T2 : ℝ) 
  (initial_amount : P = 7644) 
  (interest_rate_first_year : R1 = 0.04) 
  (interest_rate_second_year : R2 = 0.05) 
  (time_first_year : T1 = 1) 
  (time_second_year : T2 = 1) : 
  (P + (P * R1 * T1) + ((P + (P * R1 * T1)) * R2 * T2) = 8347.248) := 
by 
  sorry

end compound_interest_two_years_l591_591406


namespace chair_cost_l591_591570

/--
Nadine went to a garage sale and spent $56. She bought a table for $34 and 2 chairs.
Each chair cost the same amount.
Prove that one chair cost $11.
-/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) (total_cost : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  total_cost = 56 - 34 →
  total_cost / num_chairs = 11 :=
by
  sorry

end chair_cost_l591_591570


namespace correct_operation_l591_591672

theorem correct_operation :
  let A := 3 * x^3 + 2 * x^3 = 5 * x^6
  let B := (x + 1)^2 = x^2 + 1
  let C := x^8 / x^4 = x^2
  let D := sqrt 4 = 2
  D :=
begin
  -- Proof of the correct operation being D goes here
  sorry
end

end correct_operation_l591_591672


namespace hyperbola_vertex_distance_l591_591415

theorem hyperbola_vertex_distance :
  (∀ x y : ℝ, (x^2 / 64 - y^2 / 49) = 1 → 16) :=
begin
  -- definitions based on conditions
  let a : ℝ := real.sqrt 64,
  have h : a = 8, from real.sqrt_eq_iff_sq_eq.2 (or.inl rfl),
  rw h,
  
  -- statement to be proven
  exact 16
end

end hyperbola_vertex_distance_l591_591415


namespace cannot_cover_all_lattice_points_l591_591384

-- Define a lattice point
structure LatticePoint (ℝ : Type*) := (x y : ℝ)

-- Predicate for a disk with at least radius 5
def Disk := {c : LatticePoint ℝ × ℝ // c.2 ≥ 5}

-- Condition that interiors of disks are disjoint
def DisjointInteriors (D : set Disk) : Prop :=
  ∀ (d1 d2 : Disk), d1 ≠ d2 → dist (d1.1.1, d2.1.1) > d1.1.2 + d2.1.2

-- The main theorem to prove
theorem cannot_cover_all_lattice_points (D : set Disk) :
  (∀ p : LatticePoint ℝ, ∃ d ∈ D, dist (p, d.1.1) < d.1.2) →
  DisjointInteriors D →
  false :=
by
  sorry

end cannot_cover_all_lattice_points_l591_591384


namespace car_trip_duration_l591_591680

theorem car_trip_duration :
  ∃ T : ℝ, T = 8 ∧ 
    (∀ (Speed_1 Speed_2 Average_Speed Time_1 : ℝ) 
      (Total_Distance Distance_1 Distance_2 : ℝ) 
      (Time_Addl : ℝ),
     Speed_1 = 70 →
     Time_1 = 4 →
     Speed_2 = 60 →
     Average_Speed = 65 →
     Distance_1 = Speed_1 * Time_1 →
     Distance_2 = Speed_2 * Time_Addl →
     Total_Distance = Distance_1 + Distance_2 →
     Total_Distance = Average_Speed * T →
     T = Time_1 + Time_Addl →
     Time_Addl = 4 → T = 8) :=
begin
  -- Formal proof would go here
  sorry
end

end car_trip_duration_l591_591680


namespace small_branches_per_branch_l591_591336

theorem small_branches_per_branch (x : ℕ) (h1 : 1 + x + x^2 = 57) : x = 7 :=
by {
  sorry
}

end small_branches_per_branch_l591_591336


namespace largest_base5_to_base7_l591_591924

-- Define the largest four-digit number in base-5
def largest_base5_four_digit_number : ℕ := 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

-- Convert this number to base-7
def convert_to_base7 (n : ℕ) : ℕ := 
  let d3 := n / (7^3)
  let r3 := n % (7^3)
  let d2 := r3 / (7^2)
  let r2 := r3 % (7^2)
  let d1 := r2 / (7^1)
  let r1 := r2 % (7^1)
  let d0 := r1
  (d3 * 10^3) + (d2 * 10^2) + (d1 * 10^1) + d0

-- Theorem to prove m in base-7
theorem largest_base5_to_base7 : 
  convert_to_base7 largest_base5_four_digit_number = 1551 :=
by 
  -- skip the proof
  sorry

end largest_base5_to_base7_l591_591924


namespace function_shifted_right_l591_591050

def shifted_function (x : ℝ) : ℝ := 2 * sin (2 * (x - π / 4) + π / 6)

theorem function_shifted_right (x : ℝ) :
  shifted_function x = 2 * sin (2 * x - π / 3) :=
sorry

end function_shifted_right_l591_591050


namespace a_equals_2t_squared_l591_591781

theorem a_equals_2t_squared {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + 4 * a = b^2) :
  ∃ t : ℕ, 0 < t ∧ a = 2 * t^2 :=
sorry

end a_equals_2t_squared_l591_591781


namespace solve_equation_1_solve_equation_2_solve_equation_3_l591_591606

theorem solve_equation_1 (x : ℝ) : (x^2 - 3 * x = 0) ↔ (x = 0 ∨ x = 3) := sorry

theorem solve_equation_2 (x : ℝ) : (4 * x^2 - x - 5 = 0) ↔ (x = 5/4 ∨ x = -1) := sorry

theorem solve_equation_3 (x : ℝ) : (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2/3) := sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l591_591606


namespace infinite_nat_with_int_means_l591_591238

theorem infinite_nat_with_int_means :
  ∃ (infinitely_many : ℕ → Prop),
  (∀ (n : ℕ), infinitely_many n → 
    (∃ (p : ℕ), prime p ∧ p ≡ 1 [MOD 3] ∧ n = p^2 ∧ 
      ∃ (am gm : ℚ), am = (1 + p + p^2) / 3 ∧ gm = p ∧ am ∈ ℤ ∧ gm ∈ ℤ))
  ∧ (∃ (n : ℕ), infinitely_many n) :=
sorry

end infinite_nat_with_int_means_l591_591238


namespace same_response_l591_591232

structure Person :=
  (is_truth_teller : Bool)
  (response : String)

axiom truth_teller : Person → Prop
axiom liar : Person → Prop
axiom always_truthful_response : ∀ p: Person, truth_teller p → p.response = "I am the truth-teller"
axiom always_lying_response : ∀ p: Person, liar p → p.response = "I am the truth-teller"

theorem same_response (p1 p2 : Person) (q : String) :
  (truth_teller p1 ∨ liar p1) ∧ (truth_teller p2 ∨ liar p2) →
  (always_truthful_response p1 ∨ always_lying_response p1) ∧ (always_truthful_response p2 ∨ always_lying_response p2) →
  p1.response = p2.response :=
by
  intros hpersons hresponses
  cases hpersons with hp1 hp2
  cases hresponses with hr1 hr2
  cases hr1; cases hr2;
  exact rfl

end same_response_l591_591232


namespace range_of_a_l591_591224

variable (a x : ℝ)

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def M (a : ℝ) : Set ℝ := if a = 2 then {2} else {x | 2 ≤ x ∧ x ≤ a}

theorem range_of_a (a : ℝ) (p : x ∈ M a) (h : a ≥ 2) (hpq : Set.Subset (M a) A) : 2 ≤ a ∧ a ≤ 4 :=
  sorry

end range_of_a_l591_591224


namespace intersection_complement_eq_l591_591838

open Set

variable (U A B : Set ℕ)
  
theorem intersection_complement_eq : 
  U = {0, 1, 2, 3, 4} → 
  A = {0, 1, 3} → 
  B = {2, 3} → 
  A ∩ (U \ B) = {0, 1} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end intersection_complement_eq_l591_591838


namespace inequality_solution_l591_591440

-- Definitions
variables {a b : ℝ}

-- Hypothesis
variable (h : a > b)

-- Theorem
theorem inequality_solution : -2 * a < -2 * b :=
sorry

end inequality_solution_l591_591440


namespace solve_eq1_solve_eq2_l591_591250

theorem solve_eq1 : ∀ x : ℝ, x^2 - 4 * x = 0 ↔ x = 0 ∨ x = 4 := 
by {
  intro x,
  sorry -- Proof to be completed
}

theorem solve_eq2 : ∀ x : ℝ, x^2 = -2 * x + 3 ↔ x = -3 ∨ x = 1 := 
by {
  intro x,
  sorry -- Proof to be completed
}

end solve_eq1_solve_eq2_l591_591250


namespace domain_transformation_l591_591142

theorem domain_transformation (f : ℝ → ℝ) :
  (∀ x, -2 < x ∧ x < 0 → ∃ y, y = x+1) →
  (∀ x, 0 < x ∧ x < 1 → ∃ y, y = 2x-1) :=
by
  intros h1 h2
  sorry

end domain_transformation_l591_591142


namespace number_of_terms_in_sequence_l591_591847

noncomputable def sequence_term (n : ℕ) : ℝ :=
  2.6 + (n - 1) * 4.5

theorem number_of_terms_in_sequence :
  ∃ n : ℕ, sequence_term n = 52.1 ∧ n = 12 :=
by
  use 12
  split
  · unfold sequence_term
    norm_num
  · rfl

end number_of_terms_in_sequence_l591_591847


namespace sales_tax_difference_l591_591366

theorem sales_tax_difference (rate1 rate2 : ℝ) (price : ℝ) (h1 : rate1 = 0.075) (h2 : rate2 = 0.0625) (hprice : price = 50) : 
  rate1 * price - rate2 * price = 0.625 :=
by
  sorry

end sales_tax_difference_l591_591366


namespace remainder_polynomial_div_l591_591097

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 2 * x^3 + x + 5

-- State the theorem using the Remainder Theorem
theorem remainder_polynomial_div (a : ℝ) (h : a = 2) : 
  (p a) = 7 :=
by
  -- using the fact we're given a = 2 directly in the theorem
  rw h
  -- evaluation of polynomial is straightforward but omitted in this statement
  sorry

end remainder_polynomial_div_l591_591097


namespace eval_fraction_expr_l591_591085

theorem eval_fraction_expr :
  (2 ^ 2010 * 3 ^ 2012) / (6 ^ 2011) = 3 / 2 := 
sorry

end eval_fraction_expr_l591_591085


namespace compute_fraction_when_x_is_7_l591_591067

theorem compute_fraction_when_x_is_7 (x : ℤ) (h : x = 7) : (x^6 - 25 * x^3 + 144) / (x^3 - 12) = 312 / 331 :=
by
  rw h
  norm_num
  sorry

end compute_fraction_when_x_is_7_l591_591067


namespace complex_fraction_l591_591981

open Complex

/-- The given complex fraction \(\frac{5 - i}{1 - i}\) evaluates to \(3 + 2i\). -/
theorem complex_fraction : (⟨5, -1⟩ : ℂ) / (⟨1, -1⟩ : ℂ) = ⟨3, 2⟩ :=
  by
  sorry

end complex_fraction_l591_591981


namespace painted_prisms_l591_591728

theorem painted_prisms (n : ℕ) (h : n > 2) :
  2 * ((n - 2) * (n - 1) + (n - 2) * n + (n - 1) * n) = (n - 2) * (n - 1) * n ↔ n = 7 :=
by sorry

end painted_prisms_l591_591728


namespace intersection_A_B_equals_01_l591_591808

open Set

noncomputable theory

def A : Set ℝ := {x | 2 * x - 1 < 1}
def B : Set ℝ := {x | x = real.sqrt (-x)}

theorem intersection_A_B_equals_01 :
  (A ∩ B) = Ico 0 1 :=
by
  sorry

end intersection_A_B_equals_01_l591_591808


namespace perpendicular_slope_solution_l591_591804

theorem perpendicular_slope_solution (a : ℝ) :
  (∀ x y : ℝ, ax + (3 - a) * y + 1 = 0) →
  (∀ x y : ℝ, x - 2 * y = 0) →
  (l1_perp_l2 : ∀ x y : ℝ, ax + (3 - a) * y + 1 = 0 → x - 2 * y = 0 → False) →
  a = 2 :=
sorry

end perpendicular_slope_solution_l591_591804


namespace inequality_order_l591_591431

theorem inequality_order (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) : 
  b > (a^4 - b^4) / (a - b) ∧ (a^4 - b^4) / (a - b) > (a + b) / 2 ∧ (a + b) / 2 > 2 * a * b :=
by 
  sorry

end inequality_order_l591_591431


namespace find_f_neg3_l591_591139

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f(-x) = f(x)
axiom h2 : ∀ x : ℝ, x > 0 → f(2 + x) = -2 * f(2 - x)
axiom h3 : f(-1) = 4

theorem find_f_neg3 : f(-3) = -8 :=
by
  sorry

end find_f_neg3_l591_591139


namespace shortest_path_between_two_points_l591_591241

def shanxi_mountainous (s : Type) [MetricSpace s] (P : s → Prop := λ x, True) : Prop :=
  (∀ x y : s, P x ∧ P y → dist x y = dist (x, y)) ∧
  (∀ x y : s, P x ∧ P y → dist x y ≠ dist (x, y))

theorem shortest_path_between_two_points (s : Type) [MetricSpace s] {x y : s}
  (h : shanxi_mountainous s) : dist x y = min {d | ∃ (h: continuous), h x = y ∧ h y = h x} :=
sorry

end shortest_path_between_two_points_l591_591241


namespace locus_of_C_equation_l591_591796

open Real

noncomputable def locus_of_C (a : ℝ) (x y : ℝ) : Prop :=
  (1 - a) * x^2 - 2 * a * x + (1 + a) * y^2 = 0

theorem locus_of_C_equation (a : ℝ) (h : a > 0) : 
  ∀ (x y : ℝ), 
  (∃ B : ℝ × ℝ, B.1 = -1 ∧ 
    ∃ C : ℝ × ℝ, C.1 = x ∧ C.2 = y ∧
    C lies_on_angle_bisector a B) 
  → locus_of_C a x y :=
by
  intro x y
  intro h_conditions
  -- Here we would fill in the proof eventually
  sorry

lemma lies_on_angle_bisector (a : ℝ) (B : ℝ × ℝ) : Prop := sorry

end locus_of_C_equation_l591_591796


namespace perimeter_of_ABFCDE_eq_64_plus_32_sqrt_2_l591_591758

theorem perimeter_of_ABFCDE_eq_64_plus_32_sqrt_2:
  let side_length : ℝ := 16 in
  let hypotenuse_length : ℝ := side_length * Real.sqrt 2 in
  perimeter_of_ABFCDE side_length hypotenuse_length = 64 + 32 * Real.sqrt 2 := 
  sorry

end perimeter_of_ABFCDE_eq_64_plus_32_sqrt_2_l591_591758


namespace hyperbola_vertex_distance_l591_591416

theorem hyperbola_vertex_distance :
  (∀ x y : ℝ, (x^2 / 64 - y^2 / 49) = 1 → 16) :=
begin
  -- definitions based on conditions
  let a : ℝ := real.sqrt 64,
  have h : a = 8, from real.sqrt_eq_iff_sq_eq.2 (or.inl rfl),
  rw h,
  
  -- statement to be proven
  exact 16
end

end hyperbola_vertex_distance_l591_591416


namespace α_perp_β_l591_591858

variable {l m n : Line}
variable {α β : Plane}

-- Given Conditions
axiom l_perp_α : l ⊥ α
axiom l_parallel_β : l ∥ β

-- To Prove
theorem α_perp_β : α ⊥ β := 
sorry

end α_perp_β_l591_591858


namespace integral_sign_negative_l591_591449

open Topology

-- Define the problem
theorem integral_sign_negative {a b : ℝ} (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) (h_lt : ∀ x ∈ Set.Icc a b, f x < 0) (h_ab : a < b) :
  ∫ x in a..b, f x < 0 := 
sorry

end integral_sign_negative_l591_591449


namespace prove_f_pi_over_three_l591_591148

-- Definitions based on conditions a) 
def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ :=
  sin (ω * x + ϕ)

-- Conditions and assumptions: ω > 0, -π/2 < ϕ < π/2
variables (ω : ℝ) (ϕ : ℝ) (hω : ω > 0) (hϕ1 : -π/2 < ϕ) (hϕ2 : ϕ < π/2)

-- The question: Proving f(π/3) = √3/2
theorem prove_f_pi_over_three :
  (f (1/2) (π/6) (π/3) = sqrt 3 / 2) :=
by
  sorry

end prove_f_pi_over_three_l591_591148


namespace KN_plus_LM_ge_AC_l591_591903

open Classical

variables {A B C K L M N : Type*}
variables [has_dist A B C K L M N : Real] -- assuming these points lie on a real line
variables (AK BL CN BM : ℝ) -- the lengths of the segments

-- Given conditions
def points_on_AB (A K B P : ℝ) := AK = BL
def points_on_BC (B C M N : ℝ) := CN = BM

-- The hypothesized conclusion
theorem KN_plus_LM_ge_AC {A B C K L M N : Type*} [has_dist A B C K L M N : Real] :
  points_on_AB A K B P →
  points_on_BC B C M N →
  KN + LM ≥ AC :=
begin
  -- Skipping the proof
  sorry,
end

end KN_plus_LM_ge_AC_l591_591903


namespace units_digit_p_plus_2_l591_591312

theorem units_digit_p_plus_2 (p : ℕ) (hp₀ : p > 0) (hp₁ : p % 2 = 0) (hp₂ : ∃ (u : ℕ), u ∈ {2, 4, 6, 8} ∧ p % 10 = u) (hp₃ : (p^3 % 10) - (p^2 % 10) = 0) : 
  (p + 2) % 10 = 8 :=
sorry

end units_digit_p_plus_2_l591_591312


namespace chickens_and_rabbits_l591_591012

theorem chickens_and_rabbits (c r : ℕ) 
    (h1 : c = 2 * r - 5)
    (h2 : 2 * c + r = 92) : ∃ c r : ℕ, (c = 2 * r - 5) ∧ (2 * c + r = 92) := 
by 
    -- proof steps
    sorry

end chickens_and_rabbits_l591_591012


namespace calculate_students_l591_591938

noncomputable def handshakes (m n : ℕ) : ℕ :=
  1/2 * (4 * 3 + 5 * (2 * (m - 2) + 2 * (n - 2)) + 8 * (m - 2) * (n - 2))

theorem calculate_students (m n : ℕ) (h_m : 3 ≤ m) (h_n : 3 ≤ n) (h_handshakes : handshakes m n = 1020) : m * n = 140 :=
by
  sorry

end calculate_students_l591_591938


namespace find_the_number_l591_591335

theorem find_the_number :
  ∃ N : ℝ, ((4/5 : ℝ) * 25 = 20) ∧ (0.40 * N = 24) ∧ (N = 60) :=
by
  sorry

end find_the_number_l591_591335


namespace range_of_m_l591_591627

theorem range_of_m (α : ℝ) (m : ℝ) (h : (α > π ∧ α < 3 * π / 2) ∨ (α > 3 * π / 2 ∧ α < 2 * π)) :
  -1 < (Real.sin α) ∧ (Real.sin α) < 0 ∧ (Real.sin α) = (2 * m - 3) / (4 - m) → 
  m ∈ Set.Ioo (-1 : ℝ) (3 / 2 : ℝ) :=
by
  sorry

end range_of_m_l591_591627


namespace circle_center_radius_sum_l591_591922

-- Defining the circle equation as given
def circle_eq (x y : ℝ) := x^2 - 8 * y - 5 = -y^2 - 6 * x

-- Defining the center of the circle
def center_a := -3
def center_b := 4

-- Defining the radius of the circle
def radius_r := Real.sqrt 30

-- Defining the expression a + b + r
def expression := center_a + center_b + radius_r

-- The theorem that we need to prove
theorem circle_center_radius_sum : 
  (∃ x y : ℝ, circle_eq x y) → expression = 1 + Real.sqrt 30 := 
by 
  sorry

end circle_center_radius_sum_l591_591922


namespace log8_a1012_l591_591196

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 4 * x - 1
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 4

theorem log8_a1012 (a : ℕ → ℝ) (h₁ : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h₂ : ∃ a₁ a2023 : ℝ, f a₁ = 0 ∧ f a2023 = 0 ∧ a 1 = a₁ ∧ a 2023 = a2023) :
  log 8 (a 1012) = 1 / 3 := by
  sorry

end log8_a1012_l591_591196


namespace height_of_cylinder_l591_591498

variable (r h : ℝ)

-- Conditions given in the problem
def lateral_area_cylinder := 2 * π * r * h = 12 * π
def volume_cylinder := π * r^2 * h = 12 * π

-- The proof problem
theorem height_of_cylinder (h_value : ℝ)
  (lateral_area : lateral_area_cylinder r h)
  (volume : volume_cylinder r h) :
  h = 3 := by
  sorry

end height_of_cylinder_l591_591498


namespace geometric_sequence_iff_q_neg_one_l591_591837

theorem geometric_sequence_iff_q_neg_one {p q : ℝ} (h1 : p ≠ 0) (h2 : p ≠ 1)
  (S : ℕ → ℝ) (hS : ∀ n, S n = p^n + q) :
  (∃ (a : ℕ → ℝ), (∀ n, a (n+1) = (p - 1) * p^n) ∧ (∀ n, a (n+1) = S (n+1) - S n) ∧
                    (∀ n, a (n+1) / a n = p)) ↔ q = -1 :=
sorry

end geometric_sequence_iff_q_neg_one_l591_591837


namespace points_blue_zone_correct_l591_591347

-- Define the necessary parameters for the problem
def radius_bullseye : ℝ := 1
def points_bullseye : ℝ := 315
def area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the areas of the bullseye and the blue (fourth) zone
def area_bullseye : ℝ := area radius_bullseye
def area_blue_zone : ℝ := area 5 - area 4

-- Define the ratio of probabilities, based on areas
def ratio_prob_blue_bullseye : ℝ := area_blue_zone / area_bullseye

-- Define the inverse proportional relationship to find points for blue zone
def points_blue_zone : ℝ := points_bullseye / ratio_prob_blue_bullseye

-- The theorem we need to prove
theorem points_blue_zone_correct : points_blue_zone = 35 := by
  -- According to given and derived information
  -- We will insert 'sorry' as the actual steps are not required
  sorry

end points_blue_zone_correct_l591_591347


namespace product_of_possible_third_sides_l591_591650

theorem product_of_possible_third_sides (a b : ℝ) (h : a = 3 ∧ b = 6) : 
  let c1 := real.sqrt (a^2 + b^2) 
  let c2 := real.sqrt (b^2 - a^2)
  c1 * c2 ≈ 34.8 :=
by
  let c1 := real.sqrt (a^2 + b^2)
  let c2 := real.sqrt (b^2 - a^2)
  sorry

end product_of_possible_third_sides_l591_591650


namespace ellipse_properties_l591_591802

noncomputable def ellipse_eccentricity_eq (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (Real.sqrt (1 - b^2 / a^2) = Real.sqrt(2) / 2)

noncomputable def ellipse_minor_axis_eq (b : ℝ) : Prop :=
  2 * b = 2

noncomputable def ellipse_eq (x y : ℝ) (a : ℝ) : Prop :=
  x^2 / a^2 + y^2 = 1

noncomputable def line_eq (x y k : ℝ) : Prop :=
  k * x - y - k = 0

noncomputable def midpoint_eq (a b x1 y1 x2 y2 : ℝ) : Prop :=
  (a + b) / 2 = (x1 + x2) / 2 ∧ (a - b) / 2 = (y1 + y2) / 2

noncomputable def perpendicular_bisector_eq (x y k : ℝ) : Prop :=
  y + k / (1 + 2 * k^2) = -1 / k * (x - 2 * k^2 / (1 + 2 * k^2))

theorem ellipse_properties (a b : ℝ) : ellipse_eccentricity_eq a b ∧ ellipse_minor_axis_eq b →
  (ellipse_eq 0 1 2) ∧ ∃ k : ℝ, (line_eq (1/3) 0 k) ∧ (k = 1 ∨ k = -1) :=
by
  sorry

end ellipse_properties_l591_591802


namespace problem_divisible_by_8_l591_591213

theorem problem_divisible_by_8 :
  (5^2001 + 7^2002 + 9^21103 + 11^2004) % 8 = 0 := 
begin
  sorry
end

end problem_divisible_by_8_l591_591213


namespace determine_fake_coin_weight_l591_591654

theorem determine_fake_coin_weight
  (coins : Fin 25 → ℤ) 
  (fake_coin : Fin 25) 
  (all_same_weight : ∀ (i j : Fin 25), i ≠ fake_coin → j ≠ fake_coin → coins i = coins j)
  (fake_diff_weight : ∃ (x : Fin 25), (coins x ≠ coins fake_coin)) :
  ∃ (is_heavy : Bool), 
    (is_heavy = true ↔ coins fake_coin > coins (Fin.ofNat 0)) ∨ 
    (is_heavy = false ↔ coins fake_coin < coins (Fin.ofNat 0)) :=
  sorry

end determine_fake_coin_weight_l591_591654


namespace k6_monochromatic_triangle_k5_no_monochromatic_triangle_l591_591189

-- Definition for a complete graph with 6 vertices and 2-colored edges
def complete_graph_6 (V : Type) := { E : V × V → Prop // ∀ v1 v2, v1 ≠ v2 → E (v1, v2) }

-- Definition of a monochromatic triangle
def monochromatic_triangle (V : Type) (E : V × V → Prop) (color : Prop) :=
  ∃ v1 v2 v3, v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ (E (v1, v2) = color) ∧ (E (v2, v3) = color) ∧ (E (v1, v3) = color)

-- Definition for a 2-colored complete graph k_6
def two_colored_complete_graph_6 (V : Type) := 
  complete_graph_6 V × (∀ (v1 v2 : V), v1 ≠ v2 → E v1 v2 = true ∨ E v1 v2 = false)

-- Theorem statement: In any 2-colored complete graph k_6, there exists a monochromatic triangle
theorem k6_monochromatic_triangle (V : Type) (G : two_colored_complete_graph_6 V) :
  ∃ color, monochromatic_triangle V G.fst color :=
sorry

-- Theorem statement: There exists a 2-colored complete graph k_5 without a monochromatic triangle
theorem k5_no_monochromatic_triangle (V : Type) :
  ∃ (G : complete_graph V), (is_two_colored V G) ∧ ¬ (∃ color, monochromatic_triangle V G color) :=
sorry

end k6_monochromatic_triangle_k5_no_monochromatic_triangle_l591_591189


namespace probability_first_greater_than_second_l591_591107

open ProbabilityTheory

/-- From cards numbered 1, 2, 3, 4, and 5, one card is drawn, replaced, and another card is drawn.
    The probability that the number on the first card drawn is greater than the number on the second
    card drawn is 2/5. -/
theorem probability_first_greater_than_second :
  let E := {1, 2, 3, 4, 5}
  in
  ∃ p : ℚ,
    (p = 2 / 5) ∧
    (∃ (draw : E × E),
      probability (
        draw.fst > draw.snd
      ) = p) :=
by
  sorry

end probability_first_greater_than_second_l591_591107


namespace percent_of_liquidX_in_solutionB_l591_591558

theorem percent_of_liquidX_in_solutionB (P : ℝ) (h₁ : 0.8 / 100 = 0.008) 
(h₂ : 1.5 / 100 = 0.015) 
(h₃ : 300 * 0.008 = 2.4) 
(h₄ : 1000 * 0.015 = 15) 
(h₅ : 15 - 2.4 = 12.6) 
(h₆ : 12.6 / 700 = P) : 
P * 100 = 1.8 :=
by sorry

end percent_of_liquidX_in_solutionB_l591_591558


namespace truck_tank_percentage_bigger_than_minivan_tank_l591_591507

theorem truck_tank_percentage_bigger_than_minivan_tank :
  ∃ (service_cost_per_vehicle fuel_cost_per_liter total_cost mv_tank_capacity : ℝ) 
    (number_of_minivans number_of_trucks : ℕ) 
    (percentage_increase : ℝ),
  service_cost_per_vehicle = 2.30 ∧
  fuel_cost_per_liter = 0.70 ∧
  total_cost = 396 ∧
  mv_tank_capacity = 65 ∧
  number_of_minivans = 4 ∧
  number_of_trucks = 2 ∧
  percentage_increase = 120 ∧
  let
    total_service_cost := (number_of_minivans + number_of_trucks) * service_cost_per_vehicle,
    total_fuel_cost := total_cost - total_service_cost,
    total_amount_of_fuel := total_fuel_cost / fuel_cost_per_liter,
    fuel_for_minivans := number_of_minivans * mv_tank_capacity,
    fuel_for_trucks := total_amount_of_fuel - fuel_for_minivans,
    truck_tank_capacity := fuel_for_trucks / number_of_trucks
  in
  percentage_increase = ((truck_tank_capacity - mv_tank_capacity) / mv_tank_capacity) * 100 := sorry

end truck_tank_percentage_bigger_than_minivan_tank_l591_591507


namespace Beth_age_proof_l591_591058

theorem Beth_age_proof (B : ℕ) (h1 : 5 + 8 = 13) 
                       (h2 : B + 8 = 2 * (5 + 8)) : 
  B = 18 := 
by 
  have eq : B + 8 = 26 := by rwa h1 at h2
  linarith

end Beth_age_proof_l591_591058


namespace find_a_find_inverse_function_l591_591547

-- Define the function property and conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Define the odd function condition
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the given function
def given_function (a : ℝ) : (ℝ → ℝ) := λ x, (a - 1) * (2^x + 1)

-- State the theorem for part (1)
theorem find_a (h : is_odd_function (given_function a)) : a = 1 :=
sorry

-- Define the inverse function
def inverse_function : (ℝ → ℝ) := λ y, logBase 2 (y - 1)

-- State the theorem for part (2)
theorem find_inverse_function (h_a : a = 1) : ∀ x, 0 < x ∧ x < 2 → (given_function a) (inverse_function x) = x :=
sorry

end find_a_find_inverse_function_l591_591547


namespace total_workers_is_28_l591_591188

noncomputable def avg_salary_total : ℝ := 750
noncomputable def num_type_a : ℕ := 5
noncomputable def avg_salary_type_a : ℝ := 900
noncomputable def num_type_b : ℕ := 4
noncomputable def avg_salary_type_b : ℝ := 800
noncomputable def avg_salary_type_c : ℝ := 700

theorem total_workers_is_28 :
  ∃ (W : ℕ) (C : ℕ),
  W * avg_salary_total = num_type_a * avg_salary_type_a + num_type_b * avg_salary_type_b + C * avg_salary_type_c ∧
  W = num_type_a + num_type_b + C ∧
  W = 28 :=
by
  sorry

end total_workers_is_28_l591_591188


namespace hyperbola_vertex_distance_l591_591414

theorem hyperbola_vertex_distance :
  (∀ x y : ℝ, (x^2 / 64 - y^2 / 49) = 1 → 16) :=
begin
  -- definitions based on conditions
  let a : ℝ := real.sqrt 64,
  have h : a = 8, from real.sqrt_eq_iff_sq_eq.2 (or.inl rfl),
  rw h,
  
  -- statement to be proven
  exact 16
end

end hyperbola_vertex_distance_l591_591414


namespace line_intersection_l591_591073

-- Definitions for the parametric lines
def line1 (t : ℝ) : ℝ × ℝ := (3 + t, 2 * t)
def line2 (u : ℝ) : ℝ × ℝ := (-1 + 3 * u, 4 - u)

-- Statement that expresses the intersection point condition
theorem line_intersection :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (30 / 7, 18 / 7) :=
by
  sorry

end line_intersection_l591_591073


namespace monotonic_decreasing_interval_l591_591623

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 1) → f x < f (x + 1) := 
by 
  -- sorry is used because the actual proof is not required
  sorry

end monotonic_decreasing_interval_l591_591623


namespace monotonicity_g_range_of_a_inequality_f_l591_591150

open Real

-- Define the logarithmic function
def f (x : ℝ) : ℝ := ln x

-- 1. Monotonicity of g(x)
theorem monotonicity_g (a : ℝ) :
  (∀ x > 0, deriv (λ (x : ℝ), a * ln x - 1 / x) x ≥ 0) 
  ↔ a ≥ 0 :=
by sorry

-- 2. Range of the real number a
theorem range_of_a (a : ℝ) :
  (∀ x > 0, ln x ≤ ax ∧ ax ≤ exp x) 
  ↔ 1 / exp 1 ≤ a ∧ a ≤ exp 1 :=
by sorry

-- 3. Inequality involving f(x)
theorem inequality_f (x₁ x₂ : ℝ) (h : x₁ > x₂) (h₀ : x₂ > 0) :
  ((f x₁ - f x₂) / (x₁ - x₂) > (2 * x₂) / (x₁ ^ 2 + x₂ ^ 2)) :=
by sorry

end monotonicity_g_range_of_a_inequality_f_l591_591150


namespace arithmetic_sequence_ineq_l591_591219

variable {α : Type*} [OrderedRing α]
  
-- Define an arithmetic sequence with a common difference d < 0
def arithmetic_sequence (a₁ d : α) : ℕ → α 
| 0       := a₁
| (n + 1) := a₁ + n * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a₁ d : α) (n : ℕ) : α :=
n • a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_ineq {a₁ d : α} (h_d : d < 0) (n : ℕ) :
  let seq := arithmetic_sequence a₁ d in
  let sum_seq := sum_arithmetic_sequence a₁ d n in
  n * seq n ≤ sum_seq ∧ sum_seq ≤ n * a₁ :=
by
suffices h₁ : n * seq n ≤ sum_seq
suffices h₂ : sum_seq ≤ n * a₁
exact ⟨h₁, h₂⟩
sorry

end arithmetic_sequence_ineq_l591_591219


namespace train_speed_l591_591678

theorem train_speed (distance time : ℝ) (h₁ : distance = 240) (h₂ : time = 4) : 
  ((distance / time) * 3.6) = 216 := 
by 
  rw [h₁, h₂] 
  sorry

end train_speed_l591_591678


namespace monotonic_decreasing_interval_l591_591622

variable (x : ℝ)

def f (x : ℝ) : ℝ := x - log x 

noncomputable def f' (x : ℝ) := deriv f x

theorem monotonic_decreasing_interval : ( {x : ℝ | 0 < x ∧ x < 1} = { x | x ∈ Ioo 0 1}) ↔ ( ∀ x, f' x < 0 ) :=
sorry

end monotonic_decreasing_interval_l591_591622


namespace closest_area_shaded_l591_591013

-- define the dimensions of the rectangle
def width := 3
def height := 4

-- define the diameter and radius of the circle
def diameter := 2
def radius := diameter / 2

-- area of the rectangle
def area_rectangle := width * height

-- area of the circle using pi
noncomputable def area_circle := Real.pi * radius * radius

-- area of the shaded region
noncomputable def area_shaded := area_rectangle - area_circle

-- the closest whole number to the area of the shaded region
theorem closest_area_shaded : 9 = Int.round area_shaded :=
by
  sorry

end closest_area_shaded_l591_591013


namespace quadratic_relationship_l591_591389

theorem quadratic_relationship :
  ∀ x y, ((x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 6) ∨ (x = 3 ∧ y = 12) ∨ (x = 4 ∧ y = 20) ∨ (x = 5 ∧ y = 30)) → 
         y = x^2 + x :=
by
  intros x y h,
  cases h,
  { rw [h.1, h.2], sorry },
  { cases h,
    { rw [h.1, h.2], sorry },
    { cases h,
      { rw [h.1, h.2], sorry },
      { cases h,
        { rw [h.1, h.2], sorry },
        { rw [h.1, h.2], sorry } } } }

end quadratic_relationship_l591_591389


namespace product_bounds_l591_591071

theorem product_bounds (A : ℝ) (h : A = ∏ k in finset.range 200, (2*k - 1) / (2*k)) :
  1/29 < A ∧ A < 1/20 := by
sorry

end product_bounds_l591_591071


namespace deer_leap_distance_correct_l591_591348

-- Definitions based on the conditions of the problem.
def tiger_leap_behind : Nat := 50
def tiger_leaps_per_minute : Nat := 5
def deer_leaps_per_minute : Nat := 4
def tiger_distance_per_leap : Nat := 8
def tiger_total_distance : Nat := 800

-- Definition for the deer's distance per leap that needs to be proved.
def deer_distance_per_leap : Nat := 5

-- Lean 4 theorem statement.
theorem deer_leap_distance_correct :
  (tiger_leap_behind * tiger_distance_per_leap + 
   (tiger_leaps_per_minute.toRational / deer_leaps_per_minute.toRational * (tiger_total_distance / tiger_distance_per_leap)) * (tiger_total_distance / tiger_distance_per_leap).toRational)
  / ((tiger_leaps_per_minute.toRational / deer_leaps_per_minute.toRational * (tiger_total_distance / tiger_distance_per_leap)).toRational) = deer_distance_per_leap := 
by
  sorry

end deer_leap_distance_correct_l591_591348


namespace monotonicity_f_geq_f_neg_l591_591459

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem monotonicity (a : ℝ) :
  (a ≤ 0 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2) ∧
  (a > 0 →
    (∀ x1 x2 : ℝ, x1 > Real.log a → x2 > Real.log a → x1 ≤ x2 → f a x1 ≤ f a x2) ∧
    (∀ x1 x2 : ℝ, x1 < Real.log a → x2 < Real.log a → x1 ≤ x2 → f a x1 ≤ f a x2)) :=
by sorry

theorem f_geq_f_neg (x : ℝ) (hx : x ≥ 0) : f 1 x ≥ f 1 (-x) :=
by sorry

end monotonicity_f_geq_f_neg_l591_591459


namespace basketball_scores_l591_591332

theorem basketball_scores :
  (∃ P : Finset ℕ, P = { P | ∃ x : ℕ, x ∈ (Finset.range 8) ∧ P = x + 14 } ∧ P.card = 8) :=
by
  sorry

end basketball_scores_l591_591332


namespace hyperbola_distance_between_vertices_l591_591413

theorem hyperbola_distance_between_vertices :
  ∀ (x y : ℝ), ((x^2) / 64 - (y^2) / 49 = 1) → (distance (8, 0) (-8, 0) = 16) :=
by
  sorry

end hyperbola_distance_between_vertices_l591_591413


namespace consignment_shop_total_items_l591_591971

variable (x y z t n : ℕ)

noncomputable def totalItems (n : ℕ) := n + n + n + 3 * n

theorem consignment_shop_total_items :
  ∃ (x y z t n : ℕ), 
    3 * n * y + n * x + n * z + n * t = 240 ∧
    t = 10 * n ∧
    z + x = y + t + 4 ∧
    x + y + 24 = t + z ∧
    y ≤ 6 ∧
    totalItems n = 18 :=
by
  sorry

end consignment_shop_total_items_l591_591971


namespace triangle_area_ratio_l591_591295

/-
Triangle MNP has sides of length 7, 24, and 25 units,
Triangle QRS has sides of length 9, 12, and 15 units.
Prove that the ratio of the area of triangle MNP to the area of triangle QRS is 14/9.
-/
theorem triangle_area_ratio :
  let MNP := (7 : ℝ, 24 : ℝ, 25 : ℝ) in
  let QRS := (9 : ℝ, 12 : ℝ, 15 : ℝ) in
  (MNP.1^2 + MNP.2^2 = MNP.3^2) →
  (QRS.1^2 + QRS.2^2 = QRS.3^2) →
  let area_MNP := MNP.1 * MNP.2 / 2 in
  let area_QRS := QRS.1 * QRS.2 / 2 in
  area_MNP / area_QRS = 14 / 9 :=
by
  intros MNP QRS MNP_is_right QRS_is_right area_MNP area_QRS
  sorry

end triangle_area_ratio_l591_591295


namespace magnitude_z_l591_591444

variable (z : ℂ) (i : ℂ)
-- The conditions in the problem
axiom ImaginaryUnit : i * i = -1
axiom Condition : (1 + i) * z = i

-- The statement to be proved
theorem magnitude_z : |z| = (Real.sqrt 2) / 2 := by
  sorry

end magnitude_z_l591_591444


namespace sector_area_l591_591969

theorem sector_area (θ : ℝ) (r : ℝ) (sin_one_pos : 0 < sin 1) (hchord : 2 * sin 1 = 2 * r * sin (θ / 2)) 
  (hθ : θ = 2) : 
  (1 / 2) * r^2 * θ = 1 :=
by
  -- conditions
  have hr : r = 1 :=
    by
      calc
        r = (2 * sin 1) / (2 * 1 * sin (1))  := sorry  -- Given \sin 1 = sin (π/3) -> r = 1
  -- proof
  calc
    (1 / 2) * r^2 * θ = _ := sorry


end sector_area_l591_591969


namespace cost_of_football_shoes_l591_591677

theorem cost_of_football_shoes (f s M N x : ℝ) 
    (hf : f = 3.75) 
    (hs : s = 2.40) 
    (hM : M = 10) 
    (hN : N = 8) 
    (hmanifestsumm : M + N = 18) 
    (httotal : f + s = 6.15) 
    (hx : x = 18 - 6.15) 
    : x = 11.85 :=
by
  rw [hx]
  norm_num
  sorry

end cost_of_football_shoes_l591_591677


namespace exist_initial_configuration_l591_591508

-- Define the conditions of the grid
structure Grid (m n : ℕ) where
  grid_state : Array (Array Bool)  -- Representation of the grid where Bool indicates alive (True) or dead (False)

-- Define the rule for the evolution of the grid based on neighboring cells
def evolve (g : Grid m n) : Grid m n :=
  sorry  -- define how the grid evolves here

-- Define the notion of life existing forever in a grid
def life_exists_forever (m n : ℕ) (initial_state : Grid m n) : Prop :=
  ∃ (g : Grid m n), ∀ t : ℕ, (evolve^t initial_state).grid_state ≠ Array.empty

-- The main theorem to prove
theorem exist_initial_configuration (m n : ℕ) :
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 3) ∨ (m = 3 ∧ n = 1) → ¬ life_exists_forever m n (Grid m n) :=
sorry

end exist_initial_configuration_l591_591508


namespace max_value_of_f_area_of_triangle_l591_591832

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sin x * cos x

theorem max_value_of_f : ∃ x : ℝ, f x = (sqrt 2) / 2 + 1 / 2 :=
by sorry

-- Given conditions for triangle
variables (A B C : ℝ) (AB AC : ℝ)
def is_triangle (A B C : ℝ) := A + B + C = π
def given_AB_AC : AB = 3 := rfl
def given_AC_AC : AC = 3 := rfl

-- Using f(x) condition
def f_condition (A : ℝ) : f (A / 2 + π / 8) = 1 := sorry

-- Area of triangle ABC
noncomputable def area_triangle (AB AC : ℝ) (A : ℝ) : ℝ := 1 / 2 * AB * AC * sin A

-- Final theorem statement
theorem area_of_triangle : ∃ S : ℝ, S = 9 * sqrt 2 / 4 :=
by
  let A := sorry -- Solve angle A with given f condition
  use area_triangle 3 3 A
  sorry

end max_value_of_f_area_of_triangle_l591_591832


namespace range_of_a_l591_591835

theorem range_of_a (a : ℝ)
  (p : ∀ x, ax^2 - 4x + a > 0)
  (q : ∀ x ∈ set.Iio (-1), 2x^2 + x > 2 + ax)
  (p_or_q : p ∨ q)
  (p_and_q_false : ¬ (p ∧ q)) :
  1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l591_591835


namespace max_volume_base_edge_length_l591_591635

open Real

def volume (x : ℝ) : ℝ := x^2 * (60 - x) / 2

-- Define the condition for domain
def domain (x : ℝ) : Prop := 0 < x ∧ x < 60

theorem max_volume_base_edge_length :
  ∀ x : ℝ, domain x → (∀ y, domain y → volume y ≤ volume 40) :=
by
  sorry

end max_volume_base_edge_length_l591_591635


namespace mutually_exclusive_not_complementary_l591_591042

-- Definitions of the events
def E1 : Prop := ¬ hit_target  -- Miss the target
def E2 : Prop := hit_target    -- Hit the target
def E3 : Prop := hit_target ∧ score > 4 -- Hit the target with score > 4 
def E4 : Prop := hit_target ∧ score ≥ 5 -- Hit the target with score ≥ 5 

-- Proof statement
theorem mutually_exclusive_not_complementary :
  (E1 ∧ E3 -> False) ∧
  (E1 ∧ E4 -> False) ∧
  ¬(E1 ↔ E3) ∧
  ¬(E1 ↔ E4) ∧
  ¬(E1 ↔ E2) ∧
  (E2 ∧ E3 -> False) ∧
  (E2 ∧ E4 -> False) ∧
  ¬(E2 ↔ E3) ∧
  ¬(E2 ↔ E4) ∧
  ¬(E3 ↔ E4) ∧ -- Possible more pairs
  (let mutually_exclusive_pairs := [(E1, E3), (E1, E4)] in 
  List.length mutually_exclusive_pairs = 2) :=
by
  sorry

end mutually_exclusive_not_complementary_l591_591042


namespace range_of_t_l591_591515

theorem range_of_t (A : ℝ × ℝ) (B : ℝ × ℝ) (D : ℝ × ℝ) (t : ℝ) 
  (hA : A = (0, 2)) (hB : B = (0, 1)) (hD : D = (t, 0)) (ht : t > 0) 
  (hM_cond : ∀ M : ℝ × ℝ, 
    (∃ λ : ℝ, (M = λ • (D.1 - A.1, D.2 - A.2) + (A.1, A.2)) ∧ 0 ≤ λ ∧ λ ≤ 1) → 
    dist M A ≤ 2 * dist M B) :
  t ≥ 2 * Real.sqrt 3 / 3 := 
begin
  sorry
end

end range_of_t_l591_591515


namespace cara_between_pairs_l591_591373

open Nat

-- Definition of the problem context as described in the conditions
def cara_neighbors : ℕ := 4
def pairs_of_people (n : ℕ) : ℕ := n.choose 2

-- The theorem stating the correct answer given the conditions
theorem cara_between_pairs : pairs_of_people cara_neighbors = 6 :=
by
  rw [cara_neighbors, pairs_of_people]
  exact Nat.choose_self (4 - 2) -- This computes 4 choose 2, which is 6
  sorry -- This can be completed with an actual proof step to verify.

end cara_between_pairs_l591_591373


namespace problem_statement_l591_591478

def a : ℝ × ℝ := (0, 2)
def b : ℝ × ℝ := (2, 2)

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem problem_statement : dot_product (vector_sub a b) a = 0 := 
by 
  -- The proof would go here
  sorry

end problem_statement_l591_591478


namespace circle_intercepts_chord_l591_591381

theorem circle_intercepts_chord (A B C O : Point) (d : ℝ) 
  (angle_ABC : angle A B C) (length_condition : d > 0) (acute_angle : isAcute angle_ABC)
  (O_on_AB : lies_on O (line_through A B)) (chord_intercept_condition : ⊥ / BC B O = OM) :
  ∃ (circle_radius : ℝ), circle_radius = sqrt (OM ^ 2 + (d / 2) ^ 2) ∧ d / 2 ≤ OM :=
by
  sorry

end circle_intercepts_chord_l591_591381


namespace B_days_to_complete_work_l591_591007

theorem B_days_to_complete_work 
  (W : ℝ) -- Define the amount of work
  (A_rate : ℝ := W / 15) -- A can complete the work in 15 days
  (B_days : ℝ) -- B can complete the work in B_days days
  (B_rate : ℝ := W / B_days) -- B's rate of work
  (total_days : ℝ := 12) -- Total days to complete the work
  (A_days_after_B_leaves : ℝ := 10) -- Days A works alone after B leaves
  (work_done_together : ℝ := 2 * (A_rate + B_rate)) -- Work done together in 2 days
  (work_done_by_A : ℝ := 10 * A_rate) -- Work done by A alone in 10 days
  (total_work_done : ℝ := work_done_together + work_done_by_A) -- Total work done
  (h_total_work_done : total_work_done = W) -- Total work equals W
  : B_days = 10 :=
sorry

end B_days_to_complete_work_l591_591007


namespace zero_point_of_f_l591_591636

-- Define the function f
def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

-- We need to prove that the zero point of the function f lies between 2 and 3,
-- and infer that the interval (k, k+1) translates to k = 2, where k is an integer.
theorem zero_point_of_f (x_0 : ℝ) (k : ℤ) (h₀ : f x_0 = 0) (h₁ : x_0 ∈ Set.Ioo k (k + 1)) : k = 2 := by
  sorry

end zero_point_of_f_l591_591636


namespace line_C_equation_curve_P_to_cartesian_distance_AB_solve_inequality_range_of_a_l591_591327

-- Define the line C with parametric equations
def line_C (t: ℝ) : ℝ × ℝ := (2 + t, t + 1)

-- Define the curve P in polar coordinates
def curve_P (ρ θ: ℝ) := ρ^2 - 4 * ρ * cos θ + 3 = 0

-- Define the rectangular coordinate equation for curve P
def curve_P_cartesian (x y: ℝ) := x^2 + y^2 - 4 * x + 3 = 0

-- Define the function f(x) = |2x + 3| + |x - 1|
def f (x: ℝ) := abs (2 * x + 3) + abs (x - 1)

-- Prove the line equation in Cartesian form
theorem line_C_equation (t: ℝ) :
    let (x, y) := line_C t in x - y - 1 = 0 :=
sorry

-- Prove the rectangular equation of curve P
theorem curve_P_to_cartesian (x y: ℝ) (h: curve_P (sqrt (x^2 + y^2)) (atan2 y x)) :
    curve_P_cartesian x y :=
sorry

-- Prove the distance between intersection points A and B is sqrt(2)
theorem distance_AB :
    let d := sqrt 2 / 2 in
    2 * sqrt (1 - d^2) = sqrt 2 :=
sorry

-- Prove the solution set to f(x) > 4
theorem solve_inequality : 
    {x : ℝ | f x > 4} = {x : ℝ | x < -2} ∪ {x | 0 < x} :=
sorry

-- Prove the range of a + 1 > f(x) for x ∈ [-3/2, 1]
theorem range_of_a :
    (∃ x ∈ Icc (-3/2) 1, ∀ a : ℝ, a > 3/2 → a + 1 > f x) ∧ ∃ x ∈ Icc (-3/2) 1, a + 1 > f x :=
sorry

end line_C_equation_curve_P_to_cartesian_distance_AB_solve_inequality_range_of_a_l591_591327


namespace necessary_but_not_sufficient_log_l591_591687

noncomputable def sufficient_but_not_necessary_condition_log (x : ℝ) : Prop :=
0 < x ∧ x < 1 → log 2 (exp (2 * x) - 1) < 2

theorem necessary_but_not_sufficient_log (x : ℝ) :
(0 < x ∧ x < 1) → log 2 (exp (2 * x) - 1) < 2 ∧
∃ y : ℝ, (log 2 (exp (2 * y) - 1) < 2) ∧ ¬(0 < y ∧ y < 1) :=
begin
  sorry,
end

end necessary_but_not_sufficient_log_l591_591687


namespace ellipse_focus_distance_l591_591819

theorem ellipse_focus_distance {b : ℝ} (h₀ : 0 < b ∧ b < 3)
  (h₁ : ∀ A B : ℝ, (| F₁ A B | + | F₂ A B |) = 12 - (| A F₁ | + | B F₁ |))
  (h₂ : max (| B F₂ | + | A F₂ |) = 10) : 
  b = real.sqrt 3 := 
sorry

end ellipse_focus_distance_l591_591819


namespace monthly_instalment_504_l591_591311

def monthly_installment 
  (C : ℝ) -- Cash price
  (d : ℝ) -- Deposit percentage
  (r : ℝ) -- Annual interest rate
  (n : ℕ) -- Number of months
  : ℝ :=
  let deposit := d * C
  let balance := C - deposit
  let t := n / 12
  let A := balance * (1 + r * t)
  A / n

theorem monthly_instalment_504 :
  monthly_installment 21000 0.10 0.12 60 = 504 :=
by
  sorry

end monthly_instalment_504_l591_591311


namespace slope_y_intercept_sum_l591_591518

noncomputable def point := (ℝ × ℝ)

def midpoint (p1 p2: point) : point := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def slope (p1 p2: point) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

def y_intercept (p: point) : ℝ := p.2  -- since we assume the point lies on the y-axis (x=0).

def A : point := (0, 10)
def B : point := (0, 2)
def C : point := (10, 0)
def D : point := midpoint A B

theorem slope_y_intercept_sum :
  let m := slope C D in
  let b := y_intercept D in
  m + b = 27 / 5 :=
by {
  let m := slope C D,
  let b := y_intercept D,
  have m_val : m = -3 / 5 := by sorry,
  have b_val : b = 6 := by sorry,
  show m + b = 27 / 5, from by {
    rw [m_val, b_val],
    norm_num,
  }
}

end slope_y_intercept_sum_l591_591518


namespace f_neg_six_eq_neg_five_l591_591555

def f (x : ℝ) : ℝ :=
if x < -5 then 2 * x + 7 else x ^ 2 - 1

theorem f_neg_six_eq_neg_five : f (-6) = -5 :=
by
-- proof to be filled in
sorry

end f_neg_six_eq_neg_five_l591_591555


namespace reflect_over_y_equals_x_l591_591267

noncomputable def reflected_function (f : ℝ → ℝ) (a b : ℝ) : (ℝ → ℝ) :=
  λ x, f⁻¹ (x - b) + a

theorem reflect_over_y_equals_x
  (f : ℝ → ℝ) (a b : ℝ) (hbij : Function.Bijective f) :
  ∀ x, reflected_function f a b x = (Function.LeftInverse.left_inv_function hbij).inv_fun (x - b) + a := by
  sorry

end reflect_over_y_equals_x_l591_591267


namespace four_digit_solution_l591_591418

-- Definitions for the conditions.
def condition1 (u z x : ℕ) : Prop := u + z - 4 * x = 1
def condition2 (u z y : ℕ) : Prop := u + 10 * z - 2 * y = 14

-- The theorem to prove that the four-digit number xyz is either 1014, 2218, or 1932
theorem four_digit_solution (x y z u : ℕ) (h1 : condition1 u z x) (h2 : condition2 u z y) :
  (x = 1 ∧ y = 0 ∧ z = 1 ∧ u = 4) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ u = 8) ∨
  (x = 1 ∧ y = 9 ∧ z = 3 ∧ u = 2) := 
sorry

end four_digit_solution_l591_591418


namespace mean_score_all_students_l591_591229

def mean_combined_score (M A : ℝ) (m a : ℕ) (hm : M = 75) (ha : A = 65) (hr : (m : ℝ)/(a : ℝ) = 2/3) : ℝ :=
  (75 * m + 65 * a) / (m + a)

theorem mean_score_all_students (M A : ℝ) (m a : ℕ) (hm : M = 75) (ha : A = 65) (hr : (m : ℝ)/(a : ℝ) = 2/3) :
  mean_combined_score M A m a hm ha hr = 69 := 
sorry

end mean_score_all_students_l591_591229


namespace each_parent_payment_l591_591706

def original_salary : ℝ := 45000
def raise_percentage : ℝ := 0.2
def num_kids : ℕ := 9

def raise_amount : ℝ := original_salary * raise_percentage
def new_salary : ℝ := original_salary + raise_amount
def payment_per_parent : ℝ := new_salary / num_kids

theorem each_parent_payment (h1: raise_amount = 9000) (h2: new_salary = 54000) (h3: payment_per_parent = 6000) : payment_per_parent = 6000 :=
by
  sorry

end each_parent_payment_l591_591706


namespace general_term_of_sequence_l591_591799

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 0.5 * a n + 0.5

theorem general_term_of_sequence :
  ∃ a : ℕ → ℝ, sequence a ∧ ∀ n, a n = 1 + (1 / 2) ^ (n - 1) :=
by
  sorry

end general_term_of_sequence_l591_591799


namespace train_lost_time_l591_591349

noncomputable def train_time_lost (speed_car : ℕ) (distance : ℕ) : ℕ :=
  let speed_train := speed_car * 3 / 2 in
  let time_car := (distance : ℚ) / speed_car * 60 in
  let time_train := (distance : ℚ) / speed_train * 60 in
  time_car - time_train

theorem train_lost_time :
  train_time_lost 120 75 = 12.5 := by
  sorry

end train_lost_time_l591_591349


namespace valid_word_count_is_correct_l591_591983

noncomputable def num_valid_words : ℕ :=
  let num_one_letter_words := 1 in
  let num_two_letter_words := 24^2 - 23^2 in
  let num_three_letter_words := 24^3 - 23^3 in
  let num_four_letter_words := 24^4 - 23^4 in
  let num_five_letter_words := 24^5 - 23^5 in
  num_one_letter_words + 
  num_two_letter_words + 
  num_three_letter_words + 
  num_four_letter_words + 
  num_five_letter_words 

theorem valid_word_count_is_correct: num_valid_words = 1580921 :=
by 
  sorry

end valid_word_count_is_correct_l591_591983


namespace line_perpendicular_to_plane_l591_591861

theorem line_perpendicular_to_plane {L : Type*} [ EuclideanSpace L ] {plane : set L} 
  (line : L) (h_1 : ∃ (s1 s2 : L), s1 ≠ s2 ∧ s1 ∈ plane ∧ s2 ∈ plane ∧ line ⊥ s1 ∧ line ⊥ s2) 
  (h_2_triangle : ∀ {A B C : L}, triangle A B C ⊆ plane → ∃ (s1 s2 : L), s1 ≠ s2 ∧ s1 ∈ {A, B, C} ∧ s2 ∈ {A, B, C} ∧ line ⊥ s1 ∧ line ⊥ s2)
  (h_3_circle : ∀ {O R1 R2 R3 R4 : L}, circle O R1 R2 ∧ circle O R3 R4 ⊆ plane → ∃ (d1 d2 : L), d1 ≠ d2 ∧ d1 ∈ {R1, R2, R3, R4} ∧ d2 ∈ {R1, R2, R3, R4} ∧ line ⊥ d1 ∧ line ⊥ d2) :
  ( ∀ (s1 s2 : L), s1 ≠ s2 ∧ line ⊥ s1 ∧ line ⊥ s2 → line ⊥ plane ) :=
sorry

end line_perpendicular_to_plane_l591_591861


namespace base6_addition_unique_solution_l591_591388

theorem base6_addition_unique_solution : 
  ∃ (triangle square : ℕ), triangle < 6 ∧ square < 6 ∧
  (43 * 6^2 + triangle * 6 + square) + (1 * 6^2 + triangle * 6 + 5) + (0 * 6^2 + square * 6 + 4) = 45 * 6^2 + square * 6 + 2 ∧
  triangle = 4 ∧ square = 4 :=
by
  sorry

end base6_addition_unique_solution_l591_591388


namespace part_a_part_b_l591_591424

-- Definitions of the conditions
def is_odd_prime (p : ℕ) : Prop :=
  (2 < p) ∧ (∀ n, n < p → n > 1 → p % n ≠ 0)

def d_p (p n : ℕ) : ℕ := n % p

def is_p_seq (p : ℕ) (a : ℕ → ℕ) : Prop :=
  (Nat.gcd a 0 p = 1) ∧ (∀ n, a (n + 1) - a n = d_p p (a n))

-- Part (a) statement
theorem part_a : ∃ (inf_primes_a : set ℕ), (∀ p ∈ inf_primes_a, is_odd_prime p) ∧ (∀ p ∈ inf_primes_a, 
  ∃ (a b : ℕ → ℕ), is_p_seq p a ∧ is_p_seq p b ∧ 
  (∀ n, ∃ (m : ℕ), a m > b m) ∧ (∀ n, ∃ (m : ℕ), b m > a m)) :=
sorry

-- Part (b) statement
theorem part_b : ∃ (inf_primes_b : set ℕ), (∀ p ∈ inf_primes_b, is_odd_prime p) ∧ (∀ p ∈ inf_primes_b, 
  ∃ (a b : ℕ → ℕ), is_p_seq p a ∧ is_p_seq p b ∧ 
  a 0 < b 0 ∧ (∀ n ≥ 1, a n > b n)) :=
sorry

end part_a_part_b_l591_591424


namespace find_number_l591_591089

theorem find_number:
  ∃ x : ℝ, x + 1.35 + 0.123 = 1.794 ∧ x = 0.321 :=
by
  sorry

end find_number_l591_591089


namespace rectangle_area_representation_l591_591898

theorem rectangle_area_representation
  (ABCD_is_rectangle : ∀ A B C D : ℝ × ℝ, A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2)
  (AB BC CD DA : ℝ)
  (angle_CDA_is_90 : angle CDA = 90)
  (AB_eq : AB = 6)
  (BC_eq : BC = 8)
  (CD_eq : CD = 6)
  (DA_eq : DA = 6)
  : ∃ a b c : ℕ, a + b + c = 49 ∧ (a = 0 ∧ b = 48 ∧ c = 1) := sorry

end rectangle_area_representation_l591_591898


namespace jean_to_jane_ratio_is_3_to_1_l591_591912

-- Definitions from the conditions
def combined_total : ℕ := 76
def jean_money : ℕ := 57

-- Calculate Jane's money
def jane_money (combined_total jean_money : ℕ) : ℕ :=
  combined_total - jean_money

-- Calculate the greatest common divisor
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- Calculate the ratio and simplify
def ratio (a b : ℕ) : (ℕ × ℕ) :=
  let d := gcd a b in (a / d, b / d)

-- Prove the ratio is 3:1
theorem jean_to_jane_ratio_is_3_to_1 : ratio jean_money (jane_money combined_total jean_money) = (3, 1) :=
  by sorry

end jean_to_jane_ratio_is_3_to_1_l591_591912


namespace coeff_x3_term_in_expansion_l591_591771

noncomputable def P(x : ℝ) := 3 * x^3 + 2 * x^2 + 5 * x + 4
noncomputable def Q(x : ℝ) := 7 * x^3 + 5 * x^2 + 6 * x + 7

theorem coeff_x3_term_in_expansion :
  (P * Q).coeff 3 = 38 :=
by
  sorry

end coeff_x3_term_in_expansion_l591_591771


namespace inf_geometric_mean_gt_3_inf_geometric_mean_le_2_l591_591686

variables {x y g : ℝ}
variables (hx : 0 < x) (hy : 0 < y)
variable (hg : g = Real.sqrt (x * y))

theorem inf_geometric_mean_gt_3 :
  g ≥ 3 → (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≥ 2 / Real.sqrt (1 + g)) :=
by
  sorry

theorem inf_geometric_mean_le_2 :
  g ≤ 2 → (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≤ 2 / Real.sqrt (1 + g)) :=
by
  sorry

end inf_geometric_mean_gt_3_inf_geometric_mean_le_2_l591_591686


namespace train_time_lost_l591_591351

-- Definitions 
def speed_car : ℝ := 120 -- speed of the car in km/h
def speed_train : ℝ := speed_car * 1.5 -- speed of the train in km/h (50% faster than the car)
def distance_A_to_B : ℝ := 75 -- distance between point A and B in km

-- Time calculations (in hours)
def time_car : ℝ := distance_A_to_B / speed_car
def time_train_without_stops : ℝ := distance_A_to_B / speed_train

-- Time lost by the train in hours
def time_lost_hours : ℝ := time_car - time_train_without_stops

-- Time lost by the train in minutes
def time_lost_minutes : ℝ := time_lost_hours * 60

-- Theorem to prove the time lost in minutes
theorem train_time_lost : time_lost_minutes = 12.5 :=
by
  sorry

end train_time_lost_l591_591351


namespace single_room_cost_l591_591055

theorem single_room_cost (total_rooms : ℕ) (single_rooms : ℕ) (double_room_cost : ℕ) 
  (total_revenue : ℤ) (x : ℤ) : 
  total_rooms = 260 → 
  single_rooms = 64 → 
  double_room_cost = 60 → 
  total_revenue = 14000 → 
  64 * x + (total_rooms - single_rooms) * double_room_cost = total_revenue → 
  x = 35 := 
by 
  intros h_total_rooms h_single_rooms h_double_room_cost h_total_revenue h_eqn 
  -- Add steps for proving if necessary
  sorry

end single_room_cost_l591_591055


namespace line_equation_max_area_l591_591791

-- Definitions of the circle and the point M
def circle (x y : ℝ) := (x + 3)^2 + (y - 6)^2 = 36
def M : ℝ × ℝ := (0, 3)

-- Theorem Ⅰ: Equation of the line l
theorem line_equation : ∃ (k b : ℝ), (∀ (x y : ℝ), y = k * x + b → x - y + 3 = 0) :=
by
  sorry

-- Theorem Ⅱ: Maximum area of triangle PAB
theorem max_area (A B P : ℝ × ℝ) (hA : circle A.1 A.2) (hB : circle B.1 B.2) (hP : circle P.1 P.2):
  ∃ k, k = 18 + 18 * Real.sqrt 2 ∧ 
  ∀ (triangle_area : ℝ), triangle_area = max (1 / 2 * |6 * Real.sqrt 2 * (3 * Real.sqrt 2 + 6)|) :=
by
  sorry

end line_equation_max_area_l591_591791


namespace estimate_time_pm_l591_591906

-- Definitions from the conditions
def school_start_time : ℕ := 12
def classes : List String := ["Maths", "History", "Geography", "Science", "Music"]
def class_time : ℕ := 45  -- in minutes
def break_time : ℕ := 15  -- in minutes
def classes_up_to_science : List String := ["Maths", "History", "Geography", "Science"]
def total_classes_time : ℕ := classes_up_to_science.length * (class_time + break_time)

-- Lean statement to prove that given the conditions, the time is 4 pm
theorem estimate_time_pm :
  school_start_time + (total_classes_time / 60) = 16 :=
by
  sorry

end estimate_time_pm_l591_591906


namespace no_int_poly_7_11_11_13_l591_591951

theorem no_int_poly_7_11_11_13 :
  ∀ (f : ℤ[X]), f.eval 7 = 11 ∧ f.eval 11 = 13 → false :=
by {
  intros f h,
  cases h with h7 h11,
  let a := 11,
  let b := 7,
  have diff := f.eval a - f.eval b,
  rw [h11, h7] at diff,
  have h_div : (a - b : ℤ) ∣ diff,
  { -- (a - b) divides (a^i - b^i) for all i
    sorry },
  have ab_eq := show a - b = 4, by norm_num,
  rw ab_eq at h_div,
  -- 4 ∣ 2, which is a contradiction
  have absurd : 4 ∣ 2 := by {
    have : 4 ∣ diff := h_div,
    exact this,
  },
  contradiction,
}

end no_int_poly_7_11_11_13_l591_591951


namespace area_of_rectangle_EFGH_l591_591783

theorem area_of_rectangle_EFGH :
  let short_side := 8
  let long_side := 3 * short_side
  let width_EFGH := short_side + short_side
  let length_EFGH := long_side
  (width_EFGH * length_EFGH = 384) :=
by
  let short_side := 8
  let long_side := 3 * short_side
  let width_EFGH := short_side + short_side
  let length_EFGH := long_side
  show width_EFGH * length_EFGH = 384 from sorry

end area_of_rectangle_EFGH_l591_591783


namespace sphere_surface_area_l591_591805

noncomputable def surface_area_of_sphere (PA PB PC : ℝ) (h1 : PA = 2) (h2 : PB = 3) (h3 : PC = 4) : ℝ :=
  let d := Real.sqrt (PA^2 + PB^2 + PC^2) in  -- calculate the diameter
  let r := d / 2 in  -- calculate the radius
  4 * Real.pi * r^2  -- calculate the surface area

theorem sphere_surface_area : surface_area_of_sphere 2 3 4 (by rfl) (by rfl) (by rfl) = 29 * Real.pi := 
  sorry

end sphere_surface_area_l591_591805


namespace seven_circles_six_circles_l591_591186

-- 1. Prove that it is possible to place 7 circles on a plane such that every ray from point O intersects at least three circles.
theorem seven_circles (O : Point) (circles : Fin 7 → Circle) :
  (∀ r : Ray, (∃ c1 c2 c3 : Fin 7, r.intersects circles c1 ∧ r.intersects circles c2 ∧ r.intersects circles c3)) → 
  true := 
sorry

-- 2. Prove that it is not possible to place 6 circles on a plane such that every ray from point O intersects at least three circles.
theorem six_circles (O : Point) (circles : Fin 6 → Circle) :
  (∃ r : Ray, (∀ c1 c2 c3 : Fin 6, ¬(r.intersects circles c1 ∧ r.intersects circles c2 ∧ r.intersects circles c3))) → 
  true := 
sorry

end seven_circles_six_circles_l591_591186


namespace profit_per_section_calculation_l591_591708

-- Define the initial conditions and variables
def hectares_to_m² (hectares : ℕ) : ℕ :=
  hectares * 10000

def land_divided (total_land : ℕ) (num_sons : ℕ) : ℕ :=
  total_land / num_sons

def annual_profit_per_son (total_annual_profit : ℕ) (num_sons : ℕ) : ℕ :=
  total_annual_profit / num_sons

def quarterly_profit (annual_profit : ℕ) : ℕ :=
  annual_profit / 4

def sections_of_750m² (land_in_m² : ℕ) : ℕ :=
  land_in_m² / 750

def profit_per_750m² (quarterly_profit : ℕ) (num_sections : ℕ) : ℕ :=
  quarterly_profit / num_sections

theorem profit_per_section_calculation :
  let total_land := hectares_to_m² 3 in
  let land_per_son := land_divided total_land 8 in
  let quarterly_profit_per_son := quarterly_profit 10000 in
  let num_sections := sections_of_750m² land_per_son in
  profit_per_750m² quarterly_profit_per_son num_sections = 500 :=
by {
  let total_land := hectares_to_m² 3,
  let land_per_son := land_divided total_land 8,
  let quarterly_profit_per_son := quarterly_profit 10000,
  let num_sections := sections_of_750m² land_per_son,
  show profit_per_750m² quarterly_profit_per_son num_sections = 500, 
  sorry
}

end profit_per_section_calculation_l591_591708


namespace water_charge_standard_l591_591260

theorem water_charge_standard
  (charge_below_5_tons : ℝ)
  (ratio_zhang_li : ℝ)
  (bill_zhang : ℝ)
  (bill_li : ℝ)
  (charge_below_5 : charge_below_5_tons = 0.85)
  (ratio_zhang_li_is_two_thirds : ratio_zhang_li = 2 / 3)
  (bill_zhang_is_14_6 : bill_zhang = 14.6)
  (bill_li_is_22_65 : bill_li = 22.65) :
  ∃ (charge_above_5_tons : ℝ), charge_above_5_tons = 1.15 :=
by
  -- formalize all conditions
  have h_charge_below_5 : charge_below_5_tons = 0.85 := charge_below_5
  have h_ratio_zhang_li : ratio_zhang_li = 2 / 3 := ratio_zhang_li_is_two_thirds
  have h_bill_zhang : bill_zhang = 14.6 := bill_zhang_is_14_6
  have h_bill_li : bill_li = 22.65 := bill_li_is_22_65

  -- we need to show there exists a charge_above_5_tons such that charge_above_5_tons = 1.15
  use 1.15
  sorry -- proof omitted

end water_charge_standard_l591_591260


namespace magnitude_of_c_l591_591779

noncomputable def polynomial (c : ℂ) : Polynomial ℂ :=
  (Polynomial.X^2 - 3 * Polynomial.X + 5) *
  (Polynomial.X^2 - c * Polynomial.X + 6) *
  (Polynomial.X^2 - 5 * Polynomial.X + 13)

theorem magnitude_of_c (c : ℂ) :
  (∀ x, polynomial c x = 0 → Polynomial.X x) ∧
  Polynomial.degree (polynomial c) = 4 →
  |c| = 2 * Real.sqrt 6 :=
sorry

end magnitude_of_c_l591_591779


namespace profit_per_package_l591_591785

theorem profit_per_package
  (packages_first_center_per_day : ℕ)
  (packages_second_center_multiplier : ℕ)
  (weekly_profit : ℕ)
  (days_per_week : ℕ)
  (H1 : packages_first_center_per_day = 10000)
  (H2 : packages_second_center_multiplier = 3)
  (H3 : weekly_profit = 14000)
  (H4 : days_per_week = 7) :
  (weekly_profit / (packages_first_center_per_day * days_per_week + 
                    packages_second_center_multiplier * packages_first_center_per_day * days_per_week) : ℝ) = 0.05 :=
by
  sorry

end profit_per_package_l591_591785


namespace travel_time_from_edmonton_to_calgary_l591_591765

def edmonton_to_red_deer_distance : ℝ := 220
def red_deer_to_calgary_distance : ℝ := 110
def travel_speed : ℝ := 110
def total_distance : ℝ := edmonton_to_red_deer_distance + red_deer_to_calgary_distance

theorem travel_time_from_edmonton_to_calgary :
  total_distance / travel_speed = 3 :=
by
  -- provided distance from Edmonton to Red Deer
  have h1 : edmonton_to_red_deer_distance = 220 := by rfl
  -- provided distance from Red Deer to Calgary
  have h2 : red_deer_to_calgary_distance = 110 := by rfl
  -- provided travel speed
  have h3 : travel_speed = 110 := by rfl
  -- define the total distance
  have h4 : total_distance = edmonton_to_red_deer_distance + red_deer_to_calgary_distance := by rfl
  -- calculate total distance
  calc total_distance 
    = 220 + 110 : by rw [h1, h2]
    ... = 330 : by norm_num
  -- prove the time calculation
  have h5 : total_distance / travel_speed = 330 / 110 := by rw [← h4, h3]
  have h6 : 330 / 110 = 3 := by norm_num
  rw [h5, h6]
  sorry

end travel_time_from_edmonton_to_calgary_l591_591765


namespace domain_of_f_l591_591977

open Set

def f (x : ℝ) : ℝ := (sqrt (4 - x)) / (x - 3)

theorem domain_of_f :
  {x : ℝ | (4 - x ≥ 0) ∧ (x - 3 ≠ 0)} = {x : ℝ | x ∈ (-∞, 3) ∪ (3, 4]} :=
by
  sorry

end domain_of_f_l591_591977


namespace max_d_for_range_of_fx_l591_591094

theorem max_d_for_range_of_fx : 
  ∀ (d : ℝ), (∃ x : ℝ, x^2 + 4*x + d = -3) → d ≤ 1 := 
by
  sorry

end max_d_for_range_of_fx_l591_591094


namespace total_hours_l591_591948

variable (Jacob_time : ℕ) (Greg_time : ℕ) (Patrick_time : ℕ) (Samantha_time : ℕ)

def Jacob_time_def : Jacob_time = 18 := by sorry

def Greg_time_def : Greg_time = Jacob_time - 6 :=
by sorry

def Patrick_time_def : Patrick_time = 2 * Greg_time - 4 :=
by sorry

def Samantha_time_def : Samantha_time = (3 * Patrick_time) / 2 :=
by sorry

theorem total_hours : Jacob_time + Greg_time + Patrick_time + Samantha_time = 80 := by
  rw [Jacob_time_def, Greg_time_def, Patrick_time_def, Samantha_time_def]
  sorry

end total_hours_l591_591948


namespace inclination_angle_correct_l591_591345

noncomputable def angle_of_inclination (a h : ℝ) : ℝ :=
  arctan (1 / (2 * sqrt 3))

theorem inclination_angle_correct (a h : ℝ) :
  let theta := angle_of_inclination a h in
  theta = arctan (1 / (2 * sqrt 3)) :=
  by
  let theta := angle_of_inclination a h
  sorry

end inclination_angle_correct_l591_591345


namespace apples_total_l591_591744

def benny_apples : ℕ := 2
def dan_apples : ℕ := 9
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_total : total_apples = 11 :=
by
    sorry

end apples_total_l591_591744


namespace triangle_AB_length_l591_591197

theorem triangle_AB_length
  (A B C : Type)
  [IsTriangle A B C]
  (angle_B : angle B = 30)
  (angle_C : angle C = 90)
  (length_BC : length (side B C) = 12) :
  length (side A B) = 6 :=
sorry

end triangle_AB_length_l591_591197


namespace baseball_game_earnings_l591_591006

theorem baseball_game_earnings (W S : ℝ) 
  (h1 : W + S = 4994.50) 
  (h2 : W = S - 1330.50) : 
  S = 3162.50 := 
by 
  sorry

end baseball_game_earnings_l591_591006


namespace four_person_planning_committee_l591_591044

theorem four_person_planning_committee
  (n : ℕ)
  (h : nat.choose n 2 = 21) :
  nat.choose n 4 = 35 :=
sorry

end four_person_planning_committee_l591_591044


namespace incorrect_statements_l591_591453

-- Definitions based on conditions from the problem.

def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c < 0
def solution_set (a b c : ℝ) : Set ℝ := {x | quadratic_inequality a b c x}
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Lean statements of the conditions and the final proof problem.
theorem incorrect_statements (a b c : ℝ) (M : Set ℝ) :
  (M = ∅ → (a < 0 ∧ discriminant a b c < 0) → false) ∧
  (M = {x | x ≠ x0} → a < b → (a + 4 * c) / (b - a) = 2 + 2 * Real.sqrt 2 → false) := sorry

end incorrect_statements_l591_591453


namespace ratio_blue_yellow_l591_591560

theorem ratio_blue_yellow (total_butterflies blue_butterflies black_butterflies : ℕ)
  (h_total : total_butterflies = 19)
  (h_blue : blue_butterflies = 6)
  (h_black : black_butterflies = 10) :
  (blue_butterflies : ℚ) / (total_butterflies - blue_butterflies - black_butterflies : ℚ) = 2 / 1 := 
by {
  sorry
}

end ratio_blue_yellow_l591_591560


namespace simplify_expression_l591_591605

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = 2 * x⁻¹ * y⁻¹ * z⁻¹ :=
by
  sorry

end simplify_expression_l591_591605


namespace shaded_area_eq_l591_591634

def side_length : Real := 8
def radius : Real := 3 * Real.sqrt 2

theorem shaded_area_eq :
  let square_area := side_length ^ 2
  let triangle_area := 1 / 2 * 3 * 3
  let sector_area := 1 / 4 * π * (radius ^ 2)
  (square_area - 4 * (triangle_area + sector_area)) = 46 - 18 * π :=
by
  sorry

end shaded_area_eq_l591_591634


namespace domain_g_l591_591077

-- Definition of the function g(x)
def g (x : ℝ) : ℝ :=
  Real.logb (1 / 3) (Real.logb 9 (Real.logb (1 / 9) (Real.logb 81 (Real.logb (1 / 81) x))))

-- Required proof that the domain of g is as stated
theorem domain_g :
  let I := (1 / 81^(81) : ℝ) in
  let a := 1 / 81 in
  ∀ x, 
    g x = g x →
    I < x ∧ x < a → 
    let m := 1 in
    let n := 81 in
    (m + n = 82) 
    := 
sorry

end domain_g_l591_591077


namespace abs_values_opposite_sign_abs_value_equality_rational_numbers_correct_statement_l591_591674

noncomputable def smallest_integer := ∃ n : ℤ, ∀ m : ℤ, m ≥ n

theorem abs_values_opposite_sign (x y : ℤ)
  (hx : x < 0) (hy : y > 0) : |x| = |y| ↔ x = -y :=
by {
  sorry
}

theorem abs_value_equality (x y : ℤ) : |x| = |y| ↔ x = y ∨ x = -y :=
by {
  sorry
}

theorem rational_numbers (x : ℚ) : x > 0 ∨ x < 0 ∨ x = 0 :=
by {
  sorry
}

theorem correct_statement : 
  (¬ smallest_integer) ∧ 
  (∀ x y : ℤ, (abs_values_opposite_sign x y (lt_of_le_of_ne (le_of_lt hx) (ne.symm hy)) (le_of_lt hy)) = true) ∧ 
  (∀ x y : ℤ, abs_value_equality x y) ∧ 
  (∀ x : ℚ, rational_numbers x) :=
by {
  sorry
}

end abs_values_opposite_sign_abs_value_equality_rational_numbers_correct_statement_l591_591674


namespace volume_of_sphere_l591_591360

-- Assume the context as given in conditions
def equilateral_triangle (A B C: Point) : Prop := -- definition of an equilateral triangle

def vertices_on_sphere (A B C O: Point) (r: ℝ) : Prop :=
  dist O A = r ∧ dist O B = r ∧ dist O C = r 

def centroid (A B C G : Point) : Prop :=
  -- definition of the centroid of a triangle

def circum_circle_area (A B C: Point) (area: ℝ) : Prop :=
  -- Area of the circumcircle of triangle ABC is area

-- Define the conditions
variables {A B C O G : Point}
variables {r : ℝ}
variables {area : ℝ}
axiom h_equilateral : equilateral_triangle A B C
axiom h_vertices_on_sphere : vertices_on_sphere A B C O r
axiom h_centroid : centroid A B C G
axiom h_og : dist O G = (sqrt 3) / 3
axiom h_circum_circle_area : circum_circle_area A B C (2 * π / 3)

-- Define the sphere's volume using its radius
def sphere_volume (radius : ℝ) : ℝ :=
  4 / 3 * π * radius^3

noncomputable def find_volume_of_sphere : ℝ :=
  if h : true then (4 / 3 * π * 1^3) else (0 : ℝ)

-- Prove the statement
theorem volume_of_sphere :
  sphere_volume 1 = 4/3 * π := sorry

end volume_of_sphere_l591_591360


namespace ball_hits_ground_time_l591_591264

def ball_height (t : ℝ) : ℝ := -20 * t^2 + 30 * t + 60

theorem ball_hits_ground_time :
  ∃ t : ℝ, ball_height t = 0 ∧ t = (3 + Real.sqrt 57) / 4 :=
sorry

end ball_hits_ground_time_l591_591264


namespace problem1_problem2_l591_591302

-- Definition of a double root equation with the given condition
def is_double_root_equation (a b c : ℝ) := 
  ∃ x1 x2 : ℝ, a * x1 = 2 * a * x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0

-- Proving that x² - 6x + 8 = 0 is a double root equation
theorem problem1 : is_double_root_equation 1 (-6) 8 :=
  sorry

-- Proving that if (x-8)(x-n) = 0 is a double root equation, n is either 4 or 16
theorem problem2 (n : ℝ) (h : is_double_root_equation 1 (-8 - n) (8 * n)) :
  n = 4 ∨ n = 16 :=
  sorry

end problem1_problem2_l591_591302


namespace triangle_side_length_l591_591451

theorem triangle_side_length (x : ℕ) 
  (h_options : x ∈ {2, 3, 6, 13}) 
  (h_5_8 : 5 < 8) :
  3 < x ∧ x < 13 → x = 6 :=
by
  sorry

end triangle_side_length_l591_591451


namespace cost_of_chairs_l591_591566

-- Given conditions
def total_spent : ℕ := 56
def cost_of_table : ℕ := 34
def number_of_chairs : ℕ := 2

-- The target definition
def cost_of_one_chair : ℕ := 11

-- Statement to prove
theorem cost_of_chairs (N : ℕ) (T : ℕ) (C : ℕ) (x : ℕ) (hN : N = total_spent) (hT : T = cost_of_table) (hC : C = number_of_chairs) : x = cost_of_one_chair ↔ N - T = C * x :=
by
  sorry

end cost_of_chairs_l591_591566


namespace ceilFunc_prop_1_ceilFunc_prop_2_ceilFunc_prop_3_ceilFunc_prop_4_l591_591759

def ceilFunc (x : ℝ) : ℤ := ⌈x⌉

theorem ceilFunc_prop_1 (x : ℝ) : ceilFunc (2 * x) ≠ 2 * ceilFunc x := 
by 
  sorry

theorem ceilFunc_prop_2 {x₁ x₂ : ℝ} 
  (h : ceilFunc x₁ = ceilFunc x₂) : abs (x₁ - x₂) < 1 :=
by
  sorry

theorem ceilFunc_prop_3 (x₁ x₂ : ℝ) : ceilFunc (x₁ + x₂) ≤ ceilFunc x₁ + ceilFunc x₂ := 
by 
  sorry

theorem ceilFunc_prop_4 (x : ℝ) : ceilFunc x + ceilFunc (x + 1/2) ≠ ceilFunc (2 * x) := 
by 
  sorry

end ceilFunc_prop_1_ceilFunc_prop_2_ceilFunc_prop_3_ceilFunc_prop_4_l591_591759


namespace switches_in_position_a_after_steps_l591_591642

theorem switches_in_position_a_after_steps :
  let initial_labels := {d : ℕ | ∃ x y z : ℕ, x ≤ 11 ∧ y ≤ 11 ∧ z ≤ 11 ∧ d = 2^x * 3^y * 5^z},
      switch_positions : ℕ → ℕ → ℕ → ℕ := λ x y z step, (step + x + y + z) % 6
  in (finset.univ.filter (λ label, ∃ (x y z : ℕ) (h : x ≤ 11 ∧ y ≤ 11 ∧ z ≤ 11 ∧ label = 2^x * 3^y * 5^z), switch_positions x y z 1000 = 0)).card = 136 := 
sorry

end switches_in_position_a_after_steps_l591_591642


namespace pentagon_area_is_1888_5_l591_591341

def pentagon_side_lengths := {14, 21, 22, 29, 35}

theorem pentagon_area_is_1888_5 (a b c d e: ℝ) (hx : {a, b, c, d, e} = pentagon_side_lengths) :
  (∃ r s: ℝ, r^2 + s^2 = e^2 ∧ r = b - d ∧ s = c - a /\
  bc = b * c ∧ triangle_area = (1/2) * r * s ∧ pentagon_area = bc - triangle_area) :=
  pentagon_area = 1888.5 :=
  sorry

end pentagon_area_is_1888_5_l591_591341


namespace probability_of_three_5s_in_eight_rolls_l591_591016

-- Conditions
def total_outcomes : ℕ := 6 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3

-- The probability that the number 5 appears exactly three times in eight rolls of a fair die
theorem probability_of_three_5s_in_eight_rolls :
  (favorable_outcomes / total_outcomes : ℚ) = (56 / 1679616 : ℚ) :=
by
  sorry

end probability_of_three_5s_in_eight_rolls_l591_591016


namespace triangle_auv_is_isosceles_l591_591871

/-- Consider \( \triangle ABC \) with points \( X \) and \( Y \) on line \( BC \) such that \( X, B, C, \) and \( Y \) are in sequence.
\( BX \cdot AC = CY \cdot AB \). Let \( O_1 \) and \( O_2 \) be the circumcenters of \( \triangle ACX \) and \( \triangle ABY \), respectively.
The line \( O_1O_2 \) intersects \( AB \) and \( AC \) at points \( U \) and \( V \) respectively.
We need to prove \( \triangle AUV \) is isosceles. -/
theorem triangle_auv_is_isosceles
  (A B C X Y O1 O2 U V D : Point)
  (h1 : collinear_points X B C Y)
  (h2 : BX * AC = CY * AB)
  (h3 : is_circumcenter O1 (triangle A C X))
  (h4 : is_circumcenter O2 (triangle A B Y))
  (h5 : intersection_points O1 O2 AB U)
  (h6 : intersection_points O1 O2 AC V)
  (h7 : line_through_point D A)
  (h8 : is_angle_bisector (angle_bisector A B C D))
  (h9 : power_of_point D O1 = power_of_point D O2)
  (h10 : perp AD UV) :
  isosceles_triangle A U V := sorry

end triangle_auv_is_isosceles_l591_591871


namespace suraj_average_increase_l591_591684

namespace SurajAverage

theorem suraj_average_increase (A : ℕ) (h : (16 * A + 112) / 17 = A + 6) : (A + 6) = 16 :=
  by
  sorry

end SurajAverage

end suraj_average_increase_l591_591684


namespace union_A_B_equals_union_set_l591_591474

def A := { x : ℝ | x^2 < 4 }
def B := { y : ℝ | ∃ x : ℝ, x ∈ A ∧ y = x^2 - 2x - 1 }
def union_set := { x : ℝ | -2 ≤ x ∧ x < 7 }

theorem union_A_B_equals_union_set : A ∪ B = union_set :=
by
  sorry

end union_A_B_equals_union_set_l591_591474


namespace fraction_zero_solution_l591_591869

theorem fraction_zero_solution (x : ℝ) (h : (|x| - 2) / (x - 2) = 0) : x = -2 :=
sorry

end fraction_zero_solution_l591_591869


namespace domain_of_sqrt_l591_591263

def domain_of_f : Set ℝ := {x | x >= 1}

theorem domain_of_sqrt (x : ℝ) : (∃ y, f(x) = y) ↔ x ∈ domain_of_f :=
by
  sorry

end domain_of_sqrt_l591_591263


namespace volume_parallelepiped_l591_591083

noncomputable def volume_of_parallelepiped (m n p d : ℝ) : ℝ :=
  if h : m > 0 ∧ n > 0 ∧ p > 0 ∧ d > 0 then
    m * n * p * (d^3) / (m^2 + n^2 + p^2)^(3/2)
  else 0

theorem volume_parallelepiped (m n p d : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hd : d > 0) :
  volume_of_parallelepiped m n p d = m * n * p * (d^3) / (m^2 + n^2 + p^2)^(3/2) := by
  sorry

end volume_parallelepiped_l591_591083


namespace santino_total_fruits_l591_591593

theorem santino_total_fruits :
  let papayas := [10, 12],
      mangos := [18, 20, 22],
      apples := [14, 15, 16, 17],
      oranges := [20, 23, 25, 27, 30] in
  (papayas.sum + mangos.sum + apples.sum + oranges.sum) = 269 := 
by
  sorry

end santino_total_fruits_l591_591593


namespace perpendicular_lines_condition_l591_591322

theorem perpendicular_lines_condition (m : ℝ) :
  m = -1 → (∀ x y : ℝ, (mx + (2m - 1) * y + 1 = 0) →
  (3x + m * y + 9 = 0) → (-(m/(2m-1)) * -(3/m) = -1)) → 
  (∀ m : ℝ, 2 * m * (m + 1) = 0 → m = 0 ∨ m = -1) :=
by intros hyp hyp_eq;
   sorry

end perpendicular_lines_condition_l591_591322


namespace lcm_of_numbers_is_91_l591_591629

def ratio (a b : ℕ) (p q : ℕ) : Prop := p * b = q * a

theorem lcm_of_numbers_is_91 (a b : ℕ) (h_ratio : ratio a b 7 13) (h_gcd : Nat.gcd a b = 15) :
  Nat.lcm a b = 91 := 
by sorry

end lcm_of_numbers_is_91_l591_591629


namespace first_stopover_distance_l591_591538

theorem first_stopover_distance 
  (total_distance : ℕ) 
  (second_stopover_distance : ℕ) 
  (distance_after_second_stopover : ℕ) :
  total_distance = 436 → 
  second_stopover_distance = 236 → 
  distance_after_second_stopover = 68 →
  second_stopover_distance - (total_distance - second_stopover_distance - distance_after_second_stopover) = 104 :=
by
  intros
  sorry

end first_stopover_distance_l591_591538


namespace log_expression_simplifies_to_one_l591_591277

theorem log_expression_simplifies_to_one :
  (Real.log 5)^2 + Real.log 50 * Real.log 2 = 1 :=
by 
  sorry

end log_expression_simplifies_to_one_l591_591277


namespace chair_cost_l591_591572

-- Define the conditions
def total_spent : ℕ := 56
def table_cost : ℕ := 34
def num_chairs : ℕ := 2

-- Define the statement we need to prove
theorem chair_cost :
  ∃ (chair_cost : ℕ), 2 * chair_cost + table_cost = total_spent ∧ chair_cost = 11 :=
by
  use 11
  split
  sorry -- proof goes here, skipped as per instructions

end chair_cost_l591_591572


namespace power_function_increasing_m_eq_2_l591_591500

theorem power_function_increasing_m_eq_2 (m : ℝ) :
  (∀ x > 0, (m^2 - m - 1) * x^m > 0) → m = 2 :=
by
  sorry

end power_function_increasing_m_eq_2_l591_591500


namespace simplify_expression_l591_591266

theorem simplify_expression (x : ℝ) 
  (h1 : x^2 - 4*x + 3 = (x-3)*(x-1))
  (h2 : x^2 - 6*x + 9 = (x-3)^2)
  (h3 : x^2 - 6*x + 8 = (x-2)*(x-4))
  (h4 : x^2 - 8*x + 15 = (x-3)*(x-5)) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = (x-1)*(x-5) / ((x-2)*(x-4)) :=
by
  sorry

end simplify_expression_l591_591266


namespace geometric_series_common_ratio_l591_591735

theorem geometric_series_common_ratio (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) 
  (h₃ : S = a / (1 - r)) : r = 21 / 25 := 
sorry

end geometric_series_common_ratio_l591_591735


namespace floor_factorial_fraction_l591_591069

theorem floor_factorial_fraction : 
  (⌊(2021.factorial + 2018.factorial) / (2020.factorial + 2019.factorial)⌋ = 2020) :=
begin
  sorry
end

end floor_factorial_fraction_l591_591069


namespace radius_of_inscribed_circle_of_triangle_ACD_l591_591714

noncomputable def inscribed_circle_radius (R : ℝ) := 
let a := R in
let b := R * Real.sqrt 3 + R in
let c := 2 * R in
(a + b - c) / 2

theorem radius_of_inscribed_circle_of_triangle_ACD : 
  inscribed_circle_radius (3 + Real.sqrt 3) = Real.sqrt 3 :=
by
  sorry

end radius_of_inscribed_circle_of_triangle_ACD_l591_591714


namespace John_writing_years_l591_591202

def books_written (total_earnings per_book_earning : ℕ) : ℕ :=
  total_earnings / per_book_earning

def books_per_year (months_in_year months_per_book : ℕ) : ℕ :=
  months_in_year / months_per_book

def years_writing (total_books books_per_year : ℕ) : ℕ :=
  total_books / books_per_year

theorem John_writing_years :
  let total_earnings := 3600000
  let per_book_earning := 30000
  let months_in_year := 12
  let months_per_book := 2
  let total_books := books_written total_earnings per_book_earning
  let books_per_year := books_per_year months_in_year months_per_book
  years_writing total_books books_per_year = 20 := by
sorry

end John_writing_years_l591_591202


namespace problem_solution_l591_591850

def count_multiples_of_5_not_15 : ℕ := 
  let count_up_to (m n : ℕ) := n / m
  let multiples_of_5_up_to_300 := count_up_to 5 299
  let multiples_of_15_up_to_300 := count_up_to 15 299
  multiples_of_5_up_to_300 - multiples_of_15_up_to_300

theorem problem_solution : count_multiples_of_5_not_15 = 40 := by
  sorry

end problem_solution_l591_591850


namespace boat_average_speed_l591_591040

-- Definitions for the conditions
def upstream_speed_wrt_water : ℝ := 4
def downstream_speed_wrt_water : ℝ := 7
def river_current : ℝ := 2
def distance_between_towns (D : ℝ) : ℝ := D

-- Lean statement to prove the average speed of the boat for the round-trip
theorem boat_average_speed (D : ℝ) (hD : D > 0) :
  let upstream_speed := upstream_speed_wrt_water - river_current
      downstream_speed := downstream_speed_wrt_water + river_current
      T_up := D / upstream_speed
      T_down := D / downstream_speed
      T_total := T_up + T_down
      total_distance := 2 * D
      V_avg := total_distance / T_total
  in V_avg = 36 / 11 := sorry

end boat_average_speed_l591_591040


namespace arithmetic_sequence_of_inverses_sum_of_bn_l591_591223

noncomputable def T (a : ℕ → ℝ) (n : ℕ) := 2 - 2 * a n
noncomputable def b (a : ℕ → ℝ) (n : ℕ) := (1 - a n) * (1 - a (n + 1))

theorem arithmetic_sequence_of_inverses {a : ℕ → ℝ} (h : ∀ n, T a n = 2 - 2 * a n) :
  ∃ d : ℝ, ∃ c : ℝ, ∀ n, (1 / T a n) = c + d * n :=
sorry

theorem sum_of_bn {a : ℕ → ℝ} (h : ∀ n, T a n = 2 - 2 * a n) (h_b : ∀ n, b a n = (1 - a n) * (1 - a (n + 1))) :
  ∀ n, (∑ k in finset.range n, b a k) = n / (3 * n + 9) :=
sorry

end arithmetic_sequence_of_inverses_sum_of_bn_l591_591223


namespace T_6_value_l591_591690

variable (x : ℝ)

def T (m : ℕ) : ℝ := x^m + (1 / x)^m

theorem T_6_value (h : x + 1 / x = 5) : T x 6 = 12098 :=
sorry

end T_6_value_l591_591690


namespace complex_add_l591_591441

theorem complex_add (a b : ℝ) (i : ℂ) (h1 : 1 + 2 * i = a + b * i) : a + b = 3 := 
by {
  have h2 : (1 : ℂ).re = a.re := sorry,
  have h3 : (2 : ℂ).im = b.im := sorry,
  sorry
}

end complex_add_l591_591441


namespace value_of_a_l591_591450

theorem value_of_a 
  (a : ℝ)
  (f : ℝ → ℝ := λ x, x^2 + a * x - 1) 
  (h₁ : ∀ x ∈ set.Icc (0 : ℝ) (3 : ℝ), f x ≥ -2)
  (h₂ : ∃ x ∈ set.Icc (0 : ℝ) (3 : ℝ), f x = -2) :
  a = -10 / 3 :=
by
  sorry

end value_of_a_l591_591450


namespace problem1_problem2_l591_591063

theorem problem1 :
  6 * real.sqrt (1 / 9) - real.cbrt 27 + (real.sqrt 2) ^ 2 = 1 :=
sorry

theorem problem2 :
  2 * real.sqrt 2 + real.sqrt 9 + real.cbrt (-8) + abs (real.sqrt 2 - 2) = real.sqrt 2 + 3 :=
sorry

end problem1_problem2_l591_591063


namespace total_number_of_chickens_is_76_l591_591530

theorem total_number_of_chickens_is_76 :
  ∀ (hens roosters chicks : ℕ),
    hens = 12 →
    (∀ n, roosters = hens / 3) →
    (∀ n, chicks = hens * 5) →
    hens + roosters + chicks = 76 :=
by
  intros hens roosters chicks h1 h2 h3
  rw [h1, h2 12, h3 12]
  ring
  sorry

end total_number_of_chickens_is_76_l591_591530


namespace difference_thursday_tuesday_l591_591940

-- Define the amounts given on each day
def amount_tuesday : ℕ := 8
def amount_wednesday : ℕ := 5 * amount_tuesday
def amount_thursday : ℕ := amount_wednesday + 9

-- Problem statement: prove that the difference between Thursday's and Tuesday's amount is $41
theorem difference_thursday_tuesday : amount_thursday - amount_tuesday = 41 := by
  sorry

end difference_thursday_tuesday_l591_591940


namespace Shekar_average_is_74_l591_591242

def Shekar_marks := {math : ℕ, science : ℕ, social_studies : ℕ, english : ℕ, biology : ℕ}

def average_marks (marks : Shekar_marks) : ℕ :=
  (marks.math + marks.science + marks.social_studies + marks.english + marks.biology) / 5

theorem Shekar_average_is_74 (marks : Shekar_marks) (hm : marks.math = 76)
  (hs : marks.science = 65) (hss : marks.social_studies = 82) 
  (he : marks.english = 62) (hb : marks.biology = 85) :
  average_marks marks = 74 :=
by
  sorry

end Shekar_average_is_74_l591_591242


namespace sum_possible_m_values_l591_591137

theorem sum_possible_m_values :
  ∑ m in Finset.filter (λ m : ℤ, 0 < 3 * m ∧ 3 * m < 21) (Finset.Icc 0 21), m = 21 :=
by
  sorry

end sum_possible_m_values_l591_591137


namespace circle_standard_form_circle_parametric_and_extremum_l591_591158

-- First part: converting polar coordinate equation to standard form.
theorem circle_standard_form :
  ∀ (ρ θ : ℝ), 
    (ρ^2 - 4 * real.sqrt 2 * ρ * real.cos (θ - real.pi / 4) + 6 = 0) →
    ∃ (x y : ℝ), (x = ρ * real.cos θ) ∧ (y = ρ * real.sin θ) ∧ ((x - 2)^2 + (y - 2)^2 = 2) :=
by sorry

-- Second part: parametic equations and extremum of \(x + y\).
theorem circle_parametric_and_extremum :
  ∀ (α : ℝ),
    ∃ (x y : ℝ), 
      (x = 2 + real.sqrt 2 * real.cos α) ∧ 
      (y = 2 + real.sqrt 2 * real.sin α) ∧ 
      (∀ (α : ℝ), 4 + 2 * real.sin (α + real.pi / 4) ∈ set.Icc (2 : ℝ) (6 : ℝ)) :=
by sorry

end circle_standard_form_circle_parametric_and_extremum_l591_591158


namespace hyperbola_x_coordinate_solution_l591_591815

noncomputable def hyperbola_x_coordinate (x y : ℝ) : ℝ :=
  if (x^2 / 16 - y^2 / 9 = 1) 
     ∧ (abs(x - 16 / 5) = (dist (x, y) (5, 0) + dist (x, y) (-5, 0)) / 2)
  then x else 0

theorem hyperbola_x_coordinate_solution : ∀ x y : ℝ, 
  (x^2 / 16 - y^2 / 9 = 1) 
  ∧ (abs(x - 16 / 5) = (dist (x, y) (5, 0) + dist (x, y) (-5, 0)) / 2) 
  → x = -64 / 5 :=
begin
  sorry
end

end hyperbola_x_coordinate_solution_l591_591815


namespace total_chickens_l591_591527

   def number_of_hens := 12
   def hens_to_roosters_ratio := 3
   def chicks_per_hen := 5

   theorem total_chickens (h : number_of_hens = 12)
                          (r : hens_to_roosters_ratio = 3)
                          (c : chicks_per_hen = 5) :
     number_of_hens + (number_of_hens / hens_to_roosters_ratio) + (number_of_hens * chicks_per_hen) = 76 :=
   by
     sorry
   
end total_chickens_l591_591527


namespace compare_values_l591_591854

noncomputable def a : ℝ := Real.log 1.01
def b : ℝ := 1 / 101
def c : ℝ := Real.sin 0.01

theorem compare_values : a > b ∧ c > a :=
by
  -- to be proved
  sorry

end compare_values_l591_591854


namespace days_per_week_equals_two_l591_591201

-- Definitions based on conditions
def hourly_rate : ℕ := 10
def hours_per_delivery : ℕ := 3
def total_weeks : ℕ := 6
def total_earnings : ℕ := 360

-- Proof statement: determine the number of days per week Jamie delivers flyers is 2
theorem days_per_week_equals_two (d : ℕ) :
  10 * (total_weeks * d * hours_per_delivery) = total_earnings → d = 2 := by
  sorry

end days_per_week_equals_two_l591_591201


namespace croissant_cost_l591_591753

-- Definitions
def starting_gift_card : ℝ := 100
def cost_latte : ℝ := 3.75
def cost_cookie : ℝ := 1.25
def lattes_per_week : ℕ := 7
def cookies_number : ℕ := 5
def end_gift_card : ℝ := 43
noncomputable def cost_croissant : ℝ := 3.5

-- Theorem stating the problem
theorem croissant_cost :
  let cost_lattes := lattes_per_week * cost_latte,
      cost_cookies := cookies_number * cost_cookie,
      cost_spent := starting_gift_card - end_gift_card in
  cost_lattes + 7 * cost_croissant + cost_cookies = cost_spent :=
by
  let cost_lattes := lattes_per_week * cost_latte
  let cost_cookies := cookies_number * cost_cookie
  let cost_spent := starting_gift_card - end_gift_card
  have : cost_lattes + 7 * cost_croissant + cost_cookies = cost_spent := sorry
  exact this

end croissant_cost_l591_591753


namespace total_chickens_l591_591528

   def number_of_hens := 12
   def hens_to_roosters_ratio := 3
   def chicks_per_hen := 5

   theorem total_chickens (h : number_of_hens = 12)
                          (r : hens_to_roosters_ratio = 3)
                          (c : chicks_per_hen = 5) :
     number_of_hens + (number_of_hens / hens_to_roosters_ratio) + (number_of_hens * chicks_per_hen) = 76 :=
   by
     sorry
   
end total_chickens_l591_591528


namespace donna_fully_loaded_truck_weight_l591_591392

-- Define conditions
def empty_truck_weight : ℕ := 12000
def soda_crate_weight : ℕ := 50
def soda_crate_count : ℕ := 20
def dryer_weight : ℕ := 3000
def dryer_count : ℕ := 3

-- Calculate derived weights
def soda_total_weight : ℕ := soda_crate_weight * soda_crate_count
def fresh_produce_weight : ℕ := 2 * soda_total_weight
def dryer_total_weight : ℕ := dryer_weight * dryer_count

-- Define target weight of fully loaded truck
def fully_loaded_truck_weight : ℕ :=
  empty_truck_weight + soda_total_weight + fresh_produce_weight + dryer_total_weight

-- State and prove the theorem
theorem donna_fully_loaded_truck_weight :
  fully_loaded_truck_weight = 24000 :=
by
  -- Provide necessary calculations and proof steps if needed
  sorry

end donna_fully_loaded_truck_weight_l591_591392


namespace perpendicular_bisector_eq_l591_591127

def Point := ℝ × ℝ

def M : Point := (2, 4)
def N : Point := (6, 2)

theorem perpendicular_bisector_eq : 
  ∃ (a b c : ℝ), a = 2 ∧ b = -1 ∧ c = -5 ∧ ∀ (P : Point), 
  let (x, y) := P in
  (x - M.fst)^2 + (y - M.snd)^2 = (x - N.fst)^2 + (y - N.snd)^2 ↔ a * x + b * y + c = 0 :=
by
  use 2, -1, -5
  sorry

end perpendicular_bisector_eq_l591_591127


namespace total_amount_collected_l591_591290

-- Define ticket prices and quantities
def adult_ticket_price : ℕ := 12
def child_ticket_price : ℕ := 4
def total_tickets_sold : ℕ := 130
def adult_tickets_sold : ℕ := 40

-- Calculate the number of child tickets sold
def child_tickets_sold : ℕ := total_tickets_sold - adult_tickets_sold

-- Calculate the total amount collected from adult tickets
def total_adult_amount_collected : ℕ := adult_tickets_sold * adult_ticket_price

-- Calculate the total amount collected from child tickets
def total_child_amount_collected : ℕ := child_tickets_sold * child_ticket_price

-- Prove the total amount collected from ticket sales
theorem total_amount_collected : total_adult_amount_collected + total_child_amount_collected = 840 := by
  sorry

end total_amount_collected_l591_591290


namespace goldfish_cost_graph_l591_591846

theorem goldfish_cost_graph :
  (∀ (n : ℕ), 3 ≤ n ∧ n ≤ 15 → ∃ (c : ℕ), c = 20 * n) →
  (∃ (points : set (ℕ × ℕ)), points = {(n, 20 * n) | 3 ≤ n ∧ n ≤ 15} ∧ points.finite) :=
begin
  intros h,
  let points := {(n, 20 * n) | n ∈ finset.range (15 + 1) ∧ 3 ≤ n},
  use points,
  split,
  { ext,
    split,
    -- fill in the proof details if necessary
    -- sorry for skipping proof
    sorry
  },
  -- prove that the set of points is finite
  exact finset.finite_to_set points
end

end goldfish_cost_graph_l591_591846


namespace y1_lt_y2_of_linear_graph_l591_591897

/-- In the plane rectangular coordinate system xOy, if points A(2, y1) and B(5, y2) 
    lie on the graph of a linear function y = x + b (where b is a constant), then y1 < y2. -/
theorem y1_lt_y2_of_linear_graph (y1 y2 b : ℝ) (hA : y1 = 2 + b) (hB : y2 = 5 + b) : y1 < y2 :=
by
  sorry

end y1_lt_y2_of_linear_graph_l591_591897


namespace relational_symbols_correct_l591_591235

-- Defining the entities and the relationships
variables {M : Type} {a : Type} {b : Type} {α : Type}

-- Conditions
variable (is_point_on_line : M ∈ a)
variable (is_line_in_plane : b ⊂ α)

-- The proof statement
theorem relational_symbols_correct :
  is_point_on_line ∧ is_line_in_plane = (M ∈ a ∧ b ⊂ α) :=
sorry

end relational_symbols_correct_l591_591235


namespace area_of_iron_plate_l591_591641

-- Define the conversion from centimeters to meters
def cm_to_m (cm : ℝ) : ℝ :=
  cm / 100

-- Define the width and length in cm
def width_cm : ℝ := 500
def length_cm : ℝ := 800

-- Define the width and length in meters using the conversion function
def width_m : ℝ := cm_to_m width_cm
def length_m : ℝ := cm_to_m length_cm

-- Define the area calculation in square meters
def area_m2 (width : ℝ) (length : ℝ) : ℝ :=
  width * length

-- State the theorem that needs to be proven
theorem area_of_iron_plate : area_m2 width_m length_m = 40 := by
  sorry

end area_of_iron_plate_l591_591641


namespace police_emergency_number_prime_divisor_l591_591035

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 1000 * k + 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n :=
sorry

end police_emergency_number_prime_divisor_l591_591035


namespace max_integer_value_l591_591493

theorem max_integer_value (x : ℝ) : ∃ (m : ℤ), m = 53 ∧ ∀ y : ℝ, (1 + 13 / (3 * y^2 + 9 * y + 7) ≤ m) := 
sorry

end max_integer_value_l591_591493


namespace problem_a_problem_b_problem_c_incorrect_problem_c_correct_problem_d_incorrect_problem_d_correct_l591_591612

-- Probability of Team A winning 
def prob_A_wins (p_A : ℚ) := p_A ^ 3 

-- Probability of Team B winning 3:0
def prob_B_wins_3_0 (p_B : ℚ) := p_B 

-- Probability of Team B winning 3:1
def prob_B_wins_3_1 (p_A p_B : ℚ) := p_A * p_B 

-- Probability of Team B winning 3:2
def prob_B_wins_3_2 (p_A p_B : ℚ) := (p_A ^ 2) * p_B 

theorem problem_a (p_A p_B : ℚ)
  (h1 : p_A = 2/3)
  (h2 : p_B = 1/3) :
  prob_A_wins p_A = 8/27 :=
by 
  simp [prob_A_wins, h1]

theorem problem_b (p_A p_B : ℚ)
  (h1 : p_A = 2/3)
  (h2 : p_B = 1/3) :
  prob_B_wins_3_0 p_B = 1/3 :=
by 
  simp [prob_B_wins_3_0, h2]

theorem problem_c_incorrect (p_A p_B : ℚ)
  (h1 : p_A = 2/3)
  (h2 : p_B = 1/3) :
  prob_B_wins_3_1 p_A p_B ≠ 1/9 :=
by 
  simp [prob_B_wins_3_1, h1, h2, ne_of_lt (by norm_num : 2/9 > 1/9)]

theorem problem_c_correct (p_A p_B : ℚ)
  (h1 : p_A = 2/3)
  (h2 : p_B = 1/3) :
  prob_B_wins_3_1 p_A p_B = 2/9 :=
by 
  simp [prob_B_wins_3_1, h1, h2]

theorem problem_d_incorrect (p_A p_B : ℚ)
  (h1 : p_A = 2/3)
  (h2 : p_B = 1/3) :
  prob_B_wins_3_2 p_A p_B ≠ 4/9 :=
by 
  simp [prob_B_wins_3_2, h1, h2, ne_of_lt (by norm_num : 4/27 < 4/9)]

theorem problem_d_correct (p_A p_B : ℚ)
  (h1 : p_A = 2/3)
  (h2 : p_B = 1/3) :
  prob_B_wins_3_2 p_A p_B = 4/27 :=
by 
  simp [prob_B_wins_3_2, h1, h2]

end problem_a_problem_b_problem_c_incorrect_problem_c_correct_problem_d_incorrect_problem_d_correct_l591_591612


namespace ratio_alan_to_ben_l591_591944

theorem ratio_alan_to_ben (A B L : ℕ) (hA : A = 48) (hL : L = 36) (hB : B = L / 3) : A / B = 4 := by
  sorry

end ratio_alan_to_ben_l591_591944


namespace player1_always_wins_l591_591286

/-- 
Player 1 can always win the card game by following an optimal strategy.
There are 2002 cards on the table with numbers 1, 2, 3, ..., 2002.
Two players take turns picking one card at a time. 
After all the cards have been taken, the winner is the one 
whose last digit of the sum of the numbers on their taken cards is greater.
-/
theorem player1_always_wins :
  ∃ (strategy : ℕ → ℕ), ∀ (P1_cards P2_cards : list ℕ),
  let P1_sum := (P1_cards ++ [2]).map (λ n, n % 10) in
  let P2_sum := (P2_cards.map (λ n, n % 10)) in
  P1_sum.sum % 10 > P2_sum.sum % 10 :=
by {
  -- Proof goes here.
  sorry
}

end player1_always_wins_l591_591286


namespace number_of_cities_from_group_B_l591_591715

theorem number_of_cities_from_group_B
  (total_cities : ℕ)
  (cities_in_A : ℕ)
  (cities_in_B : ℕ)
  (cities_in_C : ℕ)
  (sampled_cities : ℕ)
  (h1 : total_cities = cities_in_A + cities_in_B + cities_in_C)
  (h2 : total_cities = 24)
  (h3 : cities_in_A = 4)
  (h4 : cities_in_B = 12)
  (h5 : cities_in_C = 8)
  (h6 : sampled_cities = 6) :
  cities_in_B * sampled_cities / total_cities = 3 := 
  by 
    sorry

end number_of_cities_from_group_B_l591_591715


namespace mens_wages_l591_591329

variable (M : ℕ) (wages_of_men : ℕ)

-- Conditions based on the problem
axiom eq1 : 15 * M = 90
axiom def_wages_of_men : wages_of_men = 5 * M

-- Prove that the total wages of the men are Rs. 30
theorem mens_wages : wages_of_men = 30 :=
by
  -- The proof would go here
  sorry

end mens_wages_l591_591329


namespace ratio_of_ages_l591_591588

theorem ratio_of_ages (S : ℝ) (R : ℝ) (h1 : S = 31.5) (h2 : S + 9 = R) :
  (S / R).nat_abs = (7 : ℝ / 9 : ℝ).nat_abs := by
screenshot
entials

end ratio_of_ages_l591_591588


namespace orchestra_members_l591_591274

theorem orchestra_members : ∃ (x : ℕ), (130 < x) ∧ (x < 260) ∧ (x % 6 = 1) ∧ (x % 5 = 2) ∧ (x % 7 = 3) ∧ (x = 241) :=
by
  sorry

end orchestra_members_l591_591274


namespace part1_part2_part3_l591_591821

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1)*x^2 - (m - 1)*x + (m - 1)

theorem part1 (m : ℝ) : (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 := 
sorry

theorem part2 (m : ℝ) (x : ℝ) : (f m x ≥ (m + 1) * x) ↔ 
  (m = -1 ∧ x ≥ 1) ∨ 
  (m > -1 ∧ (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1)) ∨ 
  (m < -1 ∧ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) := 
sorry

theorem part3 (m : ℝ) : (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) ↔
  m ≥ 1 := 
sorry

end part1_part2_part3_l591_591821


namespace third_row_number_l591_591087

-- Define the conditions to fill the grid
def grid (n : Nat) := Fin 4 → Fin 4 → Fin n

-- Ensure each number 1-4 in each cell such that numbers do not repeat
def unique_in_row (g : grid 4) : Prop :=
  ∀ i j1 j2, j1 ≠ j2 → g i j1 ≠ g i j2

def unique_in_col (g : grid 4) : Prop :=
  ∀ j i1 i2, i1 ≠ i2 → g i1 j ≠ g i1 j

-- Define the external hints condition, encapsulating the provided hints.
def hints_condition (g : grid 4) : Prop :=
  -- Example placeholders for hint conditions that would be expanded accordingly.
  g 0 0 = 3 ∨ g 0 1 = 2 -- First row hints interpreted constraints
  -- Additional hint conditions to be added accordingly

-- Prove the correct number formed by the numbers in the third row is 4213
theorem third_row_number (g : grid 4) :
  unique_in_row g ∧ unique_in_col g ∧ hints_condition g →
  (g 2 0 = 4 ∧ g 2 1 = 2 ∧ g 2 2 = 1 ∧ g 2 3 = 3) :=
by
  sorry

end third_row_number_l591_591087


namespace divisibility_2_pow_a_plus_1_l591_591362

theorem divisibility_2_pow_a_plus_1 (a b : ℕ) (h_b_pos : 0 < b) (h_b_ge_2 : 2 ≤ b) 
  (h_div : (2^a + 1) % (2^b - 1) = 0) : b = 2 := by
  sorry

end divisibility_2_pow_a_plus_1_l591_591362


namespace maximum_colored_squares_l591_591377

theorem maximum_colored_squares (grid : fin 10 × fin 10 → bool) :
  (∀ c1 c2 : fin 10 × fin 10, c1 ≠ c2 → 
    grid c1 → grid c2 → ¬ (c1.1 ≤ c2.1 ∧ c1.2 ≤ c2.2)) →
  ∃ (n : ℕ), n = 15 ∧ (∀ other_n : ℕ, other_n > n → 
    ¬ ∃ color_scheme : fin other_n → fin 10 × fin 10, (∀ i, grid (color_scheme i) = true))

end maximum_colored_squares_l591_591377


namespace uniform_probabilities_l591_591343

-- Define the uniform distribution density function
def uniform_density (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a ∨ x ≥ b then 0 else 1 / (b - a)

-- Define the cumulative distribution function based on the density function
def uniform_cdf (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then 0 else if x < b then (x - a) / (b - a) else 1

-- Prove the probability calculations for the uniform distribution over (-4, 6)
theorem uniform_probabilities :
  let a := -4
  let b := 6
  ∀ X : ℝ → ℝ, (uniform_density a b = X) →
    (P (λ x, 0 ≤ x ∧ x ≤ 1) = 0.1)
    ∧ (P (λ x, 5 ≤ x ∧ x ≤ 10) = 0.1) :=
by {
  -- Definitions
  let a := -4,
  let b := 6,
  
  -- Density function definition
  let f := uniform_density a b,
  
  -- CDF function definition
  let F := uniform_cdf a b,
 
  -- Probability calculations
  let P0_1 := F 1 - F 0,
  let P5_10 := F 10 - F 5,
  
  -- Assertions
  have hP0_1 : P0_1 = 0.1 := sorry,
  have hP5_10 : P5_10 = 0.1 := sorry,
  exact ⟨hP0_1, hP5_10⟩
}

end uniform_probabilities_l591_591343


namespace minimum_sum_distances_parabola_l591_591138

theorem minimum_sum_distances_parabola 
  (P : ℝ × ℝ) 
  (on_parabola : P.2^2 = 4 * P.1) 
  (l1 : ∀ (x y : ℝ), 3 * x - 4 * y + 12 = 0) 
  (l2 : ∀ (x : ℝ), x + 2 = 0) :
  ∀ (P : ℝ × ℝ), (on_parabola P) → 
    (3 * P.1 - 4 * P.2 + 12).abs / (real.sqrt (3^2 + 4^2)) + 
    (P.1 + 2).abs = 4 :=
by
  sorry

end minimum_sum_distances_parabola_l591_591138


namespace determine_A_l591_591488

theorem determine_A (x y A : ℝ) 
  (h : (x + y) ^ 3 - x * y * (x + y) = (x + y) * A) : 
  A = x^2 + x * y + y^2 := 
by
  sorry

end determine_A_l591_591488


namespace simplify_complex_fractions_l591_591596

theorem simplify_complex_fractions :
  let a := 4
  let b := 7
  let i := Complex.I
  (a + b * i) / (a - b * i) + (a - b * i) / (a + b * i) = -66/65 :=
by
  let a := 4
  let b := 7
  let i := Complex.I
  have h1 : (a + b * i) / (a - b * i) = (Complex (a + b * i) : ℂ) / Complex (a - b * i),
  from rfl,
  have h2 : (a - b * i) / (a + b * i) = (Complex (a - b * i) : ℂ) / Complex (a + b * i),
  from rfl,
  rw [h1, h2],
  -- blocks for the calculation steps
  sorry

end simplify_complex_fractions_l591_591596


namespace product_of_distances_l591_591794

-- Definitions based on the conditions
def curve (x y : ℝ) : Prop := x * y = 2

-- The theorem to prove
theorem product_of_distances (x y : ℝ) (h : curve x y) : abs x * abs y = 2 := by
  -- This is where the proof would go
  sorry

end product_of_distances_l591_591794


namespace find_a_to_make_f_odd_l591_591152

noncomputable def f (a : ℝ) (x : ℝ): ℝ := x^3 * (Real.log (Real.exp x + 1) + a * x)

theorem find_a_to_make_f_odd :
  (∃ a : ℝ, ∀ x : ℝ, f a (-x) = -f a x) ↔ a = -1/2 :=
by 
  sorry

end find_a_to_make_f_odd_l591_591152


namespace length_of_arc_CK_l591_591644

-- Definitions for Lean
variables (A B C K : Type) [affirmative_on_sphere : ∀ (p : Type), Type]
-- Radius of the sphere
variable (R : ℝ)
-- Arc length of BC < πR
variable (l : ℝ) (h_l : l < π * R)
-- Midpoints and great circles assumptions
variable (midpoints_great_circle : ∀ (mid_AB mid_AC : Type) (h1 : Type), 
  intersect (extension BC) K)

theorem length_of_arc_CK : 
  ∃ (CK : ℝ), (CK = (π * R / 2 + l / 2) ∨ CK = (π * R / 2 - l / 2)) :=
sorry

end length_of_arc_CK_l591_591644


namespace correct_operation_l591_591673

theorem correct_operation :
  let A := 3 * x^3 + 2 * x^3 = 5 * x^6
  let B := (x + 1)^2 = x^2 + 1
  let C := x^8 / x^4 = x^2
  let D := sqrt 4 = 2
  D :=
begin
  -- Proof of the correct operation being D goes here
  sorry
end

end correct_operation_l591_591673


namespace domain_of_v_l591_591660

noncomputable def v (x : ℝ) : ℝ := 1 / (x - 1)^(1 / 3)

theorem domain_of_v :
  {x : ℝ | ∃ y : ℝ, y ≠ 0 ∧ y = (v x)} = {x | x ≠ 1} := by
  sorry

end domain_of_v_l591_591660


namespace conjugate_of_conjugate_z_l591_591865

variable {z : ℂ}
variable (h : (conj z) * complex.I = 1 + complex.I)

theorem conjugate_of_conjugate_z : complex.conj (conj z) = 1 + complex.I :=
  sorry

end conjugate_of_conjugate_z_l591_591865


namespace other_number_more_than_42_l591_591632

theorem other_number_more_than_42 (a b : ℕ) (h1 : a + b = 96) (h2 : a = 42) : b - a = 12 := by
  sorry

end other_number_more_than_42_l591_591632


namespace hexagon_points_distance_l591_591524

-- Definitions for the conditions given in the problem
def hexagon (s : ℝ) : Type :=
{x : ℝ × ℝ // ∃ (n : fin 6), ∥x - (s * (cos (2 * n * π / 6)), s * (sin (2 * n * π / 6)))∥ ≤ s}

def points_in_hexagon (pts : set (ℝ × ℝ)) (s : ℝ) : Prop :=
∀ p ∈ pts, ∃ h : hexagon s, p = h.val

-- Definition for the side length of the hexagon and the points within it
def side_length : ℝ := 1
def num_points : ℕ := 7

-- The set of points in the hexagon
noncomputable def point_set : set (ℝ × ℝ) := {p : ℝ × ℝ | true} -- placeholder for any 7 points

theorem hexagon_points_distance :
  ∃ p1 p2 ∈ point_set, p1 ≠ p2 ∧ dist p1 p2 ≤ side_length :=
by sorry

end hexagon_points_distance_l591_591524


namespace shaded_area_correct_l591_591521

-- Defining the given conditions
def rA : ℝ := 2  -- Radius of arc AXB
def rB : ℝ := 3  -- Radius of arc BYC
def rC : ℝ := 2  -- Radius of arc XYZ

-- Calculating the area of the shaded regions
def area_sector (r : ℝ) (θ : ℝ) : ℝ :=
  (1/2) * r^2 * θ

def area_shaded : ℝ := area_sector rA real.pi + area_sector rB real.pi

-- Proving the correct answer
theorem shaded_area_correct :
  area_shaded = (13/2) * real.pi :=
by
  -- The proof is omitted
  sorry

end shaded_area_correct_l591_591521


namespace trapezoid_to_square_l591_591382

theorem trapezoid_to_square (T : Trapezoid) (h_area : T.area = 5) :
  ∃ (parts : List Shape), (Dissect T parts) ∧ (Square_formed_from parts) :=
sorry

end trapezoid_to_square_l591_591382


namespace max_number_of_children_l591_591984

theorem max_number_of_children (apples cookies chocolates : ℕ) (remaining_apples remaining_cookies remaining_chocolates : ℕ) 
  (h₁ : apples = 55) 
  (h₂ : cookies = 114) 
  (h₃ : chocolates = 83) 
  (h₄ : remaining_apples = 3) 
  (h₅ : remaining_cookies = 10) 
  (h₆ : remaining_chocolates = 5) : 
  gcd (apples - remaining_apples) (gcd (cookies - remaining_cookies) (chocolates - remaining_chocolates)) = 26 :=
by
  sorry

end max_number_of_children_l591_591984


namespace non_neg_int_solutions_count_l591_591273

theorem non_neg_int_solutions_count :
  {x : ℤ // (3 - 2 * x > 0) ∧ (2 * x - 7 ≤ 4 * x + 7) ∧ (x ≥ 0)}.to_finset.card = 2 := by
  sorry

end non_neg_int_solutions_count_l591_591273


namespace lower_bound_m_l591_591698

theorem lower_bound_m (h : ℝ) (m : ℝ) (H : 0 < h) :
  (∀ h : ℝ, 0 < h →  m = 2 / h + 2 + 2 * Real.sqrt (1 + h^2 / 4)) →
  filter.Tendsto (λ h, 2 / h + 2 + 2 * Real.sqrt (1 + h^2 / 4)) filter.at_top (nhds 3) :=
by
  sorry

end lower_bound_m_l591_591698


namespace new_cost_of_article_l591_591049

theorem new_cost_of_article (C : ℝ) (hC : C = 1078.95) (h_decrease : true) : 
  let C_new := C * (76 / 100) in
  C_new = 819.562 :=
by 
  sorry

end new_cost_of_article_l591_591049


namespace total_dog_weight_l591_591086

theorem total_dog_weight (weight_evans_dog weight_ivans_dog : ℕ)
  (h₁ : weight_evans_dog = 63)
  (h₂ : weight_evans_dog = 7 * weight_ivans_dog) :
  weight_evans_dog + weight_ivans_dog = 72 :=
sorry

end total_dog_weight_l591_591086


namespace ellipse_problem_l591_591208

-- Definitions of conditions from the problem
def F1 := (0, 0)
def F2 := (6, 0)
def ellipse_equation (x y h k a b : ℝ) := ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

-- The main statement to be proved
theorem ellipse_problem :
  let h := 3
  let k := 0
  let a := 5
  let c := 3
  let b := Real.sqrt (a^2 - c^2)
  h + k + a + b = 12 :=
by
  -- Proof would go here
  sorry

end ellipse_problem_l591_591208


namespace ratio_of_areas_l591_591517

theorem ratio_of_areas 
  (A B C D E F : Type)
  (AB AC AD : ℝ)
  (h1 : AB = 130)
  (h2 : AC = 130)
  (h3 : AD = 26)
  (CF : ℝ)
  (h4 : CF = 91)
  (BD : ℝ)
  (h5 : BD = 104)
  (AF : ℝ)
  (h6 : AF = 221)
  (EF DE BE CE : ℝ)
  (h7 : EF / DE = 91 / 104)
  (h8 : CE / BE = 3.5) :
  EF * CE = 318.5 * DE * BE :=
sorry

end ratio_of_areas_l591_591517


namespace num_consolidated_functions_eq_nine_l591_591115

def is_valid_domain (f : ℝ → ℝ) (domain : set ℝ) (range : set ℝ) : Prop :=
  ∀ y ∈ range, ∃ x ∈ domain, f x = y

def count_consolidated_functions (f : ℝ → ℝ) (range : set ℝ) : ℕ :=
  {domain : set ℝ | is_valid_domain f domain range}.to_finset.card

def f (x : ℝ) : ℝ := 2 * x ^ 2 - 1

theorem num_consolidated_functions_eq_nine :
  count_consolidated_functions f {1, 7} = 9 := sorry

end num_consolidated_functions_eq_nine_l591_591115


namespace line_integral_value_l591_591749

noncomputable def helical_line_integral : ℝ :=
  let x := λ t : ℝ, Real.cos t
  let y := λ t : ℝ, Real.sin t
  let z := λ t : ℝ, t
  let integrand := λ t : ℝ, (z t)^2 / ((x t)^2 + (y t)^2)
  let dl := λ t : ℝ, Real.sqrt(2) * t
  ∫ t in set.Icc 0 (2 * Real.pi), integrand t * dl t

theorem line_integral_value :
  helical_line_integral = 8 * Real.sqrt(2) * (Real.pi^3) / 3 :=
sorry

end line_integral_value_l591_591749


namespace find_line_l591_591834

variable {A B C1 C2 : ℝ}

-- Define the lines l1 and l2
def l1 (x y : ℝ) := A * x + B * y + C1 = 0
def l2 (x y : ℝ) := A * x + B * y + C2 = 0

-- Given conditions
variable (h1 : A - B + C1 + C2 = 0)
variable (h2 : C1 ≠ C2)
variable (H : (x, y) -> l1 x y ∧ l2 x y)

theorem find_line (hH : l1 (-1) 1) (hx1 : A - B + x - y - 1 = 0) : 
  (∀ x y, x + y ≠ 0) :=
by 
  sorry

end find_line_l591_591834


namespace part1_part2_l591_591494

-- Define the main problem condition
def satisfies_condition (z : ℂ) : Prop := z = complex.I * (2 - z)

-- Part (1): Prove the solution for \( z \)
theorem part1 : ∃ z : ℂ, satisfies_condition z ∧ z = 1 + complex.I := by
  sorry

-- Part (2): Given \( z = 1 + i \), prove the required modulus difference
theorem part2 (z : ℂ) (h : satisfies_condition z) (hz : z = 1 + complex.I) : abs (z - (2 - complex.I)) = real.sqrt 5 := by
  sorry

end part1_part2_l591_591494


namespace quadratic_inequality_solution_set_l591_591156

variable (a b c : ℝ)

theorem quadratic_inequality_solution_set (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 → (-1 / 3 < x ∧ x < 2)) :
  ∀ x : ℝ, cx^2 + bx + a < 0 → (-3 < x ∧ x < 1 / 2) :=
by
  sorry

end quadratic_inequality_solution_set_l591_591156


namespace cot_sum_arccot_roots_eq_l591_591929

-- Define the five roots of the polynomial
def z1 := sorry -- root 1 of z^5 - 3z^4 + 6z^3 - 10z^2 + 15z - 21 = 0
def z2 := sorry -- root 2 of z^5 - 3z^4 + 6z^3 - 10z^2 + 15z - 21 = 0
def z3 := sorry -- root 3 of z^5 - 3z^4 + 6z^3 - 10z^2 + 15z - 21 = 0
def z4 := sorry -- root 4 of z^5 - 3z^4 + 6z^3 - 10z^2 + 15z - 21 = 0
def z5 := sorry -- root 5 of z^5 - 3z^4 + 6z^3 - 10z^2 + 15z - 21 = 0

noncomputable def cot_sum_arccot_roots : ℝ :=
  Real.cot (∑ k in [1, 2, 3, 4, 5], Real.arccot (List.nthLe [z1, z2, z3, z4, z5] (k-1) sorry))

theorem cot_sum_arccot_roots_eq : cot_sum_arccot_roots = 14/9 := 
by sorry

end cot_sum_arccot_roots_eq_l591_591929


namespace count_three_digit_integers_divisible_by_11_and_5_l591_591481

def count_three_digit_multiples (a b: ℕ) : ℕ :=
  let lcm := Nat.lcm a b
  let first_multiple := (100 + lcm - 1) / lcm
  let last_multiple := 999 / lcm
  last_multiple - first_multiple + 1

theorem count_three_digit_integers_divisible_by_11_and_5 : 
  count_three_digit_multiples 11 5 = 17 := by 
  sorry

end count_three_digit_integers_divisible_by_11_and_5_l591_591481


namespace frog_jump_distance_l591_591231

-- Definitions based on problem conditions
def P : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (a_1, a_2)
def B : ℝ × ℝ := (b_1, b_2)
def C : ℝ × ℝ := (c_1, c_2)
def distance (p1 p2 : ℝ × ℝ) : ℝ := (real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))

-- Given initial distance between point P and C
axiom P_to_C_dist : distance P C = 0.27

-- Main theorem to prove: Distance between P and P2009
theorem frog_jump_distance : distance P P_2009 = 0.54 :=
begin
  sorry -- The proof placeholder
end

end frog_jump_distance_l591_591231


namespace f_min_value_g_greater_than_neg_20_l591_591828

noncomputable def f (x : ℝ) : ℝ := x - Real.log x
noncomputable def g (x : ℝ) : ℝ := x^3 + x^2 * f(x) - 16 * x

theorem f_min_value : ∀ x > 0, f(x) >= 1 := by
  sorry

theorem g_greater_than_neg_20 : ∀ x > 0, g(x) > -20 := by
  sorry

end f_min_value_g_greater_than_neg_20_l591_591828


namespace sum_of_solutions_eq_eight_l591_591419

theorem sum_of_solutions_eq_eight (x : ℂ) (h : x + (25 / x) = 8) :
  (∃ y, x + y = 8 ∧ ( y = 4 + 3 * complex.I ∨ y = 4 - 3 * complex.I)) :=
sorry

end sum_of_solutions_eq_eight_l591_591419


namespace compute_k_l591_591218

theorem compute_k (a b : ℝ) (x k : ℝ)
  (h1 : tan x = a / b) 
  (h2 : tan (2 * x) = b / (2 * a + b)) 
  (h3 : x = arctan k) : k = 1 / 4 :=
sorry

end compute_k_l591_591218


namespace exists_prime_gt_2019_div_property_l591_591239

theorem exists_prime_gt_2019_div_property :
  ∃ p : ℕ, prime p ∧ 2019 < p ∧ ∀ α : ℕ, α > 0 → ∃ n : ℕ, n > 0 ∧ p^α ∣ (2019^n - n^2017) := 
sorry

end exists_prime_gt_2019_div_property_l591_591239


namespace distance_le_one_l591_591280

def point := ℝ × ℝ × ℝ

def fixed_point : point := (1, 0, 0)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def point_set (d : ℝ) : set point :=
  {p | distance p fixed_point ≤ d}

theorem distance_le_one :
  point_set 1 = {p | (p.1 - 1)^2 + p.2^2 + p.3^2 ≤ 1} :=
sorry

end distance_le_one_l591_591280


namespace total_questions_asked_l591_591511

theorem total_questions_asked (drew_correct : ℕ := 20) (drew_wrong : ℕ := 6) (carla_correct : ℕ := 14):
  let carla_wrong := 2 * drew_wrong in
  let drew_total := drew_correct + drew_wrong in
  let carla_total := carla_correct + carla_wrong in
  (drew_total + carla_total) = 52 := by
  sorry

end total_questions_asked_l591_591511


namespace triangle_area_proof_l591_591447

-- Define the parameters and the given conditions
variables {α : ℝ} {a d : ℝ}

-- Assume the conditions given in the problem
def is_triangle_conditions (b c : ℝ) : Prop :=
  b - c = d

noncomputable def triangle_area (α a d : ℝ) :=
  (a^2 - d^2) / 4 * Real.cot (α / 2)

-- State the theorem we need to prove
theorem triangle_area_proof :
  ∃ (b c : ℝ), is_triangle_conditions b c → 
  ∃ t : ℝ, t = triangle_area α a d :=
by {
  sorry
}

end triangle_area_proof_l591_591447


namespace concyclic_points_l591_591970

-- Definitions for the circles, points, and tangents
variables {P : Type*} [MetricSpace P]
(C : P) (A B C D E F G : P) 
(zeta1 zeta2 : Set P)

-- Conditions as definitions in Lean 4
def circle_touch_at_A := (A ∈ zeta1) ∧ (A ∈ zeta2) ∧ metric_space.distance (center_of zeta1) (center_of zeta2) = (radius_of zeta1 + radius_of zeta2)
def line_through_A := B ∈ line_through A ∧ C ∈ line_through A
def tangent_at_B_intersects_zeta2_at_D_E := (D ∈ tangents zeta1 B) ∧ (E ∈ tangents zeta1 B) ∧ (D ∈ zeta2) ∧ (E ∈ zeta2)
def tangents_at_C_touch_zeta2_at_F_G := (F ∈ tangents C zeta1) ∧ (G ∈ tangents C zeta1) ∧ (F ∈ zeta2) ∧ (G ∈ zeta2)

-- Final proof statement
theorem concyclic_points : 
  circle_touch_at_A ∧ line_through_A ∧ tangent_at_B_intersects_zeta2_at_D_E ∧ tangents_at_C_touch_zeta2_at_F_G 
  → concyclic {D, E, F, G} :=
by 
  sorry

end concyclic_points_l591_591970


namespace beads_division_l591_591048

theorem beads_division :
  ∃ n : ℕ, let total_beads := 23 + 16 in
           let initial_beads_per_part := total_beads / n in
           let beads_after_removal := initial_beads_per_part - 10 in
           let final_beads_per_part := 2 * beads_after_removal in
           final_beads_per_part = 6 ∧ n = 3 :=
by {
  sorry
}

end beads_division_l591_591048


namespace coeff_x5_in_expansion_l591_591091

-- Define the problem statement as a theorem in Lean 4
theorem coeff_x5_in_expansion :
  (Nat.choose 8 4) - (Nat.choose 8 5) = 14 :=
sorry

end coeff_x5_in_expansion_l591_591091


namespace min_value_fraction_l591_591095

theorem min_value_fraction (x : ℝ) (h : x > 9) : (x^2 + 81) / (x - 9) ≥ 27 := 
  sorry

end min_value_fraction_l591_591095


namespace determine_top3_by_median_l591_591874

theorem determine_top3_by_median (scores : Fin 7 → ℝ) (h_distinct : ∀ i j, i ≠ j → scores i ≠ scores j) (own_score : ℝ) :
  (∃ i, scores i = own_score) →
  (∃ i, own_score < scores i) →
  (∃ i₀ i₁ i₂ i₃ i₄ i₅ i₆, 
    (List.sort (≤) [scores i₀, scores i₁, scores i₂, scores i₃, scores i₄, scores i₅, scores i₆] = [scores i₀, scores i₁, scores i₂, scores i₃, scores i₄, scores i₅, scores i₆]) ∧ 
    own_score > List.get (List.sort (≤) [scores i₀, scores i₁, scores i₂, scores i₃, scores i₄, scores i₅, scores i₆]) ⟨3, ⟨dec_trivial⟩⟩) ↔ 
  (∃ i₇ i₈ i₉, own_score ∈ [scores i₇, scores i₈, scores i₉]) :=
by
  sorry

end determine_top3_by_median_l591_591874


namespace simplify_complex_expr_l591_591599

theorem simplify_complex_expr (a b : ℂ) (hz : b = 7 * complex.I) (ha : a = 4) :
  (a + b) / (a - b) + (a - b) / (a + b) = -66 / 65 := sorry

end simplify_complex_expr_l591_591599


namespace nth_term_sequence_l591_591185

-- Definition of the sequence
def sequence (n : ℕ) : ℚ := 2^n / (2^n + 3)

-- The conjecture we need to prove
theorem nth_term_sequence (n : ℕ) : sequence n = 2^n / (2^n + 3) :=
by simp [sequence]

end nth_term_sequence_l591_591185


namespace min_pos_diff_composites_sum_96_l591_591693

-- Define what it means to be a composite number
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ p1 p2 : ℕ, p1 > 1 ∧ p2 > 1 ∧ p1 * p2 = n

-- Theorem statement: prove the minimum positive difference between two composite numbers that sum to 96 is 2
theorem min_pos_diff_composites_sum_96 : 
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ a + b = 96 ∧ (a ≠ b) ∧ (∀ (x y : ℕ), is_composite x → is_composite y → x + y = 96 → x ≠ y → nat.abs (x - y) ≥ 2) :=
begin
  -- Proof would go here
  sorry
end

end min_pos_diff_composites_sum_96_l591_591693


namespace continuity_at_2_l591_591320

theorem continuity_at_2 (f : ℝ → ℝ) (x0 : ℝ) (hf : ∀ x, f x = -4 * x ^ 2 - 8) :
  x0 = 2 → ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x0| < δ → |f x + 24| < ε := by
  sorry

end continuity_at_2_l591_591320


namespace simplify_complex_fractions_l591_591597

theorem simplify_complex_fractions :
  let a := 4
  let b := 7
  let i := Complex.I
  (a + b * i) / (a - b * i) + (a - b * i) / (a + b * i) = -66/65 :=
by
  let a := 4
  let b := 7
  let i := Complex.I
  have h1 : (a + b * i) / (a - b * i) = (Complex (a + b * i) : ℂ) / Complex (a - b * i),
  from rfl,
  have h2 : (a - b * i) / (a + b * i) = (Complex (a - b * i) : ℂ) / Complex (a + b * i),
  from rfl,
  rw [h1, h2],
  -- blocks for the calculation steps
  sorry

end simplify_complex_fractions_l591_591597


namespace greatest_multiple_of_4_l591_591254

theorem greatest_multiple_of_4 (x : ℕ) (h₀ : 0 < x) (h₁ : x % 4 = 0) (h₂ : x^3 < 5000) : x ≤ 16 :=
by {
  -- Skipping the proof as per instructions.
  sorry,
}

example : ∃ x : ℕ, 0 < x ∧ x % 4 = 0 ∧ x^3 < 5000 ∧ x = 16 :=
by {
  use 16,
  split,
  -- Proof steps for 0 < 16
  exact nat.zero_lt_bit0 nat.zero_lt_bit0,
  split,
  -- Proof steps for 16 % 4 = 0
  norm_num,
  split,
  -- Proof steps for 16^3 < 5000
  norm_num,
  -- Proof that x = 16
  refl,
}

end greatest_multiple_of_4_l591_591254


namespace geometric_series_common_ratio_l591_591734

theorem geometric_series_common_ratio (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) 
  (h₃ : S = a / (1 - r)) : r = 21 / 25 := 
sorry

end geometric_series_common_ratio_l591_591734


namespace correct_propositions_l591_591786

variables {Plane Line : Type}
variables (alpha beta : Plane) 
variables (l m : Line)

-- Given conditions
def plane_distinct (α β : Plane) := α ≠ β
def line_distinct (l m : Line) := l ≠ m
def line_perpendicular_plane (l : Line) (α : Plane) := perp l α
def line_in_plane (m : Line) (β : Plane) := m ⊆ β

-- Propositions
def proposition_1 (α β : Plane) (l m : Line) := parallel α β → perp l m
def proposition_2 (α β : Plane) (l m : Line) := perp α β → parallel l m
def proposition_3 (α : Plane) (m l : Line) := parallel m α → perp l β
def proposition_4 (β : Plane) (l m : Line) := perp l β → parallel m α

-- The proof problem
theorem correct_propositions (α β : Plane) (l m : Line) 
  (hαβ : plane_distinct α β) (hlm : line_distinct l m)
  (hlα : line_perpendicular_plane l α) (hmβ : line_in_plane m β) :
  (proposition_1 α β l m) ∧ ¬(proposition_2 α β l m) ∧ ¬(proposition_3 α m l) ∧ (proposition_4 β l m) :=
  by sorry

end correct_propositions_l591_591786


namespace minimum_value_of_fraction_l591_591433

theorem minimum_value_of_fraction {x : ℝ} (hx : x ≥ 3/2) :
  ∃ y : ℝ, y = (2 * (x - 1) + (1 / (x - 1)) + 2) ∧ y = 2 * Real.sqrt 2 + 2 :=
sorry

end minimum_value_of_fraction_l591_591433


namespace calculate_capacity_l591_591727

noncomputable def tank_capacity : ℝ :=
  let x := 120 in 
  let thirty_percent := 0.3 * x in
  let additional_liters := 36 in
  thirty_percent = additional_liters → x

theorem calculate_capacity : tank_capacity = 120 := by
  sorry

end calculate_capacity_l591_591727


namespace product_of_roots_eq_one_sixth_l591_591778

noncomputable def lg (x : ℝ) : ℝ := real.log10 x

theorem product_of_roots_eq_one_sixth
    (x1 x2 : ℝ)
    (h : (lg x1)^2 + (lg 2 + lg 3) * (lg x1) + (lg 2) * (lg 3) = 0 ∧
         (lg x2)^2 + (lg 2 + lg 3) * (lg x2) + (lg 2) * (lg 3) = 0):
  x1 * x2 = 1 / 6 :=
sorry

end product_of_roots_eq_one_sixth_l591_591778


namespace max_min_m_l591_591445

variable {x y z : ℝ}

theorem max_min_m (h : x^2 + y^2 + z^2 = 1) : ∃ m_max m_min, (m = xy + yz + zx) ∧ m_min ≤ xy + yz + zx ∧ xy + yz + zx ≤ m_max := 
sorry

end max_min_m_l591_591445


namespace cost_per_ball_is_two_l591_591563

def cost_per_tennis_ball (packs : ℕ) (balls_per_pack : ℕ) (total_cost : ℕ) : ℝ :=
  total_cost / (packs * balls_per_pack)

theorem cost_per_ball_is_two :
  (cost_per_tennis_ball 4 3 24) = 2 :=
by
  sorry

end cost_per_ball_is_two_l591_591563


namespace max_s_square_in_circle_l591_591324

theorem max_s_square_in_circle (r : ℝ) (C : ℝ × ℝ) (hC : C ≠ (r, 0) ∧ C ≠ (-r, 0))
  (hr : ∀ P : ℝ × ℝ, (P.1 - r) ^ 2 + P.2 ^ 2 = r^2) :
  let s := (√((C.1 - r) ^ 2 + C.2 ^ 2) + √((C.1 + r) ^ 2 + C.2 ^ 2))
  in s^2 ≤ 8 * r^2 :=
by
  -- proof
  sorry

end max_s_square_in_circle_l591_591324


namespace integer_solutions_k_l591_591769

theorem integer_solutions_k (k n m : ℤ) (h1 : k + 1 = n^2) (h2 : 16 * k + 1 = m^2) :
  k = 0 ∨ k = 3 :=
by sorry

end integer_solutions_k_l591_591769


namespace dave_initial_apps_l591_591383

theorem dave_initial_apps (x : ℕ) (h1 : x - 18 = 5) : x = 23 :=
by {
  -- This is where the proof would go 
  sorry -- The proof is omitted as per instructions
}

end dave_initial_apps_l591_591383


namespace angle_between_AC1_and_lateral_face_l591_591038

noncomputable def angle_between_vectors := sorry

theorem angle_between_AC1_and_lateral_face
  (a : ℝ)
  (prism : ∀ v, v ∈ ({ | A(0,0,0), B (a / 2, -√3 / 2 * a, 0), A1 (0, 0, √2 * a), C1 (a, 0, √2 * a) })) 
  (AC1_vector : ℝ × ℝ × ℝ := (a, 0, √2 * a))   
  (AD_vector : ℝ × ℝ × ℝ := (a / 4, -√3 / 4 * a, √2 * a)):
  angle_between_vectors AC1_vector AD_vector = 30 :=
sorry

end angle_between_AC1_and_lateral_face_l591_591038


namespace common_tangents_between_circles_l591_591988

theorem common_tangents_between_circles :
  let M := (1:ℝ, -2:ℝ),
      N := (2:ℝ, 2:ℝ),
      r1 := 1,
      r2 := 3,
      d := real.sqrt ((2 - 1)^2 + (2 + 2)^2) in
  d > r1 + r2 → 4 :=
by
  let M := (1, -2)
  let N := (2, 2)
  let r1 := 1
  let r2 := 3
  let d := real.sqrt ((2 - 1)^2 + (2 + 2)^2)
  -- Conditions and calculations not shown in theorem statement for brevity
  sorry

end common_tangents_between_circles_l591_591988


namespace cost_of_bike_l591_591310

noncomputable def weekly_allowance := 5
noncomputable def lawn_payment := 10
noncomputable def babysitting_rate := 7
noncomputable def initial_savings := 65
noncomputable def babysitting_hours := 2
noncomputable def additional_amount_needed := 6

theorem cost_of_bike : 
  let total_earnings := weekly_allowance + lawn_payment + babysitting_rate * babysitting_hours,
      total_savings := initial_savings + total_earnings in
  total_savings + additional_amount_needed = 100 :=
by
  sorry

end cost_of_bike_l591_591310


namespace polynomials_7_times_poly_condition_l591_591385

-- Define the concept of "7 times coefficient polynomial"
def is_7_times_coefficient_polynomial (p : Polynomial ℤ) : Prop :=
  (p.coeffs.sum % 7 = 0)

-- Given polynomials
def poly1 := Polynomial.C 2 * X^2 - Polynomial.C 9 * X
def poly2 := Polynomial.C 3 * Polynomial.a + Polynomial.C 5 * Polynomial.b
def poly3 := Polynomial.C 19 * X^2 - Polynomial.C 4 * X + Polynomial.C 2 * Y - Polynomial.C 3 * X * Y

-- Lean Proof Problem Statements
theorem polynomials_7_times (p1 p2 p3 : Polynomial ℤ) : 
  is_7_times_coefficient_polynomial p1 ∧ ¬ is_7_times_coefficient_polynomial p2 ∧ is_7_times_coefficient_polynomial p3 := by
  sorry

variables (m n : ℤ)

-- Given condition for part 2
axiom h : is_7_times_coefficient_polynomial (4.X * m - Y * n)

-- Part 2 Proof Statement
theorem poly_condition (h : is_7_times_coefficient_polynomial (Polynomial.C (4 * m) * X - Polynomial.C n * Y)) :
  is_7_times_coefficient_polynomial (Polynomial.C (2 * m) * X + Polynomial.C (3 * n) * Y) := by
  sorry

end polynomials_7_times_poly_condition_l591_591385


namespace simplify_complex_expr_l591_591601

theorem simplify_complex_expr (a b : ℂ) (hz : b = 7 * complex.I) (ha : a = 4) :
  (a + b) / (a - b) + (a - b) / (a + b) = -66 / 65 := sorry

end simplify_complex_expr_l591_591601


namespace calc_2n_minus_2np1_l591_591365

theorem calc_2n_minus_2np1 (n : ℤ) : 2^n - 2^(n+1) = -2^n := by
  sorry

end calc_2n_minus_2np1_l591_591365


namespace camp_more_than_home_l591_591764

theorem camp_more_than_home (camp home diff: ℕ) (h_camp: camp = 819058) (h_home: home = 668278) (h_diff: diff = 150780) : camp - home = diff :=
by
  { rw [h_camp, h_home, h_diff],
    exact Nat.sub_self _ }

end camp_more_than_home_l591_591764


namespace tangent_line_at_origin_is_y_eq_0_local_minimum_at_zero_number_of_zeros_l591_591463

-- The function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log(x + 1.0) - a * x^2

-- (Ⅰ) Prove the equation of the tangent line at (0, f(0)) is y = 0
theorem tangent_line_at_origin_is_y_eq_0 {a : ℝ} : 
  (deriv (λ x => f x a) 0) = 0 :=
by
  sorry

-- (Ⅱ) Prove that f(x) has a local minimum at x = 0 when a < 0
theorem local_minimum_at_zero {a : ℝ} (h : a < 0) :
  ∀ x > -1, deriv (λ x => deriv (λ x => f x a) x) x > 0 :=
by
  sorry

-- (Ⅲ) Determine the number of zeros of the function f(x) given conditions on a
theorem number_of_zeros {a : ℝ} : 
  ((a ≤ 0) ∨ (a = 1) → ∃! x, f x a = 0) ∧ 
  ((a > 0) ∧ (a ≠ 1) → (∃ x1 x2, f x1 a = 0 ∧ f x2 a = 0 ∧ (x1 ≠ x2))) :=
by
  sorry

end tangent_line_at_origin_is_y_eq_0_local_minimum_at_zero_number_of_zeros_l591_591463


namespace bananas_more_than_pears_l591_591967

theorem bananas_more_than_pears (A P B : ℕ) (h1 : P = A + 2) (h2 : A + P + B = 19) (h3 : B = 9) : B - P = 3 :=
  by
  sorry

end bananas_more_than_pears_l591_591967


namespace proof1_proof2_proof3_proof4_l591_591751

-- Define variables.
variable (m n x y z : ℝ)

-- Prove the expressions equalities.
theorem proof1 : (m + 2 * n) - (m - 2 * n) = 4 * n := sorry
theorem proof2 : 2 * (x - 3) - (-x + 4) = 3 * x - 10 := sorry
theorem proof3 : 2 * x - 3 * (x - 2 * y + 3 * x) + 2 * (3 * x - 3 * y + 2 * z) = -4 * x + 4 * z := sorry
theorem proof4 : 8 * m^2 - (4 * m^2 - 2 * m - 4 * (2 * m^2 - 5 * m)) = 12 * m^2 - 18 * m := sorry

end proof1_proof2_proof3_proof4_l591_591751


namespace sum_f_2_to_2_pow_10_l591_591443

def f (x : ℝ) : ℝ := sorry -- The definition is derived from f(3^x) = x log₂(3)

theorem sum_f_2_to_2_pow_10 (h : ∀ x, f (3^x) = x * log 3 / log 2) :
  (f 2) + (f (2^2)) + (f (2^3)) + (f (2^4)) + (f (2^5)) + (f (2^6)) +
  (f (2^7)) + (f (2^8)) + (f (2^9)) + (f (2^10)) = 55 :=
sorry

end sum_f_2_to_2_pow_10_l591_591443


namespace sin_cos_system_solution_l591_591475

theorem sin_cos_system_solution (x y : ℝ) (m n : ℤ) :
  (sin x * cos y = 0.25) ∧ (sin y * cos x = 0.75) →
  (∃ m n : ℤ, x = (π / 6) + π * (m - n) ∧ y = (π / 3) + π * (m + n)) ∨
  (∃ m n : ℤ, x = (- π / 6) + π * (m - n) ∧ y = (2 * π / 3) + π * (m + n)) :=
sorry

end sin_cos_system_solution_l591_591475


namespace tangents_and_fraction_l591_591491

theorem tangents_and_fraction
  (α β : ℝ)
  (tan_diff : Real.tan (α - β) = 2)
  (tan_beta : Real.tan β = 4) :
  (7 * Real.sin α - Real.cos α) / (7 * Real.sin α + Real.cos α) = 7 / 5 :=
sorry

end tangents_and_fraction_l591_591491


namespace number_of_values_l591_591163

open Real

theorem number_of_values (x : ℝ) (h₁ : -19 < x) (h₂ : x < 98) (h₃ : cos x ^ 2 + 2 * sin x ^ 2 = 1) : 
  ∃ n : ℕ, n = 38 :=
by
  sorry

end number_of_values_l591_591163


namespace min_value_fraction_l591_591836

noncomputable def f (a c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - 2 * x + c

theorem min_value_fraction (a c : ℝ) (h1 : a * c = 1) (h2 : 0 < a) (h3 : 0 < c) :
  (∀ a c, a * c = 1 ∧ 0 < a ∧ 0 < c → ∃ m, m = (frac 9 a) + (frac 1 c) ∧ m = 6) :=
sorry

end min_value_fraction_l591_591836


namespace sphere_radius_equal_l591_591284

theorem sphere_radius_equal (r : ℝ) 
  (hvol : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end sphere_radius_equal_l591_591284


namespace incorrect_statements_l591_591454

-- Definitions based on conditions from the problem.

def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c < 0
def solution_set (a b c : ℝ) : Set ℝ := {x | quadratic_inequality a b c x}
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Lean statements of the conditions and the final proof problem.
theorem incorrect_statements (a b c : ℝ) (M : Set ℝ) :
  (M = ∅ → (a < 0 ∧ discriminant a b c < 0) → false) ∧
  (M = {x | x ≠ x0} → a < b → (a + 4 * c) / (b - a) = 2 + 2 * Real.sqrt 2 → false) := sorry

end incorrect_statements_l591_591454


namespace altitude_in_triangle_l591_591724

theorem altitude_in_triangle :
  ∀ (a b c : ℝ), a = 40 → b = 50 → c = 70 →
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  let h := 2 * area / c in
  h = (80 * Real.sqrt 7) / 7 :=
by
  intro a b c ha hb hc
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h := 2 * area / c
  have ha : a = 40 := ha
  have hb : b = 50 := hb
  have hc : c = 70 := hc
  sorry

end altitude_in_triangle_l591_591724


namespace rational_function_domain_l591_591078

noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 - 4*x + 5) / (x^2 - 5*x + 4)

theorem rational_function_domain :
  {x : ℝ | ∃ y, h y = h x } = {x : ℝ | x ≠ 1 ∧ x ≠ 4} := 
sorry

end rational_function_domain_l591_591078


namespace number_of_parallelograms_l591_591519

def f (n : ℕ) : ℕ := 3 * ((n + 2) * (n + 1) * n * (n - 1)) / 24

theorem number_of_parallelograms (n : ℕ) : f n = 3 * ((n + 2) * (n + 1) * n * (n - 1)) / 24 :=
by
  rw f
  sorry

end number_of_parallelograms_l591_591519


namespace factor_present_l591_591755

noncomputable def given_expr := (x^2 - y^2 - z^2 + 2 * y * z + x + y - z)

theorem factor_present:
  ∃ f: Polynomial ℤ, ∃ g: Polynomial ℤ, given_expr = f * g ∧ ( f = x - y + z + 1 ∨ g = x - y + z + 1 ) :=
sorry

end factor_present_l591_591755


namespace final_percent_decrease_l591_591533

open Real

def originalPriceEuro : ℝ := 100
def discountRate : ℝ := 0.30
def germanVATRate : ℝ := 0.19
def usVATRate : ℝ := 0.08
def exchangeRate : ℝ := 1.18

theorem final_percent_decrease :
  let priceAfterDiscount := originalPriceEuro * (1 - discountRate),
      priceWithGermanVAT := priceAfterDiscount * (1 + germanVATRate),
      priceInUSD := priceWithGermanVAT * exchangeRate,
      finalPriceInUSD := priceInUSD * (1 + usVATRate),
      originalPriceUSD := originalPriceEuro * exchangeRate,
      percentDecrease := ((originalPriceUSD - finalPriceInUSD) / originalPriceUSD) * 100 in
  abs (percentDecrease - 10.04) < 0.001 :=
  by
  sorry

end final_percent_decrease_l591_591533


namespace complex_magnitude_solution_l591_591427

theorem complex_magnitude_solution (n : ℝ) (h1 : |Complex.mk 5 n| = 5 * Real.sqrt 13) (h2 : n ^ 2 + 5 * n > 50) :
  n = 10 * Real.sqrt 3 := sorry

end complex_magnitude_solution_l591_591427


namespace angle_H_is_60_l591_591192

variable (EFGH : Type) [parallelogram EFGH]
variable (F H : EFGH) (G E : EFGH)
variable (O : EFGH)

-- Angle measure function
variable [angle_measure : EFGH → ℝ]

-- Given conditions
axiom angle_F_is_120 : angle_measure F = 120
axiom diagonals_intersect : ∃ O : EFGH, O = intersection E G F H
axiom angle_EOF_is_40 : angle_measure (intersection_angle E F O) = 40

-- Prove that angle H is 60
theorem angle_H_is_60 : angle_measure H = 60 := sorry

end angle_H_is_60_l591_591192


namespace minimum_removal_to_prevent_triangles_l591_591428

noncomputable def triangle_grid (num_toothpicks : ℕ) (upward_triangles : ℕ) (downward_triangles : ℕ) : Prop :=
  num_toothpicks = 40 ∧ upward_triangles = 10 ∧ downward_triangles = 15

theorem minimum_removal_to_prevent_triangles (num_toothpicks upward_triangles downward_triangles : ℕ) :
  triangle_grid num_toothpicks upward_triangles downward_triangles →
  ∃ (min_toothpicks_to_remove : ℕ), min_toothpicks_to_remove = 10 :=
by 
  intros h,
  sorry

end minimum_removal_to_prevent_triangles_l591_591428


namespace f_double_minus_f_single_eq_n_squared_l591_591216

-- Definition of k(n) as the largest odd divisor of n
def largest_odd_divisor (n : ℕ) : ℕ :=
  let d := list.filter (λ x, x % 2 = 1) (nat.divisors n)
  list.foldr max 1 d

-- Definition of f(n) as the sum of the largest odd divisor of 1 to n
def f (n : ℕ) : ℕ := 
  (list.range (n + 1)).map largest_odd_divisor |>.sum

-- Problem statement: Prove that f(2n) - f(n) = n^2
theorem f_double_minus_f_single_eq_n_squared (n : ℕ) : 
  f (2 * n) - f n = n ^ 2 :=
sorry

end f_double_minus_f_single_eq_n_squared_l591_591216


namespace simona_treatment_cost_l591_591246

section SimonaTreatment

variables (x : ℕ) (cost_per_complex : ℕ := 197)

/-- Simona was left with only one complex after three treatments where 
each treatment frees her from half the remaining complexes and half of one of the complexes. 
Find the total cost if each reduction costs 197 francs per complex. -/
theorem simona_treatment_cost 
  (h_first_treatment: (1/2 : ℝ) * ↑x + (1/2 : ℝ) = ↑((1/2 : ℝ) * (x + 1).to_nat))
  (h_second_treatment: (1/2 : ℝ) * (1/2 : ℝ) * ↑(x + 1) + (1/2 : ℝ) = ↑((1/4 : ℝ) * (x + 3).to_nat))
  (h_third_treatment: (1/2 : ℝ) * (1/4 : ℝ) * ↑(x + 3) + (1/2 : ℝ) = ↑((1/8 : ℝ) * (x + 7).to_nat))
  (h_final_complex: (1/8 : ℝ) * ↑(x + 7) = 1) :
  cost_per_complex * (x - 1) = 1379 
:= sorry

end SimonaTreatment

end simona_treatment_cost_l591_591246


namespace simplify_complex_fractions_l591_591595

theorem simplify_complex_fractions :
  let a := 4
  let b := 7
  let i := Complex.I
  (a + b * i) / (a - b * i) + (a - b * i) / (a + b * i) = -66/65 :=
by
  let a := 4
  let b := 7
  let i := Complex.I
  have h1 : (a + b * i) / (a - b * i) = (Complex (a + b * i) : ℂ) / Complex (a - b * i),
  from rfl,
  have h2 : (a - b * i) / (a + b * i) = (Complex (a - b * i) : ℂ) / Complex (a + b * i),
  from rfl,
  rw [h1, h2],
  -- blocks for the calculation steps
  sorry

end simplify_complex_fractions_l591_591595


namespace range_cos_pi_over_3_omega_l591_591852

theorem range_cos_pi_over_3_omega (ω : ℝ) (f : ℝ → ℝ)
  (hω : ω > 0)
  (hf : ∀ x, f x = 3 * sin (ω * x) + 4 * cos (ω * x))
  (h₀ : ∀ x, 0 ≤ x ∧ x ≤ π / 3 → 4 ≤ f x ∧ f x ≤ 5) :
  ∀ y, y = cos (π / 3 * ω) → 7 / 25 ≤ y ∧ y ≤ 4 / 5 :=
by
  sorry

end range_cos_pi_over_3_omega_l591_591852


namespace chair_cost_l591_591574

-- Define the conditions
def total_spent : ℕ := 56
def table_cost : ℕ := 34
def num_chairs : ℕ := 2

-- Define the statement we need to prove
theorem chair_cost :
  ∃ (chair_cost : ℕ), 2 * chair_cost + table_cost = total_spent ∧ chair_cost = 11 :=
by
  use 11
  split
  sorry -- proof goes here, skipped as per instructions

end chair_cost_l591_591574


namespace sum_binom_divisible_l591_591237

theorem sum_binom_divisible (n : ℕ) : 
  (∑ k in Ico 0 ((n-1)/2 + 1), 5^k * (Nat.choose n (2*k+1))) % 2^(n-1) = 0 := 
by 
  sorry

end sum_binom_divisible_l591_591237


namespace lattice_points_subset_l591_591200

theorem lattice_points_subset (H : Finset (ℤ × ℤ)) :
  ∃ K ⊆ H, 
    (∀ x, (H.filter (λ p, p.1 = x)).card ≤ 2) ∧ 
    (∀ y, (H.filter (λ p, p.2 = y)).card ≤ 2) ∧ 
    (∀ p ∈ H \ K, ∃ q r ∈ K, (p.1 = q.1 ∨ p.1 = r.1) ∧ (p.2 = q.2 ∨ p.2 = r.2)) :=
by
  sorry

end lattice_points_subset_l591_591200


namespace part1_part2_case1_part2_case2_part2_case3_part3_l591_591823

variable (m : ℝ)
def f (x : ℝ) := (m + 1) * x^2 - (m - 1) * x + (m - 1)

-- Part (1)
theorem part1 (h : ∀ x : ℝ, f m x < 1) : m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

-- Part (2)
theorem part2_case1 (h : m = -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≥ 1 :=
sorry

theorem part2_case2 (h : m > -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≤ (m - 1) / (m + 1) ∨ x ≥ 1 :=
sorry

theorem part2_case3 (h : m < -1) : ∀ x, f m x ≥ (m + 1) * x ↔ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1) :=
sorry

-- Part (3)
theorem part3 (h : ∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), f m x ≥ 0) : m ≥ 1 :=
sorry

end part1_part2_case1_part2_case2_part2_case3_part3_l591_591823


namespace minimum_distance_l591_591814

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 2 * P.1

-- Define point A
def A : ℝ × ℝ := (7 / 2, 4)

-- Define the focus point of the parabola y^2 = 2x
def F : ℝ × ℝ := (1 / 2, 0)

-- Define the distance from P to the y-axis
def d1 (P : ℝ × ℝ) : ℝ := abs P.1

-- Define the distance from P to point A
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

def d2 (P : ℝ × ℝ) : ℝ := distance P A

-- Definition for the total distance
def total_distance (P : ℝ × ℝ) : ℝ := d1 P + d2 P

-- Prove the minimum value of d1 + d2 is 9 / 2
theorem minimum_distance :
  ∃ P : ℝ × ℝ, parabola P ∧ total_distance P = 9 / 2 :=
sorry

end minimum_distance_l591_591814


namespace seashells_total_l591_591591

theorem seashells_total (sam_seashells : ℕ) (mary_seashells : ℕ) (h1 : sam_seashells = 18) (h2 : mary_seashells = 47) : sam_seashells + mary_seashells = 65 :=
by
  rw [h1, h2]
  exact rfl

end seashells_total_l591_591591


namespace second_set_avg_var_l591_591842

variables {α : Type*} [field α]

def first_set_avg (x1 x2 x3 x4 x5 : α) : α := (x1 + x2 + x3 + x4 + x5) / 5

def first_set_variance : α := 1 / 2

def second_set (x1 x2 x3 x4 x5 : α) : list α := [3 * x1 - 2, 3 * x2 - 2, 3 * x3 - 2, 3 * x4 - 2, 3 * x5 - 2]

theorem second_set_avg_var {x1 x2 x3 x4 x5 : α} 
  (h_avg : first_set_avg x1 x2 x3 x4 x5 = 4)
  (h_var : first_set_variance = 1 / 2) :
  ((3 * (x1 + x2 + x3 + x4 + x5) - 10) / 5 = 10) ∧
  (9 * (1 / 2) = 9 / 2) :=
by 
  sorry

end second_set_avg_var_l591_591842


namespace space_creature_perimeter_correct_l591_591509

-- Definitions and conditions
def radius := 2 -- radius of the circle in cm
def mouth_angle := 90 -- central angle of the mouth in degrees

-- The fraction of the circle forming the arc
def arc_fraction := 3 / 4

-- The circumference of the full circle
def circumference := 2 * π * radius

-- Calculations
def arc_length := arc_fraction * circumference
def radii_length := 2 * radius

-- Proposition to prove: Perimeter of the space creature
def space_creature_perimeter := arc_length + radii_length

-- The target theorem we want to prove
theorem space_creature_perimeter_correct : space_creature_perimeter = 3 * π + 4 := by
  sorry

end space_creature_perimeter_correct_l591_591509


namespace cube_root_eq_self_l591_591170

theorem cube_root_eq_self (a : ℝ) (h : a^(3:ℕ) = a) : a = 1 ∨ a = -1 ∨ a = 0 := 
sorry

end cube_root_eq_self_l591_591170


namespace triangle_equilateral_max_area_OACB_l591_591870

open Real

variables {A B C θ : ℝ} {a b c OA OB : ℝ}

theorem triangle_equilateral (h1 : b = c) 
    (h2 : (sin B) / (sin A) = (1 - cos B) / (cos A)) : 
    A = π / 3 :=
by
  -- Proof is omitted; begin by solving the given trigonometric equation
  sorry

theorem max_area_OACB (h_theta : 0 < θ) (h_theta_pi : θ < π) (h_OA : OA = 2) (h_OB : OB = 1)
    (h_equilateral : triangle_equilateral) :
    let a := 2 in
    let S := area_oa (θ + π / 3) + (sqrt 3 / 4) * (1 + 4 - 4 * cos θ) in
    ∀ θ (h_theta : 0 < θ) (h_theta_pi : θ < π),
    S = (8 + 5 * sqrt 3) / 4 :=
by
  -- Proof is omitted; maximize the area function with trigonometric substitution
  sorry

end triangle_equilateral_max_area_OACB_l591_591870


namespace hyperbola_distance_between_vertices_l591_591412

theorem hyperbola_distance_between_vertices :
  ∀ (x y : ℝ), ((x^2) / 64 - (y^2) / 49 = 1) → (distance (8, 0) (-8, 0) = 16) :=
by
  sorry

end hyperbola_distance_between_vertices_l591_591412


namespace group_element_decomposition_l591_591328

variables {G : Type*} [group G] [fintype G]
variables (K : set G) (hK : fintype.card K > fintype.card G / 2)

theorem group_element_decomposition (g : G) :
  ∃ h k ∈ K, g = h * k :=
by sorry

end group_element_decomposition_l591_591328


namespace total_questions_asked_l591_591512

theorem total_questions_asked (drew_correct : ℕ := 20) (drew_wrong : ℕ := 6) (carla_correct : ℕ := 14):
  let carla_wrong := 2 * drew_wrong in
  let drew_total := drew_correct + drew_wrong in
  let carla_total := carla_correct + carla_wrong in
  (drew_total + carla_total) = 52 := by
  sorry

end total_questions_asked_l591_591512


namespace abs_cube_root_neg_64_l591_591325

-- Definitions required for the problem
def cube_root (x : ℝ) : ℝ := x^(1/3)
def abs_value (x : ℝ) : ℝ := abs x

-- The statement of the problem
theorem abs_cube_root_neg_64 : abs_value (cube_root (-64)) = 4 :=
by sorry

end abs_cube_root_neg_64_l591_591325


namespace cos_phi_l591_591296

theorem cos_phi (
  (P Q R S T : Type)
  (midpoint_Q : Q = (S + T) / 2)
  (PQ ST : P → Q → ℝ)
  (PQ_val : PQ = 2)
  (ST_val : ST = 2)
  (QR RP : Q → R → ℝ)
  (QR_val : QR = 8)
  (RP_val : RP = real.sqrt 72)
  (dot_product_condition : ((λ u v : ℝ, u * v) PQ PS + (λ u v : ℝ, u * v) PR PT = 4))
  ) : cos (angle ST QR) = 1 :=
sorry

end cos_phi_l591_591296


namespace alexandra_writing_time_l591_591731

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem alexandra_writing_time : 
  (8! / 16) / 60 = 42 := 
by
  sorry

end alexandra_writing_time_l591_591731


namespace cost_per_ball_is_two_l591_591564

def cost_per_tennis_ball (packs : ℕ) (balls_per_pack : ℕ) (total_cost : ℕ) : ℝ :=
  total_cost / (packs * balls_per_pack)

theorem cost_per_ball_is_two :
  (cost_per_tennis_ball 4 3 24) = 2 :=
by
  sorry

end cost_per_ball_is_two_l591_591564


namespace circumcircle_passes_through_midpoint_l591_591257

open EuclideanGeometry

variables {A B C D E F P Q R : Point}

-- Assume the conditions
variables (triangle_ABC : Triangle A B C)
variables (acute_angle_ABC : isAcuteAngle ∠ A B C)
variables (D_def : foot_of_altitude A B C D)
variables (E_def : foot_of_altitude B A C E)
variables (F_def : foot_of_altitude C A B F)
variables (line_parallel_EF_DQ : parallel (line_through D (parallel_line EF)) (line_through Q AC))
variables (line_parallel_EF_DR : parallel (line_through D (parallel_line EF)) (line_through R AB))
variables (line_intersect_EF_P : line_intersects EF BC P)

theorem circumcircle_passes_through_midpoint :
  is_cyclic_quad P Q R (midpoint B C) := sorry

end circumcircle_passes_through_midpoint_l591_591257


namespace karen_paddling_time_l591_591537

-- Define the given conditions
def karen_speed_still_water : ℕ := 10
def river_current_speed : ℕ := 4
def river_length : ℕ := 12

-- Compute Karen's net speed
def karen_net_speed : ℕ := karen_speed_still_water - river_current_speed

-- Define the target time
def target_time : ℕ := 2

-- The proof statement
theorem karen_paddling_time : river_length / karen_net_speed = target_time := 
by 
  have h1 : karen_net_speed = 6 := rfl
  have h2 : river_length = 12 := rfl
  rw [h1, h2]
  calc
  12 / 6 = 2 : rfl

sorry -- Fill in the complete proof

end karen_paddling_time_l591_591537


namespace parabola_focus_directrix_proof_l591_591093

noncomputable def parabola_focus_directrix (x y : ℝ) : Prop :=
  (x - 2) = (y - 3)^2

theorem parabola_focus_directrix_proof :
  ∃ f d,
  parabola_focus_directrix f 3 ∧ f = 2.25 ∧ d = 1.75 :=
begin
  -- what we are given in the problem
  have h1 : ∀ y, parabola_focus_directrix (2 + (y - 3)^2) y,
  { intros y,
    simp [parabola_focus_directrix],
  },
  -- specific y value for the focus
  set y0 := 3 with hy0,
  have h2 : parabola_focus_directrix (2.25) y0,
  { change (2.25 - 2) = (y0 - 3)^2,
    simp [hy0] },
  -- finding the focus and the directrix
  use [2.25, 1.75],
  split,
  { assumption },
  split,
  { refl },
  { refl },
end

end parabola_focus_directrix_proof_l591_591093


namespace police_emergency_number_prime_divisor_l591_591036

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 1000 * k + 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n :=
sorry

end police_emergency_number_prime_divisor_l591_591036


namespace domain_phi_inequality_a_gt_1_inequality_0_lt_a_lt_1_l591_591154

noncomputable def f (a x : ℝ) : ℝ := Real.log.base a (x - 1)
noncomputable def g (a x : ℝ) : ℝ := Real.log.base a (6 - 2 * x)
noncomputable def phi (a x : ℝ) : ℝ := f a x + g a x

theorem domain_phi (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  {x : ℝ | 1 < x ∧ x < 3} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

theorem inequality_a_gt_1 (a : ℝ) (h1 : a > 1) (x : ℝ) : 
  1 < x ∧ x ≤ 7 / 3 ↔ f a x ≤ g a x :=
sorry

theorem inequality_0_lt_a_lt_1 (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (x : ℝ) : 
  7 / 3 ≤ x ∧ x < 3 ↔ f a x ≤ g a x :=
sorry

end domain_phi_inequality_a_gt_1_inequality_0_lt_a_lt_1_l591_591154


namespace min_dist_parabola_l591_591921

open Real

-- Definitions of points and parabola
def C : ℝ × ℝ := (2, 0)
def D : ℝ × ℝ := (6, 3)
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Definition of distance in Euclidean space
def dist (P Q : ℝ × ℝ) : ℝ := sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Main theorem
theorem min_dist_parabola : 
  ∀ Q : ℝ × ℝ, parabola Q.1 Q.2 → dist C Q + dist D Q ≥ 6 :=
by
  sorry

end min_dist_parabola_l591_591921


namespace min_value_a_plus_2b_l591_591128

theorem min_value_a_plus_2b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_condition : (a + b) / (a * b) = 1) : a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_a_plus_2b_l591_591128


namespace set_representation_l591_591992

open Nat

def isInPositiveNaturals (x : ℕ) : Prop :=
  x ≠ 0

def isPositiveDivisor (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

theorem set_representation :
  {x | isInPositiveNaturals x ∧ isPositiveDivisor 6 (6 - x)} = {3, 4, 5} :=
by
  sorry

end set_representation_l591_591992


namespace star_evaluation_l591_591075

def star (X Y : ℚ) := (X + Y) / 4

theorem star_evaluation : star (star 3 8) 6 = 35 / 16 := by
  sorry

end star_evaluation_l591_591075


namespace total_number_of_chickens_is_76_l591_591531

theorem total_number_of_chickens_is_76 :
  ∀ (hens roosters chicks : ℕ),
    hens = 12 →
    (∀ n, roosters = hens / 3) →
    (∀ n, chicks = hens * 5) →
    hens + roosters + chicks = 76 :=
by
  intros hens roosters chicks h1 h2 h3
  rw [h1, h2 12, h3 12]
  ring
  sorry

end total_number_of_chickens_is_76_l591_591531


namespace inequality_case_1_inequality_case_2_l591_591960

-- Define the main theorem for the first condition
theorem inequality_case_1 (x : ℝ) : 
  (1 : ℝ)^2 * x + (0 : ℝ)^2 * (1 - x) ≥ (1 : ℝ * x + 0 : ℝ * (1 - x)) ^ 2 ↔ 0 ≤ x ∧ x ≤ 1 :=
by
  sorry

-- Define the main theorem for the second condition
theorem inequality_case_2 (x a b : ℝ) (h : a ≠ b) : 
  a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2 ↔ 0 ≤ x ∧ x ≤ 1 :=
by 
  sorry

end inequality_case_1_inequality_case_2_l591_591960


namespace spring_summer_work_hours_l591_591203

def john_works_spring_summer : Prop :=
  ∀ (work_hours_winter_week : ℕ) (weeks_winter : ℕ) (earnings_winter : ℕ)
    (weeks_spring_summer : ℕ) (earnings_spring_summer : ℕ) (hourly_rate : ℕ),
    work_hours_winter_week = 40 →
    weeks_winter = 8 →
    earnings_winter = 3200 →
    weeks_spring_summer = 24 →
    earnings_spring_summer = 4800 →
    hourly_rate = earnings_winter / (work_hours_winter_week * weeks_winter) →
    (earnings_spring_summer / hourly_rate) / weeks_spring_summer = 20

theorem spring_summer_work_hours : john_works_spring_summer :=
  sorry

end spring_summer_work_hours_l591_591203


namespace greatest_int_less_than_neg_19_div_3_l591_591661

theorem greatest_int_less_than_neg_19_div_3 : ∃ n : ℤ, n = -7 ∧ n < (-19 / 3 : ℚ) ∧ (-19 / 3 : ℚ) < n + 1 := 
by
  sorry

end greatest_int_less_than_neg_19_div_3_l591_591661


namespace initial_persimmons_l591_591485

axiom eaten_persimmons : ℕ := 5
axiom left_persimmons : ℕ := 12

theorem initial_persimmons : eaten_persimmons + left_persimmons = 17 := 
by
  sorry

end initial_persimmons_l591_591485


namespace trigonometric_identity_l591_591109

theorem trigonometric_identity
  (α : ℝ) 
  (h : Real.tan α = -1 / 2) :
  (Real.cos α - Real.sin α)^2 / Real.cos (2 * α) = 3 := 
by 
  sorry

end trigonometric_identity_l591_591109


namespace equivalent_fraction_l591_591962

theorem equivalent_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = -3) :
  (2 * x + 8 * y) / (4 * x - 2 * y) = 38 / 13 :=
by
  sorry

end equivalent_fraction_l591_591962


namespace smallest_perimeter_of_acute_triangle_with_consecutive_sides_l591_591663

theorem smallest_perimeter_of_acute_triangle_with_consecutive_sides :
  ∃ (a : ℕ), (a > 1) ∧ (∃ (b c : ℕ), b = a + 1 ∧ c = a + 2 ∧ (∃ (C : ℝ), a^2 + b^2 - c^2 < 0 ∧ c = 4)) ∧ (a + (a + 1) + (a + 2) = 9) :=
by {
  sorry
}

end smallest_perimeter_of_acute_triangle_with_consecutive_sides_l591_591663


namespace max_distance_729_plus_81_sqrt_5_l591_591211

noncomputable theory

open Complex

def max_distance (w : ℂ) (h : abs w = 3) : ℝ :=
  let z := (1 + 2 * Complex.I) * w^4 - w^6
  in Complex.abs z

theorem max_distance_729_plus_81_sqrt_5 :
  ∀ (w : ℂ) (h : abs w = 3), max_distance w h = 729 + 81 * Real.sqrt 5 :=
by 
  intros 
  sorry

end max_distance_729_plus_81_sqrt_5_l591_591211


namespace factorize_2mn_cube_arithmetic_calculation_l591_591326

-- Problem 1: Factorization problem
theorem factorize_2mn_cube (m n : ℝ) : 
  2 * m^3 * n - 8 * m * n^3 = 2 * m * n * (m + 2 * n) * (m - 2 * n) :=
by sorry

-- Problem 2: Arithmetic calculation problem
theorem arithmetic_calculation : 
  |1 - Real.sqrt 3| + 3 * Real.tan (Real.pi / 6) - ((Real.pi - 3)^0) + (-1/3)⁻¹ = 2 * Real.sqrt 3 - 5 :=
by sorry

end factorize_2mn_cube_arithmetic_calculation_l591_591326


namespace find_vector_OB_l591_591787

-- Define the vectors
def vector_OA := (-2, 3) : ℤ × ℤ
def vector_AB := (-1, -4) : ℤ × ℤ

-- State the theorem to prove
theorem find_vector_OB : 
  let vector_OB := vector_OA.1 + vector_AB.1, vector_OA.2 + vector_AB.2
  in vector_OB = (-3, -1) :=
by
  sorry

end find_vector_OB_l591_591787


namespace number_of_lines_dist_l591_591875

theorem number_of_lines_dist {A B : ℝ × ℝ} (hA : A = (3, 0)) (hB : B = (0, 4)) : 
  ∃ n : ℕ, n = 3 ∧
  ∀ l : ℝ → ℝ → Prop, 
  (∀ p : ℝ × ℝ, l p.1 p.2 → p ≠ A → dist A p = 2) ∧ 
  (∀ p : ℝ × ℝ, l p.1 p.2 → p ≠ B → dist B p = 3) → n = 3 := 
by sorry

end number_of_lines_dist_l591_591875


namespace diagonal_blocks_sum_l591_591952

theorem diagonal_blocks_sum :
  let n_values := [838, 839, 840]  -- Possible values of n
  in n_values.sum = 2517 := 
by
  -- The proof is omitted as instructed.
  sorry

end diagonal_blocks_sum_l591_591952


namespace polynomial_irreducible_over_Z_l591_591955

noncomputable def irreducible_polynomial (n : ℕ) : Prop :=
  irreducible (X^n + 3*X^(n-1) + 5)

theorem polynomial_irreducible_over_Z (n : ℕ) (h : n ≥ 2) : irreducible_polynomial n :=
sorry

end polynomial_irreducible_over_Z_l591_591955


namespace regular_polygons_enclosing_hexagon_l591_591344

theorem regular_polygons_enclosing_hexagon (m n : ℕ) 
  (hm : m = 6)
  (h_exterior_angle_central : 180 - ((m - 2) * 180 / m) = 60)
  (h_exterior_angle_enclosing : 2 * 60 = 120): 
  n = 3 := sorry

end regular_polygons_enclosing_hexagon_l591_591344


namespace length_of_paper_l591_591722

-- Let conditions be defined in Lean
def paper_width : ℝ := 4
def initial_diameter : ℝ := 3
def wrap_count : ℕ := 400
def final_diameter : ℝ := 11

-- Statement of the proof problem in Lean
theorem length_of_paper
  (paper_width : ℝ)
  (initial_diameter : ℝ)
  (wrap_count : ℕ)
  (final_diameter : ℝ) :
  (2 * wrap_count * paper_width) * real.pi / 100 = 28 * real.pi :=
by
  sorry

end length_of_paper_l591_591722


namespace infinite_geometric_series_common_ratio_l591_591737

theorem infinite_geometric_series_common_ratio 
  (a S r : ℝ) 
  (ha : a = 400) 
  (hS : S = 2500)
  (h_sum : S = a / (1 - r)) :
  r = 0.84 :=
by
  -- Proof will go here
  sorry

end infinite_geometric_series_common_ratio_l591_591737


namespace exists_infinitely_many_n_with_divisor_property_l591_591244

theorem exists_infinitely_many_n_with_divisor_property :
  ∃ᶠ n in at_top, (∑ m in n.divisors \ {n}, m) = n + 12
:= sorry

end exists_infinitely_many_n_with_divisor_property_l591_591244


namespace cos_theta_geom_progression_l591_591544

theorem cos_theta_geom_progression (θ : ℝ) (h1: 0 < θ ∧ θ < π / 2)
  (h2: ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  (a = sin θ ∨ a = sin (2 * θ) ∨ a = sin (3 * θ)) ∧
  (b = sin θ ∨ b = sin (2 * θ) ∨ b = sin (3 * θ)) ∧
  (c = sin θ ∨ c = sin (2 * θ) ∨ c = sin (3 * θ)) ∧ a * c = b^2) :
  cos θ = sqrt (3 / 8) :=
sorry

end cos_theta_geom_progression_l591_591544


namespace tan_alpha_eq_two_l591_591429

/--
Given that tan(α) = 2, prove that
1 / (2 * sin(α) * cos(α) + cos(α)^2) = 1.
-/
theorem tan_alpha_eq_two :
  (∀ α : ℝ, tan α = 2 → 1 / (2 * sin α * cos α + cos α ^ 2) = 1) :=
by
  intro α h
  sorry

end tan_alpha_eq_two_l591_591429


namespace more_freshmen_than_sophomores_l591_591316

variable (total_students juniors_percent non_sophomores_percent seniors_count : ℕ)
variable (total_students_pos : total_students = 800)
variable (juniors_percent_correct : juniors_percent = 22)
variable (non_sophomores_percent_correct : non_sophomores_percent = 75)
variable (seniors_count_correct : seniors_count = 160)

theorem more_freshmen_than_sophomores (h1 : total_students = 800)
    (h2 : juniors_percent = 22)
    (h3 : non_sophomores_percent = 75)
    (h4 : seniors_count = 160) : 
    let juniors := (juniors_percent * total_students) / 100 in
    let non_sophomores := (non_sophomores_percent * total_students) / 100 in
    let sophomores := total_students - non_sophomores in
    let seniors := seniors_count in
    let freshmen := total_students - (juniors + sophomores + seniors) in
    freshmen - sophomores = 64 :=
by
  sorry

end more_freshmen_than_sophomores_l591_591316


namespace find_pairs_l591_591404

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

theorem find_pairs {a b : ℝ} :
  (0 < b) → (b ≤ 1) → (0 < a) → (a < 1) → (2 * a + b ≤ 2) →
  (∀ x y : ℝ, f a b (x * y) + f a b (x + y) ≥ f a b x * f a b y) :=
by
  intros h_b_gt_zero h_b_le_one h_a_gt_zero h_a_lt_one h_2a_b_le_2
  sorry

end find_pairs_l591_591404


namespace cost_per_kg_paint_l591_591973

-- Define the basic parameters
variables {sqft_per_kg : ℝ} -- the area covered by 1 kg of paint
variables {total_cost : ℝ} -- the total cost to paint the cube
variables {side_length : ℝ} -- the side length of the cube
variables {num_faces : ℕ} -- the number of faces of the cube

-- Define the conditions given in the problem
def conditions (sqft_per_kg total_cost side_length : ℝ) (num_faces : ℕ) : Prop :=
  sqft_per_kg = 16 ∧
  total_cost = 876 ∧
  side_length = 8 ∧
  num_faces = 6

-- Define the statement to prove, which is the cost per kg of paint
theorem cost_per_kg_paint (sqft_per_kg total_cost side_length : ℝ) (num_faces : ℕ) :
  conditions sqft_per_kg total_cost side_length num_faces →
  ∃ cost_per_kg : ℝ, cost_per_kg = 36.5 :=
by
  sorry

end cost_per_kg_paint_l591_591973


namespace probability_not_below_x_axis_half_l591_591234

-- Define the vertices of the parallelogram
def P : (ℝ × ℝ) := (4, 4)
def Q : (ℝ × ℝ) := (-2, -2)
def R : (ℝ × ℝ) := (-8, -2)
def S : (ℝ × ℝ) := (-2, 4)

-- Define a predicate for points within the parallelogram
def in_parallelogram (A B C D : ℝ × ℝ) (p : ℝ × ℝ) : Prop := sorry

-- Define the area function
def area_of_parallelogram (A B C D : ℝ × ℝ) : ℝ := sorry

noncomputable def probability_not_below_x_axis (A B C D : ℝ × ℝ) : ℝ :=
  let total_area := area_of_parallelogram A B C D
  let area_above_x_axis := area_of_parallelogram (0, 0) D A (0, 0) / 2
  area_above_x_axis / total_area

theorem probability_not_below_x_axis_half :
  probability_not_below_x_axis P Q R S = 1 / 2 :=
sorry

end probability_not_below_x_axis_half_l591_591234


namespace distance_hyperbola_vertices_l591_591409

noncomputable def distance_between_vertices (a : ℝ) : ℝ :=
2 * a

theorem distance_hyperbola_vertices :
  ∃ a : ℝ, a^2 = 64 ∧ distance_between_vertices a = 16 :=
by
  have h : a = sqrt 64 := by
    apply sqrt_eq
    rw [pow_two, nat.cast_comm]
  use 8
  rw h
  apply sqrt_pos
  norm_num
  use 8
  split
  exact rfl
  norm_num
  sorry

end distance_hyperbola_vertices_l591_591409


namespace Woojoo_initial_score_l591_591270

theorem Woojoo_initial_score
  (n_students : ℕ := 10)
  (avg_initial : ℕ := 42)
  (avg_new : ℕ := 44)
  (Woojoo_rescored : ℕ := 50) :
  (initial_score_Woojoo : ℕ) :
  avg_initial * n_students + Woojoo_rescored - initial_score_Woojoo = avg_new * n_students → 
  initial_score_Woojoo = 30 := 
by
  sorry

end Woojoo_initial_score_l591_591270


namespace min_students_wearing_both_l591_591879

theorem min_students_wearing_both (n : ℕ) (h1 : n % 7 = 0) (h2 : n % 6 = 0) : 
  let x := (3 * n / 7 + 5 * n / 6 - n) in x = 11 :=
by
  -- Proof will go here
  sorry

end min_students_wearing_both_l591_591879


namespace probability_factor_less_than_10_l591_591305

def num_factors (n : ℕ) : ℕ :=
  (n.divisors).to_finset.card

def num_factors_less_than (n : ℕ) (m : ℕ) : ℕ :=
  ((n.divisors).filter (λ x, x < m)).to_finset.card

theorem probability_factor_less_than_10 (n : ℕ) (h : n = 120) :
  (num_factors_less_than n 10 : ℚ) / num_factors n = 7 / 16 := by
  sorry

end probability_factor_less_than_10_l591_591305


namespace cos_105_correct_l591_591375

noncomputable def cos_105 : ℝ :=
  -(sqrt 3) / 4 - sqrt 2 / 4

theorem cos_105_correct :
  cos (105 * real.pi / 180) = cos_105 := by
    have h1 : cos (60 * real.pi / 180) = 1 / 2 := by sorry
    have h2 : cos (45 * real.pi / 180) = sqrt 2 / 2 := by sorry
    have h3 : sin (60 * real.pi / 180) = sqrt 3 / 2 := by sorry
    have h4 : sin (45 * real.pi / 180) = sqrt 2 / 2 := by sorry
    -- Use the angle addition formula
    show cos (105 * real.pi / 180) = (1/2) * (sqrt 2 / 2) - (sqrt 3 / 2) * (sqrt 2 / 2)
    sorry

end cos_105_correct_l591_591375


namespace measure_weights_l591_591939

theorem measure_weights (w1 w3 w7 : Nat) (h1 : w1 = 1) (h3 : w3 = 3) (h7 : w7 = 7) :
  ∃ s : Finset Nat, s.card = 7 ∧ 
    (1 ∈ s) ∧ (3 ∈ s) ∧ (7 ∈ s) ∧
    (4 ∈ s) ∧ (8 ∈ s) ∧ (10 ∈ s) ∧ 
    (11 ∈ s) := 
by
  sorry

end measure_weights_l591_591939


namespace tetrahedron_mistaken_sum_l591_591907

theorem tetrahedron_mistaken_sum :
  let edges := 6
  let vertices := 4
  let faces := 4
  let joe_count := vertices + 1  -- Joe counts one vertex twice
  edges + joe_count + faces = 15 := by
  sorry

end tetrahedron_mistaken_sum_l591_591907


namespace original_average_l591_591258

theorem original_average (n : ℕ) (k : ℕ) (new_avg : ℝ) 
  (h1 : n = 35) 
  (h2 : k = 5) 
  (h3 : new_avg = 125) : 
  (new_avg / k) = 25 :=
by
  rw [h2, h3]
  simp
  sorry

end original_average_l591_591258


namespace least_x_divisible_by_3_l591_591304

theorem least_x_divisible_by_3 : ∃ x : ℕ, (∀ y : ℕ, (2 + 3 + 5 + 7 + y) % 3 = 0 → y = 1) :=
by
  sorry

end least_x_divisible_by_3_l591_591304


namespace geometry_problem_l591_591193

theorem geometry_problem:
  ∀ θ : ℝ,
  let C1 := ∀ x y : ℝ, x ^ 2 + y ^ 2 = 1,
      l := λ ρ θ, ρ * (2 * Real.cos θ - Real.sin θ) = 6,
      stretched_x := λ x, x * Real.sqrt 3,
      stretched_y := λ y, y * 2,
      C2 := ∀ x y : ℝ, (x / Real.sqrt 3) ^ 2 + (y / 2) ^ 2 = 1,
      parametric_C2 := ∃ θ : ℝ, (x = stretched_x (Real.cos θ), y = stretched_y (Real.sin θ)),
      distance_to_l := λ x y, (|2 * x - y - 6| / Real.sqrt 5) in

  -- Prove the Cartesian equation of l
  (∀ ρ θ : ℝ, 2 * ρ * Real.cos θ - ρ * Real.sin θ - 6 = 0) ∧

  -- Prove the parametric equation of C2
  (∀ θ : ℝ, ∃ x y : ℝ, x = Real.sqrt 3 * Real.cos θ ∧ y = 2 * Real.sin θ) ∧

  -- Prove the point P maximizing the distance to l
  (∃ θ : ℝ, point_P = (-Real.sqrt 3 / 2, 1) ∧
     (distance_to_l (-Real.sqrt 3 / 2) 1 = 2 * Real.sqrt 5)) := sorry

end geometry_problem_l591_591193


namespace determine_omega_l591_591143

theorem determine_omega (ω : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f (x) = sin (ω * x + π / 3))
  (h2 : ∀ x, f (π / 6 - x) = f (π / 6 + x)) : (ω = 1 ∨ ω = -5) :=
by
  sorry

end determine_omega_l591_591143


namespace prob_conditional_l591_591729

variable (A B : Prop)

-- Representing probabilities as real numbers as per the conditions
variable P_A : ℝ
variable P_B : ℝ
variable P_A_and_B : ℝ

-- Conditions given
axiom h1 : P_A = 9 / 30
axiom h2 : P_B = 11 / 30
axiom h3 : P_A_and_B = 8 / 30

-- The proof goal
theorem prob_conditional (h1 : P_A = 9 / 30) (h3 : P_A_and_B = 8 / 30) : P_A_and_B / P_A = 8 / 9 := sorry

end prob_conditional_l591_591729


namespace find_angle_degree_l591_591092

theorem find_angle_degree (x : ℝ) (h : 90 - x = 0.4 * (180 - x)) : x = 30 := by
  sorry

end find_angle_degree_l591_591092


namespace combined_weight_difference_l591_591917

def john_weight : ℕ := 81
def roy_weight : ℕ := 79
def derek_weight : ℕ := 91
def samantha_weight : ℕ := 72

theorem combined_weight_difference :
  derek_weight - samantha_weight = 19 :=
by
  sorry

end combined_weight_difference_l591_591917


namespace area_of_BCD_l591_591895

theorem area_of_BCD (S_ABC : ℝ) (a_CD : ℝ) (h_ratio : ℝ) (h_ABC : ℝ) :
  S_ABC = 36 ∧ a_CD = 30 ∧ h_ratio = 0.5 ∧ h_ABC = 12 → 
  (1 / 2) * a_CD * (h_ratio * h_ABC) = 90 :=
by
  intros h
  sorry

end area_of_BCD_l591_591895


namespace pages_per_day_read_l591_591745

theorem pages_per_day_read (start_date : ℕ) (end_date : ℕ) (total_pages : ℕ) (fraction_covered : ℚ) (pages_read : ℕ) (days : ℕ) :
  start_date = 1 →
  end_date = 12 →
  total_pages = 144 →
  fraction_covered = 2/3 →
  pages_read = fraction_covered * total_pages →
  days = end_date - start_date + 1 →
  pages_read / days = 8 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pages_per_day_read_l591_591745


namespace min_value_of_a_plus_2b_l591_591499

theorem min_value_of_a_plus_2b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 2 / a + 1 / b = 1) : a + 2 * b = 4 :=
sorry

end min_value_of_a_plus_2b_l591_591499


namespace train_lost_time_l591_591350

noncomputable def train_time_lost (speed_car : ℕ) (distance : ℕ) : ℕ :=
  let speed_train := speed_car * 3 / 2 in
  let time_car := (distance : ℚ) / speed_car * 60 in
  let time_train := (distance : ℚ) / speed_train * 60 in
  time_car - time_train

theorem train_lost_time :
  train_time_lost 120 75 = 12.5 := by
  sorry

end train_lost_time_l591_591350


namespace total_size_of_prairie_percentage_untouched_of_prairie_l591_591884

variable (ds1 ds2 flood wildfire untouched affected total_size : ℕ)

-- Conditions from the problem
axiom dust_storm_1 : ds1 = 75000
axiom dust_storm_2 : ds2 = 120000
axiom flood_event : flood = 30000
axiom wildfire_event : wildfire = 80000
axiom untouched_area : untouched = 5000
axiom combined_affected_area : affected = 290000

-- Proof statement: Total size of the prairie
theorem total_size_of_prairie :
  total_size = affected + untouched :=
sorry

-- Proof statement: Percentage of untouched prairie
theorem percentage_untouched_of_prairie :
  (untouched * 100 / total_size).toFloat ≈ 1.6949 :=
sorry

end total_size_of_prairie_percentage_untouched_of_prairie_l591_591884


namespace proof_vec_magnitude_l591_591457

variables {V : Type*} [inner_product_space ℝ V]

-- Define the vectors a, b, c
variables {a b c : V}
-- Define the conditions about the magnitudes
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1)
-- Define the conditions about the angles between the vectors
variables (hab : inner a b = 1 / 2) (hbc : inner b c = 1 / 2) (hac : inner a c = 1 / 2)

-- The statement to be proved
def vec_magnitude : Prop :=
  ∥a - b + 2 • c∥ = real.sqrt 5

-- Assertion in Lean
theorem proof_vec_magnitude : vec_magnitude ha hb hc hab hbc hac :=
by
  sorry

end proof_vec_magnitude_l591_591457


namespace triangle_area_third_points_l591_591885

theorem triangle_area_third_points (T : ℝ) 
  (h : ∃ (∆ : Type) [inst : nonempty ∆] [t : Tetrahed ∆], t.is_triangle T) :
  ∃ (M : ℝ), M = (1 / 9) * T :=
begin
  sorry
end

end triangle_area_third_points_l591_591885


namespace compare_abc_l591_591442

noncomputable def a : ℝ := (1 / 3) ^ (2 / 3)
noncomputable def b : ℝ := (1 / 4) ^ (1 / 3)
noncomputable def c : ℝ := Real.log pi / Real.log 3

theorem compare_abc : c > b ∧ b > a :=
by {
  have h_a : 0 < a := by apply Real.rpow_pos_of_pos; linarith,
  have h_b : 0 < b := by apply Real.rpow_pos_of_pos; linarith,

  have : a = (1 / 3) ^ (2 / 3) := rfl,
  have : b = (1 / 4) ^ (1 / 3) := rfl,
  have : c = Real.log pi / Real.log 3 := by {
    field_simp [c],
  },

  have h1 : a < b := by {
    have : (1 / 3) ^ (2 / 3) < (1 / 2) ^ (2 / 3) := by {
      apply Real.rpow_lt_rpow_of_exponent_lt _ _ (div_pos zero_lt_one (by norm_num)),
      linarith,
      linarith,
    },
    exact this,
  },

  have h2 : 1 < c := by {
    have : 3 < pi := by simp [Real.pi_gt3],
    apply Real.log_lt_log zero_lt_three pi_pos this,
  },

  exact ⟨h2, h1⟩,
}

end compare_abc_l591_591442


namespace mr_a_net_loss_is_240_l591_591019

theorem mr_a_net_loss_is_240 (initial_value sale1_percentage_loss sale2_percentage_gain : ℝ) (sale1_value sale2_value net_loss : ℝ) :
  initial_value = 12000 →
  sale1_percentage_loss = 15 / 100 →
  sale2_percentage_gain = 20 / 100 →
  sale1_value = initial_value * (1 - sale1_percentage_loss) →
  sale2_value = sale1_value * (1 + sale2_percentage_gain) →
  net_loss = sale2_value - initial_value →
  net_loss = 240 :=
by
  intro h_initial_value h_sale1_percentage_loss h_sale2_percentage_gain h_sale1_value h_sale2_value h_net_loss
  rw [h_initial_value, h_sale1_percentage_loss, h_sale2_percentage_gain, h_sale1_value, h_sale2_value, h_net_loss]
  -- omitting the proof steps
  sorry

end mr_a_net_loss_is_240_l591_591019


namespace sin_pi_add_alpha_value_l591_591458

noncomputable def hypotenuse (x y : ℝ) : ℝ := Real.sqrt (x * x + y * y)

theorem sin_pi_add_alpha_value
  (x y : ℝ) (r : ℝ)
  (hx : x = Real.sqrt 5)
  (hy : y = -2)
  (hr : r = hypotenuse (Real.sqrt 5) (-2)) :
  sin (π + atan2 y x) = 2 / 3 := by
  sorry

end sin_pi_add_alpha_value_l591_591458


namespace optimal_direction_l591_591041

-- Define the conditions as hypotheses
variables (a : ℝ) (V_first V_second : ℝ) (d : ℝ)
variable (speed_rel : V_first = 2 * V_second)
variable (dist : d = a)

-- Create a theorem statement for the problem
theorem optimal_direction (H : d = a) (vel_rel : V_first = 2 * V_second) : true := 
  sorry

end optimal_direction_l591_591041


namespace sum_first_8_terms_l591_591438

-- Define the arithmetic sequence
def is_arith_seq (a : ℕ → ℤ) : Prop :=
∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

-- Given conditions
variables {a : ℕ → ℤ} (h_arith : is_arith_seq a)

-- Specific conditions from the problem
variables (h_a4 : a 4 = 7) (h_sum57 : a 5 + a 7 = 26)

-- The goal is to prove that the sum of the first 8 terms equals 68
theorem sum_first_8_terms : (∑ i in finset.range 8, a i) = 68 :=
sorry

end sum_first_8_terms_l591_591438


namespace pencils_to_sell_l591_591721

/--
A store owner bought 1500 pencils at $0.10 each. 
Each pencil is sold for $0.25. 
He wants to make a profit of exactly $100. 
Prove that he must sell 1000 pencils to achieve this profit.
-/
theorem pencils_to_sell (total_pencils : ℕ) (cost_per_pencil : ℝ) (selling_price_per_pencil : ℝ) (desired_profit : ℝ)
  (h1 : total_pencils = 1500)
  (h2 : cost_per_pencil = 0.10)
  (h3 : selling_price_per_pencil = 0.25)
  (h4 : desired_profit = 100) :
  total_pencils * cost_per_pencil + desired_profit = 1000 * selling_price_per_pencil :=
by
  -- Since Lean code requires some proof content, we put sorry to skip it.
  sorry

end pencils_to_sell_l591_591721


namespace Buratino_math_problem_l591_591934

theorem Buratino_math_problem (x : ℝ) : 4 * x + 15 = 15 * x + 4 → x = 1 :=
by
  intro h
  sorry

end Buratino_math_problem_l591_591934


namespace isabella_total_gallons_purchased_l591_591763

theorem isabella_total_gallons_purchased :
  ∀ (kim_gallons: ℕ) (isa_per_gal_discount_perc: ℚ),
  kim_gallons = 20 →
  isa_per_gal_discount_perc = 108.57142857142861 →
  ∃ (isa_gallons: ℕ), isa_gallons = 21 :=
begin
  intros kim_gallons isa_per_gal_discount_perc kim_gallons_equ isa_per_gal_discount_perc_equ,
  have h_kim_discounted := kim_gallons - 6,
  have h_kim_total_discount := h_kim_discounted * 0.10,
  have h_isa_total_discount := h_kim_total_discount * isa_per_gal_discount_perc / 100,
  have h_isa_discounted := h_isa_total_discount / 0.10,
  have h_isa_gallons := h_isa_discounted + 6,
  use h_isa_gallons,
  exact rfl,
end

end isabella_total_gallons_purchased_l591_591763


namespace systematic_sampling_twentieth_group_number_l591_591301

theorem systematic_sampling_twentieth_group_number 
  (total_students : ℕ) 
  (total_groups : ℕ) 
  (first_group_number : ℕ) 
  (interval : ℕ) 
  (n : ℕ) 
  (drawn_number : ℕ) :
  total_students = 400 →
  total_groups = 20 →
  first_group_number = 11 →
  interval = 20 →
  n = 20 →
  drawn_number = 11 + 20 * (n - 1) →
  drawn_number = 391 :=
by
  sorry

end systematic_sampling_twentieth_group_number_l591_591301


namespace cone_sphere_surface_area_l591_591716

theorem cone_sphere_surface_area :
  ∀ (r l R : ℝ), r = real.sqrt 3 → l = 2 → 
  (R ^ 2 = (R - 1) ^ 2 + r ^ 2) → 4 * real.pi * R ^ 2 = 16 * real.pi :=
by
  intros r l R h_r h_l h_R
  sorry

end cone_sphere_surface_area_l591_591716


namespace Grandfather_Huang_total_payment_l591_591740

noncomputable def senior_ticket_price : ℝ := 7
def regular_ticket_price : ℝ := senior_ticket_price / 0.7
def child_ticket_price : ℝ := 0.6 * regular_ticket_price
def total_cost : ℝ := 2 * senior_ticket_price + 2 * regular_ticket_price + 2 * child_ticket_price

theorem Grandfather_Huang_total_payment : total_cost = 46 := by
  sorry

end Grandfather_Huang_total_payment_l591_591740


namespace range_of_a_l591_591464

def quadratic_function (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, 1 ≤ x ∧ x ≤ y → quadratic_function a x ≤ quadratic_function a y) : a ≤ 1 :=
sorry

end range_of_a_l591_591464


namespace final_score_correct_l591_591010

def innovation_score : ℕ := 88
def comprehensive_score : ℕ := 80
def language_score : ℕ := 75

def weight_innovation : ℕ := 5
def weight_comprehensive : ℕ := 3
def weight_language : ℕ := 2

def final_score : ℕ :=
  (innovation_score * weight_innovation + comprehensive_score * weight_comprehensive +
   language_score * weight_language) /
  (weight_innovation + weight_comprehensive + weight_language)

theorem final_score_correct :
  final_score = 83 :=
by
  -- proof goes here
  sorry

end final_score_correct_l591_591010


namespace find_b_value_l591_591130

noncomputable def ellipse_b_value (a b : ℝ) (F1 F2 P : ℝ × ℝ) (area : ℝ) : Prop :=
  let c := Real.sqrt (a^2 - b^2)
  in P ∈ {p | (p.1 / a)^2 + (p.2 / b)^2 = 1} ∧
     dist P F1 = dist P F2 ∧
     ∠ P F1 F2 = 90 ∧
     1 / 2 * dist P F1 * dist P F2 = area → b = 3

theorem find_b_value (a b : ℝ) (F1 F2 P : ℝ × ℝ)
  (h1 : a > b ∧ b > 0)
  (h2 : P ∈ {p | (p.1 / a)^2 + (p.2 / b)^2 = 1})
  (h3 : (∠ P F1 F2) = 90)
  (h4 : 1 / 2 * (dist P F1) * (dist P F2) = 9) :
  b = 3 :=
sorry

end find_b_value_l591_591130


namespace fraction_evaluation_l591_591376

theorem fraction_evaluation :
  (2 + 4 - 8 + 16 + 32 - 64 + 128 : ℚ) / (4 + 8 - 16 + 32 + 64 - 128 + 256) = 1 / 2 :=
by
  sorry

end fraction_evaluation_l591_591376


namespace quadrilateral_AD_lt_BC_l591_591888

noncomputable def quadrilateral (A B C D: Type) [metric_space A] :=
  ∃ (a b c d: A), (angle a b c = angle b a c) ∧ (angle d a c > angle c d a)

theorem quadrilateral_AD_lt_BC (A B C D: Type) [metric_space A] (quad : quadrilateral A B C D):
  ∃ (a b c d: A), (angle a b c = angle b a c) ∧ (angle d a c > angle c d a) → (dist a d < dist b c) :=
begin
  sorry
end

end quadrilateral_AD_lt_BC_l591_591888


namespace max_a2009_value_l591_591495

theorem max_a2009_value :
  ∃ (a : Fin 2009 → ℝ), 
    a 0 = 3 ∧ 
    (∀ n : ℕ, 1 ≤ n < 2009 → 
      a (Fin.succ (Fin.ofNat n))^2 - 
      (a (Fin.ofNat n) / 2009 + 1 / a (Fin.ofNat n)) *
      a (Fin.succ (Fin.ofNat n)) +
      1 / 2009 = 0) ∧
    (∀ b : Fin 2009 → ℝ, 
      b 0 = 3 ∧ 
      (∀ n : ℕ, 1 ≤ n < 2009 →
        b (Fin.succ (Fin.ofNat n))^2 - 
        (b (Fin.ofNat n) / 2009 + 1 / b (Fin.ofNat n)) *
        b (Fin.succ (Fin.ofNat n)) +
        1 / 2009 = 0) → 
      b (Fin.ofNat 2008) ≤ a (Fin.ofNat 2008)) :=
begin
  use λ n, ite (n = Fin.ofNat 0) 3 (λ m, sorry), -- Placeholders for sequence definition
  sorry -- Proof required
end

end max_a2009_value_l591_591495


namespace revenue_fall_percentage_l591_591262

theorem revenue_fall_percentage:
  let oldRevenue := 72.0
  let newRevenue := 48.0
  (oldRevenue - newRevenue) / oldRevenue * 100 = 33.33 :=
by
  let oldRevenue := 72.0
  let newRevenue := 48.0
  sorry

end revenue_fall_percentage_l591_591262


namespace constant_term_in_expansion_l591_591772

theorem constant_term_in_expansion :
  let x : ℂ := complex.i in
  let term_expansion (r : ℕ) := (x + 1 / (2 * x)) ^ 6 in
  let constant_term := term_expansion 3 in
  constant_term = 5 / 2 := 
by
  sorry

end constant_term_in_expansion_l591_591772


namespace police_emergency_number_has_prime_divisor_gt_7_l591_591028

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := sorry

end police_emergency_number_has_prime_divisor_gt_7_l591_591028


namespace arithmetic_geometric_sequences_T_sum_S_sum_l591_591437

def a (n : ℕ) : ℤ := 2 * n - 1
def b (n : ℕ) : ℤ := 3^n
def d (n : ℕ) : ℚ := 2 / (a n * a (n + 1))
def T (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)
def c (n : ℕ) : ℤ := a n * b n
noncomputable def S (n : ℕ) : ℤ := 3 + (n - 1) * 3^(n + 1)

theorem arithmetic_geometric_sequences (a_2 a_3 a_4 : ℤ) (ha_2 : a 2 = a_2) (ha_3 : a 3 = a_3) (ha_4 : a 4 = a_4)
  (cond1 : a_2 + a_3 + a_4 = 15)
  (cond2 : a_2 > 0) (cond3 : a_2, a_3 + 4, a_4 + 20 are first three terms of a geometric sequence):
  a_n = 2 * n - 1 ∧ b_n = 3^n :=
sorry

theorem T_sum (n : ℕ) : T n = (2 * n) / (2 * n + 1) :=
sorry

theorem S_sum (n : ℕ) : S n = 3 + (n - 1) * 3^(n + 1) :=
sorry

end arithmetic_geometric_sequences_T_sum_S_sum_l591_591437


namespace planes_perpendicular_l591_591297

variables {z y : ℝ}

def vec_u : ℝ × ℝ × ℝ := (3, -1, z)
def vec_v : ℝ × ℝ × ℝ := (-2, -y, 1)

theorem planes_perpendicular (h : vec_u.1 * vec_v.1 + vec_u.2 * vec_v.2 + vec_u.3 * vec_v.3 = 0) : y + z = 6 :=
by sorry

end planes_perpendicular_l591_591297


namespace sequence_a_n_general_formula_l591_591119

def a_n (n : ℕ) (h : n > 0) : ℚ := 2 - 1/(2^n)

def S (n : ℕ) (a : ℕ → ℚ) : ℚ :=
  if h : n > 0 then
    2 * ↑n + 1 - a n h
  else
    0

theorem sequence_a_n (n : ℕ) (h : n > 0) : S n (a_n) + a_n n h = 2 * ↑n + 1 :=
by sorry

theorem general_formula (n : ℕ) (h : n > 0) : a_n n h = 2 - 1/(2^n) :=
by sorry

end sequence_a_n_general_formula_l591_591119


namespace solution_system_l591_591633

theorem solution_system (a : ℝ) : 
    (a = real.sqrt 2 ∨ a = -real.sqrt 2 → 
     ∃! (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) ∧ 
    (a = 1 ∨ a = -1 → 
     ∃! (x y : ℝ) in [{p : ℝ × ℝ | (p.fst = 0 ↔ p.snd^2 = 1 - a^2)}, {p : ℝ × ℝ | p.fst ≠ 0}], 
     x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) :=
sorry

end solution_system_l591_591633


namespace group_arrangements_l591_591191

-- Define the parameters
def men : ℕ := 4
def women : ℕ := 5
def group_size : ℕ := 3

-- The main theorem
theorem group_arrangements : 
  (∃ f : fin 3 → (fin men × fin women), 
    (∀ i, ∃ m, f i = (m, _)) ∧ 
    (∀ i, ∃ w, f i = (_, w)) ∧ 
    (∀ i j, i ≠ j → 
      ∃ m_w, (f i = m_w ∧ f j ≠ m_w) ∧ (f i ≠ f j))) → 
  (number of ways = 180) :=
sorry

end group_arrangements_l591_591191


namespace largest_prime_factor_sequence_divides_sum_l591_591700

theorem largest_prime_factor_sequence_divides_sum (d₀ d₁ d₂ d₃ : ℕ) (h₀ : d₀ < 10) (h₁ : d₁ < 10) (h₂ : d₂ < 10) (h₃ : d₃ < 10) (n : ℕ) (hn : n > 0) : 
  let T := n * (1000 * d₀ + 100 * d₁ + 10 * d₂ + d₃ + 1000 * d₃ + 100 * d₀ + 10 * d₁ + d₂ + 1000 * d₂ + 100 * d₃ + 10 * d₀ + d₁ + 1000 * d₁ + 100 * d₂ + 10 * d₃ + d₀)
  in 11 ∣ T :=
by
  sorry

end largest_prime_factor_sequence_divides_sum_l591_591700


namespace trigonometric_identity_example_l591_591000

theorem trigonometric_identity_example :
  (cos (5 * Real.pi / 8) * cos (Real.pi / 8) + sin (5 * Real.pi / 8) * sin (Real.pi / 8) = 0) :=
by
  -- Apply the cosine of angle difference identity
  -- cos (5π/8 - π/8) = cos π/2 = 0
  sorry

end trigonometric_identity_example_l591_591000


namespace expectation_X_is_one_probability_A_at_least_4_out_of_5_l591_591293

open Probability

noncomputable def classAProb := 3/5
noncomputable def classBProb := 1/2

def X_distribution : PMF ℚ :=
  PMF.of_fintype (
    [(-10, 1/5), (0, 1/2), (10, 3/10)].to_finset
  ) sorry

def expectation_X : ℚ :=
  expectation X_distribution

theorem expectation_X_is_one :
  expectation_X = 1 :=
sorry

def rounds (i : ℕ) (outcomes : Fin i → ℚ) :=
(fin_fun (λ n, if outcomes n = 10 then 1 else 0))

def probability_A_wins (n : ℕ) :=
let p_A := (3/5) * (1/2) + (2/5) * (1/2) in
(probability (binomial n (p_A) ≥ 4) with sorry)

theorem probability_A_at_least_4_out_of_5 : probability_A_wins 5 = 2304 / 3125 :=
sorry

end expectation_X_is_one_probability_A_at_least_4_out_of_5_l591_591293


namespace milk_left_is_correct_l591_591557

noncomputable def lily_initial_milk : ℚ := 5
noncomputable def milk_given_to_james : ℚ := 11 / 4

theorem milk_left_is_correct : lily_initial_milk - milk_given_to_james = 9 / 4 :=
by
  sorry

end milk_left_is_correct_l591_591557


namespace sum_of_elements_in_M_l591_591777

theorem sum_of_elements_in_M : 
  let M := {a : ℤ | 
               ∃ (x y z t : ℤ), 
               a = (x + y + z) / t ∧ 
               3^x + 3^y + 3^z = 3^t ∧ 
               x ∈ ℤ ∧ y ∈ ℤ ∧ z ∈ ℤ ∧ t ∈ ℤ} 
  in M.sum = 12 :=
by
  let M := {a : ℤ | 
               ∃ (x y z t : ℤ), 
               a = (x + y + z) / t ∧ 
               3^x + 3^y + 3^z = 3^t ∧ 
               x ∈ ℤ ∧ y ∈ ℤ ∧ z ∈ ℤ ∧ t ∈ ℤ}
  have hM : M = {0, 2, 4, 6} := by sorry
  rw [hM]
  norm_num

end sum_of_elements_in_M_l591_591777


namespace cost_of_chairs_l591_591568

-- Given conditions
def total_spent : ℕ := 56
def cost_of_table : ℕ := 34
def number_of_chairs : ℕ := 2

-- The target definition
def cost_of_one_chair : ℕ := 11

-- Statement to prove
theorem cost_of_chairs (N : ℕ) (T : ℕ) (C : ℕ) (x : ℕ) (hN : N = total_spent) (hT : T = cost_of_table) (hC : C = number_of_chairs) : x = cost_of_one_chair ↔ N - T = C * x :=
by
  sorry

end cost_of_chairs_l591_591568


namespace problem_1_problem_2_problem_3_problem_4_l591_591161

noncomputable def a := (λ x, (Real.sin (Real.pi * x / 2), Real.sin (Real.pi / 3)))
noncomputable def b := (λ x, (Real.cos (Real.pi * x / 2), Real.cos (Real.pi / 3)))
noncomputable def f := (λ x, Real.sin (Real.pi * x / 2 - Real.pi / 3))

theorem problem_1 (x : ℝ) (h : (a x).fst * (b x).snd = (a x).snd * (b x).fst) : 
  Real.sin (Real.pi * x / 2 - Real.pi / 3) = 0 := sorry

theorem problem_2 (k : ℤ) : 
  ∃ (x : ℝ), f x = 0 → x = (5 : ℝ) / 3 + (2 : ℝ) * k := sorry

theorem problem_3 : 
  (f 1 + f 2 + f 3 + List.sum (List.map f (List.range 2014).tail)) = (1 : ℝ) / 2 := sorry

theorem problem_4 (A B C: ℝ) (h1: 0 < A) (h2: A < B) (h3: B < Real.pi) (h4: A + B + C = Real.pi) 
  (h5: f (4 * A / Real.pi) = 1 / 2) (h6: f (4 * B / Real.pi) = 1 / 2) : 
  Real.sin B / Real.sin C = (Real.sqrt 6 + Real.sqrt 2) / 2 := sorry

end problem_1_problem_2_problem_3_problem_4_l591_591161


namespace B_completes_remaining_work_in_2_days_l591_591679

theorem B_completes_remaining_work_in_2_days 
  (A_work_rate : ℝ) (B_work_rate : ℝ) (total_work : ℝ) 
  (A_days_to_complete : A_work_rate = 1 / 2) 
  (B_days_to_complete : B_work_rate = 1 / 6) 
  (combined_work_1_day : A_work_rate + B_work_rate = 2 / 3) : 
  (total_work - (A_work_rate + B_work_rate)) / B_work_rate = 2 := 
by
  sorry

end B_completes_remaining_work_in_2_days_l591_591679


namespace jacobs_toy_bin_l591_591872

theorem jacobs_toy_bin :
  ∃ (R : ℕ), R + (R + 7) + (R + 14) = 75 ∧ R = 18 :=
by
  use 18
  split
  · simp
  · rfl

end jacobs_toy_bin_l591_591872


namespace police_emergency_number_has_prime_gt_7_l591_591031

theorem police_emergency_number_has_prime_gt_7 (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
sorry

end police_emergency_number_has_prime_gt_7_l591_591031


namespace butterflies_equal_distribution_l591_591887

theorem butterflies_equal_distribution (N : ℕ) : (∃ t : ℕ, 
    (N - t) % 8 = 0 ∧ (N - t) / 8 > 0) ↔ ∃ k : ℕ, N = 45 * k :=
by sorry

end butterflies_equal_distribution_l591_591887


namespace triangle_smallest_angle_l591_591513

-- Define a function to calculate the smallest angle in a triangle
def smallest_angle (a b c : ℝ) : ℝ :=
  if a < b ∧ a < c then a else if b < c then b else c

theorem triangle_smallest_angle {α β γ : ℝ} 
  (h1 : α = 40)
  (h2 : β = 35)
  (h3 : γ = 105)
  (h4 : α + β + γ = 180)
  (h_ratio : β / α = 1 / 3) :
  smallest_angle α β γ = 35 := by
  sorry

end triangle_smallest_angle_l591_591513


namespace sum_alternating_sums_l591_591780

noncomputable def set_alternating_sum (s : Finset ℕ) : ℤ := 
  if s = ∅ then 0 
  else
    let l := s.to_list in
    let l_sorted := l.sort (≥) in
    (l_sorted.zipWithIndex.map (λ ⟨x, i⟩, if i % 2 = 0 then ↑x else -↑x)).sum

theorem sum_alternating_sums (n : ℕ) :
  ∑ s in (Finset.powerset (Finset.range (n+1))).filter (λ s, s ≠ ∅), set_alternating_sum s = n * 2^(n - 1) :=
sorry

end sum_alternating_sums_l591_591780


namespace system_of_equations_correct_l591_591892

theorem system_of_equations_correct (x y : ℤ) 
  (h1 : 9 * x - 5 = y)
  (h2 : 6 * x + 4 = y) : 
  (9 * x - 5 = y) ∧ (6 * x + 4 = y) :=
by {
  split,
  exact h1,
  exact h2,
}

end system_of_equations_correct_l591_591892


namespace difference_thursday_tuesday_l591_591941

-- Define the amounts given on each day
def amount_tuesday : ℕ := 8
def amount_wednesday : ℕ := 5 * amount_tuesday
def amount_thursday : ℕ := amount_wednesday + 9

-- Problem statement: prove that the difference between Thursday's and Tuesday's amount is $41
theorem difference_thursday_tuesday : amount_thursday - amount_tuesday = 41 := by
  sorry

end difference_thursday_tuesday_l591_591941


namespace find_k_and_center_l591_591157

noncomputable def line_l (ρ θ : ℝ) : Prop := ρ * sin(θ - π / 4) = 4

noncomputable def circle_C (ρ θ k : ℝ) : Prop := ρ = 2 * k * cos(θ + π / 4)

theorem find_k_and_center (k : ℝ) (hk : k ≠ 0) :
  (∃ x y : ℝ, ∀ ρ θ : ℝ, line_l ρ θ → circle_C ρ θ k → 
  min_dist := (|k + 4| - |k| = 2)) ) ->
  k = -1 ∧ ∃ x y : ℝ, (x = - √2 / 2 ∧ y = √2 / 2) :=
begin
  sorry
end

end find_k_and_center_l591_591157


namespace mode_and_median_of_book_counts_l591_591694

-- Define the data given in the problem
def book_counts : List ℕ := [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5]

-- Function to calculate mode
def mode (l : List ℕ) : ℕ :=
  let freq := l.groupBy id
  let maxGroup := List.maximumBy (List.length ∘ snd) freq
  maxGroup.1.getD 0

-- Function to calculate median
def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  if sorted.length % 2 = 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

theorem mode_and_median_of_book_counts :
  mode book_counts = 3 ∧ median book_counts = 3 := by
  sorry

end mode_and_median_of_book_counts_l591_591694


namespace goose_eggs_laied_l591_591738

theorem goose_eggs_laied (z : ℕ) (hatch_rate : ℚ := 2 / 3) (first_month_survival_rate : ℚ := 3 / 4) 
  (first_year_survival_rate : ℚ := 2 / 5) (geese_survived_first_year : ℕ := 126) :
  (hatch_rate * z) = 420 ∧ (first_month_survival_rate * 315 = 315) ∧ (first_year_survival_rate * 315 = 126) →
  z = 630 :=
by
  sorry

end goose_eggs_laied_l591_591738


namespace min_value_of_f_l591_591096

noncomputable def f (x : ℝ) := x + 2 * Real.cos x

theorem min_value_of_f :
  ∀ (x : ℝ), -Real.pi / 2 ≤ x ∧ x ≤ 0 → f x ≥ f (-Real.pi / 2) :=
by
  intro x hx
  -- conditions are given, statement declared, but proof is not provided
  sorry

end min_value_of_f_l591_591096


namespace initial_percentage_decrease_l591_591276

theorem initial_percentage_decrease (x : ℝ) (P : ℝ) (h₀ : P > 0)
  (initial_decrease : ∀ (x : ℝ), P * (1 - x / 100) * 1.3 = P * 1.04) :
  x = 20 :=
by 
  sorry

end initial_percentage_decrease_l591_591276


namespace years_to_double_approx_l591_591179

def initial_capacity : ℝ := 0.6
def final_capacity : ℝ := 6150
def years_passed : ℕ := 2050 - 2000
def growth_formula : ℝ := final_capacity / initial_capacity

noncomputable def years_to_double : ℝ := 50 / Real.log2 growth_formula

theorem years_to_double_approx :
  Real.log2 growth_formula ≠ 0 → 
  years_to_double ≈ 3.76 :=
by
  intros h
  simp [years_to_double, growth_formula, initial_capacity, final_capacity, years_passed]
  sorry

end years_to_double_approx_l591_591179


namespace coefficient_of_monomial_l591_591261

-- Definition of the monomial
def monomial : ℝ := -2 * Real.pi * a * b^2

-- Theorem statement for the coefficient of the monomial
theorem coefficient_of_monomial : coefficient monomial = -2 * Real.pi :=
sorry

end coefficient_of_monomial_l591_591261


namespace sin_alpha_plus_3pi_over_2_l591_591487

theorem sin_alpha_plus_3pi_over_2
  (α : ℝ)
  (h : cos (α + π) = - (2 / 3)) :
  sin (α + (3 * π) / 2) = - (2 / 3) :=
by
  sorry

end sin_alpha_plus_3pi_over_2_l591_591487


namespace surface_area_of_tunneled_cube_l591_591692

-- Definition of the initial cube and its properties.
def cube (side_length : ℕ) := side_length * side_length * side_length

-- Initial side length of the large cube
def large_cube_side : ℕ := 12

-- Each small cube side length
def small_cube_side : ℕ := 3

-- Number of small cubes that fit into the large cube
def num_small_cubes : ℕ := (cube large_cube_side) / (cube small_cube_side)

-- Number of cubes removed initially
def removed_cubes : ℕ := 27

-- Number of remaining cubes after initial removal
def remaining_cubes : ℕ := num_small_cubes - removed_cubes

-- Surface area of each unmodified small cube
def small_cube_surface : ℕ := 54

-- Additional surface area due to removal of center units
def additional_surface : ℕ := 24

-- Surface area of each modified small cube
def modified_cube_surface : ℕ := small_cube_surface + additional_surface

-- Total surface area before adjustment for shared faces
def total_surface_before_adjustment : ℕ := remaining_cubes * modified_cube_surface

-- Shared surface area to be subtracted
def shared_surface : ℕ := 432

-- Final surface area of the resulting figure
def final_surface_area : ℕ := total_surface_before_adjustment - shared_surface

-- Theorem statement
theorem surface_area_of_tunneled_cube : final_surface_area = 2454 :=
by {
  -- Proof required here
  sorry
}

end surface_area_of_tunneled_cube_l591_591692


namespace box_dimension_reduction_not_always_possible_l591_591993

-- Define the conditions and the main theorem

theorem box_dimension_reduction_not_always_possible
  (P : set (ℝ × ℝ × ℝ)) -- Set of parallelepipeds with dimensions (a, b, c)
  (B : ℝ × ℝ × ℝ) -- Dimensions of the box (A, B, C)
  (h1 : ∀ p ∈ P, p.1 ≤ B.1 ∧ p.2 ≤ B.2 ∧ p.3 ≤ B.3) -- Condition: All parallelepipeds fit initially
  (h2 : ∀ p ∈ P, ∃ p', p'.1 < p.1 ∧ p'.2 = p.2 ∧ p'.3 = p.3 ∨ p'.1 = p.1 ∧ p'.2 < p.2 ∧ p'.3 = p.3 ∨ p'.1 = p.1 ∧ p'.2 = p.2 ∧ p'.3 < p.3) -- Condition: One dimension is reduced
  : ¬(∀ B', B'.1 < B.1 ∨ B'.2 < B.2 ∨ B'.3 < B.3 → (∀ p ∈ P, p.1 ≤ B'.1 ∧ p.2 ≤ B'.2 ∧ p.3 ≤ B'.3)) := -- Conclusion: It's not always possible to reduce one dimension of the box
  sorry

end box_dimension_reduction_not_always_possible_l591_591993


namespace distance_pole_ratio_l591_591946

noncomputable def distance_ratio (x : ℝ) (d1 d2 : ℝ) : ℝ := 
  if d1 > d2 then d1 / d2 else d2 / d1

theorem distance_pole_ratio :
  ∃ (x : ℝ), x > 0 → (
    let y := x in
    let d1 := 3*x + 10 in
    let d2 := x - 10 in
    distance_ratio x d1 d2 = 7) :=
begin
  use 20,
  intro hx,
  simp only [distance_ratio],
  split_ifs,
  { exact 7 },
  { solve_by_elim },
end

#eval distance_pole_ratio
#eval distance_ratio 20 (3 * 20 + 10) (20 - 10) -- This should evaluate to 7

end distance_pole_ratio_l591_591946


namespace unit_prices_max_helmets_A_l591_591668

open Nat Real

-- Given conditions
variables (x y : ℝ)
variables (m : ℕ)

def wholesale_price_A := 30
def wholesale_price_B := 20
def price_difference := 15
def revenue_A := 450
def revenue_B := 600
def total_helmets := 100
def budget := 2350

-- Part 1: Prove the unit prices of helmets A and B
theorem unit_prices :
  ∃ (price_A price_B : ℝ), 
    (price_A = price_B + price_difference) ∧ 
    (revenue_B / price_B = 2 * revenue_A / price_A) ∧
    (price_B = 30) ∧
    (price_A = 45) :=
by
  sorry

-- Part 2: Prove the maximum number of helmets of type A that can be purchased
theorem max_helmets_A :
  ∃ (m : ℕ), 
    (30 * m + 20 * (total_helmets - m) ≤ budget) ∧
    (m ≤ 35) :=
by
  sorry

end unit_prices_max_helmets_A_l591_591668


namespace find_angle_B_find_range_f_l591_591209

variable (a b c : ℝ) (A B C : ℝ)
variable (f : ℝ → ℝ)

theorem find_angle_B (h1 : a + c = 1 + real.sqrt 3) (h2 : b = 1) (h3 : real.sin C = real.sqrt 3 * real.sin A) :
  B = real.pi / 6 :=
sorry

theorem find_range_f (B : ℝ) (hB : B = real.pi / 6) :
  (∀ x ∈ set.Icc 0 (real.pi / 2), 
     f x = 2 * real.sin (2 * x + B) + 4 * real.cos x ^ 2) →
  function.surjective (λ y, ∃ x ∈ set.Icc 0 (real.pi / 2), f x = y) :=
sorry

end find_angle_B_find_range_f_l591_591209


namespace rectangle_dimensions_l591_591556

theorem rectangle_dimensions (x : ℝ) (h : 4 * x * x = 120) : x = Real.sqrt 30 ∧ 4 * x = 4 * Real.sqrt 30 :=
by
  sorry

end rectangle_dimensions_l591_591556


namespace trigonometric_identity1_trigonometric_identity2_l591_591809

theorem trigonometric_identity1 (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin (Real.pi - θ) + Real.cos (θ - Real.pi)) / (Real.sin (θ + Real.pi) + Real.cos (θ + Real.pi)) = -1/3 :=
by
  sorry

theorem trigonometric_identity2 (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ) = 4/5 :=
by
  sorry

end trigonometric_identity1_trigonometric_identity2_l591_591809


namespace quadratic_roots_squared_sum_l591_591169

theorem quadratic_roots_squared_sum :
  ∀ x1 x2 : ℝ, (x1^2 - 3 * x1 + 1 = 0) → (x2^2 - 3 * x2 + 1 = 0) → (x1 ≠ x2) → (x1^2 + x2^2 = 7) :=
by
  intro x1 x2 h1 h2 hneq
  let s := x1 + x2
  let p := x1 * x2
  have h_sum : s = 3 := 
    begin
      sorry,
    end
  have h_prod : p = 1 := 
    begin
      sorry,
    end
  have h_res : x1^2 + x2^2 = s^2 - 2 * p := 
    begin
      sorry,
    end
  rw [h_res, h_sum, h_prod]
  norm_num

end quadratic_roots_squared_sum_l591_591169


namespace cos_distance_Q_R_l591_591064

def cos_similarity (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 * x2 + y1 * y2) / (Real.sqrt (x1^2 + y1^2) * Real.sqrt (x2^2 + y2^2))

def cos_distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  1 - cos_similarity x1 y1 x2 y2

theorem cos_distance_Q_R (α β : ℝ) 
  (h1 : cos_distance (Real.cos α) (Real.sin α) (Real.cos β) (Real.sin β) = 1 / 3)
  (h2 : Real.tan α * Real.tan β = 1 / 7 ) :
  cos_distance (Real.cos β) (Real.sin β) (Real.cos α) (-Real.sin α) = 1 / 2 :=
sorry

end cos_distance_Q_R_l591_591064


namespace average_of_second_and_third_smallest_l591_591259

def avg_of_two_numbers (a b : ℕ) : ℚ := (a + b) / 2

theorem average_of_second_and_third_smallest (a b c d e : ℕ)
  (h_distinct : list.nodup [a, b, c, d, e])
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (h_average_five : (a + b + c + d + e) / 5 = 5)
  (h_max_diff : max a (max b (max c (max d e))) 
             - min a (min b (min c (min d e))) = 14) :
  avg_of_two_numbers (list.nth_le (list.sort nat.lt [a, b, c, d, e]) 1 sorry)
                     (list.nth_le (list.sort nat.lt [a, b, c, d, e]) 2 sorry)
  = 2.5 := 
sorry

end average_of_second_and_third_smallest_l591_591259


namespace points_on_axes_set_notation_l591_591631

-- Defining the set of points on the coordinate axes in Cartesian coordinate system
def points_on_axes : set (ℝ × ℝ) := {p | p.fst * p.snd = 0}

-- Prove the statement that the set of coordinates of all points on the coordinate axes can be represented as points where the product of their coordinates is zero.
theorem points_on_axes_set_notation :
  ∀ p : ℝ × ℝ, p ∈ points_on_axes ↔ (∃ x y : ℝ, p = (x, y) ∧ x * y = 0) :=
sorry

end points_on_axes_set_notation_l591_591631


namespace sequence_can_be_both_arithmetic_and_geometric_l591_591754

-- Definitions based on the given conditions
def is_arithmetic_sequence (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, s (n + 1) - s n = s 1 - s 0

def is_geometric_sequence (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, s (n + 1) / s n = s 1 / s 0

-- Sequence definition example: 3, 9, ..., 729
def seq : ℕ → ℕ
| 0       := 3
| 1       := 9
| 2       := 729 -- insufficient sequence definition, illustrative purposes only
| (n+3) := sorry -- Need more information

-- Theorem that states that given the sequence, we can say it's both arithmetic and geometric by conditions stated
theorem sequence_can_be_both_arithmetic_and_geometric :
  ( is_arithmetic_sequence seq ∨ ¬is_arithmetic_sequence seq ) ∧
  ( is_geometric_sequence seq ∨ ¬is_geometric_sequence seq ) := 
  sorry

end sequence_can_be_both_arithmetic_and_geometric_l591_591754


namespace geometric_body_view_circle_l591_591502

theorem geometric_body_view_circle (P : Type) (is_circle : P → Prop) (is_sphere : P → Prop)
  (is_cylinder : P → Prop) (is_cone : P → Prop) (is_rectangular_prism : P → Prop) :
  (∀ x, is_sphere x → is_circle x) →
  (∃ x, is_cylinder x ∧ is_circle x) →
  (∃ x, is_cone x ∧ is_circle x) →
  ¬ (∃ x, is_rectangular_prism x ∧ is_circle x) :=
by
  intros h_sphere h_cylinder h_cone h_rectangular_prism
  sorry

end geometric_body_view_circle_l591_591502


namespace dot_product_of_unit_vectors_l591_591844

variables {V : Type*} [inner_product_space ℝ V]

def is_unit_vector (v : V) : Prop := ∥v∥ = 1

theorem dot_product_of_unit_vectors (a b : V) (ha : is_unit_vector a) (hb : is_unit_vector b) :
  inner (2 • a + b) (2 • a - b) = 3 :=
by {
  sorry
}

end dot_product_of_unit_vectors_l591_591844


namespace track_width_radii_l591_591718

theorem track_width_radii (r1 r2 : ℝ) 
  (h1 : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi)
  (h2 : 2 * Real.pi * r2 = 40 * Real.pi) :
  r2 = 20 ∧ r1 = 30 ∧ (r1 - r2) = 10 := 
by {
  sorry,
}

end track_width_radii_l591_591718


namespace chickens_initial_count_l591_591043

-- Definitions of initial counts and daily loss
def initial_turkeys := 200
def initial_guinea_fowls := 80
def daily_loss_chickens := 20
def daily_loss_turkeys := 8
def daily_loss_guinea_fowls := 5
def days_in_a_week := 7
def remaining_birds := 349

-- Initial chickens to be proven
def initial_chickens := 300

-- Lean 4 statement to prove the initial chickens count
theorem chickens_initial_count : 
  ∃ (chickens : ℕ), chickens = initial_chickens ∧ 
  (remaining_birds = 
    initial_turkeys - daily_loss_turkeys * days_in_a_week +
    initial_guinea_fowls - daily_loss_guinea_fowls * days_in_a_week + 
    chickens - daily_loss_chickens * days_in_a_week) :=
by
  -- Skipping the proof
  exact ⟨initial_chickens, rfl⟩
  sorry

end chickens_initial_count_l591_591043


namespace parallel_intersects_perpendicular_bisector_l591_591108

theorem parallel_intersects_perpendicular_bisector
  {A B C P : Point} (circumcircle_k : Circle)
  (h_tangent : ∀ {X}, Tangent P circumcircle_k)
  (h_angle : ∀ {X}, Angle X B C = Angle X circumcircle_k)
  (h_non_right : ∠ A B C ≠ 90) :
  ∀ {E : Point}, Parallel (Line_through P E) (Line_through A B) → 
    (Intersection (Line_through P E) (Line_through A C) = Perpendicular_bisector A B) :=
sorry

end parallel_intersects_perpendicular_bisector_l591_591108


namespace tangent_line_equation_max_min_values_l591_591460

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_equation :
  let p : ℝ × ℝ := (0, f 0) in
  let k : ℝ := deriv f 0 in
  y = (f 0) :
  y = 1 := sorry

theorem max_min_values :
  let I : Set ℝ := Set.Icc 0 (Real.pi / 2) in
  let M := Real.sup (f '' I) in
  let m := Real.inf (f '' I) in
  M = 1 ∧ m = -Real.pi / 2 := sorry

end tangent_line_equation_max_min_values_l591_591460


namespace plastering_cost_l591_591346

variable (l w d : ℝ) (c : ℝ)

theorem plastering_cost :
  l = 60 → w = 25 → d = 10 → c = 0.90 →
    let A_bottom := l * w;
    let A_long_walls := 2 * (l * d);
    let A_short_walls := 2 * (w * d);
    let A_total := A_bottom + A_long_walls + A_short_walls;
    let C_total := A_total * c;
    C_total = 2880 :=
by sorry

end plastering_cost_l591_591346


namespace least_positive_24x_16y_l591_591314

theorem least_positive_24x_16y (x y : ℤ) : ∃ a : ℕ, a > 0 ∧ a = 24 * x + 16 * y ∧ ∀ b : ℕ, b = 24 * x + 16 * y → b > 0 → b ≥ a :=
sorry

end least_positive_24x_16y_l591_591314


namespace equilateral_triangle_distance_sum_is_6sqrt3_l591_591098

-- Definitions
def equilateral_triangle (a : ℝ) := a > 0

def midpoint_distance_sum (a : ℝ) : ℝ :=
  let h := (sqrt 3 / 2) * a in
  (1 / 2) * h + h + (1 / 2) * h

-- Condition
def side_length := 6

-- Theorem Statement
theorem equilateral_triangle_distance_sum_is_6sqrt3 :
  equilateral_triangle side_length →
  midpoint_distance_sum side_length = 6 * sqrt 3 :=
by
  intro h_pos
  -- Proof omitted
  sorry

end equilateral_triangle_distance_sum_is_6sqrt3_l591_591098


namespace solution_l591_591371

def is_15_pretty (n : ℕ) : Prop :=
  nat.divisors_count n = 15 ∧ 15 ∣ n

def is_15_pretty_and_less_than_3000 (n : ℕ) : Prop :=
  is_15_pretty n ∧ n < 3000

def sum_of_15_pretty_less_than_3000 : ℕ :=
  (Finset.filter is_15_pretty_and_less_than_3000 (Finset.range 3000)).sum id

theorem solution : sum_of_15_pretty_less_than_3000 / 15 = 135 :=
  sorry

end solution_l591_591371


namespace slope_of_line_through_intersection_of_circles_l591_591070

theorem slope_of_line_through_intersection_of_circles :
  let circle1 := (λ x y : ℝ, x^2 + y^2 - 6 * x + 4 * y - 20)
  let circle2 := (λ x y : ℝ, x^2 + y^2 - 8 * x + 6 * y + 9)
  ∃ (m : ℝ), m = 1 ∧ ∀ x y : ℝ, circle1 x y = 0 ∧ circle2 x y = 0 → y = m * x - 14.5 :=
by
  sorry  -- the actual proof should be written here

end slope_of_line_through_intersection_of_circles_l591_591070


namespace police_emergency_number_has_prime_gt_7_l591_591033

theorem police_emergency_number_has_prime_gt_7 (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
sorry

end police_emergency_number_has_prime_gt_7_l591_591033


namespace car_washes_in_package_l591_591913

-- Definitions based on conditions
def normal_cost : ℝ := 15
def discount_rate : ℝ := 0.60
def total_paid : ℝ := 180

-- Derived definitions
def discounted_cost := discount_rate * normal_cost 
def num_car_washes := total_paid / discounted_cost

-- The theorem to be proved
theorem car_washes_in_package : num_car_washes = 20 :=
by
  sorry

end car_washes_in_package_l591_591913


namespace cricket_bat_price_l591_591719

theorem cricket_bat_price (A_cost B_profit_percentage C_profit_percentage : ℝ) 
    (A_cost_eq : A_cost = 156) 
    (B_profit_percentage_eq : B_profit_percentage = 0.20) 
    (C_profit_percentage_eq : C_profit_percentage = 0.25) : 
    let B_cost := A_cost * (1 + B_profit_percentage) in
    let C_cost := B_cost * (1 + C_profit_percentage) in
    C_cost = 234 :=
by
  have A_cost_eq: A_cost = 156 := sorry
  have B_profit_percentage_eq: B_profit_percentage = 0.20 := sorry
  have C_profit_percentage_eq: C_profit_percentage = 0.25 := sorry
  let B_cost := A_cost * (1 + B_profit_percentage)
  let C_cost := B_cost * (1 + C_profit_percentage)
  show C_cost = 234 from sorry

end cricket_bat_price_l591_591719


namespace salary_changes_l591_591339

/-- 
A company has a total of 41 employees (including the manager), where 
the manager's salary is higher than that of the other employees. This year, 
the manager's salary increased from 200,000 yuan last year to 230,000 yuan, 
while the salaries of the other employees remained the same as last year. 
Prove that the average salary increases, while the median salary remains unchanged.
-/
theorem salary_changes (total_employees : ℕ)
 (manager_prev_salary manager_cur_salary : ℕ)
 (other_employee_salaries : List ℕ)
 (h_total : total_employees = 41)
 (h_manager_prev : manager_prev_salary = 200000)
 (h_manager_cur : manager_cur_salary = 230000)
 (h_unchanged_salaries : ∀ (s : ℕ), s ∈ other_employee_salaries → s = s) :
 (average_salary : ℕ) 
 (median_salary : ℕ) 
 : 
 average_salary < average_salary + (manager_cur_salary - manager_prev_salary) ∧
 median_salary = median_salary :=
sorry

end salary_changes_l591_591339


namespace find_YW_in_right_triangle_l591_591546

noncomputable def right_triangle_YW_length : Prop :=
  ∃ (X Y Z W : ℝ) (A : ℝ) (base : ℝ), 
  let h_triangle : (Y = 0) ∧ (X * YZ = base * height / 2 = A) 
  ∧ (A = 200)
  ∧ (base = 30) 
  ∧ (angle YWZ = 90) 
  ∧ (YW is altitude of triangle XYZ)
  ⇒ (YW = 40 / 3)

theorem find_YW_in_right_triangle : right_triangle_YW_length :=
sorry

end find_YW_in_right_triangle_l591_591546


namespace range_of_k_l591_591171

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  k * x - Real.log x

def f_prime (k : ℝ) (x : ℝ) : ℝ :=
  k - 1 / x

def is_monotonically_increasing_on (f : ℝ → ℝ) (a : ℝ) (b : ℝ) :=
  ∀ x ∈ Ioo a b, ∀ y ∈ Ioo a b, x < y → f x ≤ f y

theorem range_of_k (k : ℝ) :
  (∀ x > 1, f_prime k x ≥ 0) ↔ k ∈ set.Ici 1 :=
sorry

end range_of_k_l591_591171


namespace chemist_solution_l591_591695

theorem chemist_solution (x : ℝ) (h1 : ∃ x, 0 < x) 
  (h2 : x + 1 > 1) : 0.60 * x = 0.10 * (x + 1) → x = 0.2 := by
  sorry

end chemist_solution_l591_591695


namespace binom_n_2_l591_591059

theorem binom_n_2 (n : ℕ) (h : 0 < n) : (nat.choose n 2) = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l591_591059


namespace number_of_friends_l591_591374

theorem number_of_friends (total_bottle_caps : ℕ) (bottle_caps_per_friend : ℕ) (h1 : total_bottle_caps = 18) (h2 : bottle_caps_per_friend = 3) :
  total_bottle_caps / bottle_caps_per_friend = 6 :=
by
  sorry

end number_of_friends_l591_591374


namespace volume_problem_l591_591901

noncomputable def volume_of_polyhedron (a : ℝ) : ℝ :=
  27 * a^3 * Real.sqrt 3 / 4

theorem volume_problem
   (a : ℝ)
   (base_edge_length : ℝ)
   (lateral_edge_length : ℝ)
   (h_planes : ∀ (A B C A1 B1 C1 : Point),
     plane_through_vertex_and_perpendicular_to_line A (line A B1) ∧
     plane_through_vertex_and_perpendicular_to_line B (line B C1) ∧
     plane_through_vertex_and_perpendicular_to_line C (line C A1)) :
  volume_of_polyhedron a = 27 * a^3 * Real.sqrt 3 / 4 :=
sorry

end volume_problem_l591_591901


namespace find_n_l591_591860

noncomputable def b_0 : ℝ := Real.cos (Real.pi / 18) ^ 2

noncomputable def b_n (n : ℕ) : ℝ :=
if n = 0 then b_0 else 4 * (b_n (n - 1)) * (1 - (b_n (n - 1)))

theorem find_n : ∀ n : ℕ, b_n n = b_0 → n = 24 := 
sorry

end find_n_l591_591860


namespace hyperbola_distance_between_vertices_l591_591411

theorem hyperbola_distance_between_vertices :
  ∀ (x y : ℝ), ((x^2) / 64 - (y^2) / 49 = 1) → (distance (8, 0) (-8, 0) = 16) :=
by
  sorry

end hyperbola_distance_between_vertices_l591_591411


namespace divisor_inequality_l591_591607

variable (k p : Nat)

-- Conditions
def is_prime (p : Nat) : Prop := Nat.Prime p
def is_divisor_of (k d : Nat) : Prop := ∃ m : Nat, d = k * m

-- Given conditions and the theorem to be proved
theorem divisor_inequality (h1 : k > 3) (h2 : is_prime p) (h3 : is_divisor_of k (2^p + 1)) : k ≥ 2 * p + 1 :=
  sorry

end divisor_inequality_l591_591607


namespace mabel_age_l591_591559

theorem mabel_age (n : ℕ) (h : n * (n + 1) / 2 = 28) : n = 7 :=
sorry

end mabel_age_l591_591559


namespace find_k_l591_591702

def f (n : ℤ) : ℤ :=
if n % 2 = 0 then n / 2 else n + 3

theorem find_k (k : ℤ) (h_odd : k % 2 = 1) (h_f_f_f_k : f (f (f k)) = 27) : k = 105 := by
  sorry

end find_k_l591_591702


namespace students_passing_subjects_l591_591505

theorem students_passing_subjects 
  (total_students : ℕ) (pass_english : ℕ) (pass_math : ℕ) (pass_science : ℕ)
  (pass_english_math : ℕ) (pass_english_science : ℕ) (pass_math_science : ℕ) (pass_all_three : ℕ) :
  total_students = 60 ∧
  pass_english = 30 ∧
  pass_math = 35 ∧
  pass_science = 28 ∧
  pass_english_math = 12 ∧
  pass_english_science = 11 ∧
  pass_math_science = 15 ∧
  pass_all_three = 6 →
  let only_english := pass_english - pass_english_math - pass_english_science + pass_all_three in
  let only_math := pass_math - pass_english_math - pass_math_science + pass_all_three in
  let only_science := pass_science - pass_english_science - pass_math_science + pass_all_three in
  only_english - (only_math + only_science) = -9 :=
by
  intros
  let only_english := pass_english - pass_english_math - pass_english_science + pass_all_three
  let only_math := pass_math - pass_english_math - pass_math_science + pass_all_three
  let only_science := pass_science - pass_english_science - pass_math_science + pass_all_three
  sorry

end students_passing_subjects_l591_591505


namespace relationship_s1_s2_l591_591542

variables {A B C G : Type}
variable [metric_space A]
variables (triangle : A → A → A → Prop) (centroid : A → A → A → A → Prop) 
          (s1 : ℝ) (s2 : ℝ)

noncomputable def GA (G A : A) := dist G A
noncomputable def GB (G B : A) := dist G B
noncomputable def GC (G C : A) := dist G C
noncomputable def AB (A B : A) := dist A B
noncomputable def BC (B C : A) := dist B C
noncomputable def CA (C A : A) := dist C A

axiom h : ∀ {A B C G : A}, triangle A B C → centroid A B C G → 
          s1 = GA G A + GB G B + GC G C ∧ 
          s2 = AB A B + BC B C + CA C A

theorem relationship_s1_s2 :
  ∀ {A B C G : A}, triangle A B C → centroid A B C G → 
  s2 ≥ 2 * s1 ∧ s1 ≤ s2 :=
sorry

end relationship_s1_s2_l591_591542


namespace students_solved_only_B_l591_591056

variable (A B C : Prop)
variable (n x y b c d : ℕ)

-- Conditions given in the problem
axiom h1 : n = 25
axiom h2 : x + y + b + c + d = n
axiom h3 : b + d = 2 * (c + d)
axiom h4 : x = y + 1
axiom h5 : x + b + c = 2 * (b + c)

-- Theorem to be proved
theorem students_solved_only_B : b = 6 :=
by
  sorry

end students_solved_only_B_l591_591056


namespace exists_perimeter_equal_point_l591_591800

variables {A B C P Q : Type}
variables [linear_ordered_field P] 
variables [add_comm_group Q] [module P Q]
include P Q

-- Given a triangle ABC
structure Triangle where
  A B C : Q

def isPerimeterEqual (t1 t2 : Triangle) : Prop :=
  t1.A + t1.B + t1.C = t2.A + t2.B + t2.C

-- Given conditions in our problem
def isPointOnSegment (P : Q) (B C : Q) : Prop :=
    ∃ t: P, (0 < t) ∧ (t < 1) ∧ P = (t • B) + ((1 - t) • C)

def isParallel (AB PQ : Q) : Prop :=
    ∃ t: P, PQ = t • AB

-- Lean 4 statement equivalent to the problem
theorem exists_perimeter_equal_point (Δ : Triangle) :
    ∃ P : Q, 
      isPointOnSegment P Δ.B Δ.C ∧
      ∃ Q : Q,
        isParallel (Δ.A - Δ.B) (Q - Δ.P) ∧
        isPerimeterEqual ⟨Δ.A, P, Q⟩ ⟨P, Δ.B, Q⟩ :=
sorry

end exists_perimeter_equal_point_l591_591800


namespace existsRectangle_l591_591434

structure Point where
  x : ℝ
  y : ℝ

structure Quadrilateral where
  A B C D : Point

structure Rectangle where
  E F G H : Point

def Segment (p1 p2 : Point) : Set Point :=
  { p : Point | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y) }

def isOnSegment (p : Point) (seg : Set Point) : Prop :=
  p ∈ seg

def isRectangle (rect : Rectangle) : Prop :=
  let ⟨E, F, G, H⟩ := rect
  (E.x - F.x) * (G.x - H.x) + (E.y - F.y) * (G.y - H.y) = 0 ∧
  (E.x - H.x) * (G.x - F.x) + (E.y - H.y) * (G.y - F.y) = 0

def rectangleLengthConstraints (E F G : Point) : Prop :=
  let EF_length := (F.x - E.x)^2 + (F.y - E.y)^2
  let EG_length := (G.x - E.x)^2 + (G.y - E.y)^2
  EG_length = 4 * EF_length

theorem existsRectangle (quad : Quadrilateral) :
  ∃ (rect : Rectangle), 
  isRectangle(rect) ∧ 
  rectangleLengthConstraints rect.E rect.F rect.G ∧
  isOnSegment quad.A (Segment rect.E rect.F) ∧
  isOnSegment quad.B (Segment rect.F rect.G) ∧
  isOnSegment quad.C (Segment rect.G rect.H) ∧
  isOnSegment quad.D (Segment rect.H rect.E) :=
sorry

end existsRectangle_l591_591434


namespace monotonic_decrease_interval_of_sinx_cosx_sinx_l591_591271

noncomputable def monotonically_decreasing_interval (f : ℝ → ℝ) (a b : ℝ) : Set ℝ :=
  {x | x ∈ Set.Icc a b ∧ ∀ y ∈ Set.Icc a b, x ≤ y → f x ≥ f y}

theorem monotonic_decrease_interval_of_sinx_cosx_sinx :
  monotonically_decreasing_interval (λ x, 2 * sin x * (cos x - sin x)) (π / 8) (5 * π / 8) = 
  Set.Icc (π / 8) (5 * π / 8) :=
sorry

end monotonic_decrease_interval_of_sinx_cosx_sinx_l591_591271


namespace donna_fully_loaded_truck_weight_l591_591393

-- Define conditions
def empty_truck_weight : ℕ := 12000
def soda_crate_weight : ℕ := 50
def soda_crate_count : ℕ := 20
def dryer_weight : ℕ := 3000
def dryer_count : ℕ := 3

-- Calculate derived weights
def soda_total_weight : ℕ := soda_crate_weight * soda_crate_count
def fresh_produce_weight : ℕ := 2 * soda_total_weight
def dryer_total_weight : ℕ := dryer_weight * dryer_count

-- Define target weight of fully loaded truck
def fully_loaded_truck_weight : ℕ :=
  empty_truck_weight + soda_total_weight + fresh_produce_weight + dryer_total_weight

-- State and prove the theorem
theorem donna_fully_loaded_truck_weight :
  fully_loaded_truck_weight = 24000 :=
by
  -- Provide necessary calculations and proof steps if needed
  sorry

end donna_fully_loaded_truck_weight_l591_591393


namespace n_salary_eq_260_l591_591649

variables (m n : ℕ)
axiom total_salary : m + n = 572
axiom m_salary : m = 120 * n / 100

theorem n_salary_eq_260 : n = 260 :=
by
  sorry

end n_salary_eq_260_l591_591649


namespace parallel_planes_by_perpendicular_line_l591_591840

variable {Point : Type} [Geometry Point] -- Assuming a type for points in a geometry context

variable {α β : Plane Point} -- Two different planes
variable {m n : Line Point}    -- Two different lines

-- Conditions extracted from option C
variable (n_perp_alpha : n ⟂ α)  -- n is perpendicular to α
variable (n_perp_beta : n ⟂ β)   -- n is perpendicular to β

-- Statement to prove
theorem parallel_planes_by_perpendicular_line : α ∥ β :=
by
  sorry

end parallel_planes_by_perpendicular_line_l591_591840


namespace max_value_of_g_l591_591425

def g (x : ℝ) : ℝ := min (min (2 * x + 2) (1 / 2 * x + 1)) (min (-3 / 4 * x + 7) (3 * x + 4))

theorem max_value_of_g : ∃ x : ℝ, g x = 17 / 5 := 
begin
  use 24 / 5,
  calc
     g (24 / 5)
     = min (min (2 * (24 / 5) + 2) (1/2 * (24 / 5) + 1)) (min (-3/4 * (24 / 5) + 7) (3 * (24 / 5) + 4)) : rfl
     ... = 17 / 5 : by norm_num,
end

end max_value_of_g_l591_591425


namespace range_of_a_l591_591147

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a + 3) * Real.exp (a * x)

theorem range_of_a (a : ℝ) : 
  (∀ x y, x ≤ y → f a x ≤ f a y) ∨ (∀ x y, x ≤ y → f a x ≥ f a y) → 
  a ∈ Set.Ico (-2 : ℝ) 0 :=
sorry

end range_of_a_l591_591147


namespace intersection_is_interval_l591_591469

-- Let M be the set of numbers where the domain of the function y = log x is defined.
def M : Set ℝ := {x | 0 < x}

-- Let N be the set of numbers where x^2 - 4 > 0.
def N : Set ℝ := {x | x^2 - 4 > 0}

-- The complement of N in the real numbers ℝ.
def complement_N : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- We need to prove that the intersection of M and the complement of N is the interval (0, 2].
theorem intersection_is_interval : (M ∩ complement_N) = {x | 0 < x ∧ x ≤ 2} := 
by 
  sorry

end intersection_is_interval_l591_591469


namespace range_of_a_l591_591172

noncomputable def has_two_distinct_extreme_points (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ is_local_min_on f {z | z > 0} x ∧ is_local_max_on f {z | z > 0} y

theorem range_of_a
  (a : ℝ)
  (f : ℝ → ℝ)
  (h : ∀ x > 0, f x = x^2 + a * x + 2 * Real.log x) :
  has_two_distinct_extreme_points f → a < -4 :=
sorry

end range_of_a_l591_591172


namespace javier_first_throw_distance_l591_591910

-- Definitions based on the conditions
def distance_second_throw : ℕ := 150  -- As solved in the solution, the second throw is 150 meters

theorem javier_first_throw_distance
  (distance_second_throw : ℕ)
  (h_first_throw : 2 * distance_second_throw = 2 * 150)
  (h_third_throw : 4 * distance_second_throw = 4 * 150)
  (h_sum_throws : 2 * distance_second_throw + distance_second_throw + 4 * distance_second_throw = 1050) :
  2 * distance_second_throw = 300 :=
by
  -- Introduce variables for the throw distances
  let distance_first_throw := 2 * distance_second_throw
  let distance_third_throw := 4 * distance_second_throw
  -- Use the provided hypothesis and solve for the first throw distance
  have h_sum : distance_first_throw + distance_second_throw + distance_third_throw = 1050,
    from h_sum_throws
  sorry

end javier_first_throw_distance_l591_591910


namespace total_number_of_chickens_is_76_l591_591529

theorem total_number_of_chickens_is_76 :
  ∀ (hens roosters chicks : ℕ),
    hens = 12 →
    (∀ n, roosters = hens / 3) →
    (∀ n, chicks = hens * 5) →
    hens + roosters + chicks = 76 :=
by
  intros hens roosters chicks h1 h2 h3
  rw [h1, h2 12, h3 12]
  ring
  sorry

end total_number_of_chickens_is_76_l591_591529


namespace total_chickens_l591_591526

   def number_of_hens := 12
   def hens_to_roosters_ratio := 3
   def chicks_per_hen := 5

   theorem total_chickens (h : number_of_hens = 12)
                          (r : hens_to_roosters_ratio = 3)
                          (c : chicks_per_hen = 5) :
     number_of_hens + (number_of_hens / hens_to_roosters_ratio) + (number_of_hens * chicks_per_hen) = 76 :=
   by
     sorry
   
end total_chickens_l591_591526


namespace asymptote_hyperbola_condition_l591_591978

theorem asymptote_hyperbola_condition : 
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = 4/3 * x ∨ y = -4/3 * x)) ∧
  ¬(∀ x y : ℝ, (y = 4/3 * x ∨ y = -4/3 * x → x^2 / 9 - y^2 / 16 = 1)) :=
by sorry

end asymptote_hyperbola_condition_l591_591978


namespace line_intersects_circle_at_two_points_midpoint_trajectory_and_chord_length_l591_591792

noncomputable def circle_eq : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 - 2*x - 4*y - 20 = 0

noncomputable def line_eq (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

noncomputable def center : ℝ × ℝ := (1, 2)
noncomputable def radius : ℝ := 5

noncomputable def point_Q : ℝ × ℝ := (3, 1)

-- Statement (1)
theorem line_intersects_circle_at_two_points (m : ℝ) :
  (∃ x y, circle_eq x y ∧ line_eq m x y) :=
by
  sorry

-- Statement (2)
theorem midpoint_trajectory_and_chord_length :
  (∀ P : ℝ × ℝ, ∃ A B : ℝ × ℝ, (circle_eq A.1 A.2) ∧ (circle_eq B.1 B.2) ∧
                  (line_eq (∃ m, line_eq m A.1 A.2 ∧ line_eq m B.1 B.2)) ∧
                  (P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →
                  (P.1 - 2)^2 + (P.2 - 3/2)^2 = 5/4) ∧
  ∃ A B : ℝ × ℝ, (circle_eq A.1 A.2) ∧ (circle_eq B.1 B.2) ∧
  (line_eq (∃ m, line_eq m A.1 A.2 ∧ line_eq m B.1 B.2)) ∧
  2 * (sqrt (radius^2 - (sqrt ((point_Q.1 - center.1)^2 + (point_Q.2 - center.2)^2)))) = 4 * sqrt 5 :=
by
  sorry

end line_intersects_circle_at_two_points_midpoint_trajectory_and_chord_length_l591_591792


namespace number_is_40_l591_591340

theorem number_is_40 (N : ℝ) (h : N = (3/8) * N + (1/4) * N + 15) : N = 40 :=
by
  sorry

end number_is_40_l591_591340


namespace number_of_zeros_F_is_zero_l591_591812

variable (f : ℝ → ℝ)
variable (hf_diff : Differentiable ℝ f)
variable (h_cond : ∀ x : ℝ, x ≠ 0 → f'' x + f x / x > 0)

def F (x : ℝ) : ℝ := f x + 1 / x

theorem number_of_zeros_F_is_zero :
  ∀ x : ℝ, F f x ≠ 0 := sorry

end number_of_zeros_F_is_zero_l591_591812


namespace minimum_possible_value_of_M_l591_591610

theorem minimum_possible_value_of_M {A B C : ℂ[X]}
  (hdegA : A.degree = 3) (hdegB : B.degree = 4) (hdegC : C.degree = 8)
  (hconstA : A.coeff 0 = 4) (hconstB : B.coeff 0 = 5) (hconstC : C.coeff 0 = 9) :
  ∃ z : ℂ, ∀ w : ℂ, (A * B - C).eval w = 0 → w = z :=
sorry

end minimum_possible_value_of_M_l591_591610


namespace hotel_charge_comparison_l591_591614

theorem hotel_charge_comparison (R G P : ℝ) 
  (h1 : P = R - 0.70 * R)
  (h2 : P = G - 0.10 * G) :
  ((R - G) / G) * 100 = 170 :=
by
  sorry

end hotel_charge_comparison_l591_591614


namespace intersection_of_A_and_B_l591_591129

open Set

-- Defining sets A and B
def A : Set ℝ := { x | x ≤ 4 }
def B : Set ℕ := {1, 3, 5, 7}

-- The proof goal
theorem intersection_of_A_and_B : A ∩ (B : Set ℝ) = {1, 3} :=
sorry

end intersection_of_A_and_B_l591_591129


namespace find_ordered_pairs_l591_591403

theorem find_ordered_pairs (a b : ℕ) (h1 : 2 * a + 1 ∣ 3 * b - 1) (h2 : 2 * b + 1 ∣ 3 * a - 1) : 
  (a = 2 ∧ b = 2) ∨ (a = 12 ∧ b = 17) ∨ (a = 17 ∧ b = 12) :=
by {
  sorry -- proof omitted
}

end find_ordered_pairs_l591_591403


namespace points_on_line_divisibility_l591_591609

theorem points_on_line_divisibility
  (x y : ℤ) 
  (p : ℤ) (hp_prime : p.prime)
  (h1 : (x * y - 1) % p = 0)
  (x1 y1 x2 y2 x3 y3 : ℤ)
  (h2 : ∃ a b c : ℤ, a * x1 + b * y1 = c ∧ a * x2 + b * y2 = c ∧ a * x3 + b * y3 = c) :
  ∃i j:fin 3, i ≠ j ∧ p ∣ (if i = 0 then x1 - if j = 1 then x2 - x3 else x3 else if i = 1 then x2 - if j = 0 then x1 - x3 else x3 else x3 - if j = 0 then x1 - x2 else x2) ∧ 
  p ∣ (if i = 0 then y1 - if j = 1 then y2 - y3 else y3 else if i = 1 then y2 - if j = 0 then y1 - y3 else y3 else y3 - if j = 0 then y1 - y2 else y2) :=
sorry

end points_on_line_divisibility_l591_591609


namespace perimeter_square_no_condition_statements_l591_591146

theorem perimeter_square_no_condition_statements :
  (∃ (A : ℕ), A = 6) → ¬(∃ (x y : nat → bool), ∀ a b : nat, x a ≠ y b) :=
by
  intro h
  -- Here we would provide the proof that calculating the perimeter 
  -- of a square with area of 6 does not require conditional statements
  sorry

end perimeter_square_no_condition_statements_l591_591146


namespace line_through_two_points_l591_591265

theorem line_through_two_points (P Q : ℝ × ℝ) (hP : P = (2, 5)) (hQ : Q = (2, -5)) :
  (∀ (x y : ℝ), (x, y) = P ∨ (x, y) = Q → x = 2) :=
by
  sorry

end line_through_two_points_l591_591265


namespace ratio_man_to_son_in_two_years_l591_591709

-- Define current ages and the conditions
def son_current_age : ℕ := 24
def man_current_age : ℕ := son_current_age + 26

-- Define ages in two years
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- State the theorem
theorem ratio_man_to_son_in_two_years : 
  (man_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2 :=
by sorry

end ratio_man_to_son_in_two_years_l591_591709


namespace quadractic_inequality_solution_l591_591177

theorem quadractic_inequality_solution (a b : ℝ) (h₁ : ∀ x : ℝ, -4 ≤ x ∧ x ≤ 3 → x^2 - (a+1) * x + b ≤ 0) : a + b = -14 :=
by 
  -- Proof construction is omitted
  sorry

end quadractic_inequality_solution_l591_591177


namespace bacteria_growth_time_l591_591966

theorem bacteria_growth_time (initial_bacteria : ℕ) (final_bacteria : ℕ) (doubling_time : ℕ) :
  initial_bacteria = 1000 →
  final_bacteria = 128000 →
  doubling_time = 3 →
  (∃ t : ℕ, final_bacteria = initial_bacteria * 2 ^ (t / doubling_time) ∧ t = 21) :=
by
  sorry

end bacteria_growth_time_l591_591966


namespace coefficient_x5_in_expansion_l591_591656

theorem coefficient_x5_in_expansion (x : ℝ) : 
  ∃ c : ℝ, (x - 2)^7 = ∑ k in finset.range 8, (binom 7 k * x^k * (-2)^(7-k)) ∧
  c = 84 ∧ 
  (binom 7 5 * x^5 * (-2)^(7-5)) = c * x^5 := 
by {
  sorry
}

end coefficient_x5_in_expansion_l591_591656


namespace center_of_XY_l591_591930

theorem center_of_XY (A B C D E F P Q X Y: Point) (k: Circle)
  (h1: triangle A B C)
  (h2: triangle.has_ne_angle A B C)
  (h3: circle.inscribed k A B C)
  (h4: k.touches_side D B C)
  (h5: k.touches_side E C A)
  (h6: k.touches_side F A B)
  (h7: line.AD A D)
  (h8: k.intersects_again P AD)
  (h9: Q = line_EF.intersection_with_perpendicular P AD)
  (h10: intersects_with_AQ X AD E)
  (h11: intersects_with_AQ Y AD F) :
  center_of_segment A X Y := sorry

end center_of_XY_l591_591930


namespace max_value_2ab_3bc_lemma_l591_591931

noncomputable def max_value_2ab_3bc (a b c : ℝ) : ℝ :=
  2 * a * b + 3 * b * c

theorem max_value_2ab_3bc_lemma
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c)
  (h : a^2 + b^2 + c^2 = 2) :
  max_value_2ab_3bc a b c ≤ 3 :=
sorry

end max_value_2ab_3bc_lemma_l591_591931


namespace sum_of_reciprocals_zero_l591_591646

open Real

variable (A B C A1 B1 C1 M : Point)
variable (f : M → Line)
variable (g : Line → Point)

noncomputable def MA1 : ℝ := sorry
noncomputable def MB1 : ℝ := sorry
noncomputable def MC1 : ℝ := sorry

theorem sum_of_reciprocals_zero 
  (hA1 : f(M) ∩ Line(B, C) = A1)
  (hB1 : f(M) ∩ Line(C, A) = B1)
  (hC1 : f(M) ∩ Line(A, B) = C1) 
  (h1 : MA1 ≠ 0)
  (h2 : MB1 ≠ 0)
  (h3 : MC1 ≠ 0) 
  : (1 / MA1) + (1 / MB1) + (1 / MC1) = 0 :=
sorry

end sum_of_reciprocals_zero_l591_591646


namespace find_circle_C_and_m_range_l591_591114

-- Definitions and given conditions
def point := ℝ × ℝ
def A : point := (3, 3)
def B : point := (2, 4)
def center_line (x : ℝ) : point := (x, 3 * x - 5)

-- Circle definition
def circle_eq (h k r : ℝ) : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - h)^2 + (y - k)^2 = r^2

-- Questions in form of hypothesis
theorem find_circle_C_and_m_range :
  ∃ h k r, (circle_eq h k r A ∧ circle_eq h k r B ∧ center_line h = (h, k) ∧ r = 1) ∧
  (∀ m > 0, ∃ M, circle_eq 3 4 1 M ∧ circle_eq 0 0 m M → 4 ≤ m ∧ m ≤ 6) :=
by {
  -- Proof is omitted for this statement.
  sorry
}

end find_circle_C_and_m_range_l591_591114


namespace inverse_of_f_at_neg_one_third_l591_591252

/-- Definition of the function f. -/
def f (x : ℝ) : ℝ := (x ^ 5 - 1) / 3

/-- Proof that the inverse of f at -1/3 is 0. -/
theorem inverse_of_f_at_neg_one_third : f 0 = -1 / 3 ↔ 0 = f⁻¹ (-1 / 3) :=
by
  sorry

end inverse_of_f_at_neg_one_third_l591_591252


namespace real_coefficient_polynomials_with_special_roots_l591_591405

noncomputable def P1 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) * (Polynomial.X ^ 2 - Polynomial.X + 1)
noncomputable def P2 : Polynomial ℝ := (Polynomial.X + 1) ^ 3 * (Polynomial.X - 1 / 2) * (Polynomial.X - 2)
noncomputable def P3 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) ^ 3 * (Polynomial.X - 2)
noncomputable def P4 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) ^ 3
noncomputable def P5 : Polynomial ℝ := (Polynomial.X + 1) ^ 2 * (Polynomial.X - 1 / 2) ^ 2 * (Polynomial.X - 2)
noncomputable def P6 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) ^ 2 * (Polynomial.X - 2) ^ 2
noncomputable def P7 : Polynomial ℝ := (Polynomial.X + 1) ^ 2 * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) ^ 2

theorem real_coefficient_polynomials_with_special_roots (P : Polynomial ℝ) :
  (∀ α, Polynomial.IsRoot P α → Polynomial.IsRoot P (1 - α) ∧ Polynomial.IsRoot P (1 / α)) →
  P = P1 ∨ P = P2 ∨ P = P3 ∨ P = P4 ∨ P = P5 ∨ P = P6 ∨ P = P7 :=
  sorry

end real_coefficient_polynomials_with_special_roots_l591_591405


namespace trisector_length_l591_591510

theorem trisector_length
  (XYZ : Triangle)
  (right_triangle : XYZ.is_right)
  (XZ YZ : ℝ)
  (XZ_eq_5 : XZ = 5) 
  (YZ_eq_12 : YZ = 12) :
  let XY := Real.sqrt (XZ^2 + YZ^2)
  let trisector_len := (5 / 6) * (XY / 3)
  XY = 13 ∧ trisector_len = 65 / 18 :=
by
  let XY := Real.sqrt (5^2 + 12^2)
  have H1 : XY = 13 :=
  by
    sorry -- proof that XY = 13

  let trisector_len := (5 / 6) * (13 / 3)
  have H2 : trisector_len = 65 / 18 :=
  by
    sorry -- proof that trisector_len = 65 / 18

  exact ⟨H1, H2⟩

end trisector_length_l591_591510


namespace power_series_rational_function_l591_591084

-- Define the power series
def power_series (f : ℚ → ℚ) : ℕ → bool := 
λ n, if f (2^(-n)) ≠ 0 then true else false

-- Define the main problem
theorem power_series_rational_function (f : ℚ → ℚ) (hf : ∀ n, f (2^(-n)) = 0 ∨ f (2^(-n)) = 1) 
(h : ∃ (q : ℚ), f 0.5 = q) :
∃ (g : ℚ → ℚ), ∃ (p q : ℚ), ∀ x, g x = (p / q) :=
sorry

end power_series_rational_function_l591_591084


namespace find_y_l591_591420

-- Definitions of points
def point1 : ℝ × ℝ := (3, -5)
def point2 : ℝ × ℝ := (5, 1)
def point3 : ℝ × ℝ := (7, y)

-- Defining the condition for collinearity
def is_on_line (y : ℝ) : Prop :=
  ∃ m b : ℝ, m = (point2.2 - point1.2) / (point2.1 - point1.1) ∧
              b = point1.2 - m * point1.1 ∧
              y = m * 7 + b

-- Statement to prove
theorem find_y (y : ℝ) (h : is_on_line y) : y = 7 :=
sorry

end find_y_l591_591420


namespace sum_of_squares_pairwise_distances_leq_nsq_Rsq_l591_591199

variables (n : ℕ) (R : ℝ)
variables (points : fin n → EuclideanSpace ℝ (n - 1))

noncomputable def pairwise_distance_sum_of_squares (points : fin n → EuclideanSpace ℝ (n - 1)) : ℝ :=
  finset.sum (finset.univ.filter (λ ⟨i, j⟩ : fin n × fin n, i < j)) (λ ⟨i, j⟩, (dist (points i) (points j))^2)

theorem sum_of_squares_pairwise_distances_leq_nsq_Rsq :
  ∀ (points : fin n → EuclideanSpace ℝ (n - 1)), 
  (∀ i, dist (0 : EuclideanSpace ℝ (n - 1)) (points i) ≤ R) →
  pairwise_distance_sum_of_squares n points ≤ n^2 * R^2 :=
by { 
  intros points h, 
  sorry 
}

end sum_of_squares_pairwise_distances_leq_nsq_Rsq_l591_591199


namespace reflection_point_sum_l591_591268

theorem reflection_point_sum (m b : ℝ) (H : ∀ x y : ℝ, (1, 2) = (x, y) ∨ (7, 6) = (x, y) → 
    y = m * x + b) : m + b = 8.5 := by
  sorry

end reflection_point_sum_l591_591268


namespace mean_weight_players_l591_591995

/-- Definitions for the weights of the players and proving the mean weight. -/
def weights : List ℕ := [62, 65, 70, 73, 73, 76, 78, 79, 81, 81, 82, 84, 87, 89, 89, 89, 90, 93, 95]

def mean (lst : List ℕ) : ℚ := (lst.sum : ℚ) / lst.length

theorem mean_weight_players : mean weights = 80.84 := by
  sorry

end mean_weight_players_l591_591995


namespace a_n_eq_3n_b_n_eq_3_pow_n_minus_1_smallest_n_for_Tn_gt_S6_l591_591891

noncomputable def a : ℕ → ℕ
| n => 3 * n

noncomputable def b : ℕ → ℕ
| n => 3^(n - 1)

noncomputable def c (n : ℕ) : ℕ :=
  4 * b n - a 5

noncomputable def T (n : ℕ) : ℕ :=
  nat.rec_on n 0 (λ n T_n, T_n + c (n + 1))

noncomputable def S (n : ℕ) : ℕ :=
  nat.rec_on n 0 (λ n S_n, S_n + a (n + 1))

-- Proof: a and b
theorem a_n_eq_3n (n : ℕ) : a n = 3 * n := sorry

theorem b_n_eq_3_pow_n_minus_1 (n : ℕ) : b n = 3^(n - 1) := sorry

-- Proof: Smallest n such that T_n > S_6
theorem smallest_n_for_Tn_gt_S6 : ∃ n : ℕ, T n > S 6 ∧ ∀ m : ℕ, m < n → T m ≤ S 6 := sorry

end a_n_eq_3n_b_n_eq_3_pow_n_minus_1_smallest_n_for_Tn_gt_S6_l591_591891


namespace unique_solution_quadratic_eq_l591_591088

theorem unique_solution_quadratic_eq (p : ℝ) (h_nonzero : p ≠ 0) : (∀ x : ℝ, p * x^2 - 20 * x + 4 = 0) → p = 25 :=
by
  sorry

end unique_solution_quadratic_eq_l591_591088


namespace M_finite_and_invariant_l591_591214

def M (a b c n : ℤ) : ℕ :=
  (finset.univ.filter (λ (xy : ℤ × ℤ), a * xy.1 ^ 2 + 2 * b * xy.1 * xy.2 + c * xy.2 ^ 2 = n)).card

theorem M_finite_and_invariant {a b c k n : ℤ}
  (ha : a > 0)
  (hP : ∃ (P : ℤ) (P1 P2 Pm : list ℤ), (ac - b^2 = P) ∧ (P = P1.prod * P2.prod * ... * Pm.prod) ∧ 
    (∀ p ∈ P1 ++ P2 ++ ... ++ Pm, nat.prime p)) : 
  M a b c n finite ∧ M a b c (P^k * n) = M a b c n := 
sorry

end M_finite_and_invariant_l591_591214


namespace kho_kho_only_players_eq_20_l591_591691

-- Define the given conditions
def total_players : Nat := 30
def kabadi_only_players : Nat := kabadi_total_players - both_games_players := 10 - 5
def both_games_players : Nat := 5

-- Prove that the number of people who play kho kho only is 20
theorem kho_kho_only_players_eq_20 (H : Nat) (H_def : total_players = kabadi_only_players + H + both_games_players):
  H = 20 :=
by
  sorry

end kho_kho_only_players_eq_20_l591_591691


namespace overall_percentage_change_in_membership_l591_591047

theorem overall_percentage_change_in_membership :
  let M := 1
  let fall_inc := 1.08
  let winter_inc := 1.15
  let spring_dec := 0.81
  (M * fall_inc * winter_inc * spring_dec - M) / M * 100 = 24.2 := by
  sorry

end overall_percentage_change_in_membership_l591_591047


namespace part1_part2_case1_part2_case2_part2_case3_part3_l591_591822

variable (m : ℝ)
def f (x : ℝ) := (m + 1) * x^2 - (m - 1) * x + (m - 1)

-- Part (1)
theorem part1 (h : ∀ x : ℝ, f m x < 1) : m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

-- Part (2)
theorem part2_case1 (h : m = -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≥ 1 :=
sorry

theorem part2_case2 (h : m > -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≤ (m - 1) / (m + 1) ∨ x ≥ 1 :=
sorry

theorem part2_case3 (h : m < -1) : ∀ x, f m x ≥ (m + 1) * x ↔ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1) :=
sorry

-- Part (3)
theorem part3 (h : ∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), f m x ≥ 0) : m ≥ 1 :=
sorry

end part1_part2_case1_part2_case2_part2_case3_part3_l591_591822


namespace f_at_9_l591_591136

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom piecewise_definition : ∀ x : ℝ, (0 < x ∧ x ≤ 1) → f(x) = |x|
axiom functional_equation : ∀ x : ℝ, f(x) + f(x + 1) = 2

theorem f_at_9 : f(9) = 1 :=
by
  -- proof goes here
  sorry

end f_at_9_l591_591136


namespace simplify_complex_expr_l591_591600

theorem simplify_complex_expr (a b : ℂ) (hz : b = 7 * complex.I) (ha : a = 4) :
  (a + b) / (a - b) + (a - b) / (a + b) = -66 / 65 := sorry

end simplify_complex_expr_l591_591600


namespace sum_of_sequence_l591_591159

def a (n : ℕ) : ℚ := 2 / (n * (n + 1))

def S (n : ℕ) : ℚ := ∑ k in Finset.range n, a (k + 1)

theorem sum_of_sequence (n : ℕ) : S n = 2 * n / (n + 1) :=
by sorry

end sum_of_sequence_l591_591159


namespace triple_cheese_pizza_count_l591_591577

theorem triple_cheese_pizza_count :
  ∃ T : ℕ, (let P := 5 in 
            let M := 9 in 
            let cost_triple_cheese := (T / 2 * P) in 
            let cost_meat_lovers := ((M / 3) * 2 * P) in
            cost_triple_cheese + cost_meat_lovers = 55) ∧ T = 10 :=
  sorry

end triple_cheese_pizza_count_l591_591577


namespace base_length_first_tri_sail_l591_591561

-- Define the areas of the sails
def area_rect_sail : ℕ := 5 * 8
def area_second_tri_sail : ℕ := (4 * 6) / 2

-- Total canvas area needed
def total_canvas_area_needed : ℕ := 58

-- Calculate the total area so far (rectangular sail + second triangular sail)
def total_area_so_far : ℕ := area_rect_sail + area_second_tri_sail

-- Define the height of the first triangular sail
def height_first_tri_sail : ℕ := 4

-- Define the area needed for the first triangular sail
def area_first_tri_sail : ℕ := total_canvas_area_needed - total_area_so_far

-- Prove that the base length of the first triangular sail is 3 inches
theorem base_length_first_tri_sail : ∃ base : ℕ, base = 3 ∧ (base * height_first_tri_sail) / 2 = area_first_tri_sail := by
  use 3
  have h1 : (3 * 4) / 2 = 6 := by sorry -- This is a placeholder for actual calculation
  exact ⟨rfl, h1⟩

end base_length_first_tri_sail_l591_591561


namespace cost_of_chairs_l591_591567

-- Given conditions
def total_spent : ℕ := 56
def cost_of_table : ℕ := 34
def number_of_chairs : ℕ := 2

-- The target definition
def cost_of_one_chair : ℕ := 11

-- Statement to prove
theorem cost_of_chairs (N : ℕ) (T : ℕ) (C : ℕ) (x : ℕ) (hN : N = total_spent) (hT : T = cost_of_table) (hC : C = number_of_chairs) : x = cost_of_one_chair ↔ N - T = C * x :=
by
  sorry

end cost_of_chairs_l591_591567


namespace randy_trip_length_l591_591585

-- Define the conditions
noncomputable def fraction_gravel := (1/4 : ℚ)
noncomputable def miles_pavement := (30 : ℚ)
noncomputable def fraction_dirt := (1/6 : ℚ)

-- The proof statement
theorem randy_trip_length :
  ∃ x : ℚ, (fraction_gravel + fraction_dirt + (miles_pavement / x) = 1) ∧ x = 360 / 7 := 
by
  sorry

end randy_trip_length_l591_591585


namespace count_polynomials_with_three_integer_roots_l591_591380

def polynomial_with_roots (n: ℕ) : Nat :=
  have h: n = 8 := by
    sorry
  if n = 8 then
    -- Apply the combinatorial argument as discussed
    52
  else
    -- Case for other n
    0

theorem count_polynomials_with_three_integer_roots:
  polynomial_with_roots 8 = 52 := 
  sorry

end count_polynomials_with_three_integer_roots_l591_591380


namespace probability_of_picking_female_brunette_in_ac_club_under_5_feet_l591_591180

noncomputable def total_students : ℕ := 200
noncomputable def female_students : ℕ := (0.60 * total_students).toNat
noncomputable def female_brunettes : ℕ := (0.50 * female_students).toNat
noncomputable def female_brunettes_under_5_feet : ℕ := (0.50 * female_brunettes).toNat
noncomputable def female_brunettes_in_academic_club : ℕ := (0.40 * female_brunettes).toNat
noncomputable def female_brunettes_in_ac_club_under_5_feet : ℕ := (0.75 * female_brunettes_in_academic_club).toNat

theorem probability_of_picking_female_brunette_in_ac_club_under_5_feet :
  (female_brunettes_in_ac_club_under_5_feet : ℚ) / total_students = 0.09 :=
by
  sorry

end probability_of_picking_female_brunette_in_ac_club_under_5_feet_l591_591180


namespace valid_arrangements_l591_591886

-- Define the digits we have and the requirement that the number does not begin with 0
def digits : List ℕ := [4, 7, 7, 7, 0]

def is_valid_number (l : List ℕ) : Prop :=
  l.length = 5 ∧ l.head ≠ some 0

-- Define the number of valid arrangements we need to prove
def valid_arrangements_count : ℕ := 16

theorem valid_arrangements : 
  (∃ (l : List ℕ) (perm : l ∈ l.permutations), is_valid_number l) → valid_arrangements_count = 16 :=
by sorry

end valid_arrangements_l591_591886


namespace mod_sum_of_inverses_l591_591303

theorem mod_sum_of_inverses :
  (7⁻¹ + 7⁻² + 7⁻³) % 25 = 9 :=
sorry

end mod_sum_of_inverses_l591_591303


namespace sum_of_areas_is_858_l591_591248

def length1 : ℕ := 1
def length2 : ℕ := 9
def length3 : ℕ := 25
def length4 : ℕ := 49
def length5 : ℕ := 81
def length6 : ℕ := 121

def base_width : ℕ := 3

def area (width : ℕ) (length : ℕ) : ℕ :=
  width * length

def total_area_of_rectangles : ℕ :=
  area base_width length1 +
  area base_width length2 +
  area base_width length3 +
  area base_width length4 +
  area base_width length5 +
  area base_width length6

theorem sum_of_areas_is_858 : total_area_of_rectangles = 858 := by
  sorry

end sum_of_areas_is_858_l591_591248


namespace percent_decrease_approx_46_l591_591741

theorem percent_decrease_approx_46 :
  let last_month_price := 7 / 3
      this_month_price := 5 / 4
      percent_decrease := ((last_month_price - this_month_price) / last_month_price) * 100
  in abs (percent_decrease - 46) < 1 :=
by
  sorry

end percent_decrease_approx_46_l591_591741


namespace two_circles_tangent_internally_l591_591477

-- Define radii and distance between centers
def R : ℝ := 7
def r : ℝ := 4
def distance_centers : ℝ := 3

-- Statement of the problem
theorem two_circles_tangent_internally :
  distance_centers = R - r → 
  -- Positional relationship: tangent internally
  (distance_centers = abs (R - r)) :=
sorry

end two_circles_tangent_internally_l591_591477


namespace no_function_satisfies_conditions_l591_591198

def fractional_part (m : ℝ) : ℝ :=
  m - m.floor

noncomputable def no_such_function : ℝ → ℝ :=
  sorry

theorem no_function_satisfies_conditions :
  ¬ ∃ f : ℝ → ℝ,
      (∀ x : ℝ, fractional_part (f x) * (Real.sin x)^2 + 
                 fractional_part x * (Real.cos (f x)) * (Real.cos x) = f x) ∧
      (∀ x : ℝ, f (f x) = f x) :=
by
  sorry

end no_function_satisfies_conditions_l591_591198


namespace domain_of_f_domain_of_f_eq_real_l591_591659
   
   noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt ((x - 2)^2 + (x + 2)^2)
   
   theorem domain_of_f : ∀ x : ℝ, (x - 2)^2 + (x + 2)^2 ≠ 0 :=
   by
     intro x
     have h₀ : (x - 2)^2 + (x + 2)^2 = 2 * x^2 + 8 := by 
       calc
         (x - 2)^2 + (x + 2)^2 
             = (x^2 - 4 * x + 4) + (x^2 + 4 * x + 4) : by ring
         _ = 2 * x^2 + 8 : by ring
     have h₁ : 2 * x^2 + 8 > 0 := by
       exact add_pos_of_pos_of_nonneg (mul_pos two_pos (pow_two_nonneg x)) zero_lt_eight
     exact ne_of_gt h₁

   -- The Lean statement proving the domain of f
   theorem domain_of_f_eq_real : ∀ x : ℝ, ∃ y = f x, true :=
   by
     intro x
     use 1 / Real.sqrt ((x - 2)^2 + (x + 2)^2)
     trivial
   
   
end domain_of_f_domain_of_f_eq_real_l591_591659


namespace factorial_trailing_zeroes_and_divisibility_l591_591387

noncomputable def count_factors : Nat -> Nat -> Nat
| 0, _ => 0
| n, p => (n / p) + count_factors (n / p) p

theorem factorial_trailing_zeroes_and_divisibility :
  count_factors 500 5 = 124 ∧ count_factors 500 3 = 247 :=
by
  sorry

end factorial_trailing_zeroes_and_divisibility_l591_591387


namespace real_solutions_count_l591_591482

theorem real_solutions_count :
  ∃ x1 x2 x3 : ℝ, 
  (|x1 + 2| = |x1 - 1| + |x1 - 4| ∧ 
   |x2 + 2| = |x2 - 1| + |x2 - 4| ∧ 
   |x3 + 2| = |x3 - 1| + |x3 - 4| ∧ 
   x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ 
   {x1, x2, x3} = {-7, 1, 7}) :=
begin
  -- existence of 3 solutions
  sorry
end

end real_solutions_count_l591_591482


namespace arithmetic_progression_x_value_l591_591979

theorem arithmetic_progression_x_value (x : ℝ) 
  (h1 : (x + 3) - (x - 3) = (3x + 5) - (x + 3)) : x = 2 :=
by
  -- sorry here to indicate proof needs to be provided
  sorry

end arithmetic_progression_x_value_l591_591979


namespace scalar_multiplication_zero_l591_591426

variable {R : Type*} [Field R]
variable {V : Type*} [AddCommGroup V] [Module R V]
variables (a b c : V) (λ : R)

theorem scalar_multiplication_zero (λ : R) (a : V) : λ • a = 0 → λ = 0 ∨ a = 0 :=
by sorry

end scalar_multiplication_zero_l591_591426


namespace wheel_radius_increase_l591_591230

noncomputable def calculate_increase_in_radius (r : ℝ) (D_old D_new : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * r
  let distance_per_rotation := circumference / 63360
  let num_rotations := D_old / distance_per_rotation
  let r' := (D_old * distance_per_rotation * 63360) / (2 * Real.pi * D_new)
  r' - r

theorem wheel_radius_increase (r D_old D_new : ℝ) (increase : ℝ) : 
  r = 12 → D_old = 300 → D_new = 290 → increase = 0.31 → 
  calculate_increase_in_radius r D_old D_new = increase := by
  intros
  sorry

end wheel_radius_increase_l591_591230


namespace distance_hyperbola_vertices_l591_591408

noncomputable def distance_between_vertices (a : ℝ) : ℝ :=
2 * a

theorem distance_hyperbola_vertices :
  ∃ a : ℝ, a^2 = 64 ∧ distance_between_vertices a = 16 :=
by
  have h : a = sqrt 64 := by
    apply sqrt_eq
    rw [pow_two, nat.cast_comm]
  use 8
  rw h
  apply sqrt_pos
  norm_num
  use 8
  split
  exact rfl
  norm_num
  sorry

end distance_hyperbola_vertices_l591_591408


namespace solve_f_l591_591541

noncomputable def f (α : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

theorem solve_f (α : ℝ) (f : ℝ → ℝ) :
      (α = 1 → (∀ x : ℝ, f x = -x)) ∧ (α = -1 → (∀ x : ℝ, f x = x)) ∧ 
      (f (f (x + y) * f (x - y)) = x^2 + α * y * f y) :=
begin
  sorry
end

end solve_f_l591_591541


namespace number_of_divisors_of_16n4_l591_591925

theorem number_of_divisors_of_16n4 (n : ℕ) (h1 : n % 2 = 1) (h2 : (n.factors.length + 1).prod = 13) : ∃ d : ℕ, d = 245 :=
by {
  sorry
}

end number_of_divisors_of_16n4_l591_591925


namespace factorial_expression_equality_l591_591060

theorem factorial_expression_equality :
  (4 * 6! + 20 * 5! + 48 * 4!) / 7! = 134 / 105 := 
by sorry

end factorial_expression_equality_l591_591060


namespace length_AB_area_AOB_l591_591522

-- Definitions for points in polar coordinates and given conditions
def O := (0, 0 : ℝ × ℝ)
def A := (2, real.pi / 3 : ℝ × ℝ)
def B := (3, 0 : ℝ × ℝ)

-- The length of line segment AB
theorem length_AB : 
  let OA := 2
  let OB := 3
  let angle_OAB := real.pi / 3
  |AB| = real.sqrt (OA^2 + OB^2 - 2 * OA * OB * real.cos angle_OAB) :=
by sorry

-- The area of triangle AOB
theorem area_AOB :
  let OA := 2
  let OB := 3
  let angle_OAB := real.pi / 3
  (1 / 2) * OA * OB * real.sin angle_OAB = (3 * real.sqrt 3) / 2 :=
by sorry

end length_AB_area_AOB_l591_591522


namespace time_addition_3pm_l591_591525

theorem time_addition_3pm (X Y Z : ℕ) (hx : X = 2) (hy : Y = 18) (hz : Z = 53) : X + Y + Z = 73 :=
by {
  have h₁ : X + Y + Z = 2 + 18 + 53, by simp [hx, hy, hz],
  have h₂ : 2 + 18 + 53 = 73, by norm_num,
  exact trans h₁ h₂
}

end time_addition_3pm_l591_591525


namespace largest_value_of_x_not_defined_l591_591762

noncomputable def quadratic_formula (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b*b - 4*a*c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2*a)
  let x2 := (-b - sqrt_discriminant) / (2*a)
  (x1, x2)

noncomputable def largest_root : ℝ :=
  let (x1, x2) := quadratic_formula 4 (-81) 49
  if x1 > x2 then x1 else x2

theorem largest_value_of_x_not_defined :
  largest_root = 19.6255 :=
by
  sorry

end largest_value_of_x_not_defined_l591_591762


namespace ratio_female_to_male_l591_591363

-- Definitions for the conditions
def average_age_female (f : ℕ) : ℕ := 40 * f
def average_age_male (m : ℕ) : ℕ := 25 * m
def average_age_total (f m : ℕ) : ℕ := (30 * (f + m))

-- Statement to prove
theorem ratio_female_to_male (f m : ℕ) 
  (h_avg_f: average_age_female f = 40 * f)
  (h_avg_m: average_age_male m = 25 * m)
  (h_avg_total: average_age_total f m = 30 * (f + m)) : 
  f / m = 1 / 2 :=
by
  sorry

end ratio_female_to_male_l591_591363


namespace find_original_number_l591_591391

-- The definition sets up the problem based on the given conditions
def original_number (x : ℝ) : Prop :=
  let final_result := (x + 3 - 3) / 3 + 3 in
  final_result = 12

-- The theorem states that given the final result is 12, the original number is 7.
theorem find_original_number : 
  ∃ (x : ℝ), original_number x ∧ x = 7 :=
by 
  exists 7
  simp [original_number]
  sorry -- skipping the detailed proof

end find_original_number_l591_591391


namespace projection_of_a_on_b_l591_591141

noncomputable def e_1 : ℝ := sorry
noncomputable def e_2 : ℝ := sorry

def angle_e1_e2 : ℝ := 2 * Real.pi / 3
def a : ℝ := e_1 + 2 * e_2
def b : ℝ := -3 * e_2

def projection (a b : ℝ) : ℝ := (a * b) / Real.sqrt (b * b)

theorem projection_of_a_on_b :
  let a := e_1 + 2 * e_2,
      b := -3 * e_2 in
  projection a b = -3 / 2 := sorry

end projection_of_a_on_b_l591_591141


namespace max_cables_cut_l591_591639

/-- 
Prove that given 200 computers connected by 345 cables initially forming a single cluster, after 
cutting cables to form 8 clusters, the maximum possible number of cables that could have been 
cut is 153.
--/
theorem max_cables_cut (computers : ℕ) (initial_cables : ℕ) (final_clusters : ℕ) (initial_clusters : ℕ) 
  (minimal_cables : ℕ) (cuts : ℕ) : 
  computers = 200 ∧ initial_cables = 345 ∧ final_clusters = 8 ∧ initial_clusters = 1 ∧ 
  minimal_cables = computers - final_clusters ∧ 
  cuts = initial_cables - minimal_cables →
  cuts = 153 := 
sorry

end max_cables_cut_l591_591639


namespace limit_s_tends_to_infinity_l591_591550

-- Let P(m) be the x-coordinate of the left endpoint of the intersection of the graphs of y = x^3 and y = m
def P(m: ℝ) := -m^(1/3)

-- Define s as (P(-m) - P(m)) / m
def s (m: ℝ) := (P(-m) - P(m)) / m

-- State the theorem which proves the value of s tends to infinity as m approaches 0
theorem limit_s_tends_to_infinity :
  ∀ m: ℝ, -2 < m → m < 2 → filter.tendsto s (nhds_within 0 (Ioo (-2 : ℝ) 2)) filter.at_top :=
sorry

end limit_s_tends_to_infinity_l591_591550


namespace general_term_formula_sum_first_n_terms_T_n_l591_591829

def f (x : ℝ) : ℝ := (sqrt 3) * Real.cos (Real.pi * x) - Real.sin (Real.pi * x)

def a_n (n : ℕ) : ℝ := n - (2 / 3)

def b_n (n : ℕ) : ℝ := (1 / 2)^n * (a_n n + (2 / 3))

def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, b_n (i+1)

theorem general_term_formula (n : ℕ) : a_n n = n - (2 / 3) :=
sorry

theorem sum_first_n_terms_T_n (n : ℕ) : T_n n = 2 - (n + 2) / 2^n :=
sorry

end general_term_formula_sum_first_n_terms_T_n_l591_591829


namespace a_1966_eq_1024_l591_591845

noncomputable def a_seq (k : ℕ) : ℕ := if k = 1 then 1966 else
    int.toNat (int.floor (real.sqrt (list.sum (list.map (λ (i : ℕ), a_seq i) (list.range (k - 1))))))

theorem a_1966_eq_1024 :
    a_seq 1966 = 1024 := 
sorry

end a_1966_eq_1024_l591_591845


namespace calculate_expression_l591_591370

noncomputable def tan (x : ℝ) : ℝ := Real.tan x

theorem calculate_expression :
  sqrt 27 - abs (2 * sqrt 3 - 9 * tan (Real.pi / 6)) + (1 / 2)⁻¹ - (1 - Real.pi)^0 = 2 * sqrt 3 + 1 :=
by
  sorry

end calculate_expression_l591_591370


namespace part_a_part_b_l591_591643

-- Define a polygon type
structure Polygon :=
  (sides : ℕ)
  (area : ℝ)
  (grid_size : ℕ)

-- Define a function to verify drawable polygon
def DrawablePolygon (p : Polygon) : Prop :=
  ∃ (n : ℕ), p.grid_size = n ∧ p.area = n ^ 2

-- Part (a): 20-sided polygon with an area of 9
theorem part_a : DrawablePolygon {sides := 20, area := 9, grid_size := 3} :=
by
  sorry

-- Part (b): 100-sided polygon with an area of 49
theorem part_b : DrawablePolygon {sides := 100, area := 49, grid_size := 7} :=
by
  sorry

end part_a_part_b_l591_591643


namespace value_of_m_l591_591174

theorem value_of_m (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : ∀ x y : ℝ, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1) 
    (eccentricity : (real.sqrt (a ^ 2 + b ^ 2)) / a = 3)
    (circle_condition : ∀ m : ℝ, (∃ d : ℝ, d = real.sqrt (9 - m) ∧ d = 1)) :
    ∃ m : ℝ, m = 8 :=
by
  sorry

end value_of_m_l591_591174


namespace coefficient_x4_expansion_zero_l591_591076

theorem coefficient_x4_expansion_zero :
  let expr := (λ x : ℝ, (x^3 / 3 - 3 / x^2) ^ 10) in
  ∀ x : ℝ, x ≠ 0 → coeff (taylor expr 10) 4 = 0 :=
by
  sorry

end coefficient_x4_expansion_zero_l591_591076


namespace angle_between_a_and_b_l591_591933

noncomputable def angle_between_vectors (a b : Vector ℝ) : ℝ :=
  Real.arccos ((a • b) / (|a| * |b|))

variable {a b : Vector ℝ}

theorem angle_between_a_and_b
  (ha : |a| = 1)
  (hb : |b| = 2)
  (h : a • (a + b) = 0) :
  angle_between_vectors a b = 2 * Real.pi / 3 :=
by
  sorry

end angle_between_a_and_b_l591_591933


namespace relationship_between_M_and_N_l591_591788

theorem relationship_between_M_and_N (a b : ℝ) (M N : ℝ) 
  (hM : M = a^2 - a * b) 
  (hN : N = a * b - b^2) : M ≥ N :=
by sorry

end relationship_between_M_and_N_l591_591788


namespace find_250th_term_l591_591760

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_excluded (n : ℕ) : Prop :=
  is_perfect_square n ∨ is_perfect_cube n

def sequence_without_squares_and_cubes : ℕ → ℕ
| 0     := 1
| (n+1) := Nat.find (λ m, (m > sequence_without_squares_and_cubes n) ∧ ¬ is_excluded m)

theorem find_250th_term :
  sequence_without_squares_and_cubes 249 = 270 :=
sorry

end find_250th_term_l591_591760


namespace count_odd_nines_2007_digit_number_l591_591987

theorem count_odd_nines_2007_digit_number :
  let nums_with_odd_nines := ({n : ℕ // (digit_count n 9 % 2 = 1) ∧ (digits n).length = 2007}) in
  ∃ count : ℕ,
  count = (1/2) * (10^2006 - 8^2006) :=
by
  sorry

end count_odd_nines_2007_digit_number_l591_591987


namespace area_of_equilateral_triangle_ABC_l591_591398

theorem area_of_equilateral_triangle_ABC (ABC_Equilateral : EquilateralTriangle ABC) 
                                         (circumcircle : Circumcircle Ω ABC)
                                         (D_on_minor_arc_AB : Point_Omega D ABC AB)
                                         (E_on_minor_arc_AC : Point_Omega E ABC AC)
                                         (BC_eq_DE : BC = DE)
                                         (area_ABE : area (Triangle A B E) = 3)
                                         (area_ACD : area (Triangle A C D) = 4) :
  area (Triangle A B C) = 37 / 7 :=
sorry

end area_of_equilateral_triangle_ABC_l591_591398


namespace necessary_but_not_sufficient_condition_l591_591810

theorem necessary_but_not_sufficient_condition {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  ((a + b > 1) ↔ (ab > 1)) → false :=
by
  sorry

end necessary_but_not_sufficient_condition_l591_591810


namespace correct_simplification_l591_591670

theorem correct_simplification (m a b x y : ℝ) :
  ¬ (4 * m - m = 3) ∧
  ¬ (a^2 * b - a * b^2 = 0) ∧
  ¬ (2 * a^3 - 3 * a^3 = a^3) ∧
  (x * y - 2 * x * y = - x * y) :=
by {
  sorry
}

end correct_simplification_l591_591670


namespace problem1_problem2_l591_591221

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 2 * a) * Real.log (x + 1) - 2 * x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + x^2 - x * Real.log (x + 1)

theorem problem1 (x : ℝ) : (x + 2) * Real.log (x + 1) - 2 * x
  is_monotonically_increasing_on Ioo (-1:ℝ) ∞ ∧
  (∀ x, (x + 2) * Real.log (x + 1) - 2 * x = 0 ↔ x = 0) := 
sorry

theorem problem2 (x1 x2 x3 y1 y2 y3 : ℝ) (hx : x1 + x2 = 2 * x3) :
  ∃ (a : ℝ), ∀ (x : ℝ), x ∈ (Ioo 0 ∞) →
    let k := (y2 - y1) / (x2 - x1),
        gx := ∂ (g a) / ∂ x ↔ g' (x2) = k in
    (a = 0) := sorry

end problem1_problem2_l591_591221


namespace fully_loaded_truck_weight_l591_591395

def empty_truck : ℕ := 12000
def weight_soda_crate : ℕ := 50
def num_soda_crates : ℕ := 20
def weight_dryer : ℕ := 3000
def num_dryers : ℕ := 3
def weight_fresh_produce : ℕ := 2 * (weight_soda_crate * num_soda_crates)

def total_loaded_truck_weight : ℕ :=
  empty_truck + (weight_soda_crate * num_soda_crates) + weight_fresh_produce + (weight_dryer * num_dryers)

theorem fully_loaded_truck_weight : total_loaded_truck_weight = 24000 := by
  sorry

end fully_loaded_truck_weight_l591_591395


namespace plane_through_points_l591_591807

noncomputable def vector (α : Type) [Add α] [Sub α] [Zero α] := (α × α × α)

noncomputable def point := vector ℝ 

def A : point := (1, 0, 1)
def B : point := (-2, 2, 1)
def C : point := (2, 0, 3)

def vec_sub (p1 p2 : point) : point :=
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def dot_prod (v1 v2 : point) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def plane_eq (a b c d x y z : ℝ) : Prop :=
  a * x + b * y + c * z + d = 0

theorem plane_through_points :
  ∃ (a b c d : ℝ), plane_eq a b c d 2 3 (-1) ∧
  (∃ a b c d, plane_eq a b c d 1 0 1) ∧
  (∃ a b c d, plane_eq a b c d (-2) 2 1) ∧
  (∃ a b c d, plane_eq a b c d 2 0 3) :=
sorry

end plane_through_points_l591_591807


namespace roofing_cost_per_foot_l591_591935

theorem roofing_cost_per_foot:
  ∀ (total_feet needed_feet free_feet : ℕ) (total_cost : ℕ),
  needed_feet = 300 →
  free_feet = 250 →
  total_cost = 400 →
  needed_feet - free_feet = 50 →
  total_cost / (needed_feet - free_feet) = 8 :=
by sorry

end roofing_cost_per_foot_l591_591935


namespace correct_selling_price_l591_591710

-- Define the cost prices in A-coins
def cost_price_A (item : String) : ℕ :=
  if item = "pencil" then 15
  else if item = "eraser" then 25
  else if item = "sharpener" then 35
  else 0

-- Define the exchange rate
def exchange_rate : ℕ := 2 -- 1 A-coin = 2 B-coins

-- Define the desired profit percentages
def profit_percentage (item : String) : ℕ :=
  if item = "pencil" then 20
  else if item = "eraser" then 25
  else if item = "sharpener" then 30
  else 0

-- Calculate the cost price in B-coins
def cost_price_B (item : String) : ℕ :=
  cost_price_A(item) * exchange_rate

-- Calculate the selling price in B-coins
def selling_price (item : String) : ℕ :=
  let profit := profit_percentage(item) in
  let cost_in_B := cost_price_B(item) in
  let profit_in_B := cost_in_B * profit / 100 in
  cost_in_B + profit_in_B

-- Prove the correctness of the selling prices
theorem correct_selling_price :
  selling_price "pencil" = 36 ∧
  selling_price "eraser" = 62 ∧ -- assuming simplification without rounding
  selling_price "sharpener" = 91 :=
by
  -- Proof is omitted
  sorry

end correct_selling_price_l591_591710


namespace curve_is_hyperbola_l591_591974

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem curve_is_hyperbola (ρ θ : ℝ) (h : ρ^2 * Real.cos (2 * θ) = 1) :
  let (x, y) := polar_to_rectangular ρ θ in
  x^2 - y^2 = 1 :=
by
  let (x, y) := polar_to_rectangular ρ θ
  sorry

end curve_is_hyperbola_l591_591974


namespace graph_passes_through_0_2_l591_591982

theorem graph_passes_through_0_2 (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (0, 2) ∈ set_of (λ p : ℝ × ℝ, p.snd = a ^ p.fst + 1) := 
by
  sorry

end graph_passes_through_0_2_l591_591982


namespace monotonic_decreasing_interval_l591_591624

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 1) → f x < f (x + 1) := 
by 
  -- sorry is used because the actual proof is not required
  sorry

end monotonic_decreasing_interval_l591_591624


namespace part1_max_price_l591_591009

theorem part1_max_price (p: ℝ) (c: ℝ) (s: ℝ) (d: ℝ): 
  (∃ a, 0 ≤ a ∧ a ≤ 5 ∧ (a + 5) * (8 - 0.8 * a) ≥ 40) ∧ p = 15 + 5 * 1 := 
begin
  let p := 15,
  let c := 10,
  let s := 80000,
  let d := 8000,
  let p_increase := 5,
  have h₁ : ∃ a, 0 ≤ a ∧ a ≤ 5 := by sorry,
  have h₂ : (a + 5) * (8 - 0.8 * a) ≥ 40 := by sorry,
  use 5,
  exact and.intro h₁ h₂,
  sorry
end

end part1_max_price_l591_591009


namespace find_AO_l591_591184

variables {A B C D O : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]
variables (AB AC BD DC AD: ℝ)
axiom condition1 : AB = 4
axiom condition2 : AC + OC = 6
axiom is_perpendicular : is_perpendicular AC BD
axiom is_perpendicular2 : is_perpendicular AC AB

theorem find_AO (AO: ℝ) : AB = 4 ∧ OC = 6 ∧ AC = AO + 6 ∧ is_perpendicular AC BD ∧ is_perpendicular AC AB  → AO = 2 :=
by
  sorry

end find_AO_l591_591184


namespace solve_for_x_l591_591958

theorem solve_for_x (x : ℝ) : 2 * x + 3 * x = 600 - (4 * x + 6 * x) → x = 40 :=
by
  intro h
  sorry

end solve_for_x_l591_591958


namespace total_number_of_cottages_is_100_l591_591743

noncomputable def total_cottages
    (x : ℕ) (n : ℕ) 
    (h1 : 2 * x = number_of_two_room_cottages)
    (h2 : n * x = number_of_three_room_cottages)
    (h3 : 3 * (n * x) = 2 * x + 25) 
    (h4 : x + 2 * x + n * x ≥ 70) : ℕ :=
x + 2 * x + n * x

theorem total_number_of_cottages_is_100 
    (x n : ℕ)
    (h1 : 2 * x = number_of_two_room_cottages)
    (h2 : n * x = number_of_three_room_cottages)
    (h3 : 3 * (n * x) = 2 * x + 25)
    (h4 : x + 2 * x + n * x ≥ 70)
    (h5 : ∃ m : ℕ, m = (x + 2 * x + n * x)) :
  total_cottages x n h1 h2 h3 h4 = 100 :=
by
  sorry

end total_number_of_cottages_is_100_l591_591743


namespace percentage_second_year_students_l591_591520

variable (TotalStudents : ℕ)
variable (NumericMethodsStudents : ℕ)
variable (AirborneControlStudents : ℕ)
variable (BothSubjectsStudents : ℕ)

theorem percentage_second_year_students 
  (h1 : TotalStudents = 676)
  (h2 : NumericMethodsStudents = 226)
  (h3 : AirborneControlStudents = 450)
  (h4 : BothSubjectsStudents = 134)
  : (NumericMethodsStudents + AirborneControlStudents - BothSubjectsStudents) / TotalStudents * 100 ≈ 80.18 := 
sorry

end percentage_second_year_students_l591_591520


namespace unique_point_in_nested_convex_polygons_l591_591540

-- Definitions
variable (P : ℕ → set ℝ^2)
variable convex : ∀ k : ℕ, convex (P k)
variable P_nested : ∀ k : ℕ, P (k+1) ⊆ P k
variable P_midpoints : ∀ k : ℕ, ∃ midpoints, ∀ x ∈ P (k+1), x is a midpoint of some side in P k

-- Theorem Statement: There exists a unique point in the intersection of P_k
theorem unique_point_in_nested_convex_polygons :
  ∃! x : ℝ^2, ∀ k : ℕ, x ∈ P k :=
sorry

end unique_point_in_nested_convex_polygons_l591_591540


namespace probability_two_slate_rocks_l591_591285

theorem probability_two_slate_rocks :
  let total_rocks := 14 + 20 + 10,
      prob_first_slate := 14 / total_rocks,
      prob_second_slate := 13 / (total_rocks - 1)
  in
    (14 * 13) / (total_rocks * (total_rocks - 1)) = 182 / 1892 :=
by
  let total_rocks := 14 + 20 + 10
  have prob_first_slate := 14 / total_rocks
  have prob_second_slate := 13 / (total_rocks - 1)
  show (14 * 13) / (total_rocks * (total_rocks - 1)) = (182 / 1892)
  sorry

end probability_two_slate_rocks_l591_591285


namespace guides_groupings_l591_591289

theorem guides_groupings : 
  ∃ n : ℕ, (∀ (t : fin 8 → fin 3), 
    (∃ g : fin 3, ∀ i : fin 8, t i ≠ g) → 
    n = 6558) := 
begin
  sorry
end

end guides_groupings_l591_591289


namespace second_quadrant_inequality_l591_591194

theorem second_quadrant_inequality (a b : ℝ) (h₁ : a < 0) (h₂ : b > 0) : a / b < 0 :=
by sorry

end second_quadrant_inequality_l591_591194


namespace integral_calculation_l591_591748

-- We define the integral problem to be solved.
def integral_problem : Prop :=
  ∫ x in 0..2, (2 * x - exp x) = 5 - exp 2

-- State the theorem we want to prove.
theorem integral_calculation : integral_problem :=
sorry

end integral_calculation_l591_591748


namespace base16_to_base2_bits_l591_591669

theorem base16_to_base2_bits :
  ∀ A B C D : ℕ,
  (A = 10) → (B = 11) → (C = 12) → (D = 13) →
  let n := 16 in
  A * n^3 + B * n^2 + C * n + D ≥ 2^15 ∧ A * n^3 + B * n^2 + C * n + D < 2^16 →
  nat.log2 (A * n^3 + B * n^2 + C * n + D) + 1 = 16 :=
by
  intros A B C D h_A h_B h_C h_D n H
  sorry

end base16_to_base2_bits_l591_591669


namespace Cody_games_l591_591065

/-- Cody had nine old video games he wanted to get rid of.
He decided to give four of the games to his friend Jake,
three games to his friend Sarah, and one game to his friend Luke.
On Saturday he bought five new games.
How many games does Cody have now? -/
theorem Cody_games (nine_games initially: ℕ) (jake_games: ℕ) (sarah_games: ℕ) (luke_games: ℕ) (saturday_games: ℕ)
  (h_initial: initially = 9)
  (h_jake: jake_games = 4)
  (h_sarah: sarah_games = 3)
  (h_luke: luke_games = 1)
  (h_saturday: saturday_games = 5) :
  ((initially - (jake_games + sarah_games + luke_games)) + saturday_games) = 6 :=
by
  sorry

end Cody_games_l591_591065


namespace isosceles_triangle_perimeter_l591_591124

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 6 ∨ a = 7) (h₂ : b = 6 ∨ b = 7) (h₃ : a ≠ b) :
  (2 * a + b = 19) ∨ (2 * b + a = 20) :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_perimeter_l591_591124


namespace puppies_per_dog_l591_591300

def dogs := 15
def puppies := 75

theorem puppies_per_dog : puppies / dogs = 5 :=
by {
  sorry
}

end puppies_per_dog_l591_591300


namespace cot_product_identity_l591_591100

theorem cot_product_identity :
  (Real.cot (Real.pi / 180 * 25) - 1) * 
  (Real.cot (Real.pi / 180 * 24) - 1) * 
  (Real.cot (Real.pi / 180 * 23) - 1) * 
  (Real.cot (Real.pi / 180 * 22) - 1) * 
  (Real.cot (Real.pi / 180 * 21) - 1) * 
  (Real.cot (Real.pi / 180 * 20) - 1) = 8 :=
by
  sorry

end cot_product_identity_l591_591100


namespace product_of_all_possible_x_l591_591489

theorem product_of_all_possible_x :
  (∀ x : ℝ, ∣20 / x + 4∣ = 3 → x = -20 ∨ x = -20 / 7) →
  (-20) * (-20 / 7) = 400 / 7 :=
by sorry

end product_of_all_possible_x_l591_591489


namespace a_2017_eq_l591_591964

noncomputable def sequence_a : ℕ → ℚ
| 0     := 1/2
| (n+1) := 2 * sequence_a n / (3 * sequence_a n + 2)

theorem a_2017_eq : sequence_a 2016 = 1 / 3026 :=
begin
  sorry -- The proof will go here
end

end a_2017_eq_l591_591964


namespace calculate_fraction_pow_l591_591367

theorem calculate_fraction_pow :
  (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
  sorry

end calculate_fraction_pow_l591_591367


namespace find_f_2013_l591_591166

-- Definition of the function f as per the conditions
noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_add4_le (x : ℝ) : f(x + 4) ≤ f(x) + 4
axiom f_add2_ge (x : ℝ) : f(x + 2) ≥ f(x) + 2
axiom f_1_eq_0 : f(1) = 0

-- The statement to be proven
theorem find_f_2013 : f(2013) = 2012 := 
by {
  sorry 
}

end find_f_2013_l591_591166


namespace smoking_lung_disease_confidence_l591_591356

/-- Prove that given the conditions, the correct statement is C:
   If it is concluded from the statistic that there is a 95% confidence 
   that smoking is related to lung disease, then there is a 5% chance of
   making a wrong judgment. -/
theorem smoking_lung_disease_confidence 
  (P Q : Prop) 
  (confidence_level : ℝ) 
  (h_conf : confidence_level = 0.95) 
  (h_PQ : P → (Q → true)) :
  ¬Q → (confidence_level = 1 - 0.05) :=
by
  sorry

end smoking_lung_disease_confidence_l591_591356


namespace geometric_sequence_a9_l591_591896

theorem geometric_sequence_a9
  (a : ℕ → ℤ)
  (q : ℤ)
  (h1 : a 3 * a 6 = -32)
  (h2 : a 4 + a 5 = 4)
  (hq : ∃ n : ℤ, q = n)
  : a 10 = -256 := 
sorry

end geometric_sequence_a9_l591_591896


namespace determine_m_l591_591839

theorem determine_m (x y m : ℝ) 
  (h1 : 3 * x + 2 * y = 4 * m - 5) 
  (h2 : 2 * x + 3 * y = m) 
  (h3 : x + y = 2) : 
  m = 3 :=
sorry

end determine_m_l591_591839


namespace min_abs_varphi_l591_591173

noncomputable def function_is_symmetric (f : ℝ → ℝ) (x0 : ℝ) (y0 : ℝ) : Prop :=
  ∀ x, f (2 * x0 - x) = f x

theorem min_abs_varphi (φ : ℝ) (k : ℤ) (h_sym : function_is_symmetric (λ x, 3 * real.cos (2 * x + φ)) (4 * real.pi / 3) 0) :
  ∃φ : ℝ, (∃k : ℤ, φ = k * real.pi - 13 * real.pi / 6) ∧ abs φ = real.pi / 6 :=
begin
  sorry,
end

end min_abs_varphi_l591_591173


namespace greatest_mult_of_4_lessthan_5000_l591_591256

theorem greatest_mult_of_4_lessthan_5000 :
  ∃ x : ℕ, (0 < x) ∧ (x % 4 = 0) ∧ (x^3 < 5000) ∧ (∀ y : ℕ, (0 < y) ∧ (y % 4 = 0) ∧ (y^3 < 5000) → y ≤ x) := 
sorry

end greatest_mult_of_4_lessthan_5000_l591_591256


namespace student_A_probability_student_B_expectation_l591_591651

theorem student_A_probability : 
  (P : ℙ(ℕ → Prop))
  (P_A : P(λ n, set.to_finset n = {1,2,3}) = (1 / 2))
  (P_A_shots : ∀ a1 a2 a3 : Prop, 
    P ({#a1, a2, a3} ∈ tfinset.univ) = (1 / 2))
  (independence : ∀ a b : Prop, 
    P (a ∩ b) = P a * P b) :
  P({
    c a1 a2 a3 s1 s2 s3 | (P a1 ∩ P a2 ∩ P a3 = 2 / 3) 
  ∨ (P a1 ∩ P a2 ∩ P a3 = 3 / 3)
  }) = (1 / 2) :=
by
  /* Detailed proof steps, assuming required here, add: */
  sorry

theorem student_B_expectation :
  (P : ℙ(ℕ → Prop))
  (P_B : P(λ n, set.to_finset n = {1,2,3}) = (2 / 3))
  (independence : ∀ a b : Prop, 
    P (a ∩ b) = P a * P b)
  (X2 : P B(2 shots) = (4 / 9))
  (X3 : P B(3 shots) = (1 / 3))
  (X4 : P B(4 shots) = (2 / 9)) :
  ∑ b • P_B(X) = (25 / 9) :=
by
  /* Expected Value computation proof here, add: */
  sorry


end student_A_probability_student_B_expectation_l591_591651


namespace part1_part2_l591_591222

def f (x : ℝ) : ℝ := abs(2 * x + 1) - abs(x - 2)

theorem part1 (x : ℝ) : f x > 2 ↔ x > 1 ∨ x < -5 := sorry

theorem part2 (t : ℝ) (x : ℝ) : (∀ x, f x ≥ t^2 - (11/2) * t) ↔ (1 / 2 ≤ t ∧ t ≤ 5) := sorry

end part1_part2_l591_591222


namespace police_emergency_number_has_prime_gt_7_l591_591032

theorem police_emergency_number_has_prime_gt_7 (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
sorry

end police_emergency_number_has_prime_gt_7_l591_591032


namespace objective_function_range_l591_591140

theorem objective_function_range (x y : ℝ) 
  (h1 : x + 2 * y > 2) 
  (h2 : 2 * x + y ≤ 4) 
  (h3 : 4 * x - y ≥ 1) : 
  ∃ z_min z_max : ℝ, (∀ z : ℝ, z = 3 * x + y → z_min ≤ z ∧ z ≤ z_max) ∧ z_min = 1 ∧ z_max = 6 := 
sorry

end objective_function_range_l591_591140


namespace circle_tangent_parabola_height_difference_l591_591338

theorem circle_tangent_parabola_height_difference
  (a b r : ℝ)
  (point_of_tangency_left : a ≠ 0)
  (points_of_tangency_on_parabola : (2 * a^2) = (2 * (-a)^2))
  (center_y_coordinate : ∃ c , c = b)
  (circle_equation_tangent_parabola : ∀ x, (x^2 + (2*x^2 - b)^2 = r^2))
  (quartic_double_root : ∀ x, (x = a ∨ x = -a) → (x^2 + (4 - 2*b)*x^2 + b^2 - r^2 = 0)) :
  b - 2 * a^2 = 2 :=
by
  sorry

end circle_tangent_parabola_height_difference_l591_591338


namespace mrs_hilt_bakes_loaves_l591_591936

theorem mrs_hilt_bakes_loaves :
  let total_flour := 5
  let flour_per_loaf := 2.5
  (total_flour / flour_per_loaf) = 2 := 
by
  sorry

end mrs_hilt_bakes_loaves_l591_591936


namespace mary_brought_stickers_l591_591225

theorem mary_brought_stickers (friends_stickers : Nat) (other_stickers : Nat) (left_stickers : Nat) 
                              (total_students : Nat) (num_friends : Nat) (stickers_per_friend : Nat) 
                              (stickers_per_other_student : Nat) :
  friends_stickers = num_friends * stickers_per_friend →
  left_stickers = 8 →
  total_students = 17 →
  num_friends = 5 →
  stickers_per_friend = 4 →
  stickers_per_other_student = 2 →
  other_stickers = (total_students - 1 - num_friends) * stickers_per_other_student →
  (friends_stickers + other_stickers + left_stickers) = 50 :=
by
  intros
  sorry

end mary_brought_stickers_l591_591225


namespace John_l591_591916

theorem John's_earnings_on_Saturday :
  ∃ S : ℝ, (S + S / 2 + 20 = 47) ∧ (S = 18) := by
    sorry

end John_l591_591916


namespace taxi_trip_distance_l591_591914

theorem taxi_trip_distance
  (initial_fee : ℝ)
  (per_segment_charge : ℝ)
  (segment_distance : ℝ)
  (total_charge : ℝ)
  (segments_traveled : ℝ)
  (total_miles : ℝ) :
  initial_fee = 2.25 →
  per_segment_charge = 0.3 →
  segment_distance = 2/5 →
  total_charge = 4.95 →
  total_miles = segments_traveled * segment_distance →
  segments_traveled = (total_charge - initial_fee) / per_segment_charge →
  total_miles = 3.6 :=
by
  intros h_initial_fee h_per_segment_charge h_segment_distance h_total_charge h_total_miles h_segments_traveled
  sorry

end taxi_trip_distance_l591_591914


namespace sum_of_hundreds_and_tens_digits_of_product_l591_591099

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def seq_num (a : ℕ) (x : ℕ) := List.foldr (λ _ acc => acc * 1000 + a) 0 (List.range x)

noncomputable def num_a := seq_num 707 101
noncomputable def num_b := seq_num 909 101

noncomputable def product := num_a * num_b

theorem sum_of_hundreds_and_tens_digits_of_product :
  hundreds_digit product + tens_digit product = 8 := by
  sorry

end sum_of_hundreds_and_tens_digits_of_product_l591_591099


namespace part1_part2_part3_l591_591497

noncomputable def f1 (x : ℝ) : ℝ := x^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := 3x - 1
noncomputable def f3 (x : ℝ) : ℝ := 4 / (x + 2)
noncomputable def g3 (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x + a^2 - 1

theorem part1 : ¬(∀ x1 ∈ (set.Icc 0 3), ∃! x2 ∈ (set.Icc 0 3), f1 x1 * f1 x2 = 2) :=
by sorry

theorem part2 (b : ℝ) : (∀ x1 ∈ (set.Icc (1/2) b), ∃! x2 ∈ (set.Icc (1/2) b), f2 x1 * f2 x2 = 1) → b = 1 :=
by sorry

theorem part3 (a : ℝ) : (∀ x1 ∈ (set.Icc 0 2), ∃! x2 ∈ (set.Icc 0 2), f3 x1 * g3 x2 a = 2)
  → a ∈ (set.Icc (-Real.sqrt 2) (2 - Real.sqrt 3)) ∪ (set.Icc (Real.sqrt 3) (2 + Real.sqrt 2)) :=
by sorry

end part1_part2_part3_l591_591497


namespace f_range_requires_numerical_evaluation_l591_591082

-- Define function components under the given conditions
def f (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + 2 * Real.pi * Real.arcsin (x / 3) -
  3 * (Real.arcsin (x / 3))^2 + (Real.pi^2 / 4) * (x^3 - x^2 + 4 * x - 8)

-- Conditions: x belonging to the interval [-3, 3]
def valid_x (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 3

-- The proof statement about the range of f(x)
theorem f_range_requires_numerical_evaluation : 
  ∃ a b : ℝ, ∀ x, valid_x x → f(x) ∈ set.Icc a b :=
sorry

end f_range_requires_numerical_evaluation_l591_591082


namespace find_leg_length_l591_591123

variable (A B : ℝ)
variable (b : ℝ)

noncomputable def is_isosceles_triangle (A B : ℝ) : Prop :=
  ∃ b : ℝ, ∃ h : ℝ, h = (1 / 2) * b ∧ 
  b ≠ 0 ∧ 
  1 / 4 * b^2 + h^2 = b^2 ∧ 
  (1 / 2) * b * h = (sqrt 3) / 2 ∧
  sin A = sqrt 3 * sin B

theorem find_leg_length (A B : ℝ) (h_area : 1 / 2 * b * (1 / 2 * b) = (sqrt 3) / 2)
  (h_sin : sin A = sqrt 3 * sin B) :
  b = sqrt 2 :=
by
  sorry

end find_leg_length_l591_591123


namespace total_pins_used_l591_591008

-- Define the given conditions
def cardboard_length : ℕ := 34
def cardboard_width : ℕ := 14
def pins_per_side : ℕ := 35
def number_of_sides : ℕ := 4

-- Define the proof problem
theorem total_pins_used (l w p_per_side n_sides : ℕ) (h1 : l = 34) (h2 : w = 14) (h3 : p_per_side = 35) (h4 : n_sides = 4) :
  p_per_side * n_sides = 140 :=
by {
  rw [h3, h4],
  exact nat.mul_def 35 4 140,
  sorry
}

end total_pins_used_l591_591008


namespace min_sum_ineq_l591_591926

noncomputable def minimum_sum (xs : Fin 50 → ℝ) : ℝ :=
  ∑ i, xs i / (1 - (xs i)^2)

theorem min_sum_ineq (xs : Fin 50 → ℝ) (hx : ∀ i, 0 < xs i ∧ xs i < 1) 
  (h_sum : ∑ i, (xs i)^2 = 1/2) :
  minimum_sum xs ≥ (3 * Real.sqrt 3) / 4 :=
by
  sorry

end min_sum_ineq_l591_591926


namespace number_of_roots_l591_591551

def S : Set ℚ := { x : ℚ | 0 < x ∧ x < (5 : ℚ)/8 }

def f (x : ℚ) : ℚ := 
  match x.num, x.den with
  | num, den => num / den + 1

theorem number_of_roots (h : ∀ q p, (p, q) = 1 → (q : ℚ) / p ∈ S → ((q + 1 : ℚ) / p = (2 : ℚ) / 3)) :
  ∃ n : ℕ, n = 7 :=
sorry

end number_of_roots_l591_591551


namespace incorrect_statements_for_quadratic_inequality_l591_591455

-- Definitions for given conditions and required expressions
def quadratic_solution_set_empty (a b c : ℝ) : Prop := 
  a > 0 ∧ (b^2 - 4 * a * c) ≤ 0

def minimum_value_expression (a b c : ℝ) (x0 : ℝ) : ℝ :=
  (a + 4 * c) / (b - a)

theorem incorrect_statements_for_quadratic_inequality (a b c x0 : ℝ) :
  (quadratic_solution_set_empty a b c ↔ M = ∅) → 
  (M = { x | x ≠ x0 } ∧ a < b → minimum_value_expression a 4 c = 2 - 2 * real.sqrt 2) :=
sorry

end incorrect_statements_for_quadratic_inequality_l591_591455


namespace police_emergency_number_prime_divisor_l591_591037

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 1000 * k + 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n :=
sorry

end police_emergency_number_prime_divisor_l591_591037


namespace four_dancers_changes_six_dancers_changes_seven_digits_telephone_number_l591_591640

-- Definitions based on conditions
def formation_changes (n : ℕ) : ℕ := (finset.range n).succ.prod id

-- Prove statements according to the Lean 4 framework
theorem four_dancers_changes : formation_changes 4 = 24 := by
  sorry

theorem six_dancers_changes : formation_changes 6 = 720 := by
  sorry

theorem seven_digits_telephone_number : formation_changes 7 = 5040 := by
  sorry

end four_dancers_changes_six_dancers_changes_seven_digits_telephone_number_l591_591640


namespace hyperbola_line_distance_l591_591155

noncomputable def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 3) - (y^2) = 1

noncomputable def focus_F : (ℝ × ℝ) :=
  (2, 0)

noncomputable def line_through_focus (x y : ℝ) : Prop :=
  y = sqrt(3) * (x - 2)

noncomputable def asymptote1 (x y : ℝ) : Prop :=
  y = (sqrt(3) / 3) * x

noncomputable def asymptote2 (x y : ℝ) : Prop :=
  y = -(sqrt(3) / 3) * x

noncomputable def triangle_right_angled (O M N : ℝ × ℝ) : Prop :=
  ∃ P : ℝ × ℝ, P ≠ O ∧ P = M ∨ P = N ∧ ∠O N P = 90

theorem hyperbola_line_distance
  (O : ℝ × ℝ)
  (h1 : O = (0, 0))
  (M N : ℝ × ℝ)
  (H : hyperbola 2 0)
  (line_through_F : ∀ x y, line_through_focus x y)
  (asym1 : ∀ x y, asymptote1 x y → ∃ M, M = (x, y))
  (asym2 : ∀ x y, asymptote2 x y → ∃ N, N = (x, y))
  : dist M N = 3 := 
sorry

end hyperbola_line_distance_l591_591155


namespace probability_cos_between_zero_and_half_l591_591584

open Real

theorem probability_cos_between_zero_and_half :
  let range := Ιcc (-π / 2) (π / 2)
  let favorable := {x ∈ range | 0 ≤ cos x ∧ cos x ≤ 1 / 2}
  (favorable.err_measureable_space.to_real) / (Ιcc (-π / 2) (π / 2).measure.to_real) = 1 / 3 :=
by
  sorry

end probability_cos_between_zero_and_half_l591_591584


namespace ratio_w_y_l591_591278

theorem ratio_w_y
  (w x y z : ℚ)
  (h1 : w / x = 5 / 2)
  (h2 : y / z = 2 / 3)
  (h3 : x / z = 10) :
  w / y = 37.5 :=
begin
  sorry
end

end ratio_w_y_l591_591278


namespace sauna_max_couples_l591_591637

def max_couples (n : ℕ) : ℕ :=
  n - 1

theorem sauna_max_couples (n : ℕ) (rooms unlimited_capacity : Prop) (no_female_male_cohabsimult : Prop)
                          (males_shared_room_constraint females_shared_room_constraint : Prop)
                          (males_known_iff_wives_known : Prop) : max_couples n = n - 1 := 
  sorry

end sauna_max_couples_l591_591637


namespace original_price_of_wand_l591_591479

-- Definitions as per the conditions
def price_paid (paid : Real) := paid = 8
def fraction_of_original (fraction : Real) := fraction = 1 / 8

-- Question and correct answer put as a theorem to prove
theorem original_price_of_wand (paid : Real) (fraction : Real) 
  (h1 : price_paid paid) (h2 : fraction_of_original fraction) : 
  (paid / fraction = 64) := 
by
  -- This 'sorry' indicates where the actual proof would go.
  sorry

end original_price_of_wand_l591_591479


namespace decompose_x_l591_591676

def x : ℝ × ℝ × ℝ := (6, 12, -1)
def p : ℝ × ℝ × ℝ := (1, 3, 0)
def q : ℝ × ℝ × ℝ := (2, -1, 1)
def r : ℝ × ℝ × ℝ := (0, -1, 2)

theorem decompose_x :
  x = (4 : ℝ) • p + q - r :=
sorry

end decompose_x_l591_591676


namespace arithmetic_sequence_sum_l591_591492

variable {a : ℕ → ℝ}

-- Define arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, n < m → a (n + 1) - a n = a (m + 1) - a m

-- Define the special property of the arithmetic sequence given in the problem
def sequence_condition (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ 
  (a 3 = root x^2 - 3x - 5) ∧ 
  (a 10 = root x^2 - 3x - 5)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : sequence_condition a) : 
  a 5 + a 8 = 3 :=
sorry

end arithmetic_sequence_sum_l591_591492


namespace problem_I_problem_II_l591_591120

noncomputable def a (n : ℕ) : ℕ := 2^n
noncomputable def b (n : ℕ) : ℕ := 2 * n - 1
noncomputable def c (n : ℕ) : ℕ := a n * b n
noncomputable def T (n : ℕ) : ℕ := (2 * n - 3) * 2^(n + 1) + 6

theorem problem_I (n : ℕ) : a n = 2^n ∧ b n = 2 * n - 1 := by
  split
  . exact rfl
  . exact rfl

theorem problem_II (n : ℕ) : 
  let T_n := ∑ i in Finset.range n, c (i + 1)
  in T_n = (2 * n - 3) * 2^(n + 1) + 6 := 
sorry

end problem_I_problem_II_l591_591120


namespace find_min_value_l591_591965

def probability_condition (a : ℕ) : ℚ :=
  ((42 - a) * (41 - a) / 2 + (a - 1) * (a - 2) / 2) / 1176

theorem find_min_value (a : ℕ) :
  (probability_condition a ≥ 1 / 2) → ∃ (m n : ℕ), ∃ (h : Nat.coprime m n), 
  (probability_condition 22 = m / n) ∧ (m + n = 499) :=
begin
  -- sorry, proof omitted.
  sorry
end

end find_min_value_l591_591965


namespace sum_p_arithemtic_k_p_l591_591580

theorem sum_p_arithemtic_k_p {p k : ℕ} (hp : Nat.Prime p) :
  (∑ a in Finset.range p, a^k) % p = 0 ∨ (∑ a in Finset.range p, a^k) % p = p-1 :=
by
  sorry

end sum_p_arithemtic_k_p_l591_591580


namespace construct_triangle_l591_591902

variables {α : ℝ} {ρ₁ ρ₂ : ℝ}

-- Define conditions for the angle and radii
def is_valid_triangle (α ρ₁ ρ₂ : ℝ) : Prop :=
(ρ₁ > ρ₂ * cos α ∧ α < 90) ∨ (ρ₂ > ρ₁ ∧ ρ₁ = 0 ∧ α = 90) ∨ (ρ₂ > ρ₁ ∧ ρ₁ > ρ₂ * abs (cos α) ∧ α > 90)

-- The goal is to prove conditions for constructing the triangle
theorem construct_triangle (α ρ₁ ρ₂ : ℝ) : 
  is_valid_triangle α ρ₁ ρ₂ :=
sorry

end construct_triangle_l591_591902


namespace lcm_24_90_128_l591_591775

theorem lcm_24_90_128 : nat.lcm (nat.lcm 24 90) 128 = 2880 := by
  sorry

end lcm_24_90_128_l591_591775


namespace not_axisymmetric_function_C_l591_591357

/-
Among the following functions, prove that the graph of y = sqrt(x) is the one that 
is not an axisymmetric figure:
A: y = 1/x
B: y = cos x, x ∈ [0, 2π]
C: y = sqrt x
D: y = log |x|
-/

def function_A (x : ℝ) : ℝ := 1 / x
def function_B (x : ℝ) : ℝ := if h : 0 ≤ x ∧ x ≤ 2 * Real.pi then Real.cos x else 0
def function_C (x : ℝ) : ℝ := Real.sqrt x
def function_D (x : ℝ) : ℝ := Real.log (Real.abs x)

theorem not_axisymmetric_function_C :
  ¬(axisymmetric (λ x, function_C x)) :=
begin
  sorry,
end

end not_axisymmetric_function_C_l591_591357


namespace f_log2_3_l591_591831

def f (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem f_log2_3 : f (Real.log2 3) = 10 / 3 := 
by
  sorry

end f_log2_3_l591_591831


namespace avg_root_cross_sectional_area_correct_avg_volume_correct_correlation_coefficient_correct_total_volume_estimate_correct_l591_591354

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i in Finset.finRange 10, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i in Finset.finRange 10, y i) / 10

noncomputable def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  let n := 10
  let x_avg := avg_root_cross_sectional_area x
  let y_avg := avg_volume y
  let num := ∑ i in Finset.finRange 10, (x i - x_avg) * (y i - y_avg)
  let denom_x := ∑ i in Finset.finRange 10, (x i - x_avg) ^ 2
  let denom_y := ∑ i in Finset.finRange 10, (y i - y_avg) ^ 2
  num / (Real.sqrt (denom_x * denom_y))

theorem avg_root_cross_sectional_area_correct (x : Fin 10 → ℝ)
  (hx_sum : ∑ i in Finset.finRange 10, x i = 0.6) :
  avg_root_cross_sectional_area x = 0.06 :=
by
  sorry

theorem avg_volume_correct (y : Fin 10 → ℝ)
  (hy_sum : ∑ i in Finset.finRange 10, y i = 3.9) :
  avg_volume y = 0.39 :=
by
  sorry

theorem correlation_coefficient_correct (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i in Finset.finRange 10, x i = 0.6)
  (hy_sum : ∑ i in Finset.finRange 10, y i = 3.9)
  (hx2_sum : ∑ i in Finset.finRange 10, x i ^ 2 = 0.038)
  (hy2_sum : ∑ i in Finset.finRange 10, y i ^ 2 = 1.6158)
  (hxy_sum : ∑ i in Finset.finRange 10, x i * y i = 0.2474) :
  correlation_coefficient x y ≈ 0.97 :=
by
  sorry

theorem total_volume_estimate_correct (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i in Finset.finRange 10, x i = 0.6)
  (hy_sum : ∑ i in Finset.finRange 10, y i = 3.9)
  (total_x : ℝ) (htotal_x : total_x = 186) :
  (avg_volume y / avg_root_cross_sectional_area x) * total_x = 1209 :=
by
  sorry

end avg_root_cross_sectional_area_correct_avg_volume_correct_correlation_coefficient_correct_total_volume_estimate_correct_l591_591354


namespace correct_statement_is_C_l591_591675

theorem correct_statement_is_C :
  (C ↔ (∃ x : ℝ, x^2 = 1 ∧ (x = 1 ∨ x = -1))) ∧ 
  ¬((∀ a x : ℝ, (a * x ∈ ℚ → a ∈ ℕ)) ∨ 
   (∀ a : ℕ, a ∈ {b : ℕ | b > 0}) ∨ 
   (set.finite {n : ℕ | n > 0 ∧ n % 2 = 0})) :=
by
  sorry

end correct_statement_is_C_l591_591675


namespace arithmetic_sequence_general_term_and_q_range_l591_591436

noncomputable def a_n (n : ℕ) : ℕ → ℝ := λ q => 8 * q^(n-1)

def is_arithmetic (S3 S4 S5 : ℕ → ℝ) : Prop :=
  S3 + 3 * S5 = 2 * (2 * S4)

noncomputable def S_n (n : ℕ) : ℕ → ℝ := λ q => Σ i in finset.range n, a_n (i + 1) q

def b_n (n : ℕ) : ℕ → ℕ → ℝ := λ n q => real.log2 (a_n n q)

noncomputable def T_n (n : ℕ) : ℕ → ℕ → ℝ := λ n q => Σ i in finset.range n, b_n (i + 1) q

theorem arithmetic_sequence_general_term_and_q_range (q : ℝ) (h_q : q ≠ 1) :
  (∀ (n : ℕ), a_n n q = 8 / 3^(n-1)) ∧
  (is_arithmetic (S_n 3 q) (S_n 4 q) (S_n 5 q) → (∀ (n : ℕ), T_n 3 q > T_n n q) → 
    real.log2 (sqrt 2 / 4) < real.log2 q ∧ real.log2 q < real.log2 (1 / 2)) :=
begin
  sorry -- Proof goes here
end

end arithmetic_sequence_general_term_and_q_range_l591_591436


namespace coefficient_x3_y4_in_expansion_l591_591658

theorem coefficient_x3_y4_in_expansion :
  let n := 7
  let a := 3
  let b := 4
  binom n a = 35 :=
by
  sorry

end coefficient_x3_y4_in_expansion_l591_591658


namespace value_of_x_2011_l591_591452

-- Conditions

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f(x) < f(y)

def is_arithmetic_sequence (x : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ a : ℝ, ∀ n : ℕ, x n = a + d * (n - 1)

def sum_to_zero (f : ℝ → ℝ) (x : ℕ → ℝ) : Prop :=
  f (x 8) + f (x 9) + f (x 10) + f (x 11) = 0

-- Problem

theorem value_of_x_2011 {f : ℝ → ℝ} {x : ℕ → ℝ} (h1 : is_odd_function f) (h2 : is_increasing_function f)
  (h3 : is_arithmetic_sequence x 2) (h4 : sum_to_zero f x) : x 2011 = 4003 :=
by
  sorry

end value_of_x_2011_l591_591452


namespace trajectory_equation_line_equation_line_equation_simplified_l591_591435

-- Definition of the distance ratio condition
def distance_ratio_condition (x y : ℝ) : Prop :=
  (Real.sqrt ((x - 1) ^ 2 + y ^ 2)) / (abs (x - 4)) = 1 / 2

-- Theorem stating the equation of the trajectory C
theorem trajectory_equation (x y : ℝ) (h : distance_ratio_condition x y) : 
  (x^2 / 4) + (y^2 / 3) = 1 :=
begin
  sorry -- Proof omitted
end

-- Definition for the midpoint condition
def midpoint_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂ = -2) ∧ (y₁ + y₂ = 2)

-- Theorem stating the equation of the line PQ
theorem line_equation (x₁ y₁ x₂ y₂ : ℝ) (h₁ : (x₁, y₁) ∈ {p | (p.fst ^ 2 / 4) + (p.snd ^ 2 / 3) = 1})
  (h₂ : (x₂, y₂) ∈ {p | (p.fst ^ 2 / 4) + (p.snd ^ 2 / 3) = 1}) (h₃ : midpoint_condition x₁ y₁ x₂ y₂) :
  ∃ k : ℝ, ∀ x y : ℝ, y - 1 = k * (x + 1) :=
begin
  sorry -- Proof omitted
end

theorem line_equation_simplified (k : ℝ) : (3 * k - 4 * 1 + 7 = 0) :=
begin
  sorry -- Proof omitted
end

end trajectory_equation_line_equation_line_equation_simplified_l591_591435


namespace median_situps_per_minute_l591_591514

-- Defining the data set of sit-ups Xiao Li does per minute
def situps_per_minute := [50, 45, 48, 47]

-- Define what the median should be
def is_median (median : ℝ) (data : List ℝ) : Prop :=
  let sorted_data := data.sort
  let len := sorted_data.length
  if len % 2 = 1 then
    median = sorted_data.get (len / 2)
  else
    median = (sorted_data.get (len / 2 - 1) + sorted_data.get (len / 2)) / 2

-- The theorem stating the median of the sit-ups_per_minute list is 47.5
theorem median_situps_per_minute :
  is_median 47.5 situps_per_minute :=
by
  sorry

end median_situps_per_minute_l591_591514


namespace difference_of_circle_areas_l591_591615

noncomputable def pi : ℝ := Real.pi

def circumference_to_radius (C : ℝ) : ℝ := C / (2 * pi)
def area_of_circle (r : ℝ) : ℝ := pi * r^2

def difference_of_areas (C1 C2 : ℝ) : ℝ :=
  let r1 := circumference_to_radius C1
  let r2 := circumference_to_radius C2
  area_of_circle r2 - area_of_circle r1

theorem difference_of_circle_areas :
  difference_of_areas 132 352 ≈ 8463.855 :=
by
  have h1 : circumference_to_radius 132 = 132 / (2 * pi) := rfl
  have h2 : circumference_to_radius 352 = 352 / (2 * pi) := rfl
  have a1 : area_of_circle (132 / (2 * pi)) = pi * (132 / (2 * pi))^2 := rfl
  have a2 : area_of_circle (352 / (2 * pi)) = pi * (352 / (2 * pi))^2 := rfl
  have : difference_of_areas 132 352 = pi * (352 / (2 * pi))^2 - pi * (132 / (2 * pi))^2 := rfl
  simp [Real.pi, pi] at *
  sorry

end difference_of_circle_areas_l591_591615


namespace shooting_performance_l591_591298

section

-- Define the scores for soldiers A and B
def scores_A : List ℝ := [8, 6, 7, 8, 6, 5, 9, 10, 4, 7]
def scores_B : List ℝ := [6, 7, 7, 8, 6, 7, 8, 7, 9, 5]

-- Define helper functions for average and variance
def average (scores : List ℝ) : ℝ := (scores.sum / scores.length)

def variance (scores : List ℝ) : ℝ :=
  let μ := average scores
  (scores.map (λ x => (x - μ) ^ 2)).sum / scores.length

-- Theorem statement
theorem shooting_performance :
  average scores_A = 7 ∧ average scores_B = 7 ∧ 
  variance scores_A = 3 ∧ variance scores_B = 1.2 ∧
  (variance scores_B < variance scores_A) :=
by {
  sorry -- the actual proof is omitted
}

end

end shooting_performance_l591_591298


namespace solve_for_x_l591_591249

theorem solve_for_x (x : ℝ) : 10^(x + 3) = 1000^x → x = 3 / 2 :=
by
  intro h
  sorry

end solve_for_x_l591_591249


namespace population_net_increase_l591_591315

theorem population_net_increase :
  ( (birth_rate_every_two_seconds = 7) ∧ (death_rate_every_two_seconds = 3) ) →
  let net_increase_per_second := (birth_rate_every_two_seconds / 2) - (death_rate_every_two_seconds / 2)
  let seconds_per_day := 24 * 60 * 60
  in net_increase_per_second * seconds_per_day = 345,600 := 
begin
  sorry
end

end population_net_increase_l591_591315


namespace range_of_m_l591_591473

theorem range_of_m (m : ℝ) :
  let A := {x : ℝ | -2 < x ∧ x ≤ 5},
      B := {x : ℝ | -m + 1 ≤ x ∧ x ≤ 2 * m - 1}
  in B ⊆ A → m < 3 :=
by
  intros A B hB
  sorry

end range_of_m_l591_591473


namespace each_parent_payment_l591_591705

def original_salary : ℝ := 45000
def raise_percentage : ℝ := 0.2
def num_kids : ℕ := 9

def raise_amount : ℝ := original_salary * raise_percentage
def new_salary : ℝ := original_salary + raise_amount
def payment_per_parent : ℝ := new_salary / num_kids

theorem each_parent_payment (h1: raise_amount = 9000) (h2: new_salary = 54000) (h3: payment_per_parent = 6000) : payment_per_parent = 6000 :=
by
  sorry

end each_parent_payment_l591_591705


namespace probability_of_D_given_T_l591_591963

-- Definitions based on the conditions given in the problem.
def pr_D : ℚ := 1 / 400
def pr_Dc : ℚ := 399 / 400
def pr_T_given_D : ℚ := 1
def pr_T_given_Dc : ℚ := 0.05
def pr_T : ℚ := pr_T_given_D * pr_D + pr_T_given_Dc * pr_Dc

-- Statement to prove 
theorem probability_of_D_given_T : pr_T ≠ 0 → (pr_T_given_D * pr_D) / pr_T = 20 / 419 :=
by
  intros h1
  unfold pr_T pr_D pr_Dc pr_T_given_D pr_T_given_Dc
  -- Mathematical steps are skipped in Lean by inserting sorry
  sorry

-- Check that the statement can be built successfully
example : pr_D = 1 / 400 := by rfl
example : pr_Dc = 399 / 400 := by rfl
example : pr_T_given_D = 1 := by rfl
example : pr_T_given_Dc = 0.05 := by rfl
example : pr_T = (1 * (1 / 400) + 0.05 * (399 / 400)) := by rfl

end probability_of_D_given_T_l591_591963


namespace one_in_A_inter_B_l591_591160

def A : Set ℚ := { x | x > -1 }

def B : Set ℝ := { x | x < 2 }

theorem one_in_A_inter_B : (1 : ℝ) ∈ (A : Set ℝ) ∩ B := by
  sorry

end one_in_A_inter_B_l591_591160


namespace minimum_value_x_plus_four_over_x_minimum_value_occurs_at_x_eq_2_l591_591790

theorem minimum_value_x_plus_four_over_x (x : ℝ) (h : x ≥ 2) : 
  x + 4 / x ≥ 4 :=
by sorry

theorem minimum_value_occurs_at_x_eq_2 : ∀ (x : ℝ), x ≥ 2 → (x + 4 / x = 4 ↔ x = 2) :=
by sorry

end minimum_value_x_plus_four_over_x_minimum_value_occurs_at_x_eq_2_l591_591790


namespace minimum_distance_of_AB_l591_591986

noncomputable def f (x : ℝ) := Real.exp x + 1
noncomputable def g (x : ℝ) := 2 * x - 1

theorem minimum_distance_of_AB :
  |(f (Real.log 2) - g (Real.log 2))| = 4 - 2 * Real.log 2 :=
sorry

end minimum_distance_of_AB_l591_591986


namespace calc_fraction_product_l591_591369

theorem calc_fraction_product : 
  (7 / 4) * (8 / 14) * (14 / 8) * (16 / 40) * (35 / 20) * (18 / 45) * (49 / 28) * (32 / 64) = 49 / 200 := 
by sorry

end calc_fraction_product_l591_591369


namespace john_total_spent_is_correct_l591_591915

noncomputable def john_spent_total (original_cost : ℝ) (discount_rate : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let discounted_cost := original_cost - (discount_rate / 100 * original_cost)
  let cost_with_tax := discounted_cost + (sales_tax_rate / 100 * discounted_cost)
  let lightsaber_cost := 2 * original_cost
  let lightsaber_cost_with_tax := lightsaber_cost + (sales_tax_rate / 100 * lightsaber_cost)
  cost_with_tax + lightsaber_cost_with_tax

theorem john_total_spent_is_correct :
  john_spent_total 1200 20 8 = 3628.80 :=
by
  sorry

end john_total_spent_is_correct_l591_591915


namespace find_a_l591_591548

noncomputable def h : ℝ → ℝ :=
λ x, if x ≤ 0 then -x else 3 * x - 50

theorem find_a (a : ℝ) (ha : a < 0) : h (h (h 15)) = h (h (h a)) → a = -55 / 3 :=
by
  sorry

end find_a_l591_591548


namespace combined_average_age_l591_591181

theorem combined_average_age (avg_age_X avg_age_Y : ℕ) (num_X num_Y : ℕ)
  (h_avg_X : avg_age_X = 30) (h_avg_Y : avg_age_Y = 45) (h_num_X : num_X = 8) (h_num_Y : num_Y = 5) :
  (num_X * avg_age_X + num_Y * avg_age_Y) / (num_X + num_Y) = 36 :=
by
  have total_age_X := num_X * avg_age_X
  have total_age_Y := num_Y * avg_age_Y
  have combined_total_age := total_age_X + total_age_Y
  have combined_total_people := num_X + num_Y
  have combined_average := combined_total_age / combined_total_people
  rw [h_avg_X, h_avg_Y, h_num_X, h_num_Y]
  have h1 : total_age_X = 240 := by norm_num
  have h2 : total_age_Y = 225 := by norm_num
  have h3 : combined_total_age = 465 := by norm_num
  have h4 : combined_total_people = 13 := by norm_num
  have h5 : combined_average = 36 := by norm_num
  exact h5

end combined_average_age_l591_591181


namespace pencils_more_than_pens_l591_591991

-- Define the problem conditions
variables (pens pencils : ℕ)
variable (ratio : ℚ := 5 / 6)
variable (num_pencils : ℕ := 42)

-- Define the statement we need to prove
theorem pencils_more_than_pens :
  (ratio * num_pencils).toNat < num_pencils →
  num_pencils - (ratio * num_pencils).toNat = 7 :=
by
  sorry

end pencils_more_than_pens_l591_591991


namespace dimension_of_y_l591_591002

variable {a b : ℝ}

-- Definition of the problem conditions
def rectangle_area (a b : ℝ) : ℝ := a * b

def square_side_length (area : ℝ) : ℝ := Real.sqrt area

def side_of_square_half (s : ℝ) : ℝ := s / 2

theorem dimension_of_y (a b : ℝ) (h_a : a = 10) (h_b : b = 15) :
  side_of_square_half (square_side_length (rectangle_area a b)) = 5 * Real.sqrt 6 / 2 := by
  -- Declaration of the desired equality
  sorry

end dimension_of_y_l591_591002


namespace triangle_inequality_complex_l591_591523

theorem triangle_inequality_complex (
  a b c a1 b1 c1 a2 b2 c2 : ℝ
) (cond: (a1 : ℝ) (b1 : ℝ) (c1 : ℝ) (a2 : ℝ) (b2 : ℝ) (c2 : ℝ)) :
  a * a1 * a2 + b * b1 * b2 + c * c1 * c2 ≥ a * b * c :=
sorry

end triangle_inequality_complex_l591_591523


namespace police_emergency_number_has_prime_divisor_gt_seven_l591_591023

theorem police_emergency_number_has_prime_divisor_gt_seven (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
by
  sorry

end police_emergency_number_has_prime_divisor_gt_seven_l591_591023


namespace find_value_of_a_l591_591833

noncomputable def value_of_a (a : ℝ) (hyp_asymptotes_tangent_circle : Prop) : Prop :=
  a = (Real.sqrt 3) / 3 → hyp_asymptotes_tangent_circle

theorem find_value_of_a (a : ℝ) (condition1 : 0 < a)
  (condition_hyperbola : ∀ x y, x^2 / a^2 - y^2 = 1)
  (condition_circle : ∀ x y, x^2 + y^2 - 4*y + 3 = 0)
  (hyp_asymptotes_tangent_circle : Prop) :
  value_of_a a hyp_asymptotes_tangent_circle := 
sorry

end find_value_of_a_l591_591833


namespace police_emergency_number_has_prime_divisor_gt_seven_l591_591024

theorem police_emergency_number_has_prime_divisor_gt_seven (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
by
  sorry

end police_emergency_number_has_prime_divisor_gt_seven_l591_591024


namespace part_a_part_b_l591_591372

-- Part (a)
def is_good (N : ℕ) : Prop :=
  N = 1 ∨ ∃ (k : ℕ), 
  k % 2 = 0 ∧ 
  (∃ (factors : multiset ℕ), 
  (∀ p ∈ factors, nat.prime p) ∧ 
  factors.prod = N ∧ factors.card = k)

def P (x a b : ℕ) : ℕ := (x - a) * (x - b)

theorem part_a : 
  ∃ (a b : ℕ), a ≠ b ∧ (∀ n, n ≥ 1 ∧ n ≤ 2010 → is_good (P n a b)) :=
by
  use [1, 2]
  sorry

-- Part (b)
theorem part_b (a b : ℕ) (h : ∀ n, is_good (P n a b)) : a = b :=
by
  sorry

end part_a_part_b_l591_591372


namespace nested_r_30_l591_591928

def r (θ : ℝ) : ℝ := 1 / (2 - θ)

theorem nested_r_30 : r (r (r (r (r (r 30))))) = 22 / 23 :=
by sorry

end nested_r_30_l591_591928


namespace jack_received_7_emails_in_afternoon_l591_591905

theorem jack_received_7_emails_in_afternoon :
  ∃ (afternoon_emails : ℕ), 
    let morning_emails := 9 in
    let evening_emails := 7 in
    let difference := 2 in
    (morning_emails = evening_emails + difference) →
    (afternoon_emails = evening_emails) :=
begin
  sorry
end

end jack_received_7_emails_in_afternoon_l591_591905


namespace Ivan_can_plant_six_trees_l591_591985

-- Define the types of trees
inductive Tree
| apple
| pear
| plum
| apricot
| cherry
| almond

-- Define a vertex type
structure Vertex :=
  (id : Nat) -- Assuming vertices are uniquely identified by natural numbers

-- Define the triangle type as a set of three vertices
structure Triangle :=
  (v1 v2 v3 : Vertex)

-- State that there are six triangles
constant triangles : Fin 6 → Triangle

-- Prove that each vertex in these triangles contains exactly one type of tree with no overlaps
theorem Ivan_can_plant_six_trees :
  ∃ (placement : Vertex → Tree), 
    (∀ t1 t2, t1 ≠ t2 → 
      triangles t1 ≠ triangles t2) ∧ 
    (∀ t, ∃ v, 
      placement v = Tree.apple ∨ 
      placement v = Tree.pear ∨ 
      placement v = Tree.plum ∨ 
      placement v = Tree.apricot ∨ 
      placement v = Tree.cherry ∨ 
      placement v = Tree.almond) :=
sorry

end Ivan_can_plant_six_trees_l591_591985


namespace math_problem_l591_591236

theorem math_problem (n : ℕ) (h : n > 0) : 
  1957 ∣ (1721^(2*n) - 73^(2*n) - 521^(2*n) + 212^(2*n)) :=
sorry

end math_problem_l591_591236


namespace additional_payment_is_65_l591_591334

def installments (n : ℕ) : ℤ := 65
def first_payment : ℕ := 20
def first_amount : ℤ := 410
def remaining_payment (x : ℤ) : ℕ := 45
def remaining_amount (x : ℤ) : ℤ := 410 + x
def average_amount : ℤ := 455

-- Define the total amount paid using both methods
def total_amount (x : ℤ) : ℤ := (20 * 410) + (45 * (410 + x))
def total_average : ℤ := 65 * 455

theorem additional_payment_is_65 :
  total_amount 65 = total_average :=
sorry

end additional_payment_is_65_l591_591334


namespace find_k_find_a_l591_591467

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := real.logb 4 (4^x + 1) + k * x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := real.logb 4 (a * 2^x - 4/3 * a)

theorem find_k (k : ℝ) (condition : ∀ x, f x k = f (-x) k) : k = -1/2 := 
by sorry

theorem find_a (a : ℝ)
  (k : ℝ) (hk : k = -1/2)
  (condition : ∃ x, f x k = g x a) : a > 1 ∨ a = -3 :=
by sorry

end find_k_find_a_l591_591467


namespace distance_circumcenters_eq_AC_l591_591206

variables {A B C H A1 C1 A2 C2 O1 O2} [EuclideanGeometry]

-- Define the conditions
variable (triangle_ABC: Triangle A B C)
variable (altitude_A: altitude A H)
variable (altitude_C: altitude C H)
variable (is_acute_angled: acute_angled_triangle A B C)
variable (foot_A1: foot A1 A H)
variable (foot_C1: foot C1 C H)
variable (reflection_A2: reflection A2 A1 AC)
variable (reflection_C2: reflection C2 C1 AC)

-- Define the circumcenters
variable (circumcenter_O1: circumcenter O1 (Triangle C2 H A1))
variable (circumcenter_O2: circumcenter O2 (Triangle C1 H A2))

-- The theorem statement
theorem distance_circumcenters_eq_AC (h1: foot A1 A H)
                                      (h2: foot C1 C H)
                                      (h3: reflection A2 A1 AC)
                                      (h4: reflection C2 C1 AC)
                                      (h5: circumcenter O1 (Triangle C2 H A1))
                                      (h6: circumcenter O2 (Triangle C1 H A2)):
  dist O1 O2 = dist A C :=
sorry

end distance_circumcenters_eq_AC_l591_591206


namespace find_y_l591_591806

variable (x1 y1 x2 y2 : ℝ)

def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

theorem find_y (y : ℝ) :
  slope (-3) 7 5 y = 1/2 ↔ y = 11 :=
by
  unfold slope
  sorry

end find_y_l591_591806


namespace smallest_value_a_l591_591733

noncomputable def smallest_a_condition (a b : ℝ) : Prop :=
  a^4 + 2 * a^2 * b + 2 * a * b + b^2 = 960

theorem smallest_value_a : ∃ a b : ℝ, smallest_a_condition a b ∧ ∀ a' b' : ℝ, smallest_a_condition a' b' → a ≥ a' :=
begin
  use [-8, 0],
  -- Proof part is omitted.
  sorry,
end

end smallest_value_a_l591_591733


namespace lecturers_scheduling_l591_591711

theorem lecturers_scheduling (lecturers : Fin 6 → Type) :
  { schedule : List (Fin 6) // schedule.nodup } →
  (DrSmith DrJones DrAllen DrBrown DrLee DrWhite : Fin 6)
  (hSmithJones : ∃ (i j : Fin 6), i < j ∧ schedule[i] = DrJones ∧ schedule[j] = DrSmith)
  (hAllenBrown : ∃ (i j : Fin 6), i < j ∧ schedule[i] = DrBrown ∧ schedule[j] = DrAllen) :
  ℕ :=
begin
  -- According to the problem solution:
  -- We need to calculate the number of valid schedules
  let total_arrangements := factorial 6, -- 6!
  have h1 : total_arrangements / 2 / 2 = 180 := by sorry, -- Gives the valid number of schedules
  exact 180,
end

end lecturers_scheduling_l591_591711


namespace find_coordinates_of_D_l591_591397

-- Given conditions:
def polar_eq_rho (theta : ℝ) : ℝ := 2 * Real.cos theta

-- Convert polar equation to the parametric equation of the semicircle C
def parametric_eq_C (alpha : ℝ) : ℝ × ℝ :=
  (1 + Real.cos alpha, Real.sin alpha)

-- Define the slope of the line l
def line_l_slope := Real.sqrt 3

-- Define that D is on C and the tangent at D is perpendicular to l
def is_point_D (D : ℝ × ℝ) : Prop :=
  ∃ alpha ∈ Set.Icc (0 : ℝ) Real.pi, parametric_eq_C alpha = D ∧
  D.snd / (D.fst - 1) = line_l_slope

-- Main proof goal using Lean statements
theorem find_coordinates_of_D :
  is_point_D (3 / 2, Real.sqrt 3 / 2) :=
sorry

end find_coordinates_of_D_l591_591397


namespace chord_length_midpoint_polar_eq_l591_591816

-- Definitions from the conditions
def line_l_polar_eq (ρ θ : ℝ) : Prop := ρ * sin (θ - π / 3) = 0

noncomputable def curve_C_param_eq (α : ℝ) : ℝ × ℝ :=
  (2 * cos α, 2 + 2 * sin α)

-- Proving the length of the chord that line l cuts from curve C is 2√3
theorem chord_length (α : ℝ) :
  let eqn := (2 * (cos α)^2 + (2 + 2 * sin α - 2)^2 = 4) in
  let xc_y := curve_C_param_eq α in
  let (x, y) := xc_y in
  let d := abs 2 / sqrt (1 + (-(sqrt 3))^2) in
  let r := 2 in
  ρ * sqrt ((r^2) - (d^2)) = 2 * sqrt 3 := by
  sorry

-- Proving the polar equation for the locus of midpoints is ρ = 2 sin θ
theorem midpoint_polar_eq (ρ θ : ℝ) :
  let midpoint_x := ρ * cos θ in
  let midpoint_y := 1 - ρ * sin θ in
  (midpoint_x)^2 + (midpoint_y - 2)^2 = 4 →
  ρ = 2 * sin θ := by
  sorry

end chord_length_midpoint_polar_eq_l591_591816


namespace blue_books_count_l591_591999

def number_of_blue_books (R B : ℕ) (p : ℚ) : Prop :=
  R = 4 ∧ p = 3/14 → B^2 + 7 * B - 44 = 0

theorem blue_books_count :
  ∃ B : ℕ, number_of_blue_books 4 B (3/14) ∧ B = 4 :=
by
  sorry

end blue_books_count_l591_591999


namespace min_max_EM_FM_max_min_EM_FM_l591_591205

-- Definitions of conditions based on the problem
def square_side_length (a : ℝ) := a > 0

def point_within_square (a : ℝ) (x y : ℝ) := 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a

def distances (a b x y : ℝ) :=
  let em := Real.sqrt (x^2 + y^2 + a^2)
  let fm := Real.sqrt ((x - a)^2 + (y - a)^2 + b^2)
  (em, fm)

-- Problem statements to prove
theorem min_max_EM_FM (a b : ℝ) : 
  square_side_length a ∧ point_within_square a x y ∧ a < b ∧ b < a * Real.sqrt 3 →
  (∃ (x y : ℝ), point_within_square a x y ∧ 
  max (distances a b x y).fst (distances a b x y).snd = a * Real.sqrt 2) :=
sorry

theorem max_min_EM_FM (a b : ℝ) : 
  square_side_length a ∧ point_within_square a x y ∧ a < b ∧ b < a * Real.sqrt 3 →
  (∃ (x y : ℝ), point_within_square a x y ∧ 
  min (distances a b x y).fst (distances a b x y).snd = Real.sqrt ((9 * a^4 + b^4 + 2 * a^2 * b^2) / (8 * a^2))) :=
sorry

end min_max_EM_FM_max_min_EM_FM_l591_591205


namespace find_cost_price_l591_591697

-- Let's define the conditions given in a)
def SP : ℝ := 1180
def G_p : ℝ := 0.3111111111111111

-- Define the calculation for the cost price based on the conditions
def cost_price (SP : ℝ) (G_p : ℝ) : ℝ := SP / (1 + G_p)

-- Now, we state the theorem we need to prove
theorem find_cost_price : 
  cost_price SP G_p ≈ 900 := 
sorry

end find_cost_price_l591_591697


namespace greatest_integer_less_than_WY_l591_591890

theorem greatest_integer_less_than_WY 
  (W X Y Z M : Type) 
  [inhabited W] [inhabited X] [inhabited Y] [inhabited Z] [inhabited M]
  (WZ : ℝ) (hWZ : WZ = 150)
  (is_square : ∃ a : ℝ, a = 150 ∧ (∀ (W X Y Z : ℝ), a = WZ ∧ a = WX ∧ a = XY ∧ a = YZ))
  (M_midpoint : 2 * dist W Y = dist W M + dist M Y)
  (perpendicular : dist W Y * dist Z M = 0) : 
  greatest_int_lt (dist W Y) = 212 := 
by 
  sorry

end greatest_integer_less_than_WY_l591_591890


namespace ratio_new_circumference_new_diameter_l591_591176

variable {r n : ℝ} (h : n ≠ 0) (h' : r > 0)

theorem ratio_new_circumference_new_diameter (h : n ∈ ℤ) : 
  let new_radius := r + n
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * Real.pi * new_radius
  ∃ (r n : ℝ), 
  (Real.pi ≠ 0) → 
  (new_diameter ≠ 0) →
  new_circumference / new_diameter = Real.pi :=
by
  intro r n h h'
  let new_radius := r + n
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * Real.pi * new_radius
  use [r, n]
  assume h_realpi h_newdiameter
  sorry

end ratio_new_circumference_new_diameter_l591_591176


namespace find_S2017_l591_591997

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- Define the arithmetic sequence {a_n}
axiom a : ℕ → ℝ

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℝ := n * (a 1 + a n) / 2

axiom a_2_minus_2 : f (a 2 - 2) = Real.sin (2014 * Real.pi / 3)
axiom a_2016_minus_2 : f (a 2016 - 2) = Real.cos (2015 * Real.pi / 6)

theorem find_S2017 : S 2017 = 4072224 := sorry

end find_S2017_l591_591997


namespace smallest_fn_correct_l591_591102

noncomputable def smallest_fn (n : ℕ) : ℕ :=
  let m := n / 6 in
  match n % 6 with
  | 0 => 4 * m + 1
  | 1 => 4 * m + 1
  | 2 => 4 * m + 2
  | 3 => 4 * m + 3
  | 4 => 4 * m + 4
  | _ => 4 * m + 4

theorem smallest_fn_correct (n : ℕ) (h : n > 2) : 
  ∃ f, (∀ S, S ⊆ finset.range (n + 1) → finset.card S = f → ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ nat.coprime a b ∧ nat.coprime b c ∧ nat.coprime a c) ∧ 
  f = smallest_fn n :=
by sorry

end smallest_fn_correct_l591_591102


namespace inscribed_circle_radius_l591_591240

-- Define the radius of the sector
def sector_radius : ℝ := 6

-- Define the radius of the inscribed circle
def inscribed_radius := 6 * (real.sqrt 2 - 1)

-- The main theorem statement
theorem inscribed_circle_radius :
  ∃ r : ℝ, r + r * real.sqrt 2 = sector_radius ∧ r = inscribed_radius :=
begin
  use (6 * (real.sqrt 2 - 1)),
  split,
  { 
    sorry -- This part would include the detailed proof steps of the computation.
  },
  { 
    refl 
  }
end

end inscribed_circle_radius_l591_591240


namespace match_Tile_C_to_Rectangle_III_l591_591647

-- Define the structure for a Tile
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the given tiles
def Tile_A : Tile := { top := 5, right := 3, bottom := 7, left := 2 }
def Tile_B : Tile := { top := 3, right := 6, bottom := 2, left := 8 }
def Tile_C : Tile := { top := 7, right := 9, bottom := 1, left := 3 }
def Tile_D : Tile := { top := 1, right := 8, bottom := 5, left := 9 }

-- The proof problem: Prove that Tile C should be matched to Rectangle III
theorem match_Tile_C_to_Rectangle_III : (Tile_C = { top := 7, right := 9, bottom := 1, left := 3 }) → true := 
by
  intros
  sorry

end match_Tile_C_to_Rectangle_III_l591_591647


namespace solution_set_of_inequality_l591_591795

-- Define the differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable [Differentiable ℝ f]

-- State the conditions
def condition1 (x : ℝ) : Prop := 
  deriv f x < f x

def condition2 : Prop :=
  ∀ x : ℝ, f(-x) = -f(x + 1)

-- State the theorem with the conclusion
theorem solution_set_of_inequality (h_cond1 : ∀ x, condition1 f x)
                                  (h_cond2 : condition2 f) :
  {x : ℝ | f x < Real.exp x} = Ioi 0 :=
by
  sorry

end solution_set_of_inequality_l591_591795


namespace problem_solution_l591_591851

theorem problem_solution (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a + b = 1 :=
sorry

end problem_solution_l591_591851


namespace derivative_at_pi_l591_591465

noncomputable def f (x : ℝ) : ℝ := real.sqrt x * real.sin x

theorem derivative_at_pi :
  deriv f real.pi = -real.sqrt real.pi :=
by
  sorry

end derivative_at_pi_l591_591465


namespace max_equilateral_triangles_l591_591432

theorem max_equilateral_triangles (length : ℕ) (n : ℕ) (segments : ℕ) : 
  (length = 2) → (segments = 6) → (∀ t, 1 ≤ t ∧ t ≤ 4 → t = 4) :=
by 
  intros length_eq segments_eq h
  sorry

end max_equilateral_triangles_l591_591432


namespace problem1_problem2_l591_591001

theorem problem1 : (Real.tan (10 * Real.pi / 180) - Real.sqrt 3) * Real.sin (40 * Real.pi / 180) = -1 := 
by
  sorry

theorem problem2 (x : ℝ) : 
  (2 * Real.cos x ^ 4 - 2 * Real.cos x ^ 2 + 1 / 2) /
  (2 * Real.tan (Real.pi / 4 - x) * Real.sin (Real.pi / 4 + x) ^ 2) = 
  Real.sin (2 * x) / 4 := 
by
  sorry

end problem1_problem2_l591_591001


namespace oranges_to_put_back_l591_591054

variables (A O x : ℕ)

theorem oranges_to_put_back
    (h1 : 40 * A + 60 * O = 560)
    (h2 : A + O = 10)
    (h3 : (40 * A + 60 * (O - x)) / (10 - x) = 50) : x = 6 := 
sorry

end oranges_to_put_back_l591_591054


namespace scheduling_arrangements_correct_l591_591046

noncomputable def arrangements_count (n k : ℕ) := Nat.choose n k

def valid_scheduling_arrangements (total_employees : ℕ) (employees_per_day : ℕ) 
    (days : ℕ) (restrictions : List (ℕ × ℕ)) : ℕ :=
  -- total arrangements without restrictions
  let total_arrangements := arrangements_count total_employees employees_per_day * arrangements_count (total_employees - employees_per_day) employees_per_day
  -- arrangements excluding those where employee A works on the 14th or employee B works on the 16th
  let invalid_A_on_14th := arrangements_count (total_employees - 1) employees_per_day * arrangements_count (total_employees - employees_per_day) employees_per_day
  let invalid_B_on_16th := arrangements_count (total_employees - 1) employees_per_day * arrangements_count (total_employees - employees_per_day) employees_per_day
  -- overlapping invalid arrangements where both employee A on the 14th and employee B on the 16th
  let overlap_A_on_14th_B_on_16th := arrangements_count (total_employees - 2) employees_per_day * arrangements_count (total_employees - employees_per_day + 1) employees_per_day
  total_arrangements - 2 * invalid_A_on_14th + overlap_A_on_14th_B_on_16th

theorem scheduling_arrangements_correct :
  valid_scheduling_arrangements 6 2 3 [(1, 0), (2, 2)] = 42 :=
by
  sorry

end scheduling_arrangements_correct_l591_591046


namespace greatest_multiple_of_4_l591_591253

theorem greatest_multiple_of_4 (x : ℕ) (h₀ : 0 < x) (h₁ : x % 4 = 0) (h₂ : x^3 < 5000) : x ≤ 16 :=
by {
  -- Skipping the proof as per instructions.
  sorry,
}

example : ∃ x : ℕ, 0 < x ∧ x % 4 = 0 ∧ x^3 < 5000 ∧ x = 16 :=
by {
  use 16,
  split,
  -- Proof steps for 0 < 16
  exact nat.zero_lt_bit0 nat.zero_lt_bit0,
  split,
  -- Proof steps for 16 % 4 = 0
  norm_num,
  split,
  -- Proof steps for 16^3 < 5000
  norm_num,
  -- Proof that x = 16
  refl,
}

end greatest_multiple_of_4_l591_591253


namespace B_subset_M_M_closed_under_mul_l591_591782

def M : Set ℤ := {a : ℤ | ∃ x y : ℤ, a = x^2 - y^2}

def B : Set ℤ := {b : ℤ | ∃ n : ℕ, b = 2 * n + 1}

theorem B_subset_M : ∀ b ∈ B, b ∈ M :=
by
  intros b hB
  cases hB with n hn
  use n + 1, n
  rw hn
  exact nat.cast_add_one n - n.val ^ 2_eq

theorem M_closed_under_mul : ∀ a1 a2 ∈ M, a1 * a2 ∈ M :=
by
  intros a1 h1 a2 h2
  cases h1 with x1 hx1
  cases h1 with y1 hy1
  cases h2 with x2 hx2
  cases h2 with y2 hy2
  use x1 * x2 + y1 * y2, x1 * y2 + x2 * y1
  rw [hx1, hx2, mul_add, add_mul, add_mul]
  ring

end B_subset_M_M_closed_under_mul_l591_591782


namespace inequality_solution_l591_591251

theorem inequality_solution (a : ℝ) (h : a > 0) :
  (if a = 2 then {x : ℝ | false}
   else if 0 < a ∧ a < 2 then {x : ℝ | 1 < x ∧ x ≤ 2 / a}
   else if a > 2 then {x : ℝ | 2 / a ≤ x ∧ x < 1}
   else ∅) =
    {x : ℝ | (a + 2) * x - 4 ≤ 2 * (x - 1)} :=
by
  sorry

end inequality_solution_l591_591251


namespace common_points_circle_ellipse_l591_591079

theorem common_points_circle_ellipse :
    (∃ (p1 p2: ℝ × ℝ),
        p1 ≠ p2 ∧
        (p1, p2).fst.1 ^ 2 + (p1, p2).fst.2 ^ 2 = 4 ∧
        9 * (p1, p2).fst.1 ^ 2 + 4 * (p1, p2).fst.2 ^ 2 = 36 ∧
        (p1, p2).snd.1 ^ 2 + (p1, p2).snd.2 ^ 2 = 4 ∧
        9 * (p1, p2).snd.1 ^ 2 + 4 * (p1, p2).snd.2 ^ 2 = 36) :=
sorry

end common_points_circle_ellipse_l591_591079


namespace simplify_complex_expr_l591_591598

theorem simplify_complex_expr (a b : ℂ) (hz : b = 7 * complex.I) (ha : a = 4) :
  (a + b) / (a - b) + (a - b) / (a + b) = -66 / 65 := sorry

end simplify_complex_expr_l591_591598


namespace new_truck_distance_l591_591021

-- Define the initial conditions
def old_truck_distance : ℝ := 150
def increase_percentage : ℝ := 0.30

-- Theorem to prove the distance traveled by the newer truck
theorem new_truck_distance :
  let new_truck_distance := old_truck_distance + (increase_percentage * old_truck_distance)
  in new_truck_distance = 195 :=
by
  sorry

end new_truck_distance_l591_591021


namespace javier_first_throw_l591_591908

theorem javier_first_throw 
  (second third first : ℝ)
  (h1 : first = 2 * second)
  (h2 : first = (1 / 2) * third)
  (h3 : first + second + third = 1050) :
  first = 300 := 
by sorry

end javier_first_throw_l591_591908


namespace quadratic_rational_root_contradiction_l591_591582

def int_coefficients (a b c : ℤ) : Prop := true  -- Placeholder for the condition that coefficients are integers

def is_rational_root (a b c p q : ℤ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ p.gcd q = 1 ∧ a * p^2 + b * p * q + c * q^2 = 0  -- p/q is a rational root in simplest form

def ear_even (b c : ℤ) : Prop :=
  b % 2 = 0 ∨ c % 2 = 0

def assume_odd (a b c : ℤ) : Prop :=
  a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0

theorem quadratic_rational_root_contradiction (a b c p q : ℤ)
  (h1 : int_coefficients a b c)
  (h2 : a ≠ 0)
  (h3 : is_rational_root a b c p q)
  (h4 : ear_even b c) :
  assume_odd a b c :=
sorry

end quadratic_rational_root_contradiction_l591_591582


namespace terminate_decimal_representation_count_l591_591103

theorem terminate_decimal_representation_count :
  let num_factors := 63 in
  let max_bound := 2000 in
  (∀ n: ℕ, (1 ≤ n ∧ n ≤ max_bound)  → (num_factors ∣ n)) =
    31 :=
begin
  sorry
end

end terminate_decimal_representation_count_l591_591103


namespace positive_int_values_satisfy_inequality_l591_591849

theorem positive_int_values_satisfy_inequality :
  {x : ℕ // x > 0 ∧ 12 < -2 * x + 16}.card = 1 :=
by
  sorry

end positive_int_values_satisfy_inequality_l591_591849


namespace diamonds_in_design_G10_l591_591883

noncomputable def T : ℕ → ℕ
| 1     := 1
| (n+1) := T n + (n + 1) ^ 2

theorem diamonds_in_design_G10 : T 10 = 385 := sorry

end diamonds_in_design_G10_l591_591883


namespace least_sales_needed_not_lose_money_l591_591318

noncomputable def old_salary : ℝ := 75000
noncomputable def new_salary_base : ℝ := 45000
noncomputable def commission_rate : ℝ := 0.15
noncomputable def sale_amount : ℝ := 750

theorem least_sales_needed_not_lose_money : 
  ∃ (n : ℕ), n * (commission_rate * sale_amount) ≥ (old_salary - new_salary_base) ∧ n = 267 := 
by
  -- The proof will show that n = 267 is the least number of sales needed to not lose money.
  existsi 267
  sorry

end least_sales_needed_not_lose_money_l591_591318


namespace eval_expression_l591_591766

theorem eval_expression : (3000 * (3000 ^ 2998) * (3000 ^ -1)) = (3000 ^ 2998) :=
by sorry

end eval_expression_l591_591766


namespace absolute_value_simplification_l591_591448

theorem absolute_value_simplification (a b : ℝ) (ha : a < 0) (hb : b > 0) : |a - b| + |b - a| = -2 * a + 2 * b := 
by 
  sorry

end absolute_value_simplification_l591_591448


namespace angles_of_triangle_range_of_values_l591_591545

-- Part 1: Proof of angles A and B when C = π/4
theorem angles_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : C = π / 4) 
  (h2 : a^2 - b^2 = (a^2 + b^2 - c^2) * c^2 / (a * b)) : 
  A = 5 * π / 8 ∧ B = π / 8 :=
sorry

-- Part 2: Proof of range of values for a / (b * cos^2 B) if the triangle is acute
theorem range_of_values (a b c : ℝ) (A B C : ℝ) 
  (h1 : a^2 - b^2 = (a^2 + b^2 - c^2) * c^2 / (a * b))
  (h2 : A < π / 2) 
  (h3 : B < π / 2) 
  (h4 : C < π / 2) : 
  2 < a / (b * (cos B)^2) ∧ a / (b * (cos B)^2) < 8 / 3 :=
sorry

end angles_of_triangle_range_of_values_l591_591545


namespace increasing_interval_of_even_fn_l591_591867

theorem increasing_interval_of_even_fn
  (a : ℝ)
  (h : ∀ x : ℝ, (a-2)*x^2 + (a-1)*x + 3 = (a-2)*(-x)^2 + (a-1)*(-x) + 3) :
  {x : ℝ | f (x) = (a-2)*x^2 + (a-1)*x + 3 ∧ f (x) = -x^2 + 3 }.increasing_on (Iic 0) := by
  sorry

end increasing_interval_of_even_fn_l591_591867


namespace chair_cost_l591_591569

/--
Nadine went to a garage sale and spent $56. She bought a table for $34 and 2 chairs.
Each chair cost the same amount.
Prove that one chair cost $11.
-/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) (total_cost : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  total_cost = 56 - 34 →
  total_cost / num_chairs = 11 :=
by
  sorry

end chair_cost_l591_591569


namespace tangent_line_at_point_inequality_proof_l591_591826

def f (t x : ℝ) : ℝ := t * x - (t - 1) * real.log x - t

theorem tangent_line_at_point (x : ℝ) (h : x = 1) : 
  ∀ t, t = 2 → (∃ (m b : ℝ), m = 1 ∧ b = -1 ∧ (∀ y, y = f x 1 → (y = x - 1))) :=
by
  sorry

theorem inequality_proof (t x : ℝ) (h1 : t ≤ 0) (h2 : x > 1) : 
  f t x < real.exp (x - 1) - 1 :=
by
  sorry

end tangent_line_at_point_inequality_proof_l591_591826


namespace cylinder_inscribed_in_prism_iff_base_polygon_allows_circle_l591_591655

theorem cylinder_inscribed_in_prism_iff_base_polygon_allows_circle (P : Type) [Polygon P] :
  (∃ C : Cylinder, inscribed C P) ↔ (∃ circ : Circle, inscribed circ P.base) :=
sorry

end cylinder_inscribed_in_prism_iff_base_polygon_allows_circle_l591_591655


namespace ellipse_equation_and_k_value_l591_591122

theorem ellipse_equation_and_k_value (a b c : ℝ) (h1 : a > b > 0) (h2 : c / a = sqrt 3 / 2) (h3 : 2 * a = 4)
                                      (k : ℝ) : 
  ((∃ ellipse_eq : ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1, ellipse_eq = (λ x y, (x^2) / 4 + y^2 = 1)) ∧
  (∃ k_val : k = sqrt 11 / 2 ∨ k = -sqrt 11 / 2, 
    ∃ l : ℝ → ℝ, l = (λ x, k * x - sqrt 3) ∧
    ∃ A B : ℝ × ℝ, 
      (∃ x1 x2 : ℝ, x1 + x2 = 8 * sqrt 3 * k / (1 + 4 * k^2) 
        ∧ x1 * x2 = 8 / (1 + 4 * k^2)) ∧
      (∃ y1 y2 : ℝ, y1 = l x1 ∧ y2 = l x2 ∧ x1 * x2 + y1 * y2 = 0))) :=
begin
  sorry
end

end ellipse_equation_and_k_value_l591_591122


namespace angle_range_of_extreme_value_l591_591125

-- Define the non-zero vectors a and b
variables (a b : EuclideanSpace ℝ n) (ha : a ≠ 0) (hb : b ≠ 0)

-- Define the condition |a| = sqrt(3) * |b|
def magnitude_condition : Prop :=
  ‖a‖ = sqrt 3 * ‖b‖

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
  1 / 3 * x^3 + ‖a‖ * x^2 + 2 * (a ⬝ b) * x + 1

-- State the theorem
theorem angle_range_of_extreme_value (hmag : magnitude_condition a b) :
  ∃ θ ∈ Icc 0 π, cos θ < sqrt (3 / 4) :=
sorry

end angle_range_of_extreme_value_l591_591125


namespace sixth_triangular_number_l591_591720

theorem sixth_triangular_number : (∑ i in finset.range 6.succ, i) = 21 := by
  sorry

end sixth_triangular_number_l591_591720


namespace inequality_proof_l591_591165

variable (a b : ℝ)

theorem inequality_proof (h : a < b) : 1 - a > 1 - b :=
sorry

end inequality_proof_l591_591165


namespace max_mom_money_difference_l591_591942

theorem max_mom_money_difference:
  let tuesday_amount := 8 in
  let wednesday_amount := 5 * tuesday_amount in
  let thursday_amount := wednesday_amount + 9 in
  (thursday_amount - tuesday_amount = 41) :=
by
  sorry

end max_mom_money_difference_l591_591942


namespace parabola_value_of_a_l591_591626

theorem parabola_value_of_a (t : ℝ) (a b c : ℝ) :
  (t = 15) →
  (∀ x, (x, 0) ∈ {(4 : ℝ, 0), (t/3, 0)}) →
  (0, 60) ∈ {(0 : ℝ, y) | y = a * 0^2 + b * 0 + c} →
  (∀ x, y = a * x^2 + b * x + c → (x, y) = (4, 0) ∨ (x, y) = (t/3, 0) ∨ (x, y) = (0, 60)) →
  a = 3 :=
by
  intros ht hroots hy_int hy_eqn
  sorry

end parabola_value_of_a_l591_591626


namespace x_intercept_of_line_l591_591770

theorem x_intercept_of_line (x y : ℚ) (h_eq : 4 * x + 7 * y = 28) (h_y : y = 0) : (x, y) = (7, 0) := 
by 
  sorry

end x_intercept_of_line_l591_591770


namespace minimum_people_in_troupe_l591_591015

-- Let n be the number of people in the troupe.
variable (n : ℕ)

-- Conditions: n must be divisible by 8, 10, and 12.
def is_divisible_by (m k : ℕ) := m % k = 0
def divides_all (n : ℕ) := is_divisible_by n 8 ∧ is_divisible_by n 10 ∧ is_divisible_by n 12

-- The minimum number of people in the troupe that can form groups of 8, 10, or 12 with none left over.
theorem minimum_people_in_troupe (n : ℕ) : divides_all n → n = 120 :=
by
  sorry

end minimum_people_in_troupe_l591_591015


namespace simplify_complex_fractions_l591_591594

theorem simplify_complex_fractions :
  let a := 4
  let b := 7
  let i := Complex.I
  (a + b * i) / (a - b * i) + (a - b * i) / (a + b * i) = -66/65 :=
by
  let a := 4
  let b := 7
  let i := Complex.I
  have h1 : (a + b * i) / (a - b * i) = (Complex (a + b * i) : ℂ) / Complex (a - b * i),
  from rfl,
  have h2 : (a - b * i) / (a + b * i) = (Complex (a - b * i) : ℂ) / Complex (a + b * i),
  from rfl,
  rw [h1, h2],
  -- blocks for the calculation steps
  sorry

end simplify_complex_fractions_l591_591594


namespace largest_prime_divisor_to_check_l591_591291

theorem largest_prime_divisor_to_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) : 
  ∃ p, p ≤ 33 ∧ prime p ∧ (∀ q, prime q → q ≤ p → ¬q ∣ n) :=
by
  sorry

end largest_prime_divisor_to_check_l591_591291


namespace shekar_average_marks_l591_591683

-- Define the given conditions as constants
def marks_math : ℕ := 76
def marks_science : ℕ := 65
def marks_social_studies : ℕ := 82
def marks_english : ℕ := 67
def marks_biology : ℕ := 55

-- Define the average function
noncomputable def average (a b c d e : ℕ) : ℕ :=
  (a + b + c + d + e) / 5

-- Define the statement to prove
theorem shekar_average_marks : 
  average marks_math marks_science marks_social_studies marks_english marks_biology = 69 :=
by
  -- Proof would go here
  sorry

end shekar_average_marks_l591_591683


namespace police_emergency_number_prime_divisor_l591_591034

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 1000 * k + 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n :=
sorry

end police_emergency_number_prime_divisor_l591_591034


namespace indices_sum_eq_19_l591_591288

theorem indices_sum_eq_19 
  (r : ℕ)
  (n : ℕ → ℕ)
  (a : ℕ → ℤ)
  (h_unique_n : ∀ i j, (i < j ↔ n i > n j))
  (h_unique_a : ∀ k, a k = 1 ∨ a k = -1)
  (h_sum_eq : ∑ k in finset.range r, a k * 3 ^ n k = 2022)
  (h_distinct : ∀ i j, i ≠ j → n i ≠ n j) : 
  finset.sum (finset.range r) n = 19 :=
by {
  sorry,
}

end indices_sum_eq_19_l591_591288


namespace taxi_fare_l591_591864

theorem taxi_fare (fare_first : ℝ) (total_fare : ℝ) (increments : ℝ) (fare_per_increment : ℝ) :
  fare_first = 10 ∧ total_fare = 59 ∧ increments = 50 ∧ fare_per_increment = (59 - 10) / 49 →
  fare_per_increment = 1 :=
by
  intros h,
  cases h with h_first rest,
  cases rest with h_total rest2,
  cases rest2 with h_increments h_fare_per_increment,
  sorry

end taxi_fare_l591_591864


namespace minimum_m_n_squared_l591_591446

theorem minimum_m_n_squared (a b c m n : ℝ) (h1 : c > a) (h2 : c > b) (h3 : c = Real.sqrt (a^2 + b^2)) 
    (h4 : a * m + b * n + c = 0) : m^2 + n^2 ≥ 1 := by
  sorry

end minimum_m_n_squared_l591_591446


namespace maximum_value_f_l591_591618

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) ^ x - Real.log2 (x + 2)

theorem maximum_value_f :
  ∃ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x = 3 :=
begin
  use -1,
  split,
  { norm_num },
  { unfold f,
    norm_num,
    exact Real.log2_one }
end

end maximum_value_f_l591_591618


namespace perfect_square_subseq_exists_l591_591111

theorem perfect_square_subseq_exists (n N : ℕ) (a : ℕ → ℕ)
  (b : ℕ → ℕ) (hb : ∀ i, 1 ≤ i → i ≤ N → b i ∈ {j | 1 ≤ j ∧ j ≤ n})
  (hN : N ≥ 2^n) :
  ∃ (k h : ℕ), 1 ≤ k ∧ k ≤ h ∧ h ≤ N ∧
  (∃ m : ℕ, (∏ i in finset.Icc k h, b i) = m^2) :=
sorry

end perfect_square_subseq_exists_l591_591111


namespace orange_juice_is_25_percent_l591_591701

-- Conditions
def total_drink : ℝ := 200 -- Total drink is 200 ounces
def watermelon_percent : ℝ := 40 -- 40 percent watermelon juice
def grape_juice : ℝ := 70 -- 70 ounces of grape juice

-- Definitions based on conditions
def watermelon_juice : ℝ := (watermelon_percent / 100) * total_drink
def orange_juice : ℝ := total_drink - watermelon_juice - grape_juice
def orange_juice_percent : ℝ := (orange_juice / total_drink) * 100

-- Prove that the percentage of orange juice is 25
theorem orange_juice_is_25_percent : orange_juice_percent = 25 := by
  sorry

end orange_juice_is_25_percent_l591_591701


namespace intersect_chord_prob_l591_591439

noncomputable def chord_probability :=
  let circle1 := ∀ x y : ℝ, x^2 + y^2 = 4
  let circle2 (a : ℝ) := ∀ x y : ℝ, (x - a)^2 + y^2 = 4
  let a_interval := set.Ioo 0 6
  let valid_interval := set.Ioo 2 4
  have intersect_common_chord (a : ℝ) : set_of (λ a, ∃ x y : ℝ, circle1 x y ∧ circle2 a x y) ⊆ valid_interval :=
    sorry
  (valid_interval.center_ratio a_interval.length) = 1/3

theorem intersect_chord_prob : chord_probability := sorry

end intersect_chord_prob_l591_591439


namespace area_of_park_l591_591628

-- Definitions of conditions
def ratio_length_breadth (L B : ℝ) : Prop := L / B = 1 / 3
def cycling_time_distance (speed time perimeter : ℝ) : Prop := perimeter = speed * time

theorem area_of_park :
  ∃ (L B : ℝ),
    ratio_length_breadth L B ∧
    cycling_time_distance 12 (8 / 60) (2 * (L + B)) ∧
    L * B = 120000 := by
  sorry

end area_of_park_l591_591628


namespace log_sqrt_30_in_terms_of_a_and_b_l591_591133

variable (a b : ℝ)
variable (h₁ : Real.log 3 2 = a)
variable (h₂ : 3^b = 5)

theorem log_sqrt_30_in_terms_of_a_and_b :
  Real.log 3 (Real.sqrt 30) = (1 / 2) * (a + b + 1) :=
sorry

end log_sqrt_30_in_terms_of_a_and_b_l591_591133


namespace N_p_le_p_c_l591_591919

def N (p : ℕ) : ℕ :=
  (Finset.filter (λ x : ℕ, x ^ x % p = 1) (Finset.range (p + 1))).card

theorem N_p_le_p_c (c : ℝ) (h1 : c < 1/2) (p0 : ℕ) :
  ∀ (p : ℕ), p.prime → p ≥ p0 → N p ≤ p ^ c := by {
  -- sorry is added to skip the actual proof
  sorry
}

end N_p_le_p_c_l591_591919


namespace hyperbola_equation_l591_591144

theorem hyperbola_equation (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_vertex : {p : ℝ × ℝ // p.1 = a ∧ p.2 = 0})
  (h_eccentricity : real.sqrt 3 = real.sqrt ((a * a + b * b) / a * a))
  (h_parallel : ∀ (p1 p2 : ℝ × ℝ), p1 = (0, -2) → p2 = (a, 0) → (2 / a) = (b / a)) :
  (\frac{x^2}{2} - \frac{y^2}{4} = 1) :=
by
  sorry

end hyperbola_equation_l591_591144


namespace area_of_triangle_ABC_l591_591761

variable (A B C : ℝ × ℝ)
variable (x1 y1 x2 y2 x3 y3 : ℝ)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC :
  let A := (1, 2)
  let B := (-2, 5)
  let C := (4, -2)
  area_of_triangle A B C = 1.5 :=
by
  sorry

end area_of_triangle_ABC_l591_591761


namespace simultaneous_inequalities_l591_591309

theorem simultaneous_inequalities (x : ℝ) 
    (h1 : x^3 - 11 * x^2 + 10 * x < 0) 
    (h2 : x^3 - 12 * x^2 + 32 * x > 0) : 
    (1 < x ∧ x < 4) ∨ (8 < x ∧ x < 10) :=
sorry

end simultaneous_inequalities_l591_591309


namespace ellipse_hyperbola_foci_coincide_l591_591768

theorem ellipse_hyperbola_foci_coincide :
  ∀ (b^2 : ℝ), (foci_coincide : ℝ)
  (ellipse_eq : ∀ (x y : ℝ), x^2 / 25 + y^2 /= b^2 → 1)
  (hyperbola_eq : ∀ (x y : ℝ), x^2 / 196 - y^2 / 121 ≠ 1 / 49 → true) :
  b^2 = 908 / 49 :=
by
  sorry

end ellipse_hyperbola_foci_coincide_l591_591768


namespace hexagon_area_proof_l591_591586

-- Define the points A and C
def A : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (8, 2)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the side length of the hexagon
def s : ℝ := distance A C

-- The area of an equilateral triangle with a given side length
def equilateral_triangle_area (side : ℝ) : ℝ := (real.sqrt 3 / 4) * side^2

-- The area of the regular hexagon consisting of 6 equilateral triangles
def hexagon_area (side : ℝ) : ℝ := 6 * equilateral_triangle_area side

-- Theorem statement
theorem hexagon_area_proof : hexagon_area s = 75 * real.sqrt 3 := by
  sorry

end hexagon_area_proof_l591_591586


namespace angle_of_extended_sides_of_octagon_l591_591245

theorem angle_of_extended_sides_of_octagon :
  ∀ (A B C D E F G H Q : Type),
    -- Conditions
    regular_polygon A B C D E F G H 8 →
    extended_sides_meet C D G H Q →
    -- Conclusion
    angle_at Q = 90 :=
by
  sorry

end angle_of_extended_sides_of_octagon_l591_591245


namespace problem_A_correct_l591_591210

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q 

-- Definition of the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(finset.range n).sum a

-- Definition of the product of the first n terms
def product_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(finset.range n).prod a

theorem problem_A_correct (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  a 1 > 1 →
  a 2019 * a 2020 > 1 →
  (a 2019 - 1) / (a 2020 - 1) < 0 →
  sum_first_n_terms a 2019 < sum_first_n_terms a 2020 :=
begin
  intros h_geo h_a1 h_a2019_2020 h_fraction,
  sorry
end

end problem_A_correct_l591_591210


namespace simplify_and_evaluate_expression_l591_591247

theorem simplify_and_evaluate_expression (m : ℝ) (h : (m + 2) * (m - 3) = 0) : 
  (m = -2 → (m^2 - 9) / (m^2 - 6*m + 9) - 3 / (m-3) ÷ (m^2 / m^3) = -4 / 5) :=
by
  sorry

end simplify_and_evaluate_expression_l591_591247


namespace conical_tent_volume_l591_591667

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

theorem conical_tent_volume :
  let d := 20 in
  let h := 10 in
  let r := d / 2 in
  volume_of_cone r h = 1000 / 3 * π :=
by
  sorry

end conical_tent_volume_l591_591667


namespace round_trip_average_gas_mileage_l591_591696

theorem round_trip_average_gas_mileage :
  ∀ (d1 d2: ℕ) (mpg1 mpg2 total_miles total_gas mileage: ℚ),
    d1 = 150 →
    d2 = 150 →
    mpg1 = 25 →
    mpg2 = 15 →
    total_miles = d1 + d2 →
    total_gas = d1 / mpg1 + d2 / mpg2 →
    mileage = total_miles / total_gas →
    mileage = 18.75 :=
by
  intros d1 d2 mpg1 mpg2 total_miles total_gas mileage
  assume h1 h2 h3 h4 h5 h6 h7
  -- Sorry to skip the proof
  sorry

end round_trip_average_gas_mileage_l591_591696


namespace tangent_line_at_point_inequality_proof_l591_591827

def f (t x : ℝ) : ℝ := t * x - (t - 1) * real.log x - t

theorem tangent_line_at_point (x : ℝ) (h : x = 1) : 
  ∀ t, t = 2 → (∃ (m b : ℝ), m = 1 ∧ b = -1 ∧ (∀ y, y = f x 1 → (y = x - 1))) :=
by
  sorry

theorem inequality_proof (t x : ℝ) (h1 : t ≤ 0) (h2 : x > 1) : 
  f t x < real.exp (x - 1) - 1 :=
by
  sorry

end tangent_line_at_point_inequality_proof_l591_591827


namespace determine_coins_in_38_bags_l591_591355

/-- Ali Baba has 40 bags of coins. Given that the genie can determine the 
    number of coins in each of two specified bags at the cost of one coin 
    being taken from one of the bags, prove that Ali Baba can determine the 
    number of coins in 38 out of 40 bags after no more than 100 procedures. 
    Each bag initially contains at least 1000 coins. -/
theorem determine_coins_in_38_bags : 
  (∀ (bags : Fin 40 → ℕ), (∀ i, 1000 ≤ bags i) → 
  (∀ (compare: (Fin 40 × Fin 40)) → ℕ × ℕ,
    ∃ (steps : Fin 100 → (Fin 40 × Fin 40)),
    ∃ (coins : Fin 40 → ℕ),
    (∀ t : Fin 38, 
      coins t = bags t - (count (λ x : (Fin 40 × Fin 40), 
                               t = x.1 ∨ t = x.2) (steps 0)))) → 
  ∃ final_step_set : Fin 38 → Fin 40, 
  (∀ f, coins (final_step_set f) = bags (final_step_set f)) :=
begin
  sorry
end

end determine_coins_in_38_bags_l591_591355


namespace cost_formula_correct_l591_591723

def cost_of_ride (T : ℤ) : ℤ :=
  if T > 5 then 10 + 5 * T - 10 else 10 + 5 * T

theorem cost_formula_correct (T : ℤ) : cost_of_ride T = 10 + 5 * T - (if T > 5 then 10 else 0) := by
  sorry

end cost_formula_correct_l591_591723


namespace mary_change_l591_591226

/-- 
Calculate the change Mary will receive after buying tickets for herself and her 3 children 
at the circus, given the ticket prices and special group rate discount.
-/
theorem mary_change :
  let adult_ticket := 2
  let child_ticket := 1
  let discounted_child_ticket := 0.5 * child_ticket
  let total_cost_with_discount := adult_ticket + 2 * child_ticket + discounted_child_ticket
  let payment := 20
  payment - total_cost_with_discount = 15.50 :=
by
  sorry

end mary_change_l591_591226


namespace magnitude_order_l591_591811

noncomputable def f (x : ℝ) : ℝ := 2 ^ x - 2 ^ (-x)

def a : ℝ := ( (7 / 9 : ℝ) ^ (-1 / 4))
def b : ℝ := ( (9 / 7 : ℝ) ^ (1 / 5))
def c : ℝ := Real.log2 (7 / 9 : ℝ)

theorem magnitude_order (ha : a = ( (9 / 7 : ℝ) ^ (1 / 4)))
                        (hb : b = ( (9 / 7 : ℝ) ^ (1 / 5)))
                        (hc : c = Real.log2 (7 / 9 : ℝ)) : f c < f b < f a :=
by
  sorry

end magnitude_order_l591_591811


namespace greatest_sum_of_other_two_roots_l591_591989

noncomputable def polynomial (x : ℝ) (k : ℝ) : ℝ :=
  x^3 - k * x^2 + 20 * x - 15

theorem greatest_sum_of_other_two_roots (k x1 x2 : ℝ) (h : polynomial 3 k = 0) (hx : x1 * x2 = 5)
  (h_prod_sum : 3 * x1 + 3 * x2 + x1 * x2 = 20) : x1 + x2 = 5 :=
by
  sorry

end greatest_sum_of_other_two_roots_l591_591989


namespace midpoint_distance_l591_591337

noncomputable def midpoint_distance_to_directrix
  (x1 x2 : ℝ) (y1 y2 : ℝ)
  (h_parabola_1 : y1^2 = 4*x1) 
  (h_parabola_2 : y2^2 = 4*x2) 
  (h_length : (y1 - y2)^2 = 4 * (x1 + x2))
  (h_chord_length : real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 7) :
  real :=
  let Mx : ℝ := (x1 + x2) / 2 in
  let My : ℝ := (y1 + y2) / 2 in
  Mx + 1

theorem midpoint_distance (x1 x2 : ℝ) (y1 y2 : ℝ)
  (h_parabola_1 : y1^2 = 4*x1) 
  (h_parabola_2 : y2^2 = 4*x2) 
  (h_length : (y1 - y2)^2 = 4 * (x1 + x2))
  (h_chord_length : real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 7) :
  midpoint_distance_to_directrix x1 x2 y1 y2 h_parabola_1 h_parabola_2 h_length h_chord_length = 7 / 2 :=
sorry

end midpoint_distance_l591_591337


namespace monotonicity_and_value_range_l591_591789

noncomputable theory

open Real

def f (a x : ℝ) : ℝ := (x - 1) * exp x - a * (x^2 + 1)

theorem monotonicity_and_value_range :
  ∀ (a : ℝ), (∀ x : ℝ, 1 ≤ x → f a x ≥ -2 * a + log x) → a ≤ (exp 1 - 1) / 2 :=
by
  sorry

end monotonicity_and_value_range_l591_591789


namespace simplify_mul_fractions_l591_591602

noncomputable def simplify (a b c d e f g h : ℚ) : ℚ :=
  a * (b / c) * (d / e) * (f / g) * (h / e)

theorem simplify_mul_fractions :
  simplify 8 (15 : ℚ) 9 (-21) 35 1 1 (-1) / 3 = -8 / 3 := by
  sorry

#simplify_mul_fractions

end simplify_mul_fractions_l591_591602


namespace solve_equation_l591_591959

theorem solve_equation :
  ∃ x : ℂ, (x - 2)^6 + (x - 6)^6 = 64 ∧ (x = 4 + complex.I * real.sqrt 2 ∨ x = 4 - complex.I * real.sqrt 2) :=
begin
  sorry
end

end solve_equation_l591_591959


namespace leap_years_count_l591_591712

theorem leap_years_count :
  let is_leap_year (y : ℕ) := (y % 900 = 150 ∨ y % 900 = 450) ∧ y % 100 = 0
  let range_start := 2100
  let range_end := 4200
  ∃ L, L = [2250, 2850, 3150, 3750, 4050] ∧ (∀ y ∈ L, is_leap_year y ∧ range_start ≤ y ∧ y ≤ range_end)
  ∧ L.length = 5 :=
by
  sorry

end leap_years_count_l591_591712


namespace no_x2_term_in_product_l591_591501

theorem no_x2_term_in_product (a : ℝ) :
  (∀ x : ℝ, ((x + 1) * (x^2 - 2 * a * x + a^2))).coeff 2 = 0 → a = 1 / 2 :=
by
  sorry

end no_x2_term_in_product_l591_591501


namespace sum_of_digits_is_13_l591_591862

theorem sum_of_digits_is_13:
  ∀ (a b c d : ℕ),
  b + c = 10 ∧
  c + d = 1 ∧
  a + d = 2 →
  a + b + c + d = 13 :=
by {
  sorry
}

end sum_of_digits_is_13_l591_591862


namespace simplify_expression_l591_591399

theorem simplify_expression : 
  (2 * Real.sqrt 7) / (Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 6) = 
  (2 * Real.sqrt 14 + 8 * Real.sqrt 21 + 2 * Real.sqrt 42 + 8 * Real.sqrt 63) / 23 :=
by
  sorry

end simplify_expression_l591_591399


namespace domain_of_function_l591_591417

-- Definitions of the conditions

def sqrt_condition (x : ℝ) : Prop := -x^2 - 3*x + 4 ≥ 0
def log_condition (x : ℝ) : Prop := x + 1 > 0 ∧ x + 1 ≠ 1

-- Statement of the problem

theorem domain_of_function :
  {x : ℝ | sqrt_condition x ∧ log_condition x} = { x | -1 < x ∧ x < 0 ∨ 0 < x ∧ x ≤ 1 } :=
sorry

end domain_of_function_l591_591417


namespace sum_of_palindromic_primes_l591_591233

def is_prime (n : ℕ) : Prop := sorry -- define the primality check
def reverse_digits (n : ℕ) : ℕ := sorry -- define the digit reversal function

def is_palindromic_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 200 ∧ (n % 10 ≠ 0) ∧ 
  (n / 100) % 2 = 1 ∧ -- hundreds digit is odd
  is_prime n ∧ is_prime (reverse_digits n)

theorem sum_of_palindromic_primes : 
  ∑ n in (Finset.filter is_palindromic_prime (Finset.range 200)), n = 868 :=
by
  sorry

end sum_of_palindromic_primes_l591_591233


namespace police_emergency_number_has_prime_divisor_gt_7_l591_591027

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := sorry

end police_emergency_number_has_prime_divisor_gt_7_l591_591027


namespace imaginary_part_of_conjugate_l591_591793

def z : Complex := (Complex.sqrt 2 / (Complex.sqrt 2 + Complex.I)) - (Complex.I / 2)

def z_conjugate : Complex := Complex.conj z

def imaginary_part (z : Complex) : ℝ := z.im

theorem imaginary_part_of_conjugate :
  imaginary_part z_conjugate = (Real.sqrt 2) / 3 + 1 / 2 := by
  sorry

end imaginary_part_of_conjugate_l591_591793
